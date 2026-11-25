// Batched decode by tolerance (architecture.md): teacher-forced decode of batch_ref's 8 sequences
// (py/dump_batch_ref.py), every position as one token per request from position 0, through GpuOps on a concurrent
// encoder (as the decode lane). 1 request: the reference's top-64 next-token scores byte for byte. 8 and 16 requests
// (each sequence on 1 or 2 rows, rows at different positions): mean KL(reference || ours) over the reference's top-64
// plus a rest bucket <= KL_MAX, and our highest-scoring token within one bf16 ulp of the reference's top score at >= 99%
// of positions.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "golden.h"
#include "lanes.h"

using namespace mllm;

// mlx-lm's own KL(1 request || 8) is 7.5e-4 mean (16: 9.2e-4) on batch_ref: 10x that. A wrong kernel lands far above
// (8 stale outputs per qmv_wide multiply: 1.8); a bad weight group doesn't (one group x 2 in each layer's
// feed_forward_gate: 1.3e-3, inside rounding noise): the 1-request byte check and test_kernels' CPU references catch
// those. Top-1 within one ulp: mlx-lm's strict top-1 is 98.5% at 8 requests, every miss a tie or 1-ulp gap at 1.
static const double KL_MAX = 7.5e-3, TOP1_MIN = 0.99;

static float bf16(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, 4);
    return value;
}

// batch_ref.safetensors: per (sequence, position) the token, the reference's top-64 token ids and their bf16 scores,
// and the log of its sum of exp(scores)
struct Reference {
    Safetensors file;
    const Tensor &tokens, &top_ids, &top_scores, &log_sum_exp;
    int sequences, positions, top_count;

    explicit Reference(const std::string& path)
        : file(path), tokens(file.at("tokens")), top_ids(file.at("ids")), top_scores(file.at("logits")),
          log_sum_exp(file.at("lse")), sequences((int)tokens.shape[0]), positions((int)tokens.shape[1]),
          top_count((int)top_ids.shape[2]) {}
    int token(int sequence, int position) const {
        return golden::load<int32_t>(file.data(tokens), (size_t)sequence * positions + position);
    }
    int top_id(int sequence, int position, int k) const {
        return golden::load<int32_t>(file.data(top_ids), ((size_t)sequence * positions + position) * top_count + k);
    }
    uint16_t top_score(int sequence, int position, int k) const {
        return golden::load<uint16_t>(file.data(top_scores), ((size_t)sequence * positions + position) * top_count + k);
    }
    double log_sum(int sequence, int position) const {
        return golden::load<float>(file.data(log_sum_exp), (size_t)sequence * positions + position);
    }
};

// Running totals of one run's checked positions
struct Agreement {
    double kl_sum = 0, kl_max = 0;
    int positions = 0, top1 = 0, within_one_ulp = 0;
};

struct BatchTest {
    const ModelSpec& model;
    const Reference& reference;
    Memory& memory;
    GpuOps& ops;
    Workspace& workspace;
    StepTables& tables;
    id<MTLBuffer> table;
    id<MTLCommandQueue> queue;
    uint32_t pages_per_row, units_per_row;
    std::vector<std::vector<uint32_t>> units;  // per row slot: its pages, then its state units
    Agreement agreement;

    // One decode step: row b = (sequences[b], positions[b]) in unit slot slots[b]; scores land in
    // workspace.next_token_scores
    void step(const std::vector<int>& sequences, const std::vector<int>& positions, const std::vector<int>& slots) {
        @autoreleasepool {
            std::vector<StepTables::Row> rows;
            for (size_t row = 0; row < sequences.size(); row++) {
                std::span<const uint32_t> slot_units(units[slots[row]]);
                rows.push_back({positions[row], 1, slot_units.first(pages_per_row), slot_units.subspan(pages_per_row)});
                ((int32_t*)workspace.token_ids.contents)[row] = reference.token(sequences[row], positions[row]);
            }
            tables.write(table, rows);
            workspace.request_count = (int)sequences.size(), workspace.tokens_per_request = 1, workspace.step_table = table;
            encode_and_wait(queue, workspace, [&] { encode_decode_step(); }, true);
        }
    }

    void encode_decode_step() {
        const int layers = model.n_layers();
        for (int i = 0; i <= layers; i++) {
            ops.apply_weights(i, workspace);
            if (i < layers) ops.apply_attention(i, workspace);
        }
        ops.pick_next_tokens(workspace);
    }

    // Row `row` of workspace.next_token_scores against the reference at (sequence, position): KL over its top-64 +
    // rest, and top-1 within 1 ulp
    void check(int row, int sequence, int position) {
        const int vocab_size = model.vocab_size;
        const uint16_t* scores = (const uint16_t*)workspace.next_token_scores.contents + (size_t)row * vocab_size;
        double highest = -INFINITY, sum = 0;
        int highest_token = 0;
        for (int token = 0; token < vocab_size; token++)
            if (bf16(scores[token]) > highest) highest = bf16(scores[token]), highest_token = token;
        for (int token = 0; token < vocab_size; token++) sum += std::exp((double)bf16(scores[token]) - highest);
        const double our_log_sum = highest + std::log(sum), reference_log_sum = reference.log_sum(sequence, position);
        double kl = 0, reference_top_mass = 0, our_top_mass = 0, reference_score_of_ours = -INFINITY;
        for (int k = 0; k < reference.top_count; k++) {
            const int token = reference.top_id(sequence, position, k);
            const double reference_log_p = bf16(reference.top_score(sequence, position, k)) - reference_log_sum;
            const double our_log_p = bf16(scores[token]) - our_log_sum;
            kl += std::exp(reference_log_p) * (reference_log_p - our_log_p);
            reference_top_mass += std::exp(reference_log_p), our_top_mass += std::exp(our_log_p);
            if (token == highest_token) reference_score_of_ours = bf16(reference.top_score(sequence, position, k));
        }
        const double reference_rest = std::fmax(1 - reference_top_mass, 1e-30), our_rest = std::fmax(1 - our_top_mass, 1e-30);
        kl += reference_rest * std::log(reference_rest / our_rest);
        const double top = bf16(reference.top_score(sequence, position, 0));
        agreement.kl_sum += kl, agreement.kl_max = std::fmax(agreement.kl_max, kl), agreement.positions++;
        agreement.top1 += highest_token == reference.top_id(sequence, position, 0);
        agreement.within_one_ulp +=
            reference_score_of_ours >= top - std::exp2(std::floor(std::log2(std::fabs(top))) - 7);
    }

    void reset() {
        memset(memory.arena_buffer.contents, 0, (size_t)16 * units_per_row * memory.arena.unit_bytes);  // fresh state: zero
        agreement = {};
    }

    void report(int request_count) const {
        const Agreement& a = agreement;
        printf("B=%2d: %d positions, KL mean %.3e max %.3e (limit %.1e), top-1 %.2f%%, within 1 ulp %.2f%%\n",
               request_count, a.positions, a.kl_sum / a.positions, a.kl_max, KL_MAX, 100.0 * a.top1 / a.positions,
               100.0 * a.within_one_ulp / a.positions);
    }

    // 1 request: byte for byte with the reference's top-64
    void check_one_request() {
        reset();
        int differing = 0;
        for (int sequence = 0; sequence < reference.sequences; sequence++)
            for (int position = 0; position < reference.positions; position++)
                differing += check_one_request_position(sequence, position);
        report(1);
        assert(differing == 0 && agreement.top1 == agreement.positions && "B = 1 decode differs from mlx-lm");
    }

    // Returns how many of the reference's top-64 scores differ
    int check_one_request_position(int sequence, int position) {
        step({sequence}, {position}, {sequence});  // a unit set per sequence: each starts from zero state
        check(0, sequence, position);
        const uint16_t* scores = (const uint16_t*)workspace.next_token_scores.contents;
        int differing = 0;
        for (int k = 0; k < reference.top_count; k++)
            differing += scores[reference.top_id(sequence, position, k)] != reference.top_score(sequence, position, k);
        return differing;
    }

    // Row r: sequence r % 8, offset[r] (distinct per row). Warm-up (not checked, the batch ramps up): row r decodes
    // positions 0 .. offset[r] - 1. Then 80 steps with every row, row r at positions offset[r] .. offset[r] + 79,
    // all checked.
    void check_requests(int request_count) {
        const int warm_up = 48;
        std::vector<int> offset(request_count);
        for (int row = 0; row < request_count; row++) offset[row] = row * 23 % warm_up;
        reset();
        for (int step_index = 0; step_index < reference.positions; step_index++) {
            std::vector<int> sequences, positions, slots;
            for (int row = 0; row < request_count; row++) {
                const int position = step_index < warm_up ? step_index - (warm_up - offset[row]) : offset[row] + step_index - warm_up;
                if (position >= 0) sequences.push_back(row % reference.sequences), positions.push_back(position), slots.push_back(row);
            }
            if (sequences.empty()) continue;
            step(sequences, positions, slots);
            if (step_index >= warm_up)
                for (int row = 0; row < request_count; row++) check(row, sequences[row], positions[row]);
        }
        report(request_count);
        assert(agreement.kl_sum / agreement.positions <= KL_MAX && "batched decode: mean KL over the limit");
        assert(agreement.within_one_ulp >= TOP1_MIN * agreement.positions && "batched decode: top-1 agreement under 99%");
    }
};

int main() {
    const ModelSpec model = spec::load(golden::MODEL);
    Reference reference(golden::dir(model) + "/batch_ref.safetensors");
    const int positions = reference.positions;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    Weights weights(model.path, device);
    Memory memory(model, device, std::max<size_t>((size_t)1 << 30, Memory::arena_for(model, 16, positions)));  // 16 rows
    GpuOps ops(memory, weights);
    Workspace workspace = ops.workspace(1, 16, positions);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    StepTables tables = memory.step_tables(16, positions);
    id<MTLBuffer> table = memory.buffer(tables.words * 4);
    memory.freeze();
    const uint32_t pages_per_row = memory.units_for(positions), units_per_row = pages_per_row + memory.state_units;
    assert((size_t)16 * units_per_row <= memory.arena.n_units);

    // fresh-per-run units for row slot b
    std::vector<std::vector<uint32_t>> units(16);
    for (uint32_t slot = 0; slot < 16; slot++)
        for (uint32_t j = 0; j < units_per_row; j++) units[slot].push_back(slot * units_per_row + j);

    BatchTest test{model, reference, memory, ops, workspace, tables, table, queue, pages_per_row, units_per_row, units, {}};
    test.check_one_request();
    for (int request_count : {8, 16}) test.check_requests(request_count);
    printf("PASS\n");
}
