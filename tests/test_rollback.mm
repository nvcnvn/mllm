// Speculative rollback (docs/speculative.md, g23): a verify of [t0, drafts] (one request, DRAFTS + 1 rows,
// Workspace::checkpoint_states) writes the linear state after each row into its own set; accepting m drafts keeps set
// m and moves the position to c + m + 1. Checked here without a draft head (any model; the 0.8B in make test):
//  1. after 0, 1 and k accepted drafts, the kept set and the decode from it equal a plain run's (tolerance: the
//     verify's multi-row kernels round differently from decode's), conv and gated-delta state of every linear layer;
//  2. nothing of a rejected row survives: rejected drafts that were wrong or right leave the same kept state and the
//     same decode after it, bit for bit.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "golden.h"
#include "lanes.h"

constexpr int PREFIX = 40, DRAFTS = mllm::DecodeLane::DRAFTS, ROWS = DRAFTS + 1, AFTER = 2;

struct Run {  // what a sequence of steps leaves: the last step's scores, the state set kept
    std::vector<uint16_t> scores;
    std::vector<uint8_t> state;
};

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    const int max_context = 256;
    mllm::Memory memory(model, device, mllm::Memory::arena_for(model, 1 + DRAFTS + 1, max_context));
    mllm::GpuOps ops(memory, weights);
    mllm::Workspace ws = ops.workspace(64, ROWS, max_context);
    mllm::StepTables tables = memory.step_tables(ROWS + 1, max_context);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    memory.freeze();
    const uint32_t state_units = memory.state_units, unit_bytes = (uint32_t)memory.arena.unit_bytes;
    std::vector<uint32_t> sets[ROWS], pages;  // sets[0] is read; after row j the state is in sets[j]
    uint32_t unit = 0;
    for (auto& set : sets)
        for (uint32_t j = 0; j < state_units; j++) set.push_back(unit++);
    while (pages.size() < memory.units_for(max_context)) pages.push_back(unit++);
    assert(unit <= memory.arena.n_units);
    uint8_t* arena = (uint8_t*)memory.arena_buffer.contents;
    const int layer_count = model.n_layers(), vocab = model.vocab_size;

    auto pass = [&](const std::vector<int>& tokens, int pos, const std::vector<uint32_t>& state, bool verify) {
        const int n = (int)tokens.size();
        std::copy(tokens.begin(), tokens.end(), (int32_t*)ws.token_ids.contents);
        ws.request_count = 1, ws.tokens_per_request = n, ws.step_table = tables.copies[0], ws.checkpoint_states = verify;
        tables.write(ws.step_table, {{pos, n, pages, state}});
        mllm::encode_and_wait(queue, ws, [&] {
            for (int layer = 0; layer < layer_count; layer++) ops.apply_weights(layer, ws), ops.apply_attention(layer, ws);
            ops.apply_weights(layer_count, ws);
            if (pos + n <= PREFIX) return;  // the prefix: no scores
            ws.request_count = n, ws.tokens_per_request = 1;  // lm_head + argmax on every row
            ops.pick_next_tokens(ws);
        });
        ws.checkpoint_states = false;
        return std::vector<int>((int32_t*)ws.next_token_ids.contents, (int32_t*)ws.next_token_ids.contents + n);
    };
    auto read_state = [&](const std::vector<uint32_t>& set) {
        std::vector<uint8_t> bytes;
        for (uint32_t u : set) bytes.insert(bytes.end(), arena + (size_t)u * unit_bytes, arena + (size_t)(u + 1) * unit_bytes);
        return bytes;
    };
    auto scores_row = [&](int row) {
        const uint16_t* scores = (const uint16_t*)ws.next_token_scores.contents + (size_t)row * vocab;
        return std::vector<uint16_t>(scores, scores + vocab);
    };
    std::vector<int> prefix(PREFIX);
    for (int j = 0; j < PREFIX; j++) prefix[j] = 1000 + 37 * j;
    auto start = [&] {  // the prefix in sets[0], deterministic: every run starts from the same bits
        for (uint32_t u : sets[0]) memset(arena + (size_t)u * unit_bytes, 0, unit_bytes);
        pass(prefix, 0, sets[0], false);
    };
    // plain decode of `tokens` from pos with the state in `set`: the last scores and the state after
    auto decode = [&](const std::vector<int>& tokens, int pos, const std::vector<uint32_t>& set) {
        Run run;
        for (int j = 0; j < (int)tokens.size(); j++) pass({tokens[j]}, pos + j, set, false), run.scores = scores_row(0);
        run.state = read_state(set);
        return run;
    };

    // the plain reference: greedy from t0 for DRAFTS + AFTER + 1 tokens; the state after each
    start();
    std::vector<int> greedy{4321};
    std::vector<std::vector<uint8_t>> plain_states;
    std::vector<std::vector<uint16_t>> plain_scores;
    for (int j = 0; j < ROWS + AFTER; j++) {
        greedy.push_back(pass({greedy[j]}, PREFIX + j, sets[0], false)[0]);
        plain_states.push_back(read_state(sets[0])), plain_scores.push_back(scores_row(0));
    }

    // state sets: per linear layer, conv (bf16) and delta (fp32): cos >= 0.999 each
    auto check_state = [&](const std::string& what, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
        double worst = 1;
        for (int layer : model.linear_layers()) {
            const size_t offset = memory.state_offset(layer);
            const auto cosine = [&](size_t at, size_t count, bool fp32) {
                double dot = 0, a2 = 0, b2 = 0;
                for (size_t i = 0; i < count; i++) {
                    const double a = fp32 ? ((const float*)&got[at])[i] : golden::to_double("BF16", &got[at], i),
                                 b = fp32 ? ((const float*)&want[at])[i] : golden::to_double("BF16", &want[at], i);
                    dot += a * b, a2 += a * a, b2 += b * b;
                }
                return a2 + b2 == 0 ? 1.0 : dot / std::sqrt(a2 * b2);
            };
            worst = std::min({worst, cosine(offset, memory.conv_bytes / 2, false),
                              cosine(offset + memory.conv_bytes, memory.delta_bytes / 4, true)});
        }
        printf("  %s: state cos >= %.6f\n", what.c_str(), worst);
        assert(worst >= 0.999);
    };
    double worst_score = 0;  // max |verify - plain| over the scores: test_speculative's near-tie threshold comes from it
    auto check_scores = [&](const std::string& what, const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
        const mllm::Tensor row{"BF16", {vocab}, 0, 0, (size_t)vocab * 2};
        for (int i = 0; i < vocab; i++)
            worst_score = std::fmax(worst_score, std::fabs(golden::to_double("BF16", (const uint8_t*)got.data(), i) -
                                                           golden::to_double("BF16", (const uint8_t*)want.data(), i)));
        // 0.998: the 27B's decode after a verify-kept state gave 0.99883 (the 0.8B: >= 0.999)
        assert(golden::matches(what, row, (const uint8_t*)want.data(), got.data(), 0.998, 1.0));
    };

    for (int accepted = 0; accepted <= DRAFTS; accepted++) {
        printf("%d of %d drafts accepted\n", accepted, DRAFTS);
        Run runs[2];  // right drafts after the accepted ones, wrong ones
        for (int wrong = 0; wrong < 2; wrong++) {
            std::vector<int> verified(greedy.begin(), greedy.begin() + ROWS);
            for (int j = accepted + 1; j < ROWS && wrong; j++) verified[j] = verified[j] + 1;
            start();
            const std::vector<int> chosen = pass(verified, PREFIX, [&] {
                std::vector<uint32_t> listed = sets[0];
                for (const auto& set : sets) listed.insert(listed.end(), set.begin(), set.end());
                return listed;
            }(), true);
            for (int j = 0; j <= accepted; j++) {
                if (chosen[j] != greedy[j + 1]) printf("  row %d: argmax %d, plain %d (near-tie?)\n", j, chosen[j], greedy[j + 1]);
                check_scores("verify row " + std::to_string(j), scores_row(j), plain_scores[j]);
            }
            check_state("kept set after " + std::to_string(accepted) + (wrong ? ", rest wrong" : ", rest right"),
                        read_state(sets[accepted]), plain_states[accepted]);
            // decode on from the kept set, AFTER tokens past the round
            const int pos = PREFIX + accepted + 1;
            runs[wrong] = decode(std::vector<int>(greedy.begin() + accepted + 1, greedy.begin() + ROWS + AFTER), pos,
                                 sets[accepted]);
            check_state("decode after", runs[wrong].state, plain_states.back());
            check_scores("decode after", runs[wrong].scores, plain_scores.back());
        }
        assert(runs[0].scores == runs[1].scores && runs[0].state == runs[1].state);
        printf("  rejected rows leave no trace: bit-exact either way\n");
    }
    printf("scores: max |multi-row - 1-row| %.4f\nPASS\n", worst_score);
}
