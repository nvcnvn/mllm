// apply_attention(i) check for linear-attention layers: replay the golden trace (prefill0..2, decode0..2 at 2 requests)
// through GpuOps::apply_attention with each request's state span on scattered, out-of-order arena units, layer by
// layer, from golden qkv / z / b / a (query_key_value, output_gate, update_rate_input, decay_input). Compares
// workspace.attention_result with <call>/LNN/attn/o and, after each call, the request's conv and delta state with
// <call>/rB/LNN/conv, delta. Decode calls run with 5 requests: both requests, copies of them on other units (rows must
// equal the originals, attention result and state) and a row with 0 tokens (its state must not change). Pass:
// everything bit-exact.
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>

#include "golden.h"
#include "lanes.h"

static double max_diff(const std::string& dtype, const uint8_t* got, const uint8_t* want, size_t count) {
    if (!memcmp(got, want, count * (dtype == "F32" ? 4 : 2))) return 0;
    double worst = 0;
    for (size_t e = 0; e < count; e++) {
        double got_value = golden::to_double(dtype, got, e), want_value = golden::to_double(dtype, want, e);
        worst = std::fmax(worst, std::isnan(got_value - want_value) ? INFINITY : std::fabs(got_value - want_value));
    }
    return worst;
}

struct FieldStats {
    int count = 0, exact = 0;
    double max_abs = 0;
};

// One call of the trace: its rows as this test runs them
struct Call {
    std::string name;
    int traced_rows;              // the trace's rows (requests)
    std::vector<int> request;     // per traced row: this test's request 0 or 1
    std::vector<int> source_row;  // per run row: the golden row its inputs come from
    std::vector<int> run_request;  // per run row: this test's request (its state span)
    bool with_copies;             // decode: + copies of both rows and a row with 0 tokens
};

struct LinearTest {
    const mllm::ModelSpec& model;
    const mllm::Safetensors& trace;
    mllm::Memory& memory;
    mllm::GpuOps& ops;
    mllm::Workspace& workspace;
    mllm::StepTables& tables;
    id<MTLBuffer> table;
    id<MTLCommandQueue> queue;
    std::vector<std::vector<uint32_t>> state_units;  // per request
    std::map<std::string, FieldStats> stats;
    std::map<int, int> request_of;  // trace request id -> 0, 1
    int copied_rows = 0;

    uint8_t* arena() const { return (uint8_t*)memory.arena_buffer.contents; }
    size_t unit_bytes() const { return memory.arena.unit_bytes; }

    // 5 requests' state spans on shuffled units: 0, 1 = the trace's requests, 2, 3 = their copies, 4 = idle row
    void place_requests() {
        const size_t units_per_span = memory.state_units;
        assert(5 * units_per_span <= memory.arena.n_units);
        std::vector<uint32_t> shuffled(memory.arena.n_units);
        std::iota(shuffled.begin(), shuffled.end(), 0);
        std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(7));
        state_units.resize(5);
        for (int request = 0; request < 5; request++)
            state_units[request].assign(shuffled.begin() + request * units_per_span,
                                        shuffled.begin() + (request + 1) * units_per_span);
        for (int request = 0; request < 4; request++)
            for (uint32_t unit : state_units[request]) memset(arena() + unit * unit_bytes(), 0, unit_bytes());
        std::mt19937 random(1);
        for (uint32_t unit : state_units[4])
            for (size_t i = 0; i < unit_bytes(); i++) arena()[unit * unit_bytes() + i] = (uint8_t)random();
    }

    // bytes [offset, offset + count) of request's state span, out of the arena
    void copy_from_span(int request, size_t offset, uint8_t* output, size_t count) const {
        for (size_t piece; count; offset += piece, output += piece, count -= piece) {
            piece = std::min(count, unit_bytes() - offset % unit_bytes());
            memcpy(output, arena() + state_units[request][offset / unit_bytes()] * unit_bytes() + offset % unit_bytes(), piece);
        }
    }
    std::vector<uint8_t> state(int request) const {
        std::vector<uint8_t> bytes(memory.state_units * unit_bytes());
        copy_from_span(request, 0, bytes.data(), bytes.size());
        return bytes;
    }
    void set_state(int request, const std::vector<uint8_t>& bytes) {
        for (size_t j = 0; j < memory.state_units; j++)
            memcpy(arena() + state_units[request][j] * unit_bytes(), bytes.data() + j * unit_bytes(), unit_bytes());
    }

    // golden rows source_row[r] of key -> workspace row r
    void load(const std::string& key, id<MTLBuffer> destination, const std::vector<int>& source_row) {
        const mllm::Tensor& tensor = trace.at(key);
        size_t row_bytes = tensor.nbytes / tensor.shape[0];
        assert(row_bytes * source_row.size() <= destination.length);
        for (size_t row = 0; row < source_row.size(); row++)
            memcpy((uint8_t*)destination.contents + row * row_bytes, trace.data(tensor) + source_row[row] * row_bytes, row_bytes);
    }

    void record(const std::string& field, const std::string& what, double diff) {
        FieldStats& stat = stats[field];
        stat.count++, stat.exact += diff == 0, stat.max_abs = std::fmax(stat.max_abs, diff);
        if (diff != 0) printf("  %s: max |diff| %g\n", what.c_str(), diff);
    }

    // The call's rows; decode calls also copy both requests' state to requests 2, 3 and add the idle row. Writes the
    // step table.
    Call prepare_call(const nlohmann::json& call) {
        Call prepared;
        prepared.name = call["name"].get<std::string>();
        const mllm::Tensor& token_ids = trace.at(prepared.name + "/tokens");
        prepared.traced_rows = (int)token_ids.shape[0];
        workspace.tokens_per_request = (int)token_ids.shape[1];
        std::vector<int> positions, token_counts;
        for (const auto& row : call["rows"]) {
            int id = row["req"];
            if (!request_of.count(id)) request_of[id] = (int)request_of.size();
            prepared.request.push_back(request_of[id]), positions.push_back(row["pos"]), token_counts.push_back(row["n"]);
        }
        prepared.with_copies = prepared.name.starts_with("decode");
        prepared.source_row.resize(prepared.traced_rows);
        std::iota(prepared.source_row.begin(), prepared.source_row.end(), 0);
        prepared.run_request = prepared.request;
        std::vector<mllm::StepTables::Row> rows;
        for (int row = 0; row < prepared.traced_rows; row++)
            rows.push_back({positions[row], token_counts[row], {}, state_units[prepared.request[row]]});
        if (prepared.with_copies) add_copies(prepared, positions, token_counts, rows);
        workspace.request_count = (int)rows.size();
        tables.write(table, rows);
        return prepared;
    }

    void add_copies(Call& call, const std::vector<int>& positions, const std::vector<int>& token_counts,
                    std::vector<mllm::StepTables::Row>& rows) {
        assert(call.traced_rows == 2);
        for (int row = 0; row < 2; row++) {
            const int copy = 2 + call.request[row];
            set_state(copy, state(call.request[row]));
            rows.push_back({positions[row], token_counts[row], {}, state_units[copy]});
            call.source_row.push_back(row), call.run_request.push_back(copy);
        }
        rows.push_back({0, 0, {}, state_units[4]}), call.source_row.push_back(0), call.run_request.push_back(4);
    }

    // apply_attention(layer) on golden inputs; the attention result against golden, copied rows against their originals
    void check_layer(const Call& call, int layer) {
        char layer_name[8];
        snprintf(layer_name, sizeof layer_name, "L%02d", layer);
        std::string prefix = call.name + "/" + layer_name + "/";
        if (!trace.tensors.count(prefix + "dense/qkv")) return;  // big models: a few layers (dump_golden kept_layers)
        load(prefix + "dense/qkv", workspace.query_key_value, call.source_row);
        load(prefix + "dense/z", workspace.output_gate, call.source_row);
        load(prefix + "dense/b", workspace.update_rate_input, call.source_row);
        load(prefix + "dense/a", workspace.decay_input, call.source_row);
        @autoreleasepool {
            mllm::encode_and_wait(queue, workspace, [&] { ops.apply_attention(layer, workspace); });
        }
        const mllm::Tensor& want = trace.at(prefix + "attn/o");
        const uint8_t* got = (const uint8_t*)workspace.attention_result.contents;
        record("o", prefix + "attn/o", max_diff("BF16", got, trace.data(want), want.numel()));
        if (!call.with_copies) return;
        const size_t row_bytes = want.nbytes / 2;
        for (int row = 2; row < 4; row++, copied_rows++)
            assert(!memcmp(got + row * row_bytes, got + call.source_row[row] * row_bytes, row_bytes));
    }

    // Each traced row's conv and delta state after the call, against golden
    void check_states(const Call& call) {
        for (int row = 0; row < call.traced_rows; row++)
            for (const char* field : {"conv", "delta"})
                for (int layer : model.linear_layers()) check_state(call, row, field, layer);
        if (!call.with_copies) return;
        for (int row = 2; row < 4; row++) assert(state(call.run_request[row]) == state(call.request[call.source_row[row]]));
    }

    void check_state(const Call& call, int row, const char* field, int layer) {
        char key[64];
        snprintf(key, sizeof key, "%s/r%d/L%02d/%s", call.name.c_str(), row, layer, field);
        if (!trace.tensors.count(key)) return;
        const mllm::Tensor& want = trace.at(key);
        size_t offset = memory.state_offset(layer) + (field[0] == 'd' ? memory.conv_bytes : 0);
        std::vector<uint8_t> got(want.nbytes);
        copy_from_span(call.request[row], offset, got.data(), want.nbytes);
        record(field, key, max_diff(want.dtype, got.data(), trace.data(want), want.numel()));
    }
};

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string directory = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(directory + "/spec.json"));
    mllm::Safetensors trace(directory + "/trace.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    mllm::Memory memory(model, device, std::max<size_t>(256 << 20, mllm::Memory::arena_for(model, 5, 0)));  // 5 state spans
    const int chunk = golden_spec["chunk"], max_batch = 5, max_rows = std::max(chunk, max_batch);
    mllm::GpuOps ops(memory, weights);
    ops.merge_projections = false;  // the fields are loaded here, not left by apply_weights' merged one-row multiplies
    mllm::Workspace workspace = ops.workspace(chunk, max_batch, 1024);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    mllm::StepTables tables = memory.step_tables(max_rows);
    memory.freeze();
    id<MTLBuffer> table = workspace.step_table = tables.copies[0];

    LinearTest test{model, trace, memory, ops, workspace, tables, table, queue, {}, {}, {}, 0};
    test.place_requests();
    const std::vector<uint8_t> idle = test.state(4);

    for (const auto& call : golden_spec["calls"]) {
        const Call prepared = test.prepare_call(call);
        for (int layer : model.linear_layers()) test.check_layer(prepared, layer);
        test.check_states(prepared);
        if (prepared.with_copies) assert(test.state(4) == idle);
    }

    bool ok = true;
    for (const auto& [name, stat] : test.stats) {
        printf("%-6s bit-exact %3d/%3d  max |diff| %g\n", name.c_str(), stat.exact, stat.count, stat.max_abs);
        ok &= stat.exact == stat.count;
    }
    int traced_layers = 0;  // linear layers in the trace
    for (int layer : model.linear_layers())
        traced_layers += trace.tensors.count("decode0/L" + std::string(layer < 10 ? "0" : "") + std::to_string(layer) + "/dense/qkv");
    printf("decode with 5 requests: %d copied rows equal to the originals (attention result and state), "
           "0-token row's state untouched\n", test.copied_rows);
    assert(ok && test.stats["o"].count == 6 * traced_layers && test.stats["delta"].count == 9 * traced_layers &&
           test.copied_rows == 3 * traced_layers * 2);
    printf("PASS\n");
}
