// apply_attention(i) check for full-attention layers: replay the golden trace (prefill0..2, decode0..2 at 2 requests)
// through GpuOps::apply_attention with each request's pages on scattered arena slots, layer by layer, from golden
// query / key / value / query_gate. Compares workspace.attention_result with <call>/LNN/attn/o and, after each call, the
// request's key / value pages with <call>/rB/LNN/k, v. Decode calls run a second time with 4 requests (both requests
// plus copies of them on other slots): every row must equal the 2-request run. Then attn.safetensors
// (py/dump_golden.py SWEEP): one row per (tokens, position) around MLX's SDPA switch points (1-pass / 2-pass and its
// blocks, vector / unfused, gemv_wide / steel, block / looped softmax), and a 1-pass and a 2-pass decode row in one
// 2-request call. Pass: everything bit-exact, except two 2-pass rows in one call: they share the longer context's block
// count (27B: 128 at 4201 keys, 1_1023 alone runs 64), which batched decode allows (KL / top-1 tolerance, test_batch);
// at more than 4 query heads per key / value head the attention result must be within max |o| / 64 (2-4 bf16 ulps at
// max |o|), at 4 or fewer (0.8B: always 64 blocks) bit-exact. Also: a padded prefill row (fewer tokens than
// workspace.tokens_per_request), a slot past the arena and a linear layer on a table without state units assert.
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>

#include <cassert>
#include <csignal>
#include <string>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "lanes.h"

// Metal buffer memory isn't mapped in a fork()ed child, so assert checks re-run this binary with a mode argument:
// true when that run dies on SIGABRT (an assert)
static bool aborts(const char* self, const char* mode) {
    pid_t child;
    const char* arguments[] = {self, mode, nullptr};
    posix_spawn_file_actions_t file_actions;
    posix_spawn_file_actions_init(&file_actions);
    posix_spawn_file_actions_addopen(&file_actions, 2, "/dev/null", O_WRONLY, 0);
    int failed = posix_spawn(&child, self, &file_actions, nullptr, (char* const*)arguments, nullptr);
    posix_spawn_file_actions_destroy(&file_actions);
    int status = 0;
    if (failed || waitpid(child, &status, 0) < 0) return false;
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static double max_diff(const uint8_t* got, const uint8_t* want, size_t count) {
    if (!memcmp(got, want, count * 2)) return 0;
    double worst = 0;
    for (size_t e = 0; e < count; e++) {
        double got_value = golden::to_double("BF16", got, e), want_value = golden::to_double("BF16", want, e);
        worst = std::fmax(worst, std::isnan(got_value - want_value) ? INFINITY : std::fabs(got_value - want_value));
    }
    return worst;
}

struct FieldStats {
    int count = 0, exact = 0;
    double max_abs = 0;
};

// A sweep row: its case name ("<tokens>_<position>") and the arena units its pages go on
using SweepRow = std::pair<std::string, const std::vector<uint32_t>*>;

struct AttentionTest {
    const mllm::ModelSpec& model;
    const mllm::Safetensors& trace;
    mllm::Memory& memory;
    mllm::GpuOps& ops;
    mllm::Workspace& workspace;
    mllm::StepTables& tables;
    id<MTLBuffer> table;
    id<MTLCommandQueue> queue;
    // pages on scattered, out-of-order slots; requests 2 and 3 are copies of 0 and 1 (the 4-request decode)
    std::vector<std::vector<uint32_t>> slots = {{5, 2}, {9}, {12, 7}, {13}};
    std::map<std::string, FieldStats> stats = {};
    int four_request_rows = 0, traced_full_layers = 0;  // full layers in the trace (big models: a few, dump_golden kept_layers)

    uint8_t* arena() const { return (uint8_t*)memory.arena_buffer.contents; }
    size_t unit_bytes() const { return memory.arena.unit_bytes; }

    // golden row source_row[r] of key -> workspace row r; query / key / value come as [requests, heads, tokens,
    // head_dim], ours are [requests, tokens, heads, head_dim]
    void load(const std::string& key, id<MTLBuffer> destination, const std::vector<int>& source_row) {
        const mllm::Tensor& tensor = trace.at(key);
        size_t row_bytes = tensor.nbytes / tensor.shape[0];
        assert(row_bytes * source_row.size() <= destination.length);
        for (size_t row = 0; row < source_row.size(); row++) {
            const uint8_t* input = trace.data(tensor) + source_row[row] * row_bytes;
            uint8_t* output = (uint8_t*)destination.contents + row * row_bytes;
            if (tensor.shape.size() < 4) memcpy(output, input, row_bytes);
            else reorder_heads(tensor, input, output);
        }
    }

    // one request's [heads, tokens, head_dim] -> [tokens, heads, head_dim]
    void reorder_heads(const mllm::Tensor& tensor, const uint8_t* input, uint8_t* output) const {
        const int64_t heads = tensor.shape[1], tokens = tensor.shape[2], head_dim = model.head_dim;
        for (int64_t head = 0; head < heads; head++)
            for (int64_t token = 0; token < tokens; token++)
                memcpy(output + ((token * heads + head) * head_dim) * 2, input + ((head * tokens + token) * head_dim) * 2,
                       head_dim * 2);
    }

    // Keys (key_or_value 0) or values (1) of full layer full_index on pages, tokens 0 .. tokens - 1, as golden's
    // [key_value_heads, tokens, head_dim]
    std::vector<uint8_t> read_pages(const std::vector<uint32_t>& pages, int full_index, int key_or_value, int tokens) const {
        const int key_value_heads = model.key_value_heads, head_dim = model.head_dim, page_tokens = memory.page_tokens;
        std::vector<uint8_t> output((size_t)key_value_heads * tokens * head_dim * 2);
        for (int head = 0; head < key_value_heads; head++)
            for (int token = 0; token < tokens; token++)
                memcpy(&output[((size_t)head * tokens + token) * head_dim * 2],
                       arena() + pages[token / page_tokens] * unit_bytes() +
                           memory.kv_offset(full_index, key_or_value, head, token % page_tokens),
                       head_dim * 2);
        return output;
    }

    void record(const std::string& field, const std::string& what, double diff) {
        FieldStats& stat = stats[field];
        stat.count++, stat.exact += diff == 0, stat.max_abs = std::fmax(stat.max_abs, diff);
        if (diff != 0) printf("  %s: max |diff| %g\n", what.c_str(), diff);
    }

    void run_attention(int layer) {
        @autoreleasepool {
            mllm::encode_and_wait(queue, workspace, [&] { ops.apply_attention(layer, workspace); });
        }
    }

    // ---- the golden trace

    // One traced call (its rows' requests, positions and token counts) on every full layer the trace has
    void check_call(const std::string& call_name, const nlohmann::json& call) {
        const mllm::Tensor& token_ids = trace.at(call_name + "/tokens");
        workspace.request_count = (int)token_ids.shape[0], workspace.tokens_per_request = (int)token_ids.shape[1];
        std::vector<int> requests, positions, token_counts;
        for (const auto& row : call["rows"])
            requests.push_back(row["req"]), positions.push_back(row["pos"]), token_counts.push_back(row["n"]);
        std::vector<int> same_rows(workspace.request_count);
        for (int row = 0; row < workspace.request_count; row++) same_rows[row] = row;
        const auto full = model.full_layers();
        for (size_t full_index = 0; full_index < full.size(); full_index++)
            check_call_layer(call_name, (int)full_index, requests, positions, token_counts, same_rows);
    }

    // Row b of the table: request run_requests[b] at the traced row b % rows' position and token count
    void write_table(const std::vector<int>& run_requests, const std::vector<int>& positions,
                     const std::vector<int>& token_counts) {
        std::vector<mllm::StepTables::Row> rows;
        for (size_t row = 0; row < run_requests.size(); row++)
            rows.push_back({positions[row % positions.size()], token_counts[row % positions.size()], slots[run_requests[row]]});
        tables.write(table, rows);
    }

    void run_layer(const std::string& prefix, int layer, const std::vector<int>& source_row,
                   const std::vector<int>& run_requests, const std::vector<int>& positions,
                   const std::vector<int>& token_counts) {
        workspace.request_count = (int)source_row.size();
        write_table(run_requests, positions, token_counts);
        load(prefix + "dense/q", workspace.query, source_row);
        load(prefix + "dense/k", workspace.key, source_row);
        load(prefix + "dense/v", workspace.value, source_row);
        load(prefix + "dense/gate", workspace.query_gate, source_row);
        run_attention(layer);
    }

    void check_call_layer(const std::string& call_name, int full_index, const std::vector<int>& requests,
                          const std::vector<int>& positions, const std::vector<int>& token_counts,
                          const std::vector<int>& same_rows) {
        const int layer = model.full_layers()[full_index];
        char layer_name[8];
        snprintf(layer_name, sizeof layer_name, "L%02d", layer);
        std::string prefix = call_name + "/" + layer_name + "/";
        if (!trace.tensors.count(prefix + "attn/o")) return;
        traced_full_layers += call_name == "decode0";
        run_layer(prefix, layer, same_rows, requests, positions, token_counts);
        const mllm::Tensor& want = trace.at(prefix + "attn/o");
        record("o", prefix + "attn/o", max_diff((const uint8_t*)workspace.attention_result.contents, trace.data(want), want.numel()));

        for (size_t row = 0; row < requests.size(); row++)
            for (int key_or_value = 0; key_or_value < 2; key_or_value++) {
                std::string key = call_name + "/r" + std::to_string(row) + "/" + layer_name + "/" + (key_or_value ? "v" : "k");
                const mllm::Tensor& pages = trace.at(key);
                const int tokens = positions[row] + token_counts[row];
                assert(pages.shape[1] == tokens);
                record("kv", key, max_diff(read_pages(slots[requests[row]], full_index, key_or_value, tokens).data(),
                                           trace.data(pages), pages.numel()));
            }
        if (call_name.starts_with("decode"))
            check_four_requests(prefix, layer, want.nbytes, requests, positions, token_counts);
    }

    // 4 requests: copies of both requests on other slots, rows must not change
    void check_four_requests(const std::string& prefix, int layer, size_t result_bytes, const std::vector<int>& requests,
                             const std::vector<int>& positions, const std::vector<int>& token_counts) {
        const uint8_t* result = (const uint8_t*)workspace.attention_result.contents;
        std::vector<uint8_t> two_request_result(result, result + result_bytes);
        for (int request = 0; request < 2; request++)
            for (size_t page = 0; page < slots[request].size(); page++)
                memcpy(arena() + slots[request + 2][page] * unit_bytes(), arena() + slots[request][page] * unit_bytes(), unit_bytes());
        run_layer(prefix, layer, {0, 1, 0, 1}, {requests[0], requests[1], 2 + requests[0], 2 + requests[1]}, positions,
                  token_counts);
        const size_t row_bytes = result_bytes / 2;
        for (int row = 0; row < 4; row++, four_request_rows++)
            assert(!memcmp(result + row * row_bytes, two_request_result.data() + (row % 2) * row_bytes, row_bytes));
        workspace.request_count = 2;
    }

    // ---- the sweep, on the last full layer

    // A sweep row's inputs into workspace row `row` and its first `position` keys / values onto its pages
    mllm::StepTables::Row load_sweep_row(const mllm::Safetensors& sweep, const SweepRow& sweep_row, size_t row,
                                         int full_index) {
        const auto& [case_name, pages] = sweep_row;
        int tokens, position;
        sscanf(case_name.c_str(), "%d_%d", &tokens, &position);
        workspace.tokens_per_request = tokens;
        for (auto [field, buffer] : {std::pair{"q", workspace.query}, {"k", workspace.key}, {"v", workspace.value},
                                     {"gate", workspace.query_gate}}) {
            const mllm::Tensor& tensor = sweep.at(case_name + "/" + field);
            memcpy((uint8_t*)buffer.contents + row * tensor.nbytes, sweep.data(tensor), tensor.nbytes);
        }
        for (int key_or_value = 0; key_or_value < 2; key_or_value++) {
            const mllm::Tensor& tensor = sweep.at(case_name + (key_or_value ? "/kv_v" : "/kv_k"));
            write_pages(*pages, full_index, key_or_value, position, sweep.data(tensor), position + tokens);
        }
        return {position, tokens, std::span(pages->data(), memory.units_for(position + tokens))};
    }

    // golden [key_value_heads, total_tokens, head_dim]'s first `tokens` tokens onto pages
    void write_pages(const std::vector<uint32_t>& pages, int full_index, int key_or_value, int tokens,
                     const uint8_t* source, int total_tokens) {
        const int head_dim = model.head_dim, page_tokens = memory.page_tokens;
        for (int head = 0; head < model.key_value_heads; head++)
            for (int token = 0; token < tokens; token++)
                memcpy(arena() + pages[token / page_tokens] * unit_bytes() +
                           memory.kv_offset(full_index, key_or_value, head, token % page_tokens),
                       source + ((size_t)head * total_tokens + token) * head_dim * 2, head_dim * 2);
    }

    // Loads the rows' inputs and pages, runs apply_attention, checks the attention result and keys / values per row.
    // within_tolerance: the attention result within max |o| / 64 instead of bit-exact
    void sweep(const mllm::Safetensors& sweep_file, const std::vector<SweepRow>& rows, bool within_tolerance = false) {
        const auto full = model.full_layers();
        const int full_index = (int)full.size() - 1;
        std::vector<mllm::StepTables::Row> table_rows;
        workspace.request_count = (int)rows.size();
        for (size_t row = 0; row < rows.size(); row++)
            table_rows.push_back(load_sweep_row(sweep_file, rows[row], row, full_index));
        tables.write(table, table_rows);
        run_attention(full[full_index]);
        for (size_t row = 0; row < rows.size(); row++) check_sweep_row(sweep_file, rows, row, full_index, within_tolerance);
    }

    void check_sweep_row(const mllm::Safetensors& sweep_file, const std::vector<SweepRow>& rows, size_t row,
                         int full_index, bool within_tolerance) {
        const auto& [case_name, pages] = rows[row];
        const mllm::Tensor& want = sweep_file.at(case_name + "/o");
        std::string what = "sweep " + case_name + (rows.size() > 1 ? " (" + std::to_string(rows.size()) + " requests)" : "");
        double diff = max_diff((const uint8_t*)workspace.attention_result.contents + row * want.nbytes,
                               sweep_file.data(want), want.numel());
        double largest = 0;
        for (size_t e = 0; e < want.numel(); e++)
            largest = std::fmax(largest, std::fabs(golden::to_double("BF16", sweep_file.data(want), e)));
        if (within_tolerance)
            printf("  %s o: max |diff| %g, max |o| %g (tolerance)\n", what.c_str(), diff, largest), assert(diff <= largest / 64);
        else record("sweep o", what + " o", diff);
        for (int key_or_value = 0; key_or_value < 2; key_or_value++) {
            const mllm::Tensor& tensor = sweep_file.at(case_name + (key_or_value ? "/kv_v" : "/kv_k"));
            record("sweep kv", what + (key_or_value ? " v" : " k"),
                   max_diff(read_pages(*pages, full_index, key_or_value, (int)tensor.shape[1]).data(),
                            sweep_file.data(tensor), tensor.numel()));
        }
    }
};

// The runs `aborts` starts: each must hit an assert. Returns 0 if it didn't.
static int run_assert_mode(const std::string& mode, const mllm::ModelSpec& model, mllm::Memory& memory, mllm::GpuOps& ops,
                           mllm::Workspace& workspace, mllm::StepTables& tables, id<MTLBuffer> table) {
    if (mode == "padded") {  // the lane contract: a prefill row is not padded (tokens_per_request = its tokens), MLX picks kernels by query length
        std::vector<uint32_t> pages = {0, 1, 2, 3, 4};
        workspace.request_count = 1, workspace.tokens_per_request = 9;
        tables.write(table, {{100, 5, pages}});
        ops.apply_attention(model.full_layers()[0], workspace);
    }
    if (mode == "nostate") {  // a linear layer with a table written without state units
        tables.write(table, {{0, 1, std::vector<uint32_t>{0}}});
        workspace.request_count = 1, workspace.tokens_per_request = 1;
        ops.apply_attention(model.linear_layers()[0], workspace);
    }
    if (mode == "slot")  // a slot past the arena
        tables.write(table, {{0, 1, std::vector<uint32_t>{memory.arena.n_units}}});
    return 0;
}

int main(int argc, char** argv) {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string directory = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(directory + "/spec.json"));
    mllm::Safetensors trace(directory + "/trace.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    mllm::Memory memory(model, device, std::max<size_t>(128 << 20, mllm::Memory::arena_for(model, 1, 9216)));  // the sweep's 9128 tokens
    const int chunk = 128, max_batch = 4, max_rows = std::max(chunk, max_batch);  // the sweep's longest chunk
    mllm::GpuOps ops(memory, weights);
    ops.merge_projections = false;  // the fields are loaded here, not left by apply_weights' merged one-row multiplies
    mllm::Workspace workspace = ops.workspace(chunk, max_batch, 9216);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    mllm::StepTables tables = memory.step_tables(max_rows);
    memory.freeze();
    id<MTLBuffer> table = workspace.step_table = tables.copies[0];
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "padded" || mode == "nostate" || mode == "slot")
        return run_assert_mode(mode, model, memory, ops, workspace, tables, table);

    AttentionTest test{model, trace, memory, ops, workspace, tables, table, queue};
    for (const auto& call : golden_spec["calls"]) test.check_call(call["name"].get<std::string>(), call);

    // the sweep; pages on the arena's last units downwards (as dump_golden) or upwards
    mllm::Safetensors sweep_file(directory + "/attn.safetensors");
    std::vector<uint32_t> downwards(memory.arena.n_units), upwards(memory.arena.n_units);
    for (uint32_t unit = 0; unit < memory.arena.n_units; unit++)
        upwards[unit] = unit, downwards[unit] = memory.arena.n_units - 1 - unit;
    std::vector<std::string> cases;
    for (const auto& [key, tensor] : sweep_file.tensors)
        if (key.ends_with("/o")) cases.push_back(key.substr(0, key.size() - 2));
    for (const std::string& case_name : cases) test.sweep(sweep_file, {{case_name, &downwards}});
    test.sweep(sweep_file, {{"1_1022", &upwards}, {"1_4200", &downwards}});  // 1-pass row and 2-pass row in one decode call
    test.sweep(sweep_file, {{"1_4200", &upwards}, {"1_0", &downwards}});
    test.sweep(sweep_file, {{"1_1023", &upwards}, {"1_4200", &downwards}},
               model.query_heads / model.key_value_heads > 4);  // two 2-pass rows, 1024 and 4201 keys

    assert(aborts(argv[0], "padded") && aborts(argv[0], "slot") && aborts(argv[0], "nostate"));

    bool ok = true;
    for (const auto& [name, stat] : test.stats) {
        printf("%-8s bit-exact %3d/%3d  max |diff| %g\n", name.c_str(), stat.exact, stat.count, stat.max_abs);
        ok &= stat.exact == stat.count;
    }
    printf("4-request decode: %d rows equal to the 2-request run\n", test.four_request_rows);
    printf("sweep: %zu cases + 3 mixed decode calls\n", cases.size());
    assert(ok && test.traced_full_layers && test.four_request_rows == 3 * test.traced_full_layers * 4 && cases.size() >= 20);
    printf("PASS\n");
}
