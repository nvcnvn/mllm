// Whole-stack check: the golden trace (prefill0..2, decode0..2 at 2 requests) runs from tokens only, through
// apply_weights(0) .. apply_weights(layers) with apply_attention(i) of both kinds in between and pick_next_tokens,
// each call in one command buffer. Every key / value page and linear state is this run's own (prefill builds them).
// Decode next-token scores and greedy ids must equal golden row by row. Each decode call also runs with 4 requests on
// copies of both requests (pages and state copied before the call, on other units): its 4 rows must equal the 2-request
// rows. Pass: everything bit-exact.
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>

#include "golden.h"
#include "lanes.h"

// What every step needs: the GPU ops, their workspace and step table, and each request's arena units
struct Stack {
    const mllm::ModelSpec& model;
    mllm::Memory& memory;
    mllm::GpuOps& ops;
    mllm::Workspace& workspace;
    mllm::StepTables& tables;
    id<MTLBuffer> table;
    id<MTLCommandQueue> queue;
    std::vector<std::vector<uint32_t>> state_units, pages;  // per request
};

// 6 requests on shuffled units: 0, 1 = the trace's; 2..5 = copies of 0, 1, 0, 1 for the 4-request runs
static void place_requests(Stack& stack, int max_context) {
    mllm::Memory& memory = stack.memory;
    const size_t unit_bytes = memory.arena.unit_bytes;
    const uint32_t state_units = memory.state_units, pages = memory.units_for(max_context);
    std::vector<uint32_t> shuffled(memory.arena.n_units);
    std::iota(shuffled.begin(), shuffled.end(), 0);
    std::shuffle(shuffled.begin(), shuffled.end(), std::mt19937(3));
    assert(6 * (state_units + pages) <= shuffled.size());
    stack.state_units.resize(6), stack.pages.resize(6);
    for (int request = 0; request < 6; request++) {
        auto start = shuffled.begin() + request * (state_units + pages);
        stack.state_units[request].assign(start, start + state_units);
        stack.pages[request].assign(start + state_units, start + state_units + pages);
    }
    uint8_t* arena = (uint8_t*)memory.arena_buffer.contents;
    for (int request = 0; request < 2; request++)
        for (uint32_t unit : stack.state_units[request]) memset(arena + unit * unit_bytes, 0, unit_bytes);  // a new request's state is zero
}

static void copy_request(Stack& stack, int from, int to) {
    uint8_t* arena = (uint8_t*)stack.memory.arena_buffer.contents;
    const size_t unit_bytes = stack.memory.arena.unit_bytes;
    for (size_t j = 0; j < stack.state_units[from].size(); j++)
        memcpy(arena + stack.state_units[to][j] * unit_bytes, arena + stack.state_units[from][j] * unit_bytes, unit_bytes);
    for (size_t j = 0; j < stack.pages[from].size(); j++)
        memcpy(arena + stack.pages[to][j] * unit_bytes, arena + stack.pages[from][j] * unit_bytes, unit_bytes);
}

// Every layer of one step into the workspace's encoder; decode also picks the next tokens
static void encode_whole_stack(Stack& stack, bool decode) {
    const int layers = stack.model.n_layers();
    for (int i = 0; i <= layers; i++) {
        if (i == layers && !decode) break;  // prefill never runs apply_weights(layers)
        stack.ops.apply_weights(i, stack.workspace);
        if (i < layers) stack.ops.apply_attention(i, stack.workspace);
    }
    if (decode) stack.ops.pick_next_tokens(stack.workspace);
}

// One step through the whole stack; decode steps also pick the next tokens and return the greedy ids
static std::vector<int> run_step(Stack& stack, const std::vector<int>& requests, const std::vector<int>& positions,
                                 int tokens_per_request, const uint8_t* token_ids) {
    mllm::Workspace& workspace = stack.workspace;
    workspace.request_count = (int)requests.size(), workspace.tokens_per_request = tokens_per_request;
    std::vector<mllm::StepTables::Row> rows;
    for (size_t row = 0; row < requests.size(); row++)
        rows.push_back({positions[row], tokens_per_request, stack.pages[requests[row]], stack.state_units[requests[row]]});
    stack.tables.write(stack.table, rows);
    memcpy(workspace.token_ids.contents, token_ids, (size_t)workspace.request_count * tokens_per_request * 4);
    const bool decode = tokens_per_request == 1;
    @autoreleasepool {
        mllm::encode_and_wait(stack.queue, workspace, [&] { encode_whole_stack(stack, decode); });
    }
    return decode ? stack.ops.next_tokens(workspace) : std::vector<int>{};
}

// A decode call of the trace: at 2 requests, scores and greedy ids equal golden row by row; then 4 requests on copies
// give the same rows
static void check_decode_call(Stack& stack, const mllm::Safetensors& trace, const std::string& call_name,
                              const std::vector<int>& requests, const std::vector<int>& positions) {
    const mllm::Tensor& token_ids = trace.at(call_name + "/tokens");
    const size_t vocab_size = stack.model.vocab_size;
    assert(requests == std::vector<int>({0, 1}));
    for (int request = 2; request < 6; request++) copy_request(stack, request % 2, request);  // before the 2-request step changes 0 and 1
    std::vector<int> ids = run_step(stack, requests, positions, 1, trace.data(token_ids));
    const mllm::Tensor &want = trace.at(call_name + "/logits"), &want_ids = trace.at(call_name + "/argmax");
    const uint8_t* scores_data = (const uint8_t*)stack.workspace.next_token_scores.contents;
    const std::vector<uint8_t> scores(scores_data, scores_data + want.nbytes);
    for (int row = 0; row < 2; row++) {  // row by row
        const size_t row_bytes = vocab_size * 2;
        if (memcmp(scores.data() + row * row_bytes, trace.data(want) + row * row_bytes, row_bytes)) {
            golden::matches(call_name + "/logits", want, trace.data(want), scores.data(), 1, 0);  // prints cos, max |diff|
            assert(!"next-token scores row differs from golden");
        }
        assert(ids[row] == golden::load<int32_t>(trace.data(want_ids), row));
    }
    // 4 requests: copies 2, 3, 4, 5 of 0, 1, 0, 1, tokens and positions repeated
    std::vector<uint8_t> four_tokens(4 * 4);
    for (int row = 0; row < 4; row++) memcpy(&four_tokens[row * 4], trace.data(token_ids) + (row % 2) * 4, 4);
    std::vector<int> four_ids =
        run_step(stack, {2, 3, 4, 5}, {positions[0], positions[1], positions[0], positions[1]}, 1, four_tokens.data());
    for (int row = 0; row < 4; row++) {
        assert(!memcmp(scores_data + row * vocab_size * 2, scores.data() + (row % 2) * vocab_size * 2, vocab_size * 2));
        assert(four_ids[row] == ids[row % 2]);
    }
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string directory = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(directory + "/spec.json"));
    mllm::Safetensors trace(directory + "/trace.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    mllm::Memory memory(model, device, std::max<size_t>(256 << 20, mllm::Memory::arena_for(model, 6, 64)));  // 6 requests
    const int chunk = golden_spec["chunk"], max_batch = 4, max_rows = std::max(chunk, max_batch), max_context = 64;
    mllm::GpuOps ops(memory, weights);
    mllm::Workspace workspace = ops.workspace(chunk, max_batch, max_context);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    mllm::StepTables tables = memory.step_tables(max_rows);
    memory.freeze();
    id<MTLBuffer> table = workspace.step_table = tables.copies[0];
    Stack stack{model, memory, ops, workspace, tables, table, queue, {}, {}};
    place_requests(stack, max_context);

    int decodes = 0;
    for (const auto& call : golden_spec["calls"]) {
        std::string call_name = call["name"].get<std::string>();
        const mllm::Tensor& token_ids = trace.at(call_name + "/tokens");
        std::vector<int> requests, positions;
        for (const auto& row : call["rows"]) requests.push_back(row["req"]), positions.push_back(row["pos"]);
        const int tokens_per_request = (int)token_ids.shape[1];
        if (tokens_per_request > 1) {
            run_step(stack, requests, positions, tokens_per_request, trace.data(token_ids));
            continue;
        }
        check_decode_call(stack, trace, call_name, requests, positions);
        decodes++;
    }
    assert(decodes == 3);
    printf("whole stack from tokens: decode0..2 next-token scores and greedy ids bit-exact row by row at 2 requests; "
           "4-request copies equal the 2-request rows\nPASS\n");
}
