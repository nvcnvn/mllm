// The decode lane's concurrent encoder (GpuOps::before_dispatch's barriers) == a serial one, byte for byte: the next
// token scores and every touched key / value and state byte, one whole decode step repeated 20 times at 7, 8, 13, 16
// and 17 requests (the batched-decode and split multiplies, which share the workspace's partial_sums; rows at mixed
// positions, some on the two-pass attention path). From g9 QA's race harness.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "golden.h"
#include "lanes.h"

using namespace mllm;

// One decode step's rows: request r at a mixed position, its pages and state units numbered from 0 in units[r].
// Returns the units used.
static uint32_t make_rows(const Memory& memory, int request_count, std::vector<std::vector<uint32_t>>& units,
                          std::vector<StepTables::Row>& rows) {
    uint32_t next_unit = 0;
    for (int request = 0; request < request_count; request++) {
        const int position = (request * 211 + 37) % 1500 + (request % 3 == 0 ? 0 : 400);  // some >= 1024 keys (2-pass)
        const uint32_t page_count = memory.units_for(position + 1);
        for (uint32_t j = 0; j < page_count + memory.state_units; j++) units[request].push_back(next_unit++);
        rows.push_back({position, 1, std::span(units[request]).first(page_count),
                        std::span(units[request]).subspan(page_count)});
    }
    return next_unit;
}

// random bf16 in [-0.5, 0.5)
static std::vector<uint16_t> random_bf16(size_t count, std::mt19937& random_engine) {
    std::vector<uint16_t> values(count);
    std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
    for (auto& value : values) {
        float drawn = uniform(random_engine);
        uint32_t bits;
        memcpy(&bits, &drawn, 4);
        value = bits >> 16;
    }
    return values;
}

// every layer's weights and attention, then the next tokens
static void encode_decode_step(GpuOps& ops, Workspace& workspace, int layer_count) {
    for (int layer = 0; layer <= layer_count; layer++) {
        ops.apply_weights(layer, workspace);
        if (layer < layer_count) ops.apply_attention(layer, workspace);
    }
    ops.pick_next_tokens(workspace);
}

int main() {
    const ModelSpec model = spec::load(golden::MODEL);
    if (Memory::unit_of(model) >= (2u << 20)) {  // 27B: 17 rows x (74 state + pages) units of 2 MiB, ~4 GB copied 200 times
        printf("SKIP: the race harness copies every touched unit per run (~1 TB of memcpy at 2 MiB units); the 0.8B covers it\n");
        return 0;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    Weights weights(model.path, device);
    const int layer_count = model.n_layers();
    Memory memory(model, device, (size_t)1 << 30);
    GpuOps ops(memory, weights);
    Workspace workspace = ops.workspace(1, 17, 4096);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    StepTables tables = memory.step_tables(17, 4096);
    id<MTLBuffer> table = memory.buffer(tables.words * 4);
    memory.freeze();
    for (int request_count : {7, 8, 13, 16, 17}) {
        std::mt19937 random_engine(request_count);
        std::vector<std::vector<uint32_t>> units(request_count);
        std::vector<StepTables::Row> rows;
        const uint32_t used_units = make_rows(memory, request_count, units, rows);
        tables.write(table, rows);
        workspace.request_count = request_count, workspace.tokens_per_request = 1, workspace.step_table = table;
        const size_t arena_bytes = (size_t)used_units * memory.arena.unit_bytes;
        const size_t score_bytes = (size_t)request_count * model.vocab_size * 2;
        const std::vector<uint16_t> initial_arena = random_bf16(arena_bytes / 2, random_engine);
        for (int request = 0; request < request_count; request++)
            ((int32_t*)workspace.token_ids.contents)[request] = (int)(random_engine() % model.vocab_size);
        // one decode step from the same arena contents; returns its scores followed by the arena bytes
        auto run_step = [&](bool concurrent) {
            memcpy(memory.arena_buffer.contents, initial_arena.data(), arena_bytes);
            encode_and_wait(queue, workspace, [&] { encode_decode_step(ops, workspace, layer_count); }, concurrent);
            std::vector<uint8_t> result(score_bytes + arena_bytes);
            memcpy(result.data(), workspace.next_token_scores.contents, score_bytes);
            memcpy(result.data() + score_bytes, memory.arena_buffer.contents, arena_bytes);
            return result;
        };
        const auto serial_result = run_step(false);
        int differing_steps = 0;
        for (int k = 0; k < 20; k++) @autoreleasepool {
                differing_steps += run_step(true) != serial_result;
            }
        printf("%d requests: concurrent != serial in %d / 20 steps\n", request_count, differing_steps);
        assert(differing_steps == 0);
    }
    printf("PASS\n");
}
