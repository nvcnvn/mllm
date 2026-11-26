// Memory checks.
// Arena storage: sizes and byte offsets equal memory.py's (golden "layout"); golden keys / values and linear state
//   written at Python's offsets read back through the GPU at the C++ offsets.
// Workspace: after freeze(), a simulated step allocates nothing (counter), and an allocation asserts.
// Step tables: the copy the CPU fills is never the one an in-flight step reads; running 2 steps ahead asserts.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <csignal>
#include <fstream>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "memory.h"

using mllm::ELEM;

// Runs function in a child; true if the child died of an assert.
template <class Function>
static bool dies_of_assert(Function function) {
    pid_t child = fork();
    if (child == 0) {
        freopen("/dev/null", "w", stderr);
        function();
        _exit(0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

// GPU copies (kernel copy_u32) from the arena at given offsets, all in one command buffer
struct GpuCopier {
    id<MTLComputePipelineState> pipeline;
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    GpuCopier(id<MTLDevice> device, id<MTLCommandQueue> queue) {
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&error];
        pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"copy_u32"] error:&error];
        assert(pipeline);
        command_buffer = [queue commandBuffer];
        encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
    }
    void copy(id<MTLBuffer> source, size_t source_offset, id<MTLBuffer> target, size_t target_offset, size_t bytes) {
        assert(source_offset % 4 == 0 && target_offset % 4 == 0 && bytes % 4 == 0);
        [encoder setBuffer:source offset:source_offset atIndex:0];
        [encoder setBuffer:target offset:target_offset atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(bytes / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    void run() {
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        assert(command_buffer.status == MTLCommandBufferStatusCompleted);
    }
};

// The arena's layout against memory.py's, and golden data written at Python's offsets read back at C++'s
struct ArenaLayoutTest {
    const mllm::ModelSpec& model;
    mllm::Memory& memory;
    const nlohmann::json& layout;  // the golden spec's "layout"
    const mllm::Safetensors& trace;
    id<MTLDevice> device;
    const size_t unit_bytes = memory.arena.unit_bytes;
    const int full_layer_count = (int)model.full_layers().size(), key_value_heads = model.key_value_heads,
              page_tokens = memory.page_tokens, head_dim = model.head_dim;
    std::vector<size_t> kv_strides = {};  // numpy kv[slot, full_layer, key_or_value, head, token, dim], in bytes

    size_t python_kv_offset(size_t slot, int full_layer, int key_or_value, int head, int token) const {
        return slot * kv_strides[0] + full_layer * kv_strides[1] + key_or_value * kv_strides[2] +
               head * kv_strides[3] + token * kv_strides[4];
    }

    // sizes and offsets equal Python's
    // With an MTP head (memory.py has none) a unit is Python's plus the head's page, after the model's full layers;
    // everything inside a unit stays at Python's offsets.
    void check_sizes_and_offsets() {
        const size_t python_unit = layout["unit_bytes"].get<size_t>(),
                     head_page = model.mtp ? 2 * key_value_heads * page_tokens * head_dim * ELEM : 0;
        assert(unit_bytes == (python_unit + head_page + 16383) / 16384 * 16384);
        if (!model.mtp)
            assert(memory.arena.n_units == layout["n_units"].get<uint32_t>() &&
                   memory.state_units == layout["state_units"].get<uint32_t>());
        assert(memory.conv_bytes == layout["conv_bytes"].get<size_t>() &&
               memory.delta_bytes == layout["delta_bytes"].get<size_t>());
        assert(memory.arena_buffer.length == (size_t)memory.arena.n_units * unit_bytes);
        kv_strides = layout["kv_strides"].get<std::vector<size_t>>();
        assert(kv_strides.size() == 6 && kv_strides[0] == python_unit && kv_strides[5] == ELEM);
        kv_strides[0] = unit_bytes;  // Python's offsets within a unit, on C++'s units
        for (int full_layer = 0; full_layer < full_layer_count; full_layer++)
            for (int key_or_value = 0; key_or_value < 2; key_or_value++)
                check_head_offsets(full_layer, key_or_value);
        for (const auto& entry : layout["kv"])
            assert(entry[0].get<size_t>() * unit_bytes + memory.kv_offset(entry[1], entry[2], entry[3], entry[4]) ==
                   entry[5].get<size_t>() + entry[0].get<size_t>() * (unit_bytes - python_unit));
        for (const auto& entry : layout["state"])
            assert(memory.state_offset(entry[0]) == entry[1].get<size_t>() &&
                   memory.state_offset(entry[0]) + memory.conv_bytes == entry[2].get<size_t>());
        assert(layout["state"].size() == model.linear_layers().size());
    }

    void check_head_offsets(int full_layer, int key_or_value) const {
        for (int head = 0; head < key_value_heads; head++)
            for (int token = 0; token < page_tokens; token++)
                assert(memory.kv_offset(full_layer, key_or_value, head, token) ==
                       python_kv_offset(0, full_layer, key_or_value, head, token));
    }

    // --- golden keys / values and state: the CPU writes at Python's offsets, the GPU reads at C++'s
    uint8_t* arena_bytes = (uint8_t*)memory.arena_buffer.contents;
    int token_count = 0;
    uint32_t page_count = 0;
    std::vector<uint32_t> pages = {}, state_units = {};  // scattered, disjoint
    size_t token_bytes = 0, page_run_bytes = 0;  // one token's head vector; one (page, full layer, key_or_value, head) run
    id<MTLBuffer> keys_values_read_back = nil, state_read_back = nil;

    const mllm::Tensor& trace_tensor(const char* name_format, int layer) const {
        char name[64];
        snprintf(name, sizeof name, name_format, layer);
        return trace.at(name);
    }
    const mllm::Tensor& key_or_value_tensor(int key_or_value, int full_layer) const {
        return trace_tensor(key_or_value ? "decode2/r0/L%02d/v" : "decode2/r0/L%02d/k", model.full_layers()[full_layer]);
    }
    // big models' traces keep a few layers (dump_golden kept_layers)
    bool layer_in_trace(int layer) const {
        return trace.tensors.count("decode2/r0/L" + std::string(layer < 10 ? "0" : "") + std::to_string(layer) +
                                   (model.kinds[layer] == "full" ? "/k" : "/conv"));
    }

    void write_golden_and_read_back() {
        token_count = (int)trace_tensor("decode2/r0/L%02d/k", model.full_layers()[0]).shape[1];
        page_count = memory.units_for(token_count);
        const uint32_t unit_count = memory.arena.n_units;
        for (uint32_t j = 0; j < page_count; j++) pages.push_back((j * 37 + 11) % unit_count);
        for (uint32_t j = 0; j < memory.state_units; j++) state_units.push_back(unit_count - 1 - 3 * j);
        token_bytes = head_dim * ELEM, page_run_bytes = page_tokens * token_bytes;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        GpuCopier copier(device, queue);
        keys_values_read_back = [device newBufferWithLength:(size_t)full_layer_count * 2 * key_value_heads * page_count *
                                                            page_run_bytes
                                                    options:MTLResourceStorageModeShared];
        for (int full_layer = 0; full_layer < full_layer_count; full_layer++)
            for (int key_or_value = 0; key_or_value < 2 && layer_in_trace(model.full_layers()[full_layer]); key_or_value++)
                write_and_copy_keys_values(copier, full_layer, key_or_value);
        for (const auto& entry : layout["state"])  // Python's offsets
            write_golden_state(entry);
        state_read_back = [device newBufferWithLength:memory.state_units * unit_bytes
                                              options:MTLResourceStorageModeShared];
        for (uint32_t j = 0; j < memory.state_units; j++)
            copier.copy(memory.arena_buffer, state_units[j] * unit_bytes, state_read_back, j * unit_bytes, unit_bytes);
        copier.run();
    }

    size_t read_back_offset(int full_layer, int key_or_value, int head) const {
        return (((size_t)full_layer * 2 + key_or_value) * key_value_heads + head) * page_count * page_run_bytes;
    }

    void write_and_copy_keys_values(GpuCopier& copier, int full_layer, int key_or_value) {
        const mllm::Tensor& tensor = key_or_value_tensor(key_or_value, full_layer);
        assert(tensor.dtype == "BF16" && tensor.shape == std::vector<int64_t>({key_value_heads, token_count, head_dim}));
        for (int head = 0; head < key_value_heads; head++) {
            for (int token = 0; token < token_count; token++)
                memcpy(arena_bytes + python_kv_offset(pages[token / page_tokens], full_layer, key_or_value, head,
                                                      token % page_tokens),
                       trace.data(tensor) + ((size_t)head * token_count + token) * token_bytes, token_bytes);
            for (uint32_t page = 0; page < page_count; page++)
                copier.copy(memory.arena_buffer,
                            pages[page] * unit_bytes + memory.kv_offset(full_layer, key_or_value, head, 0),
                            keys_values_read_back,
                            read_back_offset(full_layer, key_or_value, head) + page * page_run_bytes, page_run_bytes);
        }
    }

    // state span byte b lives at state_units[b / unit_bytes] * unit_bytes + b % unit_bytes (memory.py load_state
    // concatenates the slots)
    void write_state_span(size_t offset, const uint8_t* source, size_t count) {
        for (size_t piece; count; offset += piece, source += piece, count -= piece) {
            piece = std::min(count, unit_bytes - offset % unit_bytes);
            memcpy(arena_bytes + state_units[offset / unit_bytes] * unit_bytes + offset % unit_bytes, source, piece);
        }
    }

    void write_golden_state(const nlohmann::json& entry) {
        int layer = entry[0];
        if (!layer_in_trace(layer)) return;
        const mllm::Tensor &conv = trace_tensor("decode2/r0/L%02d/conv", layer),
                           &delta = trace_tensor("decode2/r0/L%02d/delta", layer);
        assert(conv.nbytes == memory.conv_bytes && delta.nbytes == memory.delta_bytes);
        write_state_span(entry[1], trace.data(conv), conv.nbytes);
        write_state_span(entry[2], trace.data(delta), delta.nbytes);
    }

    void check_read_back() const {
        const uint8_t* keys_values = (const uint8_t*)keys_values_read_back.contents;
        for (int full_layer = 0; full_layer < full_layer_count; full_layer++)
            for (int key_or_value = 0; key_or_value < 2 && layer_in_trace(model.full_layers()[full_layer]); key_or_value++)
                check_heads_read_back(keys_values, full_layer, key_or_value);
        const uint8_t* state = (const uint8_t*)state_read_back.contents;
        for (int layer : model.linear_layers()) {
            if (!layer_in_trace(layer)) continue;
            const mllm::Tensor &conv = trace_tensor("decode2/r0/L%02d/conv", layer),
                               &delta = trace_tensor("decode2/r0/L%02d/delta", layer);
            assert(!memcmp(state + memory.state_offset(layer), trace.data(conv), conv.nbytes));
            assert(!memcmp(state + memory.state_offset(layer) + memory.conv_bytes, trace.data(delta), delta.nbytes));
        }
    }

    void check_heads_read_back(const uint8_t* keys_values, int full_layer, int key_or_value) const {
        const mllm::Tensor& tensor = key_or_value_tensor(key_or_value, full_layer);
        for (int head = 0; head < key_value_heads; head++)
            assert(!memcmp(keys_values + read_back_offset(full_layer, key_or_value, head),
                           trace.data(tensor) + (size_t)head * token_count * token_bytes, token_count * token_bytes));
    }
};

// A run of steps: the GPU finishes steps with varying lag; the CPU fills step N's table once step N-2 is done
static void run_steps_with_lag(mllm::Memory& memory, mllm::StepTables& tables) {
    mllm::Arena& arena = memory.arena;
    uint64_t signaled = 0;
    std::vector<uint32_t> held;
    for (uint64_t step = 1; step <= 100; step++) {
        if (step % 3 == 0) signaled = step - 1;      // GPU caught up
        while (signaled + 2 < step) signaled++;      // CPU waits on the event
        id<MTLBuffer> table = tables.begin(step, signaled);
        for (uint64_t in_flight = signaled + 1; in_flight < step; in_flight++)
            assert(table != tables.copies[in_flight % 2]);
        arena.retire(signaled);
        if (arena.admit(1)) held.push_back(arena.alloc());
        tables.write(table, {{0, 1, held}});  // one row holding every unit allocated so far
        const uint32_t* words = (const uint32_t*)table.contents;
        assert(words[0] == 1 && words[1] == 0 && words[4] == 0 && words[5] == held.size() &&
               std::equal(held.begin(), held.end(), words + 6));
        if (step % 10 == 0) {
            arena.committed = step;
            arena.release(held);
            held.clear();
        }
        memory.request_residency();
    }
}

// Two rows with pages and state units:
// [rows][state units per row][start_position x rows][token_count x rows][rows + 1 offsets][pages][state x rows]
static void check_two_row_table(mllm::StepTables& tables) {
    std::vector<uint32_t> pages_0 = {4, 9}, pages_1 = {7}, state_0 = {1, 3}, state_1 = {8, 2};
    tables.write(tables.copies[0], {{5, 1, pages_0, state_0}, {0, 3, pages_1, state_1}});
    const uint32_t expected[] = {2, 2, 5, 0, 1, 3, 0, 2, 3, 4, 9, 7, 1, 3, 8, 2};
    assert(!memcmp(tables.copies[0].contents, expected, sizeof expected));
    assert(dies_of_assert([&] {  // state count differs
        tables.write(tables.copies[0], {{5, 1, pages_0, state_0}, {0, 3, pages_1, {}}});
    }));
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string golden_dir = golden::dir(model);
    std::ifstream spec_file(golden_dir + "/spec.json");
    nlohmann::json golden_spec = nlohmann::json::parse(spec_file);
    const auto& layout = golden_spec["layout"];
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Memory memory(model, device, layout["arena_bytes"].get<size_t>(), golden_spec["page_tokens"].get<int>());
    mllm::Safetensors trace(golden_dir + "/trace.safetensors");

    ArenaLayoutTest layout_test{model, memory, layout, trace, device};
    layout_test.check_sizes_and_offsets();
    layout_test.write_golden_and_read_back();
    layout_test.check_read_back();

    // --- workspace + step tables at startup, then nothing allocates
    const int chunk = golden_spec["chunk"], max_batch = 8;
    mllm::Workspace workspace = memory.workspace(chunk, max_batch);
    assert(workspace.residual.length >= (size_t)chunk * model.hidden * ELEM &&
           workspace.next_token_scores.length == (size_t)max_batch * model.vocab_size * ELEM);
    mllm::StepTables tables = memory.step_tables(max_batch);
    memory.freeze();
    const int startup_allocations = memory.allocations;
    assert(startup_allocations == 1 + 12 + 2);  // arena, workspace fields, table copies

    run_steps_with_lag(memory, tables);
    check_two_row_table(tables);
    assert(memory.allocations == startup_allocations);
    assert(dies_of_assert([&] { memory.buffer(16); }));        // allocation after startup
    assert(dies_of_assert([&] { tables.begin(10, 7); }));      // 2 steps ahead: step 9 may still read copy 0
    assert(!dies_of_assert([&] { tables.begin(10, 8); }));
    printf("layout = memory.py; %d K/V pages + %u state units read back through the GPU; %d buffers, none after "
           "startup\nPASS\n",
           layout_test.full_layer_count * 2 * layout_test.key_value_heads * (int)layout_test.page_count,
           memory.state_units, startup_allocations);
}
