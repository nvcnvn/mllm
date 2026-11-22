// Memory: the only component that allocates (py/mllm/memory.py): Weights, Arena, Workspace, StepTables.
// Layout and allocator rules: docs/memory-design.md.
#pragma once
#import <Metal/Metal.h>

#include <deque>
#include <span>
#include <string>
#include <vector>

#include "safetensors.h"
#include "spec.h"

namespace mllm {

// The model's safetensors (or header-padded copies, see memory.mm), mmapped; each file wrapped once as a no-copy MTLBuffer. A tensor = buffer + offset.
struct Weights {
    Weights(const std::string& path, id<MTLDevice> device);

    struct View {
        id<MTLBuffer> buffer;
        size_t offset;
        const Tensor& t;
    };
    View operator[](const std::string& name) const;

    Safetensors st;
    std::vector<id<MTLBuffer>> buffers;  // one per st.files
};

constexpr size_t ELEM = 2;  // bytes per activation element (bf16 for Qwen3.5)

// Arena bookkeeping (memory-design "Bookkeeping"). CPU only; one engine thread calls it, no locks.
// Slot states: free -> live (alloc) -> pending (last ref released) -> free (retire, once the GPU passed that step).
class Arena {
public:
    Arena(size_t unit_bytes, uint32_t n_units);

    // memory-design's form; reserved <= free_units always holds, so neither side wraps
    bool fits(uint32_t n) const { return n <= free_units - reserved; }
    bool can_ever(uint32_t n) const { return n <= free_units + cache_only_units - reserved; }
    bool admit(uint32_t n);  // reserve n units; false = wait (the engine rejects n > n_units on arrival)
    uint32_t alloc();        // one unit out of the caller's reservation; needs a free (retired) unit
    void unreserve(uint32_t n);
    void release(const std::vector<uint32_t>& slots);  // drop one request ref per slot
    void retire(uint64_t signaled);  // pending units whose step <= signaled (GPU event value) become free

    size_t unit_bytes;
    uint32_t n_units;
    uint64_t committed = 0;  // last committed step: released units stay pending until the GPU passes it (lanes set it, g5)

    std::vector<uint64_t> free;  // bitmap, bit set = reusable now
    std::vector<uint16_t> req_refs, cache_refs;  // cache_refs: nothing sets it until the prefix cache exists
    std::deque<std::pair<uint32_t, uint64_t>> pending;  // (slot, step), steps non-decreasing
    uint32_t free_units;  // no references: free or pending
    uint32_t cache_only_units = 0, reserved = 0;
};

// Activation buffers, sized once for rows = max(chunk, max_batch) (py/mllm/gpu.py docstring has the fields).
// Row-major [rows, ...] in bf16, except tokens (int32).
struct Workspace {
    id<MTLBuffer> tokens, x, o, q, k, v, gate, qkv, z, b, a;
    id<MTLBuffer> logits;  // [max_batch, vocab]
    int B = 0, S = 0;      // tokens' shape this step (the lane sets it); dense ops work on B * S rows
    id<MTLBuffer> table;   // this step's StepTables copy, row b = workspace row b (the lane sets it; attn reads it)
};

// Per-step uint32 slot lists the GPU reads, 2 copies by step parity (memory-design "Step tables").
// Layout (uint32 words, R rows): [R][pos x R][n x R][R + 1 offsets][slots...]; row b's pages are
// slots[off[b] .. off[b + 1]). pos: the row's first token position, n: its valid tokens this step. The attention
// kernels read it as is (src/kernels/attention.metal Table); rope reads the pos words as its int offsets.
struct StepTables {
    struct Row {
        int pos, n;
        std::span<const uint32_t> slots;
    };
    id<MTLBuffer> copies[2];
    size_t words;      // capacity of each copy, in uint32
    uint32_t n_units;  // slots are arena units
    // The copy to fill for step. The CPU may run one step ahead of the GPU: asserts signaled + 2 >= step.
    id<MTLBuffer> begin(uint64_t step, uint64_t signaled);
    void write(id<MTLBuffer> table, const std::vector<Row>& rows) const;
};

class Memory {
public:
    Memory(const ModelSpec& spec, id<MTLDevice> device, size_t arena_bytes, int page_tokens = 32);

    uint32_t units_for(size_t n_tokens) const { return (uint32_t)((n_tokens + T - 1) / T); }
    // Byte offsets, identical to memory.py's numpy views. KV within a unit: [g, kv, head, token, dim], g = index
    // among full layers. State: a slot list concatenated into one span; per linear layer conv then delta.
    size_t kv_offset(int g, int kv, int head, int token) const;
    size_t state_offset(int layer) const;

    Workspace workspace(int chunk, int max_batch);
    StepTables step_tables(int max_rows);
    // ponytail: the engine calls this periodically so the OS keeps the arena wired (g5 owns the timer).
    void request_residency() { [residency requestResidency]; }

    id<MTLBuffer> buffer(size_t bytes, MTLResourceOptions options = MTLResourceStorageModeShared);  // every allocation
    void freeze() { frozen = true; }  // after startup: any further buffer() asserts

    ModelSpec spec;
    id<MTLDevice> device;
    int T;  // tokens per page
    Arena arena;
    id<MTLBuffer> arena_buffer;  // n_units * unit_bytes, shared, untracked, pre-touched, in the residency set
    size_t conv_bytes, delta_bytes;
    uint32_t state_units;
    int allocations = 0;

private:
    std::vector<size_t> state_off;  // per layer; linear layers only
    id<MTLResidencySet> residency;
    bool frozen = false;
};

}  // namespace mllm
