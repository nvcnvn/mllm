// Memory: the only component that allocates (py/mllm/memory.py): Weights, Arena, Workspace, StepTables.
// Layout and allocator rules: docs/memory-design.md.
#pragma once
#import <Metal/Metal.h>

#include <deque>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "safetensors.h"
#include "spec.h"

namespace mllm {

// Projections GpuOps runs as one matmul at M = 1 (its merged QLinears), by module name: the weights cache keeps each
// chain's weight / scales / biases back to back (memory.mm aligned).
inline const std::vector<std::vector<std::string>> MERGED = {
    {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"},
    {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"},
    {"mlp.gate_proj", "mlp.up_proj"}};

// The model's safetensors, rewritten once into the cache (memory.mm aligned), mmapped; each file wrapped once as a no-copy MTLBuffer. A tensor = buffer + offset.
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
    // reserve n units; false = wait (the engine rejects n > n_units on arrival). evict: the prefix cache's next
    // victim (false = nothing left), called while !fits(n); can_ever(n) means it gets there.
    bool admit(uint32_t n, const std::function<bool()>& evict = {});
    uint32_t alloc();        // one unit out of the caller's reservation; needs a free (retired) unit
    void unreserve(uint32_t n);
    void release(const std::vector<uint32_t>& slots);  // drop one request ref per slot
    void ref(uint32_t s);            // one more request ref on a held unit (a cached page or snapshot)
    void cache_ref(uint32_t s);      // the prefix cache takes a unit a request holds
    void cache_release(uint32_t s);  // the prefix cache drops it
    void retire(uint64_t signaled);  // pending units whose step <= signaled (GPU event value) become free

    size_t unit_bytes;
    uint32_t n_units;
    uint64_t committed = 0;  // last committed step: released units stay pending until the GPU passes it (lanes set it, g5)

    std::vector<uint64_t> free;  // bitmap, bit set = reusable now
    std::vector<uint16_t> req_refs, cache_refs;
    std::deque<std::pair<uint32_t, uint64_t>> pending;  // (slot, step), steps non-decreasing
    uint32_t free_units;  // no references: free or pending
    uint32_t cache_only_units = 0, reserved = 0;

private:
    void drop(uint32_t s);  // no refs left: pending until the GPU passes committed
};

// Activation buffers, sized once for rows = max(chunk, max_batch) (py/mllm/gpu.py docstring has the fields).
// Row-major [rows, ...] in bf16, except tokens (int32).
struct Workspace {
    id<MTLBuffer> tokens, x, o, q, k, v, gate, qkv, z, b, a;
    id<MTLBuffer> logits;  // [max_batch, vocab]
    int B = 0, S = 0;      // tokens' shape this step (the lane sets it); dense ops work on B * S rows
    id<MTLBuffer> table;   // this step's StepTables copy, row b = workspace row b (the lane sets it; attn reads it)
    id<MTLComputeCommandEncoder> enc;  // GpuOps encodes into it (the lane sets it; GpuOps::pre orders it)
    int rows = 0, max_ctx = 0;         // capacity: B * S <= rows, pos + n <= max_ctx

    // GpuOps' scratch (GpuOps::workspace allocates it), one set per workspace so lanes on two queues never share:
    // [rows, ...] bf16: h (residual after attn), xn (normed), t (projection out), g / u (MLP), qg (q_proj out);
    // splitk: qmm_t_splitk's and qmm_t_decode's fp32 partial sums. Attention: ao (attention out, MLX's [B, H, S, D]); partials / sums /
    // maxs (sdpa 2-pass); kc / vc (prefill: the row's K/V gathered, [Hkv, max_ctx, D]); scores [H, rows, max_ctx].
    // Linear attention: nqkv (normed q / k, [rows, conv_dim]), gdn_g (fp32) / beta [rows, Hv].
    // ids: lm_head's argmax per row (int32), read by sample after the command buffer completed.
    id<MTLBuffer> h, xn, t, g, u, qg, splitk, ao, partials, sums, maxs, kc, vc, scores, nqkv, gdn_g, beta, ids;
};

// Per-step uint32 slot lists the GPU reads, 2 copies by step parity (memory-design "Step tables").
// Layout (uint32 words, R rows): [R][n_state][pos x R][n x R][R + 1 offsets][slots...][state x R]; row b's pages are
// slots[off[b] .. off[b + 1]), its state units (linear layers; every row the same count, 0 or state_units) follow
// all pages: state[b * count .. (b + 1) * count). pos: the row's first token position, n: its valid tokens this
// step. The kernels read it as is (src/kernels/table.h Table); rope reads the pos words as its int offsets.
struct StepTables {
    struct Row {
        int pos, n;
        std::span<const uint32_t> slots;
        std::span<const uint32_t> state = {};  // the request's state units, in span order
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
    // what the constructor computes, before one exists: unit bytes, and units per state slot
    static size_t unit_of(const ModelSpec& s, int page_tokens = 32);
    static uint32_t state_units_of(const ModelSpec& s, int page_tokens = 32);
    // arena bytes for `slots` requests of `tokens` tokens each: pages plus a state slot per request
    static size_t arena_for(const ModelSpec& s, size_t slots, size_t tokens, int page_tokens = 32) {
        return slots * ((tokens + page_tokens - 1) / page_tokens + state_units_of(s, page_tokens)) * unit_of(s, page_tokens);
    }

    uint32_t units_for(size_t n_tokens) const { return (uint32_t)((n_tokens + T - 1) / T); }
    // Byte offsets, identical to memory.py's numpy views. KV within a unit: [g, kv, head, token, dim], g = index
    // among full layers. State: a slot list concatenated into one span; per linear layer conv then delta.
    size_t kv_offset(int g, int kv, int head, int token) const;
    size_t state_offset(int layer) const;

    Workspace workspace(int chunk, int max_batch);
    // max_ctx bounds a row's pages (default: only the arena does)
    StepTables step_tables(int max_rows, size_t max_ctx = SIZE_MAX / 2);
    // The engine calls this periodically so the OS keeps the arena and weights wired.
    void request_residency() { [residency requestResidency]; }

    id<MTLBuffer> buffer(size_t bytes, MTLResourceOptions options = MTLResourceStorageModeShared);  // every allocation
    // ANE I/O: an IOSurface-backed fp16 [rows, cols] CVPixelBuffer (Core ML reads / writes it) and a no-copy MTLBuffer
    // over the same bytes (tools/ane_iosurface.swift). Element (r, c) at r * row + c.
    struct Surface {
        id pixel_buffer;  // CVPixelBufferRef, owned
        id<MTLBuffer> buffer;
        uint32_t row;  // elements per row (bytesPerRow / 2)
    };
    Surface surface(int rows, int cols);
    void freeze() { frozen = true; }  // after startup: any further buffer() asserts

    ModelSpec spec;
    id<MTLDevice> device;
    int T;  // tokens per page
    Arena arena;
    id<MTLBuffer> arena_buffer;  // n_units * unit_bytes, shared, untracked, pre-touched, in the residency set
    size_t conv_bytes, delta_bytes;
    uint32_t state_units;
    int allocations = 0;
    size_t allocated = 0;  // bytes of every allocation above (the arena included)

    id<MTLResidencySet> residency;  // the arena; the engine adds the weights and attaches it to the lanes' queues

private:
    std::vector<size_t> state_off;  // per layer; linear layers only
    bool frozen = false;
};

}  // namespace mllm
