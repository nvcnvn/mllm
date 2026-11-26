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

// Projections GpuOps runs as one multiply when there is a single row (its merged QuantizedWeights), by module name:
// the weights cache keeps each chain's weight / scales / biases back to back (memory.mm aligned_weight_files).
inline const std::vector<std::vector<std::string>> MERGED = {
    {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"},
    {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"},
    {"mlp.gate_proj", "mlp.up_proj"}};

// The model's safetensors, rewritten once into the cache (memory.mm aligned_weight_files), mmapped; each file wrapped
// once as a no-copy MTLBuffer. A tensor = buffer + offset.
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
// Unit states: free -> live (alloc) -> pending (last ref released) -> free (retire, once the GPU passed that step).
class Arena {
public:
    Arena(size_t unit_bytes, uint32_t n_units);

    // memory-design's form; reserved <= free_units always holds, so neither side wraps
    bool fits(uint32_t count) const { return count <= free_units - reserved; }
    bool can_ever(uint32_t count) const { return count <= free_units + cache_only_units - reserved; }
    // reserve count units; false = wait (the engine rejects count > n_units on arrival). evict: the prefix cache's
    // next victim (false = nothing left), called while !fits(count); can_ever(count) means it gets there.
    bool admit(uint32_t count, const std::function<bool()>& evict = {});
    uint32_t alloc();  // one unit out of the caller's reservation; needs a free (retired) unit
    void unreserve(uint32_t count);
    void release(const std::vector<uint32_t>& units);  // drop one request ref per unit
    void ref(uint32_t unit);            // one more request ref on a held unit (a cached page or snapshot)
    void cache_ref(uint32_t unit);      // the prefix cache takes a unit a request holds
    void cache_release(uint32_t unit);  // the prefix cache drops it
    void retire(uint64_t signaled);     // pending units whose step <= signaled (GPU event value) become free

    size_t unit_bytes;
    uint32_t n_units;
    uint64_t committed = 0;  // last committed step: released units stay pending until the GPU passes it (lanes set it, g5)

    std::vector<uint64_t> free_bitmap;               // bit set = reusable now
    std::vector<uint16_t> request_refs, cache_refs;  // per unit
    std::deque<std::pair<uint32_t, uint64_t>> pending;  // (unit, step), steps non-decreasing
    uint32_t free_units;  // no references: free or pending
    uint32_t cache_only_units = 0, reserved = 0;

private:
    void make_pending(uint32_t unit);  // no refs left: pending until the GPU passes committed
};

// Activation buffers, sized once for rows = max(chunk, max_batch) (py/mllm/gpu.py's docstring has the fields under
// their old names; docs/glossary.md maps them). Row-major [rows, ...] in bf16, except token_ids (int32).
struct Workspace {
    id<MTLBuffer> token_ids, residual, attention_result, query, key, value, query_gate;
    id<MTLBuffer> query_key_value, output_gate, update_rate_input, decay_input;  // linear attention's inputs
    id<MTLBuffer> next_token_scores;  // [max_batch, vocab]
    // token_ids' shape this step (the lane sets it): apply_weights works on request_count * tokens_per_request rows
    int request_count = 0, tokens_per_request = 0;
    id<MTLBuffer> step_table;  // this step's StepTables copy, row b = workspace row b (the lane sets it)
    // speculative verify (one request, n tokens): linear layers read the row's first state set and write the state
    // after token t into set 1 + t, so a rollback picks a set (the row lists 1 + n sets of state units)
    bool checkpoint_states = false;
    id<MTLComputeCommandEncoder> encoder;  // GpuOps encodes into it (the lane sets it)
    int max_rows = 0, max_context = 0;     // capacity: rows <= max_rows, position + tokens <= max_context

    // GpuOps' scratch (GpuOps::workspace allocates it), one set per workspace so lanes on two queues never share.
    // [rows, ...] bf16: after_attention (the residual after attention), normalized, projected (a projection's
    // output), gate_values / up_values (feed-forward), query_and_gate (query_projection's output).
    // partial_sums: the fp32 partial sums of the split multiplies (MLX qmm_t_splitk and qmm_t_decode).
    // Attention: attention_heads (attention's output, MLX's [requests, heads, tokens, head_dim]); two_pass_partials /
    // two_pass_sums / two_pass_maxes (MLX sdpa 2-pass); gathered_keys / gathered_values (prefill: the row's keys and
    // values gathered, [key_value_heads, max_context, head_dim]); attention_scores [heads_per_key_value, rows,
    // max_context] (one key_value head group at a time).
    // Linear attention: normalized_query_key (normalized query / key, [rows, conv_dim]), decay (fp32) / update_rate
    // [rows, linear_value_heads].
    // next_token_ids: the highest-scoring token per row (int32), read by sample_tokens after the command buffer
    // completed.
    id<MTLBuffer> after_attention, normalized, projected, gate_values, up_values, query_and_gate, partial_sums;
    id<MTLBuffer> attention_heads, two_pass_partials, two_pass_sums, two_pass_maxes, gathered_keys, gathered_values;
    id<MTLBuffer> attention_scores, normalized_query_key, decay, update_rate, next_token_ids;
};

// Per-step uint32 unit lists the GPU reads, 2 copies by step parity (memory-design "Step tables").
// Layout (uint32 words, R rows): [R][state units per row][start_position x R][token_count x R][R + 1 offsets]
// [pages...][state x R]; row b's pages are pages[offset[b] .. offset[b + 1]), its state units (linear layers; every
// row the same count, 0 or state_units) follow all pages: state[b * count .. (b + 1) * count). start_position: the
// row's first token position, token_count: its valid tokens this step. The kernels read it as is
// (src/kernels/table.h Table); rope reads the start_position words as its int offsets.
struct StepTables {
    struct Row {
        int start_position, token_count;
        std::span<const uint32_t> pages;
        std::span<const uint32_t> state = {};  // the request's state units, in span order
    };
    id<MTLBuffer> copies[2];
    size_t words;      // capacity of each copy, in uint32
    uint32_t n_units;  // the table lists arena units
    // The copy to fill for step. The CPU may run one step ahead of the GPU: asserts signaled + 2 >= step.
    id<MTLBuffer> begin(uint64_t step, uint64_t signaled);
    void write(id<MTLBuffer> table, const std::vector<Row>& rows) const;
};

class Memory {
public:
    Memory(const ModelSpec& spec, id<MTLDevice> device, size_t arena_bytes, int page_tokens = 32);
    // what the constructor computes, before one exists: unit bytes, and units per state slot
    static size_t unit_of(const ModelSpec& model, int page_tokens = 32);
    static uint32_t state_units_of(const ModelSpec& model, int page_tokens = 32);
    // arena bytes for `slots` requests of `tokens` tokens each: pages plus a state slot per request
    static size_t arena_for(const ModelSpec& model, size_t slots, size_t tokens, int page_tokens = 32) {
        return slots * ((tokens + page_tokens - 1) / page_tokens + state_units_of(model, page_tokens)) *
               unit_of(model, page_tokens);
    }

    uint32_t units_for(size_t tokens) const { return (uint32_t)((tokens + page_tokens - 1) / page_tokens); }
    // Byte offsets, identical to memory.py's numpy views. Keys and values within a unit:
    // [full_layer, key_or_value, head, token, head_dim]; full_layer = the index among full-attention layers,
    // key_or_value 0 = key, 1 = value. State: a slot list concatenated into one span; per linear layer conv then delta.
    size_t kv_offset(int full_layer, int key_or_value, int head, int token) const;
    size_t state_offset(int layer) const;

    Workspace workspace(int chunk, int max_batch);
    // max_context bounds a row's pages (default: only the arena does)
    StepTables step_tables(int max_rows, size_t max_context = SIZE_MAX / 2);
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
    int page_tokens;  // tokens per page
    Arena arena;
    id<MTLBuffer> arena_buffer;  // n_units * unit_bytes, shared, untracked, pre-touched, in the residency set
    size_t conv_bytes, delta_bytes;
    uint32_t state_units;
    int allocations = 0;
    size_t allocated = 0;  // bytes of every allocation above (the arena included)

    id<MTLResidencySet> residency;  // the arena; the engine adds the weights and attaches it to the lanes' queues

private:
    std::vector<size_t> state_offsets;  // per layer; linear layers only (SIZE_MAX for the others)
    bool frozen = false;
};

}  // namespace mllm
