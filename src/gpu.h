// GpuOps: apply_weights(i), apply_attention(i), pick_next_tokens and sampling on Metal (py/mllm/gpu.py: dense, attn,
// lm_head; Python samples on the CPU). Kernels are MLX's (src/vendor/mlx), dispatched the way MLX v0.32.4 dispatches
// them, so results match mlx-lm bit for bit.
#pragma once
#import <Metal/Metal.h>

#include <cassert>
#include <functional>
#include <map>
#include <random>
#include <thread>

#include "gpu_rules.h"
#include "memory.h"

namespace mllm {

// A tensor bound for a kernel: buffer + byte offset.
struct Buf {
    id<MTLBuffer> buffer;
    size_t offset = 0;
};

// 4-bit affine weights, MLX format: packed U32 [out_size, in_size / 8], scales / biases bf16 [out_size, in_size / 64].
// output = input @ dequantize(packed)ᵀ.
struct QuantizedWeights {
    Buf packed, scales, biases;
    int out_size = 0, in_size = 0;
};

// One row's sampler, mlx-lm make_sampler(temp, top_p): temperature 0 = greedy; top_p in (0, 1) keeps the nucleus.
struct SampleParams {
    float temperature = 0, top_p = 0;
    std::mt19937_64* rng = nullptr;  // the request's; needed when temperature > 0
};

class GpuOps {
public:
    GpuOps(Memory& mem, const Weights& weights);

    // mem.workspace(chunk, max_batch) plus GpuOps' scratch, sized for max_ctx tokens per request. One per lane.
    Workspace workspace(int chunk, int max_batch, int max_ctx);

    // Every op below encodes into ws.encoder and uses ws's scratch.
    // All the big weight multiplies between apply_attention(i - 1) and apply_attention(i), i = 0 .. n_layers, over
    // ws.request_count * ws.tokens_per_request rows. Fields as in gpu.py, except query, key, value are row-major
    // [requests, tokens, heads, head_dim] (gpu.py's [requests, heads, tokens, head_dim] transpose is only for MLX's
    // attention kernel).
    void apply_weights(int i, Workspace& ws);

    // Everything of layer i that touches the arena (gpu.py attn): ws.query / key / value / query_gate of
    // ws.request_count rows x ws.tokens_per_request tokens -> ws.attention_result [requests, tokens, heads * head_dim].
    // ws.step_table: this step's StepTables copy (row b = ws row b); only each row's n valid tokens are written.
    // Full layers: RoPE, keys / values into pages, attention over pos + n keys, * sigmoid(query_gate).
    // Linear layers: ws.query_key_value / output_gate / update_rate_input / decay_input -> conv (its state updated in
    // the row's state span), gated delta rule (state updated in place), gated RMSNorm -> ws.attention_result
    // [requests, tokens, linear_value_heads * linear_value_dim]; ws.query_key_value is overwritten.
    // Rows may differ in pos and pages; more than one request needs tokens_per_request == 1 (decode).
    void apply_attention(int i, Workspace& ws);

    // gpu.py lm_head and sample: scores of ws.residual's request_count rows (decode, one token each: prefill never
    // samples) -> ws.next_token_scores [requests, vocab_size] bf16, then each row's token -> ws.next_token_ids:
    // params[b] (empty: all greedy) as sample_tokens() below.
    // next_tokens, after that command buffer completed: ws.next_token_ids' request_count tokens.
    void pick_next_tokens(Workspace& ws, const std::vector<SampleParams>& params = {});
    std::vector<int> next_tokens(const Workspace& ws) const;
    // set: next_tokens hands it the step's scores too, request_count rows of vocab_size bf16 (generate --logits)
    std::function<void(const uint16_t* scores, size_t count)> on_scores;
    // bf16 scores [rows = params.size(), vocab_size] -> int32 token per row (sample.metal): temperature 0 = argmax;
    // else top_p, then a draw seeded from the row's rng (one 64-bit draw per call, so a request's tokens follow its seed).
    void sample_tokens(Buf scores, Buf output, int vocab_size, const std::vector<SampleParams>& params);

    // Decode only, no Python counterpart (Python never runs ahead):
    // ws.token_ids[b] = previous_token_ids[source_rows[b]] for source_rows[b] >= 0, so a greedy row's next step is
    // encoded before the CPU has seen its token. Encode it before apply_weights(0).
    void feed_previous_tokens(Workspace& ws, id<MTLBuffer> previous_token_ids, const std::vector<int>& source_rows);


private:
    friend struct GpuOpsTest;  // tests and bench-ops reach the kernel helpers below (Metal types, rule 1)
    friend class AneOps;       // look_up_embeddings and pipeline() for the ANE path

    // The kernels apply_weights() is made of. Encoded into the current command buffer.
    void look_up_embeddings(Buf token_ids, Buf output, int rows);
    // output may be input; weight_stride 0: one weight for all (mx.fast.rms_norm without weight).
    // ponytail: eps default = rms_norm_eps, 1e-6 for every Qwen3.5 (mlx-lm reads config.json); add it to ModelSpec
    // when a model differs.
    void normalize(Buf weight, Buf input, Buf output, int rows, int width, float eps = 1e-6f, uint32_t weight_stride = 1);
    // RMSNormGated(input, gate); output may be input
    void normalize_and_gate(Buf weight, Buf input, Buf gate, Buf output, int rows, int width);
    void multiply_weights(const QuantizedWeights& weights, Buf input, Buf output, int rows);
    void add(Buf first, Buf second, Buf output, size_t count);
    void activate_gate(Buf gate, Buf input, Buf output, size_t count);
    void greedy_tokens(Buf scores, Buf output, int rows, int vocab_size) {
        sample_tokens(scores, output, vocab_size, std::vector<SampleParams>(rows));
    }

    // multiply_weights' paths, by row count (see there)
    void multiply_decode_batch(const QuantizedWeights& weights, Buf input, Buf output, int rows);
    void multiply_many_rows(const QuantizedWeights& weights, Buf input, Buf output, int rows);
    void bind_weights(id<MTLComputePipelineState> kernel, const QuantizedWeights& weights, Buf input);
    void sum_split_parts(Buf output, int rows, int out_size, int split_count, int split_stride);

    struct Layer {
        bool is_linear;
        int full_index = -1;  // index among full layers (its keys / values in a unit)
        Buf attention_norm, feed_forward_norm;
        // attention_output: o_proj (full) or out_proj (linear)
        QuantizedWeights feed_forward_gate, feed_forward_up, feed_forward_down, attention_output;
        QuantizedWeights query_projection, key_projection, value_projection;  // full
        Buf query_norm, key_norm;
        QuantizedWeights query_key_value_projection, output_gate_projection, update_rate_projection, decay_projection;  // linear
        Buf conv_weights, decay_log_base, decay_bias, output_norm;
        // One row (see merge_projections): the projections that read ws.normalized as one multiply, weights back to
        // back along out_size. merged_linear_projections: query_key_value | output_gate | update_rate | decay (linear),
        // merged_query_key_value: query | key | value (full), merged_gate_up: gate | up. The parts above are views into it.
        QuantizedWeights merged_linear_projections, merged_query_key_value, merged_gate_up;
    };
    std::vector<Layer> layers;
    void load_kernels();
    Layer load_layer(const Weights& weights, int i, int& full_layers_so_far);

    // apply_weights' two halves: the rest of layer i - 1 after its attention, then layer i up to its attention
    void finish_layer(const Layer& previous, Workspace& ws, int rows);
    void start_layer(const Layer& layer, Workspace& ws, int rows);

    void full_attention(const Layer& layer, Workspace& ws);
    void rotate_positions(id<MTLBuffer> heads_buffer, int heads, const Workspace& ws);
    void write_keys_values(const Layer& layer, Workspace& ws);
    void attention_fused(const Layer& layer, const Workspace& ws, int shortest_context, int longest_context);
    void attention_unfused(const Layer& layer, const Workspace& ws, int longest_context);

    void linear_attention(int i, const Layer& layer, Workspace& ws);
    void conv_and_normalize(const Layer& layer, Workspace& ws, uint64_t conv_offset);
    void update_state(Buf decay_input, Buf update_rate_input, const Layer& layer, Workspace& ws, uint64_t delta_offset);

    QuantizedWeights embed_tokens;
    QuantizedWeights lm;  // language_model.lm_head, or embed_tokens when tied (Qwen3.5 <= 9B: embed_tokens.as_linear)
    Buf final_norm;

    void elementwise(id<MTLComputePipelineState> kernel, Buf first, Buf second, Buf output, size_t count);
    id<MTLComputePipelineState> pipeline(NSString* name, MTLFunctionConstantValues* constants = nil);

    ModelSpec spec;
    Memory& mem;
    id<MTLDevice> device;
    id<MTLLibrary> library;
    // the current workspace's encoder and split-k scratch, for the helpers above (set by each public op)
    // One thread per GpuOps: these members are per call (the ANE lane's thread has its own copy, AneOps::gpu)
    void begin(const Workspace& ws) {
        if (owner == std::thread::id()) owner = std::this_thread::get_id();
        assert(owner == std::this_thread::get_id() && "GpuOps used from two threads");
        encoder = ws.encoder, partial_sums = ws.partial_sums, is_decode = ws.tokens_per_request == 1;
    }
    std::thread::id owner;  // the first thread that encoded with this object
    id<MTLComputeCommandEncoder> encoder;
    id<MTLBuffer> partial_sums;
    // ws.tokens_per_request == 1: multiply_weights() may pick multiply_decode_batch (batched decode, checked by tolerance)
    bool is_decode = false;

    // Weight multiplies (MLX kernel names in load_kernels): one row, 2..5 rows per threadgroup (index = rows),
    // tiles of 32 x 32 / 64 x 64 [aligned out_size], split along in_size [aligned out_size] and the sum of the splits.
    id<MTLComputePipelineState> multiply_one_row, multiply_few_rows[6], multiply_tiles_32[2], multiply_tiles_64[2],
        multiply_split_inner[2], sum_splits;
    // multiply_tiles_32 in 64 x 64 tiles from this many rows (prefill); shorter tails keep MLX's 32 x 32 (a 64-row tile
    // of a <= 32-row tail is mostly padding). At 33..63 rows 64 x 64 is ~1% faster (27B, rows 40 / 48 / 56: -2..-3% on
    // query / gate / query_key_value_projection, +2% on feed_forward_down). Tests set it to compare the two paths.
    int large_tiles_from_rows = 33;
    // ours, for batched decode: [8 / 16 rows per tile][split-k float partials / bf16 output]
    id<MTLComputePipelineState> multiply_decode_rows[2][2];
    id<MTLComputePipelineState> embed_kernel, normalize_kernel, normalize_wide_kernel, add_kernel, activate_gate_kernel,
        activate_gate_precise_kernel, split_query_gate_kernel;
    // full attention: attention_long_context_blocks [threads per group fit / exceed the limit]
    id<MTLComputePipelineState> rotate_positions_kernel, write_keys_values_kernel, attention_short_context,
        attention_long_context_blocks[2], attention_long_context_combine, sigmoid_gate_kernel;
    // the unfused fallback; softmax_kernel [one block / looped]
    id<MTLComputePipelineState> gather_keys_values_kernel, causal_mask_kernel, scale_kernel, softmax_kernel[2];
    // linear attention, sampling, feed_previous_tokens
    id<MTLComputePipelineState> conv_kernel, conv_and_normalize_kernel, scale_query_key_kernel, decay_and_update_rate_kernel,
        decay_and_update_rate_bf16_kernel, gated_delta_kernel, sample_kernel, feed_tokens_kernel;
    // the fallback's bf16 multiply variants: multiply_few_rows_bf16 [rows per pass 3/4/5][16 / 32 lanes along the
    // inner size]; multiply_tiles_bf16: steel_gemm_fused by name + aligned rows / columns / inner bits, made on first use
    id<MTLComputePipelineState> multiply_few_rows_bf16[3][2];
    std::map<std::string, id<MTLComputePipelineState>> multiply_tiles_bf16;
    GpuClass gpu;       // MLX's device class: picks kernels and tiles (gpu_rules.h)
    int two_pass_from;  // sdpa_vector_2pass from this many keys
    id<MTLComputePipelineState> pipeline(NSString* name, std::initializer_list<std::pair<int, bool>> constants);
    // MLX's bf16 multiply for the attention fallback over the (key_value head, repeat) batches: steel_gemm_fused
    // (second transposed or not) or gemv_wide (second transposed, rows 2..15). *_batch_strides: over
    // [key_value_heads, heads_per_key_value].
    void multiply_bf16(bool second_transposed, Buf first, Buf second, Buf output, int rows, int columns, int inner,
                       int first_row_stride, int second_row_stride, int output_row_stride,
                       const int64_t first_batch_strides[2], const int64_t second_batch_strides[2],
                       int64_t output_batch_stride);

    id<MTLBuffer> one_bf16;  // bf16 1, read only

    // Before every dispatch. The decode lane's encoder is concurrent: a buffer-scope barrier (it covers the untracked
    // arena too) orders each dispatch after the ones before it, unless the call site set independent: the next
    // dispatch neither reads what the dispatches since the last barrier write, nor writes what they read or write
    // (scratch such as ws.partial_sums counts: multiply_weights() clears independent for split-k). Serial encoders
    // ignore barriers.
    bool independent = false;
    void before_dispatch() {
        dispatch_count++;
        if (!independent) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        independent = false;
    }

public:
    // apply_weights(i) on one row runs each Layer's merged multiplies: one output row [part | part ...] in
    // ws.query_key_value (merged_linear_projections), ws.query_and_gate (merged_query_key_value) or ws.gate_values
    // (merged_gate_up), and apply_attention(i) reads output_gate / update_rate / decay / value as slices of it.
    // multiply_one_row computes each output from its own weight row the same way for any out_size, so the values equal
    // the separate multiplies'. False in AneOps' copy (the ANE writes each workspace field) and in tests that load the
    // fields themselves.
    bool merge_projections = true;
    size_t dispatch_count = 0;  // kernels encoded so far (bench-ops reports it per decode step)
};

}  // namespace mllm
