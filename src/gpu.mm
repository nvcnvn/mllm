#include "gpu.h"

#include "mlx/mlx/backend/metal/kernels/steel/gemm/params.h"

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace mllm {

// ponytail: Qwen3.5's RoPE (mlx-lm nn.RoPE(64, traditional=False, base=1e7); mrope is a no-op for text). spec.py has
// no rope fields yet; add them when a model differs.
constexpr int ROTATED_DIMS = 64;        // the first 64 numbers of each head rotate by position
constexpr float ROTATION_BASE = 1e7f;  // RoPE's theta
// MLX's 'medium' (applegpu_g*s) attention rules: the two-pass kernel from 1024 keys
constexpr int TWO_PASS_FROM_KEYS = 1024;

// How many blocks the two-pass attention splits the keys into on an 's' GPU (scaled_dot_product_attention.cpp
// sdpa_vector_2pass). simdgroups: query heads per key_value head x queries.
static int two_pass_blocks(int keys, int simdgroups) {
    if (keys > 1024 && simdgroups > 4) return keys <= 8192 ? 128 : keys <= 32768 ? 256 : keys <= 65536 ? 512 : 1024;
    return 64;
}

// MLX's has_fused_kernel: the fused attention kernel (sdpa_vector) only while queries <= 8 and
// queries x heads_per_key_value <= 32 (27B, 6 query heads per key_value head: queries <= 5), else the unfused fallback
static bool fused_attention_fits(int queries, int heads_per_key_value) {
    return queries <= 8 && queries * heads_per_key_value <= 32;
}

// attention.metal KVLayout: where a full layer's keys / values sit in the arena's pages
struct KeyValueLayout {
    uint64_t unit_elements;  // bf16 elements per arena unit
    int page_tokens, key_value_heads, full_layer;
};

static std::runtime_error error(const std::string& what, NSError* err) {
    return std::runtime_error(what + (err ? std::string(": ") + err.localizedDescription.UTF8String : ""));
}

static void set_buffer(id<MTLComputeCommandEncoder> encoder, Buf buffer, int index) {
    [encoder setBuffer:buffer.buffer offset:buffer.offset atIndex:index];
}
template <class T>
static void set_value(id<MTLComputeCommandEncoder> encoder, T value, int index) {
    [encoder setBytes:&value length:sizeof(T) atIndex:index];
}

id<MTLComputePipelineState> GpuOps::pipeline(NSString* name, MTLFunctionConstantValues* constants) {
    NSError* err = nil;
    id<MTLFunction> function = constants ? [library newFunctionWithName:name constantValues:constants error:&err]
                                         : [library newFunctionWithName:name];
    if (!function) throw error(std::string("no kernel ") + name.UTF8String + " in " + METALLIB, err);
    id<MTLComputePipelineState> kernel = [device newComputePipelineStateWithFunction:function error:&err];
    if (!kernel) throw error(std::string("pipeline ") + name.UTF8String, err);
    return kernel;
}

// bool function constants (index, value)
id<MTLComputePipelineState> GpuOps::pipeline(NSString* name, std::initializer_list<std::pair<int, bool>> constants) {
    MTLFunctionConstantValues* values = [MTLFunctionConstantValues new];
    for (auto [index, value] : constants) [values setConstantValue:&value type:MTLDataTypeBool atIndex:index];
    return pipeline(name, values);
}

// A tensor of the model file as a kernel binding
static Buf tensor(const Weights& weights, const std::string& name) {
    auto view = weights[name];
    return Buf{view.buffer, view.offset};
}

static QuantizedWeights quantized_weights(const Weights& weights, const std::string& prefix) {
    auto packed = weights[prefix + ".weight"];
    auto scales = weights[prefix + ".scales"];
    QuantizedWeights result{tensor(weights, prefix + ".weight"), tensor(weights, prefix + ".scales"),
                            tensor(weights, prefix + ".biases"), (int)packed.t.shape[0], (int)packed.t.shape[1] * 8};
    if (packed.t.dtype != "U32" || scales.t.dtype != "BF16" || scales.t.shape[1] * 64 != result.in_size)
        throw std::runtime_error(prefix + ": not 4-bit affine, group 64, bf16");
    // multiply_weights() ports only qmv_fast for one row: no qmv_quad (in_size 64 / 128), qmv (in_size % 512) or
    // partial rows (out_size % 8)
    if (result.in_size % 512 || result.out_size % 8)
        throw std::runtime_error(prefix + ": K % 512 or N % 8 not 0 (kernel not ported)");
    return result;
}

// The merged weights: the weights cache keeps each chain's parts back to back (memory.h MERGED), so the merged
// multiply is a view starting at the first part.
static QuantizedWeights merged_weights(std::initializer_list<QuantizedWeights*> parts) {
    const QuantizedWeights& first = **parts.begin();
    QuantizedWeights merged{.packed = first.packed, .scales = first.scales, .biases = first.biases, .in_size = first.in_size};
    for (QuantizedWeights* part : parts) {
        const size_t packed_row = merged.in_size / 2, scales_row = merged.in_size / 64 * ELEM;
        for (auto [whole, piece, row_bytes] : {std::tuple{&merged.packed, &part->packed, packed_row},
                                               {&merged.scales, &part->scales, scales_row},
                                               {&merged.biases, &part->biases, scales_row}}) {
            if (part->in_size != merged.in_size || piece->buffer != whole->buffer ||
                piece->offset != whole->offset + merged.out_size * row_bytes)
                throw std::runtime_error("GpuOps: merged projection parts not adjacent in the weights cache");
        }
        merged.out_size += part->out_size;
    }
    return merged;
}

GpuOps::GpuOps(Memory& mem, const Weights& weights) : spec(mem.spec), mem(mem), device(mem.device) {
    // MLX picks kernels and tile sizes by GPU generation. Only the M4 family's branch (gen 15/16, not a 'd' Max/Ultra
    // die) is copied below. ponytail: port the other branches of quantized.cpp when another GPU shows up.
    NSString* architecture = device.architecture.name;
    if (!([architecture hasPrefix:@"applegpu_g15"] || [architecture hasPrefix:@"applegpu_g16"]) ||
        [architecture hasSuffix:@"d"])
        throw error(std::string("GpuOps: dispatch rules not ported for ") + architecture.UTF8String, nil);

    NSError* err = nil;
    unsigned long embedded_size = 0;  // build/mllm carries the metallib in a section (Makefile); other binaries load the file
    const uint8_t* embedded =
        getsectiondata((const mach_header_64*)_dyld_get_image_header(0), "__TEXT", "__metallib", &embedded_size);
    library = embedded ? [device newLibraryWithData:dispatch_data_create(embedded, embedded_size, nullptr, ^{}) error:&err]
                       : [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&err];
    if (!library) throw error(std::string("cannot load ") + METALLIB, err);
    load_kernels();

    const std::string model = "language_model.model.";
    embed_tokens = quantized_weights(weights, model + "embed_tokens");
    lm = weights.st.tensors.count("language_model.lm_head.weight") ? quantized_weights(weights, "language_model.lm_head")
                                                                   : embed_tokens;
    final_norm = tensor(weights, model + "norm.weight");
    for (int i = 0, full_layers_so_far = 0; i < spec.n_layers(); i++)
        layers.push_back(load_layer(weights, i, full_layers_so_far));
    for (Layer& layer : layers) {
        if (layer.is_linear)
            layer.merged_linear_projections =
                merged_weights({&layer.query_key_value_projection, &layer.output_gate_projection,
                                &layer.update_rate_projection, &layer.decay_projection});
        else
            layer.merged_query_key_value =
                merged_weights({&layer.query_projection, &layer.key_projection, &layer.value_projection});
        layer.merged_gate_up = merged_weights({&layer.feed_forward_gate, &layer.feed_forward_up});
    }

    one_bf16 = mem.buffer(ELEM);
    *(uint16_t*)one_bf16.contents = 0x3F80;  // bf16 1.0
}

// Every kernel, by its name in the metallib (MLX's names, or ours from src/kernels).
void GpuOps::load_kernels() {
    multiply_one_row = pipeline(@"affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4");
    for (int rows = 2; rows <= 5; rows++)
        multiply_few_rows[rows] =
            pipeline([NSString stringWithFormat:@"affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_%d_kl_8_batch_0", rows]);
    for (int aligned = 0; aligned < 2; aligned++) {
        NSString* name = aligned ? @"true" : @"false";
        multiply_split_inner[aligned] =
            pipeline([NSString stringWithFormat:@"affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_%@", name]);
        multiply_tiles_32[aligned] =
            pipeline([NSString stringWithFormat:@"affine_qmm_t_bfloat16_t_gs_64_b_4_alN_%@_batch_0", name]);
        multiply_tiles_64[aligned] = pipeline([NSString stringWithFormat:@"qmm_t_64_alN_%@", name]);
    }
    sum_splits = pipeline(@"steel_gemm_splitk_accum_bfloat16_float32");
    for (int sixteen_rows = 0; sixteen_rows < 2; sixteen_rows++)
        for (int split = 0; split < 2; split++)
            multiply_decode_rows[sixteen_rows][split] = pipeline(
                [NSString stringWithFormat:@"qmm_t_decode_%d_%@", sixteen_rows ? 16 : 8, split ? @"float" : @"bf16"]);
    embed_kernel = pipeline(@"embed");
    normalize_kernel = pipeline(@"rmsbfloat16");
    normalize_wide_kernel = pipeline(@"rms_loopedbfloat16");
    add_kernel = pipeline(@"add");
    activate_gate_kernel = pipeline(@"swiglu");
    activate_gate_precise_kernel = pipeline(@"precise_swiglu");
    split_query_gate_kernel = pipeline(@"split_q_gate");

    // full attention: kernel names fix head_dim 256 (bf16)
    if (!spec.full_layers().empty() && spec.head_dim != 256)
        throw std::runtime_error("GpuOps: attention kernels instantiated for head_dim 256 only");
    rotate_positions_kernel = pipeline(@"rope_bfloat16", {{1, true}, {2, false}, {3, false}});  // forward, traditional, hs_transpose
    write_keys_values_kernel = pipeline(@"write_kv_bfloat16_256");
    attention_short_context = pipeline(@"sdpa_vector_paged_bfloat16_256");
    attention_long_context_blocks[0] = pipeline(@"sdpa_vector_2pass_1_paged_bfloat16_256");
    attention_long_context_blocks[1] = pipeline(@"sdpa_vector_2pass_1_paged_bfloat16_256_tk1");
    attention_long_context_combine = pipeline(@"sdpa_vector_2pass_2_bfloat16_256");
    sigmoid_gate_kernel = pipeline(@"sigmoid_gate");
    // linear attention: kernel names fix the conv kernel and head shapes
    if (!spec.linear_layers().empty() &&
        (spec.conv_kernel != 4 || spec.linear_key_heads != 16 ||
         (spec.linear_value_heads != 16 && spec.linear_value_heads != 48) || spec.linear_key_dim != 128 ||
         spec.linear_value_dim != 128))
        throw std::runtime_error("GpuOps: linear attention kernels instantiated for Qwen3.5-0.8B's and 27B's shapes only");
    conv_kernel = pipeline(@"conv1d_silu_4");
    conv_and_normalize_kernel = pipeline(@"conv_norm_qk_4");
    scale_query_key_kernel = pipeline(@"scale_qk");
    decay_and_update_rate_kernel = pipeline(@"gdn_gates_float");
    decay_and_update_rate_bf16_kernel = pipeline(@"gdn_gates_bfloat16");
    gated_delta_kernel =
        pipeline([NSString stringWithFormat:@"gated_delta_paged_bfloat16_128_128_16_%d", spec.linear_value_heads]);
    sample_kernel = pipeline(@"sample_bf16");
    feed_tokens_kernel = pipeline(@"feed_tokens");
    gather_keys_values_kernel = pipeline(@"gather_kv_bfloat16_256");
    causal_mask_kernel = pipeline(@"causal_mask");
    scale_kernel = pipeline(@"scale_bf16");
    softmax_kernel[0] = pipeline(@"block_softmax_precise_bfloat16");
    softmax_kernel[1] = pipeline(@"looped_softmax_precise_bfloat16");
    for (int rows = 0; rows < 3; rows++)
        for (int wide = 0; wide < 2; wide++)  // has_batch, do_axpby
            multiply_few_rows_bf16[rows][wide] = pipeline(
                [NSString stringWithFormat:@"gemv_wide_bfloat16_nv%d_kl%d", 3 + rows, 16 << wide], {{0, true}, {1, false}});
    // the 16 tile variants: bit 3 = not transposed (nn), bits 2..0 = rows, columns, inner size aligned
    for (int variant = 0; variant < 16; variant++) {
        const bool not_transposed = variant & 8, rows_aligned = variant & 4, columns_aligned = variant & 2,
                   inner_aligned = variant & 1;
        // has_batch, use_out_source, do_axpby, align_M, align_N, align_K
        multiply_tiles_bf16[not_transposed][rows_aligned][columns_aligned][inner_aligned] = pipeline(
            not_transposed ? @"steel_gemm_fused_nn_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2"
                           : @"steel_gemm_fused_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2",
            {{10, true}, {100, false}, {110, false}, {200, rows_aligned}, {201, columns_aligned}, {202, inner_aligned}});
    }
}

// Layer i's weights, each field from its model-file key (docs/glossary.md maps one to the other)
GpuOps::Layer GpuOps::load_layer(const Weights& weights, int i, int& full_layers_so_far) {
    const std::string prefix = "language_model.model.layers." + std::to_string(i) + ".";
    auto tensor_at = [&](const std::string& name) { return tensor(weights, prefix + name); };
    auto weights_at = [&](const std::string& name) { return quantized_weights(weights, prefix + name); };
    Layer layer{};
    layer.is_linear = spec.kinds[i] == "linear";
    if (!layer.is_linear) layer.full_index = full_layers_so_far++;
    layer.attention_norm = tensor_at("input_layernorm.weight");
    layer.feed_forward_norm = tensor_at("post_attention_layernorm.weight");
    layer.feed_forward_gate = weights_at("mlp.gate_proj");
    layer.feed_forward_up = weights_at("mlp.up_proj");
    layer.feed_forward_down = weights_at("mlp.down_proj");
    if (!layer.is_linear) {
        layer.attention_output = weights_at("self_attn.o_proj");
        layer.query_projection = weights_at("self_attn.q_proj");
        layer.key_projection = weights_at("self_attn.k_proj");
        layer.value_projection = weights_at("self_attn.v_proj");
        layer.query_norm = tensor_at("self_attn.q_norm.weight");
        layer.key_norm = tensor_at("self_attn.k_norm.weight");
        return layer;
    }
    layer.attention_output = weights_at("linear_attn.out_proj");
    layer.query_key_value_projection = weights_at("linear_attn.in_proj_qkv");
    layer.output_gate_projection = weights_at("linear_attn.in_proj_z");
    layer.update_rate_projection = weights_at("linear_attn.in_proj_b");
    layer.decay_projection = weights_at("linear_attn.in_proj_a");
    layer.conv_weights = tensor_at("linear_attn.conv1d.weight");
    layer.decay_log_base = tensor_at("linear_attn.A_log");
    layer.decay_bias = tensor_at("linear_attn.dt_bias");
    layer.output_norm = tensor_at("linear_attn.norm.weight");
    const std::string decay_log_base_type = weights[prefix + "linear_attn.A_log"].t.dtype;
    if ((decay_log_base_type != "F32" && decay_log_base_type != "BF16") ||
        weights[prefix + "linear_attn.conv1d.weight"].t.dtype != "BF16")
        throw std::runtime_error(prefix + "linear_attn: A_log F32 / BF16 and conv1d BF16 expected");
    if (decay_log_base_type == "BF16") decay_and_update_rate_kernel = decay_and_update_rate_bf16_kernel;
    return layer;
}

Workspace GpuOps::workspace(int chunk, int max_batch, int max_ctx) {
    Workspace ws = mem.workspace(chunk, max_batch);
    ws.max_rows = std::max(chunk, max_batch), ws.max_context = max_ctx;
    size_t row_bytes = (size_t)ws.max_rows * ELEM;
    ws.after_attention = mem.buffer(row_bytes * spec.hidden);
    ws.normalized = mem.buffer(row_bytes * spec.hidden);
    ws.projected = mem.buffer(row_bytes * spec.hidden);
    // merge_projections: the one-row merged outputs land in ws.gate_values, ws.query_and_gate and ws.query_key_value
    // (max_rows may be 1)
    ws.gate_values = mem.buffer(std::max(row_bytes * layers[0].feed_forward_gate.out_size, layers[0].merged_gate_up.out_size * ELEM));
    ws.up_values = mem.buffer(row_bytes * layers[0].feed_forward_gate.out_size);
    size_t merged_query_key_value_size = 0, merged_linear_size = 0;
    for (const Layer& layer : layers) {
        merged_query_key_value_size = std::max<size_t>(merged_query_key_value_size, layer.merged_query_key_value.out_size);
        merged_linear_size = std::max<size_t>(merged_linear_size, layer.merged_linear_projections.out_size);
    }
    ws.query_and_gate = mem.buffer(std::max(row_bytes * 2 * spec.query_heads * spec.head_dim, merged_query_key_value_size * ELEM));
    if (ws.query_key_value.length < merged_linear_size * ELEM) ws.query_key_value = mem.buffer(merged_linear_size * ELEM);
    // split count * rows * out_size <= 512 tiles of 32 x 32 (see multiply_weights)
    ws.partial_sums = mem.buffer(512 * 32 * 32 * sizeof(float));
    const size_t heads = spec.query_heads, head_dim = spec.head_dim, heads_per_key_value = heads / spec.key_value_heads;
    ws.attention_heads = mem.buffer(row_bytes * heads * head_dim);
    // two-pass partials at max_ctx keys: decode (max_batch rows of 1 query; more than 4 heads per key_value head takes
    // up to 1024 blocks) or one prefill row of up to fused_queries queries (the fused path's longest)
    const int fused_queries = std::min({chunk, 8, 32 / (int)heads_per_key_value});
    size_t partial_rows = std::max(max_batch * heads * two_pass_blocks(max_ctx, heads_per_key_value),
                                   fused_queries * heads * two_pass_blocks(max_ctx, heads_per_key_value * fused_queries));
    ws.two_pass_partials = mem.buffer(partial_rows * head_dim * ELEM);
    ws.two_pass_sums = mem.buffer(partial_rows * sizeof(float));
    ws.two_pass_maxes = mem.buffer(partial_rows * sizeof(float));
    // the unfused fallback only: decode workspaces (chunk on the fused path) never gather
    const size_t gathered_rows = fused_attention_fits(chunk, (int)heads_per_key_value) ? 1 : ws.max_context;
    ws.gathered_keys = mem.buffer(spec.key_value_heads * gathered_rows * head_dim * ELEM);
    ws.gathered_values = mem.buffer(spec.key_value_heads * gathered_rows * head_dim * ELEM);
    ws.attention_scores = mem.buffer((size_t)std::max(chunk, 1) * ELEM * heads_per_key_value * gathered_rows);
    ws.normalized_query_key = mem.buffer(row_bytes * spec.conv_dim());
    ws.decay = mem.buffer((size_t)ws.max_rows * spec.linear_value_heads * sizeof(float));
    ws.update_rate = mem.buffer(row_bytes * spec.linear_value_heads);
    ws.next_token_ids = mem.buffer((size_t)ws.max_rows * sizeof(int32_t));
    return ws;
}

void GpuOps::look_up_embeddings(Buf token_ids, Buf output, int rows) {
    const QuantizedWeights& table = embed_tokens;
    [encoder setComputePipelineState:embed_kernel];
    set_buffer(encoder, table.packed, 0), set_buffer(encoder, table.scales, 1), set_buffer(encoder, table.biases, 2);
    set_buffer(encoder, token_ids, 3), set_buffer(encoder, output, 4);
    set_value(encoder, table.in_size, 5);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(table.in_size / 8, rows, 1)
                           threadsPerThreadgroup:MTLSizeMake(std::min(table.in_size / 8, 256), 1, 1)];
}

// mlx RMSNorm::eval_gpu: rms_single_row up to RMS_LOOPED_LIMIT (4096) wide, rms_looped above (27B hidden 5120)
void GpuOps::normalize(Buf weight, Buf input, Buf output, int rows, int width, float eps, uint32_t weight_stride) {
    const bool looped = width > 4096;
    id<MTLComputePipelineState> kernel = looped ? normalize_wide_kernel : normalize_kernel;
    uint32_t threads_per_group = looped ? (uint32_t)kernel.maxTotalThreadsPerThreadgroup : 32 * (((width + 3) / 4 + 31) / 32),
             axis_size = width;
    assert(threads_per_group <= kernel.maxTotalThreadsPerThreadgroup);
    [encoder setComputePipelineState:kernel];
    set_buffer(encoder, input, 0), set_buffer(encoder, weight, 1), set_buffer(encoder, output, 2);
    set_value(encoder, eps, 3), set_value(encoder, axis_size, 4), set_value(encoder, weight_stride, 5);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake((size_t)rows * threads_per_group, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
}

// Qwen3NextRMSNormGated: precise_swiglu(input, gate, normalize(input, weight))
void GpuOps::normalize_and_gate(Buf weight, Buf input, Buf gate, Buf output, int rows, int width) {
    normalize(weight, input, output, rows, width);
    elementwise(activate_gate_precise_kernel, gate, output, output, (size_t)rows * width);
}

void GpuOps::elementwise(id<MTLComputePipelineState> kernel, Buf first, Buf second, Buf output, size_t count) {
    [encoder setComputePipelineState:kernel];
    set_buffer(encoder, first, 0), set_buffer(encoder, second, 1), set_buffer(encoder, output, 2);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(count, 256), 1, 1)];
}

void GpuOps::add(Buf first, Buf second, Buf output, size_t count) { elementwise(add_kernel, first, second, output, count); }
void GpuOps::activate_gate(Buf gate, Buf input, Buf output, size_t count) {
    elementwise(activate_gate_kernel, gate, input, output, count);
}

void GpuOps::bind_weights(id<MTLComputePipelineState> kernel, const QuantizedWeights& weights, Buf input) {
    [encoder setComputePipelineState:kernel];
    set_buffer(encoder, weights.packed, 0), set_buffer(encoder, weights.scales, 1), set_buffer(encoder, weights.biases, 2);
    set_buffer(encoder, input, 3);
}

// mlx QuantizedMatmul::eval_gpu with transpose, one batch, input row-contiguous [rows, in_size]: the same kernel, tile
// and grid choice (quantized.cpp v0.32.4: get_qmv_batch_limit, dispatch_qmv, qmv_wide, qmv, qmm_splitk, qmm).
void GpuOps::multiply_weights(const QuantizedWeights& weights, Buf input, Buf output, int rows) {
    const int out_size = weights.out_size, in_size = weights.in_size;
    // Batched decode, 7 <= rows <= 16: ours. Below 7 rows, small out_size (key / value_projection,
    // decay / update_rate_projection) and above 16 rows, MLX's kernels are as fast or faster (g12).
    if (is_decode && rows >= 7 && rows <= 16 && out_size >= 1024 && out_size % 32 == 0)
        return multiply_decode_batch(weights, input, output, rows);
    // get_qmv_batch_limit, gen 15/16, not 'd'
    const int many_rows_from = in_size <= 2048 && out_size <= 2048 ? 13 : in_size <= 4096 && out_size <= 4096 ? 15 : 13;
    if (rows >= many_rows_from) return multiply_many_rows(weights, input, output, rows);
    if (rows >= 2) {  // qmv_wide: rows_per_group rows of input per threadgroup, 8 output rows per threadgroup
        const int passes = (rows + 4) / 5, rows_per_group = (rows + passes - 1) / passes;
        assert(rows_per_group >= 2 && rows_per_group <= 5);
        bind_weights(multiply_few_rows[rows_per_group], weights, input);
        set_buffer(encoder, output, 4), set_value(encoder, in_size, 5), set_value(encoder, out_size, 6);
        set_value(encoder, rows, 7);
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake((rows + rows_per_group - 1) / rows_per_group, (out_size + 7) / 8, 1)
                               threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        return;
    }
    // qmv_fast, no partial rows (shapes checked at load)
    bind_weights(multiply_one_row, weights, input);
    set_buffer(encoder, output, 4), set_value(encoder, in_size, 5), set_value(encoder, out_size, 6);
    before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(rows, out_size / 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

// qmm_t_decode (quantized.metal), not MLX's kernel, so not bit-exact with mlx-lm (checked by tolerance: test_batch).
// One row tile of 8 / 16 rows; split along in_size as MLX's heuristic, capped at 8 splits.
void GpuOps::multiply_decode_batch(const QuantizedWeights& weights, Buf input, Buf output, int rows) {
    const int out_size = weights.out_size, in_size = weights.in_size;
    const int column_tiles = out_size / 32, sixteen_rows = rows > 8;
    int split_count = std::min({std::max(1, 512 / column_tiles), 8, in_size / 64});
    while (split_count > 1 && in_size % (split_count * 64)) split_count--;
    const int split_size = in_size / split_count, split_stride = rows * out_size;
    const bool split = split_count > 1;
    bind_weights(multiply_decode_rows[sixteen_rows][split], weights, input);
    set_buffer(encoder, split ? Buf{partial_sums} : output, 4), set_value(encoder, in_size, 5);
    set_value(encoder, out_size, 6), set_value(encoder, rows, 7);
    set_value(encoder, split_size, 8), set_value(encoder, split_stride, 9);
    if (split) {
        assert((size_t)split_count * split_stride * sizeof(float) <= partial_sums.length);
        // ws.partial_sums, as in multiply_many_rows: an independent neighbour's sum of splits may still be reading it
        // (write-after-read, the g9 race). Guarded by review only: no test fails without it, because the partials are
        // written only after the ~20 us loop over in_size, by which time the sum has finished.
        independent = false;
    }
    before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(column_tiles, 1, split_count)
                           threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    if (split) sum_split_parts(output, rows, out_size, split_count, split_stride);
}

// MLX's qmm_t, or qmm_splitk while there are few tiles to spread over the GPU
void GpuOps::multiply_many_rows(const QuantizedWeights& weights, Buf input, Buf output, int rows) {
    const int out_size = weights.out_size, in_size = weights.in_size;
    const int column_tiles = (out_size + 31) / 32, row_tiles = (rows + 31) / 32;
    int split_count = std::min(std::max(1, 512 / (column_tiles * row_tiles)), in_size / 64);
    while (split_count > 1 && in_size % (split_count * 64)) split_count--;
    const bool aligned = out_size % 32 == 0;
    if (split_count <= 1) {  // MLX's qmm_t; 64 x 64 tiles from large_tiles_from_rows rows (quantized.metal: same values)
        const int tile = rows >= large_tiles_from_rows ? 64 : 32;
        bind_weights(tile == 64 ? multiply_tiles_64[out_size % 64 == 0] : multiply_tiles_32[aligned], weights, input);
        set_buffer(encoder, output, 4), set_value(encoder, in_size, 5), set_value(encoder, out_size, 6);
        set_value(encoder, rows, 7);
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake((out_size + tile - 1) / tile, (rows + tile - 1) / tile, 1)
                               threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        return;
    }
    const int split_size = in_size / split_count, split_stride = rows * out_size;
    assert((size_t)split_count * split_stride * sizeof(float) <= partial_sums.length);
    bind_weights(multiply_split_inner[aligned], weights, input);
    set_buffer(encoder, {partial_sums}, 4), set_value(encoder, in_size, 5), set_value(encoder, out_size, 6);
    set_value(encoder, rows, 7), set_value(encoder, split_size, 8), set_value(encoder, split_stride, 9);
    independent = false;  // every multiply shares ws.partial_sums: an earlier sum of splits may still be reading it
    before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(column_tiles, row_tiles, split_count)
                           threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
    sum_split_parts(output, rows, out_size, split_count, split_stride);
}

// ws.partial_sums -> output (steel_gemm_splitk_accum), rows of out_size
void GpuOps::sum_split_parts(Buf output, int rows, int out_size, int split_count, int split_stride) {
    // get_block_dims(out_size, rows, 1): threadgroup sides 2^columns_log2 x 2^rows_log2, at most 1024 threads
    int columns_log2 = 0, rows_log2 = 0, total_log2 = 0;
    for (;;) {
        int before = total_log2;
        if (out_size >= 2 << columns_log2) columns_log2++, total_log2++;
        if (total_log2 == 10) break;
        if (rows >= 2 << rows_log2) rows_log2++, total_log2++;
        if (total_log2 == 10 || total_log2 == before) break;
    }
    [encoder setComputePipelineState:sum_splits];
    set_buffer(encoder, {partial_sums}, 0), set_buffer(encoder, output, 1), set_value(encoder, split_count, 2);
    set_value(encoder, split_stride, 3), set_value(encoder, out_size, 4);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(out_size, rows, 1)
                           threadsPerThreadgroup:MTLSizeMake(1 << columns_log2, 1 << rows_log2, 1)];
}

void GpuOps::apply_weights(int i, Workspace& ws) {
    begin(ws);
    const int rows = ws.request_count * ws.tokens_per_request;
    assert(rows > 0 && rows <= ws.max_rows && i >= 0 && i <= spec.n_layers());
    if (i == 0)
        look_up_embeddings({ws.token_ids}, {ws.residual}, rows);
    else
        finish_layer(layers[i - 1], ws, rows);
    if (i == spec.n_layers()) {
        normalize(final_norm, {ws.residual}, {ws.residual}, rows, spec.hidden);
        return;
    }
    start_layer(layers[i], ws, rows);
}

// after_attention = residual + attention_output(attention_result); residual = after_attention + feed-forward of
// feed_forward_norm(after_attention)
void GpuOps::finish_layer(const Layer& previous, Workspace& ws, int rows) {
    const int hidden = spec.hidden, gate_size = previous.feed_forward_gate.out_size;
    multiply_weights(previous.attention_output, {ws.attention_result}, {ws.projected}, rows);
    add({ws.residual}, {ws.projected}, {ws.after_attention}, (size_t)rows * hidden);
    normalize(previous.feed_forward_norm, {ws.after_attention}, {ws.normalized}, rows, hidden);
    if (merge_projections && rows == 1) {
        multiply_weights(previous.merged_gate_up, {ws.normalized}, {ws.gate_values}, rows);
        activate_gate({ws.gate_values}, {ws.gate_values, gate_size * ELEM}, {ws.gate_values}, gate_size);
    } else {
        multiply_weights(previous.feed_forward_gate, {ws.normalized}, {ws.gate_values}, rows);
        independent = true, multiply_weights(previous.feed_forward_up, {ws.normalized}, {ws.up_values}, rows);
        activate_gate({ws.gate_values}, {ws.up_values}, {ws.gate_values}, (size_t)rows * gate_size);
    }
    multiply_weights(previous.feed_forward_down, {ws.gate_values}, {ws.projected}, rows);
    add({ws.after_attention}, {ws.projected}, {ws.residual}, (size_t)rows * hidden);
}

// normalized = attention_norm(residual), then the projections attention reads
void GpuOps::start_layer(const Layer& layer, Workspace& ws, int rows) {
    normalize(layer.attention_norm, {ws.residual}, {ws.normalized}, rows, spec.hidden);
    const bool merged = merge_projections && rows == 1;
    if (layer.is_linear && merged) {
        multiply_weights(layer.merged_linear_projections, {ws.normalized}, {ws.query_key_value}, rows);
        return;
    }
    if (layer.is_linear) {
        multiply_weights(layer.query_key_value_projection, {ws.normalized}, {ws.query_key_value}, rows);
        independent = true, multiply_weights(layer.output_gate_projection, {ws.normalized}, {ws.output_gate}, rows);
        independent = true, multiply_weights(layer.update_rate_projection, {ws.normalized}, {ws.update_rate_input}, rows);
        independent = true, multiply_weights(layer.decay_projection, {ws.normalized}, {ws.decay_input}, rows);
        return;
    }
    const int head_dim = spec.head_dim;
    multiply_weights(merged ? layer.merged_query_key_value : layer.query_projection, {ws.normalized}, {ws.query_and_gate}, rows);
    [encoder setComputePipelineState:split_query_gate_kernel];
    set_buffer(encoder, {ws.query_and_gate}, 0), set_buffer(encoder, {ws.query}, 1), set_buffer(encoder, {ws.query_gate}, 2);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(head_dim, (size_t)rows * spec.query_heads, 1)
                           threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    normalize(layer.query_norm, {ws.query}, {ws.query}, rows * spec.query_heads, head_dim);
    if (merged) {  // key, value follow query | gate in ws.query_and_gate; value stays there (full_attention reads it)
        independent = true, normalize(layer.key_norm, {ws.query_and_gate, layer.query_projection.out_size * ELEM}, {ws.key},
                                      spec.key_value_heads, head_dim);
        return;
    }
    independent = true, multiply_weights(layer.key_projection, {ws.normalized}, {ws.key}, rows);
    normalize(layer.key_norm, {ws.key}, {ws.key}, rows * spec.key_value_heads, head_dim);
    independent = true, multiply_weights(layer.value_projection, {ws.normalized}, {ws.value}, rows);
}

// mlx Matmul::eval_gpu for the fallback's two multiplies (matmul.cpp): batch [key_value_heads, heads_per_key_value]
// never collapses (the repeat stride of keys / values is 0), that many batches rule out split-k, so: gemv_wide when
// the second is transposed and 2 <= rows <= 15, else steel_matmul_regular_axpby with the 's' tiles
// (GEMM_TPARAM_MACRO: bm 64, bn 64, bk 16, wm 2, wn 2). Run here one key_value head at a time (batch
// [1, heads_per_key_value]): every batch is computed alone, so the split is bit-exact.
void GpuOps::multiply_bf16(bool second_transposed, Buf first, Buf second, Buf output, int rows, int columns, int inner,
                           int first_row_stride, int second_row_stride, int output_row_stride,
                           const int64_t first_batch_strides[2], const int64_t second_batch_strides[2],
                           int64_t output_batch_stride) {
    const int batches = spec.query_heads / spec.key_value_heads, batch_shape[2] = {1, batches};
    int passes = (rows + 4) / 5;
    if (second_transposed && rows >= 2 && passes <= 3 && inner % 4 == 0) {  // gemv_wide_config (gen >= 15, aligned)
        assert(output_batch_stride == (int64_t)rows * columns);              // gemv_wide writes [batch, rows, columns] densely
        int rows_per_pass = (rows + passes - 1) / passes, inner_lanes = passes == 1 || columns <= 64 ? 32 : 16;
        assert(rows_per_pass >= 3 && rows_per_pass <= 5);  // rows 6..15: the fallback runs above 8 queries, or above 32 / heads_per_key_value
        [encoder setComputePipelineState:multiply_few_rows_bf16[rows_per_pass - 3][inner_lanes == 32]];
        set_buffer(encoder, second, 0), set_buffer(encoder, first, 1), set_buffer(encoder, output, 3);
        set_value(encoder, inner, 4), set_value(encoder, columns, 5), set_value(encoder, rows, 6);
        set_value(encoder, second_row_stride, 7), set_value(encoder, first_row_stride, 8);
        set_value(encoder, 2, 11);
        [encoder setBytes:batch_shape length:sizeof batch_shape atIndex:12];
        [encoder setBytes:first_batch_strides length:2 * sizeof(int64_t) atIndex:13];
        [encoder setBytes:second_batch_strides length:2 * sizeof(int64_t) atIndex:14];
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(columns >= 65536 ? 1 : passes, (columns + 3) / 4, batches)
                               threadsPerThreadgroup:MTLSizeMake(32, inner_lanes / 8, 1)];
        return;
    }
    const int tile_rows = 64, tile_columns = 64, tile_inner = 16;
    [encoder setComputePipelineState:multiply_tiles_bf16[!second_transposed][rows % tile_rows == 0]
                                                        [columns % tile_columns == 0][inner % tile_inner == 0]];
    const int column_tiles = (columns + tile_columns - 1) / tile_columns, row_tiles = (rows + tile_rows - 1) / tile_rows;
    mlx::steel::GEMMParams params{rows, columns, inner, first_row_stride, second_row_stride, output_row_stride,
                                  column_tiles, row_tiles, first_batch_strides[1], second_batch_strides[1],
                                  output_batch_stride, 0, inner / tile_inner, 2};
    const int64_t strides[4] = {first_batch_strides[0], first_batch_strides[1], second_batch_strides[0], second_batch_strides[1]};
    set_buffer(encoder, first, 0), set_buffer(encoder, second, 1), set_buffer(encoder, output, 3), set_value(encoder, params, 4);
    [encoder setBytes:batch_shape length:sizeof batch_shape atIndex:6];
    [encoder setBytes:strides length:sizeof strides atIndex:7];
    before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(column_tiles, row_tiles, batches)
                           threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
}

void GpuOps::apply_attention(int i, Workspace& ws) {
    begin(ws);
    assert(i >= 0 && i < spec.n_layers());
    if (layers[i].is_linear)
        linear_attention(i, layers[i], ws);
    else
        full_attention(layers[i], ws);
}

// gpu.py _linear, batched: one dispatch per op over all rows (MLX runs them one row at a time; every kernel here
// computes a row the same way regardless of the request count). Same lane contract as full_attention: one request is
// a prefill row with ws.tokens_per_request == n, more are decode (one token each, rows with n == 0 skipped).
void GpuOps::linear_attention(int i, const Layer& layer, Workspace& ws) {
    const int requests = ws.request_count, tokens = ws.tokens_per_request, rows = requests * tokens;
    const int conv_dim = spec.conv_dim(), value_heads = spec.linear_value_heads;
    const uint32_t* table_words = (const uint32_t*)ws.step_table.contents;
    // the table must carry every row's state span: the kernels index it with state_units = mem.state_units
    assert((int)table_words[0] == requests && table_words[1] == mem.state_units && rows <= ws.max_rows &&
           (requests == 1 || tokens == 1));
    for (int b = 0; b < requests; b++)
        assert(requests == 1 ? (int)table_words[2 + requests + b] == tokens : (int)table_words[2 + requests + b] <= tokens);
    const uint64_t conv_offset = mem.state_offset(i), delta_offset = conv_offset + mem.conv_bytes;
    // gated_delta_paged reads each lane's 32 floats as one run: runs must not cross a unit
    assert(mem.arena.unit_bytes % 128 == 0 && delta_offset % 128 == 0);
    // merge_projections: apply_weights wrote [query_key_value | output_gate | update_rate | decay] as one row in
    // ws.query_key_value
    const bool merged = merge_projections && rows == 1;
    const size_t value_size = (size_t)value_heads * spec.linear_value_dim;
    const Buf output_gate = merged ? Buf{ws.query_key_value, conv_dim * ELEM} : Buf{ws.output_gate};
    const Buf update_rate_input = merged ? Buf{ws.query_key_value, (conv_dim + value_size) * ELEM} : Buf{ws.update_rate_input};
    const Buf decay_input = merged ? Buf{ws.query_key_value, (conv_dim + value_size + value_heads) * ELEM} : Buf{ws.decay_input};

    conv_and_normalize(layer, ws, conv_offset);
    update_state(decay_input, update_rate_input, layer, ws, delta_offset);
    normalize_and_gate(layer.output_norm, {ws.attention_result}, output_gate, {ws.attention_result}, rows * value_heads,
                       spec.linear_value_dim);
}

// The conv over the last conv_kernel tokens (its state in the row's span), then query and key normalized and scaled
// into ws.normalized_query_key
void GpuOps::conv_and_normalize(const Layer& layer, Workspace& ws, uint64_t conv_offset) {
    const int requests = ws.request_count, tokens = ws.tokens_per_request, rows = requests * tokens;
    const int conv_dim = spec.conv_dim(), key_head_dim = spec.linear_key_dim, key_dim = spec.linear_key_heads * key_head_dim;
    const uint64_t unit_bytes = mem.arena.unit_bytes;
    const int state_units = (int)mem.state_units;
    id<MTLBuffer> arena = mem.arena_buffer, table = ws.step_table;
    const double inverse_scale = std::pow((double)key_head_dim, -0.5);
    if (tokens == 1) {  // decode: conv, norm and scales fused (conv_norm_qk), same eps and scales as below
        const float eps = (float)(1e-6 * inverse_scale * inverse_scale), query_scale = (float)(inverse_scale * inverse_scale);
        assert(conv_dim % key_head_dim == 0 && key_dim % key_head_dim == 0);
        [encoder setComputePipelineState:conv_and_normalize_kernel];
        set_buffer(encoder, {ws.query_key_value}, 0), set_buffer(encoder, layer.conv_weights, 1);
        set_buffer(encoder, {arena}, 2), set_buffer(encoder, {table}, 3);
        set_value(encoder, unit_bytes, 4), set_value(encoder, state_units, 5), set_value(encoder, conv_offset, 6);
        set_buffer(encoder, {ws.normalized_query_key}, 7);
        set_value(encoder, eps, 8), set_value(encoder, query_scale, 9), set_value(encoder, (float)inverse_scale, 10);
        set_value(encoder, key_dim, 11);
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(conv_dim / key_head_dim, requests, 1)
                               threadsPerThreadgroup:MTLSizeMake(key_head_dim, 1, 1)];
        return;
    }
    [encoder setComputePipelineState:conv_kernel];
    set_buffer(encoder, {ws.query_key_value}, 0), set_buffer(encoder, layer.conv_weights, 1);
    set_buffer(encoder, {arena}, 2), set_buffer(encoder, {table}, 3);
    set_value(encoder, unit_bytes, 4), set_value(encoder, state_units, 5), set_value(encoder, conv_offset, 6);
    set_value(encoder, tokens, 7);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(conv_dim, requests, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    // normalize_qk: rms_norm without weight (eps scaled to the mean), then the scales. Rows of 128 over the whole
    // conv output (value too, into scratch; only query / key are used), as MLX norms the query / key slices.
    normalize({one_bf16}, {ws.query_key_value}, {ws.normalized_query_key}, rows * (conv_dim / key_head_dim), key_head_dim,
              (float)(1e-6 * inverse_scale * inverse_scale), 0);
    [encoder setComputePipelineState:scale_query_key_kernel];
    set_buffer(encoder, {ws.normalized_query_key}, 0), set_value(encoder, (float)(inverse_scale * inverse_scale), 1);
    set_value(encoder, (float)inverse_scale, 2), set_value(encoder, conv_dim, 3);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(2 * key_dim, rows, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// The decay and update rate per head (gdn_gates), then the gated delta rule over the state in place (gated_delta_paged)
// -> ws.attention_result
void GpuOps::update_state(Buf decay_input, Buf update_rate_input, const Layer& layer, Workspace& ws, uint64_t delta_offset) {
    const int requests = ws.request_count, tokens = ws.tokens_per_request, rows = requests * tokens;
    const int conv_dim = spec.conv_dim(), value_heads = spec.linear_value_heads;
    const int key_dim = spec.linear_key_heads * spec.linear_key_dim;
    const uint64_t unit_bytes = mem.arena.unit_bytes;
    const int state_units = (int)mem.state_units;
    id<MTLBuffer> arena = mem.arena_buffer, table = ws.step_table;

    independent = true;
    [encoder setComputePipelineState:decay_and_update_rate_kernel];
    set_buffer(encoder, decay_input, 0), set_buffer(encoder, update_rate_input, 1), set_buffer(encoder, layer.decay_log_base, 2);
    set_buffer(encoder, layer.decay_bias, 3), set_buffer(encoder, {ws.decay}, 4), set_buffer(encoder, {ws.update_rate}, 5);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(value_heads, rows, 1) threadsPerThreadgroup:MTLSizeMake(value_heads, 1, 1)];

    // mlx-lm: grid (32, linear_value_dim / 8, requests * value_heads), threadgroup (32, 2, 1)
    [encoder setComputePipelineState:gated_delta_kernel];
    set_buffer(encoder, {ws.normalized_query_key}, 0), set_buffer(encoder, {ws.normalized_query_key, (size_t)key_dim * ELEM}, 1);
    set_buffer(encoder, {ws.query_key_value, 2 * (size_t)key_dim * ELEM}, 2);
    set_buffer(encoder, {ws.decay}, 3), set_buffer(encoder, {ws.update_rate}, 4), set_buffer(encoder, {arena}, 5);
    set_buffer(encoder, {ws.attention_result}, 6), set_buffer(encoder, {table}, 7);
    set_value(encoder, tokens, 8), set_value(encoder, conv_dim, 9), set_value(encoder, unit_bytes, 10);
    set_value(encoder, state_units, 11), set_value(encoder, delta_offset, 12);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(32, spec.linear_value_dim / 8, requests * value_heads)
                           threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
}

static KeyValueLayout key_value_layout(const Memory& mem, const ModelSpec& spec, int full_layer) {
    return KeyValueLayout{mem.arena.unit_bytes / ELEM, mem.page_tokens, spec.key_value_heads, full_layer};
}

// mlx-lm: head_dim ** -0.5
static float attention_scale(int head_dim) { return 1.0f / std::sqrt((float)head_dim); }

// gpu.py _full, batched: one dispatch per op over all rows. MLX v0.32.4 ops in Python's order: fast::rope on query
// and key, write_kv, fast::scaled_dot_product_attention (sdpa_vector / 2-pass while fused_attention_fits),
// * sigmoid(query_gate). Lane contract: Python's GPU path never pads, and MLX picks kernels by query length, so
// ws.tokens_per_request is the row's n_valid when there is one request (prefill; g6's ANE lane converts its padded
// output to n rows before apply_attention). More requests is decode: one token each, rows with n == 0 are skipped.
void GpuOps::full_attention(const Layer& layer, Workspace& ws) {
    const int requests = ws.request_count, tokens = ws.tokens_per_request, heads = spec.query_heads,
              head_dim = spec.head_dim;
    const uint32_t* table_words = (const uint32_t*)ws.step_table.contents;  // the CPU wrote it this step
    assert((int)table_words[0] == requests && requests * tokens <= ws.max_rows && (requests == 1 || tokens == 1));
    int shortest_context = INT_MAX, longest_context = 0;
    for (int b = 0; b < requests; b++) {
        int pos = table_words[2 + b], n = table_words[2 + requests + b], context = pos + n;
        assert((requests == 1 ? n == tokens : n <= tokens) && context <= ws.max_context &&
               table_words[3 + 2 * requests + b] - table_words[2 + 2 * requests + b] >= mem.units_for(context));
        if (n) shortest_context = std::min(shortest_context, context), longest_context = std::max(longest_context, context);
    }
    if (!longest_context) return;

    rotate_positions(ws.query, heads, ws);
    independent = true, rotate_positions(ws.key, spec.key_value_heads, ws);
    write_keys_values(layer, ws);
    if (fused_attention_fits(tokens, heads / spec.key_value_heads))
        attention_fused(layer, ws, shortest_context, longest_context);
    else
        attention_unfused(layer, ws, longest_context);

    [encoder setComputePipelineState:sigmoid_gate_kernel];
    set_buffer(encoder, {ws.attention_heads}, 0), set_buffer(encoder, {ws.query_gate}, 1);
    set_buffer(encoder, {ws.attention_result}, 2), set_value(encoder, tokens, 3);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(head_dim, heads, requests * tokens)
                           threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
}

// mlx fast::rope (rope.cpp), in place: the first ROTATED_DIMS of each head rotate, the rest stays (MLX copies it).
// Batched kernel "rope_bfloat16", per-row offsets = the table's pos words.
void GpuOps::rotate_positions(id<MTLBuffer> heads_buffer, int heads, const Workspace& ws) {
    const int head_dim = spec.head_dim;
    [encoder setComputePipelineState:rotate_positions_kernel];
    set_buffer(encoder, {heads_buffer}, 0), set_buffer(encoder, {heads_buffer}, 1);
    set_buffer(encoder, {ws.step_table, 8}, 2), set_value(encoder, 1.0f, 3);
    int64_t strides[3] = {head_dim, (int64_t)heads * head_dim, 1};
    [encoder setBytes:strides length:sizeof strides atIndex:4];
    [encoder setBytes:strides length:sizeof strides atIndex:5];
    set_value(encoder, (int64_t)1, 6), set_value(encoder, heads, 7), set_value(encoder, std::log2(ROTATION_BASE), 10);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(ROTATED_DIMS / 2, ws.tokens_per_request, ws.request_count * ((heads + 3) / 4))
                           threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void GpuOps::write_keys_values(const Layer& layer, Workspace& ws) {
    const int tokens = ws.tokens_per_request, head_dim = spec.head_dim;
    [encoder setComputePipelineState:write_keys_values_kernel];
    // merge_projections: value is still in apply_weights' [query | gate | key | value] row in ws.query_and_gate
    const Buf value = merge_projections && ws.request_count * tokens == 1
                          ? Buf{ws.query_and_gate, (size_t)(layer.query_projection.out_size + layer.key_projection.out_size) * ELEM}
                          : Buf{ws.value};
    set_buffer(encoder, {ws.key}, 0), set_buffer(encoder, value, 1), set_buffer(encoder, {mem.arena_buffer}, 2);
    set_buffer(encoder, {ws.step_table}, 3);
    set_value(encoder, key_value_layout(mem, spec, layer.full_index), 4), set_value(encoder, tokens, 5);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(head_dim, spec.key_value_heads, ws.request_count * tokens)
                           threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
}

// MLX: sdpa_vector, the two-pass kernel from TWO_PASS_FROM_KEYS keys. Rows pick their path by context length.
void GpuOps::attention_fused(const Layer& layer, const Workspace& ws, int shortest_context, int longest_context) {
    const int requests = ws.request_count, tokens = ws.tokens_per_request, heads = spec.query_heads;
    const int key_value_heads = spec.key_value_heads, heads_per_key_value = heads / key_value_heads;
    const KeyValueLayout layout = key_value_layout(mem, spec, layer.full_index);
    const float scale = attention_scale(spec.head_dim);
    id<MTLBuffer> arena = mem.arena_buffer, table = ws.step_table;
    if (longest_context >= TWO_PASS_FROM_KEYS) {
        const int blocks = two_pass_blocks(longest_context, heads_per_key_value * tokens);
        assert((size_t)requests * heads * tokens * blocks <= ws.two_pass_sums.length / sizeof(float));
        const bool many_threads =
            32 * heads_per_key_value * tokens > (int)attention_long_context_blocks[0].maxTotalThreadsPerThreadgroup;
        [encoder setComputePipelineState:attention_long_context_blocks[many_threads]];
        set_buffer(encoder, {ws.query}, 0), set_buffer(encoder, {arena}, 1), set_buffer(encoder, {ws.two_pass_partials}, 3);
        set_buffer(encoder, {ws.two_pass_sums}, 4), set_buffer(encoder, {ws.two_pass_maxes}, 5);
        set_buffer(encoder, {table}, 6), set_value(encoder, layout, 7), set_value(encoder, TWO_PASS_FROM_KEYS, 8);
        set_value(encoder, INT_MAX, 9), set_value(encoder, blocks, 11), set_value(encoder, scale, 12);
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(key_value_heads, requests, blocks)
                               threadsPerThreadgroup:MTLSizeMake(32, heads_per_key_value, tokens)];
        // over every row; the one-pass kernel below then overwrites the rows that are not on this path
        [encoder setComputePipelineState:attention_long_context_combine];
        set_buffer(encoder, {ws.two_pass_partials}, 0), set_buffer(encoder, {ws.two_pass_sums}, 1);
        set_buffer(encoder, {ws.two_pass_maxes}, 2), set_buffer(encoder, {ws.attention_heads}, 3), set_value(encoder, blocks, 4);
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(requests * heads, tokens, 1)
                               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
    if (shortest_context < TWO_PASS_FROM_KEYS) {
        [encoder setComputePipelineState:attention_short_context];
        set_buffer(encoder, {ws.query}, 0), set_buffer(encoder, {arena}, 1), set_buffer(encoder, {ws.attention_heads}, 3);
        set_value(encoder, heads_per_key_value, 4), set_buffer(encoder, {table}, 5);
        set_value(encoder, layout, 6), set_value(encoder, 0, 7), set_value(encoder, TWO_PASS_FROM_KEYS, 8);
        set_value(encoder, scale, 10);
        before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(requests * heads, tokens, 1)
                               threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
    }
}

// MLX's unfused fallback (fast.cpp; use_fallback: head_dim 256, !fused_attention_fits), row 0 only
void GpuOps::attention_unfused(const Layer& layer, const Workspace& ws, int longest_context) {
    const int heads = spec.query_heads, key_value_heads = spec.key_value_heads, head_dim = spec.head_dim;
    const int heads_per_key_value = heads / key_value_heads;
    assert(ws.request_count == 1);
    const int queries = ws.tokens_per_request, keys = longest_context;
    // ponytail: MLX v0.32.4 runs fused steel attention for queries >= 2048 == keys (no prefix); not ported (chunks are
    // <= 512). Port it if a lane sends such a chunk.
    assert(!(queries >= 2048 && keys == queries));
    assert((size_t)queries * heads_per_key_value * keys * ELEM <= ws.attention_scores.length &&
           (size_t)key_value_heads * keys * head_dim * ELEM <= ws.gathered_keys.length);
    id<MTLBuffer> table = ws.step_table;
    [encoder setComputePipelineState:gather_keys_values_kernel];
    set_buffer(encoder, {mem.arena_buffer}, 0), set_buffer(encoder, {ws.gathered_keys}, 1);
    set_buffer(encoder, {ws.gathered_values}, 2), set_buffer(encoder, {table}, 3);
    set_value(encoder, key_value_layout(mem, spec, layer.full_index), 4);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(head_dim, keys, key_value_heads)
                           threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
    [encoder setComputePipelineState:scale_kernel];
    set_buffer(encoder, {ws.query}, 0), set_value(encoder, attention_scale(head_dim), 1);
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake((size_t)queries * heads * head_dim, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    // per key_value head g: scores [heads_per_key_value, queries, keys] = query keyᵀ, so scores holds one head group
    // (MLX's [heads, queries, keys] split by g; bit-exact, every row is computed alone). query is our
    // [queries, heads, head_dim]; repeat r -> head g * heads_per_key_value + r
    const int64_t query_batch_strides[2] = {0, head_dim}, key_value_batch_strides[2] = {0, 0},
                  score_batch_strides[2] = {0, (int64_t)queries * keys};
    const bool looped = keys > 4096;  // softmax(precise=True), in place (softmax.cpp)
    id<MTLComputePipelineState> softmax = softmax_kernel[looped];
    size_t threads_per_group = looped ? softmax.maxTotalThreadsPerThreadgroup : 32 * (((keys + 3) / 4 + 31) / 32);
    assert(threads_per_group <= softmax.maxTotalThreadsPerThreadgroup);
    for (int g = 0; g < key_value_heads; g++) {
        const size_t key_value_offset = (size_t)g * keys * head_dim * ELEM;
        multiply_bf16(true, {ws.query, (size_t)g * heads_per_key_value * head_dim * ELEM}, {ws.gathered_keys, key_value_offset},
                      {ws.attention_scores}, queries, keys, head_dim, heads * head_dim, head_dim, keys, query_batch_strides,
                      key_value_batch_strides, (int64_t)queries * keys);
        [encoder setComputePipelineState:causal_mask_kernel];
        set_buffer(encoder, {ws.attention_scores}, 0), set_buffer(encoder, {table}, 1);
        before_dispatch(), [encoder dispatchThreads:MTLSizeMake(keys, queries, heads_per_key_value)
                               threadsPerThreadgroup:MTLSizeMake(std::min(keys, 256), 1, 1)];
        [encoder setComputePipelineState:softmax];
        set_buffer(encoder, {ws.attention_scores}, 0), set_buffer(encoder, {ws.attention_scores}, 1), set_value(encoder, keys, 2);
        before_dispatch(), [encoder dispatchThreads:MTLSizeMake((size_t)heads_per_key_value * queries * threads_per_group, 1, 1)
                               threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        // attention_heads [heads, queries, head_dim] = scores value
        multiply_bf16(false, {ws.attention_scores}, {ws.gathered_values, key_value_offset},
                      {ws.attention_heads, (size_t)g * heads_per_key_value * queries * head_dim * ELEM}, queries, head_dim,
                      keys, keys, head_dim, head_dim, score_batch_strides, key_value_batch_strides, (int64_t)queries * head_dim);
    }
}

void GpuOps::pick_next_tokens(Workspace& ws, const std::vector<SampleParams>& params) {
    begin(ws);
    const int requests = ws.request_count, vocab_size = lm.out_size;
    assert(ws.tokens_per_request == 1 && requests <= ws.max_rows &&
           ws.next_token_scores.length >= (size_t)requests * vocab_size * ELEM);
    assert(params.empty() || (int)params.size() == requests);
    multiply_weights(lm, {ws.residual}, {ws.next_token_scores}, requests);
    sample_tokens({ws.next_token_scores}, {ws.next_token_ids}, vocab_size,
                  params.empty() ? std::vector<SampleParams>(requests) : params);
}

std::vector<int> GpuOps::next_tokens(const Workspace& ws) const {
    const int32_t* tokens = (const int32_t*)ws.next_token_ids.contents;
    return std::vector<int>(tokens, tokens + ws.request_count);
}

void GpuOps::feed_previous_tokens(Workspace& ws, id<MTLBuffer> previous_token_ids, const std::vector<int>& source_rows) {
    begin(ws);
    assert((int)source_rows.size() == ws.request_count && ws.tokens_per_request == 1);
    [encoder setComputePipelineState:feed_tokens_kernel];
    set_buffer(encoder, {previous_token_ids}, 0), set_buffer(encoder, {ws.token_ids}, 1);
    [encoder setBytes:source_rows.data() length:source_rows.size() * sizeof(int) atIndex:2];
    before_dispatch(), [encoder dispatchThreads:MTLSizeMake(source_rows.size(), 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(source_rows.size(), 1, 1)];
}

void GpuOps::sample_tokens(Buf scores, Buf output, int vocab_size, const std::vector<SampleParams>& params) {
    struct SampleRow {  // sample.metal SampleRow
        float temperature, top_p;
        uint64_t seed;
    };
    std::vector<SampleRow> rows;
    for (const SampleParams& row : params) {
        assert(row.temperature >= 0 && (row.temperature == 0 || row.rng));
        rows.push_back({row.temperature, row.top_p, row.temperature > 0 ? (*row.rng)() : 0});
    }
    [encoder setComputePipelineState:sample_kernel];
    set_buffer(encoder, scores, 0), set_buffer(encoder, output, 1), set_value(encoder, vocab_size, 2);
    [encoder setBytes:rows.data() length:rows.size() * sizeof(SampleRow) atIndex:3];
    before_dispatch(), [encoder dispatchThreadgroups:MTLSizeMake(rows.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

}  // namespace mllm
