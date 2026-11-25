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
constexpr int ROPE_DIMS = 64;
constexpr float ROPE_THETA = 1e7f;
// MLX's 'medium' (applegpu_g*s) SDPA rules: 2-pass from 1024 keys
constexpr int SDPA_2PASS_KEYS = 1024;

// sdpa_vector_2pass blocks on an 's' GPU (scaled_dot_product_attention.cpp sdpa_vector_2pass)
static int sdpa_blocks(int N, int n_simds) {
    if (N > 1024 && n_simds > 4) return N <= 8192 ? 128 : N <= 32768 ? 256 : N <= 65536 ? 512 : 1024;
    return 64;
}

// MLX's has_fused_kernel: sdpa_vector only while q length <= 8 and q length x gqa <= 32 (27B, gqa 6: q length <= 5),
// else the unfused fallback
static bool sdpa_vector_ok(int S, int gqa) { return S <= 8 && S * gqa <= 32; }

// attention.metal KVLayout
struct KVLayout {
    uint64_t unit;
    int T, key_value_heads, g;
};

static std::runtime_error error(const std::string& what, NSError* err) {
    return std::runtime_error(what + (err ? std::string(": ") + err.localizedDescription.UTF8String : ""));
}

id<MTLComputePipelineState> GpuOps::pipeline(NSString* name, MTLFunctionConstantValues* constants) {
    NSError* err = nil;
    id<MTLFunction> f = constants ? [library newFunctionWithName:name constantValues:constants error:&err]
                                  : [library newFunctionWithName:name];
    if (!f) throw error(std::string("no kernel ") + name.UTF8String + " in " + METALLIB, err);
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:f error:&err];
    if (!ps) throw error(std::string("pipeline ") + name.UTF8String, err);
    return ps;
}

// bool function constants (index, value)
id<MTLComputePipelineState> GpuOps::pipeline(NSString* name, std::initializer_list<std::pair<int, bool>> constants) {
    MTLFunctionConstantValues* fc = [MTLFunctionConstantValues new];
    for (auto [i, v] : constants) [fc setConstantValue:&v type:MTLDataTypeBool atIndex:i];
    return pipeline(name, fc);
}

GpuOps::GpuOps(Memory& mem, const Weights& weights) : spec(mem.spec), mem(mem), device(mem.device) {
    // MLX picks kernels and tile sizes by GPU generation. Only the M4 family's branch (gen 15/16, not a 'd' Max/Ultra
    // die) is copied below. ponytail: port the other branches of quantized.cpp when another GPU shows up.
    NSString* arch = device.architecture.name;
    if (!([arch hasPrefix:@"applegpu_g15"] || [arch hasPrefix:@"applegpu_g16"]) || [arch hasSuffix:@"d"])
        throw error(std::string("GpuOps: dispatch rules not ported for ") + arch.UTF8String, nil);

    NSError* err = nil;
    unsigned long n = 0;  // build/mllm carries the metallib in a section (Makefile); other binaries load the file
    const uint8_t* embedded = getsectiondata((const mach_header_64*)_dyld_get_image_header(0), "__TEXT", "__metallib", &n);
    library = embedded ? [device newLibraryWithData:dispatch_data_create(embedded, n, nullptr, ^{}) error:&err]
                       : [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&err];
    if (!library) throw error(std::string("cannot load ") + METALLIB, err);

    qmv_fast = pipeline(@"affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4");
    for (int nv = 2; nv <= 5; nv++)
        qmv_wide[nv] = pipeline([NSString stringWithFormat:@"affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_%d_kl_8_batch_0", nv]);
    for (int al = 0; al < 2; al++) {
        NSString* a = al ? @"true" : @"false";
        qmm_t_splitk[al] = pipeline([NSString stringWithFormat:@"affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_%@", a]);
        qmm_t[al] = pipeline([NSString stringWithFormat:@"affine_qmm_t_bfloat16_t_gs_64_b_4_alN_%@_batch_0", a]);
        qmm_t_64[al] = pipeline([NSString stringWithFormat:@"qmm_t_64_alN_%@", a]);
    }
    splitk_accum = pipeline(@"steel_gemm_splitk_accum_bfloat16_float32");
    for (int bm = 0; bm < 2; bm++)
        for (int sk = 0; sk < 2; sk++)
            qmm_t_decode[bm][sk] =
                pipeline([NSString stringWithFormat:@"qmm_t_decode_%d_%@", bm ? 16 : 8, sk ? @"float" : @"bf16"]);
    embed_ps = pipeline(@"embed");
    rms_ps = pipeline(@"rmsbfloat16");
    rms_looped_ps = pipeline(@"rms_loopedbfloat16");
    add_ps = pipeline(@"add");
    swiglu_ps = pipeline(@"swiglu");
    precise_swiglu_ps = pipeline(@"precise_swiglu");
    split_q_gate_ps = pipeline(@"split_q_gate");

    // full attention: kernel names fix head_dim 256 (bf16)
    if (!spec.full_layers().empty() && spec.head_dim != 256)
        throw std::runtime_error("GpuOps: attention kernels instantiated for head_dim 256 only");
    rope_ps = pipeline(@"rope_bfloat16", {{1, true}, {2, false}, {3, false}});  // forward, traditional, hs_transpose
    write_kv_ps = pipeline(@"write_kv_bfloat16_256");
    sdpa_ps = pipeline(@"sdpa_vector_paged_bfloat16_256");
    sdpa_2pass_1[0] = pipeline(@"sdpa_vector_2pass_1_paged_bfloat16_256");
    sdpa_2pass_1[1] = pipeline(@"sdpa_vector_2pass_1_paged_bfloat16_256_tk1");
    sdpa_2pass_2 = pipeline(@"sdpa_vector_2pass_2_bfloat16_256");
    sigmoid_gate_ps = pipeline(@"sigmoid_gate");
    // linear attention: kernel names fix the conv kernel and head shapes
    if (!spec.linear_layers().empty() && (spec.conv_kernel != 4 || spec.linear_key_heads != 16 ||
                                          (spec.linear_value_heads != 16 && spec.linear_value_heads != 48) || spec.linear_key_dim != 128 ||
                                          spec.linear_value_dim != 128))
        throw std::runtime_error("GpuOps: linear attention kernels instantiated for Qwen3.5-0.8B's and 27B's shapes only");
    conv1d_silu_ps = pipeline(@"conv1d_silu_4");
    conv_norm_qk_ps = pipeline(@"conv_norm_qk_4");
    scale_qk_ps = pipeline(@"scale_qk");
    gdn_gates_ps = pipeline(@"gdn_gates_float");
    gdn_gates_bf16_ps = pipeline(@"gdn_gates_bfloat16");
    gated_delta_ps = pipeline([NSString stringWithFormat:@"gated_delta_paged_bfloat16_128_128_16_%d", spec.linear_value_heads]);
    sample_ps = pipeline(@"sample_bf16");
    feed_ps = pipeline(@"feed_tokens");
    gather_kv_ps = pipeline(@"gather_kv_bfloat16_256");
    causal_mask_ps = pipeline(@"causal_mask");
    scale_ps = pipeline(@"scale_bf16");
    softmax_ps[0] = pipeline(@"block_softmax_precise_bfloat16");
    softmax_ps[1] = pipeline(@"looped_softmax_precise_bfloat16");
    for (int nv = 0; nv < 3; nv++)
        for (int kl = 0; kl < 2; kl++)  // has_batch, do_axpby
            gemv_wide_ps[nv][kl] = pipeline([NSString stringWithFormat:@"gemv_wide_bfloat16_nv%d_kl%d", 3 + nv, 16 << kl],
                                            {{0, true}, {1, false}});
    for (int nn = 0; nn < 2; nn++)
        for (int am = 0; am < 2; am++)
            for (int an = 0; an < 2; an++)
                for (int ak = 0; ak < 2; ak++)  // has_batch, use_out_source, do_axpby, align_M, align_N, align_K
                    steel_ps[nn][am][an][ak] = pipeline(
                        nn ? @"steel_gemm_fused_nn_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2"
                           : @"steel_gemm_fused_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2",
                        {{10, true}, {100, false}, {110, false}, {200, am}, {201, an}, {202, ak}});

    auto buf = [&](const std::string& name) {
        auto v = weights[name];
        return Buf{v.buffer, v.offset};
    };
    auto qlinear = [&](const std::string& p) {
        auto w = weights[p + ".weight"];
        auto s = weights[p + ".scales"];
        QuantizedWeights l{buf(p + ".weight"), buf(p + ".scales"), buf(p + ".biases"), (int)w.t.shape[0], (int)w.t.shape[1] * 8};
        if (w.t.dtype != "U32" || s.t.dtype != "BF16" || s.t.shape[1] * 64 != l.in_size)
            throw std::runtime_error(p + ": not 4-bit affine, group 64, bf16");
        // multiply_weights() ports only qmv_fast for small M: no qmv_quad (K 64 / 128), qmv (K % 512) or partial rows (N % 8)
        if (l.in_size % 512 || l.out_size % 8) throw std::runtime_error(p + ": K % 512 or N % 8 not 0 (kernel not ported)");
        return l;
    };
    const std::string P = "language_model.model.";
    embed_tokens = qlinear(P + "embed_tokens");
    lm = weights.st.tensors.count("language_model.lm_head.weight") ? qlinear("language_model.lm_head") : embed_tokens;
    norm = buf(P + "norm.weight");
    for (int i = 0, n_full = 0; i < spec.n_layers(); i++) {
        std::string p = P + "layers." + std::to_string(i) + ".";
        Layer l{};
        l.is_linear = spec.kinds[i] == "linear";
        if (!l.is_linear) l.full_index = n_full++;
        l.attention_norm = buf(p + "input_layernorm.weight");
        l.feed_forward_norm = buf(p + "post_attention_layernorm.weight");
        l.feed_forward_gate = qlinear(p + "mlp.gate_proj");
        l.feed_forward_up = qlinear(p + "mlp.up_proj");
        l.feed_forward_down = qlinear(p + "mlp.down_proj");
        if (l.is_linear) {
            l.attention_output = qlinear(p + "linear_attn.out_proj");
            l.query_key_value_projection = qlinear(p + "linear_attn.in_proj_qkv");
            l.output_gate_projection = qlinear(p + "linear_attn.in_proj_z");
            l.update_rate_projection = qlinear(p + "linear_attn.in_proj_b");
            l.decay_projection = qlinear(p + "linear_attn.in_proj_a");
            l.conv_weights = buf(p + "linear_attn.conv1d.weight");
            l.decay_log_base = buf(p + "linear_attn.A_log");
            l.decay_bias = buf(p + "linear_attn.dt_bias");
            l.output_norm = buf(p + "linear_attn.norm.weight");
            const std::string a_dt = weights[p + "linear_attn.A_log"].t.dtype;
            if ((a_dt != "F32" && a_dt != "BF16") || weights[p + "linear_attn.conv1d.weight"].t.dtype != "BF16")
                throw std::runtime_error(p + "linear_attn: A_log F32 / BF16 and conv1d BF16 expected");
            if (a_dt == "BF16") gdn_gates_ps = gdn_gates_bf16_ps;
        } else {
            l.attention_output = qlinear(p + "self_attn.o_proj");
            l.query_projection = qlinear(p + "self_attn.q_proj");
            l.key_projection = qlinear(p + "self_attn.k_proj");
            l.value_projection = qlinear(p + "self_attn.v_proj");
            l.query_norm = buf(p + "self_attn.q_norm.weight");
            l.key_norm = buf(p + "self_attn.k_norm.weight");
        }
        layers.push_back(l);
    }
    // the merged weights: the weights cache keeps each chain's parts back to back (memory.h MERGED), so the merged
    // matmul is a view starting at the first part
    auto merged = [&](std::initializer_list<QuantizedWeights*> parts) {
        const QuantizedWeights& f = **parts.begin();
        QuantizedWeights m{.packed = f.packed, .scales = f.scales, .biases = f.biases, .in_size = f.in_size};
        for (QuantizedWeights* p : parts) {
            const size_t wrow = m.in_size / 2, srow = m.in_size / 64 * ELEM;
            for (auto [whole, part, row] : {std::tuple{&m.packed, &p->packed, wrow}, {&m.scales, &p->scales, srow}, {&m.biases, &p->biases, srow}})
                if (p->in_size != m.in_size || part->buffer != whole->buffer || part->offset != whole->offset + m.out_size * row)
                    throw std::runtime_error("GpuOps: merged projection parts not adjacent in the weights cache");
            m.out_size += p->out_size;
        }
        return m;
    };
    for (Layer& l : layers) {
        if (l.is_linear)
            l.merged_linear_projections = merged({&l.query_key_value_projection, &l.output_gate_projection, &l.update_rate_projection, &l.decay_projection});
        else
            l.merged_query_key_value = merged({&l.query_projection, &l.key_projection, &l.value_projection});
        l.merged_gate_up = merged({&l.feed_forward_gate, &l.feed_forward_up});
    }

    one = mem.buffer(ELEM);
    *(uint16_t*)one.contents = 0x3F80;  // bf16 1.0
}

Workspace GpuOps::workspace(int chunk, int max_batch, int max_ctx) {
    Workspace ws = mem.workspace(chunk, max_batch);
    ws.max_rows = std::max(chunk, max_batch), ws.max_context = max_ctx;
    size_t row = (size_t)ws.max_rows * ELEM;
    ws.after_attention = mem.buffer(row * spec.hidden);
    ws.normalized = mem.buffer(row * spec.hidden);
    ws.projected = mem.buffer(row * spec.hidden);
    // merge_projections: the M = 1 merged rows land in ws.gate_values, ws.query_and_gate and ws.query_key_value (rows may be 1)
    ws.gate_values = mem.buffer(std::max(row * layers[0].feed_forward_gate.out_size, layers[0].merged_gate_up.out_size * ELEM));
    ws.up_values = mem.buffer(row * layers[0].feed_forward_gate.out_size);
    size_t qkv_N = 0, in_N = 0;
    for (const Layer& l : layers) qkv_N = std::max<size_t>(qkv_N, l.merged_query_key_value.out_size), in_N = std::max<size_t>(in_N, l.merged_linear_projections.out_size);
    ws.query_and_gate = mem.buffer(std::max(row * 2 * spec.query_heads * spec.head_dim, qkv_N * ELEM));
    if (ws.query_key_value.length < in_N * ELEM) ws.query_key_value = mem.buffer(in_N * ELEM);
    ws.partial_sums = mem.buffer(512 * 32 * 32 * sizeof(float));  // split_k * M * N <= 512 tiles of 32 x 32 (see linear)
    const size_t H = spec.query_heads, D = spec.head_dim, gqa = H / spec.key_value_heads;
    ws.attention_heads = mem.buffer(row * H * D);
    // 2-pass partials at max_ctx keys: decode (max_batch rows of 1 query; gqa > 4 takes up to 1024 blocks) or one
    // prefill row of up to qv queries (the vector path's longest)
    const int qv = std::min({chunk, 8, 32 / (int)gqa});
    size_t n_part = std::max(max_batch * H * sdpa_blocks(max_ctx, gqa), qv * H * sdpa_blocks(max_ctx, gqa * qv));
    ws.two_pass_partials = mem.buffer(n_part * D * ELEM);
    ws.two_pass_sums = mem.buffer(n_part * sizeof(float));
    ws.two_pass_maxes = mem.buffer(n_part * sizeof(float));
    // the unfused fallback only: decode workspaces (chunk on the vector path) never gather
    const size_t kv_rows = sdpa_vector_ok(chunk, (int)gqa) ? 1 : ws.max_context;
    ws.gathered_keys = mem.buffer(spec.key_value_heads * kv_rows * D * ELEM);
    ws.gathered_values = mem.buffer(spec.key_value_heads * kv_rows * D * ELEM);
    ws.attention_scores = mem.buffer((size_t)std::max(chunk, 1) * ELEM * H * kv_rows);
    ws.normalized_query_key = mem.buffer(row * spec.conv_dim());
    ws.decay = mem.buffer((size_t)ws.max_rows * spec.linear_value_heads * sizeof(float));
    ws.update_rate = mem.buffer(row * spec.linear_value_heads);
    ws.next_token_ids = mem.buffer((size_t)ws.max_rows * sizeof(int32_t));
    return ws;
}

static void set(id<MTLComputeCommandEncoder> enc, Buf b, int i) { [enc setBuffer:b.buffer offset:b.offset atIndex:i]; }
template <class T>
static void set_bytes(id<MTLComputeCommandEncoder> enc, T v, int i) { [enc setBytes:&v length:sizeof(T) atIndex:i]; }

void GpuOps::look_up_embeddings(Buf tokens, Buf out, int rows) {
    const QuantizedWeights& e = embed_tokens;
    [enc setComputePipelineState:embed_ps];
    set(enc, e.packed, 0), set(enc, e.scales, 1), set(enc, e.biases, 2), set(enc, tokens, 3), set(enc, out, 4);
    set_bytes(enc, e.in_size, 5);
    pre(), [enc dispatchThreads:MTLSizeMake(e.in_size / 8, rows, 1) threadsPerThreadgroup:MTLSizeMake(std::min(e.in_size / 8, 256), 1, 1)];
}

// mlx RMSNorm::eval_gpu: rms_single_row up to RMS_LOOPED_LIMIT (4096) wide, rms_looped above (27B hidden 5120)
void GpuOps::normalize(Buf weight, Buf x, Buf out, int rows, int dim, float eps, uint32_t w_stride) {
    const bool looped = dim > 4096;
    id<MTLComputePipelineState> ps = looped ? rms_looped_ps : rms_ps;
    uint32_t tg = looped ? (uint32_t)ps.maxTotalThreadsPerThreadgroup : 32 * (((dim + 3) / 4 + 31) / 32), axis = dim;
    assert(tg <= ps.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:ps];
    set(enc, x, 0), set(enc, weight, 1), set(enc, out, 2);
    set_bytes(enc, eps, 3), set_bytes(enc, axis, 4), set_bytes(enc, w_stride, 5);
    pre(), [enc dispatchThreads:MTLSizeMake((size_t)rows * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Qwen3NextRMSNormGated: precise_swiglu(x, z, normalize(x, weight))
void GpuOps::normalize_and_gate(Buf weight, Buf x, Buf z, Buf out, int rows, int dim) {
    normalize(weight, x, out, rows, dim);
    elementwise(precise_swiglu_ps, z, out, out, (size_t)rows * dim);
}

void GpuOps::elementwise(id<MTLComputePipelineState> ps, Buf a, Buf b, Buf out, size_t n) {
    [enc setComputePipelineState:ps];
    set(enc, a, 0), set(enc, b, 1), set(enc, out, 2);
    pre(), [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(n, 256), 1, 1)];
}

void GpuOps::add(Buf x, Buf y, Buf out, size_t n) { elementwise(add_ps, x, y, out, n); }
void GpuOps::activate_gate(Buf gate, Buf x, Buf out, size_t n) { elementwise(swiglu_ps, gate, x, out, n); }

// mlx QuantizedMatmul::eval_gpu with transpose, one batch, x row-contiguous [M, K]: the same kernel, tile and grid
// choice (quantized.cpp v0.32.4: get_qmv_batch_limit, dispatch_qmv, qmv_wide, qmv, qmm_splitk, qmm).
void GpuOps::multiply_weights(const QuantizedWeights& l, Buf x, Buf y, int M) {
    const int N = l.out_size, K = l.in_size;
    auto bind = [&](id<MTLComputePipelineState> ps) {
        [enc setComputePipelineState:ps];
        set(enc, l.packed, 0), set(enc, l.scales, 1), set(enc, l.biases, 2), set(enc, x, 3);
    };
    // split-k scratch -> y (steel_gemm_splitk_accum), M rows of N
    auto accum = [&](int split_k, int stride) {
        // get_block_dims(N, M, 1): powers of two, at most 1024 threads
        int p0 = 0, p1 = 0, sum = 0;
        for (;;) {
            int before = sum;
            if (N >= 2 << p0) p0++, sum++;
            if (sum == 10) break;
            if (M >= 2 << p1) p1++, sum++;
            if (sum == 10 || sum == before) break;
        }
        [enc setComputePipelineState:splitk_accum];
        set(enc, {splitk}, 0), set(enc, y, 1), set_bytes(enc, split_k, 2), set_bytes(enc, stride, 3), set_bytes(enc, N, 4);
        pre(), [enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(1 << p0, 1 << p1, 1)];
    };
    // Batched decode, 7 <= M <= 16: qmm_t_decode (quantized.metal), not MLX's kernel, so not bit-exact with mlx-lm
    // (checked by tolerance: test_batch). One M tile of 8 / 16 rows; split-k as MLX's heuristic, capped at 8.
    // Below M = 7, small N (k/value_projection, decay_projection/b) and above M = 16, MLX's kernels are as fast or faster (g12).
    if (decode && M >= 7 && M <= 16 && N >= 1024 && N % 32 == 0) {
        const int n_tiles = N / 32, bm = M > 8;
        int split_k = std::min({std::max(1, 512 / n_tiles), 8, K / 64});
        while (split_k > 1 && K % (split_k * 64)) split_k--;
        const int k_part = K / split_k, stride = M * N;
        bind(qmm_t_decode[bm][split_k > 1]);
        set(enc, split_k > 1 ? Buf{splitk} : y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
        set_bytes(enc, k_part, 8), set_bytes(enc, stride, 9);
        if (split_k > 1) {
            assert((size_t)split_k * stride * sizeof(float) <= splitk.length);
            // ws.partial_sums, as below: an indep neighbour's accumulate may still be reading it (write-after-read, the g9
            // race). Guarded by review only: no test fails without it, because the partials are written only after
            // the ~20 us K loop, by which time the accumulate has finished.
            indep = false;
        }
        pre(), [enc dispatchThreadgroups:MTLSizeMake(n_tiles, 1, split_k) threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
        if (split_k > 1) accum(split_k, stride);
        return;
    }
    int limit = K <= 2048 && N <= 2048 ? 13 : K <= 4096 && N <= 4096 ? 15 : 13;  // gen 15/16, not 'd'
    if (M >= limit) {
        int n_tiles = (N + 31) / 32, m_tiles = (M + 31) / 32;
        int split_k = std::min(std::max(1, 512 / (n_tiles * m_tiles)), K / 64);
        while (split_k > 1 && K % (split_k * 64)) split_k--;
        bool aligned = N % 32 == 0;
        if (split_k <= 1) {  // MLX's qmm_t; 64 x 64 tiles from tile64_rows rows (quantized.metal: same values)
            const int t = M >= tile64_rows ? 64 : 32;
            bind(t == 64 ? qmm_t_64[N % 64 == 0] : qmm_t[aligned]);
            set(enc, y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
            pre(), [enc dispatchThreadgroups:MTLSizeMake((N + t - 1) / t, (M + t - 1) / t, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
            return;
        }
        int k_part = K / split_k, stride = M * N;
        assert((size_t)split_k * stride * sizeof(float) <= splitk.length);
        bind(qmm_t_splitk[aligned]);
        set(enc, {splitk}, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
        set_bytes(enc, k_part, 8), set_bytes(enc, stride, 9);
        indep = false;  // every linear shares ws.partial_sums: an earlier accumulate may still be reading it
        pre(), [enc dispatchThreadgroups:MTLSizeMake(n_tiles, m_tiles, split_k) threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
        accum(split_k, stride);
    } else if (M >= 2) {  // qmv_wide: vecs_per_tg rows of x per threadgroup, 8 output rows per threadgroup
        int n_tiles = (M + 4) / 5, nv = (M + n_tiles - 1) / n_tiles;
        assert(nv >= 2 && nv <= 5);
        bind(qmv_wide[nv]);
        set(enc, y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
        pre(), [enc dispatchThreadgroups:MTLSizeMake((M + nv - 1) / nv, (N + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {  // qmv_fast, no partial rows (shapes checked at load)
        bind(qmv_fast);
        set(enc, y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6);
        pre(), [enc dispatchThreadgroups:MTLSizeMake(M, N / 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    }
}

void GpuOps::apply_weights(int i, Workspace& ws) {
    begin(ws);
    const int M = ws.request_count * ws.tokens_per_request, H = spec.hidden;
    assert(M > 0 && M <= ws.max_rows && i >= 0 && i <= spec.n_layers());
    if (i == 0) {
        look_up_embeddings({ws.token_ids}, {ws.residual}, M);
    } else {  // the rest of layer i - 1: h = x + proj(o); x = h + mlp(feed_forward_norm(h))
        const Layer& p = layers[i - 1];
        multiply_weights(p.attention_output, {ws.attention_result}, {ws.projected}, M);
        add({ws.residual}, {ws.projected}, {ws.after_attention}, (size_t)M * H);
        normalize(p.feed_forward_norm, {ws.after_attention}, {ws.normalized}, M, H);
        if (merge_projections && M == 1) {
            multiply_weights(p.merged_gate_up, {ws.normalized}, {ws.gate_values}, M);
            activate_gate({ws.gate_values}, {ws.gate_values, p.feed_forward_gate.out_size * ELEM}, {ws.gate_values}, p.feed_forward_gate.out_size);
        } else {
            multiply_weights(p.feed_forward_gate, {ws.normalized}, {ws.gate_values}, M);
            indep = true, multiply_weights(p.feed_forward_up, {ws.normalized}, {ws.up_values}, M);
            activate_gate({ws.gate_values}, {ws.up_values}, {ws.gate_values}, (size_t)M * p.feed_forward_gate.out_size);
        }
        multiply_weights(p.feed_forward_down, {ws.gate_values}, {ws.projected}, M);
        add({ws.after_attention}, {ws.projected}, {ws.residual}, (size_t)M * H);
    }
    if (i == spec.n_layers()) {
        normalize(norm, {ws.residual}, {ws.residual}, M, H);
        return;
    }
    const Layer& l = layers[i];
    normalize(l.attention_norm, {ws.residual}, {ws.normalized}, M, H);
    const bool m1 = merge_projections && M == 1;
    if (l.is_linear && m1) {
        multiply_weights(l.merged_linear_projections, {ws.normalized}, {ws.query_key_value}, M);
        return;
    }
    if (l.is_linear) {
        multiply_weights(l.query_key_value_projection, {ws.normalized}, {ws.query_key_value}, M);
        indep = true, multiply_weights(l.output_gate_projection, {ws.normalized}, {ws.output_gate}, M);
        indep = true, multiply_weights(l.update_rate_projection, {ws.normalized}, {ws.update_rate_input}, M);
        indep = true, multiply_weights(l.decay_projection, {ws.normalized}, {ws.decay_input}, M);
        return;
    }
    const int D = spec.head_dim;
    multiply_weights(m1 ? l.merged_query_key_value : l.query_projection, {ws.normalized}, {ws.query_and_gate}, M);
    [enc setComputePipelineState:split_q_gate_ps];
    set(enc, {ws.query_and_gate}, 0), set(enc, {ws.query}, 1), set(enc, {ws.query_gate}, 2);
    pre(), [enc dispatchThreads:MTLSizeMake(D, (size_t)M * spec.query_heads, 1) threadsPerThreadgroup:MTLSizeMake(D, 1, 1)];
    normalize(l.query_norm, {ws.query}, {ws.query}, M * spec.query_heads, D);
    if (m1) {  // k, v follow q | gate in ws.query_and_gate; v stays there (full_attn reads it)
        indep = true, normalize(l.key_norm, {ws.query_and_gate, l.query_projection.out_size * ELEM}, {ws.key}, spec.key_value_heads, D);
        return;
    }
    indep = true, multiply_weights(l.key_projection, {ws.normalized}, {ws.key}, M);
    normalize(l.key_norm, {ws.key}, {ws.key}, M * spec.key_value_heads, D);
    indep = true, multiply_weights(l.value_projection, {ws.normalized}, {ws.value}, M);
}

// mlx Matmul::eval_gpu for the fallback's two matmuls (matmul.cpp): batch [Hkv, gqa] never collapses (the repeat
// stride of K/V is 0), H batches rule out split-k, so: gemv_wide when transposed B and 2 <= M <= 15, else
// steel_matmul_regular_axpby with the 's' tiles (GEMM_TPARAM_MACRO: bm 64, bn 64, bk 16, wm 2, wn 2).
void GpuOps::matmul(bool transpose_b, Buf a, Buf b, Buf d, int M, int N, int K, int lda, int ldb, int ldd,
                    const int64_t a_bs[2], const int64_t b_bs[2], int64_t d_bs) {
    const int batch_shape[2] = {spec.key_value_heads, spec.query_heads / spec.key_value_heads}, batches = spec.query_heads;
    int passes = (M + 4) / 5;
    if (transpose_b && M >= 2 && passes <= 3 && K % 4 == 0) {  // gemv_wide_config (gen >= 15, aligned)
        assert(d_bs == (int64_t)M * N);                         // gemv_wide writes [batch, M, N] densely
        int vecs = (M + passes - 1) / passes, k_lanes = passes == 1 || N <= 64 ? 32 : 16;
        assert(vecs >= 3 && vecs <= 5);  // M 6..15: the fallback runs above 8 queries, or above 32 / gqa
        [enc setComputePipelineState:gemv_wide_ps[vecs - 3][k_lanes == 32]];
        set(enc, b, 0), set(enc, a, 1), set(enc, d, 3);
        set_bytes(enc, K, 4), set_bytes(enc, N, 5), set_bytes(enc, M, 6), set_bytes(enc, ldb, 7), set_bytes(enc, lda, 8);
        set_bytes(enc, 2, 11);
        [enc setBytes:batch_shape length:sizeof batch_shape atIndex:12];
        [enc setBytes:a_bs length:2 * sizeof(int64_t) atIndex:13];
        [enc setBytes:b_bs length:2 * sizeof(int64_t) atIndex:14];
        pre(), [enc dispatchThreadgroups:MTLSizeMake(N >= 65536 ? 1 : passes, (N + 3) / 4, batches)
            threadsPerThreadgroup:MTLSizeMake(32, k_lanes / 8, 1)];
        return;
    }
    const int bm = 64, bn = 64, bk = 16;
    [enc setComputePipelineState:steel_ps[!transpose_b][M % bm == 0][N % bn == 0][K % bk == 0]];
    const int tn = (N + bn - 1) / bn, tm = (M + bm - 1) / bm;
    mlx::steel::GEMMParams params{M, N, K, lda, ldb, ldd, tn, tm, a_bs[1], b_bs[1], d_bs, 0, K / bk, 2};
    const int64_t strides[4] = {a_bs[0], a_bs[1], b_bs[0], b_bs[1]};
    set(enc, a, 0), set(enc, b, 1), set(enc, d, 3), set_bytes(enc, params, 4);
    [enc setBytes:batch_shape length:sizeof batch_shape atIndex:6];
    [enc setBytes:strides length:sizeof strides atIndex:7];
    pre(), [enc dispatchThreadgroups:MTLSizeMake(tn, tm, batches) threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
}

void GpuOps::apply_attention(int i, Workspace& ws) {
    begin(ws);
    assert(i >= 0 && i < spec.n_layers());
    if (layers[i].is_linear)
        linear_attn(i, layers[i], ws);
    else
        full_attn(layers[i], ws);
}

// gpu.py _linear, batched: one dispatch per op over all rows (MLX runs them one row at a time; every kernel here
// computes a row the same way regardless of B). Same lane contract as full_attn: B == 1 is a prefill row with
// ws.tokens_per_request == n, B > 1 is decode (S == 1, rows with n == 0 skipped).
void GpuOps::linear_attn(int i, const Layer& l, Workspace& ws) {
    const int B = ws.request_count, S = ws.tokens_per_request, M = B * S, C = spec.conv_dim(), Hv = spec.linear_value_heads, Dk = spec.linear_key_dim;
    const int key_dim = spec.linear_key_heads * Dk;
    id<MTLBuffer> table = ws.step_table;
    const uint32_t* tb = (const uint32_t*)table.contents;
    // the table must carry every row's state span: the kernels index it with n_state = mem.state_units
    assert((int)tb[0] == B && tb[1] == mem.state_units && M <= ws.max_rows && (B == 1 || S == 1));
    for (int b = 0; b < B; b++) assert(B == 1 ? (int)tb[2 + B + b] == S : (int)tb[2 + B + b] <= S);
    const uint64_t unit = mem.arena.unit_bytes, conv_off = mem.state_offset(i), delta_off = conv_off + mem.conv_bytes;
    const int n_state = (int)mem.state_units;
    // gated_delta_paged reads each lane's 32 floats as one run: runs must not cross a unit
    assert(unit % 128 == 0 && delta_off % 128 == 0);
    id<MTLBuffer> arena = mem.arena_buffer;
    // merge_projections: dense wrote [qkv | z | b | a] as one row in ws.query_key_value
    const bool m1 = merge_projections && M == 1;
    const size_t vd = (size_t)Hv * spec.linear_value_dim;
    const Buf z = m1 ? Buf{ws.query_key_value, C * ELEM} : Buf{ws.output_gate}, b = m1 ? Buf{ws.query_key_value, (C + vd) * ELEM} : Buf{ws.update_rate_input},
              a = m1 ? Buf{ws.query_key_value, (C + vd + Hv) * ELEM} : Buf{ws.decay_input};

    const double inv_scale = std::pow((double)Dk, -0.5);
    if (S == 1) {  // decode: conv, norm and scales fused (conv_norm_qk), same eps and scales as below
        const float eps = (float)(1e-6 * inv_scale * inv_scale), q_scale = (float)(inv_scale * inv_scale);
        assert(C % Dk == 0 && key_dim % Dk == 0);
        [enc setComputePipelineState:conv_norm_qk_ps];
        set(enc, {ws.query_key_value}, 0), set(enc, l.conv_weights, 1), set(enc, {arena}, 2), set(enc, {table}, 3);
        set_bytes(enc, unit, 4), set_bytes(enc, n_state, 5), set_bytes(enc, conv_off, 6), set(enc, {ws.normalized_query_key}, 7);
        set_bytes(enc, eps, 8), set_bytes(enc, q_scale, 9), set_bytes(enc, (float)inv_scale, 10), set_bytes(enc, key_dim, 11);
        pre(), [enc dispatchThreadgroups:MTLSizeMake(C / Dk, B, 1) threadsPerThreadgroup:MTLSizeMake(Dk, 1, 1)];
    } else {
        [enc setComputePipelineState:conv1d_silu_ps];
        set(enc, {ws.query_key_value}, 0), set(enc, l.conv_weights, 1), set(enc, {arena}, 2), set(enc, {table}, 3);
        set_bytes(enc, unit, 4), set_bytes(enc, n_state, 5), set_bytes(enc, conv_off, 6), set_bytes(enc, S, 7);
        pre(), [enc dispatchThreads:MTLSizeMake(C, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // normalize_qk: rms_norm without weight (eps scaled to the mean), then the scales. Rows of 128 over the whole
        // conv output (v too, into scratch; only q / k are used), as MLX norms the q / k slices.
        normalize({one}, {ws.query_key_value}, {ws.normalized_query_key}, M * (C / Dk), Dk, (float)(1e-6 * inv_scale * inv_scale), 0);
        [enc setComputePipelineState:scale_qk_ps];
        set(enc, {ws.normalized_query_key}, 0), set_bytes(enc, (float)(inv_scale * inv_scale), 1), set_bytes(enc, (float)inv_scale, 2);
        set_bytes(enc, C, 3);
        pre(), [enc dispatchThreads:MTLSizeMake(2 * key_dim, M, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    indep = true;
    [enc setComputePipelineState:gdn_gates_ps];
    set(enc, a, 0), set(enc, b, 1), set(enc, l.decay_log_base, 2), set(enc, l.decay_bias, 3), set(enc, {ws.decay}, 4);
    set(enc, {ws.update_rate}, 5);
    pre(), [enc dispatchThreads:MTLSizeMake(Hv, M, 1) threadsPerThreadgroup:MTLSizeMake(Hv, 1, 1)];

    // mlx-lm: grid (32, Dv / 8, B * Hv), threadgroup (32, 2, 1)
    [enc setComputePipelineState:gated_delta_ps];
    set(enc, {ws.normalized_query_key}, 0), set(enc, {ws.normalized_query_key, (size_t)key_dim * ELEM}, 1), set(enc, {ws.query_key_value, 2 * (size_t)key_dim * ELEM}, 2);
    set(enc, {ws.decay}, 3), set(enc, {ws.update_rate}, 4), set(enc, {arena}, 5), set(enc, {ws.attention_result}, 6), set(enc, {table}, 7);
    set_bytes(enc, S, 8), set_bytes(enc, C, 9), set_bytes(enc, unit, 10), set_bytes(enc, n_state, 11);
    set_bytes(enc, delta_off, 12);
    pre(), [enc dispatchThreads:MTLSizeMake(32, spec.linear_value_dim / 8, B * Hv) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];

    normalize_and_gate(l.output_norm, {ws.attention_result}, z, {ws.attention_result}, M * Hv, spec.linear_value_dim);
}

// gpu.py _full, batched: one dispatch per op over all rows. MLX v0.32.4 ops in Python's order: fast::rope on q and k,
// write_kv, fast::scaled_dot_product_attention (sdpa_vector / 2-pass while sdpa_vector_ok), * sigmoid(gate).
// Lane contract: Python's GPU path never pads, and MLX picks kernels by q length, so ws.tokens_per_request is the row's n_valid when
// B == 1 (prefill; g6's ANE lane converts its padded output to n rows before attn). B > 1 is decode: S == 1, rows
// with n == 0 are skipped.
void GpuOps::full_attn(const Layer& l, Workspace& ws) {
    const int B = ws.request_count, S = ws.tokens_per_request, H = spec.query_heads, Hkv = spec.key_value_heads, D = spec.head_dim, gqa = H / Hkv;
    id<MTLBuffer> table = ws.step_table;
    const uint32_t* tb = (const uint32_t*)table.contents;  // the CPU wrote it this step
    assert((int)tb[0] == B && B * S <= ws.max_rows && (B == 1 || S == 1));
    int min_N = INT_MAX, max_N = 0;
    for (int b = 0; b < B; b++) {
        int pos = tb[2 + b], n = tb[2 + B + b], N = pos + n;
        assert((B == 1 ? n == S : n <= S) && N <= ws.max_context && tb[3 + 2 * B + b] - tb[2 + 2 * B + b] >= mem.units_for(N));
        if (n) min_N = std::min(min_N, N), max_N = std::max(max_N, N);
    }
    if (!max_N) return;

    // mlx fast::rope (rope.cpp), in place: the first ROPE_DIMS of each head rotate, the rest stays (MLX copies it).
    // Batched kernel "rope_bfloat16", per-row offsets = the table's pos words.
    auto rope = [&](id<MTLBuffer> x, int heads) {
        [enc setComputePipelineState:rope_ps];
        set(enc, {x}, 0), set(enc, {x}, 1), set(enc, {table, 8}, 2), set_bytes(enc, 1.0f, 3);
        int64_t strides[3] = {D, (int64_t)heads * D, 1};
        [enc setBytes:strides length:sizeof strides atIndex:4];
        [enc setBytes:strides length:sizeof strides atIndex:5];
        set_bytes(enc, (int64_t)1, 6), set_bytes(enc, heads, 7), set_bytes(enc, std::log2(ROPE_THETA), 10);
        pre(), [enc dispatchThreads:MTLSizeMake(ROPE_DIMS / 2, S, B * ((heads + 3) / 4)) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    };
    rope(ws.query, H);
    indep = true, rope(ws.key, Hkv);

    KVLayout L{mem.arena.unit_bytes / ELEM, mem.page_tokens, Hkv, l.full_index};
    id<MTLBuffer> arena = mem.arena_buffer;
    [enc setComputePipelineState:write_kv_ps];
    // merge_projections: v is still in dense's [q | gate | k | v] row in ws.query_and_gate
    const Buf v = merge_projections && B * S == 1 ? Buf{ws.query_and_gate, (size_t)(l.query_projection.out_size + l.key_projection.out_size) * ELEM} : Buf{ws.value};
    set(enc, {ws.key}, 0), set(enc, v, 1), set(enc, {arena}, 2), set(enc, {table}, 3);
    set_bytes(enc, L, 4), set_bytes(enc, S, 5);
    pre(), [enc dispatchThreads:MTLSizeMake(D, Hkv, B * S) threadsPerThreadgroup:MTLSizeMake(D, 1, 1)];

    const float scale = 1.0f / std::sqrt((float)D);  // mlx-lm: head_dim ** -0.5
    if (sdpa_vector_ok(S, gqa)) {  // MLX: sdpa_vector, 2-pass from SDPA_2PASS_KEYS keys. Rows pick their path by N.
        if (max_N >= SDPA_2PASS_KEYS) {
            const int blocks = sdpa_blocks(max_N, gqa * S);
            assert((size_t)B * H * S * blocks <= ws.two_pass_sums.length / sizeof(float));
            id<MTLComputePipelineState> ps = sdpa_2pass_1[32 * gqa * S > (int)sdpa_2pass_1[0].maxTotalThreadsPerThreadgroup];
            [enc setComputePipelineState:ps];
            set(enc, {ws.query}, 0), set(enc, {arena}, 1), set(enc, {ws.two_pass_partials}, 3), set(enc, {ws.two_pass_sums}, 4), set(enc, {ws.two_pass_maxes}, 5);
            set(enc, {table}, 6), set_bytes(enc, L, 7), set_bytes(enc, SDPA_2PASS_KEYS, 8), set_bytes(enc, INT_MAX, 9);
            set_bytes(enc, blocks, 11), set_bytes(enc, scale, 12);
            pre(), [enc dispatchThreadgroups:MTLSizeMake(Hkv, B, blocks) threadsPerThreadgroup:MTLSizeMake(32, gqa, S)];
            // over every row; the 1-pass below then overwrites the rows that are not on this path
            [enc setComputePipelineState:sdpa_2pass_2];
            set(enc, {ws.two_pass_partials}, 0), set(enc, {ws.two_pass_sums}, 1), set(enc, {ws.two_pass_maxes}, 2), set(enc, {ws.attention_heads}, 3), set_bytes(enc, blocks, 4);
            pre(), [enc dispatchThreadgroups:MTLSizeMake(B * H, S, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
        if (min_N < SDPA_2PASS_KEYS) {
            [enc setComputePipelineState:sdpa_ps];
            set(enc, {ws.query}, 0), set(enc, {arena}, 1), set(enc, {ws.attention_heads}, 3), set_bytes(enc, gqa, 4), set(enc, {table}, 5);
            set_bytes(enc, L, 6), set_bytes(enc, 0, 7), set_bytes(enc, SDPA_2PASS_KEYS, 8), set_bytes(enc, scale, 10);
            pre(), [enc dispatchThreadgroups:MTLSizeMake(B * H, S, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
        }
    } else {  // MLX's unfused fallback (fast.cpp; use_fallback: head_dim 256, !sdpa_vector_ok), row 0 only
        assert(B == 1);
        const int n = S, kL = max_N;
        // ponytail: MLX v0.32.4 runs fused steel attention for n >= 2048 == kL (no prefix); not ported (chunks are
        // <= 512). Port it if a lane sends such a chunk.
        assert(!(n >= 2048 && kL == n));
        assert((size_t)n * H * kL * ELEM <= ws.attention_scores.length && (size_t)Hkv * kL * D * ELEM <= ws.gathered_keys.length);
        [enc setComputePipelineState:gather_kv_ps];
        set(enc, {arena}, 0), set(enc, {ws.gathered_keys}, 1), set(enc, {ws.gathered_values}, 2), set(enc, {table}, 3), set_bytes(enc, L, 4);
        pre(), [enc dispatchThreads:MTLSizeMake(D, kL, Hkv) threadsPerThreadgroup:MTLSizeMake(D, 1, 1)];
        [enc setComputePipelineState:scale_ps];
        set(enc, {ws.query}, 0), set_bytes(enc, scale, 1);
        pre(), [enc dispatchThreads:MTLSizeMake((size_t)n * H * D, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // scores [H, n, kL] = q kᵀ; q is our [n, H, D], (kv head, repeat) -> head kv * gqa + repeat
        const int64_t q_bs[2] = {(int64_t)gqa * D, D}, kv_bs[2] = {(int64_t)kL * D, 0};
        matmul(true, {ws.query}, {ws.gathered_keys}, {ws.attention_scores}, n, kL, D, H * D, D, kL, q_bs, kv_bs, (int64_t)n * kL);
        [enc setComputePipelineState:causal_mask_ps];
        set(enc, {ws.attention_scores}, 0), set(enc, {table}, 1);
        pre(), [enc dispatchThreads:MTLSizeMake(kL, n, H) threadsPerThreadgroup:MTLSizeMake(std::min(kL, 256), 1, 1)];
        // softmax(precise=True), in place (softmax.cpp)
        const bool looped = kL > 4096;
        id<MTLComputePipelineState> sm = softmax_ps[looped];
        size_t tg = looped ? sm.maxTotalThreadsPerThreadgroup : 32 * (((kL + 3) / 4 + 31) / 32);
        assert(tg <= sm.maxTotalThreadsPerThreadgroup);
        [enc setComputePipelineState:sm];
        set(enc, {ws.attention_scores}, 0), set(enc, {ws.attention_scores}, 1), set_bytes(enc, kL, 2);
        pre(), [enc dispatchThreads:MTLSizeMake((size_t)H * n * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        // ao [H, S, D] = p v
        const int64_t p_bs[2] = {(int64_t)gqa * n * kL, (int64_t)n * kL};
        matmul(false, {ws.attention_scores}, {ws.gathered_values}, {ws.attention_heads}, n, D, kL, kL, D, D, p_bs, kv_bs, (int64_t)S * D);
    }

    [enc setComputePipelineState:sigmoid_gate_ps];
    set(enc, {ws.attention_heads}, 0), set(enc, {ws.query_gate}, 1), set(enc, {ws.attention_result}, 2), set_bytes(enc, S, 3);
    pre(), [enc dispatchThreads:MTLSizeMake(D, H, B * S) threadsPerThreadgroup:MTLSizeMake(D, 1, 1)];
}

}  // namespace mllm

namespace mllm {

void GpuOps::pick_next_tokens(Workspace& ws, const std::vector<SampleParams>& params) {
    begin(ws);
    const int B = ws.request_count, V = lm.out_size;
    assert(ws.tokens_per_request == 1 && B <= ws.max_rows && ws.next_token_scores.length >= (size_t)B * V * ELEM);
    assert(params.empty() || (int)params.size() == B);
    multiply_weights(lm, {ws.residual}, {ws.next_token_scores}, B);
    sample_tokens({ws.next_token_scores}, {ws.next_token_ids}, V, params.empty() ? std::vector<SampleParams>(B) : params);
}

std::vector<int> GpuOps::next_tokens(const Workspace& ws) const {
    const int32_t* p = (const int32_t*)ws.next_token_ids.contents;
    return std::vector<int>(p, p + ws.request_count);
}

void GpuOps::feed_previous_tokens(Workspace& ws, id<MTLBuffer> prev_ids, const std::vector<int>& src) {
    begin(ws);
    assert((int)src.size() == ws.request_count && ws.tokens_per_request == 1);
    [enc setComputePipelineState:feed_ps];
    set(enc, {prev_ids}, 0), set(enc, {ws.token_ids}, 1);
    [enc setBytes:src.data() length:src.size() * sizeof(int) atIndex:2];
    pre(), [enc dispatchThreads:MTLSizeMake(src.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(src.size(), 1, 1)];
}

void GpuOps::sample_tokens(Buf x, Buf out, int V, const std::vector<SampleParams>& params) {
    struct SampleRow {  // sample.metal SampleRow
        float temp, top_p;
        uint64_t seed;
    };
    std::vector<SampleRow> rows;
    for (const SampleParams& p : params) {
        assert(p.temperature >= 0 && (p.temperature == 0 || p.rng));
        rows.push_back({p.temperature, p.top_p, p.temperature > 0 ? (*p.rng)() : 0});
    }
    [enc setComputePipelineState:sample_ps];
    set(enc, x, 0), set(enc, out, 1), set_bytes(enc, V, 2);
    [enc setBytes:rows.data() length:rows.size() * sizeof(SampleRow) atIndex:3];
    pre(), [enc dispatchThreadgroups:MTLSizeMake(rows.size(), 1, 1) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];
}

}  // namespace mllm
