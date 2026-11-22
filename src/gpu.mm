#include "gpu.h"

#include <cassert>
#include <stdexcept>

namespace mllm {

// ponytail: mlx-lm reads rms_norm_eps from config.json; every Qwen3.5 uses 1e-6. Add it to ModelSpec when a model
// differs.
constexpr float EPS = 1e-6f;

static std::runtime_error error(const std::string& what, NSError* err) {
    return std::runtime_error(what + (err ? std::string(": ") + err.localizedDescription.UTF8String : ""));
}

id<MTLComputePipelineState> GpuOps::pipeline(NSString* name) {
    id<MTLFunction> f = [library newFunctionWithName:name];
    if (!f) throw error(std::string("no kernel ") + name.UTF8String + " in " + METALLIB, nil);
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:f error:&err];
    if (!ps) throw error(std::string("pipeline ") + name.UTF8String, err);
    return ps;
}

GpuOps::GpuOps(Memory& mem, const Weights& weights, int max_rows)
    : spec(mem.spec), device(mem.device), max_rows(max_rows) {
    // MLX picks kernels and tile sizes by GPU generation. Only the M4 family's branch (gen 15/16, not a 'd' Max/Ultra
    // die) is copied below. ponytail: port the other branches of quantized.cpp when another GPU shows up.
    NSString* arch = device.architecture.name;
    if (!([arch hasPrefix:@"applegpu_g15"] || [arch hasPrefix:@"applegpu_g16"]) || [arch hasSuffix:@"d"])
        throw error(std::string("GpuOps: dispatch rules not ported for ") + arch.UTF8String, nil);

    NSError* err = nil;
    library = [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&err];
    if (!library) throw error(std::string("cannot load ") + METALLIB, err);
    queue = [device newCommandQueue];

    qmv_fast = pipeline(@"affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4");
    for (int nv = 2; nv <= 5; nv++)
        qmv_wide[nv] = pipeline([NSString stringWithFormat:@"affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_%d_kl_8_batch_0", nv]);
    for (int al = 0; al < 2; al++) {
        NSString* a = al ? @"true" : @"false";
        qmm_t[al] = pipeline([NSString stringWithFormat:@"affine_qmm_t_bfloat16_t_gs_64_b_4_alN_%@_batch_0", a]);
        qmm_t_splitk[al] = pipeline([NSString stringWithFormat:@"affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_%@", a]);
    }
    splitk_accum = pipeline(@"steel_gemm_splitk_accum_bfloat16_float32");
    embed_ps = pipeline(@"embed");
    rms_ps = pipeline(@"rmsbfloat16");
    add_ps = pipeline(@"add");
    swiglu_ps = pipeline(@"swiglu");
    precise_swiglu_ps = pipeline(@"precise_swiglu");
    split_q_gate_ps = pipeline(@"split_q_gate");

    auto buf = [&](const std::string& name) {
        auto v = weights[name];
        return Buf{v.buffer, v.offset};
    };
    auto qlinear = [&](const std::string& p) {
        auto w = weights[p + ".weight"];
        auto s = weights[p + ".scales"];
        QLinear l{buf(p + ".weight"), buf(p + ".scales"), buf(p + ".biases"), (int)w.t.shape[0], (int)w.t.shape[1] * 8};
        if (w.t.dtype != "U32" || s.t.dtype != "BF16" || s.t.shape[1] * 64 != l.K)
            throw std::runtime_error(p + ": not 4-bit affine, group 64, bf16");
        // linear() ports only qmv_fast for small M: no qmv_quad (K 64 / 128), qmv (K % 512) or partial rows (N % 8)
        if (l.K % 512 || l.N % 8) throw std::runtime_error(p + ": K % 512 or N % 8 not 0 (kernel not ported)");
        return l;
    };
    const std::string P = "language_model.model.";
    embed_tokens = qlinear(P + "embed_tokens");
    norm = buf(P + "norm.weight");
    for (int i = 0; i < spec.n_layers(); i++) {
        std::string p = P + "layers." + std::to_string(i) + ".";
        Layer l{};
        l.is_linear = spec.kinds[i] == "linear";
        l.input_layernorm = buf(p + "input_layernorm.weight");
        l.post_attention_layernorm = buf(p + "post_attention_layernorm.weight");
        l.gate_proj = qlinear(p + "mlp.gate_proj");
        l.up_proj = qlinear(p + "mlp.up_proj");
        l.down_proj = qlinear(p + "mlp.down_proj");
        if (l.is_linear) {
            l.out = qlinear(p + "linear_attn.out_proj");
            l.in_proj_qkv = qlinear(p + "linear_attn.in_proj_qkv");
            l.in_proj_z = qlinear(p + "linear_attn.in_proj_z");
            l.in_proj_b = qlinear(p + "linear_attn.in_proj_b");
            l.in_proj_a = qlinear(p + "linear_attn.in_proj_a");
        } else {
            l.out = qlinear(p + "self_attn.o_proj");
            l.q_proj = qlinear(p + "self_attn.q_proj");
            l.k_proj = qlinear(p + "self_attn.k_proj");
            l.v_proj = qlinear(p + "self_attn.v_proj");
            l.q_norm = buf(p + "self_attn.q_norm.weight");
            l.k_norm = buf(p + "self_attn.k_norm.weight");
        }
        layers.push_back(l);
    }

    size_t row = (size_t)max_rows * ELEM;
    h = mem.buffer(row * spec.hidden);
    xn = mem.buffer(row * spec.hidden);
    t = mem.buffer(row * spec.hidden);
    g = mem.buffer(row * layers[0].gate_proj.N);
    u = mem.buffer(row * layers[0].gate_proj.N);
    qg = mem.buffer(row * 2 * spec.n_heads * spec.head_dim);
    splitk = mem.buffer(512 * 32 * 32 * sizeof(float));  // split_k * M * N <= 512 tiles of 32 x 32 (see linear)
}

void GpuOps::begin() {
    cb = [queue commandBuffer];
    enc = [cb computeCommandEncoder];  // serial: each dispatch sees the previous one's writes
}

double GpuOps::commit_and_wait() {
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) throw error("GpuOps: command buffer failed", cb.error);
    double seconds = cb.GPUEndTime - cb.GPUStartTime;
    enc = nil, cb = nil;
    return seconds;
}

static void set(id<MTLComputeCommandEncoder> enc, Buf b, int i) { [enc setBuffer:b.buffer offset:b.offset atIndex:i]; }
template <class T>
static void set_bytes(id<MTLComputeCommandEncoder> enc, T v, int i) { [enc setBytes:&v length:sizeof(T) atIndex:i]; }

void GpuOps::embed(Buf tokens, Buf out, int rows) {
    const QLinear& e = embed_tokens;
    [enc setComputePipelineState:embed_ps];
    set(enc, e.w, 0), set(enc, e.scales, 1), set(enc, e.biases, 2), set(enc, tokens, 3), set(enc, out, 4);
    set_bytes(enc, e.K, 5);
    [enc dispatchThreads:MTLSizeMake(e.K / 8, rows, 1) threadsPerThreadgroup:MTLSizeMake(std::min(e.K / 8, 256), 1, 1)];
}

// mlx RMSNorm::eval_gpu, rms_single_row branch
void GpuOps::rms_norm(Buf weight, Buf x, Buf out, int rows, int dim) {
    assert(dim <= 4096);  // RMS_LOOPED_LIMIT; wider rows need rms_looped
    uint32_t tg = 32 * (((dim + 3) / 4 + 31) / 32), axis = dim, w_stride = 1;
    assert(tg <= rms_ps.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:rms_ps];
    set(enc, x, 0), set(enc, weight, 1), set(enc, out, 2);
    set_bytes(enc, EPS, 3), set_bytes(enc, axis, 4), set_bytes(enc, w_stride, 5);
    [enc dispatchThreads:MTLSizeMake((size_t)rows * tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// Qwen3NextRMSNormGated: precise_swiglu(x, z, rms_norm(x, weight))
void GpuOps::gated_rms_norm(Buf weight, Buf x, Buf z, Buf out, int rows, int dim) {
    rms_norm(weight, x, out, rows, dim);
    elementwise(precise_swiglu_ps, z, out, out, (size_t)rows * dim);
}

void GpuOps::elementwise(id<MTLComputePipelineState> ps, Buf a, Buf b, Buf out, size_t n) {
    [enc setComputePipelineState:ps];
    set(enc, a, 0), set(enc, b, 1), set(enc, out, 2);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(n, 256), 1, 1)];
}

void GpuOps::add(Buf x, Buf y, Buf out, size_t n) { elementwise(add_ps, x, y, out, n); }
void GpuOps::swiglu(Buf gate, Buf x, Buf out, size_t n) { elementwise(swiglu_ps, gate, x, out, n); }

// mlx QuantizedMatmul::eval_gpu with transpose, one batch, x row-contiguous [M, K]: the same kernel, tile and grid
// choice (quantized.cpp v0.32.4: get_qmv_batch_limit, dispatch_qmv, qmv_wide, qmv, qmm_splitk, qmm).
void GpuOps::linear(const QLinear& l, Buf x, Buf y, int M) {
    const int N = l.N, K = l.K;
    auto bind = [&](id<MTLComputePipelineState> ps) {
        [enc setComputePipelineState:ps];
        set(enc, l.w, 0), set(enc, l.scales, 1), set(enc, l.biases, 2), set(enc, x, 3);
    };
    int limit = K <= 2048 && N <= 2048 ? 13 : K <= 4096 && N <= 4096 ? 15 : 13;  // gen 15/16, not 'd'
    if (M >= limit) {
        int n_tiles = (N + 31) / 32, m_tiles = (M + 31) / 32;
        int split_k = std::min(std::max(1, 512 / (n_tiles * m_tiles)), K / 64);
        while (split_k > 1 && K % (split_k * 64)) split_k--;
        bool aligned = N % 32 == 0;
        if (split_k <= 1) {
            bind(qmm_t[aligned]);
            set(enc, y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
            [enc dispatchThreadgroups:MTLSizeMake(n_tiles, m_tiles, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
            return;
        }
        int k_part = K / split_k, stride = M * N;
        assert((size_t)split_k * stride * sizeof(float) <= splitk.length);
        bind(qmm_t_splitk[aligned]);
        set(enc, {splitk}, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
        set_bytes(enc, k_part, 8), set_bytes(enc, stride, 9);
        [enc dispatchThreadgroups:MTLSizeMake(n_tiles, m_tiles, split_k) threadsPerThreadgroup:MTLSizeMake(32, 2, 2)];
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
        [enc dispatchThreads:MTLSizeMake(N, M, 1) threadsPerThreadgroup:MTLSizeMake(1 << p0, 1 << p1, 1)];
    } else if (M >= 2) {  // qmv_wide: vecs_per_tg rows of x per threadgroup, 8 output rows per threadgroup
        int n_tiles = (M + 4) / 5, nv = (M + n_tiles - 1) / n_tiles;
        assert(nv >= 2 && nv <= 5);
        bind(qmv_wide[nv]);
        set(enc, y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6), set_bytes(enc, M, 7);
        [enc dispatchThreadgroups:MTLSizeMake((M + nv - 1) / nv, (N + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    } else {  // qmv_fast, no partial rows (shapes checked at load)
        bind(qmv_fast);
        set(enc, y, 4), set_bytes(enc, K, 5), set_bytes(enc, N, 6);
        [enc dispatchThreadgroups:MTLSizeMake(M, N / 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    }
}

void GpuOps::dense(int i, Workspace& ws) {
    const int M = ws.B * ws.S, H = spec.hidden;
    assert(M > 0 && M <= max_rows && i >= 0 && i <= spec.n_layers());
    if (i == 0) {
        embed({ws.tokens}, {ws.x}, M);
    } else {  // the rest of layer i - 1: h = x + proj(o); x = h + mlp(post_attention_layernorm(h))
        const Layer& p = layers[i - 1];
        linear(p.out, {ws.o}, {t}, M);
        add({ws.x}, {t}, {h}, (size_t)M * H);
        rms_norm(p.post_attention_layernorm, {h}, {xn}, M, H);
        linear(p.gate_proj, {xn}, {g}, M);
        linear(p.up_proj, {xn}, {u}, M);
        swiglu({g}, {u}, {g}, (size_t)M * p.gate_proj.N);
        linear(p.down_proj, {g}, {t}, M);
        add({h}, {t}, {ws.x}, (size_t)M * H);
    }
    if (i == spec.n_layers()) {
        rms_norm(norm, {ws.x}, {ws.x}, M, H);
        return;
    }
    const Layer& l = layers[i];
    rms_norm(l.input_layernorm, {ws.x}, {xn}, M, H);
    if (l.is_linear) {
        linear(l.in_proj_qkv, {xn}, {ws.qkv}, M);
        linear(l.in_proj_z, {xn}, {ws.z}, M);
        linear(l.in_proj_b, {xn}, {ws.b}, M);
        linear(l.in_proj_a, {xn}, {ws.a}, M);
        return;
    }
    const int D = spec.head_dim;
    linear(l.q_proj, {xn}, {qg}, M);
    [enc setComputePipelineState:split_q_gate_ps];
    set(enc, {qg}, 0), set(enc, {ws.q}, 1), set(enc, {ws.gate}, 2);
    [enc dispatchThreads:MTLSizeMake(D, (size_t)M * spec.n_heads, 1) threadsPerThreadgroup:MTLSizeMake(D, 1, 1)];
    rms_norm(l.q_norm, {ws.q}, {ws.q}, M * spec.n_heads, D);
    linear(l.k_proj, {xn}, {ws.k}, M);
    rms_norm(l.k_norm, {ws.k}, {ws.k}, M * spec.n_kv_heads, D);
    linear(l.v_proj, {xn}, {ws.v}, M);
}

}  // namespace mllm
