// GpuOps: dense(i), attn(i), lm_head on Metal, sample on the CPU (py/mllm/gpu.py). Kernels are MLX's (src/vendor/mlx),
// dispatched the way MLX v0.32.4 dispatches them, so results match mlx-lm bit for bit.
#pragma once
#import <Metal/Metal.h>

#include <cassert>
#include <random>
#include <thread>

#include "memory.h"

namespace mllm {

// A tensor bound for a kernel: buffer + byte offset.
struct Buf {
    id<MTLBuffer> buffer;
    size_t offset = 0;
};

// 4-bit affine linear, MLX format: weight U32 [N, K / 8], scales / biases bf16 [N, K / 64]. y = x @ dequant(w)ᵀ.
struct QLinear {
    Buf w, scales, biases;
    int N = 0, K = 0;
};

// One row's sampler, mlx-lm make_sampler(temp, top_p): temp 0 = greedy; top_p in (0, 1) keeps the nucleus.
struct SampleParams {
    float temp = 0, top_p = 0;
    std::mt19937_64* rng = nullptr;  // the request's; needed when temp > 0
};

class GpuOps {
public:
    GpuOps(Memory& mem, const Weights& weights);

    // mem.workspace(chunk, max_batch) plus GpuOps' scratch, sized for max_ctx tokens per request. One per lane.
    Workspace workspace(int chunk, int max_batch, int max_ctx);

    // Every op below encodes into ws.enc and uses ws's scratch.
    // All the big matmuls between attn(i - 1) and attn(i), i = 0 .. n_layers, over ws.B * ws.S rows.
    // Fields as in gpu.py, except q, k, v are row-major [B, S, H, D] (gpu.py's [B, H, S, D] transpose is only
    // for MLX's SDPA).
    void dense(int i, Workspace& ws);

    // Everything of layer i that touches the arena (gpu.py attn): ws.q/k/v/gate of ws.B rows x ws.S tokens -> ws.o
    // [B, S, H * D]. ws.table: this step's StepTables copy (row b = ws row b); only each row's n valid tokens are
    // written. Full layers: RoPE, K/V into pages, attention over pos + n keys, * sigmoid(gate).
    // Linear layers: ws.qkv/z/b/a -> conv (its state updated in the row's state span), gated delta rule (state
    // updated in place), gated RMSNorm -> ws.o [B, S, Hv * Dv]; ws.qkv is overwritten.
    // Rows may differ in pos and pages; B > 1 needs S == 1 (decode).
    void attn(int i, Workspace& ws);

    // gpu.py lm_head and sample, split at the GPU/CPU boundary. lm_head: logits of ws.x's B rows (decode, S == 1:
    // prefill never samples) -> ws.logits [B, vocab] bf16, and their argmax. sample, after that command buffer
    // completed: one id per row, greedy rows from the GPU argmax, the rest drawn on the CPU from ws.logits.
    void lm_head(Workspace& ws);
    std::vector<int> sample(const Workspace& ws, const std::vector<SampleParams>& params) const;

    // Decode only, no Python counterpart (Python never runs ahead): ws.tokens[b] = prev_ids[src[b]] for src[b] >= 0,
    // so a greedy row's next step is encoded before the CPU has seen its token. Encode it before dense(0).
    void feed_tokens(Workspace& ws, id<MTLBuffer> prev_ids, const std::vector<int>& src);


private:
    friend struct GpuOpsTest;  // tests and bench-ops reach the kernel helpers below (Metal types, rule 1)
    friend class AneOps;       // embed and pipeline() for the ANE path

    // The kernels dense() is made of. Encoded into the current command buffer.
    void embed(Buf tokens, Buf out, int rows);
    // out may be x; w_stride 0: one weight for all (mx.fast.rms_norm without weight).
    // ponytail: eps default = rms_norm_eps, 1e-6 for every Qwen3.5 (mlx-lm reads config.json); add it to ModelSpec
    // when a model differs.
    void rms_norm(Buf weight, Buf x, Buf out, int rows, int dim, float eps = 1e-6f, uint32_t w_stride = 1);
    void gated_rms_norm(Buf weight, Buf x, Buf z, Buf out, int rows, int dim);  // RMSNormGated(x, z); out may be x
    void linear(const QLinear& l, Buf x, Buf y, int M);
    void add(Buf x, Buf y, Buf out, size_t n);
    void swiglu(Buf gate, Buf x, Buf out, size_t n);
    void argmax(Buf x, Buf out, int rows, int V);  // bf16 [rows, V] -> int32 [rows]

    struct Layer {
        bool is_linear;
        int g = -1;  // index among full layers (its K/V in a unit)
        Buf input_layernorm, post_attention_layernorm;
        QLinear gate_proj, up_proj, down_proj, out;  // out: o_proj (full) or out_proj (linear)
        QLinear q_proj, k_proj, v_proj;              // full
        Buf q_norm, k_norm;
        QLinear in_proj_qkv, in_proj_z, in_proj_b, in_proj_a;  // linear
        Buf conv1d, A_log, dt_bias, gated_norm;
        // M = 1 (see merge): the projections that read xn as one matmul, weights concatenated along N. in_proj:
        // qkv | z | b | a (linear), qkv_proj: q | k | v (full), gate_up: gate | up. The parts above are views into it.
        QLinear in_proj, qkv_proj, gate_up;
    };
    std::vector<Layer> layers;
    void full_attn(const Layer& l, Workspace& ws);
    void linear_attn(int i, const Layer& l, Workspace& ws);
    QLinear embed_tokens;
    QLinear lm;  // language_model.lm_head, or embed_tokens when tied (Qwen3.5 <= 9B: embed_tokens.as_linear)
    Buf norm;

    void elementwise(id<MTLComputePipelineState> ps, Buf a, Buf b, Buf out, size_t n);
    id<MTLComputePipelineState> pipeline(NSString* name, MTLFunctionConstantValues* constants = nil);

    ModelSpec spec;
    Memory& mem;
    id<MTLDevice> device;
    id<MTLLibrary> library;
    // the current workspace's encoder and split-k scratch, for the helpers above (set by each public op)
    // One thread per GpuOps: these members are per call (the ANE lane's thread has its own copy, AneOps::gpu)
    void use(const Workspace& ws) {
        if (owner == std::thread::id()) owner = std::this_thread::get_id();
        assert(owner == std::this_thread::get_id() && "GpuOps used from two threads");
        enc = ws.enc, splitk = ws.splitk, decode = ws.S == 1;
    }
    std::thread::id owner;  // the first thread that encoded with this object
    id<MTLComputeCommandEncoder> enc;
    id<MTLBuffer> splitk;
    bool decode = false;  // ws.S == 1: linear() may pick qmm_t_decode (batched decode, by tolerance)

    id<MTLComputePipelineState> qmv_fast, qmv_wide[6], qmm_t[2], qmm_t_64[2], qmm_t_splitk[2], splitk_accum;
    // qmm_t in 64 x 64 tiles from this many rows (prefill); shorter tails keep MLX's 32 x 32 (a 64-row tile of a
    // <= 32-row tail is mostly padding). At 33..63 rows 64 x 64 is ~1% faster (27B, M 40 / 48 / 56: -2..-3% on
    // q / gate / in_proj_qkv, +2% on down_proj). Tests set it to compare the two paths.
    int tile64_rows = 33;
    id<MTLComputePipelineState> qmm_t_decode[2][2];  // [BM 8 / 16][split-k float partials / bf16 output]
    id<MTLComputePipelineState> embed_ps, rms_ps, rms_looped_ps, add_ps, swiglu_ps, precise_swiglu_ps, split_q_gate_ps;
    id<MTLComputePipelineState> rope_ps, write_kv_ps, sdpa_ps, sdpa_2pass_1[2], sdpa_2pass_2, sigmoid_gate_ps;
    id<MTLComputePipelineState> gather_kv_ps, causal_mask_ps, scale_ps, softmax_ps[2];
    id<MTLComputePipelineState> conv1d_silu_ps, conv_norm_qk_ps, scale_qk_ps, gdn_gates_ps, gdn_gates_bf16_ps, gated_delta_ps, argmax_ps, feed_ps;
    // the fallback's matmul variants: gemv_wide [nv 4/5][k_lanes 16/32]; steel [nt/nn][align M][align N][align K]
    id<MTLComputePipelineState> gemv_wide_ps[3][2], steel_ps[2][2][2][2];
    id<MTLComputePipelineState> pipeline(NSString* name, std::initializer_list<std::pair<int, bool>> constants);
    // MLX's bf16 matmul for the attention fallback over the 8 (kv head, repeat) batches: steel_gemm_fused
    // (transpose_b: nt) or gemv_wide (nt, M 2..15). a_bs / b_bs: batch strides over [Hkv, gqa].
    void matmul(bool transpose_b, Buf a, Buf b, Buf d, int M, int N, int K, int lda, int ldb, int ldd,
                const int64_t a_bs[2], const int64_t b_bs[2], int64_t d_bs);

    id<MTLBuffer> one;  // bf16 1, read only

    // Before every dispatch. The decode lane's encoder is concurrent: a buffer-scope barrier (it covers the untracked
    // arena too) orders each dispatch after the ones before it, unless the call site set indep: the next dispatch
    // neither reads what the dispatches since the last barrier write, nor writes what they read or write (scratch
    // such as ws.splitk counts: linear() clears indep for split-k). Serial encoders ignore barriers.
    bool indep = false;
    void pre() {
        dispatches++;
        if (!indep) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        indep = false;
    }

public:
    // dense(i) at M = 1 runs each Layer's merged matmuls: one output row [part | part ...] in ws.qkv (in_proj),
    // ws.qg (qkv_proj) or ws.g (gate_up), and attn(i) reads z / b / a / v as slices of it. qmv_fast computes each
    // output from its own weight row the same way for any N, so the values equal the separate matmuls'. False in
    // AneOps' copy (the ANE writes each workspace field) and in tests that load the fields themselves.
    bool merge = true;
    size_t dispatches = 0;  // kernels encoded so far (bench-ops reports it per decode step)
};

}  // namespace mllm
