// GpuOps: dense(i), attn(i), lm_head_sample on Metal (py/mllm/gpu.py). Kernels are MLX's (src/vendor/mlx), dispatched
// the way MLX v0.32.4 dispatches them, so results match mlx-lm bit for bit.
#pragma once
#import <Metal/Metal.h>

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

class GpuOps {
public:
    // max_rows: the workspace's rows (max of chunk and batch); sizes GpuOps' own scratch buffers.
    GpuOps(Memory& mem, const Weights& weights, int max_rows);

    // All the big matmuls between attn(i - 1) and attn(i), i = 0 .. n_layers, over ws.B * ws.S rows.
    // Fields as in gpu.py, except q, k, v are row-major [B, S, H, D] (gpu.py's [B, H, S, D] transpose is only
    // for MLX's SDPA).
    void dense(int i, Workspace& ws);

    // ponytail: one command buffer per begin/commit_and_wait, waited on; g5 replaces this with lane-owned
    // command buffers and events. Both autorelease Metal objects: a long-running caller (g5's step loop) wraps
    // each step in @autoreleasepool.
    void begin();
    double commit_and_wait();  // GPU seconds the command buffer took

private:
    friend struct GpuOpsTest;  // tests and bench-ops reach the kernel helpers below (Metal types, rule 1)

    // The kernels dense() is made of. Encoded into the current command buffer.
    void embed(Buf tokens, Buf out, int rows);
    void rms_norm(Buf weight, Buf x, Buf out, int rows, int dim);  // out may be x
    void gated_rms_norm(Buf weight, Buf x, Buf z, Buf out, int rows, int dim);  // RMSNormGated(x, z); out may be x
    void linear(const QLinear& l, Buf x, Buf y, int M);
    void add(Buf x, Buf y, Buf out, size_t n);
    void swiglu(Buf gate, Buf x, Buf out, size_t n);

    struct Layer {
        bool is_linear;
        Buf input_layernorm, post_attention_layernorm;
        QLinear gate_proj, up_proj, down_proj, out;  // out: o_proj (full) or out_proj (linear)
        QLinear q_proj, k_proj, v_proj;              // full
        Buf q_norm, k_norm;
        QLinear in_proj_qkv, in_proj_z, in_proj_b, in_proj_a;  // linear
    };
    std::vector<Layer> layers;
    QLinear embed_tokens;  // its scales/biases also make it the tied lm_head (g4)
    Buf norm;

    void elementwise(id<MTLComputePipelineState> ps, Buf a, Buf b, Buf out, size_t n);
    id<MTLComputePipelineState> pipeline(NSString* name);

    ModelSpec spec;
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;

    id<MTLComputePipelineState> qmv_fast, qmv_wide[6], qmm_t[2], qmm_t_splitk[2], splitk_accum;
    id<MTLComputePipelineState> embed_ps, rms_ps, add_ps, swiglu_ps, precise_swiglu_ps, split_q_gate_ps;

    // scratch, [max_rows, ...] bf16: h (residual after attn), xn (normed), t (projection out), g / u (MLP), qg
    // (q_proj out); splitk: qmm_t_splitk's fp32 partial sums
    id<MTLBuffer> h, xn, t, g, u, qg, splitk;
    int max_rows;
};

}  // namespace mllm
