// Element-wise ops of dense(i). Each one repeats the MLX graph op by op (MLX's Sigmoid / Add / Multiply, an
// intermediate rounded to its array's dtype), so the results match mlx-lm bit for bit. One thread per element.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/unary_ops.h"

// x + y (residual adds)
[[kernel]] void add(const device bfloat16_t* x [[buffer(0)]], const device bfloat16_t* y [[buffer(1)]],
                    device bfloat16_t* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  out[i] = Add{}(x[i], y[i]);
}

// mlx_lm.layers.common.swiglu, compiled: silu(gate) * x with silu(g) = g * sigmoid(g), all bf16.
// out may alias gate.
[[kernel]] void swiglu(const device bfloat16_t* gate [[buffer(0)]], const device bfloat16_t* x [[buffer(1)]],
                       device bfloat16_t* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  bfloat16_t g = gate[i];
  bfloat16_t s = Sigmoid{}(g);
  bfloat16_t silu = Multiply{}(g, s);
  out[i] = Multiply{}(silu, x[i]);
}

// mlx_lm.layers.common.precise_swiglu(h, gate, x), compiled: (silu(float(gate)) * float(x)) -> bf16.
// The gated RMSNorm's second half: x = rms_norm(h), gate = z. out may alias x.
[[kernel]] void precise_swiglu(const device bfloat16_t* gate [[buffer(0)]], const device bfloat16_t* x [[buffer(1)]],
                               device bfloat16_t* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float g = static_cast<float>(gate[i]);
  float s = Sigmoid{}(g);
  float silu = Multiply{}(g, s);
  out[i] = static_cast<bfloat16_t>(Multiply{}(silu, static_cast<float>(x[i])));
}

// q_proj output [rows, H, 2D] -> q [rows, H, D] and gate [rows, H * D] (gpu.py: reshape [B, S, H, 2D], split
// the last axis in 2). Grid: (D, rows * H).
[[kernel]] void split_q_gate(const device bfloat16_t* qg [[buffer(0)]], device bfloat16_t* q [[buffer(1)]],
                             device bfloat16_t* gate [[buffer(2)]], uint2 i [[thread_position_in_grid]],
                             uint2 n [[threads_per_grid]]) {
  size_t src = size_t(i.y) * 2 * n.x + i.x, dst = size_t(i.y) * n.x + i.x;
  q[dst] = qg[src];
  gate[dst] = qg[src + n.x];
}

// gpu.py: o.transpose(0, 2, 1, 3).reshape(B, S, H * D) * mx.sigmoid(gate), each op rounded to bf16 as MLX does.
// ao [B, H, S, D] -> o [B, S, H * D]. Grid (D, H, B * S).
[[kernel]] void sigmoid_gate(
    const device bfloat16_t* ao [[buffer(0)]],
    const device bfloat16_t* gate [[buffer(1)]],
    device bfloat16_t* o [[buffer(2)]],
    const constant int& S [[buffer(3)]],
    uint3 i [[thread_position_in_grid]],
    uint3 n [[threads_per_grid]]) {
  ulong b = i.z / S, s = i.z % S;
  ulong src = ((b * n.y + i.y) * S + s) * n.x + i.x, dst = (ulong(i.z) * n.y + i.y) * n.x + i.x;
  bfloat16_t g = Sigmoid{}(gate[dst]);
  o[dst] = Multiply{}(ao[src], g);
}

// mlx fast.cpp SDPA fallback: q = multiply(array(scale, bf16), q). In place.
[[kernel]] void scale_bf16(device bfloat16_t* x [[buffer(0)]], const constant float& scale [[buffer(1)]],
                           uint i [[thread_position_in_grid]]) {
  x[i] = Multiply{}(static_cast<bfloat16_t>(scale), x[i]);
}
