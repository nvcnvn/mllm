// Element-wise ops of dense(i). Each one repeats the MLX graph op by op (MLX's Sigmoid / Add / Multiply, an
// intermediate rounded to its array's dtype), so the results match mlx-lm bit for bit. One thread per element.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/unary_ops.h"

#include "table.h"

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

// --- linear attention (gpu.py _linear); the MLX-generated gated delta kernel is in linear_attn.metal ---

// conv_in = concat(conv state [K - 1, C], the row's n qkv rows); conv state = conv_in's last K - 1 rows;
// out = silu(depthwise conv1d(conv_in)): MLX's depthwise_conv_1d (float acc over the taps in order) then the compiled
// nn.silu (x * sigmoid(x), bf16). In place: qkv row t becomes out row t. The conv state sits in the row's state
// span at conv_off. One thread per (channel, row), grid (C, B).
template <int K>
[[kernel]] void conv1d_silu(
    device bfloat16_t* qkv [[buffer(0)]],
    const device bfloat16_t* w [[buffer(1)]],  // [C, K]
    device uchar* arena [[buffer(2)]],
    const device uint* table [[buffer(3)]],
    const constant ulong& unit [[buffer(4)]],
    const constant int& n_state [[buffer(5)]],
    const constant ulong& conv_off [[buffer(6)]],
    const constant int& S [[buffer(7)]],
    uint2 i [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  Table tb{table};
  const int b = i.y, n = tb.n(b), C = grid.x;
  if (!n) return;
  device bfloat16_t* st[K - 1];
  bfloat16_t win[K];
  for (int r = 0; r < K - 1; r++) {
    st[r] = (device bfloat16_t*)(arena + tb.state_byte(b, n_state, unit, conv_off + (ulong(r) * C + i.x) * 2));
    win[r] = *st[r];
  }
  device bfloat16_t* x = qkv + ulong(b) * S * C + i.x;
  w += i.x * K;
  for (int t = 0; t < n; t++, x += C) {
    win[K - 1] = *x;
    float acc = 0.0;
    for (int k = 0; k < K; ++k) acc += static_cast<float>(win[k]) * w[k];
    bfloat16_t y = static_cast<bfloat16_t>(acc);
    *x = Multiply{}(y, Sigmoid{}(y));
    for (int k = 0; k < K - 1; k++) win[k] = win[k + 1];
  }
  for (int r = 0; r < K - 1; r++) *st[r] = win[r];
}
template [[host_name("conv1d_silu_4")]] [[kernel]] decltype(conv1d_silu<4>) conv1d_silu<4>;

// normalize_qk's scales after rms_norm: q * bf16(inv_scale ** 2), k * bf16(inv_scale), in place. x rows hold
// [q | k | ...] with stride ld; grid (2 * key_dim, rows).
[[kernel]] void scale_qk(device bfloat16_t* x [[buffer(0)]], const constant float& q_scale [[buffer(1)]],
                         const constant float& k_scale [[buffer(2)]], const constant int& ld [[buffer(3)]],
                         uint2 i [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  device bfloat16_t* p = x + ulong(i.y) * ld + i.x;
  *p = Multiply{}(static_cast<bfloat16_t>(i.x < grid.x / 2 ? q_scale : k_scale), *p);
}

// gated_delta_update's gates: beta = sigmoid(b) (bf16); g = compute_g, compiled: exp(-exp(A_log) * float(softplus(a +
// dt_bias))), softplus = logaddexp(x, 0) in bf16. Grid (Hv, rows).
[[kernel]] void gdn_gates(const device bfloat16_t* a [[buffer(0)]], const device bfloat16_t* b [[buffer(1)]],
                          const device float* A_log [[buffer(2)]], const device bfloat16_t* dt_bias [[buffer(3)]],
                          device float* g [[buffer(4)]], device bfloat16_t* beta [[buffer(5)]],
                          uint2 i [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  ulong j = ulong(i.y) * grid.x + i.x;
  beta[j] = Sigmoid{}(b[j]);
  float e = Negative{}(Exp{}(A_log[i.x]));
  bfloat16_t sp = LogAddExp{}(Add{}(a[j], dt_bias[i.x]), static_cast<bfloat16_t>(0));
  g[j] = Exp{}(Multiply{}(e, static_cast<float>(sp)));
}

// Decode token feedback: row b's input token is the previous step's argmax of row src[b] (src[b] < 0: the CPU wrote it).
[[kernel]] void feed_tokens(const device int* ids [[buffer(0)]], device int* tokens [[buffer(1)]],
                            const constant int* src [[buffer(2)]], uint b [[thread_position_in_grid]]) {
    if (src[b] >= 0) tokens[b] = ids[src[b]];
}

// ANE output, fp16 channels-first [C, row] (an IOSurface; columns past n are padding) -> workspace bf16 [n, C],
// rounded to nearest even like mx.astype(bfloat16). Grid: (C, n).
[[kernel]] void from_ane(const device half* src [[buffer(0)]], device bfloat16_t* dst [[buffer(1)]],
                         constant uint& row [[buffer(2)]], uint2 i [[thread_position_in_grid]],
                         uint2 n [[threads_per_grid]]) {
  dst[size_t(i.y) * n.x + i.x] = static_cast<bfloat16_t>(static_cast<float>(src[size_t(i.x) * row + i.y]));
}

// Workspace bf16 [n, C] -> ANE input, fp16 channels-first [C, row], zero in columns n .. S. Grid: (C, S).
[[kernel]] void to_ane(const device bfloat16_t* src [[buffer(0)]], device half* dst [[buffer(1)]],
                       constant uint& row [[buffer(2)]], constant uint& n [[buffer(3)]],
                       uint2 i [[thread_position_in_grid]], uint2 g [[threads_per_grid]]) {
  dst[size_t(i.x) * row + i.y] = i.y < n ? static_cast<half>(static_cast<float>(src[size_t(i.y) * g.x + i.x])) : 0.0h;
}
