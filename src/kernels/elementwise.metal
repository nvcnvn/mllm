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
// dt_bias))), softplus = logaddexp(x, 0) in bf16. Grid (Hv, rows). A_log: F32 (0.8B) or BF16 (27B; astype(float32)
// is exact).
template <typename AT>
[[kernel]] void gdn_gates(const device bfloat16_t* a [[buffer(0)]], const device bfloat16_t* b [[buffer(1)]],
                          const device AT* A_log [[buffer(2)]], const device bfloat16_t* dt_bias [[buffer(3)]],
                          device float* g [[buffer(4)]], device bfloat16_t* beta [[buffer(5)]],
                          uint2 i [[thread_position_in_grid]], uint2 grid [[threads_per_grid]]) {
  ulong j = ulong(i.y) * grid.x + i.x;
  beta[j] = Sigmoid{}(b[j]);
  float e = Negative{}(Exp{}(static_cast<float>(A_log[i.x])));
  bfloat16_t sp = LogAddExp{}(Add{}(a[j], dt_bias[i.x]), static_cast<bfloat16_t>(0));
  g[j] = Exp{}(Multiply{}(e, static_cast<float>(sp)));
}
template [[host_name("gdn_gates_float")]] [[kernel]] decltype(gdn_gates<float>) gdn_gates<float>;
template [[host_name("gdn_gates_bfloat16")]] [[kernel]] decltype(gdn_gates<bfloat16_t>) gdn_gates<bfloat16_t>;

// --- fused kernels: each does the same floating-point operations in the same order as the kernels it replaces ---

// MLX rms_single_row's reduction (rms_norm.metal, v0.32.4), verbatim: each thread's sum of squares over its N_READS
// values -> rsqrt(mean + eps), broadcast to the threadgroup. local_sums [32] / local_inv_mean [1]: the kernel's.
METAL_FUNC float rms_inv_mean(float acc, uint axis_size, float eps, uint simd_lane_id, uint simd_group_id,
                              threadgroup float* local_sums, threadgroup float* local_inv_mean) {
  acc = simd_sum(acc);
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return local_inv_mean[0];
}

// Fuses, for decode (S = 1): conv1d_silu<K>, then rms_norm (no weight, rows of 128) of the q / k part into nqkv,
// then scale_qk. v rows only get the conv (the unfused path normed them into unused scratch). Threadgroups
// (C / 128, B) of 128 threads: thread t convolves channel t; then threads 0..31 reduce 4 values each as MLX's
// rms_single_row (32 threads x 4 reads) does; the other simdgroups add zeros, as its unused local_sums do.
template <int K>
[[kernel]] void conv_norm_qk(
    device bfloat16_t* qkv [[buffer(0)]],
    const device bfloat16_t* w [[buffer(1)]],  // [C, K]
    device uchar* arena [[buffer(2)]],
    const device uint* table [[buffer(3)]],
    const constant ulong& unit [[buffer(4)]],
    const constant int& n_state [[buffer(5)]],
    const constant ulong& conv_off [[buffer(6)]],
    device bfloat16_t* nqkv [[buffer(7)]],
    const constant float& eps [[buffer(8)]],
    const constant float& q_scale [[buffer(9)]],
    const constant float& k_scale [[buffer(10)]],
    const constant int& key_dim [[buffer(11)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tgs [[threadgroups_per_grid]],
    uint2 lid2 [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  constexpr int N_READS = 4, D = 128;
  threadgroup float local_inv_mean[1];
  threadgroup float local_sums[32];
  threadgroup float conv_out[D];
  Table tb{table};
  const int b = tg.y, C = tgs.x * D, t = lid2.x, c = tg.x * D + t;
  if (!tb.n(b)) return;
  // conv1d_silu, n = 1
  device bfloat16_t* st[K - 1];
  bfloat16_t win[K];
  for (int r = 0; r < K - 1; r++) {
    st[r] = (device bfloat16_t*)(arena + tb.state_byte(b, n_state, unit, conv_off + (ulong(r) * C + c) * 2));
    win[r] = *st[r];
  }
  device bfloat16_t* x = qkv + ulong(b) * C + c;
  win[K - 1] = *x;
  float a = 0.0;
  for (int k = 0; k < K; ++k) a += static_cast<float>(win[k]) * w[c * K + k];
  bfloat16_t y = static_cast<bfloat16_t>(a);
  bfloat16_t out = Multiply{}(y, Sigmoid{}(y));
  *x = out;
  for (int k = 0; k < K - 1; k++) win[k] = win[k + 1];
  for (int r = 0; r < K - 1; r++) *st[r] = win[r];
  if (int(tg.x) * D >= 2 * key_dim) return;  // v: conv only
  conv_out[t] = out;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc = 0;  // rms_single_row: thread_x[i] = x[i]; acc += thread_x[i] * thread_x[i]
  if (t < D / N_READS)
    for (int i = 0; i < N_READS; i++) {
      float thread_x = conv_out[t * N_READS + i];
      acc += thread_x * thread_x;
    }
  const float inv = rms_inv_mean(acc, D, eps, simd_lane_id, simd_group_id, local_sums, local_inv_mean);
  const bfloat16_t one = static_cast<bfloat16_t>(1.0f), scale = static_cast<bfloat16_t>(c < key_dim ? q_scale : k_scale);
  bfloat16_t normed = one * static_cast<bfloat16_t>(conv_out[t] * inv);  // rms_norm's out, w = 1
  nqkv[ulong(b) * C + c] = Multiply{}(scale, normed);                    // scale_qk
}
template [[host_name("conv_norm_qk_4")]] [[kernel]] decltype(conv_norm_qk<4>) conv_norm_qk<4>;

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
