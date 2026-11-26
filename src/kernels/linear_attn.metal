// The gated delta rule, mlx-lm's packed kernel (mlx_lm/kernels/gated_delta.py _make_gated_delta_packed_kernel, the
// variant gated_delta_kernel picks for scalar gates, Dk 128, fp32 state, no mask), as mx.fast.metal_kernel generates
// it (MLX's utils.h + a template over InT, StT, Dk, Dv, Hk, Hv). Changed only: where things live. (a) The state
// [Hv, Dv, Dk] fp32 is updated in place in the arena, in the row's state span at delta_off (table.h); each lane's
// 32 floats are one 128-byte run, which the host keeps inside one unit. (b) q / k / v rows have stride ld (they
// are slices of the conv output); row b's tokens start at workspace row b * S, T = the row's n (0: skipped).
// The arithmetic is mlx-lm's statement for statement, so y and the state match it bit for bit.
#include "mlx/backend/metal/kernels/utils.h"

#include "table.h"

template <typename InT, typename StT, int Dk, int Dv, int Hk, int Hv>
[[kernel]] void gated_delta_paged(
    const device InT* q [[buffer(0)]],
    const device InT* k [[buffer(1)]],
    const device InT* v [[buffer(2)]],
    const device float* g [[buffer(3)]],
    const device InT* beta [[buffer(4)]],
    device uchar* arena [[buffer(5)]],
    device InT* y [[buffer(6)]],
    const device uint* table [[buffer(7)]],
    const constant int& S [[buffer(8)]],
    const constant int& ld [[buffer(9)]],
    const constant ulong& unit [[buffer(10)]],
    const constant int& n_state [[buffer(11)]],
    const constant ulong& delta_off [[buffer(12)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {
  constexpr int lanes_per_row = 4;
  constexpr int rows_per_simdgroup = 32 / lanes_per_row;
  constexpr int values_per_lane = Dk / lanes_per_row;
  constexpr int partials_per_lane = values_per_lane / 4;

  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  Table tb{table};
  const int T = tb.n(b_idx);
  if (!T) return;

  auto lane = thread_index_in_simdgroup;
  auto row_in_simdgroup = lane / lanes_per_row;
  auto lane_in_row = lane & (lanes_per_row - 1);
  auto row_group = thread_position_in_grid.y;
  auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

  // q, k, v: [B, S, *] with row stride ld, heads [Hk, Dk] / [Hv, Dv] at the row start
  auto q_ = q + ulong(b_idx) * S * ld + hk_idx * Dk + lane_in_row * values_per_lane;
  auto k_ = k + ulong(b_idx) * S * ld + hk_idx * Dk + lane_in_row * values_per_lane;
  auto v_ = v + ulong(b_idx) * S * ld + hv_idx * Dv;

  // y: [B, S, Hv, Dv]
  y += (ulong(b_idx) * S * Hv + hv_idx) * Dv;

  // state: [Hv, Dv, Dk] in the row's span
  ulong x = delta_off + ((ulong(hv_idx) * Dv + dv_idx) * Dk + lane_in_row * values_per_lane) * sizeof(StT);
  auto i_state = (device StT*)(arena + tb.state_byte(b_idx, n_state, unit, x));
  auto o_state = i_state;

  float state[values_per_lane];
  for (int i = 0; i < values_per_lane; ++i) {
    state[i] = static_cast<float>(i_state[i]);
  }

  // g, beta: [B, S, Hv]
  auto g_ = g + ulong(b_idx) * S * Hv;
  auto beta_ = beta + ulong(b_idx) * S * Hv;

  for (int t = 0; t < T; ++t) {
    float gt = static_cast<float>(g_[hv_idx]);

    // Partials mirror the generic kernel: each 4-element chain is one
    // original lane's sequential accumulation.
    float part[partials_per_lane];
    for (int pb = 0; pb < partials_per_lane; ++pb) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        int e = pb * 4 + i;
        state[e] = state[e] * gt;
        acc += state[e] * static_cast<float>(k_[e]);
      }
      part[pb] = acc;
    }
    // Butterfly levels xor 1,2,4 stay inside this lane (commutative
    // pairwise tree); levels xor 8,16 become the row-group shuffles.
    float kv_mem =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    kv_mem += simd_shuffle_xor(kv_mem, 1);
    kv_mem += simd_shuffle_xor(kv_mem, 2);

    auto delta =
        (static_cast<float>(v_[dv_idx]) - kv_mem) *
        static_cast<float>(beta_[hv_idx]);

    for (int pb = 0; pb < partials_per_lane; ++pb) {
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        int e = pb * 4 + i;
        state[e] = state[e] + static_cast<float>(k_[e]) * delta;
        acc += state[e] * static_cast<float>(q_[e]);
      }
      part[pb] = acc;
    }
    float out =
        ((part[0] + part[1]) + (part[2] + part[3])) +
        ((part[4] + part[5]) + (part[6] + part[7]));
    out += simd_shuffle_xor(out, 1);
    out += simd_shuffle_xor(out, 2);
    if (lane_in_row == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }

    q_ += ld;
    k_ += ld;
    v_ += ld;
    y += Hv * Dv;
    g_ += Hv;
    beta_ += Hv;
  }

  for (int i = 0; i < values_per_lane; ++i) {
    o_state[i] = static_cast<StT>(state[i]);
  }
}

// ponytail: Qwen3.5-0.8B's heads (Hk = Hv = 16), Qwen3.5-4B / 9B's (Hk 16, Hv 32) and Qwen3.6-27B's (Hk 16, Hv 48),
// Dk = Dv = 128; instantiate others when a model needs them
template [[host_name("gated_delta_paged_bfloat16_128_128_16_16")]] [[kernel]]
decltype(gated_delta_paged<bfloat16_t, float, 128, 128, 16, 16>) gated_delta_paged<bfloat16_t, float, 128, 128, 16, 16>;
template [[host_name("gated_delta_paged_bfloat16_128_128_16_32")]] [[kernel]]
decltype(gated_delta_paged<bfloat16_t, float, 128, 128, 16, 32>) gated_delta_paged<bfloat16_t, float, 128, 128, 16, 32>;
template [[host_name("gated_delta_paged_bfloat16_128_128_16_48")]] [[kernel]]
decltype(gated_delta_paged<bfloat16_t, float, 128, 128, 16, 48>) gated_delta_paged<bfloat16_t, float, 128, 128, 16, 48>;
