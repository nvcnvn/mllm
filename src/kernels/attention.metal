// Full-attention kernels that read the arena: K/V write into pages, and paged copies of MLX v0.32.4's sdpa_vector /
// sdpa_vector_2pass_1 (src/vendor/mlx/.../sdpa_vector.h). The paged copies change only (a) where key i comes from:
// the row's page slots[i / T] at kv_offset(g, kv, head, i % T) (memory.h), (b) per-row N = pos + n and q length n
// from the step table, rows outside [min_N, max_N) or past n return early, (c) q is read from our [B, S, H, D].
// The arithmetic is MLX's, statement for statement, so outputs match mx.fast.scaled_dot_product_attention bit for bit.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sdpa_vector.h"

#include "table.h"

// The arena's K/V layout for one full layer (memory.h kv_offset), in elements.
struct KVLayout {
  ulong unit;  // elements per unit
  int T, n_kv_heads, g;
};

template <int D>
inline ulong kv_index(constant KVLayout& L, uint slot, int kv, int head, int tok) {
  return slot * L.unit + ulong(((L.g * 2 + kv) * L.n_kv_heads + head) * L.T + tok) * D;
}

// k, v [B, S, Hkv, D] -> the rows' pages, tokens pos .. pos + n - 1. Grid (D, Hkv, B * S).
template <typename T, int D>
[[kernel]] void write_kv(
    const device T* k [[buffer(0)]],
    const device T* v [[buffer(1)]],
    device T* arena [[buffer(2)]],
    const device uint* table [[buffer(3)]],
    constant KVLayout& L [[buffer(4)]],
    const constant int& S [[buffer(5)]],
    uint3 i [[thread_position_in_grid]]) {
  Table tb{table};
  int b = i.z / S, s = i.z % S;
  if (s >= tb.n(b)) return;
  int t = tb.pos(b) + s;
  uint slot = tb.slot(b, t / L.T);
  ulong src = (ulong(i.z) * L.n_kv_heads + i.y) * D + i.x;
  arena[kv_index<D>(L, slot, 0, i.y, t % L.T) + i.x] = k[src];
  arena[kv_index<D>(L, slot, 1, i.y, t % L.T) + i.x] = v[src];
}

// sdpa_vector, no mask / sinks; causal always applied (a no-op at q length 1, as MLX's do_causal = q_len > 1).
// Grid (B * H, S), threadgroup 1024. out [B, H, S, D].
template <typename T, int D>
[[kernel]] void sdpa_vector_paged(
    const device T* queries [[buffer(0)]],
    const device T* arena [[buffer(1)]],
    device T* out [[buffer(3)]],
    const constant int& gqa_factor [[buffer(4)]],
    const device uint* table [[buffer(5)]],
    constant KVLayout& L [[buffer(6)]],
    const constant int& min_N [[buffer(7)]],
    const constant int& max_N [[buffer(8)]],
    const constant float& scale [[buffer(10)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int num_q_heads = L.n_kv_heads * gqa_factor;
  const int batch_idx = q_batch_head_idx / num_q_heads;
  const int q_head_idx = q_batch_head_idx % num_q_heads;
  const int kv_head_idx = q_head_idx / gqa_factor;
  Table tb{table};
  const int q_len = tb.n(batch_idx);
  const int N = tb.pos(batch_idx) + q_len;
  if (q_seq_idx >= q_len || N < min_N || N >= max_N) return;  // uniform per threadgroup
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  queries += ((batch_idx * int(tpg.y) + q_seq_idx) * num_q_heads + q_head_idx) * D +
      simd_lid * qk_per_thread;
  out += o_offset * D + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = i <= (N - q_len + q_seq_idx);
    if (use_key) {
      uint slot = tb.slot(batch_idx, i / L.T);
      const device T* keys = arena + kv_index<D>(L, slot, 0, kv_head_idx, i % L.T) + simd_lid * qk_per_thread;
      const device T* values = arena + kv_index<D>(L, slot, 1, kv_head_idx, i % L.T) + simd_lid * v_per_thread;
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}

// This lane's share of q . k (k prefetched into registers), summed left to right; simd_sum finishes it.
template <int N, typename T>
static inline float partial_qk(thread const float* q, thread const T* k) {
  float score = 0;
  for (int j = 0; j < N; j++) {
    score += q[j] * static_cast<float>(k[j]);
  }
  return score;
}

// o = o * factor + exp_score * v over this lane's share (v prefetched into registers).
template <int N, typename T>
static inline void rescale_add_v(thread float* o, float factor, float exp_score, thread const T* v) {
  for (int j = 0; j < N; j++) {
    o[j] = o[j] * factor + exp_score * static_cast<float>(v[j]);
  }
}

// sdpa_vector_2pass_1, no mask / sinks, causal always applied. blocks is MLX's function constant 26, passed here as
// n_blocks. Grid (Hkv, B, n_blocks), threadgroup (32, gqa, S). Partials [B, H, S, n_blocks, D]; sums, maxs fp32
// [B, H, S, n_blocks]. MLX's sdpa_vector_2pass_2 (verbatim) reduces them.
template <typename T, int D, int TK>
[[kernel]] void sdpa_vector_2pass_1_paged(
    const device T* queries [[buffer(0)]],
    const device T* arena [[buffer(1)]],
    device T* out [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    const device uint* table [[buffer(6)]],
    constant KVLayout& L [[buffer(7)]],
    const constant int& min_N [[buffer(8)]],
    const constant int& max_N [[buffer(9)]],
    const constant int& n_blocks [[buffer(11)]],
    const constant float& scale [[buffer(12)]],
    uint3 tptg [[threads_per_threadgroup]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = D / BD;
  const int blocks = n_blocks;

  typedef float U;

  thread U q[qk_per_thread];
  thread U o[v_per_thread] = {0};

  // Adjust positions
  const int kv_head_idx = tid.x;
  const int batch_idx = tid.y;
  const int block_idx = tid.z;
  const int gqa_factor = tptg.y;
  const int S = tptg.z;
  const int q_seq_idx = tidtg.z;
  const int q_head_idx = gqa_factor * kv_head_idx + tidtg.y;
  const int num_kv_heads = tpg.x;
  const int num_q_heads = num_kv_heads * gqa_factor;
  const int q_batch_head_idx = (batch_idx * num_q_heads + q_head_idx);
  const int o_offset = q_batch_head_idx * S + q_seq_idx;
  Table tb{table};
  const int q_seq_len = tb.n(batch_idx);
  const int N = tb.pos(batch_idx) + q_seq_len;
  if (q_seq_idx >= q_seq_len || N < min_N || N >= max_N) return;  // uniform per simdgroup; no barriers below

  queries += ((batch_idx * S + q_seq_idx) * num_q_heads + q_head_idx) * D + simd_lid * qk_per_thread;
  out += o_offset * blocks * D + block_idx * D + simd_lid * v_per_thread;
  sums += o_offset * blocks + block_idx;
  maxs += o_offset * blocks + block_idx;

  // Read the query
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;

  auto key = [&](int i, int kv, int per_thread) {
    return arena + kv_index<D>(L, tb.slot(batch_idx, i / L.T), kv, kv_head_idx, i % L.T) + simd_lid * per_thread;
  };

  // For each key
  int i = block_idx;
  for (; i + (TK - 1) * blocks < N; i += blocks * TK) {
    // Prefetch: issue all TK K/V loads first so the HW can overlap them.
    T k_reg[TK][qk_per_thread];
    T v_reg[TK][v_per_thread];
    for (int t = 0; t < TK; t++) {
      const device T* kp = key(i + t * blocks, 0, qk_per_thread);
      for (int j = 0; j < qk_per_thread; j++) {
        k_reg[t][j] = kp[j];
      }
    }
    for (int t = 0; t < TK; t++) {
      const device T* vp = key(i + t * blocks, 1, v_per_thread);
      for (int j = 0; j < v_per_thread; j++) {
        v_reg[t][j] = vp[j];
      }
    }

    for (int t = 0; t < TK; t++) {
      int ii = i + t * blocks;
      bool use_key = ii <= (N - q_seq_len + int(q_seq_idx));
      if (use_key) {
        U score = partial_qk<qk_per_thread>(q, k_reg[t]);
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        rescale_add_v<v_per_thread>(o, factor, exp_score, v_reg[t]);
      }
    }
  }

  // Tail (fewer than TK remaining keys)
  for (; i < N; i += blocks) {
    bool use_key = i <= (N - q_seq_len + int(q_seq_idx));
    if (use_key) {
      const device T* keys = key(i, 0, qk_per_thread);
      const device T* values = key(i, 1, v_per_thread);
      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * keys[j];
      }
      score = simd_sum(score);

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }
  }

  // Write the sum and max and outputs
  if (simd_lid == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = max_score;
  }

  for (int i = 0; i < v_per_thread; i++) {
    out[i] = static_cast<T>(o[i]);
  }
}

// ponytail (prefill): the steel GEMM loads K/V tiles from one base + ld, not per page, so the row's pages are copied
// into contiguous kc / vc [Hkv, kL, D] first. Ceiling: an extra read + write of the request's K/V per layer per
// chunk; a paged tile loader in the GEMM removes it. Row 0 only (prefill is B = 1). Grid (D, kL, Hkv).
template <typename T, int D>
[[kernel]] void gather_kv(
    const device T* arena [[buffer(0)]],
    device T* kc [[buffer(1)]],
    device T* vc [[buffer(2)]],
    const device uint* table [[buffer(3)]],
    constant KVLayout& L [[buffer(4)]],
    uint3 i [[thread_position_in_grid]],
    uint3 n [[threads_per_grid]]) {
  Table tb{table};
  uint slot = tb.slot(0, i.y / L.T);
  ulong dst = (ulong(i.z) * n.y + i.y) * D + i.x;
  kc[dst] = arena[kv_index<D>(L, slot, 0, i.z, i.y % L.T) + i.x];
  vc[dst] = arena[kv_index<D>(L, slot, 1, i.z, i.y % L.T) + i.x];
}

// MLX's fallback: where(q_idx >= k_idx, scores, finfo(bf16).min) with q_idx = pos + s (kL - qL = pos).
// scores [H, n, kL] in place, row 0's pos. Grid (kL, n, H).
[[kernel]] void causal_mask(
    device bfloat16_t* scores [[buffer(0)]],
    const device uint* table [[buffer(1)]],
    uint3 i [[thread_position_in_grid]],
    uint3 n [[threads_per_grid]]) {
  if (int(i.x) > Table{table}.pos(0) + int(i.y))
    scores[(ulong(i.z) * n.y + i.y) * n.x + i.x] = Limits<bfloat16_t>::finite_min;
}

// clang-format off
instantiate_kernel("write_kv_bfloat16_256", write_kv, bfloat16_t, 256)
instantiate_kernel("gather_kv_bfloat16_256", gather_kv, bfloat16_t, 256)
instantiate_kernel("sdpa_vector_paged_bfloat16_256", sdpa_vector_paged, bfloat16_t, 256)
instantiate_kernel("sdpa_vector_2pass_1_paged_bfloat16_256", sdpa_vector_2pass_1_paged, bfloat16_t, 256, 4)
instantiate_kernel("sdpa_vector_2pass_1_paged_bfloat16_256_tk1", sdpa_vector_2pass_1_paged, bfloat16_t, 256, 1)
instantiate_kernel("sdpa_vector_2pass_2_bfloat16_256", sdpa_vector_2pass_2, bfloat16_t, 256)
// clang-format on
