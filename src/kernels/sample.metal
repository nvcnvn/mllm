// Token choice per logits row, one threadgroup per row. Greedy (temp 0): mx.argmax, ties to the lowest index (MLX
// ArgMax). Sampled: mlx-lm make_sampler(temp, top_p): apply_top_p on the temperature-1 probabilities (keep every
// token >= the smallest value whose larger-or-equal tokens hold >= top_p of the mass), then categorical(logits / temp)
// by Gumbel-max, as mx.random.categorical; its uniforms come from a hash of the row's seed and the token index.
#include "mlx/backend/metal/kernels/utils.h"

constant constexpr int TG = 1024;
constant constexpr uint CAP = 4096;  // top_p candidates kept in threadgroup memory (24 KiB)

struct SampleRow {  // gpu.mm SampleRow
  float temp, top_p;
  ulong seed;
};

static inline float bf16(ushort b) { return as_type<float>(uint(b) << 16); }
// bf16 bits -> a key ordered as the values (-NaN < -inf < ... < +inf < +NaN)
static inline uint key(ushort b) { return b & 0x8000 ? uint(ushort(~b)) : uint(b) | 0x8000; }

// uniform in (0, 1) from (seed, i): splitmix64's finalizer, 23 bits (with 24, the top value rounds to 1.0f and its
// Gumbel noise is +inf: any token, however unlikely, would win)
static inline float unit_rand(ulong seed, uint i) {
  ulong z = seed + (ulong(i) + 1) * 0x9E3779B97F4A7C15ul;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
  z ^= z >> 31;
  return (float(uint(z >> 41)) + 0.5f) / 8388608.0f;
}

// threadgroup sum, the same value in every thread
static inline float tg_sum(float v, threadgroup float* buf, uint lane, uint simd) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);  // buf's last readers are done
  if (lane == 0) buf[simd] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return simd_sum(buf[lane]);  // TG / 32 == 32 simdgroups: one per lane
}

// (value, index) max over the threadgroup, ties to the lower index; the result in every thread
static inline void tg_argmax(thread float& best, thread int& bi, threadgroup float* vals, threadgroup int* idxs,
                             uint lane, uint simd) {
  auto merge = [&](float v, int i) {
    if (v > best || (v == best && i < bi)) best = v, bi = i;
  };
  for (ushort off = 16; off; off /= 2) merge(simd_shuffle_down(best, off), simd_shuffle_down(bi, off));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) vals[simd] = best, idxs[simd] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  best = vals[lane], bi = idxs[lane];
  for (ushort off = 16; off; off /= 2) merge(simd_shuffle_down(best, off), simd_shuffle_down(bi, off));
  best = simd_broadcast(best, 0), bi = simd_broadcast(bi, 0);
}

// This thread's share of the mass of tokens with key >= k (exp(v - m) each): over the n candidates ...
static inline float candidate_mass(threadgroup const ushort* cand_b, uint n, uint k, float m, uint lid) {
  float s = 0;
  for (uint j = lid; j < n; j += TG)
    if (key(cand_b[j]) >= k) s += exp(bf16(cand_b[j]) - m);
  return s;
}

// ... or over the whole row, skipping NaNs.
static inline float row_mass(const device ushort* x, int V, uint k, float m, uint lid) {
  float s = 0;
  for (int i = lid; i < V; i += TG) {
    const ushort b = x[i];
    const float v = bf16(b);
    if (key(b) >= k && !isnan(v)) s += exp(v - m);
  }
  return s;
}

[[kernel]] void sample_bf16(
    const device ushort* x [[buffer(0)]],
    device int* ids [[buffer(1)]],
    const constant int& V [[buffer(2)]],
    const constant SampleRow* rows [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup float vals[TG / 32];
  threadgroup int idxs[TG / 32];
  x += ulong(row) * V;
  float m = -INFINITY;
  int bi = 0;
  for (int i = lid; i < V; i += TG) {  // ascending: strict > keeps the lowest index
    float v = bf16(x[i]);
    if (v > m) m = v, bi = i;  // differs from mx.argmax on NaN (MLX returns a NaN's index; never here)
  }
  tg_argmax(m, bi, vals, idxs, lane, simd);
  const SampleRow p = rows[row];
  if (p.temp == 0 || !isfinite(m)) {  // greedy; or no finite logit: garbage in, the argmax out (all -inf: 0)
    if (lid == 0) ids[row] = bi;
    return;
  }

  float best = -INFINITY;  // Gumbel-max over the kept tokens: (v - m) / temp + Gumbel noise
  int pick = bi;
  const float inv = 1.0f / p.temp;
  auto draw = [&](int i, float v) {
    const float s = (v - m) * inv - log(-log(unit_rand(p.seed, i)));
    if (s > best) best = s, pick = i;
  };
  if (!(p.top_p > 0 && p.top_p < 1)) {  // no top_p: every finite token
    for (int i = lid; i < V; i += TG) {
      const float v = bf16(x[i]);
      if (isfinite(v)) draw(i, v);
    }
  } else {
    // Candidates: tokens within cut of the max. Each token below holds < exp(-cut) = (1 - top_p) / V of the max
    // token's mass, so all of them < (1 - top_p) of the total: the top_p cutoff is a candidate's key, and only
    // candidates are kept. So one pass over the row, then the search and the draw over the candidates alone
    // (unless there are more than CAP: then over the row, <= 16 more passes).
    const float cut = log(float(V) / (1 - p.top_p));
    threadgroup ushort cand_b[CAP];
    threadgroup int cand_i[CAP];
    threadgroup atomic_uint n_cand;
    if (lid == 0) atomic_store_explicit(&n_cand, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (int i = lid; i < V; i += TG) {
      const ushort b = x[i];
      const float v = bf16(b);
      if (isnan(v)) continue;
      total += exp(v - m);
      if (v >= m - cut) {
        const uint k = atomic_fetch_add_explicit(&n_cand, 1, memory_order_relaxed);
        if (k < CAP) cand_b[k] = b, cand_i[k] = i;
      }
    }
    const float target = p.top_p * tg_sum(total, vals, lane, simd);  // its barriers publish the candidates
    const uint n = atomic_load_explicit(&n_cand, memory_order_relaxed);
    const bool few = n <= CAP;
    // the mass of tokens with key >= k (exp(v - m) each), over the candidates or the row
    auto mass = [&](uint k) {
      const float s = few ? candidate_mass(cand_b, n, k, m, lid) : row_mass(x, V, k, m, lid);
      return tg_sum(s, vals, lane, simd);
    };
    // the largest key whose larger-or-equal tokens hold >= target: bisection over the 16-bit keys
    uint lo = 0, hi = key(as_type<uint>(m) >> 16);
    while (lo < hi) {
      const uint mid = (lo + hi + 1) / 2;
      if (mass(mid) >= target) lo = mid;
      else hi = mid - 1;
    }
    if (few) {
      for (uint j = lid; j < n; j += TG)
        if (key(cand_b[j]) >= lo) draw(cand_i[j], bf16(cand_b[j]));
    } else {
      for (int i = lid; i < V; i += TG) {
        const ushort b = x[i];
        const float v = bf16(b);
        if (key(b) >= lo && isfinite(v)) draw(i, v);
      }
    }
  }
  tg_argmax(best, pick, vals, idxs, lane, simd);
  if (lid == 0) ids[row] = pick;
}
