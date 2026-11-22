// Greedy sampling: mx.argmax over each logits row, ties to the lowest index (MLX ArgMax). One threadgroup per row.
#include "mlx/backend/metal/kernels/utils.h"

constant constexpr int TG = 1024;

[[kernel]] void argmax_bf16(
    const device bfloat16_t* x [[buffer(0)]],
    device int* ids [[buffer(1)]],
    const constant int& V [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  x += ulong(row) * V;
  float best = -INFINITY;
  int bi = 0;
  for (int i = lid; i < V; i += TG) {  // ascending: strict > keeps the lowest index
    float v = x[i];
    if (v > best) best = v, bi = i;  // differs from mx.argmax on NaN (MLX returns a NaN's index; never here)
  }
  auto merge = [&](float v, int i) {
    if (v > best || (v == best && i < bi)) best = v, bi = i;
  };
  for (ushort off = 16; off; off /= 2) merge(simd_shuffle_down(best, off), simd_shuffle_down(bi, off));
  threadgroup float vals[TG / 32];
  threadgroup int idxs[TG / 32];
  if (lane == 0) vals[simd] = best, idxs[simd] = bi;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd) return;
  best = vals[lane], bi = idxs[lane];
  for (ushort off = 16; off; off /= 2) merge(simd_shuffle_down(best, off), simd_shuffle_down(bi, off));
  if (lane == 0) ids[row] = bi;
}
