// 4-bit affine (group 64, bf16) kernels from MLX v0.32.4 (src/vendor/mlx), only the variants GpuOps dispatches.
// Names match MLX's, so gpu.mm's dispatch reads like mlx/backend/metal/quantized.cpp.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"

// M == 1 (dispatch_qmv -> qmv, K % 512 == 0, N % 8 == 0)
instantiate_kernel("affine_qmv_fast_bfloat16_t_gs_64_b_4_batch_0_pr_0_hgs_0_r_4", affine_qmv_fast, bfloat16_t, 64, 4, false, false, false, 4)

// 2 <= M < batch limit (qmv_wide; vecs_per_tg = 2..5 for M up to 15)
#define instantiate_qmv_wide(nv) \
  instantiate_kernel("affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_" #nv "_kl_8_batch_0", affine_qmv_wide, bfloat16_t, 64, 4, nv, 8, false)
instantiate_qmv_wide(2)
instantiate_qmv_wide(3)
instantiate_qmv_wide(4)
instantiate_qmv_wide(5)

// M >= batch limit: qmm_t_splitk + accum, or qmm_t when split_k comes out as 1
instantiate_kernel("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0", affine_qmm_t, bfloat16_t, 64, 4, true, false)
instantiate_kernel("affine_qmm_t_bfloat16_t_gs_64_b_4_alN_false_batch_0", affine_qmm_t, bfloat16_t, 64, 4, false, false)
instantiate_kernel("affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_true", affine_qmm_t_splitk, bfloat16_t, 64, 4, true)
instantiate_kernel("affine_qmm_t_splitk_bfloat16_t_gs_64_b_4_alN_false", affine_qmm_t_splitk, bfloat16_t, 64, 4, false)
instantiate_kernel("steel_gemm_splitk_accum_bfloat16_float32", gemm_splitk_accum, float, bfloat16_t)
// clang-format on

// Batched decode (S = 1, 7 <= M <= 16): not MLX's, checked by tolerance (test_batch), see GpuOps::linear.
// MLX's qmm_t (steel BlockMMA, x and dequantized W tiles in threadgroup memory) with one M tile of BM = 8 or 16 rows
// and a 1 x 4 simdgroup grid, BK = 64 (one quantization group per K step), BN = 32. MLX fixes BM = 32 with 2 x 2
// simdgroups, so at M <= 16 half to 3/4 of its MMA work is padding (g12 profile). N % 32 == 0, one M tile (M <= BM).
// U = float: split-k partials (grid z = split, y = ws.splitk + z * stride, summed by steel_gemm_splitk_accum).
// U = bfloat16_t: grid z = 1, y is the output.
template <int BM, typename U>
[[kernel]] void qmm_t_decode(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device bfloat16_t* x [[buffer(3)]],
    device U* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& k_part [[buffer(8)]],
    const constant int& stride [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64, BN = 32, WM = 1, WN = 4, BK_padded = BK + 8, group_size = 64;
  using mma_t = mlx::steel::BlockMMA<bfloat16_t, U, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_x_t = mlx::steel::BlockLoader<bfloat16_t, BM, BK, BK_padded, 1, WM * WN * SIMD_SIZE>;
  using loader_w_t = QuantizedBlockLoader<bfloat16_t, BN, BK, BK_padded, 1, WM * WN * SIMD_SIZE, group_size, 4>;
  threadgroup bfloat16_t Xs[BM * BK_padded];
  threadgroup bfloat16_t Ws[BN * BK_padded];
  const int k0 = tid.z * k_part, n0 = tid.x * BN;
  const device uint8_t* wl = (const device uint8_t*)w + (size_t)n0 * (K / 2) + k0 / 2;
  scales += (size_t)n0 * (K / group_size) + k0 / group_size;
  biases += (size_t)n0 * (K / group_size) + k0 / group_size;
  y += tid.z * (size_t)stride + n0;
  loader_x_t loader_x(x + k0, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  const short rows = M;
  for (int k = 0; k < k_part; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rows < BM) loader_x.load_safe(short2(BK, rows));
    else loader_x.load_unsafe();
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_x.next();
    loader_w.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rows < BM) mma_op.store_result_safe(y, N, short2(BN, rows));
  else mma_op.store_result(y, N);
}
template [[host_name("qmm_t_decode_8_float")]] [[kernel]] decltype(qmm_t_decode<8, float>) qmm_t_decode<8, float>;
template [[host_name("qmm_t_decode_16_float")]] [[kernel]] decltype(qmm_t_decode<16, float>) qmm_t_decode<16, float>;
template [[host_name("qmm_t_decode_8_bf16")]] [[kernel]] decltype(qmm_t_decode<8, bfloat16_t>) qmm_t_decode<8, bfloat16_t>;
template [[host_name("qmm_t_decode_16_bf16")]] [[kernel]] decltype(qmm_t_decode<16, bfloat16_t>) qmm_t_decode<16, bfloat16_t>;

// QuantizedEmbedding: MLX gathers the rows (weight[ids], scales[ids], biases[ids]) then runs affine_dequantize;
// this fuses the gather and keeps affine_dequantize's arithmetic (bits == 4 branch). Grid: (K / 8, rows).
[[kernel]] void embed(
    const device uint32_t* w [[buffer(0)]],
    const device bfloat16_t* scales [[buffer(1)]],
    const device bfloat16_t* biases [[buffer(2)]],
    const device int32_t* tokens [[buffer(3)]],
    device bfloat16_t* out [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]) {
  constexpr int group_size = 64, pack_factor = 8;
  size_t row = tokens[index.y];
  size_t offset = row * (K / pack_factor) + index.x;  // the uint32 holding 8 values
  size_t gindex = row * (K / group_size) + index.x * pack_factor / group_size;
  bfloat16_t scale = scales[gindex];
  bfloat16_t bias = biases[gindex];
  out += size_t(index.y) * K + index.x * pack_factor;
  uint val = w[offset];
#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    uint8_t d = (val >> (4 * i)) & 0x0f;
    out[i] = scale * d + bias;
  }
}
