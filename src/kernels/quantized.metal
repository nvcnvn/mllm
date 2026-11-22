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
