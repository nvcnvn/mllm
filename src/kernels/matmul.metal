// bf16 matmuls from MLX v0.32.4 (src/vendor/mlx), only the variants the unfused attention (MLX's SDPA fallback)
// dispatches on an 's' GPU. Names match MLX's, so gpu.mm's dispatch reads like mlx/backend/metal/matmul.cpp.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"
#include "mlx/backend/metal/kernels/gemv.h"

// scores = q @ kᵀ (nt) and out = p @ v (nn); GEMM_TPARAM_MACRO's medium-device tiles
instantiate_kernel("steel_gemm_fused_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2", gemm, bfloat16_t, 64, 64, 16, 2, 2, false, true, float)
instantiate_kernel("steel_gemm_fused_nn_bfloat16_bfloat16_bm64_bn64_bk16_wm2_wn2", gemm, bfloat16_t, 64, 64, 16, 2, 2, false, false, float)

// scores for 6 <= q_len <= 15 (gemv_wide_config: vecs_per_tg 3 to 5; 3 only at gqa 6, q_len 6)
instantiate_kernel("gemv_wide_bfloat16_nv3_kl16", gemv_wide, bfloat16_t, 3, 16)
instantiate_kernel("gemv_wide_bfloat16_nv3_kl32", gemv_wide, bfloat16_t, 3, 32)
instantiate_kernel("gemv_wide_bfloat16_nv4_kl16", gemv_wide, bfloat16_t, 4, 16)
instantiate_kernel("gemv_wide_bfloat16_nv5_kl16", gemv_wide, bfloat16_t, 5, 16)
instantiate_kernel("gemv_wide_bfloat16_nv4_kl32", gemv_wide, bfloat16_t, 4, 32)
instantiate_kernel("gemv_wide_bfloat16_nv5_kl32", gemv_wide, bfloat16_t, 5, 32)
// clang-format on
