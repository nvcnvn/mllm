// bf16 matmuls from MLX v0.32.4 (src/vendor/mlx), only the variants the unfused attention (MLX's SDPA fallback)
// dispatches (gpu_rules.h gemm_tiles). Names match MLX's, so gpu.mm's dispatch reads like mlx/backend/metal/matmul.cpp.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"
#include "mlx/backend/metal/kernels/gemv.h"

// scores = q @ kᵀ (nt) and out = p @ v (nn); GEMM_TPARAM_MACRO's tiles for every device size
#define instantiate_gemm(tb, transpose_b, bm, bn, bk, wm, wn) \
  instantiate_kernel("steel_gemm_fused_n" #tb "_bfloat16_bfloat16_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn, \
                     gemm, bfloat16_t, bm, bn, bk, wm, wn, false, transpose_b, float)
instantiate_gemm(t, true, 64, 64, 16, 2, 2)   // medium ('s')
instantiate_gemm(n, false, 64, 64, 16, 2, 2)
instantiate_gemm(t, true, 64, 32, 32, 2, 2)   // small ('g'), large ('d') smaller or large K
instantiate_gemm(n, false, 64, 64, 16, 1, 2)  // small, large
instantiate_gemm(t, true, 64, 64, 16, 1, 2)   // large, reasonable K
instantiate_gemm(n, false, 32, 64, 16, 1, 2)  // large, large K

// scores for 6 <= q_len <= 15 (gemv_wide_config: vecs_per_tg 3 to 5; 3 only at gqa 6, q_len 6)
instantiate_kernel("gemv_wide_bfloat16_nv3_kl16", gemv_wide, bfloat16_t, 3, 16)
instantiate_kernel("gemv_wide_bfloat16_nv3_kl32", gemv_wide, bfloat16_t, 3, 32)
instantiate_kernel("gemv_wide_bfloat16_nv4_kl16", gemv_wide, bfloat16_t, 4, 16)
instantiate_kernel("gemv_wide_bfloat16_nv5_kl16", gemv_wide, bfloat16_t, 5, 16)
instantiate_kernel("gemv_wide_bfloat16_nv4_kl32", gemv_wide, bfloat16_t, 4, 32)
instantiate_kernel("gemv_wide_bfloat16_nv5_kl32", gemv_wide, bfloat16_t, 5, 32)
// clang-format on
