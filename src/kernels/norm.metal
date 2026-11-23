// RMSNorm from MLX v0.32.4, verbatim (src/vendor/mlx). GpuOps uses "rmsbfloat16" (rows <= 4096 wide) and "rms_loopedbfloat16".
#include "mlx/backend/metal/kernels/rms_norm.metal"
