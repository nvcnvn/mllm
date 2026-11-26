// MLX v0.32.4's per-GPU dispatch rules, as pure functions of the GPU class and the shapes (tests/test_gpu_rules.cpp
// checks them in CI). Each names its MLX source; branch for branch as MLX, so every chip MLX runs picks MLX's
// kernels and results stay bit-exact with mlx-lm there. Only the branches GpuOps reaches are ported (4-bit affine
// weights, bf16 activations, transposed weights, no nax).
#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>

namespace mllm {

// device.cpp Device::Device: arch "applegpu_g<gen><size>" (MTLDevice.architecture.name, or $MLX_METAL_GPU_ARCH as
// MLX), gen = the two digits before the last letter, size = the last letter ('g', 's', 'c', 'd', 'p').
struct GpuClass {
    int gen = 0;
    char size = 0;
};
inline GpuClass gpu_class(const std::string& arch) {
    GpuClass gpu;
    if (arch.size() >= 3) {
        int tens = arch[arch.size() - 3] - '0', ones = arch[arch.size() - 2] - '0';
        gpu.gen = (tens >= 0 && tens < 10 ? tens : 0) * 10 + (ones >= 0 && ones < 10 ? ones : 0);
    }
    gpu.size = arch.empty() ? 0 : arch.back();
    return gpu;
}

// Why GpuOps can't run on this GPU, or "" if it can. Gen 13..16 = M1..M4. Gen 17+ (M5): MLX switches its matmuls
// and attention to neural-accelerator (nax) kernels there (device.cpp is_nax_available), which aren't ported.
inline std::string gpu_unsupported(const std::string& arch) {
    GpuClass gpu = gpu_class(arch);
    if (arch.rfind("applegpu_g", 0) != 0 || gpu.gen < 13) return arch + " is not an Apple M1-M4 GPU";
    if (gpu.gen >= 17)
        return arch + " (M5 or later): MLX uses its neural-accelerator kernels there, which mmmllm hasn't ported";
    return "";
}

// quantized.cpp get_qmv_batch_limit (gen >= 13 branches): from this many rows, qmm (qmm_splitk) instead of qmv.
// in_size = D (K), out_size = O (N).
inline int qmv_batch_limit(GpuClass gpu, int in_size, int out_size) {
    const bool small = in_size <= 2048 && out_size <= 2048, medium = in_size <= 4096 && out_size <= 4096;
    if (gpu.gen >= 15 && gpu.size != 'd') return small ? 13 : medium ? 15 : 13;
    if (gpu.size == 'd') return small ? 32 : medium ? 18 : 12;
    return small ? 14 : medium ? 10 : (gpu.gen == 13 && gpu.size == 's') ? 7 : 6;
}

// quantized.cpp use_qmv_wide (affine): qmv_wide for 2 rows up to the batch limit only from gen 15; before, qmv.
// matmul.cpp gemv_wide_config: the same gen cut for bf16 gemv_wide.
inline bool wide_vector_kernels(GpuClass gpu) { return gpu.gen >= 15; }

// matmul.cpp steel_matmul_regular_axpby: bm, bn, bk, wm, wn of steel_gemm_fused (GEMM_TPARAM_MACRO, bf16 output,
// a not transposed). outputs = batch_size_out * M * N.
struct GemmTiles {
    int bm, bn, bk, wm, wn;
    bool operator==(const GemmTiles&) const = default;
};
inline GemmTiles gemm_tiles(GpuClass gpu, bool b_transposed, int64_t outputs, int M, int N, int K) {
    const GemmTiles nt{64, 32, 32, 2, 2}, nn{64, 64, 16, 1, 2};
    if (gpu.size == 'g' || gpu.size == 'p') return b_transposed ? nt : nn;  // small device
    if (gpu.size == 'd') {                                                 // large device
        if (outputs < (int64_t)1 << 20) return b_transposed ? nt : nn;     // smaller matmul
        if (2 * std::max(M, N) > K) return nn;                             // reasonable K: {64, 64, 16, 1, 2}
        return b_transposed ? nt : GemmTiles{32, 64, 16, 1, 2};            // large K
    }
    return {64, 64, 16, 2, 2};  // medium ('s', 'c', ...): the defaults
}

// scaled_dot_product_attention.cpp ScaledDotProductAttention::eval_gpu, vector mode: sdpa_vector_2pass from this
// many keys ('s' / 'd' from 1024; others only with fewer key_value heads than query heads, from 4096).
inline int two_pass_from_keys(GpuClass gpu, bool grouped_query) {
    if (gpu.size == 'd' || gpu.size == 's') return 1024;
    return grouped_query ? 4096 : INT_MAX;
}

// sdpa_vector_2pass: how many blocks the keys split into. simdgroups = query heads per key_value head x queries.
inline int two_pass_blocks(GpuClass gpu, int keys, int simdgroups) {
    if (gpu.size == 's') {
        if (keys > 1024 && simdgroups > 4) return keys <= 8192 ? 128 : keys <= 32768 ? 256 : keys <= 65536 ? 512 : 1024;
        return 64;
    }
    if (gpu.size == 'd') {
        if (simdgroups <= 2 && keys > 8192) return 256;
        if (simdgroups >= 6) return keys >= 65536 ? 1024 : keys >= 16384 ? 512 : 128;
        return 128;
    }
    return simdgroups >= 4 ? 64 : 32;
}

}  // namespace mllm
