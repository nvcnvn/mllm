// gpu_rules.h against values read off MLX v0.32.4's source (quantized.cpp get_qmv_batch_limit / use_qmv_wide,
// matmul.cpp GEMM_TPARAM_MACRO, scaled_dot_product_attention.cpp sdpa_vector_2pass) for a few chips. Pure: no GPU.
#include <cstdio>

#include "gpu_rules.h"

using namespace mllm;

static int failures = 0;
#define CHECK(x) ((x) ? (void)0 : (void)(failures++, fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x)))

int main() {
    // device.cpp parse; MLX's size letters: 'g' base / Pro, 's' Pro / Max ("medium"), 'c' Max, 'd' Ultra
    GpuClass m1 = gpu_class("applegpu_g13g"), m1_pro = gpu_class("applegpu_g13s"), m1_max = gpu_class("applegpu_g13c"),
             m1_ultra = gpu_class("applegpu_g13d"), m2_pro = gpu_class("applegpu_g14s"), m3 = gpu_class("applegpu_g15g"),
             m3_ultra = gpu_class("applegpu_g15d"), m4_pro = gpu_class("applegpu_g16s");
    CHECK(m1.gen == 13 && m1.size == 'g' && m4_pro.gen == 16 && m4_pro.size == 's');

    CHECK(gpu_unsupported("applegpu_g16s").empty() && gpu_unsupported("applegpu_g13g").empty() &&
          gpu_unsupported("applegpu_g15d").empty());
    CHECK(!gpu_unsupported("applegpu_g17s").empty() && !gpu_unsupported("applegpu_g18p").empty() &&
          !gpu_unsupported("applegpu_g12g").empty() && !gpu_unsupported("AMD Radeon Pro 5500M").empty());

    // get_qmv_batch_limit(D, O): small (<= 2048 both), medium (<= 4096), else (27B's 5120 x 17408)
    struct { GpuClass gpu; int small, medium, large; } limits[] = {
        {m1, 14, 10, 6},      {m1_pro, 14, 10, 7},    {m1_max, 14, 10, 6}, {m1_ultra, 32, 18, 12},
        {m2_pro, 14, 10, 6},  {m3, 13, 15, 13},       {m3_ultra, 32, 18, 12}, {m4_pro, 13, 15, 13},
    };
    for (auto& l : limits) {
        CHECK(qmv_batch_limit(l.gpu, 1024, 2048) == l.small);
        CHECK(qmv_batch_limit(l.gpu, 4096, 4096) == l.medium);
        CHECK(qmv_batch_limit(l.gpu, 5120, 17408) == l.large);
    }

    // use_qmv_wide / gemv_wide_config: gen 15+
    CHECK(!wide_vector_kernels(m1) && !wide_vector_kernels(m2_pro) && wide_vector_kernels(m3) &&
          wide_vector_kernels(m3_ultra) && wide_vector_kernels(m4_pro));

    // GEMM_TPARAM_MACRO, bf16; 27B prefill chunk: 24 heads x 1024 queries x 1024 keys, head_dim 256
    const GemmTiles medium{64, 64, 16, 2, 2}, nt{64, 32, 32, 2, 2}, nn{64, 64, 16, 1, 2};
    const int64_t big = 24 * 1024 * 1024, few = 16 * 16 * 16;
    CHECK(gemm_tiles(m4_pro, true, big, 1024, 1024, 256) == medium && gemm_tiles(m4_pro, false, big, 1024, 256, 1024) == medium);
    CHECK(gemm_tiles(m1_max, true, big, 1024, 1024, 256) == medium);
    CHECK(gemm_tiles(m1, true, big, 1024, 1024, 256) == nt && gemm_tiles(m3, false, big, 1024, 256, 1024) == nn);
    CHECK(gemm_tiles(m1_ultra, true, few, 16, 16, 256) == nt && gemm_tiles(m1_ultra, false, few, 16, 16, 256) == nn);
    CHECK(gemm_tiles(m1_ultra, true, big, 1024, 1024, 256) == nn);    // reasonable K: 64 64 16 1 2 either way
    CHECK(gemm_tiles(m1_ultra, false, big, 1024, 256, 4096) == (GemmTiles{32, 64, 16, 1, 2}));  // nn, large K
    CHECK(gemm_tiles(m1_ultra, true, big, 1024, 1024, 4096) == nt);   // nt, large K

    // sdpa_vector_2pass: from 1024 keys on 's' / 'd', from 4096 with GQA elsewhere, never without
    CHECK(two_pass_from_keys(m4_pro, true) == 1024 && two_pass_from_keys(m1_ultra, false) == 1024);
    CHECK(two_pass_from_keys(m1, true) == 4096 && two_pass_from_keys(m1_max, true) == 4096);
    CHECK(two_pass_from_keys(m3, false) == INT_MAX);
    // blocks
    CHECK(two_pass_blocks(m4_pro, 1024, 6) == 64 && two_pass_blocks(m4_pro, 2000, 6) == 128 &&
          two_pass_blocks(m4_pro, 2000, 4) == 64 && two_pass_blocks(m4_pro, 40000, 6) == 512 &&
          two_pass_blocks(m4_pro, 100000, 6) == 1024);
    CHECK(two_pass_blocks(m1_ultra, 2000, 4) == 128 && two_pass_blocks(m1_ultra, 10000, 2) == 256 &&
          two_pass_blocks(m1_ultra, 20000, 6) == 512 && two_pass_blocks(m1_ultra, 70000, 6) == 1024 &&
          two_pass_blocks(m1_ultra, 10000, 6) == 128);
    CHECK(two_pass_blocks(m1, 5000, 6) == 64 && two_pass_blocks(m1, 5000, 2) == 32 && two_pass_blocks(m1_max, 5000, 3) == 32);

    printf(failures ? "test_gpu_rules: %d FAILED\n" : "test_gpu_rules: ok\n", failures);
    return failures != 0;
}
