// Kernel checks the golden trace can't give, against CPU fp32/fp64 references on random bf16 inputs:
//  - RMSNorm and the gated RMSNorm (linear attn(i) uses it in g4; the trace has no input/output pair for it alone)
//  - GpuOps::linear on every kernel it can pick: qmv_fast (M = 1), qmv_wide nv 2..5, qmm_t_splitk aligned and not,
//    qmm_t aligned and not (the trace only reaches qmv_wide nv 2 and qmm_t_splitk)
// Then times each projection shape at M = 1, 8, 16, 512 next to tools/roofline.py's bound.
#include <cassert>
#include <random>

#include "golden.h"
#include "gpu.h"

static float bf16(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static uint16_t to_bf16(float f) {  // round to nearest even (no NaNs here)
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static const uint16_t* bits(mllm::Buf b) { return (const uint16_t*)((const uint8_t*)b.buffer.contents + b.offset); }

static bool close(const char* what, const uint16_t* got, const std::vector<double>& want) {
    double dot = 0, gg = 0, ww = 0;
    for (size_t i = 0; i < want.size(); i++) {
        double a = bf16(got[i]);
        dot += a * want[i], gg += a * a, ww += want[i] * want[i];
    }
    double cos = dot / std::sqrt(gg * ww);
    if (cos < 0.99999) fprintf(stderr, "%s: cos %.7f\n", what, cos);
    return cos >= 0.99999;
}

// GpuOps' friend: the kernel helpers are private (they take Metal buffers).
struct mllm::GpuOpsTest {
    static void run();
};

int main() {
    @autoreleasepool {
        mllm::GpuOpsTest::run();
    }
}

void mllm::GpuOpsTest::run() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 64 << 20);
    mllm::GpuOps ops(mem, w, 1);
    std::mt19937 rng(0);
    std::normal_distribution<float> normal;
    const size_t big = (size_t)8224 * 1024;  // the largest x / y below, in elements
    id<MTLBuffer> x = mem.buffer(big * 2), y = mem.buffer(big * 2), z = mem.buffer(big * 2);
    auto fill = [&](id<MTLBuffer> b, size_t n, float scale) {
        uint16_t* p = (uint16_t*)b.contents;
        for (size_t i = 0; i < n; i++) p[i] = to_bf16(normal(rng) * scale);
    };
    const uint16_t* xs = (const uint16_t*)x.contents;
    const uint16_t* zs = (const uint16_t*)z.contents;
    const uint16_t* ys = (const uint16_t*)y.contents;

    // RMSNorm (hidden, final norm weight) and gated RMSNorm (head_v_dim, layer 0's linear_attn.norm), x in place
    const mllm::GpuOps::Layer& l0 = ops.layers[0];
    auto lnorm = w["language_model.model.layers.0.linear_attn.norm.weight"];
    for (bool gated : {false, true}) {
        int rows = 64, dim = gated ? s.lin_v_dim : s.hidden;
        mllm::Buf weight = gated ? mllm::Buf{lnorm.buffer, lnorm.offset} : ops.norm;
        fill(x, (size_t)rows * dim, 3), fill(z, (size_t)rows * dim, 2);
        std::vector<double> want((size_t)rows * dim);
        for (int r = 0; r < rows; r++) {
            const uint16_t* xr = xs + (size_t)r * dim;
            float ss = 0;
            for (int i = 0; i < dim; i++) ss += bf16(xr[i]) * bf16(xr[i]);
            float inv = 1 / std::sqrt(ss / dim + 1e-6f);
            for (int i = 0; i < dim; i++) {
                double v = bf16(to_bf16(bf16(xr[i]) * inv)) * bf16(bits(weight)[i]);
                if (gated) {
                    double g = bf16(zs[(size_t)r * dim + i]);
                    v *= g / (1 + std::exp(-g));
                }
                want[(size_t)r * dim + i] = v;
            }
        }
        ops.begin();
        if (gated) ops.gated_rms_norm(weight, {x}, {z}, {x}, rows, dim);
        else ops.rms_norm(weight, {x}, {x}, rows, dim);
        ops.commit_and_wait();
        assert(close(gated ? "gated_rms_norm" : "rms_norm", xs, want));
    }

    // linear: y = x @ dequant(W)^T in fp64 from the same bf16 inputs
    auto reference = [&](const mllm::QLinear& l, int M) {
        std::vector<double> want((size_t)M * l.N, 0), wrow(l.K);
        const uint32_t* q = (const uint32_t*)((const uint8_t*)l.w.buffer.contents + l.w.offset);
        for (int n = 0; n < l.N; n++) {
            for (int k = 0; k < l.K; k++) {
                int gi = n * (l.K / 64) + k / 64;
                wrow[k] = bf16(bits(l.scales)[gi]) * ((q[(size_t)n * (l.K / 8) + k / 8] >> (4 * (k % 8))) & 15) +
                          bf16(bits(l.biases)[gi]);
            }
            for (int m = 0; m < M; m++) {
                double acc = 0;
                for (int k = 0; k < l.K; k++) acc += bf16(xs[(size_t)m * l.K + k]) * wrow[k];
                want[(size_t)m * l.N + n] = acc;
            }
        }
        return want;
    };
    const mllm::GpuOps::Layer& l3 = ops.layers[3];
    struct Case {
        const char* kernel;
        const mllm::QLinear& l;
        int M;
    } cases[] = {
        {"qmv_fast", l0.down_proj, 1},          {"qmv_wide nv2", l0.in_proj_z, 2},
        {"qmv_wide nv3", l0.out, 3},            {"qmv_wide nv4", l3.q_proj, 8},
        {"qmv_wide nv5", l0.in_proj_b, 5},      {"qmm_t_splitk aligned", l3.k_proj, 16},
        {"qmm_t_splitk unaligned", l0.in_proj_a, 32}, {"qmm_t aligned", l3.v_proj, 1024},
        {"qmm_t unaligned", l0.in_proj_a, 8224},
    };
    for (const Case& c : cases) {
        fill(x, (size_t)c.M * c.l.K, 1);
        ops.begin();
        ops.linear(c.l, {x}, {y}, c.M);
        ops.commit_and_wait();
        assert(close(c.kernel, ys, reference(c.l, c.M)));
    }
    printf("rms_norm, gated_rms_norm and %zu linear kernel paths match the CPU\n", std::size(cases));

    // Timing: each shape rotates through >= 64 MiB of weight copies (each layer's, repeated), so weights come from
    // DRAM as in a real step; fewer copies stay in the SLC and read faster than DRAM peak (273 GB/s, flagged).
    // Roofline (tools/roofline.py knobs): max(bytes / BW, flops / FLOPS), BW = 210 GB/s, FLOPS = 6 TFLOPS,
    // bytes = weights at 4.5 bits + bf16 x and y.
    using P = mllm::QLinear mllm::GpuOps::Layer::*;
    using L = mllm::GpuOps::Layer;
    struct Shape {
        const char* name;
        P proj;
    } shapes[] = {{"in_proj_qkv", &L::in_proj_qkv}, {"in_proj_z", &L::in_proj_z}, {"in_proj_a/b", &L::in_proj_a},
                  {"q_proj", &L::q_proj}, {"k/v_proj", &L::k_proj}, {"o/out_proj", &L::out},
                  {"gate/up_proj", &L::gate_proj}, {"down_proj", &L::down_proj}};
    fill(x, (size_t)512 * 3584, 1);
    @autoreleasepool {
        ops.begin();  // ~0.2 s of work first, so the GPU clock has ramped up before the first timed shape
        for (int r = 0; r < 200; r++) ops.linear(l0.in_proj_qkv, {x}, {y}, 512);
        ops.commit_and_wait();
    }
    printf("%-13s %5s %5s %4s %9s %9s %6s %7s\n", "proj", "N", "K", "M", "us", "bound us", "%bound", "GB/s");
    for (const Shape& sh : shapes) @autoreleasepool {
        std::vector<const mllm::QLinear*> src;
        for (const L& l : ops.layers)
            if ((l.*sh.proj).N) src.push_back(&(l.*sh.proj));
        const int N = src[0]->N, K = src[0]->K;
        const size_t wb = (size_t)N * K / 2, sb = (size_t)N * K / 64 * 2, per = wb + 2 * sb;  // all 16-aligned
        const size_t n = std::max(src.size(), ((size_t)64 << 20) / per + 1);
        id<MTLBuffer> pool = mem.buffer(n * per);
        std::vector<mllm::QLinear> copies;
        for (size_t c = 0; c < n; c++) {
            const mllm::QLinear& l = *src[c % src.size()];
            size_t o = c * per;
            auto copy = [&](mllm::Buf b, size_t off, size_t len) {
                memcpy((uint8_t*)pool.contents + off, (const uint8_t*)b.buffer.contents + b.offset, len);
                return mllm::Buf{pool, off};
            };
            copies.push_back({copy(l.w, o, wb), copy(l.scales, o + wb, sb), copy(l.biases, o + wb + sb, sb), N, K});
        }
        for (int M : {1, 8, 16, 512}) {
            double best = 1e9;
            for (int trial = 0; trial < 3; trial++) @autoreleasepool {
                ops.begin();
                for (const mllm::QLinear& l : copies) ops.linear(l, {x}, {y}, M);
                best = std::fmin(best, ops.commit_and_wait() / copies.size());
            }
            double bytes = (double)N * K * 4.5 / 8 + 2.0 * M * K + 2.0 * M * N;
            double bound = std::fmax(bytes / 210e9, 2.0 * M * N * K / 6e12), gbs = bytes / best / 1e9;
            printf("%-13s %5d %5d %4d %9.1f %9.1f %5.0f%% %7.0f%s\n", sh.name, N, K, M, best * 1e6, bound * 1e6,
                   100 * bound / best, gbs, gbs > 273 ? "  above DRAM peak: cached" : "");
        }
    }
    printf("PASS\n");
}
