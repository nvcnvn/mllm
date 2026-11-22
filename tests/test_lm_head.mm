// lm_head + sampling. (1) golden decode0..2: ws.x after dense(L) -> logits and greedy ids equal decodeN/logits,
// decodeN/argmax. (2) lm_head.safetensors: the quantized matmul at N = vocab for M = 1..16 (MLX switches kernels by
// M) bit-exact, argmax = the lowest index of the row's max. (3) argmax ties and edges. (4) make_sampler semantics:
// 5k seeded draws match the expected distribution (chi-square), top_p drops exactly the tail apply_top_p drops.
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <string>

#include "golden.h"
#include "lanes.h"

struct mllm::GpuOpsTest {
    static void argmax(GpuOps& ops, id<MTLCommandQueue> q, Workspace& ws, id<MTLBuffer> x, id<MTLBuffer> out, int rows,
                       int V) {
        mllm::encode_and_wait(q, ws, [&] {
            ops.use(ws);
            ops.argmax({x}, {out}, rows, V);
        });
    }
};

static uint16_t to_bf16(float f) {  // round to nearest even
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
static float from_bf16(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static int cpu_argmax(const uint8_t* row, int V) {  // bytes: golden tensors may sit at odd offsets
    int best = 0;
    for (int i = 1; i < V; i++)
        if (golden::to_double("BF16", row, i) > golden::to_double("BF16", row, best)) best = i;
    return best;
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors"), sw(d + "/lm_head.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 64 << 20);
    const int max_batch = 16, V = s.vocab;
    mllm::GpuOps ops(mem, w);
    mllm::Workspace ws = ops.workspace(1, max_batch, 64);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> ids = mem.buffer(max_batch * 4);
    mem.freeze();
    uint8_t* logits = (uint8_t*)ws.logits.contents;
    ws.S = 1;
    auto exact = [&](const std::string& name, const mllm::Tensor& want, const uint8_t* data) {
        bool same = !memcmp(logits, data, want.nbytes);
        if (!same) golden::matches(name, want, data, logits, 1, 0);  // prints cos and max |diff|
        return same;
    };
    auto run = [&](int B) {
        ws.B = B;
        @autoreleasepool {
            mllm::encode_and_wait(q, ws, [&] {
                ops.lm_head(ws);
            });
        }
        return ops.sample(ws, std::vector<mllm::SampleParams>(B));  // greedy
    };

    // (1) golden decode steps
    int steps = 0;
    for (const auto& call : g["calls"]) {
        std::string c = call["name"];
        if (!c.starts_with("decode")) continue;
        char key[64];
        snprintf(key, sizeof key, "%s/L%02d/dense/x", c.c_str(), s.n_layers());
        const mllm::Tensor& x = tr.at(key);
        const mllm::Tensor &want = tr.at(c + "/logits"), &am = tr.at(c + "/argmax");
        memcpy(ws.x.contents, tr.data(x), x.nbytes);
        std::vector<int> got = run((int)x.shape[0]);
        assert(exact(c + "/logits", want, tr.data(want)));
        for (size_t b = 0; b < got.size(); b++) assert(got[b] == golden::load<int32_t>(tr.data(am), b));
        steps++;
    }
    // (2) M = 1..16 at N = vocab
    const mllm::Tensor& xs = sw.at("x");
    memcpy(ws.x.contents, sw.data(xs), xs.nbytes);
    for (int M = 1; M <= 16; M++) {
        const mllm::Tensor& want = sw.at(std::to_string(M) + "/logits");
        std::vector<int> got = run(M);
        assert(exact("M = " + std::to_string(M), want, sw.data(want)));
        for (int b = 0; b < M; b++) assert(got[b] == cpu_argmax(sw.data(want) + (size_t)b * V * 2, V));
    }
    // (3) argmax: ties to the lowest index, max at the last index, all -inf -> 0, NaN-free rows only
    uint16_t* L = (uint16_t*)logits;
    const uint16_t ninf = 0xFF80;
    std::fill(L, L + 4 * (size_t)V, to_bf16(-3));
    L[1000] = L[200000] = L[V - 1] = to_bf16(5);           // row 0: tie, lowest wins
    L[(size_t)V + V - 1] = to_bf16(1);                      // row 1: the last index
    std::fill(L + 2 * (size_t)V, L + 3 * (size_t)V, ninf);  // row 2: all -inf (MLX: 0)
    L[3 * (size_t)V + 7] = L[3 * (size_t)V + 6] = to_bf16(-2.5);  // row 3: adjacent tie
    mllm::GpuOpsTest::argmax(ops, q, ws, ws.logits, ids, 4, V);
    const int32_t* id = (const int32_t*)ids.contents;
    assert(id[0] == 1000 && id[1] == V - 1 && id[2] == 0 && id[3] == 6);

    // (4) sampling on one row: 4 live tokens, the rest at -60 .. -75.5 (~e-27 of the mass in total). 32 values,
    // not one: a row of one repeated value makes the counting pass a serial chain (~4x slower per draw)
    for (int i = 0; i < V; i++) L[i] = to_bf16(-60 - (i % 32) * 0.5f);
    const std::pair<int, float> live[] = {{5, 2.0f}, {77, 1.3f}, {V - 1, 0.8f}, {0, 0.4f}};
    for (auto [i, v] : live) L[i] = to_bf16(v);
    ws.B = 1;
    // (temp, top_p, tokens kept: apply_top_p in Python on this row)
    struct Case {
        float temp, top_p;
        int kept;
    };
    // chi-square over the kept tokens, 99.9% quantile per degree of freedom (1..3)
    const int draws = 5000;
    const double chi2_max[] = {0, 10.83, 13.82, 16.27};
    for (Case k : {Case{1, 0, 4}, Case{0.7f, 0.7f, 2}, Case{0.7f, 0.8f, 3}, Case{1.5f, 0.95f, 4}}) {
        std::mt19937_64 rng(42);
        mllm::SampleParams p{k.temp, k.top_p, &rng};
        std::map<int, int> n;
        for (int i = 0; i < draws; i++) n[ops.sample(ws, {p})[0]]++;
        double z = 0, chi2 = 0;
        for (int j = 0; j < k.kept; j++) z += std::exp(from_bf16(L[live[j].first]) / k.temp);
        for (int j = 0; j < 4; j++) {
            int t = live[j].first;
            double want = j < k.kept ? draws * std::exp(from_bf16(L[t]) / k.temp) / z : 0;
            if (j < k.kept) chi2 += (n[t] - want) * (n[t] - want) / want;
            assert(j < k.kept || n[t] == 0);
        }
        printf("  temp %.1f top_p %.2f: chi2 %.2f (df %d, max %.2f)\n", k.temp, k.top_p, chi2, k.kept - 1, chi2_max[k.kept - 1]);
        assert(chi2 < chi2_max[k.kept - 1]);
        assert((int)std::count_if(n.begin(), n.end(), [](auto& e) { return e.second > 0; }) == k.kept);
        std::mt19937_64 r1(7), r2(7);  // seeded: the same draws
        for (int i = 0; i < 100; i++)
            assert(ops.sample(ws, {{k.temp, k.top_p, &r1}})[0] == ops.sample(ws, {{k.temp, k.top_p, &r2}})[0]);
    }
    // after those rows (the reused counts must be clean): -inf tokens never drawn; an all -inf row gives 0
    std::fill(L, L + V, ninf);
    L[123] = to_bf16(1);
    std::mt19937_64 r3(3);
    for (int i = 0; i < 200; i++) assert(ops.sample(ws, {{1, 0.5f, &r3}})[0] == 123);
    L[123] = ninf;
    assert(ops.sample(ws, {{1, 0, &r3}})[0] == 0);
    printf("decode steps: %d, logits and greedy ids bit-exact; lm_head M = 1..16 bit-exact; argmax ties ok;\n"
           "sampling: 4 cases x %d draws match make_sampler's distribution (chi-square, p 0.001), top_p tails dropped\nPASS\n",
           steps, draws);
    assert(steps == 3);
}
