// A decodes (B = 1, 64 random tokens) while B prefills P random tokens: A's token gaps alone and during B's prefill, and B's
// prefill time (target-27b.md §4: no decode step > 1 s). Built like tools/long_ctx.mm:
//   clang++ ... tools/overlap.mm build/libmllm.a ... -o build/overlap; build/overlap MODEL P [gpu|ane]
//   (ane: 1024-token chunks; a P below 256 runs as one short chunk on the GPU path, in the ANE lane)
#include <chrono>
#include <random>
#include "engine.h"
using clk = std::chrono::steady_clock;
int main(int argc, char** argv) {
    if (argc < 3) return fprintf(stderr, "usage: overlap MODEL P [gpu|ane]\n"), 2;
    const int P = atoi(argv[2]);
    const std::string prefill_on = argc > 3 ? argv[3] : "gpu";
    mllm::Engine e(argv[1], (size_t)12 << 30, prefill_on == "ane" ? 1024 : 512, 32, 16, 32768, prefill_on, false);
    e.tokenizer.eos.clear();
    std::mt19937 rng(1);
    std::uniform_int_distribution<int> tok(1000, 100000);
    auto ids = [&](int n) { std::vector<int> v(n); for (int& t : v) t = tok(rng); return v; };
    for (int rep = 0; rep < 2; rep++) {
        const int a = e.submit(ids(64), {.max_tokens = 2000});
        int na = 0;
        auto last = clk::now();
        std::vector<double> solo, during;
        int b = -1;
        clk::time_point tb;
        double b_time = 0;
        while (true) {
            for (auto& ev : e.step()) {
                if (ev.req_id == a) {
                    const double g = std::chrono::duration<double>(clk::now() - last).count() * 1e3;
                    last = clk::now();
                    if (++na > 5 && b < 0) solo.push_back(g);
                    else if (b >= 0) during.push_back(g);
                } else if (ev.req_id == b) {
                    b_time = std::chrono::duration<double>(clk::now() - tb).count();
                }
            }
            if (na == 25 && b < 0) b = e.submit(ids(P), {.max_tokens = 1}), tb = clk::now();
            if (b >= 0 && b_time > 0) break;
        }
        e.cancel(a);
        while (!e.requests.empty()) e.step();
        auto mean = [](auto& v) { double s = 0; for (double x : v) s += x; return s / v.size(); };
        printf("rep %d: solo gap mean %.1f ms max %.1f; during B (%d tok, %.2f s, %.1f tok/s): %zu tokens, gap mean %.1f "
               "max %.1f ms\n", rep,
               mean(solo), *std::max_element(solo.begin(), solo.end()), P, b_time, P / b_time, during.size(), mean(during),
               *std::max_element(during.begin(), during.end()));
        fflush(stdout);
    }
}
