// One long request alone: prefill progress every 4096 tokens, then decode tok/s and the longest decode step; first,
// the prefix cache's state copy (Engine::copy_units: state_units units, CPU memcpy on the arena) and admit's memset.
// Not built by make (a measuring tool for target-27b.md §4 / §6):
//   clang++ -std=c++20 -O2 -fobjc-arc -Isrc -isystem src/vendor tools/long_ctx.mm build/libmllm.a -framework Metal \
//     -framework Foundation -framework CoreML -framework CoreVideo -framework IOSurface -o build/long_ctx
//   build/long_ctx MODEL MAX_CTX P [P ...]   (each P: a fresh random prompt, then 128 greedy tokens; EOS ignored)
#include <mach/mach.h>

#include <chrono>
#include <cstring>
#include <random>

#include "engine.h"

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

static double footprint_gb() {
    task_vm_info_data_t vi;
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n);
    return vi.phys_footprint / 1e9;
}

int main(int argc, char** argv) {
    if (argc < 4) return fprintf(stderr, "usage: long_ctx MODEL MAX_CTX P [P ...]\n"), 2;
    const int max_ctx = atoi(argv[2]), G = 128;
    mllm::Engine e(argv[1], (size_t)12 << 30, 512, 32, 16, max_ctx, "gpu", false);
    e.tokenizer.eos.clear();
    printf("footprint after load %.2f GB\n", footprint_gb());

    // the copy a snapshot (or a hit's admit) makes, and a cold admit's memset, on idle arena units
    uint8_t* base = (uint8_t*)e.mem.arena_buffer.contents;
    const size_t ub = e.mem.arena.unit_bytes, su = e.mem.state_units;
    for (int i = 0; i < 3; i++) {
        const uint32_t src = 2 * su * i, dst = src + su;  // units far apart, as in a busy arena
        auto t = clk::now();
        for (size_t j = 0; j < su; j++) memcpy(base + (dst + j) * ub, base + (src + j) * ub, ub);
        const double c = since(t);
        t = clk::now();
        for (size_t j = 0; j < su; j++) memset(base + (dst + j) * ub, 0, ub);
        printf("state copy %zu MiB: %.1f ms (%.1f GB/s); memset %.1f ms\n", su * ub >> 20, c * 1e3, su * ub / c / 1e9,
               since(t) * 1e3);
    }

    std::mt19937 rng(0);
    std::uniform_int_distribution<int> tok(1000, 100000);
    for (int a = 3; a < argc; a++) {
        const int P = atoi(argv[a]);
        std::vector<int> ids(P);
        for (int& t : ids) t = tok(rng);
        const auto t0 = clk::now();
        const int id = e.submit(ids, {.max_tokens = G});
        int mark = 4096, last_pos = 0;
        double last_t = 0, t_first = 0, prev = 0, worst = 0;
        int n = 0;
        while (e.requests.count(id)) {
            const auto ev = e.step();
            auto it = e.requests.find(id);
            if (it != e.requests.end() && it->second.state != mllm::State::WAITING && it->second.position >= mark && !n) {
                const int pos = it->second.position;
                const double t = since(t0);
                printf("  P %d: pos %6d at %7.1f s (%.1f tok/s since pos %d)\n", P, pos, t, (pos - last_pos) / (t - last_t),
                       last_pos);
                fflush(stdout);
                last_pos = pos, last_t = t, mark = pos / 4096 * 4096 + 4096;
            }
            for (size_t k = 0; k < ev.size(); k++) {
                const double t = since(t0);
                if (n++ == 0) t_first = t;
                else worst = std::max(worst, t - prev);
                prev = t;
            }
        }
        printf("P %d: prefill + first token %.1f s (%.1f tok/s); decode %.2f tok/s over %d tokens, longest step %.0f ms; "
               "footprint %.2f GB\n", P, t_first, P / t_first, (n - 1) / (prev - t_first), n - 1, worst * 1e3,
               footprint_gb());
        fflush(stdout);
    }
}
