// budget.h: the engine's sizes on Macs of 8 / 16 / 24 / 32 / 64 GB (recommendedMaxWorkingSetSize as Metal reports it:
// 5.72 / 11.45 / 17.18 / 22.9 GB, 55.66 on this 64 GB M4 Pro) for the 0.8B and 27B's shapes. Pure: no GPU.
#include <cstdio>

#include "budget.h"

using namespace mllm;

static int failures = 0;
#define CHECK(x) ((x) ? (void)0 : (void)(failures++, fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x)))

static ModelSpec shape(int layers, int hidden, int heads, int kv_heads, int linear_value_heads) {
    ModelSpec s{.hidden = hidden, .vocab_size = 248320, .query_heads = heads, .key_value_heads = kv_heads,
                .head_dim = 256, .linear_key_heads = 16, .linear_value_heads = linear_value_heads,
                .linear_key_dim = 128, .linear_value_dim = 128, .conv_kernel = 4};
    for (int i = 0; i < layers; i++) s.kinds.push_back(i % 4 == 3 ? "full" : "linear");
    return s;
}

int main() {
    const GpuClass m4_pro = gpu_class("applegpu_g16s");
    const size_t GiB = (size_t)1 << 30, MiB = 1 << 20;
    // weights = the language_model.* tensors (the engine's "weights" line); unit / state units = Memory::unit_of /
    // state_units_of
    struct Model { ModelSpec spec; size_t weights, unit, state_units; } small{shape(24, 1024, 8, 2, 16), 420000000, 384 << 10, 50},
        big{shape(64, 5120, 24, 4, 48), 15130000000, 2 * MiB, 74};
    auto plan = [&](const Model& m, double budget_gb, EngineSizes defaults, EngineSizes flags = {}, bool ane = false) {
        return plan_memory(m.spec, m4_pro, ane ? 1024 : 512, ane, m.weights, m.unit, m.state_units,
                           (size_t)(budget_gb * 1e9), defaults, flags);
    };
    const EngineSizes small_generate{GiB, 16384, 16}, small_serve{2 * GiB, 16384, 32}, big_generate{12 * GiB, 32768, 16},
        big_serve{12 * GiB, 32768, 32};

    // the engine's "workspace" line: 0.8B at 4k / 16k / 64k ctx (chunk 512, max batch 16), 27B at 32k; with ANE
    // prefill (chunk 1024): 0.8B at 4k / 16k, 27B at 32k
    const struct { const Model& m; int chunk, ctx; bool ane; double mib; } measured[] = {
        {small, 512, 4096, false, 98.3},  {small, 512, 16384, false, 174.4}, {small, 512, 65536, false, 470.7},
        {big, 512, 32768, false, 580.1},  {small, 1024, 4096, true, 389.6},  {small, 1024, 16384, true, 513.7},
        {big, 1024, 32768, true, 2991.8}};
    for (const auto& w : measured) {
        const double estimate = workspace_estimate(w.m.spec, m4_pro, w.chunk, 16, w.ctx, w.ane) / double(MiB);
        CHECK(estimate >= w.mib && estimate < w.mib * 1.05);
    }

    // the 0.8B: defaults on every Mac
    for (double budget : {5.72, 11.45, 17.18, 55.66}) {
        MemoryPlan p = plan(small, budget, small_generate), s = plan(small, budget, small_serve);
        CHECK(p.error.empty() && p.lowered.empty() && p.sizes.arena == GiB && p.sizes.max_context == 16384 &&
              p.sizes.max_batch == 16);
        CHECK(s.error.empty() && s.lowered.empty() && s.sizes.arena == 2 * GiB && s.sizes.max_batch == 32);
    }
    // the 27B: this Mac keeps today's defaults (also with ANE prefill); 8 / 16 / 24 GB can't hold it, and the error
    // names the largest smaller model that fits
    for (bool ane : {false, true}) {
        MemoryPlan here = plan(big, 55.66, big_serve, {}, ane);
        CHECK(here.error.empty() && here.lowered.empty() && here.sizes.arena == 12 * GiB &&
              here.sizes.max_context == 32768 && here.sizes.max_batch == 32);
    }
    const struct { double budget; const char* suggested; } small_macs[] = {
        {5.72, "Qwen3.5-0.8B"}, {11.45, "Qwen3.5-9B"}, {17.18, "Qwen3.5-9B"}};
    for (const auto& mac : small_macs) {
        MemoryPlan p = plan(big, mac.budget, big_generate);
        CHECK(p.error.find(mac.suggested) != std::string::npos && p.error.find("--arena-gib") == std::string::npos);
    }
    // 32 GB: a smaller arena, still one 32k request; serve's batch to the 1024-token requests it holds; ANE prefill's
    // surfaces leave a smaller arena
    MemoryPlan mid = plan(big, 22.9, big_generate), mid_serve = plan(big, 22.9, big_serve),
               mid_ane = plan(big, 22.9, big_generate, {}, true);
    CHECK(mid.error.empty() && mid.sizes.arena > 4 * GiB && mid.sizes.arena < 5 * GiB &&
          mid.sizes.max_context == 32768 && mid.sizes.max_batch == 16);
    CHECK(mid.lowered.starts_with("arena 12.00 -> 4.") && mid.lowered.ends_with(" GiB"));
    CHECK(mid.lowered.find("max_ctx") == std::string::npos && mid.lowered.find("batch") == std::string::npos);
    CHECK(mid_serve.error.empty() && mid_serve.sizes.max_batch > 8 && mid_serve.sizes.max_batch < 32 &&
          mid_serve.lowered.find("max batch 32 -> ") != std::string::npos);
    CHECK(big.weights + workspace_estimate(big.spec, m4_pro, 512, 16, 32768) + mid.sizes.arena <= 22.9e9 * 0.9);
    CHECK(mid_ane.error.empty() && mid_ane.sizes.arena + 2 * GiB < mid.sizes.arena &&
          big.weights + workspace_estimate(big.spec, m4_pro, 1024, 16, mid_ane.sizes.max_context, true) +
                  mid_ane.sizes.arena <= 22.9e9 * 0.9);

    // a tight budget: max_ctx halves until the arena holds one request; below 4096, an error with no smaller model
    MemoryPlan tight = plan(small, 0.8, small_generate);
    CHECK(tight.error.empty() && tight.sizes.max_context == 8192 &&
          tight.sizes.arena >= (8192 / 32 + 50) * (384 << 10) && tight.sizes.max_batch < 16);
    CHECK(tight.lowered.find("max_ctx 16384 -> 8192") != std::string::npos);
    const std::string too_small = plan(small, 0.6, small_generate).error;
    CHECK(!too_small.empty() && too_small.find("try") == std::string::npos);
    // flags win: kept as given, or the error (with the flags' hint)
    MemoryPlan kept = plan(small, 0.8, small_generate, {0, 4096, 0});
    CHECK(kept.error.empty() && kept.sizes.max_context == 4096);
    CHECK(plan(small, 0.8, small_generate, {0, 16384, 0}).error.find("--max-ctx") != std::string::npos);
    CHECK(!plan(big, 55.66, big_generate, {48 * GiB, 0, 0}).error.empty());
    MemoryPlan arena_given = plan(big, 55.66, big_generate, {20 * GiB, 65536, 0});
    CHECK(arena_given.error.empty() && arena_given.lowered.empty() && arena_given.sizes.arena == 20 * GiB &&
          arena_given.sizes.max_context == 65536);

    printf(failures ? "test_budget: %d FAILED\n" : "test_budget: ok\n", failures);
    return failures != 0;
}
