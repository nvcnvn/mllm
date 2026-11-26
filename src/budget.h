// The engine's memory sized to what this Mac's GPU can keep resident (MTLDevice.recommendedMaxWorkingSetSize:
// 55.7 GB on the 64 GB M4 Pro, about 5.7 / 11.5 / 17.2 GB on 8 / 16 / 24 GB Macs). Pure functions of the model's
// shapes and sizes; tests/test_budget.cpp checks them in CI.
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>

#include "gpu_rules.h"
#include "spec.h"

namespace mllm {

// Bytes of Metal scratch the engine allocates besides the weights and the arena: GpuOps::workspace for the prefill
// lane (`chunk` rows) and the decode lane's two (max_batch rows), each with attention scratch for max_ctx keys; with
// `ane`, AneOps' 2 surface sets (one per chunk in flight), each a layer's weights in fp16 and a chunk's activations.
// Within 2% above the engine's "workspace" line (0.8B: 98.3 / 174.4 / 470.7 MiB at 4k / 16k / 64k ctx; ANE 513.7).
// ponytail: mirrors memory.mm / gpu.mm / ane.mm's buffers by hand, the feed-forward width taken as 4 x hidden
// (Qwen3.5/3.6: 3.4-3.5); Core ML's own memory (~2 GB at 27B) isn't Metal's and isn't counted. Re-measure when a
// workspace buffer changes.
inline size_t workspace_estimate(const ModelSpec& s, GpuClass gpu, int chunk, int max_batch, int max_context,
                                 bool ane = false) {
    const size_t q = (size_t)s.query_heads * s.head_dim, kv = (size_t)s.key_value_heads * s.head_dim;
    const size_t lv = (size_t)s.linear_value_heads * s.linear_value_dim, ff = 4 * (size_t)s.hidden;
    const size_t conv = s.conv_dim(), lvh = s.linear_value_heads;
    const int heads_per_kv = s.query_heads / s.key_value_heads;
    // bf16 elements per row: Memory::workspace's buffers, then GpuOps::workspace's
    const size_t row_bytes = 2 * (s.hidden + std::max(q, lv) + 2 * q + 2 * kv + conv + lv + 2 * lvh + 2 +
                                  3 * s.hidden + 2 * ff + 3 * q + conv + 3 * lvh);
    auto one = [&](int rows, int batch) {
        size_t bytes = std::max(rows, batch) * row_bytes + (size_t)batch * s.vocab_size * 2 + 512 * 32 * 32 * 4;
        size_t partial_rows = (size_t)batch * s.query_heads * two_pass_blocks(gpu, max_context, heads_per_kv);
        for (int queries = 1; queries <= std::min({rows, 8, 32 / heads_per_kv}); queries++)
            partial_rows = std::max(partial_rows, (size_t)queries * s.query_heads *
                                                      two_pass_blocks(gpu, max_context, heads_per_kv * queries));
        bytes += partial_rows * (s.head_dim * 2 + 8);
        // gpu.mm fused_attention_fits fails: gathered keys and values, scores per key
        if (!(rows <= 8 && rows * heads_per_kv <= 32))
            bytes += (size_t)max_context * (4 * kv + (size_t)rows * heads_per_kv * 2);
        return bytes;
    };
    // a layer's weights: the larger attention kind's projections, then the MLP
    const size_t layer = (size_t)s.hidden * (std::max(3 * q + 2 * kv, conv + 2 * lv + 2 * lvh) + 3 * ff);
    return one(chunk, 1) + 2 * one(1, max_batch) + (ane ? 2 * (2 * layer + chunk * row_bytes) : 0);
}

struct EngineSizes {
    size_t arena = 0;  // bytes
    int max_context = 0, max_batch = 0;
};
struct MemoryPlan {
    EngineSizes sizes;
    std::string lowered;  // what was shrunk to fit, e.g. "arena 12.0 -> 4.5 GiB, max_ctx 32768 -> 8192"
    std::string error;    // non-empty: don't load
};

// Smaller models to suggest, largest first: repo, weights loaded, GB they need at their defaults (weights, workspace,
// 1 GiB arena). ponytail: a fixed list; extend when another model size is checked bit-exact.
struct SmallerModel {
    const char* repo;
    double weights, need;
};
inline constexpr SmallerModel SMALLER_MODELS[] = {{"mlx-community/Qwen3.5-9B-4bit", 5.04e9, 6.4e9},
                                                  {"mlx-community/Qwen3.5-0.8B-4bit", 0.42e9, 1.7e9}};

// The engine's sizes: each flag given (nonzero in `flags`) as is, the rest `defaults`. When weights + workspace +
// arena pass 90% of `budget` (the rest for Metal's own buffers and other apps), the arena takes what's left (at most
// its default), max_ctx halves until that arena holds one request of it (down to 4096), and max_batch drops to the
// 1024-token requests the arena holds. An error when even that, or the flags as given, can't fit, naming the largest
// smaller model that fits. weights: bytes loaded (language_model.* tensors). unit / state_units: Memory::unit_of /
// state_units_of (32-token pages). ane: prefill on the Neural Engine (workspace_estimate).
inline MemoryPlan plan_memory(const ModelSpec& s, GpuClass gpu, int chunk, bool ane, size_t weights, size_t unit,
                              size_t state_units, size_t budget, EngineSizes defaults, EngineSizes flags) {
    const size_t usable = budget / 10 * 9;
    const EngineSizes asked{flags.arena ? flags.arena : defaults.arena,
                            flags.max_context ? flags.max_context : defaults.max_context,
                            flags.max_batch ? flags.max_batch : defaults.max_batch};
    EngineSizes p = asked;
    auto workspace = [&] { return workspace_estimate(s, gpu, chunk, p.max_batch, p.max_context, ane); };
    auto request = [&](int tokens) { return ((size_t)(tokens + 31) / 32 + state_units) * unit; };
    if (weights + workspace() + p.arena <= usable) return {p, {}, {}};
    const int min_context = std::min(4096, defaults.max_context);
    for (;;) {
        if (!flags.arena) {
            const size_t taken = weights + workspace();
            p.arena = std::min(defaults.arena, taken < usable ? (usable - taken) / unit * unit : 0);
        }
        if (weights + workspace() + p.arena <= usable && request(p.max_context) <= p.arena) break;
        if (flags.max_context || p.max_context <= min_context) {
            const double need = weights + workspace() + std::max(p.arena, request(p.max_context));
            std::string suggestion;
            for (const SmallerModel& m : SMALLER_MODELS)
                if (m.weights < weights * 0.9 && m.need <= usable) {
                    suggestion = std::string(": try a smaller model, e.g. ") + m.repo;
                    break;
                }
            char line[400];
            snprintf(line, sizeof line, "this model needs %.1f GB of GPU memory (weights %.2f GB, max_ctx %d), this "
                     "Mac's GPU can use %.1f GB%s%s", need / 1e9, weights / 1e9, p.max_context, usable / 1e9,
                     suggestion.c_str(), flags.arena || flags.max_context ? "; or a smaller --arena-gib / --max-ctx" : "");
            return {p, {}, line};
        }
        p.max_context = std::max(min_context, p.max_context / 2);
    }
    if (!flags.max_batch) p.max_batch = (int)std::clamp<size_t>(p.arena / request(1024), 1, p.max_batch);
    std::string lowered;
    char part[80];
    auto note = [&](bool changed, const char* format, auto from, auto to) {
        if (!changed) return;
        snprintf(part, sizeof part, format, from, to);
        lowered += (lowered.empty() ? "" : ", ") + std::string(part);
    };
    note(p.arena != asked.arena, "arena %.2f -> %.2f GiB", asked.arena / double(1 << 30), p.arena / double(1 << 30));
    note(p.max_context != asked.max_context, "max_ctx %d -> %d", asked.max_context, p.max_context);
    note(p.max_batch != asked.max_batch, "max batch %d -> %d", asked.max_batch, p.max_batch);
    return {p, lowered, {}};
}

}  // namespace mllm
