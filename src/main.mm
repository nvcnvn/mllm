// mllm CLI: print the model spec, generate (streams text), chat (multi-turn in the terminal), bench (mlx_lm.benchmark's fields), bench-ops (GPU time per op).
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <random>
#include <string>

#include <unistd.h>

#include "budget.h"
#include "safetensors.h"
#include "engine.h"
#include "lanes.h"
#include "server.h"
#include "spec.h"

using SteadyClock = std::chrono::steady_clock;

static double seconds(SteadyClock::duration duration) { return std::chrono::duration<double>(duration).count(); }

namespace mllm {

// tools/roofline.py calibration knobs (keep the two in step): effective bandwidth and FP16 rate; 273 GB/s is the M4
// Pro's peak
constexpr double EFFECTIVE_BANDWIDTH = 210e9, PEAK_BANDWIDTH = 273e9, FP16_FLOPS = 7.5e12;
// rotate weights / K/V / state over at least this many bytes, so reads come from DRAM
constexpr size_t ROTATION_BYTES = 64 << 20;

static const char* TINY_NOTE = "tiny: dispatch / fill floor of a few us, not bandwidth";
static const char* CACHED_NOTE = "activations stay in cache (no rotation, as in a real step): a few-us floor when small";

// One line of the bench-ops table: the op's time next to its roofline bound (the larger of its bytes at the effective
// bandwidth and its flops at the FP16 rate); the note shows below 70% of the bound or above the peak bandwidth
static void print_row(const std::string& op, const std::string& shape, double op_seconds, double bytes, double flops,
                      const char* note) {
    const double bound = std::max(bytes / EFFECTIVE_BANDWIDTH, flops / FP16_FLOPS);
    const double gb_per_second = bytes / op_seconds / 1e9, percent_of_bound = 100 * bound / op_seconds;
    const bool above_peak = gb_per_second * 1e9 > PEAK_BANDWIDTH;
    printf("  %-34s %-14s %9.1f %7.1f%s %9.1f %5.0f%%  %s\n", op.c_str(), shape.c_str(), op_seconds * 1e6, gb_per_second,
           above_peak ? "!" : " ", bound * 1e6, percent_of_bound, percent_of_bound < 70 || above_peak ? note : "");
}

// bench-ops needs GpuOps' kernel helpers and layers (gpu.h friend). Holds one run's setup; each step is a member.
struct GpuOpsTest {
    static void bench_ops(const std::string& model_name);

    const ModelSpec& model;
    Memory& memory;
    GpuOps& ops;
    Workspace& ws;
    id<MTLCommandQueue> queue;
    StepTables& tables;
    std::vector<id<MTLBuffer>>& table_sets;  // one step table per rotation set
    id<MTLBuffer> input, output;             // activations in and out of the timed ops
    id<MTLBuffer> pool;                      // where rotated_copies puts its copies
    const GpuOps::Layer &full_layer, &linear_layer;
    std::vector<std::vector<uint32_t>> units = {};  // per request; the tables are written from these spans

    // GPU seconds for `repeats` encode(repeat) calls in one command buffer
    template <class Encode>
    double seconds_for_repeats(Encode& encode, int repeats) {
        @autoreleasepool {
            return encode_and_wait(queue, ws, [&] {
                ops.begin(ws);
                for (int repeat = 0; repeat < repeats; repeat++) encode(repeat);
            });
        }
    }
    // GPU seconds per encode(repeat) call: enough calls in one command buffer to take ~20 ms, best of 3
    template <class Encode>
    double gpu_seconds(Encode&& encode) {
        const double one_call = seconds_for_repeats(encode, 1);
        const int repeats = std::clamp((int)(0.02 / std::max(one_call, 1e-6)), 3, 2000);
        double best = INFINITY;
        for (int attempt = 0; attempt < 3; attempt++)
            best = std::min(best, seconds_for_repeats(encode, repeats) / repeats);
        return best;
    }

    std::vector<QuantizedWeights> rotated_copies(const QuantizedWeights& weights);
    int make_table_sets(int requests, int position, int tokens, bool linear, double set_bytes, int layers);
    void bench_multiply(const std::string& name, const QuantizedWeights& weights, int rows, bool decode);
    void bench_norms(int rows);
    void bench_attention(int rows, bool decode);
    double time_decode_step(double& wall_seconds);
    void bench_whole_decode_step(int requests, double mlx_baseline);
    void bench_verify_step(int tokens);
};

// Copies of weights over >= ROTATION_BYTES of pool (the weights themselves when they are that big)
std::vector<QuantizedWeights> GpuOpsTest::rotated_copies(const QuantizedWeights& weights) {
    const size_t packed_bytes = (size_t)weights.out_size * weights.in_size / 2;
    const size_t scale_bytes = (size_t)weights.out_size * weights.in_size / 64 * ELEM;  // scales, and again biases
    if (packed_bytes + 2 * scale_bytes >= ROTATION_BYTES) return std::vector<QuantizedWeights>{weights};
    auto aligned = [](size_t bytes) { return (bytes + 255) & ~(size_t)255; };
    const size_t copies = (ROTATION_BYTES + packed_bytes + 2 * scale_bytes - 1) / (packed_bytes + 2 * scale_bytes);
    const size_t stride = aligned(packed_bytes) + 2 * aligned(scale_bytes);
    assert(copies * stride <= pool.length);
    std::vector<QuantizedWeights> out;
    uint8_t* pool_bytes = (uint8_t*)pool.contents;
    for (size_t copy_index = 0; copy_index < copies; copy_index++) {
        QuantizedWeights copy = weights;
        const size_t offset = copy_index * stride;
        copy.packed = {pool, offset};
        copy.scales = {pool, offset + aligned(packed_bytes)};
        copy.biases = {pool, offset + aligned(packed_bytes) + aligned(scale_bytes)};
        memcpy(pool_bytes + offset, (uint8_t*)weights.packed.buffer.contents + weights.packed.offset, packed_bytes);
        memcpy(pool_bytes + copy.scales.offset, (uint8_t*)weights.scales.buffer.contents + weights.scales.offset, scale_bytes);
        memcpy(pool_bytes + copy.biases.offset, (uint8_t*)weights.biases.buffer.contents + weights.biases.offset, scale_bytes);
        out.push_back(copy);
    }
    return out;
}

// Sets of `requests` requests on their own units (pages for position + tokens tokens, or a state span), one step table
// each; enough sets that set_bytes (one layer's distinct footprint per set) * sets * layers >= ROTATION_BYTES. Returns
// the number of sets (in table_sets' first entries).
int GpuOpsTest::make_table_sets(int requests, int position, int tokens, bool linear, double set_bytes, int layers) {
    const uint32_t units_per_request = linear ? memory.state_units : memory.units_for(position + tokens);
    const int fit = std::min<int>((int)table_sets.size(), memory.arena.n_units / (requests * units_per_request));
    assert(fit >= 1);
    const int sets = std::clamp((int)std::ceil(ROTATION_BYTES / (set_bytes * layers)), 1, fit);
    units.assign((size_t)requests * sets, {});
    uint32_t next = 0;
    for (int set = 0; set < sets; set++) {
        std::vector<StepTables::Row> rows;
        for (int request = 0; request < requests; request++) {
            std::vector<uint32_t>& request_units = units[set * requests + request];
            for (uint32_t j = 0; j < units_per_request; j++) request_units.push_back(next++);
            const std::span<const uint32_t> none{};
            rows.push_back({position, tokens, linear ? none : request_units, linear ? request_units : none});
        }
        tables.write(table_sets[set], rows);
    }
    return sets;
}

// Why a weight multiply may be below its bound, by shape and kernel (MLX's kernel names)
static const char* multiply_note(const QuantizedWeights& weights, int rows, bool decode) {
    if (weights.out_size * weights.in_size < 1 << 20) return TINY_NOTE;  // key, value, update rate, decay projections
    if (!decode) return "qmm_t (64 x 64 tiles): dequantizing W costs ~10% next to the bf16 GEMM";
    if (rows == 1) return "MLX qmv_fast";
    return rows < 13 ? "MLX qmv_wide: known ~40% at M = 8-16, MLX the same; faster kernel breaks bit-exactness"
                     : "MLX qmm_t_splitk: known ~40% at M = 8-16, MLX the same; faster kernel breaks bit-exactness";
}

void GpuOpsTest::bench_multiply(const std::string& name, const QuantizedWeights& weights, int rows, bool decode) {
    const std::vector<QuantizedWeights> copies = rotated_copies(weights);
    const double weight_bytes =
        (double)weights.out_size * weights.in_size / 2 + 2.0 * weights.out_size * weights.in_size / 64 * ELEM;
    const double bytes = weight_bytes + (double)rows * (weights.in_size + weights.out_size) * ELEM;
    const double time = gpu_seconds(
        [&](int repeat) { ops.multiply_weights(copies[repeat % copies.size()], {input}, {output}, rows); });
    print_row(name, std::to_string(weights.in_size) + " -> " + std::to_string(weights.out_size), time, bytes,
              2.0 * rows * weights.out_size * weights.in_size, multiply_note(weights, rows, decode));
}

// Norms: their activations stay in cache in a real step too, so no rotation
void GpuOpsTest::bench_norms(int rows) {
    const int value_heads = model.linear_value_heads, value_dim = model.linear_value_dim;
    double time = gpu_seconds([&](int) { ops.normalize(full_layer.attention_norm, {input}, {output}, rows, model.hidden); });
    print_row("rms_norm", std::to_string(rows) + " x " + std::to_string(model.hidden), time,
              2.0 * rows * model.hidden * ELEM, 0, CACHED_NOTE);
    time = gpu_seconds([&](int) {
        ops.normalize_and_gate(linear_layer.output_norm, {input}, {ws.output_gate}, {output}, rows * value_heads, value_dim);
    });
    print_row("gated_rms_norm (2 dispatches)", std::to_string(rows * value_heads) + " x " + std::to_string(value_dim),
              time, 3.0 * rows * value_heads * value_dim * ELEM, 0, CACHED_NOTE);
}

// apply_attention: rows on their own pages / state, rotated over sets and layers
void GpuOpsTest::bench_attention(int rows, bool decode) {
    const int heads = model.query_heads, key_value_heads = model.key_value_heads, head_dim = model.head_dim;
    const int value_heads = model.linear_value_heads, value_dim = model.linear_value_dim;
    const auto full_layers = model.full_layers(), linear_layers = model.linear_layers();
    const double key_value_bytes_per_token = (double)key_value_heads * head_dim * 2 * ELEM;  // K + V, one layer
    const double state_bytes = 2.0 * (memory.conv_bytes + memory.delta_bytes);  // one row's state, read + written
    // per token: decay, kᵀS, update, qᵀS
    const double linear_flops_per_token = 7.0 * value_heads * value_dim * model.linear_key_dim;
    const int requests = decode ? rows : 1, tokens = decode ? 1 : rows;
    ws.request_count = requests, ws.tokens_per_request = tokens;
    for (int context : decode ? std::vector<int>{512, 2048, 8192} : std::vector<int>{512, 4096}) {
        const double key_value_bytes = (double)requests * context * key_value_bytes_per_token;
        const double activation_bytes = (double)requests * tokens * (3 * heads + 2 * key_value_heads) * head_dim * ELEM;
        const int sets = make_table_sets(requests, context - tokens, tokens, false, key_value_bytes, (int)full_layers.size());
        const double time = gpu_seconds([&](int repeat) {
            ws.step_table = table_sets[repeat % sets];
            ops.apply_attention(full_layers[(repeat / sets) % full_layers.size()], ws);
        });
        print_row("attn full, " + std::to_string(context) + " keys",
                  decode ? std::to_string(requests) + " rows" : "512 queries", time, key_value_bytes + activation_bytes,
                  decode ? 4.0 * requests * heads * head_dim * context : 4.0 * heads * head_dim * tokens * context,
                  decode ? "5 dispatches (rope q, rope k, write_kv, sdpa_vector, gate): their floor dominates short contexts"
                         : "MLX's unfused fallback (gather, 2 GEMMs, softmax over S x keys): no fused path for D 256");
    }
    const double activation_bytes = (double)requests * tokens * (model.conv_dim() + 2 * value_heads * value_dim) * ELEM;
    // state: read + written
    const int sets = make_table_sets(requests, 0, tokens, true, requests * state_bytes / 2, (int)linear_layers.size());
    const double time = gpu_seconds([&](int repeat) {
        ws.step_table = table_sets[repeat % sets];
        ops.apply_attention(linear_layers[(repeat / sets) % linear_layers.size()], ws);
    });
    print_row("attn linear (conv, gated delta)", decode ? std::to_string(requests) + " rows" : "512 tokens", time,
              requests * state_bytes + activation_bytes, (double)requests * tokens * linear_flops_per_token,
              decode ? "5 dispatches (conv + norm + scale, gates, delta, gated norm x2): their floor dominates at small B"
                     : "mlx-lm's gated delta kernel walks the 512 tokens in order (128 threadgroups): latency-bound");
}

// One whole decode step in one command buffer: returns its GPU seconds, sets wall_seconds (encode + commit + wait)
double GpuOpsTest::time_decode_step(double& wall_seconds) {
    const int layers = model.n_layers();
    @autoreleasepool {
        const auto wall_start = SteadyClock::now();
        const double gpu_time = encode_and_wait(queue, ws, [&] {
            for (int layer = 0; layer <= layers; layer++) {
                ops.apply_weights(layer, ws);
                if (layer < layers) ops.apply_attention(layer, ws);
            }
            ops.pick_next_tokens(ws);
        }, true);
        wall_seconds = seconds(SteadyClock::now() - wall_start);
        return gpu_time;
    }
}

// A whole decode step for `requests` rows at 512 keys each, median of 10 (after 3 warm-up steps); mlx_baseline: mlx-lm's
// tok/s at this batch size, 0 when not measured
void GpuOpsTest::bench_whole_decode_step(int requests, double mlx_baseline) {
    ws.request_count = requests, ws.tokens_per_request = 1;
    const uint32_t pages = memory.units_for(512);
    units.assign(requests, {});
    std::vector<StepTables::Row> rows;
    for (uint32_t request = 0, next = 0; request < (uint32_t)requests; request++) {
        for (uint32_t j = 0; j < pages + memory.state_units; j++) units[request].push_back(next++);
        rows.push_back({511, 1, std::span(units[request]).first(pages), std::span(units[request]).subspan(pages)});
    }
    ws.step_table = table_sets[0];
    tables.write(ws.step_table, rows);
    std::vector<double> gpu_times, wall_times;
    size_t dispatches_before = ops.dispatch_count, dispatches_per_step = 0;
    for (int step = 0; step < 13; step++) {
        double wall_time;
        const double gpu_time = time_decode_step(wall_time);
        if (!step) dispatches_per_step = ops.dispatch_count - dispatches_before;
        if (step >= 3) gpu_times.push_back(gpu_time), wall_times.push_back(wall_time);
    }
    std::sort(gpu_times.begin(), gpu_times.end()), std::sort(wall_times.begin(), wall_times.end());
    const double step_time = gpu_times[gpu_times.size() / 2], wall_time = wall_times[wall_times.size() / 2];
    // the sampling kernel alone on this step's scores (greedy: its argmax pass only, already in the step)
    std::mt19937_64 rng(1);
    const std::vector<SampleParams> params(requests, SampleParams{0.7f, 0.9f, &rng});
    const double sampling_time = gpu_seconds(
        [&](int) { ops.sample_tokens({ws.next_token_scores}, {ws.next_token_ids}, model.vocab_size, params); });
    // mlx-lm's figure is end to end, so the ratio uses wall time (greedy)
    printf("  B = %2d: %zu dispatches; GPU %6.2f ms (%5.1f tok/s); wall %6.2f ms (%5.1f tok/s)", requests,
           dispatches_per_step, step_time * 1e3, requests / step_time, wall_time * 1e3, requests / wall_time);
    if (mlx_baseline) printf(" vs mlx-lm %5.1f: %.2fx", mlx_baseline, requests / wall_time / mlx_baseline);
    printf("; GPU sampling (temp 0.7, top_p 0.9) %.3f ms/step\n", sampling_time * 1e3);
}

// A speculative verify (docs/speculative.md): one request, `tokens` rows ending at key 512 through the prefill path,
// then lm_head + argmax on every row; median of 10 GPU times (after 3 warm-up steps)
void GpuOpsTest::bench_verify_step(int tokens) {
    const int layers = model.n_layers();
    const uint32_t pages = memory.units_for(512);
    units.assign(1, {});
    for (uint32_t j = 0; j < pages + memory.state_units; j++) units[0].push_back(j);
    ws.step_table = table_sets[0];
    tables.write(ws.step_table, {{512 - tokens, tokens, std::span(units[0]).first(pages), std::span(units[0]).subspan(pages)}});
    std::vector<double> gpu_times;
    for (int step = 0; step < 13; step++) {
        @autoreleasepool {
            const double gpu_time = encode_and_wait(queue, ws, [&] {
                ws.request_count = 1, ws.tokens_per_request = tokens;
                for (int layer = 0; layer <= layers; layer++) {
                    ops.apply_weights(layer, ws);
                    if (layer < layers) ops.apply_attention(layer, ws);
                }
                ws.request_count = tokens, ws.tokens_per_request = 1;  // lm_head + argmax on every row
                ops.pick_next_tokens(ws);
            }, true);
            if (step >= 3) gpu_times.push_back(gpu_time);
        }
    }
    std::sort(gpu_times.begin(), gpu_times.end());
    printf("  verify %d tokens: GPU %6.2f ms\n", tokens, gpu_times[gpu_times.size() / 2] * 1e3);
}

void GpuOpsTest::bench_ops(const std::string& model_name) {
    const ModelSpec model = spec::load(model_name);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    Weights weights(model.path, device);
    const int chunk = 512, max_batch = 16, max_context = 8192;
    // 27B: 16 rows of 8k
    Memory memory(model, device, std::max<size_t>((size_t)2 << 30, Memory::arena_for(model, max_batch, max_context)));
    GpuOps ops(memory, weights);
    Workspace ws = ops.workspace(chunk, max_batch, max_context);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    StepTables tables = memory.step_tables(chunk, max_context);
    std::vector<id<MTLBuffer>> table_sets;
    for (int i = 0; i < 16; i++) table_sets.push_back(memory.buffer(tables.words * 4));
    const GpuOps::Layer& full_layer = ops.layers[model.full_layers()[0]];
    const GpuOps::Layer& linear_layer = ops.layers[model.linear_layers()[0]];
    const int max_in_size = std::max(
        {full_layer.feed_forward_down.in_size, full_layer.attention_output.in_size, linear_layer.attention_output.in_size});
    id<MTLBuffer> input = memory.buffer((size_t)chunk * max_in_size * ELEM);
    id<MTLBuffer> output = memory.buffer(std::max(ws.next_token_scores.length, (size_t)chunk * model.conv_dim() * ELEM));
    id<MTLBuffer> pool = memory.buffer(96 << 20);
    memory.freeze();
    GpuOpsTest bench{model, memory, ops, ws, queue, tables, table_sets, input, output, pool, full_layer, linear_layer};

    printf("bench-ops %s\nroofline: BW %.0f GB/s, FP16 %.1f TFLOPS (tools/roofline.py); bound = max(bytes / BW, "
           "flops / FP16);\nweights, K/V and state rotated over >= %zu MB; ! = above the %.0f GB/s peak\n",
           model.path.c_str(), EFFECTIVE_BANDWIDTH / 1e9, FP16_FLOPS / 1e12, ROTATION_BYTES >> 20, PEAK_BANDWIDTH / 1e9);
    // warm the GPU (the first shapes run 2x slow otherwise)
    for (auto start = SteadyClock::now(); SteadyClock::now() - start < std::chrono::milliseconds(300);)
        bench.gpu_seconds([&](int) { ops.multiply_weights(ops.lm, {ws.residual}, {output}, 1); });

    // projections, deduplicated by shape (names as in the model file)
    std::vector<std::pair<std::string, QuantizedWeights>> projections;
    auto add_projection = [&](const char* name, const QuantizedWeights& weights) {
        for (auto& [names, seen] : projections)
            if (seen.out_size == weights.out_size && seen.in_size == weights.in_size)
                return (void)(names += std::string("/") + name);
        projections.push_back({name, weights});
    };
    add_projection("q_proj", full_layer.query_projection);
    add_projection("k_proj", full_layer.key_projection);
    add_projection("v_proj", full_layer.value_projection);
    add_projection("o_proj", full_layer.attention_output);
    add_projection("in_proj_qkv", linear_layer.query_key_value_projection);
    add_projection("in_proj_z", linear_layer.output_gate_projection);
    add_projection("in_proj_b", linear_layer.update_rate_projection);
    add_projection("in_proj_a", linear_layer.decay_projection);
    add_projection("out_proj", linear_layer.attention_output);
    add_projection("gate_proj", full_layer.feed_forward_gate);
    add_projection("up_proj", full_layer.feed_forward_up);
    add_projection("down_proj", full_layer.feed_forward_down);

    for (int rows : {1, 4, 8, 16, 512}) {
        const bool decode = rows <= max_batch;
        const std::string title = decode ? "decode B = " + std::to_string(rows) : "prefill S = 512 (one row)";
        printf("\n%s\n  %-34s %-14s %9s %8s %9s %6s  %s\n", title.c_str(), "op", "shape", "us", "GB/s", "bound us",
               "%bound", "note (below 70% or !)");
        for (auto& [name, projection] : projections) bench.bench_multiply(name, projection, rows, decode);
        if (decode) bench.bench_multiply("lm_head", ops.lm, rows, decode);
        bench.bench_norms(rows);
        bench.bench_attention(rows, decode);
        if (!decode) continue;
        const double time =
            bench.gpu_seconds([&](int) { ops.greedy_tokens({ws.next_token_scores}, {output}, rows, model.vocab_size); });
        print_row("argmax", std::to_string(rows) + " x " + std::to_string(model.vocab_size), time,
                  (double)rows * model.vocab_size * ELEM, 0,
                  "one threadgroup per row: B threadgroups can't reach full bandwidth");
    }

    // a whole decode step at 512 keys: apply_weights(0 .. layers) with apply_attention of both kinds, then
    // pick_next_tokens; one command buffer
    printf("\nwhole decode step (embed .. lm_head + argmax in one command buffer), 512 keys per row; "
           "GPU sampling at temp 0.7, top_p 0.9 timed alone; median of 10 steps\n");
    // mlx-lm tok/s at B = 1 / 4 / 8 / 16 by model (reports/mlx-lm baseline.md; 27B: B = 1 from two quiet holds,
    // B = 4 / 8 from one busy run); 0: not measured, no ratio
    const bool is_0_8b = model.path.find("Qwen3.5-0.8B-4bit") != std::string::npos;
    const bool is_27b = model.path.find("Qwen3.6-27B-4bit") != std::string::npos;
    const double mlx_baseline[] = {is_0_8b ? 307 : is_27b ? 15.5 : 0, is_0_8b ? 780 : is_27b ? 35.1 : 0,
                                   is_0_8b ? 898 : is_27b ? 37.1 : 0, is_0_8b ? 1064.0 : 0};
    const int batch_sizes[] = {1, 4, 8, 16};
    for (int i = 0; i < 4; i++) bench.bench_whole_decode_step(batch_sizes[i], mlx_baseline[i]);

    printf("\nspeculative verify: one request, k + 1 tokens ending at key 512 (docs/speculative.md)\n");
    for (int tokens = 1; tokens <= 7; tokens++) bench.bench_verify_step(tokens);
}

}  // namespace mllm

// ENGINE: --arena-gib and --max-ctx, for generate / chat / bench / serve (0: not given)
static double arena_gib = 0;
static int max_context = 0;
static const char* program_name = "mllm";  // argv[0]'s name: mmmllm through the package
// prefill chunk: ANE blocks run 1024-token chunks (1.5x GPU prefill at 27B vs 1.3x at 512,
// target-27b.md §5; shorter ones run the 512 block). Fixed at 1024 for this project's M4 Pro (user, 2026-11-25).
static int prefill_chunk(const std::string& prefill_on) {
    return prefill_on == "ane" ? 1024 : 512;
}

// The engine's sizes on this Mac (budget.h): the flags given, the rest the defaults shrunk to fit the GPU's working
// set ($MLLM_GPU_BUDGET_GB in GB overrides it, to try a smaller Mac's). Defaults by unit size: 2 MiB units (27B) get
// target-27b.md's 12 GiB arena and max_ctx 32k, smaller models the command's arena and 16k. Throws before loading.
static mllm::EngineSizes fit(const std::string& model, const std::string& prefill_on, double default_gib,
                             int max_batch, bool batch_given) {
    const mllm::ModelSpec spec = mllm::spec::load(model);
    const size_t unit = mllm::Memory::unit_of(spec);
    const bool big = unit >= (2u << 20);
    size_t weights = 0;  // what Weights loads (the headers only are read here)
    for (const auto& [name, tensor] : mllm::Safetensors(spec.path).tensors)
        if (name.starts_with("language_model.")) weights += tensor.nbytes;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    size_t budget = device.recommendedMaxWorkingSetSize;
    if (const char* budget_gb = getenv("MLLM_GPU_BUDGET_GB")) {
        char* end;
        const double gb = strtod(budget_gb, &end);
        if (end == budget_gb || *end || !(gb > 0 && gb < 1e6))
            throw std::invalid_argument(std::string("MLLM_GPU_BUDGET_GB: a positive number of GB, not '") + budget_gb + "'");
        budget = (size_t)(gb * 1e9);
    }
    const mllm::MemoryPlan plan = mllm::plan_memory(
        spec, mllm::gpu_class(device.architecture.name.UTF8String), prefill_chunk(prefill_on), prefill_on == "ane",
        weights, unit, mllm::Memory::state_units_of(spec), budget,
        {(size_t)((big ? 12 : default_gib) * (1 << 30)), big ? 32768 : 16384, max_batch},
        {(size_t)(arena_gib * (1 << 30)), max_context, batch_given ? max_batch : 0});
    if (!plan.error.empty()) throw std::runtime_error(plan.error);
    if (!plan.lowered.empty())
        fprintf(stderr, "%s: %s, to fit this Mac's GPU (%.1f GB usable)\n", program_name, plan.lowered.c_str(),
                budget / 10 * 9 / 1e9);
    return plan.sizes;
}

// mllm generate: the prompt as one user message (chat template), text streamed to stdout. logits_path: each decode
// step's logits, vocab_size bf16 a step, appended raw (mmmllm compare checks them against mlx-lm's)
static void generate(const std::string& model, const std::string& prompt, mllm::GenParams params,
                     const std::string& prefill_on, bool prefix_cache, const std::string& logits_path) {
    const mllm::EngineSizes sizes = fit(model, prefill_on, 1, 16, false);
    mllm::Engine engine(model, sizes.arena, prefill_chunk(prefill_on), 32, sizes.max_batch, sizes.max_context,
                        prefill_on, prefix_cache);
    std::unique_ptr<FILE, int (*)(FILE*)> logits(logits_path.empty() ? nullptr : fopen(logits_path.c_str(), "wb"),
                                                 fclose);
    if (!logits_path.empty() && !logits) throw std::runtime_error("can't write " + logits_path);
    if (logits)
        engine.gpu.on_scores = [&](const uint16_t* scores, size_t count) { fwrite(scores, 2, count, logits.get()); };
    engine.generate({{"user", prompt}}, params, [](const std::string& text) {
        fputs(text.c_str(), stdout);
        fflush(stdout);
        return true;
    });
    printf("\n");
}

// mllm chat: a REPL over one engine. The history is kept as tokens (Tokenizer::continue_chat) and a finished reply's
// pages and state stay in the prefix cache, so a turn prefills only the new message. Ctrl-C stops a reply (the turn
// is dropped), at the prompt it quits, as does Ctrl-D.
static std::atomic<bool> interrupted;

struct Reply {
    std::vector<int> tokens;
    int cached_tokens = 0;
    SteadyClock::time_point first_token_at;
};

// Steps the engine, printing the request's text as it comes, until it is done or Ctrl-C
static Reply stream_reply(mllm::Engine& engine, int request_id) {
    Reply reply;
    for (bool done = false; !done && !interrupted;)
        for (const mllm::Event& event : engine.step()) {
            if (event.req_id != request_id) continue;
            if (event.token < 0) throw std::runtime_error("request aborted");
            if (reply.tokens.empty()) reply.first_token_at = SteadyClock::now();
            reply.tokens.push_back(event.token);
            reply.cached_tokens = event.cached, done = event.done;
            fputs(event.text.c_str(), stdout);
            fflush(stdout);
        }
    return reply;
}

static void chat(const std::string& model, mllm::GenParams params, bool default_max_tokens,
                 const std::string& prefill_on, bool prefix_cache) {
    const mllm::EngineSizes sizes = fit(model, prefill_on, 1, 16, false);
    mllm::Engine engine(model, sizes.arena, prefill_chunk(prefill_on), 32, sizes.max_batch, sizes.max_context,
                        prefill_on, prefix_cache);
    const bool tty = isatty(0);
    if (tty) fprintf(stderr, "%s chat: Ctrl-C stops a reply, Ctrl-D quits\n", program_name);
    std::vector<int> history;  // the last prompt and its reply
    for (std::string line;;) {
        if (tty) fputs(">>> ", stdout), fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        const auto start = SteadyClock::now();
        std::vector<int> tokens = history.empty() ? engine.tokenizer.encode_chat({{"user", line}})
                                                  : engine.tokenizer.continue_chat(history, line);
        // the history (reasoning kept) fills max_ctx in a few long turns: then a new conversation starts with this line
        const int least_new_tokens = default_max_tokens ? 1 : params.max_tokens;
        if (!history.empty() && (int)tokens.size() + least_new_tokens > engine.context_limit()) {
            fprintf(stderr, "%s: context full (%zu tokens): starting a new conversation\n", program_name,
                    tokens.size());
            history.clear();
            tokens = engine.tokenizer.encode_chat({{"user", line}});
        }
        const int prompt_tokens = tokens.size();
        if (default_max_tokens) params.max_tokens = std::max(1, std::min(4096, engine.context_limit() - prompt_tokens));
        int request_id;
        try {
            request_id = engine.submit(tokens, params);
        } catch (const std::invalid_argument& x) {
            fprintf(stderr, "%s: %s\n", program_name, x.what());
            continue;
        }
        interrupted = false;
        signal(SIGINT, [](int) { interrupted = true; });
        const Reply reply = stream_reply(engine, request_id);
        signal(SIGINT, SIG_DFL);
        const auto end = SteadyClock::now();
        if (interrupted) engine.cancel(request_id);
        // the step that fed its last token: only after it is the reply in the prefix cache for the next turn
        while (engine.requests.count(request_id)) engine.step();
        if (interrupted) {
            fprintf(stderr, "\n[stopped]\n");
            continue;
        }
        history = tokens;
        history.insert(history.end(), reply.tokens.begin(), reply.tokens.end());
        const int count = reply.tokens.size();
        fprintf(stderr, "\n[prompt %d tokens, %d cached, first token %.2f s; %d tokens at %.1f tok/s]\n", prompt_tokens,
                reply.cached_tokens, seconds(reply.first_token_at - start), count,
                count > 1 ? (count - 1) / seconds(end - reply.first_token_at) : 0.0);
    }
}

// mllm bench: mlx_lm.benchmark's protocol and fields. Random prompt tokens, EOS ignored, one warmup run, n trials.
// prompt_tps = B * P / time until every request has its first token (prefill + first decode step, as mlx-lm);
// generation_tps = tokens emitted after that / the time after it (B = 1: G - 1 tokens, mlx-lm counts G).
// peak_memory = Metal allocations (weights, workspaces, the arena's peak in-use units; not its unused rest).
// stream > 0 (not in mlx-lm): meanwhile keep one more request of `stream` random tokens prefilling (max_tokens 1, a new
// one as each finishes), left out of the fields above; stream_tps = its prompt tokens done / the trial's time. Measures
// what a concurrent prefill costs decode (architecture.md "Concurrency").
struct Trial {
    double prompt_tps, generation_tps, peak_memory, total_time, stream_tps;
};

static Trial run_trial(mllm::Engine& engine, const std::vector<std::vector<int>>& prompts,
                       const std::vector<int>& stream_prompt, int generation_tokens) {
    const mllm::Arena& arena = engine.mem.arena;
    const long batch_size = prompts.size(), stream_tokens = stream_prompt.size();
    const auto start = SteadyClock::now();
    std::map<int, int> tokens_by_request;
    for (const auto& prompt : prompts) tokens_by_request[engine.submit(prompt, {.max_tokens = generation_tokens})] = 0;
    int stream_id = -1;
    long streamed = 0;
    uint32_t peak_units = 0;
    SteadyClock::time_point all_started_at;
    long tokens_after = 0, started = 0;
    auto any_live = [&] {
        return std::any_of(tokens_by_request.begin(), tokens_by_request.end(),
                           [&](auto& entry) { return engine.requests.count(entry.first); });
    };
    while (any_live()) {
        if (stream_tokens && !engine.requests.count(stream_id)) {
            streamed += stream_id < 0 ? 0 : stream_tokens;
            stream_id = engine.submit(stream_prompt, {.max_tokens = 1});
        }
        for (const mllm::Event& event : engine.step()) {
            if (!tokens_by_request.count(event.req_id)) continue;
            if (tokens_by_request[event.req_id]++ == 0 && ++started == batch_size) all_started_at = SteadyClock::now();
            else if (started == batch_size) tokens_after++;
        }
        peak_units = std::max(peak_units, arena.n_units - arena.free_units);
    }
    const auto end = SteadyClock::now();
    if (stream_id >= 0) engine.cancel(stream_id);
    while (!engine.requests.empty()) engine.step();
    const double memory =
        (double)engine.device.currentAllocatedSize - engine.mem.arena_buffer.length + peak_units * arena.unit_bytes;
    return Trial{(double)batch_size * prompts[0].size() / seconds(all_started_at - start),
                 tokens_after / seconds(end - all_started_at), memory / 1e9, seconds(end - start),
                 streamed / seconds(end - start)};
}

static void bench(const std::string& model, int prompt_tokens, int generation_tokens, int batch_size, int trials,
                  const std::string& prefill_on, int stream_tokens, bool prefix_cache) {
    const mllm::EngineSizes sizes = fit(model, prefill_on, 1, 16, false);
    mllm::Engine engine(model, sizes.arena, prefill_chunk(prefill_on), 32, sizes.max_batch, sizes.max_context,
                        prefill_on, prefix_cache);
    engine.tokenizer.eos.clear();  // as mlx_lm.benchmark: never stop early
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> random_token(0, engine.spec.vocab_size - 1);
    std::vector<std::vector<int>> prompts(batch_size, std::vector<int>(prompt_tokens));
    for (auto& prompt : prompts)
        for (int& token : prompt) token = random_token(rng);
    std::vector<int> stream_prompt(stream_tokens);
    for (int& token : stream_prompt) token = random_token(rng);
    printf("Running warmup..\n");
    run_trial(engine, prompts, stream_prompt, generation_tokens);
    printf("Timing with prompt_tokens=%d, generation_tokens=%d, batch_size=%d.\n", prompt_tokens, generation_tokens,
           batch_size);
    Trial sum{};
    for (int i = 0; i < trials; i++) {
        Trial trial = run_trial(engine, prompts, stream_prompt, generation_tokens);
        printf("Trial %d:  prompt_tps=%.3f, generation_tps=%.3f, peak_memory=%.3f, total_time=%.3f", i + 1,
               trial.prompt_tps, trial.generation_tps, trial.peak_memory, trial.total_time);
        if (stream_tokens) printf(", stream_tps=%.1f", trial.stream_tps);
        printf("\n");
        fflush(stdout);
        sum = {sum.prompt_tps + trial.prompt_tps, sum.generation_tps + trial.generation_tps,
               sum.peak_memory + trial.peak_memory, 0, sum.stream_tps + trial.stream_tps};
    }
    printf("Averages: prompt_tps=%.3f, generation_tps=%.3f, peak_memory=%.3f", sum.prompt_tps / trials,
           sum.generation_tps / trials, sum.peak_memory / trials);
    if (stream_tokens) printf(", stream_tps=%.1f", sum.stream_tps / trials);
    printf("\n");
}

// mllm serve: /v1/completions on 127.0.0.1 until killed. A 2 GiB arena: 32 requests of 512 + 256 tokens need 0.9.
static int serve(const std::string& model, int port, const std::string& prefill_on, int max_batch, bool batch_given,
                 bool prefix_cache) {
    const mllm::EngineSizes sizes = fit(model, prefill_on, 2, max_batch, batch_given);
    max_batch = sizes.max_batch;
    mllm::Engine engine(model, sizes.arena, prefill_chunk(prefill_on), 32, max_batch, sizes.max_context, prefill_on,
                        prefix_cache);
    mllm::Server server(engine, model);
    fprintf(stderr, "%s serve: http://127.0.0.1:%d/v1 (chat/completions, completions; prefill on %s, max batch %d, "
            "prefix cache %s)\n", program_name, port, prefill_on.c_str(), max_batch, prefix_cache ? "on" : "off");
    if (!server.listen("127.0.0.1", port)) throw std::runtime_error("can't listen on port " + std::to_string(port));
    return 0;
}

static int usage() {
    fprintf(stderr,
            "usage: %1$s MODEL                                print the model spec\n"
            "       %1$s generate MODEL PROMPT [--max-tokens N] [--temp T] [--top-p P] [--logits FILE] [PREFILL]\n"
            "         --logits FILE: each step's logits (vocab bf16 a step) appended to FILE\n"
            "       %1$s chat MODEL [--max-tokens N] [--temp T] [--top-p P] [PREFILL]   multi-turn chat in the terminal\n"
            "       %1$s bench MODEL [-p 512] [-g 1024] [-b 1] [-n 5] [--stream P] [PREFILL]   mlx_lm.benchmark's\n"
            "         fields; --stream P: a P-token prefill runs alongside throughout, its own tok/s reported\n"
            "       %1$s serve MODEL [--port 8080] [--max-batch 32] [PREFILL]   OpenAI-style /v1/chat/completions and\n"
            "         /v1/completions on 127.0.0.1\n"
            "       %1$s bench-ops MODEL                      GPU time per op and per decode step, next to the roofline\n"
            "MODEL: repo id under $HF_HOME, or a local dir\n"
            "PREFILL: [--prefill gpu|ane]: prefill's matmuls on the Neural Engine, the GPU feeding it its own weights a\n"
            "         layer at a time (Core ML blocks: mmmllm[ane] builds them first), 1024-token chunks\n"
            "         [--prefix-cache on|off]: reuse prompt prefixes across requests (default on; off for bench, whose\n"
            "         trials repeat the same prompts)\n"
            "ENGINE (generate, chat, bench, serve): [--arena-gib G] (default 1, serve 2; 12 at 2 MiB units, the 27B)\n"
            "         [--max-ctx N] (default 16384; 32768 at 2 MiB units); defaults (and serve's max batch) lowered to\n"
            "         fit the GPU's working set\n", program_name);
    return 2;
}

// Runs one command; its flags after the positional arguments, each taken out of flags as it is read
static int run_command(int argc, char** argv, std::map<std::string, std::string>& flags) {
    const std::string command = argv[1];
    auto take_flag = [&](const char* name, const char* default_value) {
        auto found = flags.find(name);
        std::string value = found == flags.end() ? default_value : found->second;
        if (found != flags.end()) flags.erase(found);
        return value;
    };
    auto take_prefix_cache = [&](const char* default_value) {
        const std::string value = take_flag("--prefix-cache", default_value);
        if (value != "on" && value != "off") throw std::invalid_argument("--prefix-cache: on or off, not " + value);
        return value == "on";
    };
    arena_gib = std::stod(take_flag("--arena-gib", "0"));
    max_context = std::stoi(take_flag("--max-ctx", "0"));
    if (arena_gib < 0 || max_context < 0) return usage();
    if (command == "generate") {
        mllm::GenParams params{std::stoi(take_flag("--max-tokens", "256")), std::stof(take_flag("--temp", "0")),
                               std::stof(take_flag("--top-p", "0"))};
        const std::string prefill_on = take_flag("--prefill", "gpu");
        const bool prefix_cache = take_prefix_cache("on");
        const std::string logits_path = take_flag("--logits", "");
        if (!flags.empty()) return usage();
        generate(argv[2], argv[3], params, prefill_on, prefix_cache, logits_path);
    } else if (command == "chat") {
        const bool default_max_tokens = !flags.count("--max-tokens");
        mllm::GenParams params{std::stoi(take_flag("--max-tokens", "1")), std::stof(take_flag("--temp", "0")),
                               std::stof(take_flag("--top-p", "0"))};
        const std::string prefill_on = take_flag("--prefill", "gpu");
        const bool prefix_cache = take_prefix_cache("on");
        if (!flags.empty()) return usage();
        chat(argv[2], params, default_max_tokens, prefill_on, prefix_cache);
    } else if (command == "bench") {
        const int prompt_tokens = std::stoi(take_flag("-p", "512")), generation_tokens = std::stoi(take_flag("-g", "1024"));
        const int batch_size = std::stoi(take_flag("-b", "1")), trials = std::stoi(take_flag("-n", "5"));
        const std::string prefill_on = take_flag("--prefill", "gpu");
        const int stream_tokens = std::stoi(take_flag("--stream", "0"));
        const bool prefix_cache = take_prefix_cache("off");
        if (!flags.empty() || prompt_tokens < 2 || generation_tokens < 1 || batch_size < 1 || trials < 1 ||
            stream_tokens < 0 || stream_tokens == 1)
            return usage();
        bench(argv[2], prompt_tokens, generation_tokens, batch_size, trials, prefill_on, stream_tokens, prefix_cache);
    } else if (command == "serve") {
        const bool batch_given = flags.count("--max-batch");
        const int port = std::stoi(take_flag("--port", "8080")), max_batch = std::stoi(take_flag("--max-batch", "32"));
        const std::string prefill_on = take_flag("--prefill", "gpu");
        const bool prefix_cache = take_prefix_cache("on");
        if (!flags.empty() || max_batch < 1) return usage();
        return serve(argv[2], port, prefill_on, max_batch, batch_given, prefix_cache);
    } else if (command == "bench-ops") {
        if (!flags.empty()) return usage();
        mllm::GpuOpsTest::bench_ops(argv[2]);
    } else {
        if (argc != 2) return usage();
        mllm::ModelSpec model = mllm::spec::load(argv[1]);
        printf("%s\n%d layers (%zu full, %zu linear), hidden %d, vocab %d\n", model.path.c_str(), model.n_layers(),
               model.full_layers().size(), model.linear_layers().size(), model.hidden, model.vocab_size);
    }
    return 0;
}

int main(int argc, char** argv) {
    if (const char* slash = strrchr(argv[0], '/')) program_name = slash + 1;
    else program_name = argv[0];
    if (argc < 2) return usage();
    const std::string command = argv[1];
    // flags after the positional arguments: name -> value
    std::map<std::string, std::string> flags;
    const bool takes_model = command == "bench" || command == "bench-ops" || command == "serve" || command == "chat";
    const int positional_count = command == "generate" ? 4 : takes_model ? 3 : 2;
    if (argc < positional_count || (argc - positional_count) % 2) return usage();
    for (int i = positional_count; i < argc; i += 2) flags[argv[i]] = argv[i + 1];
    try {
        return run_command(argc, argv, flags);
    } catch (const std::exception& e) {
        fprintf(stderr, "%s: %s\n", program_name, e.what());
        return 1;
    }
}
