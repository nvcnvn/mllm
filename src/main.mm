// mllm CLI. `mllm MODEL`: print the model spec. `mllm bench-ops MODEL`: GPU time per op and per decode step.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "gpu.h"
#include "spec.h"

namespace mllm {

// tools/roofline.py calibration knobs: effective bandwidth and FP16 rate; 273 GB/s is the M4 Pro's peak
constexpr double BW = 210e9, PEAK_BW = 273e9, FLOPS = 6.0e12;
constexpr size_t POOL = 64 << 20;  // rotate weights / K/V / state over at least this, so reads come from DRAM

static void row(const std::string& op, const std::string& shape, double sec, double bytes, double flops,
                const char* note) {
    const double bound = std::max(bytes / BW, flops / FLOPS), gbs = bytes / sec / 1e9, pct = 100 * bound / sec;
    printf("  %-34s %-14s %9.1f %7.1f%s %9.1f %5.0f%%  %s\n", op.c_str(), shape.c_str(), sec * 1e6, gbs,
           gbs * 1e9 > PEAK_BW ? "!" : " ", bound * 1e6, pct, pct < 70 || gbs * 1e9 > PEAK_BW ? note : "");
}

// bench-ops needs GpuOps' kernel helpers and layers (gpu.h friend)
struct GpuOpsTest {
    static void bench_ops(const std::string& model);
};

void GpuOpsTest::bench_ops(const std::string& model) {
    const ModelSpec s = spec::load(model);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    Weights w(s.path, dev);
    const int chunk = 512, max_batch = 16, max_ctx = 8192, L = s.n_layers();
    Memory mem(s, dev, (size_t)2 << 30);
    Workspace ws = mem.workspace(chunk, max_batch);
    GpuOps ops(mem, w, chunk, max_ctx);
    StepTables tables = mem.step_tables(chunk);
    std::vector<id<MTLBuffer>> tabs;  // one per rotation set
    for (int i = 0; i < 16; i++) tabs.push_back(mem.buffer(tables.words * 4));
    const GpuOps::Layer &fl = ops.layers[s.full_layers()[0]], &ll = ops.layers[s.linear_layers()[0]];
    const int max_K = std::max({fl.down_proj.K, fl.out.K, ll.out.K});
    id<MTLBuffer> xin = mem.buffer((size_t)chunk * max_K * ELEM);
    id<MTLBuffer> yout = mem.buffer(std::max(ws.logits.length, (size_t)chunk * s.conv_dim() * ELEM));
    id<MTLBuffer> pool = mem.buffer(96 << 20);
    mem.freeze();

    // GPU seconds per encode(r) call: reps calls in one command buffer, best of 3
    auto time = [&](auto&& encode) {
        double t1;
        @autoreleasepool {
            ops.begin();
            encode(0);
            t1 = ops.commit_and_wait();
        }
        const int reps = std::clamp((int)(0.02 / std::max(t1, 1e-6)), 3, 2000);
        double best = INFINITY;
        for (int k = 0; k < 3; k++) @autoreleasepool {
                ops.begin();
                for (int r = 0; r < reps; r++) encode(r);
                best = std::min(best, ops.commit_and_wait() / reps);
            }
        return best;
    };
    // copies of l over >= POOL bytes of pool (l itself when it is that big)
    auto rotate = [&](const QLinear& l) {
        const size_t wb = (size_t)l.N * l.K / 2, sb = (size_t)l.N * l.K / 64 * ELEM;
        if (wb + 2 * sb >= POOL) return std::vector<QLinear>{l};
        auto al = [](size_t x) { return (x + 255) & ~(size_t)255; };
        const size_t n = (POOL + wb + 2 * sb - 1) / (wb + 2 * sb), stride = al(wb) + 2 * al(sb);
        assert(n * stride <= pool.length);
        std::vector<QLinear> out;
        uint8_t* p = (uint8_t*)pool.contents;
        for (size_t c = 0; c < n; c++) {
            QLinear q = l;
            const size_t o = c * stride;
            q.w = {pool, o}, q.scales = {pool, o + al(wb)}, q.biases = {pool, o + al(wb) + al(sb)};
            memcpy(p + o, (uint8_t*)l.w.buffer.contents + l.w.offset, wb);
            memcpy(p + q.scales.offset, (uint8_t*)l.scales.buffer.contents + l.scales.offset, sb);
            memcpy(p + q.biases.offset, (uint8_t*)l.biases.buffer.contents + l.biases.offset, sb);
            out.push_back(q);
        }
        return out;
    };
    // sets of B requests on their own units (pages for pos + n tokens, or a state span), one table each; enough
    // sets that set_bytes (one layer's distinct footprint per set) * sets * layers >= POOL
    const uint32_t SU = mem.state_units;
    std::vector<std::vector<uint32_t>> units;  // per request; the tables are written from these spans
    auto make_sets = [&](int B, int pos, int n, bool linear, double set_bytes, int layers) {
        const uint32_t per_row = linear ? SU : mem.units_for(pos + n);
        const int fit = std::min<int>((int)tabs.size(), mem.arena.n_units / (B * per_row));
        assert(fit >= 1);
        const int sets = std::clamp((int)std::ceil(POOL / (set_bytes * layers)), 1, fit);
        units.assign((size_t)B * sets, {});
        uint32_t next = 0;
        for (int k = 0; k < sets; k++) {
            std::vector<StepTables::Row> rows;
            for (int b = 0; b < B; b++) {
                std::vector<uint32_t>& u = units[k * B + b];
                for (uint32_t j = 0; j < per_row; j++) u.push_back(next++);
                rows.push_back({pos, n, linear ? std::span<const uint32_t>{} : u, linear ? u : std::span<const uint32_t>{}});
            }
            tables.write(tabs[k], rows);
        }
        return sets;
    };

    printf("bench-ops %s\nroofline: BW %.0f GB/s, FP16 %.1f TFLOPS (tools/roofline.py); bound = max(bytes / BW, "
           "flops / FP16);\nweights, K/V and state rotated over >= %zu MB; ! = above the %.0f GB/s peak\n",
           s.path.c_str(), BW / 1e9, FLOPS / 1e12, POOL >> 20, PEAK_BW / 1e9);
    // warm the GPU (the first shapes run 2x slow otherwise)
    for (auto t0 = std::chrono::steady_clock::now(); std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(300);)
        time([&](int) { ops.linear(ops.embed_tokens, {ws.x}, {yout}, 1); });

    // projections, deduplicated by shape
    std::vector<std::pair<std::string, QLinear>> proj;
    auto add_proj = [&](const char* name, const QLinear& q) {
        for (auto& [n, p] : proj)
            if (p.N == q.N && p.K == q.K) return (void)(n += std::string("/") + name);
        proj.push_back({name, q});
    };
    add_proj("q_proj", fl.q_proj), add_proj("k_proj", fl.k_proj), add_proj("v_proj", fl.v_proj);
    add_proj("o_proj", fl.out), add_proj("in_proj_qkv", ll.in_proj_qkv), add_proj("in_proj_z", ll.in_proj_z);
    add_proj("in_proj_b", ll.in_proj_b), add_proj("in_proj_a", ll.in_proj_a), add_proj("out_proj", ll.out);
    add_proj("gate_proj", fl.gate_proj), add_proj("up_proj", fl.up_proj), add_proj("down_proj", fl.down_proj);

    const int H = s.n_heads, Hkv = s.n_kv_heads, D = s.head_dim, Hv = s.lin_v_heads, Dv = s.lin_v_dim;
    const auto full = s.full_layers(), lin = s.linear_layers();
    const double kv_tok = (double)Hkv * D * 2 * ELEM;               // K + V bytes per token, one layer
    const double state = 2.0 * (mem.conv_bytes + mem.delta_bytes);  // one row's state, read + written
    const double gdn_flops = 7.0 * Hv * Dv * s.lin_k_dim;           // per token: decay, kᵀS, update, qᵀS
    const char* tiny = "tiny: dispatch / fill floor of a few us, not bandwidth";
    const char* cached = "activations stay in cache (no rotation, as in a real step): a few-us floor when small";
    for (int M : {1, 8, 16, 512}) {
        const bool decode = M <= max_batch;
        const std::string title = decode ? "decode B = " + std::to_string(M) : "prefill S = 512 (one row)";
        printf("\n%s\n  %-34s %-14s %9s %8s %9s %6s  %s\n", title.c_str(), "op", "shape", "us", "GB/s", "bound us",
               "%bound", "note (below 70% or !)");
        auto lin_note = [&](const QLinear& q) {
            if (q.N * q.K < 1 << 20) return tiny;  // k_proj, v_proj, in_proj_b, in_proj_a
            if (!decode) return "MLX qmm_t: 32x32 tiles, below the 6 TFLOPS knob";
            if (M == 1) return "MLX qmv_fast";
            return M < 13 ? "MLX qmv_wide: known ~40% at M = 8-16, MLX the same; faster kernel breaks bit-exactness"
                          : "MLX qmm_t_splitk: known ~40% at M = 8-16, MLX the same; faster kernel breaks bit-exactness";
        };
        auto bench_linear = [&](const std::string& name, const QLinear& q) {
            const std::vector<QLinear> rot = rotate(q);
            const double wbytes = (double)q.N * q.K / 2 + 2.0 * q.N * q.K / 64 * ELEM;
            const double bytes = wbytes + (double)M * (q.K + q.N) * ELEM;
            const double t = time([&](int r) { ops.linear(rot[r % rot.size()], {xin}, {yout}, M); });
            row(name, std::to_string(q.K) + " -> " + std::to_string(q.N), t, bytes, 2.0 * M * q.N * q.K, lin_note(q));
        };
        for (auto& [name, q] : proj) bench_linear(name, q);
        if (decode) bench_linear("lm_head (embed_tokens)", ops.embed_tokens);

        // norms: their activations stay in cache in a real step too, so no rotation
        double t = time([&](int) { ops.rms_norm(fl.input_layernorm, {xin}, {yout}, M, s.hidden); });
        row("rms_norm", std::to_string(M) + " x " + std::to_string(s.hidden), t, 2.0 * M * s.hidden * ELEM, 0, cached);
        t = time([&](int) { ops.gated_rms_norm(ll.gated_norm, {xin}, {ws.z}, {yout}, M * Hv, Dv); });
        row("gated_rms_norm (2 dispatches)", std::to_string(M * Hv) + " x " + std::to_string(Dv), t,
            3.0 * M * Hv * Dv * ELEM, 0, cached);

        // attn(i): rows on their own pages / state, rotated over sets and layers
        const int B = decode ? M : 1, S = decode ? 1 : M;
        ws.B = B, ws.S = S;
        for (int ctx : decode ? std::vector<int>{512, 2048, 8192} : std::vector<int>{512, 4096}) {
            const double kv = (double)B * ctx * kv_tok, act = (double)B * S * (3 * H + 2 * Hkv) * D * ELEM;
            const int sets = make_sets(B, ctx - S, S, false, kv, (int)full.size());
            t = time([&](int r) {
                ws.table = tabs[r % sets];
                ops.attn(full[(r / sets) % full.size()], ws);
            });
            row("attn full, " + std::to_string(ctx) + " keys", decode ? std::to_string(B) + " rows" : "512 queries", t,
                kv + act, decode ? 4.0 * B * H * D * ctx : 4.0 * H * D * S * ctx,
                decode ? "5 dispatches (rope q, rope k, write_kv, sdpa_vector, gate): their floor dominates short contexts"
                       : "MLX's unfused fallback (gather, 2 GEMMs, softmax over S x keys): no fused path for D 256");
        }
        const double lact = (double)B * S * (s.conv_dim() + 2 * Hv * Dv) * ELEM;
        const int sets = make_sets(B, 0, S, true, B * state / 2, (int)lin.size());  // state: read + written
        t = time([&](int r) {
            ws.table = tabs[r % sets];
            ops.attn(lin[(r / sets) % lin.size()], ws);
        });
        row("attn linear (conv, gated delta)", decode ? std::to_string(B) + " rows" : "512 tokens", t, B * state + lact,
            (double)B * S * gdn_flops,
            decode ? "7 dispatches (conv, norm, scale, gates, delta, gated norm x2): their floor dominates at small B"
                   : "mlx-lm's gated delta kernel walks the 512 tokens in order (128 threadgroups): latency-bound");
        if (decode) {
            t = time([&](int) { ops.argmax({ws.logits}, {yout}, M, s.vocab); });
            row("argmax", std::to_string(M) + " x " + std::to_string(s.vocab), t, (double)M * s.vocab * ELEM, 0,
                "one threadgroup per row: B threadgroups can't reach full bandwidth");
        }
    }

    // a whole decode step at 512 keys: dense(0 .. L) with attn of both kinds, lm_head; one command buffer
    printf("\nwhole decode step (embed .. lm_head + argmax in one command buffer), 512 keys per row; "
           "CPU sampling at temp 0.7, top_p 0.9; median of 10 steps\n");
    const double baseline[] = {307, 898, 1064};  // mlx-lm tok/s at B = 1 / 8 / 16
    int bi = 0;
    for (int B : {1, 8, 16}) {
        ws.B = B, ws.S = 1;
        const uint32_t pages = mem.units_for(512);
        units.assign(B, {});
        std::vector<StepTables::Row> rows;
        for (uint32_t b = 0, next = 0; b < (uint32_t)B; b++) {
            for (uint32_t j = 0; j < pages + SU; j++) units[b].push_back(next++);
            rows.push_back({511, 1, std::span(units[b]).first(pages), std::span(units[b]).subspan(pages)});
        }
        ws.table = tabs[0];
        tables.write(ws.table, rows);
        std::vector<double> ts, walls;  // GPU time; wall = encode + commit + wait
        for (int k = 0; k < 13; k++) @autoreleasepool {
                const auto w0 = std::chrono::steady_clock::now();
                ops.begin();
                for (int i = 0; i <= L; i++) {
                    ops.dense(i, ws);
                    if (i < L) ops.attn(i, ws);
                }
                ops.lm_head(ws);
                const double t = ops.commit_and_wait();
                const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                if (k >= 3) ts.push_back(t), walls.push_back(wall);
            }
        std::sort(ts.begin(), ts.end()), std::sort(walls.begin(), walls.end());
        const double step = ts[ts.size() / 2], wall = walls[walls.size() / 2];
        std::mt19937_64 rng(1);
        const std::vector<SampleParams> params(B, SampleParams{0.7f, 0.9f, &rng});
        const int draws = 5;
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < draws; k++) ops.sample(ws, params);
        const double samp = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / draws;
        // mlx-lm's figure is end to end, so the ratio uses wall time (greedy: the GPU argmax, no CPU sampling)
        printf("  B = %2d: GPU %6.2f ms (%5.0f tok/s); wall %6.2f ms (%5.0f tok/s) vs mlx-lm %4.0f: %.2fx; "
               "CPU sampling %.2f ms/row = %4.2f ms/step -> %5.0f tok/s\n", B, step * 1e3, B / step, wall * 1e3,
               B / wall, baseline[bi], B / wall / baseline[bi], samp / B * 1e3, samp * 1e3, B / (wall + samp));
        bi++;
    }
    // ponytail: sampling stays on the CPU (GpuOps::sample, ~B x per-row ms per step). Move it to the GPU once the
    // lanes (g5) can't hide it behind the next step's GPU work, i.e. when a step's sampling time exceeds ~10% of
    // the GPU step at the served batch.
}

}  // namespace mllm

int main(int argc, char** argv) {
    const bool bench = argc == 3 && std::string(argv[1]) == "bench-ops";
    if (argc != 2 && !bench) {
        fprintf(stderr, "usage: mllm MODEL             print the model spec\n"
                        "       mllm bench-ops MODEL   GPU time per op and per decode step, next to the roofline\n"
                        "MODEL: repo id under $HF_HOME, or a local dir\n");
        return 2;
    }
    try {
        if (bench) {
            mllm::GpuOpsTest::bench_ops(argv[2]);
            return 0;
        }
        mllm::ModelSpec s = mllm::spec::load(argv[1]);
        printf("%s\n%d layers (%zu full, %zu linear), hidden %d, vocab %d\n", s.path.c_str(), s.n_layers(),
               s.full_layers().size(), s.linear_layers().size(), s.hidden, s.vocab);
    } catch (const std::exception& e) {
        fprintf(stderr, "mllm: %s\n", e.what());
        return 1;
    }
}
