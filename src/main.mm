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

#include "engine.h"
#include "lanes.h"
#include "server.h"
#include "spec.h"

namespace mllm {

// tools/roofline.py calibration knobs (keep the two in step): effective bandwidth and FP16 rate; 273 GB/s is the M4
// Pro's peak
constexpr double BW = 210e9, PEAK_BW = 273e9, FLOPS = 7.5e12;
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
    Memory mem(s, dev, std::max<size_t>((size_t)2 << 30, Memory::arena_for(s, max_batch, max_ctx)));  // 27B: 16 rows of 8k
    GpuOps ops(mem, w);
    Workspace ws = ops.workspace(chunk, max_batch, max_ctx);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    StepTables tables = mem.step_tables(chunk, max_ctx);
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
            t1 = encode_and_wait(q, ws, [&] {
                ops.use(ws);
                encode(0);
            });
        }
        const int reps = std::clamp((int)(0.02 / std::max(t1, 1e-6)), 3, 2000);
        double best = INFINITY;
        for (int k = 0; k < 3; k++) @autoreleasepool {
                best = std::min(best, encode_and_wait(q, ws, [&] {
                    ops.use(ws);
                    for (int r = 0; r < reps; r++) encode(r);
                }) / reps);
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
        time([&](int) { ops.linear(ops.lm, {ws.x}, {yout}, 1); });

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
    for (int M : {1, 4, 8, 16, 512}) {
        const bool decode = M <= max_batch;
        const std::string title = decode ? "decode B = " + std::to_string(M) : "prefill S = 512 (one row)";
        printf("\n%s\n  %-34s %-14s %9s %8s %9s %6s  %s\n", title.c_str(), "op", "shape", "us", "GB/s", "bound us",
               "%bound", "note (below 70% or !)");
        auto lin_note = [&](const QLinear& q) {
            if (q.N * q.K < 1 << 20) return tiny;  // k_proj, v_proj, in_proj_b, in_proj_a
            if (!decode) return "qmm_t (64 x 64 tiles): dequantizing W costs ~10% next to the bf16 GEMM";
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
        if (decode) bench_linear("lm_head", ops.lm);

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
            decode ? "5 dispatches (conv + norm + scale, gates, delta, gated norm x2): their floor dominates at small B"
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
    // mlx-lm tok/s at B = 1 / 4 / 8 / 16 by model (reports/mlx-lm baseline.md; 27B: B = 1 from two quiet holds,
    // B = 4 / 8 from one busy run); 0: not measured, no ratio
    const bool is08 = s.path.find("Qwen3.5-0.8B-4bit") != std::string::npos;
    const bool is27 = s.path.find("Qwen3.6-27B-4bit") != std::string::npos;
    const double baseline[] = {is08 ? 307 : is27 ? 15.5 : 0, is08 ? 780 : is27 ? 35.1 : 0, is08 ? 898 : is27 ? 37.1 : 0,
                               is08 ? 1064.0 : 0};
    int bi = 0;
    for (int B : {1, 4, 8, 16}) {
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
        size_t d0 = ops.dispatches, per_step = 0;
        for (int k = 0; k < 13; k++) @autoreleasepool {
                const auto w0 = std::chrono::steady_clock::now();
                const double t = encode_and_wait(q, ws, [&] {
                    for (int i = 0; i <= L; i++) {
                        ops.dense(i, ws);
                        if (i < L) ops.attn(i, ws);
                    }
                    ops.lm_head(ws);
                }, true);
                if (!k) per_step = ops.dispatches - d0;
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
        printf("  B = %2d: %zu dispatches; GPU %6.2f ms (%5.1f tok/s); wall %6.2f ms (%5.1f tok/s)", B, per_step, step * 1e3,
               B / step, wall * 1e3, B / wall);
        if (baseline[bi]) printf(" vs mlx-lm %5.1f: %.2fx", baseline[bi], B / wall / baseline[bi]);
        printf("; CPU sampling %.2f ms/row = %4.2f ms/step -> %5.0f tok/s\n", samp / B * 1e3, samp * 1e3, B / (wall + samp));
        bi++;
    }
    // ponytail: sampling stays on the CPU (GpuOps::sample, ~B x per-row ms per step). Move it to the GPU once the
    // lanes (g5) can't hide it behind the next step's GPU work, i.e. when a step's sampling time exceeds ~10% of
    // the GPU step at the served batch.
}

}  // namespace mllm

// mllm generate: the prompt as one user message (chat template), text streamed to stdout
// ENGINE: --arena-gib (0: the command's default) and --max-ctx, for generate / bench / serve
static double arena_gib = 0;
static int max_ctx = 16384;
static const char* prog = "mllm";  // argv[0]'s name: mmmllm through the package
static size_t arena(double def_gib) { return (size_t)((arena_gib ? arena_gib : def_gib) * (1 << 30)); }

static void generate(const std::string& model, const std::string& prompt, mllm::GenParams p, const std::string& on,
                     const std::string& ane_weights, bool prefix_cache) {
    mllm::Engine e(model, arena(1), 512, 32, 16, max_ctx, on, ane_weights, prefix_cache);
    e.generate({{"user", prompt}}, p, [](const std::string& t) {
        fputs(t.c_str(), stdout);
        fflush(stdout);
        return true;
    });
    printf("\n");
}

// mllm chat: a REPL over one engine. The history is re-templated each turn; the prefix cache supplies the part the
// last turn already prefilled (up to its last full page before the reply), so a turn prefills about the last reply
// plus the new message. Ctrl-C stops a reply (the turn is dropped), at the prompt it quits, as does Ctrl-D.
static std::atomic<bool> interrupted;

static void chat(const std::string& model, mllm::GenParams p, bool default_max, const std::string& on,
                 const std::string& ane_weights, bool prefix_cache) {
    mllm::Engine e(model, arena(1), 512, 32, 16, max_ctx, on, ane_weights, prefix_cache);
    const bool tty = isatty(0);
    if (tty) fprintf(stderr, "%s chat: Ctrl-C stops a reply, Ctrl-D quits\n", prog);
    std::vector<mllm::Message> history;
    for (std::string line;;) {
        if (tty) fputs(">>> ", stdout), fflush(stdout);
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;
        history.push_back({"user", line});
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        std::vector<int> ids = e.tok.encode_chat(history);
        const int n = ids.size();
        if (default_max) p.max_tokens = std::max(1, std::min(4096, e.ctx_limit() - n));
        int rid;
        try {
            rid = e.submit(std::move(ids), p);
        } catch (const std::invalid_argument& x) {
            fprintf(stderr, "%s: %s\n", prog, x.what());
            history.pop_back();
            continue;
        }
        interrupted = false;
        signal(SIGINT, [](int) { interrupted = true; });
        std::string reply;
        int out = 0, cached = 0;
        clock::time_point t1;
        for (bool done = false; !done && !interrupted;)
            for (const mllm::Event& ev : e.step()) {
                if (ev.req_id != rid) continue;
                if (ev.token < 0) throw std::runtime_error("request aborted");
                if (!out++) t1 = clock::now();
                cached = ev.cached, done = ev.done;
                reply += ev.text;
                fputs(ev.text.c_str(), stdout);
                fflush(stdout);
            }
        signal(SIGINT, SIG_DFL);
        const auto t2 = clock::now();
        auto sec = [](auto d) { return std::chrono::duration<double>(d).count(); };
        if (interrupted) {
            e.cancel(rid);
            history.pop_back();
            fprintf(stderr, "\n[stopped]\n");
            continue;
        }
        history.push_back({"assistant", reply});
        fprintf(stderr, "\n[prompt %d tokens, %d cached, first token %.2f s; %d tokens at %.1f tok/s]\n", n, cached,
                sec(t1 - t0), out, out > 1 ? (out - 1) / sec(t2 - t1) : 0.0);
    }
}

// mllm bench: mlx_lm.benchmark's protocol and fields. Random prompt tokens, EOS ignored, one warmup run, n trials.
// prompt_tps = B * P / time until every request has its first token (prefill + first decode step, as mlx-lm);
// generation_tps = tokens emitted after that / the time after it (B = 1: G - 1 tokens, mlx-lm counts G).
// peak_memory = Metal allocations (weights, workspaces, the arena's peak in-use units; not its unused rest).
// stream > 0 (not in mlx-lm): meanwhile keep one more request of `stream` random tokens prefilling (max_tokens 1, a new
// one as each finishes), left out of the fields above; stream_tps = its prompt tokens done / the trial's time. Measures
// what a concurrent prefill costs decode (architecture.md "Concurrency").
static void bench(const std::string& model, int P, int G, int B, int n, const std::string& on,
                  const std::string& ane_weights, int stream, bool prefix_cache) {
    mllm::Engine e(model, arena(1), 512, 32, 16, max_ctx, on, ane_weights, prefix_cache);
    e.tok.eos.clear();  // as mlx_lm.benchmark: never stop early
    std::mt19937 rng(0);
    std::uniform_int_distribution<int> tok(0, e.spec.vocab - 1);
    std::vector<std::vector<int>> prompts(B, std::vector<int>(P));
    for (auto& p : prompts)
        for (int& t : p) t = tok(rng);
    std::vector<int> stream_prompt(stream);
    for (int& t : stream_prompt) t = tok(rng);
    const mllm::Arena& a = e.mem.arena;
    struct Trial {
        double prompt_tps, generation_tps, peak_memory, total_time, stream_tps;
    };
    auto trial = [&] {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        std::map<int, int> got;
        for (const auto& p : prompts) got[e.submit(p, {.max_tokens = G})] = 0;
        int sid = -1;
        long streamed = 0;
        uint32_t peak_units = 0;
        clock::time_point t_prompt;
        long after = 0, started = 0;
        while (std::any_of(got.begin(), got.end(), [&](auto& kv) { return e.reqs.count(kv.first); })) {
            if (stream && !e.reqs.count(sid)) {
                streamed += sid < 0 ? 0 : stream;
                sid = e.submit(stream_prompt, {.max_tokens = 1});
            }
            for (const mllm::Event& ev : e.step()) {
                if (!got.count(ev.req_id)) continue;
                if (got[ev.req_id]++ == 0 && ++started == B) t_prompt = clock::now();
                else if (started == B) after++;
            }
            peak_units = std::max(peak_units, a.n_units - a.free_units);
        }
        const auto t1 = clock::now();
        if (sid >= 0) e.cancel(sid);
        while (!e.reqs.empty()) e.step();
        auto sec = [](auto d) { return std::chrono::duration<double>(d).count(); };
        const double mem = (double)e.device.currentAllocatedSize - e.mem.arena_buffer.length + peak_units * a.unit_bytes;
        return Trial{(double)B * P / sec(t_prompt - t0), after / sec(t1 - t_prompt), mem / 1e9, sec(t1 - t0),
                     streamed / sec(t1 - t0)};
    };
    printf("Running warmup..\n");
    trial();
    printf("Timing with prompt_tokens=%d, generation_tokens=%d, batch_size=%d.\n", P, G, B);
    Trial sum{};
    for (int i = 0; i < n; i++) {
        Trial t = trial();
        printf("Trial %d:  prompt_tps=%.3f, generation_tps=%.3f, peak_memory=%.3f, total_time=%.3f", i + 1,
               t.prompt_tps, t.generation_tps, t.peak_memory, t.total_time);
        if (stream) printf(", stream_tps=%.1f", t.stream_tps);
        printf("\n");
        fflush(stdout);
        sum = {sum.prompt_tps + t.prompt_tps, sum.generation_tps + t.generation_tps, sum.peak_memory + t.peak_memory, 0,
               sum.stream_tps + t.stream_tps};
    }
    printf("Averages: prompt_tps=%.3f, generation_tps=%.3f, peak_memory=%.3f", sum.prompt_tps / n,
           sum.generation_tps / n, sum.peak_memory / n);
    if (stream) printf(", stream_tps=%.1f", sum.stream_tps / n);
    printf("\n");
}

// mllm serve: /v1/completions on 127.0.0.1 until killed. A 2 GiB arena: 32 requests of 512 + 256 tokens need 0.9.
static int serve(const std::string& model, int port, const std::string& on, const std::string& ane_weights,
                 int max_batch, bool prefix_cache) {
    mllm::Engine e(model, arena(2), 512, 32, max_batch, max_ctx, on, ane_weights, prefix_cache);
    mllm::Server s(e, model);
    fprintf(stderr, "%s serve: http://127.0.0.1:%d/v1 (chat/completions, completions; prefill on %s, max batch %d, "
            "prefix cache %s)\n", prog, port, on.c_str(), max_batch, prefix_cache ? "on" : "off");
    if (!s.listen("127.0.0.1", port)) throw std::runtime_error("can't listen on port " + std::to_string(port));
    return 0;
}

static int usage() {
    fprintf(stderr,
            "usage: %1$s MODEL                                print the model spec\n"
            "       %1$s generate MODEL PROMPT [--max-tokens N] [--temp T] [--top-p P] [PREFILL]\n"
            "       %1$s chat MODEL [--max-tokens N] [--temp T] [--top-p P] [PREFILL]   multi-turn chat in the terminal\n"
            "       %1$s bench MODEL [-p 512] [-g 1024] [-b 1] [-n 5] [--stream P] [PREFILL]   mlx_lm.benchmark's\n"
            "         fields; --stream P: a P-token prefill runs alongside throughout, its own tok/s reported\n"
            "       %1$s serve MODEL [--port 8080] [--max-batch 32] [PREFILL]   OpenAI-style /v1/chat/completions and\n"
            "         /v1/completions on 127.0.0.1\n"
            "       %1$s bench-ops MODEL                      GPU time per op and per decode step, next to the roofline\n"
            "MODEL: repo id under $HF_HOME, or a local dir\n"
            "PREFILL: [--prefill gpu|ane] [--ane-weights fp16|int8|lut6|lut4]: dense blocks of prefill on the Neural\n"
            "         Engine (Core ML models built by py/mllm/ane.py for S = 512)\n"
            "         [--prefix-cache on|off]: reuse prompt prefixes across requests (default on; off for bench, whose\n"
            "         trials repeat the same prompts)\n"
            "ENGINE (generate, bench, serve): [--arena-gib G] (default 1, serve 2; 12 at 2 MiB units, the 27B)\n"
            "         [--max-ctx N] (default 16384; 32768 at 2 MiB units)\n", prog);
    return 2;
}

int main(int argc, char** argv) {
    if (const char* sl = strrchr(argv[0], '/')) prog = sl + 1;
    else prog = argv[0];
    if (argc < 2) return usage();
    const std::string cmd = argv[1];
    // flags after the positional arguments: name -> value
    std::map<std::string, std::string> flags;
    const int npos = cmd == "generate" ? 4 : cmd == "bench" || cmd == "bench-ops" || cmd == "serve" || cmd == "chat" ? 3 : 2;
    if (argc < npos || (argc - npos) % 2) return usage();
    for (int i = npos; i < argc; i += 2) flags[argv[i]] = argv[i + 1];
    auto flag = [&](const char* name, const char* def) {
        auto it = flags.find(name);
        std::string v = it == flags.end() ? def : it->second;
        if (it != flags.end()) flags.erase(it);
        return v;
    };
    auto prefix_cache = [&](const char* def) {
        const std::string v = flag("--prefix-cache", def);
        if (v != "on" && v != "off") throw std::invalid_argument("--prefix-cache: on or off, not " + v);
        return v == "on";
    };
    try {
        // defaults by unit size: 2 MiB units (27B) get target-27b.md's budget, a 12 GiB arena and max_ctx 32k; smaller
        // models keep 0 (the command's default arena) and 16k
        const bool big = npos >= 3 && cmd != "bench-ops" && mllm::Memory::unit_of(mllm::spec::load(argv[2])) >= (2u << 20);
        arena_gib = std::stod(flag("--arena-gib", big ? "12" : "0"));
        max_ctx = std::stoi(flag("--max-ctx", big ? "32768" : "16384"));
        if (arena_gib < 0 || max_ctx < 1) return usage();
        if (cmd == "generate") {
            mllm::GenParams p{std::stoi(flag("--max-tokens", "256")), std::stof(flag("--temp", "0")),
                              std::stof(flag("--top-p", "0"))};
            const std::string on = flag("--prefill", "gpu"), aw = flag("--ane-weights", "fp16");
            const bool pc = prefix_cache("on");
            if (!flags.empty()) return usage();
            generate(argv[2], argv[3], p, on, aw, pc);
        } else if (cmd == "chat") {
            const bool default_max = !flags.count("--max-tokens");
            mllm::GenParams p{std::stoi(flag("--max-tokens", "1")), std::stof(flag("--temp", "0")),
                              std::stof(flag("--top-p", "0"))};
            const std::string on = flag("--prefill", "gpu"), aw = flag("--ane-weights", "fp16");
            const bool pc = prefix_cache("on");
            if (!flags.empty()) return usage();
            chat(argv[2], p, default_max, on, aw, pc);
        } else if (cmd == "bench") {
            const int P = std::stoi(flag("-p", "512")), G = std::stoi(flag("-g", "1024")), B = std::stoi(flag("-b", "1"));
            const int n = std::stoi(flag("-n", "5"));
            const std::string on = flag("--prefill", "gpu"), aw = flag("--ane-weights", "fp16");
            const int stream = std::stoi(flag("--stream", "0"));
            const bool pc = prefix_cache("off");
            if (!flags.empty() || P < 2 || G < 1 || B < 1 || n < 1 || stream < 0 || stream == 1) return usage();
            bench(argv[2], P, G, B, n, on, aw, stream, pc);
        } else if (cmd == "serve") {
            const int port = std::stoi(flag("--port", "8080")), max_batch = std::stoi(flag("--max-batch", "32"));
            const std::string on = flag("--prefill", "gpu"), aw = flag("--ane-weights", "fp16");
            const bool pc = prefix_cache("on");
            if (!flags.empty() || max_batch < 1) return usage();
            return serve(argv[2], port, on, aw, max_batch, pc);
        } else if (cmd == "bench-ops") {
            if (!flags.empty()) return usage();
            mllm::GpuOpsTest::bench_ops(argv[2]);
        } else {
            if (argc != 2) return usage();
            mllm::ModelSpec s = mllm::spec::load(argv[1]);
            printf("%s\n%d layers (%zu full, %zu linear), hidden %d, vocab %d\n", s.path.c_str(), s.n_layers(),
                   s.full_layers().size(), s.linear_layers().size(), s.hidden, s.vocab);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "%s: %s\n", prog, e.what());
        return 1;
    }
}
