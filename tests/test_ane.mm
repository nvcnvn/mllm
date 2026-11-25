// AneOps (ane.h): dense blocks on the ANE, the GPU feeding them its weights. Chunks of 1024 (512 blocks for shorter
// ones). Needs the 0.8B's blocks built for 512 and 1024 (the test prints the command if not).
// a. a size that isn't built names the build command; every block loads.
// b. blocks 0 (linear), 3 (full) and 4 (linear, o from a full layer) on golden prefill0 inputs (32 tokens, padded to
//    512): input surfaces zero past n; every output within cos 0.999 of the golden GPU tensors.
// c. PrefillLane, every chunk on the ANE, vs on the GPU: worst cos over every full layer's K / V and linear layer's
//    conv / recurrent state, for the golden chat prompts and repeated-prompt lengths around the chunk sizes. Floors:
//    the measured values less ~1e-3 (fp16 blocks). Two chunks in flight (each with its own weight surfaces)
//    bit-exact with one at a time, also for two requests at once.
// d. Chunks below gpu_below tokens (tails, prefix-cache splits) run on the GPU path.
// e. Engine with ANE prefill: greedy 64 equals golden up to each prompt's first exact tie; decode alongside 2-chunk
//    prefills; a random stress run; cancel / abort with two chunks in flight; a long prompt's greedy vs GPU prefill.
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unistd.h>

#include "ane.h"
#include "engine.h"
#include "golden.h"

static const std::map<std::string, id<MTLBuffer> mllm::Workspace::*> FIELDS = {
    {"q", &mllm::Workspace::query},       {"k", &mllm::Workspace::key},     {"v", &mllm::Workspace::value},
    {"gate", &mllm::Workspace::query_gate}, {"qkv", &mllm::Workspace::query_key_value}, {"z", &mllm::Workspace::output_gate},
    {"b", &mllm::Workspace::update_rate_input},       {"a", &mllm::Workspace::decay_input},     {"x", &mllm::Workspace::residual}};

static float bf16(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float f;
    memcpy(&f, &w, 4);
    return f;
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    const std::string d = golden::dir(s);
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 512 << 20);
    mllm::GpuOps ops(mem, w);

    // a.
    try {
        mllm::AneOps none(ops, mem, 2048);  // 512 and 2048: 2048 isn't built
        assert(!"loaded a missing build");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("build('" + s.path + "', [512, 2048])") != std::string::npos);
    }
    try {
        mllm::AneOps none(ops, mem, 1000);
        assert(!"accepted a chunk that isn't a multiple of 32");
    } catch (const std::invalid_argument&) {
    }
    auto t0 = std::chrono::steady_clock::now();
    const int S = 1024, max_ctx = 3072;
    mllm::AneOps ane(ops, mem, S);
    assert(ane.chunk_sizes == std::vector<int>({512, 1024}) && (int)ane.weight_inputs.size() == s.n_layers());
    printf("a. loaded %zu blocks in %.1f s, %zu surfaces per set\n", ane.blocks.size(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ane.surfaces[0].size());
    mllm::Workspace ws = ops.workspace(S, 1, S);
    mllm::AneOps ane_l(ops, mem, S);  // the lane's (b uses `ane` on this thread)
    mllm::PrefillLane gpu_lane(ops, mem, S, max_ctx), ane_lane(ops, mem, S, max_ctx, &ane_l);
    mem.freeze();
    id<MTLCommandQueue> q = [dev newCommandQueue];

    // b.
    {
        const int n = 32;  // golden prefill0: 32 tokens at pos 0
        ws.request_count = 1, ws.tokens_per_request = n;
        auto put = [&](id<MTLBuffer> b, const std::string& name) {  // a golden bf16 tensor -> workspace rows
            const mllm::Tensor& t = tr.at(name);
            memcpy(b.contents, tr.data(t), t.nbytes);
        };
        const mllm::Tensor& tok = tr.at("prefill0/tokens");
        memcpy(ws.token_ids.contents, tr.data(tok), tok.nbytes);
        for (int i : {0, 3, 4}) {
            const mllm::AneOps::Block& b = ane.block(i, n);
            assert(b.chunk_size == 512);
            char L[8];
            snprintf(L, sizeof L, "L%02d", i - 1);
            if (i > 0) {
                put(ws.residual, "prefill0/" + std::string(L) + "/dense/x");
                put(ws.attention_result, "prefill0/" + std::string(L) + "/attn/o");
            }
            mllm::encode_and_wait(q, ws, [&] { ane.prepare_inputs(i, ws, 0); });
            for (const mllm::AneOps::Activation& io : b.inputs_from_workspace) {  // padding: zero past n
                const uint16_t* p = (const uint16_t*)io.surface[0]->backing.buffer.contents;
                for (size_t t = n; t < (size_t)b.chunk_size; t++)
                    for (int c = 0; c < io.surface[0]->cols; c++) assert(p[t * io.surface[0]->backing.row + c] == 0);
            }
            mllm::encode_and_wait(q, ws, [&] { ane.apply_weights(i, ws, 0); });
            snprintf(L, sizeof L, "L%02d", i);
            for (const mllm::AneOps::Activation& io : b.outputs_to_workspace) {
                std::string name;
                for (const auto& [f, field] : FIELDS)
                    if (ws.*io.field == ws.*field) name = f;
                const int C = io.surface[0]->cols;
                const uint16_t* got = (const uint16_t*)(ws.*io.field).contents;
                // golden, gpu.py layout: q / k / v are [1, H, n, D], the rest [1, n, C]
                const mllm::Tensor& g = tr.at("prefill0/" + std::string(L) + "/dense/" + name);
                const bool heads = g.shape.size() == 4 && g.shape[2] == n;
                const int D = heads ? (int)g.shape[3] : C;
                double dot = 0, gg = 0, ww = 0;
                for (int t = 0; t < n; t++)
                    for (int c = 0; c < C; c++) {
                        size_t gi = heads ? ((size_t)(c / D) * n + t) * D + c % D : (size_t)t * C + c;
                        double x = bf16(got[(size_t)t * C + c]), z = bf16(golden::load<uint16_t>(tr.data(g), gi));
                        dot += x * z, gg += x * x, ww += z * z;
                    }
                const double cos = dot / std::sqrt(gg * ww);
                printf("b. block %d %-4s: cos vs golden %.6f\n", i, name.c_str(), cos);
                assert(cos >= 0.999);
            }
        }
    }

    // c.
    mllm::Safetensors tk(d + "/tokens.safetensors");
    auto ids_of = [&](const std::string& name) {
        const mllm::Tensor& t = tk.at(name);
        std::vector<int> v(t.nbytes / 4);
        for (size_t j = 0; j < v.size(); j++) v[j] = golden::load<int32_t>(tk.data(t), j);
        return v;
    };
    std::vector<std::vector<int>> prompts = {ids_of("chat/0"), ids_of("chat/1"), ids_of("chat/2")};
    std::vector<int> longp;
    while (longp.size() < 2600) longp.insert(longp.end(), prompts[0].begin(), prompts[0].end());
    mllm::Arena& ar = mem.arena;
    uint8_t* arena = (uint8_t*)mem.arena_buffer.contents;
    struct Units {
        std::vector<uint32_t> pages, state;
    };
    auto units = [&](int n) {  // fresh units for n tokens, state zeroed
        Units u;
        assert(ar.admit(mem.units_for(n) + mem.state_units));
        for (uint32_t j = 0; j < mem.units_for(n); j++) u.pages.push_back(ar.alloc());
        for (uint32_t j = 0; j < mem.state_units; j++) u.state.push_back(ar.alloc());
        for (uint32_t x : u.state) memset(arena + (size_t)x * ar.unit_bytes, 0, ar.unit_bytes);
        return u;
    };
    // every prompt's ids[:-1] into fresh units in chunks of S, chunk j of each prompt before chunk j + 1 of any; two:
    // up to 2 chunks in flight, else one at a time
    auto prefill_all = [&](mllm::PrefillLane& lane, const std::vector<std::vector<int>>& ps, bool two) {
        std::vector<Units> us;
        std::vector<mllm::PrefillWork> works;
        for (const std::vector<int>& ids : ps) us.push_back(units((int)ids.size() - 1));
        for (int pos = 0;; pos += S) {
            const size_t before = works.size();
            for (size_t p = 0; p < ps.size(); p++)
                if (const int n = (int)ps[p].size() - 1; pos < n)
                    works.push_back({(int)p, std::vector<int>(ps[p].begin() + pos, ps[p].begin() + std::min(pos + S, n)),
                                     pos, us[p].pages, us[p].state});
            if (works.size() == before) break;
        }
        size_t next = 0, finished = 0;
        while (finished < works.size()) {
            while (next < works.size() && (two ? lane.can_submit() : !lane.busy())) lane.submit(works[next++]);
            std::vector<mllm::PrefillWork> done = lane.poll();
            for (const mllm::PrefillWork& w : done) assert(w.start_pos == works[finished++].start_pos);
            if (done.empty()) lane.wait_oldest();
        }
        return us;
    };
    auto prefill = [&](mllm::PrefillLane& lane, const std::vector<int>& ids) { return prefill_all(lane, {ids}, false)[0]; };
    auto reset = [&] { ar = mllm::Arena(ar.unit_bytes, ar.n_units); };
    auto cos = [](const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0, aa = 0, bb = 0;
        for (size_t j = 0; j < a.size(); j++) dot += a[j] * b[j], aa += a[j] * a[j], bb += b[j] * b[j];
        return dot / std::sqrt(aa * bb);
    };
    // full layer g's K (0) or V (1), all heads, tokens from .. n
    auto kv = [&](const Units& u, int g, int which, int n, int from = 0) {
        std::vector<double> v;
        for (int h = 0; h < s.key_value_heads; h++)
            for (int t = from; t < n; t++) {
                const uint16_t* p = (const uint16_t*)(arena + (size_t)u.pages[t / mem.page_tokens] * ar.unit_bytes +
                                                      mem.kv_offset(g, which, h, t % mem.page_tokens));
                for (int j = 0; j < s.head_dim; j++) v.push_back(bf16(p[j]));
            }
        return v;
    };
    auto state = [&](const Units& u, int l, bool delta) {  // linear layer l's conv (bf16) or delta (fp32) state
        std::vector<double> v;
        const size_t off = mem.state_offset(l) + (delta ? mem.conv_bytes : 0), len = delta ? mem.delta_bytes : mem.conv_bytes;
        for (size_t b = off; b < off + len; b += delta ? 4 : 2) {
            const uint8_t* p = arena + (size_t)u.state[b / ar.unit_bytes] * ar.unit_bytes + b % ar.unit_bytes;
            float f;
            if (delta) memcpy(&f, p, 4);
            else f = bf16(*(const uint16_t*)p);
            v.push_back(f);
        }
        return v;
    };
    // {worst K/V cos over tokens from .. n, worst state cos}
    auto worst_cos = [&](const Units& a, const Units& b, int n, int from = 0) {
        double kvc = 1, st = 1;
        for (int g = 0, l = 0; l < s.n_layers(); l++)
            if (s.kinds[l] == "full") {
                for (int which : {0, 1}) kvc = std::fmin(kvc, cos(kv(a, g, which, n, from), kv(b, g, which, n, from)));
                g++;
            } else {
                for (bool delta : {false, true}) st = std::fmin(st, cos(state(a, l, delta), state(b, l, delta)));
            }
        return std::pair{kvc, st};
    };
    auto same = [&](const Units& a, const Units& b, int n) {  // bit-exact K/V and states
        for (int g = 0, l = 0; l < s.n_layers(); l++)
            if (s.kinds[l] == "full") {
                if (kv(a, g, 0, n) != kv(b, g, 0, n) || kv(a, g, 1, n) != kv(b, g, 1, n)) return false;
                g++;
            } else if (state(a, l, false) != state(b, l, false) || state(a, l, true) != state(b, l, true)) {
                return false;
            }
        return true;
    };
    const int gpu_below = ane_lane.gpu_below;
    ane_lane.gpu_below = 0;  // every chunk on the ANE, padded
    prefill(ane_lane, prompts[0]);  // warm up (first predicts are slow)
    reset();
    {
        struct Case {
            std::string name;
            std::vector<int> ids;
            double floor;
        };
        std::vector<Case> cases = {{"chat/0", prompts[0], 0.998}, {"chat/1", prompts[1], 0.998}, {"chat/2", prompts[2], 0.997}};
        for (auto [n, f] : {std::pair{500, 0.996}, {1024, 0.995}, {1500, 0.995}, {2560, 0.994}})
            cases.push_back({"repeated " + std::to_string(n), std::vector<int>(longp.begin(), longp.begin() + n + 1), f});
        for (const Case& c : cases) {
            const int n = (int)c.ids.size() - 1;
            auto [k, st] = worst_cos(prefill(gpu_lane, c.ids), prefill(ane_lane, c.ids), n);
            const double worst = std::fmin(k, st);
            printf("c. %-13s %4d tokens: worst K/V/state cos vs GPU lane %.6f (floor %.3f)\n", c.name.c_str(), n, worst, c.floor);
            assert(worst >= c.floor);
            reset();
        }
    }
    const std::vector<int> three(longp.begin(), longp.begin() + 2561), two(longp.begin(), longp.begin() + 1501);
    for (const auto* ids : {&two, &three}) {
        Units a = prefill_all(ane_lane, {*ids}, false)[0], b = prefill_all(ane_lane, {*ids}, true)[0];
        const bool eq = same(a, b, (int)ids->size() - 1);
        printf("c. %zu tokens, 2 chunks in flight: K/V and state %s one at a time\n", ids->size() - 1,
               eq ? "bit-exact with" : "DIFFER from");
        assert(eq);
        reset();
    }
    {
        Units x = prefill(ane_lane, three), y = prefill(ane_lane, two);
        std::vector<Units> xy = prefill_all(ane_lane, {three, two}, true);
        const bool eq = same(x, xy[0], (int)three.size() - 1) && same(y, xy[1], (int)two.size() - 1);
        printf("c. 2560 + 1500 tokens prefilling at once: each %s its solo run\n", eq ? "equals" : "DIFFERS from");
        assert(eq);
        reset();
    }
    ane_lane.gpu_below = gpu_below;

    // d. tails on the GPU: an ANE chunk of fewer than gpu_below tokens runs on the GPU path (ws.tokens_per_request = n, no padding).
    // Bit-exact with the ANE lane's full chunks followed by the GPU lane's tail. Against GPU prefill, on the tail tokens'
    // K/V and the final state: >= 0.999 on random prompts (the repetitive one's ANE chunks drift more).
    {
        std::vector<int> randp;
        std::mt19937 rng(3);
        for (int j = 0; j < 2101; j++) randp.push_back((int)(rng() % s.vocab_size));
        for (const auto* src : {&longp, &randp})
            for (int n : {1100, 2100}) {
                const std::vector<int> ids(src->begin(), src->begin() + n + 1);
                const int full = n / S * S;
                Units g = prefill(gpu_lane, ids);
                const int before = ane_lane.gpu_chunks;
                Units m = prefill(ane_lane, ids);
                assert(ane_lane.gpu_chunks == before + 1);
                Units c = units(n);
                for (int pos = 0; pos < n; pos += S) {
                    mllm::PrefillLane& l = pos < full ? ane_lane : gpu_lane;
                    l.submit({0, std::vector<int>(ids.begin() + pos, ids.begin() + std::min(pos + S, n)), pos, c.pages, c.state});
                    while (l.poll().empty()) l.wait_oldest();
                }
                const auto [mk, ms] = worst_cos(g, m, n, full);
                const bool eq = same(m, c, n);
                printf("d. %s %4d tokens, %2d-token tail on the GPU: cos vs GPU prefill, tail K/V %.6f, state %.6f; %s "
                       "ANE chunks + GPU lane tail\n",
                       src == &longp ? "repeated" : "random  ", n, n - full, mk, ms, eq ? "bit-exact with" : "DIFFERS from");
                assert(eq && (src == &longp || (mk >= 0.999 && ms >= 0.999)));
                reset();
            }
    }

    // e. engine
    auto greedy = [](mllm::Engine& e, const std::vector<int>& ids, int N) {
        const int id = e.submit(ids, {.max_tokens = N});
        std::vector<int> out;
        while (e.requests.count(id))
            for (const mllm::Event& ev : e.step()) out.push_back(ev.token);
        return out;
    };
    mllm::Engine ea(golden::MODEL, 512 << 20, S, 32, 16, 4096, "ane");
    // up to the first exact tie (top-2 logits equal in bf16 under mlx-lm, teacher-forced: chat/1 at token 38, chat/2 at
    // 28; chat/0 has none): the fp16 blocks' drift flips those
    {
        ea.prefill.gpu_below = 0;  // the chat prompts are short: on the ANE anyway
        const size_t tie[3] = {64, 38, 28};
        for (int i = 0; i < 3; i++) {
            const std::vector<int> got = greedy(ea, prompts[i], 64), want = ids_of("greedy/" + std::to_string(i));
            size_t first = 0;
            while (first < got.size() && first < want.size() && got[first] == want[first]) first++;
            printf("e. engine, ANE prefill: prompt %d greedy: first %zu of %zu tokens equal golden (first tie at %zu)\n",
                   i, first, got.size(), tie[i]);
            assert(got.size() == want.size() && first >= tie[i]);
        }
        ea.prefill.gpu_below = gpu_below;
    }
    // d. prefix-cache splits: a fresh 600-token prompt stops at its last full page (576, a snapshot point): 576 tokens
    // on the ANE, then the 23-token tail on the GPU
    {
        std::mt19937 rng(4);
        std::vector<int> ids(600);
        for (int& t : ids) t = (int)(rng() % s.vocab_size);
        const int before = ea.prefill.gpu_chunks;
        greedy(ea, ids, 1);
        assert(ea.prefill.gpu_chunks == before + 1);
        printf("d. engine: a 600-token prompt's tail after its prefix-cache split (23 tokens) ran on the GPU\n");
    }
    // d. a chunk ending at a snapshot point holds back its request's next chunk until poll has copied the state. Polled
    // late (40 ms between steps), both chunks would be done by then and the copy would hold the next chunk's state too.
    // The cached snapshot at 1088 (the last full page: chunks 1024, 64, then 11) equals a lane prefill of those tokens.
    {
        std::mt19937 rng(5);
        std::vector<int> ids(1100);
        for (int& t : ids) t = (int)(rng() % s.vocab_size);
        const int id = ea.submit(ids, {.max_tokens = 1});
        while (ea.requests.count(id)) ea.step(), usleep(40000);
        const std::vector<int> head(ids.begin(), ids.begin() + 1089);
        mllm::PrefixCache::Match m = ea.cache->match(head);
        assert(m.resume_pages * mem.page_tokens == 1088 && m.snapshot.size() == mem.state_units);
        Units u = prefill(ane_lane, head);
        const uint8_t* ea_arena = (const uint8_t*)ea.mem.arena_buffer.contents;
        size_t diff = 0;
        for (size_t j = 0; j < m.snapshot.size(); j++)
            diff += memcmp(ea_arena + (size_t)m.snapshot[j] * ar.unit_bytes, arena + (size_t)u.state[j] * ar.unit_bytes,
                           ar.unit_bytes) != 0;
        printf("d. engine, late polls: the snapshot at 1088 %s a lane prefill of 1088 tokens\n", diff ? "DIFFERS from" : "equals");
        assert(!diff);
        reset();
    }
    // decode on the engine thread while the ANE lane's thread prefills (both encode with GpuOps: the lane has its own
    // copy; sharing one crashed `mllm bench -b 8 --stream 512 --prefill ane`): prompt 0 still decodes golden's tokens
    // while 2-chunk prompts prefill alongside, one after another (chat/0 has no tie)
    const std::vector<int> two_chunks(longp.begin(), longp.begin() + 1301);
    {
        const int id = ea.submit(prompts[0], {.max_tokens = 64});
        std::vector<int> out;
        int other = -1;
        while (ea.requests.count(id)) {
            if (!ea.requests.count(other)) other = ea.submit(two_chunks, {.max_tokens = 1});
            for (const mllm::Event& ev : ea.step())
                if (ev.req_id == id) out.push_back(ev.token);
        }
        ea.cancel(other);
        while (!ea.requests.empty()) ea.step();
        assert(out == ids_of("greedy/0"));
        printf("e. engine: greedy 64 still golden with 2-chunk ANE prefills running alongside\n");
    }
    // random stress: 10 requests of 2..2100 random tokens (up to 3 chunks), greedy and sampled, a few cancels; every
    // step returns, events keep coming (watchdog 30 s)
    {
        std::mt19937 rng(1);
        int submitted = 0;
        auto last = std::chrono::steady_clock::now();
        while (submitted < 10 || !ea.requests.empty()) {
            if (submitted < 10 && ea.requests.size() < 16 && rng() % 3 == 0) {
                std::vector<int> ids(2 + rng() % 2099);
                for (int& t : ids) t = (int)(rng() % s.vocab_size);
                ea.submit(ids, {.max_tokens = 1 + (int)(rng() % 64), .temperature = rng() % 2 ? 0.7f : 0.0f});
                submitted++;
            }
            if (!ea.requests.empty() && rng() % 300 == 0) ea.cancel(std::next(ea.requests.begin(), rng() % ea.requests.size())->first);
            std::vector<mllm::Event> ev = ea.step();
            for (const mllm::Event& e : ev) assert(e.token >= 0);
            if (!ev.empty()) last = std::chrono::steady_clock::now();
            assert(std::chrono::steady_clock::now() - last < std::chrono::seconds(30));
        }
        printf("e. engine: random stress with ANE prefill, %d requests, no failure\n", submitted);
    }
    // cancel and abort with two chunks in flight (one request's, or two requests'): every unit comes back, no hang,
    // and the lane works afterwards. Random prompts: no prefix-cache hits shorten them.
    {
        mllm::Arena& a = ea.mem.arena;
        auto all_free = [&] { return a.free_units + a.cache_only_units == a.n_units && a.reserved == 0 && !ea.prefill.busy(); };
        auto in_flight = [&] {
            int k = 0;
            for (auto& [_, r] : ea.requests) k += r.state == mllm::State::PREFILL ? r.in_flight : 0;
            return k;
        };
        std::mt19937 rng(2);
        auto rand_ids = [&](int n) {
            std::vector<int> v(n);
            for (int& t : v) t = (int)(rng() % s.vocab_size);
            return v;
        };
        auto drain = [&](std::vector<mllm::Event>* evs) {
            auto t0 = std::chrono::steady_clock::now();
            while (!ea.requests.empty()) {
                for (const mllm::Event& e : ea.step()) evs->push_back(e);
                assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30));
            }
        };
        assert(all_free());
        std::vector<mllm::Event> evs;
        for (int reqs : {1, 2}) {
            // 2 requests: x's one chunk and y's first (FIFO takes a request's second chunk before the next request)
            const int x = ea.submit(rand_ids(reqs == 2 ? 300 : 1300), {.max_tokens = 4});
            const int y = reqs == 2 ? ea.submit(rand_ids(1300), {.max_tokens = 4}) : -1;
            while (in_flight() < 2) ea.step();
            if (reqs == 2) assert(ea.requests.at(x).in_flight == 1 && ea.requests.at(y).in_flight == 1);
            ea.cancel(x);  // y (if any) runs on to its tokens
            evs.clear();
            drain(&evs);
            assert(all_free() && std::count_if(evs.begin(), evs.end(), [&](auto& e) { return e.req_id == y; }) == (reqs == 2 ? 4 : 0));
            printf("e. engine: cancel with 2 chunks in flight (%d request%s): all units back\n", reqs, reqs == 2 ? "s" : "");
        }
        for (int reqs : {1, 2}) {
            const int x = ea.submit(rand_ids(reqs == 2 ? 300 : 1300), {.max_tokens = 4});
            const int y = reqs == 2 ? ea.submit(rand_ids(1300), {.max_tokens = 4}) : -1;
            while (in_flight() < 2) ea.step();
            ea.prefill.fail_next = true;
            bool threw = false;
            evs.clear();
            auto t0 = std::chrono::steady_clock::now();
            while (!threw) {
                try {
                    for (const mllm::Event& e : ea.step()) evs.push_back(e);
                } catch (const std::runtime_error& e) {
                    threw = std::string(e.what()).find("injected") != std::string::npos;
                    assert(threw);
                }
                assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30));
            }
            drain(&evs);
            for (int id : {x, y})
                if (id >= 0) assert(std::count_if(evs.begin(), evs.end(), [&](auto& e) { return e.req_id == id && e.token < 0; }) == 1);
            assert(all_free());
            printf("e. engine: abort with 2 chunks in flight (%d request%s): aborted, all units back\n", reqs, reqs == 2 ? "s" : "");
        }
        // the submit of a request's second chunk throws while its first is in the lane: aborted (token -1), the first
        // chunk finishes in the lane and poll ends the request
        {
            const int x = ea.submit(rand_ids(1300), {.max_tokens = 4});
            ea.prefill.fail_submit = 2;
            evs.clear();
            try {
                ea.step();
                assert(!"the second submit didn't throw");
            } catch (const std::runtime_error& e) {
                assert(std::string(e.what()).find("injected submit") != std::string::npos);
            }
            drain(&evs);
            assert(std::count_if(evs.begin(), evs.end(), [&](auto& e) { return e.req_id == x && e.token < 0; }) == 1);
            assert(all_free());
            printf("e. engine: a failed submit of a second chunk: aborted, all units back\n");
        }
        assert(greedy(ea, prompts[0], 64) == ids_of("greedy/0"));
        printf("e. engine: greedy 64 golden again after the aborts\n");
    }
    // a 1300-token prompt (two ANE chunks): greedy 30 after ANE prefill vs after GPU prefill
    {
        const std::vector<int> long_ane = greedy(ea, two_chunks, 30);
        mllm::Engine eg(golden::MODEL, 512 << 20, 512, 32, 16, 4096);
        const std::vector<int> long_gpu = greedy(eg, two_chunks, 30);
        size_t same = 0;
        while (same < long_gpu.size() && long_ane[same] == long_gpu[same]) same++;
        printf("e. engine, %zu-token prompt: ANE prefill's greedy 30 equal GPU prefill's for %zu tokens\n",
               two_chunks.size(), same);
        assert(long_ane == long_gpu);
    }
    printf("PASS\n");
}
