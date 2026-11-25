// AneOps "shared" (ane.h): blocks whose weights the GPU feeds in, chunks of 1024 (and 512). Needs the 0.8B's shared
// blocks built for 512 and 1024 (the test prints the command if not).
// a. a size that isn't built names the build_shared command.
// b. PrefillLane, shared ANE (every chunk on the ANE) vs GPU: worst cos over every full layer's K / V and linear
//    layer's conv / recurrent state, for the golden chat prompts and repeated-prompt lengths around the chunk sizes
//    (one 512 chunk, one 1024, 1024 + 512, 1024 + 1024 + 512). Floors: the measured values less 2e-4 (fp16 blocks
//    drift like the baked fp16 ones, test_ane.mm).
// c. Two chunks in flight (each with its own weight surfaces) bit-exact with one at a time, also for two requests.
// d. Engine with shared ANE prefill: greedy 64 equals golden for the 3 chat prompts up to their first exact tie (top-2
//    logits equal in bf16 under mlx-lm, teacher-forced: chat/1 at token 38, chat/2 at 28; prompt 0 has none). The
//    fp16 blocks' drift flips those; the baked fp16 blocks happen to land on golden's side.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "ane.h"
#include "engine.h"
#include "golden.h"

static float bf16(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float f;
    memcpy(&f, &w, 4);
    return f;
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    const std::string d = golden::dir(s);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 512 << 20);
    mllm::GpuOps ops(mem, w);

    // a.
    try {
        mllm::AneOps none(ops, mem, 2048, "shared");  // 512 and 2048: 2048 isn't built
        assert(!"loaded a missing build");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("build_shared('" + s.path + "', [512, 2048])") != std::string::npos);
    }
    const int S = 1024, max_ctx = 3072;
    mllm::AneOps ane(ops, mem, S, "shared");
    assert(ane.shared && ane.sizes == std::vector<int>({512, 1024}) && ane.blocks.empty());
    printf("a. loaded %zu shared blocks, %zu surfaces per set\n", ane.kinds.size(), ane.surfaces[0].size());
    mllm::PrefillLane gpu_lane(ops, mem, S, max_ctx), ane_lane(ops, mem, S, max_ctx, &ane);
    ane_lane.gpu_below = 0;  // every chunk on the ANE
    mem.freeze();

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
    auto cos = [](const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0, aa = 0, bb = 0;
        for (size_t j = 0; j < a.size(); j++) dot += a[j] * b[j], aa += a[j] * a[j], bb += b[j] * b[j];
        return dot / std::sqrt(aa * bb);
    };
    auto kv = [&](const Units& u, int g, int which, int n) {  // full layer g's K (0) or V (1), all heads, n tokens
        std::vector<double> v;
        for (int h = 0; h < s.n_kv_heads; h++)
            for (int t = 0; t < n; t++) {
                const uint16_t* p = (const uint16_t*)(arena + (size_t)u.pages[t / mem.T] * ar.unit_bytes +
                                                      mem.kv_offset(g, which, h, t % mem.T));
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
    auto each = [&](const Units& a, const Units& b, int n, auto f) {  // f(vector a, vector b) over every K/V and state
        for (int g = 0, l = 0; l < s.n_layers(); l++)
            if (s.kinds[l] == "full") {
                for (int which = 0; which < 2; which++)
                    if (!f(kv(a, g, which, n), kv(b, g, which, n))) return false;
                g++;
            } else {
                for (bool delta : {false, true})
                    if (!f(state(a, l, delta), state(b, l, delta))) return false;
            }
        return true;
    };

    // b.
    prefill(ane_lane, prompts[0]);  // warm up (first predicts are slow)
    ar = mllm::Arena(ar.unit_bytes, ar.n_units);
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
        Units g = prefill(gpu_lane, c.ids), a = prefill(ane_lane, c.ids);
        double worst = 1;
        each(g, a, n, [&](const std::vector<double>& x, const std::vector<double>& y) {
            worst = std::fmin(worst, cos(x, y));
            return true;
        });
        printf("b. %-13s %4d tokens: worst K/V/state cos vs GPU lane %.6f (floor %.3f)\n", c.name.c_str(), n, worst, c.floor);
        assert(worst >= c.floor);
        ar = mllm::Arena(ar.unit_bytes, ar.n_units);
    }

    // c.
    auto same = [&](const Units& a, const Units& b, int n) {
        return each(a, b, n, [](const std::vector<double>& x, const std::vector<double>& y) { return x == y; });
    };
    const std::vector<int> three(longp.begin(), longp.begin() + 2561), two(longp.begin(), longp.begin() + 1501);
    for (const auto* ids : {&two, &three}) {
        Units a = prefill_all(ane_lane, {*ids}, false)[0], b = prefill_all(ane_lane, {*ids}, true)[0];
        const bool eq = same(a, b, (int)ids->size() - 1);
        printf("c. %zu tokens, 2 chunks in flight: K/V and state %s one at a time\n", ids->size() - 1,
               eq ? "bit-exact with" : "DIFFER from");
        assert(eq);
        ar = mllm::Arena(ar.unit_bytes, ar.n_units);
    }
    {
        Units x = prefill(ane_lane, three), y = prefill(ane_lane, two);
        std::vector<Units> xy = prefill_all(ane_lane, {three, two}, true);
        const bool eq = same(x, xy[0], (int)three.size() - 1) && same(y, xy[1], (int)two.size() - 1);
        printf("c. 2560 + 1500 tokens prefilling at once: each %s its solo run\n", eq ? "equals" : "DIFFERS from");
        assert(eq);
        ar = mllm::Arena(ar.unit_bytes, ar.n_units);
    }

    // d.
    mllm::Engine e(golden::MODEL, 512 << 20, S, 32, 16, 4096, "ane", "shared");
    e.prefill.gpu_below = 0;  // the chat prompts are short: on the ANE anyway
    const size_t tie[3] = {64, 38, 28};
    for (int i = 0; i < 3; i++) {
        const int id = e.submit(prompts[i], {.max_tokens = 64});
        std::vector<int> got;
        while (e.reqs.count(id))
            for (const mllm::Event& ev : e.step()) got.push_back(ev.token);
        const std::vector<int> want = ids_of("greedy/" + std::to_string(i));
        size_t first = 0;
        while (first < got.size() && first < want.size() && got[first] == want[first]) first++;
        printf("d. engine, shared ANE prefill: prompt %d greedy: first %zu of %zu tokens equal golden (first tie at %zu)\n",
               i, first, got.size(), tie[i]);
        assert(got.size() == want.size() && first >= tie[i]);
    }
    printf("PASS\n");
}
