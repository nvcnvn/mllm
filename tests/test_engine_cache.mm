// Engine + prefix cache (g8): greedy tokens with the cache equal those without, cold and on hits; hits land where
// snapshots were taken (a prompt's last full page, a branch point); after a hit the K/V and linear state match the
// cache-off run (bit-exact when the chunk boundaries do); the same on ANE prefill; a small arena forcing evictions
// under 20 mixed requests ends with every unit free or cache-held.
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <string>

#include "engine.h"
#include "golden.h"

constexpr int N = 64, CHUNK = 512;
// 512 MiB at 0.8B (1365 units): ~16 requests of ~900 tokens (pages + state) cached at once; the same count elsewhere
static const size_t ARENA = std::max<size_t>(512 << 20, mllm::Memory::arena_for(mllm::spec::load(golden::MODEL), 16, 900));
// 64 MiB at 0.8B (170 units: one state slot of 50 and 3840 tokens of pages); the same elsewhere, in its units
static const size_t SMALL = std::max<size_t>(64 << 20, mllm::Memory::arena_for(mllm::spec::load(golden::MODEL), 1, 3840));

struct Run {
    std::vector<int> toks;
    int cached = -1;
    std::vector<float> kv, state;  // after prefill + one decode step (max_tokens 1 runs only)
};

static float bf16(const uint8_t* p) {
    uint32_t u = (uint32_t)(p[0] | p[1] << 8) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// one request alone; with max_tokens 1, also its K/V for positions [0, len) and its state, read from the arena after
// it finished (its units are released but nothing reuses them while the engine is idle)
static Run run(mllm::Engine& e, const std::vector<int>& ids, int max_tokens = N) {
    const int id = e.submit(ids, {.max_tokens = max_tokens});
    Run r;
    std::vector<uint32_t> pages, st;
    while (e.reqs.count(id)) {
        for (const mllm::Event& ev : e.step()) r.toks.push_back(ev.token);
        auto it = e.reqs.find(id);
        if (it != e.reqs.end() && it->second.state != mllm::State::WAITING)
            pages = it->second.pages, st = it->second.state_slots, r.cached = it->second.cached;
    }
    if (max_tokens > 1) return r;
    const mllm::Memory& m = e.mem;
    const uint8_t* base = (const uint8_t*)m.arena_buffer.contents;
    const size_t ub = m.arena.unit_bytes;
    for (int t = 0; t < (int)ids.size(); t++)
        for (int g = 0; g < (int)m.spec.full_layers().size(); g++)
            for (int kv = 0; kv < 2; kv++)
                for (int h = 0; h < m.spec.n_kv_heads; h++)
                    for (int d = 0; d < m.spec.head_dim; d++)
                        r.kv.push_back(bf16(base + pages[t / m.T] * ub + m.kv_offset(g, kv, h, t % m.T) + d * 2));
    auto at = [&](size_t b) { return base + st[b / ub] * ub + b % ub; };  // the state span's byte b
    for (int l : m.spec.linear_layers()) {
        const size_t off = m.state_offset(l);
        for (size_t b = 0; b < m.conv_bytes; b += 2) r.state.push_back(bf16(at(off + b)));
        for (size_t b = 0; b < m.delta_bytes; b += 4) {
            float f;
            memcpy(&f, at(off + m.conv_bytes + b), 4);
            r.state.push_back(f);
        }
    }
    return r;
}

// cos floor when chunk boundaries move. mlx-lm's own long1300, prefilled whole vs split at 1280 (chunks of 512):
// 0.8B K/V cos 0.99998, state 0.99996; 27B (64 layers) K/V 0.99338, state 0.96575. Ours match both to ~1e-3. The
// bit-exact checks (chunk 32, same boundaries) are the strict ones.
static const double COS_MIN = getenv("MLLM_MODEL") ? 0.95 : 0.9999;

static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& b, bool exact) {
    assert(a.size() == b.size() && !a.empty());
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (size_t i = 0; i < a.size(); i++) dot += (double)a[i] * b[i], na += (double)a[i] * a[i], nb += (double)b[i] * b[i];
    for (size_t i = 0; i < a.size(); i++) mx = std::max(mx, (double)std::fabs(a[i] - b[i]));
    const double cos = dot / std::sqrt(na * nb);
    const bool same = !memcmp(a.data(), b.data(), a.size() * 4);
    printf("  %-28s max |diff| %.3g, cos %.8f%s\n", what, mx, cos, same ? " (bit-exact)" : "");
    assert(exact ? same : cos >= COS_MIN);
}

static std::vector<int> head(const std::vector<int>& a, size_t n) {
    assert(a.size() >= n);
    return {a.begin(), a.begin() + n};
}

static std::string repeat(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; i++) out += s;
    return out;
}

static void check_idle(const mllm::Engine& e) {  // every unit free or held by the cache only
    const mllm::Arena& a = e.mem.arena;
    uint32_t cache_only = 0;
    for (uint32_t s = 0; s < a.n_units; s++) assert(!a.req_refs[s]), cache_only += a.cache_refs[s] > 0;
    assert(a.reserved == 0 && a.cache_only_units == cache_only && a.free_units + cache_only == a.n_units);
}

// every refcount from its holders (cache nodes, live requests), the counts from the refcounts, child counts
static void invariants(const mllm::Engine& e) {
    const mllm::Arena& a = e.mem.arena;
    std::vector<uint32_t> cache(a.n_units), req(a.n_units);
    std::map<uint64_t, int> kids;
    for (auto& [h, n] : e.cache->nodes) {
        cache[n.unit]++;
        for (uint32_t u : n.snapshot) cache[u]++;
        if (e.cache->nodes.count(n.parent)) kids[n.parent]++;
    }
    uint32_t reserved = 0, unheld = 0, cache_only = 0;
    for (auto& [_, r] : e.reqs) {
        if (r.state == mllm::State::DONE) continue;
        reserved += r.reserved;
        for (uint32_t u : r.pages) req[u]++;
        for (uint32_t u : r.state_slots) req[u]++;
    }
    for (uint32_t s = 0; s < a.n_units; s++) {
        assert(a.cache_refs[s] == cache[s] && a.req_refs[s] == req[s]);
        unheld += !req[s] && !cache[s], cache_only += !req[s] && cache[s];
    }
    assert(unheld == a.free_units && cache_only == a.cache_only_units && reserved == a.reserved);
    for (auto& [h, n] : e.cache->nodes) assert(n.children == kids[h]);
}

// QA's stress (g8qa_cache): 80 random requests (shared bases, multi-turn extensions, repeats; random cancels in
// prefill and decode) on a small arena. Invariants every step, cached units never change while cached, no stall.
static void stress(int chunk, int seed) {
    mllm::Engine e(golden::MODEL, SMALL, chunk, 32, 16, 16384, "gpu", true);
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> pick(0, 99), tokd(1000, 100000);
    auto rand_ids = [&](int n) {
        std::vector<int> v(n);
        for (int& t : v) t = tokd(g);
        return v;
    };
    std::vector<std::vector<int>> bases, history;
    for (int n : {300, 650, 900}) bases.push_back(rand_ids(n));
    std::map<int, std::pair<int, int>> plans;  // req -> (cancel at prefill pos, cancel at output count), -1 = no
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> sums;  // (node, unit) -> checksum while the node holds it
    auto sum = [&](uint32_t u) {
        const uint64_t* p = (const uint64_t*)((const uint8_t*)e.mem.arena_buffer.contents + (size_t)u * e.mem.arena.unit_bytes);
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < e.mem.arena.unit_bytes / 8; i += 61) h = (h ^ p[i]) * 1099511628211ull;  // sampled
        return h;
    };
    auto last_ev = std::chrono::steady_clock::now();
    int submitted = 0;
    std::set<int> hit;
    while (submitted < 80 || !e.reqs.empty()) {
        if (submitted < 80 && e.reqs.size() < 10 && pick(g) < 30) {
            std::vector<int> p;
            const int k = pick(g);
            if (k < 30 && !history.empty()) {  // multi-turn: an earlier prompt + more
                p = history[pick(g) % history.size()];
                auto more = rand_ids(1 + pick(g));
                p.insert(p.end(), more.begin(), more.end());
            } else if (k < 45 && !history.empty()) {
                p = history[pick(g) % history.size()];  // repeated
            } else {
                p = bases[pick(g) % 3];
                p.resize(std::max<size_t>(2, p.size() - pick(g) * 2));
                auto more = rand_ids(pick(g) * 3);
                p.insert(p.end(), more.begin(), more.end());
            }
            if (p.size() > 2500) p.resize(2500);
            history.push_back(p);
            const int mt = 1 + pick(g) % 40, c = pick(g);
            const int id = e.submit(p, {.max_tokens = mt});
            plans[id] = {c < 15 ? pick(g) * (int)p.size() / 100 : -1, c >= 15 && c < 30 ? pick(g) % mt : -1};
            submitted++;
        }
        if (!e.step().empty()) last_ev = std::chrono::steady_clock::now();
        // no token for this long = stalled; 0.8B prefills 2500 tokens in < 1 s, the 27B on the GPU in ~20-35 s
        assert(std::chrono::steady_clock::now() - last_ev < std::chrono::seconds(getenv("MLLM_MODEL") ? 120 : 10) &&
               "stalled");
        for (auto& [id, r] : e.reqs) {
            if (r.state == mllm::State::DONE || r.cancelled) continue;
            const auto [at_pos, at_out] = plans[id];
            if ((at_pos >= 0 && r.state == mllm::State::PREFILL && r.pos >= at_pos) ||
                (at_out >= 0 && r.state == mllm::State::DECODE && (int)r.out.size() >= at_out))
                e.cancel(id);
            if (r.cached > 0 && r.state != mllm::State::WAITING) hit.insert(id);
        }
        invariants(e);
        std::map<std::pair<uint64_t, uint32_t>, uint64_t> now;
        for (auto& [h, n] : e.cache->nodes) {
            now[{h, n.unit}] = sum(n.unit);
            for (uint32_t u : n.snapshot) now[{h ^ 0x9e3779b97f4a7c15ull, u}] = sum(u);
        }
        for (auto& [k, v] : now) assert((!sums.count(k) || sums[k] == v) && "a cached unit changed");
        sums = std::move(now);
    }
    check_idle(e);
    while (e.cache->evict()) {}
    assert(e.mem.arena.cache_only_units == 0 && e.mem.arena.free_units == e.mem.arena.n_units);
    printf("  stress chunk %d seed %d, %zu MiB: 80 requests, %zu hits; invariants held\n", chunk, seed, SMALL >> 20, hit.size());
    assert(!hit.empty());
}

int main(int argc, char** argv) {
    // At 27B the whole test takes ~28 min (g15 logs: `cache` 652 s, stress 508 + 494 s), too close to a 30-min hold:
    // `cache` runs all but the stress runs, `stress512` / `stress32` one stress run each. No argument: everything.
    const std::string part = argc > 1 ? argv[1] : "";
    if (part == "stress512" || part == "stress32") {
        part == "stress512" ? stress(512, 3) : stress(32, 1);
        printf("PASS\n");
        return 0;
    }
    assert(part.empty() || part == "cache");
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tk(d + "/tokens.safetensors");
    mllm::Tokenizer tok(s.path);
    std::vector<std::vector<int>> prompts, golden_ids;  // the 3 golden chats, then the long prompts
    for (int i = 0; i < 3; i++) {
        std::vector<mllm::Message> chat;
        for (const auto& m : g["chat"][i]) chat.push_back({m["role"], m["content"]});
        prompts.push_back(tok.encode_chat(chat));
        const mllm::Tensor& t = tk.at("greedy/" + std::to_string(i));
        golden_ids.emplace_back(t.nbytes / 4);
        for (size_t j = 0; j < golden_ids.back().size(); j++) golden_ids.back()[j] = golden::load<int32_t>(tk.data(t), j);
    }
    const std::vector<int> text = tok.encode(repeat(
        "The river runs past the mill, and the miller counts the sacks. Each morning the baker buys flour; each "
        "evening the children walk home along the water. ", 60));
    const std::vector<int> long1300 = head(text, 1300), long1281 = head(text, 1281);  // 1281: 1280 = 40 full pages
    // A shared system prompt (~790 tokens), then different questions. Chat, not the repeated text: greedy on that
    // flips at near-ties when only the chunk boundaries move (cache off, chunk 32 vs 512: from token 16 on).
    auto chat = [&](const char* what, const char* q) {
        std::string sys = std::string("You are a helpful assistant for a shipping company. Reference data follows.\n");
        const char* cargo[] = {"salt", "iron", "letters", "grain", "timber"};
        for (int k = 1; k <= 40; k++)
            sys += std::string(what) + " " + std::to_string(k) + ": departs port " + std::to_string(k * 7 % 13) +
                   " at " + std::to_string(k % 12 + 1) + " o'clock carrying " + cargo[k % 5] + ".\n";
        return tok.encode_chat({{"system", sys}, {"user", q}});
    };
    // X's last full page lies past the branch point (a long question), so Y misses; Y snapshots the branch point
    const std::vector<int> X = chat("Ship", "Which ships carry salt, which of them leave before noon, and from which "
                                            "ports do they depart? Answer as a short list with one ship per line."),
                           Y = chat("Ship", "Write a haiku about ship 7."), Z = chat("Ship", "When does ship 12 depart?"),
                           Z2 = chat("Ship", "List the ships that carry grain."),
                           Z3 = chat("Ship", "Does any ship carry timber?");
    size_t c = 0;  // the shared prefix
    while (X[c] == Y[c] && X[c] == Z[c]) c++;
    const int branch = (int)c / 32 * 32;
    // aligned: W's last full page ends at its last token, so V (W's prompt + more) hits W's own snapshot (at 768)
    const std::vector<int> V = chat("Truck", "Which trucks carry iron?"), W = head(V, 769);
    const std::vector<std::vector<int>> all = {prompts[0], prompts[1], prompts[2], long1300, long1281, X, Y, Z, Z2, Z3,
                                               V, W};
    auto first_diff = [](const std::vector<int>& a, const std::vector<int>& b) {
        size_t i = 0;
        while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
        return i;
    };
    {
        // chunk 512, the served configuration
        std::map<std::vector<int>, std::vector<int>> off_toks;  // greedy 64, cache off
        mllm::Engine off(golden::MODEL, ARENA, CHUNK, 32, 16, 16384, "gpu", false);
        mllm::Engine on(golden::MODEL, ARENA, CHUNK, 32, 16, 16384, "gpu", true);
        assert(!off.cache && on.cache);
        for (const auto& p : all) {
            Run o = run(off, p);
            off_toks[p] = o.toks;
            assert(o.cached == 0);
        }
        for (int i = 0; i < 3; i++) assert(off_toks[prompts[i]] == golden_ids[i]);
        printf("off: 3 golden prompts equal golden\n");
        // cold, then the same prompt again: hits at its last full page
        for (const auto& p : {prompts[0], prompts[1], prompts[2], long1300}) {
            Run cold = run(on, p), hit = run(on, p);
            const int P = ((int)p.size() - 1) / 32 * 32;
            printf("  len %4zu: cold cached %d, again cached %d (P %d), tokens %s\n", p.size(), cold.cached, hit.cached,
                   P, cold.toks == off_toks[p] && hit.toks == cold.toks ? "equal" : "DIFFER");
            assert(cold.cached == 0 && hit.cached == P && cold.toks == off_toks[p] && hit.toks == cold.toks);
        }
        // Shared prefix: X caches its last full page (not on Y's path); Y misses and snapshots the branch point, Z
        // hits there. Their chunk boundaries move (X: P splits a chunk; Z: resumes at the branch point), which flips
        // greedy near-ties as much as cache-off chunk 64 vs 512 does (K/V cos 0.99991 there): tokens are only
        // reported here; the chunk-32 block below checks them where the boundaries are the same.
        Run x = run(on, X), y = run(on, Y), z = run(on, Z);
        printf("  shared %zu: X cached %d, Y %d, Z %d; tokens equal cache off up to %zu, %zu, %zu of 64\n", c, x.cached,
               y.cached, z.cached, first_diff(x.toks, off_toks[X]), first_diff(y.toks, off_toks[Y]),
               first_diff(z.toks, off_toks[Z]));
        assert((int)(X.size() - 1) / 32 * 32 > branch && x.cached == 0 && y.cached == 0 && z.cached == branch);
        Run w = run(on, W), v = run(on, V);  // W's chunks are off's (its snapshot point is its prefill's end)
        printf("  aligned: W (%zu) cached %d, V cached %d\n", W.size(), w.cached, v.cached);
        assert(w.cached == 0 && v.cached == 768 && w.toks == off_toks[W]);

        // K/V and state after a hit vs cache off. long1281 hits the snapshot long1300 left at 1280 and goes straight
        // to decode, its chunks were off's: bit-exact. long1300 and Z2 resume mid-prompt (chunk boundaries move).
        Run h1 = run(on, long1281, 1), o1 = run(off, long1281, 1);
        assert(h1.cached == 1280 && h1.toks == o1.toks);
        compare("long 1281, K/V", h1.kv, o1.kv, true);
        compare("long 1281, state", h1.state, o1.state, true);
        Run h2 = run(on, long1300, 1), o2 = run(off, long1300, 1);
        assert(h2.cached == 1280);
        compare("long 1300, K/V", h2.kv, o2.kv, false);
        compare("long 1300, state", h2.state, o2.state, false);
        Run h3 = run(on, Z2, 1), o3 = run(off, Z2, 1);  // Z2: Y's branch point (Z left a snapshot at its own end)
        assert(h3.cached == branch);
        compare("Z2 (branch point), K/V", h3.kv, o3.kv, false);
        compare("Z2 (branch point), state", h3.state, o3.state, false);
        check_idle(on);
        printf("gpu chunk 512: cold == off, hits == cold\n");
    }
    std::map<std::vector<int>, std::vector<int>> off32;  // greedy 64, cache off, chunk 32
    {
        // chunk 32: every chunk is one page with or without the cache, so a hit must be bit-exact: tokens, K/V, state
        mllm::Engine off(golden::MODEL, ARENA, 32, 32, 16, 16384, "gpu", false);
        mllm::Engine on(golden::MODEL, ARENA, 32, 32, 16, 16384, "gpu", true);
        for (const auto& p : all) off32[p] = run(off, p).toks;
        for (const auto& p : {X, Y, Z, Z2, X}) {
            Run h = run(on, p), h1 = run(on, p, 1), o1 = run(off, p, 1);
            printf("  chunk 32, len %zu: cached %d then %d\n", p.size(), h.cached, h1.cached);
            assert(h.toks == off32[p] && h1.toks == o1.toks && h1.kv == o1.kv && h1.state == o1.state);
            assert(h1.cached == ((int)p.size() - 1) / 32 * 32);
            assert(&p != &Z2 || h.cached == branch);
        }
        check_idle(on);
        printf("gpu chunk 32: hits bit-exact with cache off (tokens, K/V, state)\n");
    }
    try {
        // ANE prefill: the same hits; long1300 tokens equal ANE's cache-off run (the others as at chunk 512)
        mllm::Engine off(golden::MODEL, ARENA, CHUNK, 32, 16, 16384, "ane", false);
        mllm::Engine on(golden::MODEL, ARENA, CHUNK, 32, 16, 16384, "ane", true);
        for (const auto& p : {long1300, X, Y, Z, Z3}) {
            Run o = run(off, p), cold = run(on, p), hit = run(on, p);
            const int P = ((int)p.size() - 1) / 32 * 32;
            printf("  ane len %4zu: cached %d then %d, tokens equal cache off up to %zu, %zu of 64\n", p.size(),
                   cold.cached, hit.cached, first_diff(cold.toks, o.toks), first_diff(hit.toks, o.toks));
            assert(hit.cached == P && (&p != &Z3 || cold.cached == branch));
            assert(&p != &long1300 || (cold.toks == o.toks && hit.toks == o.toks));
        }
        check_idle(on);
        printf("ane: hits where expected\n");
    } catch (const std::runtime_error& e) {  // another model ($MLLM_MODEL) may have no ANE blocks built yet
        if (!getenv("MLLM_MODEL")) throw;
        printf("ane: skipped (%.60s...)\n", e.what());
    }
    {
        // SMALL (0.8B: 170 units; one request's state is 50): admissions evict; 20 mixed requests at once. Chunk 32
        // so that outputs must equal cache off's solo runs.
        mllm::Engine on(golden::MODEL, SMALL, 32, 32, 16, 16384, "gpu", true);
        std::map<int, std::vector<int>> want, got;
        for (int i = 0; i < 20; i++) {
            const auto& p = all[(i / 2 * 5) % all.size()];  // in pairs: the second can hit what the first left
            want[on.submit(p, {.max_tokens = 16})] = head(off32[p], std::min<size_t>(16, off32[p].size()));
        }
        std::set<int> doomed;  // cancelled mid-prefill: their later pages must not be cached, their units return
        for (const auto& p : {long1300, X, long1300})
            doomed.insert(on.submit(p, {.max_tokens = 16}));
        std::set<uint64_t> seen;
        std::set<int> hit;
        while (!on.reqs.empty()) {
            for (const mllm::Event& ev : on.step()) got[ev.req_id].push_back(ev.token);
            for (int id : doomed)
                if (on.reqs.count(id) && on.reqs.at(id).pos >= 256 && !on.reqs.at(id).cancelled) on.cancel(id);
            for (auto& [h, _] : on.cache->nodes) seen.insert(h);
            for (auto& [id, r] : on.reqs)
                if (r.state != mllm::State::WAITING && r.cached > 0) hit.insert(id);
        }
        int evicted = 0;
        for (uint64_t h : seen) evicted += !on.cache->nodes.count(h);
        int same = 0;
        for (auto& [id, t] : want) same += got[id] == t;
        for (int id : doomed) assert(got[id].empty());
        printf("  small arena: %u units, %zu hits, %d nodes evicted, %zu left; %d / 20 outputs equal cache off, "
               "%zu cancelled\n", on.mem.arena.n_units, hit.size(), evicted, on.cache->nodes.size(), same, doomed.size());
        assert(same == 20 && evicted > 0 && !hit.empty());
        check_idle(on);
        // a hit whose demand plus its held pages and snapshot exceed the arena is admitted (without copies, or
        // cold), not kept waiting forever (QA: demand == n_units, the same prompt again)
        run(on, long1300, 16);  // cached at 1280 (two copies don't fit: it keeps the last one)
        const int id = on.submit(long1300, {.max_tokens = (int)(on.mem.arena.n_units - on.mem.state_units) * 32 - 1300});
        assert(on.cache->match(long1300).S > 0);
        const auto t0 = std::chrono::steady_clock::now();
        while (on.reqs.at(id).state == mllm::State::WAITING) {
            on.step();
            assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5) && "a big hit waits forever");
        }
        on.cancel(id);
        while (!on.reqs.empty()) on.step();
        check_idle(on);
    }
    if (part.empty()) stress(512, 3), stress(32, 1);
    printf("PASS\n");
}
