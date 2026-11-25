// Arena + PrefixCache under the engine's protocol, randomized (CPU only, CI). Sim is a model of Engine::step /
// admit / snapshot / finish / cancel and of the two lanes (decode: one step ahead, greedy rows fed into the next step;
// prefill: up to 2 chunks in flight on its own queue). It drives random requests: shared prefixes, repeats,
// multi-turn extensions, early EOS, cancels, requests as big as the arena, an arena small enough that admits evict.
// Every unit carries a shadow content: a hash of the tokens whose K/V or state it holds. Checked:
//   - after every operation: refcounts equal their holders (requests, cache nodes); free_units, cache_only_units and
//     reserved follow from them; reserved <= free_units; every unreferenced unit is in exactly one of the free bitmap
//     and pending, a referenced one in neither (no leak, no double free); node child counts, no orphan nodes
//   - alloc never returns a unit an in-flight decode step or prefill chunk reads (pending reclaim)
//   - nothing writes a unit the cache holds or another request shares (shared pages are immutable)
//   - a hit's pages and snapshot hold exactly that prefix's K/V and state (nothing stale, nothing evicted under it)
//   - every request ends (FIFO admission, no deadlock); afterwards, evicting everything frees every unit
//   - each scenario above actually happened (counters), so the test can't pass by not getting there.
// Mirrors src/engine.mm: when the engine's rules change, change Sim too. Hash collisions: test_prefix_cache.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <deque>
#include <map>
#include <random>
#include <set>

#include "prefix_cache.h"

constexpr int T = 4, STATE = 3, CHUNK = 8;

static uint64_t H(const std::vector<int>& seq, size_t n) {  // the content of a unit that holds seq[0, n)
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ (uint32_t)seq[i]) * 1099511628211ull;
    return h ^ n;
}

struct Stats {
    long hits = 0, evictions = 0, dup_snapshots = 0, snapshots = 0, released_in_flight = 0, two_chunks = 0,
         fed_ahead = 0, alloc_waits = 0, too_big_hits = 0, cancels = 0, eos = 0;
};

struct Req {
    int id;
    std::vector<int> ids, seq;  // seq: ids, then the tokens it will generate (fixed up front, so fed rows know them)
    int g;                      // max tokens
    enum { WAITING, PREFILL, DECODE } state = WAITING;
    int pos = 0, sent = 0;
    std::vector<int> snaps;
    std::vector<uint32_t> pages, state_slots;
    uint32_t reserved = 0;
    int out = 0;
    std::deque<uint64_t> steps;   // decode steps it is in
    std::deque<uint64_t> chunks;  // prefill chunks it is in
    bool cancelled = false;
    int in_flight() const { return (int)(steps.size() + chunks.size()); }
};

struct Lane {  // a queue with an event: work n is done when signaled >= n
    uint64_t committed = 0, signaled = 0;
    std::map<uint64_t, std::set<uint32_t>> reads;  // in-flight work -> units it reads
    void advance(uint64_t to) {
        signaled = std::max(signaled, std::min(to, committed));
        reads.erase(reads.begin(), reads.upper_bound(signaled));
    }
};

struct Sim {
    mllm::Arena a;
    mllm::PrefixCache c;
    std::mt19937& rng;
    Stats& st;
    std::vector<uint64_t> content;
    std::map<int, Req> reqs;  // by id = arrival order
    Lane decode, prefill;
    std::map<uint64_t, std::pair<int, int>> chunk_of;  // prefill chunk -> (req, end)
    int next_id = 1, ended = 0;

    Sim(uint32_t n, std::mt19937& rng, Stats& st) : a(16384, n), c(a, T), rng(rng), st(st), content(n) {}

    uint32_t demand(const Req& r) const { return (uint32_t)((r.ids.size() + r.g + T - 1) / T) + STATE; }

    uint32_t alloc(Req& r) {  // Engine::alloc
        a.retire(decode.signaled);
        if (std::none_of(a.free_bitmap.begin(), a.free_bitmap.end(), [](uint64_t w) { return w; })) {
            decode.advance(decode.committed);  // decode.wait()
            a.retire(decode.signaled);
            st.alloc_waits++;
        }
        assert(r.reserved > 0 && "request allocates past its reservation");
        r.reserved--;
        const uint32_t u = a.alloc();
        for (const Lane* l : {&decode, &prefill})
            for (auto& [_, units] : l->reads) assert(!units.count(u) && "alloc gave out a unit in-flight work reads");
        return u;
    }

    void write(uint32_t u, uint64_t v) {
        assert(a.request_refs[u] == 1 && !a.cache_refs[u] && "write to a shared or cached unit");
        content[u] = v;
    }

    bool admit(Req& r) {  // Engine::admit
        const int last = (int)r.ids.size() - 1;
        mllm::PrefixCache::Match m = c.match(r.ids);
        assert((int)m.pages.size() == m.resume_pages && m.snapshot.size() == (m.resume_pages ? STATE : 0u) && m.resume_pages <= m.matched_pages);
        std::vector<int> snaps;
        if (m.matched_pages > m.resume_pages) snaps.push_back(m.matched_pages * T);
        if (const int p = last / T * T; p > m.resume_pages * T && (snaps.empty() || p > snaps[0])) snaps.push_back(p);
        const auto need = [&] { return demand(r) - m.resume_pages + snaps.size() * STATE + m.resume_pages + m.snapshot.size(); };
        if (need() > a.n_units) st.too_big_hits++;
        while (need() > a.n_units && !snaps.empty()) snaps.erase(snaps.begin());
        if (need() > a.n_units) m = {};
        for (uint32_t u : m.pages) a.ref(u);
        for (uint32_t u : m.snapshot) a.ref(u);
        check(false);
        const uint32_t n = demand(r) - m.resume_pages + (uint32_t)snaps.size() * STATE;
        if (!a.admit(n, [&] { return c.evict() && ++st.evictions; })) {
            a.release(m.pages), a.release(m.snapshot);
            return false;
        }
        for (int i = 0; i < m.resume_pages; i++) assert(content[m.pages[i]] == H(r.ids, (i + 1) * T) && "stale page on a hit");
        for (uint32_t u : m.snapshot) assert(content[u] == H(r.ids, m.resume_pages * T) && "stale snapshot on a hit");
        st.hits += m.resume_pages > 0;
        r.reserved = n, r.pages = m.pages, r.pos = r.sent = m.resume_pages * T, r.snaps = std::move(snaps);
        for (int j = 0; j < STATE; j++) r.state_slots.push_back(alloc(r));
        for (uint32_t u : r.state_slots) write(u, m.resume_pages ? content[m.snapshot[0]] : H(r.ids, 0));
        a.release(m.snapshot);
        r.state = r.pos < last ? Req::PREFILL : Req::DECODE;
        return true;
    }

    void snapshot(Req& r) {  // Engine::snapshot
        const int n_pages = r.pos / T;
        assert(content[r.state_slots[0]] == H(r.ids, r.pos) && "snapshot of a state past its point");
        if (c.has_snapshot(r.ids, n_pages)) {
            a.unreserve(STATE);
            r.reserved -= STATE;
            st.dup_snapshots++;
            return;
        }
        std::vector<uint32_t> snap;
        for (int j = 0; j < STATE; j++) snap.push_back(alloc(r));
        for (uint32_t u : snap) write(u, content[r.state_slots[0]]);
        c.insert(r.ids, n_pages, r.pages, snap);
        a.release(snap);
        st.snapshots++;
    }

    void finish(Req& r) {  // Engine::finish; the caller erases r
        st.released_in_flight += !r.steps.empty();
        a.unreserve(r.reserved);
        std::vector<uint32_t> all = r.pages;
        all.insert(all.end(), r.state_slots.begin(), r.state_slots.end());
        a.release(all);
        r.pages.clear(), r.state_slots.clear(), r.reserved = 0;
        ended++;
    }

    void ensure_pages(Req& r, int n) {
        while ((int)r.pages.size() * T < n) r.pages.push_back(alloc(r));
    }

    void cancel(int id) {  // Engine::cancel
        Req& r = reqs.at(id);
        r.cancelled = true, st.cancels++;
        if (!r.in_flight()) finish(r), reqs.erase(id);
    }

    void step() {  // Engine::step, minus the GPU work
        a.retire(decode.signaled);
        std::set<int> done;
        // prefill completions, in chunk order
        for (auto it = chunk_of.begin(); it != chunk_of.end() && it->first <= prefill.signaled;) {
            auto [id, end] = it->second;
            it = chunk_of.erase(it);
            Req& r = reqs.at(id);
            assert(r.chunks.front() <= prefill.signaled);
            r.chunks.pop_front();
            if (r.cancelled) {
                if (!r.in_flight()) finish(r), done.insert(id);
                continue;
            }
            r.pos = end;
            if (!r.snaps.empty() && r.pos == r.snaps.front()) snapshot(r), r.snaps.erase(r.snaps.begin());
            if (r.pos == (int)r.ids.size() - 1) r.state = Req::DECODE;
            check(false);
        }
        // decode completions, in step order (a fed row has two)
        std::vector<std::pair<uint64_t, int>> toks;
        for (auto& [id, r] : reqs)
            for (uint64_t s : r.steps)
                if (s <= decode.signaled) toks.push_back({s, id});
        std::sort(toks.begin(), toks.end());
        for (auto [s, id] : toks) {
            if (done.count(id)) continue;  // ran one step past its end
            Req& r = reqs.at(id);
            assert(r.steps.front() == s);
            r.steps.pop_front();
            if (r.cancelled) {
                finish(r), done.insert(id);
                continue;
            }
            r.pos++, r.out++;
            const bool eos = rng() % 12 == 0;
            st.eos += eos;
            if (eos || r.out == r.g) finish(r), done.insert(id);
            check(false);
        }
        for (int id : done) reqs.erase(id);

        for (auto& [_, r] : reqs) {
            if (r.state != Req::WAITING) continue;
            if (!admit(r)) break;  // later requests never overtake the oldest waiter
            check(true);
        }

        // prefill: the oldest request with tokens left, up to 2 chunks in flight; never past a snapshot point
        // while the chunk ending there is still in flight
        while (prefill.committed - prefill.signaled < 2) {
            auto it = std::find_if(reqs.begin(), reqs.end(), [](auto& kv) {
                const Req& r = kv.second;
                return r.state == Req::PREFILL && !r.cancelled && r.sent < (int)r.ids.size() - 1 &&
                       !(r.in_flight() && !r.snaps.empty() && r.sent == r.snaps.front());
            });
            if (it == reqs.end()) break;
            Req& r = it->second;
            int end = std::min(r.sent + CHUNK, (int)r.ids.size() - 1);
            if (!r.snaps.empty()) end = std::min(end, r.snaps.front());
            ensure_pages(r, end);
            for (int p = r.sent / T; p * T < end; p++) write(r.pages[p], H(r.ids, std::min((p + 1) * T, end)));
            for (uint32_t u : r.state_slots) write(u, H(r.ids, end));
            const uint64_t k = ++prefill.committed;
            st.two_chunks += !r.chunks.empty();
            r.chunks.push_back(k), chunk_of[k] = {r.id, end};
            auto& units = prefill.reads[k];
            units.insert(r.pages.begin(), r.pages.end());
            units.insert(r.state_slots.begin(), r.state_slots.end());
            r.sent = end;
            check(true);
        }

        // decode: a row still in the step in flight joins the next one too, unless that step makes its last token
        std::vector<Req*> batch;
        for (auto& [_, r] : reqs)
            if (r.state == Req::DECODE && !r.cancelled &&
                (r.steps.empty() || (r.steps.size() == 1 && r.out + 1 < r.g)))
                batch.push_back(&r);
        if (!batch.empty() && decode.committed - decode.signaled < 2) {
            const uint64_t s = decode.committed + 1;
            std::set<uint32_t> units;
            for (Req* r : batch) {
                const int p = r->pos + (int)r->steps.size();
                st.fed_ahead += !r->steps.empty();
                ensure_pages(*r, p + 1);
                write(r->pages[p / T], H(r->seq, p + 1));
                for (uint32_t u : r->state_slots) write(u, H(r->seq, p + 1));
                r->steps.push_back(s);
                units.insert(r->pages.begin(), r->pages.end());
                units.insert(r->state_slots.begin(), r->state_slots.end());
            }
            decode.committed = a.committed = s;  // DecodeLane::submit
            decode.reads[s] = std::move(units);
            check(true);
        }
    }

    // exact: no admit in progress, so request refs equal their holders (admit holds a hit's pages before it owns them)
    void check(bool exact) const {
        std::vector<uint32_t> req(a.n_units), cache(a.n_units);
        std::map<uint64_t, int> kids;
        for (auto& [h, n] : c.nodes) {
            cache[n.unit]++;
            for (uint32_t u : n.snapshot) cache[u]++;
            if (n.parent != 14695981039346656037ull) assert(c.nodes.count(n.parent) && "orphan node"), kids[n.parent]++;
        }
        uint32_t reserved = 0;
        bool waiting = false;  // FIFO: the waiting requests are the newest ones
        for (auto& [_, r] : reqs) {
            assert((!waiting || r.state == Req::WAITING) && "a request overtook an older waiter");
            waiting |= r.state == Req::WAITING;
            reserved += r.reserved;
            for (uint32_t u : r.pages) req[u]++;
            for (uint32_t u : r.state_slots) req[u]++;
        }
        std::vector<int> pend(a.n_units);
        for (auto [s, step] : a.pending) pend[s]++, assert(step <= a.committed);
        uint32_t unheld = 0, cache_only = 0;
        for (uint32_t s = 0; s < a.n_units; s++) {
            assert(exact ? a.request_refs[s] == req[s] : a.request_refs[s] >= req[s]);
            assert(a.cache_refs[s] == cache[s] && cache[s] <= 1 && "a unit in two cache entries");
            const bool bit = a.free_bitmap[s / 64] >> (s % 64) & 1, refs = a.request_refs[s] || cache[s];
            assert((refs ? !bit && !pend[s] : bit + pend[s] == 1) && "leaked, double-freed or freed while held");
            unheld += !refs, cache_only += !a.request_refs[s] && cache[s];
        }
        assert(unheld == a.free_units && cache_only == a.cache_only_units && a.reserved <= a.free_units);
        assert(reserved == a.reserved);
        for (auto& [h, n] : c.nodes) assert(n.children == kids[h]);
    }
};

static void run(unsigned seed, uint32_t n_units, int n_requests, Stats& st) {
    std::mt19937 g(seed);
    Sim sim(n_units, g, st);
    std::vector<std::vector<int>> bases, history;
    for (int b = 0; b < 4; b++) {
        std::vector<int> p;
        for (int i = 0; i < (int)(g() % 6 + 1) * T; i++) p.push_back(g() % 8);
        bases.push_back(p);
    }
    std::vector<int> huge;  // demand == n_units: its hit plus copies can't fit next to it
    for (int i = 0; i < (int)(n_units - STATE) * T - 4; i++) huge.push_back(g() % 8);
    int submitted = 0;
    for (int tick = 0; sim.ended < n_requests; tick++) {
        assert(tick < n_requests * 400 && "no progress: a request never ended");
        if (submitted < n_requests && sim.reqs.size() < 10 && g() % 3 == 0) {
            std::vector<int> ids;
            const int k = g() % 40;
            if (k == 0) ids = huge;
            else if (k < 8 && !history.empty()) ids = history[g() % history.size()];   // repeated
            else if (k < 16 && !history.empty()) ids = history[g() % history.size()];  // multi-turn: + more below
            else if (k < 32) ids = bases[g() % bases.size()];                          // shared system prompt
            const int extra = k < 8 ? 0 : g() % 14 + 1;
            for (int i = 0; i < extra && ids != huge; i++) ids.push_back(g() % 8);
            if (ids.empty()) ids.push_back(g() % 8);
            int mt = g() % 6 + 1;
            if (ids == huge) mt = 4;
            else if ((ids.size() + mt + T - 1) / T + STATE > n_units) ids.resize(T);  // Engine::submit rejects these
            if (ids != huge) history.push_back(ids);
            Req r{.id = sim.next_id++, .ids = ids, .seq = ids, .g = mt};
            for (int i = 0; i < mt; i++) r.seq.push_back(g() % 8);
            sim.reqs.emplace(r.id, std::move(r));
            submitted++;
        }
        if (!sim.reqs.empty() && g() % 30 == 0) {
            auto it = sim.reqs.begin();
            std::advance(it, g() % sim.reqs.size());
            if (!it->second.cancelled) sim.cancel(it->first);
        }
        sim.step();
        sim.check(true);
        sim.decode.advance(sim.decode.signaled + g() % 2);  // the GPU: each lane finishes 0 or 1 more
        sim.prefill.advance(sim.prefill.signaled + g() % 2);
    }
    sim.decode.advance(sim.decode.committed);
    sim.a.retire(sim.decode.signaled);
    while (sim.c.evict()) sim.check(true);
    sim.a.retire(sim.decode.signaled);
    assert(sim.a.free_units == n_units && sim.a.reserved == 0 && sim.a.cache_only_units == 0 && sim.a.pending.empty());
    for (uint32_t s = 0; s < n_units; s++) assert(sim.a.free_bitmap[s / 64] >> (s % 64) & 1);
}

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 200;
    Stats st;
    for (int s = 0; s < seeds; s++) run(s, 40 + s % 3 * 30, 300, st);  // 40: admits evict constantly; 100: deep cache
    printf("hits %ld, evictions %ld, snapshots %ld (dup %ld), released in flight %ld, 2 chunks %ld, fed ahead %ld, "
           "alloc waits %ld, too-big hits %ld, cancels %ld, eos %ld\n",
           st.hits, st.evictions, st.snapshots, st.dup_snapshots, st.released_in_flight, st.two_chunks, st.fed_ahead,
           st.alloc_waits, st.too_big_hits, st.cancels, st.eos);
    for (long n : {st.hits, st.evictions, st.snapshots, st.dup_snapshots, st.released_in_flight, st.two_chunks,
                   st.fed_ahead, st.alloc_waits, st.too_big_hits, st.cancels, st.eos})
        assert(n > 0 && "a scenario never happened: the test lost its point");
    printf("PASS (%d seeds)\n", seeds);
}
