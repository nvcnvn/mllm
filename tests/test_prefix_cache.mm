// PrefixCache + Arena (CPU only): insert, match with a branch, snapshots only where taken, a hash collision never
// matches, LRU leaf eviction freeing exactly the units no request holds, cache_only_units exact after every op, and
// admit evicting until the request fits.
#include <cassert>
#include <cstdio>

#include "prefix_cache.h"

constexpr int T = 4, STATE = 3;

static void check(const mllm::Arena& a) {  // the counts from the refcounts
    uint32_t cache_only = 0, unheld = 0;
    for (uint32_t s = 0; s < a.n_units; s++) {
        cache_only += !a.request_refs[s] && a.cache_refs[s];
        unheld += !a.request_refs[s] && !a.cache_refs[s];
    }
    assert(cache_only == a.cache_only_units && unheld == a.free_units);
}

struct Req {  // what the engine does for a request: own pages, a snapshot copy handed to the cache
    std::vector<int> ids;
    std::vector<uint32_t> pages;
};

static std::vector<uint32_t> alloc(mllm::Arena& a, int n) {
    assert(a.admit(n));
    a.retire(a.committed);  // released units are reusable (no GPU here)
    std::vector<uint32_t> u;
    for (int i = 0; i < n; i++) u.push_back(a.alloc());
    return u;
}

// a request that matched m, computed the rest of its pages and snapshots at page n_pages
static Req run(mllm::Arena& a, mllm::PrefixCache& c, std::vector<int> ids, int n_pages) {
    mllm::PrefixCache::Match m = c.match(ids);
    for (uint32_t u : m.pages) a.ref(u);
    Req r{std::move(ids), m.pages};
    for (uint32_t u : alloc(a, n_pages - m.resume_pages)) r.pages.push_back(u);
    std::vector<uint32_t> snap = alloc(a, STATE);
    c.insert(r.ids, n_pages, r.pages, snap);
    a.release(snap);
    check(a);
    return r;
}

static std::vector<int> prompt(std::vector<int> pages, int tail) {  // page k = T copies of k; then tail tokens
    std::vector<int> ids;
    for (int p : pages) ids.insert(ids.end(), T, p);
    ids.insert(ids.end(), tail, 99);
    return ids;
}

int main() {
    mllm::Arena a(16384, 64);
    mllm::PrefixCache c(a, T);

    // A: 5 full pages + 1 token (the last token never counts), snapshot at its end
    const std::vector<int> ia = prompt({1, 2, 3, 4, 5}, 1);
    Req A = run(a, c, ia, 5);
    assert(c.nodes.size() == 5 && a.cache_only_units == STATE);
    a.release(A.pages);
    check(a);
    assert(a.cache_only_units == 5 + STATE && a.free_units == 64 - 5 - STATE);

    // the same prompt: everything; one token shorter: page 5 ends at len - 1, so it isn't matched
    mllm::PrefixCache::Match m = c.match(ia);
    assert(m.matched_pages == 5 && m.resume_pages == 5 && m.pages == A.pages && m.snapshot.size() == STATE);
    Req A2 = run(a, c, ia, 5);  // computes nothing; its snapshot copy is dropped: page 5 keeps A's
    assert(A2.pages == A.pages && c.match(ia).snapshot == m.snapshot && a.cache_only_units == 5 + STATE - 5);
    a.release(A2.pages);
    m = c.match(prompt({1, 2, 3, 4, 5}, 0));
    assert(m.matched_pages == 4 && m.resume_pages == 0 && m.pages.empty() && m.snapshot.empty());

    // B branches after page 3: K/V matches 3 pages, but no snapshot there; B snapshots at the branch point
    const std::vector<int> ib = prompt({1, 2, 3, 7, 8}, 2);
    m = c.match(ib);
    assert(m.matched_pages == 3 && m.resume_pages == 0);
    Req B = run(a, c, ib, 3);  // recomputes pages 0..2 into its own units; the cache keeps A's
    assert(c.nodes.size() == 5 && a.cache_only_units == 5 + 2 * STATE && c.has_snapshot(ib, 3));
    for (int i = 0; i < 3; i++) assert(B.pages[i] != A.pages[i] && a.cache_refs[B.pages[i]] == 0);
    a.release(B.pages);
    check(a);
    assert(a.cache_only_units == 5 + 2 * STATE);
    // differs from A only in page 1's last token: matches page 0 alone, never a deeper page or snapshot of A's
    // (llama.cpp f174553e1: a checkpoint reused for a prompt that diverged before it)
    std::vector<int> id = ia;
    id[2 * T - 1] = 77;
    m = c.match(id);
    assert(m.matched_pages == 1 && m.resume_pages == 0 && m.pages.empty() && m.snapshot.empty());
    // C: same three pages, then its own: hits B's snapshot, reuses A's pages
    const std::vector<int> ic = prompt({1, 2, 3, 9}, 3);
    m = c.match(ic);
    assert(m.matched_pages == 3 && m.resume_pages == 3 && m.pages == std::vector<uint32_t>(A.pages.begin(), A.pages.begin() + 3));
    Req C = run(a, c, ic, 4);  // inserts page 3 = 9 under A's page 2
    assert(c.nodes.size() == 6 && a.cache_only_units == 5 + 3 * STATE - 3);  // C holds pages 0..2
    for (int i = 0; i < 3; i++) assert(a.request_refs[A.pages[i]] == 1);  // C holds them

    // collision: a node whose hash matches but whose tokens differ is a miss, and insert stops there
    uint64_t h4 = 0;
    for (auto& [h, n] : c.nodes)
        if (n.tokens[0] == 4) h4 = h;
    c.nodes.at(h4).tokens[1] = 42;
    m = c.match(ia);
    assert(m.matched_pages == 3 && m.resume_pages == 3);
    const uint32_t before = a.cache_only_units;
    std::vector<uint32_t> own = alloc(a, 5), snap = alloc(a, STATE);
    c.insert(ia, 5, own, snap);
    a.release(snap), a.release(own);
    check(a);
    assert(c.nodes.size() == 6 && a.cache_only_units == before);
    c.nodes.at(h4).tokens[1] = 4;

    // LRU leaf eviction in age order; each victim frees only the units no request holds (C holds A's pages 0..2
    // and its own page 3)
    c.match(ic);  // C's path newest
    uint32_t free0 = a.free_units;
    assert(c.evict());  // A's page 5 (oldest leaf): its unit + A's snapshot
    assert(a.free_units == free0 + 1 + STATE && !c.has_snapshot(ia, 5));
    check(a);
    assert(c.evict());  // A's page 4: became a leaf, no snapshot
    assert(a.free_units == free0 + 2 + STATE && c.nodes.size() == 4);
    check(a);
    assert(c.evict());  // C's page 3: C holds it, so only its snapshot frees
    assert(a.free_units == free0 + 2 + 2 * STATE && a.request_refs[C.pages[3]] == 1 && !a.cache_refs[C.pages[3]]);
    check(a);
    a.release(C.pages);  // pages 0..2 still cached (B's snapshot on page 2): cache-only again; page 3 frees
    check(a);
    assert(a.cache_only_units == 3 + STATE && a.free_units == 64 - 3 - STATE);

    // admit with eviction: fits fails, can_ever holds, admit evicts until the request fits
    std::vector<uint32_t> big = alloc(a, a.free_units - 2);  // 2 unheld units left, 6 cache-only
    assert(!a.fits(5) && a.can_ever(5) && !a.can_ever(9));
    assert(!a.admit(9, [&] { return c.evict(); }) && c.nodes.size() == 3);  // can_ever false: evicts nothing
    uint64_t h1 = 0;
    for (auto& [h, n] : c.nodes)
        if (n.tokens[0] == 1) h1 = h;
    c.nodes.at(h1).last_used = 0;  // oldest, but not a leaf
    assert(a.admit(5, [&] { return c.evict(); }));  // evicts page 2 (+ snapshot: 4 units), enough
    assert(c.nodes.size() == 2 && c.nodes.count(h1) && a.reserved == 5 && a.cache_only_units == 2);
    check(a);
    a.unreserve(5);
    assert(a.admit(8, [&] { return c.evict(); }) && c.nodes.empty() && a.cache_only_units == 0);  // both left
    assert(!c.evict());
    a.unreserve(8), a.release(big);
    check(a);
    assert(a.free_units == 64 && a.reserved == 0);
    printf("PASS\n");
}
