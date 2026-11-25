// PrefixCache + Arena (CPU only): insert, match with a branch, snapshots only where taken, a hash collision never
// matches, LRU leaf eviction freeing exactly the units no request holds, cache_only_units exact after every op, and
// admit evicting until the request fits.
#include <cassert>
#include <cstdio>

#include "prefix_cache.h"

constexpr int TOKENS_PER_PAGE = 4, STATE_UNITS = 3;
constexpr uint32_t ARENA_UNITS = 64;

// the arena's counts equal what its refcounts say
static void check_counts(const mllm::Arena& arena) {
    uint32_t cache_only = 0, unheld = 0;
    for (uint32_t unit = 0; unit < arena.n_units; unit++) {
        cache_only += !arena.request_refs[unit] && arena.cache_refs[unit];
        unheld += !arena.request_refs[unit] && !arena.cache_refs[unit];
    }
    assert(cache_only == arena.cache_only_units && unheld == arena.free_units);
}

// what the engine does for a request: own pages, a snapshot copy handed to the cache
struct CachedRequest {
    std::vector<int> token_ids;
    std::vector<uint32_t> pages;
};

static std::vector<uint32_t> allocate_units(mllm::Arena& arena, int count) {
    assert(arena.admit(count));
    arena.retire(arena.committed);  // released units are reusable (no GPU here)
    std::vector<uint32_t> units;
    for (int i = 0; i < count; i++) units.push_back(arena.alloc());
    return units;
}

// a request that matched what the cache has, computed the rest of its pages and snapshots at page page_count
static CachedRequest run_request(mllm::Arena& arena, mllm::PrefixCache& cache, std::vector<int> token_ids,
                                 int page_count) {
    mllm::PrefixCache::Match match = cache.match(token_ids);
    for (uint32_t unit : match.pages) arena.ref(unit);
    CachedRequest request{std::move(token_ids), match.pages};
    for (uint32_t unit : allocate_units(arena, page_count - match.resume_pages)) request.pages.push_back(unit);
    std::vector<uint32_t> snapshot = allocate_units(arena, STATE_UNITS);
    cache.insert(request.token_ids, page_count, request.pages, snapshot);
    arena.release(snapshot);
    check_counts(arena);
    return request;
}

// page k = TOKENS_PER_PAGE copies of k; then tail tokens
static std::vector<int> make_prompt(std::vector<int> page_values, int tail) {
    std::vector<int> token_ids;
    for (int value : page_values) token_ids.insert(token_ids.end(), TOKENS_PER_PAGE, value);
    token_ids.insert(token_ids.end(), tail, 99);
    return token_ids;
}

static uint64_t hash_of_node_starting_with(const mllm::PrefixCache& cache, int first_token) {
    uint64_t found = 0;
    for (auto& [hash, node] : cache.nodes)
        if (node.tokens[0] == first_token) found = hash;
    return found;
}

// The sections below run in order on one arena and cache; each picks up the state the one before left.
struct PrefixCacheTest {
    mllm::Arena arena{16384, ARENA_UNITS};
    mllm::PrefixCache cache{arena, TOKENS_PER_PAGE};
    const std::vector<int> prompt_a = make_prompt({1, 2, 3, 4, 5}, 1);
    const std::vector<int> prompt_c = make_prompt({1, 2, 3, 9}, 3);
    CachedRequest request_a, request_c;

    // A: 5 full pages + 1 token (the last token never counts), snapshot at its end
    void insert_first_request() {
        request_a = run_request(arena, cache, prompt_a, 5);
        assert(cache.nodes.size() == 5 && arena.cache_only_units == STATE_UNITS);
        arena.release(request_a.pages);
        check_counts(arena);
        assert(arena.cache_only_units == 5 + STATE_UNITS && arena.free_units == ARENA_UNITS - 5 - STATE_UNITS);
    }

    // the same prompt: everything; one token shorter: page 5 ends at length - 1, so it isn't matched
    void match_same_and_shorter_prompt() {
        mllm::PrefixCache::Match match = cache.match(prompt_a);
        assert(match.matched_pages == 5 && match.resume_pages == 5 && match.pages == request_a.pages &&
               match.snapshot.size() == STATE_UNITS);
        // computes nothing; its snapshot copy is dropped: page 5 keeps A's
        CachedRequest request_a_again = run_request(arena, cache, prompt_a, 5);
        assert(request_a_again.pages == request_a.pages && cache.match(prompt_a).snapshot == match.snapshot &&
               arena.cache_only_units == 5 + STATE_UNITS - 5);
        arena.release(request_a_again.pages);
        match = cache.match(make_prompt({1, 2, 3, 4, 5}, 0));
        assert(match.matched_pages == 4 && match.resume_pages == 0 && match.pages.empty() && match.snapshot.empty());
    }

    // B branches after page 3: keys / values match 3 pages, but no snapshot there; B snapshots at the branch point
    void branch_after_third_page() {
        const std::vector<int> prompt_b = make_prompt({1, 2, 3, 7, 8}, 2);
        mllm::PrefixCache::Match match = cache.match(prompt_b);
        assert(match.matched_pages == 3 && match.resume_pages == 0);
        // recomputes pages 0..2 into its own units; the cache keeps A's
        CachedRequest request_b = run_request(arena, cache, prompt_b, 3);
        assert(cache.nodes.size() == 5 && arena.cache_only_units == 5 + 2 * STATE_UNITS &&
               cache.has_snapshot(prompt_b, 3));
        for (int i = 0; i < 3; i++)
            assert(request_b.pages[i] != request_a.pages[i] && arena.cache_refs[request_b.pages[i]] == 0);
        arena.release(request_b.pages);
        check_counts(arena);
        assert(arena.cache_only_units == 5 + 2 * STATE_UNITS);
    }

    // differs from A only in page 1's last token: matches page 0 alone, never a deeper page or snapshot of A's
    // (llama.cpp f174553e1: a checkpoint reused for a prompt that diverged before it)
    void prompt_diverging_inside_a_page() {
        std::vector<int> prompt_d = prompt_a;
        prompt_d[2 * TOKENS_PER_PAGE - 1] = 77;
        mllm::PrefixCache::Match match = cache.match(prompt_d);
        assert(match.matched_pages == 1 && match.resume_pages == 0 && match.pages.empty() && match.snapshot.empty());
    }

    // C: same three pages, then its own: hits B's snapshot, reuses A's pages
    void resume_at_branch_snapshot() {
        mllm::PrefixCache::Match match = cache.match(prompt_c);
        assert(match.matched_pages == 3 && match.resume_pages == 3 &&
               match.pages == std::vector<uint32_t>(request_a.pages.begin(), request_a.pages.begin() + 3));
        request_c = run_request(arena, cache, prompt_c, 4);  // inserts page 3 = 9 under A's page 2
        assert(cache.nodes.size() == 6 && arena.cache_only_units == 5 + 3 * STATE_UNITS - 3);  // C holds pages 0..2
        for (int i = 0; i < 3; i++) assert(arena.request_refs[request_a.pages[i]] == 1);  // C holds them
    }

    // collision: a node whose hash matches but whose tokens differ is a miss, and insert stops there
    void hash_collision_is_a_miss() {
        const uint64_t page_4_hash = hash_of_node_starting_with(cache, 4);
        cache.nodes.at(page_4_hash).tokens[1] = 42;
        mllm::PrefixCache::Match match = cache.match(prompt_a);
        assert(match.matched_pages == 3 && match.resume_pages == 3);
        const uint32_t cache_only_before = arena.cache_only_units;
        std::vector<uint32_t> own_pages = allocate_units(arena, 5), snapshot = allocate_units(arena, STATE_UNITS);
        cache.insert(prompt_a, 5, own_pages, snapshot);
        arena.release(snapshot), arena.release(own_pages);
        check_counts(arena);
        assert(cache.nodes.size() == 6 && arena.cache_only_units == cache_only_before);
        cache.nodes.at(page_4_hash).tokens[1] = 4;
    }

    // LRU leaf eviction in age order; each victim frees only the units no request holds (C holds A's pages 0..2
    // and its own page 3)
    void evict_least_recently_used_leaves() {
        cache.match(prompt_c);  // C's path newest
        const uint32_t free_before = arena.free_units;
        assert(cache.evict());  // A's page 5 (oldest leaf): its unit + A's snapshot
        assert(arena.free_units == free_before + 1 + STATE_UNITS && !cache.has_snapshot(prompt_a, 5));
        check_counts(arena);
        assert(cache.evict());  // A's page 4: became a leaf, no snapshot
        assert(arena.free_units == free_before + 2 + STATE_UNITS && cache.nodes.size() == 4);
        check_counts(arena);
        assert(cache.evict());  // C's page 3: C holds it, so only its snapshot frees
        assert(arena.free_units == free_before + 2 + 2 * STATE_UNITS && arena.request_refs[request_c.pages[3]] == 1 &&
               !arena.cache_refs[request_c.pages[3]]);
        check_counts(arena);
        // pages 0..2 still cached (B's snapshot on page 2): cache-only again; page 3 frees
        arena.release(request_c.pages);
        check_counts(arena);
        assert(arena.cache_only_units == 3 + STATE_UNITS && arena.free_units == ARENA_UNITS - 3 - STATE_UNITS);
    }

    // admit with eviction: fits fails, can_ever holds, admit evicts until the request fits
    void admit_evicts_until_it_fits() {
        std::vector<uint32_t> big = allocate_units(arena, arena.free_units - 2);  // 2 unheld units left, 6 cache-only
        assert(!arena.fits(5) && arena.can_ever(5) && !arena.can_ever(9));
        auto evict = [&] { return cache.evict(); };
        assert(!arena.admit(9, evict) && cache.nodes.size() == 3);  // can_ever false: evicts nothing
        const uint64_t page_1_hash = hash_of_node_starting_with(cache, 1);
        cache.nodes.at(page_1_hash).last_used = 0;  // oldest, but not a leaf
        assert(arena.admit(5, evict));  // evicts page 2 (+ snapshot: 4 units), enough
        assert(cache.nodes.size() == 2 && cache.nodes.count(page_1_hash) && arena.reserved == 5 &&
               arena.cache_only_units == 2);
        check_counts(arena);
        arena.unreserve(5);
        assert(arena.admit(8, evict) && cache.nodes.empty() && arena.cache_only_units == 0);  // both left
        assert(!cache.evict());
        arena.unreserve(8), arena.release(big);
        check_counts(arena);
        assert(arena.free_units == ARENA_UNITS && arena.reserved == 0);
    }
};

int main() {
    PrefixCacheTest test;
    test.insert_first_request();
    test.match_same_and_shorter_prompt();
    test.branch_after_third_page();
    test.prompt_diverging_inside_a_page();
    test.resume_at_branch_snapshot();
    test.hash_collision_is_a_miss();
    test.evict_least_recently_used_leaves();
    test.admit_evicts_until_it_fits();
    printf("PASS\n");
}
