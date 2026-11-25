// Arena bookkeeping check (CPU only): admit up to the limit, alloc never fails after admit, refcounted release,
// pending reclaim gated by the GPU's step, and admit refusing a big request while accepting small ones.
#include <cassert>
#include <set>

#include "memory.h"

static int count_free_bits(const mllm::Arena& arena) {
    int count = 0;
    for (uint64_t word : arena.free_bitmap) count += __builtin_popcountll(word);
    return count;
}

// admit up to the limit, then refuse; every reserved unit allocates, all distinct
static void check_admit_then_alloc(mllm::Arena& arena) {
    assert(arena.admit(40) && arena.admit(30));
    assert(!arena.admit(1) && arena.reserved == 70);
    std::set<uint32_t> allocated;
    for (int i = 0; i < 70; i++) allocated.insert(arena.alloc());
    assert(allocated.size() == 70 && *allocated.rbegin() == 69 && arena.reserved == 0 && arena.free_units == 0 &&
           count_free_bits(arena) == 0);
}

// release at committed step 10: pending, counted in free_units at once, not reusable before the GPU passes 10
static void check_pending_reclaim(mllm::Arena& arena) {
    arena.committed = 10;
    arena.release({3, 4, 5});
    assert(arena.free_units == 3 && arena.pending.size() == 3 && count_free_bits(arena) == 0);
    assert(arena.admit(3));
    arena.retire(9);
    assert(count_free_bits(arena) == 0);  // step 10 may still read them
    arena.retire(10);
    assert(count_free_bits(arena) == 3 && arena.pending.empty());
    std::set<uint32_t> reallocated{arena.alloc(), arena.alloc(), arena.alloc()};
    assert(reallocated == std::set<uint32_t>({3, 4, 5}) && arena.reserved == 0);
}

// refcounts: a unit shared by two requests frees on the second release; one the cache holds becomes cache-only
static void check_reference_counts(mllm::Arena& arena) {
    arena.ref(7);
    arena.committed = 11;
    arena.release({7});
    assert(arena.request_refs[7] == 1 && arena.pending.empty() && arena.free_units == 0);
    arena.release({7});
    assert(arena.pending.size() == 1 && arena.free_units == 1);
    arena.cache_ref(8);
    arena.release({8});
    assert(arena.free_units == 1 && arena.cache_only_units == 1 && arena.pending.size() == 1);
    arena.ref(8), arena.cache_release(8), arena.release({8});  // cache-only -> held -> uncached -> pending
    arena.retire(11);
}

// unreserve returns an unused reservation
static void check_unreserve(mllm::Arena& arena) {
    assert(arena.admit(1));
    arena.unreserve(1);
    assert(arena.reserved == 0);
}

// No starvation: a big request is refused while a small one fits. The arena alone would let small requests
// overtake the big one forever; the engine's FIFO admission (g5) stops at the first refusal, so the oldest
// waiter gets the units as they free up.
static void check_big_request_refused_while_small_fits(mllm::Arena& arena) {
    arena.release({10, 11, 12, 13});  // free_units = 2 + 4
    arena.retire(arena.committed);
    assert(arena.free_units == 6);
    assert(!arena.admit(7) && arena.admit(2) && !arena.admit(5));
    arena.unreserve(2);
    assert(arena.admit(6));
}

// a huge count plus what is reserved would wrap uint32: still refused
static void check_no_wraparound() {
    mllm::Arena arena(16384, 10);
    assert(arena.admit(1) && !arena.admit(UINT32_MAX) && !arena.can_ever(UINT32_MAX) && arena.reserved == 1);
}

int main() {
    mllm::Arena arena(16384, 70);  // 70: the last bitmap word is partial
    assert(count_free_bits(arena) == 70 && arena.free_units == 70);
    check_admit_then_alloc(arena);
    check_pending_reclaim(arena);
    check_reference_counts(arena);
    check_unreserve(arena);
    check_big_request_refused_while_small_fits(arena);
    check_no_wraparound();
    printf("PASS\n");
}
