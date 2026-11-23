// Arena bookkeeping check (CPU only): admit up to the limit, alloc never fails after admit, refcounted release,
// pending reclaim gated by the GPU's step, and admit refusing a big request while accepting small ones.
#include <cassert>
#include <set>

#include "memory.h"

static int n_free_bits(const mllm::Arena& a) {
    int n = 0;
    for (uint64_t w : a.free) n += __builtin_popcountll(w);
    return n;
}

int main() {
    mllm::Arena a(16384, 70);  // 70: the last bitmap word is partial
    assert(n_free_bits(a) == 70 && a.free_units == 70);

    // admit up to the limit, then refuse
    assert(a.admit(40) && a.admit(30));
    assert(!a.admit(1) && a.reserved == 70);
    // every reserved unit allocates, all distinct
    std::set<uint32_t> got;
    for (int i = 0; i < 70; i++) got.insert(a.alloc());
    assert(got.size() == 70 && *got.rbegin() == 69 && a.reserved == 0 && a.free_units == 0 && n_free_bits(a) == 0);

    // release at committed step 10: pending, counted in free_units at once, not reusable before the GPU passes 10
    a.committed = 10;
    a.release({3, 4, 5});
    assert(a.free_units == 3 && a.pending.size() == 3 && n_free_bits(a) == 0);
    assert(a.admit(3));
    a.retire(9);
    assert(n_free_bits(a) == 0);  // step 10 may still read them
    a.retire(10);
    assert(n_free_bits(a) == 3 && a.pending.empty());
    std::set<uint32_t> again{a.alloc(), a.alloc(), a.alloc()};
    assert(again == std::set<uint32_t>({3, 4, 5}) && a.reserved == 0);

    // refcounts: a unit shared by two requests frees on the second release; one the cache holds becomes cache-only
    a.ref(7);
    a.committed = 11;
    a.release({7});
    assert(a.req_refs[7] == 1 && a.pending.empty() && a.free_units == 0);
    a.release({7});
    assert(a.pending.size() == 1 && a.free_units == 1);
    a.cache_ref(8);
    a.release({8});
    assert(a.free_units == 1 && a.cache_only_units == 1 && a.pending.size() == 1);
    a.ref(8), a.cache_release(8), a.release({8});  // cache-only -> held -> uncached -> pending
    a.retire(11);

    // unreserve returns an unused reservation
    assert(a.admit(1));
    a.unreserve(1);
    assert(a.reserved == 0);

    // no starvation: a big request is refused while a small one fits. The arena alone would let small requests
    // overtake the big one forever; the engine's FIFO admission (g5) stops at the first refusal, so the oldest
    // waiter gets the units as they free up.
    a.release({10, 11, 12, 13});  // free_units = 2 + 4
    a.retire(a.committed);
    assert(a.free_units == 6);
    assert(!a.admit(7) && a.admit(2) && !a.admit(5));
    a.unreserve(2);
    assert(a.admit(6));
    mllm::Arena b(16384, 10);  // n + reserved would wrap uint32
    assert(b.admit(1) && !b.admit(UINT32_MAX) && !b.can_ever(UINT32_MAX) && b.reserved == 1);
    printf("PASS\n");
}
