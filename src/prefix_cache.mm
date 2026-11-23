#include "prefix_cache.h"

#include <algorithm>
#include <cassert>

namespace mllm {

uint64_t PrefixCache::hash(uint64_t prev, const int* page) const {  // FNV-1a over prev, then the page's ids
    uint64_t h = ROOT;
    auto mix = [&](const void* p, size_t n) {
        for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    };
    mix(&prev, 8);
    mix(page, T * sizeof(int));
    return h;
}

const PrefixCache::Node* PrefixCache::find(uint64_t prev, const int* page, uint64_t& h) const {
    h = hash(prev, page);
    auto it = nodes.find(h);
    if (it == nodes.end() || it->second.parent != prev || !std::equal(page, page + T, it->second.tokens.begin()))
        return nullptr;
    return &it->second;
}

PrefixCache::Match PrefixCache::match(const std::vector<int>& ids) {
    Match m;
    tick++;
    const int n = ids.empty() ? 0 : ((int)ids.size() - 1) / T;
    uint64_t prev = ROOT, h;
    for (int i = 0; i < n; i++, prev = h) {
        const Node* node = find(prev, ids.data() + i * T, h);
        if (!node) break;
        nodes.at(h).used = tick;
        m.L = i + 1;
        m.pages.push_back(node->unit);
        if (!node->snapshot.empty()) m.S = m.L, m.snapshot = node->snapshot;
    }
    m.pages.resize(m.S);
    return m;
}

bool PrefixCache::has_snapshot(const std::vector<int>& ids, int n_pages) const {
    uint64_t prev = ROOT, h;
    const Node* node = nullptr;
    for (int i = 0; i < n_pages; i++, prev = h)
        if (!(node = find(prev, ids.data() + i * T, h))) return false;
    return node && !node->snapshot.empty();
}

void PrefixCache::insert(const std::vector<int>& ids, int n_pages, const std::vector<uint32_t>& pages,
                         const std::vector<uint32_t>& snapshot) {
    assert(n_pages > 0 && n_pages * T <= (int)ids.size() && (int)pages.size() >= n_pages);
    tick++;
    uint64_t prev = ROOT, h = 0;
    for (int i = 0; i < n_pages; i++, prev = h) {
        const int* page = ids.data() + i * T;
        if (!find(prev, page, h)) {
            if (nodes.count(h)) return;  // ponytail: a collision; the page (and all below it) stays uncached
            nodes.emplace(h, Node{prev, std::vector<int>(page, page + T), pages[i], {}});
            arena.cache_ref(pages[i]);
            if (prev != ROOT) nodes.at(prev).children++;
        }
        nodes.at(h).used = tick;
    }
    Node& last = nodes.at(h);
    if (!last.snapshot.empty()) return;
    last.snapshot = snapshot;
    for (uint32_t u : snapshot) arena.cache_ref(u);
}

bool PrefixCache::evict() {
    // ponytail: an O(nodes) scan per victim; a heap of leaves if admit's eviction loop shows up in profiles
    auto victim = nodes.end();
    for (auto it = nodes.begin(); it != nodes.end(); ++it)
        if (!it->second.children && (victim == nodes.end() || it->second.used < victim->second.used)) victim = it;
    if (victim == nodes.end()) return false;
    const Node& v = victim->second;
    arena.cache_release(v.unit);
    for (uint32_t u : v.snapshot) arena.cache_release(u);
    if (v.parent != ROOT) nodes.at(v.parent).children--;
    nodes.erase(victim);
    return true;
}

}  // namespace mllm
