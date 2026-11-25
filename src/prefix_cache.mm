#include "prefix_cache.h"

#include <algorithm>
#include <cassert>

namespace mllm {

uint64_t PrefixCache::hash(uint64_t parent_hash, const int* page) const {  // FNV-1a over the parent, then the page's ids
    uint64_t result = ROOT_HASH;
    auto mix = [&](const void* bytes, size_t byte_count) {
        for (size_t i = 0; i < byte_count; i++) result = (result ^ ((const uint8_t*)bytes)[i]) * 1099511628211ull;
    };
    mix(&parent_hash, 8);
    mix(page, page_tokens * sizeof(int));
    return result;
}

const PrefixCache::Node* PrefixCache::find(uint64_t parent_hash, const int* page, uint64_t& page_hash) const {
    page_hash = hash(parent_hash, page);
    auto it = nodes.find(page_hash);
    if (it == nodes.end() || it->second.parent != parent_hash ||
        !std::equal(page, page + page_tokens, it->second.tokens.begin()))
        return nullptr;
    return &it->second;
}

PrefixCache::Match PrefixCache::match(const std::vector<int>& prompt) {
    Match found;
    use_clock++;
    const int full_pages = prompt.empty() ? 0 : ((int)prompt.size() - 1) / page_tokens;
    uint64_t parent_hash = ROOT_HASH, page_hash;
    for (int i = 0; i < full_pages; i++, parent_hash = page_hash) {
        const Node* node = find(parent_hash, prompt.data() + i * page_tokens, page_hash);
        if (!node) break;
        nodes.at(page_hash).last_used = use_clock;
        found.matched_pages = i + 1;
        found.pages.push_back(node->unit);
        if (!node->snapshot.empty()) found.resume_pages = found.matched_pages, found.snapshot = node->snapshot;
    }
    found.pages.resize(found.resume_pages);
    return found;
}

bool PrefixCache::has_snapshot(const std::vector<int>& prompt, int page_count) const {
    uint64_t parent_hash = ROOT_HASH, page_hash;
    const Node* node = nullptr;
    for (int i = 0; i < page_count; i++, parent_hash = page_hash)
        if (!(node = find(parent_hash, prompt.data() + i * page_tokens, page_hash))) return false;
    return node && !node->snapshot.empty();
}

void PrefixCache::insert(const std::vector<int>& prompt, int page_count, const std::vector<uint32_t>& pages,
                         const std::vector<uint32_t>& snapshot) {
    assert(page_count > 0 && page_count * page_tokens <= (int)prompt.size() && (int)pages.size() >= page_count);
    use_clock++;
    uint64_t parent_hash = ROOT_HASH, page_hash = 0;
    for (int i = 0; i < page_count; i++, parent_hash = page_hash) {
        const int* page = prompt.data() + i * page_tokens;
        if (!find(parent_hash, page, page_hash)) {
            if (nodes.count(page_hash)) return;  // ponytail: a collision; the page (and all below it) stays uncached
            nodes.emplace(page_hash, Node{parent_hash, std::vector<int>(page, page + page_tokens), pages[i], {}});
            arena.cache_ref(pages[i]);
            if (parent_hash != ROOT_HASH) nodes.at(parent_hash).children++;
        }
        nodes.at(page_hash).last_used = use_clock;
    }
    Node& last = nodes.at(page_hash);
    if (!last.snapshot.empty()) return;
    last.snapshot = snapshot;
    for (uint32_t unit : snapshot) arena.cache_ref(unit);
}

bool PrefixCache::evict() {
    // ponytail: an O(nodes) scan per victim; a heap of leaves if admit's eviction loop shows up in profiles
    auto victim = nodes.end();
    for (auto it = nodes.begin(); it != nodes.end(); ++it)
        if (!it->second.children && (victim == nodes.end() || it->second.last_used < victim->second.last_used))
            victim = it;
    if (victim == nodes.end()) return false;
    const Node& node = victim->second;
    arena.cache_release(node.unit);
    for (uint32_t unit : node.snapshot) arena.cache_release(unit);
    if (node.parent != ROOT_HASH) nodes.at(node.parent).children--;
    nodes.erase(victim);
    return true;
}

}  // namespace mllm
