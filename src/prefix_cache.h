// PrefixCache: full prompt pages shared across requests, with linear-state snapshots on some pages (C++ only; no
// py/ counterpart). memory-design "Sharing". Engine thread only, no locks.
//
// A tree of pages, keyed by chained hashes: page i's hash = hash(page i - 1's hash, page i's page_tokens token ids).
// Every node holds its page's K/V unit; a node may also hold a copy of the linear-attention state after its page
// (state_units units). K/V can be reused up to any matched page, but a request resumes only at a node with a
// snapshot (the state is per request and updated in place). Units are held by Arena cache refs; a request using one
// takes a request ref on top.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "memory.h"

namespace mllm {

class PrefixCache {
public:
    PrefixCache(Arena& arena, int page_tokens) : arena(arena), page_tokens(page_tokens) {}

    struct Match {
        int matched_pages = 0;  // pages found in the cache
        int resume_pages = 0;   // pages up to the deepest matched snapshot (<= matched_pages): the request starts there
        std::vector<uint32_t> pages;     // the K/V units of pages [0, resume_pages)
        std::vector<uint32_t> snapshot;  // page resume_pages - 1's state copy (empty when resume_pages = 0)
    };
    // Only pages fully below prompt.size() - 1 (the last prompt token is fed by decode). Marks pages
    // [0, matched_pages) as just used.
    Match match(const std::vector<int>& prompt);
    bool has_snapshot(const std::vector<int>& prompt, int page_count) const;
    // Nodes for pages [0, page_count) of prompt (units: pages[i]); nodes that exist are kept (the caller's duplicate
    // page stays its own). snapshot goes on page page_count - 1 unless that node has one. The caller's units must be
    // live.
    void insert(const std::vector<int>& prompt, int page_count, const std::vector<uint32_t>& pages,
                const std::vector<uint32_t>& snapshot);
    bool evict();  // the least recently used leaf; false = empty

    struct Node {
        uint64_t parent;          // the previous page's hash (ROOT_HASH for page 0)
        std::vector<int> tokens;  // verified on lookup: a hash collision must never return wrong K/V
        uint32_t unit;
        std::vector<uint32_t> snapshot;
        int children = 0;
        uint64_t last_used = 0;
    };
    std::unordered_map<uint64_t, Node> nodes;  // by page hash

private:
    static constexpr uint64_t ROOT_HASH = 14695981039346656037ull;  // FNV-1a offset basis: the hash before page 0
    uint64_t hash(uint64_t parent_hash, const int* page) const;
    // the page's node if it exists and matches (tokens and parent), else nullptr; page_hash: its hash either way
    const Node* find(uint64_t parent_hash, const int* page, uint64_t& page_hash) const;

    Arena& arena;
    int page_tokens;
    uint64_t use_clock = 0;  // ticks once per match / insert; Node::last_used holds its value then
};

}  // namespace mllm
