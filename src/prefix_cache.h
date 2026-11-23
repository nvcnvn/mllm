// PrefixCache: full prompt pages shared across requests, with linear-state snapshots on some pages (C++ only; no
// py/ counterpart). memory-design "Sharing". Engine thread only, no locks.
//
// A page-level trie keyed by chained hashes (h_i = hash(h_{i-1}, page i's T token ids)). Every node holds its page's
// K/V unit; a node may also hold a copy of the linear-attention state after its page (state_units units). K/V can
// be reused up to any matched page, but a request resumes only at a node with a snapshot (the state is per request
// and updated in place). Units are held by Arena cache refs; a request using one takes a request ref on top.
#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "memory.h"

namespace mllm {

class PrefixCache {
public:
    PrefixCache(Arena& arena, int page_tokens) : arena(arena), T(page_tokens) {}

    struct Match {
        int L = 0, S = 0;                 // pages matched; pages up to the deepest matched snapshot (S <= L)
        std::vector<uint32_t> pages;      // the K/V units of pages [0, S)
        std::vector<uint32_t> snapshot;   // page S - 1's state copy (empty when S = 0)
    };
    // Only pages fully below ids.size() - 1 (the last prompt token is fed by decode). Bumps last-used along [0, L).
    Match match(const std::vector<int>& ids);
    bool has_snapshot(const std::vector<int>& ids, int n_pages) const;
    // Nodes for pages [0, n_pages) of ids (units: pages[i]); nodes that exist are kept (the caller's duplicate page
    // stays its own). snapshot goes on page n_pages - 1 unless that node has one. The caller's units must be live.
    void insert(const std::vector<int>& ids, int n_pages, const std::vector<uint32_t>& pages,
                const std::vector<uint32_t>& snapshot);
    bool evict();  // the least recently used leaf; false = empty

    struct Node {
        uint64_t parent;
        std::vector<int> tokens;  // verified on lookup: a hash collision must never return wrong K/V
        uint32_t unit;
        std::vector<uint32_t> snapshot;
        int children = 0;
        uint64_t used = 0;
    };
    std::unordered_map<uint64_t, Node> nodes;

private:
    static constexpr uint64_t ROOT = 14695981039346656037ull;  // FNV-1a offset basis: h_{-1}
    uint64_t hash(uint64_t prev, const int* page) const;
    // page i's node if it exists and matches (tokens and parent), else nullptr; h: its hash
    const Node* find(uint64_t prev, const int* page, uint64_t& h) const;

    Arena& arena;
    int T;
    uint64_t tick = 0;
};

}  // namespace mllm
