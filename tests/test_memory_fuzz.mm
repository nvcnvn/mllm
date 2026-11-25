// Arena + PrefixCache under the engine's protocol, randomized (CPU only, CI). Simulation is a model of Engine::step /
// admit / save_snapshot / finish / cancel and of the two lanes (decode: one step ahead, greedy rows fed into the next
// step; prefill: up to 2 chunks in flight on its own queue). It drives random requests: shared prefixes, repeats,
// multi-turn extensions, early end of sequence, cancels, requests as big as the arena, an arena small enough that
// admits evict. Every unit carries a shadow content: a hash of the tokens whose keys / values or state it holds.
// Checked:
//   - after every operation: refcounts equal their holders (requests, cache nodes); free_units, cache_only_units and
//     reserved follow from them; reserved <= free_units; every unreferenced unit is in exactly one of the free bitmap
//     and pending, a referenced one in neither (no leak, no double free); node child counts, no orphan nodes
//   - alloc never returns a unit an in-flight decode step or prefill chunk reads (pending reclaim)
//   - nothing writes a unit the cache holds or another request shares (shared pages are immutable)
//   - a hit's pages and snapshot hold exactly that prefix's keys / values and state (nothing stale, nothing evicted
//     under it)
//   - every request ends (FIFO admission, no deadlock); afterwards, evicting everything frees every unit
//   - each scenario above actually happened (counters), so the test can't pass by not getting there.
// Mirrors src/engine.mm: when the engine's rules change, change Simulation too. Hash collisions: test_prefix_cache.
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <deque>
#include <map>
#include <random>
#include <set>

#include "prefix_cache.h"

constexpr int TOKENS_PER_PAGE = 4, STATE_UNITS = 3, CHUNK_TOKENS = 8;
constexpr uint64_t CACHE_ROOT_HASH = 14695981039346656037ull;  // PrefixCache's hash before page 0: a node's parent

// the content of a unit that holds tokens[0, count)
static uint64_t content_hash(const std::vector<int>& tokens, size_t count) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < count; i++) hash = (hash ^ (uint32_t)tokens[i]) * 1099511628211ull;
    return hash ^ count;
}

// how often each scenario happened, over every seed
struct Stats {
    long hits = 0, evictions = 0, duplicate_snapshots = 0, snapshots = 0, released_in_flight = 0, two_chunks = 0,
         fed_ahead = 0, alloc_waits = 0, too_big_hits = 0, cancels = 0, early_ends = 0;
};

struct SimRequest {
    int id;
    // sequence: the prompt, then the tokens it will generate (fixed up front, so fed rows know them)
    std::vector<int> prompt, sequence;
    int max_tokens;
    enum { WAITING, PREFILL, DECODE } state = WAITING;
    int position = 0, sent = 0;
    std::vector<int> snapshot_points;
    std::vector<uint32_t> pages, state_slots;
    uint32_t reserved = 0;
    int generated = 0;
    std::deque<uint64_t> decode_steps;    // decode steps it is in
    std::deque<uint64_t> prefill_chunks;  // prefill chunks it is in
    bool cancelled = false;
    int in_flight() const { return (int)(decode_steps.size() + prefill_chunks.size()); }
};

// a queue with an event: work n is done when signaled >= n
struct SimLane {
    uint64_t committed = 0, signaled = 0;
    std::map<uint64_t, std::set<uint32_t>> units_read;  // in-flight work -> units it reads
    void advance(uint64_t to) {
        signaled = std::max(signaled, std::min(to, committed));
        units_read.erase(units_read.begin(), units_read.upper_bound(signaled));
    }
};

struct Simulation {
    mllm::Arena arena;
    mllm::PrefixCache cache;
    std::mt19937& random_engine;
    Stats& stats;
    std::vector<uint64_t> unit_content;
    std::map<int, SimRequest> requests;  // by id = arrival order
    SimLane decode, prefill;
    std::map<uint64_t, std::pair<int, int>> chunk_owner;  // prefill chunk -> (request id, end position)
    int next_request_id = 1, ended_requests = 0;

    Simulation(uint32_t unit_count, std::mt19937& random_engine, Stats& stats)
        : arena(16384, unit_count), cache(arena, TOKENS_PER_PAGE), random_engine(random_engine), stats(stats),
          unit_content(unit_count) {}

    uint32_t units_needed(const SimRequest& request) const {
        return (uint32_t)((request.prompt.size() + request.max_tokens + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE) +
               STATE_UNITS;
    }

    // Engine::allocate_unit
    uint32_t allocate_unit(SimRequest& request) {
        arena.retire(decode.signaled);
        if (std::none_of(arena.free_bitmap.begin(), arena.free_bitmap.end(), [](uint64_t word) { return word; })) {
            decode.advance(decode.committed);  // decode.wait()
            arena.retire(decode.signaled);
            stats.alloc_waits++;
        }
        assert(request.reserved > 0 && "request allocates past its reservation");
        request.reserved--;
        const uint32_t unit = arena.alloc();
        for (const SimLane* lane : {&decode, &prefill})
            for (auto& [_, units] : lane->units_read)
                assert(!units.count(unit) && "alloc gave out a unit in-flight work reads");
        return unit;
    }

    void write_unit(uint32_t unit, uint64_t content) {
        assert(arena.request_refs[unit] == 1 && !arena.cache_refs[unit] && "write to a shared or cached unit");
        unit_content[unit] = content;
    }

    // Engine::admit
    bool admit(SimRequest& request) {
        const int last_position = (int)request.prompt.size() - 1;
        mllm::PrefixCache::Match match = cache.match(request.prompt);
        assert((int)match.pages.size() == match.resume_pages &&
               match.snapshot.size() == (match.resume_pages ? STATE_UNITS : 0u) &&
               match.resume_pages <= match.matched_pages);
        std::vector<int> snapshot_points;
        if (match.matched_pages > match.resume_pages) snapshot_points.push_back(match.matched_pages * TOKENS_PER_PAGE);
        if (const int page_start = last_position / TOKENS_PER_PAGE * TOKENS_PER_PAGE;
            page_start > match.resume_pages * TOKENS_PER_PAGE &&
            (snapshot_points.empty() || page_start > snapshot_points[0]))
            snapshot_points.push_back(page_start);
        const auto units_with_hit = [&] {
            return units_needed(request) - match.resume_pages + snapshot_points.size() * STATE_UNITS +
                   match.resume_pages + match.snapshot.size();
        };
        if (units_with_hit() > arena.n_units) stats.too_big_hits++;
        while (units_with_hit() > arena.n_units && !snapshot_points.empty())
            snapshot_points.erase(snapshot_points.begin());
        if (units_with_hit() > arena.n_units) match = {};
        for (uint32_t unit : match.pages) arena.ref(unit);
        for (uint32_t unit : match.snapshot) arena.ref(unit);
        check_bookkeeping(false);
        const uint32_t units_to_reserve =
            units_needed(request) - match.resume_pages + (uint32_t)snapshot_points.size() * STATE_UNITS;
        if (!arena.admit(units_to_reserve, [&] { return cache.evict() && ++stats.evictions; })) {
            arena.release(match.pages), arena.release(match.snapshot);
            return false;
        }
        for (int i = 0; i < match.resume_pages; i++)
            assert(unit_content[match.pages[i]] == content_hash(request.prompt, (i + 1) * TOKENS_PER_PAGE) &&
                   "stale page on a hit");
        for (uint32_t unit : match.snapshot)
            assert(unit_content[unit] == content_hash(request.prompt, match.resume_pages * TOKENS_PER_PAGE) &&
                   "stale snapshot on a hit");
        stats.hits += match.resume_pages > 0;
        request.reserved = units_to_reserve, request.pages = match.pages;
        request.position = request.sent = match.resume_pages * TOKENS_PER_PAGE;
        request.snapshot_points = std::move(snapshot_points);
        for (int j = 0; j < STATE_UNITS; j++) request.state_slots.push_back(allocate_unit(request));
        for (uint32_t unit : request.state_slots)
            write_unit(unit, match.resume_pages ? unit_content[match.snapshot[0]] : content_hash(request.prompt, 0));
        arena.release(match.snapshot);
        request.state = request.position < last_position ? SimRequest::PREFILL : SimRequest::DECODE;
        return true;
    }

    // Engine::save_snapshot
    void save_snapshot(SimRequest& request) {
        const int page_count = request.position / TOKENS_PER_PAGE;
        assert(unit_content[request.state_slots[0]] == content_hash(request.prompt, request.position) &&
               "snapshot of a state past its point");
        if (cache.has_snapshot(request.prompt, page_count)) {
            arena.unreserve(STATE_UNITS);
            request.reserved -= STATE_UNITS;
            stats.duplicate_snapshots++;
            return;
        }
        std::vector<uint32_t> snapshot;
        for (int j = 0; j < STATE_UNITS; j++) snapshot.push_back(allocate_unit(request));
        for (uint32_t unit : snapshot) write_unit(unit, unit_content[request.state_slots[0]]);
        cache.insert(request.prompt, page_count, request.pages, snapshot);
        arena.release(snapshot);
        stats.snapshots++;
    }

    // Engine::finish; the caller erases the request
    void finish(SimRequest& request) {
        stats.released_in_flight += !request.decode_steps.empty();
        arena.unreserve(request.reserved);
        std::vector<uint32_t> all_units = request.pages;
        all_units.insert(all_units.end(), request.state_slots.begin(), request.state_slots.end());
        arena.release(all_units);
        request.pages.clear(), request.state_slots.clear(), request.reserved = 0;
        ended_requests++;
    }

    void ensure_pages(SimRequest& request, int tokens) {
        while ((int)request.pages.size() * TOKENS_PER_PAGE < tokens) request.pages.push_back(allocate_unit(request));
    }

    // Engine::cancel
    void cancel(int request_id) {
        SimRequest& request = requests.at(request_id);
        request.cancelled = true, stats.cancels++;
        if (!request.in_flight()) finish(request), requests.erase(request_id);
    }

    // Engine::step, minus the GPU work
    void step() {
        arena.retire(decode.signaled);
        std::set<int> finished;
        take_finished_chunks(finished);
        take_decoded_tokens(finished);
        for (int request_id : finished) requests.erase(request_id);
        admit_waiting_requests();
        submit_prefill_chunks();
        submit_decode_step();
    }

    // prefill completions, in chunk order
    void take_finished_chunks(std::set<int>& finished) {
        for (auto it = chunk_owner.begin(); it != chunk_owner.end() && it->first <= prefill.signaled;) {
            auto [request_id, end] = it->second;
            it = chunk_owner.erase(it);
            SimRequest& request = requests.at(request_id);
            assert(request.prefill_chunks.front() <= prefill.signaled);
            request.prefill_chunks.pop_front();
            if (request.cancelled) {
                if (!request.in_flight()) finish(request), finished.insert(request_id);
                continue;
            }
            request.position = end;
            if (!request.snapshot_points.empty() && request.position == request.snapshot_points.front())
                save_snapshot(request), request.snapshot_points.erase(request.snapshot_points.begin());
            if (request.position == (int)request.prompt.size() - 1) request.state = SimRequest::DECODE;
            check_bookkeeping(false);
        }
    }

    // decode completions, in step order (a fed row has two)
    void take_decoded_tokens(std::set<int>& finished) {
        std::vector<std::pair<uint64_t, int>> decoded;  // (step, request id)
        for (auto& [request_id, request] : requests)
            for (uint64_t step : request.decode_steps)
                if (step <= decode.signaled) decoded.push_back({step, request_id});
        std::sort(decoded.begin(), decoded.end());
        for (auto [step, request_id] : decoded) {
            if (finished.count(request_id)) continue;  // ran one step past its end
            SimRequest& request = requests.at(request_id);
            assert(request.decode_steps.front() == step);
            request.decode_steps.pop_front();
            if (request.cancelled) {
                finish(request), finished.insert(request_id);
                continue;
            }
            request.position++, request.generated++;
            const bool ends_early = random_engine() % 12 == 0;
            stats.early_ends += ends_early;
            if (ends_early || request.generated == request.max_tokens) finish(request), finished.insert(request_id);
            check_bookkeeping(false);
        }
    }

    void admit_waiting_requests() {
        for (auto& [_, request] : requests) {
            if (request.state != SimRequest::WAITING) continue;
            if (!admit(request)) break;  // later requests never overtake the oldest waiter
            check_bookkeeping(true);
        }
    }

    // the oldest request with tokens left to prefill; never past a snapshot point while the chunk ending there is
    // still in flight
    SimRequest* next_to_prefill() {
        auto it = std::find_if(requests.begin(), requests.end(), [](auto& entry) {
            const SimRequest& request = entry.second;
            return request.state == SimRequest::PREFILL && !request.cancelled &&
                   request.sent < (int)request.prompt.size() - 1 &&
                   !(request.in_flight() && !request.snapshot_points.empty() &&
                     request.sent == request.snapshot_points.front());
        });
        return it == requests.end() ? nullptr : &it->second;
    }

    // prefill: up to 2 chunks in flight
    void submit_prefill_chunks() {
        while (prefill.committed - prefill.signaled < 2) {
            SimRequest* next = next_to_prefill();
            if (!next) break;
            SimRequest& request = *next;
            int end = std::min(request.sent + CHUNK_TOKENS, (int)request.prompt.size() - 1);
            if (!request.snapshot_points.empty()) end = std::min(end, request.snapshot_points.front());
            ensure_pages(request, end);
            for (int page = request.sent / TOKENS_PER_PAGE; page * TOKENS_PER_PAGE < end; page++)
                write_unit(request.pages[page],
                           content_hash(request.prompt, std::min((page + 1) * TOKENS_PER_PAGE, end)));
            for (uint32_t unit : request.state_slots) write_unit(unit, content_hash(request.prompt, end));
            const uint64_t chunk_number = ++prefill.committed;
            stats.two_chunks += !request.prefill_chunks.empty();
            request.prefill_chunks.push_back(chunk_number), chunk_owner[chunk_number] = {request.id, end};
            auto& units = prefill.units_read[chunk_number];
            units.insert(request.pages.begin(), request.pages.end());
            units.insert(request.state_slots.begin(), request.state_slots.end());
            request.sent = end;
            check_bookkeeping(true);
        }
    }

    // decode: a row still in the step in flight joins the next one too, unless that step makes its last token
    void submit_decode_step() {
        std::vector<SimRequest*> batch;
        for (auto& [_, request] : requests)
            if (request.state == SimRequest::DECODE && !request.cancelled &&
                (request.decode_steps.empty() ||
                 (request.decode_steps.size() == 1 && request.generated + 1 < request.max_tokens)))
                batch.push_back(&request);
        if (batch.empty() || decode.committed - decode.signaled >= 2) return;
        const uint64_t step = decode.committed + 1;
        std::set<uint32_t> units;
        for (SimRequest* request : batch) add_decode_row(*request, step, units);
        decode.committed = arena.committed = step;  // DecodeLane::submit
        decode.units_read[step] = std::move(units);
        check_bookkeeping(true);
    }

    void add_decode_row(SimRequest& request, uint64_t step, std::set<uint32_t>& units) {
        const int position = request.position + (int)request.decode_steps.size();
        stats.fed_ahead += !request.decode_steps.empty();
        ensure_pages(request, position + 1);
        write_unit(request.pages[position / TOKENS_PER_PAGE], content_hash(request.sequence, position + 1));
        for (uint32_t unit : request.state_slots) write_unit(unit, content_hash(request.sequence, position + 1));
        request.decode_steps.push_back(step);
        units.insert(request.pages.begin(), request.pages.end());
        units.insert(request.state_slots.begin(), request.state_slots.end());
    }

    // exact: no admit in progress, so request refs equal their holders (admit holds a hit's pages before it owns them)
    void check_bookkeeping(bool exact) const {
        std::vector<uint32_t> expected_request_refs(arena.n_units), expected_cache_refs(arena.n_units);
        std::map<uint64_t, int> child_counts;
        for (auto& [hash, node] : cache.nodes) {
            expected_cache_refs[node.unit]++;
            for (uint32_t unit : node.snapshot) expected_cache_refs[unit]++;
            if (node.parent != CACHE_ROOT_HASH)
                assert(cache.nodes.count(node.parent) && "orphan node"), child_counts[node.parent]++;
        }
        uint32_t reserved = 0;
        bool waiting = false;  // FIFO: the waiting requests are the newest ones
        for (auto& [_, request] : requests) {
            assert((!waiting || request.state == SimRequest::WAITING) && "a request overtook an older waiter");
            waiting |= request.state == SimRequest::WAITING;
            reserved += request.reserved;
            for (uint32_t unit : request.pages) expected_request_refs[unit]++;
            for (uint32_t unit : request.state_slots) expected_request_refs[unit]++;
        }
        std::vector<int> pending_count(arena.n_units);
        for (auto [unit, step] : arena.pending) pending_count[unit]++, assert(step <= arena.committed);
        uint32_t unheld = 0, cache_only = 0;
        for (uint32_t unit = 0; unit < arena.n_units; unit++) {
            assert(exact ? arena.request_refs[unit] == expected_request_refs[unit]
                         : arena.request_refs[unit] >= expected_request_refs[unit]);
            assert(arena.cache_refs[unit] == expected_cache_refs[unit] && expected_cache_refs[unit] <= 1 &&
                   "a unit in two cache entries");
            const bool free_bit = arena.free_bitmap[unit / 64] >> (unit % 64) & 1,
                       has_refs = arena.request_refs[unit] || expected_cache_refs[unit];
            assert((has_refs ? !free_bit && !pending_count[unit] : free_bit + pending_count[unit] == 1) &&
                   "leaked, double-freed or freed while held");
            unheld += !has_refs, cache_only += !arena.request_refs[unit] && expected_cache_refs[unit];
        }
        assert(unheld == arena.free_units && cache_only == arena.cache_only_units && arena.reserved <= arena.free_units);
        assert(reserved == arena.reserved);
        for (auto& [hash, node] : cache.nodes) assert(node.children == child_counts[hash]);
    }
};

// One run's random prompts. Every draw comes from the run's random engine, in a fixed order.
struct PromptMaker {
    std::mt19937& random_engine;
    uint32_t unit_count;
    std::vector<std::vector<int>> shared_prompts, past_prompts;
    std::vector<int> huge_prompt;  // demand == unit_count: its hit plus copies can't fit next to it

    PromptMaker(std::mt19937& random_engine, uint32_t unit_count) : random_engine(random_engine), unit_count(unit_count) {
        for (int prompt = 0; prompt < 4; prompt++) shared_prompts.push_back(random_shared_prompt());
        for (int i = 0; i < (int)(unit_count - STATE_UNITS) * TOKENS_PER_PAGE - 4; i++)
            huge_prompt.push_back(random_engine() % 8);
    }

    std::vector<int> random_shared_prompt() {
        std::vector<int> prompt;
        for (int i = 0; i < (int)(random_engine() % 6 + 1) * TOKENS_PER_PAGE; i++) prompt.push_back(random_engine() % 8);
        return prompt;
    }

    // a new request: huge, repeated, a multi-turn extension, a shared system prompt plus more, or fresh
    SimRequest random_request(int request_id) {
        std::vector<int> prompt;
        const int kind = random_engine() % 40;
        if (kind == 0) prompt = huge_prompt;
        else if (kind < 8 && !past_prompts.empty())  // repeated
            prompt = past_prompts[random_engine() % past_prompts.size()];
        else if (kind < 16 && !past_prompts.empty())  // multi-turn: + more below
            prompt = past_prompts[random_engine() % past_prompts.size()];
        else if (kind < 32)  // shared system prompt
            prompt = shared_prompts[random_engine() % shared_prompts.size()];
        const int extra_tokens = kind < 8 ? 0 : random_engine() % 14 + 1;
        for (int i = 0; i < extra_tokens && prompt != huge_prompt; i++) prompt.push_back(random_engine() % 8);
        if (prompt.empty()) prompt.push_back(random_engine() % 8);
        int max_tokens = random_engine() % 6 + 1;
        if (prompt == huge_prompt) max_tokens = 4;
        else if ((prompt.size() + max_tokens + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE + STATE_UNITS > unit_count)
            prompt.resize(TOKENS_PER_PAGE);  // Engine::submit rejects these
        if (prompt != huge_prompt) past_prompts.push_back(prompt);
        SimRequest request{.id = request_id, .prompt = prompt, .sequence = prompt, .max_tokens = max_tokens};
        for (int i = 0; i < max_tokens; i++) request.sequence.push_back(random_engine() % 8);
        return request;
    }
};

static void cancel_random_request(Simulation& simulation, std::mt19937& random_engine) {
    auto it = simulation.requests.begin();
    std::advance(it, random_engine() % simulation.requests.size());
    if (!it->second.cancelled) simulation.cancel(it->first);
}

// afterwards: the GPU finishes, the cache evicts everything, and every unit is free
static void drain_and_check_all_free(Simulation& simulation, uint32_t unit_count) {
    mllm::Arena& arena = simulation.arena;
    simulation.decode.advance(simulation.decode.committed);
    arena.retire(simulation.decode.signaled);
    while (simulation.cache.evict()) simulation.check_bookkeeping(true);
    arena.retire(simulation.decode.signaled);
    assert(arena.free_units == unit_count && arena.reserved == 0 && arena.cache_only_units == 0 &&
           arena.pending.empty());
    for (uint32_t unit = 0; unit < unit_count; unit++) assert(arena.free_bitmap[unit / 64] >> (unit % 64) & 1);
}

static void run(unsigned seed, uint32_t unit_count, int request_count, Stats& stats) {
    std::mt19937 random_engine(seed);
    Simulation simulation(unit_count, random_engine, stats);
    PromptMaker prompts(random_engine, unit_count);
    int submitted = 0;
    for (int tick = 0; simulation.ended_requests < request_count; tick++) {
        assert(tick < request_count * 400 && "no progress: a request never ended");
        if (submitted < request_count && simulation.requests.size() < 10 && random_engine() % 3 == 0) {
            SimRequest request = prompts.random_request(simulation.next_request_id++);
            simulation.requests.emplace(request.id, std::move(request));
            submitted++;
        }
        if (!simulation.requests.empty() && random_engine() % 30 == 0) cancel_random_request(simulation, random_engine);
        simulation.step();
        simulation.check_bookkeeping(true);
        // the GPU: each lane finishes 0 or 1 more
        simulation.decode.advance(simulation.decode.signaled + random_engine() % 2);
        simulation.prefill.advance(simulation.prefill.signaled + random_engine() % 2);
    }
    drain_and_check_all_free(simulation, unit_count);
}

int main(int argc, char** argv) {
    const int seeds = argc > 1 ? atoi(argv[1]) : 200;
    Stats stats;
    // 40 units: admits evict constantly; 100: deep cache
    for (int seed = 0; seed < seeds; seed++) run(seed, 40 + seed % 3 * 30, 300, stats);
    printf("hits %ld, evictions %ld, snapshots %ld (dup %ld), released in flight %ld, 2 chunks %ld, fed ahead %ld, "
           "alloc waits %ld, too-big hits %ld, cancels %ld, eos %ld\n",
           stats.hits, stats.evictions, stats.snapshots, stats.duplicate_snapshots, stats.released_in_flight,
           stats.two_chunks, stats.fed_ahead, stats.alloc_waits, stats.too_big_hits, stats.cancels, stats.early_ends);
    for (long count : {stats.hits, stats.evictions, stats.snapshots, stats.duplicate_snapshots, stats.released_in_flight,
                       stats.two_chunks, stats.fed_ahead, stats.alloc_waits, stats.too_big_hits, stats.cancels,
                       stats.early_ends})
        assert(count > 0 && "a scenario never happened: the test lost its point");
    printf("PASS (%d seeds)\n", seeds);
}
