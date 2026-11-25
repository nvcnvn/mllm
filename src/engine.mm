#include "engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace mllm {

// ponytail: a fixed interval; the OS may unwire idle memory sooner under pressure, then the first step after pays.
constexpr auto RESIDENCY_INTERVAL = std::chrono::seconds(10);

static std::unique_ptr<AneOps> make_ane_ops(const std::string& prefill_on, GpuOps& gpu, Memory& mem, int chunk) {
    if (prefill_on != "gpu" && prefill_on != "ane")
        throw std::invalid_argument("prefill_on: \"gpu\" or \"ane\", not " + prefill_on);
    return prefill_on == "ane" ? std::make_unique<AneOps>(gpu, mem, chunk) : nullptr;
}

Engine::Engine(const std::string& model, size_t arena_bytes, int chunk, int page_tokens, int max_batch,
               int max_context, const std::string& prefill_on, bool prefix_cache)
    : spec(spec::load(model)), tokenizer(spec.path), device(MTLCreateSystemDefaultDevice()), weights(spec.path, device),
      mem(spec, device, arena_bytes, page_tokens), gpu(mem, weights), ane(make_ane_ops(prefill_on, gpu, mem, chunk)),
      prefill(gpu, mem, chunk, max_context, ane.get()), decode(gpu, mem, max_batch, max_context),
      cache(prefix_cache ? std::make_unique<PrefixCache>(mem.arena, page_tokens) : nullptr), chunk(chunk),
      max_batch(max_batch), max_context(max_context), residency_requested_at(std::chrono::steady_clock::now()) {
    if (chunk % page_tokens) throw std::invalid_argument("chunks start on a page boundary: chunk % page_tokens != 0");
    // ponytail: full attention has no steel attention (MLX's path for a first chunk >= 2048); lift when it's ported
    if (chunk >= 2048) throw std::invalid_argument("chunk >= 2048 needs steel attention (not ported)");
    for (id<MTLBuffer> buffer : weights.buffers) [mem.residency addAllocation:buffer];  // the lanes' queues hold the set
    [mem.residency commit];
    [mem.residency requestResidency];
    mem.freeze();
    size_t weight_bytes = 0;
    for (id<MTLBuffer> buffer : weights.buffers) weight_bytes += buffer.length;
    fprintf(stderr, "mllm: weights %.2f GB, arena %.2f GiB (%u units of %zu KiB, %u per state slot), workspace %.1f MiB "
            "(chunk %d, max batch %d, max_ctx %d)\n", weight_bytes / 1e9, mem.arena_buffer.length / double(1 << 30),
            mem.arena.n_units, mem.arena.unit_bytes >> 10, mem.state_units,
            (mem.allocated - mem.arena_buffer.length) / double(1 << 20), chunk, max_batch, max_context);
}

int Engine::submit(std::vector<int> prompt, GenParams params) {
    if (prompt.empty() || params.max_tokens < 1) throw std::invalid_argument("empty prompt or max_tokens < 1");
    if (!(std::isfinite(params.temperature) && params.temperature >= 0 && params.top_p >= 0 && params.top_p <= 1))
        throw std::invalid_argument("need finite temperature >= 0 and 0 <= top_p <= 1");
    for (int token : prompt)  // look_up_embeddings doesn't bounds-check
        if (token < 0 || token >= spec.vocab_size)
            throw std::invalid_argument("token id out of range: " + std::to_string(token));
    if (prompt.size() + params.max_tokens > (size_t)max_context)
        throw std::invalid_argument("prompt + max_tokens > max_ctx " + std::to_string(max_context));
    Request r{.id = next_request_id, .prompt = std::move(prompt), .max_tokens = params.max_tokens,
              .sampler = {params.temperature, params.top_p}, .random_engine = std::mt19937_64(std::random_device{}()),
              .detokenizer = tokenizer.detokenizer()};
    if (units_needed(r) > mem.arena.n_units)
        throw std::invalid_argument("request needs " + std::to_string(units_needed(r)) + " units, arena has " +
                                    std::to_string(mem.arena.n_units));
    Request& stored = requests.emplace(r.id, std::move(r)).first->second;
    stored.sampler.rng = &stored.random_engine;
    return next_request_id++;
}

int Engine::submit(const std::vector<Message>& chat, GenParams params) {
    return submit(tokenizer.encode_chat(chat), params);
}

void Engine::cancel(int req_id) {
    auto it = requests.find(req_id);
    if (it == requests.end() || it->second.state == State::DONE) return;
    it->second.cancelled = true;
    if (!it->second.in_flight) finish(it->second);
}

std::vector<Event> Engine::step() {
    @autoreleasepool {
        keep_memory_resident();
        mem.arena.retire(decode.signaled());
        bool progress = take_finished_chunks();  // polled, admitted or submitted anything
        progress |= take_decoded_tokens();
        progress |= admit_waiting_requests();
        progress |= submit_prefill_chunks();
        progress |= submit_decode_step();
        std::erase_if(requests, [](auto& entry) { return entry.second.state == State::DONE; });
        // Nothing to do until a lane finishes: block instead of handing a busy loop to the caller (a 1000-token
        // generate spun 3.7 s of CPU, now 0.5 s).
        if (!progress && events.empty()) wait_for_a_lane();
        return std::exchange(events, {});
    }
}

void Engine::keep_memory_resident() {
    if (std::chrono::steady_clock::now() - residency_requested_at <= RESIDENCY_INTERVAL) return;
    mem.request_residency();
    residency_requested_at = std::chrono::steady_clock::now();
}

// A failed poll aborts the lane's in-flight rows, like a failed submit.
std::vector<Request*> Engine::requests_in_lane(State state) {
    std::vector<Request*> rows;
    for (auto& [_, r] : requests)
        if (r.in_flight && r.state == state) rows.push_back(&r);
    return rows;
}

// Prefill chunks the lane finished: their tokens are now in the arena.
bool Engine::take_finished_chunks() {
    std::vector<PrefillWork> chunks;
    try {
        chunks = prefill.poll();
    } catch (...) {
        abort(requests_in_lane(State::PREFILL));
        throw;
    }
    for (const PrefillWork& work : chunks) {
        Request& r = requests.at(work.req_id);
        r.in_flight--;
        if (r.cancelled) {
            if (!r.in_flight) finish(r);
            continue;
        }
        r.position += work.n_valid();
        // ponytail: only completed snapshot points are cached; a cancelled or aborted request's later pages aren't
        if (!r.snapshot_points.empty() && r.position == r.snapshot_points.front()) {
            save_snapshot(r);
            r.snapshot_points.erase(r.snapshot_points.begin());
        }
        if (r.position == (int)r.prompt.size() - 1) r.state = State::DECODE;
    }
    return !chunks.empty();
}

// Tokens the decode lane chose: each becomes an event (with its text), and ends its request at EOS or max_tokens.
bool Engine::take_decoded_tokens() {
    std::vector<std::pair<int, int>> tokens;
    try {
        tokens = decode.poll();
    } catch (...) {
        abort(requests_in_lane(State::DECODE));
        throw;
    }
    for (auto [req_id, token] : tokens) {
        auto it = requests.find(req_id);
        if (it == requests.end() || it->second.state == State::DONE) continue;  // ran one step past its end (fed)
        Request& r = it->second;
        r.in_flight--;
        if (r.cancelled) {
            finish(r);
            continue;
        }
        r.position += 1;
        r.generated.push_back(token);
        const bool end_of_text = tokenizer.eos.count(token);
        const bool done = end_of_text || (int)r.generated.size() >= r.max_tokens;
        std::string text = end_of_text ? "" : r.detokenizer.push(token);
        if (done) {
            text += r.detokenizer.flush();
            finish(r);
        }
        emit({req_id, token, std::move(text), done, r.cached});
    }
    return !tokens.empty();
}

bool Engine::admit_waiting_requests() {
    bool admitted = false;
    for (auto& [_, r] : requests) {
        if (r.state != State::WAITING) continue;
        if (!admit(r)) break;  // arrival order: later requests never overtake the oldest waiter
        admitted = true;
    }
    return admitted;
}

// The oldest request with prompt tokens left to send. One with a chunk in flight may send a second chunk too, unless
// that first chunk ends at a snapshot point (the copy needs that chunk done and the next one not started).
Request* Engine::next_to_prefill() {
    for (auto& [_, r] : requests)
        if (r.state == State::PREFILL && !r.cancelled && r.sent < (int)r.prompt.size() - 1 &&
            !(r.in_flight && !r.snapshot_points.empty() && r.sent == r.snapshot_points.front()))
            return &r;
    return nullptr;
}

bool Engine::submit_prefill_chunks() {
    bool submitted = false;
    while (prefill.can_submit()) {
        Request* next = next_to_prefill();
        if (!next) break;
        Request& r = *next;
        int end = std::min(r.sent + chunk, (int)r.prompt.size() - 1);
        if (!r.snapshot_points.empty()) end = std::min(end, r.snapshot_points.front());
        ensure_pages(r, end);
        r.in_flight++, submitted = true;
        try {
            prefill.submit({r.id, std::vector<int>(r.prompt.begin() + r.sent, r.prompt.begin() + end), r.sent, r.pages,
                            r.state_slots});
        } catch (...) {
            abort_failed_chunk(r);
            throw;
        }
        r.sent = end;
    }
    return submitted;
}

// prefill.submit threw for r's chunk (already counted in in_flight).
void Engine::abort_failed_chunk(Request& r) {
    if (--r.in_flight == 0) {
        abort({&r});
        return;
    }
    // its other chunk is still in the lane: end it like a cancel, poll finishes it
    r.cancelled = true;
    emit({r.id, -1, "", true});
}

// A request still in the step in flight joins the next one too: its token (greedy or sampled, both chosen on the GPU)
// is fed there. Unless that step makes its last token; an EOS there leaves one wasted row (its result is dropped at
// poll).
static bool ready_to_decode(const Request& r) {
    return r.state == State::DECODE && !r.cancelled &&
           (!r.in_flight || (r.in_flight == 1 && (int)r.generated.size() + 1 < r.max_tokens));
}

bool Engine::submit_decode_step() {
    std::vector<Request*> batch;
    for (auto& [_, r] : requests)
        if ((int)batch.size() < max_batch && ready_to_decode(r)) batch.push_back(&r);
    if (batch.empty() || !decode.can_submit()) return false;
    DecodeWork work{decode_steps, {}};
    for (Request* r : batch) {
        const int position = r->position + r->in_flight;
        ensure_pages(*r, position + 1);
        // -1: the GPU feeds the token the step in flight chooses
        const int fed_token = r->in_flight ? -1 : r->generated.empty() ? r->prompt.back() : r->generated.back();
        work.rows.push_back({r->id, fed_token, position, r->pages, r->state_slots, r->sampler});
        r->in_flight++;
    }
    try {
        decode.submit(std::move(work));
    } catch (...) {
        abort(batch);
        throw;
    }
    decode_steps++;
    return true;
}

// Waits on the decode step when one runs: it is the short one, and a completion-handler semaphore that woke on either
// lane cost 2% at one request. ponytail: prefill may then sit done for up to one decode step; a shared event for both
// lanes if that shows.
void Engine::wait_for_a_lane() {
    if (decode.busy()) decode.wait_oldest();
    else prefill.wait_oldest();
}

void Engine::generate(const std::vector<Message>& chat, GenParams params,
                      const std::function<bool(const std::string&)>& on_text) {
    const int request_id = submit(chat, params);
    std::deque<Event>& mine = inbox[request_id];
    auto cleanup = [&] {
        inbox.erase(request_id);
        cancel(request_id);
    };
    try {
        for (;;) {
            while (mine.empty()) {
                std::vector<Event> others = step();  // not ours: hand back to the next step() caller
                events.insert(events.end(), std::make_move_iterator(others.begin()),
                              std::make_move_iterator(others.end()));
            }
            Event event = std::move(mine.front());
            mine.pop_front();
            if (event.token < 0) throw std::runtime_error("request " + std::to_string(request_id) + " aborted");
            if (!event.text.empty() && !on_text(event.text)) break;
            if (event.done) break;
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

// Units for the whole request up front. ponytail: no generation top-up yet (memory-design `admit`).
uint32_t Engine::units_needed(const Request& r) const {
    return mem.units_for(r.prompt.size() + r.max_tokens) + mem.state_units;
}

uint32_t Engine::allocate_unit(Request& r) {
    Arena& arena = mem.arena;
    arena.retire(decode.signaled());
    if (std::none_of(arena.free_bitmap.begin(), arena.free_bitmap.end(), [](uint64_t word) { return word; })) {
        decode.wait();  // admit counted units still pending on the step in flight
        arena.retire(decode.signaled());
    }
    // units_needed() must cover every allocation, or this request quietly takes another's reservation
    // (vLLM 0ff1bf9bb)
    assert(r.reserved > 0 && "request allocates past its reservation");
    r.reserved--;
    return arena.alloc();
}

// With the prefix cache: the deepest cached snapshot of r's prompt, and the positions ahead where r's state is to be
// copied into the cache (snapshot_points). Without it: no match.
PrefixCache::Match Engine::match_prefix(const Request& r, std::vector<int>& snapshot_points) {
    if (!cache) return {};
    const int page_tokens = mem.page_tokens, last = (int)r.prompt.size() - 1;  // the last prompt token is fed by decode
    PrefixCache::Match match = cache->match(r.prompt);
    // the branch point (the next request with this prefix resumes there: shared system prompts) and the prompt's last
    // full page (multi-turn: the next turn extends this prompt)
    if (match.matched_pages > match.resume_pages) snapshot_points.push_back(match.matched_pages * page_tokens);
    if (const int last_page_start = last / page_tokens * page_tokens;
        last_page_start > match.resume_pages * page_tokens &&
        (snapshot_points.empty() || last_page_start > snapshot_points[0]))
        snapshot_points.push_back(last_page_start);
    // While admitting, a hit holds its resume_pages pages and its snapshot: units neither free nor cache-only.
    // ponytail: a request that can't fit with all that drops its copies (the branch point's first), then its hit (cold
    // always fits: submit checks units_needed), instead of waiting forever
    const auto need = [&] {
        return units_needed(r) - match.resume_pages + snapshot_points.size() * mem.state_units + match.resume_pages +
               match.snapshot.size();
    };
    while (need() > mem.arena.n_units && !snapshot_points.empty()) snapshot_points.erase(snapshot_points.begin());
    if (need() > mem.arena.n_units) match = {};
    return match;
}

// Reserve the request's units and give it its state; with the prefix cache, start at the deepest cached snapshot.
bool Engine::admit(Request& r) {
    Arena& arena = mem.arena;
    std::vector<int> snapshot_points;
    const PrefixCache::Match match = match_prefix(r, snapshot_points);
    // hold what the match returned, so admit's evictions can't free it
    for (uint32_t unit : match.pages) arena.ref(unit);
    for (uint32_t unit : match.snapshot) arena.ref(unit);
    const uint32_t count = units_needed(r) - match.resume_pages + (uint32_t)snapshot_points.size() * mem.state_units;
    if (!arena.admit(count, [&] { return cache && cache->evict(); })) {
        arena.release(match.pages), arena.release(match.snapshot);
        return false;
    }
    r.reserved = count, r.pages = match.pages;
    r.position = r.sent = r.cached = match.resume_pages * mem.page_tokens;
    r.snapshot_points = std::move(snapshot_points);
    start_state(r, match);
    r.state = r.position < (int)r.prompt.size() - 1 ? State::PREFILL : State::DECODE;
    return true;
}

// r's state units: a copy of the matched snapshot (requests update their state in place), or zeros.
void Engine::start_state(Request& r, const PrefixCache::Match& match) {
    for (uint32_t j = 0; j < mem.state_units; j++) r.state_slots.push_back(allocate_unit(r));
    if (match.resume_pages) {
        copy_units(r.state_slots, match.snapshot);
        mem.arena.release(match.snapshot);
        return;
    }
    for (uint32_t unit : r.state_slots)  // reused units hold an old request's state
        memset((uint8_t*)mem.arena_buffer.contents + unit * mem.arena.unit_bytes, 0, mem.arena.unit_bytes);
}

// r.position is a snapshot point and its chunk has completed (the next one isn't submitted): copy its state into
// fresh units and cache its pages up to here. Its pages from position on stay its own.
void Engine::save_snapshot(Request& r) {
    const int page_count = r.position / mem.page_tokens;
    if (cache->has_snapshot(r.prompt, page_count)) {  // another request got here first: return the copy's reservation
        mem.arena.unreserve(mem.state_units);
        r.reserved -= mem.state_units;
        return;
    }
    std::vector<uint32_t> snapshot;
    for (uint32_t j = 0; j < mem.state_units; j++) snapshot.push_back(allocate_unit(r));
    copy_units(snapshot, r.state_slots);
    cache->insert(r.prompt, page_count, r.pages, snapshot);
    mem.arena.release(snapshot);  // the cache's refs keep them
}

// CPU copy on the shared arena (~19 MB for Qwen3.5-0.8B's state): no GPU work reads or writes either side now
void Engine::copy_units(const std::vector<uint32_t>& destination, const std::vector<uint32_t>& source) {
    uint8_t* base = (uint8_t*)mem.arena_buffer.contents;
    const size_t unit_bytes = mem.arena.unit_bytes;
    for (size_t j = 0; j < destination.size(); j++)
        memcpy(base + destination[j] * unit_bytes, base + source[j] * unit_bytes, unit_bytes);
}

void Engine::ensure_pages(Request& r, int tokens) {
    while ((int)r.pages.size() * mem.page_tokens < tokens) r.pages.push_back(allocate_unit(r));
}

void Engine::emit(Event event) {
    auto it = inbox.find(event.req_id);
    if (it != inbox.end()) it->second.push_back(std::move(event));
    else events.push_back(std::move(event));
}

// A lane call failed: rows may be half-written (in-place state), so end them, never retry.
void Engine::abort(const std::vector<Request*>& rows) {
    for (Request* r : rows) {
        r->in_flight = 0;
        finish(*r);
        emit({r->id, -1, "", true});
    }
}

void Engine::finish(Request& r) {
    // pending reclaim follows decode steps only: a unit a prefill chunk still reads must not be released (cancel waits
    // for the chunk; abort zeroes in_flight once the lane let go of it)
    assert((r.in_flight == 0 || r.state == State::DECODE) && "released while a prefill chunk reads its units");
    Arena& arena = mem.arena;
    arena.unreserve(r.reserved);
    std::vector<uint32_t> units = r.pages;
    units.insert(units.end(), r.state_slots.begin(), r.state_slots.end());
    arena.release(units);
    r.pages.clear(), r.state_slots.clear(), r.reserved = 0, r.state = State::DONE;
}

}  // namespace mllm
