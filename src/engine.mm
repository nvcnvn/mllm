#include "engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace mllm {

// ponytail: a fixed interval; the OS may unwire idle memory sooner under pressure, then the first step after pays.
constexpr auto RESIDENCY_EVERY = std::chrono::seconds(10);

Engine::Engine(const std::string& model, size_t arena_bytes, int chunk, int page_tokens, int max_batch, int max_ctx)
    : spec(spec::load(model)), tok(spec.path), device(MTLCreateSystemDefaultDevice()), weights(spec.path, device),
      mem(spec, device, arena_bytes, page_tokens), ops(mem, weights),
      prefill(ops, mem, chunk, max_ctx),
      decode(ops, mem, max_batch, max_ctx), chunk(chunk), max_batch(max_batch), max_ctx(max_ctx),
      residency_at(std::chrono::steady_clock::now()) {
    if (chunk % page_tokens) throw std::invalid_argument("chunks start on a page boundary: chunk % page_tokens != 0");
    // ponytail: full_attn has no steel attention (MLX's path for a first chunk >= 2048); lift when it's ported
    if (chunk >= 2048) throw std::invalid_argument("chunk >= 2048 needs steel attention (not ported)");
    for (id<MTLBuffer> b : weights.buffers) [mem.residency addAllocation:b];  // the lanes' queues hold the set
    [mem.residency commit];
    [mem.residency requestResidency];
    mem.freeze();
}

int Engine::submit(std::vector<int> ids, GenParams p) {
    if (ids.empty() || p.max_tokens < 1) throw std::invalid_argument("empty prompt or max_tokens < 1");
    if (!(std::isfinite(p.temperature) && p.temperature >= 0 && p.top_p >= 0 && p.top_p <= 1))
        throw std::invalid_argument("need finite temperature >= 0 and 0 <= top_p <= 1");
    for (int t : ids)  // embed doesn't bounds-check
        if (t < 0 || t >= spec.vocab) throw std::invalid_argument("token id out of range: " + std::to_string(t));
    if (ids.size() + p.max_tokens > (size_t)max_ctx)
        throw std::invalid_argument("prompt + max_tokens > max_ctx " + std::to_string(max_ctx));
    Request r{.id = next_id, .ids = std::move(ids), .max_tokens = p.max_tokens, .sampler = {p.temperature, p.top_p},
              .rng = std::mt19937_64(std::random_device{}()), .detok = tok.detokenizer()};
    if (demand(r) > mem.arena.n_units)
        throw std::invalid_argument("request needs " + std::to_string(demand(r)) + " units, arena has " +
                                    std::to_string(mem.arena.n_units));
    Request& in = reqs.emplace(r.id, std::move(r)).first->second;
    in.sampler.rng = &in.rng;
    return next_id++;
}

int Engine::submit(const std::vector<Message>& chat, GenParams p) { return submit(tok.encode_chat(chat), p); }

void Engine::cancel(int req_id) {
    auto it = reqs.find(req_id);
    if (it == reqs.end() || it->second.state == State::DONE) return;
    it->second.cancelled = true;
    if (!it->second.in_flight) finish(it->second);
}

std::vector<Event> Engine::step() {
    @autoreleasepool {
        if (std::chrono::steady_clock::now() - residency_at > RESIDENCY_EVERY) {
            mem.request_residency();
            residency_at = std::chrono::steady_clock::now();
        }
        mem.arena.retire(decode.signaled());

        // a failed poll aborts the lane's in-flight rows, like a failed submit
        auto in_lane = [&](State s) {
            std::vector<Request*> rows;
            for (auto& [_, r] : reqs)
                if (r.in_flight && r.state == s) rows.push_back(&r);
            return rows;
        };
        bool progress = false;  // polled, admitted or submitted anything
        std::vector<PrefillWork> chunks;
        try {
            chunks = prefill.poll();
        } catch (...) {
            abort(in_lane(State::PREFILL));
            throw;
        }
        progress |= !chunks.empty();
        for (const PrefillWork& w : chunks) {
            Request& r = reqs.at(w.req_id);
            r.in_flight = 0;
            if (r.cancelled) {
                finish(r);
                continue;
            }
            r.pos += w.n_valid();
            if (r.pos == (int)r.ids.size() - 1) r.state = State::DECODE;
        }
        std::vector<std::pair<int, int>> toks;
        try {
            toks = decode.poll();
        } catch (...) {
            abort(in_lane(State::DECODE));
            throw;
        }
        progress |= !toks.empty();
        for (auto [req_id, t] : toks) {
            auto it = reqs.find(req_id);
            if (it == reqs.end() || it->second.state == State::DONE) continue;  // ran one step past its end (fed)
            Request& r = it->second;
            r.in_flight--;
            if (r.cancelled) {
                finish(r);
                continue;
            }
            r.pos += 1;
            r.out.push_back(t);
            const bool eos = tok.eos.count(t);
            const bool done = eos || (int)r.out.size() >= r.max_tokens;
            std::string text = eos ? "" : r.detok.push(t);
            if (done) {
                text += r.detok.flush();
                finish(r);
            }
            emit({req_id, t, std::move(text), done});
        }

        for (auto& [_, r] : reqs) {
            if (r.state != State::WAITING) continue;
            if (!mem.arena.admit(demand(r))) break;  // arrival order: later requests never overtake the oldest waiter
            r.reserved = demand(r), progress = true;
            for (uint32_t j = 0; j < mem.state_units; j++) r.state_slots.push_back(alloc(r));
            for (uint32_t u : r.state_slots)  // reused units hold an old request's state
                memset((uint8_t*)mem.arena_buffer.contents + u * mem.arena.unit_bytes, 0, mem.arena.unit_bytes);
            r.state = r.ids.size() > 1 ? State::PREFILL : State::DECODE;
        }

        if (!prefill.busy()) {
            auto it = std::find_if(reqs.begin(), reqs.end(),
                                   [](auto& kv) { return kv.second.state == State::PREFILL && !kv.second.in_flight; });
            if (it != reqs.end()) {
                Request& r = it->second;
                const int n = std::min(chunk, (int)r.ids.size() - 1 - r.pos);
                ensure_pages(r, r.pos + n);
                r.in_flight = 1, progress = true;
                try {
                    prefill.submit({r.id, std::vector<int>(r.ids.begin() + r.pos, r.ids.begin() + r.pos + n), r.pos,
                                    r.pages, r.state_slots});
                } catch (...) {
                    abort({&r});
                    throw;
                }
            }
        }

        // A greedy request still in the step in flight joins the next one too: its token is fed on the GPU. Unless
        // that step makes its last token; an EOS there leaves one wasted row (its result is dropped at poll).
        // Not while a sampled row decodes: it waits for the CPU each step, so feeding greedy rows ahead would split
        // the rows into alternating steps and halve its rate; all rows step together then, one step in flight.
        // ponytail: sampling on the GPU lets sampled rows be fed too.
        const bool sampled = std::any_of(reqs.begin(), reqs.end(), [](auto& kv) {
            return kv.second.state == State::DECODE && !kv.second.cancelled && kv.second.sampler.temp > 0;
        });
        std::vector<Request*> batch;
        for (auto& [_, r] : reqs)
            if (r.state == State::DECODE && !r.cancelled && (int)batch.size() < max_batch &&
                (!r.in_flight ||
                 (!sampled && r.in_flight == 1 && r.sampler.temp == 0 && (int)r.out.size() + 1 < r.max_tokens)))
                batch.push_back(&r);
        if (!batch.empty() && decode.can_submit() && !(sampled && decode.busy())) {
            progress = true;
            DecodeWork w{step_no, {}};
            for (Request* r : batch) {
                const int pos = r->pos + r->in_flight;
                ensure_pages(*r, pos + 1);
                w.rows.push_back({r->id, r->in_flight ? -1 : r->out.empty() ? r->ids.back() : r->out.back(), pos,
                                  r->pages, r->state_slots, r->sampler});
                r->in_flight++;
            }
            try {
                decode.submit(std::move(w));
            } catch (...) {
                abort(batch);
                throw;
            }
            step_no++;
        }

        std::erase_if(reqs, [](auto& kv) { return kv.second.state == State::DONE; });
        // Nothing to do until a lane finishes: block instead of handing a busy loop to the caller (a 1000-token
        // generate spun 3.7 s of CPU, now 0.5 s). On the decode step when one runs: it is the short one, and a
        // completion-handler semaphore that woke on either lane cost B = 1 2%. ponytail: prefill may then sit done
        // for up to one decode step; a shared event for both lanes if that shows.
        if (!progress && events.empty()) {
            if (decode.busy()) decode.wait_oldest();
            else prefill.wait_oldest();
        }
        return std::exchange(events, {});
    }
}

void Engine::generate(const std::vector<Message>& chat, GenParams p,
                      const std::function<bool(const std::string&)>& on_text) {
    const int rid = submit(chat, p);
    std::deque<Event>& q = inbox[rid];
    auto cleanup = [&] {
        inbox.erase(rid);
        cancel(rid);
    };
    try {
        for (;;) {
            while (q.empty()) {
                std::vector<Event> ev = step();  // not ours: hand back to the next step() caller
                events.insert(events.end(), std::make_move_iterator(ev.begin()), std::make_move_iterator(ev.end()));
            }
            Event e = std::move(q.front());
            q.pop_front();
            if (e.token < 0) throw std::runtime_error("request " + std::to_string(rid) + " aborted");
            if (!e.text.empty() && !on_text(e.text)) break;
            if (e.done) break;
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

// Units for the whole request up front. ponytail: no generation top-up yet (memory-design `admit`).
uint32_t Engine::demand(const Request& r) const {
    return mem.units_for(r.ids.size() + r.max_tokens) + mem.state_units;
}

uint32_t Engine::alloc(Request& r) {
    Arena& a = mem.arena;
    a.retire(decode.signaled());
    if (std::none_of(a.free.begin(), a.free.end(), [](uint64_t w) { return w; })) {
        decode.wait();  // admit counted units still pending on the step in flight
        a.retire(decode.signaled());
    }
    r.reserved--;
    return a.alloc();
}

void Engine::ensure_pages(Request& r, int n_tokens) {
    while ((int)r.pages.size() * mem.T < n_tokens) r.pages.push_back(alloc(r));
}

void Engine::emit(Event ev) {
    auto it = inbox.find(ev.req_id);
    if (it != inbox.end()) it->second.push_back(std::move(ev));
    else events.push_back(std::move(ev));
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
    Arena& a = mem.arena;
    a.unreserve(r.reserved);
    std::vector<uint32_t> all = r.pages;
    all.insert(all.end(), r.state_slots.begin(), r.state_slots.end());
    a.release(all);
    r.pages.clear(), r.state_slots.clear(), r.reserved = 0, r.state = State::DONE;
}

}  // namespace mllm
