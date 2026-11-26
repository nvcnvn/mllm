// Engine: requests and the loop (py/mllm/engine.py). submit() + step() + cancel() is the API; generate() is sugar.
//
//   mllm::Engine engine("mlx-community/Qwen3.5-0.8B-4bit");
//   engine.generate({{"user", "hi"}}, {}, [](const std::string& text) { printf("%s", text.c_str()); return true; });
#pragma once
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "lanes.h"
#include "prefix_cache.h"
#include "tokenizer.h"

namespace mllm {

struct GenParams {
    int max_tokens = 256;
    float temperature = 0, top_p = 0;  // mlx-lm make_sampler: temperature 0 = greedy
};

// step()'s output. token -1 = aborted (a lane call failed); done is then true.
struct Event {
    int req_id, token;
    std::string text;
    bool done;
    int cached = 0;  // prompt tokens the prefix cache supplied (Request::cached)
    int rounds = 0, round_tokens = 0;  // its speculative rounds so far and the tokens they kept
};

enum class State { WAITING, PREFILL, DECODE, DONE };

struct Request {
    int id;
    std::vector<int> prompt;  // token ids
    int max_tokens;
    SampleParams sampler;
    std::mt19937_64 random_engine;  // the sampler's
    Detokenizer detokenizer;
    State state = State::WAITING;
    int position = 0;  // tokens whose keys / values (and state) are in the arena
    int sent = 0;      // prefill: position + the tokens of its chunks in flight
    int cached = 0;    // prompt tokens taken from the prefix cache (its first position)
    std::vector<int> snapshot_points;  // prefix cache: positions still ahead where the state is copied into the cache
    std::vector<uint32_t> pages, state_slots;
    // Speculative decoding (greedy, a model with an MTP head: docs/speculative.md): DecodeLane::DRAFTS more state sets
    // for the verify's checkpoints, held only while it decodes alone (take_spares); the step it last decoded in and
    // its row there (the head's output to draft from); its step in flight is a speculative round
    bool speculative = false;
    std::vector<uint32_t> spare_state;
    int rounds = 0, round_tokens = 0;
    uint64_t last_step = UINT64_MAX;
    int last_row = 0;
    bool speculating = false;
    uint32_t reserved = 0;  // units promised to this request, not yet allocated
    std::vector<int> generated;  // the tokens decoded so far
    int in_flight = 0;  // lane work items holding it: up to 2 prefill chunks, or up to 2 decode steps (greedy rows fed)
    bool cancelled = false;  // finish at the next poll (architecture.md: aborts wait for their chunk)
    bool ended = false;      // its last token is out (EOS or max_tokens): its sequence goes into the prefix cache
};

class Engine {
public:
    // chunk: prefill tokens per submit (a multiple of page_tokens, so chunks start on a page). max_batch: decode rows
    // per step. max_context: the longest request (prompt + max_tokens); sizes the prefill attention scratch.
    // prefill_on: "gpu" or "ane" (dense blocks on the Neural Engine, the GPU's weights fed in per layer: ane.h; the
    // blocks are built by Python for 512 and chunk). Decode always runs on the GPU.
    // prefix_cache: reuse prompt pages and linear state across requests (prefix_cache.h); off = no lookups, no splits.
    explicit Engine(const std::string& model, size_t arena_bytes = (size_t)1 << 30, int chunk = 512,
                    int page_tokens = 32, int max_batch = 16, int max_context = 16384,
                    const std::string& prefill_on = "gpu", bool prefix_cache = true);

    int submit(std::vector<int> prompt, GenParams params = {});  // throws std::invalid_argument
    int submit(const std::vector<Message>& chat, GenParams params = {});
    void cancel(int req_id);  // one in a lane finishes when step() polls it
    int context_limit() const { return max_context; }
    std::vector<Event> step();
    // on_text gets each piece of text; returning false stops early (cancels the request). Throws if it aborts.
    // Other requests' events still come out of step().
    void generate(const std::vector<Message>& chat, GenParams params,
                  const std::function<bool(const std::string&)>& on_text);

    ModelSpec spec;
    Tokenizer tokenizer;
    id<MTLDevice> device;
    Weights weights;
    Memory mem;
    GpuOps gpu;
    std::unique_ptr<AneOps> ane;  // prefill_on == "ane"
    PrefillLane prefill;
    DecodeLane decode;
    std::map<int, Request> requests;  // live requests by id = arrival order
    std::unique_ptr<PrefixCache> cache;  // nullptr = off
    // Greedy requests decode speculatively when the model has an MTP head, alone in the decode lane (others join as
    // plain rows). Read at submit. Off: plain decode, bit-exact with mlx-lm (golden tests); gpu.on_scores turns it off
    // too (one row of scores per token).
    bool speculate = true;
    uint64_t rounds = 0, round_tokens = 0;  // speculative rounds so far and the tokens they kept

private:
    // step()'s parts, in order; each but the first returns whether it polled, admitted or submitted anything
    void keep_memory_resident();
    bool take_finished_chunks();
    bool take_decoded_tokens();
    bool admit_waiting_requests();
    bool submit_prefill_chunks();
    bool submit_decode_step();
    void wait_for_a_lane();

    std::vector<Request*> requests_in_lane(State state);
    Request* next_to_prefill();
    void abort_failed_chunk(Request& r);
    uint32_t units_needed(const Request& r) const;
    uint32_t allocate_unit(Request& r);
    bool take_spares(Request& r);
    void release_spares(Request& r);
    void ensure_pages(Request& r, int tokens);
    bool admit(Request& r);
    PrefixCache::Match match_prefix(const Request& r, std::vector<int>& snapshot_points);
    void start_state(Request& r, const PrefixCache::Match& match);
    void save_snapshot(Request& r);
    void cache_sequence(Request& r);
    void copy_units(const std::vector<uint32_t>& destination, const std::vector<uint32_t>& source);
    void emit(Event event);
    void take_tokens(Request& r, const std::vector<int>& chosen);
    void abort(const std::vector<Request*>& rows);
    void finish(Request& r);

    int chunk, max_batch, max_context;
    int next_request_id = 0;
    uint64_t decode_steps = 0;                // decode steps submitted: the next one's number
    std::vector<Event> events;                // step()'s output; kept if step() throws midway
    std::map<int, std::deque<Event>> inbox;  // req id -> events, for requests a generate() is reading
    std::chrono::steady_clock::time_point residency_requested_at;
};

}  // namespace mllm
