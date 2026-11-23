// Engine: requests and the loop (py/mllm/engine.py). submit() + step() + cancel() is the API; generate() is sugar.
//
//   mllm::Engine e("mlx-community/Qwen3.5-0.8B-4bit");
//   e.generate({{"user", "hi"}}, {}, [](const std::string& t) { printf("%s", t.c_str()); return true; });
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
};

enum class State { WAITING, PREFILL, DECODE, DONE };

struct Request {
    int id;
    std::vector<int> ids;
    int max_tokens;
    SampleParams sampler;
    std::mt19937_64 rng;  // the sampler's
    Detokenizer detok;
    State state = State::WAITING;
    int pos = 0;   // tokens whose K/V (and state) are in the arena
    int sent = 0;  // prefill: pos + the tokens of its chunks in flight
    int cached = 0;          // prompt tokens taken from the prefix cache (its first pos)
    std::vector<int> snaps;  // prefix cache: positions still ahead where the state is copied into the cache
    std::vector<uint32_t> pages, state_slots;
    uint32_t reserved = 0;  // units promised to this request, not yet allocated
    std::vector<int> out;
    int in_flight = 0;  // lane work items holding it: up to 2 prefill chunks, or up to 2 decode steps (greedy rows fed)
    bool cancelled = false;  // finish at the next poll (architecture.md: aborts wait for their chunk)
};

class Engine {
public:
    // chunk: prefill tokens per submit (a multiple of page_tokens, so chunks start on a page). max_batch: decode rows
    // per step. max_ctx: the longest request (prompt + max_tokens); sizes the prefill attention scratch.
    // prefill_on: "gpu" or "ane" (dense blocks on the Neural Engine, built by Python for S = chunk; ane_weights:
    // fp16, int8, lut6, lut4 as ane.py _compress). Decode always runs on the GPU.
    // prefix_cache: reuse prompt pages and linear state across requests (prefix_cache.h); off = no lookups, no splits.
    explicit Engine(const std::string& model, size_t arena_bytes = (size_t)1 << 30, int chunk = 512,
                    int page_tokens = 32, int max_batch = 16, int max_ctx = 16384, const std::string& prefill_on = "gpu",
                    const std::string& ane_weights = "fp16", bool prefix_cache = true);

    int submit(std::vector<int> ids, GenParams p = {});  // throws std::invalid_argument
    int submit(const std::vector<Message>& chat, GenParams p = {});
    void cancel(int req_id);  // one in a lane finishes when step() polls it
    std::vector<Event> step();
    // on_text gets each piece of text; returning false stops early (cancels the request). Throws if it aborts.
    // Other requests' events still come out of step().
    void generate(const std::vector<Message>& chat, GenParams p, const std::function<bool(const std::string&)>& on_text);

    ModelSpec spec;
    Tokenizer tok;
    id<MTLDevice> device;
    Weights weights;
    Memory mem;
    GpuOps ops;
    std::unique_ptr<AneOps> ane;  // prefill_on == "ane"
    PrefillLane prefill;
    DecodeLane decode;
    std::map<int, Request> reqs;  // live requests by id = arrival order
    std::unique_ptr<PrefixCache> cache;  // nullptr = off

private:
    uint32_t demand(const Request& r) const;
    uint32_t alloc(Request& r);
    void ensure_pages(Request& r, int n_tokens);
    bool admit(Request& r);
    void snapshot(Request& r);
    void copy_units(const std::vector<uint32_t>& dst, const std::vector<uint32_t>& src);
    void emit(Event ev);
    void abort(const std::vector<Request*>& rows);
    void finish(Request& r);

    int chunk, max_batch, max_ctx;
    int next_id = 0;
    uint64_t step_no = 0;
    std::vector<Event> events;                 // step()'s output; kept if step() throws midway
    std::map<int, std::deque<Event>> inbox;    // req id -> events, for requests a generate() is reading
    std::chrono::steady_clock::time_point residency_at;
};

}  // namespace mllm
