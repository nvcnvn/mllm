// Engine: requests and the loop (py/mllm/engine.py). submit() + step() + cancel() is the API; generate() is sugar.
//
//   mllm::Engine e("mlx-community/Qwen3.5-0.8B-4bit");
//   e.generate({{"user", "hi"}}, {}, [](const std::string& t) { printf("%s", t.c_str()); return true; });
#pragma once
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "lanes.h"
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
    int pos = 0;  // tokens whose K/V (and state) are in the arena
    std::vector<uint32_t> pages, state_slots;
    uint32_t reserved = 0;  // units promised to this request, not yet allocated
    std::vector<int> out;
    int in_flight = 0;  // lane work items holding it: a prefill chunk, or up to 2 decode steps (greedy rows fed)
    bool cancelled = false;  // finish at the next poll (architecture.md: aborts wait for their chunk)
};

class Engine {
public:
    // chunk: prefill tokens per submit (a multiple of page_tokens, so chunks start on a page). max_batch: decode rows
    // per step. max_ctx: the longest request (prompt + max_tokens); sizes the prefill attention scratch.
    explicit Engine(const std::string& model, size_t arena_bytes = (size_t)1 << 30, int chunk = 512,
                    int page_tokens = 32, int max_batch = 16, int max_ctx = 16384);

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
    PrefillLane prefill;
    DecodeLane decode;
    std::map<int, Request> reqs;  // live requests by id = arrival order

private:
    uint32_t demand(const Request& r) const;
    uint32_t alloc(Request& r);
    void ensure_pages(Request& r, int n_tokens);
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
