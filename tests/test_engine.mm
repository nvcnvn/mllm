// Engine (cpp-port.md §5, py/test_engine.py): 3 prompts x 64 greedy tokens equal golden, solo, batched, and waiting on
// a one-request arena; FIFO admission (a big waiting request isn't overtaken); a generate() stopped early and a lane
// failure both return every unit.
#include <cassert>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "engine.h"
#include "golden.h"

constexpr int MAX_TOKENS = 64, CHUNK_TOKENS = 32;  // small chunks, so a prompt spans several chunks and pages

using Chat = std::vector<mllm::Message>;
using TokenLists = std::vector<std::vector<int>>;

static void check_no_units_leaked(const mllm::Engine& engine) {
    const mllm::Arena& arena = engine.mem.arena;
    assert(arena.free_units + arena.cache_only_units == arena.n_units && arena.reserved == 0 && "arena units leaked");
}

// submits every prompt, steps until all are done; each prompt's tokens, in prompt order
static TokenLists run_to_completion(mllm::Engine& engine, const std::vector<Chat>& prompts) {
    std::vector<int> request_ids;
    for (const auto& prompt : prompts) request_ids.push_back(engine.submit(prompt, {.max_tokens = MAX_TOKENS}));
    std::map<int, std::vector<int>> tokens_by_request;
    while (!engine.requests.empty())
        for (const mllm::Event& event : engine.step()) tokens_by_request[event.req_id].push_back(event.token);
    check_no_units_leaked(engine);
    TokenLists tokens;
    for (int request_id : request_ids) tokens.push_back(tokens_by_request[request_id]);
    return tokens;
}

static void load_golden(const nlohmann::json& golden_spec, const mllm::Safetensors& golden_tokens,
                        std::vector<Chat>& prompts, TokenLists& golden_ids) {
    for (int i = 0; i < 3; i++) {
        prompts.emplace_back();
        for (const auto& message : golden_spec["chat"][i])
            prompts.back().push_back({message["role"], message["content"]});
        const mllm::Tensor& tensor = golden_tokens.at("greedy/" + std::to_string(i));
        golden_ids.emplace_back(tensor.nbytes / 4);
        for (size_t j = 0; j < golden_ids.back().size(); j++)
            golden_ids.back()[j] = golden::load<int32_t>(golden_tokens.data(tensor), j);
    }
}

// 1. one request at a time == mlx-lm greedy, token for token
static TokenLists check_solo(mllm::Engine& engine, const std::vector<Chat>& prompts, const TokenLists& golden_ids) {
    TokenLists solo;
    for (int i = 0; i < 3; i++) {
        solo.push_back(run_to_completion(engine, {prompts[i]})[0]);
        assert(solo.back() == golden_ids[i]);
    }
    printf("solo: 3 x %d greedy tokens equal golden\n", MAX_TOKENS);
    return solo;
}

// 2. all at once (batched decode) == each alone; every unit nothing holds is NaN first (bf16 0xFFFF): a kernel that
// reads a reused unit past what this request wrote (masked keys / values, a cold state) would show it
// (llama.cpp 0a7675746)
static void check_batched(mllm::Engine& engine, const std::vector<Chat>& prompts, const TokenLists& solo) {
    const mllm::Arena& arena = engine.mem.arena;
    for (uint32_t unit = 0; unit < arena.n_units; unit++)
        if (!arena.request_refs[unit] && !arena.cache_refs[unit])
            memset((uint8_t*)engine.mem.arena_buffer.contents + (size_t)unit * arena.unit_bytes, 0xFF,
                   arena.unit_bytes);
    assert(run_to_completion(engine, prompts) == solo);
    printf("batched: equal solo\n");
}

// 3. arena fits one request at a time: the rest WAIT, then run, same tokens
static void check_one_request_arena(mllm::Engine& engine, const std::vector<Chat>& prompts, const TokenLists& solo) {
    size_t longest = 0;
    for (const auto& prompt : prompts) longest = std::max(longest, engine.tokenizer.encode_chat(prompt).size());
    const uint32_t units_needed = engine.mem.units_for(longest + MAX_TOKENS) + engine.mem.state_units;
    {
        mllm::Engine small(golden::MODEL, units_needed * engine.mem.arena.unit_bytes, CHUNK_TOKENS);
        assert(run_to_completion(small, prompts) == solo);

        check_no_units_leaked(small);
    }
    printf("wait ok with a %u-unit arena\n", units_needed);
}

// FIFO: B runs; A waits (doesn't fit next to B); C would fit next to B but must not overtake A
static void check_fifo_admission(mllm::Engine& engine, const std::vector<Chat>& prompts) {
    const std::vector<int> prompt_a = engine.tokenizer.encode_chat(prompts[0]),
                           prompt_b = engine.tokenizer.encode_chat(prompts[2]), prompt_c = {1, 2, 3};
    auto units_for_request = [&](size_t prompt_tokens, int max_tokens) {
        return engine.mem.units_for(prompt_tokens + max_tokens) + engine.mem.state_units;
    };
    const uint32_t units_b = units_for_request(prompt_b.size(), 8),
                   units_a = units_for_request(prompt_a.size(), MAX_TOKENS),
                   units_c = units_for_request(prompt_c.size(), 4);
    assert(units_b + units_a > units_b + units_c && units_a <= units_b + units_c);
    mllm::Engine fifo(golden::MODEL, (units_b + units_c) * engine.mem.arena.unit_bytes, CHUNK_TOKENS);
    const int request_b = fifo.submit(prompt_b, {.max_tokens = 8}),
              request_a = fifo.submit(prompt_a, {.max_tokens = MAX_TOKENS});
    const int request_c = fifo.submit(prompt_c, {.max_tokens = 4});
    std::vector<int> admitted;  // in admission order
    auto was_admitted = [&](int request_id) {
        return !fifo.requests.count(request_id) || fifo.requests.at(request_id).state != mllm::State::WAITING;
    };
    while (!fifo.requests.empty()) {
        fifo.step();
        for (int request_id : {request_a, request_b, request_c})
            if (std::find(admitted.begin(), admitted.end(), request_id) == admitted.end() && was_admitted(request_id))
                admitted.push_back(request_id);
    }
    assert(admitted == std::vector<int>({request_b, request_a, request_c}));
    check_no_units_leaked(fifo);
    printf("FIFO: the big waiter is not overtaken\n");
}

// a sampled request next to a greedy one gets a token every step (greedy rows aren't fed ahead of it)
static void check_mixed_greedy_and_sampled(mllm::Engine& engine) {
    const int greedy_id = engine.submit(std::vector<int>{1}, {.max_tokens = 24});
    const int sampled_id =
        engine.submit(std::vector<int>{2}, {.max_tokens = 12, .temperature = 0.8f, .top_p = 0.9f});
    int greedy_tokens = 0, sampled_tokens = 0, greedy_tokens_at_last_sample = 0;
    while (!engine.requests.empty())
        for (const mllm::Event& event : engine.step()) {
            if (event.req_id == greedy_id) greedy_tokens++;
            if (event.req_id != sampled_id) continue;
            // greedy tokens between two sampled ones
            assert(sampled_tokens++ == 0 || greedy_tokens - greedy_tokens_at_last_sample <= 1);
            greedy_tokens_at_last_sample = greedy_tokens;
        }
    assert(sampled_tokens >= 2);
    check_no_units_leaked(engine);
    printf("mixed greedy + sampled: one token per step each\n");
}

static void check_invalid_params_rejected(mllm::Engine& engine) {
    for (mllm::GenParams params :
         {mllm::GenParams{.temperature = -1}, {.temperature = NAN}, {.top_p = 1.5f}, {.top_p = NAN}}) {
        bool rejected = false;
        try {
            engine.submit(std::vector<int>{1}, params);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected && engine.requests.empty());
    }
}

// 4. a generate() stopped early (cancel mid-decode), and a lane call that fails, both return every unit
static void check_stop_and_lane_failure(mllm::Engine& engine, const std::vector<Chat>& prompts) {
    int pieces = 0;
    engine.generate(prompts[0], {.max_tokens = MAX_TOKENS}, [&](const std::string&) { return ++pieces < 3; });
    assert(pieces == 3);
    engine.decode.fail_next = true;
    bool threw = false;
    try {
        engine.generate(prompts[1], {.max_tokens = MAX_TOKENS}, [](const std::string&) { return true; });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw && "lane error swallowed");
    run_to_completion(engine, {});  // steps until the cancelled request is polled, then checks the arena
}

// the same through step(): the failed rows end with token -1, done, in that very step
static void check_abort_through_step(mllm::Engine& engine, const std::vector<Chat>& prompts) {
    const int failing_id = engine.submit(prompts[2], {.max_tokens = MAX_TOKENS});
    engine.decode.fail_next = true;
    for (bool aborted = false; !aborted;) {
        try {
            engine.step();
        } catch (const std::runtime_error&) {
            std::vector<mllm::Event> events = engine.step();  // the abort event was kept for the next step() caller
            aborted = events.size() == 1 && events[0].req_id == failing_id && events[0].token == -1 && events[0].done;
            assert(aborted && engine.requests.empty());
        }
    }
    check_no_units_leaked(engine);
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string golden_dir = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(golden_dir + "/spec.json"));
    mllm::Safetensors golden_tokens(golden_dir + "/tokens.safetensors");
    std::vector<Chat> prompts;
    TokenLists golden_ids;
    load_golden(golden_spec, golden_tokens, prompts, golden_ids);

    // 0.8B 256 MiB, 27B 1 GiB
    mllm::Engine engine(golden::MODEL, golden_spec["layout"]["arena_bytes"].get<size_t>(), CHUNK_TOKENS);
    const TokenLists solo = check_solo(engine, prompts, golden_ids);
    check_batched(engine, prompts, solo);
    check_one_request_arena(engine, prompts, solo);
    check_fifo_admission(engine, prompts);
    check_mixed_greedy_and_sampled(engine);
    check_invalid_params_rejected(engine);
    check_stop_and_lane_failure(engine, prompts);
    check_abort_through_step(engine, prompts);
    // cancel before it ever ran: gone at once
    engine.cancel(engine.submit(prompts[2], {.max_tokens = MAX_TOKENS}));
    engine.step();
    assert(engine.requests.empty());
    check_no_units_leaked(engine);
    // the engine still works after the abort
    assert(run_to_completion(engine, {prompts[2]})[0] == solo[2]);
    printf("cancel and abort return every unit\nPASS\n");
}
