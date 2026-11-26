// Engine + prefix cache (g8): greedy tokens with the cache equal those without, cold and on hits; hits land where
// snapshots were taken (a prompt's last full page, a branch point); after a hit the keys / values and linear state
// match the cache-off run (bit-exact when the chunk boundaries do); a chat's next turn resumes after the reply (a
// finished request's sequence is cached); the same on ANE prefill; a small arena forcing
// evictions under 20 mixed requests ends with every unit free or cache-held.
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <string>

#include "engine.h"
#include "golden.h"

constexpr int MAX_TOKENS = 64, CHUNK_TOKENS = 512;
// 512 MiB at 0.8B (1365 units): ~16 requests of ~900 tokens (pages + state) cached at once; the same count elsewhere
static const size_t ARENA_BYTES =
    std::max<size_t>(512 << 20, mllm::Memory::arena_for(mllm::spec::load(golden::MODEL), 16, 900));
// 64 MiB at 0.8B (170 units: one state slot of 50 and 3840 tokens of pages); the same elsewhere, in its units
static const size_t SMALL_ARENA_BYTES =
    std::max<size_t>(64 << 20, mllm::Memory::arena_for(mllm::spec::load(golden::MODEL), 1, 3840));

using Tokens = std::vector<int>;
using TokensByPrompt = std::map<Tokens, Tokens>;

struct RunResult {
    Tokens tokens;
    int cached = -1;
    std::vector<float> keys_values, state;  // after prefill + one decode step (max_tokens 1 runs only)
};

static float read_bf16(const uint8_t* bytes) {
    uint32_t bits = (uint32_t)(bytes[0] | bytes[1] << 8) << 16;
    float value;
    memcpy(&value, &bits, 4);
    return value;
}

// one head's vector of one token, as floats
static void append_head_vector(const mllm::Memory& memory, const uint8_t* page, int full_layer, int key_or_value,
                               int head, int token_in_page, std::vector<float>& values) {
    for (int dim = 0; dim < memory.spec.head_dim; dim++)
        values.push_back(read_bf16(page + memory.kv_offset(full_layer, key_or_value, head, token_in_page) + dim * 2));
}

// one token's keys and values over every full layer and head
static void append_token_keys_values(const mllm::Memory& memory, const uint8_t* page, int token_in_page,
                                     std::vector<float>& values) {
    for (int full_layer = 0; full_layer < (int)memory.spec.full_layers().size(); full_layer++)
        for (int key_or_value = 0; key_or_value < 2; key_or_value++)
            for (int head = 0; head < memory.spec.key_value_heads; head++)
                append_head_vector(memory, page, full_layer, key_or_value, head, token_in_page, values);
}

// the state span's conv (bf16) and delta (fp32) parts of every linear layer, as floats
static void append_state(const mllm::Memory& memory, const std::vector<uint32_t>& state_slots,
                         std::vector<float>& values) {
    const uint8_t* arena = (const uint8_t*)memory.arena_buffer.contents;
    const size_t unit_bytes = memory.arena.unit_bytes;
    // the state span's byte b
    auto span_byte = [&](size_t b) { return arena + state_slots[b / unit_bytes] * unit_bytes + b % unit_bytes; };
    for (int layer : memory.spec.linear_layers()) {
        const size_t offset = memory.state_offset(layer);
        for (size_t b = 0; b < memory.conv_bytes; b += 2) values.push_back(read_bf16(span_byte(offset + b)));
        for (size_t b = 0; b < memory.delta_bytes; b += 4) {
            float value;
            memcpy(&value, span_byte(offset + memory.conv_bytes + b), 4);
            values.push_back(value);
        }
    }
}

// One request alone; with max_tokens 1, also its keys / values for positions [0, length) and its state, read from
// the arena after it finished (its units are released but nothing reuses them while the engine is idle)
static RunResult run_alone(mllm::Engine& engine, const Tokens& prompt, int max_tokens = MAX_TOKENS) {
    const int request_id = engine.submit(prompt, {.max_tokens = max_tokens});
    RunResult result;
    std::vector<uint32_t> pages, state_slots;
    while (engine.requests.count(request_id)) {
        for (const mllm::Event& event : engine.step()) result.tokens.push_back(event.token);
        auto it = engine.requests.find(request_id);
        if (it != engine.requests.end() && it->second.state != mllm::State::WAITING)
            pages = it->second.pages, state_slots = it->second.state_slots, result.cached = it->second.cached;
    }
    if (max_tokens > 1) return result;
    const mllm::Memory& memory = engine.mem;
    const uint8_t* arena = (const uint8_t*)memory.arena_buffer.contents;
    for (int token = 0; token < (int)prompt.size(); token++)
        append_token_keys_values(memory, arena + pages[token / memory.page_tokens] * memory.arena.unit_bytes,
                                 token % memory.page_tokens, result.keys_values);
    append_state(memory, state_slots, result.state);
    return result;
}

// cos floor when chunk boundaries move. mlx-lm's own long1300, prefilled whole vs split at 1280 (chunks of 512):
// 0.8B keys / values cos 0.99998, state 0.99996; 27B (64 layers) keys / values 0.99338, state 0.96575. Ours match
// both to ~1e-3. The bit-exact checks (chunk 32, same boundaries) are the strict ones.
static const double MIN_COSINE = getenv("MLLM_MODEL") ? 0.95 : 0.9999;

static void compare_values(const char* what, const std::vector<float>& first, const std::vector<float>& second,
                           bool exact, double min_cosine = MIN_COSINE) {
    assert(first.size() == second.size() && !first.empty());
    double dot = 0, first_norm = 0, second_norm = 0, max_difference = 0;
    for (size_t i = 0; i < first.size(); i++)
        dot += (double)first[i] * second[i], first_norm += (double)first[i] * first[i],
            second_norm += (double)second[i] * second[i];
    for (size_t i = 0; i < first.size(); i++)
        max_difference = std::max(max_difference, (double)std::fabs(first[i] - second[i]));
    const double cosine = dot / std::sqrt(first_norm * second_norm);
    const bool same = !memcmp(first.data(), second.data(), first.size() * 4);
    printf("  %-28s max |diff| %.3g, cos %.8f%s\n", what, max_difference, cosine, same ? " (bit-exact)" : "");
    assert(exact ? same : cosine >= min_cosine);
}

static Tokens first_tokens(const Tokens& tokens, size_t count) {
    assert(tokens.size() >= count);
    return {tokens.begin(), tokens.begin() + count};
}

static std::string repeat_text(const std::string& text, int times) {
    std::string out;
    for (int i = 0; i < times; i++) out += text;
    return out;
}

static size_t matching_prefix_length(const Tokens& first, const Tokens& second) {
    size_t i = 0;
    while (i < first.size() && i < second.size() && first[i] == second[i]) i++;
    return i;
}

// every unit free or held by the cache only
static void check_idle_arena(const mllm::Engine& engine) {
    const mllm::Arena& arena = engine.mem.arena;
    uint32_t cache_only = 0;
    for (uint32_t unit = 0; unit < arena.n_units; unit++)
        assert(!arena.request_refs[unit]), cache_only += arena.cache_refs[unit] > 0;
    assert(arena.reserved == 0 && arena.cache_only_units == cache_only && arena.free_units + cache_only == arena.n_units);
}

// every refcount from its holders (cache nodes, live requests), the counts from the refcounts, child counts
static void check_invariants(const mllm::Engine& engine) {
    const mllm::Arena& arena = engine.mem.arena;
    std::vector<uint32_t> expected_cache_refs(arena.n_units), expected_request_refs(arena.n_units);
    std::map<uint64_t, int> child_counts;
    for (auto& [hash, node] : engine.cache->nodes) {
        expected_cache_refs[node.unit]++;
        for (uint32_t unit : node.snapshot) expected_cache_refs[unit]++;
        if (engine.cache->nodes.count(node.parent)) child_counts[node.parent]++;
    }
    uint32_t reserved = 0, unheld = 0, cache_only = 0;
    for (auto& [_, request] : engine.requests) {
        if (request.state == mllm::State::DONE) continue;
        reserved += request.reserved;
        for (uint32_t unit : request.pages) expected_request_refs[unit]++;
        for (uint32_t unit : request.state_slots) expected_request_refs[unit]++;
    }
    for (uint32_t unit = 0; unit < arena.n_units; unit++) {
        assert(arena.cache_refs[unit] == expected_cache_refs[unit] &&
               arena.request_refs[unit] == expected_request_refs[unit]);
        unheld += !expected_request_refs[unit] && !expected_cache_refs[unit];
        cache_only += !expected_request_refs[unit] && expected_cache_refs[unit];
    }
    assert(unheld == arena.free_units && cache_only == arena.cache_only_units && reserved == arena.reserved);
    for (auto& [hash, node] : engine.cache->nodes) assert(node.children == child_counts[hash]);
}

// QA's stress (g8qa_cache): 80 random requests (shared bases, multi-turn extensions, repeats; random cancels in
// prefill and decode) on a small arena. Invariants every step, cached units never change while cached, no stall.
struct StressTest {
    const int chunk, seed;
    mllm::Engine engine{golden::MODEL, SMALL_ARENA_BYTES, chunk, 32, 16, 16384, "gpu", true};
    std::mt19937 random_engine{(std::mt19937::result_type)seed};
    std::uniform_int_distribution<int> percent{0, 99}, random_token{1000, 100000};
    std::vector<Tokens> shared_prompts = {}, past_prompts = {};
    // request id -> (cancel at prefill position, cancel at output count), -1 = no
    std::map<int, std::pair<int, int>> cancel_plans = {};
    std::map<std::pair<uint64_t, uint32_t>, uint64_t> checksums = {};  // (node, unit) -> checksum while the node holds it
    std::set<int> hit_requests = {};
    int submitted = 0;

    Tokens random_tokens(int count) {
        Tokens tokens(count);
        for (int& token : tokens) token = random_token(random_engine);
        return tokens;
    }

    uint64_t unit_checksum(uint32_t unit) const {
        const uint64_t* words = (const uint64_t*)((const uint8_t*)engine.mem.arena_buffer.contents +
                                                  (size_t)unit * engine.mem.arena.unit_bytes);
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i < engine.mem.arena.unit_bytes / 8; i += 61)  // sampled
            hash = (hash ^ words[i]) * 1099511628211ull;
        return hash;
    }

    Tokens random_prompt() {
        Tokens prompt;
        const int kind = percent(random_engine);
        if (kind < 30 && !past_prompts.empty()) {  // multi-turn: an earlier prompt + more
            prompt = past_prompts[percent(random_engine) % past_prompts.size()];
            auto more = random_tokens(1 + percent(random_engine));
            prompt.insert(prompt.end(), more.begin(), more.end());
        } else if (kind < 45 && !past_prompts.empty()) {
            prompt = past_prompts[percent(random_engine) % past_prompts.size()];  // repeated
        } else {
            prompt = shared_prompts[percent(random_engine) % 3];
            prompt.resize(std::max<size_t>(2, prompt.size() - percent(random_engine) * 2));
            auto more = random_tokens(percent(random_engine) * 3);
            prompt.insert(prompt.end(), more.begin(), more.end());
        }
        if (prompt.size() > 2500) prompt.resize(2500);
        return prompt;
    }

    void submit_random_request() {
        Tokens prompt = random_prompt();
        past_prompts.push_back(prompt);
        const int max_tokens = 1 + percent(random_engine) % 40, cancel_kind = percent(random_engine);
        const int request_id = engine.submit(prompt, {.max_tokens = max_tokens});
        cancel_plans[request_id] = {cancel_kind < 15 ? percent(random_engine) * (int)prompt.size() / 100 : -1,
                                    cancel_kind >= 15 && cancel_kind < 30 ? percent(random_engine) % max_tokens : -1};
        submitted++;
    }

    void cancel_planned_and_note_hits() {
        for (auto& [request_id, request] : engine.requests) {
            if (request.state == mllm::State::DONE || request.cancelled) continue;
            const auto [cancel_at_position, cancel_at_output] = cancel_plans[request_id];
            if ((cancel_at_position >= 0 && request.state == mllm::State::PREFILL &&
                 request.position >= cancel_at_position) ||
                (cancel_at_output >= 0 && request.state == mllm::State::DECODE &&
                 (int)request.generated.size() >= cancel_at_output))
                engine.cancel(request_id);
            if (request.cached > 0 && request.state != mllm::State::WAITING) hit_requests.insert(request_id);
        }
    }

    void check_cached_units_unchanged() {
        std::map<std::pair<uint64_t, uint32_t>, uint64_t> current;
        for (auto& [hash, node] : engine.cache->nodes) {
            current[{hash, node.unit}] = unit_checksum(node.unit);
            for (uint32_t unit : node.snapshot) current[{hash ^ 0x9e3779b97f4a7c15ull, unit}] = unit_checksum(unit);
        }
        for (auto& [key, checksum] : current)
            assert((!checksums.count(key) || checksums[key] == checksum) && "a cached unit changed");
        checksums = std::move(current);
    }

    void run() {
        for (int count : {300, 650, 900}) shared_prompts.push_back(random_tokens(count));
        auto last_event_at = std::chrono::steady_clock::now();
        while (submitted < 80 || !engine.requests.empty()) {
            if (submitted < 80 && engine.requests.size() < 10 && percent(random_engine) < 30) submit_random_request();
            if (!engine.step().empty()) last_event_at = std::chrono::steady_clock::now();
            // no token for this long = stalled; 0.8B prefills 2500 tokens in < 1 s, the 27B on the GPU in ~20-35 s
            assert(std::chrono::steady_clock::now() - last_event_at <
                       std::chrono::seconds(getenv("MLLM_MODEL") ? 120 : 10) &&
                   "stalled");
            cancel_planned_and_note_hits();
            check_invariants(engine);
            check_cached_units_unchanged();
        }
        check_idle_arena(engine);
        while (engine.cache->evict()) {}
        assert(engine.mem.arena.cache_only_units == 0 && engine.mem.arena.free_units == engine.mem.arena.n_units);
        printf("  stress chunk %d seed %d, %zu MiB: 80 requests, %zu hits; invariants held\n", chunk, seed,
               SMALL_ARENA_BYTES >> 20, hit_requests.size());
        assert(!hit_requests.empty());
    }
};

static void stress(int chunk, int seed) {
    StressTest test{chunk, seed};
    test.run();
}

// The prompts the sections share
struct TestPrompts {
    std::vector<Tokens> golden_chats;  // the 3 golden chats
    Tokens long_text_1300, long_text_1281;  // 1281: 1280 = 40 full pages
    // A shared system prompt (~790 tokens), then different questions. Chat, not the repeated text: greedy on that
    // flips at near-ties when only the chunk boundaries move (cache off, chunk 32 vs 512: from token 16 on).
    // long_question's last full page lies past the branch point (a long question), so haiku_question misses;
    // haiku_question snapshots the branch point.
    Tokens long_question, haiku_question, departure_question, grain_question, timber_question;
    size_t shared_prefix_tokens = 0;
    int branch_point = 0;
    // aligned: truck_head's last full page ends at its last token, so truck_question (truck_head + more) hits
    // truck_head's own snapshot (at 768)
    Tokens truck_question, truck_head;
    std::vector<Tokens> all;
};

static TestPrompts make_prompts(const nlohmann::json& golden_spec, const mllm::Tokenizer& tokenizer) {
    TestPrompts prompts;
    for (int i = 0; i < 3; i++) {
        std::vector<mllm::Message> chat;
        for (const auto& message : golden_spec["chat"][i]) chat.push_back({message["role"], message["content"]});
        prompts.golden_chats.push_back(tokenizer.encode_chat(chat));
    }
    const Tokens text = tokenizer.encode(repeat_text(
        "The river runs past the mill, and the miller counts the sacks. Each morning the baker buys flour; each "
        "evening the children walk home along the water. ", 60));
    prompts.long_text_1300 = first_tokens(text, 1300), prompts.long_text_1281 = first_tokens(text, 1281);
    auto reference_chat = [&](const char* vehicle, const char* question) {
        std::string system = std::string("You are a helpful assistant for a shipping company. Reference data follows.\n");
        const char* cargo[] = {"salt", "iron", "letters", "grain", "timber"};
        for (int k = 1; k <= 40; k++)
            system += std::string(vehicle) + " " + std::to_string(k) + ": departs port " + std::to_string(k * 7 % 13) +
                      " at " + std::to_string(k % 12 + 1) + " o'clock carrying " + cargo[k % 5] + ".\n";
        return tokenizer.encode_chat({{"system", system}, {"user", question}});
    };
    prompts.long_question = reference_chat("Ship", "Which ships carry salt, which of them leave before noon, and from "
                                                   "which ports do they depart? Answer as a short list with one ship "
                                                   "per line.");
    prompts.haiku_question = reference_chat("Ship", "Write a haiku about ship 7.");
    prompts.departure_question = reference_chat("Ship", "When does ship 12 depart?");
    prompts.grain_question = reference_chat("Ship", "List the ships that carry grain.");
    prompts.timber_question = reference_chat("Ship", "Does any ship carry timber?");
    size_t& shared = prompts.shared_prefix_tokens;
    while (prompts.long_question[shared] == prompts.haiku_question[shared] &&
           prompts.long_question[shared] == prompts.departure_question[shared])
        shared++;
    prompts.branch_point = (int)shared / 32 * 32;
    prompts.truck_question = reference_chat("Truck", "Which trucks carry iron?");
    prompts.truck_head = first_tokens(prompts.truck_question, 769);
    prompts.all = {prompts.golden_chats[0], prompts.golden_chats[1], prompts.golden_chats[2], prompts.long_text_1300,
                   prompts.long_text_1281, prompts.long_question, prompts.haiku_question, prompts.departure_question,
                   prompts.grain_question, prompts.timber_question, prompts.truck_question, prompts.truck_head};
    return prompts;
}

static std::vector<Tokens> load_golden_ids(const mllm::Safetensors& golden_tokens) {
    std::vector<Tokens> golden_ids;
    for (int i = 0; i < 3; i++) {
        const mllm::Tensor& tensor = golden_tokens.at("greedy/" + std::to_string(i));
        golden_ids.emplace_back(tensor.nbytes / 4);
        for (size_t j = 0; j < golden_ids.back().size(); j++)
            golden_ids.back()[j] = golden::load<int32_t>(golden_tokens.data(tensor), j);
    }
    return golden_ids;
}

// cold, then the same prompt again: hits at its last full page
static void check_repeat_hits_last_full_page(mllm::Engine& on, const TestPrompts& prompts, TokensByPrompt& off_tokens) {
    for (const auto& prompt : {prompts.golden_chats[0], prompts.golden_chats[1], prompts.golden_chats[2],
                               prompts.long_text_1300}) {
        RunResult cold = run_alone(on, prompt), hit = run_alone(on, prompt);
        const int last_full_page = ((int)prompt.size() - 1) / 32 * 32;
        printf("  len %4zu: cold cached %d, again cached %d (last full page %d), tokens %s\n", prompt.size(),
               cold.cached, hit.cached, last_full_page,
               cold.tokens == off_tokens[prompt] && hit.tokens == cold.tokens ? "equal" : "DIFFER");
        assert(cold.cached == 0 && hit.cached == last_full_page && cold.tokens == off_tokens[prompt] &&
               hit.tokens == cold.tokens);
    }
}

// Shared prefix: long_question caches its last full page (not on haiku_question's path); haiku_question misses and
// snapshots the branch point, departure_question hits there. Their chunk boundaries move (long_question: its last
// full page splits a chunk; departure_question: resumes at the branch point), which flips greedy near-ties as much
// as cache-off chunk 64 vs 512 does (keys / values cos 0.99991 there): tokens are only reported here; the chunk-32
// section below checks them where the boundaries are the same.
static void check_shared_prefix_hits(mllm::Engine& on, const TestPrompts& prompts, TokensByPrompt& off_tokens) {
    RunResult long_run = run_alone(on, prompts.long_question), haiku_run = run_alone(on, prompts.haiku_question),
              departure_run = run_alone(on, prompts.departure_question);
    printf("  shared %zu: long question cached %d, haiku %d, departure %d; tokens equal cache off up to %zu, %zu, %zu "
           "of 64\n",
           prompts.shared_prefix_tokens, long_run.cached, haiku_run.cached, departure_run.cached,
           matching_prefix_length(long_run.tokens, off_tokens[prompts.long_question]),
           matching_prefix_length(haiku_run.tokens, off_tokens[prompts.haiku_question]),
           matching_prefix_length(departure_run.tokens, off_tokens[prompts.departure_question]));
    assert((int)(prompts.long_question.size() - 1) / 32 * 32 > prompts.branch_point && long_run.cached == 0 &&
           haiku_run.cached == 0 && departure_run.cached == prompts.branch_point);
    // truck_head's chunks are cache off's (its snapshot point is its prefill's end)
    RunResult head_run = run_alone(on, prompts.truck_head), truck_run = run_alone(on, prompts.truck_question);
    printf("  aligned: truck head (%zu) cached %d, truck question cached %d\n", prompts.truck_head.size(),
           head_run.cached, truck_run.cached);
    assert(head_run.cached == 0 && truck_run.cached == 768 && head_run.tokens == off_tokens[prompts.truck_head]);
}

// Keys / values and state after a hit vs cache off. long_text_1281 hits the snapshot long_text_1300 left at 1280 and
// goes straight to decode, its chunks were cache off's: bit-exact. long_text_1300 (at 1281, where long_text_1281's
// sequence ended: a tail node) and grain_question resume mid-prompt (chunk boundaries move).
static void check_state_after_hit(mllm::Engine& on, mllm::Engine& off, const TestPrompts& prompts) {
    RunResult hit_1281 = run_alone(on, prompts.long_text_1281, 1), off_1281 = run_alone(off, prompts.long_text_1281, 1);
    assert(hit_1281.cached == 1280 && hit_1281.tokens == off_1281.tokens);
    compare_values("long 1281, K/V", hit_1281.keys_values, off_1281.keys_values, true);
    compare_values("long 1281, state", hit_1281.state, off_1281.state, true);
    RunResult hit_1300 = run_alone(on, prompts.long_text_1300, 1), off_1300 = run_alone(off, prompts.long_text_1300, 1);
    assert(hit_1300.cached == 1281);
    compare_values("long 1300, K/V", hit_1300.keys_values, off_1300.keys_values, false);
    compare_values("long 1300, state", hit_1300.state, off_1300.state, false);
    // grain_question: haiku_question's branch point (departure_question left a snapshot at its own end)
    RunResult hit_grain = run_alone(on, prompts.grain_question, 1), off_grain = run_alone(off, prompts.grain_question, 1);
    assert(hit_grain.cached == prompts.branch_point);
    compare_values("grain (branch point), K/V", hit_grain.keys_values, off_grain.keys_values, false);
    compare_values("grain (branch point), state", hit_grain.state, off_grain.state, false);
}

// A chat's next turn (Tokenizer::continue_chat: prompt + reply + the next message) resumes where the reply's sequence
// ended, mid-page (a tail node): after the EOS when the reply ended there (the step feeding it ran), before the last
// token when max_tokens cut it (never fed). Keys / values and state then match cache off, loosely: decode wrote the
// reply's part, cache off prefills it (0.8B: cos 0.99988 .. 0.99992).
static void check_chat_turn_resumes_after_reply(mllm::Engine& on, mllm::Engine& off) {
    const mllm::Tokenizer& tokenizer = on.tokenizer;
    // the reply already written: greedy ends it at once
    const Tokens answered = tokenizer.encode("<|im_start|>user\nCapital of France?<|im_end|>\n<|im_start|>assistant\n"
                                             "<think>\n\n</think>\n\nThe capital of France is Paris.");
    const Tokens open_question = tokenizer.encode_chat({{"user", "Tell me a long story about a lighthouse."}});
    for (const bool answered_turn : {true, false}) {
        const Tokens& turn = answered_turn ? answered : open_question;
        Tokens sequence = turn;
        const Tokens reply = run_alone(on, turn, 16).tokens;
        sequence.insert(sequence.end(), reply.begin(), reply.end());
        const bool ended_at_eos = tokenizer.eos.count(reply.back());
        assert(ended_at_eos == answered_turn);
        const Tokens next_turn = tokenizer.continue_chat(sequence, "And of Italy?");
        RunResult hit = run_alone(on, next_turn, 1), cold = run_alone(off, next_turn, 1);
        const int want = (int)sequence.size() - (ended_at_eos ? 0 : 1);
        printf("  chat: turn %zu + reply %zu (%s), next turn %zu tokens cached %d\n", turn.size(), reply.size(),
               ended_at_eos ? "EOS" : "max_tokens", next_turn.size(), hit.cached);
        assert(hit.cached == want && want % 32 && hit.tokens == cold.tokens);
        compare_values("chat next turn, K/V", hit.keys_values, cold.keys_values, false, std::min(MIN_COSINE, 0.999));
        compare_values("chat next turn, state", hit.state, cold.state, false, std::min(MIN_COSINE, 0.999));
    }
}

// chunk 512, the served configuration
static void check_gpu_chunk_512(const TestPrompts& prompts, const std::vector<Tokens>& golden_ids) {
    TokensByPrompt off_tokens;  // greedy 64, cache off
    mllm::Engine off(golden::MODEL, ARENA_BYTES, CHUNK_TOKENS, 32, 16, 16384, "gpu", false);
    mllm::Engine on(golden::MODEL, ARENA_BYTES, CHUNK_TOKENS, 32, 16, 16384, "gpu", true);
    assert(!off.cache && on.cache);
    for (const auto& prompt : prompts.all) {
        RunResult off_run = run_alone(off, prompt);
        off_tokens[prompt] = off_run.tokens;
        assert(off_run.cached == 0);
    }
    for (int i = 0; i < 3; i++) assert(off_tokens[prompts.golden_chats[i]] == golden_ids[i]);
    printf("off: 3 golden prompts equal golden\n");
    check_repeat_hits_last_full_page(on, prompts, off_tokens);
    check_shared_prefix_hits(on, prompts, off_tokens);
    check_state_after_hit(on, off, prompts);
    check_chat_turn_resumes_after_reply(on, off);
    check_idle_arena(on);
    printf("gpu chunk 512: cold == off, hits == cold\n");
}

// chunk 32: every chunk is one page with or without the cache, so a hit must be bit-exact: tokens, keys / values,
// state. Returns cache off's greedy 64 tokens per prompt.
static TokensByPrompt check_gpu_chunk_32(const TestPrompts& prompts) {
    TokensByPrompt off_tokens;
    mllm::Engine off(golden::MODEL, ARENA_BYTES, 32, 32, 16, 16384, "gpu", false);
    mllm::Engine on(golden::MODEL, ARENA_BYTES, 32, 32, 16, 16384, "gpu", true);
    for (const auto& prompt : prompts.all) off_tokens[prompt] = run_alone(off, prompt).tokens;
    for (const std::vector<int>* each : {&prompts.long_question, &prompts.haiku_question, &prompts.departure_question,
                                         &prompts.grain_question, &prompts.long_question}) {
        const std::vector<int>& prompt = *each;  // by address, so the checks below can tell which prompt this is
        RunResult hit = run_alone(on, prompt), hit_once = run_alone(on, prompt, 1), off_once = run_alone(off, prompt, 1);
        printf("  chunk 32, len %zu: cached %d then %d\n", prompt.size(), hit.cached, hit_once.cached);
        assert(hit.tokens == off_tokens[prompt] && hit_once.tokens == off_once.tokens &&
               hit_once.keys_values == off_once.keys_values && hit_once.state == off_once.state);
        assert(hit_once.cached == ((int)prompt.size() - 1) / 32 * 32);
        assert(&prompt != &prompts.grain_question || hit.cached == prompts.branch_point);
    }
    check_idle_arena(on);
    printf("gpu chunk 32: hits bit-exact with cache off (tokens, K/V, state)\n");
    return off_tokens;
}

// ANE prefill: the same hits; long_text_1300's tokens equal ANE's cache-off run (the others as at chunk 512)
static void check_ane_prefill(const TestPrompts& prompts) {
    mllm::Engine off(golden::MODEL, ARENA_BYTES, CHUNK_TOKENS, 32, 16, 16384, "ane", false);
    mllm::Engine on(golden::MODEL, ARENA_BYTES, CHUNK_TOKENS, 32, 16, 16384, "ane", true);
    for (const std::vector<int>* each : {&prompts.long_text_1300, &prompts.long_question, &prompts.haiku_question,
                                         &prompts.departure_question, &prompts.timber_question}) {
        const std::vector<int>& prompt = *each;  // by address, so the checks below can tell which prompt this is
        RunResult off_run = run_alone(off, prompt), cold = run_alone(on, prompt), hit = run_alone(on, prompt);
        const int last_full_page = ((int)prompt.size() - 1) / 32 * 32;
        printf("  ane len %4zu: cached %d then %d, tokens equal cache off up to %zu, %zu of 64\n", prompt.size(),
               cold.cached, hit.cached, matching_prefix_length(cold.tokens, off_run.tokens),
               matching_prefix_length(hit.tokens, off_run.tokens));
        assert(hit.cached == last_full_page &&
               (&prompt != &prompts.timber_question || cold.cached == prompts.branch_point));
        assert(&prompt != &prompts.long_text_1300 || (cold.tokens == off_run.tokens && hit.tokens == off_run.tokens));
    }
    check_idle_arena(on);
    printf("ane: hits where expected\n");
}

// SMALL_ARENA_BYTES (0.8B: 170 units; one request's state is 50): admissions evict; 20 mixed requests at once.
// Chunk 32 so that outputs must equal cache off's solo runs.
static void check_small_arena_evictions(mllm::Engine& on, const TestPrompts& prompts, TokensByPrompt& off_tokens) {
    std::map<int, Tokens> expected, received;
    for (int i = 0; i < 20; i++) {
        // in pairs: the second can hit what the first left
        const auto& prompt = prompts.all[(i / 2 * 5) % prompts.all.size()];
        expected[on.submit(prompt, {.max_tokens = 16})] =
            first_tokens(off_tokens[prompt], std::min<size_t>(16, off_tokens[prompt].size()));
    }
    std::set<int> doomed;  // cancelled mid-prefill: their later pages must not be cached, their units return
    for (const auto& prompt : {prompts.long_text_1300, prompts.long_question, prompts.long_text_1300})
        doomed.insert(on.submit(prompt, {.max_tokens = 16}));
    std::set<uint64_t> seen_nodes;
    std::set<int> hit_requests;
    auto should_cancel = [&](int request_id) {
        return on.requests.count(request_id) && on.requests.at(request_id).position >= 256 &&
               !on.requests.at(request_id).cancelled;
    };
    while (!on.requests.empty()) {
        for (const mllm::Event& event : on.step()) received[event.req_id].push_back(event.token);
        for (int request_id : doomed)
            if (should_cancel(request_id)) on.cancel(request_id);
        for (auto& [hash, _] : on.cache->nodes) seen_nodes.insert(hash);
        for (auto& [request_id, request] : on.requests)
            if (request.state != mllm::State::WAITING && request.cached > 0) hit_requests.insert(request_id);
    }
    int evicted = 0;
    for (uint64_t hash : seen_nodes) evicted += !on.cache->nodes.count(hash);
    int same = 0;
    for (auto& [request_id, tokens] : expected) same += received[request_id] == tokens;
    for (int request_id : doomed) assert(received[request_id].empty());
    printf("  small arena: %u units, %zu hits, %d nodes evicted, %zu left; %d / 20 outputs equal cache off, "
           "%zu cancelled\n", on.mem.arena.n_units, hit_requests.size(), evicted, on.cache->nodes.size(), same,
           doomed.size());
    assert(same == 20 && evicted > 0 && !hit_requests.empty());
    check_idle_arena(on);
}

// a hit whose demand plus its held pages and snapshot exceed the arena is admitted (without copies, or cold), not
// kept waiting forever (QA: demand == n_units, the same prompt again)
static void check_oversized_hit_admitted(mllm::Engine& on, const TestPrompts& prompts) {
    run_alone(on, prompts.long_text_1300, 16);  // cached at 1280 (two copies don't fit: it keeps the last one)
    const int request_id = on.submit(
        prompts.long_text_1300, {.max_tokens = (int)(on.mem.arena.n_units - on.mem.state_units) * 32 - 1300});
    assert(on.cache->match(prompts.long_text_1300).resume_pages > 0);
    const auto submitted_at = std::chrono::steady_clock::now();
    while (on.requests.at(request_id).state == mllm::State::WAITING) {
        on.step();
        assert(std::chrono::steady_clock::now() - submitted_at < std::chrono::seconds(5) && "a big hit waits forever");
    }
    on.cancel(request_id);
    while (!on.requests.empty()) on.step();
    check_idle_arena(on);
}

int main(int argc, char** argv) {
    // At 27B the whole test takes ~28 min (g15 logs: `cache` 652 s, stress 508 + 494 s), too close to a 30-min hold:
    // `cache` runs all but the stress runs, `stress512` / `stress32` one stress run each. No argument: everything.
    const std::string part = argc > 1 ? argv[1] : "";
    if (part == "stress512" || part == "stress32") {
        part == "stress512" ? stress(512, 3) : stress(32, 1);
        printf("PASS\n");
        return 0;
    }
    assert(part.empty() || part == "cache");
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string golden_dir = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(golden_dir + "/spec.json"));
    mllm::Safetensors golden_tokens(golden_dir + "/tokens.safetensors");
    mllm::Tokenizer tokenizer(model.path);
    const TestPrompts prompts = make_prompts(golden_spec, tokenizer);
    const std::vector<Tokens> golden_ids = load_golden_ids(golden_tokens);

    check_gpu_chunk_512(prompts, golden_ids);
    TokensByPrompt off_tokens_chunk_32 = check_gpu_chunk_32(prompts);
    try {
        check_ane_prefill(prompts);
    } catch (const std::runtime_error& error) {  // another model ($MLLM_MODEL) may have no ANE blocks built yet
        if (!getenv("MLLM_MODEL")) throw;
        printf("ane: skipped (%.60s...)\n", error.what());
    }
    {
        mllm::Engine on(golden::MODEL, SMALL_ARENA_BYTES, 32, 32, 16, 16384, "gpu", true);
        check_small_arena_evictions(on, prompts, off_tokens_chunk_32);
        check_oversized_hit_admitted(on, prompts);
    }
    if (part.empty()) stress(512, 3), stress(32, 1);
    printf("PASS\n");
}
