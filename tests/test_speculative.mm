// Speculative decoding end to end (g23, docs/speculative.md): Engine with speculate on vs off, B = 1 greedy.
//  1. tolerance: on the 3 golden chats and tools/spec/accept.py's 3 long prompts, speculative output equals plain greedy
//     up to the first divergence; there the plain step's scores for the two tokens are within NEAR_TIE (a near-tie the
//     verify's multi-row kernels may round the other way);
//  2. a poisoned head (its keys / values NaN for the whole prompt): output still within tolerance, acceptance drops;
//  3. prefix cache: the next turn after a speculative reply (a hit) is within tolerance of a cold plain run and drafts
//     about as well as a cold speculative run.
// `test_speculative bench`: decode tok/s on accept.py's 6 prompts, speculative vs plain interleaved (S P S P).
// Models without an MTP head (the 0.8B) skip. At 27B: under the heavy lock, MLLM_MODEL=mlx-community/Qwen3.6-27B-4bit.
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

#include "engine.h"
#include "golden.h"

using Tokens = std::vector<int>;
// The verify can flip two tokens only if each score moves: test_rollback at 27B measured max |3-row verify (and the
// decode after it) - 1-row| = 0.60 over the vocab (g23b), so 2 x 0.60. Measured gaps at divergences: 0 and 0.125.
constexpr double NEAR_TIE = 1.2;

static std::string read_file(const std::string& path) {
    std::ifstream file(path);
    assert(file && "run from the repo root");
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

// Python's text[:n]: the first n code points of UTF-8
static std::string first_chars(const std::string& text, size_t n) {
    size_t i = 0, chars = 0;
    for (; i < text.size(); i++)
        if ((text[i] & 0xC0) != 0x80 && chars++ == n) break;
    return text.substr(0, i);
}

// accept.py's PROMPTS, templated as it does (enable_thinking off: an empty think block)
static std::vector<std::pair<std::string, Tokens>> accept_prompts(const mllm::Tokenizer& tokenizer) {
    const std::vector<std::tuple<std::string, std::string, bool>> prompts = {
        {"chat", "What are the main differences between TCP and UDP, and when would you pick each? Give concrete examples.",
         false},
        {"code",
         "Write a Python function that parses an ISO 8601 duration such as 'P3DT4H5M6S' into a number of seconds. "
         "Reject malformed input with a ValueError, and add a few unit tests with unittest.",
         false},
        {"code-edit",
         "Refactor this function to be readable and fix any bugs, then show the full new version:\n\n" +
             first_chars(read_file("tools/serve_bench.py"), 3000),
         false},
        {"reasoning",
         "A train leaves city A at 9:00 at 80 km/h toward city B, 300 km away. Another leaves B at 9:30 at 100 km/h "
         "toward A. At what time do they meet, and how far from A? Explain step by step.",
         true},
        {"summary",
         "Summarize the following design document in about ten bullet points:\n\n" +
             first_chars(read_file("docs/architecture.md"), 12000),
         false},
        {"plan-think",
         "Plan a three-day trip to Kyoto for two people who like food and quiet temples, on a moderate budget. Give a "
         "day-by-day plan.",
         true},
    };
    std::vector<std::pair<std::string, Tokens>> out;
    for (const auto& [name, text, think] : prompts) {
        Tokens tokens = tokenizer.encode_chat({{"user", text}});  // ends "<think>", "\n"
        if (!think) {
            tokens.pop_back();
            for (int token : tokenizer.encode("\n\n</think>\n\n")) tokens.push_back(token);
        }
        out.push_back({name, tokens});
    }
    return out;
}

struct Run {
    Tokens tokens;
    double tok_s = 0;  // decode: tokens after the first / the time after it
    uint64_t rounds = 0, round_tokens = 0;
    int cached = 0;
    std::vector<std::vector<uint16_t>> scores;  // plain runs with scores: each step's
    double per_round() const { return rounds ? (double)round_tokens / rounds : 0; }
};

// the head's keys / values (the full layer after the model's) of every position before `end`, NaN
static void poison_head(mllm::Engine& engine, const mllm::Request& r, int end) {
    const mllm::Memory& mem = engine.mem;
    const int head_layer = (int)mem.spec.full_layers().size();
    for (int token = 0; token < end; token++)
        for (int key_or_value = 0; key_or_value < 2; key_or_value++)
            for (int head = 0; head < mem.spec.key_value_heads; head++)
                memset((uint8_t*)mem.arena_buffer.contents + (size_t)r.pages[token / mem.page_tokens] * mem.arena.unit_bytes +
                           mem.kv_offset(head_layer, key_or_value, head, token % mem.page_tokens),
                       0xFF, mem.spec.head_dim * mllm::ELEM);
}

static Run run(mllm::Engine& engine, const Tokens& prompt, int max_tokens, bool speculate, bool scores = false,
               bool poison = false) {
    Run out;
    engine.speculate = speculate;
    if (scores)
        engine.gpu.on_scores = [&](const uint16_t* row, size_t count) { out.scores.emplace_back(row, row + count); };
    const uint64_t rounds = engine.rounds, round_tokens = engine.round_tokens;
    const int id = engine.submit(prompt, {.max_tokens = max_tokens});
    std::chrono::steady_clock::time_point first, last;
    for (bool done = false; !done;)
        for (const mllm::Event& event : engine.step()) {
            if (event.req_id != id) continue;
            assert(event.token >= 0);
            last = std::chrono::steady_clock::now();
            if (out.tokens.empty()) {
                first = last;
                // the prompt's head keys / values are in place now (prefill); decode writes the rest
                if (poison) poison_head(engine, engine.requests.at(id), engine.requests.at(id).position);
            }
            out.tokens.push_back(event.token), out.cached = event.cached, done = event.done;
        }
    while (engine.requests.count(id)) engine.step();  // a fed step past its end
    engine.gpu.on_scores = nullptr, engine.speculate = true;
    out.rounds = engine.rounds - rounds, out.round_tokens = engine.round_tokens - round_tokens;
    out.tok_s = out.tokens.size() > 1 ? (out.tokens.size() - 1) / std::chrono::duration<double>(last - first).count() : 0;
    return out;
}

static void evict_all(mllm::Engine& engine) {
    while (engine.cache->evict()) {}
}

// spec equals plain up to the first divergence; there the plain step's scores for both tokens are a near-tie
static void check_tolerance(const std::string& what, const Run& plain, const Run& spec, int vocab) {
    size_t d = 0;
    while (d < plain.tokens.size() && d < spec.tokens.size() && plain.tokens[d] == spec.tokens[d]) d++;
    if (d == plain.tokens.size() && d == spec.tokens.size()) {
        printf("  %s: %zu tokens equal plain; %.2f tokens per round\n", what.c_str(), d, spec.per_round());
        return;
    }
    assert(d < plain.tokens.size() && d < spec.tokens.size() && "one stopped early with the other's same tokens");
    const std::vector<uint16_t>& row = plain.scores.at(d);
    assert((int)row.size() == vocab);
    const auto score = [&](int token) { return golden::to_double("BF16", (const uint8_t*)row.data(), token); };
    const double gap = score(plain.tokens[d]) - score(spec.tokens[d]);
    printf("  %s: equal plain for %zu of %zu tokens, then a near-tie: plain's scores %.3f vs %.3f (gap %.3f); %.2f "
           "tokens per round\n", what.c_str(), d, plain.tokens.size(), score(plain.tokens[d]), score(spec.tokens[d]), gap,
           spec.per_round());
    assert(gap >= 0 && gap <= NEAR_TIE && "speculative output left plain greedy at no near-tie");
}

static std::vector<Tokens> golden_chats(const mllm::Engine& engine) {
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(golden::dir(engine.spec) + "/spec.json"));
    std::vector<Tokens> chats;
    for (int i = 0; i < 3; i++) {
        std::vector<mllm::Message> chat;
        for (const auto& message : golden_spec["chat"][i]) chat.push_back({message["role"], message["content"]});
        chats.push_back(engine.tokenizer.encode_chat(chat));
    }
    return chats;
}

static void bench(mllm::Engine& engine) {
    constexpr int TOKENS = 256;
    printf("prompt      tokens  speculative tok/s (tokens per round)   plain tok/s\n");
    double spec_sum = 0, plain_sum = 0;
    const auto prompts = accept_prompts(engine.tokenizer);
    for (const auto& [name, prompt] : prompts) {
        run(engine, prompt, 8, false);  // warm: the prompt's prefill into the cache
        Run spec[2], plain[2];
        for (int trial = 0; trial < 2; trial++)
            spec[trial] = run(engine, prompt, TOKENS, true), plain[trial] = run(engine, prompt, TOKENS, false);
        const double spec_tps = (spec[0].tok_s + spec[1].tok_s) / 2, plain_tps = (plain[0].tok_s + plain[1].tok_s) / 2;
        printf("%-11s %6zu  %5.1f %5.1f (%.2f)                    %5.1f %5.1f   x%.2f\n", name.c_str(), prompt.size(),
               spec[0].tok_s, spec[1].tok_s, spec[0].per_round(), plain[0].tok_s, plain[1].tok_s, spec_tps / plain_tps);
        spec_sum += spec_tps, plain_sum += plain_tps;
        evict_all(engine);
    }
    printf("mean: speculative %.1f tok/s, plain %.1f tok/s (x%.2f)\n", spec_sum / prompts.size(),
           plain_sum / prompts.size(), spec_sum / plain_sum);
}

int main(int argc, char** argv) {
    const mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    if (!model.mtp) {
        printf("no MTP head for %s (tools/spec/mtp.py weights): skipped\nPASS\n", model.path.c_str());
        return 0;
    }
    if (argc == 1) {  // 0. capacity: an arena for exactly 4 requests runs 4 greedy ones at once (no spare sets held)
        constexpr int REQUESTS = 4, MAX_TOKENS = 32;
        const size_t bytes = mllm::Memory::arena_for(model, REQUESTS, 64 + MAX_TOKENS);
        mllm::Engine small(golden::MODEL, bytes, 1024, 32, 16, 8192, "gpu", false);
        Tokens prompt = small.tokenizer.encode_chat({{"user", "Count from one to twenty in words."}});
        prompt.resize(64, prompt.back());
        for (int j = 0; j < REQUESTS; j++) small.submit(prompt, {.max_tokens = MAX_TOKENS});
        uint64_t steps = 0;
        while (!small.requests.empty()) {
            small.step(), steps++;
            for (auto& [_, r] : small.requests) assert(r.state != mllm::State::WAITING && "greedy requests no longer fit");
        }
        printf("0. %d greedy requests in an arena for %d: all ran at once (%llu steps)\n", REQUESTS, REQUESTS, steps);
    }
    mllm::Engine engine(golden::MODEL, (size_t)4 << 30, 1024, 32, 16, 8192);
    if (argc > 1 && std::string(argv[1]) == "bench") return bench(engine), 0;
    const int vocab = model.vocab_size;

    printf("1. speculative vs plain greedy, 128 tokens\n");
    std::vector<std::pair<std::string, Tokens>> prompts;
    int i = 0;
    for (const Tokens& chat : golden_chats(engine)) prompts.push_back({"golden " + std::to_string(i++), chat});
    for (const auto& [name, prompt] : accept_prompts(engine.tokenizer))
        if (name == "code-edit" || name == "summary" || name == "plan-think") prompts.push_back({name, prompt});
    std::vector<Run> plains;
    for (const auto& [name, prompt] : prompts) {
        evict_all(engine);
        plains.push_back(run(engine, prompt, 128, false, true));
        evict_all(engine);
        const Run spec = run(engine, prompt, 128, true);
        assert(spec.rounds > 0);
        check_tolerance(name + " (" + std::to_string(prompt.size()) + " prompt tokens)", plains.back(), spec, vocab);
    }

    printf("2. the head's keys / values NaN for the prompt: output within tolerance, acceptance lower\n");
    for (int p : {0, 4}) {
        evict_all(engine);
        const Run clean = run(engine, prompts[p].second, 128, true);
        evict_all(engine);
        const Run poisoned = run(engine, prompts[p].second, 128, true, false, true);
        check_tolerance(prompts[p].first + " poisoned", plains[p], poisoned, vocab);
        printf("    tokens per round: %.2f clean, %.2f poisoned\n", clean.per_round(), poisoned.per_round());
        assert(poisoned.per_round() < clean.per_round());
    }
    evict_all(engine);  // the poisoned pages

    printf("3. the next turn after a speculative reply: a prefix-cache hit\n");
    const Tokens& first_turn = prompts[0].second;
    const Run reply = run(engine, first_turn, 128, true);
    Tokens history = first_turn;
    history.insert(history.end(), reply.tokens.begin(), reply.tokens.end());
    const Tokens next_turn = engine.tokenizer.continue_chat(history, "Now say the same in two sentences.");
    const Run hit = run(engine, next_turn, 128, true);
    printf("  %d of %zu prompt tokens cached\n", hit.cached, next_turn.size());
    assert(hit.cached > (int)first_turn.size() && "the reply wasn't cached");
    evict_all(engine);
    const Run cold = run(engine, next_turn, 128, true);
    evict_all(engine);
    const Run plain = run(engine, next_turn, 128, false, true);
    check_tolerance("hit", plain, hit, vocab);
    check_tolerance("cold", plain, cold, vocab);
    assert(hit.per_round() >= 0.9 * cold.per_round() && "the next turn drafts worse after a hit");
    printf("PASS\n");
}
