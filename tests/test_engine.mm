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

constexpr int N = 64, CHUNK = 32;  // small chunks, so a prompt spans several chunks and pages

static void check_arena(const mllm::Engine& e) {
    const mllm::Arena& a = e.mem.arena;
    assert(a.free_units + a.cache_only_units == a.n_units && a.reserved == 0 && "arena units leaked");
}

static std::vector<std::vector<int>> run(mllm::Engine& e, const std::vector<std::vector<mllm::Message>>& prompts) {
    std::vector<int> ids;
    for (const auto& p : prompts) ids.push_back(e.submit(p, {.max_tokens = N}));
    std::map<int, std::vector<int>> out;
    while (!e.requests.empty())
        for (const mllm::Event& ev : e.step()) out[ev.req_id].push_back(ev.token);
    check_arena(e);
    std::vector<std::vector<int>> res;
    for (int i : ids) res.push_back(out[i]);
    return res;
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tk(d + "/tokens.safetensors");
    std::vector<std::vector<mllm::Message>> prompts;
    std::vector<std::vector<int>> golden_ids;
    for (int i = 0; i < 3; i++) {
        prompts.emplace_back();
        for (const auto& m : g["chat"][i]) prompts.back().push_back({m["role"], m["content"]});
        const mllm::Tensor& t = tk.at("greedy/" + std::to_string(i));
        golden_ids.emplace_back(t.nbytes / 4);
        for (size_t j = 0; j < golden_ids.back().size(); j++) golden_ids.back()[j] = golden::load<int32_t>(tk.data(t), j);
    }

    mllm::Engine e(golden::MODEL, g["layout"]["arena_bytes"].get<size_t>(), CHUNK);  // 0.8B 256 MiB, 27B 1 GiB
    // 1. one request at a time == mlx-lm greedy, token for token
    std::vector<std::vector<int>> solo;
    for (int i = 0; i < 3; i++) {
        solo.push_back(run(e, {prompts[i]})[0]);
        assert(solo.back() == golden_ids[i]);
    }
    printf("solo: 3 x %d greedy tokens equal golden\n", N);
    // 2. all at once (batched decode) == each alone; every unit nothing holds is NaN first (bf16 0xFFFF): a kernel that
    // reads a reused unit past what this request wrote (masked K/V, a cold state) would show it (llama.cpp 0a7675746)
    {
        const mllm::Arena& a = e.mem.arena;
        for (uint32_t u = 0; u < a.n_units; u++)
            if (!a.request_refs[u] && !a.cache_refs[u])
                memset((uint8_t*)e.mem.arena_buffer.contents + (size_t)u * a.unit_bytes, 0xFF, a.unit_bytes);
    }
    assert(run(e, prompts) == solo);
    printf("batched: equal solo\n");

    // 3. arena fits one request at a time: the rest WAIT, then run, same tokens
    size_t longest = 0;
    for (const auto& p : prompts) longest = std::max(longest, e.tokenizer.encode_chat(p).size());
    const uint32_t need = e.mem.units_for(longest + N) + e.mem.state_units;
    {
        mllm::Engine small(golden::MODEL, need * e.mem.arena.unit_bytes, CHUNK);
        assert(run(small, prompts) == solo);

        check_arena(small);
    }
    printf("wait ok with a %u-unit arena\n", need);
    {
        // FIFO: B runs; A waits (doesn't fit next to B); C would fit next to B but must not overtake A
        const std::vector<int> pa = e.tokenizer.encode_chat(prompts[0]), pb = e.tokenizer.encode_chat(prompts[2]), pc = {1, 2, 3};
        auto demand = [&](size_t n, int gen) { return e.mem.units_for(n + gen) + e.mem.state_units; };
        const uint32_t db = demand(pb.size(), 8), da = demand(pa.size(), N), dc = demand(pc.size(), 4);
        assert(db + da > db + dc && da <= db + dc);
        mllm::Engine fifo(golden::MODEL, (db + dc) * e.mem.arena.unit_bytes, CHUNK);
        const int b = fifo.submit(pb, {.max_tokens = 8}), a = fifo.submit(pa, {.max_tokens = N});
        const int c = fifo.submit(pc, {.max_tokens = 4});
        std::vector<int> admitted;  // in admission order
        while (!fifo.requests.empty()) {
            fifo.step();
            for (int r : {a, b, c})
                if (std::find(admitted.begin(), admitted.end(), r) == admitted.end() &&
                    (!fifo.requests.count(r) || fifo.requests.at(r).state != mllm::State::WAITING))
                    admitted.push_back(r);
        }
        assert(admitted == std::vector<int>({b, a, c}));
        check_arena(fifo);
    }
    printf("FIFO: the big waiter is not overtaken\n");
    {
        // a sampled request next to a greedy one gets a token every step (greedy rows aren't fed ahead of it)
        const int g = e.submit(std::vector<int>{1}, {.max_tokens = 24});
        const int smp = e.submit(std::vector<int>{2}, {.max_tokens = 12, .temperature = 0.8f, .top_p = 0.9f});
        int n_g = 0, n_s = 0, last = 0;
        while (!e.requests.empty())
            for (const mllm::Event& ev : e.step()) {
                if (ev.req_id == g) n_g++;
                if (ev.req_id != smp) continue;
                assert(n_s++ == 0 || n_g - last <= 1);  // greedy tokens between two sampled ones
                last = n_g;
            }
        assert(n_s >= 2);
        check_arena(e);
    }
    printf("mixed greedy + sampled: one token per step each\n");
    for (mllm::GenParams p : {mllm::GenParams{.temperature = -1}, {.temperature = NAN}, {.top_p = 1.5f}, {.top_p = NAN}}) {
        bool bad = false;
        try {
            e.submit(std::vector<int>{1}, p);
        } catch (const std::invalid_argument&) {
            bad = true;
        }
        assert(bad && e.requests.empty());
    }

    // 4. a generate() stopped early (cancel mid-decode), and a lane call that fails, both return every unit
    int pieces = 0;
    e.generate(prompts[0], {.max_tokens = N}, [&](const std::string&) { return ++pieces < 3; });
    assert(pieces == 3);
    e.decode.fail_next = true;
    bool threw = false;
    try {
        e.generate(prompts[1], {.max_tokens = N}, [](const std::string&) { return true; });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw && "lane error swallowed");
    run(e, {});  // steps until the cancelled request is polled, then checks the arena
    // the same through step(): the failed rows end with token -1, done, in that very step
    const int f = e.submit(prompts[2], {.max_tokens = N});
    e.decode.fail_next = true;
    for (bool aborted = false; !aborted;) {
        try {
            e.step();
        } catch (const std::runtime_error&) {
            std::vector<mllm::Event> ev = e.step();  // the abort event was kept for the next step() caller
            aborted = ev.size() == 1 && ev[0].req_id == f && ev[0].token == -1 && ev[0].done;
            assert(aborted && e.requests.empty());
        }
    }
    check_arena(e);
    // cancel before it ever ran: gone at once
    e.cancel(e.submit(prompts[2], {.max_tokens = N}));
    e.step();
    assert(e.requests.empty());
    check_arena(e);
    // the engine still works after the abort
    assert(run(e, {prompts[2]})[0] == solo[2]);
    printf("cancel and abort return every unit\nPASS\n");
}
