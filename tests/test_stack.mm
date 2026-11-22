// Whole-stack check: the golden trace (prefill0..2, decode0..2 at batch 2) runs from tokens only, through
// dense(0) .. dense(L) with attn(i) of both kinds in between and lm_head, each call in one command buffer. Every
// K/V page and linear state is this run's own (prefill builds them). Decode logits and greedy ids must equal
// golden row by row. Each decode call also runs at B = 4 on copies of both requests (pages and state copied
// before the call, on other units): its 4 rows must equal the B = 2 rows. Pass: everything bit-exact.
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>

#include "golden.h"
#include "lanes.h"

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 256 << 20);
    const int chunk = g["chunk"], max_batch = 4, rows_max = std::max(chunk, max_batch), max_ctx = 64, L = s.n_layers();
    mllm::GpuOps ops(mem, w);
    mllm::Workspace ws = ops.workspace(chunk, max_batch, max_ctx);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    mllm::StepTables tables = mem.step_tables(rows_max);
    mem.freeze();
    id<MTLBuffer> table = ws.table = tables.copies[0];

    // 6 requests on shuffled units: 0, 1 = the trace's; 2..5 = copies of 0, 1, 0, 1 for the B = 4 runs
    const size_t unit = mem.arena.unit_bytes, V = s.vocab;
    const uint32_t SU = mem.state_units, P = mem.units_for(max_ctx);
    std::vector<uint32_t> perm(mem.arena.n_units);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(3));
    assert(6 * (SU + P) <= perm.size());
    std::vector<std::vector<uint32_t>> st(6), pg(6);
    for (int r = 0; r < 6; r++) {
        auto at = perm.begin() + r * (SU + P);
        st[r].assign(at, at + SU), pg[r].assign(at + SU, at + SU + P);
    }
    uint8_t* arena = (uint8_t*)mem.arena_buffer.contents;
    for (int r = 0; r < 2; r++)
        for (uint32_t u : st[r]) memset(arena + u * unit, 0, unit);  // a new request's state is zero
    auto copy_request = [&](int from, int to) {
        for (uint32_t j = 0; j < SU; j++) memcpy(arena + st[to][j] * unit, arena + st[from][j] * unit, unit);
        for (uint32_t j = 0; j < P; j++) memcpy(arena + pg[to][j] * unit, arena + pg[from][j] * unit, unit);
    };

    // one step through the whole stack; decode steps also run lm_head and return the greedy ids
    auto step = [&](const std::vector<int>& reqs, const std::vector<int>& pos, int n, const uint8_t* tokens) {
        ws.B = (int)reqs.size(), ws.S = n;
        std::vector<mllm::StepTables::Row> rows;
        for (size_t b = 0; b < reqs.size(); b++) rows.push_back({pos[b], n, pg[reqs[b]], st[reqs[b]]});
        tables.write(table, rows);
        memcpy(ws.tokens.contents, tokens, (size_t)ws.B * n * 4);
        const bool decode = n == 1;
        @autoreleasepool {
            mllm::encode_and_wait(q, ws, [&] {
                for (int i = 0; i <= L; i++) {
                    if (i == L && !decode) break;  // prefill never runs dense(L)
                    ops.dense(i, ws);
                    if (i < L) ops.attn(i, ws);
                }
                if (decode) ops.lm_head(ws);
            });
        }
        return decode ? ops.sample(ws, std::vector<mllm::SampleParams>(ws.B)) : std::vector<int>{};
    };

    int decodes = 0;
    for (const auto& call : g["calls"]) {
        std::string c = call["name"].get<std::string>();
        const mllm::Tensor& tok = tr.at(c + "/tokens");
        std::vector<int> req, pos;
        for (const auto& r : call["rows"]) req.push_back(r["req"]), pos.push_back(r["pos"]);
        const int n = (int)tok.shape[1];
        if (n > 1) {
            step(req, pos, n, tr.data(tok));
            continue;
        }
        assert(req == std::vector<int>({0, 1}));
        for (int r = 2; r < 6; r++) copy_request(r % 2, r);  // before the B = 2 step changes 0 and 1
        std::vector<int> ids = step(req, pos, 1, tr.data(tok));
        const mllm::Tensor &want = tr.at(c + "/logits"), &am = tr.at(c + "/argmax");
        const std::vector<uint8_t> logits((uint8_t*)ws.logits.contents, (uint8_t*)ws.logits.contents + want.nbytes);
        for (int b = 0; b < 2; b++) {  // row by row
            const size_t row = V * 2;
            if (memcmp(logits.data() + b * row, tr.data(want) + b * row, row)) {
                golden::matches(c + "/logits", want, tr.data(want), logits.data(), 1, 0);  // prints cos, max |diff|
                assert(!"logits row differs from golden");
            }
            assert(ids[b] == golden::load<int32_t>(tr.data(am), b));
        }
        // B = 4: copies 2, 3, 4, 5 of 0, 1, 0, 1, tokens and positions repeated
        std::vector<uint8_t> tok4(4 * 4);
        for (int b = 0; b < 4; b++) memcpy(&tok4[b * 4], tr.data(tok) + (b % 2) * 4, 4);
        std::vector<int> ids4 = step({2, 3, 4, 5}, {pos[0], pos[1], pos[0], pos[1]}, 1, tok4.data());
        for (int b = 0; b < 4; b++) {
            assert(!memcmp((uint8_t*)ws.logits.contents + b * V * 2, logits.data() + (b % 2) * V * 2, V * 2));
            assert(ids4[b] == ids[b % 2]);
        }
        decodes++;
    }
    assert(decodes == 3);
    printf("whole stack from tokens: decode0..2 logits and greedy ids bit-exact row by row at B = 2; "
           "B = 4 copies equal the B = 2 rows\nPASS\n");
}
