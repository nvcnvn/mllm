// The decode lane's concurrent encoder (GpuOps::pre's barriers) == a serial one, byte for byte: logits and every
// touched K/V and state byte, one whole decode step repeated 20 times at B = 7, 8, 13, 16, 17 (qmm_t_decode and
// split-k matmuls, which share ws.splitk; rows at mixed positions, some on the 2-pass SDPA path). From g9 QA's race
// harness.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "golden.h"
#include "lanes.h"

int main() {
    using namespace mllm;
    const ModelSpec s = spec::load(golden::MODEL);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    Weights w(s.path, dev);
    const int L = s.n_layers();
    Memory mem(s, dev, (size_t)1 << 30);
    GpuOps ops(mem, w);
    Workspace ws = ops.workspace(1, 17, 4096);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    StepTables tables = mem.step_tables(17, 4096);
    id<MTLBuffer> tab = mem.buffer(tables.words * 4);
    mem.freeze();
    for (int B : {7, 8, 13, 16, 17}) {
        std::mt19937 rng(B);
        std::vector<std::vector<uint32_t>> units(B);
        std::vector<StepTables::Row> rows;
        uint32_t next = 0;
        for (int b = 0; b < B; b++) {
            const int pos = (b * 211 + 37) % 1500 + (b % 3 == 0 ? 0 : 400);  // some >= 1024 keys (2-pass)
            const uint32_t pages = mem.units_for(pos + 1);
            for (uint32_t j = 0; j < pages + mem.state_units; j++) units[b].push_back(next++);
            rows.push_back({pos, 1, std::span(units[b]).first(pages), std::span(units[b]).subspan(pages)});
        }
        tables.write(tab, rows);
        ws.B = B, ws.S = 1, ws.table = tab;
        const size_t bytes = (size_t)next * mem.arena.unit_bytes, lb = (size_t)B * s.vocab * 2;
        std::vector<uint16_t> init(bytes / 2);  // random bf16 in [-0.5, 0.5)
        std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (auto& x : init) {
            float v = u(rng);
            uint32_t bits;
            memcpy(&bits, &v, 4);
            x = bits >> 16;
        }
        for (int b = 0; b < B; b++) ((int32_t*)ws.tokens.contents)[b] = (int)(rng() % s.vocab);
        auto step = [&](bool concurrent) {
            memcpy(mem.arena_buffer.contents, init.data(), bytes);
            encode_and_wait(q, ws, [&] {
                for (int i = 0; i <= L; i++) {
                    ops.dense(i, ws);
                    if (i < L) ops.attn(i, ws);
                }
                ops.lm_head(ws);
            }, concurrent);
            std::vector<uint8_t> r(lb + bytes);
            memcpy(r.data(), ws.logits.contents, lb);
            memcpy(r.data() + lb, mem.arena_buffer.contents, bytes);
            return r;
        };
        const auto ref = step(false);
        int diff = 0;
        for (int k = 0; k < 20; k++) @autoreleasepool {
                diff += step(true) != ref;
            }
        printf("B = %d: concurrent != serial in %d / 20 steps\n", B, diff);
        assert(diff == 0);
    }
    printf("PASS\n");
}
