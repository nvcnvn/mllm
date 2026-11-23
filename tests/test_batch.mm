// Batched decode by tolerance (architecture.md): teacher-forced decode of batch_ref's 8 sequences
// (py/dump_batch_ref.py), every position at S = 1 from position 0, through GpuOps on a concurrent encoder (as the
// decode lane). B = 1: the reference's top-64 logits byte for byte. B = 8 and 16 (each sequence on 1 or 2 rows, rows
// at different positions): mean KL(ref || ours) over the reference's top-64 plus a rest bucket <= KL_MAX, and our
// argmax within one bf16 ulp of the reference's top logit at >= 99% of positions.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "golden.h"
#include "lanes.h"

// mlx-lm's own KL(B=1 || B=8) is 7.5e-4 mean (B=16 9.2e-4) on batch_ref: 10x that. A wrong kernel lands far above
// (8 stale outputs per qmv_wide matmul: 1.8); a bad weight group doesn't (one group x 2 in each layer's gate_proj:
// 1.3e-3, inside rounding noise): the B = 1 byte check and test_kernels' CPU references catch those.
// Top-1 within one ulp: mlx-lm's strict top-1 is 98.5% at B = 8, every miss a tie or 1-ulp gap at B = 1.
static const double KL_MAX = 7.5e-3, TOP1_MIN = 0.99;

static float bf16(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

int main() {
    using namespace mllm;
    const ModelSpec s = spec::load(golden::MODEL);
    Safetensors ref(golden::dir(s) + "/batch_ref.safetensors");
    const Tensor &tok_t = ref.at("tokens"), &ids_t = ref.at("ids"), &top_t = ref.at("logits"), &lse_t = ref.at("lse");
    const int NS = (int)tok_t.shape[0], T = (int)tok_t.shape[1], TOP = (int)ids_t.shape[2], V = s.vocab, L = s.n_layers();
    auto tok = [&](int q, int p) { return golden::load<int32_t>(ref.data(tok_t), (size_t)q * T + p); };
    auto rid = [&](int q, int p, int k) { return golden::load<int32_t>(ref.data(ids_t), ((size_t)q * T + p) * TOP + k); };
    auto rtop = [&](int q, int p, int k) { return golden::load<uint16_t>(ref.data(top_t), ((size_t)q * T + p) * TOP + k); };

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    Weights w(s.path, dev);
    Memory mem(s, dev, (size_t)1 << 30);
    GpuOps ops(mem, w);
    Workspace ws = ops.workspace(1, 16, T);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    StepTables tables = mem.step_tables(16, T);
    id<MTLBuffer> tab = mem.buffer(tables.words * 4);
    mem.freeze();
    const uint32_t pages = mem.units_for(T), per_row = pages + mem.state_units;
    assert((size_t)16 * per_row <= mem.arena.n_units);

    // one decode step: row b = (seq[b], pos[b]) in fresh-per-run units b; returns nothing, logits in ws.logits
    std::vector<std::vector<uint32_t>> units(16);
    for (uint32_t b = 0; b < 16; b++)
        for (uint32_t j = 0; j < per_row; j++) units[b].push_back(b * per_row + j);
    auto step = [&](const std::vector<int>& seq, const std::vector<int>& pos, const std::vector<int>& slot) {
        @autoreleasepool {
            std::vector<StepTables::Row> rows;
            for (size_t b = 0; b < seq.size(); b++) {
                rows.push_back({pos[b], 1, std::span(units[slot[b]]).first(pages), std::span(units[slot[b]]).subspan(pages)});
                ((int32_t*)ws.tokens.contents)[b] = tok(seq[b], pos[b]);
            }
            tables.write(tab, rows);
            ws.B = (int)seq.size(), ws.S = 1, ws.table = tab;
            encode_and_wait(q, ws, [&] {
                for (int i = 0; i <= L; i++) {
                    ops.dense(i, ws);
                    if (i < L) ops.attn(i, ws);
                }
                ops.lm_head(ws);
            }, true);
        }
    };
    // row b of ws.logits against the reference at (sq, p): KL over its top-64 + rest, and top-1 within 1 ulp
    double kl_sum = 0, kl_max = 0;
    int n = 0, top1 = 0, near = 0;
    auto check = [&](int b, int sq, int p) {
        const uint16_t* row = (const uint16_t*)ws.logits.contents + (size_t)b * V;
        double m = -INFINITY, sum = 0;
        int arg = 0;
        for (int v = 0; v < V; v++)
            if (bf16(row[v]) > m) m = bf16(row[v]), arg = v;
        for (int v = 0; v < V; v++) sum += std::exp((double)bf16(row[v]) - m);
        const double lse_q = m + std::log(sum), lse_p = golden::load<float>(ref.data(lse_t), (size_t)sq * T + p);
        double kl = 0, p_top = 0, q_top = 0, ref_arg = -INFINITY;
        for (int k = 0; k < TOP; k++) {
            const int id = rid(sq, p, k);
            const double lp = bf16(rtop(sq, p, k)) - lse_p, lq = bf16(row[id]) - lse_q;
            kl += std::exp(lp) * (lp - lq), p_top += std::exp(lp), q_top += std::exp(lq);
            if (id == arg) ref_arg = bf16(rtop(sq, p, k));
        }
        const double p_rest = std::fmax(1 - p_top, 1e-30), q_rest = std::fmax(1 - q_top, 1e-30);
        kl += p_rest * std::log(p_rest / q_rest);
        const double top = bf16(rtop(sq, p, 0));
        kl_sum += kl, kl_max = std::fmax(kl_max, kl), n++;
        top1 += arg == rid(sq, p, 0);
        near += ref_arg >= top - std::exp2(std::floor(std::log2(std::fabs(top))) - 7);
    };
    auto reset = [&] {
        memset(mem.arena_buffer.contents, 0, (size_t)16 * per_row * mem.arena.unit_bytes);  // fresh state: zero
        kl_sum = kl_max = 0, n = top1 = near = 0;
    };
    auto report = [&](int B) {
        printf("B=%2d: %d positions, KL mean %.3e max %.3e (limit %.1e), top-1 %.2f%%, within 1 ulp %.2f%%\n", B, n,
               kl_sum / n, kl_max, KL_MAX, 100.0 * top1 / n, 100.0 * near / n);
    };

    // B = 1: byte for byte with the reference's top-64
    reset();
    int bad = 0;
    for (int sq = 0; sq < NS; sq++)
        for (int p = 0; p < T; p++) {
            step({sq}, {p}, {sq});  // a unit set per sequence: each starts from zero state
            check(0, sq, p);
            const uint16_t* row = (const uint16_t*)ws.logits.contents;
            for (int k = 0; k < TOP; k++) bad += row[rid(sq, p, k)] != rtop(sq, p, k);
        }
    report(1);
    assert(bad == 0 && top1 == n && "B = 1 decode differs from mlx-lm");
    for (int B : {8, 16}) {
        // row r: sequence r % 8, offset d (distinct per row). Warm-up (not checked, B ramps up): row r decodes
        // positions 0 .. d - 1. Then 80 steps at the full B, row r at positions d .. d + 79, all checked.
        const int D = 48;
        std::vector<int> d(B);
        for (int r = 0; r < B; r++) d[r] = r * 23 % D;
        reset();
        for (int t = 0; t < T; t++) {
            std::vector<int> seq, pos, slot;
            for (int r = 0; r < B; r++) {
                const int p = t < D ? t - (D - d[r]) : d[r] + t - D;
                if (p >= 0) seq.push_back(r % NS), pos.push_back(p), slot.push_back(r);
            }
            if (seq.empty()) continue;
            step(seq, pos, slot);
            if (t >= D)
                for (int b = 0; b < B; b++) check(b, seq[b], pos[b]);
        }
        report(B);
        assert(kl_sum / n <= KL_MAX && "batched decode: mean KL over the limit");
        assert(near >= TOP1_MIN * n && "batched decode: top-1 agreement under 99%");
    }
    printf("PASS\n");
}
