// attn(i) check for full-attention layers: replay the golden trace (prefill0..2, decode0..2 at batch 2) through
// GpuOps::attn with each request's pages on scattered arena slots, layer by layer, from golden q/k/v/gate.
// Compares ws.o with <call>/LNN/attn/o and, after each call, the request's K/V pages with <call>/rB/LNN/k, v.
// Decode calls run a second time at B = 4 (both requests plus copies of them on other slots): every row must
// equal the B = 2 run. Then attn.safetensors (py/dump_golden.py SWEEP): one row per (n, pos) around MLX's SDPA
// switch points (1-pass / 2-pass and its blocks, vector / unfused, gemv_wide / steel, block / looped softmax), and
// a 1-pass and a 2-pass decode row in one B = 2 call. Pass: everything bit-exact. Also: a padded prefill row
// (n < ws.S), a slot past the arena and a linear layer on a table without state units assert.
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>

#include <cassert>
#include <csignal>
#include <string>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "gpu.h"

// Metal buffer memory isn't mapped in a fork()ed child, so assert checks re-run this binary with a mode argument:
// true when that run dies on SIGABRT (an assert)
static bool aborts(const char* self, const char* mode) {
    pid_t p;
    const char* argv[] = {self, mode, nullptr};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    int rc = posix_spawn(&p, self, &fa, nullptr, (char* const*)argv, nullptr);
    posix_spawn_file_actions_destroy(&fa);
    int st = 0;
    if (rc || waitpid(p, &st, 0) < 0) return false;
    return WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
}

static double max_diff(const uint8_t* got, const uint8_t* want, size_t n) {
    if (!memcmp(got, want, n * 2)) return 0;
    double m = 0;
    for (size_t e = 0; e < n; e++) {
        double a = golden::to_double("BF16", got, e), b = golden::to_double("BF16", want, e);
        m = std::fmax(m, std::isnan(a - b) ? INFINITY : std::fabs(a - b));
    }
    return m;
}

int main(int argc, char** argv) {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 128 << 20);
    const int chunk = 128, max_batch = 4, rows_max = std::max(chunk, max_batch);  // the sweep's longest chunk
    mllm::Workspace ws = mem.workspace(chunk, max_batch);
    mllm::GpuOps ops(mem, w, rows_max, 9216);
    mllm::StepTables tables = mem.step_tables(rows_max);
    mem.freeze();
    id<MTLBuffer> table = ws.table = tables.copies[0];
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "padded") {  // the lane contract: a prefill row is not padded (ws.S = n), MLX picks kernels by q length
        std::vector<uint32_t> sl = {0, 1, 2, 3, 4};
        ws.B = 1, ws.S = 9;
        tables.write(table, {{100, 5, sl}});
        ops.begin();
        ops.attn(s.full_layers()[0], ws);
        return 0;
    }
    if (mode == "nostate") {  // a linear layer with a table written without state units
        tables.write(table, {{0, 1, std::vector<uint32_t>{0}}});
        ws.B = 1, ws.S = 1;
        ops.begin();
        ops.attn(s.linear_layers()[0], ws);
        return 0;
    }
    if (mode == "slot") {  // a slot past the arena
        tables.write(table, {{0, 1, std::vector<uint32_t>{mem.arena.n_units}}});
        return 0;
    }

    const int Hkv = s.n_kv_heads, D = s.head_dim, T = mem.T;
    const size_t unit = mem.arena.unit_bytes;
    uint8_t* arena = (uint8_t*)mem.arena_buffer.contents;
    // pages on scattered, out-of-order slots; requests 2 and 3 are copies of 0 and 1 (the B = 4 decode)
    std::vector<std::vector<uint32_t>> slots = {{5, 2}, {9}, {12, 7}, {13}};

    // golden row src[r] of key -> ws row r; q/k/v come as [B, H, S, D], ours are [B, S, H, D]
    auto load = [&](const std::string& key, id<MTLBuffer> dst, const std::vector<int>& src) {
        const mllm::Tensor& t = tr.at(key);
        const uint8_t* p = tr.data(t);
        size_t row = t.nbytes / t.shape[0];
        assert(row * src.size() <= dst.length);
        uint8_t* o = (uint8_t*)dst.contents;
        for (size_t r = 0; r < src.size(); r++) {
            const uint8_t* in = p + src[r] * row;
            if (t.shape.size() < 4) {
                memcpy(o + r * row, in, row);
                continue;
            }
            int64_t Hh = t.shape[1], S = t.shape[2];
            for (int64_t h = 0; h < Hh; h++)
                for (int64_t si = 0; si < S; si++) memcpy(o + r * row + ((si * Hh + h) * D) * 2, in + ((h * S + si) * D) * 2, D * 2);
        }
    };
    // K/V of full layer gi on pages sl, tokens 0 .. n - 1, as golden's [Hkv, n, D]
    auto pages = [&](const std::vector<uint32_t>& sl, int gi, int kv, int n) {
        std::vector<uint8_t> out((size_t)Hkv * n * D * 2);
        for (int h = 0; h < Hkv; h++)
            for (int t = 0; t < n; t++)
                memcpy(&out[((size_t)h * n + t) * D * 2], arena + sl[t / T] * unit + mem.kv_offset(gi, kv, h, t % T), D * 2);
        return out;
    };

    struct Stat {
        int n = 0, exact = 0;
        double max_abs = 0;
    };
    std::map<std::string, Stat> stats;
    auto record = [&](const std::string& field, const std::string& what, double diff) {
        Stat& st = stats[field];
        st.n++, st.exact += diff == 0, st.max_abs = std::fmax(st.max_abs, diff);
        if (diff != 0) printf("  %s: max |diff| %g\n", what.c_str(), diff);
    };

    const auto full = s.full_layers();
    int b4_rows = 0;
    for (const auto& call : g["calls"]) {
        std::string c = call["name"].get<std::string>();
        const mllm::Tensor& tok = tr.at(c + "/tokens");
        ws.B = (int)tok.shape[0], ws.S = (int)tok.shape[1];
        std::vector<int> req, pos, n;
        for (const auto& r : call["rows"]) req.push_back(r["req"]), pos.push_back(r["pos"]), n.push_back(r["n"]);
        auto write_table = [&](const std::vector<int>& reqs) {
            std::vector<mllm::StepTables::Row> rows;
            for (size_t b = 0; b < reqs.size(); b++) rows.push_back({pos[b % pos.size()], n[b % pos.size()], slots[reqs[b]]});
            tables.write(table, rows);
        };
        std::vector<int> ident(ws.B);
        for (int b = 0; b < ws.B; b++) ident[b] = b;

        for (size_t gi = 0; gi < full.size(); gi++) {
            char li[8];
            snprintf(li, sizeof li, "L%02d", full[gi]);
            std::string pre = c + "/" + li + "/";
            auto run = [&](const std::vector<int>& src, const std::vector<int>& reqs) {
                ws.B = (int)src.size();
                write_table(reqs);
                for (const char* f : {"q", "k", "v", "gate"})
                    load(pre + "dense/" + f, f[0] == 'q' ? ws.q : f[0] == 'k' ? ws.k : f[0] == 'v' ? ws.v : ws.gate, src);
                @autoreleasepool {
                    ops.begin();
                    ops.attn(full[gi], ws);
                    ops.commit_and_wait();
                }
            };
            run(ident, req);
            const mllm::Tensor& o = tr.at(pre + "attn/o");
            record("o", pre + "attn/o", max_diff((const uint8_t*)ws.o.contents, tr.data(o), o.numel()));

            for (size_t b = 0; b < req.size(); b++)
                for (int kv = 0; kv < 2; kv++) {
                    std::string key = c + "/r" + std::to_string(b) + "/" + li + "/" + (kv ? "v" : "k");
                    const mllm::Tensor& t = tr.at(key);
                    assert(t.shape[1] == pos[b] + n[b]);
                    record("kv", key, max_diff(pages(slots[req[b]], (int)gi, kv, pos[b] + n[b]).data(), tr.data(t), t.numel()));
                }
            if (c.starts_with("decode")) {  // B = 4: copies of both requests on other slots, rows must not change
                std::vector<uint8_t> o2((const uint8_t*)ws.o.contents, (const uint8_t*)ws.o.contents + o.nbytes);
                for (int r = 0; r < 2; r++)
                    for (size_t p = 0; p < slots[r].size(); p++) memcpy(arena + slots[r + 2][p] * unit, arena + slots[r][p] * unit, unit);
                run({0, 1, 0, 1}, {req[0], req[1], 2 + req[0], 2 + req[1]});
                for (int b = 0; b < 4; b++, b4_rows++)
                    assert(!memcmp((const uint8_t*)ws.o.contents + b * o.nbytes / 2, o2.data() + (b % 2) * o.nbytes / 2, o.nbytes / 2));
                ws.B = 2;
            }
        }
    }
    // --- the sweep, on the last full layer; pages on the arena's last units downwards (as dump_golden) or upwards
    mllm::Safetensors sw(d + "/attn.safetensors");
    const int gl = (int)full.size() - 1;
    std::vector<uint32_t> down(mem.arena.n_units), up(mem.arena.n_units);
    for (uint32_t u = 0; u < mem.arena.n_units; u++) up[u] = u, down[u] = mem.arena.n_units - 1 - u;
    // rows: (case, pages); loads the inputs and the first pos tokens of the case's pages, runs attn, checks o / K/V
    auto sweep = [&](const std::vector<std::pair<std::string, const std::vector<uint32_t>*>>& rows) {
        std::vector<mllm::StepTables::Row> trows;
        size_t at = 0;
        for (auto& [c, sl] : rows) {
            int n, pos;
            sscanf(c.c_str(), "%d_%d", &n, &pos);
            ws.B = (int)rows.size(), ws.S = n;
            for (auto [f, buf] : {std::pair{"q", ws.q}, {"k", ws.k}, {"v", ws.v}, {"gate", ws.gate}}) {
                const mllm::Tensor& t = sw.at(c + "/" + f);
                memcpy((uint8_t*)buf.contents + at * t.nbytes, sw.data(t), t.nbytes);
            }
            for (int kv = 0; kv < 2; kv++) {
                const mllm::Tensor& t = sw.at(c + (kv ? "/kv_v" : "/kv_k"));
                for (int h = 0; h < Hkv; h++)
                    for (int tk = 0; tk < pos; tk++)
                        memcpy(arena + (*sl)[tk / T] * unit + mem.kv_offset(gl, kv, h, tk % T),
                               sw.data(t) + ((size_t)h * (pos + n) + tk) * D * 2, D * 2);
            }
            trows.push_back({pos, n, std::span(sl->data(), mem.units_for(pos + n))});
            at++;
        }
        tables.write(table, trows);
        @autoreleasepool {
            ops.begin();
            ops.attn(full[gl], ws);
            ops.commit_and_wait();
        }
        at = 0;
        for (auto& [c, sl] : rows) {
            const mllm::Tensor& o = sw.at(c + "/o");
            std::string what = "sweep " + c + (rows.size() > 1 ? " (B = " + std::to_string(rows.size()) + ")" : "");
            record("sweep o", what + " o", max_diff((const uint8_t*)ws.o.contents + at++ * o.nbytes, sw.data(o), o.numel()));
            for (int kv = 0; kv < 2; kv++) {
                const mllm::Tensor& t = sw.at(c + (kv ? "/kv_v" : "/kv_k"));
                record("sweep kv", what + (kv ? " v" : " k"), max_diff(pages(*sl, gl, kv, (int)t.shape[1]).data(), sw.data(t), t.numel()));
            }
        }
    };
    std::vector<std::string> cases;
    for (const auto& [k, t] : sw.tensors)
        if (k.ends_with("/o")) cases.push_back(k.substr(0, k.size() - 2));
    for (const std::string& c : cases) sweep({{c, &down}});
    sweep({{"1_1023", &up}, {"1_4200", &down}});  // 1-pass row and 2-pass row in one decode call
    sweep({{"1_4200", &up}, {"1_0", &down}});

    assert(aborts(argv[0], "padded") && aborts(argv[0], "slot") && aborts(argv[0], "nostate"));

    bool ok = true;
    for (const auto& [name, st] : stats) {
        printf("%-8s bit-exact %3d/%3d  max |diff| %g\n", name.c_str(), st.exact, st.n, st.max_abs);
        ok &= st.exact == st.n;
    }
    printf("B = 4 decode: %d rows equal to the B = 2 run\n", b4_rows);
    printf("sweep: %zu cases + 2 mixed decode calls\n", cases.size());
    assert(ok && b4_rows == 3 * 6 * 4 && cases.size() >= 20);
    printf("PASS\n");
}
