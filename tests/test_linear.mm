// attn(i) check for linear-attention layers: replay the golden trace (prefill0..2, decode0..2 at batch 2) through
// GpuOps::attn with each request's state span on scattered, out-of-order arena units, layer by layer, from golden
// qkv/z/b/a. Compares ws.o with <call>/LNN/attn/o and, after each call, the request's conv and delta state with
// <call>/rB/LNN/conv, delta. Decode calls run at B = 5: both requests, copies of them on other units (rows must equal
// the originals, o and state) and a row with n = 0 (its state must not change). Pass: everything bit-exact.
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <string>

#include "golden.h"
#include "lanes.h"

static double max_diff(const std::string& dtype, const uint8_t* got, const uint8_t* want, size_t n) {
    if (!memcmp(got, want, n * (dtype == "F32" ? 4 : 2))) return 0;
    double m = 0;
    for (size_t e = 0; e < n; e++) {
        double a = golden::to_double(dtype, got, e), b = golden::to_double(dtype, want, e);
        m = std::fmax(m, std::isnan(a - b) ? INFINITY : std::fabs(a - b));
    }
    return m;
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 256 << 20);
    const int chunk = g["chunk"], max_batch = 5, rows_max = std::max(chunk, max_batch);
    mllm::GpuOps ops(mem, w);
    ops.merge = false;  // the fields are loaded here, not left by dense's merged M = 1 matmuls
    mllm::Workspace ws = ops.workspace(chunk, max_batch, 1024);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    mllm::StepTables tables = mem.step_tables(rows_max);
    mem.freeze();
    id<MTLBuffer> table = ws.table = tables.copies[0];

    // 5 requests' state spans on shuffled units: 0, 1 = the trace's requests, 2, 3 = their copies, 4 = idle row
    const size_t unit = mem.arena.unit_bytes, SU = mem.state_units, span = SU * unit;
    assert(5 * SU <= mem.arena.n_units);
    std::vector<uint32_t> perm(mem.arena.n_units);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(7));
    std::vector<std::vector<uint32_t>> st(5);
    for (int r = 0; r < 5; r++) st[r].assign(perm.begin() + r * SU, perm.begin() + (r + 1) * SU);
    uint8_t* arena = (uint8_t*)mem.arena_buffer.contents;
    for (int r = 0; r < 4; r++)
        for (uint32_t u : st[r]) memset(arena + u * unit, 0, unit);
    std::mt19937 rng(1);
    for (uint32_t u : st[4])
        for (size_t i = 0; i < unit; i++) arena[u * unit + i] = (uint8_t)rng();
    // span byte x of request r <-> arena
    auto span_copy = [&](int r, size_t x, uint8_t* out, size_t n) {
        for (size_t k; n; x += k, out += k, n -= k) {
            k = std::min(n, unit - x % unit);
            memcpy(out, arena + st[r][x / unit] * unit + x % unit, k);
        }
    };
    auto state = [&](int r) {
        std::vector<uint8_t> v(span);
        span_copy(r, 0, v.data(), span);
        return v;
    };
    auto set_state = [&](int r, const std::vector<uint8_t>& v) {
        for (size_t j = 0; j < SU; j++) memcpy(arena + st[r][j] * unit, v.data() + j * unit, unit);
    };
    const std::vector<uint8_t> idle = state(4);

    // golden rows src[b] of key -> ws row b
    auto load = [&](const std::string& key, id<MTLBuffer> dst, const std::vector<int>& src) {
        const mllm::Tensor& t = tr.at(key);
        size_t row = t.nbytes / t.shape[0];
        assert(row * src.size() <= dst.length);
        for (size_t b = 0; b < src.size(); b++) memcpy((uint8_t*)dst.contents + b * row, tr.data(t) + src[b] * row, row);
    };

    struct Stat {
        int n = 0, exact = 0;
        double max_abs = 0;
    };
    std::map<std::string, Stat> stats;
    auto record = [&](const std::string& field, const std::string& what, double diff) {
        Stat& x = stats[field];
        x.n++, x.exact += diff == 0, x.max_abs = std::fmax(x.max_abs, diff);
        if (diff != 0) printf("  %s: max |diff| %g\n", what.c_str(), diff);
    };

    std::map<int, int> req_of;  // trace request id -> 0, 1
    int dup_rows = 0;
    for (const auto& call : g["calls"]) {
        std::string c = call["name"].get<std::string>();
        const mllm::Tensor& tok = tr.at(c + "/tokens");
        const int B = (int)tok.shape[0];
        ws.S = (int)tok.shape[1];
        std::vector<int> req, pos, n;
        for (const auto& r : call["rows"]) {
            int id = r["req"];
            if (!req_of.count(id)) req_of[id] = (int)req_of.size();
            req.push_back(req_of[id]), pos.push_back(r["pos"]), n.push_back(r["n"]);
        }
        const bool dup = c.starts_with("decode");  // + copies of both rows and an n = 0 row
        std::vector<int> src(B), reqs = req;
        std::iota(src.begin(), src.end(), 0);
        std::vector<mllm::StepTables::Row> rows;
        for (int b = 0; b < B; b++) rows.push_back({pos[b], n[b], {}, st[req[b]]});
        if (dup) {
            assert(B == 2);
            for (int b = 0; b < 2; b++) {
                set_state(2 + req[b], state(req[b]));
                rows.push_back({pos[b], n[b], {}, st[2 + req[b]]}), src.push_back(b), reqs.push_back(2 + req[b]);
            }
            rows.push_back({0, 0, {}, st[4]}), src.push_back(0), reqs.push_back(4);
        }
        ws.B = (int)rows.size();
        tables.write(table, rows);

        for (int li : s.linear_layers()) {
            char lname[8];
            snprintf(lname, sizeof lname, "L%02d", li);
            std::string pre = c + "/" + lname + "/";
            load(pre + "dense/qkv", ws.qkv, src), load(pre + "dense/z", ws.z, src);
            load(pre + "dense/b", ws.b, src), load(pre + "dense/a", ws.a, src);
            @autoreleasepool {
                mllm::encode_and_wait(q, ws, [&] {
                    ops.attn(li, ws);
                });
            }
            const mllm::Tensor& o = tr.at(pre + "attn/o");
            const uint8_t* got = (const uint8_t*)ws.o.contents;
            record("o", pre + "attn/o", max_diff("BF16", got, tr.data(o), o.numel()));
            if (dup)
                for (int b = 2; b < 4; b++, dup_rows++) assert(!memcmp(got + b * o.nbytes / 2, got + src[b] * o.nbytes / 2, o.nbytes / 2));
        }
        // states after the call
        for (int b = 0; b < B; b++)
            for (const char* f : {"conv", "delta"}) {
                char key[64];
                for (int li : s.linear_layers()) {
                    snprintf(key, sizeof key, "%s/r%d/L%02d/%s", c.c_str(), b, li, f);
                    const mllm::Tensor& t = tr.at(key);
                    size_t off = mem.state_offset(li) + (f[0] == 'd' ? mem.conv_bytes : 0);
                    std::vector<uint8_t> got(t.nbytes);
                    span_copy(req[b], off, got.data(), t.nbytes);
                    record(f, key, max_diff(t.dtype, got.data(), tr.data(t), t.numel()));
                }
            }
        if (dup) {
            for (int b = 2; b < 4; b++) assert(state(reqs[b]) == state(req[src[b]]));
            assert(state(4) == idle);
        }
    }

    bool ok = true;
    for (const auto& [name, x] : stats) {
        printf("%-6s bit-exact %3d/%3d  max |diff| %g\n", name.c_str(), x.exact, x.n, x.max_abs);
        ok &= x.exact == x.n;
    }
    const int L = (int)s.linear_layers().size();
    printf("decode at B = 5: %d copied rows equal to the originals (o and state), n = 0 row's state untouched\n", dup_rows);
    assert(ok && stats["o"].n == 6 * L && stats["delta"].n == 9 * L && dup_rows == 3 * L * 2);
    printf("PASS\n");
}
