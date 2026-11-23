// dense(i) check: for every dense(i) call in the golden trace (147: prefill never runs dense(n_layers)), load its inputs from golden
// (tokens, or x from dense(i - 1) and o from attn(i - 1)), run GpuOps::dense, compare every field it wrote.
// Pass: cosine >= 0.99999. Prints per field how many calls were bit-exact and the worst max |diff|.
// Covers: embedding (L00 x), RMSNorm (input norm feeds every projection; q/k norm; final norm = L24 x),
// every projection kind and both qmv_wide (decode, 2 rows) and qmm_t_splitk (prefill, 24-32 rows).
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "lanes.h"

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 64 << 20);
    int chunk = g["chunk"].get<int>(), max_batch = 2;
    mllm::GpuOps ops(mem, w);
    mllm::Workspace ws = ops.workspace(chunk, max_batch, 64);
    id<MTLCommandQueue> q = [dev newCommandQueue];
    mem.freeze();

    std::map<std::string, id<MTLBuffer>> field = {{"x", ws.x}, {"q", ws.q}, {"k", ws.k}, {"v", ws.v},
        {"gate", ws.gate}, {"qkv", ws.qkv}, {"z", ws.z}, {"b", ws.b}, {"a", ws.a}};
    auto load = [&](const std::string& key, id<MTLBuffer> dst) {
        const mllm::Tensor& t = tr.at(key);
        assert(t.nbytes <= dst.length);
        memcpy(dst.contents, tr.data(t), t.nbytes);
    };
    struct Stat {
        int calls = 0, exact = 0;
        double worst_cos = 1, max_abs = 0;
    };
    std::map<std::string, Stat> stats;
    bool ok = true;
    const int L = s.n_layers();
    int runs = 0;
    std::vector<uint8_t> got;
    for (const auto& call : g["calls"]) {
        std::string c = call["name"].get<std::string>();
        const mllm::Tensor& tok = tr.at(c + "/tokens");
        ws.B = (int)tok.shape[0], ws.S = (int)tok.shape[1];
        for (int i = 0; i <= L; i++) {
            char li[8], lp[8];
            snprintf(li, sizeof li, "L%02d", i), snprintf(lp, sizeof lp, "L%02d", i - 1);
            if (!tr.tensors.count(c + "/" + li + "/dense/x")) continue;  // dense(L) in a prefill call
            runs++;
            if (i == 0) {
                load(c + "/tokens", ws.tokens);
            } else {
                load(c + "/" + lp + "/dense/x", ws.x);
                load(c + "/" + lp + "/attn/o", ws.o);
            }
            @autoreleasepool {
                mllm::encode_and_wait(q, ws, [&] {
                    ops.dense(i, ws);
                });
            }

            std::string pre = c + "/" + li + "/dense/";
            for (const auto& [name, buf] : field) {
                auto it = tr.tensors.find(pre + name);
                if (it == tr.tensors.end()) continue;
                const mllm::Tensor& t = it->second;
                const uint8_t* src = (const uint8_t*)buf.contents;
                got.assign(src, src + t.nbytes);
                if (t.shape.size() == 4 && (name == "q" || name == "k" || name == "v")) {
                    // golden [B, H, S, D]; ours [B, S, H, D]: bring ours to golden's order
                    int64_t B = t.shape[0], H = t.shape[1], S = t.shape[2], D = t.shape[3];
                    for (int64_t b = 0; b < B; b++)
                        for (int64_t h = 0; h < H; h++)
                            for (int64_t si = 0; si < S; si++)
                                memcpy(&got[(((b * H + h) * S + si) * D) * 2], src + (((b * S + si) * H + h) * D) * 2, D * 2);
                }
                Stat& st = stats[name];
                st.calls++;
                const uint8_t* want = tr.data(t);
                if (!memcmp(got.data(), want, t.nbytes)) {
                    st.exact++;
                    continue;
                }
                double dot = 0, gg = 0, ww = 0;
                for (size_t e = 0; e < t.numel(); e++) {
                    double a = golden::to_double("BF16", got.data(), e), b = golden::to_double("BF16", want, e);
                    dot += a * b, gg += a * a, ww += b * b;
                    st.max_abs = std::fmax(st.max_abs, std::fabs(a - b));
                }
                st.worst_cos = std::fmin(st.worst_cos, dot / std::sqrt(gg * ww));
                if (!golden::matches(pre + name, t, want, got.data(), 0.99999, INFINITY)) ok = false;
            }
        }
    }
    for (const auto& [name, st] : stats)
        printf("%-5s bit-exact %3d/%3d  worst cos %.7f  max |diff| %g\n", name.c_str(), st.exact, st.calls,
               st.worst_cos, st.max_abs);
    int prefills = 0;
    for (const auto& call : g["calls"]) prefills += call["name"].get<std::string>().starts_with("prefill");
    assert(runs == (int)g["calls"].size() * (L + 1) - prefills && stats["x"].calls == runs);
    assert(ok);

    // M = 1 (no golden call has it): the merged matmuls (GpuOps::merge) give the same bytes as the separate ones.
    // Row 0 of decode0's inputs; z / b / a / v are read where the merged row leaves them (ws.qkv, ws.qg).
    const size_t E = 2, C = s.conv_dim(), vd = (size_t)s.lin_v_heads * s.lin_v_dim, Hv = s.lin_v_heads;
    const size_t qd = (size_t)s.n_heads * s.head_dim, kvd = (size_t)s.n_kv_heads * s.head_dim;
    ws.B = 1, ws.S = 1;
    int m1_checked = 0;
    for (int i = 0; i <= L; i++) {
        char lp[8];
        snprintf(lp, sizeof lp, "L%02d", i - 1);
        const bool lin = i < L && s.kinds[i] == "linear", full = i < L && !lin;
        // (name, separate path's buffer, merged path's buffer and byte offset, bytes)
        std::vector<std::tuple<std::string, id<MTLBuffer>, id<MTLBuffer>, size_t, size_t>> f = {
            {"x", ws.x, ws.x, 0, s.hidden * E}};
        if (lin)
            f.insert(f.end(), {{"qkv", ws.qkv, ws.qkv, 0, C * E}, {"z", ws.z, ws.qkv, C * E, vd * E},
                               {"b", ws.b, ws.qkv, (C + vd) * E, Hv * E}, {"a", ws.a, ws.qkv, (C + vd + Hv) * E, Hv * E}});
        if (full)
            f.insert(f.end(), {{"q", ws.q, ws.q, 0, qd * E}, {"gate", ws.gate, ws.gate, 0, qd * E}, {"k", ws.k, ws.k, 0, kvd * E},
                               {"v", ws.v, ws.qg, (2 * qd + kvd) * E, kvd * E}});
        std::vector<std::vector<uint8_t>> want;
        for (bool merge : {false, true}) {
            if (i == 0) {
                load("decode0/tokens", ws.tokens);
            } else {
                load(std::string("decode0/") + lp + "/dense/x", ws.x);
                load(std::string("decode0/") + lp + "/attn/o", ws.o);
            }
            for (auto& [name, sep, mb, off, n] : f) memset((uint8_t*)mb.contents + off, 0xFF, n), memset(sep.contents, 0xFF, n);
            ops.merge = merge;
            @autoreleasepool {
                mllm::encode_and_wait(q, ws, [&] { ops.dense(i, ws); });
            }
            for (size_t k = 0; k < f.size(); k++) {
                auto& [name, sep, mb, off, n] = f[k];
                const uint8_t* p = merge ? (const uint8_t*)mb.contents + off : (const uint8_t*)sep.contents;
                if (!merge) want.emplace_back(p, p + n);
                else if (memcmp(p, want[k].data(), n)) printf("L%02d %s: merged M = 1 differs\n", i, name.c_str()), ok = false;
            }
        }
        m1_checked++;
    }
    assert(ok && m1_checked == L + 1);
    printf("M = 1 merged == separate, %d dense calls\n", m1_checked);
    printf("PASS\n");
}
