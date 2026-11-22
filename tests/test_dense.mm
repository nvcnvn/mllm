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
#include "gpu.h"

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 64 << 20);
    int chunk = g["chunk"].get<int>(), max_batch = 2;
    mllm::Workspace ws = mem.workspace(chunk, max_batch);
    mllm::GpuOps ops(mem, w, std::max(chunk, max_batch), 64);
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
                ops.begin();
                ops.dense(i, ws);
                ops.commit_and_wait();
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
    printf("PASS\n");
}
