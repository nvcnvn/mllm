// tokenizer check: encode == golden for every string, encode_chat == golden for every chat, and streaming
// golden token sequences through a Detokenizer gives Python's segments byte for byte (push per id, then flush).
#include <cassert>
#include <chrono>
#include <fstream>
#include <random>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "tokenizer.h"

static std::vector<int> golden_ids(const mllm::Safetensors& st, const std::string& name) {
    const mllm::Tensor& t = st.at(name);
    assert(t.dtype == "I32");
    std::vector<int> v(t.numel());
    for (size_t i = 0; i < v.size(); i++) v[i] = golden::load<int32_t>(st.data(t), i);
    return v;
}

static bool same(const std::string& what, const std::vector<int>& got, const std::vector<int>& want) {
    if (got == want) return true;
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
    fprintf(stderr, "%s: %zu vs %zu ids, first diff at %zu: %d vs %d\n", what.c_str(), got.size(), want.size(), i,
            i < got.size() ? got[i] : -1, i < want.size() ? want[i] : -1);
    return false;
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    std::ifstream f(d + "/spec.json");
    nlohmann::json g = nlohmann::json::parse(f);
    mllm::Safetensors st(d + "/tokens.safetensors");
    mllm::Tokenizer tok(s.path);
    assert(tok.eos == g["eos"].get<std::set<int>>());

    bool ok = true;
    const auto& strings = g["strings"];
    for (size_t i = 0; i < strings.size(); i++) {
        char key[16];
        snprintf(key, sizeof key, "str/%02zu", i);
        ok &= same(key, tok.encode(strings[i].get<std::string>()), golden_ids(st, key));
    }
    const auto& chats = g["chat"];
    for (size_t i = 0; i < chats.size(); i++) {
        std::vector<mllm::Message> msgs;
        for (const auto& m : chats[i]) msgs.push_back({m["role"].get<std::string>(), m["content"].get<std::string>()});
        std::string key = "chat/" + std::to_string(i);
        ok &= same(key, tok.encode_chat(msgs), golden_ids(st, key));
    }

    size_t held = 0, n_segments = 0;
    for (const auto& c : g["detok"]) {
        auto ids = c["ids"].get<std::vector<int>>();
        auto want = c["segments"].get<std::vector<std::string>>();
        assert(want.size() == ids.size() + 1);
        mllm::Detokenizer dt = tok.detokenizer();
        for (size_t i = 0; i <= ids.size(); i++) {
            std::string got = i < ids.size() ? dt.push(ids[i]) : dt.flush();
            if (got != want[i]) {
                fprintf(stderr, "detok %zu: segment %zu = '%s', want '%s'\n", n_segments, i, got.c_str(), want[i].c_str());
                ok = false;
                break;
            }
            held += got.empty();
        }
        n_segments++;
    }
    assert(ok);

    // BPE is n log n in a pre-token's length: 32K random letters (one pre-token) encode in well under 100 ms
    std::mt19937 rng(1);
    std::string letters(32768, 'a');
    for (char& c : letters) c = 'a' + rng() % 26;
    auto t0 = std::chrono::steady_clock::now();
    size_t n_ids = tok.encode(letters).size();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    assert(n_ids > 0 && ms < 100);
    mllm::Detokenizer bad = tok.detokenizer();
    assert(bad.push(-5) == "!");  // out-of-range ids (negative too) decode as "!", never read out of bounds

    printf("32K letters in %.1f ms\n", ms);
    printf("%zu strings, %zu chats, %zu detok sequences (%zu held-back pushes) equal golden\nPASS\n", strings.size(),
           chats.size(), n_segments, held);
}
