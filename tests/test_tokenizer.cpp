// tokenizer check: encode == golden for every string, encode_chat == golden for every chat, and streaming
// golden token sequences through a Detokenizer gives Python's segments byte for byte (push per id, then flush).
#include <cassert>
#include <chrono>
#include <fstream>
#include <random>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "tokenizer.h"

static std::vector<int> golden_ids(const mllm::Safetensors& tokens, const std::string& name) {
    const mllm::Tensor& tensor = tokens.at(name);
    assert(tensor.dtype == "I32");
    std::vector<int> ids(tensor.numel());
    for (size_t i = 0; i < ids.size(); i++) ids[i] = golden::load<int32_t>(tokens.data(tensor), i);
    return ids;
}

static bool same(const std::string& what, const std::vector<int>& got, const std::vector<int>& want) {
    if (got == want) return true;
    size_t i = 0;
    while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
    fprintf(stderr, "%s: %zu vs %zu ids, first diff at %zu: %d vs %d\n", what.c_str(), got.size(), want.size(), i,
            i < got.size() ? got[i] : -1, i < want.size() ? want[i] : -1);
    return false;
}

// Every golden string: encode == golden "str/NN"
static bool check_strings(const mllm::Tokenizer& tokenizer, const nlohmann::json& strings,
                          const mllm::Safetensors& tokens) {
    bool ok = true;
    for (size_t i = 0; i < strings.size(); i++) {
        char key[16];
        snprintf(key, sizeof key, "str/%02zu", i);
        ok &= same(key, tokenizer.encode(strings[i].get<std::string>()), golden_ids(tokens, key));
    }
    return ok;
}

// Every golden chat: encode_chat == golden "chat/N"
static bool check_chats(const mllm::Tokenizer& tokenizer, const nlohmann::json& chats, const mllm::Safetensors& tokens) {
    bool ok = true;
    for (size_t i = 0; i < chats.size(); i++) {
        std::vector<mllm::Message> messages;
        for (const auto& message : chats[i])
            messages.push_back({message["role"].get<std::string>(), message["content"].get<std::string>()});
        std::string key = "chat/" + std::to_string(i);
        ok &= same(key, tokenizer.encode_chat(messages), golden_ids(tokens, key));
    }
    return ok;
}

// One golden sequence through a Detokenizer: each push, then the flush, gives the next golden segment.
// held counts the pushes that returned nothing yet.
static bool check_detokenizer_sequence(const mllm::Tokenizer& tokenizer, const nlohmann::json& sequence,
                                       size_t sequence_number, size_t& held) {
    auto ids = sequence["ids"].get<std::vector<int>>();
    auto want = sequence["segments"].get<std::vector<std::string>>();
    assert(want.size() == ids.size() + 1);
    mllm::Detokenizer detokenizer = tokenizer.detokenizer();
    for (size_t i = 0; i <= ids.size(); i++) {
        std::string got = i < ids.size() ? detokenizer.push(ids[i]) : detokenizer.flush();
        if (got != want[i]) {
            fprintf(stderr, "detok %zu: segment %zu = '%s', want '%s'\n", sequence_number, i, got.c_str(),
                    want[i].c_str());
            return false;
        }
        held += got.empty();
    }
    return true;
}

// BPE is n log n in a pre-token's length: 32K random letters (one pre-token) encode in well under 100 ms
static double check_long_word_speed(const mllm::Tokenizer& tokenizer) {
    std::mt19937 random(1);
    std::string letters(32768, 'a');
    for (char& letter : letters) letter = 'a' + random() % 26;
    auto start = std::chrono::steady_clock::now();
    size_t id_count = tokenizer.encode(letters).size();
    double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    assert(id_count > 0 && milliseconds < 100);
    return milliseconds;
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string directory = golden::dir(model);
    std::ifstream file(directory + "/spec.json");
    nlohmann::json want = nlohmann::json::parse(file);
    mllm::Safetensors tokens(directory + "/tokens.safetensors");
    mllm::Tokenizer tokenizer(model.path);
    assert(tokenizer.eos == want["eos"].get<std::set<int>>());

    bool ok = check_strings(tokenizer, want["strings"], tokens);
    ok &= check_chats(tokenizer, want["chat"], tokens);
    size_t held = 0, sequences = 0;
    for (const auto& sequence : want["detok"]) ok &= check_detokenizer_sequence(tokenizer, sequence, sequences++, held);
    assert(ok);

    double milliseconds = check_long_word_speed(tokenizer);
    mllm::Detokenizer bad = tokenizer.detokenizer();
    assert(bad.push(-5) == "!");  // out-of-range ids (negative too) decode as "!", never read out of bounds

    printf("32K letters in %.1f ms\n", milliseconds);
    printf("%zu strings, %zu chats, %zu detok sequences (%zu held-back pushes) equal golden\nPASS\n",
           want["strings"].size(), want["chat"].size(), sequences, held);
}
