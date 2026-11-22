// Tokenizer + streaming Detokenizer (py/mllm/tokenizer.py): byte-level BPE from tokenizer.json, matching the
// HF tokenizer mlx-lm wraps. Pre-tokenizer split and NFC come from llama.cpp's unicode tables (src/vendor).
#pragma once
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace mllm {

struct Message {
    std::string role, content;  // system | user | assistant; text only
};

class Detokenizer;

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path);  // model dir with tokenizer.json, tokenizer_config.json

    std::vector<int> encode(const std::string& text) const;
    std::vector<int> encode_chat(const std::vector<Message>& messages) const;
    Detokenizer detokenizer() const;

    std::set<int> eos;

private:
    friend class Detokenizer;
    void bpe(const std::string& word, std::vector<int>& out) const;

    std::unordered_map<std::string, int> vocab;  // byte-encoded token -> id
    std::unordered_map<uint64_t, std::pair<int, int>> merges;  // (left id << 32 | right id) -> (rank, merged id)
    std::vector<std::pair<std::string, int>> added;  // added tokens (content, id), split out before BPE
    std::vector<std::string> bytes;  // id -> the bytes it decodes to
    int space = -1;  // id of the single-space token
};

// Holds back bytes until they form valid text. push(id) returns the text it completes; flush() the rest.
// Same output as mlx-lm's BPEStreamingDetokenizer, segment for segment.
class Detokenizer {
public:
    explicit Detokenizer(const Tokenizer& t) : t(&t) {}
    std::string push(int token);
    std::string flush();

private:
    std::string emit(std::string text);
    const Tokenizer* t;
    std::string unflushed;
    bool started = false;  // any text emitted yet
};

}  // namespace mllm
