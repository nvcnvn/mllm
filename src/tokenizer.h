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

    std::set<int> eos;  // end-of-sequence token ids

private:
    friend class Detokenizer;
    // One pre-token (byte-level string) -> its token ids, appended to out
    void apply_merges(const std::string& word, std::vector<int>& out) const;
    // Text without added tokens -> token ids, appended to out: NFC, the pre-tokenizer split, then apply_merges
    void encode_plain(const std::string& text, std::vector<int>& out) const;

    std::unordered_map<std::string, int> vocab;  // byte-encoded token -> id
    // (left id << 32 | right id) -> (rank, merged id)
    std::unordered_map<uint64_t, std::pair<int, int>> merges;
    std::vector<std::pair<std::string, int>> added_tokens;  // (content, id), split out before BPE
    std::vector<std::string> bytes_of_token;                // id -> the bytes it decodes to
    int space_token = -1;                                   // id of the single-space token
};

// Holds back bytes until they form valid text. push(id) returns the text it completes; flush() the rest.
// Same output as mlx-lm's BPEStreamingDetokenizer, segment for segment.
class Detokenizer {
public:
    explicit Detokenizer(const Tokenizer& tokenizer) : tokenizer(&tokenizer) {}
    std::string push(int token);
    std::string flush();

private:
    std::string emit(std::string text);
    const Tokenizer* tokenizer;
    std::string unflushed;
    bool started = false;  // any text emitted yet
};

}  // namespace mllm
