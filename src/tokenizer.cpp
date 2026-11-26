#include "tokenizer.h"

#include <fstream>
#include <queue>
#include <llama-unicode/unicode.h>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace mllm {

// tokenizer.json's Split regex, and the same regex as llama.cpp spells it (LLAMA_VOCAB_PRE_TYPE_QWEN35), which
// routes unicode_regex_split to its hand-written qwen35 splitter.
static const char* HF_REGEX =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const char* LLAMA_REGEX =
    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
static const std::string REPLACEMENT_CHARACTER = "\xEF\xBF\xBD";  // U+FFFD

// A token's string -> its bytes: each GPT-2 byte-level character becomes its byte, anything else (added tokens'
// text) stays as UTF-8. mlx-lm decodes the character for byte 0 as its UTF-8 instead (a falsy-0 slip); we don't.
static std::string token_string_to_bytes(const std::string& token) {
    static const auto byte_of_character = [] {
        std::unordered_map<uint32_t, uint8_t> table;
        for (int byte = 0; byte < 256; byte++) table[unicode_cpts_from_utf8(unicode_byte_to_utf8(byte))[0]] = byte;
        return table;
    }();
    std::string out;
    for (uint32_t character : unicode_cpts_from_utf8(token)) {
        auto found = byte_of_character.find(character);
        out += found != byte_of_character.end() ? std::string(1, (char)found->second) : unicode_cpt_to_utf8(character);
    }
    return out;
}

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream file(path + "/tokenizer.json");
    if (!file) throw std::runtime_error(path + ": no tokenizer.json");
    nlohmann::json config = nlohmann::json::parse(file);
    const auto& model = config["model"];
    const auto& pre_tokenizer = config["pre_tokenizer"];
    // the one tokenizer this file implements; anything else would tokenize wrong without an error
    if (model["type"] != "BPE" || config["normalizer"]["type"] != "NFC" || pre_tokenizer["type"] != "Sequence" ||
        pre_tokenizer["pretokenizers"][0]["pattern"]["Regex"] != HF_REGEX ||
        pre_tokenizer["pretokenizers"][1]["type"] != "ByteLevel")
        throw std::runtime_error(path + ": tokenizer.json is not Qwen3.5's byte-level BPE");

    for (const auto& [token, id] : model["vocab"].items()) vocab.emplace(token, id.get<int>());
    int rank = 0;
    for (const auto& merge : model["merges"]) {  // "left right", or ["left", "right"]
        std::string left, right;
        if (merge.is_string()) {
            std::string both = merge.get<std::string>();
            size_t space = both.find(' ');
            left = both.substr(0, space), right = both.substr(space + 1);
        } else {
            left = merge[0].get<std::string>(), right = merge[1].get<std::string>();
        }
        merges.emplace((uint64_t)vocab.at(left) << 32 | (uint32_t)vocab.at(right), std::pair{rank++, vocab.at(left + right)});
    }
    for (const auto& token : config["added_tokens"])
        added_tokens.emplace_back(token["content"].get<std::string>(), token["id"].get<int>());

    auto set_bytes = [&](const std::string& token, int id) {
        bytes_of_token.resize(std::max<size_t>(bytes_of_token.size(), id + 1));
        bytes_of_token[id] = token_string_to_bytes(token);
    };
    for (const auto& [token, id] : vocab) set_bytes(token, id);
    for (const auto& [token, id] : added_tokens) set_bytes(token, id);
    space_token = vocab.at(unicode_byte_to_utf8(' '));

    std::ifstream config_file(path + "/tokenizer_config.json");
    std::string eos_text = nlohmann::json::parse(config_file)["eos_token"].get<std::string>();
    for (const auto& [token, id] : added_tokens)
        if (token == eos_text) eos.insert(id);
    if (eos.empty()) eos.insert(vocab.at(eos_text));
}

// Merge the lowest-ranked adjacent pair (leftmost on a tie) until none is left. Symbols form a linked list; a min-heap
// holds candidate pairs, stale ones are skipped when popped.
void Tokenizer::apply_merges(const std::string& word, std::vector<int>& out) const {
    struct Symbol {
        int id, previous, next;
    };
    std::vector<Symbol> symbols;
    for (size_t at = 0; at < word.size();) {
        size_t length = unicode_len_utf8(word[at]);
        symbols.push_back({vocab.at(word.substr(at, length)), (int)symbols.size() - 1, (int)symbols.size() + 1});
        at += length;
    }
    if (symbols.empty()) return;
    symbols.back().next = -1;
    struct Candidate {
        int rank, left, right, left_id, right_id, merged_id;  // left / right: symbol indexes
        bool operator>(const Candidate& o) const { return rank != o.rank ? rank > o.rank : left > o.left; }
    };
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> candidates;
    auto push_candidate = [&](int left, int right) {
        if (left < 0 || right < 0) return;
        auto found = merges.find((uint64_t)symbols[left].id << 32 | (uint32_t)symbols[right].id);
        if (found != merges.end())
            candidates.push({found->second.first, left, right, symbols[left].id, symbols[right].id, found->second.second});
    };
    for (int i = 0; i + 1 < (int)symbols.size(); i++) push_candidate(i, i + 1);
    while (!candidates.empty()) {
        Candidate pair = candidates.top();
        candidates.pop();
        Symbol &left = symbols[pair.left], &right = symbols[pair.right];
        // stale: either side was merged away or changed since this pair was pushed
        if (left.id != pair.left_id || right.id != pair.right_id || left.next != pair.right) continue;
        left.id = pair.merged_id, left.next = right.next, right.id = -1;
        if (left.next >= 0) symbols[left.next].previous = pair.left;
        push_candidate(left.previous, pair.left);
        push_candidate(pair.left, left.next);
    }
    for (int i = 0; i >= 0; i = symbols[i].next) out.push_back(symbols[i].id);
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
    if (text.empty()) return;
    std::string normalized;
    for (uint32_t character : unicode_cpts_normalize_nfc(unicode_cpts_from_utf8(text)))
        normalized += unicode_cpt_to_utf8(character);
    // ponytail: llama.cpp's NFC data predates Unicode 10, so marks assigned since then (e.g. U+0C3C) reorder
    // differently from HF's normalizer. If parity matters there, override their combining classes.
    for (const auto& word : unicode_regex_split(normalized, {LLAMA_REGEX})) apply_merges(word, out);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    size_t plain_start = 0;  // where the text since the last added token starts
    // added tokens first, longest match at the leftmost position (HF: aho-corasick, leftmost-longest)
    for (size_t at = 0; at < text.size();) {
        const std::pair<std::string, int>* longest = nullptr;
        for (const auto& token : added_tokens)
            if (text.compare(at, token.first.size(), token.first) == 0 &&
                (!longest || token.first.size() > longest->first.size()))
                longest = &token;
        if (!longest) {
            at++;
            continue;
        }
        encode_plain(text.substr(plain_start, at - plain_start), out);
        out.push_back(longest->second);
        plain_start = at += longest->first.size();
    }
    encode_plain(text.substr(plain_start), out);
    return out;
}

// Jinja's |trim = Python's str.strip(): the str.isspace() set.
static bool is_python_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static std::string trim(const std::string& text) {
    size_t first = text.size(), end = 0;  // the first non-space byte, and the end of the last non-space character
    for (size_t at = 0; at < text.size();) {
        size_t character_start = at;
        uint32_t character;
        try {
            character = unicode_cpt_from_utf8(text, at);
        } catch (const std::invalid_argument&) {  // invalid UTF-8: a non-space byte
            character = 0, at = character_start + 1;
        }
        if (!is_python_space(character)) first = std::min(first, character_start), end = at;
    }
    return first < end ? text.substr(first, end - first) : "";
}

// An assistant message's "...<think>reasoning</think>content": splits off the reasoning as the template does
static void split_off_reasoning(std::string& content, std::string& reasoning) {
    size_t think_end = content.find("</think>");
    if (think_end == std::string::npos) return;
    reasoning = content.substr(0, think_end);
    reasoning.erase(reasoning.find_last_not_of('\n') + 1);  // rstrip('\n')
    if (size_t think_start = reasoning.rfind("<think>"); think_start != std::string::npos)
        reasoning.erase(0, think_start + 7);
    reasoning.erase(0, reasoning.find_first_not_of('\n'));  // lstrip('\n'); npos clears
    content.erase(0, content.rfind("</think>") + 8);
    content.erase(0, content.find_first_not_of('\n'));
}

// qwen3_5's chat_template.jinja for text-only system/user/assistant messages, add_generation_prompt=True and
// enable_thinking=True (mlx-lm passes it for any tokenizer with think tokens), so generation opens "<think>\n". No tools.
// ponytail: hard-coded ChatML; a Jinja engine (llama.cpp common/jinja) comes with the second model family.
std::vector<int> Tokenizer::encode_chat(const std::vector<Message>& messages) const {
    if (messages.empty()) throw std::invalid_argument("No messages provided.");
    size_t last_query = messages.size();
    for (size_t i = messages.size(); i-- > 0;) {
        std::string content = trim(messages[i].content);
        if (messages[i].role == "user" &&
            !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"))) {
            last_query = i;
            break;
        }
    }
    if (last_query == messages.size()) throw std::invalid_argument("No user query found in messages.");

    std::string text;
    for (size_t i = 0; i < messages.size(); i++) {
        const std::string& role = messages[i].role;
        std::string content = trim(messages[i].content);
        if (role == "system") {
            if (i) throw std::invalid_argument("System message must be at the beginning.");
            text += "<|im_start|>system\n" + content + "<|im_end|>\n";
        } else if (role == "user") {
            text += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string reasoning;
            split_off_reasoning(content, reasoning);
            reasoning = trim(reasoning);
            text += i > last_query ? "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content
                                   : "<|im_start|>assistant\n" + content;
            text += "<|im_end|>\n";
        } else {
            throw std::invalid_argument("Unexpected message role: " + role);
        }
    }
    return encode(text + "<|im_start|>assistant\n<think>\n");
}

// encode_chat's end of an assistant turn, then the user turn and the generation prompt. The template drops past turns'
// reasoning, so re-templating a chat would never match what the prefix cache holds of the reply.
std::vector<int> Tokenizer::continue_chat(std::vector<int> tokens, const std::string& message) const {
    if (!tokens.empty() && eos.count(tokens.back())) tokens.pop_back();
    for (int token : encode("<|im_end|>\n")) tokens.push_back(token);
    for (int token : encode_chat({{"user", message}})) tokens.push_back(token);
    return tokens;
}

Detokenizer Tokenizer::detokenizer() const { return Detokenizer(*this); }

// bytes -> UTF-8 with each maximal invalid subsequence replaced by U+FFFD, as Python's decode("utf-8", "replace").
static std::string decode_replace(const std::string& bytes) {
    std::string out;
    for (size_t at = 0, size = bytes.size(); at < size;) {
        uint8_t lead = bytes[at];
        if (lead < 0x80) {
            out += (char)lead, at++;
            continue;
        }
        // continuation bytes the lead byte needs, and the range its first continuation byte must be in
        int continuations = lead >= 0xC2 && lead <= 0xDF ? 1 : lead >= 0xE0 && lead <= 0xEF ? 2 : lead >= 0xF0 && lead <= 0xF4 ? 3 : 0;
        uint8_t first_min = lead == 0xE0 ? 0xA0 : lead == 0xF0 ? 0x90 : 0x80;
        uint8_t first_max = lead == 0xED ? 0x9F : lead == 0xF4 ? 0x8F : 0xBF;
        size_t end = at + 1;
        for (int k = 0; k < continuations && end < size; k++, end++) {
            uint8_t next = bytes[end];
            if (next < (k ? 0x80 : first_min) || next > (k ? 0xBF : first_max)) break;
        }
        out += continuations && end - at == (size_t)continuations + 1 ? bytes.substr(at, end - at) : REPLACEMENT_CHARACTER;
        at = end;
    }
    return out;
}

// mlx-lm drops a leading space from the first text it emits.
std::string Detokenizer::emit(std::string text) {
    if (!started && !text.empty() && text[0] == ' ') text.erase(0, 1);
    started = started || !text.empty();
    unflushed.clear();
    return text;
}

std::string Detokenizer::push(int token) {
    // mlx-lm: ids past the tokenizer -> "!"
    unflushed += (unsigned)token < tokenizer->bytes_of_token.size() ? tokenizer->bytes_of_token[token] : "!";
    std::string text = decode_replace(unflushed);
    // incomplete UTF-8 waits for more bytes; a lone space token waits for the next token (mlx-lm)
    if (text.ends_with(REPLACEMENT_CHARACTER) || token == tokenizer->space_token) return "";
    return emit(text);
}

std::string Detokenizer::flush() { return emit(decode_replace(unflushed)); }

}  // namespace mllm
