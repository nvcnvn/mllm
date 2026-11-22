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
static const std::string FFFD = "\xEF\xBF\xBD";

// A token's string -> its bytes: each GPT-2 byte-level character becomes its byte, anything else (added tokens'
// text) stays as UTF-8. mlx-lm decodes the character for byte 0 as its UTF-8 instead (a falsy-0 slip); we don't.
static std::string token_bytes(const std::string& s) {
    static const auto byte_of = [] {
        std::unordered_map<uint32_t, uint8_t> m;
        for (int b = 0; b < 256; b++) m[unicode_cpts_from_utf8(unicode_byte_to_utf8(b))[0]] = b;
        return m;
    }();
    std::string out;
    for (uint32_t c : unicode_cpts_from_utf8(s)) {
        auto it = byte_of.find(c);
        out += it != byte_of.end() ? std::string(1, (char)it->second) : unicode_cpt_to_utf8(c);
    }
    return out;
}

Tokenizer::Tokenizer(const std::string& path) {
    std::ifstream f(path + "/tokenizer.json");
    if (!f) throw std::runtime_error(path + ": no tokenizer.json");
    nlohmann::json j = nlohmann::json::parse(f);
    const auto& m = j["model"];
    const auto& pre = j["pre_tokenizer"];
    // the one tokenizer this file implements; anything else would tokenize wrong without an error
    if (m["type"] != "BPE" || j["normalizer"]["type"] != "NFC" || pre["type"] != "Sequence" ||
        pre["pretokenizers"][0]["pattern"]["Regex"] != HF_REGEX || pre["pretokenizers"][1]["type"] != "ByteLevel")
        throw std::runtime_error(path + ": tokenizer.json is not Qwen3.5's byte-level BPE");

    for (const auto& [k, v] : m["vocab"].items()) vocab.emplace(k, v.get<int>());
    int rank = 0;
    for (const auto& x : m["merges"]) {
        std::string l, r;
        if (x.is_string()) {
            std::string lr = x.get<std::string>();
            size_t sp = lr.find(' ');
            l = lr.substr(0, sp), r = lr.substr(sp + 1);
        } else {
            l = x[0].get<std::string>(), r = x[1].get<std::string>();
        }
        merges.emplace((uint64_t)vocab.at(l) << 32 | (uint32_t)vocab.at(r), std::pair{rank++, vocab.at(l + r)});
    }
    for (const auto& a : j["added_tokens"]) added.emplace_back(a["content"].get<std::string>(), a["id"].get<int>());

    for (const auto& [s, id] : vocab) bytes.resize(std::max<size_t>(bytes.size(), id + 1)), bytes[id] = token_bytes(s);
    for (const auto& [s, id] : added) bytes.resize(std::max<size_t>(bytes.size(), id + 1)), bytes[id] = token_bytes(s);
    space = vocab.at(unicode_byte_to_utf8(' '));

    std::ifstream fc(path + "/tokenizer_config.json");
    std::string eos_text = nlohmann::json::parse(fc)["eos_token"].get<std::string>();
    for (const auto& [s, id] : added)
        if (s == eos_text) eos.insert(id);
    if (eos.empty()) eos.insert(vocab.at(eos_text));
}

// One pre-token (byte-level string) -> ids: merge the lowest-ranked adjacent pair (leftmost on a tie) until none
// is left. Symbols form a linked list; a min-heap holds candidate pairs, stale ones are skipped when popped.
void Tokenizer::bpe(const std::string& word, std::vector<int>& out) const {
    struct Sym {
        int id, prev, next;
    };
    std::vector<Sym> sym;
    for (size_t i = 0; i < word.size();) {
        size_t n = unicode_len_utf8(word[i]);
        sym.push_back({vocab.at(word.substr(i, n)), (int)sym.size() - 1, (int)sym.size() + 1});
        i += n;
    }
    if (sym.empty()) return;
    sym.back().next = -1;
    struct Pair {
        int rank, left, right, lid, rid, merged;
        bool operator>(const Pair& o) const { return rank != o.rank ? rank > o.rank : left > o.left; }
    };
    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> heap;
    auto add = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        auto it = merges.find((uint64_t)sym[l].id << 32 | (uint32_t)sym[r].id);
        if (it != merges.end()) heap.push({it->second.first, l, r, sym[l].id, sym[r].id, it->second.second});
    };
    for (int i = 0; i + 1 < (int)sym.size(); i++) add(i, i + 1);
    while (!heap.empty()) {
        Pair p = heap.top();
        heap.pop();
        Sym &l = sym[p.left], &r = sym[p.right];
        // stale: either side was merged away or changed since this pair was pushed
        if (l.id != p.lid || r.id != p.rid || l.next != p.right) continue;
        l.id = p.merged, l.next = r.next, r.id = -1;
        if (l.next >= 0) sym[l.next].prev = p.left;
        add(l.prev, p.left);
        add(p.left, l.next);
    }
    for (int i = 0; i >= 0; i = sym[i].next) out.push_back(sym[i].id);
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    size_t start = 0;
    auto plain = [&](size_t end) {  // text[start:end] has no added token: NFC, split, BPE
        if (end == start) return;
        std::string s;
        for (uint32_t c : unicode_cpts_normalize_nfc(unicode_cpts_from_utf8(text.substr(start, end - start))))
            s += unicode_cpt_to_utf8(c);
        // ponytail: llama.cpp's NFC data predates Unicode 10, so marks assigned since then (e.g. U+0C3C) reorder
        // differently from HF's normalizer. If parity matters there, override their combining classes.
        for (const auto& w : unicode_regex_split(s, {LLAMA_REGEX})) bpe(w, out);
    };
    // added tokens first, longest match at the leftmost position (HF: aho-corasick, leftmost-longest)
    for (size_t i = 0; i < text.size();) {
        const std::pair<std::string, int>* hit = nullptr;
        for (const auto& a : added)
            if (text.compare(i, a.first.size(), a.first) == 0 && (!hit || a.first.size() > hit->first.size())) hit = &a;
        if (!hit) {
            i++;
            continue;
        }
        plain(i);
        out.push_back(hit->second);
        start = i += hit->first.size();
    }
    plain(text.size());
    return out;
}

// Jinja's |trim = Python's str.strip(): the str.isspace() set.
static bool py_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static std::string trim(const std::string& s) {
    size_t a = s.size(), b = 0;
    for (size_t i = 0; i < s.size();) {
        size_t at = i;
        uint32_t c;
        try {
            c = unicode_cpt_from_utf8(s, i);
        } catch (const std::invalid_argument&) {  // invalid UTF-8: a non-space byte
            c = 0, i = at + 1;
        }
        if (!py_space(c)) a = std::min(a, at), b = i;
    }
    return a < b ? s.substr(a, b - a) : "";
}

// qwen3_5's chat_template.jinja for text-only system/user/assistant messages, add_generation_prompt=True and
// enable_thinking=True (mlx-lm passes it for any tokenizer with think tokens), so generation opens "<think>\n". No tools.
// ponytail: hard-coded ChatML; a Jinja engine (llama.cpp common/jinja) comes with the second model family.
std::vector<int> Tokenizer::encode_chat(const std::vector<Message>& messages) const {
    if (messages.empty()) throw std::invalid_argument("No messages provided.");
    size_t last_query = messages.size();
    for (size_t i = messages.size(); i-- > 0;) {
        std::string c = trim(messages[i].content);
        if (messages[i].role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))) {
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
            if (size_t e = content.find("</think>"); e != std::string::npos) {
                reasoning = content.substr(0, e);
                reasoning.erase(reasoning.find_last_not_of('\n') + 1);  // rstrip('\n')
                if (size_t s = reasoning.rfind("<think>"); s != std::string::npos) reasoning.erase(0, s + 7);
                reasoning.erase(0, reasoning.find_first_not_of('\n'));  // lstrip('\n'); npos clears
                content.erase(0, content.rfind("</think>") + 8);
                content.erase(0, content.find_first_not_of('\n'));
            }
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

Detokenizer Tokenizer::detokenizer() const { return Detokenizer(*this); }

// bytes -> UTF-8 with each maximal invalid subsequence replaced by U+FFFD, as Python's decode("utf-8", "replace").
static std::string decode_replace(const std::string& b) {
    std::string out;
    for (size_t i = 0, n = b.size(); i < n;) {
        uint8_t c = b[i];
        if (c < 0x80) {
            out += (char)c, i++;
            continue;
        }
        int need = c >= 0xC2 && c <= 0xDF ? 1 : c >= 0xE0 && c <= 0xEF ? 2 : c >= 0xF0 && c <= 0xF4 ? 3 : 0;
        uint8_t lo = c == 0xE0 ? 0xA0 : c == 0xF0 ? 0x90 : 0x80, hi = c == 0xED ? 0x9F : c == 0xF4 ? 0x8F : 0xBF;
        size_t j = i + 1;
        for (int k = 0; k < need && j < n; k++, j++) {
            uint8_t d = b[j];
            if (d < (k ? 0x80 : lo) || d > (k ? 0xBF : hi)) break;
        }
        out += need && j - i == (size_t)need + 1 ? b.substr(i, j - i) : FFFD;
        i = j;
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
    unflushed += (unsigned)token < t->bytes.size() ? t->bytes[token] : "!";  // mlx-lm: ids past the tokenizer -> "!"
    std::string text = decode_replace(unflushed);
    // incomplete UTF-8 waits for more bytes; a lone space token waits for the next token (mlx-lm)
    if (text.ends_with(FFFD) || token == t->space) return "";
    return emit(text);
}

std::string Detokenizer::flush() { return emit(decode_replace(unflushed)); }

}  // namespace mllm
