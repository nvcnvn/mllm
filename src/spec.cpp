#include "spec.h"

#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mllm {

std::vector<int> ModelSpec::layers_of(const char* kind) const {
    std::vector<int> out;
    for (int i = 0; i < n_layers(); i++)
        if (kinds[i] == kind) out.push_back(i);
    return out;
}

namespace spec {

std::string resolve(const std::string& model) {
    if (fs::is_directory(model)) return model;
    const char *hub = getenv("HF_HUB_CACHE"), *hf = getenv("HF_HOME"), *home = getenv("HOME");
    if (!hub && !hf && !home) throw std::runtime_error(model + ": set HF_HOME (or HF_HUB_CACHE)");
    fs::path root = hub ? fs::path(hub) : (hf ? fs::path(hf) : fs::path(home) / ".cache/huggingface") / "hub";
    std::string dir = "models--" + model;
    for (size_t i; (i = dir.find('/')) != std::string::npos;) dir.replace(i, 1, "--");  // HF cache: org--name
    fs::path repo = root / dir;
    std::ifstream ref(repo / "refs/main");
    std::string hash;
    if (!(ref >> hash) || !fs::is_directory(repo / "snapshots" / hash))
        throw std::runtime_error(model + ": not a directory and not downloaded under " + repo.string() +
                                 " (run the Python engine once, or download it with HF_HOME set)");
    return (repo / "snapshots" / hash).string();
}

ModelSpec load(const std::string& model) {
    std::string path = resolve(model);
    std::ifstream f(fs::path(path) / "config.json");
    if (!f) throw std::runtime_error(path + ": no config.json");
    nlohmann::json c = nlohmann::json::parse(f);
    const nlohmann::json& t = c.contains("text_config") ? c["text_config"] : c;
    std::string mt = t.at("model_type");
    if (mt != "qwen3_5" && mt != "qwen3_5_text")  // ponytail: one family until a second one runs
        throw std::runtime_error("unsupported model_type " + mt);
    ModelSpec s;
    s.path = path;
    for (const auto& k : t.at("layer_types")) s.kinds.push_back(k == "full_attention" ? "full" : "linear");
    s.hidden = t.at("hidden_size");
    s.vocab = t.at("vocab_size");
    s.n_heads = t.at("num_attention_heads");
    s.n_kv_heads = t.at("num_key_value_heads");
    s.head_dim = t.at("head_dim");
    s.lin_k_heads = t.at("linear_num_key_heads");
    s.lin_v_heads = t.at("linear_num_value_heads");
    s.lin_k_dim = t.at("linear_key_head_dim");
    s.lin_v_dim = t.at("linear_value_head_dim");
    s.conv_kernel = t.at("linear_conv_kernel_dim");
    return s;
}

std::string cache_dir() {
    if (const char* c = getenv("MLLM_CACHE")) return c;
    if (fs::is_directory("/Volumes/T5")) return "/Volumes/T5/cache/mllm";
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("set MLLM_CACHE");
    return std::string(home) + "/Library/Caches/mllm";
}

std::string model_key(const std::string& path) {
    fs::path p = fs::canonical(path);
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(p))
        if (e.path().extension() == ".safetensors") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    files.push_back(p / "config.json");
    std::string h = p.string();
    for (const auto& f : files) {
        struct stat st;
        if (stat(f.c_str(), &st)) throw std::runtime_error(f.string() + ": stat failed");
        long long ns = (long long)st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
        h += f.filename().string() + ":" + std::to_string(st.st_size) + ":" + std::to_string(ns);
    }
    unsigned char d[CC_SHA1_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // SHA-1 names a cache dir; not security
    CC_SHA1(h.data(), (CC_LONG)h.size(), d);
#pragma clang diagnostic pop
    char hex[13];
    for (int i = 0; i < 6; i++) snprintf(hex + 2 * i, 3, "%02x", d[i]);
    std::string name = p.parent_path().filename() == "snapshots" ? p.parent_path().parent_path().filename().string()
                                                                  : p.filename().string();
    return name + "-" + hex;
}

}  // namespace spec
}  // namespace mllm
