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
    std::vector<int> layers;
    for (int layer = 0; layer < n_layers(); layer++)
        if (kinds[layer] == kind) layers.push_back(layer);
    return layers;
}

namespace spec {

std::string resolve(const std::string& model) {
    if (fs::is_directory(model)) return model;
    const char *hub_cache = getenv("HF_HUB_CACHE"), *hf_home = getenv("HF_HOME"), *home = getenv("HOME");
    if (!hub_cache && !hf_home && !home) throw std::runtime_error(model + ": set HF_HOME (or HF_HUB_CACHE)");
    fs::path hub = hub_cache ? fs::path(hub_cache)
                             : (hf_home ? fs::path(hf_home) : fs::path(home) / ".cache/huggingface") / "hub";
    std::string repo_dir_name = "models--" + model;
    for (size_t slash; (slash = repo_dir_name.find('/')) != std::string::npos;)
        repo_dir_name.replace(slash, 1, "--");  // HF cache: org--name
    fs::path repo = hub / repo_dir_name;
    std::ifstream main_ref(repo / "refs/main");
    std::string snapshot;
    if (!(main_ref >> snapshot) || !fs::is_directory(repo / "snapshots" / snapshot))
        throw std::runtime_error(model + ": not a directory and not downloaded under " + repo.string() +
                                 " (run the Python engine once, or download it with HF_HOME set)");
    return (repo / "snapshots" / snapshot).string();
}

ModelSpec load(const std::string& model) {
    std::string path = resolve(model);
    std::ifstream config_file(fs::path(path) / "config.json");
    if (!config_file) throw std::runtime_error(path + ": no config.json");
    nlohmann::json config = nlohmann::json::parse(config_file);
    const nlohmann::json& text = config.contains("text_config") ? config["text_config"] : config;
    std::string model_type = text.at("model_type");
    if (model_type != "qwen3_5" && model_type != "qwen3_5_text")  // ponytail: one family until a second one runs
        throw std::runtime_error("unsupported model_type " + model_type);
    ModelSpec spec;
    spec.path = path;
    for (const auto& kind : text.at("layer_types")) spec.kinds.push_back(kind == "full_attention" ? "full" : "linear");
    spec.hidden = text.at("hidden_size");
    spec.vocab_size = text.at("vocab_size");
    spec.query_heads = text.at("num_attention_heads");
    spec.key_value_heads = text.at("num_key_value_heads");
    spec.head_dim = text.at("head_dim");
    spec.linear_key_heads = text.at("linear_num_key_heads");
    spec.linear_value_heads = text.at("linear_num_value_heads");
    spec.linear_key_dim = text.at("linear_key_head_dim");
    spec.linear_value_dim = text.at("linear_value_head_dim");
    spec.conv_kernel = text.at("linear_conv_kernel_dim");
    spec.mtp = fs::exists(weights_dir(path) + "/mtp/mtp.safetensors");
    return spec;
}

std::string cache_dir() {
    if (const char* cache = getenv("MLLM_CACHE")) return cache;
    if (fs::is_directory("/Volumes/T5")) return "/Volumes/T5/cache/mllm";
    const char* home = getenv("HOME");
    if (!home) throw std::runtime_error("set MLLM_CACHE");
    return std::string(home) + "/Library/Caches/mllm";
}

std::string weights_dir(const std::string& path) { return cache_dir() + "/weights/" + model_key(path) + "-v3"; }

std::string model_key(const std::string& path) {
    fs::path dir = fs::canonical(path);
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir))
        if (entry.path().extension() == ".safetensors") files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    files.push_back(dir / "config.json");
    std::string fingerprint = dir.string();
    for (const auto& file : files) {
        struct stat status;
        if (stat(file.c_str(), &status)) throw std::runtime_error(file.string() + ": stat failed");
        long long modified_ns = (long long)status.st_mtimespec.tv_sec * 1000000000LL + status.st_mtimespec.tv_nsec;
        fingerprint += file.filename().string() + ":" + std::to_string(status.st_size) + ":" + std::to_string(modified_ns);
    }
    unsigned char digest[CC_SHA1_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // SHA-1 names a cache dir; not security
    CC_SHA1(fingerprint.data(), (CC_LONG)fingerprint.size(), digest);
#pragma clang diagnostic pop
    char hex[13];
    for (int i = 0; i < 6; i++) snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    std::string name = dir.parent_path().filename() == "snapshots" ? dir.parent_path().parent_path().filename().string()
                                                                    : dir.filename().string();
    return name + "-" + hex;
}

}  // namespace spec
}  // namespace mllm
