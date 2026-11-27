// ModelSpec: the shapes every other component is sized from. Plain data, read from config.json (py/mmmllm/spec.py).
#pragma once
#include <string>
#include <vector>

namespace mllm {

struct ModelSpec {
    std::string path;
    std::vector<std::string> kinds;  // per layer: "full" | "linear"
    int hidden, vocab_size;
    int query_heads, key_value_heads, head_dim;  // full attention
    int linear_key_heads, linear_value_heads, linear_key_dim, linear_value_dim, conv_kernel;  // Gated DeltaNet
    // The MTP draft head's weights: one more full-attention layer, its K/V in every page unit after the model's full layers
    // (docs/speculative.md). From tools/spec/mtp.py (weights_dir/mtp/mtp.safetensors), else $MLLM_DRAFT (the Python
    // wrapper downloads the head's HF repo and sets it); empty = no head, plain decode.
    std::string mtp_file;
    bool mtp = false;

    int n_layers() const { return (int)kinds.size(); }
    std::vector<int> full_layers() const { return layers_of("full"); }
    std::vector<int> linear_layers() const { return layers_of("linear"); }
    int conv_dim() const { return 2 * linear_key_heads * linear_key_dim + linear_value_heads * linear_value_dim; }

private:
    std::vector<int> layers_of(const char* kind) const;
};

namespace spec {
// Repo id or local dir -> local dir. A repo id must already be in $HF_HOME/hub; never downloads.
std::string resolve(const std::string& model);
ModelSpec load(const std::string& model);
// ane.py CACHE: $MLLM_CACHE, else /Volumes/T5/cache/mllm when T5 is mounted, else ~/Library/Caches/mllm.
std::string cache_dir();
// Mirrors ane.py _model_key (lives here until ane is ported): cache dir name unique to these weights (resolved path + size/mtime of config and weights).
std::string model_key(const std::string& path);
// <cache>/weights/<model_key>-v3: the weights as GpuOps reads them (memory.mm aligned_weight_files) and mtp/mtp.safetensors
std::string weights_dir(const std::string& path);
}  // namespace spec

}  // namespace mllm
