// spec check: every field equals the golden spec (written by py/dump_golden.py from spec.py, under its short names).
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>

#include "golden.h"

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::ifstream file(golden::dir(model) + "/spec.json");
    nlohmann::json want = nlohmann::json::parse(file)["spec"];
    assert(want.size() == 12);  // a new spec.py field needs a line here
    assert(model.path == want["path"].get<std::string>());
    assert(model.kinds == want["kinds"].get<std::vector<std::string>>());
    assert(model.hidden == want["hidden"].get<int>() && model.vocab_size == want["vocab"].get<int>());
    assert(model.query_heads == want["n_heads"].get<int>() && model.key_value_heads == want["n_kv_heads"].get<int>() &&
           model.head_dim == want["head_dim"].get<int>());
    assert(model.linear_key_heads == want["lin_k_heads"].get<int>() &&
           model.linear_value_heads == want["lin_v_heads"].get<int>());
    assert(model.linear_key_dim == want["lin_k_dim"].get<int>() && model.linear_value_dim == want["lin_v_dim"].get<int>() &&
           model.conv_kernel == want["conv_kernel"].get<int>());
    assert(model.full_layers().size() + model.linear_layers().size() == (size_t)model.n_layers());

    assert(mllm::spec::resolve(model.path) == model.path);  // a directory is used as-is
    try {
        mllm::spec::resolve("no-such-org/no-such-model");
        assert(!"resolve of a missing model must throw");
    } catch (const std::runtime_error&) {
    }
    printf("PASS\n");
}
