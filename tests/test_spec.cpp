// spec check: every field equals the golden spec (written by py/dump_golden.py from spec.py).
#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>

#include "golden.h"

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::ifstream f(golden::dir(s) + "/spec.json");
    nlohmann::json g = nlohmann::json::parse(f)["spec"];
    assert(g.size() == 12);  // a new spec.py field needs a line here
    assert(s.path == g["path"].get<std::string>());
    assert(s.kinds == g["kinds"].get<std::vector<std::string>>());
    assert(s.hidden == g["hidden"].get<int>() && s.vocab == g["vocab"].get<int>());
    assert(s.n_heads == g["n_heads"].get<int>() && s.n_kv_heads == g["n_kv_heads"].get<int>() && s.head_dim == g["head_dim"].get<int>());
    assert(s.lin_k_heads == g["lin_k_heads"].get<int>() && s.lin_v_heads == g["lin_v_heads"].get<int>());
    assert(s.lin_k_dim == g["lin_k_dim"].get<int>() && s.lin_v_dim == g["lin_v_dim"].get<int>() && s.conv_kernel == g["conv_kernel"].get<int>());
    assert(s.full_layers().size() + s.linear_layers().size() == (size_t)s.n_layers());

    assert(mllm::spec::resolve(s.path) == s.path);  // a directory is used as-is
    try {
        mllm::spec::resolve("no-such-org/no-such-model");
        assert(!"resolve of a missing model must throw");
    } catch (const std::runtime_error&) {
    }
    printf("PASS\n");
}
