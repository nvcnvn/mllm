// mllm CLI. So far: print the model spec.
#include <cstdio>
#include <exception>

#include "spec.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: mllm MODEL   (repo id under $HF_HOME, or a local dir)\n");
        return 2;
    }
    try {
        mllm::ModelSpec s = mllm::spec::load(argv[1]);
        printf("%s\n%d layers (%zu full, %zu linear), hidden %d, vocab %d\n", s.path.c_str(), s.n_layers(),
               s.full_layers().size(), s.linear_layers().size(), s.hidden, s.vocab);
    } catch (const std::exception& e) {
        fprintf(stderr, "mllm: %s\n", e.what());
        return 1;
    }
}
