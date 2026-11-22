// golden.h check: every golden tensor matches itself; a copy with one element changed does not.
#include <cassert>
#include <vector>

#include "golden.h"

int main() {
    std::string d = golden::dir(mllm::spec::load(golden::MODEL));
    int n = 0;
    for (const char* f : {"tokens.safetensors", "trace.safetensors"}) {
        mllm::Safetensors st(d + "/" + f);
        for (const auto& [name, t] : st.tensors) {
            const uint8_t* p = st.data(t);
            assert(golden::matches(name, t, p, p));
            if (!t.numel()) continue;
            std::vector<uint8_t> c(p, p + t.nbytes);
            size_t i = t.numel() / 2;
            if (t.dtype == "I32") ((int32_t*)c.data())[i] += 1;
            else if (t.dtype == "F32") ((float*)c.data())[i] += 1 + std::fabs(((float*)c.data())[i]);
            else if (t.dtype == "BF16") ((uint16_t*)c.data())[i] ^= 0x4000;  // flip the exponent's top bit
            else assert(!"unexpected golden dtype");
            golden::quiet = true;
            assert(!golden::matches(name, t, p, c.data()));
            golden::quiet = false;
            n++;
        }
    }
    printf("%d tensors: self match, perturbed copy fails\nPASS\n", n);
}
