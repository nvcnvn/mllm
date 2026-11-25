// golden.h check: every golden tensor matches itself; a copy with one element changed does not.
#include <cassert>
#include <vector>

#include "golden.h"

// A copy of the tensor with its middle element changed
static std::vector<uint8_t> changed_copy(const mllm::Tensor& tensor, const uint8_t* data) {
    std::vector<uint8_t> copy(data, data + tensor.nbytes);
    size_t middle = tensor.numel() / 2;
    if (tensor.dtype == "I32") ((int32_t*)copy.data())[middle] += 1;
    else if (tensor.dtype == "F32") ((float*)copy.data())[middle] += 1 + std::fabs(((float*)copy.data())[middle]);
    else if (tensor.dtype == "BF16") ((uint16_t*)copy.data())[middle] ^= 0x4000;  // flip the exponent's top bit
    else assert(!"unexpected golden dtype");
    return copy;
}

int main() {
    std::string directory = golden::dir(mllm::spec::load(golden::MODEL));
    int checked = 0;
    for (const char* file : {"tokens.safetensors", "trace.safetensors"}) {
        mllm::Safetensors tensors(directory + "/" + file);
        for (const auto& [name, tensor] : tensors.tensors) {
            const uint8_t* data = tensors.data(tensor);
            assert(golden::matches(name, tensor, data, data));
            if (!tensor.numel()) continue;
            std::vector<uint8_t> copy = changed_copy(tensor, data);
            golden::quiet = true;
            assert(!golden::matches(name, tensor, data, copy.data()));
            golden::quiet = false;
            checked++;
        }
    }
    printf("%d tensors: self match, perturbed copy fails\nPASS\n", checked);
}
