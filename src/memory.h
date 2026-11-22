// Memory: the only component that allocates (py/mllm/memory.py). So far: Weights.
#pragma once
#import <Metal/Metal.h>

#include <string>
#include <vector>

#include "safetensors.h"

namespace mllm {

// The model's safetensors (or header-padded copies, see memory.mm), mmapped; each file wrapped once as a no-copy MTLBuffer. A tensor = buffer + offset.
struct Weights {
    Weights(const std::string& path, id<MTLDevice> device);

    struct View {
        id<MTLBuffer> buffer;
        size_t offset;
        const Tensor& t;
    };
    View operator[](const std::string& name) const;

    Safetensors st;
    std::vector<id<MTLBuffer>> buffers;  // one per st.files
};

}  // namespace mllm
