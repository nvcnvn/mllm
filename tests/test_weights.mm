// Weights check: every tensor, bound at its offset in its no-copy MTLBuffer and copied out by the GPU,
// hashes (sha256) to what Python saw in the original files.
#include <CommonCrypto/CommonDigest.h>

#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "memory.h"

// The tensor copied by the GPU into output (copy_u32 from its buffer and offset), waited for
static void copy_on_gpu(id<MTLCommandQueue> queue, id<MTLComputePipelineState> copy, const mllm::Weights::View& tensor,
                        id<MTLBuffer> output) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:copy];
    [encoder setBuffer:tensor.buffer offset:tensor.offset atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder dispatchThreads:MTLSizeMake((tensor.t.nbytes + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

static std::string sha256_hex(const void* data, size_t bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, (CC_LONG)bytes, digest);
    char hex[65];
    for (int j = 0; j < 32; j++) snprintf(hex + 2 * j, 3, "%02x", digest[j]);
    return hex;
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::ifstream file(golden::dir(model) + "/weights.json");
    nlohmann::json want = nlohmann::json::parse(file);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    size_t language_model_tensors = 0;  // the weights cache keeps language_model.* only (memory.mm aligned)
    for (const auto& [name, _] : want.items()) language_model_tensors += name.starts_with("language_model.");
    assert(weights.st.tensors.size() == language_model_tensors);
    for (size_t i = 0; i < weights.buffers.size(); i++)
        assert(weights.buffers[i].contents == weights.st.files[i].base);  // no copy

    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&error];
    id<MTLComputePipelineState> copy =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"copy_u32"] error:&error];
    assert(copy);
    size_t biggest = 0;
    for (const auto& [_, tensor] : weights.st.tensors) biggest = std::max(biggest, tensor.nbytes);
    id<MTLBuffer> output = [device newBufferWithLength:biggest + 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    for (const auto& [name, expected] : want.items()) {
        if (!name.starts_with("language_model.")) continue;
        auto tensor = weights[name];
        assert(tensor.t.dtype == expected["dtype"].get<std::string>() &&
               tensor.t.shape == expected["shape"].get<std::vector<int64_t>>());
        assert(tensor.offset % 16 == 0);
        // words past the tensor's end stay inside the page-rounded buffer; only nbytes are hashed
        copy_on_gpu(queue, copy, tensor, output);
        std::string hex = sha256_hex(output.contents, tensor.t.nbytes);
        std::string want_hex = expected["sha256"];
        if (want_hex != hex) fprintf(stderr, "%s: sha256 of the GPU read-back differs\n", name.c_str());
        assert(want_hex == hex);
    }
    printf("%zu tensors in %zu files: GPU read-back hashes equal (%zu others skipped)\nPASS\n", language_model_tensors,
           weights.buffers.size(), want.size() - language_model_tensors);
}
