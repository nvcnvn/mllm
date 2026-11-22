// Weights check: every tensor, bound at its offset in its no-copy MTLBuffer and copied out by the GPU,
// hashes (sha256) to what Python saw in the original files.
#include <CommonCrypto/CommonDigest.h>

#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "memory.h"

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::ifstream f(golden::dir(s) + "/weights.json");
    nlohmann::json want = nlohmann::json::parse(f);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    assert(w.st.tensors.size() == want.size());
    for (size_t i = 0; i < w.buffers.size(); i++) assert(w.buffers[i].contents == w.st.files[i].base);  // no copy

    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&err];
    id<MTLComputePipelineState> copy = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy_u32"]
                                                                          error:&err];
    assert(copy);
    size_t biggest = 0;
    for (const auto& [_, t] : w.st.tensors) biggest = std::max(biggest, t.nbytes);
    id<MTLBuffer> out = [dev newBufferWithLength:biggest + 4 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [dev newCommandQueue];

    for (const auto& [name, g] : want.items()) {
        auto v = w[name];
        assert(v.t.dtype == g["dtype"].get<std::string>() && v.t.shape == g["shape"].get<std::vector<int64_t>>());
        assert(v.offset % 16 == 0);
        // words past the tensor's end stay inside the page-rounded buffer; only nbytes are hashed
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:copy];
        [enc setBuffer:v.buffer offset:v.offset atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake((v.t.nbytes + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        unsigned char d[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256(out.contents, (CC_LONG)v.t.nbytes, d);
        char hex[65];
        for (int j = 0; j < 32; j++) snprintf(hex + 2 * j, 3, "%02x", d[j]);
        std::string want_hex = g["sha256"];
        if (want_hex != hex) fprintf(stderr, "%s: sha256 of the GPU read-back differs\n", name.c_str());
        assert(want_hex == hex);
    }
    printf("%zu tensors in %zu files: GPU read-back hashes equal\nPASS\n", want.size(), w.buffers.size());
}
