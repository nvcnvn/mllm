// Proves the metallib builds and loads (tests/test_metal.mm); copy_u32 reads weights back (tests/test_weights.mm).
// Real kernels arrive with gpu.
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] += 1.0f;
}

kernel void copy_u32(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}
