// Metal path end to end: load mllm.metallib (path from the Makefile), run add_one once.
#import <Metal/Metal.h>

#include <cassert>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&error];
    if (!library) fprintf(stderr, "%s\n", error.localizedDescription.UTF8String);
    assert(library);
    id<MTLComputePipelineState> add_one =
        [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"add_one"] error:&error];
    assert(add_one);
    const int count = 1024;
    id<MTLBuffer> buffer = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
    float* values = (float*)buffer.contents;
    for (int i = 0; i < count; i++) values[i] = i;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:add_one];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    for (int i = 0; i < count; i++) assert(values[i] == i + 1);
    printf("PASS\n");
}
