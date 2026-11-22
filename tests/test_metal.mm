// Metal path end to end: load mllm.metallib (path from the Makefile), run add_one once.
#import <Metal/Metal.h>

#include <cassert>

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&err];
    if (!lib) fprintf(stderr, "%s\n", err.localizedDescription.UTF8String);
    assert(lib);
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_one"]
                                                                        error:&err];
    assert(ps);
    const int n = 1024;
    id<MTLBuffer> buf = [dev newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    float* x = (float*)buf.contents;
    for (int i = 0; i < n; i++) x[i] = i;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    for (int i = 0; i < n; i++) assert(x[i] == i + 1);
    printf("PASS\n");
}
