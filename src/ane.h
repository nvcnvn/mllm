// AneOps: dense(i) on the Neural Engine with no weight copy of its own (target-27b.md §5). Prefill only.
//
// One Core ML model per block kind (the previous layer's kind, this layer's: none_linear, linear_full, ...) and chunk
// size, built by Python (ane.py build: <cache>/ane/<model_key>/shared_S<S>/<kind>/{model.mlmodelc, io.json}). Every
// linear weight and norm gamma is a model input: inputs(i) dequantizes block i's 4-bit weights from the GPU's copy into
// fp16 weight surfaces (one set per chunk in flight, ~0.9 GB a set at 27B) next to block i's activations. Every input
// and output is an IOSurface (Memory::surface), fp16 row-major [rows, cols] like the workspace's bf16 [n, C]. S is the
// largest chunk size (512 and S are built); a chunk runs on the smallest that holds it, zero-padded.
//
// The residual x goes through ws.x (bf16) between blocks: keeping it fp16 on the ANE side drifted further from the GPU
// path (worst state cos 0.99868 vs 0.99884 on golden chat/0 with the earlier baked-weight blocks).
#pragma once
#import <CoreML/CoreML.h>

#include <map>
#include <string>
#include <vector>

#include "gpu.h"

namespace mllm {

class AneOps {
public:
    // Loads every block (compute units CPU + NE) and allocates its surfaces. Throws, naming the Python command that
    // builds them, if a block is missing.
    AneOps(GpuOps& gpu, Memory& mem, int S);

    // set: which of the 2 surface sets (two chunks in flight, one set each; PrefillLane uses chunk parity).
    // Encode into ws.enc: block i's inputs (ws.x, and ws.o for i > 0) into its surfaces, zero past ws.S tokens, and
    // its weights and gammas into the set's weight surfaces. i = 0 first embeds ws.tokens into ws.x.
    void inputs(int i, Workspace& ws, int set);
    // Runs block i on the ANE (blocks the thread until done; its inputs must be on the surfaces, i.e. the command
    // buffer that encoded inputs(i) completed), then encodes into ws.enc its outputs' first ws.S tokens -> ws.x and
    // ws.q / k / v / gate or ws.qkv / z / b / a.
    void dense(int i, Workspace& ws, int set);

    struct Surface {
        Memory::Surface s;
        MLMultiArray* array;  // over s.pixel_buffer, [rows, cols]
        int C;                // cols
    };
    struct Io {
        id<MTLBuffer> Workspace::*field;
        Surface* surface[2];  // by set
    };
    struct Block {
        int S;  // compiled chunk size
        MLModel* model;
        MLDictionaryFeatureProvider* in[2];  // by set
        MLPredictionOptions* opts[2];        // outputBackings: every output into its surface
        std::vector<Io> ins, outs;           // the ones converted from / to the workspace
    };
    // What inputs(i) writes into a weight surface: a 4-bit weight (w), or a norm's bf16 gamma of dim values
    struct Param {
        const QLinear* w;
        Buf gamma;
        int dim;
        Surface* surface[2];
    };
    int S;
    std::vector<int> sizes;  // compiled chunk sizes, ascending
    std::string root;
    std::map<std::pair<int, std::string>, Block> blocks;  // by (S, kind)
    std::vector<std::vector<Param>> params;               // per layer
    // activations by name and S, weights / gammas by name and shape (shared by every S)
    std::map<std::string, Surface> surfaces[2];
    const Block& block(int i, int n) const;  // the block that runs layer i's dense on a chunk of n tokens

    // A copy of the engine's GpuOps (same pipelines and weights, no allocation) for the ANE lane's thread: GpuOps keeps
    // the encoder it is encoding into in members, so decode on the engine thread and prefill here can't share one.
    GpuOps gpu;

private:
    id<MTLComputePipelineState> from_rows_ps, to_rows_ps, dequant_ps;
    std::vector<std::string> kind_of;  // layer i's block kind
};

}  // namespace mllm
