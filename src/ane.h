// AneOps: dense(i) as one Core ML model per dense block, run on the Neural Engine (py/mllm/ane.py). Prefill only:
// each block is compiled for a static chunk of S tokens; a shorter chunk is zero-padded. The models are built and
// cached by Python only (ane.py _build): <cache>/ane/<model_key>/S<S>_<weights>/dense_<iii>/{model.mlmodelc, io.json}.
//
// Every block input and output is an IOSurface (Memory::surface), fp16 channels-first (1, C, 1, S) (q / k:
// (1, H, D, S), the same bytes as [C = H * D, S]). Two GPU kernels move data between them and the workspace's
// row-major bf16 [n, C]; GPU attn then runs on the n valid tokens (ws.S = n) as usual. The residual x goes through
// ws.x between blocks too, rounded to bf16 as in ane.py: keeping it fp16 on the ANE side drifted further from the
// GPU path (worst state cos 0.99868 vs 0.99884 on golden chat/0; ane.py: 0.99884).
//
// weights "shared" (target-27b.md §5): no weight copy of the ANE's own. One model per block kind (previous layer's
// kind, this layer's) and chunk size (ane.py build_shared: <cache>/ane/<model_key>/shared_S<S>/<kind>/), every
// linear weight and norm gamma an input. inputs(i) also dequantizes block i's 4-bit weights from the GPU's copy into
// fp16 weight surfaces (one set per chunk in flight); layout row-major [S, C], the workspace's. S is the largest
// chunk size (512 and S are built); a chunk runs on the smallest that holds it.
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
    // builds them, if a block is missing. weights: fp16, int8, lut6, lut4 (baked into per-layer blocks) or shared.
    AneOps(GpuOps& gpu, Memory& mem, int S, const std::string& weights = "shared");

    // set: which of the 2 surface sets (two chunks in flight, one set each; PrefillLane uses chunk parity).
    // Encode into ws.enc: block i's inputs (ws.x, and ws.o for i > 0) into its surfaces, zero past ws.S tokens.
    // i = 0 first embeds ws.tokens into ws.x.
    void inputs(int i, Workspace& ws, int set);
    // Runs block i on the ANE (blocks the thread until done; its inputs must be on the surfaces, i.e. the command
    // buffer that encoded inputs(i) completed), then encodes into ws.enc its outputs' first ws.S tokens -> ws.x and
    // ws.q / k / v / gate or ws.qkv / z / b / a.
    void dense(int i, Workspace& ws, int set);

    struct Surface {
        Memory::Surface s;
        MLMultiArray* array;  // over s.pixel_buffer, in the model's shape
        int C;                // channels (rows of the surface)
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
    int S;
    std::string root;
    std::vector<Block> blocks;  // one per layer: dense(L) never runs in prefill (shared: empty)
    // shared: the blocks by (S, kind), the compiled sizes (ascending), and per layer what inputs(i) writes into the
    // weight surfaces: a 4-bit weight (w) or a norm's bf16 gamma of dim values
    bool shared = false;
    std::map<std::pair<int, std::string>, Block> kinds;
    std::vector<int> sizes;
    struct Param {
        const QLinear* w;
        Buf gamma;
        int dim;
        Surface* surface[2];
    };
    std::vector<std::vector<Param>> params;
    std::map<std::string, Surface> surfaces[2];  // by name + channels, shared by the blocks (x_out: x as an output)

    // A copy of the engine's GpuOps (same pipelines and weights, no allocation) for the ANE lane's thread: GpuOps keeps
    // the encoder it is encoding into in members, so decode on the engine thread and prefill here can't share one.
    GpuOps gpu;

private:
    id<MTLComputePipelineState> from_ane_ps, to_ane_ps, from_rows_ps, to_rows_ps, dequant_ps;
    std::vector<std::string> kind_of;  // shared: layer i's block kind
    void load_shared(Memory& mem);
    const Block& block(int i, int n) const;  // the block that runs layer i's dense on a chunk of n tokens
};

}  // namespace mllm
