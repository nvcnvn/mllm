// AneOps: GpuOps::apply_weights on the Neural Engine with no weight copy of its own (target-27b.md §5). Prefill only.
//
// One Core ML model ("block") per block kind (the previous layer's kind, this layer's: none_linear, linear_full, ...)
// and chunk size, built by Python (ane.py build: <cache>/ane/<model_key>/shared_S<chunk size>/<kind>/{model.mlmodelc,
// io.json}). Every weight matrix and norm scale is a model input: prepare_inputs(layer) dequantizes that layer's block's
// 4-bit weights from the GPU's copy into fp16 weight surfaces (one set per chunk in flight, ~0.9 GB a set at 27B) next
// to the block's activations. Every input and output is an IOSurface (Memory::surface), fp16 row-major [rows, cols]
// like the workspace's bf16 [tokens, cols]. S is the largest chunk size (512 and S are built); a chunk runs on the
// smallest block that holds it, zero-padded.
//
// The residual goes through ws.residual (bf16) between blocks: keeping it fp16 on the ANE side drifted further from the
// GPU path (worst state cos 0.99868 vs 0.99884 on golden chat/0 with the earlier baked-weight blocks).
#pragma once
#import <CoreML/CoreML.h>

#include <map>
#include <string>
#include <vector>

#include "gpu.h"

namespace mllm {

class AneOps {
public:
    // Loads every block (compute units CPU + Neural Engine) and allocates its surfaces. Throws, naming the Python
    // command that builds them, if a block is missing.
    AneOps(GpuOps& engine_gpu, Memory& memory, int chunk_size);

    // set: which of the 2 surface sets (two chunks in flight, one set each; PrefillLane uses chunk parity).
    // Encodes into ws.encoder: the layer's block inputs (ws.residual, and ws.attention_result for layer > 0) into its
    // surfaces, zero past ws.tokens_per_request tokens, and its weights and norm scales into the set's weight surfaces.
    // Layer 0 first looks up ws.token_ids' embeddings into ws.residual.
    void prepare_inputs(int layer, Workspace& ws, int set);
    // Runs the layer's block on the ANE (blocks the thread until done; its inputs must be on the surfaces, i.e. the
    // command buffer that encoded prepare_inputs(layer) completed), then encodes into ws.encoder the copy of its
    // outputs' first ws.tokens_per_request tokens -> ws.residual and ws.query / key / value / query_gate (full
    // attention) or ws.query_key_value / output_gate / update_rate_input / decay_input (linear attention).
    void apply_weights(int layer, Workspace& ws, int set);

    struct Surface {
        Memory::Surface backing;  // the IOSurface and its Metal buffer
        MLMultiArray* array;      // over backing.pixel_buffer, [rows, cols]
        int cols;
    };
    // A block input or output that is copied from / to a workspace field
    struct Activation {
        id<MTLBuffer> Workspace::*field;
        Surface* surface[2];  // by set
    };
    struct Block {
        int chunk_size;  // compiled chunk size
        MLModel* model;
        MLDictionaryFeatureProvider* input_features[2];  // by set
        MLPredictionOptions* prediction_options[2];      // outputBackings: every output into its surface
        std::vector<Activation> inputs_from_workspace, outputs_to_workspace;
    };
    // What prepare_inputs(layer) writes into a weight surface: a 4-bit weight matrix, or (weights == nullptr) a norm's
    // bf16 scale of norm_size values
    struct WeightInput {
        const QuantizedWeights* weights;
        Buf norm_scale;
        int norm_size;
        Surface* surface[2];  // by set
    };
    int S;                         // the largest compiled chunk size
    std::vector<int> chunk_sizes;  // compiled chunk sizes, ascending
    std::string root;              // the blocks' directory for chunk size S
    std::map<std::pair<int, std::string>, Block> blocks;  // by (chunk size, kind)
    std::vector<std::vector<WeightInput>> weight_inputs;  // per layer
    // activations by name and chunk size, weights / norm scales by name and shape (shared by every chunk size)
    std::map<std::string, Surface> surfaces[2];
    const Block& block(int layer, int tokens) const;  // the block that runs `layer` on a chunk of `tokens` tokens

    // A copy of the engine's GpuOps (same pipelines and weights, no allocation) for the ANE lane's thread: GpuOps keeps
    // the encoder it is encoding into in members, so decode on the engine thread and prefill here can't share one.
    GpuOps gpu;

private:
    id<MTLComputePipelineState> copy_from_surface_pipeline;  // kernel "from_rows": fp16 surface -> bf16 workspace
    id<MTLComputePipelineState> copy_to_surface_pipeline;    // kernel "to_rows": bf16 workspace -> fp16 surface
    id<MTLComputePipelineState> dequantize_pipeline;         // kernel "dequant_q4": 4-bit weights -> fp16 surface
    std::vector<std::string> block_kind;                     // per layer

    Surface* find_or_make_surface(Memory& memory, int set, const std::string& key, int rows, int cols);
    Block load_block(Memory& memory, const std::string& dir, int chunk_size, MLModelConfiguration* config);
    std::vector<WeightInput> layer_weight_inputs(int layer, const ModelSpec& model);
    void copy_to_surface(Workspace& ws, Buf source, const Surface& target, uint32_t tokens, int rows);
    void dequantize_to_surface(Workspace& ws, const QuantizedWeights& weights, const Surface& target);
};

}  // namespace mllm
