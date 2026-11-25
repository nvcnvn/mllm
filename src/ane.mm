#include "ane.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace mllm {

namespace fs = std::filesystem;

static std::runtime_error error(const std::string& what, NSError* err) {
    return std::runtime_error(what + (err ? std::string(": ") + err.localizedDescription.UTF8String : ""));
}

// Core ML feature names (ane.py's) -> the workspace field each one is copied from / to
static const std::map<std::string, id<MTLBuffer> Workspace::*> WORKSPACE_FIELD_BY_NAME = {
    {"q", &Workspace::query},
    {"k", &Workspace::key},
    {"v", &Workspace::value},
    {"gate", &Workspace::query_gate},
    {"qkv", &Workspace::query_key_value},
    {"z", &Workspace::output_gate},
    {"b", &Workspace::update_rate_input},
    {"a", &Workspace::decay_input},
    {"x", &Workspace::residual},
    {"o", &Workspace::attention_result}};

// The error for a block that isn't built: says how to build every chunk size with Python
static std::runtime_error missing_block_error(const std::string& dir, const std::string& model_path,
                                              const std::vector<int>& chunk_sizes) {
    std::string size_list;
    for (int size : chunk_sizes) size_list += (size_list.empty() ? "" : ", ") + std::to_string(size);
    return std::runtime_error("ANE model missing: " + dir + "\nBuild the blocks with Python (~30 s):\n  HF_HOME=" +
                              (getenv("HF_HOME") ? getenv("HF_HOME") : "/Volumes/T5/huggingface") +
                              " .venv/bin/python -c \"import sys; sys.path.insert(0, 'py'); from mllm.ane import build; "
                              "build('" + model_path + "', [" + size_list + "])\"");
}

AneOps::AneOps(GpuOps& engine_gpu, Memory& memory, int chunk_size)
    : S(chunk_size), gpu(engine_gpu), copy_from_surface_pipeline(engine_gpu.pipeline(@"from_rows")),
      copy_to_surface_pipeline(engine_gpu.pipeline(@"to_rows")), dequantize_pipeline(engine_gpu.pipeline(@"dequant_q4")) {
    // apply_weights runs here and writes every workspace field: apply_attention must read those, not merged outputs
    this->gpu.merge_projections = false;
    const ModelSpec& model = memory.spec;
    if (S % 32 || S < 32) throw std::invalid_argument("ANE: chunk must be a multiple of 32");
    chunk_sizes = S > 512 ? std::vector<int>{512, S} : std::vector<int>{S};
    for (int layer = 0; layer < model.n_layers(); layer++)
        block_kind.push_back((layer ? model.kinds[layer - 1] : std::string("none")) + "_" + model.kinds[layer]);
    const std::string base = spec::cache_dir() + "/ane/" + spec::model_key(model.path);
    root = base + "/shared_S" + std::to_string(S);
    MLModelConfiguration* config = [MLModelConfiguration new];
    config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    for (int size : chunk_sizes)
        for (const std::string& kind : std::set<std::string>(block_kind.begin(), block_kind.end())) {
            const std::string dir = base + "/shared_S" + std::to_string(size) + "/" + kind;
            if (!fs::exists(dir + "/model.mlmodelc") || !fs::exists(dir + "/io.json"))
                throw missing_block_error(dir, model.path, chunk_sizes);
            blocks.emplace(std::pair{size, kind}, load_block(memory, dir, size, config));
        }
    for (int layer = 0; layer < model.n_layers(); layer++) weight_inputs.push_back(layer_weight_inputs(layer, model));
}

// Activations by name and chunk size, weights / norm scales by name and shape (the same for every chunk size), all
// fp16 [rows, cols]. Made on first use.
AneOps::Surface* AneOps::find_or_make_surface(Memory& memory, int set, const std::string& key, int rows, int cols) {
    auto found = surfaces[set].find(key);
    if (found == surfaces[set].end()) {
        Memory::Surface backing = memory.surface(rows, cols);
        MLMultiArray* array = [[MLMultiArray alloc] initWithPixelBuffer:(__bridge CVPixelBufferRef)backing.pixel_buffer
                                                                  shape:@[@(rows), @(cols)]];
        found = surfaces[set].emplace(key, Surface{backing, array, cols}).first;
    }
    return &found->second;
}

// One block: its model, and a surface per input and output (both sets), outputs written in place (outputBackings).
AneOps::Block AneOps::load_block(Memory& memory, const std::string& dir, int chunk_size, MLModelConfiguration* config) {
    NSError* err = nil;
    NSData* io_file = [NSData dataWithContentsOfFile:@((dir + "/io.json").c_str())];
    NSDictionary* io_json = io_file ? [NSJSONSerialization JSONObjectWithData:io_file options:0 error:&err] : nil;
    if (![io_json isKindOfClass:NSDictionary.class] || ![io_json[@"outputs"] isKindOfClass:NSDictionary.class])
        throw error(dir + "/io.json: unreadable", err);
    MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@((dir + "/model.mlmodelc").c_str())]
                                       configuration:config error:&err];
    if (!model) throw error(dir + ": cannot load", err);
    Block block{chunk_size, model, {}, {[MLPredictionOptions new], [MLPredictionOptions new]}, {}, {}};
    NSMutableDictionary* features[2] = {[NSMutableDictionary new], [NSMutableDictionary new]};
    NSMutableDictionary* backings[2] = {[NSMutableDictionary new], [NSMutableDictionary new]};
    auto shape_of = [&](MLFeatureDescription* description, const std::string& name) {
        MLMultiArrayConstraint* constraint = description.multiArrayConstraint;
        if (!constraint || constraint.dataType != MLMultiArrayDataTypeFloat16 || constraint.shape.count != 2)
            throw std::runtime_error(dir + ": " + name + " is not fp16 [rows, cols]");
        return std::pair{constraint.shape[0].intValue, constraint.shape[1].intValue};
    };
    auto both_sets = [&](const std::string& key, int rows, int cols) {
        return std::array<Surface*, 2>{find_or_make_surface(memory, 0, key, rows, cols),
                                       find_or_make_surface(memory, 1, key, rows, cols)};
    };

    // inputs: activations x / o (from the workspace), weights w_* and norm scales g_* (written by prepare_inputs)
    NSDictionary* inputs = model.modelDescription.inputDescriptionsByName;
    for (NSString* feature in inputs) {
        const std::string name = feature.UTF8String;
        const auto [rows, cols] = shape_of(inputs[feature], name);
        const bool activation = name == "x" || name == "o";
        if (activation && rows != chunk_size)
            throw std::runtime_error(dir + ": " + name + " has " + std::to_string(rows) + " rows");
        if (!activation && name.rfind("w_", 0) && name.rfind("g_", 0))
            throw std::runtime_error(dir + ": unknown input " + name);
        const std::string key = activation ? name + "@" + std::to_string(chunk_size)
                                           : name + ":" + std::to_string(rows) + "x" + std::to_string(cols);
        const auto surface = both_sets(key, rows, cols);
        for (int set = 0; set < 2; set++) features[set][feature] = surface[set]->array;
        if (activation) block.inputs_from_workspace.push_back({WORKSPACE_FIELD_BY_NAME.at(name), {surface[0], surface[1]}});
    }
    // outputs: io.json names the workspace field each Core ML output goes to
    NSDictionary<NSString*, NSString*>* field_names = io_json[@"outputs"];
    NSDictionary* outputs = model.modelDescription.outputDescriptionsByName;
    for (NSString* feature in outputs) {
        if (!field_names[feature] || !WORKSPACE_FIELD_BY_NAME.count(field_names[feature].UTF8String))
            throw error(dir + "/io.json: no name for an output", nil);
        const std::string name = field_names[feature].UTF8String;
        const auto [rows, cols] = shape_of(outputs[feature], name);
        const auto surface = both_sets(name + "_out@" + std::to_string(chunk_size), rows, cols);
        for (int set = 0; set < 2; set++) backings[set][feature] = surface[set]->array;
        block.outputs_to_workspace.push_back({WORKSPACE_FIELD_BY_NAME.at(name), {surface[0], surface[1]}});
    }
    for (int set = 0; set < 2; set++) {
        block.input_features[set] = [[MLDictionaryFeatureProvider alloc] initWithDictionary:features[set] error:&err];
        if (!block.input_features[set]) throw error(dir + ": inputs", err);
        block.prediction_options[set].outputBackings = backings[set];
    }
    return block;
}

// What prepare_inputs(layer) writes for the layer's block: the previous layer's attention output projection,
// feed-forward weights and feed-forward norm (the block for layer i finishes layer i - 1), then the layer's attention
// norm and its query / key / value (full attention) or query_key_value / output_gate / update_rate / decay (linear
// attention) projections. Each goes into the weight surface load_block() made for that name and shape.
std::vector<AneOps::WeightInput> AneOps::layer_weight_inputs(int layer, const ModelSpec& model) {
    auto find = [&](int set, const std::string& name, int rows, int cols) {
        auto found = surfaces[set].find(name + ":" + std::to_string(rows) + "x" + std::to_string(cols));
        if (found == surfaces[set].end())
            throw std::runtime_error(root + ": no block input " + name + " [" + std::to_string(rows) + ", " +
                                     std::to_string(cols) + "] (blocks built for another model?)");
        return &found->second;
    };
    auto weights = [&](const std::string& name, const QuantizedWeights& matrix) {
        if (matrix.in_size % 64) throw std::runtime_error("ANE: K not a multiple of the group size 64");
        return WeightInput{&matrix, {}, 0,
                           {find(0, name, matrix.out_size, matrix.in_size), find(1, name, matrix.out_size, matrix.in_size)}};
    };
    auto norm_scale = [&](const std::string& name, Buf scale, int size) {
        return WeightInput{nullptr, scale, size, {find(0, name, 1, size), find(1, name, 1, size)}};
    };
    std::vector<WeightInput> inputs;
    if (layer > 0) {
        const GpuOps::Layer& previous = gpu.layers[layer - 1];
        inputs = {weights("w_o", previous.attention_output), weights("w_gate", previous.feed_forward_gate),
                  weights("w_up", previous.feed_forward_up), weights("w_down", previous.feed_forward_down),
                  norm_scale("g_post", previous.feed_forward_norm, model.hidden)};
    }
    const GpuOps::Layer& current = gpu.layers[layer];
    inputs.push_back(norm_scale("g_in", current.attention_norm, model.hidden));
    if (current.is_linear) {
        inputs.push_back(weights("w_qkv", current.query_key_value_projection));
        inputs.push_back(weights("w_z", current.output_gate_projection));
        inputs.push_back(weights("w_b", current.update_rate_projection));
        inputs.push_back(weights("w_a", current.decay_projection));
    } else {
        inputs.push_back(weights("w_q", current.query_projection));
        inputs.push_back(weights("w_k", current.key_projection));
        inputs.push_back(weights("w_v", current.value_projection));
        inputs.push_back(norm_scale("g_q", current.query_norm, model.head_dim));
        inputs.push_back(norm_scale("g_k", current.key_norm, model.head_dim));
    }
    return inputs;
}

const AneOps::Block& AneOps::block(int layer, int tokens) const {
    return blocks.at({*std::lower_bound(chunk_sizes.begin(), chunk_sizes.end(), tokens), block_kind[layer]});
}

void AneOps::prepare_inputs(int layer, Workspace& ws, int set) {
    assert(ws.request_count == 1 && ws.tokens_per_request > 0 && ws.tokens_per_request <= S && (set == 0 || set == 1));
    gpu.begin(ws);
    if (layer == 0) gpu.look_up_embeddings({ws.token_ids}, {ws.residual}, ws.tokens_per_request);
    const Block& current = block(layer, ws.tokens_per_request);
    for (const Activation& input : current.inputs_from_workspace)
        copy_to_surface(ws, {ws.*input.field}, *input.surface[set], ws.tokens_per_request, current.chunk_size);
    // the layer's weights and norm scales into this set's weight surfaces (the ANE is done with them: this set's last
    // prediction returned before this command buffer was encoded)
    for (const WeightInput& input : weight_inputs[layer]) {
        if (input.weights) dequantize_to_surface(ws, *input.weights, *input.surface[set]);
        else copy_to_surface(ws, input.norm_scale, *input.surface[set], 1, 1);
    }
}

// bf16 [tokens, target.cols] at source -> target's first `rows` fp16 rows, zero in rows tokens .. rows
void AneOps::copy_to_surface(Workspace& ws, Buf source, const Surface& target, uint32_t tokens, int rows) {
    [ws.encoder setComputePipelineState:copy_to_surface_pipeline];
    [ws.encoder setBuffer:source.buffer offset:source.offset atIndex:0];
    [ws.encoder setBuffer:target.backing.buffer offset:0 atIndex:1];
    [ws.encoder setBytes:&target.backing.row length:4 atIndex:2];
    [ws.encoder setBytes:&tokens length:4 atIndex:3];
    [ws.encoder dispatchThreads:MTLSizeMake(target.cols, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, rows > 1 ? 8 : 1, 1)];
}

// 4-bit weights [out_size, in_size] -> target's fp16 rows; one thread per 8 weights
void AneOps::dequantize_to_surface(Workspace& ws, const QuantizedWeights& weights, const Surface& target) {
    [ws.encoder setComputePipelineState:dequantize_pipeline];
    [ws.encoder setBuffer:weights.packed.buffer offset:weights.packed.offset atIndex:0];
    [ws.encoder setBuffer:weights.scales.buffer offset:weights.scales.offset atIndex:1];
    [ws.encoder setBuffer:weights.biases.buffer offset:weights.biases.offset atIndex:2];
    [ws.encoder setBuffer:target.backing.buffer offset:0 atIndex:3];
    [ws.encoder setBytes:&target.backing.row length:4 atIndex:4];
    [ws.encoder dispatchThreads:MTLSizeMake(weights.in_size / 8, weights.out_size, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
}

void AneOps::apply_weights(int layer, Workspace& ws, int set) {
    assert(ws.request_count == 1 && ws.tokens_per_request > 0 && ws.tokens_per_request <= S && (set == 0 || set == 1));
    const Block& current = block(layer, ws.tokens_per_request);
    NSError* err = nil;
    id<MLFeatureProvider> result = [current.model predictionFromFeatures:current.input_features[set]
                                                                 options:current.prediction_options[set]
                                                                   error:&err];
    if (!result) throw error("ANE dense block " + std::to_string(layer), err);
    // Core ML copies into a fresh buffer when it can't use a backing: our surfaces would then hold stale data
    NSDictionary* backings = current.prediction_options[set].outputBackings;
    for (NSString* feature in backings)
        if ([result featureValueForName:feature].multiArrayValue.pixelBuffer != ((MLMultiArray*)backings[feature]).pixelBuffer)
            throw std::runtime_error("ANE dense block " + std::to_string(layer) + ": output not written in place");
    for (const Activation& output : current.outputs_to_workspace) {
        const Surface& source = *output.surface[set];
        [ws.encoder setComputePipelineState:copy_from_surface_pipeline];
        [ws.encoder setBuffer:source.backing.buffer offset:0 atIndex:0];
        [ws.encoder setBuffer:ws.*output.field offset:0 atIndex:1];
        [ws.encoder setBytes:&source.backing.row length:4 atIndex:2];
        [ws.encoder dispatchThreads:MTLSizeMake(source.cols, ws.tokens_per_request, 1)
              threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }
}

}  // namespace mllm
