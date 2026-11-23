#include "ane.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace mllm {

namespace fs = std::filesystem;

static std::runtime_error error(const std::string& what, NSError* err) {
    return std::runtime_error(what + (err ? std::string(": ") + err.localizedDescription.UTF8String : ""));
}

static const std::map<std::string, id<MTLBuffer> Workspace::*> FIELDS = {
    {"q", &Workspace::q},     {"k", &Workspace::k},     {"v", &Workspace::v}, {"gate", &Workspace::gate},
    {"qkv", &Workspace::qkv}, {"z", &Workspace::z},     {"b", &Workspace::b}, {"a", &Workspace::a},
    {"x", &Workspace::x},     {"o", &Workspace::o}};

AneOps::AneOps(GpuOps& gpu, Memory& mem, int S, const std::string& weights)
    : S(S), root(spec::cache_dir() + "/ane/" + spec::model_key(mem.spec.path) + "/S" + std::to_string(S) + "_" + weights),
      gpu(gpu), from_ane_ps(gpu.pipeline(@"from_ane")), to_ane_ps(gpu.pipeline(@"to_ane")) {
    this->gpu.merge = false;  // dense runs here and writes every workspace field: attn must read those
    // The surface for a feature: shared by every block with the same name and channels
    auto surface = [&](int set, const std::string& name, MLFeatureDescription* d, const std::string& where) -> Surface* {
        MLMultiArrayConstraint* c = d.multiArrayConstraint;
        NSArray<NSNumber*>* shape = c.shape;
        if (!c || c.dataType != MLMultiArrayDataTypeFloat16 || shape.count != 4 || shape[0].intValue != 1 ||
            shape[3].intValue != S)
            throw std::runtime_error(where + ": " + name + " is not fp16 (1, C, H, " + std::to_string(S) + ")");
        const int C = shape[1].intValue * shape[2].intValue;
        const std::string key = name + std::to_string(C);
        auto it = surfaces[set].find(key);
        if (it == surfaces[set].end()) {
            Memory::Surface m = mem.surface(C, S);
            MLMultiArray* a = [[MLMultiArray alloc] initWithPixelBuffer:(__bridge CVPixelBufferRef)m.pixel_buffer
                                                                  shape:shape];
            it = surfaces[set].emplace(key, Surface{m, a, C}).first;
        }
        if (it->second.C != C || ![it->second.array.shape isEqualToArray:shape])
            throw std::runtime_error(where + ": " + name + " changes shape between blocks");
        return &it->second;
    };
    if (weights != "fp16" && weights != "int8" && weights != "lut6" && weights != "lut4")  // ane.py _compress
        throw std::invalid_argument("ane_weights: fp16, int8, lut6 or lut4, not " + weights);
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    for (int i = 0; i < mem.spec.n_layers(); i++) {
        char name[16];
        snprintf(name, sizeof name, "/dense_%03d", i);
        const std::string dir = root + name;
        if (!fs::exists(dir + "/model.mlmodelc") || !fs::exists(dir + "/io.json"))
            throw std::runtime_error(
                "ANE model missing: " + dir + "\nBuild the blocks with Python (takes minutes):\n  HF_HOME=" +
                (getenv("HF_HOME") ? getenv("HF_HOME") : "/Volumes/T5/huggingface") +
                " .venv/bin/python -c \"import sys; sys.path.insert(0, 'py'); from mllm.engine import Engine; "
                "Engine('" + mem.spec.path + "', chunk=" + std::to_string(S) + ", prefill='ane', ane_weights='" +
                weights + "')\"");
        NSError* err = nil;
        NSData* io = [NSData dataWithContentsOfFile:@((dir + "/io.json").c_str())];
        NSDictionary* j = io ? [NSJSONSerialization JSONObjectWithData:io options:0 error:&err] : nil;
        if (![j isKindOfClass:NSDictionary.class] || ![j[@"outputs"] isKindOfClass:NSDictionary.class])
            throw error(dir + "/io.json: unreadable", err);
        MLModel* m = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@((dir + "/model.mlmodelc").c_str())]
                                       configuration:cfg error:&err];
        if (!m) throw error(dir + ": cannot load", err);
        Block b{m, {}, {[MLPredictionOptions new], [MLPredictionOptions new]}, {}, {}};
        NSMutableDictionary* feed[2] = {[NSMutableDictionary new], [NSMutableDictionary new]};
        NSMutableDictionary* backings[2] = {[NSMutableDictionary new], [NSMutableDictionary new]};
        NSDictionary* in = m.modelDescription.inputDescriptionsByName;
        for (NSString* k in in) {
            const std::string name = k.UTF8String;
            if (name != "x" && name != "o") throw std::runtime_error(dir + ": unknown input " + name);
            Io io{FIELDS.at(name), {surface(0, name, in[k], dir), surface(1, name, in[k], dir)}};
            for (int set = 0; set < 2; set++) feed[set][k] = io.surface[set]->array;
            b.ins.push_back(io);
        }
        NSDictionary<NSString*, NSString*>* ours = j[@"outputs"];
        NSDictionary* out = m.modelDescription.outputDescriptionsByName;
        for (NSString* k in out) {
            if (!ours[k] || !FIELDS.count(ours[k].UTF8String)) throw error(dir + "/io.json: no name for an output", nil);
            const std::string name = ours[k].UTF8String, sf = name == "x" ? "x_out" : name;
            Io io{FIELDS.at(name), {surface(0, sf, out[k], dir), surface(1, sf, out[k], dir)}};
            for (int set = 0; set < 2; set++) backings[set][k] = io.surface[set]->array;
            b.outs.push_back(io);
        }
        for (int set = 0; set < 2; set++) {
            b.in[set] = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed[set] error:&err];
            if (!b.in[set]) throw error(dir + ": inputs", err);
            b.opts[set].outputBackings = backings[set];
        }
        blocks.push_back(b);
    }
}

void AneOps::inputs(int i, Workspace& ws, int set) {
    assert(ws.B == 1 && ws.S > 0 && ws.S <= S && (set == 0 || set == 1));
    gpu.use(ws);
    if (i == 0) gpu.embed({ws.tokens}, {ws.x}, ws.S);
    for (const Io& io : blocks[i].ins) {
        [ws.enc setComputePipelineState:to_ane_ps];
        [ws.enc setBuffer:ws.*io.field offset:0 atIndex:0];
        [ws.enc setBuffer:io.surface[set]->s.buffer offset:0 atIndex:1];
        [ws.enc setBytes:&io.surface[set]->s.row length:4 atIndex:2];
        const uint32_t n = ws.S;
        [ws.enc setBytes:&n length:4 atIndex:3];
        [ws.enc dispatchThreads:MTLSizeMake(io.surface[set]->C, S, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }
}

void AneOps::dense(int i, Workspace& ws, int set) {
    assert(ws.B == 1 && ws.S > 0 && ws.S <= S && (set == 0 || set == 1));
    const Block& b = blocks[i];
    NSError* err = nil;
    id<MLFeatureProvider> r = [b.model predictionFromFeatures:b.in[set] options:b.opts[set] error:&err];
    if (!r) throw error("ANE dense block " + std::to_string(i), err);
    // Core ML copies into a fresh buffer when it can't use a backing: our surfaces would then hold stale data
    NSDictionary* backings = b.opts[set].outputBackings;
    for (NSString* k in backings)
        if ([r featureValueForName:k].multiArrayValue.pixelBuffer != ((MLMultiArray*)backings[k]).pixelBuffer)
            throw std::runtime_error("ANE dense block " + std::to_string(i) + ": output not written in place");
    for (const Io& io : b.outs) {
        [ws.enc setComputePipelineState:from_ane_ps];
        [ws.enc setBuffer:io.surface[set]->s.buffer offset:0 atIndex:0];
        [ws.enc setBuffer:ws.*io.field offset:0 atIndex:1];
        [ws.enc setBytes:&io.surface[set]->s.row length:4 atIndex:2];
        [ws.enc dispatchThreads:MTLSizeMake(io.surface[set]->C, ws.S, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }
}

}  // namespace mllm
