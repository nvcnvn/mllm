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

static const std::map<std::string, id<MTLBuffer> Workspace::*> FIELDS = {
    {"q", &Workspace::q},     {"k", &Workspace::k},     {"v", &Workspace::v}, {"gate", &Workspace::gate},
    {"qkv", &Workspace::qkv}, {"z", &Workspace::z},     {"b", &Workspace::b}, {"a", &Workspace::a},
    {"x", &Workspace::x},     {"o", &Workspace::o}};

AneOps::AneOps(GpuOps& gpu, Memory& mem, int S)
    : S(S), gpu(gpu), from_rows_ps(gpu.pipeline(@"from_rows")), to_rows_ps(gpu.pipeline(@"to_rows")),
      dequant_ps(gpu.pipeline(@"dequant_q4")) {
    this->gpu.merge = false;  // dense runs here and writes every workspace field: attn must read those
    const ModelSpec& sp = mem.spec;
    if (S % 32 || S < 32) throw std::invalid_argument("ANE: chunk must be a multiple of 32");
    sizes = S > 512 ? std::vector<int>{512, S} : std::vector<int>{S};
    for (int i = 0; i < sp.n_layers(); i++)
        kind_of.push_back((i ? sp.kinds[i - 1] : std::string("none")) + "_" + sp.kinds[i]);
    const std::string base = spec::cache_dir() + "/ane/" + spec::model_key(sp.path);
    root = base + "/shared_S" + std::to_string(S);
    std::string list;
    for (int Sk : sizes) list += (list.empty() ? "" : ", ") + std::to_string(Sk);
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    for (int Sk : sizes)
        for (const std::string& kind : std::set<std::string>(kind_of.begin(), kind_of.end())) {
            const std::string dir = base + "/shared_S" + std::to_string(Sk) + "/" + kind;
            if (!fs::exists(dir + "/model.mlmodelc") || !fs::exists(dir + "/io.json"))
                throw std::runtime_error(
                    "ANE model missing: " + dir + "\nBuild the blocks with Python (~30 s):\n  HF_HOME=" +
                    (getenv("HF_HOME") ? getenv("HF_HOME") : "/Volumes/T5/huggingface") +
                    " .venv/bin/python -c \"import sys; sys.path.insert(0, 'py'); from mllm.ane import build; "
                    "build('" + sp.path + "', [" + list + "])\"");
            blocks.emplace(std::pair{Sk, kind}, load(mem, dir, Sk, cfg));
        }
    for (int i = 0; i < sp.n_layers(); i++) params.push_back(layer_params(i, sp));
}

// Activations by name and S, weights / gammas by name and shape (the same for every S), all fp16 [rows, cols].
AneOps::Surface* AneOps::surface(Memory& mem, int set, const std::string& key, int rows, int cols) {
    auto it = surfaces[set].find(key);
    if (it == surfaces[set].end()) {
        Memory::Surface m = mem.surface(rows, cols);
        MLMultiArray* a = [[MLMultiArray alloc] initWithPixelBuffer:(__bridge CVPixelBufferRef)m.pixel_buffer
                                                              shape:@[@(rows), @(cols)]];
        it = surfaces[set].emplace(key, Surface{m, a, cols}).first;
    }
    return &it->second;
}

// One block: its model, and a surface per input and output (both sets), outputs written in place (outputBackings).
AneOps::Block AneOps::load(Memory& mem, const std::string& dir, int Sk, MLModelConfiguration* cfg) {
    NSError* err = nil;
    NSData* io = [NSData dataWithContentsOfFile:@((dir + "/io.json").c_str())];
    NSDictionary* j = io ? [NSJSONSerialization JSONObjectWithData:io options:0 error:&err] : nil;
    if (![j isKindOfClass:NSDictionary.class] || ![j[@"outputs"] isKindOfClass:NSDictionary.class])
        throw error(dir + "/io.json: unreadable", err);
    MLModel* m = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:@((dir + "/model.mlmodelc").c_str())]
                                   configuration:cfg error:&err];
    if (!m) throw error(dir + ": cannot load", err);
    Block b{Sk, m, {}, {[MLPredictionOptions new], [MLPredictionOptions new]}, {}, {}};
    NSMutableDictionary* feed[2] = {[NSMutableDictionary new], [NSMutableDictionary new]};
    NSMutableDictionary* backings[2] = {[NSMutableDictionary new], [NSMutableDictionary new]};
    auto dims = [&](MLFeatureDescription* d, const std::string& name) {
        MLMultiArrayConstraint* c = d.multiArrayConstraint;
        if (!c || c.dataType != MLMultiArrayDataTypeFloat16 || c.shape.count != 2)
            throw std::runtime_error(dir + ": " + name + " is not fp16 [rows, cols]");
        return std::pair{c.shape[0].intValue, c.shape[1].intValue};
    };
    auto both = [&](const std::string& key, int rows, int cols) {
        return std::array<Surface*, 2>{surface(mem, 0, key, rows, cols), surface(mem, 1, key, rows, cols)};
    };

    // inputs: activations x / o (from the workspace), weights w_* and gammas g_* (written by inputs(i))
    NSDictionary* in = m.modelDescription.inputDescriptionsByName;
    for (NSString* k in in) {
        const std::string name = k.UTF8String;
        const auto [rows, cols] = dims(in[k], name);
        const bool act = name == "x" || name == "o";
        if (act && rows != Sk) throw std::runtime_error(dir + ": " + name + " has " + std::to_string(rows) + " rows");
        if (!act && name.rfind("w_", 0) && name.rfind("g_", 0)) throw std::runtime_error(dir + ": unknown input " + name);
        const std::string key =
            act ? name + "@" + std::to_string(Sk) : name + ":" + std::to_string(rows) + "x" + std::to_string(cols);
        const auto sf = both(key, rows, cols);
        for (int set = 0; set < 2; set++) feed[set][k] = sf[set]->array;
        if (act) b.ins.push_back({FIELDS.at(name), {sf[0], sf[1]}});
    }
    // outputs: io.json names the workspace field each Core ML output goes to
    NSDictionary<NSString*, NSString*>* ours = j[@"outputs"];
    NSDictionary* out = m.modelDescription.outputDescriptionsByName;
    for (NSString* k in out) {
        if (!ours[k] || !FIELDS.count(ours[k].UTF8String)) throw error(dir + "/io.json: no name for an output", nil);
        const std::string name = ours[k].UTF8String;
        const auto [rows, cols] = dims(out[k], name);
        const auto sf = both(name + "_out@" + std::to_string(Sk), rows, cols);
        for (int set = 0; set < 2; set++) backings[set][k] = sf[set]->array;
        b.outs.push_back({FIELDS.at(name), {sf[0], sf[1]}});
    }
    for (int set = 0; set < 2; set++) {
        b.in[set] = [[MLDictionaryFeatureProvider alloc] initWithDictionary:feed[set] error:&err];
        if (!b.in[set]) throw error(dir + ": inputs", err);
        b.opts[set].outputBackings = backings[set];
    }
    return b;
}

// What inputs(i) writes for layer i's block: the previous layer's o_proj, FFN and post norm (block i finishes layer
// i - 1), then layer i's input norm and q/k/v (full) or qkv/z/b/a (linear) projections. Each goes into the weight
// surface load() made for that name and shape.
std::vector<AneOps::Param> AneOps::layer_params(int i, const ModelSpec& sp) {
    auto find = [&](int set, const std::string& name, int rows, int cols) {
        auto it = surfaces[set].find(name + ":" + std::to_string(rows) + "x" + std::to_string(cols));
        if (it == surfaces[set].end())
            throw std::runtime_error(root + ": no block input " + name + " [" + std::to_string(rows) + ", " +
                                     std::to_string(cols) + "] (blocks built for another model?)");
        return &it->second;
    };
    auto weight = [&](const std::string& name, const QLinear& w) {
        if (w.K % 64) throw std::runtime_error("ANE: K not a multiple of the group size 64");
        return Param{&w, {}, 0, {find(0, name, w.N, w.K), find(1, name, w.N, w.K)}};
    };
    auto gamma = [&](const std::string& name, Buf g, int dim) {
        return Param{nullptr, g, dim, {find(0, name, 1, dim), find(1, name, 1, dim)}};
    };
    std::vector<Param> ps;
    if (i > 0) {
        const GpuOps::Layer& p = gpu.layers[i - 1];
        ps = {weight("w_o", p.out), weight("w_gate", p.gate_proj), weight("w_up", p.up_proj),
              weight("w_down", p.down_proj), gamma("g_post", p.post_attention_layernorm, sp.hidden)};
    }
    const GpuOps::Layer& l = gpu.layers[i];
    ps.push_back(gamma("g_in", l.input_layernorm, sp.hidden));
    if (l.is_linear) {
        ps.push_back(weight("w_qkv", l.in_proj_qkv)), ps.push_back(weight("w_z", l.in_proj_z));
        ps.push_back(weight("w_b", l.in_proj_b)), ps.push_back(weight("w_a", l.in_proj_a));
    } else {
        ps.push_back(weight("w_q", l.q_proj)), ps.push_back(weight("w_k", l.k_proj));
        ps.push_back(weight("w_v", l.v_proj));
        ps.push_back(gamma("g_q", l.q_norm, sp.head_dim)), ps.push_back(gamma("g_k", l.k_norm, sp.head_dim));
    }
    return ps;
}

const AneOps::Block& AneOps::block(int i, int n) const {
    return blocks.at({*std::lower_bound(sizes.begin(), sizes.end(), n), kind_of[i]});
}

void AneOps::inputs(int i, Workspace& ws, int set) {
    assert(ws.B == 1 && ws.S > 0 && ws.S <= S && (set == 0 || set == 1));
    gpu.use(ws);
    if (i == 0) gpu.embed({ws.tokens}, {ws.x}, ws.S);
    const Block& b = block(i, ws.S);
    for (const Io& io : b.ins) to_rows(ws, {ws.*io.field}, *io.surface[set], ws.S, b.S);
    // block i's weights and gammas into this set's weight surfaces (the ANE is done with them: this set's last
    // predict returned before this command buffer was encoded)
    for (const Param& p : params[i]) {
        const Memory::Surface& d = p.surface[set]->s;
        if (!p.w) {
            to_rows(ws, p.gamma, *p.surface[set], 1, 1);
            continue;
        }
        [ws.enc setComputePipelineState:dequant_ps];
        [ws.enc setBuffer:p.w->w.buffer offset:p.w->w.offset atIndex:0];
        [ws.enc setBuffer:p.w->scales.buffer offset:p.w->scales.offset atIndex:1];
        [ws.enc setBuffer:p.w->biases.buffer offset:p.w->biases.offset atIndex:2];
        [ws.enc setBuffer:d.buffer offset:0 atIndex:3];
        [ws.enc setBytes:&d.row length:4 atIndex:4];
        [ws.enc dispatchThreads:MTLSizeMake(p.w->K / 8, p.w->N, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }
}

// bf16 [n, dst.C] at src -> dst's first `rows` fp16 rows, zero in rows n .. rows
void AneOps::to_rows(Workspace& ws, Buf src, const Surface& dst, uint32_t n, int rows) {
    [ws.enc setComputePipelineState:to_rows_ps];
    [ws.enc setBuffer:src.buffer offset:src.offset atIndex:0];
    [ws.enc setBuffer:dst.s.buffer offset:0 atIndex:1];
    [ws.enc setBytes:&dst.s.row length:4 atIndex:2];
    [ws.enc setBytes:&n length:4 atIndex:3];
    [ws.enc dispatchThreads:MTLSizeMake(dst.C, rows, 1) threadsPerThreadgroup:MTLSizeMake(32, rows > 1 ? 8 : 1, 1)];
}

void AneOps::dense(int i, Workspace& ws, int set) {
    assert(ws.B == 1 && ws.S > 0 && ws.S <= S && (set == 0 || set == 1));
    const Block& b = block(i, ws.S);
    NSError* err = nil;
    id<MLFeatureProvider> r = [b.model predictionFromFeatures:b.in[set] options:b.opts[set] error:&err];
    if (!r) throw error("ANE dense block " + std::to_string(i), err);
    // Core ML copies into a fresh buffer when it can't use a backing: our surfaces would then hold stale data
    NSDictionary* backings = b.opts[set].outputBackings;
    for (NSString* k in backings)
        if ([r featureValueForName:k].multiArrayValue.pixelBuffer != ((MLMultiArray*)backings[k]).pixelBuffer)
            throw std::runtime_error("ANE dense block " + std::to_string(i) + ": output not written in place");
    for (const Io& io : b.outs) {
        [ws.enc setComputePipelineState:from_rows_ps];
        [ws.enc setBuffer:io.surface[set]->s.buffer offset:0 atIndex:0];
        [ws.enc setBuffer:ws.*io.field offset:0 atIndex:1];
        [ws.enc setBytes:&io.surface[set]->s.row length:4 atIndex:2];
        [ws.enc dispatchThreads:MTLSizeMake(io.surface[set]->C, ws.S, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    }
}

}  // namespace mllm
