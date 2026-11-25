// AneOps (cpp-port.md §6). Needs the S = 512 fp16 blocks built (the test prints the command if not).
// a. every block loads; a missing build names the Python command.
// b. blocks 0 (linear), 3 (full) and 4 (linear, o from a full layer) on golden prefill0 inputs (32 tokens, padded to
//    512) through the IOSurfaces: the workspace fields the GPU converted from the surfaces are bit-exact with the
//    same block run with plain, CPU-copied MLMultiArrays, and within cos 0.999 of the golden GPU tensors.
// c. PrefillLane on the ANE (its own thread) vs on the GPU, for the 3 golden chat prompts (one padded chunk each) and
//    a 627-token prompt (2 chunks): the worst cos over every full layer's K / V and every linear layer's conv /
//    recurrent state equals ane.py's (fp16 and int8 blocks). Then an Engine with ANE prefill: greedy 64 equals golden
//    for the 3 prompts, also while long prompts prefill alongside, and the long prompt's greedy tokens equal the
//    GPU-prefill engine's.
// c. Two chunks in flight on the ANE lane: bit-exact with one at a time; cancel / abort with two in flight.
// d. Chunks below gpu_below tokens (tails, prefix-cache splits) run on the GPU path.
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unistd.h>

#include "ane.h"
#include "engine.h"
#include "golden.h"

static const std::map<std::string, id<MTLBuffer> mllm::Workspace::*> FIELDS = {
    {"q", &mllm::Workspace::q},       {"k", &mllm::Workspace::k},     {"v", &mllm::Workspace::v},
    {"gate", &mllm::Workspace::gate}, {"qkv", &mllm::Workspace::qkv}, {"z", &mllm::Workspace::z},
    {"b", &mllm::Workspace::b},       {"a", &mllm::Workspace::a},     {"x", &mllm::Workspace::x}};

static float bf16(uint16_t u) {
    uint32_t w = (uint32_t)u << 16;
    float f;
    memcpy(&f, &w, 4);
    return f;
}

static uint16_t to_bf16(float f) {  // round to nearest even (no NaNs here)
    uint32_t w;
    memcpy(&w, &f, 4);
    return (uint16_t)((w + 0x7fff + ((w >> 16) & 1)) >> 16);
}

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    const std::string d = golden::dir(s);
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, 512 << 20);
    mllm::GpuOps ops(mem, w);

    try {
        mllm::AneOps none(ops, mem, 256, "lut6");  // not built
        assert(!"loaded a missing build");
    } catch (const std::runtime_error& e) {
        assert(std::string(e.what()).find("chunk=256, prefill='ane', ane_weights='lut6'") != std::string::npos);
    }
    try {
        mllm::AneOps none(ops, mem, 512, "no_such_format");
        assert(!"accepted an unknown format");
    } catch (const std::invalid_argument&) {
    }
    auto t0 = std::chrono::steady_clock::now();
    const int S = 512;
    mllm::AneOps ane(ops, mem, S, "fp16");  // fails with the build command when the models aren't built
    assert((int)ane.blocks.size() == s.n_layers());
    printf("a. loaded %zu blocks in %.1f s, %zu surfaces\n", ane.blocks.size(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ane.surfaces[0].size());
    mllm::Workspace ws = ops.workspace(S, 1, S);
    const int max_ctx = 1536;  // the 3-chunk prompt (1300 tokens)
    mllm::AneOps ane8(ops, mem, S, "int8"), ane_l(ops, mem, S, "fp16");  // the lanes' (b used `ane` on this thread)
    mllm::PrefillLane gpu_lane(ops, mem, S, max_ctx), ane_lane(ops, mem, S, max_ctx, &ane_l),
        ane8_lane(ops, mem, S, max_ctx, &ane8);
    mem.freeze();
    id<MTLCommandQueue> q = [dev newCommandQueue];

    const int n = 32;  // golden prefill0: 32 tokens at pos 0
    ws.B = 1, ws.S = n;
    auto put = [&](id<MTLBuffer> b, const std::string& name) {  // a golden bf16 tensor -> workspace rows
        const mllm::Tensor& t = tr.at(name);
        memcpy(b.contents, tr.data(t), t.nbytes);
    };
    const mllm::Tensor& tok = tr.at("prefill0/tokens");
    memcpy(ws.tokens.contents, tr.data(tok), tok.nbytes);
    for (int i : {0, 3, 4}) {
        mllm::AneOps::Block& b = ane.blocks[i];
        char L[8];
        snprintf(L, sizeof L, "L%02d", i - 1);
        if (i > 0) {
            put(ws.x, "prefill0/" + std::string(L) + "/dense/x");
            put(ws.o, "prefill0/" + std::string(L) + "/attn/o");
        }
        mllm::encode_and_wait(q, ws, [&] { ane.inputs(i, ws, 0); });
        // padding: every input surface is zero past n
        for (NSString* k in b.in[0].featureNames) {
            MLMultiArray* a = [b.in[0] featureValueForName:k].multiArrayValue;
            CVPixelBufferRef pb = a.pixelBuffer;
            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
            const uint16_t* p = (const uint16_t*)CVPixelBufferGetBaseAddress(pb);
            const size_t row = CVPixelBufferGetBytesPerRow(pb) / 2, C = CVPixelBufferGetHeight(pb);
            for (size_t c = 0; c < C; c++)
                for (int t = n; t < S; t++) assert(p[c * row + t] == 0);
            CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        }
        mllm::encode_and_wait(q, ws, [&] { ane.dense(i, ws, 0); });

        // the same block, inputs copied on the CPU into plain arrays, outputs in Core ML's own buffers
        NSMutableDictionary* feed = [NSMutableDictionary new];
        for (NSString* k in b.in[0].featureNames) {
            CVPixelBufferRef pb = [b.in[0] featureValueForName:k].multiArrayValue.pixelBuffer;
            MLMultiArray* a = [[MLMultiArray alloc] initWithShape:[b.in[0] featureValueForName:k].multiArrayValue.shape
                                                         dataType:MLMultiArrayDataTypeFloat16 error:nil];
            assert(a.strides[1].intValue * a.shape[1].intValue == a.count);  // dense
            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
            const size_t row = CVPixelBufferGetBytesPerRow(pb) / 2, C = CVPixelBufferGetHeight(pb);
            for (size_t c = 0; c < C; c++)
                memcpy((uint16_t*)a.dataPointer + c * S, (const uint16_t*)CVPixelBufferGetBaseAddress(pb) + c * row, S * 2);
            CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
            feed[k] = a;
        }
        id<MLFeatureProvider> ref = [b.model predictionFromFeatures:[[MLDictionaryFeatureProvider alloc]
                                                                        initWithDictionary:feed error:nil]
                                                              error:nil];
        assert(ref && "plain prediction failed");
        snprintf(L, sizeof L, "L%02d", i);
        for (const mllm::AneOps::Io& io : b.outs) {
            NSString* k = [b.opts[0].outputBackings allKeysForObject:io.surface[0]->array].firstObject;
            std::string name;
            for (const auto& [f, field] : FIELDS)
                if (ws.*io.field == ws.*field) name = f;
            MLMultiArray* r = [ref featureValueForName:k].multiArrayValue;
            const int C = io.surface[0]->C, inner = r.shape[2].intValue;
            const size_t cs = r.strides[1].intValue, hs = r.strides[2].intValue, ts = r.strides[3].intValue;
            const uint16_t* got = (const uint16_t*)(ws.*io.field).contents;
            __block size_t diff = 0;
            [r getBytesWithHandler:^(const void* p, NSInteger) {
              const _Float16* h = (const _Float16*)p;
              for (int c = 0; c < C; c++)
                  for (int t = 0; t < n; t++)
                      diff += got[(size_t)t * C + c] != to_bf16((float)h[(c / inner) * cs + (c % inner) * hs + t * ts]);
            }];
            // golden, gpu.py layout: q / k / v are [1, H, n, D], the rest [1, n, C]
            const mllm::Tensor& g = tr.at("prefill0/" + std::string(L) + "/dense/" + name);
            const bool heads = g.shape.size() == 4 && g.shape[2] == n;
            const int D = heads ? (int)g.shape[3] : C;
            double dot = 0, gg = 0, ww = 0;
            for (int t = 0; t < n; t++)
                for (int c = 0; c < C; c++) {
                    size_t gi = heads ? ((size_t)(c / D) * n + t) * D + c % D : (size_t)t * C + c;
                    double x = bf16(got[(size_t)t * C + c]), z = bf16(golden::load<uint16_t>(tr.data(g), gi));
                    dot += x * z, gg += x * x, ww += z * z;
                }
            const double cos = dot / std::sqrt(gg * ww);
            printf("b. block %d %-4s: %zu / %d differ from the CPU-copy run, cos vs golden %.6f\n", i, name.c_str(),
                   diff, C * n, cos);
            assert(diff == 0 && cos >= 0.999);
        }
    }

    // c. lanes
    mllm::Safetensors tk(d + "/tokens.safetensors");
    auto ids_of = [&](const std::string& name) {
        const mllm::Tensor& t = tk.at(name);
        std::vector<int> v(t.nbytes / 4);
        for (size_t j = 0; j < v.size(); j++) v[j] = golden::load<int32_t>(tk.data(t), j);
        return v;
    };
    std::vector<std::vector<int>> prompts = {ids_of("chat/0"), ids_of("chat/1"), ids_of("chat/2"), {}};
    while (prompts[3].size() < 600) prompts[3].insert(prompts[3].end(), prompts[0].begin(), prompts[0].end());
    mllm::Arena& ar = mem.arena;
    uint8_t* arena = (uint8_t*)mem.arena_buffer.contents;
    struct Units {
        std::vector<uint32_t> pages, state;
    };
    auto units = [&](int n) {  // fresh units for n tokens, state zeroed
        Units u;
        assert(ar.admit(mem.units_for(n) + mem.state_units));
        for (uint32_t j = 0; j < mem.units_for(n); j++) u.pages.push_back(ar.alloc());
        for (uint32_t j = 0; j < mem.state_units; j++) u.state.push_back(ar.alloc());
        for (uint32_t x : u.state) memset(arena + (size_t)x * ar.unit_bytes, 0, ar.unit_bytes);
        return u;
    };
    // every prompt's ids[:-1] into fresh units, chunk j of each prompt before chunk j + 1 of any; two: up to 2 chunks
    // in flight (the ANE lane's pipeline), else one at a time (the lane before g11)
    auto prefill_all = [&](mllm::PrefillLane& lane, const std::vector<std::vector<int>>& ps, bool two) {
        std::vector<Units> us;
        std::vector<mllm::PrefillWork> works;
        for (const std::vector<int>& ids : ps) us.push_back(units((int)ids.size() - 1));
        for (int pos = 0;; pos += S) {
            const size_t before = works.size();
            for (size_t p = 0; p < ps.size(); p++)
                if (const int n = (int)ps[p].size() - 1; pos < n)
                    works.push_back({(int)p, std::vector<int>(ps[p].begin() + pos, ps[p].begin() + std::min(pos + S, n)),
                                     pos, us[p].pages, us[p].state});
            if (works.size() == before) break;
        }
        size_t next = 0, finished = 0;
        while (finished < works.size()) {
            while (next < works.size() && (two ? lane.can_submit() : !lane.busy())) lane.submit(works[next++]);
            std::vector<mllm::PrefillWork> d = lane.poll();
            for (const mllm::PrefillWork& w : d) assert(w.start_pos == works[finished++].start_pos);  // in order
            if (d.empty()) lane.wait_oldest();
        }
        return us;
    };
    auto prefill = [&](mllm::PrefillLane& lane, const std::vector<int>& ids) { return prefill_all(lane, {ids}, false)[0]; };
    auto cos = [](const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0, aa = 0, bb = 0;
        for (size_t j = 0; j < a.size(); j++) dot += a[j] * b[j], aa += a[j] * a[j], bb += b[j] * b[j];
        return dot / std::sqrt(aa * bb);
    };
    // full layer g's K (0) or V (1), all heads, tokens from .. n
    auto kv = [&](const Units& u, int g, int which, int n, int from = 0) {
        std::vector<double> v;
        for (int h = 0; h < s.n_kv_heads; h++)
            for (int t = from; t < n; t++) {
                const uint16_t* p = (const uint16_t*)(arena + (size_t)u.pages[t / mem.T] * ar.unit_bytes +
                                                      mem.kv_offset(g, which, h, t % mem.T));
                for (int j = 0; j < s.head_dim; j++) v.push_back(bf16(p[j]));
            }
        return v;
    };
    auto state = [&](const Units& u, int l, bool delta) {  // linear layer l's conv (bf16) or delta (fp32) state
        std::vector<double> v;
        const size_t off = mem.state_offset(l) + (delta ? mem.conv_bytes : 0), len = delta ? mem.delta_bytes : mem.conv_bytes;
        for (size_t b = off; b < off + len; b += delta ? 4 : 2) {
            const uint8_t* p = arena + (size_t)u.state[b / ar.unit_bytes] * ar.unit_bytes + b % ar.unit_bytes;
            float f;
            if (delta) memcpy(&f, p, 4);
            else f = bf16(*(const uint16_t*)p);
            v.push_back(f);
        }
        return v;
    };
    prefill(ane_lane, prompts[0]), prefill(ane8_lane, prompts[0]);  // warm up (first predicts are slow)
    ar = mllm::Arena(ar.unit_bytes, ar.n_units);
    // ane.py's worst cos (ANE vs GPU prefill) on the same prompts; ours equals it to 6 digits (a margin of 2e-5 fails
    // keeping x fp16 between blocks, 1.6e-4 worse). test_ane.py's 0.999 (fp16) was measured on its own prompt: these
    // drift more, in Python too (the deepest linear layers' state). Then fp16 at chunk-boundary lengths n: the repeated
    // prompt's first n + 1 ids.
    struct Case {
        mllm::PrefillLane* lane;
        const char* name;
        std::vector<int> ids;
        double py;
    };
    std::vector<Case> cases;
    const double py[2][4] = {{0.998841, 0.998781, 0.998194, 0.996718}, {0.998534, 0.998563, 0.997660, 0.993370}};
    for (int pi = 0; pi < 4; pi++) cases.push_back({&ane_lane, "fp16", prompts[pi], py[0][pi]});
    for (int pi = 0; pi < 4; pi++) cases.push_back({&ane8_lane, "int8", prompts[pi], py[1][pi]});
    for (auto [n, c] : {std::pair{1, 0.999482}, {31, 0.999541}, {32, 0.999537}, {511, 0.996967}, {512, 0.996840},
                        {513, 0.996340}})
        cases.push_back({&ane_lane, "fp16", std::vector<int>(prompts[3].begin(), prompts[3].begin() + n + 1), c});
    auto worst_cos = [&](const Units& g, const Units& a, int n) {  // over every full layer's K / V, linear layer's states
        double worst = 1;
        for (int gi = 0, l = 0; l < s.n_layers(); l++)
            if (s.kinds[l] == "full") {
                double c0 = cos(kv(g, gi, 0, n), kv(a, gi, 0, n)), c1 = cos(kv(g, gi, 1, n), kv(a, gi, 1, n));
                if (getenv("VERBOSE")) printf("   layer %2d full   K %.6f V %.6f\n", l, c0, c1);
                worst = std::fmin(worst, std::fmin(c0, c1));
                gi++;
            } else {
                double c0 = cos(state(g, l, false), state(a, l, false)), c1 = cos(state(g, l, true), state(a, l, true));
                if (getenv("VERBOSE")) printf("   layer %2d linear conv %.6f state %.6f\n", l, c0, c1);
                worst = std::fmin(worst, std::fmin(c0, c1));
            }
        return worst;
    };
    auto same = [&](const Units& a, const Units& b, int n) {  // bit-exact K/V and states
        for (int g = 0, l = 0; l < s.n_layers(); l++)
            if (s.kinds[l] == "full") {
                if (kv(a, g, 0, n) != kv(b, g, 0, n) || kv(a, g, 1, n) != kv(b, g, 1, n)) return false;
                g++;
            } else if (state(a, l, false) != state(b, l, false) || state(a, l, true) != state(b, l, true)) {
                return false;
            }
        return true;
    };
    const int gpu_below = ane_lane.gpu_below;
    ane_lane.gpu_below = ane8_lane.gpu_below = 0;  // every chunk on the ANE, padded, as ane.py
    for (const auto& [lane, name, ids, py_cos] : cases) {
        const double min_cos = py_cos - 2e-5;
        const int n = (int)ids.size() - 1;
        auto t0 = std::chrono::steady_clock::now();
        Units g = prefill(gpu_lane, ids);
        auto t1 = std::chrono::steady_clock::now();
        Units a = prefill(*lane, ids);
        auto t2 = std::chrono::steady_clock::now();
        const double worst = worst_cos(g, a, n);
        auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        printf("c. %s prefill %3d tokens: worst K/V/state cos vs GPU lane %.6f (min %.6f); GPU %.1f ms, ANE %.1f ms\n",
               name, n, worst, min_cos, ms(t1 - t0), ms(t2 - t1));
        assert(worst >= min_cos);
        ar = mllm::Arena(ar.unit_bytes, ar.n_units);
    }
    ane_lane.gpu_below = ane8_lane.gpu_below = gpu_below;

    // c. two chunks in flight: K/V and state bit-exact with one at a time (same arithmetic, only the overlap moves),
    // for one request of 2 and 3 chunks (with and without a GPU tail) and for two requests prefilling at once (each
    // equals its solo run)
    {
        std::vector<int> longp;
        while (longp.size() < 1301) longp.insert(longp.end(), prompts[0].begin(), prompts[0].end());
        const std::vector<int> two(longp.begin(), longp.begin() + 1025), three(longp.begin(), longp.begin() + 1301);
        // 530 / 1100: the tail (18 / 76 tokens) runs on the GPU path, after its request's ANE chunk committed
        const std::vector<int> t530(longp.begin(), longp.begin() + 531), t1100(longp.begin(), longp.begin() + 1101);
        for (const auto* ids : {&two, &three, &t530, &t1100}) {
            Units a = prefill_all(ane_lane, {*ids}, false)[0], b = prefill_all(ane_lane, {*ids}, true)[0];
            const bool eq = same(a, b, (int)ids->size() - 1);
            printf("c. ANE lane, %zu tokens, 2 chunks in flight: K/V and state %s one at a time\n", ids->size() - 1,
                   eq ? "bit-exact with" : "DIFFER from");
            assert(eq);
            ar = mllm::Arena(ar.unit_bytes, ar.n_units);
        }
        Units x = prefill(ane_lane, three), y = prefill(ane_lane, prompts[3]);
        std::vector<Units> xy = prefill_all(ane_lane, {three, prompts[3]}, true);
        const bool eq = same(x, xy[0], (int)three.size() - 1) && same(y, xy[1], (int)prompts[3].size() - 1);
        printf("c. ANE lane, 1300 + 626 tokens prefilling at once: each %s its solo run\n", eq ? "equals" : "DIFFERS from");
        assert(eq);
        ar = mllm::Arena(ar.unit_bytes, ar.n_units);
        // noisy (the GPU may be shared): two 512 chunks back to back, best of 3
        double t[2] = {1e9, 1e9};
        for (int rep = 0; rep < 3; rep++)
            for (int k = 0; k < 2; k++) {
                auto t0 = std::chrono::steady_clock::now();
                prefill_all(ane_lane, {two}, k);
                t[k] = std::fmin(t[k], std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                ar = mllm::Arena(ar.unit_bytes, ar.n_units);
            }
        printf("c. ANE lane, 1024 tokens (noisy): one chunk at a time %.1f ms, two in flight %.1f ms\n", t[0], t[1]);
    }

    // d. tails on the GPU: an ANE chunk of fewer than gpu_below tokens runs on the GPU path (ws.S = n, no padding).
    // Bit-exact with the ANE lane's full chunks followed by the GPU lane's tail. Against GPU prefill, on the tail tokens'
    // K/V and the final state: >= 0.999 on random prompts (the repetitive one's ANE chunks drift more, as in ane.py).
    {
        std::vector<int> longp, randp;
        while (longp.size() < 1101) longp.insert(longp.end(), prompts[0].begin(), prompts[0].end());
        std::mt19937 rng(3);
        for (int j = 0; j < 1101; j++) randp.push_back((int)(rng() % s.vocab));
        for (const auto* src : {&longp, &randp})
            for (int len : {530, 1100}) {
                const std::vector<int> ids(src->begin(), src->begin() + len + 1);
                const int n = len, full = n / S * S;
                Units g = prefill(gpu_lane, ids);
                const int before = ane_lane.gpu_chunks;
                Units m = prefill(ane_lane, ids);
                assert(ane_lane.gpu_chunks == before + 1);
                ane_lane.gpu_below = 0;
                Units a = prefill(ane_lane, ids);
                ane_lane.gpu_below = gpu_below;
                Units c = units(n);
                for (int pos = 0; pos < n; pos += S) {
                    mllm::PrefillLane& l = pos < full ? ane_lane : gpu_lane;
                    l.submit({0, std::vector<int>(ids.begin() + pos, ids.begin() + std::min(pos + S, n)), pos, c.pages, c.state});
                    while (l.poll().empty()) l.wait_oldest();
                }
                auto tail_cos = [&](const Units& x) {  // {worst K/V cos over the tail tokens, worst final state cos}
                    double kvc = 1, st = 1;
                    for (int gi = 0, l = 0; l < s.n_layers(); l++)
                        if (s.kinds[l] == "full") {
                            for (int w : {0, 1}) kvc = std::fmin(kvc, cos(kv(g, gi, w, n, full), kv(x, gi, w, n, full)));
                            gi++;
                        } else {
                            for (bool d : {false, true}) st = std::fmin(st, cos(state(g, l, d), state(x, l, d)));
                        }
                    return std::pair{kvc, st};
                };
                const auto [mk, ms] = tail_cos(m);
                const auto [ak, as] = tail_cos(a);
                const bool eq = same(m, c, n);
                printf("d. %s %4d tokens, %2d-token tail on the GPU: cos vs GPU prefill, tail K/V %.6f, state %.6f (all-ANE "
                       "%.6f, %.6f); %s ANE chunks + GPU lane tail\n",
                       src == &longp ? "repeated" : "random  ", n, n - full, mk, ms, ak, as, eq ? "bit-exact with" : "DIFFERS from");
                assert(eq && (src == &longp || (mk >= 0.999 && ms >= 0.999)));
                ar = mllm::Arena(ar.unit_bytes, ar.n_units);
            }
    }

    // c. engine: greedy after ANE prefill
    auto greedy = [](mllm::Engine& e, const std::vector<int>& ids, int N) {
        const int id = e.submit(ids, {.max_tokens = N});
        std::vector<int> out;
        while (e.reqs.count(id))
            for (const mllm::Event& ev : e.step()) out.push_back(ev.token);
        return out;
    };
    mllm::Engine ea(golden::MODEL, 512 << 20, S, 32, 16, 4096, "ane", "fp16");
    for (int i = 0; i < 3; i++) {
        std::vector<int> got = greedy(ea, prompts[i], 64), want = ids_of("greedy/" + std::to_string(i));
        printf("c. engine, ANE prefill: prompt %d greedy %zu tokens %s golden\n", i, got.size(),
               got == want ? "equal" : "DIFFER from");
        assert(got == want);
    }
    // d. prefix-cache splits: a fresh 600-token prompt stops at its last full page (576, a snapshot point): 512 tokens
    // on the ANE, then 64 and 23 on the GPU
    {
        std::mt19937 rng(4);
        std::vector<int> ids(600);
        for (int& t : ids) t = (int)(rng() % s.vocab);
        const int before = ea.prefill.gpu_chunks;
        greedy(ea, ids, 1);
        assert(ea.prefill.gpu_chunks == before + 2);
        printf("d. engine: a 600-token prompt's prefix-cache split (64 tokens) and tail (23) ran on the GPU\n");
    }
    // d. a chunk ending at a snapshot point holds back its request's next chunk until poll has copied the state. Polled
    // late (40 ms between steps), both chunks would be done by then and the copy would hold the next chunk's state too.
    // The cached snapshot at 1088 (the last full page: chunks 512, 512, 64) equals a lane prefill of those tokens.
    {
        std::mt19937 rng(5);
        std::vector<int> ids(1100);
        for (int& t : ids) t = (int)(rng() % s.vocab);
        const int id = ea.submit(ids, {.max_tokens = 1});
        while (ea.reqs.count(id)) ea.step(), usleep(40000);
        const std::vector<int> head(ids.begin(), ids.begin() + 1089);
        mllm::PrefixCache::Match m = ea.cache->match(head);
        assert(m.S * mem.T == 1088 && m.snapshot.size() == mem.state_units);
        Units u = prefill(ane_lane, head);
        const uint8_t* ea_arena = (const uint8_t*)ea.mem.arena_buffer.contents;
        size_t diff = 0;
        for (size_t j = 0; j < m.snapshot.size(); j++)
            diff += memcmp(ea_arena + (size_t)m.snapshot[j] * ar.unit_bytes, arena + (size_t)u.state[j] * ar.unit_bytes,
                           ar.unit_bytes) != 0;
        printf("d. engine, late polls: the snapshot at 1088 %s a lane prefill of 1088 tokens\n", diff ? "DIFFERS from" : "equals");
        assert(!diff);
        ar = mllm::Arena(ar.unit_bytes, ar.n_units);
    }
    // decode on the engine thread while the ANE lane's thread prefills (both encode with GpuOps: the lane has its own
    // copy; sharing one crashed `mllm bench -b 8 --stream 512 --prefill ane`): prompt 0 still decodes golden's tokens
    // while 2-chunk prompts prefill alongside, one after another. ponytail: this didn't catch the shared-GpuOps race
    // (the bench did); 3 batched prompts + the stream row differ from golden by batch-size rounding, so not used.
    {
        const int id = ea.submit(prompts[0], {.max_tokens = 64});
        std::vector<int> out;
        int other = -1;
        while (ea.reqs.count(id)) {
            if (!ea.reqs.count(other)) other = ea.submit(prompts[3], {.max_tokens = 1});
            for (const mllm::Event& ev : ea.step())
                if (ev.req_id == id) out.push_back(ev.token);
        }
        ea.cancel(other);
        while (!ea.reqs.empty()) ea.step();
        assert(out == ids_of("greedy/0"));
        printf("c. engine: greedy 64 still golden with 2-chunk ANE prefills running alongside\n");
    }
    // random stress with ANE prefill: 10 requests of 2..1100 random tokens (up to 3 chunks), greedy and sampled, a few
    // cancels; every step returns, events keep coming (watchdog 30 s). Sharing the engine's GpuOps with the lane
    // thread crashed this within ~1.5 s.
    {
        std::mt19937 rng(1);
        int submitted = 0;
        auto last = std::chrono::steady_clock::now();
        while (submitted < 10 || !ea.reqs.empty()) {
            if (submitted < 10 && ea.reqs.size() < 16 && rng() % 3 == 0) {
                std::vector<int> ids(2 + rng() % 1099);
                for (int& t : ids) t = (int)(rng() % s.vocab);
                ea.submit(ids, {.max_tokens = 1 + (int)(rng() % 64), .temperature = rng() % 2 ? 0.7f : 0.0f});
                submitted++;
            }
            if (!ea.reqs.empty() && rng() % 300 == 0) ea.cancel(std::next(ea.reqs.begin(), rng() % ea.reqs.size())->first);
            std::vector<mllm::Event> ev = ea.step();
            for (const mllm::Event& e : ev) assert(e.token >= 0);
            if (!ev.empty()) last = std::chrono::steady_clock::now();
            assert(std::chrono::steady_clock::now() - last < std::chrono::seconds(30));
        }
        printf("c. engine: random stress with ANE prefill, %d requests, no failure\n", submitted);
    }
    // cancel and abort with two chunks in flight (one request's, or two requests'): every unit comes back, no hang,
    // and the lane works afterwards. Random prompts: no prefix-cache hits shorten them.
    {
        mllm::Arena& a = ea.mem.arena;
        auto all_free = [&] { return a.free_units + a.cache_only_units == a.n_units && a.reserved == 0 && !ea.prefill.busy(); };
        auto in_flight = [&] {
            int k = 0;
            for (auto& [_, r] : ea.reqs) k += r.state == mllm::State::PREFILL ? r.in_flight : 0;
            return k;
        };
        std::mt19937 rng(2);
        auto rand_ids = [&](int n) {
            std::vector<int> v(n);
            for (int& t : v) t = (int)(rng() % s.vocab);
            return v;
        };
        auto drain = [&](std::vector<mllm::Event>* evs) {
            auto t0 = std::chrono::steady_clock::now();
            while (!ea.reqs.empty()) {
                for (const mllm::Event& e : ea.step()) evs->push_back(e);
                assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30));
            }
        };
        assert(all_free());
        std::vector<mllm::Event> evs;
        for (int reqs : {1, 2}) {
            // 2 requests: x's one chunk and y's first (FIFO takes a request's second chunk before the next request)
            const int x = ea.submit(rand_ids(reqs == 2 ? 300 : 1300), {.max_tokens = 4});
            const int y = reqs == 2 ? ea.submit(rand_ids(1300), {.max_tokens = 4}) : -1;
            while (in_flight() < 2) ea.step();
            if (reqs == 2) assert(ea.reqs.at(x).in_flight == 1 && ea.reqs.at(y).in_flight == 1);
            ea.cancel(x);  // y (if any) runs on to its tokens
            evs.clear();
            drain(&evs);
            assert(all_free() && std::count_if(evs.begin(), evs.end(), [&](auto& e) { return e.req_id == y; }) == (reqs == 2 ? 4 : 0));
            printf("c. engine: cancel with 2 chunks in flight (%d request%s): all units back\n", reqs, reqs == 2 ? "s" : "");
        }
        for (int reqs : {1, 2}) {
            // 2 requests: x's one chunk and y's first (FIFO takes a request's second chunk before the next request)
            const int x = ea.submit(rand_ids(reqs == 2 ? 300 : 1300), {.max_tokens = 4});
            const int y = reqs == 2 ? ea.submit(rand_ids(1300), {.max_tokens = 4}) : -1;
            while (in_flight() < 2) ea.step();
            ea.prefill.fail_next = true;
            bool threw = false;
            evs.clear();
            auto t0 = std::chrono::steady_clock::now();
            while (!threw) {
                try {
                    for (const mllm::Event& e : ea.step()) evs.push_back(e);
                } catch (const std::runtime_error& e) {
                    threw = std::string(e.what()).find("injected") != std::string::npos;
                    assert(threw);
                }
                assert(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(30));
            }
            drain(&evs);
            for (int id : {x, y})
                if (id >= 0) assert(std::count_if(evs.begin(), evs.end(), [&](auto& e) { return e.req_id == id && e.token < 0; }) == 1);
            assert(all_free());
            printf("c. engine: abort with 2 chunks in flight (%d request%s): aborted, all units back\n", reqs, reqs == 2 ? "s" : "");
        }
        // the submit of a request's second chunk throws while its first is in the lane: aborted (token -1), the first
        // chunk finishes in the lane and poll ends the request
        {
            const int x = ea.submit(rand_ids(1300), {.max_tokens = 4});
            ea.prefill.fail_submit = 2;
            evs.clear();
            try {
                ea.step();
                assert(!"the second submit didn't throw");
            } catch (const std::runtime_error& e) {
                assert(std::string(e.what()).find("injected submit") != std::string::npos);
            }
            drain(&evs);
            assert(std::count_if(evs.begin(), evs.end(), [&](auto& e) { return e.req_id == x && e.token < 0; }) == 1);
            assert(all_free());
            printf("c. engine: a failed submit of a second chunk: aborted, all units back\n");
        }
        assert(greedy(ea, prompts[0], 64) == ids_of("greedy/0"));
        printf("c. engine: greedy 64 golden again after the aborts\n");
    }
    // 30: at token 31 GPU and ANE prefill pick different tokens (a near-tie after this repetitive prompt); ane.py's
    // ANE engine picks the same 32 tokens as this one
    std::vector<int> long_ane = greedy(ea, prompts[3], 30);
    {
        mllm::Engine eg(golden::MODEL, 512 << 20, S, 32, 16, 4096);
        std::vector<int> long_gpu = greedy(eg, prompts[3], 30);
        size_t same = 0;
        while (same < long_gpu.size() && long_ane[same] == long_gpu[same]) same++;
        printf("c. engine, %zu-token prompt: ANE prefill's greedy 30 equal GPU prefill's for %zu tokens\n",
               prompts[3].size(), same);
        assert(long_ane == long_gpu);
    }
    printf("PASS\n");
}
