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
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>

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
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), ane.surfaces.size());
    mllm::Workspace ws = ops.workspace(S, 1, S);
    const int max_ctx = 1024;
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
        mllm::encode_and_wait(q, ws, [&] { ane.inputs(i, ws); });
        // padding: every input surface is zero past n
        for (NSString* k in b.in.featureNames) {
            MLMultiArray* a = [b.in featureValueForName:k].multiArrayValue;
            CVPixelBufferRef pb = a.pixelBuffer;
            CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
            const uint16_t* p = (const uint16_t*)CVPixelBufferGetBaseAddress(pb);
            const size_t row = CVPixelBufferGetBytesPerRow(pb) / 2, C = CVPixelBufferGetHeight(pb);
            for (size_t c = 0; c < C; c++)
                for (int t = n; t < S; t++) assert(p[c * row + t] == 0);
            CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        }
        mllm::encode_and_wait(q, ws, [&] { ane.dense(i, ws); });

        // the same block, inputs copied on the CPU into plain arrays, outputs in Core ML's own buffers
        NSMutableDictionary* feed = [NSMutableDictionary new];
        for (NSString* k in b.in.featureNames) {
            CVPixelBufferRef pb = [b.in featureValueForName:k].multiArrayValue.pixelBuffer;
            MLMultiArray* a = [[MLMultiArray alloc] initWithShape:[b.in featureValueForName:k].multiArrayValue.shape
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
            NSString* k = [b.opts.outputBackings allKeysForObject:io.surface->array].firstObject;
            std::string name;
            for (const auto& [f, field] : FIELDS)
                if (ws.*io.field == ws.*field) name = f;
            MLMultiArray* r = [ref featureValueForName:k].multiArrayValue;
            const int C = io.surface->C, inner = r.shape[2].intValue;
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
    auto prefill = [&](mllm::PrefillLane& lane, const std::vector<int>& ids) {  // ids[:-1] into fresh units
        const int n = (int)ids.size() - 1;
        Units u;
        assert(ar.admit(mem.units_for(n) + mem.state_units));
        for (uint32_t j = 0; j < mem.units_for(n); j++) u.pages.push_back(ar.alloc());
        for (uint32_t j = 0; j < mem.state_units; j++) u.state.push_back(ar.alloc());
        for (uint32_t x : u.state) memset(arena + (size_t)x * ar.unit_bytes, 0, ar.unit_bytes);
        for (int pos = 0; pos < n; pos += S) {
            lane.submit({0, std::vector<int>(ids.begin() + pos, ids.begin() + std::min(pos + S, n)), pos, u.pages, u.state});
            while (lane.poll().empty()) lane.wait_oldest();
        }
        return u;
    };
    auto cos = [](const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0, aa = 0, bb = 0;
        for (size_t j = 0; j < a.size(); j++) dot += a[j] * b[j], aa += a[j] * a[j], bb += b[j] * b[j];
        return dot / std::sqrt(aa * bb);
    };
    auto kv = [&](const Units& u, int g, int which, int n) {  // full layer g's K (0) or V (1), all heads, n tokens
        std::vector<double> v;
        for (int h = 0; h < s.n_kv_heads; h++)
            for (int t = 0; t < n; t++) {
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
    {
        for (const auto& [lane, name, ids, py_cos] : cases) {
            const double min_cos = py_cos - 2e-5;
            const int n = (int)ids.size() - 1;
            auto t0 = std::chrono::steady_clock::now();
            Units g = prefill(gpu_lane, ids);
            auto t1 = std::chrono::steady_clock::now();
            Units a = prefill(*lane, ids);
            auto t2 = std::chrono::steady_clock::now();
            double worst = 1;
            int gi = 0;
            for (int l = 0; l < s.n_layers(); l++)
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
            auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
            printf("c. %s prefill %3d tokens: worst K/V/state cos vs GPU lane %.6f (min %.6f); GPU %.1f ms, ANE %.1f ms\n",
                   name, n, worst, min_cos, ms(t1 - t0), ms(t2 - t1));
            assert(worst >= min_cos);
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
    mllm::Engine ea(golden::MODEL, 512 << 20, S, 32, 16, 4096, "ane");
    for (int i = 0; i < 3; i++) {
        std::vector<int> got = greedy(ea, prompts[i], 64), want = ids_of("greedy/" + std::to_string(i));
        printf("c. engine, ANE prefill: prompt %d greedy %zu tokens %s golden\n", i, got.size(),
               got == want ? "equal" : "DIFFER from");
        assert(got == want);
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
