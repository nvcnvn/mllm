// Memory checks.
// Arena storage: sizes and byte offsets equal memory.py's (golden "layout"); golden K/V and linear state written at
//   Python's offsets read back through the GPU at the C++ offsets.
// Workspace: after freeze(), a simulated step allocates nothing (counter), and an allocation asserts.
// Step tables: the copy the CPU fills is never the one an in-flight step reads; running 2 steps ahead asserts.
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <csignal>
#include <fstream>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "memory.h"

using mllm::ELEM;

// Runs f in a child; true if the child died of an assert.
template <class F>
static bool aborts(F f) {
    pid_t p = fork();
    if (p == 0) {
        freopen("/dev/null", "w", stderr);
        f();
        _exit(0);
    }
    int st = 0;
    waitpid(p, &st, 0);
    return WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
}

struct Copier {  // GPU copies (copy_u32) from the arena at given offsets, all in one command buffer
    id<MTLComputePipelineState> ps;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    Copier(id<MTLDevice> dev, id<MTLCommandQueue> q) {
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@METALLIB] error:&err];
        ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy_u32"] error:&err];
        assert(ps);
        cb = [q commandBuffer];
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
    }
    void copy(id<MTLBuffer> src, size_t src_off, id<MTLBuffer> dst, size_t dst_off, size_t bytes) {
        assert(src_off % 4 == 0 && dst_off % 4 == 0 && bytes % 4 == 0);
        [enc setBuffer:src offset:src_off atIndex:0];
        [enc setBuffer:dst offset:dst_off atIndex:1];
        [enc dispatchThreads:MTLSizeMake(bytes / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    void run() {
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        assert(cb.status == MTLCommandBufferStatusCompleted);
    }
};

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    std::ifstream f(d + "/spec.json");
    nlohmann::json g = nlohmann::json::parse(f);
    const auto& L = g["layout"];
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Memory mem(s, dev, L["arena_bytes"].get<size_t>(), g["page_tokens"].get<int>());
    const size_t unit = mem.arena.unit_bytes;

    // --- sizes and offsets equal Python's
    assert(unit == L["unit_bytes"].get<size_t>() && mem.arena.n_units == L["n_units"].get<uint32_t>());
    assert(mem.state_units == L["state_units"].get<uint32_t>());
    assert(mem.conv_bytes == L["conv_bytes"].get<size_t>() && mem.delta_bytes == L["delta_bytes"].get<size_t>());
    assert(mem.arena_buffer.length == (size_t)mem.arena.n_units * unit);
    auto st = L["kv_strides"].get<std::vector<size_t>>();  // numpy kv[slot, g, kv, head, token, dim], bytes
    assert(st.size() == 6 && st[0] == unit && st[5] == ELEM);
    auto py_kv = [&](size_t slot, int gi, int kv, int h, int t) {
        return slot * st[0] + gi * st[1] + kv * st[2] + h * st[3] + t * st[4];
    };
    const int G = (int)s.full_layers().size(), H = s.n_kv_heads, T = mem.T, D = s.head_dim;
    for (int gi = 0; gi < G; gi++)
        for (int kv = 0; kv < 2; kv++)
            for (int h = 0; h < H; h++)
                for (int t = 0; t < T; t++) assert(mem.kv_offset(gi, kv, h, t) == py_kv(0, gi, kv, h, t));
    for (const auto& p : L["kv"])
        assert(p[0].get<size_t>() * unit + mem.kv_offset(p[1], p[2], p[3], p[4]) == p[5].get<size_t>());
    for (const auto& p : L["state"])
        assert(mem.state_offset(p[0]) == p[1].get<size_t>() && mem.state_offset(p[0]) + mem.conv_bytes == p[2].get<size_t>());
    assert(L["state"].size() == s.linear_layers().size());

    // --- golden K/V and state: CPU writes at Python's offsets, the GPU reads at C++'s
    mllm::Safetensors tr(d + "/trace.safetensors");
    uint8_t* A = (uint8_t*)mem.arena_buffer.contents;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    Copier gpu(dev, q);
    char key[64];
    auto at = [&](const char* fmt, int l) -> const mllm::Tensor& {
        snprintf(key, sizeof key, fmt, l);
        return tr.at(key);
    };
    const int n_tok = (int)at("decode2/r0/L%02d/k", s.full_layers()[0]).shape[1];
    const uint32_t n_p = mem.units_for(n_tok), n_units = mem.arena.n_units;
    std::vector<uint32_t> pages, state;  // scattered, disjoint
    for (uint32_t j = 0; j < n_p; j++) pages.push_back((j * 37 + 11) % n_units);
    for (uint32_t j = 0; j < mem.state_units; j++) state.push_back(n_units - 1 - 3 * j);
    const size_t row = D * ELEM, run = T * row;  // one (page, g, kv, head) run
    id<MTLBuffer> kv_out = [dev newBufferWithLength:(size_t)G * 2 * H * n_p * run options:MTLResourceStorageModeShared];
    for (int gi = 0; gi < G; gi++)
        for (int kv = 0; kv < 2; kv++) {
            const mllm::Tensor& t = at(kv ? "decode2/r0/L%02d/v" : "decode2/r0/L%02d/k", s.full_layers()[gi]);
            assert(t.dtype == "BF16" && t.shape == std::vector<int64_t>({H, n_tok, D}));
            for (int h = 0; h < H; h++) {
                for (int i = 0; i < n_tok; i++)
                    memcpy(A + py_kv(pages[i / T], gi, kv, h, i % T), tr.data(t) + ((size_t)h * n_tok + i) * row, row);
                for (uint32_t p = 0; p < n_p; p++)
                    gpu.copy(mem.arena_buffer, pages[p] * unit + mem.kv_offset(gi, kv, h, 0), kv_out,
                             ((((size_t)gi * 2 + kv) * H + h) * n_p + p) * run, run);
            }
        }
    // state span byte b lives at state[b / unit] * unit + b % unit (memory.py load_state concatenates the slots)
    auto span_write = [&](size_t off, const uint8_t* src, size_t n) {
        for (size_t k; n; off += k, src += k, n -= k) {
            k = std::min(n, unit - off % unit);
            memcpy(A + state[off / unit] * unit + off % unit, src, k);
        }
    };
    for (const auto& p : L["state"]) {  // Python's offsets
        int l = p[0];
        const mllm::Tensor &conv = at("decode2/r0/L%02d/conv", l), &delta = at("decode2/r0/L%02d/delta", l);
        assert(conv.nbytes == mem.conv_bytes && delta.nbytes == mem.delta_bytes);
        span_write(p[1], tr.data(conv), conv.nbytes);
        span_write(p[2], tr.data(delta), delta.nbytes);
    }
    id<MTLBuffer> st_out = [dev newBufferWithLength:mem.state_units * unit options:MTLResourceStorageModeShared];
    for (uint32_t j = 0; j < mem.state_units; j++) gpu.copy(mem.arena_buffer, state[j] * unit, st_out, j * unit, unit);
    gpu.run();

    const uint8_t* K = (const uint8_t*)kv_out.contents;
    for (int gi = 0; gi < G; gi++)
        for (int kv = 0; kv < 2; kv++) {
            const mllm::Tensor& t = at(kv ? "decode2/r0/L%02d/v" : "decode2/r0/L%02d/k", s.full_layers()[gi]);
            for (int h = 0; h < H; h++)
                assert(!memcmp(K + (((size_t)gi * 2 + kv) * H + h) * n_p * run, tr.data(t) + (size_t)h * n_tok * row,
                               n_tok * row));
        }
    const uint8_t* S = (const uint8_t*)st_out.contents;
    for (int l : s.linear_layers()) {
        const mllm::Tensor &conv = at("decode2/r0/L%02d/conv", l), &delta = at("decode2/r0/L%02d/delta", l);
        assert(!memcmp(S + mem.state_offset(l), tr.data(conv), conv.nbytes));
        assert(!memcmp(S + mem.state_offset(l) + mem.conv_bytes, tr.data(delta), delta.nbytes));
    }

    // --- workspace + step tables at startup, then nothing allocates
    const int chunk = g["chunk"], max_batch = 8;
    mllm::Workspace ws = mem.workspace(chunk, max_batch);
    assert(ws.x.length >= (size_t)chunk * s.hidden * ELEM && ws.logits.length == (size_t)max_batch * s.vocab * ELEM);
    mllm::StepTables tables = mem.step_tables(max_batch);
    mem.freeze();
    const int n_alloc = mem.allocations;
    assert(n_alloc == 1 + 12 + 2);  // arena, workspace fields, table copies

    // a run of steps: the GPU finishes steps with varying lag; the CPU fills step N's table once step N-2 is done
    mllm::Arena& a = mem.arena;
    uint64_t signaled = 0;
    std::vector<uint32_t> held;
    for (uint64_t step = 1; step <= 100; step++) {
        if (step % 3 == 0) signaled = step - 1;      // GPU caught up
        while (signaled + 2 < step) signaled++;      // CPU waits on the event
        id<MTLBuffer> w = tables.begin(step, signaled);
        for (uint64_t inflight = signaled + 1; inflight < step; inflight++) assert(w != tables.copies[inflight % 2]);
        a.retire(signaled);
        if (a.admit(1)) held.push_back(a.alloc());
        uint32_t* t = (uint32_t*)w.contents;  // one row holding every unit allocated so far
        t[0] = 1, t[1] = 0, t[2] = (uint32_t)held.size();
        std::copy(held.begin(), held.end(), t + 3);
        if (step % 10 == 0) {
            a.committed = step;
            a.release(held);
            held.clear();
        }
        mem.request_residency();
    }
    assert(mem.allocations == n_alloc);
    assert(aborts([&] { mem.buffer(16); }));              // allocation after startup
    assert(aborts([&] { tables.begin(10, 7); }));         // 2 steps ahead: step 9 may still read copy 0
    assert(!aborts([&] { tables.begin(10, 8); }));
    printf("layout = memory.py; %d K/V pages + %u state units read back through the GPU; %d buffers, none after "
           "startup\nPASS\n",
           G * 2 * H * (int)n_p, mem.state_units, n_alloc);
}
