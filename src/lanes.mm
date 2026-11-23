#include "lanes.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace mllm {

static std::runtime_error failed(const char* lane, id<MTLCommandBuffer> cb) {
    return std::runtime_error(std::string(lane) + ": command buffer failed" +
                              (cb.error ? std::string(": ") + cb.error.localizedDescription.UTF8String : ""));
}

// One command buffer on q: f encodes into ws.enc (after fence's last update, then updating it, if a fence is given).
// Commits it.
static id<MTLCommandBuffer> commit(id<MTLCommandQueue> q, Workspace& ws, const std::function<void()>& f, bool concurrent,
                                   id<MTLFence> fence = nil) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    ws.enc = [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    if (fence) [ws.enc waitForFence:fence];
    try {
        f();
    } catch (...) {  // the ANE's predict failed midway: close the encoder, drop the command buffer
        [ws.enc endEncoding];
        ws.enc = nil;
        throw;
    }
    if (fence) [ws.enc updateFence:fence];
    [ws.enc endEncoding];
    ws.enc = nil;
    [cb commit];
    return cb;
}

double encode_and_wait(id<MTLCommandQueue> q, Workspace& ws, const std::function<void()>& f, bool concurrent) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = commit(q, ws, f, concurrent);
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) throw failed("encode_and_wait", cb);
        return cb.GPUEndTime - cb.GPUStartTime;
    }
}

static id<MTLCommandQueue> lane_queue(Memory& mem) {
    id<MTLCommandQueue> q = [mem.device newCommandQueue];
    [q addResidencySet:mem.residency];
    return q;
}

// lanes.py _layers: dense(i) then attn(i) for every layer, into ws.enc
static void layers(GpuOps& ops, Workspace& ws, int L) {
    for (int i = 0; i < L; i++) {
        ops.dense(i, ws);
        ops.attn(i, ws);
    }
}

PrefillLane::PrefillLane(GpuOps& ops, Memory& mem, int chunk, int max_ctx, AneOps* ane)
    : queue(lane_queue(mem)), event([mem.device newSharedEvent]), ops(ane ? ane->gpu : ops), mem(mem),
      ws{ops.workspace(chunk, 1, max_ctx), {}}, tables(mem.step_tables(1, max_ctx)), ane(ane),
      fence(ane ? [mem.device newFence] : nil) {
    if (ane && ane->S != chunk) throw std::invalid_argument("ANE prefill: chunk must equal the blocks' compiled S");
    if (ane) {
        const Workspace own = mem.workspace(chunk, 1);
        ws[1] = ws[0];
        ws[1].tokens = own.tokens, ws[1].x = own.x, ws[1].o = own.o, ws[1].q = own.q, ws[1].k = own.k, ws[1].v = own.v;
        ws[1].gate = own.gate, ws[1].qkv = own.qkv, ws[1].z = own.z, ws[1].b = own.b, ws[1].a = own.a;
    }
    if (ane) thread = std::thread(&PrefillLane::run, this);
}

PrefillLane::~PrefillLane() {
    if (!thread.joinable()) return;
    {
        std::lock_guard l(m);
        stop = true;
    }
    cv.notify_one();
    thread.join();  // after the chunks it started, if any
}

// The ANE thread. Turns take the started chunks in rotation, oldest first. A turn waits only for the command buffer
// its own predict needs (its chunk's last one: it wrote the block's inputs), so the ANE runs one chunk's dense(i)
// while the GPU runs the other's attn. The older chunk is never behind, so a request's later chunk commits each
// layer's attn after its earlier chunk's.
void PrefillLane::run() {
    std::deque<Chunk> active;  // up to 2, in turn order
    for (;;) {
        @autoreleasepool {
            {
                std::unique_lock l(m);
                cv.wait(l, [&] { return stop || !active.empty() || !todo.empty(); });
                if (stop && active.empty()) return;
                while (!stop && active.size() < 2 && !todo.empty()) {
                    if (err) {  // a failed chunk dropped the ones in flight: drop later ones too, until poll sees it
                        done = todo.front().k;
                        event.signaledValue = done;
                    } else {
                        active.push_back(todo.front());
                    }
                    todo.pop_front();
                }
            }
            if (active.empty()) continue;
            Chunk c = active.front();
            active.pop_front();
            try {
                if (turn(c, active)) active.push_back(c);
            } catch (...) {  // drop every started chunk once the GPU is done with it
                active.push_back(c);
                uint64_t last = 0;
                for (const Chunk& a : active) {
                    if (a.cb) [a.cb waitUntilCompleted];
                    last = std::max(last, a.k);
                }
                active.clear();
                std::lock_guard l(m);
                err = std::current_exception();
                done = last;
                event.signaledValue = last;  // wakes wait_oldest; poll trusts done
            }
        }
    }
}

// One step of chunk c: start it, run one layer, or finish it. False when it's done.
bool PrefillLane::turn(Chunk& c, const std::deque<Chunk>& others) {
    const int L = mem.spec.n_layers(), set = (int)(c.k % 2);
    Workspace& ws = this->ws[set];
    auto wait = [&] {
        [c.cb waitUntilCompleted];
        if (c.cb.status != MTLCommandBufferStatusCompleted) throw failed("PrefillLane", c.cb);
    };
    // ponytail: a GPU chunk is one command buffer of all 24 layers (~10-50 ms); the other chunk's per-layer buffers
    // queue behind it, so the ANE idles that long. Commit it per layer (or a few) to interleave if tails show in TTFT.
    if (c.layer < 0 && c.gpu) {
        if (std::any_of(others.begin(), others.end(), [&](const Chunk& o) { return o.k < c.k && o.req == c.req && o.layer < L; }))
            return true;
        c.cb = commit(queue, ws, [&] { layers(ops, ws, L); }, false, fence);  // as the GPU lane: no o_proj / FFN of L - 1
        c.layer = L;
        return true;
    }
    if (c.layer < 0) {
        c.cb = commit(queue, ws, [&] { ane->inputs(0, ws, set); }, false, fence);
        c.layer = 0;
        return true;
    }
    if (c.layer == L) {  // done in submit order: the event counts chunks
        if (std::any_of(others.begin(), others.end(), [&](const Chunk& o) { return o.k < c.k; })) return true;
        wait();
        std::lock_guard l(m);
        done = c.k;
        event.signaledValue = c.k;  // wakes wait_oldest; poll trusts done
        return false;
    }
    wait();
    if (fail_next.exchange(false)) throw std::runtime_error("PrefillLane: injected failure");
    const int i = c.layer;
    c.cb = commit(queue, ws, [&] {  // K/V and state only: the last layer's o_proj + FFN never run
        ane->dense(i, ws, set);
        ops.attn(i, ws);
        if (i + 1 < L) ane->inputs(i + 1, ws, set);
    }, false, fence);
    c.layer++;
    return true;
}

void PrefillLane::submit(PrefillWork work) {
    @autoreleasepool {
        assert(can_submit() && work.n_valid() > 0 && work.n_valid() <= ws[0].rows);
        if (fail_submit && --fail_submit == 0) throw std::runtime_error("PrefillLane: injected submit failure");
        Workspace& ws = this->ws[ane ? (n + 1) % 2 : 0];
        ws.B = 1, ws.S = work.n_valid();
        std::copy(work.tokens.begin(), work.tokens.end(), (int32_t*)ws.tokens.contents);
        ws.table = tables.begin(n + 1, event.signaledValue);
        tables.write(ws.table, {{work.start_pos, work.n_valid(), work.slots, work.state_slots}});
        n++;
        if (ane) {
            const bool gpu = work.n_valid() < gpu_below;
            gpu_chunks += gpu;
            {
                std::lock_guard l(m);
                todo.push_back({n, work.req_id, gpu, -1, nil});
            }
            cv.notify_one();
        } else {
            cb = [queue commandBuffer];
            ws.enc = [cb computeCommandEncoder];
            // K/V and state only: the last layer's o_proj + FFN and lm_head never run
            layers(ops, ws, mem.spec.n_layers());
            [ws.enc endEncoding];
            ws.enc = nil;
            [cb encodeSignalEvent:event value:n];
            [cb commit];
        }
        q.push_back(std::move(work));
    }
}

std::vector<PrefillWork> PrefillLane::poll() {
    @autoreleasepool {
        if (q.empty()) return {};
        uint64_t d;
        if (ane) {
            std::exception_ptr e;
            {
                std::lock_guard l(m);
                d = done;
                if (err && d < n) return {};  // the thread is still dropping chunks
                e = std::exchange(err, nullptr);
            }
            if (e) {
                q.clear();
                std::rethrow_exception(e);
            }
        } else {
            if (cb.status == MTLCommandBufferStatusError) {
                auto e = failed("PrefillLane", cb);
                cb = nil, q.clear();
                throw e;
            }
            d = event.signaledValue;
        }
        std::vector<PrefillWork> out;
        while (!q.empty() && n - q.size() + 1 <= d) out.push_back(std::move(q.front())), q.pop_front();
        if (q.empty()) cb = nil;
        return out;
    }
}

DecodeLane::DecodeLane(GpuOps& ops, Memory& mem, int max_batch, int max_ctx)
    : queue(lane_queue(mem)), event([mem.device newSharedEvent]), fence([mem.device newFence]), ops(ops), mem(mem),
      ws{ops.workspace(1, max_batch, max_ctx), ops.workspace(1, max_batch, max_ctx)},
      tables(mem.step_tables(max_batch, max_ctx)) {}

void DecodeLane::submit(DecodeWork work) {
    @autoreleasepool {
        if (fail_next) {
            fail_next = false;
            throw std::runtime_error("DecodeLane: injected failure");
        }
        Workspace& ws = this->ws[work.step % 2];
        const int B = (int)work.rows.size();
        assert(can_submit() && B > 0 && B <= ws.rows && (q.empty() || q.back().w.step + 1 == work.step));
        ws.B = B, ws.S = 1;
        std::vector<StepTables::Row> rows;
        std::vector<int> src(B, -1);  // fed rows: their row in the step in flight
        bool fed = false;
        for (int b = 0; b < B; b++) {
            const DecodeRow& r = work.rows[b];
            if (r.token < 0) {
                assert(!q.empty());
                const auto& prev = q.back().w.rows;
                auto it = std::find_if(prev.begin(), prev.end(), [&](const DecodeRow& p) { return p.req_id == r.req_id; });
                assert(it != prev.end() && it->pos + 1 == r.pos);
                src[b] = (int)(it - prev.begin()), fed = true;
            }
            ((int32_t*)ws.tokens.contents)[b] = r.token;
            rows.push_back({r.pos, 1, r.slots, r.state_slots});
        }
        ws.table = tables.begin(work.step + 1, event.signaledValue);
        tables.write(ws.table, rows);
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        ws.enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        // Defensive today: a row in both step N and N + 1 is fed, and feeding reads N's ids, a tracked buffer, so
        // Metal already orders the two encoders (and with them the untracked arena's K/V and state). Needed if
        // workspace buffers become untracked. A fence, not encodeWaitForEvent: that cost B = 1 ~3.5% (297 vs 287).
        [ws.enc waitForFence:fence];
        if (fed) ops.feed_tokens(ws, this->ws[(work.step + 1) % 2].ids, src);
        const int L = mem.spec.n_layers();
        layers(ops, ws, L);
        ops.dense(L, ws);
        ops.lm_head(ws);
        [ws.enc updateFence:fence];
        [ws.enc endEncoding];
        ws.enc = nil;
        [cb encodeSignalEvent:event value:work.step + 1];
        [cb commit];
        mem.arena.committed = work.step + 1;  // units released from now on wait for this step
        q.push_back({cb, std::move(work)});
    }
}

std::vector<std::pair<int, int>> DecodeLane::poll() {
    @autoreleasepool {
        if (q.empty()) return {};
        for (const InFlight& f : q)
            if (f.cb.status == MTLCommandBufferStatusError) {
                auto e = failed("DecodeLane", f.cb);
                wait();
                // a failed command buffer may skip its signal: units released at its step would stay pending for
                // good. Everything committed has completed now, so mark it done (one queue, in order).
                event.signaledValue = std::max(event.signaledValue, q.back().w.step + 1);
                q.clear();
                throw e;
            }
        const DecodeWork& w = q.front().w;
        if (event.signaledValue < w.step + 1) return {};
        std::vector<SampleParams> params;
        for (const DecodeRow& r : w.rows) params.push_back(r.sampler);
        std::vector<int> ids = ops.sample(ws[w.step % 2], params);
        std::vector<std::pair<int, int>> done;
        for (size_t b = 0; b < ids.size(); b++) done.emplace_back(w.rows[b].req_id, ids[b]);
        q.pop_front();
        return done;
    }
}

// The event's value of the oldest work in flight, not signaledValue + 1: that may already have passed since the poll.
void PrefillLane::wait_oldest() const {
    if (busy()) [event waitUntilSignaledValue:n - q.size() + 1 timeoutMS:1000];
}

void DecodeLane::wait_oldest() const {
    if (!q.empty()) [event waitUntilSignaledValue:q.front().w.step + 1 timeoutMS:1000];
}

void DecodeLane::wait() {
    if (!q.empty()) [q.back().cb waitUntilCompleted];
}

}  // namespace mllm
