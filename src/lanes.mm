#include "lanes.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace mllm {

static std::runtime_error failed(const char* lane, id<MTLCommandBuffer> cb) {
    return std::runtime_error(std::string(lane) + ": command buffer failed" +
                              (cb.error ? std::string(": ") + cb.error.localizedDescription.UTF8String : ""));
}

double encode_and_wait(id<MTLCommandQueue> q, Workspace& ws, const std::function<void()>& f) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        ws.enc = [cb computeCommandEncoder];
        try {
            f();
        } catch (...) {  // the ANE's predict failed midway: close the encoder, drop the command buffer
            [ws.enc endEncoding];
            ws.enc = nil;
            throw;
        }
        [ws.enc endEncoding];
        ws.enc = nil;
        [cb commit];
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
      ws(ops.workspace(chunk, 1, max_ctx)), tables(mem.step_tables(1)), ane(ane) {
    if (ane && ane->S != chunk) throw std::invalid_argument("ANE prefill: chunk must equal the blocks' compiled S");
    if (ane) thread = std::thread(&PrefillLane::run, this);
}

PrefillLane::~PrefillLane() {
    if (!thread.joinable()) return;
    {
        std::lock_guard l(m);
        stop = true;
    }
    cv.notify_one();
    thread.join();  // after the chunk in flight, if any
}

// The ANE thread. ponytail: the CPU waits for the GPU every layer and for the ANE every block; pipelining
// ANE dense(i + 1) with GPU attn(i) (two chunks in flight) is the next step if prefill speed matters.
void PrefillLane::run() {
    for (;;) {
        uint64_t k;
        {
            std::unique_lock l(m);
            cv.wait(l, [&] { return stop || todo; });
            if (stop) return;
            k = std::exchange(todo, 0);
        }
        try {
            encode_and_wait(queue, ws, [&] { ane->inputs(0, ws); });
            const int L = mem.spec.n_layers();
            for (int i = 0; i < L; i++)  // K/V and state only: the last layer's o_proj + FFN never run
                encode_and_wait(queue, ws, [&] {
                    ane->dense(i, ws);
                    ops.attn(i, ws);
                    if (i + 1 < L) ane->inputs(i + 1, ws);
                });
        } catch (...) {
            std::lock_guard l(m);
            err = std::current_exception();
        }
        {
            std::lock_guard l(m);
            done = k;
        }
        event.signaledValue = k;  // wakes wait_oldest; poll trusts done
    }
}

void PrefillLane::submit(PrefillWork work) {
    @autoreleasepool {
        assert(!busy() && work.n_valid() > 0 && work.n_valid() <= ws.rows);
        ws.B = 1, ws.S = work.n_valid();
        std::copy(work.tokens.begin(), work.tokens.end(), (int32_t*)ws.tokens.contents);
        ws.table = tables.begin(n + 1, event.signaledValue);
        tables.write(ws.table, {{work.start_pos, work.n_valid(), work.slots, work.state_slots}});
        if (ane) {
            w = std::move(work);
            {
                std::lock_guard l(m);
                todo = ++n;
            }
            cv.notify_one();
            return;
        }
        cb = [queue commandBuffer];
        ws.enc = [cb computeCommandEncoder];
        // K/V and state only: the last layer's o_proj + FFN and lm_head never run
        layers(ops, ws, mem.spec.n_layers());
        [ws.enc endEncoding];
        ws.enc = nil;
        [cb encodeSignalEvent:event value:++n];
        [cb commit];
        w = std::move(work);
    }
}

std::vector<PrefillWork> PrefillLane::poll() {
    @autoreleasepool {
        if (!busy()) return {};
        if (ane) {
            std::exception_ptr e;
            {
                std::lock_guard l(m);
                if (done < n) return {};
                e = std::exchange(err, nullptr);
            }
            if (e) {
                w.reset();
                std::rethrow_exception(e);
            }
        }
        if (cb.status == MTLCommandBufferStatusError) {
            auto e = failed("PrefillLane", cb);
            cb = nil, w.reset();
            throw e;
        }
        if (event.signaledValue < n) return {};
        std::vector<PrefillWork> out;
        out.push_back(std::move(*w));
        cb = nil, w.reset();
        return out;
    }
}

DecodeLane::DecodeLane(GpuOps& ops, Memory& mem, int max_batch, int max_ctx)
    : queue(lane_queue(mem)), event([mem.device newSharedEvent]), fence([mem.device newFence]), ops(ops), mem(mem),
      ws{ops.workspace(1, max_batch, max_ctx), ops.workspace(1, max_batch, max_ctx)},
      tables(mem.step_tables(max_batch)) {}

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
        ws.enc = [cb computeCommandEncoder];
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
    if (busy()) [event waitUntilSignaledValue:n timeoutMS:1000];
}

void DecodeLane::wait_oldest() const {
    if (!q.empty()) [event waitUntilSignaledValue:q.front().w.step + 1 timeoutMS:1000];
}

void DecodeLane::wait() {
    if (!q.empty()) [q.back().cb waitUntilCompleted];
}

}  // namespace mllm
