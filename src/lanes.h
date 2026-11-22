// PrefillLane and DecodeLane: submit() / poll() (py/mllm/lanes.py). submit encodes the work into a command buffer on
// the lane's own MTLCommandQueue, commits it and returns; poll checks the lane's MTLSharedEvent (never blocks) and
// returns finished work. Both lanes live on the engine thread; prefill and decode overlap on the GPU through the two
// queues. Lanes never touch the allocator: every unit a work item needs is allocated before submit.
#pragma once
#import <Metal/Metal.h>

#include <deque>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "gpu.h"

namespace mllm {

struct PrefillWork {
    int req_id;
    std::vector<int> tokens;  // exact length (the GPU path never pads)
    int start_pos;
    std::vector<uint32_t> slots;        // KV pages covering start_pos + n_valid
    std::vector<uint32_t> state_slots;  // the request's state units
    int n_valid() const { return (int)tokens.size(); }
};

struct DecodeRow {
    int req_id, token, pos;  // token -1: the request's row in the step in flight, its argmax fed on the GPU
    std::vector<uint32_t> slots, state_slots;
    SampleParams sampler;
};

struct DecodeWork {
    uint64_t step;
    std::vector<DecodeRow> rows;
};

// One command buffer on q: f encodes into ws.enc. Waits; returns GPU seconds. For tests and bench-ops.
double encode_and_wait(id<MTLCommandQueue> q, Workspace& ws, const std::function<void()>& f);

class PrefillLane {
public:
    PrefillLane(GpuOps& ops, Memory& mem, int chunk, int max_ctx);
    bool busy() const { return w.has_value(); }  // one chunk in flight at a time
    void submit(PrefillWork work);
    std::vector<PrefillWork> poll();  // throws if the chunk's command buffer failed (the lane is free again)
    void wait_oldest() const;  // until the chunk in flight is done (or 1 s: a failed one never signals)

    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event;  // signaled with the chunk count

private:
    GpuOps& ops;
    Memory& mem;
    Workspace ws;
    StepTables tables;
    uint64_t n = 0;  // chunks committed
    id<MTLCommandBuffer> cb;
    std::optional<PrefillWork> w;
};

class DecodeLane {
public:
    DecodeLane(GpuOps& ops, Memory& mem, int max_batch, int max_ctx);
    // At most one step ahead: step N + 1 is encoded and committed while N runs. Its greedy rows can take N's argmax
    // on the GPU (DecodeRow::token -1), so B = 1 decode never waits for the CPU's encode and commit.
    bool can_submit() const { return q.size() < 2; }
    void submit(DecodeWork work);  // sets mem.arena.committed = work.step + 1
    // The oldest step's (req_id, token) once its event passed; throws if a step failed (every step in flight is
    // dropped: a later one may have fed on its tokens).
    std::vector<std::pair<int, int>> poll();
    uint64_t signaled() const { return event.signaledValue; }  // steps done: units pending at <= this are free
    bool busy() const { return !q.empty(); }
    void wait();  // until every step in flight is done
    void wait_oldest() const;  // until the oldest step in flight is done (or 1 s: a failed one never signals)

    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event;  // signaled with step + 1 when a step completes
    id<MTLFence> fence;        // step N + 1 waits for step N's last dispatch
    bool fail_next = false;    // tests: the next submit throws (engine abort path)

private:
    GpuOps& ops;
    Memory& mem;
    Workspace ws[2];  // by step parity: the CPU writes step N + 1's tokens and reads N's ids / logits meanwhile
    StepTables tables;
    struct InFlight {
        id<MTLCommandBuffer> cb;
        DecodeWork w;
    };
    std::deque<InFlight> q;
};

}  // namespace mllm
