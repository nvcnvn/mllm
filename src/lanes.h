// PrefillLane and DecodeLane: submit() / poll() (py/mllm/lanes.py). submit encodes the work into command buffers on
// the lane's own MTLCommandQueue (prefill: a chain of 8-layer buffers per chunk), commits and returns; poll checks the
// lane's MTLSharedEvent (never blocks) and returns finished work. Both lanes live on the engine thread; the GPU
// time-shares the two queues' command buffers in commit order (g15), it doesn't run them side by side. Lanes never touch the allocator: every unit a work item needs is allocated before submit.
// With AneOps, PrefillLane runs chunks on a thread of its own (Core ML's predict blocks): per layer, ANE dense(i)
// then GPU attn(i) on the lane's queue, two chunks in flight so the ANE runs one's dense while the GPU runs the
// other's attn; the engine sees the same submit / poll / event as on the GPU.
#pragma once
#import <Metal/Metal.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "ane.h"
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

// One command buffer on q: f encodes into ws.enc (concurrent: as the decode lane's). Waits; returns GPU seconds.
double encode_and_wait(id<MTLCommandQueue> q, Workspace& ws, const std::function<void()>& f, bool concurrent = false);

class PrefillLane {
public:
    // ane: dense on the Neural Engine (chunk must be its compiled S), attn with ane->gpu; nullptr: all on the GPU
    PrefillLane(GpuOps& ops, Memory& mem, int chunk, int max_ctx, AneOps* ane = nullptr);
    ~PrefillLane();
    bool busy() const { return !q.empty(); }
    // ANE: two chunks in flight, of any requests (a request's later chunk runs each layer's attn after its earlier
    // chunk's: one queue, one fence). GPU: one.
    bool can_submit() const { return q.size() < (ane ? 2u : 1u); }
    void submit(PrefillWork work);
    // the chunks done, in submit order. Throws if one failed: every chunk in flight is dropped (after the thread
    // let go of them), the lane is free again.
    std::vector<PrefillWork> poll();
    void wait_oldest() const;  // until the oldest chunk in flight is done (or 1 s: a failed one never signals)

    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event;  // signaled with the chunk count
    std::atomic<bool> fail_next = false;  // tests: the ANE thread's next layer throws (a failed predict)
    int fail_submit = 0;                  // tests: the fail_submit-th submit from now throws
    // ANE: a chunk of fewer tokens runs on the GPU path instead (ws.S = n, no padding to S: tails and prefix-cache
    // splits). The crossover of GPU prefill time at n against one ANE chunk (reports/mlx-lm baseline.md).
    int gpu_below = 256;
    int gpu_chunks = 0;  // ANE: chunks submitted to the GPU path (tests)

private:
    GpuOps& ops;
    Memory& mem;
    // by chunk parity (GPU: ws[0] only). ws[1] has its own inputs and outputs (tokens, x, o, q / k / v / gate,
    // qkv / z / b / a), the GPU scratch is ws[0]'s: the lane's command buffers run one after another (one queue, the
    // fence) and no scratch is live from one to the next
    Workspace ws[2];
    StepTables tables;
    uint64_t n = 0;  // chunks committed
    std::vector<id<MTLCommandBuffer>> cbs;  // GPU: the chunk in flight's
    std::deque<PrefillWork> q;  // in flight, oldest first: chunks n - q.size() + 1 .. n

    // ANE: the engine queues chunk k in `todo` (its workspace and surface set: k % 2); the thread signals `event`
    // with k when done (or failed: err set first), in submit order. A GPU chunk also runs on this thread: its
    // request's earlier chunk must have committed every layer first (one queue orders them).
    struct Chunk {
        uint64_t k;
        int req;
        bool gpu;                // all layers on the GPU, in one command buffer
        int layer = -1;          // layers whose attn is committed (-1: not started)
        id<MTLCommandBuffer> cb;  // the last one committed
    };
    AneOps* ane;
    id<MTLFence> fence;  // the untracked arena's K/V and state: each command buffer after the one before (both paths)
    std::mutex m;
    std::condition_variable cv;
    std::deque<Chunk> todo;
    uint64_t done = 0;  // the last chunk the thread finished (read under m: memory ordering)
    bool stop = false;
    std::exception_ptr err;
    std::thread thread;
    void run();
    bool turn(Chunk& c, const std::deque<Chunk>& others);
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
