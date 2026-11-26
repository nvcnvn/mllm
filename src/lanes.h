// PrefillLane and DecodeLane: submit() / poll() (py/mllm/lanes.py). submit encodes the work into command buffers on
// the lane's own MTLCommandQueue (prefill: a chain of 8-layer buffers per chunk), commits and returns; poll checks the
// lane's MTLSharedEvent (never blocks) and returns finished work. Both lanes live on the engine thread; the GPU
// time-shares the two queues' command buffers in commit order (g15), it doesn't run them side by side. Lanes never
// touch the allocator: every unit a work item needs is allocated before submit.
// With AneOps, PrefillLane runs chunks on a thread of its own (Core ML's predict blocks): per layer, the ANE's
// apply_weights(i), then the GPU's apply_attention(i) on the lane's queue, two chunks in flight so the ANE runs one
// chunk's weights while the GPU runs the other's attention; the engine sees the same submit / poll / event as on the
// GPU.
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
    std::vector<uint32_t> pages;        // KV pages covering start_pos + n_valid
    std::vector<uint32_t> state_slots;  // the request's state units
    int next_token = 0;  // the prompt's token after this chunk (an MTP head's last row reads it)
    int n_valid() const { return (int)tokens.size(); }
};

struct DecodeRow {
    int req_id, token, pos;  // token -1: the request's row in the step in flight, its token fed on the GPU
    std::vector<uint32_t> pages, state_slots;
    SampleParams sampler;
    // >= 0: a speculative round (greedy, the step's only row, docs/speculative.md), drafting from the head's output in
    // this row of the step just before. Its verify writes the linear state after each of its DRAFTS + 1 tokens into
    // its own set: after token 0 into state_slots, after token j into spare_state's set j - 1 (state_units each).
    int draft_from = -1;
    std::vector<uint32_t> spare_state;
};

struct DecodeWork {
    uint64_t step;
    std::vector<DecodeRow> rows;
};

// One command buffer on queue: encode_work encodes into workspace.encoder (concurrent: as the decode lane's). Waits;
// returns GPU seconds.
double encode_and_wait(id<MTLCommandQueue> queue, Workspace& workspace, const std::function<void()>& encode_work,
                       bool concurrent = false);

class PrefillLane {
public:
    // ane: apply_weights on the Neural Engine (chunk_tokens must be its compiled S), apply_attention with ane->gpu;
    // nullptr: all on the GPU
    PrefillLane(GpuOps& ops, Memory& mem, int chunk_tokens, int max_context, AneOps* ane = nullptr);
    ~PrefillLane();
    bool busy() const { return !in_flight.empty(); }
    // ANE: two chunks in flight, of any requests (a request's later chunk runs each layer's attention after its
    // earlier chunk's: one queue, one fence). GPU: one.
    bool can_submit() const { return in_flight.size() < (ane ? 2u : 1u); }
    void submit(PrefillWork work);
    // the chunks done, in submit order. Throws if one failed: every chunk in flight is dropped (after the thread
    // let go of them), the lane is free again.
    std::vector<PrefillWork> poll();
    void wait_oldest() const;  // until the oldest chunk in flight is done (or 1 s: a failed one never signals)

    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event;  // signaled with the chunk count
    std::atomic<bool> fail_next = false;  // tests: the ANE thread's next layer throws (a failed predict)
    int fail_submit = 0;                  // tests: the fail_submit-th submit from now throws
    // ANE: a chunk of fewer tokens runs on the GPU path instead (workspace.tokens_per_request = its token count, no
    // padding to S: tails and prefix-cache splits). The crossover of GPU prefill time at that many tokens against one
    // ANE chunk (reports/mlx-lm baseline.md).
    int gpu_below = 256;
    int gpu_chunks = 0;  // ANE: chunks submitted to the GPU path (tests)

private:
    GpuOps& ops;
    Memory& mem;
    // By chunk parity (GPU: workspaces[0] only). workspaces[1] has its own inputs and outputs (token_ids, residual,
    // attention_result, query / key / value / query_gate, query_key_value / output_gate / update_rate_input /
    // decay_input); the GPU scratch is workspaces[0]'s: the lane's command buffers run one after another (one queue,
    // the fence) and no scratch is live from one to the next.
    Workspace workspaces[2];
    StepTables step_tables;
    uint64_t chunks_committed = 0;
    std::vector<id<MTLCommandBuffer>> gpu_command_buffers;  // GPU: the chunk in flight's
    // oldest first: chunks chunks_committed - in_flight.size() + 1 .. chunks_committed
    std::deque<PrefillWork> in_flight;

    // ANE: the engine queues chunk k in `queued_chunks` (its workspace and surface set: k % 2); the thread signals
    // `event` with k when done (or failed: `failure` set first), in submit order. A GPU chunk also runs on this
    // thread: its request's earlier chunk must have committed every layer first (one queue orders them).
    struct Chunk {
        uint64_t number;  // k: its place in submit order, from 1
        int request_id;
        bool all_on_gpu;            // all layers on the GPU, LAYERS_PER_BUFFER per turn (lanes.mm)
        int layers_committed = -1;  // layers whose attention is committed (-1: not started)
        id<MTLCommandBuffer> last_command_buffer;  // the last one committed
    };
    AneOps* ane;
    id<MTLFence> fence;  // the untracked arena's K/V and state: each command buffer after the one before (both paths)
    std::mutex mutex;
    std::condition_variable wakeup;
    std::deque<Chunk> queued_chunks;
    uint64_t last_chunk_done = 0;  // the last chunk the thread finished (read under mutex: memory ordering)
    bool stopping = false;
    std::exception_ptr failure;
    std::thread ane_thread;
    void run_ane_thread();
    bool take_queued_chunks(std::deque<Chunk>& started);
    void drop_started_chunks(std::deque<Chunk>& started, std::exception_ptr error);
    bool take_turn(Chunk& chunk, const std::deque<Chunk>& others);
    void commit_gpu_chunk(Workspace& workspace);
    void throw_if_gpu_chunk_failed();
};

class DecodeLane {
public:
    // Speculative rounds draft this many tokens (k: docs/speculative.md, 4-row verify costs too much for k = 3)
    static constexpr int DRAFTS = 2;
    DecodeLane(GpuOps& ops, Memory& mem, int max_batch, int max_context);
    // At most one step ahead: step N + 1 is encoded and committed while N runs. Its rows can take N's token on the
    // GPU (DecodeRow::token -1; greedy or sampled, both chosen there), so a decode of one request never waits for the
    // CPU's encode and commit.
    bool can_submit() const { return in_flight.size() < 2; }
    // Sets mem.arena.committed = work.step + 1. Every step ends with the MTP head (if any) over its rows and the tokens
    // they chose, so the head's K/V covers every decoded position and its output can seed the next round's drafts.
    // A speculative round (one row, draft_from >= 0, nothing in flight, the step before it was this lane's last):
    // draft_tokens from that step's head output, then verify [token, drafts] as one request of DRAFTS + 1 rows
    // (multi-row kernels, so checked by tolerance), lm_head and argmax on every row, then the head over those rows.
    void submit(DecodeWork work);
    // The oldest step's (req_id, token) once its event passed; throws if a step failed (every step in flight is
    // dropped: a later one may have fed on its tokens). A speculative round gives its request 1 + n tokens: the
    // 27B's argmax after each of its n accepted drafts (equal to them) and after the last (the bonus or correction).
    std::vector<std::pair<int, int>> poll();
    uint64_t signaled() const { return event.signaledValue; }  // steps done: units pending at <= this are free
    bool busy() const { return !in_flight.empty(); }
    void wait();  // until every step in flight is done
    void wait_oldest() const;  // until the oldest step in flight is done (or 1 s: a failed one never signals)

    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event;  // signaled with step + 1 when a step completes
    id<MTLFence> fence;        // step N + 1 waits for step N's last dispatch
    bool fail_next = false;    // tests: the next submit throws (engine abort path)

private:
    GpuOps& ops;
    Memory& mem;
    // by step parity: the CPU writes step N + 1's token_ids and reads N's next_token_ids / next_token_scores meanwhile
    Workspace workspaces[2];
    StepTables step_tables;
    // a speculative round's: the verify's (one row, its 2 + DRAFTS state sets) and the draft chain's one-row tables
    StepTables verify_tables, chain_tables[DRAFTS - 1];
    struct InFlightStep {
        id<MTLCommandBuffer> command_buffer;
        DecodeWork work;
    };
    std::deque<InFlightStep> in_flight;
    bool write_step_inputs(const DecodeWork& work, Workspace& workspace, std::vector<int>& source_rows);
    void encode_round(const DecodeWork& work, Workspace& workspace);
};

}  // namespace mllm
