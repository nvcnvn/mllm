#include "lanes.h"

#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace mllm {

static std::runtime_error command_buffer_failed(const char* lane_name, id<MTLCommandBuffer> command_buffer) {
    return std::runtime_error(std::string(lane_name) + ": command buffer failed" +
                              (command_buffer.error
                                   ? std::string(": ") + command_buffer.error.localizedDescription.UTF8String
                                   : ""));
}

// Waits for the command buffer; throws if it did not complete.
static void wait_until_completed(const char* lane_name, id<MTLCommandBuffer> command_buffer) {
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) throw command_buffer_failed(lane_name, command_buffer);
}

// One command buffer on queue: encode_work encodes into workspace.encoder (after the fence's last update, then
// updating it, if a fence is given), signaling `signal_event` with `signal_value` at its end if given. Not committed.
static id<MTLCommandBuffer> encode_command_buffer(id<MTLCommandQueue> queue, Workspace& workspace,
                                                  const std::function<void()>& encode_work, bool concurrent,
                                                  id<MTLFence> fence = nil, id<MTLSharedEvent> signal_event = nil,
                                                  uint64_t signal_value = 0) {
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    workspace.encoder = [command_buffer
        computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    if (fence) [workspace.encoder waitForFence:fence];
    try {
        encode_work();
    } catch (...) {  // the ANE's predict failed midway: close the encoder, drop the command buffer
        [workspace.encoder endEncoding];
        workspace.encoder = nil;
        throw;
    }
    if (fence) [workspace.encoder updateFence:fence];
    [workspace.encoder endEncoding];
    workspace.encoder = nil;
    if (signal_event) [command_buffer encodeSignalEvent:signal_event value:signal_value];
    return command_buffer;
}

static id<MTLCommandBuffer> encode_and_commit(id<MTLCommandQueue> queue, Workspace& workspace,
                                              const std::function<void()>& encode_work, bool concurrent,
                                              id<MTLFence> fence = nil) {
    id<MTLCommandBuffer> command_buffer = encode_command_buffer(queue, workspace, encode_work, concurrent, fence);
    [command_buffer commit];
    return command_buffer;
}

double encode_and_wait(id<MTLCommandQueue> queue, Workspace& workspace, const std::function<void()>& encode_work,
                       bool concurrent) {
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = encode_and_commit(queue, workspace, encode_work, concurrent);
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted)
            throw command_buffer_failed("encode_and_wait", command_buffer);
        return command_buffer.GPUEndTime - command_buffer.GPUStartTime;
    }
}

static id<MTLCommandQueue> new_lane_queue(Memory& mem) {
    id<MTLCommandQueue> queue = [mem.device newCommandQueue];
    if (mem.residency) [queue addResidencySet:mem.residency];
    return queue;
}

// The GPU runs both lanes' command buffers in commit order: a decode step committed after a whole prefill chunk waits
// for all of it (27B: ~4 s for 512 tokens). So a prefill chunk's GPU layers go in command buffers of this many layers,
// each committed once the one before completes: a decode step waits for one at most. 8 at 27B (~0.5 s each): a
// 2048-token prefill next to a decode of one request keeps 107 tok/s (131 alone) and the decode's longest step is
// 0.54 s (3.9 s committed whole); 4: 85 tok/s, 0.31 s (report, g15). A lone prefill loses nothing to the chain: 4096
// tokens at 128.1-128.2 tok/s chained, 127.7-129.2 as one buffer per chunk (g18).
static constexpr int LAYERS_PER_BUFFER = 8;

// lanes.py _layers: apply_weights(i) then apply_attention(i) for layers first .. end - 1, into workspace.encoder
static void encode_layers(GpuOps& ops, Workspace& workspace, int first, int end) {
    for (int layer = first; layer < end; layer++) {
        ops.apply_weights(layer, workspace);
        ops.apply_attention(layer, workspace);
    }
}

// Prefill, after a chunk's last layer, with an MTP head: the rest of the last layer and the final norm (the 27B's
// hidden states), then the head over every row with the token after it (token_ids[1 .. n], PrefillWork::next_token
// last) -> the head's K/V for the chunk's positions
static void encode_mtp(GpuOps& ops, Workspace& workspace, int layer_count) {
    if (!ops.has_mtp()) return;
    ops.apply_weights(layer_count, workspace);
    ops.apply_mtp(workspace, {workspace.residual}, {workspace.token_ids, sizeof(int32_t)});
}

PrefillLane::PrefillLane(GpuOps& ops, Memory& mem, int chunk_tokens, int max_context, AneOps* ane)
    : queue(new_lane_queue(mem)), event([mem.device newSharedEvent]), ops(ane ? ane->gpu : ops), mem(mem),
      workspaces{ops.workspace(chunk_tokens, 1, max_context), {}}, step_tables(mem.step_tables(1, max_context)),
      ane(ane), fence([mem.device newFence]) {
    if (ane && ane->S != chunk_tokens)
        throw std::invalid_argument("ANE prefill: chunk must equal the blocks' compiled S");
    if (ane) {
        // the second workspace shares the first's GPU scratch, with inputs and outputs of its own
        const Workspace own = mem.workspace(chunk_tokens, 1);
        Workspace& second = workspaces[1];
        second = workspaces[0];
        second.token_ids = own.token_ids;
        second.residual = own.residual;
        second.attention_result = own.attention_result;
        second.query = own.query;
        second.key = own.key;
        second.value = own.value;
        second.query_gate = own.query_gate;
        second.query_key_value = own.query_key_value;
        second.output_gate = own.output_gate;
        second.update_rate_input = own.update_rate_input;
        second.decay_input = own.decay_input;
    }
    if (ane) ane_thread = std::thread(&PrefillLane::run_ane_thread, this);
}

PrefillLane::~PrefillLane() {
    if (!ane_thread.joinable()) return;
    {
        std::lock_guard lock(mutex);
        stopping = true;
    }
    wakeup.notify_one();
    ane_thread.join();  // after the chunks it started, if any
}

// The ANE thread. Turns take the started chunks in rotation, oldest first. A turn waits only for the command buffer
// its own predict needs (its chunk's last one: it wrote the block's inputs), so the ANE runs one chunk's
// apply_weights(i) while the GPU runs the other's apply_attention. The older chunk is never behind, so a request's
// later chunk commits each layer's attention after its earlier chunk's.
void PrefillLane::run_ane_thread() {
    std::deque<Chunk> started;  // up to 2, in turn order
    for (;;) {
        @autoreleasepool {
            if (!take_queued_chunks(started)) return;
            if (started.empty()) continue;
            Chunk chunk = started.front();
            started.pop_front();
            try {
                if (take_turn(chunk, started)) started.push_back(chunk);
            } catch (...) {
                started.push_back(chunk);
                drop_started_chunks(started, std::current_exception());
            }
        }
    }
}

// Waits until there is something to do, then moves queued chunks into `started` (up to 2). False when the lane is
// stopping and no chunk is started.
bool PrefillLane::take_queued_chunks(std::deque<Chunk>& started) {
    std::unique_lock lock(mutex);
    wakeup.wait(lock, [&] { return stopping || !started.empty() || !queued_chunks.empty(); });
    if (stopping && started.empty()) return false;
    while (!stopping && started.size() < 2 && !queued_chunks.empty()) {
        if (failure) {  // a failed chunk dropped the ones in flight: drop later ones too, until poll sees it
            last_chunk_done = queued_chunks.front().number;
            event.signaledValue = last_chunk_done;
        } else {
            started.push_back(queued_chunks.front());
        }
        queued_chunks.pop_front();
    }
    return true;
}

// A turn threw: drop every started chunk once the GPU is done with it.
void PrefillLane::drop_started_chunks(std::deque<Chunk>& started, std::exception_ptr error) {
    uint64_t last_started = 0;
    for (const Chunk& chunk : started) {
        if (chunk.last_command_buffer) [chunk.last_command_buffer waitUntilCompleted];
        last_started = std::max(last_started, chunk.number);
    }
    started.clear();
    std::lock_guard lock(mutex);
    failure = error;
    last_chunk_done = last_started;
    event.signaledValue = last_started;  // wakes wait_oldest; poll trusts last_chunk_done
}

// One step of the chunk: start it, run one layer, or finish it. False when it's done.
bool PrefillLane::take_turn(Chunk& chunk, const std::deque<Chunk>& others) {
    const int layer_count = mem.spec.n_layers(), surface_set = (int)(chunk.number % 2);
    Workspace& workspace = workspaces[surface_set];
    // A GPU chunk, as the GPU lane: LAYERS_PER_BUFFER layers per turn, each after the one before completes (whole, a
    // 255-token tail held a decode step ~2 s at 27B); no attention output projection / feed-forward of the last layer
    if (chunk.all_on_gpu && chunk.layers_committed < layer_count) {
        if (chunk.layers_committed < 0) {
            const bool earlier_chunk_of_request_running =
                std::any_of(others.begin(), others.end(), [&](const Chunk& other) {
                    return other.number < chunk.number && other.request_id == chunk.request_id &&
                           other.layers_committed < layer_count;
                });
            if (earlier_chunk_of_request_running) return true;
            chunk.layers_committed = 0;
        } else {
            wait_until_completed("PrefillLane", chunk.last_command_buffer);
        }
        const int first = chunk.layers_committed, end = std::min(first + LAYERS_PER_BUFFER, layer_count);
        chunk.last_command_buffer =
            encode_and_commit(queue, workspace, [&] {
                encode_layers(ops, workspace, first, end);
                if (end == layer_count) encode_mtp(ops, workspace, layer_count);
            }, false, fence);
        chunk.layers_committed = end;
        return true;
    }
    if (chunk.layers_committed < 0) {
        chunk.last_command_buffer =
            encode_and_commit(queue, workspace, [&] { ane->prepare_inputs(0, workspace, surface_set); }, false, fence);
        chunk.layers_committed = 0;
        return true;
    }
    if (chunk.layers_committed == layer_count) {  // done in submit order: the event counts chunks
        if (std::any_of(others.begin(), others.end(), [&](const Chunk& other) { return other.number < chunk.number; }))
            return true;
        wait_until_completed("PrefillLane", chunk.last_command_buffer);
        std::lock_guard lock(mutex);
        last_chunk_done = chunk.number;
        event.signaledValue = chunk.number;  // wakes wait_oldest; poll trusts last_chunk_done
        return false;
    }
    wait_until_completed("PrefillLane", chunk.last_command_buffer);
    if (fail_next.exchange(false)) throw std::runtime_error("PrefillLane: injected failure");
    const int layer = chunk.layers_committed;
    // K/V and state only: the last layer's attention output projection + feed-forward never run (but for the MTP
    // head, on the GPU)
    chunk.last_command_buffer = encode_and_commit(queue, workspace, [&] {
        ane->apply_weights(layer, workspace, surface_set);
        ops.apply_attention(layer, workspace);
        if (layer + 1 < layer_count) ane->prepare_inputs(layer + 1, workspace, surface_set);
        else encode_mtp(ops, workspace, layer_count);
    }, false, fence);
    chunk.layers_committed++;
    return true;
}

void PrefillLane::submit(PrefillWork work) {
    @autoreleasepool {
        assert(can_submit() && work.n_valid() > 0 && work.n_valid() <= workspaces[0].max_rows);
        if (fail_submit && --fail_submit == 0) throw std::runtime_error("PrefillLane: injected submit failure");
        Workspace& workspace = workspaces[ane ? (chunks_committed + 1) % 2 : 0];
        workspace.request_count = 1, workspace.tokens_per_request = work.n_valid();
        std::copy(work.tokens.begin(), work.tokens.end(), (int32_t*)workspace.token_ids.contents);
        ((int32_t*)workspace.token_ids.contents)[work.n_valid()] = work.next_token;  // encode_mtp's last row
        workspace.step_table = step_tables.begin(chunks_committed + 1, event.signaledValue);
        step_tables.write(workspace.step_table, {{work.start_pos, work.n_valid(), work.pages, work.state_slots}});
        chunks_committed++;
        if (ane) {
            const bool on_gpu = work.n_valid() < gpu_below;
            gpu_chunks += on_gpu;
            {
                std::lock_guard lock(mutex);
                queued_chunks.push_back({chunks_committed, work.req_id, on_gpu, -1, nil});
            }
            wakeup.notify_one();
        } else {
            commit_gpu_chunk(workspace);
        }
        in_flight.push_back(std::move(work));
    }
}

// The chunk is encoded now as command buffers of LAYERS_PER_BUFFER layers, each committed from the completion handler
// of the one before. K/V and state only: the last layer's attention output projection + feed-forward and
// pick_next_tokens never run (but for the MTP head: encode_mtp).
void PrefillLane::commit_gpu_chunk(Workspace& workspace) {
    const int layer_count = mem.spec.n_layers(), layers_per_buffer = LAYERS_PER_BUFFER;
    for (int first_layer = 0; first_layer < layer_count; first_layer += layers_per_buffer) {
        const bool last_buffer = first_layer + layers_per_buffer >= layer_count;
        gpu_command_buffers.push_back(encode_command_buffer(queue, workspace, [&] {
            for (int layer = first_layer; layer < std::min(first_layer + layers_per_buffer, layer_count); layer++)
                ops.apply_weights(layer, workspace), ops.apply_attention(layer, workspace);
            if (last_buffer) encode_mtp(ops, workspace, layer_count);
        }, false, fence, last_buffer ? event : nil, chunks_committed));
    }
    for (size_t j = 0; j + 1 < gpu_command_buffers.size(); j++) {
        id<MTLCommandBuffer> next_buffer = gpu_command_buffers[j + 1];
        [gpu_command_buffers[j] addCompletedHandler:^(id<MTLCommandBuffer>) {
            [next_buffer commit];  // also after an error
        }];
    }
    [gpu_command_buffers[0] commit];
}

std::vector<PrefillWork> PrefillLane::poll() {
    @autoreleasepool {
        if (in_flight.empty()) return {};
        uint64_t chunks_done;
        if (ane) {
            std::exception_ptr error;
            {
                std::lock_guard lock(mutex);
                chunks_done = last_chunk_done;
                if (failure && chunks_done < chunks_committed) return {};  // the thread is still dropping chunks
                error = std::exchange(failure, nullptr);
            }
            if (error) {
                in_flight.clear();
                std::rethrow_exception(error);
            }
        } else {
            chunks_done = event.signaledValue;  // before the scan: once the last buffer signals, every status is final
            throw_if_gpu_chunk_failed();
        }
        std::vector<PrefillWork> finished;
        while (!in_flight.empty() && chunks_committed - in_flight.size() + 1 <= chunks_done)
            finished.push_back(std::move(in_flight.front())), in_flight.pop_front();
        if (in_flight.empty()) gpu_command_buffers.clear();
        return finished;
    }
}

void PrefillLane::throw_if_gpu_chunk_failed() {
    for (id<MTLCommandBuffer> command_buffer : gpu_command_buffers) {
        if (command_buffer.status != MTLCommandBufferStatusError) continue;
        auto error = command_buffer_failed("PrefillLane", command_buffer);
        // The chain commits them all, also after an error: let it end before the next chunk. Not
        // waitUntilCompleted alone: on a buffer not committed yet it asserts under API validation (without
        // validation it waits for the commit; g15 checked both).
        id<MTLCommandBuffer> last_buffer = gpu_command_buffers.back();
        while (last_buffer.status < MTLCommandBufferStatusCommitted) std::this_thread::yield();
        [last_buffer waitUntilCompleted];
        // a failed last buffer may skip its signal: mark the chunk done, as DecodeLane::poll does
        event.signaledValue = std::max(event.signaledValue, chunks_committed);
        gpu_command_buffers.clear(), in_flight.clear();
        throw error;
    }
}

DecodeLane::DecodeLane(GpuOps& ops, Memory& mem, int max_batch, int max_context)
    : queue(new_lane_queue(mem)), event([mem.device newSharedEvent]), fence([mem.device newFence]), ops(ops), mem(mem),
      // with a head, room for a speculative verify's DRAFTS + 1 queries per request (attention's partials)
      workspaces{ops.workspace(ops.has_mtp() ? DRAFTS + 1 : 1, max_batch, max_context),
                 ops.workspace(ops.has_mtp() ? DRAFTS + 1 : 1, max_batch, max_context)},
      step_tables(mem.step_tables(max_batch, max_context)), verify_tables(mem.step_tables(DRAFTS + 2, max_context)),
      chain_tables{mem.step_tables(1, max_context)} {
    static_assert(DRAFTS == 2, "chain_tables' initializer");
}

void DecodeLane::submit(DecodeWork work) {
    @autoreleasepool {
        if (fail_next) {
            fail_next = false;
            throw std::runtime_error("DecodeLane: injected failure");
        }
        Workspace& workspace = workspaces[work.step % 2];
        const int row_count = (int)work.rows.size();
        assert(can_submit() && row_count > 0 && row_count <= workspace.max_rows &&
               (in_flight.empty() || in_flight.back().work.step + 1 == work.step));
        const bool speculative = work.rows[0].draft_from >= 0;
        assert(!speculative || (row_count == 1 && in_flight.empty() && ops.has_mtp()));
        std::vector<int> source_rows(row_count, -1);  // fed rows: their row in the step in flight
        const bool any_fed = !speculative && write_step_inputs(work, workspace, source_rows);
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        workspace.encoder = [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        // Defensive today: a row in both step N and N + 1 is fed, and feeding reads N's next_token_ids, a tracked
        // buffer, so Metal already orders the two encoders (and with them the untracked arena's K/V and state).
        // Needed if workspace buffers become untracked. A fence, not encodeWaitForEvent: that cost a decode of one
        // request ~3.5% (297 vs 287 tok/s).
        [workspace.encoder waitForFence:fence];
        if (speculative) {
            encode_round(work, workspace);
        } else {
            if (any_fed)
                ops.feed_previous_tokens(workspace, workspaces[(work.step + 1) % 2].next_token_ids, source_rows);
            const int layer_count = mem.spec.n_layers();
            encode_layers(ops, workspace, 0, layer_count);
            ops.apply_weights(layer_count, workspace);
            std::vector<SampleParams> sampler_params;
            for (const DecodeRow& row : work.rows) sampler_params.push_back(row.sampler);
            ops.pick_next_tokens(workspace, sampler_params);
            if (ops.has_mtp()) ops.apply_mtp(workspace, {workspace.residual}, {workspace.next_token_ids});
        }
        [workspace.encoder updateFence:fence];
        [workspace.encoder endEncoding];
        workspace.encoder = nil;
        [command_buffer encodeSignalEvent:event value:work.step + 1];
        [command_buffer commit];
        mem.arena.committed = work.step + 1;  // units released from now on wait for this step
        in_flight.push_back({command_buffer, std::move(work)});
    }
}

// The speculative round of submit(), into workspace.encoder: drafts, then the verify of [row.token, drafts] at
// row.pos .. row.pos + DRAFTS.
void DecodeLane::encode_round(const DecodeWork& work, Workspace& workspace) {
    const DecodeRow& row = work.rows[0];
    const int rows = DRAFTS + 1, layer_count = mem.spec.n_layers();
    const Workspace& previous = workspaces[(work.step + 1) % 2];
    // drafts: from the head's output at row.pos - 1, the chain at row.pos, row.pos + 1, ...
    std::vector<id<MTLBuffer>> chain;
    for (int j = 0; j + 1 < DRAFTS; j++) {
        chain.push_back(chain_tables[j].begin(work.step + 1, event.signaledValue));
        chain_tables[j].write(chain.back(), {{row.pos + j, 1, row.pages}});
    }
    ops.draft_tokens(workspace, {previous.residual, (size_t)row.draft_from * mem.spec.hidden * ELEM}, chain);
    // verify: token_ids = [row.token, drafts...] (fed on the GPU), one request of DRAFTS + 1 rows; the linear layers
    // write the state after each row into its own set
    ((int32_t*)workspace.token_ids.contents)[0] = row.token;
    std::vector<int> source_rows{-1};
    for (int j = 0; j < DRAFTS; j++) source_rows.push_back(j);
    workspace.request_count = rows, workspace.tokens_per_request = 1;
    ops.feed_previous_tokens(workspace, workspace.next_token_ids, source_rows);
    std::vector<uint32_t> state_sets = row.state_slots;  // read; then after token 0 (the same units), 1, 2, ...
    state_sets.insert(state_sets.end(), row.state_slots.begin(), row.state_slots.end());
    state_sets.insert(state_sets.end(), row.spare_state.begin(), row.spare_state.end());
    assert(row.spare_state.size() == DRAFTS * row.state_slots.size());
    workspace.step_table = verify_tables.begin(work.step + 1, event.signaledValue);
    verify_tables.write(workspace.step_table, {{row.pos, rows, row.pages, state_sets}});
    workspace.request_count = 1, workspace.tokens_per_request = rows, workspace.checkpoint_states = true;
    encode_layers(ops, workspace, 0, layer_count);
    ops.apply_weights(layer_count, workspace);
    workspace.checkpoint_states = false;
    workspace.request_count = rows, workspace.tokens_per_request = 1;  // lm_head and argmax on every row
    ops.pick_next_tokens(workspace);
    workspace.request_count = 1, workspace.tokens_per_request = rows;
    ops.apply_mtp(workspace, {workspace.residual}, {workspace.next_token_ids});
}

// Writes the step's shape, token ids and step table into the workspace. A row whose token is -1 takes its token from
// the step in flight on the GPU: source_rows gets its row there. True if any row does.
bool DecodeLane::write_step_inputs(const DecodeWork& work, Workspace& workspace, std::vector<int>& source_rows) {
    const int row_count = (int)work.rows.size();
    workspace.request_count = row_count, workspace.tokens_per_request = 1;
    std::vector<StepTables::Row> table_rows;
    bool any_fed = false;
    for (int b = 0; b < row_count; b++) {
        const DecodeRow& row = work.rows[b];
        if (row.token < 0) {
            assert(!in_flight.empty());
            const auto& previous_rows = in_flight.back().work.rows;
            auto previous = std::find_if(previous_rows.begin(), previous_rows.end(),
                                         [&](const DecodeRow& earlier) { return earlier.req_id == row.req_id; });
            assert(previous != previous_rows.end() && previous->pos + 1 == row.pos);
            source_rows[b] = (int)(previous - previous_rows.begin()), any_fed = true;
        }
        ((int32_t*)workspace.token_ids.contents)[b] = row.token;
        table_rows.push_back({row.pos, 1, row.pages, row.state_slots});
    }
    workspace.step_table = step_tables.begin(work.step + 1, event.signaledValue);
    step_tables.write(workspace.step_table, table_rows);
    return any_fed;
}

std::vector<std::pair<int, int>> DecodeLane::poll() {
    @autoreleasepool {
        if (in_flight.empty()) return {};
        for (const InFlightStep& step : in_flight) {
            if (step.command_buffer.status != MTLCommandBufferStatusError) continue;
            auto error = command_buffer_failed("DecodeLane", step.command_buffer);
            wait();
            // a failed command buffer may skip its signal: units released at its step would stay pending for
            // good. Everything committed has completed now, so mark it done (one queue, in order).
            event.signaledValue = std::max(event.signaledValue, in_flight.back().work.step + 1);
            in_flight.clear();
            throw error;
        }
        const DecodeWork& oldest = in_flight.front().work;
        if (event.signaledValue < oldest.step + 1) return {};
        const Workspace& workspace = workspaces[oldest.step % 2];
        std::vector<std::pair<int, int>> finished;
        if (oldest.rows[0].draft_from >= 0) {  // the argmax after each row, while the draft before it matched
            const int32_t *chosen = (const int32_t*)workspace.next_token_ids.contents,
                          *drafts = (const int32_t*)workspace.token_ids.contents + 1;
            for (int j = 0; j <= DRAFTS; j++) {
                finished.emplace_back(oldest.rows[0].req_id, chosen[j]);
                if (j == DRAFTS || drafts[j] != chosen[j]) break;
            }
        } else {
            std::vector<int> tokens = ops.next_tokens(workspace);
            for (size_t b = 0; b < tokens.size(); b++) finished.emplace_back(oldest.rows[b].req_id, tokens[b]);
        }
        in_flight.pop_front();
        return finished;
    }
}

// The event's value of the oldest work in flight, not signaledValue + 1: that may already have passed since the poll.
void PrefillLane::wait_oldest() const {
    if (busy()) [event waitUntilSignaledValue:chunks_committed - in_flight.size() + 1 timeoutMS:1000];
}

void DecodeLane::wait_oldest() const {
    if (!in_flight.empty()) [event waitUntilSignaledValue:in_flight.front().work.step + 1 timeoutMS:1000];
}

void DecodeLane::wait() {
    if (!in_flight.empty()) [in_flight.back().command_buffer waitUntilCompleted];
}

}  // namespace mllm
