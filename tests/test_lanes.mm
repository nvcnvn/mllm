// Lanes (cpp-port.md §5): the golden trace through PrefillLane and DecodeLane (prefill chunks, then decode steps of
// 2 requests) gives the golden greedy ids; and pending reclaim: a unit released while a decode step is in flight is
// not reallocated until the lane's event passes that step.
#include <unistd.h>

#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "golden.h"
#include "lanes.h"

// A request's arena units, taken as the engine would take them
struct RequestUnits {
    std::vector<uint32_t> state_units, pages;
};

static RequestUnits take_request_units(mllm::Memory& memory, int max_context) {
    mllm::Arena& arena = memory.arena;
    RequestUnits units;
    assert(arena.admit(memory.state_units + memory.units_for(max_context)));
    for (uint32_t j = 0; j < memory.state_units; j++) units.state_units.push_back(arena.alloc());
    for (uint32_t j = 0; j < memory.units_for(max_context); j++) units.pages.push_back(arena.alloc());
    for (uint32_t unit : units.state_units)
        memset((uint8_t*)memory.arena_buffer.contents + unit * arena.unit_bytes, 0, arena.unit_bytes);
    return units;
}

static std::vector<int> trace_token_ids(const mllm::Safetensors& trace, const std::string& name) {
    const mllm::Tensor& tensor = trace.at(name);
    std::vector<int> ids(tensor.nbytes / 4);
    for (size_t i = 0; i < ids.size(); i++) ids[i] = golden::load<int32_t>(trace.data(tensor), i);
    return ids;
}

// poll the oldest decode step: its greedy ids equal golden's for that call
static void check_oldest_decode_step(mllm::DecodeLane& decode, const mllm::Safetensors& trace,
                                     const std::string& call_name) {
    std::vector<std::pair<int, int>> polled;
    while (polled.empty()) polled = decode.poll();
    std::vector<int> expected = trace_token_ids(trace, call_name + "/argmax");
    assert(polled.size() == expected.size());
    for (size_t row = 0; row < polled.size(); row++) assert(polled[row].second == expected[row]);
}

static void run_prefill_chunk(mllm::PrefillLane& prefill, const nlohmann::json& row, const std::vector<int>& tokens,
                              const RequestUnits requests[2]) {
    int request_id = row["req"];
    assert(!prefill.busy());
    prefill.submit({request_id, tokens, row["pos"], requests[request_id].pages, requests[request_id].state_units});
    assert(prefill.busy());
    std::vector<mllm::PrefillWork> done;
    while (done.empty()) done = prefill.poll();
    assert(done.size() == 1 && done[0].req_id == request_id && done[0].n_valid() == (int)tokens.size() &&
           !prefill.busy());
}

// The golden trace's calls through the two lanes. Returns the decode steps submitted.
static uint64_t run_golden_trace(const nlohmann::json& golden_spec, const mllm::Safetensors& trace,
                                 mllm::PrefillLane& prefill, mllm::DecodeLane& decode, const mllm::Arena& arena,
                                 const RequestUnits requests[2]) {
    uint64_t step = 0;
    int decode_calls = 0;
    std::string previous_call;  // the decode call in flight
    for (const auto& call : golden_spec["calls"]) {
        std::string call_name = call["name"].get<std::string>();
        std::vector<int> tokens = trace_token_ids(trace, call_name + "/tokens");
        if (call["rows"].size() == 1 && tokens.size() > 1) {
            run_prefill_chunk(prefill, call["rows"][0], tokens, requests);
            continue;
        }
        // decode1 and decode2 are fed: submitted before the previous step is polled, tokens from its GPU argmax
        const bool fed = decode_calls > 0;
        if (fed) assert(tokens == trace_token_ids(trace, previous_call + "/argmax"));
        mllm::DecodeWork work{step++, {}};
        for (size_t row_index = 0; row_index < call["rows"].size(); row_index++) {
            const auto& row = call["rows"][row_index];
            int request_id = row["req"];
            work.rows.push_back({request_id, fed ? -1 : tokens[row_index], row["pos"], requests[request_id].pages,
                                 requests[request_id].state_units, {}});
        }
        assert(decode.can_submit());
        decode.submit(work);
        assert(decode.can_submit() == !fed && arena.committed == step);
        if (fed) check_oldest_decode_step(decode, trace, previous_call);
        previous_call = call_name;
        decode_calls++;
    }
    check_oldest_decode_step(decode, trace, previous_call);
    assert(decode_calls == 3);
    return step;
}

// Pending reclaim. A gate holds the decode queue, so the step stays in flight until the test opens it.
static void check_pending_reclaim(id<MTLDevice> device, mllm::DecodeLane& decode, mllm::Arena& arena, uint64_t step,
                                  const RequestUnits& request) {
    id<MTLSharedEvent> gate = [device newSharedEvent];
    @autoreleasepool {
        id<MTLCommandBuffer> command_buffer = [decode.queue commandBuffer];
        [command_buffer encodeWaitForEvent:gate value:1];
        [command_buffer commit];
    }
    assert(arena.admit(1));
    const uint32_t released_unit = arena.alloc();
    assert(decode.can_submit() && decode.signaled() == step);
    decode.submit({step, {{0, 0, 0, request.pages, request.state_units, {}}}});
    arena.release({released_unit});  // its last reference drops while the step may still read it
    usleep(50000);
    assert(decode.poll().empty() && decode.signaled() == step);
    arena.retire(decode.signaled());
    auto is_free = [&](uint32_t unit) { return (arena.free_bitmap[unit / 64] >> (unit % 64)) & 1; };
    assert(!is_free(released_unit) && arena.pending.size() == 1);
    // every other unit can be taken; the released one is not among them
    const uint32_t other_units = arena.free_units - 1;
    assert(arena.admit(other_units));
    std::vector<uint32_t> taken;
    for (uint32_t i = 0; i < other_units; i++) taken.push_back(arena.alloc()), assert(taken.back() != released_unit);
    gate.signaledValue = 1;
    decode.wait();
    assert(decode.signaled() == step + 1 && decode.poll().size() == 1);
    arena.retire(decode.signaled());
    assert(is_free(released_unit) && arena.admit(1) && arena.alloc() == released_unit);
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string golden_dir = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(golden_dir + "/spec.json"));
    mllm::Safetensors trace(golden_dir + "/trace.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    // two requests of max_context, then spare units
    mllm::Memory memory(model, device, std::max<size_t>(64 << 20, mllm::Memory::arena_for(model, 3, 128)));
    const int chunk = golden_spec["chunk"], max_context = 128;
    mllm::GpuOps ops(memory, weights);
    mllm::PrefillLane prefill(ops, memory, chunk, max_context);
    mllm::DecodeLane decode(ops, memory, 4, max_context);
    memory.freeze();

    const RequestUnits requests[2] = {take_request_units(memory, max_context), take_request_units(memory, max_context)};
    const uint64_t step = run_golden_trace(golden_spec, trace, prefill, decode, memory.arena, requests);
    printf("golden trace through the lanes: 3 prefill chunks, 3 batch-2 decode steps (2 fed on the GPU, each\n"
           "committed before the previous one was polled), greedy ids equal golden\n");

    check_pending_reclaim(device, decode, memory.arena, step, requests[0]);
    printf("pending reclaim: a unit released during step %llu is reused only after the event passes it\nPASS\n",
           (unsigned long long)step);
}
