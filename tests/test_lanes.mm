// Lanes (cpp-port.md §5): the golden trace through PrefillLane and DecodeLane (prefill chunks, then batch-2 decode)
// gives the golden greedy ids; and pending reclaim: a unit released while a decode step is in flight is not
// reallocated until the lane's event passes that step.
#include <unistd.h>

#include <cassert>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "golden.h"
#include "lanes.h"

int main() {
    mllm::ModelSpec s = mllm::spec::load(golden::MODEL);
    std::string d = golden::dir(s);
    nlohmann::json g = nlohmann::json::parse(std::ifstream(d + "/spec.json"));
    mllm::Safetensors tr(d + "/trace.safetensors");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    mllm::Weights w(s.path, dev);
    mllm::Memory mem(s, dev, std::max<size_t>(64 << 20, mllm::Memory::arena_for(s, 3, 128)));  // two requests of max_ctx, then spare units
    const int chunk = g["chunk"], max_ctx = 128;
    mllm::GpuOps ops(mem, w);
    mllm::PrefillLane prefill(ops, mem, chunk, max_ctx);
    mllm::DecodeLane decode(ops, mem, 4, max_ctx);
    mem.freeze();
    mllm::Arena& a = mem.arena;

    // two requests, units from the arena as the engine would take them
    std::vector<uint32_t> st[2], pg[2];
    for (int r = 0; r < 2; r++) {
        assert(a.admit(mem.state_units + mem.units_for(max_ctx)));
        for (uint32_t j = 0; j < mem.state_units; j++) st[r].push_back(a.alloc());
        for (uint32_t j = 0; j < mem.units_for(max_ctx); j++) pg[r].push_back(a.alloc());
        for (uint32_t u : st[r]) memset((uint8_t*)mem.arena_buffer.contents + u * a.unit_bytes, 0, a.unit_bytes);
    }
    auto ids_of = [&](const std::string& name) {
        const mllm::Tensor& t = tr.at(name);
        std::vector<int> v(t.nbytes / 4);
        for (size_t i = 0; i < v.size(); i++) v[i] = golden::load<int32_t>(tr.data(t), i);
        return v;
    };

    uint64_t step = 0;
    int decodes = 0;
    std::string prev;  // the decode call in flight
    auto check = [&](const std::string& c) {  // poll the oldest step: its greedy ids equal golden
        std::vector<std::pair<int, int>> got;
        while (got.empty()) got = decode.poll();
        std::vector<int> want = ids_of(c + "/argmax");
        assert(got.size() == want.size());
        for (size_t b = 0; b < got.size(); b++) assert(got[b].second == want[b]);
    };
    for (const auto& call : g["calls"]) {
        std::string c = call["name"].get<std::string>();
        std::vector<int> tok = ids_of(c + "/tokens");
        if (call["rows"].size() == 1 && tok.size() > 1) {
            const auto& r = call["rows"][0];
            int req = r["req"];
            assert(!prefill.busy());
            prefill.submit({req, tok, r["pos"], pg[req], st[req]});
            assert(prefill.busy());
            std::vector<mllm::PrefillWork> done;
            while (done.empty()) done = prefill.poll();
            assert(done.size() == 1 && done[0].req_id == req && done[0].n_valid() == (int)tok.size() && !prefill.busy());
            continue;
        }
        // decode1 and decode2 are fed: submitted before the previous step is polled, tokens from its GPU argmax
        const bool fed = decodes > 0;
        if (fed) assert(tok == ids_of(prev + "/argmax"));
        mllm::DecodeWork work{step++, {}};
        for (size_t b = 0; b < call["rows"].size(); b++) {
            const auto& r = call["rows"][b];
            int req = r["req"];
            work.rows.push_back({req, fed ? -1 : tok[b], r["pos"], pg[req], st[req], {}});
        }
        assert(decode.can_submit());
        decode.submit(work);
        assert(decode.can_submit() == !fed && a.committed == step);
        if (fed) check(prev);
        prev = c;
        decodes++;
    }
    check(prev);
    assert(decodes == 3);
    printf("golden trace through the lanes: 3 prefill chunks, 3 batch-2 decode steps (2 fed on the GPU, each\n"
           "committed before the previous one was polled), greedy ids equal golden\n");

    // Pending reclaim. A gate holds the decode queue, so the step stays in flight until the test opens it.
    id<MTLSharedEvent> gate = [dev newSharedEvent];
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [decode.queue commandBuffer];
        [cb encodeWaitForEvent:gate value:1];
        [cb commit];
    }
    assert(a.admit(1));
    const uint32_t u = a.alloc();
    assert(decode.can_submit() && decode.signaled() == step);
    decode.submit({step, {{0, 0, 0, pg[0], st[0], {}}}});
    a.release({u});  // u's last reference drops while step may still read it
    usleep(50000);
    assert(decode.poll().empty() && decode.signaled() == step);
    a.retire(decode.signaled());
    auto is_free = [&](uint32_t x) { return (a.free[x / 64] >> (x % 64)) & 1; };
    assert(!is_free(u) && a.pending.size() == 1);
    // every other unit can be taken; u is not among them
    const uint32_t others = a.free_units - 1;
    assert(a.admit(others));
    std::vector<uint32_t> taken;
    for (uint32_t i = 0; i < others; i++) taken.push_back(a.alloc()), assert(taken.back() != u);
    gate.signaledValue = 1;
    decode.wait();
    assert(decode.signaled() == step + 1 && decode.poll().size() == 1);
    a.retire(decode.signaled());
    assert(is_free(u) && a.admit(1) && a.alloc() == u);
    printf("pending reclaim: a unit released during step %llu is reused only after the event passes it\nPASS\n",
           (unsigned long long)step);
}
