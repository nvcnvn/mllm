// AneOps (ane.h): dense blocks on the ANE, the GPU feeding them its weights. Chunks of 1024 (512 blocks for shorter
// ones). Needs the 0.8B's blocks built for 512 and 1024 (the test prints the command if not).
// a. a size that isn't built names the build command; every block loads.
// b. blocks 0 (linear), 3 (full) and 4 (linear, attention_result from a full layer) on golden prefill0 inputs
//    (32 tokens, padded to 512): input surfaces zero past the 32 tokens; every output within cos 0.999 of the golden
//    GPU tensors.
// c. PrefillLane, every chunk on the ANE, vs on the GPU: worst cos over every full layer's keys / values and linear
//    layer's conv / recurrent state, for the golden chat prompts and repeated-prompt lengths around the chunk sizes.
//    Floors: the measured values less ~1e-3 (fp16 blocks). Two chunks in flight (each with its own weight surfaces)
//    bit-exact with one at a time, also for two requests at once.
// d. Chunks below gpu_below tokens (tails, prefix-cache splits) run on the GPU path.
// e. Engine with ANE prefill: greedy 64 equals golden up to each prompt's first exact tie; decode alongside 2-chunk
//    prefills; a random stress run; cancel / abort with two chunks in flight; a long prompt's greedy vs GPU prefill.
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <unistd.h>

#include "ane.h"
#include "engine.h"
#include "golden.h"

using Tokens = std::vector<int>;

// the golden trace's tensor names (gpu.py's fields) -> the workspace fields they correspond to
static const std::map<std::string, id<MTLBuffer> mllm::Workspace::*> WORKSPACE_FIELD_BY_GOLDEN_NAME = {
    {"q", &mllm::Workspace::query},
    {"k", &mllm::Workspace::key},
    {"v", &mllm::Workspace::value},
    {"gate", &mllm::Workspace::query_gate},
    {"qkv", &mllm::Workspace::query_key_value},
    {"z", &mllm::Workspace::output_gate},
    {"b", &mllm::Workspace::update_rate_input},
    {"a", &mllm::Workspace::decay_input},
    {"x", &mllm::Workspace::residual}};

static float bf16_to_float(uint16_t bits) {
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &word, 4);
    return value;
}

static double cosine(const std::vector<double>& first, const std::vector<double>& second) {
    double dot = 0, first_norm = 0, second_norm = 0;
    for (size_t j = 0; j < first.size(); j++)
        dot += first[j] * second[j], first_norm += first[j] * first[j], second_norm += second[j] * second[j];
    return dot / std::sqrt(first_norm * second_norm);
}

static Tokens golden_token_ids(const mllm::Safetensors& golden_tokens, const std::string& name) {
    const mllm::Tensor& tensor = golden_tokens.at(name);
    Tokens ids(tensor.nbytes / 4);
    for (size_t j = 0; j < ids.size(); j++) ids[j] = golden::load<int32_t>(golden_tokens.data(tensor), j);
    return ids;
}

static Tokens random_tokens(std::mt19937& random_engine, int count, int vocab_size) {
    Tokens tokens(count);
    for (int& token : tokens) token = (int)(random_engine() % vocab_size);
    return tokens;
}

// ---- a. a size that isn't built names the build command; a chunk that isn't a multiple of 32 is refused
static void check_missing_build_is_named(mllm::GpuOps& ops, mllm::Memory& memory, const mllm::ModelSpec& model) {
    try {
        mllm::AneOps none(ops, memory, 2048);  // 512 and 2048: 2048 isn't built
        assert(!"loaded a missing build");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("build('" + model.path + "', [512, 2048])") != std::string::npos);
    }
    try {
        mllm::AneOps none(ops, memory, 1000);
        assert(!"accepted a chunk that isn't a multiple of 32");
    } catch (const std::invalid_argument&) {
    }
}

// ---- b. blocks on golden prefill0 inputs

static std::string layer_name(int layer) {
    char name[8];
    snprintf(name, sizeof name, "L%02d", layer);
    return name;
}

// a golden bf16 tensor -> workspace rows
static void put_golden(const mllm::Safetensors& trace, id<MTLBuffer> buffer, const std::string& name) {
    const mllm::Tensor& tensor = trace.at(name);
    memcpy(buffer.contents, trace.data(tensor), tensor.nbytes);
}

// padding: every input surface is zero past the chunk's tokens
static void check_inputs_zero_padded(const mllm::AneOps::Block& block, int token_count) {
    for (const mllm::AneOps::Activation& input : block.inputs_from_workspace) {
        const mllm::AneOps::Surface& surface = *input.surface[0];
        const uint16_t* values = (const uint16_t*)surface.backing.buffer.contents;
        for (size_t token = token_count; token < (size_t)block.chunk_size; token++)
            for (int column = 0; column < surface.cols; column++)
                assert(values[token * surface.backing.row + column] == 0);
    }
}

// Cos of an output (workspace rows [tokens, columns]) vs the golden tensor in gpu.py's layout: query / key / value
// are [1, heads, tokens, head_dim], the rest [1, tokens, columns]
static double cosine_vs_golden(const uint16_t* output, const mllm::Tensor& golden_tensor, const uint8_t* golden_data,
                               int token_count, int columns) {
    const bool per_head = golden_tensor.shape.size() == 4 && golden_tensor.shape[2] == token_count;
    const int head_dim = per_head ? (int)golden_tensor.shape[3] : columns;
    double dot = 0, output_norm = 0, golden_norm = 0;
    for (int token = 0; token < token_count; token++)
        for (int column = 0; column < columns; column++) {
            size_t golden_index = per_head ? ((size_t)(column / head_dim) * token_count + token) * head_dim +
                                                 column % head_dim
                                           : (size_t)token * columns + column;
            double ours = bf16_to_float(output[(size_t)token * columns + column]),
                   golden_value = bf16_to_float(golden::load<uint16_t>(golden_data, golden_index));
            dot += ours * golden_value, output_norm += ours * ours, golden_norm += golden_value * golden_value;
        }
    return dot / std::sqrt(output_norm * golden_norm);
}

static void check_outputs_against_golden(const mllm::Safetensors& trace, mllm::Workspace& workspace,
                                         const mllm::AneOps::Block& block, int layer, int token_count) {
    for (const mllm::AneOps::Activation& output : block.outputs_to_workspace) {
        std::string name;
        for (const auto& [golden_name, field] : WORKSPACE_FIELD_BY_GOLDEN_NAME)
            if (workspace.*output.field == workspace.*field) name = golden_name;
        const mllm::Tensor& golden_tensor = trace.at("prefill0/" + layer_name(layer) + "/dense/" + name);
        const double cos = cosine_vs_golden((const uint16_t*)(workspace.*output.field).contents, golden_tensor,
                                            trace.data(golden_tensor), token_count, output.surface[0]->cols);
        printf("b. block %d %-4s: cos vs golden %.6f\n", layer, name.c_str(), cos);
        assert(cos >= 0.999);
    }
}

static void check_blocks_on_golden_inputs(mllm::AneOps& ane, mllm::Workspace& workspace, id<MTLCommandQueue> queue,
                                          const mllm::Safetensors& trace) {
    const int token_count = 32;  // golden prefill0: 32 tokens at position 0
    workspace.request_count = 1, workspace.tokens_per_request = token_count;
    const mllm::Tensor& tokens = trace.at("prefill0/tokens");
    memcpy(workspace.token_ids.contents, trace.data(tokens), tokens.nbytes);
    for (int layer : {0, 3, 4}) {
        const mllm::AneOps::Block& block = ane.block(layer, token_count);
        assert(block.chunk_size == 512);
        if (layer > 0) {
            put_golden(trace, workspace.residual, "prefill0/" + layer_name(layer - 1) + "/dense/x");
            put_golden(trace, workspace.attention_result, "prefill0/" + layer_name(layer - 1) + "/attn/o");
        }
        mllm::encode_and_wait(queue, workspace, [&] { ane.prepare_inputs(layer, workspace, 0); });
        check_inputs_zero_padded(block, token_count);
        mllm::encode_and_wait(queue, workspace, [&] { ane.apply_weights(layer, workspace, 0); });
        check_outputs_against_golden(trace, workspace, block, layer, token_count);
    }
}

// ---- c. and d.: prefill through the lanes into arena units, and the arena's keys / values and state read back

struct Units {
    std::vector<uint32_t> pages, state;
};

struct LaneHarness {
    const mllm::ModelSpec& model;
    mllm::Memory& memory;
    const int chunk;
    mllm::Arena& arena = memory.arena;
    uint8_t* arena_bytes = (uint8_t*)memory.arena_buffer.contents;

    // fresh units for token_count tokens, state zeroed
    Units fresh_units(int token_count) {
        Units units;
        assert(arena.admit(memory.units_for(token_count) + memory.state_units));
        for (uint32_t j = 0; j < memory.units_for(token_count); j++) units.pages.push_back(arena.alloc());
        for (uint32_t j = 0; j < memory.state_units; j++) units.state.push_back(arena.alloc());
        for (uint32_t unit : units.state) memset(arena_bytes + (size_t)unit * arena.unit_bytes, 0, arena.unit_bytes);
        return units;
    }

    // Every prompt's tokens but the last into fresh units in chunks, chunk j of each prompt before chunk j + 1 of
    // any; two_in_flight: up to 2 chunks in flight, else one at a time
    std::vector<Units> prefill_all(mllm::PrefillLane& lane, const std::vector<Tokens>& prompts, bool two_in_flight) {
        std::vector<Units> units;
        std::vector<mllm::PrefillWork> works;
        for (const Tokens& prompt : prompts) units.push_back(fresh_units((int)prompt.size() - 1));
        for (int position = 0;; position += chunk) {
            const size_t works_before = works.size();
            for (size_t p = 0; p < prompts.size(); p++)
                if (const int length = (int)prompts[p].size() - 1; position < length)
                    works.push_back({(int)p,
                                     Tokens(prompts[p].begin() + position,
                                            prompts[p].begin() + std::min(position + chunk, length)),
                                     position, units[p].pages, units[p].state});
            if (works.size() == works_before) break;
        }
        size_t next = 0, finished = 0;
        while (finished < works.size()) {
            while (next < works.size() && (two_in_flight ? lane.can_submit() : !lane.busy())) lane.submit(works[next++]);
            std::vector<mllm::PrefillWork> done = lane.poll();
            for (const mllm::PrefillWork& work : done) assert(work.start_pos == works[finished++].start_pos);
            if (done.empty()) lane.wait_oldest();
        }
        return units;
    }

    Units prefill(mllm::PrefillLane& lane, const Tokens& prompt) { return prefill_all(lane, {prompt}, false)[0]; }

    void reset() { arena = mllm::Arena(arena.unit_bytes, arena.n_units); }

    // full layer full_layer's keys (0) or values (1), all heads, tokens from .. token_count
    std::vector<double> key_or_value_values(const Units& units, int full_layer, int key_or_value, int token_count,
                                            int from = 0) const {
        std::vector<double> values;
        for (int head = 0; head < model.key_value_heads; head++)
            for (int token = from; token < token_count; token++) {
                const uint8_t* page = arena_bytes + (size_t)units.pages[token / memory.page_tokens] * arena.unit_bytes;
                const uint16_t* vector = (const uint16_t*)(page + memory.kv_offset(full_layer, key_or_value, head,
                                                                                   token % memory.page_tokens));
                for (int j = 0; j < model.head_dim; j++) values.push_back(bf16_to_float(vector[j]));
            }
        return values;
    }

    // linear layer `layer`'s conv (bf16) or delta (fp32) state
    std::vector<double> state_values(const Units& units, int layer, bool delta) const {
        std::vector<double> values;
        const size_t offset = memory.state_offset(layer) + (delta ? memory.conv_bytes : 0),
                     length = delta ? memory.delta_bytes : memory.conv_bytes;
        for (size_t b = offset; b < offset + length; b += delta ? 4 : 2) {
            const uint8_t* bytes = arena_bytes + (size_t)units.state[b / arena.unit_bytes] * arena.unit_bytes +
                                   b % arena.unit_bytes;
            float value;
            if (delta) memcpy(&value, bytes, 4);
            else value = bf16_to_float(*(const uint16_t*)bytes);
            values.push_back(value);
        }
        return values;
    }

    // {worst keys / values cos over tokens from .. token_count, worst state cos}
    std::pair<double, double> worst_cosines(const Units& first, const Units& second, int token_count,
                                            int from = 0) const {
        double worst_key_value = 1, worst_state = 1;
        for (int full_layer = 0, layer = 0; layer < model.n_layers(); layer++)
            if (model.kinds[layer] == "full") {
                for (int key_or_value : {0, 1})
                    worst_key_value = std::fmin(
                        worst_key_value, cosine(key_or_value_values(first, full_layer, key_or_value, token_count, from),
                                                key_or_value_values(second, full_layer, key_or_value, token_count, from)));
                full_layer++;
            } else {
                for (bool delta : {false, true})
                    worst_state = std::fmin(worst_state,
                                            cosine(state_values(first, layer, delta), state_values(second, layer, delta)));
            }
        return {worst_key_value, worst_state};
    }

    // bit-exact keys / values and states
    bool bit_exact(const Units& first, const Units& second, int token_count) const {
        for (int full_layer = 0, layer = 0; layer < model.n_layers(); layer++)
            if (model.kinds[layer] == "full") {
                if (key_or_value_values(first, full_layer, 0, token_count) !=
                        key_or_value_values(second, full_layer, 0, token_count) ||
                    key_or_value_values(first, full_layer, 1, token_count) !=
                        key_or_value_values(second, full_layer, 1, token_count))
                    return false;
                full_layer++;
            } else if (state_values(first, layer, false) != state_values(second, layer, false) ||
                       state_values(first, layer, true) != state_values(second, layer, true)) {
                return false;
            }
        return true;
    }
};

// c. every chunk on the ANE vs on the GPU lane, against floors
static void check_ane_lane_against_gpu_lane(LaneHarness& harness, mllm::PrefillLane& gpu_lane,
                                            mllm::PrefillLane& ane_lane, const std::vector<Tokens>& prompts,
                                            const Tokens& repeated_prompt) {
    struct Case {
        std::string name;
        Tokens prompt;
        double min_cosine;
    };
    std::vector<Case> cases = {
        {"chat/0", prompts[0], 0.998}, {"chat/1", prompts[1], 0.998}, {"chat/2", prompts[2], 0.997}};
    for (auto [length, min_cosine] : {std::pair{500, 0.996}, {1024, 0.995}, {1500, 0.995}, {2560, 0.994}})
        cases.push_back({"repeated " + std::to_string(length),
                         Tokens(repeated_prompt.begin(), repeated_prompt.begin() + length + 1), min_cosine});
    for (const Case& test_case : cases) {
        const int token_count = (int)test_case.prompt.size() - 1;
        auto [worst_key_value, worst_state] = harness.worst_cosines(
            harness.prefill(gpu_lane, test_case.prompt), harness.prefill(ane_lane, test_case.prompt), token_count);
        const double worst = std::fmin(worst_key_value, worst_state);
        printf("c. %-13s %4d tokens: worst K/V/state cos vs GPU lane %.6f (floor %.3f)\n", test_case.name.c_str(),
               token_count, worst, test_case.min_cosine);
        assert(worst >= test_case.min_cosine);
        harness.reset();
    }
}

// c. two chunks in flight == one at a time, bit for bit
static void check_two_chunks_in_flight(LaneHarness& harness, mllm::PrefillLane& ane_lane, const Tokens& two_chunk_prompt,
                                       const Tokens& three_chunk_prompt) {
    for (const auto* prompt : {&two_chunk_prompt, &three_chunk_prompt}) {
        Units one_at_a_time = harness.prefill_all(ane_lane, {*prompt}, false)[0],
              two_in_flight = harness.prefill_all(ane_lane, {*prompt}, true)[0];
        const bool equal = harness.bit_exact(one_at_a_time, two_in_flight, (int)prompt->size() - 1);
        printf("c. %zu tokens, 2 chunks in flight: K/V and state %s one at a time\n", prompt->size() - 1,
               equal ? "bit-exact with" : "DIFFER from");
        assert(equal);
        harness.reset();
    }
}

// c. two requests prefilling at once: each equals its solo run
static void check_two_requests_at_once(LaneHarness& harness, mllm::PrefillLane& ane_lane, const Tokens& three_chunk_prompt,
                                       const Tokens& two_chunk_prompt) {
    Units three_alone = harness.prefill(ane_lane, three_chunk_prompt),
          two_alone = harness.prefill(ane_lane, two_chunk_prompt);
    std::vector<Units> together = harness.prefill_all(ane_lane, {three_chunk_prompt, two_chunk_prompt}, true);
    const bool equal = harness.bit_exact(three_alone, together[0], (int)three_chunk_prompt.size() - 1) &&
                       harness.bit_exact(two_alone, together[1], (int)two_chunk_prompt.size() - 1);
    printf("c. 2560 + 1500 tokens prefilling at once: each %s its solo run\n", equal ? "equals" : "DIFFERS from");
    assert(equal);
    harness.reset();
}

// d. one prompt's full chunks on the ANE lane, then its tail on the GPU lane, chunk by chunk
static Units prefill_ane_chunks_then_gpu_tail(LaneHarness& harness, mllm::PrefillLane& ane_lane,
                                              mllm::PrefillLane& gpu_lane, const Tokens& prompt, int token_count,
                                              int full_chunk_tokens) {
    Units units = harness.fresh_units(token_count);
    for (int position = 0; position < token_count; position += harness.chunk) {
        mllm::PrefillLane& lane = position < full_chunk_tokens ? ane_lane : gpu_lane;
        const int chunk_end = std::min(position + harness.chunk, token_count);
        lane.submit({0, Tokens(prompt.begin() + position, prompt.begin() + chunk_end), position, units.pages, units.state});
        while (lane.poll().empty()) lane.wait_oldest();
    }
    return units;
}

static void check_gpu_tail(LaneHarness& harness, mllm::PrefillLane& gpu_lane, mllm::PrefillLane& ane_lane,
                           const Tokens& source, bool repeated, int token_count) {
    const Tokens prompt(source.begin(), source.begin() + token_count + 1);
    const int full_chunk_tokens = token_count / harness.chunk * harness.chunk;
    Units gpu_units = harness.prefill(gpu_lane, prompt);
    const int gpu_chunks_before = ane_lane.gpu_chunks;
    Units ane_units = harness.prefill(ane_lane, prompt);
    assert(ane_lane.gpu_chunks == gpu_chunks_before + 1);
    Units split_units =
        prefill_ane_chunks_then_gpu_tail(harness, ane_lane, gpu_lane, prompt, token_count, full_chunk_tokens);
    const auto [tail_key_value_cos, state_cos] =
        harness.worst_cosines(gpu_units, ane_units, token_count, full_chunk_tokens);
    const bool equal = harness.bit_exact(ane_units, split_units, token_count);
    printf("d. %s %4d tokens, %2d-token tail on the GPU: cos vs GPU prefill, tail K/V %.6f, state %.6f; %s "
           "ANE chunks + GPU lane tail\n",
           repeated ? "repeated" : "random  ", token_count, token_count - full_chunk_tokens, tail_key_value_cos,
           state_cos, equal ? "bit-exact with" : "DIFFERS from");
    assert(equal && (repeated || (tail_key_value_cos >= 0.999 && state_cos >= 0.999)));
    harness.reset();
}

// d. Tails on the GPU: an ANE chunk of fewer than gpu_below tokens runs on the GPU path
// (workspace.tokens_per_request = its token count, no padding). Bit-exact with the ANE lane's full chunks followed by
// the GPU lane's tail. Against GPU prefill, on the tail tokens' keys / values and the final state: >= 0.999 on random
// prompts (the repetitive one's ANE chunks drift more).
static void check_gpu_tails(LaneHarness& harness, mllm::PrefillLane& gpu_lane, mllm::PrefillLane& ane_lane,
                            const Tokens& repeated_prompt) {
    Tokens random_prompt;
    std::mt19937 random_engine(3);
    for (int j = 0; j < 2101; j++) random_prompt.push_back((int)(random_engine() % harness.model.vocab_size));
    for (const Tokens* source : std::initializer_list<const Tokens*>{&repeated_prompt, &random_prompt})
        for (int token_count : {1100, 2100})
            check_gpu_tail(harness, gpu_lane, ane_lane, *source, source == &repeated_prompt, token_count);
}

// ---- e. the engine with ANE prefill

static Tokens greedy_tokens(mllm::Engine& engine, const Tokens& prompt, int max_tokens) {
    const int request_id = engine.submit(prompt, {.max_tokens = max_tokens});
    Tokens tokens;
    while (engine.requests.count(request_id))
        for (const mllm::Event& event : engine.step()) tokens.push_back(event.token);
    return tokens;
}

// Up to the first exact tie (top-2 scores equal in bf16 under mlx-lm, teacher-forced: chat/1 at token 38, chat/2 at
// 28; chat/0 has none): the fp16 blocks' drift flips those
static void check_greedy_until_first_tie(mllm::Engine& engine, const std::vector<Tokens>& prompts,
                                         const mllm::Safetensors& golden_tokens, int gpu_below) {
    engine.prefill.gpu_below = 0;  // the chat prompts are short: on the ANE anyway
    const size_t first_tie[3] = {64, 38, 28};
    for (int i = 0; i < 3; i++) {
        const Tokens tokens = greedy_tokens(engine, prompts[i], 64),
                     expected = golden_token_ids(golden_tokens, "greedy/" + std::to_string(i));
        size_t matching = 0;
        while (matching < tokens.size() && matching < expected.size() && tokens[matching] == expected[matching])
            matching++;
        printf("e. engine, ANE prefill: prompt %d greedy: first %zu of %zu tokens equal golden (first tie at %zu)\n", i,
               matching, tokens.size(), first_tie[i]);
        assert(tokens.size() == expected.size() && matching >= first_tie[i]);
    }
    engine.prefill.gpu_below = gpu_below;
}

// d. prefix-cache splits: a fresh 600-token prompt stops at its last full page (576, a snapshot point): 576 tokens
// on the ANE, then the 23-token tail on the GPU
static void check_split_tail_on_gpu(mllm::Engine& engine) {
    std::mt19937 random_engine(4);
    Tokens prompt(600);
    for (int& token : prompt) token = (int)(random_engine() % engine.spec.vocab_size);
    const int gpu_chunks_before = engine.prefill.gpu_chunks;
    greedy_tokens(engine, prompt, 1);
    assert(engine.prefill.gpu_chunks == gpu_chunks_before + 1);
    printf("d. engine: a 600-token prompt's tail after its prefix-cache split (23 tokens) ran on the GPU\n");
}

// d. A chunk ending at a snapshot point holds back its request's next chunk until poll has copied the state. Polled
// late (40 ms between steps), both chunks would be done by then and the copy would hold the next chunk's state too.
// The cached snapshot at 1088 (the last full page: chunks 1024, 64, then 11) equals a lane prefill of those tokens.
static void check_snapshot_after_late_polls(mllm::Engine& engine, LaneHarness& harness, mllm::PrefillLane& ane_lane) {
    std::mt19937 random_engine(5);
    Tokens prompt(1100);
    for (int& token : prompt) token = (int)(random_engine() % engine.spec.vocab_size);
    const int request_id = engine.submit(prompt, {.max_tokens = 1});
    while (engine.requests.count(request_id)) engine.step(), usleep(40000);
    const Tokens head(prompt.begin(), prompt.begin() + 1089);
    mllm::PrefixCache::Match match = engine.cache->match(head);
    assert(match.resume_pages * harness.memory.page_tokens == 1088 && match.snapshot.size() == harness.memory.state_units);
    Units units = harness.prefill(ane_lane, head);
    const uint8_t* engine_arena = (const uint8_t*)engine.mem.arena_buffer.contents;
    const size_t unit_bytes = harness.arena.unit_bytes;
    size_t differing_units = 0;
    for (size_t j = 0; j < match.snapshot.size(); j++)
        differing_units += memcmp(engine_arena + (size_t)match.snapshot[j] * unit_bytes,
                                  harness.arena_bytes + (size_t)units.state[j] * unit_bytes, unit_bytes) != 0;
    printf("d. engine, late polls: the snapshot at 1088 %s a lane prefill of 1088 tokens\n",
           differing_units ? "DIFFERS from" : "equals");
    assert(!differing_units);
    harness.reset();
}

// Decode on the engine thread while the ANE lane's thread prefills (both encode with GpuOps: the lane has its own
// copy; sharing one crashed `mllm bench -b 8 --stream 512 --prefill ane`): prompt 0 still decodes golden's tokens
// while 2-chunk prompts prefill alongside, one after another (chat/0 has no tie)
static void check_decode_alongside_ane_prefill(mllm::Engine& engine, const Tokens& prompt, const Tokens& long_prompt,
                                               const Tokens& golden_greedy) {
    const int request_id = engine.submit(prompt, {.max_tokens = 64});
    Tokens tokens;
    int other_id = -1;
    while (engine.requests.count(request_id)) {
        if (!engine.requests.count(other_id)) other_id = engine.submit(long_prompt, {.max_tokens = 1});
        for (const mllm::Event& event : engine.step())
            if (event.req_id == request_id) tokens.push_back(event.token);
    }
    engine.cancel(other_id);
    while (!engine.requests.empty()) engine.step();
    assert(tokens == golden_greedy);
    printf("e. engine: greedy 64 still golden with 2-chunk ANE prefills running alongside\n");
}

// Random stress: 10 requests of 2..2100 random tokens (up to 3 chunks), greedy and sampled, a few cancels; every
// step returns, events keep coming (watchdog 30 s)
static void check_random_stress(mllm::Engine& engine) {
    std::mt19937 random_engine(1);
    int submitted = 0;
    auto last_event_at = std::chrono::steady_clock::now();
    while (submitted < 10 || !engine.requests.empty()) {
        if (submitted < 10 && engine.requests.size() < 16 && random_engine() % 3 == 0) {
            Tokens prompt(2 + random_engine() % 2099);
            for (int& token : prompt) token = (int)(random_engine() % engine.spec.vocab_size);
            engine.submit(prompt, {.max_tokens = 1 + (int)(random_engine() % 64),
                                   .temperature = random_engine() % 2 ? 0.7f : 0.0f});
            submitted++;
        }
        if (!engine.requests.empty() && random_engine() % 300 == 0)
            engine.cancel(std::next(engine.requests.begin(), random_engine() % engine.requests.size())->first);
        std::vector<mllm::Event> events = engine.step();
        for (const mllm::Event& event : events) assert(event.token >= 0);
        if (!events.empty()) last_event_at = std::chrono::steady_clock::now();
        assert(std::chrono::steady_clock::now() - last_event_at < std::chrono::seconds(30));
    }
    printf("e. engine: random stress with ANE prefill, %d requests, no failure\n", submitted);
}

static bool all_units_free(const mllm::Engine& engine) {
    const mllm::Arena& arena = engine.mem.arena;
    return arena.free_units + arena.cache_only_units == arena.n_units && arena.reserved == 0 && !engine.prefill.busy();
}

static int prefill_chunks_in_flight(const mllm::Engine& engine) {
    int count = 0;
    for (auto& [_, request] : engine.requests) count += request.state == mllm::State::PREFILL ? request.in_flight : 0;
    return count;
}

// steps until no request is left, collecting the events (watchdog 30 s)
static void drain(mllm::Engine& engine, std::vector<mllm::Event>& events) {
    auto started_at = std::chrono::steady_clock::now();
    while (!engine.requests.empty()) {
        for (const mllm::Event& event : engine.step()) events.push_back(event);
        assert(std::chrono::steady_clock::now() - started_at < std::chrono::seconds(30));
    }
}

static long count_events(const std::vector<mllm::Event>& events, int request_id, bool aborted_only) {
    return std::count_if(events.begin(), events.end(), [&](auto& event) {
        return event.req_id == request_id && (!aborted_only || event.token < 0);
    });
}

// 2 requests: x's one chunk and y's first (FIFO takes a request's second chunk before the next request)
static std::pair<int, int> submit_two_chunks_in_flight(mllm::Engine& engine, std::mt19937& random_engine,
                                                       int request_count) {
    const int vocab_size = engine.spec.vocab_size;
    const int x =
        engine.submit(random_tokens(random_engine, request_count == 2 ? 300 : 1300, vocab_size), {.max_tokens = 4});
    const int y =
        request_count == 2 ? engine.submit(random_tokens(random_engine, 1300, vocab_size), {.max_tokens = 4}) : -1;
    while (prefill_chunks_in_flight(engine) < 2) engine.step();
    return {x, y};
}

static void check_cancel_with_two_chunks_in_flight(mllm::Engine& engine, std::mt19937& random_engine, int request_count) {
    const auto [x, y] = submit_two_chunks_in_flight(engine, random_engine, request_count);
    if (request_count == 2) assert(engine.requests.at(x).in_flight == 1 && engine.requests.at(y).in_flight == 1);
    engine.cancel(x);  // y (if any) runs on to its tokens
    std::vector<mllm::Event> events;
    drain(engine, events);
    assert(all_units_free(engine) && count_events(events, y, false) == (request_count == 2 ? 4 : 0));
    printf("e. engine: cancel with 2 chunks in flight (%d request%s): all units back\n", request_count,
           request_count == 2 ? "s" : "");
}

static void check_abort_with_two_chunks_in_flight(mllm::Engine& engine, std::mt19937& random_engine, int request_count) {
    const auto [x, y] = submit_two_chunks_in_flight(engine, random_engine, request_count);
    engine.prefill.fail_next = true;
    bool threw = false;
    std::vector<mllm::Event> events;
    auto started_at = std::chrono::steady_clock::now();
    while (!threw) {
        try {
            for (const mllm::Event& event : engine.step()) events.push_back(event);
        } catch (const std::runtime_error& error) {
            threw = std::string(error.what()).find("injected") != std::string::npos;
            assert(threw);
        }
        assert(std::chrono::steady_clock::now() - started_at < std::chrono::seconds(30));
    }
    drain(engine, events);
    for (int request_id : {x, y})
        if (request_id >= 0) assert(count_events(events, request_id, true) == 1);
    assert(all_units_free(engine));
    printf("e. engine: abort with 2 chunks in flight (%d request%s): aborted, all units back\n", request_count,
           request_count == 2 ? "s" : "");
}

// the submit of a request's second chunk throws while its first is in the lane: aborted (token -1), the first chunk
// finishes in the lane and poll ends the request
static void check_failed_second_submit(mllm::Engine& engine, std::mt19937& random_engine) {
    const int x = engine.submit(random_tokens(random_engine, 1300, engine.spec.vocab_size), {.max_tokens = 4});
    engine.prefill.fail_submit = 2;
    std::vector<mllm::Event> events;
    try {
        engine.step();
        assert(!"the second submit didn't throw");
    } catch (const std::runtime_error& error) {
        assert(std::string(error.what()).find("injected submit") != std::string::npos);
    }
    drain(engine, events);
    assert(count_events(events, x, true) == 1);
    assert(all_units_free(engine));
    printf("e. engine: a failed submit of a second chunk: aborted, all units back\n");
}

// Cancel and abort with two chunks in flight (one request's, or two requests'): every unit comes back, no hang, and
// the lane works afterwards. Random prompts: no prefix-cache hits shorten them.
static void check_cancel_and_abort(mllm::Engine& engine, const Tokens& prompt, const Tokens& golden_greedy) {
    std::mt19937 random_engine(2);
    assert(all_units_free(engine));
    for (int request_count : {1, 2}) check_cancel_with_two_chunks_in_flight(engine, random_engine, request_count);
    for (int request_count : {1, 2}) check_abort_with_two_chunks_in_flight(engine, random_engine, request_count);
    check_failed_second_submit(engine, random_engine);
    assert(greedy_tokens(engine, prompt, 64) == golden_greedy);
    printf("e. engine: greedy 64 golden again after the aborts\n");
}

// a 1300-token prompt (two ANE chunks): greedy 30 after ANE prefill vs after GPU prefill
static void check_long_prompt_against_gpu_prefill(mllm::Engine& ane_engine, const Tokens& long_prompt) {
    const Tokens after_ane = greedy_tokens(ane_engine, long_prompt, 30);
    mllm::Engine gpu_engine(golden::MODEL, 512 << 20, 512, 32, 16, 4096);
    const Tokens after_gpu = greedy_tokens(gpu_engine, long_prompt, 30);
    size_t same = 0;
    while (same < after_gpu.size() && after_ane[same] == after_gpu[same]) same++;
    printf("e. engine, %zu-token prompt: ANE prefill's greedy 30 equal GPU prefill's for %zu tokens\n",
           long_prompt.size(), same);
    assert(after_ane == after_gpu);
}

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    const std::string golden_dir = golden::dir(model);
    mllm::Safetensors trace(golden_dir + "/trace.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    mllm::Memory memory(model, device, 512 << 20);
    mllm::GpuOps ops(memory, weights);

    // a.
    check_missing_build_is_named(ops, memory, model);
    auto load_started_at = std::chrono::steady_clock::now();
    const int chunk = 1024, max_context = 3072;
    mllm::AneOps ane(ops, memory, chunk);
    assert(ane.chunk_sizes == std::vector<int>({512, 1024}) && (int)ane.weight_inputs.size() == model.n_layers());
    printf("a. loaded %zu blocks in %.1f s, %zu surfaces per set\n", ane.blocks.size(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - load_started_at).count(),
           ane.surfaces[0].size());
    mllm::Workspace workspace = ops.workspace(chunk, 1, chunk);
    mllm::AneOps lane_ane(ops, memory, chunk);  // the lane's (b. uses `ane` on this thread)
    mllm::PrefillLane gpu_lane(ops, memory, chunk, max_context), ane_lane(ops, memory, chunk, max_context, &lane_ane);
    memory.freeze();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // b.
    check_blocks_on_golden_inputs(ane, workspace, queue, trace);

    // c.
    mllm::Safetensors golden_tokens(golden_dir + "/tokens.safetensors");
    std::vector<Tokens> prompts = {golden_token_ids(golden_tokens, "chat/0"), golden_token_ids(golden_tokens, "chat/1"),
                                   golden_token_ids(golden_tokens, "chat/2")};
    Tokens repeated_prompt;
    while (repeated_prompt.size() < 2600)
        repeated_prompt.insert(repeated_prompt.end(), prompts[0].begin(), prompts[0].end());
    LaneHarness harness{model, memory, chunk};
    const int gpu_below = ane_lane.gpu_below;
    ane_lane.gpu_below = 0;  // every chunk on the ANE, padded
    harness.prefill(ane_lane, prompts[0]);  // warm up (first predicts are slow)
    harness.reset();
    check_ane_lane_against_gpu_lane(harness, gpu_lane, ane_lane, prompts, repeated_prompt);
    const Tokens three_chunk_prompt(repeated_prompt.begin(), repeated_prompt.begin() + 2561),
        two_chunk_prompt(repeated_prompt.begin(), repeated_prompt.begin() + 1501);
    check_two_chunks_in_flight(harness, ane_lane, two_chunk_prompt, three_chunk_prompt);
    check_two_requests_at_once(harness, ane_lane, three_chunk_prompt, two_chunk_prompt);
    ane_lane.gpu_below = gpu_below;

    // d.
    check_gpu_tails(harness, gpu_lane, ane_lane, repeated_prompt);

    // e.
    mllm::Engine ane_engine(golden::MODEL, 512 << 20, chunk, 32, 16, 4096, "ane");
    const Tokens golden_greedy_0 = golden_token_ids(golden_tokens, "greedy/0");
    check_greedy_until_first_tie(ane_engine, prompts, golden_tokens, gpu_below);
    check_split_tail_on_gpu(ane_engine);
    check_snapshot_after_late_polls(ane_engine, harness, ane_lane);
    const Tokens long_prompt_1300(repeated_prompt.begin(), repeated_prompt.begin() + 1301);
    check_decode_alongside_ane_prefill(ane_engine, prompts[0], long_prompt_1300, golden_greedy_0);
    check_random_stress(ane_engine);
    check_cancel_and_abort(ane_engine, prompts[0], golden_greedy_0);
    check_long_prompt_against_gpu_prefill(ane_engine, long_prompt_1300);
    printf("PASS\n");
}
