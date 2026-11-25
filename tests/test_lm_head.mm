// pick_next_tokens + sampling. (1) golden decode0..2: workspace.residual after apply_weights(layers) -> next-token
// scores and greedy ids equal decodeN/logits, decodeN/argmax. (2) lm_head.safetensors: the quantized multiply at
// out_size = vocab_size for 1..16 rows (MLX switches kernels by row count) bit-exact at 1..6 and 13..16 rows, by
// tolerance at 7..12 (qmm_t_decode), rows past the row count untouched, the greedy token = the lowest index of the
// row's highest score. (3) greedy ties and edges. (4) make_sampler semantics, drawn on the GPU: 5k seeded draws match
// the expected distribution (chi-square), top_p drops exactly the tail apply_top_p drops. (5) a batch mixing greedy and
// sampled rows: each row as if alone.
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <string>

#include "golden.h"
#include "lanes.h"

struct mllm::GpuOpsTest {
    static void greedy_tokens(GpuOps& ops, id<MTLCommandQueue> queue, Workspace& workspace, id<MTLBuffer> scores,
                              id<MTLBuffer> output, int rows, int vocab_size) {
        mllm::encode_and_wait(queue, workspace, [&] {
            ops.begin(workspace);
            ops.greedy_tokens({scores}, {output}, rows, vocab_size);
        });
    }
    static std::vector<int> sample_tokens(GpuOps& ops, id<MTLCommandQueue> queue, Workspace& workspace,
                                          const std::vector<SampleParams>& params) {
        workspace.request_count = (int)params.size();
        @autoreleasepool {
            mllm::encode_and_wait(queue, workspace, [&] {
                ops.begin(workspace);
                ops.sample_tokens({workspace.next_token_scores}, {workspace.next_token_ids}, ops.spec.vocab_size, params);
            });
        }
        return ops.next_tokens(workspace);
    }
};

static uint16_t to_bf16(float value) {  // round to nearest even
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return (uint16_t)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
static float from_bf16(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, 4);
    return value;
}
static int cpu_greedy(const uint8_t* row, int vocab_size) {  // bytes: golden tensors may sit at odd offsets
    int best = 0;
    for (int i = 1; i < vocab_size; i++)
        if (golden::to_double("BF16", row, i) > golden::to_double("BF16", row, best)) best = i;
    return best;
}

static const uint16_t MINUS_INFINITY = 0xFF80;

// The case-(4) row: 4 live tokens, the rest at -60 .. -75.5 (~e-27 of the mass in total). 32 values, not one: a row of
// one repeated value makes the counting pass a serial chain (~4x slower per draw)
static const int LIVE_COUNT = 4;

struct LmHeadTest {
    const mllm::ModelSpec& model;
    mllm::GpuOps& ops;
    mllm::Workspace& workspace;
    id<MTLCommandQueue> queue;
    id<MTLBuffer> ids;
    int vocab_size, max_batch;
    std::pair<int, float> live[LIVE_COUNT];

    uint8_t* score_bytes() const { return (uint8_t*)workspace.next_token_scores.contents; }
    uint16_t* scores() const { return (uint16_t*)workspace.next_token_scores.contents; }

    bool exact(const std::string& name, const mllm::Tensor& want, const uint8_t* data) const {
        bool same = !memcmp(score_bytes(), data, want.nbytes);
        if (!same) golden::matches(name, want, data, score_bytes(), 1, 0);  // prints cos and max |diff|
        return same;
    }

    std::vector<int> pick(int request_count) {
        workspace.request_count = request_count;
        @autoreleasepool {
            mllm::encode_and_wait(queue, workspace, [&] { ops.pick_next_tokens(workspace); });
        }
        return ops.next_tokens(workspace);  // greedy
    }

    int draw(mllm::SampleParams params) { return mllm::GpuOpsTest::sample_tokens(ops, queue, workspace, {params})[0]; }

    // (1) golden decode steps; returns how many
    int check_golden_decode_steps(const nlohmann::json& calls, const mllm::Safetensors& trace) {
        int steps = 0;
        for (const auto& call : calls) {
            std::string call_name = call["name"];
            if (!call_name.starts_with("decode")) continue;
            char key[64];
            snprintf(key, sizeof key, "%s/L%02d/dense/x", call_name.c_str(), model.n_layers());
            const mllm::Tensor& residual = trace.at(key);
            const mllm::Tensor &want = trace.at(call_name + "/logits"), &want_ids = trace.at(call_name + "/argmax");
            memcpy(workspace.residual.contents, trace.data(residual), residual.nbytes);
            std::vector<int> got = pick((int)residual.shape[0]);
            assert(exact(call_name + "/logits", want, trace.data(want)));
            for (size_t row = 0; row < got.size(); row++) assert(got[row] == golden::load<int32_t>(trace.data(want_ids), row));
            steps++;
        }
        return steps;
    }

    // (2) 1..16 rows at out_size = vocab_size
    void check_row_counts(const mllm::Safetensors& sweep) {
        const mllm::Tensor& residual = sweep.at("x");
        memcpy(workspace.residual.contents, sweep.data(residual), residual.nbytes);
        for (int rows = 1; rows <= 16; rows++) check_row_count(sweep, rows);
    }

    void check_row_count(const mllm::Safetensors& sweep, int rows) {
        const mllm::Tensor& want = sweep.at(std::to_string(rows) + "/logits");
        const size_t row_bytes = (size_t)vocab_size * 2, rest = (max_batch - rows) * row_bytes;
        memset(score_bytes() + rows * row_bytes, 0x5a, rest);  // sentinel: rows past `rows` must stay untouched (padding-row stores)
        std::vector<int> got = pick(rows);
        for (size_t i = 0; i < rest; i++) assert(score_bytes()[rows * row_bytes + i] == 0x5a);
        const std::string name = std::to_string(rows) + " rows";
        if (rows < 7 || rows >= 13) {  // qmm_t_decode at 13..16 rows happens to round as MLX's qmm_t does: keep it exact
            assert(exact(name, want, sweep.data(want)));
            for (int row = 0; row < rows; row++)
                assert(got[row] == cpu_greedy(sweep.data(want) + (size_t)row * vocab_size * 2, vocab_size));
        } else {  // qmm_t_decode (GpuOps::multiply_weights, decode at 7..16 rows) at 7..12: by tolerance, greedy of our own scores
            assert(golden::matches(name, want, sweep.data(want), score_bytes(), 0.99999, 0.0625));
            for (int row = 0; row < rows; row++)
                assert(got[row] == cpu_greedy(score_bytes() + (size_t)row * vocab_size * 2, vocab_size));
        }
    }

    // (3) greedy: ties to the lowest index, the highest score at the last index, all -inf -> 0, NaN-free rows only
    void check_greedy_edges() {
        uint16_t* score = scores();
        const size_t vocab = vocab_size;
        std::fill(score, score + 4 * vocab, to_bf16(-3));
        score[1000] = score[200000] = score[vocab - 1] = to_bf16(5);         // row 0: tie, lowest wins
        score[vocab + vocab - 1] = to_bf16(1);                               // row 1: the last index
        std::fill(score + 2 * vocab, score + 3 * vocab, MINUS_INFINITY);     // row 2: all -inf (MLX: 0)
        score[3 * vocab + 7] = score[3 * vocab + 6] = to_bf16(-2.5);          // row 3: adjacent tie
        mllm::GpuOpsTest::greedy_tokens(ops, queue, workspace, workspace.next_token_scores, ids, 4, vocab_size);
        const int32_t* token = (const int32_t*)ids.contents;
        assert(token[0] == 1000 && token[1] == vocab_size - 1 && token[2] == 0 && token[3] == 6);
    }

    void write_live_row() {
        uint16_t* score = scores();
        for (int i = 0; i < vocab_size; i++) score[i] = to_bf16(-60 - (i % 32) * 0.5f);
        for (auto [token, value] : live) score[token] = to_bf16(value);
    }

    // (temperature, top_p, tokens kept: apply_top_p in Python on this row)
    struct SamplingCase {
        float temperature, top_p;
        int kept;
    };

    // (4) sampling on one row: chi-square over the kept tokens, 99.9% quantile per degree of freedom (1..3)
    static constexpr int DRAWS = 5000;
    static constexpr double CHI_SQUARED_MAX[] = {0, 10.83, 13.82, 16.27};

    void check_sampling_case(SamplingCase sampling) {
        const uint16_t* score = scores();
        std::mt19937_64 random(42);
        mllm::SampleParams params{sampling.temperature, sampling.top_p, &random};
        std::map<int, int> drawn;
        for (int i = 0; i < DRAWS; i++) drawn[draw(params)]++;
        double total = 0, chi_squared = 0;
        for (int j = 0; j < sampling.kept; j++) total += std::exp(from_bf16(score[live[j].first]) / sampling.temperature);
        for (int j = 0; j < LIVE_COUNT; j++) {
            int token = live[j].first;
            double want = j < sampling.kept ? DRAWS * std::exp(from_bf16(score[token]) / sampling.temperature) / total : 0;
            if (j < sampling.kept) chi_squared += (drawn[token] - want) * (drawn[token] - want) / want;
            assert(j < sampling.kept || drawn[token] == 0);
        }
        printf("  temp %.1f top_p %.2f: chi2 %.2f (df %d, max %.2f)\n", sampling.temperature, sampling.top_p, chi_squared,
               sampling.kept - 1, CHI_SQUARED_MAX[sampling.kept - 1]);
        assert(chi_squared < CHI_SQUARED_MAX[sampling.kept - 1]);
        assert((int)std::count_if(drawn.begin(), drawn.end(), [](auto& entry) { return entry.second > 0; }) == sampling.kept);
        std::mt19937_64 first(7), second(7);  // seeded: the same draws
        for (int i = 0; i < 100; i++)
            assert(draw({sampling.temperature, sampling.top_p, &first}) == draw({sampling.temperature, sampling.top_p, &second}));
    }

    // -inf tokens never drawn; an all -inf row gives 0
    void check_minus_infinity() {
        uint16_t* score = scores();
        std::fill(score, score + vocab_size, MINUS_INFINITY);
        score[123] = to_bf16(1);
        std::mt19937_64 random(3);
        for (int i = 0; i < 200; i++) assert(draw({1, 0.5f, &random}) == 123);
        score[123] = MINUS_INFINITY;
        assert(draw({1, 0, &random}) == 0);
    }

    // top_p with many candidates: token 7 at 5, `middle` tokens at 0, the rest at -10; top_p 0.5 keeps token 7 and the
    // middle tokens (the -10s hold ~0.2% of the mass). middle = 4000 fits the kernel's candidate buffer (4096); 5000
    // takes the whole-row path. Both: no -10 token drawn, token 7 at its odds (chi-square, 1 degree of freedom)
    void check_many_candidates(int middle) {
        uint16_t* score = scores();
        for (int i = 0; i < vocab_size; i++) score[i] = to_bf16(i < 100 + middle && i >= 100 ? 0.0f : -10.0f);
        score[7] = to_bf16(5);
        const double total = std::exp(5.0) + middle, token_7_odds = std::exp(5.0) / total;
        std::mt19937_64 random(5);
        int token_7_draws = 0;
        const int draws = 3000;
        for (int i = 0; i < draws; i++) {
            const int token = draw({1, 0.5f, &random});
            assert(token == 7 || (token >= 100 && token < 100 + middle));
            token_7_draws += token == 7;
        }
        const double expected = draws * token_7_odds;
        const double chi_squared = (token_7_draws - expected) * (token_7_draws - expected) / expected +
                                   (token_7_draws - expected) * (token_7_draws - expected) / (draws - expected);
        printf("  %d candidates: token 7 drawn %d of %d (expected %.0f), chi2 %.2f\n", middle + 1, token_7_draws, draws,
               expected, chi_squared);
        assert(chi_squared < CHI_SQUARED_MAX[1]);
    }

    // (5) rows 0..3 (the case-(4) row in each): greedy, sampled, greedy, sampled. Each row's token as if alone: the
    // greedy token for greedy rows, and for sampled rows the same draw as the same seed gives alone (on row 0)
    void check_mixed_rows() {
        uint16_t* score = scores();
        write_live_row();  // the edge cases above overwrote row 0
        for (int row = 1; row < 4; row++) std::copy(score, score + vocab_size, score + (size_t)row * vocab_size);
        std::mt19937_64 first_batched(11), first_alone(11), second_batched(12), second_alone(12);
        for (int i = 0; i < 50; i++) {
            const std::vector<int> got = mllm::GpuOpsTest::sample_tokens(
                ops, queue, workspace, {{}, {0.7f, 0.7f, &first_batched}, {}, {1.5f, 0.95f, &second_batched}});
            assert(got[0] == 5 && got[2] == 5 && (got[1] == 5 || got[1] == 77));
            assert(got[1] == draw({0.7f, 0.7f, &first_alone}) && got[3] == draw({1.5f, 0.95f, &second_alone}));
        }
    }
};

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string directory = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(directory + "/spec.json"));
    mllm::Safetensors trace(directory + "/trace.safetensors"), sweep(directory + "/lm_head.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    mllm::Memory memory(model, device, 64 << 20);
    const int max_batch = 16, vocab_size = model.vocab_size;
    mllm::GpuOps ops(memory, weights);
    mllm::Workspace workspace = ops.workspace(1, max_batch, 64);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> ids = memory.buffer(max_batch * 4);
    memory.freeze();
    workspace.tokens_per_request = 1;
    LmHeadTest test{model, ops, workspace, queue, ids, vocab_size, max_batch,
                    {{5, 2.0f}, {77, 1.3f}, {vocab_size - 1, 0.8f}, {0, 0.4f}}};

    const int steps = test.check_golden_decode_steps(golden_spec["calls"], trace);
    test.check_row_counts(sweep);
    test.check_greedy_edges();

    test.write_live_row();
    workspace.request_count = 1;
    for (LmHeadTest::SamplingCase sampling : {LmHeadTest::SamplingCase{1, 0, 4}, LmHeadTest::SamplingCase{0.7f, 0.7f, 2},
                                              LmHeadTest::SamplingCase{0.7f, 0.8f, 3},
                                              LmHeadTest::SamplingCase{1.5f, 0.95f, 4}})
        test.check_sampling_case(sampling);
    test.check_minus_infinity();
    for (int middle : {4000, 5000}) test.check_many_candidates(middle);
    test.check_mixed_rows();

    printf("decode steps: %d, next-token scores and greedy ids bit-exact; scores at 1..6, 13..16 rows bit-exact, "
           "7..12 within tolerance, rows past the count untouched; "
           "greedy ties ok;\n"
           "sampling (GPU): 4 cases x %d draws match make_sampler's distribution (chi-square, p 0.001), top_p tails dropped;\n"
           "mixed greedy + sampled rows ok\nPASS\n",
           steps, LmHeadTest::DRAWS);
    assert(steps == 3);
}
