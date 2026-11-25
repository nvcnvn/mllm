// Kernel checks the golden trace can't give, against CPU fp32/fp64 references on random bf16 inputs:
//  - normalize (RMSNorm) and normalize_and_gate (the gated RMSNorm linear attention uses; the trace has no input /
//    output pair for it alone)
//  - GpuOps::multiply_weights on every kernel it can pick: qmv_fast (1 row), qmv_wide nv 2..5, qmm_t_splitk aligned and
//    not, qmm_t aligned and not (the trace only reaches qmv_wide nv 2 and qmm_t_splitk); decode
//    (workspace.tokens_per_request == 1, 7..16 rows): qmm_t_decode BM 8 / 16, split-k and not (the next-token scores'
//    multiply), and the 6 / 17-row neighbours that stay MLX's
// Then times each projection shape at 1, 8, 16, 512 rows next to tools/roofline.py's bound ("8d": decode).
#include <cassert>
#include <random>

#include "golden.h"
#include "lanes.h"

static float bf16(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &widened, 4);
    return value;
}
static uint16_t to_bf16(float value) {  // round to nearest even (no NaNs here)
    uint32_t bits;
    memcpy(&bits, &value, 4);
    return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
static const uint16_t* bits(mllm::Buf buffer) {
    return (const uint16_t*)((const uint8_t*)buffer.buffer.contents + buffer.offset);
}

static bool close(const char* what, const uint16_t* got, const std::vector<double>& want) {
    double dot = 0, got_squared = 0, want_squared = 0;
    for (size_t i = 0; i < want.size(); i++) {
        double got_value = bf16(got[i]);
        dot += got_value * want[i], got_squared += got_value * got_value, want_squared += want[i] * want[i];
    }
    double cos = dot / std::sqrt(got_squared * want_squared);
    if (cos < 0.99999) fprintf(stderr, "%s: cos %.7f\n", what, cos);
    return cos >= 0.99999;
}

// GpuOps' friend: the kernel helpers are private (they take Metal buffers).
struct mllm::GpuOpsTest {
    const ModelSpec& model;
    const Weights& weights;
    Memory& memory;
    GpuOps& ops;
    Workspace& workspace;
    id<MTLCommandQueue> queue;
    // input: the kernels' input; output: their output; second: the gate of normalize_and_gate, or a second output
    id<MTLBuffer> input, output, second;
    std::mt19937 random{0};
    std::normal_distribution<float> normal{};

    static void run();

    // count random bf16 values of standard deviation `scale`
    void fill(id<MTLBuffer> buffer, size_t count, float scale) {
        uint16_t* values = (uint16_t*)buffer.contents;
        for (size_t i = 0; i < count; i++) values[i] = to_bf16(normal(random) * scale);
    }
    const uint16_t* values(id<MTLBuffer> buffer) const { return (const uint16_t*)buffer.contents; }

    void check_norms();
    void check_multiply_paths();
    void check_large_tiles();
    void time_projections();
    std::vector<double> cpu_multiply(const QuantizedWeights& projection, int rows) const;
    std::vector<QuantizedWeights> weight_copies(QuantizedWeights GpuOps::Layer::*projection);
    void time_shape(const char* name, const std::vector<QuantizedWeights>& copies, int rows, bool decode);
};

int main() {
    @autoreleasepool {
        mllm::GpuOpsTest::run();
    }
}

void mllm::GpuOpsTest::run() {
    ModelSpec model = spec::load(golden::MODEL);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    Weights weights(model.path, device);
    Memory memory(model, device, 64 << 20);
    GpuOps ops(memory, weights);
    Workspace workspace = ops.workspace(1, 1, 64);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    GpuOpsTest test{model, weights, memory, ops, workspace, queue, nil, nil, nil};
    const size_t largest = (size_t)8224 * model.hidden;  // the largest input / output below (qmm_t unaligned), in elements
    test.input = memory.buffer(largest * 2), test.output = memory.buffer(largest * 2), test.second = memory.buffer(largest * 2);

    test.check_norms();
    test.check_multiply_paths();
    test.check_large_tiles();
    test.time_projections();
    printf("PASS\n");
}

// normalize (hidden, final norm weight) and normalize_and_gate (linear_value_dim, layer 0's linear_attn.norm), input in
// place
void mllm::GpuOpsTest::check_norms() {
    auto layer_0_norm = weights["language_model.model.layers.0.linear_attn.norm.weight"];
    const uint16_t* input_values = values(input);
    const uint16_t* gate_values = values(second);
    for (bool gated : {false, true}) {
        int rows = 64, width = gated ? model.linear_value_dim : model.hidden;
        Buf weight = gated ? Buf{layer_0_norm.buffer, layer_0_norm.offset} : ops.final_norm;
        fill(input, (size_t)rows * width, 3), fill(second, (size_t)rows * width, 2);
        std::vector<double> want((size_t)rows * width);
        for (int row = 0; row < rows; row++) {
            const uint16_t* row_values = input_values + (size_t)row * width;
            float sum_of_squares = 0;
            for (int i = 0; i < width; i++) sum_of_squares += bf16(row_values[i]) * bf16(row_values[i]);
            float inverse_rms = 1 / std::sqrt(sum_of_squares / width + 1e-6f);
            for (int i = 0; i < width; i++) {
                double value = bf16(to_bf16(bf16(row_values[i]) * inverse_rms)) * bf16(bits(weight)[i]);
                double gate = bf16(gate_values[(size_t)row * width + i]);
                want[(size_t)row * width + i] = gated ? value * (gate / (1 + std::exp(-gate))) : value;
            }
        }
        encode_and_wait(queue, workspace, [&] {
            ops.begin(workspace);
            if (gated) ops.normalize_and_gate(weight, {input}, {second}, {input}, rows, width);
            else ops.normalize(weight, {input}, {input}, rows, width);
        });
        assert(close(gated ? "gated_rms_norm" : "rms_norm", input_values, want));
    }
}

// multiply_weights: output = input @ dequantize(weights)ᵀ in fp64 from the same bf16 inputs
std::vector<double> mllm::GpuOpsTest::cpu_multiply(const QuantizedWeights& projection, int rows) const {
    std::vector<double> want((size_t)rows * projection.out_size, 0), weight_row(projection.in_size);
    const uint32_t* packed = (const uint32_t*)((const uint8_t*)projection.packed.buffer.contents + projection.packed.offset);
    const uint16_t* input_values = values(input);
    for (int out = 0; out < projection.out_size; out++) {
        for (int in = 0; in < projection.in_size; in++) {
            int group = out * (projection.in_size / 64) + in / 64;
            int quantized = (packed[(size_t)out * (projection.in_size / 8) + in / 8] >> (4 * (in % 8))) & 15;
            weight_row[in] = bf16(bits(projection.scales)[group]) * quantized + bf16(bits(projection.biases)[group]);
        }
        for (int row = 0; row < rows; row++) {
            double sum = 0;
            for (int in = 0; in < projection.in_size; in++) sum += bf16(input_values[(size_t)row * projection.in_size + in]) * weight_row[in];
            want[(size_t)row * projection.out_size + out] = sum;
        }
    }
    return want;
}

void mllm::GpuOpsTest::check_multiply_paths() {
    const GpuOps::Layer& layer_0 = ops.layers[0];
    const GpuOps::Layer& layer_3 = ops.layers[3];
    struct Case {
        const char* kernel;
        const QuantizedWeights& weights;
        int rows;
        bool decode = false;
    } cases[] = {
        {"qmv_fast", layer_0.feed_forward_down, 1},
        {"qmv_wide nv2", layer_0.output_gate_projection, 2},
        {"qmv_wide nv3", layer_0.attention_output, 3},
        {"qmv_wide nv4", layer_3.query_projection, 8},
        {"qmv_wide nv5", layer_0.update_rate_projection, 5},
        {"qmm_t_splitk aligned", layer_3.key_projection, 16},
        {"qmm_t_splitk unaligned", layer_0.decay_projection, 32},
        {"qmm_t aligned", layer_3.value_projection, 1024},
        {"qmm_t unaligned", layer_0.decay_projection, 8224},
        {"decode M 6 (qmv_wide)", layer_0.query_key_value_projection, 6, true},
        {"qmm_t_decode BM 8 M 7", layer_0.feed_forward_down, 7, true},
        {"qmm_t_decode BM 8 M 8", layer_0.query_key_value_projection, 8, true},
        {"qmm_t_decode BM 16 M 9", layer_3.query_projection, 9, true},
        {"qmm_t_decode BM 16 M 13", layer_0.attention_output, 13, true},
        {"qmm_t_decode BM 16 M 16", layer_0.feed_forward_gate, 16, true},
        {"decode M 16 k_proj (MLX)", layer_3.key_projection, 16, true},
        {"decode M 17 (splitk)", layer_0.feed_forward_down, 17, true},
        {"qmm_t_decode no split (lm_head) M 9", ops.lm, 9, true},
    };
    for (const Case& test_case : cases) {
        fill(input, (size_t)test_case.rows * test_case.weights.in_size, 1);
        workspace.tokens_per_request = test_case.decode ? 1 : test_case.rows;
        encode_and_wait(queue, workspace, [&] {
            ops.begin(workspace);
            ops.multiply_weights(test_case.weights, {input}, {output}, test_case.rows);
        });
        assert(close(test_case.kernel, values(output), cpu_multiply(test_case.weights, test_case.rows)));
    }
    printf("normalize, normalize_and_gate and %zu multiply_weights kernel paths match the CPU\n", std::size(cases));
}

// qmm_t: MLX's 32 x 32 tiles and the 64 x 64 ones give the same bytes (each output's inner sum in the same order
// through the same MMA), on full and partial row tiles (100 = 64 + 36, 200 = 3 x 64 + 8), out_size % 64 == 0 and == 32
void mllm::GpuOpsTest::check_large_tiles() {
    const int large_tiles_from_rows = ops.large_tiles_from_rows;
    for (int rows : {100, 200})
        for (int cut : {0, 32}) {
            QuantizedWeights gate_weights = ops.layers[0].feed_forward_gate;
            gate_weights.out_size -= cut;
            fill(input, (size_t)rows * gate_weights.in_size, 1);
            workspace.tokens_per_request = rows;
            for (int from_rows : {INT_MAX, 0}) {
                const size_t dispatches_before = ops.dispatch_count;
                ops.large_tiles_from_rows = from_rows;
                encode_and_wait(queue, workspace, [&] {
                    ops.begin(workspace);
                    ops.multiply_weights(gate_weights, {input}, {from_rows ? output : second}, rows);
                });
                assert(ops.dispatch_count - dispatches_before == 1 && "qmm_t, not split-k");
            }
            ops.large_tiles_from_rows = large_tiles_from_rows;
            assert(!memcmp(values(output), values(second), (size_t)rows * gate_weights.out_size * 2) &&
                   "qmm_t 64 x 64 tiles != MLX's 32 x 32");
        }
    printf("qmm_t 64 x 64 tiles == 32 x 32, 100 / 200 rows, out_size %% 64 = 0 / 32\n");
}

// Each layer's weights of one projection, copied (repeated) into >= 64 MiB of fresh buffer
std::vector<mllm::QuantizedWeights> mllm::GpuOpsTest::weight_copies(QuantizedWeights GpuOps::Layer::*projection) {
    std::vector<const QuantizedWeights*> sources;
    for (const GpuOps::Layer& layer : ops.layers)
        if ((layer.*projection).out_size) sources.push_back(&(layer.*projection));
    const int out_size = sources[0]->out_size, in_size = sources[0]->in_size;
    const size_t packed_bytes = (size_t)out_size * in_size / 2, scale_bytes = (size_t)out_size * in_size / 64 * 2;
    const size_t copy_bytes = packed_bytes + 2 * scale_bytes;  // all 16-aligned
    const size_t copy_count = std::max(sources.size(), ((size_t)64 << 20) / copy_bytes + 1);
    id<MTLBuffer> pool = memory.buffer(copy_count * copy_bytes);
    auto copy = [&](Buf source, size_t offset, size_t bytes) {
        memcpy((uint8_t*)pool.contents + offset, (const uint8_t*)source.buffer.contents + source.offset, bytes);
        return Buf{pool, offset};
    };
    std::vector<QuantizedWeights> copies;
    for (size_t c = 0; c < copy_count; c++) {
        const QuantizedWeights& source = *sources[c % sources.size()];
        size_t offset = c * copy_bytes;
        copies.push_back({copy(source.packed, offset, packed_bytes), copy(source.scales, offset + packed_bytes, scale_bytes),
                          copy(source.biases, offset + packed_bytes + scale_bytes, scale_bytes), out_size, in_size});
    }
    return copies;
}

// Best of 3 runs through every copy, per multiply, next to the roofline bound
void mllm::GpuOpsTest::time_shape(const char* name, const std::vector<QuantizedWeights>& copies, int rows, bool decode) {
    const int out_size = copies[0].out_size, in_size = copies[0].in_size;
    workspace.tokens_per_request = decode ? 1 : rows;
    double best = 1e9;
    for (int trial = 0; trial < 3; trial++) @autoreleasepool {
        best = std::fmin(best, encode_and_wait(queue, workspace, [&] {
            ops.begin(workspace);
            for (const QuantizedWeights& copy : copies) ops.multiply_weights(copy, {input}, {output}, rows);
        }) / copies.size());
    }
    double bytes = (double)out_size * in_size * 4.5 / 8 + 2.0 * rows * in_size + 2.0 * rows * out_size;
    double bound = std::fmax(bytes / 210e9, 2.0 * rows * out_size * in_size / 7.5e12), gigabytes_per_second = bytes / best / 1e9;
    printf("%-13s %5d %5d %4d%s %9.1f %9.1f %5.0f%% %7.0f%s\n", name, out_size, in_size, rows, decode ? "d" : " ",
           best * 1e6, bound * 1e6, 100 * bound / best, gigabytes_per_second,
           gigabytes_per_second > 273 ? "  above DRAM peak: cached" : "");
}

// Timing: each shape rotates through >= 64 MiB of weight copies (each layer's, repeated), so weights come from
// DRAM as in a real step; fewer copies stay in the SLC and read faster than DRAM peak (273 GB/s, flagged).
// Roofline (tools/roofline.py knobs): max(bytes / bandwidth, flops / FLOPS), bandwidth = 210 GB/s, FLOPS = 7.5 TFLOPS,
// bytes = weights at 4.5 bits + bf16 input and output.
void mllm::GpuOpsTest::time_projections() {
    using Layer = GpuOps::Layer;
    struct Shape {
        const char* name;
        QuantizedWeights Layer::*projection;
    } shapes[] = {{"in_proj_qkv", &Layer::query_key_value_projection}, {"in_proj_z", &Layer::output_gate_projection},
                  {"in_proj_a/b", &Layer::decay_projection},           {"q_proj", &Layer::query_projection},
                  {"k/v_proj", &Layer::key_projection},                {"o/out_proj", &Layer::attention_output},
                  {"gate/up_proj", &Layer::feed_forward_gate},         {"down_proj", &Layer::feed_forward_down}};
    fill(input, (size_t)512 * 3584, 1);
    @autoreleasepool {
        // ~0.2 s of work first, so the GPU clock has ramped up before the first timed shape
        encode_and_wait(queue, workspace, [&] {
            ops.begin(workspace);
            for (int r = 0; r < 200; r++) ops.multiply_weights(ops.layers[0].query_key_value_projection, {input}, {output}, 512);
        });
    }
    printf("%-13s %5s %5s %5s %9s %9s %6s %7s\n", "proj", "out", "in", "rows", "us", "bound us", "%bound", "GB/s");
    for (const Shape& shape : shapes) @autoreleasepool {
        const std::vector<QuantizedWeights> copies = weight_copies(shape.projection);
        for (auto [rows, decode] : {std::pair{1, false}, {8, false}, {8, true}, {16, false}, {16, true}, {512, false}})
            time_shape(shape.name, copies, rows, decode);
    }
}
