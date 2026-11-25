// apply_weights(i) check: for every apply_weights(i) call in the golden trace (147: prefill never runs
// apply_weights(layers)), load its inputs from golden (tokens, or the residual from apply_weights(i - 1) and the
// attention result from apply_attention(i - 1)), run GpuOps::apply_weights, compare every field it wrote.
// Pass: cosine >= 0.99999. Prints per field how many calls were bit-exact and the worst max |diff|.
// Covers: embedding (L00 x), RMSNorm (the attention norm feeds every projection; query / key norm; final norm = L24 x),
// every projection kind and both qmv_wide (decode, 2 rows) and qmm_t_splitk (prefill, 24-32 rows).
#include <cassert>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "lanes.h"

struct FieldStats {
    int calls = 0, exact = 0;
    double worst_cos = 1, max_abs = 0;
};

// golden [requests, heads, tokens, head_dim]; ours [requests, tokens, heads, head_dim]: ours in golden's order
static void reorder_heads_like_golden(const mllm::Tensor& tensor, const uint8_t* ours, std::vector<uint8_t>& reordered) {
    int64_t requests = tensor.shape[0], heads = tensor.shape[1], tokens = tensor.shape[2], head_dim = tensor.shape[3];
    for (int64_t request = 0; request < requests; request++)
        for (int64_t head = 0; head < heads; head++)
            for (int64_t token = 0; token < tokens; token++)
                memcpy(&reordered[(((request * heads + head) * tokens + token) * head_dim) * 2],
                       ours + (((request * tokens + token) * heads + head) * head_dim) * 2, head_dim * 2);
}

struct DenseTest {
    const mllm::ModelSpec& model;
    const mllm::Safetensors& trace;
    mllm::GpuOps& ops;
    mllm::Workspace& workspace;
    id<MTLCommandQueue> queue;
    std::map<std::string, FieldStats> stats;
    bool ok = true;

    void load(const std::string& key, id<MTLBuffer> destination) {
        const mllm::Tensor& tensor = trace.at(key);
        assert(tensor.nbytes <= destination.length);
        memcpy(destination.contents, trace.data(tensor), tensor.nbytes);
    }

    // apply_weights(layer)'s inputs from golden call `call_name`: tokens for layer 0, else the layer before's residual
    // and attention result (previous_layer: "LNN" of layer - 1)
    void load_inputs(const std::string& call_name, int layer, const std::string& previous_layer) {
        if (layer == 0) {
            load(call_name + "/tokens", workspace.token_ids);
        } else {
            load(call_name + "/" + previous_layer + "/dense/x", workspace.residual);
            load(call_name + "/" + previous_layer + "/attn/o", workspace.attention_result);
        }
    }

    void run_apply_weights(int layer) {
        @autoreleasepool {
            mllm::encode_and_wait(queue, workspace, [&] { ops.apply_weights(layer, workspace); });
        }
    }

    // One field the call wrote (golden name `name`, in buffer) against golden `prefix + name`
    void compare_field(const std::string& prefix, const std::string& name, id<MTLBuffer> buffer, std::vector<uint8_t>& got) {
        auto found = trace.tensors.find(prefix + name);
        if (found == trace.tensors.end()) return;
        const mllm::Tensor& tensor = found->second;
        const uint8_t* ours = (const uint8_t*)buffer.contents;
        got.assign(ours, ours + tensor.nbytes);
        if (tensor.shape.size() == 4 && (name == "q" || name == "k" || name == "v"))
            reorder_heads_like_golden(tensor, ours, got);
        FieldStats& field = stats[name];
        field.calls++;
        const uint8_t* want = trace.data(tensor);
        if (!memcmp(got.data(), want, tensor.nbytes)) {
            field.exact++;
            return;
        }
        double dot = 0, got_squared = 0, want_squared = 0;
        for (size_t e = 0; e < tensor.numel(); e++) {
            double got_value = golden::to_double("BF16", got.data(), e), want_value = golden::to_double("BF16", want, e);
            dot += got_value * want_value, got_squared += got_value * got_value, want_squared += want_value * want_value;
            field.max_abs = std::fmax(field.max_abs, std::fabs(got_value - want_value));
        }
        field.worst_cos = std::fmin(field.worst_cos, dot / std::sqrt(got_squared * want_squared));
        if (!golden::matches(prefix + name, tensor, want, got.data(), 0.99999, INFINITY)) ok = false;
    }

    // Every apply_weights call of the trace; returns how many ran
    int check_traced_calls(const nlohmann::json& calls) {
        const std::map<std::string, id<MTLBuffer>> fields = {{"x", workspace.residual}, {"q", workspace.query},
            {"k", workspace.key}, {"v", workspace.value}, {"gate", workspace.query_gate},
            {"qkv", workspace.query_key_value}, {"z", workspace.output_gate}, {"b", workspace.update_rate_input},
            {"a", workspace.decay_input}};
        const int layers = model.n_layers();
        int runs = 0;
        std::vector<uint8_t> got;
        for (const auto& call : calls) {
            std::string call_name = call["name"].get<std::string>();
            const mllm::Tensor& token_ids = trace.at(call_name + "/tokens");
            workspace.request_count = (int)token_ids.shape[0], workspace.tokens_per_request = (int)token_ids.shape[1];
            for (int i = 0; i <= layers; i++) {
                char layer_name[8], previous_layer[8];
                snprintf(layer_name, sizeof layer_name, "L%02d", i), snprintf(previous_layer, sizeof previous_layer, "L%02d", i - 1);
                if (!trace.tensors.count(call_name + "/" + layer_name + "/dense/x")) continue;  // apply_weights(layers) in a prefill call
                if (i && !trace.tensors.count(call_name + "/" + previous_layer + "/attn/o")) continue;  // big models: a few layers (dump_golden)
                runs++;
                load_inputs(call_name, i, previous_layer);
                run_apply_weights(i);
                for (const auto& [name, buffer] : fields) compare_field(call_name + "/" + layer_name + "/dense/", name, buffer, got);
            }
        }
        return runs;
    }

    // One row (no golden call has it): the merged multiplies (GpuOps::merge_projections) give the same bytes as the
    // separate ones. Row 0 of decode0's inputs; output_gate / update_rate / decay / value are read where the merged row
    // leaves them (workspace.query_key_value, workspace.query_and_gate). Returns how many layers were checked.
    int check_merged_single_row() {
        const int layers = model.n_layers();
        workspace.request_count = 1, workspace.tokens_per_request = 1;
        int checked = 0;
        for (int i = 0; i <= layers; i++) {
            char previous_layer[8];
            snprintf(previous_layer, sizeof previous_layer, "L%02d", i - 1);
            if (i && !trace.tensors.count(std::string("decode0/") + previous_layer + "/attn/o")) continue;
            check_merged_layer(i, previous_layer);
            checked++;
        }
        return checked;
    }

    // (name, the separate path's buffer, the merged path's buffer and byte offset, bytes)
    using MergedField = std::tuple<std::string, id<MTLBuffer>, id<MTLBuffer>, size_t, size_t>;

    std::vector<MergedField> merged_fields(int layer) const {
        const int layers = model.n_layers();
        const size_t element = 2, conv_dim = model.conv_dim();
        const size_t linear_value_size = (size_t)model.linear_value_heads * model.linear_value_dim;
        const size_t linear_value_heads = model.linear_value_heads;
        const size_t query_size = (size_t)model.query_heads * model.head_dim;
        const size_t key_value_size = (size_t)model.key_value_heads * model.head_dim;
        const bool linear = layer < layers && model.kinds[layer] == "linear", full = layer < layers && !linear;
        std::vector<MergedField> fields = {{"x", workspace.residual, workspace.residual, 0, model.hidden * element}};
        if (linear)
            fields.insert(fields.end(),
                {{"qkv", workspace.query_key_value, workspace.query_key_value, 0, conv_dim * element},
                 {"z", workspace.output_gate, workspace.query_key_value, conv_dim * element, linear_value_size * element},
                 {"b", workspace.update_rate_input, workspace.query_key_value, (conv_dim + linear_value_size) * element,
                  linear_value_heads * element},
                 {"a", workspace.decay_input, workspace.query_key_value, (conv_dim + linear_value_size + linear_value_heads) * element,
                  linear_value_heads * element}});
        if (full)
            fields.insert(fields.end(),
                {{"q", workspace.query, workspace.query, 0, query_size * element},
                 {"gate", workspace.query_gate, workspace.query_gate, 0, query_size * element},
                 {"k", workspace.key, workspace.key, 0, key_value_size * element},
                 {"v", workspace.value, workspace.query_and_gate, (2 * query_size + key_value_size) * element, key_value_size * element}});
        return fields;
    }

    void check_merged_layer(int layer, const char* previous_layer) {
        std::vector<MergedField> fields = merged_fields(layer);
        std::vector<std::vector<uint8_t>> want;
        for (bool merge : {false, true}) {
            load_inputs("decode0", layer, previous_layer);
            for (auto& [name, separate, merged, offset, bytes] : fields)
                memset((uint8_t*)merged.contents + offset, 0xFF, bytes), memset(separate.contents, 0xFF, bytes);
            ops.merge_projections = merge;
            run_apply_weights(layer);
            for (size_t k = 0; k < fields.size(); k++) {
                auto& [name, separate, merged, offset, bytes] = fields[k];
                const uint8_t* output = merge ? (const uint8_t*)merged.contents + offset : (const uint8_t*)separate.contents;
                if (!merge) want.emplace_back(output, output + bytes);
                else if (memcmp(output, want[k].data(), bytes))
                    printf("L%02d %s: merged one-row multiply differs\n", layer, name.c_str()), ok = false;
            }
        }
    }
};

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    std::string directory = golden::dir(model);
    nlohmann::json golden_spec = nlohmann::json::parse(std::ifstream(directory + "/spec.json"));
    mllm::Safetensors trace(directory + "/trace.safetensors");
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    mllm::Memory memory(model, device, 64 << 20);
    int chunk = golden_spec["chunk"].get<int>(), max_batch = 2;
    mllm::GpuOps ops(memory, weights);
    mllm::Workspace workspace = ops.workspace(chunk, max_batch, 64);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    memory.freeze();
    DenseTest test{model, trace, ops, workspace, queue, {}, true};
    const int layers = model.n_layers();

    const int runs = test.check_traced_calls(golden_spec["calls"]);
    for (const auto& [name, field] : test.stats)
        printf("%-5s bit-exact %3d/%3d  worst cos %.7f  max |diff| %g\n", name.c_str(), field.exact, field.calls,
               field.worst_cos, field.max_abs);
    int prefills = 0;
    for (const auto& call : golden_spec["calls"]) prefills += call["name"].get<std::string>().starts_with("prefill");
    assert((layers > 24 || runs == (int)golden_spec["calls"].size() * (layers + 1) - prefills) && runs &&
           test.stats["x"].calls == runs);
    assert(test.ok);

    const int checked = test.check_merged_single_row();
    assert(test.ok && (layers > 24 || checked == layers + 1) && checked);
    printf("one row: merged == separate, %d apply_weights calls\n", checked);
    printf("PASS\n");
}
