// The MTP draft head (g22, docs/speculative.md) against MLX's (tools/spec/accept.py), on tools/spec/mtp.py golden's
// three golden chat prompts + 128 greedy tokens of the 27B. Checked by tolerance (user decision, 2026-11-26):
//  1. prefill: PrefillLane over the whole sequence feeds the head; its K/V ~ the reference's teacher-forced pass.
//  2. the head alone over the reference's hidden states (one pass): its K/V ~ the reference's.
//  3. drafts, k = 2, at every state c of the generated part: draft_tokens at position c - 1 from the reference's
//     hidden state -> both drafts equal accept.py's on >= 99% of positions; step 1's scores ~ the reference's logits.
// Models without mtp.safetensors (the 0.8B) skip. At 27B: under the heavy lock, MLLM_MODEL=mlx-community/Qwen3.6-27B-4bit.
#include <cassert>
#include <cstdio>
#include <vector>

#include "golden.h"
#include "lanes.h"

using mllm::Buf;

struct Prompt {
    std::vector<int> seq;
    int prompt_tokens;
};

int main() {
    mllm::ModelSpec model = mllm::spec::load(golden::MODEL);
    if (!model.mtp) {
        printf("no MTP head for %s (tools/spec/mtp.py weights): skipped\nPASS\n", model.path.c_str());
        return 0;
    }
    const std::string reference_path = golden::dir(model) + "/mtp_ref.safetensors";
    if (!std::filesystem::exists(reference_path)) {
        fprintf(stderr, "no %s\nrun under the heavy lock: .venv/bin/python tools/spec/mtp.py golden\n", reference_path.c_str());
        return 1;
    }
    mllm::Safetensors reference(reference_path);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mllm::Weights weights(model.path, device);
    const int chunk = 1024, max_context = 1024, hidden = model.hidden, head_dim = model.head_dim,
              key_value_heads = model.key_value_heads, mtp_index = (int)model.full_layers().size();
    mllm::Memory memory(model, device, mllm::Memory::arena_for(model, 1, max_context));
    mllm::GpuOps ops(memory, weights);
    mllm::Workspace ws = ops.workspace(chunk, 2, max_context);
    mllm::PrefillLane prefill(ops, memory, chunk, max_context);
    mllm::StepTables tables = memory.step_tables(1, max_context);
    id<MTLBuffer> hidden_rows = memory.buffer((size_t)max_context * hidden * mllm::ELEM);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    memory.freeze();
    std::vector<uint32_t> state_units, pages;
    for (uint32_t unit = 0; unit < memory.arena.n_units; unit++)
        (unit < memory.state_units ? state_units : pages).push_back(unit);
    uint8_t* arena = (uint8_t*)memory.arena_buffer.contents;

    // the head's keys or values at positions 0 .. count - 1, as [kv_heads, count, head_dim]
    auto head_keys_values = [&](int key_or_value, int count) {
        std::vector<uint16_t> out((size_t)key_value_heads * count * head_dim);
        for (int head = 0; head < key_value_heads; head++)
            for (int token = 0; token < count; token++)
                memcpy(&out[((size_t)head * count + token) * head_dim],
                       arena + pages[token / memory.page_tokens] * memory.arena.unit_bytes +
                           memory.kv_offset(mtp_index, key_or_value, head, token % memory.page_tokens),
                       head_dim * mllm::ELEM);
        return out;
    };
    // cos >= 0.999 and max |diff| <= max_abs (bf16 K/V after k_norm and rope: a few bf16 steps)
    auto check_keys_values = [&](const std::string& what, const std::string& n, int count, double max_abs) {
        for (int key_or_value = 0; key_or_value < 2; key_or_value++) {
            const std::string name = "p" + n + (key_or_value ? "/v" : "/k");
            const mllm::Tensor& want = reference.at(name);
            const std::vector<uint16_t> got = head_keys_values(key_or_value, count);
            const bool exact = !memcmp(got.data(), reference.data(want), want.nbytes);
            assert(golden::matches(what + " " + name, want, reference.data(want), got.data(), 0.999, max_abs));
            printf("  %s %s: within tolerance%s\n", what.c_str(), name.c_str(), exact ? " (bit-exact)" : "");
        }
    };

    size_t drafts_compared = 0, drafts_equal = 0;
    double worst_logit = 0;
    for (int prompt = 0; prompt < 3; prompt++) {
        const std::string n = std::to_string(prompt);
        const mllm::Tensor &seq_tensor = reference.at("p" + n + "/seq"), &hidden_tensor = reference.at("p" + n + "/hidden");
        std::vector<int> seq(seq_tensor.numel());
        memcpy(seq.data(), reference.data(seq_tensor), seq_tensor.nbytes);
        const int S = (int)seq.size(), P = golden::load<int32_t>(reference.data(reference.at("p" + n + "/prompt")), 0);
        assert(S - 1 <= chunk && hidden_tensor.shape[0] == S - 1);
        printf("prompt %d: %d + %d tokens\n", prompt, P, S - P);

        // 1. the engine's own prefill over positions 0 .. S - 2 (the head's last row reads seq[S - 1])
        for (uint32_t unit : state_units) memset(arena + unit * memory.arena.unit_bytes, 0, memory.arena.unit_bytes);
        prefill.submit({0, std::vector<int>(seq.begin(), seq.end() - 1), 0, pages, state_units, seq.back()});
        while (prefill.poll().empty()) prefill.wait_oldest();
        check_keys_values("prefill", n, S - 1, INFINITY);  // cos only: its own hidden states (MLX decoded one row at a time) move V by up to ~2

        // 2. the head alone over the reference's hidden states and the tokens after them, one pass
        memcpy(hidden_rows.contents, reference.data(hidden_tensor), hidden_tensor.nbytes);
        memcpy(ws.token_ids.contents, seq.data() + 1, (S - 1) * sizeof(int32_t));
        ws.request_count = 1, ws.tokens_per_request = S - 1, ws.step_table = tables.copies[0];
        tables.write(ws.step_table, {{0, S - 1, pages}});
        mllm::encode_and_wait(queue, ws, [&] { ops.apply_mtp(ws, {hidden_rows}, {ws.token_ids}); });
        check_keys_values("head alone", n, S - 1, 0.25);

        // 3. at state c (seq[c] just accepted): the head at c - 1 from (hidden[c - 1], seq[c]), then one chained step at c
        const mllm::Tensor &drafts = reference.at("p" + n + "/drafts"), &rows = reference.at("p" + n + "/rows"),
                           &logits = reference.at("p" + n + "/logits");
        int logit_row = 0;
        for (int c = P; c < S - 1; c++) {
            tables.write(tables.copies[0], {{c - 1, 1, pages}});
            tables.write(tables.copies[1], {{c, 1, pages}});
            ((int32_t*)ws.token_ids.contents)[0] = seq[c];
            ws.request_count = 1, ws.tokens_per_request = 1, ws.step_table = tables.copies[0];
            mllm::encode_and_wait(queue, ws, [&] {
                ops.draft_tokens(ws, {hidden_rows, (size_t)(c - 1) * hidden * mllm::ELEM}, {ws.token_ids}, {tables.copies[1]});
            }, true);
            const int32_t* got = (const int32_t*)ws.next_token_ids.contents;
            const int first = golden::load<int32_t>(reference.data(drafts), 2 * (c - P)),
                      second = golden::load<int32_t>(reference.data(drafts), 2 * (c - P) + 1);
            drafts_compared++, drafts_equal += got[0] == first;
            if (second >= 0 && got[0] == first) drafts_compared++, drafts_equal += got[1] == second;
            if (logit_row < (int)rows.numel() && golden::load<int32_t>(reference.data(rows), logit_row) == c) {
                const size_t row_bytes = (size_t)model.vocab_size * mllm::ELEM;
                const mllm::Tensor row{"BF16", {model.vocab_size}, 0, 0, row_bytes};
                const uint8_t* want = reference.data(logits) + logit_row * row_bytes;
                assert(golden::matches("p" + n + " step 1 scores at " + std::to_string(c), row, want,
                                       ws.next_token_scores.contents, 0.999, 1.0));
                for (int j = 0; j < model.vocab_size; j++)
                    worst_logit = std::fmax(worst_logit, std::fabs(golden::to_double("BF16", want, j) -
                                                                   golden::to_double("BF16", (const uint8_t*)ws.next_token_scores.contents, j)));
                logit_row++;
            }
        }
        assert(logit_row == (int)rows.numel());
    }
    const double agreement = (double)drafts_equal / drafts_compared;
    printf("drafts (k = 2) equal accept.py's on %zu of %zu (%.2f%%); step 1 scores max |diff| %.3f at %d rows\n",
           drafts_equal, drafts_compared, 100 * agreement, worst_logit, 3 * 8);
    assert(agreement >= 0.99);
    printf("PASS\n");
}
