# C++ port checklist

This is the whole `py/` → `src/` port. Interfaces, names and the engine loop come from [architecture.md](architecture.md); memory layout from [memory-design.md](memory-design.md). Every item ends in a **Check**. An item is done only when its check passes.

## Decisions
- **No Python bindings.** `py/` is the reference, not a host. `py/dump_golden.py` writes the reference data once: token ids, every workspace field after each `dense(i)` and `attn(i)`, K/V pages, linear-attention state, logits and greedy tokens. The C++ tests compare against those files. That gives the same layer-by-layer verification bindings would, without nanobind or MLX interop code.
- **Raw Metal, not libmlx.** The design needs fixed buffers, one untracked arena buffer, one command buffer per step, and ANE IOSurfaces read with no copy. libmlx owns its own allocation and command buffers, so it conflicts with all four. Kernels are vendored (MIT): MLX's `quantized.metal`, `rms_norm.metal`, `rope.metal`, `sdpa_vector.h`, and mlx-lm's gated-delta kernel. Starting from the same kernels also gives us a fair chance at bit-exact tokens.
- **Languages:** Objective-C++ (`.mm`) for files that touch Metal or Core ML; C++20 everywhere else. Build with CMake, which produces `libmllm`, the `mllm` CLI, the test executables, and `mllm.metallib` (via `xcrun metal`). Tests are plain executables with `assert`, no test framework.
- **Tokenizer:** our own byte-level BPE. The pre-tokenizer regex uses Unicode classes (`\p{L}\p{M}\p{N}`), so we vendor llama.cpp's `unicode.cpp` / `unicode-data.cpp` (MIT). The chat template is hard-coded ChatML for `qwen3_5`, text only. *ponytail: Jinja (llama.cpp `common/jinja`) arrives with the second model family.*
- **Core ML models are still built by Python** (`ane.py`, coremltools). C++ only loads the compiled `.mlmodelc` from the same cache path (`_model_key`).
- Same component, file and method names as `py/mllm` (architecture rule 5).

## 0. Scaffolding and reference data
- [ ] `CMakeLists.txt`, `src/`, `src/kernels/`, `tests/`; vendored code under `src/vendor/` with its licenses. **Check:** `cmake --build` produces the CLI, the tests and the metallib on a clean clone.
- [ ] `py/dump_golden.py MODEL OUT`. Writes safetensors to `/Volumes/T5/cache/mllm/golden/<model_key>/`, which is not in git. Contents:
  - Spec fields.
  - Token ids for a string set: ASCII, CJK, emoji, digits, whitespace runs, special tokens, and the 3 chat prompts from `test_engine.py`.
  - For one prompt with 2 prefill chunks then 3 decode steps of batch 2: every workspace field after each `dense(i)` / `attn(i)`, each layer's K/V pages, each linear layer's conv and recurrent state, and the logits.
  - Greedy 64 tokens for each prompt.

  **Check:** reruns are byte-identical.
- [ ] `tests/golden.h`: loads golden tensors and compares them, exact for integers, cosine plus max-abs for floats. **Check:** comparing a golden file with itself passes; comparing it with a perturbed copy fails.

## 1. `spec` and weights
- [ ] `spec.h/.cpp`: `config.json` → `ModelSpec` with the same fields as `spec.py` (vendored nlohmann/json). `resolve()` finds the HF snapshot via `$HF_HOME/hub/models--org--name/refs/main` and never downloads. **Check:** every field equals the golden spec.
- [ ] Safetensors reader: `mmap` each file, parse the header, and keep a view of each tensor (dtype, shape, offset). The whole mapping is wrapped once with `newBufferWithBytesNoCopy` (16 KiB-aligned), and each tensor is a buffer plus offset. **Check:** a hash of every tensor's bytes matches what Python sees.

## 2. `tokenizer`
- [ ] BPE from `tokenizer.json`: vocab, merges, and special/added tokens split out before the BPE runs. **Check:** ids equal the golden ids for the whole string set.
- [ ] `encode_chat`: ChatML with `add_generation_prompt` and the template's default thinking behaviour. **Check:** ids equal the golden ids for the 3 chat prompts.
- [ ] `Detokenizer` `push`/`flush`: byte-level decoding that holds back incomplete UTF-8. **Check:** streaming the golden outputs one token at a time gives byte-exact text, including multibyte characters split across tokens.

## 3. `memory`
- [ ] Arena bookkeeping (CPU only), from memory-design: free bitmap, `req_refs` / `cache_refs`, `pending`, `admit` / `alloc` / `release` / `unreserve` / `retire`. **Check:** a unit test covering admit up to the limit, `alloc` never failing after `admit`, refcounted release, pending reclaim gated by step, and no starvation of the oldest waiter.
- [ ] Arena storage: one shared, untracked `MTLBuffer`, pre-touched with `memset`, in a residency set, with `requestResidency` called again on a timer. The layout must be byte-identical to `memory.py` (unit size, KV group offsets, state slot layout), so golden K/V compares directly. **Check:** for the same config, `unit_bytes`, `state_units` and `n_units` equal Python's, and the golden K/V copied in at the Python offsets reads back correctly through the C++ offsets.
- [ ] Workspace: fixed named `MTLBuffer`s, sized for the largest of chunk × batch. **Check:** nothing allocates after startup. Assert it with an allocation counter in a debug build.
- [ ] Step tables: 2 copies, chosen by step parity. **Check:** a unit test where the CPU never writes the copy the in-flight step reads.

## 4. `gpu`: kernels and `GpuOps`
Each item is checked against the golden data for its layer. The goal is matching greedy tokens (64/64). Intermediate tensors need cosine ≥ 0.99999 against the bf16 golden data.
- [ ] Quantized embedding gather (4-bit affine, group 64). **Check:** `ws.x` after `dense(0)`'s embedding.
- [ ] RMSNorm, plus RMSNorm gated by `z`. **Check:** the normalized inputs the dense ops consume.
- [ ] 4-bit affine matmul, two kernels: `qmv` for decode (B ≤ 32) and `qmm` for prefill. **Check:** each projection output; time compared with `tools/roofline.py`.
- [ ] `dense(i)`, both layer kinds:
  - Every layer: o/out projection + residual, post-attention norm, SwiGLU MLP + residual, input norm.
  - Full layers: q/gate split, `q_norm` / `k_norm`.
  - Linear layers: `in_proj_qkv` / `z` / `b` / `a`.
  - `dense(L)`: the final norm.

  **Check:** every workspace field after each `dense(i)`.
- [ ] Full-attention `attn(i)`: RoPE at the right positions, K/V written into pages through the step table, then paged attention. Decode: one query per row over its pages (`sdpa_vector`-style). Prefill: causal attention over the prefix pages plus the chunk. Then the sigmoid gate. **Check:** golden K/V pages and `ws.o`.
- [ ] Linear `attn(i)`: depthwise conv1d with the conv window updated in place, SiLU, q/k/v split, `normalize_qk`, the gated-delta update in place on the arena state, then the gated norm. **Check:** golden conv state, recurrent state and `ws.o`.
- [ ] `lm_head` (tied embedding used as a quantized linear) and sampling. Greedy argmax runs on the GPU; temperature and top-p match `make_sampler`. **Check:** logits and greedy ids; a seeded sampling test gives the same distribution shape.
- [ ] Batch decode in one dispatch per op for all B rows, with no loop over rows (this removes the Python version's ponytail). **Check:** golden decode for batch 2 matches row by row.
- [ ] `mllm bench-ops`: time per op at decode B = 1/8/16 and prefill S = 512. **Check:** listed next to the roofline. Any op below about 70% of its bound gets a note.

## 5. `lanes` and `engine` (GPU only)
- [ ] `DecodeLane`: one command buffer per step, `MTLSharedEvent` signaled with the step number, `submit` / `poll` / `can_submit`, at most one step ahead.
- [ ] `PrefillLane` on its own `MTLCommandQueue`: `submit` / `poll` / `busy`.
- [ ] Pending reclaim: a unit freed while a step may still read it returns only after the event passes that step.
- [ ] `engine.h/.cpp`, a 1:1 port of `engine.py`: `submit` / `step` / `cancel` / `generate`, FIFO admission, abort with token −1 when a lane fails, an event queue per request.
- [ ] **Check (`tests/test_engine`):** the same cases as `py/test_engine.py`. 3 prompts × 64 greedy tokens equal golden, solo, batched, and waiting on a one-request arena. Cancel and abort return every unit.
- [ ] `mllm generate MODEL PROMPT` and `mllm bench -p -g -b -n`, printing the same fields as `mlx_lm.benchmark`.
- [ ] **Check (speed):** the same rows as [reports/mlx-lm baseline.md](../reports/mlx-lm%20baseline.md), recorded next to them. Targets: decode B=1 ≥ 300 tok/s, B=16 ≥ 1,064 tok/s; GPU prefill at 512 tokens ≥ 5,000 tok/s.

## 6. `ane`
- [ ] Load the cached `.mlmodelc` for each dense block (`MLModel`, compute units CPU + NE). If a block is missing, tell the user to build it with `py/test_ane.py` / `ane.py`. **Check:** loads all 24 blocks for S = 512.
- [ ] IOSurface I/O: inputs and `outputBackings` are IOSurface-backed `MLMultiArray`s, and the GPU reads them with no copy. Apply the `tools/ane_iosurface.swift` findings: size = `bytesPerRow × rows` rounded to 16 KiB, check base alignment at startup, fall back to the texture view. **Check:** one block's output, read on the GPU, is bit-exact with the CPU copy.
- [ ] Layout and padding: channels-first `(1, C, 1, S)` ↔ the workspace, either as GPU kernels or by having `attn` read channels-first directly. Zero-pad the last chunk and slice it back. **Check:** golden K/V and state, cosine ≥ 0.999 for fp16 and ≥ 0.99 for int8 / LUT6, the same thresholds as `test_ane.py`.
- [ ] `PrefillLane` (ANE) on its own thread, because `predict` blocks. GPU `attn` stays on the prefill queue. **Check:** greedy 64 after ANE prefill equals golden.
- [ ] Overlap: ANE prefill runs while GPU decode runs. **Check:** decode tok/s with a concurrent prefill against without (earlier spike: −2…−12%), and ANE prefill tok/s against GPU prefill.

## 7. Serving and the comparison
- [ ] A minimal streaming `/v1/completions` server (vendored cpp-httplib), just enough for `tools/serve_bench.py`. Requests go through `Engine::submit`. *Load under concurrency is where the design should win, so this belongs to the port rather than being deferred.*
- [ ] **Check:** `serve_bench.py` with the same parameters as the baseline report (1–32 clients, 512 and 2048 tokens). Add ours to the report. The target is time to first token under load clearly below mlx-lm's, which was 0.99 s with 8 clients and 4.0 s with 8 clients at 2048 tokens.

## 8. Close out
- [ ] `architecture.md`: fill the C++ column with what was actually built, and close the questions the port answered.
- [ ] `py/` stays as the reference and the source of golden data; new engine features go into C++ only.

## After the port (not in this checklist)
These are triggered by the items in architecture.md's and memory-design.md's **Deferred** tables: prefix cache and SSD tier, 8-bit KV, more than one step encoded ahead, the Qwen3.6-27B target and its ANE weight format, `mlock`, a second model family with Jinja templates.
