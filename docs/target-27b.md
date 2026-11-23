# 27B target checklist

The next milestone after the POC ([cpp-port.md](cpp-port.md), closed). The design pays off on a model the size people actually run locally: at 27B, prefill is ~30× slower per token than on 0.8B, and the weights take a quarter of RAM. That is where the prefix cache, ANE prefill and the memory design either work or don't. The target and its open questions are in [architecture.md](architecture.md) ("First model", "Open questions"). This list is the work. Every item ends in a **Check**. The comparison with other engines comes after this, at 27B: [prior-art.md](prior-art.md).

Expected shape (*estimates* from the 0.8B numbers, replaced by measurements as items close):

| | 0.8B (measured) | 27B 4-bit (*estimate*) |
|---|---|---|
| GPU weights | 0.6 GB | ~15 GB |
| Decode, B = 1 | 335 tok/s (dispatch-bound) | ~13–15 tok/s (bandwidth: ~15 GB read per token) |
| GPU prefill | ~5,000 tok/s | ~150 tok/s (compute: ~54 GFLOP per token) |
| 8k-token prompt, cold | 1.6 s | ~55 s |
| Unit (32 tokens × 16 full layers × 4 KV heads × 256) | — | 2 MiB; 64 KiB per token |

## 0. Model and reference
- [ ] Pick the checkpoint: Qwen3.6-27B, mlx-community 4-bit text-only. Read its `config.json` (`layer_types`, linear-attention head counts, dims) and download it to `/Volumes/T5/huggingface`. If no 4-bit conversion exists, convert with `mlx_lm.convert`. **Check:** `spec::load` reads it and every field matches mlx-lm's `ModelArgs`.
- [ ] mlx-lm reference at 27B: `mlx_lm.benchmark` at -p 512 / 2048 / 8192 and B = 1 / 4, plus peak memory. **Check:** a 27B section in [reports/mlx-lm baseline.md](../reports/mlx-lm%20baseline.md).
- [ ] Golden data: `py/dump_golden.py` on the 27B. If the per-layer dumps are too big, keep a few layers (first, a full-attention one, a linear one, last), plus logits and greedy tokens. **Check:** reruns are byte-identical; the size is recorded.

## 1. Load
- [ ] Weights load from the T5 (USB). The header-padded copy (`aligned()` in memory.mm) costs a second ~15 GB file if mlx-community's headers aren't 16-byte aligned. **Check:** cold load time (first start) and warm (pages cached), recorded.
- [ ] Merged projections stored once (architecture.md Deferred, "before the 27B target"): write the merged tensors into the padded weights cache and drop the originals. **Check:** RSS after load is within 1 GB of the weights' size, not ~2×.

## 2. GPU path, correct
- [ ] Kernels at 27B shapes: head dim 256 in decode and prefill attention, the bigger `qmv` / `qmm` matrices, the linear layers' head counts. **Check:** per-layer cosine against golden as in cpp-port §4; greedy 64 tokens equal mlx-lm for the 3 prompts.
- [ ] Workspace and prefill scratch at max_ctx: `scores` and `kc` / `vc` grow with heads × max_ctx. **Check:** total workspace bytes printed at startup; nothing allocates after `freeze()`; max_ctx 32k fits next to the weights and a 12 GiB arena.
- [ ] Arena layout at 27B: unit 2 MiB, `state_units` from the 48 linear layers. **Check:** `test_memory`'s layout check passes against `memory.py` for the 27B config.

## 3. GPU path, fast
- [ ] Decode at B = 1 / 4 / 8. **Check:** B = 1 ≥ mlx-lm and within ~15% of the bandwidth bound (weights bytes / measured 220 GB/s). Record what B = 4 costs per row: batching should be close to free.
- [ ] Prefill at 512 / 2048 / 8192, and chunk size (512 vs 1024; >= 2048 needs steel attention, not ported). **Check:** ≥ mlx-lm; tok/s against the compute bound.

## 4. Memory, under real conditions
- [ ] Budget: GPU weights + arena (12 GiB) + workspace, with typical apps open (browser, editor). **Check:** `footprint` / wired numbers recorded; swap growth during a 30-minute session of mixed requests; no decode step > 1 s.
- [ ] The memory-design "To verify" item: does `mlock` on the mmapped weights keep them in RAM under pressure? **Check:** the pressure test from `tools/residency_check.swift` at 27B size, with and without `mlock`.
- [ ] Long context: one request at 32k and one at 128k (8 GiB of K/V at 128k). **Check:** both admit and decode; decode tok/s at 32k and 128k against the 512-token number.

## 5. ANE prefill at 27B (architecture.md "Open questions")
- [ ] Weight format: LUT4 built from the **bf16 original** weights (LUT4 from the 4-bit MLX weights failed on 0.8B, cos 0.93). **Check:** per-block cos against fp32 reference; whole-model greedy tokens against GPU prefill; total weights (GPU 4-bit + ANE) next to a 12 GiB arena and ~16 GB for macOS fit in 64 GB.
- [ ] Build and compile the 64 dense blocks. **Check:** build time and `.mlmodelc` size, recorded; all blocks load.
- [ ] Speed alone and next to decode. **Check:** ANE prefill tok/s against GPU prefill at 27B. Then decides the default `--prefill` for 27B: ANE only if it beats the GPU, or keeps decode faster under load.

## 6. Prefix cache at 27B
- [ ] Snapshot copies: the linear state is ~150 MiB per request (*estimate*), copied at up to two points per request. **Check:** copy time measured; it stays a small fraction of the prefill it saves.
- [ ] Multi-turn: a 10-turn conversation with a long system prompt. **Check:** TTFT of every turn. Turn 10 should be within ~2× of turn 1, against a cold re-prefill of the whole history.

## 7. Close out
- [ ] 27B numbers in the reports; architecture.md's open questions closed with what was measured; the estimates table above replaced by measurements.
