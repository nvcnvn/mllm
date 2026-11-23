# 27B target checklist

The next milestone after the POC ([cpp-port.md](cpp-port.md), closed). The design pays off on a model the size people actually run locally: at 27B, prefill is ~30× slower per token than on 0.8B, and the weights take a quarter of RAM. That is where the prefix cache, ANE prefill and the memory design either work or don't. The target and its open questions are in [architecture.md](architecture.md) ("First model", "Open questions"). This list is the work. Every item ends in a **Check**. The comparison with other engines comes after this, at 27B: [prior-art.md](prior-art.md).

Expected shape (*estimates* from the 0.8B numbers, replaced by measurements as items close):

| | 0.8B (measured) | 27B 4-bit (*estimate*) |
|---|---|---|
| GPU weights | 0.42 GB measured (language model only) | 15.13 GB measured (weights cache v3: language model only; the 0.92 GB vision tower is left out) |
| Decode, B = 1 | 335 tok/s (dispatch-bound) | 14.7-15.1 tok/s measured at 512-8192 keys (mlx-lm 14.9-15.5, 0.97-0.99x; two interleaved holds); 96-98% of 14.4 GB / 220 GB/s |
| GPU prefill | ~5,000 tok/s | 129-134 tok/s measured at 512-8192 (mlx-lm 117-122, 1.06-1.13x; two interleaved holds); bound ~154 (48.7 GFLOP / 7.5 TFLOPS) |
| 8k-token prompt, cold | 1.6 s | ~62 s *derived* (8192 / 132 tok/s from the bench's timed prefill, weights warm; TTFT not timed) |
| Unit (32 tokens × 16 full layers × 4 KV heads × 256) | — | 2 MiB; 64 KiB per token |

## 0. Model and reference

## 1. Load (g13: done, see the report)

## 2. GPU path, correct (g13, g15: done, see the report)

## 3. GPU path, fast
- [ ] Decode at B = 1 ≥ mlx-lm: 0.97-0.99x in two interleaved holds (g14; see the report). **Check:** ≥ 1.00 in an interleaved hold. The GPU step (64.4 ms, bench-ops) already equals mlx-lm's per-token time; ~1-2 ms is outside it, and the dispatch floor (948 dispatches, ~2-4 ms, *estimate*) is the lever.
- [ ] Decode at B = 4 per-row cost: a B = 4 step costs 1.42x the B = 1 step (91.5 vs 64.4 ms GPU, bench-ops; qmv_wide at 164-177 GB/s; qmm_t_decode at M = 4 is slower). **Check:** B = 4 ≥ mlx-lm (35.1; engine 32.7, one run on a busy machine) and closer to free.
- [ ] Prefill chunk 512 vs 1024 (>= 2048 needs steel attention, not ported; *estimate* < 3%). **Check:** interleaved A/B recorded; chunk 1024 the 27B default if it wins.

## 4. Memory, under real conditions (g15: budget and long context done, see the report)
- [ ] The memory-design "To verify" item: does `mlock` on the mmapped weights keep them in RAM under pressure? **Check:** the pressure test from `tools/residency_check.swift` at 27B size, with and without `mlock`. *Waits for the user: this Mac has crashed under memory pressure, and other sessions run on it; needs an idle machine (each pressure test grew swap by 7-10 GiB).*

## 5. ANE prefill at 27B (architecture.md "Open questions")
- [ ] Weight format: LUT4 built from the **bf16 original** weights (LUT4 from the 4-bit MLX weights failed on 0.8B, cos 0.93). **Check:** per-block cos against fp32 reference; whole-model greedy tokens against GPU prefill; total weights (GPU 4-bit + ANE) next to a 12 GiB arena and ~16 GB for macOS fit in 64 GB.
- [ ] Build and compile the 64 dense blocks. **Check:** build time and `.mlmodelc` size, recorded; all blocks load.
- [ ] Speed alone and next to decode. **Check:** ANE prefill tok/s against GPU prefill at 27B. Then decides the default `--prefill` for 27B: ANE only if it beats the GPU, or keeps decode faster under load.

## 6. Prefix cache at 27B (g15: done, see the report)

## 7. Close out
- [ ] 27B numbers in the reports; architecture.md's open questions closed with what was measured; the estimates table above replaced by measurements.
