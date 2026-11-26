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
- [ ] Decode at B = 4 per-row cost (g18: tried, not kept): a qmv_fast over 2..6 rows (one weight pass, 8 values per thread) took the B = 4 GPU step 92.8 → 89.5 ms (bench-ops; matmuls 163-177 → 171-183 GB/s), but test_engine_cache's small-arena check (batched outputs equal cache-off solo runs, 20 / 20) got 19 / 20, so it was reverted; that test's exact-equality assumption has to change first (batched decode is checked by tolerance). On a quiet machine mlx-lm gives 38.7-39.0 tok/s at B = 4 and the engine (before qmv_decode) 35.6-37.7: the bench's staggered starts (each prompt prefills alone, so the batch ramps up and tails off at B < 4) cost more than the kernel gains. **Check:** B = 4 ≥ mlx-lm in bench, interleaved.
- [ ] Prefill chunk 512 vs 1024 (>= 2048 needs steel attention, not ported; *estimate* < 3%). **Check:** interleaved A/B recorded; chunk 1024 the 27B default if it wins.

## 4. Memory, under real conditions (g15: budget and long context done, see the report)
- [ ] The memory-design "To verify" item: does `mlock` on the mmapped weights keep them in RAM under pressure? **Check:** the pressure test from `tools/residency_check.swift` at 27B size, with and without `mlock`. *Waits for the user: this Mac has crashed under memory pressure, and other sessions run on it; needs an idle machine (each pressure test grew swap by 7-10 GiB).* Notes below.

### Memory-pressure test: notes (2026-11-24, not run)
**Why it matters at 27B.**
- **Weights:** the 15.13 GB of weights are file-backed (mmapped, no copy since g13), so macOS can evict them like any file cache.
- **Residency sets aren't a pin.** The earlier residency test showed that an `MTLResidencySet` doesn't pin memory: under pressure, file-backed pages fell from 100% to 60% resident and anonymous pages to 0%.
- **The cost of eviction:** an idle server would lose its weights to a browser, then pay for re-reading them from the T5 (USB) on the next request. The first warm-up after a cold start took 46 s at 27B (g13b).
- **What the 27B holds:** wired memory while serving was 29.5-31.6 GiB (12 GiB arena + workspace + Metal) against `vm.user_wire_limit` = 52.5 GB, so `mlock` of the 15 GB of weights fits under the limit on paper.

**What `tools/residency_check.swift` does today.**
- It builds two files of N GiB each (default 4) on the weights volume. File A sits in a residency set; file B is the control.
- It adds 1 GiB of incompressible anonymous memory at a time.
- It reports the fraction of each still resident (`mincore`) and the GPU read speed afterwards.
- It stops at the first of: B < 10% resident, pressure level critical, swap growth > 2 GiB, or 52 GiB allocated. GPU reads under pressure page memory back in, so swap has grown to 7-10 GiB anyway.
- It has no `mlock` mode yet: one more mode is needed, where A is `mlock`ed instead of (or as well as) being in the residency set.

**How to run it safely.**
- **Idle machine only:** the user present, no other sessions running heavy jobs, and the shared lock free (`lsof /Users/nvcnvn/.cache/rock-solid/heavy.lock`). Message the storage-engine and mobile-app sessions first, and take the lock (`/Volumes/T5/cache/heavy pressure ...`) so nothing starts during the run.
- **Close the big apps first:** VMs (a 16 GB Virtualization VM and a qemu VM ran all night) and Docker. Note the starting swap (`sysctl vm.swapusage`) and the internal disk's free space; swap lands on the internal SSD.
- **Start at 27B-like sizes in steps:** a 4 GiB file first, then 16 GiB (the weights' size), not the whole sweep at once. Watch `memory_pressure` from a second terminal. If the system goes yellow or red for more than a few seconds, kill the test's PID; it frees everything on exit.
- **Run each mode once:** file / residency set (the baseline, already known), file / `mlock`, and anon.

**What to record.**
- **Residency:** the fraction of A still resident, with and without `mlock`, at the moment B falls under 10%.
- **Speed:** GPU read speed of A afterwards.
- **Swap:** growth during the run.
- **Duration:** how long the machine stayed slow.

**Decides.** If `mlock` holds, `mllm serve` should `mlock` the weights at load (behind a flag, default on for serve). If it doesn't, the fallback is anonymous memory plus re-reading on demand, or accepting a cold first request after pressure.

**Also untested:** what the server does when macOS evicts its weights mid-session. Measure TTFT and decode-step times after a pressure event.

## 5. ANE prefill at 27B (architecture.md "Open questions")
- [x] **One weight copy** (2026-11-25): the ANE keeps a private wired copy of baked-in weights, so LUT4 for it means
  ~15 GB next to the GPU's 15 GB. Instead the GPU dequantizes its 4-bit weights a layer at a time into fp16 scratch the
  ANE takes as a matmul *input*, with the same values as the GPU's (no LUT4 precision problem either).
  - `tools/ane_dynamic_weights.py` (Python) showed a ~5.6 GB/s weight intake. That was mostly Python's numpy ->
    MLMultiArray conversion, not Core ML.
  - `tools/ane_weight_input.swift`, IOSurface-backed CVPixelBuffer inputs and output (as `src/ane.mm` already does):
    CPU time per call stays 0.08-0.15 ms whatever W's size (no CPU copy), and results follow W rewritten between calls.
    The extra time over baked W is ANE-side, 20-44 GB/s (S=1 baked-weight streaming was 39 GB/s, so this is the
    ANE's own memory path). Matmul speed with W as input vs baked, IOSurface, M4 Pro:

    | S x Cin x Cout | baked | input |
    |---|---|---|
    | 512 x 5120 x 5120 | 5.38 TFLOPS | 3.53 |
    | 512 x 5120 x 1024 | 4.56 | 3.16 |
    | 512 x 1024 x 4096 | 11.25 | 7.47 |
    | 2048 x 1024 x 4096 | 14.44 | 11.65 |

  - At 27B GPU prefill is ~6.7 TFLOPS (~61 ms per layer per 512 tokens). With fine weight splits (Cin <= 1024) the
    ANE with input weights is about even at 512-token chunks and ~1.7x at 2048, before per-call overheads of a real
    layer, plus the GPU's dequant (~0.8 GB fp16 written per layer, ~4 ms) and ~1.7 GB double-buffered scratch.
    Route open. Its case is still mainly concurrency (decode keeps its speed during prefill), not single-user speed.
- [x] **One 27B layer with weight inputs** (2026-11-25, `tools/ane_layer.py` builds, `tools/ane_time.swift` times
  with IOSurface I/O; random weights, Qwen3.6-27B shapes). Dense block = o/out_proj + FFN of layer i-1 and the
  projections of layer i, row-major [S, C], ~730 MiB fp16 of weight inputs. Splits: one input per linear sliced in
  the graph into Cin <= 1024 x Cout <= 4096 pieces (47 ms at S=512); unsplit 200 ms; the pieces as 133 separate
  inputs 95 ms (37 ms of CPU per call). All matmuls on the ANE. One compiled model per layer kind and S serves all
  layers (weights are inputs), loading in 4-14 s.

  | S | linear-attn block | full-attn block | dense only, x 64 layers | vs GPU prefill (131 tok/s) |
  |---|---|---|---|---|
  | 512 | 47.4 ms (8.3 TFLOPS) | 47.4 ms | 169 tok/s | 1.3x |
  | 1024 | 82.5 ms (9.5) | 82.8 ms | 194 tok/s | 1.5x |
  | 2048 | 153 ms (10.3) | 154 ms | 209 tok/s | 1.6x |

  - GPU side per layer and chunk: dequant 4-bit -> fp16 scratch 5.4 ms (MLX, 175 GB/s), plus attention / DeltaNet.
    Both well under the ANE's time, so with two chunks in flight prefill is ANE-bound (estimate, not measured).
  - Decode alongside (27B, B=1): 14.9 tok/s alone, 12.6 with the S=2048 block looping on the ANE (-16%; the dequant
    would add ~3.5% GPU time). GPU prefill alongside costs decode ~70%.
  - Scratch: two layers' weights, ~1.5 GB. S=2048 also needs steel attention on the GPU (engine.mm rejects chunk >= 2048).
- [x] **Precision** (2026-11-25, `tools/ane_fp16_check.py`, emulated on the GPU): the dense blocks in fp16 (residual
  stream included), attention / DeltaNet in bf16; prefill 2048 tokens, then the bf16 GPU path scores the next 256 from
  that cache and state. Against an fp32-dense reference, fp16 prefill is as close as the bf16 GPU path itself:

  | text | KL mean bf16 / fp16 | top-1 agree bf16 / fp16 | NLL fp32 / bf16 / fp16 |
  |---|---|---|---|
  | prose (docs) | 1.23e-3 / 1.43e-3 | 99.6% / 98.8% | 2.7878 / 2.7892 / 2.7912 |
  | code | 1.10e-3 / 1.13e-3 | 98.0% / 98.4% | 0.8653 / 0.8611 / 0.8646 |

  Largest |value| in the fp16 dense: residual 868, down_proj out 502, FFN act 354 (fp16 max 65504). Not covered:
  the ANE's own accumulation (MLX fp16 matmuls accumulate in fp32). **Check** in the engine work: a real layer's
  ANE output against this emulation.
- [x] **Engine** (2026-11-25): `--prefill ane` (AneOps, src/ane.mm; the baked-weight modes were removed; blocks from
  `ane.py build`, 4 kinds x S 512 / 1024, 32 s to build at 27B). Chunks of 1024; a shorter one runs the 512
  block; two chunks in flight, each with its own weight surfaces (~0.9 GB a set at 27B). Tests: test_ane (0.8B:
  K/V/state cos vs GPU 0.9952-0.9987, like the baked fp16 blocks; two in flight bit-exact; greedy = golden up to the
  first exact tie). 27B, `mllm bench`, M4 Pro:

  | | GPU prefill | shared ANE prefill |
  |---|---|---|
  | prompt 1024 tok/s | 136 | 157 (1.15x: one chunk, ANE and GPU take turns per layer) |
  | prompt 4096 tok/s | 131 | 191 (1.46x: chunks overlap) |
  | decode alone tok/s | 15.0 | 15.1-15.3 |
  | decode with a 2048-token prefill alongside | 3.3 (prefill 79 tok/s) | 12.1 (prefill 95 tok/s) |
  | peak memory | 16.6-16.8 GB | 19.7-19.9 GB |

- [ ] **Next:** a single chunk takes turns (ANE dense, then GPU attn + the next layer's dequant): dequant layer i + 1
  while the ANE runs layer i (a third weight set) for the one-chunk case. Tune `gpu_below` for 27B (256, from the
  0.8B). Startup time of the 8 models not measured. Chunks stay 1024 (user, 2026-11-25: no 2048 / steel attention).
- LUT4 as the *GPU's* format too (one format, not one copy: the ANE still compiles its own): ruled out 2026-11-25, ~10%
  less to read per decode step isn't worth the precision and losing bit-exactness with mlx-lm.
- [ ] Weight format: LUT4 built from the **bf16 original** weights (LUT4 from the 4-bit MLX weights failed on 0.8B, cos 0.93). **Check:** per-block cos against fp32 reference; whole-model greedy tokens against GPU prefill; total weights (GPU 4-bit + ANE) next to a 12 GiB arena and ~16 GB for macOS fit in 64 GB.
- [ ] Build and compile the 64 dense blocks. **Check:** build time and `.mlmodelc` size, recorded; all blocks load.
- [ ] Speed alone and next to decode. **Check:** ANE prefill tok/s against GPU prefill at 27B. Then decides the default `--prefill` for 27B: ANE only if it beats the GPU, or keeps decode faster under load.

## 6. Prefix cache at 27B (g15: done, see the report)

## 7. Lower-bit weights (g16, in progress, not on main)
The idea: decode at B = 1 is bandwidth-bound (64.4 ms per step reading 14.4 GB), so 3-bit (~11.5 GB, *estimate*) or 2-bit (~8 GB, *estimate*) weights should decode proportionally faster and leave room for longer context, if the quality holds.

**Done so far**, on branch `g16` in worktree `/Volumes/T5/Codes/mllm-g16`: WIP commits bdbbb66 and e65a6f3 on top of 7836048, not merged.
- Per-tensor bit widths of 3, 4 and 6, read from the packed shapes; 2 and 8 bits are a small list change.
- Mixed recipes run their unmerged parts (e.g. `v_proj` at 6 bits next to 3-bit q/k) one by one into the same output row.
- `mlx_lm.convert` tensor names are renamed in the weights cache.
- **Tests:** all pass on a 0.8B mixed_3_6 model (B = 1 bit-exact against new golden data). The one exception is test_engine's "batched equals solo" check, which fails because the model is badly damaged: mlx-lm's own B = 8 vs B = 1 on it shows the same spread.
- **Not run on that branch:** the full gate and any 27B test.

**0.8B dry run**, which says little about 27B:

| | KL vs bf16 | top-1 | perplexity |
|---|---|---|---|
| 4-bit | 0.021 | 90.6% | 31 |
| 3-bit | 2.27 | 33% | 233 |
| mixed_3_6 | 2.01 | 38% | 186 |

The reference's own perplexity is 30.4.

**Not done:**
- [ ] 27B conversions from the bf16 original (Qwen/Qwen3.6-27B, 55.6 GB, in the HF cache): 8-bit (quality reference), 3-bit, mixed_3_6, 2-bit, mixed_2_6.
  - The GPU convert hit "GPU Timeout Error" reading cold bf16 shards from the T5. The CPU convert is untested at 27B: the 3-bit run was stopped after ~1 min, still reading.
  - The scripts (convert with a 30 GiB watchdog, quality, benchmark, table) are in `tools/g16/` (`chain_all.sh` runs all of step 1; logs go to /Volumes/T5/cache/mllm/g16/logs).
  - **Check:** per variant, KL / top-1 / perplexity against the 8-bit reference on ~4k tokens of real text, greedy 64-token agreement on the 3 golden prompts, and mlx-lm decode tok/s and peak memory, run twice. A variant passes if KL ≤ ~2× the 4-bit's and top-1 ≥ 95%.
- [ ] For the variants that pass:
  - 27B golden data;
  - the 27B tests with `MLLM_MODEL` set to the variant;
  - the gate;
  - engine vs mlx-lm decode at B = 1/4 and prefill at 512/2048, interleaved.
  - If none passes, the next step is calibrated quantization (DWQ/AWQ-style).

## 8. Other gaps found on the way (2026-11-23/24)
- **Prefill vs decode trade-off (g15, g18: closed):** the 8-layer chain costs a lone prefill nothing (4096 tokens: 128.1-128.2 tok/s chained, 127.7-129.2 as one buffer), so the mixed session's TTFT rise is the GPU time decode takes, not the chain; splitting only while a request decodes was dropped. The ANE lane's short all-GPU chunks now use the same chain: a 200-token prompt next to a decode held a step 1.86-2.06 s, now 0.32 s (see the report, g18).
- **Workspace at 128k (g18: done):** 4183 → 1879 MiB (32k: 1156 → 580). The unfused fallback runs one KV head group at a time, so `scores` is 4× smaller; bit-exact.
- **Long-context decode against mlx-lm:** not compared. The engine gives 12.95 tok/s at 32k and 8.49 at 128k, at 86% of a bound that reads K/V as fast as the weights.
- **Unmeasured:**
  - Cold load with an empty page cache (needs `sudo purge`). The weights-cache rewrite (~2 min) and the warm start (~4 s) are measured.
  - A full 30-minute mixed session: QA ran 23 min to fit a lock hold.
  - test_engine_cache `stress32` at 27B on the final g15 build: `cache` and `stress512` passed.
- **Untested error path:** the prefill chain's failure handling (`PrefillLane::poll`). Nothing can make a GPU command buffer fail in tests; a deliberately faulting kernel would.
- **Tests that don't run at 27B:**
  - test_concurrent: skips at 2 MiB units, about 400 GB of memcpy at 27B.
  - test_ane: waits on §5.
- **Batched speed:** engine vs mlx-lm at B = 4/8 hasn't been rerun on a quiet machine. Only bench-ops' GPU step times exist: 91.5 / 115.9 ms.
- **Speculative decoding (g21: measured, not built):** see [speculative.md](speculative.md). The 27B's own MTP head (4-bit, from the bf16 checkpoint) at k = 2 gives 2.56 tokens per verify on six prompts. Verify of 3 rows costs 72.8 ms against 64.9 for one (bench-ops). *Estimate:* ~30 tok/s greedy at B = 1, 2.0x.
- **Old weight caches:** they're deleted. Only `-v3` dirs remain under `/Volumes/T5/cache/mllm/weights/`.

## 9. Close out
- [ ] 27B numbers in the reports; architecture.md's open questions closed with what was measured; the estimates table above replaced by measurements.
