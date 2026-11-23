# mlx-lm baseline: Qwen3.5-0.8B-4bit on M4 Pro

Measured 2026-11-22. These are the numbers our engine has to beat on the same model (the only family `py/mllm` runs).
M4 Pro, 64 GB, macOS 27.0, MLX 0.32.4, mlx-lm `1bf8da0` (editable install from `/Volumes/T5/Codes/mlx-lm`).
The machine was otherwise idle (an editor running). Greedy decoding, random-token prompts.

## 1. Offline: `mlx_lm.benchmark` (no server)

```
HF_HOME=/Volumes/T5/huggingface .venv/bin/python -m mlx_lm benchmark --model mlx-community/Qwen3.5-0.8B-4bit -p P -g G -b B -n 3
```
Each row is the average of 3 trials. Trials stayed within 3% of each other, except one decode trial at 8k (-6%).

| prompt | gen | batch | prefill tok/s | decode tok/s (all rows) | peak mem GB |
|---:|---:|---:|---:|---:|---:|
| 512 | 128 | 1 | 5033 | 313 | 1.17 |
| 2048 | 128 | 1 | 5083 | 305 | 1.79 |
| 8192 | 128 | 1 | 4298 | 267 | 2.06 |
| 512 | 256 | 1 | 4816 | 307 | 1.17 |
| 512 | 256 | 4 | 4338 | 780 | 1.91 |
| 512 | 256 | 8 | 4239 | 898 | 2.39 |
| 512 | 256 | 16 | 4225 | 1064 | 2.63 |

### C++ engine (this repo)

```
HF_HOME=/Volumes/T5/huggingface build/mllm bench mlx-community/Qwen3.5-0.8B-4bit -p P -g G -b B -n 3
```
Measured 2026-11-23 on the commit that adds this table (branch `port/g5-engine`), same fields and averaging as `mlx_lm.benchmark` (random
prompt tokens, EOS ignored, greedy). Prefill chunk 512, decode up to 16 rows, one decode step in flight ahead of the
CPU with greedy tokens fed on the GPU. The machine was **not** idle: a java process ran at ~400% CPU throughout.
Trials stayed within 3%; across today's runs of this code decode moved 295–299 at B=1, 921–961 at B=8 and 1109–1115
at B=16. Peak memory counts weights, workspaces and the arena units in use, not the
arena's unused rest, so it isn't comparable to MLX's (which includes its buffer cache).

| prompt | gen | batch | prefill tok/s | decode tok/s (all rows) | peak mem GB | ours / mlx-lm (prefill, decode) | what limits it |
|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 128 | 1 | 5333 | 299 | 1.04 | 1.06, 0.95 | Decode: GPU time per step is 3.23 ms (310 tok/s), ~400 dispatches of which ~0.5 ms is the small-dispatch floor; the measured 3.35 ms per token leaves ~0.12 ms between command buffers. Prefill: one 512 chunk, one command buffer. |
| 2048 | 128 | 1 | 5272 | 288 | 1.06 | 1.04, 0.94 | Decode loses 11 tok/s to the longer K/V read (mlx-lm 8); the gap is the B=1 step overhead above. Prefill: 4 chunks of 512 (mlx-lm prefills 2048 at once). |
| 8192 | 128 | 1 | 4471 | 260 | 1.13 | 1.04, 0.97 | Attention at long context (same sdpa kernels as MLX) in both prefill (16 chunks over a growing prefix) and decode. |
| 512 | 256 | 1 | 5272 | 325–336 (poc 300–303) | 1.04 | 1.09, 0.97 | As the first row. |
| 512 | 256 | 4 | 4708 | 851 (poc 791) | 1.13 | 1.09, 1.00 | Decode: same MLX kernels, the per-step overhead of B=1 spread over 4 rows. Prefill: one request's chunk at a time, decode of the earlier requests running alongside on the other queue. |
| 512 | 256 | 8 | 4445 | 1143–1182 (poc 931–961) | 1.25 | 1.05, 1.27 | Decode (g12): `qmm_t_decode` (MLX's steel qmm with an 8-row M tile) instead of `qmv_wide`, 50–58% of the bandwidth bound on the big projections; the lm_head is now the largest matmul (870 µs a step). Prefill as B=4. |
| 512 | 256 | 16 | 4145 | 1416–1480 (poc ~1088–1123) | 1.48 | 0.98, 1.33 | Decode (g12): `qmm_t_decode` with a 16-row M tile instead of `qmm_t_splitk`'s 32-row one (half its MMA work was padding). Prefill: the 16 chunks share the GPU with decode of the requests already prefilled (up to 15 rows by the last chunk), and prompt_tps runs until the last request's first token; mlx-lm prefills every prompt before decoding. |

Targets: decode B=1 ≥ 300 tok/s **met** since g9: +10–12% over poc, interleaved (QA: 306 → 337–339; after QA's
split-k race fix, 325–336 vs 300–303). Decode B=16 ≥ 1,064 met (1079–1108; 1416–1480 since g12, +30–32% over poc), GPU prefill at 512 ≥ 5,000 met
(5272–5339). The decode column of the B = 8 / 16 rows is g12's (below); of the other 512 / 256 rows, g9's after the fix, from 3 rounds interleaved with poc
(the parentheses), except B=4 (one earlier round; M = 4 doesn't touch the fix). Their other columns (prefill, ratios, peak memory) are still poc's.
g9's merged weight copies add 0.2 GB of Metal allocations (bench peak 1.04 → 1.24 at B=1) and ~0.4 GB RSS (QA, `/usr/bin/time -l`: max RSS 1.19 → 1.59 GB), because the originals stay paged in. Deferred until the weights cache stores them once.

g9 (dispatch floor), each item interleaved A/B against the previous one, 3 rounds; B=1 / B=8 per decode step:

| item | dispatches | bench-ops GPU ms | bench decode tok/s | kept |
|---|---|---|---|---|
| poc | 484 / 484 | 3.35–3.41 / 8.11–8.15 | 282–293 / 928–951 | — |
| a: count dispatches (`GpuOps::dispatches`) | 484 / 484 | — | — | yes |
| b: merged projections at M = 1 (in_proj ×4, q/k/v, gate/up; weights concatenated at load, +0.4 GB RSS) | 394 / 484 | 3.16–3.17 / 8.09–8.17 | 314–318 / 927–943 | yes |
| c: concurrent decode encoder, buffer barrier before each dependent dispatch (split-k always waits: shared `ws.splitk`, a race QA found at M ≥ 13) | 394 / 484 | ≈ b / −3% | +0.2% / +2.2% vs b; after the fix, vs a serial build: B=8 +3–5%, B=16 −6% / +1% / +1% | yes |
| d1: conv1d + q/k rms_norm + scale fused at S = 1 (`conv_norm_qk`) | 358 / 448 | −1% / flat | +2.3% / flat vs c | yes |
| d2: gated rms_norm + precise_swiglu fused (`rms_swiglu`) | 340 / 430 | op 2.7 → 1.75 µs; step flat | −0.3% / −1.5% vs d1 | reverted |

Not started (each estimated below 1% of the B=1 step from bench-ops): rope q + rope k + write_kv (~2 µs × 2 × 6
layers), argmax across threadgroups (28 µs → ~3 µs; 0.8% at B=1, 0.4% at B=8). Merging at M > 1 would need
consumers that take a row stride, because qmv_wide / qmm split-k choose by N.

Where B=1's time goes (2026-11-23, traced `mllm bench -p 512 -g 256`, 1,020 steps; machine loaded, java ~275% CPU,
so 278–283 tok/s that day): each decode command buffer is committed ~3.3 ms before the previous one ends, and the
GPU sits idle **~1 µs** between steps (p90 1 µs; same at B=8). The step time is all GPU execution (3.51–3.59 ms
traced, 3.60 ms in `bench-ops` the same hour). So CPU encode / launch cost is fully hidden. A recorded and replayed
step (an indirect command buffer, Metal's CUDA-graph equivalent) has nothing to remove. The gap is inside the
GPU: ~400 dependent dispatches in one serial encoder, each draining before the next starts. Next candidates:
concurrent dispatch with explicit barriers (independent projections overlap; same kernels, so bit-exact), then
fusing small ops.
After g9: 358 dispatches per B=1 step (from 484) and 3.04–3.16 ms GPU (from 3.22–3.53 in the same session). Most
of the gain came from the merged projections: in_proj_b / in_proj_a / k_proj / v_proj each cost a ~2.6–3.5 µs
dispatch floor for under 0.2 µs of bytes. What's left over the bound is mostly attention's small dispatches
(full: 5 at ~26 µs vs a 5 µs bound) and argmax (28 µs).

#### Batched decode kernels (g12)

Profile of the decode matmuls at M = 8–16 (standalone harness, weights rotated from DRAM, median of 5, quiet machine;
no GPU counters reachable from the CLI, so by scaling in M, executed FLOPs and ablations):

| kernel (MLX unless noted) | in_proj_qkv M 8 / 16, µs | down_proj M 8 / 16, µs | limiter |
|---|---|---|---|
| `qmv_wide` (picked at M 2–12/14) | 45.0 / 86.1 | 27.4 / 51.1 | scalar issue: linear in M, ~2.2 TFLOPS, a bf16 x load + convert + FMA per MAC |
| `qmm_t_splitk` (M ≥ 13/15) | 72.1 / 73.3 | 45.1 / 46.0 | MMA compute on padding: BM = 32, 2 × 2 simdgroups, ~5.5 TFLOPS executed, 2–4× the useful work |
| g10 `qmv_batch` (not on poc) | 79.2 / 82.5 | 54.8 / 55.4 | per-simdgroup latency, register-capped (extra accumulators spill) |
| **g12 `qmm_t_decode`** | **30.2 / 44.0** | **20.9 / 27.9** | — |
| bound (4.5-bit weights / 210 GB/s) | 16.9 | 9.8 | |

`qmm_t_decode` (`src/kernels/quantized.metal`) is MLX's steel qmm_t with one M tile of 8 (M ≤ 8) or 16 rows, a
1 × 4 simdgroup grid, BK = 64, and split-k by MLX's heuristic capped at 8; split 1 (the lm_head, 248320 × 1024)
stores bf16 directly: 1654 → 870 µs at M = 8. It runs for decode at 7 ≤ M ≤ 16 and N ≥ 1024 (`GpuOps::linear`);
below M = 7, at N = 512 and above M = 16 MLX's kernels are as fast. Not bit-exact with mlx-lm (the correctness
rule allows it): `test_batch` B = 8 KL mean 7.9e-4 / max 8.6e-3, B = 16 7.9e-4 / 1.6e-2, top-1 within 1 ulp 100%.

`mllm bench -p 512 -g 256 -n 3`, decode tok/s, g12 6b3876e against poc a0c21ec, interleaved, 2 rounds with the order
alternating (2026-11-23, idle machine):

| batch | g12 | poc | |
|---:|---:|---:|---:|
| 4 | 845 / 828 | 844 / 830 | flat (M = 4 keeps `qmv_wide`) |
| 8 | 1143 / 1145 | 932 / 931 | +23% |
| 16 | 1416 / 1416 | 1088 / 982 | +30% (the 982 is a poc outlier) |

QA's interleaved run of the same build: B = 8 1182 / 1155 vs poc 961 / 936 (+23%), B = 16 1437 / 1480 vs
1095 / 1123 (+31–32%). So B = 16 is +30–32% over poc (~1088–1123), not more.

g9's merged projections stay M = 1 only: the split-k accumulate and the consumers (z / b / a slices, swiglu over
gate | up) would all need a row stride, not a small diff.

#### ANE prefill (C++, `--prefill ane`, fp16 blocks at S = 512)

```
build/mllm bench mlx-community/Qwen3.5-0.8B-4bit -p P -g G -b B -n 3 [--prefill ane] [--stream 512]
```
Same day and load (java ~400% CPU), interleaved GPU/ANE runs, 2 rounds. `--stream 512`: one more 512-token request
(max_tokens 1) prefills throughout, a new one as each finishes; `stream_tps` is its prefill rate.

| run | prefill tok/s | decode tok/s | stream prefill tok/s |
|---|---:|---:|---:|
| P=512 G=128 B=1, GPU prefill | 5221–5338 | 297–298 | |
| P=512 G=128 B=1, ANE prefill | 5667–5688 | 295–298 | |
| P=2048 G=128 B=1, GPU prefill | 4963–5341 | 290 | |
| P=2048 G=128 B=1, ANE prefill | 5558–5585 | 289 | |
| P=512 G=512 B=8, no stream (GPU / ANE prefill) | 4139–4511 / 2348 | 851–921 / 840 | |
| P=512 G=512 B=8, stream on the GPU | 3930–4022 | 263–268 | 3190–3250 |
| P=512 G=512 B=8, stream on the ANE | 2356 | 810 | 1136 |
| P=512 G=2048 B=8, no stream, GPU prefill (pure decode) | 4480 | 882 (QA: 877) | |
| P=512 G=2048 B=8, no stream, ANE prefill | 2252 | 805 (QA: 856) | |
| P=512 G=2048 B=8, stream on the ANE | 2279 | 800 (QA: 816) | 1204 |

- One 512 chunk alone: 24 `predict`s 56–59 ms + GPU attn/conversions 17 ms, ~86–93 ms wall (~5,700 tok/s).
- With decode running, an ANE chunk takes 170–225 ms: the lane waits for the GPU after every layer (24 round trips),
  and each of its small command buffers queues behind decode steps. So B=8's prompt_tps (prefills overlapping the
  earlier requests' decode) is lower on the ANE (2,350) than on the GPU (4,100–4,500).
- Overlap: against pure decode (GPU prefill, long G=2048 run) a concurrent ANE prefill stream costs B=8 decode
  7–9% (882 → 800; QA's run 877 → 816), at the edge of the spike's −2…−12%; a GPU prefill stream costs it ~70%
  (→ 265). Don't read the "no stream, ANE prefill" rows as the baseline: its decode is depressed by staggered
  starts (the slower ANE prefill under decode load leaves fewer rows decoding in the tail), not by the ANE. The ANE
  stream's own rate (1,136–1,204 tok/s) is limited by the per-layer GPU round trips; pipelining two chunks (ANE dense
  of one while the GPU runs attn of the other) is the next step.

**Since then (g11): two chunks in flight, tails on the GPU.** The table above is from before g11; the measured
comparison is under "g11 measured" below and §2's "ANE prefill after g11".
- The ANE lane's thread keeps two chunks in flight (any requests, a request's next chunk too): while the GPU runs
  `attn(i)` of one, the ANE runs `dense` of the other. It waits only for the command buffer its next `predict` needs,
  not after every layer. K/V and state are bit-exact with one chunk at a time.
- A chunk under 256 tokens (a tail, or a prefix-cache split) runs on the GPU path (`ws.S = n`, no padding to 512).
- Noisy timings (the GPU was shared; lane only, no decode running, best of 3–5):
  - two 512 chunks back to back: 180–222 ms one at a time (the old schedule), 118–125 ms with two in flight;
  - crossover: GPU prefill 47.5 ms at n = 256, 58.5 at 320, 83.8 at 448, 95.6 at 512, against one ANE chunk at
    85.9 ms alone or 56.8 ms per chunk with two in flight. So the GPU wins below ~310–460 tokens. The threshold
    stays at 256: below both, and it leaves more of the GPU to decode.

**g11 measured** (2026-11-23, `g11-bench.sh`: g11 468f1bb against poc 5839ac4, interleaved, 3 rounds with the
order alternating; java ~70% CPU and a VM ~50% throughout, nothing else on the GPU). Per round, `mllm bench` averages
(3 trials each), shown as round 1 / 2 / 3:

| run, `--prefill ane` unless noted | prefill tok/s g11 | prefill tok/s poc | decode tok/s g11 | decode tok/s poc | stream g11 | stream poc |
|---|---|---|---|---|---|---|
| P=512 G=128 B=1 | 5666 / 4739 / 5304 | 5678 / 4761 / 5491 | 335 / 298 / 315 | 336 / 285 / 311 | | |
| P=2048 G=128 B=1 | **9002 / 8036 / 8524** | 5555 / 5053 / 5526 | 323 / 307 / 304 | 325 / 302 / 315 | | |
| P=600 G=128 B=1 (512 + an 87-token tail) | **5475 / 5153 / 5353** | 3686 / 3471 / 3597 | 333 / 316 / 324 | 333 / 317 / 318 | | |
| P=512 G=512 B=8 | **4426 / 3986 / 4086** | 2505 / 2355 / 2408 | 959 / 881 / 831 | 873 / 816 / 829 | | |
| P=512 G=512 B=8, ANE stream | **4357 / 3887 / 4082** | 2506 / 2406 / 2405 | 847 / 830 / 826 | 785 / 783 / 784 | 1207 / 1175 / 1175 | 1128 / 1122 / 1118 |
| P=512 G=512 B=8, GPU prefill + GPU stream | 3607 / 3347 / 3734 | 3663 / 3474 / 3525 | 254 / 262 / 262 | 272 / 261 / 267 | 3076 / 3146 / 3170 | 3281 / 3146 / 3207 |
| P=512 G=2048 B=8, GPU prefill (pure decode) | 4330 / 4265 / 4200 | 4225 / 4101 / 4246 | 906 / 903 / 888 | 927 / 905 / 917 | | |
| P=512 G=2048 B=8, ANE stream | **3180 / 3549 / 3508** | 2219 / 2266 / 2219 | 852 / 848 / 857 | 835 / 850 / 844 | 1084 / 1206 / 1218 | 1245 / 1260 / 1216 |

- Two chunks in flight lift ANE prefill from ~5,500 to ~8,000–9,000 tok/s when there are 2+ chunks (P=2048),
  and from ~2,400 to ~4,000–4,400 tok/s under B=8 decode. The 87-token tail no longer costs a full ANE chunk
  (P=600: 3,600 → 5,300).
- Decode next to an ANE prefill stream stays where it was (−5…−8% against pure decode); `--prefill gpu` rows are
  unchanged within noise. The ANE stream's own rate is flat (1,080–1,210 vs 1,120–1,260): with one stream request
  of 512 tokens (511 prefilled: one ANE chunk, no tail), there is never a second chunk to overlap.
- Cost: peak memory +35–41 MB (the second chunk's own buffers; GPU scratch is shared).

## 2. Serving: `mlx_lm.server` under concurrent clients

```
HF_HOME=/Volumes/T5/huggingface .venv/bin/python -m mlx_lm server --model mlx-community/Qwen3.5-0.8B-4bit --port 8080   # defaults: decode-concurrency 32, prompt-concurrency 8
.venv/bin/python tools/serve_bench.py --clients 1 4 8 16 32 -p 512 -g 256 -n 3
.venv/bin/python tools/serve_bench.py --clients 1 8 -p 2048 -g 128 -n 3
```
All C clients start together, each with its own prompt (no prompt-cache hits). Each value is the median of 3 rounds.
- **ttft**: time to the first streamed token, including queueing.
- **tok/s/client**: decode speed after a client's first token.
- **total**: all completion tokens divided by the round's wall time.

| prompt | clients | ttft s (median / p90) | tok/s per client | total tok/s |
|---:|---:|---:|---:|---:|
| 512 | 1 | 0.14 / 0.14 | 207 | 185 |
| 512 | 4 | 0.48 / 0.48 | 159 | 441 |
| 512 | 8 | 0.99 / 0.99 | 94 | 536 |
| 512 | 16 | 1.96 / 1.98 | 61 | 626 |
| 512 | 32 | 3.43 / 4.02 | 38 | 777 |
| 2048 | 1 | 0.49 / 0.49 | 245 | 127 |
| 2048 | 8 | 4.01 / 4.01 | 94 | 191 |

### C++ engine (`mllm serve`)

```
build/mllm serve mlx-community/Qwen3.5-0.8B-4bit [--prefill ane]   # defaults: port 8080, max batch 32, chunk 512, 2 GiB arena
.venv/bin/python tools/serve_bench.py --clients 1 4 8 16 32 -p 512 -g 256 -n 3
.venv/bin/python tools/serve_bench.py --clients 1 8 -p 2048 -g 128 -n 3
```
Measured 2026-11-23 (branch `port/g7-serve-close`, before its QA fixes: input limits and a 100 ms idle wake-up, off the token path; QA reproduced TTFT within 1–4% after them), serve_bench unchanged. GPU prefill, ANE
prefill (fp16 blocks, S = 512) and `mlx_lm.server` were run back to back in one session, on the same port. The
machine was **not** idle (java at ~400% CPU). mlx-lm's numbers that day match the table above to within its round-to-round spread,
so the load didn't move the comparison. Each value is the median of 3 rounds. Greedy tokens are the same as mlx-lm's: the
rounds that stop early on EOS (`g tok` 129 at 4 × 512, 7 at 1 × 2048) stop at the same token count in all three servers.

| prompt | clients | ttft s (median / p90), GPU prefill | ANE prefill | mlx-lm (same session) | tok/s per client, GPU / ANE / mlx-lm | total tok/s, GPU / ANE / mlx-lm |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 0.11 / 0.11 | 0.17 / 0.17 | 0.13 / 0.13 | 298 / 294 / 227 | 266 / 248 / 205 |
| 512 | 4 | 0.37 / 0.43 | 0.67 / 0.86 | 0.48 / 0.48 | 190 / 233 / 157 | 588 / 452 / 481 |
| 512 | 8 | **0.62 / 0.94** | 1.46 / 2.48 | 1.00 / 1.00 | 110 / 218 / 94 | 683 / 526 / 548 |
| 512 | 16 | 1.21 / 2.03 | 3.02 / 5.11 | 2.02 / 2.05 | 63 / 211 / 56 | 690 / 516 / 571 |
| 512 | 32 | 2.56 / 4.60 | 6.53 / 11.34 | 3.85 / 4.46 | 42 / 204 / 38 | 857 / 556 / 738 |
| 2048 | 1 | 0.45 / 0.45 | 0.45 / 0.45 | 0.50 / 0.50 | 287 / 289 / 225 | 143 / 144 / 117 |
| 2048 | 8 | **2.31 / 3.60** | 2.75 / 4.35 | 4.45 / 4.45 | 50 / 250 / 87 | 183 / 165 / 173 |

Targets (TTFT clearly below mlx-lm's 0.99 s at 8 × 512 and 4.0 s at 8 × 2048):
- **GPU prefill: met at the median, p90 only just.** At 8 × 512 the median is 0.62 s against 0.99–1.00, the p90 0.94; at 8 × 2048 the median is 2.31 s against 4.0–4.45, the p90 3.60.
- **ANE prefill: missed at 512** (1.46 / 2.48 s at 8 clients), **median met, p90 missed at 2048** (2.75 / 4.35 s).

Why:
- **We prefill FIFO, one chunk of one request at a time; mlx-lm prefills all waiting prompts in one batch.** So in each of
  our rounds the first client gets its token after one prompt's prefill and the last after all of them, and the median
  client sits in the middle. In mlx-lm everyone waits for the whole batch, so median = p90. The last client is
  bounded by the same total prefill work in both (8 × ~536 tokens at ~4,500 tok/s ≈ 0.95 s), so our p90 can't go much
  below mlx-lm's while prefill runs on the GPU at the same speed. Chunking only spreads that wait out. The median is where we win, and it keeps winning as
  clients grow (2.56 vs 3.85 s at 32).
- **GPU prefill and decode share the GPU.** The first clients start decoding while the others prefill, and decode next
  to a GPU prefill stream loses ~70% (§1, ANE prefill). Hence the low per-client decode at 8 × 2048 (50 tok/s, mlx-lm 87):
  most of each client's 128 tokens overlap someone's prefill. At 512 the prefills end sooner and we stay ahead (110 vs
  94 at 8, 42 vs 38 at 32). With GPU prefill, total tok/s beats mlx-lm in every row (by 6% at 8 × 2048, up to 30% at 1 client).
- **The server costs us nothing at one client:** 298 tok/s served, the same as offline `mllm bench` (295–299), where
  `mlx_lm.server` loses 25–30% (227 vs 307). The handlers never touch the engine, and the engine thread runs `step()` back to back.
- **ANE prefill slows under decode load, and pads the tail chunk.** Alone a 512 chunk takes ~90 ms, but with decode
  running it takes 170–225 ms, because the lane waits for the GPU after each of the 24 layers. On top of that, the
  ANE models are compiled for S = 512, so a ~536-token prompt (serve_bench's 512 random tokens, retokenized) costs two full chunks, one of which is ~23 tokens of
  real work (even at 1 client, TTFT is 0.17 s against 0.11 on the GPU). The GPU prefills the 25-token tail in a few ms.
  So 8 × 512 is ~16 slow ANE chunks in series, about 2.5 s for the last client. The payoff is decode: the ANE leaves
  the GPU to decode, so each client gets 204–250 tok/s at every load (GPU prefill: 42–110), but total tok/s is lower
  because new clients wait longer to join the batch.
- **Ways to close it:** tails on the GPU and a pipelined ANE lane are **done in g11** (below). Still open, for p90:
  prefill several waiting prompts per GPU submission as mlx-lm does, or give the first chunk of each waiting prompt
  priority over later chunks (shortest-first scheduling instead of FIFO).

**ANE prefill after g11** (2026-11-23, `--prefill ane`, g11 468f1bb against poc 5839ac4 interleaved, 3 rounds of
serve_bench per build, alternating order; median of the 9 trials per row; java ~70% CPU and a VM ~50% throughout).
mlx-lm and GPU-prefill columns are from the table above (earlier session):

| prompt | clients | ttft s (median / p90), ANE g11 | ANE poc | GPU prefill | mlx-lm | tok/s per client, g11 / poc | total tok/s, g11 / poc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 0.10 / 0.10 | 0.17 / 0.17 | 0.11 / 0.11 | 0.13 / 0.13 | 326 / 318 | 289 / 264 |
| 512 | 4 | 0.27 / 0.39 | 0.63 / 0.79 | 0.37 / 0.43 | 0.48 / 0.48 | 222 / 254 | 628 / 480 |
| 512 | 8 | **0.56 / 1.05** | 1.38 / 2.33 | 0.62 / 0.94 | 1.00 / 1.00 | 141 / 227 | 728 / 560 |
| 512 | 16 | 1.45 / 2.74 | 2.89 / 4.90 | 1.21 / 2.03 | 2.02 / 2.05 | 102 / 231 | 737 / 549 |
| 2048 | 1 | 0.27 / 0.27 | 0.53 / 0.53 | 0.45 / 0.45 | 0.50 / 0.50 | 316 / 308 | 188 / 137 |
| 2048 | 8 | **1.82 / 2.84** | 2.70 / 4.32 | 2.31 / 3.60 | 4.45 / 4.45 | 218 / 263 | 288 / 173 |

- TTFT halves at 512 and drops ~35% at 2048 (the 8-client medians: 1.38 → 0.56 s, 2.70 → 1.82 s). The targets are
  now **met at the median** for ANE prefill too (0.56 < 0.99 at 8 × 512; 1.82 < 4.0 at 8 × 2048), with the 8 × 2048 p90
  (2.84 s) below mlx-lm's 4.45 as well. The 8 × 512 p90 (1.05 s) is still level with mlx-lm's 1.00, for the FIFO
  reason above. At 2048 ANE prefill now beats GPU prefill at both median and p90.
- Total tok/s rises 9–66% (8 × 2048: 173 → 288). Per-client decode falls (8 × 512: 227 → 141) because clients now
  start decoding sooner and side by side, so more rows share each step and more of each client's decode overlaps
  other clients' prefill (the GPU tails and attn). The total is the throughput number; per-client decode was high
  before partly because the slow ANE prefill staggered clients into a near-serial queue.

### Prefix cache (g8)

```
build/mllm serve mlx-community/Qwen3.5-0.8B-4bit --port 18008 --prefix-cache on|off   # on is the default
.venv/bin/python tools/serve_bench.py --url http://127.0.0.1:18008/v1/completions --clients 8 -p 2048 -g 128 --shared-prefix 1536 -n 3
.venv/bin/python tools/serve_bench.py --url http://127.0.0.1:18008/v1/completions --clients 1 -p 2048 -g 128 --shared-prefix 2048 -n 3
```
Measured 2026-11-23 (branch `g8-prefix-cache`), GPU prefill, off then on, back to back; java at ~400% CPU again.
`--shared-prefix N`: every prompt starts with the same N random tokens (the same for the whole run), then its own;
N = -p repeats one prompt. `cached` is the usage's `prompt_tokens_details.cached_tokens` (mean over clients).

| run | trial | ttft s (median / p90), off | on | tok/s per client, off / on | total tok/s, off / on | cached (on) |
|---|---:|---:|---:|---:|---:|---:|
| 8 × 2048, 1536 shared | 1 | 2.37 / 3.69 | 2.23 / 3.54 | 50 / 50 | 202 / 191 | 0 |
| | 2 | 2.30 / 3.60 | **0.75 / 1.16** | 51 / 93 | 206 / 462 | 1600 of 2156 |
| | 3 | 2.28 / 3.57 | **0.76 / 1.16** | 51 / 93 | 207 / 461 | 1600 |
| 1 × 2048, the same prompt again | 1–3 | 0.435–0.441 | **0.019** | 288–290 / 287–289 | 146 / 278 | 2144 of 2161 |

- **Shared prefix: TTFT 3× lower (median and p90), total 2.2×** once the prefix is cached. Each client then prefills
  ~556 tokens instead of ~2156, and decode has less prefill to share the GPU with (93 tok/s per client vs 51).
- **Trial 1 doesn't hit**, by design: the warmup left only each prompt's last full page (past the shared part), so the
  first 8 clients match the 1536 shared tokens' K/V but find no state snapshot on them. The linear layers' state can
  only be resumed where a copy was taken. They all take the branch-point copy; the first to get there inserts it, and
  every later request resumes there. Requests admitted together don't wait for each other's copy (deferred).
- **A repeated prompt costs one chunk of 17 tokens** (2144 of 2161 from the cache; the last prompt token is always fed
  by decode): TTFT 19 ms against 435 ms.
- **When nothing hits** (trial 1, which also pays the extra chunk split and a 19 MB state copy per request), TTFT
  (2.23 / 3.54 s vs 2.28–2.37 / 3.57–3.69) and per-client decode (50 vs 50–51) match the cache-off rounds. Total tok/s
  is lower, 191 vs 202–207: that round produced fewer tokens, 112 per client on average against 128 (serve_bench's
  `g tok` column, not in the table above), in a shorter wall time (4.7 vs 5.0 s), because some clients hit EOS early
  (see the next point). One round can't separate a small cost from that. `mllm bench` defaults to off (its trials
  reuse their prompts); with it, B = 1 and B = 8 at 512 / 256 measured 282–296 and 877–958 tok/s, interleaved with
  the parent commit's 284–296 and 843–870.
- Greedy tokens: equal to cache off when chunk boundaries don't move (bit-exact K/V and state in `test_engine_cache`);
  when they do (a hit resumes mid-chunk, or the last-full-page split), K/V cosine is ≥ 0.99997, state ≥ 0.9999, and
  greedy can flip at a near-tie, as it does with the cache off between chunk 64 and 512. Trial 1's early EOS comes
  from such flips.

## What it says
- **Time to first token grows linearly with the number of clients.** In each round, all prompts are prefilled before any client gets its first token. TTFT ≈ C × prompt / ~4.2k tok/s, so 8 clients with 2k prompts wait 4 s each. The design targets exactly this: prefill on the ANE, decode on the GPU, and chunked prefill so decode never waits on a whole batch.
- **Single-stream decode is already close to memory bandwidth.** The weights are 625 MB, and nearly all of them are read per token (the tied lm_head included). So 310 tok/s ≈ 190 GB/s, against the 211–238 GB/s we measured as the practical ceiling for GPU 4-bit decode. A C++ loop can gain at most about 15–25% at batch 1. Batching is where the headroom is: 16 rows reach 1064 tok/s total (3.4× one row), because they share each weight read.
- **The server costs about 25–30% of single-stream decode** (207–245 tok/s served vs 307 offline): HTTP, SSE, detokenizing and its scheduler.
- **mlx-lm's batched prefill is slower per token than batch 1** (4.2k vs 4.8–5.1k tok/s at 512).

## Caveats
- **Some requests stop early.** mlx_lm.server has no `ignore_eos`, so a few requests hit EOS before `-g` (the `g tok` column in the raw output). That lowers **total** in those rounds (e.g. one 2048-prompt, 1-client round stopped at 7 tokens and got 13 tok/s total). Medians absorb it.
- **Connection retries at 16–32 clients.** `mlx_lm.server` runs on Python `http.server`, which queues only 5 pending connections. Extra connects get reset, and the client retries. The retry time counts in TTFT; at 32 clients it adds to the p90.
- **One slow trial.** The first C=1 round of the final run was slow (134 tok/s per client). Earlier runs gave 219–229 tok/s at C=1, which matches the median.

## 27B: loads and runs correctly on the GPU (g13)

mlx-community/Qwen3.6-27B-4bit (64 layers: 16 full, 48 linear; hidden 5120, 24 / 4 heads x 256, linear Hk 16 / Hv 48 x 128, vocab 248320, untied lm_head, A_log in bf16). 2026-11-23, machine shared with other agents (no timings below are speed claims).
- Spec: `spec::load` = spec.py = mlx-lm `TextModelArgs` field by field (dump_golden `check_spec` asserts it, plus eps 1e-6, rope theta 1e7, 64 rotary dims, which gpu.mm hardcodes); `MLLM_MODEL=... test_spec` passes.
- Golden (`py/dump_golden.py`, layers 0, 2-3, 31-32, 62-63 and the final norm kept): 369 MB for the 6 files (343 MB before g13b's 3 sweep cases), plus batch_ref 0.4 MB; two reruns byte-identical (`cmp` on all 6 files). 101 s, peak RSS 18 GB. 0.8B golden unchanged (rerun byte-identical).
- Weights cache (merged chains back to back, header padded, language_model.* only; `<cache>/weights/<key>-v3/`): first start writes 15.13 GB (v2 kept the 0.92 GB vision tower: 16.05 GB in 104 s) from and to the T5; with the v3 rewrite, a 16-token generate took 130.4 s wall. Warm start + 16 tokens (`mllm generate ... --max-tokens 16`): 3.9 s wall. Cold start with the cache already written: not measured (needs a page-cache purge).
- Merged projections stored once: no copies at load; `generate` peak footprint 1.85 GB = 1 GiB arena + workspace + Metal (weights are file-backed, 16.05 GB in v2 / 15.13 GB in v3, not in footprint). Before: in_proj / qkv / gate_up copies would have been ~+9 GB at 27B (*estimate*, 0.8B ratio).
- Arena: unit 2 MiB, 74 units per state slot (148 MiB); `test_memory` layout = memory.py.
- Kernels vs golden (`MLLM_MODEL=mlx-community/Qwen3.6-27B-4bit build/tests/...`): test_dense (embed, RMSNorm incl. rms_looped at 5120, every projection, qmv_wide / qmm_t_splitk), test_linear (Hv 48 GDN, conv, state), test_lm_head (untied, M 1..16), test_stack (whole stack, decode logits), test_lanes: all bit-exact. test_engine: 3 prompts x 64 greedy tokens equal mlx-lm, solo and batched.
- Attention (g13b): test_attn trace and sweep bit-exact (o 18/18, K/V 54/54; sweep o 29/29, K/V 62/62). The q length 8 mismatch at gqa 6 was a dispatch rule, not the kernel: MLX v0.32.4 (`has_fused_kernel`) runs sdpa_vector only while q length x gqa <= 32, so at 27B (gqa 6) q lengths 6..8 take the unfused fallback (we ran sdpa_vector up to 8; 0.8B's gqa 4 gives 8 x 4 = 32, never past the limit). The fallback's q kᵀ at q length 6 needs gemv_wide nv3 (added); the sweep gained 6_30, 6_1000, 7_1100 (old cases byte-identical). Not a threads-per-threadgroup problem: 8_40 runs the 1-pass kernel, 1024 threads at any gqa.
- Batched decode at gqa 6: two 2-pass rows in one call share max N's block count (MLX's 's' rule: 64 blocks up to 1024 keys, 128 to 8192, 256 to 32k when gqa x q length > 4; 0.8B is always 64). A row at N 1024 next to one at 4201 runs 128 blocks instead of its solo 64: 1 bf16 ulp off (max |diff| 9.8e-4, max |o| 0.16). Batched decode is tolerance-checked, so test_attn checks that call within 2 ulps of max |o|; its first mixed call is now a real 1-pass + 2-pass pair (1_1022 + 1_4200), bit-exact. test_batch at 27B (new batch_ref, `py/dump_batch_ref.py`, 205 s): B = 1 byte-equal to mlx-lm; KL mean 5.3e-4 at B = 8 and 5.5e-4 at B = 16 (limit 7.5e-3; mlx-lm's own B = 1 vs B = 8 / 16: 4.8e-4 / 5.5e-4), top-1 within 1 ulp 100%.
- Workspace at max_ctx (`mllm generate mlx-community/Qwen3.6-27B-4bit ... --max-ctx 32768 --arena-gib 12`, chunk 512, max batch 16; now the defaults for 2 MiB units, `Memory::unit_of`, the 0.8B keeps 1 GiB / 16k): workspace 1156.1 MiB (1463.9 before the 2-pass partials fix: the prefill lane reserved partials for 512 decode rows it never runs, and the decode lanes too few for 16 rows at gqa 6, where rows past 8192 keys take 256 blocks: 9+ such rows would have hit full_attn's assert. Now max_batch x H x blocks(max_ctx), or one prefill row of the vector path's longest q). Of it, prefill's `scores` [chunk, H, max_ctx] bf16 is 768 MiB and `kc` / `vc` 2 x 64 MiB; both scale with max_ctx (128k: 3 GiB of scores; a blocked / steel attention removes it). Peak footprint 14.24 GB (12 GiB arena + workspace + Metal; weights file-backed, 16.05 GB): ~30.3 GB with the weights, of 64 GB. A 7.5k-token prompt (one long prefill, fallback attention past 4096 keys, looped softmax) + 8 tokens ran clean, 104.7 s wall; a 16-token short prompt 46.1 s wall (cold weights cache read from the T5); machine shared (another process at 400% CPU), not speed numbers. Nothing allocates after `freeze()`: `Memory::buffer` / `surface` assert when frozen (asserts on in this build), and they are the only Metal allocations besides the weights' no-copy buffers at load.
- 27B test status (each binary alone, `MLLM_MODEL=mlx-community/Qwen3.6-27B-4bit`; test arenas now `max(0.8B size, Memory::arena_for(spec, slots, tokens))`, so the 0.8B runs are unchanged): pass: test_arena, test_attn, test_batch, test_dense, test_engine, test_golden, test_kernels (scratch sized by hidden: it wrote 8224 x 5120 into an 8224 x 1024 buffer, SIGBUS), test_lanes, test_linear, test_lm_head, test_memory, test_memory_fuzz, test_metal, test_prefix_cache, test_server, test_spec, test_stack, test_tokenizer, test_weights. test_concurrent skips (its race harness would memcpy ~1 TB at 2 MiB units; the 0.8B covers it). test_engine_cache: chunk 512 hits, chunk 32 bit-exact hits and the small arena pass (747 s); moved chunk boundaries give K/V cos 0.99339 and state cos 0.96650, where mlx-lm's own whole-vs-split prefill of the same 1300 tokens gives 0.99338 / 0.96575 (0.8B: 0.99998 / 0.99996), so the floor off the 0.8B is 0.95; its two 80-request stress runs stopped on a 10 s stall limit tuned for 0.8B speed (now 120 s off the 0.8B), not rerun (~30 min at 27B). test_ane: no 27B ANE blocks (§5, ANE section of test_engine_cache skips for the same reason).

## 27B: speed on the GPU path (g14)

mlx-community/Qwen3.6-27B-4bit, MLX 0.32.4, mlx-lm `1bf8da0` (same as the 0.8B tables), `-n 3`, engine = g14
(fda2e42: qmm_t in 64 x 64 tiles). Two holds of the machine-wide heavy lock, mlx-lm and the engine interleaved per
shape, order swapped between holds (`/Volumes/T5/cache/mllm/g14/hold_ab.sh 1|2`, logs `hold_ab1.log`,
`hold_ab2.log` there):

```
HF_HOME=/Volumes/T5/huggingface .venv/bin/python -m mlx_lm benchmark --model mlx-community/Qwen3.6-27B-4bit -p P -g 128 -b 1 -n 3
HF_HOME=/Volumes/T5/huggingface build/mllm bench mlx-community/Qwen3.6-27B-4bit -p P -g 128 -b 1 -n 3
```
- Hold 1 (2026-11-24 01:24-01:42, mlx-lm first): a Virtualization VM at 63-72% CPU throughout; at the end imagent
  66% and familycircled 37%. Nothing else heavy (other heavy jobs wait on the same lock).
- Hold 2 (01:55-02:12, engine first): the VM at 52-61% CPU; nothing else above 10%.

| prompt | hold | mlx-lm prefill / decode tok/s, peak GB | engine prefill / decode tok/s, peak GB | ours / mlx-lm (prefill, decode) |
|---:|---:|---|---|---:|
| 512 | 1 | 122.4 / 15.51, 16.40 | 129.9 / 15.05, 16.54 | 1.06, 0.97 |
| 512 | 2 | 117.9 / 15.40, 16.40 | 129.2 / 15.01, 16.54 | 1.10, 0.97 |
| 2048 | 1 | 118.6 / 15.29, 18.13 | 134.5 / 15.14, 16.64 | 1.13, 0.99 |
| 2048 | 2 | 120.4 / 15.41, 18.13 | 133.1 / 15.02, 16.64 | 1.11, 0.97 |
| 8192 | 1 | 117.3 / 14.93, 19.13 | 132.3 / 14.74, 17.05 | 1.13, 0.99 |
| 8192 | 2 | 117.6 / 14.89, 19.13 | 132.0 / 14.74, 17.05 | 1.12, 0.99 |

All 128 generated tokens, B = 1. Engine peak memory counts weights + workspace + arena units in use (not the 12 GiB
arena's unused rest). An 8192-token prefill takes 61.9-62.1 s in the engine (8192 / prompt tok/s; the bench times
the prefill alone, weights warm, no prefix hit) against 69.7-69.8 s in mlx-lm.

- **Prefill ≥ mlx-lm: met**, 1.06-1.13x in all six pairs (p512's trials spread 114-138 tok/s in both engines,
  falling trial by trial; p2048 trials within 4%, p8192 within 0.2%).
- **Decode at B = 1 is not ≥ mlx-lm:** 0.97-0.99x in all six pairs (15.0-15.1 vs 15.3-15.5 tok/s at 512 / 2048,
  14.74 vs 14.89-14.93 at 8192). The gap is ~1-2 ms per token of a ~66 ms step, outside the GPU step
  (bench-ops below).
- The earlier one-run table (2026-11-23, main 2be5a93 before qmm_t_64, java at 165-330% CPU and Go tests running)
  read 100-108 tok/s prefill and 12.5-14.2 decode for both engines: a busy machine cut both by 10-20%. Its B > 1
  rows are the only B = 4 / 8 figures for the bench (one run each, same busy machine, main 2be5a93, not rerun):

| prompt | gen | batch | mlx-lm prefill / decode tok/s, peak GB | engine prefill / decode tok/s, peak GB | ours / mlx-lm (prefill, decode) |
|---:|---:|---:|---|---|---:|
| 512 | 256 | 4 | 101.7 / 35.11, 18.61 | 101.7 / 32.71, 17.17 | 1.00, 0.93 |
| 512 | 256 | 8 | 98.4 / 37.06, 21.61 | 106.6 / 56.84, 17.99 | 1.08, 1.53 |

**Bounds.** Decode reads all weights but the embedding per token: 15.13 - 0.72 = 14.4 GB. At the 220 GB/s of the
checklist that is 65.5 ms (15.3 tok/s); mlx-lm's 15.5 is already past it (223 GB/s), and qmv_fast itself runs at
240-250 GB/s on the big projections (bench-ops), so the floor of this kernel is ~59 ms (*estimate*). The engine's
15.0-15.1 tok/s is ~98% of the 220 GB/s figure. Prefill: 48.7 GFLOP per token in matmuls (24.4 B params) against
MLX's own bf16 GEMM, measured 7.48 / 7.49 / 7.58 TFLOPS at 512 / 2048 x 5120 x 17408 and 4096³ (MLX's 4-bit
qmm: 6.76-6.81; `tools/gemm_peak.py`, `hold1.log`, VM at ~55% and spotlight at ~85% CPU): 6.5 ms per token,
**~154 tok/s compute bound**. The engine reaches 129-134 (84-87%), mlx-lm 117-122 (76-80%). `tools/roofline.py`'s
FLOPS knob (bench-ops, test_kernels) is now 7.5e12 from that run.

**Change: qmm_t in 64 x 64 tiles** (`qmm_t_64`, quantized.metal; MLX fixes 32 x 32 for affine). Same template,
same BK 32 and 2 x 2 simdgroups, so every output accumulates K in the same order through the same float MMA:
bit-exact. test_kernels compares the 64 x 64 and 32 x 32 outputs byte for byte (M 100 / 200, N % 64 = 0 / 32),
since no golden test reaches a full 64-row tile (golden chunks are 32 / 24 / 25 tokens). Interleaved main vs g14
(QA, one hold, VM at 54-73% CPU, `qa/hold.log`, 2 rounds, `-g 64` / `-g 16`): p512 prefill 125.0 / 117.3 ->
132.5 / 131.4 tok/s, p2048 119.2 / 124.3 -> 135.1 / 136.7, decode unchanged (15.1-15.4 both); 0.8B p2048 prefill
5274-5316 -> 5415-5430 (3 rounds, +2-3%). Prefill chunk GPU time (512 tokens, traced, 2 rounds x 4 chunks,
`-p 1024 -g 2`, `tiles.log`, busy machine): 32 x 32 3.93-4.54 s, 64 x 64 3.71-3.98 s.
Tile rows: M ≤ 32 keeps MLX's 32 x 32 (a 64-row tile of a ≤ 32-row tail is mostly padding); at M 40 / 48 / 56
64 x 64 is ~1% faster summed (`hold1.log`, 2 rounds, best-of-3 per op: -2..-3% on q_proj, gate_proj and
in_proj_qkv, +2% on down_proj), so 64 x 64 starts at 33 rows (`GpuOps::tile64_rows`).
Tried and rejected (same rounds as the chunk trace, `tiles.log`): BK 64 (+3% vs BK 32 at 64 x 64), 128 x 64
(+5%); M-tiles-first grid order (no gain at 32 x 32 or 64 x 64, numbers not logged).

**Decode at B = 1 / 4 / 8 / 16** (bench-ops whole step on fda2e42, 512 keys per row, median of 10 steps;
`hold_ops.sh`, `hold_ops.log`, 02:21, one hold, the VM at 62-70% and duetexpertd ~50% CPU):

| B | dispatches | GPU ms | wall ms | tok/s (wall) | GPU ms per row |
|---:|---:|---:|---:|---:|---:|
| 1 | 948 | 64.38 | 65.19 | 15.3 | 64.4 |
| 4 | 1188 | 91.45 | 92.25 | 43.4 | 22.9 |
| 8 | 1396 | 115.88 | 117.01 | 68.4 | 14.5 |
| 16 | 1492 | 181.14 | 182.17 | 87.8 | 11.3 |

The B = 1 GPU step (64.4 ms) matches mlx-lm's whole per-token time at p512 (64.5-64.9 ms), while the bench's
engine decode is 66.4-66.6 ms per token (wall step 65.2): the 1-3% decode gap is ~1-2 ms outside the GPU step
(host time between steps or the bench's per-token work; not split). Inside it, the dispatch floor (948 dispatches,
~2-4 ms, *estimate*) is the lever to get under mlx-lm. B = 4 costs 1.42x the B = 1 step: the big projections
run at 164-177 GB/s there (qmv_wide; q_proj 204 us vs 139 at M = 1) against 248-255 GB/s at B = 1.
Earlier runs on the busy machine (2026-11-23, java at 330% CPU, `ops_steps.log`) gave B = 4 104-118 ms, so the
1.5-1.7x ratio quoted before was load. Tried: g12's `qmm_t_decode` from M = 2 instead of 7 (experiment binary,
`qmin.log`, busy machine): B = 4 136 ms vs 104-118; it takes ~254 us on q_proj at any M <= 8 (~137 GB/s), so it is
bandwidth-latency-bound, not MMA-bound. Not landed. B = 8 runs `qmm_t_decode` (g12) and is 1.5x mlx-lm in the
bench (one busy run, table above).

Left (one line each in docs/target-27b.md): B = 1 decode 1-3% behind mlx-lm (~1-2 ms outside the GPU step; the
dispatch floor, 948 dispatches, is ~2-4 ms of the step, *estimate* from sum of ops vs whole step); B = 4 per-row
cost (1.42x the B = 1 step; matmuls at 164-177 GB/s vs 248-255 at B = 1; qmm_t_decode needs more bytes in
flight: split-k > 1 for large N, or register double-buffering of W); chunk 1024 vs 512 (not measured; 1024 doubles prefill's `scores` scratch to 1.5 GiB at
max_ctx 32k, and the matmuls are already at full occupancy at M = 512, so *estimate* < 3%).

## 27B: long context, memory budget, prefix cache (g15)

Measured 2026-11-24 on the g15 branch, each run alone under the shared heavy lock. Long context, snapshot copy,
multi-turn and test_engine_cache's `cache` part ran before the rebase onto g14 (main 2be5a93: prefill ~8% slower than
now); the stress runs, the mixed session and the overlap runs on g14 (a0479ef). The machine was not idle: a VM at 55-75% CPU (16 GB) throughout, docker idle, at times a qemu
VM (46%) and Spotlight; logs in `/Volumes/T5/cache/mllm/g15/`. GPU prefill, chunk 512, 12 GiB arena.

**Long context** (`tools/long_ctx.mm`: one request of P random tokens, then 128 greedy tokens, EOS ignored, cache off):
```
build/long_ctx mlx-community/Qwen3.6-27B-4bit 32768 512 32640       # max_ctx 32768: workspace 1156.1 MiB
build/long_ctx mlx-community/Qwen3.6-27B-4bit 131072 130944         # max_ctx 131072: workspace 4182.7 MiB
```
| prompt | cold prefill + first token | prefill tok/s, last 4k | decode tok/s (127 steps) | ms / step | longest step | footprint |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4.0 s (130 tok/s) | | 15.54 | 64.4 | 66 ms | 14.14 GB |
| 32,640 | 307 s (106 tok/s) | 101 | 12.95 | 77.2 | 81 ms | 14.24 GB |
| 130,944 | 1689 s (77.5 tok/s) | 56.5 | 8.49 | 117.8 | 124 ms | 17.37 GB |

- **Both admit and decode.** 128k: 4096 pages + 74 state units of the arena's 6144; the cold prefill took 28 min,
  inside one 30-min hold with ~50 s to spare (it won't fit in a hold on a busier machine). Swap unchanged (1583 MB used
  before and after).
- **Decode time per step grows linearly with the K/V read**: K/V is 64 KiB per token, so 2.15 GB at 32k and 8.59 GB at
  128k. Over the 512 step that adds 12.8 ms (168 GB/s) and 53.4 ms (161 GB/s); the weights stream at 235 GB/s
  (15.13 GB / 64.4 ms). Nothing else grows: it is the full layers' K/V reads, by sdpa_vector_2pass (MLX's kernel, ported
  bit-exact) at ~70% of the weights' rate. Against a bound with K/V at the weights' rate (101 ms at 128k, 9.9 tok/s),
  decode is at 86%. Not compared with mlx-lm at these lengths.
- **Prefill slows with position**, 8.2 ms per token at the start to 17.7 ms at 127k: MLX's unfused fallback attention
  (prefix chunks) runs q kᵀ and p v over every key and writes [512, 24, keys] bf16 scores (3 GiB per layer at 128k) that
  the mask and softmax read and write again. At max_ctx 131072 that buffer makes the workspace 4.2 GiB (1.2 at 32k).
  A blocked (flash-style) attention removes both, but MLX v0.32.4 runs this fallback for prefix chunks, so it would cost
  bit-exactness with mlx-lm.

**Memory budget, mixed session** (`tools/serve_session.py mixed`: 4 clients, each sending a request then pausing
0-20 s, prompts 64-8192 tokens (a third start with one shared 2048-token prefix), 16-256 tokens out; 24 min, 85
requests; `mllm serve` defaults: 12 GiB arena, max_ctx 32768, max batch 32, prefix cache on). Open meanwhile: a
Virtualization VM (16 GB, 43-75% CPU), docker, iTerm2, Finder, several Claude Code sessions; no browser.
- Server footprint 12-13 GB through the session (`footprint`: 13 GB IOAccelerator = arena + workspace 1276 MiB +
  Metal; peak 13 GB); weights 15.13 GB file-backed on top, in the residency set. Wired: 3.4-3.6 GB with the server
  down, 29.5-31.6 GiB (1.93-2.07 M pages of 16 KiB) while it ran: arena, workspace and the weights are wired.
- **Swap: no growth** (1582.9 MB used before, 1574.9 MB after; swapouts unchanged at 908016, swapins +171, pageouts
  +151). Memory free 84% before (the VM's 16 GB included).
- **Decode steps > 1 s: yes, broken.** The longest gap between two streamed tokens was 4979 ms; 44 of 85 requests had
  one over 1 s (median of per-request longest 1068 ms). Cause (`overlap.mm`, A decodes alone at 64 ms per token while B
  prefills 2048 tokens): the GPU runs the two lanes' command buffers in commit order, so a decode step committed after
  a 512-token chunk (one command buffer, ~3.9 s at 27B) waits out the whole chunk: 11 tokens in B's 15.2-15.9 s, gaps
  up to 3.75-3.92 s. Splitting the chunk into command buffers of K layers but committing them together did not help
  (K = 16 / 8 / 4: max 3.66-3.94 s; K = 1: 2.16-2.19 s, and prefill 3% slower). Fix (g15, `PrefillLane::submit`): the
  chunk is encoded as command buffers of K layers, each committed from the completion handler of the one before, so a
  decode step waits for one K-layer buffer at most. Bit-exact (same dispatches in the same order).
  `tools/overlap.mm`, `build/overlap mlx-community/Qwen3.6-27B-4bit 2048`, interleaved in one hold, two reps per run
  (K = 4, 2 and 64 run twice, K = 8 once). Every row, the K = 8 one included, is from an experimental build with an env
  knob for K, not the committed code (K fixed at 8, knob removed); the mixed rerun below is on the committed build:

  | K layers per buffer | B's prefill, tok/s | A's tokens during it | A's gap mean / max, ms |
  |---|---:|---:|---:|
  | 64, committed whole (before) | 128.9-133.6 | 11 | 1393-1444 / 3791-3889 |
  | 8 (**now**) | 106.2-108.3 | 67 | 282-288 / 539-545 |
  | 4 | 82.9-89.4 | 131 | 175-189 / 300-332 |
  | 2 | 62.7-65.3 | 259 | 121-126 / 188-195 |

  The GPU is time-shared, not overlapped: B's extra time equals A's extra steps x 64 ms. K = 8 keeps every decode step
  under ~0.55 s while giving up the least prefill.
- **Rerun with the fix** (QA, committed build 3355c08, same session for 23 min, not 30; logs
  `/Volumes/T5/cache/mllm/g15/qa/h2*`, `mixed*.log`): 82 requests; **longest gap 814 ms, 0 requests over 1 s** (47
  over 500 ms; median of per-request longest 553 ms). Footprint 12-13 GB; wired 29.2-31.5 GiB while serving, 3.4 GB
  idle; swap 1526.9 → 1350.9 MB used, no growth. The cost: TTFT mean 35.8 → 49.8 s and max 100.5 → 146.0 s against
  the first session, prefill now sharing the GPU with decode (and partly a busier machine).

**Prefix cache: snapshot copy** (`long_ctx`'s first lines: `Engine::copy_units`' memcpy of one state, 74 units of
2 MiB, on idle arena units): 148 MiB in 2.4-2.8 ms (56-66 GB/s, 6 samples in two runs); the memset that zeroes a cold
request's state 0.6 ms. A request makes at most two copies (~5 ms) and one at admit on a hit; one 512-token prefill
chunk takes ~4 s at 27B, so a copy costs ~0.1% of the chunk it saves, and the multi-turn hits below each saved ~4.2k
tokens (34 s).

**test_engine_cache at 27B** (`MLLM_MODEL=mlx-community/Qwen3.6-27B-4bit build/tests/test_engine_cache [part]`): passes.
The whole test is ~28 min of GPU here, too close to a 30-min hold, so it takes a part argument (`cache`, `stress512`,
`stress32`; none = all, as the gate runs it). `cache`: 652 s pre-g14, passed again on the final build (00a1a77, 662 s; hits, bit-exact chunk-32 hits, moved
boundaries K/V cos 0.99339 / state 0.96650 as g13, small arena 20 / 20 equal cache off; ANE part skips, no 27B blocks).
`stress512`: 508 s, 80 requests, 9 hits; `stress32`: 494 s, 4 hits (rebased on g14, but before the prefill
command-buffer chain); invariants held every step, no stall past 120 s, every unit free or cache-held at the end. QA
reran `stress512` on the final build (3355c08, with the chain): PASS, 80 requests, 9 hits, 559 s on a busy machine
(`qa/h4*`). `stress32` was not rerun on the final build.

**Multi-turn** (`tools/serve_session.py turns`: a ~4.2k-token system prompt, then 10 user turns; each prompt is the
previous prompt + its reply (up to 64 tokens) + the next question):
```
build/mllm serve mlx-community/Qwen3.6-27B-4bit --port 18015 [--prefix-cache off]
.venv/bin/python tools/serve_session.py turns --system 4000 --turns 10 -g 64 --save turn10.txt
.venv/bin/python tools/serve_session.py turns --replay turn10.txt          # on the --prefix-cache off server
```
| turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | turn 10, cache off |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| prompt tokens | 4249 | 4339 | 4396 | 4453 | 4510 | 4566 | 4622 | 4679 | 4736 | 4794 | 4794 |
| cached | 0 | 4224 | 4320 | 4384 | 4448 | 4480 | 4544 | 4608 | 4672 | 4704 | 0 |
| TTFT s | 33.86 | 1.18 | 0.94 | 0.78 | 0.74 | 0.97 | 0.97 | 0.84 | 0.73 | 0.99 | 38.70 |

- Every turn after the first resumes at the previous prompt's last full page (the snapshot taken there) and prefills
  only the reply, the new question and the < 32-token page tail: 62-115 tokens, 0.7-1.2 s. **Turn 10 is 0.99 s against
  38.7 s for a cold re-prefill of the same 4794 tokens (39x), and within 2x of every hit turn.** Turn 1 is the cold
  system prompt (33.9 s).
- A new conversation on the same system prompt hit 4224 tokens (turn 1's last full page lay inside the shared part):
  TTFT 0.42 and 0.38 s.

## Raw output (final run)
```
p=512 g=256
  C trial  ttft s   p90 s  tok/s/client  total tok/s  p tok  g tok
  1     1   0.142   0.142         133.9        125.1    538    256
  1     2   0.150   0.150         206.9        185.1    541    256
  1     3   0.133   0.133         223.0        200.6    531    256
  4     1   0.482   0.482         138.5        440.7    537    256
  4     2   0.491   0.491         177.7        267.3    534    129
  4     3   0.483   0.483         159.1        490.5    538    256
  8     1   0.949   0.950          93.7        557.6    536    256
  8     2   0.995   0.996          90.3        535.9    534    256
  8     3   0.993   0.994         103.3        518.6    534    224
 16     1   1.970   1.994          59.7        654.7    538    256
 16     2   1.958   1.983          61.1        582.8    537    224
 16     3   1.940   1.963          61.0        625.9    538    240
 32     1   3.241   3.798          43.3        796.4    536    240
 32     2   3.426   4.019          37.8        763.2    537    232
 32     3   3.427   4.022          37.0        777.4    537    240
p=2048 g=128
  1     1   0.490   0.490         244.6        126.8   2140    128
  1     2   0.489   0.489         244.8        127.0   2144    128
  1     3   0.486   0.486         239.1         13.7   2138      7
  8     1   3.951   3.951          92.9        192.5   2148    128
  8     2   4.022   4.022          95.8        167.9   2147    112
  8     3   4.011   4.012          94.3        191.1   2143    128
```
