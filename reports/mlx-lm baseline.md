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
