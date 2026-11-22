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
