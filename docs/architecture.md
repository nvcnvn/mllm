# Engine architecture (v1)

One structure for the throwaway Python version (`py/`) and the C++ version (`src/`). The engine loop, the components and their interfaces stay the same. Only what is inside a component changes, and we port one component at a time.

Memory layout and allocator rules live in [memory-design.md](memory-design.md). This doc is about who calls whom.

## Rules that keep the port mechanical
1. **Interfaces carry plain data**: ints, int arrays (token ids, slot numbers, positions), and named workspace buffers. Framework objects (MLX arrays, `MLModel`, `MTLBuffer`) never appear in a signature. They only live inside a component or inside a workspace field.
2. **Only `Memory` allocates.** Every buffer is sized at startup from `ModelSpec` + config. Python cheats underneath (MLX and numpy allocate), but its call sites look like the C++ ones.
3. **One engine thread owns all bookkeeping** (arena counts, requests, prefix cache). Lanes get read-only work items and return results. No locks.
4. **Lanes are asynchronous: `submit()` and `poll()`.** In Python, `submit` runs the work to completion and `poll` returns it. In C++, `submit` returns once the work is committed. The engine loop is the same code in both.
5. **One component = one Python module = one C++ `.h/.cpp` pair**, with the same name and the same method names.

## The flow
```
 generate(prompt)
      │
      ▼
 Tokenizer.encode ──► Request(ids) ──► Engine.step() ◄─────────── loop until done ─────────┐
                                          │                                               │
                         admit (Arena)    │ split prompt into chunks                       │
                                          ▼                                               │
            ┌──────────── PrefillLane ──────────────┐     ┌──────── DecodeLane ──────────┐ │
            │ one chunk of one request (C++ ANE:    │     │ one token for every request  │ │
            │ two chunks, any requests)             │     │ in DECODE state:             │ │
            │ for each layer i:                     │     │                              │ │
            │   dense(i)  ANE  (GPU fallback)       │     │   dense(i)  GPU              │ │
            │   attn(i)   GPU  write K/V to units,  │     │   attn(i)   GPU  paged       │ │
            │                  attend prefix+chunk  │     │   lm_head + sample  GPU      │ │
            │ output: K/V in the arena, no logits   │     │ output: token ids            │ │
            └───────────────────┬───────────────────┘     └──────────────┬───────────────┘ │
                                │ chunk done                             │ token ids       │
                                └──────────────► Engine ◄────────────────┘                 │
                                                   │                                       │
                                                   ▼                                       │
                                       Detokenizer.push(id) ──► text to caller ────────────┘

 Every box above reads and writes only through Memory:
 ┌─────────────────────────────────────── Memory ────────────────────────────────────────┐
 │  Weights (GPU 4-bit, ANE LUT4)   Arena (KV units)   Workspace (activations)   StepTables │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

**Prefill produces K/V only.** We prefill `prompt[:-1]`. The last prompt token is the first input to decode. So prefill never runs `lm_head` and skips the last layer's o_proj and FFN. Decode samples the first output token like any other.

## The layer seam: `dense` and `attn`
Your `attn()` / `ffn()` split, cut where the hardware cuts:
```
 x ─► [dense 0] ─► q,k,v ─► [attn 0] ─► a ─► [dense 1] ─► q,k,v ─► [attn 1] ─► a ─► … ─► [dense L] ─► h
      embed                  rope             o_proj + residual                           o_proj + residual
      norm                   K/V → units      norm, FFN, residual                         norm, FFN, residual
      q,k,v proj             attention        norm, q,k,v proj (next layer)               final norm
```
- `dense(i)` is everything with weights between two attentions. It holds no state, so it suits the ANE (static shapes, weights only).
- `attn(i)` is everything that touches the arena: RoPE (needs positions), K/V write, attention over the request's pages. It always runs on the GPU, because only the GPU reads units through step tables. It may own small per-layer weights: for Gated DeltaNet that is the depthwise conv1d, `A_log`, `dt_bias` and the gated norm. Its output goes to `ws.o`, before the out/o projection, which belongs to `dense(i+1)`.
- Putting the boundary after the next layer's q,k,v halves the ANE↔GPU handoffs: 1 Core ML call per layer, not 2. The spike measured `qkv` and `o+FFN` as separate models. Fusing them into one model is *(to verify)*.
- Layer kinds (full / window / linear) change only `attn(i)`. `dense(i)` is the same for every kind.

| Phase | `dense` | `attn` | Sampling |
|---|---|---|---|
| Prefill | ANE (`--prefill ane`; fp16 or int8 blocks, LUT4 fails on the 0.8B) or GPU (default), same interface | GPU | — |
| Decode | GPU, 4-bit affine (MLX format) | GPU, paged | GPU |

## Components
| Component | File | Owns | Interface |
|---|---|---|---|
| `ModelSpec` | `spec` | Shapes read from `config.json`: layers, kind per layer, heads, kv_heads, head_dim, hidden, vocab, RoPE, eos ids, compiled ANE chunk sizes | plain struct, `load(model_dir)` |
| `Tokenizer` | `tokenizer` | Vocab, chat template | `encode(text) → ids`, `encode_chat(messages) → ids`, `detokenizer() → Detokenizer` |
| `Detokenizer` | `tokenizer` | Bytes not yet valid UTF-8 | `push(id) → str`, `flush() → str` |
| `Memory` | `memory` | Weights, Arena, Workspace, StepTables | see below |
| `Request` | `engine` | ids, state, `pos` (tokens with K/V), slot lists, sampling params, output ids, detokenizer | plain struct |
| `GpuOps` | `gpu` | GPU kernels / MLX calls | `dense(i, ws)`, `attn(i, ws)` (step table in `ws.table`), `lm_head(ws)` (GPU: logits + argmax, encoded), `sample(ws, params) → ids` (CPU, after the command buffer completed) |
| `AneOps` | `ane` | One compiled Core ML model per dense block and chunk size | `dense(i, ws)` |
| `PrefillLane` | `lanes` | In-flight prefill chunks (GPU: 1, ANE: 2) | `submit(PrefillWork)`, `poll() → [PrefillWork]`, `busy()`, `can_submit()` |
| `DecodeLane` | `lanes` | In-flight decode step(s), step counter | `submit(DecodeWork)`, `poll() → [(req_id, token)]`, `can_submit()` |
| `PrefixCache` (C++ only) | `prefix_cache` | A page-level trie of prompt pages by chained hashes; linear-state snapshots on some nodes | `match(ids) → {L, S, pages, snapshot}` at admission, `insert(ids, n_pages, pages, snapshot)` at a snapshot point, `evict()` from `Arena::admit` |
| `Engine` | `engine` | Requests, the loop | `submit(prompt, params) → req_id`, `step() → [(req_id, text, done)]`, `generate(prompt, params) → iterator[str]` |
| `Server` (C++ only) | `server` | httplib server, the engine thread, one event queue per HTTP request | `listen(host, port)`, `stop()`; streaming `POST /v1/completions`, `GET /v1/models` (`mllm serve`) |

`generate` is sugar: `submit`, then `step` until that request is done. `submit` + `step` is the real API, so many requests later need no new structure. `mllm serve` (`server.mm`) is a thin HTTP layer over `submit` + `step` + `cancel`; the engine doesn't know about it.

### Memory
| Region | Python (throwaway) | C++ |
|---|---|---|
| Weights | `mlx_lm.load()` modules; compiled Core ML model per dense block, cached under `/Volumes/T5/cache/mllm/ane/<model>/S<chunk>_<format>/` | `mmap` + one `newBufferWithBytesNoCopy` per file (a header-padded copy under the cache when the data doesn't start 16-aligned; mlx-community files don't), in the residency set; the same cached `.mlmodelc` per block, loaded by `AneOps` |
| Arena storage | one numpy `uint8` array `[N, unit_bytes]`, with typed views per object: KV units as fp16 `[G, 2, kv_heads, T, head_dim]` (same bytes as `offset_in_unit`, so it checks the layout), state units as fp32 | one shared, untracked `MTLBuffer`, pre-touched with `memset`, in a residency set (`requestResidency` again every 10 s); byte-identical layout to Python's, so golden K/V compares directly |
| Arena bookkeeping | Python class with the exact API: `admit(n)`, `alloc()`, `release(slot)`, `retire(step)` | C++ class (memory-design build step 2), unit-tested on its own ([cpp-port.md](cpp-port.md) §3) |
| Workspace | object with named fields (`x`, `q`, `k`, `v`, `a`, `logits`) holding MLX/numpy arrays; ops reassign them | fixed `MTLBuffer`s, plus every scratch buffer `GpuOps` writes during a step (argmax `ids`, attention partials, split-k, linear-attn temporaries), so lanes on two queues never share one; `ws.enc` is the lane's current encoder. Each lane owns its workspaces: prefill one (sized for the chunk; the ANE lane a second one by chunk parity, with its own ANE surface set, that owns only its inputs and outputs, tokens / x / o / q / k / v / gate / qkv / z / b / a, ~16 MiB at chunk 512 on the 0.8B, and shares the first one's GPU scratch, ~335 MiB at max_ctx 16384: the lane's command buffers run one after another), decode two (by step parity). ANE in/out are IOSurfaces read by Metal without a copy |
| Step tables | the slot lists inside each work item | 2 `MTLBuffer`s per lane, alternated by step parity |

Ops receive the workspace and read and write its named fields. That gives both languages the same call shape. Python's `ws.q = ...` becomes C++ writing into `ws.q`'s buffer. One exception (g9): at M = 1 the GPU `dense` runs merged projections, and `attn` reads `z` / `b` / `a` / `v` as slices of `ws.qkv` / `ws.qg` (`GpuOps::merge`; off in AneOps' copy).

### Work items (plain data, what lanes see)
```
PrefillWork:  req_id, tokens[int] (padded to a compiled chunk size), n_valid, start_pos,
              slots[int]           # every page up to start_pos + n_valid, already allocated
DecodeWork:   step, entries: [(req_id, input_token, pos, slots[int])], sampling params per entry
```
- Chunk sizes are multiples of 32 tokens (one page). Prefix-cache hits also end on a page boundary, so every chunk starts at a fresh page. `attn` writes only `n_valid` rows.
- The engine allocates all units a piece of work needs **before** `submit`. Lanes never touch the allocator.

## Engine loop
Same code in Python and C++:
```
step():
    for w in prefill.poll():                 # chunk finished, its K/V is written
        req.pos += w.n_valid
        if req.pos == len(req.ids) - 1: req.state = DECODE
    for req_id, tok in decode.poll():
        req.pos += 1; req.out.append(tok)
        emit req.detok.push(tok)
        if stop(req): finish(req)            # units → pending at the last submitted step

    for req in waiting:                      # admission = the one capacity question
        if arena.admit(demand(req)) == OK: req.state = PREFILL

    if not prefill.busy() and (req := next_prefill()):   # C++: while prefill.can_submit() (ANE lane: 2 chunks,
        prefill.submit(next_chunk(req))      # a request's next one too); allocs the chunk's pages from req's reservation

    if decode.can_submit() and (batch := decoding()):
        decode.submit(build_step(batch))     # allocs a page for each req crossing a page boundary
```

### Concurrency
| | Python | C++ |
|---|---|---|
| Engine | caller's thread | one thread (the caller's; in `mllm serve` the server's engine thread) owns the bookkeeping and both lanes; the only other thread is the ANE prefill lane's (see PrefillLane), which never touches bookkeeping. A `step()` with nothing to poll or submit blocks on the oldest lane event (decode's if one runs) |
| PrefillLane | runs inside `submit` | GPU: own `MTLCommandQueue`, one chunk in flight; `submit` encodes and commits, `poll` reads the lane's `MTLSharedEvent` (never blocks). ANE (`AneOps`): a lane thread (Core ML `predict` blocks) with two chunks in flight, of any requests, taking turns: a turn waits only for its own chunk's last command buffer, runs ANE `dense(i)`, and commits GPU `attn(i)` plus the next block's inputs without waiting, so the ANE runs one chunk while the GPU runs the other's attn. One queue and an `MTLFence` order a request's later chunk after its earlier one, layer by layer; chunks finish in submit order. A chunk under `gpu_below` (256) tokens (tails, prefix-cache splits) runs all layers on the GPU path on this thread instead of a padded ANE chunk. A chunk ending at a snapshot point holds back its request's next chunk until `poll` (the copy needs it done). It encodes with its own copy of `GpuOps` (GpuOps keeps the current encoder in members); the engine sees the same submit / poll / event |
| Server | — | httplib handler threads (a pool of max(8, cores − 1), growing to 4× that: 52 here) only parse, queue `{prompt, params}` and stream their request's events; the engine thread (`Server::loop`) tokenizes, calls `submit` / `cancel` / `step` and fills per-request queues. It sleeps on a condition variable when no request is live. A client that hangs up is cancelled within ~100 ms, even before its first token (the stream wakes every 100 ms so httplib re-checks the peer). Bodies over 1 MiB get 413: the engine thread tokenizes, ~170 ms per MB, stalling every stream |
| DecodeLane | runs inside `submit`, `mx.eval` per step | own `MTLCommandQueue`, one command buffer per step, one `MTLSharedEvent` signaled with step + 1; a concurrent compute encoder, with a buffer barrier before each dependent dispatch (`GpuOps::pre`; split-k matmuls always wait, since every linear shares `ws.splitk`; `test_concurrent` checks concurrent == serial at B = 13 and 16); up to 2 steps in flight, ordered by an `MTLFence` (defensive: a fed step already reads the previous step's tracked `ids`). Not while a sampled row decodes: then all rows step together, one step in flight. Greedy rows of step N + 1 take step N's argmax on the GPU (`GpuOps::feed_tokens`); sampled rows wait for the CPU |

- The two lanes write disjoint units: a request is in one lane at a time. So the two GPU queues need no ordering between them.
- A request moves PREFILL → DECODE only after `prefill.poll()` returns its last chunk, which means its GPU writes have completed.
- Units freed by decode go through `pending` (memory-design), because the next step is already encoded. Prefill units are freed only after `poll()`: an aborted request waits for its chunk to finish.
- In Python, prefill and decode do not overlap. In C++ they overlap on the GPU through the two queues. Measured: a GPU prefill stream costs B=8 decode ~70%, an ANE one 7–9% ([report](../reports/mlx-lm%20baseline.md) §1, ANE prefill).

## Files
```
py/mllm/spec.py  tokenizer.py  memory.py  gpu.py  ane.py  lanes.py  engine.py
py/test_engine.py
src/     spec  safetensors  tokenizer  memory  gpu  ane  lanes  prefix_cache  engine  server   (.h + .cpp/.mm)
src/main.mm         mllm CLI: generate, bench, bench-ops, serve
src/kernels/*.metal attention  elementwise  linear_attn  matmul  norm  quantized  rope  sample  softmax  smoke, table.h
src/vendor/         mlx (kernel headers), nlohmann, llama-unicode, cpp-httplib
tests/test_*        one executable per component + test_golden / test_stack / test_server / test_engine_cache; golden.h
```
Python reuses mlx-lm wherever it can: `mlx_lm.load` for weights and the GPU dense modules, `TokenizerWrapper` and `StreamingDetokenizer` for the tokenizer, `mx.fast.scaled_dot_product_attention` over gathered pages for `attn`. *ponytail: the gather copies every page of a request into MLX every layer, every step. That is fine for checking correctness and useless for speed. The C++ paged kernel replaces it.*

## Checks
- `py/test_engine.py`: with GPU-only prefill, greedy 64 tokens on a fixed prompt must equal `mlx_lm.generate` on the same weights. The arena layout and the `prompt[:-1]` trick are then proven against a reference.
- `py/test_ane.py [fp16|int8|lut6|lut4]`: ANE prefill and GPU prefill write the same prompt into separate units. Every full layer's K/V and every linear layer's conv and recurrent state must reach cosine ≥ 0.999 (fp16) or ≥ 0.99 (compressed formats). Then greedy decode after ANE prefill is compared with mlx-lm.
- The arena bookkeeping gets its own unit test (memory-design build step 2). Python and C++ run the same test cases.
- Greedy batched output equals solo only away from near-ties: at batch ≥ 2 the matmuls run other kernels (`qmv_wide` / split-k instead of `qmv_fast` at M=1) that round differently, so near-tied top-2 tokens can flip, as in MLX itself (QA: 11 divergences, all with MLX's top-2 within 0–2 bf16 ulps). The golden batched check passes because its prompts have no such tie.

## Build order
1. `spec`, `tokenizer`, `engine`, both lanes GPU-only, `attn` over the numpy arena for both layer kinds of the first model: full attention (K/V pages) and Gated DeltaNet (state slot). Pass the mlx-lm match on Qwen3.5-0.8B. **Done (2026-11-22, `py/test_engine.py`): bit-exact with mlx-lm greedy for 3 prompts × 64 tokens, solo and batched, and with requests waiting on a one-request arena. Unit 384 KiB, state slot 50 units (~19.5 MiB per request).** The basic arena bookkeeping from step 2 (`admit` / `alloc` / `release`) is already in, without refcounts or `pending`.
2. Real arena bookkeeping (`admit` / `alloc` / `release` / `pending`) and several requests through `submit` + `step`.
3. `AneOps.dense` for prefill: one Core ML model per dense block, built with `tools/hetero_bench.py`'s builder from real weights. Pass the K/V comparison. **Done (2026-11-22, `py/mllm/ane.py`, `py/test_ane.py`), on Qwen3.5-0.8B, S = 128, a 392-token prompt (4 chunks, last one padded):**
   - **One model per block** (`o + FFN + norm + next q,k,v`), 24 blocks. Every op lands on the ANE, including RMSNorm (as `layer_norm([x, −x])`, Anemll's trick) and the per-head q/k norms. Embedding is a GPU gather before block 0.
   - **Speed (Python):** `predict()` alone takes 27 ms per chunk, 4,700 tok/s (≈ 4.5 TFLOPS, small shapes). The whole lane, with numpy copies and GPU `attn`, does 1,370 tok/s, against 2,400–3,100 for GPU `dense` in the same Python lane. The ANE time is mostly copies and Python, which C++ removes.
   - **fp16 precision:** the ANE is ~6× further from an fp32 GPU run than bf16 is (median 1−cos 3.2e-4 vs 5e-5). fp16 has more mantissa than bf16, so this comes from the ANE's internal arithmetic, not storage. It is still small: greedy decode after ANE prefill gives mlx-lm's exact 64 tokens.
   - **Weight formats**, built from the dequantized MLX 4-bit weights:

     | ANE weights | Size (0.8B) | Per-block worst cos | 24-layer K/V + state worst cos | Greedy 64 vs mlx-lm |
     |---|---|---|---|---|
     | fp16 | 925 MB | 0.9999 | 0.99901 | 64/64 |
     | int8 per-channel | 464 MB | 0.9999 | 0.99873 | 64/64 |
     | LUT6 k-means, groups of 16 rows | 351 MB | 0.9996 | 0.99281 | 64/64 |
     | LUT4 k-means, groups of 4 rows | 235 MB | 0.994 | 0.934 | — (fails) |
     | LUT4 uniform, groups of 16 (the spike's) | 233 MB | 0.964 | 0.40 | — |
     | LUT4 + per-channel scale | — | 0.968 | — | — (runs on the CPU, 5–9× slower) |

     All of these run at the same ANE speed (1.1–1.3 ms per block), except LUT4 per-row k-means (2.4 ms). LUT4 fails because a 16-value table shared by whole rows can't represent weights that already have a separate scale for every 64 inputs.
4. Port to C++ without Python bindings: Python stays the reference through dumped golden data. Checklist and decisions: [cpp-port.md](cpp-port.md).

## Deferred, with triggers
| Item | Add when |
|---|---|
| ~~Prefix cache~~ Done (g8): `PrefixCache`, on by default (`--prefix-cache on|off`; off for `bench`). Resumes only at a state snapshot (the prompt's last full page, or a branch point) | Generated tokens cached (multi-turn without resending), or requests waiting for a snapshot an in-flight request is about to take |
| Page-level sparse decode on full layers (Quest-style: a min/max key summary per page, attend to the top-k pages). Saves bandwidth. Saves RAM only if cold pages are spilled to the SSD tier | Full-layer K/V reads show up in decode time at long context |
| Sliding-window `attn` kind (window pages) | Running Gemma 4 or another sliding-window model |
| ~~HTTP / OpenAI-style server~~ Done (cpp-port §7): `mllm serve`, streaming `/v1/completions` only | Non-streaming, chat completions, more params: a client that needs them |
| ~~C++ tokenizer choice~~ Done: our own byte-level BPE with llama.cpp's unicode tables, ChatML hard-coded (cpp-port §2) | Jinja templates: a second model family |
| ~~GPU-side token feedback~~ Done for greedy rows (one step ahead, g5). Sampled rows (temp > 0) still wait for the CPU each step: sampling on the GPU | Sampled decode is the hot path |

## First model
**Qwen3.5 / 3.6.** Develop on Qwen3.5-0.8B or 2B, then run the target, Qwen3.6-27B. They share one architecture (mlx-lm `qwen3_5`):
- Every 4th layer is full attention (gated). The other 3 are Gated DeltaNet linear attention, which keeps a fixed-size state.
- 27B: 64 layers, 16 of them full attention, 4 KV heads × 256 dims → 16 × 2 × 4 × 256 × 2 B = **64 KiB / token**. The 48 linear layers hold a fixed state per request.
- Both arena object kinds get exercised from day one: KV pages, plus a state slot.
- The seam still holds. The DeltaNet conv1d and the recurrence carry state, so they go in `attn(i)` on the GPU. Its projections go in `dense(i)`.

**Why hybrid linear attention, not sparse attention.** Sparse attention (DeepSeek DSA, Quest) reads fewer tokens per step, but it must still store every token's K/V, because any token can be picked. So it saves bandwidth and compute, not memory. Linear layers store a fixed state, so they do save memory.
- Qwen3.6-27B at 128k tokens: 64 KiB × 128k = **8 GiB** (4 GiB with 8-bit KV), plus ~150 MiB of linear state *(estimate)*.
- A full-attention model of the same class (Qwen3-32B: 64 layers × 8 KV heads × 128 dims) needs 256 KiB/token = **32 GiB**.
- Fully linear models (Mamba, RWKV-7) store no K/V at all, but they recall poorly from long contexts, and none competes at this size.

Sparse reads can still go on top later, on the 16 full layers (see Deferred).

Gemma 4 comes second. It adds sliding-window pages, plus quirks that would each need special cases in the arena: shared K=V on global layers, KV shared across layers, and per-layer input embeddings in the small sizes. Its small and large models also differ in which of these quirks they have, so a small dev model wouldn't test the big one.

Before step 1: read the dev model's `config.json` for `layer_types` and the linear-attention head counts, and check that mlx-community has 4-bit text-only conversions.

## Open questions
| Question | Decides |
|---|---|
| ~~Does one Core ML model per dense block work on the ANE?~~ **Yes** (build step 3): every op on the ANE, one call per layer. Its speed at 27B shapes (needs the spike's splits?) is still unmeasured | Whether the ANE beats the GPU for 27B prefill |
| Which ANE weight format for the 27B? fp16 = 54 GB (impossible); int8 = 27 GB; LUT6 ≈ 20 GB; LUT4 ≈ 14 GB, but LUT4 made from the 4-bit MLX weights fails (cos 0.93). Try LUT4 made from the **bf16 original** weights, and check whether the 27B tolerates LUT4 better than the 0.8B. Next to 15 GB of GPU 4-bit weights and ~12–16 GB for macOS, only LUT4 leaves room for a real arena | Whether ANE prefill fits in 64 GB next to the GPU weights at all |
| ~~Does ANE prefill beat the GPU once C++ removes Python's copies?~~ **Alone, a little** (C++ port): 5,670 vs 5,300 tok/s at 512. **Under decode load, no**: a chunk slows to 170–225 ms (per-layer GPU round trips), but decode next to it loses 7–9% instead of ~70%. Served: GPU prefill gives the better TTFT (0.62 vs 1.46 s median at 8 × 512), ANE prefill the better per-client decode (218 vs 110 tok/s) ([report](../reports/mlx-lm%20baseline.md) §1, §2) | Default `--prefill`. Since then the ANE lane keeps two chunks in flight and runs tails on the GPU (two 512 chunks 180 → 119 ms alone, noisy); the served numbers wait for a rerun |
| Where does the ANE's fp16 error come from (accumulation, `layer_norm`, `silu`)? | Whether fp32 or split ops are worth it for longer prompts |
