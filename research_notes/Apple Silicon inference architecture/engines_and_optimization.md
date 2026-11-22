# LLM Inference Engines on Apple Silicon (non-MLX) + Performance Model and Optimization Techniques

Notes as of 2026-11-21. Target machine: Mac mini M4 Pro (20-core GPU, 64 GB, 273 GB/s).
Labels: **[measured]** = benchmark in a paper, repo, or vendor post with a stated method. **[anecdotal]** = community/blog number with unclear method. **[derived]** = my arithmetic from cited numbers.
Local repo clones read: llama.cpp at commit `19bc41d` (2026-11-21), exo, candle, mistral.rs, Anemll, metal-benchmarks, mlx-lm. GitHub URLs are given with file paths.

---

## 1. llama.cpp / ggml Metal backend: architecture, GGUF/mmap, quant formats, flash attention, CPU/GPU split, M-series performance

### Takeaway
ggml-metal is a hand-written Metal backend. It encodes the whole compute graph into a few command buffers, with multi-threaded encoding, concurrent dispatch plus explicit barriers, and a pattern-based fusion pass. It maps mmapped GGUF weights zero-copy into MTLBuffers and keeps them wired with residency sets. Its decode GEMV kernels reach about 70–85% of peak bandwidth on M4 Pro (on the 2023 reference build). Prefill scales with GPU core count and decode scales with bandwidth, which is the cleanest evidence for the performance model in Section 3.

### Cited Findings
**Architecture (source at commit 19bc41d)**
- The backend is split into `ggml-metal-context.m` (graph encoding and command buffers), `ggml-metal-device.m` (device, buffers, residency), `ggml-metal-ops.cpp`, `ggml-metal-fusion.cpp`, and `ggml-metal-tuning.cpp`. Kernels live in per-op `.metal` files: `mul_mv.metal`, `mul_mv_mma.metal`, `mul_mm.metal`, `fa_*.metal`, `fa_vec_*.metal`, `rope.metal`, `norm.metal`, `softmax.metal`, `ssm.metal`, `gated_delta_net.metal`, and others — [llama.cpp ggml-metal dir](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal)
- **Parallel command-buffer encoding.** The main thread encodes the first nodes while `n_cb` extra threads, on a concurrent GCD queue via `dispatch_apply`, encode the rest of the graph into separate command buffers. A source comment says "tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2", and the code warns that n_cb > 2 "can degrade the performance" — [ggml-metal-context.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m) (lines ~488–607, 730–741)
- **Concurrent dispatch.** Encoders are created with `MTLDispatchTypeConcurrent`, and hazards are handled with `memoryBarrierWithScope:MTLBarrierScopeBuffers`. A `ggml_mem_ranges` structure tracks the read and write buffer intervals of each op, so a barrier is inserted only when ranges overlap. A `ggml_graph_optimize` pass reorders nodes, enabled by default — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (L842, L895); [ggml-metal-common.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-common.cpp) (L160–190); ggml-metal-context.m L684
- **Fusion patterns implemented:**
  - NORM/RMS_NORM+MUL(+ADD), NORM+SCALE
  - ADD chains of 2–7
  - TOPK_MOE (SOFT_MAX+ARGSORT+GET_ROWS+norm/scale/bias, for MoE routing)
  - MOE_REDUCE (expert-weighted sum)
  - MUL_MAT+ADD (residual folded into the matmul store)
  - SSM_CONV+SILU, GATED_DELTA_NET+CPY (for hybrid/linear-attention models)

  — [ggml-metal-fusion.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-fusion.h) L34–46; [ggml-metal-fusion.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-fusion.cpp) L866–890
- **Zero-copy mmap.** Host (mmapped) memory is wrapped with `newBufferWithBytesNoCopy`, so GGUF weights are not copied into GPU memory. Unified memory lets the GPU read the page-cache pages directly — ggml-metal-device.m L2207; ggml-metal-context.m L375
- **Residency sets, macOS 15 and later.** The backend creates an MTLResidencySet collection. A background heartbeat thread periodically calls `requestResidency` so the weights stay wired. The default keep-alive is 3 minutes, overridable with the env var `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`. A workaround handles residency-set memory not being released when no GPU work happens — ggml-metal-device.m L19–24, L933–1025
- **Feature detection.** `has_simdgroup_mm` is set when the device supports `MTLGPUFamilyApple7` (M1 and later), which enables simdgroup_matrix GEMM. `has_tensor` tests for the Metal 4 tensor API, and kernels compile with `GGML_METAL_HAS_TENSOR`, which is used by `fa_tensor.metal` (the M5 neural-accelerator path, not present on M4) — ggml-metal-device.m L1138–1209
- **Per-quant GEMV tiling.** Each quant type has constants `N_R0_*` (rows per simdgroup) and `N_SG_*` (simdgroups per threadgroup), e.g. Q4_0: N_R0=4, N_SG=2; Q8_0: 2/4; MXFP4: 2/2. Separate `mul_mv_mma` ("few-row MMA") kernels handle small batches — [ggml-metal-impl.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-impl.h) L29–56
- **Flash attention kernels** exist per K/V cache type, which means FA runs directly on quantized KV cache:
  - Prefill/batch: `fa_f16/f32/q4_0/q4_1/q5_0/q5_1/q8_0`
  - Single-token decode: `fa_vec_*`

  — [kernels dir](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal/kernels)

**Quant block layouts** — [ggml-common.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)

| Format | Block structure | Bits/weight |
|---|---|---|
| Q4_0 | 32 weights: fp16 scale + 16 bytes of nibbles | 4.5 |
| Q8_0 | 32 weights + fp16 scale | 8.5 |
| MXFP4 | 32 E2M1 values + 1 E8M0 shared exponent | 4.25 |
| Q4_K | 256-weight super-block, 2 fp16 (d, dmin) + packed 6-bit sub-block scales/mins + 128 bytes of quants | — |
| IQ2_XXS | lattice/codebook (`iq2xxs_grid` lookup table) | ~2.06 |
| IQ4_XS | non-linear 4-bit codebook | ~4.25 |
| NVFP4 | also present | — |

- **Measured bits/weight for Llama-3.1-8B** (llama.cpp quantize README; hardware for its t/s columns not stated there, so use only the bpw):

| Format | bpw | Format | bpw |
|---|---|---|---|
| Q4_K_S | 4.67 | Q4_K_M | 4.89 |
| Q5_K_M | 5.70 | Q6_K | 6.56 |
| Q8_0 | 8.50 | IQ4_XS | 4.46 |
| IQ3_XXS | 3.25 | IQ2_XXS | 2.38 |
| F16 | 16.0 | | |

  — [tools/quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

**Discussion #4167, M-series table** [measured]
- Method: LLaMA-2 7B, `llama-bench -p 512 -n 128 -ngl 99`, frozen build `8e672ef` (2023-11-21) so rows stay comparable. Newer builds with FA are faster: M2 Ultra F16 TG went 41.02 → 49.55 on `c1d0e7a`+FA — [llama.cpp discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)

| Chip | GPU cores | BW (GB/s) | F16 PP512 / TG128 | Q8_0 PP / TG | Q4_0 PP / TG |
|---|---|---|---|---|---|
| M4 | 10 | 120 | 230 / 7.43 | 224 / 13.54 | 221 / 24.11 |
| M4 Pro | 16 | 273 | 381 / 17.19 | 367 / 30.54 | 364 / 49.64 |
| **M4 Pro** | **20** | **273** | **464 / 17.18** | **450 / 30.69** | **440 / 50.74** |
| M4 Max | 32 | 410 | 736 / 24.29 | 719 / 43.87 | 714 / 69.95 |
| M4 Max | 40 | 546 | 923 / 31.64 | 892 / 54.05 | 886 / 83.06 |
| M3 Max | 40 | 400 | 779 / 25.09 | 758 / 42.75 | 760 / 66.31 |
| M3 Ultra | 80 | 800 | 1538 / 39.78 | 1488 / 63.93 | 1471 / 92.14 |
| M2 Ultra | 76 | 800 | 1402 / 41.02 | 1249 / 66.64 | 1238 / 94.27 |

**Recent comparisons**
- llama.cpp (GGUF) vs MLX on the same models [measured]:
  - GGUF leads on prefill; MLX usually leads on decode.
  - LFM2.5-8B-A1B on M2 Pro: GGUF-Q4_0 prefill 1,180 tok/s, decode 129.3 tok/s; MLX-W4-DWQ decode 119.9 tok/s, prefill ~550 tok/s.
  - On M3 Ultra: GGUF-Q4_0 prefill 3,820 vs 3,416 for MLX-W4-Affine (+12%); MLX decode 247.0 vs 232.2 (+6%).
  - K-quants are slower than Q4_0 on Apple prefill (Q4_K_M prefill 1,035 vs Q4_0 1,180 on M2 Pro). That k-quant prefill penalty is "largely absent on the RTX 4090".

  — [Yemets et al., "Evaluating the Applicability of LLM Inference Optimizations on Apple Silicon", arXiv 2610.31479, Oct 2026](https://arxiv.org/abs/2610.31479)
- Moving from Q8_0 to Q4_0 speeds decode by 1.64× on M2 Pro (200 GB/s) but only 1.20× on M3 Ultra (800 GB/s) [measured] — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)

### Inferences
- **Decode is bandwidth-bound, prefill is compute-bound [derived from #4167].** M4 Pro 16-core vs 20-core: TG is identical (17.19 vs 17.18 t/s F16; 49.64 vs 50.74 Q4_0), while PP scales 1.22× against a 1.25× core ratio. Same bandwidth gives the same TG; more cores give more PP.
- **Achieved bandwidth on M4 Pro 20c [derived].** Llama-2 7B file sizes: F16 ≈ 13.5 GB, Q8_0 ≈ 7.16 GB, Q4_0 ≈ 3.83 GB (standard llama-bench sizes of 12.55 / 6.67 / 3.56 GiB).

  | Format | Effective GB/s | % of 273 GB/s |
  |---|---|---|
  | F16 | 17.18 × 13.5 ≈ 231 | ~85% |
  | Q8_0 | ≈ 220 | ~80% |
  | Q4_0 | ≈ 194 | ~71% |

  Efficiency drops as bits shrink because dequant ALU work and fixed per-token overhead stay the same while the bytes shrink.
- **Bigger chips use their bandwidth less well on small models [derived].** Q4_0 TG reaches only ~58% of peak on M4 Max 40c (83 × 3.83 / 546) and ~44% on M3 Ultra (92 × 3.83 / 800). Fixed per-token latency (dispatch, sync, small kernels) dominates, so an engine on an M4 Pro should target ≥ 80% of 273 GB/s for 7–32B models.
- **Quant choice on Apple:**
  - Q4_0 and MXFP4 (simple blocks) are cheapest to dequantize and are the fastest for both prefill and decode.
  - K-quants and IQ quants trade ALU for quality. IQ formats need codebook lookups, which matter on prefill.
  - For a custom engine, a simple group-wise affine 4-bit format (like MLX's, or Q4_0 with group size 32–64) is the pragmatic default.

### Gaps
- No current-build (late 2026) #4167 table for M4 Pro with FA enabled was found; only the M2 Ultra FA delta was visible.
- How the CPU/GPU graph split behaves when not all layers are offloaded (`-ngl < n_layers`) was not investigated in source. The scheduler code is in `ggml-backend.cpp`. On a 64 GB Mac the normal case is full offload.
- Exact % of peak for K-quants and IQ formats on M4 Pro: no measured source found.

---

## 2. Other engines on Apple Silicon

### Takeaway
By late 2026 the Mac ecosystem has consolidated on two GPU kernel stacks, ggml-metal (llama.cpp) and MLX, with most front-ends wrapping one of them:
- Ollama: the MLX engine became the default in 2026.
- LM Studio: ships both a llama.cpp and an MLX engine.
- vllm-metal and vllm-mlx: serving layers on MLX.
- exo: distributed inference on MLX.

There are also independent native-Metal runtimes (BaseRT, uzu, MetalRT), Rust stacks (candle, mistral.rs), TVM-compiled MLC-LLM, and Core ML and ANE paths. The ANE paths remain niche for LLMs: they save power but are not a throughput win at batch 1 on Macs.

### Cited Findings
**Core ML (Apple)** [measured]
- Apple's "On Device Llama 3.1 with Core ML" (2024-11-01; M1 Max, Llama-3.1-8B-Instruct, 2048 ctx) shows the effect of each step:

  | Configuration | Extend throughput (tok/s) | Prompt TTFT |
  |---|---|---|
  | Baseline (static shapes, no KV cache) | 0.19 | — |
  | KV cache as model I/O | 1.25 | — |
  | KV cache as Core ML **state** (fp16) | 16.26 | 128 ms |
  | State + **int4 block-wise (block 32)** quant | 33.67 | 52 ms |

  - The int4 model is 4.2 GB vs 16 GB fp16.
  - Decode falls with the max context the state is sized for: 39.4 tok/s @128 → 33.7 @2048 → 21.5 @8192.
  - Uses fused SDPA (macOS 15 target) and flexible input shapes.
  - It **targets the GPU, not the ANE**, because 8B models are bandwidth-bound and the GPU has the best FLOPs plus bandwidth.

  — [Apple ML Research](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- On macOS 26.3, Core ML `compute_units=ALL` routed a Qwen3.5 prefill to the **GPU, not the ANE**. Measured ANE power was ≈0 W (max 0.024 W). Core ML GPU kernels still drew less power than MLX: 22.5 W vs 53.2 W total for 9B at the long prompt [measured] — [AtomGradient hybrid-ane-mlx-bench, 2026, M2 Ultra](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)

**Apple Foundation Models (on-device, ~3B)**
- The model uses KV-cache sharing: block 2 (37.5% of layers) reuses block 1's KV, cutting KV memory by 37.5% and TTFT by about 37.5%. It also uses 2-bit quantization-aware training with a learnable per-tensor scale. Quality is reported competitive with Qwen-2.5-3B, and with Qwen-3-4B and Gemma-3-4B in English — [Apple Intelligence Foundation Language Models Tech Report 2025, arXiv 2507.13575](https://arxiv.org/abs/2507.13575); [InfoQ](https://www.infoq.com/news/2025/07/apple-foundation-models-ios26)
- The Foundation Models framework exposes only Apple's model, not custom weights. Its techniques (KV sharing, 2-bit QAT) are design ideas, not an engine you can reuse — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575)

**ANE projects**
- **ANEMLL** (0.3.5 beta, needs coremltools ≥ 9.0):
  - Splits models into ANE-sized chunks (embeddings, FFN chunks, LM head) because monolithic models fall back to GPU/CPU.
  - Recent additions: monolithic multifunction models with "ANEMLL-Dedup" (~50% size reduction), and in-model argmax to reduce ANE-to-host transfer.
  - Uses IOSurface-backed buffers and ping-pong buffers to avoid ANE races.
  - Supports LLaMA, Qwen/Qwen2.5/Qwen3, and Gemma 3.

  — [github.com/Anemll/Anemll README](https://github.com/Anemll/Anemll)
- **apple/ml-ane-transformers** is the reference for ANE-friendly layouts, for example Linear → Conv2d(1×1) and channels-first. AtomGradient's conversion applies the same "Linear → Conv2d(kernel_size=1)" rule — [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers); [AtomGradient paper](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
- **Private-API ANE** (`AppleNeuralEngine.framework`, "ANE-LM") on M2 Ultra [measured]:
  - Sequential dispatch costs ~42 ms per token regardless of prompt length. Qwen3.5-0.8B TTFT is 17.8 s for 422 tokens vs 96 ms on the MLX GPU.
  - Batched dispatch (32 tokens per call) reaches 268 tok/s prefill (0.8B) and 173 tok/s (2B), 11.3× and 7.3× over sequential.
  - Offloading prefill to the ANE cuts GPU power from 62.05 W to 0.22 W.

  — [AtomGradient paper](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
- **Orion** (arXiv 2603.06728, Mar 2026) is a direct-ANE runtime using the private `_ANEClient`/`_ANECompiler` APIs [measured]:
  - Documents 20 ANE restrictions, 14 of them previously undocumented.
  - Weights are baked in at compile time: about 4.2 s to recompile, or 494 ms by patching the weight file.
  - GPT-2 124M decodes at 170+ tok/s on M4 Max.
  - Stability across OS updates is a risk.

  — [arXiv 2603.06728](https://arxiv.org/abs/2603.06728)
- uzu (Mirai) routes work across GPU kernels and MPSGraph, the latter potentially reaching the ANE. The BaseRT authors only *suggest* MPSGraph/ANE "likely accounts for" uzu's 1–13% prefill lead; this is unverified — [BaseRT, arXiv 2607.00501](https://arxiv.org/abs/2607.00501)

**MLC-LLM (TVM-compiled Metal kernels)**
- On M2 Ultra with Qwen-2.5: about 190 tok/s vs MLX about 230 tok/s, but with lower TTFT for prompts ≤ 16k and a paged KV cache.
- The same paper reports llama.cpp at about 150 tok/s (short context only), Ollama at 20–40, and PyTorch MPS at 7–9 tok/s. PyTorch MPS is described as hampered by a 4 GB tensor limit and irregular bf16/int4 support.
- Caution: the same paper's claim of llama.cpp at "~1.2 tok/s at 32k" looks implausible next to FA-enabled llama.cpp, so treat its llama.cpp and Ollama numbers as low-confidence.

— [Rajesh et al., arXiv 2511.05502, Nov 2025](https://arxiv.org/abs/2511.05502)

**Ollama**
- Ollama historically used a ggml/llama.cpp-based engine.
- MLX-powered preview in v0.19 (announced ~2026-03-30/31), opt-in and requiring ≥ 32 GB. Third-party numbers: decode ~58 → 112 tok/s, prefill 1,154 → 1,810 tok/s [anecdotal].
- MLX reportedly became the default Apple path in v0.30 (May 2026); I did not verify this against Ollama's release notes.
- Tom's Hardware reports 1.5× prefill and 2–3× decode for Qwen on M4 Pro vs the old backend [anecdotal].

— [Ollama blog](https://ollama.com/blog/mlx-apple-silicon); [DEV](https://dev.to/alanwest/ollama-just-got-93-faster-on-mac-heres-how-to-enable-it-3gce); [DEV v0.30](https://dev.to/jovan_chan_9500711396d4e6/ollama-v030-on-apple-silicon-what-the-stable-mlx-release-actually-changed-from-the-preview-25h7); [Tom's Hardware](https://www.tomshardware.com/desktops/exploring-apple-silicons-local-ai-performance-with-the-mac-studio-and-m4-max-m4-max-beats-gb10-and-strix-halo-in-decode-throughput-but-memory-bandwidth-isnt-everything/2)

**LM Studio**
- Ships two runtimes: llama.cpp for GGUF and `mlx-engine` for MLX. Not re-verified this session — [lmstudio-ai/mlx-engine](https://github.com/lmstudio-ai/mlx-engine)

**vLLM on Mac**
- **vllm-metal** (vLLM blog 2026-09-22, v0.28.0):
  - Architecture: a vLLM V1 scheduler, paged KV blocks, chunked prefill, and the OpenAI server. Model layers come from mlx_lm; only attention is replaced, by a paged varlen Metal kernel ported from vLLM's Triton kernel. Prefill and decode are packed into one forward pass.
  - Features: prefix caching, which can save hybrid-model GDN state, and speculative decoding (MTP, a draft model, or prompt-lookup n-gram).
  - Mixed prompt lengths: batch wall time 3.64 s vs mlx_lm 10.99 s and llama.cpp 5.33 s (Qwen3.6-35B-A3B 4-bit, 8 requests) [measured, vendor].
  - Gemma 4 E4B with MTP: +20% output tok/s at concurrency 1, +0% at 8.

  — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)
- Docker's benchmark found llama.cpp ahead at 1 request: 333 vs 251 tok/s for Llama 3.2 1B 4-bit [measured, single test] — [Docker blog](https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/)
- **vllm-mlx** (Barrios, arXiv 2601.19139): MLX plus continuous batching and a paged KV cache. Self-reported 21–87% higher throughput than llama.cpp, 4.3× aggregate throughput at 16 concurrent requests, and up to 525 tok/s (Qwen3-0.6B, M4 Max) [measured, self-reported] — [arXiv 2601.19139](https://arxiv.org/abs/2601.19139)

**Native-Metal runtimes**
- **BaseRT** (Base Compute, arXiv 2607.00501, Jul 2026) is C++ calling Metal directly, with no MLX:
  - Zero-allocation decode loop (all buffers, including the KV cache, preallocated)
  - Fused attention and FFN operator sequences
  - Dequantization in the inner loop
  - Per-chip threadgroup tuning
  - CPU–GPU sync amortized across several tokens
  - simdgroup_matrix tiled GEMM with chunked prefill
  - FlashAttention-style online softmax

  Results are in Section 5. No per-technique ablation is given — [arXiv 2607.00501](https://arxiv.org/abs/2607.00501); [github.com/basecompute/baseRT](https://github.com/basecompute/baseRT)

**Rust stacks**
- **candle** has a Metal backend with its own `candle-metal-kernels` crate — [huggingface/candle](https://github.com/huggingface/candle/tree/main/candle-metal-kernels)
- **mistral.rs** (built on candle) supports Metal and in-situ quantization (ISQ), plus GGUF, GPTQ, and UQFF. As of v0.9.5, "paged attention is now the default on supported Metal models", with continuous batching and prefix caching on Apple Silicon — [EricLBuehler/mistral.rs README](https://github.com/EricLBuehler/mistral.rs)

**exo (distributed)**
- Clusters Macs with "RDMA over Thunderbolt 5" ("99% reduction in latency between devices") and tensor parallelism. Claims up to 1.8× speedup on 2 devices and 3.2× on 4 devices; demonstrated Qwen3-235B 8-bit and DeepSeek v3.1 671B 8-bit on 4× M3 Ultra (Jeff Geerling benchmarks) — [exo-explore/exo README](https://github.com/exo-explore/exo)

**M5 generation** (context only, not on M4 Pro)
- MLX uses the M5 GPU "Neural Accelerators", which need macOS 26.2 or later — [Apple ML Research, Exploring LLMs with MLX on M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- llama.cpp has a Metal-4 tensor path (`has_tensor`, `fa_tensor.metal`) — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)
- vllm-metal uses a "NAX" attention kernel for M5 prefill — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)

### Inferences
- For a custom engine on M4 Pro, the relevant competitors are llama.cpp, MLX/mlx-lm, and native-Metal runtimes (BaseRT, uzu). Core ML, ANE, PyTorch MPS, and candle are not decode-throughput leaders.
- The ANE is useful for low-power prefill or small models, but:
  - its public path (Core ML) may not even run on the ANE for these graphs;
  - its fast path needs private APIs;
  - it shares the same DRAM bandwidth, so it cannot speed up bandwidth-bound decode.
- vLLM-style paged/varlen attention only matters with concurrent requests (agents, servers). For single-user chat it adds complexity with little gain at batch 1.

### Gaps
- No primary source this session for LM Studio internals beyond the mlx-engine repo name.
- No benchmark found for the MLC-LLM, candle, or mistral.rs Metal backends on M4 Pro.
- No public numbers for Apple's on-device Foundation Model on Macs (tok/s) were found.
- Did not find whether Anemll publishes M4 Pro tok/s numbers (README excerpt had only accuracy tables).

---

## 3. Performance model: decode bandwidth bound, prefill roofline, KV growth, MoE, batching

### Takeaway
On M4 Pro, decode at batch 1 is bounded by **273 GB/s ÷ (bytes of active weights + bytes of KV read per token)**. Realistic engines reach 70–85% of that. Prefill is compute-bound above roughly 8 tokens per weight read for 4-bit weights, or about 30 for fp16. KV traffic grows linearly with context and can halve decode speed at 32k+. MoE models decode in proportion to *active* parameters, which makes them the best fit for this machine.

### Cited Findings
- Decode upper bound is T_max = B_mem / W_bytes. "Decode throughput scales linearly with memory bandwidth, confirming the decode phase is bandwidth-bound" (MLX on M2 Ultra; Qwen3.5-2B 8-bit 141.8 tok/s vs 2B BF16 101.3 tok/s) [measured] — [AtomGradient paper](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
- llama.cpp #4167 notes that at batch 512 "the computation is compute bound", and F16 is faster than quantized formats on PP because of dequant cost — [discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- **KV-cache bytes per token** = 2 · b · Σ_layers(h_kv · d) for global layers, with sliding-window layers capped at the window size [measured, derived from configs]:
  - Qwen3-4B (36 layers, 8 KV heads × 128, bf16): 144 MiB at 1k tokens, 4,608 MiB at 32k.
  - Gemma-3-4B (5 global layers, 29 local layers with a 1,024 window): 136 → 756 MiB.
  - Hybrid models grow by only 8–16 KiB per token (Granite-4.0-H-Micro, LFM2.5-2.6B).

  — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479) §2.1.2
- Context hurts decode in practice:
  - Core ML int4 Llama-3.1-8B: 39.4 tok/s at 128 max ctx → 21.5 at 8192 on M1 Max [measured] — [Apple](https://machinelearning.apple.com/research/core-ml-on-device-llama)
  - gpt-oss-20B 4-bit on M4 Pro: ~54 tok/s at 16K → ~33 at 64K [anecdotal] — [Hardware Corner](https://www.hardware-corner.net/is-mac-mini-m4-pro-good-for-llms/)
- **Apple GPU microarchitecture:**
  - 128 ALUs per GPU core.
  - Dual-dispatch is required to fully use FP16 at low occupancy.
  - Large register file (~512 KB registers/L1 per core) and a small L1D (8 KB).
  - "on bandwidth bound tasks, the GPU has no advantage over the CPU" at per-core I/O level (32 B/cycle).

  — [philipturner/metal-benchmarks README](https://github.com/philipturner/metal-benchmarks)
- **Dispatch overhead on M4 Pro with MLX** [measured]:
  - A GPU-side dispatch boundary costs ~3.04 µs.
  - CPU graph build and encode costs 0.54 ms per token but is fully hidden: pipelined 6.04 ms vs synchronous 6.53 ms per token, because mlx-lm overlaps token t+1's CPU work with token t's GPU work.
  - 24 dependent matvecs run at 77 GB/s vs 114 GB/s independently, a 32.7% serialization loss.
  - 2 ms of a 6.04 ms token was unexplained by summed op times (LFM2.5-8B-A1B).

  — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479) §2.3.5
- Fixed per-token overhead matters most for small models. BaseRT's decode advantage over llama.cpp shrinks from 1.56× (Qwen3-0.6B) to 1.04× (Qwen3-30B-A3B) on M4 Pro [measured] — [arXiv 2607.00501](https://arxiv.org/abs/2607.00501)

### Inferences
**Roofline for M4 Pro 20c [derived; FLOPs peak is an estimate]**
- PP512 for F16 Llama-2 7B is 464 t/s × ~13.4 GFLOP/token ≈ **6.2 TFLOPS achieved**. Peak FP16 is likely ~8–9 TFLOPS (20 cores × 128 ALUs × 2 FLOP × ~1.6–1.8 GHz; clock not verified this session), so prefill sits at ~70–75% of peak.
- Ridge point ≈ 8.5 TFLOPS / 273 GB/s ≈ **31 FLOP/byte**.

**Arithmetic intensity for a weight GEMM with M tokens [derived]**
- Intensity ≈ 2M / (bytes per weight).
  - fp16 (2 B/weight): intensity = M, so compute-bound above M ≈ 30.
  - 4.5-bit (0.5625 B/weight): intensity ≈ 3.6·M, so compute-bound above **M ≈ 8–9**.
- Consequence: verifying 4–8 speculative tokens, or batching 4–8 requests, costs nearly the same weight traffic as 1 token. This is the theoretical headroom for speculative decoding and batching on a 4-bit model.
- Dequant ALU cost lowers the effective ridge, which is consistent with Marlin's reuse only paying at M > 1 ([arXiv 2610.31479](https://arxiv.org/abs/2610.31479)).

**Decode ceilings on M4 Pro at 273 GB/s [derived; realistic = 70–85% of ceiling]**

| Model | Active bytes per token | Ceiling (tok/s) | Realistic (tok/s) | Notes |
|---|---|---|---|---|
| Llama 3 8B Q4_K_M | ~4.9 GB | ~56 | ~40–45 | — |
| Qwen 2.5/3 32B Q4_K_M | ~19.5 GB | ~14 | ~10–12 | Q4_0 / 4-bit MLX ~18 GB gives a ~15 ceiling |
| Qwen3-30B-A3B 4-bit | ~1.9 GB (~3.3B active) | ~140 | 80–84 measured, ~58% | Expert routing, attention, and the LM head add overhead |
| gpt-oss-20b MXFP4 | ~2 GB (3.6B active) | — | ~54 @16K measured | — |
| gpt-oss-120b MXFP4 | ~5.1B active; ~61–63 GB total | — | — | Does not fit usefully in 64 GB (see Gaps) |

**KV-cache cost**
- Llama-3-8B style GQA: 32 layers × 8 KV heads × 128 × 2 (K, V) × 2 B = 128 KiB per token in fp16, about 4 GiB at 32k context [derived].
- At 32k context the KV read per token is about the size of the Q4 weights, so decode roughly halves. q8 KV cuts this by about 2×; q4 by about 4×.

**MoE on a 64 GB Mac**
- Memory capacity holds all experts, while bandwidth only pays for the active ones. MoE models in the 30–35B-total / 3B-active class are the sweet spot: around 80 tok/s decode vs about 11 tok/s for a dense 32B.

**Batching**
- At batch 1 the GPU's compute sits mostly idle. Aggregate throughput scales roughly linearly with concurrency until M ≈ 8 (4-bit). vllm-mlx's 4.3× at 16 concurrent requests is consistent with this.

### Gaps
- No authoritative published FP16/BF16 TFLOPS figure or GPU clock for M4 Pro was found this session (the estimate above is unverified).
- No measured "achieved % of peak bandwidth" figure for MLX or llama.cpp on M4 Pro from a primary source; the % above are derived.
- The default macOS GPU wired-memory cap on 64 GB machines is not confirmed this session; it is commonly said to be ~75% of RAM. Whether gpt-oss-120b (~61–63 GB) can run on 64 GB with `iogpu.wired_limit_mb` raised close to 64 GB is not verified. It is likely impractical, with almost no RAM left for the OS and KV cache.

---

## 4. Concrete optimization techniques and measured impact on Apple Silicon (prioritized for a custom engine on M4 Pro)

### Takeaway
Ranked by expected payoff for single-user batch-1 work on M4 Pro:
1. 4-bit weights with dequant fused into the GEMV
2. Bandwidth-efficient GEMV kernels plus near-zero per-token CPU/sync overhead (whole-graph encoding, CPU/GPU overlap, preallocated buffers)
3. Flash/split-KV decode attention and KV quantization for long context
4. Prefix/prompt caching
5. MoE-specific kernels (gather, fused top-k)
6. Selective speculative decoding, which pays mainly for larger or unquantized targets and repetitive text

Generic elementwise fusion, megakernels, QuaRot-style rotations, and ANE/CPU co-execution give small or negative returns on this hardware. Batch-1 gains measured on NVIDIA often do not transfer.

### Cited Findings (in priority order)

**1. Weight quantization with dequant in registers (biggest lever)**
- Q8_0 → Q4_0 speeds decode 1.64× on M2 Pro and 1.20× on M3 Ultra.
- Materializing dequantized weights "throws away the entire point"; reconstruct in registers instead (Marlin principle).
- MLX native affine/DWQ 4-bit gives the best speed/quality trade-off.
- QuaRot rotations cost 2.4–4.2% of decode with no speed benefit without low-precision activation kernels.

— [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)

- Core ML int4 (block 32) doubled Llama-3.1-8B decode, 16.26 → 33.67 tok/s, and cut TTFT 128 → 52 ms [measured] — [Apple](https://machinelearning.apple.com/research/core-ml-on-device-llama)
- 2-bit QAT is viable for a purpose-trained 3B model (Apple Foundation Model) — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575)

**2. Low per-token overhead and kernel efficiency**
- MLX already hides CPU encode (0.54 ms per token) by overlapping token t+1's encode with token t's GPU work. The remaining loss is dependency serialization, which costs 32.7% of matvec bandwidth when matvecs are dependent — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- llama.cpp techniques, all in source:
  - 1–2 parallel encoding threads
  - Concurrent dispatch with range-based barriers
  - Graph reordering
  - Residency sets kept wired

  — [ggml-metal-context.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-context.m), [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)
- BaseRT's zero-allocation loop, CPU–GPU sync amortized over several tokens, per-chip tuning, and fused ops give a decode lead on M4 Pro. Over llama.cpp: 1.56× (Qwen3-0.6B), 1.28× (Llama-3.2-1B), 1.15× (3B), 1.04× (Qwen3-30B-A3B). Over MLX: 1.35×, 1.15×, 1.05×, and 1.01× on the same models [measured, vendor paper] — [arXiv 2607.00501](https://arxiv.org/abs/2607.00501)

**3. Attention for long context: flash/split-KV and KV quantization**
- A FlashInfer-style split-KV decode attention port gave up to **1.20× decode on M4 Pro**, rising with context (Qwen3-4B: 1.02× → 1.08× → 1.15× → 1.20×). It gave 0.97× on M2, where the extra dispatch and partials traffic did not pay back [measured] — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- FA added +21% F16 TG on M2 Ultra in llama.cpp (41.02 → 49.55 at `c1d0e7a`+FA; this also mixes in a build change) — [#4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- llama.cpp ships FA kernels for q4_0/q4_1/q5_0/q5_1/q8_0 KV caches, so quantized-KV attention exists on Metal — [kernels](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal/kernels)
- KV-cache pruning (SnapKV, ExpectedAttention, and others) at 4k context with 256 output tokens [measured, quality not evaluated]:
  - **1.72× on M4 Pro** with ExpectedAttention; 1.36–1.50× with the other methods.
  - About 0.98× on RTX 4090.
  - FastKVzip was always slower (0.28–0.96×).
  - At 512 context the best Apple gain was 1.24×, and peak memory could *increase* by up to 155%.

  — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- Architectural KV sharing (37.5% fewer layers producing KV) cuts KV memory and TTFT by 37.5% — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575)

**4. Prompt/prefix caching**
- A persistent conversational cache in MLX-LM cut wall time by 33%, 42%, and 57% at turns 3, 5, and 9 (Qwen3-4B), from avoided prefill [measured] — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- vllm-metal supports prefix caching, including GDN recurrent state for hybrid models — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)

**5. MoE-specific kernels**
- A gather-based MoE kernel at batch 1 gave **+3–12% on every machine** tested, 1.051–1.104× on M4 Pro, and 1.25× at M=8 — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- llama.cpp fuses top-k MoE routing and the expert reduction — [ggml-metal-fusion.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-fusion.h)
- BaseRT's MoE prefill at pp128 is 1.59–1.81× faster than llama.cpp and 1.42–1.78× faster than MLX, converging by pp2048 — [arXiv 2607.00501](https://arxiv.org/abs/2607.00501)

**6. Speculative decoding (selective; often negative at batch 1 on small quantized targets)**
- In the principal test (Qwen3-4B 4-bit, M2 Pro and M3 Ultra, long-prompt chat), **no method gave a net E2E speedup**: Classic, n-gram, cascade, EAGLE-3, DFlash, and DDTree all failed.
- Positive cases [measured]:
  - Classic draft on M4 Pro Qwen3-4B-4bit: +10% decode but −5% E2E.
  - N-gram on M3 Max: +13% decode, +2% E2E.
  - DDTree on M3 Pro BF16: +40.4% decode, +15.5% E2E.
- Negative cases:
  - EAGLE-3 on M4 Pro: −16% decode.
  - DFlash on M4 Pro: −15% E2E.
- Classic draft also cut prefill throughput by 29.6%.
- Gains rise with target-step cost (bigger or unquantized targets) and with predictable text (math, code).

— [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)

- vllm-metal with Gemma 4 MTP: +20% output tok/s at concurrency 1 — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)

**7. Continuous batching / paged KV (multi-request only)**
- vllm-mlx: 4.3× aggregate throughput at 16 concurrent requests [self-reported] — [arXiv 2601.19139](https://arxiv.org/abs/2601.19139)
- vllm-metal's varlen packing avoids padding blow-ups: mlx_lm wall time rose 143% on a mixed-length batch, vllm-metal's fell 6% — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)
- Marlin-style weight reuse pays only at M > 1 — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)

**8. Prefill-specific techniques**
- simdgroup_matrix tiled GEMM, chunked prefill with a bounded chunk, and FA with online softmax — [BaseRT](https://arxiv.org/abs/2607.00501)
- Simple quant formats (Q4_0, MXFP4) prefill faster than K-quants on Apple — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- llama.cpp GGUF beat MLX on prefill by about 2× on M2 Pro (1,180 vs ~550 tok/s) — same source.

**9. Memory residency / wired memory**
- `sudo sysctl iogpu.wired_limit_mb=N` (N larger than the model size and smaller than RAM) speeds up models that are close to the memory limit — [mlx-lm README](https://github.com/ml-explore/mlx-lm/blob/main/README.md)
- llama.cpp keeps residency sets alive with a heartbeat thread (default 3 min) and uses zero-copy `newBufferWithBytesNoCopy` over mmap — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m)

**10. Low or negative value on this hardware**
- Small elementwise fusion: fused add+RMSNorm gave only +1 to +1.8% E2E.
- A megakernel (11 fused MoE ops in one persistent dispatch) measured **1.8× slower** (174 vs 96 µs), for three reasons:
  - public Metal has no cooperative launch, grid barrier, or occupancy query;
  - without a proper grid barrier, plain stores were 96% stale across threadgroups;
  - removing every kernel boundary in the block recovers only ~15 µs.
- Public Metal also lacks async copy (no `cp.async` equivalent).

— [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)

- ANE co-execution:
  - Core ML may silently route to the GPU (macOS 26.3).
  - The private-API ANE achieved 268 tok/s prefill for a 0.8B model with batched dispatch, far below the GPU for real models.
  - Hybrid ANE-prefill/GPU-decode lost 11–16% decode from cache-bridge overhead in one configuration.

  — [AtomGradient paper](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)

### Inferences
**Recommended build order for an M4 Pro engine:**
1. Fused-dequant 4-bit GEMV (group 32–64, simple affine or Q4_0/MXFP4-like layout) hitting ≥ 80% of 273 GB/s.
2. Whole-token graph pre-encoded or replayed with preallocated buffers, CPU encode overlapped with GPU execution, and no per-token `waitUntilCompleted` except for sampling. Consider GPU-side argmax/sampling to avoid readback.
3. A split-KV flash-decode kernel plus a q8 KV option.
4. Persistent prefix cache with disk spill.
5. MoE gather and fused routing.
6. simdgroup_matrix prefill GEMM with chunked prefill.
7. Optional: prompt-lookup/n-gram drafting (cheap, no extra model), and draft/MTP only for dense ≥ 14B targets where one step costs ≥ 50 ms.

**Speculative decoding fits the M4 Pro's dense 32B case better than the papers' 4B targets.** A ~11 tok/s target has expensive steps, and the 4-bit ridge (~8 tokens) means verifying K ≤ 8 tokens is nearly free in bandwidth terms. This is a hypothesis to measure, not a cited result.

**Don't spend early effort on:** megakernels, ANE offload, CPU+GPU split decode (the CPU shares the same DRAM bandwidth), or QuaRot.

### Gaps
- No measured M4 Pro numbers found for KV-cache quantization's decode speedup (only kernel availability).
- No measured Medusa results on Apple Silicon were found.
- No source quantified the benefit of GPU-side sampling or argmax in isolation.
- Concurrent CPU (AMX/SME) plus GPU execution for LLMs: no measured study found this session.

---

## 5. Benchmark numbers: M4 Pro vs M4 Max / M3 Ultra, MLX vs llama.cpp

### Takeaway
On M4 Pro the measured numbers are consistent with the bandwidth model. Dense 7–8B Q4 runs at about 50 tok/s decode and 440 tok/s prefill (llama.cpp 2023 build). MoE 30B-A3B runs at about 80–84 tok/s decode across llama.cpp, MLX, and BaseRT. gpt-oss-20b runs at about 54 tok/s at 16K context. Dense 32B Q4 should be around 10–12 tok/s.

MLX vs llama.cpp is close for decode on large models; MLX is slightly ahead on dense decode and llama.cpp often ahead on prefill. Claims of 3× MLX advantages are outliers.

### Cited Findings
**llama.cpp #4167** (Llama-2 7B, build 8e672ef) [measured] — [#4167](https://github.com/ggml-org/llama.cpp/discussions/4167)

| Chip | Q4_0 PP512 | Q4_0 TG128 | F16 TG128 |
|---|---|---|---|
| M4 Pro 20c | 440 | 50.7 | 17.2 |
| M4 Max 40c | 886 | 83.1 | 31.6 |
| M3 Ultra 80c | 1471 | 92.1 | 39.8 |

**BaseRT paper, M4 Pro 16-core / 24 GB, tg128, Q4** [measured; decode is bandwidth-bound, so a 20c should match] — [arXiv 2607.00501](https://arxiv.org/abs/2607.00501)

| Model | llama.cpp (b9630) | MLX 0.31.2 | BaseRT |
|---|---|---|---|
| Qwen3-0.6B | 297.4 | 343.6 | 464.5 |
| Llama-3.2-1B | 230.4 | 257.8 | 295.4 |
| Llama-3.2-3B | 102.4 | 112.1 | 117.3 |
| Gemma-4-26B-A4B | 58.0 | 69.3 | 62.2 |
| Qwen3-30B-A3B | 80.7 | 83.1 | 84.1 |

- Qwen3-30B-A3B prefill at pp2048: BaseRT 721 tok/s, MLX 720 tok/s.

**Other M4 Pro numbers**
- Qwen3-Coder-30B-A3B 4-bit, MLX (oMLX), M4 Pro 64 GB: ~59–76 tok/s decode, ~734 tok/s prefill at 1k context [anecdotal, community log] — [oMLX community benchmarks](https://omlx.ai/benchmarks?page=5864)
- A Reddit report (via a secondary source) claims 43 (llama.cpp) vs 130 (MLX) tok/s for the same model on M4 Pro 64 GB [anecdotal]. 130 tok/s would be ~90% of the bandwidth ceiling and conflicts with BaseRT's measured 80–84, so treat it as unreliable — summarized in search results; original not verified.
- gpt-oss-20B 4-bit on M4 Pro [anecdotal]:
  - ~54 tok/s decode and 713 tok/s prefill at 16K; ~33 tok/s at 64K — [Hardware Corner](https://www.hardware-corner.net/is-mac-mini-m4-pro-good-for-llms/)
  - 8-bit on M4 Pro 48 GB: ~44–49 tok/s via oMLX [anecdotal] — [oMLX](https://omlx.ai/benchmarks?page=6533)
- Qwen3-32B on M4 Pro: 12–17 tok/s llama.cpp, 15–22 MLX [anecdotal, unclear method]. These **exceed** the ~14 tok/s 4.9-bpw bandwidth ceiling, so they are suspect unless a lower-bit quant was used — [SpecPicks](https://specpicks.com/reviews/run-qwen-3-32b-on-m4)

**Larger chips, gpt-oss-120b MXFP4** [anecdotal, community logs, oMLX]
- M4 Max 40c 128 GB: ~650–732 tok/s prefill, ~75 tok/s decode at 1k, peak memory ~60 GB.
- M3 Ultra 80c: ~584–631 prefill, ~65–74.5 decode.
- A community claim puts llama.cpp/Ollama on M4 Max at ~40–50 tok/s [anecdotal, unverified].

— [oMLX benchmark yjj3dzhc](https://omlx.ai/benchmarks/yjj3dzhc), [ohfrl8hz](https://omlx.ai/benchmarks/ohfrl8hz)

**Other comparisons**
- LFM2.5-8B-A1B: M3 Ultra GGUF-Q4_0 decode 232.2 tok/s vs MLX-W4-Affine 247.0 tok/s; RTX 4090 Q4_0 469.1 tok/s. M2 Pro 129.3 (GGUF) vs 119.9 (MLX DWQ) [measured] — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- Ollama on M4 Pro: MLX backend vs old llama.cpp backend gives 1.5× prefill and 2–3× decode for Qwen [anecdotal / press] — [Tom's Hardware](https://www.tomshardware.com/desktops/exploring-apple-silicons-local-ai-performance-with-the-mac-studio-and-m4-max-m4-max-beats-gb10-and-strix-halo-in-decode-throughput-but-memory-bandwidth-isnt-everything/2)

### Inferences
- M4 Pro → M4 Max 40c (2× bandwidth) gives about 1.64× Q4 decode. M4 Pro → M3 Ultra (2.9× bandwidth) gives only about 1.8× on 7B Q4, showing that big chips are overhead-bound on small models [derived from #4167].
- For gpt-oss-120b-class models, M4 Max and M3 Ultra decode about equally (~65–75 tok/s). The only real reason to choose the M3 Ultra over the M4 Max is capacity.
- MLX vs llama.cpp on M4 Pro:
  - Decode on large MoE is within ±5% (80.7 vs 83.1).
  - On small dense models MLX leads by 10–15%.
  - On a Gemma-4 MoE, MLX leads by 19%.
  - On prefill, llama.cpp often leads, especially on older chips.
- A custom engine's realistic decode upside over these is about 5–15% on 3B+ models and 30–55% on sub-1B models (BaseRT data).

### Gaps
- No primary, same-build MLX vs llama.cpp benchmark on M4 Pro **20-core 64 GB** for Llama 3 8B, Qwen3-32B dense, or gpt-oss-20b was found; numbers above mix chips, builds, and context lengths.
- No reliable gpt-oss-120b result on a 64 GB M4 Pro (it likely doesn't fit).
- No primary benchmark of llama.cpp (current build, FA on) vs MLX for Qwen 2.5 32B on M4 Pro.
- SiliconBench (NeurIPS 2026 poster) evaluates ten Apple Silicon stacks at concurrency 1, 8, and 16, but its tables were not retrieved — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/139658)

---

## 6. (Added scope) Heterogeneous CPU+GPU+ANE execution (disaggregated prefill/decode on one SoC), CPU vs GPU bandwidth and contention, and KV-cache designs on Apple Silicon

### Takeaway
Prior art on phones splits the work by phase: the NPU does compute-bound prefill, and the CPU or GPU does bandwidth-bound decode (llm.npu, PowerInfer-2, HeteroLLM). The same split is feasible on Apple Silicon, but only through the **private** ANE API. Measured gains there are about **power and overlap across requests, not single-request latency**.

The hard constraint for an M4 Pro engine is that CPU, GPU, and ANE all share one ~273 GB/s DRAM interface. The CPU alone sustains about 200 GB/s and the GPU about 200+ GB/s. Running them concurrently measurably slows the GPU (16% under a matched CPU stream). So **parallel decode on two units cannot add bandwidth**. The useful pattern is overlapping compute-bound prefill (ANE or GPU) of request k+1 with bandwidth-bound decode of request k, or running a cheap drafter on a different unit.

For the KV cache, the existing designs on Metal are:
- llama.cpp: a unified cell-based buffer with multi-sequence cells, quantized K/V types, KV shifting for reuse, and slot save/restore.
- mistral.rs and vllm-metal: paged block pools with native Metal paged kernels, FP8 KV, and block-level prefix sharing.

### Cited Findings
**Mobile and SoC prior art (non-Apple)**
- **llm.npu** (Xu et al., arXiv 2407.05858, Jul 2024, rev. Dec 2024) [measured, vs unnamed baselines]:
  - Techniques: chunked prompts with fixed shapes (NPUs need static graphs), "shadow outlier execution" (outlier activations run on CPU/GPU in parallel so the NPU can stay in INT8), and out-of-order subgraph scheduling across CPU/GPU and NPU by hardware affinity and accuracy sensitivity.
  - Results: 22.4× average prefill speedup and 30.7× energy savings, and the first >1,000 tok/s prefill for a billion-parameter model.
  - The abstract focuses on prefill; where decode runs is not stated on the abstract page.

  — [arXiv 2407.05858](https://arxiv.org/abs/2407.05858)
- **PowerInfer-2** (SJTU IPADS, arXiv 2406.06282) [measured, self-reported]:
  - Neuron-cluster granularity. The NPU runs dense prefill (all neurons in one big cluster). Decode uses CPU cores only, with hot/cold neuron clusters exploiting activation sparsity, plus a pipeline overlapping flash I/O and compute through a segmented neuron cache.
  - Up to 27.8× over prior frameworks; a 47B model at 11.68 tok/s on a phone.
  - Arxiv v2's text conflicts on the prefill placement; v3 says the NPU does prefill.

  — [arXiv 2406.06282v3](https://arxiv.org/html/2406.06282v3)
- **HeteroLLM / HeteroInfer** (arXiv 2501.14794, Jan 2025; v2 retitled "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference"; Snapdragon 8 Gen 3) [measured, self-reported]:
  - The NPU is the main compute unit and the GPU a secondary one that raises the NPU's performance floor. The CPU is only the control plane.
  - Uses different tensor-partition strategies for prefill and decode, chosen by a profiler-driven solver.
  - GPU–NPU sync costs about 400 µs, comparable to one kernel, so they built a fast sync based on predictable kernel wait times.
  - Results:
    - v2: 1.34–6.02× end-to-end over GPU-only and NPU-only engines.
    - Prefill: up to 3.69× over the NPU baseline and 8.68× over MNN-GPU.
    - Decode: **+23.4% tokens over GPU-only** (bandwidth-bound, so the gain is modest).

  — [arXiv 2501.14794v2](https://arxiv.org/html/2501.14794v2)

**Apple Silicon ANE+GPU disaggregation**
- **AtomGradient "Disaggregated LLM Inference on Apple Silicon"** (2026, M2 Ultra, Qwen3.5 0.8B–9B) [measured]. Five pipelines: MLX GPU, Core ML prefill + MLX decode, private-API ANE sequential, ANE-seq prefill + MLX decode, and ANE batch prefill + MLX decode.
  - Core ML `compute_units=ALL` ran on the GPU, not the ANE (macOS 26.3).
  - Private-API sequential ANE costs ~42 ms per token.
  - Decode scales with bandwidth.
  - ANE prefill cuts GPU power from 62.05 W to 0.22 W.
  - The hybrid lost 11–16% decode in one configuration from the mixed-precision cache bridge.

  — [paper](https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf)
- **AtomGradient "ANE Batch Prefill … Concurrent ANE Prefill and GPU Decode"** (Mar 2026, M2 Ultra) [measured]:
  - Up to 32 tokens per ANE dispatch (fused matrix–vector kernels compiled at load).
  - Qwen3.5-0.8B prefill: ANE 268 tok/s vs 23.7 sequential; ANE beats the GPU only on very short prompts (149 vs 138 tok/s at 13 tokens; at 28 tokens the GPU wins, 290 vs 237).
  - Hybrid decode is 39–54 tok/s vs ~61–64 pure MLX and ~18 pure ANE. State transfer costs under 30 ms.
  - Single-request hybrid wall time is **worse** than pure MLX (6,050 vs 3,679 ms).
  - The claimed value is multi-request pipelining (ANE prefills request k+1 while the GPU decodes request k), at ~1.8 W for ANE prefill vs ~62 W for GPU prefill.
  - The authors state "**No concurrent measurement**: … true concurrent execution was not yet benchmarked end-to-end."

  — [paper](https://atomgradient.github.io/hybird-batch-prefill-on-ane/paper.pdf)
- Orion: ANE weights are baked in at compile time, recompile takes 4.2 s (494 ms with weight patching), and there are 20 documented restrictions. Large or dynamic LLM graphs fit poorly — [arXiv 2603.06728](https://arxiv.org/abs/2603.06728)
- ANEMLL chunks models to fit ANE limits and moves argmax into the model "to reduce ANE-to-host data transfer". This confirms that the ANE↔host boundary is a real cost — [Anemll README](https://github.com/Anemll/Anemll)
- uzu reportedly mixes GPU kernels with MPSGraph, which may reach the ANE; the effect is unverified — [arXiv 2607.00501](https://arxiv.org/abs/2607.00501)

**CPU vs GPU achievable bandwidth and contention**
- Base M1–M4 chips, STREAM [measured] (Hübner et al., KTH, arXiv 2502.05317, Feb 2025):
  - CPU and GPU each reach roughly the same sustained bandwidth, about 85% of the theoretical peak.

  | Chip | CPU (GB/s) | GPU (GB/s) | Theoretical (GB/s) |
  |---|---|---|---|
  | M1 | 59 | 60 | 67 |
  | M2 | 78 | 91 | 100 |
  | M3 | 92 | 92 | 100 |
  | M4 | 103 | 100 | 120 |

  - Measured GPU FP32 (MPS): 2.9 TFLOPS on M4 (10-core GPU).

  — [arXiv 2502.05317](https://arxiv.org/html/2502.05317v1)
- **M4 Pro** (14-core CPU, 20-core GPU, 273 GB/s), Alpay & Başaran, arXiv 2606.27098, Jun 2026 [measured]:
  - **CPU-only, 10 threads:** STREAM triad 204.2 GB/s, BabelStream 199.4 GB/s (~73–75% of 273).
  - **GPU-only:** a Metal linear copy drives the memory-controller histogram to ~200 GB/s.
  - **Concurrent CPU+GPU:** paired median **16% GPU slowdown** under a matched user-interactive 10-thread CPU stream.
    - With background QoS the GPU is statistically unaffected, but the CPU stream collapses from **189.8 to 15.6 GB/s**, because background QoS moves the threads to E-cores and throttles them from 4.3 to 1.1 GHz.
  - **Sequential handoff:** after a GPU kernel touches 64 MiB or 512 MiB, the first CPU pass over a 16 MiB working set is 31% or 63% slower. This comes from residual system-level-cache displacement; a second pass recovers.

  — [arXiv 2606.27098](https://arxiv.org/abs/2606.27098)
- metal-benchmarks: "On bandwidth bound tasks, the GPU has no advantage over the CPU" at the core I/O level — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)

**KV-cache designs on Apple Silicon engines**
- **llama.cpp** (source at commit 19bc41d):
  - One unified KV buffer shared across all sequences (`-kvu/--kv-unified`). The cell metadata (`llama-kv-cells.h`) lets one KV cell "be part of multiple sequences at the same time", so a shared prefix is stored once.
  - K/V cache types (`-ctk/-ctv`): f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1.
  - Metal has FA kernels for each quantized KV type (`fa_q8_0.metal`, `fa_vec_q4_0.metal`, …).
  - `--cache-reuse N` reuses cached chunks "via KV shifting" (RoPE re-rotation) for non-prefix matches.
  - `--slot-save-path` persists KV slots to disk.
  - `--swa-full` and the iSWA cache handle sliding-window models; separate memory classes exist for recurrent and hybrid models (`llama-memory-recurrent`, `llama-memory-hybrid`) and for DeepSeek sparse attention (`llama-kv-cache-dsa`).

  — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp) (L281–291, L1673, L1708, L2421, L3631, L3663); [src/llama-kv-cells.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cells.h); [src/](https://github.com/ggml-org/llama.cpp/tree/master/src)
- **mistral.rs**:
  - Paged attention is on by default on Metal (v0.9.5). The KV cache is a pool of blocks that requests take and return; requests with the same prompt share blocks.
  - Memory: default headroom on unified memory is 25% (max 16 GiB), capped at Metal's `recommendedMaxWorkingSetSize`. The docs say "On a 64 GiB Mac … mistral.rs plans to use at most 48 GiB".
  - `--pa-cache-type f8e4m3` gives an FP8 KV cache at half the memory of BF16, using per-layer fixed scales.
  - On a full cache it preempts a request and recomputes it later.
  - Metal layout is HND `[blocks, kv_heads, block_size, head_dim]`. Native Metal kernels: `paged_decode_hnd.metal`, `paged_prefill.metal`, `reshape_and_cache.metal`, `copy_blocks.metal`, `gather_kv_cache.metal`, `float8.metal`, `kv_scale_update.metal`.
  - Decode chooses between FlashInfer, a standard paged kernel, or gather + SDPA.

  — [docs paged-attention.mdx](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/guides/perf/paged-attention.mdx); [docs inference-runtime.md](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/developer/inference-runtime.md); [mistralrs-paged-attn/src/metal/kernels](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs-paged-attn/src/metal/kernels)
- **vllm-metal**:
  - A fixed KV page pool sized by a warmup pass under `--gpu-memory-utilization`, with per-request block tables.
  - A packed varlen paged Metal attention kernel (ported from vLLM's Triton kernel).
  - Prefix caching, with an experimental mode that also snapshots GDN recurrent state for hybrid models; it cannot yet be combined with speculative decoding.
  - Paged attention is opt-in in the community plugin (`VLLM_METAL_USE_PAGED_ATTENTION=1`).

  — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0); [a2s-ai/A2S_vllm-metal](https://github.com/a2s-ai/A2S_vllm-metal)
- KV-size math, prefix-reuse savings (−33/−42/−57% at turns 3/5/9), and pruning trade-offs on Apple (up to 1.72× on M4 Pro, peak-memory increases at short context) — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)
- Architectural KV reduction: Apple's KV sharing across 37.5% of layers — [arXiv 2507.13575](https://arxiv.org/abs/2507.13575). Hybrid models (Mamba2 or short-conv) add only 8–16 KiB per token — [arXiv 2610.31479](https://arxiv.org/abs/2610.31479)

### Inferences
**What heterogeneous execution can and can't buy on M4 Pro [derived]**
- GPU decode already uses ~70–85% of 273 GB/s, about 190–230 GB/s. The CPU alone can also pull ~200 GB/s. Running CPU decode next to GPU decode just splits the same pipe and slows the GPU by about 16% or more.
- So for a *single stream*, splitting decode across CPU and GPU (or ANE) has no bandwidth upside.
- HeteroLLM's +23% decode on Snapdragon likely reflects that the GPU alone could not saturate DRAM on that SoC. Apple's GPU nearly can.

**Where co-execution can pay**
- **(a) Multi-request pipelining.** Overlap compute-bound prefill of request k+1 with decode of request k. On the GPU alone, continuous batching or chunked prefill already gets most of this, because prefill tokens ride along with decode GEMVs at low marginal bandwidth cost. Putting that prefill on the ANE frees GPU ALUs and saves power, but needs the private API (fragile across macOS updates), fp16 conv-style graphs with static shapes, and a KV handoff. AtomGradient measured the handoff under 30 ms for small models.
- **(b) Drafting on another unit.** A small draft model or n-gram/prompt lookup on the CPU or ANE while the GPU verifies. The drafter's weight reads compete for the same bandwidth, so the draft must be tiny; n-gram lookup is effectively free.
- **(c) CPU-side work.** Tokenization, sampling, grammar constraints, and scheduling on CPU cores, overlapped with GPU execution.
- **(d) Power and thermals.** Mostly irrelevant for a mains-powered Mac mini, but the ~60 W → ~2 W prefill power drop (AtomGradient) can matter for sustained clocks.

**Scheduling hygiene, if mixing units**
- Run any concurrent CPU helpers at background QoS to protect GPU bandwidth, accepting that they run on E-cores. Note that there is no core pinning on macOS.
- Expect cache-displacement penalties right after large GPU phases.
- Use events or shared-event fences (`MTLSharedEvent`) rather than `waitUntilCompleted`. HeteroLLM's mobile sync cost (~400 µs) suggests cross-unit sync must be amortized over large chunks.

**KV cache design recommendations for a custom engine [derived from cited designs]**
1. Preallocate one KV arena sized from the wired budget: on 64 GB, with weights ~20–35 GB, leave ~12–20 GB for KV (cf. mistral.rs's 48 GiB plan on 64 GiB).
2. Use paged blocks of 16–64 tokens in a head-major (HND) layout so a decode kernel streams contiguous `head_dim` vectors per KV head.
3. Store K/V in q8 (per-block or per-head scales) with dequantization in the flash-decode kernel. q4 is an option for very long contexts.
4. Use split-KV flash decode for long contexts (measured +20% on M4 Pro).
5. Use a radix/prefix tree over blocks for prompt and prefix reuse across turns and agents, with optional disk spill (llama.cpp slot save; oMLX's SSD cache appears in the vllm-metal comparisons).
6. Keep separate fixed-size state for SWA and recurrent layers.

Paging matters mostly for concurrency. For a single user, a contiguous per-sequence buffer with a prefix cache gets ~all the benefit with simpler kernels.

### Gaps
- **No end-to-end measured concurrent ANE-prefill + GPU-decode benchmark on Apple Silicon was found.** AtomGradient explicitly did not measure true concurrency, and nothing was measured on M4 Pro.
- No measurement found of ANE DRAM bandwidth or of ANE-vs-GPU bandwidth contention on any M-series chip.
- No study found of CPU (AMX/SME on M4) plus GPU co-execution for LLM decode or prefill on a Mac. The STREAM-based contention numbers above are the closest proxy.
- llm.npu's target devices and decode placement are not stated on the abstract page; the full paper was not retrieved.
- No measured comparison found of paged vs contiguous KV decode speed on Metal at batch 1, or of q8 vs f16 KV decode speedup on M4 Pro.
