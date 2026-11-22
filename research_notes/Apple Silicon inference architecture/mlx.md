# Apple MLX and mlx-lm: architecture, internals and lessons for a custom Mac inference framework

Scope: MLX core (ml-explore/mlx) and mlx-lm, as of November 2026. Latest releases: **MLX v0.32.4 (2026-10-14)**, **mlx-lm v0.32.2 (2026-10-16)**. Source-code facts come from shallow clones of `main` taken 2026-11-21 (mlx commit `dc5ea6e`). Paths are cited as GitHub URLs. Release dates come from `gh api repos/ml-explore/{mlx,mlx-lm}/releases`.

Shorthand used below:
- MLX = https://github.com/ml-explore/mlx
- LM = https://github.com/ml-explore/mlx-lm
- REL(x) = https://github.com/ml-explore/mlx/releases/tag/x

---

## 1. Core design: unified-memory arrays, lazy graphs, eval/async_eval, streams, transforms, compile, bindings

### Takeaway
MLX is a NumPy/JAX-style array library. Arrays have no device. You pick the device per op through a stream, and CPU and GPU ops share the same buffers with no copies; the scheduler adds cross-stream dependencies. All ops are lazy and only record a graph. `mx.eval` and `mx.async_eval` materialize it. Function transforms (grad, vmap) act on that graph. `mx.compile` only simplifies the graph and fuses elementwise and broadcast chains into JIT-generated kernels. Matmul, attention and norms are not fused by compile; they are hand-written "fast" kernels. A C++ core underpins thin Python (nanobind), C (mlx-c) and Swift (mlx-swift, built on mlx-c) bindings.

### Cited Findings
**Unified memory and streams**
- Arrays are not tied to a device at creation. The device is chosen when an op runs (`stream=mx.cpu` / `mx.gpu`), and "arrays are not moved". Independent CPU and GPU ops on the same inputs can run in parallel. If a GPU op consumes the output of a CPU op, MLX inserts the dependency automatically. — [MLX docs: Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- Docs example (M1 Max): a matmul on the GPU plus a chain of 500 small `exp` ops on the CPU takes ~1.4 ms, versus ~2.8 ms with everything on the GPU. Small, overhead-bound ops are better placed on the CPU stream. — [MLX docs: Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- Every Metal buffer is allocated with `MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked`. Memory is shared with the CPU, and MLX tracks hazards itself rather than letting Metal do it. — [MLX `mlx/backend/metal/allocator.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- v0.23.0 added a "Faster synchronization `Fence` for synchronizing CPU-GPU". — [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0)
- v0.32.1 added zero-copy CPU import with `mx.array(host_buffer, copy=False)` on unified memory. v0.32.0 added Metal DLPack zero-copy sharing. — [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1), [REL(v0.32.0)](https://github.com/ml-explore/mlx/releases/tag/v0.32.0)
- Externally owned memory is wrapped with `newBuffer(ptr, size, ...)` (no copy) and deliberately not wired. The source comment reads: "Wiring a file-backed mapping would read the whole file and keep it in memory." — [MLX `allocator.cpp` `make_buffer`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- Recent stream APIs: `new_thread_unsafe_stream` (v0.32.0). mlx-lm generates on `mx.new_thread_local_stream(mx.default_device())`. — [REL(v0.32.0)](https://github.com/ml-explore/mlx/releases/tag/v0.32.0), [LM `mlx_lm/generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)

**Lazy evaluation**
- Ops record a compute graph and nothing runs until `eval()`. Laziness is what enables `grad`/`vmap` transforms and graph optimizations. Unused outputs are never computed, but building their graph still costs something. — [MLX docs: Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- Memory benefit: a model initialized in fp32 and then loaded with fp16 weights never allocates the fp32 copies, which roughly halves peak memory. — [MLX docs: Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- Guidance: each `eval` has fixed overhead and very large graphs also cost; "a few tens to many thousands of operations per evaluation" is reasonable. Implicit eval triggers are `print`, NumPy conversion or memoryview, saving, `.item()`, and scalar arrays in Python control flow. — [MLX docs: Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- v0.20.0 introduced "BFS graph optimizations" (cited in the release alongside "Over 120tks with Mistral 7B"). v0.21.0 added a "Bfs width limit to reduce memory consumption during eval" (env `MLX_BFS_MAX_WIDTH`) and "More buffer donation with no-ops to reduce memory use". v0.18.0 improved donation heuristics. — [REL(v0.20.0)](https://github.com/ml-explore/mlx/releases/tag/v0.20.0), [REL(v0.21.0)](https://github.com/ml-explore/mlx/releases/tag/v0.21.0), [REL(v0.18.0)](https://github.com/ml-explore/mlx/releases/tag/v0.18.0)
- `mx.async_eval(*arrays)` releases the GIL and schedules evaluation without blocking. Its docstring still says "experimental API". — [MLX `python/src/transforms.cpp`](https://github.com/ml-explore/mlx/blob/main/python/src/transforms.cpp)
- mlx-lm's decode loop pipelines one step ahead. It builds the graph for step n+1 and calls `mx.async_eval(next_y, next_logprobs)` before calling `y.item()` on step n. Python graph construction for the next token therefore overlaps with GPU execution of the current one. It also calls `mx.clear_cache()` every 256 tokens. — [LM `generate.py` `generate_step`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
- v0.18.0: "Make eval just wait if all arrays are scheduled". — [REL(v0.18.0)](https://github.com/ml-explore/mlx/releases/tag/v0.18.0)

**mx.compile and fusion**
- The compile pipeline is `compile_simplify` (graph simplification and CSE-like merging), then `compile_fuse`. `is_fusable()` returns true only for unary, binary, ternary and broadcast primitives. Fusion is capped by `max_compile_depth = 11` and `max_compile_arrays = 24` inputs. Modes are `disabled`, `no_simplify`, `no_fuse` and `enabled`, and the env var is `MLX_DISABLE_COMPILE`. — [MLX `mlx/compile.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/compile.cpp)
- Fused kernels are generated as source and JIT-compiled: Metal source on the GPU, C++ compiled to a shared library on the CPU (env `MLX_CPU_CACHE_DIR`, "Separate cpu compilation cache by versions" in v0.29.0). Recompilation happens when the stream changes (v0.22.0) or `shapeless` changes (v0.23.0). Scan ops can be shapeless-compiled since v0.32.3. — [REL(v0.29.0)](https://github.com/ml-explore/mlx/releases/tag/v0.29.0), [REL(v0.22.0)](https://github.com/ml-explore/mlx/releases/tag/v0.22.0), [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0), [REL(v0.32.3)](https://github.com/ml-explore/mlx/releases/tag/v0.32.3)
- Thread-safety of the compile cache was fixed late: "Use thread local storage for frontend compile cache" (v0.31.2), "Make mx.compile cache erasing thread safe" (v0.32.1), and a "cross-process race in the CPU compile cache" fix (v0.32.4). — [REL(v0.31.2)](https://github.com/ml-explore/mlx/releases/tag/v0.31.2), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1), [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)

**Bindings and export**
- MLX provides C++, C, Python and Swift APIs. — [WWDC25 session 315 "Get started with MLX for Apple silicon"](https://developer.apple.com/videos/play/wwdc2025/315/)
- v0.22.0: "Functions can be exported from Python and run in C++ and vice versa", via `mx.export_function`. Graphs can therefore be traced in Python and executed from a C++ host without Python. — [REL(v0.22.0)](https://github.com/ml-explore/mlx/releases/tag/v0.22.0)
- mlx-swift vendors mlx-c (target `Cmlx`, sources `mlx-c/mlx/c/*.cpp`) and now also has a CUDA build plugin. Swift therefore goes through the C API. — [mlx-swift `Package.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Package.swift)
- Ollama's MLX runner (Ollama v0.19.0, announced 2026-03-30 as a preview) uses MLX-C. Breaking changes between MLX-C 0.5.0 and 0.6.0 were linked to Ollama issues. — [yage.ai survey (secondary)](https://yage.ai/share/mlx-apple-silicon-en-20260331.html), [Ollama #14432](https://github.com/ollama/ollama/issues/14432), [Ollama #14611](https://github.com/ollama/ollama/issues/14611)
- A CUDA backend shipped on PyPI in v0.27.1 (2025-07-25). Its initial release notes say it "works well with mlx-lm" and uses CUDA graphs. NCCL distributed support arrived in v0.29.0. — [REL(v0.27.1)](https://github.com/ml-explore/mlx/releases/tag/v0.27.1), [REL(v0.29.0)](https://github.com/ml-explore/mlx/releases/tag/v0.29.0)

### Inferences
- Worth copying: (a) "device per op, not per array" on unified memory, (b) lazy graph plus a one-step-ahead async decode pipeline, which hides host overhead without CUDA-graph-style capture, and (c) buffer donation, where an op reuses its input's buffer when the refcount is 1.
- MLX's compile is weak as a fusion engine. It never fuses reductions or matmuls, so all the real LLM wins come from hand-written `mx.fast` kernels and quantized matmuls. A purpose-built LLM engine can skip a general graph compiler and directly hand-schedule a fixed per-layer kernel sequence, which is the llama.cpp-style approach.
- Rebuilding the graph every token is per-token Python and C++ overhead. MLX mitigates it with async_eval rather than graph capture/replay; a custom engine could record the decode-step command sequence once.

### Gaps
- No primary source measured per-token graph-construction overhead in microseconds on M4 Pro.
- The "Using Streams" docs page was not fetched, so the detailed stream semantics (events/fences across command queues) come from the release notes and source names only.

---

## 2. Metal backend internals: allocator/buffer cache, wired memory, command-buffer batching, JIT vs precompiled kernels, custom kernels, mx.fast

### Takeaway
The allocator is a size-bucketed buffer cache over `MTLBuffer`s, with a small-allocation `MTLHeap` and soft limits derived from `recommendedMaxWorkingSetSize`. Wiring is done with `MTLResidencySet`s (macOS 15+), up to `set_wired_limit`. Ops are encoded into a command buffer that is committed after N ops or M MB, where N and M are tuned per chip class. Kernels are precompiled into a `.metallib` by default, with an optional JIT build that is used for compiled/fused kernels, `metal_kernel`, NAX kernels in Swift and some GEMV. The `mx.fast` namespace holds the hand-tuned LLM kernels.

### Cited Findings
**Allocator**
- Allocator limits are set at construction: `block_limit_ = min(1.5 × max_recommended_working_set_size, 0.95 × memsize)`, `gc_limit_ = min(0.95 × max_rec, block_limit_)`, and the cache cap `max_pool_size_ = block_limit_`. — [MLX `allocator.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- `malloc` path: round sizes above one page up to a `vm_page_size` multiple, then try `buffer_cache_.reuse_from_cache(size)`. If `active+cache+size >= gc_limit_` or the resource count is near the Metal resource limit, release cached buffers first. Allocations under `small_size_` come from a shared `MTLHeap`; otherwise `device->newBuffer`. Each new non-heap buffer is inserted into the residency set. If cache memory exceeds `max_pool_size_`, the cache is trimmed. — [MLX `allocator.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- `free` returns the buffer to the cache if the cache is below `max_pool_size_`; otherwise it releases it and removes it from the residency set. `set_cache_limit` sets `max_pool_size_`, `set_memory_limit` sets `block_limit_`, and `clear_cache()` empties the pool. — [MLX `allocator.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- The memory API moved from `mx.metal.*` to top-level `mx.*` in v0.25.0 (`mx.set_cache_limit`, `mx.clear_cache`, `mx.get_peak_memory`). The `mx.metal.*` names now emit deprecation warnings. — [REL(v0.25.0)](https://github.com/ml-explore/mlx/releases/tag/v0.25.0), [MLX `python/src/metal.cpp`](https://github.com/ml-explore/mlx/blob/main/python/src/metal.cpp)

**Wired memory**
- "Wired buffer support for large models" landed in v0.20.0 (2024-11-05). v0.25.0 improved usage of the residency set. v0.32.1 split the monolithic `MTLResidencySet` into smaller sets. v0.32.4 added an env-based residency refresh interval (`MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`), and there is also `MLX_RESIDENCY_SET_MAX_PCT`. — [REL(v0.20.0)](https://github.com/ml-explore/mlx/releases/tag/v0.20.0), [REL(v0.25.0)](https://github.com/ml-explore/mlx/releases/tag/v0.25.0), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1), [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
- `mx.set_wired_limit(limit)` docs: "only useful on macOS 15.0 or higher". The default is 0. It may not exceed the system wired limit, which can be raised with `sudo sysctl iogpu.wired_limit_mb=<MB>`. Query `mx.device_info()["max_recommended_working_set_size"]`. — [MLX `python/src/memory.cpp`](https://github.com/ml-explore/mlx/blob/main/python/src/memory.cpp)
- mlx-lm wraps generation in a `wired_limit` context manager. It sets the wired limit to `max_recommended_working_set_size`, warns if the model is more than 90% of that, and synchronizes the generation streams before restoring the limit, because "the wired limit should not be changed during an async eval". The README advises raising `iogpu.wired_limit_mb` when models are large relative to RAM. — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [LM `utils.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py), [LM README "Large Models"](https://github.com/ml-explore/mlx-lm/blob/main/README.md)
- mlx-lm v0.29.0: "Fix slow batch generation in server by setting wired_limit". — [LM releases](https://github.com/ml-explore/mlx-lm/releases/tag/v0.29.0)

**Command-buffer batching**
- `CommandEncoder::needs_commit()` commits when the op count or the MB of bound buffers exceeds a per-architecture threshold (env overrides `MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`). Defaults are keyed off the last letter of the GPU arch string:

  | Arch suffix | Chip class | Max ops | Max MB |
  |---|---|---|---|
  | `p` | phone | 20 | 40 |
  | `g` | base/Pro, including M4 Pro | 40 | 40 |
  | `s` | Max | 50 | 50 |
  | `d` | Ultra | 50 | 50 |

  Residency sets are attached at commit time because "Metal locks a command buffer's residency at commit time". — [MLX `mlx/backend/metal/device.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp)
- `MLX_METAL_FAST_SYNCH` exists as an env var for faster CPU-GPU synchronization. A v0.31.0 fix addressed "fence synchronization across command buffers". — [MLX source env-var list (grep `MLX_METAL_FAST_SYNCH`)](https://github.com/ml-explore/mlx), [REL(v0.31.0)](https://github.com/ml-explore/mlx/releases/tag/v0.31.0)

**Kernel compilation**
- The CMake option `MLX_METAL_JIT` ("Use JIT compilation for Metal kernels") defaults OFF, so release builds ship a precompiled metallib. The `jit_kernels.cpp` and `nojit_kernels.cpp` sources reflect the two modes. The Metal version is chosen from `CMAKE_OSX_DEPLOYMENT_TARGET` (v0.21.0). — [MLX `CMakeLists.txt`](https://github.com/ml-explore/mlx/blob/main/CMakeLists.txt), [REL(v0.21.0)](https://github.com/ml-explore/mlx/releases/tag/v0.21.0)
- NAX (M5 Neural Accelerator) kernels got JIT support in v0.30.1 "so that they can be used in MLX Swift". GEMV became JIT-compilable in v0.32.0. On macOS 27, v0.32.1 requests MSL 4.1 from the runtime compiler. — [REL(v0.30.1)](https://github.com/ml-explore/mlx/releases/tag/v0.30.1), [REL(v0.32.0)](https://github.com/ml-explore/mlx/releases/tag/v0.32.0), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1)
- Metal kernel source tree includes `quantized.metal`, `quantized_nax.metal`, `fp_quantized(.metal|_nax.metal)`, `fp4.h`, `fp8.h`, `gemv.metal`, `rms_norm.metal`, `rope.metal`, `scaled_dot_product_attention.metal`, `sdpa_vector.h`, `sdpa_blocked.h`, `gated_delta_update(_nax).metal` and `steel/`, the tiled GEMM/attention template library. — [MLX `mlx/backend/metal/kernels/`](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels)

**Custom kernels and mx.fast**
- `mx.fast.metal_kernel` (custom Metal kernels from Python source strings, JIT-compiled and cached) landed in **v0.17.0** (2024-08-23). Optional headers were added in v0.18.0 along with overhead reductions. `mx.fast.cuda_kernel` came in v0.29.0. — [REL(v0.17.0)](https://github.com/ml-explore/mlx/releases/tag/v0.17.0), [docs: custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html), [REL(v0.18.0)](https://github.com/ml-explore/mlx/releases/tag/v0.18.0), [REL(v0.29.0)](https://github.com/ml-explore/mlx/releases/tag/v0.29.0)
- The current `mx.fast` namespace is `rms_norm`, `layer_norm`, `rope`, `scaled_dot_product_attention`, `cross_entropy` (fused on Metal in v0.32.3), `gated_delta_update` (Qwen3-Next/3.5-style linear attention), `metal_kernel` and `cuda_kernel`. — [MLX `python/src/fast.cpp`](https://github.com/ml-explore/mlx/blob/main/python/src/fast.cpp), [REL(v0.32.3)](https://github.com/ml-explore/mlx/releases/tag/v0.32.3)
- SDPA evolution:

  | Version | Change |
  |---|---|
  | v0.24.0 | Much faster fused attention with causal masking; small-batch query SDPA for speculative decoding; fused head-dim 128 |
  | v0.25.0 | Fused vector attention supports D=256 and additive masks |
  | v0.30.0 | SDPA with attention sinks (gpt-oss) |
  | v0.30.4 | "Much faster vector fused grouped-query attention for long context" (two-pass SDPA) |
  | v0.32.x | D=72/80/96/512 support, `force_fused`, `MLX_SDPA_BLOCKS`, GQA-8 decode reading each K/V byte once, SDPA VJP (v0.32.4) |

  — [REL(v0.24.0)](https://github.com/ml-explore/mlx/releases/tag/v0.24.0), [REL(v0.25.0)](https://github.com/ml-explore/mlx/releases/tag/v0.25.0), [REL(v0.30.0)](https://github.com/ml-explore/mlx/releases/tag/v0.30.0), [REL(v0.30.4)](https://github.com/ml-explore/mlx/releases/tag/v0.30.4), [REL(v0.32.2)](https://github.com/ml-explore/mlx/releases/tag/v0.32.2), [REL(v0.32.3)](https://github.com/ml-explore/mlx/releases/tag/v0.32.3)
- RoPE: array offsets in v0.22.0, batch offsets and no-copy batch rope in v0.30.0. RMSNorm without scale arrived in v0.24.0. — [REL(v0.22.0)](https://github.com/ml-explore/mlx/releases/tag/v0.22.0), [REL(v0.30.0)](https://github.com/ml-explore/mlx/releases/tag/v0.30.0), [REL(v0.24.0)](https://github.com/ml-explore/mlx/releases/tag/v0.24.0)

### Inferences
- The allocator design is simple and worth copying: page-rounded size buckets, a soft cache cap, a GC threshold at 95% of the recommended working set, and a heap for tiny buffers. For a single-model LLM server, a bump/arena allocator for activations plus fixed preallocated KV pages would avoid MLX's cache growth and the need for periodic `clear_cache()` calls.
- Committing every 40 ops / 40 MB on Pro-class chips is a measured trade-off between CPU-GPU overlap and per-commit overhead. Benchmark your own threshold rather than committing once per token.
- On a 64 GB M4 Pro the default recommended working set is below 64 GB. Big models need `iogpu.wired_limit_mb` raised, and an engine should do what mlx-lm does: wire model plus KV with residency sets.

### Gaps
- The exact `small_size_` and heap size constants were not read (they sit just past line 66 of `allocator.cpp`).
- The default `max_recommended_working_set_size` for a 64 GB M4 Pro was not found in a primary source; it is commonly reported as about 75% of RAM, but that is unverified here.

---

## 3. Quantization: formats, kernels, newer FP formats, learned quantization (DWQ/AWQ/GPTQ)

### Takeaway
MLX's native format is affine group-wise quantization with an fp16/bf16 scale and bias per group, packed little-end-first into uint32. Bits are 1 (Metal only), 2, 3, 4, 5, 6 and 8; group sizes are 32, 64 (default) and 128. Microscaling FP formats mxfp4 (v0.29.0), nvfp4 and mxfp8 (v0.30.3) are supported, as are "qq" ops that quantize activations too. Dedicated qmv (decode), qmm (prefill), split-K, gather_qmm (MoE) and NAX (M5) kernels exist. mlx-lm adds DWQ, AWQ, GPTQ and dynamic mixed-precision quantization.

### Cited Findings
- Quantization mode table from the `mx.quantize` docstring:

  | Mode | Group size | Bits | Scale | Bias |
  |---|---|---|---|---|
  | affine | 32, 64*, 128 | 1, 2, 3, 4*, 5, 6, 8 | same dtype as input | yes |
  | mxfp4 | 32 | 4 (E2M1) | E8M0 | no |
  | mxfp8 | 32 | 8 (E4M3) | E8M0 | no |
  | nvfp4 | 16 | 4 (E2M1) | E4M3 (+ optional fp32 `global_scale`) | no |

  \* = default. Affine: `s = (max−min)/(2^b−1)`, `q = round((w−min)/s)`, packed into uint32 from the low bits (8 elements per uint32 at 4-bit). — [MLX `python/src/ops.cpp` quantize docstring](https://github.com/ml-explore/mlx/blob/main/python/src/ops.cpp)
- `quantize` validates `bits < 1 || bits > 8 || bits == 7` as invalid. — [MLX `mlx/ops.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp)
- Landing timeline:

  | Version (date) | Feature |
  |---|---|
  | v0.17.0 (2024-08-23) | Fused GPU quantization ops |
  | v0.22.0 | Mixed quantization via `nn.quantize` |
  | v0.26.0 (2025-06-02) | 5-bit quantization |
  | v0.29.0 (2025-08-29) | mxfp4 (Metal, CPU) and a `mode` param |
  | v0.30.3 (2026-01-13) | nvfp4 and mxfp8 on Metal |
  | v0.31.0 | Tensor-scale nvfp4, `QQLinear` (quantized-activation × quantized-weight) |
  | v0.32.4 (2026-10-14) | 1-bit affine on Metal |

  — [REL(v0.17.0)](https://github.com/ml-explore/mlx/releases/tag/v0.17.0), [REL(v0.22.0)](https://github.com/ml-explore/mlx/releases/tag/v0.22.0), [REL(v0.26.0)](https://github.com/ml-explore/mlx/releases/tag/v0.26.0), [REL(v0.29.0)](https://github.com/ml-explore/mlx/releases/tag/v0.29.0), [REL(v0.30.3)](https://github.com/ml-explore/mlx/releases/tag/v0.30.3), [REL(v0.31.0)](https://github.com/ml-explore/mlx/releases/tag/v0.31.0), [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
- Kernel work:

  | Version | Change |
  |---|---|
  | v0.23.0 | Faster small-batch quantized matmuls, which speed up speculative decoding |
  | v0.25.0 | Tuned qmv dispatch for small batches |
  | v0.31.2 | Split-K quantized matmul for small M |
  | v0.32.0 | `qmv_wide` small-batch quantized matvec |
  | v0.32.1 | Large NVFP4 QMV tuned for M5 Max; dequantize in fp32 |
  | v0.32.4 | Split-K partials kept in fp32; fast qmv for N not divisible by 8; M1 qmm six-row tuning |

  MoE uses `gather_qmm`, including `gather_qmm_rhs` and NAX variants. — [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0), [REL(v0.25.0)](https://github.com/ml-explore/mlx/releases/tag/v0.25.0), [REL(v0.31.2)](https://github.com/ml-explore/mlx/releases/tag/v0.31.2), [REL(v0.32.0)](https://github.com/ml-explore/mlx/releases/tag/v0.32.0), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1), [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
- fp8 handling: F8_E4M3 safetensors loading came in v0.23.0. v0.30.3 exposed to/from fp8 in Python and stopped auto-converting fp8 on safetensors load. — [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0), [REL(v0.30.3)](https://github.com/ml-explore/mlx/releases/tag/v0.30.3)
- mlx-lm quantization methods: DWQ (distill from the unquantized teacher, tuning scales, biases and non-quantized params), AWQ, GPTQ and dynamic quantization (per-layer sensitivity leads to mixed precision); methods can be cascaded. DWQ "works best distilling to lower precision, anywhere from 2-bit to 4-bit". `--group-size 32` doubles the tunable parameters, and distilling to 6- or 8-bit "often doesn't work well". — [LM `mlx_lm/LEARNED_QUANTS.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md)
- Other mlx-lm milestones: mxfp4 support in mlx-lm v0.27.0 (2025-08-29, for gpt-oss), mxfp8/nvfp4 in v0.30.1, AWQ/GPTQ weight-transformation utilities (to import AutoAWQ/GPTQ checkpoints) in v0.30.4, DWQ for very large models in v0.28.4, global scale for nvfp4 MoEs in v0.32.0, and a Hadamard-rotated 2-bit Qwen3.5 ("prism") in v0.32.0. — [LM releases](https://github.com/ml-explore/mlx-lm/releases)
- Quality versus speed on a 64 GB M4 Max (mlx 0.29.2, mlx-lm 0.28.2, `-p 2048 -g 128`), Qwen3-4B-Instruct-2507:

  | Precision | MMLU-Pro | Prompt tok/s | Gen tok/s | Memory |
  |---|---|---|---|---|
  | bf16 | 64.05 | 1781 | 52.5 | 9.0 GB |
  | q8 | 63.85 | 1607 | 86.9 | — |
  | q6 | 63.53 | — | 104.7 | — |
  | q5 | 62.38 | — | 116.4 | — |
  | q4 | 60.72 | 1622 | 134.5 | 3.35 GB |

  Prefill is flat-to-slower when quantized, while decode scales with bytes per weight. — [LM `mlx_lm/BENCHMARKS.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md)

### Inferences
- Affine int4 with group size 64 and fp16 scale+bias costs about 4.5 bits per weight. The format is simple, and kernels can dequantize in-register with one FMA.
- MLX is converging on OCP/NVIDIA FP4 formats, mainly so it can load upstream checkpoints (gpt-oss in MXFP4, and NVFP4 releases) without re-quantizing.
- For a new engine, supporting MLX affine safetensors (huge `mlx-community` HF catalog) is a cheap compatibility win.

### Gaps
- No primary source compares MLX q4 and GGUF Q4_K_M perplexity directly.
- Per-kernel bandwidth utilization of qmv on M4 Pro was not found.

---

## 4. mlx-lm: KV caches, prompt caching, speculative decoding, batched generation/server, model loading

### Takeaway
mlx-lm is a pure-Python layer: models are written in `mlx.nn` and the generation loop is Python. KV caches are contiguous `[B, H, T, D]` tensors grown in 256-token steps. Variants include rotating (sliding window with `keep` sink tokens), quantized, chunked, batch and batch-rotating caches, an `ArraysCache` for SSM/linear-attention state, and an LRU prompt-prefix cache (a trie) in the server. Speculative decoding uses a separate draft model with cache rewind. Batched generation (v0.28.0) and server continuous batching (v0.28.4) exist but use padding, with no paged attention. Weights load through a `pread`-based parallel file reader into MLX-allocated buffers, not mmap.

### Cited Findings
**KV caches**
- `KVCache.step = 256`. When capacity is exceeded it allocates `zeros` in whole 256-step chunks and `concatenate`s them onto the existing tensor, then slice-assigns new K/V. It returns a view `[..., :offset, :]`. — [LM `mlx_lm/models/cache.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- Cache classes: `ConcatenateKVCache`, `QuantizedKVCache(group_size=64, bits=8)` (K/V stored as packed uint32 plus scales and biases), `KVCache`, `RotatingKVCache(max_size, keep)` (`make_prompt_cache` uses `keep=4` when `max_kv_size` is set), `ArraysCache` (SSM state), `ChunkedKVCache`, `CacheList`, `BatchKVCache` and `BatchRotatingKVCache`. Prompt-cache structures include `PromptTrie` and `LRUPromptCache`. — [LM `cache.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- KV quantization is enabled with `--kv-bits`. It starts at token `quantized_kv_start` (default 5000), at which point the cache is converted with `to_quantized`. Server docs: "Attention on a quantized cache is not fused, so it keeps a score matrix of `prefill_step_size x context_length`". A quantized KV cache "does not support batching". — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [LM `mlx_lm/SERVER.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- A NeurIPS 2026 paper ("The Prefill Ceiling") reports that mlx-lm prefill falls back to materializing the full attention score matrix at the query-head count, about 256 KiB per context token in their setting. This outweighs the savings from quantizing KV and shortens the usable context; their model predicts peak memory within about 1% from 8k to 65k tokens. (Seen via search-result summary; paper not fetched.) — [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/170026)

**Prefill and decode loop**
- Prefill is chunked with `prefill_step_size` (default 2048 in `generate_step`, 512 in speculative). It calls `mx.eval` on the cache state after each chunk and then `mx.clear_cache()`. The last prompt token goes into the first decode step. `--prefill-step-size` was exposed as a CLI flag in v0.31.0. — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [LM v0.31.0](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.0)

**Speculative decoding**
- `speculative_generate_step(..., num_draft_tokens=2)` uses a separate draft model. After verification it calls `_rewind_cache(num_draft, num_accept)`, which trims the target cache by `num_draft - num_accept` and the draft cache by one less. The fused SDPA for small query batches (v0.24.0) and faster small-M quantized matmuls (v0.23.0) target this path. — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [REL(v0.24.0)](https://github.com/ml-explore/mlx/releases/tag/v0.24.0), [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0)
- A fix for "output corruption in speculative decoding" landed in v0.31.2. — [LM v0.31.2](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.2)

**Batching and server**
- Batch generation arrived in mlx-lm v0.28.0 (2025-09-17), and server batching plus a multi-prompt prompt cache in v0.28.4 (2025-12-03). v0.30.4 added a continuous-batching server benchmark. v0.30.6 added distributed inference in the server. v0.31.0 brought "Better caching in the server" and cache limits. v0.31.2 refactored the batch generator, added caching of system and user messages for non-trimmable (SSM) caches, and refactored `LRUPromptCache`. — [LM releases](https://github.com/ml-explore/mlx-lm/releases)
- A PR exists to persist the server's LRU prompt cache to disk across restarts (#1405); its status was not checked. — [LM PR #1405](https://github.com/ml-explore/mlx-lm/pull/1405)
- vLLM's official `vllm-metal` plugin (blog 2026-09-22) reuses mlx_lm models but replaces attention because "MLX's `scaled_dot_product_attention` has no varlen interface". mlx_lm pads batches to `[B,H,T_max,D]` and uses contiguous caches. On Qwen3.6-35B-A3B 4-bit with 8 requests, a mixed-length batch (3000 + 7×430 prompt tokens) took 10.99 s on mlx_lm versus 3.64 s on vllm-metal with paged varlen attention. In an M5 Pro 64 GB concurrent agent benchmark, "mlx_lm completed only a minority of its requests". — [vLLM blog: vllm-metal](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)
- vllm-mlx (third-party) adds a paged KV cache with prefix sharing and continuous batching. Its arXiv paper reports 21–87% higher throughput than llama.cpp and 4.3× aggregate throughput at 16 concurrent requests. — [arXiv 2601.19139](https://arxiv.org/abs/2601.19139)

**Model loading**
- `mlx_lm.utils.load_model` calls `mx.load(wf)` per safetensors shard. With `lazy=False` (the default) it evaluates the parameters. Loading is lazy, so weights are only read on eval. — [LM `mlx_lm/utils.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py)
- MLX reads files with `ParallelFileReader`, which uses `pread` on chunked offsets in parallel into MLX-allocated buffers. No `mmap` appears in `mlx/io`. "Bulk eval in safetensors to avoid unnecessary serialization of work" (v0.19.0) and "Up to 2x faster I/O" (v0.18.0) came earlier. The new `copy=False` zero-copy import (v0.32.1) wraps caller memory, for example an mmap made by the user, which is then not wired. — [MLX `mlx/io/load.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/io/load.cpp), [REL(v0.19.0)](https://github.com/ml-explore/mlx/releases/tag/v0.19.0), [REL(v0.18.0)](https://github.com/ml-explore/mlx/releases/tag/v0.18.0), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1)
- MLX also loads GGUF (build option `MLX_BUILD_GGUF` ON by default). — [MLX `CMakeLists.txt`](https://github.com/ml-explore/mlx/blob/main/CMakeLists.txt)

### Inferences
- The "grow by concat" KV cache copies the whole cache every 256 tokens. That is an O(T²/256) copy cost plus fragmentation, which may contribute to long-context decode slowdowns. A custom engine should preallocate KV (max context) or use pages, since unified memory makes paging cheap.
- Paged, varlen attention is the main architectural gap for serving. vllm-metal shows the gain is large with mixed prompt lengths.
- Copying mlx-lm's one-step-ahead async decode loop, chunked prefill with eval and clear_cache per chunk, and trie-based prefix cache is cheap and effective.

### Gaps
- Whether mlx-lm has merged MTP (multi-token prediction) or an EAGLE-style speculative path was not confirmed; only the draft-model path was seen in `generate.py`.
- The status of PR #1405 (on-disk prompt cache) is unknown.

---

## 5. Distributed: mx.distributed backends (MPI, ring, JACCL/RDMA, NCCL), pipeline and tensor parallel

### Takeaway
`mx.distributed` has four backends in `mlx/distributed/{mpi,ring,jaccl,nccl}`. The ring backend uses TCP over Thunderbolt or Ethernet (v0.23.0). JACCL uses RDMA over Thunderbolt 5 (v0.30.1, macOS ≥ 26.2, with large bandwidth gains on macOS ≥ 26.3 in v0.30.6). MPI is OpenMPI only, and NCCL is for CUDA. mlx-lm implements pipeline parallelism (layers split across ranks) and per-model tensor-parallel `shard()` methods using sharded linear layers (v0.25.0). Multi-Mac prefill scales well; decode scales weakly.

### Cited Findings
- Backend timeline:

  | Version (date) | Change |
  |---|---|
  | v0.18.0 | send/recv and `average_gradients` |
  | v0.23.0 (2025-02-14) | Ring backend and `mlx.launch` helper |
  | v0.24.0 | "prepare tb ring" script |
  | v0.25.0 | MPI from all platforms (OpenMPI only), ring all_gather, all_min/all_max, sharded layers for tensor parallelism |
  | v0.29.0 | NCCL backend for CUDA |
  | v0.30.1 (2025-12-18) | "RDMA over thunderbolt with the JACCL backend (macOS >= 26.2)" |
  | v0.30.6 (2026-02-06) | "Much faster bandwidth with JACCL on macOS >= 26.3" |
  | v0.31.0 | JACCL mesh bandwidth improvements |
  | v0.31.2 | JACCL becomes a standalone lib |
  | v0.32.1 | JACCL coordinator optional, multi-threaded rings, scatter reduce, >2 GiB ring payloads |
  | v0.32.4 (2026-10-14) | Second cable for the JACCL mesh |

  — [REL(v0.18.0)](https://github.com/ml-explore/mlx/releases/tag/v0.18.0), [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0), [REL(v0.24.0)](https://github.com/ml-explore/mlx/releases/tag/v0.24.0), [REL(v0.25.0)](https://github.com/ml-explore/mlx/releases/tag/v0.25.0), [REL(v0.29.0)](https://github.com/ml-explore/mlx/releases/tag/v0.29.0), [REL(v0.30.1)](https://github.com/ml-explore/mlx/releases/tag/v0.30.1), [REL(v0.30.6)](https://github.com/ml-explore/mlx/releases/tag/v0.30.6), [REL(v0.31.0)](https://github.com/ml-explore/mlx/releases/tag/v0.31.0), [REL(v0.31.2)](https://github.com/ml-explore/mlx/releases/tag/v0.31.2), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1), [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)
- PR #2808 (JACCL): "enables low-latency distributed communication via thunderbolt RDMA". It was initially latency-optimized and full-mesh only. It is configured with `mlx.distributed_config --backend jaccl --over thunderbolt --auto-setup`. The PR compares 4-way all-reduce latency on M3 Ultras against the TCP ring (plots only, no numbers in the text). — [MLX PR #2808](https://github.com/ml-explore/mlx/pull/2808)
- PR #3094 (v0.30.6) added large buffers (≥ macOS 26.3) and a JACCL ring for arbitrary node counts. Benchmarks on 4× M3 Ultra, 2048-token prompt, 128 generated tokens, batch 1:

  | Model | Phase | 1 node | 2 nodes (new) | 4 nodes (new) |
  |---|---|---|---|---|
  | Devstral2 123B 4-bit | Prompt | 91.5 tok/s | 166 (mesh) / 167 (ring) | 255 (mesh) / 303 (ring) |
  | Devstral2 123B 4-bit | Generation | 9.0 tok/s | 15.8 | 26.2 |
  | GLM-4.7 358B 6-bit | Prompt | 265 tok/s | 446 | 719 (ring) |
  | GLM-4.7 358B 6-bit | Generation | 16.9 tok/s | 22.3 | 27.7 |

  For MoE at batch 8, the 4-machine speedup goes "from 1.6x to 2.6x". — [MLX PR #3094](https://github.com/ml-explore/mlx/pull/3094)
- Env vars: `MLX_HOSTFILE`, `MLX_RANK`, `MLX_WORLD_SIZE`, `MLX_RING_VERBOSE`, `MLX_IBV_DEVICES`, `MLX_JACCL_COORDINATOR`, `MLX_JACCL_RING` and `MLX_MPI_LIBNAME`. — [MLX source (grep `MLX_` env vars)](https://github.com/ml-explore/mlx/tree/main/mlx/distributed)
- mlx-lm `sharded_load(pipeline_group=..., tensor_group=...)` requires either `model.model.pipeline` (pipeline parallel) or `model.shard` (tensor parallel). Tensor-parallel sharding exists per model, including GLM, gpt-oss, MiniMax, LongCat Flash (v0.30.x) and Qwen 3.5 (v0.31.0, with a parallel head in v0.32.0). "Use depends in pipeline parallel" arrived in v0.28.1. — [LM `utils.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/utils.py), [LM releases](https://github.com/ml-explore/mlx-lm/releases)

### Inferences
- For a single Mac mini M4 Pro, distributed support is irrelevant unless you add a second Mac. Note that the M4 Pro Mac mini has Thunderbolt 5 ports, so JACCL RDMA is in principle applicable. Tensor-parallel decode gains (~1.5–1.6× on 2 nodes for a dense 123B) are much smaller than prefill gains.
- Design lesson: compute-communication overlap and RDMA latency matter more than bandwidth for decode; per-token all-reduces are small.

### Gaps
- No JACCL benchmark on M4 Pro Mac minis (only M3 Ultra) was found.
- The exact all-reduce latency numbers in µs are only in PR images.

---

## 6. M5 Neural Accelerators (Metal 4 tensor ops, "NAX") and reported speedups

### Takeaway
MLX v0.30.0 (2025-11-19) added support for the per-GPU-core Neural Accelerators on M5, using Metal 4 TensorOps / Metal Performance Primitives. It requires macOS ≥ 26.2 (`MACOSX_DEPLOYMENT_TARGET=26.2` for builds since v0.32.0). Apple reports a 3.3–4.1× TTFT (prefill) speedup over M4, but only 1.19–1.27× for decode, which is bandwidth-bound (153 versus 120 GB/s). This does not apply to an M4 Pro.

### Cited Findings
- Apple's ML Research post uses a 24 GB MacBook Pro M5 versus a similar M4, with `mlx_lm.generate`, a 4096-token prompt and 128 generated tokens:

  | Model | TTFT speedup vs M4 | Decode speedup vs M4 |
  |---|---|---|
  | Qwen3-1.7B bf16 | 3.57× | 1.27× |
  | Qwen3-8B bf16 | 3.62× | 1.24× |
  | Qwen3-8B 4-bit | 3.97× | 1.24× |
  | Qwen3-14B 4-bit | 4.06× | 1.19× |
  | gpt-oss-20b MXFP4 | 3.33× | 1.24× |
  | Qwen3-30B-A3B 4-bit | 3.52× | 1.25× |

  Memory bandwidth is 120 GB/s on M4 and 153 GB/s on M5. Requires macOS 26.2+. — [Apple ML Research: Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- MLX internals call these kernels "NAX" (`quantized_nax.metal`, `fp_quantized_nax.metal`, `gated_delta_update_nax.metal`, steel NAX GEMM and attention). Work after the initial release:

  | Version | Change |
  |---|---|
  | v0.30.1 | NAX JIT for Swift |
  | v0.30.3 | Don't use NAX if kernels are absent |
  | v0.30.4 | NAX split-K GEMM |
  | v0.31.2 | NAX refactor and segmented-mm NAX |
  | v0.32.2 | Fused full attention for D=256 on NAX |
  | v0.32.3 | NAX attention for D72/D80; M5 Ultra GEMM tunings |

  — [MLX kernels dir](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels), [MLX releases](https://github.com/ml-explore/mlx/releases)
- The releases also mention "M5 and M6" split-K test compatibility (v0.32.4) and "Optimize large NVFP4 QMV on M5 Max" (v0.32.1). — [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1)

### Inferences
- On M4 Pro (no NAX), prefill is limited by simdgroup-matrix GEMM throughput. An engine targeting M4 Pro gets nothing from NAX, but should keep a GEMM abstraction so Metal 4 tensor ops can be swapped in for M5+.

### Gaps
- Apple's post gives only ratios. Absolute TTFT and tok/s are only in figures.

---

## 7. Published benchmarks: MLX on M4 Pro / M4 Max and versus llama.cpp

### Takeaway
On M4-class chips MLX generally decodes faster than llama.cpp Metal, by a reported 1.2–1.8× for 4-bit models and more for MoE, and much faster than Ollama's older llama.cpp path. llama.cpp is competitive or better on prefill and TTFT for short prompts, and notably better on long-context (30K+) decode with flash attention. Most comparative numbers are community-sourced and not controlled.

### Cited Findings
- Official mlx-lm numbers on a 64 GB M4 Max (mlx 0.29.2, mlx-lm 0.28.2, macOS 26.1):
  - Qwen3-4B q4: 1622 tok/s prefill at 2048 tokens, 134.5 tok/s decode.
  - Qwen3-30B-A3B q4: 1754 prefill / 113.3 decode, 18.2 GB.
  - Qwen3-30B-A3B q8: 1719 / 83.2, 33.5 GB.

  — [LM `BENCHMARKS.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/BENCHMARKS.md)
- Single-Mac baselines on M3 Ultra (from PR #3094): Devstral2 123B 4-bit 91.5 tok/s prefill / 9.0 tok/s decode; GLM-4.7 358B 6-bit 265 / 16.9. — [MLX PR #3094](https://github.com/ml-explore/mlx/pull/3094)
- A Mac mini M4 Pro 64 GB running Qwen3-Coder-30B-A3B decodes about 130 tok/s on MLX (34.7 GB) versus about 43 tok/s on Ollama's llama.cpp backend (40 GB GGUF), per a Reddit report. Raw tuned llama.cpp Metal reached 89.4 tok/s on comparable hardware per Ollama issue #14861. The survey estimates MLX's real advantage over raw llama.cpp at 1.4–1.8× (dense 1.4–1.6×, MoE up to ~3× against Ollama), attributing much of the "3×" headline to Ollama overhead. — [yage.ai survey (secondary, 2026-03-31)](https://yage.ai/share/mlx-apple-silicon-en-20260331.html), [Reddit r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1s18yrt/), [Ollama #14861](https://github.com/ollama/ollama/issues/14861)
- Other M4 Pro reports, from search-result summaries only: Nemotron 3 Nano on a 48 GB M4 Pro at ~83 tok/s decode and ~702 prefill tok/s on MLX versus ~52 tok/s on llama.cpp (Medium; the page failed to load), and Qwen2.5-14B 4-bit at 68 versus 52 tok/s. Treat both as unverified. — [Medium (andreask)](https://medium.com/@andreask_75652/apples-m4-pro-is-faster-than-an-m2-max-for-nemotron-3-nano-with-mlx-2f19efbc5e50), [aiagentskit blog](https://aiagentskit.com/blog/mlx-vs-llama-cpp-apple-silicon/)
- Long context: in mlx-lm issue #763 on an M3 Ultra 256 GB with MiniMax-M2.1 4-bit via LM Studio, decode was 25 tok/s on MLX versus 32 on llama.cpp+FA at 30K context, and 5.95 versus 12.12 tok/s at 146K. Prompt processing was similar (82.5 s versus 78 s at 30K). The maintainer (awni) asked for repro commands, and no resolution appears in the thread. — [mlx-lm issue #763](https://github.com/ml-explore/mlx-lm/issues/763)
- Short prompts: one M1 Max report (~650-token prompt) found GGUF effective throughput at 20 versus 13 tok/s for MLX, with MLX spending 94% of its time in prefill. — [yage.ai survey citing r/LocalLLaMA](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)
- Release-note anchors: "4-bit Mistral 7B generates at 131 toks/sec out of the box on an M2 Ultra" (v0.23.0) and "Over 120tks with Mistral 7B" (v0.20.0). — [REL(v0.23.0)](https://github.com/ml-explore/mlx/releases/tag/v0.23.0), [REL(v0.20.0)](https://github.com/ml-explore/mlx/releases/tag/v0.20.0)

### Inferences
- For a Mac mini M4 Pro 64 GB, the realistic targets set by MLX are about 130 tok/s decode on a 30B-A3B 4-bit MoE and roughly 50–60% of M4 Max prefill throughput, since the M4 Pro has fewer GPU cores. The second is an extrapolation, not a measurement.
- The long-context decode gap suggests MLX's SDPA vector kernels and contiguous KV handling scale worse than llama.cpp's FA path. v0.30.4's "much faster vector fused GQA for long context" postdates some of these reports, so re-benchmark before concluding.

### Gaps
- No controlled, version-pinned MLX versus llama.cpp benchmark on M4 Pro (same model weights, prompt lengths, `llama-bench` versus `mlx_lm.benchmark`) was found from a primary source.
- M4 Pro memory bandwidth (Apple spec, commonly cited as 273 GB/s) was not verified in this pass.

---

## 8. Known limitations and criticisms

### Takeaway
The main criticisms are:
- No paged or varlen attention, and padded batching.
- Contiguous grow-by-concat KV caches, and unfused attention on a quantized KV cache.
- Transient prefill memory spikes (full score matrices) and allocator cache growth that needs `clear_cache()`.
- Long-context decode slower than llama.cpp with flash attention.
- Python-level loop overhead, mitigated rather than removed by async_eval.
- Wired-memory limits that need sysctl tweaks.
- Numerical non-determinism across batch sizes.
- Narrower model and quant coverage when embedded (MLX-C) than llama.cpp.

### Cited Findings
- No varlen SDPA; mlx_lm pads batches and uses contiguous KV, which motivated vllm-metal's custom paged kernel. — [vLLM blog](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)
- A quantized KV cache keeps an unfused `prefill_step_size × context_length` score matrix and disables server batching. — [LM `SERVER.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- Prefill score-matrix materialization dominates peak memory ("Prefill Ceiling" paper). An oMLX issue describes a transient 42.8 GB prefill activation spike (not resident KV) that hit a hard ceiling. — [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/170026), [oMLX issue #1522](https://github.com/jundot/omlx/issues/1522)
- mlx-lm calls `mx.clear_cache()` after every prefill chunk and every 256 decode tokens, and later releases added cache-clearing in the server and batch generation (v0.31.0). The allocator cache otherwise holds freed buffers up to `max_pool_size_`, which defaults to the full block limit. — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [LM v0.31.0](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.0), [MLX `allocator.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp)
- Long-context decode is about 50% of llama.cpp+FA at 30K–146K tokens on M3 Ultra. — [mlx-lm #763](https://github.com/ml-explore/mlx-lm/issues/763)
- Downstream engines added their own fixes. LM Studio's mlx-engine reported much lower extra RAM after concurrent long prompts (v1.8.5, ~82% less), and oMLX persists KV to SSD, reportedly cutting repeated-prefix TTFT from 30–90 s to 1–3 s. Both are secondary or vendor claims. — [LM Studio blog](https://lmstudio.ai/blog/mlx-engine-agentic-workloads), [yage.ai survey](https://yage.ai/share/mlx-apple-silicon-en-20260331.html)
- Batch-size-dependent numerical non-determinism has been reported. — [adityakarnam.com (cited by yage.ai)](https://adityakarnam.com/mlx-non-determinism-apple-silicon/)
- Embedding issues: Ollama's MLX runner supported only 6 architectures at v0.19.0, and Metal 4 bf16/half type-check issues appeared on M5. — [yage.ai survey](https://yage.ai/share/mlx-apple-silicon-en-20260331.html), [Ollama #14432](https://github.com/ollama/ollama/issues/14432)
- Model memory over ~90% of the recommended working set triggers an mlx-lm warning that generation "can be slow". The fix is to raise `iogpu.wired_limit_mb`. — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [LM README](https://github.com/ml-explore/mlx-lm/blob/main/README.md)
- `async_eval` is still labeled experimental. Compile-cache thread-safety bugs were being fixed as late as v0.32.x. — [MLX `transforms.cpp`](https://github.com/ml-explore/mlx/blob/main/python/src/transforms.cpp), [REL(v0.32.1)](https://github.com/ml-explore/mlx/releases/tag/v0.32.1), [REL(v0.32.4)](https://github.com/ml-explore/mlx/releases/tag/v0.32.4)

### Inferences: copy, avoid, improve
**Copy:**
- Unified-memory buffers with storage mode shared and untracked hazards, plus your own fences.
- A size-bucketed buffer cache.
- Residency-set wiring up to the recommended working set.
- Per-chip command-buffer commit thresholds.
- Hand-written qmv/qmm, split-K and gather_qmm kernels.
- The affine group-64 int4 format, for compatibility with mlx-community.
- One-step-ahead async decode.
- Chunked prefill with a barrier after each chunk.
- A prefix-trie prompt cache.
- Export of traced graphs to run without Python.

**Avoid:**
- Grow-by-concat KV.
- Padded batches.
- Unfused attention over a quantized KV.
- Relying on a general elementwise-only compiler for performance.
- Per-token graph rebuild in Python.
- Unbounded allocator caches that need manual `clear_cache()`.

**Improve on:**
- Paged, varlen flash-attention (prefill and decode) with block tables, sized for M4 Pro threadgroup memory.
- A fused quantized-KV attention kernel.
- A preallocated KV arena.
- Static per-layer command encoding.
- A NAX/TensorOps path behind a GEMM interface for M5+.

### Gaps
- No primary-source measurement of Python overhead per token in mlx-lm, for example a µs breakdown.
- No official MLX roadmap statement on paged attention was found.

---

## 9. (Extra scope) CPU+GPU concurrency on shared arrays, KV cache memory layout, paged-attention status, per-request caches in batched generation

### Takeaway
MLX can run CPU-stream and GPU-stream work concurrently on the same unified-memory buffers. Each CPU stream has its own worker thread, and the GPU executes asynchronously after commit. Cross-stream dependencies are enforced with fences: by default `MTLSharedEvent` signal/wait, and optionally a "fast" mode in which a shared-memory counter is spin-waited by a one-thread GPU kernel or a CPU loop. Graph encoding for one `eval` call is serialized on the calling thread. MLX has no ANE device: `DeviceType` is only `{cpu, gpu}`.

KV caches in mlx-lm are dense, contiguous `[B, n_kv_heads, T_alloc, head_dim]` tensors per layer, grown by 256-token concatenation. Batched generation uses left-padding inside one shared tensor, plus per-row `offset` and `left_padding` vectors. Merging new requests pads and concatenates; finishing requests filters rows.

Maintainers have explicitly declined paged attention for now (September 2026). They say they would build their own implementation later, and external projects (vllm-metal, vllm-mlx, mistral.rs) supply paged kernels.

### Cited Findings
**Scheduling and stream threads**
- `Scheduler` keeps one `StreamThread` per stream: a `std::thread` with a `std::queue<std::function<void()>>`. `enqueue(stream, task)` pushes to that thread. `wait_for_capacity()` throttles the producer: if active tasks exceed `MAX_ACTIVE_TASKS`, or active memory exceeds the memory limit, it commits open GPU streams (`gpu::finalize`) and blocks until tasks drain. — [MLX `mlx/scheduler.h`](https://github.com/ml-explore/mlx/blob/main/mlx/scheduler.h), [MLX `mlx/scheduler.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/scheduler.cpp)
- `eval_impl` walks the topologically sorted tape on the calling thread. For each array it calls either `gpu::eval(arr)` (encode a kernel into the stream's current command buffer) or `cpu::eval(arr)`. `cpu::eval` runs `primitive.eval_cpu` to build the work, then `encoder.dispatch(...)`, which `scheduler::enqueue`s it onto that CPU stream's thread. CPU kernels therefore execute on a background thread while the main thread keeps encoding GPU work. — [MLX `mlx/transforms.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/transforms.cpp), [MLX `mlx/backend/cpu/eval.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/eval.cpp), [MLX `mlx/backend/cpu/encoder.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/encoder.h)
- Cross-stream sync inside one eval works as follows:
  - When a consumer's stream differs from an input's producer stream, the input gets a `FenceInfo{stream_index, cross_device}`.
  - After the producer runs, `Fence::update` increments a counter.
  - Before the consumer runs, `Fence::wait(stream, in, value)` is issued.
  - Across separate (async) evals, per-stream `Event`s are attached to outputs, and a consumer on another stream calls `in.event().wait(stream)`.
  - At the end of eval, each open stream's event is signalled and GPU streams are finalized (committed).

  — [MLX `mlx/transforms.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/transforms.cpp)
- Metal fence implementation:
  - Default path (`use_fast=false`): an `Event` backed by `MTLSharedEvent` (`newSharedEvent()`), with signal/wait on the stream.
  - With `MLX_METAL_FAST_SYNCH=1`, a 4-byte shared buffer counter is used instead:
    - The GPU producer runs a `fence_update` one-thread kernel. It writes the value with a system-scope `atomic_thread_fence`.
    - For cross-device consumers, the producer first runs an `input_coherent` kernel that rewrites the array with `coherent(system)` to make it visible to the CPU.
    - A GPU consumer runs a spin-wait `fence_wait` kernel.
    - A CPU consumer spin-loops on `cpu_value() < value` inside its stream thread.

  — [MLX `mlx/backend/metal/fence.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/fence.cpp), [MLX `kernels/fence.metal`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fence.metal), [MLX `mlx/backend/metal/event.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/event.cpp)
- Measured CPU+GPU overlap: the docs' split of a GPU matmul plus a CPU chain of 500 small ops halves runtime (2.8 ms to 1.4 ms on M1 Max). — [MLX docs: Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
- Thread-local streams exist (`mx.new_thread_local_stream`, used by mlx-lm as `generation_stream`), along with `new_thread_unsafe_stream` (v0.32.0). Different host threads can therefore build and evaluate graphs on their own streams. A related mlx-lm bug: "There is no Stream(gpu, 3) in current thread at KVCache `__del__` teardown" (#1888). — [LM `generate.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py), [MLX `backend/cpu/eval.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/cpu/eval.cpp), [mlx-lm #1888](https://github.com/ml-explore/mlx-lm/issues/1888)
- No ANE support: `Device::DeviceType` is `{cpu, gpu}` only. — [MLX `mlx/device.h`](https://github.com/ml-explore/mlx/blob/main/mlx/device.h)

**KV cache layout**
- `KVCache`, `QuantizedKVCache`, `RotatingKVCache` and the batch caches all use step 256.
  - `KVCache` stores per layer `keys: [B, n_kv_heads, n_steps*256, k_head_dim]` and `values: [..., v_head_dim]` in the model dtype.
  - On overflow it allocates a new zeros block and `mx.concatenate`s it, which copies the whole cache. Writes use slice assignment `keys[..., prev:offset, :] = k`, and the returned view is `[..., :offset, :]`.
  - `QuantizedKVCache` keeps a tuple `(packed uint32 [B,H,T,D*bits/32], scales [B,H,T,D/group], biases [B,H,T,D/group])` for each of K and V.
  - `RotatingKVCache(max_size, keep)` is a ring buffer of `max_size` tokens that preserves the first `keep` tokens (attention-sink style; default `keep=4` via `make_prompt_cache`). It grows in 256 steps until it reaches `max_size`.

  — [LM `mlx_lm/models/cache.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- Prompt caches can be saved and loaded as safetensors (`save_prompt_cache`), trimmed (`trim_prompt_cache`) and checked for trimmability. `ArraysCache` holds SSM / linear-attention states, which are not trimmable, so the server caches system and user message boundaries for those (v0.31.2). — [LM `cache.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py), [LM v0.31.2](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.2)

**Paged attention status**
- mlx#2228 (2025-05) proposed integrating mistral.rs's Metal PagedAttention kernels (shown to beat llama-server under continuous batching on M3 Max). The maintainer asked for benchmark details, and the issue was closed. — [mlx #2228](https://github.com/ml-explore/mlx/issues/2228)
- mlx PR #2760 (2025-11, Metal paged attention plus batched KV writes) was closed: "looks a bit early stage... refining in your own fork". PR #2814 (flagged paged attention) was closed by awni with "Closing for now since we have continuous batching in mlx-lm." — [mlx PR #2760](https://github.com/ml-explore/mlx/pull/2760), [mlx PR #2814](https://github.com/ml-explore/mlx/pull/2814)
- mlx#2955 (2025-12, FlashAttention/PagedAttention proposal) was closed. zcbenz said the referenced slowness issue was outdated and that "after improvements MLX became faster than llama.cpp". — [mlx #2955](https://github.com/ml-explore/mlx/issues/2955)
- mlx-lm#1850 (2026-09, external paged-attention integration via `cache.update_and_attend(...)`) was closed by zcbenz: "at the moment we don't plan to support paged attention due to lack of resources... we will eventually want to support it once we have more hires but we would then develop our own implementation rather than relying on external contributions". — [mlx-lm #1850](https://github.com/ml-explore/mlx-lm/issues/1850)
- mlx-lm#1849 (2026-09, still open) proposes paged KV storage that gathers pages into contiguous arrays before the existing SDPA. It includes refcounting, copy-on-write and admission control, and was validated on MLX 0.32.2. Without a paged kernel, decode must gather every step. — [mlx-lm #1849](https://github.com/ml-explore/mlx-lm/issues/1849)

**Batched generation and per-request caches**
- `BatchKVCache(left_padding)` expects left-padded inputs. For example, prompts `[1,3,5]`, `[7]` and `[2,6,8,9]` become rows `[0,1,3,5]`, `[0,0,0,7]` and `[2,6,8,9]` with `left_padding=[1,3,0]`.
  - It keeps one shared `[B,H,T,D]` tensor, a per-row `offset` array (which starts negative at `-left_padding`), and a shared write index `_idx`.
  - The causal mask is built from `offset=_idx` and `left_padding`.
  - `filter(indices)` drops finished rows, then shifts left by the minimum remaining left padding.
  - `extend(other)` pads both caches to the same length (left and right) and concatenates along batch.
  - `merge(caches)` builds a batch cache from individual caches.
  - `extract(idx)` slices one row back out as a contiguous single-request `KVCache` for prompt caching.
  - `prepare(right_padding)` / `finalize()` use `dynamic_roll` for right-padded segments.

  `BatchRotatingKVCache` is the sliding-window analogue. — [LM `cache.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py)
- `BatchGenerator` ("implements continuous batching") has defaults `completion_batch_size=32`, `prefill_batch_size=8` and `prefill_step_size=2048`, and keeps a separate `PromptProcessingBatch` and `GenerationBatch`. Each `_next()` does the following, all on one stream (`generation_stream`):
  1. One decode step for the generation batch, with `mx.clear_cache()` every 512 steps.
  2. Return early if the generation batch is full.
  3. Otherwise admit up to `prefill_batch_size` waiting prompts.
  4. Move prompts that have finished prefill into the generation batch, using `split(...).generate()` and `extend`.
  5. Run one prefill chunk of up to `prefill_step_size` tokens per prompting sequence.

  Prefill chunks and decode steps therefore alternate serially on the GPU rather than running concurrently. — [LM `mlx_lm/generate.py` `BatchGenerator`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py)
- The quantized KV cache does not support batching: the server serializes requests when `--kv-bits` is set. — [LM `SERVER.md`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- External comparison: with mixed prompt lengths, padded batching costs about 3× wall time versus paged varlen attention (10.99 s versus 3.64 s). — [vLLM blog: vllm-metal](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)

### Inferences
- **Prefill on one device, decode on another.** MLX's primitives permit it: put the prefill graph on `mx.gpu` and a small decode or draft model on `mx.cpu`, from separate threads with thread-local streams, sharing weights and KV buffers with no copies. Two cautions apply:
  - The CPU (Accelerate/NEON/AMX via BLAS) has much lower matmul throughput than the GPU.
  - CPU and GPU share the same DRAM bandwidth (on the order of 270 GB/s on M4 Pro, unverified here). Bandwidth-bound decode on both at once will contend.

  The realistic split is GPU prefill + GPU decode interleaved (as BatchGenerator does), with the CPU taking sampling, tokenization, grammar masks and small elementwise ops. The ANE requires Core ML / private APIs and is outside MLX.
- **Sync cost.** The default `MTLSharedEvent` waits are cheap at coarse granularity. MLX's fast-synch spin kernels show that fine-grained CPU-GPU handoffs on unified memory can be done with a shared atomic counter plus system-scope fences, without ending command buffers. That is a useful pattern for a custom engine doing per-layer CPU/GPU pipelining.
- **KV design implications:**
  - mlx-lm's layout (`[B,H,T,D]`, head-major, contiguous per head) is SDPA-friendly but forces whole-cache copies on growth and on batch merges.
  - A custom cache should preallocate a page pool of, say, 16–256 tokens × H × D per layer in one wired `MTLBuffer` / `MTLHeap`, and use block tables plus a paged decode kernel. This removes concat/pad/roll copies, enables prefix sharing with copy-on-write (as in mlx-lm#1849), and lets prefill and decode requests coexist without padding.
  - Since MLX maintainers won't ship this soon, it is a clear differentiation point.

### Gaps
- The value of `MAX_ACTIVE_TASKS` and the default for `MLX_METAL_FAST_SYNCH` (code reads `config::get("MLX_METAL_FAST_SYNCH", 0)`, so it appears off by default) were not checked further.
- No benchmark was found of concurrent CPU-stream and GPU-stream LLM workloads, such as a CPU draft model with GPU verification, in MLX.
