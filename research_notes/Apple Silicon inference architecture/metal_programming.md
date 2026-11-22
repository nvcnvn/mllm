# Metal / MSL Programming for LLM Inference Kernels on Apple Silicon

Target: Mac mini M4 Pro (20-core GPU, 64 GB), macOS 26/27 with Metal 4 / 4.1. Notes current as of 2026-11-21. Where I cite MLX or llama.cpp source, I read local shallow clones made that day (MLX `dc5ea6e`, 2026-11-21; llama.cpp `19bc41d`, 2026-11-21). Many 2026 arXiv papers cited below are preprints. Treat their numbers as indicative, not authoritative.

**Chip naming used below:** Apple GPU family 7 = M1/A14. Family 8 = M2/A15/A16. Family 9 = M3/M4/A17 Pro/A18. Family 10 = M5/A19. The M4 Pro has **no** GPU neural accelerators. Those arrived with M5/A19.

---

## Memory model: storage modes, no-copy/mmap buffers, hazard tracking, residency sets, working-set limits, CPU-GPU sync

### Takeaway
On Apple Silicon, use `MTLStorageModeShared` and `newBufferWithBytesNoCopy` over a page-aligned mmap of the weight file. This gives GPU-visible weights with zero copies. llama.cpp does exactly this. Production engines (MLX and llama.cpp) also disable Metal's automatic hazard tracking (untracked resources) and synchronize manually. They use `MTLResidencySet`s, which are explicitly kept resident, so the OS does not page weights out between tokens. CPU-GPU round trips (`waitUntilCompleted`) cost on the order of 100 µs. Keep them out of the per-token loop. Pipeline CPU encoding of token t+1 with GPU execution of token t.

### Cited Findings
**Storage modes**
- Apple GPUs have unified memory, but "CPU and GPU access to that memory depends on the storage mode you choose." Each mode has a stated use:
  - `shared`: "system memory that both the CPU and the GPU can access". It is the default for buffers and textures. Use it for data the CPU populates and the GPU reads often.
  - `private`: GPU-only. Use it when the GPU populates the resource, such as intermediates.
  - `memoryless`: tile memory. It works for **textures only, not buffers**, so it is irrelevant to compute buffers.
  — [Apple: Choosing a resource storage mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)
- MLX allocates every buffer with `MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked`. — [mlx/backend/metal/allocator.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp) (line 16)
- llama.cpp sets `use_shared_buffers = has_unified_memory`. It creates private buffers only for discrete or external GPUs, or when forced by an env var. — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~line 1271)

**No-copy buffers and mmap'ing weights**
- `makeBuffer(bytesNoCopy:length:options:deallocator:)` has these documented requirements:
  - pointer: "A page-aligned pointer to the starting memory address".
  - length: "results in a page-aligned region of memory".
  - Important: "The existing memory allocation needs to exist within a single virtual memory (VM) region."
  - It returns `nil` on failure.
  — [Apple docs](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))
- llama.cpp's `ggml_metal_buffer_map(ptr,size)` wraps the mmap'd GGUF in place:
  - It rounds the pointer *down* to `sysconf(_SC_PAGESIZE)` and rounds the size up to a page multiple.
  - It calls `newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil`.
  - If the file exceeds `maxBufferLength`, it creates several **overlapping views** sized `max_buffer_size`. The overlap guarantees the largest tensor fits fully inside one view.
  — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (lines ~2240–2300)
- MLX's `make_buffer(void* ptr, size)` wraps caller-owned memory. It deliberately does **not** wire it: "Wiring a file-backed mapping would read the whole file and keep it in memory." — [mlx/backend/metal/allocator.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp) (~line 220)

**Residency sets and wired memory**
- `MTLResidencySet` is "a collection of resource allocations that can move in and out of resident memory". Its `allocatedSize` reports resident bytes. — [Apple: MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset)
- In Metal 4, residency sets are populated at startup because "contents rarely change". They are attached to an `MTL4CommandQueue`, so "all command buffers committed to that queue will then include those resources". They can also be attached per command buffer (`useResidencySet:`). Updates are cheap and can happen on another thread. — [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/), [WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/)
- MLX's residency policy:
  - It keeps allocations resident up to a wired limit set by `set_wired_limit`, which is 0 by default.
  - It spreads them across several size-capped `MTL::ResidencySet`s. The reason given: "macOS makes residency decisions per residency set, so when a set loses residency under GPU memory pressure only the allocations in that set have to be made resident again".
  - Each set holds a standing `requestResidency()` and is attached to every command queue. The env var `MLX_RESIDENCY_SET_MAX_PCT` sets the per-set cap.
  — [mlx/backend/metal/resident.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.h)
- `mx.set_wired_limit` cannot exceed the system wired limit (`max_recommended_working_set_size` from `device_info()`). Setting a larger value is an error. — [MLX docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
- `mlx_lm.server` wires about 75% of RAM at startup via `set_wired_limit(max_recommended_working_set_size)`, for example about 72 GB on a 96 GB machine. A user argued in an issue that this is too aggressive. — [mlx-lm issue #883](https://github.com/ml-explore/mlx-lm/issues/883)
- llama.cpp also keeps a residency-set collection. A background loop keeps requesting residency until `keep_alive_s` of inactivity passes. The default is 3×60 s, overridable with `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`. — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~lines 934–1000)

**recommendedMaxWorkingSetSize**
- Apple defines it as "an approximation of how much memory, in bytes, this GPU device can allocate without affecting its runtime performance". — [Apple docs](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize)
- llama.cpp reports it as the device's "total" memory. A source comment notes that "it's possible to allocate more than `recommendedMaxWorkingSetSize`". — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~line 1503)
- On iPhone, the value was 8192 MiB and larger models failed to load. — [Apple forum thread 805161](https://developer.apple.com/forums/thread/805161)

**Synchronization and round-trip cost**
- MLX synchronizes with `addCompletedHandler` on command buffers and with `MTL::SharedEvent` (`newSharedEvent`) for cross-stream events. Fences (`fence.cpp`) handle ordering between untracked buffers. — [mlx/backend/metal/event.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/event.cpp), [fence.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/fence.cpp)
- llama.cpp exposes `ggml_metal_event` objects backed by `id<MTLSharedEvent>`. — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~line 1448)
- An empty kernel followed by `waitUntilCompleted` took about **0.12 ms** on an M2 Max, versus about 0.001 ms on CUDA. This is a sync round-trip cost, not dispatch cost. — [Apple dev forum 767853](https://developer.apple.com/forums/thread/767853) (via search summary)
- One study of MLX decode measured CPU graph build and encode at about 0.54 ms/token. This is "essentially invisible" because MLX overlaps CPU work for token t+1 with GPU work for token t. — [arXiv 2610.31479](https://arxiv.org/pdf/2610.31479) (preprint, search summary)

### Inferences
- For an engine whose weights are GGUF or safetensors on disk, the plan has four steps:
  1. mmap the file read-only.
  2. Page-align each region.
  3. Wrap it with `newBufferWithBytesNoCopy` (shared, untracked).
  4. Add the buffers to a residency set with `requestResidency()`.
- The first touch then faults pages in from SSD. Prefault (`madvise(MADV_WILLNEED)` or a read pass) if cold-start latency matters.
- With 64 GB, expect `recommendedMaxWorkingSetSize` to be roughly 70–75% of RAM by default. This follows from the mlx-lm "~75%" observation. Raising the system wired limit (the `iogpu.wired_limit_mb` sysctl) is a common practice, but I did not verify it in this pass.
- Untracked hazard tracking with explicit barriers or fences is the norm. Tracked mode adds driver-side dependency analysis per dispatch.

### Gaps
- **Managed storage mode:** My sources did not cover it. Apple's page does not describe it for Apple GPUs. Common knowledge says it is a no-op or unnecessary on Apple Silicon, but that is unverified here.
- **Page size:** I did not confirm the exact page size from a primary source. Apple Silicon uses 16 KB pages, which is why llama.cpp uses `sysconf(_SC_PAGESIZE)` rather than a hard-coded 4096.
- **Default value:** I found no primary-source number for the default `recommendedMaxWorkingSetSize` on a 64 GB M4 Pro.

---

## Command submission overhead: command buffers, encoders, concurrent dispatch, ICBs, fusion

### Takeaway
GPU-side dispatch costs a few µs each: about 1.7 µs for an MTL4 dispatch, and about 3–3.5 µs with a barrier. Decode runs hundreds of small kernels per token, so dispatch and barrier overhead plus memory round trips between kernels become significant at small model sizes. Kernel fusion therefore matters most for small and dense models. Both MLX and llama.cpp use a single concurrent compute encoder with explicit memory barriers. MLX also splits work into multiple command buffers sized by op count and MB.

### Cited Findings
- **Measured dispatch costs (base M5):**
  - An MTL4 dispatch costs **1.66 µs**, or **3.5 µs** with a Dispatch→Dispatch barrier.
  - A spin grid barrier costs 1.9–5 µs per phase, and it risks deadlock without a forward-progress guarantee.
  - `recommendedPersistentThreadgroupsPerGrid` was 0 in all 12 tested combinations, so persistent kernels are not viable there.
  - A relaxed `threadgroup_barrier` is 1.76× cheaper than seq_cst.
  — [scratchy issue #118](https://github.com/AI-native-Systems-Research/scratchy/issues/118) (third-party measurements)
- A GPU-side dispatch boundary of about **3.04 µs** was measured for MLX. The study concluded: "In this Apple/MLX configuration, dispatch boundaries are nearly free and byte traffic dominates." Decode ran at about 170 GB/s effective, against per-op rates of 250–440 GB/s. — [arXiv 2610.31479](https://arxiv.org/pdf/2610.31479) (preprint, via search summary)
- BaseRT, a native-Metal runtime, argues that fusion removes one launch and one memory round trip per fusion. It reports its largest gains on small dense models, for example up to 56% faster than llama.cpp on Qwen3-0.6B. These are vendor-reported numbers. — [BaseRT blog](https://huggingface.co/blog/basecompute/basert-release)
- **MLX submission model:**
  - It uses `computeCommandEncoder(MTL::DispatchTypeConcurrent)`.
  - It commits a command buffer every `max_ops_per_buffer` ops or `max_mb_per_buffer` MB. The values are 40 ops / 40 MB for base and Pro chips (arch suffix `g`), 50/50 for Max (`s`) and Ultra (`d`), and 20/40 for phones. Env vars `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` override them.
  - Splitting lets the GPU start work while the CPU is still encoding.
  — [mlx/backend/metal/device.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp) (~lines 569, 676, 700–720)
- **llama.cpp submission model:**
  - It uses `computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent` and inserts `memoryBarrierWithScope:MTLBarrierScopeBuffers` only between dependent nodes.
  - A graph-level fusion table (`ggml-metal-fusion.cpp`) is the "single source of truth for the fusions supported by the Metal backend". The graph optimizer and the op encoders both consult it to pack chains of nodes into one kernel.
  — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~lines 842, 895), [ggml-metal-fusion.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-fusion.h)
- **Metal 4 command model:**
  - `MTL4CommandBuffer`s are decoupled from queues and independent, so they can be encoded in parallel.
  - `MTL4CommandAllocator` gives explicit control of command-buffer memory.
  - A unified compute encoder absorbs blit and acceleration-structure work, reducing encoder count.
  - `MTL4ArgumentTable` replaces per-dispatch bind points ("in the bindless case, the argument table just needs one buffer binding").
  - A new low-overhead stage-to-stage Barrier API was added.
  - `MTL4Compiler` is separate from the device and inherits the calling thread's QoS.
  — [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)
- `queue commit:&commands count:1` submits batches of command buffers. ML work is identified as `MTLStageMachineLearning` in barriers, for example `barrierAfterStages:MTLStageMachineLearning beforeQueueStages:... visibilityOptions:MTL4VisibilityOptionDevice`. — [WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/)

### Inferences
- **Fusion targets for decode** (hundreds of dispatches per token):
  - RMSNorm into the next GEMV.
  - RoPE into QKV.
  - The KV-cache write into the attention kernel.
  - SwiGLU into the down-projection input.
  - Residual adds into GEMV epilogues.
- **Command-buffer pattern:** Encode each token into 1–3 command buffers with a concurrent encoder and explicit barriers. Overlap CPU encoding of token t+1 with GPU execution of token t. Never `waitUntilCompleted` inside the loop. Read the sampled token through a shared buffer plus a completion handler or `MTLSharedEvent`.
- **Indirect command buffers:** They could cut CPU encoding cost further. Because decode shapes are fixed per step, an ICB or a pre-built argument table can be replayed. However, neither MLX nor llama.cpp relies on ICBs for compute.

### Gaps
- I found no primary-source Apple number for per-dispatch overhead on M4 Pro specifically.
- I found no measurements of indirect command buffer (ICB) benefit for LLM decode on Metal.
- MLX and llama.cpp do not appear to use MTL4 command buffers yet. I grepped MLX for `MTL4` and found no hits. I did not check llama.cpp for MTL4 usage.

---

## Kernel techniques: SIMD-group functions, simdgroup_matrix, threadgroup tiling, bfloat, occupancy, Dynamic Caching

### Takeaway
Apple GPUs have 32-wide SIMD groups and 128 ALUs per core. Before family 9, each core had about 208 KB of registers and about 60 KB of threadgroup memory. The core tools are:
- `simd_sum` / `simd_shuffle` for reductions in decode GEMV.
- `simdgroup_matrix<T,8,8>` with `simdgroup_multiply_accumulate` for GEMM and attention.

Decode GEMV is bandwidth-bound, so the design priorities are full coalesced weight reads, in-register dequantization, and multiple output rows per SIMD group to reuse the x vector. M3 and later have Dynamic Caching: registers, threadgroup, stack and buffer memory share one on-chip cache that is allocated dynamically. This makes occupancy far less sensitive to peak register use, and FP32, FP16 and integer pipelines can dual-issue across SIMD groups.

### Cited Findings
**Hardware basics (Apple 7/8, i.e. M1/M2)**
- Core structure: SIMD width 32. Each core has 4 schedulers and 128 ALUs. ALU utilization maxes out at 24 SIMD groups per core.
- On-chip memory per core:
  - Register file: about 208 KB.
  - Threadgroup memory: about 60 KB.
  - L1 data cache: 8 KB.
  - Cache line: 128 B.
  - SIMD shuffle bandwidth: 256 B/cycle.
- Throughput and latency:
  - Per core-cycle: 256 FP32 FMA and 256 FP16 FMA. FP16 has no peak speedup on M1/M2.
  - Matrix FFMA16 / FFMA32: about 102 per core-cycle.
  - `simdgroup_matrix` 8×8 FP16 matmul: throughput about 17 cycles, latency about 34 cycles.
  - EXP2_32: 4 cycles. RSQRT32: 8. RECIP32: 6.
  - IMUL32: 32 per core-cycle, versus 128 I32 adds. Integer multiplies are slow, which affects index math.
- Spilling: Apple "would rather you spill to device memory" than reduce occupancy.
- SLC (L3) sizes are 8 MB for M1/M2, 24 MB for Pro, and 48 MB for Max. No M3/M4 data is given.
- Native float atomics: none.
— [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)

**Dynamic Caching (M3 and later)**
- Pre-M3, a SIMD group held registers equal to the kernel's **maximum** use for its whole lifetime. On M3 and later, registers are allocated and freed dynamically, and "the register file acts as a cache".
- Register, threadgroup, tile, stack and buffer memory share fewer, larger on-chip caches. Storage that a shader does not use is reassigned to the types it does use.
- When the working set exceeds on-chip capacity, data spills to the next cache level. The hardware **adjusts occupancy** automatically to keep data on chip.
- FP32, FP16 and integer pipelines run in parallel when instructions come from *multiple* SIMD groups, for "up to 2x ALU performance". FP16 is recommended "wherever possible", and FP16 conversions are free.
— [Apple Tech Talk: Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/)

**bfloat support**
- llama.cpp enables bf16 when the device supports `MTLGPUFamilyMetal3` or `MTLGPUFamilyApple6`. It then double-checks by compiling a test kernel, and disables bf16 if compilation fails. — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~lines 1141–1260)
- Metal's `bfloat` uses IEEE-compliant rounding, which is "a major overhead" on chips without hardware BF16. MFA therefore uses BF16 emulation for mixed precision. — [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)

**Quantized GEMV (decode)**
- MLX's `qmv_fast_impl` geometry:
  - 2 SIMD groups per threadgroup.
  - 4 output rows per SIMD group (`results_per_simdgroup = 4`).
  - Each lane loads `values_per_thread = pack_factor × packs_per_thread` x values into registers, in `float` (`typedef float U`), once per K block.
  - Each lane streams packed `uint32` weights for all 4 rows. It applies per-group scale and bias as `scale·Σ(w·x) + bias·Σx`, using the precomputed `sum` of x.
  - Partial rows reuse the last valid row's weights to avoid bounds checks inside the loop.
  — [mlx/backend/metal/kernels/quantized.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h) (~line 851)
- MLX qmv path selection by M (from an issue author and a PR):
  - For M ≥ 2, affine quantized matmul uses `qmv_wide` up to a batch limit of about 10–12, then `qmm`.
  - `qmv_wide` dequantizes each weight sub-chunk once and reuses it across rows. Each extra row is FMA/issue-bound, at about 40% of scalar fp32 peak.
  - `qmv_wide` tiles M into ceil(M/5) blocks, and each block re-reads the whole weight matrix.
  — [MLX issue #3852](https://upd.dev/ml-explore/mlx/issues/3852), [MLX PR #3863](https://github.com/ml-explore/mlx/pull/3863)
- llama.cpp's Q4_K mat-vec kernel uses 2 rows per SIMD group and 2 SIMD groups per threadgroup. This comes from a third-party adaptation of `kernel_mul_mv_q4_K_f32_impl`. — [ferrum-kernels q4_k_gemv_v2.metal](https://docs.rs/crate/ferrum-kernels/0.7.2/source/src/q4_k_gemv_v2.metal)
- A PyTorch MPS PR got a 5× speedup in its int4 kernel by "exploiting SIMD groups and thread-local caches", following MLX's `gemv.metal`. — [PyTorch PR #125704](https://github.com/pytorch/pytorch/pull/125704)

**GEMM and attention tiling**
- MLX steel GEMM:
  - `BlockMMA` uses 8×8 fragments (`kFragSize = 8`) and `simdgroup_multiply_accumulate`.
  - Tile configs are (BM, BN, BK, WM, WN) = (64,32,8,4,1), (64,32,32,2,2) and (64,64,16,1,2), chosen per dtype, shape and arch.
  — [mlx/backend/metal/matmul.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp), [kernels/steel/gemm/mma.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h)
- llama.cpp `mul_mm` uses NR0 = 64 × NR1 = 32 output tiles. It stages A and B in threadgroup memory and loads them with `simdgroup_load`, then accumulates in `simdgroup_matrix<float,8>`. — [kernels/mul_mm.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal)
- MLX decode attention (`sdpa_vector`):
  - One threadgroup handles one (batch·head, query) pair.
  - BN = 32 SIMD groups each walk a strided subset of keys. BD = 32 lanes each hold D/32 elements of q, k and the output in float registers.
  - Each SIMD group keeps a partial online softmax. Partials are combined through threadgroup memory (`outputs[BN*BD]`, `max_scores`, `sum_exp_scores`).
  - Function constants specialize mask, causal, sinks and sliding window. GQA works via `gqa_factor`.
  — [mlx/backend/metal/kernels/sdpa_vector.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h)
- Philip Turner's metal-flash-attention (MFA):
  - At large D (for example 256), the design *intentionally* spills registers and blocks along D as well.
  - Block aspect ratios are skewed, 16–32 along the parallelized dimension and 80–128 along the traversal dimension, to reduce spill bandwidth.
  - The backward pass avoids FP32 atomics, which are emulated on Apple GPUs. It uses 7 GEMMs instead of 5.
  - Everything is JIT-compiled.
  - Results: 86% ALU utilization forward on M1 and 82–94% forward on M3. On M1 Max that is about 4400 GINSTRS/s, or 83%.
  — [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)

### Inferences
- **On M4 Pro (family 9, Dynamic Caching):**
  - Register-heavy kernels are less likely to tank occupancy than on M1/M2, so larger per-thread tiles become viable.
  - Check the "Occupancy Manager Target" counter (next sections) to detect hardware-throttled occupancy.
  - Mixing FP16 math with FP32 accumulation across SIMD groups can exploit the dual-issue pipelines.
- **Specialization:** Use function constants (as MLX does) rather than runtime branches for mask, causal, quant bits and group size. This keeps register allocation tight and loops unrolled.
- **Decode GEMV recipe** (the pattern MLX and llama.cpp converge on):
  - 1 SIMD group handles R = 2–8 rows.
  - Each lane reads 16–32 contiguous bytes of weights per iteration.
  - Hold x in registers, accumulate in fp32, and finish with one `simd_sum` per row.
  - Threadgroups stay small (64 threads) so many are in flight to saturate bandwidth.

### Gaps
- I found no published M3/M4 microbenchmarks with the same detail as metal-benchmarks: register file size, `simdgroup_matrix` latency, or BF16 throughput. metal-benchmarks covers only Apple 7/8.
- `simdgroup_async_copy` is an undocumented intrinsic used by older MFA and MLX versions. I did not verify its current status.
- I found no primary-source statement of the MSL version or chip that introduced native `bfloat`. Common knowledge says MSL 3.1 (2023), with hardware acceleration on M2 and later. llama.cpp gates on the Metal3/Apple6 family plus a compile test.

---

## Metal 4 (WWDC25) and Metal 4.1 (WWDC26): MTLTensor, Metal Performance Primitives / TensorOps, ML encoder, M5 neural accelerators

### Takeaway
Metal 4 runs on M1/A14 and later. It adds three ML pieces:
- `MTLTensor`, a rank-N resource with strides that can be buffer-backed.
- MSL `tensor<>` types and Metal Performance Primitives (MPP, also called "TensorOps"), with `mpp::tensor_ops::matmul2d`, cooperative tensors and reductions.
- `MTL4MachineLearningCommandEncoder`, which runs Core ML–converted `.mtlpackage` networks on the GPU timeline.

On **M5/A19**, `matmul2d` maps onto per-core **neural accelerators**. Apple reports a 3.3–4× TTFT speedup versus M4 but only 1.19–1.27× in decode, since decode is bandwidth-bound. On **M1–M4, including the target M4 Pro**, `matmul2d` runs on the ordinary shader cores via the simdgroup_matrix/ALU path. There it gives at best about 5–21% over hand-written simdgroup_matrix kernels, and llama.cpp disables it by default. Metal 4.1 (macOS 27) adds FP8, FP4, INT2 and MX block-scaled quantized tensor types.

### Cited Findings
**Availability**
- Metal 4 is "supported by devices equipped with the Apple M1 and later, as well as the A14 Bionic and later". It uses the same `MTLDevice` and requires Xcode 26 / macOS 26. — [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)

**MTLTensor**
- Descriptor fields: rank, extents, `dataType` and `usage` (`MTLTensorUsageMachineLearning | Compute | Render`).
- Device-created tensors (`newTensorWithDescriptor:offset:error:`) have an **opaque, optimized layout**.
- Buffer-backed tensors need explicit `strides`. "The innermost stride must always be 1."
- In MSL: `tensor<device half, dextents<int,2>>` binds through buffer slots. Inline tensors created in-shader are assumed tightly packed.
— [WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/)

**MPP matmul2d**
- `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`.
- `constexpr tensor_ops::matmul2d_descriptor desc(M, N, K, transposeLeft, transposeRight, reducedPrecision); tensor_ops::matmul2d<desc, execution_thread> op; op.run(A, B, C);`
- Execution scope guidance:
  - Use larger scopes (simdgroup, threadgroup) when the whole group does uniform work.
  - Use `execution_thread` for divergent work.
— [WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/)
- For small networks, "the OS shader compiler inlines shader code, optimized for the device being used, directly into your shader". — [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)

**ML command encoder**
- Workflow:
  1. Convert with coremltools (`convert_to='mlprogram'`, `minimum_deployment_target=macOS16`).
  2. Run `metal-package-builder` to produce a `.mtlpackage`.
  3. Load it with `newLibraryWithURL`, then build an `MTL4MachineLearningPipelineDescriptor` (`setInputDimensions:` for dynamic shapes).
  4. Create a placement heap of `pipeline.intermediatesHeapSize`.
  5. Call `encoder dispatchNetworkWithIntermediatesHeap:`.
- Only ML programs are supported.
— [WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/)

**WWDC26 "Optimize custom machine learning operations with Metal tensors"**
- Hardware: M5 and A19 have a "neural accelerator located directly in each shader core" for "dense compute-bound work such as the prefill stage of an LLM". TensorOps "uses any available hardware acceleration across all Apple Silicon GPU generations."
- Type support by OS:
  - macOS/iOS 26: 4- and 8-bit **integer** tensor types, plus SIMD-group storing of cooperative tensors to threadgroup memory.
  - macOS/iOS 27: FP8 and FP4, INT2, multi-plane tensors with scale planes (FP8 E8M0 block scales), and MX formats.
- Quantized matmul: `matmul2d<descriptor, execution_simdgroups<4>> op; op.run(tA,tB,tC)` on sliced quantized tensors, and "TensorOps handles dequantization automatically". For custom formats, dequantize into a **cooperative tensor**, which lives in per-thread private memory, to skip a threadgroup round trip.
- Cooperative-tensor APIs: `get_destination_cooperative_tensor`, `get_row_reduction_destination_cooperative_tensor`, `reduce_rows(ct, rowMax, reduction_operation::max, -INFINITY)`, `map_iterator` for elementwise softmax, and `is_compatible_as_left_input` / `get_left_input_cooperative_tensor`. The last pair lets the S tile feed the P·V matmul without a memory write. If layouts are incompatible, the fallback is store, `simdgroup_barrier` and reload.
- Flash attention: the example uses `execution_simdgroup`, where each SIMD group owns full rows, so softmax needs no cross-SIMD-group exchange.
— [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/)

**M5 neural accelerators in practice**
- MLX needs **macOS 26.2 or later** to use them.
- Versus M4 (MacBook Pro, 24 GB):
  - TTFT: 3.33–4.06× faster.
  - Decode: 1.19–1.27× faster.
  - Bandwidth: 153 GB/s versus 120 GB/s.
- Example: Qwen3-14B-4bit TTFT was 4.06× faster and generation 1.19× faster.
— [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- MLX gating:
  - `is_nax_available()` requires `__builtin_available(macOS 26.2, ...)` and architecture generation ≥ 17 for Macs (≥ 18 for phones).
  - It routes quantized matmul, GEMM (`steel/gemm/gemm_nax.h`), SDPA and fp_quantized kernels to `*_nax` variants built on `MetalPerformancePrimitives.h`. Non-NAX devices keep the simdgroup_matrix "steel" kernels.
  — [mlx/backend/metal/device.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp) (~line 1046), [quantized.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp)
- llama.cpp gating:
  - It enables the tensor API only when the device name contains M5, M6, A19 or A20, unless `GGML_METAL_TENSOR_ENABLE` is set. The source comment reads: "disable the tensor API by default for old chips because with the current implementation it is not useful – M2 Ultra: ~5% slower – M4, M4 Max: no significant difference".
  - Tensor-API kernels exist for flash attention (`fa_tensor.metal`, `matmul2d<..., execution_simdgroups<NSG>>`) and convolution.
  — [ggml-metal-device.m](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m) (~lines 1147–1165), [kernels/fa_tensor.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/fa_tensor.metal)
- Rigel (M4 Max, preprint):
  - fp16 `matmul2d` reached 14.8 TFLOP/s at 2048³, about 46% of a "~32 TFLOP/s scalar-ALU fp16 roofline".
  - It was only **1.05–1.21×** faster than the legacy simdgroup_matrix path.
  - fp8 E4M3 ran at 0.87–0.94× of fp16, so it is emulated.
  - The authors conclude matmul2d on M4 "runs on the GPU shader cores via the simdgroup_matrix and FP32-ALU path, with no dedicated matrix unit".
  - Fusing epilogues (bias, GELU) on the cooperative-tensor fragment gained 6.5–12.9%.
  - The contracted dimension must be a multiple of 128 bytes for `tensor_inline` (64 halfs).
  - Metal 4.1 binaries need macOS 27 and Xcode 27, and do not load on 26.5.
  - The cooperative-tensor fragment is a repeating 8×8 base, with 2·(M/8)·(N/8) elements per lane.
  — [arXiv 2606.12765](https://arxiv.org/html/2606.12765v1)
- Base M5 (third-party):
  - `matmul2d` f16/bf16 reached 16.3–16.6 TFLOP/s. int8×int8 reached 31–33 TOPS (2×). int8×int4 reached 25.5–29 TOPS. fp8 ran at 1.0× (no double rate) and fp4 at 0.8×.
  - Cooperative tensors as matmul inputs must use `execution_simdgroup`.
  - The matrix unit helps 5–16-token batched decode steps. For 2–4 tokens it ties with `qmv_fast`.
  - Version requirements: int8/int4 `matmul2d` needs MSL 4.0 with macOS ≥ 26.2. int2, fp8/fp4 and MX need MSL 4.1 with macOS 27.
  - On M4 and earlier, `matmul2d` is emulated and "gives a wrong layout", so they gate it on NAX capability.
  — [scratchy issue #118](https://github.com/AI-native-Systems-Research/scratchy/issues/118)

**Sources conflict on execution scopes**
- Rigel describes `execution_simdgroups<N>` as N SIMD groups cooperating on one tile. Apple's WWDC26 example uses `execution_simdgroup` for an independent matmul per SIMD group. — [arXiv 2606.12765](https://arxiv.org/html/2606.12765v1), [WWDC26 330](https://developer.apple.com/videos/play/wwdc2026/330/)
- These readings look complementary (two distinct scope types), but the MPP headers are authoritative.

### Inferences
- **For the M4 Pro target:**
  - MPP/TensorOps is a portability and maintainability choice, not a speed win. Expect at most about 0–20% for prefill GEMM, and nothing for decode GEMV, which is bandwidth-bound.
  - Write simdgroup_matrix kernels in the style of MLX steel and llama.cpp as the primary path. Optionally add an MPP path gated on M5+ (`supportsFamily` Apple10 or a name check) for prefill.
- **Version targeting:** Metal 4.1 quantized types such as MXFP4 and FP8 with scale planes would only save memory on M4 Pro. Targeting MSL 4.1 also forces macOS 27.
- **ML command encoder:** It suits fixed graphs converted from Core ML. Hand-written LLM engines with dynamic KV caches would likely not use it.

### Gaps
- I found no official Apple statement of exactly how TensorOps executes on M1–M4, such as an emulation path or its performance. Only third-party measurements exist.
- Apple did not publish neural-accelerator peak TFLOPS for M5. The base-M5 numbers come from third-party measurements.
- I did not look up an Apple feature-set table listing per-family support for `MTLTensor` usages.

---

## Metal Performance Shaders / MPSGraph for ML

### Takeaway
My sources say little about MPS/MPSGraph for LLM inference. Both high-performance open-source engines (MLX, llama.cpp) bypass MPS and use custom Metal kernels.

### Cited Findings
- MLX's Metal backend implements all kernels itself: GEMM via steel, quantized ops, SDPA, norms, RoPE and so on. Its `kernels/` directory has no MPS dependency for these ops. — [mlx/backend/metal/kernels](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels)
- PyTorch MPS needed custom Metal kernels for int4 matmul, inspired by MLX's `gemv.metal`. Its SIMD-group and thread-local cache approach gave a 5× jump over the initial vectorized version. — [PyTorch PR #125704](https://github.com/pytorch/pytorch/pull/125704)
- PyTorch has since added an MPP-based attention path. — [PyTorch PR #182256](https://github.com/pytorch/pytorch/pull/182256)

### Inferences
- MPSGraph is useful as a correctness oracle or a quick baseline for dense fp16 GEMM. Its lack of support for custom quantization formats and fused decode kernels makes it a poor fit for an inference engine's hot path.

### Gaps
- I found no recent MPSGraph versus MLX GEMM benchmarks.
- I did not fetch current MPSGraph quantization support (MPSGraph added int4/int8 dequantize ops around macOS 15). This needs verification.

---

## Profiling: Xcode GPU capture, Metal System Trace, counters, bandwidth

### Takeaway
For family 9 GPUs (M3/M4), Xcode's GPU capture exposes several tools: per-pipeline **shader cost graphs** with per-line costs, compute **heat maps** by thread ID, and limiter/utilization counters. The counters cover occupancy, launch limiter, the Occupancy Manager Target, L1 eviction rate, LLC and MMU limiters, and ALU sub-unit limiters. Measure achieved bandwidth as bytes moved divided by kernel time, from GPU timestamps, and compare against the chip's peak.

### Cited Findings
- **Counters on M3/A17 Pro and later:**
  - **Occupancy:** concurrent SIMD groups per core.
  - **Compute Shader Launch Limiter:** low means the work is too small to fill the GPU. High means back pressure, often from memory.
  - **Occupancy Manager Target:** below 100% means the GPU is throttling occupancy to keep registers, threadgroup and stack memory on chip.
  - **Per-ALU-subunit limiters** (for example FP16): near 100% means that unit is saturated, so more occupancy won't help.
  - **L1 eviction rate** and L1 bandwidth/residency by memory type.
  - **Last-level cache limiter versus utilization:** a much higher limiter signals thrashing or memory back pressure.
  - **MMU limiter versus utilization:** a much higher limiter signals TLB misses from incoherent access.
  - **Definitions:** utilization = work ÷ peak. A limiter includes stalls as well as work.
  — [Apple Tech Talk 111374: Discover new Metal profiling tools for M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111374/)
- **Tools and workflow:**
  - Shader Cost Graph: a flame graph per pipeline plus per-line annotations with instruction counts by category.
  - Heat maps: shader execution cost, thread divergence, instruction count.
  - Shader Execution History.
  - Workflow: find expensive pipelines, check ALU limiters, then occupancy, then L1, then LLC/MMU, and iterate until ALU- or bandwidth-limited.
  — [Apple Tech Talk 111374](https://developer.apple.com/videos/play/tech-talks/111374/)
- API and Shader Validation, the Metal Debugger, the Metal Performance HUD and Metal System Trace all support Metal 4. — [WWDC25 Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)
- The Metal 4 debugger has a Dependency Viewer for barriers and events, an MTLTensor viewer, and an ML Network Debugger. — [WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/)
- Rigel observed bimodal results at 512³, which it attributed to GPU clock state. Benchmarks should warm up the GPU and report distributions. — [arXiv 2606.12765](https://arxiv.org/html/2606.12765v1)

### Inferences
- **Decode GEMV bandwidth check:** Time the kernel with command-buffer `GPUStartTime`/`GPUEndTime` or counter sample buffers. Compute weight bytes ÷ time and compare against peak.
- **Benchmark expectations:** Per-op rates of 250–440 GB/s in arXiv 2610.31479 suggest a high-end chip in that study. Calibrate against your own M4 Pro peak, which I did not verify in this pass (see Gaps).
- **Fusion signal:** A low launch limiter on small kernels suggests that kernel needs more threadgroups or should be fused.

### Gaps
- I did not fetch Metal System Trace documentation or a counter-sampling API reference (`MTLCounterSampleBuffer`, `sampleCountersInBuffer`).
- No primary source in this pass gave M4 Pro peak memory bandwidth. Apple's spec is 273 GB/s, but this is unverified here.

---

## High-quality open-source Metal kernels worth studying, and what makes them fast

### Takeaway
Study three codebases:
- **MLX** `mlx/backend/metal/kernels` for steel GEMM, `qmv_fast`/`qmv_wide`/`qmm`, `sdpa_vector`, and the NAX/MPP variants for M5.
- **llama.cpp** `ggml/src/ggml-metal` for per-quant-format `mul_mv` and `mul_mm` kernels, flash attention with quantized KV including `fa_vec_*`, a graph fusion table, and residency-set and no-copy mmap handling.
- **metal-flash-attention** for register-spill-aware attention blocking and atomic-free backward.

The shared patterns:
- SIMD-group-centric work decomposition.
- fp32 accumulation in registers.
- Function-constant or template specialization.
- Coalesced packed-weight loads with in-register dequantization.
- Concurrent encoders with minimal barriers.
- Aggressive fusion.

### Cited Findings
- **MLX kernel inventory:**
  - `quantized.metal` / `quantized.h`: affine quant, qmv/qmm.
  - `fp_quantized*.metal`: fp4/fp8/MX formats.
  - `gemv.metal` and `steel/gemm` (`mma.h`, `loader.h`, `gemm_nax.h`).
  - `steel/attn` (`attn.h`, `nax.h`), plus `sdpa_vector.h` and `sdpa_blocked.h`.
  - `rms_norm.metal`, `rope.metal`.
  - `gated_delta_update*.metal` for linear-attention models.
  — [mlx/backend/metal/kernels](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels)
- **llama.cpp kernel inventory:**
  - `mul_mv.metal` and `mul_mv_mma.metal`: GEMV.
  - `mul_mm.metal`: GEMM, with 64×32 tiles using simdgroup_matrix.
  - `fa_vec_{f16,q4_0,q8_0,...}.metal`: decode flash attention over quantized KV.
  - `fa_tensor.metal`: an MPP path.
  - `dequantize.h`, `ssm.metal`, `gated_delta_net.metal`.
  — [ggml-metal/kernels](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal/kernels)
- llama.cpp has a quirk: a quantized V cache requires flash attention on Metal. Mixed q8_0/q4_0 K/V configurations failed in one report. — [llama.cpp issue #21450](https://github.com/ggml-org/llama.cpp/issues/21450)
- MFA design points: intentional spilling at large D, skewed block aspect ratios, an atomic-free 7-GEMM backward, JIT compilation, and 82–94% forward ALU utilization on M3. — [metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- metal-benchmarks is the reference for Apple 7/8 microarchitecture: instruction latencies, `simdgroup_matrix` throughput, cache sizes and occupancy tables. Its local clone also includes a `CommandConcurrency` benchmark. — [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- On M4 Max, a fused scalar flash attention with online softmax was about 9× slower than a decomposed path that keeps simdgroup_matrix GEMMs. Matrix-unit and split-K variants were still 3.6–5× slower up to seq 8192. The authors blame small tiles and barriers. A naive fused FA is therefore worse than well-tiled separate GEMMs. — [arXiv 2606.12765](https://arxiv.org/html/2606.12765v1)

### Inferences
- **Starting point:** Port or reimplement MLX's `qmv_fast` (decode GEMV), `sdpa_vector` (decode attention), and steel GEMM (prefill) for M4 Pro. Use llama.cpp's fusion list as a checklist of decode fusions worth doing.
- **Attention:** Use the MLX `sdpa_vector` design for decode. Use an MFA-style or steel_attn tiled kernel for prefill, with careful block sizing. The Rigel result shows that poorly tiled fusion loses.

### Gaps
- I did not deeply read llama.cpp's `mul_mv` per-format kernels or the `ggml-metal-tuning.cpp` heuristics.
- I did not quantify MLX's versus llama.cpp's achieved bandwidth on M4 Pro from a primary benchmark.
- I did not survey other engineering blogs on hand-written Metal LLM kernels. BaseRT (vendor) and Open-TQ-Metal ([arXiv 2604.16957](https://arxiv.org/pdf/2604.16957)) exist but were not read in full.
