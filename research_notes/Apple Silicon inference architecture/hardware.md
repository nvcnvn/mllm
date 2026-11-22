# Apple M-series SoC hardware for on-device LLM inference (M1 → M5/M6, focus on M4 Pro Mac mini)

Target machine: Mac mini (2024), M4 Pro, 14-core CPU (10P + 4E), 20-core GPU, 64 GB LPDDR5X, 273 GB/s.
Research date: 2026-11-21. Labels used throughout: **[Apple]** = vendor claim/spec, **[measured]** = third-party measurement, **[RE]** = reverse-engineered/community, **[derived]** = arithmetic from cited numbers, **[spec?]** = speculative/unverified.

**Landscape as of Nov 2026 (important context):** M5 (Oct 2025), M5 Pro/Max (Mar 2026), M5 Ultra (Aug 2026, Mac Studio), and **M6** (Aug 2026, Mac mini) have all shipped. The Mac mini line moved to M6 / M5 Pro on 2026-08-25, replacing the M4/M4 Pro models — the target M4 Pro Mac mini is now the previous generation. — [MacRumors](https://www.macrumors.com/2026/08/25/apple-announces-2026-mac-mini/); [Apple Newsroom, Aug 2026](https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/)

---

## 1. Unified memory architecture, bandwidth, SLC, IOMMU/coherency, zero-copy, GPU wired limit

### Takeaway
CPU, GPU and ANE all hang off one LPDDR pool via the SoC fabric and a shared System Level Cache; the M4 Pro's 273 GB/s comes from LPDDR5X-8533 on a 256-bit (16×16-bit channel) bus, which caps LLM decode at roughly `273 GB/s ÷ bytes-of-weights-read-per-token`. By default macOS lets the GPU wire roughly 75% of RAM (~48 GB on a 64 GB machine, community-reported); `iogpu.wired_limit_mb` raises it.

### Cited Findings

**Memory bandwidth / type per chip**

| Chip | Memory | Bandwidth | Max RAM | Source |
|---|---|---|---|---|
| M1 | LPDDR4X | ~68 GB/s (68.25) | 16 GB | [Wikipedia M3 comparison table](https://en.wikipedia.org/wiki/Apple_M3) (lists "68 GB/s") |
| M1 Pro / M2 Pro | LPDDR5 | 200 GB/s | 32 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) ("200 GB/sec for its predecessors") |
| M1 Max / M2 Max | LPDDR5 | 400 GB/s | 64 / 96 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) ("400 GB/sec for all models of the M1 and M2 Max") |
| M1 Ultra / M2 Ultra | LPDDR5 | 800 GB/s | 128 / 192 GB | **unsourced in this session** — from prior knowledge of Apple specs; verify |
| M2 | LPDDR5 | ~100 GB/s (102.4) | 24 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) (lists 102 GB/s) |
| M3 | LPDDR5-6400 | 102.4 GB/s | 24 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) |
| M3 Pro | LPDDR5-6400 | 153.6 GB/s (Apple: "150") | 36 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) |
| M3 Max | LPDDR5-6400 | 307.2 (30-core GPU) / 409.6 (40-core) | 96 / 128 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) |
| M3 Ultra | LPDDR5-6400 | 819 GB/s | 256 / 512 GB | [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3) |
| M4 | LPDDR5X-7500, 8 controllers | 120 GB/s | 32 GB | [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4) |
| **M4 Pro** | **LPDDR5X-8533, 16 controllers** | **273 GB/s** | **64 GB** | [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4); [MacRumors (Apple: "273GB/s … 75% increase … vs M3 Pro")](https://www.macrumors.com/2024/10/29/apple-announces-m4-pro-chip) |
| M4 Max | LPDDR5X-8533, 32 controllers | 410 (32-core GPU) / 546 (40-core) | 128 GB | [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4) |
| M5 | LPDDR5X-9600, 8 controllers | 153.6 GB/s | 32 GB | [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5); Apple MLX blog says "153GB/s for the M5" [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) |
| M5 Pro | LPDDR5X-9600 | 307 GB/s | 64 GB | [Apple Newsroom Mar 2026](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/) |
| M5 Max | LPDDR5X-9600 | 460 (32-core GPU) / 614 (40-core) | 128 GB | [Apple Newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/); [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5) |
| M5 Ultra | LPDDR5X-9600, 64 controllers | 1,228.8 GB/s | 512 GB (96/256/512 options) | [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5) (secondary; Apple newsroom not fetched) |
| M6 | ? | 153 or 170 GB/s ("up to 170GB/s") | 32 GB | [Apple Newsroom Aug 2026](https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/); [Apple Mac mini specs](https://www.apple.com/mac-mini/specs/) |

- M4 Pro bandwidth **confirmed at 273 GB/s** — [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4); [MacRumors / Apple announcement](https://www.macrumors.com/2024/10/29/apple-announces-m4-pro-chip). Wikipedia notes each memory controller is a 16-bit channel; 16 controllers → 256-bit bus [derived] → 32 B × 8.533 GT/s = 273.1 GB/s [derived].
- M5 Pro: 307 GB/s = 32 B × 9.6 GT/s (same 256-bit width, faster DRAM) [derived from Wikipedia M5 LPDDR5X-9600].
- The MLX team attributes M5's +19–27% decode speed over M4 to bandwidth: "120GB/s for the M4, 153GB/s for the M5, which is 28% higher" — [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5). (Strong empirical evidence decode is bandwidth-bound on Apple GPUs.)
- On M5 Pro/Max the Neural Engine gets a "higher-bandwidth connection to memory" — [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5) (paraphrasing Apple).
- M5 Pro/Max use "Fusion Architecture" that "combines two dies into a single system on a chip"; M5 Ultra is quad-die "UltraFusion" — [Apple Newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/); [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5). (M4 Pro is a monolithic die; M1/M2/M3 Ultra are 2× Max via UltraFusion.)

**System Level Cache (SLC) / GPU cache hierarchy**
- SLC (GPU "L3") sizes measured/estimated by Philip Turner: M1 8 MB, M1 Pro 24 MB, M1 Max 48 MB, M1 Ultra 96 MB; M2 8 MB, M2 Pro 24 MB, M2 Max 48 MB. GPU L2: M1 768 KB, M1 Pro 256 KB, M1 Max 512 KB, M1 Ultra 1 MB, M2 ~1.5 MB, M2 Pro ~512 KB, M2 Max ~1 MB — [metal-benchmarks README](https://github.com/philipturner/metal-benchmarks) (local: `/Volumes/T5/Codes/metal-benchmarks/README.md`).
- SLC bandwidth ≈ 15.4–19.8 B/cycle vs RAM 7.7–9.9 B/cycle per GPU core (i.e. SLC ≈ 2× DRAM bandwidth) — [metal-benchmarks](https://github.com/philipturner/metal-benchmarks).
- **M4 Pro SLC size: no primary source found.** A Reddit comment estimates 32 MB — unverified [spec?] — [r/hardware thread via search](https://lr.eu.psf.lt/r/hardware/comments/1gyh42k/david_huang_tests_apple_m4_pro).
- Chips and Cheese (M2 Pro): GPU L2 miss → SLC latency 234 ns; DRAM >342 ns; 256-bit bus >200 GB/s — [Chips and Cheese, M2 Pro iGPU](https://chipsandcheese.com/2023/10/31/a-brief-look-at-apples-m2-pro-igpu/).
- On M4 (base), SME load bandwidth from one P-core drops sharply above 8 MiB, suggesting a ~8 MiB-range L2 effect — [Uni Jena SME microbenchmarks](https://scalable.uni-jena.de/opt/sme/micro.html). (Note: that is CPU-cluster L2 on base M4, not SLC.)

**Page tables, IOMMU (DART), GPU MMU (UAT), coherency**
- Each non-CPU DMA master sits behind an Apple **DART** IOMMU; the M1 DART is hardwired to **16 KB pages** — [LKML: Apple M1 DART IOMMU driver v4](https://lkml.indiana.edu/hypermail/linux/kernel/2106.3/02533.html). M1 Pro/Max/Ultra DARTs use a different PTE format — [lore.kernel.org](https://lore.kernel.org/all/0076bef7-9d3c-b5e9-00dc-a350f96e9e5f@arm.com/).
- The GPU (AGX) has its own MMU, **UAT**, with ARM64-like page tables shared between the kernel driver and GPU firmware; GPU pages are always 16 KB — [Asahi Linux AGX docs](https://asahilinux.org/docs/HW-AGX); [LWN: UAT io-pgtable](https://lwn.net/Articles/920839). Only 16K mode, GPU huge pages not enabled in the Linux driver — [LKML review](https://lkml.rescloud.iu.edu/2301.2/09318.html).
- macOS userspace on Apple silicon also uses 16 KB pages (well known; `sysctl hw.pagesize` → 16384) — **not re-verified this session**.
- CPU↔GPU coherency: Metal `MTLStorageModeShared` buffers are visible to both CPU and GPU without copies; Apple docs say `makeBuffer(bytesNoCopy:…)` requires a page-aligned pointer and page-multiple length — [Apple Metal docs](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)) (**content from prior knowledge, page not fetched this session**).

**GPU-wireable memory limit**
- `MTLDevice.recommendedMaxWorkingSetSize` reports the GPU working-set budget — [Apple docs](https://developer.apple.com/tutorials/data/documentation/metal/mtldevice/recommendedmaxworkingsetsize.md); [Apple dev forums thread](https://developer.apple.com/forums/thread/732035).
- Default ≈ 75% of RAM is **community-reported, not Apple-documented**; a 64 GB Mac typically allows ~48 GB — [mlx-lm issue #883](https://github.com/ml-explore/mlx-lm/issues/883); [devnote-override-macos-metal-vram-cap](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap).
- Tunable: `sudo sysctl iogpu.wired_limit_mb=<MB>` (Sonoma+; older macOS used `debug.iogpu.wired_limit` in bytes); `0` = default; resets on reboot unless persisted — [devnote](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap); [llmconfigurator guide](https://llmconfigurator.com/en/guides/troubleshooting/increase-metal-vram-limit-apple-silicon); [OSXDaily](https://osxdaily.com/2025/05/07/how-to-increase-vram-allocation-on-apple-silicon-mac/).
- Raising the sysctl does not guarantee apps use it (LM Studio still capped ~64 GB after raising to ~124 GB) — [LM Studio bug #651](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/651).
- MLX's allocator: GC limit = `min(memory_limit, 0.95 × recommendedMaxWorkingSetSize)`; `mx.set_wired_limit()` warns if set above `max_recommended_working_set_size`; uses Metal **residency sets** to keep weights wired — [mlx allocator.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp) (local: `/Volumes/T5/Codes/mlx/mlx/backend/metal/allocator.cpp` lines 85–103, 266–273); [mlx docs: set_wired_limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html).

### Inferences
- **Decode roofline on M4 Pro [derived]:** tok/s ≤ 273 GB/s ÷ bytes read per token. 8B @ 4-bit (~4.5–5 GB) → ≤ ~55–60 tok/s; 32B @ 4-bit (~18 GB) → ≤ ~15 tok/s; 70B @ 4-bit (~40 GB) → ≤ ~6.8 tok/s, and 40 GB already approaches the ~48 GB default GPU wired limit (KV cache must also fit). Real engines typically hit 60–80% of this bound.
- "Zero-copy" in practice means: weights mmap'd from disk can be wrapped as shared Metal buffers (16 KB-aligned) with no copy and no PCIe transfer; CPU and GPU see the same physical pages. Limits: (a) GPU-resident set capped by the wired limit; (b) 16 KB alignment/size granularity; (c) coherency is at command-buffer boundaries — the CPU must not mutate a buffer the GPU is reading in an in-flight command buffer (synchronization still needed); (d) ANE does not see Metal buffers directly — it uses IOSurface-backed tensors in a fixed layout (see §4), so CPU/GPU↔ANE handoff often involves layout conversion.
- M5 Pro (same 64 GB ceiling, 307 GB/s) gives only ~12% more decode bandwidth than M4 Pro; its big win is prefill (Neural Accelerators, §2).

### Gaps
- No primary source for M4 Pro SLC size, M4 GPU L2 size, or measured M4 Pro DRAM latency (Chips and Cheese has no M4 Pro GPU article; Geekerwan videos not checked).
- Measured achievable (STREAM-like) bandwidth on M4 Pro from the GPU vs the 273 GB/s spec not found.
- Apple has never documented the default wired-limit formula; 75% is community observation (some report it differs on very-low/high RAM configs).
- M1/M2 Ultra 800 GB/s not re-verified this session.

---

## 2. GPU microarchitecture & throughput (incl. M3 Dynamic Caching, M5 Neural Accelerators)

### Takeaway
Apple GPU cores have 128 FP32 ALUs, 32-wide SIMD-groups, ~208 KB register file and ~60 KB threadgroup memory (M1/M2 measurements); FP16 runs at the *same* FLOP rate as FP32 (pre-M5), so M4 Pro's GPU is roughly ~8 TFLOPS-class FP16/FP32 [spec?]. M3+ replaced fixed register/threadgroup partitions with a dynamically allocated on-chip cache ("Dynamic Caching"). M5's per-core Neural Accelerators (matrix units) give ~3.3–4× faster prefill/TTFT but only ~19–27% faster decode vs M4.

### Cited Findings

**Per-core microarchitecture (Apple GPU family 7/8 = M1/M2, measured by Philip Turner)** — all from [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) (local `/Volumes/T5/Codes/metal-benchmarks/README.md`):
- 128 ALUs per core (Apple's description); 4 schedulers per core; SIMD-group = **32 threads**.
- Register file **~208 KB/core**; threadgroup ("shared") memory **~60 KB/core** (Metal exposes 32 KB max per threadgroup — [not re-verified]); L1 data cache 8 KB; instruction cache 12 KB; 128 B cache line; 384–3,072 threads/core; up to 24 simds/core for ALU utilization.
- SIMD shuffle bandwidth 256 B/cycle (2× AMD/Nvidia); on-core data bandwidth 64 B/cycle. Turner notes Apple deliberately gives *less* threadgroup memory bandwidth and invests in shuffle bandwidth and matrix (`simdgroup_matrix`) instructions; recommends using the large register file as "extended threadgroup memory" for FlashAttention-style kernels.
- Throughput per core-cycle: FP32 FMA = 256 ops (128 FMA), **FP16 FMA = 256 ops (same as FP32)**, I32 add 128, I32 mul 32, I16 mul 32. `simdgroup_matrix` 8×8: FFMA16 ≈ 102.5 and FFMA32 ≈ 101.7 FMAs/core-cycle (~80% of scalar FMA peak). MATMUL<8x8xF16> latency ~34 cycles.
- Per-chip GPU peak (GFLOPS32 = GFLOPS16): M1 2,617 (8 cores @1.278 GHz); M1 Pro 5,308; M1 Max 10,617; M1 Ultra 21,233; M2 3,579 (10 cores @1.398 GHz); M2 Pro 6,800 (19 cores); M2 Max 13,600 (38 cores).
- M1 Max can run up to 96 compute commands concurrently.
- No int8 dot-product hardware is documented in these tables (I16/I32 mul are slow: 32/cycle).

**M3 / A17 Pro GPU (Apple family 9) — Dynamic Caching** — [Apple Tech Talk "Explore GPU advancements in M3 and A17 Pro"](https://developer.apple.com/videos/play/tech-talks/111375/):
- Before family 9, a SIMD-group had to allocate its *peak* register count before launch, capping occupancy. Now "On-chip register memory is now dynamically allocated and deallocated over the lifetime of the shader"; "the register file is now a cache instead of the permanent storage for the registers."
- On-chip memories redesigned "into fewer larger caches that service all these memory types" (registers, threadgroup, tile, stack, buffer). Threadgroup and tile memory become caches too.
- Hardware monitors shaders and lowers occupancy automatically if the working set would spill.
- FP16: "Apple GPUs are highly optimized to execute FP16 arithmetic … FP16 math instructions execute at peak throughput. They use fewer registers than their FP32 equivalents."
- Family 9 can execute FP32, FP16 and integer instructions from different SIMD-groups in parallel: "up to 2x ALU performance compared to prior Apple GPUs" (workload-dependent).
- Talk gives **no** cache sizes, register-file sizes, or TFLOPS.

**M4 family GPU**
- GPU cores: M4 8/9/10; **M4 Pro 16/20**; M4 Max 32/40 — [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4). Apple publishes no TFLOPS. M4 (base) measured peak **2.9 FP32 TFLOPS** on GPU — [arXiv 2502.05317 "Apple vs. Oranges"](https://arxiv.org/html/2502.05317v1).
- Metal-benchmarks covers only M1/M2 — no measured M3/M4 per-core register/threadgroup numbers — [metal-benchmarks](https://github.com/philipturner/metal-benchmarks).

**M5 family GPU + Neural Accelerators**
- Each GPU core contains a Neural Accelerator: M5 10 (or 8) cores; M5 Pro 16/20; M5 Max 32/40; M5 Ultra 64/80 — [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5); [Apple Newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/).
- Apple claims [Apple]: M5 Pro/Max "more than four times the peak GPU compute for AI" vs M4 Pro/M4 Max; LLM prompt processing M5 Pro up to **3.9× vs M4 Pro** (6.9× vs M1 Pro); M5 Max up to 4× vs M4 Max — [Apple Newsroom Mar 2026](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/); [AppleInsider](https://appleinsider.com/articles/26/03/03/how-m5-pro-and-m5-max-push-macbook-pro-into-high-bandwidth-ai-era).
- Measured by Apple's MLX team [Apple-measured], M5 vs M4 base, 4096-token prompt: TTFT speedup 3.57× (Qwen3-1.7B bf16), 3.62× (Qwen3-8B bf16), 3.97× (8B 4-bit), 4.06× (14B 4-bit), 3.33× (gpt-oss-20b MXFP4), 3.52× (Qwen3-30B-A3B 4-bit); generation speedup only 1.19–1.27×. "The Neural Accelerators provide dedicated matrix-multiplication operations"; MLX uses Metal 4 **TensorOps** and **Metal Performance Primitives** — [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- MLX gates its "NAX" (Neural Accelerator) kernel paths on macOS ≥ 26.2 and GPU architecture generation ≥ 17 (≥ 18 for 'p'-suffixed architectures) — [mlx device.cpp `is_nax_available`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp) (local `/Volumes/T5/Codes/mlx/mlx/backend/metal/device.cpp` ~L1046–1063). NAX-specific code exists in matmul, quantized matmul, SDPA kernels (same dir).
- M6 (Mac mini, Aug 2026): 12-core GPU with Neural Accelerators; Apple claims up to 4.8× faster LLM prompt processing (LM Studio) vs M4 and "up to 4x faster AI performance" — [Apple Newsroom Aug 2026](https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/).

### Inferences
- **M4 Pro GPU peak [spec?/derived]:** 20 cores × 256 FLOP/cycle × GPU clock. Assuming ~1.5–1.6 GHz (M4-generation clocks not sourced here) → ~7.7–8.2 TFLOPS FP32 = FP16. Cross-check: base M4 measured 2.9 TFLOPS on 10 cores; doubling the cores gives ≈ 5.8 TFLOPS. 2.9 TFLOPS on 10 cores implies only ~1.13 GHz × 256 FLOP/cycle, so either the benchmark falls short of peak or the clock assumption is wrong. Treat M4 Pro as **~6–8 TFLOPS FP16/FP32** effective.
- No BF16 or INT8 speedup on M1–M4 GPUs: BF16 is at best FP16-rate (often emulated/converted), INT8 has no dot-product path; quantized weights (4-bit/8-bit) help only by cutting bandwidth, dequantized to FP16 in-register. This makes 4-bit weight-only quantization + FP16 compute the natural design point on M4 Pro.
- **Prefill on M4 Pro is compute-bound at roughly 8 TFLOPS** [derived]: an 8B model needs ~16 GFLOP/token, so ~300–450 tok/s ideal prefill at 60–80% utilization; M5 Pro's Neural Accelerators should push this ~4×. Decode, being bandwidth-bound, gains only ~12% (273→307 GB/s).
- Kernel design implications: SIMD width 32, 8×8 `simdgroup_matrix` tiles, keep blocks in registers (large RF, small/slow threadgroup memory), FP16 accumulation where safe; on M3+/M4 occupancy is managed dynamically, so register pressure is less of a cliff than on M1/M2.

### Gaps
- No measured M4/M4 Pro GPU clock, register file size, threadgroup-memory size, or simdgroup_matrix throughput found (metal-benchmarks stops at M2; Chips and Cheese/Geekerwan M4 GPU data not located).
- No published TFLOPS for M5 Neural Accelerators (FP16/BF16/INT8 per core per cycle) — only Apple's "4×" claims and MLX TTFT measurements. Supported precisions of the Neural Accelerators not confirmed from a primary source.
- Whether M3/M4 added faster BF16 on the GPU: not found.

---

## 3. CPU side: P/E cores, AMX (M1–M3), SME/SME2 on M4, achievable matmul throughput

### Takeaway
Every M-series P-cluster has a shared matrix coprocessor; on M1–M3 it is reached only through undocumented AMX instructions (via Accelerate/BNNS), and on M4 it is additionally exposed as standard ARM SME/SME2 (SVL = 512 bits). One P-cluster's unit delivers ~2 TFLOPS FP32/FP16/BF16 and ~4 TOPS INT8 on M4 — but it is **shared per cluster**, so adding threads on the same cluster does not add throughput.

### Cited Findings

**Core layouts**
- M4: 3–4 P + 4–6 E; **M4 Pro: 8 or 10 P + 4 E (12/14 cores)**; M4 Max: 10/12 P + 4 E — [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4).
- M3: 4P/4E; M3 Pro 5P/6E or 6P/6E; M3 Max 10P/4E or 12P/4E; M3 Ultra 20P/8E or 24P/8E — [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3).
- M5 renames cores: base M5 = 4 "super" + 6 efficiency; M5 Pro = 5 super + 10 performance (15-core) or 6 + 12 (18-core); M5 Max 6 + 12; M5 Ultra 10+20 / 12+24. The Pro/Max "performance" cores are a new mid-tier design (no E-cores) — [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5); [Apple Newsroom](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/); [Mac mini specs](https://www.apple.com/mac-mini/specs/).
- M6 (Mac mini): 12-core CPU = 2 super + 4 performance + 6 efficiency per Apple specs page — [Mac mini specs](https://www.apple.com/mac-mini/specs/) (newsroom only says 12-core).
- M4 is "based on ARMv9.2a", supports **SME but not SVE** (LLVM therefore models it as ARMv8.7a + SME) — [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4).

**AMX (M1–M4, undocumented)** — [corsix/amx](https://github.com/corsix/amx) (local `/Volumes/T5/Codes/amx/`):
- AMX instructions are issued from the CPU but "executed on a special accelerator execution unit"; "neither documented nor supported by Apple"; distinct from Intel AMX. Research done on M1 Max, M2, M3 and M4 Max. M1→M2 added BF16; M2→M3 added modes to `ldx`/`ldy`/`matint`; M3→M4 changed some offset behaviors (AMX still present on M4) — `README.md`.
- M1 Max measured (no loads/stores, i.e. upper bound) — `fma.md`:
  - `fma32` matrix mode (f32 16×16 accumulators): 1 thread ×4 accumulators ≈ **1,475 GFLOPS**; 2 threads ≈ 2,957 GFLOPS (two P-clusters on M1 Max); max ≈ 3,119 GFLOPS at 6 threads.
  - `fma16` matrix mode (f16 32×32 accumulators): 1 thread ×2 acc ≈ 2,959 GFLOPS; 2 threads ≈ 5,916 GFLOPS; 6 threads ≈ 6,263 GFLOPS.
  - `fma64`: ~370 GFLOPS/thread, ~780 GFLOPS max.

**SME/SME2 on M4 (measured, base M4: 4P + 6E)** — [Uni Jena SME microbenchmarks](https://scalable.uni-jena.de/opt/sme/micro.html):
- SVL = 512 bits; four ZA tiles, each 16×16 FP32.
- Single P-core: FMOPA FP32 **2,008 GFLOPS**; FP16→FP32 2,010; BF16→FP32 (BFMOPA) 2,011; FP64 503; **SMOPA INT8→INT32 4,020 GOPS**; I16→I32 2,010. E-core: FP32 357, INT8 716.
- Need ≥4 ZA tiles in flight (1 tile 503, 2 tiles 1,005, 4 tiles 2,008 GFLOPS).
- SMSTART/SMSTOP overhead: only 434 GFLOPS with 8 FMOPAs per streaming block, 1,947 with 128 → keep long streaming-mode regions.
- **Multicore: 1–4 P threads ≈ 1,983–2,009 GFLOPS total (one shared SME unit for the P-cluster)**; adding E-threads reaches ~2,340; one P + one E thread = 2,371. Conclusion: two SME units (one per cluster).
- Comparisons per P-core FP32: Neon FMLA 107 GFLOPS (10 Neon threads → 636); SSVE FMLA 31; SME2 FMLA (multi-vector) 502; **AMX FMA 2,006 ≈ SME FMOPA 2,008** (same hardware).
- Load bandwidth into SME from 1 P-core: up to 925 GiB/s (LD1W ×4 regs) in-cache; drops above 8 MiB.
- Real GEMM 512×512 FP32 ≈ 1,755 GFLOPS on M4 — [Jena SME page (via search)](https://scalable.uni-jena.de/opt/sme); "Hello SME!" JIT kernels reach >2.3 TFLOPS FP32 on M4 — [arXiv 2409.18779](https://arxiv.org/pdf/2409.18779).
- Separate HPC study: M4 CPU ~1.5 FP32 TFLOPS (not SME-specific) vs GPU 2.9 — [arXiv 2502.05317](https://arxiv.org/html/2502.05317v1).
- Community project combining ARM SME2 assembly kernels with ANE training — [colonelpanichacks/neural-engine](https://github.com/colonelpanichacks/neural-engine).

### Inferences
- **M4 Pro has two P-clusters (10 P-cores, likely 5+5) [spec?]** — if each has its own SME unit, peak ≈ 2 × 2 TFLOPS FP32/FP16/BF16 + ~0.36 from the E-cluster ≈ **~4.3 TFLOPS**, INT8 ≈ ~8.7 TOPS. Cluster topology of M4 Pro was not confirmed from a source in this session. This is in the same ballpark as half the GPU's FP16 peak — useful for CPU-side prefill offload or small-batch work, but the matrix units share the same 273 GB/s DRAM, so CPU decode is no faster than GPU decode.
- SME widening FP16/BF16 runs at the FP32 rate (no 2× for half precision); only INT8 doubles throughput → CPU int8 GEMM (e.g. W8A8) is the CPU's best-case.
- Threading model for an engine: one SME-issuing thread per P-cluster (plus optionally one E-cluster thread); more threads per cluster just contend for the same unit. Neon-only kernels (e.g. llama.cpp's default quantized dot products) top out at ~0.6 TFLOPS on base M4.
- Accelerate (BLAS/BNNS) is the supported way to reach AMX on M1–M3 and AMX/SME on M4; direct SME is supported by Apple's compiler toolchain on M4 (inference: SME is an architectural ISA extension, so it is "legal" to use, unlike AMX).

### Gaps
- No source fetched on how Accelerate/BNNS internally dispatch to AMX vs SME on M4, or on BNNS Graph performance.
- No SME measurements on M4 Pro/Max specifically (cluster count and per-cluster SME unit not measured), nor on M5 (whether M5's super/performance cores have SME2 units, and their rate).
- M4 Pro P-core clocks (commonly cited ~4.5 GHz) not sourced this session.

---

## 4. Apple Neural Engine (ANE)

### Takeaway
The ANE is a 16-core fp16 engine reachable officially only through Core ML; its 38 TOPS (M4) is an INT8 marketing number — measured fp16 peak is ~19 TFLOPS, INT8 only saves bandwidth (disputed). Reverse-engineered private APIs (`_ANEClient`, `_ANECompiler`, MIL→E5) allow direct use, but static shapes, a fixed [1,C,1,S] fp16 IOSurface layout, ~32 MB on-chip SRAM, ~1 GB chunk limits and per-process compile limits make it a poor fit for large-model decode on a 64 GB M4 Pro.

### Cited Findings
- TOPS by generation [Apple]: M3 18 TOPS; M3 Ultra 36 TOPS — [Wikipedia M3](https://en.wikipedia.org/wiki/Apple_M3); M4 38 TOPS (16 cores) — [Wikipedia M4](https://en.wikipedia.org/wiki/Apple_M4); M5 42 TOPS, M5 Ultra 84 TOPS, all M5 variants 16-core — [Wikipedia M5](https://en.wikipedia.org/wiki/Apple_M5). M6: "Dual 16-core Neural Engine", "up to 2x faster" than previous gen, no TOPS — [Apple Newsroom Aug 2026](https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/). M5 Pro/Max: "faster and more power-efficient Neural Engine", no TOPS — [Apple Newsroom Mar 2026](https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/). (M1 11 TOPS / M2 15.8 TOPS: prior knowledge, not re-verified.)
- **Orion paper (M4 Max, H16 ANE)** — [arXiv 2603.06728](https://arxiv.org/html/2603.06728):
  - Apple 38 TOPS is INT8; **measured fp16 peak ~19 TFLOPS**; INT8 is dequantized to fp16 before compute, so it "saves only memory bandwidth, not compute cycles."
  - **32 MB on-chip SRAM**; throughput drops ~30% when the working set exceeds it.
  - Deep graphs (16–64 ops) reach ~94% utilization vs ~30% for single ops; 1×1 convolutions ~3× faster than equivalent matmuls.
  - Native precision fp16 (±65,504). All tensor I/O in a fixed **[1, C, 1, S] fp16 layout in IOSurface-backed memory**; min IOSurface ~49 KB → sequence padded to ≥16. Convolutions with 32,000 output channels (vocab projection) rejected. Prefill uses bucketed sequence lengths (32–1024) — i.e. static shapes.
  - ~**119 compilations per process** before compiles silently fail; first call compiling 24 programs ≈ 1,015 ms; eval queue depth 127; dispatch overhead ~0.095 ms (XPC + IOKit; another figure says ~0.03 ms); IOSurface round-trip ~2.3 ms per dispatch.
  - GPT-2 124M decode: CPU 283 tok/s vs ANE 170 tok/s; authors state GPU/MLX "currently achieves higher absolute throughput" than ANE. Idle power zero via hard power gating.
  - Private APIs: `_ANEClient`, `_ANECompiler` (MIL → E5 microcode), `_ANEInMemoryModel(Descriptor)`, `_ANEModel`, `_ANERequest`, `_ANEIOSurfaceObject`, from `/System/Library/PrivateFrameworks/AppleNeuralEngine.framework`.
- maderix/ANE: training via reverse-engineered private APIs without Core ML/Metal; reports ~15.8–18.6 TFLOPS fp16 and **35.1 TOPS INT8 (1.88× fp16)** on M4 — [maderix/ANE](https://github.com/maderix/ANE). **Conflicts** with Orion's "INT8 = no compute gain" — unresolved [RE].
- colonelpanichacks/neural-engine reports ~18.5 TOPS fp16 on M4 — [GitHub](https://github.com/colonelpanichacks/neural-engine).
- M1 Max ANE ~12 fp16 TFLOP/s; int4 LUT weights ~2.37× faster than fp16 — [arXiv 2606.22283 "Apple Neural Engine: Architecture, Programming, and Performance"](https://arxiv.org/pdf/2606.22283). Weight encoding affects LM placement/perf on ANE — [arXiv 2608.22110](https://arxiv.org/pdf/2608.22110) (not read in detail).
- Anemll (open-source LLM→ANE via Core ML): models split into chunks (embeddings / FFN chunks / LM head); chunk calculator targets ≤ ~950 MB per chunk; LUT (palettized) 4–6-bit weights; context up to 4K, "512-2048 recommended for optimal ANE performance"; FP16 overflow issues (e.g. Gemma 3 activations > 65,504); in-model argmax to reduce ANE→host transfer; IOSurface-backed buffers and serialized prediction queue to avoid ANE race conditions — [Anemll README](https://github.com/Anemll/Anemll) and [docs/calc_chunk_split.md](https://github.com/Anemll/Anemll/blob/main/docs/calc_chunk_split.md) (local `/Volumes/T5/Codes/Anemll/`).
- Apple's own ANE transformer guidance repo (channels-first [B,C,1,S] layout, 1×1 conv in place of linear) — [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers) (local clone present; not re-read this session).

### Inferences
- M4's 38 TOPS vs M3's 18 TOPS is largely a change of counting basis (INT8 vs FP16) — measured fp16 ~19 TFLOPS on M4 ≈ M3's 18 TOPS figure [derived from Orion + Wikipedia; Apple has not stated this].
- For the M4 Pro target: ANE fp16 (~19 TFLOPS claimed-measured) exceeds the GPU's ~6–8 TFLOPS on paper, so ANE is attractive for **prefill of small/medium models** or as a power-efficient co-processor, but (a) static shapes force bucketed prefill and fixed KV-cache sizes, (b) ~1 GB per-model chunks mean many chunks for 8B+ models, (c) bandwidth is the same shared DRAM (and ANE's path may be narrower than the GPU's on M4 — Apple only advertises a "higher-bandwidth connection" starting M5 Pro/Max), (d) decode results (GPT-2: ANE slower than CPU) suggest dispatch overhead dominates small per-token work. A GPU-first engine with optional ANE prefill/draft-model offload is the realistic design.

### Gaps
- ANE memory bandwidth (achievable GB/s) on M4/M4 Pro not found.
- Whether ANE on M4 Pro equals M4 (16 cores, same clock) — assumed, not sourced.
- INT8 compute speedup on ANE: sources conflict (maderix 1.88× vs Orion none).
- No authoritative max model size per Core ML ANE program; the ~950 MB chunk target is Anemll's practice, not an Apple-documented limit.

---

## 5. Power and thermals (Mac mini M4 Pro, sustained inference)

### Takeaway
Apple rates the M4 Pro Mac mini at 5 W idle / 140 W max wall power; measured sustained combined CPU+GPU load is only ~50 W, and the small single-fan chassis stays quiet at moderate loads. LLM decode (bandwidth-bound) should sit well under that envelope; no review measuring sustained LLM inference throttling on this machine was found.

### Cited Findings
- Wall power (includes PSU loss) [Apple]: Mac mini 2024 M4 (16 GB) 4 W idle / 65 W max; **M4 Pro (64 GB, 8 TB) 5 W idle / 140 W max, 17 / 478 BTU/h**; Mac mini M5 Pro 6 W / 145 W; Mac mini M6 4 W / 70 W — [Apple Support: Mac mini power consumption](https://support.apple.com/en-us/103253). Max continuous power rating for the platform: 155 W — [Mac mini specs](https://www.apple.com/mac-mini/specs/).
- Measured: M4 Pro (20-core GPU) Mac mini typical sustained power with CPU and GPU tests running together ≈ **50–52 W** (SoC-reported) — [Eclectic Light, "M4 Pro full on"](https://eclecticlight.co/2025/01/22/m4-pro-full-on-when-cpu-and-gpu-draw-over-50-w-and-how-low-power-mode-changes-that/). The article also examines how Low Power Mode cuts this (details not extracted).
- Notebookcheck: M4 Pro CPU sustained/PL1 ≈ 40 W; Mac mini "virtually silent even under average load", fan rarely spins up — [Notebookcheck M4 Pro Mac mini review](https://www.notebookcheck.net/Apple-Mac-Mini-M4-Pro-review-The-compact-and-frugal-desktop-PC-with-top-performance-and-expensive-upgrades.946883.0.html).
- Forum/user reports (low quality): base M4 mini fan ~2,970 RPM and ~35–40 dB(A) under full CPU+GPU load — [Apple Community](https://discussions.apple.com/thread/255913357); complaints of high-pitched fan noise on some M4 Pro minis — [MacRumors forums](https://forums.macrumors.com/threads/mac-mini-m4-pro-loud-fan-noise-and-thoughts.2442548/).
- Apple says the M5 Pro Mac mini is quiet and efficient for always-on use ("industry-leading performance per watt") — [Apple Newsroom Aug 2026](https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/).

### Inferences
- Per-token energy [derived]: if GPU-bound decode draws ~30–50 W while delivering ~15 tok/s on a 32B 4-bit model, that is ~2–3 J/token; at ~55 tok/s on an 8B model, ~0.6–0.9 J/token. These are estimates, not measurements.
- The 140 W max includes Thunderbolt/USB bus power and SSD; SoC-only sustained draw (~50 W) suggests the M4 Pro mini is unlikely to thermally throttle on inference workloads, but prefill (compute-bound GPU + optional SME) is the scenario most likely to approach the SoC power limit. Monitor with `powermetrics` (GPU/CPU/ANE power, frequency) during long runs.
- Low Power Mode will reduce GPU clocks/power — relevant for an always-on server; Eclectic Light article is the source to consult for magnitude.

### Gaps
- No measured GPU-only sustained power, GPU clock behavior under long LLM runs, or thermal throttling data for M4 Pro Mac mini.
- No ANE power figures under load (Orion only notes zero idle via power gating).
- No measured tokens-per-joule for M4 Pro on standard LLM benchmarks.
