# Memory design for M-series (v1)

Target: Mac mini M4 Pro, 64 GB. Numbers measured on this machine unless marked *(estimate)* or *(to verify)*.

## Layer 1 — what the hardware and OS give us

### Physical memory
- One pool of LPDDR5X, **273 GB/s peak**, shared by CPU, GPU and Neural Engine. GPU reaches **210–238 GB/s** on large sequential reads (measured).
- Cache line: **128 bytes** (`hw.cachelinesize`). Smallest unit the memory system moves.
- CPU: L1d is 128 KiB per P-core, 64 KiB per E-core; **two P-clusters of 5 cores, 16 MiB L2 each**; E-cluster 4 cores, 4 MiB L2.
- A System Level Cache sits between all units and DRAM (size unpublished, ~32 MiB *(estimate)*). Anything bigger streams from DRAM.

### Virtual memory
- OS page = **16 KiB** (`pagesize`). The GPU's MMU also maps 16 KiB pages.
- A Metal buffer in `StorageModeShared` is one range of physical pages mapped into **both** the CPU's and the GPU's page tables. Same bytes, no copy.
- Visibility is not instant: CPU writes made before a command buffer is committed are visible to that command buffer; GPU writes are visible to the CPU after it completes. In between, the CPU must not write bytes an in-flight command buffer reads.
- `HazardTrackingModeUntracked`: Metal does not order accesses for us. We do (one queue, explicit barriers).

### Limits on this machine
| Limit | Value | Consequence |
|---|---|---|
| `recommendedMaxWorkingSetSize` | 55.7 GB | Everything the GPU touches (weights + arena + workspace) must fit |
| `maxBufferLength` | 41.7 GB | One `MTLBuffer` can't exceed this → arena ≤ 41.7 GB, or several buffers |
| `iogpu.wired_limit_mb` | 0 (default) | Raise only if 55.7 GB isn't enough |
| Internal SSD free | 34 GiB | Kept for swap. Model files and the spill tier go on `/Volumes/T5` (USB) by default |

### Costs that shape the design
- Creating a buffer = syscalls + zero-filling + GPU page-table updates. **Allocate once at startup, never in the token loop.**
- First touch of a fresh page is a page fault. **Pre-touch the arena at startup** (`memset`).
- GPU memory must stay resident while the GPU uses it. The arena and weights go in `MTLResidencySet`s, and `requestResidency` is repeated on a timer, because the OS can take residency back (llama.cpp does the same). A residency set is **not a pin**: under memory pressure the OS evicted idle buffers in a set, both file-backed and anonymous (measured, `tools/residency_check.swift`). Metal pages them back in before the next command buffer runs, which stalls it (4 GiB: ~5 s from the USB volume, ~0.6 s from swap/compressor).
- Weight files are `mmap`ed and wrapped with `newBufferWithBytesNoCopy` (address and length 16 KiB-aligned). Once resident they are not read from disk again, so the files can live on the USB volume: the cost is load time only (~30–60 s cold for 30 GB *(estimate)*).
- Runs of 4 KiB or more at random slots read at full stream speed, ~245 GB/s (measured, `tools/kv_gather_bench.swift`). Scattered 128-byte reads are untested.

## Layer 2 — our layout

### Regions
| Region | Contents | Lifetime | Owner |
|---|---|---|---|
| Weights | `mmap`ed model files (4-bit, MLX format) | Process | GPU reads |
| ANE weights | Core ML model (4-bit LUT) | Process | Core ML / Neural Engine |
| **Arena** | Per-token and per-request state | Process; contents churn | This design |
| Workspace | Fixed activation buffers per chunk size (ANE I/O in IOSurfaces, GPU scratch). ANE prefill's second chunk in flight adds only its own inputs / outputs (~16 MiB at chunk 512, 0.8B) and a second IOSurface set; it shares the GPU scratch (~335 MiB at max_ctx 16384) | Process | Reused every step |
| Step tables | Small `uint32` arrays the GPU reads each step, **2 copies** | Process | CPU writes, GPU reads |

### The arena: one buffer, one unit size
```
slot     0       1       2       3       4       5      …     N−1
     ├───────┼───────┼───────┼───────┼───────┼───────┼── … ──┼───────┤
       full    window  free    window  full    window          free
               └──── one window page = 3 units, anywhere ────┘
```
- The arena is **N equal units**. Slot `s` starts at byte `s × unit`.
- The unit is a multiple of 16 KiB, fixed when the model loads.
- Every kind of state is a whole number of units. The units of one object need not be adjacent, because the GPU finds each one through the step table.
- Any free unit serves any need. No bands, no ratio setting, nothing to compact.
- A model with three kinds of state (full + window + linear) needs nothing extra.

### A unit, in bytes
Example model: 8 full-attention layers, 24 window layers, 8 KV heads, head size 128, fp16, **32 tokens per page**.
```
row   = 128 dims × 2 B        = 256 B     one token, one head
block = 32 rows               = 8 KiB     one (layer, K/V, head)
layer = 2 (K,V) × 8 heads     = 128 KiB
unit  = 8 layers              = 1 MiB     (64 OS pages)

offset_in_unit(l, kv, head, token, dim) =
    ((l × 2 + kv) × 8 + head) × 8192      ← l = layer's position inside the unit, 0–7
    + token × 256
    + dim × 2
```
- Each (layer, head) is a **contiguous 8 KiB run**: the attention kernel reads 64 cache lines in a row.
- Unit size rule: one unit holds `G` layers, and `G` defaults to the number of full-attention layers, so a full-attention page is exactly one unit with no waste. Any divisor of that number is also legal.
- 8-bit KV: 1-byte values plus one fp16 scale per 128-byte row, stored at the end of each block. *Later.*

### Objects built from units
| Object | Units | Notes |
|---|---|---|
| Full-attention page (32 tokens, 8 layers) | 1 | Grows with the request |
| Window page (32 tokens, 24 layers) | 3 | Layers 0–7, 8–15, 16–23. A layer count that doesn't divide rounds up; the waste is bounded because window pages are |
| State slot (linear attention) | 24 *(illustrative)* | 24 layers × 16 heads × 128 × 128 × 4 B ≈ 24 MiB, fixed per request |

| Per-token cost (this model) | Bytes |
|---|---|
| Full-attention layers | 8 × 2 × 8 × 128 × 2 = **32 KiB / token** |
| 32k-token request, full layers | 1024 pages × 1 unit = **1 GiB** |
| Same request, window layers (1024-token window) | (32 + 1) pages × 3 units = **99 MiB**, constant |

### Addresses
- A request holds lists of **slot numbers** (`uint32`): its full-attention pages in order, its window pages as a ring, its state units.
- Each step, the CPU copies those lists into the step table. The GPU computes `address = slot × unit + offset_in_unit` (64-bit).
- Units never move, so there are no virtual ids and no translation table.

### Sharing
The prefix cache and live requests share pages by reference count.
- Each slot has two counts: `req_refs` (live requests) and `cache_refs` (prefix-cache entries). A unit is free when both are 0.
- **Cache entries end on a page boundary.** Shared pages are therefore full and never written again, so there is no copy-on-write.
- The cache is a trie of pages (one node per full prompt page, keyed by a chained hash, tokens verified). A node holds its full-attention page; the window pages covering the last window before it (sliding-window models, not built yet); and, for linear-attention models, optionally a **copy** of the state slot (requests update state in place).
- A request can only resume at a node with a state copy, so a hit recomputes from the deepest such node (any amount, not just ≤ 31 tokens). Copies are taken at two points per request: the prompt's last full page (the next turn of a conversation extends it), and the **branch point** where its match ended (the next request with that shared prefix resumes there). Pages between the copy and the branch point are recomputed into the request's own pages.

### Bookkeeping (CPU only, plain arrays, one scheduler thread, no locks)
| Field | Type | Meaning |
|---|---|---|
| `free` | bitmap, 1 bit per slot | Reusable now |
| `req_refs`, `cache_refs` | `uint16[]` per slot | Who holds the unit |
| `pending` | list of (slot, step) | No references left, but a GPU step may still read it |
| `free_units` | count | Units with no references (free or pending) |
| `cache_only_units` | count | Units held by the cache and by no request |
| `reserved` | count | Units promised to admitted requests, not yet allocated |

Slot states: **free → live → pending → free.**

### The one capacity question
Every caller (new request, generation top-up, loading from SSD, cache state copy) goes through one function:
```
fits(n):      n ≤ free_units − reserved
can_ever(n):  n ≤ free_units + cache_only_units − reserved

admit(n):
    if not can_ever(n): return WAIT        # evict nothing
    while not fits(n): evict the cache's next victim
    reserved += n
```
- `n` comes from one demand function: pages for the uncached part of the prompt, rounded up, plus a generation allowance (config, in tokens), plus the request's bounded state (window ring with one page in transit, or one state slot), plus a state slot for each copy it will hand the prefix cache. While it is admitted, a cache hit also holds its matched pages and its snapshot (neither free nor cache-only), so a hit that can't fit next to them is admitted without its copies, or cold.
- Each later `alloc()` takes one unit out of the request's reservation. `reserved ≤ free_units` always holds, so an admitted request's `alloc()` cannot fail. Unused reservation is returned when the request ends.
- Top-up: when a request has used its allowance it calls `admit` for another. On `WAIT` it pauses and keeps its pages until units free up. If every live request is paused, the newest one ends with stop reason `length`.
- A request that would not fit in an empty arena is rejected on arrival.
- The prefix cache picks victims in its own order (least recently used leaf node). Evicting an entry frees only the units no one else holds.

### Free now, reuse later
A step is one command buffer: one decode token for every active request, plus any prefill chunk. The CPU builds step N+1 while the GPU runs step N, so a unit released now may still be read.
- The GPU signals an `MTLSharedEvent` with the step number when each step completes.
- When a unit's last reference drops, `pending.push(slot, N)`, where N is the last committed step. It counts in `free_units` immediately.
- `alloc()` first moves every pending slot with `step ≤ event.signaledValue` to `free`, then takes any free slot. If there is none, it waits for the event to advance (at most one step) and retries.
- The rule is the same for every caller. Strictly, only CPU writers (SSD loads) need it, since GPU writes are ordered by the queue; one rule is simpler than two.

### Step tables
- For each active request, the table lists its slot numbers: full-attention pages, window pages, state units.
- 2 copies, alternated by step parity, so the CPU never writes the copy the GPU is reading.
- This holds only while the CPU is **at most one step ahead**: before filling the table for step N+2, wait for `event.signaledValue ≥ N`.

### SSD tier
- Only pages held by the cache alone are spilled. A request never references a page that is not in memory.
- Spill directory is a config setting. Default `/Volumes/T5/mllm-cache` (the internal SSD is nearly full; the T5 is USB, ~1 GB/s, still far faster than re-running a long prefill). Point it at the internal SSD when there is room.
- One file per prefix-cache node in the spill directory, holding that node's own pages in order (page `i` at `i × page bytes`). Opened with `F_NOCACHE` (macOS has no `O_DIRECT`), so spilled pages are not kept a second time in the OS file cache. `F_PREALLOCATE` on create to avoid fragmentation; `F_RDAHEAD 0` on load.
- Direct I/O needs buffer, offset and length aligned to the 4 KiB device block (both disks report 4096). Units are 1 MiB-aligned and pages sit at unit multiples, so `pwrite`/`pread` go straight from/to the `MTLBuffer`, no staging copy.
- No `fsync`: a spilled page is only a cache. Each file has a header (model id, page size, token count) and a checksum; a file that fails the check after a crash is deleted.
- Spill = `pwrite` the pages, then drop the cache's references. Safe while the GPU reads them, because shared pages are never written.
- Load = `admit`, `alloc`, then `pread` straight into the units (shared memory, no GPU upload), finished before the step that reads them is committed.
- Spilling stops when free space on the spill volume would fall below a floor (config, default 16 GiB, leaving room for swap if it is the internal SSD). Past that, eviction drops pages instead.

Rules from other engines' SSD-tier bugs (oMLX, vllm-mlx, vllm-metal; [reports/prior art.md](../reports/prior%20art.md)):
- **Key files on everything that changes the bytes**, not only the tokens. The header holds:
  - a format version;
  - the model path and a hash of its weights;
  - `T`, `unit_bytes`, `state_units`, conv bytes and delta bytes;
  - the prefill path (`gpu`, or `ane` + its weight format). LUT weights give other K/V.

  Plus the parent hash and the node's T token ids, verified on load. A file that exists is not proof of a spill: compare headers. Leave other models' files alone.
- **The checksum covers the whole payload**, not the size. `F_PREALLOCATE` gives a full-length file even when the `pwrite` never finished. Write under a per-writer tmp name, then rename.
- **The spill unit is a node with its snapshot.** Pages without the snapshot below them are useless to a hybrid model: S drops to 0. A load brings pages [0, S) plus one snapshot (the deepest), never every snapshot on the chain.
- **CPU I/O is not covered by the GPU event.** `pending` is gated on decode steps, so a `pread` into a unit needs its own ref until it returns. A cancel mid-load must not let the unit go free under it. Loads go through `admit`, not `fits`.
- **Every load failure is a miss.** That covers a missing file, a bad checksum, a short read and an unmounted volume. Drop the node and its subtree, and keep the refcounts balanced.
- **Drop the cache's refs only after `pwrite` succeeded.** On `ENOSPC` or `EIO` the node stays in memory, not spilled. The space budget counts every file in the directory, tmp files included, and re-reads free space.

### Budget example (64 GB, a ~27B mixed-attention model)
| Item | GB |
|---|---|
| macOS + your other apps (VM, browser) | 12–16 |
| GPU weights, 4-bit | ~15 |
| ANE weights, 4-bit LUT | ~15 *(estimate)* |
| Arena | 12 (config) |
| Workspace + step tables | 1–2 |

The arena size is the one memory setting. It is rounded down to a whole number of units (12 GiB at 1 MiB = 12,288 slots).

## To verify
| Question | Decides |
|---|---|
| ~~Tokens per page, and whether 8 KiB runs keep full bandwidth~~ **Bandwidth doesn't decide it** (2026-11-22, `tools/kv_gather_bench.swift`): decode-shaped reads over shuffled units hit 243–249 GB/s at T = 16/32/64/128, same as one contiguous stream (243 GB/s), even reading one isolated 4 KiB block per unit. int8 KV reads in half the time (4.5 vs 8.7 ms per step for 16 × 4k tokens × 8 layers); dequant cost not measured | Keep 32 tokens per page for the other reasons (≤ 31-token recompute, half-page waste per request). Kernel compute, not layout, decides 8-bit KV |
| ~~Does a residency set keep file-backed weight pages resident?~~ **No, and not anonymous memory either** (2026-11-22, `tools/residency_check.swift`, 2 × 4 GiB, one in a set, one control, idle during pressure). File-backed: set 100 → 60 %, control 100 → 42 %. Anonymous: set 100 → 0 % in one step, control → 31 %. So `pread` into anonymous buffers doesn't help, and it costs swap on the small internal SSD, while evicted file pages are just dropped | Keep `mmap` weights. Avoid eviction by not overcommitting RAM; weights read every decode step stay hot |
| Does `mlock` on the `mmap`ed weights keep them in RAM under pressure? (`vm.user_wire_limit` = 52.5 GB, no sudo needed) | Whether an idle server can keep its weights against other apps. Run on an idle machine: each pressure test grew swap by 7–10 GiB |
| ~~Can the GPU read ANE output IOSurfaces without a copy?~~ **Yes** (2026-11-22, `tools/ane_iosurface.swift`): Core ML `outputBackings` writes into our IOSurface; Metal reads it as an `r16Float` texture or a `bytesNoCopy` buffer, bit-exact. Wrap `bytesPerRow × rows` rounded to 16 KiB, not `IOSurfaceGetAllocSize` (it has a 64 B tail); check base alignment at startup, fall back to the texture view. Unknown: whether Core ML copies internally from the ANE into the surface | Prefill K/V reaches the arena from a GPU kernel (workspace IOSurface → units), no CPU copy |

## Deferred, with triggers
| Item | Add when |
|---|---|
| Cache entries that end mid-page (copy the last page on a hit) | The ≤ 31-token recompute shows up in time to first token |
| Preempt a live request and spill it | Paused requests show up in real use |
| Arena > 41.7 GB (several buffers; slot → buffer + offset) | Arena setting exceeds `maxBufferLength` |
| 8-bit KV layout | Kernel benchmark shows the decode win |
| Draft model state in the same arena | Speculative decoding proves worth it |
| More than one step encoded ahead (K+1 step-table copies) | GPU-side sampling removes the per-token CPU round trip |
| Eviction skips leaves a live request holds. Today the oldest leaf goes even when that frees only its snapshot, which destroys a hot entry (SGLang `lock_ref`, mistral.rs b95c8ecb). Fall back to any leaf so `admit` still terminates | Hit rate under load falls below what the arena could hold |
| Admitting the queue head doesn't evict prefixes the next waiters will match (touch their matches first: SGLang 20d1f3f56a) | Same |
| Random hash root per process, and cache salt per user (offline collisions block a prefix from caching; TTFT shows hits: vLLM 73b35cca7, 77073c77b) | Serving more than one user |
| Top-up with fed steps: step N+1 needs its page before N's top-up runs (SGLang 01e3b3f3a3); preemption keeps the linear state, not only K/V (SGLang 54ec2c4699) | Generation top-up or preemption gets built |

## Build order
1. Throwaway attention benchmark: a hand-filled step table over shuffled units, 16/32/64/128 tokens per page against contiguous, fp16 against 8-bit. Fixes tokens per page and KV precision. *Read side done (`tools/kv_gather_bench.swift`); 8-bit dequant cost still to measure in the real kernel (step 4).*
2. Allocator in C++, CPU only: bitmap, reference counts, pending, `admit`, eviction loop. Unit tests, no GPU.
3. Wrap it in one `MTLBuffer` + residency set; step tables; event-driven reclaim.
4. Attention kernels that read units through the step table.
