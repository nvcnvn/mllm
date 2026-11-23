# Prior art: what other engines' KV-cache bugs teach us

Measured and read 2026-11-23, for [docs/prior-art.md](../docs/prior-art.md) §3. Benchmarks (§1) and code reading (§2) are still to do.

## How
Full git history of vLLM, SGLang, vllm-metal, vllm-mlx, oMLX, mlx-lm, mlx-engine, llama.cpp and mistral.rs, cloned under
`/Volumes/T5/Codes/`. Four agents searched the commit logs for memory-management fixes (prefix cache, block refcounts, eviction, admission,
freeing during GPU work, hybrid/linear state, SSD tiers, cancel/abort). They read about 130 diffs in total and mapped each bug to our
`Arena` / `PrefixCache` / `Engine`. Kernel, distributed, API and tokenizer bugs were skipped.

## Result
Out of about 80 bug reports, one correctness bug applied to our code. Three of the four agents found it on their own. Two cost cache hits.
Our design already prevents everything else, mostly because of three choices:
- one thread with no locks;
- the whole demand is reserved at admission;
- shared pages are full and never written again.

Each prevented case now has a test or an assert.

| # | Scenario (their bug) | Source | Ours | Covered by |
|---|---|---|---|---|
| 1 | A failed GPU step never signals its event, so units released at that step stay pending for good. Later `alloc` asserts "no free unit", or `StepTables::begin` asserts | llama.cpp 1a5631bea; vLLM d467a2a7f, 84030bbe3; SGLang 34f49c9d71 | **Was reachable. Fixed:** `DecodeLane::poll`'s error path sets the event to the last committed step after `wait()` | Reasoned, not tested: no hook commits a failing command buffer (`fail_next` throws before commit) |
| 2 | Unit reused while an in-flight step still reads it (abort, cancel, EOS of a fed row) | vLLM fbb3cf698; SGLang b2ccf36d4d, e75e3b8a98; llama.cpp d1f563a74 | Prevented: pending until the step's event; prefill chunks hold request refs until polled | `test_memory_fuzz`: alloc vs every in-flight step's reads. Assert in `Engine::finish`: never with a prefill chunk in flight |
| 3 | Matched block evicted between lookup and use (by the request's own admission) | vLLM 2c5718809, 588db1836; oMLX 30e7204a | Prevented: `admit` refs the hit before evicting | `test_memory_fuzz` (exact refcounts, stale-hit content check) |
| 4 | Hit returns stale or wrong K/V or state (bookkeeping drift, wrong parent, inverted eviction test) | vLLM 32142b3c6, 1b8a0d71c; llama.cpp f174553e1 | Prevented: chained hash + token and parent check; leaf-only eviction | `test_memory_fuzz` shadow contents; `test_prefix_cache` (collision, a prompt that diverges inside page 1) |
| 5 | Snapshot taken after the state moved past its point (a later chunk already ran) | oMLX 8986da14; SGLang 6460082c05; vLLM 75f3967ac | Prevented: no chunk past a snapshot point while the one ending there is in flight; ends clamped | `test_memory_fuzz` asserts the state at copy time equals its prefix |
| 6 | Resumed request writes into the cached state (no copy-on-write) | vLLM ce0711866; vllm-mlx 59b43c4 | Prevented: `copy_units` into private slots | `test_memory_fuzz`: no write to a cached or shared unit |
| 7 | Refs leaked on a failed admission or an abort/cancel path | SGLang e2715cf871, 4c6afbeeaa; oMLX eafcad6e | Prevented | `test_memory_fuzz` exact accounting after every op; idle drain frees every unit; `test_engine` abort |
| 8 | Double free / unit in two cache entries / double enqueue | mistral.rs 722e46e4, b95c8ecb; oMLX ce862486 | Prevented | `test_memory_fuzz`: each unreferenced unit in exactly one of {free, pending}; `cache_refs <= 1` |
| 9 | Request allocates past its reservation (admission estimate ≠ actual allocations) | vLLM 0ff1bf9bb, 7b1bc0a3e | Prevented: `demand` covers pages + copies | New assert in `Engine::alloc`; fuzz sums reservations |
| 10 | Full-prompt hit leaves no token to compute | vLLM f0d610a8a; mistral.rs 45ccd26f | Prevented: match stops at `(len-1)/T` | `test_prefix_cache` |
| 11 | Admission livelock: the hit plus its copies never fit | SGLang 69a3c54c70 | Prevented: drops copies, then the hit | `test_memory_fuzz` (requests with demand == n_units, repeated); `test_engine_cache` |
| 12 | Starvation / FCFS reversed | mistral.rs bf86b718 | Prevented: FIFO, stop at the first refusal | `test_memory_fuzz`: waiting requests are always the newest; `test_arena` |
| 13 | Stale data or NaN in a reused unit leaks into results | vLLM 4ff8c3c8f, ad720aefe; llama.cpp 0a7675746 | Prevented: cold state memset; attention reads below pos + n only | `test_engine`: every unheld unit filled with NaN, then batched output equals solo |
| 14 | Earlier of two prefill chunks fails unnoticed | vLLM 9a9f48dff | Can't happen: the GPU path has one chunk in flight; the ANE thread waits on each buffer | — |
| 15 | Eviction takes leaves that a live request still holds (frees only their snapshot) before cache-only entries | mistral.rs b95c8ecb; SGLang `lock_ref` | **Hit rate only.** Deferred (memory-design) | `test_prefix_cache` pins today's order |
| 16 | Head-of-queue admission evicts prefixes the next waiters need | SGLang 20d1f3f56a | **Hit rate only.** Deferred (memory-design) | — |
| 17 | Predictable hash root: offline collisions stop a prefix from being cached; TTFT shows cache hits | vLLM 73b35cca7, 77073c77b | Never wrong K/V (tokens verified). Deferred until multi-user serving | — |
| 18 | Chunk split differs on a hit, so numerics differ from a cold run | llama.cpp 52975342f; oMLX ae06d836 | Known: bit-exact only when chunk boundaries match (g8); ANE vs GPU tail chunk too | `test_engine_cache` |

Not built yet, so these became design rules (memory-design.md "SSD tier" and "Deferred"): ten SSD-tier failures from oMLX, vllm-mlx and
vllm-metal (stale files after a config change, torn writes, spilling pages without their snapshot, CPU I/O racing the pending rule, the
index outliving its file, disk full, volume unmounted), plus preemption losing linear state (SGLang 54ec2c4699) and top-up with fed steps
(SGLang 01e3b3f3a3).

## The fuzz test
`tests/test_memory_fuzz.mm` is CPU-only and runs in CI in about 1 s. It models `Engine::step` / `admit` / `snapshot` / `finish` /
`cancel` and both lanes:
- decode steps one ahead, with greedy rows fed into the next step;
- up to 2 prefill chunks in flight.

It runs 200 seeds × 300 requests on arenas of 40 / 70 / 100 units.

It counts that every scenario actually happened, and fails if one never did. Over the 200 seeds: 5.9k hits, 131k evictions, 50k
snapshots (1.2k duplicates), 8.5k releases while a fed step was in flight, 60k double-chunk prefills, 2.7k waits for the GPU, 11k
cancels and 12k early EOS.

I checked that it catches bugs by planting 11 into the code, rebuilding each from clean. It caught 9. The other 2 change nothing observable:
- a guard that's dead under the engine's protocol (`insert` over an existing snapshot);
- an empty chunk the end clamp already makes harmless.

The 9 it caught:
- reclaim one step early;
- `cache_only_units` not counted;
- `cache_release` frees a unit a request still holds;
- eviction of a non-leaf;
- resuming without a snapshot;
- a hit not held during admit;
- a duplicate snapshot keeping its reservation;
- finishing a cancelled request with a prefill chunk still in flight;
- non-FIFO admission.

It mirrors `engine.mm`'s rules, so a change to admit / snapshot / finish there needs the same change in `Sim`.
