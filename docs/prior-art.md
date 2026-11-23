# Prior art checklist: compare, benchmark, learn

Starts after the POC ([cpp-port.md](cpp-port.md) closed). Its benchmarks run at 27B, so they come after [target-27b.md](target-27b.md). Other teams have built paged KV, prefix caching and SSD tiers, on Apple Silicon too. This list measures us against them and reads their code and bug history. Every item ends in a **Check**. A lesson counts as done only when it has changed something here: a [memory-design.md](memory-design.md) entry, a test, or a written rejection with the reason.

## What we compete on
Local use: one person on one Mac, not a server farm. That means Ollama and LM Studio more than vLLM. Concurrency is low (1–4), and it comes from agents: subagents, parallel tool calls, autocomplete next to chat. The requests share a long prefix, and the context grows every step. So the headline numbers are **time to first token (TTFT) on a growing conversation** and **wall time of an agent trace**. Throughput at 32 clients is a tie-breaker. Single-stream decode is bandwidth-bound for everyone at 27B, so it rarely separates engines.

## Rules
- **Headline runs at 27B.** Qwen3.6-27B 4-bit, once [target-27b.md](target-27b.md) closes. On Qwen3.5-0.8B, first run everything once as a dry run of the harness and the fairness setup, and don't publish those numbers: at 0.8B prefill is ~30× faster per token, so the prefix cache and ANE prefill look smaller than they are.
- **Peer groups by weights.** MLX 4-bit affine g64: mlx-lm, LM Studio (mlx-engine), oMLX, vllm-mlx, vllm-metal, us. GGUF Q4_K_M: Ollama, llama.cpp. It's a different quantization with different quality, so it gets its own table and isn't ranked against the MLX group on speed alone. mistral.rs goes in whichever group its weights match.
- **Two configs per system:** what a user gets out of the box (Ollama's defaults included: context size, model unload after idle), and its documented best config. Report both.
- **Real tokens, fixed lengths.** Chat text, not random ids, so prefix caches can hit. Fixed output lengths (ignore EOS where the server allows it, otherwise report generated tokens) so every engine does the same work.
- **Cold and warm separately:** first request after start, first request after 10 minutes idle, steady state.
- **Two machine states:** idle, and with the usual apps open (browser, editor, a VM). Our residency tests showed macOS takes back idle memory, and a local user always has other apps.
- **Record the version** (commit or wheel) and every non-default flag of each system.
- Clones and venvs go on `/Volumes/T5` (`/Volumes/T5/Codes/<name>`, each with its own venv). All cloned with full history (blob-less): `vllm`, `sglang`, `vllm-metal`, `vllm-mlx`, `omlx` (jundot/omlx), `mlx-engine`, `mistral.rs`, `llama.cpp`, `mlx-lm`; `mlc-llm` is shallow.
- Results go in [reports/prior art.md](../reports/prior%20art.md), in the same format as the baseline report.

## 0. Setup
- [ ] Install and serve the model on each: **Ollama**, **LM Studio** (mlx-engine), **oMLX**, **vllm-mlx**, **vllm-metal**, **mistral.rs** (Metal, paged attention on), **llama.cpp** `llama-server`, **mlx-lm** server. **Check:** each answers the 3 chat prompts from `test_engine.py` with greedy decoding; a system that can't run the model is recorded and skipped.
- [ ] `serve_bench.py` grows two modes: `--turns N` (a conversation that resends its history plus ~300 new user tokens each turn) and `--trace FILE` (replay recorded requests with their original timing and parallelism). **Check:** both run against `mllm serve` and mlx-lm; `cached_tokens` read where a server reports it, listed as missing where not.
- [ ] Record an agent trace: point a real coding agent at `mllm serve` for one real task and log every request (body, start time, which ones overlap). Nobody can tune for it in advance. **Check:** the trace file, its token counts and its peak concurrency.

## 1. Benchmarks (local use)
Each item: every system, both configs, median of 3 rounds, at 27B (0.8B dry run first).
- [ ] **Chat:** a 500-token prompt, 256 tokens out, 1 client. **Check:** TTFT and decode tok/s.
- [ ] **Long prompt:** 8k / 16k / 32k tokens, 1 client. **Check:** TTFT (prefill tok/s).
- [ ] **Multi-turn** (the main number): `--turns 10` with a 2k system prompt. **Check:** TTFT for every turn. For hybrid-model engines, the tokens they recompute on a hit show whether they keep linear-state snapshots.
- [ ] **Agent trace:** replay the recorded trace. **Check:** wall time to finish it, and p50 / p90 TTFT.
- [ ] **Interference:** one long generation running while a short chat arrives. **Check:** the chat's TTFT and tok/s, against running alone. Our ANE-prefill / GPU-decode split targets exactly this.
- [ ] **Cold start:** process start → first token; first token after 10 minutes idle (machine with apps open). **Check:** seconds, per system.
- [ ] **Footprint:** memory at idle with the model loaded, and peak during the 32k prompt (`footprint`, Activity Monitor "Memory"). Then overfill (prompts longer than fit) and note what happens: wait, preempt, swap or crash. **Check:** GB per system and the overflow behaviour in one line each.
- [ ] **SSD tier (oMLX, vllm-mlx; llama.cpp slot save/restore):** 8k and 16k prompts. Measure TTFT cold, then with a memory hit, then with an SSD hit after eviction or a restart, with the cache on the T5 and on the internal SSD. **Check:** three TTFTs per size and disk; they become our SSD tier's targets.
- [ ] **Headroom (secondary):** the baseline report's sweep, `--clients 1 4 8 16 32` at 512 and 2048 tokens. **Check:** TTFT median / p90 and total tok/s, kept as a tie-breaker.
- [ ] **Correctness:** greedy 64 tokens for the 3 prompts against mlx-lm B = 1; batched decode by the top-1 / KL tolerance from `py/dump_batch_ref.py`. **Check:** the step where each system first diverges and its top-1 agreement. A system that's faster and wrong gets flagged, not ranked.

## 2. Learn from code
Each item produces a short note in the report (what they do, how it differs from ours) and a decision: **adopt**, **reject (reason)**, or **defer (trigger)**. Adopt and defer land in memory-design.md.
- [ ] **mistral.rs Metal kernels** (`mistralrs-paged-attn/src/metal/kernels/`: `paged_decode*.metal`, `paged_prefill.metal`, `gather_kv_cache.metal`): block size, K/V layout (`_hnd` vs ours), how the block table is read, prefill over paged prefix. **Check:** note, plus one decode shape (B = 8, 4k tokens) timed against our kernel if it can be lifted out.
- [ ] **vllm-metal:** the unified varlen paged kernel (decode and prefill rows in one dispatch, against our one-command-buffer step), the scheduler's chunked-prefill budget and preemption policy, TurboQuant KV. **Check:** note. Decide whether the varlen kernel replaces our separate decode and prefill paths, and whether TurboQuant changes the "8-bit KV layout" deferred item.
- [ ] **vLLM v1 hybrid KV cache manager** (`vllm/v1/core/`, upstream CUDA): page size shared across attention and Mamba/linear state, and prefix caching for hybrid models (where they keep state snapshots). **Check:** compare with our branch-point / last-full-page copies; note any snapshot placement that beats ours on the g8 trial-1 miss.
- [ ] **SGLang radix cache** (`python/sglang/srt/mem_cache/radix_cache.py`): `lock_ref` on nodes used by in-flight requests, LRU leaf eviction, node split on a partial match. **Check:** map each to ours (hit holds matched pages, `cache_refs`), list the differences.
- [ ] **vllm-mlx / oMLX SSD tier:** file format and granularity, checksums, eviction order between memory and SSD, the load path (do they copy or deserialize?), what they do about partial blocks. **Check:** note. Update memory-design.md's SSD tier section where theirs is simpler or measured better.
- [ ] **mlx-lm `PromptTrie` / `LRUPromptCache`** (`mlx_lm/models/cache.py`) and **LM Studio mlx-engine** (256-token save boundary): how they snapshot `ArraysCache` (linear state) for hybrid models. **Check:** note. Compare the 256-token boundary with our 32-token page plus snapshot points.
- [ ] **llama.cpp:** residency-set refresh timer, `llama-kv-cache-iswa` (sliding window plus full), slot save/restore to disk. **Check:** note. Compare their timer interval with ours.
- [ ] **Papers:** *Agent Memory Below the Prompt* (arXiv 2603.04428: Q4 KV persisted to SSD) and *Native LLM and MLLM Inference at Scale on Apple Silicon* (arXiv 2601.19139). **Check:** one paragraph each. Decide whether spilled pages should be quantized (it would halve SSD traffic on the USB T5).

## 3. Learn from their bugs
- [x] Read the closed issues and PRs on block / prefix-cache memory in **vLLM**, **SGLang**, **vllm-metal**, **vllm-mlx** and **mistral.rs** (search: `prefix cache`, `block leak`, `refcount`, `hash collision`, `eviction`, `preempt`, `hybrid`, `mamba`). Write each failure as a scenario (e.g. "evict a node while a hit on it is being admitted", "two requests insert the same branch point", "free a block still in a running batch", "hash collides but tokens differ", "a cancel during a chunk leaks reserved units"). **Check:** a table of scenarios, each with a test in `tests/` (memory or engine-cache) that covers it, or a line on why our design can't hit it.
- [x] A randomized test: admit / alloc / release / evict / cache insert / hit / cancel / advance fence in random order. After every operation it checks `reserved ≤ free_units`, that no unit is free while a reference or an in-flight step holds it, and that the unit counts add up to `n_units`. **Check:** runs 10⁶ operations clean on CPU, in CI. *Result: `tests/test_memory_fuzz.mm`; scenario table and the one bug found (fixed) in [reports/prior art.md](../reports/prior%20art.md). Read from commit history, not issues/PRs.*

## 4. Close out
- [ ] `reports/prior art.md`: the tables from §1, the notes from §2, and a "What it says" section: where we lead, where we trail, and why.
- [ ] Every adopt / defer decision is in memory-design.md (To verify, Deferred, or the design itself) or architecture.md. **Check:** no lesson lives only in the report.
