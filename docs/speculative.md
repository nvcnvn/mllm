# Speculative decoding at 27B: measured (g21), built (g22-g23)

**Verdict:** worth building. Use the 27B's own MTP head, quantized to 4 bits, with k = 2 draft tokens. The *estimate*
is ~30 tok/s greedy at B = 1, about 2.0x today's 15.3. A third draft token only pays once 4-row verify is cheaper.

## How Qwen3.5/3.6's MTP head works
`mlx-community/Qwen3.6-27B-4bit` ships without the head. Its 15 `mtp.*` tensors are in the bf16 `Qwen/Qwen3.6-27B`
(0.85 GB bf16). It matches vLLM's `qwen3_next_mtp.py`:

    x   = fc( concat( pre_fc_norm_embedding(embed(t[i+1])), pre_fc_norm_hidden(h[i]) ) )   # 10240 -> 5120
    x   = decoder_layer(x, position i)   # a full-attention layer like the 27B's, with its own K/V cache
    out = norm(x);  logits = lm_head(out)                                                  # predicts t[i+2]

- `embed` and `lm_head` are the 27B's own (`mtp_use_dedicated_embeddings: false`).
- `h[i]` is the 27B's hidden state **after** its final norm, as vLLM passes it. Feeding the pre-norm hidden state
  accepted fewer tokens: 1.76-1.95 against 1.84-1.96 tokens per verify at k = 1.
- All six norms are stored as an offset from 1, as in the main model.
- Chained drafting (k > 1) reuses the one layer. Step j + 1 takes step j's `out` as `h` and its draft token as `t`,
  one position further on. It attends to step 1's K/V for the verified positions, plus the chain's own entries.

mlx-lm (0.32.x, the pinned dev checkout) has no MTP: `qwen3_5/checkpoint.py` drops `mtp.*`. Its `--draft-model`
speculative generation refuses this model, because `ArraysCache` (the gated-delta state) is not trimmable.

## Acceptance (greedy, 256 tokens per prompt)
Tool: `tools/spec/accept.py`. It reports tokens per verify (accepted drafts + 1), from an exact greedy replay. The
draft runs teacher-forced over the 27B's own greedy output, and chained MTP steps run per position as a real draft
would. The six prompts are: chat Q&A, code, a code edit on an 868-token file, a reasoning problem with thinking on, a
3303-token document summary, and a trip plan with thinking on.

| draft | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 | k = 6 | k = 2, per prompt |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-4bit (same tokenizer, checked) | 1.71 | 2.21 | 2.58 | 2.87 | 3.08 | 3.25 | 1.90-2.53 |
| MTP bf16 | 1.90 | 2.58 | 3.11 | 3.55 | 3.81 | 3.95 | 2.41-2.80 |
| **MTP 4-bit** (group 64, `nn.quantize`) | **1.88** | **2.56** | **3.08** | 3.43 | 3.65 | 3.86 | 2.34-2.77 |

- The MTP head beats the 0.8B on every prompt and at every k.
- 4-bit costs the head almost nothing.
- Acceptance is lowest on long-input work (summary, code edit) and on free-form thinking. It is highest on reasoning
  and code.

## Costs
Tools: `tools/spec/timing.py` (MLX) and `mllm bench-ops`, whose new "speculative verify" lines run one request with n
rows ending at key 512, through the prefill path, with lm_head and argmax on every row. Both were run under the heavy
lock.

| rows in one 27B pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| engine, GPU ms (512 keys) | 64.9 | 67.3 | 72.8 | 89.4 | 109.1 | 141.9 | 171.0 |
| MLX, wall ms (512 keys) | 68.1 | 77.4 | 78.0 | 97.6 | 121.9 | 151.1 | 179.6 |
| MLX, wall ms (4096 keys) | 69.7 | 81.4 | 91.3 | 110.0 | 131.7 | 173.5 | 207.0 |

Up to 3 rows a verify costs nearly one step. From 4 rows on, the multi-row matmul stops being bandwidth-bound. In
bench-ops the engine's B = 4 batched decode step is 92.7 ms, the same cliff.

These are draft costs per chained step, evaluated once per chain (MLX, k = 1-4):

| draft step | ms |
|---|---|
| MTP 4-bit (fc, the layer and the full 248k lm_head) | 4.1-4.3 |
| MTP bf16 | 6.6-6.8 |
| 0.8B | ~3.6-5.1 (engine: 3.0, at 335 tok/s) |

The lm_head is 0.72 GB of the MTP step's ~0.95 GB, so it dominates.

## Estimated engine speed (*estimate*, greedy, B = 1, short context)
tok/s = tokens per verify / (verify(k + 1) + k × draft + k × 0.65 ms state checkpoints + 1.5 ms CPU sync per round)

| draft, k | tokens per verify | ms per round | tok/s | vs 15.3 today |
|---|---|---|---|---|
| MTP 4-bit, k = 1 | 1.88 | 67.3 + 4.2 + 0.65 + 1.5 = 73.7 | 25.5 | 1.67x |
| **MTP 4-bit, k = 2** | **2.56** | 72.8 + 8.4 + 1.3 + 1.5 = 84.0 | **30.5** | **1.99x** |
| MTP 4-bit, k = 3 | 3.08 | 89.4 + 12.6 + 1.95 + 1.5 = 105.5 | 29.2 | 1.91x |
| MTP 4-bit, k = 3, if 4-row verify drops to 75 ms | 3.08 | 91.1 | 33.8 | 2.2x |
| 0.8B, k = 2 | 2.21 | 72.8 + 6.0 + 1.3 + 1.5 = 81.6 | 27.1 | 1.77x |

- Per prompt, MTP k = 2 gives 2.34-2.77 tokens per verify, so 28-33 tok/s.
- The 0.8B is slower, and it needs a second model in the engine: its own weights, K/V, state and prefix-cache entries.
  The MTP head is one extra layer that shares `embed` and `lm_head`.
- Sampling (temperature > 0) accepts fewer drafts. Measure it once rejection sampling exists.

## Plan (steps for the engine)
1. **Weights.** A tool reads only the `mtp.*` tensors from the bf16 checkpoint (`accept.py`'s `bf16_tensor`). It adds 1
   to the norms, quantizes the matmuls to 4-bit group 64 with `mx.quantize`, and writes `mtp.safetensors` (~0.25 GB)
   into the weights-cache dir. `Weights` loads it as one more full-attention `Layer`, plus `fc` and three norms. Check:
   the engine's MTP logits on a fixed hidden state equal MLX's `MTP` from `accept.py`, bit for bit.
2. **The MTP layer's K/V lives in the page unit.** It takes 4 KiB per token, +6% on the 64 KiB unit. That way
   prefix-cache hits keep it: the MTP's K/V can't be rebuilt without the 27B's hidden states, which aren't kept. Prefill
   runs the head over every prompt row (one more layer, ~1.6%) and keeps the last row's final-norm hidden state.
3. **Drafting.** After each verify, the head runs over the newly accepted positions (teacher-forced, from the verify's
   hidden states), then chains k - 1 more steps. The argmax feeds the next embed on the GPU, in one command buffer.
4. **Verify in the decode lane.** One request with k + 1 rows (`tokens_per_request = k + 1`, the prefill path), with
   lm_head and argmax on every row. Greedy acceptance is the longest matching prefix, then one bonus token. Start with a
   CPU sync per round (it drops the lane's one-step-ahead pipelining). A GPU-side accept count can win that back later.
5. **Linear-state rollback.** In verify, the gated-delta and conv kernels write the state after each of the k + 1
   tokens into k + 1 checkpoint state sets (extra stores; ~0.65 ms per extra set: 152 MB of state per request at 27B).
   Accepting n drafts makes set n the request's `state_slots` and frees the others, with no copy. Full-attention K/V
   needs no rollback: the position moves back, and the next verify overwrites the rejected rows.
6. **Prefix-cache invariant.** `r.position`, `r.generated` and `r.state_slots` only ever reflect accepted tokens.
   `cache_sequence` then inserts only the accepted sequence and the state after its last token. K/V rows past
   `position` in the tail page are ignored, as today.
7. **Tests.**
   - B = 1 greedy with speculation equals non-speculative output token for token, for 256 tokens at 0.8B. The 0.8B has
     no MTP head, so use a stub draft that is sometimes right, and run the 27B MTP case under the lock.
   - Rollback after 0, 1 and k accepted drafts.
   - A prefix-cache hit after a speculative reply equals a cold run.
8. **Later.**
   - Rejection sampling for temperature > 0.
   - A multi-row qmv kernel to make 4-row verify cheap, then k = 3.
   - A draft lm_head over the most frequent ~32k tokens (~4 ms to ~1.3 ms per draft step).
   - B > 1: verify rows of several requests in one pass.

## In the engine so far (g22: steps 1-3)
- **Weights.** `tools/spec/mtp.py weights` writes `<weights cache>/mtp/mtp.safetensors` (31 tensors, 0.239 GB; HF's
  `mtp.*` names), in a subdirectory of its own. The file equals
  `accept.py`'s 4-bit head byte for byte, and `test_weights` checks the engine's GPU read-back against its hashes.
  `ModelSpec::mtp` is set when the file exists; without it (0.8B, other models) nothing changes.
- **K/V.** The head is one more full layer in the unit, after the model's: 27B unit 2048 -> 2176 KiB (+6.25%), 73 -> 70
  units per state slot. Prefill now also runs the last layer's rest and the final norm, then the head over every row
  with the next prompt token (`PrefillWork::next_token`). Prefill at 2048 tokens: 124.0 vs 129.7 tok/s without the head
  (-4.4%, 3 interleaved pairs; decode 14.88 vs 15.00, noise). The plan's 1.6% left out the last layer's o_proj + FFN.
- **Drafting.** `GpuOps::draft_tokens` (k = 2, argmax fed on the GPU, one command buffer): 8.4 ms after 2-3 accepted
  rows, 10.0 ms after 1 (g22's design). Since g23 the head runs over all 3 verify rows inside the verify, so the draft
  no longer depends on how many rows were accepted (bench-ops "speculative draft": head over 3 rows + draft).
- **Checks** (`tests/test_mtp.mm`, 27B, three golden prompts + 128 greedy tokens): the head alone on MLX's hidden states
  gives K/V bit-exact with MLX; the head's K/V from the engine's own prefill: cos >= 0.99988; drafts equal `accept.py`'s
  on 723 of 725 (99.72%); step 1's scores within 0.25 at 24 rows. QA (g23b) reran the 27B golden suite with the head (test_engine, test_stack, test_lanes, speculation off):
  bit-exact.
- Plain decode now writes the head's K/V too (g23, below).

## In the engine (g23: verify, rollback, wiring; on by default)
- **What runs.** A lone B = 1 greedy request on a model with the head decodes in rounds (`DecodeLane::encode_round`).
  Each round drafts k = 2 from the previous step's head output, verifies 3 rows through the 27B (the prefill path,
  lm_head + argmax on every row), then runs the head over the 3 rows with the verify's argmax. The linear layers write
  the state after each row into its own set (`Workspace::checkpoint_states`; 2 spare sets, taken only while the request decodes alone: `Engine::take_spares`,
  so batched greedy requests need no more units than sampled ones). Acceptance
  swaps the kept set into `state_slots`. The head also runs at the end of every plain step, so its K/V covers every
  position and no validity tracking is needed. Off for temp > 0, B > 1 (the batch decodes plainly), `--logits`, and
  `mllm bench` (mlx_lm.benchmark's protocol). Golden tests set `Engine::speculate = false`.
- **Speed** (`build/tests/test_speculative bench`, 27B, 256 tokens per prompt, interleaved S P S P on a quiet
  machine, g23b):

  | prompt | prompt tokens | speculative tok/s | tokens per round | plain tok/s | speedup |
  |---|---|---|---|---|---|
  | chat | 33 | 31.0, 29.0 | 2.63 | 14.6, 14.0 | 2.09x |
  | code | 61 | 27.6, 24.9 | 2.71 | 13.9, 12.6 | 1.99x |
  | code-edit | 868 | 23.0, 22.7 | 2.43 | 13.9, 13.9 | 1.64x |
  | reasoning | 75 | 26.7, 26.7 | 2.77 | 14.0, 14.0 | 1.90x |
  | summary | 3303 | 25.9, 24.2 | 2.54 | 14.0, 14.0 | 1.79x |
  | plan-think | 39 | 22.5, 22.5 | 2.29 | 14.4, 14.3 | 1.57x |
  | **mean** | | **25.6** | 2.56 | **14.0** | **1.83x** |

  Tokens per round match accept.py's k = 2 (2.34-2.77). One round's GPU time is 83.5 ms, plus a 2.7 ms gap for the CPU
  sync (`generate`, the code prompt, 2.70 tokens per round: 31 tok/s). The estimate was 84 ms. Per-prompt tok/s is
  below 2.70 / 86 ms on some prompts; not yet explained. Under another session's CPU load (~860% CPU, outside the lock)
  the same bench gave a 23.7 tok/s mean: the per-round CPU sync is exposed to it, plain decode is not. QA saw rounds
  drift from 84 to ~100 ms under sustained load while plain steps stayed flat (likely GPU clock / power): sustained
  speculative decode is ~23-27 tok/s, and the 31.0 is a cool-GPU number.
- **The head's cost on plain decode** (`mllm bench -p 512 -g 128 -n 3`, 3 interleaved pairs vs main 561525f, which
  has no head): B = 1 14.81 -> 14.57 tok/s (-1.6%), B = 4 35.19 -> 34.55 (-1.8%).
- **Checks** (`tests/test_speculative.mm`, 27B, 128 tokens): speculative output equals plain greedy on golden 0 and 2
  and plan-think. Golden 1, code-edit and summary diverge at tokens 97, 48 and 54, each at a near-tie: plain's scores
  for the two tokens differ by 0, 0.125 and 0. Threshold 1.2 = 2 x test_rollback's max |3-row verify - 1-row| score at
  27B (0.60); test_speculative passed with it (QA, g23b). A NaN head K/V leaves the output equal (tolerance) and drops acceptance to 1.01 tokens per round (2.62 /
  2.54 clean). The next turn after a speculative reply hits the cache (184 of 205 tokens). It equals the cold run and
  drafts as well (2.80 tokens per round on both). test_rollback at 27B: kept state cos >= 0.9993, rejected rows leave
  no trace (bit-exact). test_memory_fuzz's Sim runs rounds and takes / releases the spare sets. `mllm serve` logs and
  reports (`usage.speculative`) rounds and tokens per round per request.

## Risks
- Verify cost past 3 rows is kernel-bound (above). k = 2 is the safe choice until that kernel changes.
- The estimate is from parts, not an end-to-end run. The CPU sync and the checkpoint stores are *estimates*.
  Bit-exactness: B = 1 greedy output stays exactly the target's only if the k + 1-row verify gives the same argmax as
  the 1-row step. The prefill path's matmuls round differently from decode's, so greedy speculative output can differ
  from non-speculative output at near-ties. Either verify with the decode kernels (B-row qmv), or test by tolerance
  as batched decode is.
- The MTP head's K/V on the page unit changes the memory layout, and with it the prefix-cache format.
