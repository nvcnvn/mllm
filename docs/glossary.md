# Glossary

The C++ engine uses plain names. The model files, mlx-lm and our Python reference engine (py/mllm) use the short
names of the papers and of Hugging Face; this page maps one to the other.

## The ideas behind the names

- **Token**: a piece of text (about 3/4 of a word) the model reads or writes, as an integer id.
- **Residual**: the running vector of numbers per token that every layer reads and adds its result back into.
- **Layer**: the model is 24 (0.8B) or 64 (27B) layers in a row. Each layer is an *attention* part, then a
  *feed-forward* part.
- **Weights**: the model's learned numbers. Ours are stored in 4 bits (`QuantizedWeights`); applying them is a
  matrix multiply (`multiply_weights`).
- **Attention**: how a token looks back at earlier tokens. Two kinds alternate:
  - *full attention* (every 4th layer): each token compares its **query** with every earlier token's **key**, and
    takes a weighted mix of their **values**. Keys and values are kept per page in the arena (the "KV cache").
  - *linear attention* (Gated DeltaNet, the other layers): a fixed-size **state** per request is updated every token
    instead; its **decay** says how fast old information fades, its **update rate** how strongly a token writes in.
- **Feed-forward**: per-token work after attention: `gate` and `up` projections, combined (SwiGLU: silu(gate) * up),
  then `down` back to the residual size.
- **Norm** (RMSNorm): rescales a vector to a standard size before a projection.
- **Prefill**: reading the prompt, in chunks of up to 1024 tokens. **Decode**: writing the answer, one token per
  request per step, many requests in one batch.

## Workspace buffers (`Workspace`, memory.h)

| Now | Was (Python, older C++) | What it holds |
|---|---|---|
| `token_ids` | `tokens` | the step's token ids |
| `residual` | `x` | the residual |
| `attention_result` | `o` | attention's output, before the output projection |
| `query`, `key`, `value`, `query_gate` | `q`, `k`, `v`, `gate` | full attention's inputs |
| `query_key_value`, `output_gate`, `update_rate_input`, `decay_input` | `qkv`, `z`, `b`, `a` | linear attention's inputs |
| `next_token_scores` | `logits` | a score per vocabulary token |
| `request_count`, `tokens_per_request` | `B`, `S` | this step's shape |
| `step_table`, `encoder` | `table`, `enc` | |
| `max_rows`, `max_context` | `rows`, `max_ctx` | capacity |
| `after_attention`, `normalized`, `projected` | `h`, `xn`, `t` | scratch |
| `gate_values`, `up_values` | `g`, `u` | feed-forward scratch |
| `query_and_gate` | `qg` | the query projection's output (query and its gate) |
| `partial_sums` | `splitk` | split-K partial sums |
| `attention_heads` | `ao` | attention output per head |
| `two_pass_partials`, `two_pass_sums`, `two_pass_maxes` | `partials`, `sums`, `maxs` | long-context attention scratch |
| `gathered_keys`, `gathered_values`, `attention_scores` | `kc`, `vc`, `scores` | prefill attention scratch |
| `normalized_query_key`, `decay`, `update_rate` | `nqkv`, `gdn_g`, `beta` | linear attention scratch |
| `next_token_ids` | `ids` | the sampled token per row |

## Layer weights (`GpuOps::Layer`, gpu.h)

GpuOps' constructor loads each field from its model-file key; that is the one place both names appear.

| Now | Model-file key (under `layers.N.`) |
|---|---|
| `attention_norm`, `feed_forward_norm` | `input_layernorm`, `post_attention_layernorm` |
| `feed_forward_gate`, `feed_forward_up`, `feed_forward_down` | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` |
| `attention_output` | `self_attn.o_proj` / `linear_attn.out_proj` |
| `query_projection`, `key_projection`, `value_projection` | `self_attn.q_proj`, `k_proj`, `v_proj` |
| `query_norm`, `key_norm` | `self_attn.q_norm`, `k_norm` |
| `query_key_value_projection`, `output_gate_projection` | `linear_attn.in_proj_qkv`, `in_proj_z` |
| `update_rate_projection`, `decay_projection` | `linear_attn.in_proj_b`, `in_proj_a` |
| `conv_weights`, `decay_log_base`, `decay_bias`, `output_norm` | `linear_attn.conv1d`, `A_log`, `dt_bias`, `norm` |
| `merged_linear_projections`, `merged_query_key_value`, `merged_gate_up` | (ours: the parts above back to back) |

`QuantizedWeights` (was `QLinear`): `packed`, `scales`, `biases`, `out_size` (N), `in_size` (K).

## Functions

| Now | Was |
|---|---|
| `GpuOps::apply_weights(layer)` | `dense(i)`: everything between two attentions that multiplies by weights |
| `GpuOps::apply_attention(layer)` | `attn(i)`: everything that reads or writes the arena (KV pages, state) |
| `pick_next_tokens`, `next_tokens`, `sample_tokens`, `greedy_tokens` | `lm_head`, `ids`, `sample`, `argmax` |
| `feed_previous_tokens`, `look_up_embeddings` | `feed_tokens`, `embed` |
| `normalize`, `normalize_and_gate`, `multiply_weights`, `activate_gate` | `rms_norm`, `gated_rms_norm`, `linear`, `swiglu` |
| `AneOps::prepare_inputs`, `AneOps::apply_weights` | `inputs`, `dense` |

## Model shape (`ModelSpec`, spec.h)

`query_heads`, `key_value_heads` (was `n_heads`, `n_kv_heads`); `linear_key_heads`, `linear_value_heads`,
`linear_key_dim`, `linear_value_dim` (was `lin_k_heads` ...); `vocab_size` (was `vocab`).
