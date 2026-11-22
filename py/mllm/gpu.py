"""GpuOps: dense(i), attn(i), lm_head_sample. Python runs them through MLX with mlx-lm's modules,
so the results match mlx-lm bit for bit. C++ replaces this file with Metal kernels.

Workspace fields:
  tokens [B, S] int        input ids (dense 0)
  x      [B, S, hidden]    residual stream
  o      [B, S, *]         attn(i) output, before its out/o projection (consumed by dense(i+1))
  full layer i:   q, k, v [B, H, S, D] (normed, no RoPE yet), gate [B, S, H*D]
  linear layer i: qkv [B, S, conv_dim], z [B, S, Hv, Dv], b, a [B, S, Hv]
"""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.layers.attention import scaled_dot_product_attention
from mlx_lm.layers.recurrent import gated_delta_update, normalize_qk


def to_bits(x):
    """MLX bf16 -> numpy uint16 (raw bits) for the arena."""
    return np.array(x.view(mx.uint16))


def from_bits(a, dtype):
    return mx.array(a).view(dtype)


class GpuOps:
    def __init__(self, spec, memory):
        self.spec, self.mem = spec, memory
        self.lm = memory.weights.model.language_model
        self.layers = self.lm.layers
        self.tied = memory.weights.args.text.tie_word_embeddings
        self.dtype = self.lm.embed_tokens(mx.array([[0]])).dtype

    def dense(self, i, ws):
        """All the big matmuls between attn(i-1) and attn(i). i = 0 .. n_layers."""
        L = self.spec.n_layers
        if i == 0:
            ws.x = self.lm.embed_tokens(ws.tokens)
        else:
            prev = self.layers[i - 1]
            proj = prev.linear_attn.out_proj if prev.is_linear else prev.self_attn.o_proj
            h = ws.x + proj(ws.o)
            ws.x = h + prev.mlp(prev.post_attention_layernorm(h))
        if i == L:
            ws.x = self.lm.norm(ws.x)
            return
        layer = self.layers[i]
        xn = layer.input_layernorm(ws.x)
        B, S, _ = xn.shape
        if layer.is_linear:
            la = layer.linear_attn
            ws.qkv = la.in_proj_qkv(xn)
            ws.z = la.in_proj_z(xn).reshape(B, S, la.num_v_heads, la.head_v_dim)
            ws.b = la.in_proj_b(xn)
            ws.a = la.in_proj_a(xn)
        else:
            sa = layer.self_attn
            q, gate = mx.split(sa.q_proj(xn).reshape(B, S, sa.num_attention_heads, -1), 2, axis=-1)
            ws.gate = gate.reshape(B, S, -1)
            ws.q = sa.q_norm(q).transpose(0, 2, 1, 3)
            ws.k = sa.k_norm(sa.k_proj(xn).reshape(B, S, sa.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
            ws.v = sa.v_proj(xn).reshape(B, S, sa.num_key_value_heads, -1).transpose(0, 2, 1, 3)

    def attn(self, i, ws, rows, states):
        """Everything that touches per-request state. rows[b] = (pos, slots); states[b] = load_state views.
        ponytail: loops over rows (batch 1 at a time); a batched paged kernel replaces this in C++."""
        f = self._linear if self.layers[i].is_linear else self._full
        ws.o = mx.concatenate([f(i, ws, b, pos, slots, states[b]) for b, (pos, slots) in enumerate(rows)])

    def _full(self, i, ws, b, pos, slots, _state):
        sa = self.layers[i].self_attn
        q, k, v = ws.q[b:b + 1], ws.k[b:b + 1], ws.v[b:b + 1]
        S = q.shape[2]
        q, k = sa.rope(q, offset=pos), sa.rope(k, offset=pos)
        self.mem.write_kv(i, slots, pos, to_bits(k[0]).transpose(1, 0, 2), to_bits(v[0]).transpose(1, 0, 2))
        K, V = self.mem.read_kv(i, slots, pos + S)
        K, V = from_bits(K, self.dtype)[None], from_bits(V, self.dtype)[None]
        o = scaled_dot_product_attention(q, K, V, cache=None, scale=sa.scale, mask="causal" if S > 1 else None)
        o = o.transpose(0, 2, 1, 3).reshape(1, S, -1)
        return o * mx.sigmoid(ws.gate[b:b + 1])

    def _linear(self, i, ws, b, pos, slots, state):
        la = self.layers[i].linear_attn
        conv_np, delta_np = state[i]
        qkv = ws.qkv[b:b + 1]
        S = qkv.shape[1]
        conv_in = mx.concatenate([from_bits(conv_np, self.dtype)[None], qkv], axis=1)
        conv_np[...] = to_bits(conv_in[0, -(la.conv_kernel_size - 1):, :])
        conv_out = nn.silu(la.conv1d(conv_in))
        q, k, v = [
            t.reshape(1, S, h, d) for t, h, d in zip(
                mx.split(conv_out, [la.key_dim, 2 * la.key_dim], -1),
                [la.num_k_heads, la.num_k_heads, la.num_v_heads],
                [la.head_k_dim, la.head_k_dim, la.head_v_dim])
        ]
        q, k = normalize_qk(q, k, inv_scale=la.head_k_dim ** -0.5, eps=1e-6)
        out, st = gated_delta_update(q, k, v, ws.a[b:b + 1], ws.b[b:b + 1], la.A_log, la.dt_bias,
                                     mx.array(delta_np)[None], None, use_kernel=True)
        delta_np[...] = np.array(st[0])
        out = la.norm(out, ws.z[b:b + 1])
        return out.reshape(1, S, -1)

    def lm_head_sample(self, ws, samplers):
        h = ws.x[:, -1, :]
        logits = self.lm.embed_tokens.as_linear(h) if self.tied else self.mem.weights.lm_head(h)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return [samplers[b](logprobs[b:b + 1]).item() for b in range(len(samplers))]
