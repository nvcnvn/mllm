"""Does fp16 dense (the ANE's precision) hurt 27B? End to end, emulated on the GPU (target-27b.md §5 "Precision").

dense16 = GpuOps.dense computed the way the ANE block would (tools/ane_layer.py): fp16 residual stream, fp16 weights
(the GPU's 4-bit weights dequantized), fp16 activations; attention / DeltaNet stay on the GPU in bf16. Per text:
prefill the first P tokens with GpuOps.dense (bf16, bit-exact with mlx-lm), dense16, or the same in fp32 (the
reference), then score the next N tokens from that KV cache and state with the bf16 GPU path (what decode does after
an ANE prefill). Reports KL from the fp32 run, top-1 agreement and NLL on those N positions, and the largest |value|
per kind in dense16 (fp16 max 65504).

  .venv/bin/python tools/ane_fp16_check.py MODEL [P=2048] [N=256]
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "py"))
from mllm import spec as spec_mod                    # noqa: E402
from mllm.gpu import GpuOps                         # noqa: E402
from mllm.lanes import _layers                      # noqa: E402
from mllm.memory import Memory                      # noqa: E402
from mllm.tokenizer import Tokenizer                # noqa: E402

F16 = mx.float16
peak = {}  # kind -> max |value| seen in dense16


def track(kind, a):
    if a.dtype != F16:
        return a
    peak[kind] = max(peak.get(kind, 0.0), mx.abs(a.astype(mx.float32)).max().item())
    return a


def dense_in(ops, F16):  # F16: the dtype (fp16 = the ANE; fp32 = the reference)
    lm, layers, bf = ops.lm, ops.layers, ops.dtype
    w16 = lambda m: (mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
                     if hasattr(m, "scales") else m.weight).astype(F16)
    lin = lambda m, x: x @ w16(m).T
    norm = lambda m, x: mx.fast.rms_norm(x, m.weight.astype(F16), m.eps)  # mlx-lm's RMSNorm, in fp16

    def dense(i, ws):
        if i == 0:
            ws.x = lm.embed_tokens(ws.tokens).astype(F16)
        else:
            prev = layers[i - 1]
            proj = prev.linear_attn.out_proj if prev.is_linear else prev.self_attn.o_proj
            h = track("residual", ws.x + track("o_proj out", lin(proj, ws.o.astype(F16))))
            m, hn = prev.mlp, norm(prev.post_attention_layernorm, h)
            act = track("ffn act", mx.sigmoid(g := lin(m.gate_proj, hn)) * g * lin(m.up_proj, hn))
            ws.x = track("residual", h + track("down out", lin(m.down_proj, act)))
        layer = layers[i]
        xn = norm(layer.input_layernorm, ws.x)
        B, S, _ = xn.shape
        if layer.is_linear:
            la = layer.linear_attn
            ws.qkv = lin(la.in_proj_qkv, xn).astype(bf)
            ws.z = lin(la.in_proj_z, xn).astype(bf).reshape(B, S, la.num_v_heads, la.head_v_dim)
            ws.b, ws.a = lin(la.in_proj_b, xn).astype(bf), lin(la.in_proj_a, xn).astype(bf)
        else:
            sa = layer.self_attn
            q, gate = mx.split(lin(sa.q_proj, xn).reshape(B, S, sa.num_attention_heads, -1), 2, axis=-1)
            ws.gate = gate.reshape(B, S, -1).astype(bf)
            ws.q = norm(sa.q_norm, q).astype(bf).transpose(0, 2, 1, 3)
            ws.k = norm(sa.k_norm, lin(sa.k_proj, xn).reshape(B, S, sa.num_key_value_heads, -1)).astype(bf)
            ws.k = ws.k.transpose(0, 2, 1, 3)
            ws.v = lin(sa.v_proj, xn).astype(bf).reshape(B, S, sa.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        mx.eval(ws.x)
    return dense


def run(ops, mem, dense, ids, P, chunk=512):
    """Prefill ids[:P] with `dense` (chunks of `chunk`), then the bf16 GPU path over ids[P:]; logprobs [N, V]."""
    L, a = ops.spec.n_layers, mem.arena
    assert a.admit(mem.units_for(len(ids)) + mem.state_units)
    pages = [a.alloc() for _ in range(mem.units_for(len(ids)))]
    state = [a.alloc() for _ in range(mem.state_units)]
    mem.zero(state)
    for s in list(range(0, P, chunk)) + [P]:
        e = min(s + chunk, P) if s < P else len(ids)
        ws = mem.workspace()
        ws.tokens = mx.array([ids[s:e]])
        _layers(dense if s < P else ops.dense, ops, mem, ws, [(s, pages)], [state])
    ops.dense(L, ws)  # final norm, bf16
    w = mem.weights
    head = ops.lm.embed_tokens.as_linear if ops.tied else getattr(w, "lm_head", None) or w.language_model.lm_head
    logits = head(ws.x)[0].astype(mx.float32)
    a.release(pages + state)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main():
    model, P, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2048, int(sys.argv[3]) if len(sys.argv) > 3 else 256
    spec = spec_mod.load(model)
    mem, tok = Memory(spec, 1 << 30), Tokenizer(spec.path)
    ops = GpuOps(spec, mem)
    root = Path(__file__).resolve().parent.parent
    texts = {"prose (docs)": "".join((root / f).read_text() for f in ("docs/architecture.md", "docs/target-27b.md")),
             "code": "".join(p.read_text() for p in sorted((root / "py/mllm").glob("*.py")))}
    print(f"prefill {P} tokens (fp32 / bf16 GPU / fp16 dense), then score {N} with the bf16 GPU path")
    for name, text in texts.items():
        ids = tok.encode(text)[: P + N]
        assert len(ids) == P + N, f"{name}: {len(ids)} tokens < {P + N}"
        lp = {k: run(ops, mem, d, ids, P) for k, d in
              (("fp32", dense_in(ops, mx.float32)), ("bf16", ops.dense), ("fp16", dense_in(ops, F16)))}
        tgt = mx.array(ids[P + 1:] + [ids[-1]])[:, None]
        nll = lambda a: -mx.take_along_axis(a, tgt, -1)[:-1].mean().item()
        print(f"{name}: NLL fp32 {nll(lp['fp32']):.4f}", flush=True)
        for k in ("bf16", "fp16"):
            ref, got = lp["fp32"], lp[k]
            kl = (mx.exp(ref) * (ref - got)).sum(-1)[:-1]
            top1 = (ref.argmax(-1) == got.argmax(-1))[:-1].astype(mx.float32).mean().item()
            print(f"  {k} prefill vs fp32: KL mean {kl.mean().item():.2e} max {kl.max().item():.2e} · top-1 agree "
                  f"{top1 * 100:.1f}% · NLL {nll(got):.4f} · finite {bool(mx.isfinite(got).all().item())}", flush=True)
    print("largest |value| in fp16 dense (limit 65504):", {k: round(v, 1) for k, v in peak.items()})


if __name__ == "__main__":
    main()
