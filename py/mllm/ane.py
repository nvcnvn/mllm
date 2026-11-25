"""Neural Engine blocks for AneOps (src/ane.h): prefill's dense matmuls with no weight copy of the ANE's own
(target-27b.md §5).

Every dense block's linear weights and norm gammas are model *inputs*: the GPU dequantizes its 4-bit weights into fp16
IOSurfaces a layer at a time, so the ANE keeps no copy of its own. One model per block kind (the previous layer's kind,
this layer's) and chunk size S serves every layer of that kind. Dense block i (architecture.md "layer seam"):
  i = 0      : x (embedding, gathered on the GPU) -> norm -> projections of layer 0
  0 < i < L  : x, o -> o/out_proj(i-1) + residual -> norm, FFN(i-1) + residual -> norm -> projections of layer i
Row-major [S, C] like the GPU's workspace. Each linear is sliced in the graph into Cin <= 1024 x Cout <= 4096 pieces:
4x the unsplit speed (tools/ane_layer.py). Inputs: x [S, H], o [S, o_dim] (i > 0), w_<name> [N, K], g_<name> [1, C].
Outputs as io.json names them: x (i > 0), then qkv / z / b / a (linear) or q / gate / k / v (full), all [S, C].

  .venv/bin/python -c "import sys; sys.path.insert(0, 'py'); from mllm.ane import build; build(MODEL, [512, 1024])"
"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

# keep big files off the internal disk: the T5 drive when mounted, else the user cache dir
CACHE = Path(os.environ.get("MLLM_CACHE") or
             ("/Volumes/T5/cache/mllm" if os.path.isdir("/Volumes/T5") else Path.home() / "Library/Caches/mllm"))


def _model_key(path):
    """Cache dir name unique to these weights: resolved path + size/mtime of config and weight files."""
    p = Path(path).resolve()
    h = hashlib.sha1(str(p).encode())
    for f in sorted(p.glob("*.safetensors")) + [p / "config.json"]:
        st = f.stat()
        h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    name = p.parent.parent.name if p.parent.name == "snapshots" else p.name  # HF: models--org--name
    return f"{name}-{h.hexdigest()[:12]}"


def _kind(i, kinds):
    return f"{kinds[i - 1] if i else 'none'}_{kinds[i]}"


def _build(spec, inter, kind, S, path):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import Function, Program, types

    prev, cur = kind.split("_")
    H, D, nh, nkv = spec.hidden, spec.head_dim, spec.n_heads, spec.n_kv_heads
    Hv, Dv = spec.lin_v_heads, spec.lin_v_dim
    o_dim = {"linear": Hv * Dv, "full": nh * D}
    mats = {}  # name -> (N, K)
    if prev != "none":
        mats.update(o=(H, o_dim[prev]), gate=(inter, H), up=(inter, H), down=(H, inter))
    if cur == "linear":
        mats.update(qkv=(spec.conv_dim, H), z=(Hv * Dv, H), b=(Hv, H), a=(Hv, H))
    else:
        mats.update(q=(2 * nh * D, H), k=(nkv * D, H), v=(nkv * D, H))
    gammas = {"in": H, **({"post": H} if prev != "none" else {}), **({"q": D, "k": D} if cur == "full" else {})}
    specs = {"x": (S, H), **({"o": (S, o_dim[prev])} if prev != "none" else {})}
    specs.update({f"w_{m}": s for m, s in mats.items()})
    specs.update({f"g_{g}": (1, c) for g, c in gammas.items()})

    def pieces(n, cap):
        return [0] + [min(s + cap, n) for s in range(0, n, cap)]

    def lin(x, m, w):
        N, K = mats[m]
        co, ci = pieces(N, 4096), pieces(K, 1024)
        xs = [mb.slice_by_index(x=x, begin=[0, ci[k]], end=[S, ci[k + 1]]) for k in range(len(ci) - 1)]
        cols = []
        for j in range(len(co) - 1):
            acc = None
            for k in range(len(ci) - 1):
                wp = mb.slice_by_index(x=w[f"w_{m}"], begin=[co[j], ci[k]], end=[co[j + 1], ci[k + 1]])
                y = mb.matmul(x=xs[k], y=wp, transpose_y=True)
                acc = y if acc is None else mb.add(x=acc, y=y)
            cols.append(acc)
        return cols[0] if len(cols) == 1 else mb.concat(values=cols, axis=1)

    def rms(x, g):  # RMSNorm on the last axis as layer_norm over [x, -x] (mean 0): no fp16 x^2 overflow (Anemll)
        n, r = x.shape[-1], len(x.shape)
        d = mb.layer_norm(x=mb.concat(values=[x, mb.mul(x=x, y=np.float16(-1))], axis=-1), axes=[-1], epsilon=1e-6)
        return mb.mul(x=mb.slice_by_size(x=d, begin=[0] * r, size=list(x.shape[:-1]) + [n]), y=g)

    def body(w):
        x, outs = w["x"], {}
        if prev != "none":
            h = mb.add(x=x, y=lin(w["o"], "o", w))
            hn = rms(h, w["g_post"])
            act = mb.mul(x=mb.silu(x=lin(hn, "gate", w)), y=lin(hn, "up", w))
            x = outs["x"] = mb.add(x=h, y=lin(act, "down", w))
        xn = rms(x, w["g_in"])
        if cur == "linear":
            outs.update({m: lin(xn, m, w) for m in ("qkv", "z", "b", "a")})
        else:
            q, gate = mb.split(x=mb.reshape(x=lin(xn, "q", w), shape=[S, nh, 2 * D]), num_splits=2, axis=2)
            outs["q"] = mb.reshape(x=rms(q, w["g_q"]), shape=[S, nh * D])
            outs["gate"] = mb.reshape(x=gate, shape=[S, nh * D])
            outs["k"] = mb.reshape(x=rms(mb.reshape(x=lin(xn, "k", w), shape=[S, nkv, D]), w["g_k"]), shape=[S, nkv * D])
            outs["v"] = lin(xn, "v", w)
        return outs

    # mb.program takes its inputs from the function's signature; a variable set needs Function
    with Function({n: mb.placeholder(shape=s, dtype=types.fp16) for n, s in specs.items()},
                  opset_version=ct.target.macOS15) as fn:
        outs = body(fn.inputs)
        fn.set_outputs(list(outs.values()))
    prog = Program()
    prog.add_function("main", fn)
    m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                   minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE)
    out_names = [o.name for o in m.get_spec().description.output]
    tmp = Path(str(path) + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(m.get_compiled_model_path(), tmp / "model.mlmodelc")
    (tmp / "io.json").write_text(json.dumps({"outputs": dict(zip(out_names, outs))}))
    tmp.rename(path)


def build(model, sizes):
    """Build the blocks for chunk sizes `sizes` (the engine loads 512 and its chunk, 1024) into <cache>/ane/<model_key>/shared_S<S>/<kind>/."""
    from . import spec as spec_mod
    spec = spec_mod.load(model)
    c = json.loads((spec.path / "config.json").read_text())
    inter = c.get("text_config", c)["intermediate_size"]
    kinds = sorted({_kind(i, spec.kinds) for i in range(spec.n_layers)})
    for S in sizes:
        for kind in kinds:
            p = CACHE / "ane" / _model_key(spec.path) / f"shared_S{S}" / kind
            if p.exists():
                continue
            print(f"[ane] building block {kind} S={S} -> {p}", flush=True)
            tmp, old = CACHE / "tmp", tempfile.tempdir
            tmp.mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(tmp)
            try:
                _build(spec, inter, kind, S, p)
            finally:
                tempfile.tempdir = old
