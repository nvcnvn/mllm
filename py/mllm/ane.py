"""AneOps.dense(i): the same contract as GpuOps.dense, run as one Core ML model per dense block on the
Neural Engine. Prefill only (static chunk size S; the last chunk is zero-padded and sliced back).

dense block i (architecture.md "layer seam"):
  i = 0      : x (embedding, gathered on the GPU) -> norm -> projections of layer 0
  0 < i < L  : x, o -> o/out_proj(i-1) + residual -> norm, FFN(i-1) + residual -> norm -> projections of layer i
Layout on the ANE is channels-first (1, C, 1, S); linears are 1x1 convs.

ponytail: numpy copies at the ANE<->MLX boundary (~GB/s). C++ uses IOSurface backings (tools/ane_iosurface.swift).
"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
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


def _fp16(linear):
    """mlx (Quantized)Linear -> fp16 numpy (out, in, 1, 1): the exact weights the GPU path uses."""
    if hasattr(linear, "scales"):
        w = mx.dequantize(linear.weight, linear.scales, linear.biases, group_size=linear.group_size,
                          bits=linear.bits, mode=getattr(linear, "mode", "affine"))
    else:
        w = linear.weight
    return np.array(w.astype(mx.float32)).astype(np.float16)[:, :, None, None]


def _vec(a):
    return np.array(a.astype(mx.float32)).astype(np.float16)


def _compress(name):
    """ANE weight formats. fp16 = the GPU's dequantized weights, exact. Others: see architecture.md."""
    import coremltools.optimize.coreml as cto
    # one k-means worker: coremltools' worker pool breaks on the second build in a process
    pal = lambda **k: cto.OpPalettizerConfig(granularity="per_grouped_channel", num_kmeans_workers=1, **k)
    return {
        "fp16": None,
        "lut6": pal(nbits=6, mode="kmeans", group_size=16),
        "lut4": pal(nbits=4, mode="kmeans", group_size=4),  # uniform g16 (the spike's) is unusable: cos 0.40
        "int8": cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"),
    }[name]


def _build(gpu, i, S, path, weights):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    spec, layers = gpu.spec, gpu.layers
    H = spec.hidden

    def lin(x, w, osp=1):
        if osp == 1:
            return mb.conv(x=x, weight=w)
        return mb.concat(values=[mb.conv(x=x, weight=p) for p in np.split(w, osp, axis=0)], axis=1)

    def rms(x, gamma, eps, axis):
        """RMSNorm along `axis` as layer_norm over [x, -x] (mean 0), as Anemll does: no fp16 x^2 underflow."""
        n = x.shape[axis]
        d = mb.layer_norm(x=mb.concat(values=[x, mb.mul(x=x, y=np.float16(-1))], axis=axis), axes=[axis], epsilon=eps)
        size, shape = [-1] * 4, [1] * 4
        size[axis] = shape[axis] = n
        g = _vec(gamma).reshape(shape)
        return mb.mul(x=mb.slice_by_size(x=d, begin=[0, 0, 0, 0], size=size), y=g)

    def ffn_split(n):
        return next(k for k in (4, 2, 1) if n % k == 0 and n // k >= 512)

    def body(x, o):
        if i > 0:
            prev = layers[i - 1]
            proj = prev.linear_attn.out_proj if prev.is_linear else prev.self_attn.o_proj
            h = mb.add(x=x, y=lin(o, _fp16(proj)))
            m, ln = prev.mlp, prev.post_attention_layernorm
            hn = rms(h, ln.weight, ln.eps, 1)
            g, u = _fp16(m.gate_proj), _fp16(m.up_proj)
            k = ffn_split(g.shape[0])
            act = mb.mul(x=mb.silu(x=lin(hn, g, k)), y=lin(hn, u, k))
            x = mb.add(x=h, y=lin(act, _fp16(m.down_proj)))
        layer = layers[i]
        xn = rms(x, layer.input_layernorm.weight, layer.input_layernorm.eps, 1)
        if layer.is_linear:
            la = layer.linear_attn
            outs = {"qkv": lin(xn, _fp16(la.in_proj_qkv)), "z": lin(xn, _fp16(la.in_proj_z)),
                    "b": lin(xn, _fp16(la.in_proj_b)), "a": lin(xn, _fp16(la.in_proj_a))}
        else:
            sa = layer.self_attn
            nh, nkv, D = sa.num_attention_heads, sa.num_key_value_heads, sa.head_dim
            qg = mb.reshape(x=lin(xn, _fp16(sa.q_proj)), shape=[1, nh, 2 * D, S])
            q, gate = mb.split(x=qg, num_splits=2, axis=2)
            k = mb.reshape(x=lin(xn, _fp16(sa.k_proj)), shape=[1, nkv, D, S])
            outs = {"q": rms(q, sa.q_norm.weight, sa.q_norm.eps, 2),          # (1, H, D, S)
                    "gate": mb.reshape(x=gate, shape=[1, nh * D, 1, S]),
                    "k": rms(k, sa.k_norm.weight, sa.k_norm.eps, 2),          # (1, Hkv, D, S)
                    "v": lin(xn, _fp16(sa.v_proj))}
        if i > 0:
            outs["x"] = x
        return outs

    inputs = [mb.TensorSpec((1, H, 1, S), dtype=types.fp16)]
    if i > 0:
        prev = layers[i - 1]
        o_dim = _fp16(prev.linear_attn.out_proj if prev.is_linear else prev.self_attn.o_proj).shape[1]
        inputs.append(mb.TensorSpec((1, o_dim, 1, S), dtype=types.fp16))

    names = []

    if i > 0:
        @mb.program(input_specs=inputs, opset_version=ct.target.macOS15)
        def prog(x, o):
            outs = body(x, o)
            names.extend(outs)
            return tuple(outs.values())
    else:
        @mb.program(input_specs=inputs, opset_version=ct.target.macOS15)
        def prog(x):
            outs = body(x, None)
            names.extend(outs)
            return tuple(outs.values())

    m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                   minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cfg = _compress(weights)
    if cfg is not None:
        import coremltools.optimize.coreml as cto
        fn = cto.linear_quantize_weights if isinstance(cfg, cto.OpLinearQuantizerConfig) else cto.palettize_weights
        m = fn(m, cto.OptimizationConfig(global_config=cfg))
    out_names = [o.name for o in m.get_spec().description.output]
    tmp = Path(str(path) + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shutil.copytree(m.get_compiled_model_path(), tmp / "model.mlmodelc")
    (tmp / "io.json").write_text(json.dumps({"outputs": dict(zip(out_names, names)), "inputs": ["x", "o"][: len(inputs)]}))
    tmp.rename(path)


class AneOps:
    def __init__(self, gpu, S, weights="fp16"):
        import coremltools as ct
        self.gpu, self.spec, self.S = gpu, gpu.spec, S
        root = CACHE / "ane" / _model_key(gpu.spec.path) / f"S{S}_{weights}"
        self.models = []
        for i in range(self.spec.n_layers):
            p = root / f"dense_{i:03d}"
            if not p.exists():
                print(f"[ane] building dense block {i} -> {p}", flush=True)
                tmp, old = CACHE / "tmp", tempfile.tempdir
                tmp.mkdir(parents=True, exist_ok=True)
                tempfile.tempdir = str(tmp)  # coremltools compiles into the temp dir; only while building
                try:
                    _build(gpu, i, S, p, weights)
                finally:
                    tempfile.tempdir = old
            io = json.loads((p / "io.json").read_text())
            m = ct.models.CompiledMLModel(str(p / "model.mlmodelc"), compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.models.append((m, io))
        self.root = root

    def _pad(self, a):
        """mx [1, n, C] -> np fp16 (1, C, 1, S), zero rows past n."""
        n, C = a.shape[1], a.shape[2]
        out = np.zeros((1, C, 1, self.S), np.float16)
        out[0, :, 0, :n] = np.array(a[0].astype(mx.float32)).T
        return out

    def dense(self, i, ws):
        L, dt = self.spec.n_layers, self.gpu.dtype
        if i == L:
            raise ValueError("prefill never runs dense(L)")  # lm_head lives in decode (architecture.md)
        if i == 0:
            ws.x = self.gpu.lm.embed_tokens(ws.tokens)
        n = ws.x.shape[1]
        assert n <= self.S, f"chunk of {n} tokens > compiled size {self.S}"
        m, io = self.models[i]
        feed = {"x": self._pad(ws.x)}
        if i > 0:
            feed["o"] = self._pad(ws.o)
        res = {io["outputs"][k]: v for k, v in m.predict(feed).items()}
        get = lambda k: mx.array(res[k][..., :n]).astype(dt)  # (1, C|H, 1|D, n)
        if "x" in res:
            ws.x = get("x")[:, :, 0, :].transpose(0, 2, 1)
        if self.gpu.layers[i].is_linear:
            la = self.gpu.layers[i].linear_attn
            ws.qkv, ws.b, ws.a = (get(k)[:, :, 0, :].transpose(0, 2, 1) for k in ("qkv", "b", "a"))
            ws.z = get("z")[:, :, 0, :].transpose(0, 2, 1).reshape(1, n, la.num_v_heads, la.head_v_dim)
        else:
            sa = self.gpu.layers[i].self_attn
            ws.q, ws.k = get("q").transpose(0, 1, 3, 2), get("k").transpose(0, 1, 3, 2)   # [1, H, n, D]
            ws.v = get("v")[:, :, 0, :].reshape(1, sa.num_key_value_heads, sa.head_dim, n).transpose(0, 1, 3, 2)
            ws.gate = get("gate")[:, :, 0, :].transpose(0, 2, 1)

    def placement(self):
        """Ops per compute device, summed over all blocks (checks the work really lands on the ANE)."""
        from collections import Counter

        from coremltools.models.compute_plan import MLComputePlan
        import coremltools as ct
        c = Counter()
        for i in range(self.spec.n_layers):
            plan = MLComputePlan.load_from_path(str(self.root / f"dense_{i:03d}" / "model.mlmodelc"),
                                                compute_units=ct.ComputeUnit.CPU_AND_NE)
            for op in plan.model_structure.program.functions["main"].block.operations:
                dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
                if dev is not None:
                    c[(op.operator_name, type(dev.preferred_compute_device).__name__)] += 1
        return c


# --- shared weights: one weight copy, the GPU's (target-27b.md §5) ---
# Every dense block's linear weights and norm gammas are model *inputs*: the GPU dequantizes its 4-bit weights into
# fp16 IOSurfaces a layer at a time (AneOps "shared", src/ane.mm), so the ANE keeps no copy of its own. One model per
# block kind (the previous layer's kind, this layer's) and chunk size S serves every layer of that kind. Row-major
# [S, C] like the GPU's workspace. Each linear is sliced in the graph into Cin <= 1024 x Cout <= 4096 pieces: 4x the
# unsplit speed (tools/ane_layer.py). Inputs: x [S, H], o [S, o_dim] (i > 0), w_<name> [N, K], g_<name> [1, C].
# Outputs as io.json names them: x (i > 0), then qkv / z / b / a (linear) or q / gate / k / v (full), all [S, C].

def _shared_kind(i, kinds):
    return f"{kinds[i - 1] if i else 'none'}_{kinds[i]}"


def _build_shared(spec, inter, kind, S, path):
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

    def rms(x, g):  # RMSNorm on the last axis as layer_norm over [x, -x] (mean 0): no fp16 x^2 overflow, as _build
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


def build_shared(model, sizes):
    """Build the shared-weight blocks for chunk sizes `sizes` into <cache>/ane/<model_key>/shared_S<S>/<kind>/."""
    from . import spec as spec_mod
    spec = spec_mod.load(model)
    c = json.loads((spec.path / "config.json").read_text())
    inter = c.get("text_config", c)["intermediate_size"]
    kinds = sorted({_shared_kind(i, spec.kinds) for i in range(spec.n_layers)})
    for S in sizes:
        for kind in kinds:
            p = CACHE / "ane" / _model_key(spec.path) / f"shared_S{S}" / kind
            if p.exists():
                continue
            print(f"[ane] building shared block {kind} S={S} -> {p}", flush=True)
            tmp, old = CACHE / "tmp", tempfile.tempdir
            tmp.mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(tmp)
            try:
                _build_shared(spec, inter, kind, S, p)
            finally:
                tempfile.tempdir = old
