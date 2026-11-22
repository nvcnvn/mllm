"""AneOps.dense(i): the same contract as GpuOps.dense, run as one Core ML model per dense block on the
Neural Engine. Prefill only (static chunk size S; the last chunk is zero-padded and sliced back).

dense block i (architecture.md "layer seam"):
  i = 0      : x (embedding, gathered on the GPU) -> norm -> projections of layer 0
  0 < i < L  : x, o -> o/out_proj(i-1) + residual -> norm, FFN(i-1) + residual -> norm -> projections of layer i
Layout on the ANE is channels-first (1, C, 1, S); linears are 1x1 convs.

ponytail: numpy copies at the ANE<->MLX boundary (~GB/s). C++ uses IOSurface backings (tools/ane_iosurface.swift).
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

CACHE = Path(os.environ.get("MLLM_CACHE", "/Volumes/T5/cache/mllm"))  # keep big files off the internal disk
os.makedirs(CACHE / "tmp", exist_ok=True)
tempfile.tempdir = str(CACHE / "tmp")  # coremltools compiles models into the temp dir


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
        root = CACHE / "ane" / gpu.spec.path.parent.parent.name / f"S{S}_{weights}"
        self.models = []
        for i in range(self.spec.n_layers):
            p = root / f"dense_{i:03d}"
            if not p.exists():
                print(f"[ane] building dense block {i} -> {p}", flush=True)
                _build(gpu, i, S, p, weights)
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
