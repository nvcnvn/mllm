"""One 27B dense block (py/mllm/ane.py's seam) for the ANE with every weight a runtime input (target-27b.md §5).

Block i, row-major [S, C] (the GPU's layout): x, o -> o/out_proj(i-1) + residual -> norm, FFN(i-1) + residual -> norm
-> projections of layer i (`kind`: linear or full attention). Norm gammas are baked (tiny); every linear's weight is
an fp16 input the GPU would fill from its 4-bit weights. Shapes from Qwen3.6-27B's config; weights random (timing).

split: none   one [Cout, Cin] input per linear
       slice  one input per linear, sliced in the graph into Cin <= 1024 x Cout <= 4096 pieces
       inputs those pieces as separate inputs
Builds to OUT/<kind>_<split>_S<S>.mlpackage and prints the matmul placement. Time it with tools/ane_weight_input.swift
(any model: every input and output an IOSurface).

  .venv/bin/python tools/ane_layer.py OUT S kind split
"""
import sys
from collections import Counter

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.models.compute_plan import MLComputePlan

H, I = 5120, 17408
LIN = dict(qkv=2 * 16 * 128 + 48 * 128, z=48 * 128, b=48, a=48)  # linear attention: in_proj_* out dims
NH, NKV, D = 24, 4, 256                                            # full attention (q_proj has the output gate)
O_IN = 48 * 128                                                    # out_proj / o_proj Cin (both 6144)


def pieces(n, cap):
    return [min(cap, n - s) for s in range(0, n, cap)]


def build(S, kind, split):
    rng = np.random.default_rng(0)
    gamma = lambda n: (1 + 0.1 * rng.standard_normal(n)).astype(np.float16)
    mats = {"o": (H, O_IN), "gate": (I, H), "up": (I, H), "down": (H, I)}
    if kind == "linear":
        mats.update({k: (n, H) for k, n in LIN.items()})
    else:
        mats.update(q=(NH * D * 2, H), k=(NKV * D, H), v=(NKV * D, H))
    # weight input names and shapes: one per linear, or one per piece
    parts = {m: [(j, k, co, ci) for j, co in enumerate(pieces(Cout, 4096)) for k, ci in enumerate(pieces(Cin, 1024))]
             for m, (Cout, Cin) in mats.items()}
    specs = [("x", (S, H)), ("oin", (S, O_IN))]
    if split == "inputs":
        specs += [(f"w_{m}_{j}_{k}", (co, ci)) for m in mats for j, k, co, ci in parts[m]]
    else:
        specs += [(f"w_{m}", mats[m]) for m in mats]

    def rms(x, n):  # RMSNorm on the last axis as layer_norm over [x, -x] (py/mllm/ane.py)
        d = mb.layer_norm(x=mb.concat(values=[x, mb.mul(x=x, y=np.float16(-1))], axis=-1), axes=[-1], epsilon=1e-6)
        r = len(x.shape)
        return mb.mul(x=mb.slice_by_size(x=d, begin=[0] * r, size=list(x.shape[:-1]) + [n]), y=gamma(n))

    def lin(x, m, w):
        Cout, Cin = mats[m]
        if split == "none":
            return mb.matmul(x=x, y=w[f"w_{m}"], transpose_y=True)
        offs = lambda sizes: np.cumsum([0] + sizes)
        co, ci = offs(pieces(Cout, 4096)), offs(pieces(Cin, 1024))
        xs = [mb.slice_by_index(x=x, begin=[0, ci[k]], end=[S, ci[k + 1]]) for k in range(len(ci) - 1)]
        cols = []
        for j in range(len(co) - 1):
            acc = None
            for k in range(len(ci) - 1):
                if split == "inputs":
                    wp = w[f"w_{m}_{j}_{k}"]
                else:
                    wp = mb.slice_by_index(x=w[f"w_{m}"], begin=[co[j], ci[k]], end=[co[j + 1], ci[k + 1]])
                y = mb.matmul(x=xs[k], y=wp, transpose_y=True)
                acc = y if acc is None else mb.add(x=acc, y=y)
            cols.append(acc)
        return cols[0] if len(cols) == 1 else mb.concat(values=cols, axis=1)

    def body(w):
        x, o = w["x"], w["oin"]
        h = mb.add(x=x, y=lin(o, "o", w))
        hn = rms(h, H)
        act = mb.mul(x=mb.silu(x=lin(hn, "gate", w)), y=lin(hn, "up", w))
        x = mb.add(x=h, y=lin(act, "down", w))
        xn = rms(x, H)
        if kind == "linear":
            return (x,) + tuple(lin(xn, m, w) for m in LIN)
        qg = mb.reshape(x=lin(xn, "q", w), shape=[S, NH, 2 * D])
        q, g = mb.split(x=qg, num_splits=2, axis=2)
        k = mb.reshape(x=lin(xn, "k", w), shape=[S, NKV, D])
        return x, rms(q, D), mb.reshape(x=g, shape=[S, NH * D]), rms(k, D), lin(xn, "v", w)

    # mb.program takes inputs from the function's signature; a variable count needs Function directly
    with Function({n: mb.placeholder(shape=s, dtype=types.fp16) for n, s in specs},
                  opset_version=ct.target.macOS15) as fn:
        fn.set_outputs(list(body(fn.inputs)))
    prog = Program()
    prog.add_function("main", fn)
    return ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                      minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE)


def placement(m):
    plan = MLComputePlan.load_from_path(m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    c = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if dev is not None and op.operator_name.endswith("matmul"):
            c[type(dev.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")] += 1
    return dict(c)


if __name__ == "__main__":
    out, S, kind, split = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    m = build(S, kind, split)
    path = f"{out}/{kind}_{split}_S{S}.mlpackage"
    m.save(path)
    n_in = len(m.get_spec().description.input)
    print(f"{path}: {n_in} inputs, matmuls on {placement(m)}", flush=True)
