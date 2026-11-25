"""Spike: can the ANE run prefill matmuls with the weight as a runtime *input* (not a baked constant)?

If yes, the ANE needs no weight copy of its own: the GPU dequantizes its 4-bit weights layer by layer into a scratch
buffer the ANE reads. Baked constants cost a private wired copy (+0.48 GB fp16 / +0.14 GB LUT4 per 8B layer measured).
Per shape: y = x @ Wᵀ, x [S, Cin], W [Cout, Cin], fp16, as (a) const W, (b) W an input. Reports placement (ANE or
not, MLComputePlan) and median predict() time -> TFLOPS. Python's predict copies inputs, so (b) includes copying W
in (the copy is timed alone too, to subtract). Run under the heavy lock (timing):

  .venv/bin/python tools/ane_dynamic_weights.py
"""
import os, sys, tempfile, time
from collections import Counter

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

SECONDS = 1.5


def build(S, Cin, Cout, dynamic, workdir):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((Cout, Cin)) / np.sqrt(Cin)).astype(np.float16)
    specs = [mb.TensorSpec((S, Cin), dtype=types.fp16)]
    if dynamic:
        specs.append(mb.TensorSpec((Cout, Cin), dtype=types.fp16))

        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def prog(x, w):
            return mb.matmul(x=x, y=w, transpose_y=True)
    else:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def prog(x):
            return mb.matmul(x=x, y=w, transpose_y=True)

    m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                   minimum_deployment_target=ct.target.macOS15)
    path = os.path.join(workdir, f"s{S}_{Cin}x{Cout}_{'dyn' if dynamic else 'const'}.mlpackage")
    m.save(path)
    return path, w


def placement(model):
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=model.compute_unit)
    where = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        if op.operator_name in ("ios18.matmul", "ios17.matmul", "matmul", "ios18.linear", "ios18.conv"):
            dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
            where[type(dev.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")] += 1
    return dict(where)


def timed(f):
    for _ in range(3):
        f()
    ts, end = [], time.perf_counter() + SECONDS
    while time.perf_counter() < end or len(ts) < 5:
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    # 27B-like shapes (hidden 5120) whole and split, plus the fine split that hit ~13 TFLOPS with const weights
    shapes = [(512, 5120, 5120), (512, 5120, 1024), (512, 1024, 4096), (512, 1024, 1024), (2048, 1024, 4096)]
    work = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    print(f"{'S x Cin x Cout':>20} {'mode':>6} {'placement':>12} {'ms':>8} {'TFLOPS':>7}  note")
    for S, Cin, Cout in shapes:
        flops = 2.0 * S * Cin * Cout
        x = np.random.default_rng(1).standard_normal((S, Cin)).astype(np.float16)
        for dynamic in (False, True):
            path, w = build(S, Cin, Cout, dynamic, work)
            model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            inputs = {"x": x, "w": w} if dynamic else {"x": x}
            out = model.predict(inputs)
            got = next(iter(out.values())).astype(np.float32)
            want = x.astype(np.float32) @ w.astype(np.float32).T
            err = float(np.abs(got - want).max() / (np.abs(want).max() + 1e-9))
            t = timed(lambda: model.predict(inputs))
            note = f"rel err {err:.1e}"
            if dynamic:
                tc = timed(lambda: w.copy())  # the copy predict() makes of W, roughly
                note += f"; W {w.nbytes / 2**20:.0f} MiB, numpy copy {tc * 1e3:.2f} ms"
            print(f"{S:>6} x {Cin:>5} x {Cout:>5} {'input' if dynamic else 'const':>6} "
                  f"{str(placement(model)):>12} {t * 1e3:8.2f} {flops / t / 1e12:7.2f}  {note}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
