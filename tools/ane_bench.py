"""Neural Engine spike via Core ML: peak compute and weight bandwidth vs Core ML GPU.

Builds synthetic 1x1-conv stacks in ANE's preferred layout [1, C, 1, S] (fp16),
checks per-op placement with MLComputePlan, then times predict().
Safe by default: <1 GB weights, ~1.5 s per test, no sudo. Run with `nice`.

  .venv/bin/python tools/ane_bench.py [--gpu]   # --gpu adds a Core ML GPU comparison
"""
import argparse, os, shutil, sys, tempfile, time
from collections import Counter

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models.compute_plan import MLComputePlan

CU = {"ANE": ct.ComputeUnit.CPU_AND_NE, "GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU": ct.ComputeUnit.CPU_ONLY}
TEST_SECONDS = 1.5


def build(C, S, L, workdir):
    rng = np.random.default_rng(0)
    ws = [(rng.standard_normal((C, C, 1, 1)) / np.sqrt(C)).astype(np.float16) for _ in range(L)]

    @mb.program(input_specs=[mb.TensorSpec((1, C, 1, S), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        for w in ws:
            x = mb.conv(x=x, weight=w)
        return x

    m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                   minimum_deployment_target=ct.target.macOS15)
    path = os.path.join(workdir, f"c{C}_s{S}_l{L}.mlpackage")
    m.save(path)
    return path, L * C * C * 2  # weight bytes


def placement(model):
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=model.compute_unit)
    ops = Counter()
    for op in plan.model_structure.program.functions["main"].block.operations:
        if "conv" in op.operator_name:
            dev = plan.get_compute_device_usage_for_mlprogram_operation(op)
            ops[type(dev.preferred_compute_device).__name__.replace("ML", "").replace("ComputeDevice", "")] += 1
    return dict(ops)


def bench(path, unit, C, S):
    model = ct.models.MLModel(path, compute_units=CU[unit])
    where = placement(model)
    x = {"x": np.random.default_rng(1).standard_normal((1, C, 1, S)).astype(np.float16)}
    for _ in range(3):
        model.predict(x)  # warm up (first call compiles for the device)
    times, end = [], time.perf_counter() + TEST_SECONDS
    while time.perf_counter() < end or len(times) < 5:
        t = time.perf_counter(); model.predict(x); times.append(time.perf_counter() - t)
    return float(np.median(times)), where, len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="also run each test on the GPU via Core ML")
    args = ap.parse_args()
    units = ["ANE"] + (["GPU"] if args.gpu else [])

    # (label, C, S, L): compute sweep over prompt sizes, bandwidth at S=1.
    # No latency test: Core ML places tiny models (<= C=512) on the CPU, so it can't measure ANE latency.
    tests = [("compute", 2048, 64, 16), ("compute", 2048, 256, 16), ("compute", 2048, 1024, 16),
             ("compute", 4096, 512, 8),
             ("bandwidth", 4096, 1, 12)]

    work = tempfile.mkdtemp(prefix="ane_bench_")
    try:
        print(f"{'test':9} {'C':>5} {'S':>5} {'L':>3} {'unit':4} {'median ms':>9} {'TFLOPS':>7} {'GB/s':>6}  placement (conv ops)")
        for label, C, S, L in tests:
            path, wbytes = build(C, S, L, work)
            for unit in units:
                t, where, n = bench(path, unit, C, S)
                flops = 2 * C * C * S * L
                print(f"{label:9} {C:>5} {S:>5} {L:>3} {unit:4} {t*1e3:>9.3f} {flops/t/1e12:>7.2f} "
                      f"{wbytes/t/1e9:>6.1f}  {where}  (n={n})", flush=True)
            shutil.rmtree(path, ignore_errors=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
