"""Heterogeneous spike on real Llama-3-8B layer shapes.

  layer       ANE (Core ML) vs MLX GPU (fp16 and 4-bit) on one decoder layer's linear ops
  contention  GPU 4-bit decode alone, ANE prefill alone, then both at once (GPU-heavy, ~30 s)
  quant       ANE prefill speed + memory with fp16 vs int8 / int4 / 4-bit LUT compressed weights
  handoff     attention/FFN split prefill: ANE qkv -> GPU attention -> ANE o+FFN; handoff cost + pipelining

  nice -n 10 .venv/bin/python tools/hetero_bench.py layer
  nice -n 10 .venv/bin/python tools/hetero_bench.py contention [--split 16 --qo-split 4 --in-split 4]
  nice -n 10 .venv/bin/python tools/hetero_bench.py handoff   # needs an idle GPU, ~2 min

layer/contention/quant skip attention (o_proj consumes q as a stand-in). handoff runs real causal
attention on the GPU, but only within the chunk. ponytail: no prefix KV yet; add it with the paged cache.
"""
import argparse, os, shutil, subprocess, sys, tempfile, time

import numpy as np

H, I, KV, LAYERS = 4096, 14336, 1024, 32           # Llama-3-8B: hidden, FFN, kv_heads*head_dim, layers
SHAPES = [(H, H), (KV, H), (KV, H), (H, H), (I, H), (I, H), (H, I)]  # q k v o gate up down (out, in)
LAYER_PARAMS = sum(o * i for o, i in SHAPES)
SECONDS = 3.0
assert LAYER_PARAMS == 218_103_808


def timed(fn, seconds=SECONDS, warmup=3):
    for _ in range(warmup):
        fn()
    ts, end = [], time.perf_counter() + seconds
    while time.perf_counter() < end or len(ts) < 5:
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), len(ts)


# ---------------- ANE via Core ML ----------------
def build_layer(S, split, path, qo_split=1, in_split=1, part="full"):
    """part: full layer, or the AFD pieces "qkv" (x -> q,k,v) and "ofn" (attn_out a, residual x -> y).
    split: FFN intermediate chunks; qo_split: q/o output chunks; in_split: reduction-dim chunks
    for every 4096-input matmul. ANE needs all three (16, 4, 4) to reach ~12 TFLOPS on 8B shapes."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    rng = np.random.default_rng(0)
    W = lambda o, i: (rng.standard_normal((o, i, 1, 1), dtype=np.float32) / np.sqrt(i)).astype(np.float16)
    wq, wk, wv, wo = W(H, H), W(KV, H), W(KV, H), W(H, H)
    ffn = [(W(I // split, H), W(I // split, H), W(H, I // split)) for _ in range(split)]

    def lin(a, w, osp=1, isp=in_split):  # out-split -> concat; in-split -> sum of partial convs
        xs = mb.split(x=a, num_splits=isp, axis=1) if isp > 1 else [a]
        outs = []
        for wo_ in np.split(w, osp, axis=0):
            parts = [mb.conv(x=xi, weight=wi) for xi, wi in zip(xs, np.split(wo_, isp, axis=1))]
            y = parts[0]
            for p in parts[1:]:
                y = mb.add(x=y, y=p)
            outs.append(y)
        return outs[0] if osp == 1 else mb.concat(values=outs, axis=1)

    spec = lambda: mb.TensorSpec((1, H, 1, S), dtype=types.fp16)

    def ofn(a, x):  # o_proj + residual + FFN + residual
        h = mb.add(x=x, y=lin(a, wo, qo_split))
        parts = [lin(mb.mul(x=mb.silu(x=lin(h, g)), y=lin(h, u)), d, isp=1) for g, u, d in ffn]
        y = parts[0]
        for p in parts[1:]:
            y = mb.add(x=y, y=p)
        return mb.add(x=h, y=y)

    if part == "qkv":
        @mb.program(input_specs=[spec()], opset_version=ct.target.macOS15)
        def prog(x):
            return lin(x, wq, qo_split), lin(x, wk), lin(x, wv)
    elif part == "ofn":
        @mb.program(input_specs=[spec(), spec()], opset_version=ct.target.macOS15)
        def prog(a, x):
            return ofn(a, x)
    else:  # o_proj consumes q as a stand-in for attention
        @mb.program(input_specs=[spec()], opset_version=ct.target.macOS15)
        def prog(x):
            q, k, v = lin(x, wq, qo_split), lin(x, wk), lin(x, wv)
            return ofn(q, x), k, v

    ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.macOS15).save(path)


def load_ane(path, S):
    import coremltools as ct
    from ane_bench import placement
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    x = {"x": np.random.default_rng(1).standard_normal((1, H, 1, S)).astype(np.float16)}
    return (lambda: m.predict(x)), placement(m)


def ane_loop(path, S, seconds):
    """Child process: warm up, print READY, wait for a line on stdin, run, print RESULT."""
    fn, _ = load_ane(path, S)
    for _ in range(3):
        fn()
    print("READY", flush=True)
    sys.stdin.readline()
    t, n = timed(fn, seconds, warmup=0)
    print(f"RESULT {t} {n}", flush=True)


# ---------------- GPU via MLX ----------------
def mlx_weights(bits, layers=1):
    import mlx.core as mx
    out = []
    for _ in range(layers):
        for o, i in SHAPES:
            w = (mx.random.normal((o, i)) / i**0.5).astype(mx.float16)
            w = mx.quantize(w, group_size=64, bits=bits) if bits else (w,)
            mx.eval(*w)
            out.append(w)
    return out


def mlx_layer_fn(ws, x, bits):
    import mlx.core as mx
    mm = ((lambda a, w: mx.quantized_matmul(a, *w, transpose=True, group_size=64, bits=bits)) if bits
          else (lambda a, w: a @ w[0].T))

    def run():
        h = x
        for l in range(len(ws) // 7):
            wq, wk, wv, wo, wg, wu, wd = ws[7 * l: 7 * l + 7]
            q, k, v = mm(h, wq), mm(h, wk), mm(h, wv)
            h = h + mm(q, wo)
            g = mm(h, wg)
            h = h + mm(g * mx.sigmoid(g) * mm(h, wu), wd)
        mx.eval(h, k, v)
    return run


def nbytes(ws):
    return sum(a.nbytes for w in ws for a in w)


# ---------------- commands ----------------
def cmd_layer(args, work):
    import mlx.core as mx
    print(f"one Llama-3-8B layer, linear ops only ({LAYER_PARAMS/1e6:.0f}M params); "
          f"'8B prefill' = S / (layer time x {LAYERS})")
    print(f"{'backend':26} {'S':>5} {'ms/layer':>9} {'TFLOPS':>7} {'8B prefill tok/s':>17}  placement")
    for S in (256, 1024):
        rows = []
        for split, qo, cin in ((1, 1, 1), (4, 1, 1), (16, 4, 1), (16, 4, 4)):
            path = os.path.join(work, f"layer_s{S}_{split}_{qo}_{cin}.mlpackage")
            build_layer(S, split, path, qo, cin)
            fn, where = load_ane(path, S)
            rows.append((f"ANE ffn/{split} qo/{qo} cin/{cin}", timed(fn)[0], where))
            shutil.rmtree(path, ignore_errors=True)
        x = mx.random.normal((S, H)).astype(mx.float16)
        for bits in (0, 4):
            ws = mlx_weights(bits)
            rows.append((f"MLX GPU {'q4' if bits else 'fp16'}", timed(mlx_layer_fn(ws, x, bits))[0], ""))
            del ws
        for name, t, where in rows:
            print(f"{name:26} {S:>5} {t*1e3:>9.2f} {2*LAYER_PARAMS*S/t/1e12:>7.2f} {S/(t*LAYERS):>17.0f}  {where}",
                  flush=True)


def cmd_contention(args, work):
    import mlx.core as mx
    S, layers, secs = 512, 16, 5.0
    path = os.path.join(work, f"layer_s{S}.mlpackage")
    build_layer(S, args.split, path, args.qo_split, args.in_split)
    ws = mlx_weights(4, layers)  # 4-bit decode sim: distinct buffers, ~2 GB, >> SLC
    gpu = mlx_layer_fn(ws, mx.random.normal((1, H)).astype(mx.float16), 4)
    gb = nbytes(ws) / 1e9

    def child():
        return subprocess.Popen([sys.executable, __file__, "ane-loop", path, str(S), str(secs)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def child_result(p):
        out = p.communicate()[0].split()
        t = float(out[out.index("RESULT") + 1])
        return S / (t * LAYERS)

    def wait_ready(p):
        while (line := p.stdout.readline()) and line.strip() != "READY":
            pass

    t_gpu, _ = timed(gpu, secs)
    p = child(); wait_ready(p); p.stdin.write("\n"); p.stdin.flush()
    ane_alone = child_result(p)
    p = child(); wait_ready(p); p.stdin.write("\n"); p.stdin.flush()
    t_gpu_both, _ = timed(gpu, secs, warmup=0)
    ane_both = child_result(p)

    tok = lambda t: layers / (t * LAYERS)  # 8B-equivalent decode tok/s (weights only)
    print(f"GPU 4-bit decode, {layers} layers = {gb:.2f} GB/token-step")
    print(f"  alone:     {gb/t_gpu:6.1f} GB/s  ~{tok(t_gpu):5.1f} tok/s (8B equiv)")
    print(f"  with ANE:  {gb/t_gpu_both:6.1f} GB/s  ~{tok(t_gpu_both):5.1f} tok/s  "
          f"({(t_gpu_both/t_gpu-1)*100:+.0f}% step time)")
    print(f"ANE prefill (S={S}, split={args.split}), 8B-equiv tok/s, linears only")
    print(f"  alone:     {ane_alone:6.0f} tok/s")
    print(f"  with GPU:  {ane_both:6.0f} tok/s  ({(ane_both/ane_alone-1)*100:+.0f}%)")


def _wired_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("Pages wired down"))
    return int(line.split(":")[1].strip(" .")) * 16384 / 1e9


def _footprint_gb():
    out = subprocess.run(["footprint", str(os.getpid())], capture_output=True, text=True).stdout
    kb = next(l for l in out.splitlines() if "phys_footprint:" in l).split(":")[1].split()
    return float(kb[0]) * {"KB": 1e3, "MB": 1e6, "GB": 1e9}[kb[1]] / 1e9


def cmd_quant(args, work):
    """Does the ANE keep its prefill speed with 4-bit / 8-bit compressed weights?"""
    import gc
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    S = 512
    variants = {
        "fp16": None,
        "int8 per-channel": cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"),
        "int4 block32": cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4",
                                                    granularity="per_block", block_size=32),
        "lut4 group16": cto.OpPalettizerConfig(mode="uniform", nbits=4,
                                               granularity="per_grouped_channel", group_size=16),
    }
    base = os.path.join(work, "fp16.mlpackage")
    build_layer(S, args.split, base, args.qo_split, args.in_split)
    fp16_gb = LAYER_PARAMS * 2 / 1e9
    print(f"one Llama-3-8B layer ({fp16_gb:.2f} GB fp16), S={S}, split {args.split}/{args.qo_split}/{args.in_split}")
    print(f"{'weights':17} {'disk GB':>7} {'ms':>7} {'TFLOPS':>7} {'8B tok/s':>8} {'proc +GB':>8} {'wired +GB':>9}  placement")
    for name, cfg in variants.items():
        path = base
        if cfg is not None:
            path = os.path.join(work, f"{name.split()[0]}.mlpackage")
            m = ct.models.MLModel(base, skip_model_load=True)
            fn_ = cto.linear_quantize_weights if isinstance(cfg, cto.OpLinearQuantizerConfig) else cto.palettize_weights
            fn_(m, cto.OptimizationConfig(global_config=cfg)).save(path)
            del m
        disk = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e9
        gc.collect()
        proc0, wired0 = _footprint_gb(), _wired_gb()
        fn, where = load_ane(path, S)
        t, _ = timed(fn)
        proc1, wired1 = _footprint_gb(), _wired_gb()
        print(f"{name:17} {disk:>7.2f} {t*1e3:>7.1f} {2*LAYER_PARAMS*S/t/1e12:>7.2f} {S/(t*LAYERS):>8.0f} "
              f"{proc1-proc0:>8.2f} {wired1-wired0:>9.2f}  {where}", flush=True)
        del fn
        gc.collect()
        if path != base:
            shutil.rmtree(path, ignore_errors=True)


def cmd_handoff(args, work):
    """AFD prefill: ANE runs qkv and o+FFN, GPU (MLX) runs causal attention in between.
    Measures what one ANE<->GPU handoff costs, and whether pipelining two chunks hides attention."""
    import coremltools as ct
    import mlx.core as mx
    from ane_bench import placement

    NQ, NKV, HD = 32, 8, 128
    rng = np.random.default_rng(1)
    print("one Llama-3-8B layer per chunk; times in ms per layer; 8B tok/s = S / (t x 32)")
    print(f"{'S':>5} {'mono':>7} {'qkv+ofn':>8} {'+call':>6} {'attn':>6} {'copies':>7} {'chain':>7} "
          f"{'per-hand':>8} {'pipelined':>9} | {'tok/s mono':>10} {'chain':>6} {'pipe':>6}")
    for S in (256, 512, 1024):
        paths = {}
        for part in ("full", "qkv", "ofn"):
            paths[part] = os.path.join(work, f"{part}_s{S}.mlpackage")
            build_layer(S, args.split, paths[part], args.qo_split, args.in_split, part=part)
        models = {p: ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE) for p, path in paths.items()}
        for p, m in models.items():
            where = placement(m)
            if set(where) != {"NeuralEngine"}:
                print(f"  warning: {p} placed on {where}")
        full, qkv, ofn = models["full"], models["qkv"], models["ofn"]
        x = rng.standard_normal((1, H, 1, S)).astype(np.float16)

        r = qkv.predict({"x": x})
        names = sorted(r, key=lambda n: -r[n].shape[1])  # q (4096 ch) first, then k, v (1024 ch)

        def attn(q, k, v):  # ANE layout [1, heads*HD, 1, S] <-> MLX [1, heads, S, HD]
            t = lambda a, n: a.reshape(1, n, HD, S).transpose(0, 1, 3, 2)
            o = mx.fast.scaled_dot_product_attention(t(q, NQ), t(k, NKV), t(v, NKV), scale=HD**-0.5, mask="causal")
            return o.transpose(0, 1, 3, 2).reshape(1, H, 1, S)

        def copies():  # numpy (Core ML) -> MLX -> numpy, same bytes as a real handoff
            a = [mx.array(r[n]) for n in names]
            mx.eval(*a)
            np.array(a[0])

        def chain():
            out = qkv.predict({"x": x})
            o = attn(*(mx.array(out[n]) for n in names))
            ofn.predict({"a": np.array(o), "x": x})

        pend = []

        def pipelined():  # two chunks in flight: GPU attention of one overlaps ANE work of the other
            out = qkv.predict({"x": x})
            o = attn(*(mx.array(out[n]) for n in names))
            mx.async_eval(o)
            if pend:
                ofn.predict({"a": np.array(pend.pop()), "x": x})
            pend.append(o)

        resident = [mx.array(r[n]) for n in names]
        mx.eval(*resident)
        t_full = timed(lambda: full.predict({"x": x}))[0]
        t_qkv = timed(lambda: qkv.predict({"x": x}))[0]
        t_ofn = timed(lambda: ofn.predict({"a": r[names[0]], "x": x}))[0]
        t_attn = timed(lambda: mx.eval(attn(*resident)))[0]
        t_copy = timed(copies)[0]
        t_chain = timed(chain)[0]
        t_pipe = timed(pipelined)[0]
        per_hand = (t_chain - t_qkv - t_ofn - t_attn) / 2
        ms = lambda t: f"{t*1e3:.2f}"
        tok = lambda t: f"{S/(t*LAYERS):.0f}"
        print(f"{S:>5} {ms(t_full):>7} {ms(t_qkv+t_ofn):>8} {ms(t_qkv+t_ofn-t_full):>6} {ms(t_attn):>6} "
              f"{ms(t_copy):>7} {ms(t_chain):>7} {ms(per_hand):>8} {ms(t_pipe):>9} | "
              f"{tok(t_full):>10} {tok(t_chain):>6} {tok(t_pipe):>6}", flush=True)
        for path in paths.values():
            shutil.rmtree(path, ignore_errors=True)
    print("mono: one Core ML call, no attention (upper bound) | +call: cost of splitting into 2 calls\n"
          "attn: GPU attention alone | copies: numpy<->MLX copies (ponytail: Python-only; zero-copy needs\n"
          "IOSurface-backed MLMultiArray + outputBackings from ObjC/C++) | per-hand: chain overhead per handoff\n"
          "pipelined: per-chunk-layer time with 2 chunks interleaved (ANE never waits for GPU attention)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "ane-loop":
        return ane_loop(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["layer", "contention", "quant", "handoff"])
    ap.add_argument("--split", type=int, default=16, help="FFN split count for the ANE model")
    ap.add_argument("--qo-split", type=int, default=4, help="q/o projection output split count")
    ap.add_argument("--in-split", type=int, default=4, help="reduction-dim split for 4096-input matmuls")
    args = ap.parse_args()
    work = tempfile.mkdtemp(prefix="hetero_bench_")
    try:
        {"layer": cmd_layer, "contention": cmd_contention, "quant": cmd_quant, "handoff": cmd_handoff}[args.cmd](args, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
