"""GPU matmul ceiling for roofline.py's FLOPS knob: MLX bf16 GEMM and 4-bit qmm (group 64) at prefill shapes.

/Volumes/T5/Codes/mllm/.venv/bin/python tools/gemm_peak.py
"""
import time

import mlx.core as mx


def tflops(f, flop, n=20):
    for _ in range(3):
        mx.eval(f())
    t = time.perf_counter()
    mx.eval([f() for _ in range(n)])  # every call evaluated (a lazy loop would run only the last)
    return flop * n / (time.perf_counter() - t) / 1e12


for M, K, N in [(512, 5120, 17408), (2048, 5120, 17408), (4096, 4096, 4096)]:
    a = mx.random.normal((M, K)).astype(mx.bfloat16)
    b = mx.random.normal((N, K)).astype(mx.bfloat16)
    wq, s, bb = mx.quantize(b, 64, 4)
    print(f"{M} x {K} x {N}: bf16 GEMM {tflops(lambda: a @ b.T, 2 * M * K * N):.2f} TFLOPS, "
          f"4-bit qmm {tflops(lambda: mx.quantized_matmul(a, wq, s, bb, True, 64, 4), 2 * M * K * N):.2f}")
