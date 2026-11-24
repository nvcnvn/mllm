"""mlx_lm.convert with a memory watchdog: convert.py SRC DST RECIPE  (RECIPE: 8bit|3bit|2bit|mixed_3_6|mixed_2_6|dequant)"""
import os, sys, threading, time
import mlx.core as mx
from mlx_lm.convert import convert

LIMIT = int(os.environ.get("CONVERT_LIMIT_GB", "34")) << 30
peak = [0]
def watch():
    while True:
        m = mx.get_active_memory() + mx.get_cache_memory()
        peak[0] = max(peak[0], m)
        if m > LIMIT:
            print(f"watchdog: {m / 2**30:.1f} GiB > limit, abort", flush=True)
            os._exit(3)
        time.sleep(0.5)
threading.Thread(target=watch, daemon=True).start()
mx.set_cache_limit(2 << 30)
if os.environ.get("CONVERT_DEVICE", "cpu") == "cpu":  # GPU kernels reading cold mmapped bf16 from the T5 hit the GPU watchdog
    mx.set_default_device(mx.cpu)
src, dst, recipe = sys.argv[1:4]
t = time.time()
if recipe == "dequant":
    convert(src, dst, dequantize=True)
elif recipe.startswith("mixed"):
    convert(src, dst, quantize=True, q_group_size=64, quant_predicate=recipe)
else:
    convert(src, dst, quantize=True, q_group_size=64, q_bits=int(recipe.removesuffix("bit")))
print(f"done {recipe} in {time.time() - t:.0f} s, mx peak {mx.get_peak_memory() / 2**30:.2f} GiB, "
      f"sampled active+cache peak {peak[0] / 2**30:.2f} GiB", flush=True)
