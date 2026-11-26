# MLX costs of the speculative pieces on the 27B (g21; docs/speculative.md): a verify forward of n = 1..7 rows on top
# of L keys (logits + argmax on every row), and a chain of k draft steps (0.8B, MTP bf16, MTP 4-bit) evaluated once.
# Median of 7 after 2 warm-ups. Under the heavy lock:
#   /Volumes/T5/cache/heavy g21 .venv/bin/python tools/spec/timing.py
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import KVCache

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from accept import DRAFT, TARGET, load_mtp  # noqa: E402


def median_ms(fn, reps=7, warm=2):
    times = []
    for i in range(warm + reps):
        t0 = time.perf_counter()
        fn()
        if i >= warm:
            times.append(time.perf_counter() - t0)
    return round(statistics.median(times) * 1e3, 2)


def prefill(model, L):
    cache = model.make_cache()
    ids = mx.random.randint(0, 200000, (L,))
    for i in range(0, L, 1024):
        model(ids[i : i + 1024][None], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def rewound(cache, fn):
    """fn() run on the caches, then the caches put back (K/V offsets, linear-state arrays)."""
    saved = [list(c.cache) if hasattr(c, "cache") else c.offset for c in cache]
    fn()
    for c, s in zip(cache, saved):
        if isinstance(s, list):
            c.cache = s
        else:
            c.offset = s


def verify(model, cache, n):
    """One n-row forward with argmax on every row."""
    mx.eval(mx.argmax(model(mx.zeros((1, n), mx.uint32), cache=cache), -1))


def draft_chain(model, cache, k):
    """k greedy single-token steps chained on the GPU, no sync between them."""
    t = mx.zeros((1, 1), mx.uint32)
    for _ in range(k):
        t = mx.argmax(model(t, cache=cache)[:, -1:], -1)
    mx.eval(t)


def main():
    model, _ = load(TARGET)
    draft, _ = load(DRAFT)
    mtp16 = load_mtp(model.args.text)
    mtp4 = load_mtp(model.args.text)
    nn.quantize(mtp4, group_size=64, bits=4)
    embed, lm_head = model.model.language_model.embed_tokens, model.lm_head
    for L in (512, 4096):
        cache = prefill(model, L)
        print(f"27B verify at {L} keys, ms by rows:",
              {n: median_ms(lambda: rewound(cache, lambda: verify(model, cache, n))) for n in range(1, 8)}, flush=True)
        dcache = prefill(draft, L)
        print(f"0.8B chain of k steps at {L} keys, ms:",
              {k: median_ms(lambda: rewound(dcache, lambda: draft_chain(draft, dcache, k))) for k in (1, 2, 3, 4)}, flush=True)
        for label, mtp in (("bf16", mtp16), ("q4", mtp4)):
            mcache = KVCache()
            h = mx.zeros((1, L, 5120), mx.bfloat16)
            mx.eval(mtp(embed(mx.zeros((1, L), mx.uint32)), h, mcache))

            def chain(k):
                offset, o, t = mcache.offset, h[:, -1:], mx.zeros((1, 1), mx.uint32)
                for _ in range(k):
                    o = mtp(embed(t), o, mcache)
                    t = mx.argmax(lm_head(o), -1)
                mx.eval(t)
                mcache.offset = offset

            print(f"MTP {label} chain of k steps at {L} keys, ms:",
                  {k: median_ms(lambda: chain(k)) for k in (1, 2, 3, 4)}, flush=True)


if __name__ == "__main__":
    main()
