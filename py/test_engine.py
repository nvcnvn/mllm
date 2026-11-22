"""Build step 1 check (docs/architecture.md): our split dense/attn engine over the arena must produce
exactly mlx-lm's greedy tokens. Also: concurrent requests, waiting on a full arena, units all returned.

  HF_HOME=/Volumes/T5/huggingface .venv/bin/python py/test_engine.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
from mlx_lm.generate import generate_step

from mllm.engine import Engine

MODEL = "mlx-community/Qwen3.5-0.8B-4bit"
N = 64
CHUNK = 32  # small, so the prompt spans several chunks and pages

PROMPTS = [
    [{"role": "user", "content": "Explain in a few sentences why the sky is blue, then list three "
      "other everyday things that Rayleigh scattering explains. Keep it short and use plain words "
      "a ten-year-old would understand. Do not use any formulas or jargon."}],
    [{"role": "user", "content": "Write a haiku about unified memory."}],
    [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
]


def run(e, prompts):
    ids = [e.submit(p, max_tokens=N) for p in prompts]
    out = {i: [] for i in ids}
    while e.reqs:
        for rid, t, _, _ in e.step():
            out[rid].append(t)
    a = e.mem.arena
    assert len(a.free) == a.n_units and a.reserved == 0, "arena units leaked"
    return [out[i] for i in ids]


def reference(e, prompt):
    ids = e.tok.encode_chat(prompt)
    toks = []
    for t, _ in generate_step(mx.array(ids), e.mem.weights, max_tokens=N, prefill_step_size=CHUNK):
        toks.append(t)
        if t in e.tok.eos:
            break
    return toks


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main():
    e = Engine(MODEL, arena_bytes=256 << 20, chunk=CHUNK)
    print(f"unit {e.mem.arena.unit_bytes >> 10} KiB, {e.mem.arena.n_units} units, "
          f"state slot {e.mem.state_units} units, prompt lens {[len(e.tok.encode_chat(p)) for p in PROMPTS]}")

    # 1. one request at a time == mlx-lm, token for token
    solo = []
    for p in PROMPTS:
        t0 = time.perf_counter()
        ours, = run(e, [p])
        dt = time.perf_counter() - t0
        ref = reference(e, p)
        d = first_diff(ours, ref)
        print(f"solo  {len(ours):3d} tok in {dt:5.2f} s  match {d}/{len(ref)}  {e.tok._t.decode(ours)[:60]!r}")
        assert ours == ref, f"diverged at token {d}"
        solo.append(ours)

    # 2. all at once (batched decode) == each alone
    together = run(e, PROMPTS)
    for s, t in zip(solo, together):
        print(f"batch match {first_diff(s, t)}/{len(s)}")
        assert s == t, "batched decode differs from solo"

    # 3. arena fits one request at a time: the rest WAIT, then run, same tokens
    need = e.mem.units_for(max(len(e.tok.encode_chat(p)) for p in PROMPTS) + N) + e.mem.state_units
    small = Engine(MODEL, arena_bytes=need * e.mem.arena.unit_bytes, chunk=CHUNK)
    assert run(small, PROMPTS) == solo, "waiting requests produced different tokens"
    print(f"wait  ok with a {need}-unit arena")
    print("PASS")


if __name__ == "__main__":
    main()
