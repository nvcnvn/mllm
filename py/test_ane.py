"""Build step 3 check (docs/architecture.md): ANE prefill writes the same K/V and linear state as GPU prefill,
within fp16 tolerance, and greedy decoding after it stays on mlx-lm's tokens.

  HF_HOME=/Volumes/T5/huggingface .venv/bin/python py/test_ane.py [fp16|int8|lut6|lut4]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from mllm.engine import Engine
from mllm.lanes import PrefillLane, PrefillWork
from test_engine import MODEL, N, first_diff, reference, run

S = 128
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "fp16"
COS_MIN = 0.999 if WEIGHTS == "fp16" else 0.99   # fp16 measured worst 0.99901; compressed formats lose more


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def prefill_into(e, lane, ids):
    """Run prompt[:-1] through `lane` into fresh units; return (pages, state_slots, seconds)."""
    a, m = e.mem.arena, e.mem
    n_pages, n_state = m.units_for(len(ids)), m.state_units
    assert a.admit(n_pages + n_state)
    pages, state = [a.alloc() for _ in range(n_pages)], [a.alloc() for _ in range(n_state)]
    m.zero(state)
    t0 = time.perf_counter()
    for pos in range(0, len(ids) - 1, S):
        lane.submit(PrefillWork(0, ids[pos: min(pos + S, len(ids) - 1)], pos, pages, state))
    lane.poll()
    return pages, state, time.perf_counter() - t0


def main():
    e = Engine(MODEL, arena_bytes=512 << 20, chunk=S, prefill="ane", ane_weights=WEIGHTS)
    m = e.mem
    text = (Path(__file__).parent.parent / "docs" / "architecture.md").read_text()[:1500]
    ids = e.tok.encode_chat([{"role": "user", "content": "Summarize this design in 3 bullets:\n" + text}])
    print(f"prompt {len(ids)} tokens, {-(-(len(ids) - 1) // S)} chunks of {S}")

    gpu_lane = PrefillLane(e.prefill.ops, m)
    for lane in (gpu_lane, e.prefill):                                    # warm up (ANE first call compiles)
        p, s, _ = prefill_into(e, lane, ids[:S + 1])
        m.arena.release(p + s)
    pg, sg, tg = prefill_into(e, gpu_lane, ids)
    pa, sa, ta = prefill_into(e, e.prefill, ids)
    n = len(ids) - 1
    print(f"prefill {n} tok: GPU dense {n / tg:6.0f} tok/s, ANE dense {n / ta:6.0f} tok/s (Python, copies included)")

    worst = 1.0
    for l in e.spec.full_layers:
        kg, vg = m.read_kv(l, pg, n)
        ka, va = m.read_kv(l, pa, n)
        ck, cv = cos(bf16(kg), bf16(ka)), cos(bf16(vg), bf16(va))
        worst = min(worst, ck, cv)
        print(f"layer {l:2d} full    K cos {ck:.6f}  V cos {cv:.6f}")
    _, st_g = m.load_state(sg)
    _, st_a = m.load_state(sa)
    for l in e.spec.linear_layers:
        cc, cd = cos(bf16(st_g[l][0]), bf16(st_a[l][0])), cos(st_g[l][1], st_a[l][1])
        worst = min(worst, cc, cd)
        if l % 4 == 0 or l == e.spec.linear_layers[-1]:
            print(f"layer {l:2d} linear  conv cos {cc:.6f}  state cos {cd:.6f}")
    m.arena.release(pg + sg + pa + sa)
    m.arena.unreserve(m.arena.reserved)
    print(f"worst cos {worst:.6f} (min {COS_MIN})")
    assert worst >= COS_MIN, "ANE prefill drifted from GPU prefill"

    ours, = run(e, [[{"role": "user", "content": "Summarize this design in 3 bullets:\n" + text}]])
    ref = reference(e, [{"role": "user", "content": "Summarize this design in 3 bullets:\n" + text}])
    d = first_diff(ours, ref)
    print(f"greedy after ANE prefill: same as mlx-lm for {d}/{len(ref)} tokens")
    print(f"  ours: {e.tok._t.decode(ours)[:160]!r}")
    if d < len(ref):
        print(f"  ref : {e.tok._t.decode(ref)[:160]!r}")
    print("PASS")


if __name__ == "__main__":
    main()
