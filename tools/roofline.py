"""Back-of-envelope LLM roofline for Mac mini M4 Pro (20c GPU, 64 GB).

decode step time = max(bytes_read / BW, flops / FLOPS) + overhead
prefill time     = (linear flops + attention flops) / FLOPS
"""

# --- calibration knobs (measure on your machine and update) ---
BW = 210e9        # effective GB/s: peak 273, GPU STREAM ~200, llama.cpp achieves 71-85%
FLOPS = 6.0e12    # effective GPU FP16, derived from llama.cpp PP512 on 20c M4 Pro
OVERHEAD = 1e-3   # fixed per-step cost (dispatch, sync, sampling), seconds
BPW = 4.5         # MLX 4-bit, group 64 (~4.5 bits/weight)

# name: (total params, active params, layers, kv_heads, head_dim, q_heads)
MODELS = {
    "Llama-3-8B":     (8.0e9, 8.0e9, 32, 8, 128, 32),
    "Qwen3-30B-A3B":  (30.5e9, 3.3e9, 48, 4, 128, 32),
    "Qwen3-32B":      (32.8e9, 32.8e9, 64, 8, 128, 64),
    "Llama-3.3-70B":  (70.6e9, 70.6e9, 80, 8, 128, 64),
}


def kv_bytes_per_token(m, kv_bytes=2):
    _, _, layers, kvh, hd, _ = m
    return 2 * layers * kvh * hd * kv_bytes  # K and V


def decode(m, ctx, batch=1, kv_bytes=2):
    """Per-request tok/s and aggregate tok/s. MoE batching ignored (assumes no expert overlap gain)."""
    total, active, layers, _, hd, qh = m
    w = active * BPW / 8 * (batch if total != active else 1)  # dense: weights read once per step
    w = min(w, total * BPW / 8)
    kv = batch * ctx * kv_bytes_per_token(m, kv_bytes)
    flops = batch * (2 * active + 4 * layers * qh * hd * ctx)
    t = max((w + kv) / BW, flops / FLOPS) + OVERHEAD
    return 1 / t, batch / t


def prefill_seconds(m, prompt):
    _, active, layers, _, hd, qh = m
    linear = 2 * active * prompt
    attn = 2 * layers * qh * hd * prompt**2  # QK^T + PV, causal half
    return (linear + attn) / FLOPS


def mem_gb(m, ctx, batch=1, kv_bytes=2):
    return (m[0] * BPW / 8 + batch * ctx * kv_bytes_per_token(m, kv_bytes)) / 1e9


if __name__ == "__main__":
    # self-check: Llama-3-8B fp16 KV = 128 KiB/token
    assert kv_bytes_per_token(MODELS["Llama-3-8B"]) == 128 * 1024

    print(f"BW={BW/1e9:.0f} GB/s  FLOPS={FLOPS/1e12:.1f} T  overhead={OVERHEAD*1e3:.1f} ms\n")
    print(f"{'model':15} {'ctx':>6} {'mem GB':>7} {'tok/s f16KV':>11} {'tok/s q8KV':>10} {'TTFT(prompt=ctx)':>17}")
    for name, m in MODELS.items():
        for ctx in (4096, 32768):
            d16, _ = decode(m, ctx)
            d8, _ = decode(m, ctx, kv_bytes=1)
            print(f"{name:15} {ctx:>6} {mem_gb(m, ctx):>7.1f} {d16:>11.1f} {d8:>10.1f} {prefill_seconds(m, ctx):>16.0f}s")

    print(f"\nprefill rate (short prompt): " + ", ".join(
        f"{n} ~{1/prefill_seconds(m, 1):.0f} tok/s" for n, m in MODELS.items()))

    print("\nbatched decode, 4k ctx each (dense models), f16 KV / q8 KV:")
    for name in ("Llama-3-8B", "Qwen3-32B"):
        for b in (1, 2, 4, 8, 16):
            p16, a16 = decode(MODELS[name], 4096, b)
            p8, a8 = decode(MODELS[name], 4096, b, kv_bytes=1)
            print(f"  {name:11} N={b:>2}: per-req {p16:5.1f} / {p8:5.1f}   aggregate {a16:6.1f} / {a8:6.1f} tok/s")

    # speculative decoding upper bound on 32B: draft 0.6B, k=4, acceptance alpha
    m = MODELS["Qwen3-32B"]
    draft_t = 4 * (0.6e9 * BPW / 8 / BW + OVERHEAD)
    verify_t = 1 / decode(m, 4096, 1)[0]  # 5 tokens verify ~ cost of 1 (mem-bound)
    print("\nspeculative (Qwen3-32B + 0.6B draft, k=4), theoretical:")
    for alpha in (0.6, 0.7, 0.8):
        tokens = (1 - alpha**5) / (1 - alpha)
        print(f"  alpha={alpha}: {tokens/(draft_t+verify_t):.1f} tok/s  (vs {decode(m, 4096)[0]:.1f} plain)")
