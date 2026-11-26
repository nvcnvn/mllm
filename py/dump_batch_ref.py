"""Reference for batched decode (tests/test_batch.mm): mlx-lm's own B = 1 vs B = 8 / 16 difference, and B = 1 logits.

  HF_HOME=/Volumes/T5/huggingface .venv/bin/python py/dump_batch_ref.py MODEL [OUT]

Teacher forcing: 8 sequences of T tokens (the golden chats, then extra text), every position decoded at S = 1 from
position 0, once alone (B = 1) and once all in one batch. Prints mean / max KL(B=1 || B=M) and top-1 agreement.
KL is over the reference's top-64 tokens plus one bucket for the rest (a lower bound on the full KL; the C++ test
computes the same thing from batch_ref). Top-1 agreement is also counted within one bf16 ulp of the reference's top
logit: every strict miss of mlx-lm's own B = 8 is a tie or a 1-ulp gap at B = 1.

OUT defaults to the golden dir. batch_ref.safetensors: tokens [8, T] int32; ids [8, T, 64] int32 and logits
[8, T, 64] bf16, the B = 1 top-64 by logit (descending); lse [8, T] f32, logsumexp of the full B = 1 logits.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_model

from dump_golden import CHATS
from mmmllm import spec as specmod
from mmmllm.ane import CACHE, _model_key
from mllm.tokenizer import Tokenizer

T, TOP = 128, 64
EXTRA = [
    "Rayleigh scattering is the elastic scattering of light by particles much smaller than its wavelength. "
    "Because the intensity scales with the inverse fourth power of the wavelength, blue light is scattered "
    "about five times more strongly than red light, which is why the daytime sky looks blue and sunsets look "
    "red: at sunset the light crosses far more air, and most of the blue has been scattered away before it "
    "reaches the eye. The same effect explains the bluish tint of distant mountains and of cigarette smoke.",
    "def merge(a, b):\n    out, i, j = [], 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n"
    "            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n"
    "    return out + a[i:] + b[j:]\n\n\ndef sort(xs):\n    if len(xs) < 2:\n        return xs\n"
    "    mid = len(xs) // 2\n    return merge(sort(xs[:mid]), sort(xs[mid:]))\n\n\nprint(sort([5, 3, 9, 1, 4]))\n",
    "统一内存架构让中央处理器和图形处理器共享同一块物理内存，因此模型权重只需加载一次，两者都可以直接读取，"
    "不必在显存和主存之间来回复制。对于在笔记本电脑上运行大型语言模型来说，这意味着可用的内存更大，延迟更低。"
    "当然，带宽仍然是解码速度的主要瓶颈：每生成一个词元，都要把全部权重从内存中读取一遍。",
    "Q3 results: revenue 1,284.5 (+12.3% y/y), operating margin 18.7%, free cash flow 342.1. Guidance for Q4: "
    "revenue between 1,310 and 1,350, margin 18-19%. Headcount rose from 4,812 to 5,067. The board approved a "
    "buyback of 250 and a dividend of 0.42 per share, payable on 2026-12-15 to holders of record on 2026-12-01. "
    "Risks: currency (EUR/USD 1.08 -> 1.15), supply of 3 nm wafers, and the outcome of case No. 24-cv-01873.",
]


def sequences(tok):
    seqs = []
    for i in range(8):
        ids = tok.encode_chat(CHATS[i]) if i < len(CHATS) else []
        ids = ids + tok.encode(EXTRA[i % len(EXTRA)]) + tok.encode(EXTRA[(i + 1) % len(EXTRA)])
        assert len(ids) >= T, (i, len(ids))
        seqs.append(ids[:T])
    return np.array(seqs, np.int32)


def decode(model, tokens):
    """tokens [B, T] -> logits [B, T, V] f32, one S = 1 step per position."""
    cache = make_prompt_cache(model)
    out = []
    for t in range(tokens.shape[1]):
        logits = model(mx.array(tokens[:, t:t + 1]), cache=cache)[:, -1]
        out.append(logits)
        mx.eval(out[-1])
    return mx.stack(out, axis=1)


def kl_top(ref_logits, ids, lse, q):
    """KL(p || q) over p's top ids plus one bucket for the rest. ref_logits / ids / lse from p; q full logits."""
    lp = ref_logits.astype(mx.float32) - lse[..., None]
    lq = mx.take_along_axis(q, ids, axis=-1) - mx.logsumexp(q, axis=-1, keepdims=True)
    p_top, q_top = mx.exp(lp), mx.exp(lq)
    p_rest = mx.maximum(1 - p_top.sum(-1), 1e-30)
    q_rest = mx.maximum(1 - q_top.sum(-1), 1e-30)
    return (p_top * (lp - lq)).sum(-1) + p_rest * (mx.log(p_rest) - mx.log(q_rest))


def main():
    path = specmod.resolve(sys.argv[1])
    model, _ = load_model(path)
    tokens = sequences(Tokenizer(path))
    ref = mx.concatenate([decode(model, tokens[b:b + 1]) for b in range(len(tokens))])  # B = 1, bf16
    lse = mx.logsumexp(ref.astype(mx.float32), axis=-1)
    ids = mx.argsort(-ref.astype(mx.float32), axis=-1)[..., :TOP].astype(mx.int32)
    top = mx.take_along_axis(ref, ids, axis=-1)
    full = ref.astype(mx.float32)
    for B in (8, 16):
        q = decode(model, np.concatenate([tokens] * (B // 8))).astype(mx.float32)[:8]
        kl = kl_top(top, ids, lse, q)
        lp, lq = full - lse[..., None], q - mx.logsumexp(q, axis=-1, keepdims=True)
        kl_full = (mx.exp(lp) * (lp - lq)).sum(-1)
        qa = mx.argmax(q, -1, keepdims=True)
        top1 = (mx.argmax(full, -1, keepdims=True) == qa).astype(mx.float32).mean().item()
        # within one bf16 ulp of the reference's top logit: the B = 1 logits can't tell those tokens apart
        mx_ = full.max(-1, keepdims=True)
        near = (mx.take_along_axis(full, qa, -1) >= mx_ - mx.power(2, mx.floor(mx.log2(mx.abs(mx_))) - 7))
        near = near.astype(mx.float32).mean().item()
        same = (q.astype(mx.bfloat16) == ref).all(-1).astype(mx.float32).mean().item()
        print(f"B={B}: {kl.size} positions, KL top-{TOP}+rest mean {kl.mean().item():.3e} max {kl.max().item():.3e}; "
              f"full-vocab KL mean {kl_full.mean().item():.3e} max {kl_full.max().item():.3e}; "
              f"top-1 {top1 * 100:.2f}% (within 1 ulp {near * 100:.2f}%); rows equal to B=1 {same * 100:.1f}%")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE / "golden" / _model_key(path)
    mx.save_safetensors(str(dst / "batch_ref.safetensors"),
                        {"tokens": mx.array(tokens), "ids": ids, "logits": top, "lse": lse})
    print(f"{dst / 'batch_ref.safetensors'}: {tokens.shape[0]} x {T} positions")


if __name__ == "__main__":
    main()
