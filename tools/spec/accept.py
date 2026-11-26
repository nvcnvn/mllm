# Greedy draft acceptance for speculative decoding on Qwen3.6-27B-4bit, in MLX (g21; docs/speculative.md).
#
# The 27B decodes 256 greedy tokens per prompt, keeping its last hidden state per position. Each draft is then run
# teacher-forced over prompt + output: under greedy verify a draft token only counts while every earlier one matched,
# and then its context IS the target's, so one teacher-forced pass gives exact acceptance (chained MTP steps excepted:
# they run per position, as a real draft would). A replay then counts verify passes for k = 1..6.
# Drafts: Qwen3.5-0.8B-4bit (same tokenizer), and the 27B's own MTP head (bf16 mtp.* from Qwen/Qwen3.6-27B, read
# tensor by tensor, never the whole checkpoint), bf16 and 4-bit (group 64). Run under the heavy lock:
#   /Volumes/T5/cache/heavy g21 .venv/bin/python tools/spec/accept.py [out.json]
import json
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.layers.attention import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5.layers import DecoderLayer

TARGET = "mlx-community/Qwen3.6-27B-4bit"
DRAFT = "mlx-community/Qwen3.5-0.8B-4bit"
BF16 = "/Volumes/T5/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5"
GEN, CHUNK, KMAX = 256, 1024, 6
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPTS = [  # (name, user message, thinking on)
    ("chat", "What are the main differences between TCP and UDP, and when would you pick each? Give concrete examples.", False),
    ("code", "Write a Python function that parses an ISO 8601 duration such as 'P3DT4H5M6S' into a number of seconds. "
             "Reject malformed input with a ValueError, and add a few unit tests with unittest.", False),
    ("code-edit", "Refactor this function to be readable and fix any bugs, then show the full new version:\n\n"
     + open(os.path.join(REPO, "tools/serve_bench.py")).read()[:3000], False),
    ("reasoning", "A train leaves city A at 9:00 at 80 km/h toward city B, 300 km away. Another leaves B at 9:30 "
                  "at 100 km/h toward A. At what time do they meet, and how far from A? Explain step by step.", True),
    ("summary", "Summarize the following design document in about ten bullet points:\n\n"
     + open(os.path.join(REPO, "docs/architecture.md")).read()[:12000], False),
    ("plan-think", "Plan a three-day trip to Kyoto for two people who like food and quiet temples, on a moderate "
                   "budget. Give a day-by-day plan.", True),
]


def bf16_tensor(name, index):
    """One tensor of the bf16 checkpoint, reading only its bytes (np.fromfile at its offset)."""
    path = os.path.join(BF16, index[name])
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    meta = header[name]
    assert meta["dtype"] == "BF16", meta
    start, end = meta["data_offsets"]
    raw = np.fromfile(path, dtype=np.uint16, count=(end - start) // 2, offset=8 + n + start)
    return mx.array(raw).view(mx.bfloat16).reshape(meta["shape"])


class MTP(nn.Module):
    """Qwen3-Next / Qwen3.5 MTP head (as vLLM's qwen3_next_mtp.py): fc(cat(norm_e(embed(t+1)), norm_h(h_t))) ->
    one full-attention decoder layer -> norm; logits through the target's (shared) lm_head."""

    def __init__(self, args):
        super().__init__()
        H, eps = args.hidden_size, args.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=eps)
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(H, eps=eps)

    def __call__(self, embeds, hidden, cache):
        x = self.fc(mx.concatenate([self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)], -1))
        x = self.layers[0](x, mask=create_attention_mask(x, cache), cache=cache)
        return self.norm(x)


def load_mtp(args):
    index = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))["weight_map"]
    weights = {}
    for name in sorted(k for k in index if k.startswith("mtp.")):
        w = bf16_tensor(name, index)
        if "norm" in name:  # HF stores Qwen3-Next RMSNorm scales as an offset from 1 (mlx-lm's sanitize adds it)
            w = w + 1.0
        weights[name.removeprefix("mtp.")] = w
    mtp = MTP(args)
    mtp.load_weights(list(weights.items()))
    mx.eval(mtp.parameters())
    return mtp


def target_decode(model, ids):
    """Greedy 256 tokens; returns (tokens, final-norm hidden [S-1, H] for every position but the last, decode tok/s).
    The MTP head takes the hidden after the final norm (as vLLM passes it): the pre-norm one accepted fewer drafts
    (k = 1, bf16: 1.76-1.95 vs 1.84-1.96 tokens per verify on the six prompts)."""
    lm = model.model.language_model
    cache = model.make_cache()
    hs = []
    for i in range(0, len(ids), CHUNK):
        h = lm(mx.array(ids[i : i + CHUNK])[None], cache)
        mx.eval(h)
        hs.append(h[0])
    head = lambda h: mx.argmax(model.lm_head(h[-1:]), -1)  # noqa: E731
    tok = head(hs[-1][-1:])
    out = []
    t0 = time.perf_counter()
    for _ in range(GEN):
        mx.eval(tok)
        out.append(tok.item())
        if len(out) == GEN:
            break
        h = lm(tok[None], cache)[0]
        hs.append(h)
        tok = head(h)
    tps = (GEN - 1) / (time.perf_counter() - t0)
    return out, mx.concatenate(hs).astype(mx.bfloat16), tps


def draft_argmax(draft, seq, first):
    """Teacher-forced greedy prediction of the 0.8B at positions first..len(seq)-2 (A[q] predicts seq[q+1])."""
    cache = draft.make_cache()
    preds = []
    for i in range(0, len(seq) - 1, CHUNK):
        logits = draft(mx.array(seq[i : min(i + CHUNK, len(seq) - 1)])[None], cache=cache)[0]
        lo = max(first - i, 0)
        if lo < logits.shape[0]:
            preds.append(mx.argmax(logits[lo:], -1))
        mx.eval(preds, [c.state for c in cache])
    return mx.concatenate(preds).tolist()


def mtp_chains(mtp, model, seq, hidden, P, kmax):
    """D[c][j]: the MTP's j-th chained draft at state c (t_c just emitted; predicts t_{c+1+j}), for c = P..S-2.
    Step 1 runs teacher-forced over all positions (as the real head re-runs on verified tokens); steps 2.. run
    per state from step 1's K/V, feeding the head's own output hidden and draft token back in."""
    embed = model.model.language_model.embed_tokens
    head = lambda o: mx.argmax(model.lm_head(o), -1)  # noqa: E731
    S = len(seq)
    cache = KVCache()
    outs = []
    for i in range(0, S - 1, CHUNK):  # position i: (h_i, t_{i+1}) -> t_{i+2}
        j = min(i + CHUNK, S - 1)
        o = mtp(embed(mx.array(seq[i + 1 : j + 1]))[None], hidden[i:j][None], cache)[0]
        outs.append(o[max(P - 1 - i, 0):] if j > P - 1 else o[:0])
        mx.eval(outs, cache.state)
    o1 = mx.concatenate(outs)  # rows for positions P-1..S-2
    d1 = head(o1).tolist()
    K, V = cache.keys, cache.values
    D = {}
    for c in range(P, S - 1):
        drafts, o = [d1[c - P]], o1[c - P][None, None]
        if kmax > 1 and drafts[0] == seq[c + 1]:  # later steps only matter while the chain matches
            ch = KVCache()
            ch.state = (K[..., :c, :], V[..., :c, :], c)
            for _ in range(kmax - 1):
                o = mtp(embed(mx.array([drafts[-1]]))[None], o, ch)
                drafts.append(head(o[0]).item())
        D[c] = drafts
    return D


def replay(match, P, S, k):
    """Greedy verify replay. match(c, j): draft j at state c equals the target. Returns (verifies, tokens)."""
    c, verifies, emitted = P, 0, 1  # the first token comes from prefill
    while c < S - 1:
        n = 0
        while n < min(k, S - 2 - c) and match(c, n):
            n += 1
        verifies += 1
        emitted += n + 1
        c += n + 1
    return verifies, emitted


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/T5/cache/mllm/spec/accept.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model, tok = load(TARGET)
    draft, _ = load(DRAFT)
    mtp16 = load_mtp(model.args.text)
    mtp4 = load_mtp(model.args.text)
    nn.quantize(mtp4, group_size=64, bits=4)
    results = []
    for name, text, think in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                      enable_thinking=think)
        P = len(ids)
        gen, hidden, tps = target_decode(model, ids)
        seq = ids + gen
        S = len(seq)
        r = {"prompt": name, "prompt_tokens": P, "decode_tps": round(tps, 2), "text": tok.decode(gen)[:300]}
        A = draft_argmax(draft, seq, P)  # A[c - P] predicts seq[c + 1]
        drafts = {"0.8b": lambda c, j: A[c - P + j] == seq[c + 1 + j]}
        for label, mtp in (("mtp-bf16", mtp16), ("mtp-q4", mtp4)):
            D = mtp_chains(mtp, model, seq, hidden, P, KMAX)
            drafts[label] = lambda c, j, D=D: j < len(D[c]) and D[c][j] == seq[c + 1 + j]
        for label, match in drafts.items():
            r[label] = {}
            for k in range(1, KMAX + 1):
                v, e = replay(match, P, S, k)
                r[label][k] = round((e - 1) / v, 3)  # tokens per verify (accepted + 1)
        print(json.dumps(r), flush=True)
        results.append(r)
    json.dump(results, open(out_path, "w"), indent=1)


def self_check():
    """replay() on hand-made matches."""
    assert replay(lambda c, j: True, 10, 20, 3) == (3, 10)   # 9 tokens after the first: 4 + 4 + 1
    assert replay(lambda c, j: False, 10, 20, 3) == (9, 10)  # no draft accepted: one token per verify
    assert replay(lambda c, j: j == 0, 10, 20, 3) == (5, 10)  # first draft always right: 2 per verify


if __name__ == "__main__":
    self_check()
    main()
