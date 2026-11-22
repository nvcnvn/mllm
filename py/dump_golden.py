"""Reference data for the C++ port: run the Python engine (GPU prefill) once and write what it saw.

  HF_HOME=/Volumes/T5/huggingface .venv/bin/python py/dump_golden.py MODEL [OUT]

OUT defaults to $MLLM_CACHE/golden/<model_key>/ (model_key as in ane.py). Reruns are byte-identical.

Files and keys (bf16 stays bf16, ints are int32):
  spec.json           {"spec": ModelSpec fields, "strings": [...], "chat": [3 message lists],
                       "chunk", "page_tokens", "calls": [{"name", "rows": [{"req", "pos", "n"}]}]}
  weights.json        {tensor name: {"file", "dtype", "shape", "sha256"}} over the model's *.safetensors
  tokens.safetensors  str/NN    ids of strings[NN] (tokenizer.encode)
                      chat/N    ids of chat[N] (encode_chat)
                      greedy/N  greedy 64 tokens of chat[N], run alone
  trace.safetensors   one prompt (chat[0]) in 2 prefill chunks, then chat[2] in 1 chunk, then 3 decode steps
                      at batch 2 (rows: chat[0], chat[2]). Calls are prefill0..2, decode0..2 (see spec.json "calls").
                      <call>/tokens                [B, S] lane input
                      <call>/LNN/dense/<field>     workspace fields that dense(NN) wrote, NN = 00..n_layers
                      <call>/LNN/attn/o            attn(NN) output
                      <call>/rB/LNN/k, v           after the call, full layer NN's K/V of row B's request,
                                                   pages in token order: [kv_heads, n_tokens, head_dim]
                      <call>/rB/LNN/conv, delta    after the call, linear layer NN's state of row B's request
                      decodeN/logits [B, vocab], decodeN/argmax [B], decodeN/sampled [B]
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
import numpy as np

from mllm.ane import CACHE, _model_key
from mllm.engine import Engine
from test_engine import CHUNK, N, PROMPTS

STRINGS = [
    "Hello, world! The quick brown fox jumps over the lazy dog.",
    "你好，世界。今天天气很好。",
    "日本語のテキストと한국어 텍스트",
    "emoji: 😀👍🏽 👨‍👩‍👧‍👦 🇻🇳",
    "digits 0 7 42 1234567890 3.14159 -1e-9",
    "spaces    four\ttab\t\ttwo tabs\n\nnewlines\r\n   trailing   ",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\n\n</think>\n\nok<|endoftext|>",
    "",
    " ",
]
DECODE_STEPS = 3


def weight_hashes(path):
    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            base = 8 + n
            for name, t in sorted(header.items()):
                if name == "__metadata__":
                    continue
                a, b = t["data_offsets"]
                fh.seek(base + a)
                out[name] = {"file": f.name, "dtype": t["dtype"], "shape": t["shape"],
                             "sha256": hashlib.sha256(fh.read(b - a)).hexdigest()}
    return out


def i32(x):
    return mx.array(np.asarray(x, np.int32))


def trace(e):
    """Run chat[0] and chat[2] through Engine.step with decode held back until both are prefilled."""
    ops, mem = e.prefill.ops, e.mem
    out, calls = {}, []
    cur = {}  # name of the lane call in progress, rows

    def record(i, kind, ws, before):
        for k, v in vars(ws).items():
            if k != "tokens" and before.get(k) is not v:
                out[f"{cur['name']}/L{i:02d}/{kind}/{k}"] = v

    def wrap(name, fn):
        def f(i, ws, *a):
            before = dict(vars(ws))
            r = fn(i, ws, *a)
            record(i, name, ws, before)
            return r
        return f

    ops.dense, ops.attn = wrap("dense", ops.dense), wrap("attn", ops.attn)
    e.prefill.dense = ops.dense

    lm_head = ops.lm_head_sample

    def lm_head_sample(ws, samplers):
        h = ws.x[:, -1, :]
        logits = ops.lm.embed_tokens.as_linear(h) if ops.tied else mem.weights.lm_head(h)  # as GpuOps
        toks = lm_head(ws, samplers)
        argmax = np.array(mx.argmax(logits, axis=-1))
        assert argmax.tolist() == toks, "recomputed logits drifted from what GpuOps sampled (greedy)"
        out[f"{cur['name']}/logits"] = logits
        out[f"{cur['name']}/argmax"] = i32(argmax)
        out[f"{cur['name']}/sampled"] = i32(toks)
        return toks
    ops.lm_head_sample = lm_head_sample

    def after(rows):
        """rows: [(req, n_tokens)] -> K/V and state of each request."""
        for b, (r, n) in enumerate(rows):
            p = f"{cur['name']}/r{b}"
            for l in e.spec.full_layers:
                k, v = mem.read_kv(l, r.pages, n)
                out[f"{p}/L{l:02d}/k"] = mx.array(np.ascontiguousarray(k)).view(mx.bfloat16)
                out[f"{p}/L{l:02d}/v"] = mx.array(np.ascontiguousarray(v)).view(mx.bfloat16)
            _, views = mem.load_state(r.state_slots)
            for l, (conv, delta) in views.items():
                out[f"{p}/L{l:02d}/conv"] = mx.array(np.ascontiguousarray(conv)).view(mx.bfloat16)
                out[f"{p}/L{l:02d}/delta"] = mx.array(np.ascontiguousarray(delta))

    psub, dsub = e.prefill.submit, e.decode.submit

    def prefill_submit(w):
        cur["name"] = f"prefill{sum(c['name'].startswith('prefill') for c in calls)}"
        calls.append({"name": cur["name"], "rows": [{"req": w.req_id, "pos": w.start_pos, "n": w.n_valid}]})
        out[f"{cur['name']}/tokens"] = i32([w.tokens])
        psub(w)
        after([(e.reqs[w.req_id], w.start_pos + w.n_valid)])

    def decode_submit(w):
        cur["name"] = f"decode{w.step}"
        calls.append({"name": cur["name"], "rows": [{"req": r.req_id, "pos": r.pos, "n": 1} for r in w.rows]})
        out[f"{cur['name']}/tokens"] = i32([[r.token] for r in w.rows])
        dsub(w)
        after([(e.reqs[r.req_id], r.pos + 1) for r in w.rows])

    e.prefill.submit, e.decode.submit = prefill_submit, decode_submit
    # hold decode until both requests are prefilled, so every decode step is batch 2
    e.decode.can_submit = lambda: all(r.state == "DECODE" for r in e.reqs.values())
    a, c = e.submit(PROMPTS[0], max_tokens=N), e.submit(PROMPTS[2], max_tokens=N)
    while e.step_no < DECODE_STEPS:
        e.step()
    assert [c["name"] for c in calls] == ["prefill0", "prefill1", "prefill2", "decode0", "decode1", "decode2"], calls
    assert all(len(c["rows"]) == 2 for c in calls[3:])
    e.cancel(a), e.cancel(c)
    return out, calls


def main():
    model = sys.argv[1]
    eng = Engine(model, arena_bytes=256 << 20, chunk=CHUNK)
    s = eng.spec
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE / "golden" / _model_key(s.path)
    dst.mkdir(parents=True, exist_ok=True)

    tokens = {f"str/{i:02d}": i32(eng.tok.encode(t)) for i, t in enumerate(STRINGS)}
    for i, p in enumerate(PROMPTS):
        tokens[f"chat/{i}"] = i32(eng.tok.encode_chat(p))
        rid = eng.submit(p, max_tokens=N)
        toks = []
        while eng.reqs:
            toks += [t for r, t, _, _ in eng.step() if r == rid]
        tokens[f"greedy/{i}"] = i32(toks)
    mx.save_safetensors(str(dst / "tokens.safetensors"), tokens)

    tr, calls = trace(Engine(model, arena_bytes=256 << 20, chunk=CHUNK))
    mx.save_safetensors(str(dst / "trace.safetensors"), tr)

    spec = {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v) for k, v in vars(s).items()}
    meta = {"spec": spec, "strings": STRINGS, "chat": PROMPTS, "chunk": CHUNK,
            "page_tokens": eng.mem.T, "calls": calls}
    (dst / "spec.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    (dst / "weights.json").write_text(json.dumps(weight_hashes(s.path), indent=1, sort_keys=True) + "\n")
    print(f"{dst}: {len(tokens)} token arrays, {len(tr)} trace tensors")


if __name__ == "__main__":
    main()
