"""Reference data for the C++ port: run the Python engine (GPU prefill) once and write what it saw.

  HF_HOME=/Volumes/T5/huggingface .venv/bin/python py/dump_golden.py MODEL [OUT]

OUT defaults to $MLLM_CACHE/golden/<model_key>/ (model_key as in ane.py). Reruns are byte-identical.

Files and keys (bf16 stays bf16, ints are int32):
  spec.json           {"spec": ModelSpec fields, "strings": [...], "chat": [message lists; the first 3 are PROMPTS],
                       "eos": [ids], "chunk", "page_tokens", "calls": [{"name", "rows": [{"req", "pos", "n"}]}],
                       "detok": [{"ids", "segments"}]  Detokenizer push(id) per id, then flush(): len(ids) + 1,
                       "layout": Memory(arena_bytes, page_tokens=32) sizes, kv view strides (bytes) and
                                 byte offsets from numpy pointers: "kv" [[slot, g, kv, head, token, offset]],
                                 "state" [[layer, conv offset, delta offset]] within a state slot span}
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
    "It's they'LL we'Re I'M he'd you've CAN'T 's",
    "a\n\n\nb\n \n\tc\r\r\n\n\t\t x",
    "Cafe\u0301 nai\u0308ve n\u0303o A\u030a \ufb01 \u1100\u1161\u11a8 é",
    "12345678901234567890 \u0661\u0662\u0663 0.000001",
    "Mixed: English, Русский, Ελληνικά, العربية, עברית, हिन्दी, ไทย",
    "<|im_start|><|im_end|><|im_start|>x<tool_call>a</tool_call><think>",
]
# system + multi-turn, with a thinking block in assistant history (the template strips it) and padded content
MULTI = [{"role": "system", "content": "You are terse. "},
         {"role": "user", "content": "  Hi there\n"},
         {"role": "assistant", "content": "<think>\nhmm\n</think>\n\nHello!"},
         {"role": "user", "content": "Bye?"}]
# Unicode whitespace padding (NBSP, U+3000): the template's |trim strips it like str.strip
PAD = [{"role": "system", "content": "\u3000sys\u00a0"}, {"role": "user", "content": "\u00a0\u3000 pad \u00a0\u3000"}]
CHATS = PROMPTS + [MULTI, PAD]
DECODE_STEPS = 3
ARENA = 256 << 20


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


def detok(tok, ids):
    d = tok.detokenizer()
    return {"ids": ids, "segments": [d.push(t) for t in ids] + [d.flush()]}


def detok_cases(tok, greedy):
    """The greedy outputs, plus one sequence that splits multibyte characters across tokens, has a lone
    continuation byte (U+FFFD), special tokens, and leading/lone space tokens."""
    from mlx_lm.tokenizer_utils import _byte_decoder
    t = tok._t
    byte_tok = {b: t.convert_tokens_to_ids(c) for c, b in _byte_decoder().items()}
    ids = (tok.encode(" Hello") + tok.encode("emoji: 😀👍🏽 👨‍👩‍👧‍👦 🇻🇳 你好") + [byte_tok[0x80]] + tok.encode("a")
           + tok.encode("<|im_end|>\n<think>") + [byte_tok[0x20]] + tok.encode("x") + [byte_tok[0xE4], byte_tok[0xBD]]
           + tok.encode("<|endoftext|>") + [byte_tok[0x20]])
    cases = [detok(tok, g) for g in greedy] + [detok(tok, ids)]
    assert "" in cases[-1]["segments"][:-1], "no token was held back"
    return cases


def layout(mem):
    base = mem.arena.bytes.ctypes.data
    kv = [[s, g, k, h, t, mem.kv[s, g, k, h, t].ctypes.data - base]
          for s, g, k, h, t in [(0, 0, 0, 0, 0), (0, 0, 1, 0, 0), (1, 0, 0, 1, 0), (2, 1, 1, 1, 31), (5, len(mem.g_of) - 1, 1, 0, 7)]]
    buf, views = mem.load_state(list(range(mem.state_units)))
    state = [[l, conv.ctypes.data - buf.ctypes.data, delta.ctypes.data - buf.ctypes.data] for l, (conv, delta) in views.items()]
    return {"unit_bytes": mem.arena.unit_bytes, "n_units": mem.arena.n_units, "state_units": mem.state_units,
            "conv_bytes": mem.conv_bytes, "delta_bytes": mem.delta_bytes, "kv_strides": list(mem.kv.strides),
            "kv": kv, "state": state}


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
    eng = Engine(model, arena_bytes=ARENA, chunk=CHUNK)
    s = eng.spec
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE / "golden" / _model_key(s.path)
    dst.mkdir(parents=True, exist_ok=True)

    tokens = {f"str/{i:02d}": i32(eng.tok.encode(t)) for i, t in enumerate(STRINGS)}
    for i, p in enumerate(CHATS):
        tokens[f"chat/{i}"] = i32(eng.tok.encode_chat(p))
    greedy = []
    for i, p in enumerate(PROMPTS):
        rid = eng.submit(p, max_tokens=N)
        toks = []
        while eng.reqs:
            toks += [t for r, t, _, _ in eng.step() if r == rid]
        tokens[f"greedy/{i}"] = i32(toks)
        greedy.append(toks)
    mx.save_safetensors(str(dst / "tokens.safetensors"), tokens)

    tr, calls = trace(Engine(model, arena_bytes=ARENA, chunk=CHUNK))
    mx.save_safetensors(str(dst / "trace.safetensors"), tr)

    spec = {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v) for k, v in vars(s).items()}
    meta = {"spec": spec, "strings": STRINGS, "chat": CHATS, "eos": sorted(eng.tok.eos), "chunk": CHUNK,
            "page_tokens": eng.mem.T, "calls": calls, "detok": detok_cases(eng.tok, greedy),
            "layout": {"arena_bytes": ARENA, **layout(eng.mem)}}
    (dst / "spec.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    (dst / "weights.json").write_text(json.dumps(weight_hashes(s.path), indent=1, sort_keys=True) + "\n")
    print(f"{dst}: {len(tokens)} token arrays, {len(tr)} trace tensors")


if __name__ == "__main__":
    main()
