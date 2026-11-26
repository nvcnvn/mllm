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
  attn.safetensors    GpuOps._full on the last full layer around MLX's SDPA switch points (random inputs, one row):
                      N_POS/q, k, v [n, heads, dim] (the C++ row-major layout), N_POS/gate [n, heads*dim]
                      N_POS/o [n, heads*dim]; N_POS/kv_k, kv_v [kv_heads, pos + n, dim] the pages after the call
                      (the first pos tokens are the random prefix the call read)
  lm_head.safetensors x [16, hidden] random bf16; M/logits [M, vocab] the tied lm_head (as GpuOps) of x[:M],
                      M = 1..16 (MLX's quantized matmul switches kernels by M)
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import mlx.core as mx
import numpy as np

from mmmllm import spec as spec_mod
from mmmllm.ane import CACHE, _model_key
from mllm.engine import Engine
from mllm.gpu import from_bits, to_bits
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


def arena_bytes(s):
    """256 MiB fits the trace's two requests on 0.8B; bigger models: 1 GiB (27B: 2 MiB units, 75 per state slot)."""
    return ARENA if s.n_layers <= 24 else 1 << 30


def kept_layers(s):
    """Layers whose per-layer trace tensors are written: all of them up to 24 layers. Above, the first layer, the first
    full one, a middle linear one and the last, each with the layer before it (its inputs), plus n_layers (the final
    norm). 27B: every layer would be ~2 GB."""
    if s.n_layers <= 24:
        return set(range(s.n_layers + 1))
    targets = {0, s.full_layers[0], s.linear_layers[len(s.linear_layers) // 2], s.n_layers - 1}
    return {j for t in targets for j in (t - 1, t) if j >= 0} | {s.n_layers}


def check_spec(s, args):
    """spec.py against mlx-lm's ModelArgs (the text config mlx-lm built the model from)."""
    t = args.text
    kinds = tuple("full" if (i + 1) % t.full_attention_interval == 0 else "linear" for i in range(t.num_hidden_layers))
    want = dict(kinds=kinds, hidden=t.hidden_size, vocab=t.vocab_size, n_heads=t.num_attention_heads,
                n_kv_heads=t.num_key_value_heads, head_dim=t.head_dim, lin_k_heads=t.linear_num_key_heads,
                lin_v_heads=t.linear_num_value_heads, lin_k_dim=t.linear_key_head_dim,
                lin_v_dim=t.linear_value_head_dim, conv_kernel=t.linear_conv_kernel_dim)
    got = {k: getattr(s, k) for k in want}
    assert got == want, {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    # what the C++ engine hardcodes (gpu.mm ROPE_*, rms_norm eps)
    assert t.rms_norm_eps == 1e-6 and t.rope_theta == 1e7 and t.partial_rotary_factor * t.head_dim == 64, t


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
    keep = kept_layers(e.spec)
    out, calls = {}, []
    cur = {}  # name of the lane call in progress, rows

    def record(i, kind, ws, before):
        for k, v in vars(ws).items():
            if k != "tokens" and before.get(k) is not v and i in keep:
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

    sample = ops.sample

    def sample_rec(ws, samplers):
        toks = sample(ws, samplers)
        argmax = np.array(mx.argmax(ws.logits, axis=-1))
        assert argmax.tolist() == toks, "greedy sample differs from argmax(logits)"
        out[f"{cur['name']}/logits"] = ws.logits
        out[f"{cur['name']}/argmax"] = i32(argmax)
        out[f"{cur['name']}/sampled"] = i32(toks)
        return toks
    ops.sample = sample_rec

    def after(rows):
        """rows: [(req, n_tokens)] -> K/V and state of each request."""
        for b, (r, n) in enumerate(rows):
            p = f"{cur['name']}/r{b}"
            for l in sorted(keep & set(e.spec.full_layers)):
                k, v = mem.read_kv(l, r.pages, n)
                out[f"{p}/L{l:02d}/k"] = mx.array(np.ascontiguousarray(k)).view(mx.bfloat16)
                out[f"{p}/L{l:02d}/v"] = mx.array(np.ascontiguousarray(v)).view(mx.bfloat16)
            _, views = mem.load_state(r.state_slots)
            for l, (conv, delta) in views.items():
                if l not in keep:
                    continue
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


# (n, pos) around the switch points of MLX v0.32.4 on an 's' GPU: sdpa_vector 1-pass / 2-pass at 1024 keys and its
# blocks (64 / 128 / 256, tk1 at 8 queries), the unfused fallback above 8 queries (27B: above 5) with gemv_wide up to 15, steel GEMM
# alignment (64-row / 64-key tiles, 16-key K), block / looped softmax at 4096 keys
SWEEP = [(1, 0), (1, 1022), (1, 1023), (1, 4200), (2, 0), (8, 40), (8, 1016), (4, 1100), (8, 9000), (9, 0), (9, 100),
         (11, 50), (13, 60), (15, 0), (15, 2000), (16, 0), (16, 48), (24, 32), (33, 7), (64, 0), (100, 4000),
         (128, 3968), (6, 30), (6, 1000), (7, 1100)]  # 6, 7: unfused at gqa 6 (gemv_wide nv3 / nv4)


def attn_sweep(eng):
    from types import SimpleNamespace
    ops, mem, s = eng.prefill.ops, eng.mem, eng.spec
    i, H, Hkv, D = s.full_layers[-1], s.n_heads, s.n_kv_heads, s.head_dim
    slots = list(range(mem.arena.n_units))[::-1]  # scattered: the last units first
    mx.random.seed(0)
    out = {}
    for n, pos in SWEEP:
        r = lambda *shape: mx.random.normal(shape).astype(mx.bfloat16)
        q, k, v, gate = r(1, n, H, D), r(1, n, Hkv, D), r(1, n, Hkv, D), r(1, n, H * D)
        prefix_k, prefix_v = r(pos, Hkv, D), r(pos, Hkv, D)
        if pos:
            mem.write_kv(i, slots, 0, to_bits(prefix_k), to_bits(prefix_v))
        ws = SimpleNamespace(q=q.transpose(0, 2, 1, 3), k=k.transpose(0, 2, 1, 3), v=v.transpose(0, 2, 1, 3), gate=gate)
        o = ops._full(i, ws, 0, pos, slots, None)
        kk, vv = mem.read_kv(i, slots, pos + n)
        p = f"{n}_{pos}"
        out |= {f"{p}/q": q[0], f"{p}/k": k[0], f"{p}/v": v[0], f"{p}/gate": gate[0], f"{p}/o": o[0],
                f"{p}/kv_k": from_bits(np.ascontiguousarray(kk), mx.bfloat16),
                f"{p}/kv_v": from_bits(np.ascontiguousarray(vv), mx.bfloat16)}
    return out


def lm_head_sweep(eng):
    ops = eng.prefill.ops
    mx.random.seed(1)
    x = mx.random.normal((16, eng.spec.hidden)).astype(mx.bfloat16)
    out = {"x": x}
    for M in range(1, 17):
        out[f"{M}/logits"] = ops.lm.embed_tokens.as_linear(x[:M]) if ops.tied else eng.mem.weights.lm_head(x[:M])
    return out


def main():
    model = sys.argv[1]
    arena = arena_bytes(spec_mod.load(model))
    eng = Engine(model, arena_bytes=arena, chunk=CHUNK)
    s = eng.spec
    check_spec(s, eng.mem.weights.args)
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

    mx.save_safetensors(str(dst / "attn.safetensors"), attn_sweep(eng))
    mx.save_safetensors(str(dst / "lm_head.safetensors"), lm_head_sweep(eng))
    meta = {"strings": STRINGS, "chat": CHATS, "eos": sorted(eng.tok.eos), "chunk": CHUNK,
            "page_tokens": eng.mem.T, "detok": detok_cases(eng.tok, greedy),
            "layout": {"arena_bytes": arena, **layout(eng.mem)}}
    del eng  # one model in memory at a time (27B: 16 GB each)
    mx.clear_cache()
    tr, calls = trace(Engine(model, arena_bytes=arena, chunk=CHUNK))
    mx.save_safetensors(str(dst / "trace.safetensors"), tr)

    spec = {k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v) for k, v in vars(s).items()}
    meta |= {"spec": spec, "calls": calls}
    (dst / "spec.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    (dst / "weights.json").write_text(json.dumps(weight_hashes(s.path), indent=1, sort_keys=True) + "\n")
    print(f"{dst}: {len(tokens)} token arrays, {len(tr)} trace tensors")


if __name__ == "__main__":
    main()
