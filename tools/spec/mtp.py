# The 27B's MTP head for the engine (g22; docs/speculative.md "Plan" steps 1-3). Two jobs:
#
#   weights  Reads only the mtp.* tensors of the bf16 Qwen/Qwen3.6-27B (accept.py's load_mtp: tensor by tensor, +1 on
#            the norms), quantizes them as accept.py's reference does (nn.quantize, 4-bit, group 64; the norms stay
#            bf16) and writes <cache>/weights/<model_key>-v3/mtp/mtp.safetensors next to the engine's weights cache, laid
#            out like it (memory.mm aligned_weight_files: data at 0 mod 16, q|k|v and gate|up back to back). Then
#            re-reads the file and checks it equals a fresh reference byte for byte, and writes the reference's sha256
#            per tensor to <cache>/golden/<model_key>/mtp_weights.json (tests/test_weights.mm checks the engine's).
#            Run once per model, like the weights-cache rewrite. ~2 GB of memory, no 27B load.
#   golden   Reference data for tests/test_mtp.mm (loads the 27B: run under the heavy lock):
#            <cache>/golden/<model_key>/mtp_ref.safetensors, per golden chat prompt N = 0..2 (tokens.safetensors chat/N):
#              pN/seq     int32 [S]          prompt + GEN greedy tokens of the 27B (MLX)
#              pN/prompt  int32 [1]          P
#              pN/hidden  bf16 [S-1, H]      the 27B's final-normed hidden state at positions 0..S-2
#              pN/k, pN/v bf16 [kv_heads, S-1, head_dim]  the head's K/V after one teacher-forced pass over 0..S-2
#              pN/drafts  int32 [S-1-P, 2]   accept.py's chained drafts D[c][0..1] for c = P..S-2 (-1: not run,
#                                            step 1 missed)
#              pN/rows    int32 [R]          positions c with logits below
#              pN/logits  bf16 [R, vocab]    step 1's logits at state c (the head at position c - 1)
#
#   HF_HOME=/Volumes/T5/huggingface .venv/bin/python tools/spec/mtp.py weights
#   HF_HOME=/Volumes/T5/huggingface /Volumes/T5/cache/heavy g22 .venv/bin/python tools/spec/mtp.py golden
import hashlib
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accept  # noqa: E402

sys.path.insert(0, os.path.join(accept.REPO, "py"))
from mmmllm.ane import CACHE, _model_key  # noqa: E402

from mlx_lm.models.qwen3_5.config import ModelArgs  # noqa: E402
from mlx_lm.utils import hf_repo_to_path  # noqa: E402

GEN, LOGIT_ROWS = 128, 8
# memory.h MERGED: the engine multiplies these as one when there is one row
CHAINS = [("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), ("mlp.gate_proj", "mlp.up_proj")]


def model_dir():
    return str(hf_repo_to_path(accept.TARGET))


def reference_mtp():
    """accept.py's 4-bit MTP head, as {"mtp.<name>": array}."""
    args = ModelArgs.from_dict(json.load(open(os.path.join(model_dir(), "config.json")))).text
    mtp = accept.load_mtp(args)
    nn.quantize(mtp, group_size=64, bits=4)
    mx.eval(mtp.parameters())
    return mtp, {"mtp." + k: v for k, v in tree_flatten(mtp.parameters())}


def raw(a):
    """(safetensors dtype, bytes) of an mlx array"""
    if a.dtype == mx.bfloat16:
        return "BF16", np.array(a.view(mx.uint16)).tobytes()
    assert a.dtype == mx.uint32, a.dtype
    return "U32", np.array(a).tobytes()


def write_order(names):
    """names with each chain's members adjacent per suffix (weight / scales / biases), as the weights cache"""
    order = []
    for name in sorted(names):
        module, suffix = name.rsplit(".", 1)
        chain = next((c for c in CHAINS if module.endswith("." + c[0])), None)
        if chain:
            prefix = module[: -len(chain[0])]
            order += [f"{prefix}{m}.{suffix}" for m in chain]
        elif not any(module.endswith("." + m) for c in CHAINS for m in c[1:]):
            order.append(name)
    assert sorted(order) == sorted(names)
    return order


def write_aligned(tensors, path):
    header, offset, blobs = {}, 0, []
    for name in write_order(tensors):
        dtype, data = raw(tensors[name])
        offset = (offset + 15) // 16 * 16
        header[name] = {"dtype": dtype, "shape": list(tensors[name].shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append((offset, data))
        offset += len(data)
    text = json.dumps(header, separators=(",", ":"))
    text += " " * ((8 + len(text) + 15) // 16 * 16 - 8 - len(text))
    tmp = f"{path}.tmp{os.getpid()}"  # weights() removes a crashed run's copy
    with open(tmp, "wb") as f:
        f.write(len(text).to_bytes(8, "little") + text.encode())
        for at, data in blobs:
            f.write(b"\0" * (8 + len(text) + at - f.tell()))
            f.write(data)
    os.replace(tmp, path)


def weights():
    key = _model_key(model_dir())
    path = CACHE / "weights" / f"{key}-v3" / "mtp" / "mtp.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob("mtp.safetensors.tmp*"):  # a crashed run's partial copy
        stale.unlink()
    _, tensors = reference_mtp()
    write_aligned(tensors, str(path))
    _, fresh = reference_mtp()
    back = mx.load(str(path))
    assert sorted(back) == sorted(fresh), "tensor names differ"
    for name, want in fresh.items():
        assert raw(back[name]) == raw(want), name
    hashes = {}
    for name, want in fresh.items():
        dtype, data = raw(want)
        hashes[name] = {"dtype": dtype, "shape": list(want.shape), "sha256": hashlib.sha256(data).hexdigest()}
    golden = CACHE / "golden" / key
    golden.mkdir(parents=True, exist_ok=True)
    json.dump(hashes, open(golden / "mtp_weights.json", "w"), indent=1)
    print(f"{path}: {len(fresh)} tensors, {os.path.getsize(path) / 1e9:.3f} GB, equal to the reference byte for byte")


def golden():
    from mlx_lm import load

    key = _model_key(model_dir())
    out = CACHE / "golden" / key
    model, _ = load(accept.TARGET)
    mtp, _ = reference_mtp()
    tokens = mx.load(str(out / "tokens.safetensors"))
    accept.GEN = GEN
    result = {}
    for n in range(3):
        ids = np.array(tokens[f"chat/{n}"]).tolist()
        P = len(ids)
        gen, hidden, _ = accept.target_decode(model, ids)
        seq = ids + gen
        S = len(seq)
        D, o1, cache = accept.mtp_chains(mtp, model, seq, hidden, P, 2)
        k, v = cache
        rows = np.linspace(P, S - 2, LOGIT_ROWS).astype(np.int32)
        logits = model.lm_head(o1[mx.array(rows - P)])
        drafts = [D[c] + [-1] * (2 - len(D[c])) for c in range(P, S - 1)]
        result |= {f"p{n}/seq": mx.array(seq, mx.int32), f"p{n}/prompt": mx.array([P], mx.int32),
                   f"p{n}/hidden": hidden, f"p{n}/k": k[0], f"p{n}/v": v[0],
                   f"p{n}/drafts": mx.array(drafts, mx.int32), f"p{n}/rows": mx.array(rows),
                   f"p{n}/logits": logits.astype(mx.bfloat16)}
        print(f"prompt {n}: P {P}, S {S}", flush=True)
    mx.save_safetensors(str(out / "mtp_ref.safetensors"), result)


if __name__ == "__main__":
    {"weights": weights, "golden": golden}[sys.argv[1]]()
