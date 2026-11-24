"""quality.py MODEL NAME [--ref]: next-token quality of one variant on a fixed corpus, against a stored reference.
--ref: write the reference log-probs (float16, [N, V]) to OUT/ref_logprobs.npy. Otherwise: KL(ref || model) per token,
top-1 agreement, perplexity. Both: greedy 64 tokens on the 3 golden chat prompts. One model in memory at a time."""
import json, os, sys, threading, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

OUT = os.environ.get("QOUT", "/Volumes/T5/cache/mllm/g16/quality")
N, CHUNK, G = 4096, 512, 64
LIMIT = 38 << 30
PROMPTS = [  # py/test_engine.py PROMPTS
    [{"role": "user", "content": "Explain in a few sentences why the sky is blue, then list three "
      "other everyday things that Rayleigh scattering explains. Keep it short and use plain words "
      "a ten-year-old would understand. Do not use any formulas or jargon."}],
    [{"role": "user", "content": "Write a haiku about unified memory."}],
    [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
]

def watch():
    while True:
        if mx.get_active_memory() + mx.get_cache_memory() > LIMIT:
            print("watchdog abort", flush=True); os._exit(3)
        time.sleep(0.5)

def main():
    path, name = sys.argv[1:3]
    ref = "--ref" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    model, tok = load(path)
    print(f"{name}: loaded in {time.time() - t0:.0f} s", flush=True)
    ids_path = f"{OUT}/corpus_ids.npy"
    if not os.path.exists(ids_path):
        ids = tok.encode(open("/Volumes/T5/cache/mllm/g16/corpus.txt").read())
        assert len(ids) >= N, len(ids)
        np.save(ids_path, np.array(ids[:N], dtype=np.int32))
    ids = np.load(ids_path)
    cache = make_prompt_cache(model)
    rows = []
    mm = np.load(f"{OUT}/ref_logprobs.npy", mmap_mode="r") if not ref else None
    out = None
    t0 = time.time()
    for s in range(0, N, CHUNK):
        logits = model(mx.array(ids[None, s:s + CHUNK]), cache=cache)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if ref:
            if out is None:
                out = np.lib.format.open_memmap(f"{OUT}/ref_logprobs.npy", "w+", np.float16, (N, lp.shape[-1]))
            out[s:s + CHUNK] = np.array(lp.astype(mx.float16))
            continue
        lr = mx.array(mm[s:s + CHUNK]).astype(mx.float32)
        kl = (mx.exp(lr) * (lr - lp)).sum(-1)
        top = mx.argmax(lp, -1) == mx.argmax(lr, -1)
        nxt = ids[s + 1:s + CHUNK + 1]
        nll = -mx.take_along_axis(lp[:len(nxt)], mx.array(nxt)[:, None], axis=-1)[:, 0]
        nll_ref = -mx.take_along_axis(lr[:len(nxt)], mx.array(nxt)[:, None], axis=-1)[:, 0]
        mx.eval(kl, top, nll, nll_ref)
        rows.append((np.array(kl), np.array(top), np.array(nll), np.array(nll_ref)))
    print(f"{name}: corpus forward {time.time() - t0:.0f} s", flush=True)
    res = {"name": name, "path": path, "tokens": N}
    if ref:
        out.flush()
    else:
        kl, top, nll, nll_ref = (np.concatenate(c) for c in zip(*rows))
        kl, top = kl[:N - 1], top[:N - 1]  # positions with a known next token, as for ppl
        res.update(kl_mean=float(kl.mean()), kl_p99=float(np.percentile(kl, 99)), kl_max=float(kl.max()),
                   top1=float(top.mean()), ppl=float(np.exp(nll.mean())), ppl_ref=float(np.exp(nll_ref.mean())))
    greedy = []
    for p in PROMPTS:
        prompt = tok.apply_chat_template(p, add_generation_prompt=True)
        toks = [t for t, _ in generate_step(mx.array(prompt), model, max_tokens=G)]
        greedy.append([int(t) for t in toks])
    res["greedy"] = greedy
    res["peak_gib"] = mx.get_peak_memory() / 2**30
    json.dump(res, open(f"{OUT}/{name}.json", "w"))
    print(json.dumps({k: v for k, v in res.items() if k != "greedy"}), flush=True)

main()
