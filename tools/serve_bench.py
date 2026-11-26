"""Concurrent-client load on an OpenAI-style /v1/completions server (mlx_lm.server now, ours later).

  HF_HOME=/Volumes/T5/huggingface .venv/bin/mlx_lm.server --model mlx-community/Qwen3.5-0.8B-4bit &
  .venv/bin/python tools/serve_bench.py --clients 1 4 16 -p 512 -g 256 -n 3

Every request gets its own random-token prompt, so the server's prompt cache never hits. --shared-prefix N: every
prompt starts with the same N random tokens (fixed for the run, warmup included), then its own; N = -p repeats one
prompt (with --clients 1: a repeated prompt's TTFT).
Per round of C clients started together:
  ttft      first streamed token, median / p90 over clients (time spent queued counts)
  decode    per-client (completion tokens - 1) / (last - first chunk), median
  total     all completion tokens / round wall time
ponytail: no ignore_eos in mlx_lm.server, so a request can stop before -g; tokens come from usage.
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request
from pathlib import Path

from mlx_lm.utils import load_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "py"))
from mmmllm.spec import resolve


def one(url, model, prompt, g, out):
    body = {"model": model, "prompt": prompt, "max_tokens": g, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, last, usage = time.perf_counter(), None, None, None
    for attempt in range(20):  # http.server's listen backlog is 5: extra connects get reset; retry time counts in ttft
        try:
            r = urllib.request.urlopen(req)
            break
        except OSError:  # URLError, ConnectionResetError
            time.sleep(0.05 * (attempt + 1))
    with r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices"):  # text or reasoning: any streamed token counts
                last = time.perf_counter()
                first = first or last
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    out.append((first - t0, last - first, usage["prompt_tokens"], usage["completion_tokens"], cached))


def random_text(tok, n):
    return tok.decode([random.randrange(1000, 100_000) for _ in range(n)])


def round_(args, tok, c, shared):
    prompts = [shared + random_text(tok, args.prompt_tokens - args.shared_prefix) for _ in range(c)]
    out, t0 = [], time.perf_counter()
    ts = [threading.Thread(target=one, args=(args.url, args.model, p, args.generation_tokens, out)) for p in prompts]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    assert len(out) == c, "a request failed"
    ttft = sorted(o[0] for o in out)
    return dict(ttft=statistics.median(ttft), ttft_p90=ttft[int(0.9 * (c - 1))],
                decode=statistics.median((o[3] - 1) / o[1] for o in out if o[1] > 0),
                total=sum(o[3] for o in out) / wall,
                p_tok=statistics.mean(o[2] for o in out), g_tok=statistics.mean(o[3] for o in out),
                c_tok=statistics.mean(o[4] for o in out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/completions")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-4bit")
    ap.add_argument("--clients", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("-p", "--prompt-tokens", type=int, default=512)
    ap.add_argument("-g", "--generation-tokens", type=int, default=256)
    ap.add_argument("-n", "--trials", type=int, default=3)
    ap.add_argument("--shared-prefix", type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.shared_prefix <= args.prompt_tokens:
        ap.error("--shared-prefix must be in [0, -p]")
    random.seed(0)
    tok = load_tokenizer(resolve(args.model))
    shared = random_text(tok, args.shared_prefix) if args.shared_prefix else ""
    round_(args, tok, 2, shared)  # warmup
    print(f"p={args.prompt_tokens} g={args.generation_tokens} shared prefix={args.shared_prefix}")
    print(f"{'C':>3} {'trial':>5} {'ttft s':>7} {'p90 s':>7} {'tok/s/client':>13} {'total tok/s':>12} {'p tok':>6} {'g tok':>6}"
          f" {'cached':>6}")
    for c in args.clients:
        for i in range(args.trials):
            s = round_(args, tok, c, shared)
            print(f"{c:3d} {i + 1:5d} {s['ttft']:7.3f} {s['ttft_p90']:7.3f} {s['decode']:13.1f} {s['total']:12.1f} "
                  f"{s['p_tok']:6.0f} {s['g_tok']:6.0f} {s['c_tok']:6.0f}", flush=True)


if __name__ == "__main__":
    main()
