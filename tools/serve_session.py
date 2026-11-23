"""Sessions against `mllm serve` (target-27b.md §4 budget, §6 multi-turn). Streams /v1/completions, greedy.

  build/mllm serve mlx-community/Qwen3.6-27B-4bit --port 18015 &
  .venv/bin/python tools/serve_session.py turns --system 4000 --turns 10 -g 64
  .venv/bin/python tools/serve_session.py mixed --minutes 25

turns: one conversation, a ~N-token system prompt then T user turns; each prompt = the previous prompt + its reply +
  the next user message (chat markup written out, so every prompt extends the last). Prints each turn's TTFT and the
  cached tokens; then the conversation's turn 1 asked again with a different question twice (the second hits the
  system prompt's branch point). --replay FILE: send only the last turn's prompt saved by an earlier run (cold
  re-prefill on a --prefix-cache off server).
mixed: --clients threads (default 4) for --minutes, each sending a request then idling 0-20 s; prompts 64..8192
  tokens (a third start with one shared 2048-token prefix), 16..256 tokens out. Prints each request (a failed one too)
  and, at the end, the longest gap between two streamed tokens.
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request
from pathlib import Path

from mlx_lm.utils import load_tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "py"))
from mllm.spec import resolve


def stream(url, prompt, g):
    """-> (ttft s, longest gap between tokens s, text, usage)"""
    body = {"prompt": prompt, "max_tokens": g, "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, first, last, gap, text, usage = time.perf_counter(), None, None, 0.0, "", None
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            if d.get("choices"):
                now = time.perf_counter()
                if last is not None:
                    gap = max(gap, now - last)
                first, last = first or now, now
                text += d["choices"][0].get("text", "")
    return first - t0, gap, text, usage


def random_text(tok, n):
    return tok.decode([random.randrange(1000, 100_000) for _ in range(n)])


def user(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def turns(args, tok):
    if args.replay:
        p = Path(args.replay).read_text()
        ttft, _, _, u = stream(args.url, p, args.g)
        print(f"replay: {u['prompt_tokens']} prompt tokens, ttft {ttft:.2f} s, cached {u['prompt_tokens_details']['cached_tokens']}")
        return
    sys_text = "You are a careful assistant. Reference notes follow.\n" + random_text(tok, args.system)
    prompt = f"<|im_start|>system\n{sys_text}<|im_end|>\n"
    print(f"{'turn':>4} {'prompt tok':>10} {'cached':>6} {'ttft s':>7}")
    for i in range(args.turns):
        prompt += user(f"Question {i + 1}: what does note {random.randrange(100)} say, in one sentence?")
        ttft, _, text, u = stream(args.url, prompt, args.g)
        print(f"{i + 1:4d} {u['prompt_tokens']:10d} {u['prompt_tokens_details']['cached_tokens']:6d} {ttft:7.2f}", flush=True)
        if i + 1 < args.turns:
            prompt += text + "<|im_end|>\n"
    if args.save:
        Path(args.save).write_text(prompt)
    for k in (1, 2):  # a new conversation on the same system prompt: the first snapshots the branch point, then hits
        p = f"<|im_start|>system\n{sys_text}<|im_end|>\n" + user(f"Other question {k}: summarize note {k}.")
        ttft, _, _, u = stream(args.url, p, args.g)
        print(f"new conversation {k}: {u['prompt_tokens']} prompt tokens, cached {u['prompt_tokens_details']['cached_tokens']}, "
              f"ttft {ttft:.2f} s", flush=True)


def mixed(args, tok):
    shared = random_text(tok, 2048)
    end = time.time() + args.minutes * 60
    lock, gaps, n = threading.Lock(), [], [0]

    def client(c):
        rnd = random.Random(c)
        while time.time() < end:
            p_tok, g = int(2 ** rnd.uniform(6, 13)), int(2 ** rnd.uniform(4, 8))
            p = (shared + random_text(tok, max(16, p_tok - 2048))) if rnd.random() < 1 / 3 else random_text(tok, p_tok)
            t0 = time.time()
            try:
                ttft, gap, _, u = stream(args.url, p, g)
            except Exception as e:  # one failed stream: report it, keep the client going
                print(f"{time.strftime('%H:%M:%S')} client {c}: request failed: {e!r}", flush=True)
                time.sleep(1)
                continue
            with lock:
                n[0] += 1
                gaps.append(gap)
                print(f"{time.strftime('%H:%M:%S')} client {c}: {u['prompt_tokens']:5d} prompt "
                      f"({u['prompt_tokens_details']['cached_tokens']:5d} cached), {u['completion_tokens']:3d} out, "
                      f"ttft {ttft:6.2f} s, total {time.time() - t0:6.1f} s, longest gap {gap * 1e3:5.0f} ms", flush=True)
            time.sleep(rnd.uniform(0, 20))

    ts = [threading.Thread(target=client, args=(c,)) for c in range(args.clients)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"mixed: {n[0]} requests, longest gap between tokens {max(gaps) * 1e3:.0f} ms, "
          f"median of per-request longest {statistics.median(gaps) * 1e3:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["turns", "mixed"])
    ap.add_argument("--url", default="http://127.0.0.1:18015/v1/completions")
    ap.add_argument("--model", default="mlx-community/Qwen3.6-27B-4bit")
    ap.add_argument("-g", type=int, default=64)
    ap.add_argument("--system", type=int, default=4000)
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--save")
    ap.add_argument("--replay")
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--clients", type=int, default=4)
    args = ap.parse_args()
    random.seed(0)
    tok = load_tokenizer(resolve(args.model))
    (turns if args.mode == "turns" else mixed)(args, tok)


if __name__ == "__main__":
    main()
