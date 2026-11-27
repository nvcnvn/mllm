"""mmmllm: downloads the model if needed (and with `--prefill ane`, builds the Neural Engine blocks), then runs the
C++ engine (generate, chat, serve, bench, bench-ops). `mmmllm compare` runs mlx-lm's benchmark and ours on the same
settings and prints the ratio."""
import importlib.metadata as md
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
# a wheel puts the binary next to this file; a source checkout (PYTHONPATH=py) uses make's build/mllm
BIN = next((p for p in (HERE / "mllm", HERE.parents[1] / "build/mllm") if p.exists()), None)
DEFAULT = "mlx-community/Qwen3.5-0.8B-4bit"
ALLOW = ["*.json", "model*.safetensors", "tokenizer.model", "*.txt", "*.jinja"]  # what mlx-lm fetches, minus code
ANE_SIZES = [512, 1024]  # the blocks AneOps loads: 512 and the chunk (src/main.mm prefill_chunk)


def local(model):
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download
    try:  # cached: no network, no "Fetching N files" progress line
        return snapshot_download(model, allow_patterns=ALLOW, local_files_only=True)
    except Exception:
        return snapshot_download(model, allow_patterns=ALLOW)


# A model name part -> the HF repo of its MTP draft head (docs/speculative.md): mlx-community's conversions drop it.
DRAFTS = {"Qwen3.6-27B": "nvcnvn/Qwen3.6-27B-MTP-4bit"}


def draft(model):
    """The draft head's file for speculative decoding, downloaded on first use; None: plain decode."""
    repo = next((r for name, r in DRAFTS.items() if name in model), None)
    if not repo:
        return None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(repo, "mtp.safetensors", local_files_only=True)
    except Exception:
        pass
    print(f"mmmllm: fetching the draft head for speculative decoding (about 1.8x faster greedy replies): {repo}, "
          "0.24 GB", file=sys.stderr, flush=True)
    try:
        return hf_hub_download(repo, "mtp.safetensors")
    except Exception as e:
        print(f"mmmllm: no draft head ({type(e).__name__}), decoding without speculation", file=sys.stderr)
        return None


def tee(cmd):
    sys.stdout.flush()  # our header before the child's stderr, also when redirected to a file
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    out = ""
    for line in p.stdout:
        print(line, end="", flush=True)
        out += line
    if p.wait():
        sys.exit(p.returncode)
    m = re.search(r"^Averages: prompt_tps=([\d.]+), generation_tps=([\d.]+)", out, re.M)
    return float(m[1]), float(m[2])


def build_ane_blocks(path, args):
    """The Neural Engine blocks `--prefill ane` loads, built once per model (~30 s at 27B); a no-op when they exist."""
    from . import ane
    if not ane.missing(path, ANE_SIZES):
        return
    # a child process: no coremltools resource tracker outlives our exec (it warned of leaked semaphores)
    code = subprocess.run([sys.executable, "-m", "mmmllm.ane", path, *map(str, ANE_SIZES)]).returncode
    if code == 3:  # coremltools has wheels up to Python 3.13
        sys.exit("mmmllm: --prefill ane first builds Neural Engine blocks with coremltools: "
                 f"uvx --python 3.13 --from 'mmmllm[ane]' mmmllm {shlex.join(args)}")
    if code:
        sys.exit(code)


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


# Each takes a path MLX picks per chip (src/gpu_rules.h): 11 prefill rows (12 tokens with the template) fall between
# chips' qmv / qmm batch limits; ~1.2k keys between their one-pass / two-pass attention thresholds (1024 / 4096).
PROMPTS = ["Hi!",
           "Write a Python function that checks whether a number is prime.",
           "Here are some facts. " + " ".join(f"Fact {i}: {i} times {i} is {i * i}." for i in range(60)) +
           " Which fact has the largest number?"]


def first_difference(tokenizer, ids, text):
    """The first of mlx-lm's tokens `ids` whose text leaves `text` (ours), or None when the texts are equal."""
    if tokenizer.decode(ids) == text:
        return None
    return next((i for i in range(len(ids)) if not text.startswith(tokenizer.decode(ids[:i + 1]))), len(ids))


def correctness(path, n=64):
    """Greedy n tokens of each prompt (chat template) from both engines. Per prompt: `bit-exact` when every step's
    logits are equal bit for bit (prefill and B = 1 decode match mlx-lm exactly under the same GPU rules), else the
    largest |difference| and the first step that differs; then whether the texts agree (a weaker check: argmax
    survives small differences)."""
    import tempfile
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    ours = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, p in enumerate(PROMPTS):  # stderr: its "mllm: weights ..." line, shown only when it fails
            out = Path(tmp) / f"{k}.bf16"
            r = subprocess.run([str(BIN), "generate", path, p, "--max-tokens", str(n), "--logits", str(out),
                                "--prefix-cache", "off"],  # its snapshot points split chunks where mlx-lm doesn't
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit(r.stderr.strip() or f"mmmllm generate: exit {r.returncode}, no message")
            bits = np.fromfile(out, dtype=np.uint16).astype(np.uint32) << 16  # bf16 -> float32, exactly
            ours.append((r.stdout.removesuffix("\n"), bits.view(np.float32)))
    model, tokenizer = load(path)
    exact = same = 0
    for k, (prompt, (text, mine)) in enumerate(zip(PROMPTS, ours), 1):
        steps = []

        def keep(_, logits):  # sees each step's logits before the (greedy) sampler
            steps.append(logits)
            return logits
        ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
        ids = [r.token for r in stream_generate(model, tokenizer, ids, max_tokens=n, logits_processors=[keep],
                                                prefill_step_size=512)]  # = our GPU chunk
        theirs = np.array(mx.concatenate(steps[:len(ids)]).astype(mx.float32))  # it may run a step ahead
        mine = mine.reshape(-1, theirs.shape[1])
        m = min(len(mine), len(theirs))
        diff = np.abs(mine[:m] - theirs[:m]).max(axis=1)
        exact += not diff.any()
        verdict = (f"bit-exact ({m} steps)" if not diff.any() else
                   f"max |diff| {diff.max():.4g}, first differing step {int(np.argmax(diff > 0))} of {m}")
        ids = [t for t in ids if t not in tokenizer.eos_token_ids]
        i = first_difference(tokenizer, ids, text)
        same += i is None
        print(f"prompt {k}: {verdict}; text " + ("identical" if i is None else f"differs from token {i} of {len(ids)}"))
        if i is not None:
            at = len(tokenizer.decode(ids[:i]))
            print(f"  mmmllm: {text[max(0, at - 40):at + 40]!r}\n  mlx-lm: {tokenizer.decode(ids)[max(0, at - 40):at + 40]!r}")
    print(f"correctness: {exact}/{len(PROMPTS)} bit-exact (every step's logits), {same}/{len(PROMPTS)} identical text")
    del model  # mlx-lm's benchmark below loads it again in its own process
    import gc, mlx.core as mx
    gc.collect()
    mx.clear_cache()


def compare(args):
    try:
        md.version("mlx-lm")
    except md.PackageNotFoundError:
        sys.exit("mmmllm compare needs mlx-lm: pip install 'mmmllm[compare]'")
    model = args.pop(0) if args and not args[0].startswith("-") else DEFAULT
    args = ["-p", "512", "-g", "128", "-b", "1", "-n", "3", *args]  # later flags win in both tools
    path = local(model)
    ver = md.version("mmmllm") if BIN.parent == HERE else "source " + sh("git", "-C", HERE, "describe", "--always", "--dirty")
    import mlx.core as mx
    print("== machine")
    print(f"chip:   {sh('sysctl', '-n', 'machdep.cpu.brand_string')} ({mx.device_info()['architecture']}), "
          f"{int(sh('sysctl', '-n', 'hw.memsize')) >> 30} GB")
    print(f"macOS:  {sh('sw_vers', '-productVersion')} ({sh('sw_vers', '-buildVersion')})")
    print(f"mmmllm: {ver}, mlx {md.version('mlx')}, mlx-lm {md.version('mlx-lm')}")
    print("load:   %.2f %.2f %.2f (a busy machine moves the numbers; top CPU users:)" % os.getloadavg())
    for line in sh("ps", "-Ao", "pcpu,comm", "-r").splitlines()[1:4]:
        cpu, comm = line.split(None, 1)
        print(f"{cpu:>6} {Path(comm).name}")
    print(f"args:   {model} {' '.join(args)}")
    print("\n== correctness (greedy 64 tokens: each step's logits and the text, mmmllm vs mlx-lm)")  # first: fails fast, as does an unsupported GPU
    correctness(path)
    print("\n== mmmllm")
    ours = tee([str(BIN), "bench", path, *args])
    print("\n== mlx-lm")
    base = tee([sys.executable, "-m", "mlx_lm", "benchmark", "--model", path, *args])
    print("\n== mmmllm / mlx-lm (peak memory isn't comparable: mlx-lm's includes its buffer cache)")
    for name, a, b in (("prefill", ours[0], base[0]), ("decode ", ours[1], base[1])):
        print(f"{name} {a:.0f} / {b:.0f} = {a / b:.2f}x")


def main():
    a = sys.argv[1:]
    if not BIN:
        sys.exit("mmmllm: no engine binary (from a source checkout, run make)")
    if a[:1] == ["compare"]:
        return compare(a[1:])
    if not a or a[0] in ("-h", "--help"):
        print("usage: mmmllm compare [MODEL] [-p 512] [-g 128] [-b 1] [-n 3]   mlx-lm and mmmllm, same settings, ratio\n"
              "       (MODEL defaults to " + DEFAULT + "; repo ids download on first use)", file=sys.stderr)
        a = []
    i = 1 if a[:1] and a[0] in ("generate", "chat", "bench", "serve", "bench-ops") else 0
    if len(a) > i and not a[i].startswith("-"):
        path = local(a[i])  # keep the repo id (API responses name the model): the engine finds it in the same cache
        from huggingface_hub import constants
        os.environ["HF_HUB_CACHE"] = constants.HF_HUB_CACHE
        if a[0] != "bench" and (head := draft(path)):  # bench always decodes plainly
            os.environ["MLLM_DRAFT"] = head
        if any(flag == "--prefill" and value == "ane" for flag, value in zip(a, a[1:])):
            build_ane_blocks(path, a)
    os.execv(BIN, ["mmmllm", *a])  # argv[0]: the name its usage and errors print
