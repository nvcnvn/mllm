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
    return snapshot_download(model, allow_patterns=ALLOW)


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


def compare(args):
    try:
        md.version("mlx-lm")
    except md.PackageNotFoundError:
        sys.exit("mmmllm compare needs mlx-lm: pip install 'mmmllm[compare]'")
    model = args.pop(0) if args and not args[0].startswith("-") else DEFAULT
    args = ["-p", "512", "-g", "128", "-b", "1", "-n", "3", *args]  # later flags win in both tools
    path = local(model)
    ver = md.version("mmmllm") if BIN.parent == HERE else "source " + sh("git", "-C", HERE, "describe", "--always", "--dirty")
    print("== machine")
    print(f"chip:   {sh('sysctl', '-n', 'machdep.cpu.brand_string')}, {int(sh('sysctl', '-n', 'hw.memsize')) >> 30} GB")
    print(f"macOS:  {sh('sw_vers', '-productVersion')} ({sh('sw_vers', '-buildVersion')})")
    print(f"mmmllm: {ver}, mlx {md.version('mlx')}, mlx-lm {md.version('mlx-lm')}")
    print("load:   %.2f %.2f %.2f (a busy machine moves the numbers; top CPU users:)" % os.getloadavg())
    print("\n".join(sh("ps", "-Ao", "pcpu,comm", "-r").splitlines()[1:4]))
    print(f"args:   {model} {' '.join(args)}")
    print("\n== mmmllm")  # first: an unsupported GPU fails here, before the baseline's minutes
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
        if any(flag == "--prefill" and value == "ane" for flag, value in zip(a, a[1:])):
            build_ane_blocks(path, a)
    os.execv(BIN, ["mmmllm", *a])  # argv[0]: the name its usage and errors print
