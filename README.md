# mllm
My LLM serving engine specially build to run on my Apple M4 Mac Mini.
C++/Metal, Qwen3.5/3.6 (mlx-community 4-bit), bit-exact with mlx-lm at batch 1. Numbers vs mlx-lm: [reports/mlx-lm baseline.md](reports/mlx-lm%20baseline.md), [docs/target-27b.md](docs/target-27b.md).

## Reproduce the benchmark on your Mac

Needs an Apple-silicon Mac, Xcode (command line tools are not enough: the Metal compiler) and Python 3.12.

```sh
git clone https://github.com/nvcnvn/mllm && cd mllm
xcrun -sdk macosx metal --version || xcodebuild -downloadComponent MetalToolchain
make -j                                   # build/mllm; don't move build/, the binary finds its kernels there
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # pinned mlx + mlx-lm baseline
.venv/bin/hf download mlx-community/Qwen3.5-0.8B-4bit              # mllm never downloads; ~0.6 GB
tools/repro.sh                            # mlx-lm and mllm, same settings, then the ratio
```

`tools/repro.sh [MODEL] [-p P] [-g G] [-b B] [-n N]` passes the flags to both `mlx_lm benchmark` and `build/mllm bench`
(defaults `-p 512 -g 128 -b 1 -n 3`). The 27B: `hf download mlx-community/Qwen3.6-27B-4bit` (~16 GB), then
`tools/repro.sh mlx-community/Qwen3.6-27B-4bit -p 2048`; it wants a 32 GB+ Mac.

Absolute tok/s depends on the chip, so compare the ratio, and close other apps first: the script prints the load and
the top CPU users because a busy machine moves the numbers by several percent.
Please open an issue with the script's whole output, whatever the result.

Models load from the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`) or a
local directory.

## Check it's correct, not just fast

```sh
.venv/bin/python py/test_engine.py                                  # Python reference engine == mlx-lm's greedy tokens
.venv/bin/python py/dump_golden.py mlx-community/Qwen3.5-0.8B-4bit   # its tokens and per-layer tensors, as golden data
build/tests/test_engine                                             # C++ engine: 3 prompts x 64 greedy tokens == golden
```

`make test` runs every test, some of which also need `py/dump_batch_ref.py` and the Neural Engine blocks
(`py/mllm/ane.py`); each prints the command it is missing.
