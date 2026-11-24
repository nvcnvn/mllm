# mmmllm
my mac mini llm: an LLM serving engine in C++/Metal, built for (and on) an Apple M4 Mac mini.
Bit-exact with mlx-lm at batch 1, ~1.05x its prefill; numbers in [reports/mlx-lm baseline.md](reports/mlx-lm%20baseline.md)
and [docs/target-27b.md](docs/target-27b.md).

**Runs on:** macOS 15+, M4 / M4 Pro GPUs (other chips stop with "dispatch rules not ported": the kernel choices are
copied from MLX for the M4 family only). **Models:** Qwen3.5 / Qwen3.6 in mlx-community 4-bit, e.g.
`mlx-community/Qwen3.5-0.8B-4bit`, `mlx-community/Qwen3.6-27B-4bit` (32 GB+ Mac).

## Use

```sh
uvx mmmllm serve mlx-community/Qwen3.5-0.8B-4bit          # http://127.0.0.1:8080/v1/completions
uvx mmmllm generate mlx-community/Qwen3.5-0.8B-4bit "Hello"
pip install mmmllm                                       # or install it; the command is the same
```

Repo ids download to the Hugging Face cache on first use. `serve` streams OpenAI-style completions:

```sh
curl -N localhost:8080/v1/completions -d '{"prompt": "The capital of France is", "max_tokens": 32, "stream": true}'
```

`mmmllm` with no arguments lists every command and flag (`--port`, `--max-batch`, `--max-ctx`, `--arena-gib`, ...).

## Verify the benchmark on your Mac

```sh
uvx --from 'mmmllm[compare]' mmmllm compare                         # 0.8B, -p 512 -g 128 -b 1 -n 3
uvx --from 'mmmllm[compare]' mmmllm compare MODEL -p 2048 -b 8       # flags go to both tools
```

It runs our `bench` and `mlx_lm benchmark` (same fields) with the same settings and prints both and the ratio.
Absolute tok/s depends on the chip, so compare the ratio, and close other apps first: it prints the load and the top
CPU users because a busy machine moves the numbers by several percent. Please open an issue with the whole output.

## From source

Needs Xcode (for the Metal compiler: `xcodebuild -downloadComponent MetalToolchain` if `xcrun metal` is missing).

```sh
uvx --from git+https://github.com/nvcnvn/mmmllm mmmllm serve MODEL   # builds the wheel (make) then runs
make -j && build/mllm serve MODEL                                  # or the binary alone; needs the model downloaded
```

Development (reference engine, golden data, the full test suite) uses `requirements.txt`, which pins the mlx-lm commit
the reports were measured with:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python py/test_engine.py                                  # Python reference engine == mlx-lm's greedy tokens
.venv/bin/python py/dump_golden.py mlx-community/Qwen3.5-0.8B-4bit   # its tokens and per-layer tensors, as golden data
build/tests/test_engine                                             # C++ engine: 3 prompts x 64 greedy tokens == golden
```

`make test` runs every test; some also need `py/dump_batch_ref.py` and the Neural Engine blocks (`py/mllm/ane.py`,
for `--prefill ane`); each prints the command it is missing.
