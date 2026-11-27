# mmmllm
my mac mini llm: an LLM serving engine in C++/Metal, built for (and on) an Apple M4 Mac mini.
Bit-exact with mlx-lm at batch 1, ~1.05x its prefill; numbers in [reports/mlx-lm baseline.md](reports/mlx-lm%20baseline.md)
and [docs/target-27b.md](docs/target-27b.md).

**Runs on:** macOS 15+, Apple M1 to M4 GPUs including Pro / Max / Ultra: kernel and tile choices follow MLX's rules
for each chip (src/gpu_rules.h), so results stay bit-exact with mlx-lm there; tested on an M4 Pro; M1-M3 are untested,
reports welcome (see below). M5 stops with a clear message (MLX's neural-accelerator kernels aren't ported). **Models:** Qwen3.5 / Qwen3.6 in mlx-community 4-bit, e.g.
`mlx-community/Qwen3.5-0.8B-4bit`, `mlx-community/Qwen3.6-27B-4bit` (32 GB+ Mac).
**Memory:** the KV cache (arena), max context and serve's max batch shrink to what the GPU may use (Metal's
recommended working set: ~72% of RAM on 8 / 16 / 24 GB Macs, ~87% at 64 GB) and a model that can't fit stops before loading, with a clear message.
Which model: the 0.8B (0.4 GB of weights) on any Mac; `mlx-community/Qwen3.5-9B-4bit` (5 GB; bit-exact with mlx-lm)
on 16 / 24 GB; the 27B (15 GB) on 32 GB+. `MLLM_GPU_BUDGET_GB=11.5` tries a 16 GB Mac's limit.

## Use

```sh
uvx mmmllm chat mlx-community/Qwen3.5-0.8B-4bit           # multi-turn chat in the terminal; Ctrl-C stops a reply
uvx mmmllm serve mlx-community/Qwen3.5-0.8B-4bit          # OpenAI-style API on http://127.0.0.1:8080/v1
uvx mmmllm generate mlx-community/Qwen3.5-0.8B-4bit "Hello"
pip install mmmllm                                       # or install it; the command is the same
```

Repo ids download to the Hugging Face cache on first use. `serve` speaks the OpenAI API (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`; streamed or not, `stop` strings), so the openai clients and chat UIs work with
`base_url="http://127.0.0.1:8080/v1"`. The model thinks first: chat replies carry that part as `reasoning_content`.
`usage` adds `ttft_ms` (arrival to first token) and `tokens_per_second` (decode, after the first token); the server
logs the same per request on stderr, and `chat` prints them after each reply.

```sh
curl localhost:8080/v1/chat/completions -d '{"messages": [{"role": "user", "content": "Hi!"}]}'
curl -N localhost:8080/v1/completions -d '{"prompt": "The capital of France is", "max_tokens": 32, "stream": true}'
```

`mmmllm` with no arguments lists every command and flag (`--port`, `--max-batch`, `--max-ctx`, `--arena-gib`, ...).

**Speculative decoding** (27B): greedy replies (temperature 0) from one request at a time go about 1.8x faster
on an M4 Pro (about 25 vs 14 tok/s). A small draft layer guesses 2 tokens ahead and the 27B checks them in one pass
([speculative.md](docs/speculative.md)). The draft is Qwen's own MTP head, which mlx-community's conversion drops: on
first use of the 27B, `mmmllm` downloads it (0.24 GB, [nvcnvn/Qwen3.6-27B-MTP-4bit](https://huggingface.co/nvcnvn/Qwen3.6-27B-MTP-4bit))
to the Hugging Face cache. Without it the 27B decodes as before. The output matches plain decoding except at
near-ties between two tokens.

**Neural Engine prefill**: `--prefill ane` runs prefill's matmuls on the Neural Engine, with the GPU feeding it its
own 4-bit weights a layer at a time (no second copy; ~3 GB more memory at 27B). At 27B on an M4 Pro: 4096-token
prompts 191 vs 131 tok/s, and decode next to a long prefill keeps 12 tok/s instead of 3
([target-27b.md](docs/target-27b.md) §5). Its Core ML blocks are built with coremltools (the `ane` extra, Python
3.13 at most) on the first run, once per model (~30 s):

```sh
uvx --python 3.13 --from 'mmmllm[ane]' mmmllm chat mlx-community/Qwen3.6-27B-4bit --prefill ane
```

## Help test on your Mac

Install [uv](https://docs.astral.sh/uv/) (`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`),
close other apps, then:

```sh
uvx --python 3.12 --from 'mmmllm[compare]' mmmllm compare
```

It runs both engines greedily on 3 fixed prompts and checks every step's logits against mlx-lm's, bit for bit
(`prompt k: bit-exact`, or the largest difference and the first step that differs; then
`correctness: k/3 bit-exact`). Then it runs our `bench` and `mlx_lm benchmark` with the same settings (0.8B,
`-p 512 -g 128 -b 1 -n 3`; add `MODEL` and flags after `compare`, both tools get them) and prints the speed ratio.
Absolute tok/s depends on the chip, so compare the ratio. It prints the load and the top CPU users because a busy
machine moves the numbers by several percent. Please paste the whole output into a
[chip report](https://github.com/nvcnvn/mmmllm/issues/new?template=chip-report.md).

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

`make test` runs every test; some also need `py/dump_batch_ref.py` and the Neural Engine blocks (`py/mmmllm/ane.py`,
for `--prefill ane`); each prints the command it is missing.
