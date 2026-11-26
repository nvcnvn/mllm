# After the first publish (mmmllm 0.1.0)

So people can try it with the tools they already have. Every item ends in a **Check**.

- [x] `POST /v1/chat/completions` (messages through the model's chat template, `encode_chat`), streaming and not.
  **Check:** openai-python `client.chat.completions.create(...)` with `base_url="http://127.0.0.1:8080/v1"`, both
  `stream=True` and `False`; a chat UI (e.g. Open WebUI) talks to it unmodified.
  *Done:* the reply's reasoning (the template opens `<think>`) comes as `reasoning_content`, the rest as `content`,
  the DeepSeek / vLLM / llama.cpp convention chat UIs read; text parts and `max_completion_tokens` accepted; no
  `max_tokens` = up to 4096 within max_ctx. tests/test_server covers both forms (same tokens as `generate`).
  openai-python passes (chat and completions, streamed and not, `stop`); `usage` adds `ttft_ms` and
  `tokens_per_second`. Open WebUI not tried.
- [x] `mmmllm chat MODEL`: interactive multi-turn chat in the terminal, engine in-process (like `mlx_lm.chat`,
  `ollama run`). History kept as tokens (`Tokenizer::continue_chat`, mlx_lm.chat's way: the reasoning stays in
  context) and a finished request's sequence (prompt + reply) goes into the prefix cache, ending mid-page in a tail
  node with its state, so each turn prefills only the new message.
  **Check:** a second turn's prefill tokens = the new message's, not the whole history; Ctrl-C stops a reply, not
  the session.
  *Done (g17).* 0.8B: turn 2 prompt 254, cached 236 (= turn 1 + its reply up to the EOS); a reply cut at max_tokens
  leaves its last token uncached (1231 / 1211). 27B, 300-token replies, turn 3: prompt 662, 639 cached, first token
  0.49 s; before (re-templated history): prompt 658, 320 cached, 3.50 s (one run each, under the heavy lock).
  A stats line per turn goes to stderr; Ctrl-C drops the turn.
- [x] Non-streaming `/v1/completions` (today 400 "only stream: true is supported"). **Check:** openai-python
  `client.completions.create(...)` without `stream`.
- [x] `stop` sequences (a string or up to 4; the request is cancelled at the match). **Check:** generation ends before the stop string, which isn't returned.
- [x] Usage text says `mmmllm`, not `mllm`, when run through the package.
- [ ] More GPUs: port MLX's dispatch rules for M1–M3 and the M4 Max (`GpuOps: dispatch rules not ported`).
  **Check:** `mmmllm compare` on each, golden test bit-exact. Open: needs those chips (or testers with them).
- [ ] `/v1/completions` `text` drops the reply's leading space (the Detokenizer strips it on purpose, as mlx-lm does), so
  a client that continues with prompt + text retokenizes differently and misses the prefix cache's tail. Fixing it needs
  a per-request keep-leading-space flag from the server through `Engine::submit`.
