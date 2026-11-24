# After the first publish (mmmllm 0.1.0)

So people can try it with the tools they already have. Every item ends in a **Check**.

- [ ] `POST /v1/chat/completions` (messages through the model's chat template, `encode_chat`), streaming and not.
  **Check:** openai-python `client.chat.completions.create(...)` with `base_url="http://127.0.0.1:8080/v1"`, both
  `stream=True` and `False`; a chat UI (e.g. Open WebUI) talks to it unmodified.
- [ ] `mmmllm chat MODEL`: interactive multi-turn chat in the terminal, engine in-process (like `mlx_lm.chat`,
  `ollama run`). History re-templated each turn; the prefix cache makes each turn prefill only the new message.
  **Check:** a second turn's prefill tokens = the new message's, not the whole history; Ctrl-C stops a reply, not
  the session.
- [ ] Non-streaming `/v1/completions` (today 400 "only stream: true is supported"). **Check:** openai-python
  `client.completions.create(...)` without `stream`.
- [ ] `stop` sequences (today ignored). **Check:** generation ends before the stop string, which isn't returned.
- [ ] Usage text says `mmmllm`, not `mllm`, when run through the package.
- [ ] More GPUs: port MLX's dispatch rules for M1–M3 and the M4 Max (`GpuOps: dispatch rules not ported`).
  **Check:** `mmmllm compare` on each, golden test bit-exact.
