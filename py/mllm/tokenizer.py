"""Tokenizer + streaming Detokenizer. Python wraps mlx-lm's; C++ will need its own (see architecture.md)."""
from mlx_lm.utils import load_tokenizer


class Tokenizer:
    def __init__(self, path):
        self._t = load_tokenizer(path)
        self.eos = frozenset(self._t.eos_token_ids)

    def encode(self, text):
        return self._t.encode(text)

    def encode_chat(self, messages):
        return self._t.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)

    def detokenizer(self):
        return Detokenizer(self._t.detokenizer)


class Detokenizer:
    """Holds back bytes until they form valid text."""

    def __init__(self, d):
        self._d = d
        d.reset()

    def push(self, token):
        self._d.add_token(token)
        return self._d.last_segment

    def flush(self):
        self._d.finalize()
        return self._d.last_segment
