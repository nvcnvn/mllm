"""Engine: requests and the loop. submit() + step() is the API; generate() is sugar over them.

    from mllm.engine import Engine
    e = Engine("mlx-community/Qwen3.5-0.8B-4bit")
    for text in e.generate([{"role": "user", "content": "hi"}]):
        print(text, end="", flush=True)
"""
from dataclasses import dataclass, field

from mlx_lm.sample_utils import make_sampler

from . import spec as spec_mod
from .gpu import GpuOps
from .lanes import DecodeLane, DecodeRow, DecodeWork, PrefillLane, PrefillWork
from .memory import Memory
from .tokenizer import Tokenizer

WAITING, PREFILL, DECODE, DONE = "WAITING", "PREFILL", "DECODE", "DONE"


@dataclass
class Request:
    id: int
    ids: list
    max_tokens: int
    sampler: object
    detok: object
    state: str = WAITING
    pos: int = 0                    # tokens whose K/V (and state) are in the arena
    pages: list = field(default_factory=list)
    state_slots: list = field(default_factory=list)
    reserved: int = 0               # units promised to this request, not yet allocated
    out: list = field(default_factory=list)
    in_flight: bool = False


class Engine:
    def __init__(self, model, arena_bytes=1 << 30, chunk=512, page_tokens=32, prefill="gpu", ane_weights="fp16"):
        """prefill: "gpu" or "ane" (dense blocks on the Neural Engine, compiled for S = chunk).
        ane_weights: a key of ane._compress (fp16, int8, lut4, ...)."""
        assert chunk % page_tokens == 0, "chunks start on a page boundary"
        self.spec = spec_mod.load(model)
        self.tok = Tokenizer(self.spec.path)
        self.mem = Memory(self.spec, arena_bytes, page_tokens)
        ops = GpuOps(self.spec, self.mem)
        dense = None
        if prefill == "ane":
            from .ane import AneOps
            self.ane = AneOps(ops, chunk, ane_weights)
            dense = self.ane.dense
        self.prefill, self.decode = PrefillLane(ops, self.mem, dense), DecodeLane(ops, self.mem)
        self.chunk = chunk
        self.reqs = {}                  # live requests by id, in arrival order
        self.next_id = 0
        self.step_no = 0

    # --- API ---
    def submit(self, prompt, max_tokens=256, temperature=0.0, top_p=0.0):
        """prompt: str, chat messages (list of dicts), or token ids. Returns a request id."""
        if isinstance(prompt, str):
            ids = self.tok.encode(prompt)
        elif prompt and isinstance(prompt[0], dict):
            ids = self.tok.encode_chat(prompt)
        else:
            ids = list(prompt)
        if not ids or max_tokens < 1:
            raise ValueError("empty prompt or max_tokens < 1")
        r = Request(self.next_id, ids, max_tokens, make_sampler(temp=temperature, top_p=top_p), self.tok.detokenizer())
        if self._demand(r) > self.mem.arena.n_units:
            raise ValueError(f"request needs {self._demand(r)} units, arena has {self.mem.arena.n_units}")
        self.reqs[r.id] = r
        self.next_id += 1
        return r.id

    def step(self):
        """One turn of the loop -> [(req_id, token, text, done)]."""
        events = []
        for w in self.prefill.poll():
            r = self.reqs[w.req_id]
            r.in_flight = False
            r.pos += w.n_valid
            if r.pos == len(r.ids) - 1:
                r.state = DECODE
        for req_id, t in self.decode.poll():
            r = self.reqs[req_id]
            r.in_flight = False
            r.pos += 1
            r.out.append(t)
            done = t in self.tok.eos or len(r.out) >= r.max_tokens
            text = "" if t in self.tok.eos else r.detok.push(t)
            if done:
                text += r.detok.flush()
                self._finish(r)
            events.append((req_id, t, text, done))

        for r in self.reqs.values():
            if r.state == WAITING and self.mem.arena.admit(self._demand(r)):
                r.reserved = self._demand(r)
                r.state_slots = [self._alloc(r) for _ in range(self.mem.state_units)]
                self.mem.zero(r.state_slots)  # reused units hold an old request's state
                r.state = PREFILL if len(r.ids) > 1 else DECODE

        if not self.prefill.busy():
            r = next((r for r in self.reqs.values() if r.state == PREFILL and not r.in_flight), None)
            if r:
                n = min(self.chunk, len(r.ids) - 1 - r.pos)
                self._ensure_pages(r, r.pos + n)
                r.in_flight = True
                self.prefill.submit(PrefillWork(r.id, r.ids[r.pos: r.pos + n], r.pos, list(r.pages), r.state_slots))

        batch = [r for r in self.reqs.values() if r.state == DECODE and not r.in_flight]
        if batch and self.decode.can_submit():
            rows = []
            for r in batch:
                self._ensure_pages(r, r.pos + 1)
                r.in_flight = True
                rows.append(DecodeRow(r.id, r.out[-1] if r.out else r.ids[-1], r.pos, list(r.pages), r.state_slots, r.sampler))
            self.decode.submit(DecodeWork(self.step_no, rows))
            self.step_no += 1

        self.reqs = {k: r for k, r in self.reqs.items() if r.state != DONE}
        return events

    def generate(self, prompt, **kw):
        rid = self.submit(prompt, **kw)
        while True:
            for req_id, _, text, done in self.step():
                if req_id == rid:
                    if text:
                        yield text
                    if done:
                        return

    # --- internals ---
    def _demand(self, r):
        """Units for the whole request up front. ponytail: no generation top-up yet (memory-design `admit`)."""
        return self.mem.units_for(len(r.ids) + r.max_tokens) + self.mem.state_units

    def _alloc(self, r):
        r.reserved -= 1
        return self.mem.arena.alloc()

    def _ensure_pages(self, r, n_tokens):
        while len(r.pages) * self.mem.T < n_tokens:
            r.pages.append(self._alloc(r))

    def _finish(self, r):
        a = self.mem.arena
        a.unreserve(r.reserved)
        a.release(r.pages + r.state_slots)
        r.pages, r.state_slots, r.reserved, r.state = [], [], 0, DONE
