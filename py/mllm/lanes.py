"""PrefillLane and DecodeLane: submit() / poll(). Python runs the work inside submit; C++ returns once it
is committed (prefill on its own thread + command queue, decode one step ahead). The engine loop is the same.
Lanes never touch the allocator: every unit a work item needs is allocated before submit."""
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class PrefillWork:
    req_id: int
    tokens: list          # ponytail: exact length on the GPU path; the ANE path pads to a compiled size
    start_pos: int
    slots: list           # KV pages covering start_pos + n_valid
    state_slots: list

    @property
    def n_valid(self):
        return len(self.tokens)


@dataclass
class DecodeRow:
    req_id: int
    token: int
    pos: int
    slots: list
    state_slots: list
    sampler: object


@dataclass
class DecodeWork:
    step: int
    rows: list


def _layers(dense, ops, mem, ws, rows, state_slot_lists):
    """dense(i) then attn(i) for every layer; state units copied in and out around it."""
    loaded = [mem.load_state(s) for s in state_slot_lists]
    states = [views for _, views in loaded]
    for i in range(ops.spec.n_layers):
        dense(i, ws)
        ops.attn(i, ws, rows, states)
    for (buf, _), s in zip(loaded, state_slot_lists):
        mem.store_state(s, buf)


class PrefillLane:
    def __init__(self, ops, mem, dense=None):
        """dense: GpuOps.dense (default) or AneOps.dense; attn always runs on the GPU."""
        self.ops, self.mem, self.done = ops, mem, []
        self.dense = dense or ops.dense

    def busy(self):
        return False

    def submit(self, w: PrefillWork):
        ws = self.mem.workspace()
        ws.tokens = mx.array([w.tokens])
        # K/V and state only: the last layer's o_proj + FFN and lm_head never run.
        _layers(self.dense, self.ops, self.mem, ws, [(w.start_pos, w.slots)], [w.state_slots])
        self.done.append(w)

    def poll(self):
        done, self.done = self.done, []
        return done


class DecodeLane:
    def __init__(self, ops, mem):
        self.ops, self.mem, self.done = ops, mem, []

    def can_submit(self):
        return True

    def submit(self, w: DecodeWork):
        ws = self.mem.workspace()
        ws.tokens = mx.array([[r.token] for r in w.rows])
        _layers(self.ops.dense, self.ops, self.mem, ws, [(r.pos, r.slots) for r in w.rows], [r.state_slots for r in w.rows])
        self.ops.dense(self.ops.spec.n_layers, ws)
        self.ops.lm_head(ws)
        toks = self.ops.sample(ws, [r.sampler for r in w.rows])
        self.done.extend((r.req_id, t) for r, t in zip(w.rows, toks))

    def poll(self):
        done, self.done = self.done, []
        return done
