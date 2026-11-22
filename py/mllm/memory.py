"""Memory: the only component that allocates. Weights, Arena (storage + bookkeeping), workspaces.

Arena layout follows docs/memory-design.md: N equal units, a unit holds one KV page (T tokens) for
every full-attention layer. A linear-attention state slot is a whole number of units.
"""
from types import SimpleNamespace

import numpy as np
from mlx_lm.utils import load_model

PAGE = 16384  # OS page; units are a multiple of it
ELEM = 2      # bytes per activation element (bf16 for Qwen3.5, kept as raw uint16 bits)


class Arena:
    """Bookkeeping per memory-design.md. One engine thread calls this; no locks.

    ponytail: no refcounts (no prefix cache yet) and no `pending` list (Python lanes finish inside
    submit, so nothing in flight can still read a released unit). Both come with the C++ allocator.
    """

    def __init__(self, unit_bytes, n_units):
        self.unit_bytes = unit_bytes
        self.bytes = np.zeros((n_units, unit_bytes), np.uint8)
        self.free = list(range(n_units - 1, -1, -1))  # ponytail: list as stack; bitmap in C++
        self.reserved = 0

    @property
    def n_units(self):
        return len(self.bytes)

    def admit(self, n):
        """Reserve n units. False = wait (the engine rejects n > n_units on arrival)."""
        if n > len(self.free) - self.reserved:
            return False
        self.reserved += n
        return True

    def alloc(self):
        """Take one unit out of the caller's reservation. Cannot fail after admit."""
        assert self.reserved > 0 and self.free, "alloc without reservation"
        self.reserved -= 1
        return self.free.pop()

    def unreserve(self, n):
        assert 0 <= n <= self.reserved
        self.reserved -= n

    def release(self, slots):
        self.free.extend(slots)


class Memory:
    def __init__(self, spec, arena_bytes, page_tokens=32):
        self.spec = spec
        self.T = page_tokens
        G = len(spec.full_layers)
        kv_bytes = G * 2 * spec.n_kv_heads * page_tokens * spec.head_dim * ELEM
        unit = -(-kv_bytes // PAGE) * PAGE
        self.arena = Arena(unit, arena_bytes // unit)
        # KV view: kv[slot, g, 0|1, head, token, dim], g = index among full layers. Same bytes as offset_in_unit.
        self.kv = self.arena.bytes[:, :kv_bytes].view(np.uint16).reshape(
            -1, G, 2, spec.n_kv_heads, page_tokens, spec.head_dim)
        assert np.shares_memory(self.kv, self.arena.bytes)
        self.g_of = {l: g for g, l in enumerate(spec.full_layers)}

        # State slot: per linear layer, conv state [(K-1), conv_dim] act dtype, then delta state [Hv, Dv, Dk] fp32.
        self.conv_bytes = (spec.conv_kernel - 1) * spec.conv_dim * ELEM
        self.delta_bytes = spec.lin_v_heads * spec.lin_v_dim * spec.lin_k_dim * 4
        assert self.conv_bytes % 4 == 0
        per_layer = self.conv_bytes + self.delta_bytes
        self.state_units = -(-per_layer * len(spec.linear_layers) // unit)
        self.state_off = {l: j * per_layer for j, l in enumerate(spec.linear_layers)}

        model, _ = load_model(spec.path)
        self.weights = model  # ponytail: mlx-lm modules; C++ mmaps the safetensors

    def units_for(self, n_tokens):
        return -(-n_tokens // self.T)

    def zero(self, slots):
        self.arena.bytes[slots] = 0

    # --- KV pages (full-attention layers) ---
    def write_kv(self, layer, slots, pos, k, v):
        """k, v: uint16 [n, heads, dim] for tokens pos..pos+n-1."""
        p = np.arange(pos, pos + len(k))
        s = np.asarray(slots)[p // self.T]
        g = self.g_of[layer]
        self.kv[s, g, 0, :, p % self.T, :] = k
        self.kv[s, g, 1, :, p % self.T, :] = v

    def read_kv(self, layer, slots, n_tokens):
        """-> (k, v) uint16 [heads, n_tokens, dim]. ponytail: copies every page, every layer, every step."""
        pages = self.kv[np.asarray(slots[: self.units_for(n_tokens)]), self.g_of[layer]]  # [P, 2, H, T, D]
        P, _, H, T, D = pages.shape
        kv = pages.transpose(1, 2, 0, 3, 4).reshape(2, H, P * T, D)[:, :, :n_tokens]
        return kv[0], kv[1]

    # --- state slots (linear-attention layers) ---
    def load_state(self, slots):
        """Gather a request's state units into one buffer -> (buf, {layer: (conv u16, delta f32)}).
        ponytail: copy in/out per lane call; C++ kernels address the units directly."""
        buf = self.arena.bytes[slots].reshape(-1)
        s = self.spec
        views = {}
        for l, off in self.state_off.items():
            conv = buf[off: off + self.conv_bytes].view(np.uint16).reshape(s.conv_kernel - 1, s.conv_dim)
            d0 = off + self.conv_bytes
            delta = buf[d0: d0 + self.delta_bytes].view(np.float32).reshape(s.lin_v_heads, s.lin_v_dim, s.lin_k_dim)
            views[l] = (conv, delta)
        return buf, views

    def store_state(self, slots, buf):
        self.arena.bytes[slots] = buf.reshape(len(slots), -1)

    def workspace(self):
        """Named activation fields. Python ops reassign them; C++ writes into fixed buffers."""
        return SimpleNamespace()
