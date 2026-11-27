#!/usr/bin/env python3
"""Draws the talk's diagrams: ram.svg and split.svg (the machine, then its regions), weights.svg (one copy, two
readers), bitmap.svg, block-life.svg (unit 10 through the story), spill.svg (the SSD tier, designed), arena-0.svg ..
arena-7.svg (one request's life through the counters), and the FAQ's hardware.svg, model.svg, layout.svg, pages.svg.
Run from anywhere: python3 docs/talk/draw.py

The arena frames and block-life.svg replay src/memory.mm's Arena and src/engine.mm's admit / snapshot / finish on a
15-unit arena, so every number drawn is what the real code would hold. Only the state slot is shrunk: 2 units here,
74 on the 27B.
"""
from collections import deque
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
T, STATE_UNITS, N_UNITS = 32, 2, 15

# ---------------------------------------------------------------------------------------------------------------
# Arena, as in src/memory.mm


class Arena:
    def __init__(s, n):
        s.n = n
        s.free = [True] * n  # the bitmap
        s.req_refs, s.cache_refs = [0] * n, [0] * n
        s.pending = deque()  # (slot, step), steps non-decreasing
        s.free_units, s.cache_only_units, s.reserved, s.committed = n, 0, 0, 0
        s.log = []

    def fits(s, n):
        return n <= s.free_units - s.reserved

    def can_ever(s, n):
        return n <= s.free_units + s.cache_only_units - s.reserved

    def admit(s, n, evict):
        assert s.can_ever(n)
        while not s.fits(n):
            assert evict()
        s.reserved += n
        s.log.append(f"admit({n})")

    def alloc(s):
        assert s.reserved > 0
        slot = s.free.index(True)
        s.free[slot] = False
        s.req_refs[slot] = 1
        s.free_units -= 1
        s.reserved -= 1
        s.log.append(f"alloc→{slot}")
        return slot

    def unreserve(s, n):
        assert n <= s.reserved
        s.reserved -= n
        s.log.append(f"unreserve({n})")

    def release(s, slots):
        for u in slots:
            assert s.req_refs[u] > 0
            s.req_refs[u] -= 1
            if s.req_refs[u]:
                continue
            if s.cache_refs[u]:
                s.cache_only_units += 1
            else:
                s.drop(u)
        s.log.append(f"release({','.join(map(str, slots))})")

    def ref(s, u):
        assert s.req_refs[u] or s.cache_refs[u]
        if not s.req_refs[u]:
            s.cache_only_units -= 1
        s.req_refs[u] += 1
        s.log.append(f"ref({u})")

    def cache_ref(s, u):
        assert s.req_refs[u]
        s.cache_refs[u] += 1
        s.log.append(f"cache_ref({u})")

    def cache_release(s, u):
        assert s.cache_refs[u] > 0
        s.cache_refs[u] -= 1
        s.log.append(f"cache_release({u})")
        if s.cache_refs[u] or s.req_refs[u]:
            return
        s.cache_only_units -= 1
        s.drop(u)

    def drop(s, u):
        s.pending.append((u, s.committed))
        s.free_units += 1

    def retire(s, signaled):
        while s.pending and s.pending[0][1] <= signaled:
            u, _ = s.pending.popleft()
            s.free[u] = True
        s.log.append(f"retire({signaled})")


# ---------------------------------------------------------------------------------------------------------------
# The story: a cache entry left by an earlier request with the same system prompt, request B decoding, then request
# A arrives (system prompt + question), prefills, decodes, finishes; request C arrives and reuses A's units.

class Req:
    def __init__(s, name, n_ids, max_tokens):
        s.name, s.n_ids, s.max_tokens = name, n_ids, max_tokens
        s.pages, s.state, s.reserved, s.pos = [], [], 0, 0


class Story:
    def __init__(s):
        s.a = Arena(N_UNITS)
        s.what = [""] * N_UNITS  # what the bytes are: "p0", "state", "★@64"
        s.owner = [""] * N_UNITS  # who wrote them, for free / pending units
        s.holders = [[] for _ in range(N_UNITS)]  # requests holding a req_ref, for the labels
        s.nodes = []  # cache trie as a chain: [{"unit", "snap", "children"}]
        s.signaled = 0
        s.tables = []
        s.frames = []

    def demand(s, r):
        return -(-(r.n_ids + r.max_tokens) // T) + STATE_UNITS

    def alloc(s, r, what):
        s.a.retire(s.signaled)
        s.a.log.pop()  # alloc's own retire: not worth a line
        assert r.reserved > 0
        r.reserved -= 1
        u = s.a.alloc()
        s.what[u], s.owner[u] = what, r.name
        s.holders[u].append(r.name)
        return u

    def ref(s, r, u):
        s.a.ref(u)
        s.holders[u].append(r.name)

    def release(s, r, slots):
        s.a.release(slots)
        for u in slots:
            s.holders[u].remove(r.name)

    def evict(s):
        leaf = next(n for n in reversed(s.nodes) if not n["children"])
        s.a.log.append("evict LRU leaf:")
        s.a.cache_release(leaf["unit"])
        for u in leaf["snap"]:
            s.a.cache_release(u)
        s.nodes.remove(leaf)
        if s.nodes:
            s.nodes[-1]["children"] -= 1
        return True

    def frame(s, slug, title, caption):
        a = s.a
        s.frames.append(dict(
            slug=slug, title=title, caption=caption, log=a.log, signaled=s.signaled,
            free=list(a.free), req=list(a.req_refs), cache=list(a.cache_refs),
            pending={u: st for u, st in a.pending}, what=list(s.what), owner=list(s.owner),
            holders=[list(h) for h in s.holders],
            counters=dict(free_units=a.free_units, cache_only_units=a.cache_only_units, reserved=a.reserved,
                          committed=a.committed, signaled=s.signaled),
            tables=[dict(t, rows=[dict(r) for r in t["rows"]]) for t in s.tables]))
        a.log = []


def row(r, n):
    return dict(name=r.name, pos=r.pos, n=n, pages=list(r.pages), state=list(r.state))


def story():
    s = Story()
    a = s.a
    # an earlier request X: 64-token system prompt + 10 tokens, snapshot at its last full page (64), finished
    x = Req("X", 74, 10)
    a.admit(s.demand(x) + STATE_UNITS, lambda: False)
    x.reserved = s.demand(x) + STATE_UNITS
    x.state = [s.alloc(x, "state") for _ in range(STATE_UNITS)]
    x.pages = [s.alloc(x, f"p{i}") for i in range(2)]
    snap = [s.alloc(x, "★@64") for _ in range(STATE_UNITS)]
    for u in x.pages:
        a.cache_ref(u)
    s.nodes += [dict(unit=x.pages[0], snap=[], children=1), dict(unit=x.pages[1], snap=snap, children=0)]
    for u in snap:
        a.cache_ref(u)
    s.release(x, snap)
    x.pages.append(s.alloc(x, "p2"))
    a.committed = 30
    a.unreserve(x.reserved)
    s.release(x, x.pages + x.state)
    s.signaled = 30
    a.retire(30)
    for u in range(N_UNITS):
        if a.free[u]:
            s.what[u] = s.owner[u] = ""
    # request B: a different 30-token prompt, decoding; 2 pages so far
    b = Req("B", 30, 40)
    a.admit(s.demand(b), lambda: False)
    b.reserved = s.demand(b)
    b.state = [s.alloc(b, "state") for _ in range(STATE_UNITS)]
    b.pages = [s.alloc(b, f"p{i}") for i in range(2)]
    b.pos = 62
    a.committed, s.signaled = 40, 39
    s.tables = [dict(title="decode copies[0]", step=40, status="GPU reading", rows=[row(b, 1)]),
                dict(title="decode copies[1]", step=39, status="done", rows=[dict(row(b, 1), pos=61)])]
    a.log = []
    s.frame("0", "The arena, its counters, and the table the GPU reads",
            "B is decoding. The cache still holds a 64-token system prompt (2 pages + a state snapshot ★) "
            "from an earlier request, which has finished.")

    # admit A: system prompt (64) + a 50-token question, up to 20 new tokens
    r = Req("A", 114, 10)
    L = S = 2
    m_pages, m_snap = [n["unit"] for n in s.nodes], s.nodes[-1]["snap"]
    last = r.n_ids - 1
    snaps = [last // T * T]  # the prompt's last full page: 96 (no branch point: L == S)
    for u in m_pages + m_snap:
        s.ref(r, u)
    n = s.demand(r) - S + len(snaps) * STATE_UNITS
    a.admit(n, s.evict)
    r.reserved, r.pages, r.pos = n, list(m_pages), S * T
    r.state = [s.alloc(r, "state") for _ in range(STATE_UNITS)]
    a.log.append(f"memcpy {','.join(map(str, m_snap))}→{','.join(map(str, r.state))}")
    s.release(r, m_snap)
    s.frame("1", "1 · admit A: a cache hit on the system prompt",
            f"match: L = {L} pages, snapshot at S = {S}. Hold them (ref) before admit can evict, reserve {n}, "
            "allocate A's state, copy the snapshot into it, let the snapshot go back to the cache.")

    # prefill chunk 64–95, then the snapshot at 96 (the chunk is done, the next not started)
    r.pages.append(s.alloc(r, "p2"))
    s.tables.insert(0, dict(title="prefill lane", step=None, status="GPU reading", rows=[row(r, 32)]))
    s.frame("2a", "2 · prefill A, chunk 64–95",
            "The prefill lane gets A's slots in its own table: pages 0–1 are shared (read only), page 2 is A's "
            "own. K/V for tokens 64–95 is written into it; A's state is overwritten in place.")
    r.pos = 96
    s.tables[0]["status"] = "done"
    snap = [s.alloc(r, "★@96") for _ in range(STATE_UNITS)]
    a.log.append(f"memcpy {','.join(map(str, r.state))}→{','.join(map(str, snap))}")
    a.log.append("insert:")
    a.cache_ref(r.pages[2])
    s.nodes[-1]["children"] += 1
    s.nodes.append(dict(unit=r.pages[2], snap=snap, children=0))
    for u in snap:
        a.cache_ref(u)
    s.release(r, snap)
    s.frame("2b", "2 · prefill A: snapshot at token 96",
            "The chunk ending at 96 is done and the next one is held back, so A's state is exactly at 96. "
            "Copy it, insert page 2 + the copy into the cache. The next turn of A's conversation resumes here.")

    # chunk 96–112 (the last prompt token is fed by decode), then decode step 41 with A and B
    r.pages.append(s.alloc(r, "p3"))
    r.pos = 113
    b.pos = 63
    s.tables = [dict(title="decode copies[1]", step=41, status="CPU wrote, committed", rows=[row(r, 1), row(b, 1)]),
                dict(title="decode copies[0]", step=40, status="GPU reading", rows=[dict(row(b, 1), pos=62)])]
    a.log.insert(0, "prefill 96–112:")
    a.log.append("build step 41 → commit")
    a.committed = 41
    s.frame("3", "3 · decode: the CPU builds step 41 while the GPU runs step 40",
            "A joins decode. Same table format as prefill, one row per request, the other copy by step parity. "
            "dense() reads only weights + workspace; attn() reads the table, then the units.")

    # step 42: needs copies[0], so step 40 must be done (signaled + 2 >= 42); B crosses into a new page
    s.signaled = 40
    b.pos, r.pos = 64, 114
    b.pages.append(s.alloc(b, "p2"))
    s.tables = [dict(title="decode copies[0]", step=42, status="CPU wrote, committed", rows=[row(r, 1), row(b, 1)]),
                dict(title="decode copies[1]", step=41, status="GPU reading",
                     rows=[dict(row(r, 1), pos=113), dict(row(b, 1), pos=63, pages=b.pages[:2])])]
    a.log.insert(0, "event = 40 → copies[0] is free")
    a.log.append("build step 42 → commit")
    a.committed = 42
    s.frame("4", "4 · decode step 42: B's token 64 starts a new page",
            "copies[0] is rewritten only after the event says step 40 is done. B's new page comes out of B's "
            "reservation, so this alloc cannot fail.")

    # v0.3.0: step 41 gave A its stop token; step 42 (in flight) feeds it, so A ends at step 42's poll. Then its whole
    # sequence (prompt + the stop token, 115 tokens) goes into the cache: pages 0–2 exist, page 3 becomes a tail node
    # (19 tokens) and A's state units become its snapshot as they are (Engine::cache_sequence), then finish releases.
    a.committed = 43
    b.pos = 65
    s.signaled = 42
    s.tables = [dict(title="decode copies[1]", step=43, status="GPU reading", rows=[row(b, 1)]),
                dict(title="decode copies[0]", step=42, status="done", rows=s.tables[0]["rows"])]
    a.log.append("event = 41: A's stop token; step 42 feeds it")
    a.log.append("commit step 43 (B only)")
    a.log.append("event = 42 → cache A's sequence:")
    a.cache_ref(r.pages[3])
    for u in r.state:
        a.cache_ref(u)
        s.what[u] = "★@115"
    s.nodes[-1]["children"] += 1
    s.nodes.append(dict(unit=r.pages[3], snap=list(r.state), children=0))
    a.log.append("finish:")
    a.unreserve(r.reserved)
    r.reserved = 0
    s.release(r, r.pages + r.state)
    s.frame("5", "5 · A finishes: its whole conversation goes into the cache",
            "A ends when step 42, which fed its stop token, is done: no GPU work reads A now. Page 13 becomes a tail "
            "node (19 tokens), and A's state units become its snapshot as they are, no copy. Everything is cache-only.")

    # C arrives (20-token prompt, up to 100 new tokens): 6 units; none free, so evict the two newest leaves
    c = Req("C", 20, 100)
    n = s.demand(c)
    a.log.append(f"fits({n})? no; can_ever({n})? yes →")
    a.admit(n, s.evict)
    c.reserved = n
    s.frame("6", "6 · C arrives: admitted at once, but nothing is free yet",
            f"C needs {n} units; none are free, but 10 are cache-only. Evict the least recently used leaf (A's tail "
            "+ its snapshot), then the next (page 2 + ★@96): unowned, but step 43 may read them, so pending.")

    # step 43 done: the evicted units retire; C's state, then its first page, take the lowest free bits
    a.committed = 44
    b.pos = 66
    s.signaled = 43
    a.log.append("alloc waits for the event")
    a.log.append("event = 43 →")
    a.retire(43)
    c.state = [s.alloc(c, "state") for _ in range(STATE_UNITS)]
    c.pages = [s.alloc(c, "p0")]
    s.tables = [dict(title="prefill lane", step=None, status="GPU reading", rows=[row(c, 19)]),
                dict(title="decode copies[0]", step=44, status="GPU reading", rows=[row(b, 1)]),
                dict(title="decode copies[1]", step=43, status="done", rows=[dict(row(b, 1), pos=65)])]
    s.frame("7", "7 · unit 10 comes back: C's first page",
            "The event reaches 43, so the evicted units retire. alloc takes the lowest free bits: C's state lands "
            "where A's was (8–9), and C's first page in unit 10, A's page 2. Nothing was erased; C overwrites it.")
    return s.frames


# ---------------------------------------------------------------------------------------------------------------
# SVG

# Visual encoding, the same in every diagram. One channel per question; a hue never means two things.
#
#   who     hue          CPU green (CPU_C), GPU blue (GPU_C), ANE purple. Processor cards, cells a processor
#                        touches, "GPU reading" / "CPU wrote" tags, the GPU / CPU bands of the arena frames.
#   when    position     prefill lane above decode lane (model.svg rows, hardware.svg sub-rows).
#   which   tag / column dense = neutral grey (it runs on the GPU or the ANE), attn = amber.
#   op
#   whether outline      solid = happens / touches it; dashed grey = never, empty, free, done.
#   how     word         read, write, read + write. Never a colour.
#
# Reserved, outside the processor hues:
#   amber   per-request memory: the arena, attn columns, cache-only units (attn is amber because it touches it).
#   red     pending (hatched): released, but a GPU step may still read it.
#   yellow  "changed since the previous frame" (HL), metadata cells and counters only.
#   grey    shared or neutral: workspace, step tables, live units, dense.
#   magenta / teal / brown: requests A / B / C (REQ_C), model.svg only; none of them is a processor or state hue.
# Layout: a grid. Columns are the thing being touched (layers, memory regions, arena units), rows are who or when.
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', Menlo, Consolas, monospace"
INK, MUTED, FAINT, LINE = "#202124", "#5F6368", "#9AA0A6", "#DADCE0"
HL, HL_LINE = "#FFF0A8", "#E3B500"
STATE = {  # fill, stroke
    "free": ("#FFFFFF", "#BDC1C6"),
    "live": ("#E8EAED", "#3C4043"),
    "cache-only": ("#FEEFC3", "#E37400"),
    "pending": ("#FCE8E6", "#D93025"),
}
GPU_C, CPU_C = "#1A73E8", "#188038"


class Svg:
    def __init__(s, w, h):
        s.w, s.h, s.out = w, h, []

    def rect(s, x, y, w, h, fill="none", stroke="none", sw=1, rx=0, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="{sw}"{d}{extra}/>')

    def text(s, x, y, t, size=14, fill=INK, anchor="start", weight=400, font=FONT, italic=False):
        st = ' font-style="italic"' if italic else ""
        s.out.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" fill="{fill}" '
                     f'text-anchor="{anchor}" font-weight="{weight}" xml:space="preserve"{st}>{escape(str(t))}</text>')

    def line(s, x1, y1, x2, y2, stroke=LINE, sw=1, dash=None, marker=False):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#arrow)"' if marker else ""
        s.out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')

    def path(s, d, stroke=INK, sw=1.2, fill="none", marker=True):
        m = ' marker-end="url(#arrow)"' if marker else ""
        s.out.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{m}/>')

    def save(s, path):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s.w} {s.h}" width="{s.w}" height="{s.h}">'
                '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{MUTED}"/></marker>'
                '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="6" height="6" fill="{STATE["pending"][0]}"/>'
                '<line x1="0" y1="0" x2="0" y2="6" stroke="#F4B4AE" stroke-width="2.5"/></pattern></defs>'
                f'<rect width="{s.w}" height="{s.h}" fill="#FFFFFF"/>')
        Path(path).write_text(head + "\n".join(s.out) + "</svg>\n")


def state_of(f, u):
    if f["free"][u]:
        return "free"
    if u in f["pending"]:
        return "pending"
    return "live" if f["req"][u] else "cache-only"


def wrap(items, width):
    lines, cur = [], ""
    for it in items:
        if cur and len(cur) + 3 + len(it) > width:
            lines.append(cur)
            cur = it
        else:
            cur = (f"{cur} {it}" if cur.endswith((":", "→")) else f"{cur} · {it}") if cur else it
    return lines + ([cur] if cur else [])


def wrap_words(text, width):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return lines + [cur]


W, H = 1280, 720
X0, UW = 140, 72  # first unit's x, unit width
LX = 24  # band labels


def draw_frame(f, prev):
    g = Svg(W, H)
    # header
    g.text(LX, 40, f["title"], 26, weight=700)
    for i, ln in enumerate(wrap_words(f["caption"], 112)):
        g.text(LX, 68 + i * 20, ln, 15, MUTED)

    # --- GPU band ---------------------------------------------------------------------------------------------
    y0 = 124
    g.rect(12, y0, W - 24, 190, fill="#F8FAFF", stroke="#D2E3FC", rx=10)
    g.text(LX, y0 + 24, "GPU reads", 13, GPU_C, weight=700)
    # dense side
    g.rect(LX, y0 + 40, 96, 40, fill="#F1F3F4", stroke=LINE, rx=6)
    g.text(LX + 48, y0 + 65, "weights", 13, MUTED, "middle")
    g.rect(LX, y0 + 88, 96, 40, fill="#F1F3F4", stroke=LINE, rx=6)
    g.text(LX + 48, y0 + 113, "workspace", 13, MUTED, "middle")
    g.text(LX, y0 + 150, "dense() reads", 12, MUTED)
    g.text(LX, y0 + 165, "only these", 12, MUTED)
    # step tables (attn side)
    tx = 140
    cw = (W - 24 - tx - 10 * (len(f["tables"]) - 1)) / max(len(f["tables"]), 1)
    status_c = {"GPU reading": GPU_C, "CPU wrote, committed": CPU_C, "done": FAINT}
    reads = {}  # unit -> labels of in-flight work reading it
    for i, t in enumerate(f["tables"]):
        x = tx + i * (cw + 10)
        col = status_c[t["status"]]
        g.rect(x, y0 + 12, cw, 120, fill="#FFFFFF", stroke=col, sw=1.5 if t["status"] != "done" else 1, rx=8,
               dash="5 4" if t["status"] == "done" else None)
        label = t["title"] + (f"  ·  step {t['step']}" if t["step"] is not None else "")
        g.text(x + 12, y0 + 34, label, 14, INK, weight=700)
        g.text(x + cw - 12, y0 + 34, t["status"], 13, col, "end", weight=700)
        g.text(x + 12, y0 + 56, "row   pos    n   pages              state", 12, FAINT, font=MONO)
        for j, r in enumerate(t["rows"]):
            yy = y0 + 80 + j * 24
            g.text(x + 12, yy, f"{r['name']:<5} {r['pos']:<6} {r['n']:<3} "
                   f"{' '.join(map(str, r['pages'])):<18} {' '.join(map(str, r['state']))}", 13, INK, font=MONO)
            if t["status"] != "done":
                tag = "pf" if t["step"] is None else str(t["step"])
                for u in r["pages"] + r["state"]:
                    reads.setdefault(u, [])
                    if tag not in reads[u]:
                        reads[u].append(tag)
    g.text(tx, y0 + 156, "attn() reads its row in the table, then the units it names "
           "(slot × unit bytes + offset). Units in flight:", 12, MUTED)
    # in-flight strip
    ys = y0 + 166
    for u in range(N_UNITS):
        x = X0 + u * UW
        if u in reads:
            tags = sorted(reads[u], key=lambda t: (t == "pf", t))
            g.rect(x + 3, ys, UW - 6, 16, fill="#D2E3FC", rx=3)
            g.text(x + UW / 2, ys + 12, " ".join(tags), 11, GPU_C, "middle", weight=700, font=MONO)

    # --- arena ------------------------------------------------------------------------------------------------
    ya = 352
    g.text(LX, ya + 26, "arena", 15, INK, weight=700)
    g.text(LX, ya + 44, "1 buffer, N units", 12, MUTED)
    for u in range(N_UNITS):
        x = X0 + u * UW
        st = state_of(f, u)
        fill, stroke = STATE[st]
        g.text(x + UW / 2, ya - 6, u, 12, FAINT, "middle", font=MONO)
        g.rect(x + 1, ya, UW - 2, 62, fill="url(#hatch)" if st == "pending" else fill, stroke=stroke,
               sw=1.4, rx=4, dash="4 3" if st == "free" else None)
        if st == "free":
            continue
        holders = f["holders"][u] + (["cache"] if f["cache"][u] else [])
        top = "+".join(holders) if holders else f"was {f['owner'][u]}"
        if top == "was X":
            top = "earlier"
        dim = st == "pending"
        g.text(x + UW / 2, ya + 26, top, 11 if len(top) > 5 else 13, MUTED if dim else INK, "middle",
               weight=400 if dim else 700)
        g.text(x + UW / 2, ya + 46, f["what"][u], 12, MUTED if dim else INK, "middle", font=MONO)

    # --- CPU metadata -----------------------------------------------------------------------------------------
    ym = 436
    g.rect(12, ym, W - 24, 178, fill="#F6FBF7", stroke="#CEEAD6", rx=10)
    g.text(LX, ym + 22, "only the CPU", 13, CPU_C, weight=700)
    g.text(LX, ym + 37, "sees these", 13, CPU_C, weight=700)
    fields = [("free bit", lambda fr, u: "1" if fr["free"][u] else "0"),
              ("req_refs", lambda fr, u: fr["req"][u] or ""),
              ("cache_refs", lambda fr, u: fr["cache"][u] or ""),
              ("pending", lambda fr, u: fr["pending"].get(u, ""))]
    for k, (name, get) in enumerate(fields):
        yy = ym + 50 + k * 28
        g.text(X0 - 10, yy + 18, name, 13, MUTED, "end", font=MONO)
        for u in range(N_UNITS):
            x = X0 + u * UW
            v = get(f, u)
            changed = prev is not None and str(get(prev, u)) != str(v)
            g.rect(x + 1, yy, UW - 2, 26, fill=HL if changed else "#FFFFFF",
                   stroke=HL_LINE if changed else LINE, sw=1.2 if changed else 1)
            if name == "free bit" and v == "0" and not changed:
                g.text(x + UW / 2, yy + 18, v, 13, FAINT, "middle", font=MONO)
            else:
                g.text(x + UW / 2, yy + 18, v, 13, INK, "middle", weight=700 if changed else 400, font=MONO)

    # counters
    yc = 640
    cx = X0
    for name, v in f["counters"].items():
        changed = prev is not None and prev["counters"][name] != v
        w = 22 + 8.2 * (len(name) + len(str(v)) + 3)
        g.rect(cx, yc - 20, w, 30, fill=HL if changed else "#FFFFFF", stroke=HL_LINE if changed else LINE, rx=6)
        g.text(cx + 11, yc, f"{name} = {v}", 14, INK, weight=700 if changed else 400, font=MONO)
        cx += w + 10
    c = f["counters"]
    ok = c["reserved"] <= c["free_units"]
    g.text(cx + 6, yc, f"reserved ≤ free_units {'✓' if ok else '✗'}", 13, CPU_C if ok else "#D93025")
    g.text(LX, yc, "counters", 13, MUTED, weight=700)

    # call log
    if f["log"]:
        g.text(LX, yc + 38, "calls", 13, MUTED, weight=700)
        for i, ln in enumerate(wrap(f["log"], 128)[:2]):
            g.text(X0, yc + 38 + i * 20, ln, 13, INK, font=MONO)
        assert len(wrap(f["log"], 128)) <= 2, f["slug"]

    # legend
    lx = W - 24
    for st in ["pending", "cache-only", "live", "free"]:
        fill, stroke = STATE[st]
        tw = 8 * len(st) + 10
        lx -= tw
        g.text(lx, 40, st, 13, MUTED)
        lx -= 22
        g.rect(lx, 28, 16, 16, fill="url(#hatch)" if st == "pending" else fill, stroke=stroke, rx=3,
               dash="3 2" if st == "free" else None)
        lx -= 12
    hl_x = W - 24 - 132
    g.rect(hl_x, 52, 16, 16, fill=HL, stroke=HL_LINE, rx=3)
    g.text(hl_x + 22, 65, "changed here", 13, MUTED)
    return g


def arena_frames():
    frames = story()
    prev = None
    for f in frames:
        draw_frame(f, prev).save(OUT / f"arena-{f['slug']}.svg")
        prev = f


# ---------------------------------------------------------------------------------------------------------------
# ram.svg and split.svg: the same picture twice. Three processors over one block of DRAM; then the block cut into
# the regions mmmllm allocates at startup, each region zoomed into a column that says who touches it.
# 27B numbers: weights 15.1 GB (target-27b.md), arena --arena-gib 12, workspace ~1 GB plus ~3 GB with ANE prefill
# (peak 16.7 vs 19.8 GB, target-27b.md §5).

ANE_C = "#8430CE"
TINT = {CPU_C: "#F3FAF5", GPU_C: "#F4F8FE", ANE_C: "#FAF4FE"}
PROCS = [  # name, colour, cores, two lines of measured facts, job
    ("CPU", CPU_C, "14 cores", ["10 performance (2 clusters of 5)", "+ 4 efficiency; no math worth it"],
     "tokenizer, counters, step tables"),
    ("GPU", GPU_C, "20 cores", ["bf16 matmul ~7.5 TFLOPS", "reads 210–238 GB/s"],
     "decode, attention, prefill"),
    ("ANE", ANE_C, "16 cores", ["~13 TFLOPS on small split matmuls", "streams only ~39 GB/s"],
     "prefill's dense layers"),
]
REGIONS = [  # name, GB drawn in the bar, size label, fill, stroke, contents, {proc: (access, why)}
    ("macOS + apps", 14, "12–16 GB", "#F1F3F4", "#BDC1C6", None, None),
    ("weights", 15.1, "15.1 GB", "#E8F0FE", GPU_C, ["4-bit MLX files, mmap'ed,", "the only copy"],
     {"CPU": ("—", "maps them at load"), "GPU": ("read", "every decode step"),
      "ANE": ("—", "gets fp16 copies of one layer")}),
    ("workspace", 4, "1 + 3 GB", "#F1F3F4", "#80868B", ["activations; with ANE prefill", "2 sets of weight surfaces"],
     {"CPU": ("write", "token ids in"), "GPU": ("read + write", "activations, dequant"),
      "ANE": ("read + write", "IOSurfaces in and out")}),
    ("arena", 12, "12 GiB", "#FEF7E0", "#E37400", ["every request's K/V pages", "and state, in 2 MiB units"],
     {"CPU": ("read + write", "snapshot copies"), "GPU": ("read + write", "attention"),
      "ANE": ("—", "the GPU moves its K/V")}),
    ("step tables", 0.35, "KBs", "#F1F3F4", "#80868B", ["slot numbers per row,", "2 copies by step parity"],
     {"CPU": ("write", "every step"), "GPU": ("read", "attention finds units"), "ANE": ("—", "")}),
    ("counters + trie", 0.35, "MBs", "#E6F4EA", CPU_C, ["free bits, refcounts,", "the prefix cache (CPU heap)"],
     {"CPU": ("read + write", "one thread, no locks"), "GPU": ("—", "sees slot numbers only"), "ANE": ("—", "")}),
    ("free", 18.2, "~18 GB", "#FFFFFF", "#DADCE0", None, None),
]


def ram_bar(g, x, y, w, h, segs, total=64.0, min_w=0):
    """Draw a proportional RAM bar; return each segment's (x, w). segs: (name, gb, size, fill, stroke)."""
    widths = [max(gb / total * w, min_w) for _, gb, *_ in segs]
    k = (w - sum(ww for ww in widths if ww == min_w)) / sum(ww for ww in widths if ww != min_w)
    widths = [ww if ww == min_w else ww * k for ww in widths]
    out, xx = [], x
    for (name, gb, size, fill, stroke), ww in zip(segs, widths):
        g.rect(f"{xx:.1f}", y, f"{ww - 3:.1f}", h, fill=fill, stroke=stroke, rx=4, sw=1.3,
               dash="4 3" if name == "free" else None)
        if ww > 96:
            g.text(f"{xx + ww / 2 - 1.5:.1f}", y + h / 2 - 2, name, 14, INK, "middle", weight=700)
            g.text(f"{xx + ww / 2 - 1.5:.1f}", y + h / 2 + 16, size, 12, MUTED, "middle")
        out.append((xx, ww - 3))
        xx += ww
    return out


def machine(split):
    W_, H_ = 1280, 720
    g = Svg(W_, H_)
    if split:
        g.text(24, 40, "How we split it: five regions, all made at startup", 26, weight=700)
        g.text(24, 66, "Qwen3.6-27B on the 64 GB Mac mini. Nothing is allocated after startup; "
               "the arena's size (--arena-gib) is the one memory knob.", 15, MUTED)
    else:
        g.text(24, 40, "One block of DRAM, three processors", 26, weight=700)
        g.text(24, 66, "M4 Pro Mac mini. No processor has memory of its own: all three read and write "
               "the same physical pages.", 15, MUTED)
    # processor cards
    cy, ch, cw = 92, 108, 392
    xs = [24 + i * (cw + 28) for i in range(3)]
    for x, (name, col, cores, facts, job) in zip(xs, PROCS):
        g.rect(x, cy, cw, ch, fill=TINT[col], stroke=col, rx=10, sw=1.6)
        g.text(x + 18, cy + 34, name, 24, col, weight=700)
        g.text(x + 82, cy + 33, cores, 16, INK, weight=700)
        for k, ln in enumerate(facts):
            g.text(x + 18, cy + 60 + k * 18, ln, 13, MUTED)
        g.text(x + cw - 16, cy + 33, job, 12, col, "end", weight=700)
    # one bus
    by = 262
    for x, (name, col, *_rest) in zip(xs, PROCS):
        g.line(x + cw / 2, cy + ch, x + cw / 2, by - 4, col, 2.2)
    g.text(240, by - 24, "273 GB/s peak, one budget for all three", 13, MUTED)
    if not split:
        g.rect(24, by, 1232, 90, fill="#FFFDF5", stroke="#5F6368", rx=6, sw=1.6)
        g.text(640, by + 40, "64 GB LPDDR5X", 24, INK, "middle", weight=700)
        g.text(640, by + 66, "one physical pool, mapped in 16 KiB pages", 14, MUTED, "middle")
        lines = [
            ("Same bytes, no copies.", "A Metal buffer is the same physical pages in the CPU's and the GPU's "
             "page tables. The ANE reads and writes IOSurfaces, which the GPU maps as buffers too."),
            ("Not the same rules.", "The GPU sees CPU writes made before a command buffer is committed; the CPU "
             "sees GPU writes after it completes. In between, nobody may overwrite what the other is reading."),
            ("One budget.", "Decode streams every weight once per token (14.4 GB on the 27B, ~64 ms). Whatever "
             "else runs at the same time takes bandwidth from it."),
        ]
        y = by + 136
        for head, body in lines:
            g.text(24, y, head, 15, INK, weight=700)
            for k, ln in enumerate(wrap_words(body, 128)):
                g.text(250, y + k * 20, ln, 14, INK)
            y += 20 * len(wrap_words(body, 128)) + 18
        g.save(OUT / "ram.svg")
        return
    segs = [r[:5] for r in REGIONS]
    pos = ram_bar(g, 24, by, 1232, 56, segs, min_w=22)
    # columns: the regions mmmllm owns, equal width, zoomed from their segment
    owned = [i for i, r in enumerate(REGIONS) if r[5]]
    top, colw, gap = by + 100, 236, 13
    for j, i in enumerate(owned):
        name, gb, size, fill, stroke, what, access = REGIONS[i]
        sx, sw = pos[i]
        x = 24 + j * (colw + gap)
        g.path(f"M {sx:.1f} {by + 58} L {x} {top - 2} L {x + colw} {top - 2} L {sx + sw:.1f} {by + 58} Z",
               stroke=stroke, sw=0.8, fill=fill, marker=False)
        g.rect(x, top, colw, 300, fill="#FFFFFF", stroke=stroke, rx=8, sw=1.4)
        g.rect(x, top, colw, 74, fill=fill, stroke=stroke, rx=8, sw=1.4)
        g.rect(x + 1, top + 60, colw - 2, 14, fill=fill)
        g.text(x + 14, top + 24, name, 16, INK, weight=700)
        g.text(x + colw - 14, top + 24, size, 13, MUTED, "end", weight=700)
        for k, ln in enumerate(what):
            g.text(x + 14, top + 44 + k * 16, ln, 12, MUTED)
        for k, (pname, col, *_r) in enumerate(PROCS):
            acc, why = access[pname]
            yy = top + 92 + k * 64
            never = acc == "—"
            g.rect(x + 10, yy, colw - 20, 54, fill="#FFFFFF" if never else TINT[col], stroke=LINE if never else col,
                   rx=6, sw=1 if never else 1.4, dash="4 3" if never else None)
            g.text(x + 22, yy + 21, pname, 13, FAINT if never else col, weight=700)
            g.text(x + 62, yy + 21, "never" if never else acc, 13, FAINT if never else INK, weight=700)
            g.text(x + 22, yy + 41, why, 12, FAINT if never else MUTED)
    g.text(24, H_ - 16, "Thin slivers are drawn wider than their size. Workspace: ~1 GB on the GPU path, ~3 GB more "
           "with ANE prefill (two sets of fp16 weight surfaces, ~0.9 GB each, plus Core ML). free = what macOS keeps.",
           12, FAINT)
    g.save(OUT / "split.svg")


# ---------------------------------------------------------------------------------------------------------------
# weights.svg: one copy of the weights, two readers. The GPU dequantizes layer i's 4-bit weights into fp16
# IOSurfaces (one set per chunk in flight) that the ANE's Core ML block takes as inputs (src/ane.h).


def weights():
    g = Svg(1280, 700)
    g.text(24, 40, "One copy of the weights, two readers", 26, weight=700)
    g.text(24, 66, "Qwen3.6-27B, 4-bit. The ANE used to need its own baked-in copy; now the GPU hands it one "
           "layer's weights at a time, decoded to fp16, through shared IOSurfaces.", 15, MUTED)
    before = [("macOS + apps", 14, "12–16 GB", "#F1F3F4", "#BDC1C6"), ("GPU weights", 15.1, "15.1 GB", "#E8F0FE", GPU_C),
              ("ANE weights", 15, "~15 GB (est.)", "#F3E8FD", ANE_C), ("arena", 12, "12 GiB", "#FEF7E0", "#E37400"),
              ("ws", 2, "", "#F1F3F4", "#80868B"), ("free", 5.9, "~6 GB", "#FFFFFF", "#DADCE0")]
    now = [("macOS + apps", 14, "12–16 GB", "#F1F3F4", "#BDC1C6"), ("weights", 15.1, "15.1 GB", "#E8F0FE", GPU_C),
           ("ws", 4, "", "#F1F3F4", "#80868B"), ("arena", 12, "12 GiB", "#FEF7E0", "#E37400"),
           ("free", 18.9, "~19 GB", "#FFFFFF", "#DADCE0")]
    for y, label, sub, segs in ((92, "before", "two copies", before), (166, "now", "one copy", now)):
        g.text(24, y + 26, label, 16, INK, weight=700)
        g.text(24, y + 44, sub, 12, MUTED)
        pos = ram_bar(g, 110, y, 1146, 56, segs)
        for (name, *_r), (x, w) in zip(segs, pos):
            if name == "ws":
                g.text(x + w / 2, y + 33, "ws", 12, MUTED, "middle", weight=700)
    g.text(1256, 244, "ws = workspace + step tables; now it holds the two weight-surface sets (~0.9 GB each)",
           12, FAINT, "end")

    # one layer of one chunk
    y = 288
    g.text(24, y, "One prefill layer, one chunk: who reads what", 17, weight=700)
    boxes = [  # title, mono sub, extra line, fill, stroke
        ("weights", "layer i, 4-bit", "the GPU's copy", "#E8F0FE", GPU_C),
        ("GPU", "dequant_q4", "~0.8 GB fp16 written", "#F4F8FE", GPU_C),
        ("weight surfaces", "set k, fp16", "IOSurface, ~0.9 GB", "#F1F3F4", "#80868B"),
        ("ANE", "dense(i)", "weights are inputs", "#FAF4FE", ANE_C),
        ("IOSurface", "q / k / v out", "no copy", "#F1F3F4", "#80868B"),
        ("GPU", "attn(i)", "", "#F4F8FE", GPU_C),
        ("arena units", "K/V, state", "", "#FEF7E0", "#E37400"),
    ]
    bw, gap, bx = 150, 31, 24
    for i, (a, b, c, fill, stroke) in enumerate(boxes):
        x = bx + i * (bw + gap)
        g.rect(x, y + 20, bw, 74, fill=fill, stroke=stroke, rx=8, sw=1.4)
        g.text(x + bw / 2, y + 44, a, 14, INK, "middle", weight=700)
        g.text(x + bw / 2, y + 64, b, 13, MUTED, "middle", font=MONO)
        if c:
            g.text(x + bw / 2, y + 82, c, 11.5, MUTED, "middle")
        if i < len(boxes) - 1:
            g.line(x + bw + 3, y + 57, x + bw + gap - 3, y + 57, MUTED, 1.4, marker=True)
    # two sets
    y2 = y + 122
    g.rect(24, y2, 1232, 74, fill="#FAFAFA", stroke=LINE, rx=8)
    g.text(40, y2 + 26, "× 2 sets: two chunks in flight", 14, INK, weight=700)
    g.text(40, y2 + 48, "The ANE runs chunk k's dense(i) from set k while the GPU fills set k+1 and runs the other "
           "chunk's attention.", 13, MUTED)
    g.text(40, y2 + 66, "Each set is rewritten only after the ANE call that read it has returned: the same "
           "\"don't overwrite what someone is reading\" rule as the arena.", 13, MUTED)

    # numbers
    yt = y2 + 106
    g.text(24, yt, "27B, measured (mllm bench, M4 Pro)", 15, weight=700)
    cols = [24, 470, 700, 930]
    for x, h in zip(cols, ["", "GPU prefill", "ANE, one copy", ""]):
        g.text(x, yt + 28, h, 13, MUTED, weight=700)
    rows = [("prompt of 4,096 tokens", "131 tok/s", "191 tok/s", "chunks overlap: ANE dense, GPU attn"),
            ("decode while a 2,048-token prefill runs", "3.3 tok/s", "12.1 tok/s", "decode alone: 15.0–15.3"),
            ("peak memory", "16.7 GB", "19.8 GB", "a second baked copy would add ~15 GB")]
    for k, (name, a, b, note) in enumerate(rows):
        yy = yt + 54 + k * 26
        g.line(24, yy - 18, 1256, yy - 18, LINE)
        g.text(24, yy, name, 14, INK)
        g.text(470, yy, a, 14, INK, font=MONO)
        g.text(700, yy, b, 14, INK, weight=700, font=MONO)
        g.text(930, yy, note, 13, MUTED)
    g.save(OUT / "weights.svg")


# ---------------------------------------------------------------------------------------------------------------
# bitmap.svg: the free bitmap at 27B defaults. 12 GiB / 2 MiB = 6,144 units = 96 words of 64 bits = 768 bytes.
# Bit s of the bitmap is unit s of the arena; unit s is bytes [s × 2 MiB, (s + 1) × 2 MiB) of the one buffer.

def bitmap():
    g = Svg(1280, 620)
    g.text(24, 40, "A very thin bitmap: bit s is unit s", 26, weight=700)
    g.text(24, 66, "27B defaults: 32 tokens per page → 2 MiB per unit. A 12 GiB arena is 6,144 units, so its free "
           "bits are 6,144 bits: 768 bytes.", 15, MUTED)
    # the whole bitmap: 96 words
    y0, x0 = 104, 150
    g.text(x0 - 12, y0 + 15, "whole bitmap", 13, MUTED, "end", weight=700)
    ww = (1256 - x0) / 96
    for w in range(96):
        g.rect(f"{x0 + w * ww:.1f}", y0, f"{ww - 1.5:.1f}", 20, fill=HL if w == 0 else "#F1F3F4",
               stroke=HL_LINE if w == 0 else LINE, rx=2)
    g.text(x0, y0 + 38, "word 0", 12, INK, weight=700)
    g.text(1256, y0 + 38, "96 words × 64 bits = 6,144 bits = 768 bytes (6 cache lines of 128 bytes)", 12, MUTED, "end")

    # word 0, bit by bit, over units 0–63
    pattern = ("LLLLLLLLCCLLLLLLFFLLLLPPLLLLLLLLLLLLLLLLLLCCCCLLLLFLLLFFFFFFFFFF")
    st_of = {"L": "live", "C": "cache-only", "P": "pending", "F": "free"}
    yb, ya, bw = 196, 300, (1256 - x0) / 64
    g.path(f"M {x0} {y0 + 22} L {x0} {yb - 4} L 1256 {yb - 4} L {x0 + ww:.1f} {y0 + 22} Z", stroke=HL_LINE,
           sw=0.8, fill="#FFFBE6", marker=False)
    g.text(x0 - 12, yb + 22, "word 0", 13, MUTED, "end", weight=700)
    g.text(x0 - 12, yb + 38, "free bits", 11, FAINT, "end")
    g.text(x0 - 12, ya + 26, "arena", 13, MUTED, "end", weight=700)
    g.text(x0 - 12, ya + 42, "units 0–63", 11, FAINT, "end")
    pick = 17
    for s, ch in enumerate(pattern):
        x = x0 + s * bw
        st = st_of[ch]
        bit = "1" if st == "free" else "0"
        on = s == pick
        g.rect(f"{x:.1f}", yb, f"{bw - 2:.1f}", 32, fill=HL if on else ("#FFFFFF" if bit == "0" else "#E6F4EA"),
               stroke=HL_LINE if on else (LINE if bit == "0" else CPU_C), sw=1.4 if on else 1)
        g.text(f"{x + bw / 2 - 1:.1f}", yb + 21, bit, 12, INK if bit == "1" else FAINT, "middle",
               weight=700 if bit == "1" else 400, font=MONO)
        fill, stroke = STATE[st]
        g.rect(f"{x:.1f}", ya, f"{bw - 2:.1f}", 48, fill="url(#hatch)" if st == "pending" else fill,
               stroke=HL_LINE if on else stroke, sw=2 if on else 1, rx=3, dash="3 2" if st == "free" else None)
        if s % 8 == 0 or on:
            g.text(f"{x + bw / 2 - 1:.1f}", ya + 66, s, 11, INK if on else FAINT, "middle", font=MONO,
                   weight=700 if on else 400)
        g.line(f"{x + bw / 2 - 1:.1f}", yb + 33, f"{x + bw / 2 - 1:.1f}", ya - 1,
               HL_LINE if on else "#EEEEEE", 1.6 if on else 0.8)
    xp = x0 + pick * bw + bw / 2
    g.text(xp + 16, ya - 18, "bit 17 = 1 → unit 17 is free", 13, INK, weight=700)
    # legend
    lx, ly = x0, ya + 96
    for st in ["live", "cache-only", "pending", "free"]:
        fill, stroke = STATE[st]
        g.rect(lx, ly - 12, 14, 14, fill="url(#hatch)" if st == "pending" else fill, stroke=stroke, rx=3,
               dash="3 2" if st == "free" else None)
        g.text(lx + 20, ly, st, 13, MUTED)
        lx += 40 + 8 * len(st)
    g.text(lx + 10, ly, "Only a free unit has bit 1. Live, cache-only and pending all read 0: someone holds them, "
           "or the GPU may still.", 13, MUTED)

    # address and the two operations
    yc = 456
    boxes = [
        ("address", ["unit s  =  bit s % 64 of word s / 64",
                     "bytes   =  s × 2 MiB … (s + 1) × 2 MiB",
                     "unit 17 =  34 MiB … 36 MiB of the arena buffer"]),
        ("alloc() and retire()", ["alloc:  first word ≠ 0, s = 64 × word + ctz(word)",
                                  "        clear the bit  → lowest free unit",
                                  "retire: set the bit again (only after the GPU is past it)"]),
    ]
    for k, (title, lines) in enumerate(boxes):
        x = 24 + k * 624
        g.rect(x, yc, 608, 118, fill="#F8F9FA", stroke=LINE, rx=8)
        g.text(x + 16, yc + 26, title, 14, INK, weight=700)
        for j, ln in enumerate(lines):
            g.text(x + 16, yc + 54 + j * 22, ln, 13.5, INK, font=MONO)
    g.text(24, 604, "Besides its bit, a unit has two uint16 counters (req_refs, cache_refs): 24 KiB for all 6,144. "
           "The GPU never sees any of it, only slot numbers.", 13, MUTED)
    g.save(OUT / "bitmap.svg")


# ---------------------------------------------------------------------------------------------------------------
# block-life.svg: unit 10 through the arena frames (story()): free, A's prefill page, shared with the cache, cache-
# only, pending, free, C's page. Every value is the replay's.

def block_life(frames):
    U = 10
    g = Svg(1280, 600)
    g.text(24, 40, "The life of one block: unit 10", 26, weight=700)
    g.text(24, 66, "The frames of one request (A) arriving and leaving, seen from a single unit. Same replay of "
           "Arena and Engine as the frames.", 15, MUTED)
    heads = {"0": "before A", "1": "A admitted", "2a": "A's prefill", "2b": "snapshot @96", "3": "decode",
             "4": "decode", "5": "A finishes", "6": "C: evict", "7": "C: reuse"}
    calls = {"2a": "alloc → 10", "2b": "insert: cache_ref", "5": "release", "6": "cache_release", "7": "retire(43), alloc"}
    x0, cw = 150, (1256 - 150) / len(frames)
    yh, yu, yg, ym = 96, 132, 256, 312
    names = ["free bit", "req_refs", "cache_refs", "pending"]
    for k, n in enumerate(names):
        g.text(x0 - 12, ym + 18 + k * 30, n, 13, MUTED, "end", font=MONO)
    g.text(x0 - 12, yu + 36, "unit 10", 14, INK, "end", weight=700)
    g.text(x0 - 12, yg + 6, "GPU", 13, GPU_C, "end", weight=700)
    g.text(x0 - 12, yg + 22, "touches it", 11, FAINT, "end")
    prev = None
    for i, f in enumerate(frames):
        x = x0 + i * cw
        g.text(x + cw / 2, yh, f["slug"], 12, FAINT, "middle", font=MONO)
        g.text(x + cw / 2, yh + 18, heads[f["slug"]], 13, INK, "middle", weight=700)
        st = state_of(f, U)
        fill, stroke = STATE[st]
        g.rect(f"{x + 8:.1f}", yu, f"{cw - 16:.1f}", 72, fill="url(#hatch)" if st == "pending" else fill,
               stroke=stroke, sw=1.6, rx=6, dash="4 3" if st == "free" else None)
        if st != "free":
            holders = f["holders"][U] + (["cache"] if f["cache"][U] else [])
            top = "+".join(holders) if holders else f"was {f['owner'][U]}"
            g.text(x + cw / 2, yu + 30, top, 13, MUTED if st == "pending" else INK, "middle", weight=700)
            g.text(x + cw / 2, yu + 50, f["what"][U], 13, MUTED if st == "pending" else INK, "middle", font=MONO)
        g.text(x + cw / 2, yu + 92, st, 12, stroke if st != "free" else FAINT, "middle", weight=700)
        # GPU: which in-flight work names unit 10
        tags = []
        for t in f["tables"]:
            if t["status"] == "done":
                continue
            for r in t["rows"]:
                if U in r["pages"]:
                    if t["step"] is None:
                        tags.append("prefill writes" if r["pages"][-1] == U else "prefill reads")
                    else:
                        tags.append(f"step {t['step']} reads")
        for k, tg in enumerate(tags[:2]):
            g.rect(f"{x + 10:.1f}", yg - 12 + k * 22, f"{cw - 20:.1f}", 18, fill="#D2E3FC", rx=3)
            g.text(x + cw / 2, yg + 1 + k * 22, tg, 11.5, GPU_C, "middle", weight=700)
        if not tags:
            g.text(x + cw / 2, yg + 1, "—", 12, FAINT, "middle")
        vals = ["1" if f["free"][U] else "0", f["req"][U] or "", f["cache"][U] or "", f["pending"].get(U, "")]
        pvals = None if prev is None else ["1" if prev["free"][U] else "0", prev["req"][U] or "",
                                           prev["cache"][U] or "", prev["pending"].get(U, "")]
        for k, v in enumerate(vals):
            ch = pvals is not None and str(pvals[k]) != str(v)
            g.rect(f"{x + 8:.1f}", ym + k * 30, f"{cw - 16:.1f}", 26, fill=HL if ch else "#FFFFFF",
                   stroke=HL_LINE if ch else LINE)
            g.text(x + cw / 2, ym + 18 + k * 30, v, 13, INK, "middle", weight=700 if ch else 400, font=MONO)
        if f["slug"] in calls:
            g.text(x + cw / 2, ym + 140, calls[f["slug"]], 11.5, INK, "middle", font=MONO)
        prev = f
    g.text(x0 - 12, ym + 140, "call", 13, MUTED, "end", weight=700)
    # what to notice
    notes = [
        ("2a → 2b", "The same unit becomes the cache's while A still holds it: two refcounts, no copy. "
         "A never writes it again: the page is full."),
        ("5", "A is gone, the cache keeps it: cache-only, valid K/V, and the next request with this prompt can ref it."),
        ("6", "Evicted while step 43 is in flight: pending records the last committed step, not the last reader."),
        ("7", "Back to free only when the event passes 43. alloc takes the lowest free bit, so it is reused at once."),
    ]
    y = ym + 176
    for k, (tag, body) in enumerate(notes):
        g.text(24, y + k * 22, tag, 13, INK, weight=700, font=MONO)
        g.text(110, y + k * 22, body, 13, MUTED)
    g.save(OUT / "block-life.svg")


# ---------------------------------------------------------------------------------------------------------------
# spill.svg: a unit's states, and the SSD tier designed in memory-design.md ("SSD tier"), not built yet.

def spill():
    g = Svg(1280, 720)
    g.text(24, 40, "Spill to disk: one more place a cache-only block can go", 26, weight=700)
    g.text(24, 66, "Designed (docs/memory-design.md, \"SSD tier\"), not built yet. Dashed = the new paths.", 15, MUTED)
    by, bh, bw = 210, 64, 190
    xs = {"free": 40, "live": 330, "cache-only": 640, "pending": 960}
    for st, x in xs.items():
        fill, stroke = STATE[st]
        g.rect(x, by, bw, bh, fill="url(#hatch)" if st == "pending" else fill, stroke=stroke, sw=1.8, rx=8,
               dash="4 3" if st == "free" else None)
        g.text(x + bw / 2, by + 39, st, 17, INK, "middle", weight=700)

    def arrow(d, label, lx, ly, col=MUTED, dash=None, anchor="middle"):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        g.out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.6"{extra} marker-end="url(#arrow)"/>')
        for k, ln in enumerate(label):
            g.text(lx, ly + k * 16, ln, 12.5, INK if k == 0 else MUTED, anchor, weight=700 if k == 0 else 400,
                   font=MONO if k == 0 else FONT)

    arrow(f"M {40 + bw} {by + 22} L 326 {by + 22}", ["alloc()"], 278, by + 12)
    arrow(f"M {330 + bw} {by + 22} L 636 {by + 22}", ["release()", "the cache holds it"], 578, by - 26)
    arrow(f"M 636 {by + 46} L {330 + bw + 4} {by + 46}", ["ref()", "cache hit"], 578, by + 92)
    arrow(f"M {640 + bw} {by + 32} L 956 {by + 32}", ["cache_release()", "evict"], 893, by - 26)
    arrow(f"M {330 + bw / 2} {by + bh} C {330 + bw / 2} {by + 150}, {960 + bw / 2} {by + 150}, {960 + bw / 2} {by + bh + 4}",
          ["release()", "nobody else holds it"], 1000, by + 140, anchor="start")
    arrow(f"M {960 + bw / 2} {by} C {960 + bw / 2} {by - 100}, {40 + bw / 2} {by - 100}, {40 + bw / 2} {by - 4}",
          ["retire(signaled)", "the GPU is past the step"], 640, by - 104)

    # the disk
    dx, dy, dw, dh = 560, 470, 520, 92
    g.rect(dx, dy, dw, dh, fill="#F8F9FA", stroke="#5F6368", sw=1.6, rx=10, dash="6 4")
    g.text(dx + 20, dy + 30, "SSD: one file per cache node", 16, INK, weight=700)
    g.text(dx + 20, dy + 52, "its pages + one state snapshot, the spill unit (pages without", 13, MUTED)
    g.text(dx + 20, dy + 70, "the snapshot below them are useless to a hybrid model)", 13, MUTED)
    arrow(f"M {640 + 60} {by + bh + 4} L {640 + 60} {dy - 4}", ["spill", "pwrite straight from the", "arena buffer, then evict"],
          716, by + bh + 130, "#5F6368", "6 4", "start")
    arrow(f"M {dx} {dy + 46} C 420 {dy + 46}, {330 + bw / 2} {dy}, {330 + bw / 2} {by + bh + 4}",
          ["load", "admit, alloc, hold a ref,", "pread straight into units"], 300, dy + 40, "#5F6368", "6 4", "end")

    # file layout
    fy = 604
    g.text(24, fy, "One file", 14, INK, weight=700)
    parts = [("header", 250, "#FFFFFF", "format, model + weights hash, T, unit bytes, prefill path, parent hash, token ids"),
             ("page 0", 110, STATE["cache-only"][0], ""), ("page 1", 110, STATE["cache-only"][0], ""),
             ("…", 60, "#FFFFFF", ""), ("snapshot: 74 units", 300, STATE["cache-only"][0], ""),
             ("checksum", 120, "#FFFFFF", "")]
    x = 24
    for name, w, fill, _ in parts:
        g.rect(x, fy + 14, w - 4, 40, fill=fill, stroke="#5F6368", rx=4)
        g.text(x + (w - 4) / 2, fy + 39, name, 13, INK, "middle", weight=700)
        x += w
    g.text(24, fy + 78, "header: format, model and weights hash, tokens per page, unit bytes, state units, prefill path "
           "(GPU or ANE: different K/V), parent hash, the node's token ids", 12.5, MUTED)
    g.text(24, fy + 96, "Pages sit at 2 MiB multiples, 4 KiB-aligned, so F_NOCACHE direct I/O needs no staging copy. "
           "No fsync: a bad checksum is just a miss.", 12.5, MUTED)
    g.save(OUT / "spill.svg")


# ---------------------------------------------------------------------------------------------------------------
# model.svg: the model as a grid. Columns are layers (one set of weights each, shared by every row); rows are the two
# phases; per-request memory sits between them, touched only from the attn columns.

REQ_C = {"A": "#C2185B", "B": "#00838F", "C": "#6D4C41"}  # hues no processor or state uses


def model():
    g = Svg(1280, 660)
    DENSE_F, DENSE_S = "#F1F3F4", "#80868B"  # neutral: dense runs on the GPU or the ANE
    ATTN_F, ATTN_S = "#FEF7E0", "#E37400"

    # the example: A's prompt, tokenized; the last token goes through decode, the rest is the prefill chunk
    toks = ["You", " are", " a", " helpful", " assistant", ".", " What", " is", " the", " capital", " of", " France", "?"]
    g.text(24, 30, "Example", 15, INK, weight=700)
    g.text(110, 30, "A's prompt", 14, REQ_C["A"], weight=700)
    g.text(200, 30, '"You are a helpful assistant. What is the capital of France?"  → tokenizer → 13 tokens, '
           'one row each (a vector of 1,024 numbers on the 0.8B):', 14, INK)
    x = 110
    for i, t in enumerate(toks):
        w = 12 + 8.4 * len(t)
        last = i == len(toks) - 1
        g.rect(x, 44, w, 24, fill="#FFFFFF" if last else "#FCE4EC", stroke=REQ_C["A"], rx=4, sw=1.2,
               dash="4 3" if last else None)
        g.text(x + w / 2, 61, t.replace(" ", "·"), 13, INK, "middle", font=MONO)
        x += w + 4
    g.text(x + 8, 61, "← C = 12 rows go through prefill; the last one,", 13, MUTED)
    g.text(x + 8, 79, "\"?\", goes through decode and yields \"·Paris\".  (· is a space)", 13, MUTED)
    g.out.append('<g transform="translate(0,110)">')
    # the inner loop: layers, left to right
    g.line(250, -6, 1036, -6, INK, 1.6, marker=True)
    g.rect(470, -18, 340, 24, fill="#FFFFFF")
    g.text(640, -1, "inner loop: one layer after another, left to right", 14, INK, "middle", weight=700)
    cols = [  # x, w, kind, title, subtitle
        (250, 100, "dense", "dense", "weights W0"),
        (370, 120, "attn", "attn, linear", "3 of 4 layers · state"),
        (510, 100, "dense", "dense", "weights W1"),
        (630, 200, "attn", "attn, full", "1 of 4 layers · K/V"),
        (850, 70, "more", "…", "24–64 layers"),
        (940, 100, "dense", "lm_head", "weights"),
    ]
    yp0, yp1 = 110, 228  # prefill lane
    ym = [268, 298, 328]  # memory rows A, B, C
    yd0 = 400  # decode lane: 3 rows
    rh = 24
    yd1 = yd0 + 3 * rh

    # column backgrounds: one set of weights, read by both lanes
    for x, w, kind, title, sub in cols:
        if kind == "more":
            g.text(x + w / 2, 60, title, 22, MUTED, "middle", weight=700)
            g.text(x + w / 2, 80, sub, 12, MUTED, "middle")
            continue
        fill, stroke = (DENSE_F, DENSE_S) if kind == "dense" else (ATTN_F, ATTN_S)
        g.rect(x - 8, 30, w + 16, yd1 + 18 - 30, fill=fill, stroke=stroke, rx=10, sw=0.8, extra=' opacity="0.55"')
        g.text(x + w / 2, 54, title, 15, INK, "middle", weight=700)
        g.text(x + w / 2, 72, sub, 12, MUTED, "middle")
    g.text(24, 54, "one column = one set of weights,", 13, MUTED)
    g.text(24, 72, "read once per call, for every row", 13, MUTED)

    # lane labels
    g.text(24, yp0 + 22, "PREFILL", 17, INK, weight=700)
    for i, ln in enumerate(["once per prompt", "C tokens of", "one request", "(a chunk)"]):
        g.text(24, yp0 + 44 + i * 18, ln, 13, MUTED)
    g.text(24, yd0 + 16, "DECODE", 17, INK, weight=700)
    for i, ln in enumerate(["every step", "1 token each,", "B requests"]):
        g.text(24, yd0 + 36 + i * 18, ln, 13, MUTED)
    g.text(24, ym[0] + 4, "per-request", 13, ATTN_S, weight=700)
    g.text(24, ym[0] + 22, "memory", 13, ATTN_S, weight=700)
    g.text(24, ym[0] + 40, "(the arena)", 13, MUTED)

    # prefill lane: a tall block of C rows (A's prompt) through every column
    def rows_block(x, w, y0, y1, n, color, fill="#FFFFFF"):
        g.rect(x, y0, w, y1 - y0, fill=fill, stroke=color, rx=4, sw=1.4)
        for i in range(1, n):
            yy = y0 + (y1 - y0) * i / n
            g.line(x + 1, yy, x + w - 1, yy, "#F8BBD0", 1)

    rows_block(140, 90, yp0, yp1, 7, REQ_C["A"], fill="#FCE4EC")
    for i, t in enumerate(["You", "·are", "·a", "⋮", "·capital", "·of", "·France"]):
        g.text(185, yp0 + (yp1 - yp0) * (i + 0.5) / 7 + 5, t, 12, INK, "middle", font=MONO)
    g.text(185, yp0 - 8, "A: C = 12 rows", 13, REQ_C["A"], "middle", weight=700)
    prev = 230
    for x, w, kind, title, sub in cols:
        if kind == "more":
            g.line(prev + 4, (yp0 + yp1) / 2, x + w - 4, (yp0 + yp1) / 2, MUTED, 1.4, dash="3 4")
            prev = x + w
            continue
        if title == "lm_head":
            g.rect(x, yp0, w, yp1 - yp0, fill="#FFFFFF", stroke=LINE, rx=4, dash="4 3")
            g.text(x + w / 2, (yp0 + yp1) / 2 - 4, "skipped", 13, FAINT, "middle")
            g.text(x + w / 2, (yp0 + yp1) / 2 + 14, "in prefill", 13, FAINT, "middle")
            g.line(prev + 4, (yp0 + yp1) / 2, x - 4, (yp0 + yp1) / 2, MUTED, 1.4, marker=True)
            break
        rows_block(x, w, yp0, yp1, 7, REQ_C["A"])
        g.line(prev + 4, (yp0 + yp1) / 2, x - 4, (yp0 + yp1) / 2, MUTED, 1.4, marker=True)
        prev = x + w
    g.text(1060, yp0 + 18, "no token comes out:", 14, INK, weight=700)
    for i, ln in enumerate(["prefill only fills A's memory.", "", "C rows share every weight", "read → compute-bound:",
                            "GPU, or the ANE for dense"]):
        g.text(1060, yp0 + 38 + i * 18, ln, 13, INK if i < 1 else MUTED)

    # memory band: state (fixed) under the linear column, KV pages (growing) under the full column
    lin, full = cols[1], cols[3]
    kv_pages = {"A": 12 / 32, "B": 2.6, "C": 1.3}  # pages filled: A holds its 12 prefilled tokens
    for (name, y) in zip("ABC", ym):
        g.text(232, y + 17, name, 14, REQ_C[name], "end", weight=700)
        # state: same size for everyone
        g.rect(lin[0] + 44, y, 70, rh, fill="#FFFFFF", stroke=REQ_C[name], rx=3, sw=1.3)
        g.text(lin[0] + 79, y + 17, "state", 12, INK, "middle", font=MONO)
        # KV: pages of 32 tokens, growing
        px, pw = full[0] + 6, 34
        n_full = int(kv_pages[name])
        for k in range(4):
            x = px + k * (pw + 3)
            frac = 1 if k < n_full else (kv_pages[name] - n_full if k == n_full else 0)
            g.rect(x, y, pw, rh, fill="#FFFFFF", stroke=REQ_C[name] if frac else LINE, rx=3,
                   sw=1.2 if frac else 1, dash=None if frac else "3 2")
            if frac:
                g.rect(x + 1, y + 1, (pw - 2) * frac, rh - 2, fill=REQ_C[name], extra=' opacity="0.22"')
    for x, w, kind, title, sub in cols:
        if kind == "dense":
            g.text(x + w / 2, ym[1] + 17, "no memory", 13, FAINT, "middle", italic=True)

    # prefill attn ↔ A's memory
    for c, lbl in ((lin, "overwrite"), (full, "write C tokens' K/V")):
        cx = c[0] + 79 if c is lin else c[0] + 60
        g.line(cx, yp1 + 3, cx, ym[0] - 3, REQ_C["A"], 1.6, marker=True)
        g.text(cx + 6, yp1 + 26, lbl, 12, REQ_C["A"])

    # decode lane: 3 rows, one per request
    def decode_block(x, w):
        for i, name in enumerate("ABC"):
            g.rect(x, yd0 + i * rh, w, rh, fill="#FFFFFF", stroke=REQ_C[name], rx=3, sw=1.3)

    decode_block(140, 90)
    for i, name in enumerate("ABC"):
        g.text(185, yd0 + i * rh + 17, 'A: "?"' if name == "A" else f"{name}: 1 token", 12,
               REQ_C[name], "middle", weight=700, font=MONO if name == "A" else FONT)
    prev = 230
    ymid = yd0 + 1.5 * rh
    for x, w, kind, title, sub in cols:
        if kind == "more":
            g.line(prev + 4, ymid, x + w - 4, ymid, MUTED, 1.4, dash="3 4")
            prev = x + w
            continue
        decode_block(x, w)
        g.line(prev + 4, ymid, x - 4, ymid, MUTED, 1.4, marker=True)
        prev = x + w
    for i, name in enumerate("ABC"):
        g.text(1060, yd0 + i * rh + 17, 'A: "·Paris"' if name == "A" else f"{name}: next token", 13,
               REQ_C[name], weight=700, font=MONO if name == "A" else FONT)
    g.line(prev + 4, ymid, 1054, ymid, MUTED, 1.4, marker=True)

    # decode attn ↔ each request's own memory: elbows beside the boxes, so no arrow crosses another row
    for k, name in enumerate("ABC"):
        yk = ym[k] + rh / 2
        x = lin[0] + 10 + k * 10
        g.path(f"M {x} {yd0 - 3} L {x} {yk} L {lin[0] + 41} {yk}", REQ_C[name], 1.3)
        x = full[0] + full[1] - 6 - k * 10
        g.path(f"M {x} {yd0 - 3} L {x} {yk} L {full[0] + 6 + 4 * 37 + 1} {yk}", REQ_C[name], 1.3)
    g.text(lin[0] + 44, ym[-1] + rh + 22, "read +", 12, INK)
    g.text(lin[0] + 44, ym[-1] + rh + 38, "overwrite", 12, INK)
    g.text(full[0] + full[1] - 40, ym[-1] + rh + 30, "read all, append 1", 12, INK, "end")

    # the hand-off: A's last prompt token goes through decode
    g.path(f"M 140 {yp1 - 20} C 96 {yp1 + 40}, 96 {yd0 - 40}, 137 {yd0 + 10}", REQ_C["A"], 1.4)
    for i, ln in enumerate(["last prompt", "token joins", "decode"]):
        g.text(104, (yp1 + yd0) / 2 + 48 + i * 16, ln, 12, REQ_C["A"], "end")
    # the loop
    g.path(f"M 1100 {yd1 + 4} L 1100 {yd1 + 34} L 185 {yd1 + 34} L 185 {yd1 + 4}", MUTED, 1.4)
    g.rect(390, yd1 + 20, 500, 22, fill="#FFFFFF")
    g.text(640, yd1 + 37, "outer loop: one decode step per new token, fed back as the next input", 14, INK,
           "middle", weight=700)
    # rows: all at once, inside every layer
    d0 = cols[0]
    g.rect(d0[0] + 8, (yp0 + yp1) / 2 - 22, d0[1] - 16, 40, fill="#FFFFFF", rx=4)
    g.text(d0[0] + d0[1] / 2, (yp0 + yp1) / 2 - 5, "all 12 rows", 12, REQ_C["A"], "middle", weight=700)
    g.text(d0[0] + d0[1] / 2, (yp0 + yp1) / 2 + 11, "at once", 12, REQ_C["A"], "middle", weight=700)
    g.text(d0[0] + d0[1] / 2, yd0 - 22, "all B rows", 12, INK, "middle", weight=700)
    g.text(d0[0] + d0[1] / 2, yd0 - 7, "at once", 12, INK, "middle", weight=700)
    for i, ln in enumerate(["B rows share every weight", "read, but each row reads", "its own memory →",
                            "bandwidth-bound: GPU"]):
        g.text(1060, ym[0] + 4 + i * 18, ln, 13, MUTED)
    g.out.append("</g>")
    g.save(OUT / "model.svg")


# ---------------------------------------------------------------------------------------------------------------
# hardware.svg: who touches which memory, in the model diagram's grid style. Columns are regions of the one RAM,
# rows are the three processors; a cell says read / write / never, and when.

def hardware():
    """Who touches which memory, and when. Hue = processor, row = phase (prefill above decode, as in model.svg),
    tag = the op doing it (grey dense, amber attn), solid = touches it, dashed = never, word = how."""
    regions = [  # name, contents, size, fill, stroke
        ("weights", "mmap'ed MLX files, one copy", "15.1 GB", "#E8F0FE", GPU_C),
        ("workspace", "activations, IOSurfaces", "1 + 3 GB", "#F1F3F4", "#9AA0A6"),
        ("arena", "every request's K/V + state", "12 GiB", "#FEF7E0", "#E37400"),
        ("step tables", "slot lists, 2 copies", "KBs", "#F1F3F4", "#9AA0A6"),
        ("counters, cache trie", "plain CPU heap", "MBs", "#E6F4EA", CPU_C),
    ]
    R, W, RW, NO = "read", "write", "read + write", "never"
    D, A, DA = "dense", "attn", "dense + attn"
    procs = [  # name, spec lines, color, {phase: cells (access, op tag, lines)}
        ("CPU", ["14 cores", "bookkeeping,", "tokenizer"], CPU_C, {
            "prefill": [(NO, None, ["only maps it, at load"]),
                        (W, None, ["the chunk's token ids"]),
                        (RW, None, ["snapshot copies,", "zeroes a reused state"]),
                        (W, None, ["the prefill lane's copy"]),
                        (RW, None, ["alloc pages, insert", "snapshots in the cache"])],
            "decode": [(NO, None, []),
                       (RW, None, ["token ids in,", "sampled tokens out"]),
                       (NO, None, ["only the GPU touches", "it while decoding"]),
                       (W, None, ["the next step's copy,", "every step"]),
                       (RW, None, ["a page every 32 tokens,", "release, retire"])]}),
        ("GPU", ["20 cores", "7.5 TFLOPS", "210–238 GB/s"], GPU_C, {
            "prefill": [(R, D, ["GPU path: all of it;", "ANE path: dequantizes it"]),
                        (RW, DA, ["activations of C rows;", "fp16 weight surfaces"]),
                        (RW, A, ["writes C tokens' K/V", "+ state, reads earlier"]),
                        (R, A, ["the lane's copy"]), (NO, None, ["sees slot numbers only"])],
            "decode": [(R, D, ["all of it, every step"]),
                       (RW, DA, ["activations of B rows"]),
                       (RW, A, ["appends 1 K/V per row,", "reads all its pages"]),
                       (R, A, ["its step's copy"]), (NO, None, [])]}),
        ("ANE", ["16 cores", "~13 TFLOPS", "~39 GB/s"], ANE_C, {
            "prefill": [(NO, None, ["gets fp16 copies of", "one layer instead"]),
                        (RW, D, ["weight surfaces in,", "activations in and out"]),
                        (NO, None, ["the GPU moves its K/V"]), (NO, None, []), (NO, None, [])],
            "decode": [(NO, None, ["~39 GB/s: a decode step", "would take 370 ms"]),
                       (NO, None, []), (NO, None, []), (NO, None, []), (NO, None, [])]}),
    ]
    tint = {CPU_C: "#F3FAF5", GPU_C: "#F4F8FE", ANE_C: "#FAF4FE"}
    tag_c = {D: ("#F1F3F4", "#80868B"), A: ("#FEEFC3", "#E37400"), DA: ("#F1F3F4", "#80868B")}

    x0, gap = 236, 4
    cw = (1256 - x0 - gap * (len(regions) - 1)) / len(regions)
    yh, ch, cg, pg = 40, 62, 4, 12  # header top, cell height, gap between phases, gap between processors
    ytop = yh + 64
    H = ytop + 3 * (2 * ch + cg) + 2 * pg + 48
    g = Svg(1280, H)
    g.text(24, 26, "one LPDDR5X pool, 64 GB, 273 GB/s: every solid cell draws on the same bandwidth", 15, INK,
           weight=700)
    for j, (name, what, size, fill, stroke) in enumerate(regions):
        x = x0 + j * (cw + gap)
        g.rect(x, yh, cw, H - 42 - yh, fill=fill, stroke=stroke, rx=10, sw=0.8, extra=' opacity="0.55"')
        g.text(x + cw / 2, yh + 22, name, 15, INK, "middle", weight=700)
        g.text(x + cw / 2, yh + 38, what, 12, MUTED, "middle")
        g.text(x + cw / 2, yh + 54, size, 12, MUTED, "middle")
    y = ytop
    for name, spec, col, phases in procs:
        g.rect(24, y, 142, 2 * ch + cg, fill="#FFFFFF", stroke=col, rx=8, sw=1.6)
        g.text(38, y + 28, name, 20, col, weight=700)
        for k, ln in enumerate(spec):
            g.text(38, y + 52 + k * 17, ln, 12, MUTED)
        for p, phase in enumerate(("prefill", "decode")):
            yy = y + p * (ch + cg)
            g.text(228, yy + ch / 2 + 5, phase, 13, col, "end", weight=700)
            for j, (acc, op, lines) in enumerate(phases[phase]):
                x = x0 + j * (cw + gap) + 4
                w = cw - 8
                never = acc == NO
                g.rect(x, yy, w, ch, fill="#FFFFFF" if never else tint[col], stroke=LINE if never else col,
                       rx=6, sw=1 if never else 1.5, dash="4 3" if never else None)
                g.text(x + 10, yy + 19, acc, 13, FAINT if never else col, weight=700)
                if op == DA:  # both ops: two tags, top right
                    tx = x + w - 6
                    for o in (A, D):
                        f, st = tag_c[o]
                        tw = 8 + 6.6 * len(o)
                        tx -= tw
                        g.rect(tx, yy + 6, tw, 17, fill=f, stroke=st, rx=8, sw=1)
                        g.text(tx + tw / 2, yy + 18, o, 11, INK, "middle", font=MONO)
                        tx -= 6
                elif op:
                    f, st = tag_c[op]
                    tw = 8 + 6.6 * len(op)
                    g.rect(x + w - tw - 6, yy + 6, tw, 17, fill=f, stroke=st, rx=8, sw=1)
                    g.text(x + w - 6 - tw / 2, yy + 18, op, 11, INK, "middle", font=MONO)
                for k, ln in enumerate(lines):
                    g.text(x + 10, yy + 37 + k * 15, ln, 11.5, FAINT if never else INK)
        y += 2 * ch + cg + pg

    # legend
    yl = H - 12
    lx = 24
    g.text(lx, yl, "colour = who", 13, INK, weight=700)
    lx += 100
    for name, col in (("CPU", CPU_C), ("GPU", GPU_C), ("ANE", ANE_C)):
        g.rect(lx, yl - 12, 14, 14, fill=tint[col], stroke=col, rx=3, sw=1.5)
        g.text(lx + 20, yl, name, 13, col, weight=700)
        lx += 66
    lx += 16
    g.text(lx, yl, "row = when", 13, INK, weight=700)
    lx += 100
    g.text(lx, yl, "tag = which op", 13, INK, weight=700)
    lx += 112
    for op in (D, A):
        f, st = tag_c[op]
        tw = 8 + 6.6 * len(op)
        g.rect(lx, yl - 13, tw, 17, fill=f, stroke=st, rx=8)
        g.text(lx + tw / 2, yl, op, 11, INK, "middle", font=MONO)
        lx += tw + 8
    lx += 16
    g.text(lx, yl, "solid = touches it", 13, INK, weight=700)
    lx += 134
    g.rect(lx, yl - 12, 14, 14, fill="#FFFFFF", stroke=LINE, rx=3, dash="3 2")
    g.text(lx + 20, yl, "never", 13, MUTED)
    lx += 70
    g.text(lx, yl, "word = how", 13, INK, weight=700)
    g.save(OUT / "hardware.svg")

# ---------------------------------------------------------------------------------------------------------------
# layout.svg: the arena at real scale (27B: 12 GiB = 6,144 units of 2 MiB), one request's step-table row pointing
# into it, then one unit and one 16 KiB run zoomed in. The map is a seeded churn of requests allocating lowest
# free unit first (as Arena::alloc), so a long-lived request ends up scattered.

def layout():
    import heapq
    import random
    rng = random.Random(10)  # a seed whose map has windows showing every kind of unit
    N, SLOT, PAGE = 6144, 74, 32
    free = list(range(N))
    heapq.heapify(free)
    owner = [None] * N  # request id, "cache", or None
    reqs, cache, nid = {}, [], 0

    def take(r):
        u = heapq.heappop(free)
        owner[u] = r
        return u

    def give(us):
        for u in us:
            owner[u] = None
            heapq.heappush(free, u)

    for step in range(12000):
        if len(reqs) < 16 and rng.random() < 0.05:
            prompt, gen = rng.randint(1000, 12000), rng.randint(500, 5000)
            need = SLOT + -(-(prompt + gen) // PAGE)  # admit reserves everything up front, as Arena::admit
            reserved = sum(r["res"] for r in reqs.values())
            while len(free) - reserved < need and cache:
                give([cache.pop(0)])
            if len(free) - reserved >= need:
                nid += 1
                r = dict(state=[take(nid) for _ in range(SLOT)],
                         pages=[take(nid) for _ in range(-(-prompt // PAGE))], pos=prompt, end=prompt + gen)
                r["res"] = need - SLOT - len(r["pages"])
                reqs[nid] = r
        for k in list(reqs):
            r = reqs[k]
            if r["pos"] % PAGE == 0:
                r["pages"].append(take(k))
                r["res"] -= 1
            r["pos"] += 1
            if r["pos"] >= r["end"]:
                keep = r["pages"][: rng.randint(0, 24)]  # the prefix cache keeps the first pages
                for u in keep:
                    owner[u] = "cache"
                cache += keep
                give(r["state"] + r["pages"][len(keep):])
                del reqs[k]
    a_id = max(reqs, key=lambda k: len(reqs[k]["pages"]))
    A = reqs[a_id]
    a_pages, a_state = A["pages"], A["state"]

    # a window of the arena: 8 rows of 48 units, where A, two other requests, the cache and free units all show
    WC, WR = 48, 8
    WN = WC * WR

    state_of = {u: k for k, r in reqs.items() for u in r["state"]}

    def fits(w0):  # every legend entry shows: A's pages, state slots, cache-only, free
        us = range(w0, w0 + WN)
        na = sum(1 for u in us if owner[u] == a_id and u not in state_of)
        ns = sum(1 for u in us if u in state_of)
        return (na >= 4 and sum(1 for u in us if owner[u] is None) >= 6 and 10 <= ns <= 150
                and sum(1 for u in us if owner[u] == "cache") >= 4)

    def score(w0):
        us = range(w0, w0 + WN)
        return min(sum(1 for u in us if owner[u] == a_id), 10) + len({owner[u] for u in us})
    w0 = max((w for w in range(0, N - WN + 1, WC) if fits(w)), key=score)
    win = range(w0, w0 + WN)
    counts = {}
    for u in win:
        if owner[u] not in (None, "cache", a_id):
            counts[owner[u]] = counts.get(owner[u], 0) + 1
    # B and C: prefer requests whose state slot shows in the window, then the most units
    ranked = sorted(counts, key=lambda k: (-any(u in win for u in reqs[k]["state"]), -counts[k]))
    b_id, c_id = ranked[:2]
    # one more request finishes in the shown step: its units go pending (a GPU step may still read them)
    d_id = min((k for k in ranked[2:] if counts[k] >= 3), key=lambda k: counts[k], default=None)
    pending = {u for u in win if d_id is not None and owner[u] == d_id}

    g = Svg(1280, 690)
    AC = REQ_C["A"]
    names = {a_id: "A", b_id: "B", c_id: "C"}
    light = {"A": "#F8BBD0", "B": "#B2EBF2", "C": "#D7CCC8"}
    mx, pitch, my = 372, 18, 92
    map_w = WC * pitch
    g.text(mx, 24, f"the arena, 27B: 12 GiB = 6,144 units of 2 MiB, one MTLBuffer · showing units "
           f"{w0:,}–{w0 + WN - 1:,}", 15, INK, weight=700)
    # where the window sits in the whole arena
    g.rect(mx, 34, map_w, 7, fill="#F1F3F4", stroke=LINE, rx=2)
    g.rect(f"{mx + map_w * w0 / N:.1f}", 32, f"{max(map_w * WN / N, 3):.1f}", 11, fill=INK, rx=2)
    g.text(mx - 8, 42, "in all 6,144 →", 11, FAINT, "end")

    def cell(x, y, kind, who=None):
        if kind == "free":
            g.rect(x, y, pitch - 3, pitch - 3, fill="#FFFFFF", stroke="#BDC1C6", rx=2, dash="2 2")
        elif kind == "pending":
            g.rect(x, y, pitch - 3, pitch - 3, fill="url(#hatch)", stroke=STATE["pending"][1], rx=2)
        elif kind == "cache":
            g.rect(x, y, pitch - 3, pitch - 3, fill=STATE["cache-only"][0], stroke=STATE["cache-only"][1], rx=2)
        else:  # live: a page or a state unit of A / B / C, or of another request
            if who in REQ_C:
                f = REQ_C[who] if kind == "page" else light[who]
                st = REQ_C[who]
            else:
                f, st = ("#9AA0A6", "#5F6368") if kind == "page" else ("#E8EAED", "#5F6368")
            g.rect(x, y, pitch - 3, pitch - 3, fill=f, stroke=st, rx=2, sw=1.2)

    # legend: who (fill), then state (outline / pattern)
    lx = mx
    g.text(lx, 66, "live, by owner:", 12, MUTED, weight=700)
    lx += 98
    for who in ("A", "B", "C", None):
        for kind, label in (("page", "page"), ("state", "state")):
            cell(lx, 55, kind, who)
            lx += 18
        g.text(lx, 66, who or "others", 12, REQ_C.get(who, MUTED) if who else MUTED, weight=700)
        lx += 14 + 8 * len(who or "others")
    g.text(lx + 6, 66, "(solid = page, light = state slot)", 12, MUTED)
    lx = mx + 560
    for kind, label in (("cache", "cache-only"), ("pending", "pending"), ("free", "free")):
        cell(lx, 55, kind)
        g.text(lx + 20, 66, label, 12, MUTED)
        lx += 32 + 7 * len(label)

    for i, u in enumerate(win):
        x, y = mx + (i % WC) * pitch, my + (i // WC) * pitch
        if i % WC == 0:
            g.text(mx - 6, y + 12, f"{u}", 10, FAINT, "end", font=MONO)
        o = owner[u]
        if u in pending:
            cell(x, y, "pending")
        elif o is None:
            cell(x, y, "free")
        elif o == "cache":
            cell(x, y, "cache")
        else:
            st = reqs[o]["state"]
            cell(x, y, "state" if u in st else "page", names.get(o))
    map_h = WR * pitch

    # A's step-table row
    cx, cy, cw = 24, my - 6, 300
    g.rect(cx, cy, cw, 200, fill="#FFFFFF", stroke=AC, rx=8, sw=1.5)
    g.text(cx + 14, cy + 24, "A's row in the step table", 14, INK, weight=700)
    g.text(cx + 14, cy + 42, "copied in by the CPU, every step", 12, MUTED)
    g.text(cx + 14, cy + 68, f"pos {A['pos']:,}   n 1", 13, INK, font=MONO)
    inwin = [u for u in a_pages if u in win]
    picks = [inwin[0], inwin[len(inwin) // 2], inwin[-1]]
    g.text(cx + 14, cy + 94, f"pages ({len(a_pages)} numbers, in token order)", 12, MUTED)
    parts = ["…", str(picks[0]), "…", str(picks[1]), "…", str(picks[2]), "…"]
    g.text(cx + 14, cy + 114, " ".join(parts), 13, INK, font=MONO)
    g.text(cx + 14, cy + 164, f"state ({SLOT} units)", 12, MUTED)
    g.text(cx + 14, cy + 182, " ".join(map(str, a_state[:4])) + " …", 13, INK, font=MONO)

    def badge(x, y, k):
        g.out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{INK}"/>')
        g.text(f"{x:.1f}", f"{y + 4:.1f}", k, 11, "#FFFFFF", "middle", weight=700)

    for k, u in enumerate(picks):  # numbered markers: the same number in the row and on the map
        pi = (1, 3, 5)[k]
        tx = cx + 14 + sum(len(v) + 1 for v in parts[:pi]) * 7.8 + len(parts[pi]) * 3.9
        badge(tx, cy + 136, k + 1)
        i = u - w0
        ux, uy = mx + (i % WC) * pitch, my + (i // WC) * pitch
        g.rect(ux - 2.5, uy - 2.5, pitch + 2, pitch + 2, stroke=INK, sw=1.8, rx=3)
        badge(ux + pitch + 6, uy - 4, k + 1)

    # zoom 1: one unit = one page of A = a table of 16 full layers × (K, V) × 4 heads
    u0 = picks[1]
    pidx = a_pages.index(u0)
    t0 = pidx * PAGE
    ux, uy = mx + ((u0 - w0) % WC) * pitch, my + ((u0 - w0) // WC) * pitch
    gx, gy = 424, 350  # grid origin (cells)
    cw2, ch2 = 42, 12
    gw, gh = 8 * cw2, 16 * ch2
    yb = my + map_h + 6
    g.line(ux + 7.5, uy + pitch, ux + 7.5, yb, INK, 1, dash="2 2")
    g.out.append(f'<path d="M {ux:.1f} {yb} L {gx} {gy - 34} L {gx + gw} {gy - 34} L {ux + 15:.1f} {yb} Z" '
                 f'fill="{AC}" fill-opacity="0.06" stroke="{AC}" stroke-opacity="0.35"/>')
    g.text(24, gy - 10, f"② unit {u0} = A's page {pidx}", 14, INK, weight=700)
    for k, ln in enumerate([f"= A's tokens {t0:,}–{t0 + 31:,}", "",
                            "Each of these 32 tokens left, in each of", "the 16 full-attention layers (rows: model",
                            "layers 3, 7, …, 63; the other 48 are linear),", "a key K and a value V for each of the",
                            "4 KV heads (columns).", "",
                            "One cell = one layer, K or V, one head,", "for all 32 tokens = 16 KiB.",
                            "128 cells = 2 MiB, stored row by row."]):
        g.text(24, gy + 10 + k * 17, ln, 12.5, INK if k == 0 else MUTED, font=MONO if k == 0 else FONT)
    # headers
    g.text(gx + 2 * cw2, gy - 18, "K", 13, INK, "middle", weight=700)
    g.text(gx + 6 * cw2, gy - 18, "V", 13, INK, "middle", weight=700)
    for h in range(8):
        g.text(gx + (h + 0.5) * cw2, gy - 4, f"head {h % 4}", 10, MUTED, "middle")
    for gi in range(16):
        if gi % 3 == 0 or gi == 15:
            g.text(gx - 6, gy + gi * ch2 + 9.5, f"layer {4 * gi + 3}", 10, MUTED, "end")
        for h in range(8):
            g.rect(gx + h * cw2, gy + gi * ch2, cw2 - 2, ch2 - 2, fill="#FCE4EC" if h < 4 else "#F8BBD0",
                   stroke=AC, sw=0.5)
    for h in range(8):  # grouped-query attention: each stored KV head serves 6 of the 24 query heads
        q0 = (h % 4) * 6
        g.text(gx + (h + 0.5) * cw2, gy + gh + 12, f"q {q0}–{q0 + 5}", 9.5, FAINT, "middle")
    g.text(gx - 6, gy + gh + 12, "serves", 9.5, FAINT, "end")
    g.text(gx, gy + gh + 30, "24 query heads share 4 stored KV heads · in memory: row by row, left to right", 11,
           MUTED)

    # zoom 2: one cell = the 32 tokens' keys for layer g 0, head 0
    cx0, cy0 = gx, gy
    g.rect(cx0 - 1, cy0 - 1, cw2, ch2, stroke=INK, sw=1.8)
    zx, zy, zw, rh2 = 860, 350, 300, 5
    g.out.append(f'<path d="M {cx0 + cw2 - 1} {cy0 - 1} L {zx} {zy} L {zx} {zy + 32 * rh2} '
                 f'L {cx0 + cw2 - 1} {cy0 + ch2 - 1} Z" fill="{AC}" fill-opacity="0.06" stroke="{AC}" '
                 f'stroke-opacity="0.35"/>')
    g.text(zx, zy - 34, "one cell: layer 3, K, head 0", 14, INK, weight=700)
    g.text(zx, zy - 16, "32 rows (tokens) × 256 numbers × 2 B = 16 KiB", 12, MUTED)
    for t in range(32):
        g.rect(zx, zy + t * rh2, zw, rh2 - 1, fill="#FCE4EC" if t % 2 else "#F8BBD0")
    g.rect(zx, zy, zw, 32 * rh2 - 1, stroke=AC, sw=1)
    g.text(zx - 6, zy + 7, f"t {t0:,}", 10, MUTED, "end", font=MONO)
    g.text(zx - 6, zy + 32 * rh2, f"t {t0 + 31:,}", 10, MUTED, "end", font=MONO)
    g.text(zx + zw + 8, zy + 7, "← one token's key:", 11, MUTED)
    g.text(zx + zw + 8, zy + 21, "256 numbers", 11, MUTED)
    for k, ln in enumerate(["Attention for (layer 3, head 0) reads this", "cell in every page of A, in token order:",
                            "one contiguous 16 KiB read per page."]):
        g.text(zx, zy + 32 * rh2 + 24 + k * 17, ln, 12, MUTED)

    # the address, each term labelled with the zoom it picks
    fy = 624
    g.rect(24, fy - 24, 1232, 84, fill="#F8F9FA", stroke=LINE, rx=8)
    g.text(40, fy, "address =", 14, INK, font=MONO, weight=700)
    expr = ["s × 2 MiB", "  +  ", "((g × 2 + kv) × 4 + head) × 16 KiB", "  +  ", "token × 512 B"]
    labels = ["which unit (from the step table)", None, "which cell: g = full layer 0–15, kv = K/V, head", None,
              "which row in the cell"]
    x = 140
    for part, lab in zip(expr, labels):
        g.text(x, fy, part, 14, INK, font=MONO)
        w = 8.45 * len(part)
        if lab:
            g.line(x, fy + 8, x + w, fy + 8, AC, 1.5)
            g.text(x + w / 2, fy + 26, lab, 12, AC, "middle")
        x += w
    g.text(40, fy + 46, "s comes from the step table; g (layer 3 → 0, 7 → 1, …, 63 → 15), K/V, head and token are "
           "known to the kernel. No pointers are stored anywhere.", 12, MUTED)
    g.save(OUT / "layout.svg")


# ---------------------------------------------------------------------------------------------------------------
# pages.svg: the simple version. A request's tokens, cut into pages of 32, each page one arena unit, found through
# A's row in the step table. Defaults of the 27B: 32 tokens per page, 2 MiB per unit, 74 units per state slot.

A_PAGES, A_STATE, A_CACHE, A_FREE = [3, 4, 5, 22], [12, 13, 14, 15, 16], {10, 11}, {17, 18, 21}


def pages():
    g = Svg(1280, 500)
    AC, AL = REQ_C["A"], "#F8BBD0"
    n_tok, prompt = 100, 80
    slots = [3, 4, 5, 30]  # prefill took 3 neighbours at once; decode's page came later, wherever was free
    # 1. the tokens
    tx, ty, tw = 150, 70, 1000
    g.text(24, 40, "A's conversation so far: 100 tokens", 15, INK, weight=700)
    for k, (name, f) in enumerate((("prompt (80)", AC), ("generated (20)", "#E57BA6"))):
        g.rect(330 + k * 130, 29, 13, 13, fill=f, rx=2)
        g.text(349 + k * 130, 40, name, 12, MUTED)
    per = tw / 128
    for t in range(128):
        x = tx + t * per
        used = t < n_tok
        g.rect(f"{x:.1f}", ty, f"{per - 1:.1f}", 26, fill=(AC if t < prompt else "#E57BA6") if used else "#FFFFFF",
               stroke=AC if used else LINE, sw=0.4, dash=None if used else "2 2")
    for p in range(4):
        x = tx + p * 32 * per
        g.rect(f"{x - 1:.1f}", ty - 4, f"{32 * per:.1f}", 34, stroke=INK, sw=1.4, rx=3)
        last = p == 3
        g.text(f"{x + 16 * per:.0f}", ty + 50, f"page {p}", 13, INK, "middle", weight=700)
        g.text(f"{x + 16 * per:.0f}", ty + 66, "tokens 96–99 + 28 empty" if last else f"tokens {p * 32}–{p * 32 + 31}",
               12, MUTED, "middle")
    g.text(tx - 10, ty + 18, "tokens", 13, MUTED, "end")

    # 2. the arena window
    ax, ay, upw = 150, 230, 41.6
    g.text(tx - 10, ay + 20, "arena", 13, MUTED, "end")
    g.text(tx - 10, ay + 36, "units 0–23", 11, FAINT, "end")
    slots, state, cache, free = A_PAGES, A_STATE, A_CACHE, A_FREE
    for u in range(24):
        x = ax + u * upw
        g.text(f"{x + upw / 2 - 2:.0f}", ay - 8, u, 11, FAINT, "middle", font=MONO)
        if u in slots:
            g.rect(f"{x:.1f}", ay, f"{upw - 4:.1f}", 44, fill=AC, stroke=AC, rx=4)
            g.text(f"{x + upw / 2 - 2:.0f}", ay + 27, f"p{slots.index(u)}", 13, "#FFFFFF", "middle", weight=700)
        elif u in state:
            g.rect(f"{x:.1f}", ay, f"{upw - 4:.1f}", 44, fill=AL, stroke=AC, rx=4)
            g.text(f"{x + upw / 2 - 2:.0f}", ay + 27, f"s{state.index(u)}", 13, INK, "middle", weight=700)
        elif u in cache:
            g.rect(f"{x:.1f}", ay, f"{upw - 4:.1f}", 44, fill=STATE["cache-only"][0], stroke=STATE["cache-only"][1],
                   rx=4)
        elif u in free:
            g.rect(f"{x:.1f}", ay, f"{upw - 4:.1f}", 44, fill="#FFFFFF", stroke="#BDC1C6", rx=4, dash="3 3")
        else:
            g.rect(f"{x:.1f}", ay, f"{upw - 4:.1f}", 44, fill="#9AA0A6", stroke="#5F6368", rx=4)
    g.text(ax + 24 * upw + 6, ay + 20, "… up to", 11, FAINT)
    g.text(ax + 24 * upw + 6, ay + 36, "6,143", 11, FAINT, font=MONO)
    # arrows: page p → its unit
    for p, u in enumerate(slots):
        x0 = tx + (p * 32 + 16) * per
        x1 = ax + u * upw + upw / 2 - 2
        g.path(f"M {x0:.0f} {ty + 74} C {x0:.0f} {ty + 120}, {x1:.0f} {ay - 70}, {x1:.0f} {ay - 22}", AC, 1.6)
    g.text(ax + 3 * upw, ay + 62, "↑ prefill: 3 pages at once, side by side", 12, AC)
    g.text(ax + 23 * upw - 4, ay + 62, "decode: a new page at token 96, wherever a unit is free ↑", 12, AC, "end")
    # legend
    lx, ly = ax, ay + 90
    for name, f, st, d in (("A's pages", AC, AC, None), ("A's state", AL, AC, None),
                           ("other requests", "#9AA0A6", "#5F6368", None),
                           ("cache-only", STATE["cache-only"][0], STATE["cache-only"][1], None),
                           ("free", "#FFFFFF", "#BDC1C6", "3 2")):
        g.rect(lx, ly - 11, 13, 13, fill=f, stroke=st, rx=3, dash=d)
        g.text(lx + 19, ly, name, 12, MUTED)
        lx += 40 + 7 * len(name)
    g.text(lx + 10, ly, "· A's state slot is 74 units: 5 here, the rest elsewhere", 12, MUTED)

    # 3. the step table row, and the one-line sizes
    sy = 360
    g.rect(24, sy, 600, 56, fill="#FFFFFF", stroke=AC, rx=8, sw=1.5)
    g.text(40, sy + 22, "A's row in the step table", 13, INK, weight=700)
    g.text(40, sy + 44, "pages  3 4 5 22    state  12 13 14 15 16 … (74)", 14, INK, font=MONO)
    g.text(644, sy + 22, "That list of numbers is all the GPU needs to find A's memory:", 13, MUTED)
    g.text(644, sy + 42, "unit s starts at byte s × 2 MiB. Units never move.", 13, MUTED)
    g.rect(24, sy + 76, 1232, 40, fill="#F8F9FA", stroke=LINE, rx=8)
    g.text(40, sy + 101, "1 page = 32 tokens = 1 unit = 2 MiB (each token: 64 KiB of K and V)  ·  "
           "every 32 new tokens, one more unit  ·  the state slot never grows", 14, INK)
    g.save(OUT / "pages.svg")


# ---------------------------------------------------------------------------------------------------------------
# zoom-*.svg: inside one block, one level per picture (27B). Each picture outlines in yellow what the next one opens.
#   K/V:   zoom-arena (units) → zoom-unit (a row per full-attention layer) → zoom-row (K/V × 4 heads) → zoom-cell
#          (a line per token)
#   state: zoom-arena → zoom-state (74 units, one byte span) → zoom-state-layer (conv + delta of one linear layer)
# Same window and request as pages.svg: A has 100 tokens, pages 3 4 5 22, state 12 13 14 15 16 …
# Layouts: memory.mm kv_offset / attention.metal (K/V), memory.mm state_offsets + state_layer_bytes (state).

Z_AC, Z_AL, Z_AL2, Z_AV = REQ_C["A"], "#F8BBD0", "#FCE4EC", "#F48FB1"
KIB, MIB = 1 << 10, 1 << 20
Z_UNIT = 2 * MIB
Z_CONV, Z_DELTA = 3 * 10240 * 2, 48 * 128 * 128 * 4  # conv: (kernel 4 − 1) × conv_dim × bf16; delta: Hv × 128 × 128 fp32
Z_PER = Z_CONV + Z_DELTA
Z_LINEAR = [l for l in range(64) if l % 4 != 3]
Z_FULL = [l for l in range(64) if l % 4 == 3]
Z_PATHS = {"kv": ["arena", "K/V unit", "layer row", "cell"], "state": ["arena", "state slot", "one layer"]}


def zoom_frame(title, sub, path, cur, h=720):
    """No title in the picture (the slide or post heading carries it, and numbers differ between them): a breadcrumb
    of zoom levels, the current one bold, then the one-line context."""
    g = Svg(1280, h)
    items = Z_PATHS[path] if path else ["arena"]
    spans = []
    for i, it in enumerate(items):
        on = i == cur
        spans.append(f'<tspan fill="{INK if on else FAINT}" font-weight="{700 if on else 400}">{escape(it)}</tspan>')
    sep = f'<tspan fill="{FAINT}">  ›  </tspan>'
    g.out.append(f'<text x="24" y="38" font-family="{FONT}" font-size="17" xml:space="preserve">{sep.join(spans)}</text>')
    for i, ln in enumerate(wrap_words(sub, 150)):
        g.text(24, 66 + i * 20, ln, 15, MUTED)
    return g


def zoom_next(g, x, y, w, h, label, below=True):
    g.rect(f"{x - 3:.1f}", f"{y - 3:.1f}", f"{w + 6:.1f}", f"{h + 6:.1f}", stroke=HL_LINE, sw=3, rx=5)
    ty = y + h + 22 if below else y - 12
    g.text(f"{x + w / 2:.1f}", f"{ty:.1f}", label, 12.5, "#9A7B00", "middle", weight=700)


def zoom_arena():
    g = zoom_frame("Zoom 1 · the arena: every unit is a page or a piece of a state",
                   "A has 100 tokens: 4 K/V pages and a 74-unit state slot. The same free bits gave out both kinds; "
                   "the units themselves are identical 2 MiB blocks.", None, 0)
    x0, y0, uw, uh = 70, 150, 47, 74
    for u in range(24):
        x = x0 + u * uw
        g.text(x + uw / 2 - 2, y0 - 10, u, 12, FAINT, "middle", font=MONO)
        if u in A_PAGES:
            g.rect(x, y0, uw - 4, uh, fill=Z_AC, stroke=Z_AC, rx=5)
            g.text(x + uw / 2 - 2, y0 + 43, f"p{A_PAGES.index(u)}", 14, "#FFFFFF", "middle", weight=700)
        elif u in A_STATE:
            g.rect(x, y0, uw - 4, uh, fill=Z_AL, stroke=Z_AC, rx=5)
            g.text(x + uw / 2 - 2, y0 + 43, f"s{A_STATE.index(u)}", 14, INK, "middle", weight=700)
        elif u in A_CACHE:
            g.rect(x, y0, uw - 4, uh, fill=STATE["cache-only"][0], stroke=STATE["cache-only"][1], rx=5)
        elif u in A_FREE:
            g.rect(x, y0, uw - 4, uh, fill="#FFFFFF", stroke="#BDC1C6", rx=5, dash="3 3")
        else:
            g.rect(x, y0, uw - 4, uh, fill="#9AA0A6", stroke="#5F6368", rx=5)
    g.text(x0 + 24 * uw + 4, y0 + 34, "…", 16, FAINT)
    g.text(x0 + 24 * uw + 4, y0 + 54, "6,143", 11, FAINT, font=MONO)
    zoom_next(g, x0 + 22 * uw, y0, uw - 4, uh, "K/V unit ↓")
    zoom_next(g, x0 + 12 * uw, y0, uw - 4, uh, "state slot ↓")
    lx, ly = x0, y0 + 130
    for name, f, st, d in (("A's K/V pages", Z_AC, Z_AC, None), ("A's state", Z_AL, Z_AC, None),
                           ("other requests", "#9AA0A6", "#5F6368", None),
                           ("cache-only", STATE["cache-only"][0], STATE["cache-only"][1], None),
                           ("free", "#FFFFFF", "#BDC1C6", "3 2")):
        g.rect(lx, ly - 11, 13, 13, fill=f, stroke=st, rx=3, dash=d)
        g.text(lx + 19, ly, name, 13, MUTED)
        lx += 46 + 7.5 * len(name)
    # what the two kinds hold, and who says which is which
    y = 340
    g.rect(24, y, 1232, 64, fill="#FFFFFF", stroke=Z_AC, rx=8, sw=1.5)
    g.text(40, y + 26, "A's row in the step table: the only place that says which unit is which", 13, INK, weight=700)
    g.text(40, y + 50, "pages  3 4 5 22        state  12 13 14 15 16 … (74 numbers)", 15, INK, font=MONO)
    cards = [(Z_AC, "#FFFFFF", "a K/V page",
              ["32 tokens of keys and values,", "for the 16 full-attention layers.", "One more every 32 tokens."]),
             (Z_AL, INK, "a piece of the state",
              ["Part of one fixed summary for the", "48 linear-attention layers.", "74 units, for the request's whole life."])]
    for k, (fill, ink, head, lines) in enumerate(cards):
        x = 24 + k * 624
        g.rect(x, 440, 608, 130, fill="#FFFFFF", stroke=LINE, rx=8)
        g.rect(x + 16, 460, 28, 28, fill=fill, stroke=Z_AC, rx=4)
        g.text(x + 58, 480, head, 15, INK, weight=700)
        for j, ln in enumerate(lines):
            g.text(x + 58, 506 + j * 20, ln, 14, INK)
    g.text(24, 610, "The bitmap, the refcounts and alloc() never look inside a unit. Only the attention kernels do, and "
           "they learn what a unit is from the table.", 14, MUTED)
    g.save(OUT / "zoom-arena.svg")


def zoom_unit():
    g = zoom_frame("Zoom 2 · one K/V unit: a row per full-attention layer",
                   "Unit 22 = A's page 3, tokens 96–127 (96–99 written so far). 16 rows, one per full-attention layer, "
                   "in memory order: row g starts at byte g × 128 KiB of the unit.", "kv", 1)
    x0, y0, cw, rh = 210, 160, 106, 26
    for j, (name, x) in enumerate((("K", x0), ("V", x0 + 4 * cw))):
        g.text(x + 2 * cw, y0 - 30, name, 16, INK, "middle", weight=700)
        for h in range(4):
            g.text(x + h * cw + cw / 2, y0 - 10, f"head {h}", 12, MUTED, "middle")
    g.text(x0 - 16, y0 - 10, "model layer", 12, MUTED, "end")
    g.text(x0 + 8 * cw + 16, y0 - 10, "byte offset", 12, MUTED)
    for r, layer in enumerate(Z_FULL):
        y = y0 + r * rh
        g.text(x0 - 16, y + 18, f"layer {layer}", 13, INK, "end")
        g.text(x0 - 80, y + 18, f"g {r}", 11, FAINT, "end", font=MONO)
        for c in range(8):
            g.rect(x0 + c * cw + 1, y + 1, cw - 2, rh - 2, fill=Z_AL if c < 4 else Z_AV, stroke=Z_AC, sw=0.7, rx=2)
        g.text(x0 + 8 * cw + 16, y + 18, f"{r * 128:,} KiB" if r else "0", 12, FAINT, font=MONO)
    zoom_next(g, x0, y0, 8 * cw, rh, "", below=False)
    g.text(x0 + 8 * cw + 104, y0 + 18, "← next", 12.5, "#9A7B00", weight=700)
    yb = y0 + 16 * rh + 40
    g.text(x0, yb, "16 rows × 128 KiB = 2 MiB. Each row: K and V for the 4 KV heads of that layer.", 14, INK)
    g.text(x0, yb + 22, "Layers 0–2, 4–6, … 60–62 are linear attention: they live in the state slot.", 14, MUTED)
    g.save(OUT / "zoom-unit.svg")


def zoom_row():
    g = zoom_frame("Zoom 3 · one row: layer 3's keys and values",
                   "8 cells of 16 KiB: a K and a V for each of the 4 KV heads, 128 KiB in all. The model has 24 query "
                   "heads; each group of 6 shares one KV head (grouped-query attention).", "kv", 2)
    x0, y0, cw, gap, ch = 24, 180, 146, 6, 170
    for c in range(8):
        kv, h = ("K", "V")[c // 4], c % 4
        x = x0 + c * (cw + gap) + (20 if c >= 4 else 0)
        g.rect(x, y0, cw, ch, fill=Z_AL if c < 4 else Z_AV, stroke=Z_AC, sw=1.2, rx=6)
        g.text(x + cw / 2, y0 + 40, f"{kv} · head {h}", 16, INK, "middle", weight=700)
        g.text(x + cw / 2, y0 + 66, "16 KiB", 14, INK, "middle")
        g.text(x + cw / 2, y0 + 96, "32 tokens ×", 12.5, MUTED, "middle")
        g.text(x + cw / 2, y0 + 114, "256 numbers × 2 B", 12.5, MUTED, "middle")
        g.text(x + cw / 2, y0 + ch + 22, f"+{c * 16} KiB" if c else "+0", 12, FAINT, "middle", font=MONO)
        g.text(x + cw / 2, y0 - 14, f"query heads {h * 6}–{h * 6 + 5}", 12, MUTED, "middle")
    zoom_next(g, x0, y0, cw, ch, "")
    g.text(x0 + cw / 2, y0 + ch + 46, "next ↓", 12.5, "#9A7B00", "middle", weight=700)
    # query heads → KV heads
    yq = 470
    g.text(24, yq - 14, "who reads them: 24 query heads, 6 per KV head", 14, INK, weight=700)
    qw = 1232 / 24
    for q in range(24):
        x = 24 + q * qw
        g.rect(f"{x + 2:.1f}", yq, f"{qw - 4:.1f}", 30, fill="#FFFFFF", stroke="#80868B", rx=4)
        g.text(f"{x + qw / 2:.1f}", yq + 20, f"q{q}", 11.5, INK, "middle", font=MONO)
    for h in range(4):
        x = 24 + h * 6 * qw
        g.path(f"M {x + 4:.1f} {yq + 38} L {x + 4:.1f} {yq + 46} L {x + 6 * qw - 4:.1f} {yq + 46} "
               f"L {x + 6 * qw - 4:.1f} {yq + 38}", MUTED, 1.2, marker=False)
        g.text(f"{x + 3 * qw:.1f}", yq + 66, f"read K head {h} and V head {h}", 12.5, MUTED, "middle")
    g.text(24, 620, "So a token stores 4 keys + 4 values per full layer, not 24 + 24: "
           "16 layers × 8 × 512 B = 64 KiB per token.", 14, INK)
    g.save(OUT / "zoom-row.svg")


def zoom_cell():
    g = zoom_frame("Zoom 4 · one cell: a line per token",
                   "Layer 3, K, head 0 of A's page 3. 32 lines, tokens 96–127; each line is one token's key: "
                   "256 numbers × bf16 = 512 B. The cell is 16 KiB with no gaps.", "kv", 3)
    x0, y0, w, lh = 190, 130, 760, 12.5
    for t in range(32):
        y = y0 + t * lh
        tok = 96 + t
        written = tok < 100
        nxt = tok == 100
        g.text(x0 - 12, y + 10, tok, 10.5, INK if written or nxt else FAINT, "end", font=MONO,
               weight=700 if nxt else 400)
        g.rect(x0, f"{y + 1:.1f}", w, f"{lh - 2:.1f}", fill=Z_AL if written else "#FFFFFF",
               stroke=Z_AC if written else "#D5D7DA", sw=0.7, dash=None if written else "3 2")
        if t == 0:
            g.text(x0 + w + 12, y + 10, "512 B: token 96's key, 256 numbers", 12, MUTED)
    yn = y0 + 4 * lh
    g.rect(x0 - 2, f"{yn - 1:.1f}", w + 4, f"{lh + 2:.1f}", stroke=HL_LINE, sw=2.5, rx=2)
    g.text(x0 + w + 12, f"{yn + 10:.1f}", "← the next decode step writes token 100 here,", 12.5, "#9A7B00", weight=700)
    g.text(x0 + w + 12, f"{yn + 26:.1f}", "   and on the same line of all 128 cells", 12.5, "#9A7B00", weight=700)
    g.text(x0 + w + 12, f"{y0 + 31 * lh + 10:.1f}", "empty until tokens 101–127 arrive", 12, FAINT)
    # the address, one term per zoom level
    ya = y0 + 32 * lh + 40
    g.rect(24, ya, 1232, 112, fill="#F8F9FA", stroke=LINE, rx=8)
    terms = [("22 × 2 MiB", "arena: which unit", "from the step table"),
             ("0 × 128 KiB", "unit: which layer row", "g = 0 for layer 3"),
             ("0 × 16 KiB", "row: K/V and head", "(kv × 4 + head)"),
             ("4 × 512 B", "cell: which line", "token 100 − 96")]
    g.text(40, ya + 40, "address =", 17, INK, weight=700, font=MONO)
    x = 150
    for k, (t, a, b) in enumerate(terms):
        g.text(x, ya + 40, t, 17, INK, weight=700, font=MONO)
        g.text(x, ya + 66, a, 12.5, Z_AC, weight=700)
        g.text(x, ya + 84, b, 12, MUTED)
        x += 250
        if k < 3:
            g.text(x - 40, ya + 40, "+", 17, INK, font=MONO)
    g.save(OUT / "zoom-cell.svg")


def zoom_state():
    g = zoom_frame("Zoom 5 · the state slot: 74 units, one byte span",
                   "A's state for the 48 linear layers is one 146.8 MiB array. Top row: the units it is stored in, in "
                   "step-table order. Bottom row: what the bytes are. Same byte axis; only the start and the end shown.",
                   "state", 1)
    x0, gap = 160, 90
    segs = [(0, 10), (142, 148)]  # MiB ranges shown, same scale, "…" between
    scale = (1256 - x0 - gap) / sum(b - a for a, b in segs)

    def X(mib):
        x = x0
        for k, (a, b) in enumerate(segs):
            if a <= mib <= b:
                return x + (mib - a) * scale
            x += (b - a) * scale + gap
        raise ValueError(mib)

    yu, yl, h = 170, 290, 64
    g.text(x0 - 16, yu + 30, "units", 15, INK, "end", weight=700)
    g.text(x0 - 16, yu + 48, "where it is", 12, MUTED, "end")
    g.text(x0 - 16, yl + 30, "layers", 15, INK, "end", weight=700)
    g.text(x0 - 16, yl + 48, "what it is", 12, MUTED, "end")
    # units: 2 MiB boxes
    names = {0: "12", 1: "13", 2: "14", 3: "15", 4: "16"}
    for (a, b) in segs:
        for u in range(a // 2, b // 2):
            xa, xb = X(2 * u), X(2 * u + 2)
            g.rect(f"{xa:.1f}", yu, f"{xb - xa:.1f}", h, fill="#FFFFFF", stroke="#3C4043", sw=1.4)
            g.text(f"{(xa + xb) / 2:.1f}", yu + 28, f"s{u}", 15, INK, "middle", weight=700)
            g.text(f"{(xa + xb) / 2:.1f}", yu + 48, f"unit {names[u]}" if u in names else "unit …", 12, MUTED,
                   "middle", font=MONO)
    # layers: 3.06 MiB bars, one colour; the unused tail
    per = Z_PER / MIB
    for j, layer in enumerate(Z_LINEAR):
        lo, hi = j * per, (j + 1) * per
        for (a, b) in segs:
            ca, cb = max(lo, a), min(hi, b)
            if ca >= cb:
                continue
            xa, xb = X(ca), X(cb)
            g.rect(f"{xa:.1f}", yl, f"{xb - xa:.1f}", h, fill=Z_AL, stroke=Z_AC, sw=1.4)
            if xb - xa > 70:
                g.text(f"{(xa + xb) / 2:.1f}", yl + 28, f"layer {layer}", 14, INK, "middle", weight=700)
                g.text(f"{(xa + xb) / 2:.1f}", yl + 48, "3.06 MiB", 12, MUTED, "middle")
            elif xb - xa > 24:
                g.text(f"{(xa + xb) / 2:.1f}", yl + 37, layer, 12, INK, "middle", weight=700)
    used = len(Z_LINEAR) * per
    xa, xb = X(used), X(148)
    g.rect(f"{xa:.1f}", yl, f"{xb - xa:.1f}", h, fill="#FFFFFF", stroke="#BDC1C6", sw=1.2, dash="4 3")
    g.text(f"{(xa + xb) / 2:.1f}", yl + 37, "unused", 12, MUTED, "middle")
    # unit borders drawn through both rows: they don't line up with the layers
    for (a, b) in segs:
        for m in range(a, b + 1, 2):
            g.line(f"{X(m):.1f}", yu + h, f"{X(m):.1f}", yl + h + 10, "#9AA0A6", 1, dash="3 3")
            anchor = "end" if m == b else ("start" if m == a and a else "middle")
            g.text(f"{X(m) + (4 if anchor == 'start' else 0):.1f}", yl + h + 28, f"{m} MiB" if m else "0", 12,
                   MUTED, anchor, font=MONO)
    # the "…" between the two ranges
    xg = X(10) + gap / 2
    for y in (yu + h / 2 + 6, yl + h / 2 + 6):
        g.text(xg, y, "…", 22, FAINT, "middle")
    g.text(xg, yl + h + 48, "68 more units", 11, FAINT, "middle")
    zoom_next(g, X(0), yl, X(per) - X(0), h, "")
    g.text(X(per / 2), yl - 14, "next ↓", 12, "#9A7B00", "middle", weight=700)
    # how to read it
    y = 470
    lines = [
        ("Units are 2 MiB, a layer's state is 3.06 MiB, so the borders don't line up. They don't need to:",
         INK, 700),
        ("the GPU finds byte x of the array in unit  state[x / 2 MiB],  at offset  x % 2 MiB.", INK, 400),
        ("Layer 1 starts at 3.06 MiB → s1 (unit 13), 1.06 MiB in. Layer 0 runs from s0 into s1.", MUTED, 400),
        ("Layers 3, 7, 11, … 63 are missing: they are full attention, and keep K/V pages instead.",
         MUTED, 400),
        ("48 layers × 3.06 MiB = 146.8 MiB → 74 units of 2 MiB = 148 MiB: the last 1.2 MiB is unused.", MUTED, 400),
    ]
    for i, (t, c, w) in enumerate(lines):
        g.text(x0, y + i * 26, t, 15 if i < 2 else 14, c, weight=w, font=MONO if i == 1 and False else FONT)
    g.save(OUT / "zoom-state.svg")


def zoom_state_layer():
    g = zoom_frame("Zoom 6 · one linear layer's state: rewritten by every token",
                   "Layer 0's 3.06 MiB, bytes 0 … 3.06 MiB of the span: it starts in s0 (unit 12) and runs into s1 "
                   "(unit 13). No line per token: every token reads all of it and writes all of it back.", "state", 2)
    x0, y0, w = 24, 150, 1232
    scale = w / Z_PER
    cw = 70  # conv drawn wider than its 60 KiB
    g.rect(x0, y0, cw, 56, fill=Z_AC, stroke=Z_AC, rx=4)
    g.text(x0 + cw / 2, y0 + 33, "conv", 13, "#FFFFFF", "middle", weight=700)
    g.rect(x0 + cw, y0, w - cw, 56, fill=Z_AL, stroke=Z_AC, rx=4)
    g.text(x0 + cw + (w - cw) / 2, y0 + 33, "delta: 48 heads × 128 × 128 × fp32 = 3 MiB", 14, INK, "middle", weight=700)
    xb = x0 + cw + (Z_UNIT - Z_CONV) * (w - cw) / Z_DELTA
    g.line(f"{xb:.1f}", y0 - 10, f"{xb:.1f}", y0 + 66, "#3C4043", 2.5, dash="5 3")
    g.text(f"{xb - 8:.1f}", y0 - 14, "s0 (unit 12)", 12, MUTED, "end", font=MONO)
    g.text(f"{xb + 8:.1f}", y0 - 14, "s1 (unit 13)", 12, MUTED, font=MONO)
    g.text(x0, y0 + 80, "60 KiB (drawn wider)", 12, MUTED)
    g.text(f"{xb:.1f}", y0 + 80, "2 MiB: a unit boundary means nothing to the layer", 12, MUTED, "middle")
    # delta heads
    yh = 290
    g.text(24, yh, "delta: one 128 × 128 matrix per value head, 64 KiB each", 15, INK, weight=700)
    hw, hh, cols = 96, 64, 12
    for h in range(48):
        r, c = divmod(h, cols)
        x = 24 + c * (hw + 7)
        y = yh + 16 + r * (hh + 8)
        g.rect(x, y, hw, hh, fill=Z_AL, stroke=Z_AC, rx=4)
        g.text(x + hw / 2, y + 28, f"head {h}", 12, INK, "middle", weight=700)
        g.text(x + hw / 2, y + 46, "128 × 128", 11, MUTED, "middle", font=MONO)
    yc = yh + 16 + 4 * (hh + 8) + 30
    g.text(24, yc, "conv: the last 3 inputs of the layer's short convolution, 3 × 10,240 × bf16 = 60 KiB.", 14, INK)
    g.text(24, yc + 24, "Compare the K/V cell: it gets one new line per token and old lines never change, so a cache can "
           "share a full page.", 14, MUTED)
    g.text(24, yc + 46, "Here every matrix changes with every token, so there is no \"state at token t\" left behind: "
           "the cache needs a copy, unless the request is done with it.", 14, MUTED)
    g.save(OUT / "zoom-state-layer.svg")


def zoom():
    zoom_arena()
    zoom_unit()
    zoom_row()
    zoom_cell()
    zoom_state()
    zoom_state_layer()


# ---------------------------------------------------------------------------------------------------------------
# engine-loop.svg: the loop around the layer loop. Time runs left to right; rows are the request queue and the two
# lanes that every Engine::step() polls and feeds (architecture.md "Engine loop"). Schematic times.

def engine_loop():
    D_C = "#455A64"
    col = dict(REQ_C, D=D_C)
    g = Svg(1280, 640)
    g.text(24, 40, "One engine loop, two lanes", 26, weight=700)
    g.text(24, 66, "Every turn of the loop: collect what the lanes finished, admit waiting requests that fit, give the "
           "prefill lane its next chunk, give the decode lane its next step.", 15, MUTED)
    g.text(24, 86, "Time runs left to right (schematic). Prefill is not outside the loop: it is a lane the loop feeds "
           "one chunk at a time, next to decode.", 15, MUTED)
    x0, x1, T = 210, 1250, 100.0

    def X(t):
        return x0 + (x1 - x0) * t / T

    rows = [("request queue", "waiting for memory", 150), ("prefill lane", "one prompt chunk at a time", 250),
            ("decode lane", "one token for every running request", 360)]
    for name, sub, y in rows:
        g.text(24, y + 22, name, 15, INK, weight=700)
        g.text(24, y + 40, sub, 12, MUTED)
        g.line(x0, y + 66, x1, y + 66, LINE)
    # events (vertical markers)
    events = [(5, "A admitted"), (68, "A's prompt done: A joins decode"), (76, "B stops: D admitted")]
    for t, label in events:
        g.line(f"{X(t):.1f}", 130, f"{X(t):.1f}", 450, FAINT, 1.2, dash="4 4")
        left = t == 68
        g.text(f"{X(t) + (-4 if left else 4):.1f}", 126, label, 12, INK, "end" if left else "start", weight=700)

    def chip(t0, t1, y, h, who, text, solid=True, dash=None):
        c = col[who]
        g.rect(f"{X(t0) + 1:.1f}", y, f"{X(t1) - X(t0) - 2:.1f}", h, fill=c if solid else "#FFFFFF", stroke=c,
               sw=1.4, rx=5, dash=dash)
        if text:
            g.text(f"{(X(t0) + X(t1)) / 2:.1f}", y + h / 2 + 5, text, 12.5, "#FFFFFF" if solid else c, "middle",
                   weight=700)

    # queue
    chip(1, 5, 162, 40, "A", "A", solid=False, dash="5 3")
    chip(58, 76, 162, 40, "D", "D waits: no room", solid=False, dash="5 3")
    # prefill lane: A's 2,499 prompt tokens before its last one, in 1,024-token chunks; then D
    for t0, t1, txt in ((6, 30, "A · tokens 0–1023"), (30, 54, "A · 1024–2047"), (54, 67, "A · 2048–2498")):
        chip(t0, t1, 262, 40, "A", txt)
    chip(78, 98, 262, 40, "D", "D · tokens 0–1023")
    # decode lane: one tick per step, coloured stripes per request in the step
    for k in range(50):
        t = 1 + k * 2
        who = ["B", "C"] if t < 68 else (["A", "B", "C"] if t < 76 else ["A", "C"])
        x = X(t)
        w = X(t + 2) - X(t) - 3
        hh = 48 / len(who)
        for j, r in enumerate(who):
            g.rect(f"{x:.1f}", f"{372 + j * hh:.1f}", f"{w:.1f}", f"{hh - 1:.1f}", fill=col[r], rx=1.5)
    for t0, t1, txt in ((1, 68, "steps with rows B, C"), (68, 76, "A, B, C"), (76, 100, "A, C")):
        g.text(f"{(X(t0) + X(t1)) / 2:.1f}", 438, txt, 12, MUTED, "middle")
    # legend
    lx, ly = x0, 480
    for r in ("A", "B", "C", "D"):
        g.rect(lx, ly - 12, 14, 14, fill=col[r], rx=3)
        g.text(lx + 20, ly, f"request {r}", 13, MUTED)
        lx += 110
    g.text(lx + 10, ly, "dashed = waiting   ·   one thin bar in the decode lane = one decode step", 13, MUTED)
    # notes
    notes = [
        "A's last prompt token is not prefilled: it is A's first input in decode, which turns it into A's first new token.",
        "While A's chunks run, B and C keep getting a token every step. On the 27B, one ANE chunk is ~5 s and one decode "
        "step ~80 ms,",
        "so in reality about 60 decode steps fit under each chunk, not the handful drawn here.",
        "Inside each bar, the same layer loop runs: a chunk pushes 1,024 rows through 64 layers, a step pushes one row per "
        "request.",
    ]
    for i, ln in enumerate(notes):
        g.text(24, 530 + i * 22, ln, 13.5, INK if i != 2 else INK)
    g.save(OUT / "engine-loop.svg")


if __name__ == "__main__":
    arena_frames()
    block_life(story())
    machine(split=False)
    machine(split=True)
    bitmap()
    spill()
    weights()
    model()
    hardware()
    layout()
    pages()
    zoom()
    engine_loop()
