Diagrams for [../talk-memory.md](../talk-memory.md).

`ram.svg`, `split.svg`, `weights.svg`, `pages.svg`, `bitmap.svg`, `block-life.svg`, `spill.svg`, `arena-*.svg` (one
request's life through the arena, its step tables and its counters), `zoom-*.svg` (inside one block, one level per
picture), the FAQ's `hardware.svg` and `model.svg`, and `layout.svg` (all zoom levels on one page, not used by the
talk) come from [draw.py](draw.py). The arena frames and `block-life.svg` replay `Arena` and
`Engine::admit` / `snapshot` / `finish` on a 15-unit arena (v0.3.0 rules), so the counters drawn are the ones the code would hold. Edit the story or the layout there, then:

```sh
python3 docs/talk/draw.py
```

`trie.svg` is a [dg](https://github.com/nvcnvn/dg) project in `trie/`. Edit its `model.hcl`, then re-export:

```sh
dg check -C docs/talk/trie && dg export -C docs/talk/trie --format svg --out docs/talk/trie.svg
```
