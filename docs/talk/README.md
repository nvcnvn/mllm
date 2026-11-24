Diagrams for [../talk-memory.md](../talk-memory.md), one [dg](https://github.com/nvcnvn/dg) project per folder.
Edit a folder's `model.hcl`, then re-export its picture next to it:

```sh
dg check -C docs/talk/trie && dg export -C docs/talk/trie --format svg --out docs/talk/trie.svg
```
