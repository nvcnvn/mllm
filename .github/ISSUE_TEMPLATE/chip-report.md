---
name: Chip report
about: Paste the output of `mmmllm compare` from your Mac
title: "Chip report: <chip, e.g. M2 Pro 16 GB>"
labels: chip-report
---

Run this in Terminal (needs uv: `brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`):

```sh
uvx --python 3.12 --from 'mmmllm[compare]' mmmllm compare
```

and paste its whole output as-is:

```
paste here
```
