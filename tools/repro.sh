#!/bin/bash
# tools/repro.sh [MODEL] [-p 512] [-g 128] [-b 1] [-n 3]
# Runs mlx_lm.benchmark and `build/mllm bench` with the same settings on this Mac and prints both plus the ratio.
# Paste the whole output into an issue. Needs `make` and `pip install -r requirements.txt` first (README).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=mlx-community/Qwen3.5-0.8B-4bit
[[ $# -gt 0 && $1 != -* ]] && { MODEL=$1; shift; }
ARGS=(-p 512 -g 128 -b 1 -n 3 "$@")  # later flags win in both tools
PY=${PYTHON:-.venv/bin/python}
[[ -x $PY ]] || PY=python3

echo "== machine"
echo "chip:   $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
echo "macOS:  $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
echo "mllm:   $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo ' (dirty)')"
echo "mlx:    $("$PY" -c 'import mlx.core as m, importlib.metadata as d; print(m.__version__, "mlx-lm", d.version("mlx-lm"))')"
echo "load:   $(sysctl -n vm.loadavg)  (a busy machine moves the numbers; top CPU users:)"
ps -Ao pcpu,comm -r | sed -n 2,4p
echo "args:   $MODEL ${ARGS[*]}"

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
echo; echo "== mlx-lm"
"$PY" -m mlx_lm benchmark --model "$MODEL" "${ARGS[@]}" | tee "$T/mlx"
echo; echo "== mllm"
build/mllm bench "$MODEL" "${ARGS[@]}" | tee "$T/ours"

echo; echo "== mllm / mlx-lm (peak memory isn't comparable: mlx-lm's includes its buffer cache)"
grep -h '^Averages' "$T/mlx" "$T/ours" | awk -F'[=,]' '
  NR == 1 { p = $2; g = $4 } NR == 2 { printf "prefill %.0f / %.0f = %.2fx\ndecode  %.0f / %.0f = %.2fx\n", $2, p, $2 / p, $4, g, $4 / g }'
