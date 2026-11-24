#!/bin/zsh
# one hold: mlx_lm.benchmark p512 g128 B=1 n3 for each variant in the given order
export HF_HOME=/Volumes/T5/huggingface
L=/Volumes/T5/cache/mllm/g16/logs; M=/Volumes/T5/cache/mllm/models; P=/Volumes/T5/Codes/mllm/.venv/bin/python
tag=$1; shift
echo "$(date +%T) bench $tag start $(memory_pressure | tail -1)" >> $L/chain.log
top -l 2 -o cpu -n 8 -stats pid,command,cpu,mem | tail -9 > $L/bench_${tag}_top.log
for r in "$@"; do
  m=$M/Qwen3.6-27B-$r; [[ $r == 4bit ]] && m=mlx-community/Qwen3.6-27B-4bit
  echo "=== $r $(date +%T)" >> $L/bench_$tag.log
  perl -e 'alarm 360; exec @ARGV' $P -m mlx_lm benchmark --model $m -p 512 -g 128 -b 1 -n 3 >> $L/bench_$tag.log 2>&1
  echo "$(date +%T) bench $tag $r rc=$?" >> $L/chain.log
done
