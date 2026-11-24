#!/bin/zsh
# one hold: reference (8bit) log-probs, then each variant against it. One model per process.
export HF_HOME=/Volumes/T5/huggingface
D=${0:A:h}
L=/Volumes/T5/cache/mllm/g16/logs; M=/Volumes/T5/cache/mllm/models; P=/Volumes/T5/Codes/mllm/.venv/bin/python
echo "$(date +%T) quality start $(memory_pressure | tail -1)" >> $L/chain.log
top -l 2 -o cpu -n 8 -stats pid,command,cpu,mem | tail -9 > $L/quality_top.log
run() { perl -e 'alarm 420; exec @ARGV' $P $D/quality.py "$@" >> $L/quality.log 2>&1; echo "$(date +%T) quality $2 rc=$?" >> $L/chain.log; }
[[ -e /Volumes/T5/cache/mllm/g16/quality/8bit.json ]] || run $M/Qwen3.6-27B-8bit 8bit --ref
run mlx-community/Qwen3.6-27B-4bit 4bit
for r in 3bit mixed_3_6 2bit mixed_2_6; do run $M/Qwen3.6-27B-$r $r; done
