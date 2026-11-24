#!/bin/zsh
# convert_step.sh RECIPE: one convert inside a hold (memory check, time -l, log)
export HF_HOME=/Volumes/T5/huggingface
D=${0:A:h}
L=/Volumes/T5/cache/mllm/g16/logs; r=$1
SRC=/Volumes/T5/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/1cda14dd71a3f47115850eb790f416f165da1cb5
DST=/Volumes/T5/cache/mllm/models/Qwen3.6-27B-$r
free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %)
echo "$(date +%T) start $r free $free%" >> $L/chain.log
if (( free < 45 )); then echo "$(date +%T) abort $r: free $free%" >> $L/chain.log; exit 1; fi
rm -rf $DST
CONVERT_LIMIT_GB=30 /usr/bin/time -l /Volumes/T5/Codes/mllm/.venv/bin/python $D/convert.py $SRC $DST $r > $L/convert_$r.log 2>&1
rc=$?
echo "$(date +%T) end $r rc=$rc $(grep '^done' $L/convert_$r.log)" >> $L/chain.log
