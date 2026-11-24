#!/bin/zsh
D=${0:A:h}
L=/Volumes/T5/cache/mllm/g16/logs
R=($@); (( $#R )) || R=(8bit 3bit mixed_3_6 2bit mixed_2_6)
for r in $R; do
  [[ -e /Volumes/T5/cache/mllm/models/Qwen3.6-27B-$r/model.safetensors.index.json ]] && continue
  echo "$(date +%T) queue $r" >> $L/chain.log
  /Volumes/T5/cache/heavy g16 perl -e 'alarm 1800; exec @ARGV' $D/convert_step.sh $r
done
echo "$(date +%T) chain done" >> $L/chain.log
