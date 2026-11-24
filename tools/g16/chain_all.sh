#!/bin/zsh
D=${0:A:h}
$D/chain_convert.sh
L=/Volumes/T5/cache/mllm/g16/logs
echo "$(date +%T) queue quality" >> $L/chain.log
/Volumes/T5/cache/heavy g16 perl -e 'alarm 1800; exec @ARGV' $D/hold_quality.sh
echo "$(date +%T) queue bench r1" >> $L/chain.log
/Volumes/T5/cache/heavy g16 perl -e 'alarm 1800; exec @ARGV' $D/hold_bench.sh r1 4bit 8bit 3bit mixed_3_6 2bit mixed_2_6
echo "$(date +%T) queue bench r2" >> $L/chain.log
/Volumes/T5/cache/heavy g16 perl -e 'alarm 1800; exec @ARGV' $D/hold_bench.sh r2 mixed_2_6 2bit mixed_3_6 3bit 8bit 4bit
echo "$(date +%T) eval chain done" >> $L/chain.log
