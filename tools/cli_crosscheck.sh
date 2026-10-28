#!/bin/zsh
# Cross-check the harness against the stock mlx-lm CLIs (the commands an episode
# would show on screen). Output: logs/cli_crosscheck.log
set -u
V=/Users/vpromisex/videos/inference-series/tools/.venv/bin
M=/Users/vpromisex/videos/inference-series/research/measurements
LOG=$M/logs/cli_crosscheck.log
: > $LOG
CHAT="$(cat $M/prompts/chat.txt)"
for R in Llama-3.2-1B-Instruct-4bit Llama-3.2-1B-Instruct-8bit Llama-3.2-1B-Instruct-bf16 \
         Meta-Llama-3.1-8B-Instruct-4bit Meta-Llama-3.1-8B-Instruct-8bit Meta-Llama-3.1-8B-Instruct-bf16; do
  for i in 0 1 2 3; do   # run 0 = warm-up (discarded in the summary)
    echo "## run $i: mlx_lm.generate --model mlx-community/$R --prompt \"\$(cat prompts/chat.txt)\" --max-tokens 256 --temp 0" >> $LOG
    $V/mlx_lm.generate --model mlx-community/$R --prompt "$CHAT" --max-tokens 256 --temp 0 2>&1 | grep -E '^(Prompt|Generation|Peak memory):' >> $LOG
  done
done
for B in 1 4 16; do
  echo "## mlx_lm.benchmark --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit -p 2048 -g 128 -b $B -n 3" >> $LOG
  $V/mlx_lm.benchmark --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit -p 2048 -g 128 -b $B -n 3 2>&1 | grep -E '^(Trial|Averages|Timing)' >> $LOG
done
echo "## mlx_lm.benchmark --model mlx-community/Meta-Llama-3.1-8B-Instruct-bf16 -p 512 -g 128 -b 1 -n 3" >> $LOG
$V/mlx_lm.benchmark --model mlx-community/Meta-Llama-3.1-8B-Instruct-bf16 -p 512 -g 128 -b 1 -n 3 2>&1 | grep -E '^(Trial|Averages|Timing)' >> $LOG
echo done >> $LOG
