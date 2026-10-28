#!/bin/zsh
# Downloads the course's measurement models into ~/.cache/huggingface (small first).
HF=/Users/vpromisex/videos/inference-series/tools/.venv/bin/hf
for r in Llama-3.2-1B-Instruct-4bit Llama-3.2-1B-Instruct-8bit Llama-3.2-1B-Instruct-bf16 Qwen3-1.7B-4bit Meta-Llama-3.1-8B-Instruct-4bit Meta-Llama-3.1-8B-Instruct-8bit Meta-Llama-3.1-8B-Instruct-bf16; do
  echo "=== $r $(date +%T)"
  $HF download mlx-community/$r --quiet && echo "OK $r $(date +%T)" || echo "FAIL $r"
done
echo ALLDONE
