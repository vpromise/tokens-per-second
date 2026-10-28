#!/bin/zsh
# My own check of finished episodes: briefs/check.sh 12 [13 ...]
# Runs verify_audio inside each folder with every ALL-CAPS term the captions contain.
PY=~/rag-video/tts/.venv/bin/python
V=~/.claude/skills/tutorial-video/scripts/verify_audio.py
for n in "$@"; do
  d=$(print -l ~/videos/inference-series/ep$(printf %02d $n)-*(/N) | head -1)
  s=$(grep -m1 -oE "^class Ep[0-9]+[A-Za-z0-9]+" $d/scene.py | cut -d' ' -f2)
  terms=$(grep -oE "\b[A-Z][A-Z0-9]+s?\b" $d/$s.srt | sed -E 's/s$//' | grep -E "^[A-Z0-9]{2,}$" | grep -vE "^[0-9]+$" | sort -u | paste -sd, -)
  echo "=== $n $s  terms: ${terms:-AI}"
  (cd $d && afinfo $s.mp4 | grep "estimated duration"; [[ -f $s.html && -f $s.srt ]] && echo "html+srt ok"; $PY $V $s.mp4 --terms ${terms:-AI} 2>&1 | grep -v -i -E "warn|semaphore|fetching|^\s*$|deprecat|librosa|audioread|transcript opens")
done
