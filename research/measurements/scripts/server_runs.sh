#!/bin/zsh
# Measurement 10 + 12b: mlx_lm.server TTFT/ITL under 1/4/16 concurrent streaming
# clients, then greedy determinism of one prompt alone vs under concurrent load.
# Usage: server_runs.sh <hf-repo> <tag>
set -u
T=/Users/vpromisex/videos/inference-series/tools
M=/Users/vpromisex/videos/inference-series/research/measurements
PY=$T/.venv/bin/python
MODEL=$1; TAG=$2; PORT=8090
LOG=$M/logs/server__$TAG.log
$T/.venv/bin/mlx_lm.server --model $MODEL --port $PORT --temp 0 --max-tokens 128 > $LOG 2>&1 &
SPID=$!
trap "kill $SPID 2>/dev/null" EXIT
for i in {1..120}; do curl -s localhost:$PORT/v1/models >/dev/null 2>&1 && break; sleep 1; done
echo "server up (pid $SPID)"
# warm-up (2 requests, loads weights + compiles kernels)
$PY -I $T/stream_client.py --port $PORT --clients 2 --requests 1 --tag warmup --out /tmp/warmup.json >/dev/null
for C in 1 4 16; do
  R=$(( C == 1 ? 10 : (C == 4 ? 5 : 3) ))
  $PY -I $T/stream_client.py --port $PORT --clients $C --requests $R --max-tokens 128 --tag ${TAG}__c$C --out $M/raw/server__${TAG}__c$C.json
done
# determinism: same prompt, 20 sequential requests alone
$PY -I $T/stream_client.py --port $PORT --clients 1 --requests 20 --max-tokens 128 --same-prompt $M/prompts/determinism.txt --tag ${TAG}__det_solo --out $M/raw/server__${TAG}__det_solo.json
# ... and the same 20 requests while 15 other clients keep the batch full
$PY -I $T/stream_client.py --port $PORT --clients 15 --requests 30 --max-tokens 128 --tag ${TAG}__det_load_background --out $M/raw/server__${TAG}__det_load_background.json &
LPID=$!
sleep 3
$PY -I $T/stream_client.py --port $PORT --clients 1 --requests 20 --max-tokens 128 --same-prompt $M/prompts/determinism.txt --tag ${TAG}__det_underload --out $M/raw/server__${TAG}__det_underload.json
wait $LPID
echo done
