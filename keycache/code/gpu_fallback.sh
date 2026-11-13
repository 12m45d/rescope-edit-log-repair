#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Fallback GPU launcher, used ONLY while the shared queue 02_research/tools/gpuq/gq does not exist.
# usage: gpu_fallback.sh <gpus in order of preference, e.g. 0,1> <need_MiB> <cmd ...>
# Waits until one of the cards has need_MiB + 1500 MiB free on two looks 25-45 s apart (nobody is loading a model
# in between), then runs the command on it (CUDA_VISIBLE_DEVICES; the command sees the card as cuda:0).  The command
# caps its own memory.  Retries up to 6 times after a non-zero exit (the capture is resumable).  Never kills anything.
# Our own launches are serialised with a lock file (the look and the first 120 s of the command happen under it).
GPUS=$1; NEED=$2; shift 2
HOLD=${GPU_HOLD:-120}
LOCK="${RESCOPE_ROOT}/02_research/experiments/10_final/cache/work/.gpu_fallback.lock"
exec 9>"$LOCK"
free() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
tries=0
while true; do
  GPU=""
  while [ -z "$GPU" ]; do
    flock 9
    for g in ${GPUS//,/ }; do
      f1=$(free $g)
      if [ "$f1" -ge $((NEED + 1500)) ]; then
        sleep $((RANDOM % 20 + 25))
        f2=$(free $g)
        if [ "$f2" -ge $((NEED + 1500)) ] && [ $((f1 - f2)) -le 400 ]; then GPU=$g; break; fi
      fi
    done
    if [ -z "$GPU" ]; then flock -u 9; sleep 45; fi
  done
  echo "[gpu_fallback $(date +%H:%M:%S)] GPU $GPU free $f2 MiB >= $NEED + 1500 (try $tries): $*"
  CUDA_VISIBLE_DEVICES=$GPU "$@" 9>&- &
  pid=$!
  ( sleep $HOLD; flock -u 9 ) 2>/dev/null &
  holder=$!
  wait $pid
  rc=$?
  kill $holder 2>/dev/null; flock -u 9
  if [ $rc -eq 0 ]; then exit 0; fi
  tries=$((tries + 1))
  echo "[gpu_fallback $(date +%H:%M:%S)] exit code $rc (try $tries)"
  if [ $tries -ge 6 ]; then exit $rc; fi
  sleep $((RANDOM % 60 + 60))
done
