#!/bin/bash
# SUPERSEDED (project rule of 2026-11-10): launch GPU jobs through the shared queue 02_research/tools/gpuq/gq
# (docs/USAGE.md, section Tests).  Kept for reference only.
# usage: gpu_launch.sh <gpus, e.g. 0 or 0,1 (order of preference)> <need_MiB> <cmd ...>
# Waits until one of the GPUs has need_MiB + 1024 MiB free on two looks 20-40 s apart (nobody is loading a model in
# between), then runs the command on it (CUDA_VISIBLE_DEVICES; the command sees the card as cuda:0).  Retries up to 8
# times on a non-zero exit (out-of-memory races with the other agents' jobs; the tests are resumable).  Never kills anything.
# Launchers of this package are serialised by a lock file: the look and the first HOLD seconds of the command (model
# load) happen under the lock, so two of our own jobs can never claim the same free memory (they did once: both looked
# at the same card in the same second and the second one ran out of memory while loading).
GPUS=$1; NEED=$2; shift 2
HOLD=${GPU_LAUNCH_HOLD:-90}
LOCK="$(cd "$(dirname "$0")" && pwd)/out/.gpu_launch.lock"
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
free() { nvidia-smi -i $1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
tries=0
while true; do
  GPU=""
  while [ -z "$GPU" ]; do
    flock 9
    for g in ${GPUS//,/ }; do
      f1=$(free $g)
      if [ "$f1" -ge $((NEED + 1024)) ]; then
        sleep $((RANDOM % 20 + 20))
        f2=$(free $g)
        if [ "$f2" -ge $((NEED + 1024)) ] && [ $((f1 - f2)) -le 400 ]; then GPU=$g; break; fi
      fi
    done
    if [ -z "$GPU" ]; then flock -u 9; sleep 45; fi
  done
  echo "[gpu_launch $(date +%H:%M:%S)] GPU $GPU free $f2 MiB >= $NEED + 1024 (try $tries): $*"
  CUDA_VISIBLE_DEVICES=$GPU "$@" 9>&- &
  pid=$!
  ( sleep $HOLD; flock -u 9 ) 2>/dev/null &
  holder=$!
  wait $pid
  rc=$?
  kill $holder 2>/dev/null; flock -u 9
  if [ $rc -eq 0 ]; then exit 0; fi
  tries=$((tries + 1))
  echo "[gpu_launch $(date +%H:%M:%S)] exit code $rc (try $tries)"
  if [ $tries -ge 8 ]; then exit $rc; fi
  sleep $((RANDOM % 60 + 60))
done
