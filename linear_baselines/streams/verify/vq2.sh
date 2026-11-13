#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# P12 verifier queue: run gpu_verify.py for one log when the card has room (the python checks the start rule itself
# and exits 75 without touching the GPU when it must wait).   usage: vq.sh <gpu> <need_mib> <cap_gb> <log> [max_minutes]
cd "$(dirname "$0")" || exit 1
PY=${RESCOPE_PYTHON}
G=$1; NEED=$2; CAP=$3; LOG=$4; MAXMIN=${5:-120}
end=$(( $(date +%s) + MAXMIN*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY gpu_verify.py --log "$LOG" --cap_gb "$CAP" --need_mib "$NEED" >> "logs/gpu_$LOG.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date +%T)] done $LOG" >> "logs/queue_$LOG.out"; exit 0; fi
  if [ $rc -ne 75 ]; then echo "[$(date +%T)] FAILED rc=$rc $LOG" >> "logs/queue_$LOG.out"; exit $rc; fi
  sleep 20
done
echo "[$(date +%T)] gave up after $MAXMIN min ($LOG)" >> "logs/queue_$LOG.out"; exit 76
