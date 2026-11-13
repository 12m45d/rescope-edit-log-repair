#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# P12 verifier queue for Qwen: card 0 first (the project's convention), card 1 only when card 0 has no window and no
# LLaVA job of this verifier is running there.  gpu_verify.py applies the start rule itself (two looks, waiters of other
# agents first) and exits 75 without touching the card when it must wait; a running job leaves the card (exit 75)
# when another agent's waiting launcher is blocked only by it.      usage: vq5.sh [max_minutes]
cd "$(dirname "$0")" || exit 1
PY=${RESCOPE_PYTHON}
LOG=TB-Q100-ae-l2_3; MAXMIN=${1:-240}
end=$(( $(date +%s) + MAXMIN*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  for G in 0 1; do
    if [ "$G" = 1 ] && pgrep -f "gpu_verify.py --log TB-L100" > /dev/null; then continue; fi
    CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY gpu_verify.py --log $LOG --cap_gb 9.2 --need_mib 10000 >> "logs/gpu_$LOG.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then echo "[$(date +%T)] done $LOG (card $G)" >> "logs/queue_$LOG.out"; exit 0; fi
    if [ $rc -ne 75 ]; then echo "[$(date +%T)] FAILED rc=$rc $LOG (card $G)" >> "logs/queue_$LOG.out"; exit $rc; fi
  done
  sleep 20
done
echo "[$(date +%T)] gave up after $MAXMIN min ($LOG)" >> "logs/queue_$LOG.out"; exit 76
