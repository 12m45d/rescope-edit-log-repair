#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# P12 verifier queue for LLaVA (GPU 1), two footprints (the python applies the start rule itself and exits 75
# without touching the card when it must wait):
#   full : every condition (REEDIT z* needs backward passes)   cap 15.0 GiB, card need 15920 MiB
#   lean : the conditions without REEDIT (forwards only)        cap 14.4 GiB, card need 15310 MiB
# usage: vq3.sh [max_minutes]
cd "$(dirname "$0")" || exit 1
PY=${RESCOPE_PYTHON}
LOG=TB-L100-ae-l2_3; G=1; MAXMIN=${1:-300}
LEAN=NONE,W0,BEST__COMP_B__nALL,COMP_B__n10,RELB8,PROTKEYS,MODDIAG
end=$(( $(date +%s) + MAXMIN*60 ))
run() { CUDA_VISIBLE_DEVICES=$G PYTHONDONTWRITEBYTECODE=1 $PY gpu_verify.py --log $LOG "$@" >> "logs/gpu_$LOG.log" 2>&1; }
while [ "$(date +%s)" -lt "$end" ]; do
  run --cap_gb 15.0 --need_mib 15920; rc=$?
  if [ $rc -eq 0 ]; then echo "[$(date +%T)] done $LOG" >> "logs/queue_$LOG.out"; exit 0; fi
  if [ $rc -ne 75 ]; then echo "[$(date +%T)] FAILED rc=$rc $LOG (full)" >> "logs/queue_$LOG.out"; exit $rc; fi
  run --cap_gb 14.4 --need_mib 15310 --conds $LEAN; rc=$?
  if [ $rc -ne 0 ] && [ $rc -ne 75 ]; then echo "[$(date +%T)] FAILED rc=$rc $LOG (lean)" >> "logs/queue_$LOG.out"; exit $rc; fi
  sleep 20
done
echo "[$(date +%T)] gave up after $MAXMIN min ($LOG)" >> "logs/queue_$LOG.out"; exit 76
