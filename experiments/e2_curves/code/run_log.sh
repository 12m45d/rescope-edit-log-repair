#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# One GPU job of the queue per log: a mechanics pass on every 40th sequence (private output directory), then the real blocks.
# usage: gq run --mem <MiB> --prio 3 -- bash code/run_log.sh <log> [blocks]
set -u
LOG=$1
BLOCKS=${2:-reedit_build,capture,base,cw,d,b1,c,b2,gen,e}
EXP=${RESCOPE_ROOT}/02_research/experiments/10_final/e2_curves
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $EXP
if [ ! -f out_smoke/PASSED_$LOG ]; then
  rm -rf out_smoke/cells/*/$LOG out_smoke/capture/$LOG.pt out_smoke/parts/*$LOG* 2>/dev/null
  E2_OUT=$EXP/out_smoke $PY code/job.py $LOG --smoke 40 --blocks capture,base,cw,d,b1,c,gen || { echo "SMOKE FAILED $LOG"; exit 3; }
  touch out_smoke/PASSED_$LOG
fi
$PY code/job.py $LOG --blocks $BLOCKS
