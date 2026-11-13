#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E9 GPU stages of one log, each stage its own queue job (so that jobs of higher priority can start in between).  Resumable.
LOG=$1; MEM=$2; TAG=$3
cd "$(dirname "$0")"
PY=${RESCOPE_PYTHON}
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for ST in base m1 m5 m10 m20 m50; do
  $GQ run --mem $MEM --prio 1 --name e9_${TAG}_$ST -- $PY e9_run.py $LOG $ST || echo "[run_gpu] stage $ST of $LOG failed with code $?"
done
echo "[run_gpu] all stages submitted and finished for $LOG"
