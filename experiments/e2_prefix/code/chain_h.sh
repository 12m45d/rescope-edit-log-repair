#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E2 on prefix logs: second pass (phase H: D2 decodes of the fixed subset of rows), after the chain of the backbone has finished.
#   chain_h.sh <mem MiB> <pid of the first chain> <log> [<log> ...]
EXP=${RESCOPE_ROOT}/02_research/experiments/10_final/e2_prefix
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $EXP
mem=$1; waitpid=$2; shift; shift
while kill -0 "$waitpid" 2>/dev/null; do sleep 30; done
for log in "$@"; do
  echo "[chain_h $(date +%H:%M:%S)] queue $log"
  $GQ run --mem $mem --prio 1 --name e2pH_${log#TB-} -- $PY code/job.py $log --phases H >> logs/jobH_$log.out 2>&1
  echo "[chain_h $(date +%H:%M:%S)] $log exit $?"
done
echo "[chain_h $(date +%H:%M:%S)] CHAIN DONE: $*"
