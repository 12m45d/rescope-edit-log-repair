#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E2 on prefix logs: one chain of queue jobs (one log per job, one after the other; every job resumable).
#   chain.sh <mem MiB> <pid to wait for | 0> <log> [<log> ...]
EXP=${RESCOPE_ROOT}/02_research/experiments/10_final/e2_prefix
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $EXP
mem=$1; waitpid=$2; shift; shift
if [ "$waitpid" != "0" ]; then while kill -0 "$waitpid" 2>/dev/null; do sleep 20; done; fi
for log in "$@"; do
  for try in 1 2; do
    echo "[chain $(date +%H:%M:%S)] queue $log (try $try)"
    $GQ run --mem $mem --prio 1 --name e2p_${log#TB-} -- $PY code/job.py $log >> logs/job_$log.out 2>&1
    rc=$?
    echo "[chain $(date +%H:%M:%S)] $log exit $rc"
    if [ $rc -eq 0 ] && tail -n 3 logs/job_$log.out | grep -q "job finished clean"; then break; fi
  done
done
echo "[chain $(date +%H:%M:%S)] CHAIN DONE: $*"
