#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# One chain of GPU jobs, run one after another through the queue (two chains = at most two GPU jobs of this task at a time).
#   run_chain.sh <log> [<log> ...]      a log with 1000 or more edits waits until the gate-level jobs of the long logs are done
#   (at most ONE process at a time on a log with 1000 or more edits).  Analysis + render after every log.
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
cd "$(dirname "$0")"
for log in "$@"; do
  case "$log" in
    *Q1000*|*Q2000*) while [ ! -f ../logs/gate_long.DONE ]; do sleep 60; done ;;
  esac
  for try in 1 2; do
    $GQ run --mem 10500 --prio 2 --name e3q_${log#TB-} -- $PY e3_behave.py "$log" >> ../logs/behave_$log.log 2>&1 && break
    echo "[chain] $log attempt $try ended with $?" >> ../logs/behave_$log.log
  done
  echo "behave $log finished $(date)" >> ../logs/chain.log
  if [ -f e3_analyze.py ] && [ -f e3_render.py ]; then
    (flock 9; $PY e3_analyze.py >> ../logs/analyze.log 2>&1; $PY e3_render.py >> ../logs/analyze.log 2>&1) 9> ../logs/.analyze.lock
  fi
done
echo "CHAIN DONE $* $(date)" >> ../logs/chain.log
