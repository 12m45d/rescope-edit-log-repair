#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E5 optimum runs: one log after the other, 8 single-threaded workers, at most 40 minutes of CPU time (core-minutes)
# per log, earlier runs of the log included (a log whose budget is spent runs nothing and only writes its meta file).
# usage: nohup bash run_opt.sh [pid ...] &     (waits for the given pids before it starts)
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${RESCOPE_PYTHON}
for p in "$@"; do
  while kill -0 "$p" 2>/dev/null; do sleep 5; done
done
for lg in TB-Q100-ae-l2_3 TB-L100-ae-l2_3; do
  nice -n 10 $PY e5_opt.py --log $lg --workers 8 --budget-cpu-min 40 --tl 10 >> ../logs/opt_${lg}.out 2>&1
done
