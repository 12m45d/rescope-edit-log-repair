#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# The GPU steps of E5 and E8 for one backbone, one after another under ONE reservation of the queue (each step its own process: one log per process):
#   E5 (f) capacity curve -> E5 (b) lineage sizes under the deployed scoping / log-only thresholds -> E8 latency.  Every step is resumable; two tries per step.
# usage: gq run --mem 12500 --prio 2 -- bash gpu_all.sh qwen25vl TB-Q100-ae-l2_3      (19500: llava TB-L100-ae-l2_3)
F=${RESCOPE_ROOT}/02_research/experiments/10_final
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
BB=$1; LOG=$2
cd $F/e5_rescope
for try in 1 2; do $PY code/e5f_capacity.py $LOG >> logs/f_$LOG.out 2>&1 && break || echo "E5f try $try failed"; done
for try in 1 2; do $PY code/e5b_where.py $LOG >> logs/b_where_$LOG.out 2>&1 && break || echo "E5b try $try failed"; done
cd $F/e8_cost
for try in 1 2; do $PY code/lat.py $BB >> logs/lat_$BB.out 2>&1 && break || echo "E8 latency try $try failed"; done
echo GPU_ALL_DONE $BB
