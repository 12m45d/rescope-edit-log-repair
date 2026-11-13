#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# FINAL GPU run, LLaVA, as launched for the delivered table:  nohup tests/extra/final_runs_L.sh [key-cache root] > tests/out/run_gpu_L.out 2>&1 &
# Resumable: results of another code version are dropped first, finished tests are skipped.  The whole GPU suite (b1 .. b14, c).
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -n "$1" ] && export RESCOPE_KEYCACHE_ROOT="$1"
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
cd ${RESCOPE_ROOT}/02_research/code/rescope
( flock 9; $PY tests/extra/drop_stale.py ) 9>tests/out/.drop.lock
$GQ run --mem 17000 --name rescope_tests_L -- $PY tests/run_tests.py gpu --log TB-L100-ae-l2_3 --mem_gb 16.1
echo "L_SUITE_EXIT $?"
echo CHAIN_L_DONE
