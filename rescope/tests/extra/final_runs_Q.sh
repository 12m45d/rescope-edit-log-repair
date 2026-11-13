#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# FINAL GPU runs, Qwen, as launched for the delivered table:  nohup tests/extra/final_runs_Q.sh [key-cache root] > tests/out/run_gpu_Q.out 2>&1 &
# Resumable: results of another code version are dropped first (extra/drop_stale.py), finished tests are skipped.
#   1. scale test (T = 2000, fresh process)   2. extras (one log per registry family; peak RAM lean / harness loader)
#   3. the whole GPU suite on the development log (b1 .. b14, c)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
[ -n "$1" ] && export RESCOPE_KEYCACHE_ROOT="$1"
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
cd ${RESCOPE_ROOT}/02_research/code/rescope
( flock 9; $PY tests/extra/drop_stale.py ) 9>tests/out/.drop.lock
$GQ run --mem 10500 --name rescope_scale -- $PY tests/run_tests.py scale --mem_gb 9.8
echo "SCALE_EXIT $?"
$GQ run --mem 10500 --name rescope_families -- $PY tests/extra/load_families.py --mem_gb 9.8
echo "FAMILIES_EXIT $?"
$GQ run --mem 10500 --name rescope_peak_lean -- $PY tests/extra/load_peak.py TB-Q2000-ae-l2_3 --lean 1 --mem_gb 9.8
$GQ run --mem 10500 --name rescope_peak_harness -- $PY tests/extra/load_peak.py TB-Q2000-ae-l2_3 --lean 0 --mem_gb 9.8
echo "PEAK_EXIT $?"
$GQ run --mem 10800 --name rescope_tests_Q -- $PY tests/run_tests.py gpu --log TB-Q100-ae-l2_3 --mem_gb 10.1
echo "Q_SUITE_EXIT $?"
echo CHAIN_Q_DONE
