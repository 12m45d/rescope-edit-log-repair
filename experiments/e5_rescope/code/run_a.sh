#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E5 (a): one log per process, sequentially; finished logs are skipped
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e5_rescope
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for L in TB-Q100-ae-l2_3 TB-Q200-ae-l2_3 TB-Q500-ae-l2_3 TB-Q100-memitw TB-Q500-memitw TB-Q1000-memitw TB-Q2000-memitw TB-L100-ae-l2_3 TB-L200-ae-l2_3 TB-L500-ae-l2_3 TB-L100-memitw TB-L500-memitw TB-L1000-memitw; do
  [ -f out/a/$L.json ] && continue
  $PY code/e5a_rebase.py $L > logs/a_$L.out 2>&1 || echo "FAILED $L"
done
echo ALLDONE
