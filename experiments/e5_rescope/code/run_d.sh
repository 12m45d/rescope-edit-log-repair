#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e5_rescope
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for L in TB-Q100-ae-l2_3 TB-L100-ae-l2_3; do
  $PY code/e5d_least.py $L > logs/d_$L.out 2>&1 || echo "FAILED $L"
done
echo ALLDONE
