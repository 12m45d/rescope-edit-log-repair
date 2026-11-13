#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Final step of E5 (linear part): waits for the given pids (the optimum runs), then aggregates and renders.
# usage: nohup bash finish.sh [pid ...] &
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=${RESCOPE_PYTHON}
for p in "$@"; do
  while kill -0 "$p" 2>/dev/null; do sleep 5; done
done
$PY e5_aggregate.py && $PY e5_render.py
