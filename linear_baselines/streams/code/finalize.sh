#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# CPU only: every derived file from the raw records (one run at a time).  usage: ./finalize.sh
cd "$(dirname "$0")"
exec 9> ../out/.finalize.lock
flock 9
PY=${RESCOPE_PYTHON}
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 P12_THREADS=4   # at most 4 BLAS threads per process
$PY identity_reedit.py 2>&1 | grep -v Warning | cut -c1-200
$PY rho_cpu.py 2>&1 | grep -v Warning | cut -c1-200
$PY protect_split_cpu.py 2>&1 | grep -v Warning | cut -c1-200
$PY aggregate_text.py 2>&1 | grep -v Warning | tail -1
$PY aggregate.py 2>&1 | grep -v Warning | tail -1
$PY render.py | tail -1
$PY plot.py 2>&1 | grep -v Warning | wc -l
