#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E6 GPU steps of one log, one after another inside ONE queue job: behaviour of the default row and of the linear-response row first
# (the main table needs it), then the gate level of the multi-position roles, then the remaining behavioural rows.  Resumable.
set -e
LOG=$1
cd "$(dirname "$0")"
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$PY e6_behave.py $LOG default resp_lin
$PY e6_multi.py $LOG
$PY e6_behave.py $LOG
