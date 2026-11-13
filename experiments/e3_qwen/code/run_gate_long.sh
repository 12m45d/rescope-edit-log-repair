#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Gate level + factor level of the two long logs, one after the other (one process at a time on a log with 1000 or more edits).
PY=${RESCOPE_PYTHON}
cd "$(dirname "$0")"
for log in TB-Q1000-memitw TB-Q2000-memitw; do
  $PY e3_gate.py $log >> ../logs/gate_$log.log 2>&1 || echo "gate $log failed $(date)" >> ../logs/chain.log
  $PY e3_factor.py $log >> ../logs/factor.log 2>&1 || echo "factor $log failed $(date)" >> ../logs/chain.log
done
touch ../logs/gate_long.DONE
