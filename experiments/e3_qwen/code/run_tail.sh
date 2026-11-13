#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# After the last behavioural job: second read-out (R8) of the gate-level-only rows on the two long logs (one process at a time on a
# log with 1000 or more edits), then the final analysis + render.
PY=${RESCOPE_PYTHON}
cd "$(dirname "$0")"
while ! grep -q "CHAIN DONE .*TB-Q2000-memitw" ../logs/chain.log 2>/dev/null; do sleep 60; done
for log in TB-Q1000-memitw TB-Q2000-memitw; do
  $PY e3_gate.py $log --modes qd >> ../logs/gate_$log.log 2>&1 || echo "gate qd $log failed $(date)" >> ../logs/chain.log
done
(flock 9; $PY e3_analyze.py >> ../logs/analyze.log 2>&1; $PY e3_render.py >> ../logs/analyze.log 2>&1) 9> ../logs/.analyze.lock
echo "TAIL DONE $(date)" >> ../logs/chain.log
