#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# remaining gate-level cells: development logs first (all levels + the selected _x forms), then the working points
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e1_main/code
for lg in TB-Q100-ae-l2_3 TB-L100-ae-l2_3; do ${RESCOPE_PYTHON} e1_gate.py $lg > ../logs/gate_rest_$lg.log 2>&1 & done
wait
for lg in TB-Q100-ae TB-Q100-ae-l2_1 TB-L100-ae; do ${RESCOPE_PYTHON} e1_gate.py $lg > ../logs/gate_rest_$lg.log 2>&1 & done
wait
echo ALL GATE DONE
