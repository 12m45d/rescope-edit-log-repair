#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# gate level of the dense gates (router, z-score) on the working points; started after the behavioural job of the same backbone (at most two GPU jobs of E1 at a time)
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e1_main/code
( while ! grep -q "JOB DONE\|Traceback" ../logs/behave_Q.log; do sleep 30; done
  ${RESCOPE_ROOT}/02_research/tools/gpuq/gq run --mem 10500 --prio 3 --name e1_wpQ10 -- ${RESCOPE_PYTHON} e1_behave.py TB-Q100-ae --dense-only > ../logs/dense_TB-Q100-ae.log 2>&1
  ${RESCOPE_ROOT}/02_research/tools/gpuq/gq run --mem 10500 --prio 3 --name e1_wpQ1 -- ${RESCOPE_PYTHON} e1_behave.py TB-Q100-ae-l2_1 --dense-only > ../logs/dense_TB-Q100-ae-l2_1.log 2>&1 ) &
( while ! grep -q "JOB DONE\|Traceback" ../logs/behave_L.log; do sleep 30; done
  ${RESCOPE_ROOT}/02_research/tools/gpuq/gq run --mem 17000 --prio 3 --name e1_wpL10 -- ${RESCOPE_PYTHON} e1_behave.py TB-L100-ae --dense-only > ../logs/dense_TB-L100-ae.log 2>&1 ) &
wait
echo WP DENSE DONE
