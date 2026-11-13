#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E3 / E4 on the LLaVA logs: behavioural jobs through the queue (two chains, the longest log last), analysis + render after every log.
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e3_llava
PY=${RESCOPE_PYTHON}
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
one() {
  $GQ run --mem 17000 --prio 2 --name e3L_$1 -- $PY code/e3_behave.py $1 >> logs/behave_$1.log 2>&1
  echo "exit $? $1 $(date)" >> logs/run_all.log
  if [ -f code/e3_analyze.py ]; then (flock 9; $PY code/e3_analyze.py >> logs/analyze.log 2>&1; $PY code/e3_render.py >> logs/analyze.log 2>&1) 9> logs/.analyze.lock; fi
}
(one TB-L100-ae-l2_3; one TB-L200-ae-l2_3; one TB-L500-ae-l2_3) &
(one TB-L100-memitw; one TB-L500-memitw) &
wait
free -g >> logs/run_all.log
one TB-L1000-memitw
echo "ALL DONE $(date)" >> logs/run_all.log
