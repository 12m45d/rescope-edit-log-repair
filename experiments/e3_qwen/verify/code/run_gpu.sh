#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Two chains through the queue (at most two GPU jobs of this verification at a time; ONE process at a time on a log with 1000 or more edits:
# TB-L1000-memitw starts only after TB-Q2000-memitw has finished).  Resumable: finished stages are skipped.
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e3_qwen/verify
one() {  # <mem> <log>
  for try in 1 2; do
    $GQ run --mem $1 --name v3_${2#TB-} -- $PY code/v_gpu.py $2 >> logs/gpu_$2.log 2>&1 && break
    echo "[chain] $2 attempt $try ended with $?" >> logs/gpu_$2.log
  done
  echo "done $2 $(date)" >> logs/chain.log
}
free -g >> logs/chain.log
(one 10500 TB-Q2000-memitw; one 10500 TB-Q500-ae-l2_3; one 10500 TB-Q500-memitw) &
(one 17000 TB-L500-ae-l2_3; one 17000 TB-L500-memitw
 while [ ! -f out/gpu/TB-Q2000-memitw/DONE.json ] && ! grep -q "done TB-Q2000-memitw" logs/chain.log; do sleep 30; done
 free -g >> logs/chain.log
 one 17000 TB-L1000-memitw) &
wait
echo "ALL DONE $(date)" >> logs/chain.log
