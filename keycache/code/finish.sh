#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage: finish.sh <backbone>      (nohup, after the capture jobs of the backbone were started)
# Waits until every capture part of the backbone exists (gives up if parts are missing and no capture process has been
# seen for 10 minutes), assembles the cache, then runs the verifications (CPU; fresh GPU capture of the seeded sample; the
# development log's handle) and the table.  GPU steps go through code/run_gpu.sh (the shared queue).
BB=$1
W=${RESCOPE_ROOT}/02_research/experiments/10_final/cache
PY=${RESCOPE_PYTHON}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $W
case $BB in qwen25vl) NEED=10500; CAP=9.7;; llava) NEED=17000; CAP=16.0;; *) echo "backbone?"; exit 2;; esac
say() { echo "[finish $BB $(date +%H:%M:%S)] $*"; }
idle=0
while true; do
  n=$($PY code/fc_missing.py $BB 2>/dev/null)
  [ "$n" = "0" ] && break
  if pgrep -f "fc_capture.py $BB " > /dev/null; then idle=0; else idle=$((idle + 1)); fi
  if [ $idle -ge 20 ]; then say "GIVING UP: $n parts missing and no capture process for 10 minutes"; exit 1; fi
  sleep 30
done
while pgrep -f "fc_capture.py $BB " > /dev/null; do sleep 10; done
say "all parts exist; assembling"
$PY code/fc_assemble.py $BB || { say "ASSEMBLY FAILED"; exit 1; }
$PY code/fc_verify.py $BB cpu > logs/verify_cpu_$BB.out 2>&1 || { say "VERIFY CPU FAILED"; tail -5 logs/verify_cpu_$BB.out; exit 1; }
say "cpu verification written"
code/run_gpu.sh $NEED fc_verify_$BB $PY code/fc_verify.py $BB gpu --mem_gb $CAP > logs/verify_gpu_$BB.out 2>&1 || { say "VERIFY GPU FAILED"; tail -5 logs/verify_gpu_$BB.out; exit 1; }
say "gpu verification written"
code/run_gpu.sh $NEED fc_verify_dev_$BB $PY code/fc_verify.py $BB dev --mem_gb $CAP > logs/verify_dev_$BB.out 2>&1 || { say "VERIFY DEV FAILED"; tail -5 logs/verify_dev_$BB.out; exit 1; }
say "dev verification written"
$PY code/fc_table.py > logs/table.out 2>&1 || { say "TABLE FAILED"; tail -5 logs/table.out; exit 1; }
say "FINISHED"
