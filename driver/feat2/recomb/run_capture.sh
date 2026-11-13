#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage: run_capture.sh <qwen25vl | llava> [extra arguments of rc_capture.py]
# One GPU job through the shared queue (02_research/tools/gpuq/gq): verify the capture path bit for bit, capture the recombination
# keys in resumable parts, re-check a sample.  Start with nohup; a restarted job skips the finished parts.
#   nohup ./run_capture.sh qwen25vl > /dev/null 2>&1 &        (log: cache2/recomb/logs/nohup_capture_<backbone>.out)
F=${RESCOPE_ROOT}/02_research/experiments/10_final/feat2/recomb
D=${RESCOPE_ROOT}/02_research/experiments/10_final/cache2/recomb
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
BB=$1; shift
case $BB in qwen25vl) MEM=10500; TAG=rcx_Q;; llava) MEM=17000; TAG=rcx_L;; *) echo "backbone?"; exit 1;; esac
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p $D/logs
cd $F
echo "[run_capture $(date '+%F %T')] $BB via gq --mem $MEM --prio 4 --name $TAG $*" >> $D/logs/nohup_capture_$BB.out
exec $GQ run --mem $MEM --prio 4 --name $TAG -- $PY $F/rc_capture.py $BB "$@" >> $D/logs/nohup_capture_$BB.out 2>&1
