#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage: run_gpu.sh <need_MiB> <job name> <cmd ...>
# Launch a GPU job of this build.  Through the shared queue 02_research/tools/gpuq/gq once work/.use_gq exists (created
# by hand after the queue's README.txt was read and the queue was found usable); until then through the fallback
# launcher code/gpu_fallback.sh (nvidia-smi: free >= need + 1500 MiB on two looks; the job caps itself).
W=${RESCOPE_ROOT}/02_research/experiments/10_final/cache
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
NEED=$1; NAME=$2; shift 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $W
if [ -f $W/work/.use_gq ] && [ -x $GQ ]; then
  echo "[run_gpu $(date +%H:%M:%S)] via gq: $NAME need $NEED MiB"
  exec $GQ run --mem $NEED --name $NAME -- "$@"
else
  echo "[run_gpu $(date +%H:%M:%S)] via fallback launcher: $NAME need $NEED MiB"
  GPU_HOLD=150 exec code/gpu_fallback.sh ${GPU_ORDER:-1,0} $NEED "$@"
fi
