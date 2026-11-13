#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage (through the GPU queue only; one queue job per backbone):
#   gq run --mem 10500|17000 --prio 3 --name mme_<backbone> -- bash run_bb.sh <qwen25vl | llava>
# The GPU steps of a backbone run one after another, each in a fresh process:
#   1. proof     30 rows of the EXISTING benign pool through fc_capture.main, compared bit for bit with the cache; a difference
#                stops the chain before anything of MME is captured (exit 2)
#   2. capture   the MME rows through fc_capture.main (resumable parts: a restart continues)
#   3. assemble  (CPU) and verify: 50 seeded MME rows captured again with the plain library calls, without the vision-feature
#                memo, in a fresh process -- only if code/READY_VERIFY exists (otherwise they are run as a separate queue job)
BB=$1
M=${RESCOPE_ROOT}/02_research/experiments/10_final/cache2/mme
PY=${RESCOPE_PYTHON}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $M/work || exit 9
say() { echo "[run_bb $BB $(date +%H:%M:%S)] $*"; }
[ -n "$GQ_MEM" ] || { say "not under gq (GQ_MEM is not set): GPU steps run only through the queue"; exit 9; }
say "start: GQ_MEM=$GQ_MEM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES pid $$"
if [ "$2" != "verify_only" ]; then
  $PY -I -B $M/code/mme_capture.py $BB proof; rc=$?
  [ $rc -eq 0 ] || { say "PROOF FAILED (exit $rc): nothing of MME was captured"; exit 2; }
  $PY -I -B $M/code/mme_capture.py $BB capture; rc=$?
  [ $rc -eq 0 ] || { say "CAPTURE FAILED (exit $rc)"; exit 3; }
fi
if [ -f $M/code/READY_VERIFY ] || [ "$2" = "verify_only" ]; then
  $PY -I -B $M/code/mme_assemble.py $BB; rc=$?
  [ $rc -eq 0 ] || { say "ASSEMBLY FAILED (exit $rc)"; exit 4; }
  $PY -I -B $M/code/mme_verify.py $BB gpu; rc=$?
  [ $rc -eq 0 ] || { say "VERIFY FAILED (exit $rc)"; exit 5; }
  say "FINISHED (assembly, fresh-process verification)"
else
  say "FINISHED (proof, capture); assembly and verification are run as a separate job"
fi
