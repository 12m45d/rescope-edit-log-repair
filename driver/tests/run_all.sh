#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# The whole test suite of the fx driver.  `run_all.sh` = CPU tests + GPU tests; `run_all.sh gpu` = GPU tests only.
# GPU tests run in groups, every group its own job of the queue (names fx_*; --prio FX_PRIO, default -8: other jobs go first; the
# queue's own policy file state/prio_override.json may replace the priority), at most three groups at a time (two Qwen-size, one
# LLaVA-size), so that the jobs of other agents interleave.  Tests a and d use private empty memo stores, b / c / f / g the shared one
# (append-only shards: safe side by side).  A test that passed with the present code (sha1 of fx/*.py and rescope/*.py) is skipped:
# the script can be re-started at any time.
cd "$(dirname "$0")" || exit 1
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PRIO=${FX_PRIO:--8}
Q=TB-Q100-ae-l2_3; L=TB-L100-ae-l2_3
mkdir -p out
if [ "$1" != "gpu" ]; then $PY t_cpu.py > out/cpu_all.log 2>&1 & fi
job() {   # job <log> <MiB> <tag> <tests ...>: development log, tests a b c d f g
  local log=$1 mem=$2 tag=$3; shift 3
  $GQ run --mem $mem --name fx_$tag --prio $PRIO -- $PY t_gpu.py $log "$@" >> out/gpu_$log.log 2>&1; }
mech() {  # mech <log> <MiB>: mechanics only (test e) on any registered log that exists; no metric is computed
  $PY -c "import sys; sys.path.insert(0, '..'); import fx; sys.exit(0 if fx.logs.exists('$1') else 1)" 2>/dev/null || { echo "$1: not there, skipped" > out/gpu_$1.log; return; }
  $GQ run --mem $2 --name fx_e_${1#TB-} --prio $PRIO -- $PY t_gpu.py $1 e > out/gpu_$1.log 2>&1; }
: > out/gpu_$Q.log; : > out/gpu_$L.log
( job $Q 10500 ac_Q a c
  job $Q 10500 d_Q d ) &
( job $Q 10500 bfg_Q b f g
  mech TB-Q2000-ae-l2_3 10500
  mech TB-Q2000-memitw 10500
  mech TB-B500-ae 10500
  mech TB-Q500-onehop-l2_3 10500 ) &
( job $L 17000 bfg_L b f g
  job $L 17000 ac_L a c
  job $L 17000 d_L d
  mech TB-L1000-memitw 17000 ) &
wait
$PY render.py > out/render.log 2>&1
echo "done: $(date)"; tail -n 3 out/render.log
