#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Stage 2 of the fx driver: EVERY test on the present code -- the tests of stage 2 (t_r1_cpu.py, t_r1_gpu.py), the tests of stage 1
# (t_ext_cpu.py, t_ext_gpu.py), the driver's own tests (t_cpu.py; t_gpu.py a b c f g in full, d as a smoke run, mechanics e on one long
# log per backbone) and the decision digest; then render_r1.py --ready2 writes tests/out/table_r1.txt and ../READY.json (stage 2).
# GPU jobs go through the queue at priority 5 (names drv2_*), two jobs per backbone.  The prefix logs are NOT rebuilt (fx/prefix.py is
# unchanged; `python -m fx.prefix build` is the stage-1 step).  `run_r1.sh` after every change of fx/*.py; results: tests/out/.
cd "$(dirname "$0")" || exit 1
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
Q=TB-Q100-ae-l2_3; L=TB-L100-ae-l2_3
mkdir -p out/logs
run() {   # run <MiB> <tag> <log file> <command ...>
  local mem=$1 tag=$2 log=$3; shift 3
  $GQ run --mem $mem --name drv2_$tag --prio 5 -- "$@" >> out/logs/$log 2>&1; }
new() {   # the GPU tests of stage 2 on a development log
  : > out/logs/r1_gpu_$3.log
  run $2 r1_$3 r1_gpu_$3.log $PY t_r1_gpu.py $1 c q h; }
old() {   # old <log> <MiB> <letter> <long log for mechanics>: the GPU tests of stage 1 and of the first driver
  local log=$1 mem=$2 x=$3 long=$4
  : > out/logs/s2_old_$x.log
  run $mem ext_$x s2_old_$x.log $PY t_ext_gpu.py $log h m r s k
  run $mem extp_$x s2_old_$x.log $PY t_ext_gpu.py $log-p10 p
  run $mem old_$x s2_old_$x.log $PY t_gpu.py $log a b f g c --force
  run $mem olds_$x s2_old_$x.log $PY t_gpu.py $log d --smoke --force
  run $mem mech_$x s2_old_$x.log $PY t_gpu.py $long e; }
new $Q 10500 Q &
new $L 17000 L &
old $Q 10500 Q TB-Q2000-memitw &
old $L 17000 L TB-L1000-memitw &
( $PY t_r1_cpu.py > out/logs/r1_cpu.log 2>&1 ) &
( $PY t_ext_cpu.py > out/logs/s2_ext_cpu.log 2>&1 ) &
( $PY t_cpu.py > out/logs/s2_cpu_all.log 2>&1; $PY digest.py after > out/logs/s2_digest_after.log 2>&1 ) &
wait
$PY render.py > out/logs/s2_render.log 2>&1
$PY render_r1.py --ready2 > out/logs/s2_render_r1.log 2>&1
echo "done: $(date)"; tail -n 4 out/logs/s2_render_r1.log
