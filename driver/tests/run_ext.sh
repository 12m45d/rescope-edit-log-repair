#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# GPU part of the stage-1 extension: the prefix logs, the extension's GPU tests and the EXISTING GPU tests of the driver on the
# two development logs (a b c f g in full, d as a smoke run; mechanics e on one long log per backbone), all through the queue
# at priority 5 (names drv1_*), one chain per backbone.  `run_ext.sh` after every change of fx/*.py; results: tests/out/.
cd "$(dirname "$0")" || exit 1
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
Q=TB-Q100-ae-l2_3; L=TB-L100-ae-l2_3
mkdir -p out/logs
run() {   # run <MiB> <tag> <log file> <command ...>
  local mem=$1 tag=$2 log=$3; shift 3
  $GQ run --mem $mem --name drv1_$tag --prio 5 -- "$@" >> out/logs/$log 2>&1; }
chain() { # chain <log> <MiB> <letter> <long log for mechanics>
  local log=$1 mem=$2 x=$3 long=$4
  : > out/logs/ext_$x.log
  ( cd .. && $GQ run --mem $mem --name drv1_prefix_$x --prio 5 -- $PY -m fx.prefix build $log >> tests/out/logs/ext_$x.log 2>&1 )
  $PY t_ext_cpu.py prefix >> out/logs/ext_$x.log 2>&1
  run $mem ext_$x ext_$x.log $PY t_ext_gpu.py $log h m r s k
  run $mem extp_$x ext_$x.log $PY t_ext_gpu.py $log-p10 p
  run $mem old_$x ext_$x.log $PY t_gpu.py $log a b f g c --force
  run $mem olds_$x ext_$x.log $PY t_gpu.py $log d --smoke --force
  run $mem mech_$x ext_$x.log $PY t_gpu.py $long e; }
chain $Q 10500 Q TB-Q2000-memitw &
chain $L 17000 L TB-L1000-memitw &
wait
echo "done: $(date)"
