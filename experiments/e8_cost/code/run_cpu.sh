#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# E8 CPU costs: one process at a time; logs with >= 1000 edits wait for the validity runs of E5 (c) (one process at a time on such logs)
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e8_cost
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for L in TB-Q100-ae-l2_3 TB-L100-ae-l2_3 TB-Q100-memitw TB-Q500-memitw TB-L100-memitw; do
  [ -f out/cpu/$L.json ] || $PY code/cost_cpu.py real $L > logs/cpu_$L.out 2>&1 || echo "FAILED $L"
done
while ! grep -q ALLDONE ../e5_rescope/logs/run_c.out 2>/dev/null; do sleep 30; done
for L in TB-Q1000-memitw TB-Q2000-memitw TB-L1000-memitw; do
  [ -f out/cpu/$L.json ] || $PY code/cost_cpu.py real $L > logs/cpu_$L.out 2>&1 || echo "FAILED $L"
done
for T in 1000 10000 100000; do for B in qwen25vl llava; do
  [ -f out/cpu/${B}__synthetic-T$T.json ] || $PY code/cost_cpu.py synth $B $T > logs/cpu_synth_${B}_$T.out 2>&1 || echo "FAILED synth $B $T"
done; done
echo ALLDONE
