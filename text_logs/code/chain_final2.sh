#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Remaining steps after the stream and the T2000 description, one GPU job of this build at a time (the cards are
# shared with the final experiments): wait for the running generation job, then generation at T2000, self-check,
# and the CPU finalisation.  usage: chain_final2.sh <pid of the running gq wrapper to wait for>
TP=${RESCOPE_ROOT}/02_research/experiments/09_testbed_scale/text_public
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $TP || exit 1
if [ -n "$1" ]; then while kill -0 "$1" 2>/dev/null; do sleep 10; done; fi
echo "[$(date -u +%H:%M:%S)] generation T2000"
$GQ run --mem 10500 --prio -8 --name tx_genG2 -- $PY code/describe.py --tags T2000 --parts gen > logs/describe_G2.out 2>&1
echo "[$(date -u +%H:%M:%S)] generation, anything left (resumable)"
$GQ run --mem 10500 --prio -8 --name tx_genG3 -- $PY code/describe.py --tags dev100 T100 W0 T500 T1000 T2000 --parts gen > logs/describe_G3.out 2>&1
echo "[$(date -u +%H:%M:%S)] self-check"
$GQ run --mem 9500 --prio -8 --name tx_selfcheck -- $PY code/selfcheck.py > logs/selfcheck.out 2>&1
echo "[$(date -u +%H:%M:%S)] finalize"
bash code/finalize.sh > logs/finalize.out 2>&1
echo "[$(date -u +%H:%M:%S)] chain done (exit $?)"
