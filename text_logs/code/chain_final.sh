#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# After the reported stream is complete: register T2000, describe it, self-check, render everything.
TP=${RESCOPE_ROOT}/02_research/experiments/09_testbed_scale/text_public
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $TP || exit 1
while [ ! -f $TP/logs_edit/stream/snapshots/W_T2000.pt ] || [ ! -f $TP/logs_edit/stream/updates/u_01999.pt ]; do sleep 20; done
sleep 15
echo "[$(date -u +%H:%M:%S)] stream complete: build logs"
$PY code/build_logs.py > logs/build_logs.out 2>&1
echo "[$(date -u +%H:%M:%S)] describe T2000"
$GQ run --mem 10500 --prio -8 --name tx_descD2 -- $PY code/describe.py --tags T2000 --parts dec,edit,public > logs/describe_D2.out 2>&1
echo "[$(date -u +%H:%M:%S)] generation T2000"
$GQ run --mem 10500 --prio -8 --name tx_genG2 -- $PY code/describe.py --tags T2000 --parts gen > logs/describe_G2.out 2>&1
echo "[$(date -u +%H:%M:%S)] self-check"
$GQ run --mem 9500 --prio -8 --name tx_selfcheck -- $PY code/selfcheck.py > logs/selfcheck.out 2>&1
echo "[$(date -u +%H:%M:%S)] finalize"
bash code/finalize.sh > logs/finalize.out 2>&1
echo "[$(date -u +%H:%M:%S)] chain done (exit $?)"
