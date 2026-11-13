#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# the gate alone for the synthetic log with 100000 edits (needs 14.5 GB on a card): after the Qwen chain of this agent, so that at most two of its GPU jobs run at a time
cd ${RESCOPE_ROOT}/02_research/experiments/10_final/e8_cost
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq; PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while ! grep -q GPU_ALL_DONE ../e5_rescope/logs/gpu_all_qwen25vl.out 2>/dev/null; do sleep 30; done
for try in 1 2; do
  $GQ run --mem 14500 --prio 2 --name e8_synth -- $PY code/synth_gpu.py >> logs/synth_gpu.out 2>&1 && break
done
echo CHAIN_DONE
