#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage: waitq.sh [seconds]   -- wait (CPU only) until one of my chains reports something new, or the timeout;
# then print a compact status of the two cards, the other agents' queues and my processes.
cd "$(dirname "$0")"
T=${1:-540}
PY=${RESCOPE_PYTHON}
sig() { cat ../logs/chain_Q.out ../logs/chain_L.out 2>/dev/null | grep -c "^\[chain \|^\[gpu_wait \|^\[p12\]"; }
s0=$(sig); t0=$(date +%s)
while [ "$(sig)" -eq "$s0" ] && [ $(( $(date +%s) - t0 )) -lt $T ]; do sleep 15; done
echo "$(date +%T) free MiB $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')| other agents' launchers waiting: $($PY gpuq.py list | grep -c waiting) | running neighbours: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) | mine: $(./mine.sh | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
echo "records: Q $(ls ../out/TB-Q100-ae-l2_3/*.json | grep -v '/_' | wc -l), L $(ls ../out/TB-L100-ae-l2_3/*.json | grep -v '/_' | wc -l); d2: Q $(ls ../out/TB-Q100-ae-l2_3/d2/*.json 2>/dev/null | wc -l), L $(ls ../out/TB-L100-ae-l2_3/d2/*.json 2>/dev/null | wc -l)"
if [ "$(sig)" -ne "$s0" ]; then echo "--- new chain lines:"; cat ../logs/chain_Q.out ../logs/chain_L.out | grep "^\[chain \|^\[gpu_wait \|^\[p12\]" | tail -n $(( $(sig) - s0 )) | cut -c1-230; fi
