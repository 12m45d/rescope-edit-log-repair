#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage: chain_resume.sh <gpu> <log> <deadline_epoch> <need_MiB> <cap_gb> <need_MiB_d2> <cap_gb_d2> [unit ...]
# Resume of 2026-11-10 17:10 UTC (both cards idle at the start; ONE process of this experiment per card: Qwen on
# GPU 0, LLaVA on GPU 1).  The same workers, launcher and caps as chain.sh (every start behind gpu_wait.sh, which
# waits for need + 1024 MiB free and for the other agents' queues; the workers' watchdog leaves the card for another
# agent's queued job), in the order of the resume task:
#   protocol  1. seed 0 completions (missing n values; corrected REEDIT runs), then 2. seeds 1 and 2 (n <= 20):
#             run_streams.py --phase protocol (p12_common.plan order: seed 0, seed 1, seed 2), one process
#   d2all     3. D2 decodes, in the order of run_d2.d2_plan("all"): the core set first (end points + COMP-B, COMP-B+,
#             REEDIT-B, RB-TOT, RB-UNION at n in {ALL, 100, 10}, seed 0), then REEDIT-S, COMP-B-ALLPOS n = 10 and the
#             sequential COMP variants.  Its first step captures the held-out answer keys: with finalize.sh that gives
#             5., the module-level diagnostics of rho_cpu.py / protect_split_cpu.py, on the CPU.
#   d2core    (the core set alone)
#   h2h       4. the text head-to-head (run_text_h2h.py, 30 text complaints)
# Default units: protocol d2all h2h = the order of the task.  (The chains launched at 17:17:34 ran with the order
# "protocol d2core h2h d2all", i.e. the head-to-head before the non-core D2 decodes: my deviation from the task's
# order.  Corrected at 19:38, before any head-to-head pass: those two chain shells were stopped, their workers were
# left running, and then_chain.sh started this script with "d2all h2h" / "protocol d2all h2h" when each worker ended.)
# The exploratory configs outside the protocol (COMP_BR, REEDIT_SC) are not part of the resume.
# No pass is started after <deadline_epoch>; every unit is resumable.
# At most 4 BLAS threads per process (exported here; p12_common.setup_process reads $P12_THREADS).
cd "$(dirname "$0")"
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 P12_THREADS=4
g=$1; log=$2; deadline=$3; need=$4; cap="--cap_gb $5"; need_d2=$6; cap_d2="--cap_gb $7"; shift 7
units="${*:-protocol d2all h2h}"
ndone() { ls ../out/$log/*.json 2>/dev/null | grep -v '/_' | grep -vc '\.\(oom\|failed\)\.json$'; }
say() { echo "[chain $(date +%T)] $log: $*"; }
late() { [ "$(date +%s)" -ge "$deadline" ]; }
fin() { ./finalize.sh >> ../logs/finalize.log 2>&1; }          # CPU only: results.json, results.md, figures
streams() {   # $1 = phase
  local ph=$1 tries=0 hard=0 a b rc
  until $PY status.py --log $log --quiet --phase $ph; do
    if late; then say "streams $ph: deadline reached, not starting another pass"; break; fi
    tries=$((tries + 1))
    if [ $tries -gt 300 ]; then say "streams $ph: giving up after $tries starts"; break; fi
    b=$(ndone)
    ./gpu_wait.sh $g $need $PY run_streams.py --log $log $cap --phase $ph
    rc=$?
    a=$(ndone)
    say "streams $ph pass $tries: exit $rc, records $b -> $a"
    fin
    if [ $rc -eq 0 ] && [ "$a" -eq "$b" ]; then say "streams $ph: clean pass without progress, stopping"; break; fi
    # exit 75 = left the card for another agent / could not start: normal, queued again.  Any other failure
    # without a new record counts; three in a row end the unit (a bug must not reload the model for hours).
    if [ $rc -ne 0 ] && [ $rc -ne 75 ] && [ "$a" -eq "$b" ]; then hard=$((hard + 1)); else hard=0; fi
    if [ $hard -ge 3 ]; then say "streams $ph: 3 failed starts in a row without progress (exit $rc), stopping this unit"; break; fi
    [ $rc -ne 0 ] && sleep 60
  done
  $PY status.py --log $log --phase $ph 2>&1 | grep -v Warning | cut -c1-400
}
retry() {     # $1 = unit name, $2 = need_MiB, $3.. = command
  local name=$1 nd=$2 t rc hard=0; shift 2
  for t in $(seq 1 300); do
    if late; then say "$name: deadline reached, not starting another pass"; return 1; fi
    ./gpu_wait.sh $g $nd "$@"
    rc=$?
    say "$name pass $t: exit $rc"
    fin
    [ $rc -eq 0 ] && return 0
    if [ $rc -ne 75 ]; then hard=$((hard + 1)); else hard=0; fi
    if [ $hard -ge 3 ]; then say "$name: 3 failures in a row (exit $rc), stopping this unit"; return 1; fi
    sleep 60
  done
  return 1
}
say "RESUME CHAIN START gpu $g, units: $units, deadline $(date -d @$deadline +%T), need $need MiB $cap, D2 need $need_d2 MiB $cap_d2, threads $P12_THREADS"
for u in $units; do
  if late; then say "deadline reached before unit $u"; break; fi
  say "unit $u"
  case $u in
    protocol) streams protocol ;;
    s0)       streams s0 ;;
    rest)     streams rest ;;
    d2core)   retry "D2 core" $need_d2 $PY run_d2.py --log $log $cap_d2 --set core ;;
    d2all)    retry "D2 rest" $need_d2 $PY run_d2.py --log $log $cap_d2 --set all ;;
    h2h)      retry "text head-to-head" $need $PY run_text_h2h.py --log $log $cap ;;
    *)        say "unknown unit $u" ;;
  esac
done
fin
say "RESUME CHAIN DONE"
