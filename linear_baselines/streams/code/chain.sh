#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# usage: chain.sh <gpu> <need_MiB> <log> [cap_gb [need_MiB_d2 cap_gb_d2]]
# All the work of one log on its ASSIGNED card, every start behind gpu_wait.sh (other agents have priority):
#   1. seed 0 of the protocol's stream configs at n in {1, 10, 100, ALL} (the summary table)    (run_streams.py --phase tab)
#   2. D2 decodes, core set (end points + the six distinct methods at n in {10, 100, ALL})    (run_d2.py --set core)
#   2b. the one exploratory COMP_BR config (outside the protocol)                              (run_streams.py --phase br)
#   2c. the rest of seed 0 (n in {2, 5, 20, 50})                                               (run_streams.py --phase s0)
#   3. seeds 1 and 2 of the protocol (n <= 20)                                               (run_streams.py --phase rest)
#   4. D2 decodes, the rest (COMP_BR, COMP-B-ALLPOS n = 10, sequential COMP variants)         (run_d2.py --set all)
#   5. second priority, ONLY when the protocol curves of BOTH logs are complete: the text head-to-head;
#      while the other log is still running, the exploratory REEDIT_SC configs are done first
# Every unit is resumable.  A worker that could not start, or left the card for another agent's job (exit 75), is
# queued again; a clean pass without progress ends a stream unit (what is left cannot run: .oom.json / .failed.json).
cd "$(dirname "$0")"
PY=${RESCOPE_PYTHON}
g=$1; need=$2; log=$3; cap=${4:+--cap_gb $4}
# the D2 decodes need less memory than the stream evaluations (no batched Loc pass): optional smaller window
need_d2=${5:-$need}; cap_d2=${6:+--cap_gb $6}; cap_d2=${cap_d2:-$cap}
ndone() { ls ../out/$log/*.json 2>/dev/null | grep -v '/_' | grep -vc '\.\(oom\|failed\)\.json$'; }
say() { echo "[chain $(date +%T)] $log: $*"; }
streams() {   # $1 = phase
  local ph=$1 tries=0 hard=0 a b rc
  until $PY status.py --log $log --quiet --phase $ph; do
    tries=$((tries + 1))
    if [ $tries -gt 300 ]; then say "streams $ph: giving up after $tries starts"; break; fi
    b=$(ndone)
    ./gpu_wait.sh $g $need $PY run_streams.py --log $log $cap --phase $ph
    rc=$?
    a=$(ndone)
    say "streams $ph pass $tries: exit $rc, records $b -> $a"
    ./finalize.sh > /dev/null 2>&1          # CPU only: results.json, results.md, figures follow the records
    if [ $rc -eq 0 ] && [ "$a" -eq "$b" ]; then say "streams $ph: clean pass without progress, stopping"; break; fi
    # exit 75 = left the card for another agent / could not start: normal, queued again.  Any other failure
    # without a new record counts; three in a row end the unit (a bug must not reload the model for hours).
    if [ $rc -ne 0 ] && [ $rc -ne 75 ] && [ "$a" -eq "$b" ]; then hard=$((hard + 1)); else hard=0; fi
    if [ $hard -ge 3 ]; then say "streams $ph: 3 failed starts in a row without progress (exit $rc), stopping this unit"; break; fi
    [ $rc -ne 0 ] && sleep 60
  done
  $PY status.py --log $log --phase $ph
}
retry() {     # $1 = unit name, $2 = need_MiB, $3.. = command
  local name=$1 nd=$2 t rc hard=0; shift 2
  for t in $(seq 1 300); do
    ./gpu_wait.sh $g $nd "$@"
    rc=$?
    say "$name pass $t: exit $rc"
    ./finalize.sh > /dev/null 2>&1
    [ $rc -eq 0 ] && return 0
    if [ $rc -ne 75 ]; then hard=$((hard + 1)); else hard=0; fi
    if [ $hard -ge 3 ]; then say "$name: 3 failures in a row (exit $rc), stopping this unit"; return 1; fi
    sleep 60
  done
  return 1
}
streams tab
retry "D2 core" $need_d2 $PY run_d2.py --log $log $cap_d2 --set core
streams br
streams s0
streams rest
retry "D2 rest" $need_d2 $PY run_d2.py --log $log $cap_d2 --set all
if ! $PY status.py --quiet --phase protocol; then streams extra; fi
w=0
until $PY status.py --quiet --phase protocol; do     # the protocol curves of BOTH logs
  w=$((w + 1))
  if [ $w -gt 360 ]; then say "curves of both logs not complete after 12 h of waiting: no text head-to-head"; exit 0; fi
  sleep 120
done
retry "text head-to-head" $need $PY run_text_h2h.py --log $log $cap
streams extra
say "CHAIN DONE"
