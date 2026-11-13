#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Determinism check of the selection: run fc_select.py again into work/<backbone>/select_recheck and compare the files
# with the selection in use (manifest, hard negatives, benign pool, sampled edits).  If they are byte-identical the new
# select_summary.json (which may hold additional counts) replaces the old one; otherwise nothing is touched.
W=${RESCOPE_ROOT}/02_research/experiments/10_final/cache
PY=${RESCOPE_PYTHON}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
rc=0
for bb in qwen25vl llava; do
  d=$W/work/$bb/select_recheck
  rm -rf $d; mkdir -p $d
  FC_SELECT_DIR=$d $PY $W/code/fc_select.py $bb > $d/log.txt 2>&1 || { echo "$bb: selection failed"; tail -5 $d/log.txt; rc=1; continue; }
  same=1
  for f in manifest.jsonl hn.jsonl benign.jsonl hn_edits.json; do
    if ! cmp -s $d/$f $W/work/$bb/select/$f; then same=0; echo "$bb: $f DIFFERS"; fi
  done
  if [ $same -eq 1 ]; then
    cp $d/select_summary.json $W/work/$bb/select/select_summary.json
    echo "$bb: selection reproduced byte for byte (manifest, hn, benign, hn_edits); summary updated"
  else
    rc=1
  fi
done
exit $rc
