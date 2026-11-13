#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Description of the weights that exist early (pre-edit, dev100, T100, T500): first the decode / teacher-forcing /
# public-probability parts, then the generation part.  Waits for the pre-edit continuations (pre_cont.py).
TP=${RESCOPE_ROOT}/02_research/experiments/09_testbed_scale/text_public
GQ=${RESCOPE_ROOT}/02_research/tools/gpuq/gq
PY=${RESCOPE_PYTHON}
cd $TP || exit 1
while [ ! -f $TP/out/pre_continuations.pt ]; do sleep 20; done
$GQ run --mem 13000 --prio -8 --name tx_descA -- $PY $TP/code/describe.py --tags W0 dev100 T100 T500 --parts dec,edit,public \
  && $GQ run --mem 13000 --prio -8 --name tx_descA_gen -- $PY $TP/code/describe.py --tags dev100 T100 T500 W0 --parts gen
