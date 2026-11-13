#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# CPU: (re)build the registry and the audit from the per-update files, then every rendered output.
#   logs_edit/registry.json, out/log_audit.json     build_logs.py
#   out/provenance.json, out/DEVIATIONS.txt         provenance.py, make_deviations.py
#   out/single_edit_check.{json,txt}                score_single.py   (public summariser + the paper's LaTeX table)
#   out/text_logs.json, out/text_logs_table.txt     make_table.py     (public summariser + this project's measurements)
TP=${RESCOPE_ROOT}/02_research/experiments/09_testbed_scale/text_public
PY=${RESCOPE_PYTHON}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd $TP || exit 1
set -e
$PY code/build_logs.py
$PY code/export_library_format.py > logs/export_library_format.out 2>&1
$PY code/provenance.py > logs/provenance.out 2>&1
$PY code/make_deviations.py > /dev/null
$PY code/score_single.py > logs/score_single.out 2>&1
$PY code/first_token_subjects.py > logs/first_token_subjects.out 2>&1
$PY code/make_table.py
$PY code/make_index.py > /dev/null
$PY code/make_summary.py > /dev/null
