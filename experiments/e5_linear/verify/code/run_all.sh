#!/bin/bash
eval "$(python3 "${RESCOPE_HOME:?set RESCOPE_HOME to the clone of the repository}/rescope_paths.py" --sh)"   # locations: the one configuration point
# Independent verification of E5 (linear part): the whole pipeline, in order.  CPU only; at most 8 workers x 2 BLAS threads.
# usage: bash run_all.sh        (about 25 minutes; writes only under ../ = e5_linear/verify)
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=${RESCOPE_PYTHON}
set -e
for lg in TB-Q100-ae-l2_3 TB-L100-ae-l2_3; do
  $PY v_cond_rollback.py --log $lg --workers 8          # tasks (1), (2): every complaint, the seeded 60 flagged
  $PY v_nu.py --log $lg --workers 8                     # extra: vertex cover number
done
$PY v_capacity.py --log TB-Q100-ae-l2_3 > ../logs/v_capacity_TB-Q100-ae-l2_3.out 2>&1 &      # task (3): two processes, 2 threads each
$PY v_capacity.py --log TB-L100-ae-l2_3 > ../logs/v_capacity_TB-L100-ae-l2_3.out 2>&1 &
wait
$PY v_target_rank.py                                    # extra for task (3): rank of the compensation target
$PY v_prefix.py                                         # extra for task (3): how the prompt prefix is shared
$PY v_opt_ub.py                                         # extra: optimum code, upper bounds
$PY v_opt_lb.py --workers 8                             # extra: optimum code, certified lower bounds
$PY v_rerun.py                                          # task (4): sandbox re-run of the measuring agent's aggregation and render
$PY v_compare.py                                        # -> out/verify_results.json, ../verdicts.json
$PY v_render.py                                         # -> out/verify_table.txt, out/final_message.txt (and the message in ../verdicts.json)
