"""Extra GPU check: peak resident memory of loading ONE log in a fresh process, lean or with the harness loader.

    $GQ run --mem 10500 -- python tests/extra/load_peak.py TB-Q2000-ae-l2_3 --lean 0 [--mem_gb 9.8]

Writes tests/out/extra_load_peak_<log>_<lean|harness>.json (peak RSS after import, after load, with / without the batch deltas)."""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402

torch.set_num_threads(4)
import run_tests  # noqa: E402
from rescope import evaluate as EV  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--lean", type=int, default=1)
ap.add_argument("--mem_gb", type=float, default=9.8)
a = ap.parse_args()
res = {"test": "extra_load_peak", "code": run_tests.CODE, "name": a.name, "lean": bool(a.lean), "rss_gb_after_import": EV.peak_rss_gb()}
t0 = time.time()
H = EV.load(a.name, device="cuda:0", mem_gb=a.mem_gb, lean=bool(a.lean))
res.update(T=H.T, n_batches=len(H.batches), seconds=time.time() - t0, peak_rss_gb=EV.peak_rss_gb(), checks=dict(H.checks),
           checks_ok=all(v for k, v in H.checks.items() if k != "collapsed_working_point"), gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
           batch_deltas_gb=len(H.batches) * H.m_out * H.d_in * 4 / 2 ** 30, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
res["pass"] = bool(res["checks_ok"] and res["peak_rss_gb"] < 40.0)
run_tests.jdump(res, os.path.join(run_tests.OUT, f"extra_load_peak_{a.name}_{'lean' if a.lean else 'harness'}.json"))
print(res)
