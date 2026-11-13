"""E10 verification (CPU): gate level of the rows the adapter builds itself (ReScope, cosine with the same calibration) on the
development log and on T100, over the 20 calibration draws, against the experiment's gate-level files.

    python v9_devgate.py  ->  out/devgate.json
Why: the experiment's gate-level driver (run/code/run_gate.py) was modified AFTER its run on the development log (whose file is
the basis of the selection of the R3 gate) and BEFORE its runs on the reported logs; no version history exists.  The adapter's own
gate-level evaluation (e10t.gate.evaluate / gate_stats; the experiment's driver does not use it) is run here for both logs: if the
development-log file (earlier driver version) and the T100 file (final version) both agree with it, the edit did not change what
the driver computes for these rows.
"""
import time

import numpy as np

import vcommon as V

import e10t  # noqa: E402
from e10t import gate as G, methods as MT  # noqa: E402

out = {}
for log in ("TX-G2XL-rome-dev100", V.T100):
    L = e10t.open_log(log)
    X = V.jload(V.XOUT + f"/gate/{log}.json")
    o = {}
    for lab in ("rescope", "cosine"):
        t0 = time.time()
        diffs, mine, theirs = [], [], []
        for d in range(20):
            M = MT.rescope(L, alpha=V.ALPHA, n=V.N_CAL, draw=d) if lab == "rescope" else MT.cosine(L, "two_block", alpha=V.ALPHA, n=V.N_CAL, draw=d)
            st = G.gate_stats(L, G.evaluate(L, M, parts=("heldout", "paraphrase", "neighborhood")))
            x = X["rows"][lab]["gon"]["per_draw"][str(d)]
            a = [st["heldout"]["fired"], st["paraphrase"]["own_released"], st["neighborhood"]["fired"]]
            b = [x["heldout"]["fired"], x["paraphrase"]["own_released"], x["neighborhood"]["fired"]]
            mine.append(a)
            theirs.append(b)
            diffs.append(max(abs(p - q) for p, q in zip(a, b)))
            ce = (X["rows"][lab]["gon"]["cal"][str(d)] or {})
            if abs(float(M.rec["cal"]["eta"]) - float(ce.get("eta"))) > 1e-12 or int(M.rec["cal"]["n_edits_capped"]) != int(ce.get("n_edits_capped")):
                diffs[-1] = max(diffs[-1], 1.0)
        o[lab] = {"n_draws": 20, "max_abs_difference_points": float(max(diffs)), "n_draws_with_a_difference": int(sum(x > 1e-9 for x in diffs)),
                  "verifier_mean_false_firing_coverage_neighbourhood": [float(v) for v in np.mean(mine, 0)], "experiment_mean": [float(v) for v in np.mean(theirs, 0)], "seconds": time.time() - t0}
        print(log, lab, o[lab], flush=True)
        G.clear_cache()
    out[log] = o
out["all_equal"] = all(v["n_draws_with_a_difference"] == 0 for o in out.values() for v in o.values())
V.jdump(out, V.OUT + "/devgate.json")
