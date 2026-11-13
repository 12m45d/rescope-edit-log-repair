"""E10 diagnostic (descriptive; no method is changed and nothing is served): which detectors set the common scale eta of the
two-block calibration of the default method on each reported log.

    python diag_eta.py  ->  out/diag_eta.json
Per log and calibration draw (alpha = 0.05, n = 200, the library's two_block_rule on the same per-query maxima as the row):
eta as calibrated; the detectors that attain the block-2 scores at and above the quantile (share that belong to updates of
exploding norm (audit's list) / to first-token-subject requests); eta recomputed on the SAME draw without the detectors of the
exploding updates, and without those of the first-token-subject requests (arithmetic on the calibration matrix only).
"""
import numpy as np

import xcommon as X
import e10t
from e10t import data as D, gate as G
from rescope.calibrate import two_block_rule
import xmeth as XM

EXPL = sorted(int(x["index"]) for x in X.jload(X.AUDIT + "/out/recon.json")["stream_updates"]["exploding"])
out = {"what": __doc__, "logs": {}}
for lg in X.LOGS:
    L = e10t.open_log(lg)
    S = XM.base_of(L, "rescope")
    T = L.T
    ex = np.array([i in EXPL for i in range(T)])
    ft = np.array([L.solver_pos[i] == 0 for i in range(T)])
    rows = []
    for d in range(X.N_DRAWS):
        dr = D.calibration(L, X.N_CAL, d)
        Q = G.features(L, S, dr.text_ids)["Q"].numpy()
        r = two_block_rule(Q, X.ALPHA, (1, 1), d)
        sc = Q[r["idx2"]] - r["b"][None, :]
        top = sc.argmax(1)
        mx = sc.max(1)
        at = mx >= r["eta"]
        row = {"draw": d, "eta": float(r["eta"]), "n_block2_at_or_above_eta": int(at.sum()),
               "share_attained_by_exploding_pct": 100.0 * float(ex[top[at]].mean()) if at.any() else None,
               "share_attained_by_first_token_pct": 100.0 * float(ft[top[at]].mean()) if at.any() else None,
               "detectors_at_or_above_eta": sorted({int(i) for i in top[at]})[:20]}
        for nm, drop in (("exploding", ex), ("first_token", ft)):
            if drop.any() and not drop.all():
                row["eta_without_" + nm] = float(two_block_rule(Q[:, ~drop], X.ALPHA, (1, 1), d)["eta"])
        rows.append(row)
    o = {"T": T, "n_exploding": int(ex.sum()), "n_first_token": int(ft.sum()), "per_draw": rows, "eta": X.over_draws([x["eta"] for x in rows])}
    for k in ("eta_without_exploding", "eta_without_first_token", "share_attained_by_exploding_pct", "share_attained_by_first_token_pct"):
        v = [x.get(k) for x in rows]
        if any(x is not None for x in v):
            o[k] = X.over_draws(v)
    o["n_draws_eta_at_least_1"] = int(sum(x["eta"] >= 1.0 for x in rows))
    o["n_draws_eta_above_0.5"] = int(sum(x["eta"] > 0.5 for x in rows))
    out["logs"][lg] = o
    print(lg, {k: (round(v["mean"], 3), round(v["max"], 3)) for k, v in o.items() if isinstance(v, dict) and "mean" in v}, o["n_draws_eta_above_0.5"])
    G.clear_cache()
X.jdump(out, X.OUT + "/diag_eta.json")
