"""Check for part c' (CPU, from the captured all-position vectors): the literal two-block rule (no guards) of ReScope-W on rows that
are exchangeable with the calibration queries -- the valid REPAIR rows whose QUERY is not in the calibration draw -- next to
heldout_disjoint, over the 20 draws.  Written because the mean false firing of c' on heldout_disjoint can exceed the bound on one
fixed test pool; nothing is selected with it.  -> out/parts/cw_check_<log>.json"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402
from rescope import calibrate_two_block  # noqa: E402

for name in e2c.LOGS:
    p = os.path.join(e2c.OUT, "capture", f"{name}.pt")
    if not os.path.exists(p):
        continue
    L = fx.logs.open(name)
    cap = torch.load(p)
    ix = {j: i for i, j in enumerate(cap["rows"])}
    S = fx.methods.base_gate(L, "ncoord", "all", "all", "lin", True, True)[0]
    rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
    per = []
    for d in range(fx.draws.N_DRAWS):
        dr = fx.draws.calibration(L, 200, d)
        S2, rec = calibrate_two_block(S, coords=cap["Rmax"][[ix[j] for j in dr.rows]], alpha=0.05, split=(1, 1), seed=int(dr.seed), rank="excess", allow_silence=True)
        th = S2.theta()
        fire = lambda rows: 100.0 * float((cap["Rmax"][[ix[j] for j in rows]] >= th[None, :]).any(1).double().mean())      # noqa: E731
        q = set(fx.draws.query_ids(L, dr.rows))
        fresh = [j for j in rep if fx.draws.query_ids(L, [j])[0] not in q]
        per.append({"draw": d, "n_fresh_repair": len(fresh), "fired_fresh_repair": fire(fresh), "fired_heldout_disjoint": fire(hd), "bound": 100.0 * rec["bound_false_firing"]})
    out = {"log": name, "rule": "literal two-block rule without guards, alpha 0.05, n 200", "draws": per,
           "mean_fired_fresh_repair": float(np.mean([x["fired_fresh_repair"] for x in per])),
           "mean_fired_heldout_disjoint": float(np.mean([x["fired_heldout_disjoint"] for x in per])), "bound": per[0]["bound"],
           "n_heldout_disjoint": len(hd), "what": "fresh REPAIR rows = valid REPAIR rows whose query is not a calibration query of the draw (exchangeable with the calibration queries)"}
    fx.io.jdump(out, os.path.join(e2c.OUT, "parts", f"cw_check_{name}.json"), indent=1)
    print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in out.items() if k not in ("draws", "what", "rule")})
