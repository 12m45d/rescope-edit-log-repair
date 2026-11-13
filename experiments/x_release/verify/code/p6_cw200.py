#!/usr/bin/env python3
"""Part 6: the 200-draw POST HOC read-out of part c' (ReScope-W with two-block thresholds), recomputed from the captured per-row maxima
(e2_curves/out/capture/<log>.pt: Rmax = largest normalised coordinate per row and edit over all positions).
The experiments' library is used for DATA ACCESS only (fx.logs.open: rows of the REPAIR and heldout_disjoint pools, query ids);
the draws, the rule and the bootstrap are my own code:
  draw d      rows = sort(default_rng(d).choice(N, 200, replace=False)) of the REPAIR pool (documented seed scheme: rescope.splits.sample);
              blocks = sort(perm[:100]), sort(perm[100:]) of default_rng(d).permutation(200) (rescope.calibrate.split_blocks)
  rule        b_i = block-1 maximum of edit i; score(row) = max_i (R[row, i] - b_i); eta = 96th smallest block-2 score
              (k = ceil(0.95 * 101)); a row fires if score > eta; bound 5 / 101
  fresh rows  valid REPAIR rows whose query id is not a calibration query of the draw
  interval    bootstrap over query clusters, the 200 draws nested (one resample of clusters used under every draw), 95 % percentile,
              MY resamples (20000, own generator).
Run with the python that has torch (reading the .pt).  -> out/mine_cw200.json"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import math
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import fx  # noqa: E402

OUT = FINAL + "/x_release/verify/out"
K = math.ceil(0.95 * 101)
res = {}
for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
    L = fx.logs.open(name)
    c = torch.load(f"{FINAL}/e2_curves/out/capture/{name}.pt")
    R = c["Rmax"].double().numpy()
    ix = {int(j): i for i, j in enumerate(c["rows"])}
    rep, hd = [int(j) for j in L.pool("repair")], [int(j) for j in L.pool("heldout_disjoint")]
    Rr, Rh = R[[ix[j] for j in rep]], R[[ix[j] for j in hd]]
    qr = np.array([L.index[j]["query_id"] for j in rep])
    qh = np.array([L.index[j]["query_id"] for j in hd])
    N, D = len(rep), 200
    _, invr = np.unique(qr, return_inverse=True)
    _, invh = np.unique(qh, return_inverse=True)
    Ur, Uh = invr.max() + 1, invh.max() + 1
    numr, denr, numh = np.zeros((D, Ur)), np.zeros((D, Ur)), np.zeros((D, Uh))
    denh = np.bincount(invh, minlength=Uh).astype(float)
    per = []
    for d in range(D):
        rows = np.sort(np.random.default_rng(d).choice(N, 200, replace=False))
        perm = np.random.default_rng(d).permutation(200)
        i1, i2 = np.sort(perm[:100]), np.sort(perm[100:])
        b = Rr[rows][i1].max(0)
        eta = np.sort((Rr[rows][i2] - b[None, :]).max(1))[K - 1]
        f = (Rr - b[None, :]).max(1) > eta
        fh = (Rh - b[None, :]).max(1) > eta
        inq = set(qr[rows].tolist())
        fresh = np.array([q not in inq for q in qr.tolist()])
        numr[d] = np.bincount(invr, weights=(f & fresh).astype(float), minlength=Ur)
        denr[d] = np.bincount(invr, weights=fresh.astype(float), minlength=Ur)
        numh[d] = np.bincount(invh, weights=fh.astype(float), minlength=Uh)
        per.append({"draw": d, "n_fresh_repair": int(fresh.sum()), "fired_fresh_repair": 100.0 * float(f[fresh].mean()), "fired_heldout_disjoint": 100.0 * float(fh.mean())})
    rng = np.random.default_rng(31337)
    bs_r, bs_h = [], []
    for _ in range(20000):
        cr = np.bincount(rng.integers(0, Ur, Ur), minlength=Ur).astype(float)
        bs_r.append(100.0 * np.mean((numr @ cr) / (denr @ cr)))
        ch = np.bincount(rng.integers(0, Uh, Uh), minlength=Uh).astype(float)
        bs_h.append(100.0 * np.mean((numh @ ch) / (denh @ ch)))
    res[name] = {"n_repair_rows": N, "n_repair_clusters": int(Ur), "n_heldout_disjoint_rows": len(hd), "n_heldout_clusters": int(Uh), "rank": K, "draws": per,
                 "fresh_repair": {"mean": float(np.mean([p["fired_fresh_repair"] for p in per])), "ci_lo": float(np.percentile(bs_r, 2.5)), "ci_hi": float(np.percentile(bs_r, 97.5)),
                                  "se": float(np.std(bs_r, ddof=1)), "n_boot": 20000},
                 "heldout_disjoint": {"mean": float(np.mean([p["fired_heldout_disjoint"] for p in per])), "ci_lo": float(np.percentile(bs_h, 2.5)), "ci_hi": float(np.percentile(bs_h, 97.5)),
                                      "se": float(np.std(bs_h, ddof=1)), "n_boot": 20000},
                 "source": f"e2_curves/out/capture/{name}.pt (Rmax; rows of the REPAIR and heldout_disjoint pools and query ids through fx.logs.open)"}
    st = json.load(open(FINAL + "/e2_curves/out/fixup/cw_check_200.json"))["logs"][name]
    dmax = max(max(abs(a["fired_fresh_repair"] - b_["fired_fresh_repair"]), abs(a["fired_heldout_disjoint"] - b_["fired_heldout_disjoint"]), abs(a["n_fresh_repair"] - b_["n_fresh_repair"]))
               for a, b_ in zip(per, st["draws"]))
    res[name]["max_abs_difference_to_the_200_stored_per_draw_records"] = float(dmax)
    print(name, N, Ur, len(hd), Uh, "per-draw max diff", dmax)
    for k in ("fresh_repair", "heldout_disjoint"):
        print("  ", k, "mine", {x: round(v, 4) for x, v in res[name][k].items()}, "stored", st[k]["mean_over_draws"], st[k]["ci_lo"], st[k]["ci_hi"], st[k]["se"])
json.dump(res, open(OUT + "/mine_cw200.json", "w"), indent=1)
