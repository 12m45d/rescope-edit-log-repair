"""E2 fix-up, item 4 (amendment of 2026-11-12 02:55 UTC, decision 4): the POST HOC read-out of part c' (code/cw_check.py) over 200
seeded calibration draws, with an interval.  CPU, from the captured all-position vectors (out/capture/<log>.pt); no model run; no
existing file is changed.  -> out/fixup/cw_check_200.json

Read-out (as in cw_check.py; POST HOC: it was added after the result on heldout_disjoint had been seen): the literal two-block rule
of ReScope-W (no guards, alpha 0.05, n 200) fires on a row if any edit's largest normalised coordinate over all positions reaches its
threshold.  Per draw d: thresholds from the calibration draw d (fx.draws.calibration(L, 200, d); split seed d); false firing on
  fresh REPAIR rows   = valid REPAIR rows whose QUERY is not a calibration query of draw d (exchangeable with the calibration queries)
  heldout_disjoint    = the fixed pool of valid held-out rows
and the mean of the per-draw rates over the draws, against the bound of the rule.

INTERVAL (95 %, percentile, 2000 resamples, seed 0) -- bootstrap over query clusters with the draws nested:
  the unit is a query cluster (rows with the same query_id: same kind of prompt, question string and image content).  One resample
  draws as many clusters as the pool has, with replacement, ONCE; the resampled clusters, each with all its rows, are then used under
  EVERY one of the 200 draws: for draw d the rate is (firing rows of the resampled clusters that are fresh under d) / (rows of the
  resampled clusters that are fresh under d), and the statistic is the mean of these 200 rates.  Draws that share queries therefore
  move together.  The thresholds of a draw are those computed from its original calibration sample (the calibration is not repeated
  inside the bootstrap), and the draws themselves are not resampled.  The interval therefore carries the sampling of the test rows and
  the dependence between draws; it does NOT carry the part of the variability of the thresholds that comes from the REPAIR pool
  itself (a cluster left out of a resample still shaped the thresholds of the draws that calibrated on it).  'crossed' (sensitivity,
  not the reported interval): the 200 draws are resampled with replacement as well, independently of the clusters.

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python fixup_cw200.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402
from rescope import calibrate_two_block  # noqa: E402

FIX = os.path.join(e2c.OUT, "fixup")
N_SEEDS, N_STORED, N_BOOT, SEED, ALPHA, N_CAL = 200, 20, 2000, 0, 0.05, 200
DOC = __doc__


def counts(n, n_boot, rng):
    idx = rng.integers(0, n, size=(n_boot, n))
    c = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(c, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return c


def interval(num, den, bound):
    """num, den: (draws, clusters).  -> statistic, nested-draw cluster bootstrap, crossed sensitivity."""
    D, U = num.shape
    stat = 100.0 * float(np.mean(num.sum(1) / den.sum(1)))
    C = counts(U, N_BOOT, np.random.default_rng([SEED, 201]))
    rates = (C @ num.T) / (C @ den.T)                                    # (resamples, draws)
    if not np.isfinite(rates).all():
        raise RuntimeError("a resample without fresh rows")
    b = 100.0 * rates.mean(1)
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    di = np.random.default_rng([SEED, 202]).integers(0, D, size=(N_BOOT, D))
    bc = 100.0 * np.take_along_axis(rates, di, axis=1).mean(1)
    clo, chi = float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))
    return {"mean_over_draws": stat, "ci_lo": lo, "ci_hi": hi, "se": float(b.std(ddof=1)), "n_clusters": int(U), "n_draws": int(D), "n_boot": N_BOOT, "seed": SEED,
            "bound": bound, "interval_contains_the_bound": bool(lo <= bound <= hi), "interval_entirely_above_the_bound": bool(lo > bound),
            "interval_entirely_below_the_bound": bool(hi < bound), "mean_minus_bound": stat - bound,
            "share_of_resamples_above_the_bound": float((b > bound).mean()),
            "crossed_sensitivity": {"ci_lo": clo, "ci_hi": chi, "se": float(bc.std(ddof=1)), "interval_contains_the_bound": bool(clo <= bound <= chi),
                                    "what": "draws resampled with replacement as well (independently of the clusters); not the reported interval"}}


def one_log(name):
    t0 = time.time()
    L = fx.logs.open(name)
    cap = torch.load(os.path.join(e2c.OUT, "capture", f"{name}.pt"))
    ix = {j: i for i, j in enumerate(cap["rows"])}
    S = fx.methods.base_gate(L, "ncoord", "all", "all", "lin", True, True)[0]
    rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
    q_rep, q_hd = np.array(fx.draws.query_ids(L, rep)), np.array(fx.draws.query_ids(L, hd))
    R_rep, R_hd = cap["Rmax"][[ix[j] for j in rep]], cap["Rmax"][[ix[j] for j in hd]]
    _, inv_rep = np.unique(q_rep, return_inverse=True)
    _, inv_hd = np.unique(q_hd, return_inverse=True)
    U_rep, U_hd = int(inv_rep.max()) + 1, int(inv_hd.max()) + 1
    num_r, den_r, num_h = np.zeros((N_SEEDS, U_rep)), np.zeros((N_SEEDS, U_rep)), np.zeros((N_SEEDS, U_hd))
    den_h = np.tile(np.bincount(inv_hd, minlength=U_hd).astype(np.float64), (N_SEEDS, 1))
    per = []
    for d in range(N_SEEDS):
        dr = fx.draws.calibration(L, N_CAL, d)
        S2, rec = calibrate_two_block(S, coords=cap["Rmax"][[ix[j] for j in dr.rows]], alpha=ALPHA, split=(1, 1), seed=int(dr.seed), rank="excess", allow_silence=True)
        th = S2.theta()
        f_rep, f_hd = (R_rep >= th[None, :]).any(1), (R_hd >= th[None, :]).any(1)
        q = set(fx.draws.query_ids(L, dr.rows))
        fresh = np.array([x not in q for x in q_rep])
        ft = torch.as_tensor(fresh)
        per.append({"draw": d, "n_fresh_repair": int(fresh.sum()), "fired_fresh_repair": 100.0 * float(f_rep[ft].double().mean()),
                    "fired_heldout_disjoint": 100.0 * float(f_hd.double().mean()), "bound": 100.0 * rec["bound_false_firing"]})
        fr = f_rep.numpy().astype(np.float64)
        num_r[d] = np.bincount(inv_rep, weights=fr * fresh, minlength=U_rep)
        den_r[d] = np.bincount(inv_rep, weights=fresh.astype(np.float64), minlength=U_rep)
        num_h[d] = np.bincount(inv_hd, weights=f_hd.numpy().astype(np.float64), minlength=U_hd)
    bounds = sorted({round(x["bound"], 12) for x in per})
    bound = per[0]["bound"]
    stored = fx.io.jload(os.path.join(e2c.OUT, "parts", f"cw_check_{name}.json"))
    diffs = []
    for a, b in zip(stored["draws"], per[:N_STORED]):
        diffs.append({"draw": a["draw"], "same_draw": a["draw"] == b["draw"], "n_fresh_equal": a["n_fresh_repair"] == b["n_fresh_repair"],
                      "abs_diff_fresh": abs(a["fired_fresh_repair"] - b["fired_fresh_repair"]), "abs_diff_heldout": abs(a["fired_heldout_disjoint"] - b["fired_heldout_disjoint"]),
                      "abs_diff_bound": abs(a["bound"] - b["bound"])})
    m20f, m20h = float(np.mean([x["fired_fresh_repair"] for x in per[:N_STORED]])), float(np.mean([x["fired_heldout_disjoint"] for x in per[:N_STORED]]))
    repro = {"n_stored_draws": len(stored["draws"]), "all_draw_indices_equal": all(x["same_draw"] for x in diffs), "all_n_fresh_equal": all(x["n_fresh_equal"] for x in diffs),
             "max_abs_diff_fresh_repair": max(x["abs_diff_fresh"] for x in diffs), "max_abs_diff_heldout_disjoint": max(x["abs_diff_heldout"] for x in diffs),
             "max_abs_diff_bound": max(x["abs_diff_bound"] for x in diffs), "mean_first_20_fresh_repair": m20f, "stored_mean_fresh_repair": stored["mean_fired_fresh_repair"],
             "mean_first_20_heldout_disjoint": m20h, "stored_mean_heldout_disjoint": stored["mean_fired_heldout_disjoint"]}
    repro["first_20_draws_reproduce_the_stored_draws"] = bool(len(stored["draws"]) == N_STORED and repro["all_draw_indices_equal"] and repro["all_n_fresh_equal"]
                                                              and max(repro["max_abs_diff_fresh_repair"], repro["max_abs_diff_heldout_disjoint"], repro["max_abs_diff_bound"]) <= 1e-9
                                                              and abs(m20f - stored["mean_fired_fresh_repair"]) <= 1e-9 and abs(m20h - stored["mean_fired_heldout_disjoint"]) <= 1e-9)
    fr = interval(num_r, den_r, bound)
    hh = interval(num_h, den_h, bound)
    vf, vh = np.array([x["fired_fresh_repair"] for x in per]), np.array([x["fired_heldout_disjoint"] for x in per])
    for o, v in ((fr, vf), (hh, vh)):
        o["over_draws"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max()),
                           "share_of_draws_above_the_bound": float((v > bound).mean())}
        o["mean_of_per_draw_rates_equals_statistic"] = bool(abs(float(v.mean()) - o["mean_over_draws"]) <= 1e-9)
    out = {"log": name, "label": "POST HOC read-out", "rule": f"literal two-block rule without guards, alpha {ALPHA}, n {N_CAL}", "seeds": [0, N_SEEDS - 1], "n_draws": N_SEEDS,
           "bound": bound, "bound_is_the_same_for_every_draw": len(bounds) == 1, "n_repair_rows": len(rep), "n_repair_query_clusters": U_rep, "n_heldout_disjoint_rows": len(hd),
           "n_heldout_disjoint_query_clusters": U_hd, "n_fresh_repair_rows_min_max": [min(x["n_fresh_repair"] for x in per), max(x["n_fresh_repair"] for x in per)],
           "fresh_repair": fr, "heldout_disjoint": hh, "reproduction_of_the_stored_20_draws": repro, "draws": per, "seconds": time.time() - t0}
    print(name, f"fresh REPAIR {fr['mean_over_draws']:.2f} [{fr['ci_lo']:.2f}, {fr['ci_hi']:.2f}]  heldout_disjoint {hh['mean_over_draws']:.2f} [{hh['ci_lo']:.2f}, {hh['ci_hi']:.2f}]  bound {bound:.2f}  "
          f"first 20 reproduce: {repro['first_20_draws_reproduce_the_stored_draws']}  [{time.time() - t0:.0f}s]", flush=True)
    return out


def main():
    os.makedirs(FIX, exist_ok=True)
    logs = {name: one_log(name) for name in e2c.LOGS}
    fx.io.jdump({"what": "POST HOC read-out of part c' (ReScope-W with two-block thresholds): false firing of the literal rule on REPAIR rows outside the calibration draw and on "
                         "heldout_disjoint, mean over 200 seeded calibration draws, with a 95 % interval; gate level, from the captured vectors (no model run)",
                 "label": "POST HOC read-out (cw_check.py was written after the result on heldout_disjoint had been seen; PROTOCOL.md, amendment of 2026-11-12 02:55 UTC, decision 4)",
                 "method": DOC, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint(), "logs": logs}, os.path.join(FIX, "cw_check_200.json"), indent=1)


if __name__ == "__main__":
    main()
