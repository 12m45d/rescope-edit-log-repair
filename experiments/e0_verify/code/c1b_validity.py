"""Check 1b: validity of the calibration rules.  Over 1000 random splits of the cached benign rows into n calibration rows and
test rows, the false-firing rate (share of test rows on which some edit is a candidate) of the UNCAPPED rule must be at most
alpha within 3 standard errors, for n in {40, 200}, alpha in {0.01, 0.05, 0.2}; two-block and shared-scale rule; normalised
coordinate and cosine feature; both development logs.  The capped rule (guards on) is reported next to it.  CPU.

  python c1b_validity.py Q|L benign|probe        -> out/c1b/<tag>_<feature>_<pool>.json   (resumable: finished files are skipped)

Pools of cached benign rows:
  benign   the label-free benign pool of the cache (kind 'benign', 3000 rows per backbone), prompt-only key (every row has it)
  probe    the valid probe rows of the log's family (the protocol's calibration source), teacher-forced key
Two estimates of every cell, on the same splits:
  lib      the library's rules on the library gate's features: rescope.two_block_rule / shared_rule (thresholds tau, a row
           fires iff some feature >= tau: the package's gate), capped: calibrate_two_block / calibrate_shared on the gate with
           guards (first 200 splits);
  naive    my implementation (vcommon.py) on my own features.
A split is a uniformly random permutation of the pool (numpy default_rng([20261111, split])): the first n rows calibrate
(two-block: the first n // 2 are block 1), all other rows are test rows.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import vcommon as V

sys.path.insert(0, V.LIB)
import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import KeyCache, add_guards, calibrate_shared, calibrate_two_block, feature_gate, from_testbed  # noqa: E402

ALPHAS = ("1/100", "1/20", "1/5")
NS = (40, 200)
N_SPLITS = 1000
N_LIB_CAPPED = 200
SEED = 20261111


def main(tag, pool_name):
    d = V.DEV[tag]
    name, bb, fam = d["log"], d["backbone"], d["family"]
    idx = V.read_index(bb)
    log = V.read_log(name)
    T, W, K = log["T"], log["W"], log["K"]
    if pool_name == "benign":
        rows = [j for j, e in enumerate(idx) if e["kind"] == "benign"]
        which, kf = "decode", "row_decode"
    else:
        rows = [j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "probe" and e["y0_valid"]]
        which, kf = "tf", "row_tf"
    X = V.read_keys(bb, which)[torch.tensor([idx[j][kf] for j in rows])].double().numpy()
    relrows = sorted([j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "rel" and e["edit"] < T], key=lambda j: idx[j]["edit"])
    X_b1 = V.read_keys(bb, "tf")[torch.tensor([idx[j]["row_tf"] for j in relrows])].double().numpy()
    gedit = np.concatenate([np.arange(T), np.arange(T)])
    kc = KeyCache(bb)
    L = from_testbed(name)
    N = len(rows)
    for feature in ("ncoord", "cos"):
        path = f"{V.OUT}/c1b/{tag}_{feature}_{pool_name}.json"
        if os.path.exists(path):
            V.say(f"skip {path}")
            continue
        t0 = time.time()
        F_my = V.FEATURES[feature](X, W, K)
        G_my = np.concatenate([V.FEATURES[feature](K, W, K), V.FEATURES[feature](X_b1, W, K)])
        Gg, _ = add_guards(feature_gate(L, feature), *kc.own_keys(L.meta["uids"]))
        F_lib = kc.coords(Gg, rows, which).numpy()
        out = {"log": name, "feature": feature, "pool": pool_name, "key": which, "n_pool": N, "n_splits": N_SPLITS, "n_lib_capped_splits": N_LIB_CAPPED,
               "max_abs_feature_diff_lib_vs_mine": float(np.abs(F_lib - F_my).max()), "cells": []}
        for n in NS:
            i1, i2 = np.arange(n // 2), np.arange(n // 2, n)
            acc = {}
            for s in range(N_SPLITS):
                perm = np.random.default_rng([SEED, s]).permutation(N)
                cal, test = perm[:n], perm[n:]
                Qm, Ql, Tm, Tl = F_my[cal], F_lib[cal], F_my[test], F_lib[test]
                for rule in ("two_block", "shared"):
                    for alpha in ALPHAS:
                        a = acc.setdefault((rule, alpha), {"naive_u": [], "lib_u": [], "naive_c": [], "lib_c": [], "n_cap_naive": [], "n_cap_lib": [],
                                                           "diff_u": 0, "dec_u": 0, "diff_c": 0, "dec_c": 0, "eta_inf": 0, "bound": None, "rank": None, "n2": None})
                        if rule == "two_block":
                            b, eta, k = V.naive_two_block(Qm, i1, i2, alpha)
                            rl = rescope.two_block_rule(Ql, alpha, blocks=(i1, i2))
                        else:
                            b, eta, k = V.naive_shared(Qm, alpha)
                            rl = rescope.shared_rule(Ql, alpha)
                        fu = ((Tm - b[None, :]) > eta).any(1)
                        cap, _ = V.naive_caps(G_my, gedit, b, eta)
                        fc = fu | (Tm >= cap[None, :]).any(1)
                        lu = (Tl >= rl["tau"][None, :]).any(1)
                        a["naive_u"].append(fu.mean())
                        a["naive_c"].append(fc.mean())
                        a["lib_u"].append(lu.mean())
                        a["n_cap_naive"].append(int(np.isfinite(cap).sum()))
                        a["diff_u"] += int((fu != lu).sum())
                        a["dec_u"] += int(fu.size)
                        a["eta_inf"] += int(not np.isfinite(eta))
                        a["bound"], a["rank"], a["n2"] = rl["bound"], rl["rank"], rl["n2"]
                        assert rl["rank"] == k, (rl["rank"], k)
                        if s < N_LIB_CAPPED:
                            kw = dict(coords=torch.from_numpy(Ql), alpha=alpha)
                            S, r = calibrate_two_block(Gg, blocks=(i1, i2), **kw) if rule == "two_block" else calibrate_shared(Gg, **kw)
                            lc = (Tl >= S.det_tau.numpy()[None, :]).any(1)
                            a["lib_c"].append(lc.mean())
                            a["n_cap_lib"].append(r["n_edits_capped"])
                            a["diff_c"] += int((fc != lc).sum())
                            a["dec_c"] += int(fc.size)
                if (s + 1) % 250 == 0:
                    V.say(f"  {name} {feature} {pool_name} n={n}: split {s + 1}/{N_SPLITS} ({time.time() - t0:.0f}s)")
            for (rule, alpha), a in acc.items():
                al = float(V.as_fraction(alpha))

                def stat(v):
                    v = np.asarray(v, dtype=np.float64)
                    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None, "n_splits": int(v.size),
                            "p95": float(np.percentile(v, 95)), "max": float(v.max())}
                c = {"n": n, "rule": rule, "alpha": alpha, "alpha_float": al, "rank": a["rank"], "n2": a["n2"], "bound": a["bound"],
                     "n_eta_infinite": a["eta_inf"], "naive_uncapped": stat(a["naive_u"]), "lib_uncapped": stat(a["lib_u"]),
                     "naive_capped": stat(a["naive_c"]), "lib_capped": stat(a["lib_c"]), "naive_capped_first": stat(a["naive_c"][:N_LIB_CAPPED]),
                     "mean_edits_capped_naive": float(np.mean(a["n_cap_naive"])), "mean_edits_capped_lib": float(np.mean(a["n_cap_lib"])),
                     "capped_counts_equal_first": a["n_cap_naive"][:N_LIB_CAPPED] == a["n_cap_lib"],
                     "uncapped_decisions": a["dec_u"], "uncapped_decisions_differing_lib_vs_naive": a["diff_u"],
                     "capped_decisions": a["dec_c"], "capped_decisions_differing_lib_vs_naive": a["diff_c"]}
                for who in ("naive_uncapped", "lib_uncapped"):
                    m, se = c[who]["mean"], c[who]["se"]
                    c[who]["within_alpha_3se"] = bool(m <= al + 3 * se)
                    c[who]["within_bound_3se"] = bool(m <= a["bound"] + 3 * se)
                    c[who]["excess_over_bound_in_se"] = float((m - a["bound"]) / se) if se and se > 0 else (0.0 if m <= a["bound"] else float("inf"))
                out["cells"].append(c)
        out["seconds"] = time.time() - t0
        V.jdump(out, path)
        V.say(f"wrote {path} ({out['seconds']:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
