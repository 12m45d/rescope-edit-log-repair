"""Check 1 (E5 c): empirical false firing of the two-block rule, n = 200, alpha = 0.05, 200 own draws; REPAIR -> heldout_disjoint (teacher-forced
keys and prompt-only keys), REPAIR -> BEN-T, REPAIR -> MME-T (prompt-only keys on both sides, R25 (5)); the calibration-conditional rank, exactly."""
from vgate import *
import fx                                    # data access only: log, key cache, pools
from scipy import stats
from math import comb

def pac_rank_exact(n2, alpha, delta):
    """smallest k with P[Beta(n2 + 1 - k, k) <= alpha] >= 1 - delta.  FF of the k-th order statistic threshold ~ Beta(n2 + 1 - k, k);
    P[Beta(a, b) <= x] with integer a, b = P[Bin(a + b - 1, x) >= a]  (exact rational arithmetic)."""
    out = {}
    for k in range(1, n2 + 1):
        a, b = n2 + 1 - k, k
        p = sum(Fraction(comb(n2, j)) * alpha ** j * (1 - alpha) ** (n2 - j) for j in range(a, n2 + 1))
        out[k] = p
    ks = [k for k in out if out[k] >= 1 - delta]
    return min(ks), out

def run(name, ndraw=200, n=200):
    L = fx.logs.open(name)
    cap, own_b1 = guard_caps(L)
    rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
    Pt, Mt = fx.pools.get(L, "BEN-T", "decode"), fx.pools.get(L, "MME-T", "decode")
    assert Pt.test_only and Mt.test_only
    R = {"rep_tf": ncoord(L.log, keys_of(L.kc, rep, "tf")), "hd_tf": ncoord(L.log, keys_of(L.kc, hd, "tf")),
         "rep_d": ncoord(L.log, keys_of(L.kc, rep, "decode")), "hd_d": ncoord(L.log, keys_of(L.kc, hd, "decode")),
         "bent": ncoord(L.log, keys_of(L.kc, Pt.rows, "decode")), "mmet": ncoord(L.log, keys_of(Mt.kc, Mt.rows, "decode"))}
    near = np.array([L.near_edit(j) is not None for j in hd])
    ben_far = ~np.asarray(fx.pools.related(L, Pt), dtype=bool)              # the pool's own flags: row unrelated to an edit or probe of the family
    ben_far_rates = []
    pt = np.array([L.index[j].get("ptype") for j in hd])
    cells = {"tf: REPAIR -> heldout_disjoint": ("rep_tf", "hd_tf"), "prompt: REPAIR -> heldout_disjoint": ("rep_d", "hd_d"),
             "prompt: REPAIR -> BEN-T": ("rep_d", "bent"), "prompt: REPAIR -> MME-T": ("rep_d", "mmet")}
    acc = {c: {"lit": [], "grd": [], "ncap": []} for c in cells}
    sub = {s: [] for s in ("near", "far", "image", "text")}
    N = len(rep)
    for d in range(ndraw):
        rng = np.random.default_rng([977, 1, d])               # own seeds
        rows = rng.permutation(N)[:n]; perm = rng.permutation(n)
        for c, (cal, test) in cells.items():
            b, eta, k = two_block(R[cal][rows], perm)
            lit, g, nc = fire(R[test], b, eta, cap)
            acc[c]["lit"].append(100 * lit.mean()); acc[c]["grd"].append(100 * g.mean()); acc[c]["ncap"].append(nc)
            if c.endswith("BEN-T"): ben_far_rates.append(100 * g[ben_far].mean())
            if c.startswith("tf"):
                sub["near"].append(100 * g[near].mean()); sub["far"].append(100 * g[~near].mean())
                sub["image"].append(100 * g[pt == "image"].mean()); sub["text"].append(100 * g[pt == "text"].mean())
    def st(x):
        x = np.asarray(x, float); return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))), "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95)),
                                          "share_above_5": float(100 * (x > 5).mean())}
    out = {"log": name, "n": n, "n_draws": ndraw, "rank": k, "bound_pct": 100 * (100 + 1 - k) / 101, "n_rows": {"repair": N, "heldout_disjoint": len(hd), "BEN-T": len(Pt.rows), "MME-T": len(Mt.rows)},
           "cells": {c: {"literal": st(v["lit"]), "guards": st(v["grd"]), "n_capped_mean": float(np.mean(v["ncap"]))} for c, v in acc.items()},
           "tf_heldout_subsets_guards": {s: st(v) for s, v in sub.items()}, "n_near": int(near.sum()),
           "own_b1_coordinate": {"min": float(own_b1.min()), "max": float(own_b1.max()), "n_below_1": int((own_b1 < 1).sum())}}
    out["BEN-T_far_rows_guards"] = dict(st(ben_far_rates), n_far=int(ben_far.sum()))
    # the experiment's numbers
    E = jload(f"{FINAL}/e5_rescope/out/c/{name}.json")
    def cell(cross, test):
        by = [x for x in E["cross"][cross]["by_n"] if x["n"] == 200][0]
        c = [x for x in by["cells"] if x["rule"] == "two_block" and x["alpha"] == 0.05][0]
        t = c["tests"][test]
        return {"rank": c["rank"], "bound": c["bound"], "n_draws": c["n_draws"], "literal": {k2: t["all"]["lit"][k2] for k2 in ("mean", "se_mean")}, "guards": {k2: t["all"]["grd"][k2] for k2 in ("mean", "se_mean")},
                "subsets_guards": {s: t[s]["grd"]["mean"] for s in ("near", "far", "image", "text") if s in t and "grd" in t[s]}, "n_capped_mean": c["n_capped_mean"]}
    out["experiment"] = {"tf: REPAIR -> heldout_disjoint": cell("repair_to_heldout", "heldout_disjoint"), "prompt: REPAIR -> heldout_disjoint": cell("repair_prompt_keys", "heldout_disjoint"),
                         "prompt: REPAIR -> BEN-T": cell("repair_prompt_keys", "BEN-T"), "prompt: REPAIR -> MME-T": cell("repair_prompt_keys", "MME-T")}
    cmp = {}
    for c in cells:
        for v in ("literal", "guards"):
            m, e = out["cells"][c][v], out["experiment"][c][v]
            z = (m["mean"] - e["mean"]) / max(1e-12, (m["se"] ** 2 + e["se_mean"] ** 2) ** .5)
            cmp[f"{c} [{v}]"] = {"mine": round(m["mean"], 3), "experiment": round(e["mean"], 3), "diff": round(m["mean"] - e["mean"], 3), "z": round(z, 2)}
    ef = out["experiment"]["prompt: REPAIR -> BEN-T"]["subsets_guards"].get("far")
    cmp["prompt: REPAIR -> BEN-T far rows [guards]"] = {"mine": round(out["BEN-T_far_rows_guards"]["mean"], 3), "experiment": None if ef is None else round(ef, 3),
                                                        "diff": None if ef is None else round(out["BEN-T_far_rows_guards"]["mean"] - ef, 3)}
    out["comparison"] = cmp
    return out

if __name__ == "__main__":
    res = {"logs": {}}
    k, tab = pac_rank_exact(100, Fraction(1, 20), Fraction(1, 10))
    kb = min(kk for kk in range(1, 101) if stats.beta.cdf(0.05, 101 - kk, kk) >= 0.9)
    res["pac_rank"] = {"n2": 100, "alpha": 0.05, "delta": 0.1, "rank_exact_rational": k, "rank_scipy_beta": kb,
                       "P[FF<=alpha] at ranks 96..100": {str(kk): float(tab[kk]) for kk in range(96, 101)}, "two_block_rank": conformal_rank(100, Fraction(1, 20))}
    for name in (QDEV, LDEV):
        res["logs"][name] = run(name)
        jdump(res, OUT + "/c1_validity.json")
        print(name, json.dumps(res["logs"][name]["comparison"], indent=0))
    print(res["pac_rank"])
    for name in (QDEV, LDEV):
        E = jload(f"{FINAL}/e5_rescope/out/c/{name}.json")
        by = [x for x in E["cross"]["repair_to_heldout"]["by_n"] if x["n"] == 200][0]
        res["logs"][name]["experiment_pac_rank"] = [x["rank"] for x in by["cells"] if x["rule"] == "pac" and x["alpha"] == 0.05][0]
    jdump(res, OUT + "/c1_validity.json")
    print({n: res["logs"][n]["experiment_pac_rank"] for n in res["logs"]})
