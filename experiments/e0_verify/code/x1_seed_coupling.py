"""Extra check X1 (CPU): the experiments use ONE seed for the calibration draw (numpy default_rng(seed).choice) and for the
two-block split (default_rng(seed).permutation).  Does the shared seed couple the split to the draw?

  python x1_seed_coupling.py        -> out/x1_seed_coupling.json

For seeds 0..5999 and both development pools (489 / 494 valid REPAIR rows, n = 200): which pool rows land in block 1 under the
shared seed, against an independently seeded split.  Statistics per seed: share of block-1 rows that are image rows, mean
pool position of block 1 minus block 2, number of block-1 rows among the 100 smallest drawn pool positions.  Under an
independent uniform split the first has mean = the share of image rows in the draw, the second mean 0, the third mean 50.
Also the end-to-end effect on the default method (naive rule, cached keys): firing rate on HELD-OUT-DISJOINT and T-Gen coverage
with the shared seed against an independent split seed, 2000 draws, paired.
"""
from __future__ import annotations

import numpy as np

import vcommon as V

import torch  # noqa: E402

N_SEEDS, N, ALPHA = 6000, 200, "1/20"


def main():
    res = {}
    for tag, d in V.DEV.items():
        name, bb, fam = d["log"], d["backbone"], d["family"]
        idx = V.read_index(bb)
        prow = [e for e in idx if e["family"] == fam and e["kind"] == "probe"]
        ann = V.annotate(prow)
        repair = [e for e, a in zip(prow, ann) if a["pool"] == "repair" and e["y0_valid"]]
        is_img = np.array([e["ptype"] == "image" for e in repair])
        P = len(repair)
        st = {"shared": {"img": [], "pos": [], "low": []}, "indep": {"img": [], "pos": [], "low": []}}
        for s in range(N_SEEDS):
            cal = V.draw(P, N, s)
            for nm, sd in (("shared", s), ("indep", 10 ** 6 + s)):
                i1, i2 = V.halves(N, sd)
                st[nm]["img"].append(is_img[cal[i1]].mean() - is_img[cal].mean())
                st[nm]["pos"].append(cal[i1].mean() - cal[i2].mean())
                st[nm]["low"].append(int((i1 < N // 2).sum()))
        out = {"n_pool": P, "n_seeds": N_SEEDS}
        for nm, v in st.items():
            out[nm] = {}
            for k, x in v.items():
                x = np.asarray(x, dtype=np.float64)
                mu0 = 50.0 if k == "low" else 0.0
                out[nm][k] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "z_vs_independent_split": float((x.mean() - mu0) / (x.std(ddof=1) / np.sqrt(x.size)))}
        # end-to-end: default method, shared seed vs independent split seed
        log = V.read_log(name)
        T, W, K, uids = log["T"], log["W"], log["K"], log["uids"]
        Ktf = V.read_keys(bb, "tf")
        key = lambda es: Ktf[torch.tensor([e["row_tf"] for e in es])].double().numpy()         # noqa: E731
        F_rep = V.feat_ncoord(key(repair), W, K)
        hd = [e for e, a in zip(prow, ann) if a["disjoint"] and e["y0_valid"]]
        F_hd = V.feat_ncoord(key(hd), W, K)
        tg = sorted([e for e in idx if e["family"] == fam and e["kind"] == "tgen" and e["edit"] < T], key=lambda e: e["edit"])
        F_tg = V.feat_ncoord(key(tg), W, K)
        ar = np.arange(T)
        diff = {"fire": [], "cov": []}
        val = {"shared": {"fire": [], "cov": []}, "indep": {"fire": [], "cov": []}}
        for s in range(2000):
            cal = V.draw(P, N, s)
            r = {}
            for nm, sd in (("shared", s), ("indep", 10 ** 6 + s)):
                i1, i2 = V.halves(N, sd)
                b, eta, _ = V.naive_two_block(F_rep[cal], i1, i2, ALPHA)
                fire = 100.0 * float(((F_hd - b) > eta).any(1).mean())
                M = F_tg - b
                cand = M > eta
                cov = 100.0 * float((cand.any(1) & (np.where(cand, M, -np.inf).argmax(1) == ar)).mean())      # uncapped rule, own edit is top-1
                r[nm] = (fire, cov)
                val[nm]["fire"].append(fire)
                val[nm]["cov"].append(cov)
            diff["fire"].append(r["shared"][0] - r["indep"][0])
            diff["cov"].append(r["shared"][1] - r["indep"][1])
        e2e = {}
        for k in ("fire", "cov"):
            x = np.asarray(diff[k])
            e2e[k] = {"shared_mean": float(np.mean(val["shared"][k])), "indep_mean": float(np.mean(val["indep"][k])), "diff_mean": float(x.mean()),
                      "diff_se": float(x.std(ddof=1) / np.sqrt(x.size)), "z": float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size)))}
        out["default_method_shared_vs_independent_split_seed_2000_draws"] = e2e
        res[tag] = out
        V.say(f"{name}: {out}")
    V.jdump(res, f"{V.OUT}/x1_seed_coupling.json")


if __name__ == "__main__":
    main()
