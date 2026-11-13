"""Extra check X4 (CPU): parts of the calibration / cached-key code that the final experiments use and checks 1 - 5 do not reach.

  python x4_more_rules.py        -> out/x4_more_rules.json

  A  features 'coord' (raw solver coordinate) and 'negdist' (minus the Euclidean distance to the stored key; conformal radius of the
     BALL baseline): library decisions against my naive implementation, both rules, capped and uncapped, seeds 0 - 2, alpha 1/20
  B  several keys per query: KeyCache.decide / coords with which = 'all' (teacher-forced key, the key at the decode's token where
     it differs, the 11 keys of a decode) for the default method calibrated on those keys: fired == OR over my per-key decisions
  C  calibrate_dense (the conformal rule of the router / z-score baselines: one scalar score per query), shared and two-block form,
     on scores with many ties: threshold decisions == the strict defining comparison, for every test score
  D  block ratios 1:3 and 3:1 and the option rank='r' (ablations of E6): library against naive
"""
from __future__ import annotations

import sys
import types

import numpy as np

import vcommon as V

sys.path.insert(0, V.LIB)
import torch  # noqa: E402

torch.set_num_threads(4)
from rescope import (KeyCache, add_guards, calibrate_dense, calibrate_shared, calibrate_two_block, feature_gate, from_testbed, splits)  # noqa: E402


def feat_coord(keys, W, K):
    return np.asarray(keys, dtype=np.float64) @ W.T


def feat_negdist(keys, W, K):
    k = np.asarray(keys, dtype=np.float64)
    out = np.empty((k.shape[0], K.shape[0]))
    for a in range(0, k.shape[0], 64):
        out[a:a + 64] = -np.linalg.norm(k[a:a + 64, None, :] - K[None, :, :], axis=2)
    return out


FEAT = dict(V.FEATURES, coord=feat_coord, negdist=feat_negdist)


def classify(F, b, eta, cap, anyd, cand):
    M = F - b[None, :]
    dist = np.abs(M - eta) if np.isfinite(eta) else np.full(M.shape, np.inf)
    if cap is not None:
        dist = np.minimum(dist, np.abs(F - cap[None, :]))
    scale = np.maximum(1.0, np.abs(F).max(1))
    srt = np.sort(np.where(cand, M, -np.inf), axis=1)
    with np.errstate(invalid="ignore"):
        gap = np.where(cand.sum(1) >= 2, srt[:, -1] - srt[:, -2], np.inf)
    tie = anyd & ((dist.min(1) <= 1e-9 * scale) | (gap <= 1e-9 * scale))
    return int(tie.sum()), int((anyd & ~tie).sum())


def main():
    res = {}
    for tag, d in V.DEV.items():
        name, bb, fam = d["log"], d["backbone"], d["family"]
        idx = V.read_index(bb)
        Ktf, Kst = V.read_keys(bb, "tf"), V.read_keys(bb, "decode_steps")
        log = V.read_log(name)
        T, uids, W, K = log["T"], log["uids"], log["W"], log["K"]
        pos = {u: i for i, u in enumerate(uids)}
        in_log = lambda e: e["family"] == fam and (e.get("edit") is None or (e["edit"] < T and (e.get("edit_b") is None or e["edit_b"] < T)))  # noqa: E731
        rows = [j for j, e in enumerate(idx) if in_log(e) and e.get("row_tf") is not None]
        prow = [j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "probe"]
        ann = V.annotate([idx[j] for j in prow])
        repair = [j for j, a in zip(prow, ann) if a["pool"] == "repair" and idx[j]["y0_valid"]]
        relrows = sorted([j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "rel" and e["edit"] < T], key=lambda j: idx[j]["edit"])
        key = lambda js: Ktf[torch.tensor([idx[j]["row_tf"] for j in js])].double().numpy()      # noqa: E731
        X, Xb1 = key(rows), key(relrows)
        posr = {j: i for i, j in enumerate(rows)}
        own = np.array([pos.get(idx[j].get("edit_uid"), -1) for j in rows])
        gedit = np.concatenate([np.arange(T), np.arange(T)])
        kc = KeyCache(bb)
        L = from_testbed(name)
        R = res[tag] = {"log": name}
        # ------------------------------------------------------------------ A + D
        tot = {"n_decisions": 0, "n_differing": 0, "n_tie": 0, "n_real": 0, "n_configs": 0, "max_abs_eta_diff": 0.0, "n_capped_sets_differing": 0}
        cfgs = [(f, r, (1, 1), "excess" if r == "two_block" else "r") for f in ("coord", "negdist") for r in ("two_block", "shared")] + \
            [("ncoord", "two_block", (1, 3), "excess"), ("ncoord", "two_block", (3, 1), "excess"), ("ncoord", "two_block", (1, 1), "r"), ("cos", "two_block", (1, 3), "excess")]
        for feature, rule, split, rank in cfgs:
            F = FEAT[feature](X, W, K)
            Gm = np.concatenate([FEAT[feature](K, W, K), FEAT[feature](Xb1, W, K)])
            Gg, _ = add_guards(feature_gate(L, feature), *kc.own_keys(uids))
            for seed in (0, 1, 2):
                cal = [repair[i] for i in V.draw(len(repair), 200, seed)]
                Q = F[[posr[j] for j in cal]]
                if rule == "two_block":
                    i1, i2 = V.halves(200, seed, split)
                    b, eta, k = V.naive_two_block(Q, i1, i2, "1/20")
                else:
                    b, eta, k = V.naive_shared(Q, "1/20")
                cap, _ = V.naive_caps(Gm, gedit, b, eta)
                ben = splits.sample(repair, 200, seed)
                coords = kc.coords(Gg, ben)
                for capped in (True, False):
                    if rule == "two_block":
                        S, rec = calibrate_two_block(Gg, coords=coords, alpha=0.05, seed=seed, split=split, rank=rank, allow_silence=not capped)
                    else:
                        S, rec = calibrate_shared(Gg, coords=coords, alpha=0.05, allow_silence=not capped)
                    dec = kc.decide(S, "tf", rows=rows, uids=uids)
                    M = F - b[None, :]
                    cand = M > eta
                    if capped:
                        cand = cand | (F >= cap[None, :])
                    fired = cand.any(1)
                    rk = F if rank == "r" else M                         # rank 'r': the candidate with the largest feature releases
                    top = np.where(fired, np.where(cand, rk, -np.inf).argmax(1), -1)
                    has = own >= 0
                    oc = np.zeros(len(rows), dtype=bool)
                    oc[has] = cand[has, own[has]]
                    ot = fired & has & (top == own)
                    df = [fired != dec["fired"], top != dec["top_edit"], oc != dec["own_cand"], ot != dec["own_top"]]
                    anyd = df[0] | df[1] | df[2] | df[3]
                    nt, nr = classify(F, b, eta, cap if capped else None, anyd, cand)
                    tot["n_decisions"] += 4 * len(rows)
                    tot["n_differing"] += int(sum(x.sum() for x in df))
                    tot["n_tie"] += nt
                    tot["n_real"] += nr
                    if capped:
                        tot["n_capped_sets_differing"] += int(sorted(int(i) for i in np.nonzero(np.isfinite(cap))[0]) != rec["capped_edits"])
                        tot["max_abs_eta_diff"] = max(tot["max_abs_eta_diff"], abs(rec["eta"] - eta) if np.isfinite(eta) else 0.0)
                tot["n_configs"] += 1
        R["A_D_features_ratios_rank"] = dict(tot, configs=[f"{f}/{r}/split {s[0]}:{s[1]}/rank {k}" for f, r, s, k in cfgs])
        # ------------------------------------------------------------------ B: several keys per query (which = 'all')
        rows_all = [j for j in rows if idx[j].get("rows_decode_steps") is not None and idx[j].get("lp_shift") is not None]
        Gg, _ = add_guards(feature_gate(L, "ncoord"), *kc.own_keys(uids))
        ben = splits.sample(repair, 200, 0)
        S, rec = calibrate_two_block(Gg, coords=kc.coords(Gg, ben, "all"), alpha=0.05, seed=0)
        dec = kc.decide(S, "all", rows=rows_all, uids=uids)

        def keys_all(j):
            e = idx[j]
            ks = [Ktf[e["row_tf"]]] + ([Ktf[e["row_tf_prompt"]]] if e["lp_shift"] else []) + [Kst[r] for r in e["rows_decode_steps"]]
            return torch.stack(ks).double().numpy()
        cal = [repair[i] for i in V.draw(len(repair), 200, 0)]
        Qa = np.stack([V.feat_ncoord(keys_all(j), W, K).max(0) for j in cal])              # per query: the largest coordinate over its keys
        i1, i2 = V.halves(200, 0)
        b, eta, k = V.naive_two_block(Qa, i1, i2, "1/20")
        Gm = np.concatenate([V.feat_ncoord(K, W, K), V.feat_ncoord(Xb1, W, K)])
        cap, _ = V.naive_caps(Gm, gedit, b, eta)
        mine = np.array([bool(V.naive_decide(V.feat_ncoord(keys_all(j), W, K), b, eta, cap)[0].any()) for j in rows_all])
        tfonly = np.array([bool(V.naive_decide(V.feat_ncoord(keys_all(j)[:1], W, K), b, eta, cap)[0].any()) for j in rows_all])
        R["B_which_all"] = {"n_rows": len(rows_all), "n_differing": int((mine != dec["fired"]).sum()), "eta_abs_diff": abs(eta - rec["eta"]), "n_fired": int(mine.sum()),
                            "n_fired_on_the_teacher_forced_key_alone": int(tfonly.sum()), "n_near_lib": int(dec["near"].sum()),
                            "capped_equal": sorted(int(i) for i in np.nonzero(np.isfinite(cap))[0]) == rec["capped_edits"]}
        V.say(f"{name}: A/D {tot}; B {R['B_which_all']}")
    # ---------------------------------------------------------------------- C: calibrate_dense on synthetic scores with ties
    rng = np.random.default_rng(20261111)
    C = {"n_cases": 0, "n_decisions": 0, "n_differing": 0, "n_threshold_not_succ_eta": 0}
    for case in range(400):
        n = int(rng.choice([20, 39, 40, 100, 199, 200]))
        alpha = str(rng.choice(["1/100", "1/20", "1/10", "1/5", "1/3"]))
        sc = np.round(rng.gamma(2.0, 1.0, n), int(rng.choice([0, 1, 6])))            # many exact ties at 0 / 1 decimals
        test = np.concatenate([np.round(rng.gamma(2.0, 1.0, 500), 1), sc, np.nextafter(sc, np.inf), np.nextafter(sc, -np.inf)])
        g = types.SimpleNamespace(thr=0.0, kind="router", name="x")
        for two in (False, True):
            seed = int(rng.integers(0, 1000))
            new, rec = calibrate_dense(g, sc, alpha, two_block=two, seed=seed)
            if two:
                i1, i2 = V.halves(n, seed)
                b = sc[i1].max()
                s2 = np.sort(sc[i2] - b)
                k = V.rank_exact(len(i2), alpha)
                eta = np.inf if k > len(i2) else s2[k - 1]
                want = (test - b) > eta
            else:
                k = V.rank_exact(n, alpha)
                eta = np.inf if k > n else np.sort(sc)[k - 1]
                want = test > eta
                C["n_threshold_not_succ_eta"] += int(np.isfinite(eta) and new.thr != np.nextafter(eta, np.inf))
            got = test >= new.thr
            C["n_cases"] += 1
            C["n_decisions"] += int(test.size)
            C["n_differing"] += int((got != want).sum())
    res["C_calibrate_dense"] = C
    V.say(f"C calibrate_dense: {C}")
    res["pass"] = all(res[t]["A_D_features_ratios_rank"]["n_real"] == 0 and res[t]["A_D_features_ratios_rank"]["n_capped_sets_differing"] == 0 and res[t]["B_which_all"]["n_differing"] == 0
                      for t in V.DEV) and C["n_differing"] == 0 and C["n_threshold_not_succ_eta"] == 0
    V.jdump(res, f"{V.OUT}/x4_more_rules.json")
    V.say(f"PASS {res['pass']}")


if __name__ == "__main__":
    main()
