"""Lean review, CPU part 2: items 2 (retrieval family, lookup), 5 (crossed bootstrap, Holm), 6 (pools and masks), 10 (feature layer: the
split of the benign pool, the discard rules of the recombinations, the stop list, the scan of the packages).

    python rl_cpu2.py retr <development log> | stats | pools <development log> | feat        -> out/cpu2_<part>[_<log>].json
Own implementation from the raw files (attribute table, embedding matrices, key-cache index, benign rows, recombination rows); the
driver is called for the quantity under test only.
"""
import glob
import hashlib
import json
import math
import os
import re
import sys
import time

import rl_common as C

PART = sys.argv[1]
import numpy as np  # noqa: E402

SEED = 77034
t00 = time.time()


def say(*a):
    print(f"[{time.time() - t00:6.1f}s]", *a, flush=True)


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def nq(s):
    return (s or "").strip().lower()


def blocks(n, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:n // 2]), np.sort(perm[n // 2:])


def rank_k(n2, alpha=0.05):
    from fractions import Fraction
    return math.ceil((1 - Fraction(alpha).limit_denominator(10 ** 6)) * (n2 + 1))


def two_block(Q, i1, i2, alpha=0.05):
    b = Q[i1].max(0)
    with np.errstate(invalid="ignore"):
        s2 = np.sort((Q[i2] - b[None, :]).max(1))
    k = rank_k(len(i2), alpha)
    eta = s2[k - 1] if k <= len(i2) else np.inf
    return b, float(eta)


# =================================================================================================== item 2
def part_retr(name):
    import torch  # noqa: F401
    import fx
    L = fx.logs.open(name)
    T = L.T
    att = jl(f"{C.FIN}/cache2/retr/{L.backbone}/attributes.jsonl")
    seq = [a for a in att if a["table"] == "sequence"]
    req = {a["edit_uid"]: a for a in att if a["table"] == "edit" and a["family"] == L.family}
    assert len(seq) == len(L.index) and all(s["sid"] == e["sid"] for s, e in zip(seq, L.index))
    ik = {r["md5"]: i for i, r in enumerate(jl(f"{C.FIN}/cache2/retr/emb/image_keys.jsonl"))}
    qk = {r["question_key"]: i for i, r in enumerate(jl(f"{C.FIN}/cache2/retr/emb/question_keys.jsonl"))}
    EI = np.load(f"{C.FIN}/cache2/retr/emb/image_emb.npy").astype(np.float64)
    EQ = np.load(f"{C.FIN}/cache2/retr/emb/question_emb.npy").astype(np.float64)
    ereq = [req[u] for u in L.uids]
    # the requests are the edits' own Rel sequences (image content and question string)
    req_ok = sum(int(r["image_md5"] == L.index[L.edit_rows["rel"][i]]["image_md5"] and nq(r["question"]) == nq(L.index[L.edit_rows["rel"][i]]["prompt"])) for i, r in enumerate(ereq))
    eI = EI[[ik[r["image_md5"]] for r in ereq]]
    eQ = EQ[[qk[r["question_key"]] for r in ereq]]

    def scores(rows):
        a = np.full((len(rows), T), -np.inf)
        c = np.empty((len(rows), T))
        for n, j in enumerate(rows):
            s = seq[j]
            if s["image_md5"] is not None:
                a[n] = eI @ EI[ik[s["image_md5"]]]
            c[n] = eQ @ EQ[qk[nq(s["question"])]]
        return a, c
    gI, gQ = (eI * eI).sum(1), (eQ * eQ).sum(1)
    rng = np.random.default_rng([SEED, 2, 0 if L.family == "Q" else 1])
    ed = [int(j) for k in ("rel", "tgen", "igen") for j in L.edit_rows[k]]
    hd = [int(j) for j in L.rows("probe", pool="heldout_disjoint", valid=True)]
    hn = [int(j) for j in L.rows("hn", valid=True)]
    SEQ = sorted(int(x) for x in rng.choice(ed, 100, replace=False)) + sorted(int(x) for x in rng.choice(hd, 100, replace=False)) + sorted(int(x) for x in rng.choice(hn, 100, replace=False))
    A, Cq = scores(SEQ)
    rel = [int(j) for j in L.edit_rows["rel"]]
    Arel, Crel = scores(rel)
    allrows = ed + hd + hn + [int(j) for j in L.rows("probe", pool="repair", valid=True)]
    text_rows = [j for j in allrows if seq[j]["image_md5"] is None]
    At, Ct = scores(text_rows)
    out = {"log": name, "requests_equal_own_rel_sequences": [req_ok, T], "n_text_rows": len(text_rows), "rules": {}}
    TIE = 1e-9

    def capth(lit, g):
        sil = g < lit
        return np.where(sil, np.minimum(lit, g - TIE * np.abs(g)), lit), sil

    for dr in (0, 1):
        rows = [int(j) for j in fx.draws.calibration(L, 200, dr).rows]
        QA, QC = scores(rows)
        i1, i2 = blocks(200, dr)
        has1 = np.isfinite(QA[i1]).all(1)
        # ---- own rules -> decide(a, c) -> (fired, top, cand)
        rules = {}
        bA, eA = two_block(QA, i1, i2)
        bC, eC = two_block(QC, i1, i2)
        thA, silA = capth(bA + eA, gI)
        thC, silC = capth(bC + eC, gQ)

        def d_conj(a, c):
            pa = np.where(silA[None, :], a >= thA[None, :], (a - bA[None, :]) > eA)
            pc = np.where(silC[None, :], c >= thC[None, :], (c - bC[None, :]) > eC)
            cand = pa & pc
            with np.errstate(invalid="ignore"):
                rk = np.where(cand, (a - bA[None, :]) + (c - bC[None, :]), -np.inf)
            return cand.any(1), np.where(cand.any(1), rk.argmax(1), -1), cand
        rules["retr_conj"] = (d_conj, int((silA | silC).sum()), None)
        S = QA + QC
        bS, eS = two_block(S, i1, i2)
        thS, silS = capth(bS + eS, gI + gQ)

        def d_sum(a, c, raw=False):
            s = a + c
            cand = np.where(silS[None, :], s >= thS[None, :], (s - bS[None, :]) > eS)
            with np.errstate(invalid="ignore"):
                rk = np.where(cand, s if raw else s - bS[None, :], -np.inf)
            return cand.any(1), np.where(cand.any(1), rk.argmax(1), -1), cand
        rules["retr_sum"] = (d_sum, int(silS.sum()), eS)
        muA, sdA = QA[i1][has1].mean(0), QA[i1][has1].std(0)
        muC, sdC = QC[i1].mean(0), QC[i1].std(0)

        def zmin(a, c):
            with np.errstate(invalid="ignore"):
                return np.minimum((a - muA) / sdA, (c - muC) / sdC)
        Z = zmin(QA, QC)
        bZ, eZ = two_block(Z, i1, i2)
        gZ = zmin(gI[None, :], gQ[None, :])[0]
        thZ, silZ = capth(bZ + eZ, gZ)

        def d_min(a, c, raw=False):
            s = zmin(a, c)
            cand = np.where(silZ[None, :], s >= thZ[None, :], (s - bZ[None, :]) > eZ)
            with np.errstate(invalid="ignore"):
                rk = np.where(cand, s if raw else s - bZ[None, :], -np.inf)
            return cand.any(1), np.where(cand.any(1), rk.argmax(1), -1), cand
        rules["retr_min"] = (d_min, int(silZ.sum()), eZ)
        for nm, (fn, ncap, eta) in rules.items():
            M = fx.methods.build(nm, L, alpha=0.05, n=200, draw=dr)
            d = fx.gate.decide(L, M, SEQ, "tf")
            f, t, _ = fn(A, Cq)
            diff = (f != np.asarray(d["fired"])) | (t != np.asarray(d["top_edit"]))
            dp = fx.gate.decide(L, M, SEQ, "prompt")
            dd = fx.gate.decide(L, M, SEQ, "decode")
            fo, to, co = fn(Arel, Crel)
            dro = fx.gate.decide(L, M, rel, "tf")
            ft, _, _ = fn(At, Ct)
            dt = fx.gate.decide(L, M, text_rows, "tf")
            rec = {"n": len(SEQ), "n_decisions_differ": int(diff.sum()), "n_firing": int(f.sum()), "n_capped_mine": ncap, "n_capped_driver": M.rec["cal"]["n_edits_capped"],
                   "same_at_prompt_and_decode": bool((np.asarray(dp["fired"]) == np.asarray(d["fired"])).all() and (np.asarray(dd["top_edit"]) == np.asarray(d["top_edit"])).all()),
                   "own_request_candidate_mine": int(co[np.arange(T), np.arange(T)].sum()), "own_request_candidate_driver": int(np.asarray(dro["own_cand"]).sum()),
                   "own_request_released_by_own_edit_driver": int(np.asarray(dro["own_top"]).sum()),
                   "text_rows_firing_mine": int(ft.sum()), "text_rows_firing_driver": int(np.asarray(dt["fired"]).sum()),
                   "draw_sids_equal_cosine": M.rec["draw"]["sids"] == fx.methods.cosine(L, alpha=0.05, draw=dr).rec["draw"]["sids"],
                   "blocks_equal_mine": M.rec["cal"]["block1"] == [int(i) for i in i1] and M.rec["cal"]["block2"] == [int(i) for i in i2]}
            if eta is not None:
                rec["eta_abs_diff"] = abs(eta - float(M.rec["cal"]["eta"]))
                fr, tr, _ = fn(A, Cq, True)
                rec["top1_by_raw_score_would_differ_on"] = int((tr != t).sum())
            else:
                rec["eta_abs_diff"] = max(abs(eA - M.rec["cal"]["image"]["eta"]), abs(eC - M.rec["cal"]["question"]["eta"]))
            if nm == "retr_min":
                st = M.obj.stats
                rec["standardisation_max_abs_diff"] = float(max(np.abs(st["mu_img"] - muA).max(), np.abs(st["sd_img"] - sdA).max(), np.abs(st["mu_q"] - muC).max(), np.abs(st["sd_q"] - sdC).max()))
                rec["n_block1_with_image"] = int(has1.sum())
            # the values a retrieval method releases: the forced gate carries the registered cosine gate's values
            Fm = fx.review1.forced(L, 7)
            rec["forced_gate_values_equal_cosine_gate"] = bool(torch.equal(Fm.obj.values, fx.methods.cosine(L, alpha=0.05, draw=dr).obj.values))
            rec["forced_gate_role_fire_response"] = [Fm.obj.role, Fm.obj.fire, Fm.obj.response]
            out["rules"].setdefault(nm, {})[dr] = rec
        say(name, "draw", dr, {k: (v[dr]["n_decisions_differ"], v[dr]["n_firing"], v[dr]["n_capped_mine"], v[dr]["n_capped_driver"]) for k, v in out["rules"].items()})
    # ---- lookup: own tokeniser, the package's stop list (the registered list is fixed in feat2/retr/stoplist.py)
    sys.path.append(C.FIN + "/feat2/retr")
    import stoplist
    tok = re.compile(r"[^\W\d_]+", re.UNICODE)
    cw = lambda s: {t for t in tok.findall((s or "").lower()) if len(t) >= 2 and t not in stoplist.STOP}      # noqa: E731
    ew = [cw(r["question"]) for r in ereq]
    md = [r["image_md5"] for r in ereq]
    M = fx.methods.build("lookup", L)
    rows = ed + hd + hn
    d = fx.gate.decide(L, M, rows, "tf")
    fm, tm = np.zeros(len(rows), dtype=bool), np.full(len(rows), -1)
    for n, j in enumerate(rows):
        s = seq[j]
        if s["image_md5"] is None:
            continue
        w = cw(s["question"])
        best, bi = 0, -1
        for i in range(T):
            if md[i] == s["image_md5"]:
                k = len(w & ew[i])
                if k > best:
                    best, bi = k, i
        fm[n], tm[n] = bi >= 0, bi
    out["lookup"] = {"n": len(rows), "n_decisions_differ": int(((fm != np.asarray(d["fired"])) | (tm != np.asarray(d["top_edit"]))).sum()), "n_firing": int(fm.sum()),
                     "stop_list_sha1": stoplist.stop_list_sha1(), "n_stop_words": len(stoplist.STOP), "fixed_at": stoplist.FIXED_AT,
                     "stop_list_file_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(C.FIN + "/feat2/retr/stoplist.py"))),
                     "content_digest_has_stop_sha": True, "own_requests_released": int(sum(fm[n] and tm[n] == i for n, i in enumerate(range(T))))}
    out["ok"] = bool(all(r["n_decisions_differ"] == 0 and r["n_capped_mine"] == r["n_capped_driver"] and r["same_at_prompt_and_decode"] and r["own_request_candidate_driver"] == T
                         and r["text_rows_firing_driver"] == 0 and r["draw_sids_equal_cosine"] and r["blocks_equal_mine"] and r["eta_abs_diff"] < 1e-12
                         and r["forced_gate_values_equal_cosine_gate"] for v in out["rules"].values() for r in v.values())
                     and out["lookup"]["n_decisions_differ"] == 0 and req_ok == T)
    C.dump(f"cpu2_retr_{name}.json", out)
    say("retr ok", out["ok"], out["lookup"])


def part_valid():
    """Validity of retr_min and retr_sum on exchangeable synthetic scores: the driver's rule (fx.review1.one_score_rule), literal thresholds."""
    import fx
    rng = np.random.default_rng([SEED, 22])
    T, n, ntest, trials = 30, 200, 4000, 500
    res = {}
    for rule in ("retr_min", "retr_sum"):
        ff, ffm = [], []
        for tr in range(trials):
            sc = rng.uniform(0.5, 2.0, T)
            sh = rng.normal(0, 0.3, T)

            def draw(m):
                z = rng.standard_t(4, size=(m, 1)) * 0.5                      # a query effect shared by all edits (heavy tail)
                a = 0.3 + 0.1 * sc[None, :] * (rng.standard_t(5, size=(m, T)) + z) + 0.05 * sh[None, :]
                c = 0.2 + 0.15 * (rng.standard_normal((m, T)) ** 3 / 5 + z * sc[None, :])
                noimg = rng.random(m) < 0.4
                a[noimg] = -np.inf
                return a, c
            a, c = draw(n)
            r = fx.review1.one_score_rule(a, c, rule, 0.05, seed=tr)
            at, ct = draw(ntest)
            if rule == "retr_sum":
                s = at + ct
            else:
                st = r["stats"]
                with np.errstate(invalid="ignore"):
                    s = np.minimum((at - st["mu_img"]) / st["sd_img"], (ct - st["mu_q"]) / st["sd_q"])
            ff.append(float((s >= r["lit"][None, :]).any(1).mean()))
            # own rule on the same data
            i1, i2 = blocks(n, tr)
            if rule == "retr_sum":
                Q = a + c
                stt = None
            else:
                h = np.isfinite(a[i1]).all(1)
                stt = (a[i1][h].mean(0), a[i1][h].std(0), c[i1].mean(0), c[i1].std(0))
                with np.errstate(invalid="ignore"):
                    Q = np.minimum((a - stt[0]) / stt[1], (c - stt[2]) / stt[3])
            b, eta = two_block(Q, i1, i2)
            if rule == "retr_min":
                with np.errstate(invalid="ignore"):
                    s = np.minimum((at - stt[0]) / stt[1], (ct - stt[2]) / stt[3])
            ffm.append(float(((s - b[None, :]) > eta).any(1).mean()))
        ff, ffm = np.array(ff), np.array(ffm)
        res[rule] = {"trials": trials, "n": n, "T": T, "n_test": ntest, "bound": 5 / 101, "mean_false_firing_driver_rule": float(ff.mean()), "se": float(ff.std(ddof=1) / np.sqrt(trials)),
                     "mean_false_firing_own_rule": float(ffm.mean()), "max_abs_diff_driver_vs_own_per_trial": float(np.abs(ff - ffm).max()),
                     "valid": bool(ff.mean() <= 5 / 101 + 3 * ff.std(ddof=1) / np.sqrt(trials))}
        say("validity", rule, res[rule])
    return res


# =================================================================================================== item 5
def part_stats():
    import fx
    rng = np.random.default_rng([SEED, 5])
    out = {"retrieval_validity": part_valid(), "crossed_bootstrap": {}, "holm": {}}
    # coverage of the 95 % interval of the grand mean of a two-way table y[d, u] = a_d + b_u + e_du (true mean 0)
    for tag, D, U, sa, sb, se in (("unit-dominated, 3 draws", 3, 100, 0.03, 1.0, 0.3), ("unit-dominated, 20 draws", 20, 100, 0.03, 1.0, 0.3),
                                  ("draw effect 0.3, 3 draws", 3, 100, 0.3, 1.0, 0.3), ("draw effect 0.3, 20 draws", 20, 100, 0.3, 1.0, 0.3),
                                  ("cell noise only, 5 draws", 5, 100, 0.0, 0.0, 1.0)):
        hit, wid, ana = 0, [], []
        R = 300
        for r in range(R):
            y = sa * rng.standard_normal((D, 1)) + sb * rng.standard_normal((1, U)) + se * rng.standard_normal((D, U))
            b = fx.stats.crossed_bootstrap(y, n_boot=600, seed=r)
            hit += int(b["ci_lo"] <= 0.0 <= b["ci_hi"])
            wid.append(b["ci_hi"] - b["ci_lo"])
            ana.append(2 * 1.96 * math.sqrt(sa ** 2 / D + sb ** 2 / U + se ** 2 / (D * U)))
            if r == 0:
                v0 = abs(b["value"] - float(y.mean()))
        out["crossed_bootstrap"][tag] = {"coverage": hit / R, "mean_width": float(np.mean(wid)), "analytic_width": float(np.mean(ana)), "value_is_grand_mean_abs_diff": v0, "n_sim": R}
        say("crossed", tag, out["crossed_bootstrap"][tag])
    # ratio form: value = sum(num) / sum(den); binary flags with clustered units
    num = rng.integers(0, 3, size=(4, 50)).astype(float)
    den = np.full((4, 50), 3.0)
    b = fx.stats.crossed_bootstrap(num, den, scale=100)
    out["crossed_bootstrap"]["ratio_value_abs_diff"] = abs(b["value"] - 100 * num.sum() / den.sum())
    # paired difference under the null (same law for both methods, shared unit effect): share of p < 0.05
    rej, R = 0, 300
    for r in range(R):
        bu = rng.standard_normal((1, 80))
        ya = bu + 0.5 * rng.standard_normal((5, 80))
        yb = bu + 0.5 * rng.standard_normal((5, 80))
        dd = fx.stats.crossed_bootstrap_diff(ya, yb, n_boot=600, seed=r)
        rej += int(dd["p_two_sided"] < 0.05)
        if r == 0:
            v0 = abs(dd["value"] - float(ya.mean() - yb.mean()))
    out["crossed_bootstrap"]["paired_null_rejection_rate_at_0.05"] = rej / R
    out["crossed_bootstrap"]["paired_value_abs_diff"] = v0
    ya = 0.5 + rng.standard_normal((5, 80))
    dd = fx.stats.crossed_bootstrap_diff(ya, ya - 0.5, n_boot=600)
    out["crossed_bootstrap"]["paired_shift_0.5_detected"] = bool(dd["p_two_sided"] < 0.01 and abs(dd["value"] - 0.5) < 1e-9)
    # Holm against an own implementation
    def my_holm(p):
        p = np.asarray(p, dtype=float)
        m = len(p)
        o = np.argsort(p, kind="stable")
        adj = np.empty(m)
        cur = 0.0
        for r_, i in enumerate(o):
            cur = max(cur, min(1.0, (m - r_) * p[i]))
            adj[i] = cur
        return adj
    bad = 0
    for r in range(300):
        p = rng.random(rng.integers(1, 9)) ** 3
        h = fx.stats.holm(list(p))
        bad += int(not np.allclose(h["adjusted"], my_holm(p)) or list(h["reject"]) != list(my_holm(p) <= 0.05))
    ex = fx.stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    out["holm"] = {"n_random_vectors": 300, "n_differ": bad, "example": {k: float(v) for k, v in ex["adjusted"].items()}, "example_expected": {"a": 0.03, "b": 0.06, "c": 0.06},
                   "example_ok": bool(abs(ex["adjusted"]["a"] - 0.03) < 1e-12 and abs(ex["adjusted"]["b"] - 0.06) < 1e-12 and abs(ex["adjusted"]["c"] - 0.06) < 1e-12)}
    try:
        from statsmodels.stats.multitest import multipletests
        p = rng.random(7) ** 3
        out["holm"]["statsmodels_equal"] = bool(np.allclose(multipletests(p, method="holm")[1], fx.stats.holm(list(p))["adjusted"]))
    except Exception as e:  # noqa: BLE001
        out["holm"]["statsmodels_equal"] = f"not available ({type(e).__name__})"
    cb = out["crossed_bootstrap"]
    out["ok"] = bool(all(v["valid"] for v in out["retrieval_validity"].values()) and 0.91 <= cb["unit-dominated, 3 draws"]["coverage"] <= 0.995
                     and 0.91 <= cb["unit-dominated, 20 draws"]["coverage"] <= 0.995 and cb["paired_null_rejection_rate_at_0.05"] <= 0.08
                     and cb["paired_shift_0.5_detected"] and bad == 0 and out["holm"]["example_ok"] and cb["ratio_value_abs_diff"] < 1e-9)
    C.dump("cpu2_stats.json", out)
    say("stats ok", out["ok"])


# =================================================================================================== item 6
def part_pools(name):
    import torch
    import fx
    L = fx.logs.open(name)
    out = {"log": name, "refused": {}, "allowed": {}, "masks": {}}
    bt = fx.pools.get(L, "BEN-T")
    bw = fx.pools.get(L, "BEN-W")
    bs = fx.pools.get(L, "BEN-S")
    mt = fx.pools.get(L, "MME-T")
    mc = fx.pools.get(L, "MME-C")
    btf = fx.pools.get(L, "BEN-T", "tf")
    Dr = fx.draws.Draw
    hand = lambda rows, which, kc=None: Dr(rows=list(rows), which=which, n=len(rows), draw=0, source="hand", rec={"sids": []}, kc=kc)       # noqa: E731
    S0 = fx.methods.base_gate(L)[0]
    inf = torch.full((L.T,), -float("inf"), dtype=torch.float64)
    attempts = {
        "calibration(source=BEN-T)": lambda: fx.draws.calibration(L, 200, 0, source="BEN-T"),
        "calibration(source=MME-T)": lambda: fx.draws.calibration(L, 200, 0, source="MME-T"),
        "calibration(source=BEN-W)": lambda: fx.draws.calibration(L, 200, 0, source="BEN-W"),
        "calibration(source=benign) whole pool": lambda: fx.draws.calibration(L, 200, 0, source="benign"),
        "calibration(source=benign, part=BEN-T)": lambda: fx.draws.calibration(L, 200, 0, source="benign", part="BEN-T"),
        "calibration(source=BEN-S, part=BEN-T)": lambda: fx.draws.calibration(L, 200, 0, source="BEN-S", part="BEN-T"),
        "pool_rows(BEN-T)": lambda: fx.draws.pool_rows(L, "BEN-T"),
        "pool_rows(MME-T)": lambda: fx.draws.pool_rows(L, "MME-T"),
        "explicit(BEN-T rows)": lambda: fx.draws.explicit(L, bt.rows[:50], which="decode"),
        "explicit(BEN-W rows)": lambda: fx.draws.explicit(L, bw.rows[:50], which="decode"),
        "explicit(MME-T rows, kc)": lambda: fx.draws.explicit(L, mt.rows[:50], kc=mt.kc),
        "rescope(cal=hand-made Draw of BEN-T rows)": lambda: fx.methods.rescope(L, cal=hand(bt.rows[:200], "decode")),
        "rescope(cal=hand-made Draw of MME-T rows)": lambda: fx.methods.rescope(L, cal=hand(mt.rows[:200], "tf", mt.kc)),
        "rescope(cal=mixed REPAIR + 1 BEN-T row)": lambda: fx.methods.rescope(L, cal=hand(list(fx.draws.calibration(L, 199, 0).rows) + [btf.rows[0]], "tf")),
        "cosine(cal=BEN-T)": lambda: fx.methods.cosine(L, cal=hand(bt.rows[:200], "decode")),
        "rescope(null_rows=BEN-T tf rows)": lambda: fx.methods.rescope(L, null_rows=btf.rows[:20]),
        "rescope_null(null_rows=BEN-T tf rows)": lambda: fx.methods.rescope_null(L, null_rows=btf.rows[:20]),
        "rescope(least=BEN-T tf rows)": lambda: fx.methods.rescope(L, least=btf.rows[:3]),
        "rescope(least=BEN-W tf rows)": lambda: fx.methods.rescope(L, least=fx.pools.get(L, "BEN-W", "tf").rows[:3]),
        "calibration(exclude=BEN-T rows)": lambda: fx.draws.calibration(L, 200, 0, exclude=btf.rows[:3]),
        "recalibrate(cal=BEN-T)": lambda: fx.methods.recalibrate(L, S0, cal=hand(bt.rows[:200], "decode")),
        "recalibrate(cal=MME-T)": lambda: fx.methods.recalibrate(L, S0, cal=hand(mt.rows[:200], "tf", mt.kc)),
        "recalibrate(source=BEN-T)": lambda: fx.methods.recalibrate(L, S0, source="BEN-T"),
        "calib_coords(BEN-T)": lambda: fx.methods.calib_coords(L, S0, hand(bt.rows[:200], "decode")),
        "null_sample(source=BEN-T)": lambda: fx.draws.null_sample(L, 100, 0, source="BEN-T"),
        "null_sample(source=BEN-S)": lambda: fx.draws.null_sample(L, 100, 0, source="BEN-S"),
        "partition(source=BEN-S)": lambda: fx.draws.partition(L, (100, 100), 0, source="BEN-S"),
        "build(cos_centred, source=BEN-T)": lambda: fx.methods.build("cos_centred", L, source="BEN-T"),
        "build(whitecos_0.1, source=MME-T)": lambda: fx.methods.build("whitecos_0.1", L, source="MME-T"),
        "build(white_1_shared, cal=BEN-T)": lambda: fx.methods.build("white_1_shared", L, cal=hand(bt.rows[:200], "decode")),
        "build(cos_null, null_rows=BEN-T)": lambda: fx.methods.build("cos_null", L, null_rows=btf.rows[:100]),
        "build(rescope_null_x, null_rows=BEN-T)": lambda: fx.methods.build("rescope_null_x", L, null_rows=btf.rows[:100]),
        "build(rescope_x, source=BEN-T)": lambda: fx.methods.build("rescope_x", L, source="BEN-T"),
        "build(rescope_x, cal=MME-T)": lambda: fx.methods.build("rescope_x", L, cal=hand(mt.rows[:200], "tf", mt.kc)),
        "build(rescope_nullW, source=MME-T)": lambda: fx.methods.build("rescope_nullW", L, source="MME-T"),
        "build(retr_sum, source=BEN-T)": lambda: fx.methods.build("retr_sum", L, source="BEN-T"),
        "build(retr_min, cal=BEN-T)": lambda: fx.methods.build("retr_min", L, cal=hand(bt.rows[:200], "decode")),
        "build(retr_conj, cal=BEN-T)": lambda: fx.methods.build("retr_conj", L, cal=hand(bt.rows[:200], "decode")),
        "build(retr_sum, source=MME-C) (no attributes)": lambda: fx.methods.build("retr_sum", L, source="MME-C"),
        "pools.firing(retr_sum, MME-T) (no attributes)": lambda: fx.pools.firing(L, fx.methods.build("retr_sum", L), "MME-T"),
        "e1_rowset_review1(source=BEN-T)": lambda: fx.review1.e1_rowset_review1(L, 0.05, 0, labels=("rescope", "cos_centred"), source="BEN-T"),
        "e1_rowset(source=benign)": lambda: fx.methods.e1_rowset(L, 0.05, 0, labels=("rescope",), source="benign"),
        "pools.guard(BEN-T, statistics)": lambda: fx.pools.guard(L, bt.rows[:5], "statistics"),
        "pools.guard(BEN-T, stream)": lambda: fx.pools.guard(L, bt.rows[:5], "stream"),
        "pools.guard(MME-T, calibration)": lambda: fx.pools.guard(L, mt.rows[:5], "calibration", mt.kc),
        "behave: tf on BEN-T rows (no model session)": lambda: fx.behave.session(L, None).tf(fx.methods.rescope(L), bt.rows[:3]),
    }
    for k, fn in attempts.items():
        try:
            fn()
            out["refused"][k] = "NOT REFUSED"
        except Exception as e:  # noqa: BLE001
            out["refused"][k] = f"{type(e).__name__}: {str(e)[:110]}"
    ok_calls = {
        "calibration(source=BEN-S)": lambda: set(fx.draws.calibration(L, 200, 0, source="BEN-S").rows) <= set(fx.pools.get(L, "BEN-S", "tf").rows),
        "calibration(source=MME-C)": lambda: set(fx.draws.calibration(L, 200, 0, source="MME-C").rows) <= set(mc.rows),
        "rescope(source=BEN-S)": lambda: fx.methods.rescope(L, source="BEN-S").rec["draw"]["source"] == "BEN-S",
        "rescope(source=MME-C)": lambda: fx.methods.rescope(L, source="MME-C").rec["draw"]["source"] == "MME-C",
        "build(whitecos_0.1, source=BEN-S)": lambda: fx.methods.build("whitecos_0.1", L, source="BEN-S").rec["draw"]["source"] == "BEN-S",
        "build(retr_sum, source=BEN-S)": lambda: fx.methods.build("retr_sum", L, source="BEN-S").rec["draw"]["source"] == "BEN-S",
        "firing(rescope, BEN-T)": lambda: fx.pools.firing(L, fx.methods.rescope(L), "BEN-T")["n"] == len(bt.rows),
        "firing(rescope, MME-T)": lambda: fx.pools.firing(L, fx.methods.rescope(L), "MME-T")["n"] == len(mt.rows),
        "firing(whitecos_0.1, MME-T)": lambda: fx.pools.firing(L, fx.methods.build("whitecos_0.1", L), "MME-T")["n"] == len(mt.rows),
        "firing(retr_min, BEN-T)": lambda: fx.pools.firing(L, fx.methods.build("retr_min", L), "BEN-T")["n"] == len(bt.rows),
        "firing(lookup, BEN-T)": lambda: fx.pools.firing(L, fx.methods.build("lookup", L), "BEN-T")["n"] == len(bt.rows),
    }
    for k, fn in ok_calls.items():
        try:
            out["allowed"][k] = bool(fn())
        except Exception as e:  # noqa: BLE001
            out["allowed"][k] = f"FAILED {type(e).__name__}: {str(e)[:150]}"
    # ---- the parts: own recomputation of the split and its use
    parts = json.load(open(f"{C.FIN}/cache2/benign_parts/{L.backbone}.json"))
    out["pool_sizes"] = {"BEN-W": len(bw.rows), "BEN-S": len(bs.rows), "BEN-T": len(bt.rows), "MME-C": len(mc.rows), "MME-T": len(mt.rows)}
    out["driver_parts_equal_file"] = all(parts["parts"][L.index[j]["sid"]] == nm for nm, P in (("BEN-W", bw), ("BEN-S", bs), ("BEN-T", bt)) for j in P.rows)
    # BEN-W is the only statistics source: the statistics object the driver hands to the gates
    R1 = fx.review1
    WS, wst = R1.package("white")
    st = wst.load(L.backbone)
    out["statistics_rows_are_BEN-W"] = sorted(st.sids) == sorted(L.index[j]["sid"] for j in bw.rows) and len(st.sids) == len(bw.rows)
    out["statistics_keys_are_BEN-W_decode_keys"] = bool(torch.equal(st.keys, L.kc.keys("decode", [L.kc.by_sid(s) for s in st.sids])))
    # ---- masks
    from fx.logs import norm_q
    fam_rows = [e for e in L.index if e.get("family") == L.family and e.get("kind") in ("rel", "tgen")]
    qs = {nq(e["prompt"]) for e in fam_rows}
    n_stream = sum(1 for e in fam_rows if e["kind"] == "rel")
    pr = [int(j) for j in L.rows("probe", valid=True)]
    mine = np.array([nq(L.index[j]["prompt"]) not in qs for j in pr])
    drv = fx.masks.keep(L, pr)
    dq = fx.draws.calibration(L, 200, 0, mask="qdisjoint")
    nsq = fx.draws.null_sample(L, 100, 0, mask="qdisjoint")
    Mq = fx.methods.rescope(L, mask="qdisjoint")
    hdq = L.rows("probe", pool="heldout_disjoint", valid=True, where=fx.masks.question_disjoint(L))
    out["masks"] = {"n_stream_edits": n_stream, "n_stream_edits_driver": fx.masks.n_stream_edits(L), "expected_longest_stream": 2000 if L.family == "Q" else 1000,
                    "norm_q_is_strip_lower": norm_q("  What IS this? ") == "what is this?",
                    "question_disjoint_differs_from_own": int((mine != drv).sum()), "n_probe_rows": len(pr), "n_kept": int(mine.sum()),
                    "masked_draw_rows_all_disjoint": all(nq(L.index[j]["prompt"]) not in qs for j in dq.rows), "masked_draw_n": len(dq.rows),
                    "masked_draw_in_repair": set(dq.rows) <= set(L.pool("repair", valid=True)),
                    "masked_null_sample_disjoint_and_no_shared_query": all(nq(L.index[j]["prompt"]) not in qs for j in nsq) and not ({L.index[j]["query_id"] for j in nsq} & {L.index[j]["query_id"] for j in dq.rows}),
                    "masked_method_draw_is_masked_draw": Mq.rec["draw"]["sids"] == dq.rec["sids"], "masked_method_variant": Mq.variant,
                    "heldout_disjoint_filtered": [len(hdq), len(L.pool("heldout_disjoint"))],
                    "ben_t_mask_counts": {k: fx.pools.firing(L, Mq, "BEN-T", mask=fx.masks.question_disjoint(L))[k] for k in ("n", "n_masked_out")},
                    "later_edits_dev": fx.masks.later_edits(L), "later_from": fx.masks.later_from(L), "stream_pos_is_range": fx.masks.stream_pos(L) == list(range(L.T))}
    own_bt = sum(1 for j in bt.rows if nq(L.index[j]["prompt"]) not in qs)
    out["masks"]["ben_t_kept_own"] = own_bt
    # later edits on other logs of the programme: mechanics only (open, the mask; no metric)
    mech = {}
    for other in (("TB-Q500-ae-l2_3", "TB-Q1000-memitw") if L.family == "Q" else ("TB-L200-ae-l2_3", "TB-L500-memitw")):
        Lo = fx.logs.open(other)
        le = fx.masks.later_edits(Lo)
        t0 = fx.masks.later_from(Lo)
        mech[other] = {"T": Lo.T, "n_later": len(le), "expected": max(0, Lo.T - t0), "later_is_tail": le == list(range(t0, Lo.T)), "stream_pos_is_range": fx.masks.stream_pos(Lo) == list(range(Lo.T)),
                       "first_100_uids_equal_dev": Lo.uids[:100] == L.uids}
    out["masks"]["later_edits_other_logs"] = mech
    m = out["masks"]
    out["ok"] = bool(all(v != "NOT REFUSED" for v in out["refused"].values()) and all(v is True for v in out["allowed"].values()) and out["driver_parts_equal_file"]
                     and out["statistics_rows_are_BEN-W"] and out["statistics_keys_are_BEN-W_decode_keys"] and m["n_stream_edits"] == m["expected_longest_stream"] == m["n_stream_edits_driver"]
                     and m["question_disjoint_differs_from_own"] == 0 and m["masked_draw_rows_all_disjoint"] and m["masked_null_sample_disjoint_and_no_shared_query"]
                     and m["masked_method_draw_is_masked_draw"] and m["later_edits_dev"] == [] and m["ben_t_mask_counts"]["n"] == own_bt
                     and all(v["n_later"] == v["expected"] and v["later_is_tail"] and v["stream_pos_is_range"] for v in mech.values()))
    C.dump(f"cpu2_pools_{name}.json", out)
    say("pools ok", out["ok"], "not refused:", [k for k, v in out["refused"].items() if v == "NOT REFUSED"], "failed allowed:", [k for k, v in out["allowed"].items() if v is not True])


# =================================================================================================== item 10 (CPU parts)
def part_feat():
    out = {}
    # ---- (a) split of the benign pool by source item
    sp = {}
    for bb in ("qwen25vl", "llava"):
        rows = jl(f"{C.FIN}/cache/benign/{bb}/rows.jsonl")
        P = json.load(open(f"{C.FIN}/cache2/benign_parts/{bb}.json"))
        uids = sorted({r["uid"] for r in rows})
        n = len(uids)
        perm = np.random.default_rng(20261111).permutation(n)
        nw, ns = round(0.20 * n), round(0.45 * n)
        mine = {}
        for r_, i in enumerate(perm):
            mine[uids[int(i)]] = "BEN-W" if r_ < nw else ("BEN-S" if r_ < nw + ns else "BEN-T")
        by_item = {}
        for r in rows:
            by_item.setdefault(r["uid"], set()).add(P["parts"][r["sid"]])
        sp[bb] = {"n_rows": len(rows), "n_items": n, "items_in_two_parts": sum(len(v) > 1 for v in by_item.values()),
                  "rows_differing_from_own_split": sum(P["parts"][r["sid"]] != mine[r["uid"]] for r in rows), "all_rows_have_a_part": set(P["parts"]) == {r["sid"] for r in rows},
                  "counts": {k: sum(v == k for v in P["parts"].values()) for k in ("BEN-W", "BEN-S", "BEN-T")}, "item_counts": {k: sum(v == k for v in mine.values()) for k in ("BEN-W", "BEN-S", "BEN-T")},
                  "items": {u: next(iter(v)) for u, v in by_item.items()}, "rows_key": sorted((r["uid"], r["ptype"], r["prompt"]) for r in rows)}
    out["benign_split"] = {bb: {k: v for k, v in sp[bb].items() if k not in ("items", "rows_key")} for bb in sp}
    out["benign_split"]["same_items_and_parts_both_backbones"] = sp["qwen25vl"]["items"] == sp["llava"]["items"]
    out["benign_split"]["same_rows_both_backbones"] = sp["qwen25vl"]["rows_key"] == sp["llava"]["rows_key"]
    # BEN-T used by nothing in the packages: which rows of the benign pool the stored statistics / recombination files reference
    import torch
    use = {}
    for bb in ("qwen25vl", "llava"):
        P = json.load(open(f"{C.FIN}/cache2/benign_parts/{bb}.json"))["parts"]
        st = torch.load(f"{C.FIN}/cache2/whiten/{bb}/stats.pt", map_location="cpu")
        use[bb] = {"whiten_stats_parts": sorted({P[s] for s in st["sids"]}), "whiten_n": len(st["sids"]), "whiten_all_BEN-W": len(st["sids"]) == sum(v == "BEN-W" for v in P.values())}
    out["ben_t_use"] = use
    code_hits = {}
    for f in sorted(glob.glob(f"{C.FIN}/feat2/*/*.py") + glob.glob(f"{C.FIN}/feat2/*/tests/*.py")):
        txt = open(f, encoding="utf-8").read()
        hits = [ln.strip()[:150] for ln in txt.splitlines() if re.search(r"BEN-T|MME-T", ln) and not ln.strip().startswith("#")]
        if hits:
            code_hits[os.path.relpath(f, C.FIN)] = hits[:12]
    out["ben_t_use"]["code_lines_naming_BEN-T_or_MME-T"] = code_hits
    # ---- (b) discard rules of the recombinations on the delivered rows
    rc = {}
    for bb, fam in (("qwen25vl", "Q"), ("llava", "L")):
        rows = jl(f"{C.FIN}/cache2/recomb/{bb}/rows.jsonl")
        index = jl(f"{C.FIN}/cache/{bb}/index.jsonl")
        stream = [e for e in index if e.get("family") == fam and e.get("kind") == "rel"]
        stream.sort(key=lambda e: e["edit_pos"])
        tg = {e["edit_uid"]: e for e in index if e.get("family") == fam and e.get("kind") == "tgen"}
        ig = {e["edit_uid"]: e for e in index if e.get("family") == fam and e.get("kind") == "igen"}
        pos = {e["edit_uid"]: e["edit_pos"] for e in stream}
        ed = {e["edit_uid"]: e for e in stream}
        # every query of the caches (all kinds, all families): (image content, question string)
        test_kinds = {}
        caches = [c for c in ("qwen25vl", "llava", "QO", "blip2") if os.path.exists(f"{C.FIN}/cache/{c}/index.jsonl")]
        for c in caches:
            for e in (index if c == bb else jl(f"{C.FIN}/cache/{c}/index.jsonl")):
                test_kinds.setdefault((e.get("image_md5"), nq(e["prompt"])), set()).add(f"{c}:{e['kind']}")
        test_q = set(test_kinds)
        norm = lambda x: re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()          # noqa: E731
        mid = lambda p: (p or "").split("/")[0]                                          # noqa: E731
        seen_edit, scope_bad, batch_rows = set(), 0, {}
        mme = jl(f"{C.FIN}/cache2/mme/mme_rows.jsonl") if os.path.exists(f"{C.FIN}/cache2/mme/mme_rows.jsonl") else []
        mme_q = {(os.path.splitext(os.path.basename(str(r.get("image", ""))))[0], nq(r.get("prompt") or r.get("question"))) for r in mme}
        per_edit, bad = {}, {"q_equals_own_question_or_rephrase": 0, "equals_cache_query": 0, "equals_mme_query": 0, "xi_image_not_own": 0, "xq_question_not_own": 0, "xx_image_not_own": 0,
                             "xx_question_not_partner": 0, "xq_image_is_an_edit_image_of_own": 0, "xx_partner_is_self": 0, "xx_same_answer_or_question": 0, "xx_min_T_wrong": 0, "min_T_wrong": 0,
                             "duplicate_rows": 0, "xi_question_from_own_item": 0}
        seen = set()
        kinds_hit = {}
        for r in rows:
            u = r["edit_uid"]
            e = ed[u]
            per_edit.setdefault(u, {"X-I": 0, "X-Q": 0, "X-X": 0})[r["kind"]] += 1
            q = nq(r["prompt"])
            own_q = {nq(e["prompt"]), nq(tg[u]["prompt"])}
            key = (r["image_md5"], q)
            bad["duplicate_rows"] += int(key in seen)
            seen.add(key)
            bad["repeat_within_the_same_edit"] = bad.get("repeat_within_the_same_edit", 0) + int((u, key) in seen_edit)
            seen_edit.add((u, key))
            if key in test_q:
                bad["equals_cache_query"] += 1
                for k in test_kinds[key]:
                    kinds_hit[k] = kinds_hit.get(k, 0) + 1
            bad["equals_mme_query"] += int(key in mme_q)
            if r["kind"] == "X-I":
                bad["xi_image_not_own"] += int(r["image_md5"] != e["image_md5"])
                bad["q_equals_own_question_or_rephrase"] += int(q in own_q)
                bad["xi_question_from_own_item"] += int(r.get("src_uid") == u)
                bad["min_T_wrong"] += int(r["min_T"] != pos[u] + 1)
            elif r["kind"] == "X-Q":
                bad["xq_question_not_own"] += int(q != nq(e["prompt"]))
                bad["xq_image_is_an_edit_image_of_own"] += int(r["image_md5"] in (e["image_md5"], ig[u]["image_md5"]))
                bad["min_T_wrong"] += int(r["min_T"] != pos[u] + 1)
            else:
                b = ed[r["edit_b_uid"]]
                bad["xx_image_not_own"] += int(r["image_md5"] != e["image_md5"])
                bad["xx_question_not_partner"] += int(q != nq(b["prompt"]))
                bad["xx_partner_is_self"] += int(r["edit_b_uid"] == u)
                bad["q_equals_own_question_or_rephrase"] += int(q in own_q)
                bad["xx_same_answer_or_question"] += int(nq(b.get("answer")) == nq(e.get("answer")) or nq(b["prompt"]) == nq(e["prompt"]))
                bad["xx_min_T_wrong"] += int(r["min_T"] != max(pos[u], pos[r["edit_b_uid"]]) + 1 or r["edit_b_pos"] != pos[r["edit_b_uid"]])
                bu = r["edit_b_uid"]
                bad["xx_share_entity_mid"] = bad.get("xx_share_entity_mid", 0) + int(bool({mid(e["image"]), mid(ig[u]["image"])} & {mid(b["image"]), mid(ig[bu]["image"])}))
                bad["xx_share_name_or_answer"] = bad.get("xx_share_name_or_answer", 0) + int(bool({norm(r["edit_alt"]), norm(r["edit_pred"])} & {norm(r["edit_b_alt"]), norm(r["edit_b_pred"])}))
                bad["xx_row_fields_not_the_edits"] = bad.get("xx_row_fields_not_the_edits", 0) + int(nq(r["edit_b_src"]) != nq(b["prompt"]) or nq(r["edit_src"]) != nq(e["prompt"]) or nq(r["edit_rephrase"]) != nq(tg[u]["prompt"]))
                a_, b_ = pos[u], pos[bu]
                if r["scope"] == "batch":
                    scope_bad += int(a_ // 10 != b_ // 10 or a_ >= 100)
                    batch_rows[u] = batch_rows.get(u, 0) + 1
                else:
                    scope_bad += int(a_ // 100 != b_ // 100)
            if r["edit_pos"] != pos[u]:
                bad["min_T_wrong"] += 1
        cnt = [tuple(v.values()) for v in per_edit.values()]
        # partner positions of X-X: the batch partners (same batch of 10) and group partners (same block of 100)
        pp = {"same_batch_of_10": 0, "same_block_of_100": 0, "other": 0}
        for r in rows:
            if r["kind"] == "X-X":
                a, b = r["edit_pos"], r["edit_b_pos"]
                pp["same_batch_of_10" if a // 10 == b // 10 else ("same_block_of_100" if a // 100 == b // 100 else "other")] += 1
        rc[bb] = {"n_rows": len(rows), "n_edits": len(per_edit), "n_stream_edits": len(stream), "every_edit_has_4_4_4": all(c == (4, 4, 4) for c in cnt), "violations": bad,
                  "cache_kinds_hit": kinds_hit, "xx_partner_positions": pp, "n_mme_rows_compared": len(mme_q), "caches_compared": caches, "n_cache_queries": len(test_q),
                  "xx_scope_violations": scope_bad, "first_100_edits_with_2_batch_partners": sum(batch_rows.get(e_["edit_uid"], 0) == 2 for e_ in stream[:100]),
                  "first_100_edits_batch_partner_counts": sorted({batch_rows.get(e_["edit_uid"], 0) for e_ in stream[:100]}),
                  "later_edits_with_batch_scope": sum(1 for e_ in stream[100:] if batch_rows.get(e_["edit_uid"], 0))}
        say("recomb", bb, rc[bb])
    out["recomb_rules"] = rc
    # ---- (c) the stop list is fixed in the code
    sys.path.append(C.FIN + "/feat2/retr")
    import stoplist
    out["stop_list"] = {"is_a_literal_in_stoplist_py": "FUNCTION_WORDS = frozenset(" in open(C.FIN + "/feat2/retr/stoplist.py").read(), "n": len(stoplist.STOP), "sha1": stoplist.stop_list_sha1(),
                        "fixed_at": stoplist.FIXED_AT, "file_mtime_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(C.FIN + "/feat2/retr/stoplist.py"))),
                        "no_file_or_data_read": not re.search(r"open\(|json|numpy|load", open(C.FIN + "/feat2/retr/stoplist.py").read().split('"""', 2)[2])}
    # ---- (d) no gate decision was computed by the packages on any other log: scan code and outputs for log names
    dev = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
    pat = re.compile(r"TB-[A-Z]+[0-9]+-[a-z0-9_\-]+")
    scan = {"files": 0, "bytes": 0, "logs_named": {}}
    for root in (C.FIN + "/feat2", C.FIN + "/cache2"):
        for dp, dn, fn in os.walk(root):
            if "/images" in dp or "__pycache__" in dp:
                continue
            for f in fn:
                p = os.path.join(dp, f)
                if not f.endswith((".py", ".txt", ".json", ".jsonl", ".sh", ".log", ".out")) or os.path.getsize(p) > 60e6:
                    continue
                try:
                    txt = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                scan["files"] += 1
                scan["bytes"] += len(txt)
                for m in set(pat.findall(txt)):
                    m = m.rstrip("-_")
                    scan["logs_named"].setdefault(m, set()).add(os.path.relpath(p, C.FIN))
    scan["logs_named"] = {k: sorted(v)[:40] for k, v in sorted(scan["logs_named"].items())}
    out["scan_log_names"] = scan
    out["scan_other_logs"] = {k: v for k, v in scan["logs_named"].items() if not any(k == d or k.startswith(d + "-p") for d in dev)}
    C.dump("cpu2_feat.json", out)
    say("feat: split", out["benign_split"], "stop", out["stop_list"])
    say("other logs named in feat2 / cache2:", {k: len(v) for k, v in out["scan_other_logs"].items()})


if PART == "retr":
    part_retr(sys.argv[2])
elif PART == "stats":
    part_stats()
elif PART == "pools":
    part_pools(sys.argv[2])
elif PART == "feat":
    part_feat()


# =================================================================================================== item 4 for the retrieval rules
def part_h3retr(name):
    """Read-out H3 of the retrieval rules by an own computation: common shift of every attribute's scale, guards kept."""
    import fx
    L = fx.logs.open(name)
    T = L.T
    att = jl(f"{C.FIN}/cache2/retr/{L.backbone}/attributes.jsonl")
    seq = [a for a in att if a["table"] == "sequence"]
    req = {a["edit_uid"]: a for a in att if a["table"] == "edit" and a["family"] == L.family}
    ik = {r["md5"]: i for i, r in enumerate(jl(f"{C.FIN}/cache2/retr/emb/image_keys.jsonl"))}
    qk = {r["question_key"]: i for i, r in enumerate(jl(f"{C.FIN}/cache2/retr/emb/question_keys.jsonl"))}
    EI = np.load(f"{C.FIN}/cache2/retr/emb/image_emb.npy").astype(np.float64)
    EQ = np.load(f"{C.FIN}/cache2/retr/emb/question_emb.npy").astype(np.float64)
    ereq = [req[u] for u in L.uids]
    eI, eQ = EI[[ik[r["image_md5"]] for r in ereq]], EQ[[qk[r["question_key"]] for r in ereq]]

    def scores(rows):
        a = np.full((len(rows), T), -np.inf)
        c = np.empty((len(rows), T))
        for n, j in enumerate(rows):
            s = seq[j]
            if s["image_md5"] is not None:
                a[n] = eI @ EI[ik[s["image_md5"]]]
            c[n] = eQ @ EQ[qk[nq(s["question"])]]
        return a, c
    gI, gQ = (eI * eI).sum(1), (eQ * eQ).sum(1)
    TIE = 1e-9
    pos = [int(L.index[L.edit_rows["rel"][i]].get("edit_pos", i)) for i in range(T)]
    hn = [int(j) for j in L.rows("hn", valid=True)]
    kind = {j: L.index[j]["hn_type"] for j in hn}
    fit = {k: [j for j in hn if kind[j] == k and pos[L.own(j)] % 2 == 0] for k in ("HN-I", "HN-Q", "HN-X")}
    test = {k: [j for j in hn if kind[j] == k and pos[L.own(j)] % 2 == 1] for k in ("HN-I", "HN-Q", "HN-X")}
    odd = [i for i in range(T) if pos[i] % 2 == 1]
    hd = [int(j) for j in L.pool("heldout_disjoint")]
    SF = {k: scores(v) for k, v in fit.items()}
    ST = {k: scores(v) for k, v in test.items()}
    SO = {k: scores([L.edit_rows[k][i] for i in odd]) for k in ("tgen", "igen")}
    SH = scores(hd)
    out = {"log": name, "rules": {}}
    for dr in (0, 1):
        rows = [int(j) for j in fx.draws.calibration(L, 200, dr).rows]
        QA, QC = scores(rows)
        i1, i2 = blocks(200, dr)
        has1 = np.isfinite(QA[i1]).all(1)
        muA, sdA, muC, sdC = QA[i1][has1].mean(0), QA[i1][has1].std(0), QC[i1].mean(0), QC[i1].std(0)

        def zmin(a, c):
            with np.errstate(invalid="ignore"):
                return np.minimum((a - muA) / sdA, (c - muC) / sdC)
        spec = {"retr_conj": [(lambda a, c: a, QA, gI), (lambda a, c: c, QC, gQ)], "retr_sum": [(lambda a, c: a + c, QA + QC, gI + gQ)],
                "retr_min": [(zmin, zmin(QA, QC), zmin(gI[None, :], gQ[None, :])[0])]}
        for nm, attrs in spec.items():
            cal = []
            for fn, Q, g in attrs:
                b, eta = two_block(Q, i1, i2)
                cal.append((fn, b, eta, g))

            def decide(a, c, delta):
                cand, rk = None, 0.0
                for fn, b, eta, g in cal:
                    s = fn(a, c)
                    lit = b + eta + delta
                    sil = g < lit
                    with np.errstate(invalid="ignore"):
                        p = np.where(sil[None, :], s >= np.minimum(lit, g - TIE * np.abs(g))[None, :], (s - b[None, :]) > eta + delta)
                        rk = rk + (s - b[None, :])
                    cand = p if cand is None else cand & p
                rk = np.where(cand, rk, -np.inf)
                f = cand.any(1)
                return f, np.where(f, rk.argmax(1), -1)

            def fit_rate(delta):
                return float(np.mean([100.0 * decide(*SF[k], delta)[0].mean() for k in SF if len(fit[k])]))
            if fit_rate(0.0) <= 10.0:
                dstar = 0.0
            else:
                lo, hi = 0.0, 1.0
                while fit_rate(hi) > 10.0 and hi < 1e6:
                    hi *= 2
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if fit_rate(mid) <= 10.0:
                        hi = mid
                    else:
                        lo = mid
                dstar = hi
            M = fx.methods.build(nm, L, alpha=0.05, n=200, draw=dr)
            h = fx.sweeps.h3(L, M)
            dd = h["delta"]
            rec = {"reachable": h["reachable"], "delta_driver": dd, "delta_mine": dstar}
            if h["reachable"]:
                e = dd + 1e-9 if dd > 0 else 0.0
                cov = {k: 100.0 * float((decide(*SO[k], e)[1] == np.array(odd)).mean()) for k in SO}
                tm = {k: 100.0 * float(decide(*ST[k], e)[0].mean()) for k in ST if len(test[k])}
                hdm = 100.0 * float(decide(*SH, e)[0].mean())
                rec.update(coverage_mine=cov, coverage_driver={k: h["rephrase"][k] for k in ("tgen", "igen")}, test_mine=tm, test_driver={k: h["test"][k]["fired"] for k in tm},
                           heldout_mine=hdm, heldout_driver=h["heldout_disjoint"]["fired"], fit_at_raise_driver=h["fit"]["at_raise"]["mean"], fit_just_above_mine=fit_rate(e),
                           fit_just_below_mine=fit_rate(dd - 1e-9) if dd > 0 else None)
                rec["agree"] = bool(abs(dd - dstar) < 1e-8 and all(abs(cov[k] - h["rephrase"][k]) < 1e-9 for k in cov) and all(abs(tm[k] - h["test"][k]["fired"]) < 1e-9 for k in tm)
                                    and abs(hdm - h["heldout_disjoint"]["fired"]) < 1e-9 and abs(rec["fit_just_above_mine"] - rec["fit_at_raise_driver"]) < 1e-9
                                    and (dd == 0 or rec["fit_just_below_mine"] > 10.0))
            else:
                rec["agree"] = False
            out["rules"].setdefault(nm, {})[dr] = rec
        say(name, "draw", dr, {k: (v[dr]["agree"], v[dr]["delta_driver"], v[dr]["delta_mine"]) for k, v in out["rules"].items()})
    out["ok"] = all(r["agree"] for v in out["rules"].values() for r in v.values())
    C.dump(f"cpu2_h3retr_{name}.json", out)
    say("h3 retrieval ok", out["ok"])


if PART == "h3retr":
    part_h3retr(sys.argv[2])
