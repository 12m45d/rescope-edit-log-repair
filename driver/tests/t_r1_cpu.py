"""CPU tests of stage 2 of the fx driver (fx.review1: the gates registered after the review).  Development logs for every number;
other logs: mechanics only (open, build, one gate-level pass; nothing of them is stored but "it ran", sizes and seconds).

  python t_r1_cpu.py [names...]      -> tests/out/r1_cpu_<name>.json        (python render_r1.py renders tests/out/table_r1.txt)

  naive   (3a) every new gate, calibration draw 0: thresholds and gate-level decisions through the driver == a naive path that calls
               the feature package and the library directly (white.scores + rescope.add_guards / calibrate_* / KeyCache.decide;
               recomb.own_baseline + recomb.calibrate_two_block_x; retr.reference; a plain-loop reference for retr_sum / retr_min)
  mech    (3b) the mechanics tables of the feature packages (feat2/*/out/mechanics*.json) reproduced for draw 0; rescope_null and
               cos_null: the table's ROW rule for the null sample reproduces it, the driver's QUERY rule differs by the reported amount
  h3      (3d) read-out H3 for every calibrated new gate (raise recomputed from the raised method object); both forms for the _x
               methods; the sweep of a retrieval rule == the decisions of its raised rule
  rules   retr_sum / retr_min: validity on exchangeable synthetic scores; _x with no recombination == the gate without _x bit for bit;
          the selection reader (missing / partial / complete); the row set; where a gate is not defined; pools; digests; switches
  logs    mechanics on the 16 logs of the reduced programme and the 6 prefix logs (a subset of the names: build + gate-level pass)
  grid    every name builds for the 20 draws at alpha = 0.05 and for the other four levels at draws 0 - 1 (development logs and working
          points); own prompts stay candidates
"""
from __future__ import annotations

import math
import os
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

R1, MT = fx.review1, fx.methods
OUT = os.path.join(HERE, "out")
DEV = fx.logs.names("development")
KEYS = ("fired", "top_edit", "own_cand", "own_top", "n_cand")
F2 = fx.FINAL + "/feat2"
INF = float("inf")
SCOPED = [n for n in R1.NAMES if n not in R1.RETRIEVAL_FAMILY + ("lookup",)]
CALIBRATED = [n for n in R1.NAMES if n != "lookup"]


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return False


def n_diff(a, b, keys=KEYS) -> int:
    """Number of sequences on which two decisions differ in any of the arrays."""
    bad = np.zeros(len(np.asarray(a["fired"])), dtype=bool)
    for k in keys:
        bad |= np.asarray(a[k]) != np.asarray(b[k])
    return int(bad.sum())


def row_sets(L):
    s = fx.gate.row_sets(L)
    return [("probe", s["probe"], "tf"), ("edit", s["edit"], "tf"), ("hn", s["hn"], "tf"), ("benign", s["benign"], "decode")]


# ----------------------------------------------------------------------------- the naive path
def spec(name):
    """(base feature, kappa, calibration, x) of a scoped name, written out here (not the driver's parser)."""
    x = name.endswith("_x")
    core = name[:-2] if x else name
    sh = core.endswith("_shared")
    core = core[:-7] if sh else core
    kap = None
    for f in ("whitecos", "white"):
        if core.startswith(f + "_"):
            kap, core = float(core[len(f) + 1:]), f
            break
    return core, kap, ("shared" if sh else "two_block"), x


def old_null_rows(L, draw=0, n=200, n_a=100):
    """The null sample of the feature packages' mechanics tables: positions n .. n + n_a - 1 of the draw's ranking (disjoint by ROW)."""
    pool = fx.draws.pool_rows(L, "repair")
    rk = fx.draws.ranking(len(pool), draw)
    return [pool[int(k)] for k in np.sort(rk[n:n + n_a])]


def naive_unthresholded(L, base, kap, A):
    from rescope import add_guards, feature_gate, null_repair
    sys.path.insert(0, F2) if F2 not in sys.path else None
    from white import scores as WS, wstats
    V = MT.released_values(L)[0]
    if base in ("rescope", "cosine", "ball", "rescope_null", "rescope_nullW"):
        S = feature_gate(L.log, {"cosine": "cos", "ball": "negdist"}.get(base, "ncoord"), values=V)
        S, _ = add_guards(S, *L.own_keys())
        if base == "rescope_null":
            S, _ = null_repair(S, L.kc.keys("tf", A), ridge=0.0)
        elif base == "rescope_nullW":
            S, _ = WS.null_w(S, stats=wstats.load(L.backbone))
        return S
    S = WS.feature_gate(L.log, base, stats=None if base == "cos_null" else wstats.load(L.backbone), kappa=kap,
                        null_keys=L.kc.keys("tf", A) if base == "cos_null" else None, values=V)
    return add_guards(S, *L.own_keys())[0]


def naive_scoped(L, name, draw=0, alpha=0.05, null_rows=None):
    from rescope import calibrate_shared, calibrate_two_block
    import recomb
    base, kap, cal, x = spec(name)
    dr = fx.draws.calibration(L, 200, draw)
    A = null_rows if null_rows is not None else (fx.draws.null_sample(L, 100, draw) if base in ("cos_null", "rescope_null") else None)
    S = naive_unthresholded(L, base, kap, A)
    Q = L.kc.coords(S, dr.rows, dr.which)
    if x:
        xv, _ = recomb.own_baseline(L, S)
        return recomb.calibrate_two_block_x(S, coords=Q, x=xv, alpha=alpha, seed=dr.seed)
    if cal == "shared":
        return calibrate_shared(S, coords=Q, alpha=alpha)
    return calibrate_two_block(S, coords=Q, alpha=alpha, seed=dr.seed)


def cosines_loop(L, rows):
    """(SI, SQ) by plain loops over sequences and edits from the stored embeddings (own code; rows without an image: -inf)."""
    import retr
    E = retr.edits(L)
    st = retr.store(E.cache)
    EI, EQ = np.asarray(st.EI, dtype=np.float64), np.asarray(st.EQ, dtype=np.float64)
    SI, SQ = np.full((len(rows), E.T), -INF), np.zeros((len(rows), E.T))
    for a, j in enumerate(rows):
        ir, qr = int(st.seq_img[j]), int(st.seq_q[j])
        for i in range(E.T):
            if ir >= 0:
                SI[a, i] = float(EI[ir] @ EI[int(E.img[i])])
            SQ[a, i] = float(EQ[qr] @ EQ[int(E.q[i])])
    gI = np.array([float(EI[int(E.img[i])] @ EI[int(E.img[i])]) for i in range(E.T)])
    gQ = np.array([float(EQ[int(E.q[i])] @ EQ[int(E.q[i])]) for i in range(E.T)])
    return SI, SQ, gI, gQ


def naive_one_score(L, rule, rows, own, draw=0, alpha=Fraction(1, 20)):
    """retr_sum / retr_min from the amendment's text with plain loops and the DEFINING comparisons (strict margin > eta; no
    floating-point-safe thresholds): block-1 statistics and baselines, block-2 quantile, own-request guard, top-1 by the margin.
    -> decision arrays + the number of sequences on which top-1 by the RAW fused score would release another edit."""
    from rescope import split_blocks
    dr = fx.draws.calibration(L, 200, draw)
    CI, CQ, gI, gQ = cosines_loop(L, dr.rows)
    n, T = CI.shape
    i1, i2 = split_blocks(n, (1, 1), dr.seed)
    if rule == "retr_min":
        mu_i, sd_i, mu_q, sd_q = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        for i in range(T):
            a = [CI[q, i] for q in i1 if CI[q, i] > -INF]
            c = [CQ[q, i] for q in i1]
            mu_i[i], mu_q[i] = math.fsum(a) / len(a), math.fsum(c) / len(c)
            sd_i[i] = math.sqrt(math.fsum((v - mu_i[i]) ** 2 for v in a) / len(a))
            sd_q[i] = math.sqrt(math.fsum((v - mu_q[i]) ** 2 for v in c) / len(c))
        score = lambda a, c, i: min((a - mu_i[i]) / sd_i[i], (c - mu_q[i]) / sd_q[i])          # noqa: E731
    else:
        score = lambda a, c, i: a + c                                                           # noqa: E731
    b = [max(score(CI[q, i], CQ[q, i], i) for q in i1) for i in range(T)]
    s2 = sorted(max(score(CI[q, i], CQ[q, i], i) - b[i] for i in range(T)) for q in i2)
    k = math.ceil((1 - alpha) * (len(i2) + 1))
    eta = INF if k > len(i2) else s2[k - 1]
    g = [score(gI[i], gQ[i], i) for i in range(T)]
    sil = [not (g[i] - b[i] > eta) for i in range(T)]
    cap = [g[i] - 1e-9 * abs(g[i]) for i in range(T)]
    SI, SQ, _, _ = cosines_loop(L, rows)
    N = len(rows)
    out = {"fired": np.zeros(N, dtype=bool), "top_edit": np.full(N, -1), "own_cand": np.zeros(N, dtype=bool), "own_top": np.zeros(N, dtype=bool), "n_cand": np.zeros(N, dtype=np.int64)}
    raw_other = multi = 0
    for a in range(N):
        best, best_m, best_raw, best_s = -1, -INF, -1, -INF
        nc = 0
        for i in range(T):
            s = score(SI[a, i], SQ[a, i], i)
            c = (s >= cap[i]) if sil[i] else (s - b[i] > eta)
            if not c:
                continue
            nc += 1
            if s - b[i] > best_m:
                best, best_m = i, s - b[i]
            if s > best_s:
                best_raw, best_s = i, s
            if i == own[a]:
                out["own_cand"][a] = True
        out["n_cand"][a] = nc
        if nc:
            out["fired"][a], out["top_edit"][a] = True, best
            out["own_top"][a] = best == own[a]
            raw_other += int(best_raw != best)
            multi += int(nc > 1)
    return out, {"eta": eta, "n_capped": int(sum(sil)), "n_top1_by_raw_score_differs": raw_other, "n_rows_with_several_candidates": multi}


def naive():
    sys.path.insert(0, F2) if F2 not in sys.path else None
    R = R1.package("retr")
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        sets = row_sets(L)
        res = {}
        for nm in R1.NAMES:
            t0 = time.time()
            M = MT.build(nm, L, alpha=0.05, draw=0)
            r = {"kind": M.kind, "n_rows": 0, "n_differ": 0, "n_capped": MT.n_capped(M)}
            if M.kind == "scoped":
                Gn, rec = naive_scoped(L, nm)
                r["thresholds_bitwise_equal"] = bool(torch.equal(M.obj.theta(), Gn.theta()))
                r["baselines_equal"] = bool((M.obj.det_b is None and Gn.det_b is None) or torch.equal(M.obj.det_b, Gn.det_b))
                r["n_capped_naive"] = rec["n_edits_capped"]
                for part, rows, w in sets:
                    a, b = fx.gate.decide(L, M, rows, w), L.kc.decide(Gn, w, rows=rows, uids=L.uids)
                    r["n_rows"] += len(rows)
                    r["n_differ"] += n_diff(a, b)
                good = r["thresholds_bitwise_equal"] and r["baselines_equal"] and r["n_differ"] == 0 and r["n_capped"] == r["n_capped_naive"]
            elif nm in ("retr_conj", "lookup"):
                dr = fx.draws.calibration(L, 200, 0)
                Rf = R.reference("retr", L, dr.rows, alpha=0.05, seed=dr.seed) if nm == "retr_conj" else R.reference("lookup", L)
                evn, evd = Rf.evaluate(positions=("tf",)), fx.gate.evaluate(L, M, positions=("tf",))
                same_rows = True
                for part, rows, w in sets:
                    a, b = evd[part][w], evn[part][w]
                    same_rows &= list(a["rows"]) == list(b["rows"]) == list(rows)
                    r["n_rows"] += len(rows)
                    r["n_differ"] += n_diff(a, b)
                r["same_row_sets"] = bool(same_rows)
                if nm == "retr_conj":
                    r["thresholds_bitwise_equal"] = bool(np.array_equal(M.obj.theta()["img"], Rf.thr.th_img) and np.array_equal(M.obj.theta()["q"], Rf.thr.th_q))
                    r["n_capped_naive"] = Rf.rec["n_edits_capped"]
                good = r["n_differ"] == 0 and same_rows and r.get("thresholds_bitwise_equal", True) and r.get("n_capped_naive", r["n_capped"]) == r["n_capped"]
            else:
                extra = {}
                for part, rows, w in sets:
                    a = fx.gate.decide(L, M, rows, w)
                    b, extra_p = naive_one_score(L, nm, rows, a["own"])
                    r["n_rows"] += len(rows)
                    r["n_differ"] += n_diff(a, b)
                    for k in ("n_top1_by_raw_score_differs", "n_rows_with_several_candidates"):
                        extra[k] = extra.get(k, 0) + extra_p[k]
                    extra["eta"], extra["n_capped_naive"] = extra_p["eta"], extra_p["n_capped"]
                r.update(extra, eta_driver=M.rec["cal"]["eta"])
                good = r["n_differ"] == 0 and r["n_capped"] == r["n_capped_naive"] and abs(r["eta"] - r["eta_driver"]) < 1e-12
            r["ok"], r["seconds"] = bool(good), time.time() - t0
            ok &= r["ok"]
            res[nm] = r
            say(f"{name} {nm}: {'ok' if r['ok'] else 'DIFFERS'} ({r['n_differ']} of {r['n_rows']} decisions differ; {r['seconds']:.1f}s)")
        out["logs"][name] = {"gates": res, "n_gates": len(res), "n_ok": sum(v["ok"] for v in res.values()), "n_decisions": sum(v["n_rows"] for v in res.values()),
                             "n_differ": sum(v["n_differ"] for v in res.values())}
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (3b) the mechanics tables
def _stats(L, M):
    return fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf",)))


def _white_row(st, M):
    return {"eta": float(M.rec["cal"]["eta"]), "capped": int(M.rec["cal"]["n_edits_capped"]), "fire": st["probe"]["heldout_disjoint"]["fired"], "rel": st["edit"]["rel"]["own_top"],
            "tgen": st["edit"]["tgen"]["own_top"], "tgen_cand": st["edit"]["tgen"]["own_cand"], "igen": st["edit"]["igen"]["own_top"], "igen_cand": st["edit"]["igen"]["own_cand"],
            "HN-I": st["hn"]["HN-I"]["fired"], "HN-Q": st["hn"]["HN-Q"]["fired"], "HN-X": st["hn"]["HN-X"]["fired"]}


def _white_ref(v):
    c, ro = v["calibration"], v["read_out"]
    return {"eta": c["eta"], "capped": c["n_edits_capped"], "fire": ro["fire_heldout_disjoint"], "rel": ro["rel_own_top"], "tgen": ro["tgen_own_top"], "tgen_cand": ro["tgen_own_cand"],
            "igen": ro["igen_own_top"], "igen_cand": ro["igen_own_cand"], "HN-I": ro["hn_fired"]["HN-I"], "HN-Q": ro["hn_fired"]["HN-Q"], "HN-X": ro["hn_fired"]["HN-X"]}


def _maxdiff(a, b):
    return max(abs(float(a[k]) - float(b[k])) for k in a)


def mech():
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        r = {}
        # ---- white: 12 variants x 2 calibrations
        m = fx.io.jload(f"{F2}/white/out/mechanics_{name}.json")
        old = old_null_rows(L)
        by = {L.sid(j): j for j in fx.draws.pool_rows(L, "repair")}
        by.update({L.index[j].get("lib_sid"): j for j in fx.draws.pool_rows(L, "repair")})
        r["old_null_rows_are_the_tables"] = sorted(old) == sorted(by[s] for s in m["null_sids"])
        new = fx.draws.null_sample(L, 100, 0)
        r["null_sample"] = {"rows_replaced_by_the_query_rule": len(set(old) - set(new)), "calibration_queries_in_the_old_sample": len(set(fx.draws.query_ids(L, old)) & set(fx.draws.query_ids(L, fx.draws.calibration(L, 200, 0).rows)))}
        rows, same, dep = {}, 0, {}
        for lab, v in m["variants"].items():
            for rule in ("two_block", "shared"):
                ref = _white_ref(v["cal"][rule])
                tau = np.asarray(v["cal"][rule]["calibration"]["tau"], dtype=np.float64)
                nm = {"rescope": "rescope", "cosine": "cosine"}.get(lab, lab)
                if lab in ("rescope", "rescope_null"):
                    M = MT.build(lab, L, alpha=0.05, draw=0, calibration=rule)
                elif lab == "cosine":
                    M = MT.cosine(L, rule, alpha=0.05, draw=0)
                else:
                    M = MT.build(nm + ("_shared" if rule == "shared" else ""), L, alpha=0.05, draw=0)
                got = _white_row(_stats(L, M), M)
                th = M.obj.theta().numpy()
                e = {"max_abs_diff_read_out": _maxdiff(ref, got), "thresholds_bitwise_equal": bool(np.array_equal(th, tau)), "max_abs_diff_thresholds": float(np.nanmax(np.abs(np.where(np.isfinite(th) & np.isfinite(tau), th - tau, 0.0))))}
                if lab in ("rescope_null", "cos_null"):                 # the table's ROW rule through the naive path; the driver's QUERY rule differs
                    Gn, rec = naive_scoped(L, lab + ("_shared" if rule == "shared" else ""), null_rows=old)
                    Mo = MT.wrap(L, Gn, lab + "@row", rec={"cal": rec})
                    go = _white_row(_stats(L, Mo), Mo)
                    e["row_rule"] = {"max_abs_diff_read_out": _maxdiff(ref, go), "thresholds_bitwise_equal": bool(np.array_equal(Gn.theta().numpy(), tau))}
                    e["query_rule_minus_table"] = {k: float(got[k]) - float(ref[k]) for k in ref}
                    e["reproduced"] = bool(e["row_rule"]["thresholds_bitwise_equal"] and e["row_rule"]["max_abs_diff_read_out"] < 1e-9)
                    dep[f"{lab}/{rule}"] = e
                else:
                    e["reproduced"] = bool(e["thresholds_bitwise_equal"] and e["max_abs_diff_read_out"] < 1e-9)
                same += int(e["reproduced"])
                rows[f"{lab}/{rule}"] = e
        r["white"] = {"n_rows": len(rows), "n_reproduced": same, "rows": rows, "null_dependent": dep}
        ok &= same == len(rows) and r["old_null_rows_are_the_tables"]
        # ---- retr: the conjunction and lookup
        mr = fx.io.jload(f"{F2}/retr/out/mechanics.json")["logs"][name]
        rr = {}
        for lab, nm in (("retr", "retr_conj"), ("lookup", "lookup")):
            M = MT.build(nm, L, alpha=0.05, draw=0)
            st, ref = _stats(L, M), mr[lab]
            d = [abs(st["probe"]["heldout_disjoint"]["fired"] - ref["heldout_disjoint"]["fired_pct"]), abs(st["benign"]["fired"] - ref["benign"]["fired_pct"])]
            for k in ("rel", "tgen", "igen"):
                d += [abs(st["edit"][k]["own_top"] - ref[k]["coverage_pct"]), abs(st["edit"][k]["fired"] - ref[k]["fired_pct"])]
            for k in ("HN-I", "HN-Q", "HN-X"):
                d += [abs(st["hn"][k]["fired"] - ref[k]["fired_pct"]), abs(st["hn"][k]["own_top"] - ref[k]["coverage_pct"])]
            rr[lab] = {"max_abs_diff": float(max(d)), "capped": MT.n_capped(M), "capped_table": ref["n_capped_edits"],
                       "fingerprint_equal": None if lab == "lookup" else M.rec["cal"]["fingerprint"] == mr["calibration"]["fingerprint"]}
            ok &= rr[lab]["max_abs_diff"] < 1e-9 and rr[lab]["capped"] == rr[lab]["capped_table"] and rr[lab]["fingerprint_equal"] in (None, True)
        r["retr"] = rr
        # ---- recomb: rescope and rescope_x
        mc = fx.io.jload(f"{F2}/recomb/out/mechanics.json")[L.backbone]
        rc = {}
        for i, nm in enumerate(("rescope", "rescope_x")):
            M = MT.build(nm, L, alpha=0.05, draw=0)
            a, b = fx.metrics.leaves(_stats(L, M)), fx.metrics.leaves(mc["gate"][i])
            common = [k for k in b if k in a and isinstance(b[k], (int, float)) and b[k] is not None and a[k] is not None]
            rc[nm] = {"n_cells": len(common), "max_abs_diff": float(max(abs(float(a[k]) - float(b[k])) for k in common)), "eta": float(M.rec["cal"]["eta"]), "eta_table": mc["thresholds"]["eta"][i]}
            ok &= rc[nm]["max_abs_diff"] < 1e-9 and abs(rc[nm]["eta"] - rc[nm]["eta_table"]) < 1e-15 and len(common) > 20
        rc["x_sha1_equal"] = MT.build("rescope_x", L).rec["recomb"]["x_sha1"] == mc["records"]["recomb"]["x_sha1"]
        ok &= rc["x_sha1_equal"]
        r["recomb"] = rc
        out["logs"][name] = r
        say(f"{name}: white {same}/{len(rows)} rows reproduced, retr {rr}, recomb {rc}")
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (3d) read-out H3
def h3():
    S = fx.sweeps
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        sp = S.h3_split(L)
        res = {}
        for nm in CALIBRATED:
            M = MT.build(nm, L, alpha=0.05, draw=0)
            t0 = time.time()
            x = nm.endswith("_x")
            hb = R1.h3_both(L, M) if x else None
            h = hb["all"] if x else S.h3(L, M)
            r = {"reachable": bool(h["reachable"]), "delta": h["delta"], "seconds": time.time() - t0, "fit_at_0": h["fit"]["at_0"]["mean"]}
            good = True
            if h["reachable"]:
                Mr = h["method"]
                per = []
                for k in h["kinds_fit"]:
                    per.append(100.0 * float(fx.gate.decide(L, Mr, sp["fit"][k], "tf")["fired"].mean()))
                r.update(fit_at_raise=h["fit"]["at_raise"]["mean"], fit_recomputed=float(np.mean(per)), coverage=h["rephrase"]["mean"], test_mean_fired=h["test"]["mean_fired"],
                         heldout_fired=h["heldout_disjoint"]["fired"], n_at_cap=h.get("n_edits_at_cap"), below=(h["fit"].get("one_float_below") or {}).get("mean"))
                d = fx.gate.decide(L, Mr, h["test_rows"]["rows"], "tf")
                good &= abs(r["fit_recomputed"] - r["fit_at_raise"]) < 1e-9 and r["fit_at_raise"] <= 10.0 and (h["delta"] == 0.0 or r["below"] > 10.0)
                good &= bool(np.array_equal(d["fired"], h["test_rows"]["fired"]) and np.array_equal(d["top_edit"], h["test_rows"]["top_edit"]))
                if x:
                    u = hb["unseen_foreign"]
                    r["subset"] = {"n": hb["subset"]["n"], "test": hb["subset"]["test"], "test_mean_fired": u["test"]["mean_fired"]}
                    good &= u["delta"] == h["delta"] and u["rephrase"]["mean"] == h["rephrase"]["mean"] and hb["subset"]["test"]["n_rows"] <= hb["subset"]["test"]["of"]
                    keep = set(R1.hn_unseen_foreign(L)["rows"])
                    good &= set(u["test_rows"]["rows"]) == {j for j in h["test_rows"]["rows"] if j in keep}
            else:
                r["at_saturation"] = h.get("at_saturation")
            r["ok"] = bool(good)
            ok &= r["ok"]
            res[nm] = r
        # near_scope of an _x method == the gate statistics; the sweep of a retrieval rule == its raised rule
        Mx = MT.build("rescope_x", L, alpha=0.05, draw=0)
        ns, st = R1.near_scope(L, Mx), fx.metrics.gate_stats(L, fx.gate.evaluate(L, Mx, positions=("tf",)))
        near_ok = all(abs(ns["all"][k]["fired"] - st["hn"][k]["fired"]) < 1e-9 and ns["unseen_foreign"].get(k, {"n": 0})["n"] == ns["subset"]["n"][k]["kept"] for k in R1.HN)
        hn = L.rows("hn", valid=True)
        rows = L.pool("heldout_disjoint") + L.edit_rows["tgen"] + L.edit_rows["igen"] + hn
        sw_bad = 0
        for nm in R1.RETRIEVAL_FAMILY:
            M = MT.build(nm, L, alpha=0.05, draw=0)
            sw = S.open_sweep(L, M, rows)
            for dl in (0.0, 0.03, 0.2, -0.05, 1.0):
                sw_bad += n_diff(sw.at(dl), fx.gate.decide(L, S.raised_method(L, M, dl), rows, "tf"))
            sw_bad += int(S.raised_method(L, M, 0.0).digest != M.digest) + int(S.raised_method(L, M, 0.1).digest == M.digest)
            Ma = S.raised_method(L, M, 0.03)                           # shifts accumulate; a raised rule at shift 0 is itself; H3 of the H3 method raises by 0
            sw_bad += int(S.raised_method(L, Ma, 0.0).digest != Ma.digest) + n_diff(fx.gate.decide(L, S.raised_method(L, Ma, 0.02), rows, "tf"), fx.gate.decide(L, S.raised_method(L, M, 0.03 + 0.02), rows, "tf"))
            sw_bad += n_diff(S.open_sweep(L, Ma, rows).at(0.0), fx.gate.decide(L, Ma, rows, "tf"))
            hh = S.h3(L, M)
            sw_bad += int(S.h3(L, hh["method"])["delta"] != 0.0) if hh["reachable"] else 0
            Md = S.raised_method(L, M, 0.05, guards="drop")
            sw_bad += n_diff(fx.gate.decide(L, S.raised_method(L, Md, 0.05, guards="drop"), rows, "tf"), S.open_sweep(L, M, rows, guards="drop").at(0.05 + 0.05))
            sw_bad += int(not raises(lambda: S.raised_method(L, Md, 0.05), ValueError))
        cv = S.shift_curves(L, MT.build("retr_conj", L), [0.0, 0.05], pools=("BEN-T",))
        lookup_no_scale = bool(raises(lambda: S.h3(L, MT.build("lookup", L)), TypeError))
        ok &= near_ok and sw_bad == 0 and lookup_no_scale and "pool.BEN-T.fired" in cv["curves"]
        out["logs"][name] = {"gates": res, "n_gates": len(res), "n_ok": sum(v["ok"] for v in res.values()), "n_reachable": sum(v["reachable"] for v in res.values()),
                             "near_scope_equals_gate_stats": bool(near_ok), "subset_sizes": ns["subset"]["n"], "retrieval_sweep_differs_from_raised_rule": sw_bad,
                             "lookup_has_no_scale": lookup_no_scale}
        say(f"{name}: H3 ran for {len(res)} gates, consistent {out['logs'][name]['n_ok']}, reachable {out['logs'][name]['n_reachable']}; subset {ns['subset']['n']}; sweep != raised rule {sw_bad}")
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- rules: validity, bit-identity, selection, row set, not defined
def _synthetic(rng, n, T, heavy):
    """Exchangeable queries: i.i.d. rows with a query effect shared by the edits; half of the queries have no image."""
    q = rng.standard_normal((n, 1))
    a = 0.3 + 0.1 * q + 0.1 * (rng.standard_t(3, (n, T)) if heavy else rng.standard_normal((n, T)))
    c = 0.5 + 0.15 * rng.standard_normal((n, 1)) + 0.1 * rng.standard_normal((n, T)) * rng.uniform(0.5, 2.0, (1, T))
    a[rng.random(n) < 0.5] = -INF
    return a, c


def validity(rule, trials=600, n=200, m=300, T=30, alpha=0.05):
    rng = np.random.default_rng(20261111)
    rate, late = [], []
    for t in range(trials):
        SI, SQ = _synthetic(rng, n + m, T, heavy=t % 2 == 1)
        r = R1.one_score_rule(SI[:n], SQ[:n], rule, alpha, seed=t)
        s = SI[n:] + SQ[n:] if rule == "retr_sum" else R1._std_min(SI[n:], SQ[n:], r["stats"])
        rate.append(float((s >= r["lit"][None, :]).any(1).mean()))
        if rule == "retr_min":                                          # control: statistics from BOTH blocks (block 2 read before its quantile)
            from rescope.calibrate import two_block_rule
            st = R1.block1_stats(SI[:n], SQ[:n], np.arange(n))
            rr = two_block_rule(R1._std_min(SI[:n], SQ[:n], st), alpha, blocks=(r["idx1"], r["idx2"]))
            late.append(float((R1._std_min(SI[n:], SQ[n:], st) >= rr["tau"][None, :]).any(1).mean()))
    rate = np.array(rate)
    out = {"trials": trials, "n": n, "n_test": m, "T": T, "alpha": alpha, "bound": r["bound"], "mean_rate": float(rate.mean()), "se": float(rate.std(ddof=1) / math.sqrt(trials)),
           "share_of_trials_above_alpha": float((rate > alpha).mean())}
    if late:
        out["mean_rate_with_statistics_from_both_blocks"] = float(np.mean(late))
    out["ok"] = bool(out["mean_rate"] <= out["bound"] + 3 * out["se"] and out["mean_rate"] >= 0.8 * out["bound"])
    return out


def rules():
    out, ok = {"logs": {}}, True
    out["validity"] = {rule: validity(rule) for rule in ("retr_sum", "retr_min")}
    ok &= all(v["ok"] for v in out["validity"].values())
    say(f"validity: {out['validity']}")
    # ---- the selection reader
    tmp = os.path.join(OUT, "r1_selection_test.json")
    full = {"qwen25vl": {"log": DEV[0], "cosine_family": {"H3": "white_0.1_shared", "H2": {"variant": "whitecos", "calibration": "two_block", "kappa": 1}}, "retrieval": {"H3": "retr_sum", "H2": "retr_min"}, "table": {}},
            "llava": {"log": DEV[1], "cosine_family": {"H3": {"variant": "cosine", "calibration": "shared"}, "H2": "cos_null"}, "retrieval": {"H3": "retr_conj", "H2": "retr_conj"}}}
    sel = {"missing_file_raises": bool(raises(lambda: R1.read_selection(os.path.join(OUT, "no_such_file.json")), FileNotFoundError)),
           "real_file_exists_now": os.path.exists(R1.SELECTION)}
    norm = R1.write_selection(full, tmp)
    got = R1.read_selection(tmp)
    sel["complete"] = got == norm and got["qwen25vl"]["cosine"] == {"H3": "white_0.1_shared", "H3_x": "white_0.1_x", "H2": "whitecos_1", "H2_x": "whitecos_1_x", "differs": True,
                                                                  "x_form_changes_calibration": {"H3": True, "H2": False}} \
        and got["llava"]["cosine"]["H3"] == "cosine_shared" and got["llava"]["cosine"]["H3_x"] == "cosine_x" and got["llava"]["retr"] == {"H3": "retr_conj", "H2": "retr_conj", "differs": False} \
        and R1.selected("qwen25vl", "H3", tmp) == {"cosine": "white_0.1_shared", "cosine_x": "white_0.1_x", "retr": "retr_sum"}
    part = []
    for bad in ({"qwen25vl": full["qwen25vl"]},                                                                     # one backbone only
                {**full, "llava": {"cosine_family": {"H3": "cosine"}, "retrieval": full["llava"]["retrieval"]}},     # H2 of the cosine family missing
                {**full, "llava": {"cosine_family": full["llava"]["cosine_family"]}},                                # no retrieval rule
                {**full, "qwen25vl": {**full["qwen25vl"], "retrieval": {"H3": "retr", "H2": "retr_sum"}}},          # not a rule of the family
                {**full, "qwen25vl": {**full["qwen25vl"], "cosine_family": {"H3": "rescope", "H2": "cosine"}}}):      # not a gate of the family
        fx.io.jdump({"backbones": bad}, tmp)
        part.append(bool(raises(lambda: R1.read_selection(tmp), ValueError)))
    with open(tmp, "w") as f:
        f.write("{ not json")
    part.append(bool(raises(lambda: R1.read_selection(tmp), ValueError)))
    os.remove(tmp)
    sel["partial_or_invalid_raise"] = part
    out["selection"] = sel
    ok &= sel["missing_file_raises"] and sel["complete"] and all(part)
    say(f"selection: {sel}")
    # ---- where a gate is not defined (mechanics on three cut logs: nothing but the reason is read)
    nd = {}
    for name in ("TB-Q100-mmke-l2_3", "TB-Q100-onehop-l2_3", "TB-B100-ae"):
        if not fx.logs.exists(name):
            continue
        Lx = fx.logs.open(name)
        nd[name] = {nm: bool(R1.why_not(Lx, nm)) for nm in ("cos_centred", "white_0.1", "whitecos_1_shared", "rescope_nullW", "cos_null", "retr_conj", "retr_sum", "retr_min", "lookup", "rescope_x", "cosine_x")}
        nd[name]["build_raises_NotDefined"] = all(bool(raises(lambda nm=nm: MT.build(nm, Lx), R1.NotDefined)) for nm, v in nd[name].items() if v is True)
        rs = R1.e1_rowset_review1(Lx, 0.05, 0, labels=("rescope", "cos_centred", "rescope_x"), skip_unavailable=False)
        nd[name]["rowset_marks_not_defined"] = [M.kind for M in rs] == ["scoped", "unavailable", "unavailable"] and all(M.rec.get("not_defined") for M in rs[1:])
    L0 = fx.logs.open(DEV[0])
    nd["always"] = {nm: bool(raises(lambda nm=nm: MT.build(nm, L0), R1.NotDefined)) for nm in ("retr_sum_x", "retr_conj_x", "lookup_x", "cosine_shared_x", "white_0.1_shared_x")}
    nd["unknown"] = [bool(raises(lambda: MT.build("white_0.5", L0), ValueError)), bool(raises(lambda: MT.build("white", L0), ValueError)), bool(raises(lambda: MT.build("no_such_gate", L0), KeyError)),
                     MT.build("white", L0, kappa=0.1, calibration="shared").label == "white_0.1_shared"]
    out["not_defined"] = nd
    want = {"TB-Q100-mmke-l2_3": dict(cos_centred=True, rescope_nullW=True, cos_null=False, rescope_x=True), "TB-Q100-onehop-l2_3": dict(cos_centred=True, retr_conj=True, rescope_x=True, cos_null=False),
            "TB-B100-ae": dict(cos_centred=True, retr_conj=True, lookup=True, rescope_x=True)}
    ok &= all(nd[k][a] == b for k, w in want.items() if k in nd for a, b in w.items()) and all(nd[k]["build_raises_NotDefined"] and nd[k]["rowset_marks_not_defined"] for k in want if k in nd)
    ok &= all(nd["always"].values()) and all(nd["unknown"])
    say(f"not defined: {nd}")
    for name in DEV:
        L = fx.logs.open(name)
        r = {}
        # ---- _x with no recombination == the gate without _x, bit for bit
        none = torch.full((L.T,), -INF, dtype=torch.float64)
        same = {}
        for b in R1.X_BASES:
            Mx = MT.build(b + "_x", L, alpha=0.05, draw=0, baseline_x=none)
            M0 = MT.build(b, L, alpha=0.05, draw=0)
            same[b] = bool(torch.equal(Mx.obj.theta(), M0.obj.theta()) and torch.equal(Mx.obj.det_b, M0.obj.det_b) and Mx.obj.eta == M0.obj.eta and Mx.digest == M0.digest)
        r["x_without_recombination_equals_gate"] = same
        # ---- the row set
        rs = R1.e1_rowset_review1(L, 0.05, 0)
        old = MT.e1_rowset(L, 0.05, 0)
        labs = [M.label for M in rs]
        cal = [M for M in rs if M.label in MT.E1_DEPENDS or M.label in R1.R1_DEPENDS]
        sids = {tuple(M.rec["draw"]["sids"]) for M in cal if "draw" in M.rec}
        allg = R1.e1_rowset_review1(L, 0.05, 0, labels=MT.E1_LABELS + R1.R1_LABELS + MT.E1_GATE_LEVEL + R1.R1_GATE_LEVEL)
        r["rowset"] = {"n_rows": len(rs), "labels_ok": labs == [M.label for M in old] + list(R1.R1_LABELS), "stage1_rows_same_content": [M.digest for M in rs[:len(old)]] == [M.digest for M in old],
                       "same_calibration_rows": len(sids) == 1, "n_calibrated": len(cal), "with_gate_level": [M.label for M in allg][-2:] == ["rescope_pac", "rescope_nullW"],
                       "other_draw_changes_calibrated_rows_only": {M.label for M, N in zip(rs, R1.e1_rowset_review1(L, 0.05, 1)) if M.digest != N.digest} <= {M.label for M in cal},
                       "n_rows_changed_by_the_draw": sum(M.digest != N.digest for M, N in zip(rs, R1.e1_rowset_review1(L, 0.05, 1))),
                       "digests_distinct": len({M.digest for M in rs}) == len(rs), "capped": {M.label: MT.n_capped(M) for M in cal}}
        # ---- pools, digests, switches
        Mw, Ms, Mc, Ml = (MT.build(n, L) for n in ("whitecos_0.1", "retr_sum", "retr_conj", "lookup"))
        P = fx.pools
        r["pools"] = {"BEN-T": {M.label: P.firing(L, M, "BEN-T")["n"] for M in (Mw, Ms, Mc, Ml)}, "MME-T_scoped_n": P.firing(L, Mw, "MME-T")["n"],
                      "MME-T_retrieval_not_defined": [bool(raises(lambda M=M: P.firing(L, M, "MME-T"), R1.NotDefined)) for M in (Ms, Mc, Ml)],
                      "BEN-T_calibration_refused": bool(raises(lambda: MT.build("retr_sum", L, cal=fx.draws.explicit(L, P.get(L, "BEN-T").rows[:200])), P.PoolGuardError))}
        sw = {"allow_silence": [MT.n_capped(MT.build(n, L, allow_silence=True)) for n in ("cos_centred", "retr_conj", "retr_min")],
              "qdisjoint": [MT.build(n, L, mask="qdisjoint").variant for n in ("cos_null", "retr_sum", "rescope_x")],
              "BEN-S": [MT.build(n, L, source="BEN-S", which="decode").rec["draw"]["source"] for n in ("white_0.1",)] + [MT.build("retr_conj", L, source="BEN-S").rec["draw"]["source"]],
              "MME-C_scoped": MT.build("whitecos_1", L, source="MME-C").rec["draw"]["source"], "MME-C_retrieval_not_defined": bool(raises(lambda: MT.build("retr_sum", L, source="MME-C"), R1.NotDefined)),
              "null_sample_sharing_a_query_refused": bool(raises(lambda: MT.build("cos_null", L, null_rows=old_null_rows(L)), ValueError)),
              "unknown_switch_raises": bool(raises(lambda: MT.build("retr_sum", L, role="all"), TypeError)) and bool(raises(lambda: MT.build("rescope_x", L, role="all"), ValueError))}
        r["switches"] = sw
        dg = {"same_build_same_digest": MT.build("retr_min", L).digest == MT.build("retr_min", L).digest and MT.build("white_1", L).digest == MT.build("white_1", L).digest,
              "request_digests_distinct": len({M.digest for M in (Ms, Mc, Ml, MT.build("retr_min", L), MT.build("retr_sum", L, draw=1))}) == 5,
              "dedupe": len(MT.dedupe([Ms, MT.build("retr_sum", L), Mc])) == 2, "mask_edits_refused": bool(raises(lambda: MT.mask_edits(L, Ms, [0]), TypeError)),
              "unthresholded": bool(torch.equal(R1.unthresholded(L, "whitecos_0.1_shared").det_w, Mw.obj.det_w) and float(R1.unthresholded(L, "cos_null", draw=0).det_tau.max()) == -INF)}
        r["digests"] = dg
        good = (all(same.values()) and all(v for k, v in r["rowset"].items() if k not in ("n_rows", "n_calibrated", "capped", "n_rows_changed_by_the_draw")) and all(r["pools"]["MME-T_retrieval_not_defined"])
                and r["pools"]["BEN-T_calibration_refused"] and len(set(r["pools"]["BEN-T"].values())) == 1 and sw["allow_silence"] == [0, 0, 0] and all("mask=qdisjoint" in v for v in sw["qdisjoint"])
                and sw["BEN-S"] == ["BEN-S", "BEN-S"] and sw["MME-C_scoped"] == "MME-C" and sw["MME-C_retrieval_not_defined"] and sw["null_sample_sharing_a_query_refused"]
                and sw["unknown_switch_raises"] and all(dg.values()))
        r["ok"] = bool(good)
        ok &= r["ok"]
        out["logs"][name] = r
        say(f"{name}: rules {'ok' if good else 'FAIL'}: {r}")
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- mechanics on the logs of the reduced programme and the prefix logs
MECH_NAMES = ("cos_centred", "cos_null_shared", "white_0.1", "whitecos_1", "rescope_nullW", "rescope_x", "whitecos_0.1_x", "cosine_x", "retr_conj", "retr_sum", "retr_min", "lookup")


def logs():
    out, ok = {"logs": {}, "names": list(MECH_NAMES)}, True
    names = list(fx.logs.programme()) + fx.logs.names("prefix")
    for name in names:
        if not fx.logs.exists(name):
            out["logs"][name] = {"exists": False}
            ok = False
            continue
        t0 = time.time()
        L = fx.logs.open(name)
        r = {"T": L.T, "role": L.role, "seconds_build": {}, "ran": {}}
        hn = L.rows("hn", valid=True)
        for nm in MECH_NAMES:
            try:
                t1 = time.time()
                M = MT.build(nm, L, alpha=0.05, draw=0)
                r["seconds_build"][nm] = time.time() - t1
                ev = fx.gate.evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn"))
                good = len(ev["probe"]["tf"]["fired"]) > 0 and len(ev["edit"]["tf"]["fired"]) == 3 * L.T and isinstance(M.digest, str)
                if hn and nm != "lookup" and name in ("TB-Q500-memitw", "TB-L500-ae-l2_3") + tuple(fx.logs.names("prefix")):
                    h = (R1.h3_both(L, M)["all"] if nm.endswith("_x") else fx.sweeps.h3(L, M))
                    good &= "reachable" in h
                r["ran"][nm] = bool(good)
            except Exception as e:                                      # noqa: BLE001
                r["ran"][nm] = f"{type(e).__name__}: {str(e)[:200]}"
        if hn:
            sub = R1.hn_unseen_foreign(L)
            r["hn_subset"] = {"n_all": sub["n_all"], "n_kept": sub["n_kept"]}
        r["n_recombination_rows"] = len(R1._recomb_rows(L))
        r["ok"] = all(v is True for v in r["ran"].values())
        r["seconds"] = time.time() - t0
        ok &= r["ok"]
        out["logs"][name] = r
        say(f"{name} (T = {L.T}): {'ran' if r['ok'] else r['ran']} ({r['seconds']:.0f}s)")
        for c in (R1._BASES, R1._XB, R1._NULLW, R1._FORCED, MT._BASE, MT._VALUES):
            c.clear()
        fx.gate.clear_cache()
    out["n_logs"], out["n_ok"] = len(names), sum(bool(v.get("ok")) for v in out["logs"].values())
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- every name, 20 draws at alpha = 0.05 and the other levels at draws 0 - 1
def grid():
    out, ok = {"logs": {}}, True
    for name in DEV + fx.logs.names("working_point"):
        L = fx.logs.open(name)
        n, bad, inf, capmax = 0, [], 0, {}
        for alpha, draws in ((0.05, range(fx.draws.N_DRAWS)), (0.01, (0, 1)), (0.02, (0, 1)), (0.10, (0, 1)), (0.20, (0, 1))):
            for d in draws:
                for nm in R1.NAMES:
                    if nm == "lookup" and (alpha, d) != (0.05, 0):
                        continue
                    try:
                        M = MT.build(nm, L, alpha=alpha, draw=d)
                        dec = fx.gate.decide(L, M, L.edit_rows["rel"], "tf")
                        n += 1
                        if nm != "lookup":
                            inf += int(bool(M.rec["cal"].get("eta_infinite")))
                            capmax[nm] = max(capmax.get(nm, 0), int(MT.n_capped(M)))
                            if not allow_missing_own(dec):
                                bad.append(f"{nm} a{alpha} d{d}: an own prompt is not a candidate with guards on")
                    except Exception as e:                              # noqa: BLE001
                        bad.append(f"{nm} a{alpha} d{d}: {type(e).__name__}: {str(e)[:120]}")
        out["logs"][name] = {"n_built": n, "failures": bad[:20], "n_failures": len(bad), "n_eta_infinite": inf, "largest_number_of_capped_edits": capmax}
        ok &= not bad
        say(f"{name}: {n} gates built over draws and levels, failures {len(bad)} {bad[:3]}")
    out["ok"] = bool(ok)
    return out


def allow_missing_own(dec) -> bool:
    """With guards on, every edit is a candidate on its own Rel prompt (cosine family: a guard key; retrieval: the own request)."""
    return bool(np.asarray(dec["own_cand"]).all())


TESTS = {"naive": naive, "mech": mech, "h3": h3, "rules": rules, "logs": logs, "grid": grid}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for nm in sys.argv[1:] or list(TESTS):
        t0 = time.time()
        res = TESTS[nm]()
        res.update(test=nm, seconds=time.time() - t0, code=fx.code_fingerprint(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        fx.io.jdump(res, os.path.join(OUT, f"r1_cpu_{nm}.json"), indent=1)
        say(f"{nm}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds']:.0f}s)")
