"""CPU tests of the stage-1 extension of the fx driver (tasks 1 - 10 of the extension; development logs for every number, other
logs: mechanics only, nothing of them is printed or stored but "it ran").

  python t_ext_cpu.py [names...]      -> tests/out/ext_cpu_<name>.json        (python render_ext.py renders tests/out/table_ext.txt)

  null       (1) rescope_null: n_A = 0 == the default method bit for bit; null_keys(L, draw) and the explicit-keys path == the seeded
                 path bit for bit; no calibration query is a null-sample query (query_id), 20 draws; refusals
  samples    (2) contaminated draws of E6 (even edits, half text / half image, nested); draws from BEN-S and MME-C; the guard:
                 whole benign pool, BEN-T, MME-T, BEN-W refused as calibration / statistics / stream data on every path
  sweeps     (3) the raised scale: GateSweep == fx.gate.decide of the raised gate object; curves at shift 0 == gate statistics;
                 coverage scale; H3 against an independent recomputation from the raised method object; default method + cosine gate
  rowset     (4) the E1 row set holds every registered row; calibrated rows share the calibration rows of a draw
  calib      (5) rescope_pac: rank against exact computation and scipy, validity on exchangeable synthetic scores; external
                 baselines (hook): no values == default bit for bit, formula check, == the feature package's calibration;
                 top-1 with capped edits: "argmax first" read-out against a brute-force recomputation
  readouts   (6) kept / gained, D2 rate with interval, grid for the crossed bootstrap, blocklist (memo only, no model)
  stats2     (7) crossed bootstrap (coverage on a two-way random-effects table, paired version, ratio form), Holm
  pools      (8) registry of the reduced programme (mechanics on every log), key-level pools, masks of R8
  prefix     (9) prefix logs (after `python -m fx.prefix build`): open, factors, hard negatives, default method; all batches == the
                 development log
  maint      (10) mask_edits (empty set, never released, added detector, guards), content digest (review finding 2)
"""
from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(HERE, "out")
DEV = fx.logs.names("development")
KEYS = ("fired", "top_edit", "own_cand", "own_top", "n_cand")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def same_gate(a, b) -> bool:
    eq = lambda x, y: (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))          # noqa: E731
    return all(eq(getattr(a, k), getattr(b, k)) for k in ("det_tau", "det_w", "det_s", "det_b", "det_u", "values", "det_key")) and a.eta == b.eta


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


# ----------------------------------------------------------------------------- (1) rescope_null
def null():
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        r = {"n_A_0_equals_default": 0, "explicit_keys_equal_seeded": 0, "draws_checked": 3}
        for d in range(3):
            base = fx.methods.rescope(L, draw=d)
            M0 = fx.methods.rescope_null(L, n_A=0, draw=d)
            r["n_A_0_equals_default"] += int(same_gate(base.obj, M0.obj) and base.digest == M0.digest)
            Mn = fx.methods.rescope_null(L, draw=d)
            K, rows = fx.methods.null_keys(L, d)
            Mk = fx.methods.rescope_null(L, draw=d, null_keys=K)
            r["explicit_keys_equal_seeded"] += int(same_gate(Mn.obj, Mk.obj) and Mn.digest == Mk.digest and rows == fx.draws.null_sample(L, 100, d)
                                                   and torch.equal(K, L.kc.keys("tf", rows)) and tuple(K.shape) == (100, L.log.d))
        q = lambda rows: {L.index[j]["query_id"] for j in rows}                                    # noqa: E731
        shared = [len(q(fx.methods.null_rows(L, d)) & q(fx.draws.calibration(L, 200, d).rows)) for d in range(fx.draws.N_DRAWS)]
        distinct = [len(q(fx.methods.null_rows(L, d))) for d in range(fx.draws.N_DRAWS)]
        in_pool = all(set(fx.methods.null_rows(L, d)) <= set(L.pool("repair")) for d in range(fx.draws.N_DRAWS))
        cal = fx.draws.calibration(L, 200, 0).rows
        r.update(n_draws=fx.draws.N_DRAWS, calibration_queries_in_null_sample_max=max(shared), null_sample_distinct_queries_min=min(distinct), null_sample_in_repair_pool=in_pool,
                 calibration_key_as_null_key_refused=raises(lambda: fx.methods.rescope_null(L, draw=0, null_keys=L.kc.keys("tf", cal[:3])), ValueError),
                 two_samples_refused=raises(lambda: fx.methods.rescope(L, null_keys=K, null_rows=rows), ValueError),
                 n_A_50=fx.methods.rescope_null(L, n_A=50).rec["null_repair"]["n_a"], in_e1_rowset="rescope_null" in fx.methods.E1_LABELS,
                 variant_n_A_0=fx.methods.rescope_null(L, n_A=0).variant)
        good = (r["n_A_0_equals_default"] == 3 and r["explicit_keys_equal_seeded"] == 3 and max(shared) == 0 and min(distinct) == 100 and in_pool
                and r["calibration_key_as_null_key_refused"] and r["two_samples_refused"] and r["n_A_50"] == 50 and r["in_e1_rowset"])
        ok &= good
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (2) samples and the guard
def samples():
    P = fx.pools
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        r, c = {}, {}
        clean = fx.draws.calibration(L, 200, 0).rows
        prev = set()
        for p in fx.draws.CONTAMINATION:
            dr = fx.draws.calibration(L, 200, 0, contamination=p, contam_half="even")
            k = int(round(p * 200))
            ins = [j for j in dr.rows if L.index[j]["kind"] in ("tgen", "igen")]
            nt, ni = sum(L.index[j]["kind"] == "tgen" for j in ins), sum(L.index[j]["kind"] == "igen" for j in ins)
            c[str(p)] = {"n": len(dr.rows), "n_in_scope": len(ins), "expected": k, "n_text": nt, "n_image": ni, "all_even_edits": all(L.own(j) % 2 == 0 for j in ins),
                         "benign_part_nested_in_clean_draw": set(dr.rows[:200 - k]) <= set(clean), "nested_in_share": prev <= set(ins)}
            prev = set(ins)
            ok &= (len(dr.rows) == 200 and len(ins) == k and nt == k - k // 2 and ni == k // 2 and c[str(p)]["all_even_edits"] and c[str(p)]["benign_part_nested_in_clean_draw"]
                   and c[str(p)]["nested_in_share"])
        r["contamination_even"] = c
        r["first_half_draw_unchanged"] = fx.draws.calibration(L, 200, 0, contamination=0.05).rec["contam_half"].startswith("A (")
        Mc = fx.methods.rescope(L, contamination=0.2, contam_half="even")
        r["contaminated_method"] = {"variant": Mc.variant, "n": Mc.n, "contaminating_edits_even": all(e % 2 == 0 for e in Mc.rec["draw"]["contaminating_edits"])}
        ok &= r["first_half_draw_unchanged"] and "contam_half=even" in Mc.variant and r["contaminated_method"]["contaminating_edits_even"]
        # ---- other key-level sources
        parts = P.benign_parts(L)
        src = {}
        for which in ("tf", "decode", "steps"):
            dr = fx.draws.calibration(L, 200, 0, source="BEN-S", which=which)
            src[f"BEN-S/{which}"] = {"n_pool": dr.rec["n_pool"], "n": len(dr.rows), "all_in_part": all(parts[L.index[j]["sid"]] == "BEN-S" for j in dr.rows), "kc_is_log_cache": dr.kc is None}
            ok &= src[f"BEN-S/{which}"]["all_in_part"] and len(dr.rows) == 200 and dr.kc is None
        same = fx.draws.calibration(L, 200, 0, source="benign", part="BEN-S").rows == fx.draws.calibration(L, 200, 0, source="BEN-S").rows
        dm = fx.draws.calibration(L, 200, 1, source="MME-C")
        mk = P.mme(L.backbone).kc
        src["MME-C/tf"] = {"n_pool": dm.rec["n_pool"], "n": len(dm.rows), "all_in_part": all(mk.index[j]["part"] == "MME-C" and mk.index[j]["y0_valid"] for j in dm.rows),
                           "kc_is_mme_cache": dm.kc is mk, "sids": dm.rec["sids"][:2]}
        ok &= same and src["MME-C/tf"]["all_in_part"] and dm.kc is mk and len(dm.rows) == 200
        Mb, Mm, Mr = fx.methods.rescope(L, source="BEN-S"), fx.methods.rescope(L, source="MME-C"), fx.methods.rescope(L)
        Qm = mk.coords(Mm.obj, dm.rows[:0] or fx.draws.calibration(L, 200, 0, source="MME-C").rows)
        b2 = [i for i in Mm.rec["cal"]["block2"]]
        fire_b2 = int((Qm[b2] >= Mm.obj.theta()[None, :]).any(1).sum())
        src["methods"] = {"variants": [Mb.variant, Mm.variant], "thresholds_differ_from_repair": not torch.equal(Mb.obj.det_tau, Mr.obj.det_tau) and not torch.equal(Mm.obj.det_tau, Mr.obj.det_tau),
                          "mme_block2_firing": fire_b2, "mme_allowed": Mm.rec["cal"]["n_allowed_above"], "mme_capped": Mm.rec["cal"]["n_edits_capped"],
                          "null_with_mme_source_builds": fx.methods.rescope_null(L, source="MME-C").rec["null_repair"]["n_queries_shared_with_calibration"] == 0,
                          "cosine_ben_s": fx.methods.cosine(L, source="BEN-S").variant}
        ok &= src["methods"]["thresholds_differ_from_repair"] and (fire_b2 <= Mm.rec["cal"]["n_allowed_above"] or Mm.rec["cal"]["capped"]) and src["methods"]["null_with_mme_source_builds"]
        r["sources"] = src
        # ---- the guard
        bt, bw = P.get(L, "BEN-T").rows, P.get(L, "BEN-W").rows
        bt_tf = P.get(L, "BEN-T", "tf").rows
        mt = P.get(L, "MME-T").rows
        from fx.draws import Draw
        g = {"whole_benign_pool_refused": raises(lambda: fx.draws.calibration(L, 200, 0, source="benign"), ValueError),
             "whole_benign_pool_refused_in_method": raises(lambda: fx.methods.rescope(L, source="benign"), ValueError),
             "BEN-T_source_refused": raises(lambda: fx.draws.calibration(L, 100, 0, source="BEN-T"), P.PoolGuardError),
             "MME-T_source_refused": raises(lambda: fx.draws.calibration(L, 100, 0, source="MME-T"), P.PoolGuardError),
             "BEN-W_source_refused": raises(lambda: fx.draws.calibration(L, 100, 0, source="BEN-W"), P.PoolGuardError),
             "benign_part_BEN-T_refused": raises(lambda: fx.draws.calibration(L, 100, 0, source="benign", part="BEN-T"), P.PoolGuardError),
             "explicit_BEN-T_refused": raises(lambda: fx.draws.explicit(L, bt_tf[:50]), P.PoolGuardError),
             "explicit_MME-T_refused": raises(lambda: fx.draws.explicit(L, mt[:50], kc=mk), P.PoolGuardError),
             "explicit_one_BEN-T_row_among_repair_refused": raises(lambda: fx.draws.explicit(L, clean[:60] + bt_tf[:1]), P.PoolGuardError),
             "handmade_draw_BEN-T_refused_at_calibration": raises(lambda: fx.methods.rescope(L, cal=Draw(rows=bt_tf[:80], which="tf", n=80, draw=0, source="x")), P.PoolGuardError),
             "handmade_draw_MME-T_refused_at_calibration": raises(lambda: fx.methods.recalibrate(L, fx.methods.base_gate(L)[0], cal=Draw(rows=mt[:80], which="tf", n=80, draw=0, source="x", kc=mk)), P.PoolGuardError),
             "null_rows_BEN-T_refused": raises(lambda: fx.methods.rescope(L, null_rows=bt_tf[:20]), P.PoolGuardError),
             "least_BEN-T_refused": raises(lambda: fx.methods.rescope(L, least=bt_tf[:2]), P.PoolGuardError),
             "exclude_BEN-T_refused": raises(lambda: fx.draws.calibration(L, 100, 0, exclude=bt_tf[:2]), P.PoolGuardError),
             "BEN-S_explicit_allowed": len(fx.draws.explicit(L, P.get(L, "BEN-S", "tf").rows[:50]).rows) == 50,
             "MME-C_explicit_allowed": len(fx.draws.explicit(L, P.get(L, "MME-C").rows[:50], kc=mk).rows) == 50,
             "BEN-W_null_rows_allowed": fx.methods.rescope(L, null_rows=P.get(L, "BEN-W", "tf").rows[:20]).rec["null_repair"]["n_a"] == 20,
             "BEN-W_calibration_refused": raises(lambda: fx.draws.explicit(L, P.get(L, "BEN-W", "tf").rows[:50]), P.PoolGuardError),
             "behave_refuses_benign_rows": raises(lambda: fx.behave.session(L, None).base_row(bt[0], "tf"), KeyError)}
        r["guard"] = g
        ok &= all(g.values())
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (3) sweeps, coverage scale, H3
def _decisions_equal(L, sw, gate, rows, delta, which="tf") -> int:
    a, b = sw.at(delta), fx.gate.decide(L, gate, rows, which)
    return sum(int((np.asarray(a[k]) != np.asarray(b[k])).sum()) for k in KEYS)


def sweeps():
    S = fx.sweeps
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        sets = fx.gate.row_sets(L, benign=False)
        rows = sets["probe"] + sets["edit"] + sets["hn"]
        r = {}
        for lab, M in (("rescope", fx.methods.rescope(L, draw=0)), ("cosine", fx.methods.cosine(L, draw=0)), ("rescope_shared", fx.methods.rescope_shared(L, draw=0))):
            m = {"n_edits_capped": M.rec["cal"]["n_edits_capped"]}
            sc = S.Scale(M.obj)
            m["scale_0_is_the_gate"] = bool(torch.equal(sc.theta(0.0), M.obj.theta())) and sc.gate(0.0) is fx.gate._cpu(M.obj)
            m["n_at_cap_0"] = sc.n_at_cap(0.0)
            sw = S.GateSweep(L, M, rows)
            nd = 0
            for d in (0.0, 0.01, 0.05, 0.2, 1.0, -0.05):
                nd += _decisions_equal(L, sw, sw.gate(d), rows, d)
            swd = S.GateSweep(L, M, rows, guards="drop")
            nd_drop = sum(_decisions_equal(L, swd, swd.gate(d), rows, d) for d in (0.0, 0.05))
            m.update(n_decisions_differing_from_gate_object=nd, n_compared=6 * 5 * len(rows), drop_differing=nd_drop,
                     drop_0_is_the_gate=bool(torch.equal(swd.scale.theta(0.0), M.obj.theta())))
            # guards kept at every scale: every own Rel prompt keeps its own edit as a candidate
            own = S.GateSweep(L, M, L.edit_rows["rel"])
            m["own_prompts_candidate_at_raise_0.5"] = int(own.at(0.5)["own_cand"].sum())
            m["own_prompts_candidate_at_raise_0.5_drop"] = int(S.GateSweep(L, M, L.edit_rows["rel"], guards="drop").at(0.5)["own_cand"].sum())
            # curves at shift 0 == the gate statistics of the method
            cv = S.shift_curves(L, M, [0.0, 0.05], pools=("BEN-T", "MME-T"))
            gs = fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, parts=("probe", "edit", "hn")))
            pairs = [(cv["curves"]["probe.heldout_disjoint.fired"][0], gs["probe"]["heldout_disjoint"]["fired"]), (cv["curves"]["edit.tgen.own_top"][0], gs["edit"]["tgen"]["own_top"]),
                     (cv["curves"]["edit.igen.own_cand"][0], gs["edit"]["igen"]["own_cand"]), (cv["curves"]["hn.HN-I.fired"][0], gs["hn"]["HN-I"]["fired"]),
                     (cv["curves"]["hn.HN-X.own_top"][0], gs["hn"]["HN-X"]["own_top"]), (cv["curves"]["probe.heldout_strict.fired"][0], gs["probe"]["heldout_strict"]["fired"]),
                     (cv["curves"]["pool.BEN-T.fired"][0], fx.pools.firing(L, M, "BEN-T")["fired"]), (cv["curves"]["pool.MME-T.fired"][0], fx.pools.firing(L, M, "MME-T")["fired"])]
            m["curves_at_0_differing_from_gate_stats"] = sum(abs(float(a) - float(b)) > 1e-9 for a, b in pairs)
            m["curves_monotone_in_firing"] = bool(cv["curves"]["probe.heldout_disjoint.fired"][1] <= cv["curves"]["probe.heldout_disjoint.fired"][0])
            # coverage scale: the level set boundary, with the exact gate decision
            cs = []
            reph = L.edit_rows["tgen"] + L.edit_rows["igen"]
            for target in (25.0, 50.0, 75.0):
                c = S.coverage_scale(L, M, target)
                if c["reachable"] and math.isfinite(c["delta"]):
                    g2 = S.Scale(M.obj).gate(c["delta"])
                    cov = 100.0 * float(fx.gate.decide(L, g2, reph)["own_top"].mean())
                    c["check"] = bool(cov >= target and abs(cov - c["coverage"]) < 1e-9 and c["coverage_next"] < target)
                else:
                    c["check"] = True
                cs.append({k: v for k, v in c.items() if k != "kinds"})
            m["coverage_scale"] = cs
            # H3 against an independent recomputation through fx.gate.decide of the raised METHOD object
            h = S.h3(L, M)
            sp = S.h3_split(L)
            m["h3_split"] = {"n_fit": sp["n"]["fit"], "n_test": sp["n"]["test"], "fit_rows_even": all(L.own(j) % 2 == 0 for k in sp["fit"] for j in sp["fit"][k]),
                             "test_rows_odd": all(L.own(j) % 2 == 1 for k in sp["test"] for j in sp["test"][k]), "n_odd_edits": len(sp["odd_edits"]),
                             "partition": sum(len(v) for s in ("fit", "test") for v in sp[s].values()) == len(L.rows("hn", valid=True))}
            hh = {"reachable": h["reachable"], "delta": h["delta"]}
            if h["reachable"]:
                Mh = h["method"]
                fitm = float(np.mean([100.0 * float(fx.gate.decide(L, Mh, sp["fit"][k])["fired"].mean()) for k in h["kinds_fit"]]))
                tst = {k: 100.0 * float(fx.gate.decide(L, Mh, sp["test"][k])["fired"].mean()) for k in S.HN if sp["test"][k]}
                odd = sp["odd_edits"]
                cov = {k: 100.0 * float(fx.gate.decide(L, Mh, [L.edit_rows[k][i] for i in odd])["own_top"].mean()) for k in ("tgen", "igen")}
                hd = 100.0 * float(fx.gate.decide(L, Mh, L.pool("heldout_disjoint"))["fired"].mean())
                below = None
                if h["delta"] > 0:                                      # minimality: no smaller raise on a grid reaches the limit, and one float below fails
                    grid = np.linspace(0.0, h["delta"], 60, endpoint=False)
                    swf = S.GateSweep(L, M, [j for k in h["kinds_fit"] for j in sp["fit"][k]])
                    kidx = np.array([i for i, k in enumerate(h["kinds_fit"]) for _ in sp["fit"][k]])
                    below = min(float(np.mean([100.0 * swf.at(float(d))["fired"][kidx == i].mean() for i in range(len(h["kinds_fit"]))])) for d in grid)
                hh.update(fit_mean_at_raise=h["fit"]["at_raise"]["mean"], fit_mean_at_0=h["fit"]["at_0"]["mean"], fit_mean_recomputed=fitm, smallest_fit_mean_below_the_raise=below,
                          one_float_below=h["fit"].get("one_float_below", {}).get("mean"), test=h["test"], test_recomputed=tst, rephrase=h["rephrase"], rephrase_recomputed=cov,
                          heldout_disjoint=h["heldout_disjoint"], heldout_recomputed=hd, n_edits_at_cap=h["n_edits_at_cap"], method_variant=Mh.variant, method_memo_path=Mh.memo,
                          rel_own_prompts_kept=h["rephrase"]["rel"], per_edit_n=len(h["per_edit"]["edits"]), digest_differs_from_base=(Mh.digest != M.digest) == (h["delta"] > 0))
                good = (abs(fitm - h["fit"]["at_raise"]["mean"]) < 1e-9 and fitm <= 10.0 and (below is None or below > 10.0)
                        and (h["delta"] == 0 or h["fit"]["one_float_below"]["mean"] > 10.0)
                        and all(abs(tst[k] - h["test"][k]["fired"]) < 1e-9 for k in tst) and all(abs(cov[k] - h["rephrase"][k]) < 1e-9 for k in cov)
                        and abs(hd - h["heldout_disjoint"]["fired"]) < 1e-9 and abs(h["rephrase"]["mean"] - 0.5 * (cov["tgen"] + cov["igen"])) < 1e-9
                        and hh["digest_differs_from_base"] and Mh.memo and len(h["test_rows"]["rows"]) == sum(len(sp["test"][k]) for k in S.HN))
                sub = S.h3(L, M, test_rows=sp["test"]["HN-I"][:50])
                good &= sub["delta"] == h["delta"] and sub["test"]["subset"]["n_rows"] == 50 and "HN-Q" not in sub["test"]
            else:
                good = h["delta"] is None and h["method"] is None
            m["h3"] = hh
            m_ok = (m["scale_0_is_the_gate"] and nd == 0 and nd_drop == 0 and m["drop_0_is_the_gate"] and m["own_prompts_candidate_at_raise_0.5"] == L.T
                    and m["curves_at_0_differing_from_gate_stats"] == 0 and m["curves_monotone_in_firing"] and all(c["check"] for c in cs) and good
                    and all(m["h3_split"][k] for k in ("fit_rows_even", "test_rows_odd", "partition")))
            m["ok"] = bool(m_ok)
            ok &= m_ok
            r[lab] = m
        # one global threshold on the feature of ANY gate object: the base gate handed in as an object gives the registered sweep
        a, b = fx.gate.sweep(L, "ncoord"), fx.gate.sweep(L, "x", gate=fx.methods.base_gate(L, "ncoord")[0])
        r["sweep_of_a_gate_object_equals_feature_sweep"] = all(np.array_equal(a["parts"][p][k], b["parts"][p][k]) for p in a["parts"] for k in ("rmax", "amax", "rown"))
        r["sweep_of_a_calibrated_gate_ignores_thresholds"] = all(np.array_equal(a["parts"][p]["rmax"], fx.gate.sweep(L, "x", gate=fx.methods.rescope(L, draw=0))["parts"][p]["rmax"]) for p in a["parts"])
        r["fixed_gate_is_a_pure_shift"] = bool(torch.equal(S.Scale(fx.methods.cosine(L, "default").obj).theta(0.1), fx.methods.cosine(L, "default").obj.theta() + 0.1))
        ok &= r["sweep_of_a_gate_object_equals_feature_sweep"] and r["sweep_of_a_calibrated_gate_ignores_thresholds"] and r["fixed_gate_is_a_pure_shift"]
        out["logs"][name] = r
        say(f"  sweeps {name}: " + ", ".join(f"{k} {'ok' if v.get('ok') else 'FAIL'}" for k, v in r.items() if isinstance(v, dict)))
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (4) row set
REQUIRED_ROWS = {"cosine": "cosine gate, two-block", "cosine_shared": "cosine gate, shared scale", "ball": "distance-ball gate, two-block", "router": "activation router, conformal",
                 "cosine_default": "cosine 0.65 at the last prompt position", "cosine_published": "cosine 0.65 at all positions", "zscore_default": "z-score gate, published setting",
                 "zscore": "z-score gate, conformal", "exact": "exact-match gate", "dual": "answer-match baseline", "rescope": "default method", "rescope_shared": "shared scale",
                 "rescope_null": "ReScope-N", "where_logonly": "Rescope-WHERE, log-only thresholds", "deployed": "deployed log", "preedit": "pre-edit model"}


def rowset():
    out, ok = {"required": REQUIRED_ROWS, "logs": {}}, True
    missing = [k for k in REQUIRED_ROWS if k not in fx.methods.E1_LABELS]
    out["missing_rows"] = missing
    out["gate_level_only"] = list(fx.methods.E1_GATE_LEVEL)
    with open(os.path.join(fx.COMMON, "USAGE.txt")) as f:
        usage = " ".join(f.read().split())
    out["undefined_calibrated_forms_documented"] = all(s in usage for s in ("NO CALIBRATED FORM", "exact-match gate", "answer-match baseline"))
    ok &= not missing and out["undefined_calibrated_forms_documented"]
    for name in DEV:
        L = fx.logs.open(name)
        rs = {M.label: M for M in fx.methods.e1_rowset(L, 0.05, 3, labels=("rescope", "rescope_shared", "rescope_null", "cosine", "cosine_shared", "ball", "rescope_pac"))}
        sids = {k: tuple(M.rec["draw"]["sids"]) for k, M in rs.items()}
        r = {"calibrated_rows": list(rs), "same_calibration_rows": len(set(sids.values())) == 1, "same_alpha_n_draw": len({(M.alpha, M.n, M.draw) for M in rs.values()}) == 1,
             "rules": {k: M.rec["cal"]["op"] for k, M in rs.items()}, "n_edits_capped": {k: M.rec["cal"]["n_edits_capped"] for k, M in rs.items()},
             "digests_distinct": len({M.digest for M in rs.values()}) == len(rs)}
        ok &= r["same_calibration_rows"] and r["same_alpha_n_draw"] and r["digests_distinct"]
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (5) calibration options
def _pac_exact(n2, a, d):
    """Independent exact computation with fractions: smallest k in 1 .. n2 with P[Bin(n2, a) >= n2 + 1 - k] >= 1 - d (n2 + 1: none)."""
    from fractions import Fraction
    a, d = Fraction(a).limit_denominator(10 ** 9), Fraction(d).limit_denominator(10 ** 9)
    pmf = [Fraction(math.comb(n2, i)) * a ** i * (1 - a) ** (n2 - i) for i in range(n2 + 1)]
    for k in range(1, n2 + 1):
        if sum(pmf[n2 + 1 - k:]) >= 1 - d:
            return k
    return n2 + 1


def calib():
    from scipy import stats as sps
    from rescope import ScopedLog, conformal_rank_exact, float_safe_threshold, split_blocks, synthetic_log
    out, ok = {}, True
    # ---- (i) the rank
    k = fx.methods.pac_rank(100, 0.05, 0.1)
    beta = [kk for kk in range(1, 101) if sps.beta.cdf(0.05, 100 + 1 - kk, kk) >= 0.9]
    grid = [(n2, a, d) for n2 in (20, 50, 100, 200, 500) for a in (0.01, 0.02, 0.05, 0.1, 0.2) for d in (0.05, 0.1, 0.2)]
    bad = [(n2, a, d) for n2, a, d in grid if fx.methods.pac_rank(n2, a, d) != _pac_exact(n2, a, d)]
    sc_bad = 0
    for n2, a, d in grid:
        kk = fx.methods.pac_rank(n2, a, d)
        ks = [x for x in range(1, n2 + 1) if sps.beta.cdf(a, n2 + 1 - x, x) >= 1 - d - 1e-12]
        sc_bad += int((ks[0] if ks else n2 + 1) != kk)
    out["rank"] = {"n2_100_alpha_0.05_delta_0.1": k, "exact_fractions": _pac_exact(100, 0.05, 0.1), "scipy_beta": beta[0], "default_rank": conformal_rank_exact(100, 0.05),
                   "grid": len(grid), "grid_differing_from_exact": len(bad), "grid_differing_from_scipy": sc_bad, "nothing_may_fire_n2_20_alpha_0.01": fx.methods.pac_rank(20, 0.01, 0.1) == 21}
    ok &= k == _pac_exact(100, 0.05, 0.1) == beta[0] == 99 and not bad and sc_bad == 0 and out["rank"]["nothing_may_fire_n2_20_alpha_0.01"]
    # ---- validity on exchangeable synthetic scores: realised false-firing probability of eta = the k-th smallest of n2 uniform scores
    g = np.random.default_rng(5)
    U = np.sort(g.random((40000, 100)), axis=1)
    rate = lambda kk: 1.0 - U[:, kk - 1]                                 # noqa: E731
    val = {"n_calibration_samples": 40000, "share_above_alpha_pac": float((rate(k) > 0.05).mean()), "share_above_alpha_one_rank_lower": float((rate(k - 1) > 0.05).mean()),
           "share_above_alpha_default_rank": float((rate(96) > 0.05).mean()), "delta": 0.1, "exact_pac": float(sps.binom.cdf(100 - k, 100, 0.05)),
           "exact_one_rank_lower": float(sps.binom.cdf(100 - k + 1, 100, 0.05)), "mean_rate_pac": float(rate(k).mean())}
    ok &= val["share_above_alpha_pac"] <= 0.1 and val["share_above_alpha_one_rank_lower"] > 0.1 and abs(val["share_above_alpha_pac"] - val["exact_pac"]) < 0.005
    # ---- the same through the whole constructor path on a synthetic scoped log with exchangeable keys
    log = synthetic_log(T=24, d=96, m=20, batch=4, seed=0)
    log = log[0] if isinstance(log, tuple) else log
    S0 = ScopedLog.from_log(log, role="last")
    S0 = S0.replace(guard_key=S0.guard_key[:0], guard_edit=S0.guard_edit[:0])
    gk = torch.Generator().manual_seed(7)
    share = {"pac": 0, "default": 0}
    n_rep, n_test = 300, 4000
    for rep in range(n_rep):
        Kc = torch.randn(200, 96, generator=gk, dtype=torch.float64)
        Kt = torch.randn(n_test, 96, generator=gk, dtype=torch.float64)
        Q = S0.scores(Kc)
        Sp, rp = fx.methods._cal_two_block_ext(S0, Q, 0.05, rep, (1, 1), True, "excess", None, 0.1)
        Sd, _ = fx.methods._cal_two_block(S0, Q, 0.05, rep, (1, 1), "max", True, "excess")
        share["pac"] += int(float(Sp.fires_on(Kt).any(1).double().mean()) > 0.05)
        share["default"] += int(float(Sd.fires_on(Kt).any(1).double().mean()) > 0.05)
    val["end_to_end"] = {"n_calibrations": n_rep, "n_test_keys": n_test, "share_of_calibrations_with_rate_above_alpha": {k_: v / n_rep for k_, v in share.items()}, "rank": rp["rank"]}
    ok &= share["pac"] / n_rep <= 0.1 and share["default"] / n_rep > 0.2 and rp["rank"] == 99
    out["validity"] = val
    out["logs"] = {}
    for name in DEV:
        L = fx.logs.open(name)
        r = {}
        base, Mp = fx.methods.rescope(L, draw=0), fx.methods.build("rescope_pac", L, draw=0)
        dr = fx.draws.calibration(L, 200, 0)
        Q = L.kc.coords(base.obj, dr.rows)
        lit = fx.sweeps.Scale(Mp.obj).lit
        b2 = Mp.rec["cal"]["block2"]
        r["pac"] = {"rank": Mp.rec["cal"]["rank"], "pac_rank": fx.methods.pac_rank(len(b2), 0.05, 0.1), "alpha": Mp.alpha, "variant": Mp.variant, "label": Mp.label,
                    "block2_firing_literal": int((Q[b2] >= lit[None, :]).any(1).sum()), "n2_minus_rank": len(b2) - Mp.rec["cal"]["rank"],
                    "thresholds_not_below_default": bool((lit >= fx.sweeps.Scale(base.obj).lit).all()), "same_draw": Mp.rec["draw"]["sids"] == base.rec["draw"]["sids"],
                    "same_baselines": bool(torch.equal(Mp.obj.det_b, base.obj.det_b)), "n_edits_capped": Mp.rec["cal"]["n_edits_capped"], "memo_path": Mp.memo,
                    "delta_0.2": fx.methods.rescope_pac(L, delta=0.2).rec["cal"]["rank"]}
        ok &= (r["pac"]["rank"] == r["pac"]["pac_rank"] == 99 and r["pac"]["block2_firing_literal"] == r["pac"]["n2_minus_rank"] and r["pac"]["thresholds_not_below_default"]
               and r["pac"]["same_draw"] and r["pac"]["same_baselines"] and Mp.alpha == 0.05 and r["pac"]["delta_0.2"] == fx.methods.pac_rank(100, 0.05, 0.2))
        # ---- (ii) external baselines
        ninf = torch.full((L.T,), -float("inf"), dtype=torch.float64)
        Mx0 = fx.methods.rescope(L, draw=0, baseline_x=ninf)
        gx = np.random.default_rng(11)
        x = torch.from_numpy(np.where(gx.random(L.T) < 0.5, gx.normal(0.2, 0.2, L.T), -np.inf))
        Mx = fx.methods.rescope(L, draw=0, baseline_x=x)
        i1, i2 = split_blocks(200, (1, 1), 0)
        Qn = Q.numpy()
        b = np.maximum(Qn[i1].max(0), x.numpy())
        sc2 = (Qn[i2] - b[None, :]).max(1)
        kk = conformal_rank_exact(len(i2), 0.05)
        eta = float(np.sort(sc2)[kk - 1])
        lit_ref = torch.from_numpy(float_safe_threshold(b, np.ones(L.T), eta))
        sx = fx.sweeps.Scale(Mx.obj)
        r["baseline_x"] = {"minus_inf_equals_default_bitwise": same_gate(Mx0.obj, base.obj), "variant": Mx.variant, "n_baselines_raised": Mx.rec["cal"]["baseline_x"]["n_baselines_raised"],
                           "baselines_equal_formula": bool(torch.equal(Mx.obj.det_b, torch.from_numpy(b))), "eta_equals_formula": Mx.rec["cal"]["eta"] == eta,
                           "literal_thresholds_equal_formula": bool(torch.equal(sx.lit, lit_ref)), "n1_n2": [Mx.rec["cal"]["n1"], Mx.rec["cal"]["n2"]], "n_edits_capped": Mx.rec["cal"]["n_edits_capped"],
                           "recalibrate_equals_constructor": bool(torch.equal(fx.methods.recalibrate(L, fx.methods.base_gate(L)[0], 0.05, 200, 0, baseline_x=x)[0].det_tau, Mx.obj.det_tau)),
                           "wrong_length_refused": raises(lambda: fx.methods.rescope(L, baseline_x=torch.zeros(3)), ValueError),
                           "with_pac": fx.methods.rescope_pac(L, baseline_x=x).rec["cal"]["rank"] == 99}
        try:                                                            # the feature package's own calibration of rescope_x (R5), if it is there
            if fx.FINAL + "/feat2" not in sys.path:
                sys.path.insert(0, fx.FINAL + "/feat2")
            import recomb
            S0b = fx.methods.base_gate(L)[0]
            xr, _ = recomb.own_baseline(L, S0b)
            Sr, _ = recomb.calibrate_two_block_x(S0b, coords=Q, x=xr, alpha=0.05, seed=dr.seed)
            Mr = fx.methods.rescope(L, draw=0, baseline_x=xr)
            r["baseline_x"]["equals_feature_package_recomb_bitwise"] = bool(torch.equal(Sr.det_tau, Mr.obj.det_tau) and torch.equal(Sr.det_b, Mr.obj.det_b))
            r["baseline_x"]["recomb_n_edits_capped"] = Mr.rec["cal"]["n_edits_capped"]
        except Exception as e:                                          # noqa: BLE001
            r["baseline_x"]["equals_feature_package_recomb_bitwise"] = f"not checked: {type(e).__name__}: {e}"
        bx = r["baseline_x"]
        ok &= (bx["minus_inf_equals_default_bitwise"] and bx["baselines_equal_formula"] and bx["eta_equals_formula"] and bx["literal_thresholds_equal_formula"] and bx["n1_n2"] == [100, 100]
               and bx["recalibrate_equals_constructor"] and bx["wrong_length_refused"] and bx["with_pac"] and bx["equals_feature_package_recomb_bitwise"] is not False)
        # ---- (iii) top-1 with capped edits: the other order of selection, against a brute-force recomputation
        t1 = {}
        sets = fx.gate.row_sets(L, benign=False)
        rows = sets["probe"] + sets["edit"] + sets["hn"]
        plain = fx.gate.evaluate(L, base, parts=("probe", "edit", "hn"))
        t1["default_output_unchanged"] = "top1" not in fx.metrics.gate_stats(L, plain) and "top1_differs" not in plain["probe"]["tf"]
        for lab, M in (("rescope_d0", base), ("cosine_d0", fx.methods.cosine(L, draw=0)), ("cosine_d1", fx.methods.cosine(L, draw=1)), ("cosine_default", fx.methods.cosine(L, "default"))):
            ev = fx.gate.evaluate(L, M, parts=("probe", "edit", "hn"), top1=True)
            rep = fx.metrics.gate_stats(L, ev)["top1"]
            g_ = fx.gate._cpu(M.obj)
            R = fx.gate.features(L, M, rows)
            th, key = g_.theta(), g_.rank_key(R)
            n_diff = 0
            for i in range(R.shape[0]):                                 # brute force, row by row
                c = (R[i] >= th).nonzero().flatten().tolist()
                lib = max(c, key=lambda j: (float(key[i, j]), -j)) if c else -1
                a = int(torch.argmax(key[i]))
                alt = a if a in c else -1
                n_diff += int(lib != alt)
            cap = (M.rec.get("cal") or {}).get("n_edits_capped")
            t1[lab] = {"n_sequences": rep["all"]["n"], "n_differs": rep["all"]["n_differs"], "brute_force": n_diff, "n_edits_capped": cap, "by_part": {k_: v["n_differs"] for k_, v in rep.items() if k_ != "all"},
                       "library_fires_only": sum(v.get("n_library_fires_only", 0) for k_, v in rep.items() if k_ != "all")}
            ok &= rep["all"]["n_differs"] == n_diff and rep["all"]["n"] == len(rows) and (n_diff == 0 or bool(cap))
        ok &= t1["default_output_unchanged"]
        r["top1"] = t1
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (6) read-outs of R7 (memo only; no model)
def readouts():
    Mx = fx.metrics
    out, ok = {"logs": {}}, True
    # synthetic rows: identities and the interval
    g = np.random.default_rng(3)
    dep = [{"kind": "edit", "metric": m, "edit": i, "exact": bool(g.random() < 0.5)} for m in ("rel", "tgen", "igen") for i in range(100)]
    met = [{"kind": "edit", "metric": r["metric"], "edit": r["edit"], "exact": bool(g.random() < 0.7)} for r in dep]
    kg = Mx.kept_gained(met, dep, ci=True)
    ident = all(kg[m]["n_kept"] + kg[m]["n_lost"] == kg[m]["n_deployed_exact"] and kg[m]["n_kept"] + kg[m]["n_gained"] == kg[m]["n_exact"] for m in kg)
    direct = fx.stats.bootstrap([float(r["kept"]) for r in Mx.kept_gained_rows(met, dep) if r["metric"] == "rel"], list(range(100)), scale=100.0)
    d2rows = [{"pre_correct": bool(i % 3), "D2": bool(g.random() < 0.2), "fired": False, "query_id": i // 2} for i in range(300)]
    d2 = Mx.d2_stats(d2rows, ci=True)
    sub = [r for r in d2rows if r["pre_correct"]]
    d2_direct = fx.stats.bootstrap([float(r["D2"]) for r in sub], [r["query_id"] for r in sub], scale=100.0)
    gr = Mx.grid({0: met, 1: met[::-1]}, "exact", unit="edit", where=lambda r: r["metric"] == "rel")
    out["synthetic"] = {"identities": ident, "kept_ci_equals_direct_bootstrap": kg["rel"]["kept"] == direct, "d2_plain_unchanged": Mx.d2_stats(d2rows)["D2"] == Mx.pct(r["D2"] for r in sub),
                        "d2_ci_equals_cluster_bootstrap": d2["D2"] == d2_direct, "d2_units": d2["D2"]["n_units"], "d2_rows": d2["n_pre_correct"],
                        "grid_shape": list(gr["num"].shape), "grid_rows_equal": bool(np.array_equal(gr["num"][0], gr["num"][1])),
                        "grid_other_units_refused": raises(lambda: Mx.grid({0: met, 1: met[:50]}, "exact", unit="edit"), ValueError),
                        "missing_deployed_row_refused": raises(lambda: Mx.kept_gained(met, dep[:10]), KeyError)}
    ok &= all(v for k, v in out["synthetic"].items() if isinstance(v, bool)) and d2["D2"]["n_units"] < d2["n_pre_correct"] and out["synthetic"]["grid_shape"] == [2, 100]
    for name in DEV:
        L = fx.logs.open(name)
        ses = fx.behave.session(L, None)
        r = {}
        try:
            rm, rd = ses.edits(fx.methods.rescope(L, draw=0)), ses.edits(fx.methods.deployed(L))
            kg = Mx.kept_gained(rm, rd)
            es_m, es_d = Mx.edit_stats(rm), Mx.edit_stats(rd)
            r["kept_gained"] = {m: {k: kg[m][k] for k in ("n", "n_deployed_exact", "n_exact", "n_kept", "n_gained", "n_lost", "kept", "gained")} for m in kg}
            r["consistent_with_edit_stats"] = all(abs(kg[m]["kept"] + kg[m]["gained"] - es_m[m]["exact"]) < 1e-9 and abs(100.0 * kg[m]["n_deployed_exact"] / kg[m]["n"] - es_d[m]["exact"]) < 1e-9 for m in kg)
            ok &= r["consistent_with_edit_stats"]
            # blocklist: listed queries take the pre-edit row, the others the deployed row
            hd = L.pool("heldout_disjoint")
            rep = hd[:25]
            bl, dep_r, pre_r = ses.blocklist(rep, hd), ses.tf(fx.methods.deployed(L), hd), ses.tf(fx.methods.preedit(L), hd)
            listed = {L.index[j]["query_id"] for j in rep}
            good = all((b["D1"] == (p["D1"] if L.index[j]["query_id"] in listed else d["D1"])) and b["blocked"] == (L.index[j]["query_id"] in listed) for j, b, d, p in zip(hd, bl, dep_r, pre_r))
            r["blocklist"] = {"n_rows": len(hd), "n_reported": len(rep), "n_blocked": sum(b["blocked"] for b in bl), "rows_consistent": good,
                              "D1_deployed": Mx.pct(d["D1"] for d in dep_r), "D1_blocklist": Mx.pct(b["D1"] for b in bl), "D1_blocked_rows": Mx.pct(b["D1"] for b in bl if b["blocked"])}
            ok &= good and r["blocklist"]["n_blocked"] >= 25 and r["blocklist"]["D1_blocked_rows"] == 0.0
        except fx.behave.MemoMiss as e:
            r["not_in_memo"] = str(e)
            ok = False
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (7) crossed bootstrap, Holm
def _reference_crossed(Y, n_boot, seed):
    """The naive definition (a loop): draws and units resampled independently, the mean over the resampled grid."""
    Y = np.asarray(Y, dtype=float)
    D_, U_ = Y.shape
    rng = np.random.default_rng(seed)
    st = np.empty(n_boot)
    for b in range(n_boot):
        st[b] = Y[np.ix_(rng.integers(0, D_, D_), rng.integers(0, U_, U_))].mean()
    return float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def stats2():
    ST = fx.stats
    out = {}
    D_, U_, sa, sb, se_ = 20, 300, 0.05, 0.4, 0.2
    true_sd = math.sqrt(sa ** 2 / D_ + sb ** 2 / U_ + se_ ** 2 / (D_ * U_))
    n_exp, cov, widths, wu, wref = 400, 0, [], [], []
    for s in range(n_exp):
        g = np.random.default_rng(9000 + s)
        Y = 0.5 + g.normal(0, sa, D_)[:, None] + g.normal(0, sb, U_)[None, :] + g.normal(0, se_, (D_, U_))
        b = ST.crossed_bootstrap(Y, n_boot=500 if s else 2000, seed=s)
        cov += int(b["ci_lo"] <= 0.5 <= b["ci_hi"])
        widths.append(b["ci_hi"] - b["ci_lo"])
        if s < 40:
            u = ST.bootstrap(Y.mean(0), n_boot=500, seed=s)
            wu.append(u["ci_hi"] - u["ci_lo"])
            lo, hi = _reference_crossed(Y, 500, s)
            wref.append(hi - lo)
    one = ST.crossed_bootstrap(Y, seed=0)
    out["crossed"] = {"n_experiments": n_exp, "coverage_95": cov / n_exp, "mean_width": float(np.mean(widths)), "analytic_width": 2 * 1.96 * true_sd,
                      "units_only_width": float(np.mean(wu)), "reference_loop_width": float(np.mean(wref)), "width_over_reference": float(np.mean(widths[:40]) / np.mean(wref)),
                      "value_is_grand_mean": abs(one["value"] - Y.mean()) < 1e-12, "n_boot_default": one["n_boot"], "n_draws": one["n_draws"], "n_units": one["n_units"],
                      "deterministic": ST.crossed_bootstrap(Y, seed=0) == one}
    c = out["crossed"]
    ok = (0.91 <= c["coverage_95"] <= 0.99 and abs(c["mean_width"] / c["analytic_width"] - 1) < 0.15 and c["units_only_width"] < c["mean_width"] and abs(c["width_over_reference"] - 1) < 0.1
          and c["value_is_grand_mean"] and c["n_boot_default"] == 2000 and c["deterministic"])
    # ratio form (clusters of unequal size) and one draw == the unit bootstrap of a ratio
    g = np.random.default_rng(1)
    den = g.integers(1, 4, 150).astype(float)
    num = g.binomial(den.astype(int), 0.3).astype(float)
    r1 = ST.crossed_bootstrap(num[None, :], den[None, :], scale=100.0)
    r2 = ST.bootstrap_ratio(num, den, list(range(150)), scale=100.0)
    out["ratio"] = {"value": r1["value"], "value_direct": 100.0 * num.sum() / den.sum(), "width": r1["ci_hi"] - r1["ci_lo"], "unit_bootstrap_width": r2["ci_hi"] - r2["ci_lo"],
                    "incomplete_grid_refused": raises(lambda: ST.crossed_bootstrap([[1.0, float("nan")]]), ValueError)}
    ok &= abs(r1["value"] - out["ratio"]["value_direct"]) < 1e-9 and abs(out["ratio"]["width"] / out["ratio"]["unit_bootstrap_width"] - 1) < 0.1 and out["ratio"]["incomplete_grid_refused"]
    # paired version: coverage of a known difference, the pairing narrows the interval, rejection rate under the null
    covd, rej, wp, wunp = 0, 0, [], []
    for s in range(300):
        g = np.random.default_rng(100 + s)
        a_d, b_u = g.normal(0, sa, D_)[:, None], g.normal(0, sb, U_)[None, :]
        A = 0.6 + a_d + b_u + g.normal(0, se_, (D_, U_))
        B = 0.5 + a_d + b_u + g.normal(0, se_, (D_, U_))
        dd = ST.crossed_bootstrap_diff(A, B, n_boot=500, seed=s)
        covd += int(dd["ci_lo"] <= 0.1 <= dd["ci_hi"])
        wp.append(dd["ci_hi"] - dd["ci_lo"])
        n0 = ST.crossed_bootstrap_diff(A - 0.1, B, n_boot=500, seed=s)
        rej += int(n0["p_two_sided"] < 0.05)
        if s < 30:
            x, y = ST.crossed_bootstrap(A, n_boot=500, seed=s), ST.crossed_bootstrap(B, n_boot=500, seed=s + 1)
            wunp.append(math.hypot(x["ci_hi"] - x["ci_lo"], y["ci_hi"] - y["ci_lo"]))
    out["paired"] = {"n_experiments": 300, "coverage_95": covd / 300, "mean_width": float(np.mean(wp)), "analytic_width": 2 * 1.96 * math.sqrt(2 * se_ ** 2 / (D_ * U_)),
                     "unpaired_width": float(np.mean(wunp)), "rejection_rate_under_the_null": rej / 300, "value_is_difference": abs(dd["value"] - (A.mean() - B.mean())) < 1e-12,
                     "other_shape_refused": raises(lambda: ST.crossed_bootstrap_diff(A, B[:, :10]), ValueError)}
    p = out["paired"]
    ok &= 0.9 <= p["coverage_95"] <= 1.0 and p["mean_width"] < 0.5 * p["unpaired_width"] and p["rejection_rate_under_the_null"] <= 0.1 and p["value_is_difference"] and p["other_shape_refused"]
    # Holm
    h = ST.holm([0.01, 0.04, 0.03, 0.005])
    hn = ST.holm({"a": 0.2, "b": 0.001, "c": 0.02})
    known = np.allclose(h["adjusted"], [0.03, 0.06, 0.06, 0.02]) and h["reject"] == [True, False, False, True] and h["order"] == [3, 0, 2, 1]
    try:
        from statsmodels.stats.multitest import multipletests
        g = np.random.default_rng(0)
        sm = all(np.allclose(ST.holm(list(pv))["adjusted"], multipletests(pv, method="holm")[1]) for pv in g.random((200, 7)) ** 3)
    except Exception as e:                                              # noqa: BLE001
        sm = f"statsmodels not available ({type(e).__name__})"
    g = np.random.default_rng(2)
    fwer = float(np.mean([any(ST.holm(list(g.random(6)))["reject"]) for _ in range(4000)]))          # six true nulls: family-wise error <= alpha
    out["holm"] = {"known_example": bool(known), "named": hn["adjusted"], "named_ok": np.allclose([hn["adjusted"][k] for k in "abc"], [0.2, 0.003, 0.04]) and hn["order"] == ["b", "c", "a"],
                   "equals_statsmodels_on_200_families": sm, "family_wise_error_under_six_true_nulls": fwer, "monotone": all(x <= 1 for x in h["adjusted"]),
                   "bad_p_refused": raises(lambda: ST.holm([0.1, 1.2]), ValueError)}
    ok &= known and bool(out["holm"]["named_ok"]) and sm is not False and fwer <= 0.06 and out["holm"]["bad_p_refused"]
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (8) registry of the reduced programme, pools, masks
def pools():
    P, MK = fx.pools, fx.masks
    out, ok = {"programme": {}, "logs": {}}, True
    prog = fx.logs.programme()
    out["programme_roles"] = {r: sorted(n for n, x in prog.items() if x == r) for r in sorted(set(prog.values()))}
    want = {"TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw",
            "TB-L500-memitw", "TB-L1000-memitw"}
    ok &= set(fx.logs.programme("unseen")) == want and not any(fx.logs.REGISTRY[n]["cut_by_R19"] for n in prog) and len(prog) == 16
    for name in prog:                                                   # mechanics only on every log that is not a development log: nothing of it is kept but counts
        r = {"role": prog[name], "exists": fx.logs.exists(name)}
        if r["exists"]:
            L = fx.logs.open(name)
            M = fx.methods.rescope(L, draw=0)
            ev = fx.gate.evaluate(L, M, parts=("probe", "edit", "hn"), top1=True)
            later = MK.later_edits(L)
            keep = MK.keep(L, L.pool("heldout_disjoint"))
            dq = fx.draws.calibration(L, 200, 0, mask="qdisjoint")
            fq = MK.question_disjoint(L)
            pt = {"n": len(P.decide(L, M, "BEN-T")["fired"])}              # the pool rows are part of the one gate-level pass; no rate is formed
            pm = {"n": len(P.decide(L, M, "MME-T")["fired"])}
            h3_ok = None
            if prog[name] in ("development", "working_point") and L.rows("hn", valid=True):      # read-outs only where numbers may be looked at
                h = fx.sweeps.h3(L, M)
                h3_ok = bool(h["reachable"] in (True, False) and (not h["reachable"] or h["method"].memo))
            r.update(opened=True, T=L.T, built=bool(M.key), gate_pass_rows=sum(len(d["tf"]["rows"]) for d in ev.values()), near_scope_flag=callable(L.near_edit),
                     n_later_edits=len(later), later_from=MK.later_from(L), later_ok=later == [i for i in range(L.T) if i >= MK.later_from(L)],
                     n_stream_edits=MK.n_stream_edits(L), masked_draw_ok=len(dq.rows) == 200 and all(fq(L.index[j]) for j in dq.rows) and dq.rec["mask"] == "qdisjoint",
                     mask_shape_ok=keep.shape[0] == len(L.pool("heldout_disjoint")), pools_ok=pt["n"] == len(P.get(L, "BEN-T").rows) and pm["n"] == len(P.get(L, "MME-T").rows),
                     h3_mechanics=h3_ok)
            ok &= r["built"] and r["later_ok"] and r["masked_draw_ok"] and r["mask_shape_ok"] and r["pools_ok"] and h3_ok is not False
            if name not in DEV:
                fx.logs._OPEN.pop(name, None)
                fx.methods._VALUES.pop(name, None)
                for k in [k for k in fx.methods._BASE if k[0] == name]:
                    fx.methods._BASE.pop(k)
                fx.gate.clear_cache()
        else:
            ok = False
        out["programme"][name] = r
    for name in DEV:
        L = fx.logs.open(name)
        parts = P.benign_parts(L)
        cnt = {p: sum(v == p for v in parts.values()) for p in P.BEN}
        mm = P.mme(L.backbone)
        r = {"benign_parts": cnt, "benign_parts_file_counts": fx.io.jload(os.path.join(P.PARTS_DIR, f"{L.backbone}.json"))["n_rows"],
             "pools": {n: {"n": len(P.get(L, n).rows), "which": P.get(L, n).which, "test_only": P.get(L, n).test_only} for n in P.NAMES},
             "mme_rows": {p: len(mm.rows(p)) for p in P.MME}, "available": P.available(L.backbone)}
        ok &= cnt == r["benign_parts_file_counts"] == {"BEN-W": 600, "BEN-S": 1375, "BEN-T": 1025} and r["mme_rows"] == {"MME-C": 1188, "MME-T": 1186}
        ok &= r["pools"]["BEN-T"]["n"] == 1025 and r["pools"]["BEN-T"]["which"] == "decode" and r["pools"]["MME-T"]["which"] == "tf" and r["pools"]["MME-T"]["test_only"]
        ok &= not (set(P.get(L, "BEN-T").rows) & set(P.get(L, "BEN-S").rows)) and not (set(P.get(L, "BEN-T").rows) & set(P.get(L, "BEN-W").rows))
        # firing on a pool == the library's own decision on the same keys (both gates; MME in its own cache)
        fr = {}
        for lab, M in (("rescope", fx.methods.rescope(L, draw=0)), ("cosine", fx.methods.cosine(L, draw=0))):
            for pn in ("BEN-T", "MME-T", "BEN-S", "MME-C"):
                Pl = P.get(L, pn)
                d = P.decide(L, M, Pl)
                ref = (L.kc if Pl.kc is None else Pl.kc).decide(M.obj, Pl.which, rows=Pl.rows)
                st = P.firing(L, M, pn, rows_out=True)
                same = bool(np.array_equal(d["fired"], ref["fired"]) and np.array_equal(d["top_edit"], ref["top_edit"]))
                fr[f"{lab}/{pn}"] = {"n": st["n"], "fired": st["fired"], "equals_library": same, "far_n": st["far"]["n"], "related_n": st["related"]["n"],
                                     "n_units": len(set(st["rows"]["unit"])), "n_edits_capped": st["n_edits_capped"]}
                ok &= same and st["n"] == len(Pl.rows) and st["far"]["n"] + st["related"]["n"] == st["n"] and abs(st["fired"] - 100.0 * float(np.mean(ref["fired"]))) < 1e-9
        r["firing"] = fr
        Mr = fx.methods.rescope(L, draw=0)
        mk_ = P.firing(L, Mr, "BEN-T", mask=MK.question_disjoint(L))
        r["masked_pool"] = {"n": mk_["n"], "n_masked_out": mk_["n_masked_out"], "by_category_mme": len(P.firing(L, Mr, "MME-T")["by_category"]),
                            "zscore_like_gate_refused": raises(lambda: P.decide(L, fx.methods.rescope(L, role="all", calibration="none"), "BEN-T"), NotImplementedError)}
        ok &= mk_["n"] + mk_["n_masked_out"] == 1025 and r["masked_pool"]["by_category_mme"] == 14 and r["masked_pool"]["zscore_like_gate_refused"]
        # masks
        hd = L.pool("heldout_disjoint")
        keep = MK.keep(L, hd)
        qs = MK.stream_questions(L)
        near_all = [j for j in hd if L.near_edit(j, rephrase=True) is not None]
        dq = fx.draws.calibration(L, 200, 0, mask="qdisjoint")
        nq = fx.methods.rescope_null(L, draw=0, mask="qdisjoint")
        r["masks"] = {"n_stream_edits": MK.n_stream_edits(L), "n_stream_question_strings": len(qs), "later_edits": len(MK.later_edits(L)), "later_from": MK.later_from(L),
                      "heldout_disjoint_kept": int(keep.sum()), "heldout_disjoint_n": len(hd), "near_scope_rows_of_this_log_all_masked_out": not any(keep[hd.index(j)] for j in near_all),
                      "masked_draw_pool": dq.rec["n_pool"], "unmasked_pool": len(L.pool("repair")), "masked_draw_rows_pass": all(MK.question_disjoint(L)(L.index[j]) for j in dq.rows),
                      "masked_draws_nested": set(fx.draws.calibration(L, 100, 0, mask="qdisjoint").rows) <= set(dq.rows),
                      "masked_null_variant": nq.variant, "masked_null_sample_disjoint": nq.rec["null_repair"]["n_queries_shared_with_calibration"] == 0,
                      "masked_null_rows_pass": all(MK.question_disjoint(L)(L.index[L.kc.by_sid(s)]) for s in nq.rec["null_repair"]["sids"]),
                      "probe_where": len(L.rows("probe", pool="heldout_disjoint", valid=True, where=MK.question_disjoint(L))) == int(keep.sum())}
        m = r["masks"]
        ok &= (m["n_stream_edits"] == {"Q": 2000, "L": 1000}[L.family] and m["later_edits"] == 0 and m["near_scope_rows_of_this_log_all_masked_out"] and m["masked_draw_rows_pass"]
               and m["masked_draws_nested"] and m["masked_null_sample_disjoint"] and m["masked_null_rows_pass"] and m["probe_where"] and m["masked_draw_pool"] < m["unmasked_pool"]
               and "mask=qdisjoint" in nq.variant)
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (9) prefix logs (CPU part; after the GPU build)
def prefix():
    PF = fx.prefix
    out, ok = {"logs": {}, "names": fx.logs.names("prefix"), "check_names": fx.logs.names("prefix_check")}, True
    ok &= len(out["names"]) == 6 and len(out["check_names"]) == 2
    for parent in DEV:
        Lp = fx.logs.open(parent)
        for k in PF.KS + (PF.CHECK_K,):
            nm = PF.name(parent, k)
            r = {"complete": PF.complete(nm), "role": fx.logs.info(nm)["role"]}
            if not r["complete"]:
                ok = False
                out["logs"][nm] = r
                continue
            L = fx.logs.open(nm)
            d = PF.describe(nm)
            E = PF.edit_log(nm)
            M = fx.methods.rescope(L, draw=0)
            hn = L.rows("hn", valid=True)
            r.update(T=L.T, n_batches=int(L.log.batch.max()) + 1, factors_are_parent_rows=bool(torch.equal(L.log.U, Lp.log.U[:k]) and torch.equal(L.log.W, Lp.log.W[:k]) and torch.equal(L.log.K, Lp.log.K[:k])),
                     uids_are_stream_prefix=L.uids == Lp.uids[:k], library_edit_log_T=int(E.T), n_probe_rows=len(L.probe_rows), pools_equal_parent=L.pool("repair") == Lp.pool("repair") and L.pool("heldout_disjoint") == Lp.pool("heldout_disjoint"),
                     hard_negatives_are_those_of_its_edits=set(hn) == {j for j in Lp.rows("hn", valid=True) if Lp.own(j) < k and (Lp.index[j].get("edit_b") is None or int(Lp.index[j]["edit_b"]) < k)},
                     n_hard_negatives=len(hn), values_are_parent_rows=bool(torch.equal(fx.methods.released_values(L)[0], fx.methods.released_values(Lp)[0][:k])),
                     default_method_builds=bool(M.key) and M.n == 200, calibration_rows_equal_parent=M.rec["draw"]["sids"] == fx.methods.rescope(Lp, draw=0).rec["draw"]["sids"],
                     n_D1_repair_complaints=len(fx.draws._repair_complaints(L)), described_repair_complaints=d["deployed"]["repair"]["n_D1"],
                     load_checks=all(d["checks"]["harness_load_checks"].get(c) for c in ("WE_float_minus_W0_equals_stored_composite", "C_equals_stored_composite", "rollback_empty_equals_WE", "factor_uids_match_stream", "probe_uids_match_records")),
                     build_hook_mismatch=d["checks"]["hook_mismatch"], deployed={x: d["deployed"][x]["exact"] for x in ("rel", "tgen", "igen")}, D1_heldout_disjoint=d["deployed"]["heldout_disjoint"]["D1"])
            good = (r["factors_are_parent_rows"] and r["uids_are_stream_prefix"] and r["library_edit_log_T"] == k and r["pools_equal_parent"] and r["hard_negatives_are_those_of_its_edits"]
                    and r["values_are_parent_rows"] and r["default_method_builds"] and r["calibration_rows_equal_parent"] and r["n_D1_repair_complaints"] == r["described_repair_complaints"]
                    and r["load_checks"] and r["build_hook_mismatch"] == 0 and d["checks"]["factors_equal_parent_rows"] and d["checks"]["WE_equals_rule"])
            try:
                if fx.FINAL + "/feat2" not in sys.path:
                    sys.path.insert(0, fx.FINAL + "/feat2")
                import recomb
                lr = recomb.log_rows(L)
                r["recombinations"] = {"n_rows": len(lr.idx), "owners_in_prefix": bool((np.asarray(lr.edit) < k).all())}
                good &= r["recombinations"]["owners_in_prefix"]
            except Exception as e:                                      # noqa: BLE001
                r["recombinations"] = f"not checked: {type(e).__name__}: {e}"
            if k == PF.CHECK_K:                                         # all batches: the development log itself
                ab = d["checks"]["all_batches"]
                Mp = fx.methods.rescope(Lp, draw=0)
                r["all_batches"] = dict(ab, thresholds_equal_development_log=bool(torch.equal(M.obj.det_tau, Mp.obj.det_tau)), digest_equal=M.digest == Mp.digest,
                                        stored_edit_exact={x: d["deployed"][x]["exact"] for x in ("rel", "tgen", "igen")})
                r["all_batches"]["batch8_metrics_equal_parent_stored"] = d["checks"]["batch8_metrics_equal_parent_stored"]
                good &= (ab["composite_equals_stored_composite_bitwise"] and ab["WE_equals_development_WE_bitwise"] and ab["n_D1_flags_differing_from_stored_records"] == 0
                         and r["all_batches"]["thresholds_equal_development_log"] and r["all_batches"]["digest_equal"] and all(d["checks"]["batch8_metrics_equal_parent_stored"].values()))
            r["ok"] = bool(good)
            ok &= good
            out["logs"][nm] = r
            fx.logs._OPEN.pop(nm, None)
    out["table_exists"] = os.path.exists(PF.TABLE)
    out["ok"] = bool(ok and out["table_exists"])
    return out


# ----------------------------------------------------------------------------- (10) masking, content digest
def maint():
    from rescope import least_repair, witness_repair
    out, ok = {"logs": {}}, True
    for name in DEV:
        L = fx.logs.open(name)
        r = {}
        sets = fx.gate.row_sets(L, benign=False)
        rows = sets["probe"] + sets["edit"] + sets["hn"]
        for lab, M in (("rescope", fx.methods.rescope(L, draw=0)), ("cosine", fx.methods.cosine(L, draw=0))):
            M0 = fx.methods.mask_edits(L, M, [])
            R = [3, 17, 18, 40, 77]
            Mm = fx.methods.mask_edits(L, M, R)
            g, gm = M.obj, Mm.obj
            keep = torch.ones(L.T, dtype=torch.bool)
            keep[R] = False
            never = 0
            for w in ("tf", "prompt", "decode", "steps"):
                d = fx.gate.decide(L, Mm, rows, w)
                never += int(np.isin(d["top_edit"], R).sum())
                if w == "steps":
                    never += int(np.isin(d["step_top"], R).sum())
            own_keys = gm.fires_on(gm.det_key[:L.T])                    # the stored solver keys of every edit as queries
            before = fx.gate.decide(L, M, rows)
            after = fx.gate.decide(L, Mm, rows)
            untouched = ~np.isin(before["top_edit"], R)                 # sequences on which no masked edit was released: nothing may change
            # top-1 selection is the one way a sequence can change: only where a masked edit WAS released
            changed = (before["fired"] != after["fired"]) | (before["top_edit"] != after["top_edit"])
            m = {"empty_set_same_object": M0.obj is M.obj, "empty_set_digest_equal": M0.digest == M.digest, "empty_set_gate_bitwise": same_gate(M0.obj, M.obj),
                 "masked_thresholds_inf": bool(torch.isinf(gm.det_tau[R]).all() and (gm.det_tau[R] > 0).all()), "other_thresholds_bitwise": bool(torch.equal(gm.det_tau[keep], g.det_tau[keep])),
                 "detectors_values_baselines_unchanged": bool(gm.det_w is g.det_w and gm.values is g.values and ((gm.det_b is None and g.det_b is None) or gm.det_b is g.det_b)),
                 "n_releases_of_a_masked_edit_all_positions": never, "masked_edit_candidate_on_any_stored_key": int(own_keys[:, R].sum()),
                 "guards_of_masked_edits_dropped": not bool(torch.isin(gm.guard_edit, torch.tensor(R)).any()), "n_guards": [int(g.guard_edit.numel()), int(gm.guard_edit.numel())],
                 "n_sequences_changed": int(changed.sum()), "n_changed_without_a_masked_release_before": int((changed & untouched).sum()),
                 "n_other_edit_released_instead": int((np.isin(before["top_edit"], R) & after["fired"]).sum()),
                 "digest_differs": Mm.digest != M.digest, "variant": Mm.variant, "memo_path": Mm.memo, "own_prompts_not_released": Mm.rec["own_prompts"]["n_not_released"],
                 "mask_twice_equals_union": fx.methods.mask_edits(L, fx.methods.mask_edits(L, M, R[:2]), R[2:]).digest == Mm.digest,
                 "out_of_range_refused": raises(lambda: fx.methods.mask_edits(L, M, [L.T]), ValueError)}
            # a later raise-only operator of the library leaves the masked edits masked (no guard pulls them back)
            fire = [j for j, f in zip(L.pool("repair"), fx.gate.decide(L, Mm, L.pool("repair"))["fired"]) if f][:2]
            if fire:
                g2 = least_repair(gm, [L.kc.keys("tf", [j]) for j in fire])[0]
                m["masked_after_least_repair"] = bool(torch.isinf(g2.det_tau[R]).all())
                ok &= m["masked_after_least_repair"]
            # the raised scale of a masked gate keeps the mask
            m["masked_after_raise"] = bool(torch.isinf(fx.sweeps.Scale(gm).theta(0.3)[R]).all())
            good = (m["empty_set_same_object"] and m["empty_set_digest_equal"] and m["masked_thresholds_inf"] and m["other_thresholds_bitwise"] and m["detectors_values_baselines_unchanged"]
                    and never == 0 and m["masked_edit_candidate_on_any_stored_key"] == 0 and m["guards_of_masked_edits_dropped"] and m["n_changed_without_a_masked_release_before"] == 0
                    and m["digest_differs"] and m["mask_twice_equals_union"] and m["out_of_range_refused"] and m["masked_after_raise"] and Mm.memo)
            ok &= good
            r[lab] = m
        # an ADDED detector of a masked edit is masked as well (why-not repair in detector mode adds one)
        M = fx.methods.rescope(L, draw=0)
        e = 5
        wkey = L.kc.keys("tf", [L.edit_rows["tgen"][e]])
        Sw, wrec = witness_repair(M.obj, e, wkey, mode="detector")
        Mw = fx.methods.wrap(L, Sw, "rescope+witness")
        Mwm = fx.methods.mask_edits(L, Mw, [e])
        dw = fx.gate.decide(L, Mw, [L.edit_rows["tgen"][e]])
        dm = [fx.gate.decide(L, Mwm, rows, w) for w in ("tf", "prompt", "decode")]
        r["added_detector"] = {"n_detectors": int(Sw.D), "witness_released_before_mask": bool(dw["own_top"][0]), "detectors_of_the_edit": int((Sw.det_edit == e).sum()),
                               "all_its_thresholds_inf": bool(torch.isinf(Mwm.obj.det_tau[Sw.det_edit == e]).all()), "n_releases_after_mask": int(sum(np.isin(d["top_edit"], [e]).sum() for d in dm)),
                               "n_candidate_after_mask": int(Mwm.obj.edit_fires(torch.cat([wkey.double(), Sw.det_key[:L.T]]))[:, e].sum())}
        a = r["added_detector"]
        ok &= a["n_detectors"] == L.T + 1 and a["witness_released_before_mask"] and a["detectors_of_the_edit"] == 2 and a["all_its_thresholds_inf"] and a["n_releases_after_mask"] == 0 and a["n_candidate_after_mask"] == 0
        # ---- content digest (review finding 2): wrapped objects that differ only in one component
        g = M.obj
        th2 = g.det_tau.clone()
        th2[0] = th2[0] + 1e-3
        b2 = g.det_b.clone()
        b2[1] += 1e-6
        v2 = g.values.clone()
        v2[2, 0] += 1e-9
        w2 = g.det_w.clone()
        w2[3, 0] = torch.nextafter(w2[3, 0], torch.tensor(float("inf"), dtype=torch.float64))
        var = {"same": fx.methods.wrap(L, g, "a"), "same_other_label": fx.methods.wrap(L, g.replace(), "b", rec={"note": "x"}), "thresholds": fx.methods.wrap(L, g.replace(det_tau=th2), "a"),
               "baselines": fx.methods.wrap(L, g.replace(det_b=b2), "a"), "values": fx.methods.wrap(L, g.replace(values=v2), "a"), "detectors": fx.methods.wrap(L, g.replace(det_w=w2), "a"),
               "eta": fx.methods.wrap(L, g.replace(eta=1.0 + 1e-12), "a"), "response": fx.methods.wrap(L, g.replace(response="gain"), "a"), "role": fx.methods.wrap(L, g.replace(role="last+answer"), "a"),
               "fire": fx.methods.wrap(L, g.replace(fire="all"), "a")}
        dg = {k: v.digest for k, v in var.items()}
        W = torch.zeros(4, 5, dtype=torch.bfloat16)
        W2 = W.clone()
        W2[0, 0] = 1.0
        r["digest"] = {"equal_content_equal_digest": dg["same"] == dg["same_other_label"] == M.digest, "n_variants": len(dg) - 2, "all_variants_differ": len(set(dg.values())) == len(dg) - 1,
                       "keys_of_baseline_variants_equal (the finding)": var["baselines"].key == var["same"].key,
                       "thresholds_only": dg["thresholds"] != dg["same"], "weights": fx.methods.content_digest(W) != fx.methods.content_digest(W2) and fx.methods.content_digest(W) == fx.methods.content_digest(W.clone()),
                       "rebuilt_method_same_digest": fx.methods.rescope(L, draw=4).digest == fx.methods.rescope(L, draw=4).digest,
                       "deployed_differs_from_preedit": fx.methods.deployed(L).digest != fx.methods.preedit(L).digest,
                       "dedupe": {k: len(v) for k, v in fx.methods.dedupe([var["same"], var["same_other_label"], var["thresholds"]]).items()},
                       "e1_rowset_digests_distinct": len({x.digest for x in fx.methods.e1_rowset(L, 0.05, 0)}) == len(fx.methods.e1_rowset(L, 0.05, 0))}
        dd = r["digest"]
        ok &= (dd["equal_content_equal_digest"] and dd["all_variants_differ"] and dd["thresholds_only"] and dd["weights"] and dd["rebuilt_method_same_digest"] and dd["deployed_differs_from_preedit"]
               and sorted(dd["dedupe"].values()) == [1, 2] and dd["e1_rowset_digests_distinct"])
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


TESTS = {"null": null, "samples": samples, "sweeps": sweeps, "rowset": rowset, "calib": calib, "readouts": readouts, "stats2": stats2, "pools": pools, "prefix": prefix, "maint": maint}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(TESTS)
    for nm in names:
        t0 = time.time()
        res = TESTS[nm]()
        res.update(test=nm, seconds=time.time() - t0, code=fx.code_fingerprint(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        fx.io.jdump(res, os.path.join(OUT, f"ext_cpu_{nm}.json"), indent=1)
        say(f"{nm}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds']:.0f}s)")
