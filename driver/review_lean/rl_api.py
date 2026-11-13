"""Lean review, items 8 (sanity numbers), 9 (every name the task texts ask for: one call each on a development log) and 10 (attributes of
100 sequences against the rendered prompts).

    python rl_api.py <development log>                                   CPU calls + sanity numbers        -> out/api_cpu_<log>.json
    gq run --mem ... -- python rl_api.py <development log> gpu           behavioural calls on tiny inputs  -> out/api_gpu_<log>.json
Every call sits in its own try: the record holds "ok" or the exception.  Numbers are looked at on development logs only; the one unseen
log opened here (mechanics of the frozen row set: open, build, one gate-level pass) prints and stores no metric.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import hashlib
import json
import os
import re
import sys
import time
import traceback

import rl_common as C

NAME = sys.argv[1]
GPU = len(sys.argv) > 2 and sys.argv[2] == "gpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402

import fx  # noqa: E402

R1, MT = fx.review1, fx.methods
t00 = time.time()
L = fx.logs.open(NAME)
out = {"log": NAME, "fx": fx.code_fingerprint(), "calls": {}}
SCR = os.path.join(C.OUT, "scratch")
os.makedirs(SCR, exist_ok=True)


def call(name, fn):
    t0 = time.time()
    try:
        v = fn()
        out["calls"][name] = {"ok": True, "info": v, "s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        out["calls"][name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1200:]}
        print("FAILED", name, out["calls"][name]["error"], flush=True)


def save():
    C.dump(f"api_{'gpu' if GPU else 'cpu'}_{NAME}.json", out)
    bad = [k for k, v in out["calls"].items() if not v["ok"]]
    print(f"[{time.time() - t00:.0f}s] {len(out['calls'])} calls, failed: {bad}", flush=True)


ALL = MT.E1_LABELS + R1.R1_LABELS
GL = MT.E1_GATE_LEVEL + R1.R1_GATE_LEVEL

if not GPU:
    # ------------------------------------------------------------------ E1: row set, gate level, selection, frontier
    rows = R1.e1_rowset_review1(L, 0.05, 0, H=None, labels=ALL + GL)
    labs = [M.label for M in rows]
    out["e1_rowset_without_model"] = {"n": len(labs), "missing_without_model": [x for x in ALL + GL if x not in labs]}

    def gl_all():
        bad, n = [], 0
        for M in rows:
            if M.kind in ("deployed", "preedit", "dual", "weight", "gate") or not (fx.gate.is_scoped(M.obj) or M.kind == "request"):
                continue
            ev = fx.gate.evaluate(L, M, positions=("tf", "prompt"), top1=True)
            st = fx.metrics.gate_stats(L, ev)
            need = [st["probe"]["heldout_disjoint"]["fired"], st["probe"]["heldout_strict"]["fired"], st["probe"]["heldout_disjoint_image"]["fired"], st["probe"]["heldout_disjoint_text"]["fired"],
                    st["edit"]["tgen"]["own_top"], st["edit"]["tgen"]["own_cand"], st["edit"]["igen"]["own_top"], st["edit"]["tgen"]["other_top"], st["hn"]["HN-I"]["fired"], st["hn"]["HN-I"]["own_top"],
                    st["hn"]["HN-Q"]["fired"], st["hn"]["HN-X"]["fired"], st["probe"]["lineage"]["width_mean"] if st["probe"]["lineage"]["n_firing"] else 0.0, st["top1"]["all"]["n_differs"]]
            st2 = fx.metrics.gate_stats(L, ev, "prompt")
            need.append(st2["probe"]["heldout_disjoint"]["fired"])
            if any(v is None for v in need):
                bad.append(M.label)
            n += 1
        return {"n_gates": n, "missing_fields": bad}
    call("E1 gate level: evaluate + gate_stats for every gate-level row", gl_all)

    def literal():
        r = {}
        for nm in ("rescope", "cosine", "cos_centred", "whitecos_0.1", "retr_conj", "retr_min", "rescope_x"):
            M, Ml = MT.build(nm, L), MT.build(nm, L, allow_silence=True)
            r[nm] = [MT.n_capped(M), MT.n_capped(Ml), "allow_silence" in Ml.variant]
        return r
    call("E1 literal rule without the cap (allow_silence=True), n_capped", literal)
    call("E1 own prompts / calibration record", lambda: {nm: [sorted((MT.build(nm, L).rec.get("own_prompts") or {}).keys())[:6], "n_own_keys_silenced_by_literal_rule" in MT.build(nm, L).rec["cal"]] for nm in ("rescope", "cosine", "whitecos_0.1")})
    call("E1 far columns: pools.firing BEN-T / MME-T", lambda: {nm: [fx.pools.firing(L, MT.build(nm, L), "BEN-T")["n"], fx.pools.firing(L, MT.build(nm, L), "MME-T")["n"]] for nm in ("rescope", "cosine", "cos_null", "white_1_shared", "rescope_nullW", "rescope_pac", "rescope_x")})

    def h3_all():
        r = {}
        for M in rows:
            if M.label in R1.COSINE_FAMILY + R1.RETRIEVAL_FAMILY + ("rescope", "rescope_shared", "rescope_null", "rescope_x", "rescope_nullW", "rescope_pac", "ball"):
                h = fx.sweeps.h3(L, M)
                r[M.label] = [bool(h["reachable"]), h["rephrase"]["mean"] is not None if h["reachable"] else None]
        return {"n": len(r), "unreachable": [k for k, v in r.items() if not v[0]]}
    call("E1 H3 for every calibrated gate (20 names)", h3_all)
    call("E1 H3 over 20 draws for the two families (selection read-out)", lambda: {nm: int(sum(bool(fx.sweeps.h3(L, MT.build(nm, L, draw=d))["reachable"]) for d in range(fx.draws.N_DRAWS))) for nm in ("cosine", "whitecos_0.1_shared", "retr_sum", "retr_min", "retr_conj")})
    call("E1 _x read-outs: h3_both, near_scope", lambda: [R1.h3_both(L, MT.build("rescope_x", L))["subset"]["n"], R1.near_scope(L, MT.build("cosine_x", L))["subset"]["n_kept"]])

    def selection():
        p = os.path.join(SCR, f"sel_{NAME}.json")
        tab = {"note": "scratch file of the review; not the real selection"}
        R1.write_selection({bb: {"log": "x", "cosine_family": {"H3": {"variant": "whitecos", "calibration": "shared", "kappa": 0.1}, "H2": "cos_centred"},
                                 "retrieval": {"H3": "retr_sum", "H2": "retr_min"}, "tables": tab} for bb in ("qwen25vl", "llava")}, p)
        s = R1.selected(L, path=p)
        s2 = R1.selected(L, "H2", path=p)
        rd = R1.read_selection(p)
        real = None
        try:
            R1.read_selection()
        except FileNotFoundError:
            real = "FileNotFoundError (not written yet)"
        Mx = MT.build(s["cosine_x"], L)
        return {"H3": s, "H2": s2, "x_form_changes_calibration": rd[L.backbone]["cosine"]["x_form_changes_calibration"], "real_file": real, "built_x": Mx.label}
    call("E1 selection: write_selection / read_selection / selected / <best>_x", selection)

    def frontier():
        r = {}
        for feat in ("ncoord", "cos"):
            fr = fx.gate.frontier(L, feat)
            mc = fx.gate.matched_coverage(L, feat, (25, 50, 75))
            r[feat] = [len(fr["thresholds"]), sorted(k for k in fr["curves"] if k.startswith("hn."))[:3], "probe.heldout_disjoint.fired" in fr["curves"], "edit.tgen.own_top" in fr["curves"], len(mc)]
        sw = fx.gate.sweep(L, "whitecos_0.1", gate=R1.unthresholded(L, "whitecos_0.1", draw=0))
        fr = fx.gate.frontier(L, sw=sw)
        sp = fx.sweeps.h3_split(L)
        pos = {j: i for i, j in enumerate(sw["parts"]["hn"]["rows"])}
        r["variant"] = [len(fr["thresholds"]), all(j in pos for k in sp["test"] for j in sp["test"][k])]
        cv = fx.sweeps.shift_curves(L, MT.build("retr_conj", L), [0.0, 0.05], pools=("BEN-T",))
        r["retr_conj_curves"] = [k in cv["curves"] for k in ("probe.heldout_disjoint.fired", "edit.rephrase.own_top", "hn.test.mean.fired", "hn.HN-I.fired", "pool.BEN-T.fired")]
        return r
    call("E1b / E1c: frontier, matched_coverage, variant sweep, retrieval shift curves", frontier)
    # ------------------------------------------------------------------ E6: ablation switches that build without the model
    def e6():
        r = {}
        sw = {"response=gain": dict(response="gain"), "response=lin": dict(response="lin"), "rebase=False": dict(rebase=False), "fire=all": dict(fire="all"), "feature=coord": dict(feature="coord"),
              "feature=cos": dict(feature="cos"), "calibration=shared": dict(calibration="shared"), "calibration=log_only": dict(calibration="log_only"),
              "calibration=running_max": dict(calibration="running_max"), "split=1:3": dict(split=(1, 3)), "split=3:1": dict(split=(3, 1)), "rank=r": dict(rank="r"),
              "allow_silence": dict(allow_silence=True), "source=BEN-S": dict(source="BEN-S"), "contamination=0.01": dict(contamination=0.01, contam_half="even"),
              "contamination=0.05": dict(contamination=0.05, contam_half="even"), "contamination=0.2": dict(contamination=0.2, contam_half="even"),
              "n=40": dict(n=40), "n=100": dict(n=100), "n=400": dict(n=400), "baseline=0.98 (cut by R19; the switch exists)": dict(baseline=0.98)}
        base = MT.rescope(L)
        for k, kw in sw.items():
            M = MT.rescope(L, **kw)
            r[k] = [M.variant, M.digest != base.digest]
        for na in (0, 50, 100, 200):
            M = MT.rescope_null(L, n_A=na)
            r[f"rescope_null n_A={na}"] = [M.variant, (M.rec.get("null_repair") or {}).get("n_a")]
        r["rescope_pac"] = [MT.rescope_pac(L).rec["cal"]["pac"]["rank"]]
        r["even/odd edits"] = [len(fx.draws.even_edits(L)), len(fx.draws.odd_edits(L))]
        c = fx.draws.calibration(L, 200, 0, contamination=0.2, contam_half="even")
        r["contaminating edits all even"] = all(e % 2 == 0 for e in c.rec["contaminating_edits"]) and c.rec["n_contaminating"] == 40
        return r
    call("E6 ablation switches (CPU builds)", e6)
    call("E9 mask_edits on a cosine-family gate and on a raised method", lambda: [MT.mask_edits(L, MT.build("whitecos_0.1", L), [1, 2]).variant, MT.mask_edits(L, fx.sweeps.raised_method(L, MT.rescope(L), 0.05), [3]).variant])
    # ------------------------------------------------------------------ E2 a-c, prefix logs
    call("E2 complaints / verify_p12", lambda: [len(fx.draws.complaints(L, 10, 0)), len(fx.draws.complaints(L, 100, 1)), set(fx.draws.complaints(L, 10, 0)) <= set(fx.draws.complaints(L, 20, 0)), str(fx.draws.verify_p12(L))[:200]])
    call("E2 c evidence curve: n = 0 / 5 / 40 / 200, rescope_shared(n=)", lambda: [MT.evidence(L, n, 0, 0.05).variant for n in (0, 5, 40, 200)] + [MT.rescope_shared(L, n=40).n, MT.rescope_logonly(L).variant])

    def prefix():
        r = {}
        for nm in fx.logs.names("prefix"):
            if not nm.startswith(NAME):
                continue
            Lp = fx.logs.open(nm)
            c = fx.draws.complaints(Lp, 10, 0, n_base=20)
            M = MT.rescope(Lp, draw=0)
            Mx = MT.build("rescope_x", Lp, draw=0)
            r[nm] = [Lp.T, len(c), M.obj.T, Mx.obj.T, fx.prefix.describe(nm).get("complete"), len(Lp.rows("hn", valid=True)), MT.rescope_logonly(Lp).obj.T]
        return r
    call("E2 prefix logs: open, complaints(n_base=20), rescope, rescope_x, describe", prefix)
    # ------------------------------------------------------------------ E3 row set on the development log + mechanics on one unseen log
    def e3(Lx, mech=False):
        p = os.path.join(SCR, f"sel_{NAME}.json")
        sel = R1.selected(Lx, path=p)
        labels = tuple(dict.fromkeys(MT.E3_LABELS + (sel["cosine"], sel["cosine_x"], sel["retr"], "lookup", "rescope_x") + ("rescope_nullW", "rescope_pac")))
        rs = R1.e1_rowset_review1(Lx, 0.05, 0, H=None, labels=labels, skip_unavailable=False)
        rq = R1.e1_rowset_review1(Lx, 0.05, 0, H=None, labels=("rescope", sel["cosine"], sel["retr"], "rescope_x"), mask="qdisjoint")
        n = 0
        for M in rs + rq:
            if M.kind == "unavailable" or not (fx.gate.is_scoped(M.obj) or M.kind == "request"):
                continue
            ev = fx.gate.evaluate(Lx, M, positions=("tf",))
            fx.metrics.gate_stats(Lx, ev)
            if M.kind != "request":
                fx.pools.firing(Lx, M, "MME-T")
            fx.pools.firing(Lx, M, "BEN-T", mask=fx.masks.question_disjoint(Lx))
            n += 1
        hq = Lx.rows("probe", pool="heldout_disjoint", valid=True, where=fx.masks.question_disjoint(Lx))
        le = fx.masks.later_edits(Lx)
        h = [bool(fx.sweeps.h3(Lx, M)["reachable"]) is not None for M in rs if M.label in ("rescope", sel["cosine"], sel["retr"])]
        hb = [R1.h3_both(Lx, M)["subset"]["n_all"] > 0 for M in rs if M.label in ("rescope_x", sel["cosine_x"])]
        return {"labels": [M.label for M in rs], "kinds": sorted({M.kind for M in rs}), "n_gate_level_passes": n, "n_later_edits": len(le), "n_question_disjoint_probes": len(hq) if not mech else "not printed",
                "h3": len(h), "h3_both": len(hb), "stored": sorted(fx.logs.stored(Lx.name))[:4] if not mech else "ok", "retr_on_MME-T": R1.why_not(Lx, sel["retr"]),
                "near_edit_flag": any(Lx.near_edit(j) is not None for j in Lx.pool("heldout_disjoint")), "lineage": int(fx.gate.decide(Lx, rs[1], Lx.pool("heldout_disjoint"))["n_cand"].max()) if not mech else "ok"}
    call("E3 row set on the development log (frozen labels, qdisjoint read-out, H3, pools)", lambda: e3(L))
    other = "TB-Q200-ae-l2_3" if L.family == "Q" else "TB-L200-ae-l2_3"
    call(f"E3 mechanics on {other} (open, build, one gate-level pass; no metric)", lambda: {k: v for k, v in e3(fx.logs.open(other), True).items() if k in ("labels", "kinds", "n_gate_level_passes", "n_later_edits", "h3", "h3_both")})
    call("E3 programme()", lambda: [len(fx.logs.programme()), sorted(set(fx.logs.programme().values()))])
    # ------------------------------------------------------------------ E5 / E8
    def e5():
        S = MT.base_gate(L)[0]
        Rm = fx.gate.features(L, S, L.pool("heldout_disjoint")[:20], "tf")
        P = fx.pools.get(L, "MME-T")
        Rk = fx.gate.features(L, S, P.rows[:20], P.which, kc=P.kc)
        Pb = fx.pools.get(L, "BEN-T")
        Rb = fx.gate.features(L, S, Pb.rows[:20], Pb.which)
        from rescope import shared_rule, two_block_rule
        r1 = two_block_rule(fx.gate.features(L, S, fx.draws.calibration(L, 200, 0).rows, "tf").numpy(), 0.05, seed=0)
        r2 = shared_rule(Rm.numpy(), 0.05)
        return [tuple(Rm.shape), tuple(Rk.shape), tuple(Rb.shape), MT.pac_rank(100, 0.05, 0.1), abs(float(r1["eta"]) - float(MT.rescope(L).rec["cal"]["eta"])) < 1e-12, "eta" in r2,
                [len(fx.pools.get(L, p).rows) for p in ("BEN-T", "BEN-S", "MME-C", "MME-T")]]
    call("E5 features on probes / MME / BEN pools, library rules on matrices, pac_rank", e5)
    save()
    # ------------------------------------------------------------------ item 8: the sanity numbers of e0_sanity/out/table.txt (gate level, 20 draws)
    def sanity():
        txt = open(C.FIN + "/e0_sanity/out/table.txt").read()
        sec = txt[txt.index("\n" + NAME + "  ("):]
        nxt = sec.find("\n####", 200)
        nxt2 = sec.find("\nTB-", 200)
        sec = sec[:nxt2] if nxt2 > 0 else sec
        num = lambda s: [float(x) for x in re.findall(r"(-?\d+\.\d+|\d+) \[", s)]           # noqa: E731
        tab = {}
        for tag, key in (("M1 ncoord two-block (default)", "rescope"), ("M2 ncoord shared scale", "rescope_shared"), ("M3 cosine two-block", "cosine"), ("M4 cosine shared scale", "cosine_shared")):
            lines = [ln for ln in sec.splitlines() if ln.strip().startswith(tag)]
            g, hn_, fur = lines[0], lines[1], lines[2]
            first = lambda ln: [float(x) for x in re.findall(r"(-?\d+\.\d+|-?\d+) \[", ln[len(tag) + 4:])]      # noqa: E731
            a, b, c = first(g), first(hn_), first(fur)
            tab[key] = {"fire": a[0], "fire_uncapped": a[1], "tgen_own": a[2], "tgen_cand": a[3], "igen_own": a[4], "igen_cand": a[5], "capped": a[-1],
                        "hn_i": b[0], "hn_i_own": b[1], "hn_q": b[2], "hn_x": b[4], "strict": c[0], "image": c[2], "text": c[3]}
        mine = {}
        for key in tab:
            acc = {k: [] for k in tab[key]}
            for d in range(20):
                M = MT.build(key, L, alpha=0.05, draw=d)
                st = fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn")))
                Ml = MT.build(key, L, alpha=0.05, draw=d, allow_silence=True)
                stl = fx.metrics.gate_stats(L, fx.gate.evaluate(L, Ml, positions=("tf",), parts=("probe",)))
                acc["fire"].append(st["probe"]["heldout_disjoint"]["fired"])
                acc["fire_uncapped"].append(stl["probe"]["heldout_disjoint"]["fired"])
                acc["tgen_own"].append(st["edit"]["tgen"]["own_top"])
                acc["tgen_cand"].append(st["edit"]["tgen"]["own_cand"])
                acc["igen_own"].append(st["edit"]["igen"]["own_top"])
                acc["igen_cand"].append(st["edit"]["igen"]["own_cand"])
                acc["capped"].append(MT.n_capped(M))
                acc["hn_i"].append(st["hn"]["HN-I"]["fired"])
                acc["hn_i_own"].append(st["hn"]["HN-I"]["own_top"])
                acc["hn_q"].append(st["hn"]["HN-Q"]["fired"])
                acc["hn_x"].append(st["hn"]["HN-X"]["fired"])
                acc["strict"].append(st["probe"]["heldout_strict"]["fired"])
                acc["image"].append(st["probe"]["heldout_disjoint_image"]["fired"])
                acc["text"].append(st["probe"]["heldout_disjoint_text"]["fired"])
            mine[key] = {k: float(np.mean(v)) for k, v in acc.items()}
        diff = {key: {k: round(mine[key][k] - tab[key][k], 3) for k in tab[key]} for key in tab}
        tol = lambda k: 0.051 if k in ("tgen_own", "tgen_cand", "igen_own", "igen_cand", "capped", "hn_i", "hn_i_own", "hn_q", "hn_x") else 0.0051       # noqa: E731
        worst = max(abs(v) for d_ in diff.values() for v in d_.values())
        return {"table": tab, "driver_mean_over_20_draws": mine, "driver_minus_table": diff, "largest_abs_difference": worst,
                "all_within_rounding": all(abs(v) <= tol(k) for d_ in diff.values() for k, v in d_.items())}
    call("item 8: sanity numbers (M1 - M4, gate level, 20 draws) against e0_sanity/out/table.txt", sanity)
    save()
else:
    # ================================================================== GPU: behavioural names on tiny inputs
    H = fx.logs.load_model(NAME)
    C.cap_gpu()
    ses = fx.behave.session(L, H)
    M = MT.rescope(L, alpha=0.05, draw=0)
    Mc = MT.cosine(L, alpha=0.05, draw=0)
    dep = MT.deployed(L)
    hd = [int(j) for j in L.pool("heldout_disjoint")]
    hnr = [int(j) for j in L.rows("hn", valid=True)]
    E3_ = [0, 1, 2, 3]
    keep = {}

    def edits():
        e = ses.edits(M, edits=E3_)
        ed_ = ses.edits(dep, edits=E3_)
        keep["e"], keep["ed"] = e, ed_
        kg = fx.metrics.kept_gained(e, ed_, ci=True)
        es = fx.metrics.edit_stats(e, ci=True)
        return {"n_rows": len(e), "fields": sorted(e[0].keys())[:40], "kept_gained": sorted(kg.keys()), "edit_stats": sorted(es.keys())[:8], "token_level": "token_acc" in e[0]}
    call("ses.edits (subset), kept_gained, edit_stats", edits)

    def probes():
        p = ses.probes(M)
        ps = ses.probes(M, pool="heldout_strict")
        keep["p"] = p
        st = fx.metrics.probe_stats(p, ci=True)
        return {"n": len(p), "n_strict": len(ps), "fields": sorted(p[0].keys())[:45], "stats": sorted(st.keys())[:12], "has_near_edit": "near_edit" in p[0], "has_tok_loc": "tok_loc" in p[0], "has_query_id": "query_id" in p[0]}
    call("ses.probes heldout_disjoint / heldout_strict, probe_stats", probes)

    def hns():
        sub = hnr[:6] + hnr[600:604] + hnr[-4:]
        t, g = ses.hard_negatives(M, rows=sub, generate=True)
        hs = fx.metrics.hn_stats(t, g, ci=False)
        t2 = ses.hard_negatives(Mc, rows=sub)
        flags = [k for k in t[0] if any(x in k for x in ("categ", "related", "new", "deg", "wrong", "hn_type"))]
        return {"n": len(t), "n_gen": len(g), "subset_flags": flags, "stats": sorted(hs.keys())[:10], "wrong_target_in_gen": any("wrong_target" in r for r in g) or any("wrong_target" in r for r in t), "n2": len(t2)}
    call("ses.hard_negatives(generate=True) on 14 rows, hn_stats, subset flags", hns)

    def gens():
        rows = [L.edit_rows["rel"][0], L.edit_rows["tgen"][1], hd[0], hd[1], hnr[0]]
        g = ses.generate(M, rows)
        gs = fx.metrics.gen_stats(g)
        d2 = ses.d2(M, rows=hd[:6])
        ds = fx.metrics.d2_stats(d2, ci=True)
        gd = ses.generate(dep, rows[:3])
        return {"n": len(g), "gen_fields": sorted(g[0].keys())[:30], "gen_stats": sorted(gs.keys())[:8], "d2_n": len(d2), "d2_stats": ds if not isinstance(ds, dict) else sorted(ds.keys()), "deployed_gen": len(gd)}
    call("ses.generate, gen_stats, ses.d2(rows=), d2_stats(ci=True)", gens)

    def dual():
        Md = MT.dual(L)
        rows = [L.edit_rows["rel"][0], hd[0], hnr[0]]
        g = ses.generate(Md, rows)
        t = ses.dual_tf(Md, rows)
        return {"n": len(g), "fields": sorted(g[0].keys())[:25], "dual_tf": len(t)}
    call("dual: ses.generate, ses.dual_tf", dual)

    def stats_():
        e2 = ses.edits(Mc, edits=E3_)
        ga = fx.metrics.grid({0: keep["e"], 1: keep["e"]}, lambda r: float(r["exact"]), unit="edit", where=lambda r: r["metric"] in ("tgen", "igen"))
        gb = fx.metrics.grid({0: e2, 1: e2}, lambda r: float(r["exact"]), unit="edit", where=lambda r: r["metric"] in ("tgen", "igen"))
        d = fx.stats.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100)
        gp = fx.metrics.grid({0: keep["p"]}, "D1", unit="query_id")
        b = fx.stats.crossed_bootstrap(gp["num"], gp["den"], scale=100)
        ep = fx.metrics.endpoint(keep["e"], e2)
        return {"diff_keys": sorted(d.keys())[:8], "probe_units": int(gp["num"].shape[1]), "d1_value": b["value"] is not None, "endpoint": sorted(ep.keys())[:8], "holm": fx.stats.holm({"a": d["p_two_sided"], "b": 0.2})["adjusted"]}
    call("fx.metrics.grid + crossed_bootstrap(_diff), endpoint, holm on behavioural rows", stats_)

    def lin():
        rows = [L.edit_rows["rel"][0], L.edit_rows["tgen"][0], hd[0], hd[1]]
        r = {}
        for k, kw in (("response=lin", dict(response="lin")), ("response=gain", dict(response="gain")), ("fire=all", dict(fire="all")), ("rebase=False", dict(rebase=False)),
                      ("role=all", dict(role="all", H=H)), ("role=last+answer", dict(role="last+answer", H=H))):
            Mx = MT.rescope(L, **kw)
            r[k] = [len(ses.tf(Mx, rows)), Mx.kind]
        return r
    call("E6 behavioural ablations on 4 sequences (lin, gain, fire all, stored values, roles)", lin)

    def where():
        Mw = MT.where_logonly(L, H)
        rows = [L.edit_rows["rel"][0], hd[0], hd[1]]
        t = ses.tf(Mw, rows)
        c = fx.draws.complaints(L, 3, 0)
        Ml = MT.where_least(L, c, H, start="logonly")
        Md = MT.where_least(L, c, H, start="deployed")
        t2 = ses.tf(Ml, rows)
        b = ses.blocklist(c, rows + list(c)[:2])
        return {"where_logonly": [len(t), "n_cand_edits" in t[0]], "where_least": [Ml.variant, Md.variant, len(t2)], "blocklist": [len(b), sorted(set(r.get("src") for r in b))]}
    call("where_logonly, where_least (logonly / deployed start), ses.blocklist", where)

    def dense():
        r = {}
        rows = [L.edit_rows["rel"][0], hd[0], hd[1]]
        for nm, fn in (("router", MT.router), ("zscore", MT.zscore)):
            Mr = fn(L, H, alpha=0.05, draw=0)
            Mdf = MT.build(nm, L, H=H, calibration="default")
            r[nm] = [Mr.label, len(ses.tf(Mr, rows)), Mdf.label]
        return r
    call("router / zscore (calibrated and published forms) on 3 sequences", dense)

    def p12():
        r = {}
        for lab in ("p12_comp_b_nall", "p12_reedit_b_nall", "p12_rb_union_nall", "p12_rb_tot_nall"):
            rs = MT.e1_rowset(L, 0.05, 0, H=H, labels=(lab,))
            r[lab] = [rs[0].kind, len(ses.tf(rs[0], [hd[0], L.edit_rows["rel"][0]]))] if rs else "not available"
        return r
    call("linear repairs of P12 at all complaints through the row set (2 sequences)", p12)
    call("full E1 row set with the model: labels returned", lambda: [M_.label for M_ in R1.e1_rowset_review1(L, 0.05, 0, H=H, labels=ALL + GL)])
    if L.family == "L":
        call("LLaVA: lp='prompt' next to lp='tf'", lambda: [len(ses.tf(M, hd[:4] + [L.edit_rows["tgen"][0]], lp="prompt")), len(ses.edits(M, edits=[0], lp="prompt"))])
    call("behavioural H3: raised method on odd edits and test rows", lambda: (lambda h: [len(ses.edits(h["method"], edits=h["odd_edits"][:2])), len(ses.hard_negatives(h["method"], rows=h["test_rows"]["rows"][:6], generate=True)[0])])(fx.sweeps.h3(L, M)))
    call("behavioural H3 of a retrieval rule", lambda: (lambda h: [len(ses.edits(h["method"], edits=h["odd_edits"][:2])), h["method"].kind])(fx.sweeps.h3(L, MT.build("retr_sum", L))))
    call("E8: rescope.Hook(module, gate, record=False)", lambda: (lambda rs: type(rs.Hook(H.mod, M.obj.to(H.dev) if hasattr(M.obj, "to") else M.obj, record=False)).__name__)(__import__("rescope")))
    save()

    # ---- item 10: image and question attributes of 100 random sequences against the rendered prompts
    def attributes():
        att = [json.loads(x) for x in open(f"{C.FIN}/cache2/retr/{L.backbone}/attributes.jsonl") if x.strip()]
        seq = [a for a in att if a["table"] == "sequence"]
        IMG = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"
        rng = np.random.default_rng(77036)
        cand = [j for j, e in enumerate(L.index) if e.get("family") == L.family and e.get("lib_sid") is not None and e.get("kind") != "benign"]
        ev, v = ses.ev_plain, H.vlm
        pick, recs, tried = [], [], 0
        for j in rng.permutation(cand):
            if len(recs) >= 100:
                break
            j = int(j)
            try:
                it = ses.item(j)
            except KeyError:
                continue
            q, ptype, ans = ev._query(it)
            question = q["m_loc_q"] if ptype == "image" else q["loc"]
            b = ev.batch(it, answer=False)
            ids = b["inputs"]["input_ids"][0]
            text = v.tok.decode([int(t) for t in ids if int(t) != getattr(v, "image_token_id", -1)])
            a = seq[j]
            md5 = None
            if ptype == "image":
                p = q["m_loc"] if os.path.isabs(q["m_loc"]) else os.path.join(IMG, q["m_loc"])
                md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
            recs.append({"j": j, "kind": L.index[j]["kind"], "question_equal": a["question"] == question, "question_in_rendered_prompt": question.strip() in text,
                         "key_is_strip_lower": a["question_key"] == question.strip().lower(), "image_md5_equal": a["image_md5"] == md5, "text_row_has_no_image": (ptype == "image") == (a["image"] is not None),
                         "n_tokens_equal_cache": int(ids.shape[0]) == int(L.index[j]["len_prompt"]), "sid_equal": a["sid"] == L.index[j]["sid"]})
        # benign rows are gate level only (no library item): the attribute table against the cache index and the image file
        ben = [j for j, e in enumerate(L.index) if e.get("kind") == "benign"]
        nb = 0
        okb = 0
        for j in rng.permutation(ben)[:30]:
            j = int(j)
            e, a = L.index[j], seq[j]
            nb += 1
            good = a["question"] == e["prompt"] and a["image_md5"] == e.get("image_md5")
            if e.get("image"):
                p = e["image"] if os.path.isabs(e["image"]) else os.path.join(IMG, e["image"])
                good = good and hashlib.md5(open(p, "rb").read()).hexdigest() == a["image_md5"]
            okb += int(good)
        keys_ = ("question_equal", "question_in_rendered_prompt", "key_is_strip_lower", "image_md5_equal", "text_row_has_no_image", "n_tokens_equal_cache", "sid_equal")
        return {"n": len(recs), "by_kind": {k: sum(r["kind"] == k for r in recs) for k in sorted({r["kind"] for r in recs})}, **{k: sum(r[k] for r in recs) for k in keys_},
                "benign_rows_checked": nb, "benign_rows_ok": okb, "all_ok": all(all(r[k] for k in keys_) for r in recs) and okb == nb and len(recs) == 100}
    call("item 10: attributes of 100 sequences against the rendered prompts", attributes)
    try:
        out["assert_clean"] = ses.assert_clean()
    except Exception as e:  # noqa: BLE001
        out["assert_clean"] = f"FAILED {type(e).__name__}: {e}"
    save()
    print("assert_clean", out["assert_clean"])
