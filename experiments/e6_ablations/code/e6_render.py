"""E6: results.json (fx.io schema), table.txt, paper_numbers.json (+ doc) from the job outputs.  No number is typed by hand.

    python e6_render.py

Reads out/gate_<log>.json (role-'last' rows, 20 draws), out/multi_<log>.json (roles last + answer / all, 20 draws), out/cells/ and
out/rows/ (behaviour, draws 0 - 2) and the sanity run e0 (comparison of the default and the shared-scale rows).  Whatever is not
there yet is rendered as 'pending'.
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, needs=("experiments/e6_ablations/out/rows",))
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import glob
import os
import time

import numpy as np

import e6_common as C
from e6_common import fx

E0 = f"{_RP.ROOT}/02_research/experiments/10_final/e0_sanity/out/results.json"
M, ST = fx.metrics, fx.stats
HN = ("HN-I", "HN-Q", "HN-X")
CAMEL = {"default": "Default", "role_lastans": "RoleLastAnswer", "role_all": "RoleAll", "resp_gain": "RespGain", "resp_lin": "RespLin",
         "values_logged": "ValuesLogged", "fire_all": "FireAll", "feat_coord": "FeatCoord", "feat_cos": "FeatCos", "cal_shared": "CalShared",
         "cal_logonly": "CalLogOnly", "cal_runmax": "CalRunMax", "split_1_3": "SplitOneThree", "split_3_1": "SplitThreeOne", "rank_r": "RankRaw",
         "guards_off": "GuardsOff", "src_bens": "SrcBenS", "src_bens_dec": "SrcBenSPrompt", "contam_01": "ContamOne", "contam_05": "ContamFive",
         "contam_20": "ContamTwenty", "n_40": "NForty", "n_100": "NHundred", "n_400": "NFourHundred", "null_50": "NullFifty", "null_100": "NullHundred",
         "null_200": "NullTwoHundred", "pac": "Pac"}


def g(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def gate_metrics(key, cell):
    """Flat gate-level numbers of one (row, draw) cell: {metric: (pool, value)}."""
    st, cov = cell["stats"], ("own_cand" if key == "fire_all" else "own_top")
    out = {"gate.fire": ("heldout_disjoint", g(st, "probe", "heldout_disjoint", "fired")),
           "gate.fire.strict": ("heldout_strict", g(st, "probe", "heldout_strict", "fired")),
           "gate.fire.near_probe": ("heldout_disjoint_near", g(st, "probe", "heldout_disjoint_near", "fired")),
           "gate.fire.far_probe": ("heldout_disjoint_far", g(st, "probe", "heldout_disjoint_far", "fired")),
           "gate.fire.literal": ("heldout_disjoint", g(cell, "literal", "heldout_disjoint")),
           "gate.fire.BEN-T": ("BEN-T", g(cell, "BEN-T", "fired")),
           "gate.fire.literal.BEN-T": ("BEN-T", g(cell, "literal", "BEN-T")),
           "gate.n_capped": ("edits", g(cell, "cal", "n_capped")), "gate.own_not_released": ("edits", g(cell, "cal", "own_not_released")),
           "gate.eta": ("", g(cell, "cal", "eta"))}
    for m in ("rel", "tgen", "igen"):
        out[f"gate.cov.{m}"] = ("edits", g(st, "edit", m, cov))
        out[f"gate.other_edit.{m}"] = ("edits", g(st, "edit", m, "other_top"))
    for par in ("even", "odd"):
        for m in ("tgen", "igen"):
            out[f"gate.cov.{par}.{m}"] = (f"edits_{par}", g(cell, "parity", par, m, cov))
    hv = []
    for t in HN:
        out[f"gate.hn.{t}.fire"] = (t, g(st, "hn", t, "fired"))
        out[f"gate.hn.{t}.own"] = (t, g(st, "hn", t, "own_top"))
        hv.append(g(st, "hn", t, "fired"))
    out["gate.hn.mean.fire"] = ("HN", float(np.mean(hv)) if all(v is not None for v in hv) else None)
    wd = g(st, "probe", "lineage", "multi")
    out["gate.lineage.multi"] = ("heldout_disjoint", wd)
    return out


def load_gate(name):
    out = {}
    for f in (f"gate_{name}.json", f"multi_{name}.json"):
        p = os.path.join(C.OUT, f)
        if os.path.exists(p):
            out.update(fx.io.jload(p))
    return out


def rows_of(tag):
    p = os.path.join(C.OUT, "rows", tag + ".jsonl.gz")
    return fx.io.load_rows(p) if os.path.exists(p) else None


def cb(rows_by_draw, value, unit, where):
    gr = M.grid(rows_by_draw, value, unit=unit, where=where)
    return ST.crossed_bootstrap(gr["num"], gr["den"], scale=100.0)


def behaviour(name, key, dep_rows, even, G):
    """Behavioural numbers of one row: per draw and crossed over the draws.  None if no cell is finished."""
    sh = C.SHORT[name]
    ds = [d for d in (C.BEHAVE_DRAWS if C.BY[key]["per_draw"] else (0,)) if os.path.exists(os.path.join(C.OUT, "cells", f"{sh}__{key}__d{d}.json"))]
    if not ds:
        return None
    rows = {d: rows_of(f"{sh}__{key}__d{d}") for d in ds}
    cells = {d: fx.io.jload(os.path.join(C.OUT, "cells", f"{sh}__{key}__d{d}.json")) for d in ds}
    ed = {d: [r for r in rows[d] if r["kind"] == "edit"] for d in ds}
    pr = {d: [r for r in rows[d] if r["kind"] == "probe"] for d in ds}
    hn = {d: [r for r in rows[d] if r["kind"] == "hn"] for d in ds}
    per, cr = {}, {}
    for d in ds:
        es, ps, hs = M.edit_stats(ed[d]), M.probe_stats(pr[d]), M.hn_stats(hn[d])
        pss = M.probe_stats([r for r in pr[d] if r["strict"]])
        o = {"beh.fire": ps["fired"], "beh.D1": ps["D1"], "beh.D1.strict": pss["D1"], "beh.fire.strict": pss["fired"], "beh.D1_given_fired": ps["D1_given_fired"],
             "beh.fire.near_probe": g(ps, "near", "fired"), "beh.D1.near_probe": g(ps, "near", "D1"), "beh.n_near_probe": g(ps, "near", "n"),
             "beh.T-Loc*": ps["T-Loc*"], "beh.I-Loc*": ps["I-Loc*"], "beh.rephrase.exact": g(es, "rephrase", "exact")}
        for m in ("rel", "tgen", "igen"):
            o[f"beh.{m}.exact"], o[f"beh.{m}.cov"] = es[m]["exact"], es[m]["own_top"]
        for par, pe in (("even", True), ("odd", False)):
            for m in ("tgen", "igen"):
                sub = [r for r in ed[d] if r["metric"] == m and (r["edit"] in even) == pe]
                o[f"beh.{par}.{m}.exact"], o[f"beh.{par}.{m}.cov"] = M.pct(r["exact"] for r in sub), M.pct(r.get("own_released_last") for r in sub)
        for t in HN:
            o[f"beh.hn.{t}.fire"], o[f"beh.hn.{t}.D1"] = hs[t]["fired"], hs[t]["D1"]
            o[f"beh.hn.{t}.n_y0_degenerate"], o[f"beh.hn.{t}.D1_y0_not_degenerate"] = hs[t].get("n_y0_degenerate"), hs[t].get("D1_y0_not_degenerate")
        o["beh.hn.mean.fire"] = float(np.mean([hs[t]["fired"] for t in HN]))
        o["beh.hn.mean.D1"] = float(np.mean([hs[t]["D1"] for t in HN]))
        kg = M.kept_gained(ed[d], dep_rows)
        for m in ("rel", "tgen", "igen"):
            for q in ("kept", "gained", "lost"):
                o[f"beh.{m}.{q}"] = kg[m][q]
        o["beh.n_capped"] = g(cells[d], "cal", "n_capped") if g(cells[d], "cal", "n_capped") is not None else g(G, key, str(d), "cal", "n_capped")
        onr = g(cells[d], "cal", "own_not_released")                    # roles beyond 'last': from the hook's flags of the Rel rows
        o["beh.own_not_released"] = onr if onr is not None else sum(not r.get("own_released_last") for r in ed[d] if r["metric"] == "rel")
        per[d] = o
    isk = lambda k: (lambda r: r["kind"] == k)                           # noqa: E731
    cr["beh.fire"] = cb(pr, lambda r: float(bool(r["fired"])), "query_id", isk("probe"))
    cr["beh.D1"] = cb(pr, lambda r: float(bool(r["D1"])), "query_id", isk("probe"))
    cr["beh.D1.strict"] = cb(pr, lambda r: float(bool(r["D1"])), "query_id", lambda r: r["strict"])
    for m in ("rel", "tgen", "igen"):
        w = (lambda mm: (lambda r: r["metric"] == mm))(m)
        cr[f"beh.{m}.exact"] = cb(ed, lambda r: float(bool(r["exact"])), "edit", w)
        cr[f"beh.{m}.cov"] = cb(ed, lambda r: float(bool(r.get("own_released_last"))), "edit", w)
        kgr = {d: M.kept_gained_rows(ed[d], dep_rows) for d in ds}
        for q in ("kept", "gained", "lost"):
            cr[f"beh.{m}.{q}"] = cb(kgr, lambda r, q=q: float(bool(r[q])), "edit", w)
    for t in HN:
        w = (lambda tt: (lambda r: r["hn_type"] == tt))(t)
        cr[f"beh.hn.{t}.fire"] = cb(hn, lambda r: float(bool(r["fired"])), "edit", w)
        cr[f"beh.hn.{t}.D1"] = cb(hn, lambda r: float(bool(r["D1"])), "edit", w)
    d2 = rows_of(f"{sh}__{key}__d0__d2")
    d2s = None
    if d2 is not None:
        s = M.d2_stats(d2, ci=True)
        d2s = {"n_pre_correct": s["n_pre_correct"], "n_D2": s["n_D2"], "D2": s["D2"], "fired": s["fired"]}
    return {"draws": ds, "per": per, "crossed": cr, "d2": d2s, "cells": cells, "variant": cells[ds[0]].get("variant", ""),
            "rows": {d: rows[d] for d in ds}}


def fm(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def ms(s, nd=1):
    """mean [p05, p95] over draws."""
    if s is None or s.get("mean") is None:
        return "-"
    if s["n_draws"] == 1:
        return f"{s['mean']:.{nd}f}"
    return f"{s['mean']:.{nd}f} [{s['p05']:.{nd}f}, {s['p95']:.{nd}f}]"


def ci(s, nd=1):
    if s is None or s.get("value") is None:
        return "-"
    return f"{s['value']:.{nd}f} [{s['ci_lo']:.{nd}f}, {s['ci_hi']:.{nd}f}]"


def e0_compare(name, G, B):
    """Default (M1) and shared-scale (M2) rows against the sanity run, cell by cell."""
    if not os.path.exists(E0):
        return ["  sanity results not found"], {}
    e0 = fx.io.jload(E0)["logs"][name]
    lines, worst = [], {}
    for key, m in (("default", "M1"), ("cal_shared", "M2")):
        if key not in G:
            continue
        for lab, path, mine in (("gate fire % heldout_disjoint", ("probe", "heldout_disjoint", "fired"), "gate.fire"),
                                ("gate fire % heldout_strict", ("probe", "heldout_strict", "fired"), "gate.fire.strict"),
                                ("gate coverage T-Gen", ("edit", "tgen", "own_top"), "gate.cov.tgen"), ("gate coverage I-Gen", ("edit", "igen", "own_top"), "gate.cov.igen"),
                                ("gate HN-I firing", ("hn", "HN-I", "fired"), "gate.hn.HN-I.fire"), ("gate HN-Q firing", ("hn", "HN-Q", "fired"), "gate.hn.HN-Q.fire"),
                                ("gate HN-X firing", ("hn", "HN-X", "fired"), "gate.hn.HN-X.fire")):
            ref = g(e0, "gate", "methods", m, "tf", *path)
            if ref is None or "values" not in ref:
                continue
            a = [gate_metrics(key, G[key][str(d)])[mine][1] for d in range(len(ref["values"])) if str(d) in G[key]]
            dv = max(abs(x - y) for x, y in zip(a, ref["values"])) if a else None
            worst[f"{key}:{lab}"] = dv
            lines.append(f"  {key:11s} {lab:30s} E6 mean {np.mean(a):6.2f}   E0 mean {ref['mean']:6.2f}   largest per-draw difference {dv:.2e} ({len(a)} draws)")
        if B.get(key):
            for lab, path, mine in (("behaviour Rel exact", ("edit", "rel", "exact"), "beh.rel.exact"), ("behaviour T-Gen exact", ("edit", "tgen", "exact"), "beh.tgen.exact"),
                                    ("behaviour I-Gen exact", ("edit", "igen", "exact"), "beh.igen.exact"),
                                    ("behaviour D1 heldout_disjoint", ("probe", "heldout_disjoint", "D1"), "beh.D1"),
                                    ("behaviour D1 heldout_strict", ("probe", "heldout_strict", "D1"), "beh.D1.strict"),
                                    ("behaviour HN-I D1", ("hn", "HN-I", "D1"), "beh.hn.HN-I.D1"), ("behaviour HN-Q D1", ("hn", "HN-Q", "D1"), "beh.hn.HN-Q.D1"),
                                    ("behaviour HN-X D1", ("hn", "HN-X", "D1"), "beh.hn.HN-X.D1")):
                ref = g(e0, "behaviour", "methods", m, "summary", *path)
                if ref is None or "values" not in ref:
                    continue
                a = [B[key]["per"][d][mine] for d in B[key]["draws"]]
                dv = max(abs(x - y) for x, y in zip(a, ref["values"]))
                worst[f"{key}:{lab}"] = dv
                lines.append(f"  {key:11s} {lab:30s} E6 {'/'.join(fm(x, 2) for x in a)}   E0 {'/'.join(fm(x, 2) for x in ref['values'])}   largest difference {dv:.2e}")
    return lines, worst


def main():
    R = fx.io.Results("E6", C.OUT)
    paper, doc, text = {}, {}, []
    text.append(f"E6 ABLATIONS OF THE METHOD      (rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/e6_render.py from out/*.json and out/rows/; no number is typed by hand)")
    text.append("alpha = 0.05, n = 200 unless the row varies it; ONE switch is changed from the default method per row (fx.methods.rescope(L, <switch>)).")
    text.append("GATE LEVEL: 20 calibration draws, cells = mean [5th, 95th percentile over the draws]; teacher-forced keys of the key cache (fx.gate); rows 'role = ...' need")
    text.append("  every position: one truncated forward per sequence (code/e6_multi.py), calibrated gates bit-identical to the driver's for draws 0 - 2 (digest check below).")
    text.append("BEHAVIOUR: calibration draws 0 - 2 (R19: only for role = all, response = LIN, logged values, shared scale, log-only, running maxima, guards off, contamination 20 %),")
    text.append("  teacher forcing at B = 1 through fx.behave; cells = mean over the draws [95 % crossed bootstrap over draws x units, 2000 resamples; units: query clusters (probes),")
    text.append("  edits (edit metrics, hard negatives)]; n_d = number of draws behind the interval (3 draws: the interval is too narrow where draws disagree; they are listed separately).")
    text.append("FF = false firing = % of valid heldout_disjoint rows on which an edit is released; 'lit' = the literal conformal rule without the guard cap (the event the bound covers);")
    text.append("BEN-T = test part of the benign pool, PROMPT-ONLY keys (fx.pools.firing; gates are calibrated on teacher-forced keys); near probes = heldout_disjoint rows whose question")
    text.append("is the question of a logged edit; cov = % of T-Gen / I-Gen sequences on which the OWN edit is released at the last prompt position (selection = all: is a candidate);")
    text.append("HN-I / HN-Q / HN-X = near-scope negatives (valid rows): % of rows on which AN edit is released, and D1; capped = edits whose threshold was lowered by the guard (bound void for them);")
    text.append("ownNR = own Rel prompts on which the own edit is not released.  Rows marked (=) change only what is released: at gate level they are the default row by construction.")
    text.append("FF strict = the same on heldout_strict.  Key-level pools hold the last-prompt key only: BEN-T is not defined for the roles that look at further positions.  MME pools: not part of E6.")
    text.append("Dropped by R19: the block-1 baseline as 0.98-quantile.  Gate level only by R19: every row without behaviour columns ('.').  rescope_null with n_A = 0 is the default row (checked).")
    allB = {}
    for name in C.LOGS:
        sh = C.SHORT[name]
        G = load_gate(name)
        dep = rows_of(f"{sh}__deployed")
        dep_e = [r for r in dep if r["kind"] == "edit"] if dep else None
        even = set(fx.io.jload(os.path.join(C.OUT, f"gate_{name}_meta.json"))["even_edits"])
        B = {k: (behaviour(name, k, dep_e, even, G) if dep_e else None) for k in C.BEHAVE}
        allB[name] = B
        text += ["", "#" * 170, f"{name}", "#" * 170]
        # ---------------------------------------------------------------- records
        S = {}
        for key in C.KEYS:
            if key not in G:
                continue
            a = C.BY[key]
            cells = G[key]
            ds = sorted(int(d) for d in cells)
            per = {d: gate_metrics(key, cells[str(d)]) for d in ds}
            S[key] = {}
            kw = dict(variant=cells[str(ds[0])]["cal"].get("variant") or "", alpha=C.ALPHA if key != "cal_logonly" and key != "cal_runmax" else None,
                      n=cells[str(ds[0])]["cal"].get("n"))
            for met in per[ds[0]]:
                pool = per[ds[0]][met][0]
                vals = {d: per[d][met][1] for d in ds}
                if all(v is None for v in vals.values()):
                    S[key][met] = None
                    continue
                S[key][met] = R.add_draws(name, key, met, vals, pool=pool, n_draws=len(ds), **kw)
        if dep:
            dp = M.probe_stats([r for r in dep if r["kind"] == "probe"], ci=True)
            de = M.edit_stats(dep_e, ci=True)
            dh = M.hn_stats([r for r in dep if r["kind"] == "hn"], ci=True)
            R.add_stat(name, "deployed", "beh.D1", dp["D1"], pool="heldout_disjoint")
            for m in ("rel", "tgen", "igen"):
                R.add_stat(name, "deployed", f"beh.{m}.exact", de[m]["exact"], pool="edits")
            for t in HN:
                R.add_stat(name, "deployed", f"beh.hn.{t}.D1", dh[t]["D1"], pool=t)
        for key, b in B.items():
            if not b:
                continue
            c0 = b["cells"][b["draws"][0]]
            kw = dict(variant=b["variant"], alpha=C.ALPHA if key not in ("cal_logonly", "cal_runmax") else None, n=g(c0, "cal", "n"))
            pool_of = lambda met: ("heldout_strict" if "strict" in met else "edits" if any(x in met for x in ("rel", "tgen", "igen", "rephrase", "capped", "own_not")) else  # noqa: E731
                                   met.split(".")[2] if ".hn." in met else "heldout_disjoint")
            for d in b["draws"]:
                for met, v in b["per"][d].items():
                    R.add(name, key, met, v, draw=d, pool=pool_of(met), **kw)
            for met, s in b["crossed"].items():
                R.add(name, key, met + "@crossed", s["value"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n_units=s["n_units"], pool=pool_of(met), n_draws=s["n_draws"], **kw)
            if b["d2"]:
                R.add_stat(name, key, "beh.D2", b["d2"]["D2"], draw=0, pool="heldout_disjoint_pre_correct", n_pre_correct=b["d2"]["n_pre_correct"], n_D2=b["d2"]["n_D2"], **kw) \
                    if isinstance(b["d2"]["D2"], dict) else R.add(name, key, "beh.D2", None, draw=0, pool="heldout_disjoint_pre_correct", **kw)

        # ---------------------------------------------------------------- compact table: one row per ablation
        text += ["", f"(1) COMPACT ABLATION TABLE, {name}: gate level over 20 draws (all rows) | behaviour over draws 0 - 2 (rows registered for behaviour; '.' = gate level only by R19)"]
        hdr = ["row", "FF", "FF strict", "FF lit", "FF BEN-T", "FF near probes", "cov T", "cov I", "HN-I rel", "HN-Q rel", "HN-X rel", "capped", "ownNR", "|", "n_d", "FF", "D1", "Rel", "T-Gen", "I-Gen",
               "HN-I D1", "HN-Q D1", "HN-X D1"]
        body = []
        for key in C.KEYS:
            a, s, b = C.BY[key], S.get(key), B.get(key)
            lab = f"{a['group'].split(' ', 1)[0]:>2} {a['text']}" + (" (=)" if key in C.GATE_EQUALS_DEFAULT else "")
            if s is None:
                gl = ["pending"] + [""] * 11
            else:
                cap, onr = s.get("gate.n_capped"), s.get("gate.own_not_released")
                gl = [ms(s["gate.fire"], 2), ms(s["gate.fire.strict"], 2), ms(s["gate.fire.literal"], 2), "not defined" if s.get("gate.fire.BEN-T") is None else ms(s["gate.fire.BEN-T"], 2),
                      ms(s["gate.fire.near_probe"], 1), ms(s["gate.cov.tgen"]), ms(s["gate.cov.igen"]), ms(s["gate.hn.HN-I.fire"]), ms(s["gate.hn.HN-Q.fire"]),
                      ms(s["gate.hn.HN-X.fire"]), "-" if cap is None else f"{cap['mean']:.1f} (max {cap['max']:.0f})", "-" if onr is None else f"{onr['mean']:.1f} (max {onr['max']:.0f})"]
            if b:
                c = b["crossed"]
                bl = [str(len(b["draws"])), ci(c["beh.fire"], 2), ci(c["beh.D1"], 2), ci(c["beh.rel.exact"]), ci(c["beh.tgen.exact"]), ci(c["beh.igen.exact"]),
                      ci(c["beh.hn.HN-I.D1"]), ci(c["beh.hn.HN-Q.D1"]), ci(c["beh.hn.HN-X.D1"])]
            else:
                bl = ["pending" if a["behave"] else "."] + [""] * 8
            body.append([lab] + gl + ["|"] + bl)
        text.append(fx.io.text_table(hdr, body))
        if dep:
            text.append(f"   deployed log W_E (reference, one run): D1 {ci(dp['D1'], 2)}; Rel / T-Gen / I-Gen exact {fm(de['rel']['exact']['value'])} / {fm(de['tgen']['exact']['value'])} / "
                        f"{fm(de['igen']['exact']['value'])}; D1 on HN-I / HN-Q / HN-X {fm(dh['HN-I']['D1']['value'])} / {fm(dh['HN-Q']['D1']['value'])} / {fm(dh['HN-X']['D1']['value'])}")

        # ---------------------------------------------------------------- behaviour detail
        text += ["", f"(2) BEHAVIOUR, {name}: coverage, heldout_strict, near-scope release next to the far rate, R7 read-outs (kept / gained against the deployed log, % of the 100 edits;",
                 "    D2 = % of the pre-correct heldout_disjoint rows whose 48-token answer loses the reference, draw 0 only, bootstrap over query clusters)"]
        hdr = ["row", "n_d", "FF far (hd)", "HN-I rel", "HN-Q rel", "HN-X rel", "FF near probes", "D1 strict", "cov T", "cov I", "Rel kept/gained", "T-Gen kept/gained", "I-Gen kept/gained",
               "D2 % (n_pre)", "capped", "ownNR"]
        body = []
        for key in C.BEHAVE:
            b = B.get(key)
            if not b:
                body.append([key, "pending"] + [""] * 14)
                continue
            c, pm = b["crossed"], (lambda met: float(np.mean([b["per"][d][met] for d in b["draws"] if b["per"][d][met] is not None] or [np.nan])))
            d2 = b["d2"]
            body.append([key, str(len(b["draws"])), ci(c["beh.fire"], 2), ci(c["beh.hn.HN-I.fire"]), ci(c["beh.hn.HN-Q.fire"]), ci(c["beh.hn.HN-X.fire"]),
                         fm(pm("beh.fire.near_probe")), ci(c["beh.D1.strict"], 2), ci(c["beh.tgen.cov"]), ci(c["beh.igen.cov"])]
                        + [f"{fm(c[f'beh.{m}.kept']['value'])} / {fm(c[f'beh.{m}.gained']['value'])}" for m in ("rel", "tgen", "igen")]
                        + ["-" if not d2 or not isinstance(d2["D2"], dict) else f"{ci(d2['D2'], 2)} ({d2['n_pre_correct']})",
                           "/".join(str(b["per"][d]["beh.n_capped"]) for d in b["draws"]), "/".join(str(b["per"][d]["beh.own_not_released"]) for d in b["draws"])])
        text.append(fx.io.text_table(hdr, body))
        text += ["", f"(3) BEHAVIOUR PER CALIBRATION DRAW, {name} (draws 0 / 1 / 2 listed separately)"]
        hdr = ["row", "FF", "D1", "Rel", "T-Gen", "I-Gen", "HN-I rel", "HN-Q rel", "HN-X rel", "HN-I D1", "HN-Q D1", "HN-X D1"]
        body = []
        for key in C.BEHAVE:
            b = B.get(key)
            if not b:
                continue
            pd = lambda met, nd=1: " / ".join(fm(b["per"][d][met], nd) for d in b["draws"])     # noqa: E731
            body.append([key, pd("beh.fire", 2), pd("beh.D1", 2), pd("beh.rel.exact", 0), pd("beh.tgen.exact", 0), pd("beh.igen.exact", 0), pd("beh.hn.HN-I.fire"), pd("beh.hn.HN-Q.fire"),
                         pd("beh.hn.HN-X.fire"), pd("beh.hn.HN-I.D1"), pd("beh.hn.HN-Q.D1"), pd("beh.hn.HN-X.D1")])
        text.append(fx.io.text_table(hdr, body))
        deg = []
        for key in C.BEHAVE:
            b = B.get(key)
            if b:
                p0 = b["per"][b["draws"][0]]
                deg.append(f"{key}: " + ", ".join(f"{t} {p0[f'beh.hn.{t}.n_y0_degenerate']} rows, D1 without them {fm(p0[f'beh.hn.{t}.D1_y0_not_degenerate'])}" for t in HN))
        if deg:
            text += ["   hard negatives with a degenerate pre-edit answer (empty, one character, '?', bare None / No), draw 0: " + deg[0]] + ["      " + x for x in deg[1:]]

        # ---------------------------------------------------------------- item 10: contamination
        text += ["", f"(4) CONTAMINATION OF THE CALIBRATION SAMPLE, {name}: share of the n = 200 calibration queries replaced by rephrase prompts (text / image half each) of the edits with EVEN stream index;",
                 "    gate level, 20 draws; exact: behaviour, draws 0 - 2 (0 % and 20 % only, R19)"]
        hdr = ["share", "FF", "FF near probes", "HN mean rel", "cov T even", "cov T odd", "cov I even", "cov I odd", "capped", "|", "exact T even", "exact T odd", "exact I even", "exact I odd"]
        body = []
        for key, lab in (("default", "0 %"), ("contam_01", "1 %"), ("contam_05", "5 %"), ("contam_20", "20 %")):
            s, b = S.get(key), B.get(key)
            if s is None:
                continue
            bl = ["." if key in ("contam_01", "contam_05") else "pending"] * 4
            if b:
                bl = [" / ".join(fm(b["per"][d][f"beh.{par}.{m}.exact"], 0) for d in b["draws"]) for m in ("tgen", "igen") for par in ("even", "odd")]
            body.append([lab, ms(s["gate.fire"], 2), ms(s["gate.fire.near_probe"]), ms(s["gate.hn.mean.fire"]), ms(s["gate.cov.even.tgen"]), ms(s["gate.cov.odd.tgen"]), ms(s["gate.cov.even.igen"]),
                         ms(s["gate.cov.odd.igen"]), f"{s['gate.n_capped']['mean']:.1f} (max {s['gate.n_capped']['max']:.0f})", "|"] + bl)
        text.append(fx.io.text_table(hdr, body))

        # ---------------------------------------------------------------- item 11: sample size
        text += ["", f"(5) SAMPLE SIZE n, {name} (gate level only): mean, [5th, 95th percentile], sd over the 20 draws; bound = floor(alpha (n2 + 1)) / (n2 + 1) of the two-block rule"]
        hdr = ["n", "bound %", "FF mean [p05, p95]", "FF sd", "FF lit", "FF BEN-T", "cov T mean [p05, p95]", "cov T sd", "cov I mean [p05, p95]", "cov I sd", "HN mean rel", "capped"]
        body = []
        for key, lab in (("n_40", "40"), ("n_100", "100"), ("default", "200"), ("n_400", "400")):
            s = S.get(key)
            if s is None:
                continue
            bd = G[key]["0"]["cal"].get("bound")
            if isinstance(bd, str) and "/" in bd:
                a_, b_ = bd.split("/")
                bd = float(a_) / float(b_)
            body.append([lab, "-" if bd is None else f"{100 * float(bd):.2f}", ms(s["gate.fire"], 2), fm(s["gate.fire"]["sd"], 2), ms(s["gate.fire.literal"], 2), ms(s["gate.fire.BEN-T"], 2),
                         ms(s["gate.cov.tgen"]), fm(s["gate.cov.tgen"]["sd"]), ms(s["gate.cov.igen"]), fm(s["gate.cov.igen"]["sd"]), ms(s["gate.hn.mean.fire"]),
                         f"{s['gate.n_capped']['mean']:.1f} (max {s['gate.n_capped']['max']:.0f})"])
        text.append(fx.io.text_table(hdr, body))

        # ---------------------------------------------------------------- per-draw listing at gate level where draws matter
        text += ["", f"(6) GATE LEVEL PER DRAW, {name}: for each of the 20 calibration draws, FF on heldout_disjoint (first line) and the release rate on near-scope negatives, mean of",
                 "    HN-I / HN-Q / HN-X (second line); all draws and all metrics are in results.json"]
        for key in C.KEYS:
            if key in G and C.BY[key]["per_draw"]:
                gm = [gate_metrics(key, G[key][str(d)]) for d in sorted(int(x) for x in G[key])]
                text.append(f"   {key:14s} FF   " + " ".join(f"{x['gate.fire'][1]:5.2f}" for x in gm))
                text.append(f"   {'':14s} near " + " ".join(f"{x['gate.hn.mean.fire'][1]:5.1f}" for x in gm))

        # ---------------------------------------------------------------- checks
        text += ["", f"(7) CHECKS, {name}"]
        mp = os.path.join(C.OUT, f"gate_{name}_meta.json")
        if os.path.exists(mp):
            ck = fx.io.jload(mp)["checks"]
            text.append(f"   identities at draws 0 - 2 (content digests / thresholds): rescope_null(n_A = 0) == default: {all(v for k, v in ck.items() if k.startswith('null0'))}; "
                        f"feature = cos == the registered cosine gate: {all(v for k, v in ck.items() if k.startswith('featcos'))}; "
                        f"GAIN / LIN / logged values have the default's detectors and thresholds: {all(v for k, v in ck.items() if 'same_thresholds' in k)}")
        mm = os.path.join(C.OUT, f"multi_{name}_meta.json")
        if os.path.exists(mm):
            ck = fx.io.jload(mm)
            text.append(f"   multi-position roles: gates calibrated from the cached per-query maxima have the driver's content digest for draws 0 - 2: "
                        f"{all(v['equal'] for v in ck['checks'].values())} ({len(ck['checks'])} gates); assert_clean {ck['assert_clean']}")
            b = B.get("role_all")
            if b and "role_all" in G:
                nmis, ntot = 0, 0
                for d in b["draws"]:
                    mine = {x["csid"]: x for x in G["role_all"][str(d)].get("per_seq", [])}
                    for r in b["rows"][d]:
                        x = mine.get(r["csid"])
                        if x is None:
                            continue
                        ntot += 1
                        nmis += (bool(r["fired"]) != x["fired"]) or (r.get("own_released_last") is not None and bool(r["own_released_last"]) != bool(x["own_last"]))
                text.append(f"   role = all: firing flags of the gate-level pass against the hook's flags of the behavioural rows (draws {b['draws']}): {nmis} of {ntot} sequences differ")
                R.add(name, "role_all", "check.gate_vs_hook.n_differ", nmis, pool="all", n_units=ntot)
        jobs = sorted(glob.glob(os.path.join(C.OUT, f"behave_{name}_job_*.json")))
        for jp in jobs:
            j = fx.io.jload(jp)
            text.append(f"   behaviour job {os.path.basename(jp)}: assert_clean {j['assert_clean']} (hook_mismatch and memo conflicts must be 0)")
        hm = [(k, d, b["cells"][d]["assert_clean"]) for k, b in B.items() if b for d in b["draws"]]
        text.append(f"   behaviour cells finished: {len(hm)}; cells with hook_mismatch or memo conflict: {sum(1 for _, _, a in hm if a['hook_mismatch'] or a['memo_conflicts'])}")
        lines, worst = e0_compare(name, G, B)
        text += [f"   comparison with the sanity run e0 (default = M1, shared scale = M2), same draws:"] + ["   " + x for x in lines]
        for k, v in worst.items():
            R.add(name, k.split(":")[0], "check.e0." + k.split(":")[1].replace(" ", "_"), v, pool="")

        # ---------------------------------------------------------------- paper numbers
        for key in C.KEYS:
            s, b = S.get(key), B.get(key)
            cm = "Esix" + sh + CAMEL[key]
            if s:
                for met, nm, what in (("gate.fire", "Fire", "false firing on heldout_disjoint, % (gate level, mean over 20 calibration draws)"),
                                      ("gate.fire.BEN-T", "FireBenT", "false firing on BEN-T, prompt-only keys, % (gate level, mean over 20 draws)"),
                                      ("gate.cov.tgen", "CovT", "coverage of text rephrases, % (gate level, mean over 20 draws)"),
                                      ("gate.cov.igen", "CovI", "coverage of image rephrases, % (gate level, mean over 20 draws)"),
                                      ("gate.hn.mean.fire", "NearFire", "release rate on near-scope negatives, mean of HN-I / HN-Q / HN-X, % (gate level, mean over 20 draws)"),
                                      ("gate.n_capped", "Capped", "number of capped edits (mean over 20 draws)")):
                    if s.get(met):
                        paper[cm + nm] = round(s[met]["mean"], 2)
                        doc[cm + nm] = f"{name}, ablation row '{C.BY[key]['text']}': {what}"
                if s.get("gate.fire"):
                    paper[cm + "FireLo"], paper[cm + "FireHi"] = round(s["gate.fire"]["p05"], 2), round(s["gate.fire"]["p95"], 2)
                    doc[cm + "FireLo"] = f"{name}, '{C.BY[key]['text']}': 5th percentile over 20 draws of the false firing on heldout_disjoint, %"
                    doc[cm + "FireHi"] = f"{name}, '{C.BY[key]['text']}': 95th percentile over 20 draws of the false firing on heldout_disjoint, %"
            if b:
                c = b["crossed"]
                for met, nm, what in (("beh.D1", "DOne", "D1 on heldout_disjoint, %"), ("beh.rel.exact", "Rel", "Rel exact, %"), ("beh.tgen.exact", "TGen", "T-Gen exact, %"),
                                      ("beh.igen.exact", "IGen", "I-Gen exact, %"), ("beh.hn.HN-I.D1", "HnIDOne", "D1 on HN-I, %"), ("beh.hn.HN-Q.D1", "HnQDOne", "D1 on HN-Q, %"),
                                      ("beh.hn.HN-X.D1", "HnXDOne", "D1 on HN-X, %")):
                    paper[cm + nm] = round(c[met]["value"], 2)
                    doc[cm + nm] = f"{name}, ablation row '{C.BY[key]['text']}': {what} (behaviour, mean over calibration draws {b['draws']})"
    R.save(alpha=C.ALPHA, n=C.N, n_draws_gate=C.N_DRAWS, draws_behaviour=list(C.BEHAVE_DRAWS), rows=[{k: v for k, v in C.BY[key].items() if k != "sw"} | {"switches": str(C.BY[key]["sw"])} for key in C.KEYS])
    if os.path.exists(R.path("paper_numbers.json")):
        os.remove(R.path("paper_numbers.json"))
    R.paper(paper)
    fx.io.jdump(dict(sorted(doc.items())), R.path("paper_numbers_doc.json"), indent=1)
    R.table("\n".join(text))
    print(f"[e6_render] {len(R.records)} records, {len(paper)} paper numbers -> {R.path('table.txt')}")


if __name__ == "__main__":
    main()
