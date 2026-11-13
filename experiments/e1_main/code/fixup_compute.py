"""E1 fix-up (amendment of 2026-11-12 00:48 UTC, decisions 1 and 4): ADDITIONS only, computed from stored files; no model run, no
existing file is changed.  Writes into out/fixup/:

  mcnemar.json                 item 1: exact McNemar tests (PROTOCOL section 4) of ReScope against every other row of the main table
  reporting.json               item 2: near-scope draws, generation / D2 / wrong-target read-outs, Holm bounds, the H2 family
  rows_from_e2.json            item 3: the rows the main table takes from E2, read again from the final E2 records, with the checks
  paper_numbers_fixup.json     flat {name: value}; every name starts with 'EoneFx' (no name of out/paper_numbers.json is reused)
  paper_numbers_fixup_doc.json one line per name
  checks.json                  every check of this script with its outcome

    PYTHONDONTWRITEBYTECODE=1 python fixup_compute.py        (CPU; then fixup_render.py)
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os
import time

from e1c import *  # noqa: F401,F403
import e1c
import e1_render

ST = fx.stats
FIX = os.path.join(OUT, "fixup")
E2 = FINAL + "/e2_curves/out"
BDRAWS = (0, 1, 2)
DW = {0: "Zero", 1: "One", 2: "Two"}
KW = {"HN-I": "HNI", "HN-Q": "HNQ", "HN-X": "HNX", "all": "HNAll"}
N_BOOT = ST.N_BOOT
PN, DOC, CHECKS = {}, {}, []
ITEM_OF, CUR = {}, [0]                   # item of the task a paper number belongs to


def put(name, value, text):
    if value is None:
        return
    if name in PN:
        raise RuntimeError("paper number defined twice: " + name)
    PN[name], DOC[name] = value, text
    ITEM_OF[name] = CUR[0]


def check(item, what, ok, detail=None):
    CHECKS.append({"item": item, "check": what, "ok": bool(ok), "detail": detail})
    print(f"  [{'ok' if ok else 'FAILED'}] item {item}: {what}" + ("" if detail is None else f"  ({str(detail)[:200]})"), flush=True)


def selections():
    """{log: selected labels} from out/best_cosine_gate.json (the file the experiment wrote before the behavioural runs)."""
    s = jload(SEL)["selected"]
    out = {}
    for bb, e in s.items():
        out[e["log"]] = {"cos_H3": e["cosine"]["H3"], "cos_H3_x": e["cosine"]["H3_x"], "cos_H2": e["cosine"]["H2"], "cos_H2_x": e["cosine"]["H2_x"],
                         "retr_H3": e["retr"]["H3"], "retr_H2": e["retr"]["H2"]}
    return out


def names(sel):
    """Short macro names of the rows of the main table (the names of e1_analyze.paper; cos_H2_x, which has none there, is added)."""
    nm = {"deployed": "Deployed", "preedit": "Preedit", "rescope": "Rescope", "rescope_shared": "RescopeShared", "rescope_null": "RescopeNull", "rescope_x": "RescopeX",
          "rescope_lin": "RescopeLin", "where_logonly": "RescopeW", "cosine": "Cosine", "cosine_shared": "CosineShared", "lookup": "Lookup", "ball": "Ball", "router": "Router",
          "cosine_default": "CosDefault", "cosine_published": "CosPublished", "zscore_default": "Zscore", "exact": "Exact", "dual": "Dual"}
    for k, v in ((sel["cos_H3"], "BestCos"), (sel["cos_H2"], "BestCosHtwo"), (sel["cos_H3_x"], "BestCosX"), (sel["cos_H2_x"], "BestCosHtwoX"), (sel["retr_H3"], "Retr"),
                 (sel["retr_H2"], "RetrHtwo")):
        nm.setdefault(k, v)
    return nm


# ----------------------------------------------------------------------------- item 1: exact McNemar
def flags_of(rows, outcome, e2=False):
    """{csid: flag} of one binary outcome: 'D1' on heldout_disjoint (probe rows of the pool) | 'Rel' exact (the 100 Rel checks)."""
    if outcome == "D1":
        sub = [r for r in rows if r["kind"] == "probe" and r.get("disjoint") and "line" not in r and not r.get("d2_row")]
        return {r["csid"]: bool(r["D1"]) for r in sub}
    sub = [r for r in rows if r["kind"] == "edit" and r["metric"] == "rel" and "line" not in r]
    return {r["csid"]: bool(r["exact"]) for r in sub}


def e1_rows(log, lab, d):
    p = os.path.join(OUT, "rows", f"{log}__{lab}__d{'x' if d is None else d}__tf.jsonl.gz")
    return fx.io.load_rows(p) if os.path.exists(p) else None


def mcnemar():
    sels = selections()
    tests, rows_meta, excluded = [], [], []
    for log in DEV:
        sel = sels[log]
        order, name, cal = e1_render.rows_for(sel)
        nm = names(sel)
        A = {d: e1_rows(log, "rescope", d) for d in BDRAWS}
        if any(v is None for v in A.values()):
            raise RuntimeError("ReScope row files missing: " + log)
        fa = {(o, d): flags_of(A[d], o) for o in ("D1", "Rel") for d in BDRAWS}
        others = [(lab, name[lab], nm[lab], "E1", lab in cal) for lab in order if lab != "rescope"]
        s = SHORT[log]
        e2src = [("COMP_B ALL", "COMP-B, all reported complaints (E2 d: rebuilt, seed 0)", "RowCompBAll", f"d__{s}__COMP_B__nALL__s0", False),
                 ("REEDIT_B ALL", "REEDIT-B, all reported complaints (E2 d: saved weight of P12)", "RowReeditBAll", f"d__{s}__REEDIT_B__nALL__s0", False),
                 ("where_cal", "ReScope-W, two-block thresholds (E2 c'; draws 0-2)", "WCal", f"cw__{s}__where_cal", True)]
        for lab, title, mn, src, has_draws in others + [(a, b, c, "E2:" + t, h) for a, b, c, t, h in e2src]:
            rows_meta.append({"log": log, "row": lab, "title": title, "macro": mn, "source": src, "has_calibration_draws": bool(has_draws)})
            for d in BDRAWS:
                if src == "E1":
                    B = e1_rows(log, lab, d if has_draws else None)
                    file = f"e1_main/out/rows/{log}__{lab}__d{d if has_draws else 'x'}__tf.jsonl.gz"
                else:
                    tag = src[3:] + (f"__d{d}" if has_draws else "")
                    p = os.path.join(E2, "rows", tag + ".jsonl.gz")
                    B = fx.io.load_rows(p) if os.path.exists(p) else None
                    file = f"e2_curves/out/rows/{tag}.jsonl.gz"
                if B is None:
                    excluded.append({"log": log, "row": lab, "draw": d, "reason": "per-row file missing: " + file})
                    continue
                for o in ("D1", "Rel"):
                    a, b = fa[(o, d)], flags_of(B, o)
                    if set(a) != set(b):
                        excluded.append({"log": log, "row": lab, "draw": d, "outcome": o, "reason": f"the two files do not hold the same rows ({len(a)} against {len(b)})"})
                        continue
                    ids = sorted(a)
                    xa, xb = [a[i] for i in ids], [b[i] for i in ids]
                    m = ST.mcnemar_exact(xa, xb)
                    n11 = sum(x and y for x, y in zip(xa, xb))
                    tests.append({"log": log, "row": lab, "macro": mn, "outcome": o, "draw": d, "draw_of_the_other_row": d if has_draws else None,
                                  "b": m["n10"], "c": m["n01"], "n11": int(n11), "n00": int(m["n"] - n11 - m["n10"] - m["n01"]), "n": m["n"], "p": m["p"],
                                  "rate_rescope": 100.0 * sum(xa) / len(xa), "rate_other": 100.0 * sum(xb) / len(xb), "file_other": file})
    # Holm: (1) every test of an outcome (backbones x rows x draws); (2) within one calibration draw (backbones x rows)
    fam = {}
    for o in ("D1", "Rel"):
        T = [t for t in tests if t["outcome"] == o]
        h = ST.holm({i: t["p"] for i, t in enumerate(T)})
        for i, t in enumerate(T):
            t["p_holm_all_tests"] = h["adjusted"][i]
        fam[o] = {"all_tests": {"m": h["m"]}, "within_draw": {}}
        for d in BDRAWS:
            Td = [t for t in T if t["draw"] == d]
            hd = ST.holm({i: t["p"] for i, t in enumerate(Td)})
            for i, t in enumerate(Td):
                t["p_holm_within_draw"] = hd["adjusted"][i]
            fam[o]["within_draw"][str(d)] = {"m": hd["m"]}
    # per comparison: the draws next to each other and the pooled table (description only)
    comp = []
    for rm in rows_meta:
        for o in ("D1", "Rel"):
            T = sorted((t for t in tests if t["log"] == rm["log"] and t["row"] == rm["row"] and t["outcome"] == o), key=lambda t: t["draw"])
            if not T:
                continue
            pb, pc = sum(t["b"] for t in T), sum(t["c"] for t in T)
            n, k = pb + pc, min(pb, pc)
            comp.append({"log": rm["log"], "row": rm["row"], "title": rm["title"], "macro": rm["macro"], "source": rm["source"], "outcome": o,
                         "has_calibration_draws": rm["has_calibration_draws"], "n_draws": len(T),
                         "per_draw": [{k2: t[k2] for k2 in ("draw", "b", "c", "n", "p", "p_holm_all_tests", "p_holm_within_draw", "rate_rescope", "rate_other")} for t in T],
                         "pooled_description_only": {"b": pb, "c": pc, "n11": sum(t["n11"] for t in T), "n00": sum(t["n00"] for t in T), "n_rows_times_draws": sum(t["n"] for t in T),
                                                     "note": "sum of the per-draw tables; the rows repeat across the draws, so this is a description, not a test, and enters no Holm family"},
                         "p_max_over_draws": max(t["p"] for t in T), "p_holm_all_tests_max_over_draws": max(t["p_holm_all_tests"] for t in T),
                         "p_holm_within_draw_max_over_draws": max(t["p_holm_within_draw"] for t in T),
                         "all_draws_reject_0.05_after_holm_all_tests": bool(all(t["p_holm_all_tests"] <= 0.05 for t in T)),
                         "direction": ("ReScope lower" if pb < pc else ("ReScope higher" if pb > pc else "tie")) + (" D1" if o == "D1" else " Rel exact")})
    out = {"what": "exact McNemar test (fx.stats.mcnemar_exact: two-sided binomial test on the discordant pairs) of ReScope (default; calibration draw d) against every other row "
                   "of the main table, for the two registered binary paired outcomes, per calibration draw d = 0, 1, 2",
           "outcomes": {"D1": "D1 flag of the valid heldout_disjoint rows (teacher forcing, B = 1); pairs = rows (csid)", "Rel": "Rel exact of the 100 edits; pairs = edits"},
           "b": "pairs with the outcome under ReScope and not under the other row", "c": "pairs with the outcome under the other row and not under ReScope",
           "draws": "a row with calibration draws is paired with ReScope at the same draw index; a row without calibration (one run) is compared with ReScope at each of its draws",
           "holm": {"all_tests": "family = every test of the outcome: backbones x rows x calibration draws (the conservative reading; field p_holm_all_tests)",
                    "within_draw": "family = the tests of the outcome at one calibration draw: backbones x rows (field p_holm_within_draw)",
                    "note": "R6 asks for Holm 'over backbones and baselines'; the tests are per draw, so both ways of counting the draws are given", "families": fam},
           "status": "secondary / descriptive comparisons (R6: H1 to H3 are the only confirmatory ones); the tests add the statistic that PROTOCOL section 4 names",
           "rows_from_e2": "COMP-B, REEDIT-B (all reported complaints) and ReScope-W two-block are rows of table (2b); their flags are read from e2_curves/out/rows (same rows, same driver)",
           "n_tests": len(tests), "excluded": excluded, "comparisons": comp}
    jdump(out, os.path.join(FIX, "mcnemar.json"))
    check(1, "no comparison excluded (every per-row file exists and holds the same rows as ReScope's)", not excluded, excluded[:3] or None)
    ev = jload(ROOT + "/verify/verdicts.json")["protocol_problems"][0]["evidence"]
    bad = []
    for log in DEV:
        for d in BDRAWS:
            t = next(t for t in tests if t["log"] == log and t["row"] == "deployed" and t["outcome"] == "D1" and t["draw"] == d)
            e = ev[log][str(d)]
            if not (t["c"] == e["only_deployed"] and t["b"] == e["only_rescope"] and abs(t["p"] - e["p_exact"]) <= 1e-9 * max(e["p_exact"], 1e-300)):
                bad.append((log, d, t["b"], t["c"], t["p"], e))
    check(1, "D1 against the deployed log reproduces the discordant counts and exact p of the verification (verify/verdicts.json, protocol problem 1; 6 tests)", not bad, bad or None)
    for c in comp:
        base = f"EoneFx{SHORT[c['log']]}{c['macro']}Mc{'DOne' if c['outcome'] == 'D1' else 'Rel'}"
        what = f"{c['log']}: exact McNemar, ReScope against {c['row']} ({c['title']}), outcome {'D1 on heldout_disjoint' if c['outcome'] == 'D1' else 'Rel exact'}"
        for t in c["per_draw"]:
            w = DW[t["draw"]]
            put(base + f"D{w}B", t["b"], f"{what}, calibration draw {t['draw']}: pairs with the outcome under ReScope only (out/fixup/mcnemar.json; flags of out/rows)")
            put(base + f"D{w}C", t["c"], f"{what}, calibration draw {t['draw']}: pairs with the outcome under the other row only")
            put(base + f"D{w}P", t["p"], f"{what}, calibration draw {t['draw']}: exact two-sided p")
            put(base + f"D{w}PHolm", t["p_holm_all_tests"], f"{what}, calibration draw {t['draw']}: Holm-adjusted p, family = all {fam[c['outcome']]['all_tests']['m']} tests of the outcome (backbones x rows x draws)")
            put(base + f"D{w}PHolmDraw", t["p_holm_within_draw"], f"{what}, calibration draw {t['draw']}: Holm-adjusted p, family = the {fam[c['outcome']]['within_draw'][str(t['draw'])]['m']} tests of the outcome at this draw (backbones x rows)")
        put(base + "PoolB", c["pooled_description_only"]["b"], f"{what}: b summed over the draws (description only, not a test)")
        put(base + "PoolC", c["pooled_description_only"]["c"], f"{what}: c summed over the draws (description only, not a test)")
        put(base + "PMax", c["p_max_over_draws"], f"{what}: largest exact p over the draws")
        put(base + "PHolmMax", c["p_holm_all_tests_max_over_draws"], f"{what}: largest Holm-adjusted p over the draws (family: all tests of the outcome)")
    for o, w in (("D1", "DOne"), ("Rel", "Rel")):
        put(f"EoneFxMc{w}NTests", fam[o]["all_tests"]["m"], f"number of exact McNemar tests of the outcome {o} (2 development logs x rows of the main table x 3 calibration draws): the Holm family 'all tests'")
        put(f"EoneFxMc{w}NTestsDraw", fam[o]["within_draw"]["0"]["m"], f"number of exact McNemar tests of the outcome {o} at one calibration draw (2 development logs x rows): the Holm family 'within draw'")
    return out


# ----------------------------------------------------------------------------- item 2: reporting
def holm_bounds(ps, floor):
    """Holm step-down with the information which adjusted values are upper bounds: a bootstrap p at the resolution 1 / n_boot stands
    for 'no resample on the other side of 0' (p <= 1 / n_boot); an adjusted value whose defining term comes from such a p is an
    upper bound.  -> {key: {p, p_holm, at_resolution, p_holm_is_upper_bound}}."""
    keys = sorted(ps, key=lambda k: ps[k])
    m, run, src, out = len(keys), 0.0, False, {}
    for rank, k in enumerate(keys):
        term = min(1.0, (m - rank) * ps[k])
        at = ps[k] <= floor * (1 + 1e-12)
        if term > run:
            run, src = term, at
        elif term == run and at:
            src = True
        out[k] = {"p": ps[k], "p_holm": run, "at_resolution": bool(at), "p_holm_is_upper_bound": bool(src)}
    return out


def reporting(recs):
    I = {}
    for r in recs:
        I.setdefault((r["log"], r["method"], r["metric"], r["pool"], r["draw"]), r)
    g = lambda *k: I.get(k)                                                        # noqa: E731
    v = lambda r: None if r is None else r["value"]                                # noqa: E731
    sels = selections()
    out = {"near_scope_draws": [], "generation": [], "holm": {}, "H2_family": {}}
    ev = jload(ROOT + "/verify/verdicts.json")["protocol_problems"][3]["evidence"]
    bad, n_cmp, bad_mean = [], 0, []
    for log in DEV:
        sel, s = sels[log], SHORT[log]
        order, name, cal = e1_render.rows_for(sel)
        nm = names(sel)
        # (a) the three behavioural draws next to every near-scope cell
        for lab in order:
            c = lab in cal
            row = {"log": log, "row": lab, "title": name[lab], "macro": nm[lab], "calibrated": bool(c), "n_draws": 3 if c else 1, "kinds": {}}
            for k in HN:
                e = {}
                for q, qn in (("fired", "Fire"), ("D1", "DOne")):
                    if c:
                        m = g(log, lab, f"hn.{k}.{q}@x", "hard_negatives", None)
                        dr = [v(g(log, lab, f"hn.{k}.{q}", "hard_negatives", d)) for d in BDRAWS]
                        if m is None or any(x is None for x in dr):
                            e[q] = None
                            continue
                        e[q] = {"mean": m["value"], "ci_lo": m["ci_lo"], "ci_hi": m["ci_hi"], "draws": dr, "spread": max(dr) - min(dr)}
                        if abs(sum(dr) / 3.0 - m["value"]) > 1e-9:
                            bad_mean.append((log, lab, k, q))
                        for d in BDRAWS:
                            put(f"EoneFx{s}{nm[lab]}{KW[k]}{qn}D{DW[d]}", dr[d], f"{log}, {lab}: {'near-scope negatives ' + k + ' on which an edit is released (%)' if q == 'fired' else 'D1 on ' + k + ' (%)'}, "
                                f"behaviour, calibration draw {d} (out/results.json: metric hn.{k}.{q}, draw {d})")
                        put(f"EoneFx{s}{nm[lab]}{KW[k]}{qn}Spread", e[q]["spread"], f"{log}, {lab}: largest minus smallest of the three behavioural draws of that cell (points)")
                        if q == "fired":
                            put(f"EoneFx{s}{nm[lab]}{KW[k]}FireMean", m["value"], f"{log}, {lab}: near-scope negatives {k} on which an edit is released (%), mean over calibration draws 0-2 (out/results.json: hn.{k}.fired@x)")
                    else:
                        m = g(log, lab, f"hn.{k}.{q}", "hard_negatives", None)
                        e[q] = None if m is None or m["value"] is None else {"mean": m["value"], "ci_lo": m["ci_lo"], "ci_hi": m["ci_hi"], "draws": None, "spread": None}
                        if q == "fired" and e[q] is not None:
                            put(f"EoneFx{s}{nm[lab]}{KW[k]}FireMean", m["value"], f"{log}, {lab}: near-scope negatives {k} on which an edit is released (%), one run without calibration (out/results.json: hn.{k}.fired)")
                row["kinds"][k] = e
                if c and lab in ev.get(log, {}) and e.get("D1"):
                    n_cmp += 3
                    if any(abs(x - y) > 1e-9 for x, y in zip(e["D1"]["draws"], ev[log][lab][k])):
                        bad.append((log, lab, k))
            out["near_scope_draws"].append(row)
        # (b) generation, D2, wrong target: the rows of table (3)
        gen_rows = [("deployed", None), ("preedit", None), ("rescope", 0), ("rescope_x", 0), ("where_logonly", None)] + [(b, 0) for b in dict.fromkeys([sel["cos_H3"], sel["cos_H2"]])] + \
                   [(r, 0) for r in dict.fromkeys([sel["retr_H3"], sel["retr_H2"]])] + [("dual", None)]
        for lab, d in gen_rows:
            if g(log, lab, "gen.rel.eq", "edit_prompts", d) is None:
                out["generation"].append({"log": log, "row": lab, "missing": True})
                continue
            src = f"out/results.json, method {lab}, " + ("one run" if d is None else f"calibration draw {d}")
            row = {"log": log, "row": lab, "title": name[lab], "macro": nm[lab], "draw": d, "edit_prompts": {}, "near_scope": {}}
            for m, mw in (("rel", "Rel"), ("tgen", "TGen"), ("igen", "IGen")):
                row["edit_prompts"][m] = {"n": v(g(log, lab, f"gen.{m}.n", "edit_prompts", d)), "eq": v(g(log, lab, f"gen.{m}.eq", "edit_prompts", d)),
                                          "contains": v(g(log, lab, f"gen.{m}.contains", "edit_prompts", d))}
                put(f"EoneFx{s}{nm[lab]}Gen{mw}Eq", row["edit_prompts"][m]["eq"], f"{log}, {lab}: generated answer of the {m} prompts, first line equals the target (% of {row['edit_prompts'][m]['n']:.0f} edits; decoder v2; {src})")
                put(f"EoneFx{s}{nm[lab]}Gen{mw}Contains", row["edit_prompts"][m]["contains"], f"{log}, {lab}: generated answer of the {m} prompts contains the target (% of {row['edit_prompts'][m]['n']:.0f} edits; {src})")
            r = g(log, lab, "gen.probe.eq_pre", "heldout_disjoint", d)
            row["probes_eq_pre"] = None if r is None else {k: r[k] for k in ("value", "ci_lo", "ci_hi", "n_units")}
            if r is not None:
                put(f"EoneFx{s}{nm[lab]}GenProbeEqPre", r["value"], f"{log}, {lab}: valid heldout_disjoint rows whose 10-token generated answer equals the pre-edit answer (%; {src})")
                put(f"EoneFx{s}{nm[lab]}GenProbeEqPreLo", r["ci_lo"], "lower end of its 95 % interval (percentile bootstrap over query clusters)")
                put(f"EoneFx{s}{nm[lab]}GenProbeEqPreHi", r["ci_hi"], "upper end of its 95 % interval")
            r = g(log, lab, "d2.D2", "heldout_disjoint", d)
            n2, npc = v(g(log, lab, "d2.n_D2", "heldout_disjoint", d)), v(g(log, lab, "d2.n_pre_correct", "heldout_disjoint", d))
            row["D2"] = None if r is None or r["value"] is None else {"rate": r["value"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n_D2": n2, "n_pre_correct": npc}
            if row["D2"] is not None:
                if npc and abs(100.0 * n2 / npc - r["value"]) > 1e-9:
                    bad_mean.append((log, lab, "D2 rate != n_D2 / n_pre_correct"))
                put(f"EoneFx{s}{nm[lab]}Dtwo", r["value"], f"{log}, {lab}: D2 rate = % of the pre-correct heldout_disjoint rows whose 48-token answer loses the reference ({n2:.0f} of {npc:.0f} rows; {src})")
                put(f"EoneFx{s}{nm[lab]}DtwoLo", r["ci_lo"], "lower end of its 95 % interval (percentile bootstrap over query clusters)")
                put(f"EoneFx{s}{nm[lab]}DtwoHi", r["ci_hi"], "upper end of its 95 % interval")
                put(f"EoneFx{s}{nm[lab]}DtwoNum", n2, f"{log}, {lab}: number of pre-correct heldout_disjoint rows with D2 (numerator of the D2 rate)")
                put(f"EoneFx{s}{nm[lab]}DtwoDen", npc, f"{log}, {lab}: number of pre-correct heldout_disjoint rows (denominator of the D2 rate)")
            for k in HN + ("all",):
                wt, ng = g(log, lab, f"hngen.{k}.wrong_target", "hard_negatives", d), v(g(log, lab, f"hngen.{k}.n_gen", "hard_negatives", d))
                eqp, eqa = g(log, lab, f"hngen.{k}.eq_pre", "hard_negatives", d), g(log, lab, f"hngen.{k}.eq_any_edit_all_rows", "hard_negatives", d)
                e = {"n_rows": ng, "wrong_target": None if wt is None or wt["value"] is None else {x: wt[x] for x in ("value", "ci_lo", "ci_hi")},
                     "eq_pre": v(eqp), "eq_any_edit_all_rows": v(eqa)}
                row["near_scope"][k] = e
                kind = "all near-scope negatives" if k == "all" else k
                if e["wrong_target"] is not None:
                    put(f"EoneFx{s}{nm[lab]}{KW[k]}WrongTarget", wt["value"], f"{log}, {lab}: wrong-target rate on {kind} = % of the {ng:.0f} valid rows whose generated answer equals the target of the released edit ({src})")
                    put(f"EoneFx{s}{nm[lab]}{KW[k]}WrongTargetLo", wt["ci_lo"], "lower end of its 95 % interval (percentile bootstrap over edits)")
                    put(f"EoneFx{s}{nm[lab]}{KW[k]}WrongTargetHi", wt["ci_hi"], "upper end of its 95 % interval")
                    put(f"EoneFx{s}{nm[lab]}{KW[k]}WrongTargetDen", ng, f"{log}, {lab}: number of valid rows of {kind} (denominator of the wrong-target rate)")
                put(f"EoneFx{s}{nm[lab]}{KW[k]}GenEqPre", e["eq_pre"], f"{log}, {lab}: {kind} whose generated answer equals the pre-edit answer (%; {src})")
                put(f"EoneFx{s}{nm[lab]}{KW[k]}GenEqAny", e["eq_any_edit_all_rows"], f"{log}, {lab}: {kind} whose generated answer equals the target of any edit (% of all rows; {src})")
            out["generation"].append(row)
    check(2, "(a) the per-draw near-scope D1 values equal the values the verification lists (verify/verdicts.json, protocol problem 4)", not bad and n_cmp > 0, f"{n_cmp} values compared; differing: {bad}")
    check(2, "(a, b) the mean of the three draws equals the stored mean; every D2 rate equals n_D2 / n_pre_correct", not bad_mean, bad_mean or None)
    # (c) Holm values at the resolution of the bootstrap, (d) the H2 family
    EP = jload(os.path.join(OUT, "endpoints.json"))
    floor = 1.0 / N_BOOT
    agree = True
    for H, hw in (("H2", "Htwo"), ("H3", "Hthree")):
        ps = {f"{log}/{k}": e["p"] for log in DEV for k, e in EP[log][H].items() if "p" in e}
        hb = holm_bounds(ps, floor)
        fam = EP["holm"][H]
        tests = []
        for key in sorted(ps):
            log, k = key.split("/")
            e = EP[log][H][k]
            agree = agree and abs(hb[key]["p_holm"] - e["p_holm"]) <= 1e-12
            b = hb[key]
            other = e["method"] if H == "H2" else f"{e['a']} minus {e['b']}"
            pr = ("<= " if b["p_holm_is_upper_bound"] else "") + f"{b['p_holm']:.3f}"
            tests.append({"log": log, "comparison": k, "method": other, "diff": e["diff"], "ci_lo": e["ci_lo"], "ci_hi": e["ci_hi"], "p": e["p"], "p_at_resolution": b["at_resolution"],
                          "p_holm": b["p_holm"], "p_holm_is_upper_bound": b["p_holm_is_upper_bound"], "p_holm_print": pr})
            base = f"EoneFx{SHORT[log]}{hw}{e1_analyze_camel(k)}"
            put(base + "PHolmPrint", pr, f"{log}: {H}, {other}: Holm-adjusted p as it is to be printed ('<=' marks an upper bound: the bootstrap p sits at its resolution 1 / {N_BOOT})")
            put(base + "PHolmIsBound", bool(b["p_holm_is_upper_bound"]), f"{log}: {H}, {other}: true if the Holm-adjusted p is an upper bound (bootstrap p at the resolution 1 / {N_BOOT} = {floor})")
            if b["p_holm_is_upper_bound"]:
                put(base + "PHolmBound", b["p_holm"], f"{log}: {H}, {other}: upper bound of the Holm-adjusted p ({fam['n_tests']} tests x resolution {floor} of the bootstrap p, or the running maximum of Holm)")
        out["holm"][H] = {"n_tests": fam["n_tests"], "resolution_of_the_bootstrap_p": floor, "n_boot": N_BOOT, "smallest_possible_adjusted_p": fam["n_tests"] * floor,
                          "rule": "p_two_sided of the paired crossed bootstrap is max(share of resamples on the other side of 0, doubled; 1 / n_boot): a value at 1 / n_boot means "
                                  "that no resample crossed 0, so p and every Holm value defined by it are upper bounds", "tests": tests,
                          "n_upper_bounds": sum(t["p_holm_is_upper_bound"] for t in tests)}
        put(f"EoneFx{hw}HolmFloor", fam["n_tests"] * floor, f"{H}: smallest Holm-adjusted p the bootstrap can show = {fam['n_tests']} tests x resolution {floor} (1 / {N_BOOT} resamples)")
        put(f"EoneFx{hw}NHolmBounds", out["holm"][H]["n_upper_bounds"], f"{H}: number of Holm-adjusted values (of {fam['n_tests']}) that are upper bounds")
    put("EoneFxBootResolution", floor, f"resolution of every bootstrap p-value: 1 / {N_BOOT} resamples")
    check(2, "(c) Holm values recomputed here equal the stored p_holm of out/endpoints.json (H2 and H3)", agree)
    fam = EP["holm"]["H2"]
    lst = []
    for key in fam["family"]:
        log, k = key.split("/")
        lst.append({"log": log, "comparison": k, "method": EP[log]["H2"][k]["method"], "calibrated_baseline": EP[log]["H2"][k].get("calibrated_baseline")})
    per_log = {log: [x["method"] for x in lst if x["log"] == log] for log in DEV}
    out["H2_family"] = {"n_comparisons": fam["n_tests"], "list": lst, "per_log": per_log,
                        "not_in_the_family": "the calibrated z-score gate and the published forms of the ball and router gates are gate level only (reduced programme; PROTOCOL.md, amendment "
                                             "of 2026-11-12 00:48 UTC, decision 2); the comparisons are listed in out/endpoints.json, key holm.H2.family",
                        "outcome": "(T-Gen + I-Gen) / 2 exact of ReScope minus the other method; paired crossed bootstrap, 3 calibration draws x 100 edits"}
    put("EoneFxHtwoNComparisons", fam["n_tests"], "H2: number of comparisons in the Holm family (out/endpoints.json, holm.H2.n_tests): 2 development logs x the methods of EoneFxHtwoComparisonsList")
    for log in DEV:
        put(f"EoneFx{SHORT[log]}HtwoComparisonsList", ", ".join(per_log[log]), f"{log}: the methods ReScope is compared with in H2 (out/endpoints.json)")
        put(f"EoneFx{SHORT[log]}HtwoNComparisons", len(per_log[log]), f"{log}: number of H2 comparisons on this log")
    check(2, "(d) the H2 family holds 18 comparisons (9 per development log)", fam["n_tests"] == 18 and len(lst) == 18 and all(len(x) == 9 for x in per_log.values()), fam["n_tests"])
    jdump(out, os.path.join(FIX, "reporting.json"))
    return out


def e1_analyze_camel(s):
    import re
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(w[:1].upper() + w[1:] for w in "".join(words.get(ch, ch) if ch.isdigit() else ch for ch in s).split())


# ----------------------------------------------------------------------------- item 3: rows from E2
KEY = ("log", "part", "method", "variant", "n_label", "draw", "pool", "metric")


def rows_from_e2():
    recs = jload(E2 + "/results.json")
    J, JL = {}, {}
    for r in recs:
        k = tuple(str(r.get(k)) if k == "n_label" else r.get(k) for k in KEY)
        J[k] = r                                                         # the last record of a key, as the renderers of E2 and E1 read it
        JL.setdefault(k, []).append(r)
    trip = lambda r: tuple(r.get(f) for f in ("value", "ci_lo", "ci_hi"))        # noqa: E731
    dup = {k: v for k, v in JL.items() if len({trip(r) for r in v}) > 1}
    meta = jload(E2 + "/results_meta.json")
    e2pn = jload(E2 + "/paper_numbers.json")
    slim = lambda r: None if r is None else {k: r.get(k) for k in ("value", "ci_lo", "ci_hi", "n_units", "n_draws", "per_draw", "interval") if r.get(k) is not None or k == "value"}   # noqa: E731
    rows, cmp_pn, bad_pn = [], 0, []

    def pn_check(name, r):
        nonlocal cmp_pn
        if name not in e2pn:
            bad_pn.append({"name": name, "problem": "not in e2_curves/out/paper_numbers.json"})
            return
        cmp_pn += 1
        if r is None or e2pn[name] != r["value"]:
            bad_pn.append({"name": name, "paper_number": e2pn[name], "record": None if r is None else r["value"]})

    for log in DEV:
        sh = SHORT[log]
        R = lambda part, method, variant, nl, metric, pool, draw=None: J.get((log, part, method, variant, str(nl), draw, pool, metric))     # noqa: E731
        # (b) least repair from the log-only thresholds
        for nl, wn in (("10", "Ten"), ("ALL", "All")):
            a = ("b", "where_least", "logonly", nl)
            cells = {c: slim(R(*a, m + "@mean", p)) for c, m, p in
                     [("Rel", "edit.rel.exact", "edits"), ("T-Gen", "edit.tgen.exact", "edits"), ("I-Gen", "edit.igen.exact", "edits"), ("rephrase", "edit.rephrase.exact", "edits"),
                      ("fire hd", "probe.fired", "heldout_disjoint"), ("D1 hd", "probe.D1", "heldout_disjoint"), ("D1 strict", "probe.D1", "heldout_strict"),
                      ("T-Loc*", "probe.T-Loc*", "heldout_disjoint"), ("I-Loc*", "probe.I-Loc*", "heldout_disjoint"), ("D2", "d2.D2", "heldout_disjoint"),
                      ("D2 n pre-correct", "d2.n_pre_correct", "heldout_disjoint"), ("refused", "meta.n_refused", "repair"), ("own out", "edit.rel.n_own_out_of_scope", "edits"),
                      ("still firing", "meta.n_reported_still_firing", "repair")] +
                     [(f"kg {m} {f}", f"kg.{m}.{f}", "edits") for m in ("rel", "tgen", "igen") for f in ("kept", "gained", "lost")] +
                     [(f"{k} fire (gate)", f"gate.hn.{k}.fired", "hard_negatives") for k in HN] + [(f"{k} D1", f"hn.{k}.D1", "hard_negatives") for k in HN]}
            r0 = R(*a, "probe.D1@mean", "heldout_disjoint")
            rows.append({"log": log, "part": "b", "row": f"ReScope-W, least repair from the log-only thresholds, n = {nl} reported complaints", "n_evidence": nl,
                         "n_streams": None if r0 is None else r0.get("n_draws"), "cells": cells})
            for suf, m, p in (("Done", "probe.D1@mean", "heldout_disjoint"), ("Rel", "edit.rel.exact@mean", "edits"), ("TGen", "edit.tgen.exact@mean", "edits"),
                              ("IGen", "edit.igen.exact@mean", "edits"), ("Refused", "meta.n_refused@mean", "repair"), ("OwnOut", "edit.rel.n_own_out_of_scope@mean", "edits")):
                pn_check(f"Etwo{sh}WLog{wn}{suf}", R(*a, m, p))
        # (c') two-block thresholds
        a = ("cw", "where_cal", "two_block", "200")
        cells = {c: slim(R(*a, m + "@mean", p)) for c, m, p in
                 [("Rel", "edit.rel.exact", "edits"), ("T-Gen", "edit.tgen.exact", "edits"), ("I-Gen", "edit.igen.exact", "edits"), ("rephrase", "edit.rephrase.exact", "edits"),
                  ("fire hd", "probe.fired", "heldout_disjoint"), ("fire hd (gate, 20 draws)", "gate.fired", "heldout_disjoint"), ("fire hd literal (gate)", "gate.fired_literal", "heldout_disjoint"),
                  ("capped", "gate.n_edits_capped", "edits"), ("D1 hd", "probe.D1", "heldout_disjoint"), ("D1 strict", "probe.D1", "heldout_strict"),
                  ("T-Loc*", "probe.T-Loc*", "heldout_disjoint"), ("I-Loc*", "probe.I-Loc*", "heldout_disjoint"), ("D2", "d2.D2", "heldout_disjoint"),
                  ("D2 n pre-correct", "d2.n_pre_correct", "heldout_disjoint")] +
                 [(f"kg {m} {f}", f"kg.{m}.{f}", "edits") for m in ("rel", "tgen", "igen") for f in ("kept", "gained", "lost")] +
                 [(f"{k} D1", f"hn.{k}.D1", "hard_negatives") for k in HN]}
        for k in HN:                                                     # two records under one key in E2's results.json: both are given, labelled
            both = JL.get((log, "cw", "where_cal", "two_block", "200", None, "hard_negatives", f"gate.hn.{k}.fired@mean"), [])
            cells[f"{k} fire (gate, 20 draws)"] = slim(next((r for r in both if r.get("n_units") == 20), None))
            cells[f"{k} fire (gate, draws 0-2)"] = slim(next((r for r in both if r.get("n_draws") == 3), None))
        rows.append({"log": log, "part": "c'", "row": "ReScope-W, two-block thresholds from 200 benign queries (alpha 0.05)", "n_evidence": "200", "n_streams": 3, "cells": cells})
        for suf, m, p in (("Fire", "gate.fired@mean", "heldout_disjoint"), ("Capped", "gate.n_edits_capped@mean", "edits"), ("Done", "probe.D1@mean", "heldout_disjoint"),
                          ("Rel", "edit.rel.exact@mean", "edits"), ("TGen", "edit.tgen.exact@mean", "edits"), ("IGen", "edit.igen.exact@mean", "edits"), ("Dtwo", "d2.D2@mean", "heldout_disjoint")):
            pn_check(f"Etwo{sh}WCal{suf}", R(*a, m, p))
        # (d) linear rows
        dkeys = sorted({(k[2], k[3], k[4]) for k in J if k[0] == log and k[1] == "d"})
        for method, variant, nl in dkeys:
            a = ("d", method, variant, nl)
            cells = {c: slim(R(*a, m, p, 0)) for c, m, p in
                     [("Rel", "edit.rel.exact", "edits"), ("T-Gen", "edit.tgen.exact", "edits"), ("I-Gen", "edit.igen.exact", "edits"), ("rephrase", "edit.rephrase.exact", "edits"),
                      ("D1 hd", "probe.D1", "heldout_disjoint"), ("D1 strict", "probe.D1", "heldout_strict"), ("T-Loc*", "probe.T-Loc*", "heldout_disjoint"),
                      ("I-Loc*", "probe.I-Loc*", "heldout_disjoint"), ("D2", "d2.D2", "heldout_disjoint"), ("D2 n pre-correct", "d2.n_pre_correct", "heldout_disjoint")] +
                     [(f"kg {m} {f}", f"kg.{m}.{f}", "edits") for m in ("rel", "tgen", "igen") for f in ("kept", "gained", "lost")] +
                     [(f"{k} D1", f"hn.{k}.D1", "hard_negatives") for k in HN] + [(f"gen {m} eq", f"gen.{m}.eq", "edits") for m in ("rel", "tgen", "igen")] +
                     [(f"gen {m} contains", f"gen.{m}.contains", "edits") for m in ("rel", "tgen", "igen")]}
            rows.append({"log": log, "part": "d", "row": f"{method}, n = {nl} reported complaints ({variant}; seed 0)", "method": method, "variant": variant, "n_evidence": nl, "n_streams": 1, "cells": cells})
            wn = {"10": "Ten", "ALL": "All"}[nl]
            nm = f"Etwo{sh}Row{method.title().replace('_', '')}{wn}"
            for suf, m, p in (("Done", "probe.D1", "heldout_disjoint"), ("Rel", "edit.rel.exact", "edits"), ("Dtwo", "d2.D2", "heldout_disjoint")):
                pn_check(nm + suf, R(*a, m, p, 0))
            if nl == "ALL" and method in ("COMP_B", "REEDIT_B"):
                for m, mw in (("rel", "Rel"), ("tgen", "TGen"), ("igen", "IGen")):
                    r = R(*a, f"gen.{m}.eq", "edits", 0)
                    put(f"EoneFx{sh}Row{method.title().replace('_', '')}AllGen{mw}Eq", None if r is None else r["value"],
                        f"{log}: {method} at all reported complaints ({variant}), generated answer of the {m} prompts equals the target (% of edits; decoder v2; e2_curves/out/results.json, part d, metric gen.{m}.eq, draw 0)")
    check(3, "every value of the re-rendered rows equals the paper number of E2 (e2_curves/out/paper_numbers.json) exactly", not bad_pn and cmp_pn > 0, f"{cmp_pn} paper numbers compared; differing: {bad_pn[:5]}")
    # the cells E1's table printed earlier (out/e2_rows_used.json: the records read at 2026-11-11 22:49)
    used = jload(os.path.join(OUT, "e2_rows_used.json"))
    bad_used, n_used = [], 0
    for r in used["records"]:
        k = tuple(str(r.get(x)) if x == "n_label" else r.get(x) for x in KEY)
        cur = JL.get(k, [])
        n_used += 1
        if trip(r) not in [trip(x) for x in cur]:
            bad_used.append({"key": list(k), "then": list(trip(r)), "now": [list(trip(x)) for x in cur]})
    check(3, "every E2 record that E1's table (2b) was rendered from (out/e2_rows_used.json) equals the final E2 record (value and interval)", not bad_used and n_used > 0,
          f"{n_used} records compared; differing: {bad_used[:5]}")
    # the block (2b) of out/table.txt, rendered again with E1's own function from the final E2 files, must be found verbatim
    lin, cw = jload(E2 + "/linear_rows.json"), jload(E2 + "/rescope_w_rows.json")
    table = open(os.path.join(OUT, "table.txt")).read()
    blocks, found = {}, {}
    n_side, bad_side = 0, 0
    for r in lin + cw:
        k = tuple(str(r.get(x)) if x == "n_label" else r.get(x) for x in KEY)
        n_side += 1
        if trip(r) not in [trip(x) for x in JL.get(k, [])]:
            bad_side += 1
    check(3, "linear_rows.json and rescope_w_rows.json of E2 hold the same values as its results.json", bad_side == 0, f"{n_side} records; differing: {bad_side}")
    for log in DEV:
        rr = []
        for nme, src, flt, mean in (("COMP-B, all reported complaints (E2: rebuilt, seed 0)", lin, lambda r: r["method"] == "COMP_B" and r.get("n_label") == "ALL", False),
                                    ("REEDIT-B, all reported complaints (E2: saved weight of P12)", lin, lambda r: r["method"] == "REEDIT_B" and r.get("n_label") == "ALL", False),
                                    ("ReScope-W, two-block thresholds (E2 c'; draws 0-2)", cw, lambda r: r["method"] == "where_cal", True)):
            rs = [r for r in src if r["log"] == log and flt(r)]
            d = {}
            for r in rs:
                m = r["metric"]
                if mean:
                    if r["draw"] is None and m.endswith("@mean"):
                        d[(m[:-5], r["pool"])] = r
                    elif r["draw"] == 0 and m.startswith(("d2.", "gen.", "hngen.")):
                        d.setdefault((m, r["pool"]), r)
                else:
                    d[(m, r["pool"])] = r
            rr.append((nme, d, rs[0].get("n") if rs else None))
        blocks[log] = e1_render.e2_table(rr)
        found[log] = blocks[log] in table
    check(3, "table (2b) of out/table.txt, rendered again with e1_render.e2_table from the final E2 files, is found verbatim in out/table.txt (both logs)", all(found.values()), found)
    dl = [{"key": list(k), "records": [{f: r.get(f) for f in ("value", "ci_lo", "ci_hi", "n_units", "n_draws", "per_draw")} for r in v]} for k, v in sorted(dup.items(), key=str)]
    check(3, "no key (log, part, method, variant, n, draw, pool, metric) of e2_curves/out/results.json holds two different records", not dup,
          f"{len(dup)} keys hold two different records: the near-scope firing of part c' at gate level (metric gate.hn.<kind>.fired@mean and gate.hn.mean_fired@mean) is stored once as "
          "the mean over the 20 gate-level draws and once as the mean over draws 0-2; both are listed in rows_from_e2.json (duplicate_keys) and printed, labelled; a renderer "
          "that indexes by key reads the later record")
    out = {"what": "the rows that the main table of E1 takes from E2, read again from the final records of e2_curves/out/results.json (parts b, c' and d of its table)",
           "read_from": E2, "read_at": time.strftime("%Y-%m-%d %H:%M:%S"), "e2_results_written": meta.get("written"), "e2_rows_read_by_e1_at": used.get("read_at"),
           "cells": "part b and c': mean over the complaint streams / calibration draws (per_draw lists them; an interval is the crossed bootstrap of E2); part d: one run (seed 0), "
                    "percentile bootstrap; gate columns of c': mean over 20 draws with the 5th and 95th percentile",
           "checks": {"e2_paper_numbers": {"n_compared": cmp_pn, "mismatches": bad_pn}, "e1_earlier_records": {"n_compared": n_used, "mismatches": bad_used},
                      "table_2b_block_found_verbatim": found, "row_files_equal_results": {"n_compared": n_side, "n_mismatches": bad_side},
                      "keys_with_two_different_records": len(dup)},
           "duplicate_keys": dl,
           "mismatch_rule": "a mismatch is reported here and nowhere repaired", "table_2b_rerendered": blocks, "rows": rows}
    jdump(out, os.path.join(FIX, "rows_from_e2.json"))
    return out


def main():
    os.makedirs(FIX, exist_ok=True)
    t0 = time.time()
    print("item 1", flush=True)
    CUR[0] = 1
    mcnemar()
    print(f"item 2  [{time.time() - t0:.0f}s]", flush=True)
    recs = jload(os.path.join(OUT, "results.json"))
    CUR[0] = 2
    reporting(recs)
    del recs
    print(f"item 3  [{time.time() - t0:.0f}s]", flush=True)
    CUR[0] = 3
    rows_from_e2()
    old = jload(os.path.join(OUT, "paper_numbers.json"))
    coll = sorted(set(PN) & set(old))
    check(0, "no name of out/fixup/paper_numbers_fixup.json exists in out/paper_numbers.json", not coll, f"collisions: {len(coll)}")
    check(0, "every paper number has a documentation line", set(PN) == set(DOC))
    jdump(dict(sorted(PN.items())), os.path.join(FIX, "paper_numbers_fixup.json"))
    jdump(dict(sorted(DOC.items())), os.path.join(FIX, "paper_numbers_fixup_doc.json"))
    by_item = {str(i): sum(v == i for v in ITEM_OF.values()) for i in (1, 2, 3)}
    jdump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint(), "n_paper_numbers": len(PN), "n_paper_numbers_by_item": by_item,
           "n_collisions_with_paper_numbers_json": len(coll), "n_checks": len(CHECKS), "n_checks_failed": sum(not c["ok"] for c in CHECKS), "checks": CHECKS},
          os.path.join(FIX, "checks.json"))
    print(f"paper numbers: {len(PN)}; collisions with out/paper_numbers.json: {len(coll)}; checks failed: {sum(not c['ok'] for c in CHECKS)} of {len(CHECKS)}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
