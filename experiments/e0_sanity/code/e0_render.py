"""E0: render out/results.json and out/table.txt from the JSON files written by the experiment scripts.

  python e0_render.py

Inputs (all written by code): out/gate_<log>.json (e0_gate.py), out/behaviour/<log>/ (e0_behaviour.py), out/referee.json
(e0_referee.py), out/checks.json (e0_checks.py).  Every number of the table is computed here from per-row flags by the
functions of e0_common (gate_stats, behaviour_stats, over_draws); nothing is typed by hand.
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)

import os
import time

import e0_common as C

import numpy as np  # noqa: E402

GATE_VARIANTS = ("tf", "tf_uncapped", "prompt", "decode")
ORDER = ("M1", "M2", "M3", "M4", "M5", "M6", "deployed", "pre-edit")
LABEL = {"M1": "M1 ncoord two-block (default)", "M2": "M2 ncoord shared scale", "M3": "M3 cosine two-block", "M4": "M4 cosine shared scale",
         "M5": "M5 cosine 0.65", "M6": "M6 exact (cos >= 0.999)", "deployed": "deployed log W_E", "pre-edit": "pre-edit W_0 (control)"}


def flags_of(e):
    return C.unbits(e["fired"]), C.unbits(e["own_cand"]), C.unbits(e["own_top"]), np.array(e["top_edit"])


def degenerate_sids(G):
    """lib sids of the rows whose pre-edit answer y0 is degenerate (empty, one character, '?', a bare None / No)."""
    return frozenset(r["lib_sid"] for r in G["rows"] if any((r.get("y0_deg") or {}).values()))


# ----------------------------------------------------------------------------- numbers
def gate_results(name):
    G = C.jload(os.path.join(C.OUT, f"gate_{name}.json"))
    info = G["rows"]
    rel = np.array([i for i, r in enumerate(info) if r["kind"] == "rel"])
    hn = [r for r in info if r["kind"].startswith("hn")]
    dj = [r for r in info if r.get("disjoint")]

    def deg(rows):
        return {k: sum(bool(r["y0_deg"][k]) for r in rows) for k in ("empty", "single_char", "qmark", "none_no")}
    counts = {"probe_valid": sum(r["kind"] == "probe" for r in info), "repair": G["n_repair_pool"],
              "heldout": sum(r.get("pool") == "heldout" for r in info), "heldout_disjoint": len(dj),
              "heldout_strict": sum(bool(r.get("strict")) for r in info), "edits": G["T"],
              "heldout_disjoint_question_in_repair": sum(bool(r.get("question_in_repair")) for r in dj),
              "hn": {t: sum(r["hn_type"] == t for r in hn) for t in C.HN_TYPES},
              "hn_flags": {"src_related_to_log": sum(bool(r["src_related_to_log"]) for r in hn), "q_edit": sum(r["q_edit"] is not None for r in hn),
                           "HN-I_q_edit": sum(r["q_edit"] is not None and r["hn_type"] == "HN-I" for r in hn),
                           "related_flag": sum(bool(r["related_flag"]) for r in hn), "same_category": sum(bool(r["same_category"]) for r in hn),
                           "new": sum(bool(r["new"]) for r in hn)},
              "y0_degenerate": {"heldout_disjoint": deg(dj), **{t: deg([r for r in hn if r["hn_type"] == t]) for t in C.HN_TYPES}},
              "lp_shift_rows": {"heldout_disjoint": sum(bool(r.get("lp_shift")) for r in dj), "repair": sum(bool(r.get("lp_shift")) for r in info if r.get("pool") == "repair"),
                                **{k: sum(bool(r.get("lp_shift")) for r in info if r["kind"] == k) for k in ("rel", "tgen", "igen")},
                                "hn": sum(bool(r.get("lp_shift")) for r in hn)}}
    ev_mask = np.array([r["kind"] in ("tgen", "igen") or r["kind"].startswith("hn") or bool(r.get("disjoint")) for r in info])   # reported rows, without Rel
    out = {"counts": counts, "cache": G["cache"], "code": G["code"], "methods": {},
           "own_b1_feature": {f: {"min": float(np.min(v["own_feature_b1_rel_key"])), "median": float(np.median(v["own_feature_b1_rel_key"])),
                                  "max": float(np.max(v["own_feature_b1_rel_key"]))} for f, v in G["features"].items()}}
    for mid in C.METHODS:
        m = G["methods"][mid]
        if "fixed" in m:
            e = m["fixed"]
            r = {v: C.gate_stats(info, *flags_of(e[v])) for v in ("tf", "prompt", "decode")}
            oc, ot = C.unbits(e["tf"]["own_cand"])[rel], C.unbits(e["tf"]["own_top"])[rel]
            r["own_prompts"] = {"n": int(rel.size), "silenced": int((~oc).sum()), "not_released": int((~ot).sum()),
                                "silenced_edits": [int(info[i]["edit"]) for i in rel[~oc]]}
            r["n_near"] = e["tf"]["n_near"]
            r["n_near_reported_rows"] = int((C.unbits(e["tf"]["near"]) & ev_mask).sum())
            r["threshold"] = e["tau"][0]
        else:
            ds = m["draws"]
            r = {v: C.summarise_draws([C.gate_stats(info, *flags_of(e[v])) for e in ds]) for v in GATE_VARIANTS}
            rec = [e["rec"] for e in ds]
            r["calibration"] = {"n1": rec[0]["n1"], "n2": rec[0]["n2"], "rank": rec[0]["rank"], "bound_false_firing_percent": 100.0 * rec[0]["bound_false_firing"],
                                "top1_rank": rec[0]["top1_rank"], "eta": C.over_draws([x["eta"] for x in rec]),
                                "n_edits_capped": C.over_draws([x["n_edits_capped"] for x in rec]),
                                "n_stored_keys_silenced_by_literal_rule": C.over_draws([x["n_own_keys_silenced_by_literal_rule"] for x in rec]),
                                "n_guards_silenced_by_literal_rule": C.over_draws([x["n_guards_silenced_by_literal_rule"] for x in rec]),
                                "n_guards": rec[0]["n_guards"], "n_guards_silenced_after_cap": C.over_draws([x["n_guards_silenced"] for x in rec]),
                                "n_draws_with_a_capped_edit": sum(x["n_edits_capped"] > 0 for x in rec),
                                "capped_edits_by_draw": {str(e["seed"]): e["rec"]["capped_edits"] for e in ds if e["rec"]["n_edits_capped"]}}
            ft = G["features"][m["spec"]["feature"]]
            capped = sorted({int(x) for e in ds for x in e["rec"]["capped_edits"]})
            r["calibration"]["capped_edits_repair_max"] = {str(i): {"largest_feature_of_a_REPAIR_row": ft["repair_max"][i], "row": ft["repair_argmax_sid"][i],
                                                                   "same_question_as_the_edit": ft["repair_argmax_same_question_as_edit"][i],
                                                                   "own_b1_feature": ft["own_feature_b1_rel_key"][i],
                                                                   "n_draws_capped": sum(i in e["rec"]["capped_edits"] for e in ds)} for i in capped}
            r["own_prompts"] = {"n": int(rel.size),
                                "silenced_before_cap": C.over_draws([int((~C.unbits(e["tf_uncapped"]["own_cand"])[rel]).sum()) for e in ds]),
                                "silenced_after_cap": C.over_draws([int((~C.unbits(e["tf"]["own_cand"])[rel]).sum()) for e in ds]),
                                "not_released_after_cap": C.over_draws([int((~C.unbits(e["tf"]["own_top"])[rel]).sum()) for e in ds])}
            r["n_near"] = sum(e["tf"]["n_near"] for e in ds)
            r["n_near_reported_rows"] = sum(int((C.unbits(e["tf"]["near"]) & ev_mask).sum()) for e in ds)
            r["n_flags_changed_by_cap_reported_rows"] = sum(int(((C.unbits(e["tf"]["fired"]) != C.unbits(e["tf_uncapped"]["fired"])) & ev_mask).sum()) for e in ds)
        r["spec"] = m["spec"]
        out["methods"][mid] = r
    return out, degenerate_sids(G)


def behaviour_results(name, degenerate):
    B = C.load_behaviour(name)
    if B is None:
        return None
    import rescope
    d = os.path.join(C.OUT, "behaviour", name)
    out = {"meta": C.jload(os.path.join(d, "meta.json")), "done": C.jload(os.path.join(d, "done.json")), "methods": {}}
    st = B["static"]
    for m in ("deployed", "pre-edit", "M5", "M6"):
        out["methods"][m] = {"single": C.behaviour_stats(st["edits"][m], st["probes"][m], st["hn"][m], degenerate)}
    for mid in C.CALIBRATED:
        per = [C.behaviour_stats(*(B[f"draw{d_}"][g][mid] for g in ("edits", "probes", "hn")), degenerate) for d_ in C.BEHAVIOUR_DRAWS]
        recs = [C.jload(os.path.join(d, f"gates_draw{d_}.json"))[mid]["rec"] for d_ in C.BEHAVIOUR_DRAWS]
        out["methods"][mid] = {"draws": {str(d_): p for d_, p in zip(C.BEHAVIOUR_DRAWS, per)}, "summary": C.summarise_draws(per),
                               "calibration": {str(d_): {"eta": r["eta"], "n_edits_capped": r["n_edits_capped"], "capped_edits": r["capped_edits"],
                                                         "bound_false_firing": r["bound_false_firing"]} for d_, r in zip(C.BEHAVIOUR_DRAWS, recs)}}
    sm = rescope.stored_metrics(name, pools=False)                    # the deployed log's own stored metrics (no model)
    dep = out["methods"]["deployed"]["single"]["probe"]["heldout_disjoint"]
    out["deployed_vs_stored"] = {"stored_locality": sm["locality"], "stored_edits_vendor_pass_batch8": {k: v["exact"] for k, v in sm["edits"].items()},
                                 "signed_diff_T-Loc*_minus_T-Loc": None if sm["locality"].get("T-Loc") is None else dep["T-Loc*"] - sm["locality"]["T-Loc"],
                                 "signed_diff_I-Loc*_minus_I-Loc": None if sm["locality"].get("I-Loc") is None else dep["I-Loc*"] - sm["locality"]["I-Loc"],
                                 "note": "different quantities: decoder-v2 answer at B = 1 on valid HELD-OUT-DISJOINT rows versus the benchmark's "
                                         "reference answer in padded batches of 8 on all 500 items"}
    return out


# ----------------------------------------------------------------------------- text
def table(headers, rows, align=None):
    cols = list(zip(*([headers] + rows)))
    w = [max(len(str(x)) for x in c) for c in cols]
    align = align or ["l"] + ["r"] * (len(headers) - 1)

    def line(r):
        return "  ".join(str(x).ljust(w[i]) if align[i] == "l" else str(x).rjust(w[i]) for i, x in enumerate(r)).rstrip()
    return "\n".join("    " + x for x in [line(headers), "  ".join("-" * x for x in w)] + [line(r) for r in rows])


def f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def dr(d, nd=1):
    """A statistic over the calibration draws: mean [5th, 95th percentile]."""
    if d is None or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{nd}f} [{d['p05']:.{nd}f}, {d['p95']:.{nd}f}]"


def rg(d, nd=1):
    """A statistic over the three behaviour draws: mean [min, max]."""
    if d is None or d.get("mean") is None:
        return "-"
    return f"{d['mean']:.{nd}f} [{d['min']:.{nd}f}, {d['max']:.{nd}f}]"


def render_gate(g):
    c = g["counts"]
    L = [f"(1) GATE LEVEL -- key cache, CPU, {C.N_DRAWS} seeded calibration draws; cells: mean [5th, 95th percentile over draws]; M5 / M6: one fixed gate.",
         f"    fire = % of the {c['heldout_disjoint']} valid HELD-OUT-DISJOINT rows on which an edit fires ('uncapped' = the literal conformal rule, the event the bound covers).",
         "    coverage = % of the 100 T-Gen / I-Gen sequences on which their OWN edit fires = is the top-1 candidate and releases ('cand' = is a candidate, top-1 or not).",
         "    own prompts = the 100 B = 1 Rel prompts: number on which the own edit is NOT a candidate before / after the guard cap; 'not top-1' = number not released by",
         "    the own edit after the cap (another edit has the larger margin)."]
    rows = []
    for mid in C.METHODS:
        m = g["methods"][mid]
        if mid in C.FIXED:
            t, op = m["tf"], m["own_prompts"]
            rows.append([LABEL[mid], f(t["probe"]["heldout_disjoint"]["fired"], 2), "-", f(t["edit"]["tgen"]["own_top"]), f(t["edit"]["tgen"]["own_cand"]),
                         f(t["edit"]["igen"]["own_top"]), f(t["edit"]["igen"]["own_cand"]), f"{op['silenced']} (no cap)", "-", str(op["not_released"]), "-"])
        else:
            t, u, op, cal = m["tf"], m["tf_uncapped"], m["own_prompts"], m["calibration"]
            rows.append([LABEL[mid], dr(t["probe"]["heldout_disjoint"]["fired"], 2), dr(u["probe"]["heldout_disjoint"]["fired"], 2),
                         dr(t["edit"]["tgen"]["own_top"]), dr(t["edit"]["tgen"]["own_cand"]), dr(t["edit"]["igen"]["own_top"]), dr(t["edit"]["igen"]["own_cand"]),
                         dr(op["silenced_before_cap"]), dr(op["silenced_after_cap"], 0), dr(op["not_released_after_cap"]), dr(cal["n_edits_capped"])])
    L.append(table(["method", "fire % (as run)", "fire % uncapped", "T-Gen own fires", "T-Gen own cand", "I-Gen own fires", "I-Gen own cand",
                    "own silenced before cap", "after cap", "own not top-1", "edits capped"], rows))
    L.append("")
    n = c["hn"]
    L.append(f"    hard negatives of the first 100 edits, valid rows (HN-I n = {n['HN-I']}, HN-Q n = {n['HN-Q']}, HN-X n = {n['HN-X']}): % of rows on which AN edit fires / on which the")
    L.append("    row's OWN edit fires (own = the edit whose image (HN-I) or question (HN-Q) the row uses; HN-X: own a = the image's edit, own b = the question's edit).")
    rows = []
    for mid in C.METHODS:
        h = g["methods"][mid]["tf"]["hn"]
        fn = f if mid in C.FIXED else dr
        rows.append([LABEL[mid], fn(h["HN-I"]["fired"]), fn(h["HN-I"]["own_top"]), fn(h["HN-Q"]["fired"]), fn(h["HN-Q"]["own_top"]),
                     fn(h["HN-X"]["fired"]), fn(h["HN-X"]["own_top"]), fn(h["HN-X"]["b_top"])])
    L.append(table(["method", "HN-I an edit", "HN-I own", "HN-Q an edit", "HN-Q own", "HN-X an edit", "HN-X own a", "HN-X own b"], rows))
    L.append("")
    L.append("    further gate-level numbers: fire % on other probe pools and at the other positions of the SAME gate (calibrated on teacher-forced keys): '@prompt' = the")
    L.append("    decode's token in the teacher-forced sequence, '@decode' = the prompt-only key; Rel = % of own prompts on which the own edit fires; 'other edit' = % of")
    L.append("    rephrase sequences on which ANOTHER edit fires (wrong target).")
    rows = []
    for mid in C.METHODS:
        m = g["methods"][mid]
        fn = f if mid in C.FIXED else dr
        t = m["tf"]
        rows.append([LABEL[mid], fn(t["probe"]["heldout_strict"]["fired"], 2), fn(t["probe"]["heldout"]["fired"], 2), fn(t["probe"]["heldout_disjoint_image"]["fired"], 2),
                     fn(t["probe"]["heldout_disjoint_text"]["fired"], 2), fn(m["prompt"]["probe"]["heldout_disjoint"]["fired"], 2),
                     fn(m["decode"]["probe"]["heldout_disjoint"]["fired"], 2), fn(t["edit"]["rel"]["own_top"]), fn(t["edit"]["tgen"]["other_top"]),
                     fn(t["edit"]["igen"]["other_top"])])
    L.append(table(["method", "strict", "HELD-OUT (all)", "disjoint image", "disjoint text", "disjoint @prompt", "disjoint @decode", "Rel own fires",
                    "T-Gen other edit", "I-Gen other edit"], rows))
    L.append("")
    for mid in C.CALIBRATED:
        v = g["methods"][mid]["calibration"]
        L.append(f"    calibration {mid}: n1 = {v['n1']}, n2 = {v['n2']}, rank {v['rank']}, bound {v['bound_false_firing_percent']:.2f} %, top-1 by {v['top1_rank']}, eta {dr(v['eta'], 3)}; "
                 f"draws with a capped edit {v['n_draws_with_a_capped_edit']}/{C.N_DRAWS}" + (f" (draw: edits {v['capped_edits_by_draw']})" if 0 < v['n_draws_with_a_capped_edit'] <= 6 else "")
                 + f"; guards silenced by the literal rule (of {v['n_guards']}: stored keys + B = 1 Rel keys) {dr(v['n_guards_silenced_by_literal_rule'])}; reported rows "
                 f"(HELD-OUT-DISJOINT, T-Gen, I-Gen, hard negatives) within 1e-9 of a threshold, all draws: {g['methods'][mid]['n_near_reported_rows']}; firing flags of reported "
                 f"rows changed by the cap, all draws: {g['methods'][mid]['n_flags_changed_by_cap_reported_rows']}")
    ob = g["own_b1_feature"]
    L.append("    own B = 1 Rel key on its own detector (signed): " + "; ".join(f"{k} min {v['min']:.4f}, median {v['median']:.4f}, max {v['max']:.4f}" for k, v in ob.items())
             + f"; M6 misses the own prompt of edits {g['methods']['M6']['own_prompts']['silenced_edits']}")
    return L


def render_behaviour(b, g):
    if b is None:
        return ["(2) BEHAVIOUR -- the GPU run of this log is not complete."]
    ms = b["methods"]
    ref = ms["deployed"]["single"]
    L = [f"(2) BEHAVIOUR -- teacher forcing at B = 1 (hook on the GPU), calibration draws {list(C.BEHAVIOUR_DRAWS)}; cells of M1 - M4: mean [min, max] over the 3 draws.",
         "    Rel / T-Gen / I-Gen = exact % over all 100 edits; D1 = % of valid rows whose pre-edit answer y0 (multi-token, decoder v2) is not reproduced:",
         f"    HELD-OUT-DISJOINT n = {ref['probe']['heldout_disjoint']['n']}, HELD-OUT-STRICT n = {ref['probe']['heldout_strict']['n']}, hard negatives HN-I n = "
         f"{ref['hn']['HN-I']['n']}, HN-Q n = {ref['hn']['HN-Q']['n']}, HN-X n = {ref['hn']['HN-X']['n']}; T-Loc* / I-Loc* = token-level locality (%) on",
         f"    HELD-OUT-DISJOINT ({ref['probe']['heldout_disjoint']['n_text']} text / {ref['probe']['heldout_disjoint']['n_image']} image rows; rescope.locality)."]
    rows, rows2, rows3 = [], [], []
    for m in ORDER:
        s, fn = (ms[m]["summary"], rg) if "summary" in ms[m] else (ms[m]["single"], f)
        p, ps, e, h = s["probe"]["heldout_disjoint"], s["probe"]["heldout_strict"], s["edit"], s["hn"]
        rows.append([LABEL[m], fn(e["rel"]["exact"], 0), fn(e["tgen"]["exact"], 0), fn(e["igen"]["exact"], 0), fn(p["D1"], 2), fn(ps["D1"], 2),
                     fn(h["HN-I"]["D1"]), fn(h["HN-Q"]["D1"]), fn(h["HN-X"]["D1"]), fn(p["T-Loc*"], 2), fn(p["I-Loc*"], 2)])
        rows2.append([LABEL[m], fn(p["fired"], 2), fn(p["D1_given_fired"]), fn(e["tgen"]["own_top"]), fn(e["igen"]["own_top"]), fn(e["gen_mean"]["exact"]),
                      fn(e["rel"]["token"]), fn(e["tgen"]["token"]), fn(e["igen"]["token"]), fn(h["all"]["fired"]), fn(h["all"]["D1"]), fn(h["all"]["tok_loc"])])
        rows3.append([LABEL[m]] + [x for t in C.HN_TYPES for x in (fn(h[t]["D1_y0_not_degenerate"]), fn(h[t]["D1_y0_degenerate"]))] + [fn(p["D1_y0_not_degenerate"], 2)])
    L.append(table(["method", "Rel", "T-Gen", "I-Gen", "D1 % disjoint", "D1 % strict", "HN-I D1 %", "HN-Q D1 %", "HN-X D1 %", "T-Loc*", "I-Loc*"], rows))
    L.append("")
    L.append("    the same runs: firing on HELD-OUT-DISJOINT (hook), D1 among the firing rows, own edit fires on its rephrase, (T-Gen + I-Gen) / 2 exact, token accuracy of")
    L.append("    the edits (%), hard negatives pooled")
    L.append(table(["method", "fire % disjoint", "D1 % | fired", "T-Gen own fires", "I-Gen own fires", "(T+I)/2 exact", "Rel tok", "T-Gen tok", "I-Gen tok", "HN fire %",
                    "HN D1 %", "HN tok-loc %"], rows2))
    dg = g["counts"]["y0_degenerate"]
    L.append("")
    L.append("    degenerate pre-edit answers are counted: rows whose y0 is empty / one character / '?' / a bare None or No: HELD-OUT-DISJOINT "
             f"{sum(dg['heldout_disjoint'].values())} of {g['counts']['heldout_disjoint']} {dg['heldout_disjoint']}; "
             + "; ".join(f"{t} {ref['hn'][t]['n_y0_degenerate']} of {ref['hn'][t]['n']}" for t in C.HN_TYPES) + ".  D1 % without / on those rows:")
    L.append(table(["method", "HN-I not deg.", "HN-I deg.", "HN-Q not deg.", "HN-Q deg.", "HN-X not deg.", "HN-X deg.", "disjoint not deg."], rows3))
    L.append("")
    L.append(f"    hard negatives by subset: fire % (gate level, mean over the {C.N_DRAWS} draws) / D1 % (mean over the 3 draws); deployed log and M5: D1 %")
    rows4 = []
    for nm in C.HN_SUBSETS:
        row = [nm, str(ref["hn_subset"][nm]["n"])]
        for m in C.CALIBRATED:
            row.append(f"{f(g['methods'][m]['tf']['hn_subset'][nm]['fired']['mean'])} / {f(ms[m]['summary']['hn_subset'][nm]['D1']['mean'])}")
        row += [f(ms["M5"]["single"]["hn_subset"][nm]["D1"]), f(ref["hn_subset"][nm]["D1"])]
        rows4.append(row)
    L.append(table(["subset", "n", "M1 fire / D1", "M2 fire / D1", "M3 fire / D1", "M4 fire / D1", "M5 D1", "deployed D1"], rows4))
    dv = b["deployed_vs_stored"]
    L.append("")
    L.append(f"    deployed log: stored benchmark locality T-Loc {f(dv['stored_locality'].get('T-Loc'), 2)} / I-Loc {f(dv['stored_locality'].get('I-Loc'), 2)}; signed difference "
             f"T-Loc* - T-Loc = {f(dv['signed_diff_T-Loc*_minus_T-Loc'], 2)}, I-Loc* - I-Loc = {f(dv['signed_diff_I-Loc*_minus_I-Loc'], 2)} points (different quantities: other answer")
    L.append(f"    text, B = 1, valid HELD-OUT-DISJOINT rows); stored edit exact of the vendor pass (padded batches of 8): Rel {f(dv['stored_edits_vendor_pass_batch8']['rel'])}, "
             f"T-Gen {f(dv['stored_edits_vendor_pass_batch8']['tgen'])}, I-Gen {f(dv['stored_edits_vendor_pass_batch8']['igen'])}.")
    cal = "; ".join(f"{mid}: " + ", ".join(f"draw {d} eta {v['eta']:.4f} capped {v['n_edits_capped']}" for d, v in ms[mid]["calibration"].items()) for mid in C.CALIBRATED)
    L.append("    gates built on the GPU: " + cal)
    nf = sum(v.get("n_nonfinite", 0) for m in ms.values() for s in ([m["single"]] if "single" in m else list(m["draws"].values()))
             for grp in s.values() for v in grp.values())
    L.append(f"    rows with a non-finite logit, all methods and draws: {nf}.  E0 runs no free generation (teacher forcing only), so there are no generated outputs to count.")
    return L


def render_referee(r):
    if r is None:
        return ["(3) REFEREE -- out/referee.json not found."]
    k = r["keys"]
    L = ["(3) M1 / M2 AGAINST THE THEORY REFEREE (rf2_c6_checks.py, section D4, alpha = 5 %) -- one difference at a time; cells: mean over halves / draws +- standard error.",
         f"    positions: {k['keys_bitwise_equal']}/{k['n_in_cache']} of the P11 sequences he read are bitwise the cache's teacher-forced keys, last prompt index equal "
         f"{k['last_prompt_index_equal']}/{k['n_in_cache']}: NO difference of position.",
         f"    his script re-run unchanged: largest difference to his stored numbers {r['rerun_max_abs_diff_REAL_section']:.1e}; the library's rule on his own 300 halves: "
         f"largest difference to his numbers {r['library_on_his_halves_max_abs_diff_to_his_numbers']:.1e} points."]
    for cal, title in (("two_block", "two-block rule -> M1"), ("shared", "shared scale -> M2")):
        rows = []
        for step, v in r["ladder"][cal].items():
            if isinstance(v["firing"], dict):
                def pm(d, nd):
                    return f"{d['mean']:.{nd}f}" + (f" +- {d['se']:.{nd}f}" if v["n_draws"] > 1 else "")
                rows.append([step, v["what"], str(v["n_draws"]), str(v["n_cal"]), str(v["n_test"]), pm(v["firing"], 2), pm(v["cov_tgen"], 1), pm(v["cov_igen"], 1),
                             pm(v["cov_rel"], 2), f"{v['mean_eta']:.3f}", f"{v['mean_edits_capped']:.2f}"])
            else:
                rows.append([step, v["what"], str(r["referee"]["resplits"]) if cal == "two_block" else "1", str(r["referee"]["n_fit"]), str(r["referee"]["n_test"]),
                             f"{v['firing']:.2f}", f"{v['cov_tgen']:.1f}", f"{v['cov_igen']:.1f}", f"{v['cov_rel']:.2f}", "", ""])
        L.append(f"    {title}")
        L.append(table(["step", "what changes", "halves/draws", "n cal", "n read-out", "rows firing %", "T-Gen cov %", "I-Gen cov %", "Rel cov %", "eta", "edits capped"],
                       rows, align=["l", "l"] + ["r"] * 9))
        L.append("")
    return L


def explain(r):
    """The differences M1 / M2 minus referee, attributed step by step (differences of the ladder means; points)."""
    out = {}
    for cal in ("two_block", "shared"):
        lad = r["ladder"][cal]

        def val(step, key):
            v = lad[step][key]
            return v if isinstance(v, float) else v["mean"]
        chain = [("Monte Carlo error of the referee's 300 halves (his number -> expectation over halves)" if cal == "two_block"
                  else "library rule in the referee's setting (no randomness)", "R0", "R1"),
                 (f"POOL, calibration: all valid REPAIR rows -> n = {C.N_CAL} drawn at random (expectation over draws)", "R1", "R2"),
                 (f"the {C.N_DRAWS} draws of this experiment instead of the expectation (draw noise)", "R2", "R2s"),
                 ("POOL, read-out: HELD-OUT -> HELD-OUT-DISJOINT (25 rows that repeat a REPAIR query removed)", "R2s", "R3"),
                 ("GUARDS: B = 1 own Rel keys as guards, capped thresholds", "R3", "R4"),
                 ("TOP-1: coverage counts the own edit only if it is the top-1 candidate", "R4", "R5")]
        keys = ("firing", "cov_tgen", "cov_igen", "cov_rel")
        out[cal] = {"steps": [{"what": w, "from": a, "to": b, **{k: val(b, k) - val(a, k) for k in keys}} for w, a, b in chain],
                    "positions": {k: 0.0 for k in keys},
                    "total": {k: val("R5", k) - val("R0", k) for k in keys},
                    "referee": {k: val("R0", k) for k in keys}, "this_experiment": {k: val("R5", k) for k in keys}}
    return out


def render_explain(ex, r):
    L = ["    attribution of the difference to the referee, in points (rows firing % / T-Gen coverage % / I-Gen coverage %); the steps add up to the total:"]
    for cal, title in (("two_block", "M1 (two-block)"), ("shared", "M2 (shared scale)")):
        e = ex[cal]
        L.append(f"    {title}: referee {e['referee']['firing']:.2f} / {e['referee']['cov_tgen']:.1f} / {e['referee']['cov_igen']:.1f}  ->  this experiment "
                 f"{e['this_experiment']['firing']:.2f} / {e['this_experiment']['cov_tgen']:.1f} / {e['this_experiment']['cov_igen']:.1f}   (total "
                 f"{e['total']['firing']:+.2f} / {e['total']['cov_tgen']:+.1f} / {e['total']['cov_igen']:+.1f})")
        for s in e["steps"]:
            L.append(f"        {s['firing']:+.2f} / {s['cov_tgen']:+.1f} / {s['cov_igen']:+.1f}   {s['what']}")
        L.append("        +0.00 / +0.0 / +0.0   POSITIONS: the same teacher-forced keys, bit for bit")
    mc = r["referee_monte_carlo"]
    L.append("    the referee's 300 halves against the expectation over halves (same rule, same keys, same pools; only the halves differ):")
    for k, lab in (("firing", "rows firing %"), ("cov_tgen", "T-Gen coverage %"), ("cov_igen", "I-Gen coverage %")):
        v = mc[k]
        L.append(f"        {lab}: his 300 halves {v['referee_300_halves']:.2f}, expectation {v['expectation']:.2f}, standard error of a 300-halves mean {v['se_of_a_300_halves_mean']:.3f} "
                 f"-> z = {v['z_of_referee']:+.2f}; of {v['n_blocks_of_300']} disjoint blocks of 300 seeded halves (range {v['block_means_min']:.2f} .. {v['block_means_max']:.2f}) "
                 f"{v['n_blocks_at_least_as_far_from_the_expectation']} lie as far from the expectation")
    tw = r["two_block_minus_shared"]
    L.append("    two-block MINUS shared scale (signed; mean +- standard error over calibrations):")
    for key, lab in (("all_REPAIR_rows_referee_readout", "referee's setting (all REPAIR rows, HELD-OUT, candidate)"),
                     ("n200_referee_readout", f"n = {C.N_CAL}, referee's read-out, paired on the same draws"),
                     ("n200_M1_minus_M2_as_defined", f"M1 - M2 as defined (n = {C.N_CAL}, HELD-OUT-DISJOINT, guards, top-1), paired on the same draws")):
        v = tw[key]
        L.append(f"        {lab} ({v['firing']['n_draws']} calibrations): rows firing {v['firing']['mean']:+.2f} +- {v['firing']['se']:.2f}, T-Gen coverage "
                 f"{v['cov_tgen']['mean']:+.2f} +- {v['cov_tgen']['se']:.2f}, I-Gen coverage {v['cov_igen']['mean']:+.2f} +- {v['cov_igen']['se']:.2f}")
    m = r["m1_m2_many_draws"]
    L.append(f"    M1 / M2 as defined over {m['two_block']['n_draws']} draws (expectation; the gate table has {C.N_DRAWS}): M1 rows firing {m['two_block']['firing']['mean']:.2f} % "
             f"(5th - 95th percentile {m['two_block']['firing']['p05']:.2f} - {m['two_block']['firing']['p95']:.2f}), T-Gen {m['two_block']['cov_tgen']['mean']:.1f}, I-Gen "
             f"{m['two_block']['cov_igen']['mean']:.1f}; M2 rows firing {m['shared']['firing']['mean']:.2f} % ({m['shared']['firing']['p05']:.2f} - {m['shared']['firing']['p95']:.2f}), "
             f"T-Gen {m['shared']['cov_tgen']['mean']:.1f}, I-Gen {m['shared']['cov_igen']['mean']:.1f}; bound {m['two_block']['bound_false_firing_percent']:.2f} % / "
             f"{m['shared']['bound_false_firing_percent']:.2f} %.")
    return L


def render_checks(ch):
    if ch is None:
        return ["(4) CHECKS -- out/checks.json not found."]
    a, b, c, d = ch["A_naive"], ch["B_ladder"], ch["C_hook"], ch["D_behaviour"]
    L = ["(4) CHECKS (e0_checks.py)",
         f"    A  naive numpy re-implementation of M1 - M6 from the raw files against the library: {a['n_differing']} of {a['n_decisions']} per-row decisions differ "
         f"(all {C.N_DRAWS} draws, capped and uncapped) -> {'OK' if a['ok'] else 'FAIL'}",
         f"    B  last ladder step against the gate-level table: largest difference {b.get('max_abs_diff', float('nan')):.1e} -> {'OK' if b.get('ok') else 'FAIL / not available'}"]
    if c.get("available"):
        L.append(f"    C  GPU hook against the CPU decisions from the key cache: {c['n_differing']} of {c['n_flags']} firing flags differ (draws 0 - 2, M1 - M6: fired, top edit, own "
                 f"candidate, own release); thresholds built on the GPU against the CPU: largest relative difference {c['max_rel_tau_diff']:.1e}, eta {c['max_abs_eta_diff']:.1e}; "
                 f"same calibration draws {c['draws_equal']}; same rows {all(v['same_set'] for v in c['row_sets'].values())} -> {'OK' if c['ok'] else 'FAIL'}")
        L.append(f"    D  deployed D1 flags == the testbed's stored D1 complaints on {d['deployed_D1_equals_stored_D1']['equal']}/{d['deployed_D1_equals_stored_D1']['n']} rows; "
                 f"pre-edit control: probe D1 {d['pre_edit_probe_D1']}, hard-negative D1 {d['pre_edit_hn_D1']}, token locality all 1: {d['pre_edit_tok_loc_all_1']}, edit rows equal "
                 f"to the cache's pre-edit outcome {d['pre_edit_edit_exact_equals_cache']['equal']}/{d['pre_edit_edit_exact_equals_cache']['n']}; silent rows of the gates equal to "
                 f"the pre-edit rows {d['silent_rows']['equal_to_pre_edit']}/{d['silent_rows']['n']}; rows with a non-finite logit {d['n_nonfinite_rows']} -> {'OK' if d['ok'] else 'FAIL'}")
    else:
        L.append("    C, D  not available (the GPU run of this log is not complete)")
    e = ch.get("E_seed_coupling")
    if e:
        L.append(f"    E  one seed for the draw and for the two-block split against an independently seeded split ({e['n_draws']} draws, M1's rule): rows firing "
                 f"{e['same_seed']['firing']['mean']:.2f} vs {e['independent_seed']['firing']['mean']:.2f} % (z = {e['diff_firing']['z']:+.2f}), T-Gen coverage "
                 f"{e['same_seed']['cov_tgen']['mean']:.1f} vs {e['independent_seed']['cov_tgen']['mean']:.1f} % (z = {e['diff_cov_tgen']['z']:+.2f}), image rows in block 1 minus "
                 f"block 2 {e['same_seed']['image_rows_block1_minus_block2']['mean']:+.2f} vs {e['independent_seed']['image_rows_block1_minus_block2']['mean']:+.2f} -> "
                 f"{'OK (no difference)' if e['ok'] else 'DIFFERENCE'}")
    return L


def library_tests():
    """The library's own test results (rescope/tests/out/*.json, read only): how many carry the fingerprint of the code used here."""
    import glob
    now = C.code_fingerprint()["rescope"]
    n = n_pass = n_same = 0
    for p in glob.glob(os.path.join(C.LIB, "rescope", "tests", "out", "*.json")):
        try:
            d = C.jload(p)
        except ValueError:
            continue
        if isinstance(d, dict) and "pass" in d and "code" in d:
            n += 1
            n_pass += bool(d["pass"])
            n_same += d["code"] == now
    if n == 0:                                                          # ARTIFACT: do not rewrite the table with "0 of its 0 stored test results"
        raise SystemExit("e0_render: input missing: the stored test results of the library, rescope/tests/out/*.json, are not part of the repository "
                         "(they are written by `python rescope/tests/run_tests.py cpu` and `... gpu`; the table as delivered with the library is "
                         "rescope/tests/TABLE_delivered.md).\n  Run the library's tests first, then this renderer; its line on the library then counts "
                         "the results of YOUR run.  Nothing was written: out/table.txt and out/results.json are the authors' rendering.")
    return {"fingerprint_of_the_library_used": now, "n_test_results": n, "n_pass": n_pass, "n_with_this_fingerprint": n_same}


def restart_check(name):
    p = os.path.join(C.OUT, "behaviour", name, "restart_check.json")
    if not os.path.exists(p):
        return None
    d = C.jload(p)
    first = C.jload(os.path.join(C.OUT, "behaviour", name, "meta.json"))
    again = C.jload(os.path.join(C.OUT, "behaviour", name, "restart_meta.json"))
    return {"ok": d["ok"], "n_gates": d["n_gates"], "n_gates_bitwise": d["n_gates_bitwise"], "max_rel_diff": max(v["max_rel_diff"] for v in d["gates"].values()),
            "n_rows": d["n_rows"], "n_rows_equal": d["n_rows_equal"], "differing": d["differing"], "spot": d["spot"],
            "first_run_started": first["started"], "restart_started": again["started"], "restart_card": d.get("cuda_visible_devices")}


def main():
    ref = C.jload(os.path.join(C.OUT, "referee.json")) if os.path.exists(os.path.join(C.OUT, "referee.json")) else None
    chk = C.jload(os.path.join(C.OUT, "checks.json")) if os.path.exists(os.path.join(C.OUT, "checks.json")) else None
    res = {"what": "E0 sanity comparison before the final experiments (no tuning; numbers reported as they are)",
           "rendered": time.strftime("%Y-%m-%d %H:%M:%S"), "alpha": C.ALPHA, "n_cal": C.N_CAL, "n_draws_gate": C.N_DRAWS,
           "draws_behaviour": list(C.BEHAVIOUR_DRAWS), "methods": C.METHODS, "code": C.code_fingerprint(),
           "definitions": {"gate_stats": C.gate_stats.__doc__, "behaviour_stats": C.behaviour_stats.__doc__, "over_draws": C.over_draws.__doc__,
                           "signs": "normalised coordinates and cosines are signed; no absolute value is taken anywhere"},
           "library_tests": library_tests(), "logs": {}}
    T = ["E0 SANITY COMPARISON BEFORE THE FINAL EXPERIMENTS      (rendered " + res["rendered"] + " by code/e0_render.py from out/*.json; no number is typed by hand)",
         f"alpha = {C.ALPHA}; n = {C.N_CAL} calibration queries = a seeded random draw from the valid REPAIR probe rows (draw d: seed d for the draw and for the two-block split);",
         "teacher-forced keys and forwards at B = 1; every gate decides at the teacher-forced last prompt position (role 'last'); coordinates and cosines are SIGNED.",
         "M1 = default method of PROTOCOL.md (normalised solver coordinate, two-block calibration, top-1, REPL, rebased values, guards on); M2 = M1 with the shared-scale",
         "calibration; M3 / M4 = cosine gate (raw cosine to the stored solver key, top-1, rebased values, guards on) with the two-block / shared-scale calibration;",
         "M5 = cosine gate at 0.65 and M6 = exact-match gate (cos >= 0.999), both role 'last', top-1, rebased values; deployed = W_E; pre-edit = W_0 (control).",
         "Guards = each edit's stored solver key and the B = 1 key of its own Rel prompt must keep firing (capped thresholds; a capped edit is outside the conformal bound).",
         "Only the two development logs were touched.  Nothing was tuned or selected on these numbers.",
         f"Library: rescope, code fingerprint {res['library_tests']['fingerprint_of_the_library_used']} (unchanged); {res['library_tests']['n_with_this_fingerprint']} of its "
         f"{res['library_tests']['n_test_results']} stored test results carry this fingerprint, {res['library_tests']['n_pass']} pass.", ""]
    for name, cfg in C.LOGS.items():
        g, degenerate = gate_results(name)
        b = behaviour_results(name, degenerate)
        r = None if ref is None else ref["logs"][name]
        ex = None if r is None else explain(r)
        ch = None if chk is None else chk["logs"][name]
        rc = restart_check(name)
        res["logs"][name] = {"gate": g, "behaviour": b, "referee": r, "referee_attribution": ex, "checks": ch, "restart_check": rc}
        c = g["counts"]
        T += ["#" * 170, f"{name}  ({cfg['short']}, T = {c['edits']} edits)", "#" * 170,
              f"rows: REPAIR {c['repair']} valid (calibration pool); HELD-OUT {c['heldout']} valid, HELD-OUT-DISJOINT {c['heldout_disjoint']} (of them {c['heldout_disjoint_question_in_repair']} "
              f"ask a question that a REPAIR row asks about another image), HELD-OUT-STRICT {c['heldout_strict']};",
              f"hard negatives (valid rows, first 100 edits) {c['hn']}: {c['hn_flags']['new']} new, source item related to an edit of this log {c['hn_flags']['src_related_to_log']}, "
              f"HN-I rows whose question is also asked by an edit of this log (cross-like) {c['hn_flags']['HN-I_q_edit']}, same category as the edit "
              f"{c['hn_flags']['same_category']}, related_flag {c['hn_flags']['related_flag']};",
              f"sequences whose teacher-forced last prompt position is one token after the decode's: {c['lp_shift_rows']}", ""]
        T += render_gate(g) + [""] + render_behaviour(b, g) + [""] + render_referee(r) + ([] if ex is None else render_explain(ex, r)) + [""] + render_checks(ch)
        if rc is not None:
            T.append(f"    F  restart of the complete GPU run (card {rc['restart_card']}): gates rebuilt bitwise {rc['n_gates_bitwise']}/{rc['n_gates']} (largest relative difference "
                     f"{rc['max_rel_diff']:.1e}); first {rc['spot']} items of every group and method re-evaluated: {rc['n_rows_equal']}/{rc['n_rows']} rows identical to the stored "
                     f"rows -> {'OK' if rc['ok'] else 'DIFFERENCE'}")
        T.append("")
    C.jdump(res, os.path.join(C.OUT, "results.json"), indent=1)
    with open(os.path.join(C.OUT, "table.txt"), "w") as fh:
        fh.write("\n".join(T) + "\n")
    print("\n".join(T))
    S = render_summary(res)
    with open(os.path.join(C.OUT, "summary.txt"), "w") as fh:
        fh.write("\n".join(S) + "\n")
    print("\n" + "\n".join(S))


def render_summary(res):
    """Compact tables of the headline cells (out/summary.txt): the same numbers as out/table.txt, read from the same results."""
    def tab(headers, rows):
        w = [max(len(str(x)) for x in c) for c in zip(*([headers] + rows))]
        return ["  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)).rstrip() for r in [headers] + rows]

    logs = [(n, C.LOGS[n]["short"][0]) for n in C.LOGS]
    S = [f"E0 summary (alpha = {C.ALPHA}, n = {C.N_CAL}; Q = TB-Q100-ae-l2_3, L = TB-L100-ae-l2_3)", "",
         f"GATE LEVEL, {C.N_DRAWS} draws: fire = % of HELD-OUT-DISJOINT rows firing, mean[5th-95th pct]; cov = % of T-Gen / I-Gen whose own edit fires (top-1);",
         "sil = own Rel prompts silenced before>after cap; HN any / HN own = % of HN-I / HN-Q / HN-X rows on which an edit / the row's own edit (HN-X: either of",
         "its two edits) fires (means)"]
    rows = []
    for name, q in logs:
        g = res["logs"][name]["gate"]["methods"]
        for mid in C.METHODS:
            m, t = g[mid], g[mid]["tf"]
            if mid in C.FIXED:
                v = lambda x: x                                                                      # noqa: E731
                fire, sil = f"{t['probe']['heldout_disjoint']['fired']:.2f}", f"{m['own_prompts']['silenced']}"
            else:
                v = lambda x: x["mean"]                                                              # noqa: E731
                d = t["probe"]["heldout_disjoint"]["fired"]
                fire = f"{d['mean']:.2f}[{d['p05']:.2f}-{d['p95']:.2f}]"
                sil = f"{m['own_prompts']['silenced_before_cap']['mean']:.1f}>{m['own_prompts']['silenced_after_cap']['mean']:.0f}"
            rows.append([q, mid, fire, f"{v(t['edit']['tgen']['own_top']):.1f}/{v(t['edit']['igen']['own_top']):.1f}", sil,
                         "/".join(f"{v(t['hn'][h]['fired']):.0f}" for h in C.HN_TYPES),
                         "/".join(f"{v(t['hn'][h]['own_or_b_top' if h == 'HN-X' else 'own_top']):.0f}" for h in C.HN_TYPES)])
    S += tab(["log", "method", "fire", "cov T/I", "sil", "HN any I/Q/X", "HN own I/Q/X"], rows)
    S += ["", f"BEHAVIOUR, draws {list(C.BEHAVIOUR_DRAWS)} (means; M5, M6, deployed: one run): exact % Rel/T-Gen/I-Gen over 100 edits; D1 % on HELD-OUT-DISJOINT / STRICT;",
          "hard-negative D1 % HN-I/HN-Q/HN-X; token-level locality T-Loc*/I-Loc* on HELD-OUT-DISJOINT"]
    rows = []
    for name, q in logs:
        b = res["logs"][name]["behaviour"]
        if b is None:
            continue
        for mid in ("M1", "M2", "M3", "M4", "M5", "M6", "deployed"):
            m = b["methods"][mid]
            s, v = (m["summary"], lambda x: x["mean"]) if "summary" in m else (m["single"], lambda x: x)
            e, p, ps, h = s["edit"], s["probe"]["heldout_disjoint"], s["probe"]["heldout_strict"], s["hn"]
            rows.append([q, mid, "/".join(f"{v(e[k]['exact']):.0f}" for k in ("rel", "tgen", "igen")), f"{v(p['D1']):.2f}/{v(ps['D1']):.2f}",
                         "/".join(f"{v(h[t]['D1']):.1f}" for t in C.HN_TYPES), f"{v(p['T-Loc*']):.2f}/{v(p['I-Loc*']):.2f}"])
    S += tab(["log", "method", "Rel/T/I", "D1 disj/strict", "HN D1 I/Q/X", "T-Loc*/I-Loc*"], rows)
    S += ["", "REFEREE -> M1 / M2 (rows firing % / T-Gen coverage %), differences in points: MC = his 300 halves -> expectation over halves; n200 = calibration on",
          f"all REPAIR rows -> {C.N_CAL} drawn; d20 = our {C.N_DRAWS} draws instead of the expectation; pool = read-out HELD-OUT -> HELD-OUT-DISJOINT; guards; top-1; positions"]
    rows = []
    for name, q in logs:
        ex = res["logs"][name]["referee_attribution"]
        if ex is None:
            continue
        for cal, lab in (("two_block", "two-block"), ("shared", "shared")):
            e = ex[cal]
            st = e["steps"]
            rows.append([q, lab, f"{e['referee']['firing']:.2f}/{e['referee']['cov_tgen']:.1f}", f"{e['this_experiment']['firing']:.2f}/{e['this_experiment']['cov_tgen']:.1f}"]
                        + [f"{x['firing']:+.2f}/{x['cov_tgen']:+.1f}" for x in st] + [f"{e['positions']['firing']:+.2f}/{e['positions']['cov_tgen']:+.1f}"])
    S += tab(["log", "rule", "referee", "this", "MC", "n200", "d20", "pool", "guards", "top-1", "positions"], rows)
    # ---- the numbers behind the remarks of the report
    S += ["", "NOTES (Q | L)"]

    def both(fn):
        return " | ".join(fn(res["logs"][name]) for name, _ in logs)

    def pm(d):
        return f"{d['mean']:+.2f}+-{d['se']:.2f}"
    if all(res["logs"][n]["referee"] is not None for n, _ in logs):
        S.append("two-block minus shared, M1 - M2 as defined, 6000 paired draws (fire, T-Gen cov, I-Gen cov): " + both(
            lambda L: ", ".join(pm(L["referee"]["two_block_minus_shared"]["n200_M1_minus_M2_as_defined"][k]) for k in ("firing", "cov_tgen", "cov_igen"))))
        S.append("  the same in the referee's setting (all REPAIR rows): " + both(
            lambda L: ", ".join(f"{L['referee']['two_block_minus_shared']['all_REPAIR_rows_referee_readout'][k]['mean']:+.2f}" for k in ("firing", "cov_tgen", "cov_igen"))))
        S.append("fire % over 6000 draws, mean (5th-95th pct), M1 then M2: " + both(
            lambda L: "; ".join(f"{L['referee']['m1_m2_many_draws'][c]['firing']['mean']:.2f} ({L['referee']['m1_m2_many_draws'][c]['firing']['p05']:.2f}-"
                                f"{L['referee']['m1_m2_many_draws'][c]['firing']['p95']:.2f})" for c in ("two_block", "shared")))
                 + "; bound 4.95 / 4.98")
        S.append("referee's 300 halves vs expectation over 60000 halves (two-block fire %): " + both(
            lambda L: f"{L['referee']['referee_monte_carlo']['firing']['referee_300_halves']:.2f} vs {L['referee']['referee_monte_carlo']['firing']['expectation']:.2f} "
                      f"(z = {L['referee']['referee_monte_carlo']['firing']['z_of_referee']:+.2f}, {L['referee']['referee_monte_carlo']['firing']['n_blocks_at_least_as_far_from_the_expectation']} "
                      f"of {L['referee']['referee_monte_carlo']['firing']['n_blocks_of_300']} blocks of 300 as far)"))
        S.append("library rule on the referee's own halves, largest difference to his numbers (points): " + both(
            lambda L: f"{L['referee']['library_on_his_halves_max_abs_diff_to_his_numbers']:.0e}") + "; P11 keys bitwise equal to the cache: " + both(
            lambda L: f"{L['referee']['keys']['keys_bitwise_equal']}/{L['referee']['keys']['n_in_cache']}"))

    def caps(L):
        out = []
        for mid in ("M1", "M3"):
            c = L["gate"]["methods"][mid]["calibration"]
            t = f"{mid} {c['n_edits_capped']['mean']:.1f} ({c['n_draws_with_a_capped_edit']}/{C.N_DRAWS} draws)"
            if mid == "M1" and c["capped_edits_repair_max"]:
                t += " [" + "; ".join(f"edit {i}: {v['n_draws_capped']} draws, largest REPAIR coordinate {v['largest_feature_of_a_REPAIR_row']:.3f} ({v['row']}, same question "
                                      f"as the edit: {v['same_question_as_the_edit']})" for i, v in c["capped_edits_repair_max"].items()) + "]"
            out.append(t)
        return ", ".join(out)
    S.append("edits capped per draw, mean: " + both(caps))
    S.append("own Rel prompts not released by the own edit after the cap (M3, mean over draws): " + both(
        lambda L: f"{L['gate']['methods']['M3']['own_prompts']['not_released_after_cap']['mean']:.2f}"))
    S.append("rephrases on which ANOTHER edit fires, M1 then M2 (T-Gen / I-Gen %): " + both(
        lambda L: "; ".join(f"{L['gate']['methods'][m]['tf']['edit']['tgen']['other_top']['mean']:.1f} / {L['gate']['methods'][m]['tf']['edit']['igen']['other_top']['mean']:.1f}"
                            for m in ("M1", "M2"))))
    S.append("HN-I fire % of M1 (20 draws): new same-category / new other-category / existing: " + both(
        lambda L: " / ".join(f"{L['gate']['methods']['M1']['tf']['hn_subset'][k]['fired']['mean']:.1f}"
                             for k in ("HN-I new, same category", "HN-I new, other category", "HN-I existing (P11, other category)"))))
    if all(res["logs"][n]["behaviour"] is not None for n, _ in logs):
        S.append("D1 % among the firing HELD-OUT-DISJOINT rows, M1 - M5 (draws 0-2): " + both(
            lambda L: " ".join(f"{(L['behaviour']['methods'][m]['summary']['probe']['heldout_disjoint']['D1_given_fired']['mean'] if 'summary' in L['behaviour']['methods'][m] else L['behaviour']['methods'][m]['single']['probe']['heldout_disjoint']['D1_given_fired']):.1f}"
                               for m in ("M1", "M2", "M3", "M4", "M5"))))
        S.append("(T-Gen + I-Gen) / 2 exact %, M1 - M5, deployed (draws 0-2): " + both(
            lambda L: " ".join(f"{(L['behaviour']['methods'][m]['summary']['edit']['gen_mean']['exact']['mean'] if 'summary' in L['behaviour']['methods'][m] else L['behaviour']['methods'][m]['single']['edit']['gen_mean']['exact']):.1f}"
                               for m in ("M1", "M2", "M3", "M4", "M5", "deployed"))))
        S.append("T-Gen / I-Gen exact % per draw 0, 1, 2: " + both(
            lambda L: "; ".join(f"{m} " + " ".join(f"{L['behaviour']['methods'][m]['draws'][str(d)]['edit']['tgen']['exact']:.0f}/{L['behaviour']['methods'][m]['draws'][str(d)]['edit']['igen']['exact']:.0f}"
                                                   for d in C.BEHAVIOUR_DRAWS) for m in ("M1", "M2"))))
        S.append("degenerate pre-edit answers (bare None / No etc.) among HN-I / HN-Q / HN-X rows: " + both(
            lambda L: " / ".join(f"{L['behaviour']['methods']['deployed']['single']['hn'][t]['n_y0_degenerate']} of {L['behaviour']['methods']['deployed']['single']['hn'][t]['n']}"
                                 for t in C.HN_TYPES)) + "; rows with a non-finite logit: 0 | 0" if all(
            res["logs"][n]["checks"]["D_behaviour"]["n_nonfinite_rows"] == 0 for n, _ in logs) else "")
    if all(res["logs"][n]["checks"] is not None for n, _ in logs):
        S.append("checks: naive re-implementation, decisions differing: " + both(lambda L: f"{L['checks']['A_naive']['n_differing']} of {L['checks']['A_naive']['n_decisions']}")
                 + "; GPU hook vs CPU cache, flags differing: " + both(lambda L: f"{L['checks']['C_hook']['n_differing']} of {L['checks']['C_hook']['n_flags']}")
                 + "; deployed D1 == stored: " + both(lambda L: f"{L['checks']['D_behaviour']['deployed_D1_equals_stored_D1']['equal']}/{L['checks']['D_behaviour']['deployed_D1_equals_stored_D1']['n']}")
                 + "; restart, rows identical: " + both(lambda L: "-" if L["restart_check"] is None else f"{L['restart_check']['n_rows_equal']}/{L['restart_check']['n_rows']}")
                 + "; all checks OK: " + both(lambda L: str(all(v.get("ok") for v in L["checks"].values()))))
    lt = res["library_tests"]
    S.append(f"library rescope {lt['fingerprint_of_the_library_used']}: {lt['n_with_this_fingerprint']}/{lt['n_test_results']} stored test results carry this fingerprint, {lt['n_pass']} pass")
    return S


if __name__ == "__main__":
    main()
