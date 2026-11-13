#!/usr/bin/env python3
"""Vision-language experiments (E1, E2, prefix logs, E3, E5, E6, E8, E9 and the fix-ups) in the paper: generated table
bodies, derived facts and the list of text macros.

Called by scripts/gen_numbers.py as a DERIVED source (manifest["derived"], function build) at every `make numbers`,
or directly:   python3 scripts/gen_vlm.py            (prints the facts)
               python3 scripts/gen_vlm.py --sync     rewrites the entries of group "vlm" in numbers_manifest.json from
                                                     MACROS below, keeping only macros that a .tex file uses

Reads (ROOT = project_3; F = 02_research/experiments/10_final; nothing is typed here)
  paper_numbers.json (+ fixup/paper_numbers_fixup.json) of e1_main, e2_curves, e2_prefix, e3_qwen, e3_llava,
  e5_rescope, e6_ablations, e8_cost, e9_maintenance, x_release (tags of PN_FILES below)
  F/e2_curves/out/results.json (records; orders, round-off floor, degenerate near-scope answers, complaint counts)
  F/e2_prefix/out/h1_prefix.json (FINAL prefix part of H1), F/e2_curves/out/fixup/h1_registered.json (development logs)
  F/e5_rescope/out/capacity_curve.json, validity_distribution.json; F/e1_main/out/best_cosine_gate.json, decision.json
  F/x_release/out/logs_b1.json (deployed logs as evaluated here), holm.json
Writes (paper directory)
  tab_vlm_main.tex     body of Table tab:main (deployment classes on the two development logs)
  tab_vlm_scale.tex    part (a) of Table tab:scale (unseen logs)
  tab_vlm_maint.tex    parts (b), (c) of Table tab:maint (revocation; cost)
  tab_vlm_tr_*.tex     tables of the technical report, Appendix C (unseen logs in full, replication read-outs,
                       ablations, near-scope cells per draw, McNemar tests)
  scripts/data/vlm_facts.json   derived facts (each with source and rule) and copies of logs_b1.json for Table tab:logs

Rules applied here (PROTOCOL.md, amendments of 2026-11-12; the verifications' lists)
  * H1: development logs from h1_registered.json, prefix logs from the FINAL h1_prefix.json (never the provisional
    prefix part of the fix-up record);
  * near-scope cells: the three behavioural draws are printed (d0/d1/d2); ReScope-W with two-block thresholds: its
    near-scope cells are the 20-draw gate-level record of E2 (behaviour exists at draw 0 only);
  * rows of methods that read only solver keys or requests are equal in both editor families at equal T: a cell of the
    second family that EQUALS (computed) the first family's is printed as "=";
  * Holm-adjusted values at the resolution of the bootstrap are upper bounds.
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import math
import os
import re
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
F = "02_research/experiments/10_final"
OUT_REL = "01_paper/v4_rescope/scripts/data/vlm_facts.json"
PN_FILES = {
    "e1": f"{F}/e1_main/out/paper_numbers.json", "e1f": f"{F}/e1_main/out/fixup/paper_numbers_fixup.json",
    "e2": f"{F}/e2_curves/out/paper_numbers.json", "e2f": f"{F}/e2_curves/out/fixup/paper_numbers_fixup.json",
    "e2p": f"{F}/e2_prefix/out/paper_numbers.json",
    "e3q": f"{F}/e3_qwen/out/paper_numbers.json", "e3qf": f"{F}/e3_qwen/out/fixup/paper_numbers_fixup.json",
    "e3l": f"{F}/e3_llava/out/paper_numbers.json", "e3lf": f"{F}/e3_llava/out/fixup/paper_numbers_fixup.json",
    "e5": f"{F}/e5_rescope/out/paper_numbers.json", "e6": f"{F}/e6_ablations/out/paper_numbers.json",
    "e8": f"{F}/e8_cost/out/paper_numbers.json", "e9": f"{F}/e9_maintenance/out/paper_numbers.json",
    "xr": f"{F}/x_release/out/paper_numbers_fixup.json",
}
OTHER = {
    "e2res": f"{F}/e2_curves/out/results.json", "h1p": f"{F}/e2_prefix/out/h1_prefix.json",
    "h1r": f"{F}/e2_curves/out/fixup/h1_registered.json", "cap": f"{F}/e5_rescope/out/capacity_curve.json",
    "vdist": f"{F}/e5_rescope/out/validity_distribution.json", "best": f"{F}/e1_main/out/best_cosine_gate.json",
    "dec": f"{F}/e1_main/out/decision.json", "logs": f"{F}/x_release/out/logs_b1.json",
    "holm": f"{F}/x_release/out/holm.json",
    "e8res": f"{F}/e8_cost/out/results.json",       # final revision (T3): the rows of mode where_logonly (ReScope-W, log-only thresholds) are in the records only
}
TEXFILES = ("tab_vlm_main.tex", "tab_vlm_scale.tex", "tab_vlm_maint.tex", "tab_vlm_tr_unseen_a.tex", "tab_vlm_tr_unseen_m.tex",
            "tab_vlm_tr_repl.tex", "tab_vlm_tr_ablation.tex", "tab_vlm_tr_draws.tex", "tab_vlm_tr_mcnemar.tex", "tab_vlm_tr_main.tex", "tab_vlm_tr_prefix.tex",
            "tab_vlm_tr_cost.tex", "tab_vlm_tr_width.tex")
DEV = {"Q": "TB-Q100-ae-l2_3", "L": "TB-L100-ae-l2_3"}
# unseen logs: (paper tag of Table tab:logs, testbed name, tag in e3 names, tag in x_release names, T)
UNSEEN = {
    "Q": [("QAcc", "TB-Q200-ae-l2_3", "AeCC", 200), ("QAd", "TB-Q500-ae-l2_3", "AeD", 500),
          ("QWc", "TB-Q100-memitw", "MeC", 100), ("QWd", "TB-Q500-memitw", "MeD", 500),
          ("QWm", "TB-Q1000-memitw", "MeM", 1000), ("QWmm", "TB-Q2000-memitw", "MeMM", 2000)],
    "L": [("LAcc", "TB-L200-ae-l2_3", "AeTwoHundred", 200), ("LAd", "TB-L500-ae-l2_3", "AeFiveHundred", 500),
          ("LWc", "TB-L100-memitw", "MeHundred", 100), ("LWd", "TB-L500-memitw", "MeFiveHundred", 500),
          ("LWm", "TB-L1000-memitw", "MeThousand", 1000)],
}
DEVTAG = {"Q": "AeC", "L": "AeHundred"}


def find_root():
    env = os.environ.get("PROJECT3_ROOT")
    if env:
        return env
    return _RP.ROOT          # the working-tree view of this repository (rescope_paths.py)


def fx(v, n=1):
    """Fixed-point text, rounded on the decimal value, half away from zero (rule of gen_numbers.fmt_number)."""
    if v is None:
        return "--"
    with localcontext() as ctx:
        ctx.prec = 400
        d = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-(n + 9)), rounding=ROUND_HALF_EVEN)
        s = format(d.quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP), "f")
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def sg(v, n=1):
    """signed number in math mode"""
    s = fx(v, n)
    return "$-$" + s[1:] if s.startswith("-") else ("+" + s if float(s) > 0 else s)


class Data:
    def __init__(self, root):
        self.root = root
        self.pn = {t: json.load(open(os.path.join(root, p))) for t, p in PN_FILES.items()}
        self.o = {t: json.load(open(os.path.join(root, p))) for t, p in OTHER.items()}
        self.rec = {}
        for r in self.o["e2res"]:
            self.rec.setdefault((r["log"], r["part"], r["method"], r["variant"], str(r["n_label"]), r["metric"]), []).append(r)
        self.used = []

    def p(self, tag, key, default=KeyError):
        d = self.pn[tag]
        if key not in d:
            if default is KeyError:
                raise KeyError(f"{tag}: {key}")
            return default
        return d[key]


def e3(D, b, tagname, name, default=KeyError):
    """paper number of E3 for backbone b: the two experiments use different names for the same quantity."""
    return D.p("e3q" if b == "Q" else "e3l", f"Ethree{b}{tagname}{name}", default)


def e3row(D, b, lt, method):
    """One (log, method) cell set of E3 under common names (percent). Qwen and LLaVA files use different key names.
    LATER EDITS (PROTOCOL.md, R8): stream positions from 500 on (Qwen) and from 100 on (LLaVA), the edits that no design
    decision saw; only the experiments' own values are used (Qwen: suffix ExactLater, which exists on the logs with 1000
    and 2000 edits; LLaVA: suffix Later, T > 100); a log without later edits has None, and GenLater then falls back to
    all edits (the captions say so).
    NEAR: release on near-scope negatives at gate level (20 draws), ALL KINDS POOLED: Qwen stores the pooled rate;
    LLaVA stores one rate per kind, pooled here with the numbers of valid rows of each kind (x_release)."""
    tag, T = lt[2], lt[3]
    out = {}
    if b == "Q":
        m = {"deployed": "Deployed", "where": "Where", "rescope": "Rescope", "white": "WhiteTenthShared", "retr": "RetrSum",
             "rescope_x": "RescopeX", "cosine": "Cosine", "lookup": "Lookup", "whiteh3": "WhiteOneShared"}[method]
        g = lambda n, dflt=None: e3(D, b, tag, m + n, dflt)                                        # noqa: E731
        out.update(D1=g("HdDone"), Rel=g("RelExact"), TGen=g("TGenExact"), IGen=g("IGenExact"), Fire=g("HdFired"),
                   GateFire=g("GateHdFired"), Near=g("GateHnFired"), HnD1=g("HnDone"), Capped=g("GateCapped"),
                   MmeT=g("GateMmeT"), NearRel=g("HdNearFired"))
        for k in ("Rel", "TGen", "IGen"):
            out[k + "Later"] = g(k + "ExactLater")
            out[k + "First"] = g(k + "ExactFirst")
    else:
        m = {"deployed": "Deployed", "where": "WhereLogonly", "rescope": "Rescope", "white": "WhiteZeroOneShared",
             "retr": "RetrSum", "rescope_x": "RescopeX", "cosine": "Cosine", "lookup": "Lookup", "whiteh3": "WhiteOneShared"}[method]
        g = lambda n, dflt=None: e3(D, b, tag, m + n, dflt)                                        # noqa: E731
        hn = [g("HnFireI"), g("HnFireQ"), g("HnFireX")]
        nk = [D.p("xr", f"Xrel{b}{tag}RescopeHn{k}Rows") for k in "IQX"]                           # valid rows per kind (the same for every method)
        out.update(D1=g("DOne"), Rel=g("Rel"), TGen=g("TGen"), IGen=g("IGen"), Fire=g("Fire"), GateFire=g("Fire"),
                   Near=None if None in hn else sum(h * n for h, n in zip(hn, nk)) / sum(nk), HnD1=g("HnDOne"), Capped=g("Capped"), MmeT=g("MmeT"),
                   NearRel=g("FireNear"))
        for k in ("Rel", "TGen", "IGen"):
            out[k + "Later"] = g(k + "Later") if T > 100 else None
    for sfx in ("", "Later"):
        t, i = out.get("TGen" + sfx), out.get("IGen" + sfx)
        out["Gen" + sfx] = None if t is None or i is None else (t + i) / 2
    out["HasLater"] = out.get("GenLater") is not None
    if out.get("GenLater") is None:
        out["GenLater"] = out["Gen"]                                          # no later edits in this log: all edits
    return out


def xr(D, b, tag, name, default=KeyError):
    return D.p("xr", f"Xrel{b}{tag}{name}", default)


def rec1(D, log, part, method, variant, n, metric):
    rs = D.rec.get((log, part, method, variant, str(n), metric), [])
    return rs


# ------------------------------------------------------------------------------------------------ facts
def facts(D):
    Fc = {}

    def put(name, value, source, rule):
        Fc[name] = {"value": value, "source": source, "rule": rule}

    # ---- H1: development logs (h1_registered.json) and prefix logs (FINAL h1_prefix.json)
    h1r, h1p = D.o["h1r"], D.o["h1p"]
    cells = {k: c for k, c in h1p["cells"].items() if c["n"] == 10}
    assert len(cells) == 6 and all(c["status"] == "complete" for c in cells.values())
    two = [c["log"] for c in cells.values() if c["verdict"]["rel_within_2_points"] and c["verdict"]["D1_at_most_10"]]
    one = [c["log"] for c in cells.values() if c["verdict"]["rel_not_more_than_2_below_deployed"] and c["verdict"]["D1_at_most_10"]]
    src = OTHER["h1p"]
    put("h1_prefix_n", len(cells), src, "prefix logs with a complete cell at n = 10")
    put("h1_prefix_two_sided", len(two), src, "prefix logs on which D1 <= 10 and |Rel - Rel deployed| <= 2: " + ", ".join(two))
    put("h1_prefix_one_sided", len(one), src, "prefix logs on which D1 <= 10 and Rel >= Rel deployed - 2: " + ", ".join(one))
    dev = h1r["development_logs"]
    put("h1_dev_two_sided", sum(1 for v in dev.values() if v["two_sided_reading"]["met_on_this_log"]), OTHER["h1r"],
        "development logs on which the two-sided reading is met (D1 <= 10 and |Rel - Rel deployed| <= 2)")
    put("h1_dev_one_sided", sum(1 for v in dev.values() if v["one_sided_reading"]["met_on_this_log"]), OTHER["h1r"],
        "development logs on which the one-sided reading is met (D1 <= 10 and Rel >= Rel deployed - 2)")
    deltas, d1s = [], []
    for b, lg in DEV.items():
        deltas.append(D.p("e2", f"Etwo{b}HoneRelMinusDeployed"))
        d1s.append(D.p("e2", f"Etwo{b}HoneWhereDone"))
    for c in cells.values():
        deltas.append(c["verdict"]["rel_signed_diff_to_deployed"])
        d1s.append(c["where_least"]["hd_D1"]["mean"])
    put("h1_rel_delta_min", min(deltas), "e2_curves paper numbers + " + src, "smallest Rel exact of the repair minus the deployed log over the 8 logs (points)")
    put("h1_rel_delta_max", max(deltas), "e2_curves paper numbers + " + src, "largest ... over the 8 logs")
    put("h1_d1_max", max(d1s), "e2_curves paper numbers + " + src, "largest held-out D1 of the repair after 10 complaints over the 8 logs")
    put("h1_n_logs", len(deltas), src, "development logs + prefix logs")
    # final revision (T30): H1 is reported as registered, without a second reading: on how many of the 8 logs Rel of the repair is
    # more than 2 points below / above the deployed log (the rule's 2 points: PROTOCOL.md, H1), and which logs these are
    names = [lg for lg in DEV.values()] + [c["log"] for c in cells.values()]
    put("h1_n_rel_below", sum(1 for d in deltas if d < -2), "e2_curves paper numbers + " + src, "logs (of 8) with Rel of the repair more than 2 points BELOW the deployed log: " + ", ".join(n_ for n_, d in zip(names, deltas) if d < -2))
    put("h1_n_rel_above", sum(1 for d in deltas if d > 2), "e2_curves paper numbers + " + src, "logs (of 8) with Rel of the repair more than 2 points ABOVE the deployed log: " + ", ".join(n_ for n_, d in zip(names, deltas) if d > 2))
    put("h1_rel_below_logs", [n_ for n_, d in zip(names, deltas) if d < -2], src, "the logs of h1_n_rel_below")
    put("h1_rel_above_logs", [n_ for n_, d in zip(names, deltas) if d > 2], src, "the logs of h1_n_rel_above")
    for b, lg in DEV.items():
        for T in (10, 20, 50):
            c = cells[f"{lg}-p{T}|n10"]
            pre = f"pfx_{b}{T}_"
            put(pre + "dep_d1", c["deployed"]["hd_D1"], src, "deployed log, held-out D1")
            put(pre + "dep_rel", c["deployed"]["rel"], src, "deployed log, Rel exact")
            put(pre + "w_d1", c["where_least"]["hd_D1"]["mean"], src, "ReScope-W after 10 complaints, held-out D1 (mean over streams)")
            put(pre + "w_rel", c["where_least"]["rel"]["mean"], src, "ReScope-W after 10 complaints, Rel exact")
            put(pre + "rel_delta", c["verdict"]["rel_signed_diff_to_deployed"], src, "Rel exact of the repair minus the deployed log")
            put(pre + "comp_d1", c["linear"]["comp_b"]["hd_D1"], src, "compensation (answer positions) after 10 complaints, held-out D1")
            put(pre + "comp_rel", c["linear"]["comp_b"]["rel"], src, "the same, Rel exact")
            ba = c["best_any"]
            assert ba["method"].startswith("rb_"), ba["method"]
            g1 = ba["hd_D1"]["mean"] if isinstance(ba["hd_D1"], dict) else ba["hd_D1"]
            g2 = ba["rel"]["mean"] if isinstance(ba["rel"], dict) else ba["rel"]
            put(pre + "rb_d1", g1, src, f"the linear repair with the lowest held-out D1 after 10 complaints (a rollback: {ba['method']}), held-out D1")
            put(pre + "rb_rel", g2, src, "the same repair, Rel exact")
    for key, fn in (("dep_d1", None), ("w_d1", None), ("comp_d1", None)):
        vals = [Fc[f"pfx_{b}{T}_{key}"]["value"] for b in DEV for T in (10, 20, 50)]
        put(f"pfx_{key}_min", min(vals), src, "minimum over the six prefix logs")
        put(f"pfx_{key}_max", max(vals), src, "maximum over the six prefix logs")
    put("h1_comp_d1_min", min([Fc[f"pfx_{b}{T}_comp_d1"]["value"] for b in DEV for T in (10, 20, 50)] + [D.p("e2", f"Etwo{b}LinCompBTenDone") for b in DEV]),
        "e2_curves paper numbers + " + src, "compensation (answer positions) after 10 complaints: smallest held-out D1 over the 8 logs")
    small = [(Fc[f"pfx_{b}{T}_rb_d1"]["value"], Fc[f"pfx_{b}{T}_rb_rel"]["value"]) for b in DEV for T in (10, 20)]
    put("pfx_rb_small_d1_max", max(s[0] for s in small), src, "rollback with the lowest D1 on the 10- and 20-edit prefixes: largest held-out D1")
    put("pfx_rb_small_rel_max", max(s[1] for s in small), src, "the same rollbacks: largest Rel exact")
    cmp_rel = [abs(Fc[f"pfx_{b}{T}_comp_rel"]["value"] - Fc[f"pfx_{b}{T}_dep_rel"]["value"]) for b in DEV for T in (10, 20, 50)]
    put("pfx_comp_rel_absdiff_max", max(cmp_rel), src, "compensation: largest |Rel - Rel deployed| over the six prefix logs")
    put("h1_holm_bound", h1r["holm"]["smallest_possible_adjusted_p"], OTHER["h1r"], "smallest Holm-adjusted value that can be shown (4 tests x resolution)")

    # ---- E2 records: orders, refusals, own requests out of scope, complaint counts, round-off floor, degenerate y0
    n_cells = n_ok = refused = own_out = 0
    for (log, part, method, variant, n, metric), rs in D.rec.items():
        if part == "b" and method == "where_least" and metric == "order.n_orders":
            for r in rs:
                eq = [x for x in D.rec[(log, part, method, variant, n, "order.n_equal_thresholds")] if x["draw"] == r["draw"]]
                n_cells += 1
                n_ok += int(eq and eq[0]["value"] == r["value"] == 20)
        if part == "b" and method == "where_least" and metric == "meta.n_refused":
            refused += sum(r["value"] for r in rs)
        if part == "b" and method == "where_least" and metric == "edit.rel.n_own_out_of_scope":
            own_out += sum(r["value"] for r in rs)
    put("least_cells", n_cells, OTHER["e2res"], "cells (log, start, n, complaint stream) of the least repair with an order check")
    put("least_cells_all_orders", n_ok, OTHER["e2res"], "of these, cells whose thresholds are bitwise equal under all 20 arrival orders")
    put("least_orders", 20, OTHER["e2res"], "arrival orders per cell (order.n_orders, equal in every cell: asserted)")
    put("least_refused", refused, OTHER["e2res"], "reported complaints refused, summed over the cells")
    put("least_own_out", own_out, OTHER["e2res"], "own requests out of scope after the repair, summed over the cells")
    for b, lg in DEV.items():
        ns = {r["n"] for (log, part, method, variant, n, metric), rs in D.rec.items()
              if log == lg and part == "a" and n == "ALL" and method == "COMP_B" and metric == "probe.d1@mean" for r in rs}
        if not ns:
            ns = {r["n"] for (log, part, method, variant, n, metric), rs in D.rec.items()
                  if log == lg and part == "b" and n == "ALL" for r in rs}
        assert len(ns) == 1, ns
        put(f"n_all_{b}", int(ns.pop()), OTHER["e2res"], "number of reported complaints at n = ALL")
        hn_n = hn_d = 0
        for kind in ("HN-I", "HN-Q", "HN-X"):
            hn_n += rec1(D, lg, "base", "deployed", "", 0, f"hn.{kind}.n")[0]["value"]
            hn_d += rec1(D, lg, "base", "deployed", "", 0, f"hn.{kind}.n_y0_degenerate")[0]["value"]
        put(f"hn_degenerate_pct_{b}", 100.0 * hn_d / hn_n, OTHER["e2res"],
            f"near-scope negatives with a degenerate pre-edit answer: {int(hn_d)} of {int(hn_n)}")

    # ---- capacity curve (E5 f)
    for b, lg in DEV.items():
        o = [D.p("e5", f"capacity.{lg}.largest_n_all_edits_preserved.order{i}") for i in range(3)]
        pr = [D.p("e5", f"capacity.{lg}.predicted_largest_n_all_edits.order{i}") for i in range(3)]
        put(f"cap_meas_{b}", (f"{min(o)}--{max(o)}" if min(o) != max(o) else str(o[0])), PN_FILES["e5"],
            "largest n of the grid with all T edits preservable, range over the three complaint orders")
        put(f"cap_meas_list_{b}", ", ".join(str(x) for x in o) if len(set(o)) > 1 else str(o[0]), PN_FILES["e5"], "the same, per order")
        put(f"cap_pred_list_{b}", ", ".join(str(x) for x in pr) if len(set(pr)) > 1 else str(pr[0]), PN_FILES["e5"], "registered prediction per order")

    capj = json.load(open(os.path.join(D.root, F, "e5_rescope/out/capacity_curve.json")))
    sil = 0
    for b, lg in DEV.items():
        L = capj[lg]
        fr = L["smallest_grid_n_with_rank_KQ_equal_d_by_order"]
        put(f"cap_fullrank_list_{b}", ", ".join(str(x) for x in fr) if len(set(fr)) > 1 else str(fr[0]), F + "/e5_rescope/out/capacity_curve.json",
            "first n of the grid at which the keys of the complaints span the key space (only the pre-edit weight restores them), per order")
        put(f"cap_grid_{b}", ", ".join(str(n) for n in L["grid"]), F + "/e5_rescope/out/capacity_curve.json", "grid of the numbers of complaints")
        sil = max([sil] + [L["grid"][n][o][k] for n in L["grid"] for o in L["grid"][n]
                           for k in ("thresholds_sacrificing_silenced_own_request", "thresholds_sacrificing_silenced_solver_key", "thresholds_protecting_refused_complaints")])
        put(f"cap_tol_rank_{b}", L["tolerance_rank"], F + "/e5_rescope/out/capacity_curve.json", "relative tolerance of the numerical rank")
        put(f"cap_tol_rank_tex_{b}", "10^{%d}" % round(math.log10(L["tolerance_rank"])), F + "/e5_rescope/out/capacity_curve.json", "the same, as a power of ten (math mode)")
        put(f"cap_stable_{b}", int(all(L["rank_stable_over_two_decades_by_order"])), F + "/e5_rescope/out/capacity_curve.json", "1 if the rank is the same over two decades of the tolerance, every order")
        # final revision (T31, T39 f): like is compared with like. The experiment's own record of the registered predictions
        # (capacity_curve.json, <log>.prediction.by_order.<k>): the prediction per complaint, the grid value that it implies
        # (rounded down to the grid for "all edits preservable up to n"; the first grid point at or above it for "only W0"),
        # the measured grid value, the value from the rank after every complaint, and the experiment's verdict per order.
        cj = F + "/e5_rescope/out/capacity_curve.json"
        po = [L["prediction"]["by_order"][str(i)] for i in range(3)]
        lst = lambda xs: ", ".join(str(x) for x in xs) if len(set(xs)) > 1 else str(xs[0])            # noqa: E731
        assert [q["measured_largest_grid_n_all_edits"] for q in po] == L["largest_n_all_edits_preserved_by_order"]
        assert [q["measured_first_grid_n_rank_d"] for q in po] == fr
        put(f"cap_pred_grid_list_{b}", lst([q["grid_value_the_prediction_implies"] for q in po]), cj, "registered prediction 'all edits preservable up to n', rounded down to the grid, per order (prediction.by_order.<k>.grid_value_the_prediction_implies)")
        put(f"cap_hit_all_{b}", sum(1 for q in po if q["hit_all_edits"]), cj, "orders (of 3) in which that grid value equals the measured one (hit_all_edits)")
        put(f"cap_every_list_{b}", lst([q["every_complaint_largest_n_all_edits"] for q in po]), cj, "largest number of complaints with rank K_Q <= d - T, from the rank after every complaint, per order")
        put(f"cap_w0_pred_list_{b}", lst([q["predicted_first_n_only_W0"] for q in po]), cj, "second registered prediction of R20: the number of complaints from which on only the pre-edit weight restores them, per order (predicted_first_n_only_W0)")
        put(f"cap_w0_pred_grid_list_{b}", lst([q["grid_value_the_prediction_implies_for_W0"] for q in po]), cj, "the grid value that this prediction implies, per order")
        put(f"cap_w0_every_list_{b}", lst([q["every_complaint_first_n_rank_d"] for q in po]), cj, "first number of complaints at which the keys of the complaints span the key space, from the rank after every complaint, per order")
        put(f"cap_w0_miss_{b}", sum(1 for q in po if not q["hit_only_W0"]), cj, "orders (of 3) in which the grid value of the second prediction differs from the measured one")
        put(f"cap_w0_hit_{b}", sum(1 for q in po if q["hit_only_W0"]), cj, "orders (of 3) in which the grid value of the second prediction equals the measured one (hit_only_W0)")
        put(f"cap_formula_holds_{b}", int(bool(L["cap_equals_min_T_d_minus_rank_at_every_grid_point"])), cj, "1 if at every grid point the number of edits that the minimum-norm restoring weight preserves equals the rank formula (cap_equals_min_T_d_minus_rank_at_every_grid_point)")
        gp = sorted(int(n) for n in L["grid"])
        lo_, hi_ = min(q["grid_value_the_prediction_implies"] for q in po), max(fr)
        steps = {b_ - a_ for a_, b_ in zip(gp, gp[1:]) if lo_ <= a_ and b_ <= hi_}
        assert len(steps) == 1, steps
        put(f"cap_grid_step_{b}", steps.pop(), cj, f"step of the grid between {lo_} and {hi_} complaints (every predicted and measured value lies there)")
    put("cap_thr_silenced_max", sil, F + "/e5_rescope/out/capacity_curve.json",
        "thresholds-only repair of the same complaints from the log-only thresholds: largest number of edits silenced (own request or solver key) or complaints refused, over both logs, the grid and the orders")
    comp_all = [D.p("e2", f"Etwo{b}Lin{m}TenDone") for b in DEV for m in ("CompB", "CompBp", "CompBAllpos")]
    for c in cells.values():
        comp_all += [c["linear"]["comp_b"]["hd_D1"], c["linear"]["comp_bp"]["hd_D1"]]
    put("h1_comp_any_d1_min", min(comp_all), "e2_curves paper numbers + " + src, "every compensation variant we run (answer positions with and without protection; all positions) after 10 complaints: smallest held-out D1 over the 8 logs")
    # W4 (2026-11-12): the range that the abstract, the introduction and the conclusion quote for "the rollback and compensation
    # repairs we run" after 10 complaints on the two development logs (the repairs of a frozen log that Sec. 5.2 names; the
    # re-run of the editor is no repair of a frozen log and is not in the range).
    lin10 = {(b, m): D.p("e2", f"Etwo{b}Lin{m}TenDone") for b in DEV for m in ("CompB", "CompBp", "CompBAllpos", "RbUnion", "RbTot")}
    put("h1_lin_dev_d1_min", min(lin10.values()), PN_FILES["e2"], "rollback (union, total) and compensation (answer positions with and without protection; all positions) after 10 complaints on the two development logs: smallest held-out D1; attained by %s on %s" % min((v, m, b) for (b, m), v in lin10.items())[1:])
    put("h1_lin_dev_d1_max", max(lin10.values()), PN_FILES["e2"], "the same, largest held-out D1; attained by %s on %s" % max((v, m, b) for (b, m), v in lin10.items())[1:])
    # ---- E1: H2 other baselines; near-scope spread; McNemar; selection
    others = []
    for b in DEV:
        for base in ("Ball", "Cosine", "CosineShared", "Dual", "Lookup", "Router"):
            others.append(D.p("e1", f"Eone{b}Htwo{base}Diff"))
    put("h2_other_min", min(others), PN_FILES["e1"], "H2, ReScope minus the registered cosine gate (both calibrations), ball, router, lookup, answer match: smallest difference over both logs")
    put("h2_other_max", max(others), PN_FILES["e1"], "the same, largest difference")
    rows_main = ["Rescope", "RescopeX", "Cosine", "BestCos", "BestCosHtwo", "Retr"]
    sp = []
    for b in DEV:
        for r in rows_main:
            for k in "IQX":
                sp.append((D.p("e1f", f"EoneFx{b}{r}HN{k}FireSpread"), f"{b} {r} HN-{k}"))
    put("near_spread_max", max(sp)[0], PN_FILES["e1f"], f"largest spread (max - min over the three behavioural draws) of a near-scope release cell of the calibrated rows of Table tab:main; attained by {max(sp)[1]}")
    mc = [D.p("e1f", f"EoneFx{b}DeployedMcDOneD{d}P") for b in DEV for d in ("Zero", "One", "Two")]
    put("mc_dep_exp", int(math.ceil(math.log10(max(mc)))), PN_FILES["e1f"], "ceil(log10) of the largest exact McNemar p of D1, ReScope against the deployed log, over both logs and the three draws")
    cr = [xr(D, b, DEVTAG[b], f"RescopeH{pool}ChangedReleased") for b in DEV for pool in ("d", "nI", "nQ", "nX")]
    put("sys_changed_released_min", min(cr), PN_FILES["xr"], "ReScope on the development logs: smallest share of released rows whose answer changes, over heldout_disjoint and the three kinds of near-scope negatives, both logs")
    put("sys_changed_released_hd_min", min(xr(D, b, DEVTAG[b], "RescopeHdChangedReleased") for b in DEV), PN_FILES["xr"], "the same on heldout_disjoint only")
    best = D.o["best"]
    put("sel_cos_family_n", _count_variants(best, "cosine"), OTHER["best"], "variants of the cosine family among which the reference gate was selected")
    put("sel_retr_family_n", _count_variants(best, "retr"), OTHER["best"], "retrieval rules among which the reference was selected")

    # ---- E3: ranges for the text
    def rng(name, vals, source, rule):
        vals = [v for v in vals if v is not None]
        put(name + "_min", min(vals), source, rule + " (minimum)")
        put(name + "_max", max(vals), source, rule + " (maximum)")

    e3src = "paper numbers of e3_qwen / e3_llava (+ fixup)"
    for b in DEV:
        fam1 = [lt for lt in UNSEEN[b] if lt[1].endswith("ae-l2_3")]
        fam2 = [lt for lt in UNSEEN[b] if lt[1].endswith("memitw")]
        for fn, logs in (("a", fam1), ("m", fam2)):
            rng(f"e3_{b}{fn}_dep_d1", [e3row(D, b, lt, "deployed")["D1"] for lt in logs], e3src, "deployed log, held-out D1 over the unseen logs of the family")
            rng(f"e3_{b}{fn}_w_d1", [e3row(D, b, lt, "where")["D1"] for lt in logs], e3src, "ReScope-W log-only, held-out D1")
            rng(f"e3_{b}{fn}_s_d1", [e3row(D, b, lt, "rescope")["D1"] for lt in logs], e3src, "ReScope, held-out D1")
            rng(f"e3_{b}{fn}_s_rel", [e3row(D, b, lt, "rescope")["Rel"] for lt in logs], e3src, "ReScope, Rel exact")
            rng(f"e3_{b}{fn}_s_cap", [fxp(D, b, lt, "RescopeCappedMean") for lt in logs], e3src, "ReScope, capped edits (mean over the 20 draws)")
            rng(f"e3_{b}{fn}_h2", [fxp(D, b, lt, "HtwoCosAllDiff") for lt in logs], e3src, "H2 read-out, ReScope minus the gate selected for H2")
            rng(f"e3_{b}{fn}_h3p1", [fxp(D, b, lt, "HthreePairOneDiff") for lt in logs], e3src, "H3 pair 1")
            rng(f"e3_{b}{fn}_h3p2", [fxp(D, b, lt, "HthreePairTwoDiff") for lt in logs], e3src, "H3 pair 2")
            rng(f"e3_{b}{fn}_errlog", [D.p("e5", f"logs.{lt[1]}.rebase.err_logged_max", None) for lt in logs] +
                ([D.p("e5", f"logs.{DEV[b]}.rebase.err_logged_max")] if fn == "a" else []), PN_FILES["e5"], "largest relative error of a logged value against its target (x 100 = percent)")
        allu = UNSEEN[b]
        rng(f"e3_{b}_retr_h3", [fxp(D, b, lt, "HthreeRetrDiff") for lt in allu], e3src, "H3 against the retrieval reference")
        rng(f"e3_{b}_draws_above", [fxp(D, b, lt, "RescopeDrawsAbove") for lt in allu], e3src, "calibration draws (of 20) whose realised release rate exceeds the level")
        devlt = ("dev", DEV[b], DEVTAG[b], 100)
        rng(f"e3_{b}_near_rel", [fxp(D, b, lt, "RescopeNearRelBeh") for lt in [devlt] + allu], e3src, "ReScope: release on held-out probes that ask the question of an edit (behaviour; development log and unseen logs)")
        # figure revision of 2026-11-12 (DECISIONS_FIG_REVISION.txt, G6; FINAL_CHECK_2 S6): Sec. 5.4 is about the UNSEEN logs; the
        # largest value of the range above is the development log of Qwen, so the range over the unseen logs is a fact of its own
        rng(f"e3_{b}_near_rel_unseen", [fxp(D, b, lt, "RescopeNearRelBeh") for lt in allu], e3src, "ReScope: release on held-out probes that ask the question of an edit (behaviour; UNSEEN logs only)")
        put(f"e3_{b}_near_rel_dev", fxp(D, b, devlt, "RescopeNearRelBeh"), e3src, "the same on the development log")
        # lists along T (text facts "a/b/c"): first family = development log + unseen logs; second family
        for fn, logs in (("a", [devlt] + fam1), ("m", fam2)):
            for meth in ("deployed", "where", "rescope", "white", "retr"):
                rows = [e3row(D, b, lt, meth) for lt in logs]
                for met, nd in (("D1", 1), ("Rel", 0), ("TGen", 0), ("IGen", 0)):
                    if all(r[met] is not None for r in rows):
                        put(f"e3l_{b}{fn}_{meth}_{met}", "/".join(fx(r[met], nd) for r in rows), e3src, "values along T: " + ", ".join(lt[1] for lt in logs))
            put(f"e3l_{b}{fn}_h2", ", ".join(fx(fxp(D, b, lt, "HtwoCosAllDiff")) for lt in logs if lt[0] != "dev"), e3src, "H2 read-out along T (unseen logs); printed in math mode")
            put(f"e3l_{b}{fn}_cap", "/".join(fx(fxp(D, b, lt, "RescopeCappedMean"), 0) for lt in logs), e3src, "ReScope: capped edits (mean of 20 draws) along T")
            put(f"e3l_{b}{fn}_T", "/".join(str(lt[3]) for lt in logs), e3src, "T of the logs of the family")
        # retention of the first 100 edits (E9 c, read from E3)
        rs = json.load(open(os.path.join(D.root, F, "e3_qwen" if b == "Q" else "e3_llava", "out", "retention_series.json")))["families"]
        for fam_key, fn in (("ae-l2_3", "a"), ("memitw", "m")):
            for meth, short in (("deployed", "dep"), ("where_logonly", "w"), ("rescope", "s")):
                x = rs[fam_key]["methods"][meth]["rel_exact"] if "methods" in rs[fam_key] else [r["rel"] for r in rs[fam_key][meth]]
                put(f"ret_{b}{fn}_{short}", "/".join(fx(v_, 0) for v_ in x), F + "/e3_*/out/retention_series.json", "Rel exact of the first 100 edits of the stream along T")
                put(f"ret_{b}{fn}_{short}_min", min(x), F + "/e3_*/out/retention_series.json", "smallest value along T")
                put(f"ret_{b}{fn}_{short}_max", max(x), F + "/e3_*/out/retention_series.json", "largest value along T")
        rng(f"e3_{b}_errreb", [D.p("e5", f"logs.{lt[1]}.rebase.err_rebased_max", None) for lt in allu] + [D.p("e5", f"logs.{DEV[b]}.rebase.err_rebased_max")], PN_FILES["e5"], "largest relative error of a rebased value (x 100 = percent)")
    put("e3_retr_h3_abs_min", min(abs(Fc[f"e3_{b}_retr_h3_max"]["value"]) for b in DEV), e3src, "H3 against retrieval on the unseen logs: smallest lead of retrieval (points)")
    put("e3_retr_h3_abs_max", max(abs(Fc[f"e3_{b}_retr_h3_min"]["value"]) for b in DEV), e3src, "the same, largest lead of retrieval")
    assert all(Fc[f"e3_{b}_retr_h3_max"]["value"] < 0 for b in DEV)
    allogs = [(b, lt) for b in DEV for lt in [("dev", DEV[b], DEVTAG[b], 100)] + UNSEEN[b]]
    for nm, key in (("p1_a", "HthreePairOneAtCapA"), ("p1_b", "HthreePairOneAtCapB"), ("p2_a", "HthreePairTwoAtCapA"), ("p2_b", "HthreePairTwoAtCapB"), ("retr_b", "HthreeRetrAtCapB")):
        put(f"h3_cap_{nm}_max", max(fxp(D, b, lt, key) for b, lt in allogs), e3src, f"H3: largest mean number of edits at their guard cap under the raised scale over the 13 logs ({key})")
    put("h3_unreachable_max", max(fxp(D, b, lt, k) for b, lt in allogs for k in ("HthreePairOneUnreachable", "HthreePairTwoUnreachable", "HthreeRetrUnreachable")), e3src,
        "H3: calibration draws in which the near-scope constraint cannot be reached, largest count over logs and comparisons")
    for nm in ("near_rel", "near_rel_unseen", "draws_above", "retr_h3", "errreb"):
        put(f"e3_{nm}_min", min(Fc[f"e3_{b}_{nm}_min"]["value"] for b in DEV), e3src, "minimum over both backbones")
        put(f"e3_{nm}_max", max(Fc[f"e3_{b}_{nm}_max"]["value"] for b in DEV), e3src, "maximum over both backbones")
    # ---- validity (E5 c): ranges over exchangeable pairs and over the reverse shift
    v = lambda lg, rule, pair, leaf: D.p("e5", f"validity.{lg}.{rule}.{pair}.{leaf}")              # noqa: E731
    same = ("repair_to_repair_rest", "ben_s_to_ben_s_rest", "mme_c_to_mme_c_rest")
    rng("val_same_above", [v(lg, "two_block", p, "share_of_draws_above_alpha") for lg in DEV.values() for p in same], PN_FILES["e5"], "two-block rule, same pool: share of draws above the level")
    rng("val_same_mean", [v(lg, "two_block", p, "ff_mean") for lg in DEV.values() for p in same], PN_FILES["e5"], "two-block rule, same pool: mean realised rate")
    exch = same + ("ben_s_to_ben_t", "mme_c_to_mme_t")
    put("val_pac_exch_above_max", max(v(lg, "pac", p, "share_of_draws_above_alpha") for lg in DEV.values() for p in exch), PN_FILES["e5"], "calibration-conditional rank, exchangeable pairs: largest share of draws above the level")
    rev = ("mme_c_to_heldout", "mme_c_to_ben_t")
    rng("val_rev", [v(lg, "two_block", p, "ff_mean") for lg in DEV.values() for p in rev], PN_FILES["e5"], "two-block rule calibrated on the out-of-benchmark workload, tested on benchmark queries or the benign pool: mean realised rate")
    rng("val_rev_pac", [v(lg, "pac", p, "ff_mean") for lg in DEV.values() for p in rev], PN_FILES["e5"], "the same for the calibration-conditional rank")
    vd = D.o["vdist"]["logs"][DEV["Q"]]["pairs"]["repair_to_repair_rest"]["two_block"]
    key = [k for k in vd if k.startswith("beta_prob_a")]
    put("val_beta_above", vd[key[0]] * (100 if vd[key[0]] <= 1 else 1), OTHER["vdist"], f"Beta({vd['beta_a']}, {vd['beta_b']}) law of the two-block rule: probability (percent) that the conditional rate exceeds the level ({key[0]})")
    put("val_beta_n2", vd["n2"], OTHER["vdist"], "calibration scores in the second block")
    put("val_bound", vd["beta_mean_pct"], OTHER["vdist"], "mean of the Beta law = the bound of the two-block rule (percent)")

    # ---- E9: ranges over m for the scoped log
    ms = ("One", "Five", "Ten", "Twenty", "Fifty")
    for b in DEV:
        put(f"rev_{b}_scoped_logits_max", max(D.p("e9", f"Enine{b}ScopedM{m}OutsideLogits") for m in ms), PN_FILES["e9"], "scoped log: other tested sequences whose logits change, largest over m")
        put(f"rev_{b}_scoped_flags_max", max(D.p("e9", f"Enine{b}ScopedM{m}OutsideFlags") for m in ms), PN_FILES["e9"], "scoped log: answer flags that change on them, largest over m")
        put(f"rev_{b}_scoped_revoked_max", max(D.p("e9", f"Enine{b}ScopedM{m}RevokedRel") for m in ms), PN_FILES["e9"], "scoped log: Rel exact of the revoked requests after masking, largest over m")
        rng(f"rev_{b}_linear_d1", [D.p("e9", f"Enine{b}LinearM{m}DOne") for m in ms], PN_FILES["e9"], "linear log after subtracting the recorded terms: held-out D1 over m")
        put(f"rev_{b}_linear_all_logits", int(all(D.p("e9", f"Enine{b}LinearM{m}OutsideLogits") == D.p("e9", f"Enine{b}LinearM{m}OutsideN") for m in ms)),
            PN_FILES["e9"], "1 if for every m the logits of EVERY other tested sequence change under subtraction")

    # ---- E8, cost of ReScope-W (final revision, T3 / FINAL_CHECK M3). The all-positions form with the calibrated thresholds
    # is in the paper numbers (keys *_all_positions); the form with LOG-ONLY thresholds (mode where_logonly) is in the
    # records only: primary latency pass, medians over 30 sequences x 9 repetitions (out/table.txt, E8 (1)).
    for b, lg in DEV.items():
        rows = {r["metric"]: r for r in D.o["e8res"] if r["log"] == lg and r["method"] == "where_logonly" and r.get("primary_pass")}
        allp = {r["metric"]: r for r in D.o["e8res"] if r["log"] == lg and r["method"] == "all_fp64" and r.get("primary_pass")}
        assert abs(allp["gate_only_ms@median"]["value"] - D.p("e8", f"{lg}.gate_ms_per_forward_all_positions")) < 1e-9, "all_fp64 records are the *_all_positions paper numbers"
        assert abs(allp["overhead_end_to_end_pct@median"]["value"] - D.p("e8", f"{lg}.overhead_end_to_end_pct_all_positions")) < 1e-9
        e8src = OTHER["e8res"]
        put(f"cost_{b}_wlog_gate_ms", rows["gate_only_ms@median"]["value"], e8src, "ReScope-W with log-only thresholds (mode where_logonly), gate alone, all positions: median ms per forward pass, primary pass")
        put(f"cost_{b}_wlog_over_pct", rows["overhead_end_to_end_pct@median"]["value"], e8src, "the same mode: median paired end-to-end overhead of a forward pass (percent), primary pass")
        for nm, rr in (("wlog", rows), ("wall", allp)):
            put(f"cost_{b}_{nm}_over_p10", rr["overhead_end_to_end_pct@median"]["ci_lo"], e8src, "10th percentile of the paired overhead over sequences x repetitions (" + nm + ")")
            put(f"cost_{b}_{nm}_over_p90", rr["overhead_end_to_end_pct@median"]["ci_hi"], e8src, "90th percentile of the same (" + nm + ")")
        # figure revision of 2026-11-12 (DECISIONS_FIG_REVISION.txt, G17; FINAL_CHECK_2 N8): the experiment timed the development
        # logs in TWO passes on a shared, loaded card (records with primary_pass false = pass 2); Sec. 5.7 prints both for the gate
        # of ReScope (mode last_fp64) and for the all-positions form of ReScope-W (mode all_fp64).
        for nm, mode in (("s", "last_fp64"), ("wall", "all_fp64"), ("wlog", "where_logonly")):
            first = {r["metric"]: r for r in D.o["e8res"] if r["log"] == lg and r["method"] == mode and r.get("primary_pass") is True}
            second = {r["metric"]: r for r in D.o["e8res"] if r["log"] == lg and r["method"] == mode and r.get("primary_pass") is False}
            assert second["gate_only_ms@median"]["latency_pass"] == 2 and first["gate_only_ms@median"]["latency_pass"] == 1
            if nm == "s":
                assert abs(first["gate_only_ms@median"]["value"] - D.p("e8", f"{lg}.gate_ms_per_forward")) < 1e-9, "last_fp64, pass 1, is the paper number of the gate of ReScope"
                assert abs(first["overhead_end_to_end_pct@median"]["value"] - D.p("e8", f"{lg}.overhead_end_to_end_pct")) < 1e-9
            put(f"cost_{b}_{nm}_gate_ms_p2", second["gate_only_ms@median"]["value"], e8src, f"mode {mode}, gate alone: median ms per forward pass, SECOND latency pass (primary_pass false)")
            put(f"cost_{b}_{nm}_over_pct_p2", second["overhead_end_to_end_pct@median"]["value"], e8src, f"mode {mode}: median paired end-to-end overhead (percent), SECOND latency pass")

    med = [D.p("e9", f"Enine{b}Rollback{k}KMedian") for b in DEV for k in ("Image", "Text")]
    put("rb_k_median_min", min(med), PN_FILES["e9"], "rollback of the linear log for ONE complaint: smallest median number of deleted edits (image / text complaints, both logs)")
    put("rb_k_median_max", max(med), PN_FILES["e9"], "the same, largest median")
    dl = [abs(D.p("e6", f"Esix{b}ValuesLogged{k}") - D.p("e6", f"Esix{b}Default{k}")) for b in DEV for k in ("Rel", "TGen", "IGen")]
    put("abl_logged_maxdiff", max(dl), PN_FILES["e6"], "logged instead of rebased values: largest |difference| of Rel / T-Gen / I-Gen exact to the default, both logs (points)")
    lv = [n for n, k in ((1, "One"), (5, "Five"), (20, "Twenty")) if all(f"Esix{b}Contam{k}CovT" in D.pn["e6"] for b in DEV)]
    put("contam_levels", ", ".join(str(x) for x in lv), PN_FILES["e6"], "contamination levels (percent of the calibration sample) for which E6 has rows")
    # ---- E1b / E1c: one common threshold per feature at matched coverage of the rephrasings (frontier.json)
    fr = json.load(open(os.path.join(D.root, F, "e1_main/out/frontier.json")))["logs"]
    for b, lg in DEV.items():
        for feat in ("ncoord", "cos", "white_0.1"):
            for row in fr[lg]["matched_coverage"][feat]:
                if row["target"] == 50 and row["reachable"]:
                    put(f"front_{b}_{feat}_50_ff", row["heldout_disjoint_fired"], F + "/e1_main/out/frontier.json", "matched coverage 50 %: false firing on heldout_disjoint under one common threshold")
                    put(f"front_{b}_{feat}_50_near", sum(row["hn_fired"][k] for k in ("HN-I", "HN-Q", "HN-X")) / 3, F + "/e1_main/out/frontier.json", "the same threshold: release on near-scope negatives, mean of the three kinds")
    # ---- deployed logs as evaluated here (Table tab:logs)
    for lg in D.o["logs"]["logs"]:
        d = lg["deployed"]
        for k_out, k_in in (("rel", "rel_exact"), ("tgen", "tgen_exact"), ("igen", "igen_exact")):
            put(f"logs_b1.{lg['log']}.{k_out}", d[k_in]["value"], OTHER["logs"], f"{d[k_in]['numerator']} of {d[k_in]['denominator']}")
        put(f"logs_b1.{lg['log']}.d1", d["D1_heldout_disjoint"]["value"], OTHER["logs"], f"{d['D1_heldout_disjoint'].get('numerator')} of {d['D1_heldout_disjoint'].get('denominator')}")
    sf = os.path.join(PAPER, "scripts", "data", "setup_facts.json")
    if os.path.exists(sf):                       # logs that the final experiments did not evaluate (working points; no row in the table)
        for name, rec in json.load(open(sf))["logs"].items():
            if f"logs_b1.{name}.rel" not in Fc and isinstance(rec, dict) and "edits" in rec:
                for k_out in ("rel", "tgen", "igen"):
                    put(f"logs_b1.{name}.{k_out}", rec["edits"][k_out]["exact"], "scripts/data/setup_facts.json", "NOT evaluated in the final experiments: value stored by the testbed (the log has no row in Table tab:logs)")
    return Fc


def _count_variants(best, kind):
    """number of variants tabulated for the selection (the same list on both backbones: asserted)."""
    key = "table_retrieval_family" if kind == "retr" else "table_cosine_family"
    ns = {len(v[key]) for v in best["backbones"].values()}
    assert len(ns) == 1, ns
    return ns.pop()


def fxp(D, b, lt, name, default=None):
    """fix-up paper number of E3 for a log"""
    return D.p("e3qf" if b == "Q" else "e3lf", f"EthreeFx{b}{lt[2]}{name}", default)


def xr_or(D, b, lt, name):
    return fxp(D, b, lt, name)


# ------------------------------------------------------------------------------------------------ tables
def near3(D, b, row):
    """mean near-scope release over the three kinds at each behavioural draw: 'd0/d1/d2'; one value where one exists."""
    try:
        per = []
        for d in ("Zero", "One", "Two"):
            per.append(sum(D.p("e1f", f"EoneFx{b}{row}HN{k}FireD{d}") for k in "IQX") / 3)
        return "/".join(fx(x, 0) for x in per)
    except KeyError:
        vals = [D.p("e1f", f"EoneFx{b}{row}HN{k}FireMean", None) for k in "IQX"]
        return "--" if None in vals else fx(sum(vals) / 3, 0)


def main_cap(D, b, row, where_cal=False, comp=False):
    """capped edits of a calibrated row (mean over the 20 gate-level draws); '--' for rows without calibration."""
    v = D.p("e2", f"Etwo{b}WCalCapped") if where_cal else D.p("e1", f"Eone{b}{row}Capped", None)
    return "--" if v is None or comp else fx(v, 1)


def main_cells(D, b, row, e1row=None, where_cal=False, comp=False):
    g1 = lambda n, d=None: D.p("e1", f"Eone{b}{e1row or row}{n}", d)                           # noqa: E731
    gf = lambda n, d=None: D.p("e1f", f"EoneFx{b}{row}{n}", d)                                  # noqa: E731
    if where_cal:
        e = lambda n: D.p("e2", f"Etwo{b}WCal{n}")                                              # noqa: E731
        hn = [rec1(D, DEV[b], "cw", "where_cal", "two_block", 200, f"gate.hn.{k}.fired@mean") for k in ("HN-I", "HN-Q", "HN-X")]
        near = [max(r, key=lambda x: (x.get("n_units") or 0, -(x.get("draw") or 0)))["value"] for r in hn]
        nearv = _cal_near_20(D, b)
        return [fx(e("Rel"), 0), fx(e("TGen"), 0), fx(e("IGen"), 0), "--", fx(xr(D, b, DEVTAG[b], "WhereCalHdReleased"), 1),
                fx(e("Done"), 1), fx(e("Dtwo"), 1), fx(nearv, 0) + "$^{g}$"]
    if comp:
        e = lambda n: D.p("e2", f"Etwo{b}RowCompBAll{n}")                                       # noqa: E731
        tg, ig = D.p("e1f", f"EoneFx{b}RowCompBAllGenTGenEq", None), D.p("e1f", f"EoneFx{b}RowCompBAllGenIGenEq", None)
        rr = D.o["e2res"]
        t = [r["value"] for r in rec1(D, DEV[b], "d", "COMP_B", "rebuilt", "ALL", "edit.tgen.exact")]
        i = [r["value"] for r in rec1(D, DEV[b], "d", "COMP_B", "rebuilt", "ALL", "edit.igen.exact")]
        gen = "--" if tg is None else fx((tg + ig) / 2, 0)
        return [fx(e("Rel"), 0), fx(t[0], 0), fx(i[0], 0), gen, "--", fx(e("Done"), 1), fx(e("Dtwo"), 1), "--"]
    tg, ig = gf("GenTGenEq"), gf("GenIGenEq")
    gen = "--" if tg is None else fx((tg + ig) / 2, 0)
    d2 = gf("Dtwo")
    fire = g1("Fire")
    return [fx(g1("Rel"), 0), fx(g1("TGen"), 0), fx(g1("IGen"), 0), gen, "--" if fire is None else fx(fire, 1),
            fx(g1("DOne"), 1), "--" if d2 is None else fx(d2, 1), near3(D, b, row) if fire is not None else "--"]


def _cal_near_20(D, b):
    """ReScope-W with two-block thresholds: mean near-scope release over the three kinds, gate level, 20 draws (the
    record of E2 with the larger number of draws; the key holds two records, see FIXUP.txt of e1_main)."""
    vals = []
    for k in ("HN-I", "HN-Q", "HN-X"):
        rs = rec1(D, DEV[b], "cw", "where_cal", "two_block", 200, f"gate.hn.{k}.fired@mean")
        assert rs, k
        # the 20-draw record is the one with an interval over draws (ci from 20 draws); pick by the note if present
        best = None
        for r in rs:
            nd = r.get("n_draws") or r.get("n_units") or 0
            if best is None or nd > best[0]:
                best = (nd, r["value"])
        vals.append(best[1])
    return sum(vals) / 3


def tab_main(D, full=False):
    rows = [
        ("--", r"deployed log", dict(row="Deployed")),
        ("log", r"\sysW, log-only thresholds", dict(row="RescopeW")),
        ("log, $n$", r"\bCompB, all complaints", dict(row="RowCompBAll", comp=True)),
        # W4, 2026-11-12 (page budget, READING_FOR_W4.txt part E, cut (2) = FIXES_FOR_FINAL_PASS F32 (i)): the rows
        # "ReScope-W, two-block thresholds" and "lookup on the requests" are rows of the REPORT's table only; their
        # numbers stay in the text of Secs. 5.2 / 5.3 (\vAQCalDone, \vBHTwoOtherMin ...). The class label "log, B" moves
        # to the row of \sys in the paper's table.
        (r"log, $\Bcal$", r"\sysW, two-block thresholds", dict(row="WCal", where_cal=True)) if full else None,
        ("" if full else r"log, $\Bcal$", r"\sys", dict(row="Rescope")),
        ("", r"\sys, log-derived negatives", dict(row="RescopeX")),
        ("", r"cosine on solver keys", dict(row="Cosine")) if full else None,
        (r"log, $\Bcal$, stat.", r"whitened gate, selected for H2", dict(row="BestCosHtwo")),
        ("", r"whitened gate, selected for H3", dict(row="BestCos")),
        ("log", r"cosine, published threshold", dict(row="CosPublished")) if full else None,
        ("requests", r"lookup on the requests", dict(row="Lookup")) if full else None,
        (r"requests, enc., $\Bcal$", r"retrieval, two encoders", dict(row="Retr")),
        ("edit targets", r"answer match (\bDual)", dict(row="Dual")) if full else None,
    ]
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers) from the paper numbers of e1_main, e2_curves and their fix-ups. Do not edit.",
           r"\begin{tabular}{@{}llcccccccccccccccc@{}}", r"\toprule",
           r"& & \multicolumn{8}{c}{\lgQAcName} & \multicolumn{8}{c}{\lgLAcName} \\",
           r"\cmidrule(lr){3-10}\cmidrule(l){11-18}",
           r"Needs & Method & \Rel & \TGen & \IGen & Gen & Rls & \Done & \Dtwo & Near & \Rel & \TGen & \IGen & Gen & Rls & \Done & \Dtwo & Near \\",
           r"\midrule"]
    if full:
        out[1] = r"\begin{tabular}{@{}llcccccccccccccccccc@{}}"
        out[3] = r"& & \multicolumn{9}{c}{\lgQAcName} & \multicolumn{9}{c}{\lgLAcName} \\"
        out[4] = r"\cmidrule(lr){3-11}\cmidrule(l){12-20}"
        out[5] = r"Needs & Method & \Rel & \TGen & \IGen & Gen & Rls & \Done & \Dtwo & Near & Cap & \Rel & \TGen & \IGen & Gen & Rls & \Done & \Dtwo & Near & Cap \\"
    for needs, label, kw in [r for r in rows if r]:
        cells = []
        for b in ("Q", "L"):
            cells += main_cells(D, b, **kw)
            if full:
                cells.append(main_cap(D, b, kw["row"], kw.get("where_cal", False), kw.get("comp", False)))
        out.append(f"{needs} & {label} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_scale(D):
    """part (a) of Table tab:scale: ONE LINE PER LOG FAMILY (page decision P5), values along T separated by slashes;
    the first family starts with its development log (T = 100). Per log and method: technical report, Appendix C."""
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers) from the paper numbers of e3_qwen, e3_llava and their fix-ups. Do not edit.",
           r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
           r"& & \multicolumn{2}{c}{Deployed} & \multicolumn{2}{c}{\sysW, log-only} & \multicolumn{4}{c}{\sys} & Retrieval \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-10}\cmidrule(l){11-11}",
           r"(a) Logs & $T$ & \Done & \Rel & \Done & \Rel & \Done & \Rel & Gen & Near & Gen \\",
           r"\midrule"]
    for b in ("Q", "L"):
        dev = ("QAc" if b == "Q" else "LAc", DEV[b], DEVTAG[b], 100)
        for fam, logs in (("a", [dev] + [lt for lt in UNSEEN[b] if lt[1].endswith("ae-l2_3")]),
                          ("m", [lt for lt in UNSEEN[b] if lt[1].endswith("memitw")])):
            rows = {m: [e3row(D, b, lt, m) for lt in logs] for m in ("deployed", "where", "rescope", "retr")}
            j = lambda m, k, n: "/".join(fx(r[k], n) for r in rows[m])                              # noqa: E731
            label = rf"{{\lg{logs[0][0]}Name}} to {{\lg{logs[-1][0]}Name}}"
            cells = ["/".join(str(lt[3]) for lt in logs), j("deployed", "D1", 0), j("deployed", "Rel", 0), j("where", "D1", 1), j("where", "Rel", 0),
                     j("rescope", "D1", 1), j("rescope", "Rel", 0), j("rescope", "GenLater", 0), j("rescope", "Near", 0), j("retr", "GenLater", 0)]
            out.append(label + " & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_maint(D):
    ms = (("One", 1), ("Fifty", 50))                                        # all five m: e9_maintenance/out/table.txt
    g = lambda b, kind, m, n: D.p("e9", f"Enine{b}{kind}M{m}{n}")                                # noqa: E731
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers) from the paper numbers of e9_maintenance and e8_cost. Do not edit.",
           r"\begin{tabular}{@{}rcccccccccccc@{}}", r"\toprule",
           r"(b) & \multicolumn{3}{c}{{\lgQAcName}: mask} & \multicolumn{3}{c}{subtract} & \multicolumn{3}{c}{{\lgLAcName}: mask} & \multicolumn{3}{c}{subtract} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}",
           r"$m$ & Lgt & Ans & \Done & Lgt & Ans & \Done & Lgt & Ans & \Done & Lgt & Ans & \Done \\", r"\midrule"]
    for m, mv in ms:
        cells = []
        for b in ("Q", "L"):
            for kind in ("Scoped", "Linear"):
                n = g(b, kind, m, "OutsideN")
                cells += [f"{fx(g(b, kind, m, 'OutsideLogits'), 0)}/{fx(n, 0)}", fx(g(b, kind, m, "OutsideFlags"), 0), fx(g(b, kind, m, "DOne"), 1)]
        out.append(f"{mv} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}\hfill"]
    e8 = lambda k: D.p("e8", k)                                                                 # noqa: E731
    out += [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
            r"(c) Log & $T$ & MB & \%par & Gate & Cal. & Rep. & Rev. \\", r"\midrule"]
    for b, lg in list(DEV.items())[:1]:                                     # the Qwen development log (LLaVA: the text of Sec. 5.7)
        bb = {"Q": "qwen25vl", "L": "llava"}[b]
        T = D.p("xr", f"Xrel{b}{DEVTAG[b]}LogT")
        share = 100.0 * T / e8(f"storage.{bb}.T_at_model_size")
        out.append(rf"{{\lg{b}AcName}} & {T} & {fx(e8(lg + '.storage_mb'), 1)} & {fx(share, 2)} & {fx(e8(lg + '.gate_ms_per_forward'), 2)} & "
                   rf"{fx(e8(lg + '.calibration_s'), 2)} & {fx(e8(lg + '.repair_ms_per_complaint'), 2)} & {fx(e8(lg + '.revocation_ms'), 2)} \\")
    for T, lab in ((100000, "10^5"),):
        k = f"synthetic.T{T}."
        out.append(rf"synthetic & ${lab}$ & {fx(e8(k + 'storage_mb'), 0)} & {fx(e8(k + 'storage_share_of_model_pct'), 2 if T < 10000 else 1)} & {fx(e8(k + 'gate_ms_per_forward'), 2)} & {fx(e8(k + 'calibration_s'), 2)} & "
                   rf"{fx(e8(k + 'repair_ms_per_complaint'), 1)} & {fx(e8(k + 'revocation_ms'), 2)} \\")
    out.append(rf"\emph{{published}}$^{{c}}$ & $10^5$ & -- & {fx(e8('published.neuraldb.storage_at_1e5_pct'), 0)} & -- & -- & -- & -- \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


# ---- technical report (Appendix C)
def tab_tr_unseen(D, first):
    """the logs of E3 of one editor family (first family: with the development logs), nine methods."""
    meth = (("deployed", "deployed log"), ("where", r"\sysW, log-only"), ("rescope", r"\sys"), ("rescope_x", r"\sys, log-derived negatives"),
            ("white", "whitened gate (H2)"), ("whiteh3", "whitened gate (H3)"), ("cosine", "cosine on solver keys"),
            ("retr", "retrieval"), ("lookup", "lookup"))
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers). Do not edit.",
           r"\begin{tabular}{@{}llccccccccc@{}}", r"\toprule",
           r"Log & Method & \Rel & \TGen & \IGen & \TGen$_{>}$ & \IGen$_{>}$ & Rls & \Done & Near & Cap \\", r"\midrule"]
    for b in ("Q", "L"):
        logs = ([("QAc" if b == "Q" else "LAc", DEV[b], DEVTAG[b], 100)] if first else []) + \
               [lt for lt in UNSEEN[b] if lt[1].endswith("ae-l2_3" if first else "memitw")]
        for lt in logs:
            j = 0
            for m, lab in meth:
                try:
                    r = e3row(D, b, lt, m)
                except KeyError:
                    continue
                if r["D1"] is None and r["Rel"] is None:
                    continue
                lat = lambda k: "--" if r.get(k) is None else fx(r[k], 0)                 # noqa: E731
                cells = [fx(r["Rel"], 0), fx(r["TGen"], 0), fx(r["IGen"], 0), lat("TGenLater"), lat("IGenLater"),
                         "--" if r["Fire"] is None else fx(r["Fire"], 1), fx(r["D1"], 1),
                         "--" if r["Near"] is None else fx(r["Near"], 0), "--" if r["Capped"] is None else fx(r["Capped"], 1)]
                out.append((rf"{{\lg{lt[0]}Name}}" if j == 0 else "") + f" & {lab} & " + " & ".join(cells) + r" \\")
                j += 1
            out.append(r"\addlinespace[1pt]")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_repl(D):
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers). Do not edit.",
           r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
           r"Log & H2, all edits & H2, later edits & H3, pair 1 & H3, pair 2 & H3, retrieval \\", r"\midrule"]

    def ci(b, lt, stem):
        d = fxp(D, b, lt, stem + "Diff")
        if d is None:
            return "--"
        return f"{sg(d)} [{sg(fxp(D, b, lt, stem + 'Lo'))}, {sg(fxp(D, b, lt, stem + 'Hi'))}]"
    for b in ("Q", "L"):
        for lt in [("QAc" if b == "Q" else "LAc", DEV[b], DEVTAG[b], 100)] + UNSEEN[b]:
            later = ci(b, lt, "HtwoCosLater") if lt[3] > 100 else "--"
            out.append(rf"{{\lg{lt[0]}Name}} & {ci(b, lt, 'HtwoCosAll')} & {later} & {ci(b, lt, 'HthreePairOne')} & {ci(b, lt, 'HthreePairTwo')} & {ci(b, lt, 'HthreeRetr')} \\")
        if b == "Q":
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_ablation(D):
    rows = (("Default", "default (two-block rule, last position, replace, rebased)"), ("CalShared", "one shared scale"),
            ("CalLogOnly", "log-only thresholds"), ("CalRunMax", "per-edit running maxima"), ("Pac", r"calibration-conditional rank"),
            ("RoleAll", "all positions eligible"), ("RespLin", "linear response"), ("RespGain", "gain response"),
            ("ValuesLogged", "logged instead of rebased values"), ("GuardsOff", "guards off"), ("FireAll", "all firing edits released"),
            ("ContamOne", r"calibration sample contaminated, 1\,\%"), ("ContamFive", r"contaminated, 5\,\%"), ("ContamTwenty", r"contaminated, 20\,\%"))
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers). Do not edit.",
           r"\begin{tabular}{@{}lcccccccccccccccc@{}}", r"\toprule",
           r"& \multicolumn{8}{c}{\lgQAcName} & \multicolumn{8}{c}{\lgLAcName} \\", r"\cmidrule(lr){2-9}\cmidrule(l){10-17}",
           r"Variant & FF & Near & Cov$_T$ & Cov$_I$ & \Rel & \TGen & \IGen & \Done & FF & Near & Cov$_T$ & Cov$_I$ & \Rel & \TGen & \IGen & \Done \\", r"\midrule"]
    for key, lab in rows:
        cells = []
        for b in ("Q", "L"):
            g = lambda n: D.p("e6", f"Esix{b}{key}{n}", None)                                    # noqa: E731
            cells += [fx(g("Fire"), 1), fx(g("NearFire"), 0), fx(g("CovT"), 0), fx(g("CovI"), 0), fx(g("Rel"), 0), fx(g("TGen"), 0), fx(g("IGen"), 0), fx(g("DOne"), 1)]
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_draws(D):
    rows = (("Rescope", r"\sys"), ("RescopeX", r"\sys, log-derived negatives"), ("Cosine", "cosine on solver keys"),
            ("BestCosHtwo", "whitened gate (H2)"), ("BestCos", "whitened gate (H3)"), ("Retr", "retrieval"))
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers). Do not edit.",
           r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
           r"& \multicolumn{4}{c}{\lgQAcName} & \multicolumn{4}{c}{\lgLAcName} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
           r"Method & \hnI & \hnQ & \hnX & WT & \hnI & \hnQ & \hnX & WT \\", r"\midrule"]
    for key, lab in rows:
        cells = []
        for b in ("Q", "L"):
            for k in "IQX":
                cells.append("/".join(fx(D.p("e1f", f"EoneFx{b}{key}HN{k}FireD{d}"), 0) for d in ("Zero", "One", "Two")))
            wt = D.p("e1f", f"EoneFx{b}{key}HNAllWrongTarget", None)
            cells.append(fx(wt, 1))
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_mcnemar(D):
    """exact McNemar tests of E1 (fix-up): ReScope against each row; calibration draw 0; b / c = discordant rows."""
    rows = (("Deployed", "deployed log"), ("RescopeW", r"\sysW, log-only"), ("RescopeX", r"\sys, log-derived negatives"),
            ("Cosine", "cosine on solver keys"), ("BestCosHtwo", "whitened gate (H2)"), ("BestCos", "whitened gate (H3)"),
            ("CosPublished", "cosine, published threshold"), ("Lookup", "lookup"), ("Retr", "retrieval"), ("Dual", "answer match"),
            ("Exact", "exact match"), ("Preedit", "pre-edit model"))
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers). Do not edit.",
           r"\begin{tabular}{@{}lcccccccccccc@{}}", r"\toprule",
           r"& \multicolumn{6}{c}{\lgQAcName} & \multicolumn{6}{c}{\lgLAcName} \\", r"\cmidrule(lr){2-7}\cmidrule(l){8-13}",
           r"& \multicolumn{3}{c}{\Done on \poolHeld} & \multicolumn{3}{c}{\Rel} & \multicolumn{3}{c}{\Done on \poolHeld} & \multicolumn{3}{c}{\Rel} \\",
           r"\sys against & $b$ & $c$ & $p$ & $b$ & $c$ & $p$ & $b$ & $c$ & $p$ & $b$ & $c$ & $p$ \\", r"\midrule"]

    def pv(x):
        if x is None:
            return "--"
        if x >= 0.001:
            return fx(x, 3)
        e = int(math.floor(math.log10(x))) if x > 0 else -300
        return rf"$<10^{{{e + 1}}}$"
    for key, lab in rows:
        cells = []
        for b in ("Q", "L"):
            for oc in ("DOne", "Rel"):
                g = lambda n: D.p("e1f", f"EoneFx{b}{key}Mc{oc}DZero{n}", None)                 # noqa: E731
                cells += ["--" if g("B") is None else str(int(g("B"))), "--" if g("C") is None else str(int(g("C"))), pv(g("PHolm"))]
        if any(c != "--" for c in cells):
            out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_prefix(D):
    """the six prefix logs after 10 reported complaints (FINAL h1_prefix.json): D1 on heldout_disjoint / Rel exact."""
    cells = {k: c for k, c in D.o["h1p"]["cells"].items() if c["n"] == 10}
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers) from e2_prefix/out/h1_prefix.json. Do not edit.",
           r"\begin{tabular}{@{}lrcccccccccccc@{}}", r"\toprule",
           r"& & \multicolumn{2}{c}{Deployed} & \multicolumn{2}{c}{\sysW} & \multicolumn{2}{c}{\bCompB} & \multicolumn{2}{c}{\bRbUnion} & \multicolumn{2}{c}{\bRbTot} & H1 & \Rel${}\ge{}$dep.${}-\hOneRelPts$ \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}",
           r"Stream & $T$ & \Done & \Rel & \Done & \Rel & \Done & \Rel & \Done & \Rel & \Done & \Rel & (registered) & (\Done${}\le\hOneDoneMax$) \\", r"\midrule"]
    for b, lg in DEV.items():
        for T in (10, 20, 50):
            c = cells[f"{lg}-p{T}|n10"]
            v = c["verdict"]
            yn = lambda x: "met" if x else "not met"                                              # noqa: E731
            lin = c["linear"]
            row = [fx(c["deployed"]["hd_D1"], 1), fx(c["deployed"]["rel"], 0), fx(c["where_least"]["hd_D1"]["mean"], 1), fx(c["where_least"]["rel"]["mean"], 0)]
            for m in ("comp_b", "rb_union", "rb_tot"):
                row += [fx(lin[m]["hd_D1"], 1), fx(lin[m]["rel"], 0)]
            # final revision (T30): the second column is not called a reading of the rule: it says whether D1 is within the rule and Rel
            # is not more than 2 points below the deployed log (yes / no); the first is the registered rule (met / not met)
            row += [yn(v["D1_at_most_10"] and v["rel_within_2_points"]), "yes" if (v["D1_at_most_10"] and v["rel_not_more_than_2_below_deployed"]) else "no"]
            out.append(rf"{{\lg{b}AcName}} & {T} & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_cost(D):
    """revocation for every m (E9) and the cost figures of E8 in full."""
    ms = (("One", 1), ("Five", 5), ("Ten", 10), ("Twenty", 20), ("Fifty", 50))
    g = lambda b, kind, m, n: D.p("e9", f"Enine{b}{kind}M{m}{n}")                                # noqa: E731
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers) from the paper numbers of e9_maintenance and e8_cost. Do not edit.",
           r"\begin{tabular}{@{}rcccccccccccccccc@{}}", r"\toprule",
           r"& \multicolumn{8}{c}{\lgQAcName} & \multicolumn{8}{c}{\lgLAcName} \\", r"\cmidrule(lr){2-9}\cmidrule(l){10-17}",
           r"& \multicolumn{4}{c}{scoped log: mask} & \multicolumn{4}{c}{linear log: subtract} & \multicolumn{4}{c}{scoped log: mask} & \multicolumn{4}{c}{linear log: subtract} \\",
           r"$m$ & Lgt & Ans & Kept & \Done & Lgt & Ans & Kept & \Done & Lgt & Ans & Kept & \Done & Lgt & Ans & Kept & \Done \\", r"\midrule"]
    for m, mv in ms:
        cells = []
        for b in ("Q", "L"):
            for kind in ("Scoped", "Linear"):
                cells += [f"{fx(g(b, kind, m, 'OutsideLogits'), 0)}/{fx(g(b, kind, m, 'OutsideN'), 0)}", fx(g(b, kind, m, "OutsideFlags"), 0),
                          fx(g(b, kind, m, "KeptRel"), 1), fx(g(b, kind, m, "DOne"), 1)]
        out.append(f"{mv} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{4pt}"]
    e8 = lambda k: D.p("e8", k)                                                                 # noqa: E731
    out += [r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
            r"Log & $T$ & numbers & MB & \%par & gate (ms) & end to end (\%) [10th, 90th pct.] & cal.\ (s) & repair (ms) & revoke (ms) \\", r"\midrule"]
    Fc = facts(D)
    for b, lg in DEV.items():
        bb = {"Q": "qwen25vl", "L": "llava"}[b]
        T = D.p("xr", f"Xrel{b}{DEVTAG[b]}LogT")
        out.append(rf"{{\lg{b}AcName}} & {T} & {e8(lg + '.storage_numbers'):,d} & {fx(e8(lg + '.storage_mb'), 1)} & {fx(100.0 * T / e8(f'storage.{bb}.T_at_model_size'), 2)} & "
                   rf"{fx(e8(lg + '.gate_ms_per_forward'), 2)} & {fx(e8(lg + '.overhead_end_to_end_pct'), 1)} [{sg(e8(lg + '.overhead_end_to_end_pct_p10'), 0)}, {sg(e8(lg + '.overhead_end_to_end_pct_p90'), 0)}] & "
                   rf"{fx(e8(lg + '.calibration_s'), 2)} & {fx(e8(lg + '.repair_ms_per_complaint'), 2)} & {fx(e8(lg + '.revocation_ms'), 2)} \\")
        # final revision (T3; FINAL_CHECK M3, S9 e): the gate of ReScope-W on the same log, which evaluates every position (same store; the first row is the gate of ReScope)
        # figure revision of 2026-11-12 (G21; FINAL_CHECK_2 S8): the row is mode all_fp64 of e8_cost/code/lat.py, i.e. all positions with the
        # thresholds of the ReScope calibration (the two-block thresholds of ReScope-W itself were not timed); the label says so
        # (S8 proposed to add '(the form of ReScope-W)': 57 pt too wide for the table; the caption and the text of the report say it)
        out.append(rf"\quad all positions, thresholds of the \sys calibration & & & & & {fx(e8(lg + '.gate_ms_per_forward_all_positions'), 2)} & {fx(e8(lg + '.overhead_end_to_end_pct_all_positions'), 1)} "
                   rf"[{sg(Fc[f'cost_{b}_wall_over_p10']['value'], 0)}, {sg(Fc[f'cost_{b}_wall_over_p90']['value'], 0)}] & & & \\")
        out.append(rf"\quad \sysW, log-only thresholds & & & & & {fx(Fc[f'cost_{b}_wlog_gate_ms']['value'], 2)} & {fx(Fc[f'cost_{b}_wlog_over_pct']['value'], 1)} "
                   rf"[{sg(Fc[f'cost_{b}_wlog_over_p10']['value'], 0)}, {sg(Fc[f'cost_{b}_wlog_over_p90']['value'], 0)}] & & & \\")
    for bb, lab in (("qwen25vl", r"\bbQs"), ("llava", r"\bbLs")):
        for T, tl in ((1000, "10^3"), (10000, "10^4"), (100000, "10^5")):
            k = f"synthetic.{bb}.T{T}."
            out.append(rf"synthetic, {lab} & ${tl}$ & {int(T * e8(f'storage.{bb}.numbers_per_edit')):,d} & {fx(e8(k + 'storage_mb'), 0)} & {fx(e8(k + 'storage_share_of_model_pct'), 2)} & "
                       rf"{fx(e8(k + 'gate_ms_per_forward'), 2)} & -- & {fx(e8(k + 'calibration_s'), 2)} & {fx(e8(k + 'repair_ms_per_complaint'), 1)} & {fx(e8(k + 'revocation_ms'), 2)} \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tab_tr_width(D):
    """lineage size (E5 b): number of edits that fire on the PRE-EDIT trace of a query (the pre-edit firing set)."""
    rows = (("deployed", "deployed scoping (no thresholds; all positions)"), ("where_logonly", r"\sysW, log-only thresholds (all positions)"),
            ("two_block", r"\sys, two-block thresholds (last prompt position; \pcDraws draws pooled)"))
    out = ["% GENERATED by scripts/gen_vlm.py (make numbers) from e5_rescope/out/paper_numbers.json (keys width.<log>.<rule>.*). Do not edit.",
           r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
           r"& \multicolumn{4}{c}{\lgQAcName} & \multicolumn{4}{c}{\lgLAcName} \\", r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
           r"Scoped log & size $0$ (\%) & median & mean & max & size $0$ (\%) & median & mean & max \\", r"\midrule"]
    for key, lab in rows:
        cells = []
        for b in ("Q", "L"):
            g = lambda n: D.p("e5", f"width.{DEV[b]}.{key}.{n}")                                 # noqa: E731
            cells += [fx(g("share_zero"), 1), fx(g("median"), 0), fx(g("mean"), 2), str(int(g("max")))]
        out.append(f"{lab} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def tables(D):
    return {"tab_vlm_main.tex": tab_main(D), "tab_vlm_tr_main.tex": tab_main(D, full=True), "tab_vlm_scale.tex": tab_scale(D), "tab_vlm_maint.tex": tab_maint(D),
            "tab_vlm_tr_unseen_a.tex": tab_tr_unseen(D, True), "tab_vlm_tr_unseen_m.tex": tab_tr_unseen(D, False),
            "tab_vlm_tr_repl.tex": tab_tr_repl(D),
            "tab_vlm_tr_ablation.tex": tab_tr_ablation(D), "tab_vlm_tr_draws.tex": tab_tr_draws(D),
            "tab_vlm_tr_mcnemar.tex": tab_tr_mcnemar(D), "tab_vlm_tr_prefix.tex": tab_tr_prefix(D),
            "tab_vlm_tr_cost.tex": tab_tr_cost(D), "tab_vlm_tr_width.tex": tab_tr_width(D)}


def inputs():
    return list(PN_FILES.values()) + list(OTHER.values())


def build(root=None, manifest=None, out_path=None, verbose=False):
    root = root or find_root()
    out_path = out_path or os.path.join(root, OUT_REL)
    D = Data(root)
    doc = {"what": "derived facts of the vision-language experiments for the paper (scripts/gen_vlm.py); every value names its source and rule",
           "inputs": {}, "facts": facts(D), "warnings": []}
    for rel in inputs():
        st = os.stat(os.path.join(root, rel))
        doc["inputs"][rel] = {"mtime": int(st.st_mtime), "size": st.st_size}
    for fn, src in tables(D).items():
        p = os.path.join(PAPER, fn)
        if not os.path.exists(p) or open(p).read() != src:
            with open(p + ".tmp", "w") as f:
                f.write(src)
            os.replace(p + ".tmp", p)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path + ".tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(out_path + ".tmp", out_path)
    return doc, doc["warnings"]


def stale(root=None, path=None):
    root = root or find_root()
    path = path or os.path.join(root, OUT_REL)
    if not os.path.exists(path):
        return ["the collection does not exist"]
    if not os.path.isdir(os.path.join(root, F)):
        return None
    doc = json.load(open(path))
    out = []
    for rel, was in (doc.get("inputs") or {}).items():
        p = os.path.join(root, rel)
        now = None
        if os.path.exists(p):
            st = os.stat(p)
            now = {"mtime": int(st.st_mtime), "size": st.st_size}
        if now != was:
            out.append(f"{rel}: " + ("vanished" if now is None else "changed"))
    out += [f"{fn}: missing" for fn in TEXFILES if not os.path.exists(os.path.join(PAPER, fn))]
    return out


if __name__ == "__main__":
    if "--sync" in sys.argv:
        import vlm_macros
        vlm_macros.sync()
    else:
        d, w = build(verbose=True)
        print(f"wrote {OUT_REL} ({len(d['facts'])} facts) and {', '.join(TEXFILES)}")
        if "-v" in sys.argv:
            for k, v in d["facts"].items():
                if not k.startswith("logs_b1.") and not k.startswith("pfx_"):
                    print(f"  {k:36s} {v['value']}")
