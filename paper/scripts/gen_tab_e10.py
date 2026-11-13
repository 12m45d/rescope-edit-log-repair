#!/usr/bin/env python3
"""Experiment E10 (public ROME logs of GPT-2 XL) in the paper: generated table bodies and derived facts.

Called by scripts/gen_numbers.py as a DERIVED source (manifest["derived"], function build) at every `make numbers`,
or directly:   python3 scripts/gen_tab_e10.py

Reads (ROOT = project_3; nothing is typed here, every number is read from these files)
  E10 = 02_research/experiments/10_final/e10_text/run
  E10/out/summary/behave_<log>.json   behaviour per row (mean over calibration draws 0-2, per_draw, crossed bootstrap)
  E10/out/summary/gate_<log>.json     gate level over the 20 calibration draws (false firing on HELD-OUT prompts)
  E10/out/gate/<log>.json             the same, draw by draw (own requests released per draw)
  E10/out/summary/equal_<log>.json    equal evidence (n reported complaints; seeds)
  E10/out/summary/r22_<log>.json      relational maintenance workload (PROTOCOL.md, R22)
  E10/out/screening.json, E10/out/cert/<log>.json
  E10/verify/out/screening_T2000.json the VERIFIER's margins on the log with 2000 requests (ties = duplicate solver keys;
                                      PROTOCOL.md, reporting clarification (1) of 2026-11-11 16:41)
Writes (paper directory)
  tab_e10_text.tex    part (b) of Table tab:scale   (clean and collapsed text log, behaviour)
  tab_e10_maint.tex   part (a) of Table tab:maint   (relational workload, both logs)
  tab_e10_equal.tex   panel (c) of Figure fig:evidence (equal evidence on the text logs)
  scripts/data/e10_facts.json   derived facts for macros of numbers.tex (each with its source and rule)

Rules applied here (PROTOCOL.md, Section 6, "Reporting clarifications" after the verification of E10; verify/verdicts.json):
  * guards off on the collapsed log: the two-block rows are bimodal over calibration draws; their cells print the
    three behavioural draws separately (d0/d1/d2), never the mean alone;
  * screening at 2000 requests: negative margins and exact ties (|margin| < 1e-9; duplicate solver keys) apart, from
    the verifier's file;
  * least repair: the number of reported complaints on which no edit is released AS SERVED is counted from the
    experiment's rows (complaints.fired), never assumed;
  * requests logged from the split on: the largest value over ALL rows (behaviour, equal evidence, relational).
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
import os
import sys
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
E10 = "02_research/experiments/10_final/e10_text/run"
OUT_REL = "01_paper/v4_rescope/scripts/data/e10_facts.json"
LOGS = {"TOne": "TX-G2XL-rome-T100", "TFive": "TX-G2XL-rome-T500"}
TEXFILES = ("tab_e10_text.tex", "tab_e10_maint.tex", "tab_e10_equal.tex")


def find_root():
    env = os.environ.get("PROJECT3_ROOT")
    if env:
        return env
    return _RP.ROOT          # the working-tree view of this repository (rescope_paths.py)


def inputs():
    rel = [f"{E10}/out/screening.json", f"{E10}/verify/out/screening_T2000.json", f"{E10}/verify/verdicts.json"]
    for lg in LOGS.values():
        rel += [f"{E10}/out/summary/behave_{lg}.json", f"{E10}/out/summary/gate_{lg}.json", f"{E10}/out/gate/{lg}.json",
                f"{E10}/out/summary/equal_{lg}.json", f"{E10}/out/summary/r22_{lg}.json", f"{E10}/out/cert/{lg}.json"]
    return rel


def fx(v, n=1):
    """Fixed-point text of a number, rounded on its decimal value, half away from zero (rule of gen_numbers.fmt_number)."""
    if v is None:
        return "--"
    if isinstance(v, int):
        v = float(v)
    with localcontext() as ctx:
        ctx.prec = 400
        d = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-(n + 9)), rounding=ROUND_HALF_EVEN)
        return format(d.quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP), "f")


def tv(v):
    return fx(v, 3)[1:] if 0 <= v < 1 else fx(v, 3)        # .237 instead of 0.237 (column width)


def cnt(v):
    """A count that may be a mean over seeds / draws: integer if it is one, else one decimal."""
    return fx(v, 0) if abs(v - round(v)) < 1e-9 else fx(v, 1)


def load(root):
    j = lambda rel: json.load(open(os.path.join(root, rel)))
    D = {"screen": j(f"{E10}/out/screening.json"), "vscreen": j(f"{E10}/verify/out/screening_T2000.json")}
    for tag, lg in LOGS.items():
        D[tag] = {"behave": j(f"{E10}/out/summary/behave_{lg}.json"), "gate": j(f"{E10}/out/summary/gate_{lg}.json")["rows"],
                  "gate_draws": j(f"{E10}/out/gate/{lg}.json")["rows"], "equal": j(f"{E10}/out/summary/equal_{lg}.json"),
                  "r22": j(f"{E10}/out/summary/r22_{lg}.json"), "cert": j(f"{E10}/out/cert/{lg}.json")}
    return D


# ------------------------------------------------------------------------------------------------ part (b) of tab:scale
# (needs, label, row of the experiment, has guards)
TEXT_ROWS = [
    ("--", "deployed log", "deployed", False),
    ("--", "pre-edit model", "preedit", False),
    ("log", r"\sysW, log-only thresholds", "where_logonly", True),
    ("log, $\\Bcal$", r"\sysW, two-block thresholds", "where_twoblock", True),
    ("log, $\\Bcal$", r"\sys", "rescope", True),
    ("log, $\\Bcal$", r"cosine on solver keys, same calibration", "cosine", False),
    ("log, $\\Bcal$, stat.", r"whitened cosine (selected), same calibration", "whitecos_0.1", False),
    ("log", r"cosine on solver keys, published threshold", "cosine_published", False),
    ("log", r"exact match of a solver key", "exact", False),
    ("requests", r"keyed store on the subject string", "lookup_subject", False),
]
BIMODAL = ("where_twoblock@goff", "rescope@goff")           # verify/verdicts.json, residual risk 4
COS_GOFF = ("cosine", "whitecos_0.1")                       # calibrated cosine rows: guards-off run not printed (see facts)
T5KEYS = ("edit.request.exact.before", "edit.paraphrase.exact.before", "heldout.D1_tf", "heldout.kept_tf")


def bm(D, tag, row, key):
    return D[tag]["behave"][row]["metrics"][key]["value"]


def gl(D, tag, row, key):
    g = D[tag]["gate"].get(row)
    return None if g is None or row in ("deployed", "preedit") else g[key]["mean"]


def ff(D, tag, row):
    return gl(D, tag, row, "heldout.fired")


def per_draw(D, tag, row, key, n=0):
    pd = D[tag]["behave"][row]["per_draw"][key]
    return "/".join(fx(pd[k], n) for k in sorted(pd, key=int))


def tab_text(D):
    L = ["% tab_e10_text.tex -- GENERATED by scripts/gen_tab_e10.py (make numbers) from the summary files of experiment E10",
         "% (02_research/experiments/10_final/e10_text/run/out/summary/behave_<log>.json, gate_<log>.json). Do not edit.",
         "% Part (b) of Table tab:scale. Cells: mean over the behavioural calibration draws 0-2 (rows with $\\Bcal$) or the",
         "% single run; FF, Near: gate level, mean over the 20 calibration draws. Cells d0/d1/d2: the three draws separately",
         "% (two-block rule without guards on the collapsed log: bimodal; PROTOCOL.md, reporting clarification (3)).",
         r"\begin{tabular}{@{}llcccccccc@{\hspace{9pt}}cccccc@{}}",
         r"  \toprule",
         r"  & & \multicolumn{8}{c}{\txGRcName (clean)} & \multicolumn{6}{c}{\txGRdName (collapsed)} \\",
         r"  \cmidrule(lr){3-10}\cmidrule(l){11-16}",
         r"  (b) Needs & Method & Req & Par & \Done & Kept & FF & Near & PS & NS & Req$_<$ & Par$_<$ & \Done & Kept & FF & Near \\",
         r"  \midrule"]
    for needs, label, row, guards in TEXT_ROWS:
        c = [needs, label]
        b1 = D["TOne"]["behave"][row]
        c += [fx(bm(D, "TOne", row, k)) for k in ("edit.request.exact", "edit.paraphrase.exact", "heldout.D1_tf", "heldout.kept_tf")]
        c += [fx(ff(D, "TOne", row)), fx(gl(D, "TOne", row, "neighborhood.fired")),
              fx(b1["public"]["all.PS"]["value"]), fx(b1["public"]["all.NS"]["value"])]
        c += [fx(bm(D, "TFive", row, k)) for k in T5KEYS] + [fx(ff(D, "TFive", row)), fx(gl(D, "TFive", row, "neighborhood.fired"))]
        L.append("  " + " & ".join(c) + r" \\  % " + row)
        if guards:
            r2 = row + "@goff"
            c = [needs, r"\quad guards off" + (r"\,$^{\dagger}$" if r2 in BIMODAL else ""), r"\multicolumn{8}{c}{--}"]
            if r2 in BIMODAL:
                c += [per_draw(D, "TFive", r2, k) for k in T5KEYS]
            else:
                c += [fx(bm(D, "TFive", r2, k)) for k in T5KEYS]
            c += [fx(ff(D, "TFive", r2)), fx(gl(D, "TFive", r2, "neighborhood.fired"))]
            L.append("  " + " & ".join(c) + r" \\  % " + r2)
        if row in ("preedit", "whitecos_0.1"):
            L.append(r"  \addlinespace[2pt]")
    L += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ part (a) of tab:maint
def r22row(D, tag, label, m=None):
    for r in D[tag]["r22"]["rows"]:
        if r["label"] == label and r["m"] == m:
            return r
    return None


# (needs, label in the paper, label of the experiment, m)
MAINT_ROWS = [
    ("--", "deployed log", "deployed", None),
    ("--", "restore base (log discarded)", "restore-base", None),
    ("log", "rollback of the screened updates", "rb_screen", None),
    ("oracle", r"delete update \txSplit", "oracle: delete update 173", None),
    ("log", r"\sysW, log-only thresholds", "ReScope-W log-only", None),
    ("log", r"\quad guards off", "ReScope-W log-only (guards off)", None),
    ("log, $\\Bcal$", r"\sysW, two-block thresholds", "ReScope-W two-block", None),
    ("log, $\\Bcal$", r"\quad guards off\,$^{\dagger}$", "ReScope-W two-block (guards off)", None),
    ("log, $\\Bcal$", r"\sys", "migration (ReScope)", None),
    ("log, $\\Bcal$", r"\quad guards off\,$^{\dagger}$", "migration (ReScope) (guards off)", None),
    "M",
    ("log, $m$", r"\sysW, least repair, sacrificing", "ReScope-W least repair (sacrificing)", 10),
    # W3b, 2026-11-12 (FIXES_FOR_FINAL_PASS F14): the row "least repair, protecting" (equal to the sacrificing one on the
    # clean log) and the row "blocklist" (equal to the deployed log but for its column Chg) left the table for the budget;
    # both are in the experiment's out/table.txt, section 5; the text's macros and checks still read them.
    ("log, $m$", "rollback to the resolving prefix", "rb_suffix", 10),
    ("log, $m$", "rollback of the firing set", "rb_lineage", 10),
    ("log, $m$", "minimum compensation", "comp_min", 10),
    "S",
    ("requests", "keyed store on the subject string", "lookup_subject", None),
]
MAINT_BIMODAL = {"ReScope-W two-block (guards off)": "where_twoblock@goff", "migration (ReScope) (guards off)": "rescope@goff"}


def maint_cells(D, tag, r, collapsed):
    if r is None:
        return [r"\multicolumn{8}{c}{--}"]
    lost = r["edits_lost"]["n"]
    c = [fx(r["reference_preservation"]["value"])]
    if collapsed:
        beh = MAINT_BIMODAL.get(r["label"])
        if beh:
            # the same quantity draw by draw: retention of an edited tuple = its request is answered with the new object
            a = D[tag]["behave"][beh]["metrics"]["edit.request.exact.before"]["value"]
            assert abs(a - r["retention_before"]["value"]) < 1e-6, (r["label"], a, r["retention_before"]["value"])
            c += [per_draw(D, tag, beh, "edit.request.exact.before"), per_draw(D, tag, beh, "edit.request.exact.from")]
        else:
            c += [fx(r["retention_before"]["value"]), fx(r["retention_from"]["value"])]
    else:
        c += [fx(r["retention"]["value"]), fx(r["queries_exact_pct"]["value"])]
    c += [tv(r["aggregate_tv_mean"]["value"]), cnt(r["repair_passes"]["value"]), fx(r["repair_seconds"]["value"]),
          cnt(r["statements_changed"]["value"])]
    rg = r.get("edits_lost_range")
    c += [f"{rg[0]}--{rg[1]}" if collapsed and r["label"] in MAINT_BIMODAL and rg else cnt(lost)]
    return c


def tab_maint(D):
    d1 = r22row(D, "TOne", "blocklist", 10)["diagnosis_passes"]["value"]
    d5 = r22row(D, "TFive", "blocklist", 10)["diagnosis_passes"]["value"]
    t1 = r22row(D, "TOne", "blocklist", 10)["n_tuple_complaints"]
    t5 = r22row(D, "TFive", "blocklist", 10)["n_tuple_complaints"]
    L = ["% tab_e10_maint.tex -- GENERATED by scripts/gen_tab_e10.py (make numbers) from",
         "% 02_research/experiments/10_final/e10_text/run/out/summary/r22_<log>.json (PROTOCOL.md, R22). Do not edit.",
         "% Part (a) of Table tab:maint. Rows with m: mean over 3 complaint seeds; rows with \\Bcal: mean over the",
         "% calibration draws 0-2; dagger rows: Ret draw by draw (from the behavioural summary, same quantity; the mean",
         "% is asserted equal), Lost as the range over the draws.",
         r"\begin{tabular}{@{}llcccccccc@{\hspace{7pt}}cccccccc@{}}",
         r"  \toprule",
         r"  & & \multicolumn{8}{c}{\txGRcName (clean)} & \multicolumn{8}{c}{\txGRdName (collapsed)} \\",
         r"  \cmidrule(lr){3-10}\cmidrule(l){11-18}",
         r"  (a) Needs & Repair or reference & Ref & Ret & Q & TV & Pass & s & Chg & Lost"
         r" & Ref & Ret$_<$ & Ret$_\ge$ & TV & Pass & s & Chg & Lost \\",
         r"  \midrule"]
    for row in MAINT_ROWS:
        if row == "M":
            L.append(r"  \addlinespace[2pt]")
            L.append(r"  \multicolumn{18}{@{}l}{\emph{after $m=10$ complained queries; diagnosis charged: "
                     f"{cnt(d1)} forward passes ({cnt(t1)} tuple complaints) on \\txGRcName, {cnt(d5)} ({cnt(t5)}) on \\txGRdName"
                     r"}} \\")
            continue
        if row == "S":
            L.append(r"  \addlinespace[2pt]")
            continue
        needs, label, lab, m = row
        c = [needs, label] + maint_cells(D, "TOne", r22row(D, "TOne", lab, m), False) \
            + maint_cells(D, "TFive", r22row(D, "TFive", lab, m), True)
        L.append("  " + " & ".join(c) + r" \\  % " + lab + (f", m = {m}" if m else ""))
    L += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ panel of fig:evidence
EQ_ROWS = [("least repair, sacrif.", "least_sacrificing"), ("rollback, prefix", "rb_suffix"),
           ("rollback, firing set", "rb_lineage"), ("compensation", "comp_min")]
NS = (1, 10, 100)
NS_TABLE = (10, 100)      # W3b, 2026-11-12 (FIXES_FOR_FINAL_PASS F14): the rows n = 1 left panel (c) for the budget; facts, macros
                          # and checks still use NS (the n = 1 values that the text quotes are macros)


def eq_lost(rec):
    xs = [m["edits_lost"]["n"] for m in rec["meta"]]
    return sum(xs) / len(xs)


def eq_cells(D, tag, key):
    rec = D[tag]["equal"][key]
    m = rec["metrics"]
    suf = ".before" if tag == "TFive" else ""
    return [fx(m["heldout.D1_tf"]["value"]), fx(m["heldout.kept_tf"]["value"]), fx(m["edit.request.exact" + suf]["value"]),
            fx(m["edit.paraphrase.exact" + suf]["value"]), cnt(eq_lost(rec))]


def same_cells(D, tag, k1, k2):
    return eq_cells(D, tag, k1) == eq_cells(D, tag, k2)


def tab_equal(D):
    L = ["% tab_e10_equal.tex -- GENERATED by scripts/gen_tab_e10.py (make numbers) from",
         "% 02_research/experiments/10_final/e10_text/run/out/summary/equal_<log>.json and behave_<log>.json. Do not edit.",
         "% Panel (c) of Figure fig:evidence: n reported complaints on the two text logs (n = 10: mean of 3 seeds; rows n = 1: out/table.txt).",
         r"\begin{tabular}{@{}l@{\hspace{3pt}}r@{\hspace{4pt}}ccccc@{\hspace{5pt}}ccccc@{}}",
         r"  \toprule",
         r"  & & \multicolumn{5}{c}{\txGRcName} & \multicolumn{5}{c}{\txGRdName} \\",
         r"  \cmidrule(lr){3-7}\cmidrule(l){8-12}",
         r"  (c) Repair & $n$ & \Done & Kept & Req & Par & Lost & \Done & Kept & Req$_<$ & Par$_<$ & Lost \\",
         r"  \midrule"]

    def dep(tag):
        suf = ".before" if tag == "TFive" else ""
        return [fx(bm(D, tag, "deployed", "heldout.D1_tf")), fx(bm(D, tag, "deployed", "heldout.kept_tf")),
                fx(bm(D, tag, "deployed", "edit.request.exact" + suf)), fx(bm(D, tag, "deployed", "edit.paraphrase.exact" + suf)), "0"]
    # the blocklist changes listed texts only: on unreported prompts and on the requests it is the deployed log
    for tag in LOGS:
        for n in NS:
            assert eq_cells(D, tag, f"blocklist|n{n}") == dep(tag), ("blocklist differs from the deployed log", tag, n)
    L.append("  deployed; blocklist & any & " + " & ".join(dep("TOne") + dep("TFive")) + r" \\")
    for label, key in EQ_ROWS:
        L.append(r"  \addlinespace[1.5pt]")
        for i, n in enumerate(NS_TABLE):
            L.append("  " + (label if i == 0 else "") + f" & {n} & "
                     + " & ".join(eq_cells(D, "TOne", f"{key}|n{n}") + eq_cells(D, "TFive", f"{key}|n{n}")) + r" \\")
        if key == "least_sacrificing":
            # protecting form: equal to the sacrificing form on the clean log; on the collapsed log it refuses every
            # complaint and equals the log-only thresholds with guards (both asserted)
            assert all(same_cells(D, "TOne", f"least_sacrificing|n{n}", f"least_protecting|n{n}") for n in NS)
            p5 = [eq_cells(D, "TFive", f"least_protecting|n{n}") for n in NS]
            assert p5[0] == p5[1] == p5[2]
            for n in NS:
                rec = D["TFive"]["equal"][f"least_protecting|n{n}"]
                assert all(m["complaints_refused"] == m["n_complaints"] for m in rec["meta"]), "protecting form: not all refused"
            L.append(r"  \quad protecting & any & \multicolumn{5}{c}{\emph{as sacrificing}} & " + " & ".join(p5[0]) + r" \\")
    L += [r"  \bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ derived facts
def facts(D):
    F = {}

    def put(name, value, source, rule):
        F[name] = {"value": value, "source": source, "rule": rule}

    # (1) requests logged from the split on: largest share answered with the new object over ALL rows
    cand = []
    for row, rec in D["TFive"]["behave"].items():
        cand.append((rec["metrics"]["edit.request.exact.from"]["value"], "behaviour: " + row))
    for key, rec in D["TFive"]["equal"].items():
        cand.append((rec["metrics"]["edit.request.exact.from"]["value"], "equal evidence: " + key))
    for r in D["TFive"]["r22"]["rows"]:
        cand.append((r["retention_from"]["value"], "relational: " + r["label"] + (f" m={r['m']}" if r["m"] else "")))
    best = max(cand)
    put("from_split_max", best[0], "summary/behave_, equal_, r22_TX-G2XL-rome-T500.json",
        f"max over {len(cand)} rows of request exact / retention from position 173 on; attained by '{best[1]}'")
    put("from_split_max_row", best[1], "as from_split_max", "row that attains the maximum")
    bb = max((rec["metrics"]["edit.request.exact.from"]["value"], row) for row, rec in D["TFive"]["behave"].items())
    put("from_split_max_behaviour", bb[0], "summary/behave_TX-G2XL-rome-T500.json", f"max over the behavioural rows ('{bb[1]}')")

    # (2) guards off, two-block rule, collapsed log: own requests silenced per gate-level draw (20 draws)
    pd = D["TFive"]["gate_draws"]["rescope"]["goff"]["per_draw"]
    T = 500
    sil = sorted((round(T * (100.0 - pd[d]["request"]["own_released"]) / 100.0) for d in pd), reverse=True)
    big = [s for s in sil if s > T / 2]
    put("goff_draws", len(sil), "out/gate/TX-G2XL-rome-T500.json rows.rescope.goff.per_draw", "calibration draws at gate level")
    put("goff_draws_majority_silenced", len(big), "same", "draws in which more than half of the 500 own requests are silenced")
    put("goff_min_silenced_in_those", min(big) if big else None, "same", "fewest own requests silenced among those draws")
    put("goff_max_silenced_in_others", max(s for s in sil if s <= T / 2), "same", "most own requests silenced in the other draws")
    put("goff_silenced_sorted", sil, "same", "own requests silenced per draw, sorted")

    # (3) rows that lie close together on the clean log
    close = ["rescope", "cosine", "whitecos_0.1", "cosine_published"]
    for nm, key in (("par", "edit.paraphrase.exact"), ("done", "heldout.D1_tf"), ("req", "edit.request.exact")):
        xs = [bm(D, "TOne", r, key) for r in close]
        put(f"clean_close_{nm}_min", min(xs), "summary/behave_TX-G2XL-rome-T100.json", f"min of {key} over {close}")
        put(f"clean_close_{nm}_max", max(xs), "summary/behave_TX-G2XL-rome-T100.json", f"max of {key} over {close}")

    xs = [abs(bm(D, "TFive", r, k) - bm(D, "TFive", r + "@goff", k)) for r in COS_GOFF for k in T5KEYS]
    xs += [abs(gl(D, "TFive", r, k) - gl(D, "TFive", r + "@goff", k)) for r in COS_GOFF for k in ("heldout.fired", "neighborhood.fired")]
    put("collapsed_cosine_goff_max_change", max(xs), "summary/behave_, gate_TX-G2XL-rome-T500.json",
        f"largest absolute change of a printed cell of the rows {COS_GOFF} between guards on and guards off")

    # (4) equal evidence: ranges and counts
    for tag in LOGS:
        eq = D[tag]["equal"]
        xs = [eq[f"{k}|n{n}"]["metrics"]["heldout.D1_tf"]["value"] for k in ("comp_min", "rb_lineage") for n in NS]
        put(f"equal_{tag}_linear_done_min", min(xs), f"summary/equal_{LOGS[tag]}.json", "min of held-out D1 over comp_min, rb_lineage, n = 1, 10, 100")
        put(f"equal_{tag}_linear_done_max", max(xs), f"summary/equal_{LOGS[tag]}.json", "max of the same")
        ls = [eq_lost(eq[f"rb_suffix|n{n}"]) for n in NS]
        put(f"equal_{tag}_prefix_lost_min", min(ls), f"summary/equal_{LOGS[tag]}.json", "fewest edits lost by rb_suffix over n (mean of seeds)")
        put(f"equal_{tag}_prefix_lost_max", max(ls), f"summary/equal_{LOGS[tag]}.json", "most edits lost by rb_suffix over n")
        for n in NS:
            rec = eq[f"least_sacrificing|n{n}"]
            tot = sum(m["n_complaints"] for m in rec["meta"])
            fired = rec["metrics"]["complaints.fired"]["value"]
            served = tot - round(tot * fired / 100.0)
            put(f"equal_{tag}_least_n{n}_reported", tot, f"summary/equal_{LOGS[tag]}.json", "reported complaints summed over the seeds")
            put(f"equal_{tag}_least_n{n}_certified", served, f"summary/equal_{LOGS[tag]}.json",
                "reported complaints on which no edit is released as served (teacher-forced pass): total - round(total * complaints.fired / 100)")
            put(f"equal_{tag}_least_n{n}_lost", eq_lost(rec), f"summary/equal_{LOGS[tag]}.json", "edits lost, mean over seeds")
    mid = D["TOne"]["equal"][f"least_sacrificing|n{NS[1]}"]["meta"]
    put("equal_n_mid", mid[0]["n_complaints"], "summary/equal_TX-G2XL-rome-T100.json", "complaints per seed at the middle grid point")
    put("equal_seeds_mid", len(mid), "summary/equal_TX-G2XL-rome-T100.json", "complaint seeds at the middle grid point")
    tot5 = sum(m["n_complaints"] for n in NS for m in D["TFive"]["equal"][f"least_protecting|n{n}"]["meta"])
    ref5 = sum(m["complaints_refused"] for n in NS for m in D["TFive"]["equal"][f"least_protecting|n{n}"]["meta"])
    put("equal_TFive_protecting_reported", tot5, "summary/equal_TX-G2XL-rome-T500.json", "complaints given to the protecting form, all n and seeds")
    put("equal_TFive_protecting_refused", ref5, "summary/equal_TX-G2XL-rome-T500.json", "of them refused")

    # (5) relational workload
    q5 = max(r["queries_exact_pct"]["value"] for r in D["TFive"]["r22"]["rows"])
    put("rel_TFive_q_max", q5, "summary/r22_TX-G2XL-rome-T500.json", "max of queries exact over all rows")
    lin = [r["queries_exact_pct"]["value"] for r in D["TOne"]["r22"]["rows"] if r["label"] in ("comp_min", "rb_lineage", "blocklist")]
    put("rel_TOne_linear_q_max", max(lin), "summary/r22_TX-G2XL-rome-T100.json", "max of queries exact over comp_min, rb_lineage, blocklist, m = 1, 3, 10")
    rs = [r["queries_exact_pct"]["value"] for r in D["TOne"]["r22"]["rows"] if r["label"] == "rb_suffix"]
    put("rel_TOne_prefix_q_max", max(rs), "summary/r22_TX-G2XL-rome-T100.json", "max of queries exact over rb_suffix, m = 1, 3, 10")

    # (6) screening: experiment's file for T100 / T500 (undisputed); verifier's file for T2000 with the tie rule
    for tag, lg in LOGS.items():
        s = D["screen"]["logs"][lg]
        put(f"screen_{tag}_flagged", s["n_flagged"], "out/screening.json", "edits with a non-positive self-separation margin")
        for ref, nm in (("exploding_norm", "expl"), ("first_token_subject", "first")):
            a = s["against"][ref]
            put(f"screen_{tag}_{nm}_ref", len(a.get("reference", [])), "out/screening.json", f"reference set {ref}")
            put(f"screen_{tag}_{nm}_tp", len(a.get("true_positives", [])), "out/screening.json", f"flagged edits in {ref}")
        ties = [e for e in s["flagged"] if abs(next(d["margin"] for d in s["flagged_detail"] if d["edit"] == e)) < 1e-9] if s["n_flagged"] else []
        put(f"screen_{tag}_ties", len(ties), "out/screening.json", "flagged edits with |margin| < 1e-9")
    v = D["vscreen"]
    t = v["ties_T2000"]
    tie = set(t["tie_edits"])
    for e, mm in t["margins"].items():
        assert abs(mm["public_factor_arithmetic"]) < 1e-9, ("tie edit with a margin that is not a tie", e)
    fl = set(v["logs"]["TX-G2XL-rome-T2000"]["flagged"])
    neg = sorted(fl - tie)
    expl, first = set(v["audit_exploding"]), set(v["first_token_subject_by_lookup_index"])
    src = "verify/out/screening_T2000.json"
    put("screen_TTwoThousand_negative", len(neg), src, "verifier's flagged edits without the tie edits (margin < -1e-9)")
    put("screen_TTwoThousand_ties", len(tie), src, "edits with |margin| < 1e-9 (duplicate solver keys)")
    put("screen_TTwoThousand_tie_pairs", len(t["groups_of_identical_solver_keys"]), src, "groups of identical solver keys")
    put("screen_TTwoThousand_expl_ref", len(expl), src, "audit's list of exploding updates")
    put("screen_TTwoThousand_expl_tp", len(expl & set(neg)), src, "exploding updates with a negative margin")
    put("screen_TTwoThousand_first_ref", len(first), src, "requests whose subject is the first prompt token")
    put("screen_TTwoThousand_first_tp", len(first & set(neg)), src, "of them with a negative margin")
    assert not (tie & expl), "a tie edit is an exploding update: the recall would depend on the tie rule"

    # (7) certificate
    n = sum(D[tag]["cert"]["total"]["n_checked_certified"] for tag in LOGS)
    viol = sum(D[tag]["cert"]["total"]["n_violations_certified"] + D[tag]["cert"]["total"]["n_violations_in_band"] for tag in LOGS)
    put("cert_checked_total", n, "out/cert/<log>.json total.n_checked_certified", "sum over the two logs")
    put("cert_violations_total", viol, "out/cert/<log>.json", "violations, certified and in the band, both logs")
    g_on = sorted({k.split("__")[0] for k, r in D["TFive"]["cert"]["rows"].items() if "@goff" not in k}
                  - {k.split("__")[0] for k, r in D["TFive"]["cert"]["rows"].items() if "@goff" not in k and r["n_certified"] > 0})
    put("cert_TFive_rows_certifying_nothing", g_on, "out/cert/TX-G2XL-rome-T500.json", "rows with guards that certify no prompt in any draw")
    return F


def build(root=None, manifest=None, out_path=None, verbose=False):
    root = root or find_root()
    out_path = out_path or os.path.join(root, OUT_REL)
    if not os.path.isdir(os.path.join(root, E10, "out", "summary")):
        raise OSError(f"{E10}/out/summary is not readable")
    D = load(root)
    doc = {"what": "derived facts of experiment E10 for the paper (scripts/gen_tab_e10.py); every value names its source and rule",
           "inputs": {}, "facts": facts(D), "warnings": []}
    for rel in inputs():
        st = os.stat(os.path.join(root, rel))
        doc["inputs"][rel] = {"mtime": int(st.st_mtime), "size": st.st_size}
    texts = {"tab_e10_text.tex": tab_text(D), "tab_e10_maint.tex": tab_maint(D), "tab_e10_equal.tex": tab_equal(D)}
    for fn, src in texts.items():
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
    if not os.path.isdir(os.path.join(root, E10, "out", "summary")):
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
    d, w = build(verbose=True)
    print(f"wrote {OUT_REL} ({len(d['facts'])} facts) and {', '.join(TEXFILES)}")
    for k, v in d["facts"].items():
        print(f"  {k:44s} {v['value']}")
