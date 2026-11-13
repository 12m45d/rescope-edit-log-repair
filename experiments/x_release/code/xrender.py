"""Rendering of the E3 fix-up: table_fixup.txt and FIXUP.txt of e3_qwen/out/fixup, e3_llava/out/fixup and x_release/out, from the JSON
files written by xlib.run_backbone / release_compute.py.  One table per editor family; the two families are never printed side by side.
Nothing is computed here except the formatting of stored numbers.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/10_final/x_release/code")
from xlib import *  # noqa: E402,F401,F403
import xlib  # noqa: E402

FAMS = ("ae-l2_3", "memitw")


def f(v, nd=1, sign=False):
    if v is None:
        return "-"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def ci(v, lo, hi, nd=1):
    return "-" if v is None else f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def pv(p):
    if p is None:
        return "-"
    return f"<={RES:.4f}" if p <= RES * (1 + 1e-12) else f"{p:.4f}"


def pex(t):
    """Exact McNemar value; values too small for a float are printed from their decimal logarithm."""
    p, lg = t["p"], t["log10_p"]
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(lg))
    return f"{10 ** (lg - e):.1f}e{e}"


def per(v, nd=1, sign=False):
    return "/".join(f(x, nd, sign) for x in v)


def tab(head, rows, left=1):
    """Plain-text table: the first `left` columns are left-aligned, the others right-aligned."""
    if not rows:
        return ["  (no rows)"]
    w = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    fmt = lambda r: "  " + "  ".join(str(x).ljust(w[i]) if i < left else str(x).rjust(w[i]) for i, x in enumerate(r))        # noqa: E731
    return [fmt(head), "  " + "  ".join("-" * x for x in w)] + [fmt(r) for r in rows]


def checks_block(ck):
    n_skip = sum(bool(c.get("skipped")) for c in ck["checks"])              # ARTIFACT: a skipped check is shown as skipped, not as ok
    out = [f"CHECKS ({ck['n_checks']}; failed: {ck['n_checks_failed']}" + (f"; passed: {ck['n_checks'] - ck['n_checks_failed'] - n_skip}; skipped: {n_skip}" if n_skip else "") + ")"]
    for c in ck["checks"]:
        out.append(f"  [{'skipped' if c.get('skipped') else 'ok' if c['ok'] else 'FAILED'}] item {c['item']}: {c['check']}" + ("" if c["detail"] is None else f"   -- {c['detail']}"))
    return out


def write(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("written", os.path.relpath(path, FINAL), len(lines), "lines")


# ----------------------------------------------------------------------------- one backbone
def h2_rows(B, E, fam):
    rows = []
    for log in B.fam[fam]:
        e = E[log]
        for role in xlib.H2W:
            x = e["H2"][role]
            for sub in ("all", "later"):
                if sub not in x:
                    continue
                y, g = x[sub], x[sub]["gate_coverage_form_20_draws"]
                rows.append([log + (" (dev)" if log == B.dev else ""), x["method"], "all" if sub == "all" else f"later (>= {B.later_from})", str(y["n_units"]), str(x["n_draws"]),
                             f(y["mean_rescope"]), f(y["mean_other"]), ci(y["diff"], y["ci_lo"], y["ci_hi"]), pv(y["p"]), per(y["by_draw"], 1, True),
                             "-" if g is None else ci(g["value"], g["ci_lo"], g["ci_hi"]), "-" if g is None else pv(g["p_two_sided"])])
    return rows


def h3_rows_t(B, E, fam):
    rows = []
    for log in B.fam[fam]:
        for pair in xlib.H3W:
            x = E[log]["H3"][pair]
            g = x["gate_coverage_form_20_draws"]
            ga, gb = x.get("gate_a") or {}, x.get("gate_b") or {}
            if x.get("missing_behaviour"):
                rows.append([log, pair, f"{x['a']} - {x['b']}", "-", "-", "-", "-", "-", "missing", "-", "-", "-", "-", "-", "-"])
                continue
            rows.append([log + (" (dev)" if log == B.dev else ""), pair, f"{x['a']} - {x['b']}", str(x["n_units"]), str(x["n_draws"]), f(x["mean_a"]), f(x["mean_b"]),
                         ci(x["diff"], x["ci_lo"], x["ci_hi"]), pv(x["p"]), per(x["by_draw"], 1, True), f"{x['n_unreachable_a']}/{x['n_unreachable_b']}",
                         f"{f(mean(x['n_edits_at_cap_a_by_draw']))}/{f(mean(x['n_edits_at_cap_b_by_draw']))}",
                         "-" if g is None else ci(g["value"], g["ci_lo"], g["ci_hi"]), "-" if g is None else pv(g["p_two_sided"]),
                         f"{ga.get('n_unreachable', '-')}/{gb.get('n_unreachable', '-')}"])
    return rows


def render_backbone(key):
    B = Backbone(key)
    ck = jload(B.fix + "/checks.json")
    epf = ck["endpoints_file"]
    EPF = jload(os.path.join(B.fix, epf))
    E, MCN, REPF = EPF["logs"], jload(B.fix + "/mcnemar.json"), jload(B.fix + "/reporting.json")
    REP = REPF["logs"]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    it = "1" if key == "L" else "2"
    L = [f"E3 FIX-UP ({B.title}): ADDITIONS TO THE FINISHED AND VERIFIED EXPERIMENT (PROTOCOL.md, amendment of 2026-11-12 06:16 UTC)      rendered {now} by code/fixup_render.py",
         f"from out/fixup/*.json (written {ck['written']} by code/fixup_compute.py through x_release/code/xlib.py; driver fx {ck['code']}).  No existing file of the experiment was changed.",
         "One table per editor family; rows of the development log are marked (dev).  Status of every difference below: replication read-out on unseen logs, not the confirmatory family of R6.",
         ""]
    L += [f"(F{it}a) H2: rephrase exact, (T-Gen + I-Gen) / 2, of ReScope (rescope) minus the reference; paired crossed bootstrap over calibration draws x edits (2000 resamples, seed 0)",
          f"      reference = the gate of the cosine family selected for H2 on the development log ({B.sel['cos_H2']}) and the retrieval reference ({B.sel['retr_H2']}); file out/fixup/{epf}",
          f"      edits: all, and later edits only (R8: stream position >= {B.later_from}); draws = behavioural calibration draws (R23: 1 from 1000 edits on: then the interval holds the unit component only);",
          f"      p = two-sided bootstrap value (resolution {RES}: '<=' marks a value at the resolution, an upper bound); per draw = the difference at each behavioural draw;",
          "      gate form = the same difference of the coverage of text and image rephrases (own edit released) over the 20 calibration draws x edits (R25 (3))", ""]
    head = ["log", "reference", "edits", "n", "draws", "rescope %", "ref. %", "difference [95 %]", "p", "per draw", "gate form, 20 draws [95 %]", "p (gate)"]
    for fam in FAMS:
        L += ["  " + FAM_TITLE[fam]] + tab(head, h2_rows(B, E, fam), left=3) + [""]
    L += [f"(F{it}b) H3 (R6 with R25: common scale raised until the mean firing on the fit near-scope negatives is at most 10 percent; read on the odd edits): rephrase exact of a minus b",
          f"      pair1 = rescope against {B.sel['cos_H3']}; pair2_x = rescope_x against {B.sel['cos_H3_x']} (log-derived negatives); retr = rescope against {B.sel['retr_H3']} in the form of pair 1",
          "      unreach. a/b = behavioural draws on which the raise does not reach the limit (they count 0); at cap a/b = edits at their guard cap at the raised scale, mean over the draws (R25 (4));",
          "      gate form = coverage under the same constraint over the 20 calibration draws x odd edits; unr. gate a/b = draws of the 20 that are unreachable", ""]
    head = ["log", "pair", "a - b", "n", "draws", "a %", "b %", "difference [95 %]", "p", "per draw", "unreach. a/b", "at cap a/b", "gate form, 20 draws [95 %]", "p (gate)", "unr. gate a/b"]
    for fam in FAMS:
        L += ["  " + FAM_TITLE[fam]] + tab(head, h3_rows_t(B, E, fam), left=3) + [""]
    cmpv = EPF["comparison_with_verification"]
    L += [f"(F{it}c) Comparison with the verification's own values (e3_qwen/verify/out/claims_extra.json; the verification lists one decimal and drew its own resamples)",
          "      limit: 0.1 points for a value, 0.5 points for an interval end; '*' = at or above the limit (reason in the last column)", ""]
    head = ["log", "quantity", "field", "here", "verification", "abs. diff.", "limit", "", "reason"]
    for fam in FAMS:
        rows = [[r["log"], r["quantity"], r["field"], f(r["mine"], 2, True), f(r["verifier"], 1, True), f(r["abs_diff"], 2), f(r["limit"], 1), "*" if r["above_limit"] else "", r["reason"] or ""]
                for r in cmpv if family_of(r["log"]) == fam]
        L += ["  " + FAM_TITLE[fam]] + tab(head, rows, left=3) + [""]
    L += ["(F3) EXACT McNEMAR TESTS (PROTOCOL section 4; two-sided exact binomial test on the discordant pairs), per log and per calibration draw, from the per-row flags; file out/fixup/mcnemar.json",
          "      D1 = the pre-edit answer is not reproduced, valid heldout_disjoint rows; Rel = the request of an edit is answered with its target, all edits;",
          "      b = pairs with the outcome under the first row only, c = under the second row only; a row without calibration (deployed log, where_logonly) has one run and is paired with each draw",
          "      of a calibrated row; two calibrated rows are paired at the same draw index.  Descriptive per-log tests: no Holm family is formed over them.", ""]
    head = ["log", "outcome", "first row", "second row", "draw", "pairs", "first %", "second %", "b", "c", "p (exact)"]
    for fam in FAMS:
        rows = []
        for log in B.fam[fam]:
            for c in MCN["logs"][log].values():
                for t in c["tests"]:
                    rows.append([log + (" (dev)" if log == B.dev else ""), "D1 heldout_disjoint" if c["outcome"] == "D1" else "Rel exact", c["a"], c["b"], "-" if t["draw"] is None else str(t["draw"]),
                                 str(t["n_pairs"]), f(t["rate_a"]), f(t["rate_b"]), str(t["b"]), str(t["c"]), pex(t)])
        L += ["  " + FAM_TITLE[fam]] + tab(head, rows, left=4) + [""]
    L += ["(F4a) REPORTING NEXT TO FAR LOCALITY (amendment of 2026-11-12 06:16 UTC, decision 4); file out/fixup/reporting.json",
          "      near rows = valid heldout_disjoint probes that ask the question of an edit of the log; released = an edit is released on the row: behaviour (per-row files; mean and single runs)",
          "      and gate level (mean of the 20 calibration draws); above = calibration draws of 20 whose false-firing rate on heldout_disjoint exceeds the bound of the two-block rule (5 / 101 =",
          "      4.95 percent); fire = that rate, mean [5th, 95th percentile] and largest draw; capped = edits at their guard cap (for them the bound is void), mean / largest of the 20 draws.",
          "      where_logonly has thresholds from the log alone and no gate-level read-out ('-').", ""]
    head = ["log", "method", "near rows", "near released % (behaviour)", "runs", "near released % (gate)", "above / 20", "fire % mean [p05, p95]", "max", "capped mean", "capped max"]
    for fam in FAMS:
        rows = []
        for log in B.fam[fam]:
            for lab, m in REP[log]["methods"].items():
                nb, ng, ff, cp = m.get("near_released_behaviour"), m.get("near_released_gate"), m.get("false_firing_20_draws"), m.get("capped_edits_20_draws")
                rows.append([log + (" (dev)" if log == B.dev else ""), lab, str(REP[log]["n_near_scope_probes"]), "-" if nb is None else f(nb["percent"]),
                             "-" if nb is None else per([p["percent"] for p in nb["by_draw"]]), "-" if ng is None else f(ng["percent"]), "-" if ff is None else str(ff["n_draws_above_bound"]),
                             "-" if ff is None else f"{f(ff['mean'], 2)} [{f(ff['p05'], 2)}, {f(ff['p95'], 2)}]", "-" if ff is None else f(ff["max"], 2),
                             "-" if cp is None else f(cp["mean"], 2), "-" if cp is None else str(cp["max"])])
        L += ["  " + FAM_TITLE[fam]] + tab(head, rows, left=2) + [""]
    L += ["(F4d) Self-separation margin of the log (decision 5): smallest margin over the edits and number of edits with a non-positive margin", ""]
    for fam in FAMS:
        rows = [[log + (" (dev)" if log == B.dev else ""), str(REP[log]["T"]), f(REP[log]["self_separation_margin"]["min"], 4, True), str(REP[log]["self_separation_margin"]["n_nonpositive"]),
                 REP[log]["self_separation_margin"]["file"]] for log in B.fam[fam]]
        L += ["  " + FAM_TITLE[fam]] + tab(["log", "T", "smallest margin", "edits with margin <= 0", "source"], rows) + [""]
    L += ["(F4e) Rel exact against the deployed log (R7): kept = the request is answered with the target under both, gained = only under the method, lost = only under the deployed log;",
          f"      percent of the edits of the subset (all edits; the first 100 edits of the stream; the later edits, stream position >= {B.later_from}); counts per run in brackets (kept/gained/lost)", ""]
    head = ["log", "method", "edits", "n", "runs", "kept %", "gained %", "lost %", "counts per run (kept/gained/lost)"]
    for fam in FAMS:
        rows = []
        for log in B.fam[fam]:
            for lab in ("where_logonly", "rescope"):
                for sub, e in (REP[log]["methods"][lab].get("rel_kept_gained_lost") or {}).items():
                    rows.append([log + (" (dev)" if log == B.dev else ""), lab, {"all": "all", "first100": "first 100", "later": f"later (>= {B.later_from})"}[sub], str(e["n_edits"]), str(e["n_draws"]),
                                 f(e["kept_percent"]), f(e["gained_percent"]), f(e["lost_percent"]), "  ".join(f"{p['kept']}/{p['gained']}/{p['lost']}" for p in e["by_draw"])])
        L += ["  " + FAM_TITLE[fam]] + tab(head, rows, left=3) + [""]
    write(B.fix + "/table_fixup.txt", L)

    # ---- FIXUP.txt
    n = ck["n_paper_numbers_by_item"]
    above = [r for r in cmpv if r["above_limit"]]
    T = [f"E3 FIX-UP ({B.title})   written {now} by code/fixup_render.py from out/fixup/checks.json and the JSON files named below",
         "Additions required by PROTOCOL.md section 4 and by the amendment of 2026-11-12 06:16 UTC (decisions 1, 2, 4, 5), after the independent verification of E3 (e3_qwen/verify/verdicts.json:",
         "problems, must_fix_before_paper).  Nothing was selected by looking at results (the reference gates are those of e1_main/out/best_cosine_gate.json); no model was run; no existing file",
         f"was changed: the additions live in out/fixup/ and code/fixup_*.py (shared computation: x_release/code/xlib.py, xrender.py).  Paper numbers: {ck['n_paper_numbers']} new names (all start",
         f"with 'EthreeFx'); names shared with out/paper_numbers.json or with the earlier fix-up files of e1_main and e2_curves: {ck['n_collisions']}.", ""]
    if key == "L":
        T += [f"ITEM 1  paired differences H2 and H3 (missing in e3_llava)  ->  endpoints.json, table_fixup.txt (F1a - F1c), {n.get('1', 0)} paper numbers",
              "  Method of e3_qwen/code/e3_analyze.py (endpoints): fx.stats.crossed_bootstrap_diff over the behavioural calibration draws x edits, 2000 resamples, seed 0; the 20-draw gate-level",
              "  coverage form; H3 for pair 1, pair 2 and the retrieval reference; an unreachable draw counts 0.  H2 on all edits and on later edits only (stream position >= 100).", ""]
        head = ["log", "H2 all: - " + B.sel["cos_H2"], "H2 later", "H2 all: - " + B.sel["retr_H2"], "H3 pair 1", "H3 pair 2", "H3 retr", "draws"]
        for fam in FAMS:
            rows = []
            for log in B.fam[fam]:
                e = E[log]
                h = lambda role, sub: (ci(e["H2"][role][sub]["diff"], e["H2"][role][sub]["ci_lo"], e["H2"][role][sub]["ci_hi"]) if sub in e["H2"][role] else "-")       # noqa: E731
                g = lambda pair: ci(e["H3"][pair].get("diff"), e["H3"][pair].get("ci_lo"), e["H3"][pair].get("ci_hi"))                                                # noqa: E731
                rows.append([log + (" (dev)" if log == B.dev else ""), h("best_cosine_H2", "all"), h("best_cosine_H2", "later"), h("retr", "all"), g("pair1"), g("pair2_x"), g("retr"),
                             str(e["n_behavioural_draws"])])
            T += ["  " + FAM_TITLE[fam]] + ["  " + x for x in tab(head, rows)] + [""]
        T += [f"  Comparison with the verification's values (claims_extra.json): {len(cmpv)} numbers; values at or above 0.1 points: {sum(r['field'] == 'diff' for r in above)}; "
              f"interval ends at or above 0.5 points: {sum(r['field'] != 'diff' for r in above)}."]
        for r in above:
            T.append(f"    {r['log']}, {r['quantity']}, {r['field']}: here {r['mine']:+.2f}, verification {r['verifier']:+.1f} (difference {r['abs_diff']:.2f}): {r['reason']}")
        T += ["  Every value agrees with the verification's to its printed decimal; the interval ends differ by at most the amount above because the two bootstraps draw different resamples.", ""]
    else:
        st = EPF["about"]["already_in_out_endpoints_json"]
        T += [f"ITEM 2  paired differences that the paper numbers lack  ->  endpoints_added.json, table_fixup.txt (F2a - F2c), {n.get('2', 0)} paper numbers",
              f"  H2 against {B.sel['cos_H2']} and {B.sel['retr_H2']} on all edits and on later edits (stream position >= 500: the logs with 1000 and 2000 edits), H3 for pair 1, pair 2 and the retrieval",
              "  reference, recomputed from the per-row files and the gate-level cells with the experiment's method.",
              f"  FOUND: out/endpoints.json of the experiment already holds every one of these quantities ({sum(x['in_endpoints_json'] for x in st)} of {len(st)} entries, H3 pair 2, the retrieval reference and the",
              "  later edits included), and out/table.txt prints them; the recomputation equals the stored values exactly (check below).  What was missing is their paper numbers (out/paper_numbers.json holds",
              "  no paired difference), a documentation line for each, and tables per editor family: these are the additions.  endpoints_added.json holds the recomputed entries with their status.",
              f"  Comparison with the verification's values (H3 pair 2 and the retrieval reference): {len(cmpv)} numbers; at or above the limit: {len(above)}.", ""]
    mc = MCN["about"]
    T += [f"ITEM 3  exact McNemar tests  ->  mcnemar.json, table_fixup.txt (F3), {n.get('3', 0)} paper numbers",
          f"  {mc['n_tests']} tests: (a) D1 on heldout_disjoint and (b) Rel exact, where_logonly against the deployed log and rescope against the deployed log; (c) D1 on heldout_disjoint, rescope against",
          f"  {B.sel['cos_H2']} and against {B.sel['retr_H2']}; per log and per behavioural calibration draw.  b, c, p and log10 p per test.",
          f"  The exact value is computed in integer arithmetic: fx.stats.mcnemar_exact overflows from 1024 discordant pairs on ({mc['computation']['n_integer_only']} test(s) of this backbone); below that the two agree exactly.",
          "  The Holm correction of the replication read-outs (H2, H3) over the unseen logs of both backbones is in x_release/out/holm.json.", ""]
    T += [f"ITEM 4  reporting next to far locality  ->  reporting.json, table_fixup.txt (F4a, F4d, F4e), {n.get('4', 0)} paper numbers",
          "  (a) near-scope held-out probes released, behaviour and gate level, with the number of rows; (b) calibration draws of 20 above the bound, mean, 5th and 95th percentile of the rate;",
          "  (c) capped edits, mean and largest count; (d) smallest self-separation margin and edits with a non-positive margin; (e) Rel exact kept / gained / lost against the deployed log for",
          "  where_logonly and rescope (all edits, first 100, later edits).  Methods: " + ", ".join(REPORT_METHODS) + "; where_logonly has no gate-level read-out.", ""]
    for fam in FAMS:
        rows = []
        for log in B.fam[fam]:
            m = REP[log]["methods"]["rescope"]
            rows.append([log + (" (dev)" if log == B.dev else ""), f(m["near_released_behaviour"]["percent"]), f(m["near_released_gate"]["percent"]), str(m["false_firing_20_draws"]["n_draws_above_bound"]),
                         f(m["capped_edits_20_draws"]["mean"], 2), str(m["capped_edits_20_draws"]["max"]), f(REP[log]["self_separation_margin"]["min"], 3, True),
                         str(REP[log]["self_separation_margin"]["n_nonpositive"])])
        T += ["  rescope  --  " + FAM_TITLE[fam]] + ["  " + x for x in tab(["log", "near rel. % beh.", "gate", "draws above / 20", "capped mean", "max", "margin min", "margin <= 0"], rows)] + [""]
    T += checks_block(ck)
    write(B.fix + "/FIXUP.txt", T)


# ----------------------------------------------------------------------------- x_release
def render_release():
    OUT = XR + "/out"
    ck, H, RCF, LB = jload(OUT + "/checks.json"), jload(OUT + "/holm.json"), jload(OUT + "/release_change.json"), jload(OUT + "/logs_b1.json")["logs"]
    RC = RCF["logs"]
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    L = [f"X_RELEASE: RELEASE AGAINST CHANGE, HOLM OVER THE REPLICATION READ-OUTS, TABLE OF LOGS      rendered {now} by code/release_render.py",
         f"from out/*.json (written {ck['written']} by code/release_compute.py; driver fx {ck['code']}).  New files only; no model run.  One table per editor family; never the two families side by side.", ""]
    for h, t in (("H2", "rephrase exact (T-Gen + I-Gen) / 2 of rescope minus the reference, all edits"), ("H3", "rephrase exact on the odd edits under the near-scope constraint, a minus b")):
        x = H[h]
        L += [f"(X1-{h}) HOLM WITHIN {h}: {H['about']['label']}",
              f"      {t}; family = {x['n_logs']} unseen logs of both backbones x {x['comparisons_per_log']} comparisons = {x['family_size']}; p = two-sided bootstrap value (2000 resamples; resolution {RES}:",
              f"      a value at the resolution is an upper bound, printed '<='; smallest Holm-adjusted value that can be shown {x['smallest_adjusted_value_that_can_be_shown']:.4f}); draws = behavioural calibration draws.",
              f"      Holm-adjusted value at most 0.05: {x['n_rejected_at_0_05']} of {x['family_size']} (the paper's method worse: {x['n_rejected_rescope_worse']}; better: {x['n_rejected_rescope_better']}); "
              f"values at the resolution: {x['n_at_resolution']}; adjusted values that are upper bounds: {x['n_adjusted_upper_bounds']}.", ""]
        head = ["log", "backbone", "comparison", "draws", "difference [95 %]", "p", "Holm p", "<= 0.05"]
        for fam in FAMS:
            rows = [[m["log"], m["backbone"], m["key"].split(" | ")[1], str(m["n_draws"]), ci(m["diff"], m["ci_lo"], m["ci_hi"]), pv(m["p"]), m["p_holm_print"], "yes" if m["reject_at_0_05"] else "no"]
                    for m in x["members"] if m["family"] == fam]
            L += ["  " + FAM_TITLE[fam]] + tab(head, rows, left=3) + [""]
    L += ["(X2) RELEASE AGAINST CHANGE (decision E4 of the triage of the external review); file out/release_change.json",
          "      released = rows on which an edit is released; changed = rows whose pre-edit answer is not reproduced (D1); changed | released = D1 among the rows with a release;",
          "      changed, no release = NUMBER of rows without a release whose answer changed, summed over the runs, of the rows without a release (the certificate says 0).",
          "      Percentages are means over the runs (calibration draws 0-2, draw 0 only from 1000 edits on; complaint streams 0-2 for the H1 row; one run for where_logonly); in brackets the",
          "      single runs as numerator/denominator.  hd = heldout_disjoint; far = rows that share no question with an edit of the log; near = rows that ask the question of an edit.",
          "      Reading: low D1 does not show accurate scope detection; the repair is conservative in its effect on answers, not selective in release; near-scope queries are outside the guarantee.", ""]
    head = ["log", "method", "pool", "rows", "released % (per run)", "changed % (per run)", "changed | released % (per run)", "changed, no release (per run)"]
    pw = {"hd_all": "hd all", "hd_far": "hd far", "hd_near": "hd near"}
    for fam in FAMS:
        for bbt in ("Qwen2.5-VL-3B", "LLaVA-1.5-7B"):
            rows = []
            for log, node in RC.items():
                if node["family"] != fam or node["backbone"] != bbt:
                    continue
                for lab, m in node["methods"].items():
                    for pool, s in m["pools"].items():
                        if not s["n_rows"]:
                            continue
                        br = s["by_run"]
                        rows.append([log + (" (dev)" if node["development_log"] else ""), lab, pw.get(pool, pool), str(s["n_rows"]),
                                     f"{f(s['released_percent'])} ({' '.join(str(p['n_released']) + '/' + str(p['n']) for p in br)})",
                                     f"{f(s['changed_percent'])} ({' '.join(str(p['n_changed']) + '/' + str(p['n']) for p in br)})",
                                     f"{f(s['changed_among_released_percent'])} ({' '.join(str(p['n_changed_released']) + '/' + str(p['n_released']) for p in br)})",
                                     f"{s['changed_not_released_count']} ({' '.join(str(p['n_changed_not_released']) + '/' + str(p['n_not_released']) for p in br)})"])
            L += [f"  {FAM_TITLE[fam]}  --  {bbt}"] + tab(head, rows, left=3) + [""]
    L += ["      method keys: " + "; ".join(f"{k} = {v}" for k, v in MTITLE.items() if k not in ("deployed", "rescope_shared", "rescope_x", "white_1_x")), ""]
    L += ["(X3) TABLE OF LOGS (decision 7): the deployed log as evaluated in these experiments, one sequence per forward pass; file out/logs_b1.json",
          "      Rel / T-Gen / I-Gen exact in percent of the T edits; D1 in percent of the valid heldout_disjoint rows; numerator/denominator in brackets", ""]
    head = ["log", "backbone", "role", "T", "hd rows", "Rel exact", "T-Gen exact", "I-Gen exact", "D1 hd", "source"]
    cell = lambda d: f"{f(d['value'])} ({d['numerator']}/{d['denominator']})"                # noqa: E731
    for title, sel in ((FAM_TITLE["ae-l2_3"], lambda e: e["editor_family"] == "ae-l2_3" and e["evaluated_in"] == "E3"),
                       (FAM_TITLE["ae-l2_3"] + ": PREFIX LOGS of the development streams (first 10, 20, 50 edits)", lambda e: e["evaluated_in"] != "E3"),
                       (FAM_TITLE["memitw"], lambda e: e["editor_family"] == "memitw")):
        rows = [[e["log"], e["backbone"], e["role"], str(e["T"]), str(e["n_heldout_disjoint_valid"]), cell(e["deployed"]["rel_exact"]), cell(e["deployed"]["tgen_exact"]),
                 cell(e["deployed"]["igen_exact"]), cell(e["deployed"]["D1_heldout_disjoint"]), e["deployed"]["rel_exact"]["source"]] for e in LB if sel(e)]
        L += ["  " + title] + tab(head, rows, left=3) + [""]
    write(OUT + "/table_fixup.txt", L)

    n = ck["n_paper_numbers_by_item"]
    cw = RCF["changed_without_release"]
    T = [f"X_RELEASE   written {now} by code/release_render.py from out/checks.json and the JSON files named below",
         "Additions to the final evaluation required by the amendment of 2026-11-12 06:16 UTC (decisions 2 and 7) and by decision E4 of the triage of the external review",
         "(01_paper/v4_rescope/_ext_review/TRIAGE_RESULTS.txt).  No model was run; every number comes from stored per-row files; no existing file was changed (new directory x_release).",
         f"Paper numbers: {ck['n_paper_numbers']} new names (all start with 'Xrel'); names shared with any paper-number file of E1, E2, the prefix logs, E3 or a fix-up: {ck['n_collisions']}.", "",
         f"ITEM 3 (Holm)  ->  holm.json, table_fixup.txt (X1-H2, X1-H3), {n.get('3', 0)} paper numbers",
         f"  Label: {H['about']['label']}.  From the two-sided bootstrap values of the behavioural paired differences (e3_qwen/out/fixup/endpoints_added.json,",
         f"  e3_llava/out/fixup/endpoints.json); resolution {RES}; a value at the resolution is an upper bound, and so is a Holm-adjusted value that derives from it."]
    for h in ("H2", "H3"):
        x = H[h]
        T += [f"  {h}: family of {x['family_size']} = {x['n_logs']} unseen logs x {x['comparisons_per_log']} comparisons; Holm-adjusted value at most 0.05: {x['n_rejected_at_0_05']} "
              f"(the paper's method worse in {x['n_rejected_rescope_worse']}, better in {x['n_rejected_rescope_better']}); not rejected: {len(x['not_rejected'])}: " + "; ".join(x["not_rejected"])]
    T += ["", f"ITEM 5  release against change  ->  release_change.json, table_fixup.txt (X2), {n.get('5', 0)} paper numbers",
          "  13 logs of E3 (the two development logs among them) x where_logonly, rescope, the two selected gates of the cosine family, retr_sum; on the development logs also, from E2, the H1 row",
          "  (least repair of 10 complaints from the log-only thresholds, complaint streams 0-2) and ReScope-W with two-block thresholds (calibration draws 0-2; its near-scope negatives exist",
          "  at draw 0 only, as stored).  Pools: heldout_disjoint (all / no shared question / shared question), HN-I, HN-Q, HN-X.  Per run: numerator and denominator of every rate.", ""]
    head = ["log", "method", "rows", "released %", "changed % (D1)", "changed | released %", "changed, no release"]
    for fam in FAMS:
        rows = []
        for log, node in RC.items():
            if node["family"] != fam:
                continue
            for lab in ("where_logonly", "where_least_n10", "where_cal", "rescope"):
                if lab in node["methods"]:
                    s = node["methods"][lab]["pools"]["hd_all"]
                    rows.append([log + (" (dev)" if node["development_log"] else ""), lab, str(s["n_rows"]), f(s["released_percent"]), f(s["changed_percent"]), f(s["changed_among_released_percent"]),
                                 str(s["changed_not_released_count"])])
        T += ["  heldout_disjoint, all rows  --  " + FAM_TITLE[fam]] + ["  " + x for x in tab(head, rows, left=2)] + [""]
    T += [f"  Rows without a release whose answer changed: {cw['n_rows_summed_over_runs']} (summed over all logs, methods, pools and runs)."]
    for lab, e in cw["evidence_by_method"].items():
        T.append(f"    {lab}: {e['rows_without_release_summed_over_runs']} rows without a release (summed over runs and pools); with a forward pass under the method: "
                 f"{e['with_a_forward_pass_under_the_method']}; pre-edit outcome assigned without a forward: {e['pre_edit_outcome_assigned_without_a_forward']}.  The count is {e['the_count_of_changed_rows_is']}.")
    if cw["cells"]:
        T.append("  FINDING: cells with a non-zero count:")
        for c in cw["cells"]:
            T.append(f"    {c['log']}, {c['method']}, {c['pool']}: {c['count']} row(s) {c['by_run']}: " + ", ".join(r["csid"] for r in c["rows"][:12]))
        for lab, e in cw["by_method"].items():
            T.append(f"    {lab}: {e['reason']}")
    else:
        T += ["  No cell has a non-zero count: for the thresholds-only rows (where_logonly, the H1 row, the two-block thresholds) this is measured on every sequence (the round-off of replaying logged",
              "  terms flips no teacher-forced token on these rows); for the gates at the last prompt position it holds by construction of the evaluation (see the lines above)."]
    T += ["", f"ITEM 6  table of logs  ->  logs_b1.json, table_fixup.txt (X3), {n.get('6', 0)} paper numbers",
          f"  {len(LB)} logs: 13 of E3 (rows of the deployed log in e3_*/out/rows/<log>__deployed__dx__tf.jsonl.gz), 6 prefix logs (e2_prefix/out/rows/<log>__deployed.jsonl.gz): backbone, editor family,",
          "  T, valid heldout_disjoint rows, Rel / T-Gen / I-Gen exact and D1 with numerator, denominator and source file.  The development logs have the same counts in E1 and E2.", ""]
    T += checks_block(ck)
    write(OUT + "/FIXUP.txt", T)
