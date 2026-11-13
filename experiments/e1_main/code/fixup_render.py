"""E1 fix-up: renders out/fixup/table_fixup.txt and out/fixup/FIXUP.txt from the JSON files that fixup_compute.py wrote.  Every number
printed is read from those files; nothing is computed or typed here.

    PYTHONDONTWRITEBYTECODE=1 python fixup_render.py
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os
import time

from e1c import *  # noqa: F401,F403

FIX = os.path.join(OUT, "fixup")
T = fx.io.text_table


def f(v, nd=1):
    return "-" if v is None or isinstance(v, str) else f"{v:.{nd}f}"


def ci(r, nd=1, key="value"):
    if r is None or r.get(key) is None:
        return "-"
    if r.get("ci_lo") is None:
        return f(r[key], nd)
    return f"{r[key]:.{nd}f} [{r['ci_lo']:.{nd}f}, {r['ci_hi']:.{nd}f}]"


def pf(p):
    return "1" if p >= 0.9995 else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def draws(v, nd=1):
    return "-" if not v else "{" + "/".join(f(x, nd) for x in v) + "}"


def main():
    MC, RP, E2R, CK = (jload(os.path.join(FIX, n)) for n in ("mcnemar.json", "reporting.json", "rows_from_e2.json", "checks.json"))
    o = []
    P = o.append
    P(f"E1 FIX-UP: ADDITIONS TO THE FINISHED AND VERIFIED EXPERIMENT (PROTOCOL.md, amendment of 2026-11-12 00:48 UTC, decisions 1 and 4)      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/fixup_render.py")
    P(f"from out/fixup/*.json (written {CK['written']} by code/fixup_compute.py; driver fx {CK['code']}).  No existing file of the experiment was changed; no model was run.")
    P("")
    P("(F1) EXACT McNEMAR TESTS (PROTOCOL section 4): ReScope (default) against every other row of the main table, per calibration draw d0 / d1 / d2")
    P(f"     {MC['what']}.")
    P(f"     b = {MC['b']}; c = {MC['c']}; p = exact two-sided binomial test on b + c.")
    P(f"     draws: {MC['draws']}.")
    P(f"     Holm 'all' = {MC['holm']['all_tests']}; Holm 'draw' = {MC['holm']['within_draw']}.")
    P(f"     pooled = {MC['comparisons'][0]['pooled_description_only']['note']}.")
    P(f"     status: {MC['status']}.  Rows from E2: {MC['rows_from_e2']}.")
    P(f"     tests: {MC['n_tests']}; excluded: {len(MC['excluded'])}")
    for log in DEV:
        for oc, title in (("D1", "D1 on heldout_disjoint (outcome = the pre-edit answer is not reproduced; b > c would mean MORE damage under ReScope)"),
                          ("Rel", "Rel exact (outcome = the edit's own request is answered exactly; b > c means MORE edits answered under ReScope)")):
            cs = [c for c in MC["comparisons"] if c["log"] == log and c["outcome"] == oc]
            fam = MC["holm"]["families"][oc]
            P("")
            P(f"  {log}   {title};  pairs per test: {cs[0]['per_draw'][0]['n']};  Holm families: all = {fam['all_tests']['m']} tests, draw = {fam['within_draw']['0']['m']} tests")
            rows = []
            for c in cs:
                d = c["per_draw"]
                rows.append([c["title"] + ("" if c["source"] == "E1" else "  [E2 rows]"), "3" if c["has_calibration_draws"] else "-", "/".join(f(t["rate_rescope"], 1) for t in d),
                             "/".join(f(t["rate_other"], 1) for t in d)] + [f"{t['b']}/{t['c']}  p {pf(t['p'])}  ({pf(t['p_holm_all_tests'])}; {pf(t['p_holm_within_draw'])})" for t in d] +
                            [f"{c['pooled_description_only']['b']}/{c['pooled_description_only']['c']}", pf(c["p_holm_all_tests_max_over_draws"])])
            P("  " + T(["ReScope against", "draws of the row", "rate ReScope d0/d1/d2 %", "rate row d0/d1/d2 %", "d0: b/c  p  (Holm all; draw)", "d1: b/c  p  (Holm all; draw)",
                         "d2: b/c  p  (Holm all; draw)", "pooled b/c", "largest Holm-all p"], rows).replace("\n", "\n  "))
    P("")
    P("(F2a) THE THREE BEHAVIOURAL DRAWS NEXT TO EVERY NEAR-SCOPE CELL OF THE MAIN ROWS (R25 (3)): fire = share of valid near-scope negatives on which an edit is released,")
    P("      D1 = share whose pre-edit answer is not reproduced (%); calibrated rows: mean over draws 0-2 [95 % crossed bootstrap, 3 draws] {d0/d1/d2}; rows without calibration: one run [bootstrap]")
    for log in DEV:
        rows = []
        for r in RP["near_scope_draws"]:
            if r["log"] != log:
                continue
            cells = []
            for k in HN:
                e = r["kinds"][k]
                fr, d1 = e.get("fired"), e.get("D1")
                cells.append(("-" if fr is None else f(fr["mean"]) + (" " + draws(fr["draws"]) if fr["draws"] else "")) + "  /  " +
                             ("-" if d1 is None else ci(d1, 1, "mean") + (" " + draws(d1["draws"]) if d1["draws"] else "")))
            sp = [r["kinds"][k]["D1"]["spread"] for k in HN if r["kinds"][k].get("D1") and r["kinds"][k]["D1"]["spread"] is not None]
            rows.append([r["title"], str(r["n_draws"]) if r["calibrated"] else "-"] + cells + [f(max(sp)) if sp else "-"])
        P("")
        P(f"  {log}")
        P("  " + T(["method", "n draws", "HN-I fire {draws}  /  D1 [95 %] {draws}", "HN-Q fire {draws}  /  D1 [95 %] {draws}", "HN-X fire {draws}  /  D1 [95 %] {draws}", "largest D1 spread"], rows).replace("\n", "\n  "))
    P("")
    P("(F2b) GENERATED ANSWERS, D2 AND WRONG-TARGET RATES OF THE ROWS OF TABLE (3), NOW ALSO PAPER NUMBERS (decoder v2; calibration draw 0 for calibrated rows).  D2 = pre-correct")
    P("      heldout_disjoint rows whose 48-token answer loses the reference: rate [95 % bootstrap] (numerator / denominator).  wrong target = valid near-scope rows whose generated")
    P("      answer equals the target of the released edit: rate [95 % bootstrap] (n rows = denominator); eq pre / eq any = answer equals the pre-edit answer / the target of any edit.")
    for log in DEV:
        rows = []
        for r in RP["generation"]:
            if r["log"] != log:
                continue
            if r.get("missing"):
                rows.append([r["row"], "missing"])
                continue
            ep = r["edit_prompts"]
            d2 = r["D2"]
            rows.append([r["title"], "-" if r["draw"] is None else str(r["draw"])] + [f(ep[m]["eq"]) + " / " + f(ep[m]["contains"]) for m in ("rel", "tgen", "igen")] +
                        [ci(r["probes_eq_pre"], 2), "-" if d2 is None else f"{d2['rate']:.2f} [{d2['ci_lo']:.2f}, {d2['ci_hi']:.2f}] ({d2['n_D2']:.0f} / {d2['n_pre_correct']:.0f})"] +
                        [(ci(r["near_scope"][k]["wrong_target"], 1) + f" (n = {f(r['near_scope'][k]['n_rows'], 0)})") + " ; " + f(r["near_scope"][k]["eq_pre"]) + " / " + f(r["near_scope"][k]["eq_any_edit_all_rows"])
                         for k in HN + ("all",)])
        P("")
        P(f"  {log}")
        P("  " + T(["method", "draw", "Rel eq / contains", "T-Gen eq / contains", "I-Gen eq / contains", "probes eq pre", "D2 rate (n D2 / n pre-correct)", "HN-I wrong target (n) ; eq pre / eq any",
                     "HN-Q wrong target (n) ; eq pre / eq any", "HN-X wrong target (n) ; eq pre / eq any", "all kinds wrong target (n) ; eq pre / eq any"], rows).replace("\n", "\n  "))
    P("")
    P("(F2c) HOLM-ADJUSTED VALUES AT THE RESOLUTION OF THE BOOTSTRAP ARE UPPER BOUNDS")
    for H in ("H2", "H3"):
        h = RP["holm"][H]
        P(f"  {H}: {h['n_tests']} tests; resolution of a bootstrap p = 1 / {h['n_boot']} = {h['resolution_of_the_bootstrap_p']}; smallest Holm-adjusted value that can be shown = "
          f"{h['n_tests']} x {h['resolution_of_the_bootstrap_p']} = {h['smallest_possible_adjusted_p']:.4f}; upper bounds: {h['n_upper_bounds']} of {h['n_tests']}")
        P("  " + T(["log", "comparison", "against", "difference [95 %], points", "bootstrap p", "at resolution", "Holm-adjusted p, as to be printed"],
                   [[SHORT[t["log"]], t["comparison"], t["method"], f"{t['diff']:+.1f} [{t['ci_lo']:+.1f}, {t['ci_hi']:+.1f}]", ("<= " if t["p_at_resolution"] else "") + f"{t['p']:.4f}",
                     "yes" if t["p_at_resolution"] else "no", t["p_holm_print"]] for t in h["tests"]]).replace("\n", "\n  "))
        P(f"  rule: {h['rule']}.")
        P("")
    hf = RP["H2_family"]
    P(f"(F2d) H2 HAS {hf['n_comparisons']} COMPARISONS ({hf['outcome']}):")
    for log, lst in hf["per_log"].items():
        P(f"  {log}: {len(lst)}: " + ", ".join(lst))
    P(f"  {hf['not_in_the_family']}.")
    P("")
    P("(F3) THE ROWS THAT THE MAIN TABLE TAKES FROM E2, RENDERED AGAIN FROM THE FINAL E2 RECORDS")
    P(f"     {E2R['what']}; read {E2R['read_at']} (E2 results written {E2R['e2_results_written']}; E1's table had read them at {E2R['e2_rows_read_by_e1_at']}).")
    P(f"     cells: {E2R['cells']}.  A linear repair has no per-edit release: 'fire' is '-'.")
    c = E2R["checks"]
    P(f"     CHECKS: paper numbers of E2 compared: {c['e2_paper_numbers']['n_compared']}, mismatches: {len(c['e2_paper_numbers']['mismatches'])};  records behind E1's table (2b) compared: "
      f"{c['e1_earlier_records']['n_compared']}, mismatches: {len(c['e1_earlier_records']['mismatches'])};  block (2b) of out/table.txt found verbatim when rendered again: "
      + ", ".join(f"{SHORT[k]} {'yes' if v else 'NO'}" for k, v in c["table_2b_block_found_verbatim"].items()) + f".  {E2R['mismatch_rule']}.")
    for m in c["e2_paper_numbers"]["mismatches"] + c["e1_earlier_records"]["mismatches"]:
        P(f"     MISMATCH: {m}")
    P(f"     KEYS OF E2's results.json WITH TWO DIFFERENT RECORDS: {c['keys_with_two_different_records']} (near-scope firing of part c' at gate level: the mean over the 20 gate-level draws and the mean over")
    P("     draws 0-2 are stored under one key; both are printed below, labelled; nothing is chosen here):")
    for d in E2R["duplicate_keys"]:
        P(f"       {SHORT[d['key'][0]]} {d['key'][7]}: " + " | ".join(f"{r['value']:.2f}" + (f" [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] (20 draws)" if r.get("n_units") == 20 else f" (draws 0-2: {draws(r.get('per_draw'), 2)})") for r in d["records"]))
    for log in DEV:
        rows = []
        for r in E2R["rows"]:
            if r["log"] != log:
                continue
            C = r["cells"]
            g = lambda k: C.get(k)                                                     # noqa: E731
            v = lambda k, nd=1: "-" if g(k) is None or g(k).get("value") is None else f(g(k)["value"], nd)     # noqa: E731
            pd = lambda k, nd=2: "" if g(k) is None or not g(k).get("per_draw") or len(g(k)["per_draw"]) < 2 else " " + draws(g(k)["per_draw"], nd)    # noqa: E731
            if r["part"] == "c'":
                hn = " / ".join(v(f"{k} fire (gate, 20 draws)") + "|" + v(f"{k} fire (gate, draws 0-2)") for k in HN)
                fire = v("fire hd", 2) + f"; gate {ci(g('fire hd (gate, 20 draws)'), 2)}; capped {v('capped', 2)}"
            else:
                hn = " / ".join(v(f"{k} fire (gate)") for k in HN) if r["part"] == "b" else "-"
                fire = ci(g("fire hd"), 2) if r["part"] == "b" else "-"
            rows.append([f"({r['part']}) {r['row']}", r["n_evidence"], str(r["n_streams"] or "-"), ci(g("Rel")), ci(g("T-Gen")), ci(g("I-Gen")), ci(g("rephrase")),
                         "/".join(v(f"kg rel {x}", 0) for x in ("kept", "gained", "lost")), "/".join(v(f"kg tgen {x}", 0) for x in ("kept", "gained")) + " ; " + "/".join(v(f"kg igen {x}", 0) for x in ("kept", "gained")),
                         fire, ci(g("D1 hd"), 2) + pd("D1 hd"), v("D1 strict", 2), v("T-Loc*", 2) + "/" + v("I-Loc*", 2), hn, " / ".join(v(f"{k} D1") for k in HN),
                         ci(g("D2"), 2) + f" (n = {v('D2 n pre-correct', 0)})", "/".join(v(f"gen {m} eq", 0) for m in ("rel", "tgen", "igen")) if r["part"] == "d" else "-"])
        P("")
        P(f"  {log}")
        P("  " + T(["row (part of E2's table)", "n evidence", "streams / draws", "Rel", "T-Gen", "I-Gen", "(T-Gen + I-Gen) / 2", "Rel kept/gained/lost", "T-Gen ; I-Gen kept/gained", "fire hd %",
                     "D1 hd % {streams / draws}", "D1 strict", "T-Loc*/I-Loc*", "NEAR fire HN-I / HN-Q / HN-X (c': 20 draws|draws 0-2)", "NEAR D1 HN-I / HN-Q / HN-X", "D2 rate of pre-correct",
                     "gen eq Rel/T-Gen/I-Gen"], rows).replace("\n", "\n  "))
    with open(os.path.join(FIX, "table_fixup.txt.tmp"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    os.replace(os.path.join(FIX, "table_fixup.txt.tmp"), os.path.join(FIX, "table_fixup.txt"))
    # ---------------------------------------------------------------- FIXUP.txt
    q = []
    Q = q.append
    Q(f"E1 FIX-UP   written {time.strftime('%Y-%m-%d %H:%M:%S')} by code/fixup_render.py from out/fixup/checks.json and the JSON files named below")
    Q("Additions required by PROTOCOL.md section 4 and by the amendment of 2026-11-12 00:48 UTC (decisions 1 and 4), after the independent verification (verify/verdicts.json,")
    Q("protocol problems 1, 3, 4, 6).  Nothing was selected by looking at results; no model was run; no existing file was changed: the additions live in out/fixup/ and code/fixup_*.py.")
    Q(f"Paper numbers: {CK['n_paper_numbers']} new names (all start with 'EoneFx'); names shared with out/paper_numbers.json: {CK['n_collisions_with_paper_numbers_json']}.")
    Q("")
    bi = CK["n_paper_numbers_by_item"]
    fam = MC["holm"]["families"]
    Q(f"ITEM 1  exact McNemar tests  ->  mcnemar.json, table_fixup.txt (F1), {bi['1']} paper numbers")
    Q(f"  {MC['n_tests']} tests: ReScope against each of the other rows of the main table (rows of E1 and the three rows of table (2b) taken from E2), outcomes D1 on heldout_disjoint and Rel exact,")
    Q(f"  calibration draws 0-2, from the stored per-row flags.  b, c and p per draw; the table pooled over the draws is a description only.  Holm, two ways of counting the draws:")
    Q(f"  'all tests' ({fam['D1']['all_tests']['m']} per outcome: backbones x rows x draws) and 'within draw' ({fam['D1']['within_draw']['0']['m']} per outcome and draw: backbones x rows).")
    nrej = {oc: sum(c["all_draws_reject_0.05_after_holm_all_tests"] for c in MC["comparisons"] if c["outcome"] == oc) for oc in ("D1", "Rel")}
    ntot = {oc: sum(1 for c in MC["comparisons"] if c["outcome"] == oc) for oc in ("D1", "Rel")}
    Q(f"  Comparisons whose three draws all stay below 0.05 after Holm (all tests): D1 {nrej['D1']} of {ntot['D1']}; Rel exact {nrej['Rel']} of {ntot['Rel']} (the table lists every one).")
    Q("")
    Q(f"ITEM 2  reporting  ->  reporting.json, table_fixup.txt (F2a - F2d), {bi['2']} paper numbers")
    Q("  (a) near-scope cells: fire and D1 of HN-I / HN-Q / HN-X at each behavioural draw, the spread, the mean firing per kind;")
    Q("  (b) rows of table (3): generation metrics, D2 rate with numerator and denominator, wrong-target rate per kind with its denominator;")
    for H in ("H2", "H3"):
        h = RP["holm"][H]
        Q(f"  (c) {H}: {h['n_upper_bounds']} of {h['n_tests']} Holm-adjusted values are upper bounds (bootstrap p at its resolution 1 / {h['n_boot']} = {h['resolution_of_the_bootstrap_p']}; "
          f"smallest value that can be shown {h['smallest_possible_adjusted_p']:.4f}); stored with '<=' (names ...PHolmPrint, ...PHolmBound, ...PHolmIsBound);")
    Q(f"  (d) H2 has {hf['n_comparisons']} comparisons: " + "; ".join(f"{SHORT[k]}: " + ", ".join(v) for k, v in hf["per_log"].items()) + ".")
    Q("")
    Q(f"ITEM 3  rows taken from E2  ->  rows_from_e2.json, table_fixup.txt (F3), {bi['3']} paper numbers (generated answers of COMP-B and REEDIT-B at all complaints, R19 a)")
    Q("  parts b (least repair from the log-only thresholds, n = 10 and ALL), c' (two-block thresholds) and d (linear rows) of E2's table, from the final records of e2_curves/out/results.json.")
    Q(f"  Agreement with E2's paper numbers: {c['e2_paper_numbers']['n_compared']} compared, {len(c['e2_paper_numbers']['mismatches'])} differ.  Agreement with the records behind E1's table (2b): "
      f"{c['e1_earlier_records']['n_compared']} compared, {len(c['e1_earlier_records']['mismatches'])} differ; the block (2b) rendered again is found verbatim in out/table.txt: "
      + ", ".join(f"{SHORT[k]} {'yes' if v else 'NO'}" for k, v in c["table_2b_block_found_verbatim"].items()) + ".")
    Q(f"  Found and reported, not repaired: {c['keys_with_two_different_records']} keys of e2_curves/out/results.json hold two different records (near-scope firing of c' at gate level: 20 draws and draws 0-2).")
    Q("")
    Q(f"CHECKS ({CK['n_checks']}; failed: {CK['n_checks_failed']})")
    for x in CK["checks"]:
        Q(f"  [{'ok' if x['ok'] else 'FAILED'}] item {x['item']}: {x['check']}" + ("" if x["detail"] is None else f"   -- {str(x['detail'])[:600]}"))
    with open(os.path.join(FIX, "FIXUP.txt.tmp"), "w") as fh:
        fh.write("\n".join(q) + "\n")
    os.replace(os.path.join(FIX, "FIXUP.txt.tmp"), os.path.join(FIX, "FIXUP.txt"))
    print("table_fixup.txt:", len(o), "lines; FIXUP.txt:", len(q), "lines")


if __name__ == "__main__":
    main()
