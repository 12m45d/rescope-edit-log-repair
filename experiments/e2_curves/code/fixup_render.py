"""E2 fix-up: renders out/fixup/table_fixup.txt and out/fixup/FIXUP.txt from the JSON files that fixup_cw200.py and fixup_compute.py
wrote.  Every number printed is read from those files; nothing is computed or typed here.

    PYTHONDONTWRITEBYTECODE=1 python fixup_render.py
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402

FIX = os.path.join(e2c.OUT, "fixup")
T = fx.io.text_table
HN = ("HN-I", "HN-Q", "HN-X")
POST = "POST HOC read-out"


def f(v, nd=2):
    return "-" if v is None or isinstance(v, str) else f"{v:.{nd}f}"


def ci(r, nd=2):
    if r is None or r.get("value") is None:
        return "-"
    return f(r["value"], nd) + ("" if r.get("ci_lo") is None else f" [{r['ci_lo']:.{nd}f}, {r['ci_hi']:.{nd}f}]")


def per(r, nd=2):
    return "" if r is None or not r.get("per_draw") or len(r["per_draw"]) < 2 else " {" + "/".join(f(x, nd) for x in r["per_draw"]) + "}"


def yn(b):
    return "yes" if b else "NO"


def main():
    CW, H1, RP, EX, CK = (fx.io.jload(os.path.join(FIX, n)) for n in ("cw_check_200.json", "h1_registered.json", "reporting.json", "running_example_flag.json", "checks.json"))
    S = e2c.SHORT
    o = []
    P = o.append
    P(f"E2 FIX-UP: ADDITIONS TO THE FINISHED AND VERIFIED EXPERIMENT (PROTOCOL.md, amendments of 2026-11-12 01:00 and 02:55 UTC)      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/fixup_render.py")
    P(f"from out/fixup/*.json (written {CK['written']} by code/fixup_compute.py; cw_check_200.json written {CW['written']} by code/fixup_cw200.py; driver fx {CK['code']}).")
    P("No existing file of the experiment was changed; no model was run.")
    P("")
    P(f"(F4) {POST.upper()} OF PART c' (ReScope-W with two-block thresholds): false firing of the literal rule without guards, alpha 0.05, n 200, gate level from the captured vectors;")
    P(f"     mean of the per-draw rates over 200 seeded calibration draws (seeds 0 to 199) against the bound.  {CW['label']}.")
    P("     INTERVAL: 95 % percentile bootstrap over QUERY CLUSTERS with the draws nested, 2000 resamples: the clusters of the pool are resampled once per resample and the same resampled")
    P("     clusters (with all their rows) are used under every one of the 200 draws; the per-draw thresholds are not recomputed and the draws are not resampled.  'crossed' = sensitivity")
    P("     in which the 200 draws are resampled as well (not the reported interval).  'fresh REPAIR' = valid REPAIR rows whose query is not a calibration query of the draw.")
    rows = []
    for name, c in CW["logs"].items():
        for key, lab in (("fresh_repair", f"fresh REPAIR rows ({c['n_fresh_repair_rows_min_max'][0]}-{c['n_fresh_repair_rows_min_max'][1]} rows per draw; {c['n_repair_query_clusters']} clusters)"),
                         ("heldout_disjoint", f"heldout_disjoint ({c['n_heldout_disjoint_rows']} rows; {c['n_heldout_disjoint_query_clusters']} clusters)")):
            e, r = c[key], c["reproduction_of_the_stored_20_draws"]
            rows.append([S[name], lab, f(r["stored_mean_fresh_repair" if key == "fresh_repair" else "stored_mean_heldout_disjoint"]),
                         f(r["mean_first_20_fresh_repair" if key == "fresh_repair" else "mean_first_20_heldout_disjoint"]), f"{e['mean_over_draws']:.2f} [{e['ci_lo']:.2f}, {e['ci_hi']:.2f}]",
                         f(e["bound"]), "contains the bound" if e["interval_contains_the_bound"] else ("entirely ABOVE the bound" if e["interval_entirely_above_the_bound"] else "entirely below the bound"),
                         f"{e['mean_minus_bound']:+.2f}", f"[{e['crossed_sensitivity']['ci_lo']:.2f}, {e['crossed_sensitivity']['ci_hi']:.2f}]",
                         f"{e['over_draws']['p05']:.2f} / {e['over_draws']['p95']:.2f}", f(100.0 * e["over_draws"]["share_of_draws_above_the_bound"], 1)])
    P("     " + T(["log", "pool (" + POST + ")", "stored mean, 20 draws", "first 20 of the 200", "mean over 200 draws [95 %]", "bound %", "interval", "mean - bound", "crossed [95 %]",
                    "5th / 95th pct over draws", "% of draws above the bound"], rows).replace("\n", "\n     "))
    P("     the first 20 draws reproduce the stored draws of out/parts/cw_check_<log>.json: " + ", ".join(f"{S[n]} {yn(c['reproduction_of_the_stored_20_draws']['first_20_draws_reproduce_the_stored_draws'])}" for n, c in CW["logs"].items()))
    P("")
    P("(F5) H1 AS REGISTERED (R6), BOTH READINGS OF 'WITHIN 2 POINTS'")
    P(f"     rule ({H1['rule_as_registered']['source']}):")
    P(f"       \"{H1['rule_as_registered']['quote']}\"")
    P(f"     two-sided reading: {H1['readings']['two_sided']};  one-sided reading: {H1['readings']['one_sided']}.")
    P(f"     {H1['readings']['status']}.")
    rows, rows2, rows3 = [], [], []
    for name, v in H1["development_logs"].items():
        rp, dp = v["repair"], v["deployed_log"]
        rows.append([S[name], f"{rp['D1_heldout_disjoint']['mean']:.2f} [{rp['D1_heldout_disjoint']['ci_lo']:.2f}, {rp['D1_heldout_disjoint']['ci_hi']:.2f}] " + "{" + "/".join(f(x) for x in rp["D1_heldout_disjoint"]["per_stream"]) + "}",
                     f(dp["D1_heldout_disjoint"]), f(rp["rel_exact"]["mean"], 1) + " {" + "/".join(f(x, 1) for x in rp["rel_exact"]["per_stream"]) + "}", f(dp["rel_exact"], 1),
                     f"{v['rel_exact_repair_minus_deployed_points']:+.1f}", yn(v["D1_at_most_10_percent"]), "met" if v["two_sided_reading"]["met_on_this_log"] else "NOT met",
                     "met" if v["one_sided_reading"]["met_on_this_log"] else "NOT met"])
        k, c, e = v["rel_edits_against_the_deployed_log"], v["edit_checks_of_300_against_the_deployed_log"], v["edits_still_released"]
        rows2.append([S[name], " / ".join(f(k[x]["value"], 1) for x in ("kept", "gained", "lost")), " ; ".join("/".join(f(y, 0) for y in k[x]["per_draw"]) for x in ("kept", "gained", "lost")),
                      f(c["lost"]["value"], 1) + " / " + f(c["gained"]["value"], 1) + f" of {f(c['n_checks']['value'], 0)}", ci(e["heldout_disjoint_percent"]) + per(e["heldout_disjoint_percent"])] +
                     [f(e["near_scope_percent"][t]["value"], 1) + per(e["near_scope_percent"][t], 1) for t in HN])
        for ck, cc in v["differences_to_linear_repairs"].items():
            rows3.append([S[name], ck, cc["linear_repair"], f(cc["d1_repair"]), f(cc["d1_linear"]) + " {" + "/".join(f(x) for x in cc["d1_linear_per_stream"]) + "}",
                          f"{cc['difference_points']:.2f} [{cc['ci_lo']:.2f}, {cc['ci_hi']:.2f}]", ("<= " if cc["p_at_resolution"] else "") + f"{cc['p_two_sided']:.4f}", cc["p_holm_print"], f(cc["rel_linear"], 1)])
        if v["why_the_readings_differ"]:
            P(f"     {S[name]}: the readings differ because {v['why_the_readings_differ']}.")
    P("     " + T(["log", "D1 % repair [95 %] {streams 0/1/2}", "D1 % deployed", "Rel exact repair {streams}", "Rel exact deployed", "Rel repair - deployed", "D1 <= 10 %", "two-sided reading", "one-sided reading"], rows).replace("\n", "\n     "))
    P("     the repaired log against the deployed log (means over the three complaint streams) and the rows on which it still releases an edit:")
    P("     " + T(["log", "Rel kept / gained / lost (edits of 100)", "per stream kept ; gained ; lost", "edit checks lost / gained", "fire % heldout_disjoint [95 %] {streams}", "NEAR fire % HN-I {streams}",
                    "NEAR fire % HN-Q {streams}", "NEAR fire % HN-X {streams}"], rows2).replace("\n", "\n     "))
    P(f"     differences to the linear repairs of the same complaints (D1 on heldout_disjoint, repair minus linear; {H1['holm']['family']}; resolution of a bootstrap p "
      f"{H1['holm']['resolution_of_the_bootstrap_p']}, smallest Holm value {H1['holm']['smallest_possible_adjusted_p']:.3f}; '<=' = upper bound):")
    P("     " + T(["log", "choice of the best linear repair", "linear repair", "D1 % repair", "D1 % linear {streams}", "difference [95 %]", "bootstrap p", "Holm-adjusted p", "Rel exact linear"], rows3).replace("\n", "\n     "))
    dp, pp = H1["development_part"], H1["prefix_part"]
    P(f"     DEVELOPMENT LOGS: two-sided reading met on both: {yn(dp['two_sided_reading_met_on_both_logs'])}; one-sided reading met on both: {yn(dp['one_sided_reading_met_on_both_logs'])}")
    P(f"     {pp['label'].upper()} -- copied from {pp['copied_from']}; file modified {pp['source']['modified']}, read {pp['source']['read_at']}; "
      f"{pp['n_complete']} of {pp['n_prefix_logs']} prefix logs complete at n = 10" + (f"; incomplete: {', '.join(pp['incomplete'])}" if pp["incomplete"] else ""))
    rows = []
    for k, e in pp["prefix_logs"].items():
        if e["complete"]:
            rows.append([k, e["T"], "complete", ci({"value": e["D1_heldout_disjoint"]["mean"], "ci_lo": e["D1_heldout_disjoint"]["ci_lo"], "ci_hi": e["D1_heldout_disjoint"]["ci_hi"]}), f(e["deployed"]["hd_D1"]),
                         f(e["rel_exact"]["mean"], 1), f(e["deployed"]["rel"], 1), f"{e['rel_exact_repair_minus_deployed_points']:+.1f}", "met" if e["two_sided_reading_met"] else "NOT met",
                         "met" if e["one_sided_reading_met"] else "NOT met"])
        else:
            rows.append([k, e["T"], f"INCOMPLETE ({e['status']})", "-", f(e["deployed"]["hd_D1"]) if e.get("deployed") else "-", "-", f(e["deployed"]["rel"], 1) if e.get("deployed") else "-", "-", "-", "-"])
    P("     " + T(["prefix log", "edits", "cell at n = 10", "D1 % repair [95 %]", "D1 % deployed", "Rel exact repair", "Rel exact deployed", "Rel repair - deployed", "two-sided reading", "one-sided reading"], rows).replace("\n", "\n     "))
    ro = H1["rule_outcome"]
    P(f"     RULE ({ro['parts']}): two-sided reading: {ro['two_sided_reading']};  one-sided reading: {ro['one_sided_reading']}")
    P("")
    P("(F6a) EDITS STILL RELEASED AFTER A LEAST REPAIR (low D1 of a repair that starts from the deployed log is a measurement, not a certificate): share of heldout_disjoint rows on which")
    P("      an edit is still released (hook; crossed bootstrap) and near-scope firing per kind (gate level); means over the complaint streams {streams}")
    rows = [[S[e["log"]], e["row"], str(e["n_streams"]), ci(e["heldout_disjoint_D1_percent"]), ci(e["heldout_disjoint_fired_percent"]) + per(e["heldout_disjoint_fired_percent"])] +
            [f(e["near_scope_fired_percent"][t]["value"], 1) + per(e["near_scope_fired_percent"][t], 1) for t in HN] for e in RP["edits_still_released"]]
    P("      " + T(["log", "row", "streams", "D1 % heldout_disjoint [95 %]", "fire % heldout_disjoint [95 %] {streams}", "NEAR fire % HN-I", "NEAR fire % HN-Q", "NEAR fire % HN-X"], rows).replace("\n", "\n      "))
    P("")
    P("(F6b) FALSE FIRING OF THE LABEL-FREE MIGRATION (ReScope: last prompt position, top-1, rebased values) ON THE OUT-OF-BENCHMARK WORKLOAD MME-T, BY n BENIGN QUERIES; gate level;")
    P("      mean [5th, 95th percentile over the draws] (draws); '*' = the mean exceeds the level of the two-block rule (5 %)")
    for name, m in RP["mme_t"].items():
        P(f"      {name}: keys: {m['keys']}; cell field 'which' of MME-T: {m['rows'][0]['key_field_of_the_cell']}; rows: {m['rows'][0]['n_rows']}; mean above the level at n = "
          + (", ".join(m["n_labels_with_mean_above_the_level"]) or "none"))
    names = list(RP["mme_t"])
    by = {n: {r["n_label"]: r for r in RP["mme_t"][n]["rows"]} for n in names}
    labels = [r["n_label"] for r in RP["mme_t"][names[0]]["rows"]]
    rows = []
    for nl in labels:
        row = [nl, by[names[0]][nl]["rule"]]
        for n in names:
            r = by[n].get(nl)
            row.append("-" if r is None else f"{r['mean']:.2f} [{r['p05']:.2f}, {r['p95']:.2f}] ({r['n_draws']})" + ("*" if r["mean_above_the_level"] else ""))
            row.append("-" if r is None or r["share_of_draws_above_the_level"] is None else f(100.0 * r["share_of_draws_above_the_level"], 0))
        rows.append(row)
    P("      " + T(["n", "rule"] + [x for n in names for x in (f"{S[n]}: MME-T fire % [5th, 95th] (draws)", f"{S[n]}: % of draws above 5 %")], rows).replace("\n", "\n      "))
    P(f"      {RP['mme_t'][names[0]]['note']}.")
    P("")
    P("(F6c) CAPPED EDITS OF PART c' WITH TWO DECIMALS (mean over the 20 gate-level draws [5th, 95th percentile]); table.txt of E2 prints one decimal")
    P("      " + T(["log", "capped edits, two decimals", "5th / 95th percentile", "draws", "as printed with one decimal"],
                   [[S[n], c["print_two_decimals"], f"{c['p05']:.2f} / {c['p95']:.2f}", c["n_draws"], c["print_one_decimal_as_in_table_txt"]] for n, c in RP["capped"].items()]).replace("\n", "\n      "))
    P("")
    P(f"(F6d) RUNNING EXAMPLE ({EX['log']}): THE COMPLAINT'S EDIT AND THE WHY-NOT REPHRASE BELONG TO TWO DIFFERENT EDITS: {yn(EX['two_different_edits'])}")
    a, b = EX["complaint_edit"], EX["whynot_edit"]
    P("      " + T(["role", "stream position (0-based)", "counted from 1", "edit: question -> target", "how it enters the example"],
                   [["edit of the complaint", a["stream_position_0_based"], a["stream_position_1_based"], f"{a['question']} -> {a['target']}", "chosen by: " + a["chosen_by"]],
                    ["edit of the why-not rephrase", b["stream_position_0_based"], b["stream_position_1_based"], f"{b['question']} -> {b['target']}",
                     f"rephrase '{b['rephrase']}': own coordinate {b['coordinate']:.3f} below its threshold {b['threshold']:.3f}"]]).replace("\n", "\n      "))
    P(f"      {EX['stream_position']}.  {EX['for_the_text']}.")
    with open(os.path.join(FIX, "table_fixup.txt.tmp"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    os.replace(os.path.join(FIX, "table_fixup.txt.tmp"), os.path.join(FIX, "table_fixup.txt"))
    # ---------------------------------------------------------------- FIXUP.txt
    q = []
    Q = q.append
    bi = CK["n_paper_numbers_by_item"]
    Q(f"E2 FIX-UP   written {time.strftime('%Y-%m-%d %H:%M:%S')} by code/fixup_render.py from out/fixup/checks.json and the JSON files named below")
    Q("Additions required by the amendments of 2026-11-12 01:00 and 02:55 UTC of PROTOCOL.md, after the independent verification (verify/verdicts.json: 'unfavourable items in the files")
    Q("missing from the summary' and the protocol and code notes).  Nothing was selected by looking at results; no model was run; no existing file was changed: the additions live in")
    Q("out/fixup/ and code/fixup_*.py.")
    Q(f"Paper numbers: {CK['n_paper_numbers']} new names (all start with 'EtwoFx'); names shared with out/paper_numbers.json: {CK['n_collisions_with_paper_numbers_json']}.")
    Q("")
    Q(f"ITEM 4  part c', {POST}, 200 seeded draws  ->  cw_check_200.json, table_fixup.txt (F4), {bi['4']} paper numbers")
    Q("  Mean false firing of the literal two-block rule over the draws with seeds 0 to 199, on the REPAIR rows outside the calibration draw and on heldout_disjoint; 95 % interval by a")
    Q("  bootstrap over query clusters with the draws nested (the same resampled clusters under every draw; thresholds of a draw not recomputed; draws not resampled; 2000 resamples).")
    for name, c in CW["logs"].items():
        e, h = c["fresh_repair"], c["heldout_disjoint"]
        Q(f"  {S[name]}: REPAIR rows outside the draw {e['mean_over_draws']:.2f} [{e['ci_lo']:.2f}, {e['ci_hi']:.2f}] against the bound {e['bound']:.2f}: the interval "
          f"{'CONTAINS' if e['interval_contains_the_bound'] else 'DOES NOT CONTAIN'} the bound (stored mean over 20 draws {c['reproduction_of_the_stored_20_draws']['stored_mean_fresh_repair']:.2f}); "
          f"heldout_disjoint {h['mean_over_draws']:.2f} [{h['ci_lo']:.2f}, {h['ci_hi']:.2f}]: the interval {'CONTAINS' if h['interval_contains_the_bound'] else 'DOES NOT CONTAIN'} the bound.")
    Q("  Every file and line that quotes this read-out carries the label '" + POST + "'.")
    Q("")
    Q(f"ITEM 5  H1 record  ->  h1_registered.json, table_fixup.txt (F5), {bi['5']} paper numbers")
    Q("  The rule of R6 quoted from PROTOCOL.md; its outcome under the two-sided and the one-sided reading of 'within 2 points' per development log, with D1 and Rel exact of the repair")
    Q("  and of the deployed log (mean and per complaint stream), Rel kept / gained / lost, edit checks lost / gained of 300, the rows on which the repaired log still releases an edit,")
    Q("  and the two differences to linear repairs with intervals and Holm-adjusted values.  The record makes no statement about the time at which a reading entered the code.")
    for name, v in H1["development_logs"].items():
        Q(f"  {S[name]}: D1 {v['repair']['D1_heldout_disjoint']['mean']:.2f} % (deployed {v['deployed_log']['D1_heldout_disjoint']:.2f}); Rel exact {v['repair']['rel_exact']['mean']:.1f} against "
          f"{v['deployed_log']['rel_exact']:.1f} ({v['rel_exact_repair_minus_deployed_points']:+.1f}): two-sided {'met' if v['two_sided_reading']['met_on_this_log'] else 'NOT met'}, "
          f"one-sided {'met' if v['one_sided_reading']['met_on_this_log'] else 'NOT met'}.")
    Q(f"  {pp['label']}: h1_prefix.json of {pp['source']['modified']}: {pp['n_complete']} of {pp['n_prefix_logs']} prefix logs complete; two-sided met on {len(pp['two_sided_reading_met_on'])}, "
      f"one-sided met on {len(pp['one_sided_reading_met_on'])}" + (f"; incomplete: {', '.join(pp['incomplete'])}" if pp["incomplete"] else "") + ".")
    Q(f"  Rule: two-sided reading: {ro['two_sided_reading']}; one-sided reading: {ro['one_sided_reading']}.")
    Q("")
    Q(f"ITEM 6  reporting items of the verification  ->  reporting.json, running_example_flag.json, table_fixup.txt (F6a - F6d), {bi['6']} paper numbers")
    Q("  (a) H1 row and the least repair from the deployed start at n = 10 and n = 100: share of heldout_disjoint rows on which an edit is still released, near-scope firing per kind;")
    Q("  (b) false firing of the label-free migration on MME-T by n, both logs, mean with the 5th and 95th percentile over draws; mean above the level 5 % at n = "
      + "; ".join(f"{S[n]}: " + (", ".join(m["n_labels_with_mean_above_the_level"]) or "none") for n, m in RP["mme_t"].items()) + ";")
    Q("  (c) capped edits of part c' with two decimals: " + ", ".join(f"{S[n]} {c['print_two_decimals']}" for n, c in RP["capped"].items()) + ";")
    Q(f"  (d) running example: two different edits = {yn(EX['two_different_edits'])}; stream positions (0-based) {a['stream_position_0_based']} (complaint's edit, largest coordinate) and "
      f"{b['stream_position_0_based']} (why-not rephrase).  out/running_example.json itself is not changed: the flag is the file running_example_flag.json.")
    Q("")
    Q(f"CHECKS ({CK['n_checks']}; failed: {CK['n_checks_failed']})")
    for x in CK["checks"]:
        Q(f"  [{'ok' if x['ok'] else 'FAILED'}] item {x['item']}: {x['check']}" + ("" if x["detail"] is None else f"   -- {str(x['detail'])[:400]}"))
    with open(os.path.join(FIX, "FIXUP.txt.tmp"), "w") as fh:
        fh.write("\n".join(q) + "\n")
    os.replace(os.path.join(FIX, "FIXUP.txt.tmp"), os.path.join(FIX, "FIXUP.txt"))
    print("table_fixup.txt:", len(o), "lines; FIXUP.txt:", len(q), "lines")


if __name__ == "__main__":
    main()
