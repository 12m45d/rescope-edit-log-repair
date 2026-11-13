"""Verification of E1: renders out/table.txt, out/paper_numbers.json (+ _doc) and verdicts.json from out/checks.json and out/results.json
(written by v_stats.py).  No number is typed here except the cells QUOTED from the experiment agent's summary (the claims under verification)."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import sys
import time

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
V = FINAL + "/e1_main/verify"
VO = V + "/out"
sys.path.insert(0, FINAL + "/common")
import fx  # noqa: E402

DEV = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
CELLS = ("edit.rel.exact", "edit.tgen.exact", "edit.igen.exact", "probe.D1", "hn.HN-I.D1", "hn.HN-Q.D1", "hn.HN-X.D1")
# the main table of the experiment agent's summary (quoted; Rel / T-Gen / I-Gen | D1 heldout_disjoint | D1 HN-I / HN-Q / HN-X), Qwen then LLaVA
SUMMARY = {"deployed": ((78, 9, 22, 54.2, 77, 78, 87), (88, 29, 66, 62.3, 63, 66, 76)), "rescope": ((100, 56, 68, 3.9, 32, 24, 49), (98, 38, 90, 3.1, 20, 36, 42)),
           "rescope_x": ((100, 21, 57, 2.5, 9, 8, 16), (98, 15, 70, 2.4, 5, 9, 14)), "where_logonly": ((78, 7, 26, 1.1, 6, 9, 13), (92, 28, 68, 2.8, 20, 35, 38)),
           "white_1_shared": ((100, 45, 75, 5.5, 19, 29, 37), (98, 29, 93, 3.6, 17, 51, 53)), "white_0.1_shared": ((100, 60, 79, 6.9, 35, 41, 56), (98, 38, 93, 3.8, 18, 52, 56)),
           "cosine_shared": ((100, 3, 49, 4.3, 5, 7, 9), (98, 2, 69, 2.6, 3, 4, 6)), "retr_sum": ((100, 86, 95, 4.5, 52, 18, 56), (98, 89, 97, 6.7, 60, 20, 69)),
           "dual": ((64, 8, 18, 0.0, 1, 4, 3), (72, 28, 51, 0.4, 5, 18, 21))}
# two counts of the summary's anomaly list (quoted)
SUMMARY_COUNTS = {"cosine_published_rows_differ_L": 5, "prompt_position_max_rows": 12}


def f(v, nd=1):
    return "-" if v is None else (str(v) if isinstance(v, (str, bool)) else f"{v:.{nd}f}")


def TT(h, rows):
    return fx.io.text_table(h, [[str(x) for x in r] for r in rows])


def main():
    C = fx.io.jload(VO + "/checks.json")
    K, A, E = C["checks"], C["A"], C["E_own_gpu"]
    o, num, doc = [], {}, {}
    P = o.append

    def put(name, value, text):
        name = "".join({"0": "Zero", "1": "One", "2": "Two", "3": "Three"}.get(ch, ch) for ch in name if ch.isalnum())      # names of LaTeX macros: letters only
        assert name.isalpha() and name not in num, name
        if value is not None:
            num[name], doc[name] = value, text
    P(f"VERIFICATION OF E1 (main comparison; E1b, E1c, E7)      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by verify/code/v_render.py from verify/out/checks.json")
    P("own evaluation: the library's plain Evaluator (rescope.Evaluator: no driver, no memo, no key cache; every row a B = 1 forward), gates built on the model's device from keys captured in the job;")
    P("calibration draw 0, alpha = 0.05, n = 200; own statistics: own crossed / paired percentile bootstrap (2000 resamples; other random numbers than the driver's).")
    P("")
    P("(0) CHECKS")
    P(TT(["check", "ok", "numbers"], [[k, v.get("ok"), "; ".join(f"{a} = {b if not isinstance(b, float) else f'{b:.3g}'}" for a, b in v.items() if a != "ok" and not isinstance(b, (list, dict)))] for k, v in K.items()]))
    P("")
    # ---------------------------------------------------------------- (1) own forwards
    P("(1) OWN FORWARDS AGAINST THE EXPERIMENT'S ROWS, calibration draw 0 (deployed: no draw): mine / experiment, %; all 100 edits, heldout_disjoint, valid near-scope negatives")
    n_flags, n_diff, missing_logs = 0, 0, []
    for log in DEV:
        e = E.get(log, {})
        if "missing" in e or not e.get("cells"):
            missing_logs.append(log)
            P(f"  {log}: " + (e.get("missing") or "teacher-forced blocks not complete") + "  [NOT FINISHED]")
            if not e.get("flags"):
                continue
        rows = []
        for m, c in e.get("cells", {}).items():
            rows.append([m] + [f"{f(c[k]['mine'], 2 if k.startswith('probe') else 1)} / {f(c[k]['experiment'], 2 if k.startswith('probe') else 1)}" if k in c else "-"
                               for k in ("probe.fired", "probe.D1", "edit.rel.exact", "edit.tgen.exact", "edit.igen.exact", "cov.tgen", "cov.igen", "hn.HN-I.D1", "hn.HN-Q.D1", "hn.HN-X.D1")])
            for k, v in c.items():
                put(f"EoneVer{SHORT[log]}{''.join(w.capitalize() for w in m.split('_'))}{''.join(ch for ch in k.title() if ch.isalpha())}", v["mine"],
                    f"{log}, {m}, calibration draw 0: {k} (%) from the verification's own forwards (experiment: {v['experiment']})")
        P(f"  {log}")
        P("  " + TT(["method", "fire hd", "D1 hd", "Rel", "T-Gen", "I-Gen", "T-Gen coverage", "I-Gen coverage", "HN-I D1", "HN-Q D1", "HN-X D1"], rows).replace("\n", "\n  "))
        rows = []
        for m, fl in e.get("flags", {}).items():
            for k, a in fl.items():
                n_flags += a["n"]
                n_diff += a["n_differ"] + a["n_missing"]
            kinds = sorted({k.split(".")[0] for k in fl})
            rows.append([m] + [f"{sum(a['n_differ'] for k, a in fl.items() if k.startswith(kd))} of {sum(a['n'] for k, a in fl.items() if k.startswith(kd))}" if kd in kinds else "not run yet" for kd in ("edit", "probe", "hn")] +
                        [", ".join(sorted({s for a in fl.values() for s in a["sids"]}))[:120]])
        P("  row-level agreement of the flags (exact, D1, released, released edit, own edit released): flags that differ / flags compared")
        P("  " + TT(["method", "edit rows", "heldout_disjoint rows", "near-scope rows", "rows that differ"], rows).replace("\n", "\n  "))
        t = e.get("thresholds_vs_driver", {})
        if t and "error" not in t:
            P("  my gates (built on the GPU from live keys) against the driver's gates of the experiment (CPU, key cache): " +
              "; ".join(f"{m}: thresholds differ by at most {v['max_rel_diff']:.1e} (relative), same calibration queries {v['calibration_sids_equal']}" +
                        (f", same null-repair queries {v['null_sids_equal']}" if "null_sids_equal" in v else "") + f", driver digest = experiment's cell {v['driver_digest_equals_experiment_cell']}" for m, v in t.items()))
        pe = e.get("endpoint_draw0_rescope_minus_cosine_shared")
        if pe:
            P("  primary endpoint at draw 0, (T-Gen + I-Gen) / 2 exact, rescope minus cosine_shared, paired bootstrap over the 100 edits: " +
              "; ".join(f"from {k.replace('_', ' ')}: {v['value']:+.1f} [{v['ci_lo']:+.1f}, {v['ci_hi']:+.1f}]" for k, v in pe.items()))
            put(f"EoneVer{SHORT[log]}EndpointDrawZeroMine", pe["mine"]["value"], f"{log}: (T-Gen + I-Gen) / 2 exact, rescope minus cosine_shared at calibration draw 0, points, from the verification's own forwards")
            put(f"EoneVer{SHORT[log]}EndpointDrawZeroExperimentRows", pe["experiment_rows"]["value"], f"{log}: the same difference from the experiment's per-edit rows")
        P("")
    put("EoneVerFlagsCompared", n_flags, "row-level flags of the verification's own forwards compared with the experiment's rows (both development logs as far as run)")
    put("EoneVerFlagsDiffer", n_diff, "of these, flags that differ or have no row in the experiment's files")
    # ---------------------------------------------------------------- (2) generation
    P("(2) FREE GENERATION, D2 AND DUAL ON SEEDED SAMPLES (own decodes; 60 edit prompts, 60 heldout_disjoint probes, 60 near-scope negatives, 60 pre-correct heldout_disjoint rows; dual: the 60 probes + 40 edit prompts)")
    for log in DEV:
        e = E.get(log, {})
        g = e.get("generation")
        if not g:
            P(f"  {log}: not run yet  [NOT FINISHED]")
            continue
        rows = []
        for m, v in g.items():
            for kind, a in v["agreement"].items():
                rt = next((x for k, x in v["rates"].items() if k.startswith(kind)), None)
                rows.append([m, kind, a["n"], a["n_text_differs"], a["n_flag_differs"], a["n_fired_differs"], f"{a['n_not_decoded_by_experiment']} ({a['n_not_decoded_and_not_pre_edit']})",
                             "-" if rt is None else f"{f(rt['mine'])} / {f(rt['experiment_same_rows'])} (n = {rt['n']})", ", ".join(a["sids"])[:90]])
        P(f"  {log}")
        P("  " + TT(["method", "kind", "rows", "48-token text differs", "eq / contains / agree differs", "released differs", "not decoded by the driver (of them not pre-edit in my decode)",
                     "eq (edit) or agree (probe, hn): mine / experiment", "rows"], rows).replace("\n", "\n  "))
        d = g["dual"]["dual"]
        P(f"  dual: release decision differs on {d['n_release_differs']} of {d['n']} queries; released mine / experiment: probes {f(d['by_kind']['probe']['released_mine'])} / {f(d['by_kind']['probe']['released_experiment'])} "
          f"(n = {d['by_kind']['probe']['n']}), edit prompts {f(d['by_kind']['edit']['released_mine'])} / {f(d['by_kind']['edit']['released_experiment'])} (n = {d['by_kind']['edit']['n']})" +
          (f"; teacher-forced composition (my decodes select the model, my forward): flags that differ {d['tf_composition']['n_flag_differs']}; exact on the edit prompts {f(d['tf_composition']['edit_exact_mine'])} / "
           f"{f(d['tf_composition']['edit_exact_experiment'])}; D1 on the probes {f(d['tf_composition']['probe_D1_mine'])} / {f(d['tf_composition']['probe_D1_experiment'])}" if "tf_composition" in d else ""))
        d2 = e.get("d2")
        if d2:
            P("  D2 on the sample: " + "; ".join(f"{m}: D2 mine / experiment {f(v['D2_mine'])} / {f(v['D2_experiment_same_rows'])} % of {v['n']} rows (flags that differ {v['n_D2_differs']}, texts that differ {v['n_text_differs']}, "
                                               f"rows in the experiment's D2 set {v['n_in_experiment_d2_set']})" for m, v in d2.items()))
        P("")
    # ---------------------------------------------------------------- (3) summary table
    P("(3) THE MAIN TABLE OF THE EXPERIMENT'S SUMMARY (alpha = 0.05, mean of draws 0-2) AGAINST THE CELLS RECOMPUTED FROM ITS ROW FILES: Rel / T-Gen / I-Gen | D1 heldout_disjoint | D1 HN-I / HN-Q / HN-X")
    rows, n_bad = [], 0
    for lab, (q, l_) in SUMMARY.items():
        for log, claim in ((DEV[0], q), (DEV[1], l_)):
            m = A[log][lab]["mean"]
            mine = [m[k] for k in CELLS]
            ok = all(abs(x - c) <= (0.0551 if k == "probe.D1" else 0.551) for x, c, k in zip(mine, claim, CELLS))     # rounding of a value printed with one decimal (two for D1)
            n_bad += int(not ok)
            rows.append([lab, SHORT[log], "/".join(f(x, 0) for x in claim[:3]) + f" | {claim[3]} | " + "/".join(f(x, 0) for x in claim[4:]),
                         "/".join(f(x, 1) for x in mine[:3]) + f" | {f(mine[3], 2)} | " + "/".join(f(x, 1) for x in mine[4:]), "yes" if ok else "NO"])
    P(TT(["row", "log", "summary", "recomputed from the rows", "agree within rounding"], rows))
    put("EoneVerSummaryCellsDisagree", n_bad, "rows of the summary's main table (18 = 9 methods x 2 logs) with a cell that is not the rounded value recomputed from the experiment's row files")
    P("")
    # ---------------------------------------------------------------- (4) H2, H3, decision, selection
    P("(4) CONFIRMATORY COMPARISONS recomputed from the experiment's per-edit rows (own crossed bootstrap, draws x edits): mine | experiment")
    for log in DEV:
        P(f"  {log}   H2: (T-Gen + I-Gen) / 2 exact, all edits, ReScope minus the baseline; 3 draws")
        gt = lambda g: "-" if g is None else f"{g['value']:+.1f} [{g['ci_lo']:+.1f}, {g['ci_hi']:+.1f}]"     # noqa: E731
        P("  " + TT(["baseline", "method", "difference [95 %] mine", "p Holm mine", "difference [95 %] experiment", "p Holm experiment", "gate coverage form (20 draws) mine", "experiment"],
                    [[role, d["method"], f"{d['value']:+.1f} [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}]", f(d["p_holm"], 4), f"{d['exp']['diff']:+.1f} [{d['exp']['ci_lo']:+.1f}, {d['exp']['ci_hi']:+.1f}]", f(d["exp"]["p_holm"], 4),
                      gt(d.get("gate")), gt((d.get("gate") or {}).get("exp"))] for role, d in C["B_H2"][log].items()]).replace("\n", "\n  "))
        P("  H3: the same quantity on the odd edits under the near-scope constraint, a minus b; behaviour 3 draws, gate-level coverage form 20 draws")
        P("  " + TT(["pair", "a", "b", "behaviour mine", "behaviour experiment", "gate form mine", "gate form experiment", "edits at cap a / b", ">= 10 and intervals exclude 0"],
                    [[pair, d["a"], d["b"], f"{d['diff']:+.1f} [{d['ci_lo']:+.1f}, {d['ci_hi']:+.1f}]", f"{d['exp']['diff']:+.1f} [{d['exp']['ci_lo']:+.1f}, {d['exp']['ci_hi']:+.1f}]",
                      f"{d['gate_diff']:+.1f} [{d['gate_ci_lo']:+.1f}, {d['gate_ci_hi']:+.1f}]", f"{d['exp']['gate']['value']:+.1f} [{d['exp']['gate']['ci_lo']:+.1f}, {d['exp']['gate']['ci_hi']:+.1f}]",
                      f"{d['n_edits_at_cap_a']} / {d['n_edits_at_cap_b']}", d["conditions_met"]] for pair, d in C["C_H3"][log].items()]).replace("\n", "\n  "))
        for pair, d in C["C_H3"][log].items():
            put(f"EoneVer{SHORT[log]}Hthree{''.join(w.capitalize() for w in pair.split('_'))}Diff", d["diff"], f"{log}: H3 difference {d['a']} minus {d['b']} (points), recomputed from the experiment's H3 rows")
        for role, d in C["B_H2"][log].items():
            put(f"EoneVer{SHORT[log]}Htwo{''.join(w.capitalize() for w in role.split('_'))}Diff", d["value"], f"{log}: H2 difference rescope minus {d['method']} (points), recomputed from the experiment's per-edit rows")
    P(f"  DECISION (R6 / R25, applied literally): mine {C['C_decision']} ; experiment {K['C_H3_decision']['experiment']}; largest H3 difference of a cosine-family pair: {K['C_H3_decision']['largest_H3_difference_points']:+.1f} points")
    P("")
    P("(5) SELECTION IN FAVOUR OF THE BASELINE recomputed from the per-edit coverage flags of the gate-level cells (20 draws): largest mean coverage; ties by the registered order")
    rows = []
    for log in DEV:
        for fam, D in C["D_selection"][log].items():
            for key, p in D["pick"].items():
                rows.append([SHORT[log], fam, key, p["selected"], f(p["value"], 3), ", ".join(p["tied"]) if len(p["tied"]) > 1 else "-", f"{p['runner_up'][1]} {f(p['runner_up'][0], 3)}", D["experiment"][key]])
    P(TT(["log", "family", "read-out", "selected (mine)", "coverage", "tied with", "runner-up", "experiment"], rows))
    P(f"  the selection file is older than the first behavioural cell of a selected gate: {K['D_selection']['selection_file_older_than_first_behavioural_cell_of_a_selected_gate']} "
      f"({f(K['D_selection']['seconds_between'], 0)} s)")
    P("")
    # ---------------------------------------------------------------- (6) other facts
    P("(6) COUNTS, BOUND, FURTHER CLAIMS OF THE SUMMARY")
    for log in DEV:
        g = C["G"][log]
        P(f"  {log}: edit rows {g['n_edit_rows']}, heldout_disjoint rows {g['n_heldout_disjoint_rows']} in {g['n_query_clusters']} query clusters (invalid pre-edit answers among them: {g['n_invalid_y0_rows_in_heldout_disjoint']}), "
          f"near-scope rows {g['n_hn']} {g['hn_by_kind']}, with a degenerate pre-edit answer {g['hn_degenerate_y0']}; the same row set for every method and draw: {g['same_row_set_for_every_method_and_draw']}; "
          f"behavioural cells {g['behavioural_cells']['n']}, hook_mismatch max {g['behavioural_cells']['hook_mismatch_max']}, memo conflicts max {g['behavioural_cells']['memo_conflicts_max']}")
        ov = g["gates_whose_mean_literal_false_firing_exceeds_the_bound"]
        P("     calibrated gates whose false firing on heldout_disjoint (literal rule, mean of 20 draws) exceeds the bound of their rule: " +
          ("none" if not ov else "; ".join(f"{k} {f(v.get('mean_literal', v['mean_fire_hd']), 2)}" + (f" (capped {f(v['n_capped_mean'])})" if v.get("n_capped_mean") else "") for k, v in ov.items())))
    Hh = C["H"]
    P(f"  logs in results.json: {C['G']['logs_in_results_json']}; with behavioural records: {C['G']['logs_with_behavioural_records']}")
    P("  MME-T firing on Qwen (20 draws; mean [min, max]): " + "; ".join(f"{k} {f(v['mean'])} [{f(v['min'])}, {f(v['max'])}]" for k, v in Hh["MME-T_firing_Qwen"].items()))
    cl = Hh["cosine_published_vs_last_position_L"]
    P(f"  cosine 0.65 at all positions against the last position: Qwen {Hh['cosine_published_vs_last_position_Q']['n_rows_any_field']} of {Hh['cosine_published_vs_last_position_Q']['n']} rows differ in any field; "
      f"LLaVA, of {cl['n']} rows: {cl['n_rows_any_field']} differ in some field, {cl['n_released_differs']} in the release flag, {cl['n_outcome_differs']} in the teacher-forced outcome (token accuracy), "
      f"{cl['n_exact_differs']} in exact / D1 (summary: {SUMMARY_COUNTS['cosine_published_rows_differ_L']} rows differ)")
    P(f"  LLaVA, gate at the decode's position: largest number of behavioural rows that change in a cell {f(Hh['prompt_position_LLaVA']['max_rows_differ'], 0)} (summary: {SUMMARY_COUNTS['prompt_position_max_rows']}); "
      f"largest change of a cell {f(Hh['prompt_position_LLaVA']['largest_cell_change_points'], 2)} points")
    P("  capped edits (mean of 20 draws): " + "; ".join(f"{k} {f(v)}" for k, v in Hh["capped"].items()))
    i = C["I"]
    if i.get("available"):
        P("  re-run of the experiment's own e1_analyze.py and e1_render.py on a private copy: identical to the experiment's files: " + "; ".join(f"{k}: {v}" for k, v in i["identical"].items()))
        P(f"     E2 rows of table (2b): E2 rendered {i['e2_meta_then'].get('rendered') if i.get('e2_meta_then') else '-'} when the experiment read them; at the re-run ({i.get('e2_read_now')}) E2 was rendered "
          f"{i['e2_meta_now'].get('rendered') if i.get('e2_meta_now') else '-'}")
    P("")
    P("(7) WHAT THE SUMMARY DOES NOT CARRY (recomputed from the experiment's row files)")
    J = C["J"]
    P("  near-scope D1 of the three calibration draws d0/d1/d2 (%; the main table of the experiment prints their mean and lists the draws for heldout_disjoint and the edits only):")
    P("  " + TT(["log", "method", "HN-I", "HN-Q", "HN-X"], [[SHORT[log], lab] + ["/".join(f(x) for x in d[k]) for k in ("HN-I", "HN-Q", "HN-X")] for log in DEV for lab, d in J["hn_D1_spread_over_draws"][log].items()]).replace("\n", "\n  "))
    P("  generated answers at draw 0: D2 (% of the pre-correct heldout_disjoint rows) and wrong-target rate on the near-scope negatives (% of all valid rows whose generated answer is the target of the released edit):")
    P("  " + TT(["log", "method", "D2 (n)", "wrong target HN-I", "HN-Q", "HN-X"], [[SHORT[log], lab, f"{f(d['D2'])} ({d['n_pre_correct']})"] + [f(d[f"wrong_target.{k}"]) for k in ("HN-I", "HN-Q", "HN-X")]
                                                                                 for log in DEV for lab, d in J["generation_draw0"][log].items()]).replace("\n", "\n  "))
    pj = J["paper_numbers_json"]
    P(f"  the experiment's paper_numbers.json ({pj['n']} names): names with a wrong-target rate {pj['n_names_with_wrong_target']}, with D2 {pj['n_names_with_D2']}, with a generation metric {pj['n_names_with_generation']}")
    P("  LLaVA near-scope negatives, D1 (%) on rows with a non-degenerate / degenerate pre-edit answer (empty, one character, '?', bare None / No), mean of draws 0-2 (n rows):")
    lg = DEV[1]
    P("  " + TT(["method"] + [f"{k} non-deg / deg" for k in ("HN-I", "HN-Q", "HN-X")],
                [[lab] + [f"{f(c[f'{k}.deg=0']['D1'])} / {f(c[f'{k}.deg=1']['D1'])} ({c[f'{k}.deg=0']['n']} / {c[f'{k}.deg=1']['n']})" for k in ("HN-I", "HN-Q", "HN-X")]
                 for lab, c in J["hn_D1_by_degenerate_pre_edit_answer"][lg].items()]).replace("\n", "\n  "))
    P("  exact McNemar (PROTOCOL section 4; not in the experiment's files), D1 on heldout_disjoint, ReScope against the deployed log, per calibration draw: rows changed only under the deployed log / only under ReScope (p):")
    for log in DEV:
        P(f"     {log}: " + "; ".join(f"d{d}: {v['only_deployed']} / {v['only_rescope']} (p = {v['p_exact']:.1e})" for d, v in J["mcnemar_D1_heldout_disjoint_rescope_vs_deployed"][log].items()))
    for log in DEV:
        for d, v in J["mcnemar_D1_heldout_disjoint_rescope_vs_deployed"][log].items():
            put(f"EoneVer{SHORT[log]}McNemarOnlyRescopeDraw{['Zero', 'One', 'Two'][int(d)]}", v["only_rescope"], f"{log}, calibration draw {d}: heldout_disjoint rows with D1 under ReScope and not under the deployed log")
    put("EoneVerSanityCells", K["F_sanity_e0"]["n_cells_compared"], "cells of the sanity run e0 compared with the cells recomputed from the experiment's rows and gate-level cells")
    put("EoneVerSanityLargestDiff", K["F_sanity_e0"]["largest_abs_diff"], "largest absolute difference of these cells (points)")
    put("EoneVerCellsRecomputed", K["A_cells_vs_results_json"]["n_cells_compared"], "behavioural cells recomputed from the experiment's row files and compared with its results.json")
    put("EoneVerCellsLargestDiff", K["A_cells_vs_results_json"]["largest_abs_diff"], "largest absolute difference of these cells (points)")
    put("EoneVerPaperNumbersCompared", K["A_cells_vs_paper_numbers"]["n_numbers_compared"] + K["H_gate_level_paper_numbers"]["n_numbers_compared"], "numbers of the experiment's paper_numbers.json recomputed here")
    put("EoneVerPaperNumbersLargestDiff", max(K["A_cells_vs_paper_numbers"]["largest_abs_diff"], K["H_gate_level_paper_numbers"]["largest_abs_diff"]), "largest absolute difference of these numbers")
    put("EoneVerDecisionEqual", bool(K["C_H3_decision"]["equal"]), "the decision of R6 / R25 recomputed here equals the experiment's decision.json (trade-off for every pair)")
    put("EoneVerSelectionEqual", bool(K["D_selection"]["equal"]), "the selection recomputed here equals best_cosine_gate.json")
    fx.io.jdump(dict(sorted(num.items())), VO + "/paper_numbers.json")
    fx.io.jdump(dict(sorted(doc.items())), VO + "/paper_numbers_doc.json")
    with open(VO + "/table.txt.tmp", "w") as fh:
        fh.write("\n".join(o) + "\n")
    os.replace(VO + "/table.txt.tmp", VO + "/table.txt")
    print("table.txt:", len(o), "lines;", len(num), "paper numbers; summary rows that disagree:", n_bad, "; logs without own forwards:", missing_logs)


if __name__ == "__main__":
    main()
