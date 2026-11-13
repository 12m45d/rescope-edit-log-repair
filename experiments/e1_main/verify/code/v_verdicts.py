"""Verification of E1: verdicts.json from out/checks.json (v_stats.py).  The verdict of every headline claim is derived by code from the checks;
the notes name the files and lines.  Re-run after every block."""
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
sys.path.insert(0, FINAL + "/common")
import fx  # noqa: E402

DEV = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")


def main():
    C = fx.io.jload(V + "/out/checks.json")
    K, E, J, Hh = C["checks"], C["E_own_gpu"], C["J"], C["H"]
    own = {}
    for log in DEV:
        e = E.get(log, {})
        fl = e.get("flags", {})
        n = sum(a["n"] for m in fl.values() for a in m.values())
        nd = sum(a["n_differ"] + a["n_missing"] for m in fl.values() for a in m.values())
        kinds = sorted({k.split(".")[0] for m in fl.values() for k in m})
        g = e.get("generation")
        gen_bad = None if not g else sum(a["n_text_differs"] + a["n_flag_differs"] + a["n_fired_differs"] + a["n_not_decoded_and_not_pre_edit"] for m in g.values() for a in m["agreement"].values())
        d2_bad = None if not e.get("d2") else sum(v["n_D2_differs"] + v["n_text_differs"] for v in e["d2"].values())
        dual_bad = None if not g else g["dual"]["dual"]["n_release_differs"] + (g["dual"]["dual"].get("tf_composition") or {}).get("n_flag_differs", 0)
        own[log] = {"tf_complete": kinds == ["edit", "hn", "probe"], "n_flags_compared": n, "n_flags_differ": nd, "generation_done": g is not None, "generation_rows_that_differ": gen_bad,
                    "d2_done": e.get("d2") is not None, "d2_rows_that_differ": d2_bad, "dual_decisions_or_composition_flags_that_differ": dual_bad,
                    "endpoint_draw0": e.get("endpoint_draw0_rescope_minus_cosine_shared")}
    tf_all = all(o["tf_complete"] and o["n_flags_differ"] == 0 for o in own.values())
    tf_some = any(o["tf_complete"] and o["n_flags_differ"] == 0 for o in own.values()) and not any(o["n_flags_differ"] for o in own.values())
    gen_all = all(o["generation_done"] and o["d2_done"] and o["generation_rows_that_differ"] == 0 and o["d2_rows_that_differ"] == 0 and o["dual_decisions_or_composition_flags_that_differ"] == 0 for o in own.values())
    gen_some = any(o["generation_done"] for o in own.values()) and not any((o["generation_rows_that_differ"] or 0) + (o["d2_rows_that_differ"] or 0) + (o["dual_decisions_or_composition_flags_that_differ"] or 0) for o in own.values())
    not_run = [log for log, o in own.items() if not (o["tf_complete"] and o["generation_done"] and o["d2_done"])]
    v3 = lambda full, part: "CONFIRMED" if full else ("PARTLY CONFIRMED" if part else "REFUTED")     # noqa: E731
    cells_ok = K["A_cells_vs_results_json"]["ok"] and K["A_cells_vs_paper_numbers"]["ok"] and K["H_gate_level_paper_numbers"]["ok"]
    anomalies_exact = Hh["cosine_published_vs_last_position_L"]["n_rows_any_field"] == 5 and Hh["prompt_position_LLaVA"]["max_rows_differ"] <= 12 and Hh["prompt_position_LLaVA"]["largest_cell_change_points"] <= 1.0
    claims = {
        "decision_R6_R25_trade_off_for_every_pair": {
            "verdict": v3(K["C_H3_decision"]["ok"], K["C_H3_decision"]["equal"]),
            "evidence": {"mine": C["C_decision"], "experiment": K["C_H3_decision"]["experiment"], "largest_H3_difference_of_a_cosine_family_pair_points": K["C_H3_decision"]["largest_H3_difference_points"],
                         "H3": {log: {p: {k: d[k] for k in ("a", "b", "diff", "ci_lo", "ci_hi", "gate_diff", "gate_ci_lo", "gate_ci_hi", "n_edits_at_cap_a", "n_edits_at_cap_b")} for p, d in C["C_H3"][log].items()} for log in DEV}},
            "note": "recomputed from the H3 row files and the gate-level cells with an own crossed bootstrap; the rule applied literally gives trade-off for both pairs and for retr"},
        "selection_best_cosine_gate_and_retr": {
            "verdict": v3(K["D_selection"]["ok"], K["D_selection"]["equal"]),
            "evidence": {log: {fam: {"pick": D["pick"], "experiment": D["experiment"]} for fam, D in C["D_selection"][log].items()} for log in DEV} | {"timing": K["D_selection"]},
            "note": "recomputed from the per-edit coverage flags of the 20 draws; on Qwen the H3 read-out is an exact tie between white_1_shared and whitecos_1_shared (the H3 coverage of a shared-scale gate does not "
                    "depend on the calibration draw); the tie is broken by the experiment's own rule (registered order; e1c.py line 36, e1_select.py lines 46-50), which PROTOCOL.md does not contain"},
        "main_table_cells_of_the_summary": {
            "verdict": v3(cells_ok and tf_all, cells_ok and tf_some),
            "evidence": {"cells_vs_results_json": K["A_cells_vs_results_json"], "cells_vs_paper_numbers": K["A_cells_vs_paper_numbers"], "own_forwards": {log: {k: o[k] for k in ("tf_complete", "n_flags_compared", "n_flags_differ")} for log, o in own.items()},
                         "gate_cells_rebuilt": K.get("I_gate_cells_rebuilt")},
            "note": "all cells of the summary's table are the rounded means of the row files; own forwards through the library's plain Evaluator (draw 0: rescope, rescope_shared, rescope_null, cosine_shared, deployed)" +
                    ("" if tf_all else "; own forwards NOT finished for: " + ", ".join(lg for lg, o in own.items() if not o["tf_complete"]))},
        "H2_confirmatory_comparisons": {
            "verdict": v3(K["B_H2"]["ok"], K["B_H2"]["largest_abs_diff_point_estimates"] < 1e-9),
            "evidence": {"check": K["B_H2"], "endpoint_draw0_from_own_forwards": {log: o["endpoint_draw0"] for log, o in own.items()}},
            "note": "point estimates equal; interval ends within the Monte Carlo error of 2000 resamples; the same intervals exclude 0; the same Holm rejections.  The family lacks the calibrated z-score gate (protocol problem 2)"},
        "H3_comparisons": {"verdict": v3(K["C_H3_decision"]["ok"], K["C_H3_decision"]["largest_abs_diff_point_estimates"] < 1e-9), "evidence": K["C_H3_decision"],
                           "note": "behavioural differences and gate-level coverage forms recomputed; the raise of the behavioural cells equals the gate-level raise; no unreachable draw"},
        "free_generation_D2_dual": {
            "verdict": v3(gen_all, gen_some),
            "evidence": {log: {k: o[k] for k in ("generation_done", "generation_rows_that_differ", "d2_done", "d2_rows_that_differ", "dual_decisions_or_composition_flags_that_differ")} for log, o in own.items()},
            "note": "own decodes on seeded samples (60 edit prompts, 60 probes, 60 near-scope negatives, 60 pre-correct rows; dual on 60 probes + 40 edit prompts)" + ("" if gen_all else "; NOT finished for: " + ", ".join(not_run))},
        "protocol_logs_pools_draws": {
            "verdict": v3(K["G_counts_and_discipline"]["ok"] and K.get("I_calibration_draws_from_REPAIR", {}).get("ok", False) and K["D_selection"]["ok"], K["G_counts_and_discipline"]["ok"]),
            "evidence": {"logs_in_results_json": C["G"]["logs_in_results_json"], "logs_with_behavioural_records": C["G"]["logs_with_behavioural_records"], "draws": K.get("I_calibration_draws_from_REPAIR"),
                         "jobs": C["G"]["jobs"], "counts": {log: {k: C["G"][log][k] for k in ("n_edit_rows", "n_heldout_disjoint_rows", "n_query_clusters", "n_hn", "hn_by_kind", "hn_degenerate_y0")} for log in DEV}},
            "note": "only the two development logs (behaviour) and the three working points (gate level) occur in code, logs and results; the 20 calibration draws are the seeded draws from the valid REPAIR rows (own "
                    "re-derivation from the library's rows), the null-repair samples lie in REPAIR and share no row with the draw, no HELD-OUT row calibrates; locality on heldout_disjoint; BEN-T / MME-T through "
                    "fx.pools.firing (e1_gate.py line 104), the benign part of fx.gate.evaluate is not used (line 64); hook_mismatch 0 and no memo conflict in every cell"},
        "sanity_run_e0_agreement": {"verdict": v3(K["F_sanity_e0"]["ok"], False), "evidence": K["F_sanity_e0"], "note": "every shared cell (behaviour draws 0-2, gate level 20 draws) recomputed from the row files / cells"},
        "internal_consistency_results_table_paper_numbers": {
            "verdict": v3(K.get("I_rerun_of_analysis_and_rendering", {}).get("ok", False) and cells_ok and K["G_counts_and_discipline"]["ok"], cells_ok),
            "evidence": {"rerun": K.get("I_rerun_of_analysis_and_rendering"), "counts": K["G_counts_and_discipline"], "identical": C["I"].get("identical")},
            "note": "the experiment's e1_analyze.py and e1_render.py re-run on a private copy give the same records, endpoints, decision, paper numbers and table"},
        "E1b_E1c_frontier_and_matched_coverage": {
            "verdict": v3(K.get("I_frontier_rerun_one_log_per_process", {}).get("ok", False) and K.get("I_E1c_matched_coverage_own_computation", {}).get("ok", False), K.get("I_E1c_matched_coverage_own_computation", {}).get("ok", False)),
            "evidence": {"frontier_rerun": K.get("I_frontier_rerun_one_log_per_process"), "E1c_own": K.get("I_E1c_matched_coverage_own_computation")},
            "note": "frontier.json is reproduced by a re-run with one log per process; the matched-coverage cells of the two registered features are reproduced by an own computation with the library's key cache"},
        "anomalies_list_of_the_summary": {
            "verdict": "CONFIRMED" if anomalies_exact else "PARTLY CONFIRMED",
            "evidence": {"cosine_published_vs_last_position": {"Q": Hh["cosine_published_vs_last_position_Q"], "L": Hh["cosine_published_vs_last_position_L"]}, "prompt_position_LLaVA": Hh["prompt_position_LLaVA"],
                         "MME-T_firing_Qwen": Hh["MME-T_firing_Qwen"], "capped": Hh["capped"],
                         "over_bound": {log: C["G"][log]["gates_whose_mean_literal_false_firing_exceeds_the_bound"] for log in DEV}},
            "note": "bound, MME-T shift, capped edits and H3 reachability are as stated; two counts are imprecise: cosine 0.65 at all positions against the last-position form on LLaVA: "
                    f"{Hh['cosine_published_vs_last_position_L']['n_rows_any_field']} rows differ in some field, {Hh['cosine_published_vs_last_position_L']['n_released_differs']} in the release flag, "
                    f"{Hh['cosine_published_vs_last_position_L']['n_outcome_differs']} in the teacher-forced outcome (the summary's 5), {Hh['cosine_published_vs_last_position_L']['n_exact_differs']} in exact / D1; at the decode's "
                    f"position up to {Hh['prompt_position_LLaVA']['max_rows_differ']:.0f} behavioural rows change in a cell and a cell moves by up to {Hh['prompt_position_LLaVA']['largest_cell_change_points']:.2f} points "
                    "(summary: at most 12 rows, 1 point or less).  Not in the summary although the files show it: further gates exceed their bound on the 20-draw mean (Qwen: ball, router, two-block whitecos, cos_null_shared; LLaVA: "
                    "cos_null, retr_min under the literal rule); the wrong-target rates and D2 of the generated answers; the spread of near-scope D1 over the three draws; the share of LLaVA near-scope negatives with a degenerate "
                    "pre-edit answer (31 percent) and the D1 on them"},
    }
    problems = [
        {"id": 1, "severity": "minor", "where": "e1_main/code/e1_analyze.py lines 380-398 (endpoints)", "what": "PROTOCOL.md section 4 asks for the exact McNemar test for binary paired outcomes; none is computed (Rel exact, near-scope D1, "
         "D1 against the deployed log are reported with the crossed bootstrap only)", "effect": "no number changes; a registered statistic is missing (verification: discordant rows in checks.json, J)",
         "evidence": J["mcnemar_D1_heldout_disjoint_rescope_vs_deployed"]},
        {"id": 2, "severity": "moderate", "where": "e1_main/code/e1_behave.py line 217 (list cal), e1_analyze.py line 363 (list base)", "what": "the primary endpoint is registered against EACH calibrated baseline gate; the calibrated z-score gate "
         "(a row of the driver's E1 row set) has no behavioural run, so H2 lacks it; the published forms of the ball gate and of the router are gate level only as well",
         "effect": "H2 has 18 instead of 20 tests; the Holm-adjusted values do not change materially if the missing comparisons sit at the resolution floor (0.009 would read 0.010); the summary states the omission"},
        {"id": 3, "severity": "moderate", "where": "e1_main/code/e1_render.py lines 399-432 (e2_rows), table (2b)", "what": "COMP-B, REEDIT-B and two-block ReScope-W are copied from e2_curves/out while E2 was running; R19 (a) asks for generated "
         "answers and D2 of COMP-B and REEDIT-B: D2 is there, the free-generation columns are absent; the registered secondary read-out 'D1 against the best linear repair' has no paired statistic (side by side only)",
         "effect": "table (2b) may be stale; re-render after E2 and its verification are final (unchanged at the time of this verification)"},
        {"id": 4, "severity": "minor", "where": "e1_main/code/e1_render.py lines 162-169 (table 2)", "what": "R25 (3): the draws are to be listed where they disagree; the near-scope cells print only the mean of three draws with a crossed "
         "interval", "effect": "the summary's near-scope D1 of ReScope hides a spread of up to 25 points over the draws", "evidence": J["hn_D1_spread_over_draws"]},
        {"id": 5, "severity": "minor", "where": "e1_main/code/e1c.py line 36, e1_select.py lines 46-50", "what": "exact tie of the H3 selection on Qwen (white_1_shared, whitecos_1_shared); tie rule not in PROTOCOL.md; whitecos_1_shared has no "
         "behavioural run", "effect": "none on the decision (trade-off with either gate at gate level)"},
        {"id": 6, "severity": "minor", "where": "e1_main/out/paper_numbers.json; e1_analyze.py lines 502-564 (paper)", "what": "no wrong-target rate, no D2 and no generation metric among the paper numbers; Holm p-values at the floor "
         "of the bootstrap (18 x 1 / 2000 = 0.009) are printed as values", "effect": "numbers a paper would quote for E7 are missing; print floor values as upper bounds", "evidence": J["paper_numbers_json"]},
        {"id": 7, "severity": "minor", "where": "summary of the experiment agent (anomalies)", "what": "two counts are inaccurate (see the claim anomalies_list_of_the_summary)", "effect": "text only"},
        {"id": 8, "severity": "none", "where": "e1_main/code/e1_frontier.py line 66, e1_analyze.py line 573", "what": "both development logs are opened in one CPU process (the driver's rule 'one log per process' is about the model session)",
         "effect": "none: frontier.json is identical when re-run with one log per process"},
    ]
    out = {"experiment": "E1 main comparison (with E1b, E1c, E7)", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "driver": fx.code_fingerprint(),
           "method_of_verification": "own B = 1 forwards and decodes through the library's plain Evaluator (no driver, no memo, no key cache), gates built on the model's device; own crossed bootstrap on the "
                                     "experiment's per-row files; re-run of the experiment's analysis, rendering and frontier scripts on a private copy; gate-level cells rebuilt for 3 of 20 draws",
           "claims": claims, "protocol_problems": problems, "not_finished": not_run, "checks": K, "own_forwards": own,
           "to_fix_before_the_numbers_enter_a_paper": [
               "re-render table (2b) from the final, verified E2 records; add the generated answers of COMP-B and REEDIT-B (R19 a) or state that they are not available",
               "add the calibrated z-score gate to the behavioural rows and to H2, or record in PROTOCOL.md that it is gate level only",
               "list the three calibration draws next to every near-scope cell; add wrong-target rates and D2 to paper_numbers.json; print Holm values at the bootstrap floor as upper bounds",
               "add the exact McNemar test for the binary paired comparisons (section 4) or amend the protocol",
               "state the tie rule of the selection in the paper's protocol text; correct the two counts of the anomalies list"],
           "paths": {"table": V + "/out/table.txt", "checks": V + "/out/checks.json", "results": V + "/out/results.json", "rows": V + "/out/rows/", "code": V + "/code/"}}
    fx.io.jdump(out, V + "/verdicts.json", indent=1)
    print(json.dumps({k: v["verdict"] for k, v in claims.items()}, indent=1))
    print("not finished:", not_run)


if __name__ == "__main__":
    main()
