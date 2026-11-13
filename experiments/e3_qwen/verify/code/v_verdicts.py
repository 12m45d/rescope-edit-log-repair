"""Verification of E3: assemble verdicts.json from the files written by v_compare.py, v_gate20.py, v_internal.py, v_rerender.py, v_claims.py,
v_misc.py and v_pub.py (every number is read from those files; the verdict words are the verifier's).

    python v_verdicts.py     -> ../verdicts.json, out/verdict_table.txt
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import time

V = f"{_RP.ROOT}/02_research/experiments/10_final/e3_qwen/verify"
J = lambda n: json.load(open(f"{V}/out/{n}.json"))                      # noqa: E731
cmp_, g20, inn, rr, cl, ex, misc, pub = J("compare"), J("gate20"), J("internal"), J("rerender"), J("claims"), J("claims_extra"), J("misc"), J("pub_TB-Q500-ae-l2_3")

# ---------------------------------------------------------------- check 2: own forwards (draw 0) and own rule (20 draws)
tot = {"probe_rows": 0, "probe_flags_differ": 0, "edit_rows": 0, "edit_flags_differ": 0, "prompt_rows": 0, "prompt_flags_differ": 0, "cov_flags_differ": 0, "cov_flags": 0, "keys": 0, "keys_equal": 0,
       "max_eta_diff": 0.0, "capped_differs": 0, "max_thr_diff": 0.0}
rows2 = []
for r in cmp_:
    assert not r.get("missing"), r
    for v in r["cache_check"].values():
        tot["keys"] += v["n"]
        tot["keys_equal"] += v["n_bitwise_equal"]
    for m, o in r["methods"].items():
        p, e, t, c = o["probes"], o["edits"], o["thresholds"], o["coverage_all_edits_gate_level"]
        tot["probe_rows"] += p["n"]
        tot["probe_flags_differ"] += p["n_differ_fired"] + p["n_differ_D1"] + p["n_missing_in_experiment"]
        tot["edit_rows"] += e["n"]
        tot["edit_flags_differ"] += e["n_differ_exact"] + e["n_differ_fired"] + e["n_differ_own_released_last"] + e["n_missing_in_experiment"]
        tot["max_eta_diff"] = max(tot["max_eta_diff"], t["eta_abs_diff"])
        tot["max_thr_diff"] = max(tot["max_thr_diff"], t["thr_summary_max_abs_diff"])
        tot["capped_differs"] += t["n_capped_mine"] != t["n_capped_experiment"]
        for k in ("tgen", "igen", "rel"):
            tot["cov_flags"] += c[k]["n"]
            tot["cov_flags_differ"] += c[k]["n_flags_differ"]
        q = o.get("probes_prompt_position")
        if q:
            tot["prompt_rows"] += q["n_shifted"]
            tot["prompt_flags_differ"] += q["n_differ_fired"] + q["n_differ_D1"]
        rows2.append({"log": r["log"], "method": m, "eta": [t["eta_mine"], t["eta_experiment"]], "capped": [t["n_capped_mine"], t["n_capped_experiment"]],
                      "hd_fired": [p["fired_mine"], p["fired_experiment_same_rows"]], "hd_D1": [p["D1_mine"], p["D1_experiment_same_rows"]], "n_hd": p["n"],
                      "rel_tgen_igen_exact": [[e[k]["exact_mine"], e[k]["exact_experiment_same_edits"]] for k in ("rel", "tgen", "igen")], "n_edits": e["n_edits"],
                      "coverage_tgen_igen_all_edits": [[c[k]["coverage_mine"], c[k]["coverage_experiment"]] for k in ("tgen", "igen")],
                      "flags_differ": p["n_differ_fired"] + p["n_differ_D1"] + e["n_differ_exact"] + e["n_differ_own_released_last"] + c["tgen"]["n_flags_differ"] + c["igen"]["n_flags_differ"],
                      "prompt_position": None if not q else {"n_shifted": q["n_shifted"], "hd_fired": [q["fired_mine"], q["fired_experiment"]], "hd_D1": [q["D1_mine"], q["D1_experiment"]],
                                                              "decisions_differing_from_tf": q["n_fired_differs_from_tf_mine"]}})
t20 = {"n_log_method": 0, "hd_flags": 0, "hd_flags_differ": 0, "cov_flags": 0, "cov_flags_differ": 0, "draws_capped_differs": 0, "max_eta_diff": 0.0, "max_mean_fire_diff": 0.0}
for r in g20["logs"]:
    for m, o in r["methods"].items():
        t20["n_log_method"] += 1
        t20["hd_flags"] += o["n_hd_flags_total"]
        t20["hd_flags_differ"] += o["n_hd_flags_differ_total"]
        t20["cov_flags"] += o["n_cov_flags_total"]
        t20["cov_flags_differ"] += o["n_cov_flags_differ_total"]
        t20["draws_capped_differs"] += o["n_draws_capped_differs"]
        t20["max_eta_diff"] = max(t20["max_eta_diff"], o["max_abs_eta_diff"])
        t20["max_mean_fire_diff"] = max(t20["max_mean_fire_diff"], abs(o["fire_mine"] - o["fire_exp"]))

# ---------------------------------------------------------------- check 3
c3 = {}
for bb, O in inn["backbones"].items():
    dv = O["deployed_vs_stored"]
    c3[bb] = {"results_json_cells_recomputed": O["results_json"]["n_cells_compared"], "results_json_disagree": O["results_json"]["n_disagree"],
              "paper_numbers": O["paper_numbers"]["n_names"], "paper_numbers_recomputed": O["paper_numbers"]["n_recomputed_here"], "paper_numbers_disagree": O["paper_numbers"]["n_disagree"],
              "paper_numbers_not_recomputed_kinds": O["paper_numbers"]["not_recomputed_kinds"],
              "rerun_of_their_analysis": rr[bb]["results"], "rerun_paper_numbers_differ": rr[bb]["paper_numbers"]["n_differ"], "rerendered_table_diff_lines": rr[bb]["table"]["n_diff_lines_other_than_time_stamps"],
              "dev_vs_e0": O["dev_vs_e0"], "row_files_have_expected_rows": all(c["all_row_files_have_my_sid_sets"] and c["ctx_hd_ok"] for c in O["counts"].values()),
              "near_scope_flags_differing": sum(c["near_flags_differing"] + c["ctx_near_differs"] for c in O["counts"].values()),
              "deployed_probe_D1_flags_differing_from_stored": sum(v["probe_D1"]["n_flags_differ"] for v in dv.values()),
              "deployed_edit_flags_differing_from_stored_vendor_pass": {g: [v[m]["n_flags_differ"] for m in ("rel", "tgen", "igen")] for g, v in dv.items()},
              "deployed_rel_exact_B1_vs_stored": {g: [round(v["rel"]["B1_rows"], 1), round(v["rel"]["stored_per_edit"], 1)] for g, v in dv.items()},
              "counts": {g: {k: c[k] for k in ("n_probe_rows", "n_probe_valid", "n_hd_all", "n_hd_valid", "n_repair_valid", "n_near_rows", "max_D1_without_firing")} for g, c in O["counts"].items()}}
pq = json.load(open(f"{_RP.ROOT}/02_research/experiments/10_final/e3_qwen/out/paper_numbers.json"))
rb = [(k, v) for k, v in pq.items() if k.endswith("RebaseErrMedian") or k.endswith("LoggedErrMedian")]
c3["qwen"]["rebase_error_medians_recomputed_pct"] = {g: [round(v["median_rebased_pct"], 2), round(v["median_logged_pct"], 1)] for g, v in misc["rebase"].items() if g.startswith("TB-Q")}
c3["qwen"]["rebase_paper_numbers_range"] = [min(v for k, v in rb if "Rebase" in k), max(v for k, v in rb if "Rebase" in k), min(v for k, v in rb if "Logged" in k), max(v for k, v in rb if "Logged" in k)]

# ---------------------------------------------------------------- check 1
c1 = {bb: {k: O["discipline"][k] for k in ("n_cells", "bad_cells", "n_failed_or_skipped_files", "digest_gate_vs_behaviour_differ", "internal_logs_mentioned", "out_internal_dir_exists", "logs_in_results")}
      | {"jobs_per_log": {g: v["n_jobs"] for g, v in O["discipline"]["jobs_per_log"].items()}} for bb, O in inn["backbones"].items()}
disagree = [r for r in cl if not r["agree"]]

claims = [
 {"id": "Q1", "claim": "Qwen: all seven logs finished, 0 missing / failed cells, hook_mismatch 0, development log = sanity run e0", "verdict": "CONFIRMED",
  "evidence": f"{c1['qwen']['n_cells']} cells present with alpha 0.05, n 200, one variant per label, driver 6a71f0d2cd71; one job record per log; no FAILED / SKIPPED file; e0: {c3['qwen']['dev_vs_e0']}"},
 {"id": "Q2", "claim": "Qwen main table (7 logs x 5 methods: Rel / T-Gen / I-Gen exact, D1 on heldout_disjoint, D1 on hard negatives)", "verdict": "CONFIRMED",
  "evidence": "all 35 rows reproduced from the per-row files; draw-0 flags of rescope / rescope_null / cosine_shared reproduced row for row by my own forwards on TB-Q500-ae-l2_3, TB-Q500-memitw, TB-Q2000-memitw"},
 {"id": "Q3", "claim": "Qwen: far locality holds (gate firing 2.3-4.5 % over 20 draws, bound 4.95 %; Rel exact >= 99.5)", "verdict": "CONFIRMED",
  "evidence": "my own rule on the raw key files, 20 draws, flag for flag; not said in the summary: single draws exceed 4.95 % on 0-8 of 20 draws per log, and near-scope held-out probes are released on 5-23 % (behaviour)",
  "detail": {"draws_above_bound": ex["Qwen: draws above 4.95 % of 20, rescope"], "near_scope_probes_released_beh_gate": ex["Qwen: rescope, near-scope held-out probes released (behaviour %) / gate level"]}},
 {"id": "Q4", "claim": "Qwen: H2 not confirmed (tie with white_0.1_shared on AlphaEdit-style logs, loss on every MEMIT-style log; later edits; gate-level coverage form)", "verdict": "CONFIRMED",
  "evidence": "differences and intervals recomputed from the per-row files with my own crossed bootstrap (all within 0.2 points of the reported intervals)"},
 {"id": "Q5", "claim": "Qwen: H3 pair 1 +1.2 / +5.1 on AlphaEdit-style, -11.3 to -4.9 on MEMIT-style", "verdict": "PARTLY CONFIRMED",
  "evidence": "numbers reproduced; the summary omits pair 2 (rescope_x minus white_1_x) and the retrieval reference, both in the files",
  "detail": {"pair2": ex["Qwen H3 pair 2 (rescope_x minus white_1_x), behaviour"], "retr": ex["Qwen H3 against retr_sum (rescope minus retr_sum), behaviour"]}},
 {"id": "Q6", "claim": "Qwen: retr_sum ahead by 28-55 points; coverage and retention decay; capped edits 5 -> 60 (rescope_x 220); literal rule fires identically; where_logonly degrades MEMIT logs; MME-T of the null variants",
  "verdict": "CONFIRMED", "evidence": "every number recomputed (claims.json)"},
 {"id": "Q7", "claim": "Qwen anomalies: baseline rows equal on both families at equal T (attributed, untested); cosine_published = cosine_default; one negative margin (-0.012); rebase error 0.8-1.1 % against 11-58 %",
  "verdict": "CONFIRMED", "evidence": "stored solver keys of the two families are bitwise equal and the rebased values differ by at most 1.3 % (gate20.json); my forwards of the all-positions cosine gate: "
  f"{pub['all_positions']['n_fired_before_last_prompt']} + {pub['all_positions']['n_fired_after_last_prompt']} releases away from the last prompt position on {pub['n']} sequences, 0 rows differ; margins and rebase error recomputed",
  "detail": {"families_at_equal_T": g20["families_at_equal_T"], "rebase_pct_rebased_logged": c3["qwen"]["rebase_error_medians_recomputed_pct"]}},
 {"id": "L1", "claim": "LLaVA: all six logs finished, nothing failed, hook_mismatch 0, development log = sanity run e0", "verdict": "CONFIRMED",
  "evidence": f"{c1['llava']['n_cells']} cells present, same settings on every log; e0: {c3['llava']['dev_vs_e0']}"},
 {"id": "L2", "claim": "LLaVA main table (6 logs x 6 methods)", "verdict": "CONFIRMED",
  "evidence": "all 36 rows reproduced from the per-row files and gate cells; draw-0 flags reproduced by my own forwards on TB-L500-ae-l2_3, TB-L500-memitw, TB-L1000-memitw at both gate positions"},
 {"id": "L3", "claim": "LLaVA: reliability and far locality hold (Rel 98-99, D1 2-3 %, mean firing below 4.95 % on every log; 9 of 20 draws above on L100-memitw)", "verdict": "CONFIRMED", "evidence": "20 draws with my own rule, flag for flag"},
 {"id": "L4", "claim": "LLaVA: text rephrases weak and worse with T; whitened cosine matches or beats rescope; retr_sum far ahead", "verdict": "PARTLY CONFIRMED",
  "evidence": "numbers reproduced; imprecise: retr_sum T-Gen is 89 on L100-memitw (summary: 58-83), HN-I firing 37-60 % (summary: 40-60); the summary gives no paired H2 / H3 difference, and the experiment computed none",
  "detail": {"H2_mine": ex["LLaVA H2 (rescope minus white_0.1_shared, rephrase exact, all edits; my bootstrap)"], "H3_beh_pair1_pair2_retr_mine": ex["LLaVA H3 pair 1 / pair 2 / retr, behaviour"],
             "rescope_minus_deployed_rephrase_exact": ex["LLaVA: rescope minus DEPLOYED, rephrase exact"]}},
 {"id": "L5", "claim": "LLaVA: H3 gate table (4 logs x 5 methods)", "verdict": "CONFIRMED", "evidence": "20 cells reproduced from the per-edit flags; no unreachable draw"},
 {"id": "L6", "claim": "LLaVA: the bound is void for many memitw edits (19 / 91 / 149 capped, rescope_x 24 / 171 / 411); smallest self-separation margin 0.02 on memitw", "verdict": "PARTLY CONFIRMED",
  "evidence": "capped counts confirmed; the margin statement is wrong for TB-L1000-memitw: smallest margin -0.107, two edits with a non-positive margin (the experiment's own files and paper numbers hold -0.107)",
  "detail": ex["LLaVA: self-separation margin min / edits with a non-positive margin"]},
 {"id": "L7", "claim": "LLaVA: rescope_x cuts hard-negative D1 to 4-8 % at a cost of 15-20 points of I-Gen", "verdict": "PARTLY CONFIRMED", "evidence": "D1 confirmed; the I-Gen cost is 5 points on L100-memitw and 16-21 elsewhere"},
 {"id": "L8", "claim": "LLaVA: retention; MME-T firing above the budget on the ae logs; counts (non-oracle targets, 12 invalid probes, shifted sequences); where_logonly at T = 1000", "verdict": "CONFIRMED", "evidence": "claims.json"},
]
problems = [
 {"where": "e3_llava/code/e3_analyze.py (behaviour(), lines 305-442; paper(), line 553)", "problem": "no paired comparison is computed: H2 and H3 exist only as per-method levels (tables 1 and 4), no difference, no paired interval "
  "(PROTOCOL section 4, R6, R10, R25 (3)); e3_qwen/code/e3_analyze.py:472 (endpoints) does it for Qwen", "effect": "no number is wrong; the LLaVA H2 / H3 differences and intervals are missing from results.json and paper_numbers.json",
  "fix": "add the paired crossed-bootstrap differences (my values: claims_extra.json)"},
 {"where": "LLaVA summary (text), not the files", "problem": "'smallest self-separation margin is 0.02 on memitw' ignores TB-L1000-memitw (-0.107, two edits non-positive)", "effect": "understates the log-level warning sign at T = 1000",
  "fix": "quote paper number EthreeLMeThousandMarginMin"},
 {"where": "e3_qwen/code/e3_analyze.py, e3_llava/code/e3_analyze.py", "problem": "no exact McNemar test for the binary paired comparisons (PROTOCOL section 4); no Holm correction (left to the coordinator; said only in the Qwen report)",
  "effect": "the per-log H2 / H3 read-outs are uncorrected and descriptive", "fix": "compute from the per-row flags when the confirmatory family is assembled"},
 {"where": "e3_qwen/code/e3_render.py:275 (MAIN TABLE, COMPACT; table.txt lines 9-139) and both summaries", "problem": "AlphaEdit-style and MEMIT-style logs ([redacted: describes a related submission under double-blind review]) are listed in one table (amendment of 2026-11-10 17:40, item iii); "
  "the baselines' rows are identical in both families at equal T by construction (equal solver keys)", "effect": "presentation only", "fix": "one table per family in the paper; state that baseline rows do not depend on the family"},
 {"where": "e3_qwen/code, e3_llava/code", "problem": "E3 as registered (section 3) lists gate latency and storage against T; neither experiment measures them", "effect": "none on the reported numbers", "fix": "take them from E8 or say so"},
 {"where": "e3_qwen/code/e3_gate.py (modified 02:49 UTC), e3_render.py (05:35), e3_analyze.py (01:28); e3_llava/code/e3_analyze.py, e3_render.py (03:06)", "problem": "scripts were edited while or after the jobs ran; there is no version history",
  "effect": "none found: the registered gate cells predate the edit, no cell was computed twice (one job record per log, no second settings), and the delivered scripts reproduce results.json, paper_numbers.json and table.txt exactly",
  "fix": "none needed; keep the code frozen"},
 {"where": "both experiments", "problem": "the deployed log's Rel / T-Gen / I-Gen are B = 1 values and differ from the testbed's stored vendor-pass flags (padded batches) on 0-13 edits per metric (largest: TB-Q500-ae-l2_3 Rel 54.6 against 56.4 stored); "
  "D1 equals the stored flags on every row", "effect": "documented limitation of the stored metrics; both values are in results.json", "fix": "the paper's table of logs must use the B = 1 values"},
]
missing_from_summaries = {
 "qwen": ["near-scope held-out probes: rescope releases an edit on 5-23 % (behaviour; 6-27 % gate level)", "single calibration draws above 4.95 % on five of six unseen logs (3-8 of 20)",
          "H3 pair 2 (rescope_x minus white_1_x) negative on every MEMIT-style log (-9.3 to -2.9); H3 against retr_sum -15 to -37",
          "the registered variant rescope_null beats the default by 7.5-9.0 points of rephrase exact on the MEMIT-style logs with 100-1000 edits; the shared-scale ablation beats it by 4.9 on TB-Q500-ae-l2_3"],
 "llava": ["H2 as a difference: -31.0, -17.8, -10.2 points on the MEMIT-style logs (T = 100, 500, 1000), +1.2 / -1.1 on the AlphaEdit-style logs",
           "behavioural H3 pair 1 negative on all five unseen logs (-2.7 to -23.3)", "rephrase exact of rescope is 10.8 points BELOW the deployed log on TB-L100-memitw",
           "negative self-separation margin at T = 1000 (two edits)", "BEN-T far firing above the level for rescope_x (6.0 % on L200-ae) and rescope_shared (5.6-5.8 % on L100 / L200-ae)"]}
out = {"experiment": "E3 (e3_qwen, e3_llava)", "written": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "verifier_directory": V,
       "checks": {"1_protocol_discipline": {"verdict": "CONFIRMED with reporting problems", "detail": c1},
                  "2_own_recomputation": {"verdict": "CONFIRMED", "draw0_own_forwards": tot, "draw0_rows": rows2, "all_20_draws_own_rule": t20,
                                          "method": "plain rescope.Evaluator, keys captured here, thresholds by my own numpy rule; no driver, no memo, no key cache (the cache is used only for gate-level coverage "
                                                    "and the 20-draw recomputation, after a bitwise check against the keys captured here)"},
                  "3_internal_consistency": {"verdict": "CONFIRMED", "detail": c3},
                  "limits_of_this_verification": "own forwards and own thresholds cover rescope, rescope_null and cosine_shared (as tasked); the rows of the other methods (white_*, retr_sum, rescope_x, "
                                                 "where_logonly, cosine, lookup, exact, deployed) were recomputed from the experiments' stored per-row files, not from forwards of my own; BEN-T / MME-T firing "
                                                 "was read from the gate cells",
                  "4_unfavourable_items_missing_from_the_summaries": {"verdict": "summaries correct but incomplete", "items": missing_from_summaries}},
       "claims": claims, "summary_numbers_checked": {"n": len(cl), "n_agree": sum(r["agree"] for r in cl), "disagreeing": disagree},
       "problems": problems,
       "must_fix_before_paper": ["compute the LLaVA paired H2 / H3 differences with intervals (missing in e3_llava)", "correct the LLaVA margin statement (T = 1000: -0.107, two edits)",
                                 "split every table by log family; no side-by-side of the two families", "McNemar and Holm for the confirmatory comparisons, from the stored per-row flags",
                                 "state next to 'far locality': releases on near-scope held-out probes (3-23 %), single draws above the level, and that the bound is void for capped edits (up to 60 / 149 for the default method)",
                                 "report H3 pair 2 and the retrieval reference next to pair 1"],
       "files": {k: f"{V}/out/{k}" for k in ("compare.json", "compare_table.txt", "gate20.json", "gate20_table.txt", "internal.json", "mine_qwen.json", "mine_llava.json", "rerender.json", "claims.json",
                                              "claims_table.txt", "claims_extra.json", "misc.json", "pub_TB-Q500-ae-l2_3.json")}}
json.dump(out, open(V + "/verdicts.json", "w"), indent=1)
L = ["E3 VERIFICATION: verdicts (from verdicts.json)"]
for c in claims:
    L.append(f"{c['id']:3s} {c['verdict']:17s} {c['claim']}")
L.append("")
L.append(f"check 2, draw 0, own forwards: {tot}")
L.append(f"check 2, 20 draws, own rule: {t20}")
L.append(f"summary numbers: {sum(r['agree'] for r in cl)} of {len(cl)} agree; disagreeing: " + "; ".join(f"{r['quantity']} | {r['experiment']} | {r['mine']}" for r in disagree))
for bb in c3:
    L.append(f"check 3 {bb}: " + json.dumps({k: v for k, v in c3[bb].items() if k not in ("counts", "deployed_edit_flags_differing_from_stored_vendor_pass", "deployed_rel_exact_B1_vs_stored", "rebase_error_medians_recomputed_pct")}))
open(V + "/out/verdict_table.txt", "w").write("\n".join(L) + "\n")
print("\n".join(L))
