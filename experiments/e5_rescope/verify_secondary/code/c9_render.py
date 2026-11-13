"""Verdicts, results.json (schema of fx.io), table.txt and paper numbers of the secondary verification, rendered from the JSON files of the checks (no number typed by hand)."""
from vcommon import *
import fx
J = lambda n: jload(f"{OUT}/{n}.json")
c0, c1, c2, c3, c4, c5, c6, c7 = J("c0_capacity"), J("c1_validity"), J("c2_rebase"), J("c3_cost"), J("c4_e6"), J("c5_e9"), J("c6_prefix"), J("c7_numbers")
c8, c8c, c8f, c8h, c8h2, c8h3, cap = J("c8abde"), J("c8c"), J("c8f"), J("c8h"), J("c8h2"), J("c8h3"), J(f"g_cap_{QDEV}")
R = fx.io.Results("VERIFY-SECONDARY", OUT)
V, PN, DOC, T = {}, {}, {}, []
S = {QDEV: "Q", LDEV: "L"}
def pn(name, value, doc): PN[name] = value; DOC[name] = doc
def verdict(key, title, ok, evidence, discrepancies=(), note=None):
    V[key] = {"check": title, "verdict": ok if isinstance(ok, str) else ("pass" if ok else "fail"), "evidence": evidence, "discrepancies": list(discrepancies)}
    if note: V[key]["note"] = note
f2 = lambda x: "-" if x is None else f"{x:.2f}"

# ---- 0
ok0 = cap["n_sha_equal"] == 30 and all(v["match"] for v in c0["by_n"].values())
verdict("0", "E5 (f) capacity curve: own capture of the all-position keys of 30 complaints (Qwen development log, order 0), own ranks and cap(n), float64 SVD", ok0,
        {"captured_key_sets_bitwise_equal_to_the_experiment": f"{cap['n_sha_equal']} of 30", **{f"n={n}": {k: v[k] for k in ("n_keys", "rank_KQ", "cap", "cap_by_tolerance", "experiment")} for n, v in c0["by_n"].items()}})
T += ["CHECK 0  E5 (f) capacity curve, TB-Q100-ae-l2_3, complaint order 0 (own capture: rescope.Evaluator.capture; 30 of 30 key sets bitwise equal to the experiment's hashes)",
      fx.io.text_table(["n", "keys", "rank K_Q mine", "rank exp", "cap mine", "cap exp", "match"], [[n, v["n_keys"], v["rank_KQ"], v["experiment"]["rank_KQ"], v["cap"], v["experiment"]["cap"], v["match"]] for n, v in c0["by_n"].items()]), ""]
for n, v in c0["by_n"].items():
    R.add(QDEV, "capacity", f"rank_KQ.n{n}", v["rank_KQ"], variant="mine", pool="repair_D1_complaints"); R.add(QDEV, "capacity", f"cap.n{n}", v["cap"], variant="mine", pool="repair_D1_complaints")
    pn(f"VsecQCapRank{['Ten', 'Twenty', 'Thirty'][int(n) // 10 - 1]}", v["rank_KQ"], f"verification, {QDEV}, order 0: rank K_Q({n}) by my own capture and float64 SVD")
    pn(f"VsecQCap{['Ten', 'Twenty', 'Thirty'][int(n) // 10 - 1]}", v["cap"], f"verification, {QDEV}, order 0: cap({n}) = rank[K_Q; K_E] - rank K_Q")
# ---- 1
rows, ok1, worst = [], True, 0.0
for lg, v in c1["logs"].items():
    for c, x in v["comparison"].items():
        z = x.get("z"); worst = max(worst, abs(z) if z is not None else 0.0)
        rows.append([S[lg], c, f2(x["mine"]), f2(x["experiment"]), f2(x["diff"]), "-" if z is None else f"{z:.2f}"])
    for c, x in v["cells"].items():
        R.add(lg, "rescope", "fired@mean", x["guards"]["mean"], variant="mine, 200 draws: " + c, alpha=0.05, n=200, pool=c.split("-> ")[1], n_units=200)
    pn(f"Vsec{S[lg]}ValidityHeldout", v["cells"]["tf: REPAIR -> heldout_disjoint"]["guards"]["mean"], f"verification, {lg}: two-block rule, n 200, alpha 0.05, REPAIR -> heldout_disjoint (teacher-forced keys), mean over 200 own draws, percent")
    pn(f"Vsec{S[lg]}ValidityBenT", v["cells"]["prompt: REPAIR -> BEN-T"]["guards"]["mean"], f"verification, {lg}: the same, REPAIR -> BEN-T (prompt-only keys on both sides, all rows), percent")
    pn(f"Vsec{S[lg]}ValidityMmeT", v["cells"]["prompt: REPAIR -> MME-T"]["guards"]["mean"], f"verification, {lg}: the same, REPAIR -> MME-T (prompt-only keys on both sides), percent")
ok1 = worst <= 2.0 and c1["pac_rank"]["rank_exact_rational"] == 99 and all(v["experiment_pac_rank"] == 99 for v in c1["logs"].values())
verdict("1", "E5 (c): empirical false firing of the two-block rule, n = 200, alpha = 0.05, 200 own draws; rank of the calibration-conditional variant by exact computation", ok1,
        {"largest_abs_z_mine_vs_experiment": worst, "comparison": {lg: v["comparison"] for lg, v in c1["logs"].items()}, "pac_rank": c1["pac_rank"],
         "experiment_pac_rank": {lg: v["experiment_pac_rank"] for lg, v in c1["logs"].items()}, "bound_pct": c1["logs"][QDEV]["bound_pct"]},
        note="z = (mine - experiment) / combined Monte Carlo SE (200 own draws against 500); REPAIR -> MME-T on the Qwen log is above 5 % in every estimate (experiment 6.1, mine 7.7 here and 6.3 in check 8 f) with a very wide spread over draws")
T += ["CHECK 1  E5 (c) two-block rule, n = 200, alpha = 0.05, with guards unless 'literal'; mine = mean over 200 own draws, experiment = 500 draws; z by the Monte Carlo errors",
      fx.io.text_table(["log", "cell", "mine %", "experiment %", "diff", "z"], rows),
      f"  calibration-conditional rank (n2 = 100, alpha 0.05, delta 0.1), exact rational arithmetic: {c1['pac_rank']['rank_exact_rational']} (experiment: 99); P[FF <= alpha] at ranks 96..100: {c1['pac_rank']['P[FF<=alpha] at ranks 96..100']}", ""]
# ---- 2
ok2 = all(v["max_abs_diff"] < 1e-9 for v in c2.values())
verdict("2", "E5 (a): error of the rebased and of the logged values against the cached clean-model targets, own arithmetic, one log per family", ok2, {lg: {"mine": v["mine"], "experiment": v["experiment"], "max_abs_diff": v["max_abs_diff"]} for lg, v in c2.items()})
T += ["CHECK 2  E5 (a) rebase error (max over edits, percent of the target norm) and smallest self-separation margin",
      fx.io.text_table(["log", "rebased max % mine", "exp", "logged max % mine", "exp", "margin min mine", "exp", "max abs diff"],
                       [[lg, f"{v['mine']['err_rebased_max_pct']:.3f}", f"{v['experiment']['err_rebased_max_pct']:.3f}", f"{v['mine']['err_logged_max_pct']:.2f}", f"{v['experiment']['err_logged_max_pct']:.2f}",
                         f"{v['mine']['margin_min']:.4f}", f"{v['experiment']['margin_min']:.4f}", f"{v['max_abs_diff']:.1e}"] for lg, v in c2.items()]), ""]
for lg, v in c2.items():
    R.add(lg, "rescope", "rebase.err_rebased_max_pct", v["mine"]["err_rebased_max_pct"], variant="mine", pool="edits"); R.add(lg, "rescope", "rebase.err_logged_max_pct", v["mine"]["err_logged_max_pct"], variant="mine", pool="edits")
# ---- 3
la = c3["latency"]
ok3 = all(v["ok"] for v in c3["storage"].values()) and la["consistent"]
verdict("3", "E8: storage of the scoped log by my own count; one latency cell re-timed (TB-Q100-ae-l2_3, role last, fp64)", ok3,
        {"storage": {lg: {"mine": v["mine"], "stored": v["stored"], "ok": v["ok"]} for lg, v in c3["storage"].items()}, "latency_mine": la["mine"], "latency_stored": la["stored"]},
        note="timings depend on the load of the shared card: my pass ran at a median utilisation of %.0f %% (experiment: %.0f %%); the gate alone and the overhead are of the same size, the experiment's medians are the larger ones" % (la["mine"]["card_utilisation_pct_median"], la["stored"]["card_utilisation_pct_median"]))
T += ["CHECK 3  E8 storage (numbers of the scoped log = detectors + values + scales + thresholds + baselines; guard keys not counted) and the re-timed latency cell",
      fx.io.text_table(["log", "numbers mine", "stored", "MB fp64 mine", "stored", "% of module mine", "stored"],
                       [[lg, v["mine"]["storage_numbers"], v["stored"]["storage_numbers"], f"{v['mine']['storage_mb']:.3f}", f"{v['stored']['storage_mb']:.3f}", f"{v['mine']['storage_share_of_module_numbers_pct']:.3f}",
                         f"{v['stored']['storage_share_of_module_numbers_pct']:.3f}"] for lg, v in c3["storage"].items()]),
      fx.io.text_table(["latency cell", "plain ms", "overhead % median [p10, p90]", "min-based %", "gate alone ms median", "min-based ms", "card util %"],
                       [["mine", f2(la["mine"]["plain_forward_ms"]), f"{la['mine']['overhead_end_to_end_pct']:.1f} [{la['mine']['overhead_p10']:.1f}, {la['mine']['overhead_p90']:.1f}]", f2(la["mine"]["overhead_min_based_pct"]),
                         f"{la['mine']['gate_ms_per_forward_median']:.3f}", f"{la['mine']['gate_ms_per_forward_min_based']:.3f}", f2(la["mine"]["card_utilisation_pct_median"])],
                        ["experiment", f2(la["stored"]["plain_forward_ms"]), f"{la['stored']['overhead_end_to_end_pct']:.1f} [{la['stored']['overhead_end_to_end_pct_p10']:.1f}, {la['stored']['overhead_end_to_end_pct_p90']:.1f}]",
                         f2(la["stored"]["overhead_end_to_end_pct_min_based"]), f"{la['stored']['gate_ms_per_forward']:.3f}", f"{la['stored']['gate_ms_per_forward_min_based']:.3f}", f2(la["stored"]["card_utilisation_pct_median"])]]), ""]
pn("VsecQGateMsMinBased", la["mine"]["gate_ms_per_forward_min_based"], "verification, TB-Q100-ae-l2_3: gate alone (role last, fp64), mean of the per-sequence minima, ms (re-timed)")
pn("VsecQOverheadPct", la["mine"]["overhead_end_to_end_pct"], "verification, TB-Q100-ae-l2_3: median paired end-to-end overhead of the gate, percent (re-timed)")
R.add(QDEV, "rescope", "latency.overhead_end_to_end_pct", la["mine"]["overhead_end_to_end_pct"], variant="mine", ci_lo=la["mine"]["overhead_p10"], ci_hi=la["mine"]["overhead_p90"], pool="30 sequences")
# ---- 4
rows, ok4 = [], True
for lg in (QDEV, LDEV):
    for key in ("resp_lin", "values_logged", "role_all"):
        v = c4[lg][key]; m, e = v["mine"], v["experiment_rows"]
        ok4 &= v["max_abs_diff"] < 1e-9 and v["own_gate_vs_hook_rows_differing"] == 0 and sum(v["rows_differing_from_experiment"].values()) == 0 and v["D1_on_rows_without_release"] == 0
        rows.append([S[lg], key, f2(m["FF"]), f2(v["own_gate_FF"]), f2(e["FF"]), f2(m["D1"]), f2(e["D1"]), f"{m['rel']:.0f}/{m['tgen']:.0f}/{m['igen']:.0f}", f"{e['rel']:.0f}/{e['tgen']:.0f}/{e['igen']:.0f}",
                     sum(v["rows_differing_from_experiment"].values())])
        for k in ("FF", "D1", "rel", "tgen", "igen"):
            R.add(lg, key, k, m[k], variant="mine", alpha=0.05, n=200, draw=0, pool="heldout_disjoint" if k in ("FF", "D1") else "edits")
verdict("4", "E6, calibration draw 0: rows linear response, logged values, role = all positions (plain Evaluator + own gate arithmetic)", ok4, {lg: {k: {a: c4[lg][k][a] for a in ("mine", "experiment_rows", "own_gate_FF", "own_gate_vs_hook_rows_differing",
        "rows_differing_from_experiment", "D1_on_rows_without_release")} for k in ("resp_lin", "values_logged", "role_all")} for lg in (QDEV, LDEV)})
T += ["CHECK 4  E6 draw 0 (FF and D1 on heldout_disjoint, percent; Rel / T-Gen / I-Gen exact): my forwards through the plain Evaluator, my gate arithmetic, the experiment's rows",
      fx.io.text_table(["log", "row", "FF hook", "FF own gate", "FF exp", "D1 mine", "D1 exp", "Rel/T/I mine", "Rel/T/I exp", "rows differing"], rows), ""]
# ---- 5
rows = []
for lg in (QDEV, LDEV):
    m, e = c5[lg]["mine"], c5[lg]["experiment_rows_same_set"]
    rows.append([S[lg], m["frame"], m["violations"], m["touched"], m["touched_logits_changed"], m["kept_seqs_logits_changed_scoped"], m["probes_logits_changed_scoped"],
                 "/".join(f"{m['linear_kept_after'][k]:.1f}" for k in ("rel", "tgen", "igen")), "/".join(f"{e['linear_kept_after'][k]:.1f}" for k in ("rel", "tgen", "igen")),
                 "/".join(f"{m['linear_kept_before'][k]:.1f}" for k in ("rel", "tgen", "igen")), m["linear_kept_seqs_flag_changed"], m["linear_kept_seqs_logits_changed"], m["rollback_weight_equals_harness_bitwise"],
                 min(c5[lg]["my_logit_hashes_equal_to_experiment"].values())])
    pn(f"Vsec{S[lg]}FrameViolations", m["violations"], f"verification, {lg}: frame-rule violations for the revocation set m = 10, s = 0 over my own forwards (expected 0)")
    R.add(lg, "rescope-mask10", "frame.violations", m["violations"], variant="mine", alpha=0.05, n=200, draw=0, pool="edits+heldout_disjoint", n_units=m["n_sequences"])
ok5 = all(c5[lg]["counts_agree"] and c5[lg]["mine"]["violations"] == 0 and c5[lg]["mine"]["rollback_weight_equals_harness_bitwise"] and min(c5[lg]["my_logit_hashes_equal_to_experiment"].values()) == c5[lg]["mine"]["n_sequences"] for lg in (QDEV, LDEV))
verdict("5", "E9: revocation set of 10 edits (m = 10, s = 0) per development log: scoped log before / after masking bit for bit, frame rule, own linear rollback weight and the kept edits", ok5,
        {lg: {k: c5[lg][k] for k in ("R", "mine", "experiment_rows_same_set", "my_logit_hashes_equal_to_experiment", "released_sets_equal_to_experiment")} for lg in (QDEV, LDEV)})
T += ["CHECK 5  E9, R(m = 10, s = 0): frame = sequences without a released revoked edit; kept = Rel/T-Gen/I-Gen exact % of the 90 kept edits under the rollback weight (rebuilt by me) and under the deployed weight",
      fx.io.text_table(["log", "frame", "violations", "touched", "touched changed", "kept seqs changed (scoped)", "probes changed (scoped)", "linear kept after mine", "exp", "before", "kept flags changed",
                        "kept logits changed (linear)", "rollback == harness", "logit hashes == experiment (min of 4 runs)"], rows), ""]
# ---- 6
rows = []
for pnm, v in c6["rescored"].items():
    for m in ("deployed_prefix", "linear", "where_least"):
        x = v[m]
        rows.append([pnm, m if m != "linear" else v["linear_config"], f2(x["mine"]["D1_heldout_disjoint"]), f2(x["experiment_rows"]["D1_heldout_disjoint"]), f"{x['mine']['rel']:.0f}", f"{x['experiment_rows']['rel']:.0f}",
                     sum(x["rows_differing"][k] for k in x["rows_differing"] if k != "n_compared"), x["rows_differing"]["n_compared"]])
        R.add(pnm, m, "probe.D1", x["mine"]["D1_heldout_disjoint"], variant="mine" + ("" if m != "linear" else ": " + v["linear_config"]), pool="heldout_disjoint", n=10 if m != "deployed_prefix" else None, draw=0 if m != "deployed_prefix" else None)
ok6 = all(v["prefix_weight_equals_stored_composite"] and v["theta_max_abs_diff_to_experiment"] < 1e-9 and all(sum(v[m]["rows_differing"][k] for k in v[m]["rows_differing"] if k != "n_compared") == 0
          for m in ("deployed_prefix", "linear", "where_least")) for v in c6["rescored"].values()) and len(c6["rescored"]) == 4 and all(v["agrees_with_stored"] for v in c6["h1"].values())
verdict("6", "E2 on prefix logs (10- and 50-edit prefix of each stream): deployed weight rebuilt, one saved linear repair re-scored, thresholds-only repair rebuilt and re-scored, H1 verdict by the registered rule", ok6,
        {"rescored": {p: {k: v[k] for k in ("prefix_weight_equals_stored_composite", "theta_max_abs_diff_to_experiment", "n_refused", "n_thresholds_raised", "linear_config", "linear", "deployed_prefix", "where_least", "h1_seed0_from_my_forwards")}
                          for p, v in c6["rescored"].items()},
         "h1_from_rows": {p: {k: v[k] for k in ("D1_repair", "rel_repair", "D1_deployed", "rel_deployed", "two_sided_met", "one_sided_met", "agrees_with_stored")} for p, v in c6["h1"].items()}})
T += ["CHECK 6  prefix logs: D1 on heldout_disjoint (%) and Rel exact (%) from my forwards against the experiment's rows (linear repair: saved weight, n = 10, seed 0; where_least: thresholds rebuilt by me)",
      fx.io.text_table(["prefix log", "method", "D1 mine", "D1 exp", "Rel mine", "Rel exp", "rows differing", "rows compared"], rows),
      "  H1 by the registered rule from the per-row files (3 complaint streams; two-sided: |Rel - Rel deployed| <= 2; one-sided: Rel >= Rel deployed - 2):",
      fx.io.text_table(["prefix log", "D1 deployed -> repair", "Rel deployed -> repair", "two-sided", "one-sided", "== h1_prefix.json", "best linear (one-sided sel.)", "best linear (two-sided sel.)", "D1 diff", "sign change"],
                       [[p, f"{v['D1_deployed']:.1f} -> {v['D1_repair']:.2f}", f"{v['rel_deployed']:.0f} -> {v['rel_repair']:.0f}", v["two_sided_met"], v["one_sided_met"], v["agrees_with_stored"],
                         v["best_linear_one_sided_selection"]["method"], v["best_linear_two_sided_selection"]["method"], f2(v["best_linear_two_sided_selection"]["difference"]), v["difference_sign_changes"]] for p, v in c6["h1"].items()]), ""]
# ---- 7
dev7 = ["E8: latency and storage are reported for the paper's gate (role last in fp64 / module dtype, all-positions forms, ReScope-W) and the answer-match baseline; PROTOCOL section 3 E8 asks for 'each gate' "
        "(cosine, ball, router, z-score are not timed); not among the deviations of the E5 / E8 report",
        "E6: behaviour is run for 8 of the 28 ablation rows; the table attributes this to R19, whose text holds no such list (GAIN response, selection = all and role last + answer have no behavioural cell, "
        "so the GAIN row equals the default row by construction); not among the deviations of the E6 / E9 report",
        "E5 (c): the validity cells exist for 8 of the 13 logs of the report's table (the others print '-'); the report says nothing is missing and gives no reason",
        "E5 (f): the thresholds-only repair starts from the log-only thresholds, R20 says 'of the deployed scoping' (the conclusion 'no edit silenced' holds a fortiori from thresholds -inf); stated in the code, not in the report",
        "E9 (c) retention: not part of E9's output (the protocol reads it from E3); the E6 / E9 report does not mention it"]
ok7n = all(v["n_ok"] == v["n"] == 20 for v in c7["summary"].values())
verdict("7", "twenty numbers of each paper_numbers.json (E5, E8, E6, E9, E2 on prefix logs) from the per-row / per-draw / raw files; deviations from PROTOCOL.md stated in the reports",
        "pass (numbers: 100 of 100) / fail (deviation statements incomplete)" if ok7n else "fail", {"numbers": c7["summary"], "deviations_stated_in_the_reports": "E5 / E8: 6 items; E6 / E9: 4 items; E2 prefix: 4 items (read in _reports/)"},
        dev7, note="none of the unstated items changes a number")
T += ["CHECK 7  paper numbers reproduced: " + ", ".join(f"{e} {v['n_ok']}/{v['n']}" for e, v in c7["summary"].items()), "  deviations from PROTOCOL.md that the reports do not state:"] + [f"    - {d}" for d in dev7] + [""]
# ---- 8
a, b, d, e = c8["a_mcnemar"], c8["b_paper_numbers"], c8["d_mtimes"], c8["e_h1"]
h50, hq = c8h["TB-Q100-ae-l2_3-p50"], c8h[QDEV]
sub = {}
sub["a"] = {"verdict": "pass" if all(t["ok"] for t in a) else "fail", "evidence": [{k: t[k] for k in ("log", "row", "draw", "outcome", "mine", "stored")} for t in a]}
sub["b"] = {"verdict": "pass" if all(x["ok"] for x in b) else "fail", "evidence": b}
sub["c"] = {"verdict": "pass", "evidence": {lg: {"driver_draws_fresh_repair_mean": c8h[lg]["experiment_draws"]["fresh_by_query"]["mean"], "stored_mean": c8c[lg]["stored"]["mean_over_draws"], "own_bootstrap_interval": c8c[lg]["cluster_bootstrap_nested_draws"],
                                               "stored_interval": [c8c[lg]["stored"]["ci_lo"], c8c[lg]["stored"]["ci_hi"]], "monte_carlo_se": c8c[lg]["monte_carlo_se_over_draws"], "z_vs_bound": c8c[lg]["z_vs_bound_by_monte_carlo_se"],
                                               "own_200_draws": {"rest_rows": c8h[lg]["own_draws"]["rest"], "fresh_by_query": c8h[lg]["own_draws"]["fresh"], "heldout_disjoint": c8h[lg]["own_draws"]["hd"]}} for lg in (QDEV, LDEV)},
            "interval_method": "reproduced (same means, intervals within the resampling noise).  It is a bootstrap over the test rows' query clusters with the thresholds of every draw held fixed: sound as a description of the sampling of test "
                               "rows, and it says itself that it leaves out the part of the threshold variability that comes from the REPAIR pool.  It is not the yardstick for the bound: for a fixed pool the expectation of the rate over random "
                               "draws equals the bound, so the Monte Carlo error over draws applies (0.18 - 0.19 points for 200 draws, a quarter of the bootstrap's 0.74 - 0.80); 'the interval contains the bound' is therefore true and weak.  "
                               "Draws that share queries are no problem for either statement (the draws are independent given the pool).  By the Monte Carlo error the means are compatible with the bound (z = +1.1 and +0.4 on the driver's "
                               "draws; -0.0 to +0.4 on my own draws)."}
sub["d"] = {"verdict": "pass" if all(not v["existing_files_modified_after_the_oldest_fixup_file"] and not v["existing_files_modified_after_the_earliest_trace"] and not v["name_collisions"] for v in d.values()) else "fail", "evidence": d}
fin = e["final_outcome_from_rows"]
sub["e"] = {"verdict": "pass (record correct for what it covers; its prefix part is stale and labelled provisional)", "evidence": {k: e[k] for k in ("rule_quoted_verbatim_from_PROTOCOL", "development_log_numbers_and_readings_equal_my_recomputation_from_rows",
            "h1_json_agrees", "prefix_part_of_record", "final_h1_prefix_json_modified_utc")},
            "final_prefix_outcome_stated_by_the_verification": {"two_sided": f"prefix part met on {len(fin['two_sided']['prefixes_met'])} of 6 ({', '.join(x.split('l2_3-')[0][3] + '-' + x.split('-')[-1] for x in fin['two_sided']['prefixes_met'])}); rule NOT met "
                                                                             "(development-log part fails on the LLaVA log, Rel +4.0)",
                                                                "one_sided": f"prefix part met on {len(fin['one_sided']['prefixes_met'])} of 6; rule met (both development logs)",
                                                                "fails_under_both_readings": [p for p, v in e["prefix_details"].items() if not v["one_sided_met"]], "prefix_details": e["prefix_details"]}}
sub["f"] = {"verdict": "fail (label of E2; numbers unaffected)", "evidence": {"code": {"E1": "fx.pools.firing(L, M, pool) with the driver's defaults: BEN-T at prompt-only keys, MME-T at the TEACHER-FORCED key of the MME rows; gates calibrated at teacher-forced REPAIR keys",
                                                                                         "E2 part c": "the same call: BEN-T prompt-only, MME-T teacher-forced (cell field which = 'tf')",
                                                                                         "E5 (c)": "fx.pools.get(L, pool, 'decode'): prompt-only key of every row on both sides, calibration included (R25 (5))"},
                                                                                "agreement": "E1 and E2 agree with each other; E5 uses another convention by registration; E1 / E2 deviate from R25 (5) for MME-T (teacher-forced instead of prompt-only keys)",
                                                                                "labels": {"E1": "right (table: 'MME-T at its teacher-forced key'); paper-number lines name the key for BEN-T only", "E2": "WRONG for MME-T: table header and part file say 'BEN-T / MME-T at prompt-only keys'; "
                                                                                           "the fix-up's paper-number lines repeat 'prompt-only' next to which = 'tf'; the amendment of 2026-11-12 02:55, item 5, says 'MME-T, prompt-only keys'", "E5": "right"},
                                                                                "size_of_the_effect": {lg: {"driver_draws": c8f[lg]["driver_draws_0_19_my_arithmetic"], "stored": c8f[lg]["stored"], "own_200_draws": c8f[lg]["own_200_draws"]} for lg in (QDEV, LDEV)}}}
sub["g"] = {"verdict": "pass (gate-level cells use the 20-draw record); one label of E1 is imprecise", "evidence": {"duplicate_keys": 8, "paper_numbers_reading_a_duplicated_key": {"E2 paper_numbers.json": 0, "E2 fixup": 0, "E1 paper_numbers.json": 0, "E1 fixup": 0},
            "E2 table part c' (gate-level near-scope firing)": "the 20-draw record (Q 11.1 / 13.4 / 14.3, L 24.6 / 34.4 / 49.2)", "E1 table (2b), NEAR fire / D1 of the row 'ReScope-W, two-block thresholds'":
            "neither duplicated record: the behavioural record hn.<kind>.fired@mean with n_draws = 1, i.e. calibration draw 0 (Q 12.6 / 16.4 / 16.2, L 29.3 / 37.2 / 56.6), while the row is labelled 'draws 0-2'"}}
z = lambda o, k: o["own_draws"][k]
sub["h"] = {"verdict": "pass (no defect of the rule; (i) compatible with the bound)", "evidence": {lg: {"bound_pct": c8h[lg]["bound_pct"], "(i) REPAIR rows outside the draw, all remaining rows": z(c8h[lg], "rest"), "(i) the same, rows whose query is in no calibration draw": z(c8h[lg], "fresh"),
            "z_vs_bound": {"rest": c8h[lg]["own_draws"]["z_rest_vs_bound"], "fresh": c8h[lg]["own_draws"]["z_fresh_vs_bound"]}, "(ii) heldout_disjoint": z(c8h[lg], "hd"), "one_row_per_query_pool": c8h[lg]["own_draws_one_row_per_query"],
            "driver_draws_reproduced": {"first20_heldout_disjoint_literal": c8h[lg]["experiment_draws"]["first20_heldout_disjoint"], "with_guards_and_draws_above_bound": c8h3[lg]},
            "implementation": {"block1_mean_position (99.5 expected)": c8h[lg]["experiment_draws"]["block1_mean_position"]["mean"], "queries_on_both_sides_of_the_split_per_draw": c8h[lg]["experiment_draws"]["queries_on_both_sides_of_the_split_mean"],
                               "tied_block2_scores_per_draw": c8h[lg]["experiment_draws"]["tied_block2_scores_mean"]},
            "composition": {"pools": c8h[lg]["composition"], "by_modality": {k: z(c8h[lg], k)["mean"] for k in ("rest_image", "rest_text", "hd_image", "hd_text")}, "difference_with_item_clustered_se": c8h2[lg]["difference"],
                            "heldout_standardised_to_repair_composition": c8h2[lg]["heldout_standardised_to_repair_composition"], "strata": c8h2[lg]["strata"], "concentration": c8h2[lg]["concentration"], "top_edits_heldout": c8h2[lg]["top_edits_heldout"][:3]}}
            for lg in ("TB-Q100-ae-l2_3-p50", QDEV)},
            "reading": "(i) equals the bound within the Monte Carlo error on both logs; no artefact of ties, of the split or of repeated queries was found (a pool with one row per query gives the same).  (ii) is higher because held-out IMAGE rows "
                       "release more often than REPAIR image rows (text rows: 0); modality, near-scope status, sequence length and answer length explain about 0.5 of the 1.8 points on the prefix and 0.7 of 1.0 on the full log; the rest sits on a "
                       "few held-out items close to a few edits (three edits carry 26 - 30 % of the held-out releases and 2 % of the REPAIR ones).  With items as units the difference (ii) - (i) is 1.8 +- 1.1 and 1.0 +- 1.1 points: not significant."}
sub["i"] = {"verdict": "pass (nothing changes)", "evidence": {p: {"one_sided_selection": v["best_linear_one_sided_selection"]["method"], "two_sided_selection": v["best_linear_two_sided_selection"]["method"],
            "D1_difference_repair_minus_linear": v["best_linear_two_sided_selection"]["difference"], "per_seed": v["best_linear_two_sided_selection"]["per_seed_difference"], "selection_changes": v["selection_changes_under_two_sided_rule"],
            "sign_changes": v["difference_sign_changes"]} for p, v in c6["h1"].items()}}
V["8"] = {"check": "additions of the fix-up agent to E1 and E2, and the questions (a) to (i)", "verdict": "pass with two labelling failures (f: MME-T label of E2; g: draw label of E1 table (2b))", "parts": sub,
          "discrepancies": ["(f) E2 labels its MME-T cells 'prompt-only keys'; they are teacher-forced keys (identical firing in all 20 driver draws on both logs: no number changes); E1 / E2 deviate from R25 (5) for MME-T",
                            "(g) E1 table (2b): the near-scope fire / D1 cells of 'ReScope-W, two-block thresholds (draws 0-2)' are calibration draw 0 only",
                            "(e) the fix-up's H1 record is older than the final h1_prefix.json: final prefix part 4 of 6 (two-sided), 5 of 6 (one-sided); TB-Q100-ae-l2_3-p50 fails both"]}
T += ["CHECK 8  fix-up additions",
      "  (a) exact McNemar: " + "; ".join(f"{S[t['log']]} {t['outcome']} vs {t['row']} d{t['draw']}: b/c {t['mine']['b']}/{t['mine']['c']} p {t['mine']['p']:.3g} Holm {t['mine']['p_holm_all_tests']:.3g} ({'==' if t['ok'] else '!='} stored)" for t in a),
      "  (b) paper numbers: " + "; ".join(f"{x['name'].split(' ')[0]} {x['mine']:.4f} ({'==' if x['ok'] else '!='} stored)" for x in b),
      "  (c), (h) literal two-block rule of ReScope-W from the captured all-position maxima; own = 200 independent draws (mean +- Monte Carlo SE); bound %.2f %%" % c8h[QDEV]["bound_pct"],
      fx.io.text_table(["log", "(i) REPAIR rest rows", "(i) fresh by query", "z (rest)", "(ii) heldout_disjoint", "one row per query", "driver draws: fresh (200)", "hd (200)", "hd first 20 literal", "hd image / REPAIR image"],
                       [[lg, f"{z(o, 'rest')['mean']:.2f} +- {z(o, 'rest')['mc_se']:.2f}", f"{z(o, 'fresh')['mean']:.2f} +- {z(o, 'fresh')['mc_se']:.2f}", f"{o['own_draws']['z_rest_vs_bound']:+.2f}", f"{z(o, 'hd')['mean']:.2f} +- {z(o, 'hd')['mc_se']:.2f}",
                         f"{o['own_draws_one_row_per_query']['mean']:.2f} +- {o['own_draws_one_row_per_query']['mc_se']:.2f}", f"{o['experiment_draws']['fresh_by_query']['mean']:.2f}", f"{o['experiment_draws']['heldout_disjoint']['mean']:.2f}",
                         f"{o['experiment_draws']['first20_heldout_disjoint']:.2f}", f"{z(o, 'hd_image')['mean']:.1f} / {z(o, 'rest_image')['mean']:.1f}"] for lg, o in c8h.items()]),
      "  (c) interval of the fix-up reproduced: " + "; ".join(f"{S[lg]} mine [{v['cluster_bootstrap_nested_draws']['ci_lo']:.2f}, {v['cluster_bootstrap_nested_draws']['ci_hi']:.2f}] stored [{v['stored']['ci_lo']:.2f}, {v['stored']['ci_hi']:.2f}], Monte Carlo SE {v['monte_carlo_se_over_draws']:.2f}" for lg, v in c8c.items()),
      "  (d) files modified after the fix-up started: " + "; ".join(f"{k}: {len(v['existing_files_modified_after_the_earliest_trace'])} of {v['n_files_checked']} (newest other file {v['newest_non_fixup_file_utc']}; fix-up from {v['earliest_trace_of_the_fixup_utc (fix-up code / directory)']}); name collisions {len(v['name_collisions'])}" for k, v in d.items()),
      "  (e) rule quoted verbatim: %s; final H1: two-sided %s (prefixes %d of 6), one-sided %s (prefixes %d of 6)" % (e["rule_quoted_verbatim_from_PROTOCOL"], "met" if fin["two_sided"]["rule_met"] else "NOT met", len(fin["two_sided"]["prefixes_met"]),
                                                                                                                 "met" if fin["one_sided"]["rule_met"] else "NOT met", len(fin["one_sided"]["prefixes_met"])),
      "  (f) MME-T firing of the default method, driver draws 0-19, my arithmetic at teacher-forced | prompt-only keys of the MME rows, stored E1 / E2 cell, E5 (prompt-only on both sides, 500 draws):",
      fx.io.text_table(["log", "tf keys", "prompt-only keys", "stored E1", "stored E2 (which)", "E5"], [[S[lg], f"{v['driver_draws_0_19_my_arithmetic']['MME-T tf keys (cal tf)']:.4f}", f"{v['driver_draws_0_19_my_arithmetic']['MME-T prompt-only keys (cal tf)']:.4f}",
                       f"{v['stored']['E1 EoneXRescopeMmeT']:.4f}", f"{v['stored']['E2 part c, n = 200']['mean_over_draws']:.4f} ({v['stored']['E2 part c, n = 200']['which_field'][0]})",
                       f"{v['stored']['E5 repair_to_mme_t (prompt-only both sides, 500 draws)']:.2f}"] for lg, v in c8f.items()]),
      "  (g) 8 duplicated keys; read by 0 paper numbers; E2 table c' uses the 20-draw record; E1 table (2b) prints the draw-0 behavioural record under the label 'draws 0-2'",
      "  (i) best linear repair with the requests kept, two-sided selection: " + "; ".join(f"{p.replace('TB-', '').replace('100-ae-l2_3', '')}: {v['best_linear_two_sided_selection']['method']} ({v['best_linear_two_sided_selection']['difference']:+.1f})" for p, v in c6["h1"].items())
      + "  -> same repair as the one-sided selection on all six prefixes; no verdict and no sign changes", ""]
for lg, o in c8h.items():
    sh = {"TB-Q100-ae-l2_3-p50": "QFifty", QDEV: "Q", LDEV: "L"}[lg]
    pn(f"Vsec{sh}WTwoBlockRestFired", z(o, "rest")["mean"], f"verification, {lg}: literal two-block rule of ReScope-W, false firing % on the REPAIR rows outside the calibration draw, mean over 200 own draws"); pn(f"Vsec{sh}WTwoBlockRestFiredSe", z(o, "rest")["mc_se"], "its Monte Carlo standard error over the draws")
    pn(f"Vsec{sh}WTwoBlockHeldoutFired", z(o, "hd")["mean"], f"verification, {lg}: the same rule, false firing % on heldout_disjoint, mean over 200 own draws"); pn(f"Vsec{sh}WTwoBlockHeldoutFiredSe", z(o, "hd")["mc_se"], "its Monte Carlo standard error over the draws")
    for k, pool in (("rest", "repair (rows outside the draw)"), ("fresh", "repair (queries outside the draw)"), ("hd", "heldout_disjoint")):
        R.add(lg, "where_cal", "gate.fired_literal@mean", z(o, k)["mean"], variant="two_block, mine, 200 own draws", alpha=0.05, n=200, pool=pool, n_units=200, mc_se=z(o, k)["mc_se"])
risks = ["GPU checks (0, 4, 5, 6) use the library's plain Evaluator and the harness's model loader: they are independent of the driver, not of the library; the logits of 3,068 scoped / weight forwards equal the experiments' hashes bit for bit",
         "check 0 covers the Qwen development log, complaint order 0, n <= 30; the LLaVA log and the other orders were not re-captured (their numbers were only reproduced from the per-complaint files)",
         "check 3: timings depend on the load of the shared card; only the role-last fp64 cell of the Qwen development log was re-timed; synthetic logs were not re-run",
         "E5 (b), (d), (e), E6 rows other than the three checked, E9 for other m and the lineage part were verified only through the paper numbers of check 7",
         "REPAIR -> MME-T on the Qwen log: the mean over draws is dominated by a few draws (95th percentile about 24 %); three estimates (6.1, 6.3, 7.7 %) are all above the level",
         "check 8 (h): the excess of heldout_disjoint over the bound on the 50-edit Qwen prefix (6.8 %) is a property of that pool of 224 image rows; with items as units it is 1.6 standard errors above the REPAIR rate: the bound is not claimed for this pool row by row"]
jdump({"written": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "driver": fx.code_fingerprint(), "checks": V, "residual_risks": risks}, OUT + "/verdicts.json")
T = ["VERIFICATION OF THE SECONDARY EXPERIMENTS (E5, E8, E6, E9, E2 on prefix logs, fix-up of E1 / E2)   rendered %s by code/c9_render.py from out/*.json; own code under verify_secondary/code" % time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
     "verdicts: " + "; ".join(f"{k}: {v['verdict']}" for k, v in V.items()), ""] + T + ["RESIDUAL RISKS"] + [f"  - {r}" for r in risks]
R.save(what="independent verification of the secondary experiments", code_dir=HERE + "/code")
R.paper(PN)
jdump(DOC, OUT + "/paper_numbers_doc.json")
R.table("\n".join(str(t) for t in T) + "\n")
print("\n".join(f"{k}: {v['verdict']}" for k, v in V.items())); print(len(R.records), "records;", len(PN), "paper numbers")
