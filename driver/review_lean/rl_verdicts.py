"""Lean review: verdicts.json from the review's own output files (no number typed by hand), patches.txt, and the review fields of READY.json.

    python rl_verdicts.py            (after every check has run; READY.json: fields "reviewed", "review", fingerprint re-computed)
"""
import glob
import hashlib
import json
import os
import time

import rl_common as C
import fx

Q, L = C.DEV
J = lambda n: json.load(open(os.path.join(C.OUT, n)))            # noqa: E731
ex = lambda n: os.path.exists(os.path.join(C.OUT, n))            # noqa: E731
fp = fx.code_fingerprint()
V = {}


def both(fn):
    return {"Q": fn(Q), "L": fn(L)}


def verdict(ok, fixed=False):
    return "fixed" if fixed else ("pass" if ok else "fail")


# ---------------------------------------------------------------- 1
c1 = both(lambda n: J(f"cpu1_{n}.json"))
alt = {a: both(lambda n: J(f"cpu1_{n}_alpha{a}.json")) for a in ("0.01", "0.2")}
g = both(lambda n: J(f"gpu_{n}.json"))
ok1 = all(c1[k]["items"]["1_gates"]["n_differ"] == 0 and c1[k]["items"]["1_gates"]["all_capped_sets_equal"] and c1[k]["items"]["1_gates"]["all_ranks_equal"]
          and not c1[k]["items"]["1_fairness"]["violations"] and g[k]["blocks"]["hook"]["ok"] for k in c1) \
    and all(alt[a][k]["items"]["1_gates"]["n_differ"] == 0 for a in alt for k in ("Q", "L")) and all(all(x[d]["draw_equals_library_sample"] and x[d]["null_queries_shared_with_calibration"] == 0 and x[d]["null_distinct_queries"] == 100 for d in x) for x in (c1[k]["items"]["draws"] for k in c1))
V["1_new_gates"] = {"verdict": verdict(ok1), "evidence": {
    "own_reimplementation": "numpy fp64 from the raw files (factors, key cache, BEN-W keys by the parts file, direct d x d Cholesky solve of C_W + rho I, own null repair, own two-block / shared rule, own guard cap, own top-1)",
    "alpha_0.05": {k: {"gates_x_draws": c1[k]["items"]["1_gates"]["n_gates"] * 2, "decisions": c1[k]["items"]["1_gates"]["n_decisions"], "differ": c1[k]["items"]["1_gates"]["n_differ"],
                       "max_eta_abs_diff": c1[k]["items"]["1_gates"]["max_eta_abs_diff"], "capped_sets_equal": c1[k]["items"]["1_gates"]["all_capped_sets_equal"]} for k in c1},
    "other_levels": {a: {k: [alt[a][k]["items"]["1_gates"]["n_decisions"], alt[a][k]["items"]["1_gates"]["n_differ"]] for k in ("Q", "L")} for a in alt},
    "fairness_vs_registered_cosine_gate": {k: {"checked": c1[k]["items"]["1_fairness"]["n_checked"], "violations": c1[k]["items"]["1_fairness"]["violations"],
                                               "what": "released values bitwise, role last, top-1, REPL, guard keys bitwise, same draw rows, same blocks"} for k in c1},
    "calibration_draw": {k: c1[k]["items"]["draws"] for k in c1},
    "hook_on_gpu_equals_gate_level": {k: {n: [v["n"], v["hook_release_differs_from_gate_level"], v["silent_not_bitwise_preedit"], v["memo_path_flags_differ_from_forward"]] for n, v in g[k]["blocks"]["hook"]["gates"].items()} for k in g}}}
# ---------------------------------------------------------------- 2
r2 = both(lambda n: J(f"cpu2_retr_{n}.json"))
st = J("cpu2_stats.json")
h3r = both(lambda n: J(f"cpu2_h3retr_{n}.json"))
ok2 = all(r2[k]["ok"] for k in r2) and all(v["valid"] for v in st["retrieval_validity"].values()) and all(g[k]["blocks"]["retr"]["ok"] for k in g) and all(h3r[k]["ok"] for k in h3r)
V["2_retrieval_family"] = {"verdict": verdict(ok2), "evidence": {
    "three_fusion_rules_own_implementation": {k: {n: {d: {"decisions": x["n"], "differ": x["n_decisions_differ"], "capped_mine_driver": [x["n_capped_mine"], x["n_capped_driver"]],
                                                          "own_request_candidate": x["own_request_candidate_driver"], "text_rows_firing": x["text_rows_firing_driver"], "eta_abs_diff": x["eta_abs_diff"],
                                                          "top1_by_raw_fused_score_would_differ_on": x.get("top1_by_raw_score_would_differ_on")} for d, x in v.items()} for n, v in r2[k]["rules"].items()} for k in r2},
    "lookup": {k: r2[k]["lookup"] for k in r2}, "n_text_rows_checked": {k: r2[k]["n_text_rows"] for k in r2},
    "validity_on_exchangeable_synthetic_scores": st["retrieval_validity"],
    "behaviour_gpu": {k: g[k]["blocks"]["retr"]["rules"] for k in g}, "h3_own_computation": {k: {n: {d: x["agree"] for d, x in v.items()} for n, v in h3r[k]["rules"].items()} for k in h3r}},
    "note": "retr_sum / retr_min release the candidate with the largest MARGIN (score - baseline); 'R4 extended' says 'top-1 by the fused score'. The two readings release another edit on the "
            "sequences counted above (none changes whether anything is released). To be confirmed by the coordinator; the driver's choice is stated in USAGE.txt."}
# ---------------------------------------------------------------- 3
ft = J("cpu2_feat.json")
ok3 = all(not c1[k]["items"]["3_x"]["violations"] for k in c1) and all(v["violations"]["equals_cache_query"] == 0 and v["violations"]["q_equals_own_question_or_rephrase"] == 0 for v in ft["recomb_rules"].values())
V["3_x_construction"] = {"verdict": verdict(ok3), "evidence": {
    k: {"gates_x_draws": c1[k]["items"]["3_x"]["n_checked"], "violations": c1[k]["items"]["3_x"]["violations"],
        "checked": "x = own maximum over the edit's OWN rows (owner by image for X-I / X-X, by question for X-Q) equals the driver's to 1e-9; baseline = max(block 1, x); block 2 and rank of the base gate; "
                   "guards of the base gate; thresholds and ranking baselines bit-identical to the base gate with x = -inf; decisions equal on 300 sequences x 2 draws",
        "example": c1[k]["items"]["3_x"]["per_gate"]["rescope_x"]["0"]} for k in c1},
    "no_recombination_is_a_test_query": {bb: {"rows": v["n_rows"], "equal_to_a_query_of_any_cache": v["violations"]["equals_cache_query"], "caches": v["caches_compared"], "cache_queries": v["n_cache_queries"],
                                              "equal_to_an_MME_query": v["violations"]["equals_mme_query"]} for bb, v in ft["recomb_rules"].items()}}
# ---------------------------------------------------------------- 4
ok4 = all(c1[k]["items"]["4_h3"]["all_agree"] for k in c1) and all(h3r[k]["ok"] for k in h3r)
V["4_h3"] = {"verdict": verdict(ok4), "evidence": {k: {"split": {"n_fit": c1[k]["items"]["4_h3"]["n_fit"], "n_test": c1[k]["items"]["4_h3"]["n_test"], "own_edit_matches_raw_index": c1[k]["items"]["4_h3"]["own_edit_matches_raw_index"]},
                                                         "gates": {n: {d: {"agree": x["agree"], "raise_driver": x.get("delta_driver"), "raise_abs_diff": x.get("delta_abs_diff"), "coverage_odd_edits": x.get("coverage_driver"),
                                                                           "test_firing": x.get("test_fired_driver"), "heldout_disjoint": x.get("heldout_disjoint_fired_driver"),
                                                                           "fit_mean_below_at_above": [x.get("fit_mean_just_below_mine"), x.get("fit_mean_at_driver_delta_driver"), x.get("fit_mean_just_above_mine")]}
                                                                       for d, x in v.items()} for n, v in c1[k]["items"]["4_h3"]["gates"].items()},
                                                         "capped_edits_of_the_checked_gates": c1[k]["items"]["4_h3"].get("n_capped")} for k in c1},
             "note": "HN-Q rows show a foreign image; the driver assigns them to the edit whose question they ask (USAGE.txt). The raise keeps the guards (thresholds = min(literal + raise, cap))."}
# ---------------------------------------------------------------- 5
i5 = {k: c1[k]["items"]["5_null_pac"] for k in c1}
cb = st["crossed_bootstrap"]
ok5 = all(i5[k]["rescope_null_nA0"]["tensors_bitwise_equal"] and i5[k]["rescope_null_nA0"]["digest_equal"] and i5[k]["rescope_pac"]["all_equal"] and i5[k]["rescope_pac"]["rank_in_record"] == i5[k]["rescope_pac"]["rank_n100"]
          and abs(i5[k]["rescope_pac"]["eta_mine"] - i5[k]["rescope_pac"]["eta_driver"]) < 1e-12 for k in i5) and st["ok"]
V["5_null_pac_bootstrap_holm"] = {"verdict": verdict(ok5), "evidence": {
    "rescope_null_n_A_0": {k: i5[k]["rescope_null_nA0"] for k in i5}, "rescope_null_detectors": {k: i5[k]["rescope_null"] for k in i5},
    "rescope_pac": {k: {x: i5[k]["rescope_pac"][x] for x in ("all_equal", "rank_n100", "rank_in_record", "eta_mine", "eta_driver", "eta_default")} for k in i5},
    "rescope_pac_rank_grid": i5["Q"]["rescope_pac"]["grid"], "crossed_bootstrap": cb, "holm": st["holm"]},
    "note": "Coverage of the crossed bootstrap is nominal where units dominate, too wide for pure cell noise and too narrow (see 'draw effect 0.3, 3 draws') when the draw component matters and only 3 draws exist."}
# ---------------------------------------------------------------- 6
p = both(lambda n: J(f"cpu2_pools_{n}.json"))
c3 = both(lambda n: J(f"cpu3_{n}.json"))
ok6 = all(p[k]["ok"] for k in p)
V["6_pools_masks"] = {"verdict": verdict(ok6), "evidence": {k: {"attempts_refused": len(p[k]["refused"]), "not_refused": [a for a, v in p[k]["refused"].items() if v == "NOT REFUSED"],
                                                                 "legitimate_calls_ok": sum(v is True for v in p[k]["allowed"].values()), "legitimate_calls_failed": [a for a, v in p[k]["allowed"].items() if v is not True],
                                                                 "statistics_rows_are_BEN-W": p[k]["statistics_rows_are_BEN-W"], "statistics_keys_are_BEN-W_decode_keys": p[k]["statistics_keys_are_BEN-W_decode_keys"],
                                                                 "pool_sizes": p[k]["pool_sizes"], "masks": p[k]["masks"], "BEN-W_overlap_with_other_pools": c3[k]["b_benw_overlap"], "MME_split": c3[k]["c_mme_split"]} for k in p},
                      "note": "The guard sees sequences, not tensors: null_keys= and library rules applied to fx.gate.features matrices are outside it (USAGE.txt, known limitations)."}
# ---------------------------------------------------------------- 7
ok7 = all(g[k]["blocks"][b]["ok"] for k in g for b in ("mask", "rollback", "prefix"))
V["7_prefix_maintenance"] = {"verdict": verdict(ok7), "evidence": {k: {"prefix": g[k]["blocks"]["prefix"]["prefixes"], "mask": {x: v for x, v in g[k]["blocks"]["mask"].items() if x != "seconds"},
                                                                         "rollback": g[k]["blocks"]["rollback"]} for k in g},
                             "note": "Load a prefix log in a fresh process: with reuse= after a long session of the same process the LLaVA load ran out of the 17000 MiB cap (review_lean/logs/gpu_L.log)."}
# ---------------------------------------------------------------- 8
api = both(lambda n: J(f"api_cpu_{n}.json"))
dg = J("digest_compare.json") if ex("digest_compare.json") else None
tdig = json.load(open(os.path.join(C.COMMON, "tests", "out", "digest_compare.json")))
tests = {}
for f in sorted(glob.glob(os.path.join(C.COMMON, "tests", "out", "*.json"))):
    try:
        r = json.load(open(f))
    except Exception:  # noqa: BLE001
        continue
    if not isinstance(r, dict) or "ok" not in r or "code" not in r:
        continue
    code = r.get("code") or {}
    b = os.path.basename(f)[:-5]
    tests[b] = {"pass": r["ok"] is True, "present_code": (code.get("fx") if isinstance(code, dict) else code) == fp["fx"]}
old = sorted(k for k, v in tests.items() if not v["present_code"])
san = {k: api[k]["calls"]["item 8: sanity numbers (M1 - M4, gate level, 20 draws) against e0_sanity/out/table.txt"] for k in api}
sr1 = json.load(open(os.path.join(C.COMMON, "tests", "out", "summary_r1.json")))
ok8 = bool(dg and not dg["differ"]) and len(tests) >= 60 and all(v["pass"] for v in tests.values()) and not sr1["not_passing"] and sr1["digest_unchanged"] is True and tdig.get("ok") is True and all(san[k]["ok"] and san[k]["info"]["all_within_rounding"] for k in san) and all(k in ("gpu_TB-B500-ae_e_mechanics", "gpu_TB-Q2000-ae-l2_3_e_mechanics", "gpu_TB-Q500-onehop-l2_3_e_mechanics") for k in old)
V["8_unchanged_behaviour"] = {"verdict": verdict(ok8), "evidence": {
    "review_digest": dg, "driver_digest_tests/digest.py": {k: tdig.get(k) for k in tdig if k in ("unchanged", "n_rows", "n_decisions", "same", "ok", "differ")} or tdig,
    "tests": {"n": len(tests), "passing": sum(v["pass"] for v in tests.values()), "on_present_code": sum(v["present_code"] for v in tests.values()), "passed_on_older_code_only": old, "driver_summary": {k: sr1[k] for k in ("n_tests", "n_pass_present_code", "pass_on_older_code_not_rerun", "not_passing", "digest_unchanged")},
              "re_run_by_the_review": "tests/t_cpu.py, t_ext_cpu.py, t_r1_cpu.py, digest.py and the FULL test d (E1 row set end to end, cold memo) on both development logs; the other GPU tests carry the present fingerprint (run 17:45 - 18:34 UTC) and were not repeated: no line of fx/*.py was changed by the review"},
    "sanity_numbers": {k: {"largest_abs_difference": san[k]["info"]["largest_abs_difference"], "all_within_rounding_of_the_table": san[k]["info"]["all_within_rounding"],
                           "driver_mean_over_20_draws": san[k]["info"]["driver_mean_over_20_draws"]} for k in san}}}
# ---------------------------------------------------------------- 9
apg = both(lambda n: J(f"api_gpu_{n}.json"))
fails = {k: [c for c, v in api[k]["calls"].items() if not v["ok"]] + [c for c, v in apg[k]["calls"].items() if not v["ok"]] for k in api}
ok9 = all(not v for v in fails.values()) and all(isinstance(apg[k]["assert_clean"], dict) and apg[k]["assert_clean"]["hook_mismatch"] == 0 for k in apg)
V["9_names_of_the_task_texts"] = {"verdict": verdict(ok9), "evidence": {k: {"cpu_calls": len(api[k]["calls"]), "gpu_calls": len(apg[k]["calls"]), "failed": fails[k], "assert_clean": apg[k]["assert_clean"],
                                                                              "rows_without_model": api[k]["e1_rowset_without_model"], "calls": sorted(api[k]["calls"]) + sorted(apg[k]["calls"])} for k in api},
                                  "not_obtainable_by_design": ["retrieval family and lookup on MME-T / MME-C (no attributes for MME rows)", "an _x form of a retrieval rule or of lookup",
                                                               "D1 / behaviour on benign-pool and MME rows (cut, R19 e)", "ReScope-W with two-block thresholds (R24): the task builds it with the library on the driver's draws",
                                                               "COMP-B+ and rebuilt harness corrections: fx.methods.wrap of the harness object"]}
# ---------------------------------------------------------------- 10
em = J("embed.json")
att = {k: apg[k]["calls"]["item 10: attributes of 100 sequences against the rendered prompts"] for k in apg}
rcv = ft["recomb_rules"]
ok10 = (ft["benign_split"]["same_items_and_parts_both_backbones"] and all(ft["benign_split"][bb]["items_in_two_parts"] == 0 and ft["benign_split"][bb]["rows_differing_from_own_split"] == 0 for bb in ("qwen25vl", "llava"))
        and all(ft["ben_t_use"][bb]["whiten_all_BEN-W"] for bb in ("qwen25vl", "llava")) and em["ok"] and all(att[k]["ok"] and att[k]["info"]["all_ok"] for k in att) and ft["stop_list"]["is_a_literal_in_stoplist_py"]
        and all(g[k]["blocks"]["recapture"]["ok"] for k in g) and all(sum(v for kk, v in rcv[bb]["violations"].items() if kk != "duplicate_rows") == 0 and rcv[bb]["every_edit_has_4_4_4"] and rcv[bb]["xx_scope_violations"] == 0 for bb in rcv)
        and all(c1[k]["items"]["1_gates"]["n_differ"] == 0 for k in c1))
V["10_feature_layer"] = {"verdict": verdict(ok10), "evidence": {
    "benign_split": ft["benign_split"], "BEN-T_in_the_packages": {"statistics_rows": ft["ben_t_use"], "note": "BEN-T rows are read by the packages only as test queries a recombination must not equal (the discard rule of R5) and in the parts file itself"},
    "whitened_scores_direct_solve": {k: {n: {d: [x["n"], x["n_decisions_differ"], x["eta_abs_diff"]] for d, x in v.items()} for n, v in c1[k]["items"]["1_gates"]["per_gate"].items() if n.startswith("white")} for k in c1},
    "whitened_scores_note": "per log and draw: 200 calibration keys + 300 sequence keys + 200 guard keys scored with M from a direct d x d Cholesky solve of C_W + rho I (not the package's Woodbury form): thresholds (eta) and every decision equal",
    "embeddings_re_embedded": {x: em[x] for x in ("images", "questions")}, "attributes_vs_rendered_prompts": {k: att[k]["info"] for k in att},
    "stop_list": ft["stop_list"], "recombination_keys_recaptured": {k: {x: g[k]["blocks"]["recapture"][x] for x in ("n", "n_bitwise", "n_lp_equal", "by_kind", "n_first_100_edits")} for k in g},
    "recombination_discard_rules": rcv, "partner_items": {k: c3[k]["d_partner_items"] for k in c3},
    "decisions_on_other_logs": {"files_scanned": ft["scan_log_names"]["files"], "other_logs_named_in": {k: len(v) for k, v in ft["scan_other_logs"].items()},
                                "finding": "other logs occur as the master logs of the long streams (captures), in the lists of streams / logs a package serves or refuses (row counts, label-free) and in tests that check the refusal; "
                                           "the mechanics outputs (white/out, retr/out, recomb/out) hold the two development logs only; white.mechanics refuses any other log"}}}
# ---------------------------------------------------------------- residual risks
ff = {k: c3[k]["e_false_firing_20_draws"]["per_method"] for k in c3}
risks = [
    "DECISION FOR THE COORDINATOR (not changed by the review): retr_sum / retr_min release the candidate with the largest margin (score - baseline), 'R4 extended' reads 'top-1 by the fused score'; "
    "firing is unaffected, the released edit differs on a few sequences (item 2). A one-line confirmation in PROTOCOL.md before E1 selects 'retr' settles it.",
    "DECISION FOR THE COORDINATOR: the _x form of a shared-scale gate of the cosine family is the two-block _x form on the same detectors (R5 is defined inside the two-block rule).",
    "H3: HN-Q rows are split by the edit whose question they ask (their image is foreign); the raise keeps the guard caps; report n_edits_at_cap with every H3 number.",
    "The subset of test hard negatives with an unseen foreign part has 4 (Qwen) / 1 (LLaVA) HN-X rows: report HN-I and HN-Q there and give the HN-X count.",
    "Capped edits void the bound: retr_min caps about 43 - 45 of 100 edits, retr_conj about 58, the cosine gate about 37 on the LLaVA development log; report fx.methods.n_capped and the literal rule (allow_silence=True) next to every calibrated row.",
    "One calibration draw is noisy: on the LLaVA development log cos_null fires on 12.15 % of heldout_disjoint at draw 0 and on " + f"{ff['L']['cos_null']['literal_mean']} % on average over the 20 draws (bound 4.95 %); "
    f"retr_sum averages {ff['L']['retr_sum']['literal_mean']} % there (the test pool is one fixed sample of 469 rows: about 1 point of sampling error). Report all 20 draws; E5 (c) is the validity experiment.",
    "Crossed bootstrap: with 3 behavioural draws the interval is too narrow when the draw-to-draw component matters (coverage " + f"{cb['draw effect 0.3, 3 draws']['coverage']:.2f} in the review's simulation) and too wide for pure cell noise; state the number of draws next to every interval.",
    "The pool guard sees sequences, not tensors: with null_keys= or with library rules on fx.gate.features matrices (E5 c) the experiment itself must keep BEN-T and MME-T on the test side.",
    "source='BEN-S' at the teacher-forced key has only 282 (Qwen) / 289 (LLaVA) rows: draws of 200 overlap heavily; which='decode' gives all 1375 rows at the prompt-only key.",
    "Retrieval family and lookup: not defined on the MME pools, no _x form, no tf_hashed / mask_edits; write_selection needs both backbones in one call.",
    "Prefix logs: open each in a fresh process (one log per process); probe records of prefixes carry no D2 fields.",
    "BEN-T firing is read at the prompt-only key while gates are calibrated at teacher-forced keys; fx.gate.evaluate's 'benign' part is still the whole pool (use fx.pools.firing for BEN-T).",
    "feat2/white/out/mechanics.txt is stale for the rows rescope_null / cos_null (row rule of the null sample); FEAT2_PROBLEMS.txt. feat2/verify holds an unfinished earlier check and no verdicts: this file replaces it.",
    "The three mechanics tests on logs cut by R19 (TB-B500-ae, TB-Q2000-ae-l2_3, TB-Q500-onehop-l2_3) passed on older code only; these logs are not used.",
]
verd = {"review": "lean review of the two-stage extension of the driver fx (stage 1 + stage 2) and of the feature layer feat2 / cache2", "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "code_fingerprint": fp, "fingerprint_before_the_review": "6a71f0d2cd71", "fx_files_changed_by_the_review": fp["fx"] != "6a71f0d2cd71",
        "summary": {k: v["verdict"] for k, v in V.items()}, "usable_as_it_is": all(v["verdict"] in ("pass", "fixed") for v in V.values()),
        "items": V, "residual_risks": risks, "false_firing_heldout_disjoint_20_draws_percent": ff,
        "code_of_the_review": sorted(os.path.basename(f) for f in glob.glob(os.path.join(C.HERE, "rl_*.py"))), "outputs": "review_lean/out/*.json, review_lean/logs/"}
with open(os.path.join(C.HERE, "verdicts.json"), "w") as f:
    json.dump(verd, f, indent=1, default=str)
print(json.dumps(verd["summary"], indent=1), "usable:", verd["usable_as_it_is"])
# ---------------------------------------------------------------- patches.txt
with open(os.path.join(C.HERE, "patches.txt"), "w") as f:
    f.write("PATCHES OF THE LEAN REVIEW (review_lean; " + verd["date"] + ")\n"
            "No line of fx/*.py was changed: the review found no defect in the driver or in the feature layer that needed a patch.\n"
            f"Code fingerprint before and after: 6a71f0d2cd71 / {fp['fx']} (library {fp['rescope']}).\n"
            "Decision digest of the existing row set (both development logs, draw 0): review_lean/out/digest_before.json == digest_after.json "
            f"({dg['rows'] if dg else '?'} rows, {dg['decisions'] if dg else '?'} decisions, {len(dg['differ']) if dg else '?'} differ); tests/digest.py: unchanged.\n"
            "Files written by the review: review_lean/ (code rl_*.py, out/, logs/, verdicts.json, this file); tests/out/gpu_<development log>_d_e1.json and e1_demo_<log>/ (the full test d, re-run on the\n"
            "present code), tests/out/*.json of the CPU suites (re-run), tests/out/table*.txt and READY.json (re-rendered; fields 'reviewed', 'review' added); USAGE.txt (the header and one line of KNOWN LIMITATIONS OF STAGE 2 now name the review).\n"
            "Memo: entries written by the review are outcomes of forwards under gates of the development logs (same keys as the experiments use).\n")
# ---------------------------------------------------------------- READY.json
rp = os.path.join(C.COMMON, "READY.json")
ready = json.load(open(rp))
d = os.path.join(C.COMMON, "fx")
h = hashlib.sha1()
for fn in sorted(os.listdir(d)):
    if fn.endswith(".py"):
        h.update(fn.encode())
        h.update(open(os.path.join(d, fn), "rb").read())
assert h.hexdigest()[:12] == fp["fx"] == ready["code_fingerprint"], (h.hexdigest()[:12], fp, ready["code_fingerprint"])
ready["reviewed"] = verd["usable_as_it_is"]
ready["functions_that_return_methods"] = list(dict.fromkeys(ready["functions_that_return_methods"]))
ready["reviewed_base"] = "b92d0532498f (review/patches.txt); the stage-1 extension and stage 2 (fx.review1) were reviewed independently on " + verd["date"] + ": review_lean/verdicts.json"
ready["review"] = {"record": "review_lean/verdicts.json", "patches": "review_lean/patches.txt (none to fx/*.py)", "date": verd["date"], "code_fingerprint": fp["fx"], "library_fingerprint": fp["rescope"],
                   "verdicts": verd["summary"], "n_residual_risks": len(risks)}
with open(rp, "w") as f:
    json.dump(ready, f, indent=1)
print("READY.json: reviewed =", ready["reviewed"], "fingerprint", ready["code_fingerprint"])
