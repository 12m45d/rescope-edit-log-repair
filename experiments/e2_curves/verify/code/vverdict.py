"""Verdicts of the independent verification of E2 a-c, computed from out/comparisons.json, out/cw_check.json, out/misc_check.json.
-> verify/verdicts.json and out/verdicts.txt (the compact table of the final message).  No number is typed by hand."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os

FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
VER = FIN + "/e2_curves/verify"
C = json.load(open(VER + "/out/comparisons.json"))
CW = json.load(open(VER + "/out/cw_check.json"))
MI = json.load(open(VER + "/out/misc_check.json"))
LOGS = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
V, rows = {}, []


def status(ok, partial=False):
    return "CONFIRMED" if ok else ("PARTLY CONFIRMED" if partial else "REFUTED")


def f(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


have = {n: bool(C["logs"].get(n, {}).get("cells")) for n in LOGS}
# 1. headline cells recomputed on the GPU
cell_ok, detail = {}, {}
for n, sh in LOGS.items():
    for k, c in C["logs"].get(n, {}).get("cells", {}).items():
        m, e, ag = c["mine"], c["theirs_reported"], c["agreement"]
        ok = all(e[q] is not None and abs(e[q]["value"] - m[q]["value"]) < 1e-9 for q in ("D1", "D1_strict", "rel", "tgen", "igen")) and ag["edit_exact_differ"] == 0 and ag["probe_exact_differ"] == 0
        cell_ok[f"{sh}/{k}"] = ok
        detail[f"{sh}/{k}"] = {"D1_exp": e["D1"]["value"] if e["D1"] else None, "D1_mine": m["D1"]["value"], "edits_exp": [e[q]["value"] if e[q] else None for q in ("rel", "tgen", "igen")],
                               "edits_mine": [m[q]["value"] for q in ("rel", "tgen", "igen")], "own_out_exp": e["own_out"]["value"] if e.get("own_out") else None, "own_out_mine": m["own_out"],
                               "flags_differ_edits": ag["edit_exact_differ"], "flags_differ_heldout": ag["probe_exact_differ"], "fired_flags_differ": ag["edit_fired_differ"] + ag["probe_fired_differ"],
                               "max_abs_threshold_diff": None if c["thresholds"] is None else c["thresholds"]["max_abs_diff"]}
        rows.append([f"{sh} {k}: D1 %; Rel/T-Gen/I-Gen", f"{f(detail[f'{sh}/{k}']['D1_exp'])}; " + "/".join(f(x, 0) for x in detail[f"{sh}/{k}"]["edits_exp"]),
                     f"{f(m['D1']['value'])}; " + "/".join(f(m[q]["value"], 0) for q in ("rel", "tgen", "igen")),
                     ("yes" if ok else "NO") + f" (flags differ {ag['edit_exact_differ']}/{ag['edit_n']}, {ag['probe_exact_differ']}/{ag['probe_n']})"])
n_cells = len(cell_ok)
V["headline_cells_recomputed_with_the_plain_evaluator"] = {
    "verdict": status(all(cell_ok.values()) and n_cells == 16, partial=any(cell_ok.values())), "n_cells": n_cells, "n_agree": sum(cell_ok.values()), "cells": detail,
    "what": "deployed log; ReScope-W least repair from the deployed scoping at n = 10 and 100 (seed 0); ReScope-W least repair from the log-only thresholds at n = 10 (seed 0, the H1 cell); "
            "COMP-B rebuilt on all complaints; ReScope label-free at n = 0, 10, 200 (draw 0); both development logs; thresholds, draws and complaint sets rebuilt by the verifier"}
# 2. linear cells
imp = {sh: C["logs"][n]["linear_import"] for n, sh in LOGS.items() if "linear_import" in C["logs"].get(n, {})}
V["imported_linear_cells_equal_the_p12_records"] = {"verdict": status(len(imp) == 2 and all(i["n_mismatch"] == 0 for i in imp.values())),
                                                    "per_log": {sh: {k: i[k] for k in ("n_cells", "n_numbers", "n_mismatch")} for sh, i in imp.items()},
                                                    "summary_rows": {sh: C["logs"][n]["linear_summary"] for n, sh in LOGS.items() if "linear_summary" in C["logs"].get(n, {})}}
for n, sh in LOGS.items():
    for k, s in C["logs"].get(n, {}).get("linear_summary", {}).items():
        rows.append([f"{sh} linear {k} ({s['n_seeds']} seeds): D1 %; Rel", f"{f(s['d1_exp'])}; {f(s['rel_exp'], 1)}", f"{f(s['d1_mine'])}; {f(s['rel_mine'], 1)}",
                     "yes" if abs(s["d1_exp"] - s["d1_mine"]) < 1e-9 and abs(s["rel_exp"] - s["rel_mine"]) < 1e-9 else "NO"])
# 3. part d
cb = {sh: C["logs"][n].get("comp_b_all_vs_p12_record") for n, sh in LOGS.items() if have[n]}
pd = {sh: MI[n]["part_d_flags_vs_record"] for n, sh in LOGS.items()}
ok_d = all([v["heldout_differ"], v["edit_differ"]] == v["experiment_states"] for sh in pd for k, v in pd[sh].items()) \
    and all(v["heldout_differ"] == 0 and v["edit_differ"] == 0 for sh in pd for k, v in pd[sh].items() if not k.startswith("REEDIT_B__n10"))
ok_cb = len(cb) == 2 and all(c and c["n_heldout_differ"] == 0 and c["n_edit_differ"] == 0 for c in cb.values())
V["rebuilt_linear_rows_reproduce_the_p12_flags"] = {"verdict": status(ok_d and ok_cb, partial=ok_d or ok_cb), "my_rebuilt_comp_b_all_vs_record": cb,
                                                    "experiment_rows_vs_record": {sh: {k: {"recounted_by_the_verifier": [v["heldout_differ"], v["edit_differ"]], "experiment_states": v["experiment_states"]} for k, v in pd[sh].items()} for sh in pd},
                                                    "note": "COMP-B on all complaints rebuilt and evaluated by the verifier; rollbacks and REEDIT rows checked from the experiment's per-row files against the records"}
for sh, c in cb.items():
    if c:
        es = pd[sh]["COMP_B__nALL__s0"]["experiment_states"]
        rows.append([f"{sh} COMP-B ALL rebuilt: flags differing from the P12 record (held-out; edits)", f"{es[0]}/{pd[sh]['COMP_B__nALL__s0']['n_heldout']}; {es[1]}/300", f"{c['n_heldout_differ']}/{c['n_heldout']}; {c['n_edit_differ']}/{c['n_edit']}",
                     "yes" if c["n_heldout_differ"] == 0 and c["n_edit_differ"] == 0 else "NO"])
# 4. part b bookkeeping
pb = {sh: CW[n]["part_b"] for n, sh in LOGS.items()}
fl = {sh: MI[n]["floor"] for n, sh in LOGS.items()}
ok_b = all(p["n_cells"] == 36 and p["n_refused_total"] == 0 and p["n_still_firing_total"] == 0 and p["n_cells_20_of_20_orders"] == 36 and p["hook_vs_vectors_differ"] == 0 for p in pb.values()) \
    and all(MI[n]["part_b"]["n_own_rel_prompts_out_of_scope_over_all_cells"] == 0 for n in LOGS) and all(float(fl[sh][k]) == float(fl[sh]["experiment_states"][k]) for sh in fl for k in ("edits", "heldout_disjoint", "hard_negatives"))
mine_b = {f"{sh}/{k}": C["logs"][n]["cells"][k]["thresholds"]["mine"] for n, sh in LOGS.items() if have[n] for k in ("wl_dep_n10", "wl_dep_n100", "wl_log_n10")}
ok_b = ok_b and all(m.get("n_refused") == 0 and m.get("still_firing") == 0 for m in mine_b.values())
V["part_b_no_refusal_own_prompts_in_scope_order_independent_floor"] = {"verdict": status(ok_b), "cells": pb, "round_off_floor": fl, "own_rel_prompts_out_of_scope": 0 if ok_b else None,
                                                                    "my_least_repairs_seed0": mine_b, "jobs_clean": {sh: CW[n]["jobs"] for n, sh in LOGS.items()}}
for sh in fl:
    rows.append([f"{sh} round-off floor: flags differing from the deployed weight (edits; held-out; hard negatives), from the rows", "; ".join(f(float(fl[sh]["experiment_states"][k]), 0) for k in ("edits", "heldout_disjoint", "hard_negatives")),
                 "; ".join(str(fl[sh][k]) for k in ("edits", "heldout_disjoint", "hard_negatives")), "yes" if ok_b else "see part b"])
# 5. H1
h = {sh: C["logs"][n]["h1"] for n, sh in LOGS.items() if "h1" in C["logs"].get(n, {})}
ok_h = len(h) == 2 and all(abs(x["d1_where_exp"] - x["d1_where_mine"]) < 1e-9 and abs(x["rel_where_exp"] - x["rel_where_mine"]) < 1e-9 and abs(x["rel_deployed_exp"] - x["rel_deployed_mine"]) < 1e-9
                           and x["literal_rule_helps_on_this_log_exp"] == x["literal_rule_helps_on_this_log_mine"] and x["one_sided_met_exp"] == x["one_sided_met_mine"]
                           and all(c["linear_exp"] == c["linear_mine"] and abs(c["diff_exp"]["value"] - c["diff_mine"]["value"]) < 1e-9 and abs(c["diff_exp"]["ci_lo"] - c["diff_mine"]["ci_lo"]) < 1.0
                                   and abs(c["diff_exp"]["ci_hi"] - c["diff_mine"]["ci_hi"]) < 1.0 for c in x["comparisons"].values())
                           and len({x["best_linear"][k] for k in x["best_linear"]}) == 1 for x in h.values())
V["h1_json"] = {"verdict": status(ok_h), "per_log": {sh: {k: x[k] for k in x if k != "candidates"} for sh, x in h.items()},
                "literal_rule": {sh: x["literal_rule_helps_on_this_log_mine"] for sh, x in h.items()}, "helps_on_both_development_logs_literal": all(x["literal_rule_helps_on_this_log_mine"] for x in h.values()) and len(h) == 2,
                "one_sided_reading_met_on_both": all(x["one_sided_met_mine"] for x in h.values()) and len(h) == 2,
                "note": "numbers and the literal verdict of h1.json are reproduced from the per-row files; the best linear repair is the same for the experiment's six methods, the four of PROTOCOL "
                        "section 2 and every P12 method with three seeds; which reading of 'within 2 points' R6 means is not a verification question; the prefix part of the rule is experiment e2_prefix"}
for sh, x in h.items():
    c = x["comparisons"]["best_lowest_d1"]
    rows.append([f"{sh} H1: D1 % ReScope-W; Rel; Rel deployed", f"{f(x['d1_where_exp'])}; {f(x['rel_where_exp'], 1)}; {f(x['rel_deployed_exp'], 1)}", f"{f(x['d1_where_mine'])}; {f(x['rel_where_mine'], 1)}; {f(x['rel_deployed_mine'], 1)}", "yes" if ok_h else "see h1"])
    rows.append([f"{sh} H1: difference to {c['linear_mine']} [95 %]", f"{f(c['diff_exp']['value'])} [{f(c['diff_exp']['ci_lo'])}, {f(c['diff_exp']['ci_hi'])}]",
                 f"{f(c['diff_mine']['value'])} [{f(c['diff_mine']['ci_lo'])}, {f(c['diff_mine']['ci_hi'])}]", "yes"])
    rows.append([f"{sh} H1: literal rule helps / one-sided reading met", f"{x['literal_rule_helps_on_this_log_exp']} / {x['one_sided_met_exp']}", f"{x['literal_rule_helps_on_this_log_mine']} / {x['one_sided_met_mine']}", "yes"])
# 6. c' gate level
ok_cw = all(max(CW[n]["max_abs_diff"][k] for k in ("fired_as_run", "fired_literal", "n_capped")) < 1e-9 and CW[n]["max_abs_diff"]["fresh_repair_literal"] < 0.1 for n in LOGS)
V["c_prime_gate_level"] = {"verdict": status(ok_cw), "mean_over_20_draws": {sh: CW[n]["mean"] for n, sh in LOGS.items()}, "bound_percent": CW["TB-Q100-ae-l2_3"]["bound_percent"],
                           "note": "recomputed by the verifier (own draws, own two-block rule, own guard caps) from the experiment's captured all-position vectors; the behavioural cells of c' "
                                   "(D1, edits) were NOT recomputed on the GPU: they are consistent with their per-row files (re-rendering) only; the fresh-REPAIR read-out differs by at most "
                                   "0.03 points per draw because the verifier identifies queries by the library's query id"}
for n, sh in LOGS.items():
    m = CW[n]["mean"]
    rows.append([f"{sh} c' false firing % as run; literal; capped edits (20 draws)", f"{f(m['fired_as_run_exp'])}; {f(m['fired_literal_exp'])}; {f(m['n_capped_exp'])}",
                 f"{f(m['fired_as_run_mine'])}; {f(m['fired_literal_mine'])}; {f(m['n_capped_mine'])}", "yes" if ok_cw else "NO"])
# 7. sanity
sa = {sh: C["logs"][n]["sanity"] for n, sh in LOGS.items() if "sanity" in C["logs"].get(n, {})}
ok_s = len(sa) == 2 and all(s["n"] == 11 and s["max_abs_diff"] < 1e-9 for s in sa.values())
V["agreement_with_the_sanity_run_e0"] = {"verdict": status(ok_s, partial=bool(sa)), "max_abs_diff_points_of_my_numbers": {sh: s["max_abs_diff"] for sh, s in sa.items()}, "n_numbers": {sh: s["n"] for sh, s in sa.items()},
                                        "experiments_cells_against_e0_recounted_from_its_rows": {sh: MI[n]["sanity_claim_from_the_experiments_rows"] for n, sh in LOGS.items()},
                                        "note": "my own deployed and default-method (draw 0) numbers against e0, and the experiment's seven shared cells (draws 0 - 2) recounted from its rows; the "
                                                "summary's '44 cells per log' are 44 numbers of 7 cells"}
ok_s = ok_s and all(MI[n]["sanity_claim_from_the_experiments_rows"]["max_abs_diff_points"] < 1e-9 for n in LOGS)
V["agreement_with_the_sanity_run_e0"]["verdict"] = status(ok_s, partial=bool(sa))
# 8. running example
ex = C.get("example")
if ex:
    V["running_example"] = {"verdict": status(ex["n_agree"] == ex["n"], partial=ex["n_agree"] >= ex["n"] - 3), "n_values": ex["n"], "n_agree": ex["n_agree"],
                            "disagreeing": [r for r in ex["rows"] if not r["agree"]], "mine_details": ex["mine_details"]}
    for r in ex["rows"]:
        if r["key"] in ("complaint.max_coordinate", "rollback.n_deleted", "rollback.edits_broken", "compensation.heldout_changed", "rescope.own_requests_firing", "whynot.coordinate", "whynot.threshold",
                        "complaint.deployed_answer", "whynot.answer") or not r["agree"]:
            rows.append([f"example {r['key']}", str(r["experiment"])[:44], str(r["mine"])[:44], "yes" if r["agree"] else "NO"])
else:
    V["running_example"] = {"verdict": "NOT FINISHED", "note": "the Qwen GPU job of the verifier has not written the example"}
# 9. internal consistency
ren = C["rerender"]
ok_r = ren.get("ran") and ren["results_json_equal"] and ren["paper_numbers_equal"] and ren["table_lines_differing_after_header"] == 0 and ren["h1_json_equal"] and ren["paper_numbers_all_documented"] \
    and not ren["paper_numbers_not_found_in_results_or_h1"]
V["internal_consistency"] = {"verdict": status(bool(ok_r), partial=bool(ren.get("ran"))), "rerender": ren, "counts": {sh: C["logs"][n].get("counts") for n, sh in LOGS.items()}}
rows.append(["re-rendering: results.json / paper_numbers.json / table.txt / h1.json identical", "-", f"{ren.get('results_json_equal')} / {ren.get('paper_numbers_equal')} / {ren.get('table_lines_differing_after_header') == 0} / {ren.get('h1_json_equal')}", "yes" if ok_r else "NO"])
# 10. what the files show and the summary does not say (numbers read from the experiment's files by code)
miss = {}
for n, sh in LOGS.items():
    hb, fh = MI[n]["part_b"]["h1_cells_logonly_n10"], MI[n]["files_hold"]
    miss[sh] = {"h1_row_fires_on_heldout_disjoint_percent_mean": fh["h1_row_fired_heldout_mean"], "h1_row_near_scope_firing_percent_gate_mean": fh["h1_row_hn_fired_gate_mean"],
                "h1_row_near_scope_D1_percent_seed0": hb["0"]["hn_D1"], "h1_row_rel_kept_gained_lost_seed0": [hb["0"]["rel_kept"], hb["0"]["rel_gained"], hb["0"]["rel_lost"]],
                "h1_row_edit_checks_lost_gained_of_300_seed0": [hb["0"]["checks_lost_rel_tgen_igen"], hb["0"]["checks_gained_rel_tgen_igen"]],
                "deployed_start_n10_fires_on_heldout_percent_mean": fh["deployed_start_n10_fired_heldout_mean"], "deployed_start_n10_near_scope_firing_percent": fh["deployed_start_n10_hn_fired_gate_mean"],
                "rescope_n200_near_scope_firing_percent_mean": fh["rescope_n200_hn_fired_mean"], "rescope_MME-T_false_firing_percent_mean_by_n": fh["rescope_MME-T_fired_mean"],
                "c_prime_fresh_repair_false_firing_percent": CW[n]["mean"]["fresh_repair_literal_exp"]}
V["unfavourable_items_in_the_files_missing_from_the_summary"] = {
    "verdict": "PARTLY CONFIRMED (the summary's list is correct but incomplete)", "numbers": miss,
    "items": ["the H1 row (ReScope-W, log-only start, n = 10) still releases an edit on a sizeable share of heldout_disjoint rows and of the near-scope negatives; the summary prints near-scope firing for c' only (R6: next to every far rate)",
              "ReScope-W does not keep the deployed edits as they are: edit checks that are exact under the deployed log are lost and others gained (Rel kept / gained / lost above); the summary gives net rates only",
              "false firing of the label-free ReScope on MME-T exceeds 5 percent on the Qwen log at n = 40, 50, 200 and ALL (shifted workload, prompt-only keys)",
              "c' on the LLaVA log exceeds the bound on the fresh REPAIR rows (the summary prints the number without saying so); cw_check.py is a post-hoc diagnostic written after the result",
              "least repair from the deployed start at n = 10 leaves an edit released on about half (Qwen) and almost all (LLaVA) held-out rows: low D1 there is not a certificate"]}
V["_protocol_and_code_notes"] = [
    {"file": "e2_curves/code/a_import.py", "line": "20-37 (rows_of), 56-76", "problem": "E2 (a) asks for the P12 repairs re-scored through the library's evaluation; the cells copy the P12 harness flags (disclosed in the summary)",
     "effect_on_numbers": "none found: 0 mismatches against the records; COMP-B ALL rebuilt by the verifier and the rebuilt rollbacks reproduce every flag; REEDIT is not reproducible by construction"},
    {"file": "e2_curves/code/tables.py", "line": "381-384, 394", "problem": "the 'best linear repair' of H1 is chosen on the reported heldout_disjoint D1 over six P12 methods (in favour of the baseline); line 394 fixes the two-sided reading of 'within 2 points'",
     "effect_on_numbers": "none: the same method (RB_UNION) is best among the four methods of PROTOCOL section 2 and among every P12 method with three seeds; on the LLaVA log the literal verdict is 'does not help' only because Rel is above the deployed log"},
    {"file": "e2_curves/code/cw_check.py", "line": "1-4, 31-35", "problem": "post-hoc read-out (false firing of c' on REPAIR rows outside the draw) added after the heldout_disjoint result was seen; REPAIR rows are not a reporting pool",
     "effect_on_numbers": "none on registered numbers; must be labelled post hoc if quoted"},
    {"file": "e2_curves/code/example.py", "line": "56, 132-139", "problem": "the default method releases no edit on the example's complaint, so 'the edit' is the one with the largest coordinate, and the why-not rephrase belongs to another edit (disclosed)",
     "effect_on_numbers": "none; the example's edit and why-not pair must not be presented as one edit"},
    {"file": "e2_curves/code/tables.py", "line": "224", "problem": "capped edits of c' on the LLaVA log printed with one decimal (0.1 for a mean of 0.05)", "effect_on_numbers": "cosmetic; paper_numbers.json holds 0.05"}]
tl = MI["_timeline"]
V["_protocol_and_code_notes"].append(
    {"file": "e2_curves/code/tables.py", "line": "394-400", "problem": "the summary says the two-sided reading of H1 was fixed in code before any ReScope-W cell existed; the file that holds the rule was last modified "
     + tl["code_last_modified"]["tables.py"] + ", the H1 cells were written " + json.dumps(tl["h1_cells_written"]) + ": the statement cannot be confirmed from the files (modified after the last cell: "
     + str(tl["tables_py_modified_after_the_last_h1_cell"]) + ")", "effect_on_numbers": "none: both readings are in h1.json and the decision is left to the coordinator"})
V["_protocol_and_code_notes"].append(
    {"file": "e2_curves/code/job.py", "line": "whole file", "problem": "job.py was last modified " + tl["code_last_modified"]["job.py"] + "; the job processes started about " + json.dumps(tl["job_process_started"])
     + " (modified after the start: " + json.dumps(tl["job_py_modified_after_the_job_started"]) + "): a job that started before the edit ran a version of the script that is not on disk",
     "effect_on_numbers": "none found: the cells of the two logs have the same tags and fields (" + str(tl["cells_of_the_two_logs_have_the_same_tags_and_fields"]) + ") and every recomputed cell of both logs agrees"})
V["_timeline"] = tl
V["_not_done_by_the_verifier"] = ["seeds 1 and 2 of the ReScope-W cells and draws 1 and 2 of the label-free curve were not recomputed on the GPU (H1 uses them from the per-row files)",
                                  "the behavioural cells of c', D2, hard-negative behaviour and free generation were not recomputed",
                                  "the LLaVA compensation of the verifier was solved on the GPU (P12: CPU); its flags equal the record",
                                  "the first Qwen attempt of the verifier ran out of GPU memory in the harness COMP solve on the GPU (logs/vjob_Q_try1_oom.out); the job was started again with "
                                  "the solve on the CPU, as P12, at the queue priority the experiment itself used"]
for nme in LOGS:
    assert nme in MI
V["_missing"] = C["missing"]
json.dump(V, open(VER + "/verdicts.json", "w"), indent=1)
w = [max(len(str(r[i])) for r in rows + [["quantity", "experiment", "mine", "agree"]]) for i in range(4)]
txt = ["VERDICTS (verify/code/vverdict.py)"] + [f"  {k}: {v['verdict']}" for k, v in V.items() if isinstance(v, dict) and "verdict" in v] + [""]
txt += ["  ".join(str(c).ljust(w[i]) for i, c in enumerate(["quantity", "experiment", "mine", "agree"]))] + ["  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) for r in rows]
open(VER + "/out/verdicts.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
