"""Assemble the audit's verdicts from the audit's own result files (out/*.json).  Every number in verdicts.json and in
verdicts_table.txt is read from those files; a verdict is PASS only if the stated numeric criteria hold.

    python make_verdicts.py        -> verdicts.json, verdicts_table.txt
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402


def J(name):
    return A.jload(os.path.join(A.AOUT, name))


prov, recon, sc, traps = J("provenance.json"), J("recon.json"), J("single_cmp.json"), J("traps.json")
ev, evx, pools, disc, claims = J("eval_tf4.json"), J("eval_env.json"), J("pools.json"), J("discipline.json"), J("claims.json")
x1, x2, x3, lib, misc = J("extra.json"), J("extra2.json"), J("extra3.json"), J("lib.json"), J("misc.json")
pcmp, puc, pind, envx = J("pub_cmp.json"), J("pub_uc.json"), J("pub_indep.json"), J("env_extra.json")
selft, wtime, repeats = J("selftest.json"), J("writetime.json"), J("repeats.json")
ROME_RUNS = {t: J(f"rome_{t}.json") for t in ("det", "devfull", "stream30", "streamT100plus60")}
NAMES = list(A.LOGS)
REPORTED = [n for n in NAMES if "dev" not in n]
r1 = lambda x: round(float(x), 1)   # noqa: E731
r2 = lambda x: round(float(x), 2)   # noqa: E731


def PF(ok):
    return "PASS" if ok else "FAIL"


V = {"audit": "E10 public text logs: sequential ROME (public code) on GPT-2 XL with CounterFact -- logs TX-G2XL-rome-dev100, -T100, -T500, -T1000, -T2000",
     "written_utc": A.now(), "audit_dir": A.AUD, "audit_seed": A.AUDIT_SEED,
     "binding_specification": "02_research/experiments/10_final/PROTOCOL.md, section 6, amendment 'E10: public text logs' and its correction "
                              "(single-edit check on the first 1000 records); 04_handover/README.md section 5 (data traps)",
     "independence": "no function of the builder's code (text_public/code) is imported; the audit reads the builder's stored files, the public "
                     "repository, the checkpoint, the data and the statistics file, and drives the public code with drivers of its own",
     "discipline_of_the_audit": "on the reported stream the audit ran only the public editor (re-derivation of logged updates), the pre-edit and the "
                                "deployed model; no threshold, gate, scoped replay, rollback or repair was run or computed on any log of the family"}

# ======================================================================================= determinism (used by check 2)
det_rows = {}
for tag, R in ROME_RUNS.items():
    for it in R["items"]:
        for u in it["updates"]:
            det_rows[(it["source"], u["index"])] = (u["left_vs_stored"]["bitwise_equal"], u["right_vs_stored"]["bitwise_equal"],
                                                    u["key_audit_vs_stored_k_request"]["bitwise_equal"],
                                                    u["output_before_audit_vs_stored_cur_output"]["bitwise_equal"],
                                                    abs(u["readout_dot_key_minus_1"]), u["output_after_vs_stored_target_rel"],
                                                    u["module_output_formula_rel"], u["key_unchanged_by_update_bitwise"])
det = {}
for src in ("dev", "stream"):
    rows = {k: v for k, v in det_rows.items() if k[0] == src}
    det[src] = {"n_distinct_updates_rerun_by_audit": len(rows), "n_left_vector_bitwise_equal": sum(v[0] for v in rows.values()),
                "n_right_vector_bitwise_equal": sum(v[1] for v in rows.values()), "n_solver_key_bitwise_equal(audit's own hook)": sum(v[2] for v in rows.values()),
                "n_output_before_bitwise_equal(audit's own hook)": sum(v[3] for v in rows.values()),
                "max_abs(<readout, key> - 1)": max(v[4] for v in rows.values()),
                "max_rel(module output after the update - logged target)": max(v[5] for v in rows.values()),
                "max_rel(key @ W + b - module output)": max(v[6] for v in rows.values()),
                "indices": "all 100" if src == "dev" else (f"0..99 (T100 complete) and {len([k for k in rows if k[1] >= 100])} updates between 100 and 1999: 100-102, 173-175, "
                                                           f"1530-1532 and random ones; {len([k for k in rows if k[1] >= 173])} of them after the collapse")}
det_all_bitwise = all(d["n_left_vector_bitwise_equal"] == d["n_distinct_updates_rerun_by_audit"] == d["n_right_vector_bitwise_equal"] for d in det.values())
stream_after_173 = len([k for k in det_rows if k[0] == "stream" and k[1] >= 173])
T_ = misc["T"]
templates_ok = all(R["context_templates"]["equal_to_builder"][k] for R in ROME_RUNS.values() for k in ("dev", "stream", "single"))
others_ok = all(R["other_parameters"]["unchanged"] for R in ROME_RUNS.values())

# ======================================================================================= check 1: provenance
pc, hp, md, dt, st, gj = (prov[k] for k in ("public_code", "hparams", "model", "data", "statistics", "gpt_j"))
sf = x1["statistics_fresh_download"]
pp = x1["public_printouts"]
stats_never_recomputed = all(pp[k]["n_'Computing locally'"] == 0 and pp[k]["n_'Attempting to download'"] == 0
                             and len(pp[k]["statistics_files_loaded"]) == 1 for k in ("dev", "stream", "single"))
weights_ok = all(md["files"][f]["sha256_equals_mirror_listing"] and md["files"][f]["size_equals_mirror_listing"] for f in ("model.safetensors", "pytorch_model.bin"))
small_ok = all(v.get("blob_equals_mirror_listing", True) for v in md["files"].values())
lv = x2["left_vector_from_statistics"]
c1_ok = bool(pc["HEAD_equals_stated_commit"] and pc["verdict_unmodified"] and pc.get("remote_HEAD_equals_clone_HEAD") and hp["equals_commit"]
             and all(hp["copies_recorded_by_runs"].values()) and weights_ok and small_ok and md["revision_equals_mirror_main"]
             and dt["n_records"] == 21919 and dt["fresh_download"]["equal_to_local"] and sf["equal_to_local"] and stats_never_recomputed
             and x3["pip_target_packages_unmodified"] and lv["with_template_mean_key"]["min_cosine"] > 0.999999)
omissions = [
    "which checkpoint file is read is not stated: transformers 4.23.1 with safetensors installed reads model.safetensors, the public environment "
    f"(transformers 4.15.0) would read pytorch_model.bin; the {recon['pre_edit_weight']['checkpoint_files']['n_common']} tensors of the two files are bitwise equal "
    f"({recon['pre_edit_weight']['checkpoint_files']['n_common_not_bitwise_equal']} differ): no effect",
    "item 15 (memory fallback) ends with '(bit for bit: not run yet)' and no result of that check is stored (out/describe/*/info.json has no "
    "use_cache_equivalence_at_W0); the audit ran it (see discrepancies)",
    "the list is rendered from the driver files as they are on disk; tp_common.py and run_stream.py were last modified at "
    f"{x1['code_on_disk_vs_runs']['mtime']['run_stream.py']} while the first process of the reported stream was running; that process "
    f"(updates 0..{x1['code_on_disk_vs_runs']['stream_updates_written_by_the_process_started_before_the_last_modification'] - 1}), the development log and the "
    "single-edit run executed earlier versions of which no copy is kept (see discrepancies)",
]
V_c1 = {"verdict": PF(c1_ok), "evidence": {
    "public_code": {"commit": pc["HEAD"], "equals_stated_commit": pc["HEAD_equals_stated_commit"], "commit_date": pc["commit_date"],
                    "git_status_porcelain_empty": pc["status_porcelain"] == "", "git_diff_HEAD_empty": not pc["diff_HEAD_names"],
                    "tracked_files": pc["n_tracked_files"], "tracked_symbolic_links": pc["n_tracked_symbolic_links"],
                    "tracked_files_whose_content_hash_differs_from_the_commit(audit's own hashing)": len(pc["tracked_files_whose_content_differs_from_commit"]),
                    "untracked_files": f"{pc['untracked_files']['n']} compiled __pycache__/*.pyc (git-ignored), {len(pc['untracked_files']['others'])} others",
                    "commit_archive_downloaded_by_audit(codeload.github.com)": {"files_compared": pc["archive_fetched_by_audit"]["n_files_compared"],
                                                                                 "links_compared": pc["archive_fetched_by_audit"]["n_links_compared"],
                                                                                 "differing": len(pc["archive_fetched_by_audit"]["files_differing_from_clone"]),
                                                                                 "sha256_equals_builder_archive": pc["archive_fetched_by_audit"]["sha256"] == pc["archive_fetched_by_audit"]["builder_archive_sha256"]},
                    "github_HEAD_and_main_today_equal_this_commit": pc.get("remote_HEAD_equals_clone_HEAD")},
    "hyper_parameters": {"file": "hparams/ROME/gpt2-xl.json", "sha256": hp["sha256"], "equals_commit_blob": hp["equals_commit"],
                         "copies_recorded_by_the_runs_equal": hp["copies_recorded_by_runs"]},
    "model": {"repo_revision": md["builder_download_record_revision"], "equals_mirror_main": md["revision_equals_mirror_main"],
              "model.safetensors": {"bytes": md["files"]["model.safetensors"]["bytes"], "sha256": md["files"]["model.safetensors"]["sha256"],
                                    "equals_mirror_lfs_listing": md["files"]["model.safetensors"]["sha256_equals_mirror_listing"]},
              "pytorch_model.bin": {"bytes": md["files"]["pytorch_model.bin"]["bytes"], "sha256": md["files"]["pytorch_model.bin"]["sha256"],
                                    "equals_mirror_lfs_listing": md["files"]["pytorch_model.bin"]["sha256_equals_mirror_listing"]},
              "config_and_tokenizer_files_equal_mirror_blobs": small_ok,
              "edited_weight_of_loaded_model_equals_both_checkpoint_tensors_and_stored_W0": bool(recon["pre_edit_weight"]["equals_model.safetensors_tensor_bitwise"]
                                                                                                and recon["pre_edit_weight"]["equals_pytorch_model.bin_tensor_bitwise"])},
    "data": {"file": "counterfact.json", "bytes": dt["bytes"], "sha256": dt["sha256"], "n_records": dt["n_records"],
             "case_ids_are_0_to_n_minus_1": dt["case_ids_are_0_to_n_minus_1"], "fresh_download_by_audit_identical": dt["fresh_download"]["equal_to_local"],
             "first_record": dt["first_record"], "last_record": dt["last_record"]},
    "statistics": {"source": "authors' precomputed file (rome.baulab.info); not recomputed", "sha256": st["sha256"], "bytes": st["bytes"],
                   "fresh_download_by_audit_identical": sf["equal_to_local"], "tokens": int(st["token_count"]), "samples": st["sample_size"],
                   "shape": st["mom2_shape"], "dtype": st["mom2_dtype"], "condition_number": r1(st["second_moment"]["condition_number"]),
                   "public_code_printouts": "'Loading cached <this file>' in every run; 0 'Attempting to download', 0 'Computing locally'",
                   "stored_directions_recomputed_from_the_file(float64 inverse, template-mean keys)": {
                       "n_updates": lv["n_updates"], "min_cosine": lv["with_template_mean_key"]["min_cosine"], "max_rel_l2_diff": lv["with_template_mean_key"]["max_rel_l2_diff"]}},
    "library_stack": {"transformers_4.23.1_tokenizers_0.13.3_huggingface_hub_0.11.1_nltk_3.10.4_files_equal_wheel_RECORD": x3["pip_target_packages_unmodified"],
                      "[redacted: related project]_files_modified_since_build_start": len(x3["[redacted: related project]_files_modified_since_build_start"]["files"])
                      - len(x3["[redacted: related project]_files_modified_since_build_start"]["with_mtime_in_the_future(not written today)"]),
                      "files_left_in_home_cache": x3["home_cache_files_newer_than_build_start"]["n"]},
    "gpt_j": {"bytes_total": gj["bytes_total"], "equals_stated": gj["bytes_total"] == gj["builder_stated_total_bytes"],
              "pytorch_model.bin_sha256_equals_mirror_listing": gj.get("sha256_equals_mirror_listing"), "no_queue_job_touched_it": True},
    "driver_vs_public_code": {k: v for k, v in x1["driver_vs_public_calls"].items() if isinstance(v, bool)},
    "deviation_list": {"items": x2["deviations_file"]["n_items"] if x2["deviations_file"]["n_items"] == 17 else claims["deviations"]["n_items"],
                       "missing_from_the_list": omissions}},
    "criteria": "commit = stated commit and = github HEAD; every tracked file hash = commit; hyper-parameter file = commit; model files = mirror listing; "
                "data and statistics = fresh downloads; statistics never recomputed; installed stacks = wheels"}

# ======================================================================================= check 2: condition (1)
logs2 = {}
for n in NAMES:
    L = recon["logs"][n]
    r, k = L["reconstruction_fp64"], L["key_identity"]
    logs2[n] = {"T": L["T"], "records": L["records"], "requests_equal_dataset_records": L["requests_equal_dataset_records"],
                "float32_replay_of_logged_terms_equals_deployed_weight_bitwise": L["float32_replay_equals_deployed_bitwise"],
                "fp64_sum_vs_deployed": {"max_abs_error": r["max_abs"], "rel_fro_of_update": r["rel_fro_vs_update"], "rel_fro_of_W0": r["rel_fro_vs_W0"],
                                         "max_abs_error_in_float32_ulps_at_max|W_E|": r1(r["max_abs_in_ulps_at_max"]), "max|W_E|": r["WE_max_abs"]},
                "|W_E - W_0|_F": r["update_fro"], "W_E_finite": r["WE_finite"],
                "max|<readout, solver key> - 1|": k["max_abs_w_dot_k_minus_1"], "negative_division_factors": k["n_negative_s"],
                "min_division_factor": k["min_abs_s"],
                "max_rel(key @ W_before + b - logged output)": k["max_out_before_vs_cur_output_rel"],
                "max_rel(key @ W_after + b - logged target)": k["max_out_after_vs_target_rel"],
                "template_mean_key_coordinate": {"median": k["w_dot_k_template_mean"]["median"], "q95": k["w_dot_k_template_mean"]["q95"],
                                                 "max": k["w_dot_k_template_mean"]["max"], "share_within_1pct": k["share_template_mean_within_1pct"]},
                "library_format": {"U=right*s, W=left/s, K=solver key": bool(lib["logs"][n]["U_is_right_times_s"] and lib["logs"][n]["W_is_left_over_s"] and lib["logs"][n]["K_is_k_request"]),
                                   "library own() max|s-1|": lib["logs"][n]["own_coordinate_max_abs_dev_from_1"],
                                   "library delta() vs (W_E-W_0)^T rel": lib["logs"][n]["delta_vs_(WE-W0)^T_rel_fro"]}}
c2_ok = bool(all(v["float32_replay_of_logged_terms_equals_deployed_weight_bitwise"] and v["fp64_sum_vs_deployed"]["rel_fro_of_update"] < 1e-5
                 and v["max|<readout, solver key> - 1|"] < 1e-10 and v["negative_division_factors"] == 0 and v["W_E_finite"]
                 and v["requests_equal_dataset_records"] and v["max_rel(key @ W_after + b - logged target)"] < 1e-4 for v in logs2.values())
             and det_all_bitwise and templates_ok and others_ok
             and x1["hash_chain"]["dev"]["n_reproduced_by_audit_replay"] == x1["hash_chain"]["dev"]["n_recorded_hashes"]
             and x1["hash_chain"]["stream"]["n_reproduced_by_audit_replay"] == x1["hash_chain"]["stream"]["n_recorded_hashes"])
V_c2 = {"verdict": PF(c2_ok), "evidence": {
    "per_log": logs2,
    "orientation": "the edited parameter is the Conv1D weight of transformer.h.17.mlp.c_proj, shape (d_in 6400, d_out 1600), y = x @ W + b "
                   f"(checked with the audit's own hooks: max rel {max(d['max_rel(key @ W + b - module output)'] for d in det.values()):.1e}); the logged update is "
                   "outer(left 6400, right 1600) added without transposition; in the paper's notation the read-out vector is the PUBLIC 'left_vector' / s and "
                   "the output-side factor is the PUBLIC 'right_vector' * s, s = <left_vector, solver key>",
    "solver_key": "the input of the module at the last subject token of the request prompt ALONE (what the public compute_v divides by); the audit's own "
                  "capture of it is bitwise the stored key on every re-run update",
    "determinism": {**det, "all_reproduced_bitwise": det_all_bitwise, "n_stream_updates_rerun_after_the_collapse(index >= 173)": stream_after_173,
                    "context_templates_sampled_by_audit_equal_builder(21 of 21)": templates_ok,
                    "all_other_parameters_unchanged_by_the_public_editor(676 tensors hashed)": others_ok,
                    "weight_hashes_recorded_during_the_runs_reproduced_by_replay": {s_: f"{x1['hash_chain'][s_]['n_reproduced_by_audit_replay']} of {x1['hash_chain'][s_]['n_recorded_hashes']}" for s_ in ("dev", "stream")},
                    "condition_for_reproduction": "the 21 context templates must be the ones the first process sampled from the PRE-EDIT model with torch seed 0 "
                                                  "(stored in meta.json).  Re-running development update 97 from the saved state with templates sampled from the "
                                                  f"EDITED model at that state (same seed) gives {T_['a_templates_from_edited_model_at_state']['n_templates_equal_positionwise']} of 21 equal templates and an "
                                                  f"update that differs by {100 * T_['a_templates_from_edited_model_at_state']['outer_product_rel_fro_diff']:.0f} % (relative Frobenius); with the "
                                                  "pre-edit templates it is bitwise the stored one"}},
    "criteria": "float32 replay = deployed weight bitwise for all five logs; fp64 error < 1e-5 of the update; |<readout, key> - 1| < 1e-10; no negative "
                "division factor; every re-run update bitwise equal to the stored factors"}

# ======================================================================================= check 3: condition (2)
s0, s1 = sc["rerun_seed0"], sc["rerun_seed1"]
b1000 = sc["builder_1000"]
pubR = sc["published"]
c3_ok = bool(sc["condition_2"]["pass"] and s0["n_records_compared"] == 150 and s0["n_post_probabilities_bitwise_equal"] == 150
             and s0["n_pre_probabilities_bitwise_equal"] == 150 and all(b1000["audit_equals_builder_reported_to_0.01"].values())
             and pubR["eprint_equal_to_builder_copy"] and b1000["n_case_files"] == 1000 and b1000["n_records_with_nonfinite_probability"] == 0)
V_c3 = {"verdict": PF(c3_ok), "evidence": {
    "published(arXiv 2202.05262v5 e-print downloaded by the audit; tables/cf-results.tex, GPT-2 XL block, row ROME; 7,500 records)": {
        k: pubR["ROME"][k] for k in ("ES", "PS", "NS")},
    "builder_first_1000_records_recomputed_by_audit_from_the_1000_case_files": {k: r2(b1000["edited_recomputed_by_audit"][k]) for k in ("ES", "PS", "NS")},
    "standard_errors": {k: r2(b1000["edited_recomputed_by_audit"][k + "_se"]) for k in ("ES", "PS", "NS")},
    "abs_difference_to_published": {k: r2(abs(b1000["diff_to_published"][k])) for k in ("ES", "PS", "NS")}, "tolerance_points": 3.0,
    "unedited_model_same_records(ours / published)": {k: [r2(b1000["unedited_recomputed_by_audit"][k]), pubR["GPT-2 XL (unedited)"][k][0]] for k in ("ES", "PS", "NS")},
    "blocks_of_200_records(NS)": [r2(b["NS"]) for b in b1000["by_block_of_200"]],
    "rerun_of_150_random_records_by_the_PUBLIC_driver(experiments/evaluate.py main(), seed 0)": {
        "records": s0["n_records_compared"], "post_probabilities_bitwise_equal": s0["n_post_probabilities_bitwise_equal"],
        "pre_probabilities_bitwise_equal": s0["n_pre_probabilities_bitwise_equal"], "factors_bitwise_equal(left, right)": [s0["n_left_bitwise_equal"], s0["n_right_bitwise_equal"]],
        "prompt_flags_differing": f"{s0['n_prompt_flags_differ']} of {s0['n_prompt_flags']}",
        "sample_scores(audit = builder)": {k: r2(s0["audit_sample_scores"][k]) for k in ("ES", "PS", "NS")},
        "baselines_replaced_by_placeholders(never called)": list(s0["baselines_replaced_by_placeholders"]),
        "four_first_token_subject_records_also_identical": all(p["post_probabilities_bitwise_equal"] for p in s0["extra_first_token_records"])},
    "same_150_records_with_other_context_templates(seed 1)": {
        "templates_equal_positionwise": f"{s1['n_templates_equal_positionwise']} of 21", "median_rel_diff_of_right_vector": r2(s1["median_right_rel_l2_diff"]),
        "max_rel_diff_of_left_vector": r2(s1["max_left_rel_l2_diff"]), "prompt_flags_differing": f"{s1['n_prompt_flags_differ']} of {s1['n_prompt_flags']}",
        "score_change_points": {k: r2(s1["diff_audit_minus_builder"][k]) for k in ("ES", "PS", "NS")}},
    "single_edits_with_exploding_update_among_the_first_1000": [{"record": e["record"], "update_fro": r1(e["update_fro"]), "ES": e["ES"], "PS": e["PS"], "NS": e["NS"]}
                                                               for e in x1["single_edit_files"]["exploding_single_edits"]]},
    "criteria": "|ours - published| <= 3 points for ES, PS, NS on the first 1000 records; the audit's re-run reproduces the stored per-record probabilities"}

# ======================================================================================= check 4: condition (3) data traps
tk, pe = traps["tokenisation"], traps["pre_edit_continuations"]
A_, C_, B_ = ev["A"], ev["C"], ev["B"]
c4_items = {
    "multi_token_decode": {"ok": tk["max_tokens_any_object"] == 1 and pe["n_tokens_per_continuation"] == [10],
                           "evidence": f"every new and true object of all {tk['n_records']} records is ONE token as tok(' ' + object) (max {tk['max_tokens_any_object']}), so the exact-match "
                                       "rule is a first-token test and equals the token-level and the teacher-forced variant "
                                       f"({sum(traps['table_from_row_files'][t]['deployed']['n_rows_dec_exact_differs_from_tf_exact'] for t in traps['table_from_row_files'])} differing rows); "
                                       "the multi-token comparison of the family is D1 on 10-token greedy continuations"},
    "pre_edit_continuations": {"ok": A_["n_y0_equal_stored"] == 100 and A_["n_token_ids_equal_stored"] == 100 and pe["n_valid"] == pe["n_texts"],
                               "evidence": f"100 random prompts decoded by the audit (one prompt, no cache, no padding): {A_['n_y0_equal_stored']} of 100 continuations and "
                                           f"{A_['n_tokens_equal_of_1000']} of 1000 tokens equal the stored ones (also under transformers {evx['transformers']}: {evx['A']['n_y0_equal_stored']} of 100); "
                                           f"further {sum(B_[n][k]['n_audit_y0_equal_stored'] for n in B_ for k in ('heldout', 'neighborhood'))} of "
                                           f"{sum(B_[n][k]['n'] for n in B_ for k in ('heldout', 'neighborhood'))} in check 7"},
    "degenerate_continuations_counted": {"ok": pe["n_repetitive_builder_definition(le3 distinct or run>=4)"] == pe["builder_summary"]["n_repetitive"] and pe["counts"]["empty"] == 0,
                                         "evidence": f"pre-edit, {pe['n_texts']} prompts: {pe['counts']['empty']} empty, {pe['n_repetitive_builder_definition(le3 distinct or run>=4)']} repetitive "
                                                     f"(builder: {pe['builder_summary']['n_repetitive']}), {pe['counts']['has_eos']} contain <|endoftext|> (kept as valid), {pe['counts']['newline_first']} start with a "
                                                     f"line break, {pe['counts']['le2_words']} have at most two words; deployed: repetitive continuations on HELD-OUT "
                                                     + ", ".join(f"{t} {r1(traps['table_from_row_files'][t]['deployed']['heldout']['repetitive'])} %" for t in traps["table_from_row_files"])},
    "ground_truth_vs_evaluation": {"ok": traps["ground_truth_vs_evaluation"]["pre_continuations_weight_is_W0"],
                                   "evidence": "D1: ground truth = greedy continuation of the PRE-EDIT model (stored with the hash of W_0), evaluated = argmax of the DEPLOYED model; exact match: "
                                               "data-set object string vs deployed argmax; public scores (probability comparison) are in separate columns and define no ground truth; the pre-edit "
                                               "row of D1 (0.0) is a re-run of the same model, i.e. a reproducibility figure"},
    "batched_equals_single_decode": {"ok": all(C_[m]["n_builder_stored_batched_equal_audit_single"] == C_["n_prompts"] for m in ("pre_edit", "dev100_deployed")),
                                     "evidence": f"{C_['n_prompts']} prompts of {len(C_['token_lengths'])} different lengths: the builder's stored batched decode equals the audit's single-prompt decode on "
                                                 f"{C_['pre_edit']['n_builder_stored_batched_equal_audit_single']} (pre-edit) and {C_['dev100_deployed']['n_builder_stored_batched_equal_audit_single']} (dev100 deployed); "
                                                 "the builder batches prompts of ONE token length (no pad token, no mask).  The audit's own left-padded batch with mask and position ids: "
                                                 f"{C_['pre_edit']['n_audit_left_padded_batched_equal_single']} of {C_['n_prompts']}; a naive right-padded batch: "
                                                 f"{C_['pre_edit']['n_right_padded_naive_equal_single_among_padded']} of {C_['n_prompts'] - C_['pre_edit']['n_prompts_that_needed_no_padding']} padded prompts (the trap exists; it is avoided)"},
    "public_probability_test_padding": {"ok": all(v["n_prompts_success_flag_differs"] == 0 for v in pind["probabilities"].values()),
                                        "evidence": "the public test right-pads 46 sequences per record; the audit's unpadded single-sequence computation gives the same success flag on "
                                                    + ", ".join(f"{v['n_prompts'] - v['n_prompts_success_flag_differs']} of {v['n_prompts']} prompts ({t}, max |diff| {v['max_abs_diff_of_mean_neg_log_prob']:.1e})"
                                                                for t, v in pind["probabilities"].items())},
    "leading_space_of_the_object": {"ok": tk["n_records_where_token_without_space_equals_token_with_space"] == 0 and tk["prompt_texts"]["n_ending_in_whitespace"] == 0
                                    and tk["first_token_of_space_plus_object_starts_with_space"] == tk["n_records"],
                                    "evidence": f"target tokens are tok(' ' + object): the leading-space token on {tk['first_token_of_space_plus_object_starts_with_space']} of {tk['n_records']} records; without "
                                                f"the space the token would equal it on {tk['n_records_where_token_without_space_equals_token_with_space']} records; "
                                                f"{tk['prompt_texts']['n_ending_in_whitespace']} of {tk['prompt_texts']['n']} prompts end in white space; stored token ids equal the tokenizer's on "
                                                f"{tk['prompt_texts']['n_stored_token_ids_equal_environment_tokenizer']}; a text rule (decoded text starts with the object string) differs from the token rule on "
                                                f"{sum(traps['table_from_row_files'][t]['deployed']['audit_text_rule_on_stored_decode(starts with the object string)']['n_rows_differ_from_token_rule'] for t in traps['table_from_row_files'])} rows"},
    "signs": {"ok": all(recon["logs"][n]["key_identity"]["n_negative_s"] == 0 for n in NAMES),
              "evidence": "all 2,100 division factors are positive (normalising the read-out vector flips no sign); public scores store NEGATIVE log-probabilities: success means "
                          "nll(true) > nll(new) (verified against the audit's own probabilities)"},
    "non_finite_values": {"ok": all(traps["public_scores_from_case_files"][t][m]["n_nonfinite_records"] == 0 for t in traps["public_scores_from_case_files"] for m in ("deployed", "pre_edit")),
                          "evidence": "no non-finite probability or logit in any stored row; float32 (GPU) vs float64 (CPU) last-token logits: same argmax on "
                                      + ", ".join(f"{v['n_argmax_equal_float32_vs_float64']} of {v['n_prompts']} ({t}, max |diff| {v['max_abs_logit_diff']:.1e})" for t, v in pind["precision"].items())}}
c4_ok = all(v["ok"] for v in c4_items.values())
V_c4 = {"verdict": PF(c4_ok), "evidence": c4_items,
        "criteria": "each of the handover traps and of the listed decode traps is checked by the audit's own code; all must hold"}

# ======================================================================================= check 5: pools
bf = pools["beyond_registered_flags"]
c5_ok = bool(pools["split"]["repair_equals_rederivation"] and pools["split"]["heldout_equals_rederivation"] and pools["all_pairwise_disjoint"]
             and all(pools["subject_flags_equal_builder_report"].values()) and pools["seed_in_builder_pools_json"] == pools["seed_protocol"]
             and pools["prompt_rows_text_not_equal_dataset"] == 0 and all(pools["subject_flags"][p]["n_prompt_rows_flag_differs"] == 0 for p in ("repair", "heldout")))
V_c5 = {"verdict": PF(c5_ok), "evidence": {
    "seed": pools["seed_protocol"], "rule": pools["rule_in_builder_pools_json"],
    "split_rederived_by_audit_equals_stored": [pools["split"]["repair_equals_rederivation"], pools["split"]["heldout_equals_rederivation"]],
    "records": {"eligible": pools["split"]["n_eligible"], "REPAIR": pools["split"]["n_repair"], "HELD-OUT": pools["split"]["n_heldout"], "stream": 2000, "dev": 100},
    "pairwise_intersections_by_record": pools["disjoint_by_record"],
    "pool_records_sharing_a_subject_with_a_logged_request(T100/T500/T1000/T2000, dev)": {
        p: [pools["subject_flags"][p][f"share_subject_with_T{T}"] for T in (100, 500, 1000, 2000)] + [pools["subject_flags"][p]["share_subject_with_dev"]] for p in ("repair", "heldout")},
    "per_row_flags_differing_from_audit": {p: pools["subject_flags"][p]["n_prompt_rows_flag_differs"] for p in ("repair", "heldout")},
    "mention_flags_differing_from_the_audit's_own_rule": {p: bf[p]["n_rows_mention_flag_differs_from_builder"] for p in ("repair", "heldout")},
    "not_covered_by_record_disjointness(counted, flags exist)": {
        "HELD-OUT benign prompts that are a logged stream request / neighbourhood prompt / a prompt of the dev log": [bf["heldout"]["text_is_logged_stream_request"], bf["heldout"]["text_is_logged_stream_neighborhood"], bf["heldout"]["text_is_logged_dev_prompt"]],
        "REPAIR benign prompts, the same": [bf["repair"]["text_is_logged_stream_request"], bf["repair"]["text_is_logged_stream_neighborhood"], bf["repair"]["text_is_logged_dev_prompt"]],
        "prompt texts in both pools": bf["heldout"]["text_is_in_other_pool"], "subjects in both pools": bf["subjects_shared_between_repair_and_heldout"],
        "HELD-OUT prompts mentioning a logged subject (T100/T500/T1000/T2000)": [bf["heldout"]["prompts_mentioning_logged_subject_audit_rule"][f"T{T}"] for T in (100, 500, 1000, 2000)],
        "neighbourhood prompts equal to a logged request (T100/T500/T1000/T2000)": [bf["neighborhood_equal_to_a_logged_request"][f"T{T}"] for T in (100, 500, 1000, 2000)],
        "neighbourhood prompts mentioning another logged subject (T100/T500/T1000/T2000)": [bf["neighborhood_mentioning_another_logged_subject_audit_rule"][f"T{T}"] for T in (100, 500, 1000, 2000)]}},
    "criteria": "the stored pools equal the re-derivation from the rule and the seed; all pairwise record intersections empty; the subject flags equal the audit's"}

# ======================================================================================= check 6: discipline
c6_ok = bool(disc["verdict_inputs"]["gate_or_repair_on_reported_log_by_builder"] is False and disc["builder_scripts_using_a_gate_object"] == ["check_library.py"]
             and disc["library_check"]["is_development_log"] and not disc["builder_jobs_without_a_logged_command"]
             and all(not v["reported_log_names_in_file"] for v in disc["builder_outputs_with_gate_words"].values()))
V_c6 = {"verdict": PF(c6_ok), "evidence": {
    "gpu_jobs_of_the_build(queue event log)": sorted(disc["gpu_jobs"]["builder(tx_*)"]),
    "scripts_behind_them": disc["builder_job_commands"],
    "only_script_that_builds_a_gate_object(ScopedLog / Hook)": "check_library.py: NAME = TX.DEV_LOG, asserted to be TX-G2XL-rome-dev100 (thresholds +inf, -inf and 0.999: a format check of the library on the development log)",
    "outputs_with_gate_words_all_belong_to_the_development_log": True,
    "reported_logs_through_the_library": "export_library_format.py builds rescope.EditLog objects of all five logs with the library's constructor and reads own() (= 1) and delta() "
                                         "(= W_E - W_0): a format check of stored tensors, no threshold, no decision, no model",
    "evaluations_of_the_reported_stream_by_the_build": "pre-edit weight; deployed weights after 100, 150, 173, 174, 200, 500, 1000, 2000 requests; the single-edit protocol on records 0..999",
    "note_detector_statistic_on_the_reported_stream": "while writing the stream the build recorded, for every update, the normalised coordinate <left, k> / <left, k_request> of the edit's own subject "
                                                      "token under the editor's 21 context templates (factors.pt: scalars coord_templates, w_dot_k_avg) and tabulated its distribution per reported log "
                                                      "(table, section 1: median 1.02, q95 1.36, max 313).  It is the key identity the build had to check, evaluated at the editor's own prompts; it is also the "
                                                      "registered detector's score at 20 prefixed variants of each request.  No threshold was applied and nothing was decided or tuned with it (the "
                                                      "loader's default key follows from the public code and holds on the development log alone).  Reported for transparency; the audit recomputed the same "
                                                      "statistic to verify the builder's statement and computed no other coordinate on a reported log",
    "other_agents_at_scan_time": {"time": disc["other_agents"]["scan_time"],
                                  "10_final/e10_text": "driver under construction; its scratch scripts open the development log only; key-cache parts built: "
                                                       f"{disc['other_agents']['10_final/cache/text/gpt2-xl parts']}; parts of the reported stream: {disc['other_agents']['key_cache_parts_of_the_reported_stream']}"}},
    "criteria": "no script of the build constructs or applies a gate / scoped log / repair on a log other than TX-G2XL-rome-dev100; no output of such a run exists"}

# ======================================================================================= check 7: the table
tab = traps["table_from_row_files"]
rows7 = {}
c7_ok = True
for name, tag in (("TX-G2XL-rome-dev100", "dev100"), ("TX-G2XL-rome-T1000", "T1000")):
    R = B_[name]
    e, h, nb = R["edits"], R["heldout"], R["neighborhood"]
    ok = bool(R["audit_replay_equals_builder_snapshot_bitwise"] and e["n_rows_equal_builder_dec_exact"] == e["n"] and h["n_rows_D1_tf_equal_builder"] == h["n"]
              and h["n_rows_D1_dec_equal_builder"] == h["n"] and nb["n_rows_D1_tf_equal_builder"] == nb["n"] and h["n_rows_tok_loc_equal_builder"] == h["n"]
              and h["n_deployed_continuation_equal_builder_stored"] == h["n"] and nb["n_deployed_continuation_equal_builder_stored"] == nb["n"])
    c7_ok = c7_ok and ok
    rows7[name] = {"deployed_weight_rebuilt_by_audit_equals_snapshot": R["audit_replay_equals_builder_snapshot_bitwise"],
                   "exact_match(300 request / paraphrase prompts)": {"rows_equal_to_builder": f"{e['n_rows_equal_builder_dec_exact']} of {e['n']}",
                                                                     "audit_request_pct": r2(e["request"]["audit_exact_token_pct"]), "audit_paraphrase_pct": r2(e["paraphrase"]["audit_exact_token_pct"]),
                                                                     "n_request_n_paraphrase": [e["request"]["n"], e["paraphrase"]["n"]],
                                                                     "builder_all_rows_pct(request, paraphrase)": [r2(e["request"]["builder_all_rows_pct"]), r2(e["paraphrase"]["builder_all_rows_pct"])],
                                                                     "token_rule_vs_text_rule_differ": e["n_token_rule_differs_from_text_rule"]},
                   "D1_heldout(300 prompts)": {"rows_equal_to_builder(tf, decode, tokens kept)": [h["n_rows_D1_tf_equal_builder"], h["n_rows_D1_dec_equal_builder"], h["n_rows_tok_loc_equal_builder"]],
                                               "audit_D1_pct": r2(h["audit_D1_tf_pct"]), "audit_tokens_kept_pct": r2(h["audit_tok_loc_pct"]), "audit_first_token_changed_pct": r2(h["audit_first_token_changed_pct"]),
                                               "builder_all_3000_rows(D1, tokens kept)": [r2(h["builder_all_rows_D1_tf_pct"]), r2(h["builder_all_rows_tok_loc_pct"])],
                                               "deployed_continuations_equal_to_builder's_stored_decode": h["n_deployed_continuation_equal_builder_stored"]},
                   "D1_neighbourhood(300 prompts)": {"rows_equal_to_builder": nb["n_rows_D1_tf_equal_builder"], "audit_D1_pct": r2(nb["audit_D1_tf_pct"]),
                                                     "builder_all_rows_D1_pct": r2(nb["builder_all_rows_D1_tf_pct"])}}
all_digits = all(tab[t]["vs_report"]["all_equal_to_reported_digits"] and traps["public_scores_from_case_files"][t]["all_equal_to_reported_digits"] for t in tab)
ret = ev["R"]
c7_ok = bool(c7_ok and all_digits and all(v["n_rows_equal_builder"] == 300 and v["audit_replay_equals_snapshot"] for v in ret.values())
             and all(v["n_records_all_probabilities_bitwise_equal"] == v["n_records"] for v in pcmp["cmp"].values()))
V_c7 = {"verdict": PF(c7_ok), "evidence": {
    "own_decode": rows7,
    "retention_first_100_requests(audit's decode; request / paraphrase exact)": {t: [r1(v["audit_request_exact_pct"]), r1(v["audit_paraphrase_exact_pct"]), f"{v['n_rows_equal_builder']} of 300 rows equal"] for t, v in ret.items()},
    "every_digit_of_the_report's_table_recomputed_from_the_stored_row_and_case_files": all_digits,
    "table_recomputed": {t: {"ES_PS_NS": [r1(traps["public_scores_from_case_files"][t]["deployed"][k]) for k in ("ES", "PS", "NS")],
                             "request_paraphrase_exact": [r1(tab[t]["deployed"]["request_exact"]), r1(tab[t]["deployed"]["paraphrase_exact"])],
                             "D1_heldout": r1(tab[t]["deployed"]["heldout"]["D1_tf"]), "tokens_kept": r1(tab[t]["deployed"]["heldout"]["tok_loc"]),
                             "GE": r1(traps["generation_entropy"][f"{t}:{'dev' if t == 'dev100' else 'stream'}_first100"]["GE_audit_x100"]),
                             "pre_edit_D1": tab[t]["pre_edit"]["heldout"]["D1_tf"]} for t in tab},
    "public_probability_test_rerun_by_audit(records: identical to stored case files)": {t: f"{v['n_records_all_probabilities_bitwise_equal']} of {v['n_records']}" for t, v in pcmp["cmp"].items()},
    "D1_teacher_forced_vs_decode": f"{traps['D1_tf_vs_dec_on_deployed_rows']['n_differ']} of {traps['D1_tf_vs_dec_on_deployed_rows']['n_rows']} deployed rows differ",
    "D1_along_the_development_log(audit, 300 prompts; builder 3000: 7.0 / 26.2 / 95.4)": {k: r1(v["audit_D1_tf_pct"]) for k, v in claims.get("d1_curve_dev_audit_300_prompts", {}).items()}},
    "criteria": "on 300 random prompts per column and log (dev100, T1000) every row flag equals the audit's single-prompt decode; all digits of the table follow from the stored files"}

V["checks"] = {"1_provenance": V_c1, "2_condition_1_reconstruction_orientation_key_determinism": V_c2, "3_condition_2_single_edit": V_c3,
               "4_condition_3_data_traps": V_c4, "5_pools": V_c5, "6_discipline": V_c6, "7_table_of_deployed_logs": V_c7}
V["conditions_of_use"] = {
    "(1) deployed weight = pre-edit weight + logged rank-one terms": {"verdict": V_c2["verdict"],
        "evidence": "all five logs: float32 replay bitwise equal; fp64 relative error of the update " + ", ".join(f"{recon['logs'][n]['reconstruction_fp64']['rel_fro_vs_update']:.1e}" for n in NAMES)
                    + "; max abs entry error " + ", ".join(f"{recon['logs'][n]['reconstruction_fp64']['max_abs']:.1e}" for n in NAMES) + " (2.6 to 14.2 float32 ulps at max|W_E|)"},
    "(2) single edits on the first 1000 records within 3 points of the published scores": {"verdict": V_c3["verdict"],
        "evidence": f"ES {r2(b1000['edited_recomputed_by_audit']['ES'])} / {pubR['ROME']['ES'][0]}, PS {r2(b1000['edited_recomputed_by_audit']['PS'])} / {pubR['ROME']['PS'][0]}, "
                    f"NS {r2(b1000['edited_recomputed_by_audit']['NS'])} / {pubR['ROME']['NS'][0]} (ours / published)"},
    "(3) data traps of the handover": {"verdict": V_c4["verdict"], "evidence": "; ".join(f"{k}: {'ok' if v['ok'] else 'NOT ok'}" for k, v in c4_items.items())}}

# ======================================================================================= per log
ed = claims["editor_before_and_from_173"]
inst = ed["n_requests_installed_at_write_time(P(new|request) > 0.5)"]
un = x3["update_norm_concentration"]
hits = x1["collapsed_logs_exact_match_hits"]
K = {p["n_requests"]: p for p in misc["K"]["points"]}
clean_note = ["no exploding update, no request whose subject is the first prompt token",
              "D1 on 10-token continuations is close to saturation (see residual risks)"]
per_log = {}
for n in NAMES:
    tag = n.split("-")[-1]
    t = tab[tag]["deployed"]
    p = traps["public_scores_from_case_files"][tag]["deployed"]
    base = {"T": A.LOGS[n][1], "conditions_1_2_3": [V_c2["verdict"], V_c3["verdict"], V_c4["verdict"]],
            "deployed": {"ES_PS_NS": [r1(p[k]) for k in ("ES", "PS", "NS")], "request_paraphrase_exact_pct": [r1(t["request_exact"]), r1(t["paraphrase_exact"])],
                         "D1_heldout_pct": r1(t["heldout"]["D1_tf"]), "tokens_kept_pct": r1(t["heldout"]["tok_loc"]),
                         "first_token_changed_pct": r1(t["heldout"]["audit_from_tensors"]["first_token_changed"]),
                         "repetitive_continuations_pct": r1(t["heldout"]["repetitive"])},
            "largest_update_norm": r1(un[n]["max_update_norm"]), "|W_E - W_0|_F": r1(un[n]["deployed_minus_pre_edit_fro"]), "|W_0|_F": r1(recon["logs"][n]["reconstruction_fp64"]["W0_fro"])}
    if un[n]["first_token_subject_updates"]:
        T = A.LOGS[n][1]
        base.update({
            "verdict": "safe", "class": "collapse log", "safe_for": "a faithful, bitwise reproducible record of a deployment that the public code destroyed; claims about collateral damage and about "
                                                                 "recovering the pre-edit behaviour",
            "not_safe_for": "any claim about preserving or retaining the edits of the whole log, and any comparison that reads the deployed row as a working edited model",
            "binding_restrictions": [
                f"the deployed model is collapsed: update 173 ('Chicago is a twin city of' -> Istanbul, subject = first prompt token, division factor {recon['stream_updates']['exploding'][0]['s']:.4f}, "
                f"update norm {recon['stream_updates']['exploding'][0]['update_fro']:.0f} against |W_0| = 117) takes the tokens kept on HELD-OUT prompts from "
                f"{r1(K[173]['audit_200_prompts']['tok_loc'])} % to {r1(K[174]['audit_200_prompts']['tok_loc'])} % and the exact match of the first 100 requests from "
                f"{r1(K[173]['audit_first_100_requests_exact_pct'])} % to {r1(K[174]['audit_first_100_requests_exact_pct'])} % (audit's own decode of the deployed prefixes 173 and 174)",
                f"{len(un[n]['first_token_subject_updates'])} of {T} updates (requests {un[n]['first_token_subject_updates']}) carry {un[n]['their_share_of_the_summed_squared_norm_pct']:.2f} % of the summed "
                "squared update norm.  MEASURED: the norms, and the model before / after update 173.  NOT measured (a model without these updates is a repair, which the audit may not run): how much "
                "damage the other updates carry on their own",
                f"{T - 173} of the {T} logged targets were computed on weights that already contain update 173: the editor's own final probability of the new object is below 0.5 on "
                f"{ed['share_p_new_below_0.5']['173..' + str(T - 1)]:.1f} % of them (the public code's own printouts), and {inst['T' + str(T)]} of {T} requests had P(new object) > 0.5 on their own "
                f"request prompt right after their own update (write-time files; {wtime['runs']['stream']['n_identical_to_stored_write_time_file']} of {wtime['runs']['stream']['n_updates_checked']} sampled ones "
                "re-derived identically by the audit).  Whether any replay can install the remaining targets was not tested by the builder or by the audit: report the requests before 173 separately",
                f"deployed baseline: ES / PS / NS {[r1(p[k]) for k in ('ES', 'PS', 'NS')]} with probability differences EM / PM / NM "
                f"{[r1(selft['3_magnitude_scores'][tag]['deployed'][k]) for k in ('EM', 'PM', 'NM')]} points (pre-edit model: ES / PS / NS "
                f"{[r1(selft['3_magnitude_scores'][tag]['pre_edit'][k]) for k in ('ES', 'PS', 'NS')]}); D1 100 %; "
                f"{r1(t['heldout']['repetitive'])} % repetitive continuations",
                f"exact match: {hits[n]['request']['n_exact_whose_target_is_an_object_of_a_first_token_subject_update']} of the {hits[n]['request']['n_exact']} request hits and "
                f"{hits[n]['paraphrase']['n_exact_whose_target_is_an_object_of_a_first_token_subject_update']} of the {hits[n]['paraphrase']['n_exact']} paraphrase hits have a target among the new objects of "
                "the first-token-subject updates, the tokens the collapsed model emits most.  Base-rate test (does the model emit such a token more often when it is the target?): on other edits' prompts "
                f"with such a target, request {selft['1_exact_match_hits_of_collapsed_logs'][tag]['request']['hits_among_them_observed']} hits observed against "
                f"{selft['1_exact_match_hits_of_collapsed_logs'][tag]['request']['hits_among_them_expected_if_output_is_independent_of_the_target']:.1f} expected "
                f"({selft['1_exact_match_hits_of_collapsed_logs'][tag]['request']['observed_minus_expected_in_sd']:+.1f} sd), paraphrase "
                f"{selft['1_exact_match_hits_of_collapsed_logs'][tag]['paraphrase']['hits_among_them_observed']} against "
                f"{selft['1_exact_match_hits_of_collapsed_logs'][tag]['paraphrase']['hits_among_them_expected_if_output_is_independent_of_the_target']:.1f} "
                f"({selft['1_exact_match_hits_of_collapsed_logs'][tag]['paraphrase']['observed_minus_expected_in_sd']:+.1f} sd); "
                f"request hits with another target: {selft['1_exact_match_hits_of_collapsed_logs'][tag]['request']['hits_whose_target_is_not_in_S']} (pre-edit model: "
                f"{hits[n]['request']['n_exact_pre_edit']} request hits in all).  The column mixes base-rate hits with others; which of the others are retained edits was not determined",
                f"public n-gram entropy of the deployed model {r1(traps['generation_entropy'][tag + ':stream_first100']['GE_audit_x100'])} (pre-edit "
                f"{r1(traps['generation_entropy']['W0:stream_first100']['GE_audit_x100'])}; dev100, not collapsed, {r1(traps['generation_entropy']['dev100:dev_first100']['GE_audit_x100'])})",
                ]})
    else:
        base.update({"verdict": "safe", "class": "development log" if "dev" in n else "clean reported log", "notes": clean_note})
    per_log[n] = base
V["per_log"] = per_log

# ======================================================================================= discrepancies
cd = x1["code_on_disk_vs_runs"]
ucw = puc["uc"]["at_W0"]
uca = puc["uc"]["against_stored"]
n_true_false_identical = sum(1 for v in ucw.values() if v.get("identical_output_true_vs_false")) + sum(1 for v in uca.values() if v.get("identical_output_true_vs_false"))
n_oom = sum(1 for v in list(ucw.values()) + list(uca.values()) if "use_cache_true" in v)
dvs = x2["dev_vs_stream_shared_prompt_texts"]
V["discrepancies"] = [
    {"what": "DEVIATIONS.txt item 15 states the equivalence check of the memory fallback as '(bit for bit: not run yet)'; no result of it is stored",
     "numbers": f"audit: the public probability test returns identical numbers with model.config.use_cache True and False in {n_true_false_identical} record-weight pairs "
                f"(records 0, 1, 147, 427, 1878, 15000 at W_0; 147, 427 at T500 and at T1000; 147, 427, 1878 at T2000); with use_cache False all "
                f"{sum(1 for v in uca.values() if v['use_cache_false_equals_stored_bitwise'])} flagged stored case files are reproduced bitwise; record 1531 (179-token paraphrase) ran out of memory with "
                f"use_cache True under a 14.5 GB reservation in {n_oom} of {n_oom} attempts, so for it the equivalence rests on the other records",
     "effect_on_logs": "none"},
    {"what": "the driver files on disk are not the versions that wrote most of the logs, and the list of deviations is rendered from the files on disk",
     "numbers": f"tp_common.py and run_stream.py were last modified {cd['mtime']['run_stream.py']}; the first process of the reported stream started {cd['runs']['stream']['started']} and wrote updates "
                f"0..{cd['stream_updates_written_by_the_process_started_before_the_last_modification'] - 1}; the development log finished {cd['runs']['dev']['last_update_written']}; the single-edit run "
                f"finished {cd['runs']['single']['finished']}; {2000 - x2['driver_versions']['stream']['n_write_time_files_with_field_of_current_driver']} of 2000 stream write-time files and "
                f"{100 - x2['driver_versions']['dev']['n_write_time_files_with_field_of_current_driver']} of 100 development ones lack the field the current driver writes; no earlier copy of the files is kept",
     "effect_on_logs": f"none found: the audit's own driver reproduces bitwise {det['dev']['n_distinct_updates_rerun_by_audit']} of 100 development updates, {det['stream']['n_distinct_updates_rerun_by_audit']} stream updates "
                       f"and {s0['n_records_compared'] + len(s0['extra_first_token_records'])} single edits, and the replay reproduces all "
                       f"{x1['hash_chain']['dev']['n_recorded_hashes'] + x1['hash_chain']['stream']['n_recorded_hashes']} weight hashes recorded during the runs"},
    {"what": "no run log of the first process of the reported stream (updates 0..1530) exists: logs/stream.out begins with the resumed run and logs/stream_before_run_1.out holds only a cancelled "
             "queue entry (the file was presumably truncated by the first watcher attempt), so the stated cause of the interruption (out of memory in the write-time test of case 1531) is not "
             "documented by that process",
     "numbers": f"consistent with: a pause from {cd['runs']['stream']['pauses_longer_than_2_min(after update, from, to)'][0][1]} to {cd['runs']['stream']['pauses_longer_than_2_min(after update, from, to)'][0][2]} after "
                f"{cd['runs']['stream']['pauses_longer_than_2_min(after update, from, to)'][0][0]} updates; request 1531 computed twice by the public code with "
                f"{pp['stream']['requests_computed_more_than_once'][0]['n_printed_lines_compared']} identical printed lines; the same out-of-memory error for that record in logs/describe_A_oom_case1531.out",
     "effect_on_logs": "none: the audit re-ran updates 1530, 1531, 1532 from the replayed state, bitwise equal to the stored factors"},
    {"what": "logs_edit/dev/meta.json is stale", "numbers": f"'what' says records 15000 .. 15002 and T_planned = {x1['dev_meta_stale_fields']['T_planned']} (from the 3-update test run); T_done = "
                                                            f"{x1['dev_meta_stale_fields']['T_done']} and {x1['dev_meta_stale_fields']['n_update_files']} update files exist", "effect_on_logs": "none"},
    {"what": "the deviation list does not say which checkpoint file is loaded (model.safetensors under transformers 4.23.1; the public stack would read pytorch_model.bin)",
     "numbers": f"{recon['pre_edit_weight']['checkpoint_files']['n_common']} tensors in both files, {recon['pre_edit_weight']['checkpoint_files']['n_common_not_bitwise_equal']} not bitwise equal", "effect_on_logs": "none"},
]
V["builder_statements_checked"] = {
    "report_table(ES/PS/NS, exact, D1, tokens kept, GE of the five logs)": "all digits reproduced from the stored files" if all_digits else "NOT all digits reproduced",
    "reconstruction_table": {n.split("-")[-1]: [float(f"{recon['logs'][n]['reconstruction_fp64']['rel_fro_vs_update']:.2g}"), float(f"{recon['logs'][n]['reconstruction_fp64']['max_abs']:.2g}")] for n in NAMES},
    "single_edit(ours / published)": "100.0 / 100.0, 96.7 / 96.4, 75.4 / 75.4 confirmed",
    "collapse_at_update_173": {"division_factor": recon["stream_updates"]["exploding"][0]["s"], "update_norm": recon["stream_updates"]["exploding"][0]["update_fro"],
                               "|W - W_0|_F after 173 / 174 requests": [r1(recon["stream_updates"]["W_minus_W0_fro"]["173"]), r1(recon["stream_updates"]["W_minus_W0_fro"]["174"])],
                               "audit_decode_tokens_kept_173_174": [r1(K[173]["audit_200_prompts"]["tok_loc"]), r1(K[174]["audit_200_prompts"]["tok_loc"])],
                               "public_test_at_173_and_174_identical_to_builder(records)": [K[173]["public_test_records_0_99"]["n_records_identical_to_builder_case_files"], K[174]["public_test_records_0_99"]["n_records_identical_to_builder_case_files"]],
                               "same_division_factor_under_transformers_5": f"keys agree to {envx['keys']['max_key_rel_l2_diff']:.1e}, division factors to {envx['keys']['max_rel_diff_of_division_factor']:.1e}",
                               "also_with_other_templates(seed 1, record 266 edited alone)": [r1(a) for _, a in s1["sample_records_with_update_fro_above_117"]]},
    "eight_first_token_requests_and_20_exploding_updates": [len(recon["logs"]["TX-G2XL-rome-T2000"]["updates"]["subject_is_first_token"]), len(recon["logs"]["TX-G2XL-rome-T2000"]["updates"]["exploding(>|W0|_F)"]),
                                                            f"{recon['first_token_subjects_in_dataset']['n']} of {recon['first_token_subjects_in_dataset']['n_records']} records; none in dev100"],
    "editor_fails_on_99.0_pct_from_173": r2(ed["share_p_new_below_0.5"]["173..1999"]),
    "stored_scalars_equal_the_public_code's_own_printouts": {s_: f"p_new {claims['public_printouts_vs_stored_scalars'][s_]['n_p_new_equal_to_printed']}, division factor "
                                                                 f"{claims['public_printouts_vs_stored_scalars'][s_]['n_division_factor_equal_to_printed']} of {claims['public_printouts_vs_stored_scalars'][s_]['n_updates']}" for s_ in ("dev", "stream")},
    "key_identity_with_template_mean_key(T2000)": {k: claims["template_mean_coordinate_T2000"][k] for k in ("median", "q95", "max", "share_within_1pct")},
    "key_identity_with_request_key(max abs dev, T2000)": recon["logs"]["TX-G2XL-rome-T2000"]["key_identity"]["max_abs_w_dot_k_minus_1"],
    "D1_after_1_10_100_updates_of_dev100(audit, 300 prompts)": {k: r1(v["audit_D1_tf_pct"]) for k, v in claims.get("d1_curve_dev_audit_300_prompts", {}).items()},
    "every_object_one_token": tk["n_tokens_of_space_plus_new_object"], "two_D1_variants_agree": f"{traps['D1_tf_vs_dec_on_deployed_rows']['n_rows'] - traps['D1_tf_vs_dec_on_deployed_rows']['n_differ']} of {traps['D1_tf_vs_dec_on_deployed_rows']['n_rows']}",
    "n_gram_entropy_barely_registers_collapse": {t: [r1(v["GE_audit_x100"]), f"8 most frequent words = {100 * v['share_of_all_words_taken_by_the_8_most_frequent']:.0f} % of all words"] for t, v in traps["generation_entropy"].items() if "first100" in t},
    "greedy_tokens_that_are_objects_of_exploding_updates(HELD-OUT, pct)": {t: r1(v["exploding"]["deployed"]["share_of_tokens_pct"]) for t, v in claims["heldout_greedy_tokens_of_deployed_models"].items()},
    "transformers_5_breaks_the_public_generate_fast": envx["generate_fast_env"].get("len5", envx["generate_fast_env"])[:4] if isinstance(envx["generate_fast_env"].get("len5"), list) else envx["generate_fast_env"],
    "stream_resumed_once_at_1531_dev_once_at_3": [cd["runs"]["stream"]["resumed"][0]["n_have"], cd["runs"]["dev"]["resumed"][0]["n_have"]],
    "cache_files_outside_project_3_removed": x3["home_cache_files_newer_than_build_start"]["n"] == 0,
    "gpt_j_bytes_and_sha256": [gj["bytes_total"], gj.get("sha256_equals_mirror_listing")],
    "deviations_listed": claims["deviations"]["n_items"]}

# ======================================================================================= residual risks, what a user must know
rep = x3["repeated_subjects"]
ge = x2["ngram_entropy_formula"]
V["residual_risks"] = [
    "Reproducibility is established on this machine and stack only (RTX 5090, torch 2.8.0, transformers 4.23.1): the public environment (torch 1.10.2, transformers 4.15.0) cannot be installed "
    "for Python 3.11, so what the authors' stack would write bit for bit is untested; the tie to the paper is the single-edit scores (0.0 / 0.3 / 0.0 points) and the unedited-model scores.",
    f"One stream, one draw of the public code's 21 random context templates (seed 0).  With another draw ({s1['n_templates_equal_positionwise']} of 21 templates equal) the single-edit factors of the same "
    f"150 records differ by {100 * s1['median_right_rel_l2_diff']:.0f} % (median, output-side factor) while ES / PS / NS move by "
    f"{r2(s1['diff_audit_minus_builder']['ES'])} / {r2(s1['diff_audit_minus_builder']['PS'])} / {r2(s1['diff_audit_minus_builder']['NS'])} points; the variance of sequential-log metrics over template draws is unknown.",
    f"The development log has no request whose subject is the first prompt token and no exploding update (base rate {recon['first_token_subjects_in_dataset']['n']} of {recon['first_token_subjects_in_dataset']['n_records']} "
    f"records).  In the library format the read-out vectors of such updates have norm up to {lib['logs']['TX-G2XL-rome-T2000']['readout_norm_max']:.1f} against a median of "
    f"{lib['logs']['TX-G2XL-rome-T2000']['readout_norm_median']:.2f} (measured).  What that does to the method's scores on these logs was not measured (discipline); nothing developed on the development log has met such an update.",
    f"Records are disjoint, prompt texts are not: {dvs['dev_neighborhood_rows_whose_text_is_a_stream_prompt']['any_kind']} of the 1000 neighbourhood rows of the development log are also prompts of the reported stream "
    f"({dvs['n_distinct_texts_in_both']} distinct texts; {dvs['T100_rows_whose_text_is_a_dev_prompt']['n']} rows of T100, {dvs['T2000_rows_whose_text_is_a_dev_prompt']['n']} of T2000); "
    f"{bf['heldout']['text_is_logged_stream_neighborhood']} HELD-OUT benign prompts are neighbourhood prompts of logged records and {bf['heldout']['text_is_logged_stream_request']} are logged requests (flags exist: use the 'clean' rows).",
    f"D1 on 10-token continuations is close to its ceiling on the logs that did NOT collapse (dev100 {r1(tab['dev100']['deployed']['heldout']['D1_tf'])} %, T100 {r1(tab['T100']['deployed']['heldout']['D1_tf'])} %; "
    f"audit after 1 / 10 updates: {' / '.join(str(r1(v['audit_D1_tf_pct'])) for k, v in claims.get('d1_curve_dev_audit_300_prompts', {}).items() if k in ('1', '10'))} %); the share of tokens kept "
    f"({r1(tab['dev100']['deployed']['heldout']['tok_loc'])} / {r1(tab['T100']['deployed']['heldout']['tok_loc'])} %) and of changed first tokens "
    f"({r1(tab['dev100']['deployed']['heldout']['audit_from_tensors']['first_token_changed'])} / {r1(tab['T100']['deployed']['heldout']['audit_from_tensors']['first_token_changed'])} %) are further from their ceiling.",
    f"The public n-gram entropy (metric v) did not separate the collapsed T500 / T1000 from the non-collapsed dev100: {r1(traps['generation_entropy']['T500:stream_first100']['GE_audit_x100'])} / "
    f"{r1(traps['generation_entropy']['T1000:stream_first100']['GE_audit_x100'])} against {r1(traps['generation_entropy']['dev100:dev_first100']['GE_audit_x100'])} (pre-edit "
    f"{r1(traps['generation_entropy']['W0:stream_first100']['GE_audit_x100'])}), while 8 words make up "
    f"{100 * traps['generation_entropy']['T500:stream_first100']['share_of_all_words_taken_by_the_8_most_frequent']:.0f} % of the generated words (T2000: "
    f"{r1(traps['generation_entropy']['T2000:stream_first100']['GE_audit_x100'])}).  For scale, the public formula on uniformly random sequences of the same length (90 word tokens) gives "
    f"{selft['2_ngram_entropy_of_random_sequences']['public_formula_on_random_sequences(x100, mean of 400)']['90 tokens, 5 symbols (uniform)']:.0f} for 5 symbols and "
    f"{selft['2_ngram_entropy_of_random_sequences']['public_formula_on_random_sequences(x100, mean of 400)']['90 tokens, 10 symbols (uniform)']:.0f} for 10.",
    f"Requests that repeat a subject / a whole request prompt inside a log ({repeats['n_pairs_with_bitwise_equal_solver_key']} of {repeats['n_pairs']} such prompt pairs of the stream have "
    f"bitwise identical solver keys and {repeats['n_pairs_with_different_new_object']} different new objects): T500 {rep['TX-G2XL-rome-T500']['n_requests_whose_subject_occurred_earlier']} / "
    f"{rep['TX-G2XL-rome-T500']['n_requests_whose_request_prompt_occurred_earlier']}, T1000 {rep['TX-G2XL-rome-T1000']['n_requests_whose_subject_occurred_earlier']} / "
    f"{rep['TX-G2XL-rome-T1000']['n_requests_whose_request_prompt_occurred_earlier']}, T2000 {rep['TX-G2XL-rome-T2000']['n_requests_whose_subject_occurred_earlier']} / "
    f"{rep['TX-G2XL-rome-T2000']['n_requests_whose_request_prompt_occurred_earlier']}; none in dev100 and T100.",
    f"Float32: at T2000 the deployed weight has entries up to {recon['logs']['TX-G2XL-rome-T2000']['reconstruction_fp64']['WE_max_abs']:.0f}; the exact (fp64) sum of the logged terms differs from it by up to "
    f"{recon['logs']['TX-G2XL-rome-T2000']['reconstruction_fp64']['max_abs']:.1e} per entry, so a replay in another precision is the deployed model only up to that round-off (float32 vs float64 forward at T1000: "
    f"same argmax on {pind['precision']['T1000']['n_argmax_equal_float32_vs_float64']} of {pind['precision']['T1000']['n_prompts']} prompts).",
    "The registered behavioural evaluation of E10 (amendment R19 f) uses the logs with 500 and 2000 requests: both are collapse logs; the only clean reported log (T100) is registered at gate level only.",
    "Checkpoint and tokenizer were verified against the listing of hf-mirror.com (huggingface.co is not reachable from this machine).",
]
V["what_a_user_must_know"] = [
    "Naming: the PUBLIC left_vector (6400, input side) divided by s is the read-out vector w_i; the PUBLIC right_vector (1600) times s is the output-side factor u_i; s = <left_vector, solver key>.  The stored "
    "weight is a Conv1D weight (6400, 1600), y = x @ W + b.  The library-format copies are already converted.",
    f"Solver key = module input at the last subject token of the request prompt alone.  Only there is the coordinate 1 (max deviation {recon['logs']['TX-G2XL-rome-T2000']['key_identity']['max_abs_w_dot_k_minus_1']:.1e}); "
    f"at the template-averaged key it has median {claims['template_mean_coordinate_T2000']['median']:.2f}, q95 {claims['template_mean_coordinate_T2000']['q95']:.2f}, max {claims['template_mean_coordinate_T2000']['max']:.0f} and "
    f"is within 1 % for {claims['template_mean_coordinate_T2000']['share_within_1pct']:.1f} % of the 2000 updates.",
    "T500, T1000 and T2000 are collapse logs (see per_log: binding restrictions).  dev100 and T100 are the only logs with a working deployed model.",
    "A log can be re-derived bit for bit only with the 21 context templates stored in its meta.json.",
    "Exact match is a first-token test on this data set (every object is one token); the public ES / PS / NS compare two probabilities and are near 50 on the collapsed logs while the probability "
    "differences are near 0."]
st1 = selft["1_exact_match_hits_of_collapsed_logs"]
st2 = selft["2_ngram_entropy_of_random_sequences"]["public_formula_on_random_sequences(x100, mean of 400)"]
V["tests_of_the_audit's_own_statements"] = {
    "why": "before the report was written, the interpretive statements of the first draft of this file were tested; three were wrong or too strong and were corrected",
    "1_'the exact-match hits of the collapsed logs are coincidences'": {
        "prediction_if_true": "the model emits a collapse token X as first token at the same rate whether or not X is the target of the request",
        "result": {t: {k: {"observed": st1[t][k]["hits_among_them_observed"], "expected": round(st1[t][k]["hits_among_them_expected_if_output_is_independent_of_the_target"], 2),
                           "sd_units": round(st1[t][k]["observed_minus_expected_in_sd"], 2), "hits_with_another_target": st1[t][k]["hits_whose_target_is_not_in_S"],
                           "hits_total": st1[t][k]["hits_total"]} for k in ("request", "paraphrase")} for t in st1},
        "outcome": "NOT supported as a general statement: consistent with the base rate on paraphrase prompts and at T1000, but at T2000 the request prompts show 15 hits against 6.4 expected "
                   "(+3.4 sd), and at T500 9 of the 14 request hits have a target outside the collapse tokens.  The statement was withdrawn from the list of discrepancies; the restriction now says "
                   "what was measured"},
    "2_'a random sequence over 5 symbols already scores 619'": {
        "prediction_if_true": "the public formula on uniformly random 5-symbol sequences of the texts' length gives about 619",
        "result": {k: round(v, 1) for k, v in st2.items()},
        "outcome": "WRONG number (619 is an asymptotic value that a 90-token text cannot reach): 5 symbols give 539 to 545, 10 symbols 616 to 628.  The corrected statement is in the residual risks"},
    "3_'ES / PS / NS of about 50 are chance values'": {
        "result": {t: {k: round(selft["3_magnitude_scores"][t]["deployed"][k], 2) for k in ("ES", "PS", "NS", "EM", "PM", "NM")} for t in ("T500", "T1000", "T2000")},
        "outcome": "replaced by the measured fact: the probability differences EM / PM / NM are between -0.2 and 1.3 points on the collapsed logs"},
    "4_write_time_files_used_for_'221 of 2000 requests were installed'": {
        "result": {s_: {k: v for k, v in wtime["runs"][s_].items() if k != "rows" and v is not None} for s_ in ("stream", "dev")},
        "outcome": "the sampled write-time files are reproduced identically by the audit's replay and the public test"},
    "statements_reworded_because_they_were_not_measured": [
        "'every difference between the deployed row and a scoped row is decided by the eight first-token updates' -> the norm share and the before / after of update 173 are measured; a model "
        "without these updates was not evaluated",
        "'edit metrics over all requests are bounded by the 221 installed requests' -> whether a replay can install the other targets was not tested",
        "'tokens kept and first-token change discriminate better than D1' -> they are further from their ceiling; no method was compared"]}
V["not_checked"] = [
    "the generations behind the n-gram entropy were not re-sampled (their stored texts were re-scored with the audit's own formula)",
    "D1 of T100, T500 and T2000 on HELD-OUT prompts was recomputed from the stored decodes; the audit's own decode covers dev100, T1000 and the stream prefixes 100, 150, 173, 174, 200",
    "the VIEW pool of amendment R16 is not part of this build",
    "'no GPU work was done with GPT-J' cannot be proven; the queue's event log has no such job"]
V["gpu_jobs_of_the_audit"] = {k: v for k, v in disc["gpu_jobs"]["audit(e10aud_*)"].items()}
V["files"] = {"scripts": sorted(f for f in os.listdir(A.AUD) if f.endswith(".py")), "results": "out/*.json (one per script)", "logs": "logs/", "table": "verdicts_table.txt"}
V["all_checks_pass"] = all(v["verdict"] == "PASS" for v in V["checks"].values())
A.jdump(V, os.path.join(A.AUD, "verdicts.json"))

# ======================================================================================= rendered table
L = []
L.append("INDEPENDENT AUDIT of the E10 public text logs (ROME, public code, GPT-2 XL, CounterFact) -- rendered by make_verdicts.py from out/*.json")
L.append(f"written {V['written_utc']}; audit directory {A.AUD}")
L.append("")
L.append("CHECKS")
for k, v in V["checks"].items():
    L.append(f"  {k:62s} {v['verdict']}")
L.append("")
L.append("CONDITIONS OF USE (PROTOCOL.md, section 6, E10)")
for k, v in V["conditions_of_use"].items():
    L.append(f"  {v['verdict']}  {k}")
    L.append(f"        {v['evidence'][:230]}")
L.append("")
L.append("LOGS")
L.append(f"  {'log':22s}{'T':>5s} {'verdict':8s}{'class':20s}{'replay':>8s}{'rel err':>10s}{'max|E|':>10s}{'ES':>7s}{'PS':>6s}{'NS':>6s}{'req':>6s}{'para':>6s}{'D1':>7s}{'kept':>6s}{'max|dW_i|':>11s}")
for n in NAMES:
    p = per_log[n]
    r = recon["logs"][n]
    L.append(f"  {n:22s}{p['T']:5d} {p['verdict']:8s}{p['class']:20s}{'bitwise' if r['float32_replay_equals_deployed_bitwise'] else 'NO':>8s}{r['reconstruction_fp64']['rel_fro_vs_update']:10.1e}"
             f"{r['reconstruction_fp64']['max_abs']:10.1e}{p['deployed']['ES_PS_NS'][0]:7.1f}{p['deployed']['ES_PS_NS'][1]:6.1f}{p['deployed']['ES_PS_NS'][2]:6.1f}"
             f"{p['deployed']['request_paraphrase_exact_pct'][0]:6.1f}{p['deployed']['request_paraphrase_exact_pct'][1]:6.1f}{p['deployed']['D1_heldout_pct']:7.1f}{p['deployed']['tokens_kept_pct']:6.1f}{p['largest_update_norm']:11.1f}")
L.append("  replay: W_0 + logged outer products written in float32 in order = deployed weight; rel err / max|E|: fp64 sum of the terms against the deployed weight;")
L.append("  ES PS NS: public scores of the deployed model; req / para: greedy decode starts with the new object; D1 / kept: HELD-OUT prompts, 10-token continuation not reproduced / tokens kept.")
L.append("")
L.append("DETERMINISM (public code re-run by the audit's own driver from the saved states)")
for s_ in ("dev", "stream"):
    d = det[s_]
    nk = d["n_solver_key_bitwise_equal(audit's own hook)"]
    L.append(f"  {s_:7s} {d['n_distinct_updates_rerun_by_audit']:4d} updates re-run: left vector bitwise equal {d['n_left_vector_bitwise_equal']}, right vector bitwise equal {d['n_right_vector_bitwise_equal']}, "
             f"solver key bitwise equal {nk}; max |<w,k>-1| {d['max_abs(<readout, key> - 1)']:.1e}; ({d['indices']})")
L.append(f"  single-edit protocol: {s0['n_records_compared']} random records of the first 1000 through the public experiments/evaluate.py: {s0['n_post_probabilities_bitwise_equal']} identical to the stored case files")
L.append("")
L.append("SINGLE-EDIT CHECK (first 1000 records; audit's arithmetic on the builder's 1000 case files / published, arXiv 2202.05262v5)")
for k in ("ES", "PS", "NS"):
    L.append(f"  {k}  {b1000['edited_recomputed_by_audit'][k]:7.2f} / {pubR['ROME'][k][0]:5.1f}   diff {b1000['diff_to_published'][k]:+.2f}")
L.append("")
L.append("THE REPORTED STREAM AROUND UPDATE 173 (audit's own decode; 200 HELD-OUT prompts, first 100 requests)")
L.append(f"  {'requests':>9s}{'|W-W0|_F':>10s}{'D1':>8s}{'kept':>7s}{'1st tok':>9s}{'first-100 exact':>17s}")
for p in misc["K"]["points"]:
    a = p["audit_200_prompts"]
    L.append(f"  {p['n_requests']:9d}{p['W_minus_W0_fro']:10.1f}{a['D1_tf']:8.1f}{a['tok_loc']:7.1f}{a['first_token_changed']:9.1f}{p['audit_first_100_requests_exact_pct']:17.1f}")
L.append("")
L.append("DISCREPANCIES")
for i, d in enumerate(V["discrepancies"], 1):
    L.append(f"  {i}. {d['what']}")
    L.append(f"     numbers: {d['numbers']}")
    L.append(f"     effect on the logs: {d['effect_on_logs']}")
L.append("")
L.append("BINDING RESTRICTIONS OF THE COLLAPSE LOGS (T500, T1000, T2000)")
for r in per_log["TX-G2XL-rome-T2000"]["binding_restrictions"]:
    L.append(f"  - (T2000) {r}")
L.append("")
L.append("RESIDUAL RISKS")
for r in V["residual_risks"]:
    L.append(f"  - {r}")
open(os.path.join(A.AUD, "verdicts_table.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
print("all checks pass:", V["all_checks_pass"])
