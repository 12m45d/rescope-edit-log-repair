"""Render tests/out/table.txt and tests/out/summary.json from tests/out/cpu_*.json and gpu_*.json (python tests/render.py)."""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402
import fx  # noqa: E402

OUT = os.path.join(HERE, "out")
ORDER = ["c_pack", "c_guard", "c_log", "c_rebase", "c_draws", "c_calib", "c_null0", "c_gate", "c_cache", "c_rows",
         "g_linear", "g_silent", "g_target", "g_null0", "g_batch", "g_hook", "g_cache", "g_public", "g_store", "g_guard", "g_usage"]
IDENT = {"g_linear": "(a)", "g_silent": "(b)", "g_target": "(c)", "c_rebase": "(c)", "g_null0": "(d)", "c_null0": "(d)"}


def ev(name, d):
    """One line of evidence per test, from its stored detail."""
    g = lambda *k: _get(d, k)                                            # noqa: E731
    if name == "c_pack":
        return (f"round trip of sets with 0 .. 700 edits: {g('pack_roundtrip')}; synthetic padded batch + cache step: fire positions {g('n_fire_pos')} (real positions 5+1, 3+1, 4+1), "
                f"80 candidate edits per row as a mask: {all(g('cand_edits_is_mask'))}, own flags {g('own_cand')[:2]}")
    if name == "c_guard":
        return "; ".join(f"{k}: capped {v['n_capped']}, lowered {v['n_lowered']}, own requests in band {v['own_requests_in_band_before']} -> {v['own_requests_in_band_after']}"
                         for k, v in d.items()) + f"; largest relative lowering {max(v.get('max_relative_lowering', 0.0) for v in d.values()):.1e}"
    if name == "g_guard":
        return (f"{g('n_cases')} capped (gate, draw, edit) cases: own edit a candidate on its own request in {g('with_margin', 'tf_own_candidate')} teacher-forced and "
                f"{g('with_margin', 'dec_own_candidate')} decode passes with the margin; with the library's caps alone: {g('library_thresholds_only', 'tf_own_candidate')} and "
                f"{g('library_thresholds_only', 'dec_own_candidate')}")
    if name == "c_log":
        return f"|delta - (W_E - W_0)^T| / |.| = {g('delta_vs_WE_minus_W0_transposed_rel_fro'):.1e}; float32 replay bitwise = deployed: {g('replay_bitwise_equals_deployed')}; prefix replay too"
    if name == "c_rebase":
        return (f"k W_0 + b + V = target: max rel. error {g('max_rel_err_rebased'):.1e} over 100 edits (logged values instead: median {g('median_rel_err_not_rebased'):.2f}); "
                f"values vs rescope.rebase {g('values_vs_rescope_rebase_max_rel'):.0e}, bitwise = the driver's released_values: {g('values_bitwise_equal_to_the_drivers_function')}")
    if name == "c_draws":
        return f"{g('draw_equals_rescope_splits_sample')}/{g('n_draws')} draws = splits.sample; {g('null_sample_disjoint_by_text')}/{g('n_draws')} null samples disjoint by text"
    if name == "c_calib":
        return (f"thresholds from cached maxima vs library on key tensors: max |diff| {g('ncoord', 'two_block', 'max_abs_threshold_diff'):.0e} (ncoord), "
                f"{g('cos', 'two_block', 'max_abs_threshold_diff'):.0e} (cos); block-2 queries firing {g('ncoord', 'two_block', 'n_block2_queries_firing')} <= {g('ncoord', 'two_block', 'n_allowed')}")
    if name == "c_null0":
        return f"13 gate tensors bitwise equal: {all(g('fields_bitwise_equal').values())}; decisions equal on {g('n_rows_compared')} rows; control n_A = 100 differs: {g('control_n_a_100_differs')}"
    if name == "c_gate":
        return "; ".join(f"{k}: {v['n_fired']} fired of {v['n_queries']}, mismatches {v['fired_mismatch'] + v['released_pattern_mismatch'] + v['candidate_edit_count_mismatch']}" for k, v in d.items())
    if name == "c_cache":
        return (f"{g('texts_needed_by_dev_log')} texts, {g('texts_missing')} missing; single-pass recomputation deviates by "
                f"{max(p['single_pass_recompute_max_rel_deviation'] for p in d['parts'].values()):.1e}; float16 would move a coordinate by {g('float16_rounding_max_dr_over_scale'):.1e} of its scale (band {g('band'):.0e})")
    if name == "c_rows":
        return f"rows {', '.join(g('labels'))}; undefined rows refused with reason: {g('undefined_row_refused_with_reason')}"
    if name == "g_linear":
        return (f"max |logit diff| to the deployed weight {g('max_abs_logit_diff_to_deployed'):.1e} (deployed vs pre-edit: {g('for_scale_median_max_abs_logit_diff_deployed_vs_pre_edit'):.1f}); "
                f"same argmax at {g('n_positions_same_argmax')}/{g('n_positions')} positions; {g('n_decodes_equal_to_deployed')}/{g('n_prompts')} decodes equal; "
                f"{g('n_pad_positions_in_batches')} PAD positions, none fires")
    if name == "g_silent":
        return (f"{g('n_silent_bitwise_tf')}/{g('n_silent_hook')} silent rows bitwise (teacher forced), {g('n_silent_bitwise_dec')} (decode); {g('n_fired')} firing rows, "
                f"{g('n_fired_bitwise_tf')} of them equal; thresholds +inf: {g('n_off_bitwise_tf')}/{g('n_rows')}")
    if name == "g_target":
        return (f"served module output vs target: max rel. error {g('rebased', 'max_rel_err'):.1e}, own edit released at the solver position {g('rebased', 'own_edit_released_at_solver_position')}/100; "
                f"logged values: median {g('logged_values', 'median_rel_err'):.2f}; deployed weight: median {g('deployed', 'median_rel_err'):.2f}")
    if name == "g_null0":
        return f"{g('n_bitwise_equal_tf_logits_and_flags')}/{g('n_prompts')} prompts bitwise (teacher forced), {g('n_bitwise_equal_decode_logits')} (decode); {g('n_fired_under_default')} of them fire"
    if name == "g_batch":
        return "; ".join(f"{k}: tok diff rows {v['tf_tok_diff_rows']}+{v['dec_tok_diff_rows']}, flag diff {v['flag_diff_rows']}, max |logit diff| {max(v['max_abs_logit_diff_tf'], v['max_abs_logit_diff_dec']):.0e}"
                         for k, v in d.items())
    if name == "g_hook":
        return "; ".join(f"{k}: {v['n_rows']} rows, hook != cache {sum(v['hook_tf_vs_cache_mismatch'].values())}, sparse != full {v['sparse_vs_full_rows_differing']}, "
                         f"passes {v['sequences_run_sparse']} vs {v['sequences_run_full']}" for k, v in d.items())
    if name == "g_cache":
        return (f"live vs cached key, coordinate shift / scale: {g('tf_max_dr_over_scale'):.1e} (teacher forced), {g('dec_max_dr_over_scale'):.1e} (decode); band {g('band'):.0e}; "
                f"float16: {g('f16_max_dr_over_scale'):.1e}; cosine feature live vs cached {g('cos_max_abs_dr_live_vs_cached'):.1e} (its band >= {g('band'):.0e}); "
                f"features cuda vs cpu {g('features_cuda_vs_cpu_max_abs_diff'):.0e}")
    if name == "g_public":
        return (f"deployed / pre-edit vs the build's case files: max |diff| {g('deployed', 'max_abs_diff_to_build_case_files'):.0e} / {g('preedit', 'max_abs_diff_to_build_case_files'):.0e}; "
                f"all-firing linear gate vs deployed scores: {g('where_all_vs_deployed_max_abs_score_diff'):.2f} points; generated texts equal to the build's: "
                + ", ".join(f"{k} {v['n_texts_equal_to_build']}/{v['n_texts']}" for k, v in d["generation"].items()))
    if name == "g_store":
        return "; ".join(f"{k}: first pass {v['first_pass_run']} run + {v['first_pass_silent']} silent, second pass {v['second_pass_run']} run, rows differing {v['rows_differing']}"
                         for k, v in d.items() if isinstance(v, dict) and "first_pass_run" in v) + (
            f"; two weights / two gates with the same label get separate outcomes: {g('same_label', 'weights_sig_differ') and g('same_label', 'gates_sig_differ')}")
    if name == "g_usage":
        return "; ".join(f"({k}) {v['n_lines']} lines, {v['seconds']:.0f}s" for k, v in d.items() if isinstance(v, dict)) + f"; hook mismatches {g('hook_mismatch')}"
    return ""


def _get(d, keys):
    for k in keys:
        d = d[k]
    return d


res = {}
for p in sorted(glob.glob(os.path.join(OUT, "cpu_*.json")) + glob.glob(os.path.join(OUT, "gpu_*.json"))):
    r = json.load(open(p))
    res[r["name"]] = r
now = e10t.code_fingerprint()
rows, summ = [], {"code": now, "tests": {}}
for name in ORDER + [n for n in res if n not in ORDER]:
    r = res.get(name)
    if r is None:
        rows.append([name, IDENT.get(name, ""), "NOT RUN", "-", "-", ""])
        summ["tests"][name] = {"status": "NOT RUN"}
        continue
    cur = r["code"]["e10t"] == now["e10t"] and r["code"]["rescope"] == now["rescope"]
    try:
        e = ev(name, r["detail"]) if r["detail"] else (r["error"] or "").strip().splitlines()[-1]
    except Exception as ex:  # noqa: BLE001
        e = f"(evidence not rendered: {type(ex).__name__}: {ex})"
    st = ("PASS" if r["pass"] else "FAIL") + ("" if cur else " (older code)")
    rows.append([name, IDENT.get(name, ""), st, f"{r['seconds']:.0f}", f"{r['peak_alloc_MiB']:.0f}" if "peak_alloc_MiB" in r else "-", r["what"]])
    rows.append(["", "", "", "", "", "    " + e])
    summ["tests"][name] = {"status": st, "pass": r["pass"], "current_code": cur, "seconds": r["seconds"], "what": r["what"], "evidence": e, "written": r["written"]}
n_pass = sum(1 for v in summ["tests"].values() if v.get("pass") and v.get("current_code"))
summ.update(n_tests=len(ORDER), n_pass_current_code=n_pass, n_fail=sum(1 for v in summ["tests"].values() if v.get("pass") is False),
            n_not_run=sum(1 for v in summ["tests"].values() if v["status"] == "NOT RUN"))
txt = [f"e10t TESTS (development log {e10t.DEV_LOG} only)   rendered {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} by tests/render.py",
       f"code: e10t {now['e10t']}, rescope {now['rescope']}, fx {now['fx']}, tx_public {now['tx_public']}   ({now['e10t_version']})",
       f"{n_pass} of {len(ORDER)} tests pass with the current code; failed: {summ['n_fail']}; not run: {summ['n_not_run']}.  c_ = CPU (tests/t_cpu.py), g_ = GPU (tests/t_gpu.py).",
       "id = the identity of the task list that the test establishes: (a) linear log = deployed weight, (b) silent prompt = pre-edit model bit for bit,",
       "(c) target attained at the solver key with rebased values, (d) null repair with n_A = 0 = the default method bit for bit.", "",
       fx.io.text_table(["test", "id", "status", "s", "MiB", "what it checks / evidence"], rows, align=["l", "l", "l", "r", "r", "l"])]
txt = "\n".join(txt) + "\n"
with open(os.path.join(OUT, "table.txt"), "w") as f:
    f.write(txt)
e10t.data.jdump(fx.io.clean(summ), os.path.join(OUT, "summary.json"))
print(txt)
