"""Pass / fail table and regression table of the rescope tests, rendered ONLY from tests/out/*.json."""
from __future__ import annotations

import json
import os

LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Qwen", "TB-L100-ae-l2_3": "LLaVA"}
REG = ("P10-HARD-SELF-noE", "R1-REL0.35-REPL", "R1-REL0.35-GAIN", "R1-POOL1-REPL", "NDB-R1-0.95")


def _g(x, nd=3):
    return "-" if x is None else f"{x:.{nd}g}"


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _frac(v):
    return f"{v[0]}/{v[1]}"


def k_a1(d):
    b = d["bf16_writes"]
    return (f"exact writes: rebased max rel err {_g(d['max_rel_err_exact_writes'])}, not rebased {_g(max(c['not_rebased_max_rel'] for c in d['cases']))}; "
            f"bf16 writes: factor form {_g(b['factor_form_max_rel'])}, exact form {_g(b['exact_form_max_rel'])}, not rebased {_g(b['not_rebased_max_rel'])}")


def k_a2(d):
    n = sum(len(c["responses"]) for c in d["cases"])
    ok = sum(v["bitwise_expected"] for c in d["cases"] for v in c["responses"].values())
    return f"{ok}/{n} (seed, dtype, response) cells bitwise dtype(W_0 k + v); fp64 max |out - z*| {_g(d['max_abs_err_fp64'])}"


def k_a3(d):
    return (f"non-firing rows bitwise {d['n_nonfiring_rows_bitwise']}/{d['n_nonfiring_rows']}; silent calls return the module's own tensor "
            f"{d['n_nonfiring_calls_same_object']}/{d['n_nonfiring_calls']}; -0.0 kept {d['n_negzero_preserved']}/{d['n_negzero_rows']}")


def k_a9(d):
    b = d["batched"]
    return (f"decisions equal {d['n_decisions_equal']}/{d['n_cases']} (role x fire x response x eta x extra detectors x dtype); "
            f"firing rows bitwise {d['n_firing_rows_bitwise']}/{d['n_firing_rows']}; padded batch B = 3 == three B = 1 calls {b['decisions_equal']}/{b['n']}")


def k_a4(d):
    n = d["n_trials"]
    return (f"order independent (bitwise) {d['order_independent_bitwise']}/{n}; = coordinatewise max {d['equals_coordinatewise_max']}/{n}; "
            f"minimal {d['minimal']}/{n}; idempotent {d['idempotent']}/{n}; default margins order independent {d['margin_order_independent']}/{n}; "
            f"own-key conflict refused: {d['own_key_conflict']['refused']}")


def k_a5(d):
    c = d["cells"]
    z = d["z_vs_theory_no_ties"]
    return (f"{sum(x['ok'] for x in c)}/{len(c)} (seed, n, alpha, ties) cells, {c[0]['n_trials']} trials x {c[0]['n_test_per_trial']} fresh keys each: mean "
            f"false-firing rate <= alpha + 3 se in every cell (<= alpha itself in {sum(x['le_alpha'] for x in c)}/{len(c)}; largest excess "
            f"{d['max_excess_over_alpha']:+.4f}); z against the exact expectation 1 - rank/(n+1), {z['n_cells']} cells without ties: mean {z['mean']:+.2f}, "
            f"range [{z['min']:+.2f}, {z['max']:+.2f}]; firing == strict conformal event in every trial; the non-strict rule AT the quantile would exceed "
            f"alpha by {d['naive_rule_max_excess_with_ties']:+.4f} under ties; own keys silenced by the literal rule: reported "
            f"{d['own_keys']['literal_silenced_reported']}, after capping {d['own_keys']['capped_silenced_actual']}")


def k_a6(d):
    e, z = d["exact"], d["zero_force_log"]
    return (f"max |r| on benign keys {_g(e['max_abs_benign_r'])}, own coordinate rel err {_g(e['max_own_rel_err'])}, vs pinv {_g(e['max_abs_diff_vs_pinv'])}; "
            f"ridge vs KKT {_g(max(r['max_abs_diff_vs_kkt'] for r in d['ridge']))}; joint zero-forcing vs pinv {_g(z['max_abs_diff_vs_pinv'])}")


def k_a7(d):
    p = d["public_operator"]
    return (f"min cut == brute force {d['n_equal_brute_force']}/{d['n_instances']} ({d['n_problems_with_ties']}/{d['n_problems']} problems with ties); "
            f"box thresholds reproduce the optimum {d['n_box_thresholds_reproduce_optimum']}/{d['n_instances']}; scope_cut on scoped logs {p['objective_equals_brute_force']}/{p['n']}")


def k_a8(d):
    k = d["kv_cache"]
    return (f"decisions equal {d['n_decisions_equal']}/{d['n_cases']}, outputs bitwise {d['n_outputs_bitwise']}/{d['n_cases']} "
            f"(cosine / exact / ball / router / zscore); KV-cache steps: positions {k['positions_equal']}/{k['n']}, outputs {k['outputs_bitwise']}/{k['n']}; "
            f"zscore in one-token steps: kv='stale' swaps {k['n_step_positions_zscore_stale']} step positions, kv='call' (official hook) {k['n_step_positions_zscore_call']}")


def k_a15(d):
    p, w, t = d["pad"], d["W0"], d["tie"]
    return (f"last prompt index of another length than the batch refused {d['n_lp_length_raise']}/{d['n_lp_length_cases']} (scoped x 3 roles, router, zscore, cosine; "
            f"shorter and longer); gate without W0 refused: {w['no_W0_raises']}, wrong module weight with set_weight=False refused: "
            f"{w['wrong_weight_set_weight_false_raises']}, weight restored: {w['restored_after']}; padded batch with seq_len == unpadded B = 1 calls "
            f"{p['masked_equals_unpadded_B1']}/{p['n']} (PAD positions firing without seq_len: last {p['n_pad_firing_unmasked']['last']}, last+answer "
            f"{p['n_pad_firing_unmasked']['last+answer']}, all {p['n_pad_firing_unmasked']['all']}; with seq_len {p['n_pad_firing_masked']}); exact top-1 tie: "
            f"the lower edit index releases {t['lowest_index_releases']}/{t['n']}")


def k_a16(d):
    s = d["setters"]
    t = "; ".join(f"{k} {v['tau_5_without_the_guard']:.3f} -> {v['tau_5']:.3f} (edit fires on {v['firing_rate_without_the_guard_pct']:.1f} % -> "
                  f"{v['firing_rate_edit_5_pct']:.1f} % of random keys, counted: {v['n_guards_not_firing_before']})" for k, v in s.items())
    return (f"add_guards refuses a key that does not fire: {d['nonfiring_refused']} (mixed batch: {d['mixed_refused']}); firing keys accepted "
            f"{d['firing_keys']['n_firing_now']}/{d['firing_keys']['n_added']}; with allow_nonfiring=True the guard is a must-fire constraint, threshold of its edit: "
            f"{t}; least_repair sees no conflict in it: {not d['least_repair']['refused'] and d['least_repair']['conflicts'] == []}")


def k_a10(d):
    return (f"threshold mode: tau = witness coordinate {d['threshold']['tau_equals_coordinate']}; conflict reported {d['conflict']['conflict']}; "
            f"detector mode (cosine): fires witness {d['detector']['fires_witness']}, random key {d['detector']['fires_random']}; witness guarded {d['witness_is_guarded']['refused']}; "
            f"a repair that would leave the witness silent is refused: {d['nonfiring_witness_refused']['cosine_gamma_above_1']} / "
            f"{d['nonfiring_witness_refused']['supplied_detector_tau_above_1']}; every guard of every granted repair fires: "
            f"{d['nonfiring_witness_refused']['every_guard_of_every_granted_repair_fires']}")


def k_a11(d):
    return f"{sum(c['ok'] for c in d['cases'])}/{len(d['cases'])} (seed, q) cells: tau = numpy quantile of the pooled cross-talk; own-key cap {d['cap']['capped_own_keys_fire']}"


def k_a12(d):
    out = []
    for lg, r in d["logs"].items():
        rb, sp = r["rebase"], r["split"]
        out.append(f"{SHORT[lg]}: rebase rel err vs cached z* - h(W_0): max {_g(rb['factor_form_max'])} (median {_g(rb['factor_form_median'])}), "
                   f"not rebased max {_g(rb['not_rebased_max'])} (median {_g(rb['not_rebased_median'])}), = P11a; HELD-OUT {sp['n_heldout']} rows, "
                   f"disjoint {sp['n_heldout_disjoint']}, twins {sp['n_twins']}; of the disjoint rows {sum(sp['n_disjoint_question_in_repair'].values())} share "
                   f"their question with a REPAIR row (strict pool {sp['n_heldout_strict']}) and {sp['n_disjoint_rows_in_duplicate_groups']} repeat another "
                   f"disjoint row ({sp['n_disjoint_distinct_queries']} distinct queries); hard negatives {r['hard_negatives']['n']} = P11a "
                   f"({r['hard_negatives']['n_p11a']}), missing file raises: {r['hard_negatives']['missing_file_raises']}; "
                   f"POOL(1) theta {_g(r['log_only']['1.0']['theta'], 4)} = P11a")
    return "; ".join(out)


def k_b1(d):
    return (f"max |logit diff| {_g(d['max_abs_logit_diff'])} (median {_g(d['median_abs_logit_diff'])}; round-off floor {_g(d['roundoff_floor_max_abs_logit_diff'])}; "
            f"W_0 vs W_E median {_g(d['median_max_abs_logit_diff_W0_vs_WE'])}); argmax equal on {d['n_argmax_equal']}/{d['n']} sequences, "
            f"{d['n_argmax_tok_equal']}/{d['n_tok']} tokens (floor {d['roundoff_floor_n_argmax_tok_equal']}/{d['n_tok']}); module output rel err max "
            f"{_g(d['module_rel_err_max'])} (floor {_g(d['roundoff_floor_module_rel_err_max'])}); own term dropped: {_g(d['control_ratio_median'])}x further at the "
            f"last prompt position; bitwise {d['n_bitwise']}/{d['n']} (not expected)")


def k_b2(d):
    return f"bitwise W_0 {d['n_bitwise_W0']}/{d['n']} (hook called {d['n_hook_called']}/{d['n']}, fired {d['n_fired']}; differs from W_E on {d['n_differs_from_WE']})"


def k_b345(d):
    t3, t4 = d["iii_total"], d["iv_total"]
    pr = "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in t3["probe_rows_by_config"].items())
    return (f"(iii) non-firing rows bitwise W_0 {t3['bitwise_W0']}/{t3['n_nonfiring']} (probe rows: {pr}); (iv) predicted == observed firing "
            f"{t4['equal']}/{t4['n']} (config, row) pairs; (v) all flags equal to P10 / P11a: {d['v_all_flags_equal']}")


def k_b6(d):
    out = []
    for tag, t in d["methods"].items():
        for mk, m in t.items():
            if isinstance(m, dict):
                out.append(f"{tag}[{mk}] edits {_frac(m['edit_flags_equal'])}, HELD-OUT {_frac(m['heldout_flags_equal'])}, complaints {_frac(m['complaint_flags_equal'])}")
    re = d.get("reedit")
    if re:
        out.append(f"REEDIT built by evaluate.reedit (z* at B = 1, chunk={re['chunk']}): {re['n_oracle_targets']}/{re['n']} oracle targets, "
                   f"{re['n_installed']}/{re['n']} complaints installed")
    return "; ".join(out)


def k_b7(d):
    a, b, c, r = d["inactive_gate"], d["deployed"], d["d2"], d["free_generation_rel"]["R1"]
    return (f"inactive gate: y0 {a['y_equals_stored_y0']}/{a['n']}, 48-token answer {a['long_equals_stored_pre']}/{a['n']}; W_E 48-token answer = stored "
            f"{b['long_equals_stored_post']}/{b['n']}; D2 flags = stored {c['D2_flags_equal_stored']}/{c['n_pre_correct']}; free-gen Rel eq {r['eq']}/{r['n']} "
            f"(teacher-forced exact {r['tf_exact']}), one firing position per decode {sum(x == 1 for x in r['n_fire_pos'])}/{r['n']}, decision at the last prompt "
            f"position not the same in every forward of a decode: {r['n_decodes_last_decision_not_constant']}/{r['n']}, position bookkeeping "
            f"{r['pos_ok']}/{r['n']} (the two paths are compared in test b9)")


def k_b9(d):
    s, p, lr = d["scan"], d["positions"], d["least_repair"]
    bk = ", ".join(f"{k} {v['n_shifted']}/{v['n']}" for k, v in s["by_kind"].items())
    sb, se, mn, dk = p["shifted_benign_rows"], p["shifted_edit_rows"], lr["margin_needed"], lr["decode_keys_bitwise"]
    names = {"tf_keys_default_margin": "teacher-forced keys, default margin", "both_paths_default_margin": "keys of both paths, default margin",
             "tf_keys_decode_margin": f"teacher-forced keys, margin {d['decode_margin']}",
             "both_paths_decode_margin": f"protocol A = keys of both paths, margin {d['decode_margin']}",
             "all_default_margin": "protocol B = teacher-forced + all decode keys, default margin"}
    var = "; ".join(f"{nm}: teacher-forced {lr[k]['tf_still_firing']}, at the decode's token {lr[k]['tf_prompt_position_still_firing']}, decode "
                    f"{lr[k]['decode_still_firing']}" + (f" (refused {lr[k]['n_refused']})" if lr[k]["n_refused"] else "") for k, nm in names.items())

    def mg(key):
        v = mn[key]
        if not v["n"]:
            return "no such complaint"
        ub = f", unbounded for {v['n_unbounded']}" if v["n_unbounded"] else ""
        return f"max {_g(v['max'])}{ub} (above the default margin for {v['n_above_default_margin']}/{v['n']}, above {d['decode_margin']} for {v['n_above_decode_margin']}/{v['n']})"
    return (f"(1) last prompt position on another token in teacher forcing than in the decode: {s['n_shifted']}/{s['n']} sequences ({bk}; edits "
            f"{s['shifted_edits']}; tokens {p['tokens_shifted']}); lp_shift == the vendor's row encoding on {_frac(s['encode_row_equal'])} sequences, == the built "
            f"batches {p['ok_tokenizer']}/{p['n_checked']}; max |dr| between the two "
            f"positions {_g(p['max_dr_shifted_tf_vs_decode_position'])} on shifted sequences, {_g(p['max_dr_unshifted_tf_vs_decode_position'])} on the others "
            f"(same token in the two sequences: {_g(p['max_dr_same_token_two_sequences'])}). (2) lp='prompt' puts the hook on the decode's token "
            f"{p['ok_hook']}/{p['n_checked']}, rows identical where nothing shifts {_frac(p['unshifted_rows_identical'])}; shifted benign rows (tau {d['tau']}): "
            f"fired {sb['fired_tf']} / {sb['fired_prompt']} of {sb['n']} at the teacher-forced / decode position (differs on {sb['fired_differs']}), D1 "
            f"{sb['D1_tf']} / {sb['D1_prompt']} (differs on {sb['D1_differs']}); shifted edit rows: fired {se['fired_tf']} / {se['fired_prompt']} of {se['n']}, exact "
            f"{se['exact_tf']} / {se['exact_prompt']}. (3) least repair of {lr['n_complaints']} firing REPAIR complaints ({lr['n_complaints_shifted']} shifted), "
            f"repaired complaints still firing afterwards: {var}. Relative margin a complaint needs above its keys to stay silent through its decode, unshifted "
            f"complaints: teacher-forced keys {mg('tf_keys/unshifted')}, both paths {mg('both_paths/unshifted')}, all decode keys {mg('all/unshifted')}; shifted "
            f"complaints: teacher-forced keys {mg('tf_keys/shifted')}, both paths {mg('both_paths/shifted')}, all decode keys {mg('all/shifted')}. (4) decode_keys "
            f"bitwise the keys of the real decodes in {dk['bitwise_equal_to_decode_keys']}/{dk['n_forwards']} full forwards (up to "
            f"{dk['max_distinct_keys_in_one_decode']} distinct keys at the last prompt position within one decode)")


def k_b10(d):
    r, c, rl, ag, dc = d["r_own_B1"], d["cos_B1_vs_stored"], d["rel"], d["add_guards"], d["decode"]
    T = d["n_edits"]
    d1, d2 = dc["cap_with_B1_guards"], dc["cap_with_B1_and_decode_guards"]
    return (f"B = 1 Rel key of the own prompt on its own detector: r in [{r['min']:.3f}, {r['max']:.3f}] (median {r['median']:.4f}; {r['n_below_1']}/{T} below 1), cosine to "
            f"the stored key min {c['min']:.5f} ({c['n_below_0.999']} below 0.999: edits {c['edits_below_0.999']}); own prompts firing through the hook: thresholds at the "
            f"caps of the stored keys {rl['cap_stored_keys']['own_fired_last']}/{T}, with the B = 1 keys as guards {rl['cap_with_B1_guards']['own_fired_last']}/{T}; conformal "
            f"with binding caps {rl['conformal_capped_stored_keys']['own_fired_last']}/{T} -> {rl['conformal_capped_with_B1_guards']['own_fired_last']}/{T}; add_guards on the "
            f"capped log refused: {ag['on_capped_log_refused']} ({ag['on_capped_log_allow_nonfiring']['n_not_firing_now']} keys do not fire); exact gate (0.999) fires "
            f"{rl['exact_gate']['own_fired_last']}/{T} own prompts, cosine 0.95 {rl['cosine_0.95']['own_fired_last']}/{T}; free generation of the own prompt "
            f"({d1['n_edits']} edits), own edit a candidate in every full forward: teacher-forced B = 1 guards {d1['n_edits_own_candidate_in_every_forward']}/{d1['n_edits']} "
            f"edits ({d1['n_forwards_own_candidate']}/{d1['n_forwards']} forwards), with the decode keys as guards too "
            f"{d2['n_edits_own_candidate_in_every_forward']}/{d2['n_edits']} ({d2['n_forwards_own_candidate']}/{d2['n_forwards']}; "
            f"{dc['n_shifted_edits_without_decode_guards']} shifted edit(s) have no decode guard); hn_items() without loaded hard negatives raises: "
            f"{d['evaluator']['hn_items_without_hard_negatives_raises']}; strict pool {d['evaluator']['n_heldout_strict']} of "
            f"{d['evaluator']['n_heldout_disjoint']} disjoint rows")


def k_c(d):
    m = d["modes"]
    g = "; ".join(f"{k} {m[k]['gate_only_ms']:.2f} ms = {100 * m[k]['gate_only_over_plain_forward']:.2f} %" for k in m if "gate_only_ms" in m[k])
    e = "; ".join(f"{k} {m[k]['paired_ratio_median']:.3f} [{m[k]['paired_ratio_q25_q75'][0]:.2f}, {m[k]['paired_ratio_q25_q75'][1]:.2f}]" for k in m if k != "plain_W0")
    sd = d["module_dtype_gate_same_decisions"]
    ld = d.get("gpu_load_during_test", {})
    return (f"plain forward {m['plain_W0']['ms_per_forward']:.1f} ms (mean {d['mean_seq_len']:.0f} tokens; edited module alone {d['module_linear_ms']:.2f} ms). "
            f"Gate only, per forward and as % of the plain forward: {g}. End to end, paired ratio to plain, median [q25, q75] (shared GPU: "
            f"{ld.get('utilization_pct')} % utilisation, {ld.get('n_compute_processes')} processes): {e}. Same decisions as the fp64 gate: module dtype, role last "
            f"{sd['last']}/{d['n_sequences']}, role all {sd['all']}/{d['n_sequences']}; fp32, role all {d['fp32_gate_same_decisions_all']}/{d['n_sequences']}")


def k_b8(d):
    g = d.get("guards", {})
    return (f"{d['n_code_lines']} code lines executed; {len(d['methods'])} methods x 5 item types; pre-edit D1 {d['pre_edit_D1']}, deployed D1 == stored "
            f"{d['deployed_D1_equals_stored']}/{d['n_probe_rows']}; lp='prompt' rows == lp='tf' rows where nothing shifts "
            f"{d.get('lp_prompt_rows_equal_where_no_shift')}/{d.get('n_rows_without_shift')}; guards firing after the session {g.get('n_firing')}/{g.get('n')}; "
            f"session ops {d['session']['ops']}")


def k_a13(d):
    return "; ".join(f"{SHORT[lg]}: " + ", ".join(f"{c} {v['fired_equal']}/{v['n']} (probe firing {v['probe_fire_rate']:.1f} %)" for c, v in r.items())
                     for lg, r in d["logs"].items())


def k_a14(d):
    out = []
    for lg, r in d["logs"].items():
        c, lr, nr, wr = r["conformal"], r["least_repair"], r["null_repair"], r["witness_repair"]
        c1 = next(x for x in c if x["n"] == 100 and x["alpha"] == 0.1)
        out.append(f"{SHORT[lg]}: conformal mean false-firing rate on HELD-OUT keys == P11b in {sum(x['ok'] for x in c)}/{len(c)} (n, alpha) cells "
                   f"(e.g. n = {c[0]['n']}, alpha = {c[0]['alpha']}: {100 * c[0]['mean_ffr']:.2f} % vs {100 * c[0]['p11b_mean_ffr']:.2f} %); the guarantee is marginal over "
                   f"the draw: n = {c1['n']}, alpha = {c1['alpha']}: mean {100 * c1['mean_ffr']:.2f} %, q95 of single draws {100 * c1['q95_ffr']:.2f} %, "
                   f"{100 * c1['share_of_draws_above_alpha']:.0f} % of the draws above alpha; least repair of "
                   f"{lr['n_firing_repair_rows']} firing REPAIR rows: {lr['n_repaired_rows_still_firing']} still fire, order independent {lr['order_independent_bitwise']}; "
                   f"null repair (d = 11008): max |r| on benign keys {_g(nr['max_abs_benign_r'])}; witness repair of 50 T-Gen keys: {wr['already_fired']} already fire, "
                   f"{wr['applied']} granted, {wr['conflicts']} conflicts with benign rows, {wr['n_silent_repair_rows_firing_after']} benign rows fire afterwards")
    return "; ".join(out)


def k_a17(d):
    e, n = d["exact_enumeration_vs_referee"], d["n_trials"]
    return (f"{n} random instances (continuous, quarter grid, small integers, few atoms; splits 1:1 / 1:3 / 3:1; u = 1 or a positive vector; rank by excess or "
            f"by r): blocks, baselines, rank, eta == naive {d['rule_equal']}/{n}; stored thresholds are THE boundary of the defining comparison "
            f"{d['tau_is_threshold']}/{n}; candidates and top-1 on {d['n_keys']} keys (incl. the keys at / next to every threshold) == naive in "
            f"{d['decisions_equal']}/{n} instances ({d['n_keys_firing']} firing keys, {d['n_keys_two_or_more_candidates']} with several candidates, "
            f"{d['n_keys_exact_tie_in_rank']} exact rank ties); top-1 by the literal excess (r - b - eta u) / u == top-1 by margin on "
            f"{d['n_top1_literal_excess_equal']}/{d['n_top1_literal_excess_checked']} multi-candidate keys (the others are exact margin ties: "
            f"{d['n_top1_literal_excess_differs_at_exact_margin_tie']}); calibration queries tied with eta that fire: {d['tied_with_eta_firing']}/{d['n_keys_tied_with_eta']}; "
            f"eta = +inf in {d['eta_infinite_trials']} instances; guard caps == naive {d['caps_equal']}/{d['n_cap_trials']} ({d['n_edits_capped_total']} edits capped); "
            f"shared-scale rule == naive {d['shared_rule_equal']}/{d['n_shared_trials']}; exact false-firing probability == the referee's exact rationals in "
            f"{e['equal']}/{e['cases']} enumerated cases, above the bound in {e['violations_of_bound']}; dense gates (one scalar score) "
            f"{d['dense']['equal']}/{d['dense']['n']}")


def k_a18(d):
    c = d["cells"]
    z = d["z_vs_bound_continuous"]
    ties = [x for x in c if not x["continuous"]]
    return (f"{sum(x['ok'] for x in c)}/{len(c)} cells ({d['n_trials']} trials x {d['n_fresh_per_trial']} fresh queries; two-block and shared rule; T = "
            f"{min(x['T'] for x in c)}..{max(x['T'] for x in c)}; n2 = {min(x['n2'] for x in c)}..{max(x['n2'] for x in c)}): mean false firing <= "
            f"floor(alpha (n2 + 1)) / (n2 + 1) + 3 se in every cell (largest excess over the bound {d['max_excess_over_bound']:+.4f}); continuous laws, z against "
            f"the bound (= the exact expectation): mean {z['mean']:+.2f}, range [{z['min']:+.2f}, {z['max']:+.2f}] over {z['n_cells']} cells; {len(ties)} cells "
            f"with ties (discrete / urn): rates " + ", ".join(f"{100 * x['mean_false_firing_rate']:.2f} % <= {100 * x['bound']:.2f} %" for x in ties)
            + f"; eta = +inf cells fire on nothing: {sum(x['eta_infinite_trials'] == d['n_trials'] for x in c)} cell(s); gate == strict event in every trial: "
            f"{all(x['gate_equals_strict_event'][0] == x['gate_equals_strict_event'][1] for x in c)}; every feature through the package gate on random keys "
            f"(2000 trials each, bound {100 * d['feature_cells'][0]['bound']:.2f} %): "
            + ", ".join(f"{x['feature']} {100 * x['mean_false_firing_rate']:.2f} % (z {x['z_vs_bound']:+.2f})" for x in d["feature_cells"])
            + f", gate == strict event {sum(x['gate_equals_strict_event'][0] for x in d['feature_cells'])}/{sum(x['gate_equals_strict_event'][1] for x in d['feature_cells'])}")


def k_a19(d):
    r, g, e, k = d["referee_c1"], d["general"], d["extreme"], d["rank"]
    return (f"referee's check C1 re-run with the library threshold ({r['cases']} cases x {r['inputs_per_case']} inputs): mismatches with the ratio gate "
            f"{r['mismatches_library_threshold_vs_ratio_gate']} (naive product threshold {r['mismatches_naive_product_threshold_vs_ratio_gate']} = referee "
            f"{r['referee_naive_mismatches']}); two-block form (b != 0, u != 1), {g['n_inputs']} boundary inputs: mismatches numpy {g['mismatch_numpy']}, torch "
            f"{g['mismatch_torch']} (product form b + eta u: {g['mismatch_naive_product']}; eta-defining point fires: {g['defining_point_fires']} vs "
            f"{g['defining_point_fires_naive_product']}); extreme magnitudes {e['ok']}/{e['n']} in {1e3 * e['seconds']:.0f} ms; exact rank == Fraction arithmetic "
            f"{k['equal_exact']}/{k['n']} (the float formula is one rank off in {k['float_formula_differs']}, e.g. alpha = 1/3, n = 8: {k['one_third_n8']} not 7)")


def k_a20(d):
    out = []
    for lg, r in d["logs"].items():
        for c in r["checks"]:
            a = c["alphas"][0]
            nd = sum(x["decisions"]["hyb_decisions_differing"] + x["decisions"]["sh_decisions_differing"] for x in c["alphas"])
            nn = sum(x["decisions"]["n_decisions"] for x in c["alphas"])
            de = r.get("default_form")
            out.append(f"{SHORT.get(lg, lg)} / {c['script']}: re-run == stored numbers {c['rerun_reproduces_stored_numbers']}; alpha = {a['alpha']}, "
                       f"{a['resplits']} halves: rows firing {100 * a['library']['two_block']['frac_rows_firing']:.2f} % (referee "
                       f"{100 * a['referee']['two_block']['frac_rows_firing']:.2f} %), T-Gen coverage {100 * a['library']['two_block']['cov_tgen']:.2f} % "
                       f"({100 * a['referee']['two_block']['cov_tgen']:.2f} %); shared scale {100 * a['library']['shared']['frac_rows_firing']:.2f} % / "
                       f"{100 * a['library']['shared']['cov_tgen']:.2f} % ({100 * a['referee']['shared']['frac_rows_firing']:.2f} % / "
                       f"{100 * a['referee']['shared']['cov_tgen']:.2f} %); decisions differing from his rule over all alphas: {nd}/{nn}")
    de = [f"{SHORT.get(lg, lg)} {100 * r['default_form']['frac_rows_firing']:.2f} % / {100 * r['default_form']['cov_tgen']:.2f} % "
          f"({r['default_form']['decisions_differing_from_referee_form']} decisions differ)" for lg, r in d["logs"].items() if r.get("default_form")]
    return "; ".join(out) + ("; the DEFAULT form (normalised coordinate, u = 1) on the same halves at alpha = 0.05: " + ", ".join(de) if de else "")


def k_a21(d):
    s, sim, lo, g = d["sources"], d["simulated_part2"], d["longest_log_cpu"], d["guards"]
    p2 = d["part2_now"]
    return (f"{d['n_logs']} logs registered ({d['n_complete']} complete): " + ", ".join(f"{k} {len(v)}" for k, v in d["logs_by_source"].items())
            + f"; part 2 now: {len(p2.get('registered', []))} registered, {len(p2.get('not_registered_incomplete', []))} not built yet; simulated: no registry at all "
            f"tolerated {sim['absent']['no_exception']} (unknown log raises {sim['absent']['unknown_log_raises']}), complete-but-not-appended log picked up "
            f"{sim['present']['pending_registered'] == sim['present']['newly_registered']}, incomplete log left out {sim['present']['incomplete'] != []}, name appended "
            f"later picked up {sim['appended']['picked_up']}; stored_metrics == testbed files for {sum(v.get('ok', False) for v in d['stored_metrics'].values())}/"
            f"{len(d['stored_metrics'])} logs; harness baseline / evalcache / edit_keys refused from T = 500: "
            f"{all(r['500'] == r['2000'] == 'refused' and r['100'] == r['499'] == 'passed' for r in g.values())}; T = {lo['T']} log on the CPU (factors, rebased "
            f"values, calibrated gate): peak RAM {d['peak_rss_gb']['after']:.1f} GB; final hard negatives: "
            + "; ".join(f"{bb} not built yet" if not v["available"] else f"{bb} " + ", ".join(
                f"{nm}: {x['n_final_rows']} valid rows {x['types']} (P11 set {x['n_p11_rows']}, contained {x['p11_rows_contained']}; source item related to an edit of "
                f"the log: {x['n_src_related_to_log']})" for nm, x in v.items() if nm != "available") for bb, v in d["final_hard_negatives"].items()))


def _where(v):
    """Which cache a test read: the final one (10_final/cache/<backbone>/) or a snapshot of a partial assembly."""
    return "final cache 10_final/cache" if v.get("is_final") else "SNAPSHOT of a partial assembly"


def k_a22(d):
    fc = d["final_caches"]
    ca = "; ".join(f"{bb} ({_where(v)}): {v['summary']['n']} sequences, complete {v['summary']['complete']}, format ok {v['ok']}" if v["available"]
                   else f"{bb}: not built yet" for bb, v in fc.items())
    return (f"synthetic cache: decide (fired, candidates, top edit, own flags) == key-by-key firing() of the gate in {d['n_equal']}/{d['n_decisions']} "
            f"(gate, key set, sequence) cells ({len(d['gates'])} gates x 6 key sets; {d['n_in_band']} cells flagged as inside the decision band, "
            f"{d['n_in_band_decided_differently_key_by_key']} of them decided differently); coords == query maxima to {_g(d['coords_max_rel_diff'])}; router "
            f"{d['router']['equal']}/{d['router']['n']}; selection helpers {sum(d['selection'].values())}/{len(d['selection'])}; files that do not belong together "
            f"(shorter key file, other index, other content with verify) detected {sum(d['consistency'].values())}/{len(d['consistency'])}; refused inputs "
            f"{sum(d['refused'].values())}/{len(d['refused'])}; band: threshold at a coordinate / 1e-12 above / 1e-6 above flagged "
            f"{d['decision_band']['at_coordinate']['near']} / {d['decision_band']['1e-12_above']['near']} / {d['decision_band']['1e-6_above']['near']}; silent-row "
            f"flags == a real hook {d['silent_flags']['equal']}/{d['silent_flags']['n']}; real caches: {ca}")


def k_b11(d):
    out = []
    for nm, v in d["variants"].items():
        bk = "; ".join(f"{k} released {b['released']}/{b['n']}" + (f", eq {b['eq']} (deployed {b['eq_deployed']}, pre-edit {b['eq_pre_edit']}), own released "
                                                                    f"{b['own_released']}" if k.startswith("edit") else f", agree with y0 {b['agree_y0']} (deployed "
                                                                    f"{b['agree_y0_deployed']})") for k, b in v["by_kind"].items())
        out.append(f"{nm}: rows == the selected model's row {v['rows_equal_selected_model']}/{v['n']}, selection == independent rule {v['match_equal']}/{v['n']}, "
                   f"two generations recorded {v['two_generations']}/{v['n']} [{bk}]")
    c = d["tf_composition"]
    return (f"{d['n_items']} items, {d['n_decodes']} decodes for 5 methods (= 2 per item: {d['n_decodes'] == 2 * d['n_items']}); " + "; ".join(out)
            + f"; refused under teacher forcing: {all(d['refused_in_teacher_forcing'].values())}; labelled composition dual_tf_rows (contains): row == teacher-"
            f"forced row of the selected model {c['rows_equal_selected_model']}/{c['n']} ({c['n_deployed_selected']} deployed), probe D1 {c['D1_probe_rows'][0]}/"
            f"{c['D1_probe_rows'][1]} (deployed log {c['D1_probe_rows_deployed']}), T-Loc* {_f(c['locality']['T-Loc*'])} / I-Loc* {_f(c['locality']['I-Loc*'])} "
            f"(deployed {_f(c['locality_deployed']['T-Loc*'])} / {_f(c['locality_deployed']['I-Loc*'])})")


def k_b12(d):
    m = d["methods"]
    r = d["recomputed"]
    out = (f"tok_loc == independent recomputation (argmax vs argmax under W_0): " + ", ".join(f"{k} {c['equal']}/{c['n']} valid + {c['equal_invalid']}/{c['n_invalid']} "
                                                                                              f"invalid-y0 rows" for k, c in r.items())
           + f"; pre-edit weight: T-Loc* {_f(m['pre-edit']['locality']['T-Loc*'])} / I-Loc* {_f(m['pre-edit']['locality']['I-Loc*'])}, invalid rows tok_loc 1 on "
           f"{m['pre-edit']['invalid_rows']['tok_loc_1']}/{m['pre-edit']['invalid_rows']['n']} (token accuracy vs y0 below 1 on "
           f"{m['pre-edit']['invalid_rows']['token_acc_below_1']}); gate: silent rows with tok_loc 1 {m['rescope']['silent_rows_tok_loc_1']}/{m['rescope']['n_silent']}, "
           f"T-Loc* {_f(m['rescope']['locality']['T-Loc*'], 2)} / I-Loc* {_f(m['rescope']['locality']['I-Loc*'], 2)} ({m['rescope']['n_fired']} of {m['rescope']['n']} "
           f"rows fire)")
    dv = d.get("deployed_vs_stored")
    if dv:
        a, j = dv["pool_all"], dv["pool_heldout_disjoint"]
        out += (f"; deployed log, all valid rows ({a['n_text']} text, {a['n_image']} image): T-Loc* {a['T-Loc*']:.2f} / I-Loc* {a['I-Loc*']:.2f} vs stored benchmark "
                f"T-Loc {a['stored']['T-Loc']:.2f} / I-Loc {a['stored']['I-Loc']:.2f}: signed difference {a['diff']['T-Loc']:+.2f} / {a['diff']['I-Loc']:+.2f} points "
                f"(different quantities: decoder-v2 answer at B = 1 on valid rows vs reference answer in padded batches on all 500 items); HELD-OUT-DISJOINT: "
                f"{j['T-Loc*']:.2f} / {j['I-Loc*']:.2f}")
    return out


def k_b13(d):
    lo, kc = d["local"], d["key_cache"]
    rows = "; ".join(f"{g}: " + ", ".join(f"{k} {v['equal']}/{v['n']} ({v['n_fired']} fire)" for k, v in r.items()) for g, r in lo["rows"].items())
    s1, s2 = lo["stats_first_call"], lo["stats_second_call"]
    bd = "; ".join(f"{g}: CPU decision == hook {b['cpu_equals_hook']}/{b['n']}, {b['in_band']} in the band" for g, b in d["decision_band"].items())
    if kc["available"]:
        v = kc["verify"]
        fhn = d.get("final_hard_negatives", {})
        cache = (f"key cache ({_where(kc)}: {kc['n_sequences']} sequences, complete {kc['complete']}; holds {kc['n_items_held']}/{kc['n_items']} items): keys bitwise "
                 f"{v['key_bitwise']}/{v['n_checked']} (+ {v['key_prompt_bitwise']}/{v['key_prompt_checked']} at the decode's token), pre-edit outcome equal "
                 f"{v['pre_equal']}/{v['n_checked']}; rows == Evaluator " + ", ".join(f"{g} {c['equal']}/{c['n']}" for g, c in kc["rows"].items())
                 + f"; {kc['stats']['silent_from_cache']} rows from the cache, {kc['stats']['forwards']} forwards; cache of the other backbone refused: "
                 f"{kc.get('other_backbone_refused', 'not checked (not built)')}; final hard negatives: "
                 + (f"{fhn['n_valid_rows']} valid rows {fhn['types']}, sample of {fhn['n_sample']} rows equal through both paths {fhn['rows_equal']}/{fhn['n_sample']}"
                    if fhn.get("available") else "not built when the test ran"))
    else:
        cache = "key cache of the final experiments: not built when the test ran"
    cal = "; ".join(f"{g} eta {_g(c['eta'], 4)}, {c['n_edits_capped']} of the edits capped (own prompt silenced by the literal rule; bound void for them), own guards "
                    f"released by their edit {c['n_guards_released_by_own_edit_top1']}/{c['n_guards']}" for g, c in d["calibration"].items())
    return (f"gates (two-block, alpha 0.05, n = {next(iter(d['calibration'].values()))['n1'] + next(iter(d['calibration'].values()))['n2']}): {cal}. "
            f"rows == Evaluator rows in every field: {rows}; second call identical: {all(v['equal_second_call'] == v['n'] for r in lo['rows'].values() for v in r.values())}; "
            f"lp='prompt' {d['lp_prompt']['equal']}/{d['lp_prompt']['n']} ({d['lp_prompt']['n_shifted']} shifted rows); invalid-y0 probe rows "
            f"{d['invalid_y0_rows']['equal']}/{d['invalid_y0_rows']['n']}; forwards: Evaluator "
            f"{2 * sum(d['n_items'].values())}, sparse first call {s1['forwards']} (of which {s1['local_captures']} captures), second call {s2['forwards']}; {bd}; "
            f"public functions {all(d['public_functions_equal'].values())}; non-sparse methods unchanged {d['not_sparse_methods_equal']}; {cache}")


def k_b14(d):
    c, k = d["calibration"], d["key_cache_block"]
    return (f"{d['n_code_lines']} code lines executed; {len(d['methods'])} methods x {d['n_items']} probe rows; calibrations: "
            + ", ".join(f"{g} eta {_g(r['eta'], 4)} (n1 {r['n1']}, n2 {r['n2']}, {r['n_edits_capped']} edits capped)" for g, r in c.items())
            + f"; deployed D1 == stored {d['deployed_D1_equals_stored']}/{d['n_items']}; sparse rows == plain Evaluator "
            f"{sum(d['sparse_rows_equal_plain_evaluator'].values())}/{len(d['sparse_rows_equal_plain_evaluator'])} gates; DUAL rows with two generations "
            f"{d['dual']['two_generations']}/{d['dual']['n']}; stored_metrics == metrics.json {d['stored_rel_exact_equals_metrics_json']}; KeyCache example "
            + (f"ran ({_where(k)}): {k['n_sequences']} sequences decided on the CPU, {k['n_fired']} fire" if k["ran"] else f"not run ({k['note']})"))


def k_d1(d):
    l, r, g, st, c = d["load"], d["rss"], d["gate"], d["stored"], d["lean_vs_harness_loader"]
    return (f"{d['name']} (T = {l['T']}, {l['n_batches']} batches; the harness loader would hold {l['batch_deltas_gb_not_held']:.0f} GB of batch deltas) "
            f"loaded lean in {l['seconds']:.0f} s: harness checks "
            f"{sum(bool(v) for k, v in l['checks'].items() if k != 'collapsed_working_point')}/{len(l['checks']) - 1}; peak RAM after load {r['after_load']:.1f} GB, at "
            f"the end {r['end']:.1f} GB (limit {d['limit_gb']:.0f} GB), GPU peak {d['gpu_peak_gb']:.1f} GB; refused: {sum(v is True for v in d['refused'].values())}/"
            f"{len(d['refused'])} harness calls; lean loader == harness loader on {c['log']}: " + ", ".join(f"{k} {c.get(k)}" for k in ("W0", "WE", "C", "U", "Wfac", "Kp_log"))
            + f", checks {c.get('checks_equal')}, lazy batch delta {c.get('lazy_batch_delta_equal')}; stored metrics == files {st['per_edit_flags_equal_file'] and st['probe_flags_equal_rows']}"
            f", B = 1 deployed D1 == stored on {st['sample']['deployed_D1_equals_stored']}/{st['sample']['n']} sampled rows (Rel exact of 20 sampled edits: B = 1 "
            f"{st['sample']['rel_B1_exact']}, stored vendor pass {st['sample']['rel_stored_exact_same_edits']}); default gate at T = {g['T']}: own Rel prompt fires "
            f"{g['rel_own_fired_last']}/{g['n_rel']}, {g['calibration']['n_edits_capped']} edits capped, {g['probe_fired']}/{g['n_probe']} sampled probes fire; "
            f"registries: " + ", ".join(f"{k} {len(v)}" for k, v in d["registry"]["by_source"].items()))


CPU = [("a1_rebase_identity", "a. rebase identity", k_a1), ("a2_replay_attains_targets", "a. replay attains every target", k_a2),
       ("a3_nonfiring_bitwise", "a. non-firing keys give W_0 k bitwise", k_a3), ("a9_hook_vs_reference", "a. hook vs position-by-position reference", k_a9),
       ("a15_hook_guards", "a. hook input guards (review F3, F4, PAD positions)", k_a15),
       ("a4_least_repair", "a. least repair", k_a4), ("a5_conformal", "a. conformal quantile", k_a5), ("a6_null_repair", "a. null repair equalities", k_a6),
       ("a7_scope_cut", "a. scope_cut vs brute force", k_a7), ("a8_baselines", "a. baselines fire where defined", k_a8),
       ("a10_witness_repair", "a. witness repair / session", k_a10), ("a11_calibrate_log_only", "a. log-only calibration", k_a11),
       ("a16_guard_contract", "a. guards are must-fire keys (review F2)", k_a16),
       ("a12_real_logs", "a. real logs (CPU): rebase vs cached targets, splits", k_a12),
       ("a13_real_logs_firing", "a. real logs (CPU): firing flags from P10's stored coordinates vs P10 / P11a", k_a13),
       ("a14_real_keys_operators", "a. real keys (CPU): conformal vs P11b, least / null / witness repair", k_a14),
       ("a17_two_block_naive", "e1. two-block / shared rule vs naive re-implementation (ties)", k_a17),
       ("a18_two_block_validity", "e1. two-block / shared rule: validity on exchangeable scores", k_a18),
       ("a19_float_safe", "e1. floating-point-safe thresholds, exact rank (referee C1)", k_a19),
       ("a20_two_block_referee", "e1. referee's reference numbers on the stored keys (C6)", k_a20),
       ("a21_scale_registry", "e4. registries, stored metrics, harness guards (CPU)", k_a21),
       ("a22_keycache", "e5. KeyCache on a synthetic cache; silent-row flags", k_a22)]
GPU = [("b1_linear", "b(i) linear log without E vs W_E", k_b1), ("b2_off", "b(ii) tau = +inf", k_b2),
       ("b345_pass", "b(iii)-(v) all rows, 5 configs", k_b345), ("b6_harness", "b(vi) harness corrections", k_b6),
       ("b7_decode", "decode pipeline (D2, free generation)", k_b7), ("b9_paths", "teacher-forced path vs decode path (review F1)", k_b9),
       ("b10_own", "own prompts at B = 1, guards (review F5)", k_b10), ("c_timing", "c. timing", k_c),
       ("b8_usage", "docs/USAGE.md example executes", k_b8), ("b11_dual", "e2. baseline DUAL in free generation", k_b11),
       ("b12_tokloc", "e3. token-level locality at B = 1", k_b12), ("b13_sparse", "e5. SparseEvaluator rows == Evaluator rows", k_b13),
       ("b14_usage2", "docs/USAGE.md examples 2 and 3 execute", k_b14)]
SCALE = [("TB-Q2000-ae-l2_3", "d1_scale", "e4. scale-safe use at T = 2000 (fresh process)", k_d1)]


def _row(path, label, key, code, tests):
    if not os.path.exists(path):
        return f"| {label} | not run | - |"
    with open(path) as f:
        d = json.load(f)
    kind = "cpu" if os.path.basename(path).startswith("cpu_") else "gpu"             # scale_*: produced by the GPU test code
    stale = "" if (d.get("code") == code and d.get("tests") == (tests or {}).get(kind)) else " (STALE: produced by an older version of the code)"
    if d.get("not_run"):                                               # [added for the public artifact] input not part of the release / not present
        return f"| {label} | not run (input not available) | {d['not_run'][:300]} |"
    if d.get("error"):
        return f"| {label} | ERROR{stale} | {d['error'].strip().splitlines()[-1][:200]} |"
    try:
        kn = key(d)
    except Exception as e:                                             # a malformed JSON must not hide the pass / fail flag
        kn = f"(key numbers unavailable: {type(e).__name__}: {e})"
    return f"| {label} | {'PASS' if d['pass'] else 'FAIL'}{stale} | {kn} |"


def x_families(d):
    out = []
    for x in d["logs"]:
        if x.get("name") is None:
            out.append(f"{x['source']}: {x['note']}")
            continue
        s, g, st = x["sample"], x["gate"], x["stored"]
        hn = x["hard_negatives"]
        out.append(f"{x['name']} ({x['source']}, T = {x['T']}, {'lean' if x['lean'] else 'harness loader'}, {x['seconds_load']:.0f} s): checks {x['checks_ok']}, deployed "
                   f"D1 == stored {s['deployed_D1_equals_stored']}/{s['n_probe']}, pre-edit D1 {s['pre_edit_D1']}, Rel B = 1 {s['rel_B1_exact']}/{s['n_edits']} (stored vendor "
                   f"pass {s['rel_stored_exact_same_edits']}), default gate: own Rel fires {g['rel_own_fired_last']}/{g['n_rel']}, probes firing {g['probe_fired']}/"
                   f"{g['n_probe']}, stored T-Loc {_f(st['T-Loc'])} / I-Loc {_f(st['I-Loc'])}"
                   + (f", hard negatives {hn['n_valid_rows']} rows {hn['types']}" if "n_valid_rows" in hn else ", no hard negatives for this log"))
    return f"one model, {len(d['logs'])} logs (load(reuse=)), peak RAM {d['peak_rss_gb']:.1f} GB: " + "; ".join(out)


def x_p11(d):
    out = []
    for bb, v in d["backbones"].items():
        out.append(f"{bb}: not built" if not v["available"] else
                   f"{bb} ({_where(v)}, {v['n_cache_sequences']} cached sequences, complete {v['cache_complete']}): {v['n_in_both']} of {v['n_p11_sequences']} P11 sequences held, keys "
                   f"bitwise equal {v['keys_bitwise_equal']}/{v['n_in_both']}, last prompt index equal {v['last_prompt_index_equal']}/{v['n_in_both']}, CPU decisions "
                   f"(thresholds 0.35) equal {v['decisions_equal']}/{v['n_in_both']}; CPU decisions from the cache == firing OBSERVED in P11a's forwards: "
                   + ", ".join(f"{c} {o['equal']}/{o['n_in_cache']} ({o['n_fired_observed']} fire)" for c, o in v.get("observed_in_forwards", {}).items()))
    return "key cache against the keys captured independently by P11b: " + "; ".join(out)


def x_peak(d):
    return (f"{d['name']} (T = {d['T']}, {d['n_batches']} batches = {d['batch_deltas_gb']:.1f} GB of applied deltas) with the "
            f"{'lean loader' if d['lean'] else 'harness loader'}: peak RAM {d['peak_rss_gb']:.1f} GB, {d['seconds']:.0f} s, harness checks {d['checks_ok']}")


def x_reg(d):
    lg = "; ".join(f"{SHORT.get(k, k)}: numbers {v['regression_numbers_identical']}, agreement counts {v['agreement_identical']}, rows identical in every previous field "
                   f"{v['rows_identical_in_every_previous_field']}/{v['n_rows']} (fields added to rows: {', '.join(v['fields_added_to_rows']) or 'none'})"
                   if v.get("available") else f"{SHORT.get(k, k)}: not available" for k, v in d["logs"].items())
    return (f"regression rows of the table identical to the table before this work (`tests/out_v1/table.md`): {d['table_regression_rows_identical']} "
            f"({d['n_table_regression_rows']} rows); b345 pass against the previous outputs (`tests/out_v1/`): {lg}")


EXTRA = [("extra_load_families.json", "e4. one log of every registry family loads and evaluates (extra)", x_families, "pass"),
         ("extra_cache_vs_p11.json", "e5. key cache vs independently captured P11 keys (extra, CPU)", x_p11, "pass"),
         ("extra_load_peak_TB-Q2000-ae-l2_3_lean.json", "e4. peak RAM, lean loader (extra)", x_peak, "pass"),
         ("extra_load_peak_TB-Q2000-ae-l2_3_harness.json", "e4. peak RAM, harness loader (extra, for comparison)", x_peak, "pass"),
         ("regression_unchanged.json", "regression unchanged (tests/extra/regression_check.py)", x_reg, "unchanged")]


def _prev_cmp(d, p_old):
    """The regression pass `d` against the same pass of the REVIEWED version of the package (JSON at p_old): are the
    regression numbers equal, and how many per-row records carry the same value in every field the old row had?"""
    if not os.path.exists(p_old):
        return None
    with open(p_old) as f:
        old = json.load(f)
    n = same = 0
    numbers = True
    for c, rows in old.get("rows", {}).items():
        new = {r["sid"]: r for r in d.get("rows", {}).get(c, [])}
        for r in rows:
            q = new.get(r["sid"])
            n += 1
            same += int(q is not None and all(q.get(k) == v for k, v in r.items()))
        numbers = numbers and old["regression"][c]["mine"] == d.get("regression", {}).get(c, {}).get("mine")
    return {"code_old": old.get("code"), "n": n, "same": same, "numbers_same": bool(numbers), "n_configs": len(old.get("rows", {}))}


def render(out_dir, code=None, tests=None):
    L = [f"Rendered from `tests/out/*.json` only (package fingerprint `{code}`, test code `{tests}`; a row marked STALE was produced by other code).",
         "", "## Tests", "", "| test | result | key numbers |", "|---|---|---|"]
    for name, label, key in CPU:
        L.append(_row(os.path.join(out_dir, f"cpu_{name}.json"), label, key, code, tests))
    for lg in LOGS:
        for name, label, key in GPU:
            L.append(_row(os.path.join(out_dir, f"gpu_{lg}_{name}.json"), f"{label} [{SHORT[lg]}]", key, code, tests))
    for lg, name, label, key in SCALE:
        L.append(_row(os.path.join(out_dir, f"scale_{lg}_{name}.json"), f"{label} [{lg}]", key, code, tests))
    ex = []
    for fname, label, key, flag in EXTRA:
        path = os.path.join(out_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        stale = "" if d.get("code") in (None, code) else f" (package fingerprint `{d.get('code')}`)"
        try:
            kn = key(d) if not d.get("error") else d["error"].strip().splitlines()[-1][:200]
        except Exception as e:
            kn = f"(key numbers unavailable: {type(e).__name__}: {e})"
        ex.append(f"| {label} | {('PASS' if flag == 'pass' else 'YES') if d.get(flag) else ('FAIL' if flag == 'pass' else 'NO')}{stale} | {kn} |")
    if ex:
        L += ["", "## Extra checks (scripts in `tests/extra/`, not part of the fingerprinted suite)", "", "| check | result | key numbers |", "|---|---|---|"] + ex
    L += ["", "## Regression against the exploratory experiments (B = 1 teacher forcing; Rel / T-Gen / I-Gen exact %, D1 % over all valid probe rows, "
          "hard-negative D1 %)", "",
          "| config | log | theirs: Rel / T-Gen / I-Gen; D1; HN D1 | this package | flags equal (exact; fired): edits | probes | hard negatives |",
          "|---|---|---|---|---|---|---|"]

    def num(x):
        hn = "-" if x["hn_D1"] is None else f"{x['hn_D1']:.1f}"
        return f"{x['rel']:.0f} / {x['tgen']:.0f} / {x['igen']:.0f}; {x['probe_D1']:.1f}; {hn}"

    def ag(a):
        return "-" if a["n"] == 0 else f"{a['exact_equal']}/{a['n']}; {a['fired_equal']}/{a['n']}"
    prev_dir = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "out_prev")        # the outputs behind the reviewed version's table
    prev = []
    for lg in LOGS:
        p = os.path.join(out_dir, f"gpu_{lg}_b345_pass.json")
        if not os.path.exists(p):
            L.append(f"| (not run) | {SHORT[lg]} | - | - | - | - | - |")
            continue
        with open(p) as f:
            d = json.load(f)
        for c in REG:
            r = d.get("regression", {}).get(c)
            if r is None:
                continue
            a = r["agreement"]
            L.append(f"| `{c}` | {SHORT[lg]} | {num(r['theirs'])} | {num(r['mine'])} | {ag(a['edit'])} | {ag(a['probe'])} | {ag(a['hn'])} |")
        cmp = _prev_cmp(d, os.path.join(prev_dir, f"gpu_{lg}_b345_pass.json"))
        if cmp is not None:
            prev.append(f"{SHORT[lg]}: regression numbers of the {cmp['n_configs']} configs identical: {cmp['numbers_same']}; per-row records identical in every "
                        f"field of the reviewed version's row (exact, token accuracy, firing flags, candidates): {cmp['same']}/{cmp['n']} (reviewed code `{cmp['code_old']}`)")
    if prev:
        L += ["", "Against the reviewed version of the package (its own outputs, kept in `tests/out_prev/`), default arguments (`lp=\"tf\"`): "
              + "; ".join(prev) + "."]
    return "\n".join(L)
