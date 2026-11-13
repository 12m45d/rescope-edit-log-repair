"""out/tests/*.json -> TESTS.md (no hand-typed numbers).  usage: python -m harness.tests.render_tests"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from harness.common import HARNESS_DIR  # noqa: E402

OUT = os.path.join(HARNESS_DIR, "out", "tests")


def g(name):
    p = os.path.join(OUT, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    md = ["# Harness unit tests", "", "Rendered by `harness/tests/render_tests.py` from `out/tests/*.json` "
          "(produced by `harness/tests/run_tests.py`). Do not edit by hand.", ""]
    z = g("zero")
    if z:
        md += ["## zero — zero correction is bit-identical; exactness of the speed-ups", "",
               f"**pass = {z['pass']}** ({z['seconds']:.0f} s, {z['gpu']})", "",
               f"- logits of {z['n_sequences']} sequences (complaints + edit rel/igen) bitwise equal to W_E under: " +
               ", ".join(f"{k} {v}" for k, v in z["logits_bitwise"].items()),
               "- full evaluate() record identical to NONE: " + ", ".join(f"{k} {v}" for k, v in z["eval_identical"].items()),
               f"- NONE record: edits broken {z['none_record_unchanged']['edits_broken']}, fixed "
               f"{z['none_record_unchanged']['edits_fixed']}, held-out rows changed {z['none_record_unchanged']['collateral_changed_rows']}, "
               f"Loc items changed {z['none_record_unchanged']['loc_items_changed']}, per-edit em/acc identical {z['none_record_unchanged']['edits_identical']}",
               "- suffix engine == full forward (bitwise) at: " + ", ".join(f"{k} {v}" for k, v in z["suffix_equals_full"].items()),
               f"- W(R = all edits) == W_0 bitwise: {z['rollback_all_equals_W0']}; module-input keys identical at W_E and W_0: "
               f"{z['keys_independent_of_edited_weight']}",
               f"- vision-feature cache bitwise (miss and hit): {z['vision_cache_bitwise']}; prebuilt eval batches == vendor "
               f"run_probes: {z['prebuilt_batches_equal_vendor_run_probes']}",
               f"- load checks {z['load_checks']}; baseline checks {z['baseline_checks']}", ""]
    c = g("comp_w2")
    if c:
        md += ["## comp_w2 — COMP reproduces the old W2 H1 (K_last, answer positions, fp64 adapter, alpha 1)", "",
               f"**pass = {c['pass']}** ({c['seconds']:.0f} s). 20 random P8 complaints (seed 2026, 10 image / 10 text), "
               f"ci = {c['complaints_ci']}.", "",
               f"- resolved: new ungated {c['resolved_new_ungated']}/{c['n']}, new gated {c['resolved_new_gated']}/{c['n']}, "
               f"old cb_run H1 {c['resolved_old']}/{c['n']} (old H1 on all 200: {c['old_200_H1']})",
               f"- per-complaint flag agreement with cb_run: ungated {c['flag_agreement_ungated']}/{c['n']}, gated "
               f"{c['flag_agreement_gated']}/{c['n']}; max rel. diff of ||correction||_F {c['dW_fro_max_rel_diff']:.2e}",
               f"- rel edits broken (sum over the 20): new {c['edits_broken_new']} (vendor batch-8 eval) vs old {c['edits_broken_old']} "
               "(cb_run B=1 full forwards)", f"- load checks {c['load_checks']}", ""]
    fo = g("fo_w2")
    if fo:
        md += ["## fo_w2 — ROLLBACK-FO reproduces P8's greedy-FO flag (budget 10, old W2)", "",
               f"**pass = {fo['pass']}** ({fo['seconds']:.0f} s). Same 20 complaints.", "",
               f"- resolved flag agreement {fo['flag_agreement']}/{fo['n']} (new {fo['resolved_new']}, P8 {fo['resolved_old']}); "
               f"identical removal sequence {fo['removal_sequence_identical']}/{fo['n']}; internal stop == evaluate() "
               f"{fo['internal_vs_evaluate_agree']}/{fo['n']}",
               f"- mean k removed new {fo['mean_k_new']:.2f} vs P8 {fo['mean_k_old']:.2f}; wall-clock per complaint new "
               f"{fo['mean_sec_new']:.2f} s (suffix engine, factorised scores) vs P8 {fo['mean_sec_old']:.1f} s",
               f"- load checks {fo['load_checks']}", ""]
        diff = [r for r in fo["rows"] if not r["same_R"]]
        if diff:
            md += ["Complaints whose removal sequence differs:", "", "| ci | ptype | new R | P8 R | new res | P8 res |", "|---|---|---|---|---|---|"]
            md += [f"| {r['ci']} | {r['ptype']} | {r['new']['R']} | {r['old']['R']} | {r['new']['resolved']} | {r['old']['resolved']} |" for r in diff]
            md.append("")
    r = g("reedit")
    if r:
        md += ["## reedit — REEDIT reproduces baselines/smoke (TB-Q100-ae, 8 complaints x {k0, k0+kp})", "",
               f"**pass = {r['pass']}** ({r['seconds']:.0f} s).", "",
               f"- agreement with the smoke: resolved {r['resolved_agree']}/{r['n']}, Rel exact after {r['rel_exact_agree']}/{r['n']}, "
               f"collateral broken (smoke's 20 probes) {r['coll_broken_agree']}/{r['n']}, z* steps {r['z_steps_agree']}/{r['n']}; "
               f"max |dRel token| {r['rel_token_max_absdiff']:.3f}; max rel. diff dW/W0 {r['dW_over_W0_max_reldiff']:.2e}",
               f"- factorised GPU solve vs vendor Filters.solve: max rel. diff of D {r['vendor_rel_diff_D_max']:.2e}, W_after entries "
               f"differing (max) {r['vendor_W_after_entries_differing_max']}",
               f"- wall-clock median: solve {r['sec_solve_new_median']:.3f} s, repair total {r['sec_total_new_median']:.2f} s "
               f"(smoke, vendor dense solve with diagnostics: {r['sec_total_old_median']:.2f} s)", "",
               "| complaint | variant | resolved new/old | Rel exact new/old | coll broken new/old | dW/W0 new/old | z steps new/old |",
               "|---|---|---|---|---|---|---|"]
        md += [f"| {x['group']} | {x['variant']} | {x['new']['resolved']}/{x['old']['resolved']} | {x['new']['rel_exact']:.0f}/"
               f"{x['old']['rel_exact']:.0f} | {x['new']['coll_broken']}/{x['old']['coll_broken']} | {x['new']['dW_over_W0']:.4e}/"
               f"{x['old']['dW_over_W0']:.4e} | {x['new']['z_steps']}/{x['old']['z_steps']} |" for x in r["rows"]]
        md.append("")
    k = g("kseq")
    if k:
        md += [f"## kseq — protecting every position of the {k['n_seq']} shortest edit sequences (K_seq:{k['n_seq']})", "",
               f"**pass = {k['pass']}** ({k['seconds']:.0f} s). {k['n_positions']} protected positions (d_in {k['d_in']}; "
               f"d_in / mean edit length = {k['n_star_d_in_over_mean_len']:.1f} sequences).", "",
               f"- repairs leaving ALL {k['n_seq']} protected sequences bit-identical (full logits, every position): gated adapter "
               f"{k['repairs_all_identical_gated']}/{k['n_repairs']}, ungated {k['repairs_all_identical_ungated']}/{k['n_repairs']}; "
               f"complaints resolved {k['resolved']}/{k['n_repairs']}", ""]
        md += ["| complaint | gated identical | ungated identical | protect rank | ||corr||_F |", "|---|---|---|---|---|"]
        md += [f"| {x['group']} | {x['gated']['n_identical']} | {x['ungated']['n_identical']} | {x['gated']['rank']} | {x['gated']['fro']:.3f} |" for x in k["rows"]]
        md.append("")
    fe = g("fasteval")
    if fe:
        md += ["## fasteval — exact fast evaluation path (suffix engine) == full forwards", "",
               f"**pass = {fe['pass']}** ({fe['seconds']:.0f} s): identical evaluate() records {fe['n_identical']}/{fe['n']} "
               f"(COMP adapter, COMP merged, ROLLBACK-FO, REEDIT on 4 complaints); median eval time fast "
               f"{fe['sec_fast_median']:.1f} s vs full {fe['sec_full_median']:.1f} s.", ""]
    lv = g("llava")
    if lv:
        md += [f"## llava — the harness on LLaVA-1.5-7B ({lv['log']})", "",
               f"**pass = {lv['pass']}** ({lv['seconds']:.0f} s); suffix == full bitwise {lv['suffix_equals_full']}; peak GPU "
               f"memory allocated {lv['max_gpu_mem_GB']:.1f} GB; load checks {lv['load_checks']}; baseline checks {lv['baseline_checks']}", "",
               "| method | cfg | resolved | edits broken | held-out non-complaints lost | ||corr||_F | passes | s |", "|---|---|---|---|---|---|---|---|"]
        md += [f"| {x['method']} | {x['cfg']} | {x['resolved']} | {x['edits_broken']} | {x['lost']} | {x['fro']:.3f} | {x['passes']} | {x['sec']:.2f} |"
               for x in lv["rows"]]
        md.append("")
    rn = g("reedit_noise")
    if rn:
        md += ["## reedit_noise — run-to-run noise of z* (original baselines/reedit.py vs harness, 2 runs each)", "",
               f"W_E identical to reedit.py's: {rn['W_E_identical_to_reedit_py']}; relative spread of the applied ||delta||/||W_0|| "
               f"over the 4 runs: " + ", ".join(f"{k} {v:.2e}" for k, v in rn["rel_spread_applied_dW"].items()), "",
               "reedit.py reports dW/W0 with CPU fp32 norms (biased by ~1e-3 vs fp64); the fp64 column is comparable across rows.", "",
               "| complaint | implementation | run | dW/W0 as reported | dW/W0 fp64 | W_after == harness (same run index) | z* ||delta||/||h|| | z* NLL | z* steps |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {x['group']} | {x['impl']} | {x['rep']} | {x['applied_dW_over_W0']:.5e} | {x['applied_dW_over_W0_fp64']:.5e} | "
               f"{x.get('W_after_bitwise_equal_harness_same_rep', '-')} | {x['z_delta_over_h']:.5f} | {x['z_nll']:.5f} | {x['z_steps']} |"
               for x in rn["rows"]]
        md.append("")
    kp = g("kpc")
    if kp:
        md += [f"## kpc — protect \"K_pc:<r>\" (top-r principal directions of K_all; RQ3 extension, {kp['log']})", "",
               f"**pass = {kp['pass']}** ({kp['seconds']:.0f} s). K_all = {kp['n_keys_K_all']} keys, d_in {kp['d_in']}; "
               f"eigenbasis orthonormality err {kp['orthonormal_err']:.1e}, eigen-residual {kp['eig_residual_rel_max']:.1e}, "
               f"descending {kp['evals_descending']}, lambda_min/lambda_max {kp['eval_min_over_max']:.2e} (K_all full rank); "
               f"energy of U_64 from the keys {kp['energy64_from_keys']:.12f} vs from the eigenvalues {kp['energy64_from_evals']:.12f}; "
               f"union K_last + K_pc:64 rank {kp['union_rank']} (independent matrix_rank {kp['union_rank_independent']}); "
               f"pre-existing specs give the bitwise same basis as before the extension: {kp['compat_bitwise_basis']}", "",
               "| complaint | A | K_pc:0 == none | K_pc:d_in infeasible | ||corr|| at d_in | module out == W_E | rank kept at d_in - A | P1 | P2 | ||corr|| at d_in - A | ||corr|| none |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {x['group']} | {x['A']} | {x['pc0_equals_none']} | {x['full_infeasible']} | {x['full_fro']:.1f} | "
               f"{x['full_module_out_bitwise_WE']} | {x['r_d_minus_A_rank_kept']} | {x['r_d_minus_A_P1_rel']:.1e} | "
               f"{x['r_d_minus_A_P2']:.1e} | {x['r_d_minus_A_fro']:.1f} | {x['none_fro']:.2f} |" for x in kp["rows"]]
        md.append("")
    q2 = g("rq2cells")
    if q2:
        md += [f"## rq2cells — RQ2 options of COMP (solve=\"alphaedit\") and REEDIT (solve=\"exact\", form=\"adapter\", targets=) ({q2['log']})", "",
               f"**pass = {q2['pass']}** ({q2['seconds']:.0f} s). {q2['n']} random complaints; K_ans protection rank "
               f"{q2['protect_rank_K_ans']}. Defaults unchanged (COMP() == COMP(solve='exact') bitwise; REEDIT() == vendor solve, "
               "W_after bitwise; reusing REEDIT's own z* reproduces W_after bitwise and counts no zopt pass).", "",
               "| complaint | COMP default bitwise | COMP-AE vs vendor (rel D) | COMP-AE attainment | REEDIT vs vendor (W entries diff) | reused z* W bitwise | REEDIT adapter vs vendor (rel D) | REEDIT-exact attainment | REEDIT-exact max abs R^T Q | protected K_ans outputs bitwise |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {x['group']} | {x['comp_default_bitwise']} | {x['comp_ae_vendor_rel_diff_D']:.1e} | {x['comp_ae_attainment']:.3f} | "
               f"{x.get('reedit_default_vendor_W_after_differing', '-')} | {x.get('reedit_reused_targets_W_after_bitwise', '-')} | "
               f"{x.get('reedit_adapter_vendor_rel_diff_D', float('nan')):.1e} | {x.get('reedit_exact_attainment', float('nan')):.12f} | "
               f"{x.get('reedit_exact_RtQ_max', float('nan')):.1e} | {x.get('reedit_exact_protected_outputs_bitwise', '-')} |"
               for x in q2["rows"]]
        md.append("")
    q4 = g("rq4_ext")
    if q4:
        sq, rb, rc = q4["sequential"], q4["rollback_offload"], q4["reedit_chunk"]
        md += ["## rq4_ext — RQ4 options (split fraction, per-item eval, sequential COMP, offloaded rollback, chunked REEDIT)", "",
               f"**pass = {q4['pass']}** ({q4['seconds']:.0f} s, {q4['log']}; `tests/test_rq4_ext.py`)", "",
               f"- split: {q4['split']}",
               f"- COMP defaults bitwise with the new options at their defaults: {q4['comp_defaults_bitwise']}; "
               f"evaluate(per_item=True): {q4['per_item']}",
               f"- sequential COMP (3 complaints, accumulating protection): restore rel. err seq {max(sq['restore_rel_err_seq']):.1e} "
               f"(batch {max(sq['restore_rel_err_batch']):.1e}), protected-span leak {sq['protect_leak_rel']:.1e}, R orthonormality "
               f"{sq['R_orthonormality_err']:.1e}, ||X|| seq {sq['fro_seq']:.6f} vs batch {sq['fro_batch']:.6f} (batch <= seq: "
               f"{sq['batch_le_seq']}), rank {sq['rank_seq']}/{sq['rank_batch']}",
               f"- ROLLBACK_FO offload: same R {rb['same_R']}, same trajectory {rb['same_trajectory']}, same W {rb['same_W']}, "
               f"flags consistent {rb['flags_consistent']}",
               f"- REEDIT chunk: text-only chunk 8 == old path bitwise {rc['text_only_chunk8_bitwise']}; mixed rows resolved "
               f"chunk 8 {rc.get('mixed_resolved')} / chunk 1 {rc.get('mixed_chunk1_resolved')}; rel. diff of the update chunk 1 vs 8 "
               f"{rc.get('mixed_chunk1_vs_chunk8_rel_diff_update', float('nan')):.3f} (batch-shape sensitivity of the bf16 capture; "
               "informative)", ""]
    open(os.path.join(HARNESS_DIR, "TESTS.md"), "w").write("\n".join(md) + "\n")
    print("wrote TESTS.md")


if __name__ == "__main__":
    main()
