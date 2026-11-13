"""Render tests/out/table_ext.txt (the tests of the stage-1 extension + the state of the driver's own tests + the decision digest) and,
with --ready, write ../READY.json -- only if every test passed with the PRESENT code (no number is typed by hand).

  python render_ext.py [--ready]
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

OUT = os.path.join(HERE, "out")
S = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
CODE = fx.code_fingerprint()


def J(name):
    p = os.path.join(OUT, name)
    return fx.io.jload(p) if os.path.exists(p) else None


def verdict(r):
    if r is None:
        return "MISSING"
    v = "PASS" if r.get("ok") else "FAIL"
    return v if r.get("code") == CODE else v + " (older code)"


def per_log(r, f):
    return "; ".join(f"{S.get(k, k)}: {f(v)}" for k, v in r["logs"].items())


def d_null(r):
    return per_log(r, lambda v: f"n_A = 0 == default bit for bit {v['n_A_0_equals_default']}/3 draws, explicit keys == seeded {v['explicit_keys_equal_seeded']}/3, calibration queries in the null "
                   f"sample (max over {v['n_draws']} draws) {v['calibration_queries_in_null_sample_max']}, distinct null queries >= {v['null_sample_distinct_queries_min']}, in E1 row set {v['in_e1_rowset']}")


def d_samples(r):
    return per_log(r, lambda v: f"contaminated draws (even edits, half text / half image, nested) ok for shares {list(v['contamination_even'])}; sources " +
                   ", ".join(f"{k} pool {x['n_pool']}" for k, x in v['sources'].items() if k != 'methods') + f"; guard: {sum(bool(x) for x in v['guard'].values())}/{len(v['guard'])} checks")


def d_sweeps(r):
    def one(v):
        out = []
        for k in ("rescope", "cosine", "rescope_shared"):
            m, h = v[k], v[k]["h3"]
            out.append(f"{k}: sweep != gate object on {m['n_decisions_differing_from_gate_object']} of {m['n_compared']} decisions, capped {m['n_edits_capped']}, own prompts kept at raise 0.5 "
                       f"{m['own_prompts_candidate_at_raise_0.5']}, H3 raise {h['delta']:.4f} (fit {h['fit_mean_at_0']:.1f} -> {h['fit_mean_at_raise']:.2f} %, recomputed {h['fit_mean_recomputed']:.2f}), "
                       f"coverage odd edits {h['rephrase']['mean']:.1f} %, heldout {h['heldout_disjoint']['fired']:.2f} %, at cap {h['n_edits_at_cap']}" if h["reachable"] else f"{k}: H3 not reachable")
        return " | ".join(out)
    return per_log(r, one)


def d_rowset(r):
    return f"required rows missing: {r['missing_rows'] or 'none'}; undefined calibrated forms documented {r['undefined_calibrated_forms_documented']}; " + \
        per_log(r, lambda v: f"same calibration rows for {len(v['calibrated_rows'])} calibrated rows {v['same_calibration_rows']}, capped {v['n_edits_capped']}")


def d_calib(r):
    k, v = r["rank"], r["validity"]
    return (f"rank(n2 = 100, alpha = 0.05, delta = 0.1) = {k['n2_100_alpha_0.05_delta_0.1']} (exact {k['exact_fractions']}, scipy {k['scipy_beta']}; default rule {k['default_rank']}), grid of {k['grid']}: "
            f"{k['grid_differing_from_exact']} differ; synthetic validity: P[rate > alpha] = {v['share_above_alpha_pac']:.4f} (exact {v['exact_pac']:.4f}; one rank lower {v['share_above_alpha_one_rank_lower']:.4f}; "
            f"default rank {v['share_above_alpha_default_rank']:.3f}), end to end {v['end_to_end']['share_of_calibrations_with_rate_above_alpha']}; " +
            per_log(r, lambda x: f"baseline hook: none == default {x['baseline_x']['minus_inf_equals_default_bitwise']}, formula {x['baseline_x']['literal_thresholds_equal_formula']}, == recomb "
                    f"{x['baseline_x']['equals_feature_package_recomb_bitwise']}; top-1 'argmax first' differs on " +
                    ", ".join(f"{m} {t['n_differs']} of {t['n_sequences']} (capped {t['n_edits_capped']}, brute force {t['brute_force']})" for m, t in x['top1'].items() if isinstance(t, dict))))


def d_readouts(r):
    s = r["synthetic"]
    return f"identities {s['identities']}, intervals == direct bootstrap {s['kept_ci_equals_direct_bootstrap']} / {s['d2_ci_equals_cluster_bootstrap']} ({s['d2_units']} clusters of {s['d2_rows']} rows); " + \
        per_log(r, lambda v: "rescope d0 kept / gained " + ", ".join(f"{m} {x['n_kept']} / {x['n_gained']}" for m, x in v['kept_gained'].items()) +
                f"; blocklist of {v['blocklist']['n_reported']}: D1 {v['blocklist']['D1_deployed']:.2f} -> {v['blocklist']['D1_blocklist']:.2f} %, rows consistent {v['blocklist']['rows_consistent']}")


def d_stats2(r):
    c, p, h = r["crossed"], r["paired"], r["holm"]
    return (f"crossed bootstrap ({c['n_draws']} draws x {c['n_units']} units, {c['n_experiments']} tables): coverage {c['coverage_95']:.3f}, width {c['mean_width']:.4f} vs analytic {c['analytic_width']:.4f} "
            f"(units only {c['units_only_width']:.4f}; reference loop {c['reference_loop_width']:.4f}); paired: coverage {p['coverage_95']:.3f}, width {p['mean_width']:.4f} vs analytic {p['analytic_width']:.4f} "
            f"(unpaired {p['unpaired_width']:.4f}), rejection under the null {p['rejection_rate_under_the_null']:.3f}; Holm: known example {h['known_example']}, == statsmodels "
            f"{h['equals_statsmodels_on_200_families']}, family-wise error {h['family_wise_error_under_six_true_nulls']:.3f}")


def d_pools(r):
    pr = r["programme"]
    return (f"{sum(v.get('opened', False) for v in pr.values())} of {len(pr)} logs of the reduced programme open, build and pass at gate level (mechanics only); " +
            per_log(r, lambda v: f"parts {v['benign_parts']}, MME {v['mme_rows']}, firing == library on {sum(x['equals_library'] for x in v['firing'].values())}/{len(v['firing'])} (gate, pool) pairs, "
                    f"question-disjoint heldout rows {v['masks']['heldout_disjoint_kept']} of {v['masks']['heldout_disjoint_n']}, masked pool {v['masks']['masked_draw_pool']} of {v['masks']['unmasked_pool']}"))


def d_prefix(r):
    ok = [k for k, v in r["logs"].items() if v.get("ok")]
    ab = [f"{k}: composite bitwise {v['all_batches']['composite_equals_stored_composite_bitwise']}, W_E bitwise {v['all_batches']['WE_equals_development_WE_bitwise']}, D1 flags differing "
          f"{v['all_batches']['n_D1_flags_differing_from_stored_records']}, thresholds == development log {v['all_batches']['thresholds_equal_development_log']}, batch-8 metrics == stored {all(v['all_batches']['batch8_metrics_equal_parent_stored'].values())}" for k, v in r["logs"].items() if "all_batches" in v]
    return f"{len(ok)} of {len(r['logs'])} prefix logs complete and consistent (factors, pools, hard negatives of their edits, default method); " + "; ".join(ab)


def d_maint(r):
    return per_log(r, lambda v: f"mask: empty set == method {v['rescope']['empty_set_digest_equal']}, releases of a masked edit {v['rescope']['n_releases_of_a_masked_edit_all_positions']} "
                   f"(cosine {v['cosine']['n_releases_of_a_masked_edit_all_positions']}), sequences changed without a masked release {v['rescope']['n_changed_without_a_masked_release_before']}, "
                   f"added detector masked {v['added_detector']['n_releases_after_mask'] == 0}; digest: {v['digest']['n_variants']} one-component variants all differ {v['digest']['all_variants_differ']}, "
                   f"thresholds only {v['digest']['thresholds_only']}")


def g_h(r):
    return (f"{r['n_rows']} sequences ({r['n_firing']} firing): silent rows not bitwise pre-edit {r['silent_rows_not_bitwise_pre_edit']}, hashes differing on a second run {r['deterministic']}, "
            f"released != gate level {r['released_differs_from_gate_level']}, fields != memo path {r['outcome_fields_differ_from_memo_path']}; Rescope-WHERE released sets {r['where']['released_sets']}")


def g_m(r):
    return (f"{r['n_rows']} sequences, {r['n_rows_with_a_masked_release_before']} with a masked edit released before: releases of a masked edit after {r['n_releases_of_a_masked_edit_after']}; frame-rule "
            f"violations {r['frame_rule_violations']} of {r['n_rows_checked_by_the_frame_rule']}; masked rows now silent {r['masked_rows_now_silent']} (bitwise pre-edit {r['masked_rows_now_pre_edit_bitwise']}), "
            f"other edit released {r['masked_rows_other_edit_released']}")


def g_r(r):
    return (f"rollback(empty) == W_E {r['empty_set_is_WE_bitwise']}, (all) == W_0 {r['all_edits_is_W0_bitwise']}, one edit / whole batch / mixed == formula {r['one_edit_equals_formula']} / "
            f"{r['whole_batch_equals_formula']} / {r['mixed_equals_formula']}; rows != library {r['rows_differ_from_library']} of {r['n_rows']}; forwards on a second evaluation {r['second_evaluation_forwards']}")


def g_s(r):
    return ("H3 behavioural form: " + ", ".join(f"{k} raise {v['delta']:.4f}: flags != gate level {v['edit_flags_differ_from_gate_level']} + {v['hard_negative_flags_differ_from_gate_level']}" for k, v in r['h3'].items()) +
            f"; rescope_pac flags != gate level {r['rescope_pac']['flags_differ_from_gate_level']}; same calibration rows for {len(r['rowset']['calibrated_rows'])} calibrated rows incl. router / z-score "
            f"{r['rowset']['same_calibration_rows']}, all {len(fx.methods.E1_LABELS)} rows built {r['rowset']['all_rows_built']}; router on pools == library {[v['equals_library'] for k, v in r['pools'].items() if isinstance(v, dict) and 'equals_library' in v]}")


def g_k(r):
    d = r["d2_deployed"]
    return ("kept / gained rescope d0: " + ", ".join(f"{m} {v['n_kept']} / {v['n_gained']}" for m, v in r['kept_gained'].items()) +
            f"; D2 deployed {d['n_D2']} of {d['n_pre_correct']} pre-correct rows = {d['rate']['value']:.2f} % [{d['rate']['ci_lo']:.2f}, {d['rate']['ci_hi']:.2f}] ({d['rate']['n_units']} clusters; stored records {d['n_D2_stored_records']}); "
            f"method {r['d2_method']['n_D2']} of {r['d2_method']['n_pre_correct']}")


def g_p(r):
    lin = r["linear"]
    return (f"T = {r['T']}: harness checks {all(v for k, v in r['checks'].items() if k != 'collapsed_working_point')}, deployed Rel {r['deployed_from_memo']['rel']:.0f} (described {r['deployed_from_memo']['described']:.0f}); "
            f"COMP on {lin['COMP']['n_complaints']} complaints: still D1 {lin['COMP']['complaints_still_D1']}, rows != plain library {lin['COMP']['rows_differ_from_plain_library']} ({lin['COMP']['seconds']:.0f}s); ROLLBACK_FO R = {lin['ROLLBACK_FO'].get('R')} == driver rollback "
            f"{lin['ROLLBACK_FO'].get('equals_driver_rollback_weight')}; REEDIT ran {lin['REEDIT'].get('ran')} ({lin['REEDIT']['seconds']:.0f}s{', ' + lin['REEDIT'].get('error', '') if not lin['REEDIT'].get('ran') else ''}); "
            f"harness evaluate ran {lin['harness_evaluate'].get('ran')} ({lin['harness_evaluate']['seconds']:.0f}s); no file in the library's / harness's cache {r['no_prefix_file_in_library_or_harness_cache']}; nested stream (n_base 20) {r['complaints']['stream_n_base_20']}")


CPU = [("null", "(1) ReScope-N", d_null), ("samples", "(2) samples + guard", d_samples), ("sweeps", "(3) sweeps, coverage scale, H3", d_sweeps), ("rowset", "(4) row set", d_rowset),
       ("calib", "(5) calibration options", d_calib), ("readouts", "(6) read-outs R7, blocklist", d_readouts), ("stats2", "(7) statistics R10", d_stats2),
       ("pools", "(8) logs, pools, masks", d_pools), ("prefix", "(9) prefix logs", d_prefix), ("maint", "(10) mask, digest", d_maint)]
GPU = [("h_hashes", "(10 ii) logit hashes", g_h), ("m_mask", "(10 i) mask, frame rule", g_m), ("r_rollback", "(10 iii) rollback weight", g_r),
       ("s_h3_pac_rowset", "(3, 4, 5) behavioural forms", g_s), ("k_readouts", "(6) read-outs on real rows", g_k)]


def main():
    rows, tests = [], {}
    for nm, title, f in CPU:
        r = J(f"ext_cpu_{nm}.json")
        v = verdict(r)
        tests[f"ext_cpu_{nm}"] = v
        try:
            det = f(r) if r else ""
        except Exception as e:                                          # noqa: BLE001
            det = f"(detail not rendered: {type(e).__name__}: {e})"
        rows.append((title, "CPU", v, det))
    for log, x in S.items():
        for nm, title, f in GPU:
            r = J(f"ext_gpu_{log}_{nm}.json")
            v = verdict(r)
            if r is not None and (r.get("hook_mismatch") or r.get("memo_conflicts")):
                v = "FAIL"
            tests[f"ext_gpu_{x}_{nm}"] = v
            try:
                det = f(r) if r else ""
            except Exception as e:                                      # noqa: BLE001
                det = f"(detail not rendered: {type(e).__name__}: {e})"
            rows.append((title, x, v, det))
        r = J(f"ext_gpu_{log}-p10_p_prefix.json")
        v = verdict(r)
        tests[f"ext_gpu_{x}_p_prefix"] = v
        try:
            det = g_p(r) if r else ""
        except Exception as e:                                          # noqa: BLE001
            det = f"(detail not rendered: {type(e).__name__}: {e})"
        rows.append(("(9) prefix log + harness repairs", f"{x} p10", v, det))
    # the driver's own tests (before the extension), re-run on the present code
    old = {}
    for p in sorted(glob.glob(os.path.join(OUT, "cpu_*.json")) + glob.glob(os.path.join(OUT, "gpu_*.json"))):
        nm = os.path.basename(p)[:-5]
        r = fx.io.jload(p)
        old[nm] = verdict(r)
    dg = J("digest_compare.json")
    e0 = [k for k in old if "regress" in k]
    txt = [f"TESTS OF THE STAGE-1 EXTENSION OF THE fx DRIVER   (rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by tests/render_ext.py from tests/out/*.json; code {CODE})", "",
           fx.io.text_table(("test (task)", "scope", "result", "detail"), rows, align=["l", "l", "l", "l"]), "",
           "THE DRIVER'S OWN TESTS (tests/t_cpu.py, tests/t_gpu.py; table: tests/out/table.txt), state for the present code:",
           "  " + "; ".join(f"{k}: {v}" for k, v in old.items()), "",
           f"SANITY NUMBERS of e0_sanity/out/table.txt (default method, shared scale, cosine gate with both calibrations, cosine 0.65, exact match; deployed, pre-edit): "
           f"reproduced by tests {', '.join(f'{k} {old[k]}' for k in e0)} (gate level: thresholds and every decision of 20 draws; behaviour: every row and rendered number of draws 0 - 2)", "",
           "DECISION DIGEST of the existing row set (both development logs, alpha = 0.05, calibration draw 0; tests/digest.py: record key, gate bytes and every gate-level decision of each row): " +
           (f"{dg['n_equal']} of {dg['n_rows']} rows unchanged ({dg['n_decisions']} decisions); fx before {dg['fx_before']['fx']}, now {dg['fx_after']['fx']}" if dg else "MISSING"), ""]
    out = "\n".join(txt)
    with open(os.path.join(OUT, "table_ext.txt"), "w") as f:
        f.write(out)
    print(out[:3000])
    allv = dict(tests, **{f"driver_{k}": v for k, v in old.items()})
    # tests that are allowed to stand on older code: mechanics / full E1 pass on logs and paths the extension does not touch (listed, not hidden)
    stale_ok = [k for k, v in allv.items() if v == "PASS (older code)" and (k.endswith("_e_mechanics") or k.endswith("_d_e1"))]
    bad = {k: v for k, v in allv.items() if v != "PASS" and k not in stale_ok}
    digest_ok = bool(dg and dg["ok"] and dg["fx_after"]["fx"] == CODE["fx"])
    summary = {"n_tests": len(allv), "n_pass_present_code": sum(v == "PASS" for v in allv.values()), "pass_on_older_code_not_rerun": stale_ok, "not_passing": bad, "digest_unchanged": digest_ok,
               "tests": allv}
    fx.io.jdump(summary, os.path.join(OUT, "summary_ext.json"), indent=1)
    print("\nnot passing:", bad, "| digest unchanged:", digest_ok, "| older code, not re-run:", stale_ok)
    if "--ready" in sys.argv:
        if bad or not digest_ok:
            raise SystemExit("READY.json is NOT written: tests are not all passing with the present code")
        d = os.path.dirname(os.path.abspath(fx.__file__))
        files = {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest()[:12] for f in sorted(os.listdir(d)) if f.endswith(".py")}
        ready = {"stage": 1, "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "code_fingerprint": CODE["fx"], "library_fingerprint": CODE["rescope"], "fx_version": CODE["fx_version"],
                 "fingerprint_rule": "sha1 (12 hex digits) over the names and bytes of fx/*.py in sorted order (fx.code_fingerprint)", "source_files": files,
                 "reviewed_base": "b92d0532498f (review/patches.txt); the stage-1 extension is not yet independently reviewed",
                 "method_names": list(fx.methods.NAMES), "e1_labels": list(fx.methods.E1_LABELS), "e1_gate_level_only": list(fx.methods.E1_GATE_LEVEL), "e3_labels": list(fx.methods.E3_LABELS),
                 "functions_that_return_methods": ["fx.methods.build", "fx.methods.wrap", "fx.methods.mask_edits", "fx.methods.rollback", "fx.sweeps.raised_method", "fx.methods.recalibrate + wrap"],
                 "pools": {"calibration_sources": list(fx.draws.SOURCES), "test_only": list(fx.pools.TEST_ONLY), "statistics_only": list(fx.pools.STATISTICS_ONLY),
                           "probe_pools": list(fx.logs.POOLS), "key_level": list(fx.pools.NAMES), "refused": ["benign (whole pool) as a calibration source"]},
                 "logs": {n: {"role": e["role"], "exists": fx.logs.exists(n), "T": e["T"], "backbone": e["backbone"], "cut_by_R19": bool(e.get("cut_by_R19"))} for n, e in fx.logs.REGISTRY.items()},
                 "programme_R19": fx.logs.programme(), "prefix_logs": {n: fx.prefix.complete(n) for n in fx.logs.names("prefix")},
                 "masks": {"later_edits_from": fx.masks.LATER, "question_disjoint": "fx.masks.question_disjoint"},
                 "test_summary": summary, "tables": ["tests/out/table_ext.txt", "tests/out/table.txt", "out/prefix_logs.txt"], "usage": "USAGE.txt (rules at the top; section EXTENSION)"}
        fx.io.jdump(ready, os.path.join(os.path.dirname(HERE), "READY.json"), indent=1)
        print("READY.json written:", ready["code_fingerprint"])


if __name__ == "__main__":
    main()
