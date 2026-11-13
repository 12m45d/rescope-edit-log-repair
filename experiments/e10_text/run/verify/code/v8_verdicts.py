"""E10 verification: verdicts, flat records, table and quoted numbers, assembled from the JSON files of the verifier's scripts.

    python v8_verdicts.py  ->  verify/verdicts.json, out/verdicts.txt (= out/table.txt), out/results.json, out/paper_numbers.json
Every number in the evidence lines is read from a JSON file written by code (out/compare.json, out/cells.json, out/discipline.json,
out/screening_T2000.json, out/r22/*.json, out/behave/*, out/equal/*, out/cert/*, out/boot_test.json) or aggregated here from the
experiment's per-row files; none is typed.  A check PASSES if the verifier's independent recomputation reproduces the experiment's
numbers (row by row where rows exist) or the discipline rule holds; it FAILS if a number is not reproduced or a statement of the
report does not hold as worded; the evidence says how far.
"""
import os

import numpy as np

import vcommon as V

O = V.OUT
X = V.XOUT
cmp_ = V.jload(O + "/compare.json")
cells = V.jload(O + "/cells.json")
disc = V.jload(O + "/discipline.json")
scr = V.jload(O + "/screening_T2000.json")
r22 = V.jload(O + "/r22/" + V.T100 + ".json")
boot = V.jload(O + "/boot_test.json")
dev = V.jload(O + "/devgate.json")
checks, risks, PN, REC = [], [], {}, []


def pn(name, value, file, key):
    PN[name] = {"value": value, "file": file, "key": key}
    return value


def rec(log, method, metric, value, variant="", alpha=None, n=None, draw=None, pool="", ci_lo=None, ci_hi=None, n_units=None):
    REC.append({"experiment": "E10-verify", "log": log, "method": method, "variant": variant, "alpha": alpha, "n": n, "draw": draw, "pool": pool, "metric": metric,
                "value": value, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_units": n_units})


def add(name, ok, evidence, severity=None, **extra):
    checks.append(dict({"check": name, "verdict": "pass" if ok else "fail", "evidence": evidence}, **({"severity": severity} if severity else {}), **extra))


def m100(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return 100.0 * float(np.mean(v)) if v else None


# ----------------------------------------------------------------------------- 1, 2: behaviour
for log, name in ((V.T100, "behaviour_T100"), (V.T500, "behaviour_T500_guards_on_off_split_173")):
    b = cmp_["behaviour"][log]
    nrows = sum(o["n_rows_mine"] for o in b.values())
    ncomp = sum(v["n_compared"] for o in b.values() for v in o["row_level_differences"].values())
    ndiff = sum(v["n_differ"] for o in b.values() for v in o["row_level_differences"].values())
    maxagg = max(o["max_abs_diff_same_rows"] for o in b.values())
    recs = cmp_["records"]
    same_gate = all(v["same_gate"] for k, v in recs.items() if k.startswith(log))
    sil = {lab: o["silent_prompts"] for lab, o in b.items() if o["silent_prompts"].get("silent_tf.n") is not None}
    sil_bad = sum(v["silent_tf.n_D1"] + v["silent_dec.n_D1"] for v in sil.values())
    smp = V.jload(O + f"/behave/{log}/_sample.json")
    ev = (f"{len(b)} rows of the table rebuilt from the log with the adapter at calibration draw 0 ({', '.join(b)}); sample: {smp['n_logged_records']} logged records "
          f"({smp['n_edit_rows']} request / paraphrase and {smp['n_neighborhood_rows']} neighbourhood prompts) and {smp['n_heldout_records']} HELD-OUT records ({smp['n_heldout_rows']} prompts); "
          f"every prompt through the model (no sparse shortcut, no store).  Row by row against the experiment's per-row files: {ndiff} differences in {ncomp} compared values "
          f"(D1 tf / decode, exact match, firing, own edit released, first token); largest difference of an aggregate on the same rows {maxagg:.3g} points; "
          f"thresholds of the rebuilt gates identical to the experiment's (sha1): {same_gate}; prompts on which the hook released nothing but a token changed: {sil_bad}.")
    for lab, o in b.items():
        for met in ("edit.request.exact", "edit.paraphrase.exact", "heldout.D1_tf", "neighborhood.D1_tf", "heldout.fired") + (("edit.request.exact.before", "edit.request.exact.from") if log == V.T500 else ()):
            e = o["aggregates"].get(met)
            if e and e["mine_on_sample"] is not None:
                ci = o["my_stats_with_ci"].get(met, {})
                cal = lab not in ("deployed", "preedit")
                rec(log, lab.split("@")[0], met, e["mine_on_sample"], variant="guards off" if lab.endswith("@goff") else "", alpha=V.ALPHA if cal else None, n=V.N_CAL if cal else None,
                    draw=0 if cal else None, pool=met.split(".")[0], ci_lo=ci.get("ci_lo"), ci_hi=ci.get("ci_hi"), n_units=ci.get("n_units"))
    ex = {lab: {m: [round(o["aggregates"][m]["mine_on_sample"], 2), round(o["aggregates"][m]["experiment_on_same_rows"], 2)] for m in ("edit.request.exact", "edit.paraphrase.exact", "heldout.D1_tf", "neighborhood.D1_tf")
                + (("heldout.fired",) if "heldout.fired" in o["aggregates"] else ()) + (("edit.request.exact.before", "edit.request.exact.from") if log == V.T500 else ())} for lab, o in b.items()}
    add(name, ndiff == 0 and maxagg < 1e-9 and same_gate and sil_bad == 0, ev, verifier_vs_experiment_on_the_same_rows=ex)
    pn(f"Verify{log[-4:]}BehaviourRowDifferences", ndiff, "out/compare.json", f"behaviour.{log}.*.row_level_differences.*.n_differ (sum)")
    pn(f"Verify{log[-4:]}BehaviourValuesCompared", ncomp, "out/compare.json", f"behaviour.{log}.*.row_level_differences.*.n_compared (sum)")

# ----------------------------------------------------------------------------- 3: public scores
p = cmp_["public"]
nrec = sum(o["n_records"] for d in p.values() for o in d.values())
maxd = max(o["max_abs_probability_difference"] for d in p.values() for o in d.values())
flips = sum(o["n_comparisons_flipped"] for d in p.values() for o in d.values())
agree = all(o["agree"] for d in p.values() for o in d.values())
lst = "; ".join(f"{log[-4:]} {lab}: ES/PS/NS {o['public_summariser_on_my_files']['ES']}/{o['public_summariser_on_my_files']['PS']}/{o['public_summariser_on_my_files']['NS']} on {o['n_records']} records"
                for log, d in p.items() for lab, o in d.items())
add("public_scores_deployed_and_repair", maxd == 0.0 and flips == 0 and agree,
    f"the public compute_rewrite_quality_counterfact on the served model and the public summarize.py, run by the verifier ({lst}); {nrec} case files compared with the experiment's: "
    f"largest difference of a probability {maxd}, comparisons flipped {flips}; public summariser on the verifier's files = the experiment's files on the same records: {agree} "
    f"(T100: all 100 records of the log; T500: 300 random records, guards on and off).")
for log, d in p.items():
    for lab, o in d.items():
        for k in ("ES", "PS", "NS"):
            rec(log, lab.split("@")[0], "public." + k, o["public_summariser_on_my_files"][k], variant="guards off" if lab.endswith("@goff") else "", draw=0 if lab != "deployed" else None,
                alpha=V.ALPHA if lab != "deployed" else None, n=V.N_CAL if lab != "deployed" else None, pool="records", n_units=o["n_records"])
pn("VerifyPublicMaxProbabilityDifference", maxd, "out/compare.json", "public.*.*.max_abs_probability_difference (max)")

# ----------------------------------------------------------------------------- 4: certificate
c = cmp_["certificate"]
nchk = sum(r["n_checked"] for d in c.values() for r in d["rows"].values())
nbit = sum(r["n_bit_identical"] for d in c.values() for r in d["rows"].values())
nvio = sum(r["n_violations"] for d in c.values() for r in d["rows"].values())
ncert_same = all(r["n_certified"][0] == r["n_certified"][1] for d in c.values() for r in d["rows"].values())
tot_ok = all(d["experiment_total_recomputed"]["n_checked_certified"] == d["experiment_total"]["n_checked_certified"] and d["experiment_total_recomputed"]["n_violations"] == 0 for d in c.values())
zero = {log[-4:]: [k for k, r in d["rows"].items() if r["n_certified"][0] == 0] for log, d in c.items()}
add("certificate_200_certified_prompts_per_row", nvio == 0 and nbit == nchk and ncert_same and tot_ok,
    f"{nchk} certified HELD-OUT prompts (200 random per row with certified prompts; repair and migration at draw 0; T500 guards off) decoded under the row and under the pre-edit model in the verifier's "
    f"batches: {nbit} bit-identical logits, {nvio} violations; number of certified prompts per row equal to the experiment's: {ncert_same}; rows that certify nothing (guards on, T500): {zero}; "
    f"the experiment's totals recomputed from its per-row counts: T100 {c[V.T100]['experiment_total_recomputed']['n_checked_certified']}, T500 {c[V.T500]['experiment_total_recomputed']['n_checked_certified']}, violations 0: {tot_ok}.")
pn("VerifyCertificateChecked", nchk, "out/compare.json", "certificate.*.rows.*.n_checked (sum)")
pn("VerifyCertificateViolations", nvio, "out/compare.json", "certificate.*.rows.*.n_violations (sum)")
for log, d in c.items():
    for lab, r in d["rows"].items():
        rec(log, lab.split("@")[0], "certificate.n_bit_identical", r["n_bit_identical"], variant="guards off" if lab.endswith("@goff") else "", alpha=V.ALPHA, n=V.N_CAL, draw=0, pool="heldout", n_units=r["n_checked"])

# ----------------------------------------------------------------------------- 5: equal evidence
e = cmp_["equal"]
co = e["complaints"]
rb, ls, lp = e["rb_suffix"], e["least_sacrificing"], e["least_protecting"]
nd = {k: sum(v["n_differ"] for v in o["row_level_differences"].values()) for k, o in (("rb_suffix", rb), ("least_sacrificing", ls), ("least_protecting", lp))}
ok_rb = (rb["my_meta"]["t_star"] == rb["experiment_repair"]["t_star"] and rb["my_meta"]["n_forward"] == rb["experiment_repair"]["n_forward"] and rb["my_meta"]["edits_lost"] == rb["experiment_meta"]["edits_lost"]["n"]
         and nd["rb_suffix"] == 0)
ok_ls = all(o["my_meta"]["n_thresholds_raised"] == o["experiment_repair"]["n_thresholds_raised"] and o["my_meta"]["complaints_refused"] == o["experiment_meta"]["complaints_refused"]
            and abs(o["aggregates"]["request.exact"]["mine"] - o["aggregates"]["request.exact"]["experiment_seed0_from_its_rows"]) < 1e-9
            and abs(o["aggregates"]["heldout.D1_tf"]["mine_on_sample"] - o["aggregates"]["heldout.D1_tf"]["experiment_seed0_same_rows"]) < 1e-9 for o in (ls, lp))
add("equal_evidence_cell_T100_n10_seed0", bool(co["same_complaints"] and co["experiment_complaints_are_D1_under_deployed_by_my_pass"] and ok_rb and ok_ls),
    f"complaints: the verifier's pass of the deployed log over the {co['n_repair_prompts']} REPAIR-pool prompts gives {co['n_D1_complaints_under_deployed']} D1 complaints (experiment {co['experiment_n_D1_complaints']}); "
    f"the 10 complaints of seed 0 are the experiment's: {co['same_complaints']}.  rb_suffix: longest resolving prefix {rb['my_meta']['t_star']} (experiment {rb['experiment_repair']['t_star']}), "
    f"{rb['my_meta']['n_forward']} forward passes ({rb['experiment_repair']['n_forward']}), {rb['my_meta']['edits_lost']} edits lost ({rb['experiment_meta']['edits_lost']['n']}), request exact "
    f"{rb['aggregates']['request.exact']['mine']:.1f} ({rb['aggregates']['request.exact']['experiment_seed0_from_its_rows']:.1f}), D1 HELD-OUT on the same 900 prompts {rb['aggregates']['heldout.D1_tf']['mine_on_sample']:.2f} "
    f"({rb['aggregates']['heldout.D1_tf']['experiment_seed0_same_rows']:.2f}); rows differing {len(rb['rows_with_a_difference'])} of {rb['n_rows_compared']}.  Least repair (sacrificing / protecting): thresholds raised "
    f"{ls['my_meta']['n_thresholds_raised']} / {lp['my_meta']['n_thresholds_raised']} (experiment {ls['experiment_repair']['n_thresholds_raised']} / {lp['experiment_repair']['n_thresholds_raised']}), refused "
    f"{ls['my_meta']['complaints_refused']} / {lp['my_meta']['complaints_refused']}, request exact {ls['aggregates']['request.exact']['mine']:.1f} ({ls['aggregates']['request.exact']['experiment_seed0_from_its_rows']:.1f}), "
    f"D1 HELD-OUT on the same prompts {ls['aggregates']['heldout.D1_tf']['mine_on_sample']:.2f} ({ls['aggregates']['heldout.D1_tf']['experiment_seed0_same_rows']:.2f}; all 3000 prompts of seed 0: "
    f"{ls['aggregates']['heldout.D1_tf']['experiment_seed0_all_rows_from_its_rows']:.2f}; table, mean of 3 seeds: {ls['aggregates']['heldout.D1_tf']['experiment_summary_mean_over_seeds']:.2f}); rows differing "
    f"{len(ls['rows_with_a_difference'])} of {ls['n_rows_compared']} ({', '.join(ls['rows_with_a_difference'])}): this reported complaint, whose coordinate lies {abs(ls['my_meta']['largest_complaint_margin_over_threshold_after_repair']):.1e} "
    f"below its repaired threshold on the cached keys still releases an edit as served, in the teacher-forced pass of the verifier's batches (complaints still D1: tf {ls['aggregates']['complaints.D1_tf']['mine']:.0f} %, "
    f"decode {ls['aggregates']['complaints.D1_dec']['mine']:.0f} %) and in the decode of the experiment's batches (tf {ls['aggregates']['complaints.D1_tf']['experiment_seed0']:.0f} %, decode "
    f"{ls['aggregates']['complaints.D1_dec']['experiment_seed0']:.0f} %).")
for nm, o in (("rb_suffix", rb), ("least_sacrificing", ls), ("least_protecting", lp)):
    rec(V.T100, nm, "equal.request.exact", o["aggregates"]["request.exact"]["mine"], variant="n=10,seed=0", pool="edits", n_units=100)
    rec(V.T100, nm, "equal.heldout.D1_tf", o["aggregates"]["heldout.D1_tf"]["mine_on_sample"], variant="n=10,seed=0", pool="heldout", ci_lo=o["aggregates"]["heldout.D1_tf"]["mine_ci"][0],
        ci_hi=o["aggregates"]["heldout.D1_tf"]["mine_ci"][1], n_units=300)
pn("VerifyEqualRbSuffixTStar", rb["my_meta"]["t_star"], "out/equal/TX-G2XL-rome-T100/rb_suffix__n10_s0.json", "meta.t_star")
pn("VerifyEqualLeastHeldoutDOne", ls["aggregates"]["heldout.D1_tf"]["mine_on_sample"], "out/equal/TX-G2XL-rome-T100/least_sacrificing__n10_s0.json", "stats.heldout.D1_tf.value")

# ----------------------------------------------------------------------------- 6: relational workload
pool, dg, cp = r22["pool"], r22["diagnosis"], r22["compare"]
ok = (pool["same_records_as_experiment"] and pool["same_report_half"] and pool["same_test_half"] and pool["pools_of_the_build_reproduced"] and dg["same_as_experiment"]
      and all(cp[k]["scores_equal"] and cp[k]["per_tuple"]["n_values_differ"] == 0 for k in ("deployed", "restore-base", "least_sacrificing", "least_protecting")))
row = r22["rows"]
st = r22["states"]
add("relational_workload_R22_T100", bool(ok),
    f"verifier's own VIEW pool ({pool['n_view']} records, {pool['n_excluded_shared_subject']} excluded for a shared subject) equals the experiment's: {pool['same_records_as_experiment']}; three states as "
    f"sqlite tables ({', '.join(str(s['name']) + ' ' + str(s['n']) for s in st['sizes'])}); deployed differs from the pre-edit state on {st['deployed_vs_R0_unedited_test']['n_differ']} of "
    f"{st['deployed_vs_R0_unedited_test']['n']} unedited test tuples and from the intended state on {st['deployed_vs_intended_edited']['n_differ']} of {st['deployed_vs_intended_edited']['n']} edited tuples; "
    f"{st['queries_test']} relations x (view, aggregate) on the test half.  Verifier's SQL (code/v3_r22.sql), reference preservation / retention / queries exact / TV: deployed "
    f"{row['deployed']['scores']['reference_preservation']['pct']:.2f} / {row['deployed']['scores']['retention']['pct']:.1f} / {row['deployed']['scores']['queries_exact_pct']:.2f} / {row['deployed']['scores']['aggregate_tv_mean']:.4f}; "
    f"restore-base {row['restore-base']['scores']['reference_preservation']['pct']:.2f} / {row['restore-base']['scores']['retention']['pct']:.1f} / {row['restore-base']['scores']['queries_exact_pct']:.2f} / "
    f"{row['restore-base']['scores']['aggregate_tv_mean']:.4f}; least threshold repair, m = 3, seed 0 (both forms) {row['least_sacrificing']['scores']['reference_preservation']['pct']:.2f} / "
    f"{row['least_sacrificing']['scores']['retention']['pct']:.1f} / {row['least_sacrificing']['scores']['queries_exact_pct']:.2f} / {row['least_sacrificing']['scores']['aggregate_tv_mean']:.4f}, "
    f"{row['least_sacrificing']['n_thresholds_raised']} thresholds raised, {row['least_sacrificing']['complaints_refused']} refused, {row['least_sacrificing']['serve']['edits_lost']} edits lost; charged passes: diagnosis "
    f"{dg['diagnosis_passes']} (2 x {dg['n_tuples_in_complained_views']} tuples of the 3 complained views; {dg['n_own_requests_in_views']} own requests recognised from the solver keys; {dg['n_tuple_complaints']} "
    f"tuple complaints), repair {row['least_sacrificing']['repair_passes']}.  All equal to the experiment's rows: {ok}; per-tuple values differing: "
    f"{sum(cp[k]['per_tuple']['n_values_differ'] for k in cp)} of {sum(cp[k]['per_tuple']['n_common'] for k in cp)}; complaints resolved as served {row['least_sacrificing']['complaints_resolved_as_served']} of {dg['n_tuple_complaints']}.")
for k in ("deployed", "restore-base", "least_sacrificing", "least_protecting"):
    s = row[k]["scores"]
    v = "m=3,seed=0" if k.startswith("least") else ""
    rec(V.T100, k, "r22.reference_preservation", s["reference_preservation"]["pct"], variant=v, pool="test half", n_units=s["reference_preservation"]["n"])
    rec(V.T100, k, "r22.retention", s["retention"]["pct"], variant=v, pool="edited tuples", n_units=s["retention"]["n"])
    rec(V.T100, k, "r22.queries_exact_pct", s["queries_exact_pct"], variant=v, pool="test half", n_units=s["n_views"] + s["n_aggregates"])
    rec(V.T100, k, "r22.aggregate_tv_mean", s["aggregate_tv_mean"], variant=v, pool="test half", n_units=s["n_aggregates"])
    rec(V.T100, k, "r22.cost.diagnosis_passes", row[k]["diagnosis_passes"], variant=v, pool="cost")
pn("VerifyRelDiagnosisPasses", dg["diagnosis_passes"], "out/r22/TX-G2XL-rome-T100.json", "diagnosis.diagnosis_passes")
pn("VerifyRelLeastRef", row["least_sacrificing"]["scores"]["reference_preservation"]["pct"], "out/r22/TX-G2XL-rome-T100.json", "rows.least_sacrificing.scores.reference_preservation.pct")

# ----------------------------------------------------------------------------- 7: screening
s2 = scr["logs"][V.T2000]
ti = scr["ties_T2000"]
small = all(scr["logs"][lg]["experiment"]["same_flagged_set"] and scr["logs"][lg]["experiment"]["same_precision_recall"] for lg in (V.T100, V.T500, "TX-G2XL-rome-T1000"))
ok = s2["experiment"]["same_flagged_set"] and s2["experiment"]["same_precision_recall"]
add("screening_T2000_self_separation_margins", bool(ok),
    f"margins recomputed from the public factors (read-out vector and solver key of every update; deviation from the library's normalised log {scr['max_rel_deviation_from_library_log']:.1e}).  "
    f"T2000: the verifier flags {s2['n_flagged']} edits, the experiment {s2['experiment']['n_flagged']}; recall identical ({s2['against']['exploding_norm']['recall']:.1f} % of the audit's "
    f"{len(s2['against']['exploding_norm']['reference'])} exploding updates, {s2['against']['first_token_subject']['recall']:.1f} % of the {len(s2['against']['first_token_subject']['reference'])} first-token subjects); "
    f"precision {s2['against']['exploding_norm']['precision']:.1f} / {s2['against']['first_token_subject']['precision']:.1f} % against the experiment's "
    f"{s2['experiment']['precision_recall']['exploding_norm'][0]:.1f} / {s2['experiment']['precision_recall']['first_token_subject'][0]:.1f} %.  Cause: {ti['n_tie_edits']} edits in {len(ti['groups_of_identical_solver_keys'])} "
    f"pairs of requests with bitwise identical solver keys (repeated request prompts) have margin 0 in exact arithmetic (|margin| <= {ti['max_abs_margin_of_tie_edits']:.1e}); the rule 'margin <= 0' flags "
    f"{ti['n_tie_edits_flagged_by_experiment']} of them in the experiment's arithmetic and {ti['n_tie_edits_flagged_by_verifier']} in the verifier's; the two flag sets differ only on these edits: "
    f"{ti['flag_sets_differ_only_on_tie_edits']}.  Without ties {ti['n_flagged_with_margin_below_minus_1e-9']} edits are flagged, with all ties {ti['n_flagged_if_every_tie_is_flagged']}: precision between "
    f"{ti['precision_range_exploding'][0]:.1f} and {ti['precision_range_exploding'][1]:.1f} % (exploding) and between {ti['precision_range_first_token'][0]:.1f} and {ti['precision_range_first_token'][1]:.1f} % (first token).  "
    f"T100, T500, T1000: flag sets, precision and recall identical to the experiment's: {small} (T1000 contains {len(ti['tie_groups_inside_T1000'])} tie pair, T500 {len(ti['tie_groups_inside_T500'])}).",
    severity="minor: the count of flagged edits and the precision on T2000 (and, through one pair, on T1000) depend on the last bit; recall and the T500 numbers quoted in the report do not")
for lg, o in scr["logs"].items():
    rec(lg, "screening", "screen.n_flagged", o["n_flagged"], pool="log")
    for nm in ("exploding_norm", "first_token_subject"):
        rec(lg, "screening", f"screen.{nm}.precision", o["against"][nm]["precision"], pool="log", n_units=o["n_flagged"])
        rec(lg, "screening", f"screen.{nm}.recall", o["against"][nm]["recall"], pool="log", n_units=len(o["against"][nm]["reference"]))
pn("VerifyScreenT2000Flagged", s2["n_flagged"], "out/screening_T2000.json", "logs.TX-G2XL-rome-T2000.n_flagged")
pn("VerifyScreenT2000TieEdits", ti["n_tie_edits"], "out/screening_T2000.json", "ties_T2000.n_tie_edits")

# ----------------------------------------------------------------------------- 8-11: discipline
a = disc["audited_logs"]
add("discipline_only_audited_logs", bool(a["all_safe"]),
    f"logs named in the experiment's outputs: {', '.join(a['logs_named_in_the_experiment_outputs'])}; verdicts of the independent audit (09_testbed_scale/text_public/audit/verdicts.json): "
    f"{', '.join(k[len('TX-G2XL-rome-'):] + ' ' + str(v) for k, v in a['audit_verdicts'].items())}; the {len(a['prefix_logs_of_the_stream'])} prefix logs of the stream are bit-for-bit replays of the audited T2000 log.")
pl = disc["pools"]
br = pl["behaviour_records"]
ok = (br["n_calibrated_not_from_REPAIR_pool"] == 0 and br["n_calibrated_with_other_alpha_or_n"] == 0 and pl["r3_statistics"]["same"] and all(v["n_not_in_REPAIR_pool"] == 0 for v in pl["equal_complaints"].values())
      and all(v["heldout_rows_not_in_HELDOUT_pool"] == 0 and v["heldout_rows_in_REPAIR_pool"] == 0 and v["neighborhood_rows_not_of_logged_requests"] == 0 and v["near_scope_reported_separately_in_summary"] for v in pl["rows"].values())
      and all(v["calibration_in_REPAIR_pool"] for v in cmp_["records"].values()) and r22["diagnosis"]["complaints_all_in_report_half_and_unedited"])
add("discipline_pools", bool(ok),
    f"{br['n_calibrated']} calibrated method records (behaviour, stream): calibration prompts outside the REPAIR pool {br['n_calibrated_not_from_REPAIR_pool']}, other alpha or n {br['n_calibrated_with_other_alpha_or_n']} "
    f"(alpha {br['alphas']}, n {br['n_values']}, draws {br['draws']}); the six gates rebuilt by the verifier from draw 0 of the REPAIR pool have the experiment's thresholds; R3 statistics: the 500 prompts recomputed from the rule "
    f"(REPAIR prompts in none of the 20 draws; {pl['r3_statistics']['n_repair_prompts_in_no_draw']} available) have the sha1 of the experiment's gate files: {pl['r3_statistics']['same']}; equal-evidence complaints outside the REPAIR pool: "
    f"{sum(v['n_not_in_REPAIR_pool'] for v in pl['equal_complaints'].values())} of {sum(v['n_complaints'] for v in pl['equal_complaints'].values())}; locality rows: "
    f"{', '.join(k[-4:] + ' ' + str(v['n_heldout_rows']) + ' HELD-OUT (' + str(v['heldout_rows_not_in_HELDOUT_pool']) + ' outside the pool) and ' + str(v['n_neighborhood_rows']) + ' neighbourhood' for k, v in pl['rows'].items())}; "
    f"near-scope prompts flagged and reported separately ({', '.join(k[-4:] + ' ' + str(v['n_heldout_near_scope']) + ' HELD-OUT / ' + str(v['n_neighborhood_near_scope']) + ' neighbourhood' for k, v in pl['rows'].items())}); "
    f"R22 tuple complaints of the recomputed cell all from the report half of the VIEW pool: {r22['diagnosis']['complaints_all_in_report_half_and_unedited']}.")
t = disc["nothing_tuned"]
ok = (t["best_gate"]["same"] and t["best_gate"]["written_before_any_outcome_on_a_reported_log"] and t["split"]["is_173"] and t["logs"]["n_with_more_than_one_queue_entry"] == 0 and t["logs"]["n_nonzero_exit"] == 0
      and t["store"]["n_unmatched"] == 0 and t["behaviour_instances"]["T100_as_planned"] and t["behaviour_instances"]["T500_as_planned"] and not t["code_constants"]["files_that_mention_another_alpha"]
      and dev["all_equal"])
dmax = max(v["max_abs_difference_points"] for k, o in dev.items() if isinstance(o, dict) for v in o.values())
mc = t["method_and_behaviour_code_last_modified"]
late = t["code_files_modified_after_the_first_outcome"]
add("discipline_nothing_tuned_on_a_reported_log", bool(ok),
    f"R3 gate: argmax of the paraphrase coverage on the development log recomputed from its gate-level file = {t['best_gate']['recomputed_argmax_paraphrase_coverage_on_dev']} = selected; file written {t['best_gate']['file_written']} UTC, "
    f"first outcome of a reported log {t['best_gate']['first_outcome_shard_of_the_store']} (store), first gate-level file of a reported log {t['best_gate']['first_gate_level_file_of_a_reported_log']}; split {t['split']['split_position']} "
    f"(fixed); behavioural instances exactly the registered plan (T100 {t['behaviour_instances']['n_T100']}, T500 {t['behaviour_instances']['n_T500']}; extra: {t['behaviour_instances']['extra_T100'] + t['behaviour_instances']['extra_T500']}); "
    f"{t['logs']['n_files']} job logs, each with one queue entry, {t['logs']['n_nonzero_exit']} non-zero exit codes, {t['logs']['n_tracebacks']} tracebacks; outcome store: {t['store']['n_signatures_in_store']} served methods "
    f"({t['store']['n_gates']} gates, {t['store']['n_weights']} weights, {t['store']['n_lookup']} keyed stores), {t['store']['n_unmatched']} without a method record under out/ (no method was served with other settings); "
    f"constants: {t['code_constants']['ALPHA, N_CAL, N_DRAWS, BEHAVE_DRAWS line']}; no other alpha in the code.  Code: the method construction (xmeth.py, {mc['xmeth.py'][11:]}) and the behavioural driver (run_behave.py, "
    f"{mc['run_behave.py'][11:]}) were last modified before the first outcome ({t['first_outcome'][11:]}); files modified after a job that imports them had started: {t['files_modified_after_a_job_that_imports_them_started']} "
    f"(jobs {[j[:-4] for j in t['jobs_started_before_such_a_modification']]}).  run_gate.py was changed between its run on the development log (basis of the R3 selection) and its runs on the reported logs: the adapter's own "
    f"gate-level evaluation, run by the verifier over the 20 draws, reproduces the experiment's gate-level files of the development log AND of T100 for ReScope and the cosine gate (false firing, coverage, neighbourhood firing, eta, "
    f"capped edits; largest difference {dmax} points), so the edit did not change what the driver computes for these rows.  xcommon.py (paths, bootstrap) was changed at {late.get('xcommon.py')[11:]}, seconds before its bootstrap "
    f"test file was rewritten ({t['tests_boot_written'][11:]}); the jobs started earlier compute no interval.  No version history exists: the earlier contents cannot be inspected.")
pc = disc["public_code_path"]
add("discipline_public_scores_from_public_code", bool(not pc["experiment_defines_its_own_probability_test"] and pc["adapter_calls_public_function"] and pc["adapter_uses_public_summariser"] and maxd == 0.0),
    f"the experiment calls {sorted({x for v in pc['experiment_calls'].values() for x in v})} of the adapter, which runs the public probability test (through the builder's wrapper of compute_rewrite_quality_counterfact) and the public "
    f"experiments/summarize.py; the experiment defines no probability test of its own; the verifier's direct calls of the public function give case files identical to the experiment's ({nrec} records, largest difference {maxd}); "
    f"ES / PS / NS are in columns of their own in every table.")

# ----------------------------------------------------------------------------- 12: numbers reproduced
mt = cells["main_table"]
rr = cells["relational_rows"]
nrel = sum(o["n_rows"] for o in rr.values())
nrel_bad = sum(o["n_values_differ_from_summary"] for o in rr.values())
ok = cells["n_reproduced"] == cells["n_cells"] and mt["n_cells_differ"] == 0 and nrel_bad == 0 and mt["n_ci_outside_1_point"] == 0
add("numbers_of_table_and_paper_numbers_reproduced_from_rows", bool(ok),
    f"{cells['n_reproduced']} of {cells['n_cells']} sampled entries of paper_numbers.json ({cells['sample_rule']}) reproduced from the per-row files by the verifier's aggregation, {cells['n_found_in_table_txt']} found in table.txt; "
    f"main table (section 1): {mt['n_rows']} rows parsed, {mt['n_cells'] - mt['n_cells_differ']} of {mt['n_cells']} cells reproduced (exact match, D1, kept, changed first tokens from the row files; ES / PS / NS from the case files); "
    f"{mt['n_ci_cells']} interval ends recomputed with the verifier's crossed bootstrap (other random numbers): largest deviation {mt['ci_max_abs_deviation']:.2f} points, {mt['n_ci_outside_1_point']} beyond 1 point; relational tables: "
    f"{nrel} rows of both logs rescored by the verifier's SQL from the experiment's per-tuple values, {nrel_bad} values differ; gate-level columns (false firing, coverage, neighbourhood firing over the 20 draws) of ReScope and of "
    f"the cosine gate on the development log and on T100 recomputed with the adapter's evaluation: largest difference {max(v['max_abs_difference_points'] for k_, o_ in dev.items() if isinstance(o_, dict) for v in o_.values())} points; "
    f"bootstrap tested on synthetic data: {boot['all_ok']}.")
pn("VerifyCellsReproduced", cells["n_reproduced"], "out/cells.json", "n_reproduced")
pn("VerifyMainTableCells", mt["n_cells"], "out/cells.json", "main_table.n_cells")
pn("VerifyMainTableCellsDiffering", mt["n_cells_differ"], "out/cells.json", "main_table.n_cells_differ")

# ----------------------------------------------------------------------------- 13, 14: statements of the report, from the experiment's per-row files
def pooled(sub, log, tags, pred, key):
    rows = []
    for tg in tags:
        rows += [r for r in V.load_rows(X + f"/{sub}/{log}/{tg}.jsonl.gz") if pred(r)]
    return m100(rows, key)


req_from = lambda r: r["kind"] == "request" and r["set"] == "stream" and r.get("part") is None and r["pos"] >= V.SPLIT      # noqa: E731
req_bef = lambda r: r["kind"] == "request" and r["set"] == "stream" and r.get("part") is None and r["pos"] < V.SPLIT       # noqa: E731
beh, eq = {}, {}
for f in sorted(os.listdir(X + f"/rows/{V.T500}")):
    beh.setdefault(f.split("__")[0], []).append(f[:-9])
for f in sorted(os.listdir(X + f"/rows_equal/{V.T500}")):
    a_, b_ = f[:-9].split("__")
    eq.setdefault(a_ + "|" + b_.split("_s")[0], []).append(f[:-9])
beh_from = {k: pooled("rows", V.T500, v, req_from, "new_exact") for k, v in beh.items()}
eq_from = {k: pooled("rows_equal", V.T500, v, req_from, "new_exact") for k, v in eq.items()}
top_b = max(beh_from, key=beh_from.get)
over = {k: v for k, v in eq_from.items() if v > 12.0}
rel_from = {k: e["retention_from_173_verifier_sql"] for k, e in cells["relational_rows"][V.T500]["rows"].items()}
rel_over = {k: v for k, v in rel_from.items() if v > 12.0}
goff = {d: pooled("rows", V.T500, [f"rescope@goff__d{d}"], req_bef, "new_exact") for d in (0, 1, 2)}
ex = disc["r21_expectations"]
ff = ex["false_firing_heldout_mean_over_20_draws"]
s1 = all(ff[lg][k]["min"] == 100.0 for lg in ff for k in ("rescope|gon", "where_twoblock|gon"))
cosr = [ff[lg][k]["mean"] for lg in ff for k in ("cosine|gon", "whitecos_0.1|gon")]
e5 = ex["eta"][V.T500]
stream = {}
for pfx in ("TX-G2XL-rome-T2000@173", "TX-G2XL-rome-T2000@174"):
    rows = V.load_rows(X + f"/rows_stream/{pfx}/deployed__dx.jsonl.gz")
    stream[pfx[-3:]] = {"exact": m100([r for r in rows if r["kind"] == "request"], "new_exact"), "ES": V.jload(X + f"/stream/{pfx}/deployed__dx.public.json")["all"]["ES"]}
gen5 = {lab: 100.0 * float(np.mean([r["ngram_entropy"] for r in V.jload(X + f"/gen/{V.T500}/{lab}.json")["rows"] if "failed" not in r])) for lab in ("deployed", "where_twoblock")}
compl = {}
for log in (V.T100, V.T500):
    rows = []
    for sd in (0, 1, 2):
        rows += [r for r in V.load_rows(X + f"/rows_equal/{log}/least_sacrificing__n10_s{sd}.jsonl.gz") if r.get("part") == "extra"]
    compl[log] = {"n": len(rows), "still_firing_tf": sum(bool(r["fired"]) for r in rows), "still_firing_dec": sum(bool(r["fired_dec"]) for r in rows), "D1_tf": sum(bool(r["D1_tf"]) for r in rows),
                  "D1_dec": sum(bool(r["D1_dec"]) for r in rows), "in_decision_band": sum(bool(r.get("near")) for r in rows)}
stm = [
    {"statement": "main table (T100, T500 guards on / off, before / from 173)", "holds": mt["n_cells_differ"] == 0,
     "evidence": f"{mt['n_cells']} cells of section 1 of table.txt reproduced from the per-row files; the rows of draw 0 recomputed by the verifier on the model agree row by row"},
    {"statement": "R21 (1): with guards the calibrated ReScope rows fire on 100 % of HELD-OUT prompts on T500, T1000, T2000; the cosine family stays at 5-7 %", "holds": bool(s1 and 5.0 <= min(cosr) and max(cosr) < 7.0),
     "evidence": f"per-row flags of the 20 draws: ReScope and ReScope-W two-block 100 % in every draw of the three logs: {s1}; cosine and whitecos_0.1 means between {min(cosr):.2f} and {max(cosr):.2f} %; "
                 f"verifier's model run at draw 0 on T500: {cmp_['behaviour'][V.T500]['rescope']['aggregates']['heldout.fired']['mine_on_sample']:.0f} % of 900 HELD-OUT prompts fire"},
    {"statement": "R21 (2): eta >= 1 in 1 of 20 draws on T500 and 12 of 20 on T2000; behavioural draw 0 silences 499 of 500 own requests; draws 1-2 give 96-97 % request exact before 173",
     "holds": bool(e5["n_draws_eta_ge_1_rebuilt"] == 1 and ex["eta"][V.T2000]["n_draws_eta_ge_1_from_gate_file"] == 12 and e5["own_requests_silenced_draws_0_1_2"][0] == 499 and 95.5 <= goff[1] <= 97.5 and 95.5 <= goff[2] <= 97.5),
     "evidence": f"T500: the 20 calibrations rebuilt by the verifier: eta >= 1 in {e5['n_draws_eta_ge_1_rebuilt']} draw (draw 0: {e5['rebuilt_etas'][0]:.3f}); own requests silenced with guards off in draws 0, 1, 2: "
                 f"{e5['own_requests_silenced_draws_0_1_2']} (over the 20 draws {sorted(e5['own_requests_silenced_guards_off_per_draw'], reverse=True)[:3]} are the three largest); T2000: {ex['eta'][V.T2000]['n_draws_eta_ge_1_from_gate_file']} "
                 f"of 20 in the experiment's per-draw records (not rebuilt); migration with guards off, request exact before 173, from the row files: draw 0 {goff[0]:.1f}, draw 1 {goff[1]:.1f}, draw 2 {goff[2]:.1f} %"},
    {"statement": "R21 (3): no row exceeds 12 % request exact from 173 on", "holds": bool(max(beh_from.values()) <= 12.0 and not over and not rel_over),
     "evidence": f"true for the {len(beh_from)} behavioural rows of T500 (largest: {top_b} {beh_from[top_b]:.2f} %); NOT true for the equal-evidence rows of the same table: "
                 + ", ".join(f"{k.replace('|', ' ')} {v:.1f} %" for k, v in sorted(over.items(), key=lambda kv: -kv[1])) + " (rollback of the lineage; experiment's own row files, table section 6); the same in the "
                 "relational workload (retention of the edited tuples from 173 on, verifier's SQL on the experiment's per-tuple values): "
                 + ", ".join(f"{k.replace('|m=None', '').replace('|', ' ')} {v:.1f} %" for k, v in sorted(rel_over.items(), key=lambda kv: -kv[1]))},
    {"statement": "equal evidence, n = 10 (table of the report)", "holds": True,
     "evidence": f"the cell recomputed on the model (T100, seed 0) agrees; {sum(c['kind'] == 'equal' and c['reproduced'] for c in cells['cells'])} sampled equal-evidence numbers reproduced from the row files"},
    {"statement": "R22 rows of the report (T100 and T500)", "holds": nrel_bad == 0 and rr[V.T500]["max_queries_exact_pct"] == 0.0,
     "evidence": f"{nrel} rows rescored by the verifier's SQL, {nrel_bad} values differ; T500: queries exact at most {rr[V.T500]['max_queries_exact_pct']:.1f} % in every row; diagnosis passes at m = 10: "
                 f"T100 {rr[V.T100]['diagnosis_passes_mean_by_m']['10']:.0f}, T500 {rr[V.T500]['diagnosis_passes_mean_by_m']['10']:.0f}; tuple complaints at m = 10 on T500: {rr[V.T500]['tuple_complaints_mean_by_m']['10']:.0f}"},
    {"statement": "split: the deployed model breaks at prefix 174 (ES of the first 100 requests 99 -> 51, exact 94 -> 1)", "holds": bool(round(stream["173"]["ES"]) == 99 and round(stream["174"]["ES"]) == 51 and round(stream["173"]["exact"]) == 94 and round(stream["174"]["exact"]) == 1),
     "evidence": f"row and case files of the stream: prefix 173 ES {stream['173']['ES']:.1f}, exact {stream['173']['exact']:.1f}; prefix 174 ES {stream['174']['ES']:.1f}, exact {stream['174']['exact']:.1f}"},
    {"statement": "certificate: 34,398 (T100) and 50,521 (T500) certified cases bit-identical, 0 violations; guards-on ReScope rows certify nothing on T500", "holds": bool(tot_ok and nvio == 0),
     "evidence": f"totals recomputed from the per-row counts; {nchk} cases re-checked by the verifier with {nvio} violations; guards-on rows with 0 certified prompts: {c[V.T500]['experiment_total_recomputed']['rows_with_zero_certified']}"},
    {"statement": "anomalies: frozen copy of the three driver files; screening on T500 (recall 62.5 / 100, precision 83 / 50); entropy 431 against 579; hook / cache mismatches 0", "holds": bool(
        scr["logs"][V.T500]["experiment"]["same_precision_recall"] and br["hook_mismatch_total"]["hook_mismatch"] == 0 and br["hook_mismatch_total"]["hook_mismatch_dec"] == 0 and round(gen5["where_twoblock"]) == 431 and round(gen5["deployed"]) == 579),
     "evidence": f"the frozen files have the sha1 of the reviewed originals the adapter was tested with (the verifier's shim refuses anything else); T500 screening recomputed: recall {scr['logs'][V.T500]['against']['exploding_norm']['recall']:.1f} / "
                 f"{scr['logs'][V.T500]['against']['first_token_subject']['recall']:.1f}, precision {scr['logs'][V.T500]['against']['exploding_norm']['precision']:.1f} / {scr['logs'][V.T500]['against']['first_token_subject']['precision']:.1f}; "
                 f"n-gram entropy from the generation files: repair with guards {gen5['where_twoblock']:.1f}, deployed {gen5['deployed']:.1f}; mismatches summed over {br['n_method_records']} method records: {br['hook_mismatch_total']}"},
]
bad = [s for s in stm if not s["holds"]]
add("headline_statements_of_the_report", not bad,
    f"{len(stm) - len(bad)} of {len(stm)} statements hold as worded.  Not as worded: " + "; ".join(f"'{s['statement']}': {s['evidence']}" for s in bad) if bad else f"all {len(stm)} statements hold",
    severity="minor: one summary sentence is too strong; the numbers behind it are in table.txt and are reproduced" if bad else None, statements=stm)
dev_notes = [
    f"least repair (R21 d, sacrificing form 'every reported complaint is certified'): with the library's margin a repaired complaint can stay inside the decision band; at n = 10 the experiment's own rows show "
    f"{compl[V.T100]['in_decision_band']} of {compl[V.T100]['n']} (T100) and {compl[V.T500]['in_decision_band']} of {compl[V.T500]['n']} (T500) complaints in the band, {compl[V.T100]['still_firing_tf']} and {compl[V.T500]['still_firing_tf']} "
    f"still releasing an edit in the teacher-forced pass, {compl[V.T100]['D1_dec']} and {compl[V.T500]['D1_dec']} still D1 in the decode; stated in DEVIATIONS.txt item 7 and in the column 'resolved' (teacher-forced), not in the report text",
    "task 5: the repair and the migration were also run (draw 0, first 100 requests) on the unregistered stream prefixes 150, 173, 174, 200 and 300; shown in table section 8, not listed among the additions; nothing is selected from them",
    "screening: 'exploding' is the audit's list of 20 updates with norm above |W_0| (stated in the table), not the 8 updates of the audit's summary table, which coincide with the first-token subjects",
]
add("deviations_from_the_protocol_stated", True,
    f"out/DEVIATIONS.txt lists 16 items (frozen driver copy, R3 statistics, run-time patch of the refinement, gate-level code, rows not run, guards-off log-only row, two forms of the least repair, rb_suffix, comp_min, blocklist, "
    f"lookup_subject, R22 choices, public-score intervals, overhead, eta diagnostic, certificate); the verifier's reading of the code and of the protocol (sections 0, 2, 4, 6: E10, R1-R16, R19 f, R21, R22) found no change of a registered "
    f"analysis that is missing from it.  Notes that a reader of the report text alone would miss: " + " | ".join(dev_notes), notes=dev_notes)

# ----------------------------------------------------------------------------- residual risks
risks += [
    f"Screening (R21 c) is unstable at exact ties: {ti['n_tie_edits']} edits of T2000 with repeated request prompts have margin 0; flagged count {s2['experiment']['n_flagged']} (experiment) / {s2['n_flagged']} (verifier) / "
    f"{ti['n_flagged_with_margin_below_minus_1e-9']} without ties / {ti['n_flagged_if_every_tie_is_flagged']} with all ties; the paper should fix a tie rule (e.g. margin < -tolerance, or count repeated prompts apart) before quoting "
    f"the T1000 / T2000 precision.",
    "R21 (3) as summarised in the report ('no row exceeds 12 %') does not cover the equal-evidence rows: rb_lineage reaches " + ", ".join(f"{v:.1f} %" for k, v in sorted(over.items(), key=lambda kv: -kv[1]))
    + " request exact on the requests from 173 on (equal evidence), and the rollbacks rb_screen / rb_lineage retain " + ", ".join(f"{v:.1f} %" for k, v in sorted(rel_over.items(), key=lambda kv: -kv[1]))
    + " of the edited tuples from 173 on in the relational workload; the expectation 'no method recovers them' is met by the gates and repairs, not by these rollbacks; a paper sentence should say so.",
    dev_notes[0],
    f"On T500 the guards-off outcome of the two calibrated ReScope rows is bimodal over calibration draws (own requests silenced in the 20 rebuilt draws: {sorted(e5['own_requests_silenced_guards_off_per_draw'], reverse=True)[:3]} "
    f"in three draws, at most {sorted(e5['own_requests_silenced_guards_off_per_draw'], reverse=True)[3]} in the others); the three behavioural draws contain one collapsed draw (draw 0), so the means over draws 0-2 "
    f"(e.g. request exact before 173) describe neither mode; the per-draw values should be quoted.",
    f"The verifier recomputed behaviour at calibration draw 0 only, on 300 logged and 300 HELD-OUT records per log; draws 1 and 2, the cosine-family rows, comp_min / rb_lineage / blocklist, R22 on T500 and gate level on T1000 / T2000 "
    f"were re-aggregated from the experiment's per-row files, not rerun on the model.",
    "Serving overhead (x1.05-1.15) and repair times are timings on shared GPUs and were not re-measured.",
    f"No version history of run/code: {t['files_modified_after_a_job_that_imports_them_started']} were modified after jobs that import them had started (among them the gate-level run on the development log that "
    f"the R3 selection reads); the verifier reproduced that file for ReScope and the cosine gate with the adapter's evaluation, not for the R3 gates themselves (built by the feature layer feat2/white, which the verifier did not rebuild).",
    "The verifier rebuilds methods with the same adapter (e10t) and library (rescope) as the experiment: an error inside them (thresholds, hook, guards, key cache) would be shared; the certificate, the public scores, the "
    "relational scores and the screening do not depend on the experiment's own code.",
    "The key caches of the reported logs were built by the experiment through the adapter's script and are reused here; the verifier's live model runs agree with the decisions taken from them on every sampled prompt.",
]
verd = {"experiment": "E10 public ROME logs (GPT-2 XL)", "verified_directory": V.RUN, "written_utc": __import__("time").strftime("%Y-%m-%d %H:%M:%S", __import__("time").gmtime()),
        "protocol": "10_final/PROTOCOL.md sections 0, 2, 4, 6 (E10, Review round 1, R16, R19 f, R21, R22)", "independence": "library rescope, adapter e10t, builder loader tx_public and the public evaluation code are "
        "imported; no function of run/code is imported; the experiment's outcome store is neither read nor written by the model runs; driver fx = hash-checked shim of the three reviewed files",
        "n_checks": len(checks), "n_pass": sum(c["verdict"] == "pass" for c in checks), "n_fail": sum(c["verdict"] == "fail" for c in checks), "checks": checks, "residual_risks": risks,
        "files": {"compare": "out/compare.json", "cells": "out/cells.json", "discipline": "out/discipline.json", "screening": "out/screening_T2000.json", "relational": "out/r22/TX-G2XL-rome-T100.json",
                  "rows": "out/rows/, out/rows_equal/", "public case files": "out/public/", "certificate": "out/cert/", "table": "out/table.txt", "flat records": "out/results.json", "quoted numbers": "out/paper_numbers.json"}}
V.jdump(verd, V.VER + "/verdicts.json")
pn("VerifyChecksPassed", verd["n_pass"], "verdicts.json", "n_pass")
pn("VerifyChecksFailed", verd["n_fail"], "verdicts.json", "n_fail")
V.jdump(REC, O + "/results.json", indent=None)
V.jdump(PN, O + "/paper_numbers.json")

# ----------------------------------------------------------------------------- rendered table
import textwrap  # noqa: E402

L_ = [f"E10 VERIFICATION: public ROME logs of GPT-2 XL  --  rendered {verd['written_utc']} UTC by code/v8_verdicts.py from JSON under verify/out (no number is typed)",
      f"verified: {V.RUN}", f"checks: {verd['n_pass']} pass, {verd['n_fail']} fail", ""]
for i, ch in enumerate(checks, 1):
    L_.append(f"{i:2d}. [{ch['verdict'].upper()}] {ch['check']}" + (f"   ({ch['severity']})" if ch.get("severity") else ""))
    L_ += textwrap.wrap(ch["evidence"], 170, initial_indent="      ", subsequent_indent="      ")
    if ch.get("statements"):
        for s in ch["statements"]:
            L_ += textwrap.wrap(f"{'holds' if s['holds'] else 'DOES NOT HOLD AS WORDED'}: {s['statement']} -- {s['evidence']}", 170, initial_indent="      * ", subsequent_indent="        ")
    L_.append("")
L_.append("VERIFIER (sample, draw 0) AGAINST THE EXPERIMENT ON THE SAME ROWS  (percent; verifier / experiment)")
hdr = f"   {'log':5s} {'row':22s} {'request exact':>16s} {'paraphrase exact':>18s} {'D1 HELD-OUT':>16s} {'D1 neighbourhood':>18s} {'false firing':>16s} {'req <173':>16s} {'req >=173':>16s}"
L_ += [hdr, "   " + "-" * (len(hdr) - 3)]
for ch in checks[:2]:
    for lab, v in ch["verifier_vs_experiment_on_the_same_rows"].items():
        f2 = lambda k: (f"{v[k][0]:.2f} / {v[k][1]:.2f}" if k in v else "-")      # noqa: E731
        L_.append(f"   {ch['check'][10:14]:5s} {lab:22s} {f2('edit.request.exact'):>16s} {f2('edit.paraphrase.exact'):>18s} {f2('heldout.D1_tf'):>16s} {f2('neighborhood.D1_tf'):>18s} {f2('heldout.fired'):>16s} "
                  f"{f2('edit.request.exact.before'):>16s} {f2('edit.request.exact.from'):>16s}")
L_ += ["", "RESIDUAL RISKS"]
for r_ in risks:
    L_ += textwrap.wrap(r_, 170, initial_indent="   - ", subsequent_indent="     ")
txt = "\n".join(L_) + "\n"
for pth in (O + "/table.txt", O + "/verdicts.txt"):
    with open(pth, "w") as fh:
        fh.write(txt)
print(txt)
