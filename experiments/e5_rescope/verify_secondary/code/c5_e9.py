"""Check 5 (E9): my own forwards (plain Evaluator) for the revocation set m = 10, s = 0: frame rule, bit-for-bit logits, own rollback weight; against the experiment's row files."""
from vcommon import *
import gzip
def rd(p):
    with gzip.open(p, "rt") as f: return [json.loads(l) for l in f]
res = {}
for name, sh in ((QDEV, "Q"), (LDEV, "L")):
    J = jload(f"{OUT}/g_e9_{name}.json"); R = set(J["R"])
    rows = rd(f"{OUT}/g_e9_{name}_rows.jsonl.gz")
    E = {t: {r["sid"]: r for r in rd(f"{FINAL}/e9_maintenance/out/hash/{sh}__{t}.jsonl.gz")} for t in ("base_scoped", "scoped_m10_s0", "linear_m10_s0", "base_deployed")}
    frame = [r for r in rows if not (set(r["scoped"]["released"]) & R)]
    touched = [r for r in rows if set(r["scoped"]["released"]) & R]
    viol = [r["sid"] for r in frame if not r["same_logits_before_after"]]
    other = [r["sid"] for r in touched if r["masked"]["released"]]
    masked_releases_revoked = [r["sid"] for r in rows if set(r["masked"]["released"]) & R]
    ed = [r for r in rows if r["kind"] == "edit"]; pr = [r for r in rows if r["kind"] == "probe"]
    kept = [r for r in ed if r["edit"] not in R]; rev = [r for r in ed if r["edit"] in R]
    pct = lambda rs, k, f="exact": 100 * float(np.mean([r[k][f] for r in rs]))
    def by_metric(rs, k): return {m: pct([r for r in rs if r["metric"] == m], k) for m in ("rel", "tgen", "igen")}
    mine = {"n_sequences": len(rows), "frame": len(frame), "violations": len(viol), "touched": len(touched), "touched_logits_changed": sum(not r["same_logits_before_after"] for r in touched),
            "other_edit_released_after_masking": len(other), "masked_gate_releases_a_revoked_edit": len(masked_releases_revoked),
            "kept_seqs_logits_changed_scoped": sum(not r["same_logits_before_after"] for r in kept), "probes_logits_changed_scoped": sum(not r["same_logits_before_after"] for r in pr),
            "kept_seqs_flag_changed_scoped": sum(r["scoped"]["exact"] != r["masked"]["exact"] for r in kept), "probes_flag_changed_scoped": sum(r["scoped"]["D1"] != r["masked"]["D1"] for r in pr),
            "scoped_revoked_after": by_metric(rev, "masked"), "scoped_revoked_before": by_metric(rev, "scoped"), "scoped_kept_after": by_metric(kept, "masked"), "scoped_kept_before": by_metric(kept, "scoped"),
            "linear_kept_after": by_metric(kept, "rollback"), "linear_kept_before": by_metric(kept, "deployed"), "linear_revoked_after": by_metric(rev, "rollback"), "linear_revoked_before": by_metric(rev, "deployed"),
            "linear_kept_seqs_flag_changed": sum(r["rollback"]["exact"] != r["deployed"]["exact"] for r in kept), "linear_kept_seqs_logits_changed": sum(not r["same_logits_rollback_deployed"] for r in kept),
            "linear_probes_logits_changed": sum(not r["same_logits_rollback_deployed"] for r in pr), "linear_probes_flag_changed": sum(r["rollback"]["D1"] != r["deployed"]["D1"] for r in pr),
            "D1_scoped_after": pct(pr, "masked", "D1"), "D1_scoped_before": pct(pr, "scoped", "D1"), "D1_linear_after": pct(pr, "rollback", "D1"), "D1_linear_before": pct(pr, "deployed", "D1"),
            "rollback_weight_equals_harness_bitwise": J["rollback_weight_equals_harness"]}
    # the experiment's rows for the same set
    def ex(t, sid, f): return E[t][sid][f]
    keptE = [s for s, r in E["base_scoped"].items() if r["kind"] == "edit" and r["edit"] not in R]
    probesE = [s for s, r in E["base_scoped"].items() if r["kind"] == "probe"]
    expm = {"n_sequences": len(E["base_scoped"]), "frame": sum(not (set(r["released"]) & R) for r in E["base_scoped"].values() if r["sid"] in E["scoped_m10_s0"]),
            "violations": sum((not (set(E["base_scoped"][s]["released"]) & R)) and E["base_scoped"][s]["logit_sha1"] != r["logit_sha1"] for s, r in E["scoped_m10_s0"].items()),
            "touched": sum(bool(set(E["base_scoped"][s]["released"]) & R) for s in E["scoped_m10_s0"]),
            "linear_kept_after": {m: 100 * float(np.mean([E["linear_m10_s0"][s]["exact"] for s in keptE if E["linear_m10_s0"][s]["metric"] == m])) for m in ("rel", "tgen", "igen")},
            "linear_kept_before": {m: 100 * float(np.mean([E["base_deployed"][s]["exact"] for s in keptE if E["base_deployed"][s]["metric"] == m])) for m in ("rel", "tgen", "igen")},
            "linear_kept_seqs_flag_changed": sum(E["linear_m10_s0"][s]["exact"] != E["base_deployed"][s]["exact"] for s in keptE),
            "D1_linear_after": 100 * float(np.mean([E["linear_m10_s0"][s]["D1"] for s in probesE])), "D1_scoped_after": 100 * float(np.mean([E["scoped_m10_s0"][s]["D1"] for s in probesE])),
            "probes_logits_changed_scoped": sum(E["scoped_m10_s0"][s]["logit_sha1"] != E["base_scoped"][s]["logit_sha1"] for s in probesE),
            "kept_seqs_logits_changed_scoped": sum(E["scoped_m10_s0"][s]["logit_sha1"] != E["base_scoped"][s]["logit_sha1"] for s in keptE)}
    sha_eq = {t: sum(r[k]["sha"] == E[t][r["sid"]]["logit_sha1"] for r in rows) for t, k in (("base_scoped", "scoped"), ("scoped_m10_s0", "masked"), ("linear_m10_s0", "rollback"), ("base_deployed", "deployed"))}
    flag_eq = {t: sum((r[k]["exact"] == E[t][r["sid"]]["exact"]) for r in rows) for t, k in (("base_scoped", "scoped"), ("scoped_m10_s0", "masked"), ("linear_m10_s0", "rollback"), ("base_deployed", "deployed"))}
    rel_eq = {t: sum(r[k]["released"] == E[t][r["sid"]]["released"] for r in rows) for t, k in (("base_scoped", "scoped"), ("scoped_m10_s0", "masked"))}
    agree = all(mine[k] == expm[k] for k in ("frame", "violations", "touched", "linear_kept_seqs_flag_changed", "probes_logits_changed_scoped", "kept_seqs_logits_changed_scoped")) and \
        all(abs(mine[k][m] - expm[k][m]) < 1e-9 for k in ("linear_kept_after", "linear_kept_before") for m in ("rel", "tgen", "igen")) and abs(mine["D1_linear_after"] - expm["D1_linear_after"]) < 1e-9
    res[name] = {"R": sorted(R), "mine": mine, "experiment_rows_same_set": expm, "my_logit_hashes_equal_to_experiment": sha_eq, "my_flags_equal_to_experiment": flag_eq,
                 "released_sets_equal_to_experiment": rel_eq, "violating_sids": viol, "counts_agree": bool(agree)}
    print(name, json.dumps(res[name], indent=0)[:3000])
jdump(res, OUT + "/c5_e9.json")
