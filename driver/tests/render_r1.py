"""Render tests/out/table_r1.txt (the tests of stage 2: fx.review1) and, with --ready2, write ../READY.json for stage 2 -- only if
every test of stage 2 AND of stage 1 passed with the PRESENT code and the decision digest of the existing row set is unchanged.
No number is typed by hand.

  python render_r1.py [--ready2]        (runs render_ext.main() first: table_ext.txt and summary_ext.json for the present code)
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fx  # noqa: E402
import render_ext  # noqa: E402

OUT = os.path.join(HERE, "out")
S = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
CODE = fx.code_fingerprint()
R1 = fx.review1


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


def d_naive(r):
    def one(v):
        g = v["gates"]
        sm, mn = g["retr_sum"], g["retr_min"]
        return (f"{v['n_ok']}/{v['n_gates']} gates: thresholds bitwise and {v['n_decisions']} decisions == naive path ({v['n_differ']} differ); stage-1 hook == recomb's function for "
                f"{sum(1 for k, x in g.items() if k.endswith('_x') and x.get('thresholds_bitwise_equal'))}/{sum(1 for k in g if k.endswith('_x'))} _x gates; "
                f"top-1 by the raw fused score instead of the margin would release another edit on {sm['n_top1_by_raw_score_differs']} (retr_sum) / {mn['n_top1_by_raw_score_differs']} (retr_min) sequences")
    return per_log(r, one)


def d_mech(r):
    def one(v):
        w, dep = v["white"], v["white"]["null_dependent"]
        dd = "; ".join(f"{k}: fire {e['query_rule_minus_table']['fire']:+.2f}, T-Gen {e['query_rule_minus_table']['tgen']:+.0f}, I-Gen {e['query_rule_minus_table']['igen']:+.0f}, "
                       f"HN-I {e['query_rule_minus_table']['HN-I']:+.1f}, eta {e['query_rule_minus_table']['eta']:+.4f}, capped {e['query_rule_minus_table']['capped']:+.0f}" for k, e in dep.items())
        return (f"white {w['n_reproduced']}/{w['n_rows']} rows reproduced (thresholds bitwise, read-outs; rescope_null / cos_null with the table's ROW rule: old sample == the table's "
                f"{v['old_null_rows_are_the_tables']}); QUERY rule replaces {v['null_sample']['rows_replaced_by_the_query_rule']} null rows, driver minus table (points): {dd}; "
                f"retr: conjunction / lookup max |diff| {v['retr']['retr']['max_abs_diff']:.1e} / {v['retr']['lookup']['max_abs_diff']:.1e}, capped {v['retr']['retr']['capped']} == {v['retr']['retr']['capped_table']}; "
                f"recomb: rescope_x {v['recomb']['rescope_x']['n_cells']} cells max |diff| {v['recomb']['rescope_x']['max_abs_diff']:.1e}, x sha1 equal {v['recomb']['x_sha1_equal']}")
    return per_log(r, one)


def d_h3(r):
    def one(v):
        g = v["gates"]
        cov = lambda k: f"{g[k]['coverage']:.0f}" if g[k]["reachable"] else "unreachable"          # noqa: E731
        return (f"H3 ran for {v['n_gates']} calibrated gates ({v['n_reachable']} reachable, {v['n_ok']} consistent with the raised method object); coverage of the odd edits: "
                + ", ".join(f"{k} {cov(k)}" for k in ("rescope_x", "cos_centred", "white_0.1", "whitecos_0.1", "retr_conj", "retr_sum", "retr_min"))
                + f"; subset with unseen foreign part (kept / all): " + ", ".join(f"{k} {x['kept']}/{x['all']}" for k, x in v["subset_sizes"].items())
                + f"; retrieval sweep != raised rule on {v['retrieval_sweep_differs_from_raised_rule']} decisions; lookup has no scale {v['lookup_has_no_scale']}")
    return per_log(r, one)


def d_rules(r):
    va = r["validity"]
    head = ("synthetic validity (600 trials, bound {:.2f} %): retr_sum {:.2f} +- {:.2f} %, retr_min {:.2f} +- {:.2f} %; ".format(100 * va["retr_sum"]["bound"], 100 * va["retr_sum"]["mean_rate"],
            100 * va["retr_sum"]["se"], 100 * va["retr_min"]["mean_rate"], 100 * va["retr_min"]["se"])
            + f"selection reader: missing file raises {r['selection']['missing_file_raises']}, partial / invalid raise {sum(r['selection']['partial_or_invalid_raise'])}/{len(r['selection']['partial_or_invalid_raise'])}, "
            f"complete {r['selection']['complete']}, real file exists {r['selection']['real_file_exists_now']}; not defined: "
            + ", ".join(f"{k} {sum(v is True for kk, v in x.items() if kk not in ('build_raises_NotDefined', 'rowset_marks_not_defined'))} names" for k, x in r["not_defined"].items() if k.startswith("TB-"))
            + f" (builder raises NotDefined, row set marks them), always {sum(r['not_defined']['always'].values())}/{len(r['not_defined']['always'])}; ")
    return head + per_log(r, lambda v: f"_x without recombination == gate {sum(v['x_without_recombination_equals_gate'].values())}/{len(v['x_without_recombination_equals_gate'])}; row set {v['rowset']['n_rows']} rows, "
                          f"stage-1 rows same content {v['rowset']['stage1_rows_same_content']}, one calibration draw for {v['rowset']['n_calibrated']} calibrated rows {v['rowset']['same_calibration_rows']}; "
                          f"capped: " + ", ".join(f"{k} {c}" for k, c in v["rowset"]["capped"].items() if c) + f"; pools / switches / digests {v['ok']}")


def d_logs(r):
    slow = {}
    for name, v in r["logs"].items():
        if v.get("T") in (100, 500, 1000, 2000) and v.get("seconds_build"):
            slow.setdefault(v["T"], []).append(v)
    cost = "; ".join(f"T = {T}: " + ", ".join(f"{k} {max(x['seconds_build'].get(k, 0) for x in vs):.1f}" for k in ("white_0.1", "whitecos_1", "rescope_nullW", "rescope_x", "whitecos_0.1_x", "retr_min")) for T, vs in sorted(slow.items()))
    return (f"{r['n_ok']} of {r['n_logs']} logs (16 of the reduced programme + 6 prefix logs): {len(r['names'])} names built and passed at gate level (mechanics only); "
            f"build seconds (largest per T): {cost}")


def d_grid(r):
    return "; ".join(f"{k}: {v['n_built']} gates built (20 draws at alpha 0.05, four other levels at draws 0 - 1), failures {v['n_failures']}, eta = +inf in {v['n_eta_infinite']}, most capped edits: "
                     + ", ".join(f"{a} {c}" for a, c in sorted(v["largest_number_of_capped_edits"].items(), key=lambda x: -x[1])[:4]) for k, v in r["logs"].items())


def g_c(r):
    g = r["gates"]
    return (f"{r['n_ok']}/{r['n_gates']} scoped gates on {r['n_sequences']} sequences each, every sequence a forward: silent sequences not bitwise pre-edit "
            f"{sum(v['n_silent_not_bitwise_pre_edit'] for v in g.values())} of {sum(v['n_silent'] for v in g.values())}; hook != gate level {sum(v['n_released_differs_from_gate_level'] for v in g.values())}; "
            f"fields != memo path {sum(v['n_fields_differ_from_memo_path'] for v in g.values())}; firing sequences {min(v['n_firing'] for v in g.values())} - {max(v['n_firing'] for v in g.values())} per gate")


def g_q(r):
    g = r["gates"]
    return "; ".join(f"{k}: {v['n_firing']} firing, driver != RequestGate route {v['n_fields_differ_driver_vs_library_route']}, silent not bitwise pre-edit {v['n_silent_not_bitwise_pre_edit']} of {v['n_silent']}, "
                     f"decodes differing {v['n_decodes_differ']} of {v['n_decodes']}, forwards on a second pass {v['driver_forwards_second_pass']}" for k, v in g.items())


def g_h(r):
    return "; ".join((f"{k}: raise {v['delta']:.4f}, flags != gate level {v['n_edit_flags_differ']} + {v['n_test_flags_differ']}" + (f" (subset {v['subset']['n_test_rows']} rows: {v['subset']['n_flags_differ']})" if "subset" in v else ""))
                     if v["reachable"] else f"{k}: not reachable" for k, v in r["gates"].items())


CPU = (("naive", "(3a) driver == naive path", d_naive), ("mech", "(3b) mechanics tables", d_mech), ("h3", "(3d) read-out H3", d_h3), ("rules", "(1, 2) rules, selection, row set", d_rules),
       ("logs", "(1, 2) all logs, mechanics", d_logs), ("grid", "(1) draws and levels", d_grid))
GPU = (("c_scoped", "(3c) behaviour, scoped gates", g_c), ("q_retrieval", "(3c) behaviour, retrieval", g_q), ("h_h3", "(3d) H3, behavioural form", g_h))
TOUCHED = ["fx/__init__.py: docstring (4 lines), VERSION, import of review1 (1 line)",
           "fx/gate.py decide(): 2 lines (an object with decide_rows decides itself: retrieval rules)",
           "fx/methods.py content_digest(): 2 lines (an object with content_digest hashes itself); build(): 4 lines (names of fx.review1; the KeyError text)",
           "fx/behave.py Session.tf(): 3 lines, Session.generate(): 3 lines (kind 'request' -> fx.review1.session_tf / session_generate); hard_negatives(): 1 line "
           "(a retrieval method decodes only the rows on which it releases); tf_hashed(): 2 lines (TypeError for kind 'request')",
           "tests/t_ext_cpu.py, tests/render_ext.py: unchanged; originals of the four fx files: review/orig_stage2/"]


def main():
    sys.argv = [a for a in sys.argv if a != "--ready"]
    render_ext.main()                                                   # stage 1: table_ext.txt + summary_ext.json for the present code
    s1 = J("summary_ext.json")
    rows, tests = [], {}
    for nm, title, f in CPU:
        r = J(f"r1_cpu_{nm}.json")
        v = verdict(r)
        tests[f"r1_cpu_{nm}"] = v
        try:
            det = f(r) if r else ""
        except Exception as e:                                          # noqa: BLE001
            det = f"(detail not rendered: {type(e).__name__}: {e})"
        rows.append((title, "CPU", v, det))
    for log, x in S.items():
        for nm, title, f in GPU:
            r = J(f"r1_gpu_{log}_{nm}.json")
            v = verdict(r)
            if r is not None and (r.get("hook_mismatch") or r.get("memo_conflicts")):
                v = "FAIL"
            tests[f"r1_gpu_{x}_{nm}"] = v
            try:
                det = f(r) if r else ""
            except Exception as e:                                      # noqa: BLE001
                det = f"(detail not rendered: {type(e).__name__}: {e})"
            rows.append((title, x, v, det))
    dg, d0, d1 = J("digest_compare.json"), J("digest_s2before.json"), J("digest_after.json")
    digest_ok = bool(dg and dg["ok"] and dg["fx_after"]["fx"] == CODE["fx"] and d0 and d1 and d0["logs"] == d1["logs"])
    bad1 = dict(s1["not_passing"])
    txt = [f"TESTS OF STAGE 2 OF THE fx DRIVER (fx.review1: the gates registered after the review)   (rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by tests/render_r1.py from tests/out/*.json; code {CODE})", "",
           fx.io.text_table(("test (task)", "scope", "result", "detail"), rows, align=["l", "l", "l", "l"]), "",
           f"(3e) EARLIER TESTS on the present code (tests/out/table_ext.txt, tests/out/table.txt): {s1['n_pass_present_code']} of {s1['n_tests']} PASS; not passing: {bad1 or 'none'}; "
           f"passed on older code and not re-run (mechanics / full E1 pass on cut logs and paths stage 2 does not touch): {s1['pass_on_older_code_not_rerun'] or 'none'}", "",
           "DECISION DIGEST of the existing row set (both development logs, alpha = 0.05, calibration draw 0; tests/digest.py): " +
           (f"{dg['n_equal']} of {dg['n_rows']} rows unchanged ({dg['n_decisions']} decisions) against the first driver ({dg['fx_before']['fx']}) and against the snapshot taken before the "
            f"first change of stage 2 (digest_s2before.json, fx {d0['fx_before_or_after']['fx']}): {digest_ok}; fx now {dg['fx_after']['fx']}" if dg and d0 else "MISSING"), "",
           "LINES TOUCHED IN EXISTING FILES (everything else of stage 2 is the new module fx/review1.py and new test files):"] + ["  " + t for t in TOUCHED] + [""]
    out = "\n".join(txt)
    with open(os.path.join(OUT, "table_r1.txt"), "w") as f:
        f.write(out)
    print(out[:6000])
    bad = {k: v for k, v in tests.items() if v != "PASS"}
    bad.update({k: v for k, v in bad1.items()})
    summary = {"n_tests": len(tests) + s1["n_tests"], "n_pass_present_code": sum(v == "PASS" for v in tests.values()) + s1["n_pass_present_code"],
               "pass_on_older_code_not_rerun": s1["pass_on_older_code_not_rerun"], "not_passing": bad, "digest_unchanged": digest_ok, "stage2_tests": tests, "stage1_tests": s1["tests"]}
    fx.io.jdump(summary, os.path.join(OUT, "summary_r1.json"), indent=1)
    print("\nnot passing:", bad, "| digest unchanged:", digest_ok)
    if "--ready2" in sys.argv:
        if bad or not digest_ok:
            raise SystemExit("READY.json is NOT written: tests are not all passing with the present code, or the digest changed")
        root = os.path.dirname(HERE)
        old = fx.io.jload(os.path.join(root, "READY.json"))
        s1fp = old["code_fingerprint"] if old.get("stage") == 1 else old["stage1_fingerprint"]
        d = os.path.dirname(os.path.abspath(fx.__file__))
        files = {f: hashlib.sha1(open(os.path.join(d, f), "rb").read()).hexdigest()[:12] for f in sorted(os.listdir(d)) if f.endswith(".py")}
        ready = dict(old)
        ready.update({"stage": 2, "date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "code_fingerprint": CODE["fx"], "stage1_fingerprint": s1fp, "library_fingerprint": CODE["rescope"],
                      "fx_version": CODE["fx_version"], "source_files": files,
                      "reviewed_base": "b92d0532498f (review/patches.txt); the stage-1 extension and stage 2 (fx.review1) are not yet independently reviewed",
                      "method_names": list(fx.methods.NAMES) + list(R1.NAMES), "method_names_stage1": list(fx.methods.NAMES), "method_names_review_round": list(R1.NAMES),
                      "e1_labels": list(fx.methods.E1_LABELS), "e1_gate_level_only": list(fx.methods.E1_GATE_LEVEL) + list(R1.R1_GATE_LEVEL), "e3_labels": list(fx.methods.E3_LABELS),
                      "r1_labels": list(R1.R1_LABELS), "e1_rowset_after_review": "fx.review1.e1_rowset_review1 (E1_LABELS + R1_LABELS; gate level only: rescope_pac, rescope_nullW)",
                      "cosine_family": list(R1.COSINE_FAMILY), "retrieval_family": list(R1.RETRIEVAL_FAMILY), "x_forms": [b + "_x" for b in R1.X_BASES],
                      "selection_file": {"path": R1.SELECTION, "exists": os.path.exists(R1.SELECTION), "reader": "fx.review1.read_selection / selected", "writer": "fx.review1.write_selection"},
                      "functions_that_return_methods": old["functions_that_return_methods"] + ["fx.review1.build (through fx.methods.build)", "fx.review1.e1_rowset_review1"],
                      "feature_packages": {"dir": R1.FEAT2, "problems": "FEAT2_PROBLEMS.txt", "not_defined": "fx.review1.NotDefined / why_not(L, name)"},
                      "lines_touched_in_existing_files": TOUCHED,
                      "logs": {n: {"role": e["role"], "exists": fx.logs.exists(n), "T": e["T"], "backbone": e["backbone"], "cut_by_R19": bool(e.get("cut_by_R19"))} for n, e in fx.logs.REGISTRY.items()},
                      "test_summary": summary, "tables": ["tests/out/table_r1.txt", "tests/out/table_ext.txt", "tests/out/table.txt", "out/prefix_logs.txt"],
                      "usage": "USAGE.txt (rules at the top; sections EXTENSION (stage 1), GATES OF THE REVIEW ROUND (stage 2), EXPERIMENT TASKS)"})
        fx.io.jdump(ready, os.path.join(root, "READY.json"), indent=1)
        print("READY.json written: stage 2,", ready["code_fingerprint"], "(stage 1:", s1fp + ")")


if __name__ == "__main__":
    main()
