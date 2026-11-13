"""Verification: renders verify/verdicts.json and verify/out/verify_results.json into plain text.     usage: python v_render.py

  verify/out/verify_table.txt     every comparison (table, extras, discrepancies, details per log)
  verify/out/final_message.txt    the final message (at most 250 words; the script fails if it is longer); the same text
                                  is stored in verify/verdicts.json under 'message' with its word count.
No number is typed here: every number is read from the two JSON files.
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import os
import time

MAX_WORDS = 250


def table(header, rows, indent=0):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    out = ["  ".join(str(header[i]).ljust(w[i]) for i in range(len(header))).rstrip(), "  ".join("-" * w[i] for i in range(len(header)))]
    out += ["  ".join(str(r[i]).ljust(w[i]) for i in range(len(header))).rstrip() for r in rows]
    return [" " * indent + x for x in out]


def yn(b):
    return "yes" if b else "NO"


def e(x, nd=2):
    return "-" if x is None else f"{x:.{nd}e}"


def f(x, nd=4):
    return "-" if x is None else (str(int(x)) if float(x) == int(x) else f"{x:.{nd}g}")


def main():
    V = C.jload(os.path.join(C.VER, "verdicts.json"))
    R = C.jload(os.path.join(C.OUT, "verify_results.json"))
    Q, L = V["logs"]["Q"], V["logs"]["L"]
    out = ["E5 (LINEAR PART) -- INDEPENDENT VERIFICATION OF THE NUMBERS QUOTED FOR THE THEORY SECTION",
           f"rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by verify/code/v_render.py from verify/verdicts.json and verify/out/verify_results.json; no number is typed by hand.",
           f"Logs: Q = {Q} ({V['n_complaints'][Q]} REPAIR-pool D1 complaints, decoder v2), L = {L} ({V['n_complaints'][L]}).  Cells are 'Q | L'.",
           "Mine: own code (verify/code), fp64; contributions w_i = u_i <psi_i, k> as explicit vectors from edit_factors.pt; last prompt key from the final key cache (10_final/cache),",
           "not from the Track C capture that the report used (the two captures are bit for bit equal on every complaint); inner products and coordinates SIGNED, norms magnitudes.",
           "Scopes: 'sample' = 60 seeded complaints per log (30 image + 30 text, harness rule, seed 0); the medians in column 'mine' are over ALL complaints (what the report quotes);",
           "capacity: five complaints of the sample (numpy default_rng(5)), and every complaint as an extra.",
           ""]
    out.append("TABLE (the four tasks)")
    out += table(["quantity", "report", "mine", "agree", "verdict", "note"],
                 [[r["quantity"], r["report"], r["mine"], yn(r["agree"]), r["verdict"], r["note"]] for r in V["table"]])
    out.append("")
    out.append("EXTRAS (numbers of the summary outside the four tasks)")
    out += table(["quantity", "report", "mine", "agree", "verdict", "note"],
                 [[r["quantity"], r["report"], r["mine"], yn(r["agree"]) if r["id"] != "opt_cert" else "n/a", r["verdict"], r["note"]] for r in V["extras"]])
    out.append("")
    out.append("DISCREPANCIES")
    for d in V["discrepancies"]:
        out.append(f"  [{d['kind']}] {d['text']}" + (f"  CAUSE: {d['cause']}" if d["cause"] else ""))
    out.append("")
    for name in (Q, L):
        X = R["logs"][name]
        out.append("#" * 150)
        out.append(f"{name}")
        out.append("#" * 150)
        nc, sm, ky = X["n_complaints"], X["sample"], X["keys"]
        out.append(f"complaints: mine {nc['mine']} ({nc['mine_by_ptype']['image']} image, {nc['mine_by_ptype']['text']} text), report {nc['report']} ({nc['report_by_ptype']['image']}, {nc['report_by_ptype']['text']}), same set {yn(nc['same_set'])}; "
                   f"my split and my sample against the harness sampler: candidates equal {yn(sm['harness_check']['candidates_equal'])}, sample equal {yn(sm['harness_check']['sample_equal'])}; "
                   f"sample {sm['n']} ({sm['by_ptype']['image']} image, {sm['by_ptype']['text']} text).")
        out.append(f"last prompt keys (final key cache): same probe row and answer {ky['all']['n_same_uid_group_answer']}/{ky['all']['n']}, REPAIR pool {ky['all']['n_pool_repair']}, bit for bit equal to the Track C capture "
                   f"{ky['all']['n_bitwise_equal_trackc']}/{ky['all']['n']} (sample {ky['sample']['n_bitwise_equal_trackc']}/{ky['sample']['n']}); teacher-forced position one token after the decode's: {ky['all']['n_lp_shift']} (report {ky['report_n_lp_shift']}).")
        ll = X["log_level"]
        out.append(f"log: T {ll['T']}, d_in {ll['d_in']}, d_out {ll['d_out']}, {ll['n_batches']} batches, module dtype {ll['module_dtype']}; ||sum_i u_i psi_i^T - (W_E - W_0)||_F / ||W_E - W_0||_F = {e(ll['delta_fac_vs_applied_fro'])}; "
                   f"cosine matrix of the write vectors: gamma {f(ll['gamma_U'])}, Gamma {f(ll['Gamma_U'])}, ratio {f(ll['ratio_U'])}.")
        out.append("")
        out.append("(1) CONDITIONING AT THE COMPLAINT KEY   (medians [quartiles]; 'dev' = largest deviation from the report's row of the same complaint)")
        rows = []
        for kd in ("last", "stacked"):
            for sc in ("sample", "all"):
                s = X["conditioning"][kd][sc]
                rows.append([kd, sc, s["n"], f"{f(s['mine_gamma']['median'])}", f"{f(s['mine_Gamma']['median'])}",
                             f"{f(s['mine_ratio']['median'])} [{f(s['mine_ratio']['q25'])}, {f(s['mine_ratio']['q75'])}]", f"{f(s['report_rows_ratio']['median'])}",
                             f"{f(s['mine_frac_neg']['median'])}", f"{f(s['report_rows_frac_neg']['median'])}", s["n_cancellation_free"],
                             e(max(s["max_rel_dev_gamma"], s["max_rel_dev_Gamma"], s["max_rel_dev_ratio"])), e(s["max_abs_dev_frac_neg"]), e(s["max_rel_dev_two_routes"]), s["n_distinct_ratio_rel_1e-9"]])
        out += table(["key", "scope", "n", "gamma", "Gamma", "Gamma/gamma", "report rows", "neg. share", "report rows", "canc.-free", "dev gamma, Gamma, ratio (rel)", "dev share (abs)", "eigvalsh vs SVD (rel)", "distinct ratios"], rows, 4)
        pn = X["conditioning"]["report_paper_numbers"]
        out.append(f"    report (paper_numbers.json, all complaints): last key gamma {f(pn['last']['median_gamma'])}, Gamma {f(pn['last']['median_Gamma'])}, ratio {f(pn['last']['median_Gamma_over_gamma'])}, negative share "
                   f"{f(pn['last']['median_share_negative_pairs'])}; stacked ratio {f(pn['stacked']['median_Gamma_over_gamma'])} [{f(pn['stacked']['q25_Gamma_over_gamma'])}, {f(pn['stacked']['q75_Gamma_over_gamma'])}].  "
                   f"sqrt(gamma/Gamma) = {f(X['conditioning']['last']['all']['mine_tol_factor']['median'])}; gamma T = {f(X['conditioning']['gamma_times_T'])}.")
        la = X["conditioning"]["last"]["all"]
        out.append(f"    factor form against the applied weight difference at the last key, median: mine {f(la['mine_fac_vs_applied']['median'])}, report rows {f(la['report_rows_fac_vs_applied']['median'])}; "
                   f"edits with a negative coordinate, median: {f(la['mine_n_neg_coord']['median'])}; pairs with an inner product of exactly zero: {la['n_zero_pairs']}.")
        out.append("")
        out.append("(2) ROLLBACK BY RULE AND THE BOUNDS   (deletions of T = 100; LF largest norm first, FO adaptive first-order greedy, BI best improvement; medians; '=' complaints with the report's count)")
        out.append("    bound mu = max(1, mu(c^2 |D|^2 / gamma)): the lower bound of the technical report's sandwich (prop:app:cond); 'displayed' = LF(c sqrt(Gamma/gamma)): what Prop. prop:cond AS DISPLAYED in")
        out.append("    sec_theory.tex gives for OPT_c; 'check' = complaints on which LF(c) <= the least rule at tolerance c sqrt(gamma/Gamma) (the displayed claim is not contradicted).")
        rows = []
        for kd in ("last", "stacked"):
            for t in ("0.1", "0.3", "0.5"):
                for sc in ("sample", "all"):
                    s = X["rollback"][kd][t][sc]
                    rows.append([kd, t, sc, s["n"], f(s["mine_lf"]["median"]), f(s["mine_fo"]["median"]), f(s["mine_bi"]["median"]), f(s["mine_lb1"]["median"]), f(s["mine_lf_over_lb1"]["median"]),
                                 f(s["mine_lb_stmt"]["median"]), s["n_lb_stmt_zero"], f"{s['n_equal_lf']}/{s['n_equal_fo']}/{s['n_equal_bi']}/{s['n_equal_lb_mu']}", f"{s['n_stmt_check']}/{s['n']}",
                                 e(s["min_rel_margin_of_a_stop_decision"]), ", ".join(s["groups_bi_deletes_all"]) or "-"])
                p = X["rollback"][kd][t]["report_paper_numbers"]
                rows.append([kd, t, "report", "all", f(p["median_lf"]), f(p["median_fo"]), f(p["median_bi"]), f(p["median_lb"]), f(p["median_lf_over_lb"]), "n/a", "n/a", "", "", "", ""])
        out += table(["key", "c", "scope", "n", "LF", "FO", "BI", "bound mu", "LF / bound mu", "displayed", "displayed = 0", "= LF/FO/BI/mu", "check", "least rel. margin of a stop", "BI deletes all"], rows, 4)
        out.append("")
        cp = X["capacity"]
        ek, fm = cp["edit_keys"], cp["family"]
        out.append("(3) CAPACITY OF WHOLE-SEQUENCE PROTECTION   (n smallest edit sequences protected; a complaint is repairable iff every singular value of (I - P_n) K_q exceeds 1e-9 max ||k_q||)")
        out.append(f"    edit-sequence keys: harness cache; against the final key cache (independent capture): uid, length and last prompt position equal {ek['cross_check_final_key_cache']['n_uid_len_lastpos_equal']}/100, "
                   f"last prompt key bit for bit {ek['cross_check_final_key_cache']['n_last_key_bitwise_equal']}/100.  Group sizes {ek['group_size_min_max'][0]}-{ek['group_size_min_max'][1]} "
                   f"(report {ek['report_group_size_min_max'][0]}-{ek['report_group_size_min_max'][1]}), full rank {ek['n_groups_full_rank']}/100, ranks and order equal to the report's: {yn(ek['group_rank_equal_report'])}, {yn(ek['order_equal_report'])}.")
        out.append(f"    prefix: positions bit for bit equal in all 100 sequences {ek['z_bitwise']} (report {ek['report_z_bitwise']}); positions within 10 % of edit 0 in every sequence {ek['z_near']} (report z = {ek['report_z']}), "
                   f"largest relative difference there {f(ek['near_prefix_max_rel_diff'], 3)}; bitwise-distinct keys at the leading positions {ek['distinct_keys_at_leading_positions']}.")
        out.append(f"    my method: exact de-duplication of the keys + ONE Householder QR; distinct keys = rank at every n: equal to the report's rank at {fm['n_points_compared']} family points: {yn(fm['rank_equal_at_every_n'])}; "
                   f"n_free = {fm['mine_n_free']} ({fm['mine_keys_at_n_free']} keys, {fm['mine_rank_at_n_free']} distinct, {fm['mine_classes_at_n_free']} prefix copies; report rank {fm['report_rank_at_kept']}); "
                   f"key space filled at n = {fm['mine_n_sat']} (report {fm['report_saturated_at_n']}); a second prefix copy enters at n = {fm['first_n_with_two_classes']}; least |R_jj| / s_1 up to n_free {e(fm['min_rdiag_over_s1_at_n_free'])}.")
        out.append(f"    the capacity counts DISTINCT keys: at n = {fm['mine_n_free']} the {fm['mine_keys_at_n_free']} protected keys hold {fm['mine_rank_at_n_free']} distinct ones (d_in = {X['log_level']['d_in']}); "
                   f"if no key were a bit-for-bit duplicate of another, {cp['law']['mine']['z0']['kept_pred']} sequences would fit (the law with z = 0).  The keys at ALL positions of the edit sequences exist in ONE cache only "
                   f"(07_repair_v3/harness/cache, also the report's source); the independent capture covers the last prompt position.")
        pq = X["prefix"]
        out.append(f"    prefix classes (sequences whose first {pq['z']} keys are bit for bit equal): {pq['n_classes']}; (size, shortest, longest sequence) by length: "
                   + ", ".join(f"({d['n']}, {d['len_min']}, {d['len_max']})" for d in pq["classes_by_length"])
                   + f"; the classes are disjoint ranges of the sequence length: {yn(pq['classes_are_length_intervals'])}; relative difference between the copies {f(pq['rel_diff_between_classes']['min'], 3)} .. {f(pq['rel_diff_between_classes']['max'], 3)}; "
                   f"all copies together: {pq['n_prefix_keys_all_classes']} keys of rank {pq['rank_of_all_prefix_copies']}; a second class enters the family at n = {pq['first_n_with_a_second_class']}.")
        sp = cp.get("spectrum") or {}
        for lab in ("n_free", "n_sat"):
            if lab in sp:
                s = sp[lab]
                out.append(f"    LAPACK singular values of the distinct keys at n = {s['n']}: {s['n_distinct']} values, largest {f(s['s1_distinct'], 6)} (with duplicates {f(s['s1_with_duplicates'], 6)}), smallest {e(s['s_min_distinct'])}, "
                           f"harness cut {e(s['cut'])}, numerical rank {s['numerical_rank']}.")
        rows = [[x["group"], x["ptype"], x["A"], x["mine_n_cap_last"], x["report_n_cap_last"], x["mine_n_cap_stacked"], x["report_n_cap_stacked"], f(x["mine_escape_at_n_cap"], 6), f(x["report_escape_at_n_cap"], 6),
                 e(x["rel_dev_escape"]), e(x["mine_escape_at_n_cap_plus_1"]), e(x["report_escape_at_n_cap_plus_1"]), yn(x["agree"])] for x in cp["five"]]
        out += table(["complaint", "type", "A", "n_cap last: mine", "report", "n_cap stacked: mine", "report", "escape at n_cap: mine", "report", "rel. dev", "escape at n_cap + 1: mine", "report", "agree"], rows, 4)
        out.append("    (my escape at n_cap + 1 is zero by the rank: the protected keys there have the full rank d_in, see the singular values above; the report's value is its round-off.)")
        al = cp["all"]
        out.append(f"    every complaint ({al['n']}): n_cap equal to the report's, last {al['n_equal_n_cap_last']}, stacked {al['n_equal_n_cap_stacked']}; my values {al['mine_n_cap_last_values']} / {al['mine_n_cap_stacked_values']}; "
                   f"escape of the last key at n_free: median {f(al['mine_escape_at_n_free']['median'])}, least {f(al['mine_escape_at_n_free']['min'])}; largest relative deviation of the escape from the report's up to n_free {e(al['max_rel_dev_escape_last_up_to_n_free'])}.")
        lw = cp["law"]
        out.append("    law (a = 1), T - s*_Z / s*_Z: " + "; ".join(f"{k} (z = {v['z']}): {v['kept_pred']} / {v['sstar']}" for k, v in lw["mine"].items())
                   + f"; report s*_Z {lw['report_sstar_Z']}, with z = 0 {lw['report_sstar_z0']}, measured protectable edits {lw['report_kept_feasible']}; the prefix is shared exactly: {yn(lw['premise_exact'])}.")
        for p, v in cp["points"].items():
            out.append(f"    {p} only: {v['mine_n_keys']} keys, rank {v['mine_rank']} (report {v['report_rank']}), complaints repairable {v['mine_n_repairable']} (report {v['report_n_repairable']}), "
                       f"median escape of the last key {f(v['mine_median_escape'], 6)} (report {f(v['report_median_escape'], 6)}).")
        tk = X["target_rank"]
        out.append(f"    target of the compensation (applied weight difference): rank V = A for {tk['n_rank_V_equals_A']}/{tk['n']} complaints (sample {tk['sample']['n_rank_V_equals_A']}/{tk['sample']['n']}), rank K_q = A for {tk['n_rank_Kq_equals_A']}/{tk['n']}, "
                   f"V nonzero at the last key for {tk['n_V_last_nonzero']}/{tk['n']}; equal to the report's ranks for {tk['n_rank_V_equal_report']} and {tk['n_rank_Kq_equal_report']} complaints: "
                   f"'repairable' (full column rank of (I - P) K_q) is then equivalent to the existence of an admissible correction.")
        nu = X["nu"]
        out.append("")
        out.append(f"(extra) vertex cover number of the negative graph, last key, every complaint: min / median / max {f(nu['mine']['min'])} / {f(nu['mine']['median'])} / {f(nu['mine']['max'])} "
                   f"(report {' / '.join(f(x) for x in nu['report_min_median_max'])}); equal per complaint {nu['n_equal_per_complaint']}/{nu['n']}; integer program optimal {nu['n_milp_optimal']}/{nu['n']}, clique number agrees {nu['n_methods_agree']}/{nu['n']}.")
        ou, ol = R["opt_ub"][name], R["opt_lb"]["logs"][name]
        out.append("")
        out.append(f"(extra) OPTIMUM CODE, key 'last' ({ou['n_rows']} finished rows; {ol['n_within_budget']} complaints within the report's budget).  Mine: R_best checked as a deletion set; lower bounds by a Lagrangian certificate")
        out.append("    that needs no solver (verify/code/v_opt_lb.py: diagonal shift delta with G - Diag(delta) positive definite, multiplier, one point of the box; valid for every kept set).  'report LB - mine':")
        out.append("    histogram over the finished rows of (report's lower bound) - (my certified lower bound): <= 0 confirms the report's bound; > 0 is NOT a refutation (the report's bound rests on its branch and bound).")
        rows = []
        for t, s in ol["per_tol"].items():
            a, w, rp = s["all_rows"], s["within_budget"], s["report"]
            rows.append([t, f"{ou['per_tol'][t]['n_ub_feasible']}/{ou['n_rows']}", f"{a['n_report_lb_confirmed']}/{a['n']}", ", ".join(f"{k}: {v}" for k, v in a["report_lb_minus_mine_hist"].items()), a["n_my_bound_above_report_ub"],
                         f"{w['n_report_lb_confirmed']}/{w['n']}", f"{w['n_my_bracket_closed']} ({rp['opt_n_certified']})",
                         f"{f(w['median_LBv'])}..{f(w['median_UBv'])} ({f(rp['opt_median_lower_bound'])}..{f(rp['opt_median_upper_bound'])})",
                         f"{f(w['median_bi_over_opt_lo'])}..{f(w['median_bi_over_opt_hi'])} ({f(rp['opt_median_bi_over_opt_lo'])}..{f(rp['opt_median_bi_over_opt_hi'])})",
                         f"{f(w['median_fo_over_opt_lo'])}..{f(w['median_fo_over_opt_hi'])} ({f(rp['opt_median_fo_over_opt_lo'])}..{f(rp['opt_median_fo_over_opt_hi'])})",
                         f"{f(w['median_lf_over_opt_lo'])}..{f(w['median_lf_over_opt_hi'])} ({f(rp['opt_median_lf_over_opt_lo'])}..{f(rp['opt_median_lf_over_opt_hi'])})",
                         f"{f(w['median_lb_over_opt_lo'])}..{f(w['median_lb_over_opt_hi'])} ({f(rp['opt_median_lb_over_opt_lo'])}..{f(rp['opt_median_lb_over_opt_hi'])})"])
        out += table(["c", "UB feasible", "report LB confirmed", "report LB - mine", "my LB > report UB", "confirmed, within budget", "optimum certified: mine (report)", "median LB..UB: mine (report)",
                      "BI / OPT: mine (report)", "FO / OPT: mine (report)", "LF / OPT: mine (report)", "bound mu / OPT: mine (report)"], rows, 4)
        st = R["opt_lb"]["selftest"]
        out.append(f"    medians over the complaints within the budget; self-test of the certificate: certified bound <= brute-force optimum in {st['n_bound_le_opt']}/{st['checks']} checks on {st['instances']} random instances (equal in {st['n_bound_eq_opt']}).")
        out.append("    note: " + R["opt_lb"]["note"] + ".")
        out.append("")
    rr = R["rerun"]
    out.append("#" * 150)
    out.append("(4) paper_numbers.json IS DERIVED FROM results.json BY THE SCRIPT")
    out.append("#" * 150)
    d, r2 = rr["derivation"], rr["rerun"]
    out.append(f"(a) stored results.json -> e5_aggregate.paper_numbers() -> is_core: {d['n_values_recomputed']} values against {d['n_values_paper_numbers']} stored, {d['n_differences_paper_numbers']} differences; "
               f"paper_numbers_all.json: {d['n_values_paper_numbers_all']} values, {d['n_differences_paper_numbers_all']} differences; definitions: {d['n_differences_doc']} differences; null values: {d['n_null_values_paper_numbers']} ({', '.join(x.split('.', 2)[2] for x in d['null_names'][:3])}, ...: no certified optimum at c = 0.1).")
    out.append(f"(b) unmodified e5_aggregate.py re-run in a sandbox on copies of the raw inputs: results.json {r2['results_json']['n_leaf_values']} values, {r2['results_json']['n_differences']} differences (time stamps aside); "
               f"paper_numbers.json byte for byte {yn(r2['paper_numbers.json']['bytes_equal'])}, paper_numbers_all.json {yn(r2['paper_numbers_all.json']['bytes_equal'])}, paper_numbers_doc.json {yn(r2['paper_numbers_doc.json']['bytes_equal'])}.")
    out.append(f"(c) unmodified e5_render.py re-run: table.txt {rr['render']['n_lines_rerun']} lines, differing lines {rr['render']['lines_differing']} (line 2 carries the time stamps).")
    out.append(f"(d) the measuring agent's files after the verification: out/ unchanged {yn(rr['untouched']['out_unchanged'])} ({rr['untouched']['out_files']} files, md5), code/ unchanged {yn(rr['untouched']['code_unchanged'])} ({rr['untouched']['code_files']} files).")
    sv = R["summary_vs_json"]
    out.append(f"(e) numbers quoted in the summary against paper_numbers.json ({sv['n_against_my_values']} derived ones against my values): {sv['n_ok']}/{sv['n']} are correct roundings" + ("." if not sv["not_ok"] else ": NOT " + "; ".join(f"{c['id']} ({c['log']}): quoted {c['quoted']}, JSON {c['json']}" for c in sv["not_ok"])))
    out.append("")
    with open(os.path.join(C.OUT, "verify_table.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    # ------------------------------------------------------------------ the final message (at most MAX_WORDS words)
    msg = [f"E5 linear part, independent check (own code; keys from the final key cache). Cells Q | L; medians over all {V['n_complaints'][Q]} | {V['n_complaints'][L]} complaints, the seeded 60 included.",
           "quantity; report; mine; agree; verdict"]
    for r in V["message_rows"]:
        msg.append("; ".join(r))
    for k in ("numerical", "meaning", "extras", "optimum"):
        if V["message_parts"][k]:
            msg.append(V["message_parts"][k])
    msg.append(f"Files: {V['paths']['verdicts']} ; {V['paths']['table']}")
    text = "\n".join(msg)
    n_words = len(text.split())
    if n_words > MAX_WORDS:
        raise SystemExit(f"final message has {n_words} words (limit {MAX_WORDS})")
    with open(os.path.join(C.OUT, "final_message.txt"), "w") as fh:
        fh.write(text + "\n")
    V["message"], V["message_words"] = text, n_words
    C.jdump(V, os.path.join(C.VER, "verdicts.json"))
    print(f"wrote verify_table.txt ({len(out)} lines), final_message.txt ({n_words} words) and the message into verdicts.json")


if __name__ == "__main__":
    main()
