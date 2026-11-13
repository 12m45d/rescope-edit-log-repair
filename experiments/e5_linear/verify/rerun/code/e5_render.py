"""E5 rendering: out/results.json + out/paper_numbers.json -> out/table.txt (plain text; every number comes from the JSON).

usage: python e5_render.py
"""
from __future__ import annotations

import os
import time

import e5_common as C

W = 190


def f(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, str):
        return x
    if isinstance(x, int) or (isinstance(x, float) and float(x).is_integer() and abs(x) < 1e15 and nd == 0):
        return str(int(x))
    return f"{x:.{nd}f}"


def cell(d, nd=3, rng=True):
    """median [q25, q75] (min .. max)"""
    if d is None or d.get("n", 0) == 0:
        return "-"
    s = f"{f(d['median'], nd)} [{f(d['q25'], nd)}, {f(d['q75'], nd)}]"
    return s + (f" ({f(d['min'], nd)} .. {f(d['max'], nd)})" if rng else "")


def icell(d, rng=True):
    """the same for integer-valued statistics (medians / quartiles may be half-integers)"""
    if d is None or d.get("n", 0) == 0:
        return "-"
    g = lambda x: (str(int(x)) if float(x).is_integer() else f"{x:.1f}")       # noqa: E731
    s = f"{g(d['median'])} [{g(d['q25'])}, {g(d['q75'])}]"
    return s + (f" ({g(d['min'])} .. {g(d['max'])})" if rng else "")


def table(header, rows, indent=0):
    rows = [[str(c) for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(header)]
    pad = " " * indent
    out = [pad + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(header)).rstrip(),
           pad + "  ".join("-" * w[i] for i in range(len(header)))]
    out += [pad + "  ".join(r[i].ljust(w[i]) for i in range(len(header))).rstrip() for r in rows]
    return out


def hist_s(h):
    return "{" + ", ".join(f"{k}: {v}" for k, v in h.items()) + "}" if h else "{}"


def quoted(res, P):
    """The numbers that the real-log sentences of the three results need, both logs side by side."""
    names = list(res["logs"])
    rows = []

    def add(label, key, nd=2, sub=None):
        vals = []
        for n in names:
            v = P.get(key.format(log=n))
            vals.append(f(v, nd) if not isinstance(v, bool) else f(v))
        rows.append([label] + vals + [key.format(log="<log>")])

    add("complaints (REPAIR-pool D1, decoder v2)", "conditioning.{log}.n_complaints", 0)
    add("  image / text", "conditioning.{log}.n_image", 0)
    rows[-1][1:1 + len(names)] = [f"{P[f'conditioning.{n}.n_image']} / {P[f'conditioning.{n}.n_text']}" for n in names]
    rows[-1][-1] = "conditioning.<log>.n_image, .n_text"
    add("Prop. cond: Gamma/gamma at the last prompt key, median over complaints", "conditioning.{log}.median_Gamma_over_gamma")
    add("  minimum .. maximum over complaints", "conditioning.{log}.min_Gamma_over_gamma")
    rows[-1][1:1 + len(names)] = [f"{f(P[f'conditioning.{n}.min_Gamma_over_gamma'], 2)} .. {f(P[f'conditioning.{n}.max_Gamma_over_gamma'], 2)}" for n in names]
    rows[-1][-1] = "conditioning.<log>.min_/max_Gamma_over_gamma"
    add("  one value for every complaint (log constant)", "conditioning.{log}.Gamma_over_gamma_is_one_value_for_all_complaints")
    add("  gamma (hard instances: gamma <= 1/T)", "conditioning.{log}.median_gamma", 3)
    add("  Gamma", "conditioning.{log}.median_Gamma", 2)
    add("  sqrt(gamma/Gamma): LF at c is within OPT at c x this", "conditioning.{log}.median_tolerance_factor", 3)
    add("  stacked answer positions: median Gamma/gamma", "conditioning.{log}.stacked.median_Gamma_over_gamma")
    add("  stacked: first / third quartile", "conditioning.{log}.stacked.q25_Gamma_over_gamma")
    rows[-1][1:1 + len(names)] = [f"{f(P[f'conditioning.{n}.stacked.q25_Gamma_over_gamma'], 2)} / {f(P[f'conditioning.{n}.stacked.q75_Gamma_over_gamma'], 2)}" for n in names]
    rows[-1][-1] = "conditioning.<log>.stacked.q25_/q75_Gamma_over_gamma"
    add("share of pairs with <w_i, w_j> < 0, median (last key)", "conditioning.{log}.median_share_negative_pairs", 3)
    add("  stacked", "conditioning.{log}.stacked.median_share_negative_pairs", 3)
    add("cancellation-free complaints (no negative pair), share", "conditioning.{log}.share_cancellation_free", 3)
    add("most obtuse pairwise cosine, median (last key)", "conditioning.{log}.median_most_obtuse_cos", 3)
    add("vertex cover number nu of the negative graph, median (exact)", "conditioning.{log}.median_nu", 1)
    add("  minimum over complaints", "conditioning.{log}.min_nu", 0)
    for t in C.TOLS:
        k = "rollback.{log}.c" + str(t) + "."
        add(f"c = {t}: largest-first deletions, median (last key)", k + "median_lf", 1)
        add(f"c = {t}: lower bound of Prop. cond, median", k + "median_lb", 1)
        add(f"c = {t}: LF / lower bound, median", k + "median_lf_over_lb", 3)
        add(f"c = {t}: FO-greedy / lower bound, median", k + "median_fo_over_lb", 3)
        add(f"c = {t}: BI-greedy / lower bound, median", k + "median_bi_over_lb", 3)
        add(f"c = {t}: optimum code within 40 CPU min: complaints reached", k + "opt_n_complaints", 0)
        add(f"c = {t}:   certified optimum for", k + "opt_n_certified", 0)
        for q, lab in (("lf", "LF"), ("fo", "FO"), ("bi", "BI")):
            add(f"c = {t}:   {lab} / OPT, median over certified", k + f"opt_median_{q}_over_opt_certified", 3)
            add(f"c = {t}:   {lab} / OPT, median over reached, at least", k + f"opt_median_{q}_over_opt_lo", 3)
            add(f"c = {t}:   {lab} / OPT, median over reached, at most", k + f"opt_median_{q}_over_opt_hi", 3)
        add(f"c = {t}:   lower bound of Prop. cond / OPT, median, at least", k + "opt_median_lb_over_opt_lo", 3)
        add(f"c = {t}:   lower bound of Prop. cond / OPT, median, at most", k + "opt_median_lb_over_opt_hi", 3)
        add(f"c = {t}: every finished optimum job: complaints", k + "optall_n_complaints", 0)
        add(f"c = {t}:   BI / OPT, median, at least", k + "optall_median_bi_over_opt_lo", 3)
        add(f"c = {t}:   BI / OPT, median, at most", k + "optall_median_bi_over_opt_hi", 3)
        add(f"c = {t}: complaints needing > 50 deletions, at least", k + "n_need_more_than_50_certain", 0)
        add(f"c = {t}: complaints needing > 50 deletions, at most", k + "n_need_more_than_50_possible", 0)
    add("stacked, c = 0.1: LF / lower bound, median", "rollback.{log}.stacked.c0.1.median_lf_over_lb", 3)
    add("Thm. capacity: T", "capacity.{log}.T", 0)
    add("  key dimension d_in", "capacity.{log}.d_in", 0)
    add("  group size (whole request sequence), minimum", "capacity.{log}.group_size_min", 0)
    add("  group size, maximum", "capacity.{log}.group_size_max", 0)
    add("  group size (needed set), minimum", "capacity.{log}.needed_set.group_size_min", 0)
    add("  group size (needed set), maximum", "capacity.{log}.needed_set.group_size_max", 0)
    add("  shared-prefix dimension z used", "capacity.{log}.prefix_z", 0)
    add("  ... of which bitwise shared by all T sequences", "capacity.{log}.prefix_z_bitwise", 0)
    add("  s*_Z under GP/Z (a = 1)", "capacity.{log}.sstar_Z", 0)
    add("  predicted protectable edits T - s*_Z", "capacity.{log}.kept_pred", 0)
    add("  measured: edits protected for every complaint", "capacity.{log}.kept_feasible", 0)
    add("  predicted with z = 0", "capacity.{log}.kept_pred_z0", 0)
    add("  needed sets: predicted / measured", "capacity.{log}.needed_set.kept_pred", 0)
    rows[-1][1:1 + len(names)] = [f"{P.get(f'capacity.{n}.needed_set.kept_pred')} / {P.get(f'capacity.{n}.needed_set.kept_feasible')}" for n in names]
    rows[-1][-1] = "capacity.<log>.needed_set.kept_pred, .kept_feasible"
    for pn, lab in (("K_ans", "only the edits' answer keys"), ("K_last", "only the edits' last prompt keys")):
        add(f"  protecting {lab}: rank", "capacity.{log}." + pn + ".rank", 0)
        add("    edits protected (law / measured)", "capacity.{log}." + pn + ".kept_pred", 0)
        rows[-1][1:1 + len(names)] = [f"{P.get(f'capacity.{n}.{pn}.kept_pred')} / {P.get(f'capacity.{n}.{pn}.kept_measured')}" for n in names]
        rows[-1][-1] = f"capacity.<log>.{pn}.kept_pred, .kept_measured"
        add("    complaints repairable", "capacity.{log}." + pn + ".n_complaints_repairable", 0)
    return table(["quantity"] + names + ["name in paper_numbers.json"], rows)


def log_block(name, L):
    out = []
    ll = L["log_level"]
    A = L["A"]
    kc = L["keycache_check"]
    out += ["#" * W, f"{name}  ({L['short']}; T = {ll['T']} edits in {ll['n_batches']} batches, d_in = {ll['d_in']}, d_out = {ll['d_out']})", "#" * W]
    out.append(f"complaints: {L['n_complaints']} REPAIR-pool D1 ({L['n_by_ptype']['image']} image, {L['n_by_ptype']['text']} text); answer positions A: "
               f"all {icell(A['all'])}, image {icell(A['image'])}, text {icell(A['text'])}; last prompt key bitwise equal to the key cache "
               f"{kc['n_last_key_bitwise_equal']}/{kc['n_complaints']}; complaints whose teacher-forced last prompt position is one token after the decode's: {kc['n_lp_shift']}")
    out.append("")
    out.append("(1) CONDITIONING AND CANCELLATION AT THE COMPLAINT KEY  (Ghat = Gram matrix of the contributions scaled to unit diagonal; gamma <= Gamma its extreme eigenvalues)")
    rows = []
    for kd in C.KEYDEFS:
        for sel in ("all",) + C.PTYPES:
            c = L["conditioning"][kd][sel]
            rows.append([kd, sel, c["n"], cell(c["gamma"]), cell(c["Gamma"], 2), cell(c["ratio"], 2), cell(c["energy"], 3, False), cell(c["frac_neg"]),
                         cell(c["min_cos"]), icell(c["nu"]), f"{c['n_cancellation_free']}/{c['n']}"])
    out += table(["key", "split", "n", "gamma", "Gamma", "Gamma / gamma", "||D||^2 / sum ||w_i||^2", "share of pairs <w_i, w_j> < 0", "most obtuse cosine",
                  "nu = vertex cover of negative graph", "canc.-free"], rows, 4)
    cl = L["conditioning"]["last"]["all"]
    cs = L["conditioning"]["stacked"]["all"]
    out.append(f"    log level: cosine matrix of the write vectors u_i has gamma = {f(ll['gamma_U'])}, Gamma = {f(ll['Gamma_U'], 2)}, Gamma / gamma = {f(ll['ratio_U'], 2)}. At ONE key w_i = c_i u_i, so Ghat is")
    out.append(f"      that matrix up to the signs of the coordinates c_i: the same spectrum for EVERY complaint (largest relative deviation {ll['last_identity_max_rel_dev']:.1e}; distinct values of Gamma / gamma: "
               f"{cl['n_distinct_ratio_values_rel_1e-9']}).")
    out.append(f"    reference: the hard instances of Thm. thm:rollback-hard have gamma <= 1/T = {f(1.0 / ll['T'], 2)}; gamma > 0 for {cl['n_gamma_positive']}/{cl['n']} (last), {cs['n_gamma_positive']}/{cs['n']} (stacked) complaints;"
               f" sqrt(gamma / Gamma) = {f(cl['tolerance_factor_sqrt_gamma_over_Gamma']['median'])} (last), median {f(cs['tolerance_factor_sqrt_gamma_over_Gamma']['median'])} (stacked).")
    out.append(f"    signs: edits with a negative coordinate <psi_i, k_q> at the last key {icell(cl['n_neg_coord'])} of {ll['T']}; pairs of write vectors with <u_i, u_j> < 0: {f(100 * ll['share_pairs_cos_u_negative'], 1)} %"
               f" (most obtuse cosine {f(ll['min_cos_u'])}); nu exact for {cl['n_nu_exact']}/{cl['n']} and {cs['n_nu_exact']}/{cs['n']} complaints.")
    out.append(f"    norms: max ||w_i|| / min ||w_i|| {cell(cl['norm_spread'], 0)} (last), {cell(cs['norm_spread'], 1)} (stacked); edits holding half of sum ||w_i||^2: {icell(cl['n_edits_for_half_w2'])} (last);"
               f" zero contributions: {cl['n_zero_contributions']}.")
    out.append(f"    factor form against the applied weight difference, ||(W_E - W_0) K - D|| / ||D||: {cell(cl['fac_vs_applied'], 4)} (last), {cell(cs['fac_vs_applied'], 4)} (stacked).")
    out.append("")
    out.append(f"(2a) ROLLBACK BY RULE  (deletions out of T = {ll['T']}; criterion ||r|| <= c ||D||, p = 2).  LF = largest-norm-first; FO = first-order greedy (delete the kept edit maximising <r, w_i> for the")
    out.append("     current residual r); BI = best-improvement greedy (minimising ||r - w_i||); FO-static = Track C's fixed order by <D, w_i>; LB = max(1, mu(c^2 ||D||^2 / gamma)) <= OPT_c (Prop. prop:cond);")
    out.append("     LF guarantee = mu(c^2 ||D||^2 / Gamma) >= LF; LB-tri = triangle-inequality bound.  Ratios per complaint, then median [quartiles] (min .. max).")
    rows = []
    for kd in C.KEYDEFS:
        for t in C.TOLS:
            for sel in ("all",) + C.PTYPES:
                r = L["rollback"][kd][str(t)]["rules"][sel]
                rows.append([kd, t, sel, r["n"], icell(r["lf"]), icell(r["fo"]), icell(r["bi"]), icell(r["fo_static"], False), icell(r["lb1"]), icell(r["lf_guar"], False),
                             icell(r["lb_tri"], False), cell(r["lf_over_lb1"]), cell(r["fo_over_lb1"], 3, False), cell(r["bi_over_lb1"]), r["n_lb_cond_zero"]])
    out += table(["key", "c", "split", "n", "LF", "FO", "BI", "FO-static", "LB (Prop. cond)", "LF guarantee", "LB-tri", "LF / LB", "FO / LB", "BI / LB", "mu = 0"], rows, 4)
    notes = []
    for kd in C.KEYDEFS:
        for t in C.TOLS:
            r = L["rollback"][kd][str(t)]["rules"]["all"]
            da = {k: v for k, v in r["n_delete_all"].items() if v}
            if da:
                notes.append(f"{kd}, c = {t}: " + ", ".join(f"{k} deletes all {ll['T']} edits on {v} complaint(s)" + (f" {r['groups_delete_all'][k]}" if k in r["groups_delete_all"] else "") for k, v in da.items()))
    out.append("    a rule deletes ALL edits: " + ("; ".join(notes) if notes else "never"))
    r0 = L["rollback"]["last"]
    out.append("    tolerance at which Prop. prop:cond compares LF with the optimum, c sqrt(gamma / Gamma): "
               + "; ".join(f"c = {t}: {f(r0[str(t)]['tolerance_of_guarantee']['median'], 4)} (last), {f(L['rollback']['stacked'][str(t)]['tolerance_of_guarantee']['median'], 4)} (stacked, median)" for t in C.TOLS))
    out.append(f"    checks: LF > LF guarantee on {sum(L['rollback'][kd][str(t)]['rules']['all']['n_lf_gt_lf_guar'] for kd in C.KEYDEFS for t in C.TOLS)} instances; lower bound > best rule on "
               f"{sum(L['rollback'][kd][str(t)]['rules']['all']['n_lb_gt_best_rule'] for kd in C.KEYDEFS for t in C.TOLS)} instances (both must be 0).")
    out.append("")
    orun = L["optimum_run"]
    out.append("(2b) AGAINST THE OPTIMUM OF THE LINEAR SURROGATE  (existing Track C code, unchanged: exhaustive search over |R| <= 4, then outer-approximation MILP (HiGHS) on the QCR-convexified")
    out.append(f"     constraint; MILP limit {f(orun['milp_time_limit_s'], 0)} s per instance; seeded order, seed {orun['seed']}, image / text interleaved; 'last' first, then 'stacked').  OPT_c in [LB*, UB*]: LB* = max(LB, LB-tri, MILP bound),")
    out.append("     UB* = fewest deletions of any feasible set found (rules, local search, MILP); certified iff LB* = UB*.  Over all reached complaints: rule / UB* <= rule / OPT_c <= rule / LB*.")
    out.append(f"     budget: {f(orun['budget_cpu_min'], 0)} minutes of CPU time per log ({orun['workers']} single-threaded workers).  CPU time spent on this log: {f(orun['cpu_min_spent'], 1)} core-minutes"
               + (" -- OVER BUDGET." if orun["over_budget"] else " (within budget)."))
    if orun["stopped_run"]:
        sr = orun["stopped_run"]
        out.append(f"       a first run was stopped at {sr['stopped']} after {f(sr['wall_s'] / 60, 1)} min of wall-clock and {f(sr['cpu_s_total'] / 60, 1)} core-minutes ({sr['n_rows_finished']} jobs finished): {sr['reason']}.")
    for r in orun["runs"] or []:
        out.append(f"       run of {r['start']}: {r['n_jobs_run']} jobs (+{r['n_jobs_copied']} copied) in {f(r['wall_s'] / 60, 1)} min of wall-clock, {f(r['cpu_s_children'] / 60, 1)} core-minutes of all workers ({f(r['cpu_s_jobs'] / 60, 1)} in jobs)"
                   + ("; INTERRUPTED" if r.get("interrupted") else "") + ".")
    out.append(f"     'within budget' = the first jobs of the job sequence whose CPU seconds sum to at most {f(orun['budget_cpu_s_for_jobs'], 0)} s: {orun['n_jobs_within_budget']['last']} complaints with the key 'last', {orun['n_jobs_within_budget']['stacked']} with 'stacked'"
               f" ({f(orun['cpu_min_jobs_within_budget'], 1)} core-minutes); 'all finished' = every job that finished: {orun['n_jobs_finished']['last']} ('last'), {orun['n_jobs_finished']['stacked']} ('stacked'); errors {len(orun['errors'])}.")
    rows = []
    blocks = [(kd, ok, lab) for kd in C.KEYDEFS for ok, lab in (("optimum", "within budget"), ("optimum_all", "all finished"))]
    for kd, ok, lab in blocks:
        for t in C.TOLS:
            o = L["rollback"][kd][str(t)].get(ok, {}).get("all", {"n_reached": 0})
            if not o["n_reached"]:
                rows.append([kd, lab, t, 0] + ["-"] * 11)
                continue
            rows.append([kd, lab, t, f"{o['n_reached']} ({L['rollback'][kd][str(t)][ok]['image']['n_reached']} + {L['rollback'][kd][str(t)][ok]['text']['n_reached']})",
                         f"{o['n_certified']} ({f(100 * o['share_certified'], 0)} %)", hist_s(o["gap_hist"]),
                         f"{icell(o['LB'], False)} / {icell(o['UB'], False)}", icell(o["opt_certified"], False),
                         f"{f(o['lf_over_opt_certified']['median'])} ({f(o['lf_over_opt_certified']['max'])})" if o["n_certified"] else "-",
                         f"{f(o['fo_over_opt_certified']['median'])} ({f(o['fo_over_opt_certified']['max'])})" if o["n_certified"] else "-",
                         f"{f(o['bi_over_opt_certified']['median'])} ({f(o['bi_over_opt_certified']['max'])})" if o["n_certified"] else "-",
                         f"{f(o['lf_over_opt_lo']['median'])} - {f(o['lf_over_opt_hi']['median'])} (<= {f(o['lf_over_opt_hi']['max'])})",
                         f"{f(o['fo_over_opt_lo']['median'])} - {f(o['fo_over_opt_hi']['median'])} (<= {f(o['fo_over_opt_hi']['max'])})",
                         f"{f(o['bi_over_opt_lo']['median'])} - {f(o['bi_over_opt_hi']['median'])} (<= {f(o['bi_over_opt_hi']['max'])})",
                         f"{f(o['lb1_over_opt_lo']['median'])} - {f(o['lb1_over_opt_hi']['median'])}"])
    out += table(["key", "jobs", "c", "reached (image + text)", "certified", "UB* - LB* histogram", "LB* / UB* median [q]", "OPT (certified)", "LF/OPT cert. med (max)", "FO/OPT cert.", "BI/OPT cert.",
                  "LF/OPT all: median in (max)", "FO/OPT all", "BI/OPT all", "LB(Prop.)/OPT all"], rows, 4)
    for kd in C.KEYDEFS:
        for ok, lab in (("optimum", "within budget"), ("optimum_all", "all finished")):
            bits = []
            for t in C.TOLS:
                o = L["rollback"][kd][str(t)][ok]["all"]
                if o["n_reached"]:
                    bits.append(f"c = {t}: BI = OPT on {o['n_bi_equals_opt_certified']}, FO = OPT on {o['n_fo_equals_opt_certified']}, LF = OPT on {o['n_lf_equals_opt_certified']} of {o['n_certified']} certified; "
                                f"Track C alone certifies {o['n_certified_by_trackc_alone']}; a rule beats its best set on {o['n_rule_bound_tighter_than_trackc']['ub']}; "
                                f"BI / FO-static differ from Track C's on {o['n_mismatch_trackc_bi']} / {o['n_mismatch_trackc_fo_static']}")
            if bits:
                out.append(f"    {kd}, {lab}: " + " | ".join(bits))
    out.append(f"    CPU seconds per job: {cell(orun['cpu_s_per_job'], 0)}.")
    out.append("")
    out.append("(2c) COMPLAINTS THAT NEED MORE THAN N DELETIONS  (OPT_c > N): at least #(LB* > N), at most #(UB* > N), with the optimum jobs within the budget; complaints without one enter with the bounds of (2a).")
    rows = []
    for kd in C.KEYDEFS:
        for t in C.TOLS:
            r = L["rollback"][kd][str(t)]["rules"]["all"]
            row = [kd, t, r["n"]]
            for N in C.NEED_MORE:
                q = r["need_more_than"][str(N)]
                row += [f"{q['n_certain']} .. {q['n_possible']}", f"{q['n_certain_bounds_only']} .. {q['n_possible_bounds_only']}",
                        f"{q['n_rule']['lf']} / {q['n_rule']['fo']} / {q['n_rule']['bi']}"]
            rows.append(row)
    hdr = ["key", "c", "n"]
    for N in C.NEED_MORE:
        hdr += [f"> {N}: at least .. at most", "(2a) bounds only", "LF / FO / BI delete more"]
    out += table(hdr, rows, 4)
    out.append("")
    cap = L["capacity"]
    out.append("(3) CAPACITY OF EXACT PROTECTION  (Thm. thm:capacity).  Protection group of edit i = keys of its own request sequence (prompt + target, teacher forced, B = 1); r_i = its numerical rank.")
    out.append("     Family: the n smallest groups (by sequence length, then edit id).  A complaint stays repairable iff (I - P) K_q keeps full column rank (singular values > 1e-9 max ||k_q||; fp64).")
    rows = []
    for g, s in cap["groups"].items():
        rows.append([g, s["definition"], icell(s["group_size"]), icell(s["group_rank"], False), f"{s['n_groups_full_rank']}/{s['T']}", s["n_keys_all_groups"], s["z_bitwise"], s["z_used"],
                     f"{s['n_prefix_classes']} {s['prefix_class_sizes']}", f(s["near_prefix_max_rel_diff"], 4)])
    out += table(["group", "definition", "size", "rank r_i", "full rank", "keys in all groups", "z bitwise", "z used (near-shared)", "prefix classes (bitwise) and sizes", "max rel. diff in prefix"], rows, 4)
    rows = []
    for g, s in cap["groups"].items():
        m = s["measured"]
        for zn in ("z0", "z_bitwise", "z_near"):
            z = s["law"][zn]
            rows.append([g, f"{zn} (z = {z['z']})", z["sstar_last"], z["kept_pred_last"], hist_s(z["kept_pred_stacked_hist"]), hist_s(m["n_cap_last_hist"]), hist_s(m["n_cap_stacked_hist"]),
                         f"{z['n_agree_measured_last']}/{s['n_complaints']}", f"{z['n_agree_measured_stacked']}/{s['n_complaints']}",
                         f"{z['n_rank_equal']}/{z['n_family_points']} (max dev {z['max_abs_dev']})", f"{z['rank_pred_at_kept']} / {m['rank_at_kept']}"])
        zc = s["law"]["z_near_per_prefix_class"]
        rows.append([g, f"per prefix class (z = {zc['z']})", "-", "-", "-", "-", "-", "-", "-", f"{zc['n_rank_equal']}/{zc['n_family_points']} (max dev {zc['max_abs_dev']})",
                     f"{zc['rank_pred_at_kept']} / {m['rank_at_kept']}"])
    out += table(["group", "law under GP/Z with", "s*_Z (a = 1)", "kept = T - s*_Z", "kept, a = A (stacked): {value: complaints}", "MEASURED kept per complaint, last {value: n}",
                  "MEASURED, stacked", "law = measured (last)", "(stacked)", "rank law = measured rank along the family", "rank at kept: law / measured"], rows, 4)
    for g, s in cap["groups"].items():
        m = s["measured"]
        out.append(f"    {g}: corrections that protect {m['kept_feasible_every_complaint']} whole groups exist for every one of the {s['n_complaints']} complaints (both key definitions); at n = "
                   f"{m['kept_feasible_every_complaint'] + 1} the span has rank {m['rank_at_next']} of {s['d_in']} and {m['n_feasible_last_at_next']} (last) / {m['n_feasible_stacked_at_next']} (stacked) complaints stay repairable;")
        out.append(f"        rank at kept {m['rank_at_kept']} (sum of group ranks {m['sum_group_rank_at_kept']}; free dimensions {m['free_dims_at_kept']}); escape ||(I - P) k_q|| / ||k_q|| of the last key at kept: {cell(m['escape_last_at_kept'], 3)};"
                   f" by probe type: image {hist_s(m['by_ptype']['image']['n_cap_last_hist'])}, text {hist_s(m['by_ptype']['text']['n_cap_last_hist'])};")
        out.append(f"        the kept groups lie in {s['n_prefix_classes_among_kept']} prefix class(es); rank V = A for {s['a_rank']['n_rankV_equals_A']}/{s['n_complaints']} and rank K_q = A for {s['a_rank']['n_rankKq_equals_A']}/{s['n_complaints']} complaints; "
                   f"rank kept = min(A, d - rank) at every n for {m['n_rank_kept_law_holds']}/{s['n_complaints']}; feasibility monotone in n for {m['n_monotone_last']} / {m['n_monotone_stacked']}; basis orthonormal to {m['orth_err_basis']:.1e}.")
    if "points" in cap:
        rows = []
        for pn, p in cap["points"].items():
            rows.append([pn, p["n_keys"], p["rank"], p["rank_pred_gp_z0"], icell(p["group_rank"], False), f"{p['kept_pred_last']} of {ll['T']}", f"{p['n_feasible_last']}/{p['n_complaints']}",
                         f"{p['n_feasible_stacked']}/{p['n_complaints']}", cell(p["escape_last"]), cell(p["escape_fro"], 3, False)])
        out.append("    protecting ONLY the edits' answer keys (K_ans: the answer-predicting positions of every edit sequence) or ONLY their last prompt keys (K_last), all T edits at once:")
        out += table(["keys", "n keys", "rank", "rank under GP (z = 0)", "group rank", "protectable (law)", "repairable complaints (last)", "(stacked)", "escape of the last complaint key",
                      "Frobenius escape of the stacked keys"], rows, 4)
    if "rq3_check" in cap:
        q = cap["rq3_check"]
        out.append(f"    existing RQ3 file (60 sampled complaints, whole sequences, stacked keys): ranks along the family equal {f(q['rank_along_family_equal'])} ({q['n_family_points_compared']} points), group ranks equal {f(q['group_rank_equal'])}, "
                   f"z equal {q['z_equal']}, measured capacity equal for {q['n_cap_equal']}/{q['n_common_complaints']} complaints, rank kept equal at every n for {q['n_rank_kept_equal_at_every_n']}/{q['n_common_complaints']}, "
                   f"point ranks equal {f(q['point_rank_equal'])} (RQ3: {q['rq3_point_rank']}).")
    out.append("")
    return out


def main():
    res = C.jload(os.path.join(C.OUT, "results.json"))
    P = C.jload(os.path.join(C.OUT, "paper_numbers_all.json"))
    core = C.jload(os.path.join(C.OUT, "paper_numbers.json"))
    out = [f"E5 (LINEAR PART) -- REAL-LOG STATISTICS FOR THE RESULTS ON ROLLBACK AND COMPENSATION (Thm. thm:rollback-hard, Prop. prop:cond, Thm. thm:capacity)",
           f"rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/e5_render.py from out/results.json (aggregated {res['generated']}) and out/paper_numbers_all.json; no number is typed by hand.",
           "Logs: the two development logs only.  Complaints: ALL decoder-v2 D1 complaints of the REPAIR pool (harness split by item, seed 20261109), split image / text by probe type.",
           "Contribution of edit i at a key k: w_i = u_i <psi_i, k> (factor form of the log, fp64); D = sum_i w_i.  Inner products <w_i, w_j> and coordinates <psi_i, k> are SIGNED; norms are magnitudes.",
           "Key definitions: 'last' = the last prompt position of the teacher-forced pre-edit answer y0 (ONE key; primary); 'stacked' = the A answer-predicting positions stacked (Frobenius norms; robustness).",
           "Cells 'a [b, c] (d .. e)': median [first quartile, third quartile] (minimum .. maximum) over the complaints.  All linear algebra in fp64 ('floating point'); nothing was tuned.",
           ""]
    out.append(f"QUOTED NUMBERS (the real-log sentences of the three results; names as in out/paper_numbers.json ({len(core)} names; optall_ names: out/paper_numbers_all.json, {len(P)} names; definitions: out/paper_numbers_doc.json))")
    out += quoted(res, P)
    out.append("")
    for name, L in res["logs"].items():
        out += log_block(name, L)
    out.append("NOTES AND ANOMALIES (generated by code/e5_aggregate.py from the numbers above)")
    for n in res.get("notes", []):
        out.append("  - " + n)
    out.append("")
    for name, L in res["logs"].items():
        v = L.get("verify")
        if not v:
            out.append(f"INDEPENDENT RECOMPUTATION ({name}): not available.")
            continue
        a = v["conditioning_rollback"]
        bits = []
        for kd in C.KEYDEFS:
            x = a[kd]
            neq = {t: {k: x["n"] - n for k, n in d.items() if n != x["n"]} for t, d in x["n_equal"].items()}
            neq = {t: d for t, d in neq.items() if d}
            bits.append(f"'{kd}' ({x['n']} complaints): gamma / Gamma / energy relative deviation <= {x['max_rel_dev_gamma']:.1e} / {x['max_rel_dev_Gamma']:.1e} / {x['max_rel_dev_energy']:.1e}, share of negative pairs and "
                        f"most obtuse cosine deviation <= {x['max_abs_dev_frac_neg']:.1e} / {x['max_abs_dev_min_cos']:.1e}, deletion counts and lower bound differing: {neq if neq else 'none'}; recomputed medians: Gamma / gamma "
                        f"{f(x['recomputed_medians']['ratio'], 2)}, share negative {f(x['recomputed_medians']['frac_neg'])}, LF / LB at c = 0.1 {f(x['recomputed_medians']['lf_over_lb_0.1'])}")
        out.append(f"INDEPENDENT RECOMPUTATION ({name}; code/e5_verify.py: explicit contribution vectors and LAPACK SVD instead of the Gram identity): " + "; ".join(bits) + ".")
        b = v.get("capacity")
        if b:
            k0, k1 = b[f"n={b['kept']}"], b[f"n={b['kept'] + 1}"]
            out.append(f"    capacity (one stacked key matrix, LAPACK singular values and pivoted QR instead of the incremental basis): n = {b['kept']}: rank {k0['rank_svd']} (stored {k0['rank_stored']}; pivoted-QR diagonal {k0['rank_qr_diag']}), "
                       f"smallest kept / first dropped singular value {k0['s_min_kept']:.3g} / {f(k0['s_first_dropped'], 6) if k0['s_first_dropped'] is None else format(k0['s_first_dropped'], '.3g')}, complaints repairable {k0['n_feasible_last']} (stored {k0['n_feasible_last_stored']}), escape of the last key: "
                       f"median {f(k0['escape_last_median'])} (stored {f(k0['escape_stored_median'])}), largest relative deviation {k0['max_rel_dev_escape']:.1e}; n = {b['kept'] + 1}: rank {k1['rank_svd']} (stored {k1['rank_stored']}).")
    out.append("")
    st = res.get("selftest")
    if st:
        r, v, k = st["rules"], st["vertex_cover"], st["capacity_law"]
        out.append("SELF-TEST (code/e5_selftest.py): " + ("ALL PASS" if st["all_pass"] else "FAILURES") + f".  Rules and bounds against the brute-force optimum on {r['instances']} random instances x 4 tolerances: "
                   f"LB <= OPT {r['lb_cond_le_opt']}/{r['checks']}, LB-tri <= OPT {r['lb_tri_le_opt']}/{r['checks']}, OPT <= every rule {r['opt_le_rules']}/{r['checks']}, LF <= LF guarantee {r['lf_le_guar']}/{r['checks']}, "
                   f"LF(c) <= OPT at c sqrt(gamma/Gamma) {r['prop_last_claim']}/{r['checks']};")
        out.append(f"    vertex cover number against networkx {v['equal_networkx']}/{v['graphs']} and brute force {v['equal_brute_force']}/{v['brute_force_graphs']}; BI and FO-static against Track C's functions on real Gram matrices: "
                   + ", ".join(f"{n} {q['bi']}/{q['n']} and {q['fo_static']}/{q['n']}" for n, q in st["ropt"].items())
                   + f"; capacity formula against the brute-force minimum sacrifice {k['opt_equals_T_minus_kept_pred']}/{k['instances']}.")
    with open(os.path.join(C.OUT, "table.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote table.txt ({len(out)} lines)")


if __name__ == "__main__":
    main()
