"""E5 aggregation: out/cond_rollback_<log>.json, out/opt_rows_<log>.jsonl, out/opt_meta_<log>.json,
out/capacity_<log>_{seq,need}.json, out/selftest.json and the RQ3 key-space files  ->  out/results.json and
out/paper_numbers.json.       usage: python e5_aggregate.py          (CPU, seconds; no number is typed here)

Conventions of every summary: 'dist' = n, min, q25, median, q75, max, mean over the complaints (numpy percentiles, linear
interpolation); splits 'all' / 'image' / 'text' (probe type); key definitions 'last' (primary) and 'stacked'.
Rollback, per tolerance c (criterion ||r|| <= c ||D||, p = 2, deletions out of T):
  rules      lf, fo (adaptive, the rule of Thm. thm:rollback-hard(c)), bi, fo_static (Track C's fixed order), best_rule
  bounds     lb_cond = mu(c^2 ||D||^2 / gamma) (Prop. prop:cond), lb1 = max(1, lb_cond) (OPT_c >= 1 for c < 1 and D != 0),
             lb_tri (triangle inequality), lb = max(lb1, lb_tri); lf_guar = mu(c^2 ||D||^2 / Gamma) >= lf
  ratios     rule / lb1 ("ratio to the lower bound of the proposition") and rule / lb
  optimum    Track C code on the complaints it reached: opt_lb <= OPT_c <= opt_ub.  Combined with the bounds above:
             LB* = max(lb, opt_lb), UB* = min(best_rule, opt_ub); certified iff LB* == UB*.
             ratio to the optimum: over the certified complaints rule / OPT; over ALL reached complaints the bracket
             rule / UB* <= rule / OPT <= rule / LB*, so median(rule / UB*) <= median(rule / OPT) <= median(rule / LB*).
             'optimum'      (primary) the jobs WITHIN THE BUDGET of 40 minutes of CPU time per log: the longest prefix of
                            the job sequence (seeded order; key definition 'last' first, then 'stacked') whose recorded
                            CPU seconds sum to at most 40 x 60 - 5 x 8 (5 s per worker reserved for loading the log);
             'optimum_all'  every job that finished, including jobs of a run that exceeded the budget before it was
                            stopped (optimum_run.over_budget).
  need > N   number of complaints with OPT_c > N lies between #(LB* > N) and #(UB* > N) (complaints that the optimum code
             did not reach contribute their rule / proposition bounds).
Capacity: see e5_capacity.py.  kept = number of protectable edits = T - (sacrifice).
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)

import os
import time
from collections import Counter

import numpy as np

import e5_common as C

RULES = ("lf", "fo", "bi", "fo_static")
BUDGET_CPU_S = C.OPT_BUDGET_MIN * 60.0 - C.OPT_INIT_S * C.OPT_WORKERS      # CPU seconds available to the jobs of one log


def hist(values) -> dict:
    """{value: count}, numeric values in increasing order, then the others by name."""
    def key(kv):
        k = kv[0]
        return (0, float(k), "") if isinstance(k, (int, float, np.integer, np.floating)) and not isinstance(k, bool) else (1, 0.0, str(k))
    return {str(k): int(v) for k, v in sorted(Counter(values).items(), key=key)}


def split(rows, sel):
    return rows if sel == "all" else [r for r in rows if r["ptype"] == sel]


# ----------------------------------------------------------------------------- task 1
def cond_summary(rows, kd) -> dict:
    out = {}
    for sel in ("all",) + C.PTYPES:
        v = [r["keydefs"][kd] for r in split(rows, sel)]
        s = {"n": len(v)}
        for k in ("gamma", "Gamma", "ratio", "energy", "frac_neg", "min_cos", "median_cos", "nu", "alpha_nonneg", "norm_spread",
                  "fac_vs_applied", "A", "n_edits_for_half_w2"):
            s[k] = C.dist([x.get(k) for x in v])
        if kd == "last":
            s["n_neg_coord"] = C.dist([x["n_neg_coord"] for x in v])
        s["tolerance_factor_sqrt_gamma_over_Gamma"] = C.dist([float(np.sqrt(x["gamma"] / x["Gamma"])) for x in v])
        s["n_cancellation_free"] = int(sum(bool(x["cancellation_free"]) for x in v))
        s["share_cancellation_free"] = s["n_cancellation_free"] / len(v) if v else None
        s["n_gamma_positive"] = int(sum(bool(x["gamma_positive"]) for x in v))
        s["n_zero_contributions"] = int(sum(x["n_zero_contributions"] for x in v))
        s["n_nu_exact"] = int(sum(bool(x["nu_exact"]) for x in v))
        s["n_distinct_ratio_values_rel_1e-9"] = int(len({round(float(np.log(x["ratio"])) / 1e-9) for x in v}))
        out[sel] = s
    return out


# ----------------------------------------------------------------------------- task 2
def merged_rows(rows, opt, kd, t, only=None):
    """Per complaint: rules, bounds and (where the optimum code reached it) its bracket, at key definition kd, tolerance t.
    only: a set of (group, keydef) jobs; the optimum rows of other jobs are ignored."""
    if only is not None:
        opt = {k: v for k, v in opt.items() if k in only}
    out = []
    for r in rows:
        x = r["keydefs"][kd]["tol"][str(t)]
        m = {"group": r["group"], "ptype": r["ptype"], "T": int(r["keydefs"][kd]["T"]), **{k: int(x[k]) for k in RULES}, "best_rule": int(x["best_rule"]),
             "lb_cond": int(x["lb_cond"]), "lb1": max(1, int(x["lb_cond"])), "lb_tri": int(x["lb_tri"]), "lf_guar": int(x["lf_guar"])}
        m["lb"] = max(m["lb1"], m["lb_tri"])
        o = opt.get((r["group"], kd))
        m["opt"] = None
        if o is not None and "tau" in o:
            q = o["tau"][str(t)]
            m["opt"] = {"lb": int(q["opt_lb"]), "ub": int(q["opt_ub"]), "certified_trackc": bool(q["certified"]), "how": q["how"],
                        "ls": int(q["ls"]), "fo_static_trackc": int(q["fo"]), "bi_trackc": int(q["bi"]),
                        "milp_status": q.get("milp", {}).get("status"), "milp_time": q.get("milp", {}).get("time"),
                        "kelley_converged": q.get("kelley", {}).get("converged"), "order_idx": o["order_idx"], "cpu": o["cpu"],
                        "copied": bool(o.get("copied_from_last"))}
        m["LB"] = max(m["lb"], m["opt"]["lb"]) if m["opt"] else m["lb"]
        m["UB"] = min(m["best_rule"], m["opt"]["ub"]) if m["opt"] else m["best_rule"]
        m["certified"] = bool(m["LB"] == m["UB"])
        out.append(m)
    return out


def rules_summary(ms) -> dict:
    out = {}
    for sel in ("all",) + C.PTYPES:
        v = split(ms, sel)
        s = {"n": len(v)}
        for k in RULES + ("best_rule", "lb_cond", "lb1", "lb_tri", "lb", "lf_guar"):
            s[k] = C.dist([x[k] for x in v])
        for k in RULES + ("best_rule",):
            s[f"{k}_over_lb1"] = C.dist([x[k] / x["lb1"] for x in v])
            s[f"{k}_over_lb"] = C.dist([x[k] / x["lb"] for x in v])
            s[f"{k}_minus_lb"] = C.dist([x[k] - x["lb"] for x in v])
        s["n_lb_cond_zero"] = int(sum(x["lb_cond"] == 0 for x in v))
        s["n_lf_gt_lf_guar"] = int(sum(x["lf"] > x["lf_guar"] for x in v))
        s["n_lb_gt_best_rule"] = int(sum(x["lb"] > x["best_rule"] for x in v))
        s["n_delete_all"] = {k: int(sum(x[k] == x["T"] for x in v)) for k in RULES}
        s["groups_delete_all"] = {k: [x["group"] for x in v if x[k] == x["T"]][:20] for k in ("lf", "fo", "bi")}
        s["best_rule_on_delete_all"] = {k: [x["best_rule"] for x in v if x[k] == x["T"]][:20] for k in ("lf", "fo", "bi")}
        s["n_rule_equal"] = {"fo_eq_bi": int(sum(x["fo"] == x["bi"] for x in v)), "lf_eq_best": int(sum(x["lf"] == x["best_rule"] for x in v)),
                             "fo_eq_best": int(sum(x["fo"] == x["best_rule"] for x in v)), "bi_eq_best": int(sum(x["bi"] == x["best_rule"] for x in v))}
        s["need_more_than"] = {str(N): {"n_rule": {k: int(sum(x[k] > N for x in v)) for k in RULES},
                                        "n_certain_bounds_only": int(sum(x["lb"] > N for x in v)),
                                        "n_possible_bounds_only": int(sum(x["best_rule"] > N for x in v)),
                                        "n_certain": int(sum(x["LB"] > N for x in v)),
                                        "n_possible": int(sum(x["UB"] > N for x in v))} for N in C.NEED_MORE}
        out[sel] = s
    return out


def opt_summary(ms) -> dict:
    """Statistics over the complaints that the optimum code reached (ms: merged rows with m['opt'] set)."""
    out = {}
    for sel in ("all",) + C.PTYPES:
        v = [x for x in split(ms, sel) if x["opt"]]
        s = {"n_reached": len(v)}
        if not v:
            out[sel] = s
            continue
        cert = [x for x in v if x["certified"]]
        s["n_certified"] = len(cert)
        s["share_certified"] = len(cert) / len(v)
        s["n_certified_by_trackc_alone"] = int(sum(x["opt"]["certified_trackc"] for x in v))
        s["how"] = hist([x["opt"]["how"] for x in v])
        s["gap_hist"] = hist([x["UB"] - x["LB"] for x in v])
        s["gap_hist_trackc"] = hist([x["opt"]["ub"] - x["opt"]["lb"] for x in v])
        s["LB"], s["UB"] = C.dist([x["LB"] for x in v]), C.dist([x["UB"] for x in v])
        s["opt_certified"] = C.dist([x["UB"] for x in cert])
        s["n_rule_bound_tighter_than_trackc"] = {"ub": int(sum(x["best_rule"] < x["opt"]["ub"] for x in v)),
                                                 "lb": int(sum(x["lb"] > x["opt"]["lb"] for x in v))}
        s["n_mismatch_trackc_bi"] = int(sum(x["opt"]["bi_trackc"] != x["bi"] for x in v))
        s["n_mismatch_trackc_fo_static"] = int(sum(x["opt"]["fo_static_trackc"] != x["fo_static"] for x in v))
        s["milp_status"] = hist([x["opt"]["milp_status"] for x in v])
        s["lb1_over_opt_certified"] = C.dist([x["lb1"] / x["UB"] for x in cert])
        s["opt_minus_lb1_certified"] = C.dist([x["UB"] - x["lb1"] for x in cert])
        for k in RULES + ("best_rule",):
            s[f"{k}_over_opt_certified"] = C.dist([x[k] / x["UB"] for x in cert])
            s[f"{k}_minus_opt_certified"] = C.dist([x[k] - x["UB"] for x in cert])
            s[f"n_{k}_equals_opt_certified"] = int(sum(x[k] == x["UB"] for x in cert))
            s[f"{k}_over_opt_lo"] = C.dist([x[k] / x["UB"] for x in v])            # rule / UB* <= rule / OPT
            s[f"{k}_over_opt_hi"] = C.dist([x[k] / x["LB"] for x in v])            # rule / OPT <= rule / LB*
            s[f"{k}_minus_opt_lo"] = C.dist([x[k] - x["UB"] for x in v])
            s[f"{k}_minus_opt_hi"] = C.dist([x[k] - x["LB"] for x in v])
        s["lb1_over_opt_lo"] = C.dist([x["lb1"] / x["UB"] for x in v])
        s["lb1_over_opt_hi"] = C.dist([x["lb1"] / x["LB"] for x in v])
        s["cpu_s"] = float(sum(x["opt"]["cpu"] for x in v))
        out[sel] = s
    return out


def budget_prefix(opt):
    """The jobs within the CPU budget: the longest prefix of the job sequence (key definition 'last' in the seeded
    order, then 'stacked' in the seeded order) whose recorded CPU seconds sum to at most BUDGET_CPU_S.  A copied row
    (one answer position: the same instance as its 'last' row) costs nothing and belongs to the prefix iff its 'last'
    row does and every job before it in the sequence is in the prefix.  -> (set of (group, keydef), CPU seconds)."""
    seq = []
    for kd in C.KEYDEFS:
        seq += sorted([o for (g, k), o in opt.items() if k == kd], key=lambda o: o["order_idx"])
    out, cpu, nxt = set(), 0.0, {kd: 0 for kd in C.KEYDEFS}
    for o in seq:
        kd = o["keydef"]
        if kd == "stacked" and nxt["last"] == 0:
            break
        if "tau" not in o or o["order_idx"] != nxt[kd] or cpu + o["cpu"] > BUDGET_CPU_S:
            break
        if o.get("copied_from_last") and (o["group"], "last") not in out:
            break
        cpu += o["cpu"]
        nxt[kd] += 1
        out.add((o["group"], kd))
    return out, cpu


# ----------------------------------------------------------------------------- task 3
def capacity_summary(name) -> dict:
    out = {"groups": {}}
    for g in ("seq", "need"):
        p = os.path.join(C.OUT, f"capacity_{name}_{g}.json")
        if not os.path.exists(p):
            continue
        d = C.jload(p)
        ek, f = d["edit_keys"], d["family"]
        T, din = d["T"], d["d_in"]
        A = d["complaints"]["A"]
        ncl, ncs = f["n_cap_last"], f["n_cap_stacked"]
        k_all = int(min(min(ncl), min(ncs)))
        ki = f["n"].index(k_all)
        s = {"definition": "whole request sequence (every position)" if g == "seq" else "needed set (positions 0 .. len - 2)",
             "T": T, "d_in": din, "n_complaints": len(A),
             "group_size": C.dist(ek["group_size"]), "group_rank": C.dist(ek["group_rank"]),
             "n_groups_full_rank": ek["group_full_rank"], "n_keys_all_groups": ek["n_keys_all_groups"],
             "z_bitwise": ek["z_bitwise"], "z_near": ek["z_near"], "z_used": ek["z_near"],
             "near_prefix_max_rel_diff": max(ek["near_prefix_max_rel_diff"][: ek["z_near"]]) if ek["z_near"] else None,
             "n_prefix_classes": ek["n_prefix_classes"], "prefix_class_sizes": ek["prefix_class_sizes"],
             "n_prefix_classes_among_kept": len(set(ek["prefix_class_along_order"][:k_all])),
             "family": "the n smallest groups (by sequence length, then edit id)",
             "saturated_at_n": f["saturated_at_n"], "n_family_steps": len(f["n"]) - 1,
             "measured": {"n_cap_last": C.dist(ncl), "n_cap_last_hist": hist(ncl), "n_cap_stacked": C.dist(ncs),
                          "n_cap_stacked_hist": hist(ncs), "kept_feasible_every_complaint": k_all,
                          "sacrifice_upper_bound": T - k_all, "rank_at_kept": f["rank"][ki],
                          "sum_group_rank_at_kept": f["sum_rank"][ki], "free_dims_at_kept": din - f["rank"][ki],
                          "escape_last_at_kept": C.dist(f["escape_last"][ki]),
                          "n_feasible_last_at_next": f["n_feasible_last"][ki + 1] if ki + 1 < len(f["n"]) else None,
                          "n_feasible_stacked_at_next": f["n_feasible_stacked"][ki + 1] if ki + 1 < len(f["n"]) else None,
                          "rank_at_next": f["rank"][ki + 1] if ki + 1 < len(f["n"]) else None,
                          "n_monotone_last": int(sum(f["monotone_last"])), "n_monotone_stacked": int(sum(f["monotone_stacked"])),
                          "n_rank_kept_law_holds": int(sum(f["rank_kept_law_holds"])), "orth_err_basis": f["orth_err_basis"],
                          "by_ptype": {p: {"n": int(sum(x == p for x in d["complaints"]["ptype"])),
                                           "n_cap_last_hist": hist([v for v, x in zip(ncl, d["complaints"]["ptype"]) if x == p]),
                                           "n_cap_stacked_hist": hist([v for v, x in zip(ncs, d["complaints"]["ptype"]) if x == p])}
                                       for p in C.PTYPES}},
             "a_rank": {"n_rankV_equals_A": int(sum(x == y for x, y in zip(d["complaints"]["a_rank_stacked"], A))),
                        "n_rankKq_equals_A": int(sum(x == y for x, y in zip(d["complaints"]["kq_rank"], A))),
                        "n_rankV_last_equals_1": int(sum(d["complaints"]["a_rank_last"])), "A": C.dist(A)},
             "law": {}}
        for zn, p in f["pred"].items():
            if "kept_pred_last" not in p:
                s["law"][zn] = {"z": p["z"], "n_rank_equal": p["n_rank_equal"], "n_family_points": len(f["n"]),
                                "max_abs_dev": p["max_abs_dev"], "rank_pred_at_kept": p["rank_pred"][ki], "note": p.get("note")}
                continue
            s["law"][zn] = {"z": p["z"], "kept_pred_last": p["kept_pred_last"], "sstar_last": p["sstar_last"],
                            "kept_pred_stacked_hist": hist(p["kept_pred_stacked"]),
                            "sstar_stacked_hist": hist([T - x for x in p["kept_pred_stacked"]]),
                            "n_agree_measured_last": int(sum(x == p["kept_pred_last"] for x in ncl)),
                            "n_agree_measured_stacked": int(sum(x == y for x, y in zip(ncs, p["kept_pred_stacked"]))),
                            "n_rank_equal": p["n_rank_equal"], "n_family_points": len(f["n"]), "max_abs_dev": p["max_abs_dev"],
                            "rank_pred_at_kept": p["rank_pred"][ki]}
        out["groups"][g] = s
        if "points" in d:
            out["points"] = {}
            for pn, p in d["points"].items():
                out["points"][pn] = {"n_keys": p["n_keys"], "rank": p["rank"], "rank_pred_gp_z0": p["rank_pred_gp"],
                                     "group_rank": C.dist(p["group_rank"]), "group_size_min_max": p["group_size_min_max"],
                                     "kept_pred_last": p["kept_pred"]["last"], "kept_pred_stacked_hist": hist(p["kept_pred"]["stacked"]),
                                     "n_feasible_last": int(sum(p["feasible_last"])), "n_feasible_stacked": int(sum(p["feasible_stacked"])),
                                     "n_complaints": len(p["feasible_last"]), "escape_last": C.dist(p["escape_last"]),
                                     "escape_fro": C.dist(p["escape_fro"]), "kept_measured": T,
                                     "note": "all T edits protected at these keys only (one protection set, no family)"}
    # cross-check with the existing RQ3 key-space file (60 sampled complaints, whole sequences, stacked keys)
    rq_p = os.path.join(C.RQ3, "out", f"theory_{name}.json")
    seq_p = os.path.join(C.OUT, f"capacity_{name}_seq.json")
    if not os.path.exists(rq_p) and os.path.exists(os.path.join(C.OUT, "results.json")):
        # ARTIFACT: the RQ3 file is of the exploratory phase and is not released.  The stored outcome of this cross-check is kept and labelled.
        prev = ((C.jload(os.path.join(C.OUT, "results.json")).get("logs") or {}).get(name) or {}).get("capacity", {}).get("rq3_check")
        if prev is not None:
            out["rq3_check"] = dict(prev, kept="from the shipped summary; input not released")
            print(f"[e5_aggregate] {name}: cross-check with the RQ3 file kept from the shipped summary (input not released: {rq_p})", flush=True)
    if os.path.exists(rq_p) and os.path.exists(seq_p):
        rq, d = C.jload(rq_p), C.jload(seq_p)
        f = d["family"]
        m = min(len(rq["kseq"]["rank"]), len(f["rank"]))
        idx = {g: j for j, g in enumerate(d["complaints"]["group"])}
        common = [(k, idx[g]) for k, g in enumerate(rq["complaints"]) if g in idx]
        out["rq3_check"] = {
            "file": rq_p, "n_rq3_complaints": len(rq["complaints"]), "n_common_complaints": len(common),
            "n_family_points_compared": m, "rank_along_family_equal": rq["kseq"]["rank"][:m] == f["rank"][:m],
            "group_rank_equal": rq["edit_keys"]["group_rank"] == d["edit_keys"]["group_rank"],
            "z_equal": [rq["edit_keys"]["z_bitwise_shared_prefix"] == d["edit_keys"]["z_bitwise"],
                        rq["edit_keys"]["z_near_shared_prefix_rel10pct"] == d["edit_keys"]["z_near"]],
            "n_cap_equal": int(sum(rq["kseq"]["n_cap_measured"][k] == f["n_cap_stacked"][j] for k, j in common)),
            "n_rank_kept_equal_at_every_n": int(sum(all(rq["kseq"]["per_c"][n][k]["rank_kept"] == f["rank_kept_stacked"][n][j]
                                                        for n in range(m)) for k, j in common)),
            "rq3_n_cap_hist": hist(rq["kseq"]["n_cap_measured"]),
            "rq3_kept_pred": {z: hist(p["n_cap_pred"]) for z, p in rq["kseq"]["pred"].items()},
            "rq3_point_rank": {k: rq["points"]["rank"][k] for k in ("K_last", "K_ans")},
            "point_rank_equal": all(rq["points"]["rank"][k] == d["points"][k]["rank"] for k in ("K_last", "K_ans")),
            "rq3_point_feasible": {k: int(sum(r["rank_kept"] == rq["A"][j] for j, r in enumerate(rq["points"]["per_c"][k])))
                                   for k in ("K_last", "K_ans")}}
    return out


# ----------------------------------------------------------------------------- paper numbers
def paper_numbers(res) -> dict:
    P = {}
    for name, L in res["logs"].items():
        c_last, c_stk = L["conditioning"]["last"]["all"], L["conditioning"]["stacked"]["all"]
        k = f"conditioning.{name}."
        P[k + "n_complaints"] = L["n_complaints"]
        P[k + "n_image"], P[k + "n_text"] = L["n_by_ptype"]["image"], L["n_by_ptype"]["text"]
        P[k + "T"] = L["log_level"]["T"]
        P[k + "one_over_T"] = 1.0 / L["log_level"]["T"]
        for tag, c in (("", c_last), ("stacked.", c_stk)):
            P[k + tag + "median_Gamma_over_gamma"] = c["ratio"]["median"]
            P[k + tag + "q25_Gamma_over_gamma"], P[k + tag + "q75_Gamma_over_gamma"] = c["ratio"]["q25"], c["ratio"]["q75"]
            P[k + tag + "min_Gamma_over_gamma"], P[k + tag + "max_Gamma_over_gamma"] = c["ratio"]["min"], c["ratio"]["max"]
            P[k + tag + "median_gamma"], P[k + tag + "median_Gamma"] = c["gamma"]["median"], c["Gamma"]["median"]
            P[k + tag + "min_gamma"] = c["gamma"]["min"]
            P[k + tag + "median_tolerance_factor"] = c["tolerance_factor_sqrt_gamma_over_Gamma"]["median"]
            P[k + tag + "median_share_negative_pairs"] = c["frac_neg"]["median"]
            P[k + tag + "min_share_negative_pairs"], P[k + tag + "max_share_negative_pairs"] = c["frac_neg"]["min"], c["frac_neg"]["max"]
            P[k + tag + "n_cancellation_free"] = c["n_cancellation_free"]
            P[k + tag + "share_cancellation_free"] = c["share_cancellation_free"]
            P[k + tag + "median_most_obtuse_cos"] = c["min_cos"]["median"]
            P[k + tag + "median_nu"], P[k + tag + "min_nu"], P[k + tag + "max_nu"] = c["nu"]["median"], c["nu"]["min"], c["nu"]["max"]
            P[k + tag + "median_energy"] = c["energy"]["median"]
        P[k + "Gamma_over_gamma_is_one_value_for_all_complaints"] = bool(c_last["n_distinct_ratio_values_rel_1e-9"] == 1)
        P[k + "median_n_edits_with_negative_coordinate"] = c_last["n_neg_coord"]["median"]
        P[k + "share_negative_pairs_of_write_vectors"] = L["log_level"]["share_pairs_cos_u_negative"]
        for kd, tag in (("last", ""), ("stacked", "stacked.")):
            for t in C.TOLS:
                R = L["rollback"][kd][str(t)]
                r, o = R["rules"]["all"], R["optimum"]["all"]
                k = f"rollback.{name}.{tag}c{t}."
                for q in ("lf", "fo", "bi"):
                    P[k + f"median_{q}"] = r[q]["median"]
                    P[k + f"median_{q}_over_lb"] = r[f"{q}_over_lb1"]["median"]
                    P[k + f"max_{q}_over_lb"] = r[f"{q}_over_lb1"]["max"]
                    P[k + f"n_{q}_deletes_all"] = r["n_delete_all"][q]
                P[k + "median_lb"], P[k + "min_lb"], P[k + "max_lb"] = r["lb1"]["median"], r["lb1"]["min"], r["lb1"]["max"]
                P[k + "median_lf_guarantee"] = r["lf_guar"]["median"]
                P[k + "median_best_rule"] = r["best_rule"]["median"]
                P[k + "median_best_rule_over_lb"] = r["best_rule_over_lb1"]["median"]
                for N in C.NEED_MORE:
                    P[k + f"n_need_more_than_{N}_certain"] = r["need_more_than"][str(N)]["n_certain"]
                    P[k + f"n_need_more_than_{N}_possible"] = r["need_more_than"][str(N)]["n_possible"]
                    P[k + f"n_lf_more_than_{N}"] = r["need_more_than"][str(N)]["n_rule"]["lf"]
                    P[k + f"n_bi_more_than_{N}"] = r["need_more_than"][str(N)]["n_rule"]["bi"]
                    P[k + f"optall_n_need_more_than_{N}_certain"] = R["need_more_than_with_all_reached"]["all"][str(N)]["n_certain"]
                    P[k + f"optall_n_need_more_than_{N}_possible"] = R["need_more_than_with_all_reached"]["all"][str(N)]["n_possible"]
                for pre, oo in (("opt_", o), ("optall_", R["optimum_all"]["all"])):
                    P[k + pre + "n_complaints"] = oo["n_reached"]
                    if not oo["n_reached"]:
                        continue
                    P[k + pre + "n_certified"] = oo["n_certified"]
                    P[k + pre + "median_certified"] = oo["opt_certified"]["median"]
                    P[k + pre + "median_lower_bound"], P[k + pre + "median_upper_bound"] = oo["LB"]["median"], oo["UB"]["median"]
                    for q in ("lf", "fo", "bi"):
                        P[k + pre + f"median_{q}_over_opt_certified"] = oo[f"{q}_over_opt_certified"]["median"]
                        P[k + pre + f"max_{q}_over_opt_certified"] = oo[f"{q}_over_opt_certified"]["max"]
                        P[k + pre + f"n_{q}_equals_opt_certified"] = oo[f"n_{q}_equals_opt_certified"]
                        P[k + pre + f"median_{q}_over_opt_lo"] = oo[f"{q}_over_opt_lo"]["median"]
                        P[k + pre + f"median_{q}_over_opt_hi"] = oo[f"{q}_over_opt_hi"]["median"]
                        P[k + pre + f"max_{q}_over_opt_hi"] = oo[f"{q}_over_opt_hi"]["max"]
                    P[k + pre + "median_lb_over_opt_lo"], P[k + pre + "median_lb_over_opt_hi"] = oo["lb1_over_opt_lo"]["median"], oo["lb1_over_opt_hi"]["median"]
        orun = L["optimum_run"]
        k = f"rollback.{name}.optimum_run."
        P[k + "budget_cpu_min"], P[k + "cpu_min_spent"], P[k + "over_budget"] = orun["budget_cpu_min"], orun["cpu_min_spent"], orun["over_budget"]
        P[k + "milp_time_limit_s"] = orun["milp_time_limit_s"]
        P[k + "n_complaints_within_budget"] = orun["n_jobs_within_budget"]["last"]
        P[k + "n_complaints_all_finished"] = orun["n_jobs_finished"]["last"]
        P[k + "n_complaints_stacked_within_budget"] = orun["n_jobs_within_budget"]["stacked"]
        cap = L["capacity"]
        k = f"capacity.{name}."
        if "seq" in cap["groups"]:
            s = cap["groups"]["seq"]
            z = s["law"]["z_near"]
            P[k + "T"], P[k + "d_in"], P[k + "n_complaints"] = s["T"], s["d_in"], s["n_complaints"]
            P[k + "group_size_min"], P[k + "group_size_max"] = int(s["group_size"]["min"]), int(s["group_size"]["max"])
            P[k + "group_size_median"] = s["group_size"]["median"]
            P[k + "group_rank_min"], P[k + "group_rank_max"] = int(s["group_rank"]["min"]), int(s["group_rank"]["max"])
            P[k + "prefix_z"], P[k + "prefix_z_bitwise"] = s["z_used"], s["z_bitwise"]
            P[k + "n_prefix_classes"] = s["n_prefix_classes"]
            P[k + "sstar_Z"], P[k + "kept_pred"] = z["sstar_last"], z["kept_pred_last"]
            P[k + "sstar_z0"], P[k + "kept_pred_z0"] = s["law"]["z0"]["sstar_last"], s["law"]["z0"]["kept_pred_last"]
            P[k + "kept_feasible"] = s["measured"]["kept_feasible_every_complaint"]
            P[k + "kept_feasible_min_per_complaint"] = int(min(s["measured"]["n_cap_last"]["min"], s["measured"]["n_cap_stacked"]["min"]))
            P[k + "kept_feasible_max_per_complaint"] = int(max(s["measured"]["n_cap_last"]["max"], s["measured"]["n_cap_stacked"]["max"]))
            P[k + "sacrifice_measured"] = s["measured"]["sacrifice_upper_bound"]
            P[k + "rank_at_kept"], P[k + "rank_pred_at_kept"] = s["measured"]["rank_at_kept"], z["rank_pred_at_kept"]
            P[k + "saturated_at_n"] = s["saturated_at_n"]
            P[k + "stacked.kept_pred_min"] = int(min(int(x) for x in z["kept_pred_stacked_hist"]))
            P[k + "stacked.kept_pred_max"] = int(max(int(x) for x in z["kept_pred_stacked_hist"]))
        if "need" in cap["groups"]:
            s = cap["groups"]["need"]
            z = s["law"]["z_near"]
            P[k + "needed_set.group_size_min"], P[k + "needed_set.group_size_max"] = int(s["group_size"]["min"]), int(s["group_size"]["max"])
            P[k + "needed_set.group_size_median"] = s["group_size"]["median"]
            P[k + "needed_set.sstar_Z"], P[k + "needed_set.kept_pred"] = z["sstar_last"], z["kept_pred_last"]
            P[k + "needed_set.kept_feasible"] = s["measured"]["kept_feasible_every_complaint"]
            P[k + "needed_set.rank_at_kept"], P[k + "needed_set.rank_pred_at_kept"] = s["measured"]["rank_at_kept"], z["rank_pred_at_kept"]
        for pn in ("K_ans", "K_last"):
            if pn in cap.get("points", {}):
                p = cap["points"][pn]
                P[k + f"{pn}.n_keys"], P[k + f"{pn}.rank"] = p["n_keys"], p["rank"]
                P[k + f"{pn}.kept_pred"], P[k + f"{pn}.kept_measured"] = p["kept_pred_last"], p["kept_measured"]
                P[k + f"{pn}.n_complaints_repairable"] = min(p["n_feasible_last"], p["n_feasible_stacked"])
                P[k + f"{pn}.median_escape_last_key"] = p["escape_last"]["median"]
    return P


DOC = {
    "conditioning.<log>.n_complaints / n_image / n_text": "REPAIR-pool D1 complaints (decoder v2) of the log, and their split by probe type",
    "conditioning.<log>.[stacked.]median_Gamma_over_gamma (q25_, q75_, min_, max_)": "Gamma / gamma of the Gram matrix of the contributions scaled to unit diagonal, over the complaints; no prefix: at the last prompt key (ONE key); 'stacked.': the A answer-predicting keys stacked",
    "conditioning.<log>.Gamma_over_gamma_is_one_value_for_all_complaints": "true iff the ratio at the last prompt key is the same number for every complaint (at one key it is the ratio of the cosine matrix of the write vectors u_i)",
    "conditioning.<log>.[stacked.]median_gamma / median_Gamma / min_gamma": "extreme eigenvalues (medians over the complaints; least gamma)",
    "conditioning.<log>.[stacked.]median_tolerance_factor": "median of sqrt(gamma / Gamma): Prop. prop:cond bounds largest-first at tolerance c by OPT at tolerance c times this factor",
    "conditioning.<log>.[stacked.]median_share_negative_pairs (min_, max_)": "share of the T (T - 1) / 2 pairs with <w_i, w_j> < 0 (signed), over the complaints",
    "conditioning.<log>.[stacked.]n_cancellation_free / share_cancellation_free": "complaints without a negative pair",
    "conditioning.<log>.[stacked.]median_most_obtuse_cos": "median over the complaints of the least pairwise cosine",
    "conditioning.<log>.[stacked.]median_nu / min_nu / max_nu": "vertex cover number of the graph of negative pairs (exact)",
    "conditioning.<log>.[stacked.]median_energy": "median of ||D||^2 / sum_i ||w_i||^2",
    "conditioning.<log>.median_n_edits_with_negative_coordinate": "median number of edits with <psi_i, k_q> < 0 at the last prompt key",
    "conditioning.<log>.share_negative_pairs_of_write_vectors": "share of pairs with <u_i, u_j> < 0 (log level)",
    "conditioning.<log>.T / one_over_T": "number of edits; the bound gamma <= 1/T of the hard instances",
    "rollback.<log>.[stacked.]c<c>.median_lf / median_fo / median_bi / median_best_rule": "deletions (of T) of largest-norm-first, first-order greedy (adaptive), best-improvement greedy, and of the best of the rules, median over all complaints, criterion ||r|| <= c ||D||",
    "rollback.<log>.[stacked.]c<c>.median_lb / min_lb / max_lb": "lower bound of Prop. prop:cond, max(1, mu(c^2 ||D||^2 / gamma)) <= OPT_c",
    "rollback.<log>.[stacked.]c<c>.median_<rule>_over_lb / max_<rule>_over_lb": "per-complaint ratio rule / lower bound: median and maximum over all complaints",
    "rollback.<log>.[stacked.]c<c>.median_lf_guarantee": "median of mu(c^2 ||D||^2 / Gamma), the upper bound of Prop. prop:cond on largest-first",
    "rollback.<log>.[stacked.]c<c>.n_<rule>_deletes_all": "complaints on which the rule deletes all T edits",
    "rollback.<log>.[stacked.]c<c>.n_need_more_than_<N>_certain / _possible": "number of complaints with OPT_c > N: at least / at most (bounds + optimum jobs within the budget); optall_: with every finished optimum job",
    "rollback.<log>.[stacked.]c<c>.n_lf_more_than_<N> / n_bi_more_than_<N>": "complaints on which the rule deletes more than N edits",
    "rollback.<log>.[stacked.]c<c>.opt_n_complaints / opt_n_certified": "complaints reached by the optimum code within the budget of 40 minutes of CPU time per log (seeded order) / of them with a certified optimum (LB* = UB*)",
    "rollback.<log>.[stacked.]c<c>.opt_median_certified": "median certified OPT_c",
    "rollback.<log>.[stacked.]c<c>.opt_median_lower_bound / opt_median_upper_bound": "medians of LB* and UB* (OPT_c in [LB*, UB*]) over the reached complaints",
    "rollback.<log>.[stacked.]c<c>.opt_median_<rule>_over_opt_certified (max_, n_<rule>_equals_opt_certified)": "rule / OPT_c over the certified complaints",
    "rollback.<log>.[stacked.]c<c>.opt_median_<rule>_over_opt_lo / _hi (opt_max_<rule>_over_opt_hi)": "medians of rule / UB* and rule / LB* over all reached complaints: the median of rule / OPT_c lies between them",
    "rollback.<log>.[stacked.]c<c>.opt_median_lb_over_opt_lo / _hi": "the same bracket for (lower bound of Prop. prop:cond) / OPT_c",
    "rollback.<log>.[stacked.]c<c>.optall_*": "the opt_ quantities over EVERY finished optimum job (for a log whose run exceeded the budget before it was stopped this is a longer prefix of the same seeded order)",
    "rollback.<log>.optimum_run.*": "budget (CPU minutes), CPU minutes spent, over_budget, MILP limit, complaints within the budget / finished",
    "capacity.<log>.T / d_in / n_complaints": "edits, key dimension, complaints",
    "capacity.<log>.group_size_min / _max / _median, group_rank_min / _max": "protection group = every position of the edit's request sequence (prompt + target): number of keys and numerical rank r_i",
    "capacity.<log>.prefix_z / prefix_z_bitwise / n_prefix_classes": "shared-prefix dimension z used (leading positions whose keys agree within 10 % in all sequences) / leading positions bitwise equal in all T sequences / number of bitwise-distinct copies of that prefix",
    "capacity.<log>.sstar_Z / kept_pred": "s*_Z of Thm. thm:capacity under GP/Z for a = 1 and T - s*_Z, the predicted number of protectable edits",
    "capacity.<log>.sstar_z0 / kept_pred_z0": "the same with z = 0 (plain general position)",
    "capacity.<log>.kept_feasible (kept_feasible_min_/max_per_complaint) / sacrifice_measured": "measured: the largest n such that a correction protecting the n smallest groups exists for EVERY complaint (both key definitions; per-complaint extremes) / T minus it",
    "capacity.<log>.rank_at_kept / rank_pred_at_kept / saturated_at_n": "rank of the protected span at n = kept_feasible, its GP/Z prediction, and the n at which the span fills the key space",
    "capacity.<log>.stacked.kept_pred_min / _max": "T - s*_Z with a = A (stacked complaint keys), extremes over the complaints",
    "capacity.<log>.needed_set.*": "the same with the needed set (positions 0 .. len - 2) as protection group",
    "capacity.<log>.K_ans.* / K_last.*": "protecting only the answer-predicting keys / only the last prompt key of every edit: number of keys, rank of their span, protectable edits by the law (z = 0) and measured (all T at once), complaints that stay repairable, median escape ||(I - P) k_q|| / ||k_q|| of the complaint's last prompt key",
}


STACKED_CORE = ("median_lf", "median_fo", "median_bi", "median_lb", "median_lf_over_lb", "median_fo_over_lb", "median_bi_over_lb")


def is_core(k: str) -> bool:
    """The names of paper_numbers.json (the numbers that the real-log sentences of the three results quote); every
    other name is in paper_numbers_all.json: the optimum quantities over jobs beyond the budget (optall_), per-rule
    counts, and the rollback statistics of the stacked keys other than the rule / bound medians."""
    if k.startswith(("conditioning.", "capacity.")) or ".optimum_run." in k:
        return True
    name = k.rsplit(".", 1)[1]
    if ".stacked." in k:
        return name in STACKED_CORE
    return not (name.startswith("optall_") or name.startswith(("n_lf_more_than", "n_bi_more_than")) or "_deletes_all" in name and not name.startswith("n_bi_")
                or name.startswith("opt_max_") or name.startswith("opt_n_") and name.endswith("_equals_opt_certified"))


def notes(res) -> list:
    """Generated remarks on anomalies (every number is read from `res`)."""
    out = []
    for name, L in res["logs"].items():
        ll, cl, cs = L["log_level"], L["conditioning"]["last"]["all"], L["conditioning"]["stacked"]["all"]
        if cl["n_distinct_ratio_values_rel_1e-9"] == 1:
            out.append(f"{name}: at ONE complaint key Gamma / gamma = {cl['ratio']['median']:.2f} for every one of the {cl['n']} complaints (gamma = {cl['gamma']['median']:.3f}, "
                       f"Gamma = {cl['Gamma']['median']:.2f}): w_i = <psi_i, k_q> u_i, so the scaled Gram matrix is the cosine matrix of the write vectors up to signs; "
                       f"a 'median over the complaints' is a constant of the log.  Stacked answer positions: median {cs['ratio']['median']:.2f} ({cs['ratio']['min']:.2f} .. {cs['ratio']['max']:.2f}).")
        out.append(f"{name}: cancellation-free complaints {cl['n_cancellation_free']}/{cl['n']} (last), {cs['n_cancellation_free']}/{cs['n']} (stacked); least vertex cover number {int(cl['nu']['min'])} (last), {int(cs['nu']['min'])} (stacked).")
        for kd in C.KEYDEFS:
            for t in C.TOLS:
                r = L["rollback"][kd][str(t)]["rules"]["all"]
                for q in ("lf", "fo", "bi"):
                    if r["n_delete_all"][q]:
                        out.append(f"{name}: {q.upper()} deletes all {ll['T']} edits on {r['n_delete_all'][q]} complaint(s) at c = {t}, key '{kd}': {r['groups_delete_all'][q]} "
                                   f"(best rule there: {[int(x) for x in r['best_rule_on_delete_all'][q]]}).")
                if r["n_delete_all"]["fo_static"]:
                    out.append(f"{name}: Track C's static first-order order (decreasing <D, w_i>, the 'fo' of rollback_opt/results.md) deletes all {ll['T']} edits on {r['n_delete_all']['fo_static']} "
                               f"of {r['n']} complaints at c = {t}, key '{kd}'; it is not the adaptive rule of Thm. thm:rollback-hard(c).")
        o = L["optimum_run"]
        if o["over_budget"]:
            out.append(f"{name}: the optimum code spent {o['cpu_min_spent']:.1f} core-minutes, more than the budget of {o['budget_cpu_min']:.0f}: {o['stopped_run']['reason'] if o['stopped_run'] else ''}.  "
                       f"Primary numbers use the first {o['n_jobs_within_budget']['last']} complaints of the seeded order ({o['cpu_min_jobs_within_budget']:.1f} core-minutes); all {o['n_jobs_finished']['last']} finished jobs are reported separately.")
        if not o["n_jobs_within_budget"]["stacked"]:
            out.append(f"{name}: the budget was used up by the key definition 'last' ({o['n_jobs_within_budget']['last']} complaints); the optimum code was not run on the stacked keys.")
        nocert = [t for t in C.TOLS if L["rollback"]["last"][str(t)]["optimum"]["all"].get("n_certified") == 0 and L["rollback"]["last"][str(t)]["optimum"]["all"]["n_reached"]]
        if nocert:
            out.append(f"{name}: no optimum is certified within the MILP limit of {o['milp_time_limit_s']:.0f} s at c in {nocert} (key 'last'); OPT is bracketed there.")
        kc = L["keycache_check"]
        if kc["n_lp_shift"]:
            out.append(f"{name}: for {kc['n_lp_shift']} of {kc['n_complaints']} complaints the teacher-forced last prompt position (used here) is one token after the last prompt position of a decode.")
        cap = L["capacity"]["groups"].get("seq")
        if cap:
            z, zc = cap["law"]["z_near"], cap["law"]["z_near_per_prefix_class"]
            if z["max_abs_dev"]:
                out.append(f"{name}: the rank law of GP/Z with z = {z['z']} misses the measured rank by up to {z['max_abs_dev']} on {z['n_family_points'] - z['n_rank_equal']} of {z['n_family_points']} family points: the prefix keys are not "
                           f"bitwise shared ({cap['z_bitwise']} positions bitwise equal in all sequences; {cap['n_prefix_classes']} bitwise-distinct copies), and each copy adds its own {z['z']} dimensions "
                           f"(law per prefix class: {zc['n_rank_equal']}/{zc['n_family_points']} points exact).  The predicted number of protectable edits is unaffected.")
            if cap["law"]["z0"]["kept_pred_last"] != cap["measured"]["kept_feasible_every_complaint"]:
                out.append(f"{name}: plain general position (z = 0) predicts {cap['law']['z0']['kept_pred_last']} protectable edits; measured {cap['measured']['kept_feasible_every_complaint']}.")
    return out


def main():
    t0 = time.time()
    res = {"definition": __doc__, "generated": time.strftime("%Y-%m-%d %H:%M:%S"), "tols": list(C.TOLS),
           "keydefs": {"last": "the last prompt position of the teacher-forced pre-edit answer (one key; primary)",
                       "stacked": "the A answer-predicting positions stacked (Frobenius norms; robustness column)"},
           "signs": "inner products <w_i, w_j> and coordinates <psi_i, k> are SIGNED; norms are magnitudes",
           "precision": "fp64 linear algebra on float32 images of the module-dtype keys and on the fp64 factors of the log",
           "logs": {}}
    for name in C.LOGS:
        p = os.path.join(C.OUT, f"cond_rollback_{name}.json")
        if not os.path.exists(p):
            continue
        d = C.jload(p)
        rows = d["rows"]
        L = {"backbone": d["backbone"], "short": C.LOGS[name]["short"], "n_complaints": d["n_complaints"], "n_by_ptype": d["n_by_ptype"],
             "keycache_check": {k: v for k, v in d["keycache_check"].items() if k != "note"}, "log_level": d["log_level"],
             "A": C.by_ptype(rows, lambda r: r["A"]),
             "conditioning": {kd: cond_summary(rows, kd) for kd in C.KEYDEFS}}
        # optimum rows
        opt, meta = {}, None
        op, mp_ = os.path.join(C.OUT, f"opt_rows_{name}.jsonl"), os.path.join(C.OUT, f"opt_meta_{name}.json")
        if os.path.exists(op):
            for o in C.jsonl_read(op):
                opt[(o["group"], o["keydef"])] = o
        if os.path.exists(mp_):
            meta = C.jload(mp_)
        errs = [(g, k, o["error"]) for (g, k), o in opt.items() if "error" in o]
        within, within_cpu = budget_prefix(opt)
        stop = C.jload(os.path.join(C.OUT, f"opt_stop_{name}.json")) if os.path.exists(os.path.join(C.OUT, f"opt_stop_{name}.json")) else None
        spent = (sum(r["cpu_s_children"] for r in meta["runs"]) if meta else 0.0) + (stop["cpu_s_total"] if stop else 0.0)
        L["optimum_run"] = {"code": "07_repair_v3/rollback_opt/code (run_ropt.solve_one, ropt.*), unchanged; tolerances " + str(list(C.TOLS)),
                            "milp_time_limit_s": C.OPT_TL, "seed": C.OPT_SEED, "workers": C.OPT_WORKERS,
                            "budget_cpu_min": C.OPT_BUDGET_MIN, "budget_cpu_s_for_jobs": BUDGET_CPU_S,
                            "cpu_min_spent": spent / 60.0, "over_budget": bool(spent > C.OPT_BUDGET_MIN * 60.0),
                            "runs": meta["runs"] if meta else None,
                            "stopped_run": {k: stop[k] for k in ("stopped", "reason", "wall_s", "cpu_s_total", "n_workers", "n_rows_finished",
                                                                 "cpu_s_rows_finished")} if stop else None,
                            "n_jobs_finished": {kd: int(sum(1 for (g, k), o in opt.items() if k == kd and "tau" in o)) for kd in C.KEYDEFS},
                            "n_jobs_within_budget": {kd: int(sum(1 for (g, k) in within if k == kd)) for kd in C.KEYDEFS},
                            "cpu_min_jobs_within_budget": within_cpu / 60.0,
                            "n_jobs_copied_from_last": int(sum(1 for o in opt.values() if o.get("copied_from_last"))),
                            "cpu_s_jobs_finished": float(sum(o["cpu"] for o in opt.values())), "errors": errs,
                            "cpu_s_per_job": C.dist([o["cpu"] for o in opt.values() if not o.get("copied_from_last")]),
                            "budget_rule": "40 minutes of CPU time (core-minutes) per log over all workers; 'optimum' uses the jobs "
                                           "within it (budget_prefix), 'optimum_all' every finished job"}
        L["rollback"] = {}
        for kd in C.KEYDEFS:
            L["rollback"][kd] = {}
            for t in C.TOLS:
                ms_all = merged_rows(rows, opt, kd, t)
                ms = merged_rows(rows, opt, kd, t, only=within)
                R = {"rules": rules_summary(ms), "optimum": opt_summary(ms), "optimum_all": opt_summary(ms_all),
                     "need_more_than_with_all_reached": {sel: v["need_more_than"] for sel, v in rules_summary(ms_all).items()}}
                R["tolerance_of_guarantee"] = C.dist([t * float(np.sqrt(r["keydefs"][kd]["gamma"] / r["keydefs"][kd]["Gamma"])) for r in rows])
                L["rollback"][kd][str(t)] = R
        L["capacity"] = capacity_summary(name)
        res["logs"][name] = L
    sp = os.path.join(C.OUT, "selftest.json")
    if os.path.exists(sp):
        st = C.jload(sp)
        res["selftest"] = {k: v for k, v in st.items() if k != "definition"}
    for name in C.LOGS:
        vp = os.path.join(C.OUT, f"verify_{name}.json")
        if os.path.exists(vp) and name in res["logs"]:
            v = C.jload(vp)
            res["logs"][name]["verify"] = {k: x for k, x in v.items() if k != "definition"}
    res["notes"] = notes(res)
    res["sec"] = time.time() - t0
    C.jdump(res, os.path.join(C.OUT, "results.json"))
    P = paper_numbers(res)
    core = {k: v for k, v in P.items() if is_core(k)}
    C.jdump(core, os.path.join(C.OUT, "paper_numbers.json"))
    C.jdump(P, os.path.join(C.OUT, "paper_numbers_all.json"))
    C.jdump(DOC, os.path.join(C.OUT, "paper_numbers_doc.json"))
    print(f"wrote results.json, paper_numbers.json ({len(core)} numbers), paper_numbers_all.json ({len(P)}) and paper_numbers_doc.json ({res['sec']:.1f}s)")


if __name__ == "__main__":
    main()
