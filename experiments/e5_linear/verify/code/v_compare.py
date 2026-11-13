"""Verification: my recomputation against the measuring agent's files -> verify/out/verify_results.json and
verify/verdicts.json (the final message as data; rendered by v_render.py).        usage: python v_compare.py

Mine (written by v_cond_rollback.py, v_capacity.py, v_nu.py, v_opt_ub.py, v_rerun.py; none of them reads the report):
  verify/out/v_cond_rollback_<log>.json, v_capacity_<log>.json, v_nu_<log>.json, v_opt_ub.json, v_rerun.json
Report (read only): out/cond_rollback_<log>.json (rows per complaint), out/capacity_<log>_seq.json, out/paper_numbers.json,
  out/results.json; and the numbers QUOTED in the measuring agent's summary (SUMMARY below: literals copied from the
  summary; they are the claims under test, never a result of mine).
Scopes: 'sample' = the seeded sample of 60 complaints (30 image + 30 text; harness rule, seed 0), 'all' = every REPAIR-pool
D1 complaint (the report's medians are over all complaints, so only this scope reproduces them exactly), 'five' = five
complaints of the sample (numpy default_rng(5).choice) for the capacity.
Agreement rules: real numbers agree if the relative deviation is below 1e-9; counts and shares of pairs must be equal;
a quoted literal agrees with its JSON value if it is a correct rounding (|literal - value| <= half a unit of its last digit).
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import os

import numpy as np

RTOL = 1e-9
Q, L = "TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"

# (id, name in out/paper_numbers.json with {log}, literal for Q, literal for L): the numbers of the measuring agent's summary
SUMMARY = [
    ("Gamma/gamma, last key", "conditioning.{log}.median_Gamma_over_gamma", "57.5", "18.1"),
    ("gamma, last key", "conditioning.{log}.median_gamma", "0.308", "0.407"),
    ("Gamma, last key", "conditioning.{log}.median_Gamma", "17.7", "7.36"),
    ("Gamma/gamma, stacked, median", "conditioning.{log}.stacked.median_Gamma_over_gamma", "10.1", "5.0"),
    ("Gamma/gamma, stacked, q25", "conditioning.{log}.stacked.q25_Gamma_over_gamma", "9.2", "4.2"),
    ("Gamma/gamma, stacked, q75", "conditioning.{log}.stacked.q75_Gamma_over_gamma", "11.1", "7.2"),
    ("negative-pair share, median", "conditioning.{log}.median_share_negative_pairs", "0.50", "0.50"),
    ("cancellation-free complaints", "conditioning.{log}.n_cancellation_free", "0", "0"),
    ("complaints", "conditioning.{log}.n_complaints", "297", "305"),
    ("least nu", "conditioning.{log}.min_nu", "40", "61"),
    ("sqrt(gamma/Gamma)", "conditioning.{log}.median_tolerance_factor", "0.13", "0.24"),
    ("c=0.1 LF", "rollback.{log}.c0.1.median_lf", "72", "72"),
    ("c=0.1 FO", "rollback.{log}.c0.1.median_fo", "70", "71"),
    ("c=0.1 BI", "rollback.{log}.c0.1.median_bi", "70", "71"),
    ("c=0.1 Prop. bound", "rollback.{log}.c0.1.median_lb", "59", "62"),
    ("LF/bound c=0.1", "rollback.{log}.c0.1.median_lf_over_lb", "1.21", "1.15"),
    ("LF/bound c=0.3", "rollback.{log}.c0.3.median_lf_over_lb", "2.00", "1.63"),
    ("LF/bound c=0.5", "rollback.{log}.c0.5.median_lf_over_lb", "6.33", "3.43"),
    ("LF/bound c=0.1, stacked", "rollback.{log}.stacked.c0.1.median_lf_over_lb", "1.03", "1.03"),
    ("optimum code: complaints", "rollback.{log}.c0.1.opt_n_complaints", "43", "52"),
    ("certified c=0.1", "rollback.{log}.c0.1.opt_n_certified", "0", "0"),
    ("certified c=0.3", "rollback.{log}.c0.3.opt_n_certified", "3", "15"),
    ("certified c=0.5", "rollback.{log}.c0.5.opt_n_certified", "20", "38"),
    ("BI/OPT lo c=0.1", "rollback.{log}.c0.1.opt_median_bi_over_opt_lo", "1.00", "1.00"),
    ("BI/OPT hi c=0.1", "rollback.{log}.c0.1.opt_median_bi_over_opt_hi", "1.04", "1.03"),
    ("BI/OPT lo c=0.3", "rollback.{log}.c0.3.opt_median_bi_over_opt_lo", "1.02", "1.00"),
    ("BI/OPT hi c=0.3", "rollback.{log}.c0.3.opt_median_bi_over_opt_hi", "1.05", "1.03"),
    ("BI/OPT lo c=0.5", "rollback.{log}.c0.5.opt_median_bi_over_opt_lo", "1.00", "1.00"),
    ("BI/OPT hi c=0.5", "rollback.{log}.c0.5.opt_median_bi_over_opt_hi", "1.05", "1.00"),
    ("T", "capacity.{log}.T", "100", "100"),
    ("d_in", "capacity.{log}.d_in", "11008", "11008"),
    ("group size, min", "capacity.{log}.group_size_min", "289", "595"),
    ("group size, max", "capacity.{log}.group_size_max", "1059", "610"),
    ("z", "capacity.{log}.prefix_z", "15", "1"),
    ("s*_Z", "capacity.{log}.sstar_Z", "63", "82"),
    ("protectable, predicted", "capacity.{log}.kept_pred", "37", "18"),
    ("protectable, measured", "capacity.{log}.kept_feasible", "37", "18"),
    ("answer keys only: rank", "capacity.{log}.K_ans.rank", "400", "483"),
    ("last keys only: rank", "capacity.{log}.K_last.rank", "100", "100"),
    ("answer keys only: edits kept", "capacity.{log}.K_ans.kept_measured", "100", "100"),
    ("last keys only: edits kept", "capacity.{log}.K_last.kept_measured", "100", "100"),
    ("answer keys only: complaints repairable", "capacity.{log}.K_ans.n_complaints_repairable", "297", "305"),
    ("last keys only: complaints repairable", "capacity.{log}.K_last.n_complaints_repairable", "297", "305"),
    ("z = 0 predicts", "capacity.{log}.kept_pred_z0", "35", None),
    ("bitwise prefix positions", "capacity.{log}.prefix_z_bitwise", "0", None),
    ("core-minutes spent", "rollback.{log}.optimum_run.cpu_min_spent", "97", None),
    ("complaints within budget", "rollback.{log}.optimum_run.n_complaints_within_budget", "43", None),
    ("complaints finished", "rollback.{log}.optimum_run.n_complaints_all_finished", "107", None),
]
# ranges over BOTH logs quoted in the summary: (id, name, 'min' | 'max' over the logs, literal)
SUMMARY_RANGES = [
    ("bound/OPT c=0.1 from", "rollback.{log}.c0.1.opt_median_lb_over_opt_lo", "min", "0.85"),
    ("bound/OPT c=0.1 to", "rollback.{log}.c0.1.opt_median_lb_over_opt_hi", "max", "0.90"),
    ("bound/OPT c=0.3 from", "rollback.{log}.c0.3.opt_median_lb_over_opt_lo", "min", "0.57"),
    ("bound/OPT c=0.3 to", "rollback.{log}.c0.3.opt_median_lb_over_opt_hi", "max", "0.64"),
    ("bound/OPT c=0.5 from", "rollback.{log}.c0.5.opt_median_lb_over_opt_lo", "min", "0.17"),
    ("bound/OPT c=0.5 to", "rollback.{log}.c0.5.opt_median_lb_over_opt_hi", "max", "0.32"),
]


def lit_ok(lit, val):
    nd = len(lit.split(".")[1]) if "." in lit else 0
    return bool(val is not None and abs(float(lit) - float(val)) <= 0.5 * 10 ** (-nd) + 1e-12)


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


def g(x, nd=3):
    """Number -> text with nd significant digits; integers stay integers."""
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    x = float(x)
    if x == int(x) and abs(x) < 1e6:
        return str(int(x))
    return f"{x:.{nd}g}"


def pair(a, b, nd=3):
    return f"{g(a, nd)} | {g(b, nd)}"


def main():
    PN = C.jload(os.path.join(C.REPORT_OUT, "paper_numbers.json"))
    RES = C.jload(os.path.join(C.REPORT_OUT, "results.json"))
    out = {"definition": __doc__, "logs": {}}
    # ------------------------------------------------------------------ the summary's literals against paper_numbers.json
    chk = []
    for cid, key, lq, ll in SUMMARY:
        for name, lit in ((Q, lq), (L, ll)):
            if lit is None:
                continue
            k = key.format(log=name)
            chk.append({"id": cid, "log": name, "name": k, "quoted": lit, "json": PN.get(k), "in_paper_numbers": k in PN, "ok": lit_ok(lit, PN.get(k))})
    for cid, key, how, lit in SUMMARY_RANGES:
        vals = [PN.get(key.format(log=n)) for n in (Q, L)]
        v = (min if how == "min" else max)(vals)
        chk.append({"id": cid, "log": "both", "name": key, "quoted": lit, "json": v, "in_paper_numbers": all(key.format(log=n) in PN for n in (Q, L)), "ok": lit_ok(lit, v)})
    out["summary_vs_json"] = {"n": len(chk), "n_ok": int(sum(c["ok"] for c in chk)), "not_ok": [c for c in chk if not c["ok"]], "checks": chk}
    for name in (Q, L):
        mine = C.jload(os.path.join(C.OUT, f"v_cond_rollback_{name}.json"))
        rep = C.jload(os.path.join(C.REPORT_OUT, f"cond_rollback_{name}.json"))
        rr = {r["group"]: r for r in rep["rows"]}
        mrows = mine["rows"]
        R = {"n_complaints": {"mine": mine["n_complaints"], "report": PN[f"conditioning.{name}.n_complaints"],
                              "mine_by_ptype": mine["n_by_ptype"], "report_by_ptype": {"image": PN[f"conditioning.{name}.n_image"], "text": PN[f"conditioning.{name}.n_text"]},
                              "same_set": set(rr) == {r["group"] for r in mrows}},
             "sample": {"n": len(mine["sample_groups"]), "by_ptype": mine["sample_by_ptype"], "seed": mine["sample_seed"], "harness_check": mine["harness_check"]},
             "keys": {"all": {k: v for k, v in mine["keys_all"].items() if k != "lp_shift_groups"},
                      "sample": {k: v for k, v in mine["keys_sample"].items() if k != "lp_shift_groups"},
                      "report_n_lp_shift": RES["logs"][name]["keycache_check"]["n_lp_shift"]},
             "log_level": mine["log_level"]}
        scopes = {"sample": [r for r in mrows if r["in_sample"]], "all": mrows}
        # -------------------------------------------------------------- (1) conditioning
        cond = {}
        for kd in ("last", "stacked"):
            cond[kd] = {}
            for sc, rows in scopes.items():
                a = [r["keydefs"][kd] for r in rows]
                b = [rr[r["group"]]["keydefs"][kd] for r in rows]
                s = {"n": len(rows)}
                for k in ("gamma", "Gamma", "ratio", "energy", "D_norm"):
                    s[f"max_rel_dev_{k}"] = float(max(rel(x[k], y[k]) for x, y in zip(a, b)))
                s["max_rel_dev_two_routes"] = float(max(x["routes_rel_dev"] for x in a))
                s["max_abs_dev_frac_neg"] = float(max(abs(x["frac_neg"] - y["frac_neg"]) for x, y in zip(a, b)))
                s["n_equal_n_neg_pairs"] = int(sum(x["n_neg_pairs"] == y["n_neg_pairs"] for x, y in zip(a, b)))
                s["n_zero_pairs"] = int(sum(x["n_zero_pairs"] for x in a))
                s["n_cancellation_free"] = int(sum(x["n_neg_pairs"] == 0 for x in a))
                s["n_gamma_positive"] = int(sum(x["gamma"] > 0 for x in a))
                s["n_distinct_ratio_rel_1e-9"] = int(len({round(float(np.log(x["ratio"])) / 1e-9) for x in a}))
                for k in ("gamma", "Gamma", "ratio", "frac_neg", "tol_factor"):
                    s[f"mine_{k}"] = C.quart([x[k] for x in a])
                for k in ("gamma", "Gamma", "ratio", "frac_neg"):
                    s[f"report_rows_{k}"] = C.quart([y[k] for y in b])
                if kd == "last":
                    s["mine_fac_vs_applied"] = C.quart([x["fac_vs_applied"] for x in a])
                    s["report_rows_fac_vs_applied"] = C.quart([y["fac_vs_applied"] for y in b])
                    s["mine_n_neg_coord"] = C.quart([x["n_neg_coord"] for x in a])
                cond[kd][sc] = s
        tag = {"last": "", "stacked": "stacked."}
        cond["report_paper_numbers"] = {kd: {k: PN[f"conditioning.{name}.{tag[kd]}{k}"] for k in
                                             ("median_Gamma_over_gamma", "q25_Gamma_over_gamma", "q75_Gamma_over_gamma", "median_gamma", "median_Gamma",
                                              "median_share_negative_pairs", "n_cancellation_free", "median_tolerance_factor")} for kd in ("last", "stacked")}
        ca, cp = cond["last"]["all"], cond["report_paper_numbers"]["last"]
        cond["agree"] = {
            "gamma": bool(ca["max_rel_dev_gamma"] < RTOL and rel(ca["mine_gamma"]["median"], cp["median_gamma"]) < RTOL),
            "Gamma": bool(ca["max_rel_dev_Gamma"] < RTOL and rel(ca["mine_Gamma"]["median"], cp["median_Gamma"]) < RTOL),
            "ratio": bool(ca["max_rel_dev_ratio"] < RTOL and rel(ca["mine_ratio"]["median"], cp["median_Gamma_over_gamma"]) < RTOL),
            "frac_neg": bool(ca["max_abs_dev_frac_neg"] == 0 and ca["mine_frac_neg"]["median"] == cp["median_share_negative_pairs"]
                             and ca["n_cancellation_free"] == cp["n_cancellation_free"]),
            "sample_per_complaint": bool(max(cond["last"]["sample"][f"max_rel_dev_{k}"] for k in ("gamma", "Gamma", "ratio")) < RTOL
                                         and cond["last"]["sample"]["max_abs_dev_frac_neg"] == 0),
            "stacked": bool(all(rel(cond["stacked"]["all"]["mine_ratio"][q], cond["report_paper_numbers"]["stacked"][k]) < RTOL
                                for q, k in (("median", "median_Gamma_over_gamma"), ("q25", "q25_Gamma_over_gamma"), ("q75", "q75_Gamma_over_gamma")))
                            and cond["stacked"]["all"]["max_rel_dev_ratio"] < RTOL)}
        cond["one_value_for_all_complaints"] = bool(ca["n_distinct_ratio_rel_1e-9"] == 1)
        cond["ratio_of_write_vectors"] = mine["log_level"]["ratio_U"]
        cond["gamma_times_T"] = ca["mine_gamma"]["median"] * mine["log_level"]["T"]
        R["conditioning"] = cond
        # -------------------------------------------------------------- (2) rollback by rule and the bounds
        rb = {}
        for kd in ("last", "stacked"):
            rb[kd] = {}
            for t in C.TOLS:
                st = {}
                for sc, rows in scopes.items():
                    a = [r["keydefs"][kd]["tol"][str(t)] for r in rows]
                    b = [rr[r["group"]]["keydefs"][kd]["tol"][str(t)] for r in rows]
                    s = {"n": len(rows)}
                    for mk, rk in (("lf", "lf"), ("fo", "fo"), ("bi", "bi"), ("lb_mu", "lb_cond"), ("lf_guar", "lf_guar")):
                        s[f"n_equal_{mk}"] = int(sum(x[mk] == y[rk] for x, y in zip(a, b)))
                        s[f"mine_{mk}"] = C.quart([x[mk] for x in a])
                        s[f"report_rows_{mk}"] = C.quart([y[rk] for y in b])
                    s["mine_lb1"] = C.quart([x["lb1"] for x in a])
                    s["mine_lf_over_lb1"] = C.quart([x["lf"] / x["lb1"] for x in a])
                    s["mine_fo_over_lb1"] = C.quart([x["fo"] / x["lb1"] for x in a])
                    s["mine_lb_stmt"] = C.quart([x["lb_stmt"] for x in a])
                    s["n_lb_stmt_zero"] = int(sum(x["lb_stmt"] == 0 for x in a))
                    s["n_lb_mu_zero"] = int(sum(x["lb_mu"] == 0 for x in a))
                    s["n_stmt_check"] = int(sum(x["stmt_check"] for x in a))
                    s["n_sandwich_ok"] = int(sum(x["sandwich_ok"] for x in a))
                    s["tol_scaled_down"] = C.quart([x["tol_scaled_down"] for x in a])
                    s["mine_min_rule_at_scaled_down"] = C.quart([min(x["rules_at_scaled_down"].values()) for x in a])
                    mm = [m for x in a for r_ in ("lf", "fo", "bi") for m in (x[r_ + "_margin_stop"], x[r_ + "_margin_prev"]) if m is not None]
                    s["min_rel_margin_of_a_stop_decision"] = float(min(mm)) if mm else None
                    s["groups_bi_deletes_all"] = [r["group"] for r, x in zip(rows, a) if x["bi"] == 100]
                    s["fo_where_bi_deletes_all"] = [x["fo"] for x in a if x["bi"] == 100]
                    s["n_lf_fo_deletes_all"] = int(sum(x["lf"] == 100 or x["fo"] == 100 for x in a))
                    st[sc] = s
                pre = f"rollback.{name}.{tag[kd]}c{t}."
                st["report_paper_numbers"] = {k: PN.get(pre + k) for k in ("median_lf", "median_fo", "median_bi", "median_lb", "median_lf_over_lb",
                                                                           "median_fo_over_lb", "median_lf_guarantee", "n_bi_deletes_all")}
                sa, sp = st["all"], st["report_paper_numbers"]
                st["agree"] = {"lf": bool(sa["n_equal_lf"] == sa["n"] and sa["mine_lf"]["median"] == sp["median_lf"]),
                               "fo": bool(sa["n_equal_fo"] == sa["n"] and sa["mine_fo"]["median"] == sp["median_fo"]),
                               "bi": bool(sa["n_equal_bi"] == sa["n"] and sa["mine_bi"]["median"] == sp["median_bi"]),
                               "lb": bool(sa["n_equal_lb_mu"] == sa["n"] and sa["mine_lb1"]["median"] == sp["median_lb"]),
                               "lf_over_lb": bool(rel(sa["mine_lf_over_lb1"]["median"], sp["median_lf_over_lb"]) < RTOL),
                               "sample_per_complaint": bool(all(st["sample"][f"n_equal_{k}"] == st["sample"]["n"] for k in ("lf", "fo", "bi", "lb_mu")))}
                rb[kd][str(t)] = st
        R["rollback"] = rb
        # -------------------------------------------------------------- (3) capacity
        mc = C.jload(os.path.join(C.OUT, f"v_capacity_{name}.json"))
        rc = C.jload(os.path.join(C.REPORT_OUT, f"capacity_{name}_seq.json"))
        ri = {gname: j for j, gname in enumerate(rc["complaints"]["group"])}
        mi = {gname: j for j, gname in enumerate(mc["complaints"]["group"])}
        fam_m, fam_r = mc["family"], rc["family"]
        nn = min(len(fam_m["n"]), len(fam_r["n"]))
        five = []
        for gname in mc["five"]:
            jm, jr = mi[gname], ri[gname]
            ncm = mc["complaints"]["n_cap_last"][jm]
            em, er = mc["complaints"]["escape_last"][ncm][jm], fam_r["escape_last"][ncm][jr]
            five.append({"group": gname, "ptype": mc["complaints"]["ptype"][jm], "A": mc["complaints"]["A"][jm],
                         "mine_n_cap_last": ncm, "report_n_cap_last": fam_r["n_cap_last"][jr],
                         "mine_n_cap_stacked": mc["complaints"]["n_cap_stacked"][jm], "report_n_cap_stacked": fam_r["n_cap_stacked"][jr],
                         "mine_escape_at_n_cap": em, "report_escape_at_n_cap": er, "rel_dev_escape": rel(em, er),
                         "mine_escape_at_n_cap_plus_1": mc["complaints"]["escape_last"][ncm + 1][jm] if ncm + 1 < len(fam_m["n"]) else None,
                         "report_escape_at_n_cap_plus_1": fam_r["escape_last"][ncm + 1][jr] if ncm + 1 < len(fam_r["n"]) else None,
                         "mine_rank_kept_stacked_at_n_cap": mc["complaints"]["rank_kept_stacked"][ncm][jm],
                         "agree": bool(ncm == fam_r["n_cap_last"][jr] and mc["complaints"]["n_cap_stacked"][jm] == fam_r["n_cap_stacked"][jr])})
        common = [gname for gname in mc["complaints"]["group"] if gname in ri]
        nf = fam_m["n_free"]
        dev_esc = max(rel(mc["complaints"]["escape_last"][n][mi[gname]], fam_r["escape_last"][n][ri[gname]]) for n in range(min(nf, nn - 1) + 1) for gname in common)
        cap = {"five": five, "n_five_agree": int(sum(f["agree"] for f in five)),
               "all": {"n": len(common), "n_equal_n_cap_last": int(sum(mc["complaints"]["n_cap_last"][mi[x]] == fam_r["n_cap_last"][ri[x]] for x in common)),
                       "n_equal_n_cap_stacked": int(sum(mc["complaints"]["n_cap_stacked"][mi[x]] == fam_r["n_cap_stacked"][ri[x]] for x in common)),
                       "mine_n_cap_last_values": sorted(set(mc["complaints"]["n_cap_last"])), "mine_n_cap_stacked_values": sorted(set(mc["complaints"]["n_cap_stacked"])),
                       "max_rel_dev_escape_last_up_to_n_free": float(dev_esc),
                       "mine_escape_at_n_free": C.quart(mc["complaints"]["escape_last"][nf]),
                       "n_monotone": [int(sum(mc["complaints"]["monotone_last"])), int(sum(mc["complaints"]["monotone_stacked"]))]},
               "family": {"n_points_compared": nn, "rank_equal_at_every_n": fam_m["rank"][:nn] == fam_r["rank"][:nn],
                          "mine_n_free": nf, "mine_n_sat": fam_m["n_sat"], "report_saturated_at_n": fam_r["saturated_at_n"],
                          "mine_rank_at_n_free": fam_m["rank"][nf], "report_rank_at_kept": PN[f"capacity.{name}.rank_at_kept"],
                          "mine_keys_at_n_free": fam_m["sum_size"][nf], "mine_classes_at_n_free": fam_m["n_first_position_classes"][nf],
                          "first_n_with_two_classes": next((n for n, c in zip(fam_m["n"], fam_m["n_first_position_classes"]) if c >= 2), None),
                          "min_rdiag_over_s1_at_n_free": fam_m["min_rdiag_over_s1_at_n_free"], "q_roundtrip_max_rel": fam_m["q_roundtrip_max_rel"]},
               "spectrum": mc.get("spectrum"),
               "edit_keys": {"cross_check_final_key_cache": mc["edit_keys"]["cross_check_final_key_cache"],
                             "group_size_min_max": [min(mc["edit_keys"]["len"]), max(mc["edit_keys"]["len"])],
                             "report_group_size_min_max": [PN[f"capacity.{name}.group_size_min"], PN[f"capacity.{name}.group_size_max"]],
                             "n_groups_full_rank": mc["edit_keys"]["n_groups_full_rank"], "group_rank_equal_report": mc["edit_keys"]["group_rank"] == rc["edit_keys"]["group_rank"],
                             "order_equal_report": mc["edit_keys"]["order_smallest_first"] == rc["edit_keys"]["order_smallest_first"],
                             "z_bitwise": mc["edit_keys"]["z_bitwise"], "z_near": mc["edit_keys"]["z_near"],
                             "report_z": PN[f"capacity.{name}.prefix_z"], "report_z_bitwise": PN[f"capacity.{name}.prefix_z_bitwise"],
                             "near_prefix_max_rel_diff": max(mc["edit_keys"]["near_prefix_max_rel_diff"][: mc["edit_keys"]["z_near"]]) if mc["edit_keys"]["z_near"] else None,
                             "distinct_keys_at_leading_positions": mc["edit_keys"]["distinct_keys_at_leading_positions"][:20]},
               "law": {"mine": mc["law"], "report_sstar_Z": PN[f"capacity.{name}.sstar_Z"], "report_kept_pred": PN[f"capacity.{name}.kept_pred"],
                       "report_sstar_z0": PN[f"capacity.{name}.sstar_z0"], "report_kept_feasible": PN[f"capacity.{name}.kept_feasible"],
                       "agree_sstar_Z": bool(mc["law"]["z_near"]["sstar"] == PN[f"capacity.{name}.sstar_Z"] and mc["law"]["z0"]["sstar"] == PN[f"capacity.{name}.sstar_z0"]),
                       "premise_exact": bool(mc["edit_keys"]["z_bitwise"] == mc["edit_keys"]["z_near"])},
               "points": {p: {"mine_rank": mc["points"][p]["rank"], "report_rank": PN[f"capacity.{name}.{p}.rank"],
                              "mine_n_keys": mc["points"][p]["n_keys"], "report_n_keys": PN[f"capacity.{name}.{p}.n_keys"],
                              "mine_n_repairable": min(mc["points"][p]["n_feasible_last"], mc["points"][p]["n_feasible_stacked"]),
                              "report_n_repairable": PN[f"capacity.{name}.{p}.n_complaints_repairable"],
                              "mine_median_escape": float(np.median(mc["points"][p]["escape_last"])), "report_median_escape": PN[f"capacity.{name}.{p}.median_escape_last_key"],
                              "agree": bool(mc["points"][p]["rank"] == PN[f"capacity.{name}.{p}.rank"]
                                            and min(mc["points"][p]["n_feasible_last"], mc["points"][p]["n_feasible_stacked"]) == PN[f"capacity.{name}.{p}.n_complaints_repairable"]
                                            and rel(float(np.median(mc["points"][p]["escape_last"])), PN[f"capacity.{name}.{p}.median_escape_last_key"]) < RTOL)}
                          for p in ("K_ans", "K_last")}}
        cap["agree_five"] = bool(cap["n_five_agree"] == len(five) and all(f["mine_n_cap_last"] == PN[f"capacity.{name}.kept_feasible"] for f in five))
        R["capacity"] = cap
        # -------------------------------------------------------------- extra: nu
        mn = C.jload(os.path.join(C.OUT, f"v_nu_{name}.json"))
        nr = {r["group"]: r for r in mn["rows"]}
        R["nu"] = {"mine": mn["nu"], "report_min_median_max": [PN[f"conditioning.{name}.min_nu"], PN[f"conditioning.{name}.median_nu"], PN[f"conditioning.{name}.max_nu"]],
                   "n_equal_per_complaint": int(sum(nr[x]["nu"] == rr[x]["keydefs"]["last"]["nu"] for x in nr)), "n": len(nr),
                   "n_milp_optimal": mn["n_milp_optimal"], "n_methods_agree": mn["n_methods_agree"],
                   "agree": bool(mn["nu"]["min"] == PN[f"conditioning.{name}.min_nu"] and mn["nu"]["median"] == PN[f"conditioning.{name}.median_nu"]
                                 and all(nr[x]["nu"] == rr[x]["keydefs"]["last"]["nu"] for x in nr))}
        out["logs"][name] = R
    # numbers of the summary that are derived quantities (no name in paper_numbers.json): against MY values
    cq = out["logs"][Q]["capacity"]
    off = cq["family"]["mine_rank_at_n_free"] - (cq["family"]["mine_keys_at_n_free"] - cq["edit_keys"]["z_near"] * (cq["family"]["mine_n_free"] - 1))
    der = [("tolerance c sqrt(gamma/Gamma) at which the proposition places OPT, c = 0.1", Q, "0.013", out["logs"][Q]["rollback"]["last"]["0.1"]["all"]["tol_scaled_down"]["median"]),
           ("tolerance c sqrt(gamma/Gamma) at which the proposition places OPT, c = 0.1", L, "0.024", out["logs"][L]["rollback"]["last"]["0.1"]["all"]["tol_scaled_down"]["median"]),
           ("gamma T (gamma against 1/T)", Q, "31", out["logs"][Q]["conditioning"]["gamma_times_T"]), ("gamma T (gamma against 1/T)", L, "41", out["logs"][L]["conditioning"]["gamma_times_T"]),
           ("rank law with z = 15 off from n", Q, "35", cq["family"]["first_n_with_two_classes"]), ("rank law with z = 15 off by", Q, "15", off),
           ("FO where BI deletes all (text_125, c = 0.1)", L, "73", (out["logs"][L]["rollback"]["last"]["0.1"]["all"]["fo_where_bi_deletes_all"] or [None])[0])]
    for cid, name, lit, val in der:
        out["summary_vs_json"]["checks"].append({"id": cid, "log": name, "name": "derived (my value)", "quoted": lit, "json": val, "in_paper_numbers": False, "ok": lit_ok(lit, val)})
    chk = out["summary_vs_json"]["checks"]
    out["summary_vs_json"].update(n=len(chk), n_ok=int(sum(c["ok"] for c in chk)), not_ok=[c for c in chk if not c["ok"]],
                                  n_against_my_values=int(sum(not c["in_paper_numbers"] for c in chk)))
    out["opt_ub"] = C.jload(os.path.join(C.OUT, "v_opt_ub.json"))["logs"]
    olb = C.jload(os.path.join(C.OUT, "v_opt_lb.json"))
    out["opt_lb"] = {"selftest": olb["selftest"], "shifts": olb["shifts"], "refine_iters": olb["refine_iters"],
                     "note": "the shifts and the refinement rule of the certificate search were fixed after trials on five complaints of TB-L100-ae-l2_3; they decide how tight a "
                             "certified bound is, never whether it is valid (any positive definite shift, any multiplier and any point of the box give a valid bound)",
                     "logs": {n: {"n_rows": v["n_rows"], "n_within_budget": v["n_within_budget"], "per_tol": v["per_tol"]} for n, v in olb["logs"].items()}}
    tr = C.jload(os.path.join(C.OUT, "v_target_rank.json"))["logs"]
    for name in (Q, L):
        rc = C.jload(os.path.join(C.REPORT_OUT, f"capacity_{name}_seq.json"))["complaints"]
        mt = {r["group"]: r for r in tr[name]["rows"]}
        out["logs"][name]["target_rank"] = {
            **{k: v for k, v in tr[name].items() if k != "rows"},
            "n_rank_V_equal_report": int(sum(mt[x]["rank_V"] == a for x, a in zip(rc["group"], rc["a_rank_stacked"]))),
            "n_rank_Kq_equal_report": int(sum(mt[x]["rank_Kq"] == a for x, a in zip(rc["group"], rc["kq_rank"]))),
            "report_n_rank_V_last_equals_1": int(sum(rc["a_rank_last"]))}
    pf = C.jload(os.path.join(C.OUT, "v_prefix.json"))["logs"]
    for name in (Q, L):
        out["logs"][name]["prefix"] = {k: v for k, v in pf[name].items() if k != "class_of_the_n_smallest"}
    rer = C.jload(os.path.join(C.OUT, "v_rerun.json"))
    out["rerun"] = {k: rer[k] for k in ("derivation", "rerun", "render", "untouched", "all_pass")}
    C.jdump(out, os.path.join(C.OUT, "verify_results.json"))

    # ================================================================== verdicts (the final message as data)
    def both(fn, nd=3):
        return pair(fn(out["logs"][Q]), fn(out["logs"][L]), nd)

    def allof(fn):
        return bool(fn(out["logs"][Q]) and fn(out["logs"][L]))

    def P(key, nd=3):
        return pair(PN[key.format(log=Q)], PN[key.format(log=L)], nd)

    rows = []

    def add(rid, quantity, report, mine, agree, verdict, note=""):
        rows.append({"id": rid, "quantity": quantity, "report": report, "mine": mine, "agree": bool(agree), "verdict": verdict, "note": note})

    one_val = allof(lambda R: R["conditioning"]["one_value_for_all_complaints"])
    add("gamma", "gamma, last key", P("conditioning.{log}.median_gamma"), both(lambda R: R["conditioning"]["last"]["all"]["mine_gamma"]["median"]),
        allof(lambda R: R["conditioning"]["agree"]["gamma"] and R["conditioning"]["agree"]["sample_per_complaint"]), "safe")
    add("Gamma", "Gamma, last key", P("conditioning.{log}.median_Gamma"), both(lambda R: R["conditioning"]["last"]["all"]["mine_Gamma"]["median"]),
        allof(lambda R: R["conditioning"]["agree"]["Gamma"] and R["conditioning"]["agree"]["sample_per_complaint"]), "safe")
    add("ratio", "Gamma/gamma, last key", P("conditioning.{log}.median_Gamma_over_gamma"), both(lambda R: R["conditioning"]["last"]["all"]["mine_ratio"]["median"]),
        allof(lambda R: R["conditioning"]["agree"]["ratio"] and R["conditioning"]["agree"]["sample_per_complaint"]),
        "safe as a constant of the log, not as a median over complaints" if one_val else "safe",
        "one value for every complaint: the ratio of the cosine matrix of the write vectors" if one_val else "")
    add("neg", "negative-pair share, median", P("conditioning.{log}.median_share_negative_pairs"),
        both(lambda R: R["conditioning"]["last"]["all"]["mine_frac_neg"]["median"]) + " (sample " + both(lambda R: R["conditioning"]["last"]["sample"]["mine_frac_neg"]["median"]) + ")",
        allof(lambda R: R["conditioning"]["agree"]["frac_neg"] and R["conditioning"]["agree"]["sample_per_complaint"]), "safe")
    for t in ("0.1", "0.5"):
        add(f"rules_c{t}", f"c={t}: LF/FO deletions, median",
            " | ".join(f"{g(PN[f'rollback.{n}.c{t}.median_lf'])}/{g(PN[f'rollback.{n}.c{t}.median_fo'])}" for n in (Q, L)),
            " | ".join(f"{g(out['logs'][n]['rollback']['last'][t]['all']['mine_lf']['median'])}/{g(out['logs'][n]['rollback']['last'][t]['all']['mine_fo']['median'])}" for n in (Q, L))
            + " (sample " + " | ".join(f"{g(out['logs'][n]['rollback']['last'][t]['sample']['mine_lf']['median'])}/{g(out['logs'][n]['rollback']['last'][t]['sample']['mine_fo']['median'])}" for n in (Q, L)) + ")",
            allof(lambda R: R["rollback"]["last"][t]["agree"]["lf"] and R["rollback"]["last"][t]["agree"]["fo"] and R["rollback"]["last"][t]["agree"]["sample_per_complaint"]), "safe",
            "counts equal for every complaint")
    def lb_cells(src):                                                      # 'c=0.1,c=0.5' per log, cells Q | L
        return " | ".join(",".join(g(src(n, t)) for t in ("0.1", "0.5")) for n in (Q, L))
    lb_mine = lb_cells(lambda n, t: out["logs"][n]["rollback"]["last"][t]["all"]["mine_lb1"]["median"])
    lb_rep = lb_cells(lambda n, t: PN[f"rollback.{n}.c{t}.median_lb"])
    lb_stmt = lb_cells(lambda n, t: out["logs"][n]["rollback"]["last"][t]["all"]["mine_lb_stmt"]["median"])
    add("bound", "lower bound, median, c=0.1,0.5", lb_rep, lb_mine,
        allof(lambda R: R["rollback"]["last"]["0.1"]["agree"]["lb"] and R["rollback"]["last"]["0.5"]["agree"]["lb"]),
        "safe only as mu(c^2|D|^2/gamma) of the report's sandwich (prop:app:cond); not as Prop. prop:cond",
        f"Prop. prop:cond as displayed gives OPT_c >= LF(c sqrt(Gamma/gamma)): median {lb_stmt}")
    add("capacity", "capacity, five complaints per log", P("capacity.{log}.kept_feasible"),
        " | ".join(",".join(str(f["mine_n_cap_last"]) for f in out["logs"][n]["capacity"]["five"]) for n in (Q, L)),
        allof(lambda R: R["capacity"]["agree_five"]), "safe (floating point)",
        "also equal for every complaint and both key definitions")
    d = out["rerun"]["derivation"]
    add("derivation", "paper_numbers.json from results.json", f"{d['n_values_paper_numbers']} values",
        f"{d['n_differences_paper_numbers']} differ; re-run from raw rows: {out['rerun']['rerun']['results_json']['n_differences']} differ",
        out["rerun"]["all_pass"], "safe")
    extras = []

    def ex(rid, quantity, report, mine, agree, verdict, note=""):
        extras.append({"id": rid, "quantity": quantity, "report": report, "mine": mine, "agree": bool(agree), "verdict": verdict, "note": note})

    ex("stacked_ratio", "Gamma/gamma, stacked keys, median [quartiles]",
       " | ".join(f"{g(PN[f'conditioning.{n}.stacked.median_Gamma_over_gamma'])} [{g(PN[f'conditioning.{n}.stacked.q25_Gamma_over_gamma'])}, {g(PN[f'conditioning.{n}.stacked.q75_Gamma_over_gamma'])}]" for n in (Q, L)),
       " | ".join(f"{g(out['logs'][n]['conditioning']['stacked']['all']['mine_ratio']['median'])} [{g(out['logs'][n]['conditioning']['stacked']['all']['mine_ratio']['q25'])}, {g(out['logs'][n]['conditioning']['stacked']['all']['mine_ratio']['q75'])}]" for n in (Q, L)),
       allof(lambda R: R["conditioning"]["agree"]["stacked"]), "safe")
    ex("lf_over_lb", "LF / bound, median, c=0.1, 0.3, 0.5; stacked c=0.1",
       "; ".join(P("rollback.{log}.c" + t + ".median_lf_over_lb") for t in ("0.1", "0.3", "0.5")) + "; " + P("rollback.{log}.stacked.c0.1.median_lf_over_lb"),
       "; ".join(both(lambda R: R["rollback"]["last"][t]["all"]["mine_lf_over_lb1"]["median"]) for t in ("0.1", "0.3", "0.5")) + "; " + both(lambda R: R["rollback"]["stacked"]["0.1"]["all"]["mine_lf_over_lb1"]["median"]),
       allof(lambda R: all(R["rollback"]["last"][t]["agree"]["lf_over_lb"] for t in ("0.1", "0.3", "0.5")) and R["rollback"]["stacked"]["0.1"]["agree"]["lf_over_lb"]),
       "safe with the bound named as in row 'bound'")
    ex("bi_c0.1", "c=0.1: BI deletions, median; complaints on which BI deletes all",
       P("rollback.{log}.c0.1.median_bi") + "; " + P("rollback.{log}.c0.1.n_bi_deletes_all"),
       both(lambda R: R["rollback"]["last"]["0.1"]["all"]["mine_bi"]["median"]) + "; " + both(lambda R: len(R["rollback"]["last"]["0.1"]["all"]["groups_bi_deletes_all"]))
       + " (" + ", ".join(f"{x} FO {y}" for x, y in zip(out["logs"][L]["rollback"]["last"]["0.1"]["all"]["groups_bi_deletes_all"], out["logs"][L]["rollback"]["last"]["0.1"]["all"]["fo_where_bi_deletes_all"])) + ")",
       allof(lambda R: R["rollback"]["last"]["0.1"]["agree"]["bi"]) and len(out["logs"][L]["rollback"]["last"]["0.1"]["all"]["groups_bi_deletes_all"]) == PN[f"rollback.{L}.c0.1.n_bi_deletes_all"], "safe")
    ex("nu", "least vertex cover number nu (exact)", P("conditioning.{log}.min_nu"), both(lambda R: R["nu"]["mine"]["min"]), allof(lambda R: R["nu"]["agree"]), "safe",
       "integer program (HiGHS, optimal) and networkx clique number, every complaint")
    exact = allof(lambda R: R["capacity"]["law"]["premise_exact"])
    ex("sstar", "s*_Z (z)", " | ".join(f"{PN[f'capacity.{n}.sstar_Z']} ({PN[f'capacity.{n}.prefix_z']})" for n in (Q, L)),
       " | ".join(f"{out['logs'][n]['capacity']['law']['mine']['z_near']['sstar']} ({out['logs'][n]['capacity']['law']['mine']['z_near']['z']}; bitwise z {out['logs'][n]['capacity']['edit_keys']['z_bitwise']}: {out['logs'][n]['capacity']['law']['mine']['z_bitwise']['sstar']})" for n in (Q, L)),
       allof(lambda R: R["capacity"]["law"]["agree_sstar_Z"]),
       "safe" if exact else "safe for L; for Q only with the caveat that the prefix is shared approximately (no position bit for bit)",
       "" if exact else f"Q: {out['logs'][Q]['capacity']['edit_keys']['distinct_keys_at_leading_positions'][0]} bitwise-distinct copies of the {out['logs'][Q]['capacity']['edit_keys']['z_near']} prefix keys, "
                        f"relative difference up to {g(out['logs'][Q]['capacity']['edit_keys']['near_prefix_max_rel_diff'], 2)}; exact z = 0 gives s* = {out['logs'][Q]['capacity']['law']['mine']['z0']['sstar']}, above the measured sacrifice")
    ex("points", "answer keys / last keys only: rank; complaints repairable",
       " | ".join(f"{PN[f'capacity.{n}.K_ans.rank']}/{PN[f'capacity.{n}.K_last.rank']}; {PN[f'capacity.{n}.K_ans.n_complaints_repairable']}/{PN[f'capacity.{n}.K_last.n_complaints_repairable']}" for n in (Q, L)),
       " | ".join(f"{out['logs'][n]['capacity']['points']['K_ans']['mine_rank']}/{out['logs'][n]['capacity']['points']['K_last']['mine_rank']}; {out['logs'][n]['capacity']['points']['K_ans']['mine_n_repairable']}/{out['logs'][n]['capacity']['points']['K_last']['mine_n_repairable']}" for n in (Q, L)),
       allof(lambda R: R["capacity"]["points"]["K_ans"]["agree"] and R["capacity"]["points"]["K_last"]["agree"]), "safe")
    ex("groups", "T, d_in; group sizes", " | ".join(f"{PN[f'capacity.{n}.T']}, {PN[f'capacity.{n}.d_in']}; {PN[f'capacity.{n}.group_size_min']}-{PN[f'capacity.{n}.group_size_max']}" for n in (Q, L)),
       " | ".join(f"{out['logs'][n]['log_level']['T']}, {out['logs'][n]['log_level']['d_in']}; {out['logs'][n]['capacity']['edit_keys']['group_size_min_max'][0]}-{out['logs'][n]['capacity']['edit_keys']['group_size_min_max'][1]}" for n in (Q, L)),
       allof(lambda R: R["capacity"]["edit_keys"]["group_size_min_max"] == R["capacity"]["edit_keys"]["report_group_size_min_max"] and R["log_level"]["T"] == 100 and R["log_level"]["d_in"] == 11008), "safe")
    ex("lp_shift", "L complaints whose teacher-forced last prompt position is one token after the decode's", str(out["logs"][L]["keys"]["report_n_lp_shift"]),
       str(out["logs"][L]["keys"]["all"]["n_lp_shift"]), out["logs"][L]["keys"]["report_n_lp_shift"] == out["logs"][L]["keys"]["all"]["n_lp_shift"], "safe")
    ou, ol = out["opt_ub"], out["opt_lb"]["logs"]
    TT = ("0.1", "0.3", "0.5")

    def per_c(fn, nd=3):                                                    # 'c=0.1,0.3,0.5' per log, cells Q | L
        return " | ".join(",".join(g(fn(n, t), nd) for t in TT) for n in (Q, L))
    ub_all = all(ou[n]["per_tol"][t]["n_ub_feasible"] == ou[n]["n_rows"] for n in (Q, L) for t in TT)
    max_excess = max(int(k) for n in (Q, L) for t in TT for k in ol[n]["per_tol"][t]["all_rows"]["report_lb_minus_mine_hist"])
    no_contra = all(ol[n]["per_tol"][t]["all_rows"]["n_my_bound_above_report_ub"] == 0 for n in (Q, L) for t in TT)
    ex("opt_ub", "optimum code: upper bounds that are feasible deletion sets of the stated size (finished rows)",
       " | ".join(str(ou[n]["n_rows"]) for n in (Q, L)), per_c(lambda n, t: ou[n]["per_tol"][t]["n_ub_feasible"]), ub_all, "safe")
    ex("opt_lb", "optimum code: lower bounds confirmed by my solver-free certificate (complaints within the budget)",
       " | ".join(str(ol[n]["n_within_budget"]) for n in (Q, L)), per_c(lambda n, t: ol[n]["per_tol"][t]["within_budget"]["n_report_lb_confirmed"]),
       no_contra, f"the confirmed ones safe; the others exceed my certified bound by at most {max_excess}: not refuted, they rest on the report's branch and bound",
       "all finished rows: " + per_c(lambda n, t: ol[n]["per_tol"][t]["all_rows"]["n_report_lb_confirmed"]) + " of " + " | ".join(str(ol[n]["n_rows"]) for n in (Q, L))
       + "; my bound above the report's upper bound: " + str(sum(ol[n]["per_tol"][t]["all_rows"]["n_my_bound_above_report_ub"] for n in (Q, L) for t in TT)))
    for rule in ("bi", "fo"):
        hi_ok = all(ol[n]["per_tol"][t]["within_budget"][f"median_{rule}_over_opt_lo"] == ol[n]["per_tol"][t]["report"][f"opt_median_{rule}_over_opt_lo"] for n in (Q, L) for t in TT)
        ex(f"opt_{rule}", f"{rule.upper()} / OPT, median over the complaints within the budget: at least; at most",
           per_c(lambda n, t: ol[n]["per_tol"][t]["report"][f"opt_median_{rule}_over_opt_lo"]) + "; " + per_c(lambda n, t: ol[n]["per_tol"][t]["report"][f"opt_median_{rule}_over_opt_hi"]),
           per_c(lambda n, t: ol[n]["per_tol"][t]["within_budget"][f"median_{rule}_over_opt_lo"]) + "; " + per_c(lambda n, t: ol[n]["per_tol"][t]["within_budget"][f"median_{rule}_over_opt_hi"]),
           hi_ok, "lower ends safe; upper ends: quote mine (certified without the solver) or the report's as 'by the optimum code'",
           "mine uses my certified lower bounds: tighter than the report at c = 0.1, wider at c = 0.5")
    ex("opt_lbratio", "(bound of the proposition) / OPT, median within the budget: at least; at most",
       per_c(lambda n, t: ol[n]["per_tol"][t]["report"]["opt_median_lb_over_opt_lo"]) + "; " + per_c(lambda n, t: ol[n]["per_tol"][t]["report"]["opt_median_lb_over_opt_hi"]),
       per_c(lambda n, t: ol[n]["per_tol"][t]["within_budget"]["median_lb_over_opt_lo"]) + "; " + per_c(lambda n, t: ol[n]["per_tol"][t]["within_budget"]["median_lb_over_opt_hi"]),
       all(ol[n]["per_tol"][t]["within_budget"]["median_lb_over_opt_lo"] == ol[n]["per_tol"][t]["report"]["opt_median_lb_over_opt_lo"] for n in (Q, L) for t in TT),
       "safe to two digits with my certified bounds (the bound is loose at c = 0.3, 0.5)")
    ex("opt_cert", "optimum certified (complaints within the budget)", per_c(lambda n, t: ol[n]["per_tol"][t]["report"]["opt_n_certified"]),
       per_c(lambda n, t: ol[n]["per_tol"][t]["within_budget"]["n_my_bracket_closed"]), False,
       "not reproduced: the report's counts rest on its branch and bound; mine are what my certificate closes (at c = 0.1 more than the report)",
       "'no optimum certified at c = 0.1' describes the report's solver, not the instances")
    sv = out["summary_vs_json"]
    ex("summary", "numbers quoted in the summary that are correct roundings of paper_numbers.json (derived ones: of my values)", f"{sv['n']} quoted", f"{sv['n_ok']} correct", sv["n_ok"] == sv["n"],
       "safe" if sv["n_ok"] == sv["n"] else "see the list", "; ".join(f"{c['id']} ({c['log']}): quoted {c['quoted']}, JSON {g(c['json'], 6)}" for c in sv["not_ok"]))
    disc = []
    num_dis = [r["id"] for r in rows + extras if not r["agree"] and not r["id"].startswith("opt_")]
    disc.append({"kind": "numerical", "text": "none: every recomputed number equals the report's, complaint by complaint" if not num_dis else "rows " + ", ".join(num_dis),
                 "cause": "" if not num_dis else "see verify_table.txt"})
    disc.append({"kind": "meaning", "text": f"the 'Prop. bound' ({lb_rep}) is mu(c^2|D|^2/gamma), the lower bound of the report's sandwich; Prop. prop:cond as displayed in sec_theory.tex "
                                            f"only gives OPT_c >= LF(c sqrt(Gamma/gamma)), median {lb_stmt}",
                 "cause": "the measuring agent used the stronger statement of the technical report (prop:app:cond), which sec_theory.tex does not display"})
    if one_val:
        disc.append({"kind": "meaning", "text": "Gamma/gamma at one key is the same number for every complaint", "cause": "at one key w_i = <psi_i, k> u_i, so the scaled Gram matrix is the cosine matrix of the write vectors up to signs"})
    if not exact:
        pq = out["logs"][Q]["prefix"]
        disc.append({"kind": "meaning", "text": f"s*_Z of Q rests on z = {pq['z']} prefix positions that are not shared bit for bit by all sequences",
                     "cause": f"the {pq['z']} prefix keys exist in {pq['n_classes']} bitwise-distinct copies, one per range of sequence lengths (ranges disjoint: {g(pq['classes_are_length_intervals'])}), "
                              f"{100 * pq['rel_diff_between_classes']['min']:.1f}-{100 * pq['rel_diff_between_classes']['max']:.1f} % apart in relative norm; the {pq['n_prefix_keys_all_classes']} copies span "
                              f"{pq['rank_of_all_prefix_copies']} dimensions, so no {pq['z']}-dimensional subspace is common to all groups through them; why the copies differ was not investigated"})
    disc.append({"kind": "sampling", "text": "medians over my 60 complaints differ from the report's medians over all complaints (e.g. LF at c = 0.1: "
                                             + both(lambda R: R["rollback"]["last"]["0.1"]["sample"]["mine_lf"]["median"]) + " against " + P("rollback.{log}.c0.1.median_lf") + ")",
                 "cause": "subsample; on the same complaints the report's rows give my medians, and over all complaints mine equal the report's"})
    # ------------------------------------------------------------------ the compact rows and sentences of the final message
    # (cells 'Q | L'; lists inside a cell are written without blanks, so that the message stays below its word limit)
    by = {r["id"]: r for r in rows + extras}
    la = {n: out["logs"][n]["conditioning"]["last"]["all"] for n in (Q, L)}
    pk = {"gamma": "median_gamma", "Gamma": "median_Gamma", "ratio": "median_Gamma_over_gamma"}

    def three(src):
        return " | ".join(",".join(g(src(n, k)) for k in ("gamma", "Gamma", "ratio")) for n in (Q, L))

    def rule_cells(t, scope=None):
        if scope is None:
            return " | ".join(f"{g(PN[f'rollback.{n}.c{t}.median_lf'])}/{g(PN[f'rollback.{n}.c{t}.median_fo'])}" for n in (Q, L))
        return " | ".join(f"{g(out['logs'][n]['rollback']['last'][t][scope]['mine_lf']['median'])}/{g(out['logs'][n]['rollback']['last'][t][scope]['mine_fo']['median'])}" for n in (Q, L))
    five_vals = {n: sorted({f["mine_n_cap_last"] for f in out["logs"][n]["capacity"]["five"]} | {f["mine_n_cap_stacked"] for f in out["logs"][n]["capacity"]["five"]}) for n in (Q, L)}
    yes = lambda b: "yes" if b else "NO"                                    # noqa: E731
    safe = lambda b, extra="": ("safe" + extra) if b else "NOT safe"        # noqa: E731
    cond_ok = by["gamma"]["agree"] and by["Gamma"]["agree"] and by["ratio"]["agree"]
    message_rows = [
        ["gamma,Gamma,Gamma/gamma (last key)", three(lambda n, k: PN[f"conditioning.{n}.{pk[k]}"]), three(lambda n, k: la[n][f"mine_{k}"]["median"]), yes(cond_ok),
         safe(cond_ok, ", but the ratio is one constant per log (cosine matrix of the write vectors), not a median over complaints" if one_val else "")],
        ["negative-pair share", P("conditioning.{log}.median_share_negative_pairs"), both(lambda R: R["conditioning"]["last"]["all"]["mine_frac_neg"]["median"]), yes(by["neg"]["agree"]), safe(by["neg"]["agree"])],
        ["LF/FO deletions c=0.1", rule_cells("0.1"), rule_cells("0.1", "all"), yes(by["rules_c0.1"]["agree"]), safe(by["rules_c0.1"]["agree"])],
        ["LF/FO deletions c=0.5", rule_cells("0.5"), rule_cells("0.5", "all"), yes(by["rules_c0.5"]["agree"]), safe(by["rules_c0.5"]["agree"])],
        ["bound c=0.1,0.5", lb_rep, lb_mine, yes(by["bound"]["agree"]),
         safe(by["bound"]["agree"], f" only as the report's mu-bound: prop:cond as displayed gives {lb_stmt}")],
        ["capacity (5 complaints/log)", P("capacity.{log}.kept_feasible"), " | ".join(",".join(str(v) for v in five_vals[n]) for n in (Q, L)), yes(by["capacity"]["agree"]), safe(by["capacity"]["agree"])],
        [f"paper_numbers.json ({d['n_values_paper_numbers']} values)", "from results.json", f"{d['n_differences_paper_numbers']} differences, re-run identical" if out["rerun"]["all_pass"] else "differences",
         yes(out["rerun"]["all_pass"]), safe(out["rerun"]["all_pass"])]]
    dev = max(la[n][f"max_rel_dev_{k}"] for n in (Q, L) for k in ("gamma", "Gamma", "ratio"))
    counts_all = allof(lambda R: all(R["rollback"]["last"][t]["all"][f"n_equal_{k}"] == R["rollback"]["last"][t]["all"]["n"] for t in TT for k in ("lf", "fo", "bi", "lb_mu")))
    zq = out["logs"][Q]["capacity"]
    bi_all = out["logs"][L]["rollback"]["last"]["0.1"]["all"]["groups_bi_deletes_all"]
    names = {"stacked_ratio": "stacked Gamma/gamma", "lf_over_lb": "LF/bound ratios", "nu": "least nu " + by["nu"]["mine"], "sstar": "s*_Z", "points": "answer/last-key ranks",
             "bi_c0.1": "BI deletes all 100 on L " + ",".join(bi_all)}
    agree_ex = [x for x in names if by[x]["agree"]]
    wb = lambda n, t: ol[n]["per_tol"][t]["within_budget"]                  # noqa: E731
    message_parts = {
        "numerical": ("Discrepancies: none numerical (per complaint: counts equal, reals within " + f"{dev:.0e}" + "). Sample-of-60 medians differ by sampling only (LF/FO c=0.1: "
                      + rule_cells("0.1", "sample") + ").") if (counts_all and all(r["agree"] for r in rows)) else
                     "Discrepancies: numerical, rows " + ", ".join(r["id"] for r in rows if not r["agree"]) + " (see table).",
        "meaning": "" if exact else f"Wording: Q's s*_Z={PN[f'capacity.{Q}.sstar_Z']} uses {zq['edit_keys']['z_near']} prefix keys shared only approximately "
                                    f"(bitwise z={zq['edit_keys']['z_bitwise']} gives {zq['law']['mine']['z_bitwise']['sstar']}).",
        "extras": "Extras agree: " + "; ".join(names[x] for x in agree_ex) + "." + ("" if len(agree_ex) == len(names) else " DISAGREE: " + ", ".join(x for x in names if x not in agree_ex) + "."),
        "optimum": ("Optimum code (my solver-free certificates; c=0.1,0.3,0.5): upper bounds " + ("all feasible" if ub_all else "NOT all feasible") + "; its lower bounds confirmed for "
                    + " | ".join(",".join(str(wb(n, t)["n_report_lb_confirmed"]) for t in TT) + f" of {ol[n]['n_within_budget']}" for n in (Q, L))
                    + "; certified medians: BI/OPT at most " + per_c(lambda n, t: wb(n, t)["median_bi_over_opt_hi"], 4)
                    + " (report " + per_c(lambda n, t: ol[n]["per_tol"][t]["report"]["opt_median_bi_over_opt_hi"], 4) + "), bound/OPT "
                    + ",".join(f"{min(wb(n, t)['median_lb_over_opt_lo'] for n in (Q, L)):.2f}-{max(wb(n, t)['median_lb_over_opt_hi'] for n in (Q, L)):.2f}" for t in TT)
                    + " (report " + ",".join(f"{min(ol[n]['per_tol'][t]['report']['opt_median_lb_over_opt_lo'] for n in (Q, L)):.2f}-{max(ol[n]['per_tol'][t]['report']['opt_median_lb_over_opt_hi'] for n in (Q, L)):.2f}" for t in TT)
                    + "); its certified counts rest on its branch and bound.")}
    ver = {"definition": "Final message of the independent verification of E5 (linear part) as data; rendered by v_render.py into out/verify_table.txt and out/final_message.txt.",
           "message_rows": message_rows, "message_parts": message_parts,
           "logs": {"Q": Q, "L": L}, "scopes": {"sample": "60 seeded complaints per log (30 image, 30 text)", "all": "every REPAIR-pool D1 complaint", "five": "five of the 60 (capacity)"},
           "table": rows, "extras": extras, "discrepancies": disc,
           "n_complaints": {n: out["logs"][n]["n_complaints"]["mine"] for n in (Q, L)},
           "per_complaint_agreement": {n: {"conditioning_max_rel_dev": max(out["logs"][n]["conditioning"]["last"]["all"][f"max_rel_dev_{k}"] for k in ("gamma", "Gamma", "ratio")),
                                           "counts_equal": {t: {k: out["logs"][n]["rollback"]["last"][t]["all"][f"n_equal_{k}"] for k in ("lf", "fo", "bi", "lb_mu")} for t in ("0.1", "0.3", "0.5")},
                                           "min_rel_margin_of_a_stop_decision": min(out["logs"][n]["rollback"]["last"][t]["all"]["min_rel_margin_of_a_stop_decision"] for t in ("0.1", "0.3", "0.5"))}
                                       for n in (Q, L)},
           "paths": {"verdicts": os.path.join(C.VER, "verdicts.json"), "table": os.path.join(C.OUT, "verify_table.txt"), "results": os.path.join(C.OUT, "verify_results.json"),
                     "code": os.path.join(C.VER, "code")}}
    C.jdump(ver, os.path.join(C.VER, "verdicts.json"))
    print(f"wrote verify_results.json and verdicts.json: {sum(r['agree'] for r in rows)}/{len(rows)} table rows agree, "
          f"{sum(r['agree'] for r in extras)}/{len(extras)} extras; summary literals {sv['n_ok']}/{sv['n']}")


if __name__ == "__main__":
    main()
