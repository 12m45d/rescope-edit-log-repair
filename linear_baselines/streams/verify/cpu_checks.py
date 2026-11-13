"""P12 verifier -- CPU checks on the experiment's raw records (no GPU, own code; the experiment's files are only read).

  A  every raw record out/<log>/*.json: reported set == prefix of MY stream (nestedness), all reported rows are valid
     REPAIR-pool D1 complaints, the held-out rows are exactly MY valid HELD-OUT rows (no REPAIR row), stored W_E flags
     match probe_records_v2, the collateral counts match the per-row flags; COMP records: options as stated, not
     infeasible, non-zero correction, protection rank; REEDIT: l2 = the log's, variant, installed counts; rollback
     budgets.
  B  my own summary of every raw record against out/results.json (aggregate) -- every cell quantity.
  C  the report's two tables (report_tables.txt, verbatim) against the raw records, cell by cell, and the report's
     headline numbers.
  D  split leak (twins) from probe_records_v2 + my split.
  E  sequential == batch: COMP-S vs COMP-B operators and REEDIT-S vs REEDIT-B weights from the saved corrections.
  F  the harness z* fallback bug: source evidence + the first-pass records.
Writes verify/out/cpu_checks.json.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402

OUT = os.path.join(VC.P12, "out")
RES = {}


def nval(n, n_all):
    return n_all if n == "ALL" else int(n)


def my_summary(d):
    """Own summary of one raw record (cell quantities)."""
    ev = d["eval"]
    hr = ev["heldout_rows"]
    aft = [bool(x["reproduced_after"]) for x in hr]
    we = [bool(x["reproduced_WE"]) for x in hr]
    is_end = d["config"] in ("NONE", "ALL")
    rc = [x for x in ev["complaints"] if (not is_end) or x.get("is_complaint_D1")]
    return {"resolved": int(sum(bool(x["D1_resolved"]) for x in rc)), "n": len(rc),
            "D1": 100.0 * sum(not a for a in aft) / len(aft), "n_D1": int(sum(not a for a in aft)), "n_rows": len(aft),
            "fixed": int(sum(a and not w for a, w in zip(aft, we))), "lost": int(sum(w and not a for a, w in zip(aft, we))),
            "n_complaints": int(sum(not w for w in we)), "n_noncomplaints": int(sum(we)),
            "rel": 100.0 * float(np.mean(ev["edits"]["rel"]["em_list"])), "tgen": 100.0 * float(np.mean(ev["edits"]["tgen"]["em_list"])),
            "igen": 100.0 * float(np.mean(ev["edits"]["igen"]["em_list"])), "norm": ev["fro_rel_to_log"],
            "iloc": ev["loc"]["iloc"]["after"], "tloc": ev["loc"]["tloc"]["after"], "iloc_WE": ev["loc"]["iloc"]["WE"],
            "uids": [x["uid"] for x in hr], "aft": aft, "we": we, "groups": [x["group"] for x in hr]}


# ============================================================================ A + B
results = VC.jload(os.path.join(OUT, "results.json"))
A = {}
B = {}
raw = {}
for log in VC.LOGS:
    rows = VC.records(log)
    pool = VC.my_split(rows)
    byg = {r["group"]: r for r in rows}
    streams = {s: [c["group"] for c in VC.my_stream(rows, pool, s)] for s in (0, 1, 2)}
    ho = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    rp = [r for r in rows if pool[r["probe_idx"]] == "repair" and r["y0_valid"]]
    setup = VC.jload(os.path.join(OUT, log, "_setup.json"))
    a = {"n_records": 0, "streams_equal_setup": all(streams[s] == setup["streams"][str(s)] for s in (0, 1, 2)),
         "prefix_ok": 0, "reported_all_repair_D1_valid": 0, "heldout_rows_exact": 0, "heldout_no_repair_row": 0,
         "stored_WE_flags_ok": 0, "collateral_counts_ok": 0, "problems": [], "comp": [], "reedit": [], "rollback": [],
         "nested_pairs_checked": 0, "nested_pairs_ok": 0, "aliases": []}
    b = {"n_rows_compared": 0, "n_fields_compared": 0, "mismatches": []}
    L = results["logs"][log]
    n_all = len(streams[0])
    recs = {}
    files = sorted(glob.glob(os.path.join(OUT, log, "*.json"))) + sorted(glob.glob(os.path.join(OUT, log, "superseded_zfallback", "*.json")))
    for fn in files:
        base = os.path.basename(fn)
        if base.startswith("_") or base.endswith((".failed.json", ".oom.json")):
            continue
        d = VC.jload(fn)
        sup = "superseded_zfallback" in fn
        name = d["config"] + ("@first_pass" if sup else "")
        recs[name] = d
        a["n_records"] += 1
        ms = my_summary(d)
        hr = d["eval"]["heldout_rows"]
        # held-out side
        ok_h = ms["groups"] == [r["group"] for r in ho]
        a["heldout_rows_exact"] += ok_h
        a["heldout_no_repair_row"] += all(pool[byg[g]["probe_idx"]] == "heldout" for g in ms["groups"])
        ok_f = ok_h and ms["we"] == [bool(r["reproduced"]) for r in ho]
        a["stored_WE_flags_ok"] += ok_f
        col = d["eval"]["collateral"]["all"]
        ok_c = (col["complaints_fixed"] == ms["fixed"] and col["noncomplaints_lost"] == ms["lost"]
                and col["n_complaints"] == ms["n_complaints"] and col["n_noncomplaints"] == ms["n_noncomplaints"])
        a["collateral_counts_ok"] += ok_c
        if not (ok_h and ok_f and ok_c):
            a["problems"].append((name, "heldout", ok_h, ok_f, ok_c))
        # reported side
        if d["config"] in ("NONE", "ALL"):
            cg = [f"{x['ptype']}_{x['probe_idx']}" for x in d["eval"]["complaints"]]
            ok_p = sorted(cg) == sorted(r["group"] for r in rp)
            ok_r = True
        else:
            n = nval(d["n"], n_all)
            cg = [f"{x['ptype']}_{x['probe_idx']}" for x in d["eval"]["complaints"]]
            ok_p = d["reported_groups"] == streams[d["seed"]][:n] and cg == d["reported_groups"] and d["n_reported"] == n
            ok_r = all(pool[byg[g]["probe_idx"]] == "repair" and byg[g]["is_complaint_D1"] and byg[g]["y0_valid"] for g in cg)
        a["prefix_ok"] += ok_p
        a["reported_all_repair_D1_valid"] += ok_r
        if not (ok_p and ok_r):
            a["problems"].append((name, "reported", ok_p, ok_r))
        if d.get("alias_of"):
            a["aliases"].append((d["config"], d["alias_of"]))
        m = d["method"]
        opt = d.get("options", {})
        if m.startswith("COMP"):
            meta, rk, ex = d["meta"], d["rank"], d["extra"]
            a["comp"].append({"config": name, "protect": opt.get("protect"), "query": opt.get("query"), "form": opt.get("form"),
                              "infeasible": meta.get("infeasible", any(s["infeasible"] for s in ex.get("steps", []))),
                              "rank": rk["correction"], "n_query": rk["n_query_keys"], "n_dropped": rk["n_dropped"],
                              "protect_rank": rk.get("protection_static", rk["protection"]), "norm": d["eval"]["fro_rel_to_log"],
                              "P1_rel": meta.get("P1_rel"), "restore_err_max": max(p["restore_rel_err"] for p in ex["per_complaint"]),
                              "n_ans_tok_sum": int(sum(byg[g]["n_ans_tok"] for g in d["reported_groups"])),
                              "passes": d["cost"]["passes"], "n_backward": d["cost"]["n_backward"]})
        elif m.startswith("REEDIT"):
            ex = d["extra"]
            a["reedit"].append({"config": name, "l2": d["meta"].get("l2"), "variant": d["meta"].get("variant"),
                                "installed": ex.get("installed"), "not_installed": ex.get("not_installed"),
                                "n": d["n_reported"], "zproc": ex.get("zproc"), "solve_attainment": ex.get("solve_attainment"),
                                "norm": d["eval"]["fro_rel_to_log"], "kind": d["meta"].get("method"),
                                "resolved": ms["resolved"]})
        elif m.startswith("RB_"):
            ex = d["extra"]
            a["rollback"].append({"config": name, "k": ex["k"], "budget": ex.get("budget", ex.get("budget_each")),
                                  "resolved_internal": ex.get("resolved_internal"), "norm": d["eval"]["fro_rel_to_log"],
                                  "per_complaint_k_max": max(ex["per_complaint_k"]) if "per_complaint_k" in ex else None})
        # ---- B: my summary against results.json
        row = (L.get("rows_first_pass", {}) if sup else L["rows"]).get(d["config"])
        if row is None:
            if not sup:
                b["mismatches"].append((name, "missing in results.json"))
            continue
        b["n_rows_compared"] += 1
        pairs = [("resolved", ms["resolved"], row["reported"]["resolved"]), ("n", ms["n"], row["reported"]["n"]),
                 ("D1", ms["D1"], row["heldout"]["D1_rate"]), ("fixed", ms["fixed"], row["heldout"]["complaints_fixed"]),
                 ("lost", ms["lost"], row["heldout"]["noncomplaints_lost"]), ("rel", ms["rel"], row["edits"]["rel"]["exact"]),
                 ("tgen", ms["tgen"], row["edits"]["tgen"]["exact"]), ("igen", ms["igen"], row["edits"]["igen"]["exact"]),
                 ("norm", ms["norm"], row["fro_rel_to_log"]), ("iloc", ms["iloc"], row["loc"]["iloc"]["after"])]
        for k, x, y in pairs:
            b["n_fields_compared"] += 1
            if abs(float(x) - float(y)) > 1e-9:
                b["mismatches"].append((name, k, x, y))
        lo, hi = VC.cluster_boot_ci([0.0 if v else 1.0 for v in ms["aft"]], ms["uids"])
        ci = row["heldout"]["D1_rate_ci"]
        b["n_fields_compared"] += 2
        if abs(100 * lo - ci["lo"]) > 1e-9 or abs(100 * hi - ci["hi"]) > 1e-9:
            b["mismatches"].append((name, "ci", (100 * lo, 100 * hi), (ci["lo"], ci["hi"])))
        p, _, _ = VC.mcnemar_exact(ms["aft"], ms["we"])
        b["n_fields_compared"] += 1
        if abs(p - row["heldout"]["paired_vs_WE"]["p_exact"]) > 1e-12:
            b["mismatches"].append((name, "mcnemar", p, row["heldout"]["paired_vs_WE"]["p_exact"]))
    # nestedness of the reported sets along n (per method, seed)
    for name1, d1 in recs.items():
        for name2, d2 in recs.items():
            if ("@" in name1) or ("@" in name2) or d1["config"] in ("NONE", "ALL") or d2["config"] in ("NONE", "ALL"):
                continue
            if d1["method"] == d2["method"] and d1["seed"] == d2["seed"] and d1["n_reported"] < d2["n_reported"]:
                a["nested_pairs_checked"] += 1
                a["nested_pairs_ok"] += d2["reported_groups"][:d1["n_reported"]] == d1["reported_groups"]
    a["n_stream"] = n_all
    a["n_heldout_valid"], a["n_repair_valid"] = len(ho), len(rp)
    a["n_repair_complaints"] = sum(1 for r in rp if r["is_complaint_D1"])
    a["n_repair_noncomplaints"] = sum(1 for r in rp if r["reproduced"] is True)
    a["n_heldout_complaints"] = sum(1 for r in ho if r["is_complaint_D1"])
    a["stratification_ok"] = all(
        sum(byg[g]["ptype"] == "image" for g in streams[0][:n]) == n // 2 for n in (1, 2, 5, 10, 20, 50, 100))
    a["comp_all_feasible"] = all((not c["infeasible"]) and c["rank"] > 0 and c["norm"] > 0 for c in a["comp"])
    # joint (batch) solves: none may be infeasible or zero.  Sequential variants: a step of rank 0 is legitimate only when
    # the complaint's model input repeats an earlier complaint of the stream (same prompt, same image BYTES, same y0 ids)
    is_seq = lambda c: ("COMP_S_" in c["config"]) or ("COMP_SP_" in c["config"])  # noqa: E731
    a["comp_batch_infeasible_or_zero"] = [c["config"] for c in a["comp"] if not is_seq(c) and (c["infeasible"] or c["rank"] <= 0 or c["norm"] <= 0)]
    a["comp_all_nonzero"] = all(c["rank"] > 0 and c["norm"] > 0 for c in a["comp"])
    import hashlib
    IMG = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"
    _md5 = {}

    def inp(g):
        r = byg[g]
        if r["image"] is not None and r["image"] not in _md5:
            _md5[r["image"]] = hashlib.md5(open(os.path.join(IMG, r["image"]), "rb").read()).hexdigest()
        return (r["ptype"], r["prompt"], _md5.get(r["image"]), tuple(r["y0_ids"]))
    zs, zs_rep = set(), set()
    for name, d in recs.items():
        if "@" in name or not d["method"].startswith("COMP_S"):
            continue
        seen = set()
        for st_ in d["extra"].get("steps", []):
            k = inp(st_["group"])
            if st_["infeasible"] or st_["rank_kept"] == 0:
                zs.add(st_["group"])
                if k in seen:
                    zs_rep.add(st_["group"])
            seen.add(k)
    a["comp_seq_zero_rank_steps"] = sorted(zs)
    a["comp_seq_zero_rank_steps_repeating_an_earlier_input"] = sorted(zs_rep)
    a["comp_seq_zero_rank_steps_all_repeats"] = zs == zs_rep
    a["comp_options_ok"] = all(c["form"] == "adapter" and c["query"] == ("all" if "ALLPOS" in c["config"] else "answer")
                               and c["protect"] == (["K_ans", "probe:100000"] if ("_BP_" in c["config"] or "_SP_" in c["config"]) else ["K_ans"])
                               for c in a["comp"])
    a["comp_answer_query_count_ok"] = all(c["n_query"] == c["n_ans_tok_sum"] for c in a["comp"] if "ALLPOS" not in c["config"])
    a["comp_no_backward"] = all(c["n_backward"] == 0 for c in a["comp"])
    a["comp_restore_err_max"] = max(c["restore_err_max"] for c in a["comp"])
    a["comp_protect_ranks"] = sorted({(("B+" if ("_BP_" in c["config"] or "_SP_" in c["config"]) else "B"), c["protect_rank"]) for c in a["comp"]})
    a["reedit_l2_values"] = sorted({c["l2"] for c in a["reedit"]})
    a["reedit_variants"] = sorted({c["variant"] for c in a["reedit"]})
    a["rb_tot_budgets"] = sorted({c["budget"] for c in a["rollback"] if "RB_TOT" in c["config"]})
    a["rb_union_budgets"] = sorted({c["budget"] for c in a["rollback"] if "RB_UNION" in c["config"]})
    a["rb_tot_k_max"] = max(c["k"] for c in a["rollback"] if "RB_TOT" in c["config"])
    a["rb_union_per_complaint_k_max"] = max(c["per_complaint_k_max"] for c in a["rollback"] if "RB_UNION" in c["config"])
    a["log_l2"] = setup["l2"]
    A[log], B[log], raw[log] = a, b, recs
RES["A_records"] = A
RES["B_aggregate"] = B

# ============================================================================ C  report tables vs raw records
LABM = {"COMP-B": "COMP_B", "COMP-B+": "COMP_BP", "ALLPOS": "COMP_B_ALLPOS", "REEDIT-B": "REEDIT_B", "RB-TOT": "RB_TOT", "RB-UNION": "RB_UNION"}
NS = (1, 10, 100, "ALL")
C = {"cells": 0, "cells_agree": 0, "disagree": [], "dagger_cells": [], "dagger_ok": True, "empty_cells_ok": True}
cur = None
for line in open(os.path.join(HERE, "report_tables.txt")):
    line = line.strip()
    if line.startswith("["):
        cur = line.strip("[]")
        continue
    if not line.startswith("|"):
        continue
    parts = [x.strip() for x in line.strip("|").split("|")]
    m = LABM[parts[0]]
    for n, cellv in zip(NS, parts[1:]):
        name = f"{m}__n{n}__s0"
        d = raw[cur].get(name)
        fp = False
        if d is None and (name + "@first_pass") in raw[cur]:
            d, fp = raw[cur][name + "@first_pass"], True
        if cellv in ("–", "-"):
            C["empty_cells_ok"] &= d is None
            continue
        dag = cellv.endswith("†")
        cellv = cellv.rstrip("†").replace("−", "-")
        C["cells"] += 1
        if d is None:
            C["disagree"].append((cur, name, "no record"))
            continue
        ms = my_summary(d)
        mine = (f"{ms['resolved']}/{ms['n']};{ms['D1']:.1f};+{ms['fixed']}/-{ms['lost']};"
                f"{ms['rel']:.0f}/{ms['tgen']:.0f}/{ms['igen']:.0f};{ms['norm']:.2f}")
        if mine == cellv:
            C["cells_agree"] += 1
        else:
            C["disagree"].append((cur, name, cellv, mine))
        if dag or fp:
            C["dagger_cells"].append((cur, name, dag, fp))
            C["dagger_ok"] &= (dag == fp)
RES["C_report_tables"] = C

# headline numbers of the report (section 0 / 3 / 4 / 6), from the raw records with my own statistics
Hd = {}
for log in VC.LOGS:
    none, best, w0 = my_summary(raw[log]["NONE"]), my_summary(raw[log]["COMP_B__nALL__s0"]), my_summary(raw[log]["ALL"])
    rows = VC.records(log)
    pool = VC.my_split(rows)
    # twins: held-out rows whose query (ptype, prompt, image) also occurs in the REPAIR pool
    key = lambda r: (r["ptype"], r["prompt"], r["image"])  # noqa: E731
    repk = {}
    for r in rows:
        if pool[r["probe_idx"]] == "repair":
            repk.setdefault(key(r), []).append(r)
    ho = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    tw = [r for r in ho if key(r) in repk]
    twg = {r["group"] for r in tw}
    dis = [k for k, g in enumerate(best["groups"]) if g not in twg]
    lo, hi = VC.cluster_boot_ci([0.0 if v else 1.0 for v in best["aft"]], best["uids"])
    p, bfix, clost = VC.mcnemar_exact(best["aft"], best["we"])
    pd, _, _ = VC.mcnemar_exact([best["aft"][k] for k in dis], [best["we"][k] for k in dis])
    dfix = sum(best["aft"][k] and not best["we"][k] for k in dis)
    dlost = sum(best["we"][k] and not best["aft"][k] for k in dis)
    dn_c = sum(not best["we"][k] for k in dis)
    dn_n = sum(best["we"][k] for k in dis)
    d = raw[log]["COMP_B__nALL__s0"]
    # selection: lowest held-out D1 among the non-endpoint configs with Rel exact >= Rel_WE - 2 (first-pass REEDIT excluded)
    cand = []
    for name, dd in raw[log].items():
        if "@" in name or dd["config"] in ("NONE", "ALL") or dd["method"] in ("REEDIT_SC", "COMP_BR"):
            continue
        s = my_summary(dd)
        if s["rel"] >= none["rel"] - 2.0:
            cand.append((s["D1"], name, s["rel"]))
    cand.sort()
    best_lin = [c for c in cand if c[0] == cand[0][0]]
    Hd[log] = {"none_D1": none["D1"], "none_n_D1": none["n_D1"], "none_n": none["n_rows"],
               "none_rel_tgen_igen": [none["rel"], none["tgen"], none["igen"]],
               "W0_n_D1_heldout": w0["n_D1"], "W0_repair_D1": int(sum(not bool(x["D1_resolved"]) for x in raw[log]["ALL"]["eval"]["complaints"])),
               "best_D1": best["D1"], "best_ci": [100 * lo, 100 * hi], "best_delta": best["D1"] - none["D1"], "best_mcnemar_p": p,
               "best_fixed": bfix, "best_lost": clost, "best_rel": best["rel"], "best_resolved": [best["resolved"], best["n"]],
               "best_rank": d["rank"]["correction"], "best_norm": best["norm"], "best_n_forward": d["cost"]["n_forward"],
               "best_n_backward": d["cost"]["n_backward"], "best_iloc": [best["iloc_WE"], best["iloc"]],
               "selection_n_candidates": len(cand), "selection_minimum": [list(c) for c in best_lin], "selection_top5": [list(c) for c in cand[:5]],
               "twins_valid_heldout": len(tw), "twins_heldout_complaints": int(sum(bool(r["is_complaint_D1"]) for r in tw)),
               "twins_by_ptype": {pt: int(sum(r["ptype"] == pt for r in tw)) for pt in ("image", "text")},
               "disjoint_n": len(dis), "disjoint_none_D1": 100.0 * sum(not best["we"][k] for k in dis) / len(dis),
               "disjoint_best_D1": 100.0 * sum(not best["aft"][k] for k in dis) / len(dis), "disjoint_mcnemar_p": pd,
               "disjoint_fix_rate": 100.0 * dfix / dn_c, "disjoint_loss_rate": 100.0 * dlost / dn_n,
               "comp_unresolved_share_at_ALL": 100.0 * (best["n"] - best["resolved"]) / best["n"],
               "reedit_net_by_n": {str(dd["n"]) + ("†" if "@" in nm else ""): my_summary(dd)["fixed"] - my_summary(dd)["lost"]
                                   for nm, dd in sorted(raw[log].items()) if dd["method"] == "REEDIT_B" and dd["seed"] == 0
                                   and not (("@" in nm) and dd["config"] in raw[log])},
               "bp_all": [my_summary(raw[log]["COMP_BP__nALL__s0"])["D1"]],
               "rb_union_all_k": raw[log]["RB_UNION__nALL__s0"]["extra"]["k"], "rb_union_100_k": raw[log]["RB_UNION__n100__s0"]["extra"]["k"],
               "rb_tot_all": [raw[log]["RB_TOT__nALL__s0"]["extra"]["k"], my_summary(raw[log]["RB_TOT__nALL__s0"])["D1"], my_summary(raw[log]["RB_TOT__nALL__s0"])["rel"]],
               "allpos": {str(dd["n"]): [my_summary(dd)["resolved"], my_summary(dd)["n"], dd["rank"]["correction"]]
                          for nm, dd in sorted(raw[log].items()) if dd["method"] == "COMP_B_ALLPOS"},
               "sanity_none": raw[log]["NONE"]["sanity"], "sanity_all": raw[log]["ALL"]["sanity"]}
RES["C_headlines"] = Hd

# ============================================================================ E  sequential == batch (saved corrections)
E = {}
try:
    import torch
    torch.set_num_threads(8)
    for log in VC.LOGS:
        cd = os.path.join(OUT, log, "corr")
        e = {}
        for n in ("10", "100", "ALL"):
            for ms_, mb in (("COMP_S", "COMP_B"), ("COMP_SP", "COMP_BP")):
                pa, pb = os.path.join(cd, f"{ms_}__n{n}__s0.pt"), os.path.join(cd, f"{mb}__n{n}__s0.pt")
                if os.path.exists(pa) and os.path.exists(pb):
                    x, y = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
                    Xa, Xb = x["Lf"] @ x["Rf"].T, y["Lf"] @ y["Rf"].T
                    e[f"{ms_}_vs_{mb}__n{n}"] = float((Xa - Xb).norm() / Xb.norm())
                    del Xa, Xb, x, y
            pa, pb = os.path.join(cd, f"REEDIT_S__n{n}__s0.pt"), os.path.join(cd, f"REEDIT_B__n{n}__s0.pt")
            if os.path.exists(pa) and os.path.exists(pb):
                x, y = torch.load(pa, map_location="cpu"), torch.load(pb, map_location="cpu")
                if x["kind"] == "weight" and y["kind"] == "weight":
                    comp = torch.load(os.path.join(VC.TB_ROOT, log, "composite_delta.pt"), map_location="cpu")
                    # W_E is not stored here: ||W_S - W_B|| relative to the recorded ||W_B - W_E|| (norm * ||log||)
                    nb = raw[log][f"REEDIT_B__n{n}__s0"]["eval"]["fro"]
                    e[f"REEDIT_S_vs_REEDIT_B__n{n}"] = float((x["W"].double() - y["W"].double()).norm() / nb)
                    del comp
        E[log] = e
except Exception as ex:      # noqa: BLE001
    E["error"] = repr(ex)
RES["E_seq_vs_batch"] = E

# ============================================================================ F  harness z* fallback bug (source evidence)
src = open(os.path.join(VC.V3, "harness", "methods.py")).read().split("\n")
i_elif = next(i for i, l in enumerate(src) if l.strip().startswith("elif chunk is None:") and "compute_targets" in src[i + 1])
i_else = next(i for i in range(i_elif, len(src)) if src[i].strip().startswith("else:") and "chunk option" in src[i])
i_for = next(i for i in range(i_else, len(src)) if src[i].strip().startswith("for k, over in enumerate("))
ind = lambda l: len(l) - len(l.lstrip())  # noqa: E731
F = {"file": "07_repair_v3/harness/methods.py", "line_elif_chunk_None": i_elif + 1, "line_else_chunk": i_else + 1,
     "line_fallback_loop": i_for + 1, "fallback_loop_indent": ind(src[i_for]), "else_indent": ind(src[i_else]),
     "fallback_loop_inside_else_branch": ind(src[i_for]) > ind(src[i_else]),
     "dead_inner_test_line": next(i + 1 for i in range(i_for, i_for + 8) if src[i].strip() == "if chunk is None:"),
     "first_pass_records": {log: {nm.split("@")[0]: {"not_installed": dd["extra"]["not_installed"], "zproc": dd["extra"].get("zproc"),
                                                    "resolved": my_summary(dd)["resolved"], "D1": my_summary(dd)["D1"]}
                                  for nm, dd in raw[log].items() if "@" in nm} for log in VC.LOGS},
     "reruns_present": {log: sorted(nm for nm, dd in raw[log].items() if "@" not in nm and dd["method"].startswith("REEDIT")
                                    and dd["extra"].get("zproc")) for log in VC.LOGS}}
RES["F_harness_bug"] = F

VC.jdump(RES, os.path.join(VC.VOUT, "cpu_checks.json"))
for log in VC.LOGS:
    a, b = A[log], B[log]
    print(log, "records", a["n_records"], "| prefix", a["prefix_ok"], "| repair-D1-valid", a["reported_all_repair_D1_valid"], "| heldout exact",
          a["heldout_rows_exact"], "| stored flags", a["stored_WE_flags_ok"], "| collateral", a["collateral_counts_ok"], "| nested",
          a["nested_pairs_ok"], "/", a["nested_pairs_checked"], "| problems", a["problems"][:5])
    print("   comp feasible", a["comp_all_feasible"], "options", a["comp_options_ok"], "nq", a["comp_answer_query_count_ok"], "protect ranks", a["comp_protect_ranks"],
          "restore err max", a["comp_restore_err_max"], "| reedit l2", a["reedit_l2_values"], a["reedit_variants"], "| rb budgets", a["rb_tot_budgets"],
          a["rb_union_budgets"], "k max", a["rb_tot_k_max"], a["rb_union_per_complaint_k_max"])
    print("   aggregate: rows", b["n_rows_compared"], "fields", b["n_fields_compared"], "mismatches", b["mismatches"][:6])
    print("   headlines", {k: v for k, v in Hd[log].items() if k not in ("sanity_none", "sanity_all")})
print("report tables:", {k: v for k, v in C.items()})
print("seq vs batch:", E)
print("harness bug:", {k: v for k, v in F.items() if k != "first_pass_records"})
print("first pass:", F["first_pass_records"])
