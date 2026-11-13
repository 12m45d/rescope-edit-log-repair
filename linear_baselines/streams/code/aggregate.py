"""P12 -- out/<log>/*.json (+ d2/*.json, superseded_zfallback/*.json, text_h2h rows) -> out/results.json.
Every number of results.md and of the figures comes from out/results.json.

usage: python aggregate.py
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, _gd + "/linear_baselines/streams", needs=("linear_baselines/streams/out/TB-Q100-ae-l2_3", "linear_baselines/streams/out/TB-L100-ae-l2_3"))

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P12 = os.path.dirname(HERE)
OUT = os.path.join(P12, "out")
sys.path.insert(0, HERE)
import p12_common as C  # noqa: E402  (protocol constants, the plan; puts 07_repair_v3 on sys.path)
from harness.stats import bootstrap_ci, mcnemar  # noqa: E402

LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
METHODS = ("COMP_B", "COMP_BP", "COMP_S", "COMP_SP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION",
           "REEDIT_SC", "COMP_BR")
EXTRA = ("REEDIT_SC", "COMP_BR")                  # exploratory, outside the protocol
N_GRID = (1, 2, 5, 10, 20, 50, 100, "ALL")
NB = 2000
REL_TOL = 2.0                     # "Rel exact within 2 points of the unrepaired log"


def jload(p):
    with open(p) as f:
        return json.load(f)


def nkey(n):
    return 10 ** 9 if n == "ALL" else int(n)


def leak_map(log):
    """HELD-OUT rows whose query (ptype, prompt, image) is IDENTICAL to a REPAIR-pool row's ("twins").  The
    harness splits the 500 probe items by uid, but VLKEB eval items share image-locality probes, so the same
    (image, question) can sit in both pools.  Computed from probe_records_v2.jsonl + the harness split (CPU)."""
    p = os.path.join(OUT, log, "_leak.json")
    if os.path.exists(p):
        return jload(p)
    from collections import Counter
    from harness.common import TB, TB_ROOT
    from harness import sampler as SM

    class _H:
        pass
    cfgl = TB.LOGS[log]
    items, _ = TB.load_stream(cfgl["stream"])
    H = _H()
    H.probes, _ = TB.select_probes(items, key_exclude=(cfgl["backbone"] == "llava"))
    H._setup = {}
    rows = TB.jsonl_read(os.path.join(TB_ROOT, log, "probe_records_v2.jsonl"))
    pl = SM.pool_of(H)
    key = lambda r: (r["ptype"], r["prompt"], r["image"])  # noqa: E731
    rep = {}
    for r in rows:
        if pl[r["probe_idx"]] == "repair":
            rep.setdefault(key(r), []).append(r)
    twins = {}
    for r in rows:
        if pl[r["probe_idx"]] == "heldout" and key(r) in rep:
            twins[r["group"]] = {"repair_groups": [x["group"] for x in rep[key(r)]], "ptype": r["ptype"], "valid": bool(r["y0_valid"]),
                                 "is_complaint_D1": bool(r["is_complaint_D1"]),
                                 "repair_is_complaint_D1": [bool(x["is_complaint_D1"]) for x in rep[key(r)]]}
    rc = [r for r in rows if pl[r["probe_idx"]] == "repair" and r["is_complaint_D1"]]
    cnt = Counter(key(r) for r in rc)
    first = {}
    for r in rc:
        first.setdefault(key(r), r)
    tv = [t for t in twins.values() if t["valid"]]
    ho_valid = [r for r in rows if pl[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    out = {"definition": "HELD-OUT rows whose (ptype, prompt, image) is identical to a REPAIR-pool row",
           "twins": twins, "n_twins_valid": len(tv), "n_twin_complaints": sum(t["is_complaint_D1"] for t in tv),
           "n_twin_noncomplaints": sum(not t["is_complaint_D1"] for t in tv),
           "n_twin_complaints_of_repair_complaint": sum(t["is_complaint_D1"] and any(t["repair_is_complaint_D1"]) for t in tv),
           "by_ptype": {pt: sum(t["ptype"] == pt for t in tv) for pt in ("image", "text")},
           "n_heldout_valid": len(ho_valid), "n_heldout_valid_complaints": sum(bool(r["is_complaint_D1"]) for r in ho_valid),
           "heldout_valid_groups": [r["group"] for r in ho_valid],
           "n_distinct_queries": {pt: len({key(r) for r in rows if r["ptype"] == pt}) for pt in ("image", "text")},
           "n_rows": {pt: sum(r["ptype"] == pt for r in rows) for pt in ("image", "text")},
           "repair_complaints": len(rc), "repair_complaints_distinct_queries": len(cnt),
           "repair_redundant_complaints": len(rc) - len(cnt),
           "repair_redundant_answer_keys": int(sum(r["n_ans_tok"] for r in rc) - sum(first[k]["n_ans_tok"] for k in cnt))}
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    return out


def pool_stats(hr, idx, ci=True):
    aft = [bool(hr[k]["reproduced_after"]) for k in idx]
    we = [bool(hr[k]["reproduced_WE"]) for k in idx]
    n = max(len(idx), 1)
    out = {"n_rows": len(idx), "D1_rate": 100.0 * sum(not a for a in aft) / n, "D1_rate_WE": 100.0 * sum(not w for w in we) / n,
           "n_D1_after": int(sum(not a for a in aft)),
           "complaints_fixed": int(sum(a and not w for a, w in zip(aft, we))), "n_complaints": int(sum(not w for w in we)),
           "noncomplaints_lost": int(sum(w and not a for a, w in zip(aft, we))), "n_noncomplaints": int(sum(we))}
    if ci and idx:
        c = bootstrap_ci([0.0 if a else 1.0 for a in aft], [hr[k]["uid"] for k in idx], NB, 0)
        out["D1_rate_ci"] = {k: (100.0 * v if k in ("mean", "lo", "hi") else v) for k, v in c.items()}
        out["paired_vs_WE"] = mcnemar(aft, we)
    return out


def summarize(d, leak=None):
    ev = d["eval"]
    hr = ev["heldout_rows"]
    uids = [r["uid"] for r in hr]
    d1_after = [0.0 if r["reproduced_after"] else 1.0 for r in hr]
    d1_we = [0.0 if r["reproduced_WE"] else 1.0 for r in hr]
    is_end = d["config"] in ("NONE", "ALL")
    rc = ev["complaints"] if not is_end else [r for r in ev["complaints"] if r["is_complaint_D1"]]
    col = ev["collateral"]
    s = {"log": d["log"], "config": d["config"], "method": d["method"], "n": d["n"], "seed": d["seed"],
         "n_reported": len(rc) if not is_end else 0, "alias_of": d.get("alias_of"),
         "reported": {"n": len(rc), "resolved": int(sum(r["D1_resolved"] for r in rc)),
                      "by_ptype": {pt: {"n": sum(r["ptype"] == pt for r in rc),
                                        "resolved": int(sum(r["D1_resolved"] for r in rc if r["ptype"] == pt))}
                                   for pt in ("image", "text")}},
         "heldout": {"n_rows": len(hr), "n_items": len(set(uids)),
                     "D1_rate": 100.0 * float(np.mean(d1_after)), "D1_rate_WE": 100.0 * float(np.mean(d1_we)),
                     "D1_rate_ci": {k: (100.0 * v if k in ("mean", "lo", "hi") else v)
                                    for k, v in bootstrap_ci(d1_after, uids, NB, 0).items()},
                     "n_D1_after": int(sum(d1_after)),
                     "complaints_fixed": col["all"]["complaints_fixed"], "n_complaints": col["all"]["n_complaints"],
                     "noncomplaints_lost": col["all"]["noncomplaints_lost"], "n_noncomplaints": col["all"]["n_noncomplaints"],
                     "by_ptype": {pt: dict(col[pt], D1_rate=100.0 * float(np.mean([x for x, r in zip(d1_after, hr) if r["ptype"] == pt])),
                                           D1_rate_WE=100.0 * float(np.mean([x for x, r in zip(d1_we, hr) if r["ptype"] == pt])))
                                  for pt in ("image", "text")},
                     "paired_vs_WE": mcnemar([r["reproduced_after"] for r in hr], [r["reproduced_WE"] for r in hr])},
         "edits": {m: {k: ev["edits"][m][k] for k in ("exact", "token", "exact_WE", "token_WE", "n_broken", "n_fixed")}
                   for m in ("rel", "tgen", "igen")},
         "edits_broken_total": ev["edits_broken_total"], "edits_fixed_total": ev["edits_fixed_total"],
         "loc": {m: {"after": ev["loc"][m]["after"], "WE": ev["loc"][m]["WE"],
                     "delta": ev["loc"][m]["after"] - ev["loc"][m]["WE"]} for m in ("tloc", "iloc")},
         "fro": ev["fro"], "fro_rel_to_log": ev["fro_rel_to_log"],
         "rank": d.get("rank", {}),
         "cost": {"passes": d["cost"].get("passes"), "sec": d["cost"].get("sec_total"),
                  "sec_zopt": d["cost"].get("sec_zopt"), "n_backward": d["cost"].get("n_backward"),
                  "n_forward": d["cost"].get("n_forward"), "flops_total": d["cost"].get("flops_total"),
                  "n_calls": d["cost"].get("n_calls")},
         "repair_wall_sec": d.get("repair_wall_sec"), "eval_sec": d.get("eval_sec"),
         "peak_gpu_gib": (d.get("peak_gpu_bytes") or 0) / 2 ** 30, "gpu_procs": d.get("gpu_procs"),
         "sanity": d.get("sanity")}
    s["rel_within_tol"] = bool(s["edits"]["rel"]["exact"] >= s["edits"]["rel"]["exact_WE"] - REL_TOL)
    if leak is not None:
        tw = leak["twins"]
        assert [r["group"] for r in hr] == leak["heldout_valid_groups"], "held-out rows differ from the split"
        dis = [k for k, r in enumerate(hr) if r["group"] not in tw]
        twi = [k for k, r in enumerate(hr) if r["group"] in tw]
        rep = set(d.get("reported_groups") or [])
        tr = [k for k in twi if rep & set(tw[hr[k]["group"]]["repair_groups"])]
        s["heldout_disjoint"] = pool_stats(hr, dis)
        s["heldout_twins"] = pool_stats(hr, twi, ci=False)
        s["heldout_twins_of_reported"] = pool_stats(hr, tr, ci=False)
        # consistency: a held-out twin of a reported complaint is the same model input, so its outcome (held-out
        # evaluation, suffix engine) must equal the reported complaint's outcome (complaint evaluation, full forward)
        res = {f"{c['ptype']}_{c['probe_idx']}": bool(c["D1_resolved"]) for c in ev["complaints"]} if not is_end else {}
        pairs = [(res[g2], bool(hr[k]["reproduced_after"])) for k in tr for g2 in tw[hr[k]["group"]]["repair_groups"] if g2 in res]
        s["twin_consistency"] = {"n": len(pairs), "agree": int(sum(a == b for a, b in pairs))}
    ex, meta = d.get("extra", {}), d.get("meta", {})
    m = d["method"]
    if m.startswith("COMP"):
        pc = ex.get("per_complaint", [])
        rmax = max((p["restore_rel_err"] for p in pc), default=None)
        unres = [(p, r) for p, r in zip(pc, rc) if not r["D1_resolved"]]
        s["comp"] = {"n_query_keys": s["rank"].get("n_query_keys"), "rank": s["rank"].get("correction"),
                     "n_dropped": s["rank"].get("n_dropped"), "protect_rank": s["rank"].get("protection"),
                     "d_in": s["rank"].get("d_in"),
                     "rank_used_frac": ((s["rank"].get("correction") or 0) + (s["rank"].get("protection") or 0)) / max(s["rank"].get("d_in") or 1, 1),
                     "restore_rel_err_max": rmax,
                     "feasible": bool(rmax is not None and rmax < 1e-6),
                     "unresolved": len(unres),
                     "unresolved_with_output_restored": int(sum(p["restore_rel_err"] < 1e-6 for p, r in unres)),
                     "unresolved_tok_acc_median": float(np.median([r["tok_acc"] for p, r in unres])) if unres else None,
                     # which reported complaints stay unresolved (listed for the all-position variant only: few rows)
                     "unresolved_detail": ([{"group": p["group"], "restore_rel_err": p["restore_rel_err"], "n_keys": p["n_keys"],
                                             "tok_acc": r["tok_acc"], "n_tok": r["n_tok"], "first_mismatch": r["first_mismatch"]}
                                            for p, r in unres] if m == "COMP_B_ALLPOS" else None),
                     "identity_vs_batch": ex.get("identity_vs_batch"),
                     "P1_rel": meta.get("P1_rel"), "cond": meta.get("cond"), "e_mean": meta.get("e_mean"),
                     "max_step_cond": max((x["cond"] for x in ex.get("steps", []) if isinstance(x["cond"], (int, float))), default=None)}
    if m.startswith("REEDIT"):
        s["reedit"] = {"installed": ex.get("installed"), "not_installed": ex.get("not_installed"),
                       "solve_attainment": ex.get("solve_attainment"), "dW_over_W0": ex.get("dW_over_W0"),
                       "zproc": ex.get("zproc"), "n_batches": len(ex.get("per_batch", [])) or 1,
                       "attainment_per_batch": [b.get("solve_attainment") for b in ex.get("per_batch", [])],
                       "installed_not_resolved": int(sum((not r["D1_resolved"]) for r in rc)) - len(ex.get("not_installed") or [])}
    if m.startswith("RB_"):
        s["rollback"] = {"k": ex.get("k"), "R": ex.get("R"), "resolved_internal": ex.get("resolved_internal"),
                         "n_resolved_individually": ex.get("n_resolved_individually")}
    return s


def seed_stats(rows, f):
    v = [f(r) for r in rows]
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_seeds": len(v),
            "values": [float(x) for x in v]}


CURVES = {"heldout_disjoint_D1_rate": lambda r: r["heldout_disjoint"]["D1_rate"],
          "heldout_disjoint_fixed": lambda r: r["heldout_disjoint"]["complaints_fixed"],
          "heldout_disjoint_lost": lambda r: r["heldout_disjoint"]["noncomplaints_lost"],
          "twins_of_reported": lambda r: r["heldout_twins_of_reported"]["n_rows"],
          "twins_of_reported_fixed": lambda r: r["heldout_twins_of_reported"]["complaints_fixed"],
          "heldout_D1_rate": lambda r: r["heldout"]["D1_rate"],
          "heldout_complaints_fixed": lambda r: r["heldout"]["complaints_fixed"],
          "heldout_noncomplaints_lost": lambda r: r["heldout"]["noncomplaints_lost"],
          "reported_resolved_frac": lambda r: 100.0 * r["reported"]["resolved"] / max(r["reported"]["n"], 1),
          "reported_resolved": lambda r: r["reported"]["resolved"],
          "rel_exact": lambda r: r["edits"]["rel"]["exact"], "tgen_exact": lambda r: r["edits"]["tgen"]["exact"],
          "igen_exact": lambda r: r["edits"]["igen"]["exact"],
          "edits_broken_total": lambda r: r["edits_broken_total"], "edits_fixed_total": lambda r: r["edits_fixed_total"],
          "rel_broken": lambda r: r["edits"]["rel"]["n_broken"],
          "tloc": lambda r: r["loc"]["tloc"]["after"], "iloc": lambda r: r["loc"]["iloc"]["after"],
          "fro_rel_to_log": lambda r: r["fro_rel_to_log"],
          "rank_correction": lambda r: r["rank"].get("correction"),
          "sec": lambda r: r["cost"]["sec"], "n_forward": lambda r: r["cost"]["n_forward"],
          "n_backward": lambda r: r["cost"]["n_backward"]}


def _med(v):
    return None if not v else float(np.median(v))


def _auc_lower(a, b):
    """P(x < y) + 0.5 P(x = y) for x drawn from a, y from b (Mann-Whitney); 0.5 = the two groups are not
    separated; > 0.5 = a tends to be lower than b."""
    if not a or not b:
        return None
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    bs = np.sort(b)
    lo = np.searchsorted(bs, a, side="left")           # number of b strictly below each a
    hi = np.searchsorted(bs, a, side="right")          # number of b <= each a
    return float(((len(b) - hi) + 0.5 * (hi - lo)).sum() / (len(a) * len(b)))


def mr_summary(log, config, mr, tw):
    """Summary of the per-row module-restoration ratios of one config (held-out rows; see run_d2.module_restore)."""
    rho, d1 = mr["rho"], mr["stored_D1"]
    rp0 = os.path.join(OUT, log, config + ".json")
    rec = jload(rp0) if os.path.exists(rp0) else None
    rep = set((rec or {}).get("reported_groups") or [])
    is_tw = [g in tw for g in mr["groups"]]
    tw_rep = [g in tw and bool(rep & set(tw[g]["repair_groups"])) for g in mr["groups"]]
    cd = [x for x, c, t in zip(rho, d1, is_tw) if c and not t]
    nd = [x for x, c, t in zip(rho, d1, is_tw) if (not c) and not t]
    m = {"n_rows": len(rho), "median_complaints": _med([x for x, c in zip(rho, d1) if c]),
         "median_noncomplaints": _med([x for x, c in zip(rho, d1) if not c]),
         "frac_complaints_below_half": float(np.mean([x < 0.5 for x, c in zip(rho, d1) if c])),
         "q25_complaints": float(np.quantile([x for x, c in zip(rho, d1) if c], 0.25)),
         "q75_complaints": float(np.quantile([x for x, c in zip(rho, d1) if c], 0.75)),
         "by_ptype": {pt: _med([x for x, c, t in zip(rho, d1, mr["ptype"]) if c and t == pt]) for pt in ("image", "text")},
         "auc_complaints_lower_disjoint": _auc_lower(cd, nd),
         "frac_above_one_disjoint": float(np.mean([x > 1.0 for x, t in zip(rho, is_tw) if not t])),
         "median_complaints_disjoint": _med(cd), "median_noncomplaints_disjoint": _med(nd),
         "frac_complaints_below_half_disjoint": float(np.mean([x < 0.5 for x in cd])) if cd else None,
         "frac_noncomplaints_below_half_disjoint": float(np.mean([x < 0.5 for x in nd])) if nd else None,
         "n_twins_of_reported": int(sum(tw_rep)),
         "median_twins_of_reported": _med([x for x, t in zip(rho, tw_rep) if t]),
         "max_twins_of_reported": max([x for x, t in zip(rho, tw_rep) if t], default=None)}
    sh = mr.get("span_share")
    if sh is not None:
        m["span"] = {"rank": mr.get("rank"),
                     "share_complaints_disjoint": _med([x for x, c, t in zip(sh, d1, is_tw) if c and not t]),
                     "share_noncomplaints_disjoint": _med([x for x, c, t in zip(sh, d1, is_tw) if (not c) and not t]),
                     "share_twins_of_reported_min": min([x for x, t in zip(sh, tw_rep) if t], default=None),
                     "share_q90_disjoint": float(np.quantile([x for x, t in zip(sh, is_tw) if not t], 0.9))}
    if rec is not None:                                  # outcome of the same rows in the stream record
        hr = rec["eval"]["heldout_rows"]
        if [r["group"] for r in hr] == mr["groups"]:
            aft = [r["reproduced_after"] for r in hr]
            m["median_complaints_fixed"] = _med([x for x, c, a in zip(rho, d1, aft) if c and a])
            m["median_complaints_not_fixed"] = _med([x for x, c, a in zip(rho, d1, aft) if c and not a])
            m["median_noncomplaints_lost"] = _med([x for x, c, a in zip(rho, d1, aft) if (not c) and not a])
            m["median_noncomplaints_kept"] = _med([x for x, c, a in zip(rho, d1, aft) if (not c) and a])
    return m


def load_d2(log, leak):
    """(d2, rho): D2 summaries per config (out/<log>/d2/*.json) and module-restoration summaries per config (from
    the D2 record when it has them, else from the CPU computation of rho_cpu.py in out/<log>/rho/*.json)."""
    out, rho = {}, {}
    tw = leak["twins"]
    for fn in sorted(glob.glob(os.path.join(OUT, log, "d2", "*.json"))):
        d = jload(fn)
        s = dict(d["summary"], seconds=d["seconds"])
        if d.get("regression_suite_in_sample"):
            s["regression_suite_in_sample"] = d["regression_suite_in_sample"]
        dj = [x for x in d["rows"] if x["group"] not in tw]
        s["disjoint"] = {"n_pre_correct": len(dj), "n_D2": int(sum(not x["post_correct"] for x in dj)),
                         "n_D2_WE": int(sum(x["stored_D2"] for x in dj)),
                         "stored_D2_resolved": int(sum(x["stored_D2"] and x["post_correct"] for x in dj)),
                         "new_D2": int(sum((not x["stored_D2"]) and not x["post_correct"] for x in dj))}
        if d.get("module_restore") is not None:
            rho[d["config"]] = dict(mr_summary(log, d["config"], d["module_restore"], tw), source="run_d2.py")
        out[d["config"]] = s
    for fn in sorted(glob.glob(os.path.join(OUT, log, "rho", "*.json"))):   # the CPU file has the span share too: preferred
        d = jload(fn)
        new = dict(mr_summary(log, d["config"], d["module_restore"], tw), source="rho_cpu.py")
        old = rho.get(d["config"])
        if old is not None:                                 # the same quantity from the GPU pass: must agree
            new["max_abs_diff_vs_run_d2_median"] = abs(old["median_complaints"] - new["median_complaints"])
        rho[d["config"]] = new
    return out, rho


def load_superseded(log):
    """First-pass REEDIT records (z* without the harness oracle fallback) next to their re-runs."""
    out = []
    for fn in sorted(glob.glob(os.path.join(OUT, log, "superseded_zfallback", "REEDIT_*.json"))):
        d = jload(fn)
        s = summarize(d)
        out.append({"config": d["config"], "first_pass": {"reported_resolved": s["reported"]["resolved"], "n": s["reported"]["n"],
                                                          "not_installed": s["reedit"]["not_installed"],
                                                          "heldout_D1_rate": s["heldout"]["D1_rate"],
                                                          "rel_exact": s["edits"]["rel"]["exact"],
                                                          "edits_broken_total": s["edits_broken_total"],
                                                          "fro_rel_to_log": s["fro_rel_to_log"]}})
    return out


def main():
    res = {"logs": {}, "protocol": {"rel_tol": REL_TOL, "bootstrap": NB, "methods": list(METHODS),
                                    "n_grid": [str(n) for n in N_GRID], "n_base": C.N_BASE, "seeds": list(C.SEEDS),
                                    "rb_tot_budget": C.RB_TOT_BUDGET, "rb_union_budget": C.RB_UNION_BUDGET,
                                    "reedit_batch": C.REEDIT_BATCH, "d2_ns": [str(n) for n in C.D2_NS], "options": {}}}
    for log in LOGS:
        od = os.path.join(OUT, log)
        if not os.path.exists(os.path.join(od, "_setup.json")):
            continue
        setup = jload(os.path.join(od, "_setup.json"))
        rows, failed, oom, raw_info, flips = {}, [], [], [], {}
        leak = leak_map(log)
        for fn in sorted(glob.glob(os.path.join(od, "*.json"))):
            b = os.path.basename(fn)
            if b.startswith("_"):
                continue
            d = jload(fn)
            if b.endswith(".failed.json"):
                failed.append({k: d.get(k) for k in ("config", "method", "n", "seed", "error")})
                continue
            if b.endswith(".oom.json"):
                oom.append({k: d.get(k) for k in ("config", "method", "n", "seed", "error", "cap_gib")})
                continue
            rows[d["config"]] = summarize(d, leak)
            hrr = d["eval"]["heldout_rows"]
            flips[d["config"]] = ({r["group"] for r in hrr if r["reproduced_after"] and not r["reproduced_WE"]},
                                  {r["group"] for r in hrr if r["reproduced_WE"] and not r["reproduced_after"]})
            if d.get("options") and d["method"] not in res["protocol"]["options"]:
                res["protocol"]["options"][d["method"]] = d["options"]
            raw_info.append((d.get("alias_of") is not None, d.get("peak_gpu_bytes") or 0, d.get("gpu_procs"),
                             d.get("repair_wall_sec") or 0.0, d.get("eval_sec") or 0.0, d.get("torch_threads"), d.get("time")))
        oom = [o for o in oom if o["config"] not in rows]
        L = {"setup": {k: v for k, v in setup.items() if k != "streams"}, "rows": rows, "failed": failed,
             "oom_unfinished": oom}
        n_all = setup["n_repair_complaints"]
        L["n_all"] = n_all
        fset = {x["config"] for x in failed}
        L["missing"] = [C.config_name(m, n, sd) for (m, n, sd) in C.plan(extra=False)
                        if C.config_name(m, n, sd) not in rows and C.config_name(m, n, sd) not in fset]
        L["missing_extra"] = [C.config_name(m, n, sd) for (m, n, sd) in C.plan(extra=True)
                              if m in C.EXTRA_METHODS and C.config_name(m, n, sd) not in rows]
        own = [x for x in raw_info if not x[0]]
        gp = [x[2] for x in own if x[2] is not None]
        L["run_info"] = {"n_alias": sum(x[0] for x in raw_info), "peak_gpu_gib_max": max((x[1] for x in own), default=0) / 2 ** 30,
                         "gpu_procs_median": float(np.median(gp)) - 1 if gp else None, "gpu_procs_max": float(max(gp)) - 1 if gp else None,
                         "repair_wall_min": sum(x[3] for x in own) / 60.0, "eval_wall_min": sum(x[4] for x in own) / 60.0}
        # records of the resume (2026-11-10 17:20 UTC on): the only ones that carry torch_threads
        rs_ = [x for x in own if x[5] is not None]
        if rs_:
            gp2 = [x[2] for x in rs_ if x[2] is not None]
            L["run_info"]["resume"] = {"n_records": len(rs_), "torch_threads": sorted({int(x[5]) for x in rs_}),
                                       "first": min(x[6] for x in rs_), "last": max(x[6] for x in rs_),
                                       "gpu_procs_max": float(max(gp2)) - 1 if gp2 else None,
                                       "peak_gpu_gib_max": max(x[1] for x in rs_) / 2 ** 30,
                                       "repair_wall_min": sum(x[3] for x in rs_) / 60.0, "eval_wall_min": sum(x[4] for x in rs_) / 60.0}
        # curves: per method, per n -> seed statistics
        curves = {}
        for m in METHODS:
            for n in N_GRID:
                rs = [r for r in rows.values() if r["method"] == m and r["n"] == n]
                if not rs:
                    continue
                rs = sorted(rs, key=lambda r: r["seed"])
                c = {"n": n, "n_value": n_all if n == "ALL" else n, "seeds": [r["seed"] for r in rs],
                     "configs": [r["config"] for r in rs]}
                for k, f in CURVES.items():
                    c[k] = seed_stats(rs, f)
                c["rel_within_tol_all_seeds"] = all(r["rel_within_tol"] for r in rs)
                curves.setdefault(m, {})[str(n)] = c
        L["curves"] = curves
        # identity checks: sequential COMP vs the joint solve
        ident = []
        for m, mb in (("COMP_S", "COMP_B"), ("COMP_SP", "COMP_BP")):
            for r in rows.values():
                if r["method"] != m:
                    continue
                o = rows.get(r["config"].replace(m + "__", mb + "__"))
                iv = (r.get("comp") or {}).get("identity_vs_batch")
                ident.append({"config": r["config"], "alias": r["alias_of"] is not None,
                              "rel_fro_diff": None if iv is None else iv["rel_fro_diff"],
                              "d_reported_resolved": None if o is None else r["reported"]["resolved"] - o["reported"]["resolved"],
                              "d_heldout_D1": None if o is None else r["heldout"]["n_D1_after"] - o["heldout"]["n_D1_after"],
                              "d_edit_checks": None if o is None else sum(abs(r["edits"][k]["exact"] - o["edits"][k]["exact"]) for k in ("rel", "tgen", "igen")),
                              "same_heldout_D1": None if o is None else r["heldout"]["n_D1_after"] == o["heldout"]["n_D1_after"],
                              "same_reported_resolved": None if o is None else r["reported"]["resolved"] == o["reported"]["resolved"],
                              "same_edits": None if o is None else all(r["edits"][k]["exact"] == o["edits"][k]["exact"] for k in ("rel", "tgen", "igen"))})
        L["seq_vs_batch_identity"] = ident
        # operating points: Rel exact within REL_TOL of the unrepaired log
        none = rows.get("NONE")
        ops = []
        if none is not None:
            for r in rows.values():
                if r["method"] in ("NONE", "ALL") or not r["rel_within_tol"]:
                    continue
                hd = r["heldout_disjoint"]
                ops.append({"config": r["config"], "method": r["method"], "n": r["n"], "seed": r["seed"],
                            "heldout_D1_rate": r["heldout"]["D1_rate"], "ci": [r["heldout"]["D1_rate_ci"]["lo"], r["heldout"]["D1_rate_ci"]["hi"]],
                            "delta_vs_none": r["heldout"]["D1_rate"] - none["heldout"]["D1_rate"],
                            "mcnemar_p": r["heldout"]["paired_vs_WE"]["p_exact"],
                            "disjoint_D1_rate": hd["D1_rate"], "disjoint_ci": [hd["D1_rate_ci"]["lo"], hd["D1_rate_ci"]["hi"]],
                            "disjoint_delta_vs_none": hd["D1_rate"] - none["heldout_disjoint"]["D1_rate"],
                            "disjoint_mcnemar_p": hd["paired_vs_WE"]["p_exact"],
                            "rel_exact": r["edits"]["rel"]["exact"]})
            L["operating_points_extra"] = sorted([o for o in ops if o["method"] in EXTRA], key=lambda o: o["heldout_D1_rate"])
            ops = [o for o in ops if o["method"] not in EXTRA]
            L["operating_points_disjoint"] = sorted(ops, key=lambda o: o["disjoint_D1_rate"])[:15]
            ops.sort(key=lambda o: o["heldout_D1_rate"])
        L["operating_points"] = ops[:15]
        L["n_operating_candidates"] = len(ops)
        L["leak"] = {k: v for k, v in leak.items() if k not in ("twins", "heldout_valid_groups")}
        own_rows = [r for r in rows.values() if r["alias_of"] is None and "twin_consistency" in r]
        L["leak"]["twin_outcome_check"] = {"n": int(sum(r["twin_consistency"]["n"] for r in own_rows)),
                                           "agree": int(sum(r["twin_consistency"]["agree"] for r in own_rows)),
                                           "n_configs": int(sum(r["twin_consistency"]["n"] > 0 for r in own_rows))}
        # the same selection on seed means (cells with every seed inside the tolerance)
        cells = []
        for m, cn in curves.items():
            for n, c in cn.items():
                if c["rel_within_tol_all_seeds"] and m not in EXTRA:
                    cells.append({"method": m, "n": n, "n_seeds": c["heldout_D1_rate"]["n_seeds"],
                                  "heldout_D1_rate_mean": c["heldout_D1_rate"]["mean"],
                                  "disjoint_D1_rate_mean": c["heldout_disjoint_D1_rate"]["mean"],
                                  "rel_exact_mean": c["rel_exact"]["mean"]})
        cells.sort(key=lambda o: o["heldout_D1_rate_mean"])
        L["operating_cells"] = cells[:15]
        L["d2"], L["rho"] = load_d2(log, leak)
        pp = os.path.join(od, "_protect_split.json")
        if os.path.exists(pp):                              # CPU diagnostic of protect_split_cpu.py
            ps = jload(pp)
            tw = leak["twins"]
            dj = [g not in tw for g in ps["groups"]]
            d1 = ps["stored_D1"]

            def med(v, comp):
                return _med([x for x, c, k in zip(v, d1, dj) if k and c == comp])
            L["protect_split"] = {
                "rank_K_ans": ps["rank_K_ans"], "rank_matches_COMP_B": ps["rank_matches_COMP_B"], "d_in": ps["d_in"],
                "D_share_on_K_ans_span": ps["fro_D_on_K_ans_span"] / ps["fro_D"],
                "a_complaints": med(ps["ans"]["a"], True), "a_noncomplaints": med(ps["ans"]["a"], False),
                "b_complaints": med(ps["ans"]["b"], True), "b_noncomplaints": med(ps["ans"]["b"], False),
                "s_complaints": med(ps["ans"]["s"], True), "s_noncomplaints": med(ps["ans"]["s"], False),
                "rel_change_complaints": med(ps["rel_change"], True), "rel_change_noncomplaints": med(ps["rel_change"], False),
                "frac_complaints_a_below_half": float(np.mean([x < 0.5 for x, c, k in zip(ps["ans"]["a"], d1, dj) if k and c])),
                "n_complaints": int(sum(1 for c, k in zip(d1, dj) if k and c)), "n_noncomplaints": int(sum(1 for c, k in zip(d1, dj) if k and not c))}
        # (a) flip rates on the disjoint held-out rows: P(fixed | held-out complaint), P(lost | held-out non-complaint)
        # (b) persistence along the NESTED reported sets of one seed: of the held-out rows a repair of n1 complaints
        #     flipped, how many are still flipped after the repair of the superset of n2 > n1 complaints
        pers = {}
        for m in METHODS:
            for sd in C.SEEDS:
                seq = [(n, C.config_name(m, n, sd)) for n in N_GRID if C.config_name(m, n, sd) in flips]
                ent = []
                for (n1, c1), (n2, c2) in zip(seq[:-1], seq[1:]):
                    f1, l1 = flips[c1]
                    f2, l2 = flips[c2]
                    ent.append({"n1": str(n1), "n2": str(n2), "fixed_n1": len(f1), "fixed_kept": len(f1 & f2), "fixed_n2": len(f2),
                                "lost_n1": len(l1), "lost_kept": len(l1 & l2), "lost_n2": len(l2)})
                if ent:
                    pers.setdefault(m, {})[str(sd)] = ent
        L["flip_persistence"] = pers
        for r in rows.values():
            hd = r["heldout_disjoint"]
            r["flip_rates_disjoint"] = {"fix": 100.0 * hd["complaints_fixed"] / max(hd["n_complaints"], 1),
                                        "loss": 100.0 * hd["noncomplaints_lost"] / max(hd["n_noncomplaints"], 1)}
        rid = []
        ip = os.path.join(od, "_reedit_identity.json")
        wid = jload(ip) if os.path.exists(ip) else {}
        for n in N_GRID:
            for sd in C.SEEDS:
                a, b = rows.get(C.config_name("REEDIT_S", n, sd)), rows.get(C.config_name("REEDIT_B", n, sd))
                if a is None or b is None:
                    continue
                w = wid.get(str(n)) if sd == 0 else None
                rid.append({"n": str(n), "seed": sd, "alias": a["alias_of"] is not None,
                            "n_batches": a["reedit"]["n_batches"],
                            "rel_fro_diff_weights": None if not w or w.get("alias") else w["rel_fro_diff"],
                            "weight_dtype": None if not w or w.get("alias") else w.get("dtype"),
                            "ulp": None if not w or w.get("alias") else w.get("ulp"),
                            "d_reported_resolved": a["reported"]["resolved"] - b["reported"]["resolved"],
                            "d_heldout_D1": a["heldout"]["n_D1_after"] - b["heldout"]["n_D1_after"],
                            "d_edit_checks": sum(abs(a["edits"][k]["exact"] - b["edits"][k]["exact"]) for k in ("rel", "tgen", "igen")),
                            "norm_S": a["fro_rel_to_log"], "norm_B": b["fro_rel_to_log"]})
        L["reedit_seq_vs_batch"] = rid
        L["extra_rows"] = sorted([r["config"] for r in rows.values() if r["method"] in EXTRA])
        # D1 flag at W_0 (record ALL: every valid REPAIR row evaluated at W_0) of the complaints listed in unresolved_detail
        want = {u["group"] for r in rows.values() for u in ((r.get("comp") or {}).get("unresolved_detail") or [])}
        ap_ = os.path.join(od, "ALL.json")
        if want and os.path.exists(ap_):
            L["w0_complaint_flags"] = {f"{c['ptype']}_{c['probe_idx']}": [bool(c["D1_resolved"])]
                                       for c in jload(ap_)["eval"]["complaints"] if f"{c['ptype']}_{c['probe_idx']}" in want}
        L["superseded_zfallback"] = load_superseded(log)
        # first-pass REEDIT records (the harness's DEFAULT call: one z* attempt, non-oracle complaints not installed)
        # as full rows: shown, clearly marked, only where the re-run with the documented fallback is missing
        fp = {}
        for fn in sorted(glob.glob(os.path.join(od, "superseded_zfallback", "REEDIT_*.json"))):
            d = jload(fn)
            if d["config"] not in rows:
                fp[d["config"]] = summarize(d, leak)
        L["rows_first_pass"] = fp
        for name, r in fp.items():                     # curve cells that exist only as a first-pass record (flagged)
            m, n = r["method"], str(r["n"])
            if m in curves and n in curves[m]:
                continue
            c = {"n": r["n"], "n_value": n_all if r["n"] == "ALL" else r["n"], "seeds": [r["seed"]], "configs": [name], "first_pass": True}
            for k, f in CURVES.items():
                c[k] = seed_stats([r], f)
            c["rel_within_tol_all_seeds"] = False      # never part of the operating-point selection
            curves.setdefault(m, {})[n] = c
        res["logs"][log] = L
    ip = os.path.join(OUT, "_incidents.json")
    if os.path.exists(ip):
        res["incidents"] = jload(ip)
    rp = os.path.join(OUT, "_resume.json")                 # operational log of the resume run (like _incidents.json)
    if os.path.exists(rp):
        res["resume"] = jload(rp)
    tp = os.path.join(OUT, "text_h2h", "results_text.json")
    if os.path.exists(tp):
        res["text_h2h"] = jload(tp)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("results.json:", {k: (len(v["rows"]), "rows", len(v["d2"]), "d2", len(v["failed"]), "failed") for k, v in res["logs"].items()})


if __name__ == "__main__":
    main()
