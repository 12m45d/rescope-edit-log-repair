"""P12, second priority -- out/text_h2h/rows_<log>.jsonl -> out/text_h2h/results_text.json.

An adapted copy of 07_repair_v3/rq1/code/aggregate.py (same definitions: uid-level percentile bootstrap, exact
McNemar, `clean` = resolved AND no logged edit broken), restricted to the five RQ1 methods and to this
experiment's TEXT complaints.  For reference the same summary is computed, read-only, over rq1's own rows
(image complaints; only complaints for which all five methods are present).

usage: python aggregate_text.py
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P12 = os.path.dirname(HERE)
V3 = f"{_RP.ROOT}/02_research/experiments/07_repair_v3"
sys.path.insert(0, V3)
from harness.stats import bootstrap_ci, mcnemar  # noqa: E402

LOGS = ["TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"]
RQ1_CFG = ["NONE", "COMP", "REEDIT", "RB_FO", "RB_BEH"]
TDIR = os.path.join(P12, "out", "text_h2h")
N_TEXT = 30                       # planned text complaints per log (run_text_h2h.py --n default)


def load_rows(path):
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path)]
    seen, out = set(), []
    for r in rows:                      # resume safety: first row per (config, complaint)
        k = (r["config"], r["group"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def resolved(r):
    return bool(r["eval"]["complaints"][0]["D1_resolved"])


def broken(r):
    return r["eval"].get("edits_broken_total")


def clean(r):
    return resolved(r) and broken(r) == 0


def ci(vals, uids):
    pairs = [(v, u) for v, u in zip(vals, uids) if v is not None]
    if not pairs:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    return bootstrap_ci([p[0] for p in pairs], [p[1] for p in pairs])


def summarise(rows):
    u = [r["uid"] for r in rows]
    full = [r for r in rows if r.get("full_eval", True)]
    uf = [r["uid"] for r in full]
    ev = [r["eval"] for r in full]
    s = {"n": len(rows), "n_full_eval": len(full),
         "D1_resolved": int(sum(resolved(r) for r in rows)),
         "D1_rate": ci([float(resolved(r)) for r in rows], u),
         "D2_pooled": {"n": sum(1 for r in rows if r["is_D2"]),
                       "resolved": int(sum(bool(r["eval"]["complaints"][0].get("D2_resolved")) for r in rows if r["is_D2"]))},
         "tok_acc": ci([r["eval"]["complaints"][0]["tok_acc"] for r in rows], u)}
    if full:
        s.update({
            "clean": int(sum(clean(r) for r in full)),
            "clean_rate": ci([float(clean(r)) for r in full], uf),
            "zero_broken_rate": ci([float(broken(r) == 0) for r in full], uf),
            "edits_broken_total": ci([e["edits_broken_total"] for e in ev], uf),
            "edits_fixed_total": ci([e["edits_fixed_total"] for e in ev], uf),
            "edits": {m: {"broken": ci([e["edits"][m]["n_broken"] for e in ev], uf),
                          "fixed": ci([e["edits"][m]["n_fixed"] for e in ev], uf),
                          "exact_after": ci([e["edits"][m]["exact"] for e in ev], uf),
                          "exact_WE": ev[0]["edits"][m]["exact_WE"]}
                      for m in ("rel", "tgen", "igen")},
            "collateral": {pt: {"noncomplaints_lost": ci([e["collateral"][pt]["noncomplaints_lost"] for e in ev], uf),
                                "complaints_fixed": ci([e["collateral"][pt]["complaints_fixed"] for e in ev], uf),
                                "n_noncomplaints": ev[0]["collateral"][pt]["n_noncomplaints"],
                                "n_complaints": ev[0]["collateral"][pt]["n_complaints"]}
                           for pt in ("image", "text", "all")},
            "loc": {m: {"delta": ci([e["loc"][m]["after"] - e["loc"][m]["WE"] for e in ev], uf),
                        "WE": ev[0]["loc"][m]["WE"]} for m in ("tloc", "iloc")},
            "fro_median": float(np.median([e["fro"] for e in ev])),
            "fro_rel_median": float(np.median([e["fro_rel_to_log"] for e in ev]))})
    kinds = sorted({k for r in rows for k in r["cost"]["passes"]})
    s["cost"] = {"passes_mean": {k: float(np.mean([r["cost"]["passes"].get(k, 0) for r in rows])) for k in kinds},
                 "n_forward": ci([float(r["cost"]["n_forward"]) for r in rows], u),
                 "n_backward": ci([float(r["cost"]["n_backward"]) for r in rows], u),
                 "flops_median": float(np.median([r["cost"]["flops_total"] for r in rows])),
                 "sec_median": float(np.median([r["cost"]["sec"]["total"] for r in rows])),
                 "sec_p90": float(np.percentile([r["cost"]["sec"]["total"] for r in rows], 90)),
                 "gpu_procs_median": float(np.median([r.get("gpu_procs") or 0 for r in rows]))}
    m = rows[0]["method"]
    if m.startswith("ROLLBACK"):
        k = [r["meta"]["k"] for r in rows]
        s["rollback"] = {"k": ci(k, u), "k_median": float(np.median(k)), "k_max": int(max(k)),
                         "budget_exhausted_unresolved": int(sum((not r["meta"]["resolved_internal"]) for r in rows))}
    if m == "REEDIT":
        s["reedit"] = {"not_installed": int(sum(len(r["meta"]["not_installed_rows"]) for r in rows)),
                       "not_installed_groups": [r["group"] for r in rows if r["meta"]["not_installed_rows"]],
                       "zopt_attempts_hist": {str(a): int(sum(r["meta"]["zopt_attempts"][0] == a for r in rows))
                                              for a in sorted({r["meta"]["zopt_attempts"][0] for r in rows})},
                       "solve_attainment": ci([r["meta"].get("solve_attainment") for r in rows], u)}
    if m == "COMP":
        s["comp"] = {"infeasible": int(sum(bool(r["meta"].get("infeasible")) for r in rows)),
                     "protect_rank": rows[0]["meta"].get("protect_rank"),
                     "rank_dropped_total": int(sum(r["meta"].get("n_dropped", 0) for r in rows)),
                     "P1_rel_max": float(max(r["meta"]["P1_rel"] for r in rows))}
    return s


def paired(byc, a, b, flag):
    A = {r["group"]: r for r in byc[a]}
    Bm = {r["group"]: r for r in byc[b]}
    gs = [g for g in A if g in Bm and (flag is resolved or (A[g].get("full_eval", True) and Bm[g].get("full_eval", True)))]
    return dict(mcnemar([flag(A[g]) for g in gs], [flag(Bm[g]) for g in gs]), a=a, b=b)


def one_set(rows):
    byc = {}
    for r in rows:
        byc.setdefault(r["config"], []).append(r)
    done = {g for g in {r["group"] for r in rows} if all(any(x["group"] == g for x in byc.get(c, [])) for c in RQ1_CFG)}
    for c in list(byc):
        byc[c] = sorted([r for r in byc[c] if r["group"] in done], key=lambda r: r["complaint_index"])
    if not done:
        return None
    ti = [r.get("text_index") for r in byc["COMP"]]
    L = {"n_complaints_complete": len(done), "groups": [r["group"] for r in byc["COMP"]],
         # this experiment's rows carry text_index (position in draw order): are the finished complaints the first k?
         "is_draw_order_prefix": (None if any(t is None for t in ti) else sorted(ti) == list(range(len(ti)))),
         "n_ans_tok_median": float(np.median([r["eval"]["complaints"][0]["n_tok"] for r in byc["COMP"]])),
         "configs": {c: summarise(byc[c]) for c in RQ1_CFG if byc.get(c)}, "paired": {}}
    ex = byc["COMP"][0]["eval"]
    L["baseline"] = {"rel_exact_WE": ex["edits"]["rel"]["exact_WE"], "tgen_exact_WE": ex["edits"]["tgen"]["exact_WE"],
                     "igen_exact_WE": ex["edits"]["igen"]["exact_WE"], "tloc_WE": ex["loc"]["tloc"]["WE"],
                     "iloc_WE": ex["loc"]["iloc"]["WE"], "heldout_noncomplaints": ex["collateral"]["all"]["n_noncomplaints"],
                     "heldout_complaints": ex["collateral"]["all"]["n_complaints"]}
    for b in ["NONE", "REEDIT", "RB_FO", "RB_BEH"]:
        L["paired"][f"COMP_vs_{b}_resolved"] = paired(byc, "COMP", b, resolved)
        if b != "NONE":
            L["paired"][f"COMP_vs_{b}_clean"] = paired(byc, "COMP", b, clean)
    L["paired"]["REEDIT_vs_RB_BEH_resolved"] = paired(byc, "REEDIT", "RB_BEH", resolved)
    L["paired"]["REEDIT_vs_RB_BEH_clean"] = paired(byc, "REEDIT", "RB_BEH", clean)
    fb = byc.get("REEDIT_FB", [])
    if fb:      # supplement rows: the documented z* fallback for complaints rq1's call left uninstalled
        L["reedit_fallback_supplement"] = [{"group": r["group"], "resolved": resolved(r), "edits_broken": broken(r),
                                            "zopt_attempts": r["meta"]["zopt_attempts"],
                                            "installed": not r["meta"]["not_installed_rows"]} for r in fb]
        ok = {r["group"] for r in fb if resolved(r)}
        n = len(byc["REEDIT"])
        L["reedit_with_fallback_resolved"] = {"resolved": int(sum(resolved(r) or r["group"] in ok for r in byc["REEDIT"])), "n": n}
    return L


def main():
    out = {"protocol": {
        "complaints": "text rows of rq1's own sample harness.sample_complaints(H, 100, seed=1) (REPAIR pool, D1 v2): the first 30 "
                      "text complaints in draw order (rq1 indices 50..79), never run by rq1",
        "methods": "NONE; COMP protect K_ans, query answer, adapter; REEDIT k0+kp; ROLLBACK_FO budget 10; ROLLBACK_BEH budget 20 "
                   "(the options of rq1/code/run_rq1.py)",
        "collateral": "HELD-OUT pool only", "ci": "uid-level percentile bootstrap, 2000 resamples, 95%",
        "paired": "exact McNemar on the same complaints",
        "clean": "resolved (D1) AND no logged edit broken (rel + T-Gen + I-Gen, teacher-forced exact)",
        "reference": "rq1 image complaints: read-only from 07_repair_v3/rq1/rows (complaints with all five methods)",
        "n_planned": N_TEXT},
        "text": {}, "rq1_image_reference": {}}
    for lg in LOGS:
        t = one_set(load_rows(os.path.join(TDIR, f"rows_{lg}.jsonl")))
        if t is not None:
            out["text"][lg] = t
        ref = one_set([r for r in load_rows(os.path.join(V3, "rq1", "rows", f"rows_{lg}.jsonl")) if r["config"] in RQ1_CFG])
        if ref is not None:
            out["rq1_image_reference"][lg] = ref
    if not out["text"]:
        print("no text head-to-head rows yet")
        return
    with open(os.path.join(TDIR, "results_text.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("results_text.json:", {k: v["n_complaints_complete"] for k, v in out["text"].items()},
          "| rq1 image reference:", {k: v["n_complaints_complete"] for k, v in out["rq1_image_reference"].items()})


if __name__ == "__main__":
    main()
