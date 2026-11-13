"""P12 verifier -- extra CPU readings of the experiment's own records (interpretation checks). -> out/cpu_extra.json
  1. the "why" paragraph: do the module-level quantities separate held-out complaints from non-complaints in the
     direction the explanation needs?  (from out/<log>/_protect_split.json: per-row s, a, b, rel_change)
  2. rollback budgets: how often do they bind?
  3. Rel tolerance: which configurations sit exactly on the 2-point boundary; T-Loc / I-Loc of the best point.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402


def auc(a, b):
    """P(x > y) + 0.5 P(x = y), x from a, y from b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = np.sort(b)
    lo, hi = np.searchsorted(bs, a, "left"), np.searchsorted(bs, a, "right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(a) * len(b)))


out = {}
for log in VC.LOGS:
    o = {}
    rows = VC.records(log)
    pool = VC.my_split(rows)
    key = lambda r: (r["ptype"], r["prompt"], r["image"])  # noqa: E731
    repk = {key(r) for r in rows if pool[r["probe_idx"]] == "repair"}
    ho = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    pp = os.path.join(VC.P12, "out", log, "_protect_split.json")
    if os.path.exists(pp):
        ps = VC.jload(pp)
        assert ps["groups"] == [r["group"] for r in ho]
        dj = np.array([key(r) not in repk for r in ho])
        d1 = np.array(ps["stored_D1"])
        s, a, b = (np.array(ps["ans"][k]) for k in ("s", "a", "b"))
        rc = np.array(ps["rel_change"])
        c, n = dj & d1, dj & ~d1
        absP = a * rc                                   # ||D P K|| / ||W_0 K||: change through the protected span, absolute scale
        absC = b * rc                                   # ||D (I-P) K|| / ||W_0 K||: change outside it
        o["why"] = {"n_complaints": int(c.sum()), "n_noncomplaints": int(n.sum()),
                    "median_s": [float(np.median(s[c])), float(np.median(s[n]))],
                    "median_a": [float(np.median(a[c])), float(np.median(a[n]))],
                    "median_b": [float(np.median(b[c])), float(np.median(b[n]))],
                    "median_rel_change": [float(np.median(rc[c])), float(np.median(rc[n]))],
                    "median_change_through_span_over_W0K": [float(np.median(absP[c])), float(np.median(absP[n]))],
                    "median_change_outside_span_over_W0K": [float(np.median(absC[c])), float(np.median(absC[n]))],
                    "auc_complaint_higher": {"s": auc(s[c], s[n]), "a": auc(a[c], a[n]), "rel_change": auc(rc[c], rc[n]),
                                             "change_through_span_over_W0K": auc(absP[c], absP[n]),
                                             "change_outside_span_over_W0K": auc(absC[c], absC[n])},
                    "by_ptype": {pt: {"n": [int((c & (np.array(ps["ptype"]) == pt)).sum()), int((n & (np.array(ps["ptype"]) == pt)).sum())],
                                      "median_s": [float(np.median(s[c & (np.array(ps["ptype"]) == pt)])), float(np.median(s[n & (np.array(ps["ptype"]) == pt)]))],
                                      "median_a": [float(np.median(a[c & (np.array(ps["ptype"]) == pt)])), float(np.median(a[n & (np.array(ps["ptype"]) == pt)]))],
                                      "auc_complaint_higher_through_span": auc(absP[c & (np.array(ps["ptype"]) == pt)], absP[n & (np.array(ps["ptype"]) == pt)]),
                                      "auc_complaint_higher_outside_span": auc(absC[c & (np.array(ps["ptype"]) == pt)], absC[n & (np.array(ps["ptype"]) == pt)])}
                                 for pt in ("image", "text")}}
    rb = {}
    for n_ in ("1", "2", "5", "10", "20", "100", "ALL"):
        for m in ("RB_TOT", "RB_UNION"):
            p = os.path.join(VC.P12, "out", log, f"{m}__n{n_}__s0.json")
            if not os.path.exists(p):
                continue
            d = VC.jload(p)
            ex = d["extra"]
            if m == "RB_TOT":
                rb[f"{m}__n{n_}"] = {"k": ex["k"], "budget": ex["budget"], "resolved_internal": ex["resolved_internal"],
                                     "budget_binds": bool(ex["k"] >= ex["budget"] and not ex["resolved_internal"])}
            else:
                ks = ex["per_complaint_k"]
                rb[f"{m}__n{n_}"] = {"k_union": ex["k"], "n": len(ks), "n_resolved_individually": ex["n_resolved_individually"],
                                     "n_at_budget": int(sum(k >= ex["budget_each"] for k in ks)), "median_k": float(np.median(ks))}
    o["rollback"] = rb
    none = VC.jload(os.path.join(VC.P12, "out", log, "NONE.json"))["eval"]
    best = VC.jload(os.path.join(VC.P12, "out", log, "COMP_B__nALL__s0.json"))["eval"]
    o["best_point"] = {"rel": [none["edits"]["rel"]["exact"], best["edits"]["rel"]["exact"]],
                       "rel_token": [none["edits"]["rel"]["token"], best["edits"]["rel"]["token"]],
                       "rel_broken_fixed": [best["edits"]["rel"]["n_broken"], best["edits"]["rel"]["n_fixed"]],
                       "tloc": [none["loc"]["tloc"]["after"], best["loc"]["tloc"]["after"]],
                       "iloc": [none["loc"]["iloc"]["after"], best["loc"]["iloc"]["after"]],
                       "on_rel_tolerance_boundary": bool(abs(best["edits"]["rel"]["exact"] - (none["edits"]["rel"]["exact"] - 2.0)) < 1e-9)}
    st = VC.jload(os.path.join(VC.P12, "out", log, "_setup.json"))["stored"]
    o["stored_vs_B1"] = {"stored_b8": [st["Rel_exact"], st["TGen_exact"], st["IGen_exact"]],
                         "B1_used_in_tables": [none["edits"][m]["exact"] for m in ("rel", "tgen", "igen")]}
    out[log] = o
    print(log, {k: v for k, v in o.items()})
VC.jdump(out, os.path.join(VC.VOUT, "cpu_extra.json"))
