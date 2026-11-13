"""P12 -- compact curves for the final message, printed from out/results.json with render.cell (the same cells as
results.md section 4; no hand-typed number).

usage: python final_curves.py
Per log and method: n: HELD-OUT D1 % on the disjoint rows, seed mean (min-max) [Rel exact, seed mean (min-max)] {seeds}
"""
from __future__ import annotations

import json
import os

import render as RD

R = json.load(open(os.path.join(RD.P12, "out", "results.json")))
for lg in RD.LOGS:
    L = R["logs"][lg]
    nn = L["rows"]["NONE"]
    print(f"\n{RD.SHORT[lg]} -- unrepaired: disjoint D1 {nn['heldout_disjoint']['D1_rate']:.1f} ({nn['heldout_disjoint']['n_rows']} rows; full pool "
          f"{nn['heldout']['D1_rate']:.1f}), Rel exact {nn['edits']['rel']['exact']:.0f}")
    for m in ("COMP_B", "COMP_BP", "REEDIT_B", "RB_TOT", "RB_UNION"):
        cn = L["curves"].get(m, {})
        parts = []
        for n in RD.NS:
            c = cn.get(n)
            if c is None:
                parts.append(f"n={n}: -")
                continue
            parts.append(f"n={n}: {RD.cell(c, 'heldout_disjoint_D1_rate', 1)} [{RD.cell(c, 'rel_exact', 0)}] {{{','.join(str(s) for s in c['seeds'])}}}")
        print(f"  {RD.LAB[m]}: " + "; ".join(parts))
    D = L["d2"]
    if D:
        nd, aa = D.get("NONE"), D.get("ALL")
        print("  D2 after / pre-correct held-out rows (stored D2 resolved / new D2):",
              "" if nd is None else f"unrepaired {nd['n_D2']}/{nd['n_pre_correct']} (flags == stored {nd['D2_flags_equal_stored']}, long answers == stored {nd['long_eq_stored_post']});",
              "" if aa is None else f"W_0 {aa['n_D2']} (long == stored pre {aa['long_eq_stored_pre']}/{aa['n_pre_correct']})")
        for m in ("COMP_B", "COMP_BP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION", "COMP_S", "COMP_SP"):
            seq = [f"n={n}: {D[RD.cfg(m, n)]['n_D2']} ({D[RD.cfg(m, n)]['stored_D2_resolved']}/{D[RD.cfg(m, n)]['new_D2']})" for n in RD.D2_NS if RD.cfg(m, n) in D]
            if seq:
                print(f"    {RD.LAB[m]}: " + "; ".join(seq))
    print(f"  missing protocol configs: {len(L.get('missing', []))}; failed: {[x['config'] for x in L['failed']]}; oom: {[x['config'] for x in L.get('oom_unfinished', [])]}")
T = R.get("text_h2h") or {}
for lg, src in (T.get("text") or {}).items():
    print(f"\ntext h2h {RD.SHORT[lg]}: {src['n_complaints_complete']} complaints")
    for c, s in src["configs"].items():
        if "clean" not in s or s["n_full_eval"] < s["n"]:
            print(f"  {c}: resolved {s['D1_resolved']}/{s['n']}")
            continue
        print(f"  {c}: resolved {s['D1_resolved']}/{s['n']}; clean {s['clean']}; edits broken/repair {s['edits_broken_total']['mean']:.2f}; held-out lost "
              f"{s['collateral']['all']['noncomplaints_lost']['mean']:.1f} / fixed {s['collateral']['all']['complaints_fixed']['mean']:.1f}; norm/log {s['fro_rel_median']:.3f}; "
              f"passes {s['cost']['n_forward']['mean']:.0f}/{s['cost']['n_backward']['mean']:.0f}; D2 {s['D2_pooled']['resolved']}/{s['D2_pooled']['n']}")
    if src.get("reedit_with_fallback_resolved"):
        print("  REEDIT with the documented fallback:", src["reedit_with_fallback_resolved"])
