"""P12 -- compact tables for the final message, printed from out/results.json (no hand-typed number).

usage: python final_tables.py
Cell format: resolved/n ; HELD-OUT D1 % (full pool) ; +held-out complaints fixed / -non-complaints lost ; Rel/T-Gen/I-Gen exact ; norm/log
"""
from __future__ import annotations

import json
import os

P12 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = json.load(open(os.path.join(P12, "out", "results.json")))
LAB = {"COMP_B": "COMP-B", "COMP_BP": "COMP-B+", "COMP_S": "COMP-S", "COMP_SP": "COMP-S+", "COMP_B_ALLPOS": "ALLPOS", "REEDIT_B": "REEDIT-B",
       "REEDIT_S": "REEDIT-S", "RB_TOT": "RB-TOT", "RB_UNION": "RB-UNION", "COMP_BR": "COMP-B-RESTORE*", "REEDIT_SC": "REEDIT-SC*"}
NS = ("1", "10", "100", "ALL")


def cell(r):
    h, e = r["heldout"], r["edits"]
    return (f"{r['reported']['resolved']}/{r['reported']['n']};{h['D1_rate']:.1f};+{h['complaints_fixed']}/-{h['noncomplaints_lost']};"
            f"{e['rel']['exact']:.0f}/{e['tgen']['exact']:.0f}/{e['igen']['exact']:.0f};{r['fro_rel_to_log']:.2f}")


def same(a, b):
    return a is not None and b is not None and cell(a) == cell(b)


for lg, L in R["logs"].items():
    rows = L["rows"]
    nn, aa = rows["NONE"], rows["ALL"]
    lk = L["leak"]
    print(f"\n### {lg} (ALL = {L['n_all']})")
    print(f"NONE: D1 {nn['heldout']['D1_rate']:.1f} ({nn['heldout']['n_D1_after']}/{nn['heldout']['n_rows']}; disjoint {nn['heldout_disjoint']['D1_rate']:.1f}), "
          f"Rel/TGen/IGen {nn['edits']['rel']['exact']:.0f}/{nn['edits']['tgen']['exact']:.0f}/{nn['edits']['igen']['exact']:.0f}; "
          f"W_0: D1 {aa['heldout']['D1_rate']:.1f}, Rel/TGen/IGen {aa['edits']['rel']['exact']:.0f}/{aa['edits']['tgen']['exact']:.0f}/{aa['edits']['igen']['exact']:.0f}; "
          f"held-out {nn['heldout']['n_complaints']} complaints / {nn['heldout']['n_noncomplaints']} non-complaints; twins {lk['n_twins_valid']} ({lk['n_twin_complaints']} complaints)")
    print("| method | " + " | ".join(f"n={n}" for n in NS) + " |")
    print("|---|" + "---|" * len(NS))
    fp = L.get("rows_first_pass", {})
    for m in LAB:
        cs = [rows.get(f"{m}__n{n}__s0") or fp.get(f"{m}__n{n}__s0") for n in NS]
        marks = ["" if rows.get(f"{m}__n{n}__s0") or not fp.get(f"{m}__n{n}__s0") else "(first pass) " for n in NS]
        if all(c is None for c in cs):
            continue
        # a sequential COMP row identical to the joint row in every shown cell is merged into it
        twin = {"COMP_S": "COMP_B", "COMP_SP": "COMP_BP"}.get(m)
        if twin and all(same(c, rows.get(f"{twin}__n{n}__s0")) for c, n in zip(cs, NS) if c is not None):
            print(f"| {LAB[m]} | identical to {LAB[twin]} in all {sum(c is not None for c in cs)} cells |" + " |" * (len(NS) - 1))
            continue
        print(f"| {LAB[m]} | " + " | ".join("-" if c is None else mk + cell(c) for c, mk in zip(cs, marks)) + " |")
    print("disjoint D1 %:", "; ".join(f"{LAB[m]} " + "/".join("-" if rows.get(f'{m}__n{n}__s0') is None else f"{rows[f'{m}__n{n}__s0']['heldout_disjoint']['D1_rate']:.1f}" for n in NS)
                                         for m in ("COMP_B", "COMP_BP", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION") if any(rows.get(f"{m}__n{n}__s0") for n in NS)))
    D = L["d2"]
    if D:
        print("D2 after (stored resolved/new) of", D.get("NONE", {}).get("n_pre_correct"), "pre-correct held-out rows; NONE", D.get("NONE", {}).get("n_D2"), "; W_0", D.get("ALL", {}).get("n_D2"))
        for m in ("COMP_B", "COMP_BP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION", "COMP_S", "COMP_SP"):
            seq = [f"n={n}: {D[f'{m}__n{n}__s0']['n_D2']} ({D[f'{m}__n{n}__s0']['stored_D2_resolved']}/{D[f'{m}__n{n}__s0']['new_D2']})" for n in ("10", "100", "ALL") if f"{m}__n{n}__s0" in D]
            if seq:
                print("  ", LAB[m], "; ".join(seq))
    RH = L.get("rho") or {}
    if RH:
        print("rho, disjoint held-out rows (median complaints / non-complaints; AUC):", "; ".join(
            f"{k.replace('__s0', '')}: {v['median_complaints_disjoint']:.2f}/{v['median_noncomplaints_disjoint']:.2f}; {v['auc_complaints_lower_disjoint']:.2f}"
            for k, v in RH.items() if k not in ("NONE", "ALL") and not k.startswith(("COMP_S", "REEDIT_S"))))
    for tag in ("operating_points", "operating_points_disjoint", "operating_points_extra"):
        ops = L.get(tag) or []
        if ops:
            o = ops[0]
            r = rows[o["config"]]
            print(f"{tag}: {o['config']} D1 {o['heldout_D1_rate']:.1f} [{o['ci'][0]:.1f},{o['ci'][1]:.1f}] ({o['delta_vs_none']:+.1f}, p={o['mcnemar_p']:.2g}); disjoint {o['disjoint_D1_rate']:.1f} "
                  f"({o['disjoint_delta_vs_none']:+.1f}, p={o['disjoint_mcnemar_p']:.2g}); Rel {o['rel_exact']:.0f}; rank {r['rank'].get('correction')}; norm {r['fro_rel_to_log']:.2f}; "
                  f"passes {r['cost'].get('n_forward')}/{r['cost'].get('n_backward')}; {r['cost'].get('sec'):.0f}s; candidates {L['n_operating_candidates']}")
    print("missing:", len(L.get("missing", [])), L.get("missing", [])[:8], "| failed:", [x["config"] for x in L["failed"]], "| oom:", [x["config"] for x in L.get("oom_unfinished", [])],
          "| extra missing:", L.get("missing_extra"))
T = R.get("text_h2h")
if T:
    for lg, src in T["text"].items():
        print(f"\n### text h2h {lg}: {src['n_complaints_complete']} complaints")
        for c, s in src["configs"].items():
            if "clean" not in s or s["n_full_eval"] < s["n"]:
                print(f"  {c}: resolved {s['D1_resolved']}/{s['n']}")
                continue
            print(f"  {c}: resolved {s['D1_resolved']}/{s['n']}; clean {s['clean']}; broken {s['edits_broken_total']['mean']:.2f}; lost {s['collateral']['all']['noncomplaints_lost']['mean']:.1f}; "
                  f"fixed {s['collateral']['all']['complaints_fixed']['mean']:.1f}; norm/log {s['fro_rel_median']:.3f}; passes {s['cost']['n_forward']['mean']:.0f}/{s['cost']['n_backward']['mean']:.0f}; D2 {s['D2_pooled']['resolved']}/{s['D2_pooled']['n']}")
        if src.get("reedit_with_fallback_resolved"):
            print("  REEDIT with fallback:", src["reedit_with_fallback_resolved"], src.get("reedit_fallback_supplement"))
        ref = T["rq1_image_reference"].get(lg)
        if ref:
            print(f"  [rq1 image reference, {ref['n_complaints_complete']} complaints]", "; ".join(f"{c}: {s['D1_resolved']}/{s['n']}, clean {s.get('clean')}" for c, s in ref["configs"].items()))
