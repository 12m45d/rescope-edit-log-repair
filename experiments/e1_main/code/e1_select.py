"""E1 (ii): the selection in favour of the baseline (R3, R4 extended, R25), from the gate-level cells of the two development logs at
alpha = 0.05 (20 calibration draws).  Writes out/best_cosine_gate.json with the driver's write_selection (both backbones in one call).

Per backbone, among ALL variants of the cosine family (the registered cosine gate with both calibrations included): the variant with
the largest H3 coverage (mean over the 20 draws of (T-Gen + I-Gen) / 2 coverage on the odd edits at the raised scale) is "the best
gate of the cosine family"; a second one for H2 (mean coverage of text and image rephrases) if it differs.  The same for the three
retrieval rules.  Rules fixed before any number was looked at: e1c.H3_UNREACHABLE_RULE, e1c.TIE_RULE.
"""
import os
import sys

from e1c import *  # noqa: F401,F403
import e1c


def read(log, lab):
    p = os.path.join(GATE, log, f"{lab}__a0.05.json")
    b = jload(p)
    cells = [b["cells"][str(d)] for d in range(N_DRAWS)]
    return cells


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def row(log, lab):
    cells = read(log, lab)
    h3 = [c["h3"]["rephrase"]["mean"] if c["h3"]["reachable"] else 0.0 for c in cells]
    rch = [c for c in cells if c["h3"]["reachable"]]
    p = fx.review1.parse(lab) or {"feature": "cosine", "kappa": None, "calibration": "two_block"}
    return {"name": lab, "variant": p["feature"], "kappa": p["kappa"], "calibration": p["calibration"], "n_draws": len(cells),
            "H3_coverage_mean": float(np.mean(h3)), "H3_coverage_by_draw": h3, "H3_n_draws_unreachable": len(cells) - len(rch),
            "H3_n_edits_at_cap_mean": mean([c["h3"]["n_edits_at_cap"] for c in rch]), "H3_raise_mean": mean([c["h3"]["delta"] for c in rch]),
            "H3_test_mean_fired_mean": mean([c["h3"]["test"]["mean_fired"] for c in rch]),
            "H3_heldout_disjoint_fired_mean": mean([c["h3"]["heldout_disjoint"]["fired"] for c in rch]),
            "H2_coverage_mean": float(np.mean([c["h2_cov"] for c in cells])), "H2_coverage_by_draw": [c["h2_cov"] for c in cells],
            "tgen_coverage_mean": mean([c["st"]["edit"]["tgen"]["own_top"] for c in cells]), "igen_coverage_mean": mean([c["st"]["edit"]["igen"]["own_top"] for c in cells]),
            "heldout_disjoint_fired_mean": mean([c["st"]["probe"]["heldout_disjoint"]["fired"] for c in cells]),
            "heldout_disjoint_fired_literal_mean": mean([c["literal"]["disjoint.all"]["fired"] for c in cells]),
            "n_capped_mean": mean([c["n_capped"] for c in cells]), "n_capped_max": max(c["n_capped"] for c in cells),
            "near_scope_fired_mean": {k: mean([c["st"]["hn"][k]["fired"] for c in cells]) for k in HN}}


def best(table, key, order):
    top = max(r[key] for r in table.values())
    for nm in order:                                                    # the tie rule: first in the registered order
        if table[nm][key] == top:
            return nm


def main():
    bbs = {}
    for log in DEV:
        L = fx.logs.open(log)
        cos = {nm: row(log, nm) for nm in COS_FAMILY}
        ret = {nm: row(log, nm) for nm in RETR_FAMILY}
        c3, c2 = best(cos, "H3_coverage_mean", COS_FAMILY), best(cos, "H2_coverage_mean", COS_FAMILY)
        r3, r2 = best(ret, "H3_coverage_mean", RETR_FAMILY), best(ret, "H2_coverage_mean", RETR_FAMILY)
        det = lambda t, nm: {k: t[nm][k] for k in ("name", "variant", "calibration", "kappa", "H3_coverage_mean", "H2_coverage_mean")}   # noqa: E731
        bbs[L.backbone] = {"log": log, "cosine_family": {"H3": c3, "H2": c2}, "retrieval": {"H3": r3, "H2": r2},
                           "cosine_family_selected": {"H3": det(cos, c3), "H2": det(cos, c2), "differs": c3 != c2,
                                                      "H3_x_form": fx.review1.x_form(c3), "H2_x_form": fx.review1.x_form(c2)},
                           "retrieval_selected": {"H3": det(ret, r3), "H2": det(ret, r2), "differs": r3 != r2},
                           "table_cosine_family": cos, "table_retrieval_family": ret}
    norm = fx.review1.write_selection(
        bbs, SEL, experiment="E1", alpha=0.05, n=200, n_draws=N_DRAWS, level="gate level (key caches), development logs only",
        read_outs={"H3": "mean over the calibration draws of the coverage (own edit released) of (T-Gen + I-Gen) / 2 on the odd edits at the smallest common "
                         "raise of the scale with mean firing <= 10 percent on the fit hard negatives (fx.sweeps.h3, guards kept)",
                   "H2": "mean over the calibration draws of (T-Gen coverage + I-Gen coverage) / 2 on all edits (fx.metrics.gate_stats)"},
        rules={"h3_unreachable": e1c.H3_UNREACHABLE_RULE, "ties": e1c.TIE_RULE,
               "retr_top1": "retr_sum / retr_min release by margin (R25, item 1)", "x_form": "the _x form of a shared-scale gate is the two-block _x form (R25, item 2)"},
        code=fx.code_fingerprint(), written_by="e1_main/code/e1_select.py")
    print(norm)
    for bb, e in bbs.items():
        print(bb, e["log"])
        for fam in ("table_cosine_family", "table_retrieval_family"):
            print(fx.io.text_table(["variant", "H3 cov", "unreach", "at cap", "H2 cov", "fire hd", "capped"],
                                   [[nm, f"{r['H3_coverage_mean']:.2f}", r["H3_n_draws_unreachable"], "-" if r["H3_n_edits_at_cap_mean"] is None else f"{r['H3_n_edits_at_cap_mean']:.1f}",
                                     f"{r['H2_coverage_mean']:.2f}", f"{r['heldout_disjoint_fired_mean']:.2f}", f"{r['n_capped_mean']:.1f}"] for nm, r in e[fam].items()]))


if __name__ == "__main__":
    if os.path.exists(SEL) and "--force" not in sys.argv:
        raise SystemExit(f"{SEL} exists (the selection is made once)")
    main()
