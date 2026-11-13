"""P12 -- out/results.json -> results.md.  Every number is read from results.json (aggregate.py); the prose
around the numbers is fixed text whose direction words are selected by the data (no hand-typed result).

usage: python render.py
"""
from __future__ import annotations

import json
import os

P12 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Qwen (TB-Q100-ae-l2_3)", "TB-L100-ae-l2_3": "LLaVA (TB-L100-ae-l2_3)"}
LAB = {"NONE": "NONE (unrepaired)", "ALL": "ALL (W_0)", "COMP_B": "COMP-B", "COMP_BP": "COMP-B+", "COMP_S": "COMP-S",
       "COMP_SP": "COMP-S+", "COMP_B_ALLPOS": "COMP-B-ALLPOS", "REEDIT_B": "REEDIT-B", "REEDIT_S": "REEDIT-S",
       "REEDIT_SC": "REEDIT-SC (expl.)", "COMP_BR": "COMP-B-RESTORE (expl.)", "RB_TOT": "RB-TOT", "RB_UNION": "RB-UNION"}
ORDER = ["COMP_B", "COMP_BP", "COMP_S", "COMP_SP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION", "REEDIT_SC", "COMP_BR"]
MAIN = ["COMP_B", "COMP_BP", "COMP_S", "COMP_SP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION"]
NS = ["1", "2", "5", "10", "20", "50", "100", "ALL"]
TAB_NS = ["1", "10", "100", "ALL"]
D2_NS = ["10", "100", "ALL"]


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def pv(p):
    return "<1e-4" if p < 1e-4 else f"{p:.2g}"


def cfg(m, n, s=0):
    return f"{m}__n{n}__s{s}"


def cell(c, key, d=1):
    """seed mean (min-max) of a curve cell; dagger = first-pass REEDIT record."""
    if c is None or c.get(key) is None:
        return "-"
    v = c[key]
    mk = "\u2020" if c.get("first_pass") else ""
    if v["n_seeds"] > 1:
        return f"{v['mean']:.{d}f} ({v['min']:.{d}f}-{v['max']:.{d}f}){mk}"
    return f"{v['mean']:.{d}f}{mk}"


def prot_s(r):
    p = r.get("rank", {}).get("protection")
    if isinstance(p, dict):
        return f"K0 {p['K0_projected_rank']} (hard) + Kp {p['Kp_soft_columns']} (soft)"
    return str(p)


def rank_s(r):
    return f"{r.get('rank', {}).get('correction', '-')} / {prot_s(r)}"


def nlab(n, L):
    return f"ALL ({L['n_all']})" if n == "ALL" else str(n)


def updown(a, b, eps=1e-9):
    return "below" if a < b - eps else ("above" if a > b + eps else "equal to")


FP = "\u2020"            # dagger: first-pass REEDIT record (harness default call), shown where the re-run is missing


def getrow(L, name):
    """(row, mark): the record of a config, or its first-pass REEDIT record (mark = dagger) if the re-run is missing."""
    r = L["rows"].get(name)
    if r is not None:
        return r, ""
    r = L.get("rows_first_pass", {}).get(name)
    return (r, FP) if r is not None else (None, "")


def row_line(r, n_lab, mark=""):
    h, e, lc, c = r["heldout"], r["edits"], r["loc"], r["cost"]
    rep = "-" if r["method"] in ("NONE", "ALL") else f"{r['reported']['resolved']}/{r['reported']['n']}"
    hd, tr = r["heldout_disjoint"], r["heldout_twins_of_reported"]
    return (f"| {LAB[r['method']]}{mark} | {n_lab} | {rep} | {h['D1_rate']:.1f} [{h['D1_rate_ci']['lo']:.1f}, {h['D1_rate_ci']['hi']:.1f}] | "
            f"{h['complaints_fixed']}/{h['n_complaints']} | {h['noncomplaints_lost']}/{h['n_noncomplaints']} | "
            f"{hd['D1_rate']:.1f} [{hd['D1_rate_ci']['lo']:.1f}, {hd['D1_rate_ci']['hi']:.1f}] (+{hd['complaints_fixed']} / -{hd['noncomplaints_lost']}) | "
            f"{tr['complaints_fixed']}/{tr['n_rows']} | "
            f"{e['rel']['exact']:.0f} / {e['tgen']['exact']:.0f} / {e['igen']['exact']:.0f} | "
            f"{r['edits_broken_total']} ({e['rel']['n_broken']}/{e['tgen']['n_broken']}/{e['igen']['n_broken']}) / {r['edits_fixed_total']} | "
            f"{lc['tloc']['after']:.1f} / {lc['iloc']['after']:.1f} | {r['fro_rel_to_log']:.3f} | {rank_s(r)} | "
            f"{f(c.get('n_forward'), 0)} / {f(c.get('n_backward'), 0)} | {f(c.get('sec'), 1)} |")


HEAD = ("| method | n | reported resolved | HELD-OUT D1 rate % [95% CI] | held-out complaints fixed | non-complaints lost | "
        "disjoint HELD-OUT D1 % [CI] (+fixed / -lost) | twins of reported complaints fixed | "
        "Rel / T-Gen / I-Gen exact | edit checks broken (rel/tgen/igen) / fixed | T-Loc / I-Loc | norm / log | rank: correction / protection | "
        "passes fwd / bwd | repair s |")
SEP = "|" + "---|" * 15


# =========================================================================== section 1
def findings(R):
    md = []
    logs = [lg for lg in LOGS if lg in R["logs"]]
    tol = R["protocol"]["rel_tol"]
    P = R["protocol"]
    # ---- sanity
    md += ["**Sanity.**"]
    for lg in logs:
        L = R["logs"][lg]
        nn, aa = L["rows"].get("NONE"), L["rows"].get("ALL")
        bc, st = L["setup"]["baseline_checks"], L["setup"]["stored"]
        if nn is None or aa is None:
            md.append(f"- {SHORT[lg]}: end points missing.")
            continue
        sn, sa = nn["sanity"], aa["sanity"]
        ok_none = sn["repair_D1_set_equals_stored"] and sn["heldout_D1_set_equals_stored"] and sn["edits_identical_to_baseline"] and bc["Rel_matches_stored"]
        ok_all = sa["repair_D1_complaints_after"] == 0 and sa["heldout_D1_complaints_after"] == 0 and sa["weight_equals_W0"]
        md.append(f"- {SHORT[lg]}: NONE {'reproduces' if ok_none else 'DOES NOT reproduce'} the stored baseline -- D1 flag of every valid row equal to "
                  f"`probe_records_v2.jsonl` on the REPAIR pool ({sn['n_repair_rows']} rows: {sn['repair_D1_set_equals_stored']}) and on the HELD-OUT pool "
                  f"({sn['n_heldout_rows']} rows: {sn['heldout_D1_set_equals_stored']}); Rel exact {bc['Rel_exact_WE']:.1f} / token {bc['Rel_token_WE']:.2f} on the "
                  f"testbed's batch-8 path = stored {st['Rel_exact']:.1f} / {st['Rel_token']:.2f} ({bc['Rel_matches_stored']}); the B=1 teacher-forced Rel exact used "
                  f"in every table below is {bc['Rel_B1_exact_WE']:.1f}. ALL (W_0) {'gives' if ok_all else 'DOES NOT give'} zero D1 complaints "
                  f"(REPAIR {sa['repair_D1_complaints_after']}, HELD-OUT {sa['heldout_D1_complaints_after']}; weight bitwise W_0: {sa['weight_equals_W0']}); "
                  f"its Rel exact is {aa['edits']['rel']['exact']:.0f}.")
    md.append("")
    # ---- leak
    md += ["**Anomaly 0 (protocol level) -- the HELD-OUT pool is not disjoint from the REPAIR pool.** The harness splits the 500 probe items by uid "
           "(`harness/sampler.py`, seed 20261109; the same split is used by every experiment that calls `sampler.pool_of`), but VLKEB eval items "
           "share image-locality probes: the same (image, question) occurs under several uids, on both sides of the split. A held-out row whose "
           "query is identical to a reported complaint is that complaint (same input, same y0): repairing the reported one repairs it, which is "
           "not generalisation. Text queries are all distinct. Below, **disjoint HELD-OUT** = the valid HELD-OUT rows whose query does not occur in "
           "the REPAIR pool; both versions are reported, the protocol's full pool first. Any HELD-OUT number computed on this split contains these rows."]
    for lg in logs:
        L = R["logs"][lg]
        lk = L["leak"]
        nn = L["rows"].get("NONE")
        extra = ""
        if nn is not None:
            extra = (f" Unrepaired D1 rate: full pool {nn['heldout']['D1_rate']:.1f}% ({nn['heldout']['n_rows']} rows), disjoint "
                     f"{nn['heldout_disjoint']['D1_rate']:.1f}% ({nn['heldout_disjoint']['n_rows']} rows), twins {nn['heldout_twins']['D1_rate']:.1f}% "
                     f"({nn['heldout_twins']['n_rows']} rows).")
        rall = L["rows"].get(cfg("COMP_B", "ALL"))
        dropped = "-" if rall is None else rall["comp"]["n_dropped"]
        tc = lk["twin_outcome_check"]
        md.append(f"- {SHORT[lg]}: {lk['n_distinct_queries']['image']} distinct image queries in {lk['n_rows']['image']} image rows "
                  f"({lk['n_distinct_queries']['text']} / {lk['n_rows']['text']} for text); {lk['n_twins_valid']} of the {lk['n_heldout_valid']} valid HELD-OUT rows "
                  f"are twins of a REPAIR-pool row ({lk['by_ptype']['image']} image, {lk['by_ptype']['text']} text): {lk['n_twin_complaints']} held-out complaints "
                  f"(all {lk['n_twin_complaints_of_repair_complaint']} twins of REPAIR D1 complaints, {100.0 * lk['n_twin_complaints'] / lk['n_heldout_valid_complaints']:.1f}% "
                  f"of the held-out complaints) and {lk['n_twin_noncomplaints']} held-out non-complaints. Inside the REPAIR pool {lk['repair_redundant_complaints']} of the "
                  f"{lk['repair_complaints']} complaints repeat an earlier query ({lk['repair_redundant_answer_keys']} redundant answer keys; the joint "
                  f"compensation of all REPAIR complaints drops {dropped} query directions).{extra} Check: over {tc['n_configs']} repaired "
                  f"configurations, a held-out twin of a reported complaint has the same outcome as that complaint in {tc['agree']}/{tc['n']} cases "
                  f"(two different evaluation paths: suffix engine for held-out rows, full forward for reported complaints).")
    md.append("")
    # ---- headline per log
    anyfp = any(R["logs"][lg].get("rows_first_pass") for lg in logs)       # a first-pass REEDIT record is still shown
    md += ["**Finding 1 -- what happens to the unreported complaints (HELD-OUT pool; seed 0; HELD-OUT D1 rate % full pool / disjoint rows, "
           "Rel exact in brackets" + (f"; {FP} = first-pass REEDIT record, see the note under the main table" if anyfp else "") + ").**"]
    for lg in logs:
        L = R["logs"][lg]
        rows = L["rows"]
        nn = rows.get("NONE")
        if nn is None:
            continue
        parts = []
        for m in ("COMP_B", "COMP_BP", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION"):
            seq = []
            for n in TAB_NS:
                r, mk = getrow(L, cfg(m, n))
                if r is not None:
                    seq.append(f"n={n}{mk}: {r['heldout']['D1_rate']:.1f} / {r['heldout_disjoint']['D1_rate']:.1f} ({r['edits']['rel']['exact']:.0f})")
            if seq:
                parts.append(f"{LAB[m]} " + ", ".join(seq))
        md.append(f"- {SHORT[lg]}: unrepaired {nn['heldout']['D1_rate']:.1f} / {nn['heldout_disjoint']['D1_rate']:.1f} "
                  f"({nn['heldout']['n_D1_after']}/{nn['heldout']['n_rows']} and {nn['heldout_disjoint']['n_D1_after']}/{nn['heldout_disjoint']['n_rows']} rows; "
                  f"Rel exact {nn['edits']['rel']['exact']:.0f}). After the repair: " + "; ".join(parts) + ".")
    md.append("")
    # ---- generalisation: fixed vs lost
    md += ["**Finding 2 -- what a repair does to unreported rows: held-out complaints fixed and held-out non-complaints lost "
           "(disjoint held-out rows, seed 0; +fixed / -lost).**"]
    for lg in logs:
        rows = R["logs"][lg]["rows"]
        parts = []
        for m in ("COMP_B", "COMP_BP", "REEDIT_B", "RB_TOT", "RB_UNION"):
            seq = []
            for n in TAB_NS:
                r, mk = getrow(R["logs"][lg], cfg(m, n))
                if r is not None:
                    h = r["heldout_disjoint"]
                    seq.append(f"n={n}{mk}: +{h['complaints_fixed']} / -{h['noncomplaints_lost']}")
            if seq:
                parts.append(f"{LAB[m]} " + ", ".join(seq))
        nn = rows.get("NONE")
        if nn is not None:
            md.append(f"- {SHORT[lg]} (disjoint rows: {nn['heldout_disjoint']['n_complaints']} held-out complaints, {nn['heldout_disjoint']['n_noncomplaints']} "
                      f"held-out non-complaints): " + "; ".join(parts) + ".")
    md.append("")
    # ---- per-family readings, every direction word chosen by the data
    md += ["**Finding 2a -- fix rate against loss rate on unreported rows (disjoint held-out rows, seed 0; fix rate % = held-out complaints "
           "fixed / held-out complaints | loss rate % = held-out non-complaints lost / held-out non-complaints).** The two rates are per-row "
           "probabilities on two different sets, so they can be compared directly: equal rates mean that a held-out row is as likely to be "
           "changed for the worse as for the better. (No control with an unrelated correction of the same norm was run: whether the flips of "
           "the compensation and re-edit methods are specific to the reported complaints is not established by these numbers.)"]
    for lg in logs:
        L = R["logs"][lg]
        parts = []
        for m in ("COMP_B", "COMP_BP", "REEDIT_B", "RB_TOT", "RB_UNION"):
            seq = []
            for n in TAB_NS:
                r, mk = getrow(L, cfg(m, n))
                if r is not None and "flip_rates_disjoint" in r:
                    fr = r["flip_rates_disjoint"]
                    seq.append(f"n={n}{mk}: {fr['fix']:.0f} | {fr['loss']:.0f}")
                elif r is not None:
                    hd = r["heldout_disjoint"]
                    seq.append(f"n={n}{mk}: {100.0 * hd['complaints_fixed'] / max(hd['n_complaints'], 1):.0f} | "
                               f"{100.0 * hd['noncomplaints_lost'] / max(hd['n_noncomplaints'], 1):.0f}")
            if seq:
                parts.append(f"{LAB[m]} " + ", ".join(seq))
        if parts:
            md.append(f"- {SHORT[lg]}: " + "; ".join(parts) + ".")
    md.append("")
    md += ["**Finding 2a' -- do the held-out flips accumulate along the nested reported sets? Of the held-out rows flipped by the repair of n1 "
           "complaints, the number still flipped after the repair of the superset of n2 complaints (seed 0; fixed kept / fixed at n1 -> fixed at n2 | "
           "lost kept / lost at n1 -> lost at n2).**"]
    for lg in logs:
        L = R["logs"][lg]
        for m in ("COMP_B", "REEDIT_B", "RB_TOT", "RB_UNION"):
            ent = L.get("flip_persistence", {}).get(m, {}).get("0")
            if ent:
                md.append(f"- {SHORT[lg]}, {LAB[m]}: " + "; ".join(
                    f"{e['n1']}->{e['n2']}: {e['fixed_kept']}/{e['fixed_n1']} -> {e['fixed_n2']} | {e['lost_kept']}/{e['lost_n1']} -> {e['lost_n2']}" for e in ent) + ".")
    md.append("")
    md += ["**Finding 2b -- protecting a regression suite of known-good rows (COMP-B+ against COMP-B).** e = mean share of a reported "
           "complaint key's norm left after projecting out the protected span; the smaller e, the larger the correction needed for the same "
           "restoration. Largest n with both records, seed 0:"]
    for lg in logs:
        L = R["logs"][lg]
        nn = L["rows"].get("NONE")
        n = next((x for x in reversed(NS) if cfg("COMP_B", x) in L["rows"] and cfg("COMP_BP", x) in L["rows"]), None)
        if n is None or nn is None:
            continue
        a, b = L["rows"][cfg("COMP_B", n)], L["rows"][cfg("COMP_BP", n)]
        d0 = nn["heldout"]["D1_rate"]
        md.append(f"- {SHORT[lg]}, n = {nlab(n, L)}: COMP-B protects rank {a['comp']['protect_rank']} (e = {a['comp']['e_mean']:.2f}), correction norm "
                  f"{a['fro_rel_to_log']:.2f} of the log, held-out +{a['heldout']['complaints_fixed']} / -{a['heldout']['noncomplaints_lost']}, D1 rate "
                  f"{a['heldout']['D1_rate']:.1f}% ({updown(a['heldout']['D1_rate'], d0)} the unrepaired {d0:.1f}%); COMP-B+ protects rank "
                  f"{b['comp']['protect_rank']} (e = {b['comp']['e_mean']:.2f}), norm {b['fro_rel_to_log']:.2f}, held-out +{b['heldout']['complaints_fixed']} / "
                  f"-{b['heldout']['noncomplaints_lost']}, D1 rate {b['heldout']['D1_rate']:.1f}% ({updown(b['heldout']['D1_rate'], d0)} the unrepaired log); "
                  f"reported complaints resolved {a['reported']['resolved']}/{a['reported']['n']} against {b['reported']['resolved']}/{b['reported']['n']}.")
    md.append("")
    md += ["**Finding 2c -- re-editing (REEDIT) and held-out rows.** Net = held-out complaints fixed minus held-out non-complaints lost "
           "(full pool, seed 0" + (f"; {FP} first pass" if anyfp else "") + "):"]
    for lg in logs:
        L = R["logs"][lg]
        seq, nets = [], []
        for n in NS:
            r, mk = getrow(L, cfg("REEDIT_B", n))
            if r is None:
                continue
            net = r["heldout"]["complaints_fixed"] - r["heldout"]["noncomplaints_lost"]
            nets.append(net)
            seq.append(f"n={n}{mk}: {net:+d} (Rel {r['edits']['rel']['exact']:.0f}, {r['edits_broken_total']} edit checks broken)")
        if seq:
            verdict = ("negative at every n measured" if all(x < 0 for x in nets) else
                       ("positive at every n measured" if all(x > 0 for x in nets) else "of mixed sign"))
            md.append(f"- {SHORT[lg]}: " + "; ".join(seq) + f". The net is {verdict}.")
    md.append("")
    md += ["**Finding 2d -- rollback: HELD-OUT D1 rate against the number of edits removed (k of the logged edits rolled back; seed 0).**"]
    for lg in logs:
        L = R["logs"][lg]
        for m in ("RB_TOT", "RB_UNION"):
            seq = []
            for n in NS:
                r = L["rows"].get(cfg(m, n))
                if r is not None and "rollback" in r:
                    seq.append(f"n={n}: k={r['rollback']['k']}, D1 {r['heldout']['D1_rate']:.1f}%, Rel {r['edits']['rel']['exact']:.0f}")
            if seq:
                md.append(f"- {SHORT[lg]}, {LAB[m]}: " + "; ".join(seq) + ".")
    md.append("")
    # ---- best operating point
    md += [f"**Finding 3 -- best linear operating point (lowest HELD-OUT D1 rate with Rel exact within {tol:g} points of the unrepaired log).**"]
    for lg in logs:
        L = R["logs"][lg]
        ops = L["operating_points"]
        nn = L["rows"].get("NONE")
        if not ops or nn is None:
            md.append(f"- {SHORT[lg]}: no configuration inside the tolerance.")
            continue
        o = ops[0]
        r = L["rows"][o["config"]]
        dist = sorted({(LAB[x["method"]], str(x["n"])) for x in ops if abs(x["heldout_D1_rate"] - o["heldout_D1_rate"]) < 1e-9})
        cellc = L["curves"][o["method"]][str(o["n"])]
        ns_ = len(cellc["seeds"])
        seed_txt = ""
        if ns_ > 1:           # the selected configuration against the other seeds of its own cell (selection effect)
            inside = sum(1 for cn_ in cellc["configs"] if L["rows"][cn_]["rel_within_tol"])
            seed_txt = (f"The same method and n over its {ns_} seeds: HELD-OUT D1 rate {cell(cellc, 'heldout_D1_rate')}%, disjoint rows "
                        f"{cell(cellc, 'heldout_disjoint_D1_rate')}%, Rel exact {cell(cellc, 'rel_exact', 0)}; {inside} of the {ns_} seeds are inside the "
                        f"tolerance. ")
        md.append(f"- {SHORT[lg]}: **{LAB[o['method']]}, n = {o['n']}, seed {o['seed']}** -- HELD-OUT D1 rate {o['heldout_D1_rate']:.1f}% "
                  f"[{o['ci'][0]:.1f}, {o['ci'][1]:.1f}] against {nn['heldout']['D1_rate']:.1f}% unrepaired ({o['delta_vs_none']:+.1f} points; exact McNemar on the "
                  f"held-out rows p = {pv(o['mcnemar_p'])}; {r['heldout']['complaints_fixed']} complaints fixed, {r['heldout']['noncomplaints_lost']} non-complaints lost), "
                  f"Rel / T-Gen / I-Gen exact {r['edits']['rel']['exact']:.0f} / {r['edits']['tgen']['exact']:.0f} / {r['edits']['igen']['exact']:.0f} "
                  f"({r['edits_broken_total']} edit checks broken, {r['edits_fixed_total']} fixed), reported resolved {r['reported']['resolved']}/{r['reported']['n']}. "
                  f"Cost: correction of rank {r['rank'].get('correction')} (protection {prot_s(r)}), norm {r['fro_rel_to_log']:.3f} of the log, "
                  f"{f(r['cost'].get('n_forward'), 0)} forward / {f(r['cost'].get('n_backward'), 0)} backward passes, {f(r['cost'].get('sec'), 1)} s. "
                  f"It is the minimum over {L['n_operating_candidates']} configurations inside the tolerance, selected on the evaluation pool itself "
                  f"({ns_} seed{'s' if ns_ != 1 else ''} at this n): read it as an optimistic bound. {seed_txt}"
                  f"Tied at the same rate: {', '.join(f'{a} n={b}' for a, b in dist)}. "
                  f"The repaired log keeps {o['heldout_D1_rate'] / max(nn['heldout']['D1_rate'], 1e-9) * 100:.0f}% of the unrepaired HELD-OUT D1 rate. "
                  f"On the disjoint rows the same point gives {o['disjoint_D1_rate']:.1f}% [{o['disjoint_ci'][0]:.1f}, {o['disjoint_ci'][1]:.1f}] against "
                  f"{nn['heldout_disjoint']['D1_rate']:.1f}% ({o['disjoint_delta_vs_none']:+.1f} points, p = {pv(o['disjoint_mcnemar_p'])}); "
                  f"{r['heldout_twins_of_reported']['complaints_fixed']} of its {r['heldout']['complaints_fixed']} fixed held-out complaints are twins of reported "
                  f"complaints ({r['heldout_twins_of_reported']['n_rows']} such twins).")
        od = L.get("operating_points_disjoint") or []
        if od:
            o2 = od[0]
            md.append(f"  Selected on the disjoint rows instead: {LAB[o2['method']]}, n = {o2['n']}, seed {o2['seed']} -- {o2['disjoint_D1_rate']:.1f}% "
                      f"({o2['disjoint_delta_vs_none']:+.1f} points, p = {pv(o2['disjoint_mcnemar_p'])}; full pool {o2['heldout_D1_rate']:.1f}%), Rel exact {o2['rel_exact']:.0f}.")
        oc = L.get("operating_cells") or []
        if oc:                # the same selection on seed means (section 7): not a single draw
            c0 = oc[0]
            md.append(f"  By seed mean, over the cells (method x n) whose every seed is inside the tolerance: {LAB[c0['method']]}, n = {c0['n']} "
                      f"({c0['n_seeds']} seed{'s' if c0['n_seeds'] != 1 else ''}) -- HELD-OUT D1 rate {c0['heldout_D1_rate_mean']:.1f}% (disjoint rows "
                      f"{c0['disjoint_D1_rate_mean']:.1f}%), Rel exact {c0['rel_exact_mean']:.1f}.")
    md.append("")
    # ---- exploratory: the largest subspace the REPAIR pool defines
    if any(cfg("COMP_BR", "ALL") in R["logs"][lg]["rows"] for lg in logs):
        md += ["**Finding 3b (exploratory, outside the protocol and outside the selection above) -- the largest subspace the REPAIR pool defines.** "
               "COMP-B-RESTORE is the same joint compensation with the answer positions of EVERY valid REPAIR-pool row as queries (the reported "
               "complaints and the known-good rows), protecting K_ans: where COMP-B+ pins the known-good rows to the edited output, this restores "
               "W_0's module output on them as well."]
        for lg in logs:
            L = R["logs"][lg]
            r, nn, b = L["rows"].get(cfg("COMP_BR", "ALL")), L["rows"].get("NONE"), L["rows"].get(cfg("COMP_B", "ALL"))
            if r is None or nn is None:
                continue
            hd = r["heldout_disjoint"]
            md.append(f"- {SHORT[lg]}: {r['comp']['n_query_keys']} query keys, rank {r['comp']['rank']} of {r['comp']['d_in']} ({r['comp']['n_dropped']} dropped), "
                      f"norm {r['fro_rel_to_log']:.3f} of the log; reported complaints resolved {r['reported']['resolved']}/{r['reported']['n']}; HELD-OUT D1 rate "
                      f"{r['heldout']['D1_rate']:.1f}% [{r['heldout']['D1_rate_ci']['lo']:.1f}, {r['heldout']['D1_rate_ci']['hi']:.1f}] "
                      f"(unrepaired {nn['heldout']['D1_rate']:.1f}%" + ("" if b is None else f", COMP-B n=ALL {b['heldout']['D1_rate']:.1f}%")
                      + f"; +{r['heldout']['complaints_fixed']} / -{r['heldout']['noncomplaints_lost']}; exact McNemar vs unrepaired p = "
                      f"{pv(r['heldout']['paired_vs_WE']['p_exact'])}); disjoint rows {hd['D1_rate']:.1f}% against {nn['heldout_disjoint']['D1_rate']:.1f}% "
                      f"(+{hd['complaints_fixed']} / -{hd['noncomplaints_lost']}, p = {pv(hd['paired_vs_WE']['p_exact'])}); Rel / T-Gen / I-Gen exact "
                      f"{r['edits']['rel']['exact']:.0f} / {r['edits']['tgen']['exact']:.0f} / {r['edits']['igen']['exact']:.0f} "
                      f"({r['edits_broken_total']} edit checks broken, {r['edits_fixed_total']} fixed); T-Loc / I-Loc {r['loc']['tloc']['after']:.1f} / "
                      f"{r['loc']['iloc']['after']:.1f} (unrepaired {nn['loc']['tloc']['after']:.1f} / {nn['loc']['iloc']['after']:.1f}).")
        md.append("")
    # ---- D2
    if any(R["logs"][lg]["d2"] for lg in logs):
        md += ["**Finding 4 -- D2 complaints (ground-truth correctness lost) among the HELD-OUT rows, decoder v2, seed 0 "
               "(D2 after the repair; stored D2 resolved / new D2).**"]
        for lg in logs:
            D = R["logs"][lg]["d2"]
            if not D:
                continue
            nn = D.get("NONE")
            head = ""
            if nn is not None:
                head = (f"unrepaired {nn['n_D2']}/{nn['n_pre_correct']} pre-correct held-out rows (stored flags reproduced on "
                        f"{nn['D2_flags_equal_stored']}/{nn['n_pre_correct']}, long answers identical to the stored ones on {nn['long_eq_stored_post']}); ")
            parts = []
            for m in ("COMP_B", "COMP_BP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT", "RB_UNION"):
                seq = []
                for n in D2_NS:
                    d = D.get(cfg(m, n))
                    if d is not None:
                        seq.append(f"n={n}: {d['n_D2']} ({d['stored_D2_resolved']} / {d['new_D2']})")
                if seq:
                    parts.append(f"{LAB[m]} " + ", ".join(seq))
            aa = D.get("ALL")
            tail = "" if aa is None else f"; W_0: {aa['n_D2']} (long answers identical to the stored pre-edit ones on {aa['long_eq_stored_pre']}/{aa['n_pre_correct']})"
            md.append(f"- {SHORT[lg]}: {head}" + "; ".join(parts) + tail + ".")
        md.append("")
    # ---- module restore
    if any(R["logs"][lg].get("rho") for lg in logs):
        md += ["**Finding 5 -- how much of the log's change at the edited module is undone on unreported rows (diagnostic; rho = "
               "||(W_repaired - W_0) K|| / ||(W_E - W_0) K|| at the answer keys of each valid HELD-OUT row: 1 = unrepaired, 0 = pre-edit, > 1 = "
               "further from the pre-edit output). Disjoint held-out rows, seed 0; per config: median rho over held-out complaints / over held-out "
               "non-complaints; AUC = probability that a held-out complaint has a lower rho than a held-out non-complaint (0.5 = the repair does "
               "not tell them apart at the module).**"]
        for lg in logs:
            RH = R["logs"][lg].get("rho") or {}
            parts = []
            for m in ("COMP_B", "COMP_BP", "COMP_BR", "COMP_B_ALLPOS", "REEDIT_B", "RB_TOT", "RB_UNION"):
                seq = []
                for n in D2_NS:
                    d = RH.get(cfg(m, n))
                    if d is not None:
                        seq.append(f"n={n}: {d['median_complaints_disjoint']:.2f} / {d['median_noncomplaints_disjoint']:.2f}, AUC "
                                   f"{d['auc_complaints_lower_disjoint']:.2f}")
                if seq:
                    parts.append(f"{LAB[m]} " + "; ".join(seq))
            tws = []
            for m in ("COMP_B", "COMP_BP"):
                d = RH.get(cfg(m, "ALL"))
                if d is not None and d.get("n_twins_of_reported"):
                    tws.append(f"{LAB[m]} n=ALL: {d['n_twins_of_reported']} held-out twins of reported complaints, max rho {d['max_twins_of_reported']:.1e}")
            # the same medians from the two computations of rho: CPU (checkpoint tensor + composite delta, rho_cpu.py)
            # against the GPU pass of run_d2.py (the harness's own W_E, W_0 and applied correction)
            xs = [d["max_abs_diff_vs_run_d2_median"] for d in RH.values() if d.get("max_abs_diff_vs_run_d2_median") is not None]
            xchk = ("" if not xs else f" Cross-check of the two computations of rho (CPU: checkpoint tensor + stored composite delta; GPU pass of "
                    f"`run_d2.py`: the harness's weights): median over held-out complaints equal within {max(xs):.1e} over {len(xs)} configurations.")
            if parts:
                md.append(f"- {SHORT[lg]}: " + " | ".join(parts) + "." + (" Check of Anomaly 0 (identical keys are restored exactly): " + "; ".join(tws) + "." if tws else "") + xchk)
            else:
                md.append(f"- {SHORT[lg]}: not computed (the held-out answer keys of this log were not captured: no GPU start of the D2 pass).")
        md.append("")
        md += ["**Finding 5a -- how much of an unreported row lies in the subspace the reported complaints define (compensation only; share = "
               "||R^T K|| / ||K|| at the row's answer keys, R = orthonormal basis of the correction's row space = the reported complaint keys "
               "with the protected span projected out; 1 = the correction acts on the whole key, 0 = it cannot act on it). Disjoint held-out rows, "
               "seed 0; rank of the correction, then median share over held-out complaints / over held-out non-complaints (90th percentile over "
               "all disjoint rows).**"]
        for lg in logs:
            RH = R["logs"][lg].get("rho") or {}
            parts = []
            for m in ("COMP_B", "COMP_BP", "COMP_BR", "COMP_B_ALLPOS"):
                seq = []
                for n in D2_NS:
                    d = (RH.get(cfg(m, n)) or {}).get("span")
                    if d is not None:
                        seq.append(f"n={n}: rank {d['rank']}, {d['share_complaints_disjoint']:.2f} / {d['share_noncomplaints_disjoint']:.2f} "
                                   f"({d['share_q90_disjoint']:.2f})")
                if seq:
                    parts.append(f"{LAB[m]} " + "; ".join(seq))
            if parts:
                md.append(f"- {SHORT[lg]} (d_in = {R['logs'][lg]['setup']['d_in']}): " + " | ".join(parts) + ".")
        md.append("")
    if any(R["logs"][lg].get("protect_split") for lg in logs):
        md += ["**Finding 5c -- where the log's change on an unreported row sits (module level, CPU diagnostic, no behavioural evaluation).** Every "
               "edit-preserving compensation here leaves the log untouched on the span of the logged edits' answer keys (K_ans, projector P): "
               "the repaired module differs from W_0 by D P + (D + X)(I - P), D = W_E - W_0. Per valid disjoint HELD-OUT row, at its answer keys K: "
               "s = ||P K|| / ||K|| (share of the key inside the protected span); a = ||D P K|| / ||D K|| = the restoration ratio rho that "
               "restoring W_0 on everything orthogonal to the edits' answer keys would give (W_0 + D P; the complaint-free limit of such a repair); "
               "b = ||D (I - P) K|| / ||D K||; a and b are not an orthogonal split (they can cancel). Medians, held-out complaints / non-complaints:"]
        for lg in logs:
            ps = R["logs"][lg].get("protect_split")
            if not ps:
                md.append(f"- {SHORT[lg]}: not computed (the held-out answer keys of this log were not captured: no GPU start of the D2 pass).")
                continue
            md.append(f"- {SHORT[lg]}: K_ans has rank {ps['rank_K_ans']} of {ps['d_in']} (equal to the protection rank recorded by COMP-B: "
                      f"{ps['rank_matches_COMP_B']}) and carries {100 * ps['D_share_on_K_ans_span']:.0f}% of ||D||_F; s = {ps['s_complaints']:.2f} / "
                      f"{ps['s_noncomplaints']:.2f}; a = {ps['a_complaints']:.2f} / {ps['a_noncomplaints']:.2f}; b = {ps['b_complaints']:.2f} / "
                      f"{ps['b_noncomplaints']:.2f}; share of held-out complaints with a < 0.5: {100 * ps['frac_complaints_a_below_half']:.0f}% "
                      f"({ps['n_complaints']} complaints, {ps['n_noncomplaints']} non-complaints). Size of the log's change itself, "
                      f"||D K|| / ||W_0 K||: {ps['rel_change_complaints']:.2f} / {ps['rel_change_noncomplaints']:.2f}.")
        md.append("")
    suite_lines = []
    for lg in logs:
        D = R["logs"][lg]["d2"]
        sw = []
        for m in ("COMP_B", "COMP_BP", "COMP_BR"):
            for n in D2_NS:
                d = D.get(cfg(m, n), {}).get("regression_suite_in_sample")
                if d is not None:
                    sw.append(f"{LAB[m]} n={n}: {d['kept']}/{d['n']}")
        if sw:
            suite_lines.append(f"- {SHORT[lg]}: " + "; ".join(sw) + ".")
    if suite_lines:
        md += ["**Finding 5b -- the regression suite itself (IN-SAMPLE diagnostic on the REPAIR pool, not a collateral number): REPAIR-pool "
               "known-good rows that still reproduce y0 under the correction. COMP-B+ protects the edited module's output at the answer keys of "
               "exactly these rows; COMP-B-RESTORE restores W_0's module output there; the other positions of the same rows are free to change.**"]
        md += suite_lines + [""]
    # ---- identity
    md += ["**Anomaly 1 -- sequential compensation is the joint compensation.** With accumulating protection (K_ans + the keys of every earlier "
           "repaired complaint) the sequential operator satisfies the same constraints in the same row space as the joint solve, so COMP-S = COMP-B and "
           "COMP-S+ = COMP-B+ as matrices (exact arithmetic, no dropped direction). Measured:"]
    for lg in logs:
        idn = [i for i in R["logs"][lg]["seq_vs_batch_identity"] if not i["alias"] and i["rel_fro_diff"] is not None]
        if not idn:
            continue
        mx = max(i["rel_fro_diff"] for i in idn)
        same = sum(bool(i["same_heldout_D1"] and i["same_reported_resolved"] and i["same_edits"]) for i in idn)
        dmax = lambda k: max(abs(i[k]) for i in idn if i[k] is not None)  # noqa: E731
        md.append(f"- {SHORT[lg]}: max relative Frobenius difference between the sequential and the joint operator {mx:.1e} over {len(idn)} "
                  f"configurations; identical outcome (reported resolved, HELD-OUT D1 count, Rel / T-Gen / I-Gen exact) in {same}/{len(idn)}; "
                  f"largest differences: {dmax('d_reported_resolved')} reported complaints, {dmax('d_heldout_D1')} held-out rows, "
                  f"{dmax('d_edit_checks'):.0f} edit checks.")
    md.append("")
    # ---- REEDIT-S = REEDIT-B
    if any(R["logs"][lg].get("reedit_seq_vs_batch") for lg in logs):
        # title chosen by the data: largest difference in HELD-OUT rows between REEDIT-S and REEDIT-B per log
        dmx = {lg: max((abs(x["d_heldout_D1"]) for x in R["logs"][lg].get("reedit_seq_vs_batch", []) if not x["alias"]), default=None) for lg in logs}
        off = [f"{SHORT[lg]} (up to {v} HELD-OUT rows)" for lg, v in dmx.items() if v is not None and v > 2]
        md += ["**Anomaly 1b -- the re-edit stream is the joint re-edit in exact arithmetic"
               + ("" if not off else "; in half precision it is not the same model on " + ", ".join(off)) + ".** "
               "REEDIT-S continues the log like the original stream (batches of 10, Kp "
               "extended by the installed keys, residual z* - h taken at the current weights, targets z* computed before the stream as in the testbed "
               "build). That is recursive least squares: with A_b = P(Kp Kp^T + sum_{j<=b} K_j K_j^T) + l2 I, the update of batch b is "
               "D_b = (R_b - D_{<b} K_b) K_b^T A_b^-1 and the sum of the updates is the joint solve of all batches. REEDIT-S can therefore differ from "
               "REEDIT-B only through half-precision arithmetic: the write after every batch, and the residuals, which the stream takes from "
               "half-precision forwards at the rounded weights. Measured (configurations with more than one batch):"]
        for lg in logs:
            rs = [x for x in R["logs"][lg].get("reedit_seq_vs_batch", []) if not x["alias"]]
            if not rs:
                continue
            w = [x for x in rs if x["rel_fro_diff_weights"] is not None]
            wtxt = ("; ".join(f"n={x['n']}: {x['rel_fro_diff_weights']:.1e}" for x in w)) if w else "not available"
            dts = sorted({x["weight_dtype"] for x in w if x.get("weight_dtype")})
            md.append(f"- {SHORT[lg]}: ||W_S - W_B|| / ||W_B - W_E|| on the saved weights"
                      + (f" ({', '.join(d.replace('torch.', '') for d in dts)})" if dts else "") + f": {wtxt}; over {len(rs)} configurations the largest "
                      f"differences are {max(abs(x['d_reported_resolved']) for x in rs)} reported complaints, {max(abs(x['d_heldout_D1']) for x in rs)} "
                      f"held-out rows, {max(x['d_edit_checks'] for x in rs):.0f} edit checks. Per configuration, REEDIT-S minus REEDIT-B "
                      f"(reported complaints resolved / HELD-OUT rows with D1 / edit checks differing): "
                      + "; ".join(f"n={x['n']} s{x['seed']}: {x['d_reported_resolved']:+d} / {x['d_heldout_D1']:+d} / {x['d_edit_checks']:.0f}" for x in rs) + "."
                      + "".join(f" Entry-wise, n={x['n']}: the differing entries differ by a median of {x['ulp']['median']:.0f} step(s) of the weight format "
                                f"({100 * x['ulp']['share_within_2']:.0f}% within 2 steps, {100 * x['ulp']['share_above_16']:.1f}% by more than 16, largest "
                                f"{x['ulp']['max']:.0f})." for x in w if x.get("ulp")))
        allr = [x for lg in logs for x in R["logs"][lg].get("reedit_seq_vs_batch", []) if not x["alias"]]
        mxr = max((abs(x["d_heldout_D1"]) for x in allr), default=0)
        md += ["  A sequential re-edit that differs from the joint one needs targets re-optimised on the current weights (REEDIT-SC, exploratory, "
               "section 8)."
               + ("" if mxr <= 2 else f" The identity is one of exact arithmetic: on the saved weights the two differ by up to {mxr} HELD-OUT rows "
                  "(listed per configuration above), so REEDIT-S and REEDIT-B are kept as separate rows in every table."), ""]
    # ---- z* fallback
    if any(R["logs"][lg]["superseded_zfallback"] for lg in logs):
        md += ["**Anomaly 2 -- the harness skips its documented z* oracle fallback unless `chunk` is given.** In `07_repair_v3/harness/methods.py` "
               "(modified 2026-11-09 18:47 for RQ4) the fallback ladder of `m_reedit` (fixed loop at 2x steps, then 200 steps / early_stop 0.01) sits "
               "inside the `chunk is not None` branch, so `repair(\"REEDIT\", ...)` with the default `chunk=None` makes one vendor attempt and silently "
               "does not install a complaint whose z* is not an oracle (the README documents the ladder for every call). The harness was not modified "
               "(read-only); here every z* is computed with `chunk=1` (same vendor attempt for one item, then the ladder). The first-pass records "
               "made before this was noticed are kept in `out/<log>/superseded_zfallback/`; first pass -> re-run:"]
        for lg in logs:
            L = R["logs"][lg]
            for s in L["superseded_zfallback"]:
                r = L["rows"].get(s["config"])
                fp = s["first_pass"]
                if r is None:
                    md.append(f"- {SHORT[lg]} `{s['config']}`: first pass resolved {fp['reported_resolved']}/{fp['n']} (not installed: {len(fp['not_installed'])}), "
                              f"HELD-OUT D1 {fp['heldout_D1_rate']:.1f}%, Rel {fp['rel_exact']:.0f}; re-run not finished.")
                else:
                    md.append(f"- {SHORT[lg]} `{s['config']}`: reported resolved {fp['reported_resolved']}/{fp['n']} -> {r['reported']['resolved']}/{r['reported']['n']} "
                              f"(not installed {len(fp['not_installed'])} -> {len(r['reedit']['not_installed'] or [])}), HELD-OUT D1 {fp['heldout_D1_rate']:.1f}% -> "
                              f"{r['heldout']['D1_rate']:.1f}%, Rel exact {fp['rel_exact']:.0f} -> {r['edits']['rel']['exact']:.0f}, norm {fp['fro_rel_to_log']:.3f} -> "
                              f"{r['fro_rel_to_log']:.3f}.")
        md += ["  The single-complaint REEDIT rows of `07_repair_v3/rq1` written after that harness change use the default call and are exposed to the same "
               "omission (see section 9 for the text complaints).", ""]
    # ---- unresolved although restored
    worst = max((r["comp"]["restore_rel_err_max"] for lg in logs for r in R["logs"][lg]["rows"].values()
                 if r["method"] in ("COMP_B", "COMP_BP", "COMP_S", "COMP_SP") and r.get("comp") and r["comp"]["restore_rel_err_max"] is not None), default=None)
    ap_tot = {"a": [0, 0], "b": [0, 0]}
    for lg in logs:
        for r in R["logs"][lg]["rows"].values():
            if r["method"] != "COMP_B_ALLPOS":
                continue
            o = R["logs"][lg]["rows"].get(r["config"].replace("COMP_B_ALLPOS__", "COMP_B__"))
            if o is None:
                continue
            ap_tot["a"][0] += r["reported"]["resolved"]; ap_tot["a"][1] += r["reported"]["n"]      # noqa: E702
            ap_tot["b"][0] += o["reported"]["resolved"]; ap_tot["b"][1] += o["reported"]["n"]      # noqa: E702
    allpos_txt = (f"{ap_tot['a'][0]}/{ap_tot['a'][1]} reported complaints against {ap_tot['b'][0]}/{ap_tot['b'][1]} for COMP-B on the same sets "
                  f"(both logs, all seeds available)") if ap_tot["a"][1] else "(no COMP-B-ALLPOS record)"
    md += ["**Anomaly 3 -- compensation at the answer positions restores the module output and still leaves reported complaints unresolved.** "
           f"W_0's module output is restored at every reported answer key in every configuration (largest relative restoration error over all "
           f"answer-position COMP configurations: {'-' if worst is None else format(worst, '.0e')}), "
           "but only the answer positions are restored; the prompt positions of the same row keep the edited module output. Restoring every "
           "position of the same complaints (COMP-B-ALLPOS, n <= 10) resolves " + allpos_txt + "."]
    for lg in logs:
        rows = R["logs"][lg]["rows"]
        seq = []
        for m in ("COMP_B", "COMP_BP"):
            for n in TAB_NS:
                r = rows.get(cfg(m, n))
                if r is None or "comp" not in r:
                    continue
                c = r["comp"]
                seq.append(f"{LAB[m]} n={n}: {c['unresolved']}/{r['reported']['n']} unresolved, {c['unresolved_with_output_restored']} of them with the "
                           f"module output restored (max restoration error {c['restore_rel_err_max']:.0e}, {c['n_dropped']} directions dropped)")
        if seq:
            md.append(f"- {SHORT[lg]}: " + "; ".join(seq) + ".")
    md.append("")
    # ---- ALLPOS
    md += ["**Anomaly 4 -- restoring ALL positions (COMP-B-ALLPOS).** Feasible = W_0's module output restored at every position of every reported "
           "complaint (restoration error < 1e-6); dropped = query directions below the rank cut (keys inside the protected span or linearly dependent)."]
    for lg in logs:
        L = R["logs"][lg]
        seq = []
        for n in ("1", "2", "5", "10"):
            rs = sorted([r for r in L["rows"].values() if r["method"] == "COMP_B_ALLPOS" and str(r["n"]) == n], key=lambda r: r["seed"])
            for r in rs:
                c = r["comp"]
                seq.append(f"n={n} s{r['seed']}: {c['n_query_keys']} keys, rank {c['rank']} ({c['n_dropped']} dropped), "
                           f"{'feasible' if c['feasible'] else 'NOT feasible'} (err {c['restore_rel_err_max']:.0e}), resolved {r['reported']['resolved']}/{r['reported']['n']}, "
                           f"held-out +{r['heldout']['complaints_fixed']} / -{r['heldout']['noncomplaints_lost']}, norm {r['fro_rel_to_log']:.2f}")
        if seq:
            md.append(f"- {SHORT[lg]} (d_in = {L['setup']['d_in']}): " + "; ".join(seq) + ".")
        # reported complaints that stay unresolved although the module output is restored at EVERY position of the row
        und = {}
        for r in sorted([r for r in L["rows"].values() if r["method"] == "COMP_B_ALLPOS"], key=lambda r: (r["seed"], r["n"])):
            for u in (r["comp"].get("unresolved_detail") or []):
                if u["restore_rel_err"] < 1e-6:
                    und.setdefault(u["group"], []).append((r, u))
        for g, lst in und.items():
            w0 = next((c for c in (L.get("w0_complaint_flags") or {}).get(g, [])), None)
            md.append(f"  {SHORT[lg]}: `{g}` stays unresolved in {len(lst)} all-position configuration(s) ("
                      + ", ".join(f"n={r['n']} s{r['seed']}" for r, u in lst) + f") although the module output is restored at each of its {lst[0][1]['n_keys']} "
                      f"positions (largest restoration error {max(u['restore_rel_err'] for r, u in lst):.0e}): token accuracy "
                      f"{min(u['tok_acc'] for r, u in lst):.2f}-{max(u['tok_acc'] for r, u in lst):.2f} of {lst[0][1]['n_tok']} answer tokens, first mismatch at answer token "
                      + ", ".join(sorted({str(u['first_mismatch']) for r, u in lst}))
                      + ("" if w0 is None else f"; at W_0 itself the row is {'reproduced' if w0 else 'NOT reproduced'}")
                      + ". Not explained. One candidate, not tested: the adapter adds the fp64 correction to the module's half-precision output and "
                        "rounds, so the repaired forward equals W_0's forward only up to that rounding.")
    md.append("")
    # ---- rollback union
    md += ["**Anomaly 5 -- what RB-UNION and RB-TOT roll back at each n (edits removed, Rel exact, HELD-OUT D1 rate, reported complaints resolved):**"]
    for lg in logs:
        L = R["logs"][lg]
        seq = []
        for m in ("RB_UNION", "RB_TOT"):
            for n in TAB_NS:
                r = L["rows"].get(cfg(m, n))
                if r is not None and "rollback" in r:
                    seq.append(f"{LAB[m]} n={n}: {r['rollback']['k']} of {L['setup']['T']} edits rolled back, Rel exact {r['edits']['rel']['exact']:.0f}, "
                               f"HELD-OUT D1 {r['heldout']['D1_rate']:.1f}%, reported resolved {r['reported']['resolved']}/{r['reported']['n']}")
        if seq:
            md.append(f"- {SHORT[lg]}: " + "; ".join(seq) + ".")
    md.append("")
    # ---- not finished
    unf = []
    for lg in logs:
        L = R["logs"][lg]
        if L["failed"]:
            unf.append(f"- {SHORT[lg]}: failed configurations: " + ", ".join(f"`{x['config']}` ({x['error']})" for x in L["failed"]) + ".")
        if L.get("oom_unfinished"):
            unf.append(f"- {SHORT[lg]}: not finished (CUDA OOM on the shared card / under the process cap; not a result): "
                       + ", ".join(f"`{x['config']}`" for x in L["oom_unfinished"]) + ".")
        if L.get("missing"):
            unf.append(f"- {SHORT[lg]}: {len(L['missing'])} protocol configurations not run: " + ", ".join(f"`{x}`" for x in L["missing"][:40])
                       + (" ..." if len(L["missing"]) > 40 else "") + ".")
    for lg in logs:
        L = R["logs"][lg]
        nd2 = [c for c in ("NONE", "ALL") + tuple(cfg(m, n) for m in ("COMP_B", "COMP_BP", "REEDIT_B", "RB_TOT", "RB_UNION") for n in D2_NS)
               if c not in L["d2"]]
        if nd2:
            unf.append(f"- {SHORT[lg]}: D2 decodes not run for {len(nd2)} of the {3 * 5 + 2} core configurations"
                       + (" (none was run)" if not L["d2"] else ": " + ", ".join(f"`{x}`" for x in nd2)) + ".")
        if L.get("missing_extra"):
            unf.append(f"- {SHORT[lg]}: exploratory configurations outside the protocol, planned and not run: "
                       + ", ".join(f"`{x}`" for x in L["missing_extra"]) + ".")
    TT = (R.get("text_h2h") or {})
    if not TT.get("text"):
        unf.append("- Second priority (text-complaint head-to-head): not run on either log.")
    else:
        npl = TT["protocol"].get("n_planned")
        for lg in logs:
            src = TT["text"].get(lg)
            if src is None:
                unf.append(f"- {SHORT[lg]}: text-complaint head-to-head not run (0 of the {npl} planned complaints).")
            elif npl is not None and src["n_complaints_complete"] < npl:
                unf.append(f"- {SHORT[lg]}: text-complaint head-to-head incomplete: {src['n_complaints_complete']} of the {npl} planned complaints done.")
    md += ["**Not finished.**"] + (unf if unf else ["- every protocol configuration of both logs has a record."]) + [""]
    _ = P
    return md


# =========================================================================== sections 2..
def main():
    R = json.load(open(os.path.join(P12, "out", "results.json")))
    logs = [lg for lg in LOGS if lg in R["logs"]]
    P = R["protocol"]
    md = ["# P12 -- linear repair streams", "",
          "Rendered by `code/render.py` from `out/results.json` (`code/aggregate.py` over `out/<log>/*.json`, written by `code/run_streams.py`, "
          "`code/run_d2.py`, `code/run_text_h2h.py` through the shared harness `07_repair_v3/harness`). Do not edit by hand. "
          "Figures: `figures/fig_heldout_d1.png`, `fig_rel_exact.png`, `fig_edits_broken.png`, `fig_tradeoff.png`.", "",
          "## 1. Findings and anomalies", ""]
    md += findings(R)
    # ---------------------------------------------------------------- protocol
    md += ["## 2. Protocol", "",
           "- Logs: " + "; ".join(f"{SHORT[lg]}: {R['logs'][lg]['setup']['T']} edits, layer {R['logs'][lg]['setup']['layer']}, l2 = {R['logs'][lg]['setup']['l2']:g}, "
                                   f"REPAIR pool {R['logs'][lg]['setup']['n_repair_rows']} valid rows = {R['logs'][lg]['setup']['n_repair_complaints']} D1 complaints + "
                                   f"{R['logs'][lg]['setup']['n_repair_noncomplaints']} non-complaints, HELD-OUT pool {R['logs'][lg]['setup']['n_heldout_rows']} valid rows"
                                   for lg in logs) + ".",
           f"- Reported sets: one nested stream per (log, seed): harness `sample_complaints(H, {P['n_base']}, seed)` (D1 v2, REPAIR pool, stratified), "
           "interleaved text / image in draw order, then the remaining REPAIR-pool complaints in a seeded random order; the reported set of size n is the "
           f"first n of the stream (n // 2 image + n - n // 2 text). n in {{{', '.join(P['n_grid'])}}}; seeds {P['seeds']} for n <= 20, seed 0 for n >= 50. "
           "Sequential methods use the stream order.",
           "- Methods (all through `harness.repair`; alpha = 1; nothing tuned; fp64 solves on the CPU for both logs):"]
    for m in ORDER:
        if m in P["options"]:
            md.append(f"  - **{LAB[m]}**: `{json.dumps(P['options'][m], sort_keys=True)}`")
    md += [f"- RB-TOT: budget {P['rb_tot_budget']}; RB-UNION: budget {P['rb_union_budget']} per complaint; REEDIT-S: batches of {P['reedit_batch']}.",
           "- Evaluation: `harness.evaluate(per_item=True)`; complaints and protection keys come from the REPAIR pool only; collateral and generalisation "
           "from the HELD-OUT pool only; y0 = decoder-v2 answers (`probe_records_v2.jsonl`).",
           f"- 95% CIs: uid-level percentile bootstrap, {P['bootstrap']} resamples. Paired test against the unrepaired log: exact McNemar on the held-out rows "
           "(descriptive; many configurations are compared).", ""]
    # ---------------------------------------------------------------- main tables
    md += ["## 3. Main table: method x n in {1, 10, 100, ALL}, seed 0", "",
           "HELD-OUT = valid-y0 probe rows of the HELD-OUT pool. D1 rate = share of these rows whose y0 is not reproduced under teacher forcing after the "
           "repair. Disjoint = the HELD-OUT rows whose query does not occur in the REPAIR pool (Anomaly 0); twins of reported complaints = HELD-OUT rows "
           "whose query is identical to one of the n reported complaints (fixed / rows). Edits: all logged edits x {Rel, T-Gen, I-Gen}, B=1 teacher-forced exact; broken = exact at W_E and not after. Norm = Frobenius norm of "
           "the correction / ||W_E - W_0||. Rank of the correction: exact for COMP (fp64 adapter), nominal for REEDIT (installed complaints) and rollbacks "
           "(edits rolled back). Passes include the z* optimisation for REEDIT. Wall-clock on shared GPUs.", ""]
    for lg in logs:
        L = R["logs"][lg]
        md += [f"### {SHORT[lg]}", "", HEAD, SEP]
        if "NONE" in L["rows"]:
            md.append(row_line(L["rows"]["NONE"], "-"))
        anyfp = False
        for m in MAIN:
            for n in TAB_NS:
                r, mk = getrow(L, cfg(m, n))
                if r is not None:
                    md.append(row_line(r, nlab(n, L), mk))
                    anyfp = anyfp or bool(mk)
        if "ALL" in L["rows"]:
            md.append(row_line(L["rows"]["ALL"], "-"))
        md.append("")
        if anyfp:
            md += [f"{FP} first-pass record: REEDIT as the harness's default call executes it (one z* attempt; a complaint whose z* is not an oracle is "
                   "not installed; Anomaly 2). The re-run with the documented oracle fallback did not get GPU time. Complaints not installed in these rows: "
                   + "; ".join(f"{LAB[r['method']]} n={r['n']}: {len(r['reedit']['not_installed'] or [])}" for r in L.get("rows_first_pass", {}).values()
                               if str(r["n"]) in TAB_NS and r["seed"] == 0) + ".", ""]
    # ---------------------------------------------------------------- curves
    md += ["## 4. Curves over n", "",
           "Cell = mean over the seeds run, with (min-max) where more than one seed was run. Protocol: seeds 0, 1, 2 for n <= 20, seed 0 for n >= 50. "
           + (f"{FP} = first-pass REEDIT record (see the note under the main table). " if any(R["logs"][lg].get("rows_first_pass") for lg in logs) else "")
           + "Seeds run per cell:", ""]
    for lg in logs:
        L = R["logs"][lg]
        md += [f"{SHORT[lg]}", "", "| method | " + " | ".join(f"n={nlab(n, L)}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for m in ORDER:
            cn = L["curves"].get(m)
            if cn is None:
                continue
            md.append(f"| {LAB[m]} | " + " | ".join("-" if n not in cn else ",".join(str(x) for x in cn[n]["seeds"]) + (FP if cn[n].get("first_pass") else "")
                                                    for n in NS) + " |")
        md.append("")
    for key, title, d in (("heldout_D1_rate", "HELD-OUT D1 rate (%), full pool (protocol)", 1),
                          ("heldout_disjoint_D1_rate", "HELD-OUT D1 rate (%), disjoint rows (no query shared with the REPAIR pool)", 1),
                          ("heldout_complaints_fixed", "held-out complaints fixed (full pool)", 1),
                          ("heldout_noncomplaints_lost", "held-out non-complaints lost (full pool)", 1),
                          ("heldout_disjoint_fixed", "held-out complaints fixed (disjoint rows)", 1),
                          ("heldout_disjoint_lost", "held-out non-complaints lost (disjoint rows)", 1),
                          ("twins_of_reported", "held-out twins of reported complaints (rows)", 1),
                          ("twins_of_reported_fixed", "held-out twins of reported complaints fixed", 1),
                          ("reported_resolved_frac", "reported complaints resolved (%)", 0),
                          ("rel_exact", "Rel exact (%)", 1), ("tgen_exact", "T-Gen exact (%)", 1), ("igen_exact", "I-Gen exact (%)", 1),
                          ("edits_broken_total", "edit checks broken (of 300)", 1), ("edits_fixed_total", "edit checks fixed", 1),
                          ("tloc", "T-Loc (held-out items)", 2), ("iloc", "I-Loc (held-out items)", 2),
                          ("fro_rel_to_log", "norm of the correction / norm of the log", 3), ("rank_correction", "rank of the correction", 0),
                          ("n_forward", "forward passes of the repair", 0), ("n_backward", "backward passes of the repair", 0),
                          ("sec", "repair wall-clock (s; shared GPUs)", 1)):
        md += [f"### {title}", ""]
        for lg in logs:
            L = R["logs"][lg]
            none = L["rows"].get("NONE")
            ref = ""
            if none is not None:
                refv = {"heldout_D1_rate": none["heldout"]["D1_rate"], "heldout_disjoint_D1_rate": none["heldout_disjoint"]["D1_rate"],
                        "rel_exact": none["edits"]["rel"]["exact"],
                        "tgen_exact": none["edits"]["tgen"]["exact"], "igen_exact": none["edits"]["igen"]["exact"],
                        "tloc": none["loc"]["tloc"]["after"], "iloc": none["loc"]["iloc"]["after"]}.get(key)
                if refv is not None:
                    ref = f" -- unrepaired: {refv:.{d}f}"
            md += [f"{SHORT[lg]}{ref}", "", "| method | " + " | ".join(f"n={nlab(n, L)}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
            for m in ORDER:
                cn = L["curves"].get(m)
                if cn is None:
                    continue
                md.append(f"| {LAB[m]} | " + " | ".join(cell(cn.get(n), key, d) for n in NS) + " |")
            md.append("")
    # ---------------------------------------------------------------- by ptype
    md += ["## 5. HELD-OUT rows by probe type (seed 0; D1 rate % after the repair; +fixed / -lost)", ""]
    for lg in logs:
        L = R["logs"][lg]
        nn = L["rows"].get("NONE")
        if nn is None:
            continue
        b = nn["heldout"]["by_ptype"]
        md += [f"{SHORT[lg]} -- unrepaired: image {b['image']['D1_rate']:.1f}% ({b['image']['n_complaints']} complaints / {b['image']['n_noncomplaints']} "
               f"non-complaints), text {b['text']['D1_rate']:.1f}% ({b['text']['n_complaints']} / {b['text']['n_noncomplaints']})", "",
               "| method | n | image D1 % | image +fixed / -lost | text D1 % | text +fixed / -lost | reported resolved image | reported resolved text |", "|---|---|---|---|---|---|---|---|"]
        for m in MAIN:
            for n in TAB_NS:
                r = L["rows"].get(cfg(m, n))
                if r is None:
                    continue
                p, q = r["heldout"]["by_ptype"], r["reported"]["by_ptype"]
                md.append(f"| {LAB[m]} | {nlab(n, L)} | {p['image']['D1_rate']:.1f} | +{p['image']['complaints_fixed']} / -{p['image']['noncomplaints_lost']} | "
                          f"{p['text']['D1_rate']:.1f} | +{p['text']['complaints_fixed']} / -{p['text']['noncomplaints_lost']} | "
                          f"{q['image']['resolved']}/{q['image']['n']} | {q['text']['resolved']}/{q['text']['n']} |")
        md.append("")
    # ---------------------------------------------------------------- D2
    md += ["## 6. D2 complaints among the HELD-OUT rows (seed 0, n in {10, 100, ALL})", "",
           "Testbed D2 pipeline (decoder v2: 10 no-cache greedy forwards + KV-cache greedy continuation to 48 tokens; `tb_common.contains_ref`): a HELD-OUT "
           "row is a D2 complaint if its pre-edit 48-token answer contains the reference (stored `pre_correct_long`) and the answer of the repaired model "
           "does not. Rows = every HELD-OUT row with `pre_correct_long`. rho = module-output restoration at the answer keys of the valid HELD-OUT rows "
           "(1 = unrepaired, 0 = pre-edit; medians).", ""]
    for lg in logs:
        D = R["logs"][lg]["d2"]
        L = R["logs"][lg]
        if not D:
            md += [f"{SHORT[lg]}: no D2 record.", ""]
            continue
        md += [f"### {SHORT[lg]}", "",
               "| method | n | pre-correct rows | D2 after (rate %) | D2 at W_E (stored) | stored D2 resolved | new D2 | disjoint rows: D2 after / n (W_E; resolved / new) | "
               "image D2 / n | text D2 / n | answers == stored post-edit | "
               "== stored pre-edit | rho complaints (q25-q75) | rho non-complaints | rho fixed / not fixed complaints | rho lost / kept non-complaints | "
               "REPAIR known-good rows kept (in-sample) |",
               "|" + "---|" * 17]

        def d2_line(name, lab_m, lab_n):
            d = D.get(name)
            if d is None:
                return None
            bp = d["by_ptype"]
            mr = (L.get("rho") or {}).get(name)
            if mr is None:
                rr = "- | - | - | -"
            else:
                rr = (f"{mr['median_complaints']:.2f} ({mr['q25_complaints']:.2f}-{mr['q75_complaints']:.2f}) | {mr['median_noncomplaints']:.2f} | "
                      f"{f(mr.get('median_complaints_fixed'), 2)} / {f(mr.get('median_complaints_not_fixed'), 2)} | "
                      f"{f(mr.get('median_noncomplaints_lost'), 2)} / {f(mr.get('median_noncomplaints_kept'), 2)}")
            dj = d["disjoint"]
            return (f"| {lab_m} | {lab_n} | {d['n_pre_correct']} | {d['n_D2']} ({d['D2_rate']:.1f}) | {d['n_D2_WE']} | {d['stored_D2_resolved']} | {d['new_D2']} | "
                    f"{dj['n_D2']}/{dj['n_pre_correct']} ({dj['n_D2_WE']}; {dj['stored_D2_resolved']} / {dj['new_D2']}) | "
                    f"{bp['image']['n_D2']}/{bp['image']['n_pre_correct']} | {bp['text']['n_D2']}/{bp['text']['n_pre_correct']} | {d['long_eq_stored_post']} | "
                    f"{d['long_eq_stored_pre']} | {rr} | "
                    + ("-" if not d.get("regression_suite_in_sample") else f"{d['regression_suite_in_sample']['kept']}/{d['regression_suite_in_sample']['n']}") + " |")
        x = d2_line("NONE", LAB["NONE"], "-")
        if x:
            md.append(x)
        for m in MAIN:
            for n in D2_NS:
                x = d2_line(cfg(m, n), LAB[m], nlab(n, L))
                if x:
                    md.append(x)
        x = d2_line("ALL", LAB["ALL"], "-")
        if x:
            md.append(x)
        md.append("")
    # ---------------------------------------------------------------- operating points
    md += [f"## 7. Operating points with Rel exact within {P['rel_tol']:g} points of the unrepaired log", "",
           "Single configurations (lowest HELD-OUT D1 rate first; p = exact McNemar against the unrepaired log on the held-out rows, uncorrected):", ""]
    for lg in logs:
        L = R["logs"][lg]
        md += [f"{SHORT[lg]} -- {L['n_operating_candidates']} configurations inside the tolerance", "",
               "| config | HELD-OUT D1 % [CI] | delta vs unrepaired | p | disjoint D1 % [CI] | delta (disjoint) | p (disjoint) | Rel exact |",
               "|---|---|---|---|---|---|---|---|"]
        for o in L["operating_points"][:10]:
            md.append(f"| {LAB[o['method']]} n={o['n']} s{o['seed']} | {o['heldout_D1_rate']:.1f} [{o['ci'][0]:.1f}, {o['ci'][1]:.1f}] | {o['delta_vs_none']:+.1f} | "
                      f"{pv(o['mcnemar_p'])} | {o['disjoint_D1_rate']:.1f} [{o['disjoint_ci'][0]:.1f}, {o['disjoint_ci'][1]:.1f}] | {o['disjoint_delta_vs_none']:+.1f} | "
                      f"{pv(o['disjoint_mcnemar_p'])} | {o['rel_exact']:.0f} |")
        md += ["", "Cells (method x n) whose every seed is inside the tolerance, by seed-mean HELD-OUT D1 rate:", "",
               "| method | n | seeds | HELD-OUT D1 % (mean) | disjoint D1 % (mean) | Rel exact (mean) |", "|---|---|---|---|---|---|"]
        for o in L["operating_cells"][:10]:
            md.append(f"| {LAB[o['method']]} | {o['n']} | {o['n_seeds']} | {o['heldout_D1_rate_mean']:.1f} | {o['disjoint_D1_rate_mean']:.1f} | {o['rel_exact_mean']:.1f} |")
        md.append("")
    # ---------------------------------------------------------------- method diagnostics
    md += ["## 8. Method diagnostics (seed 0)", "", "### COMP: rank budget, restoration, unresolved reported complaints", ""]
    for lg in logs:
        L = R["logs"][lg]
        md += [f"{SHORT[lg]} (d_in = {L['setup']['d_in']})", "",
               "| method | n | query keys | rank of the correction | dropped | protection rank | (rank + protection) / d_in | e (key share outside the protected span) | "
               "condition number | max restoration error | "
               "unresolved reported | ... with the module output restored | max ||X_seq - X_joint|| / ||X_joint|| |", "|" + "---|" * 13]
        for m in ("COMP_B", "COMP_BP", "COMP_S", "COMP_SP", "COMP_B_ALLPOS"):
            for n in NS:
                r = L["rows"].get(cfg(m, n))
                if r is None or "comp" not in r:
                    continue
                c = r["comp"]
                iv = c.get("identity_vs_batch")
                md.append(f"| {LAB[m]} | {nlab(n, L)} | {c['n_query_keys']} | {c['rank']} | {c['n_dropped']} | {c['protect_rank']} | {c['rank_used_frac']:.2f} | "
                          f"{f(c.get('e_mean'), 2)} | {'-' if not isinstance(c.get('cond'), (int, float)) else format(c['cond'], '.1e')} | "
                          f"{c['restore_rel_err_max']:.1e} | {c['unresolved']}/{r['reported']['n']} | {c['unresolved_with_output_restored']} | "
                          f"{'-' if iv is None else format(iv['rel_fro_diff'], '.1e')} |")
        md.append("")
    md += ["### REEDIT: installed complaints and solve attainment", ""]
    for lg in logs:
        L = R["logs"][lg]
        md += [f"{SHORT[lg]}", "", "| method | n | installed | not installed (no oracle z*) | installed but not resolved | solve attainment (joint, or first-last batch) | "
               "||D|| / ||W_0|| (joint solve) | norm / log |", "|" + "---|" * 8]
        for m in ("REEDIT_B", "REEDIT_S", "REEDIT_SC"):
            for n in NS:
                r, mk = getrow(L, cfg(m, n))
                if r is None or "reedit" not in r:
                    continue
                c = r["reedit"]
                apb = [x for x in c.get("attainment_per_batch") or [] if x is not None]
                att = f(c.get("solve_attainment"), 3) if not apb else f"{apb[0]:.3f} - {apb[-1]:.3f} ({len(apb)} batches)"
                md.append(f"| {LAB[m]}{mk} | {nlab(n, L)} | {c['installed']} | {len(c['not_installed'] or [])} | {c['installed_not_resolved']} | {att} | "
                          f"{f(c.get('dW_over_W0'), 3)} | {r['fro_rel_to_log']:.3f} |")
        md.append("")
    md += ["### Rollback: edits rolled back", ""]
    for lg in logs:
        L = R["logs"][lg]
        md += [f"{SHORT[lg]}", "", "| method | n | edits rolled back (of " + str(L["setup"]["T"]) + ") | all reported resolved at stop | resolved individually (per-complaint sets) | "
               "reported resolved | Rel exact |", "|" + "---|" * 7]
        for m in ("RB_TOT", "RB_UNION"):
            for n in NS:
                r = L["rows"].get(cfg(m, n))
                if r is None or "rollback" not in r:
                    continue
                c = r["rollback"]
                md.append(f"| {LAB[m]} | {nlab(n, L)} | {c['k']} | {c.get('resolved_internal') if c.get('resolved_internal') is not None else '-'} | "
                          f"{c.get('n_resolved_individually') if c.get('n_resolved_individually') is not None else '-'} | "
                          f"{r['reported']['resolved']}/{r['reported']['n']} | {r['edits']['rel']['exact']:.0f} |")
        md.append("")
    # ---------------------------------------------------------------- text h2h
    md += ["## 9. Second priority: per-complaint head-to-head on TEXT complaints (rq1 continued)", ""]
    T = R.get("text_h2h")
    if not T or not T.get("text"):
        md += ["Not run (the curves of both logs come first).", ""]
    else:
        md += [f"- complaints: {T['protocol']['complaints']}", f"- methods: {T['protocol']['methods']}", f"- clean: {T['protocol']['clean']}",
               f"- CI: {T['protocol']['ci']}; paired: {T['protocol']['paired']}; reference rows: {T['protocol']['reference']}", ""]
        for lg in logs:
            for tag, src in (("TEXT complaints (this experiment)", T["text"].get(lg)), ("image complaints (rq1 rows, read-only reference)", T["rq1_image_reference"].get(lg))):
                if src is None:
                    continue
                b = src["baseline"]
                npl = T["protocol"].get("n_planned")
                part = ""
                if tag.startswith("TEXT") and npl is not None and src["n_complaints_complete"] < npl:
                    part = (f" of the {npl} planned" + (" (the first " + str(src["n_complaints_complete"]) + " in draw order: a uniform random subset; "
                                                         "the intervals below are those of this smaller sample)" if src.get("is_draw_order_prefix") else
                                                         " (NOT the first ones in draw order)"))
                md += [f"### {SHORT[lg]} -- {tag}: {src['n_complaints_complete']} complaints{part}", "",
                       f"W_E: Rel / T-Gen / I-Gen exact {b['rel_exact_WE']:.0f} / {b['tgen_exact_WE']:.0f} / {b['igen_exact_WE']:.0f}; held-out rows: "
                       f"{b['heldout_noncomplaints']} non-complaints, {b['heldout_complaints']} complaints; median answer length {src['n_ans_tok_median']:.0f} tokens.", "",
                       "| method | D1 resolved | rate [95% CI] | clean (resolved, 0 broken) [CI] | edits broken / repair [CI] | rel / T-Gen / I-Gen broken | edits fixed | "
                       "held-out non-compl. lost [CI] | held-out compl. fixed [CI] | dT-Loc | dI-Loc | median norm (/ log) | D2 res / n | passes fwd / bwd | median s |",
                       "|" + "---|" * 15]
                for c in ("NONE", "COMP", "REEDIT", "RB_FO", "RB_BEH"):
                    s = src["configs"].get(c)
                    if s is None:
                        continue
                    if c == "NONE" or "clean" not in s or s["n_full_eval"] < s["n"]:
                        md.append(f"| {c} | {s['D1_resolved']}/{s['n']} | {s['D1_rate']['mean']:.2f} [{s['D1_rate']['lo']:.2f}, {s['D1_rate']['hi']:.2f}] | "
                                  f"(full evaluation on {s['n_full_eval']} complaint) | - | - | - | - | - | - | - | - | {s['D2_pooled']['resolved']}/{s['D2_pooled']['n']} | - | - |")
                        continue

                    def ci(x, d=2):
                        return "-" if not x or x.get("mean") is None else f"{x['mean']:.{d}f} [{x['lo']:.{d}f}, {x['hi']:.{d}f}]"
                    e = s["edits"]
                    md.append(f"| {c} | {s['D1_resolved']}/{s['n']} | {ci(s['D1_rate'])} | {s['clean']} ({ci(s['clean_rate'])}) | {ci(s['edits_broken_total'])} | "
                              f"{e['rel']['broken']['mean']:.2f} / {e['tgen']['broken']['mean']:.2f} / {e['igen']['broken']['mean']:.2f} | {s['edits_fixed_total']['mean']:.2f} | "
                              f"{ci(s['collateral']['all']['noncomplaints_lost'])} | {ci(s['collateral']['all']['complaints_fixed'])} | "
                              f"{ci(s['loc']['tloc']['delta'])} | {ci(s['loc']['iloc']['delta'])} | {s['fro_median']:.2f} ({s['fro_rel_median']:.3f}) | "
                              f"{s['D2_pooled']['resolved']}/{s['D2_pooled']['n']} | {s['cost']['n_forward']['mean']:.1f} / {s['cost']['n_backward']['mean']:.1f} | "
                              f"{s['cost']['sec_median']:.2f} |")
                md.append("")
                re_ = src["configs"].get("REEDIT", {}).get("reedit")
                if re_:
                    line = (f"REEDIT: {re_['not_installed']} complaint(s) not installed (z* not an oracle after the single vendor attempt of the default call: "
                            f"{', '.join(re_['not_installed_groups']) or 'none'}); z* attempts {re_['zopt_attempts_hist']}.")
                    if src.get("reedit_fallback_supplement"):
                        fb = src["reedit_fallback_supplement"]
                        w = src["reedit_with_fallback_resolved"]
                        line += (f" Supplement with the documented fallback ladder (`chunk=1`) on these complaints: "
                                 + "; ".join(f"{x['group']}: installed {x['installed']}, resolved {x['resolved']}, attempts {x['zopt_attempts']}, edits broken {x['edits_broken']}" for x in fb)
                                 + f" -> REEDIT with the documented fallback would resolve {w['resolved']}/{w['n']}.")
                    md += [line, ""]
                md += ["| comparison | outcome | n | both | first only | second only | neither | p |", "|---|---|---|---|---|---|---|---|"]
                for k, v in src["paired"].items():
                    a, bb, oc = k.rsplit("_", 1)[0].split("_vs_") + [k.rsplit("_", 1)[1]]
                    md.append(f"| {a} vs {bb} | {oc} | {v['n']} | {v['both']} | {v['a_only']} | {v['b_only']} | {v['neither']} | {pv(v['p_exact'])} |")
                md.append("")
    # ---------------------------------------------------------------- run info
    md += ["## 10. Run information", ""]
    for lg in logs:
        L = R["logs"][lg]
        ri = L.get("run_info", {})
        md.append(f"- {SHORT[lg]}: {len(L['rows'])} records ({ri.get('n_alias', 0)} copies of an identical computation: REEDIT-S with one batch, COMP-S / COMP-S+ "
                  f"at n = 1); peak GPU memory allocated {f(ri.get('peak_gpu_gib_max'), 2)} GiB; other compute processes on the card during the runs: "
                  f"median {f(ri.get('gpu_procs_median'), 0)}, max {f(ri.get('gpu_procs_max'), 0)}; total repair wall-clock {f(ri.get('repair_wall_min'), 0)} min, "
                  f"evaluation {f(ri.get('eval_wall_min'), 0)} min.")
    md += ["- Wall-clock numbers come from shared GPUs and CPU solves under load: use pass counts and ranks for cost claims.", ""]
    if R.get("resume") or any(R["logs"][lg].get("run_info", {}).get("resume") for lg in logs):
        md += ["### Resume run (operational log `out/_resume.json`; record counts from the raw records)", ""]
        for x in (R.get("resume") or {}).get("notes", []):
            md.append(f"- {x}")
        for lg in logs:
            rr = R["logs"][lg].get("run_info", {}).get("resume")
            if rr:
                md.append(f"- {SHORT[lg]}: {rr['n_records']} stream records written by the resume ({rr['first']} to {rr['last']} UTC), torch CPU threads "
                          f"{', '.join(str(t) for t in rr['torch_threads'])} (the records of the first run do not carry the field; its setting was 8); "
                          f"peak GPU memory allocated {f(rr.get('peak_gpu_gib_max'), 2)} GiB; other compute processes on the card: max "
                          f"{f(rr.get('gpu_procs_max'), 0)}; repair wall-clock {f(rr.get('repair_wall_min'), 0)} min, evaluation {f(rr.get('eval_wall_min'), 0)} min.")
        md.append("")
    if R.get("incidents"):
        md += ["### Shared-GPU incidents caused by this experiment (operational log `out/_incidents.json`; not experimental results)", ""]
        for x in R["incidents"]:
            md.append(f"- **{x['when']}, {x['card']}.** {x['what']} Impact on other agents: {x['impact_on_others']}. Cause: {x['cause']} "
                      f"Fix: {x['fix']}")
        md.append("")
    with open(os.path.join(P12, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("results.md written:", len(md), "lines")


if __name__ == "__main__":
    main()
