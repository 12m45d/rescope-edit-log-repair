"""E2 tables, H1 read-out and paper numbers (called by render.py; every number comes from the records / cells)."""
from __future__ import annotations

import os
import time

import numpy as np

import e2c
from e2c import fx

M, ST = fx.metrics, fx.stats
NL = ("1", "2", "5", "10", "20", "50", "100", "ALL")


def write(S, recs, floor, cgate, wgate, missing, logs):
    idx = {}
    for r in recs:
        idx[(r["log"], r.get("part"), r["method"], r["variant"], r.get("n_label"), r["draw"], r["pool"], r["metric"])] = r

    def rec(log, part, method, variant, nl, metric, pool, draw=None):
        return idx.get((log, part, method, variant, str(nl), draw, pool, metric))

    def val(*a, **k):
        r = rec(*a, **k)
        return None if r is None else r["value"]

    def f(r, nd=1, ci=True, nd_n=False):
        if r is None or r["value"] is None:
            return "-"
        s = f"{r['value']:.{nd}f}"
        if ci and r.get("ci_lo") is not None:
            s += f" [{r['ci_lo']:.{nd}f}, {r['ci_hi']:.{nd}f}]"
        if nd_n and r.get("n_draws"):
            s += f" ({r['n_draws']})"
        return s

    def mean(log, part, method, variant, nl, metric, pool, nd=1, ci=True, nd_n=False):
        return f(rec(log, part, method, variant, nl, metric + "@mean", pool), nd, ci, nd_n)

    def spread(log, part, method, variant, nl, metric, pool, nd=1):
        r = rec(log, part, method, variant, nl, metric + "@mean", pool)
        if r is None or not r.get("per_draw") or len(r["per_draw"]) < 2:
            return ""
        return "{" + ", ".join(f"{v:.{nd}f}" for v in r["per_draw"]) + "}"

    T = fx.io.text_table
    out = [f"E2 EVIDENCE CURVES (parts a - c, c'), LINEAR ROWS (d), H1      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/render.py + tables.py from out/cells, out/rows, out/parts",
           "Development logs only.  Locality pool: heldout_disjoint (valid rows; heldout_strict as a second column).  D1 = % of rows whose cached decoder-v2 pre-edit answer is",
           "not reproduced under teacher forcing at B = 1.  Intervals: percentile bootstrap, 2000 resamples (probe unit = query cluster, edit unit = edit); over seeds / calibration",
           "draws: crossed bootstrap (draws x units); '(k)' = number of seeds / draws behind a mean; '{..}' = the single draws.  Near-scope firing (hard negatives HN-I / HN-Q / HN-X)",
           "is printed next to every far rate; for a linear repair it is not defined (the weight acts at every key: '-').  BEN-T / MME-T: false firing at gate level through",
           "fx.pools.firing (PROMPT-ONLY keys; gates calibrated at teacher-forced REPAIR keys); not defined for all-positions gates (ReScope-W) and for linear repairs.",
           "The retrieval family / lookup and the _x methods are not part of E2.  No behavioural evaluation on benign-pool or MME rows (R19).",
           f"NOT FINISHED / MISSING at render time: {missing if missing else 'nothing'}", ""]
    h1 = {}
    paper, doc = {}, {}

    def pn(name, value, text):
        paper[name], doc[name] = value, text

    for name in e2c.LOGS:
        L, sh = logs[name], e2c.SHORT[name]
        nall_c = len(fx.draws.complaints(L, "ALL", 0))
        out += ["#" * 150, f"{name}   (T = {L.T}; REPAIR pool {len(L.pool('repair'))} valid rows, {nall_c} reported D1 complaints = 'ALL'; heldout_disjoint {len(L.pool('heldout_disjoint'))} valid rows, "
                f"heldout_strict {len(L.pool('heldout_strict'))})", "#" * 150]
        dep = lambda metric, pool: rec(name, "base", "deployed", "", "0", metric, pool)                     # noqa: E731
        ds = L.describe()
        out += [f"rows: {ds['n_probe_rows']} probe rows, {ds['n_probe_valid']} with a valid pre-edit answer (the others are excluded); hard negatives valid / invalid: "
                + ", ".join(f"{t} {ds['hard_negatives'].get(t)} / {ds['hard_negatives'].get(t + ' (invalid y0)')}" for t in M.HN_TYPES)
                + "; degenerate pre-edit answers (empty, one character, '?', bare None / No) among the valid rows: heldout_disjoint "
                + f(rec(name, "base", "deployed", "", "0", "probe.n_y0_degenerate", "heldout_disjoint"), 0) + ", "
                + ", ".join(f"{t} {f(rec(name, 'base', 'deployed', '', '0', f'hn.{t}.n_y0_degenerate', 'hard_negatives'), 0)}" for t in M.HN_TYPES)
                + " (records: probe.D1_y0_not_degenerate, hn.<kind>.D1_y0_not_degenerate)",
                "every behavioural cell also holds the read-outs of R7 in results.json: kg.<rel|tgen|igen>.<kept|gained|lost> (% of all edits, against the deployed log), d2.D2", ""]
        sc = sanity_compare(S, name)
        if sc is not None:
            out += [f"agreement with the sanity run e0 (same cells: {', '.join(sc['cells'])}): {sc['n_numbers']} numbers compared, largest absolute difference "
                    f"{sc['largest']:.4f} points (" + ", ".join(f"{k}: {x:.4f}" for k, x in sc["max_abs_diff"].items()) + ")", ""]
            pn(f"Etwo{sh}SanityMaxDiff", sc["largest"], f"{name}: largest absolute difference (points) between this experiment and the sanity run e0 on the {sc['n_numbers']} shared numbers")
        if dep("probe.D1", "heldout_disjoint") is not None:
            fl = floor.get(name)
            out += ["(0) REFERENCE ROWS (driver, B = 1)",
                    T(["row", "D1 % disjoint", "D1 % strict", "Rel", "T-Gen", "I-Gen", "D2 % (n pre-correct)", "HN-I / HN-Q / HN-X D1 %"],
                      [[lab, f(rec(name, "base", m, v, "0", "probe.D1", "heldout_disjoint")), f(rec(name, "base", m, v, "0", "probe.D1", "heldout_strict")),
                        f(rec(name, "base", m, v, "0", "edit.rel.exact", "edits"), 0, False), f(rec(name, "base", m, v, "0", "edit.tgen.exact", "edits"), 0, False),
                        f(rec(name, "base", m, v, "0", "edit.igen.exact", "edits"), 0, False),
                        f(rec(name, "base", m, v, "0", "d2.D2", "heldout_disjoint")) + f" ({f(rec(name, 'base', m, v, '0', 'd2.n_pre_correct', 'heldout_disjoint'), 0)})",
                        " / ".join(f(rec(name, "base", m, v, "0", f"hn.{t}.D1", "hard_negatives"), 1, False) for t in M.HN_TYPES)]
                       for lab, m, v in (("deployed log W_E", "deployed", ""), ("pre-edit model W_0", "preedit", ""),
                                         ("ReScope-W, all thresholds -inf (round-off floor)", "where_floor", "all thresholds -inf"))]),
                    "" if fl is None else f"    round-off floor: flags of ReScope-W at thresholds -inf that differ from the deployed weight: edits {fl['edits']} of {fl['n_edits']}, "
                    f"heldout_disjoint {fl['heldout_disjoint']} of {fl['n_heldout_disjoint']}, hard negatives {fl['hard_negatives']} of {fl['n_hard_negatives']}", ""]
            pn(f"Etwo{sh}DepDone", val(name, "base", "deployed", "", "0", "probe.D1", "heldout_disjoint"), f"{name}: deployed log, D1 % on heldout_disjoint (driver, B = 1)")
            pn(f"Etwo{sh}DepRel", val(name, "base", "deployed", "", "0", "edit.rel.exact", "edits"), f"{name}: deployed log, Rel exact %")
            if fl is not None:
                pn(f"Etwo{sh}FloorEdits", fl["edits"], f"{name}: edit flags (of {fl['n_edits']}) of ReScope-W with all thresholds at -inf that differ from the deployed weight")
                pn(f"Etwo{sh}FloorHeld", fl["heldout_disjoint"], f"{name}: heldout_disjoint D1 flags (of {fl['n_heldout_disjoint']}) of ReScope-W at -inf that differ from the deployed weight")
        # ---------------- (a) linear curves
        ex = os.path.join(e2c.OUT, "parts", f"a_exists_{name}.json")
        if os.path.exists(ex):
            have = fx.io.jload(ex)["cells_that_exist"]
            out += ["(a) LINEAR REPAIRS OF n REPORTED COMPLAINTS -- imported P12 records (08_rescope/p12_linear_streams), metrics recomputed here from their per-row flags;",
                    "    cells: mean over the seeds that exist [crossed bootstrap] (number of seeds).  Cells that exist (seeds): "
                    + "; ".join(f"{m}: " + ",".join(f"{n}:{''.join(map(str, s))}" for n, s in d.items()) for m, d in have.items()),
                    "    restoration criterion (from the record options; nothing recomputed): " + "; ".join(f"{m}: {e2c.RESTORES[m]}" for m in e2c.LINEAR), ""]
            for title, metric, pool, nd, ci in (("held-out D1 % (heldout_disjoint)", "probe.D1", "heldout_disjoint", 1, True),
                                                ("held-out D1 % (heldout_strict)", "probe.D1", "heldout_strict", 1, False),
                                                ("held-out complaints fixed / non-complaints lost (heldout_disjoint; mean over seeds)", None, None, 1, False),
                                                ("Rel exact %", "edit.rel.exact", "edits", 1, False), ("T-Gen exact %", "edit.tgen.exact", "edits", 1, False),
                                                ("I-Gen exact %", "edit.igen.exact", "edits", 1, False), ("edit checks broken (of 300; mean over seeds)", "edit.checks_broken", "edits", 1, False),
                                                ("reported complaints resolved (count; mean over seeds)", "meta.n_reported_resolved", "repair", 1, False)):
                rows = []
                for m in e2c.LINEAR:
                    if metric is None:
                        rows.append([m] + [mean(name, "a", m, "p12_record", nl, "probe.complaints_fixed", "heldout_disjoint", 1, False) + " / "
                                           + mean(name, "a", m, "p12_record", nl, "probe.noncomplaints_lost", "heldout_disjoint", 1, False) for nl in NL])
                    else:
                        rows.append([m] + [mean(name, "a", m, "p12_record", nl, metric, pool, nd, ci, nd_n=ci) for nl in NL])
                if metric == "probe.D1" and pool == "heldout_disjoint":
                    rows.append(["blocklist (driver)"] + [mean(name, "bl", "blocklist", "deployed + exact-match list of the reported queries", nl, metric, pool, nd, ci, nd_n=True) for nl in NL])
                if metric in ("edit.rel.exact", "edit.tgen.exact", "edit.igen.exact"):
                    rows.append(["blocklist (driver)"] + [mean(name, "bl", "blocklist", "deployed + exact-match list of the reported queries", nl, metric, pool, nd, False) for nl in NL])
                out += [f"    {title}", "    " + T(["method \\ n"] + list(NL), rows).replace("\n", "\n    "), ""]
            r0 = rec(name, "a", "deployed", "p12_record", "0", "probe.D1", "heldout_disjoint")
            out += [f"    unrepaired log in the P12 records: D1 {f(r0)} on heldout_disjoint; near-scope firing of a linear repair: not defined (-)", ""]
            for m in e2c.LINEAR:
                for nl, w in (("10", "Ten"), ("ALL", "All")):
                    v = val(name, "a", m, "p12_record", nl, "probe.D1@mean", "heldout_disjoint")
                    if v is not None:
                        pn(f"Etwo{sh}Lin{m.title().replace('_', '')}{w}Done", v, f"{name}: {m} at n = {nl} reported complaints, D1 % on heldout_disjoint, mean over the seeds of the P12 records")
                        pn(f"Etwo{sh}Lin{m.title().replace('_', '')}{w}Rel", val(name, "a", m, "p12_record", nl, "edit.rel.exact@mean", "edits"), f"{name}: {m} at n = {nl}, Rel exact %, mean over seeds (P12 records)")
        # ---------------- (b) ReScope-W least repair
        if any(k[0] == "b" and k[1] == name for k in S.cells):
            out += ["(b) ReScope-W (all positions, logged values, linear response, no write round-off term): LEAST REPAIR of the same nested complaint sets (library least_repair,",
                    "    one complaint at a time, coordinates at every position of the teacher-forced pre-edit trace, default margins, on_conflict = 'refuse': a complaint whose",
                    "    repair would silence an own request is refused and counted).  start 'deployed' = all thresholds -inf; 'logonly' = log-only thresholds.",
                    "    fire = % of heldout_disjoint rows on which an edit is released (hook); HN fire = gate level from the captured all-position vectors;",
                    "    own out = own Rel prompts on which the own edit is not released at the last prompt position (of 100); refused / still = reported complaints refused /",
                    "    still firing after the repair; orders = arrival orders (of 20 random ones) that give bitwise the same thresholds; means over seeds, '{..}' single seeds", ""]
            for start in ("deployed", "logonly"):
                rows = []
                for nl in NL:
                    a = (name, "b", "where_least", start, nl)
                    if rec(*a, "probe.D1@mean", "heldout_disjoint") is None:
                        continue
                    rows.append([nl, mean(*a, "probe.D1", "heldout_disjoint", nd_n=True), spread(*a, "probe.D1", "heldout_disjoint"), mean(*a, "probe.D1", "heldout_strict", ci=False),
                                 mean(*a, "probe.fired", "heldout_disjoint", ci=False),
                                 " / ".join(mean(*a, f"gate.hn.{t}.fired", "hard_negatives", ci=False) for t in M.HN_TYPES),
                                 mean(*a, "edit.rel.exact", "edits", ci=False), mean(*a, "edit.tgen.exact", "edits", ci=False), mean(*a, "edit.igen.exact", "edits", ci=False),
                                 mean(*a, "kg.rel.kept", "edits", ci=False) + "/" + mean(*a, "kg.rel.gained", "edits", ci=False),
                                 mean(*a, "edit.rel.n_own_out_of_scope", "edits", ci=False), mean(*a, "meta.n_refused", "repair", ci=False),
                                 mean(*a, "meta.n_reported_still_firing", "repair", ci=False), mean(*a, "order.n_equal_thresholds", "repair", 0, ci=False),
                                 mean(*a, "probe.T-Loc*", "heldout_disjoint", ci=False) + "/" + mean(*a, "probe.I-Loc*", "heldout_disjoint", ci=False),
                                 mean(*a, "d2.D2", "heldout_disjoint", ci=False), " / ".join(mean(*a, f"hn.{t}.D1", "hard_negatives", ci=False) for t in M.HN_TYPES)])
                out += [f"    start = {start}",
                        "    " + T(["n", "D1 % disjoint (seeds)", "single seeds", "D1 strict", "fire %", "HN-I/Q/X fire %", "Rel", "T-Gen", "I-Gen", "Rel kept/gained", "own out", "refused",
                                    "still", "orders", "T-Loc*/I-Loc*", "D2 %", "HN-I/Q/X D1 %"], rows).replace("\n", "\n    "), ""]
            rows = []
            for nl in NL:
                a = (name, "bl", "blocklist", "deployed + exact-match list of the reported queries", nl)
                if rec(*a, "probe.D1@mean", "heldout_disjoint") is not None:
                    rows.append([nl, mean(*a, "probe.D1", "heldout_disjoint", nd_n=True), mean(*a, "probe.D1", "heldout_strict", ci=False), mean(*a, "probe.fired", "heldout_disjoint", 2, ci=False),
                                 mean(*a, "edit.rel.exact", "edits", ci=False), mean(*a, "edit.tgen.exact", "edits", ci=False), mean(*a, "edit.igen.exact", "edits", ci=False)])
            out += ["    BLOCKLIST line (deployed model + exact-match list of the n reported queries, answered by the pre-edit model; measured through the driver);",
                    "    'listed %' = heldout_disjoint rows whose query is on the list (0 by the construction of the pool unless a query repeats)",
                    "    " + T(["n", "D1 % disjoint (seeds)", "D1 strict", "listed %", "Rel", "T-Gen", "I-Gen"], rows).replace("\n", "\n    "), ""]
            for start, w in (("deployed", "Dep"), ("logonly", "Log")):
                for nl, wn in (("10", "Ten"), ("100", "Hundred"), ("ALL", "All")):
                    a = (name, "b", "where_least", start, nl)
                    if rec(*a, "probe.D1@mean", "heldout_disjoint") is not None:
                        pn(f"Etwo{sh}W{w}{wn}Done", val(*a, "probe.D1@mean", "heldout_disjoint"), f"{name}: ReScope-W, least repair of n = {nl} reported complaints from the {start} thresholds: D1 % on heldout_disjoint (mean over seeds)")
                        pn(f"Etwo{sh}W{w}{wn}Rel", val(*a, "edit.rel.exact@mean", "edits"), f"{name}: the same repair, Rel exact % (mean over seeds)")
                        pn(f"Etwo{sh}W{w}{wn}TGen", val(*a, "edit.tgen.exact@mean", "edits"), f"{name}: the same repair, T-Gen exact % (mean over seeds)")
                        pn(f"Etwo{sh}W{w}{wn}IGen", val(*a, "edit.igen.exact@mean", "edits"), f"{name}: the same repair, I-Gen exact % (mean over seeds)")
                        pn(f"Etwo{sh}W{w}{wn}Refused", val(*a, "meta.n_refused@mean", "repair"), f"{name}: the same repair, reported complaints refused (mean over seeds)")
                        pn(f"Etwo{sh}W{w}{wn}OwnOut", val(*a, "edit.rel.n_own_out_of_scope@mean", "edits"), f"{name}: the same repair, own Rel prompts out of scope (mean over seeds)")
            for nl, wn in (("10", "Ten"), ("ALL", "All")):
                a = (name, "bl", "blocklist", "deployed + exact-match list of the reported queries", nl)
                if rec(*a, "probe.D1@mean", "heldout_disjoint") is not None:
                    pn(f"Etwo{sh}Block{wn}Done", val(*a, "probe.D1@mean", "heldout_disjoint"), f"{name}: blocklist of the n = {nl} reported queries, D1 % on heldout_disjoint (mean over seeds)")
        # ---------------- (c) ReScope on label-free evidence
        if name in cgate:
            out += ["(c) ReScope (migration: last prompt position, top-1, rebased values) ON LABEL-FREE EVIDENCE: n benign REPAIR queries.  n = 0: log-only thresholds; n <= 20:",
                    "    + least repair of the n received queries (3 seeds); n >= 40: two-block rule at alpha = 0.05 (gate level 20 draws: mean [5th, 95th percentile over draws];",
                    "    at n = ALL the sample is the whole pool and a draw only changes the split into blocks); 'literal' = the rule without guards; capped = edits at their guard cap",
                    "    (bound void for them); HN = near-scope firing; BEN-T / MME-T at prompt-only keys.  Behaviour: seeds / draws 0 - 2 (mean, crossed bootstrap, (k) draws)", ""]
            for meth in ("rescope", "rescope_shared"):
                rows = []
                for nl in ("0", "1", "2", "5", "10", "20", "40", "50", "100", "200", "ALL"):
                    var = "logonly" if nl == "0" else ("logonly+least" if nl in ("1", "2", "5", "10", "20") else "")
                    g = (name, "c", meth, var, nl)
                    if rec(*g, "gate.fired@mean", "heldout_disjoint") is None:
                        continue
                    bv = {"0": "logonly", "": ""}.get(nl, None)
                    cells = [c for k, (c, _) in S.cells.items() if k[0] == "c" and k[1] == name and c["method"] == meth and str(c["n_label"]) == nl]
                    bvar = cells[0].get("variant", "") if cells else ""
                    b = (name, "c", meth, bvar, nl)
                    rg = rec(*g, "gate.fired@mean", "heldout_disjoint")
                    rows.append([nl, f(rg, 2) + f" ({rg['n_units']})", mean(*g, "gate.fired_literal", "heldout_disjoint", 2, False), mean(*g, "gate.n_edits_capped", "edits", 1, False),
                                 mean(*g, "gate.fired_strict", "heldout_strict", 2, False), mean(*g, "gate.BEN-T.fired", "BEN-T", 2, False), mean(*g, "gate.MME-T.fired", "MME-T", 2, False),
                                 " / ".join(mean(*g, f"gate.hn.{t}.fired", "hard_negatives", ci=False) for t in M.HN_TYPES),
                                 "/".join(mean(*g, f"gate.{m}.own_top", "edits", 0, False) for m in ("rel", "tgen", "igen")),
                                 mean(*b, "probe.D1", "heldout_disjoint", 2, nd_n=True), spread(*b, "probe.D1", "heldout_disjoint", 2),
                                 "/".join(mean(*b, f"edit.{m}.exact", "edits", 0, False) for m in ("rel", "tgen", "igen")),
                                 mean(*b, "d2.D2", "heldout_disjoint", ci=False), " / ".join(mean(*b, f"hn.{t}.D1", "hard_negatives", ci=False) for t in M.HN_TYPES)])
                out += [f"    {meth}", "    " + T(["n", "fire % disjoint (draws)", "literal", "capped", "fire strict", "BEN-T", "MME-T", "HN-I/Q/X fire %", "cover Rel/T/I %",
                                                   "D1 % disjoint (draws)", "single draws", "Rel/T-Gen/I-Gen exact", "D2 %", "HN-I/Q/X D1 %"], rows).replace("\n", "\n    "), ""]
            for nl, wn in (("0", "Zero"), ("10", "Ten"), ("200", "TwoHundred"), ("ALL", "All")):
                var = "logonly" if nl == "0" else ("logonly+least" if nl == "10" else "")
                g = (name, "c", "rescope", var, nl)
                if rec(*g, "gate.fired@mean", "heldout_disjoint") is not None:
                    pn(f"Etwo{sh}Rescope{wn}Fire", val(*g, "gate.fired@mean", "heldout_disjoint"), f"{name}: ReScope on n = {nl} benign queries, false firing % on heldout_disjoint, gate level, mean over draws")
                    pn(f"Etwo{sh}Rescope{wn}TGenCov", val(*g, "gate.tgen.own_top@mean", "edits"), f"{name}: the same, text rephrases released %, gate level, mean over draws")
                    pn(f"Etwo{sh}Rescope{wn}IGenCov", val(*g, "gate.igen.own_top@mean", "edits"), f"{name}: the same, image rephrases released %, gate level, mean over draws")
        # ---------------- (c') ReScope-W two-block
        if name in wgate:
            g = (name, "cw", "where_cal", "two_block", "200")
            rg = rec(*g, "gate.fired@mean", "heldout_disjoint")
            out += ["(c') ReScope-W WITH TWO-BLOCK THRESHOLDS FROM BENIGN QUERIES (R24): alpha = 0.05, n = 200, score = largest normalised coordinate over all positions of the",
                    "    teacher-forced pre-edit trace; gate level over the 20 draws from the captured vectors: mean [5th, 95th percentile]; behaviour draws 0 - 2",
                    "    " + T(["read-out", "value"],
                               [["false firing % heldout_disjoint (20 draws)", f(rg, 2)], ["  literal rule without guards", mean(*g, "gate.fired_literal", "heldout_disjoint", 2)],
                                ["  heldout_strict", mean(*g, "gate.fired_strict", "heldout_strict", 2)], ["capped edits", mean(*g, "gate.n_edits_capped", "edits", 1)],
                                ["own requests released at the last prompt position % (Rel / T-Gen / I-Gen)", " / ".join(mean(*g, f"gate.{m}.own_last", "edits", 1) for m in ("rel", "tgen", "igen"))],
                                ["  released at any position %", " / ".join(mean(*g, f"gate.{m}.own_any", "edits", 1, False) for m in ("rel", "tgen", "igen"))],
                                ["near-scope firing % HN-I / HN-Q / HN-X", " / ".join(mean(*g, f"gate.hn.{t}.fired", "hard_negatives", 1) for t in M.HN_TYPES)],
                                ["BEN-T / MME-T", "not defined (key-level pools hold the last-prompt key only)"],
                                ["behaviour: D1 % heldout_disjoint (draws)", mean(*g, "probe.D1", "heldout_disjoint", 2, nd_n=True) + " " + spread(*g, "probe.D1", "heldout_disjoint", 2)],
                                ["  D1 % heldout_strict", mean(*g, "probe.D1", "heldout_strict", 2, False)], ["  fire % heldout_disjoint (hook)", mean(*g, "probe.fired", "heldout_disjoint", 2) + " " + spread(*g, "probe.fired", "heldout_disjoint", 2)],
                                ["  Rel / T-Gen / I-Gen exact %", " / ".join(mean(*g, f"edit.{m}.exact", "edits", 1) for m in ("rel", "tgen", "igen"))],
                                ["  Rel kept / gained against the deployed log % of edits", mean(*g, "kg.rel.kept", "edits", 1, False) + " / " + mean(*g, "kg.rel.gained", "edits", 1, False)],
                                ["  T-Gen kept / gained; I-Gen kept / gained", mean(*g, "kg.tgen.kept", "edits", 1, False) + " / " + mean(*g, "kg.tgen.gained", "edits", 1, False) + "; "
                                 + mean(*g, "kg.igen.kept", "edits", 1, False) + " / " + mean(*g, "kg.igen.gained", "edits", 1, False)],
                                ["  T-Loc* / I-Loc*", mean(*g, "probe.T-Loc*", "heldout_disjoint", 2, False) + " / " + mean(*g, "probe.I-Loc*", "heldout_disjoint", 2, False)],
                                ["  D2 % of pre-correct rows (draw 0)", mean(*g, "d2.D2", "heldout_disjoint", 1)],
                                ["  HN-I / HN-Q / HN-X D1 % (draw 0)", " / ".join(mean(*g, f"hn.{t}.D1", "hard_negatives", 1, False) for t in M.HN_TYPES)],
                                ["  hook decision != gate-level decision (rows, draws 0 - 2)", mean(*g, "check.hook_vs_vectors.n_differ", "all", 0, False) + " of " + mean(*g, "check.hook_vs_vectors.n_rows", "all", 0, False)]]).replace("\n", "\n    "), ""]
            pc = os.path.join(e2c.OUT, "parts", f"cw_check_{name}.json")
            if os.path.exists(pc):
                ck = fx.io.jload(pc)
                out += [f"    check (code/cw_check.py; literal rule, 20 draws): mean false firing {ck['mean_fired_heldout_disjoint']:.2f} % on heldout_disjoint (one fixed pool of "
                        f"{ck['n_heldout_disjoint']} rows) and {ck['mean_fired_fresh_repair']:.2f} % on the REPAIR rows whose query is in no calibration draw (exchangeable rows); "
                        f"bound {ck['bound']:.2f} %", ""]
                pn(f"Etwo{sh}WCalFireLiteral", ck["mean_fired_heldout_disjoint"], f"{name}: ReScope-W two-block, literal rule without guards, false firing % on heldout_disjoint, mean over 20 draws")
                pn(f"Etwo{sh}WCalFireFreshRepair", ck["mean_fired_fresh_repair"], f"{name}: the same rule, false firing % on REPAIR rows whose query is in no calibration draw, mean over 20 draws")
            for k, metric, pool, text in (("Fire", "gate.fired@mean", "heldout_disjoint", "false firing % on heldout_disjoint, gate level, mean over 20 draws"),
                                          ("Capped", "gate.n_edits_capped@mean", "edits", "capped edits, mean over 20 draws"),
                                          ("Done", "probe.D1@mean", "heldout_disjoint", "D1 % on heldout_disjoint, mean over behavioural draws 0 - 2"),
                                          ("Rel", "edit.rel.exact@mean", "edits", "Rel exact %, mean over draws 0 - 2"), ("TGen", "edit.tgen.exact@mean", "edits", "T-Gen exact %, mean over draws 0 - 2"),
                                          ("IGen", "edit.igen.exact@mean", "edits", "I-Gen exact %, mean over draws 0 - 2"), ("Dtwo", "d2.D2@mean", "heldout_disjoint", "D2 % of the pre-correct rows, draw 0")):
                if rec(*g, metric, pool) is not None:
                    pn(f"Etwo{sh}WCal{k}", val(*g, metric, pool), f"{name}: ReScope-W with two-block thresholds (alpha 0.05, n 200): {text}")
        # ---------------- (d) linear rows
        dcells = sorted((c for k, (c, _) in S.cells.items() if k[0] == "d" and k[1] == name), key=lambda c: c["tag"])
        if dcells:
            rows = []
            for c in dcells:
                a = (name, "d", c["method"], c["variant"], c["n_label"])
                fr = c.get("flags_vs_record") or {}
                rows.append([f"{c['method']} n={c['n_label']} ({c['variant']})", f(rec(*a, "probe.D1", "heldout_disjoint", draw=0)), f(rec(*a, "probe.D1", "heldout_strict", draw=0), ci=False),
                             "/".join(f(rec(*a, f"edit.{m}.exact", "edits", draw=0), 0, False) for m in ("rel", "tgen", "igen")),
                             "/".join(f(rec(*a, f"kg.{m}.kept", "edits", draw=0), 0) + "+" + f(rec(*a, f"kg.{m}.gained", "edits", draw=0), 0) for m in ("rel", "tgen", "igen")),
                             f(rec(*a, "probe.T-Loc*", "heldout_disjoint", draw=0), 1) + "/" + f(rec(*a, "probe.I-Loc*", "heldout_disjoint", draw=0), 1),
                             " / ".join(f(rec(*a, f"hn.{t}.D1", "hard_negatives", draw=0), 1, False) for t in M.HN_TYPES), f(rec(*a, "d2.D2", "heldout_disjoint", draw=0)),
                             "/".join(f(rec(*a, f"gen.{m}.eq", "edits", draw=0), 0) for m in ("rel", "tgen", "igen")),
                             f"{fr.get('n_heldout_differ', '-')}/{fr.get('n_heldout', '-')}; {fr.get('n_edit_differ', '-')}/{fr.get('n_edit', '-')}"])
                wn = {"10": "Ten", "ALL": "All"}[str(c["n_label"])]
                nm = f"Etwo{sh}Row{c['method'].title().replace('_', '')}{wn}"
                pn(nm + "Done", val(*a, "probe.D1", "heldout_disjoint", draw=0), f"{name}: {c['method']} at n = {c['n_label']} ({c['variant']}), D1 % on heldout_disjoint, driver at B = 1")
                pn(nm + "Rel", val(*a, "edit.rel.exact", "edits", draw=0), f"{name}: the same row, Rel exact %")
                pn(nm + "Dtwo", val(*a, "d2.D2", "heldout_disjoint", draw=0), f"{name}: the same row, D2 % of the pre-correct heldout_disjoint rows")
                pn(nm + "FlagsDiffer", fr.get("n_heldout_differ"), f"{name}: the same row, held-out D1 flags (of {fr.get('n_heldout')}) that differ from the P12 record")
            out += ["(d) LINEAR ROWS FOR THE MAIN TABLE AND THE LIMITS (alpha does not apply; seed 0; evaluated at B = 1 through the driver).  near-scope firing: not defined for a",
                    "    linear repair.  'flags vs record' = held-out D1 flags (of the 490 / 494 valid held-out rows) and edit flags (of 300) that differ from the P12 record",
                    "    " + T(["row", "D1 % disjoint", "D1 strict", "Rel/T-Gen/I-Gen", "kept+gained Rel/T/I", "T-Loc*/I-Loc*", "HN-I/Q/X D1 %", "D2 %", "gen eq Rel/T/I", "flags vs record (held-out; edits)"],
                               rows).replace("\n", "\n    "), ""]
        # ---------------- H1
        hb = h1_log(S, name, L, rec)
        if hb is not None:
            h1[name] = hb
            out += ["(H1) REPAIR AT EQUAL EVIDENCE, n = 10 reported complaints (R6): ReScope-W (least repair from the log-only thresholds) against the best linear repair of the same",
                    "    complaints (P12 records, seeds 0 - 2); paired crossed bootstrap over seeds and probe clusters; differences = ReScope-W minus linear",
                    "    " + T(["comparison", "linear repair", "D1 % ReScope-W", "D1 % linear", "difference [95 %] (p)", "Rel ReScope-W", "Rel linear", "Rel deployed"],
                               [[k, v["linear"], f"{v['d1_where']:.2f}", f"{v['d1_linear']:.2f}", f"{v['d1_diff']['value']:.2f} [{v['d1_diff']['ci_lo']:.2f}, {v['d1_diff']['ci_hi']:.2f}] (p = {v['d1_diff']['p_two_sided']:.4f})",
                                 f"{v['rel_where']:.1f}", f"{v['rel_linear']:.1f}", f"{hb['rel_deployed']:.1f}"] for k, v in hb["comparisons"].items()]).replace("\n", "\n    "),
                    f"    rule of R6 on this log: D1 of ReScope-W at most 10 % ({hb['d1_where']:.2f}: {hb['d1_at_most_10']}) and Rel exact within 2 points of the deployed log "
                    f"({hb['rel_where']:.1f} vs {hb['rel_deployed']:.1f}, difference {hb['rel_minus_deployed']:+.1f}: literal two-sided reading {hb['rel_within_2']}; one-sided reading "
                    f"'not more than 2 points below' {hb['rel_not_more_than_2_below_deployed']}) -> literal: {hb['verdict_on_this_log']}; one-sided reading met: {hb['one_sided_reading_met_on_this_log']}", ""]
            pn(f"Etwo{sh}HoneWhereDone", hb["d1_where"], f"{name}: H1, ReScope-W (least repair of 10 complaints from the log-only thresholds), D1 % on heldout_disjoint, mean over 3 seeds")
            pn(f"Etwo{sh}HoneWhereRel", hb["rel_where"], f"{name}: H1, the same repair, Rel exact %, mean over 3 seeds")
            for k, w in (("best_lowest_d1", "Best"), ("best_rel_within_2", "BestRel")):
                if k in hb["comparisons"]:
                    v = hb["comparisons"][k]
                    pn(f"Etwo{sh}Hone{w}Name", v["linear"], f"{name}: H1, the linear repair selected as '{k}' at n = 10")
                    pn(f"Etwo{sh}Hone{w}Done", v["d1_linear"], f"{name}: H1, D1 % on heldout_disjoint of that linear repair, mean over 3 seeds")
                    pn(f"Etwo{sh}Hone{w}Diff", v["d1_diff"]["value"], f"{name}: H1, D1 difference ReScope-W minus that linear repair (points)")
                    pn(f"Etwo{sh}Hone{w}DiffLo", v["d1_diff"]["ci_lo"], f"{name}: H1, lower end of the 95 % interval of that difference (paired crossed bootstrap, 3 seeds x query clusters)")
                    pn(f"Etwo{sh}Hone{w}DiffHi", v["d1_diff"]["ci_hi"], f"{name}: H1, upper end of that interval")
            pn(f"Etwo{sh}HoneVerdict", hb["verdict_on_this_log"], f"{name}: H1, the rule of R6 applied literally (|Rel - Rel deployed| <= 2) on this log (the full rule also needs one mild prefix: e2_prefix)")
            pn(f"Etwo{sh}HoneRelMinusDeployed", hb["rel_minus_deployed"], f"{name}: H1, Rel exact of ReScope-W minus Rel exact of the deployed log (points)")
    if h1:
        ps = {f"{e2c.SHORT[n]}/{k}": v["d1_diff"]["p_two_sided"] for n, hb in h1.items() for k, v in hb["comparisons"].items()}
        hol = ST.holm(ps)
        blob = {"what": "H1 (R6): held-out D1 after n = 10 reported complaints, ReScope-W (least repair from the log-only thresholds) against the best linear repair of the same "
                        "complaints; development logs; paired crossed bootstrap over seeds (3) and query clusters; Holm over backbones and baselines",
                "rule": "helps iff D1 <= 10 % with Rel exact within 2 points of the deployed log, on the development logs AND on at least one mild prefix (R12; the prefix part is "
                        "experiment e2_prefix, not this file)",
                "logs": h1, "holm": hol, "helps_on_both_development_logs": all(hb["helps_on_this_log"] for hb in h1.values()) and len(h1) == 2,
                "one_sided_reading_met_on_both_development_logs": all(hb["one_sided_reading_met_on_this_log"] for hb in h1.values()) and len(h1) == 2,
                "n_seeds": 3, "note_intervals": "three seeds: the crossed bootstrap is too narrow where the seed-to-seed component matters (R25); per-seed values are listed"}
        fx.io.jdump(blob, os.path.join(e2c.OUT, "h1.json"), indent=1)
        out += [f"H1 over the development logs ({len(h1)} of 2 available): literal reading: " + ("the rule holds on both" if blob["helps_on_both_development_logs"] else "the rule does NOT hold on both")
                + "; one-sided reading: " + ("met on both" if blob["one_sided_reading_met_on_both_development_logs"] else "not met on both")
                + "; Holm-adjusted p of the D1 differences: " + ", ".join(f"{k}: {v:.4f}" for k, v in hol["adjusted"].items()) if isinstance(hol.get("adjusted"), dict) else "H1 written to out/h1.json", ""]
    S.R.table("\n".join(out))
    S.R.paper({k: v for k, v in paper.items() if v is not None})
    fx.io.jdump(dict(sorted((k, v) for k, v in doc.items() if paper.get(k) is not None)), os.path.join(e2c.OUT, "paper_numbers_doc.json"), indent=1)


def sanity_compare(S, name):
    """The cells this experiment shares with the sanity run e0 (deployed log; rescope / rescope_shared at alpha 0.05, n 200, draws
    0 - 2; hard negatives at the draws that have them): largest absolute difference per metric, in points."""
    p = os.path.join(e2c.FIN, "e0_sanity", "out", "results.json")
    if not os.path.exists(p):
        return None
    e0 = fx.io.jload(p)["logs"][name]
    bm = e0["behaviour"]["methods"]
    pairs = []
    if ("base", name, "deployed") in S.cells:
        pairs.append(("deployed", bm["deployed"]["single"], S.cells[("base", name, "deployed")][1]))
    for d in (0, 1, 2):
        for mine, theirs in (("rescope", "M1"), ("rescope_shared", "M2")):
            k = ("c", name, f"{mine}_n200__d{d}")
            if k in S.cells and str(d) in bm[theirs]["draws"]:
                pairs.append((f"{mine} draw {d}", bm[theirs]["draws"][str(d)], S.cells[k][1]))
    if not pairs:
        return None
    v = lambda x: x["value"] if isinstance(x, dict) else x                # noqa: E731
    worst, n = {}, 0
    for lab, a, st in pairs:
        cmp_ = [("D1 heldout_disjoint", a["probe"]["heldout_disjoint"]["D1"], st.get(("heldout_disjoint", "probe.D1"))),
                ("D1 heldout_strict", a["probe"]["heldout_strict"]["D1"], st.get(("heldout_strict", "probe.D1"))),
                ("fired heldout_disjoint", a["probe"]["heldout_disjoint"].get("fired"), st.get(("heldout_disjoint", "probe.fired")))]
        cmp_ += [(f"{m} exact", a["edit"][m]["exact"], st.get(("edits", f"edit.{m}.exact"))) for m in ("rel", "tgen", "igen")]
        cmp_ += [(f"{t} D1", a["hn"][t]["D1"], st.get(("hard_negatives", f"hn.{t}.D1"))) for t in M.HN_TYPES]
        for nm, x, y in cmp_:
            if x is None or y is None:
                continue
            n += 1
            dv = abs(float(v(x)) - float(v(y)))
            if dv > worst.get(nm, (-1, ""))[0]:
                worst[nm] = (dv, lab)
    return {"n_cells": len(pairs), "n_numbers": n, "max_abs_diff": {k: w[0] for k, w in worst.items()}, "cells": [p_[0] for p_ in pairs],
            "largest": max(w[0] for w in worst.values()) if worst else None}


def h1_log(S, name, L, rec):
    """H1 read-out of one log from the per-row files (None until the cells exist)."""
    wk = [("b", name, f"where_least__logonly__n10__s{s}") for s in (0, 1, 2)]
    if any(k not in S.cells for k in wk) or ("base", name, "deployed") not in S.cells:
        return None
    isp = lambda r: r["kind"] == "probe" and r.get("disjoint") and "line" not in r and not r.get("d2_row")      # noqa: E731
    isr = lambda r: r["kind"] == "edit" and r["metric"] == "rel" and "line" not in r                             # noqa: E731
    d1 = lambda keys: M.grid({i: S.rows[k] for i, k in enumerate(keys)}, lambda r: float(bool(r["D1"])), unit="query_id", where=isp)      # noqa: E731
    rel = lambda keys: M.grid({i: S.rows[k] for i, k in enumerate(keys)}, lambda r: float(bool(r["exact"])), unit="edit", where=isr)      # noqa: E731
    gw, rw = d1(wk), rel(wk)
    mw = 100.0 * float(gw["num"].sum() / gw["den"].sum())
    relw = 100.0 * float(rw["num"].sum() / rw["den"].sum())
    dep_rel = 100.0 * float(np.mean([bool(r["exact"]) for r in S.rows[("base", name, "deployed")] if isr(r)]))
    cand = {}
    for m in e2c.LINEAR:
        lk = [("a", name, f"{m}__n10__s{s}") for s in (0, 1, 2)]
        if any(k not in S.cells for k in lk):
            continue
        g, r = d1(lk), rel(lk)
        cand[m] = {"keys": lk, "g": g, "r": r, "d1": 100.0 * float(g["num"].sum() / g["den"].sum()), "rel": 100.0 * float(r["num"].sum() / r["den"].sum())}
    if not cand:
        return None
    comps = {}
    sel = {"best_lowest_d1": min(cand, key=lambda m: cand[m]["d1"])}
    ok = [m for m in cand if cand[m]["rel"] >= dep_rel - 2.0]
    if ok:
        sel["best_rel_within_2"] = min(ok, key=lambda m: cand[m]["d1"])
    for k, m in sel.items():
        c = cand[m]
        if gw["units"] != c["g"]["units"]:
            raise RuntimeError("H1: the two methods do not hold the same query clusters")
        dd = ST.crossed_bootstrap_diff(gw["num"], c["g"]["num"], gw["den"], c["g"]["den"], scale=100.0)
        dr = ST.crossed_bootstrap_diff(rw["num"], c["r"]["num"], rw["den"], c["r"]["den"], scale=100.0)
        comps[k] = {"linear": m, "d1_where": mw, "d1_linear": c["d1"], "d1_diff": dd, "rel_where": relw, "rel_linear": c["rel"], "rel_diff": dr,
                    "d1_where_per_seed": [100.0 * float(gw["num"][i].sum() / gw["den"][i].sum()) for i in range(3)],
                    "d1_linear_per_seed": [100.0 * float(c["g"]["num"][i].sum() / c["g"]["den"][i].sum()) for i in range(3)]}
    a, b = mw <= 10.0, abs(relw - dep_rel) <= 2.0                       # R6 literally: "within 2 points of the deployed log"
    return {"log": name, "n": 10, "seeds": [0, 1, 2], "d1_where": mw, "rel_where": relw, "rel_deployed": dep_rel,
            "linear_candidates": {m: {"d1": c["d1"], "rel": c["rel"]} for m, c in cand.items()}, "comparisons": comps,
            "d1_at_most_10": bool(a), "rel_within_2": bool(b), "rel_not_more_than_2_below_deployed": bool(relw >= dep_rel - 2.0), "helps_on_this_log": bool(a and b),
            "rel_minus_deployed": relw - dep_rel, "one_sided_reading_met_on_this_log": bool(a and relw >= dep_rel - 2.0),
            "readings": "literal (fixed in the code before any ReScope-W cell existed): |Rel - Rel deployed| <= 2 points; one-sided: Rel >= Rel deployed - 2 points. "
                        "The two differ only where the repair has a HIGHER Rel exact than the deployed log; which reading R6 means is the coordinator's decision.",
            "selection_of_best_rel_within_2": "linear repairs whose mean Rel exact is not more than 2 points below the deployed log (one-sided: in favour of the baseline)",
            "verdict_on_this_log": "helps (criteria of R6 met on this log)" if a and b else "does not help (criteria of R6 not met on this log)",
            "linear_flags_from": "P12 records (harness evaluation); ReScope-W flags: driver at B = 1; the same heldout_disjoint rows"}
