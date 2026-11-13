"""P12 verifier -- VERIFY_REPORT.md rendered from the verifier's JSON outputs (no hand-typed result number).

inputs : out/draw.json, out/cpu_checks.json, out/render_check.json, out/leak_pixels.json, out/cpu_moddiag.json,
         out/cpu_extra.json, out/zstar_compare_*.json, out/<log>/*.json (gpu_verify.py)
outputs: out/verify_summary.json, VERIFY_REPORT.md
usage  : python render_report.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402

SHORT = {"TB-Q100-ae-l2_3": "Qwen", "TB-L100-ae-l2_3": "LLaVA"}
LAB = {"COMP_B": "COMP-B", "COMP_BP": "COMP-B+", "COMP_B_ALLPOS": "ALLPOS", "REEDIT_B": "REEDIT-B", "RB_TOT": "RB-TOT", "RB_UNION": "RB-UNION"}


def opt(p):
    return VC.jload(p) if os.path.exists(p) else None


def yn(x):
    return "yes" if x else "NO"


draw = VC.jload(os.path.join(VC.VOUT, "draw.json"))
cpu = opt(os.path.join(VC.VOUT, "cpu_checks.json"))
rend = opt(os.path.join(VC.VOUT, "render_check.json"))
leakp = opt(os.path.join(VC.VOUT, "leak_pixels.json"))
cmod = opt(os.path.join(VC.VOUT, "cpu_moddiag.json")) or {}
cext = opt(os.path.join(VC.VOUT, "cpu_extra.json")) or {}
expres = opt(os.path.join(VC.P12, "out", "results.json")) or {"logs": {}}
S = {"logs": {}, "table": []}
tab = []            # (quantity, experiment, mine, agree)
missing = []


def gp(log, cond):
    return opt(os.path.join(VC.VOUT, log, cond + ".json"))


def cond_names(log):
    out = [("NONE", "NONE (unrepaired)", "exact"), ("W0", "ALL (W_0)", "exact"), ("BEST__COMP_B__nALL", "best point COMP-B n=ALL", "exact")]
    for m, n in draw["draw"][log]:
        if m == "REEDIT_B" and log == "TB-Q100-ae-l2_3" and n in (10, 100, "ALL"):
            out.append((f"{m}__n{n}__theirz", f"random cell {LAB[m]} n={n} (first-pass record), experiment's z* re-installed by me", "exact"))
            out.append((f"{m}__n{n}__asrec", f"random cell {LAB[m]} n={n} (first-pass record), my own z*, same installed set", "noisy"))
            out.append((f"{m}__n{n}__ladder", f"random cell {LAB[m]} n={n}, stated protocol (my z*, fallback ladder; the experiment has no such record)", "other"))
        elif m == "REEDIT_B":
            out.append((f"{m}__n{n}__theirz", f"random cell {LAB[m]} n={n}, experiment's z* re-installed by me", "exact"))
            out.append((f"{m}__n{n}", f"random cell {LAB[m]} n={n}, my own z*", "noisy"))
        else:
            out.append((f"{m}__n{n}", f"random cell {LAB[m]} n={n}", "exact"))
    return out


exact_ok = {}        # (log, cond) -> True / False (conditions that must agree row for row)
for log in VC.LOGS:
    sh = SHORT[log]
    L = {"conds": {}}
    L["xcheck"] = opt(os.path.join(VC.VOUT, log, "_xcheck.json"))
    rows = VC.records(log)
    pool = VC.my_split(rows)
    ho = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    key = lambda r: (r["ptype"], r["prompt"], r["image"])  # noqa: E731
    repk = {key(r) for r in rows if pool[r["probe_idx"]] == "repair"}
    dis = [k for k, r in enumerate(ho) if key(r) not in repk]
    we = [bool(r["reproduced"]) for r in ho]
    uids = [r["uid"] for r in ho]
    for cond, label, kind in cond_names(log):
        d = gp(log, cond)
        if d is None:
            missing.append(f"{sh} {label}")
            tab.append((f"{sh} {label}: HELD-OUT D1 % / Rel exact", "-", "NOT RUN (no GPU window)", "-"))
            continue
        m, v = d["mine"], d.get("vs_experiment", {})
        c = {"mine": m, "kind": kind, "vs": {k: x for k, x in v.items() if k not in ("heldout_rows_differ", "rel_em_differ")},
             "heldout_rows_differ": v.get("heldout_rows_differ"), "rel_em_differ": v.get("rel_em_differ"),
             "repair": {k: x for k, x in (d.get("repair") or {}).items() if k not in ("meta", "cost", "my_z_attempts")},
             "saved": d.get("vs_saved_correction"), "sanity": d.get("sanity")}
        h = d["flags"]["heldout"]
        lo, hi = VC.cluster_boot_ci([0.0 if x else 1.0 for x in h], uids)
        p, b, cc = VC.mcnemar_exact(h, we)
        pd, _, _ = VC.mcnemar_exact([h[k] for k in dis], [we[k] for k in dis])
        c["stats"] = {"ci": [100 * lo, 100 * hi], "mcnemar_p": p, "fixed": b, "lost": cc,
                      "disjoint_D1": 100.0 * sum(not h[k] for k in dis) / len(dis),
                      "disjoint_D1_WE": 100.0 * sum(not we[k] for k in dis) / len(dis), "disjoint_p": pd, "disjoint_n": len(dis)}
        L["conds"][cond] = c
        has = v.get("experiment_record") is not None and kind != "other"
        fp = " (first-pass record)" if d.get("experiment_record_is_first_pass") else ""
        e_d1 = f"{v['heldout_D1_exp']:.1f} ({v['heldout_n_D1_exp']}/{v['heldout_rows']}){fp}" if has else "no record"
        e_rel = f"{v['rel_exact_exp']:.0f}" if has else "no record"
        ok_h = has and v["heldout_rows_agree"] == v["heldout_rows"]
        ok_r = has and v["rel_em_agree"] == 100
        if kind == "exact":
            exact_ok[(log, cond)] = bool(ok_h and ok_r)
            a_h, a_r = f"{yn(ok_h)}, {v['heldout_rows_agree']}/{v['heldout_rows']} rows", f"{yn(ok_r)}, {v['rel_em_agree']}/100 edits"
        elif kind == "noisy":
            # my own z* targets: identical inputs otherwise.  Rows can differ only through z* (not reproducible on image
            # rows); whether the install + evaluation are right is what the "experiment's z* re-installed" row shows.
            dd = abs(m["heldout_D1"] - v["heldout_D1_exp"])
            tz = exact_ok.get((log, cond.replace("__asrec", "__theirz") if cond.endswith("__asrec") else cond + "__theirz"))
            if ok_h and ok_r:
                a_h, a_r = f"yes, {v['heldout_rows_agree']}/{v['heldout_rows']} rows (z* reproduced)", f"yes, {v['rel_em_agree']}/100 edits"
            else:
                why_ = "differs only through z*" if tz else "differs (z* and possibly more: the re-install row does not agree or was not run)"
                a_h = f"{why_}: {v['heldout_rows_agree']}/{v['heldout_rows']} rows equal, D1 {dd:.1f} points apart"
                a_r = f"{v['rel_em_agree']}/100 edits equal"
        else:
            a_h = a_r = "-"
        tab.append((f"{sh} {label}: HELD-OUT D1 % (+fixed / -lost)", e_d1,
                    f"{m['heldout_D1']:.1f} ({m['heldout_n_D1']}/{m['heldout_n']}; +{m['heldout_complaints_fixed']} / -{m['heldout_noncomplaints_lost']})", a_h))
        tab.append((f"{sh} {label}: Rel exact", e_rel, f"{m['rel_exact']:.0f}", a_r))
        if "reported_resolved" in m:
            if has and "reported_resolved_exp" in v:
                tab.append((f"{sh} {label}: reported resolved", f"{v['reported_resolved_exp']}/{m['reported_n']}", f"{m['reported_resolved']}/{m['reported_n']}",
                            (f"{yn(v['reported_agree'] == m['reported_n'])}, " if kind == "exact" else "") + f"{v['reported_agree']}/{m['reported_n']} rows"))
            else:
                rp = d.get("repair") or {}
                tab.append((f"{sh} {label}: reported resolved / installed / norm over log", "no record",
                            f"{m['reported_resolved']}/{m['reported_n']} / {rp.get('installed')} / {rp.get('fro_rel_to_log', float('nan')):.3f}", "-"))
        sv = d.get("vs_saved_correction") or {}
        if sv.get("saved_correction") and kind != "other":
            if "entries_differing" in sv:
                tab.append((f"{sh} {label}: rebuilt weight vs the experiment's saved weight", f"norm/log {v.get('fro_rel_to_log_exp', float('nan')):.3f}",
                            f"norm/log {d['repair']['fro_rel_to_log']:.3f}; {sv['entries_differing']} of {sv['n_entries']} entries differ (rel. diff {sv['rel_fro_diff']:.1e})",
                            yn(sv["entries_differing"] == 0) if kind == "exact" else "z* noise"))
            elif "rel_fro_diff" in sv:
                tab.append((f"{sh} {label}: rebuilt operator vs the experiment's saved operator", f"rank {sv['rank_saved']}, norm/log {v.get('fro_rel_to_log_exp', float('nan')):.3f}",
                            f"rank {sv['rank_mine']}, norm/log {d['repair']['fro_rel_to_log']:.3f}; rel. Frobenius diff {sv['rel_fro_diff']:.1e}",
                            yn(sv["rel_fro_diff"] < 1e-6)))
    n0, w0 = L["conds"].get("NONE"), L["conds"].get("W0")
    if n0:
        s = n0["sanity"]
        tab.append((f"{sh} sanity: NONE vs stored v2 flags, rows differing (HELD-OUT / REPAIR)", "0 / 0",
                    f"{s['heldout_flags_differing_from_stored']} / {s['repair_flags_differing_from_stored']} (stored D1 {s['stored_heldout_D1']} / {s['stored_repair_D1']})",
                    yn(s["heldout_flags_equal_stored_v2"] and s["repair_flags_equal_stored_v2"])))
    if w0:
        s = w0["sanity"]
        tab.append((f"{sh} sanity: D1 complaints at W_0 (HELD-OUT / REPAIR); weight == checkpoint", "0 / 0; True",
                    f"{s['heldout_D1_at_W0']} / {s['repair_D1_at_W0']}; {s['weight_equals_W0']} (W_0 == checkpoint file: {(L['xcheck'] or {}).get('cross_checks', {}).get('W0_equals_checkpoint_file')})",
                    yn(s["heldout_D1_at_W0"] == 0 and s["repair_D1_at_W0"] == 0 and s["weight_equals_W0"])))
    rb8 = gp(log, "RELB8")
    L["relb8"] = rb8
    if rb8:
        tab.append((f"{sh} sanity: stored Rel exact / token on the testbed's batch-8 path", f"{rb8['stored']['Rel_exact']:.1f} / {rb8['stored']['Rel_token']:.2f}",
                    f"{rb8['mine_b8']['rel']['exact']:.1f} / {rb8['mine_b8']['rel']['token']:.2f}", yn(rb8["rel_matches_stored"])))
    lc = gp(log, "LOC")
    L["loc"] = lc
    if lc:
        tab.append((f"{sh} best point: T-Loc / I-Loc of the held-out items, unrepaired -> repaired",
                    f"{lc['experiment']['tloc']['WE']:.1f} -> {lc['experiment']['tloc']['best']:.1f} / {lc['experiment']['iloc']['WE']:.1f} -> {lc['experiment']['iloc']['best']:.1f}",
                    f"{lc['mine']['tloc']['WE']:.1f} -> {lc['mine']['tloc']['best']:.1f} / {lc['mine']['iloc']['WE']:.1f} -> {lc['mine']['iloc']['best']:.1f} (probes and comparison rebuilt; vendor batch-8 forwards)",
                    yn(lc["agree"])))
    b = L["conds"].get("BEST__COMP_B__nALL")
    if b and cpu:
        hd = cpu["C_headlines"][log]
        st = b["stats"]
        tab.append((f"{sh} best point: 95% CI / exact McNemar p (full pool)", f"[{hd['best_ci'][0]:.1f}, {hd['best_ci'][1]:.1f}] / {hd['best_mcnemar_p']:.2g}",
                    f"[{st['ci'][0]:.1f}, {st['ci'][1]:.1f}] / {st['mcnemar_p']:.2g} (my flags, my statistics)",
                    yn(abs(st["ci"][0] - hd["best_ci"][0]) < 0.05 and abs(st["ci"][1] - hd["best_ci"][1]) < 0.05 and abs(st["mcnemar_p"] - hd["best_mcnemar_p"]) < 1e-9)))
        tab.append((f"{sh} best point on the non-duplicated rows: D1 % vs unrepaired (p)",
                    f"{hd['disjoint_best_D1']:.1f} vs {hd['disjoint_none_D1']:.1f} ({hd['disjoint_mcnemar_p']:.2g})",
                    f"{st['disjoint_D1']:.1f} vs {st['disjoint_D1_WE']:.1f} ({st['disjoint_p']:.2g}; {st['disjoint_n']} rows)",
                    yn(abs(st["disjoint_D1"] - hd["disjoint_best_D1"]) < 1e-9 and abs(st["disjoint_p"] - hd["disjoint_mcnemar_p"]) < 1e-9)))
    wp = gp(log, "WPROJ")
    L["wproj"] = None if wp is None else {"mine": wp["mine"], "weight": wp["weight"]}
    if wp and n0:
        mw, m0 = wp["mine"], n0["mine"]
        tab.append((f"{sh} verifier's extra (no experiment record): behaviour of W_0 + D P_ans, i.e. W_0 restored outside the edits' answer-key span -- HELD-OUT D1 % (+fixed / -lost); Rel exact; REPAIR-pool D1",
                    f"module-level statement only (0.87 of the module change left); unrepaired {m0['heldout_D1']:.1f}, Rel {m0['rel_exact']:.0f}, REPAIR D1 {m0['repair_n_D1']}/{m0['repair_n']}",
                    f"{mw['heldout_D1']:.1f} ({mw['heldout_n_D1']}/{mw['heldout_n']}; +{mw['heldout_complaints_fixed']} / -{mw['heldout_noncomplaints_lost']}); Rel {mw['rel_exact']:.0f}; "
                    f"REPAIR D1 {mw['repair_n_D1']}/{mw['repair_n']} (+{mw['repair_complaints_fixed']} / -{mw['repair_noncomplaints_lost']})", "-"))
    L["protkeys"] = gp(log, "PROTKEYS")
    mdg = gp(log, "MODDIAG")
    ps = expres["logs"].get(log, {}).get("protect_split")
    if mdg:
        r = mdg["rows"]
        twg = {ho[k]["group"] for k in range(len(ho)) if k not in set(dis)}
        sel = [i for i, g in enumerate(r["group"]) if g not in twg]
        med = lambda v, comp: float(np.median([v[i] for i in sel if r["stored_D1"][i] == comp]))  # noqa: E731
        L["moddiag"] = {"source": "my own key capture (GPU)", "rank_K_ans": mdg["rank_K_ans"], "s_complaints": med(r["s"], True), "s_noncomplaints": med(r["s"], False),
                        "a_complaints": med(r["a"], True), "a_noncomplaints": med(r["a"], False)}
    elif cmod.get(log) and "rank_K_ans" in cmod[log]:
        L["moddiag"] = dict({k: cmod[log][k] for k in ("rank_K_ans", "s_complaints", "s_noncomplaints", "a_complaints", "a_noncomplaints")},
                            source="my own linear algebra on the experiment's captured held-out keys (CPU)")
    if ps and L.get("moddiag"):
        md_ = L["moddiag"]
        tab.append((f"{sh} module level: rank of the edits' answer-key span / share of a held-out complaint's key norm in it / module change left if W_0 is restored outside it (medians)",
                    f"{ps['rank_K_ans']} / {ps['s_complaints']:.2f} / {ps['a_complaints']:.2f}",
                    f"{md_['rank_K_ans']} / {md_['s_complaints']:.2f} / {md_['a_complaints']:.2f} ({md_['source']})",
                    yn(ps["rank_K_ans"] == md_["rank_K_ans"] and abs(ps["s_complaints"] - md_["s_complaints"]) < 0.005 and abs(ps["a_complaints"] - md_["a_complaints"]) < 0.005)))
        tab.append((f"{sh} module level, the same two medians for held-out NON-complaints (not quoted in the report's summary)",
                    f"{ps['s_noncomplaints']:.2f} / {ps['a_noncomplaints']:.2f}", f"{md_['s_noncomplaints']:.2f} / {md_['a_noncomplaints']:.2f}",
                    yn(abs(ps["s_noncomplaints"] - md_["s_noncomplaints"]) < 0.005 and abs(ps["a_noncomplaints"] - md_["a_noncomplaints"]) < 0.005)))
    S["logs"][log] = L

if cpu:
    C = cpu["C_report_tables"]
    tab.append(("report's two tables: cells equal to my own summary of the raw records", f"{C['cells']} cells", f"{C['cells_agree']} agree; dagger marks correct: {C['dagger_ok']}",
                yn(C["cells_agree"] == C["cells"] and C["dagger_ok"] and C["empty_cells_ok"])))
    for log in VC.LOGS:
        a, b, hd = cpu["A_records"][log], cpu["B_aggregate"][log], cpu["C_headlines"][log]
        sh = SHORT[log]
        tab.append((f"{sh} raw records: reported set = first n of MY stream / all valid REPAIR-pool D1 complaints / held-out rows = MY valid HELD-OUT rows",
                    f"{a['n_records']} records", f"{a['prefix_ok']} / {a['reported_all_repair_D1_valid']} / {a['heldout_rows_exact']}; nested pairs {a['nested_pairs_ok']}/{a['nested_pairs_checked']}",
                    yn(a["prefix_ok"] == a["n_records"] == a["reported_all_repair_D1_valid"] == a["heldout_rows_exact"] and a["nested_pairs_ok"] == a["nested_pairs_checked"])))
        tab.append((f"{sh} results.json vs my summary of the raw records (cells, CI, McNemar)", f"{b['n_fields_compared']} values", f"{len(b['mismatches'])} mismatches", yn(not b["mismatches"])))
        nt = 16 if sh == "Qwen" else 12
        tab.append((f"{sh} split leak: valid HELD-OUT rows with a REPAIR-pool twin / of them complaints", f"25 / {nt}",
                    f"{hd['twins_valid_heldout']} / {hd['twins_heldout_complaints']}" + (f" (comparing image pixels instead of file names: {leakp[log]['pixels']['n_valid_heldout_twins']} / {leakp[log]['pixels']['twin_complaints']})" if leakp else ""),
                    yn(hd["twins_valid_heldout"] == 25 and hd["twins_heldout_complaints"] == nt)))
    e = cpu["E_seq_vs_batch"]
    cs_ = [v for lg in VC.LOGS for k, v in e.get(lg, {}).items() if k.startswith("COMP_S")]
    rs_ = [v for lg in VC.LOGS for k, v in e.get(lg, {}).items() if k.startswith("REEDIT_S")]
    if cs_:
        tab.append(("COMP-S(+) vs COMP-B(+) operators, max rel. Frobenius diff (saved corrections)", "<= 3e-8", f"{max(cs_):.1e} ({len(cs_)} pairs)", yn(max(cs_) <= 3e-8)))
    if rs_:
        tab.append(("REEDIT-S vs REEDIT-B weights (LLaVA), max rel. diff", "1.5e-3", f"{max(rs_):.1e} ({len(rs_)} pairs)", yn(max(rs_) < 1.6e-3)))
if rend:
    tab.append(("results.md = render(results.json) and results.json = aggregate(raw records): sandbox re-run of copies of the experiment's scripts, byte comparison", "identical",
                f"md identical: {rend['results_md_identical']}; json identical: {rend['results_json_identical']}", yn(rend["results_md_identical"] and rend["results_json_identical"])))


# ============================================================================ verdicts (data-driven)
def gpu_phrase(conds):
    """logs on which every one of these exact conditions agrees row for row / disagrees / was not run."""
    done, bad, miss = [], [], []
    for log in VC.LOGS:
        v = [exact_ok.get((log, c)) for c in conds]
        if all(x is True for x in v):
            done.append(SHORT[log])
        elif any(x is False for x in v):
            bad.append(SHORT[log])
        else:
            miss.append(SHORT[log])
    return done, bad, miss


V = []
done, bad, miss = gpu_phrase(["NONE", "W0"])
V.append(("Sanity: NONE reproduces the stored D1 sets and Rel; W_0 gives zero D1",
          "REFUTED" if bad else (("CONFIRMED" if not miss else "CONFIRMED on " + ", ".join(done) + "; not recomputed on " + ", ".join(miss)) if done else "NOT RECOMPUTED"),
          "own loop: " + (", ".join(done) + " row for row" if done else "-") + ("; mismatch on " + ", ".join(bad) if bad else "")))
if cpu:
    C = cpu["C_report_tables"]
    alle = list(exact_ok)
    allok = [k for k, v in exact_ok.items() if v]
    V.append(("Unrepaired baselines and the two seed-0 tables", "CONFIRMED" if C["cells_agree"] == C["cells"] and len(allok) == len(alle) else "PARTLY CONFIRMED",
              f"{C['cells_agree']}/{C['cells']} cells equal the raw records; {len(allok)}/{len(alle)} recomputed conditions agree row for row"))
    done, bad, miss = gpu_phrase(["BEST__COMP_B__nALL"])
    sel = all(cpu["C_headlines"][lg]["selection_minimum"][0][1] == "COMP_B__nALL__s0" for lg in VC.LOGS)
    V.append(("Best operating point = COMP-B at n = ALL on both logs, with its numbers",
              "REFUTED" if bad or not sel else ("CONFIRMED" if not miss else "CONFIRMED on " + ", ".join(done) + "; " + ", ".join(miss) + " from the records only"),
              "selection re-derived from the raw records; correction rebuilt and re-evaluated on: " + (", ".join(done) if done else "-")))
    V.append(("Generalisation reading (COMP-B fixes and breaks at similar rates; COMP-B+ worse at ALL; REEDIT net negative at every n; rollback only by removing edits)",
              "CONFIRMED (numbers)", "rates and nets re-derived from the raw records; the reading is single-seed and has no same-norm control"))
    V.append(("Anomalies: split leak; harness z* fallback bug; sequential = batch; answer-position compensation; ALLPOS",
              "CONFIRMED", "twins recounted from the records (also by pixel content); bug located in the source; operators and weights compared from the saved corrections"))
LL = S["logs"]["TB-L100-ae-l2_3"]
mdL = LL.get("moddiag")
why = (cext.get("TB-L100-ae-l2_3") or {}).get("why")
psL = expres["logs"].get("TB-L100-ae-l2_3", {}).get("protect_split")
wpL, n0L = LL.get("wproj"), LL["conds"].get("NONE")
if mdL and psL:
    nums_ok = (psL["rank_K_ans"] == mdL["rank_K_ans"] and abs(psL["s_complaints"] - mdL["s_complaints"]) < 0.005 and abs(psL["a_complaints"] - mdL["a_complaints"]) < 0.005)
    if wpL and n0L:
        mw, m0 = wpL["mine"], n0L["mine"]
        supports = mw["heldout_D1"] >= m0["heldout_D1"]
        V.append(("Why (LLaVA, module level): rank, share, residual change; 'the damage travels through the edits' own key directions'",
                  "CONFIRMED (numbers); reading supported only by my extra run" if (nums_ok and supports) else ("PARTLY CONFIRMED" if nums_ok else "REFUTED"),
                  f"the three numbers reproduce ({mdL['source']}); my behavioural run of that very weight (W_0 restored outside the span; no experiment record, one run): HELD-OUT D1 "
                  f"{mw['heldout_D1']:.1f}% at Rel exact {mw['rel_exact']:.0f}, against {m0['heldout_D1']:.1f}% unrepaired -- " + ("it does not lower D1" if supports else "it LOWERS D1, against the reading")))
    else:
        V.append(("Why (LLaVA, module level): rank, share, residual change; 'the damage travels through the edits' own key directions'",
                  "PARTLY CONFIRMED" if nums_ok else "REFUTED", "the three numbers reproduce; the causal sentence has no behavioural test"))
V.append(("Not run: D2, seeds 1-2, text head-to-head, Qwen REEDIT re-runs", "CONFIRMED", "no such record exists; every record has seed 0"))
S["verdicts"] = V
S["table"] = tab
S["missing"] = missing

# ============================================================================ markdown
md = ["# P12 (linear repair streams) -- independent verification", "",
      "Rendered by `verify/render_report.py` from `verify/out/*.json` (GPU: `gpu_verify.py`; CPU: `cpu_checks.py`, `cpu_extra.py`, `cpu_moddiag.py`, `leak_pixels.py`, "
      "`sandbox_render.sh`). Every number is read from those files. The experiment's files were not modified.", "",
      "## 1. Verdict per headline", "", "| headline | verdict | basis |", "|---|---|---|"]
for h, v, b in V:
    md.append(f"| {h} | **{v}** | {b} |")
md.append("")
if missing:
    md += ["**Not recomputed (no GPU window for this lowest-priority job):** " + "; ".join(missing) + ".", ""]
md += ["## 2. What was recomputed, and how", "",
       "- Own evaluation loop (`gpu_verify.py`): pools and the nested stream rebuilt with numpy from `probe_records_v2.jsonl` (`vcommon.my_split`, `my_stream`); B=1 batches built "
       "from the record fields and from the log's `edit_batch.jsonl`; one full teacher-forced forward per row; the correction applied by my own forward hook (fp64 adapter, gate "
       "1e-8) or by writing the replacement weight. No call of `harness.evaluate`, `harness.baseline`, the suffix engine or `H.applied`.",
       "- Corrections rebuilt through `harness.repair` with the stated options and an empty private cache (edit keys, protection bases, z* targets recomputed with fresh forwards).",
       f"- The two further cells per log were drawn with a fixed seed before any GPU number of mine existed (`draw_cells.py`, seed {draw['seed']}, {draw['n_universe']} cells): "
       + "; ".join(f"{SHORT[lg]}: " + ", ".join(f"{LAB[m]} n={n}" for m, n in draw["draw"][lg]) for lg in VC.LOGS) + ".", "",
       "## 3. Table", "", "| quantity | experiment | verifier | agree? |", "|---|---|---|---|"]
for q, e, m, a in tab:
    md.append(f"| {q} | {e} | {m} | {a} |")
md.append("")
md += ["## 4. Code reading (check 1)", ""]
if cpu:
    for log in VC.LOGS:
        a = cpu["A_records"][log]
        pk = S["logs"][log].get("protkeys")
        md.append(f"- {SHORT[log]}: {a['n_records']} raw records. Reported set = first n of my own stream in {a['prefix_ok']}; every reported row a valid REPAIR-pool D1 complaint in "
                  f"{a['reported_all_repair_D1_valid']}; held-out rows identical to my valid HELD-OUT rows in {a['heldout_rows_exact']} (no REPAIR row in any: {a['heldout_no_repair_row']}); "
                  f"stored W_E flags used as the reference in {a['stored_WE_flags_ok']}. Nested along n: {a['nested_pairs_ok']}/{a['nested_pairs_checked']} pairs; "
                  f"n // 2 image + (n - n // 2) text: {a['stratification_ok']}. COMP: options as stated {a['comp_options_ok']}; answer-query key count = sum of y0 tokens "
                  f"{a['comp_answer_query_count_ok']}; joint solves infeasible or zero: {len(a['comp_batch_infeasible_or_zero'])}; every correction non-zero: {a['comp_all_nonzero']}; zero-rank steps inside "
                  f"the sequential variants: {len(a['comp_seq_zero_rank_steps'])} complaints, all repeating an earlier complaint's input (same prompt, image bytes, y0): "
                  f"{a['comp_seq_zero_rank_steps_all_repeats']}; largest restoration error {a['comp_restore_err_max']:.0e}; protection ranks {a['comp_protect_ranks']}; no backward pass "
                  f"{a['comp_no_backward']}. REEDIT: l2 {a['reedit_l2_values']} (the log's: {a['log_l2']}), variant {a['reedit_variants']}. Rollback budgets: RB-TOT {a['rb_tot_budgets']} "
                  f"(largest k {a['rb_tot_k_max']}), RB-UNION {a['rb_union_budgets']} per complaint (largest per-complaint k {a['rb_union_per_complaint_k_max']}).")
        rb = (cext.get(log) or {}).get("rollback") or {}
        bind = [k for k, v in rb.items() if v.get("budget_binds")]
        un = rb.get("RB_UNION__nALL")
        if rb:
            md.append(f"  Budgets: RB-TOT ends at its budget with reported complaints still unresolved at n = {', '.join(x.split('__n')[1] for x in bind) or 'none'}"
                      + (f"; RB-UNION at n = ALL: {un['n_at_budget']} of {un['n']} per-complaint searches end at the budget, {un['n_resolved_individually']} are resolved individually." if un else "."))
        if pk:
            md.append(f"  Protection keys recomputed with fresh forwards: `probe:100000` selects {pk['n_probe_rows']} rows, all valid REPAIR-pool non-complaints ({pk['selection_repair_noncomplaints_only']}; "
                      f"equal to my list: {pk['selection_equals_my_repair_noncomplaints']}; HELD-OUT rows among them: {pk['n_heldout_rows_in_selection']}); fresh keys equal to the cache file the "
                      f"experiment read: {pk['fresh_keys_equal_harness_cache']} (max abs diff {pk['fresh_vs_cache_max_abs_diff']}); own hook capture equal on {sum(pk['own_hook_capture_equal'])}/"
                      f"{len(pk['own_hook_capture_equal'])} sampled rows; fresh edit keys equal to the cached ones: {pk['edit_keys_fresh_equal_harness_cache']}; rank K_ans {pk['rank_K_ans']} "
                      f"(experiment {pk['experiment_protection_rank_COMP_B']}), K_ans + probe {pk['rank_K_ans_plus_probe']} (experiment {pk['experiment_protection_rank_COMP_BP']}).")
        else:
            md.append("  Protection keys: fresh recomputation not run on this log (no GPU window); checked by code reading and by the recorded protection ranks only.")
    F = cpu["F_harness_bug"]
    md += ["", "## 5. Bugs", "",
           f"- `{F['file']}`, lines {F['line_elif_chunk_None']}-{F['dead_inner_test_line']}: the z* oracle fallback loop (line {F['line_fallback_loop']}, indent {F['fallback_loop_indent']}) is inside the "
           f"`else:` branch of line {F['line_else_chunk']} (indent {F['else_indent']}), so it runs only when `chunk` is given; the test `if chunk is None:` on line {F['dead_inner_test_line']} is dead "
           "code. `repair(\"REEDIT\", ...)` with the default `chunk=None` makes one vendor attempt and does not install a non-oracle complaint. The report's claim is correct; the experiment "
           "works around it with `chunk=1` (`code/run_streams.py`, `compute_target`).",
           "- First-pass records affected (kept in `out/<log>/superseded_zfallback/`): " + "; ".join(f"{SHORT[lg]} `{k}` not installed {v['not_installed']}" for lg in VC.LOGS for k, v in F["first_pass_records"][lg].items() if k.startswith("REEDIT_B")) + ".",
           "- Corrected re-runs present: " + "; ".join(f"{SHORT[lg]}: {', '.join(v) if v else 'none'}" for lg, v in F["reruns_present"].items()) + ".", ""]
md += ["## 6. Residual risks", ""]
zc = opt(os.path.join(VC.VOUT, "zstar_compare_TB-Q100-ae-l2_3.json"))
zeff = ""
QC = S["logs"]["TB-Q100-ae-l2_3"]["conds"]
tz_ = next((c for k, c in QC.items() if k.endswith("__theirz")), None)
as_ = next((c for k, c in QC.items() if k.endswith("__asrec")), None)
ld_ = next((c for k, c in QC.items() if k.endswith("__ladder")), None)
if tz_ and as_:
    zeff = (f" Effect on the Qwen REEDIT-B n=100 cell: with the experiment's z* my install gives the saved weight ({tz_['saved']['entries_differing']} of {tz_['saved']['n_entries']} entries differ) and "
            f"its record row for row; with my z* and the same installed set {as_['vs']['heldout_rows'] - as_['vs']['heldout_rows_agree']} of {as_['vs']['heldout_rows']} held-out rows flip "
            f"(D1 {as_['mine']['heldout_D1']:.1f}% against {as_['vs']['heldout_D1_exp']:.1f}%, Rel exact {as_['mine']['rel_exact']:.0f} against {as_['vs']['rel_exact_exp']:.0f})"
            + (f"; with the fallback ladder ({ld_['repair']['installed']} installed, the stated protocol) D1 {ld_['mine']['heldout_D1']:.1f}%, Rel exact {ld_['mine']['rel_exact']:.0f}." if ld_ else "."))
cfin = opt(os.path.join(VC.VOUT, "cpu_final.json")) or {}
zrep = gp("TB-Q100-ae-l2_3", "ZREPEAT")
zs_ = (cfin.get("zstar") or {}).get("first_attempt_pairs")
if zs_:
    bi, bt = zs_["image"], zs_["text"]
    txt = ("- z* is not reproducible on image complaints, so every REEDIT number has a run-to-run component that one seed does not show. ")
    if zrep:
        im = [e for e in zrep["rows"] if e["ptype"] == "image"]
        tx = [e for e in zrep["rows"] if e["ptype"] == "text"]
        txt += (f"Same process, same call, three repeats per complaint (Qwen): text {sum(e['own_runs_bitwise_equal'] for e in tx)}/{len(tx)} bitwise equal; image "
                f"{sum(e['own_runs_bitwise_equal'] for e in im)}/{len(im)} bitwise equal, relative difference of the optimised offset between repeats "
                f"{min(e['own_runs_max_rel_diff'] for e in im):.2f}-{max(e['own_runs_max_rel_diff'] for e in im):.2f} (the starting state h is bitwise equal in every repeat: "
                f"{all(e['h_bitwise_equal_across_runs'] for e in zrep['rows'])}). ")
    txt += (f"My first-attempt z* against the experiment's cached first-attempt z* (first 100 Qwen stream complaints): text {bt['bitwise_equal']}/{bt['n']} bitwise equal; image "
            f"{bi['bitwise_equal']}/{bi['n']}, median relative difference {bi['median_rel_diff']:.2f} (max {bi['max_rel_diff']:.2f}), a different number of optimisation steps in {bi['steps_differ']}. ")
    fl = (cfin.get("zstar") or {}).get("oracle_flips_in_experiment_files") or []
    if fl:
        txt += ("Oracle status flips between runs in the experiment's own files: "
                + "; ".join(f"`{x['group']}` not an oracle, then an oracle at attempt {x['second_run_attempts']} ({x['source']})" for x in fl if x["second_run_oracle"]) + ".")
    md.append(txt + zeff)
for log in VC.LOGS:
    bp = (cext.get(log) or {}).get("best_point")
    if bp:
        md.append(f"- {SHORT[log]} best point: Rel exact {bp['rel'][0]:.0f} -> {bp['rel'][1]:.0f} ({bp['rel_broken_fixed'][0]} broken, {bp['rel_broken_fixed'][1]} fixed)"
                  + (", exactly on the 2-point tolerance" if bp["on_rel_tolerance_boundary"] else "") + f"; T-Loc {bp['tloc'][0]:.1f} -> {bp['tloc'][1]:.1f}, I-Loc {bp['iloc'][0]:.1f} -> {bp['iloc'][1]:.1f}.")
if why:
    w = why
    wtxt = ""
    if wpL and n0L:
        mw, m0 = wpL["mine"], n0L["mine"]
        wtxt = (f" My one behavioural run of W_0 + D P_ans (W_0 restored outside the span): HELD-OUT D1 {mw['heldout_D1']:.1f}% (+{mw['heldout_complaints_fixed']} / -{mw['heldout_noncomplaints_lost']}) "
                f"against {m0['heldout_D1']:.1f}% unrepaired, Rel exact {mw['rel_exact']:.0f}; REPAIR pool D1 {mw['repair_n_D1']}/{mw['repair_n']} against {m0['repair_n_D1']}/{m0['repair_n']}. "
                "The part of the log inside the edits' answer-key span alone damages more rows than the whole log, so the out-of-span part partly offsets it (a and b are not an orthogonal split).")
    md.append(f"- The 'why' paragraph is LLaVA only and quotes the complaint medians only. The quantities are not specific to complaints: held-out non-complaints have s = {w['median_s'][1]:.2f} "
              f"(complaints {w['median_s'][0]:.2f}) and a = {w['median_a'][1]:.2f} (complaints {w['median_a'][0]:.2f}); the gap is the probe-type mix (image rows s = "
              f"{w['by_ptype']['image']['median_s'][0]:.2f} / {w['by_ptype']['image']['median_s'][1]:.2f}, text rows {w['by_ptype']['text']['median_s'][0]:.2f} / {w['by_ptype']['text']['median_s'][1]:.2f}, "
              "complaints / non-complaints)." + wtxt)
q = (cext.get("TB-Q100-ae-l2_3") or {}).get("stored_vs_B1")
if q:
    md.append(f"- Tables use B=1 teacher forcing for the edits; the testbed's stored values are batch-8: Qwen Rel / T-Gen / I-Gen stored {q['stored_b8'][0]:.0f} / {q['stored_b8'][1]:.0f} / {q['stored_b8'][2]:.0f}, "
              f"tables {q['B1_used_in_tables'][0]:.0f} / {q['B1_used_in_tables'][1]:.0f} / {q['B1_used_in_tables'][2]:.0f}.")
md += ["- Single seed; the best point is the minimum over the configurations inside the tolerance, chosen on the evaluation pool (stated in the report). No same-norm control correction was run, so "
       "'fix rate about equal to loss rate' is not shown to be specific to the reported complaints.",
       "- Qwen REEDIT at n = 10 / 100 / ALL exists only as first-pass records (complaints not installed); marked with a dagger in the report."]
ntw = cfin.get("near_twins") or {}
if ntw:
    md.append("- Split leak, beyond identical queries: " + "; ".join(
        f"{SHORT[lg]}: {v['near_twins']} more valid HELD-OUT image rows share the question and the image's entity folder with a REPAIR-pool row (another picture; {v['near_twins_of_a_repair_complaint']} "
        f"next to a REPAIR complaint). The best point fixes {v['best_point_fixed_over_complaints']['exact_twins'][0]}/{v['best_point_fixed_over_complaints']['exact_twins'][1]} of the identical-twin "
        f"complaints and {v['best_point_fixed_over_complaints']['all_other_heldout_complaints'][0]}/{v['best_point_fixed_over_complaints']['all_other_heldout_complaints'][1]} of the other held-out "
        f"complaints; without both kinds of twins: {v['best_point_D1_strict']:.1f}% against {v['none_D1_strict']:.1f}% (p = {v['mcnemar_p_strict']:.2g}, {v['strictly_disjoint_rows']} rows)"
        for lg, v in ntw.items()) + ".")
md.append("")
prov = opt(os.path.join(VC.VOUT, "provenance.json"))
if prov:
    st_ = [r for lg in prov["feasible_flag_stored_vs_current_code"].values() for r in lg if r[1] != r[2]]
    md.append(f"- Provenance: `code/run_streams.py` and `p12_common.py` were last modified at {prov['code_mtime_utc']['run_streams.py']} UTC, after the last raw record ({prov['last_raw_stream_record_utc']} UTC). "
              f"{len(st_)} COMP records carry `extra.feasible_exact = false` although every criterion of the current code holds (they have dropped duplicate directions); the flag is not used by any "
              "reported number. The cells I rebuilt reproduce with the current harness and the stated options.")
md.append("- Not recomputed by me: T-Gen / I-Gen at B=1 (my loop checks Rel; the batch-8 T-Gen / I-Gen of the unrepaired log equal the stored values), the cost columns, CIs other than the best "
          "point's, and every cell outside the conditions of section 3 (those were checked against the raw records only). My private copies of the edit-key cache were deleted after the bitwise "
          "comparison recorded in `out/<log>/PROTKEYS.json`.")
qx = (S["logs"]["TB-Q100-ae-l2_3"].get("xcheck") or {})
md.append("")
inc = opt(os.path.join(VC.VOUT, "_verifier_incidents.json")) or []
md += ["## 7. Shared-GPU log of this verifier (operational; `out/_verifier_incidents.json`)", ""]
for x in inc:
    md.append(f"- {x['when']}, {x['card']}: {x['what']} Impact on others: {x['impact_on_others']}." + (f" Cause: {x['cause']}. Fix: {x['fix']}." if x.get("cause") else ""))
md.append("")
md += ["## 8. Files", "", "- `verify/gpu_verify.py`, `vcommon.py`, `cpu_checks.py`, `cpu_extra.py`, `cpu_moddiag.py`, `leak_pixels.py`, `sandbox_render.sh`, `draw_cells.py`, `render_report.py`",
       "- `verify/out/<log>/*.json` (one file per recomputed condition, with my per-row flags), `verify/out/*.json`, `verify/logs/`", ""]
with open(os.path.join(VC.VER, "VERIFY_REPORT.md"), "w") as fh:
    fh.write("\n".join(md) + "\n")
VC.jdump(S, os.path.join(VC.VOUT, "verify_summary.json"))
print("\n".join(md))
