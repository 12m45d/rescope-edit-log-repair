"""E2 on prefix logs: out/table.txt rendered from out/summary.json, out/h1_prefix.json and out/status.json (no number typed by hand).

    python render.py
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcommon import *  # noqa: F401,F403,E402
from pcommon import fx  # noqa: E402

LINM = ("comp_b", "comp_bp", "reedit_b", "rb_tot", "rb_union")


def f1(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def stat(g, nd=1, seeds=True):
    """'mean [lo, hi] (a/b/c)': the interval is the unit bootstrap (1 seed) or the crossed bootstrap over seeds / draws x units."""
    if g is None:
        return "-"
    s = f1(g["mean"], nd)
    if g.get("ci_lo") is not None:
        s += f" [{f1(g['ci_lo'], nd)}, {f1(g['ci_hi'], nd)}]"
    ps = g.get("per_seed") or []
    if seeds and len(ps) > 1:
        s += " (" + "/".join(f1(x, nd) for x in ps) + ")"
    return s


def dsum(g, nd=1):
    """Gate level over calibration draws: 'mean [p05, p95]'."""
    if g is None:
        return "-"
    if g.get("n_draws", len(g.get("values", []))) <= 1 or g.get("p05") is None:
        return f1(g["mean"], nd)
    return f"{f1(g['mean'], nd)} [{f1(g['p05'], nd)}, {f1(g['p95'], nd)}]"


def near(r):
    nr = r.get("near")
    if not nr:
        return "n.d."
    return " / ".join("-" if nr.get(t) is None else f1(nr[t]["mean"]) for t in ("HN-I", "HN-Q", "HN-X"))


def trip(r, a, b, c, nd=0):
    return " / ".join(f1(r[k]["mean"], nd) if r.get(k) else "-" for k in (a, b, c))


def main():
    summ = fx.io.jload(os.path.join(OUT, "summary.json"))
    h1 = fx.io.jload(os.path.join(OUT, "h1_prefix.json"))
    status = fx.io.jload(os.path.join(OUT, "status.json"))
    meta = fx.io.jload(os.path.join(OUT, "results_meta.json"))
    L = []
    w = L.append
    w(f"E2 ON PREFIX LOGS (amendment R12, reduced by R19 d)      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/render.py from out/summary.json, h1_prefix.json, status.json")
    w(f"driver {meta['code']}; one GPU job per log (code/job.py), gate level by code/gate.py; every number is recomputed from out/rows/*.jsonl.gz by code/aggregate.py")
    w("")
    w("WHAT THE ROWS ARE")
    w("  prefix log = the development stream after its first 10 / 20 / 50 edits (pre-edit weight + the stored batch updates of the first 1 / 2 / 5 batches; driver: fx.prefix).")
    w("  complaints = D1 complaints of the REPAIR pool under the prefix's deployed weight; nested sets fx.draws.complaints(L, n, seed, n_base=20); n in {1, 10, 20, all};")
    w("               3 seeds for n <= 20, seed 0 for all.  Every repair of a cell receives the same complaints.")
    w("  COMP-B / COMP-B+ / REEDIT-B / RB-TOT / RB-UNION: the shared harness with the calls and options of P12 (run_streams.make; fp64 solves on the CPU), built once, saved")
    w("               (out/corr/), and re-scored through the driver at B = 1.  blocklist: the deployed log + an exact-match list of the reported queries (ses.blocklist).")
    w("  ReScope-W  = thresholds-only repair (all positions, logged values, linear response, no round-off term): log-only thresholds (n = 0), + least repair of the n")
    w("               complaints (library default: a complaint whose repair would silence an own key is refused), or two-block thresholds from n = 200 benign REPAIR queries")
    w("               (R24: library rule on the per-row maxima over all positions; alpha = 0.05).  ReScope(0.05) = the default method (migration; last prompt position, REPL).")
    w("COLUMNS")
    w("  D1 hd      = % of the valid heldout_disjoint rows whose pre-edit answer is not reproduced (teacher forcing, B = 1).  'mean [lo, hi] (per seed)': one seed: unit bootstrap")
    w("               over query clusters; 3 seeds / 3 calibration draws: crossed bootstrap over seeds (draws) x query clusters, 2000 resamples -- WITH 3 SEEDS / DRAWS THE")
    w("               INTERVAL IS TOO NARROW WHEN THE SEED COMPONENT MATTERS (R25 (3)); the values per seed / draw are listed in brackets.")
    w("  D1 strict  = the same on heldout_strict.   fired hd = % of heldout_disjoint rows on which an edit is released (gates only; behaviour rows).")
    w("  near HN-I/HN-Q/HN-X = % of the valid hard negatives of the prefix's edits on which an edit is released (GATE LEVEL; mean over seeds / over 20 calibration draws).")
    w("  BEN-T      = false firing % on BEN-T at gate level (fx.pools.firing, PROMPT-ONLY keys; calibration at teacher-forced REPAIR keys), mean [5th, 95th] over 20 draws;")
    w("               n.d. = not defined (a repaired weight has no firing event; ReScope-W needs every position, the pool holds the last-prompt key only).")
    w("  Rel/T-Gen/I-Gen = exact % over the prefix's edits (teacher forcing, B = 1; mean over seeds).  Rel k/g/l = edits kept / gained / lost against the deployed log, % of edits (R7).")
    w("  +fix/-lost = held-out complaints fixed / held-out non-complaints lost against the deployed log (mean over seeds).  rep. res. = % of the reported complaints resolved.")
    w("  D2 (R7) is measured for a fixed subset of rows only (line 'D2' under each table; prefix probe records carry no D2 fields, R25 (5)).  Part (e) (T = 200 logs): not run (R19 d).")
    w("")
    for log in LOGS:
        S = summ.get(log)
        st = status.get(log, {})
        w("#" * 200)
        if S is None:
            w(f"{log}: NOT RUN YET ({st.get('missing')})")
            continue
        dep = S["rows"][0]
        w(f"{log}   T = {S['T']} edits; deployed: D1 hd {f1(dep['hd_D1']['mean'], 2)} %, Rel/T-Gen/I-Gen {trip(dep, 'rel', 'tgen', 'igen')}; REPAIR complaints (n = all) {S['n_repair_complaints']}; "
          f"heldout_disjoint {S['pools']['heldout_disjoint']} rows, heldout_strict {S['pools']['heldout_strict']}; valid hard negatives of its edits {S['hn_valid']}")
        w("#" * 200)
        head = ("method", "n", "seeds", "D1 hd % [CI] (per seed)", "D1 strict %", "fired hd %", "near HN-I/HN-Q/HN-X", "BEN-T", "Rel/T-Gen/I-Gen", "Rel k/g/l", "+fix/-lost", "rep. res. %", "notes")
        rows = []
        for r in S["rows"]:
            notes = []
            if r["method"] in LINM:
                if "edits_rolled_back" in r:
                    notes.append("rolled back " + "/".join(str(x) for x in r["edits_rolled_back"]) + f" of {S['T']}")
                if "installed" in r:
                    notes.append("installed " + "/".join(str(x) for x in r["installed"]) + (" (not: " + "/".join(str(x) for x in r["not_installed"]) + ")" if any(r["not_installed"]) else ""))
                notes.append("norm/log " + "/".join(f1(x, 2) for x in r["fro_rel_to_log"]))
                if r["correction_kind"] != ["adapter"] and r["correction_kind"] != ["weight"]:
                    notes.append("kind " + ",".join(r["correction_kind"]))
            if r["method"] == "where_least":
                oc = r["order_check"]
                notes.append(f"refused {'/'.join(str(x) for x in r['n_refused'])}; orders {oc['n_equal_thresholds']}/{oc['n_orders']} same thresholds; own out {'/'.join(str(x) for x in r['own_prompts_not_candidate'])}; "
                             f"hook!=gate {r['hook_vs_gate_n_differ']}/{r['hook_vs_gate_n_rows']}")
            if r["method"] == "where_logonly":
                notes.append(f"own out {'/'.join(str(x) for x in r['own_prompts_not_candidate'])}; hook!=gate {r['hook_vs_gate_n_differ']}/{r['hook_vs_gate_n_rows']}")
            if r["method"] in ("where_two_block", "rescope"):
                g = r.get("gate20")
                if g:
                    notes.append(f"gate level, 20 draws: fired hd {dsum(g['hd_fired'], 2)}, literal rule {dsum(g['literal_hd_fired'], 2)} (bound {f1(g['bound']['mean'], 2)}), capped edits {dsum(g['n_capped'])} "
                                 f"max {int(max(g['n_capped']['values']))}, T-Gen/I-Gen released {f1(g['tgen_own']['mean'])}/{f1(g['igen_own']['mean'])}")
                if r["method"] == "where_two_block":
                    notes.append(f"hook!=gate {r['hook_vs_gate_n_differ']}/{r['hook_vs_gate_n_rows']}")
            ben = dsum(r["ben_t"]["fired"], 2) if r.get("ben_t") else ("-" if r["method"] in ("deployed", "preedit") else "n.d.")
            rows.append((r["label"], str(r.get("n_reported", r["n"])) if r["n"] != "ALL" else f"all={r['n_reported']}", str(r["n_seeds"]) if r["method"] not in ("deployed", "preedit", "where_logonly") else "-",
                         stat(r["hd_D1"], 1), stat(r["strict_D1"], 1, seeds=False), stat(r["hd_fired"], 1, seeds=False) if r.get("hd_fired") else "-", near(r) if r["method"] not in LINM + ("deployed", "preedit", "blocklist") else "-",
                         ben, trip(r, "rel", "tgen", "igen"), trip(r, "rel_kept", "rel_gained", "rel_lost"),
                         f"+{f1(r['fixed']['mean'], 0)}/-{f1(r['lost']['mean'], 0)}" if r.get("fixed") else "-", f1(r["reported_resolved"]["mean"], 0) if r.get("reported_resolved") else "-", "; ".join(notes)))
        w(fx.io.text_table(head, rows, align=["l"] + ["r"] * 11 + ["l"]))
        fl = S.get("where_floor")
        if fl:
            w(f"  round-off floor of ReScope-W (every threshold at minus infinity = the deployed log without its write round-off term): flags that differ from the deployed weight: "
              f"D1 on heldout_disjoint {fl['heldout_disjoint']['n_differ']}/{fl['heldout_disjoint']['n']} (D1 {f1(fl['heldout_disjoint']['D1_floor'])} %), "
              + ", ".join(f"{m} exact {fl[m]['n_differ']}/{fl[m]['n']} (exact {fl[m]['n_exact_floor']} against {fl[m]['n_exact_deployed']} deployed)" for m in ("rel", "tgen", "igen")))
        if S.get("d2"):
            w("  D2 (R7): % of the pre-correct heldout_disjoint rows whose 48-token answer loses the reference [unit bootstrap over query clusters]; seed 0 / draw 0 cells only: "
              + "; ".join(f"{d['label']} {f1(d['D2']['value'])} [{f1(d['D2']['ci_lo'])}, {f1(d['D2']['ci_hi'])}] ({d['n_D2']}/{d['n_pre_correct']})" for d in S["d2"] if d["D2"]))
        eq = S.get("where_two_block_equals_driver_constructor")
        if eq:
            w(f"  ReScope-W two-block, draw 0: thresholds from the captured vectors == the driver's own constructor (fx.methods.rescope(role='all', fire='all', response='lin')): bitwise {eq['thresholds_bitwise']}, "
              f"largest difference {eq['max_abs_diff']:.2e}")
        for j in S.get("jobs_clean", []):
            w(f"  job clean: hook_mismatch {j['hook_mismatch']}, memo conflicts {j['memo_conflicts']}, forwards {j['forwards']}, {j['seconds'] / 3600:.2f} h, peak GPU {j['peak_gpu_gib']:.2f} GiB, phases {','.join(j['phases'])}")
        if st.get("missing"):
            w(f"  MISSING CELLS ({len(st['missing'])}): {', '.join(st['missing'][:40])}")
        if st.get("failed"):
            for k, v in st["failed"].items():
                w(f"  FAILED CELL {k} (attempts {v['attempts']}): {v['error'][:220]}")
        w("")
    w("#" * 200)
    w("H1 (R6), PREFIX PART: thresholds-only repair (log-only thresholds + least repair) against the best linear repair of the same complaints; held-out D1 on heldout_disjoint")
    w("#" * 200)
    w("  best any = the linear repair with the lowest mean held-out D1 (selected on the evaluation pool, in favour of the baseline); best rel = the lowest among those whose Rel is not more than")
    w("  2 points below the deployed log (the selection of e2_curves).  diff = ReScope-W minus linear, points, paired crossed bootstrap over 3 seeds x units [95 %; 3 seeds: too narrow where the")
    w("  seed component matters], p two-sided.  Rule (literal): helps iff D1 <= 10 and |Rel - Rel deployed| <= 2; the one-sided reading (Rel >= Rel deployed - 2) is printed where it differs.")
    head = ("log", "n", "deployed D1 / Rel", "ReScope-W D1 [CI] (per seed)", "Rel", "best any: D1, Rel", "diff D1 [CI], p", "best rel: D1, Rel", "diff D1 [CI], p", "diff Rel [CI]", "D1<=10", "Rel within 2", "helps")
    rows = []
    for key, c in h1["cells"].items():
        if c.get("status") != "complete":
            rows.append((c["log"], str(c["n"]), f"{f1(c['deployed']['hd_D1'])} / {f1(c['deployed']['rel'], 0)}", c.get("status", "not run")) + ("-",) * 9)
            continue
        wl = c["where_least"]

        def best(tag):
            b = c.get(tag)
            if not b:
                return "none", "-", "-"
            d = b["diff_D1_where_minus_linear"]
            dr = b["diff_rel_where_minus_linear"]
            return (f"{b['method']}: {f1(b['hd_D1']['mean'])}, {f1(b['rel']['mean'], 0)}", f"{f1(d['value'])} [{f1(d['ci_lo'])}, {f1(d['ci_hi'])}], p {d['p_two_sided']:.4f}",
                    f"{f1(dr['value'])} [{f1(dr['ci_lo'])}, {f1(dr['ci_hi'])}]")
        a, b = best("best_any"), best("best_rel")
        v = c["verdict"]
        rows.append((c["log"], str(c["n"]), f"{f1(c['deployed']['hd_D1'])} / {f1(c['deployed']['rel'], 0)}", stat(wl["hd_D1"]), f1(wl["rel"]["mean"], 0), a[0], a[1], b[0], b[1], b[2],
                     "yes" if v["D1_at_most_10"] else "no", "yes" if v["rel_within_2_points"] else "no",
                     ("YES" if v["helps_on_this_log"] else "NO") + ("" if v["helps_on_this_log"] == v["one_sided_reading_met_on_this_log"] else " (one-sided reading: " + ("yes" if v["one_sided_reading_met_on_this_log"] else "no") + ")")
                     + ("" if v["counts_for_the_endpoint"] else " (n=1: extra)")))
    w(fx.io.text_table(head, rows, align=["l"] + ["r"] * 12))
    w(f"  prefix part of the endpoint (n = 10): complete cells {h1['n_prefix_cells_complete_n10']} of 6; prefixes on which the rule holds: {h1['prefixes_that_help_n10']}; "
      f"at least one mild prefix helps: {h1['at_least_one_mild_prefix_helps']}")
    w(f"  {h1['note']}")
    w("")
    w("#" * 200)
    w("ONE TABLE OVER THE PREFIXES AND THE FULL DEVELOPMENT LOGS (n = 10 reported complaints for the complaint-driven columns; means over 3 seeds / 3 calibration draws)")
    w("#" * 200)
    head = ("log", "T", "deployed D1 | Rel/T-Gen/I-Gen", "best linear (any): D1 | Rel/T-Gen/I-Gen", "best linear (Rel within 2): D1 | Rel/T-Gen/I-Gen", "blocklist D1",
            "ReScope-W log-only: D1 | fired | Rel/T-Gen/I-Gen", "ReScope-W + least repair: D1 | fired | near | Rel/T-Gen/I-Gen", "ReScope-W two-block: D1 | fired | near | Rel/T-Gen/I-Gen",
            "ReScope(0.05): D1 | fired | near | Rel/T-Gen/I-Gen")
    rows = []
    full = summ["_full_development_logs"]
    for fam, dev in (("Q", "TB-Q100-ae-l2_3"), ("L", "TB-L100-ae-l2_3")):
        for log in [x for x in LOGS if x.startswith(dev)]:
            S = summ.get(log)
            if S is None:
                rows.append((log, "", "not run") + ("-",) * 7)
                continue
            by = {(r["method"], r["n"]): r for r in S["rows"]}
            dep = by[("deployed", "-")]
            c = h1["cells"].get(f"{log}|n10", {})

            def lin(tag):
                b = c.get(tag)
                if not b:
                    return "none" if c.get("status") == "complete" else "-"
                r = by[(b["method"], 10)]
                return f"{b['method']}: {f1(r['hd_D1']['mean'])} | {trip(r, 'rel', 'tgen', 'igen')}"

            def gate(key, with_near=True):
                r = by.get(key)
                if r is None:
                    return "-"
                s = f"{f1(r['hd_D1']['mean'])} | {f1(r['hd_fired']['mean']) if r.get('hd_fired') else '-'}"
                if with_near:
                    s += f" | {near(r)}"
                return s + f" | {trip(r, 'rel', 'tgen', 'igen')}"
            bl = by.get(("blocklist", 10))
            rows.append((log, str(S["T"]), f"{f1(dep['hd_D1']['mean'])} | {trip(dep, 'rel', 'tgen', 'igen')}", lin("best_any"), lin("best_rel"), f1(bl["hd_D1"]["mean"]) if bl else "-",
                         gate(("where_logonly", 0), False), gate(("where_least", 10)), gate(("where_two_block", 200)), gate(("rescope", 200))))
        f = full[dev]
        t3 = lambda x: " / ".join(f1(x.get(k), 0) for k in ("rel", "tgen", "igen"))                     # noqa: E731
        la = f"{f['best_any']['method']}: {f1(f['best_any']['hd_D1'])} | {t3(f['best_any'])}" if f.get("best_any") else "-"
        lr = f"{f['best_rel']['method']}: {f1(f['best_rel']['hd_D1'])} | {t3(f['best_rel'])}" if f.get("best_rel") else "none"
        wl = f.get("where_least_n10")
        rows.append((dev + " (full) *", "100", f"{f1(f['deployed']['hd_D1'])} | {t3(f['deployed'])}", la, lr, "see e2_curves", "see e2_curves",
                     "see e2_curves" if not wl else f"{f1(wl['hd_D1'])} | see e2_curves | {f1(wl['rel'], 0)} / - / -", "see e2_curves",
                     f"{f1(f['rescope']['hd_D1'])} | {f1(f['rescope']['hd_fired'])} | see e0 | {t3(f['rescope'])}"))
    w(fx.io.text_table(head, rows, align=["l"] + ["r"] * 9))
    w("  * full development logs, from files of other experiments (read, not re-run here): deployed = the driver's all-batches construction check (B = 1); best linear repairs and the")
    w("    thresholds-only repair at n = 10 (3 seeds) = " + full["TB-Q100-ae-l2_3"]["where_least_source"] + ";")
    w("    linear: " + full["TB-Q100-ae-l2_3"]["linear_source"] + ";")
    w("    default method = sanity run e0, M1, draws 0 - 2; blocklist, log-only and two-block cells of the full logs: e2_curves/out (results.json, table.txt), not copied here.")
    for dev in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
        wl = full[dev].get("where_least_n10")
        if wl:
            w(f"    H1 on {dev} as e2_curves wrote it: literal rule {'met' if wl['helps_literal'] else 'NOT met'} (Rel minus deployed {f1(wl['rel_minus_deployed'])}), one-sided reading {'met' if wl['helps_one_sided'] else 'NOT met'}.")
    w("  near = firing % on HN-I / HN-Q / HN-X of the log's edits at gate level; fired = % of heldout_disjoint rows on which an edit is released (behaviour rows).")
    w("")
    w(summ["_ben_t"])
    w(f"E2 part (e): {summ['_e2_part_e']['status']} -- {summ['_e2_part_e']['cause']}")
    tm = {log: (summ[log].get("timing") or {}) for log in LOGS if log in summ}
    if tm:
        w("measured cost on the prefix logs (seconds per phase; A = harness builds, B = re-scoring of 50 corrections, C = capture, D = thresholds-only cells, F / G = calibrated rows): "
          + "; ".join(f"{log.split('l2_3-')[1]}{'Q' if '-Q' in log else 'L'}: " + ", ".join(f"{ph} {v['seconds']:.0f}" for ph, v in sorted(t.items())) for log, t in tm.items()))
    fx.io.Results(EXPERIMENT, OUT).table("\n".join(L))


if __name__ == "__main__":
    main()
