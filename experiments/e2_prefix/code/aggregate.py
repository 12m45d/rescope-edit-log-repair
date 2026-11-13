"""E2 on prefix logs: aggregate the cells and per-row files into the common schema, the H1 read-out and the rendered table.

    python aggregate.py            (CPU; no log is opened: every number is recomputed from out/rows/*.jsonl.gz and out/cells/)

Writes out/results.json (+ results_meta.json), out/summary.json, out/h1_prefix.json, out/status.json, out/paper_numbers.json,
out/paper_numbers_doc.json and out/table.txt (rendered from the JSON files just written; no number is typed by hand).

Conventions of the records (fx.io schema): for complaint-driven cells (linear repairs, where_least, blocklist) `n` = number of
reported complaints, `draw` = seed of the complaint stream, `variant` = 'n1' | 'n10' | 'n20' | 'nALL'; for calibrated rows
(rescope, where_two_block) `alpha` = 0.05, `n` = 200 benign queries, `draw` = calibration draw.  Summaries over seeds / draws:
metric@mean (ci = 5th / 95th percentile over seeds or draws) and metric@crossed (crossed bootstrap over seeds or draws x units;
units = query clusters for probe metrics, edits for edit metrics; R10); a single-seed cell carries the unit bootstrap."""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, needs=("experiments/e2_prefix/out/cells", "experiments/e2_prefix/out/rows"))

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcommon import *  # noqa: F401,F403,E402
from pcommon import fx, np  # noqa: E402

M, ST = fx.metrics, fx.stats
DEV = {"Q": "TB-Q100-ae-l2_3", "L": "TB-L100-ae-l2_3"}
E2C = os.path.join(FIN, "e2_curves", "out")


def hd(rows):
    return [r for r in rows if r["kind"] == "probe" and r["pool"] != "repair"]


def rp(rows):
    return [r for r in rows if r["kind"] == "probe" and r["pool"] == "repair"]


def ed(rows):
    return [r for r in rows if r["kind"] == "edit"]


def cell_stats(rows, dep_rows, reported=True):
    """Numbers of one evaluated cell (one seed / draw): heldout_disjoint, heldout_strict, edits, reported complaints."""
    h = hd(rows)
    if not h or not all(r["disjoint"] and r["y0_valid"] for r in h):
        raise RuntimeError("held-out rows must be valid heldout_disjoint rows")
    ps = M.probe_stats(h, ci=True)
    pstrict = M.probe_stats([r for r in h if r["strict"]], ci=True)
    d0 = {r["csid"]: bool(r["D1"]) for r in hd(dep_rows)}
    gate = h[0].get("fired") is not None
    out = {"heldout_disjoint": {"n": ps["n"], "D1": ps["D1"], "n_D1": ps["n_D1"], "fired": ps["fired"] if gate else None, "T-Loc*": ps["T-Loc*"], "I-Loc*": ps["I-Loc*"],
                                "D1_near": ps["near"].get("D1") if ps["near"]["n"] else None, "D1_far": ps["far"].get("D1") if ps["far"]["n"] else None,
                                "n_near": ps["near"]["n"], "fired_near": ps["near"].get("fired") if gate and ps["near"]["n"] else None,
                                "fired_far": ps["far"].get("fired") if gate and ps["far"]["n"] else None, "n_y0_degenerate": ps.get("n_y0_degenerate"),
                                "n_complaints_deployed": sum(d0.values()), "complaints_fixed": sum(d0[r["csid"]] and not r["D1"] for r in h),
                                "noncomplaints_lost": sum((not d0[r["csid"]]) and bool(r["D1"]) for r in h)},
           "heldout_strict": {"n": pstrict["n"], "D1": pstrict["D1"], "fired": pstrict["fired"] if gate else None}}
    es = M.edit_stats(ed(rows), ci=True)
    kg = M.kept_gained(ed(rows), ed(dep_rows), ci=True)
    out["edits"] = {m: {"n": es[m]["n"], "exact": es[m]["exact"], "token": es[m]["token"], "own_released": es[m].get("own_cand"),
                        "kept": kg[m]["kept"], "gained": kg[m]["gained"], "lost": kg[m]["lost"], "n_kept": kg[m]["n_kept"], "n_gained": kg[m]["n_gained"], "n_lost": kg[m]["n_lost"]}
                    for m in ("rel", "tgen", "igen")}
    out["edits"]["rephrase"] = {"n": es["rephrase"]["n"], "exact": es["rephrase"]["exact"]}
    if reported:
        r_ = rp(rows)
        out["reported"] = {"n": len(r_), "n_resolved": sum(not r["D1"] for r in r_), "resolved": 100.0 * sum(not r["D1"] for r in r_) / len(r_) if r_ else None}
    return out


def val(x):
    return x["value"] if isinstance(x, dict) else x


def crossed(rows_by, value, unit, where):
    g = M.grid(rows_by, value, unit=unit, where=where)
    return ST.crossed_bootstrap(g["num"], g["den"], scale=100.0)


def crossed_diff(rows_a, rows_b, value, unit, where):
    ga, gb = M.grid(rows_a, value, unit=unit, where=where), M.grid(rows_b, value, unit=unit, where=where)
    if ga["units"] != gb["units"] or ga["draws"] != gb["draws"]:
        raise RuntimeError("paired grids need the same seeds and units")
    return ST.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100.0)


IS_HD = lambda r: r["kind"] == "probe" and r["pool"] != "repair"                                       # noqa: E731
IS_STRICT = lambda r: r["kind"] == "probe" and r["pool"] != "repair" and r["strict"]                   # noqa: E731
IS_M = lambda m: (lambda r: r["kind"] == "edit" and r["metric"] == m)                                   # noqa: E731


def group_summary(cells: dict, rows_by: dict) -> dict:
    """Summary over the seeds / draws of one (method, n): means, 5th - 95th percentile, crossed bootstrap, the values per seed."""
    ks = sorted(cells)
    out = {"n_seeds": len(ks), "seeds": ks}

    def per(path):
        v = []
        for k in ks:
            x = cells[k]
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
            v.append(val(x))
        return v
    for nm, path in (("hd_D1", ("heldout_disjoint", "D1")), ("strict_D1", ("heldout_strict", "D1")), ("hd_fired", ("heldout_disjoint", "fired")),
                     ("strict_fired", ("heldout_strict", "fired")), ("hd_D1_far", ("heldout_disjoint", "D1_far")), ("hd_D1_near", ("heldout_disjoint", "D1_near")),
                     ("hd_fired_far", ("heldout_disjoint", "fired_far")), ("hd_fired_near", ("heldout_disjoint", "fired_near")),
                     ("fixed", ("heldout_disjoint", "complaints_fixed")), ("lost", ("heldout_disjoint", "noncomplaints_lost")),
                     ("tloc", ("heldout_disjoint", "T-Loc*")), ("iloc", ("heldout_disjoint", "I-Loc*")),
                     ("rel", ("edits", "rel", "exact")), ("tgen", ("edits", "tgen", "exact")), ("igen", ("edits", "igen", "exact")), ("rephrase", ("edits", "rephrase", "exact")),
                     ("rel_kept", ("edits", "rel", "kept")), ("rel_gained", ("edits", "rel", "gained")), ("rel_lost", ("edits", "rel", "lost")),
                     ("tgen_kept", ("edits", "tgen", "kept")), ("tgen_gained", ("edits", "tgen", "gained")), ("tgen_lost", ("edits", "tgen", "lost")),
                     ("igen_kept", ("edits", "igen", "kept")), ("igen_gained", ("edits", "igen", "gained")), ("igen_lost", ("edits", "igen", "lost")),
                     ("reported_resolved", ("reported", "resolved")), ("reported_n", ("reported", "n"))):
        v = per(path)
        if all(x is None for x in v):
            out[nm] = None
            continue
        od = ST.over_draws(v)
        out[nm] = {"mean": od["mean"], "p05": od["p05"], "p95": od["p95"], "per_seed": v}
    for nm, value, unit, where in (("hd_D1", "D1", "query_id", IS_HD), ("strict_D1", "D1", "query_id", IS_STRICT), ("hd_fired", "fired", "query_id", IS_HD),
                                   ("rel", "exact", "edit", IS_M("rel")), ("tgen", "exact", "edit", IS_M("tgen")), ("igen", "exact", "edit", IS_M("igen"))):
        if out.get(nm) is None:
            continue
        cb = crossed({k: rows_by[k] for k in ks}, (lambda r, f=value: float(bool(r[f]))), unit, where)
        out[nm].update(ci_lo=cb["ci_lo"], ci_hi=cb["ci_hi"], n_units=cb["n_units"], n_draws=cb["n_draws"], crossed_value=cb["value"])
    return out


def add_cell_records(R, log, method, cs, **kw):
    for pool in ("heldout_disjoint", "heldout_strict"):
        R.add_nested(log, method, {k: v for k, v in cs[pool].items()}, pool=pool, **kw)
    R.add_nested(log, method, cs["edits"], pool="edits", **kw)
    if "reported" in cs:
        R.add_nested(log, method, cs["reported"], pool="reported", **kw)


def add_group_records(R, log, method, gs, **kw):
    for nm, pool, metric in (("hd_D1", "heldout_disjoint", "D1"), ("strict_D1", "heldout_strict", "D1"), ("hd_fired", "heldout_disjoint", "fired"),
                             ("rel", "edits", "rel.exact"), ("tgen", "edits", "tgen.exact"), ("igen", "edits", "igen.exact"), ("rephrase", "edits", "rephrase.exact"),
                             ("fixed", "heldout_disjoint", "complaints_fixed"), ("lost", "heldout_disjoint", "noncomplaints_lost"),
                             ("reported_resolved", "reported", "resolved")):
        g = gs.get(nm)
        if g is None:
            continue
        R.add(log, method, metric + "@mean", g["mean"], ci_lo=g["p05"], ci_hi=g["p95"], n_units=gs["n_seeds"], pool=pool, **kw)
        if "ci_lo" in g:
            R.add(log, method, metric + "@crossed", g["crossed_value"], ci_lo=g["ci_lo"], ci_hi=g["ci_hi"], n_units=g["n_units"], pool=pool, n_draws=g["n_draws"], **kw)


def draws_summary(values):
    od = ST.over_draws(values)
    return {"mean": od["mean"], "p05": od["p05"], "p95": od["p95"], "min": od["min"], "max": od["max"], "n_draws": od["n_draws"], "values": list(values)}


def main():
    t0 = time.time()
    R = fx.io.Results(EXPERIMENT, OUT)
    summ, status, h1 = {}, {}, {"rule": "R6, literal: the thresholds-only repair 'helps' on a log if its held-out D1 (heldout_disjoint, mean over the seeds of the "
                                        "complaint stream) after n = 10 reported complaints is at most 10 percent AND its Rel exact is within 2 points (absolute difference) "
                                        "of the deployed log's; the endpoint asks for this on both development logs (e2_curves) and on at least one mild prefix (this file).",
                                "best_linear": "per log and n: the linear repair (COMP-B, COMP-B+, REEDIT-B, RB-TOT, RB-UNION) with the lowest mean held-out D1 over the "
                                               "seeds ('best_any'), and the lowest among those whose mean Rel exact is not more than 2 points below the deployed log "
                                               "('best_rel': the one-sided selection of e2_curves/out/h1.json, adopted so that the prefix read-out is comparable with it; in "
                                               "favour of the baseline); the selection is made on the evaluation pool",
                                "readings": "literal: |Rel - Rel deployed| <= 2 points (the verdict of this file); one-sided: Rel >= Rel deployed - 2 points (reported next to "
                                            "it, as e2_curves does); the two differ only where a repair has a higher Rel exact than the deployed log",
                                "intervals": "paired crossed bootstrap over the 3 seeds of the complaint stream and the units (query clusters / edits), 2000 resamples",
                                "cells": {}}
    for log in LOGS:
        setup = get(log, "setup")
        st = {"setup": setup is not None, "missing": [], "failed": {}, "n_cells": 0}
        status[log] = st
        if setup is None or not done(log, "eval__deployed"):
            st["missing"].append("everything (the job has not written the deployed rows)")
            continue
        cdir = os.path.join(CELLS, log)
        for f in sorted(os.listdir(cdir)):
            if f.endswith(".failed.json"):
                fr = fx.io.jload(os.path.join(cdir, f))
                st["failed"][f[:-len(".failed.json")]] = {"attempts": fr.get("attempts"), "error": fr.get("error", "")[:300]}
        dep_rows = load_rows(log, "deployed")
        pre_rows = load_rows(log, "preedit")
        T = setup["T"]
        S = {"T": T, "n_repair_complaints": setup["n_repair_complaints"], "pools": setup["pools"], "hn_valid": setup["hn_valid"], "rows": []}
        summ[log] = S
        kept_rows = {}

        def single(method, label, rows, n_label, evidence, reported=False, extra=None, **kw):
            cs = cell_stats(rows, dep_rows, reported=reported)
            add_cell_records(R, log, method, cs, **kw)
            gs = group_summary({0: cs}, {0: rows})
            row = dict(method=method, label=label, n=n_label, evidence=evidence, **gs)
            row.update(extra or {})
            S["rows"].append(row)
            st["n_cells"] += 1
            return row

        single("deployed", "deployed log", [r for r in dep_rows if not (r["kind"] == "probe" and r["pool"] == "repair")], "-", "none")
        single("preedit", "pre-edit model", [r for r in pre_rows if not (r["kind"] == "probe" and r["pool"] == "repair")], "-", "none")
        dep_rel = S["rows"][0]["rel"]["mean"]

        # ---- complaint-driven cells
        def group(method, label, prefix, n, evidence="complaints", cell_extra=None):
            cells, rows_by, recs = {}, {}, {}
            for (n_, s) in cells_plan():
                if n_ != n:
                    continue
                c = cfg(prefix, n, s)
                rec = get(log, "eval__" + c)
                if rec is None:
                    st["missing"].append(c)
                    continue
                rows = load_rows(log, c)
                cs = cell_stats(rows, dep_rows, reported=True)
                nrep = cs["reported"]["n"]
                add_cell_records(R, log, method, cs, variant=f"n{n}", n=nrep, draw=s)
                cells[s], rows_by[s], recs[s] = cs, rows, rec
                st["n_cells"] += 1
            if not cells:
                return None
            gs = group_summary(cells, rows_by)
            nrep = int(round(gs["reported_n"]["mean"]))
            add_group_records(R, log, method, gs, variant=f"n{n}", n=nrep)
            row = dict(method=method, label=label, n=n, n_reported=nrep, evidence=evidence, **gs)
            if cell_extra is not None:
                row.update(cell_extra(recs))
            S["rows"].append(row)
            kept_rows[(method, n)] = rows_by
            return row

        def lin_extra(recs):
            b = {s: get(log, "build__" + r["config"]) for s, r in recs.items()}
            out = {"correction_kind": sorted({r["correction_kind"] for r in recs.values()}),
                   "fro_rel_to_log": [b[s]["fro_rel_to_log"] for s in sorted(b)], "rank": [b[s]["rank"].get("correction") for s in sorted(b)],
                   "repair_wall_sec": [b[s]["repair_wall_sec"] for s in sorted(b)]}
            ex = [b[s].get("extra") or {} for s in sorted(b)]
            if any("k" in e for e in ex):
                out["edits_rolled_back"] = [e.get("k") for e in ex]
            if any("installed" in e for e in ex):
                out["installed"] = [e.get("installed") for e in ex]
                out["not_installed"] = [len(e.get("not_installed") or []) for e in ex]
            return out

        def where_extra(recs):
            ks = sorted(recs)
            gl = [recs[s]["gate_level"] for s in ks]
            out = {"near": {t: draws_summary([g["hn"][t]["fired"] for g in gl]) if gl[0]["hn"][t]["n"] else None for t in ("HN-I", "HN-Q", "HN-X")},
                   "near_n": {t: gl[0]["hn"][t]["n"] for t in ("HN-I", "HN-Q", "HN-X")},
                   "gl_hd_fired": draws_summary([g["probe"]["heldout_disjoint"]["fired"] for g in gl]),
                   "gl_own": {m: draws_summary([g["edit"][m]["own_released"] for g in gl]) for m in ("rel", "tgen", "igen")},
                   "hook_vs_gate_n_differ": sum(recs[s]["hook_vs_gate_level"]["n_differ"] for s in ks),
                   "hook_vs_gate_n_rows": sum(recs[s]["hook_vs_gate_level"]["n_rows"] for s in ks),
                   "own_prompts_not_candidate": [g["edit"]["rel"]["n"] - g["edit"]["rel"]["n_own_released"] for g in gl]}      # own Rel sequences on which the own edit is not released
            if "least_repair" in recs[ks[0]]:
                lr = [recs[s]["least_repair"] for s in ks]
                oc = [recs[s]["order_check"] for s in ks]
                out.update(n_refused=[x["n_refused"] for x in lr], n_resolved_by_rule=[x["n_resolved"] for x in lr], thresholds_raised=[x["n_thresholds_raised_total"] for x in lr],
                           order_check={"n_orders": sum(o["n_orders"] for o in oc), "n_equal_thresholds": sum(o["n_equal_thresholds"] for o in oc),
                                        "n_equal_refused_sets": sum(o["n_equal_refused_sets"] for o in oc), "max_abs_diff": max(o["max_abs_diff"] for o in oc)})
            return out

        def record_where_gate(method, recs, **kw):
            for s, rec in recs.items():
                g = rec["gate_level"]
                for t in ("HN-I", "HN-Q", "HN-X"):
                    R.add(log, method, f"{t}.fired", g["hn"][t]["fired"], pool="hn", n_units=g["hn"][t]["n"], draw=s, level="gate", **kw)
                R.add(log, method, "fired", g["probe"]["heldout_disjoint"]["fired"], pool="heldout_disjoint", n_units=g["probe"]["heldout_disjoint"]["n"], draw=s, level="gate", **kw)

        for n in NS:
            for method in LIN:
                group(LIN_LABEL[method], method.replace("_BP", "-B+").replace("_", "-"), method, n, cell_extra=lin_extra)
            group("blocklist", "blocklist", "BLOCKLIST", n)
            row = group("where_least", "ReScope-W: log-only thresholds + least repair", "WHERE_LEAST", n, cell_extra=where_extra)
            if row is not None:
                recs = {s: get(log, "eval__" + cfg("WHERE_LEAST", n, s)) for (n_, s) in cells_plan() if n_ == n and done(log, "eval__" + cfg("WHERE_LEAST", n, s))}
                record_where_gate("where_least", recs, variant=f"n{n}", n=row["n_reported"])

        # ---- ReScope-W, log-only thresholds (no complaint)
        rec = get(log, "eval__where_logonly")
        if rec is None:
            st["missing"].append("where_logonly")
        else:
            rows = load_rows(log, "where_logonly")
            single("where_logonly", "ReScope-W: log-only thresholds", rows, 0, "log only", extra=where_extra({0: rec}))
            record_where_gate("where_logonly", {None: rec})

        # ---- calibrated rows: behaviour for draws 0 - 2, gate level for 20 draws
        def calibrated(method, label, prefix, gate_cell, where):
            cells, rows_by, recs = {}, {}, {}
            for d in DRAWS_B:
                c = f"{prefix}__d{d}"
                rec = get(log, "eval__" + c)
                if rec is None:
                    st["missing"].append(c)
                    continue
                rows = load_rows(log, c)
                cs = cell_stats(rows, dep_rows, reported=False)
                add_cell_records(R, log, method, cs, alpha=ALPHA, n=N_CAL, draw=d)
                R.add(log, method, "n_edits_capped", rec.get("n_capped"), pool="edits", alpha=ALPHA, n=N_CAL, draw=d, level="behaviour")
                cells[d], rows_by[d], recs[d] = cs, rows, rec
                st["n_cells"] += 1
            if not cells:
                return
            gs = group_summary(cells, rows_by)
            add_group_records(R, log, method, gs, alpha=ALPHA, n=N_CAL)
            row = dict(method=method, label=label, n=N_CAL, evidence="benign queries", **gs)
            row["n_capped_behaviour"] = [recs[d].get("n_capped") for d in sorted(recs)]
            if where:
                row.update(where_extra(recs))
            g = get(log, gate_cell)
            if g is None:
                st["missing"].append(gate_cell)
            else:
                per = g["draws"]
                ds = sorted(per, key=int)
                if where:
                    f = lambda fn: [fn(per[d]["gate_level"]) for d in ds]                                # noqa: E731
                    gl = {"hd_fired": f(lambda x: x["probe"]["heldout_disjoint"]["fired"]), "strict_fired": f(lambda x: x["probe"]["heldout_strict"]["fired"]),
                          "hd_fired_far": f(lambda x: x["probe"]["heldout_disjoint_far"]["fired"]), "hd_fired_near": f(lambda x: x["probe"]["heldout_disjoint_near"]["fired"]),
                          "rel_own": f(lambda x: x["edit"]["rel"]["own_released"]), "tgen_own": f(lambda x: x["edit"]["tgen"]["own_released"]),
                          "igen_own": f(lambda x: x["edit"]["igen"]["own_released"]),
                          "HN-I": f(lambda x: x["hn"]["HN-I"]["fired"]), "HN-Q": f(lambda x: x["hn"]["HN-Q"]["fired"]), "HN-X": f(lambda x: x["hn"]["HN-X"]["fired"]),
                          "hn_mean": f(lambda x: x["hn"]["mean_fired"])}
                    nhn = {t: per[ds[0]]["gate_level"]["hn"][t]["n"] for t in ("HN-I", "HN-Q", "HN-X")}
                    ben = None
                else:
                    f = lambda fn: [fn(per[d]["gate_stats"]) for d in ds]                                # noqa: E731
                    gl = {"hd_fired": f(lambda x: x["probe"]["heldout_disjoint"]["fired"]), "strict_fired": f(lambda x: x["probe"]["heldout_strict"]["fired"]),
                          "hd_fired_far": f(lambda x: x["probe"]["heldout_disjoint_far"]["fired"]), "hd_fired_near": f(lambda x: x["probe"]["heldout_disjoint_near"]["fired"]),
                          "rel_own": f(lambda x: x["edit"]["rel"]["own_top"]), "tgen_own": f(lambda x: x["edit"]["tgen"]["own_top"]), "igen_own": f(lambda x: x["edit"]["igen"]["own_top"]),
                          "HN-I": f(lambda x: x["hn"]["HN-I"]["fired"]), "HN-Q": f(lambda x: x["hn"]["HN-Q"]["fired"]), "HN-X": f(lambda x: x["hn"]["HN-X"]["fired"]),
                          "hn_mean": [per[d]["hn_mean_fired"] for d in ds], "top1_n_differs": f(lambda x: x["top1"]["all"]["n_differs"])}
                    nhn = {t: per[ds[0]]["gate_stats"]["hn"][t]["n"] for t in ("HN-I", "HN-Q", "HN-X")}
                    ben = {"fired": [per[d]["BEN-T"]["fired"] for d in ds], "far": [per[d]["BEN-T"]["far"]["fired"] for d in ds], "n": per[ds[0]]["BEN-T"]["n"],
                           "which": per[ds[0]]["BEN-T"]["which"]}
                gl["n_capped"] = [per[d]["n_capped"] for d in ds]
                gl["literal_hd_fired"] = [per[d]["literal_no_guards"]["heldout_disjoint_fired"] for d in ds]
                gl["bound"] = [100.0 * float(per[d]["bound_false_firing"]) for d in ds]
                row["gate20"] = {k: (draws_summary(v) if not all(x is None for x in v) else None) for k, v in gl.items()}
                row["gate20"]["n_hn"] = nhn
                row["near"] = {t: row["gate20"][t] for t in ("HN-I", "HN-Q", "HN-X")}
                row["near_n"] = nhn
                for k, v in gl.items():
                    if all(x is None for x in v):
                        continue
                    pool = "hn" if k.startswith("HN") or k == "hn_mean" else ("edits" if k.endswith("_own") or k == "n_capped" else "heldout_disjoint")
                    R.add_draws(log, method, "gate." + k, {int(d): x for d, x in zip(ds, v)}, pool=pool, alpha=ALPHA, n=N_CAL, level="gate")
                if ben is not None:
                    row["ben_t"] = {"fired": draws_summary(ben["fired"]), "far": draws_summary(ben["far"]), "n": ben["n"], "which": ben["which"]}
                    R.add_draws(log, method, "gate.fired", {int(d): x for d, x in zip(ds, ben["fired"])}, pool="BEN-T", alpha=ALPHA, n=N_CAL, level="gate", keys="prompt-only")
                    R.add_draws(log, method, "gate.fired_far", {int(d): x for d, x in zip(ds, ben["far"])}, pool="BEN-T", alpha=ALPHA, n=N_CAL, level="gate", keys="prompt-only")
            S["rows"].append(row)
            kept_rows[(method, N_CAL)] = rows_by

        calibrated("where_two_block", "ReScope-W: two-block thresholds", "WHERE_TB", "gate__where_two_block", True)
        calibrated("rescope", "ReScope(0.05), default method", "RESCOPE", "gate__rescope", False)
        r0 = get(log, "eval__WHERE_TB__d0")
        if r0 is not None:
            S["where_two_block_equals_driver_constructor"] = r0.get("equals_driver_constructor")
        # ---- the round-off floor of ReScope-W: flags that differ from the deployed weight when every threshold is at minus infinity
        if done(log, "eval__where_floor"):
            fr = {r["csid"]: r for r in load_rows(log, "where_floor")}
            dr_ = {r["csid"]: r for r in dep_rows if r["csid"] in fr}
            fl = {"heldout_disjoint": {"n": 0, "n_differ": 0}, "rel": {"n": 0, "n_differ": 0, "n_exact_floor": 0, "n_exact_deployed": 0},
                  "tgen": {"n": 0, "n_differ": 0, "n_exact_floor": 0, "n_exact_deployed": 0}, "igen": {"n": 0, "n_differ": 0, "n_exact_floor": 0, "n_exact_deployed": 0}}
            for k_, a_ in fr.items():
                b_ = dr_[k_]
                if a_["kind"] == "probe":
                    fl["heldout_disjoint"]["n"] += 1
                    fl["heldout_disjoint"]["n_differ"] += int(bool(a_["D1"]) != bool(b_["D1"]))
                else:
                    g_ = fl[a_["metric"]]
                    g_["n"] += 1
                    g_["n_differ"] += int(bool(a_["exact"]) != bool(b_["exact"]))
                    g_["n_exact_floor"] += int(bool(a_["exact"]))
                    g_["n_exact_deployed"] += int(bool(b_["exact"]))
            fl["heldout_disjoint"]["D1_floor"] = M.pct(r["D1"] for r in fr.values() if r["kind"] == "probe")
            S["where_floor"] = fl
            for k_, g_ in fl.items():
                R.add(log, "where_floor", "n_flags_differ_from_deployed", g_["n_differ"], pool="heldout_disjoint" if k_ == "heldout_disjoint" else "edits." + k_, n_units=g_["n"])
        else:
            st["missing"].append("where_floor")
        # ---- D2 as a rate of the pre-correct heldout_disjoint rows (R7): the cells of phase H (seed 0 / draw 0; n = 10)
        d2 = []
        for cname, method, label, kw in ([("deployed", "deployed", "deployed log", {})]
                                         + [(cfg(m, 10, 0), LIN_LABEL[m], m.replace("_BP", "-B+").replace("_", "-") + " (n = 10, seed 0)", {"variant": "n10", "n": 10, "draw": 0}) for m in LIN]
                                         + [("where_logonly", "where_logonly", "ReScope-W: log-only thresholds", {}),
                                            (cfg("WHERE_LEAST", 10, 0), "where_least", "ReScope-W: + least repair (n = 10, seed 0)", {"variant": "n10", "n": 10, "draw": 0}),
                                            ("WHERE_TB__d0", "where_two_block", "ReScope-W: two-block thresholds (draw 0)", {"alpha": ALPHA, "n": N_CAL, "draw": 0}),
                                            ("RESCOPE__d0", "rescope", "ReScope(0.05) (draw 0)", {"alpha": ALPHA, "n": N_CAL, "draw": 0})]):
            if not done(log, "d2__" + cname):
                if os.path.exists(cell_path(log, "d2__" + cname).replace(".json", ".failed.json")) or S.get("timing", None) is None:
                    pass
                st["missing"].append("d2__" + cname)
                continue
            ds = M.d2_stats(load_rows(log, "d2__" + cname), ci=True)
            if ds["D2"] is not None:
                R.add_stat(log, method, "D2", ds["D2"], pool="heldout_disjoint_pre_correct", **kw)
            d2.append({"method": method, "label": label, "n_pre_correct": ds["n_pre_correct"], "n_D2": ds["n_D2"], "D2": ds["D2"], "fired": ds["fired"]})
        S["d2"] = d2
        S["timing"] = get(log, "timing")
        clean = [fx.io.jload(os.path.join(cdir, f)) for f in sorted(os.listdir(cdir)) if f.startswith("clean__")]
        S["jobs_clean"] = [{"hook_mismatch": c["hook_mismatch"], "memo_conflicts": c["memo_conflicts"], "forwards": c["forwards"], "seconds": c["seconds"], "phases": c["phases"],
                            "peak_gpu_gib": c["peak_gpu_bytes"] / 2 ** 30} for c in clean]

        # ---- H1 per log
        for n in ((1, 10) if T == 10 else (10,)):
            wr = kept_rows.get(("where_least", n))
            cell = {"log": log, "T": T, "n": n, "deployed": {"hd_D1": S["rows"][0]["hd_D1"]["mean"], "rel": dep_rel}}
            h1["cells"][f"{log}|n{n}"] = cell
            if wr is None or len(wr) < len(SEEDS):
                cell["status"] = "thresholds-only repair incomplete"
                continue
            wrow = next(r for r in S["rows"] if r["method"] == "where_least" and r["n"] == n)
            cell["where_least"] = {"hd_D1": {k: wrow["hd_D1"][k] for k in ("mean", "ci_lo", "ci_hi", "per_seed")}, "rel": {k: wrow["rel"][k] for k in ("mean", "ci_lo", "ci_hi", "per_seed")},
                                   "n_seeds": wrow["n_seeds"]}
            lin = [r for r in S["rows"] if r["method"] in LIN_LABEL.values() and r["n"] == n and r["n_seeds"] == len(SEEDS)]
            cell["linear"] = {r["method"]: {"hd_D1": r["hd_D1"]["mean"], "rel": r["rel"]["mean"], "rel_within_2_literal": abs(r["rel"]["mean"] - dep_rel) <= 2.0 + 1e-9,
                                            "rel_not_more_than_2_below_deployed": r["rel"]["mean"] >= dep_rel - 2.0 - 1e-9} for r in lin}
            cell["linear_incomplete"] = [m for m in LIN_LABEL.values() if m not in cell["linear"]]
            for tag, cand in (("best_any", lin), ("best_rel", [r for r in lin if r["rel"]["mean"] >= dep_rel - 2.0 - 1e-9])):
                if not cand:
                    cell[tag] = None
                    continue
                b = min(cand, key=lambda r: r["hd_D1"]["mean"])
                lr = kept_rows[(b["method"], n)]
                dd = crossed_diff(wr, lr, (lambda r: float(bool(r["D1"]))), "query_id", IS_HD)
                dr = crossed_diff(wr, lr, (lambda r: float(bool(r["exact"]))), "edit", IS_M("rel"))
                cell[tag] = {"method": b["method"], "hd_D1": {k: b["hd_D1"][k] for k in ("mean", "ci_lo", "ci_hi", "per_seed")},
                             "rel": {k: b["rel"][k] for k in ("mean", "ci_lo", "ci_hi", "per_seed")},
                             "diff_D1_where_minus_linear": {k: dd[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "n_draws", "n_units")},
                             "diff_rel_where_minus_linear": {k: dr[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "n_draws", "n_units")}}
                R.add(log, "where_least", f"H1.diff_D1_vs_{tag}", dd["value"], ci_lo=dd["ci_lo"], ci_hi=dd["ci_hi"], n_units=dd["n_units"], pool="heldout_disjoint", variant=f"n{n}", n=n,
                      n_draws=dd["n_draws"], p_two_sided=dd["p_two_sided"], against=b["method"])
                R.add(log, "where_least", f"H1.diff_rel_vs_{tag}", dr["value"], ci_lo=dr["ci_lo"], ci_hi=dr["ci_hi"], n_units=dr["n_units"], pool="edits", variant=f"n{n}", n=n,
                      n_draws=dr["n_draws"], p_two_sided=dr["p_two_sided"], against=b["method"])
            d1, rel = wrow["hd_D1"]["mean"], wrow["rel"]["mean"]
            cell["verdict"] = {"D1_at_most_10": bool(d1 <= 10.0), "rel_abs_diff_to_deployed": abs(rel - dep_rel), "rel_signed_diff_to_deployed": rel - dep_rel,
                               "rel_within_2_points": bool(abs(rel - dep_rel) <= 2.0 + 1e-9), "helps_on_this_log": bool(d1 <= 10.0 and abs(rel - dep_rel) <= 2.0 + 1e-9),
                               "rel_not_more_than_2_below_deployed": bool(rel >= dep_rel - 2.0 - 1e-9), "one_sided_reading_met_on_this_log": bool(d1 <= 10.0 and rel >= dep_rel - 2.0 - 1e-9),
                               "D1_upper_interval_at_most_10": bool(wrow["hd_D1"]["ci_hi"] <= 10.0), "counts_for_the_endpoint": n == 10}
            cell["status"] = "complete"
    cells10 = [c for c in h1["cells"].values() if c["n"] == 10]
    h1["n_prefix_cells_complete_n10"] = sum(c.get("status") == "complete" for c in cells10)
    h1["prefixes_that_help_n10"] = [c["log"] for c in cells10 if c.get("status") == "complete" and c["verdict"]["helps_on_this_log"]]
    h1["at_least_one_mild_prefix_helps"] = bool(h1["prefixes_that_help_n10"]) if h1["n_prefix_cells_complete_n10"] else None
    h1["note"] = ("the endpoint H1 also needs the two development logs (e2_curves/out/h1.json); this file decides the prefix part only. The n = 1 cell of the "
                  "10-edit prefixes is an additional read-out and does not count for the endpoint (R6 names n = 10).")
    p = os.path.join(E2C, "h1.json")
    h1["development_logs_file"] = p if os.path.exists(p) else "not present when this file was written"

    # ---- the full development logs, for the table over prefixes and full logs (files of other experiments; read-only)
    full = {}
    p12 = fx.io.jload(os.path.join(P12, "out", "results.json"))["logs"]
    e0 = fx.io.jload(os.path.join(FIN, "e0_sanity", "out", "results.json"))["logs"]
    for dev in DEV.values():
        pj = fx.prefix.describe(dev + "-p100")["deployed"]
        f = {"deployed": {"hd_D1": pj["heldout_disjoint"]["D1"], "rel": pj["rel"]["exact"], "tgen": pj["tgen"]["exact"], "igen": pj["igen"]["exact"],
                          "source": "common/out/prefix_logs/<log>-p100/prefix.json (the construction check with all batches = the development log; B = 1)"}}
        rows = p12[dev]["rows"]
        lin = {}
        for m in LIN:
            rs = [rows[cfg(m, 10, s)] for s in SEEDS if cfg(m, 10, s) in rows]
            if len(rs) == len(SEEDS):
                lin[LIN_LABEL[m]] = {"hd_D1": float(np.mean([r["heldout_disjoint"]["D1_rate"] for r in rs])), "rel": float(np.mean([r["edits"]["rel"]["exact"] for r in rs])),
                                     "tgen": float(np.mean([r["edits"]["tgen"]["exact"] for r in rs])), "igen": float(np.mean([r["edits"]["igen"]["exact"] for r in rs])), "n_seeds": len(rs)}
        f["linear_n10"] = lin
        f["linear_source"] = "P12 records (08_rescope/p12_linear_streams/out/results.json: the harness's own evaluation, not re-scored through the library; e2_curves re-scores them)"
        if lin:
            ba = min(lin, key=lambda k: lin[k]["hd_D1"])
            f["best_any"] = dict(lin[ba], method=ba)
            ok = [k for k in lin if abs(lin[k]["rel"] - f["deployed"]["rel"]) <= 2.0 + 1e-9]
            f["best_rel"] = dict(lin[min(ok, key=lambda k: lin[k]["hd_D1"])], method=min(ok, key=lambda k: lin[k]["hd_D1"])) if ok else None
        m1 = e0[dev]["behaviour"]["methods"]["M1"]["summary"]
        f["rescope"] = {"hd_D1": m1["probe"]["heldout_disjoint"]["D1"]["mean"], "hd_fired": m1["probe"]["heldout_disjoint"]["fired"]["mean"], "rel": m1["edit"]["rel"]["exact"]["mean"],
                        "tgen": m1["edit"]["tgen"]["exact"]["mean"], "igen": m1["edit"]["igen"]["exact"]["mean"],
                        "source": "e0_sanity/out/results.json, M1, mean over calibration draws 0 - 2"}
        f["where_least_n10"] = None
        f["where_least_source"] = "e2_curves/out/h1.json: not present when this table was rendered"
        pc = os.path.join(E2C, "h1.json")
        if os.path.exists(pc):
            hc = fx.io.jload(pc)["logs"].get(dev)
            if hc is not None and hc.get("n") == 10:
                f["where_least_n10"] = {"hd_D1": hc["d1_where"], "rel": hc["rel_where"], "helps_literal": hc["helps_on_this_log"], "helps_one_sided": hc["one_sided_reading_met_on_this_log"],
                                        "rel_minus_deployed": hc["rel_minus_deployed"]}
                f["where_least_source"] = "e2_curves/out/h1.json (file of the sibling experiment; read, not recomputed)"

                def from_curves(c):
                    m = c["linear"]
                    rs = [rows[cfg(m, 10, s_)] for s_ in SEEDS if cfg(m, 10, s_) in rows]
                    return {"method": m.lower(), "hd_D1": c["d1_linear"], "rel": c["rel_linear"], "tgen": float(np.mean([r["edits"]["tgen"]["exact"] for r in rs])) if rs else None,
                            "igen": float(np.mean([r["edits"]["igen"]["exact"] for r in rs])) if rs else None, "diff_D1": c["d1_diff"]}
                f["best_any"] = from_curves(hc["comparisons"]["best_lowest_d1"])
                f["best_rel"] = from_curves(hc["comparisons"]["best_rel_within_2"]) if hc["comparisons"].get("best_rel_within_2") else None
                f["linear_source"] = ("e2_curves/out/h1.json: best linear repair at n = 10 as selected there (its candidates also hold COMP-B-ALLPOS); T-Gen / I-Gen of that repair from "
                                      "P12's records")
        full[dev] = f
    summ["_full_development_logs"] = full
    summ["_e2_part_e"] = {"status": "not run", "cause": "cut by amendment R19 (d): R12 is run on the prefix logs only; TB-Q200-ae-l2_3 and TB-L200-ae-l2_3 were not opened by this experiment"}
    summ["_d2"] = {"status": "subset", "what": "D2 = % of the pre-correct heldout_disjoint rows whose 48-token answer no longer contains the reference (decoder v2), measured by "
                   "ses.d2 for: deployed; the five linear repairs and the thresholds-only repair at n = 10, seed 0; the log-only thresholds; ReScope-W two-block and the "
                   "default method at calibration draw 0.  The probe records of prefix logs carry no D2 fields (R25 (5)): nothing is read from them; D2 complaints are not "
                   "used.  Not every behavioural row (cost); the subset was fixed before any result existed."}
    summ["_ben_t"] = ("BEN-T: false firing at gate level from fx.pools.firing (prompt-only keys; the gate is calibrated at teacher-forced REPAIR keys); defined for the default "
                      "method only: a repaired weight has no firing event, and ReScope-W looks at every position while the pool holds the last-prompt key only")

    R.save(logs=list(LOGS), rows_dir=ROWS, cells_dir=CELLS, conventions=__doc__)
    fx.io.jdump(fx.io.clean(summ), os.path.join(OUT, "summary.json"), indent=1)
    fx.io.jdump(fx.io.clean(h1), os.path.join(OUT, "h1_prefix.json"), indent=1)
    fx.io.jdump(fx.io.clean(status), os.path.join(OUT, "status.json"), indent=1)
    paper(R)
    import render
    render.main()
    print(f"aggregated {len(R.records)} records in {time.time() - t0:.0f}s; cells per log: { {k: v['n_cells'] for k, v in status.items()} }")


TAG = {"TB-Q100-ae-l2_3-p10": "QTen", "TB-Q100-ae-l2_3-p20": "QTwenty", "TB-Q100-ae-l2_3-p50": "QFifty",
       "TB-L100-ae-l2_3-p10": "LTen", "TB-L100-ae-l2_3-p20": "LTwenty", "TB-L100-ae-l2_3-p50": "LFifty"}
NTAG = {1: "One", 10: "Ten", 20: "Twenty", "ALL": "All"}


def paper(R):
    """paper_numbers.json (name -> value) and paper_numbers_doc.json (name -> one line), from the JSON files just written."""
    summ, h1 = fx.io.jload(os.path.join(OUT, "summary.json")), fx.io.jload(os.path.join(OUT, "h1_prefix.json"))
    num, doc = {}, {}

    def put_(name, v, text):
        num[name], doc[name] = v, text
    for log, tag in TAG.items():
        S = summ.get(log)
        if S is None:
            continue
        what = f"{log} ({S['T']} edits)"
        by = {(r["method"], r["n"]): r for r in S["rows"]}
        d = by[("deployed", "-")]
        put_(f"Pfx{tag}DeployedDOne", d["hd_D1"]["mean"], f"{what}: deployed log, D1 % on heldout_disjoint")
        put_(f"Pfx{tag}DeployedRel", d["rel"]["mean"], f"{what}: deployed log, Rel exact %")
        put_(f"Pfx{tag}DeployedTGen", d["tgen"]["mean"], f"{what}: deployed log, T-Gen exact %")
        put_(f"Pfx{tag}DeployedIGen", d["igen"]["mean"], f"{what}: deployed log, I-Gen exact %")
        put_(f"Pfx{tag}RepairComplaints", S["n_repair_complaints"], f"{what}: number of D1 complaints in the REPAIR pool (= n 'all')")
        for (m, mt, ml) in (("where_logonly", "WLogOnly", "ReScope-W with log-only thresholds (no complaint)"),):
            r = by.get((m, 0))
            if r:
                put_(f"Pfx{tag}{mt}DOne", r["hd_D1"]["mean"], f"{what}: {ml}, D1 % on heldout_disjoint")
                put_(f"Pfx{tag}{mt}Rel", r["rel"]["mean"], f"{what}: {ml}, Rel exact %")
                put_(f"Pfx{tag}{mt}Fired", r["hd_fired"]["mean"], f"{what}: {ml}, % of heldout_disjoint rows on which an edit is released")
        for n in NS:
            for m, mt, ml in (("where_least", "WLeast", "ReScope-W, log-only thresholds + least repair"), ("blocklist", "Blocklist", "blocklist line"),
                              ("comp_b", "CompB", "COMP-B"), ("comp_bp", "CompBPlus", "COMP-B+"), ("reedit_b", "ReeditB", "REEDIT-B"), ("rb_tot", "RbTot", "RB-TOT"),
                              ("rb_union", "RbUnion", "RB-UNION")):
                r = by.get((m, n))
                if not r:
                    continue
                sd = f"mean over {r['n_seeds']} seed(s)"
                put_(f"Pfx{tag}{mt}N{NTAG[n]}DOne", r["hd_D1"]["mean"], f"{what}: {ml} of n = {n} reported complaints, D1 % on heldout_disjoint, {sd}")
                put_(f"Pfx{tag}{mt}N{NTAG[n]}Rel", r["rel"]["mean"], f"{what}: {ml} of n = {n} reported complaints, Rel exact %, {sd}")
        for m, mt, ml in (("rescope", "ReScope", "default method ReScope(0.05), n = 200 benign queries"), ("where_two_block", "WTwoBlock", "ReScope-W with two-block thresholds, n = 200 benign queries")):
            r = by.get((m, N_CAL))
            if not r:
                continue
            sd = f"mean over {r['n_seeds']} calibration draw(s)"
            put_(f"Pfx{tag}{mt}DOne", r["hd_D1"]["mean"], f"{what}: {ml}, D1 % on heldout_disjoint, {sd} (behaviour)")
            put_(f"Pfx{tag}{mt}Fired", r["hd_fired"]["mean"], f"{what}: {ml}, % of heldout_disjoint rows on which an edit is released, {sd} (behaviour)")
            for e, et in (("rel", "Rel"), ("tgen", "TGen"), ("igen", "IGen")):
                put_(f"Pfx{tag}{mt}{et}", r[e]["mean"], f"{what}: {ml}, {et} exact %, {sd}")
            g = r.get("gate20")
            if g:
                put_(f"Pfx{tag}{mt}GateFired", g["hd_fired"]["mean"], f"{what}: {ml}, false firing % on heldout_disjoint at gate level, mean over 20 calibration draws")
                put_(f"Pfx{tag}{mt}GateLiteralFired", g["literal_hd_fired"]["mean"], f"{what}: {ml}, false firing % of the literal rule without guards, mean over 20 draws")
                put_(f"Pfx{tag}{mt}CappedMax", max(g["n_capped"]["values"]), f"{what}: {ml}, largest number of capped edits over 20 draws")
                for t, tt in (("HN-I", "HnI"), ("HN-Q", "HnQ")):
                    if g.get(t):
                        put_(f"Pfx{tag}{mt}Gate{tt}", g[t]["mean"], f"{what}: {ml}, firing % on {t} hard negatives of the prefix's edits, gate level, mean over 20 draws")
            if r.get("ben_t"):
                put_(f"Pfx{tag}{mt}BenT", r["ben_t"]["fired"]["mean"], f"{what}: {ml}, false firing % on BEN-T (prompt-only keys), gate level, mean over 20 draws")
        for n in (1, 10):
            c = h1["cells"].get(f"{log}|n{n}")
            if not c or c.get("status") != "complete":
                continue
            nt = NTAG[n]
            put_(f"Pfx{tag}HOneN{nt}Helps", "yes" if c["verdict"]["helps_on_this_log"] else "no", f"{what}: H1 rule of R6 applied literally at n = {n} (D1 <= 10 % and Rel within 2 points of the deployed log)")
            for bt, bl in (("best_any", "BestAny"), ("best_rel", "BestRel")):
                b = c.get(bt)
                if not b:
                    put_(f"Pfx{tag}HOneN{nt}{bl}Method", "none", f"{what}: H1 at n = {n}: no linear repair qualifies as {bt}")
                    continue
                put_(f"Pfx{tag}HOneN{nt}{bl}Method", b["method"], f"{what}: H1 at n = {n}: the linear repair selected as {bt} (lowest mean held-out D1" + ("; Rel within 2 points)" if bt == "best_rel" else ")"))
                put_(f"Pfx{tag}HOneN{nt}{bl}DOne", b["hd_D1"]["mean"], f"{what}: H1 at n = {n}: held-out D1 % of {bt}, mean over 3 seeds")
                put_(f"Pfx{tag}HOneN{nt}{bl}Rel", b["rel"]["mean"], f"{what}: H1 at n = {n}: Rel exact % of {bt}, mean over 3 seeds")
                put_(f"Pfx{tag}HOneN{nt}{bl}DiffDOne", b["diff_D1_where_minus_linear"]["value"], f"{what}: H1 at n = {n}: held-out D1 of the thresholds-only repair minus {bt}, points")
                put_(f"Pfx{tag}HOneN{nt}{bl}DiffDOneLo", b["diff_D1_where_minus_linear"]["ci_lo"], f"{what}: H1 at n = {n}: lower end of the paired crossed-bootstrap interval of that difference (3 seeds)")
                put_(f"Pfx{tag}HOneN{nt}{bl}DiffDOneHi", b["diff_D1_where_minus_linear"]["ci_hi"], f"{what}: H1 at n = {n}: upper end of the paired crossed-bootstrap interval of that difference (3 seeds)")
    if h1.get("at_least_one_mild_prefix_helps") is not None:
        put_("PfxHOneMildPrefixHelps", "yes" if h1["at_least_one_mild_prefix_helps"] else "no", "H1, prefix part: the thresholds-only repair helps (rule of R6, n = 10) on at least one prefix log")
        put_("PfxHOnePrefixesHelping", len(h1["prefixes_that_help_n10"]), "H1, prefix part: number of prefix logs (of those complete) on which the rule of R6 holds at n = 10")
        put_("PfxHOnePrefixesComplete", h1["n_prefix_cells_complete_n10"], "H1, prefix part: number of prefix logs whose n = 10 cell is complete")
    p = os.path.join(OUT, "paper_numbers.json")
    if os.path.exists(p):
        os.remove(p)                                                    # rebuilt from scratch at every aggregation (no stale name)
    R.paper(num)
    fx.io.jdump(dict(sorted(doc.items())), os.path.join(OUT, "paper_numbers_doc.json"), indent=1)


if __name__ == "__main__":
    main()
