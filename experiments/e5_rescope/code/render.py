"""E5: render everything that exists (resumable; missing parts are listed).  Reads out/{a,b,c,d,e,f}/ and writes
out/results.json (fx.io schema), out/table.txt, out/paper_numbers.json + paper_numbers_doc.json, out/calibration_series.json,
out/validity_distribution.json, out/capacity_curve.json, out/capacity_table.txt.     python render.py"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, needs=("experiments/e5_rescope/out/c",))
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob
import gzip

from e5common import *

R = fx.io.Results("E5", OUT)
PN, DOC, TXT, MISSING = {}, {}, [], []
tt = fx.io.text_table


def pn(key, value, doc):
    PN[key] = value
    DOC[key] = doc


def f(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def sci(x):
    return "-" if x is None else f"{x:.1e}"


# ----------------------------------------------------------------------------- (a)
def part_a():
    rows = []
    for name in LOGS_A:
        p = f"{OUT}/a/{name}.json"
        if not os.path.exists(p):
            MISSING.append(f"a: {name}")
            continue
        r = jload(p)
        rb, ok, oc, mg, cs = r["rebase"], r["own_keys"], r["own_coordinate_b1"], r["self_separation_margin"], r["correction_size_rel_to_value"]
        vals = {"T": r["T"], "rebase.n_targets_found": r["n_targets_found"],
                "rebase.err_rebased_max": rb and rb["err_rebased"]["max"], "rebase.err_rebased_median": rb and rb["err_rebased"]["median"],
                "rebase.err_logged_max": rb and rb["err_logged"]["max"], "rebase.err_logged_median": rb and rb["err_logged"]["median"],
                "rebase.correction_rel_median": cs["median"], "rebase.correction_rel_max": cs["max"],
                "rebase.deployed_err_at_own_keys_median": ok and ok["deployed_log_rel_err_vs_target"]["median"],
                "rebase.deployed_err_at_own_keys_max": ok and ok["deployed_log_rel_err_vs_target"]["max"],
                "rebase.scoped_err_at_own_keys_median": ok and ok["scoped_replay_rel_err_vs_target"]["median"],
                "rebase.scoped_err_at_own_keys_max": ok and ok["scoped_replay_rel_err_vs_target"]["max"],
                "rebase.margin_min": mg["min"], "rebase.margin_median": mg["median"], "rebase.margin_n_nonpositive": mg["n_nonpositive"],
                "rebase.own_coordinate_min": oc["min"], "rebase.own_coordinate_median": oc["median"], "rebase.own_coordinate_drop_max": oc["drop_max"],
                "rebase.own_keys_firing_at_threshold_one": oc["n_firing_at_threshold_one"], "rebase.own_keys_silenced_at_threshold_one": oc["n_below_one"],
                "rebase.own_keys_not_above_own_margin": oc["n_below_own_margin"]}
        for k, v in vals.items():
            if k != "T":
                R.add(name, "log", k, v, pool="edits", n_units=r["T"])
                pn(f"logs.{name}.{k}", v, f"E5 (a), {name}: {k} (factor level; out/a/{name}.json)")
        rows.append([name, r["T"], sci(vals["rebase.err_rebased_max"]), sci(vals["rebase.err_rebased_median"]), f(vals["rebase.err_logged_max"], 3), f(vals["rebase.err_logged_median"], 3),
                     f(cs["median"], 3), f(cs["max"], 3), f(vals["rebase.deployed_err_at_own_keys_median"], 3), f(vals["rebase.deployed_err_at_own_keys_max"], 3),
                     f(mg["min"], 3), f(mg["median"], 3), mg["n_nonpositive"], f(oc["min"], 3), f(oc["median"], 3), oc["n_below_one"], oc["n_firing_at_threshold_one"]])
    TXT.append("E5 (a)  REBASE AND RETROFIT AT FACTOR LEVEL (all T edits of each log; relative errors against the cached clean-model targets V* = z* - h(W_0))\n"
               "  err reb / err log: |V_i - V*_i| / |V*_i| with the rebased value / with the logged value U_i; corr: |V_i - U_i| / |V_i|; dep@own: module output of the\n"
               "  deployed log at its solver key against the target (second identity; factors, no write round-off), the scoped replay's is err reb (W_0 k_i + V_i);\n"
               "  margin: 1 - largest normalised coordinate of edit i at another solver key; own r: coordinate of the B = 1 own Rel prompt; <1: own prompts that a threshold of one silences\n"
               + tt(["log", "T", "err reb max", "err reb med", "err log max", "err log med", "corr med", "corr max", "dep@own med", "dep@own max", "margin min", "margin med", "margin<=0",
                     "own r min", "own r med", "own r <1", "own r >=1"], rows))


# ----------------------------------------------------------------------------- (b)
def summ(nc):
    nc = np.asarray(nc)
    nz = nc[nc > 0]
    return {"n": int(nc.size), "share_zero": float((nc == 0).mean() * 100) if nc.size else None, "median": float(np.median(nz)) if nz.size else None,
            "mean": float(nz.mean()) if nz.size else None, "max": int(nz.max()) if nz.size else None}


def part_b():
    rows = []
    for name in DEVS:
        per = {}
        for rule in ("deployed_scoping", "where_logonly"):
            p = f"{OUT}/b/{name}__{rule}.jsonl.gz"
            if not os.path.exists(p) or not os.path.exists(f"{OUT}/b/{name}__where.json"):
                MISSING.append(f"b: {name} {rule} (GPU)")
                continue
            rr = fx.io.load_rows(p)
            hd = [x for x in rr if x["set"] == "heldout_disjoint"]
            per[rule] = {"heldout_disjoint": summ([(x["n_cand_edits"] or 0) for x in hd]), "heldout_disjoint_far": summ([(x["n_cand_edits"] or 0) for x in hd if not x["near"]]),
                         "heldout_disjoint_near": summ([(x["n_cand_edits"] or 0) for x in hd if x["near"]]),
                         "repair_D1_complaints": summ([(x["n_cand_edits"] or 0) for x in rr if x["set"] == "repair_D1_complaint"]), "n_draws": None}
        p = f"{OUT}/b/{name}__two_block.json"
        if os.path.exists(p):
            r = jload(p)
            per["two_block"] = dict(r["pooled_over_draws"], n_draws=r["n_draws"])
            sz = [r["draws"][str(d)]["heldout_disjoint"]["share_zero"] for d in range(r["n_draws"])]
            per["two_block"]["share_zero_by_draw"] = sz
            per["two_block"]["n_capped_by_draw"] = [r["draws"][str(d)]["n_edits_capped"] for d in range(r["n_draws"])]
            for d in range(r["n_draws"]):
                for pool in ("heldout_disjoint", "heldout_disjoint_far", "heldout_disjoint_near", "repair_D1_complaints", "BEN-T_far", "heldout_disjoint_literal_rule"):
                    for k in ("share_zero", "median", "mean", "max"):
                        R.add(name, "rescope", f"width.{k}", r["draws"][str(d)][pool][k], alpha=0.05, n=200, draw=d, pool=pool, n_units=r["draws"][str(d)][pool]["n"],
                              n_edits_capped=r["draws"][str(d)]["n_edits_capped"])
        else:
            MISSING.append(f"b: {name} two_block")
        for rule, v in per.items():
            lab = {"deployed_scoping": "deployed", "where_logonly": "where_logonly", "two_block": "two_block"}[rule]
            for pool, s in v.items():
                if not isinstance(s, dict) or "share_zero" not in s:
                    continue
                if rule != "two_block":
                    for k in ("share_zero", "median", "mean", "max"):
                        R.add(name, {"deployed_scoping": "deployed_scoping", "where_logonly": "where_logonly"}[rule], f"width.{k}", s[k], pool=pool, n_units=s["n"])
                key = f"width.{name}.{lab}" if pool == "heldout_disjoint" else f"width.{name}.{lab}.{pool}"
                for k in ("share_zero", "median", "mean", "max"):
                    pn(f"{key}.{k}", s[k], f"E5 (b), {name}, rule {lab}, rows {pool}: lineage size {k} (share_zero in percent; median / mean / max over the queries with size > 0"
                                           + ("; pooled over 20 calibration draws" if rule == "two_block" else "") + ")")
                extra = ""
                if rule == "two_block" and pool == "heldout_disjoint":
                    extra = f"zero by draw {min(v['share_zero_by_draw']):.1f}-{max(v['share_zero_by_draw']):.1f}; capped {min(v['n_capped_by_draw'])}-{max(v['n_capped_by_draw'])}"
                rows.append([name, lab, pool, s["n"], f(s["share_zero"], 1), f(s["median"], 1), f(s["mean"], 2), s["max"] if s["max"] is not None else "-", v.get("n_draws") or "-", extra])
    TXT.append("E5 (b)  LINEAGE SIZE (number of edits that fire on the pre-edit trace of a query); zero % = share of queries with size 0; median / mean / max over the others\n"
               "  deployed = the deployed scoping replayed (all thresholds -inf, all positions; measured through the hook); where_logonly = ReScope-W with log-only thresholds (all positions);\n"
               "  two_block = candidates of the calibrated default method at the last prompt position, 20 calibration draws pooled (rows x draws).  far / near: probes that do not / do share a\n"
               "  question string with an edit of the log; repair_D1_complaints: the D1 complaints of the REPAIR pool under the deployed log; BEN-T: prompt-only keys, far = unrelated rows\n"
               + tt(["log", "rule", "rows", "n", "zero %", "median", "mean", "max", "draws", "note"], rows))


# ----------------------------------------------------------------------------- (c)
def tot_se(se_mc, a, n_test):
    """Standard error (percent) of a mean realised rate against the level a: Monte-Carlo error over draws + binomial sampling error of a FIXED test pool."""
    return float(np.sqrt(se_mc ** 2 + (0.0 if not n_test else a * (1 - a) / n_test * 1e4)))


def cell_of(by_n, n, rule, alpha):
    for bn in by_n:
        if bn.get("n") == n and "cells" in bn:
            for c in bn["cells"]:
                if c["rule"] == rule and c["alpha"] == alpha:
                    return c
    return None


def g(c, test, sub, v, k="mean"):
    try:
        return c["tests"][test][sub][v][k]
    except (KeyError, TypeError):
        return None


def part_c():
    series, dist = {}, {}
    names = list(DEVS) + list(LOGS_C_GATE)
    have = {}
    for name in names:
        p = f"{OUT}/c/{name}.json"
        if not os.path.exists(p) or "seconds" not in (jload(p)):
            MISSING.append(f"c: {name}" + (" (partial)" if os.path.exists(p) else ""))
            if not os.path.exists(p):
                continue
        have[name] = jload(p)
    # --- series file + records
    for name, r in have.items():
        s = series.setdefault(name, {"T": r["T"], "cells": {}})
        groups = [("probes_all", r["within"].get("probes_all"), "rest")]
        if "BEN-S" in r["within"]:
            groups.append(("BEN-S", r["within"]["BEN-S"], "rest"))
        for cname, cv in r.get("cross", {}).items():
            groups.append((cname, cv, None))
        for gname, gv, _ in groups:
            if not gv:
                continue
            for bn in gv["by_n"]:
                if "cells" not in bn:
                    continue
                for c in bn["cells"]:
                    for test, tv in c["tests"].items():
                        key = f"{gname}->{test}"
                        for sub in ("all", "far", "near", "image", "text", "not_q_edit"):
                            if sub not in tv or "lit" not in tv[sub]:
                                continue
                            e = {"n": c["n"], "alpha": c["alpha"], "rule": c["rule"], "bound_pct": None if c["bound"] is None else c["bound"] * 100, "subset": sub,
                                 "ff_literal_mean": tv[sub]["lit"]["mean"], "ff_literal_p05": tv[sub]["lit"]["p05"], "ff_literal_p95": tv[sub]["lit"]["p95"],
                                 "ff_guards_mean": tv[sub]["grd"]["mean"], "ff_guards_p05": tv[sub]["grd"]["p05"], "ff_guards_p95": tv[sub]["grd"]["p95"],
                                 "share_of_draws_above_alpha_literal": tv[sub]["lit"].get("share_of_draws_above_alpha"),
                                 "share_of_draws_above_alpha_guards": tv[sub]["grd"].get("share_of_draws_above_alpha"),
                                 "n_capped_mean": c["n_capped_mean"], "coverage_tgen": c["coverage"]["tgen"]["grd"], "coverage_igen": c["coverage"]["igen"]["grd"],
                                 "n_draws": c["n_draws"], "n_test_rows": tv.get("n_rows", {}).get("n")}
                            s["cells"].setdefault(key, []).append(e)
                            if sub in ("all", "far", "near"):
                                kw = dict(alpha=c["alpha"], n=c["n"], pool=f"{key}:{sub}", n_units=c["n_draws"], n_edits_capped_mean=c["n_capped_mean"],
                                          bound_pct=e["bound_pct"])
                                R.add(name, c["rule"], "false_firing_literal@mean", e["ff_literal_mean"], ci_lo=e["ff_literal_p05"], ci_hi=e["ff_literal_p95"], **kw)
                                R.add(name, c["rule"], "false_firing_guards@mean", e["ff_guards_mean"], ci_lo=e["ff_guards_p05"], ci_hi=e["ff_guards_p95"], **kw)
                    kw = dict(alpha=c["alpha"], n=c["n"], pool=f"{gname}:rephrases", n_units=c["n_draws"])
                    R.add(name, c["rule"], "coverage_tgen@mean", c["coverage"]["tgen"]["grd"], **kw)
                    R.add(name, c["rule"], "coverage_igen@mean", c["coverage"]["igen"]["grd"], **kw)
    jdump({"doc": "E5 (c): realised false-firing rates (percent) over 500 calibration draws per cell; cells[calibration pool->test pool] = list of entries "
                  "(rule, n, alpha, subset of test rows, literal rule and rule with guards: mean, 5th, 95th percentile; bound; coverage of rephrases). "
                  "Keys: probes_all and repair_to_heldout = teacher-forced keys; *_prompt_keys and BEN-S = prompt-only keys.", "logs": series}, f"{OUT}/calibration_series.json")
    # --- table 1: within pool, development logs
    for name in DEVS:
        if name not in have:
            continue
        r = have[name]
        for gname in ("probes_all", "BEN-S"):
            if gname not in r["within"]:
                continue
            by_n = r["within"][gname]["by_n"]
            rows = []
            for n in NSLIST:
                for a in (0.01, 0.02, 0.05, 0.10, 0.20):
                    tb, sh, pc = cell_of(by_n, n, "two_block", a), cell_of(by_n, n, "shared", a), cell_of(by_n, n, "pac", a)
                    if tb is None:
                        continue
                    rows.append([n, a, f(tb["bound"] * 100), f"{f(g(tb, 'rest', 'all', 'lit'))} [{f(g(tb, 'rest', 'all', 'lit', 'p05'), 1)}, {f(g(tb, 'rest', 'all', 'lit', 'p95'), 1)}]",
                                 f(g(tb, "rest", "all", "grd")), f(g(tb, "rest", "far", "grd")), f(g(tb, "rest", "near", "grd")), f(g(tb, "rest", "not_q_edit", "grd")), f(tb["n_capped_mean"], 1),
                                 f"{f(tb['coverage']['tgen']['grd'], 1)}/{f(tb['coverage']['igen']['grd'], 1)}",
                                 f(sh["bound"] * 100), f"{f(g(sh, 'rest', 'all', 'lit'))} [{f(g(sh, 'rest', 'all', 'lit', 'p05'), 1)}, {f(g(sh, 'rest', 'all', 'lit', 'p95'), 1)}]",
                                 f(g(sh, "rest", "all", "grd")), f(g(sh, "rest", "far", "grd")), f(g(sh, "rest", "near", "grd")), f(sh["n_capped_mean"], 1),
                                 f"{f(sh['coverage']['tgen']['grd'], 1)}/{f(sh['coverage']['igen']['grd'], 1)}",
                                 f(g(pc, "rest", "all", "lit")), f(g(pc, "rest", "all", "lit", "share_of_draws_above_alpha"), 1)])
            rm = []
            for n in NSLIST:
                c = cell_of(by_n, n, "runmax", None)
                if c:
                    rm.append([n, f(c["bound"] * 100), f"{f(g(c, 'rest', 'all', 'lit'))} [{f(g(c, 'rest', 'all', 'lit', 'p05'), 1)}, {f(g(c, 'rest', 'all', 'lit', 'p95'), 1)}]",
                               f(g(c, "rest", "all", "grd")), f(g(c, "rest", "far", "grd")), f(g(c, "rest", "near", "grd")), f(c["n_capped_mean"], 1),
                               f"{f(c['coverage']['tgen']['grd'], 1)}/{f(c['coverage']['igen']['grd'], 1)}"])
            TXT.append(f"E5 (c)  VALIDITY WITHIN A POOL: {name}, pool {gname} ({r['within'][gname]['pool']}; {r['within'][gname]['n_pool']} rows); 500 draws of n calibration rows, test = the rest\n"
                       "  false firing in percent: literal rule mean [5th, 95th percentile over draws]; grd = with guards (the method); far / near = test rows unrelated / related to an edit (see (b));\n"
                       "  no q-edit (benign pool only) = without the rows whose question string is the question of an edit of the log; capped = mean number of capped edits; cov = coverage of text /\n"
                       "  image rephrases (guards); pac = calibration-conditional rank (delta 0.1): mean, % of draws above alpha\n"
                       + tt(["n", "alpha", "2blk bound", "2blk literal", "grd", "far", "near", "no q-edit", "capped", "cov t/i", "shared bound", "shared literal", "grd", "far", "near", "capped", "cov t/i",
                             "pac lit", "pac >alpha %"], rows)
                       + "\n  per-edit running maxima (least repair of the n queries from thresholds -inf), bound min(1, T / (n + 1)):\n"
                       + tt(["n", "bound", "literal", "grd", "far", "near", "capped", "cov t/i"], rm))
    # --- table 2: cross-pool cells at n = 200
    for name in DEVS:
        if name not in have or not have[name].get("cross"):
            continue
        r = have[name]
        rows, miss = [], []
        for cname, cv in r["cross"].items():
            for a in (0.01, 0.02, 0.05, 0.10, 0.20):
                for rule in ("two_block", "pac", "shared"):
                    c = cell_of(cv["by_n"], 200, rule, a)
                    if c is None:
                        continue
                    for test in sorted(c["tests"]):
                        if a != 0.05 and rule != "two_block":
                            continue
                        tv = c["tests"][test]
                        main = "far" if test == "BEN-T" else "all"
                        lit, grd = g(c, test, main, "lit"), g(c, test, main, "grd")
                        rows.append([cname, test + (" (far)" if main == "far" else ""), rule, a, "-" if c["bound"] is None else f(c["bound"] * 100),
                                     f"{f(lit)} [{f(g(c, test, main, 'lit', 'p05'), 1)}, {f(g(c, test, main, 'lit', 'p95'), 1)}]", f(grd),
                                     f(g(c, test, "all", "grd")), f(g(c, test, "near", "grd")), f(g(c, test, "image", "grd")), f(g(c, test, "text", "grd")),
                                     f(g(c, test, main, "lit", "share_of_draws_above_alpha"), 1), f(c["n_capped_mean"], 1), tv["n_rows"]["n"], tv["n_rows"].get("near", "-")])
                        se = tot_se((tv[main]["lit"].get("se_mean") or 0.0) if main in tv else 0.0, a, None if test == "rest" else (tv["n_rows"]["n"] if main == "all" else tv["n_rows"].get(main)))
                        if rule in ("two_block", "shared") and lit is not None and lit > a * 100 + 2 * se:
                            miss.append(f"{rule} {cname}->{test} alpha {a}: {lit:.2f} % (SE {se:.2f})")
        TXT.append(f"E5 (c)  VALIDITY ACROSS POOLS: {name}, n = 200, 500 draws of the calibration rows; test = every row of the test pool ('rest' = the remaining rows of the calibration pool)\n"
                   "  repair_to_heldout: teacher-forced keys (registered cell); *_prompt_keys: prompt-only key of every row on both sides.  main = far rows for BEN-T (related rows excluded, counted\n"
                   "  in column near n), all rows otherwise; all / near / image / text: rule with guards on those rows; >alpha % = share of draws whose realised literal rate exceeds alpha\n"
                   + tt(["calibration", "test", "rule", "alpha", "bound", "literal main [p05, p95]", "grd main", "grd all", "grd near", "image", "text", ">alpha %", "capped", "n test", "near n"], rows)
                   + "\n  cells (two_block at every level, shared at 0.05) whose mean literal rate exceeds the level by more than 2 SE (Monte-Carlo error and sampling error of a fixed test pool): " + ("; ".join(miss) if miss else "none"))
    # --- table 3: T series at gate level
    rows = []
    for name in list(DEVS) + list(LOGS_C_GATE):
        if name not in have:
            continue
        by_n = have[name]["within"]["probes_all"]["by_n"]
        for n in NSLIST:
            rmx, tb, sh = cell_of(by_n, n, "runmax", None), cell_of(by_n, n, "two_block", 0.05), cell_of(by_n, n, "shared", 0.05)
            if rmx is None:
                continue
            rows.append([name, have[name]["T"], n, f(rmx["bound"] * 100, 1), f(g(rmx, "rest", "all", "lit")), f(g(rmx, "rest", "far", "lit")), f(g(rmx, "rest", "near", "lit")),
                         f(tb["bound"] * 100), f(g(tb, "rest", "all", "lit")), f(g(tb, "rest", "all", "grd")), f(g(tb, "rest", "far", "grd")), f(g(tb, "rest", "near", "grd")), f(tb["n_capped_mean"], 1),
                         f"{f(tb['coverage']['tgen']['grd'], 1)}/{f(tb['coverage']['igen']['grd'], 1)}",
                         f(g(sh, "rest", "all", "lit")), f(g(sh, "rest", "all", "grd")), f(sh["n_capped_mean"], 1), f"{f(sh['coverage']['tgen']['grd'], 1)}/{f(sh['coverage']['igen']['grd'], 1)}"])
    TXT.append("E5 (c)  RUNNING MAXIMA AGAINST THE CALIBRATED RULES ALONG T (gate level; pool: valid REPAIR-plus-held-out probes of each log, teacher-forced keys; test = the rest; alpha = 0.05)\n"
               "  false firing in percent, mean over 500 draws; lit = literal rule, grd = with guards (capped edits fire below the calibrated level: their bound is void); near = probes that share a question with an edit\n"
               + tt(["log", "T", "n", "runmax bound", "runmax lit", "far", "near", "2blk bound", "2blk lit", "2blk grd", "far", "near", "capped", "cov t/i", "shared lit", "shared grd", "capped", "cov t/i"], rows))
    # --- distribution file + paper numbers
    from scipy import stats as st
    for name in DEVS:
        if name not in have or not have[name].get("cross"):
            continue
        r = have[name]
        d = dist.setdefault(name, {"n": 200, "alpha": 0.05, "pairs": {}})
        EXCH = {"repair_to_heldout": "approximately (probe items split by item)", "repair_to_repair_rest": "yes (same pool)", "repair_to_heldout_prompt_keys": "approximately (split by item)",
                "repair_to_ben_t": "no (another workload)", "repair_to_mme_t": "no (another workload)", "ben_s_to_heldout": "no (another workload)",
                "ben_s_to_ben_t": "yes (parts of one pool, split by source item)", "ben_s_to_mme_t": "no (another workload)", "ben_s_to_ben_s_rest": "yes (same pool)",
                "mme_c_to_heldout": "no (another workload)", "mme_c_to_ben_t": "no (another workload)", "mme_c_to_mme_t": "yes (parts of one pool, split by image)",
                "mme_c_to_mme_c_rest": "yes (same pool)"}
        pairs = [("repair_to_heldout", "repair_to_heldout", "heldout_disjoint", "all"), ("repair_to_repair_rest", "repair_to_heldout", "rest", "all"),
                 ("repair_to_heldout_prompt_keys", "repair_prompt_keys", "heldout_disjoint", "all"), ("repair_to_ben_t", "repair_prompt_keys", "BEN-T", "far"),
                 ("repair_to_mme_t", "repair_prompt_keys", "MME-T", "all"),
                 ("ben_s_to_heldout", "ben_s_prompt_keys", "heldout_disjoint", "all"), ("ben_s_to_ben_t", "ben_s_prompt_keys", "BEN-T", "far"), ("ben_s_to_mme_t", "ben_s_prompt_keys", "MME-T", "all"),
                 ("ben_s_to_ben_s_rest", "ben_s_prompt_keys", "rest", "all"),
                 ("mme_c_to_heldout", "mme_c_prompt_keys", "heldout_disjoint", "all"), ("mme_c_to_ben_t", "mme_c_prompt_keys", "BEN-T", "far"), ("mme_c_to_mme_t", "mme_c_prompt_keys", "MME-T", "all"),
                 ("mme_c_to_mme_c_rest", "mme_c_prompt_keys", "rest", "all")]
        rowsd = []
        for pname, cname, test, sub in pairs:
            cv = r["cross"].get(cname)
            if not cv:
                continue
            for rule in ("two_block", "pac", "shared"):
                c = cell_of(cv["by_n"], 200, rule, 0.05)
                if c is None or test not in c["tests"]:
                    continue
                tv = c["tests"][test]
                key = f"validity.{name}.{rule}.{pname}"
                e = tv[sub]
                pn(f"{key}.ff_mean", e["grd"]["mean"], f"E5 (c), {name}, {rule}, alpha 0.05, n 200, calibration->test {pname} ({cv['keys']}), rows '{sub}': realised false-firing rate with guards, mean over 500 draws, percent")
                pn(f"{key}.ff_p95", e["grd"]["p95"], "same cell: 95th percentile over the 500 draws")
                pn(f"{key}.ff_p05", e["grd"]["p05"], "same cell: 5th percentile over the 500 draws")
                pn(f"{key}.ff_literal_mean", e["lit"]["mean"], "same cell: literal rule without guards, mean")
                pn(f"{key}.share_of_draws_above_alpha", e["lit"]["share_of_draws_above_alpha"], "same cell: percent of the 500 draws whose realised literal rate exceeds alpha = 5 %")
                pn(f"{key}.n_capped_mean", c["n_capped_mean"], "same cell: mean number of capped edits (bound void for them)")
                for sub2, lab in (("image", "image_rows"), ("text", "text_rows"), ("far", "far_rows"), ("near", "near_rows"), ("all", "all_rows")):
                    if sub2 in tv and "grd" in tv[sub2]:
                        pn(f"{key}.{lab}.ff_mean", tv[sub2]["grd"]["mean"], f"same cell, test rows '{sub2}': with guards, mean over draws, percent")
                if rule in ("two_block", "pac"):
                    n2 = 100
                    k = c["rank"]
                    a_, b_ = n2 + 1 - k, k
                    x = np.linspace(0, 0.25, 251)
                    lit = e["lit"].get("per_draw")
                    d["pairs"].setdefault(pname, {"exchangeable": EXCH[pname], "keys": cv["keys"], "test": test, "rows": sub, "n_test_rows": tv["n_rows"]["n"]})[rule] = {
                        "rank": k, "n2": n2, "beta_a": a_, "beta_b": b_, "beta_mean_pct": a_ / (a_ + b_) * 100 if k <= n2 else 0.0,
                        "beta_prob_above_alpha": float(st.beta.sf(0.05, a_, b_)) if k <= n2 else 0.0,
                        "beta_pdf_grid_x_pct": [float(v * 100) for v in x], "beta_pdf_grid": [float(v) for v in st.beta.pdf(x, a_, b_)] if k <= n2 else None,
                        "realised_literal_pct_per_draw": lit, "realised_guards_pct_per_draw": e["grd"].get("per_draw"),
                        "realised_mean_pct": e["lit"]["mean"], "share_of_draws_above_alpha_pct": e["lit"]["share_of_draws_above_alpha"],
                        "ks_distance_to_beta": float(st.kstest(np.asarray(lit) / 100, lambda v: st.beta.cdf(v, a_, b_)).statistic) if lit and k <= n2 else None}
                    dd = d["pairs"][pname][rule]
                    rowsd.append([pname, rule, f"Beta({a_}, {b_})", f(dd["beta_mean_pct"]), f(dd["beta_prob_above_alpha"] * 100, 1), f(e["lit"]["mean"]), f(e["lit"]["p05"], 1), f(e["lit"]["p95"], 1),
                                  f(e["lit"]["share_of_draws_above_alpha"], 1), f(dd["ks_distance_to_beta"], 3), tv["n_rows"]["n"] if sub == "all" else tv["n_rows"].get(sub), EXCH[pname],
                                  ("MISSED" if e["lit"]["mean"] > 5.0 + 2 * tot_se(e["lit"].get("se_mean", 0.0), 0.05, None if test == "rest" else (tv["n_rows"]["n"] if sub == "all" else tv["n_rows"].get(sub)))
                                   else ("above, within 2 SE" if e["lit"]["mean"] > 5.0 else "met")) if rule == "two_block"
                                  else ("MISSED" if e["lit"]["share_of_draws_above_alpha"] > 10.0 else "met")])
                    pn(f"{key}.exchangeable", EXCH[pname], "same cell: are calibration and test queries exchangeable")
                    pn(f"{key}.level", rowsd[-1][-1], "same cell: two_block: mean realised rate against alpha = 5 % (MISSED = above by more than 2 standard errors: Monte-Carlo error of the 500 draws and, for a fixed test pool, its binomial sampling error); pac: share of draws above alpha against delta = 10 %")
                    pn(f"{key}.ff_se_mean", e["lit"].get("se_mean"), "same cell: Monte-Carlo standard error of the mean over the 500 draws (literal rule)")
        TXT.append(f"E5 (c)  DISTRIBUTION OF THE REALISED RATE OVER 500 DRAWS AGAINST THE BETA LAW: {name}, n = 200 (n2 = 100), alpha = 0.05, literal rule (percent)\n"
                   "  law: conditional false-firing probability ~ Beta(n2 + 1 - k, k) for exchangeable continuous scores; the realised rate on a finite test pool adds sampling noise, and with a FIXED\n"
                   "  test pool (all cells except 'rest') the draws share one test sample.  P>alpha law / realised: share of draws above 5 %; rescope_pac must stay at about 10 % or less where exchangeable\n"
                   + tt(["calibration->test", "rule", "law", "law mean", "law P>alpha %", "realised mean", "p05", "p95", "realised >alpha %", "KS", "n test", "exchangeable", "level"], rowsd)
                   + "\n  level: two_block = mean realised rate against alpha (MISSED: above 5 % by more than 2 SE; SE = Monte-Carlo error of the mean over draws and, for a fixed test pool, the binomial"
                   + "\n  sampling error sqrt(alpha (1 - alpha) / n test) of that pool, rows taken as independent); pac = share of draws above alpha against delta = 10 %"
                   + "\n  NOMINAL LEVEL MISSED in: " + ("; ".join(f"{x[0]} ({x[1]}; exchangeable: {x[11]})" for x in rowsd if x[-1] == "MISSED") or "no cell"))
    jdump({"doc": "E5 (c): per development log and calibration->test pair at n = 200, alpha = 0.05: the realised literal false-firing rate of each of the 500 calibration draws (percent) "
                  "for the two-block rule and for rescope_pac, with the Beta law of the conditional rate (parameters, pdf on a grid) for overlay.", "logs": dist}, f"{OUT}/validity_distribution.json")


NSLIST = (20, 50, 100, 200, 500)


# ----------------------------------------------------------------------------- (d)
def part_d():
    rows = []
    for name in DEVS:
        for start in ("logonly", "neginf"):
            p = f"{OUT}/d/{name}__{start}.json"
            if not os.path.exists(p):
                MISSING.append(f"d: {name} {start}")
                continue
            r = jload(p)
            s = r["summary"]
            vals = {"streams_identical_for_all_orders": s["streams_with_identical_thresholds_for_all_orders"], "n_streams": s["n_streams"], "writes_mean": s["writes_mean"],
                    "writes_max": s["writes_max"], "bound_writes": s["bound_writes"], "n_regressions": s["n_regressions_total"], "n_refused": s["refused_total_order0"],
                    "queries_firing_at_start_mean": s["queries_firing_at_start_mean"], "false_firing_start": r["start"]["heldout_disjoint"],
                    "false_firing_after": s["heldout_disjoint"]["mean"], "false_firing_after_far": s["heldout_disjoint_far"]["mean"],
                    "false_firing_after_near": (s["heldout_disjoint_near"] or {}).get("mean"), "ben_t_far_start": r["start"]["BEN-T"]["far"], "ben_t_far_after": s["BEN-T_far"]["mean"],
                    "bound_false_firing_pct": s["bound_false_firing_pct"]}
            for k, v in vals.items():
                R.add(name, f"least_repair_from_{start}", f"least.{k}", v, pool="repair streams (N = 50)", n_units=s["n_streams"])
                pn(f"least.{name}.{start}.{k}", v, f"E5 (d), {name}, start {start}: {k} (100 streams of 50 benign REPAIR queries, 20 arrival orders each)")
            rows.append([name, start, f"{vals['streams_identical_for_all_orders']}/{vals['n_streams']}", f(vals["writes_mean"], 1), vals["writes_max"], f(vals["bound_writes"], 1),
                         vals["n_regressions"], vals["n_refused"], f(vals["queries_firing_at_start_mean"], 1), f(vals["false_firing_start"]), f(vals["false_firing_after"]),
                         f(vals["false_firing_after_far"]), f(vals["false_firing_after_near"]), f(vals["ben_t_far_start"]), f(vals["ben_t_far_after"]), f(vals["bound_false_firing_pct"], 0)])
    TXT.append("E5 (d)  LEAST REPAIR ON REAL STREAMS (role last; 100 streams of N = 50 benign REPAIR queries, 20 arrival orders per stream; library least_repair, one query at a time)\n"
               "  identical: streams whose final thresholds are the same bits for all 20 orders; writes: threshold writes per stream and order (mean, max) against T (1 + ... + 1/N), a bound on the\n"
               "  mean over arrival orders; regressions: certified queries that fire again later; ff = firing on heldout_disjoint (percent; far / near probes) and BEN-T far rows (prompt-only keys)\n"
               "  before / after the stream, against min(1, T / (N + 1)).  logonly = the start the task names (silent on Qwen: nothing to repair); neginf = all thresholds -inf\n"
               + tt(["log", "start", "identical", "writes mean", "writes max", "bound", "regressions", "refused", "firing at start", "ff start", "ff after", "far", "near", "BEN-T far start",
                     "BEN-T far after", "bound %"], rows))


# ----------------------------------------------------------------------------- (e)
def part_e():
    p = f"{OUT}/e/rereport.json"
    if not os.path.exists(p):
        MISSING.append("e")
        return
    r = jload(p)
    ex = r["exactness"]
    src = "08_rescope/p11_scopecut/results.md"
    pn("cut.exact_instances", ex["n_equal_brute_force"], f"E5 (e) re-report: instances on which the minimum cut equals brute force ({src}:{ex['line']})")
    pn("cut.instances", ex["n_instances"], f"E5 (e) re-report: instances compared with brute force ({src}:{ex['line']})")
    rows = [[x["instance"], x["split"], x["rows_kind"][:28], x["edits"], x["rows"], x["graph_nodes"], x["graph_edges"], x["build_ms"], x["one_cut_ms_median"], x["one_cut_ms_max"], f"{src}:{x['line']}"]
            for x in r["run_time"]]
    big = [x for x in r["run_time"] if x["instance"] == "SYNTHETIC" and x["edits"] == 5000]
    if big:
        pn("cut.ms_at_5000_edits", big[0]["one_cut_ms_median"], f"E5 (e) re-report: median ms of one cut, synthetic instance with 5000 edits and 100000 rows ({src}:{big[0]['line']})")
    real = [x["one_cut_ms_median"] for x in r["run_time"] if "TB-" in x["instance"] and x["edits"] == 100]
    if real:
        pn("cut.ms_real_100_edits_max", max(real), f"E5 (e) re-report: largest median ms of one cut over the real instances with 100 edits ({src}:{r['run_time_header']['line']})")
    rows2 = [[x["log"], x["split"], f(x["scope_cut_firing_pct"], 1), f(x["at_coverage"]), f(x["indep_firing_pct"], 1), f(x["G-ABS"], 1), f(x["G-REL"], 1), f(x["G-COS"], 1), x["best_global"],
              f(x["ratio_cut_over_best_global"]), x["ratio_ci"], f(x["cut_over_indep"]), f"{src}:{x['line']}"] for x in r["literal_table"]]
    for x in r["literal_table"]:
        k = f"cut.oos.{x['log']}.{x['split']}"
        for a, b in (("scope_cut_firing_pct", "min_cut"), ("indep_firing_pct", "independent"), ("G-ABS", "shared_abs"), ("G-REL", "shared_rel"), ("ratio_cut_over_best_global", "ratio_to_best_shared")):
            pn(f"{k}.{b}", x[a], f"E5 (e) re-report: firing test rows (percent) at 70 % test-witness coverage, {x['log']} split {x['split']}, {b} ({src}:{x['line']})")
    TXT.append(f"E5 (e)  MINIMUM CUT: RE-REPORT of the verified record {src} (nothing new is run, R19)\n"
               f"  exactness ({src}:{ex['line']}): min cut == brute force on {ex['n_equal_brute_force']}/{ex['n_instances']} instances; independent per-edit thresholds <= optimum on "
               f"{ex['indep_at_most_optimum']}/{ex['n_instances']} (equal on {ex['indep_equal_optimum']})\n  run time ({src}:{r['run_time_header']['line']}):\n"
               + tt(["instance", "split", "rows", "edits", "rows N", "nodes", "edges", "build ms", "cut ms med", "cut ms max", "file:line"], rows)
               + f"\n  out of sample, firing test rows (percent) at 70 % test-witness coverage; verdict {r['cut_pass'].get('verdict')} ({src}:{r['cut_pass'].get('line')}); NR = coverage not reached on the lambda grid:\n"
               + tt(["log", "split", "min cut", "at coverage", "independent", "shared abs", "shared rel", "cosine", "best shared", "cut / best", "95 % CI", "cut / indep", "file:line"], rows2)
               + "\n  " + "\n  ".join(f"{src}:{q['line']}: {q['text'][:330]}" for q in r["quotes"][:3])
               + "\n  not in the record and not run: " + r["not_in_the_record"])


# ----------------------------------------------------------------------------- (f)
def part_f():
    p_pred = f"{_RP.ROOT}/03_analysis/theory/round4_review/P1b_capacity/predictions_P1b.json"
    if not os.path.exists(p_pred):                                      # ARTIFACT: stop instead of writing the summaries without the capacity numbers
        raise SystemExit(f"e5 render: input missing: {p_pred}\n  It is experiments/e5_rescope/inputs/capacity_predictions.json of the repository, mounted at this path by "
                         "rescope_paths.py (table LAYOUT).  Restore that file (git checkout) and run again.  Nothing was written.")
    pred = jload(p_pred)
    out, lines = {"doc": "E5 (f), amendment R20: capacity curve of linear certification; code/e5f_capacity.py (definitions in its docstring). largest_n_all_edits_preserved: the largest n of "
                         "the grid (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60) up to which the minimum-norm weight that restores the n complaints at all positions and keeps the solver "
                         "keys (keep_greedy) preserves all T edits (|X k_i| <= 1e-6 |D k_i|): minimum over the three seeded orders; the three values next to it.",
                  "prediction_file": "03_analysis/theory/round4_review/P1b_capacity/predictions_P1b.json"}, []
    for name in DEVS:
        seeds = {}
        for s in (0, 1, 2):
            p = f"{OUT}/f/{name}__s{s}.json"
            if os.path.exists(p):
                seeds[s] = jload(p)
            else:
                MISSING.append(f"f: {name} order {s}")
        ndp = f"{OUT}/f/{name}__nd.json"
        if not seeds or not os.path.exists(ndp):
            continue
        nd = jload(ndp)
        T, d = nd["T"], nd["d_in"]
        e = {k: v for k, v in nd.items() if k.startswith("nd_")}
        e.update(T=T, d_in=d)
        chk = f"{OUT}/f/{name}__p12check.json"
        if os.path.exists(chk):
            e["check_against_p12_record"] = jload(chk)
        L1 = pred["logs"][name]["L1_predictions"]
        largest, first_w0, by_n = {}, {}, {}
        for s, r in seeds.items():
            ok = [gq["n"] for gq in r["grid"] if gq["keep_greedy"]["n_preserved"] == T]
            notok = [gq["n"] for gq in r["grid"] if gq["keep_greedy"]["n_preserved"] < T]
            lim = min(notok) if notok else None
            largest[s] = max([n for n in ok if lim is None or n < lim], default=0)
            full = [gq["n"] for gq in r["grid"] if gq["rank_KQ"] >= d]
            first_w0[s] = min(full) if full else None
            for gq in r["grid"]:
                by_n.setdefault(gq["n"], {})[s] = gq
        exact_all, exact_w0, cap_is_min = {}, {}, True
        for s_, r in seeds.items():                                    # from the rank after every complaint (cap = min(T, d - rank) is checked at every grid point)
            ranks = [x["rank_after"] for x in r["per_complaint"]]
            exact_all[s_] = max([i + 1 for i, rk in enumerate(ranks) if rk <= d - T], default=0)
            exact_w0[s_] = min([i + 1 for i, rk in enumerate(ranks) if rk >= d], default=None)
            cap_is_min &= all(gq["cap"] == min(T, d - gq["rank_KQ"]) for gq in r["grid"])
        e["cap_equals_min_T_d_minus_rank_at_every_grid_point"] = bool(cap_is_min)
        e["largest_n_with_rank_KQ_at_most_d_minus_T_by_order"] = [exact_all.get(s_) for s_ in (0, 1, 2)]
        e["smallest_n_with_rank_KQ_equal_d_by_order"] = [exact_w0.get(s_) for s_ in (0, 1, 2)]
        e["largest_n_all_edits_preserved"] = min(largest.values()) if len(largest) == 3 else None
        e["largest_n_all_edits_preserved_by_order"] = [largest.get(s) for s in (0, 1, 2)]
        e["largest_n_all_edits_preserved_range"] = f"{min(largest.values())}-{max(largest.values())}" if largest else None
        e["smallest_grid_n_with_rank_KQ_equal_d_by_order"] = [first_w0.get(s) for s in (0, 1, 2)]
        e["n_orders_finished"] = len(seeds)
        e["rank_stable_over_two_decades_by_order"] = [seeds[s]["rank_stable_over_two_decades"] if s in seeds else None for s in (0, 1, 2)]
        e["residual_singular_values_in_bands_by_order"] = [seeds[s]["residual_singular_values_in_bands"] if s in seeds else None for s in (0, 1, 2)]
        e["tolerance_rank"], e["tolerance_preserved"] = 1e-9, 1e-6
        pr = {}
        for s in (0, 1, 2):
            ps = L1.get(f"s{s}", {})
            n_all = ps.get("N_all_T_edits_kept_own_keys__floor((d-T)/rate)")
            n_w0 = ps.get("N_from_which_only_W0_certifies__ceil(d/rate)")
            pr[s] = {"predicted_largest_n_all_edits": n_all, "predicted_first_n_only_W0": n_w0}
            if s in seeds:
                grid = [gq["n"] for gq in seeds[s]["grid"]]
                exp_ok = max([n for n in grid if n <= n_all], default=0)
                pr[s].update(measured_largest_grid_n_all_edits=largest[s], grid_value_the_prediction_implies=exp_ok, hit_all_edits=bool(largest[s] == exp_ok),
                             measured_first_grid_n_rank_d=first_w0[s], grid_value_the_prediction_implies_for_W0=min([n for n in grid if n >= n_w0], default=None),
                             hit_only_W0=bool(first_w0[s] == min([n for n in grid if n >= n_w0], default=None)),
                             every_complaint_largest_n_all_edits=exact_all[s], every_complaint_first_n_rank_d=exact_w0[s],
                             error_all_edits_in_complaints=exact_all[s] - n_all, error_only_W0_in_complaints=None if exact_w0[s] is None else exact_w0[s] - n_w0)
        e["prediction"] = {"by_order": pr, "pooled": L1.get("pooled"), "statement": "all T edits can survive up to 23 to 34 complaints (Qwen) / about 35 (LLaVA); from about 25 to 36 complaints on only "
                           "the pre-edit weight restores them (PROTOCOL.md R20)"}
        e["prediction_hit"] = bool(all(v.get("hit_all_edits") for v in pr.values() if "hit_all_edits" in v)) if len(seeds) == 3 else None
        e["grid"] = {str(n): {f"s{s}": {"n_keys": gq["n_keys"], "rank_KQ": gq["rank_KQ"], "cap": gq["cap"], "cap_by_tolerance": gq["cap_by_tolerance"],
                                         "preserved_keep_greedy": gq["keep_greedy"]["n_preserved"], "preserved_keep_greedy_tol_1e-9": gq["keep_greedy"]["n_preserved_tol_1e-9"],
                                         "preserved_keep_greedy_tol_1e-3": gq["keep_greedy"]["n_preserved_tol_1e-3"], "preserved_keep_ls": gq["keep_ls"]["n_preserved"],
                                         "preserved_plain": gq["plain"]["n_preserved"], "plain_rel_change_median": gq["plain"]["rel_change_median"],
                                         "restoration_rel_err_last_trace": gq["keep_greedy"].get("restoration_rel_err_last_trace"),
                                         "thresholds_sacrificing_silenced_solver_key": gq["thresholds"]["sacrificing"]["n_silenced_at_solver_key"],
                                         "thresholds_sacrificing_silenced_own_request": gq["thresholds"]["sacrificing"]["n_silenced_on_own_request"],
                                         "thresholds_protecting_refused_complaints": gq["thresholds"]["protecting"]["n_complaints_refused"],
                                         "thresholds_protecting_silenced": gq["thresholds"]["protecting"]["n_silenced_at_solver_key"] + gq["thresholds"]["protecting"]["n_silenced_on_own_request"]}
                              for s, gq in sorted(v.items())} for n, v in sorted(by_n.items())}
        out[name] = e
        rows = []
        for n, v in sorted(by_n.items()):
            def col(fn):
                return " / ".join(str(fn(v[s])) if s in v else "-" for s in (0, 1, 2))
            predn = " / ".join(str(100 if n <= pr[s]["predicted_largest_n_all_edits"] else (0 if n >= pr[s]["predicted_first_n_only_W0"] else "partial")) for s in (0, 1, 2))
            rows.append([n, col(lambda q: q["n_keys"]), col(lambda q: q["rank_KQ"]), col(lambda q: q["cap"]), predn, col(lambda q: q["keep_greedy"]["n_preserved"]),
                         col(lambda q: q["keep_ls"]["n_preserved"]), col(lambda q: q["plain"]["n_preserved"]), col(lambda q: f"{q['plain']['rel_change_median']:.2f}"),
                         col(lambda q: q["thresholds"]["sacrificing"]["n_silenced_at_solver_key"]), col(lambda q: q["thresholds"]["sacrificing"]["n_silenced_on_own_request"]),
                         col(lambda q: q["thresholds"]["protecting"]["n_complaints_refused"])])
        for i_, s_ in enumerate((0, 1, 2)):
            pn(f"capacity.{name}.largest_n_all_edits_preserved.order{s_}", e["largest_n_all_edits_preserved_by_order"][i_], f"E5 (f), {name}, order {s_}: largest n of the grid with all T edits preserved")
            pn(f"capacity.{name}.largest_n_rank_at_most_d_minus_T.order{s_}", e["largest_n_with_rank_KQ_at_most_d_minus_T_by_order"][i_], f"E5 (f), {name}, order {s_}: largest number of complaints with rank K_Q <= d_in - T (from the rank after every complaint)")
            pn(f"capacity.{name}.predicted_largest_n_all_edits.order{s_}", pr[s_]["predicted_largest_n_all_edits"], f"prediction written before the measurement (predictions_P1b.json), {name}, order {s_}")
        for k in ("largest_n_all_edits_preserved", "nd_holds", "nd_rank_write_vectors", "nd_coordinate_matrix_smallest_singular_value", "nd_coordinate_matrix_condition_number", "prediction_hit"):
            v = e[k]
            pn(f"capacity.{name}.{k}", (str(v) if isinstance(v, bool) else v), f"E5 (f), {name}: {k} (out/capacity_curve.json)")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                R.add(name, "linear_certification", f"capacity.{k}", v, pool="repair D1 complaints, 3 orders")
        for n, v in sorted(by_n.items()):
            for s, gq in v.items():
                for k, val in (("rank_KQ", gq["rank_KQ"]), ("cap", gq["cap"]), ("preserved_keep_greedy", gq["keep_greedy"]["n_preserved"]), ("preserved_plain", gq["plain"]["n_preserved"]),
                               ("thresholds_silenced_solver_key", gq["thresholds"]["sacrificing"]["n_silenced_at_solver_key"]),
                               ("thresholds_silenced_own_request", gq["thresholds"]["sacrificing"]["n_silenced_on_own_request"])):
                    R.add(name, "linear_certification" if "thresholds" not in k else "thresholds_only_repair", f"capacity.{k}", val, n=n, draw=s, pool="repair D1 complaints", n_units=T)
        ck = e.get("check_against_p12_record", {})
        lines.append(f"E5 (f)  CAPACITY CURVE OF LINEAR CERTIFICATION: {name} (T = {T}, d_in = {d}); three seeded orders of REPAIR-pool D1 complaints (values: order 0 / 1 / 2)\n"
                     f"  hypothesis of the theorem: rank of the write vectors {e['nd_rank_write_vectors']} of {T}; coordinate matrix: smallest singular value {e['nd_coordinate_matrix_smallest_singular_value']:.3e}, "
                     f"condition number {e['nd_coordinate_matrix_condition_number']:.3e}; nd_holds = {e['nd_holds']} (float64 SVD, relative tolerance 1e-10)\n"
                     f"  capture check against P12 (ten complaints of order 0): query keys {ck.get('n_query_keys')} (stored {ck.get('stored_n_query_keys')}), rank after projecting K_ans "
                     f"{ck.get('rank_after_projecting_K_ans')} (stored {ck.get('stored_rank')}): match = {ck.get('match')}\n"
                     f"  rank tolerance 1e-9 x largest key norm; residual singular values in (1e-10, 1e-9] / (1e-9, 1e-8] x scale along the stream, per order: "
                     + " ; ".join(f"{b['in_1e-10_1e-9']} / {b['in_1e-9_1e-8']}" if b else "-" for b in e["residual_singular_values_in_bands_by_order"])
                     + f" -> rank stable over two decades: {e['rank_stable_over_two_decades_by_order']}\n"
                     "  cap = rank[K_Q, K_E] - rank K_Q; pred = edits the prediction says can be preserved (per order); keep = edits preserved by the minimum-norm restoring weight that keeps the\n"
                     "  solver keys (greedy subset / least squares over all); plain = by the minimum-norm restoring change -D P_Q alone, with the median relative change of the module output\n"
                     "  at the solver keys; thresholds-only repair of the deployed scoping from log-only thresholds: edits silenced at their solver key / on their own request (sacrificing form),\n"
                     "  complaints refused by the protecting form (it silences no edit)\n"
                     + tt(["n", "keys", "rank K_Q", "cap", "pred", "keep greedy", "keep ls", "plain", "plain rel change", "thr: solver key", "thr: own request", "thr: refused"], rows)
                     + f"\n  largest n of the grid with all {T} edits preserved, per order: {e['largest_n_all_edits_preserved_by_order']} (minimum {e['largest_n_all_edits_preserved']}); "
                     f"first grid n with rank K_Q = d_in: {e['smallest_grid_n_with_rank_KQ_equal_d_by_order']}\n"
                     f"  prediction (written before the measurement), per order: all edits up to n = {[pr[s]['predicted_largest_n_all_edits'] for s in (0, 1, 2)]}, only W_0 from n = "
                     f"{[pr[s]['predicted_first_n_only_W0'] for s in (0, 1, 2)]}; on the grid this implies {[pr[s].get('grid_value_the_prediction_implies') for s in (0, 1, 2)]} and "
                     f"{[pr[s].get('grid_value_the_prediction_implies_for_W0') for s in (0, 1, 2)]}: "
                     f"HIT per order (all edits) {[pr[s].get('hit_all_edits') for s in (0, 1, 2)]}, (only W_0) {[pr[s].get('hit_only_W0') for s in (0, 1, 2)]}"
                     + ("" if e["prediction_hit"] in (True, None) else "   ** THE PREDICTION IS MISSED on at least one order **")
                     + f"\n  from the rank after every complaint (cap = min(T, d_in - rank K_Q) at every grid point: {e['cap_equals_min_T_d_minus_rank_at_every_grid_point']}): largest n with rank K_Q <= d_in - T, per order: "
                     f"{e['largest_n_with_rank_KQ_at_most_d_minus_T_by_order']} (predicted {[pr[s]['predicted_largest_n_all_edits'] for s in (0, 1, 2)]}); smallest n with rank K_Q = d_in: "
                     f"{e['smallest_n_with_rank_KQ_equal_d_by_order']} (predicted {[pr[s]['predicted_first_n_only_W0'] for s in (0, 1, 2)]})")
    jdump(out, f"{OUT}/capacity_curve.json")
    if lines:
        with open(f"{OUT}/capacity_table.txt", "w") as fh:
            fh.write("\n\n".join(lines) + "\n")
        TXT.extend(lines)


def main():
    for fn in (part_a, part_b, part_c, part_d, part_e, part_f):
        try:
            fn()
        except Exception as ex:                                         # a part that cannot be rendered yet must not hide the others
            import traceback
            MISSING.append(f"{fn.__name__}: render error {ex!r}")
            traceback.print_exc()
    head = ("E5  CHECKS OF THE THEORY ON REAL LOGS (rescoping part).  Rendered by code/render.py from out/{a,b,c,d,e,f}/ on " + time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
            + f"\n    driver fx {fx.code_fingerprint()}; gate level throughout except (b) deployed / where_logonly and the capture of (f) (GPU).  Not finished / missing: "
            + ("; ".join(MISSING) if MISSING else "nothing"))
    if any("render error" in m for m in MISSING):                       # ARTIFACT: a part that failed must not silently drop its paper numbers
        raise SystemExit("e5 render: stopped, nothing written: " + "; ".join(m for m in MISSING if "render error" in m)
                         + "\n  Inputs: out/{a,b,c,d,e,f}/ from results_rows.tar.gz (unpacked into the clone) and experiments/e5_rescope/inputs/.")
    R.save(missing=MISSING)
    R.table(head + "\n\n" + "\n\n".join(TXT))
    fx.io.jdump(dict(sorted(PN.items())), f"{OUT}/paper_numbers.json", indent=1)
    fx.io.jdump(dict(sorted(DOC.items())), f"{OUT}/paper_numbers_doc.json", indent=1)
    print(head)
    print(len(R.records), "records;", len(PN), "paper numbers")


if __name__ == "__main__":
    main()
