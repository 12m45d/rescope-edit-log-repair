"""Independent verification of E2 a-c: CPU analysis and rendering.  Reads my own GPU rows (out/<log>/tf_chunk_*.json.gz, written by
vjob.py through the library's plain Evaluator), the experiment's per-row files, cells and results.json, the P12 records, the sanity
run e0.  Writes out/results.json (fx.io schema), out/rows/, out/comparisons.json, out/table.txt, out/paper_numbers.json (+ _doc),
verdicts.json.  Every number printed anywhere comes from here.

    python vcpu.py            (safe at any time: a log whose GPU job has not finished is listed as missing)
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob
import gzip
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np

FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
EXP = FIN + "/e2_curves"
VER = EXP + "/verify"
OUT = VER + "/out"
P12 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p12_linear_streams"
PY = f"{_RP.PYTHON}"
sys.path.insert(0, FIN + "/common")
import fx                                                                # fx.io: the ONE result schema; fx.methods on the CPU only for the threshold comparison

LOGS = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
LINEAR = ("COMP_B", "COMP_BP", "COMP_B_ALLPOS", "REEDIT_B", "RB_TOT", "RB_UNION")
REGISTERED = ("COMP_B", "REEDIT_B", "RB_TOT", "RB_UNION")                # PROTOCOL section 2
CELLS = {"deployed": ("base", "deployed"), "wl_dep_n10": ("b", "where_least__deployed__n10__s0"), "wl_dep_n100": ("b", "where_least__deployed__n100__s0"),
         "wl_log_n10": ("b", "where_least__logonly__n10__s0"), "comp_b_all": ("d", "COMP_B__nALL__s0"), "c_n0": ("c", "rescope_n0__s0"),
         "c_n10": ("c", "rescope_n10__s0"), "c_n200": ("c", "rescope_n200__d0")}
LABEL = {"deployed": "deployed log", "wl_dep_n10": "ReScope-W least repair, deployed start, n=10, seed 0", "wl_dep_n100": "ReScope-W least repair, deployed start, n=100, seed 0",
         "wl_log_n10": "ReScope-W least repair, log-only start, n=10, seed 0 (H1 cell)", "comp_b_all": "COMP-B rebuilt on ALL complaints, seed 0",
         "c_n0": "ReScope label-free, n=0 (log-only)", "c_n10": "ReScope label-free, n=10, draw 0", "c_n200": "ReScope label-free, n=200, draw 0"}


def jload(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as fh:
        return json.load(fh)


def jl(p):
    with gzip.open(p, "rt") as fh:
        return [json.loads(x) for x in fh]


def num(x):
    return float(x) if isinstance(x, str) else x


def boot(flags, units, B=2000, seed=0):
    """Percentile bootstrap of a rate in percent, whole clusters resampled (my own implementation)."""
    f = np.asarray(flags, dtype=np.float64)
    inv = np.unique(np.asarray([str(u) for u in units]), return_inverse=True)[1]
    nu = int(inv.max()) + 1
    s, c = np.bincount(inv, weights=f, minlength=nu), np.bincount(inv, minlength=nu).astype(np.float64)
    idx = np.random.default_rng(seed).integers(0, nu, (B, nu))
    st = 100.0 * s[idx].sum(1) / c[idx].sum(1)
    return {"value": 100.0 * float(f.mean()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5)), "n_units": nu, "n_rows": int(f.size)}


def crossed_diff(Na, Da, Nb, Db, B=2000, seed=0):
    """Paired crossed bootstrap (seeds x clusters) of the difference of two rates in percent (my own implementation)."""
    nd, nu = Na.shape
    rng = np.random.default_rng(seed)
    st = np.empty(B)
    for b in range(B):
        d, u = rng.integers(0, nd, nd), rng.integers(0, nu, nu)
        st[b] = 100.0 * (Na[d][:, u].sum() / Da[d][:, u].sum() - Nb[d][:, u].sum() / Db[d][:, u].sum())
    p = 2.0 * min(float((st <= 0).mean()), float((st >= 0).mean()))
    return {"value": 100.0 * float(Na.sum() / Da.sum() - Nb.sum() / Db.sum()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5)),
            "p_two_sided": max(p, 1.0 / B), "n_draws": nd, "n_units": nu}


def summ(rows):
    """The headline numbers of one cell from teacher-forced rows (mine or the experiment's)."""
    tf = [r for r in rows if "line" not in r and not r.get("d2_row")]
    e = [r for r in tf if r["kind"] == "edit"]
    p = [r for r in tf if r["kind"] == "probe" and r.get("disjoint")]
    out = {"n_edit_rows": len(e), "n_heldout_disjoint": len(p), "n_heldout_strict": sum(bool(r.get("strict")) for r in p)}
    out["D1"] = boot([bool(r["D1"]) for r in p], [r["query_id"] for r in p])
    ps = [r for r in p if r.get("strict")]
    out["D1_strict"] = boot([bool(r["D1"]) for r in ps], [r["query_id"] for r in ps])
    for m in ("rel", "tgen", "igen"):
        sub = [r for r in e if r["metric"] == m]
        out[m] = boot([bool(r["exact"]) for r in sub], [r["edit"] for r in sub])
    rel = [r for r in e if r["metric"] == "rel"]
    out["own_out"] = sum(not r["own_fired_last"] for r in rel) if rel and rel[0].get("own_fired_last") is not None else None
    out["fired"] = 100.0 * float(np.mean([bool(r["fired"]) for r in p])) if p and p[0].get("fired") is not None else None
    return out


def agree(mine, theirs):
    """Row-level agreement of the flags of two row sets joined by the library sid (edits + heldout_disjoint probes)."""
    a = {r["sid"]: r for r in mine if "line" not in r and not r.get("d2_row") and (r["kind"] == "edit" or (r["kind"] == "probe" and r.get("disjoint")))}
    b = {r["sid"]: r for r in theirs if "line" not in r and not r.get("d2_row") and (r["kind"] == "edit" or (r["kind"] == "probe" and r.get("disjoint")))}
    out = {"n_mine": len(a), "n_theirs": len(b), "n_joined": len(set(a) & set(b)), "same_sids": set(a) == set(b)}
    for kind in ("edit", "probe"):
        ks = [k for k in a if k in b and a[k]["kind"] == kind]
        out[f"{kind}_n"] = len(ks)
        out[f"{kind}_exact_differ"] = sum(bool(a[k]["exact"]) != bool(b[k]["exact"]) for k in ks)
        fk = [k for k in ks if a[k].get("fired") is not None and b[k].get("fired") is not None]
        out[f"{kind}_fired_n"] = len(fk)
        out[f"{kind}_fired_differ"] = sum(bool(a[k]["fired"]) != bool(b[k]["fired"]) for k in fk)
        out[f"{kind}_differing"] = [k for k in ks if bool(a[k]["exact"]) != bool(b[k]["exact"])][:12]
    ok = [k for k in a if k in b and a[k]["kind"] == "edit" and a[k].get("own_fired_last") is not None and b[k].get("own_fired_last") is not None]
    out["own_fired_last_differ"] = sum(bool(a[k]["own_fired_last"]) != bool(b[k]["own_fired_last"]) for k in ok)
    return out


def main():
    t0 = time.time()
    res = jload(EXP + "/out/results.json")
    idx = {}
    for r in res:
        idx[(r["log"], r.get("part"), r["method"], r["variant"], str(r.get("n_label")), r["draw"], r["pool"], r["metric"])] = r
    paper_exp = jload(EXP + "/out/paper_numbers.json")
    h1_exp = jload(EXP + "/out/h1.json")
    e0 = jload(FIN + "/e0_sanity/out/results.json")
    R = fx.io.Results("E2-verify", OUT)
    C = {"logs": {}, "missing": []}
    lines, paper, doc, checks = [], {}, {}, {}

    def pn(k, v, t):
        if v is not None:
            paper[k], doc[k] = v, t

    for name, sh in LOGS.items():
        cl = C["logs"].setdefault(name, {})
        pools = jload(f"{OUT}/{name}/pools.json") if os.path.exists(f"{OUT}/{name}/pools.json") else None
        if pools is None:
            C["missing"].append(f"{name}: no GPU output")
            continue
        g_dis = {h["group"]: h for h in pools["heldout"]}
        # ---------------------------------------------------------------- 1. my GPU cells against the experiment's
        mine = {}
        for p in sorted(glob.glob(f"{OUT}/{name}/tf_chunk_*.json.gz")):
            for k, v in jload(p).items():
                mine.setdefault(k, []).extend(v)
        n_expect = 300 + pools["n_heldout_valid"]
        complete = bool(mine) and all(len(v) == n_expect for v in mine.values())
        cl["gpu_pass_complete"] = complete
        if not complete:
            C["missing"].append(f"{name}: evaluation pass incomplete ({ {k: len(v) for k, v in mine.items()} } of {n_expect})")
        cells = {}
        th_mine = jload(f"{OUT}/{name}/thresholds.json")["thresholds"]
        L = fx.logs.open(name)
        for k, (part, tag) in CELLS.items():
            if k not in mine or not complete:
                continue
            meta = jload(f"{EXP}/out/cells/{part}/{name}/{tag}.json")
            theirs = jl(f"{EXP}/out/rows/{part}__{sh}__{tag}.jsonl.gz")
            d = meta.get("draw") if meta.get("draw") is not None else meta.get("seed")
            key = (name, part, meta["method"], meta.get("variant", ""), str(meta.get("n_label")), d)
            sm, st = summ(mine[k]), summ(theirs)
            rep = {}                                                    # the experiment's REPORTED numbers (results.json)
            for nm, pool, metric in (("D1", "heldout_disjoint", "probe.D1"), ("D1_strict", "heldout_strict", "probe.D1"), ("rel", "edits", "edit.rel.exact"),
                                     ("tgen", "edits", "edit.tgen.exact"), ("igen", "edits", "edit.igen.exact"), ("own_out", "edits", "edit.rel.n_own_out_of_scope"),
                                     ("fired", "heldout_disjoint", "probe.fired"), ("n", "heldout_disjoint", "probe.n")):
                r = idx.get(key + (pool, metric))
                rep[nm] = None if r is None else {"value": r["value"], "ci_lo": r.get("ci_lo"), "ci_hi": r.get("ci_hi")}
            ag = agree(mine[k], theirs)
            th = None
            if k in th_mine:                                            # thresholds: mine against the experiment's
                tm = np.array([num(x) for x in th_mine[k]["tau"]], dtype=np.float64)
                if meta.get("theta") is not None:
                    tt = np.array([num(x) for x in meta["theta"]], dtype=np.float64)
                    src = "cell file"
                else:
                    n = {"c_n0": 0, "c_n10": 10, "c_n200": 200}[k]
                    tt = fx.methods.evidence(L, n, 0, 0.05).obj.theta().numpy()
                    src = "driver constructor rebuilt on the CPU"
                fin = np.isfinite(tm) & np.isfinite(tt)
                th = {"source_of_theirs": src, "n": int(tm.size), "n_both_finite": int(fin.sum()), "n_inf_pattern_differs": int((np.isfinite(tm) != np.isfinite(tt)).sum()),
                      "max_abs_diff": float(np.abs(tm[fin] - tt[fin]).max()) if fin.any() else None,
                      "max_rel_diff": float((np.abs(tm[fin] - tt[fin]) / np.abs(tt[fin]).clip(1e-300)).max()) if fin.any() else None,
                      "n_differ_more_than_1e-5_rel": int((np.abs(tm[fin] - tt[fin]) > 1e-5 * np.abs(tt[fin])).sum()),
                      "mine": {a: b for a, b in th_mine[k].items() if isinstance(b, (int, float))}}
            cells[k] = {"mine": sm, "theirs_rows": st, "theirs_reported": rep, "agreement": ag, "thresholds": th, "cell": f"{part}/{tag}",
                        "experiment_meta": {a: meta.get(a) for a in ("n_refused", "n_reported_still_firing", "n_thresholds_raised_total", "n_edits_capped", "flags_vs_record")}}
            kw = dict(variant="verify (plain Evaluator, own thresholds)", n=meta.get("n"), draw=d, alpha=meta.get("alpha"))
            R.add_stat(name, k, "probe.D1", sm["D1"], pool="heldout_disjoint", **kw)
            R.add_stat(name, k, "probe.D1", sm["D1_strict"], pool="heldout_strict", **kw)
            for m in ("rel", "tgen", "igen"):
                R.add_stat(name, k, f"edit.{m}.exact", sm[m], pool="edits", **kw)
            if sm["own_out"] is not None:
                R.add(name, k, "edit.rel.n_own_out_of_scope", float(sm["own_out"]), pool="edits", **kw)
                R.add(name, k, "probe.fired", sm["fired"], pool="heldout_disjoint", **kw)
            for a, b in (("edit_exact_differ", "edits"), ("probe_exact_differ", "heldout_disjoint"), ("edit_fired_differ", "edits"), ("probe_fired_differ", "heldout_disjoint")):
                R.add(name, k, "agree.n_" + a, float(ag[a]), pool=b, **kw)
            R.rows(f"{sh}__{k}", mine[k])
        cl["cells"] = cells
        # the rebuilt compensation against the ORIGINAL P12 record (all valid held-out rows, edits)
        if "comp_b_all" in mine and complete:
            rec = jload(f"{P12}/out/{name}/COMP_B__nALL__s0.json")
            ho = {h["group"]: bool(h["reproduced_after"]) for h in rec["eval"]["heldout_rows"]}
            pr = [r for r in mine["comp_b_all"] if r["kind"] == "probe"]
            ed = [r for r in mine["comp_b_all"] if r["kind"] == "edit"]
            cl["comp_b_all_vs_p12_record"] = {"n_heldout": len(pr), "n_heldout_in_record": len(ho), "n_heldout_differ": sum(bool(r["exact"]) != ho[r["group"]] for r in pr),
                                              "n_heldout_disjoint_differ": sum(bool(r["exact"]) != ho[r["group"]] for r in pr if r["disjoint"]),
                                              "n_edit": len(ed), "n_edit_differ": sum(bool(r["exact"]) != bool(rec["eval"]["edits"][r["metric"]]["em_list"][r["edit"]]) for r in ed),
                                              "solve_device": "see out/<log>/comp_b_all.pt (cpu as P12 on Qwen; the LLaVA build of this job ran on the GPU)"}
            dep = {h["group"]: bool(h["reproduced_WE"]) for h in rec["eval"]["heldout_rows"]}
            prd = [r for r in mine["deployed"] if r["kind"] == "probe"]
            cl["deployed_vs_p12_record"] = {"n_heldout": len(prd), "n_heldout_differ": sum(bool(r["exact"]) != dep[r["group"]] for r in prd)}
        # ---------------------------------------------------------------- 2. imported linear cells against the original records
        imp = {"n_cells": 0, "n_numbers": 0, "n_mismatch": 0, "max_abs_diff": 0.0, "examples": []}
        lin = {}
        for cf in sorted(glob.glob(f"{EXP}/out/cells/a/{name}/*.json")):
            meta = jload(cf)
            rec = jload(meta["file"])
            hr = rec["eval"]["heldout_rows"]
            dis = [h for h in hr if g_dis[h["group"]]["disjoint"]]
            st_ = [h for h in dis if g_dis[h["group"]]["strict"]]
            v = {("heldout_disjoint", "probe.D1"): 100.0 * np.mean([not h["reproduced_after"] for h in dis]),
                 ("heldout_strict", "probe.D1"): 100.0 * np.mean([not h["reproduced_after"] for h in st_]),
                 ("heldout_disjoint", "probe.n"): float(len(dis))}
            for m in ("rel", "tgen", "igen"):
                v[("edits", f"edit.{m}.exact")] = 100.0 * np.mean([bool(x) for x in rec["eval"]["edits"][m]["em_list"]])
            imp["n_cells"] += 1
            key = (name, "a", meta["method"], meta.get("variant", ""), str(meta.get("n_label")), meta.get("seed"))
            for (pool, metric), x in v.items():
                r = idx.get(key + (pool, metric))
                imp["n_numbers"] += 1
                dv = None if r is None or r["value"] is None else abs(float(r["value"]) - float(x))
                if dv is None or dv > 1e-9:
                    imp["n_mismatch"] += 1
                    imp["examples"].append([meta["tag"], pool, metric, None if r is None else r["value"], float(x)])
                else:
                    imp["max_abs_diff"] = max(imp["max_abs_diff"], dv)
            lin[(meta["method"], str(meta.get("n_label")), meta.get("seed"))] = {"d1": float(v[("heldout_disjoint", "probe.D1")]), "rel": float(v[("edits", "edit.rel.exact")]),
                                                                              "tgen": float(v[("edits", "edit.tgen.exact")]), "igen": float(v[("edits", "edit.igen.exact")])}
        imp["examples"] = imp["examples"][:10]
        cl["linear_import"] = imp
        # summary rows of the linear part that the experiment's summary quotes (mean over the seeds of the records)
        lsum = {}
        for m, nl in (("COMP_B", "10"), ("RB_UNION", "10"), ("REEDIT_B", "10"), ("COMP_B", "ALL"), ("REEDIT_B", "ALL"), ("RB_UNION", "ALL")):
            ss = [lin[k] for k in lin if k[0] == m and k[1] == nl]
            mine_d1, mine_rel = float(np.mean([s["d1"] for s in ss])), float(np.mean([s["rel"] for s in ss]))
            r1 = idx.get((name, "a", m, "p12_record", nl, None, "heldout_disjoint", "probe.D1@mean"))
            r2 = idx.get((name, "a", m, "p12_record", nl, None, "edits", "edit.rel.exact@mean"))
            lsum[f"{m}@{nl}"] = {"n_seeds": len(ss), "d1_mine": mine_d1, "d1_exp": None if r1 is None else r1["value"], "rel_mine": mine_rel, "rel_exp": None if r2 is None else r2["value"]}
            R.add(name, m, "probe.D1@mean", mine_d1, variant="p12_record, recomputed by the verifier", n=None, pool="heldout_disjoint", n_label=nl, n_seeds=len(ss))
            R.add(name, m, "edit.rel.exact@mean", mine_rel, variant="p12_record, recomputed by the verifier", n=None, pool="edits", n_label=nl, n_seeds=len(ss))
        cl["linear_summary"] = lsum
        # ---------------------------------------------------------------- 3. H1 from the per-row files, rule applied literally
        qid = {h["group"]: h["query_id"] for h in pools["heldout"] if h["disjoint"]}
        units = sorted({str(q) for q in qid.values()})
        ui = {u: i for i, u in enumerate(units)}

        def grid_w(s):
            rows = [r for r in jl(f"{EXP}/out/rows/b__{sh}__where_least__logonly__n10__s{s}.jsonl.gz") if r["kind"] == "probe" and r.get("disjoint") and "line" not in r and not r.get("d2_row")]
            N, D = np.zeros(len(units)), np.zeros(len(units))
            for r in rows:
                N[ui[str(qid[r["group"]])]] += float(bool(r["D1"]))
                D[ui[str(qid[r["group"]])]] += 1
            return N, D, rows

        def grid_p12(m, s):
            rec = jload(f"{P12}/out/{name}/{m}__n10__s{s}.json")
            N, D = np.zeros(len(units)), np.zeros(len(units))
            for h in rec["eval"]["heldout_rows"]:
                if h["group"] in qid:
                    N[ui[str(qid[h["group"]])]] += float(not h["reproduced_after"])
                    D[ui[str(qid[h["group"]])]] += 1
            return N, D, 100.0 * float(np.mean([bool(x) for x in rec["eval"]["edits"]["rel"]["em_list"]]))

        gw = [grid_w(s) for s in (0, 1, 2)]
        Nw, Dw = np.stack([g[0] for g in gw]), np.stack([g[1] for g in gw])
        d1_w = 100.0 * float(Nw.sum() / Dw.sum())
        rel_w = float(np.mean([100.0 * np.mean([bool(r["exact"]) for r in jl(f"{EXP}/out/rows/b__{sh}__where_least__logonly__n10__s{s}.jsonl.gz")
                                                if r["kind"] == "edit" and r["metric"] == "rel" and "line" not in r]) for s in (0, 1, 2)]))
        rel_dep = 100.0 * float(np.mean([bool(r["exact"]) for r in jl(f"{EXP}/out/rows/base__{sh}__deployed.jsonl.gz") if r["kind"] == "edit" and r["metric"] == "rel" and "line" not in r]))
        cand = {}
        allm = sorted({os.path.basename(p).split("__n10__")[0] for p in glob.glob(f"{P12}/out/{name}/*__n10__s0.json")})
        for m in allm:
            if not all(os.path.exists(f"{P12}/out/{name}/{m}__n10__s{s}.json") for s in (0, 1, 2)):
                continue
            g = [grid_p12(m, s) for s in (0, 1, 2)]
            N, D = np.stack([x[0] for x in g]), np.stack([x[1] for x in g])
            cand[m] = {"d1": 100.0 * float(N.sum() / D.sum()), "rel": float(np.mean([x[2] for x in g])), "N": N, "D": D}
        best = {"registered_set_of_the_experiment": min((m for m in cand if m in LINEAR), key=lambda m: cand[m]["d1"]),
                "protocol_section_2_set": min((m for m in cand if m in REGISTERED), key=lambda m: cand[m]["d1"]),
                "every_p12_method_with_3_seeds": min(cand, key=lambda m: cand[m]["d1"])}
        okrel = [m for m in cand if m in LINEAR and cand[m]["rel"] >= rel_dep - 2.0]
        best_rel = min(okrel, key=lambda m: cand[m]["d1"]) if okrel else None
        he = h1_exp["logs"][name]
        comp = {}
        for lab, m in (("best_lowest_d1", best["registered_set_of_the_experiment"]), ("best_rel_within_2", best_rel)):
            if m is None:
                continue
            dd = crossed_diff(Nw, Dw, cand[m]["N"], cand[m]["D"])
            ex = he["comparisons"].get(lab, {})
            comp[lab] = {"linear_mine": m, "linear_exp": ex.get("linear"), "d1_linear_mine": cand[m]["d1"], "d1_linear_exp": ex.get("d1_linear"), "diff_mine": dd,
                         "diff_exp": {k: ex.get("d1_diff", {}).get(k) for k in ("value", "ci_lo", "ci_hi", "p_two_sided")}}
        a, b2, b1 = d1_w <= 10.0, abs(rel_w - rel_dep) <= 2.0, rel_w >= rel_dep - 2.0
        h1 = {"d1_where_mine": d1_w, "d1_where_exp": he["d1_where"], "rel_where_mine": rel_w, "rel_where_exp": he["rel_where"], "rel_deployed_mine": rel_dep,
              "rel_deployed_exp": he["rel_deployed"], "best_linear": best, "candidates": {m: {"d1": c["d1"], "rel": c["rel"]} for m, c in cand.items()}, "comparisons": comp,
              "d1_at_most_10": bool(a), "rel_within_2_two_sided": bool(b2), "rel_not_more_than_2_below": bool(b1),
              "literal_rule_helps_on_this_log_mine": bool(a and b2), "literal_rule_helps_on_this_log_exp": bool(he["helps_on_this_log"]),
              "one_sided_met_mine": bool(a and b1), "one_sided_met_exp": bool(he["one_sided_reading_met_on_this_log"]),
              "where_d1_per_seed": [100.0 * float(Nw[i].sum() / Dw[i].sum()) for i in range(3)]}
        if "wl_log_n10" in cells:                                       # the H1 cell of seed 0 recomputed on the GPU by me
            h1["seed0_d1_mine_gpu"] = cells["wl_log_n10"]["mine"]["D1"]["value"]
            h1["seed0_rel_mine_gpu"] = cells["wl_log_n10"]["mine"]["rel"]["value"]
            h1["deployed_rel_mine_gpu"] = cells["deployed"]["mine"]["rel"]["value"]
        cl["h1"] = h1
        R.add(name, "where_least", "h1.probe.D1@mean", d1_w, variant="logonly, from the experiment's per-row files", n=10, pool="heldout_disjoint", n_seeds=3)
        R.add(name, "where_least", "h1.edit.rel.exact@mean", rel_w, variant="logonly, from the experiment's per-row files", n=10, pool="edits", n_seeds=3)
        for lab, c in comp.items():
            R.add_stat(name, "where_least", f"h1.diff.{lab}", {k: c["diff_mine"][k] for k in ("value", "ci_lo", "ci_hi", "n_units")}, variant=f"minus {c['linear_mine']}", n=10,
                       pool="heldout_disjoint", n_seeds=3, p_two_sided=c["diff_mine"]["p_two_sided"])
        # ---------------------------------------------------------------- 4. the sanity run e0
        bm = e0["logs"][name]["behaviour"]["methods"]
        v = lambda x: x["value"] if isinstance(x, dict) else x          # noqa: E731
        san = []
        for k, a_ in (("deployed", bm["deployed"]["single"]), ("c_n200", bm["M1"]["draws"]["0"])):
            if k not in cells:
                continue
            sm = cells[k]["mine"]
            for nm, x, y in (("D1 heldout_disjoint", a_["probe"]["heldout_disjoint"]["D1"], sm["D1"]["value"]), ("D1 heldout_strict", a_["probe"]["heldout_strict"]["D1"], sm["D1_strict"]["value"]),
                             ("Rel", a_["edit"]["rel"]["exact"], sm["rel"]["value"]), ("T-Gen", a_["edit"]["tgen"]["exact"], sm["tgen"]["value"]),
                             ("I-Gen", a_["edit"]["igen"]["exact"], sm["igen"]["value"]), ("fired heldout_disjoint", a_["probe"]["heldout_disjoint"].get("fired"), sm["fired"])):
                if x is not None and y is not None:
                    san.append({"cell": k, "quantity": nm, "e0": float(v(x)), "mine": float(y), "abs_diff": abs(float(v(x)) - float(y))})
        cl["sanity"] = {"rows": san, "max_abs_diff": max((s["abs_diff"] for s in san), default=None), "n": len(san)}
        # ---------------------------------------------------------------- 5. counts
        cl["counts"] = {"mine": {k: pools[k] for k in ("n_probe_rows", "n_probe_valid", "n_heldout_valid", "n_heldout_disjoint", "n_heldout_strict", "n_repair_valid")},
                        "n_complaints_all_p12": len(jload(f"{P12}/out/{name}/COMP_B__nALL__s0.json")["reported_groups"]),
                        "exp_probe_n_values": sorted({r["value"] for r in res if r["log"] == name and r["metric"] == "probe.n" and r["pool"] == "heldout_disjoint"}),
                        "exp_strict_n_values": sorted({r["value"] for r in res if r["log"] == name and r["metric"] == "probe.n" and r["pool"] == "heldout_strict"}),
                        "exp_edit_n_values": sorted({r["value"] for r in res if r["log"] == name and r["metric"] in ("edit.rel.n", "edit.tgen.n", "edit.igen.n")}),
                        "exp_n_records": sum(r["log"] == name for r in res)}
        print(name, "analysed", f"{time.time() - t0:.0f}s", flush=True)

    # -------------------------------------------------------------------- 6. running example
    exq = f"{OUT}/TB-Q100-ae-l2_3/example.json"
    if os.path.exists(exq):
        me, ex = jload(exq), jload(FIN + "/out/running_example.json")
        ex2 = jload(EXP + "/out/running_example.json")
        rows = [("complaint sid", ex["_selection"]["complaint_sid"].split(":", 1)[1], me["selection"]["sid"]),
                ("stream position (seed 0)", ex["_selection"]["stream_position_seed0"], me["selection"]["stream_position_seed0"]),
                ("complaint.query", ex["complaint"]["query"], me["complaint"]["query"]), ("complaint.pre_answer", ex["complaint"]["pre_answer"], me["complaint"]["pre_answer"]),
                ("complaint.deployed_answer", ex["complaint"]["deployed_answer"], me["complaint"]["deployed_answer"]),
                ("complaint.max_coordinate", ex["complaint"]["max_coordinate"], me["complaint"]["max_coordinate"]),
                ("edit index", ex["_selection"]["edit_index"], me["edit_index"]),
                ("edit.target", ex["edit"]["target"], next((v for k, v in me["edit_item"].items() if v == ex["edit"]["target"]), None)),
                ("edit.question", ex["edit"]["question"], next((v for k, v in me["edit_item"].items() if v == ex["edit"]["question"]), None)),
                ("rollback.n_deleted", ex["rollback"]["n_deleted"], me["rollback"]["n_deleted"]), ("rollback removal set", sorted(ex["_details"]["rollback"]["R"]), sorted(me["rollback"]["R"])),
                ("rollback.edits_broken", ex["rollback"]["edits_broken"], me["rollback"]["edits_broken"]),
                ("compensation.heldout_changed", ex["compensation"]["heldout_changed"], me["compensation"]["heldout_changed"]), ("heldout.n", ex["heldout"]["n"], me["heldout"]["n"]),
                ("rescope.own_requests_firing", ex["rescope"]["own_requests_firing"], me["rescope"]["own_requests_firing"]),
                ("whynot edit", ex["_details"]["whynot"]["edit"], me["whynot"]["edit"]),
                ("whynot.query", ex["whynot"]["query"], next((v for k, v in me["whynot"]["edit_item"].items() if v == ex["whynot"]["query"]), None)),
                ("whynot.answer", ex["whynot"]["answer"], me["whynot"]["answer"]), ("whynot.coordinate", ex["whynot"]["coordinate"], me["whynot"]["coordinate"]),
                ("whynot.threshold", ex["whynot"]["threshold"], me["whynot"]["threshold"])]

        def same(a, b):
            if isinstance(a, float) or isinstance(b, float):
                return a is not None and b is not None and abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)))
            return a == b
        C["example"] = {"rows": [{"key": k, "experiment": a, "mine": b, "agree": bool(same(a, b))} for k, a, b in rows],
                        "two_copies_of_the_experiment_file_equal": {k: ex[k] for k in ex if k != "written"} == {k: ex2[k] for k in ex2 if k != "written"},
                        "mine_details": {"rollback": me["rollback"], "rescope": me["rescope"], "compensation": me["compensation"], "edit_chosen_by": me["edit_chosen_by"]},
                        "n_agree": sum(bool(same(a, b)) for _, a, b in rows), "n": len(rows)}
    else:
        C["missing"].append("running example (Qwen GPU job)")

    # -------------------------------------------------------------------- 7. internal consistency: re-run of the experiment's rendering
    rr = f"{OUT}/rerender"
    os.makedirs(rr, exist_ok=True)
    for d in ("cells", "rows", "parts", "capture"):
        if not os.path.lexists(f"{rr}/{d}"):
            os.symlink(f"{EXP}/out/{d}", f"{rr}/{d}")
    if not os.path.exists(f"{rr}/results.json") or "--rerender" in sys.argv:
        env = dict(os.environ, E2_OUT=rr)
        pr = subprocess.run([PY, f"{EXP}/code/render.py"], env=env, capture_output=True, text=True)
        open(f"{VER}/logs/rerender.log", "w").write(pr.stdout[-4000:] + "\n" + pr.stderr[-4000:])
    ren = {"ran": os.path.exists(f"{rr}/results.json")}
    if ren["ran"]:
        r2 = jload(f"{rr}/results.json")
        strip = lambda x: json.dumps(x, sort_keys=True)                  # noqa: E731
        ren["results_json_n_records"] = [len(res), len(r2)]
        ren["results_json_equal"] = len(res) == len(r2) and all(strip(a) == strip(b) for a, b in zip(res, r2))
        p2 = jload(f"{rr}/paper_numbers.json")
        ren["paper_numbers_equal"] = p2 == paper_exp
        ren["paper_numbers_n"] = len(paper_exp)
        ren["paper_numbers_differing"] = [k for k in set(p2) | set(paper_exp) if p2.get(k) != paper_exp.get(k)][:20]
        t1, t2 = open(f"{EXP}/out/table.txt").read().split("\n"), open(f"{rr}/table.txt").read().split("\n")
        ren["table_lines"] = [len(t1), len(t2)]
        ren["table_lines_differing_after_header"] = sum(a != b for a, b in zip(t1[1:], t2[1:])) + abs(len(t1) - len(t2))
        h2 = jload(f"{rr}/h1.json")
        ren["h1_json_equal"] = strip(h2) == strip(h1_exp)
        for f in ("linear_rows.json", "rescope_w_rows.json"):
            ren[f + "_equal"] = strip(jload(f"{rr}/{f}")) == strip(jload(f"{EXP}/out/{f}"))
        doc_exp = jload(EXP + "/out/paper_numbers_doc.json")
        ren["paper_numbers_all_documented"] = set(doc_exp) == set(paper_exp)
        # every paper number against the record of results.json it names (value equality by search)
        vals = {}
        for r in res:
            if isinstance(r["value"], (int, float)):
                vals.setdefault(round(float(r["value"]), 9), 0)
        h1vals = set()

        def walk(x):
            if isinstance(x, dict):
                for y in x.values():
                    walk(y)
            elif isinstance(x, list):
                for y in x:
                    walk(y)
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                h1vals.add(round(float(x), 9))
        walk(h1_exp)
        for f in glob.glob(f"{EXP}/out/parts/cw_check_*.json"):
            walk(jload(f))
        nf = [k for k, x in paper_exp.items() if isinstance(x, (int, float)) and not isinstance(x, bool) and round(float(x), 9) not in vals and round(float(x), 9) not in h1vals]
        ren["paper_numbers_not_found_in_results_or_h1"] = nf
    C["rerender"] = ren
    fx.io.jdump(C, f"{OUT}/comparisons.json", indent=1)
    R.save(what="independent verification of E2 a-c", missing=C["missing"])
    render(C, R, paper, doc, pn)
    print("done", f"{time.time() - t0:.0f}s; missing: {C['missing']}")


def f1(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))


def render(C, R, paper, doc, pn):
    T = fx.io.text_table
    out = [f"VERIFICATION OF E2 a-c (evidence curves, linear rows, running example)      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by verify/code/vcpu.py",
           "mine = recomputed by the verifier: rescope.load + rescope.Evaluator.tf_rows at B = 1 (no driver, no memo, no key cache), thresholds computed by the verifier from",
           "coordinates captured by the verifier; seed / draw 0.  exp = the experiment's reported number (results.json).  D1 on heldout_disjoint, percentile bootstrap over",
           "query clusters (2000 resamples, the verifier's own generator: interval ends differ from the experiment's in the first decimal).  'flags differ' = rows whose",
           f"exact flag differs between my rows and the experiment's per-row file (edits of 300 / heldout_disjoint rows).  NOT FINISHED: {C['missing'] if C['missing'] else 'nothing'}", ""]
    verdicts = {}
    allok = {}
    for name, sh in LOGS.items():
        cl = C["logs"].get(name, {})
        if not cl.get("cells"):
            continue
        out += ["#" * 140, name, "#" * 140, "(1) headline cells: experiment's reported number | mine"]
        rows = []
        for k, c in cl["cells"].items():
            m, e, ag, th = c["mine"], c["theirs_reported"], c["agreement"], c["thresholds"]
            ev = lambda q: None if e.get(q) is None else e[q]["value"]   # noqa: E731
            ok = all(ev(q) is not None and abs(ev(q) - m[q]["value"]) < 1e-9 for q in ("D1", "D1_strict", "rel", "tgen", "igen")) and ag["edit_exact_differ"] == 0 and ag["probe_exact_differ"] == 0
            allok[(name, k)] = ok
            rows.append([LABEL[k], f"{f1(ev('D1'))} | {f1(m['D1']['value'])} [{f1(m['D1']['ci_lo'], 1)}, {f1(m['D1']['ci_hi'], 1)}]", f"{f1(ev('D1_strict'))} | {f1(m['D1_strict']['value'])}",
                         "/".join(f1(ev(q), 0) for q in ("rel", "tgen", "igen")) + " | " + "/".join(f1(m[q]["value"], 0) for q in ("rel", "tgen", "igen")),
                         f"{f1(ev('own_out'), 0)} | {f1(m['own_out'], 0)}", f"{f1(ev('fired'))} | {f1(m['fired'])}",
                         f"{ag['edit_exact_differ']}/{ag['edit_n']}; {ag['probe_exact_differ']}/{ag['probe_n']}", f"{ag['edit_fired_differ']}/{ag['edit_fired_n']}; {ag['probe_fired_differ']}/{ag['probe_fired_n']}",
                         "-" if th is None else f"{f1(th['max_abs_diff'], 10)} ({th['n_inf_pattern_differs']})", "yes" if ok else "NO"])
            pre = f"EtwoV{sh}{''.join(w.title() for w in k.replace('_', ' ').split())}".replace("0", "Zero").replace("1", "One").replace("2", "Two")
            pn(pre + "Done", m["D1"]["value"], f"{name}: {LABEL[k]}: D1 % on heldout_disjoint, recomputed by the verifier (plain Evaluator)")
            pn(pre + "Rel", m["rel"]["value"], f"{name}: {LABEL[k]}: Rel exact %, recomputed by the verifier")
            pn(pre + "TGen", m["tgen"]["value"], f"{name}: {LABEL[k]}: T-Gen exact %, recomputed by the verifier")
            pn(pre + "IGen", m["igen"]["value"], f"{name}: {LABEL[k]}: I-Gen exact %, recomputed by the verifier")
            pn(pre + "FlagsDiffer", ag["edit_exact_differ"] + ag["probe_exact_differ"], f"{name}: {LABEL[k]}: exact flags (300 edit rows + heldout_disjoint rows) that differ between the verifier's rows and the experiment's")
        out += [T(["cell", "D1 % disjoint exp | mine [95 %]", "D1 strict", "Rel/T-Gen/I-Gen", "own out", "fire % disjoint", "flags differ (edits; held-out)", "fired flags differ", "max |threshold diff| (inf pattern)", "agree"], rows), ""]
        cb = cl.get("comp_b_all_vs_p12_record")
        if cb:
            out += [f"    rebuilt COMP-B (ALL) against the ORIGINAL P12 record: held-out flags differing {cb['n_heldout_differ']} of {cb['n_heldout']} (heldout_disjoint: {cb['n_heldout_disjoint_differ']}), "
                    f"edit flags {cb['n_edit_differ']} of {cb['n_edit']}; my deployed flags against the record's deployed flags: {cl['deployed_vs_p12_record']['n_heldout_differ']} of {cl['deployed_vs_p12_record']['n_heldout']}", ""]
            pn(f"EtwoV{sh}CompRecordFlagsDiffer", cb["n_heldout_differ"] + cb["n_edit_differ"], f"{name}: COMP-B rebuilt by the verifier on all complaints: held-out + edit flags that differ from the P12 record")
        imp = cl["linear_import"]
        out += [f"(2) imported linear cells (part a) against the original P12 records, recomputed from the records with the library's split: {imp['n_cells']} cells, {imp['n_numbers']} numbers, "
                f"{imp['n_mismatch']} mismatches (largest difference of the others {imp['max_abs_diff']:.2e})" + (f"; examples {imp['examples']}" if imp["examples"] else ""),
                T(["linear cell (mean over the record's seeds)", "seeds", "D1 % exp", "D1 % mine", "Rel exp", "Rel mine", "agree"],
                  [[k, s["n_seeds"], f1(s["d1_exp"]), f1(s["d1_mine"]), f1(s["rel_exp"]), f1(s["rel_mine"]),
                    "yes" if s["d1_exp"] is not None and abs(s["d1_exp"] - s["d1_mine"]) < 1e-9 and abs(s["rel_exp"] - s["rel_mine"]) < 1e-9 else "NO"] for k, s in cl["linear_summary"].items()]), ""]
        pn(f"EtwoV{sh}LinearImportMismatch", imp["n_mismatch"], f"{name}: numbers of the imported linear cells (of {imp['n_numbers']}) that differ from the verifier's recomputation from the P12 records")
        h = cl["h1"]
        out += ["(3) H1 recomputed from the per-row files (ReScope-W rows of the experiment, seeds 0-2; linear side from the P12 records); rule of R6 applied literally",
                T(["quantity", "exp", "mine", "agree"],
                  [["D1 % ReScope-W (log-only start, n = 10, 3 seeds)", f1(h["d1_where_exp"]), f1(h["d1_where_mine"]), "yes" if abs(h["d1_where_exp"] - h["d1_where_mine"]) < 1e-9 else "NO"],
                   ["Rel exact % ReScope-W", f1(h["rel_where_exp"]), f1(h["rel_where_mine"]), "yes" if abs(h["rel_where_exp"] - h["rel_where_mine"]) < 1e-9 else "NO"],
                   ["Rel exact % deployed", f1(h["rel_deployed_exp"]), f1(h["rel_deployed_mine"]), "yes" if abs(h["rel_deployed_exp"] - h["rel_deployed_mine"]) < 1e-9 else "NO"]]
                  + [[f"{lab}: linear repair", c["linear_exp"], c["linear_mine"], "yes" if c["linear_exp"] == c["linear_mine"] else "NO"] for lab, c in h["comparisons"].items()]
                  + [[f"{lab}: D1 difference [95 %] (p)", f"{f1(c['diff_exp']['value'])} [{f1(c['diff_exp']['ci_lo'])}, {f1(c['diff_exp']['ci_hi'])}] ({f1(c['diff_exp']['p_two_sided'], 4)})",
                      f"{f1(c['diff_mine']['value'])} [{f1(c['diff_mine']['ci_lo'])}, {f1(c['diff_mine']['ci_hi'])}] ({f1(c['diff_mine']['p_two_sided'], 4)})",
                      "yes" if abs(c["diff_exp"]["value"] - c["diff_mine"]["value"]) < 1e-9 and abs(c["diff_exp"]["ci_lo"] - c["diff_mine"]["ci_lo"]) < 1.0 and abs(c["diff_exp"]["ci_hi"] - c["diff_mine"]["ci_hi"]) < 1.0 else "NO"]
                     for lab, c in h["comparisons"].items()]
                  + [["literal rule (D1 <= 10 and |Rel - Rel deployed| <= 2): helps on this log", h["literal_rule_helps_on_this_log_exp"], h["literal_rule_helps_on_this_log_mine"],
                      "yes" if h["literal_rule_helps_on_this_log_exp"] == h["literal_rule_helps_on_this_log_mine"] else "NO"],
                     ["one-sided reading (Rel >= Rel deployed - 2) met", h["one_sided_met_exp"], h["one_sided_met_mine"], "yes" if h["one_sided_met_exp"] == h["one_sided_met_mine"] else "NO"]]),
                f"    best linear repair at n = 10 by lowest D1: the experiment's six methods -> {h['best_linear']['registered_set_of_the_experiment']}; the four of PROTOCOL section 2 -> {h['best_linear']['protocol_section_2_set']}; "
                f"every P12 method with 3 seeds -> {h['best_linear']['every_p12_method_with_3_seeds']}",
                "    candidates (D1 %, Rel %): " + "; ".join(f"{m} {c['d1']:.1f}, {c['rel']:.1f}" for m, c in sorted(h["candidates"].items(), key=lambda kv: kv[1]["d1"])),
                f"    ReScope-W D1 per seed {[round(x, 2) for x in h['where_d1_per_seed']]}" + (f"; the seed-0 cell recomputed by me on the GPU: D1 {h['seed0_d1_mine_gpu']:.2f}, Rel {h['seed0_rel_mine_gpu']:.1f} (deployed Rel {h['deployed_rel_mine_gpu']:.1f})" if "seed0_d1_mine_gpu" in h else ""), ""]
        pn(f"EtwoV{sh}HoneWhereDone", h["d1_where_mine"], f"{name}: H1, ReScope-W D1 % (3 seeds), recomputed by the verifier from the per-row files")
        pn(f"EtwoV{sh}HoneWhereRel", h["rel_where_mine"], f"{name}: H1, ReScope-W Rel exact % (3 seeds), recomputed by the verifier")
        for lab, w in (("best_lowest_d1", "Best"), ("best_rel_within_2", "BestRel")):
            if lab in h["comparisons"]:
                pn(f"EtwoV{sh}Hone{w}Diff", h["comparisons"][lab]["diff_mine"]["value"], f"{name}: H1, D1 difference ReScope-W minus {h['comparisons'][lab]['linear_mine']} (points), verifier")
                pn(f"EtwoV{sh}Hone{w}DiffLo", h["comparisons"][lab]["diff_mine"]["ci_lo"], f"{name}: H1, lower end of the 95 % interval of that difference (verifier's paired crossed bootstrap)")
                pn(f"EtwoV{sh}Hone{w}DiffHi", h["comparisons"][lab]["diff_mine"]["ci_hi"], f"{name}: H1, upper end of that interval")
        pn(f"EtwoV{sh}HoneLiteralHelps", h["literal_rule_helps_on_this_log_mine"], f"{name}: H1, rule of R6 applied literally by the verifier (D1 <= 10 and |Rel - Rel deployed| <= 2)")
        s = cl["sanity"]
        out += [f"(4) sanity run e0 (deployed; default method, draw 0) against MY numbers: {s['n']} numbers, largest absolute difference {f1(s['max_abs_diff'], 4)} points",
                T(["cell", "quantity", "e0", "mine"], [[r["cell"], r["quantity"], f1(r["e0"]), f1(r["mine"])] for r in s["rows"]]), ""]
        pn(f"EtwoV{sh}SanityMaxDiff", s["max_abs_diff"], f"{name}: largest absolute difference (points) between the verifier's numbers and the sanity run e0 on {s['n']} shared numbers")
        cn = cl["counts"]
        out += [f"(5) counts: mine {cn['mine']}; complaints 'ALL' {cn['n_complaints_all_p12']}; in the experiment's records: heldout_disjoint n {cn['exp_probe_n_values']}, heldout_strict n {cn['exp_strict_n_values']}, "
                f"edits per metric {cn['exp_edit_n_values']}; {cn['exp_n_records']} records", ""]
    if "example" in C:
        ex = C["example"]
        out += ["#" * 140, f"RUNNING EXAMPLE recomputed from scratch (Qwen development log): {ex['n_agree']} of {ex['n']} values agree; the two copies of the experiment's file are equal: {ex['two_copies_of_the_experiment_file_equal']}",
                T(["key", "experiment", "mine", "agree"], [[r["key"], str(r["experiment"])[:60], str(r["mine"])[:60], "yes" if r["agree"] else "NO"] for r in ex["rows"]]),
                f"    details (mine): rollback {ex['mine_details']['rollback']}", f"    rescope {ex['mine_details']['rescope']}; compensation {ex['mine_details']['compensation']}; edit chosen by: {ex['mine_details']['edit_chosen_by']}", ""]
        pn("EtwoVExampleAgree", ex["n_agree"], f"running example: values (of {ex['n']}) on which the verifier's recomputation from scratch agrees with the experiment")
    ren = C["rerender"]
    out += ["#" * 140, "INTERNAL CONSISTENCY: the experiment's render.py re-run on its own cells into verify/out/rerender", "    " + json.dumps(ren), ""]
    R.table("\n".join(out))
    R.paper(paper)
    fx.io.jdump(dict(sorted(doc.items())), f"{OUT}/paper_numbers_doc.json", indent=1)
    fx.io.jdump({"cells_agree": {f"{n}/{k}": v for (n, k), v in allok.items()}}, f"{OUT}/cells_agree.json", indent=1)


if __name__ == "__main__":
    main()
