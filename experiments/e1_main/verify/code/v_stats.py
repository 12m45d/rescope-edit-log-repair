"""Verification of E1, statistics (CPU).  Everything here is recomputed with this file's own code from
  (a) the experiment's per-row files and gate-level cells (e1_main/out/rows, out/gate, out/behave), and
  (b) the rows of this verification's own GPU evaluation through the library's plain Evaluator (verify/out/rows; v_gpu.py),
and is compared with what the experiment reports (results.json, endpoints.json, decision.json, best_cosine_gate.json, paper_numbers.json)
and with the sanity run e0.  The driver is imported for the result schema (fx.io) and for the registered ORDER of the gate families only.

    python v_stats.py            -> verify/out/results.json, checks.json, paper_numbers.json (+ _doc), table.txt (v_render.py renders)
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob
import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
E1, V = FINAL + "/e1_main", FINAL + "/e1_main/verify"
EO, VO = E1 + "/out", V + "/out"
sys.path.insert(0, FINAL + "/common")
import fx  # noqa: E402

DEV = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
HN = ("HN-I", "HN-Q", "HN-X")
DRAWS = (0, 1, 2)
NB = 2000
FIXED = ("deployed", "preedit", "exact", "cosine_default", "lookup", "zscore_default", "cosine_published", "where_logonly", "dual")
CHK = {}                                                                 # name -> {...}: every check, written to checks.json


def jl(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(x) for x in f]


def jload(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return json.load(f)


def erows(log, lab, d, part="tf"):
    p = f"{EO}/rows/{log}__{lab}__d{'x' if d is None else d}__{part}.jsonl.gz"
    return jl(p) if os.path.exists(p) else None


def pct(v):
    v = list(v)
    return 100.0 * sum(bool(x) for x in v) / len(v) if v else None


# ----------------------------------------------------------------------------- own bootstrap (not the driver's)
def boot_counts(n, rng):
    return rng.multinomial(n, np.full(n, 1.0 / n), size=NB).astype(np.float64)


def crossed(num, den, seed=7):
    """num, den: (draws, units).  Percentile interval of sum(num) / sum(den) under independent resampling of draws and units."""
    num, den = np.atleast_2d(np.asarray(num, float)), np.atleast_2d(np.asarray(den, float))
    rng = np.random.default_rng(seed)
    cd, cu = boot_counts(num.shape[0], rng), boot_counts(num.shape[1], rng)
    st = 100.0 * ((cd @ num) * cu).sum(1) / ((cd @ den) * cu).sum(1)
    return {"value": 100.0 * num.sum() / den.sum(), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5))}


def crossed_diff(na, da, nb, db, seed=7):
    na, da, nb, db = (np.atleast_2d(np.asarray(x, float)) for x in (na, da, nb, db))
    rng = np.random.default_rng(seed)
    cd, cu = boot_counts(na.shape[0], rng), boot_counts(na.shape[1], rng)
    f = lambda M: ((cd @ M) * cu).sum(1)                                 # noqa: E731
    st = 100.0 * (f(na) / f(da) - f(nb) / f(db))
    p = 2.0 * min(float((st <= 0).mean()), float((st >= 0).mean()))
    return {"value": 100.0 * (na.sum() / da.sum() - nb.sum() / db.sum()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5)),
            "p": min(1.0, max(p, 1.0 / NB)), "mean_a": 100.0 * na.sum() / da.sum(), "mean_b": 100.0 * nb.sum() / db.sum()}


def holm(ps):
    keys = sorted(ps, key=lambda k: ps[k])
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def grid(rows_by_draw, val, unit, where):
    """(draws, units) sums and counts; every draw must hold the same units."""
    per = []
    for d in sorted(rows_by_draw):
        acc = {}
        for r in rows_by_draw[d]:
            if where(r):
                a = acc.setdefault(r[unit], [0.0, 0])
                a[0] += float(bool(val(r)))
                a[1] += 1
        per.append(acc)
    units = sorted(per[0], key=str)
    assert all(set(p) == set(units) for p in per), "draws hold different units"
    return np.array([[p[u][0] for u in units] for p in per]), np.array([[p[u][1] for u in units] for p in per])


REPH = lambda r: r["kind"] == "edit" and r["metric"] in ("tgen", "igen")   # noqa: E731
EX = lambda r: r["exact"]                                                  # noqa: E731


def rep3(lab, by):
    return {d: by[None] for d in DRAWS} if None in by else by


# ----------------------------------------------------------------------------- A. cells from the experiment's rows
def cells_from_rows(rows):
    e = [r for r in rows if r["kind"] == "edit"]
    p = [r for r in rows if r["kind"] == "probe" and r["disjoint"]]
    h = [r for r in rows if r["kind"] == "hn"]
    c = {f"edit.{m}.exact": pct(r["exact"] for r in e if r["metric"] == m) for m in ("rel", "tgen", "igen")}
    c["edit.rephrase.exact"] = pct(r["exact"] for r in e if r["metric"] in ("tgen", "igen"))
    c["probe.D1"] = pct(r["D1"] for r in p)
    c["probe.fired"] = pct(r["fired"] for r in p) if p and p[0].get("fired") is not None else None
    for k in HN:
        c[f"hn.{k}.D1"] = pct(r["D1"] for r in h if r["hn_type"] == k)
        c[f"hn.{k}.fired"] = pct(r["fired"] for r in h if r["hn_type"] == k) if h and h[0].get("fired") is not None else None
    c["n.edit"], c["n.probe"], c["n.hn"] = len(e), len(p), len(h)
    c["n.probe.invalid_y0"] = sum(not r.get("y0_valid", True) for r in p)
    c["n.hn.by_kind"] = {k: sum(r["hn_type"] == k for r in h) for k in HN}
    c["n.hn.y0_deg"] = {k: sum(bool(r.get("y0_deg")) for r in h if r["hn_type"] == k) for k in HN}
    c["n.clusters"] = len({r["query_id"] for r in p})
    return c


POOL = {"edit": "edits", "probe": "heldout_disjoint", "hn": "hard_negatives"}


def part_a(R, RES, PN, sel):
    """Main-table cells recomputed from the row files; compared with results.json and paper_numbers.json."""
    idx = {(r["log"], r["method"], r["metric"], r["pool"], r["draw"]): r for r in RES}
    out, TF = {}, {}
    nm = {"deployed": "Deployed", "preedit": "Preedit", "rescope": "Rescope", "rescope_shared": "RescopeShared", "rescope_null": "RescopeNull", "rescope_x": "RescopeX",
          "rescope_lin": "RescopeLin", "where_logonly": "RescopeW", "cosine": "Cosine", "cosine_shared": "CosineShared", "lookup": "Lookup", "ball": "Ball", "router": "Router",
          "cosine_default": "CosDefault", "cosine_published": "CosPublished", "zscore_default": "Zscore", "exact": "Exact", "dual": "Dual"}
    mn = {"edit.rel.exact": "Rel", "edit.tgen.exact": "TGen", "edit.igen.exact": "IGen", "edit.rephrase.exact": "Reph", "probe.D1": "DOne", "probe.fired": "Fire",
          "hn.HN-I.D1": "HNIDOne", "hn.HN-Q.D1": "HNQDOne", "hn.HN-X.D1": "HNXDOne"}
    worst, worst_pn, n_cmp, n_pn = 0.0, 0.0, 0, 0
    for log in DEV:
        s = sel[log]
        names = dict(nm)
        names[s["cos_H3"]], names[s["retr_H3"]] = "BestCos", "Retr"
        names.setdefault(s["cos_H2"], "BestCosHtwo")
        names.setdefault(s["cos_H3_x"], "BestCosX")
        labs = sorted({x.split("__")[1] for x in os.listdir(f"{EO}/rows") if x.startswith(log + "__") and x.endswith("__tf.jsonl.gz")})
        out[log], TF[log] = {}, {}
        for lab in labs:
            fixed = lab in FIXED
            by = {d: erows(log, lab, d) for d in ((None,) if fixed else DRAWS)}
            by = {d: v for d, v in by.items() if v is not None}
            TF[log][lab] = by
            per = {d: cells_from_rows(v) for d, v in by.items()}
            mean = {k: (float(np.mean([per[d][k] for d in per])) if all(isinstance(per[d][k], float) for d in per) else None) for k in mn}
            out[log][lab] = {"per_draw": per, "mean": mean, "n_draws": len(per)}
            for k, v in mean.items():
                if v is None:
                    continue
                pool = POOL[k.split(".")[0]]
                e = idx.get((log, lab, k if fixed else k + "@x", pool, None))
                ev = None if e is None else e["value"]
                R.add(log, lab, k, v, pool=pool, alpha=None if fixed else 0.05, n=None if fixed else 200, experiment_value=ev, source="recomputed from the experiment's row files",
                      n_draws=len(per))
                if ev is not None:
                    n_cmp += 1
                    worst = max(worst, abs(v - ev))
                key = f"Eone{SHORT[log]}{names.get(lab, '?')}{mn[k]}"
                if key in PN:
                    n_pn += 1
                    worst_pn = max(worst_pn, abs(v - PN[key]))
    CHK["A_cells_vs_results_json"] = {"n_cells_compared": n_cmp, "largest_abs_diff": worst, "ok": worst < 1e-9}
    CHK["A_cells_vs_paper_numbers"] = {"n_numbers_compared": n_pn, "largest_abs_diff": worst_pn, "ok": worst_pn < 1e-9}
    return out, TF


# ----------------------------------------------------------------------------- B. H2
def part_b(R, TF, EP):
    out, ps = {}, {}
    for log in DEV:
        A = TF[log]["rescope"]
        na, da = grid(A, EX, "edit", REPH)
        out[log] = {}
        for role, e in EP[log]["H2"].items():
            lab = e["method"]
            B = rep3(lab, TF[log][lab])
            nb, db = grid(B, EX, "edit", REPH)
            d = crossed_diff(na, da, nb, db)
            d["by_draw"] = [100.0 * (na[i].sum() / da[i].sum() - nb[i].sum() / db[i].sum()) for i in range(3)]
            hn = lambda r: r["kind"] == "hn"                               # noqa: E731
            h = crossed_diff(*grid(A, lambda r: r["D1"], "edit", hn), *grid(B, lambda r: r["D1"], "edit", hn))
            rl = crossed_diff(*grid(A, EX, "edit", lambda r: r["kind"] == "edit" and r["metric"] == "rel"), *grid(B, EX, "edit", lambda r: r["kind"] == "edit" and r["metric"] == "rel"))
            d.update(method=lab, n_units=int(na.shape[1]), near_scope_D1_diff=h["value"], rel_exact_diff=rl["value"],
                     exp={"diff": e["diff"], "ci_lo": e["ci_lo"], "ci_hi": e["ci_hi"], "p": e["p"], "p_holm": e.get("p_holm"), "by_draw": e["by_draw"],
                          "near_scope_D1_diff": e["near_scope_D1_diff"]["value"], "rel_exact_diff": e["rel_exact_diff"]["value"]})
            eg = e.get("gate_coverage_form_20_draws")
            if eg is not None:
                def cov(lb):
                    a_ = "0.05" if os.path.exists(f"{EO}/gate/{log}/{lb}__a0.05.json") else "none"
                    cs = gate_cells(log, lb, a_)
                    cs = cs * 20 if len(cs) == 1 else cs
                    return np.array([0.5 * (unflag(c["per_edit"]["tgen"]) + unflag(c["per_edit"]["igen"])) for c in cs])
                Ma, Mb = cov("rescope"), cov(lab)
                g = crossed_diff(Ma, np.ones_like(Ma), Mb, np.ones_like(Mb))
                d["gate"] = {"value": g["value"], "ci_lo": g["ci_lo"], "ci_hi": g["ci_hi"], "n_draws": int(Ma.shape[0]), "exp": {k: eg[k] for k in ("value", "ci_lo", "ci_hi")}}
            out[log][role] = d
            ps[f"{log}/{role}"] = d["p"]
    adj = holm(ps)
    wv, wc, wn = 0.0, 0.0, 0.0
    wg = max(abs(d["gate"]["value"] - d["gate"]["exp"]["value"]) for log in DEV for d in out[log].values() if "gate" in d)
    wgc = max(max(abs(d["gate"]["ci_lo"] - d["gate"]["exp"]["ci_lo"]), abs(d["gate"]["ci_hi"] - d["gate"]["exp"]["ci_hi"])) for log in DEV for d in out[log].values() if "gate" in d)
    for log in DEV:
        for role, d in out[log].items():
            d["p_holm"] = adj[f"{log}/{role}"]
            wv = max(wv, abs(d["value"] - d["exp"]["diff"]), max(abs(a - b) for a, b in zip(d["by_draw"], d["exp"]["by_draw"])))
            wn = max(wn, abs(d["near_scope_D1_diff"] - d["exp"]["near_scope_D1_diff"]), abs(d["rel_exact_diff"] - d["exp"]["rel_exact_diff"]))
            wc = max(wc, abs(d["ci_lo"] - d["exp"]["ci_lo"]), abs(d["ci_hi"] - d["exp"]["ci_hi"]))
            R.add(log, "rescope", "endpoint.H2.rephrase_exact_diff", d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_units=d["n_units"], variant=f"minus {d['method']} ({role})", alpha=0.05, n=200,
                  pool="edits", p_two_sided=d["p"], p_holm=d["p_holm"], n_draws=3, experiment_value=d["exp"]["diff"], experiment_ci=[d["exp"]["ci_lo"], d["exp"]["ci_hi"]],
                  experiment_p_holm=d["exp"]["p_holm"], source="own crossed bootstrap on the experiment's per-edit rows")
    sign_same = all(((d["ci_lo"] > 0) == (d["exp"]["ci_lo"] > 0)) and ((d["ci_hi"] < 0) == (d["exp"]["ci_hi"] < 0)) for log in DEV for d in out[log].values())
    rej_same = all((d["p_holm"] <= 0.05) == (d["exp"]["p_holm"] <= 0.05) for log in DEV for d in out[log].values())
    CHK["B_H2"] = {"n_tests": len(ps), "gate_form_largest_abs_diff_point_estimates": wg, "gate_form_largest_abs_diff_interval_ends": wgc, "largest_abs_diff_point_estimates": wv, "largest_abs_diff_secondary_points": wn, "largest_abs_diff_interval_ends": wc,
                   "intervals_exclude_0_in_the_same_cases": sign_same, "holm_rejections_equal": rej_same, "ok": wv < 1e-9 and wn < 1e-9 and wc < 1.5 and sign_same and rej_same and wg < 1e-9 and wgc < 1.5}
    return out


# ----------------------------------------------------------------------------- C. H3 and the decision
def gate_cells(log, lab, alpha="0.05"):
    b = jload(f"{EO}/gate/{log}/{lab}__a{alpha}.json")
    return [b["cells"][k] for k in sorted(b["cells"], key=int)]


def unflag(s):
    return np.array([c == "1" for c in s], dtype=float)


def part_c(R, EP, DEC):
    out, ps = {}, {}
    for log in DEV:
        out[log] = {}
        for pair, e in EP[log]["H3"].items():
            a, b = e["a"], e["b"]
            rows, info = {}, {}
            for lab in (a, b):
                rows[lab], info[lab] = {}, {}
                for d in DRAWS:
                    rec = jload(f"{EO}/behave/{log}/{lab}__d{d}__h3.json")
                    info[lab][d] = rec
                    rows[lab][d] = [r for r in erows(log, lab, d, "h3") if r["kind"] == "edit"] if rec["reachable"] else \
                        [{"kind": "edit", "metric": m, "edit": x, "exact": False} for m in ("tgen", "igen") for x in rec["odd_edits"]]
                    odd = sorted({r["edit"] for r in rows[lab][d]})
                    assert odd == sorted(rec["odd_edits"]) and all(x % 2 == 1 for x in odd), "H3 edit rows are not the odd edits"
            d_ = crossed_diff(*grid(rows[a], EX, "edit", REPH), *grid(rows[b], EX, "edit", REPH))
            # gate-level coverage form over the 20 draws (per-edit coverage flags of the odd edits at the raised scale)
            def mat(lab):
                M = []
                for c in gate_cells(log, lab):
                    h = c["h3"]
                    M.append(0.5 * (unflag(h["per_edit"]["tgen"]) + unflag(h["per_edit"]["igen"])) if h["reachable"] else None)
                n = next(len(x) for x in M if x is not None)
                return np.array([x if x is not None else np.zeros(n) for x in M]), sum(x is None for x in M)
            (Ma, ua), (Mb, ub) = mat(a), mat(b)
            g = crossed_diff(Ma, np.ones_like(Ma), Mb, np.ones_like(Mb))
            # the cell record's raise equals the gate-level cell's raise of the same draw
            same_raise = all(abs((info[lab][d].get("delta") or 0.0) - (gate_cells(log, lab)[d]["h3"].get("delta") or 0.0)) < 1e-12 for lab in (a, b) for d in DRAWS)
            c = {"a": a, "b": b, "diff": d_["value"], "ci_lo": d_["ci_lo"], "ci_hi": d_["ci_hi"], "p": d_["p"], "mean_a": d_["mean_a"], "mean_b": d_["mean_b"],
                 "gate_diff": g["value"], "gate_ci_lo": g["ci_lo"], "gate_ci_hi": g["ci_hi"], "gate_unreachable_draws": [int(ua), int(ub)],
                 "n_edits_at_cap_a": [info[a][d].get("n_edits_at_cap") for d in DRAWS], "n_edits_at_cap_b": [info[b][d].get("n_edits_at_cap") for d in DRAWS],
                 "behavioural_raise_equals_gate_level_raise": same_raise,
                 "exp": {"diff": e["diff"], "ci_lo": e["ci_lo"], "ci_hi": e["ci_hi"], "p_holm": e.get("p_holm"), "gate": e["gate_coverage_form_20_draws"]}}
            c["conditions_met"] = bool(c["diff"] >= 10.0 and c["ci_lo"] > 0 and c["gate_ci_lo"] > 0)
            out[log][pair] = c
            ps[f"{log}/{pair}"] = d_["p"]
    adj = holm(ps)
    dec, wv, wc = {}, 0.0, 0.0
    for log in DEV:
        for pair, c in out[log].items():
            c["p_holm"] = adj[f"{log}/{pair}"]
            wv = max(wv, abs(c["diff"] - c["exp"]["diff"]), abs(c["gate_diff"] - c["exp"]["gate"]["value"]))
            wc = max(wc, abs(c["ci_lo"] - c["exp"]["ci_lo"]), abs(c["ci_hi"] - c["exp"]["ci_hi"]), abs(c["gate_ci_lo"] - c["exp"]["gate"]["ci_lo"]), abs(c["gate_ci_hi"] - c["exp"]["gate"]["ci_hi"]))
            R.add(log, c["a"], "endpoint.H3.rephrase_exact_diff", c["diff"], ci_lo=c["ci_lo"], ci_hi=c["ci_hi"], n_units=50, variant=f"minus {c['b']} ({pair})", alpha=0.05, n=200, pool="odd_edits",
                  p_two_sided=c["p"], p_holm=c["p_holm"], n_draws=3, experiment_value=c["exp"]["diff"], experiment_ci=[c["exp"]["ci_lo"], c["exp"]["ci_hi"]],
                  n_edits_at_cap_a=c["n_edits_at_cap_a"], n_edits_at_cap_b=c["n_edits_at_cap_b"], source="own crossed bootstrap on the experiment's H3 rows")
            R.add(log, c["a"], "endpoint.H3.gate_coverage_diff", c["gate_diff"], ci_lo=c["gate_ci_lo"], ci_hi=c["gate_ci_hi"], n_units=50, variant=f"minus {c['b']} ({pair})", alpha=0.05, n=200,
                  pool="odd_edits", n_draws=20, experiment_value=c["exp"]["gate"]["value"], experiment_ci=[c["exp"]["gate"]["ci_lo"], c["exp"]["gate"]["ci_hi"]],
                  source="own crossed bootstrap on the experiment's gate-level cells")
    for pair in ("pair1", "pair2_x", "retr"):
        dec[pair] = "dominates" if all(out[log][pair]["conditions_met"] for log in DEV) else "trade-off"
    same = all(dec[p] == DEC["pairs"][p]["verdict"] for p in dec)
    CHK["C_H3_decision"] = {"mine": dec, "experiment": {p: DEC["pairs"][p]["verdict"] for p in dec}, "equal": same, "largest_abs_diff_point_estimates": wv,
                            "largest_abs_diff_interval_ends": wc, "raise_consistent": all(c["behavioural_raise_equals_gate_level_raise"] for log in DEV for c in out[log].values()),
                            "largest_H3_difference_points": max(c["diff"] for log in DEV for p, c in out[log].items() if p != "retr"), "ok": same and wv < 1e-9 and wc < 1.5}
    return out, dec


# ----------------------------------------------------------------------------- D. the selection
def part_d(R, SELF):
    out, ok = {}, True
    fam = {"cosine": list(fx.review1.COSINE_FAMILY), "retr": list(fx.review1.RETRIEVAL_FAMILY)}
    for log in DEV:
        bb = fx.logs.info(log)["backbone"]
        out[log] = {}
        for f, names in fam.items():
            tab = {}
            for nm in names:
                cs = gate_cells(log, nm)
                assert len(cs) == 20
                # from the per-edit flags (not from the stored aggregates)
                h3 = [100.0 * float(np.mean(0.5 * (unflag(c["h3"]["per_edit"]["tgen"]) + unflag(c["h3"]["per_edit"]["igen"])))) if c["h3"]["reachable"] else 0.0 for c in cs]
                h2 = [100.0 * float(np.mean(0.5 * (unflag(c["per_edit"]["tgen"]) + unflag(c["per_edit"]["igen"])))) for c in cs]
                tab[nm] = {"H3": float(np.mean(h3)), "H2": float(np.mean(h2)), "n_unreachable": sum(not c["h3"]["reachable"] for c in cs), "n_odd_edits": len(cs[0]["h3"]["per_edit"]["tgen"]) if cs[0]["h3"]["reachable"] else None,
                           "fit_mean_fired_at_raise_max": max(((c["h3"].get("fit") or {}).get("at_raise") or {}).get("mean", 0.0) for c in cs),
                           "n_edits_at_cap_mean": float(np.mean([c["h3"].get("n_edits_at_cap") or 0 for c in cs]))}
            pick = {}
            for key in ("H3", "H2"):
                top = max(t[key] for t in tab.values())
                tied = [nm for nm in names if abs(tab[nm][key] - top) < 1e-9]
                pick[key] = {"selected": tied[0], "value": top, "tied": tied, "runner_up": sorted(((t[key], nm) for nm, t in tab.items() if nm not in tied), reverse=True)[0]}
            exp = SELF["selected"][bb]["cosine" if f == "cosine" else "retr"]
            eq = pick["H3"]["selected"] == exp["H3"] and pick["H2"]["selected"] == exp["H2"]
            ok = ok and eq
            out[log][f] = {"table": tab, "pick": pick, "experiment": {"H3": exp["H3"], "H2": exp["H2"]}, "equal": eq}
            for key in ("H3", "H2"):
                R.add(log, pick[key]["selected"], f"selection.{f}.{key}.coverage", pick[key]["value"], alpha=0.05, n=200, pool="gate_level", n_draws=20, experiment_selected=exp[key],
                      tied=pick[key]["tied"], runner_up=pick[key]["runner_up"], source="recomputed from the per-edit coverage flags of the experiment's gate-level cells")
    t_sel, t_first = os.path.getmtime(f"{EO}/best_cosine_gate.json"), None
    for p in glob.glob(f"{EO}/behave/*/*.json"):
        b = os.path.basename(p)
        if any(b.startswith(x) for x in ("white_", "whitecos_", "cos_", "retr_")):
            t_first = os.path.getmtime(p) if t_first is None else min(t_first, os.path.getmtime(p))
    CHK["D_selection"] = {"equal": ok, "selection_file_older_than_first_behavioural_cell_of_a_selected_gate": bool(t_first is None or t_sel < t_first),
                          "seconds_between": None if t_first is None else t_first - t_sel, "ok": ok and (t_first is None or t_sel < t_first)}
    return out


# ----------------------------------------------------------------------------- E. own GPU rows against the experiment's rows (draw 0)
def mine(log, blk):
    p = f"{VO}/rows/{log}__{blk}.json.gz"
    return jload(p) if os.path.exists(p) else None


FLAGS = {"edit": ("exact", "fired", "top_last", "own_released_last"), "probe": ("D1", "exact", "fired", "top_last"), "hn": ("D1", "exact", "fired", "top_last")}


def boot_single(da, db, seed=7):
    """Paired bootstrap over edits of one draw: per-edit means of the two rephrase flags."""
    rng = np.random.default_rng(seed)
    d = np.asarray(da, float) - np.asarray(db, float)
    st = 100.0 * (boot_counts(len(d), rng) @ d) / len(d)
    return {"value": 100.0 * float(d.mean()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5))}


def per_edit(rows):
    acc = {}
    for r in rows:
        if REPH(r):
            acc.setdefault(r["edit"], []).append(float(bool(r["exact"])))
    return [float(np.mean(acc[k])) for k in sorted(acc)]


def part_e(R):
    out = {}
    for log in DEV:
        g = mine(log, "gates")
        if g is None:
            out[log] = {"missing": "no GPU rows of the verification for this log"}
            continue
        o = {"gates": {k: {"eta": v.get("eta"), "n_edits_capped": v.get("n_edits_capped")} for k, v in g["cal"].items()}, "cells": {}, "flags": {}, "n": {}}
        my = {}
        for blk in ("tf_edit", "tf_probe", "tf_hn"):
            b = mine(log, blk)
            if b is None:
                continue
            for m, rows in b.items():
                my.setdefault(m, []).extend(rows)
        for m, rows in my.items():
            ex = erows(log, m, None if m == "deployed" else 0)
            exs = {r["sid"]: r for r in ex}
            kinds = sorted({r["kind"] for r in rows})
            fl = {}
            for r in rows:
                e = exs.get(r["sid"])
                for k in FLAGS[r["kind"]]:
                    if m == "deployed" and k in ("fired", "top_last", "own_released_last"):
                        continue
                    a = fl.setdefault(f"{r['kind']}.{k}", {"n": 0, "n_differ": 0, "n_missing": 0, "sids": []})
                    a["n"] += 1
                    if e is None:
                        a["n_missing"] += 1
                    elif e.get(k) != r.get(k):
                        a["n_differ"] += 1
                        a["sids"].append(r["sid"]) if len(a["sids"]) < 8 else None
            o["flags"][m] = fl
            o["n"][m] = {"mine": len(rows), "experiment_rows": len(ex), "kinds": kinds, "experiment_rows_not_in_mine": len(set(exs) - {r["sid"] for r in rows}) if len(kinds) == 3 else None}
            cm = cells_from_rows([dict(r, disjoint=True) if r["kind"] == "probe" else r for r in rows]) if len(kinds) == 3 else None
            ce = cells_from_rows(ex)
            if cm is not None:
                cov_m = {k: pct(r.get("own_released_last") for r in rows if r["kind"] == "edit" and r["metric"] == k) for k in ("tgen", "igen")} if m != "deployed" else {}
                cov_e = {k: pct(r.get("own_released_last") for r in ex if r["kind"] == "edit" and r["metric"] == k) for k in ("tgen", "igen")} if m != "deployed" else {}
                o["cells"][m] = {}
                for k in ("probe.fired", "probe.D1", "edit.rel.exact", "edit.tgen.exact", "edit.igen.exact", "hn.HN-I.D1", "hn.HN-Q.D1", "hn.HN-X.D1"):
                    o["cells"][m][k] = {"mine": cm[k], "experiment": ce[k]}
                for k in cov_m:
                    o["cells"][m][f"cov.{k}"] = {"mine": cov_m[k], "experiment": cov_e[k]}
                for k, v in o["cells"][m].items():
                    if v["mine"] is not None:
                        R.add(log, m, k, v["mine"], pool=POOL.get(k.split(".")[0], "edits"), alpha=None if m == "deployed" else 0.05, n=None if m == "deployed" else 200,
                              draw=None if m == "deployed" else 0, experiment_value=v["experiment"], source="own forwards through the library's plain Evaluator")
        # primary endpoint at draw 0: rescope minus cosine_shared, from the experiment's rows and from mine
        if "rescope" in my and "cosine_shared" in my and any(r["kind"] == "edit" for r in my["rescope"]):
            pe = {"mine": boot_single(per_edit(my["rescope"]), per_edit(my["cosine_shared"])),
                  "experiment_rows": boot_single(per_edit(erows(log, "rescope", 0)), per_edit(erows(log, "cosine_shared", 0)))}
            o["endpoint_draw0_rescope_minus_cosine_shared"] = pe
            for k, v in pe.items():
                R.add(log, "rescope", "endpoint.H2.rephrase_exact_diff.draw0", v["value"], ci_lo=v["ci_lo"], ci_hi=v["ci_hi"], n_units=100, variant=f"minus cosine_shared; rows: {k}", alpha=0.05, n=200,
                      draw=0, pool="edits", source="paired bootstrap over edits, one draw")
        # thresholds of my GPU-built gates against the driver's CPU-built gates of the experiment (same content digest as the experiment's cell)
        try:
            L = fx.logs.open(log)
            thr = {}
            for m in ("rescope", "rescope_shared", "rescope_null", "cosine_shared"):
                M = fx.methods.build(m, L, alpha=0.05, n=200, draw=0)
                t = (M.obj.eta * M.obj.det_tau).numpy()
                a = np.array(g["eff_threshold"][m])
                cell = jload(f"{EO}/behave/{log}/{m}__d0__tf.json")
                thr[m] = {"max_rel_diff": float(np.max(np.abs(a - t) / np.maximum(np.abs(t), 1e-300))), "driver_digest_equals_experiment_cell": M.digest == cell["digest"],
                          "calibration_sids_equal": [s.split(":", 1)[1] if s[:2] in ("Q:", "L:") else s for s in M.rec["draw"]["sids"]] == g["calibration_sids"]}
                if m == "rescope_null":
                    thr[m]["null_sids_equal"] = [s.split(":", 1)[1] if s[:2] in ("Q:", "L:") else s for s in M.rec["null_repair"]["sids"]] == g["null_sids"]
            o["thresholds_vs_driver"] = thr
        except Exception as e:                                           # noqa: BLE001
            o["thresholds_vs_driver"] = {"error": repr(e)[:300]}
        # generation, D2, dual
        gen, d2, w0 = mine(log, "gen"), mine(log, "d2"), mine(log, "w0")
        if gen is not None:
            G = {}
            for m, lab, d in (("deployed", "deployed", None), ("preedit", "preedit", None), ("rescope", "rescope", 0), ("dual", "dual", None)):
                ex = {r["sid"]: r for r in erows(log, lab, d, "gen")}
                acc = {}
                for r in gen[m]:
                    e = ex[r["sid"]]
                    a = acc.setdefault(r["kind"], {"n": 0, "n_text_differs": 0, "n_flag_differs": 0, "n_fired_differs": 0, "n_not_decoded_by_experiment": 0, "n_not_decoded_and_not_pre_edit": 0, "sids": []})
                    a["n"] += 1
                    if "long" not in e:                                  # the driver did not decode: nothing is released, hence the pre-edit answer
                        a["n_not_decoded_by_experiment"] += 1
                        bad = bool(r.get("fired")) or not r.get("long_eq_pre", True)
                        a["n_not_decoded_and_not_pre_edit"] += int(bad)
                        if bad and len(a["sids"]) < 8:
                            a["sids"].append(r["sid"])
                        continue
                    td = e["long"] != r["long"]
                    key = {"edit": ("eq", "contains"), "probe": ("agree",), "hn": ("agree",)}[r["kind"]]
                    fd = any(e.get(k) != r.get(k) for k in key)
                    a["n_text_differs"] += int(td)
                    a["n_flag_differs"] += int(fd)
                    if m != "deployed" and m != "preedit":
                        a["n_fired_differs"] += int(bool(e.get("fired")) != bool(r.get("fired")))
                    if (td or fd) and len(a["sids"]) < 8:
                        a["sids"].append(r["sid"])
                rates = {}
                for kind, key in (("edit", "eq"), ("probe", "agree"), ("hn", "agree")):
                    sub = [r for r in gen[m] if r["kind"] == kind and r.get(key) is not None]
                    if sub:
                        es = [ex[r["sid"]] for r in sub]
                        ev = [(e[key] if "long" in e else True) for e in es]
                        rates[f"{kind}.{key}"] = {"n": len(sub), "mine": pct(r[key] for r in sub), "experiment_same_rows": pct(ev)}
                        R.add(log, m, f"gen.sample.{kind}.{key}", rates[f"{kind}.{key}"]["mine"], n_units=len(sub), pool="seeded sample", draw=0 if m == "rescope" else None,
                              experiment_value=rates[f"{kind}.{key}"]["experiment_same_rows"], source="own decodes through the library's plain Evaluator")
                G[m] = {"agreement": acc, "rates": rates}
            # DUAL: release decision and the labelled teacher-forced composition
            exd = {r["sid"]: r for r in erows(log, "dual", None, "gen")}
            ext = {r["sid"]: r for r in erows(log, "dual", None, "tf")}
            dd = {"n": len(gen["dual"]), "n_release_differs": sum(bool(exd[r["sid"]]["fired"]) != bool(r["fired"]) for r in gen["dual"]),
                  "released_mine": pct(r["fired"] for r in gen["dual"]), "released_experiment_same_rows": pct(exd[r["sid"]]["fired"] for r in gen["dual"]),
                  "by_kind": {k: {"n": sum(r["kind"] == k for r in gen["dual"]), "released_mine": pct(r["fired"] for r in gen["dual"] if r["kind"] == k),
                                  "released_experiment": pct(exd[r["sid"]]["fired"] for r in gen["dual"] if r["kind"] == k)} for k in ("probe", "edit")}}
            if w0 is not None:
                tfm = {"deployed": {r["sid"]: r for r in w0["deployed"]}, "preedit": {r["sid"]: r for r in w0["preedit"]}}
                nd, comp = 0, {"edit": [], "probe": []}
                for r in gen["dual"]:
                    row = tfm["deployed" if r["fired"] else "preedit"][r["sid"]]
                    e = ext[r["sid"]]
                    key = "exact" if r["kind"] == "edit" else "D1"
                    nd += int(bool(row[key]) != bool(e[key]))
                    comp[r["kind"]].append((bool(row[key]), bool(e[key])))
                dd["tf_composition"] = {"n_flag_differs": nd, "edit_exact_mine": pct(a for a, _ in comp["edit"]), "edit_exact_experiment": pct(b for _, b in comp["edit"]),
                                        "probe_D1_mine": pct(a for a, _ in comp["probe"]), "probe_D1_experiment": pct(b for _, b in comp["probe"])}
                R.add(log, "dual", "tf_composition.sample.edit.exact", dd["tf_composition"]["edit_exact_mine"], n_units=len(comp["edit"]), pool="seeded sample",
                      experiment_value=dd["tf_composition"]["edit_exact_experiment"], source="own composition: own decodes select the model, own teacher-forced forward")
                R.add(log, "dual", "tf_composition.sample.probe.D1", dd["tf_composition"]["probe_D1_mine"], n_units=len(comp["probe"]), pool="seeded sample",
                      experiment_value=dd["tf_composition"]["probe_D1_experiment"], source="own composition: own decodes select the model, own teacher-forced forward")
            R.add(log, "dual", "gen.sample.released", dd["released_mine"], n_units=dd["n"], pool="seeded sample", experiment_value=dd["released_experiment_same_rows"],
                  source="own decodes through the library's plain Evaluator")
            G["dual"]["dual"] = dd
            o["generation"] = G
        if d2 is not None:
            D = {}
            for m, lab, d in (("deployed", "deployed", None), ("rescope", "rescope", 0)):
                ex = {r["sid"]: r for r in erows(log, lab, d, "gen") if r["kind"] == "probe"}
                sub = d2[m]
                D[m] = {"n": len(sub), "n_in_experiment_d2_set": sum(bool(ex[r["sid"]].get("in_d2_set")) for r in sub), "n_D2_differs": sum(bool(ex[r["sid"]]["D2"]) != bool(r["D2"]) for r in sub),
                        "n_text_differs": sum(ex[r["sid"]]["long"] != r["long"] for r in sub), "D2_mine": pct(r["D2"] for r in sub), "D2_experiment_same_rows": pct(ex[r["sid"]]["D2"] for r in sub),
                        "n_pre_correct": sum(bool(r["pre_correct"]) for r in sub)}
                R.add(log, m, "d2.sample.D2", D[m]["D2_mine"], n_units=len(sub), pool="seeded sample of the pre-correct heldout_disjoint rows", draw=0 if m == "rescope" else None,
                      experiment_value=D[m]["D2_experiment_same_rows"], source="own decodes through the library's plain Evaluator")
            o["d2"] = D
        out[log] = o
    return out


# ----------------------------------------------------------------------------- F. the sanity run e0
def part_f(A):
    E0 = jload(FINAL + "/e0_sanity/out/results.json")
    MAP = {"M1": "rescope", "M2": "rescope_shared", "M3": "cosine", "M4": "cosine_shared", "M5": "cosine_default", "M6": "exact", "deployed": "deployed", "pre-edit": "preedit"}
    worst, n, bad = 0.0, 0, []
    for log in DEV:
        e0 = E0["logs"][log]
        for m0, lab in MAP.items():
            b = e0["behaviour"]["methods"].get(m0)
            if b is None:
                continue
            per = {None: b["single"]} if "single" in b else {int(k): v for k, v in b["draws"].items()}
            for d, v in per.items():
                mine_ = A[log][lab]["per_draw"].get(d)
                if mine_ is None:
                    continue
                for path, k in ((("edit", "rel", "exact"), "edit.rel.exact"), (("edit", "tgen", "exact"), "edit.tgen.exact"), (("edit", "igen", "exact"), "edit.igen.exact"),
                                (("probe", "heldout_disjoint", "D1"), "probe.D1"), (("hn", "HN-I", "D1"), "hn.HN-I.D1"), (("hn", "HN-Q", "D1"), "hn.HN-Q.D1"), (("hn", "HN-X", "D1"), "hn.HN-X.D1")):
                    x = v
                    for kk in path:
                        x = x[kk]
                    df = abs(float(x) - mine_[k])
                    n += 1
                    worst = max(worst, df)
                    if df > 1e-9:
                        bad.append([log, lab, d, k, float(x), mine_[k]])
            g = e0["gate"]["methods"].get(m0)
            if g is not None:
                a = "0.05" if os.path.exists(f"{EO}/gate/{log}/{lab}__a0.05.json") else "none"
                cs = gate_cells(log, lab, a)
                for path, f in ((("probe", "heldout_disjoint", "fired"), lambda c: c["st"]["probe"]["heldout_disjoint"]["fired"]), (("edit", "tgen", "own_top"), lambda c: c["st"]["edit"]["tgen"]["own_top"]),
                                (("edit", "igen", "own_top"), lambda c: c["st"]["edit"]["igen"]["own_top"]), (("hn", "HN-I", "fired"), lambda c: c["st"]["hn"]["HN-I"]["fired"]),
                                (("hn", "HN-Q", "fired"), lambda c: c["st"]["hn"]["HN-Q"]["fired"]), (("hn", "HN-X", "fired"), lambda c: c["st"]["hn"]["HN-X"]["fired"])):
                    x = g["tf"]
                    for kk in path:
                        x = x[kk]
                    th = x["values"] if isinstance(x, dict) else [x]
                    ou = [f(c) for c in cs]
                    df = float(np.max(np.abs(np.array(th, float) - np.array(ou, float))))
                    n += 1
                    worst = max(worst, df)
                    if df > 1e-9:
                        bad.append([log, lab, "gate", "/".join(path), float(np.mean(th)), float(np.mean(ou))])
    CHK["F_sanity_e0"] = {"n_cells_compared": n, "largest_abs_diff": worst, "differing": bad[:20], "ok": worst < 1e-9}


# ----------------------------------------------------------------------------- G. counts, bound, protocol facts read from the files
def part_g(A, RES):
    o = {}
    for log in DEV:
        c = A[log]["rescope"]["per_draw"][0]
        dep = A[log]["deployed"]["per_draw"][None]
        o[log] = {"n_edit_rows": c["n.edit"], "n_heldout_disjoint_rows": c["n.probe"], "n_query_clusters": c["n.clusters"], "n_invalid_y0_rows_in_heldout_disjoint": c["n.probe.invalid_y0"],
                  "n_hn": c["n.hn"], "hn_by_kind": c["n.hn.by_kind"], "hn_degenerate_y0": c["n.hn.y0_deg"],
                  "same_row_set_for_every_method_and_draw": len({(v["n.edit"], v["n.probe"], v["n.hn"]) for lab in A[log] for v in A[log][lab]["per_draw"].values()}) == 1,
                  "preedit_D1": A[log]["preedit"]["mean"]["probe.D1"], "deployed_D1": dep["probe.D1"]}
        # gate-level false firing over the 20 draws against the bound of the rule (two-block: 5 / 101; shared: 10 / 201)
        over = {}
        for p in sorted(glob.glob(f"{EO}/gate/{log}/*__a0.05.json")):
            b = jload(p)
            cs = [b["cells"][k] for k in sorted(b["cells"], key=int)]
            if "st" not in cs[0] or len(cs) < 20:
                if b.get("dense") and len(cs) == 20:
                    v = [c["split"]["disjoint.all"]["fired"] for c in cs]
                    if np.mean(v) > 5.0:
                        over[b["label"]] = {"mean_fire_hd": float(np.mean(v)), "dense": True}
                continue
            v = [c["split"]["disjoint.all"]["fired"] for c in cs]
            lit = [c["literal"]["disjoint.all"]["fired"] for c in cs] if cs[0].get("literal") else v
            bound = 100.0 * float(cs[0]["cal"].get("bound_false_firing", 0.05)) if cs[0].get("cal") else 5.0
            if np.mean(lit) > bound:
                over[b["label"]] = {"mean_fire_hd": float(np.mean(v)), "mean_literal": float(np.mean(lit)), "bound": bound, "n_capped_mean": float(np.mean([c["n_capped"] for c in cs])),
                                    "p95": float(np.percentile(v, 95))}
        o[log]["gates_whose_mean_literal_false_firing_exceeds_the_bound"] = over
    jobs = {}
    for log in DEV:
        js = sorted(glob.glob(f"{EO}/behave/{log}/_job_*.json"))
        jobs[log] = [{"hook_mismatch": j["assert_clean"].get("hook_mismatch"), "memo_conflicts": j["assert_clean"].get("memo_conflicts", j["assert_clean"].get("conflicts"))} for j in map(jload, js)]
        cells = [jload(p) for p in glob.glob(f"{EO}/behave/{log}/*.json") if not os.path.basename(p).startswith("_")]
        o[log]["behavioural_cells"] = {"n": len(cells), "hook_mismatch_max": max(c["hook_mismatch"] for c in cells), "memo_conflicts_max": max(c["memo_conflicts"] for c in cells),
                                       "driver_fingerprints": sorted({json.dumps(c["code"], sort_keys=True) for c in cells})}
    o["jobs"] = jobs
    logs_seen = sorted({r["log"] for r in RES})
    o["logs_in_results_json"] = logs_seen
    allowed = set(fx.logs.names("development")) | set(fx.logs.names("working_point"))
    o["only_development_logs_and_working_points"] = set(logs_seen) <= allowed
    beh = sorted({r["log"] for r in RES if r["pool"] in ("edits", "heldout_disjoint", "heldout_strict", "hard_negatives") and r["metric"].startswith(("edit.", "probe.", "hn."))})
    o["logs_with_behavioural_records"] = beh
    CHK["G_counts_and_discipline"] = {"ok": o["only_development_logs_and_working_points"] and set(beh) <= set(DEV) and all(o[lg]["same_row_set_for_every_method_and_draw"] for lg in DEV)
                                      and all(o[lg]["behavioural_cells"]["hook_mismatch_max"] == 0 and o[lg]["behavioural_cells"]["memo_conflicts_max"] == 0 for lg in DEV)}
    return o


# ----------------------------------------------------------------------------- H. gate-level paper numbers and further claims of the summary
def part_h(RES, PN, sel):
    o = {}
    # (1) gate-level paper numbers: mean over the 20 draws, recomputed from the cells
    base = {"rescope": "Rescope", "rescope_shared": "RescopeShared", "rescope_null": "RescopeNull", "rescope_x": "RescopeX", "cosine": "Cosine", "cosine_shared": "CosineShared",
            "ball": "Ball", "router": "Router"}
    worst, n = 0.0, 0
    for log in DEV:
        names = dict(base)
        names[sel[log]["cos_H3"]], names[sel[log]["retr_H3"]] = "BestCos", "Retr"
        names.setdefault(sel[log]["cos_H2"], "BestCosHtwo")
        names.setdefault(sel[log]["cos_H3_x"], "BestCosX")
        for lab, nm in names.items():
            cs = gate_cells(log, lab)
            dense = "st" not in cs[0]
            vals = {"GateFire": [c["split"]["disjoint.all"]["fired"] for c in cs], "BenT": [c["BEN-T"]["fired"] for c in cs if "fired" in c.get("BEN-T", {})],
                    "MmeT": [c["MME-T"]["fired"] for c in cs if "fired" in c.get("MME-T", {})]}
            if not dense:
                vals["Capped"] = [c["n_capped"] for c in cs]
                vals["GateCov"] = [50.0 * float(np.mean(unflag(c["per_edit"]["tgen"]) + unflag(c["per_edit"]["igen"]))) for c in cs]
                vals["HthreeCov"] = [50.0 * float(np.mean(unflag(c["h3"]["per_edit"]["tgen"]) + unflag(c["h3"]["per_edit"]["igen"]))) if c["h3"]["reachable"] else 0.0 for c in cs]
            for suf, v in vals.items():
                key = f"Eone{SHORT[log]}{nm}{suf}"
                if key in PN and v:
                    n += 1
                    worst = max(worst, abs(float(np.mean(v)) - PN[key]))
    CHK["H_gate_level_paper_numbers"] = {"n_numbers_compared": n, "largest_abs_diff": worst, "ok": worst < 1e-9}
    # (2) endpoint and decision paper numbers against the endpoint files
    # (3) claims of the summary read from the files
    q = "TB-Q100-ae-l2_3"
    mme = {lab: [c["MME-T"]["fired"] for c in gate_cells(q, lab)] for lab in ("rescope", "rescope_null", "rescope_shared", "router")}
    o["MME-T_firing_Qwen"] = {lab: {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))} for lab, v in mme.items()}
    for log in DEV:
        a, b = erows(log, "cosine_published", None), erows(log, "cosine_default", None)
        bs = {r["sid"]: r for r in b}
        o[f"cosine_published_vs_last_position_{SHORT[log]}"] = {"n": len(a), "n_rows_differ": sum(any(r.get(k) != bs[r["sid"]].get(k) for k in ("exact", "D1", "fired", "top_last")) for r in a),
                                                                "n_rows_any_field": sum(any(r.get(k) != bs[r["sid"]].get(k) for k in ("exact", "D1", "fired", "top_last", "token_acc")) for r in a),
                                                                "n_released_differs": sum(r.get("fired") != bs[r["sid"]].get("fired") for r in a),
                                                                "n_outcome_differs": sum(r.get("token_acc") != bs[r["sid"]].get("token_acc") for r in a),
                                                                "n_exact_differs": sum(r.get("exact") != bs[r["sid"]].get("exact") for r in a)}
    pp = [r for r in RES if r["metric"] == "prompt_position.n_rows_differ"]
    o["prompt_position_LLaVA"] = {"n_cells": len(pp), "max_rows_differ": max(r["value"] for r in pp) if pp else None}
    mv = 0.0
    idx = {(r["log"], r["method"], r["metric"], r["pool"], r["draw"]): r["value"] for r in RES}
    for (log, lab, metric, pool, d), v in idx.items():
        if metric.endswith("@prompt") and v is not None and idx.get((log, lab, metric[:-7], pool, d)) is not None:
            mv = max(mv, abs(v - idx[(log, lab, metric[:-7], pool, d)]))
    o["prompt_position_LLaVA"]["largest_cell_change_points"] = mv
    o["capped"] = {f"{SHORT[log]}/{lab}": float(np.mean([c["n_capped"] for c in gate_cells(log, lab)])) for log in DEV for lab in ("cosine", "retr_conj", "retr_min", "rescope", "rescope_x")}
    return o


# ----------------------------------------------------------------------------- I. re-run of the experiment's own analysis / rendering; gate-level spot check
def part_i():
    rr = V + "/work/rerender/out"
    o = {"available": os.path.exists(rr + "/table.txt")}
    if o["available"]:
        same = {}
        for f in ("endpoints.json", "decision.json", "paper_numbers.json", "paper_numbers_doc.json", "gate_draws.json", "missing.json"):
            same[f] = open(f"{rr}/{f}", "rb").read() == open(f"{EO}/{f}", "rb").read()
        strip = lambda r: json.dumps({k: v for k, v in r.items() if k != "written"}, sort_keys=True)     # noqa: E731
        same["results.json (records)"] = [strip(r) for r in jload(rr + "/results.json")] == [strip(r) for r in jload(EO + "/results.json")]
        ta, tb = open(rr + "/table.txt").read().split("\n")[1:], open(EO + "/table.txt").read().split("\n")[1:]
        same["table.txt (all lines but the time stamp)"] = ta == tb
        ua, ub = jload(rr + "/e2_rows_used.json"), jload(EO + "/e2_rows_used.json")
        same["e2_rows_used.json (records)"] = ua["records"] == ub["records"]
        o.update(identical=same, e2_meta_then=ub.get("e2_meta"), e2_meta_now=ua.get("e2_meta"), e2_read_now=ua.get("read_at"))
        CHK["I_rerun_of_analysis_and_rendering"] = {"n_files": len(same), "n_identical": sum(same.values()), "ok": all(same.values())}
    gc = {}
    for p in sorted(glob.glob(f"{VO}/gatecheck_*.json")):
        g = jload(p)
        gc[g["log"]] = {k: g[k] for k in ("n_cells", "n_digest_differs", "largest_abs_diff", "ok")}
    if gc:
        CHK["I_gate_cells_rebuilt"] = {"n_logs": len(gc), "n_cells": sum(g["n_cells"] for g in gc.values()), "n_digest_differs": sum(g["n_digest_differs"] for g in gc.values()),
                                       "largest_abs_diff": max(g["largest_abs_diff"] for g in gc.values()), "ok": all(g["ok"] for g in gc.values())}
    o["gatecheck"] = gc
    fr, FRE = {}, None
    for log in DEV:
        p = f"{V}/work/frontier/{log}/frontier.json"
        if os.path.exists(p):
            FRE = FRE or jload(EO + "/frontier.json")
            fr[log] = json.dumps(jload(p)["logs"][log], sort_keys=True) == json.dumps(FRE["logs"][log], sort_keys=True)
    if fr:
        CHK["I_frontier_rerun_one_log_per_process"] = {"n_logs": len(fr), "n_identical": sum(fr.values()), "ok": all(fr.values())}
    if os.path.exists(VO + "/draws_check.json"):
        dc = jload(VO + "/draws_check.json")
        CHK["I_calibration_draws_from_REPAIR"] = {"n_draws_per_log": 20, "draws_equal_to_own_seeded_draw": min(v["draws_equal_to_own_seeded_draw"] for v in dc["logs"].values()),
                                                  "rows_of_HELD-OUT_in_a_calibration_or_null_sample": sum(v["rows_of_HELD-OUT_in_a_calibration_or_null_sample"] for v in dc["logs"].values()),
                                                  "rescope_digest_equal_all_draws": min(v["rescope_digest_equal_all_draws"] for v in dc["logs"].values()), "ok": bool(dc["ok"])}
    if os.path.exists(VO + "/e1c_check.json"):
        e = jload(VO + "/e1c_check.json")
        CHK["I_E1c_matched_coverage_own_computation"] = {"n_cells": sum(len(f_["cells"]) for l_ in e["logs"].values() for f_ in l_.values()),
                                                         "n_reachable": sum(c["reachable"] for l_ in e["logs"].values() for f_ in l_.values() for c in f_["cells"]),
                                                         "reachability_equal": all(c["reachable"] == c["experiment_reachable"] for l_ in e["logs"].values() for f_ in l_.values() for c in f_["cells"]),
                                                         "largest_abs_diff": e["largest_abs_diff"], "ok": bool(e["ok"])}
    return o


# ----------------------------------------------------------------------------- J. what the summary does not carry
def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts."""
    from math import comb
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def part_j(A, TF, RES, PN):
    o = {"mcnemar_D1_heldout_disjoint_rescope_vs_deployed": {}, "hn_D1_by_degenerate_pre_edit_answer": {}, "hn_D1_spread_over_draws": {}, "generation_draw0": {}}
    for log in DEV:
        dep = {r["sid"]: r["D1"] for r in TF[log]["deployed"][None] if r["kind"] == "probe" and r["disjoint"]}
        per = {}
        for d in DRAWS:
            a = {r["sid"]: r["D1"] for r in TF[log]["rescope"][d] if r["kind"] == "probe" and r["disjoint"]}
            b = sum(dep[s_] and not a[s_] for s_ in dep)
            c = sum(a[s_] and not dep[s_] for s_ in dep)
            per[d] = {"only_deployed": b, "only_rescope": c, "p_exact": mcnemar_exact(b, c)}
        o["mcnemar_D1_heldout_disjoint_rescope_vs_deployed"][log] = per
        sub = {}
        for lab in ("deployed", "rescope", "rescope_x", "cosine_shared", "white_1_shared", "retr_sum"):
            by = TF[log][lab]
            cell = {}
            for k in HN:
                for deg in (False, True):
                    v = [pct(r["D1"] for r in rows if r["kind"] == "hn" and r["hn_type"] == k and bool(r.get("y0_deg")) == deg) for rows in by.values()]
                    n = sum(1 for r in next(iter(by.values())) if r["kind"] == "hn" and r["hn_type"] == k and bool(r.get("y0_deg")) == deg)
                    cell[f"{k}.deg={int(deg)}"] = {"n": n, "D1": None if n == 0 else float(np.mean(v))}
            sub[lab] = cell
        o["hn_D1_by_degenerate_pre_edit_answer"][log] = sub
        o["hn_D1_spread_over_draws"][log] = {lab: {k: [A[log][lab]["per_draw"][str(d) if str(d) in A[log][lab]["per_draw"] else d][f"hn.{k}.D1"] for d in DRAWS] for k in HN}
                                             for lab in ("rescope", "rescope_x", "cosine", "white_1_shared", "retr_sum")}
        g = {}
        for lab, d in (("deployed", None), ("rescope", 0), ("rescope_x", 0), ("white_1_shared", 0), ("retr_sum", 0), ("dual", None)):
            rows = erows(log, lab, d, "gen")
            hn = [r for r in rows if r["kind"] == "hn"]
            pr = [r for r in rows if r["kind"] == "probe" and r.get("in_d2_set")]
            g[lab] = {"D2": pct(r["D2"] for r in pr), "n_pre_correct": len(pr),
                      **{f"wrong_target.{k}": (pct(r["wrong_target"] for r in hn if r["hn_type"] == k) if all(r.get("wrong_target") is not None for r in hn) else None) for k in HN}}
        o["generation_draw0"][log] = g
    keys = list(PN)
    o["paper_numbers_json"] = {"n": len(keys), "n_names_with_wrong_target": sum("rong" in k for k in keys), "n_names_with_D2": sum("DTwo" in k or "Dtwo" in k for k in keys),
                               "n_names_with_generation": sum("Gen" in k and "TGen" not in k and "IGen" not in k for k in keys)}
    return o


def main():
    os.makedirs(VO, exist_ok=True)
    R = fx.io.Results("E1-verify", VO)
    RES = jload(EO + "/results.json")
    PN = jload(EO + "/paper_numbers.json")
    EP, DEC, SELF = jload(EO + "/endpoints.json"), jload(EO + "/decision.json"), jload(EO + "/best_cosine_gate.json")
    sel = {}
    for log in DEV:
        b = SELF["selected"][fx.logs.info(log)["backbone"]]
        sel[log] = {"cos_H3": b["cosine"]["H3"], "cos_H3_x": b["cosine"]["H3_x"], "cos_H2": b["cosine"]["H2"], "cos_H2_x": b["cosine"]["H2_x"], "retr_H3": b["retr"]["H3"], "retr_H2": b["retr"]["H2"]}
    A, TF = part_a(R, RES, PN, sel)
    B = part_b(R, TF, EP)
    C, dec = part_c(R, EP, DEC)
    D = part_d(R, SELF)
    E = part_e(R)
    part_f(A)
    G = part_g(A, RES)
    Hh = part_h(RES, PN, sel)
    Ii = part_i()
    Jj = part_j(A, TF, RES, PN)
    fx.io.jdump({"checks": CHK, "A": {lg: {lab: {"mean": v["mean"], "n_draws": v["n_draws"], "per_draw": {str(d): {k: x for k, x in c.items() if not k.startswith("n.")} for d, c in v["per_draw"].items()}}
                                           for lab, v in A[lg].items()} for lg in A}, "B_H2": B, "C_H3": C, "C_decision": dec, "D_selection": D, "E_own_gpu": E, "G": G, "H": Hh, "I": Ii, "J": Jj, "selection": sel},
                VO + "/checks.json")
    R.save(script="v_stats.py")
    print(json.dumps({k: {x: y for x, y in v.items() if not isinstance(y, (list, dict))} for k, v in CHK.items()}, indent=1))


if __name__ == "__main__":
    main()
