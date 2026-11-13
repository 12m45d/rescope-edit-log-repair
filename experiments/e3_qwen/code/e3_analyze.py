"""E3 (Qwen): statistics.  Reads the gate-level cells (out/gate), the behavioural row files (out/rows) with their cell records
(out/behave) and the factor-level files (out/factor); writes out/results.json (schema of fx.io), out/gate_draws.json (every gate-level
number of every calibration draw), out/endpoints.json, out/scale_series.json, out/retention_series.json, out/missing.json,
out/paper_numbers.json + paper_numbers_doc.json and, for the development log, out/sanity_compare.json.  No number is typed by hand.

    python e3_analyze.py [--no-cache]        (CPU; re-runnable after every block; per-log results are cached by the state of their inputs)

Statistics (PROTOCOL.md section 4, R10, R25 (3)): crossed bootstrap over calibration draws x units (2000 resamples; probe units = query
clusters, edit units = edits; hard negatives: the owning edit); every summary record carries n_draws and by_draw (the per-draw values).
A method without calibration has one row set (n_draws = 1, calibrated = False).
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, _go.path.join(_gd, "experiments", "e3_qwen", "out", "cache"), needs=("experiments/e3_qwen/out/gate", "experiments/e3_qwen/out/behave", "experiments/e3_qwen/out/rows"))
import argparse
import glob
import hashlib
import os
import time

from e3c import *  # noqa: F401,F403
import e3c

MX, ST = fx.metrics, fx.stats
MISSING = []
SUB = ("all", "first100", "later")


# ----------------------------------------------------------------------------- helpers
def rec(recs, method, metric, value, pool, **kw):
    r = {"method": method, "metric": metric, "value": value, "pool": pool}
    r.update(kw)
    recs.append(r)
    return r


def pct(v):
    v = [float(x) for x in v]
    return 100.0 * float(np.mean(v)) if v else None


def rate(by, value, unit, where=None, ci=True):
    """Rate (%) of `value` over the rows selected by `where`, per calibration draw and pooled with the crossed bootstrap."""
    val = value if callable(value) else (lambda r: r[value])
    per, n = [], 0
    for d, rows in by.items():
        v = [val(r) for r in rows if where is None or where(r)]
        if any(x is None for x in v):
            return None
        per.append(pct(v))
        n = len(v)
    if not per or any(x is None for x in per):
        return None
    out = {"value": float(np.mean(per)), "by_draw": per, "n_draws": len(per), "n_rows": n, "ci_lo": None, "ci_hi": None, "n_units": None}
    if ci:
        g = MX.grid(by, lambda r: float(val(r)), unit=unit, where=where)
        s = ST.crossed_bootstrap(g["num"], g["den"], scale=100.0)
        out.update(value=s["value"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n_units=s["n_units"])
    return out


def put(recs, method, metric, st, pool, **kw):
    if st is None:
        return
    rec(recs, method, metric, st["value"], pool, ci_lo=st.get("ci_lo"), ci_hi=st.get("ci_hi"), n_units=st.get("n_units"), n_draws=st.get("n_draws"),
        by_draw=st.get("by_draw"), n_rows=st.get("n_rows"), **kw)


def rows_of(log, label, d, part):
    tag = f"{label}__d{'x' if d is None else d}__{part}"
    p = os.path.join(OUT, "rows", f"{log}__{tag}.jsonl.gz")
    c = os.path.join(BEH, log, tag + ".json")
    if not (os.path.exists(p) and os.path.exists(c)):
        return None
    return fx.io.load_rows(p)


def cellrec(log, label, d, part):
    c = os.path.join(BEH, log, f"{label}__d{'x' if d is None else d}__{part}.json")
    return jload(c) if os.path.exists(c) else None


def sub_where(sub, later):
    if sub == "all":
        return lambda r: True
    if sub == "first100":
        return lambda r: r["edit"] < 100
    return lambda r: r["edit"] in later


# ----------------------------------------------------------------------------- gate level
def g_metrics(c):
    m = {}
    st, sp = c["st"], c["split"]
    for k in ("all", "image", "text", "near", "far", "qd"):
        m["fire.hd" + ("" if k == "all" else "." + k)] = sp[k]["fired"]
    m["n.hd"], m["n.hd.near"], m["n.hd.qd"] = sp["all"]["n"], sp["near"]["n"], sp["qd"]["n"]
    lit = c.get("literal")
    if lit and "all" in lit:
        m["fire.hd.literal"], m["fire.hd.qd.literal"], m["fire.hd.near.literal"] = lit["all"]["fired"], lit["qd"]["fired"], lit["near"]["fired"]
        if lit.get("built"):
            m["literal.rel_cov"] = lit.get("rel_own_top")
    for k in ("tgen", "igen", "rel"):
        e = st["edit"][k]
        m[f"cov.{k}"], m[f"cand.{k}"], m[f"other.{k}"], m[f"fired.{k}"] = e["own_top"], e["own_cand"], e["other_top"], e["fired"]
    m["h2.cov"] = c["h2_cov"]
    for k in HN:
        if k in st["hn"]:
            m[f"hn.{k}.fired"], m[f"hn.{k}.own"], m[f"hn.{k}.n"] = st["hn"][k]["fired"], st["hn"][k]["own_top"], st["hn"][k]["n"]
    hv = [st["hn"][k]["fired"] for k in HN if k in st["hn"] and st["hn"][k]["fired"] is not None]
    m["hn.mean.fired"] = float(np.mean(hv)) if hv else None
    m["hn.all.fired"] = st["hn"]["all"]["fired"]
    ln = c["lineage_hd"]
    w = {int(k): v for k, v in ln["hist"].items()}
    m["lineage.hd.n_firing"] = ln["n_firing"]
    m["lineage.hd.width_mean"] = float(sum(k * v for k, v in w.items()) / ln["n_firing"]) if ln["n_firing"] else None
    m["lineage.hd.width_max"] = max(w) if w else None
    m["lineage.hd.multi"] = 100.0 * sum(v for k, v in w.items() if k > 1) / ln["n_firing"] if ln["n_firing"] else None
    cal = c.get("cal") or {}
    if c.get("alpha") is not None:
        m["n_capped"] = c["n_capped"]
        m["own.silenced_before_cap"] = cal.get("n_edits_silenced_by_literal_rule", cal.get("n_own_requests_silenced_by_literal_rule"))
        m["eta"] = cal.get("eta") if not cal.get("eta_infinite") else None
        m["eta_infinite"] = 1.0 if cal.get("eta_infinite") else 0.0
        m["bound"] = None if cal.get("bound_false_firing") is None else 100.0 * cal["bound_false_firing"]
    th = c.get("thresholds")
    if th and th.get("n_finite"):
        m["thr.min"], m["thr.median"], m["thr.max"] = th["min"], th["median"], th["max"]
    op = c.get("own_prompts")
    T = st["edit"]["rel"]["n"]
    m["own.not_candidate_after_cap"] = op["n_not_candidate"] if op else None
    m["own.not_released"] = op["n_not_released"] if op else round((100.0 - st["edit"]["rel"]["own_top"]) * T / 100.0)
    m["own.released_by_other"] = c["per_edit_other"]["rel"].count("1")
    b = c.get("BEN-T") or {}
    if "fired" in b:
        m["BEN-T.fired"], m["BEN-T.n"] = b["fired"], b["n"]
        m["BEN-T.far.fired"], m["BEN-T.far.n"], m["BEN-T.related.n"], m["BEN-T.related.fired"] = b["far"]["fired"], b["far"]["n"], b["related"]["n"], b["related"]["fired"]
    bq = c.get("BEN-T.qd") or (b if c.get("mode") == "qd" else {})
    if "fired" in bq:
        m["BEN-T.qd.fired"], m["BEN-T.qd.n"] = bq["fired"], bq["n"]
    mm = c.get("MME-T") or {}
    if "fired" in mm:
        m["MME-T.fired"], m["MME-T.n"] = mm["fired"], mm["n"]
    m["top1.n_differs"] = (st.get("top1") or {}).get("all", {}).get("n_differs")
    m["prompt.n_rows_differ"] = sum(c["n_rows_differ_tf_vs_prompt"].values())
    for key, pre in (("h3", "h3"), ("h3_unseen", "h3u")):
        h = c.get(key)
        if h is None:
            continue
        m[pre + ".reachable"] = 1.0 if h["reachable"] else 0.0
        m[pre + ".n_at_cap"] = h.get("n_edits_at_cap")
        if h["reachable"]:
            m[pre + ".cov"], m[pre + ".tgen"], m[pre + ".igen"] = h["rephrase"]["mean"], h["rephrase"]["tgen"], h["rephrase"]["igen"]
            m[pre + ".delta"] = h["delta"]
            for k in HN:
                if k in h["test"]:
                    m[f"{pre}.test.{k}.fired"], m[f"{pre}.test.{k}.own"], m[f"{pre}.test.{k}.n"] = h["test"][k]["fired"], h["test"][k]["own_top"], h["test"][k]["n"]
            m[pre + ".test.mean"] = h["test"]["mean_fired"]
            m[pre + ".hd.fired"] = h["heldout_disjoint"]["fired"]
            m[pre + ".hd.near.fired"] = h["heldout_disjoint"]["near"]["fired"]
            m[pre + ".fit.mean_at_0"] = h["fit"]["at_0"]["mean"]
        else:
            m[pre + ".cov"] = 0.0                                       # e3c.H3_UNREACHABLE_RULE
    if "near_scope" in c:
        for form, pre in (("all", "ns"), ("unseen_foreign", "nsu")):
            for k in HN:
                if k in c["near_scope"][form]:
                    m[f"{pre}.{k}.fired"], m[f"{pre}.{k}.n"] = c["near_scope"][form][k]["fired"], c["near_scope"][form][k]["n"]
    return m


def cbm(mat, den=None):
    s = ST.crossed_bootstrap(np.asarray(mat, dtype=float), None if den is None else np.asarray(den, dtype=float), scale=100.0)
    return {"value": s["value"], "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"], "n_units": s["n_units"], "n_draws": s["n_draws"]}


def unit_mat(flag_rows, units, keep=None):
    """(D, U) sums and counts per unit from (D, n) flags and the unit of every column."""
    F = np.asarray(flag_rows, dtype=float)
    u = np.asarray(units)
    if keep is not None:
        F, u = F[:, keep], u[keep]
    if u.size == 0:
        return None, None
    names, inv = np.unique(u, return_inverse=True)
    num = np.zeros((F.shape[0], names.size))
    den = np.zeros((F.shape[0], names.size))
    for i in range(F.shape[0]):
        np.add.at(num[i], inv, F[i])
        np.add.at(den[i], inv, 1.0)
    return num, den


def gate_level(log, recs):
    """-> draws[label][mode] = {metric: [value per draw]}, cells[label][mode] = [cell per draw]."""
    draws, cells = {}, {}
    cp = os.path.join(GATE, log, "_context.json")
    if not os.path.exists(cp):
        MISSING.append({"log": log, "part": "gate level"})
        return draws, cells
    ctx = jload(cp)
    later = np.array(ctx["later_edits"], dtype=np.int64)
    T = ctx["T"]
    hd_q = np.array(ctx["hd"]["query_id"])
    hd_near, hd_qd = unflags(ctx["hd"]["near"]), unflags(ctx["hd"]["qd"])
    hn_edit, hn_type = np.array(ctx["hn"]["edit"]), np.array(ctx["hn"]["hn_type"])
    for p in sorted(glob.glob(os.path.join(GATE, log, "*__*.json"))):
        b = jload(p)
        lab, mode = b["label"], b["mode"]
        cs = [b["cells"][k] for k in sorted(b["cells"], key=int)]
        kw = dict(alpha=None if b["fixed"] else ALPHA, n=None if b["fixed"] else N_CAL, variant="qdisjoint" if mode == "qd" else "")
        if "not_defined" in cs[0]:
            rec(recs, lab, "gate.not_defined", None, "gate_level", note=cs[0]["not_defined"], **kw)
            continue
        ms = [g_metrics(c) for c in cs]
        keys = list(dict.fromkeys(k for m in ms for k in m))
        d = {k: [m.get(k) for m in ms] for k in keys}
        draws.setdefault(lab, {})[mode] = d
        cells.setdefault(lab, {})[mode] = cs
        for k, v in d.items():
            vals = [x for x in v if x is not None]
            if not vals:
                continue
            s = ST.over_draws(vals)
            rec(recs, lab, "gate." + k, s["mean"], "gate_level", ci_lo=s["p05"], ci_hi=s["p95"], n_units=s["n_draws"], n_draws=s["n_draws"], by_draw=v,
                interval="5th-95th percentile over calibration draws", vmin=s["min"], vmax=s["max"], **kw)
        # crossed bootstrap over the calibration draws x units (R10)
        H = np.array([unflags(c["hd_flags"]) for c in cs])
        for nm, keep in (("fire.hd", None), ("fire.hd.near", hd_near), ("fire.hd.far", ~hd_near), ("fire.hd.qd", hd_qd)):
            num, den = unit_mat(H, hd_q, keep)
            if num is not None:
                rec(recs, lab, f"gate.{nm}@x", None, "gate_level", **kw).update(cbm(num, den))
        for k in HN + ("all",):
            keep = np.ones(hn_type.size, dtype=bool) if k == "all" else hn_type == k
            if keep.any():
                num, den = unit_mat(np.array([unflags(c["hn_flags"]) for c in cs]), hn_edit, keep)
                rec(recs, lab, f"gate.hn.{k}.fired@x", None, "gate_level", **kw).update(cbm(num, den))
        for sub, idx in (("all", np.arange(T)), ("first100", np.arange(min(100, T))), ("later", later)):
            if idx.size == 0:
                continue
            for k in ("tgen", "igen", "rel"):
                rec(recs, lab, f"gate.cov.{sub}.{k}@x", None, "gate_level", **kw).update(cbm(np.array([unflags(c["per_edit"][k])[idx] for c in cs])))
            rec(recs, lab, f"gate.cov.{sub}.rephrase@x", None, "gate_level", **kw).update(
                cbm(np.array([0.5 * (unflags(c["per_edit"]["tgen"])[idx].astype(float) + unflags(c["per_edit"]["igen"])[idx]) for c in cs])))
    return draws, cells


def gate_cov_diff(cells, a, b, key="per_edit", idx=None):
    """Gate-level coverage form over the 20 draws: paired crossed bootstrap of (T-Gen + I-Gen) / 2 coverage, units = edits."""
    def mat(lab):
        cs = cells[lab]["reg"]
        if len(cs) == 1:
            cs = cs * N_DRAWS
        out = []
        for c in cs:
            pe = c["per_edit"] if key == "per_edit" else (c["h3"]["per_edit"] if c["h3"]["reachable"] else None)
            if pe is None:
                out.append(None)
                continue
            v = 0.5 * (unflags(pe["tgen"]).astype(float) + unflags(pe["igen"]).astype(float))
            out.append(v if idx is None else v[idx])
        n = next((len(x) for x in out if x is not None), None)
        if n is None:
            return None
        return np.array([x if x is not None else np.zeros(n) for x in out])        # unreachable draw: coverage 0 (e3c.H3_UNREACHABLE_RULE)
    if a not in cells or b not in cells or "reg" not in cells[a] or "reg" not in cells[b]:
        return None
    A, B = mat(a), mat(b)
    if A is None or B is None or A.shape != B.shape:
        return None
    d = ST.crossed_bootstrap_diff(A, B, scale=100.0)
    return {k: d[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "mean_a", "mean_b")} | {"n_draws": int(A.shape[0]), "n_units": int(A.shape[1])}


# ----------------------------------------------------------------------------- behaviour
def split_tf(rows):
    return ([r for r in rows if r["kind"] == "edit"], [r for r in rows if r["kind"] == "probe"], [r for r in rows if r["kind"] == "hn"])


def tf_stats(recs, lab, by, later, qd_csid, dep_e, kw, masked=False):
    pre = "qd." if masked else ""
    var = dict(kw, variant="qdisjoint" if masked else "")
    parts = {d: split_tf(rows) for d, rows in by.items()}
    E, P, Hn = ({d: p[i] for d, p in parts.items()} for i in range(3))
    ex = lambda r: bool(r["exact"])                                                    # noqa: E731
    # (1) edits: all T edits, the first 100 edits of the stream, later edits only (R8)
    for sub in SUB:
        w0 = sub_where(sub, later)
        if not any(w0(r) for r in next(iter(E.values()))):
            continue
        for m in ("rel", "tgen", "igen", "rephrase"):
            w = (lambda r, m=m, w0=w0: w0(r) and (r["metric"] in ("tgen", "igen") if m == "rephrase" else r["metric"] == m))
            put(recs, lab, f"{pre}edit.{sub}.{m}.exact", rate(E, ex, "edit", w), "edits", **var)
            put(recs, lab, f"{pre}edit.{sub}.{m}.token", rate(E, "token_acc", "edit", w, ci=(sub == "all" and not masked)), "edits", **var)
            if next(iter(E.values()))[0].get("own_released_last") is not None:
                put(recs, lab, f"{pre}edit.{sub}.{m}.released_own", rate(E, lambda r: bool(r.get("own_released_last")), "edit", w, ci=False), "edits", **var)
                put(recs, lab, f"{pre}edit.{sub}.{m}.released_other", rate(E, lambda r: bool(r.get("fired")) and not bool(r.get("own_released_last")), "edit", w, ci=False), "edits", **var)
    # R7: edits kept / gained against the deployed log
    if dep_e is not None and not masked:
        KG = {d: MX.kept_gained_rows(E[d], dep_e) for d in E}
        for sub in SUB:
            w0 = sub_where(sub, later)
            if not any(w0(r) for r in next(iter(KG.values()))):
                continue
            for m in ("rel", "tgen", "igen"):
                for f in ("kept", "gained", "lost"):
                    put(recs, lab, f"kg.{sub}.{m}.{f}", rate(KG, lambda r, f=f: bool(r[f]), "edit", lambda r, m=m, w0=w0: w0(r) and r["metric"] == m, ci=(sub == "all")), "edits", **var)
    # (2) heldout_disjoint probes; near-scope probes (question string of an edit of the log) separately; question-disjoint rows
    gate = next(iter(P.values()))[0].get("fired") is not None
    sets = (("", lambda r: True), ("near.", lambda r: r.get("near_edit") is not None), ("far.", lambda r: r.get("near_edit") is None),
            ("qdrows.", lambda r: r["csid"] in qd_csid))
    for nm, w in sets:
        if masked and nm != "qdrows.":
            continue
        tag = f"{pre}probe." if masked else f"probe.{nm}"
        if not any(w(r) for r in next(iter(P.values()))):
            continue
        if gate:
            put(recs, lab, tag + "fired", rate(P, lambda r: bool(r["fired"]), "query_id", w), "heldout_disjoint", **var)
        put(recs, lab, tag + "D1", rate(P, lambda r: bool(r["D1"]), "query_id", w), "heldout_disjoint", **var)
        put(recs, lab, tag + "tok_loc", rate(P, "tok_loc", "query_id", w, ci=False), "heldout_disjoint", **var)
        rec(recs, lab, tag + "n", sum(1 for r in next(iter(P.values())) if w(r)), "heldout_disjoint", **var)
    if not masked:
        p0 = next(iter(P.values()))
        rec(recs, lab, "probe.n_y0_degenerate", sum(1 for r in p0 if r.get("y0_deg")), "heldout_disjoint", **var)
        rec(recs, lab, "probe.n_invalid_excluded", None, "heldout_disjoint", note="invalid pre-edit answers are excluded by the driver's pool (counts: describe.n_probe_rows - n_probe_valid)", **var)
    # (3) near-scope negatives of the log's edits, by kind
    if next(iter(Hn.values())):
        hgate = next(iter(Hn.values()))[0].get("fired") is not None
        for k in HN + ("all",):
            w = (lambda r: True) if k == "all" else (lambda r, k=k: r["hn_type"] == k)
            if not any(w(r) for r in next(iter(Hn.values()))):
                continue
            if hgate:
                put(recs, lab, f"{pre}hn.{k}.fired", rate(Hn, lambda r: bool(r["fired"]), "edit", w), "hard_negatives", **var)
                put(recs, lab, f"{pre}hn.{k}.own", rate(Hn, lambda r: bool(r.get("own_released_last")), "edit", w, ci=False), "hard_negatives", **var)
            put(recs, lab, f"{pre}hn.{k}.D1", rate(Hn, lambda r: bool(r["D1"]), "edit", w), "hard_negatives", **var)
            put(recs, lab, f"{pre}hn.{k}.tok_loc", rate(Hn, "tok_loc", "edit", w, ci=False), "hard_negatives", **var)
            rec(recs, lab, f"{pre}hn.{k}.n", sum(1 for r in next(iter(Hn.values())) if w(r)), "hard_negatives", **var)
    return E, P, Hn


def h3_rows(log, lab, draws):
    out, info = {}, {}
    for d in draws:
        r = cellrec(log, lab, d, "h3")
        if r is None:
            MISSING.append({"log": log, "label": lab, "draw": d, "part": "h3"})
            return None, None
        info[d] = r
        if r["reachable"]:
            out[d] = fx.io.load_rows(os.path.join(OUT, "rows", f"{log}__{lab}__d{d}__h3.jsonl.gz"))
        else:
            out[d] = [{"kind": "edit", "metric": m, "edit": e, "exact": False, "token_acc": 0.0, "own_released_last": False} for m in ("tgen", "igen") for e in r["odd_edits"]]
    return out, info


def behaviour(log, recs, lb, ctx, cells):
    draws = bdraws(log)
    later = set(ctx["later_edits"])
    qd_csid = {s for s, f in zip(ctx["hd"]["csid"], ctx["hd"]["qd"]) if f == "1"}
    tf = {}
    dep = rows_of(log, "deployed", None, "tf")
    dep_e = split_tf(dep)[0] if dep else None
    fixed = lb["weights"] + lb["fixed"] + lb["lib"]
    for lab in fixed + lb["cal"]:
        is_fixed = lab in fixed
        kw = dict(alpha=None if is_fixed else ALPHA, n=None if is_fixed else N_CAL, calibrated=not is_fixed)
        sk = os.path.join(BEH, log, f"{lab}__dx__SKIPPED.json")
        if os.path.exists(sk):
            s = jload(sk)
            rec(recs, lab, "skipped_for_cost", s["timing"]["projected_pass_s"], "behaviour", note="one pass does not fit in 40 minutes (projection from 100 seeded sequences; seconds)",
                timing=s["timing"], **kw)
            continue
        by = {}
        for d in ([None] if is_fixed else draws):
            rows = rows_of(log, lab, d, "tf")
            if rows is not None:
                by[d] = rows
            else:
                MISSING.append({"log": log, "label": lab, "draw": d, "part": "tf"})
        if not by:
            continue
        tf[lab] = by
        tf_stats(recs, lab, by, later, qd_csid, dep_e, kw)
        crs = [cellrec(log, lab, d, "tf") for d in by]
        rec(recs, lab, "n_capped.behaviour_draws", float(np.mean([c["n_capped"] for c in crs])) if not is_fixed else None, "behaviour", by_draw=[c.get("n_capped") for c in crs],
            n_draws=len(by), draws=[d for d in by], **kw)
        if lab in lb["lib"] and crs[0].get("timing"):
            rec(recs, lab, "timing.pass_s", crs[0].get("seconds_cell"), "behaviour", timing=crs[0]["timing"], where_cal=crs[0].get("where_cal"), **kw)
        # second read-out (R8): calibration from the question-disjoint REPAIR rows
        if not is_fixed:
            byq = {d: r for d in draws if (r := rows_of(log, lab, d, "tfq")) is not None}
            if len(byq) == len(draws):
                tf_stats(recs, lab, byq, later, qd_csid, None, kw, masked=True)
            else:
                MISSING.append({"log": log, "label": lab, "part": "tfq", "have": sorted(byq)})
        # (6) / R7: D2 as a rate of the pre-correct rows (draw 0; rows without calibration: their one row set)
        d2 = rows_of(log, lab, None if is_fixed else 0, "d2")
        if d2 is None:
            MISSING.append({"log": log, "label": lab, "part": "d2"})
        else:
            s = MX.d2_stats(d2, ci=True)
            rec(recs, lab, "d2.rate", s["D2"]["value"], "heldout_disjoint", ci_lo=s["D2"]["ci_lo"], ci_hi=s["D2"]["ci_hi"], n_units=s["D2"]["n_units"], n_draws=1, draw=None if is_fixed else 0,
                n_pre_correct=s["n_pre_correct"], n_D2=s["n_D2"], **kw)
            rec(recs, lab, "d2.n_pre_correct", s["n_pre_correct"], "heldout_disjoint", draw=None if is_fixed else 0, **kw)
            rec(recs, lab, "d2.n_D2", s["n_D2"], "heldout_disjoint", draw=None if is_fixed else 0, **kw)
    # the published all-positions cosine gate against its last-position form: rows whose flags differ
    if "cosine_published" in tf and "cosine_default" in tf:
        a, b = next(iter(tf["cosine_published"].values())), {r["csid"]: r for r in next(iter(tf["cosine_default"].values()))}
        nd = sum(any(r.get(k) != b[r["csid"]].get(k) for k in ("exact", "token_acc", "D1", "tok_loc", "fired")) for r in a)
        rec(recs, "cosine_published", "n_rows_differ_from_cosine_default", nd, "behaviour", n_rows=len(a), note="teacher-forced rows (edits, heldout_disjoint, hard negatives) whose exact / token / D1 / firing flags differ from the last-position form")
    # free generation of the default method (draw 0)
    g = rows_of(log, "rescope", 0, "gen")
    if g is None:
        MISSING.append({"log": log, "label": "rescope", "part": "gen"})
    else:
        ge, gp = [r for r in g if r["kind"] == "edit"], [r for r in g if r["kind"] == "probe"]
        gs = MX.gen_stats(ge)
        for m in ("rel", "tgen", "igen"):
            for f in ("eq", "contains", "starts", "fired", "n", "n_degenerate"):
                rec(recs, "rescope", f"gen.first100.{m}.{f}", gs[m][f], "edit_prompts", alpha=ALPHA, n=N_CAL, draw=0)
        v = [r for r in gp if r.get("y0_valid")]
        if v and all("agree" in r for r in v):
            s = ST.bootstrap([float(bool(r["agree"])) for r in v], [r["query_id"] for r in v], scale=100.0)
            rec(recs, "rescope", "gen.probe200.eq_pre", s["value"], "heldout_disjoint", ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n_units=s["n_units"], alpha=ALPHA, n=N_CAL, draw=0, n_rows=len(v))
            rec(recs, "rescope", "gen.probe200.fired", pct(bool(r.get("fired")) for r in v), "heldout_disjoint", alpha=ALPHA, n=N_CAL, draw=0, n_rows=len(v))
    # H3 (behaviour): rephrase exact on the odd edits at the raised scale, test hard negatives, heldout_disjoint
    h3 = {}
    for lab in lb["h3"]:
        rows, info = h3_rows(log, lab, draws)
        if rows is None:
            continue
        h3[lab] = (rows, info)
        kw = dict(alpha=ALPHA, n=N_CAL, calibrated=True)
        nun = sum(not info[d]["reachable"] for d in draws)
        rec(recs, lab, "h3.n_unreachable", nun, "h3", n_draws=len(draws), **kw)
        rec(recs, lab, "h3.n_at_cap", None, "h3", by_draw=[info[d].get("n_edits_at_cap") for d in draws], n_draws=len(draws), **kw)["value"] = (
            float(np.mean([info[d]["n_edits_at_cap"] for d in draws if info[d].get("n_edits_at_cap") is not None])) if any(info[d].get("n_edits_at_cap") is not None for d in draws) else None)
        rec(recs, lab, "h3.delta", None, "h3", by_draw=[info[d].get("delta") for d in draws], n_draws=len(draws), **kw)
        ed = lambda r: r["kind"] == "edit"                                                # noqa: E731
        for m in ("tgen", "igen", "rephrase"):
            w = (lambda r, m=m: ed(r) and (r["metric"] in ("tgen", "igen") if m == "rephrase" else r["metric"] == m))
            put(recs, lab, f"h3.{m}.exact", rate(rows, lambda r: bool(r["exact"]), "edit", w), "h3", **kw)
            put(recs, lab, f"h3.{m}.released_own", rate(rows, lambda r: bool(r.get("own_released_last")), "edit", w), "h3", **kw)
        rr = {d: rows[d] for d in draws if info[d]["reachable"]}
        if rr:
            forms = [("", None)]
            if lab.endswith("_x"):
                uf = next(iter(info.values())).get("unseen_foreign")
                if uf:
                    forms.append(("unseen.", set(uf["csids"])))
                    for k in HN:
                        rec(recs, lab, f"h3.unseen.subset.{k}.n_kept_of_all", uf["n"][k]["kept"], "h3", n_all=uf["n"][k]["all"], **kw)
            for pre, keep in forms:
                for k in HN + ("all",):
                    wk = (lambda r, k=k, keep=keep: r["kind"] == "hn" and (k == "all" or r["hn_type"] == k) and (keep is None or r["csid"] in keep))
                    wt, wg = (lambda r, wk=wk: wk(r) and r.get("part") != "gen"), (lambda r, wk=wk: wk(r) and r.get("part") == "gen")
                    n = sum(1 for r in next(iter(rr.values())) if wt(r))
                    if not n:
                        continue
                    ci = n >= 20
                    put(recs, lab, f"h3.{pre}test.{k}.fired", rate(rr, lambda r: bool(r["fired"]), "edit", wt, ci=ci), "h3", **kw)
                    put(recs, lab, f"h3.{pre}test.{k}.own", rate(rr, lambda r: bool(r.get("own_released_last")), "edit", wt, ci=False), "h3", **kw)
                    put(recs, lab, f"h3.{pre}test.{k}.D1", rate(rr, lambda r: bool(r["D1"]), "edit", wt, ci=ci), "h3", **kw)
                    put(recs, lab, f"h3.{pre}test.{k}.wrong_target", rate(rr, lambda r: bool(r.get("wrong_target")), "edit", wg, ci=ci), "h3", **kw)
                    rec(recs, lab, f"h3.{pre}test.{k}.n", n, "h3", **kw)
            pw = lambda r: r["kind"] == "probe"                                           # noqa: E731
            put(recs, lab, "h3.hd.fired", rate(rr, lambda r: bool(r["fired"]), "query_id", pw), "h3", **kw)
            put(recs, lab, "h3.hd.D1", rate(rr, lambda r: bool(r["D1"]), "query_id", pw), "h3", **kw)
            put(recs, lab, "h3.hd.near.fired", rate(rr, lambda r: bool(r["fired"]), "query_id", lambda r: pw(r) and r.get("near_edit") is not None, ci=False), "h3", **kw)
    return tf, h3


def endpoints(log, recs, lb, tf, h3, cells, ctx):
    """H2 (registered primary endpoint of E1, on all edits and on LATER EDITS ONLY) and H3 on this log; descriptive per log (the Holm
    correction over backbones and baselines is the coordinator's)."""
    draws = bdraws(log)
    later = set(ctx["later_edits"])
    out = {"H2": {}, "H3": {}, "n_behavioural_draws": len(draws)}
    s = lb["sel"]
    fixed = set(lb["weights"] + lb["fixed"] + lb["lib"])

    def rep(lab):
        if lab not in tf:
            return None
        by = tf[lab]
        if lab in fixed:
            r = next(iter(by.values()))
            return {d: r for d in draws}
        return {d: by[d] for d in draws} if all(d in by for d in draws) else None

    A = rep("rescope")
    ex = lambda r: float(bool(r["exact"]))                                             # noqa: E731
    if A is not None:
        for role, lab in (("best_cosine_H2", s["cos_H2"]), ("best_cosine_H3", s["cos_H3"]), ("retr", s["retr_H2"]), ("lookup", "lookup"), ("cosine", "cosine"),
                          ("cosine_shared", "cosine_shared")):
            B = rep(lab)
            if B is None:
                out["H2"][role] = {"method": lab, "missing": True}
                continue
            e = {"method": lab, "role": role, "n_draws": len(draws), "calibrated_baseline": lab not in fixed}
            for sub in SUB:
                w0 = sub_where(sub, later)
                w = lambda r, w0=w0: r["kind"] == "edit" and r["metric"] in ("tgen", "igen") and w0(r)        # noqa: E731
                if not any(w(r) for r in A[draws[0]]):
                    continue
                ga, gb = MX.grid(A, ex, unit="edit", where=w), MX.grid(B, ex, unit="edit", where=w)
                d = ST.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100.0)
                bd = [float(100 * np.mean([ex(r) for r in A[k] if w(r)]) - 100 * np.mean([ex(r) for r in B[k] if w(r)])) for k in draws]
                e[sub] = {"diff": d["value"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p_two_sided"], "mean_rescope": d["mean_a"], "mean_other": d["mean_b"],
                          "n_units": len(ga["units"]), "by_draw": bd}
                idx = None if sub == "all" else (np.arange(min(100, ctx["T"])) if sub == "first100" else np.array(sorted(later), dtype=np.int64))
                e[sub]["gate_coverage_form_20_draws"] = gate_cov_diff(cells, "rescope", lab, idx=idx)
                rec(recs, "rescope", f"endpoint.H2.{sub}.rephrase_exact_diff", d["value"], "edits", ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_units=len(ga["units"]), variant=f"minus {lab} ({role})",
                    alpha=ALPHA, n=N_CAL, p_two_sided=d["p_two_sided"], n_draws=len(draws), by_draw=bd, gate_coverage_form_20_draws=e[sub]["gate_coverage_form_20_draws"])
            out["H2"][role] = e
    for pair, a, b in (("pair1", "rescope", s["cos_H3"]), ("pair2_x", "rescope_x", s["cos_H3_x"]), ("retr", "rescope", s["retr"])):
        e = {"a": a, "b": b, "n_draws": len(draws)}
        g = gate_cov_diff(cells, a, b, key="h3")
        e["gate_coverage_form_20_draws"] = g
        for nm, lab in (("a", a), ("b", b)):
            if lab in cells and "reg" in cells[lab] and "h3" in cells[lab]["reg"][0]:
                cs = cells[lab]["reg"]
                rch = [c["h3"] for c in cs if c["h3"]["reachable"]]
                e["gate_" + nm] = {"n_draws": len(cs), "n_unreachable": len(cs) - len(rch), "n_edits_at_cap_by_draw": [c["h3"].get("n_edits_at_cap") for c in cs],
                                   "coverage_mean": float(np.mean([c["h3"]["rephrase"]["mean"] if c["h3"]["reachable"] else 0.0 for c in cs])),
                                   "test_mean_fired": float(np.mean([h["test"]["mean_fired"] for h in rch])) if rch else None,
                                   "heldout_disjoint_fired_mean": float(np.mean([h["heldout_disjoint"]["fired"] for h in rch])) if rch else None}
        if a in h3 and b in h3:
            ra, rb = h3[a][0], h3[b][0]
            w = lambda r: r["kind"] == "edit" and r["metric"] in ("tgen", "igen")           # noqa: E731
            ga, gb = MX.grid(ra, ex, unit="edit", where=w), MX.grid(rb, ex, unit="edit", where=w)
            d = ST.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100.0)
            bd = [float(100 * np.mean([ex(r) for r in ra[k] if w(r)]) - 100 * np.mean([ex(r) for r in rb[k] if w(r)])) for k in draws]
            e.update(diff=d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], p=d["p_two_sided"], mean_a=d["mean_a"], mean_b=d["mean_b"], n_units=len(ga["units"]), by_draw=bd)
            rec(recs, a, "endpoint.H3.rephrase_exact_diff", d["value"], "h3", ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_units=len(ga["units"]), variant=f"minus {b} ({pair})", alpha=ALPHA, n=N_CAL,
                p_two_sided=d["p_two_sided"], n_draws=len(draws), by_draw=bd, gate_coverage_form_20_draws=g)
        else:
            e["missing_behaviour"] = True
        if g is not None:
            rec(recs, a, "endpoint.H3.gate_coverage_diff", g["value"], "gate_level", ci_lo=g["ci_lo"], ci_hi=g["ci_hi"], n_units=g["n_units"], variant=f"minus {b} ({pair})", alpha=ALPHA,
                n=N_CAL, p_two_sided=g["p_two_sided"], n_draws=g["n_draws"])
        out["H3"][pair] = e
    return out


def factor(log, recs):
    p = os.path.join(FAC, log + ".json")
    if not os.path.exists(p):
        MISSING.append({"log": log, "part": "factor level"})
        return
    f = jload(p)
    st = f.get("stored") or {}
    for m in ("rel", "tgen", "igen"):
        if "edits" in st:
            rec(recs, "deployed", f"stored.edit.{m}.exact", st["edits"][m]["exact"], "stored", note="the log's own stored metrics (testbed build; not B = 1 re-scored)")
            rec(recs, "deployed", f"stored.edit.{m}.token", st["edits"][m]["token"], "stored")
    if "probes" in st and "heldout_disjoint" in st["probes"]:
        hp = st["probes"]["heldout_disjoint"]
        rec(recs, "deployed", "stored.probe.D1", hp["D1_rate"], "stored", n_valid=hp["n_valid"], n_rows=hp["n_rows"], n_D1=hp["n_D1"])
        rec(recs, "deployed", "stored.probe.n_invalid", hp["n_rows"] - hp["n_valid"], "stored")
        if hp.get("n_pre_correct"):
            rec(recs, "deployed", "stored.probe.D2_rate", 100.0 * hp["n_D2"] / hp["n_pre_correct"], "stored", n_pre_correct=hp["n_pre_correct"], n_D2=hp["n_D2"])
    rb = f.get("rebase")
    if rb:
        for k in ("err_rebased", "err_logged", "err_rebased_first100", "err_logged_first100", "err_rebased_later", "err_logged_later"):
            if rb.get(k):
                for q in ("median", "max", "mean"):
                    rec(recs, "log", f"factor.rebase.{k}.{q}", rb[k][q], "factor_level", n_units=rb[k]["n"])
    else:
        rec(recs, "log", "factor.rebase.not_available", None, "factor_level", note=f.get("rebase_error") or f"targets found: {f.get('n_targets_found')}")
    mg = f["self_separation_margin"]
    for k in ("n_positive", "n_nonpositive", "n_ties", "min", "median", "n_below_0_1"):
        rec(recs, "log", f"factor.margin.{k}", mg[k], "factor_level", n_units=mg["n"])
    d = f["describe"]
    rec(recs, "log", "describe.n_probe_invalid", d["n_probe_rows"] - d["n_probe_valid"], "probes")
    rec(recs, "log", "describe.n_heldout_disjoint", d["pools_valid"]["heldout_disjoint"], "probes")
    rec(recs, "log", "describe.n_heldout_disjoint_near", d["near_scope_valid"]["heldout_disjoint"], "probes")
    for k, v in d["hard_negatives"].items():
        rec(recs, "log", f"describe.hn.{k}", v, "hard_negatives")


# ----------------------------------------------------------------------------- per log (cached by the state of the inputs)
def signature(log):
    h = hashlib.sha1()
    with open(os.path.abspath(__file__), "rb") as f:
        h.update(f.read())
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e3c.py"), "rb") as f:
        h.update(f.read())
    for pat in (os.path.join(GATE, log, "*.json"), os.path.join(BEH, log, "*.json"), os.path.join(FAC, log + ".json")):
        for p in sorted(glob.glob(pat)):
            s = os.stat(p)
            h.update(f"{os.path.basename(p)}:{s.st_mtime_ns}:{s.st_size};".encode())
    return h.hexdigest()


def analyze_log(log, use_cache=True):
    global MISSING
    cp = os.path.join(OUT, "cache", log + ".json")
    sig = signature(log)
    if use_cache and os.path.exists(cp):
        c = jload(cp)
        if c.get("signature") == sig:
            MISSING.extend(c["missing"])
            return c
    keep, MISSING = MISSING, []
    recs = []
    L = fx.logs.open(log)
    lb = e3c.labels(L)
    lb = {k: (list(v) if isinstance(v, tuple) else v) for k, v in lb.items()}
    for k in ("cal", "gl", "fixed", "weights", "lib", "h3"):
        lb[k] = tuple(lb[k])
    draws, cells = gate_level(log, recs)
    ep, tf = None, {}
    cpx = os.path.join(GATE, log, "_context.json")
    if os.path.exists(cpx):
        ctx = jload(cpx)
        tf, h3 = behaviour(log, recs, lb, ctx, cells)
        ep = endpoints(log, recs, lb, tf, h3, cells, ctx)
    factor(log, recs)
    c = {"signature": sig, "log": log, "records": recs, "gate_draws": draws, "endpoints": ep, "missing": MISSING, "labels": {k: list(v) for k, v in lb.items() if isinstance(v, tuple)},
         "selection": lb["sel"], "have_tf": sorted(tf)}
    jdump(c, cp)
    MISSING = keep + MISSING
    return c


# ----------------------------------------------------------------------------- series, sanity, paper numbers
def idx_of(recs):
    return {(r["method"], r["metric"], r.get("variant", "")): r for r in recs}


def series(per):
    """Scale series (per family and method: one list per quantity over T) and retention series (E9 c)."""
    scale = {"what": "per family and method: lists over the logs of the family (increasing T); registered read-out and the second read-out of R8 (qdisjoint); values in percent; "
                     "*_ci = crossed-bootstrap interval; gate-level quantities are means over the 20 calibration draws", "qdisjoint_note": QD_NOTE, "families": {}}
    ret = {"what": "E9 (c) retention: Rel / T-Gen / I-Gen exact (percent, B = 1 teacher forcing) of the FIRST 100 EDITS of the stream as T grows; deployed log, thresholds-only repair "
                   "(Rescope-WHERE with log-only thresholds) and the default method (calibration draws as listed)", "families": {}}
    B = {"rel_exact": "edit.all.rel.exact", "tgen_exact": "edit.all.tgen.exact", "igen_exact": "edit.all.igen.exact", "rel_exact_first100": "edit.first100.rel.exact",
         "tgen_exact_first100": "edit.first100.tgen.exact", "igen_exact_first100": "edit.first100.igen.exact", "rel_exact_later": "edit.later.rel.exact",
         "tgen_exact_later": "edit.later.tgen.exact", "igen_exact_later": "edit.later.igen.exact", "hd_fired": "probe.fired", "hd_D1": "probe.D1", "hd_near_fired": "probe.near.fired",
         "hd_near_D1": "probe.near.D1", "near_scope_D1": "hn.all.D1", "near_scope_fired": "hn.all.fired"}
    G = {"gate_cov_tgen": "gate.cov.tgen", "gate_cov_igen": "gate.cov.igen", "gate_hd_fired": "gate.fire.hd", "gate_hd_fired_literal": "gate.fire.hd.literal", "gate_hd_near_fired": "gate.fire.hd.near",
         "gate_BENT_far_fired": "gate.BEN-T.far.fired", "gate_MMET_fired": "gate.MME-T.fired", "gate_near_scope_fired": "gate.hn.all.fired", "gate_n_capped": "gate.n_capped"}
    Q = {"rel_exact": "qd.edit.all.rel.exact", "tgen_exact": "qd.edit.all.tgen.exact", "igen_exact": "qd.edit.all.igen.exact", "hd_fired": "qd.probe.fired", "hd_D1": "qd.probe.D1",
         "near_scope_D1": "qd.hn.all.D1", "near_scope_fired": "qd.hn.all.fired"}
    QF = {"hd_fired": "probe.qdrows.fired", "hd_D1": "probe.qdrows.D1"}             # rows without calibration: the same decisions on the filtered rows
    QG = {"gate_cov_tgen": "gate.cov.tgen", "gate_cov_igen": "gate.cov.igen", "gate_hd_fired": "gate.fire.hd.qd", "gate_BENT_fired": "gate.BEN-T.qd.fired", "gate_near_scope_fired": "gate.hn.all.fired",
          "gate_n_capped": "gate.n_capped"}
    for fam, logs in FAMILIES.items():
        logs = [g for g in logs if g in per]
        if not logs:
            continue
        ix = {g: idx_of(per[g]["records"]) for g in logs}
        methods = order({r["method"] for g in logs for r in per[g]["records"] if r["method"] != "log"})
        fs = {"logs": logs, "T": [T_of(g) for g in logs], "behavioural_draws": [list(bdraws(g)) for g in logs], "methods": {}}
        for m in methods:
            def col(metric, var="", field="value"):
                return [(ix[g].get((m, metric, var)) or {}).get(field) for g in logs]
            e = {"registered": {}, "qdisjoint": {}}
            for k, metric in B.items():
                e["registered"][k] = col(metric)
                e["registered"][k + "_ci"] = [[a, b] for a, b in zip(col(metric, field="ci_lo"), col(metric, field="ci_hi"))]
            for k, metric in G.items():
                e["registered"][k] = col(metric)
            for k, metric in Q.items():
                v = col(metric, "qdisjoint")
                if all(x is None for x in v) and k in QF:
                    v = col(QF[k])
                elif all(x is None for x in v) and k.endswith("_exact"):
                    v = col(B[k])                                       # no calibration: the same rows
                elif all(x is None for x in v) and k.startswith("near_scope"):
                    v = col(B[k])
                e["qdisjoint"][k] = v
            for k, metric in QG.items():
                v = col(metric, "qdisjoint")
                if all(x is None for x in v) and k == "gate_hd_fired":
                    v = col("gate.fire.hd.qd")
                elif all(x is None for x in v) and k == "gate_BENT_fired":
                    v = col("gate.BEN-T.qd.fired")
                elif all(x is None for x in v):
                    v = col(metric)
                e["qdisjoint"][k] = v
            fs["methods"][m] = e
        scale["families"][fam] = fs
        rf = {"logs": logs, "T": [T_of(g) for g in logs], "behavioural_draws": [list(bdraws(g)) for g in logs], "methods": {}}
        for m in ("deployed", "where_logonly", "rescope"):
            e = {}
            for k in ("rel", "tgen", "igen"):
                rr = [ix[g].get((m, f"edit.first100.{k}.exact", "")) for g in logs]
                e[k + "_exact"] = [None if r is None else r["value"] for r in rr]
                e[k + "_exact_ci"] = [None if r is None else [r.get("ci_lo"), r.get("ci_hi")] for r in rr]
                e[k + "_exact_by_draw"] = [None if r is None else r.get("by_draw") for r in rr]
            e["skipped_for_cost"] = [(m, "skipped_for_cost", "") in ix[g] for g in logs]
            rf["methods"][m] = e
        ret["families"][fam] = rf
    jdump(scale, os.path.join(OUT, "scale_series.json"))
    jdump(ret, os.path.join(OUT, "retention_series.json"))


def sanity(per):
    """The development log against the sanity run e0 (same cells: M1 - M6, deployed; gate level 20 draws, behaviour draws 0 - 2)."""
    p = FINAL + "/e0_sanity/out/results.json"
    if DEV not in per or not os.path.exists(p):
        return None
    e0 = jload(p)["logs"][DEV]
    MAP = {"M1": "rescope", "M2": "rescope_shared", "M3": "cosine", "M4": "cosine_shared", "M5": "cosine_default", "M6": "exact", "deployed": "deployed"}
    ix = idx_of(per[DEV]["records"])
    G = per[DEV]["gate_draws"]
    cells, worst = [], 0.0
    for m0, lab in MAP.items():
        if m0 in e0["gate"]["methods"] and lab in G:
            g = e0["gate"]["methods"][m0]["tf"]
            for path, mine in ((("probe", "heldout_disjoint", "fired"), "fire.hd"), (("edit", "tgen", "own_top"), "cov.tgen"), (("edit", "igen", "own_top"), "cov.igen"),
                               (("hn", "HN-I", "fired"), "hn.HN-I.fired"), (("hn", "HN-Q", "fired"), "hn.HN-Q.fired"), (("hn", "HN-X", "fired"), "hn.HN-X.fired")):
                v = g
                for k in path:
                    v = v[k]
                theirs = v["values"] if isinstance(v, dict) else [v]
                ours = G[lab]["reg"][mine]
                d = float(np.max(np.abs(np.array(theirs, dtype=float) - np.array(ours, dtype=float))))
                cells.append({"level": "gate", "method": lab, "e0": m0, "metric": mine, "n_draws": len(theirs), "max_abs_diff": d})
                worst = max(worst, d)
        b = e0["behaviour"]["methods"].get(m0)
        if b is None:
            continue
        per_d = {None: b["single"]} if "single" in b else {int(k): v for k, v in b["draws"].items()}
        for path, metric in ((("edit", "rel", "exact"), "edit.all.rel.exact"), (("edit", "tgen", "exact"), "edit.all.tgen.exact"), (("edit", "igen", "exact"), "edit.all.igen.exact"),
                             (("probe", "heldout_disjoint", "D1"), "probe.D1"), (("hn", "HN-I", "D1"), "hn.HN-I.D1"), (("hn", "HN-Q", "D1"), "hn.HN-Q.D1"), (("hn", "HN-X", "D1"), "hn.HN-X.D1")):
            r = ix.get((lab, metric, ""))
            if r is None or r.get("by_draw") is None:
                continue
            th, ou = [], []
            for i, (d, v) in enumerate(sorted(per_d.items(), key=lambda kv: -1 if kv[0] is None else kv[0])):
                for k in path:
                    v = v[k]
                if i < len(r["by_draw"]):
                    th.append(float(v))
                    ou.append(float(r["by_draw"][i]))
            if th:
                d = float(np.max(np.abs(np.array(th) - np.array(ou))))
                cells.append({"level": "behaviour", "method": lab, "e0": m0, "metric": metric, "n_draws": len(th), "max_abs_diff": d, "e0": th, "e3": ou})
                worst = max(worst, d)
    out = {"what": sanity.__doc__, "log": DEV, "n_cells": len(cells), "n_cells_differ_above_1e-9": sum(c["max_abs_diff"] > 1e-9 for c in cells), "largest_abs_diff": worst,
           "differing": [c for c in cells if c["max_abs_diff"] > 1e-9], "cells": cells}
    jdump(out, os.path.join(OUT, "sanity_compare.json"))
    return out


ROMAN = {100: "C", 200: "CC", 500: "D", 1000: "M", 2000: "MM"}
MCODE = {"deployed": "Deployed", "where_logonly": "Where", "rescope": "Rescope", "rescope_shared": "RescopeShared", "rescope_null": "RescopeNull", "rescope_x": "RescopeX",
         "rescope_nullW": "RescopeNullW", "rescope_pac": "RescopePac", "cosine": "Cosine", "cosine_shared": "CosineShared", "white_1_shared": "WhiteOneShared",
         "white_0.1_shared": "WhiteTenthShared", "white_1_x": "WhiteOneX", "retr_sum": "RetrSum", "lookup": "Lookup", "cosine_default": "CosineDefault",
         "cosine_published": "CosinePublished", "exact": "Exact"}
PAPER = (("RelExact", "edit.all.rel.exact", "Rel exact, all T edits (%)"), ("TGenExact", "edit.all.tgen.exact", "T-Gen exact, all T edits (%)"), ("IGenExact", "edit.all.igen.exact", "I-Gen exact, all T edits (%)"),
         ("RelExactLater", "edit.later.rel.exact", "Rel exact, later edits only (stream position >= 500) (%)"), ("TGenExactLater", "edit.later.tgen.exact", "T-Gen exact, later edits only (%)"),
         ("IGenExactLater", "edit.later.igen.exact", "I-Gen exact, later edits only (%)"), ("RelExactFirst", "edit.first100.rel.exact", "Rel exact, first 100 edits of the stream (%)"),
         ("TGenExactFirst", "edit.first100.tgen.exact", "T-Gen exact, first 100 edits (%)"), ("IGenExactFirst", "edit.first100.igen.exact", "I-Gen exact, first 100 edits (%)"),
         ("HdFired", "probe.fired", "heldout_disjoint rows on which an edit is released, behavioural draws (%)"), ("HdDone", "probe.D1", "D1 on heldout_disjoint (%)"),
         ("HdNearFired", "probe.near.fired", "near-scope held-out probes on which an edit is released (%)"), ("HdNearDone", "probe.near.D1", "D1 on near-scope held-out probes (%)"),
         ("HnDone", "hn.all.D1", "D1 on the near-scope negatives of the log's edits, all kinds (%)"), ("HnFired", "hn.all.fired", "near-scope negatives on which an edit is released (%)"),
         ("DtwoRate", "d2.rate", "D2 as a rate of the pre-correct heldout_disjoint rows, draw 0 (%)"),
         ("GateHdFired", "gate.fire.hd", "gate level, mean over 20 calibration draws: firing on heldout_disjoint (%)"),
         ("GateHdFiredLiteral", "gate.fire.hd.literal", "gate level, mean over 20 draws: firing on heldout_disjoint of the literal rule without guards (%)"),
         ("GateHdNearFired", "gate.fire.hd.near", "gate level, mean over 20 draws: firing on near-scope held-out probes (%)"),
         ("GateCovTGen", "gate.cov.tgen", "gate level, mean over 20 draws: coverage of text rephrases (%)"), ("GateCovIGen", "gate.cov.igen", "gate level, mean over 20 draws: coverage of image rephrases (%)"),
         ("GateCapped", "gate.n_capped", "gate level, mean over 20 draws: number of capped edits"), ("GateBenTFar", "gate.BEN-T.far.fired", "gate level, mean over 20 draws: firing on BEN-T rows not related to the family, prompt-only keys (%)"),
         ("GateMmeT", "gate.MME-T.fired", "gate level, mean over 20 draws: firing on MME-T (%)"), ("GateHnFired", "gate.hn.all.fired", "gate level, mean over 20 draws: firing on near-scope negatives, all kinds (%)"),
         ("HthreeCov", "gate.h3.cov", "H3, gate level, mean over 20 draws: rephrase coverage on the odd edits at the raised scale (%)"),
         ("HthreeExact", "h3.rephrase.exact", "H3, behaviour: (T-Gen + I-Gen) / 2 exact on the odd edits at the raised scale (%)"))


def paper(per):
    nums, doc = {}, {}
    for log, c in per.items():
        fam = "Ae" if "-ae-" in log else "Me"
        base = "EthreeQ" + fam + ROMAN[T_of(log)]
        ix = idx_of(c["records"])
        for m, code in MCODE.items():
            for nm, metric, text in PAPER:
                r = ix.get((m, metric, ""))
                if r is None or r.get("value") is None:
                    continue
                k = base + code + nm
                nums[k] = round(float(r["value"]), 2)
                doc[k] = f"{log}, {m}: {text}" + (f"; behavioural draws {r['n_draws']}" if r.get("n_draws") and not metric.startswith("gate.") else "")
        for nm, metric, text in (("RebaseErrMedian", "factor.rebase.err_rebased.median", "median relative error of the rebased values against the cached targets"),
                                 ("LoggedErrMedian", "factor.rebase.err_logged.median", "median relative error of the logged residuals (no rebasing) against the cached targets"),
                                 ("MarginPositive", "factor.margin.n_positive", "number of edits with a positive self-separation margin"),
                                 ("MarginMin", "factor.margin.min", "smallest self-separation margin")):
            r = ix.get(("log", metric, ""))
            if r is not None and r.get("value") is not None:
                nums[base + nm] = round(float(r["value"]), 4)
                doc[base + nm] = f"{log}: {text}"
    return nums, doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    R = fx.io.Results("E3", OUT)
    per = {}
    for log in LOGS:
        if not os.path.isdir(os.path.join(GATE, log)):
            continue
        c = analyze_log(log, not a.no_cache)
        per[log] = c
        for r in c["records"]:
            r = dict(r)
            m, metric, value = r.pop("method"), r.pop("metric"), r.pop("value")
            R.add(log, m, metric, value, **r)
        print(f"[{time.time() - t0:6.0f}s] {log}: {len(c['records'])} records, tf rows of {len(c['have_tf'])} methods, missing {len(c['missing'])}", flush=True)
    R.save(logs=list(per), alpha=ALPHA, n=N_CAL, n_gate_draws=N_DRAWS, behavioural_draws={g: list(bdraws(g)) for g in per}, h3_unreachable_rule=H3_UNREACHABLE_RULE, qdisjoint=QD_NOTE,
           selection={g: per[g]["selection"] for g in per}, statistics=__doc__)
    jdump({"what": "every gate-level number of every calibration draw: [log][method][reg|qd][metric] = list over the draws 0..19 (a method without calibration: one value)",
           "draws": {g: per[g]["gate_draws"] for g in per}}, os.path.join(OUT, "gate_draws.json"))
    jdump({g: per[g]["endpoints"] for g in per}, os.path.join(OUT, "endpoints.json"))
    jdump({"n_missing": len(MISSING), "missing": MISSING}, os.path.join(OUT, "missing.json"))
    series(per)
    s = sanity(per)
    nums, doc = paper(per)
    p = os.path.join(OUT, "paper_numbers.json")
    jdump(dict(sorted(nums.items())), p)
    jdump(dict(sorted(doc.items())), os.path.join(OUT, "paper_numbers_doc.json"))
    print(f"[{time.time() - t0:6.0f}s] records {len(R.records)}, paper numbers {len(nums)}, missing cells {len(MISSING)}, sanity {None if s is None else (s['n_cells'], s['n_cells_differ_above_1e-9'], s['largest_abs_diff'])}", flush=True)


if __name__ == "__main__":
    main()
