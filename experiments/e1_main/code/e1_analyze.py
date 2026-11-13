"""E1: statistics.  Reads the gate-level cells (out/gate), the behavioural row files (out/rows) and cell records (out/behave) and
writes out/results.json (schema of fx.io), out/gate_draws.json (every gate-level number of every calibration draw, compact),
out/endpoints.json, out/decision.json, out/paper_numbers.json + paper_numbers_doc.json.  No number is typed by hand.

    python e1_analyze.py           (CPU; re-runnable after every block; missing cells are listed in out/missing.json)
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import glob
import os
import re
import time

from e1c import *  # noqa: F401,F403
import e1c

MX, ST = fx.metrics, fx.stats
ALPHA = 0.05
BDRAWS = (0, 1, 2)
MISSING = []
CORE = ("fire.hd", "fire.hd.literal", "fire.strict", "cov.tgen", "cov.igen", "h2.cov", "hn.HN-I.fired", "hn.HN-Q.fired", "hn.HN-X.fired", "hn.mean.fired",
        "n_capped", "BEN-T.fired", "MME-T.fired", "h3.cov", "h3.test.mean", "h3.n_at_cap")


# ----------------------------------------------------------------------------- gate level
def g_metrics(c):
    """Flat {metric: number} of one gate-level cell."""
    m = {}
    if c.get("dense") or "st" not in c:                                # dense gates (router, z-score): firing only
        for k, v in c["split"].items():
            pool, mod = k.split(".")
            m["fire." + ("hd" if pool == "disjoint" else "strict") + ("" if mod == "all" else "." + mod)] = v["fired"]
        for k in ("rel", "tgen", "igen"):
            m[f"fired.{k}"] = c["edit_fired"][k]
        for k in HN:
            m[f"hn.{k}.fired"] = c["hn_fired"][k]
        m["hn.mean.fired"] = float(np.mean([c["hn_fired"][k] for k in HN]))
        for p in ("BEN-T", "MME-T"):
            if p in c and "fired" in c[p]:
                m[p + ".fired"] = c[p]["fired"]
        return m
    st = c["st"]
    for k, v in c["split"].items():
        pool, mod = k.split(".")
        nm = "fire." + ("hd" if pool == "disjoint" else "strict") + ("" if mod == "all" else "." + mod)
        m[nm] = v["fired"]
        if c.get("literal"):
            m[nm + ".literal"] = c["literal"][k]["fired"]
    m["fire.hd.near"], m["fire.hd.far"] = st["probe"]["heldout_disjoint_near"]["fired"], st["probe"]["heldout_disjoint_far"]["fired"]
    for k in ("tgen", "igen", "rel"):
        e = st["edit"][k]
        m[f"cov.{k}"], m[f"cand.{k}"], m[f"other.{k}"], m[f"fired.{k}"] = e["own_top"], e["own_cand"], e["other_top"], e["fired"]
    m["h2.cov"] = c["h2_cov"]
    for k in HN:
        m[f"hn.{k}.fired"], m[f"hn.{k}.own"] = st["hn"][k]["fired"], st["hn"][k]["own_top"]
    m["hn.mean.fired"] = float(np.mean([st["hn"][k]["fired"] for k in HN]))
    m["lineage.hd.width_mean"], m["lineage.hd.multi"] = st["probe"]["lineage"]["width_mean"], st["probe"]["lineage"]["multi"]
    m["lineage.hn.width_mean"], m["lineage.hn.multi"] = st["hn"]["all"]["lineage"]["width_mean"], st["hn"]["all"]["lineage"]["multi"]
    m["lineage.reph.width_mean"] = np.mean([x for x in (st["edit"][k]["lineage"]["width_mean"] for k in ("tgen", "igen")) if x is not None]) if any(
        st["edit"][k]["lineage"]["width_mean"] is not None for k in ("tgen", "igen")) else None
    cal = c.get("cal") or {}
    if c.get("alpha") is not None:
        m["n_capped"] = c["n_capped"]
        m["own.silenced_before_cap"] = cal.get("n_edits_silenced_by_literal_rule", cal.get("n_own_requests_silenced_by_literal_rule"))
        m["eta"] = cal.get("eta") if not cal.get("eta_infinite") else None
    op = c.get("own_prompts")
    m["own.not_candidate_after_cap"] = op["n_not_candidate"] if op else None
    m["own.not_released"] = op["n_not_released"] if op else round((100.0 - st["edit"]["rel"]["own_top"]) * st["edit"]["rel"]["n"] / 100.0)
    for p in ("BEN-T", "MME-T"):
        if p in c and "fired" in c[p]:
            m[p + ".fired"] = c[p]["fired"]
            for s in ("image", "text"):
                if s in c[p]:
                    m[f"{p}.{s}.fired"] = c[p][s]["fired"]
    m["top1.n_differs"] = (st.get("top1") or {}).get("all", {}).get("n_differs")
    m["prompt.n_rows_differ"] = sum(c["n_rows_differ_tf_vs_prompt"].values())
    if "st_prompt" in c:
        sp = c["st_prompt"]
        m["fire.hd@prompt"], m["cov.tgen@prompt"], m["cov.igen@prompt"] = sp["probe"]["heldout_disjoint"]["fired"], sp["edit"]["tgen"]["own_top"], sp["edit"]["igen"]["own_top"]
        for k in HN:
            m[f"hn.{k}.fired@prompt"] = sp["hn"][k]["fired"]
    for key, pre in (("h3", "h3"), ("h3_unseen", "h3u")):
        h = c.get(key)
        if h is None:
            continue
        m[pre + ".reachable"] = 1.0 if h["reachable"] else 0.0
        if h["reachable"]:
            m[pre + ".cov"], m[pre + ".tgen"], m[pre + ".igen"] = h["rephrase"]["mean"], h["rephrase"]["tgen"], h["rephrase"]["igen"]
            m[pre + ".delta"], m[pre + ".n_at_cap"] = h["delta"], h["n_edits_at_cap"]
            for k in HN:
                if k in h["test"]:
                    m[f"{pre}.test.{k}.fired"], m[f"{pre}.test.{k}.own"], m[f"{pre}.test.{k}.n"] = h["test"][k]["fired"], h["test"][k]["own_top"], h["test"][k]["n"]
            m[pre + ".test.mean"] = h["test"]["mean_fired"]
            m[pre + ".hd.fired"] = h["heldout_disjoint"]["fired"]
            m[pre + ".fit.mean_at_0"] = h["fit"]["at_0"]["mean"]
        else:
            m[pre + ".cov"] = 0.0                                       # e1c.H3_UNREACHABLE_RULE
    if "near_scope" in c:
        for form, pre in (("all", "ns"), ("unseen_foreign", "nsu")):
            for k in HN:
                if k in c["near_scope"][form]:
                    m[f"{pre}.{k}.fired"], m[f"{pre}.{k}.n"] = c["near_scope"][form][k]["fired"], c["near_scope"][form][k]["n"]
    return m


def gate_level(R):
    """-> draws[log][label][alpha] = {metric: [value per draw]}, cells[log][label][alpha] = [cell per draw]."""
    draws, cells = {}, {}
    for log in DEV + WP:
        for p in sorted(glob.glob(os.path.join(GATE, log, "*.json"))):
            if os.path.basename(p).startswith("_"):                     # job records
                continue
            b = jload(p)
            lab, alpha = b["label"], b["alpha"]
            cs = [b["cells"][k] for k in sorted(b["cells"], key=int)]
            if "not_defined" in cs[0]:
                R.add(log, lab, "not_defined", None, alpha=alpha, pool="gate_level", note=cs[0]["not_defined"])
                continue
            if b.get("dense"):
                for c in cs:
                    c["dense"] = True
            ms = [g_metrics(c) for c in cs]
            keys = list(dict.fromkeys(k for m in ms for k in m))
            d = {k: [m.get(k) for m in ms] for k in keys}
            a = "none" if alpha is None else afmt(alpha)
            draws.setdefault(log, {}).setdefault(lab, {})[a] = d
            cells.setdefault(log, {}).setdefault(lab, {})[a] = cs
            kw = dict(alpha=alpha, n=None if alpha is None else 200, pool="gate_level", variant=cs[0].get("variant", ""))
            for k, v in d.items():
                vals = [x for x in v if x is not None]
                if not vals:
                    continue
                if alpha is None:
                    R.add(log, lab, "gate." + k, vals[0], **kw)
                elif k in CORE and abs(alpha - ALPHA) < 1e-12:
                    R.add_draws(log, lab, "gate." + k, {i: x for i, x in enumerate(v) if x is not None}, **kw)
                else:
                    s = ST.over_draws(vals)
                    R.add(log, lab, "gate." + k + "@mean", s["mean"], ci_lo=s["p05"], ci_hi=s["p95"], n_units=s["n_draws"], **kw)
    jdump({"what": "every gate-level number of every calibration draw: [log][method][alpha][metric] = list over the draws 0..19 (a method without calibration: one value)",
           "draws": draws}, os.path.join(OUT, "gate_draws.json"))
    return draws, cells


# ----------------------------------------------------------------------------- behaviour
def rows_of(log, label, d, part):
    p = os.path.join(OUT, "rows", f"{log}__{label}__d{'x' if d is None else d}__{part}.jsonl.gz")
    c = os.path.join(BEH, log, f"{label}__d{'x' if d is None else d}__{part}.json")
    if part == "tf" and label == "dual":
        c = os.path.join(BEH, log, "dual__dx__gen.json")
    if not (os.path.exists(p) and os.path.exists(c)):
        MISSING.append({"log": log, "label": label, "draw": d, "part": part})
        return None
    return fx.io.load_rows(p)


def split_tf(rows):
    e = [r for r in rows if r["kind"] == "edit"]
    hd = [r for r in rows if r["kind"] == "probe" and r["disjoint"]]
    hs = [r for r in rows if r["kind"] == "probe" and r["strict"]]
    h = [r for r in rows if r["kind"] == "hn"]
    return e, hd, hs, h


def cb(rows_by_draw, value, unit, where=None):
    g = MX.grid(rows_by_draw, value, unit=unit, where=where)
    return ST.crossed_bootstrap(g["num"], g["den"], scale=100.0)


def cbd(a_by_draw, b_by_draw, value, unit, where=None):
    ga, gb = MX.grid(a_by_draw, value, unit=unit, where=where), MX.grid(b_by_draw, value, unit=unit, where=where)
    if ga["units"] != gb["units"]:
        raise ValueError("paired comparison on different units")
    return ST.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"], scale=100.0)


SUBSETS = (("same_category", "samecat"), ("related_flag", "related"), ("new", "new"), ("y0_deg", "y0deg"))


def behaviour(R, log, labels_fixed, labels_cal):
    """-> tf[label] = {draw: rows} (fixed methods: the same rows under draws 0 - 2, flagged), meta."""
    tf, fixed = {}, set()
    dep = rows_of(log, "deployed", None, "tf")
    dep_e = split_tf(dep)[0] if dep else None
    for lab in labels_fixed + labels_cal:
        is_fixed = lab in labels_fixed
        by = {}
        for d in ([None] if is_fixed else BDRAWS):
            rows = rows_of(log, lab, d, "tf")
            if rows is not None:
                by[d] = rows
        if not by:
            continue
        if is_fixed:
            fixed.add(lab)
        tf[lab] = by
        for d, rows in by.items():
            e, hd, hs, h = split_tf(rows)
            kw = dict(alpha=None if is_fixed else ALPHA, n=None if is_fixed else 200, draw=d)
            R.add_nested(log, lab, MX.edit_stats(e, ci=True), prefix="edit.", pool="edits", **kw)
            R.add_nested(log, lab, MX.probe_stats(hd, ci=True), prefix="probe.", pool="heldout_disjoint", **kw)
            R.add_nested(log, lab, MX.probe_stats(hs, ci=True), prefix="probe.", pool="heldout_strict", **kw)
            R.add_nested(log, lab, MX.hn_stats(h, ci=True), prefix="hn.", pool="hard_negatives", **kw)
            if dep_e is not None:
                R.add_nested(log, lab, MX.kept_gained(e, dep_e, ci=True), prefix="kg.", pool="edits", **kw)
            for k in HN:                                                # subsets that the cache flags
                for fld, nm in SUBSETS:
                    for val in (True, False):
                        sub = [r for r in h if r["hn_type"] == k and bool(r.get(fld)) == val]
                        if sub:
                            pre = f"hnsub.{k}.{nm}={int(val)}."
                            R.add(log, lab, pre + "n", len(sub), pool="hard_negatives", **kw)
                            R.add(log, lab, pre + "fired", MX.pct(r.get("fired") for r in sub), pool="hard_negatives", **kw)
                            R.add(log, lab, pre + "D1", MX.pct(r["D1"] for r in sub), pool="hard_negatives", **kw)
        # summary over the calibration draws: crossed bootstrap (R10); fixed rows: the single-draw records above carry their interval
        if not is_fixed and len(by) >= 2:
            kw = dict(alpha=ALPHA, n=200, draw=None, n_draws=len(by), draws=sorted(by))
            parts = {d: split_tf(rows) for d, rows in by.items()}
            E, HD, HS, H = ({d: p[i] for d, p in parts.items()} for i in range(4))
            for m in ("rel", "tgen", "igen"):
                R.add_stat(log, lab, f"edit.{m}.exact@x", cb(E, lambda r: float(bool(r["exact"])), "edit", lambda r, m=m: r["metric"] == m), pool="edits", **kw)
                R.add_stat(log, lab, f"edit.{m}.token@x", cb(E, "token_acc", "edit", lambda r, m=m: r["metric"] == m), pool="edits", **kw)
                R.add_stat(log, lab, f"edit.{m}.own_top@x", cb(E, lambda r: float(bool(r.get("own_released_last"))), "edit", lambda r, m=m: r["metric"] == m), pool="edits", **kw)
            R.add_stat(log, lab, "edit.rephrase.exact@x", cb(E, lambda r: float(bool(r["exact"])), "edit", lambda r: r["metric"] in ("tgen", "igen")), pool="edits", **kw)
            for pool, P in (("heldout_disjoint", HD), ("heldout_strict", HS)):
                gate = next(iter(P.values()))[0].get("fired") is not None
                if gate:
                    R.add_stat(log, lab, "probe.fired@x", cb(P, lambda r: float(bool(r["fired"])), "query_id"), pool=pool, **kw)
                R.add_stat(log, lab, "probe.D1@x", cb(P, lambda r: float(bool(r["D1"])), "query_id"), pool=pool, **kw)
                R.add_stat(log, lab, "probe.T-Loc*@x", cb(P, "tok_loc", "query_id", lambda r: r["ptype"] == "text"), pool=pool, **kw)
                R.add_stat(log, lab, "probe.I-Loc*@x", cb(P, "tok_loc", "query_id", lambda r: r["ptype"] == "image"), pool=pool, **kw)
            for k in HN + ("all",):
                w = (lambda r: True) if k == "all" else (lambda r, k=k: r["hn_type"] == k)
                if next(iter(H.values()))[0].get("fired") is not None:
                    R.add_stat(log, lab, f"hn.{k}.fired@x", cb(H, lambda r: float(bool(r["fired"])), "edit", w), pool="hard_negatives", **kw)
                R.add_stat(log, lab, f"hn.{k}.D1@x", cb(H, lambda r: float(bool(r["D1"])), "edit", w), pool="hard_negatives", **kw)
            if dep_e is not None:
                KG = {d: MX.kept_gained_rows(E[d], dep_e) for d in E}
                for m in ("rel", "tgen", "igen"):
                    for f in ("kept", "gained", "lost"):
                        R.add_stat(log, lab, f"kg.{m}.{f}@x", cb(KG, lambda r, f=f: float(r[f]), "edit", lambda r, m=m: r["metric"] == m), pool="edits", **kw)
    return tf, fixed


def prompt_position(R, log, tf, fixed):
    """LLaVA: the gate at the decode's position on the sequences whose two positions differ; records only where a row differs."""
    out = {}
    for lab, by in tf.items():
        for d, rows in by.items():
            alt = rows_of(log, lab, d, "tfp")
            if alt is None:
                continue
            a = {r["csid"]: r for r in alt}
            base = {r["csid"]: r for r in rows}
            diff = [s for s in a if any(a[s].get(k) != base[s].get(k) for k in ("fired", "top_last", "exact", "D1"))]
            out[(lab, d)] = {"n_shift_rows": len(a), "n_rows_differ": len(diff)}
            kw = dict(alpha=None if lab in fixed else ALPHA, n=None if lab in fixed else 200, draw=d)
            R.add(log, lab, "prompt_position.n_shift_rows", len(a), pool="lp_shift", **kw)
            R.add(log, lab, "prompt_position.n_rows_differ", len(diff), pool="lp_shift", **kw)
            if diff:
                rows2 = [a.get(r["csid"], r) for r in rows]
                e, hd, hs, h = split_tf(rows2)
                es, ps, hs_ = MX.edit_stats(e), MX.probe_stats(hd), MX.hn_stats(h)
                for m in ("rel", "tgen", "igen"):
                    R.add(log, lab, f"edit.{m}.exact@prompt", es[m]["exact"], pool="edits", **kw)
                R.add(log, lab, "probe.fired@prompt", ps["fired"], pool="heldout_disjoint", **kw)
                R.add(log, lab, "probe.D1@prompt", ps["D1"], pool="heldout_disjoint", **kw)
                for k in HN:
                    R.add(log, lab, f"hn.{k}.fired@prompt", hs_[k]["fired"], pool="hard_negatives", **kw)
                    R.add(log, lab, f"hn.{k}.D1@prompt", hs_[k]["D1"], pool="hard_negatives", **kw)
    return out


def generation(R, log, gen_rows):
    """Generation metrics (draw 0 / fixed rows; dual): edit prompts eq / contains; probes agree, D2 rate; hard negatives."""
    for lab, d in gen_rows:
        rows = rows_of(log, lab, d, "gen")
        if rows is None:
            continue
        kw = dict(alpha=None if d is None else ALPHA, n=None if d is None else 200, draw=d)
        ge = [r for r in rows if r["kind"] == "edit"]
        gp = [r for r in rows if r["kind"] == "probe"]
        gh = [r for r in rows if r["kind"] == "hn"]
        R.add_nested(log, lab, MX.gen_stats(ge), prefix="gen.", pool="edit_prompts", **kw)
        hd = [r for r in gp if r.get("disjoint") and r.get("y0_valid")]
        if hd and all("agree" in r for r in hd):
            R.add_stat(log, lab, "gen.probe.eq_pre", ST.bootstrap([float(bool(r["agree"])) for r in hd], [r["query_id"] for r in hd], scale=100.0), pool="heldout_disjoint", **kw)
        d2 = [r for r in gp if r.get("in_d2_set")]
        R.add_nested(log, lab, MX.d2_stats(d2, ci=True), prefix="d2.", pool="heldout_disjoint", **kw)
        hs = MX.hn_stats(None, gh, ci=True)
        R.add_nested(log, lab, hs, prefix="hngen.", pool="hard_negatives", **kw)
        for k in HN + ("all",):                                         # equals the pre-edit answer: a row without a decode IS the pre-edit model (nothing fired)
            sub = [r for r in gh if k == "all" or r["hn_type"] == k]
            eq = [bool(r["agree"]) if "line" in r else (True if r.get("fired") is False else None) for r in sub]
            if sub and all(x is not None for x in eq):
                R.add_stat(log, lab, f"hngen.{k}.eq_pre", ST.bootstrap([float(x) for x in eq], [r["edit"] for r in sub], scale=100.0), pool="hard_negatives", **kw)
            anyt = [bool(r.get("eq_any_edit")) if "line" in r else (False if r.get("fired") is False else None) for r in sub]
            if sub and all(x is not None for x in anyt):
                R.add_stat(log, lab, f"hngen.{k}.eq_any_edit_all_rows", ST.bootstrap([float(x) for x in anyt], [r["edit"] for r in sub], scale=100.0), pool="hard_negatives", **kw)
            for fld, nm in SUBSETS:
                for val in (True, False):
                    s2 = [r for r in sub if bool(r.get(fld)) == val]
                    wt = [r.get("wrong_target") for r in s2]
                    if k != "all" and s2 and all(x is not None for x in wt):
                        R.add(log, lab, f"hngensub.{k}.{nm}={int(val)}.wrong_target", MX.pct(wt), pool="hard_negatives", n_units=len(s2), **kw)


# ----------------------------------------------------------------------------- endpoints
def rep3(by, fixed):
    """Rows under the draws 0 - 2 (a method without calibration: its one row set under each draw)."""
    if fixed:
        r = next(iter(by.values()))
        return {d: r for d in BDRAWS}
    return {d: by[d] for d in BDRAWS if d in by}


def gate_cov_diff(cells, log, a, b, key="per_edit"):
    """Gate-level coverage form over the 20 draws: paired crossed bootstrap of (T-Gen + I-Gen) / 2 coverage, units = edits."""
    def mat(lab):
        cs = cells[log][lab]["0.05" if "0.05" in cells[log][lab] else "none"]
        if len(cs) == 1:
            cs = cs * N_DRAWS
        out = []
        for c in cs:
            if key == "per_edit":
                pe = c["per_edit"]
            else:
                h = c["h3"]
                pe = h["per_edit"] if h["reachable"] else None
            if pe is None:
                out.append(None)
            else:
                out.append(0.5 * (unflags(pe["tgen"]).astype(float) + unflags(pe["igen"]).astype(float)))
        n = next(len(x) for x in out if x is not None)
        return np.array([x if x is not None else np.zeros(n) for x in out])        # unreachable draw: coverage 0 (e1c.H3_UNREACHABLE_RULE)
    A, B = mat(a), mat(b)
    d = ST.crossed_bootstrap_diff(A, B, scale=100.0)
    return {k: d[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "mean_a", "mean_b")} | {"n_draws": int(A.shape[0]), "n_units": int(A.shape[1])}


def h3_rows(log, lab):
    """{draw: edit rows on the odd edits at the raised scale}; an unreachable draw -> rows with exact False (rule)."""
    out, info = {}, {}
    for d in BDRAWS:
        c = os.path.join(BEH, log, f"{lab}__d{d}__h3.json")
        if not os.path.exists(c):
            MISSING.append({"log": log, "label": lab, "draw": d, "part": "h3"})
            return None, None
        rec = jload(c)
        info[d] = rec
        if rec["reachable"]:
            out[d] = fx.io.load_rows(os.path.join(OUT, "rows", f"{log}__{lab}__d{d}__h3.jsonl.gz"))
        else:
            out[d] = [{"kind": "edit", "metric": m, "edit": e, "exact": False, "token_acc": 0.0, "own_released_last": False} for m in ("tgen", "igen") for e in rec["odd_edits"]]
    return out, info


def endpoints(R, log, tf, fixed, cells, sel):
    out = {"H2": {}, "H3": {}, "secondary": {}}
    if "rescope" not in tf or len(tf["rescope"]) < 3:
        return out
    A = rep3(tf["rescope"], False)
    reph = lambda r: r["kind"] == "edit" and r["metric"] in ("tgen", "igen")           # noqa: E731
    ex = lambda r: float(bool(r["exact"]))                                             # noqa: E731
    base = [("best_cosine_H2", sel["cos_H2"]), ("best_cosine_H3", sel["cos_H3"]), ("retr", sel["retr_H2"]), ("retr_H3", sel["retr_H3"]), ("lookup", "lookup"),
            ("cosine", "cosine"), ("cosine_shared", "cosine_shared"), ("ball", "ball"), ("router", "router"), ("dual", "dual")]
    seen = set()
    for role, lab in base:
        if lab in seen and role in ("best_cosine_H3", "retr_H3"):
            continue
        seen.add(lab)
        if lab not in tf or len(rep3(tf[lab], lab in fixed)) < 3:
            out["H2"][role] = {"method": lab, "missing": True}
            continue
        B = rep3(tf[lab], lab in fixed)
        d = cbd(A, B, ex, "edit", reph)
        e = {"method": lab, "role": role, "diff": d["value"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p_two_sided"], "mean_rescope": d["mean_a"], "mean_other": d["mean_b"],
             "n_draws": 3, "n_units": 100, "by_draw": [float(100 * np.mean([ex(r) for r in A[k] if reph(r)]) - 100 * np.mean([ex(r) for r in B[k] if reph(r)])) for k in BDRAWS],
             "calibrated_baseline": lab not in fixed}
        if lab in cells[log] and "per_edit" in next(iter(cells[log][lab].values()))[0]:
            e["gate_coverage_form_20_draws"] = gate_cov_diff(cells, log, "rescope", lab)
        dr = cbd(A, B, ex, "edit", lambda r: r["kind"] == "edit" and r["metric"] == "rel")
        e["rel_exact_diff"] = {k: dr[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided")}
        hn = lambda r: r["kind"] == "hn"                                                  # noqa: E731
        dh = cbd(A, B, lambda r: float(bool(r["D1"])), "edit", hn)
        e["near_scope_D1_diff"] = {k: dh[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "mean_a", "mean_b")}
        out["H2"][role] = e
        R.add(log, "rescope", "endpoint.H2.rephrase_exact_diff", d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_units=100, variant=f"minus {lab} ({role})", alpha=ALPHA, n=200,
              pool="edits", p_two_sided=d["p_two_sided"], n_draws=3)
        R.add(log, "rescope", "endpoint.rel_exact_diff", dr["value"], ci_lo=dr["ci_lo"], ci_hi=dr["ci_hi"], n_units=100, variant=f"minus {lab} ({role})", alpha=ALPHA, n=200,
              pool="edits", p_two_sided=dr["p_two_sided"], n_draws=3)
        R.add(log, "rescope", "endpoint.near_scope_D1_diff", dh["value"], ci_lo=dh["ci_lo"], ci_hi=dh["ci_hi"], variant=f"minus {lab} ({role})", alpha=ALPHA, n=200,
              pool="hard_negatives", p_two_sided=dh["p_two_sided"], n_draws=3)
    # secondary: D1 on heldout_disjoint against the deployed log
    if "deployed" in tf:
        B = rep3(tf["deployed"], True)
        d = cbd(A, B, lambda r: float(bool(r["D1"])), "query_id", lambda r: r["kind"] == "probe" and r["disjoint"])
        out["secondary"]["D1_heldout_disjoint_rescope_minus_deployed"] = {k: d[k] for k in ("value", "ci_lo", "ci_hi", "p_two_sided", "mean_a", "mean_b")} | {"n_draws": 3}
        R.add(log, "rescope", "endpoint.D1_diff", d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], variant="minus deployed", alpha=ALPHA, n=200, pool="heldout_disjoint",
              p_two_sided=d["p_two_sided"], n_draws=3)
    # H3: two pairs + retr in the first pair's form
    for pair, a, b in (("pair1", "rescope", sel["cos_H3"]), ("pair2_x", "rescope_x", sel["cos_H3_x"]), ("retr", "rescope", sel["retr_H3"])):
        ra, ia = h3_rows(log, a)
        rb, ib = h3_rows(log, b)
        e = {"a": a, "b": b}
        if a in cells[log] and b in cells[log]:
            e["gate_coverage_form_20_draws"] = gate_cov_diff(cells, log, a, b, key="h3")
            ga, gb = cells[log][a]["0.05"], cells[log][b]["0.05"]
            for nm, cs in (("a", ga), ("b", gb)):
                rch = [c["h3"] for c in cs if c["h3"]["reachable"]]
                e["gate_" + nm] = {"n_draws": len(cs), "n_unreachable": len(cs) - len(rch), "n_edits_at_cap_mean": float(np.mean([h["n_edits_at_cap"] for h in rch])) if rch else None,
                                   "n_edits_at_cap_by_draw": [c["h3"].get("n_edits_at_cap") for c in cs],
                                   "coverage_mean": float(np.mean([c["h3"]["rephrase"]["mean"] if c["h3"]["reachable"] else 0.0 for c in cs])),
                                   "test_fired_mean": {k: float(np.mean([h["test"][k]["fired"] for h in rch])) for k in HN} if rch else None,
                                   "test_mean_fired": float(np.mean([h["test"]["mean_fired"] for h in rch])) if rch else None,
                                   "heldout_disjoint_fired_mean": float(np.mean([h["heldout_disjoint"]["fired"] for h in rch])) if rch else None}
        if ra is None or rb is None:
            e["missing"] = True
            out["H3"][pair] = e
            continue
        d = cbd(ra, rb, ex, "edit", reph)
        e.update(diff=d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], p=d["p_two_sided"], mean_a=d["mean_a"], mean_b=d["mean_b"], n_draws=3, n_units=len(ia[0]["odd_edits"]),
                 by_draw=[float(100 * np.mean([ex(r) for r in ra[k] if reph(r)]) - 100 * np.mean([ex(r) for r in rb[k] if reph(r)])) for k in BDRAWS],
                 n_edits_at_cap_a=[ia[k].get("n_edits_at_cap") for k in BDRAWS], n_edits_at_cap_b=[ib[k].get("n_edits_at_cap") for k in BDRAWS],
                 reachable_a=[ia[k]["reachable"] for k in BDRAWS], reachable_b=[ib[k]["reachable"] for k in BDRAWS])
        out["H3"][pair] = e
        R.add(log, a, "endpoint.H3.rephrase_exact_diff", d["value"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_units=e["n_units"], variant=f"minus {b} ({pair})", alpha=ALPHA, n=200,
              pool="odd_edits", p_two_sided=d["p_two_sided"], n_draws=3)
        # behavioural read-outs of H3 per method: exact / coverage on the odd edits, firing / D1 / wrong target per kind on the test rows, heldout_disjoint
        for lab, rr, ii in ((a, ra, ia), (b, rb, ib)):
            if any(r["method"] == lab and r["metric"] == "h3.rephrase.exact@x" and r["log"] == log for r in R.records):
                continue
            kw = dict(alpha=ALPHA, n=200, draw=None, n_draws=3, n_edits_at_cap=[ii[k].get("n_edits_at_cap") for k in BDRAWS])
            R.add_stat(log, lab, "h3.rephrase.exact@x", cb(rr, ex, "edit", reph), pool="odd_edits", **kw)
            for m in ("tgen", "igen"):
                R.add_stat(log, lab, f"h3.{m}.exact@x", cb(rr, ex, "edit", lambda r, m=m: r["kind"] == "edit" and r["metric"] == m), pool="odd_edits", **kw)
                R.add_stat(log, lab, f"h3.{m}.own_top@x", cb(rr, lambda r: float(bool(r.get("own_released_last"))), "edit", lambda r, m=m: r["kind"] == "edit" and r["metric"] == m), pool="odd_edits", **kw)
            if all(ii[k]["reachable"] for k in BDRAWS):
                forms = [("", None)]
                if lab.endswith("_x"):
                    forms.append(("u", set(ii[0]["unseen_foreign"]["csids"])))
                for suf, keep in forms:
                    for k in HN:
                        if suf == "u" and k == "HN-X":
                            R.add(log, lab, "h3u.test.HN-X.n", sum(1 for r in rr[0] if r["kind"] == "hn" and r.get("part") != "gen" and r["hn_type"] == k and r["csid"] in keep), pool="test_hard_negatives", **kw)
                            continue
                        w = lambda r, k=k, keep=keep: r["kind"] == "hn" and r.get("part") != "gen" and r["hn_type"] == k and (keep is None or r["csid"] in keep)   # noqa: E731
                        wg = lambda r, k=k, keep=keep: r["kind"] == "hn" and r.get("part") == "gen" and r["hn_type"] == k and (keep is None or r["csid"] in keep)  # noqa: E731
                        R.add_stat(log, lab, f"h3{suf}.test.{k}.fired@x", cb(rr, lambda r: float(bool(r["fired"])), "edit", w), pool="test_hard_negatives", **kw)
                        R.add_stat(log, lab, f"h3{suf}.test.{k}.D1@x", cb(rr, lambda r: float(bool(r["D1"])), "edit", w), pool="test_hard_negatives", **kw)
                        R.add_stat(log, lab, f"h3{suf}.test.{k}.wrong_target@x", cb(rr, lambda r: float(bool(r["wrong_target"])), "edit", wg), pool="test_hard_negatives", **kw)
                R.add_stat(log, lab, "h3.probe.fired@x", cb(rr, lambda r: float(bool(r["fired"])), "query_id", lambda r: r["kind"] == "probe"), pool="heldout_disjoint", **kw)
                R.add_stat(log, lab, "h3.probe.D1@x", cb(rr, lambda r: float(bool(r["D1"])), "query_id", lambda r: r["kind"] == "probe"), pool="heldout_disjoint", **kw)
    return out


def decide(ep):
    """Holm within H2 and within H3 (over backbones, baselines and the two pairs) and the decision rule of R6 / R25, literally."""
    for H in ("H2", "H3"):
        ps = {f"{log}/{k}": e["p"] for log in DEV for k, e in ep.get(log, {}).get(H, {}).items() if "p" in e}
        if ps:
            h = ST.holm(ps)
            adj = h["adjusted"] if isinstance(h["adjusted"], dict) else dict(zip(ps, h["adjusted"]))
            rej = h["reject"] if isinstance(h["reject"], dict) else dict(zip(ps, h["reject"]))
            for key in ps:
                log, k = key.split("/")
                ep[log][H][k]["p_holm"] = adj[key]
                ep[log][H][k]["holm_reject_0.05"] = bool(rej[key])
            ep.setdefault("holm", {})[H] = {"n_tests": len(ps), "family": sorted(ps)}
    dec = {"rule": ("R6: 'dominates' only if ReScope exceeds the best gate of the cosine family by at least 10 points of H3 ((T-Gen + I-Gen) / 2 exact on the odd edits under the "
                    "near-scope constraint) on both backbones with an interval that excludes 0; R25 (3): additionally the gate-level coverage-form difference over the 20 "
                    "calibration draws must exclude 0 with the same sign; otherwise 'trade-off'.  Intervals: paired crossed bootstrap (draws x edits, 2000 resamples, 95 percent); "
                    "behavioural intervals rest on 3 calibration draws."), "pairs": {}}
    for pair in ("pair1", "pair2_x", "retr"):
        per, ok, complete = {}, True, True
        for log in DEV:
            e = ep.get(log, {}).get("H3", {}).get(pair)
            if e is None or e.get("missing"):
                complete = False
                per[log] = {"missing": True}
                continue
            g = e.get("gate_coverage_form_20_draws")
            c = {"a": e["a"], "b": e["b"], "behaviour_diff_points": e["diff"], "behaviour_interval": [e["ci_lo"], e["ci_hi"]], "behaviour_n_draws": e["n_draws"],
                 "behaviour_by_draw": e["by_draw"], "p": e["p"], "p_holm": e.get("p_holm"), "gate_coverage_diff_points": g["value"], "gate_coverage_interval": [g["ci_lo"], g["ci_hi"]],
                 "gate_n_draws": g["n_draws"], "n_edits_at_cap_a": e["n_edits_at_cap_a"], "n_edits_at_cap_b": e["n_edits_at_cap_b"],
                 "at_least_10_points": bool(e["diff"] >= 10.0), "behaviour_interval_excludes_0": bool(e["ci_lo"] > 0 or e["ci_hi"] < 0),
                 "gate_interval_excludes_0_same_sign": bool((g["ci_lo"] > 0 and e["diff"] > 0) or (g["ci_hi"] < 0 and e["diff"] < 0))}
            c["conditions_met"] = bool(c["at_least_10_points"] and c["behaviour_interval_excludes_0"] and e["ci_lo"] > 0 and c["gate_interval_excludes_0_same_sign"])
            ok = ok and c["conditions_met"]
            per[log] = c
        dec["pairs"][pair] = {"verdict": ("dominates" if ok else "trade-off") if complete else "incomplete", "per_log": per,
                              "claim_form": {"pair1": "ReScope against the best gate of the cosine family", "pair2_x": "with log-derived negatives (rescope_x against the same gate in its _x form)",
                                             "retr": "ReScope against the selected retrieval rule, in the first pair's form (reported; the rule of R6 names the cosine family)"}[pair]}
    return dec


# ----------------------------------------------------------------------------- paper numbers
def camel(s):
    s = re.sub(r"[^A-Za-z0-9]+", " ", s)
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(w[:1].upper() + w[1:] for w in "".join(words.get(ch, ch) if ch.isdigit() else ch for ch in s).split())


def paper(R, ep, dec, sel_by_log):
    num, doc = {}, {}

    def put(name, value, text):
        if value is None:
            return
        num[name], doc[name] = value, text
    idx = {}
    for r in R.records:
        idx.setdefault((r["log"], r["method"], r["metric"], r["pool"], r["draw"], r.get("alpha")), r)
    for log in DEV:
        s, sel = SHORT[log], sel_by_log[log]
        rows = [("deployed", "Deployed", None), ("preedit", "Preedit", None), ("rescope", "Rescope", ALPHA), ("rescope_shared", "RescopeShared", ALPHA), ("rescope_null", "RescopeNull", ALPHA),
                ("rescope_x", "RescopeX", ALPHA), ("rescope_lin", "RescopeLin", ALPHA), ("where_logonly", "RescopeW", None), (sel["cos_H3"], "BestCos", ALPHA),
                (sel["cos_H2"], "BestCosHtwo", ALPHA), (sel["cos_H3_x"], "BestCosX", ALPHA), ("cosine", "Cosine", ALPHA), ("cosine_shared", "CosineShared", ALPHA),
                (sel["retr_H3"], "Retr", ALPHA), ("lookup", "Lookup", None), ("ball", "Ball", ALPHA), ("router", "Router", ALPHA), ("cosine_default", "CosDefault", None),
                ("cosine_published", "CosPublished", None), ("zscore_default", "Zscore", None), ("exact", "Exact", None), ("dual", "Dual", None)]
        for lab, nm, alpha in rows:
            sfx = "@x" if alpha is not None else ""
            for metric, pool, mn, what in ((f"edit.rel.exact{sfx}", "edits", "Rel", "Rel exact (%)"), (f"edit.tgen.exact{sfx}", "edits", "TGen", "T-Gen exact (%)"),
                                           (f"edit.igen.exact{sfx}", "edits", "IGen", "I-Gen exact (%)"), (f"edit.rephrase.exact{sfx}", "edits", "Reph", "(T-Gen + I-Gen) / 2 exact (%)"),
                                           (f"probe.D1{sfx}", "heldout_disjoint", "DOne", "D1 on heldout_disjoint (%)"), (f"probe.fired{sfx}", "heldout_disjoint", "Fire", "rows of heldout_disjoint on which an edit is released (%, behaviour)"),
                                           (f"hn.HN-I.D1{sfx}", "hard_negatives", "HNIDOne", "D1 on HN-I (%)"), (f"hn.HN-Q.D1{sfx}", "hard_negatives", "HNQDOne", "D1 on HN-Q (%)"),
                                           (f"hn.HN-X.D1{sfx}", "hard_negatives", "HNXDOne", "D1 on HN-X (%)"), (f"hn.all.fired{sfx}", "hard_negatives", "HNFire", "near-scope negatives on which an edit is released (%)")):
                r = idx.get((log, lab, metric, pool, None, alpha))
                if r is not None:
                    put(f"Eone{s}{nm}{mn}", r["value"], f"{log}, {lab}: {what}" + ("; mean over calibration draws 0-2, alpha = 0.05" if alpha else ""))
            g = idx.get((log, lab, "gate.fire.hd@mean", "gate_level", None, alpha))
            if g is not None:
                put(f"Eone{s}{nm}GateFire", g["value"], f"{log}, {lab}: gate-level false firing on heldout_disjoint (%), mean over 20 calibration draws, alpha = 0.05")
            for gm, mn, what in (("gate.BEN-T.fired@mean", "BenT", "gate-level firing on BEN-T (%; prompt-only keys)"), ("gate.MME-T.fired@mean", "MmeT", "gate-level firing on MME-T (%)"),
                                 ("gate.n_capped@mean", "Capped", "number of capped edits (mean over 20 draws)"), ("gate.h2.cov@mean", "GateCov", "gate-level coverage of text and image rephrases (%), mean over 20 draws"),
                                 ("gate.h3.cov@mean", "HthreeCov", "H3 coverage at gate level (%), mean over 20 draws")):
                g = idx.get((log, lab, gm, "gate_level", None, alpha))
                if g is not None:
                    put(f"Eone{s}{nm}{mn}", g["value"], f"{log}, {lab}: {what}")
        for role, e in ep.get(log, {}).get("H2", {}).items():
            if "diff" in e:
                put(f"Eone{s}Htwo{camel(role)}Diff", e["diff"], f"{log}: H2, (T-Gen + I-Gen) / 2 exact of rescope minus {e['method']} ({role}), points; paired crossed bootstrap, 3 draws")
                put(f"Eone{s}Htwo{camel(role)}Lo", e["ci_lo"], "lower end of its 95 % interval")
                put(f"Eone{s}Htwo{camel(role)}Hi", e["ci_hi"], "upper end of its 95 % interval")
                put(f"Eone{s}Htwo{camel(role)}PHolm", e.get("p_holm"), "its Holm-adjusted p-value (within H2)")
        for pair, e in ep.get(log, {}).get("H3", {}).items():
            if "diff" in e:
                put(f"Eone{s}Hthree{camel(pair)}Diff", e["diff"], f"{log}: H3, (T-Gen + I-Gen) / 2 exact on the odd edits under the near-scope constraint, {e['a']} minus {e['b']}, points; 3 draws")
                put(f"Eone{s}Hthree{camel(pair)}Lo", e["ci_lo"], "lower end of its 95 % interval")
                put(f"Eone{s}Hthree{camel(pair)}Hi", e["ci_hi"], "upper end of its 95 % interval")
                put(f"Eone{s}Hthree{camel(pair)}PHolm", e.get("p_holm"), "its Holm-adjusted p-value (within H3)")
                g = e.get("gate_coverage_form_20_draws")
                if g:
                    put(f"Eone{s}Hthree{camel(pair)}GateDiff", g["value"], f"{log}: the same difference in its gate-level coverage form over 20 draws, points")
                    put(f"Eone{s}Hthree{camel(pair)}GateLo", g["ci_lo"], "lower end of its 95 % interval")
                    put(f"Eone{s}Hthree{camel(pair)}GateHi", g["ci_hi"], "upper end of its 95 % interval")
    for pair, d in dec["pairs"].items():
        put(f"EoneDecision{camel(pair)}", d["verdict"], f"decision rule of R6 / R25 for {d['claim_form']}")
    for bb, e in (e1c.selection() or {}).items():
        b = "Q" if bb == "qwen25vl" else "L"
        put(f"Eone{b}BestCosName", e["cosine"]["H3"], f"{bb}: the best gate of the cosine family (largest H3 coverage on the development log)")
        put(f"Eone{b}BestCosHtwoName", e["cosine"]["H2"], f"{bb}: the best gate of the cosine family for H2")
        put(f"Eone{b}RetrName", e["retr"]["H3"], f"{bb}: the selected retrieval rule")
    jdump(dict(sorted(num.items())), os.path.join(OUT, "paper_numbers.json"))
    jdump(dict(sorted(doc.items())), os.path.join(OUT, "paper_numbers_doc.json"))
    return num


def main():
    t0 = time.time()
    R = fx.io.Results("E1", OUT)
    draws, cells = gate_level(R)
    print(f"[{time.time() - t0:5.0f}s] gate level: {len(R.records)} records", flush=True)
    ep, sel_by_log = {}, {}
    for log in DEV:
        L = fx.logs.open(log)
        sel = e1c.selected_labels(L)
        sel_by_log[log] = sel
        best = list(dict.fromkeys([sel["cos_H3"], sel["cos_H2"]]))
        best_x = list(dict.fromkeys([sel["cos_H3_x"], sel["cos_H2_x"]]))
        retr = list(dict.fromkeys([sel["retr_H3"], sel["retr_H2"]]))
        fixed = ["deployed", "preedit", "exact", "cosine_default", "lookup", "zscore_default", "cosine_published", "where_logonly", "dual"]
        cal = list(dict.fromkeys(["rescope", "rescope_shared", "rescope_null", "rescope_x", "rescope_lin"] + best + best_x + ["cosine", "cosine_shared"] + retr + ["ball", "router"]))
        tf, fx_ = behaviour(R, log, fixed, cal)
        if L.backbone == "llava":
            prompt_position(R, log, tf, fx_)
        generation(R, log, [("deployed", None), ("preedit", None), ("rescope", 0), ("rescope_x", 0)] + [(b, 0) for b in best] + [(r, 0) for r in retr] + [("where_logonly", None), ("dual", None)])
        ep[log] = endpoints(R, log, tf, fx_, cells, sel)
        print(f"[{time.time() - t0:5.0f}s] behaviour {log}: {len(R.records)} records", flush=True)
    dec = decide(ep)
    jdump(ep, os.path.join(OUT, "endpoints.json"))
    complete = all(d["verdict"] != "incomplete" for d in dec["pairs"].values())
    dec["complete"] = complete
    jdump(dec, os.path.join(OUT, "decision.json"))
    seen, miss = set(), []
    for m in MISSING:
        k = (m["log"], m["label"], m["draw"], m["part"])
        if k not in seen and m["part"] != "tfp":
            seen.add(k)
            miss.append(m)
    jdump({"missing_behavioural_cells": miss, "n": len(miss)}, os.path.join(OUT, "missing.json"))
    R.save(script="e1_analyze.py", n_missing_cells=len(miss))
    paper(R, ep, dec, sel_by_log)
    print(f"[{time.time() - t0:5.0f}s] {len(R.records)} records; missing behavioural cells: {len(miss)}; decision: { {k: v['verdict'] for k, v in dec['pairs'].items()} }")


if __name__ == "__main__":
    main()
