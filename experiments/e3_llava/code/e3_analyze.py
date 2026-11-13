"""E3 / E4 on the LLaVA logs: statistics.  Reads the gate-level cells (out/gate), the behavioural row files (out/rows) and cell records
(out/behave) and writes out/results.json (schema of fx.io), out/summary.json (what the tables are rendered from), out/gate_draws.json
(every gate-level number of every calibration draw), out/scale_series.json, out/retention_series.json, out/dev_check.json,
out/paper_numbers.json + paper_numbers_doc.json, out/missing.json.  No number is typed by hand.

    python e3_analyze.py           (CPU; re-runnable after every log; cells that do not exist yet are listed in out/missing.json)
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__, needs=("experiments/e3_llava/out/gate", "experiments/e3_llava/out/behave", "experiments/e3_llava/out/rows"))
import glob
import os
import re
import time

from e3c import *  # noqa: F401,F403
import e3c

MX, ST = fx.metrics, fx.stats
MISSING = []
SUB = ("all", "first100", "later")


def camel(s):
    s = re.sub(r"[^A-Za-z0-9]+", " ", str(s))
    words = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    return "".join(w[:1].upper() + w[1:] for w in "".join(words.get(ch, ch) if ch.isdigit() else ch for ch in s).split())


def pct(x):
    x = np.asarray(x, dtype=float)
    return 100.0 * float(x.mean()) if x.size else None


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ----------------------------------------------------------------------------- gate level
def cbm(F, units=None):
    """Crossed bootstrap (R10) of a rate from a (draws, rows) flag matrix; units: unit id per row (None: every row a unit)."""
    F = np.asarray(F, dtype=float)
    if F.shape[1] == 0:
        return None
    if units is None:
        num, den = F, np.ones_like(F)
    else:
        u, inv = np.unique(np.asarray(units), return_inverse=True)
        A = np.zeros((F.shape[1], len(u)))
        A[np.arange(F.shape[1]), inv] = 1.0
        num, den = F @ A, np.tile(A.sum(0), (F.shape[0], 1))
    s = ST.crossed_bootstrap(num, den, scale=100.0)
    return {k: s[k] for k in ("value", "ci_lo", "ci_hi", "n_draws", "n_units")}


def edit_subsets(T, later):
    m = {"all": np.ones(T, dtype=bool), "first100": np.arange(T) < 100, "later": np.isin(np.arange(T), later)}
    return {k: v for k, v in m.items() if v.any() and not (k == "first100" and T == 100)}


def g_metrics(c, meta, es):
    """Flat {metric: number} of one gate-level cell (one calibration draw)."""
    m = {}
    for k, v in c["hd"].items():
        m["fire." + k] = v["fired"]
    m["fire.hd@prompt"] = c["hd_prompt"]["hd"]["fired"]
    if c.get("literal"):
        m["fire.hd.literal"], m["fire.hd_qd.literal"] = c["literal"]["hd"]["hd"]["fired"], c["literal"]["hd"]["hd_qd"]["fired"]
    for p in ("BEN-T", "MME-T"):
        b = c.get(p) or {}
        if "fired" in b:
            m[p + ".fired"], m[p + ".n"] = b["fired"], b["n"]
            if "far" in b:
                m[p + ".far.fired"], m[p + ".far.n"], m[p + ".related.n"] = b["far"]["fired"], b["far"]["n"], b["related"]["n"]
    cov = {}
    for k in ("rel", "tgen", "igen"):
        ot, oth = unflags(c["edit"][k]["own_top"]), unflags(c["edit"][k]["other_top"])
        cov[k] = ot
        for s, msk in es.items():
            sfx = "" if s == "all" else "." + s
            m[f"cov.{k}{sfx}"], m[f"other.{k}{sfx}"] = pct(ot[msk]), pct(oth[msk])
        m[f"cov.{k}@prompt"] = c["edit"][k]["own_top_prompt_pct"]
    for s, msk in es.items():
        m["h2cov" + ("" if s == "all" else "." + s)] = 0.5 * (pct(cov["tgen"][msk]) + pct(cov["igen"][msk]))
    hk, fh, oh = np.array(meta["hn"]["kind"]), unflags(c["hn_fired"]), unflags(c["hn_own_top"])
    uf = unflags(meta["hn"]["unseen_foreign"]) if meta["hn"]["unseen_foreign"] else None
    if hk.size:
        for k in HN:
            if (hk == k).any():
                m[f"hn.{k}.fired"], m[f"hn.{k}.own"], m[f"hn.{k}.n"] = pct(fh[hk == k]), pct(oh[hk == k]), int((hk == k).sum())
                m[f"hn.{k}.fired@prompt"] = c["hn_prompt"].get(k)
                if uf is not None:
                    m[f"hnu.{k}.n"] = int((uf & (hk == k)).sum())
                    m[f"hnu.{k}.fired"] = pct(fh[uf & (hk == k)]) if m[f"hnu.{k}.n"] else None
        m["hn.mean.fired"] = mean([m.get(f"hn.{k}.fired") for k in HN])
    w = c["lineage_hd"]["widths"]
    m["lineage.n_firing"], m["lineage.width_mean"], m["lineage.width_max"] = len(w), (float(np.mean(w)) if w else None), (max(w) if w else None)
    m["lineage.multi"] = pct(np.array(w) > 1) if w else None
    cal = c.get("cal") or {}
    if c.get("alpha") is not None:
        m["n_capped"] = c["n_capped"]
        m["own.silenced_before_cap"] = cal.get("n_edits_silenced_by_literal_rule", cal.get("n_own_requests_silenced_by_literal_rule"))
        m["eta"] = cal.get("eta") if not cal.get("eta_infinite") else None
        m["eta_infinite"] = 1.0 if cal.get("eta_infinite") else 0.0
        m["bound"] = cal.get("bound_false_firing")
        th = cal.get("thresholds") or {}
        if "median" in th:
            m["thr.min"], m["thr.median"], m["thr.max"], m["thr.n_infinite"] = th["min"], th["median"], th["max"], th["n"] - th["n_finite"]
    op = c.get("own_prompts")
    m["own.not_released_after_cap"] = op["n_not_released"] if op else int(round((~cov["rel"]).sum()))
    m["own.released_by_other"] = int(unflags(c["edit"]["rel"]["other_top"]).sum())
    m["top1.n_differs"] = ((c["st"].get("top1") or {}).get("all") or {}).get("n_differs")
    m["prompt.n_rows_differ"] = sum(c["n_rows_differ_tf_vs_prompt"].values())
    for key, pre in (("h3", "h3"), ("h3_unseen", "h3u")):
        h = c.get(key)
        if h is None or "not_defined" in h:
            continue
        m[pre + ".reachable"] = 1.0 if h["reachable"] else 0.0
        m[pre + ".n_at_cap"] = h.get("n_edits_at_cap")
        if h["reachable"]:
            m[pre + ".cov"], m[pre + ".tgen"], m[pre + ".igen"], m[pre + ".delta"] = h["rephrase"]["mean"], h["rephrase"]["tgen"], h["rephrase"]["igen"], h["delta"]
            for k in HN:
                if k in h["test"]:
                    m[f"{pre}.test.{k}.fired"], m[f"{pre}.test.{k}.own"], m[f"{pre}.test.{k}.n"] = h["test"][k]["fired"], h["test"][k]["own_top"], h["test"][k]["n"]
            m[pre + ".test.mean"], m[pre + ".hd.fired"] = h["test"]["mean_fired"], h["heldout_disjoint"]["fired"]
        else:
            m[pre + ".cov"] = 0.0                                    # rule of e1_main/out/best_cosine_gate.json: an unreachable draw counts 0
    return m


G_CI = ("fire.hd", "fire.hd_near", "fire.hd_far", "fire.hd_qd", "cov.tgen", "cov.igen", "cov.tgen.later", "cov.igen.later", "h2cov", "hn.HN-I.fired", "hn.HN-Q.fired", "hn.HN-X.fired")


def gate_level(R, log, S, GD):
    gdir = os.path.join(GATE, log)
    if not os.path.exists(os.path.join(gdir, "_meta.json")):
        MISSING.append({"log": log, "level": "gate", "what": "no gate-level output"})
        return None
    meta = jload(os.path.join(gdir, "_meta.json"))
    lb, T = meta["labels"], meta["T"]
    es = edit_subsets(T, meta["later_edits"])
    pm = {k: unflags(v) for k, v in meta["probe"]["masks"].items()}
    qid = np.array(meta["probe"]["query_id"])
    hk, hown = np.array(meta["hn"]["kind"]), np.array(meta["hn"]["own"])
    for lab in lb["cal"] + lb["cal_gl"] + lb["fixed_cpu"]:
        for ro in ("reg", "qd"):
            p = os.path.join(gdir, f"{lab}__{ro}.json")
            if not os.path.exists(p):
                MISSING.append({"log": log, "level": "gate", "label": lab, "readout": ro})
                continue
            f = jload(p)
            cells = f["cells"]
            node = S.setdefault(log, {}).setdefault(lab, {}).setdefault(ro, {})
            if "not_defined" in cells["0"]:
                node["not_defined"] = cells["0"]["not_defined"]
                R.add(log, lab, "not_defined", str(cells["0"]["not_defined"])[:300], pool="gate", variant=ro)
                continue
            ds = sorted(int(d) for d in cells)
            per = {d: g_metrics(cells[str(d)], meta, es) for d in ds}
            names = sorted({k for d in ds for k in per[d]})
            g = node.setdefault("gate", {})
            kw = dict(alpha=None if f["fixed"] else ALPHA, n=None if f["fixed"] else N, variant=ro, pool="gate")
            for nm in names:
                vals = [per[d].get(nm) for d in ds]
                GD.setdefault(log, {}).setdefault(lab, {}).setdefault(ro, {})[nm] = vals
                num = [v for v in vals if isinstance(v, (int, float))]
                if not num:
                    continue
                s = ST.over_draws(num)
                g[nm] = {"mean": s["mean"], "p05": s["p05"], "p95": s["p95"], "min": s["min"], "max": s["max"], "n_draws": s["n_draws"]}
                if f["fixed"]:
                    R.add(log, lab, "g." + nm, num[0], draw=None, **kw)
                else:
                    R.add(log, lab, "g." + nm + "@mean", s["mean"], ci_lo=s["p05"], ci_hi=s["p95"], n_units=s["n_draws"], draw=None, **kw)
            # crossed bootstrap over calibration draws x units (R10): probe units = query clusters, edit units = edits
            C = [cells[str(d)] for d in ds]
            PF = np.array([unflags(c["probe_fired"]) for c in C])
            HF = np.array([unflags(c["hn_fired"]) for c in C]) if hk.size else None
            for nm in G_CI:
                if nm not in g:
                    continue
                if nm.startswith("fire."):
                    msk = pm[nm.split(".", 1)[1]]
                    cb = cbm(PF[:, msk], qid[msk])
                elif nm.startswith("cov.") or nm == "h2cov":
                    parts = nm.split(".")
                    msk = es[parts[2]] if len(parts) > 2 else es["all"]
                    if nm == "h2cov":
                        F = 0.5 * (np.array([unflags(c["edit"]["tgen"]["own_top"]) for c in C]).astype(float) + np.array([unflags(c["edit"]["igen"]["own_top"]) for c in C]))
                    else:
                        F = np.array([unflags(c["edit"][parts[1]]["own_top"]) for c in C])
                    cb = cbm(F[:, msk])
                else:
                    k = nm.split(".")[1]
                    cb = cbm(HF[:, hk == k], hown[hk == k])
                if cb:
                    g[nm].update(ci_lo=cb["ci_lo"], ci_hi=cb["ci_hi"], n_units=cb["n_units"])
                    R.add(log, lab, "g." + nm + "@x", cb["value"], ci_lo=cb["ci_lo"], ci_hi=cb["ci_hi"], n_units=cb["n_units"], draw=None, n_draws=cb["n_draws"], **kw)
            node["n_draws_gate"] = len(ds)
            if "h3_subset" in C[0]:
                node["h3_subset"] = C[0]["h3_subset"]
    S[log]["_meta"] = {k: meta[k] for k in ("T", "n_probe_rows_all", "n_probe_rows_valid", "n_probe_invalid_excluded", "degenerate_y0", "hn_invalid_excluded", "lp_shift_rows",
                                            "selection", "describe")}
    S[log]["_meta"]["n_later_edits"] = len(meta["later_edits"])
    S[log]["_meta"]["hn_unseen_foreign_n"] = meta["hn"]["unseen_foreign_n"]
    S[log]["_meta"]["n_hd"] = {k: int(v.sum()) for k, v in pm.items()}
    fp = os.path.join(gdir, "_factor.json")
    if os.path.exists(fp):
        fa = jload(fp)
        S[log]["_factor"] = {"non_oracle_edits": fa["non_oracle_edits"], "n_non_oracle": len(fa["non_oracle_edits"]), "non_oracle_matches_report": fa["non_oracle_matches_report"],
                             "n_targets_found": fa["n_targets_found"], "rebase": None if fa["rebase"] is None else {k: fa["rebase"][k] for k in ("rebased", "not_rebased")},
                             "margin": {k: v for k, v in fa["self_separation_margin"].items() if k != "per_edit"}}
        for k, v in (("factor.n_non_oracle", len(fa["non_oracle_edits"])), ("factor.margin.n_positive", fa["self_separation_margin"]["n_positive"]),
                     ("factor.margin.min", fa["self_separation_margin"]["min"])):
            R.add(log, "log", k, v, pool="factors")
        if fa["rebase"]:
            for a in ("rebased", "not_rebased"):
                for b in ("max", "median"):
                    R.add(log, "log", f"factor.rebase_error.{a}.{b}", fa["rebase"][a][b], pool="factors")
    return meta


# ----------------------------------------------------------------------------- behaviour
def rows_of(log, label, d, part, note=True):
    tag = f"{label}__d{'x' if d is None else d}__{part}"
    p, c = os.path.join(OUT, "rows", f"{log}__{tag}.jsonl.gz"), os.path.join(BEH, log, tag + ".json")
    if not os.path.exists(c):
        if note:
            MISSING.append({"log": log, "level": "behaviour", "cell": tag})
        return None, None
    rec = jload(c)
    if not os.path.exists(p):
        return None, rec
    return fx.io.load_rows(p), rec


def flag(f):
    return lambda r: float(bool(r[f]))


def specs(T, later, qd_sids, uf_sids, has_fired, has_own):
    """(name, kind, value, unit, where, with_interval) of every behavioural read-out of the teacher-forced rows."""
    out = []
    later = set(later)
    sub = {"all": lambda r: True, "first100": lambda r: r["edit"] < 100, "later": lambda r: r["edit"] in later}
    if T == 100:
        sub.pop("first100")
    if not later:
        sub.pop("later")
    for s, w in sub.items():
        sfx = "" if s == "all" else "." + s
        for m in ("rel", "tgen", "igen"):
            wm = (lambda r, m=m, w=w: r["metric"] == m and w(r))
            out.append((f"edit.{m}.exact{sfx}", "edit", flag("exact"), "edit", wm, True))
            out.append((f"edit.{m}.token{sfx}", "edit", lambda r: float(r["token_acc"]), "edit", wm, False))
            if has_own:
                out.append((f"edit.{m}.own_top{sfx}", "edit", flag("own_released_last"), "edit", wm, False))
            for f in ("kept", "gained", "lost"):
                out.append((f"kg.{m}.{f}{sfx}", "kg", flag(f), "edit", wm, s == "all" and f != "lost"))
        out.append((f"edit.reph.exact{sfx}", "edit", flag("exact"), "edit", (lambda r, w=w: r["metric"] in ("tgen", "igen") and w(r)), True))
    pw = {"": lambda r: True, ".near": lambda r: r["near_edit"] is not None, ".far": lambda r: r["near_edit"] is None, ".qd": lambda r: r["csid"] in qd_sids}
    for s, w in pw.items():
        if has_fired:
            out.append((f"probe.fired{s}", "probe", flag("fired"), "query_id", w, s in ("", ".near")))
        out.append((f"probe.D1{s}", "probe", flag("D1"), "query_id", w, s in ("", ".near", ".qd")))
    out.append(("probe.tok_loc", "probe", lambda r: float(r["tok_loc"]), "query_id", lambda r: True, False))
    out.append(("probe.T-Loc*", "probe", lambda r: float(r["tok_loc"]), "query_id", lambda r: r["ptype"] == "text", False))
    out.append(("probe.I-Loc*", "probe", lambda r: float(r["tok_loc"]), "query_id", lambda r: r["ptype"] == "image", False))
    for k in HN + ("all",):
        w = (lambda r: True) if k == "all" else (lambda r, k=k: r["hn_type"] == k)
        if has_fired:
            out.append((f"hn.{k}.fired", "hn", flag("fired"), "edit", w, False))
            if has_own:
                out.append((f"hn.{k}.own", "hn", flag("own_released_last"), "edit", w, False))
        out.append((f"hn.{k}.D1", "hn", flag("D1"), "edit", w, True))
        if k != "all" and uf_sids is not None:
            wu = (lambda r, k=k: r["hn_type"] == k and r["csid"] in uf_sids)
            if has_fired:
                out.append((f"hnu.{k}.fired", "hn", flag("fired"), "edit", wu, False))
            out.append((f"hnu.{k}.D1", "hn", flag("D1"), "edit", wu, False))
    return out


def summarise(by, kind_rows, sp):
    """One read-out over the behavioural draws: per-draw values, mean, crossed-bootstrap interval (R10; n_draws stated)."""
    name, kind, val, unit, where, ci = sp
    per = {}
    for d, parts in by.items():
        rows = [r for r in parts[kind] if where(r)]
        if not rows:
            return None
        per[d] = 100.0 * float(np.mean([val(r) for r in rows]))
    out = {"mean": float(np.mean(list(per.values()))), "per_draw": {str(d): v for d, v in per.items()}, "n_draws": len(per), "n_rows": len(rows)}
    if ci:
        g = MX.grid({d: parts[kind] for d, parts in by.items()}, val, unit=unit, where=where)
        s = ST.crossed_bootstrap(g["num"], g["den"], scale=100.0)
        out.update(ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], n_units=s["n_units"])
    return out


def split(rows, dep_e=None):
    e = [r for r in rows if r["kind"] == "edit"]
    return {"edit": e, "probe": [r for r in rows if r["kind"] == "probe"], "hn": [r for r in rows if r["kind"] == "hn" and r.get("part") != "gen"],
            "kg": MX.kept_gained_rows(e, dep_e) if dep_e is not None else []}


def behaviour(R, log, S, meta):
    T, lb = meta["T"], meta["labels"]
    draws = bdraws(T)
    qd_sids = {s for s, f in zip(meta["probe"]["sid"], meta["probe"]["masks"]["hd_qd"]) if f == "1"}
    uf_sids = {s for s, f in zip(meta["hn"]["sid"], meta["hn"]["unseen_foreign"]) if f == "1"} if meta["hn"]["unseen_foreign"] else None
    dep, _ = rows_of(log, "deployed", None, "tf")
    dep_e = [r for r in dep if r["kind"] == "edit"] if dep else None
    fixed = ["deployed"] + lb["fixed_cpu"] + ["cosine_published", "where_logonly"]
    for lab in fixed + lb["cal"]:
        is_fixed = lab in fixed
        for ro, part in (("reg", "tf"),) + ((("qd", "tfq"),) if not is_fixed else ()):
            by, recs = {}, {}
            for d in ([None] if is_fixed else draws):
                rows, rec = rows_of(log, lab, d, part)
                if rec is not None:
                    recs[d] = rec
                if rows is not None:
                    by[d] = split(rows, dep_e)
            node = S.setdefault(log, {}).setdefault(lab, {}).setdefault(ro, {})
            for d, rec in recs.items():
                if rec.get("not_defined"):
                    node["not_defined"] = rec["not_defined"]
                if rec.get("skipped_for_cost"):
                    node["skipped_for_cost"] = rec["skipped_for_cost"]
                    node["timing"] = {k: rec.get(k) for k in ("seconds_build", "seconds_per_sequence", "seconds_one_pass_estimated", "n_sequences")}
                    R.add(log, lab, "skipped_for_cost", rec["skipped_for_cost"], pool="behaviour", variant=ro)
                if rec.get("seconds_one_pass"):
                    node["timing"] = {k: rec.get(k) for k in ("seconds_build", "seconds_one_pass", "n_sequences")}
            if not by:
                continue
            r0 = next(iter(by.values()))
            has_fired = r0["probe"][0].get("fired") is not None
            has_own = r0["edit"][0].get("own_released_last") is not None
            b = node.setdefault("beh", {})
            kw = dict(alpha=None if is_fixed else ALPHA, n=None if is_fixed else N, variant=ro)
            for sp in specs(T, meta["later_edits"], qd_sids, uf_sids, has_fired, has_own):
                if sp[1] == "kg" and dep_e is None:
                    continue
                s = summarise(by, None, sp)
                if s is None:
                    continue
                b[sp[0]] = s
                pool = {"edit": "edits", "kg": "edits", "probe": "heldout_disjoint", "hn": "hard_negatives"}[sp[1]]
                for d, v in s["per_draw"].items():
                    R.add(log, lab, "b." + sp[0], v, draw=None if d == "None" else int(d), pool=pool, **kw)
                if not is_fixed or "ci_lo" in s:
                    R.add(log, lab, "b." + sp[0] + "@x", s["mean"], ci_lo=s.get("ci_lo"), ci_hi=s.get("ci_hi"), n_units=s.get("n_units"), draw=None, pool=pool, n_draws=s["n_draws"], **kw)
            node["n_draws_beh"] = len(by)
            node["draws_beh"] = [d for d in by]
            node["n_capped_beh"] = {str(d): recs[d].get("n_capped") for d in recs}
            node["counts"] = {"n_edit_rows": len(r0["edit"]), "n_hd": len(r0["probe"]), "n_hn": len(r0["hn"]),
                              "n_hd_y0_degenerate": sum(bool(r.get("y0_deg")) for r in r0["probe"]), "n_nonfinite": sum(not r.get("finite", True) for k in ("edit", "probe", "hn") for r in r0[k])}
            # the decode's position on the sequences where it differs (LLaVA)
            if ro == "reg" and lab not in ("deployed", "cosine_published", "where_logonly"):
                tot = {"n_rows": [], "n_fired_differs": [], "n_outcome_differs": []}
                for d in by:
                    rp, _ = rows_of(log, lab, d, "tfp")
                    if rp is None:
                        continue
                    tfm = {r["csid"]: r for k in ("edit", "probe", "hn") for r in by[d][k]}
                    tot["n_rows"].append(len(rp))
                    tot["n_fired_differs"].append(sum(bool(r["fired"]) != bool(tfm[r["csid"]]["fired"]) for r in rp))
                    tot["n_outcome_differs"].append(sum((r["exact"] != tfm[r["csid"]]["exact"]) for r in rp))
                if tot["n_rows"]:
                    node["prompt_position"] = {k: float(np.mean(v)) for k, v in tot.items()}
                    for k, v in node["prompt_position"].items():
                        R.add(log, lab, "b.prompt_position." + k, v, pool="lp_shift_sequences", draw=None, **kw)
    # stored metrics of the deployed log (vendor pass, padded batches) and its probe flags on heldout_disjoint
    L = fx.logs.open(log)
    st = fx.logs.stored(log, pools=True)
    hd = [L.probe_rec(j) for j in L.pool("heldout_disjoint")]
    pc = [r for r in hd if r.get("pre_correct_long")]
    sd = {"edits": {m: {"exact": st["edits"][m]["exact"], "token": st["edits"][m]["token"]} for m in ("rel", "tgen", "igen")},
          "hd": {"n": len(hd), "D1": pct([bool(r["is_complaint_D1"]) for r in hd]), "n_pre_correct": len(pc), "n_D2": sum(bool(r["is_complaint_D2_long"]) for r in pc),
                 "D2": pct([bool(r["is_complaint_D2_long"]) for r in pc]) if pc else None}}
    S[log].setdefault("deployed", {}).setdefault("reg", {})["stored"] = sd
    for m in ("rel", "tgen", "igen"):
        R.add(log, "deployed", f"stored.edit.{m}.exact", sd["edits"][m]["exact"], pool="edits", note="vendor pass, padded batches")
    R.add(log, "deployed", "stored.probe.D1", sd["hd"]["D1"], pool="heldout_disjoint", n_units=len(hd))
    R.add(log, "deployed", "stored.probe.D2", sd["hd"]["D2"], pool="heldout_disjoint", n_units=len(pc))
    # draw 0: D2 on heldout_disjoint (rate of the pre-correct rows, R7); free generation for rescope
    for lab in lb["d2"]:
        rows, rec = rows_of(log, lab, 0, "d2")
        if rows is None:
            continue
        s = MX.d2_stats(rows, ci=True)
        v = s["D2"] or {}
        S[log][lab]["reg"]["d2"] = {"n_pre_correct": s["n_pre_correct"], "n_D2": s["n_D2"], "D2": v.get("value"), "ci_lo": v.get("ci_lo"), "ci_hi": v.get("ci_hi"), "fired": s["fired"], "draw": 0}
        R.add(log, lab, "b.D2", v.get("value"), ci_lo=v.get("ci_lo"), ci_hi=v.get("ci_hi"), n_units=v.get("n_units"), draw=0, alpha=ALPHA, n=N, pool="heldout_disjoint", n_pre_correct=s["n_pre_correct"], n_D2=s["n_D2"])
    rows, rec = rows_of(log, "rescope", 0, "gen")
    if rows is not None:
        e, p = [r for r in rows if r["kind"] == "edit"], [r for r in rows if r["kind"] == "probe"]
        gs = MX.gen_stats(e)
        gp = {"n": len(p), "fired": pct([bool(r.get("fired")) for r in p]), "n_degenerate": sum(any((r.get("degenerate") or {}).values()) for r in p)}
        for f in ("agree", "long_eq_pre"):                          # generated answer agrees with / equals the pre-edit answer
            if p and p[0].get(f) is not None:
                gp[f] = pct([bool(r[f]) for r in p])
        S[log]["rescope"]["reg"]["gen"] = {"edits_first100": gs, "probes200": gp, "draw": 0}
        for m in gs:
            for f in ("eq", "contains", "starts", "n_degenerate"):
                R.add(log, "rescope", f"b.gen.{m}.{f}", gs[m][f], draw=0, alpha=ALPHA, n=N, pool="edits_first100")
        for f, v in gp.items():
            R.add(log, "rescope", f"b.gen.probe.{f}", v, draw=0, alpha=ALPHA, n=N, pool="heldout_disjoint_200")
    # behavioural H3
    for lab in lb["h3"]:
        per = {}
        for d in draws:
            rows, rec = rows_of(log, lab, d, "h3")
            if rec is None or rec.get("not_defined"):
                continue
            o = {"reachable": bool(rec["reachable"]), "delta": rec.get("delta"), "n_at_cap": rec.get("n_edits_at_cap")}
            if rows:
                e = [r for r in rows if r["kind"] == "edit"]
                t = [r for r in rows if r["kind"] == "hn" and r.get("part") != "gen"]
                g = [r for r in rows if r["kind"] == "hn" and r.get("part") == "gen"]
                p = [r for r in rows if r["kind"] == "probe"]
                ex = {m: pct([bool(r["exact"]) for r in e if r["metric"] == m]) for m in ("rel", "tgen", "igen")}
                cv = {m: pct([bool(r["own_released_last"]) for r in e if r["metric"] == m]) for m in ("tgen", "igen")}
                o.update(exact=0.5 * (ex["tgen"] + ex["igen"]), exact_tgen=ex["tgen"], exact_igen=ex["igen"], rel=ex["rel"], cov=0.5 * (cv["tgen"] + cv["igen"]),
                         hd_fired=pct([bool(r["fired"]) for r in p]), hd_D1=pct([bool(r["D1"]) for r in p]), n_odd_edits=len(rec["odd_edits"]))
                ufs = set((rec.get("unseen_foreign") or {}).get("csids") or [])
                for k in HN:
                    tk, gk = [r for r in t if r["hn_type"] == k], [r for r in g if r["hn_type"] == k]
                    if tk:
                        o[f"test.{k}.n"], o[f"test.{k}.fired"], o[f"test.{k}.D1"] = len(tk), pct([bool(r["fired"]) for r in tk]), pct([bool(r["D1"]) for r in tk])
                        o[f"test.{k}.wrong_target"] = pct([bool(r["wrong_target"]) for r in gk]) if gk else None
                        if ufs:
                            tu = [r for r in tk if r["csid"] in ufs]
                            o[f"testu.{k}.n"] = len(tu)
                            o[f"testu.{k}.fired"] = pct([bool(r["fired"]) for r in tu]) if tu else None
                            o[f"testu.{k}.wrong_target"] = pct([bool(r["wrong_target"]) for r in gk if r["csid"] in ufs]) if tu else None
                o["test.mean.fired"] = mean([o.get(f"test.{k}.fired") for k in HN])
            per[d] = o
        if per:
            names = sorted({k for o in per.values() for k in o})
            h = {nm: {"mean": mean([float(o[nm]) for o in per.values() if o.get(nm) is not None]), "per_draw": {str(d): o.get(nm) for d, o in per.items()}, "n_draws": len(per)} for nm in names}
            S[log][lab]["reg"]["h3_beh"] = h
            for nm in names:
                for d, o in per.items():
                    if o.get(nm) is not None:
                        R.add(log, lab, "b.h3." + nm, float(o[nm]), draw=d, alpha=ALPHA, n=N, pool="h3")


# ----------------------------------------------------------------------------- series, development-log check, paper numbers
def gv(S, log, lab, ro, level, nm, f="mean"):
    try:
        return S[log][lab][ro][level][nm][f]
    except (KeyError, TypeError):
        return None


def series(S):
    sc, rt = {}, {}
    for fam, logs in FAMILIES.items():
        labs = sorted({lab for log in logs for lab in S.get(log, {}) if not lab.startswith("_")})
        for lab in labs:
            pts = []
            for log in logs:
                if lab not in S.get(log, {}):
                    continue
                pt = {"log": log, "T": S[log]["_meta"]["T"]}
                for ro in ("reg", "qd"):
                    nd = S[log][lab].get(ro) or {}
                    if nd.get("not_defined") or nd.get("skipped_for_cost"):
                        pt[ro] = {"not_run": nd.get("not_defined") or nd.get("skipped_for_cost")}
                        continue
                    q = "_qd" if ro == "qd" else ""
                    pt[ro] = {"Rel": gv(S, log, lab, ro, "beh", "edit.rel.exact"), "T-Gen": gv(S, log, lab, ro, "beh", "edit.tgen.exact"), "I-Gen": gv(S, log, lab, ro, "beh", "edit.igen.exact"),
                              "coverage_tgen": gv(S, log, lab, ro, "gate", "cov.tgen"), "coverage_igen": gv(S, log, lab, ro, "gate", "cov.igen"),
                              "false_firing_hd": gv(S, log, lab, ro, "gate", "fire.hd" + q), "false_firing_hd_ci": [gv(S, log, lab, ro, "gate", "fire.hd" + q, "ci_lo"), gv(S, log, lab, ro, "gate", "fire.hd" + q, "ci_hi")],
                              "D1_hd": gv(S, log, lab, ro, "beh", "probe.D1" + (".qd" if ro == "qd" else "")), "near_scope_D1": gv(S, log, lab, ro, "beh", "hn.all.D1"),
                              "near_scope_fired": gv(S, log, lab, ro, "gate", "hn.mean.fired"), "n_draws_gate": nd.get("n_draws_gate"), "n_draws_behaviour": nd.get("n_draws_beh")}
                pts.append(pt)
            sc.setdefault(fam, {})[lab] = pts
        for lab in ("deployed", "where_logonly", "rescope"):
            rt.setdefault(fam, {})[lab] = [{"log": log, "T": S[log]["_meta"]["T"], **{m: gv(S, log, lab, "reg", "beh", f"edit.{m}.exact" + ("" if S[log]["_meta"]["T"] == 100 else ".first100")) for m in ("rel", "tgen", "igen")},
                                            "n_draws": (S[log].get(lab, {}).get("reg") or {}).get("n_draws_beh"),
                                            "not_run": (S[log].get(lab, {}).get("reg") or {}).get("skipped_for_cost")} for log in logs if log in S and "_meta" in S[log]]
    jdump({"what": "scale series per log family and method: registered read-out (reg) and the read-out with calibration draws and probe rows restricted to rows whose question string is "
                   "not a question or rephrase of any edit of the longest stream (qd; false firing and D1 on the filtered heldout_disjoint rows).  Rel / T-Gen / I-Gen exact, D1 and "
                   "near-scope D1: behaviour (mean over the behavioural draws); coverage, false firing, near-scope firing: gate level (mean over 20 draws; interval: crossed bootstrap)",
           "families": sc}, os.path.join(OUT, "scale_series.json"))
    jdump({"what": "retention (E9 c): Rel / T-Gen / I-Gen exact of the FIRST 100 EDITS of the stream as T grows; deployed log, thresholds-only repair with log-only thresholds, default method "
                   "(mean over the behavioural draws)", "families": rt}, os.path.join(OUT, "retention_series.json"))


def dev_check(S, GD):
    """The development log against the sanity run e0 (same cells as e1_main/code/e1_sanity.py)."""
    p = FINAL + "/e0_sanity/out/results.json"
    if DEV not in S or not os.path.exists(p):
        return
    e0 = jload(p)["logs"][DEV]
    MAP = {"M1": "rescope", "M2": "rescope_shared", "M3": "cosine", "M4": "cosine_shared", "M5": "cosine_default", "M6": "exact", "deployed": "deployed"}
    cells = []
    for m0, lab in MAP.items():
        g = (e0["gate"]["methods"].get(m0) or {}).get("tf")
        if g and lab in GD.get(DEV, {}):
            for path, mine in ((("probe", "heldout_disjoint", "fired"), "fire.hd"), (("edit", "tgen", "own_top"), "cov.tgen"), (("edit", "igen", "own_top"), "cov.igen"),
                               (("hn", "HN-I", "fired"), "hn.HN-I.fired"), (("hn", "HN-Q", "fired"), "hn.HN-Q.fired"), (("hn", "HN-X", "fired"), "hn.HN-X.fired")):
                v = g
                for k in path:
                    v = v[k]
                th = v["values"] if isinstance(v, dict) else [v]
                ou = GD[DEV][lab]["reg"][mine][:len(th)]
                cells.append({"level": "gate", "method": lab, "metric": mine, "n_draws": len(th), "max_abs_diff": float(np.max(np.abs(np.array(th, dtype=float) - np.array(ou, dtype=float)))),
                              "e0_mean": float(np.mean(th)), "e3_mean": float(np.mean(ou))})
        b = e0["behaviour"]["methods"].get(m0)
        if b is None:
            continue
        per = {"None": b["single"]} if "single" in b else {str(k): v for k, v in b["draws"].items()}
        for path, mine in ((("edit", "rel", "exact"), "edit.rel.exact"), (("edit", "tgen", "exact"), "edit.tgen.exact"), (("edit", "igen", "exact"), "edit.igen.exact"),
                           (("probe", "heldout_disjoint", "D1"), "probe.D1"), (("hn", "HN-I", "D1"), "hn.HN-I.D1"), (("hn", "HN-Q", "D1"), "hn.HN-Q.D1"), (("hn", "HN-X", "D1"), "hn.HN-X.D1")):
            mine_pd = gv(S, DEV, lab, "reg", "beh", mine, "per_draw") or {}
            th, ou = [], []
            for d, v in per.items():
                for k in path:
                    v = v[k]
                if d in mine_pd:
                    th.append(float(v))
                    ou.append(float(mine_pd[d]))
            if th:
                cells.append({"level": "behaviour", "method": lab, "metric": mine, "n_draws": len(th), "max_abs_diff": float(np.max(np.abs(np.array(th) - np.array(ou)))),
                              "e0_mean": float(np.mean(th)), "e3_mean": float(np.mean(ou))})
    out = {"what": "development log TB-L100-ae-l2_3: this experiment's numbers against the sanity run e0_sanity/out/results.json (gate level: every draw; behaviour: draws 0 - 2)",
           "n_cells": len(cells), "n_cells_differ_above_1e-9": sum(c["max_abs_diff"] > 1e-9 for c in cells), "largest_abs_diff": max([c["max_abs_diff"] for c in cells], default=None),
           "differing": [c for c in cells if c["max_abs_diff"] > 1e-9], "cells": cells}
    jdump(out, os.path.join(OUT, "dev_check.json"))
    S["_dev_check"] = {k: out[k] for k in ("n_cells", "n_cells_differ_above_1e-9", "largest_abs_diff", "differing")}


TW = {100: "Hundred", 200: "TwoHundred", 500: "FiveHundred", 1000: "Thousand"}
PAPER = (("Rel", "beh", "edit.rel.exact", "Rel exact, all edits (percent; behaviour, mean over the behavioural draws)"),
         ("TGen", "beh", "edit.tgen.exact", "T-Gen exact, all edits"), ("IGen", "beh", "edit.igen.exact", "I-Gen exact, all edits"),
         ("RelLater", "beh", "edit.rel.exact.later", "Rel exact, later edits only (stream position >= 100)"), ("TGenLater", "beh", "edit.tgen.exact.later", "T-Gen exact, later edits only"),
         ("IGenLater", "beh", "edit.igen.exact.later", "I-Gen exact, later edits only"), ("RephLater", "beh", "edit.reph.exact.later", "(T-Gen + I-Gen) / 2 exact, later edits only"),
         ("Reph", "beh", "edit.reph.exact", "(T-Gen + I-Gen) / 2 exact, all edits (H2 form)"),
         ("DOne", "beh", "probe.D1", "D1 on heldout_disjoint (percent of valid rows)"), ("DOneNear", "beh", "probe.D1.near", "D1 on near-scope heldout_disjoint probes"),
         ("HnDOne", "beh", "hn.all.D1", "D1 on the hard negatives of the log's edits (all kinds)"),
         ("CovT", "gate", "cov.tgen", "coverage of text rephrases (own edit released; gate level, mean over 20 draws)"), ("CovI", "gate", "cov.igen", "coverage of image rephrases (gate level, 20 draws)"),
         ("CovTLater", "gate", "cov.tgen.later", "coverage of text rephrases, later edits only (gate level, 20 draws)"), ("CovILater", "gate", "cov.igen.later", "coverage of image rephrases, later edits only"),
         ("Fire", "gate", "fire.hd", "false firing on heldout_disjoint (gate level, mean over 20 draws)"), ("FireNear", "gate", "fire.hd_near", "firing on near-scope heldout_disjoint probes (gate level)"),
         ("FireFar", "gate", "fire.hd_far", "firing on the other heldout_disjoint probes (gate level)"), ("FireLiteral", "gate", "fire.hd.literal", "false firing of the literal rule without guard caps"),
         ("HnFireI", "gate", "hn.HN-I.fired", "firing on HN-I (gate level)"), ("HnFireQ", "gate", "hn.HN-Q.fired", "firing on HN-Q (gate level)"), ("HnFireX", "gate", "hn.HN-X.fired", "firing on HN-X (gate level)"),
         ("BenT", "gate", "BEN-T.far.fired", "false firing on BEN-T (prompt-only keys; rows related to the family excluded; gate level)"),
         ("MmeT", "gate", "MME-T.fired", "false firing on MME-T (out-of-benchmark workload; gate level)"), ("Capped", "gate", "n_capped", "edits at their guard cap (mean over 20 draws)"),
         ("HthreeCov", "gate", "h3.cov", "H3: coverage of (T-Gen + I-Gen) / 2 on the odd edits under the near-scope constraint (gate level, 20 draws; unreachable draws count 0)"))


def paper(R, S):
    num, doc = {}, {}
    for log in LOGS:
        if log not in S or "_meta" not in S[log]:
            continue
        T = S[log]["_meta"]["T"]
        pre = "Ethree" + "L" + ("Ae" if family_of(log) == "ae-l2_3" else "Me") + TW[T]
        for lab in S[log]:
            if lab.startswith("_"):
                continue
            for short, level, nm, text in PAPER:
                v = gv(S, log, lab, "reg", level, nm)
                if v is not None:
                    k = pre + camel(lab) + short
                    num[k], doc[k] = v, f"{log}, {lab}: {text}"
        fa = S[log].get("_factor")
        if fa:
            num[pre + "NonOracle"], doc[pre + "NonOracle"] = fa["n_non_oracle"], f"{log}: edits whose target is not an oracle (cannot become exact under a value-releasing method)"
            num[pre + "MarginPositive"], doc[pre + "MarginPositive"] = fa["margin"]["n_positive"], f"{log}: edits with a positive self-separation margin of the solver keys"
            num[pre + "MarginMin"], doc[pre + "MarginMin"] = fa["margin"]["min"], f"{log}: smallest self-separation margin"
            if fa["rebase"]:
                num[pre + "RebaseErrMax"], doc[pre + "RebaseErrMax"] = fa["rebase"]["rebased"]["max"], f"{log}: largest relative error of the rebased values against the cached targets"
                num[pre + "LoggedErrMax"], doc[pre + "LoggedErrMax"] = fa["rebase"]["not_rebased"]["max"], f"{log}: largest relative error of the logged values (no rebasing) against the cached targets"
    p = os.path.join(OUT, "paper_numbers.json")
    if os.path.exists(p):
        os.remove(p)                                                # rebuilt from scratch at every run (no stale name)
    R.paper(num)
    jdump(dict(sorted(doc.items())), os.path.join(OUT, "paper_numbers_doc.json"))


def main():
    t0 = time.time()
    R = fx.io.Results("E3", OUT)
    S, GD = {}, {}
    for log in LOGS:
        meta = gate_level(R, log, S, GD)
        if meta is not None:
            behaviour(R, log, S, meta)
        print(f"[{time.time() - t0:5.0f}s] {log}", flush=True)
    dev_check(S, GD)
    series(S)
    jobs = {log: [jload(p).get("assert_clean") for p in sorted(glob.glob(os.path.join(BEH, log, "_job_*.json")))] for log in LOGS}
    fails = sorted(os.path.basename(p) for log in LOGS for p in glob.glob(os.path.join(BEH, log, "*__FAILED*.json")))
    S["_run"] = {"jobs_assert_clean": jobs, "failed_cells": fails, "n_missing": len(MISSING), "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    jdump(S, os.path.join(OUT, "summary.json"))
    jdump({"what": "every gate-level number of every calibration draw: draws[log][method][readout][metric] = list over draws 0 .. 19", "draws": GD}, os.path.join(OUT, "gate_draws.json"))
    jdump(MISSING, os.path.join(OUT, "missing.json"))
    R.save(alpha=ALPHA, n=N, logs=list(LOGS), families=FAMILIES, experiment="E3 (scale) and E4 (other editor) on the LLaVA-1.5-7B logs",
           n_draws_gate=N_DRAWS, behavioural_draws={log: list(bdraws(S[log]["_meta"]["T"])) for log in LOGS if log in S and "_meta" in S[log]})
    paper(R, S)
    print(f"[{time.time() - t0:5.0f}s] records {len(R.records)} missing {len(MISSING)}", flush=True)


if __name__ == "__main__":
    main()
