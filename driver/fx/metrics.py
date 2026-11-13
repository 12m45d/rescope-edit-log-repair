"""fx.metrics: the ONE definition of every reported number, computed from per-row flags.

  gate_stats(L, ev, which)        gate level (fx.gate.evaluate): firing on probe pools, coverage on edit sequences, hard negatives,
                                  the benign pool, lineage widths
  edit_stats(rows)                Rel / T-Gen / I-Gen: exact, token accuracy, share in scope (own edit releases), rephrase mean
  probe_stats(rows)               firing rate, D1 rate, D1 among firing rows, token-level locality T-Loc* / I-Loc*; near-scope split
  hn_stats(rows, gen_rows)        hard negatives per type: firing, D1, own edit releases, wrong-target rate (generated answers)
  d2_stats(rows), gen_stats(rows) D2 count; free generation of edit prompts (eq / contains / degenerate)
  endpoint(rows_a, rows_b)        primary endpoint of E1: (T-Gen + I-Gen) / 2 exact of a versus b, paired bootstrap over edits; McNemar
All rates are percentages of rows.  ci=True turns every rate into {value, ci_lo, ci_hi, n_units, ...} (fx.stats.bootstrap: 2000
resamples; unit = query cluster (query_id: identical queries are one unit; protocol amendment R10) for probe rows, edit for edit
rows and hard negatives).  Rows: fx.behave rows (the library's own rows have no query_id: pass unit='uid').
"""
from __future__ import annotations

import numpy as np

from . import stats

HN_TYPES = ("HN-I", "HN-Q", "HN-X")
POOLS = {"heldout_disjoint": lambda r: bool(r.get("disjoint")), "heldout_strict": lambda r: bool(r.get("strict")),
         "heldout": lambda r: r.get("pool") == "heldout", "repair": lambda r: r.get("pool") == "repair"}


def pct(flags):
    """Percentage of true flags; None for an empty list or if a flag is None (methods that are not gates have no firing flags)."""
    flags = list(flags)
    if not flags or any(x is None for x in flags):
        return None
    return 100.0 * sum(bool(x) for x in flags) / len(flags)


def mean100(values):
    v = [x for x in values if x is not None]
    return 100.0 * float(np.mean(v)) if v else None


def _rate(flags, units, ci):
    if not ci:
        return pct(flags)
    flags = list(flags)
    if not flags or any(x is None for x in flags):
        return None
    return stats.bootstrap([float(bool(x)) for x in flags], units, scale=100.0)


def _mean(values, units, ci):
    if not ci:
        return mean100(values)
    return stats.bootstrap(list(values), units, scale=100.0)


# ----------------------------------------------------------------------------- gate level
def gate_stats(L, ev: dict, which: str = "tf") -> dict:
    """Numbers of one gate-level pass (fx.gate.evaluate(L, M)) at position `which`:
      probe.<pool>            n, fired = % of valid probe rows of the pool on which an edit fires; pools heldout_disjoint (reporting),
                              heldout_strict, heldout, repair; heldout_disjoint_image / _text; heldout_disjoint_near / _far = rows whose
                              question string is / is not the question of an edit of the log (near-scope probes, LogData.near_edit)
      probe.lineage           width = number of candidate edits per firing HELD-OUT-DISJOINT row: n_firing, mean, max, % with more than one
      edit.<metric>           n, fired, own_cand (the own edit is a candidate), own_top = coverage (the own edit releases), other_top, lineage
      hn.<type>, hn.all       n, fired, own_cand, own_top; HN-X: b_top (the edit whose question the row uses releases), own_or_b_top; hn.all.lineage
      benign                  n, fired (prompt-only key), by ptype; benign_near / benign_far as for probes"""
    from . import gate as G
    out = {}
    if "probe" in ev and which in ev["probe"]:
        d = ev["probe"][which]
        inf = G.info(L, d["rows"])
        f, nc = np.asarray(d["fired"]), np.asarray(d["n_cand"])
        sel = lambda p: np.array([i for i, r in enumerate(inf) if p(r)], dtype=np.int64)           # noqa: E731
        pr = {nm: {"n": int(sel(p).size), "fired": pct(f[sel(p)])} for nm, p in POOLS.items()}
        dj = POOLS["heldout_disjoint"]
        for nm, p in (("image", lambda r: r["ptype"] == "image"), ("text", lambda r: r["ptype"] == "text"),
                      ("near", lambda r: r["near_edit"] is not None), ("far", lambda r: r["near_edit"] is None)):
            idx = sel(lambda r: dj(r) and p(r))
            pr["heldout_disjoint_" + nm] = {"n": int(idx.size), "fired": pct(f[idx])}
        pr["lineage"] = _width(nc[sel(dj)], f[sel(dj)])
        out["probe"] = pr
    if "edit" in ev and which in ev["edit"]:
        d = ev["edit"][which]
        kinds = np.array([L.index[j]["kind"] for j in d["rows"]])
        f, oc, ot, nc = (np.asarray(d[k]) for k in ("fired", "own_cand", "own_top", "n_cand"))
        out["edit"] = {m: {"n": int((kinds == m).sum()), "fired": pct(f[kinds == m]), "own_cand": pct(oc[kinds == m]), "own_top": pct(ot[kinds == m]),
                           "other_top": pct(f[kinds == m] & ~ot[kinds == m]), "lineage": _width(nc[kinds == m], f[kinds == m])} for m in ("rel", "tgen", "igen")}
    if "hn" in ev and which in ev["hn"]:
        d = ev["hn"][which]
        inf = G.info(L, d["rows"])
        f, oc, ot, top = (np.asarray(d[k]) for k in ("fired", "own_cand", "own_top", "top_edit"))
        hn = {}
        for t in HN_TYPES:
            idx = np.array([i for i, r in enumerate(inf) if r["hn_type"] == t], dtype=np.int64)
            h = {"n": int(idx.size), "fired": pct(f[idx]), "own_cand": pct(oc[idx]), "own_top": pct(ot[idx])}
            if t == "HN-X":
                eb = np.array([inf[i]["edit_b"] for i in idx], dtype=np.int64)
                bt = f[idx] & (top[idx] == eb)
                h.update(b_top=pct(bt), own_or_b_top=pct(ot[idx] | bt))
            hn[t] = h
        hn["all"] = {"n": len(inf), "fired": pct(f), "lineage": _width(np.asarray(d["n_cand"]), f)}
        out["hn"] = hn
    if "benign" in ev:
        d = ev["benign"]["decode"]
        inf = G.info(L, d["rows"])
        f = np.asarray(d["fired"])
        b = {"n": len(inf), "fired": pct(f), "which": "decode"}
        for nm, p in (("image", lambda r: r["ptype"] == "image"), ("text", lambda r: r["ptype"] == "text"),
                      ("near", lambda r: r["near_edit"] is not None), ("far", lambda r: r["near_edit"] is None)):
            idx = np.array([i for i, r in enumerate(inf) if p(r)], dtype=np.int64)
            b[nm] = {"n": int(idx.size), "fired": pct(f[idx])}
        out["benign"] = b
    t1 = top1_report(ev, which)
    if t1 is not None:                                                  # only with fx.gate.evaluate(..., top1=True): the default output is unchanged
        out["top1"] = t1
    return out


def top1_report(ev: dict, which: str = "tf"):
    """TOP-1 SELECTION WITH CAPPED EDITS -- the definition used by every experiment is the library's: the candidates are the
    detectors whose coordinate passes its (capped) threshold, and the best candidate by the ranking key releases ("threshold
    first, argmax second").  The other order, "argmax first, threshold second" (the first-ranked detector of ALL releases iff it
    is a candidate), agrees whenever no threshold was capped below its calibrated level; this read-out counts, per part of a
    gate-level pass made with fx.gate.evaluate(..., top1=True), the sequences on which the two orders decide differently
    (n_differs; of them n_library_fires_only = the library releases a (capped) edit where the other order releases nothing,
    n_other_edit = both release, different edits).  None if the pass carries no such arrays."""
    out, tot = {}, {"n": 0, "n_differs": 0}
    for part, by in ev.items():
        d = by.get("decode" if part == "benign" else which)
        if d is None or "top1_differs" not in d:
            continue
        df, f, af = np.asarray(d["top1_differs"], dtype=bool), np.asarray(d["fired"], dtype=bool), np.asarray(d["alt_fired"], dtype=bool)
        out[part] = {"n": int(df.size), "n_differs": int(df.sum()), "n_library_fires_only": int((df & f & ~af).sum()), "n_other_edit": int((df & f & af).sum())}
        tot["n"] += int(df.size)
        tot["n_differs"] += int(df.sum())
    if not out:
        return None
    out["all"] = tot
    return out


def _width(n_cand, fired) -> dict:
    """Lineage width = number of candidate edits of a firing query: n_firing, mean, max, share of firing queries with more than one."""
    w = np.asarray(n_cand)[np.asarray(fired, dtype=bool)]
    return {"n_firing": int(w.size), "width_mean": float(w.mean()) if w.size else None, "width_max": int(w.max()) if w.size else None,
            "multi": 100.0 * float((w > 1).mean()) if w.size else None}


def leaves(d, prefix=()) -> dict:
    """Flatten a nested dict of numbers: {(path): value}."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and not ("value" in v and "n_units" in v):
            out.update(leaves(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def over_draws(per_draw: list) -> dict:
    """Per-draw nested statistics -> {(path): fx.stats.over_draws(...)} = mean, 5th / 95th percentile, min, max, sd over the draws.
    A number that is the same in every draw (row counts such as n) is kept as that number."""
    flat = [leaves(d) for d in per_draw]
    out = {}
    for p in flat[0]:
        vals = [f.get(p) for f in flat]
        same = all(v == vals[0] for v in vals) and (vals[0] is None or isinstance(vals[0], (int, str)) or str(p[-1]) == "n")
        out[p] = vals[0] if same else stats.over_draws(vals)
    return out


# ----------------------------------------------------------------------------- behaviour
def edit_stats(rows, ci: bool = False) -> dict:
    """Per metric: n, exact (% of edits whose target is reproduced token by token), token (mean token accuracy, %), fired, own_cand
    (own edit is a candidate), own_top (own edit releases = the sequence is in scope), n_nonfinite; 'rephrase': exact = mean over
    edits of (T-Gen exact + I-Gen exact) / 2 (the primary endpoint of E1)."""
    out = {}
    for m in ("rel", "tgen", "igen"):
        sub = [r for r in rows if r.get("kind") == "edit" and r.get("metric") == m]
        if not sub:
            continue
        u = [r["edit"] for r in sub]
        out[m] = {"n": len(sub), "exact": _rate([r["exact"] for r in sub], u, ci), "token": _mean([r["token_acc"] for r in sub], u, ci),
                  "fired": pct(r.get("fired") for r in sub), "own_cand": pct(r.get("own_fired_last") for r in sub),
                  "own_top": pct(r.get("own_released_last") for r in sub), "n_nonfinite": sum(not r["finite"] for r in sub)}
    pe = rephrase_exact(rows)
    if pe:
        ed = sorted(pe)
        out["rephrase"] = {"n": len(ed), "exact": _mean([pe[e] for e in ed], ed, ci)}
    return out


def rephrase_exact(rows) -> dict:
    """{edit: (T-Gen exact + I-Gen exact) / 2} for the edits that have both rows."""
    t = {r["edit"]: float(bool(r["exact"])) for r in rows if r.get("kind") == "edit" and r.get("metric") == "tgen"}
    i = {r["edit"]: float(bool(r["exact"])) for r in rows if r.get("kind") == "edit" and r.get("metric") == "igen"}
    return {e: 0.5 * (t[e] + i[e]) for e in t if e in i}


def probe_stats(rows, ci: bool = False, unit: str = "query_id") -> dict:
    """Valid probe rows of ONE pool: n, fired, n_fired, D1 (% of rows whose pre-edit answer is no longer reproduced), n_D1,
    D1_given_fired, T-Loc* / I-Loc* (token-level agreement with the pre-edit argmax, % over text / image rows; a row with a
    non-finite logit counts 0), tok_loc (all rows), n_nonfinite; n_y0_degenerate, D1_y0_not_degenerate, D1_y0_degenerate (rows whose
    pre-edit answer is empty / one character / '?' / a bare None-No); near / far: the same counts for rows that share / do not
    share a question string with an edit of the log.  unit (used only with ci=True): the bootstrap unit, 'query_id' (the query
    cluster: rows with the same question string and image content are resampled together; the unit of amendment R10) or 'uid'
    (the probe item: its text and its image probe are resampled together; the unit of the protocol before R10)."""
    def one(sub, full=True):
        if not sub:
            return {"n": 0}
        un = (lambda rr: [r[unit] for r in rr]) if ci else (lambda rr: None)                       # noqa: E731
        u = un(sub)
        gate = sub[0].get("fired") is not None
        fired = [r for r in sub if r.get("fired")]
        o = {"n": len(sub), "fired": _rate([r.get("fired") for r in sub], u, ci) if gate else None, "n_fired": len(fired) if gate else None,
             "D1": _rate([r["D1"] for r in sub], u, ci), "n_D1": sum(bool(r["D1"]) for r in sub),
             "D1_given_fired": pct(r["D1"] for r in fired) if gate and fired else None}
        if full:
            tx, im = [r for r in sub if r["ptype"] == "text"], [r for r in sub if r["ptype"] == "image"]
            o.update({"T-Loc*": _mean([r["tok_loc"] for r in tx], un(tx), ci) if tx else None,
                      "I-Loc*": _mean([r["tok_loc"] for r in im], un(im), ci) if im else None,
                      "tok_loc": _mean([r["tok_loc"] for r in sub], u, ci), "n_text": len(tx), "n_image": len(im),
                      "n_nonfinite": sum(not r["finite"] for r in sub), **_deg(sub)})
        return o
    pr = [r for r in rows if r.get("kind") == "probe"]
    if any(not r.get("y0_valid") for r in pr):
        raise ValueError("probe_stats: D1 and token locality are defined on rows with a valid pre-edit answer")
    out = one(pr)
    if pr and "near_edit" in pr[0]:
        out["near"] = one([r for r in pr if r["near_edit"] is not None], full=False)
        out["far"] = one([r for r in pr if r["near_edit"] is None], full=False)
    return out


def _deg(sub) -> dict:
    """Rows whose pre-edit answer is degenerate (y0_deg: empty, one character, '?', bare None / No) are counted, and D1 is also given
    without them and on them (rows without the flag, e.g. the library's own rows: {})."""
    if not sub or "y0_deg" not in sub[0]:
        return {}
    deg, rest = [r for r in sub if r["y0_deg"]], [r for r in sub if not r["y0_deg"]]
    return {"n_y0_degenerate": len(deg), "D1_y0_not_degenerate": pct(r["D1"] for r in rest), "D1_y0_degenerate": pct(r["D1"] for r in deg)}


def hn_stats(rows, gen_rows=None, ci: bool = False) -> dict:
    """Hard negatives per type (HN-I, HN-Q, HN-X) and 'all': n, fired, D1, own_top (the row's own edit releases), tok_loc,
    n_nonfinite; with generation rows (fx.behave.hard_negatives(generate=True)): wrong_target (% of rows whose generated answer
    equals the target of the firing edit), eq_own, eq_any_edit, n_decoded."""
    out = {}
    hn = [r for r in rows if r.get("kind") == "hn"] if rows is not None else []
    for t in HN_TYPES + ("all",):
        sub = [r for r in hn if t == "all" or r["hn_type"] == t]
        o = {}
        if sub:
            u = [r["edit"] for r in sub]
            o = {"n": len(sub), "fired": _rate([r.get("fired") for r in sub], u, ci) if sub[0].get("fired") is not None else None,
                 "D1": _rate([r["D1"] for r in sub], u, ci), "own_top": pct(r.get("own_released_last") for r in sub),
                 "tok_loc": _mean([r["tok_loc"] for r in sub], u, ci), "n_nonfinite": sum(not r["finite"] for r in sub), **_deg(sub)}
        if gen_rows is not None:
            g = [r for r in gen_rows if t == "all" or r["hn_type"] == t]
            if g:
                u = [r["edit"] for r in g]
                wt = [r.get("wrong_target") for r in g]
                o.update(n_gen=len(g), n_decoded=sum("line" in r for r in g), wrong_target=_rate(wt, u, ci) if all(x is not None for x in wt) else None,
                         eq_own=pct(bool(r.get("eq_own")) for r in g if "line" in r) if any("line" in r for r in g) else None,
                         eq_any_edit=pct(bool(r.get("eq_any_edit")) for r in g if "line" in r) if any("line" in r for r in g) else None,
                         fired_decode=pct(r.get("fired") for r in g))
        if o:
            out[t] = o
    return out


def d2_stats(rows, ci: bool = False, unit: str = "query_id") -> dict:
    """Pre-correct probe rows under a 48-token decode: n_pre_correct, n_D2 (the reference is no longer contained), D2 = the RATE
    of the pre-correct rows (percent; amendment R7), fired.  ci=True: D2 becomes {value, ci_lo, ci_hi, n_units, ...} (percentile
    bootstrap, 2000 resamples, unit = query cluster `unit`; R7: "D2 as a rate of the pre-correct rows with interval").  Over
    several calibration draws use fx.metrics.grid(rows_by_draw, "D2", where=lambda r: r.get("pre_correct")) + fx.stats.crossed_bootstrap."""
    sub = [r for r in rows if r.get("pre_correct")]
    out = {"n_pre_correct": len(sub), "n_D2": sum(bool(r["D2"]) for r in sub), "D2": pct(r["D2"] for r in sub), "fired": pct(r.get("fired") for r in sub)}
    if ci:
        out["D2"] = _rate([r["D2"] for r in sub], [r.get(unit, i) for i, r in enumerate(sub)], True) if sub else None
    return out


def kept_gained_rows(rows, rows_deployed) -> list:
    """Per edit sequence (Rel / T-Gen / I-Gen): {metric, edit, exact, exact_deployed, kept, gained, lost} with kept = exact under
    the deployed log AND under the method, gained = not exact under the deployed log, exact under the method, lost = exact under
    the deployed log, not under the method (amendment R7).  rows: edit rows of the method; rows_deployed: edit rows of the
    deployed log (fx.behave: ses.edits(fx.methods.deployed(L))) on the same edits (a missing edit is an error)."""
    dep = {(r["metric"], r["edit"]): bool(r["exact"]) for r in rows_deployed if r.get("kind") == "edit"}
    out = []
    for r in rows:
        if r.get("kind") != "edit":
            continue
        k = (r["metric"], r["edit"])
        if k not in dep:
            raise KeyError(f"no row of the deployed log for {k}")
        m, d = bool(r["exact"]), dep[k]
        out.append({"kind": "edit", "metric": r["metric"], "edit": r["edit"], "exact": m, "exact_deployed": d, "kept": m and d, "gained": m and not d, "lost": d and not m})
    return out


def kept_gained(rows, rows_deployed, ci: bool = False) -> dict:
    """Read-out of amendment R7 per metric (rel, tgen, igen): n, n_deployed_exact, n_exact, n_kept, n_gained, n_lost, and the
    rates kept / gained / lost in percent OF ALL n EDITS (kept + lost = the deployed log's exact rate; kept + gained = the
    method's), kept_of_deployed = n_kept / n_deployed_exact (percent).  ci=True: the three rates with bootstrap intervals over
    edits.  Over calibration draws: fx.metrics.grid(rows_by_draw, "kept", ...) on kept_gained_rows + fx.stats.crossed_bootstrap."""
    kg = kept_gained_rows(rows, rows_deployed)
    out = {}
    for m in ("rel", "tgen", "igen"):
        sub = [r for r in kg if r["metric"] == m]
        if not sub:
            continue
        u = [r["edit"] for r in sub]
        nd, nk = sum(r["exact_deployed"] for r in sub), sum(r["kept"] for r in sub)
        out[m] = {"n": len(sub), "n_deployed_exact": nd, "n_exact": sum(r["exact"] for r in sub), "n_kept": nk, "n_gained": sum(r["gained"] for r in sub),
                  "n_lost": sum(r["lost"] for r in sub), "kept": _rate([r["kept"] for r in sub], u, ci), "gained": _rate([r["gained"] for r in sub], u, ci),
                  "lost": _rate([r["lost"] for r in sub], u, ci), "kept_of_deployed": 100.0 * nk / nd if nd else None}
    return out


def grid(rows_by_draw, value, unit: str = "query_id", where=None) -> dict:
    """The (draws, units) matrices of the crossed bootstrap (fx.stats.crossed_bootstrap, R10) from behavioural rows per
    calibration draw.  rows_by_draw: {draw: rows} (or a list of row lists); value: a row field (flag / number) or a function of a
    row; unit: the row field that names the unit ('query_id' for probe rows: the query cluster; 'edit' for edit rows and hard
    negatives) or a function; where: predicate selecting the rows.  -> {num (D, U) sum of the values of the unit's rows, den
    (D, U) number of its rows, units, draws}: fx.stats.crossed_bootstrap(g["num"], g["den"], scale=100) is the rate over rows with
    its interval; for two methods on the same rows fx.stats.crossed_bootstrap_diff(ga["num"], gb["num"], ga["den"], gb["den"]).
    Every draw must hold the same units (a unit without rows in some draw is an error: the grid would not be crossed)."""
    items = list(rows_by_draw.items()) if isinstance(rows_by_draw, dict) else list(enumerate(rows_by_draw))
    val = value if callable(value) else (lambda r: r[value])
    un = unit if callable(unit) else (lambda r: r[unit])
    per, units = [], None
    for d, rows in items:
        acc = {}
        for r in rows:
            if where is not None and not where(r):
                continue
            v = val(r)
            if v is None:
                raise ValueError(f"draw {d}: a row without a value")
            a = acc.setdefault(un(r), [0.0, 0])
            a[0] += float(v)
            a[1] += 1
        if units is None:
            units = sorted(acc, key=str)
        elif set(acc) != set(units):
            raise ValueError(f"draw {d} holds other units than the first draw ({len(acc)} vs {len(units)})")
        per.append(acc)
    if not per or not units:
        raise ValueError("no rows")
    return {"num": np.array([[p[u][0] for u in units] for p in per]), "den": np.array([[float(p[u][1]) for u in units] for p in per]),
            "units": units, "draws": [d for d, _ in items]}


def gen_stats(rows) -> dict:
    """Free generation of edit prompts per metric: n, eq (first line equals the target after normalisation), contains, starts,
    n_degenerate, fired."""
    out = {}
    for m in ("rel", "tgen", "igen"):
        sub = [r for r in rows if r.get("kind") == "edit" and r.get("metric") == m and "eq" in r]
        if sub:
            out[m] = {"n": len(sub), "eq": pct(r["eq"] for r in sub), "contains": pct(r["contains"] for r in sub), "starts": pct(r["starts"] for r in sub),
                      "n_degenerate": sum(any(r["degenerate"].values()) for r in sub), "fired": pct(r.get("fired") for r in sub)}
    return out


def endpoint(rows_a, rows_b) -> dict:
    """Primary endpoint of E1 for two methods on the same edits: (T-Gen + I-Gen) / 2 exact of a minus b, paired bootstrap over
    edits (points), and per metric the exact McNemar test of the exact flags."""
    a, b = rephrase_exact(rows_a), rephrase_exact(rows_b)
    ed = sorted(set(a) & set(b))
    out = {"rephrase": stats.bootstrap_diff([a[e] for e in ed], [b[e] for e in ed], ed, scale=100.0)}
    for m in ("rel", "tgen", "igen"):
        xa = {r["edit"]: bool(r["exact"]) for r in rows_a if r.get("kind") == "edit" and r.get("metric") == m}
        xb = {r["edit"]: bool(r["exact"]) for r in rows_b if r.get("kind") == "edit" and r.get("metric") == m}
        e2 = sorted(set(xa) & set(xb))
        if e2:
            out[m] = dict(stats.bootstrap_diff([float(xa[e]) for e in e2], [float(xb[e]) for e in e2], e2, scale=100.0),
                          mcnemar=stats.mcnemar_exact([xa[e] for e in e2], [xb[e] for e in e2]))
    return out
