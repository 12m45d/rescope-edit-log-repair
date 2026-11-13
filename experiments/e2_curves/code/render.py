"""E2 renderer: every reported number is recomputed here from the cell files (e2c) and written to out/results.json (fx.io schema),
out/linear_rows.json, out/rescope_w_rows.json, out/h1.json, out/paper_numbers.json (+ _doc), out/table.txt.  CPU only.

    python render.py            (safe to run at any time: parts that are not finished are listed as missing)
Statistics: fx.stats (percentile bootstrap, 2000 resamples; probe unit = query cluster, edit unit = edit); summaries over seeds /
calibration draws: crossed bootstrap (draws x units), the number of draws is stored next to every interval (n_draws).
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402

M, ST = fx.metrics, fx.stats
EXPN = "E2"
PARTS = ("a", "base", "b", "bl", "c", "cw", "d")
_boot = {}
SMOKE_OK = bool(os.environ.get("E2_RENDER_SMOKE"))       # mechanics test of the renderer on the smoke cells (private output directory)


def boot(flags, units):
    """Percentile bootstrap of a rate (percent), cached by content."""
    k = (tuple(bool(x) for x in flags), tuple(units))
    if k not in _boot:
        _boot[k] = ST.bootstrap([float(bool(x)) for x in flags], list(units), scale=100.0) if flags else None
    return _boot[k]


def split(rows):
    tf = [r for r in rows if "line" not in r and not r.get("d2_row")]
    return ([r for r in tf if r["kind"] == "edit"], [r for r in tf if r["kind"] == "probe" and r.get("disjoint")],
            [r for r in tf if r["kind"] == "hn"], [r for r in rows if r.get("d2_row")])


def cell_stats(rows):
    """{(pool, metric): number | stat dict} of one cell, from its per-row flags."""
    e, p, hn, d2 = split(rows)
    out = {}
    for pool, sub in (("heldout_disjoint", p), ("heldout_strict", [r for r in p if r.get("strict")])):
        if not sub:
            continue
        u = [r["query_id"] for r in sub]
        gate = sub[0].get("fired") is not None
        out[(pool, "probe.n")] = len(sub)
        out[(pool, "probe.D1")] = boot([r["D1"] for r in sub], u)
        out[(pool, "probe.n_D1")] = sum(bool(r["D1"]) for r in sub)
        if gate:
            out[(pool, "probe.fired")] = boot([r["fired"] for r in sub], u)
            out[(pool, "probe.n_fired")] = sum(bool(r["fired"]) for r in sub)
            fr = [r for r in sub if r["fired"]]
            out[(pool, "probe.D1_given_fired")] = M.pct(r["D1"] for r in fr) if fr else None
        if all("D1_dep" in r for r in sub):
            out[(pool, "probe.n_complaints_deployed")] = sum(bool(r["D1_dep"]) for r in sub)
            out[(pool, "probe.complaints_fixed")] = sum(bool(r["D1_dep"]) and not r["D1"] for r in sub)
            out[(pool, "probe.noncomplaints_lost")] = sum((not r["D1_dep"]) and bool(r["D1"]) for r in sub)
        if pool == "heldout_disjoint":
            for nm, ss in (("near", [r for r in sub if r.get("near_edit") is not None]), ("far", [r for r in sub if r.get("near_edit") is None])):
                out[(pool, f"probe.{nm}.n")] = len(ss)
                out[(pool, f"probe.{nm}.D1")] = M.pct(r["D1"] for r in ss) if ss else None
                if gate and ss:
                    out[(pool, f"probe.{nm}.fired")] = M.pct(r["fired"] for r in ss)
            if "tok_loc" in sub[0]:
                tx, im = [r for r in sub if r["ptype"] == "text"], [r for r in sub if r["ptype"] == "image"]
                out[(pool, "probe.T-Loc*")] = M.mean100([r["tok_loc"] for r in tx])
                out[(pool, "probe.I-Loc*")] = M.mean100([r["tok_loc"] for r in im])
                out[(pool, "probe.tok_loc")] = M.mean100([r["tok_loc"] for r in sub])
                out[(pool, "probe.n_nonfinite")] = sum(not r["finite"] for r in sub)
            if "y0_deg" in sub[0]:
                deg = [r for r in sub if r["y0_deg"]]
                out[(pool, "probe.n_y0_degenerate")] = len(deg)
                out[(pool, "probe.D1_y0_not_degenerate")] = M.pct(r["D1"] for r in sub if not r["y0_deg"])
    if e:
        for m in ("rel", "tgen", "igen"):
            sub = [r for r in e if r["metric"] == m]
            if not sub:
                continue
            u = [r["edit"] for r in sub]
            out[("edits", f"edit.{m}.n")] = len(sub)
            out[("edits", f"edit.{m}.exact")] = boot([r["exact"] for r in sub], u)
            out[("edits", f"edit.{m}.token")] = M.mean100([r["token_acc"] for r in sub])
            if sub[0].get("own_fired_last") is not None:
                out[("edits", f"edit.{m}.own_cand")] = M.pct(r["own_fired_last"] for r in sub)
                out[("edits", f"edit.{m}.n_own_out_of_scope")] = sum(not r["own_fired_last"] for r in sub)
                out[("edits", f"edit.{m}.fired")] = M.pct(r["fired"] for r in sub)
            if all("exact_dep" in r for r in sub):                       # R7: edits kept / gained against the deployed log (percent of all edits)
                n = len(sub)
                out[("edits", f"kg.{m}.kept")] = 100.0 * sum(r["exact"] and r["exact_dep"] for r in sub) / n
                out[("edits", f"kg.{m}.gained")] = 100.0 * sum(r["exact"] and not r["exact_dep"] for r in sub) / n
                out[("edits", f"kg.{m}.lost")] = 100.0 * sum(r["exact_dep"] and not r["exact"] for r in sub) / n
        pe = M.rephrase_exact(e)
        if pe:
            out[("edits", "edit.rephrase.exact")] = ST.bootstrap([pe[k] for k in sorted(pe)], sorted(pe), scale=100.0)
        if all("exact_dep" in r for r in e):
            out[("edits", "edit.checks_broken")] = sum(r["exact_dep"] and not r["exact"] for r in e)
            out[("edits", "edit.checks_fixed")] = sum(r["exact"] and not r["exact_dep"] for r in e)
            out[("edits", "edit.n_checks")] = len(e)
    for t in M.HN_TYPES + ("all",):
        sub = [r for r in hn if t == "all" or r["hn_type"] == t]
        if not sub:
            continue
        u = [r["edit"] for r in sub]
        out[("hard_negatives", f"hn.{t}.n")] = len(sub)
        out[("hard_negatives", f"hn.{t}.D1")] = boot([r["D1"] for r in sub], u)
        if sub[0].get("fired") is not None:
            out[("hard_negatives", f"hn.{t}.fired")] = M.pct(r["fired"] for r in sub)
        if "y0_deg" in sub[0]:
            out[("hard_negatives", f"hn.{t}.n_y0_degenerate")] = sum(bool(r["y0_deg"]) for r in sub)
            out[("hard_negatives", f"hn.{t}.D1_y0_not_degenerate")] = M.pct(r["D1"] for r in sub if not r["y0_deg"])
    if d2:
        s = M.d2_stats(d2, ci=True)
        out[("heldout_disjoint", "d2.D2")] = s["D2"]
        out[("heldout_disjoint", "d2.n_pre_correct")] = s["n_pre_correct"]
        out[("heldout_disjoint", "d2.n_D2")] = s["n_D2"]
    return out


CROSSED = {("heldout_disjoint", "probe.D1"): ("D1", "query_id", lambda r: r["kind"] == "probe" and r.get("disjoint") and "line" not in r),
           ("heldout_strict", "probe.D1"): ("D1", "query_id", lambda r: r["kind"] == "probe" and r.get("strict") and "line" not in r),
           ("heldout_disjoint", "probe.fired"): ("fired", "query_id", lambda r: r["kind"] == "probe" and r.get("disjoint") and "line" not in r),
           ("edits", "edit.rel.exact"): ("exact", "edit", lambda r: r["kind"] == "edit" and r["metric"] == "rel" and "line" not in r),
           ("edits", "edit.tgen.exact"): ("exact", "edit", lambda r: r["kind"] == "edit" and r["metric"] == "tgen" and "line" not in r),
           ("edits", "edit.igen.exact"): ("exact", "edit", lambda r: r["kind"] == "edit" and r["metric"] == "igen" and "line" not in r),
           ("heldout_disjoint", "d2.D2"): ("D2", "query_id", lambda r: r.get("d2_row") and r.get("pre_correct"))}


class Store:
    def __init__(self):
        self.R = fx.io.Results(EXPN, e2c.OUT)
        self.cells = {}                      # (part, log, tag) -> (meta, stats)
        self.rows = {}

    def load(self, logs, caps):
        for part in PARTS:
            for name in e2c.LOGS:
                for c in e2c.read_cells(part, name):
                    if c.get("smoke") and not SMOKE_OK:
                        continue
                    rows = e2c.read_rows(part, name, c["tag"])
                    st = cell_stats(rows)
                    st.update(extra_stats(c, logs, caps))
                    self.cells[(part, name, c["tag"])] = (c, st)
                    self.rows[(part, name, c["tag"])] = rows

    def add(self, c, pool, metric, v, draw="cell", **extra):
        d = (c.get("draw") if c.get("draw") is not None else c.get("seed")) if draw == "cell" else draw
        kw = dict(variant=c.get("variant", ""), alpha=c.get("alpha"), n=c.get("n"), draw=d, pool=pool, part=c["part"], n_label=str(c.get("n_label", "")), **extra)
        if isinstance(v, dict):
            return self.R.add_stat(c["log"], c["method"], metric, v, **kw)
        return self.R.add(c["log"], c["method"], metric, None if v is None else float(v), **kw)


def extra_stats(c, logs, caps):
    """Cell-level numbers that are not row statistics: counts of the repair, the order check, the hook check, the comparison with
    the P12 record, near-scope firing at gate level from the captured vectors (ReScope-W cells)."""
    out = {}
    for k in ("n_refused", "n_reported_still_firing", "n_thresholds_raised_total", "n_finite_thresholds", "n_reported_resolved", "rollback_k", "n_edits_capped",
              "n_listed_queries"):
        if c.get(k) is not None:
            out[("repair", "meta." + k)] = c[k]
    if c.get("order_check"):
        out[("repair", "order.n_orders")] = c["order_check"]["n_orders"]
        out[("repair", "order.n_equal_thresholds")] = c["order_check"]["n_equal_thresholds"]
        out[("repair", "order.vector_rule_equals_driver")] = float(c["order_check"]["vector_rule_equals_driver_thresholds"])
    if c.get("hook_vs_vectors"):
        out[("all", "check.hook_vs_vectors.n_rows")] = c["hook_vs_vectors"]["n_rows"]
        out[("all", "check.hook_vs_vectors.n_differ")] = c["hook_vs_vectors"]["n_fired_differs"] + c["hook_vs_vectors"]["n_own_last_differs"]
    if c.get("flags_vs_record"):
        f = c["flags_vs_record"]
        out[("heldout", "check.flags_vs_record.n_heldout_differ")] = f["n_heldout_differ"]
        out[("heldout_disjoint", "check.flags_vs_record.n_heldout_differ")] = f["n_heldout_disjoint_differ"]
        out[("edits", "check.flags_vs_record.n_edit_differ")] = f["n_edit_differ"]
    if c.get("theta") is not None and c["log"] in caps and c["part"] in ("b", "cw", "base"):
        g = gate_from_vectors(logs[c["log"]], caps[c["log"]], c["theta"])
        for t in M.HN_TYPES:
            out[("hard_negatives", f"gate.hn.{t}.fired")] = g[t]
        out[("hard_negatives", "gate.hn.mean_fired")] = float(np.mean([g[t] for t in M.HN_TYPES]))
        out[("heldout_disjoint", "gate.fired_from_vectors")] = g["heldout_disjoint"]
    return out


def group_key(c):
    return (c["part"], c["log"], c["method"], c.get("variant", ""), str(c.get("n_label")))


def gate_from_vectors(L, cap, theta):
    """Gate-level firing of a ReScope-W threshold vector on the hard negatives / held-out rows, from the captured vectors."""
    import torch
    th = torch.tensor([float(x) for x in theta], dtype=torch.float64)
    ix = {j: i for i, j in enumerate(cap["rows"])}
    fired = (cap["Rmax"] >= th[None, :]).any(1).numpy()
    out = {}
    for t in M.HN_TYPES:
        js = [ix[j] for j in L.rows(("hn_i", "hn_q", "hn_x"), valid=True) if j in ix and L.index[j]["hn_type"] == t]
        out[t] = 100.0 * float(fired[js].mean()) if js else None
    hd = [ix[j] for j in L.pool("heldout_disjoint") if j in ix]
    out["heldout_disjoint"] = 100.0 * float(fired[hd].mean())
    return out


def main():
    t0 = time.time()
    S = Store()
    R = S.R
    logs = {n: fx.logs.open(n) for n in e2c.LOGS}
    caps = {}
    for n in e2c.LOGS:
        p = os.path.join(e2c.OUT, "capture", f"{n}.pt")
        if not os.path.exists(p):                                       # ARTIFACT: stop instead of dropping the gate-level records of the captured vectors
            raise SystemExit(f"e2 render: input missing: {p}\n  The captured vectors are in results_rows.tar.gz of the release (unpack it into the clone: "
                             "tar -xzf results_rows.tar.gz -C <clone>); they are written by `job.py <log> --blocks capture` (GPU).  Nothing was written.")
        if os.path.exists(p):
            import torch
            caps[n] = torch.load(p)
    S.load(logs, caps)
    missing = []
    # ---- per-cell records
    groups = {}
    for key, (c, st) in sorted(S.cells.items()):
        for (pool, metric), v in st.items():
            S.add(c, pool, metric, v)
        groups.setdefault(group_key(c), []).append(key)
    # ---- summaries over seeds / draws (crossed bootstrap; n_draws stored)
    for gk, keys in sorted(groups.items()):
        c0 = S.cells[keys[0]][0]
        draws = [(S.cells[k][0].get("draw") if S.cells[k][0].get("draw") is not None else S.cells[k][0].get("seed")) for k in keys]
        if len(keys) < 2 and c0["part"] not in ("a", "b", "bl", "c", "cw"):
            continue
        cs = dict(c0, draw=None, seed=None)
        allm = sorted({pm for k in keys for pm in S.cells[k][1]})
        for pm in allm:
            vals, kk, dd = [], [], []
            for k, d in zip(keys, draws):                                # a metric measured on a subset of the seeds / draws (D2, hard negatives:
                v = S.cells[k][1].get(pm)                               # seed / draw 0) is summarised over the cells that hold it; n_draws says how many
                v = v["value"] if isinstance(v, dict) else v
                if v is not None:
                    vals.append(v)
                    kk.append(k)
                    dd.append(d)
            if not vals:
                continue
            extra = dict(n_draws=len(kk), draws=dd, per_draw=[float(v) for v in vals])
            if pm in CROSSED:
                fld, unit, where = CROSSED[pm]
                try:
                    g = M.grid({i: S.rows[k] for i, k in enumerate(kk)}, lambda r, f=fld: float(bool(r[f])), unit=unit, where=where)
                    s = ST.crossed_bootstrap(g["num"], g["den"], scale=100.0)
                    S.add(cs, pm[0], pm[1] + "@mean", s, draw=None, interval="crossed bootstrap (draws x units)", **extra)
                    continue
                except (ValueError, KeyError):
                    pass
            S.add(cs, pm[0], pm[1] + "@mean", float(np.mean(vals)), draw=None, **extra)
    # ---- round-off floor: flags of ReScope-W with all thresholds at -inf against the deployed weight
    floor = {}
    for n in e2c.LOGS:
        k = ("base", n, "where_floor")
        if k not in S.cells:
            missing.append(f"{n}: base/where_floor")
            continue
        e, p, hn, _ = split(S.rows[k])
        floor[n] = {"edits": sum(r["exact"] != r["exact_dep"] for r in e), "n_edits": len(e),
                    "heldout_disjoint": sum(r["D1"] != r["D1_dep"] for r in p), "n_heldout_disjoint": len(p),
                    "hard_negatives": sum(r["D1"] != r["D1_dep"] for r in hn), "n_hard_negatives": len(hn)}
        c = S.cells[k][0]
        for pool, key in (("edits", "edits"), ("heldout_disjoint", "heldout_disjoint"), ("hard_negatives", "hard_negatives")):
            S.add(c, pool, "floor.n_flags_differ_from_deployed", floor[n][key])
    # ---- gate level: part c (c_gate.py) and part c' (job block cw)
    cgate = {}
    for n in e2c.LOGS:
        p = os.path.join(e2c.OUT, "parts", f"c_gate_{n}.json")
        if not os.path.exists(p):
            missing.append(f"{n}: part c gate level")
            continue
        blob = fx.io.jload(p)
        by = {}
        for c in blob["cells"]:
            by.setdefault((c["method"], c["n_label"]), []).append(c)
        cgate[n] = by
        for (meth, nl), cc in by.items():
            var = {"log-only": "logonly", "log-only + least repair": "logonly+least", "two-block": "", "shared scale": ""}[cc[0]["rule"]]
            base = dict(variant=var, alpha=cc[0].get("alpha"), n=cc[0]["n"], part="c", n_label=nl)
            series = {"gate.fired": ("heldout_disjoint", lambda s: s["probe"]["heldout_disjoint"]["fired"]),
                      "gate.fired_strict": ("heldout_strict", lambda s: s["probe"]["heldout_strict"]["fired"]),
                      "gate.fired_near": ("heldout_disjoint", lambda s: s["probe"]["heldout_disjoint_near"]["fired"]),
                      "gate.fired_far": ("heldout_disjoint", lambda s: s["probe"]["heldout_disjoint_far"]["fired"]),
                      "gate.fired_literal": ("heldout_disjoint", lambda s: s.get("literal", {}).get("heldout_disjoint_fired")),
                      "gate.n_edits_capped": ("edits", lambda s: s["n_edits_capped"]),
                      "gate.rel.own_top": ("edits", lambda s: s["edit"]["rel"]["own_top"]),
                      "gate.tgen.own_top": ("edits", lambda s: s["edit"]["tgen"]["own_top"]),
                      "gate.igen.own_top": ("edits", lambda s: s["edit"]["igen"]["own_top"]),
                      "gate.hn.HN-I.fired": ("hard_negatives", lambda s: s["hn"]["HN-I"]["fired"]),
                      "gate.hn.HN-Q.fired": ("hard_negatives", lambda s: s["hn"]["HN-Q"]["fired"]),
                      "gate.hn.HN-X.fired": ("hard_negatives", lambda s: s["hn"]["HN-X"]["fired"]),
                      "gate.hn.mean_fired": ("hard_negatives", lambda s: s["hn"]["mean_fired"]),
                      "gate.BEN-T.fired": ("BEN-T", lambda s: s["BEN-T"].get("fired")),
                      "gate.MME-T.fired": ("MME-T", lambda s: s["MME-T"].get("fired"))}
            for metric, (pool, fn) in series.items():
                vals = {c["draw"]: fn(c["st"]) for c in cc}
                if any(v is None for v in vals.values()):
                    continue
                R.add_draws(n, meth, metric, {d: float(v) for d, v in vals.items()}, pool=pool, **base)
    wgate = {}
    for n in e2c.LOGS:
        p = os.path.join(e2c.OUT, "parts", f"cw_gate_{n}.json")
        if not os.path.exists(p):
            missing.append(f"{n}: part c' gate level")
            continue
        blob = fx.io.jload(p)
        if blob.get("smoke") and not SMOKE_OK:
            continue
        wgate[n] = blob["draws"]
        base = dict(variant="two_block", alpha=0.05, n=200, part="cw", n_label="200")
        series = {"gate.fired": ("heldout_disjoint", lambda d: d["gate"]["probe"]["heldout_disjoint"]["fired"]),
                  "gate.fired_strict": ("heldout_strict", lambda d: d["gate"]["probe"]["heldout_strict"]["fired"]),
                  "gate.fired_near": ("heldout_disjoint", lambda d: d["gate"]["probe"]["heldout_disjoint_near"]["fired"]),
                  "gate.fired_far": ("heldout_disjoint", lambda d: d["gate"]["probe"]["heldout_disjoint_far"]["fired"]),
                  "gate.fired_literal": ("heldout_disjoint", lambda d: d["gate_literal"]["probe"]["heldout_disjoint"]["fired"]),
                  "gate.n_edits_capped": ("edits", lambda d: d["n_edits_capped"]),
                  "gate.rel.own_last": ("edits", lambda d: d["gate"]["edit"]["rel"]["own_last"]),
                  "gate.tgen.own_last": ("edits", lambda d: d["gate"]["edit"]["tgen"]["own_last"]),
                  "gate.igen.own_last": ("edits", lambda d: d["gate"]["edit"]["igen"]["own_last"]),
                  "gate.rel.own_any": ("edits", lambda d: d["gate"]["edit"]["rel"]["own_any"]),
                  "gate.tgen.own_any": ("edits", lambda d: d["gate"]["edit"]["tgen"]["own_any"]),
                  "gate.igen.own_any": ("edits", lambda d: d["gate"]["edit"]["igen"]["own_any"]),
                  "gate.rel.own_last_literal": ("edits", lambda d: d["gate_literal"]["edit"]["rel"]["own_last"]),
                  "gate.hn.HN-I.fired": ("hard_negatives", lambda d: d["gate"]["hn"]["HN-I"]["fired"]),
                  "gate.hn.HN-Q.fired": ("hard_negatives", lambda d: d["gate"]["hn"]["HN-Q"]["fired"]),
                  "gate.hn.HN-X.fired": ("hard_negatives", lambda d: d["gate"]["hn"]["HN-X"]["fired"])}
        for metric, (pool, fn) in series.items():
            R.add_draws(n, "where_cal", metric, {d["draw"]: float(fn(d)) for d in blob["draws"]}, pool=pool, **base)
        R.add_draws(n, "where_cal", "gate.hn.mean_fired", {d["draw"]: float(np.mean([d["gate"]["hn"][t]["fired"] for t in M.HN_TYPES])) for d in blob["draws"]},
                    pool="hard_negatives", **base)
    # ---- free generation of the edit prompts (block gen)
    for name in e2c.LOGS:
        for c in e2c.read_cells("gen", name):
            if c.get("smoke") and not SMOKE_OK:
                continue
            gs = M.gen_stats(e2c.read_rows("gen", name, c["tag"]))
            src = S.cells.get((c["of_part"], name, c["of_tag"]))
            if src is None:
                continue
            for m, d in gs.items():
                for k in ("eq", "contains", "starts", "n_degenerate"):
                    S.add(src[0], "edits", f"gen.{m}.{k}", d[k])
    R.save(missing=missing, rendered=time.strftime("%Y-%m-%d %H:%M:%S"))
    recs = R.records
    # ---- files for E1: linear rows and the ReScope-W rows of the main table
    fx.io.jdump([r for r in recs if r.get("part") == "d"], os.path.join(e2c.OUT, "linear_rows.json"))
    fx.io.jdump([r for r in recs if r.get("part") == "cw" or (r.get("part") == "base")], os.path.join(e2c.OUT, "rescope_w_rows.json"))
    fx.io.jdump({"linear_rows.json": {"what": "part d: linear repairs for the main table, evaluated at B = 1 through the driver (edits, heldout_disjoint, heldout_strict, hard "
                                              "negatives, D2 of the pre-correct heldout_disjoint rows, free generation where finished); method / variant: COMP_B rebuilt (ALL "
                                              "reported complaints, seed 0), REEDIT_B rebuilt (n = 10) and p12_saved_weight (ALL), RB_UNION / RB_TOT rebuilt; n_label = '10' | 'ALL'",
                                      "cells": sorted(k[1] + "/" + k[2] for k in S.cells if k[0] == "d"),
                                      "rows": "out/rows/d__<Q|L>__<tag>.jsonl.gz"},
                 "rescope_w_rows.json": {"what": "part c': ReScope-W with two-block thresholds (method where_cal; behaviour draws 0 - 2, D2 / hard negatives at draw 0; gate "
                                                 "level 20 draws: metrics gate.*), next to the base rows (deployed, preedit, where_floor)",
                                         "cells": sorted(k[1] + "/" + k[2] for k in S.cells if k[0] == "cw"), "rows": "out/rows/cw__<Q|L>__<tag>.jsonl.gz"},
                 "missing": missing, "rendered": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(e2c.OUT, "rows_for_e1_meta.json"), indent=1)
    import tables
    tables.write(S, recs, floor, cgate, wgate, missing, logs)
    print(f"rendered {len(recs)} records in {time.time() - t0:.0f}s; missing: {missing}")


if __name__ == "__main__":
    main()
