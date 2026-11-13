"""E3 (Qwen): tables.  Renders out/table.txt from out/results.json, out/endpoints.json, out/retention_series.json, out/missing.json and
out/sanity_compare.json (all written by e3_analyze.py).  One block of tables per log family; rows = T and method.

    python e3_render.py
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os
import time

from e3c import *  # noqa: F401,F403

TT = fx.io.text_table


def load():
    recs = jload(os.path.join(OUT, "results.json"))
    ix = {}
    for r in recs:
        ix[(r["log"], r["method"], r["metric"], r.get("variant") or "")] = r
    return recs, ix


def f1(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


class V:
    def __init__(self, ix):
        self.ix = ix

    def r(self, log, m, metric, var=""):
        return self.ix.get((log, m, metric, var))

    def v(self, log, m, metric, var="", nd=1):
        r = self.r(log, m, metric, var)
        return "-" if r is None else f1(r["value"], nd)

    def ci(self, log, m, metric, var="", nd=1):
        r = self.r(log, m, metric, var)
        if r is None or r["value"] is None:
            return "-"
        if r.get("ci_lo") is None:
            return f1(r["value"], nd)
        return f"{f1(r['value'], nd)} [{f1(r['ci_lo'], nd)},{f1(r['ci_hi'], nd)}]"

    def bd(self, log, m, metric, var="", nd=1):
        r = self.r(log, m, metric, var)
        if r is None or not r.get("by_draw"):
            return "-"
        return "/".join(f1(x, nd) for x in r["by_draw"])

    def rng(self, log, m, metric, var="", nd=0):
        r = self.r(log, m, metric, var)
        if r is None or r["value"] is None:
            return "-"
        if r.get("vmin") is None:
            return f1(r["value"], nd)
        return f"{f1(r['value'], 1)} ({f1(r['vmin'], nd)}-{f1(r['vmax'], nd)})"

    def tri(self, log, m, fmt, keys, var="", nd=1, ci=False):
        return " / ".join((self.ci if ci else self.v)(log, m, fmt.format(k), var, nd) for k in keys)


def methods_of(recs, log):
    return order({r["method"] for r in recs if r["log"] == log and r["method"] != "log"})


def ndraws(V_, log, m):
    r = V_.r(log, m, "edit.all.rel.exact")
    if r is None:
        return "-"
    cal = V_.r(log, m, "n_capped.behaviour_draws")
    return str(r.get("n_draws")) if (cal is not None and cal.get("calibrated")) else "none"


def family_tables(fam, logs, recs, ix, ep, lines):
    v = V(ix)
    logs = [g for g in logs if any(r["log"] == g for r in recs)]
    if not logs:
        return
    lines.append("=" * 200)
    lines.append(f"FAMILY {fam}: {FAMILY_TITLE[fam]}")
    lines.append("=" * 200)
    R3 = ("rel", "tgen", "igen")

    def beh_rows(fn, need="edit.all.rel.exact"):
        out = []
        for g in logs:
            for m in methods_of(recs, g):
                if v.r(g, m, need) is None:
                    sk = v.r(g, m, "skipped_for_cost")
                    if sk is not None and need == "edit.all.rel.exact":
                        out.append([str(T_of(g)), m, "-", f"SKIPPED FOR COST: build + one pass projected at {sk['value'] / 60:.0f} min > 40 min"])
                    continue
                out.append([str(T_of(g)), m, ndraws(v, g, m)] + fn(g, m))
        return out

    # ---- A. edits (behaviour)
    lines.append("")
    lines.append(f"[{fam}] A. EDITS, behaviour (B = 1 teacher forcing; percent).  draws = calibration draws behind the row (none: no calibration).  x [lo,hi] = crossed bootstrap over draws x edits (R10).")
    lines.append("    all = all T edits; first100 = the first 100 edits of the stream; later = stream position >= 500 (edits no development step has seen).  kept / gained: against the deployed log (R7), all T edits.")
    hdr = ["T", "method", "draws", "Rel exact all", "T-Gen exact all", "I-Gen exact all", "token Rel/TG/IG all", "first100 exact Rel/TG/IG", "later Rel exact", "later T-Gen exact", "later I-Gen exact",
           "later token Rel/TG/IG", "kept Rel/TG/IG", "gained Rel/TG/IG", "lost Rel/TG/IG"]
    lines.append(TT(hdr, beh_rows(lambda g, m: [v.ci(g, m, "edit.all.rel.exact"), v.ci(g, m, "edit.all.tgen.exact"), v.ci(g, m, "edit.all.igen.exact"), v.tri(g, m, "edit.all.{}.token", R3),
                                                 v.tri(g, m, "edit.first100.{}.exact", R3), v.ci(g, m, "edit.later.rel.exact"), v.ci(g, m, "edit.later.tgen.exact"), v.ci(g, m, "edit.later.igen.exact"),
                                                 v.tri(g, m, "edit.later.{}.token", R3), v.tri(g, m, "kg.all.{}.kept", R3), v.tri(g, m, "kg.all.{}.gained", R3), v.tri(g, m, "kg.all.{}.lost", R3)])))
    # ---- B. locality and near scope (behaviour)
    lines.append("")
    lines.append(f"[{fam}] B. LOCALITY AND NEAR SCOPE, behaviour (percent).  hd = heldout_disjoint (valid rows); near probes = held-out probes whose question string is the question of an edit of the log;")
    lines.append("    HN = near-scope negatives of the log's edits (first 100 edits + sampled later edits): fired / own edit released / D1 per kind.  D2 = rate of the pre-correct hd rows (draw 0; 48-token decode).")
    hdr = ["T", "method", "draws", "capped (per draw)", "hd fired", "hd D1", "hd tok-loc", "n near", "NEAR probes fired", "NEAR probes D1", "far probes fired", "far probes D1",
           "HN-I fired/own/D1", "HN-Q fired/own/D1", "HN-X fired/own/D1", "HN all D1", "D2 rate (n D2 / n pre-correct)"]

    def b_row(g, m):
        d2 = v.r(g, m, "d2.rate")
        return [v.bd(g, m, "n_capped.behaviour_draws", nd=0), v.ci(g, m, "probe.fired"), v.ci(g, m, "probe.D1"), v.v(g, m, "probe.tok_loc", nd=2), v.v(g, m, "probe.near.n", nd=0),
                v.ci(g, m, "probe.near.fired"), v.ci(g, m, "probe.near.D1"), v.v(g, m, "probe.far.fired"), v.v(g, m, "probe.far.D1")] + [
                v.tri(g, m, "hn." + k + ".{}", ("fired", "own", "D1")) for k in HN] + [v.ci(g, m, "hn.all.D1"),
                "-" if d2 is None else f"{v.ci(g, m, 'd2.rate')} ({d2['n_D2']}/{d2['n_pre_correct']})"]
    lines.append(TT(hdr, beh_rows(b_row)))
    # ---- C. gate level, 20 draws
    lines.append("")
    lines.append(f"[{fam}] C. GATE LEVEL, 20 calibration draws (key caches; percent; mean [5th,95th percentile over draws]; rows without calibration: one gate).")
    lines.append("    literal = the conformal rule without the guard cap (R25).  BEN-T far: rows of BEN-T that the pool's flags do not relate to the family (related rows excluded, counted), PROMPT-ONLY keys,")
    lines.append("    gate calibrated at teacher-forced REPAIR keys (fx.pools.firing).  MME-T: out-of-benchmark workload, calibrated on REPAIR (fx.pools.firing).  capped = edits at their guard cap, mean (min-max):")
    lines.append("    the bound is void for capped edits.  own prompts: silenced by the literal rule (before the cap) / not a candidate after the cap / released by ANOTHER edit (top-1 confusion).")
    hdr = ["T", "method", "eta", "bound", "hd fired", "hd fired literal", "NEAR probes fired", "BEN-T far fired (n far; n related excl.)", "MME-T fired", "cov T-Gen", "cov I-Gen",
           "cov later TG/IG", "HN-I/Q/X fired", "capped", "own: sil.before / not cand. / by other", "lineage hd: mean width / % >1 / max", "thr median"]
    rows = []
    for g in logs:
        for m in methods_of(recs, g):
            if v.r(g, m, "gate.fire.hd") is None:
                nd_ = v.r(g, m, "gate.not_defined")
                if nd_ is not None:
                    rows.append([str(T_of(g)), m, "not defined: " + str(nd_.get("note"))[:120]])
                continue
            bt = v.r(g, m, "gate.BEN-T.far.fired")
            ben = "-" if bt is None else f"{v.ci(g, m, 'gate.BEN-T.far.fired')} ({v.v(g, m, 'gate.BEN-T.far.n', nd=0)}; {v.v(g, m, 'gate.BEN-T.related.n', nd=0)})"
            mme = v.ci(g, m, "gate.MME-T.fired") if v.r(g, m, "gate.MME-T.fired") is not None else "not defined"
            rows.append([str(T_of(g)), m, v.v(g, m, "gate.eta", nd=3), v.v(g, m, "gate.bound", nd=2), v.ci(g, m, "gate.fire.hd"), v.v(g, m, "gate.fire.hd.literal"), v.ci(g, m, "gate.fire.hd.near"), ben, mme,
                         v.ci(g, m, "gate.cov.tgen"), v.ci(g, m, "gate.cov.igen"), v.tri(g, m, "gate.cov.later.{}@x", ("tgen", "igen")), v.tri(g, m, "gate.hn.{}.fired", HN),
                         v.rng(g, m, "gate.n_capped"), " / ".join(v.v(g, m, "gate.own." + k, nd=1) for k in ("silenced_before_cap", "not_candidate_after_cap", "released_by_other")),
                         " / ".join((v.v(g, m, "gate.lineage.hd.width_mean", nd=2), v.v(g, m, "gate.lineage.hd.multi"), v.v(g, m, "gate.lineage.hd.width_max", nd=1))), v.v(g, m, "gate.thr.median", nd=3)])
    lines.append(TT(hdr, rows))
    # ---- D. H3
    lines.append("")
    lines.append(f"[{fam}] D. READ-OUT H3 (near-scope-constrained; R6, R25 (4)): the common scale is raised (guard caps kept) until mean firing on the FIT hard negatives (even edits) is <= 10 %; read on the ODD edits.")
    lines.append("    HN-Q rows belong to the edit whose question they ask.  gate level: 20 draws; behaviour: the draws of column 'draws'.  unreachable draws count coverage 0 (rule of E1).  wrong-target: generated answer")
    lines.append("    equals the target of the releasing edit.  _x methods, second form: test hard negatives whose foreign part occurs in no recombination row (HN-I, HN-Q; HN-X count given, nearly empty).")
    hdr = ["T", "method", "gate: unreach./20", "gate: at cap", "gate: coverage TG/IG/mean", "gate: test fired I/Q/X", "gate: hd fired", "gate: NEAR probes", "draws", "beh: reph. exact", "beh: TG / IG exact",
           "beh: test fired I/Q/X", "beh: test D1 I/Q/X", "beh: wrong-target I/Q/X", "beh: hd fired", "beh: at cap (per draw)", "unseen-foreign: fired I/Q (n I, Q; n HN-X)"]
    rows = []
    for g in logs:
        for m in methods_of(recs, g):
            if v.r(g, m, "gate.h3.cov") is None:
                continue
            nun = v.r(g, m, "gate.h3.reachable")
            unr = "-" if nun is None else f"{sum(1 for x in nun['by_draw'] if not x)}"
            uf = "-"
            if v.r(g, m, "gate.h3u.test.HN-I.fired") is not None:
                uf = (f"gate {v.v(g, m, 'gate.h3u.test.HN-I.fired')} / {v.v(g, m, 'gate.h3u.test.HN-Q.fired')}; beh {v.v(g, m, 'h3.unseen.test.HN-I.fired')} / {v.v(g, m, 'h3.unseen.test.HN-Q.fired')} "
                      f"(n {v.v(g, m, 'gate.h3u.test.HN-I.n', nd=0)}, {v.v(g, m, 'gate.h3u.test.HN-Q.n', nd=0)}; HN-X {v.v(g, m, 'gate.h3u.test.HN-X.n', nd=0)})")
            hb = v.r(g, m, "h3.rephrase.exact")
            rows.append([str(T_of(g)), m, unr, v.rng(g, m, "gate.h3.n_at_cap"), v.tri(g, m, "gate.h3.{}", ("tgen", "igen", "cov")), v.tri(g, m, "gate.h3.test.{}.fired", HN), v.v(g, m, "gate.h3.hd.fired", nd=2),
                         v.v(g, m, "gate.h3.hd.near.fired"), "-" if hb is None else str(hb.get("n_draws")), v.ci(g, m, "h3.rephrase.exact"), v.tri(g, m, "h3.{}.exact", ("tgen", "igen")),
                         v.tri(g, m, "h3.test.{}.fired", HN), v.tri(g, m, "h3.test.{}.D1", HN), v.tri(g, m, "h3.test.{}.wrong_target", HN), v.v(g, m, "h3.hd.fired", nd=2), v.bd(g, m, "h3.n_at_cap", nd=0), uf])
    lines.append(TT(hdr, rows))
    # ---- E. near-scope read-outs of the _x methods, twice
    rows = []
    for g in logs:
        for m in methods_of(recs, g):
            if v.r(g, m, "gate.ns.HN-I.fired") is None:
                continue
            rows.append([str(T_of(g)), m, v.tri(g, m, "gate.ns.{}.fired", HN), v.tri(g, m, "gate.ns.{}.n", HN, nd=0), v.tri(g, m, "gate.nsu.{}.fired", ("HN-I", "HN-Q")),
                         v.tri(g, m, "gate.nsu.{}.n", HN, nd=0), v.rng(g, m, "gate.n_capped")])
    if rows:
        lines.append("")
        lines.append(f"[{fam}] E. NEAR-SCOPE FIRING OF THE _x METHODS, reported twice (gate level, 20 draws, at the calibrated scale): all hard negatives of the log, and the subset whose foreign part occurs in no")
        lines.append("    recombination row of the log (clarification of 09:03; HN-X subset nearly empty: count given, rate not reported).")
        lines.append(TT(["T", "method", "all: fired HN-I/Q/X", "all: n I/Q/X", "unseen foreign: fired HN-I/Q", "unseen foreign: n I/Q/X", "capped"], rows))
    # ---- F. second read-out (R8)
    lines.append("")
    lines.append(f"[{fam}] F. SECOND READ-OUT OF THE SCALE SERIES (R8), next to the registered one.  " + QD_NOTE + ".")
    lines.append("    reg = registered calibration; qd = calibration from the question-disjoint REPAIR rows.  qd rows = heldout_disjoint rows passing the filter.  Rows without calibration: the filter only.")
    hdr = ["T", "method", "n qd rows", "gate hd fired: reg, all rows", "gate: reg, qd rows", "gate: qd cal, qd rows", "gate BEN-T qd: reg / qd cal", "gate cov TG/IG: qd cal", "gate capped: qd cal",
           "gate HN all fired: reg / qd cal", "draws", "beh Rel/TG/IG exact: qd cal", "beh hd fired, qd rows: reg / qd cal", "beh hd D1, qd rows: reg / qd cal", "beh HN all D1: reg / qd cal"]
    rows = []
    for g in logs:
        for m in methods_of(recs, g):
            if v.r(g, m, "gate.fire.hd") is None and v.r(g, m, "probe.qdrows.D1") is None:
                continue
            q = "qdisjoint"
            hasq = v.r(g, m, "gate.fire.hd.qd", q) is not None
            hasb = v.r(g, m, "qd.probe.D1", q) is not None
            rows.append([str(T_of(g)), m, v.v(g, m, "gate.n.hd.qd", nd=0) if v.r(g, m, "gate.n.hd.qd") else v.v(g, m, "probe.qdrows.n", nd=0), v.ci(g, m, "gate.fire.hd"), v.ci(g, m, "gate.fire.hd.qd"),
                         v.ci(g, m, "gate.fire.hd.qd", q) if hasq else ("(no calibration)" if v.r(g, m, "gate.eta") is None else "(not yet run)"), f"{v.v(g, m, 'gate.BEN-T.qd.fired')} / {v.v(g, m, 'gate.BEN-T.qd.fired', q) if hasq else '-'}",
                         v.tri(g, m, "gate.cov.{}", ("tgen", "igen"), q) if hasq else "-", v.rng(g, m, "gate.n_capped", q) if hasq else "-",
                         f"{v.v(g, m, 'gate.hn.all.fired')} / {v.v(g, m, 'gate.hn.all.fired', q) if hasq else '-'}", ndraws(v, g, m),
                         v.tri(g, m, "qd.edit.all.{}.exact", R3, q) if hasb else "-", f"{v.v(g, m, 'probe.qdrows.fired')} / {v.v(g, m, 'qd.probe.fired', q) if hasb else '-'}",
                         f"{v.ci(g, m, 'probe.qdrows.D1')} / {v.ci(g, m, 'qd.probe.D1', q) if hasb else '-'}", f"{v.v(g, m, 'hn.all.D1')} / {v.v(g, m, 'qd.hn.all.D1', q) if hasb else '-'}"])
    lines.append(TT(hdr, rows))
    # ---- G. endpoints
    lines.append("")
    lines.append(f"[{fam}] G. ENDPOINTS PER LOG (descriptive per log; Holm over backbones and baselines is the coordinator's).  H2: (T-Gen + I-Gen) / 2 exact of rescope minus the baseline, paired crossed bootstrap")
    lines.append("    over draws x edits; on all edits and on LATER EDITS ONLY (the confirmation of H2 on unseen logs is read on later edits); gate form = the same difference in coverage over the 20 draws.")
    rows = []
    for g in logs:
        e = (ep.get(g) or {})
        for role, x in (e.get("H2") or {}).items():
            if x.get("missing"):
                rows.append([str(T_of(g)), "H2", role, x["method"], "missing"])
                continue
            for sub in ("all", "later", "first100"):
                if sub not in x:
                    continue
                s = x[sub]
                gc = s.get("gate_coverage_form_20_draws")
                rows.append([str(T_of(g)), "H2 " + sub, role, x["method"], str(x["n_draws"]), f"{f1(s['mean_rescope'])} vs {f1(s['mean_other'])}", f"{f1(s['diff'])} [{f1(s['ci_lo'])},{f1(s['ci_hi'])}]",
                             f"{s['p']:.4f}", "/".join(f1(b) for b in s["by_draw"]), "-" if gc is None else f"{f1(gc['value'])} [{f1(gc['ci_lo'])},{f1(gc['ci_hi'])}] ({f1(gc['mean_a'])} vs {f1(gc['mean_b'])})", str(s["n_units"])])
        for pair, x in (e.get("H3") or {}).items():
            gc = x.get("gate_coverage_form_20_draws")
            gtxt = "-" if gc is None else f"{f1(gc['value'])} [{f1(gc['ci_lo'])},{f1(gc['ci_hi'])}] ({f1(gc['mean_a'])} vs {f1(gc['mean_b'])})"
            if "diff" in x:
                rows.append([str(T_of(g)), "H3", pair, f"{x['a']} - {x['b']}", str(x["n_draws"]), f"{f1(x['mean_a'])} vs {f1(x['mean_b'])}", f"{f1(x['diff'])} [{f1(x['ci_lo'])},{f1(x['ci_hi'])}]", f"{x['p']:.4f}",
                             "/".join(f1(b) for b in x["by_draw"]), gtxt, str(x["n_units"])])
            else:
                rows.append([str(T_of(g)), "H3", pair, f"{x['a']} - {x['b']}", "-", "-", "behaviour missing", "-", "-", gtxt, "-"])
    lines.append(TT(["T", "endpoint", "role", "baseline / pair", "draws", "exact: a vs b", "difference [CI]", "p", "per draw", "gate coverage form, 20 draws: diff [CI] (a vs b)", "n edits"], rows))
    # ---- H. draws listed separately
    lines.append("")
    lines.append(f"[{fam}] H. CALIBRATION DRAWS LISTED SEPARATELY (behaviour; with three draws the crossed bootstrap is too narrow when draws matter: R25 (3)).  Values per draw, in the order of the draws.")
    rows = []
    for g in logs:
        for m in methods_of(recs, g):
            c = v.r(g, m, "n_capped.behaviour_draws")
            if c is None or not c.get("calibrated"):
                continue
            rows.append([str(T_of(g)), m, "/".join(str(d) for d in c["draws"]), v.bd(g, m, "n_capped.behaviour_draws", nd=0), v.bd(g, m, "edit.all.rel.exact"), v.bd(g, m, "edit.all.tgen.exact"),
                         v.bd(g, m, "edit.all.igen.exact"), v.bd(g, m, "probe.fired", nd=2), v.bd(g, m, "probe.D1", nd=2), v.bd(g, m, "probe.near.fired"), v.bd(g, m, "hn.all.D1")])
    lines.append(TT(["T", "method", "draws", "capped", "Rel exact", "T-Gen exact", "I-Gen exact", "hd fired", "hd D1", "NEAR probes fired", "HN all D1"], rows))
    # ---- I. log level
    lines.append("")
    lines.append(f"[{fam}] I. LOGS: stored metrics of the deployed log (testbed build), factor level (rebase lemma: relative error of the released values against the cached clean-model targets, with and without")
    lines.append("    rebasing; self-separation margin_i = 1 - largest normalised coordinate of edit i at another solver key), pools, free generation of the default method (draw 0).")
    rows = []
    for g in logs:
        rows.append([str(T_of(g)), g, v.tri(g, "deployed", "stored.edit.{}.exact", R3), v.v(g, "deployed", "stored.probe.D1"), v.tri(g, "log", "factor.rebase.err_rebased.{}", ("median", "max"), nd=4),
                     v.tri(g, "log", "factor.rebase.err_logged.{}", ("median", "max"), nd=4), v.tri(g, "log", "factor.rebase.err_rebased_later.{}", ("median", "max"), nd=4),
                     v.tri(g, "log", "factor.rebase.err_logged_later.{}", ("median", "max"), nd=4), f"{v.v(g, 'log', 'factor.margin.n_positive', nd=0)} of {T_of(g)}",
                     v.v(g, "log", "factor.margin.min", nd=4), v.v(g, "log", "describe.n_heldout_disjoint", nd=0), v.v(g, "log", "describe.n_heldout_disjoint_near", nd=0), v.v(g, "log", "describe.n_probe_invalid", nd=0),
                     v.v(g, "deployed", "probe.n_y0_degenerate", nd=0), " / ".join(v.v(g, "log", "describe.hn." + k, nd=0) for k in HN),
                     v.tri(g, "rescope", "gen.first100.{}.eq", R3), v.tri(g, "rescope", "gen.first100.{}.contains", R3), v.ci(g, "rescope", "gen.probe200.eq_pre")])
    lines.append(TT(["T", "log", "stored exact Rel/TG/IG", "stored hd D1", "err rebased med/max", "err logged med/max", "later: rebased med/max", "later: logged med/max", "margin > 0", "margin min", "n hd",
                     "n hd near", "invalid probe rows (excl.)", "degenerate y0 (hd)", "valid HN-I/Q/X", "rescope gen first100: eq Rel/TG/IG", "contains Rel/TG/IG", "gen 200 hd probes = pre-edit"], rows))


def main():
    recs, ix = load()
    ep = jload(os.path.join(OUT, "endpoints.json"))
    ret = jload(os.path.join(OUT, "retention_series.json"))
    miss = jload(os.path.join(OUT, "missing.json"))
    meta = jload(os.path.join(OUT, "results_meta.json")) if os.path.exists(os.path.join(OUT, "results_meta.json")) else {}
    lines = [f"E3 / E4 ON THE Qwen2.5-VL-3B LOGS (e3_qwen)   rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by e3_render.py from out/results.json ({len(recs)} records; code fx {fx.code_fingerprint()})",
             "Frozen row set, alpha = 0.05, n = 200; gate level: 20 calibration draws; behaviour: draws 0-2 on logs with at most 500 edits, draw 0 on logs with 1000 or more (R23).  Nothing is tuned on any log.",
             "Selection (e1_main/out/best_cosine_gate.json through fx.review1.selected): best gate of the cosine family for H3 = white_1_shared (its _x form white_1_x), for H2 = white_0.1_shared; retr = retr_sum.",
             "Not run by decision of the protocol (R19, R24): dual on unseen logs, the two-block thresholds-only repair, behaviour of rescope_nullW / rescope_pac and of BEN-T / MME rows (gate level only).",
             "Not defined (reported, not substituted): the retrieval family and lookup on MME-T (no retrieval attributes for MME rows); they have no _x form.",
             f"Logs present: {', '.join(sorted({r['log'] for r in recs}, key=lambda g: LOGS.index(g)))}.  Missing cells at render time: {miss['n_missing']} (out/missing.json).", ""]
    # compact main table
    v = V(ix)
    lines.append("=" * 200)
    lines.append("MAIN TABLE, COMPACT (one row per log and method; percent).  Behaviour: mean over the behavioural draws (column draws); gate: mean over 20 calibration draws.  Details and intervals: the family tables below.")
    lines.append("    hd = heldout_disjoint; NEAR = held-out probes asking the question of an edit of the log; HN = near-scope negatives of the log's edits (all kinds); later = stream position >= 500; capped = gate level, mean over 20 draws.")
    rows = []
    for g in LOGS:
        for m in methods_of(recs, g):
            if v.r(g, m, "edit.all.rel.exact") is None and v.r(g, m, "gate.fire.hd") is None:
                if v.r(g, m, "skipped_for_cost") is not None:
                    rows.append([g.replace("TB-", ""), m, "-", "skipped for cost"])
                continue
            rows.append([g.replace("TB-", ""), m, ndraws(v, g, m), v.tri(g, m, "edit.all.{}.exact", ("rel", "tgen", "igen")), v.tri(g, m, "edit.later.{}.exact", ("rel", "tgen", "igen")),
                         v.tri(g, m, "edit.first100.{}.exact", ("rel", "tgen", "igen")), v.v(g, m, "probe.fired"), v.v(g, m, "probe.D1"), v.v(g, m, "probe.near.fired"), v.v(g, m, "probe.near.D1"),
                         v.v(g, m, "hn.all.fired"), v.v(g, m, "hn.all.D1"), v.v(g, m, "d2.rate"), v.v(g, m, "gate.fire.hd"), v.v(g, m, "gate.fire.hd.literal"), v.v(g, m, "gate.fire.hd.near"),
                         v.v(g, m, "gate.BEN-T.far.fired"), v.v(g, m, "gate.MME-T.fired") if v.r(g, m, "gate.MME-T.fired") is not None else ("n.d." if v.r(g, m, "gate.fire.hd") is not None else "-"),
                         v.tri(g, m, "gate.cov.{}", ("tgen", "igen")), v.v(g, m, "gate.hn.all.fired"), v.v(g, m, "gate.n_capped")])
    lines.append(TT(["log", "method", "draws", "exact Rel/TG/IG all", "exact Rel/TG/IG later", "exact Rel/TG/IG first100", "hd fired", "hd D1", "NEAR fired", "NEAR D1", "HN fired", "HN D1", "D2 rate",
                     "gate hd fired", "gate literal", "gate NEAR", "gate BEN-T far", "gate MME-T", "gate cov TG/IG", "gate HN fired", "capped"], rows))
    lines.append("")
    for fam, logs in FAMILIES.items():
        family_tables(fam, logs, recs, ix, ep, lines)
        lines.append("")
    # the published cosine gate
    v = V(ix)
    lines.append("=" * 200)
    lines.append("PUBLISHED COSINE GATE (0.65, all positions) against its last-position form (cosine_default): teacher-forced rows whose exact / token / D1 / firing flags differ, per log: "
                 + "; ".join(f"{g}: {v.v(g, 'cosine_published', 'n_rows_differ_from_cosine_default', nd=0)}" for g in LOGS if v.r(g, "cosine_published", "n_rows_differ_from_cosine_default") is not None))
    lines.append("Rescope-WHERE (log-only thresholds), cost of one pass (seconds; build + all sequences): "
                 + "; ".join(f"{g}: {f1(v.r(g, 'where_logonly', 'timing.pass_s')['value'], 0)}" for g in LOGS if v.r(g, "where_logonly", "timing.pass_s") is not None)
                 + "; skipped for cost: " + (", ".join(g for g in LOGS if v.r(g, "where_logonly", "skipped_for_cost") is not None) or "none"))
    # retention
    lines.append("=" * 200)
    lines.append("RETENTION (E9 c): Rel / T-Gen / I-Gen exact of the FIRST 100 EDITS of the stream as T grows (percent; [crossed-bootstrap interval]); from out/retention_series.json")
    rows = []
    for fam, f in ret["families"].items():
        for m, e in f["methods"].items():
            for i, g in enumerate(f["logs"]):
                def cell(k):
                    x, c = e[k + "_exact"][i], e[k + "_exact_ci"][i]
                    if x is None:
                        return "skipped for cost" if e["skipped_for_cost"][i] else "-"
                    return f"{f1(x)}" + (f" [{f1(c[0])},{f1(c[1])}]" if c and c[0] is not None else "")
                rows.append([fam, m, str(f["T"][i]), str(len(f["behavioural_draws"][i])) if m == "rescope" else "none", cell("rel"), cell("tgen"), cell("igen")])
    lines.append(TT(["family", "method", "T", "draws", "Rel exact, first 100", "T-Gen exact, first 100", "I-Gen exact, first 100"], rows))
    # sanity
    sp = os.path.join(OUT, "sanity_compare.json")
    if os.path.exists(sp):
        s = jload(sp)
        lines.append("")
        lines.append(f"AGREEMENT WITH THE SANITY RUN e0 on {s['log']}: {s['n_cells']} cells compared (gate level 20 draws; behaviour draws 0-2; rescope, rescope_shared, cosine, cosine_shared, cosine_default, exact, deployed), "
                     f"{s['n_cells_differ_above_1e-9']} differ by more than 1e-9, largest absolute difference {s['largest_abs_diff']:.3g}.")
    if miss["n_missing"]:
        lines.append("")
        lines.append("MISSING CELLS (not finished or failed; out/missing.json): " + "; ".join(sorted({f"{x['log']}:{x.get('label', '')}:{x.get('part', '')}" for x in miss["missing"]}))[:6000])
    fx.io.Results("E3", OUT).table("\n".join(lines))
    print("table.txt", len(lines), "lines")


if __name__ == "__main__":
    main()
