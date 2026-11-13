"""E1: renders out/table.txt from out/results.json, gate_draws.json, endpoints.json, decision.json, frontier.json, best_cosine_gate.json
and the cell records.  Every number printed is read from a JSON file written by code."""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import glob
import os
import time

from e1c import *  # noqa: F401,F403
import e1c



def TT(headers, rows, align=None):
    """fx.io.text_table with rows padded / cut to the header length."""
    n = len(headers)
    return fx.io.text_table(headers, [(list(r) + [""] * n)[:n] for r in rows], align)


E2 = FINAL + "/e2_curves/out"


def f(v, nd=1):
    return "-" if v is None or isinstance(v, str) else f"{v:.{nd}f}"


def ci(r, nd=1):
    if r is None or r.get("value") is None:
        return "-"
    if r.get("ci_lo") is None:
        return f(r["value"], nd)
    return f"{r['value']:.{nd}f} [{r['ci_lo']:.{nd}f}, {r['ci_hi']:.{nd}f}]"


class Idx:
    def __init__(self, records):
        self.d = {}
        for r in records:
            self.d.setdefault((r["log"], r["method"], r["metric"], r["pool"], r["draw"]), r)

    def get(self, log, method, metric, pool, draw=None):
        return self.d.get((log, method, metric, pool, draw))

    def beh(self, log, lab, metric, pool, cal):
        """A behavioural cell: calibrated rows -> the crossed-bootstrap record over the draws; other rows -> the single record."""
        return self.get(log, lab, metric + "@x", pool) if cal else self.get(log, lab, metric, pool)

    def by_draw(self, log, lab, metric, pool, nd=1):
        v = [self.get(log, lab, metric, pool, d) for d in (0, 1, 2)]
        return "/".join(f(x["value"], nd) if x else "-" for x in v)


def gd(G, log, lab, metric, alpha="0.05"):
    """Gate-level values over the draws (list) or None."""
    e = G.get(log, {}).get(lab)
    if e is None:
        return None
    a = alpha if alpha in e else ("none" if "none" in e else None)
    if a is None or metric not in e[a]:
        return None
    v = [x for x in e[a][metric] if x is not None]
    return v or None


def gs(G, log, lab, metric, alpha="0.05", nd=1, spread=True):
    v = gd(G, log, lab, metric, alpha)
    if v is None:
        return "-"
    if len(v) == 1:
        return f(v[0], nd)
    m, lo, hi = float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if spread else f"{m:.{nd}f}"


def rows_for(sel):
    best = list(dict.fromkeys([sel["cos_H3"], sel["cos_H2"]]))
    best_x = list(dict.fromkeys([sel["cos_H3_x"], sel["cos_H2_x"]]))
    retr = list(dict.fromkeys([sel["retr_H3"], sel["retr_H2"]]))
    name = {"deployed": "deployed log W_E", "preedit": "pre-edit W_0", "rescope": "ReScope (default)", "rescope_shared": "ReScope shared scale", "rescope_null": "ReScope-N",
            "rescope_x": "rescope_x (log-derived negatives)", "where_logonly": "ReScope-W log-only thresholds", "rescope_lin": "ReScope, linear response (LIN)",
            "cosine": "cosine two-block", "cosine_shared": "cosine shared scale", "lookup": "lookup (requests)", "cosine_default": "cosine 0.65, last position",
            "cosine_published": "cosine 0.65, all positions", "zscore_default": "z-score 2.2 (published)", "exact": "exact match (cos >= 0.999)", "ball": "ball, conformal radius",
            "router": "router, conformal threshold", "dual": "dual (answer match; tf = composition)"}
    for i, b in enumerate(best):
        name[b] = f"best cosine-family gate ({'H3' if b == sel['cos_H3'] else 'H2'}{' = H2' if len(best) == 1 else ''}): {b}"
    for b in best_x:
        name[b] = f"its _x form: {b}" + (" (H3 selection)" if b == sel["cos_H3_x"] and len(best_x) > 1 else (" (H2 selection)" if len(best_x) > 1 else ""))
    for r in retr:
        name[r] = f"retr = {r} (requests + encoders)"
    order = ["deployed", "preedit", "rescope", "rescope_shared", "rescope_null", "rescope_x", "where_logonly", "rescope_lin"] + best + best_x + ["cosine", "cosine_shared"] + retr + \
            ["lookup", "cosine_default", "cosine_published", "zscore_default", "exact", "ball", "router", "dual"]
    cal = set(["rescope", "rescope_shared", "rescope_null", "rescope_x", "rescope_lin", "cosine", "cosine_shared", "ball", "router"] + best + best_x + retr)
    return order, name, cal


def main():
    recs = jload(os.path.join(OUT, "results.json"))
    I = Idx(recs)
    G = jload(os.path.join(OUT, "gate_draws.json"))["draws"]
    EP = jload(os.path.join(OUT, "endpoints.json"))
    DEC = jload(os.path.join(OUT, "decision.json"))
    MISS = jload(os.path.join(OUT, "missing.json"))
    SELF = jload(SEL)
    FR = jload(os.path.join(OUT, "frontier.json")) if os.path.exists(os.path.join(OUT, "frontier.json")) else None
    o = []
    P = o.append
    P(f"E1 MAIN COMPARISON AT A MATCHED FALSE-FIRING BUDGET (+ E1b frontier, E1c matched coverage, E7 near-scope negatives)      rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/e1_render.py")
    P(f"driver fx {fx.code_fingerprint()}; READY.json stage {jload(FINAL + '/common/READY.json')['stage']}; reduced programme (R19 a): behaviour at alpha = 0.05, calibration draws 0-2, development logs;")
    P("gate level: 20 calibration draws, alpha in {0.01, 0.02, 0.05, 0.10, 0.20}, n = 200 REPAIR-pool queries; working points: gate level only.")
    P("INTERVALS.  Behavioural cells of calibrated rows: value = mean over the 3 calibration draws, [95 % crossed bootstrap over draws x units, 2000 resamples] (units: edits for edit and")
    P("  hard-negative metrics, query clusters for probe metrics); '(n draws = 3)' applies to every such interval: with 3 draws it is too narrow where the draws disagree, so the three draws are")
    P("  listed next to it as d0/d1/d2.  Rows without calibration: one run, [95 % percentile bootstrap over units].  Gate-level cells: mean [5th, 95th percentile] over the 20 draws.")
    P("FAR COLUMNS.  heldout_disjoint / heldout_strict: behaviour (teacher forcing, B = 1).  BEN-T and MME-T: GATE LEVEL ONLY, fx.pools.firing; gates are calibrated at teacher-forced REPAIR keys,")
    P("  BEN-T rows are read at PROMPT-ONLY keys (MME-T at its teacher-forced key); no D1 on these pools (R19 e).  The retrieval family and lookup are NOT DEFINED on the MME pools.")
    P("NEAR SCOPE is printed next to every far rate (R6): HN-I / HN-Q / HN-X = share of valid near-scope negatives on which an edit is released.")
    P("CAPPED EDITS void the conformal bound for those edits: 'capped' = fx.methods.n_capped; 'literal' = false firing of the literal rule without the guard cap.")
    P(f"H3, unreachable draws: {e1c.H3_UNREACHABLE_RULE}.")
    P(f"Missing behavioural cells at render time: {MISS['n']}" + ("" if not MISS["n"] else "  " + "; ".join(sorted({m['label'] + '/' + m['part'] + '/' + SHORT[m['log']] for m in MISS['missing_behavioural_cells']}))[:1500]))
    E2R, e2note = e2_rows()
    P("Rows COMP-B, REEDIT-B (all reported complaints) and 'ReScope-W with two-block thresholds' (R24), table (2b) of each log: " + e2note)
    P("")
    P("SELECTION IN FAVOUR OF THE BASELINE (out/best_cosine_gate.json; gate level, development logs, alpha = 0.05, 20 draws; written before any behavioural run of these gates)")
    for bb, e in SELF["backbones"].items():
        s = SELF["selected"][bb]
        P(f"  {bb} ({e['log']}): best gate of the cosine family for H3 = {s['cosine']['H3']} (_x form {s['cosine']['H3_x']}), for H2 = {s['cosine']['H2']} (_x form {s['cosine']['H2_x']}); retr for H3 = {s['retr']['H3']}, for H2 = {s['retr']['H2']}")
        for fam in ("table_cosine_family", "table_retrieval_family"):
            P("    " + TT(["variant", "H3 coverage", "draws unreachable", "edits at cap (H3)", "H3 test near-scope fire", "H2 coverage", "fire hd", "fire hd literal", "capped mean", "capped max"],
                          [[nm, f(r["H3_coverage_mean"], 2), r["H3_n_draws_unreachable"], f(r["H3_n_edits_at_cap_mean"]), f(r["H3_test_mean_fired_mean"]), f(r["H2_coverage_mean"], 2),
                            f(r["heldout_disjoint_fired_mean"], 2), f(r["heldout_disjoint_fired_literal_mean"], 2), f(r["n_capped_mean"]), r["n_capped_max"]] for nm, r in e[fam].items()]).replace("\n", "\n    "))
    P(f"  rules: {SELF['rules']['ties']}; retr_sum / retr_min release by margin; the _x form of a shared-scale gate is the two-block _x form (R25).")
    for log in DEV:
        L = fx.logs.open(log)
        sel = e1c.selected_labels(L)
        order, name, cal = rows_for(sel)
        d = L.describe()
        P("")
        P("#" * 200)
        P(f"{log}   ({L.backbone}, T = {L.T});  heldout_disjoint n = {d['pools_valid']['heldout_disjoint']}, heldout_strict n = {d['pools_valid']['heldout_strict']}; hard negatives (valid) "
          f"{ {k: v for k, v in d['hard_negatives'].items() if 'invalid' not in k} }, excluded for an invalid pre-edit answer { {k: v for k, v in d['hard_negatives'].items() if 'invalid' in k} }; "
          f"probe rows excluded (invalid pre-edit answer): {d['n_probe_rows'] - d['n_probe_valid']}")
        P("#" * 200)
        # ---- T1 edits
        P("(1) MAIN TABLE, EDITS (behaviour; exact %, all 100 edits; R7: kept = exact under the deployed log and under the method, gained = not exact under the deployed log, exact under the method, % of all edits)")
        rows = []
        for lab in order:
            c = lab in cal
            if I.beh(log, lab, "edit.rel.exact", "edits", c) is None:
                rows.append([name[lab], "not run / missing"] + [""] * 9)
                continue
            rows.append([name[lab], "3" if c else "-"] + [ci(I.beh(log, lab, f"edit.{m}.exact", "edits", c)) for m in ("rel", "tgen", "igen", "rephrase")] +
                        [(I.by_draw(log, lab, "edit.tgen.exact", "edits", 0) + " | " + I.by_draw(log, lab, "edit.igen.exact", "edits", 0)) if c else "-"] +
                        [f(getattr(I.beh(log, lab, f"edit.{m}.token", "edits", c), "get", lambda k: None)("value")) for m in ("tgen", "igen")] +
                        ["/".join(f(getattr(I.beh(log, lab, f"kg.{m}.{k}", "edits", c), "get", lambda k: None)("value"), 0) for k in ("kept", "gained")) for m in ("tgen", "igen")])
        P(TT(["method", "n draws", "Rel", "T-Gen", "I-Gen", "(T-Gen + I-Gen) / 2", "T-Gen d0/d1/d2 | I-Gen d0/d1/d2", "T-Gen tok", "I-Gen tok", "T-Gen kept/gained", "I-Gen kept/gained"], rows))
        P("")
        # ---- T1b locality
        P("(2) MAIN TABLE, FAR LOCALITY WITH NEAR-SCOPE FIRING NEXT TO IT (behaviour: fire = an edit is released, D1 = the pre-edit answer is not reproduced, %; gate columns: mean over 20 draws)")
        rows = []
        for lab in order:
            c = lab in cal
            if I.beh(log, lab, "probe.D1", "heldout_disjoint", c) is None:
                rows.append([name[lab], "not run / missing"] + [""] * 14)
                continue
            rows.append([name[lab], ci(I.beh(log, lab, "probe.fired", "heldout_disjoint", c), 2), ci(I.beh(log, lab, "probe.D1", "heldout_disjoint", c), 2),
                         I.by_draw(log, lab, "probe.D1", "heldout_disjoint", 2) if c else "-", ci(I.beh(log, lab, "probe.D1", "heldout_strict", c), 2),
                         f(getattr(I.beh(log, lab, "probe.T-Loc*", "heldout_disjoint", c), "get", lambda k: None)("value"), 2), f(getattr(I.beh(log, lab, "probe.I-Loc*", "heldout_disjoint", c), "get", lambda k: None)("value"), 2),
                         gs(G, log, lab, "BEN-T.fired", nd=2, spread=False), gs(G, log, lab, "MME-T.fired", nd=2, spread=False) if lab not in RETR_FAMILY + ("lookup",) else "not defined"] +
                        [f(getattr(I.beh(log, lab, f"hn.{k}.fired", "hard_negatives", c), "get", lambda k: None)("value")) + " / " + ci(I.beh(log, lab, f"hn.{k}.D1", "hard_negatives", c)) for k in HN] +
                        [gs(G, log, lab, "n_capped", nd=1, spread=False), gs(G, log, lab, "fire.hd", nd=2, spread=False), gs(G, log, lab, "fire.hd.literal", nd=2, spread=False)])
        P(TT(["method", "fire hd", "D1 hd", "D1 hd d0/d1/d2", "D1 strict", "T-Loc*", "I-Loc*", "BEN-T fire (gate)", "MME-T fire (gate)", "NEAR HN-I fire / D1", "NEAR HN-Q fire / D1", "NEAR HN-X fire / D1",
              "capped", "gate fire hd", "gate literal"], rows))
        P("   dual: its teacher-forced cells are the labelled composition ses.dual_tf (the model DUAL selects from its two decodes); 'fire' = the deployed answer was released.")
        P("   deployed / pre-edit / z-score rows have no per-edit release; the router and z-score gates release the deployed output (no per-edit decision).")
        P("")
        P("(2b) ROWS TAKEN FROM E2 (e2_curves/out; evaluated there at B = 1 through the same driver; linear repairs: one run, [95 % bootstrap]; ReScope-W two-block: mean over draws 0-2, D2 at draw 0; a linear repair has no")
        P("     per-edit release, so 'fire' is '-'); n evidence = number of reported complaints (linear repairs) / benign calibration queries (ReScope-W)")
        P(e2_table(E2R[log]) if log in E2R else "     EMPTY: " + e2note)
        P("")
        # ---- T3 generation
        P("(3) GENERATED ANSWERS AND D2 (decoder v2; calibration draw 0 for calibrated rows; rows of the main table + dual).  edit prompts: the first line equals / contains the target (%);")
        P("    probes: 10-token answer equals the pre-edit answer (% of heldout_disjoint); D2 = % of the pre-correct heldout_disjoint rows whose 48-token answer loses the reference [95 % bootstrap, query clusters];")
        P("    near-scope negatives: eq pre = answer equals the pre-edit answer; wrong target = answer equals the target of the released edit; eq any = equals the target of any edit (all rows; a row on which")
        P("    nothing is released in the decode is the pre-edit model and is not decoded).")
        rows = []
        for lab, dr in [("deployed", None), ("preedit", None), ("rescope", 0), ("rescope_x", 0), ("where_logonly", None)] + [(b, 0) for b in dict.fromkeys([sel["cos_H3"], sel["cos_H2"]])] + \
                       [(r, 0) for r in dict.fromkeys([sel["retr_H3"], sel["retr_H2"]])] + [("dual", None)]:
            g = lambda m, pool: I.get(log, lab, m, pool, dr)                                          # noqa: E731
            if g("gen.rel.eq", "edit_prompts") is None:
                rows.append([name[lab], "not run / missing"] + [""] * 9)
                continue
            v = lambda r, nd=1: f(r["value"], nd) if r else "-"                                       # noqa: E731
            rows.append([name[lab]] + [v(g(f"gen.{m}.eq", "edit_prompts")) + " / " + v(g(f"gen.{m}.contains", "edit_prompts")) for m in ("rel", "tgen", "igen")] +
                        [ci(g("gen.probe.eq_pre", "heldout_disjoint"), 2), ci(g("d2.D2", "heldout_disjoint"), 2) + f" (n = {v(g('d2.n_pre_correct', 'heldout_disjoint'), 0)})"] +
                        [v(g(f"hngen.{k}.eq_pre", "hard_negatives")) + " / " + v(g(f"hngen.{k}.wrong_target", "hard_negatives")) + " / " + v(g(f"hngen.{k}.eq_any_edit_all_rows", "hard_negatives")) for k in HN])
        P(TT(["method", "Rel eq / contains", "T-Gen eq / contains", "I-Gen eq / contains", "probes eq pre", "D2 rate of pre-correct", "HN-I eq pre / wrong target / eq any",
              "HN-Q eq pre / wrong target / eq any", "HN-X eq pre / wrong target / eq any"], rows))
        P("")
        # ---- T4 gate level at alpha = 0.05
        P("(4) GATE LEVEL, alpha = 0.05, n = 200, 20 calibration draws (CPU, key caches; dense gates: built with the model's weights): mean [5th, 95th percentile over draws]")
        glabs = [x for x in ("rescope", "rescope_shared", "rescope_null", "rescope_nullW", "rescope_pac", "rescope_x") if x in G.get(log, {})] + list(COS_FAMILY) + \
                [x for x in dict.fromkeys([sel["cos_H3_x"], sel["cos_H2_x"]])] + list(RETR_FAMILY) + ["lookup", "ball", "ball_default", "cosine_default", "exact", "router", "router_default", "zscore", "zscore_default"]
        P(gate_table(G, log, glabs))
        P("   own silenced b/a = own prompts (of 100) on which the own edit is not a candidate under the literal rule (before the cap) / as run (after the cap); cand = own edit is a candidate; other = another edit is")
        P("   released on the rephrase; lineage = mean number of candidate edits on a firing heldout_disjoint row.  Dense gates: 'cov' = the gate fires on the rephrase (it releases the deployed output).")
        P("   z-score, cosine at all positions and ReScope-W are not decided from last-position keys: their firing is in table (2) (behaviour).")
        P("")
        P("(4b) NEAR-SCOPE NEGATIVES AT GATE LEVEL, alpha = 0.05 (E7): % of rows on which an edit is released / on which the row's own edit is released; mean over 20 draws")
        P(TT(["method", "HN-I an edit / own", "HN-Q an edit / own", "HN-X an edit / own a", "mean of kinds", "lineage width on HN"],
             [[lab] + [gs(G, log, lab, f"hn.{k}.fired", spread=False) + " / " + gs(G, log, lab, f"hn.{k}.own", spread=False) for k in HN] +
              [gs(G, log, lab, "hn.mean.fired"), gs(G, log, lab, "lineage.hn.width_mean", nd=2, spread=False)] for lab in glabs if gd(G, log, lab, "hn.HN-I.fired")]))
        P("   _x methods, second form (test hard negatives whose foreign part occurs in no recombination row of the log; whole set, unraised scale):")
        P(TT(["method", "HN-I all (n)", "HN-I unseen (n)", "HN-Q all (n)", "HN-Q unseen (n)", "HN-X unseen: n rows only"],
             [[lab] + [gs(G, log, lab, f"{p}.{k}.fired", spread=False) + f" ({f((gd(G, log, lab, f'{p}.{k}.n') or [None])[0], 0)})" for k in ("HN-I", "HN-Q") for p in ("ns", "nsu")] +
              [f((gd(G, log, lab, "nsu.HN-X.n") or [None])[0], 0)] for lab in glabs if gd(G, log, lab, "nsu.HN-I.fired")]))
        P("")
        # ---- T5 alphas
        P("(5) GATE LEVEL ALONG alpha (mean over 20 draws): false firing on heldout_disjoint as run (literal rule) | (T-Gen + I-Gen) / 2 coverage | mean near-scope firing over the three kinds | capped edits")
        cl = [x for x in glabs if x in G.get(log, {}) and any(a in G[log][x] for a in ("0.01", "0.05"))]
        for metric, ttl, nd in (("fire.hd", "false firing on heldout_disjoint, as run (literal)", 2), ("h2.cov", "(T-Gen + I-Gen) / 2 coverage", 1), ("hn.mean.fired", "near-scope firing, mean of HN-I / HN-Q / HN-X", 1),
                                ("BEN-T.fired", "firing on BEN-T (prompt-only keys)", 2), ("n_capped", "capped edits", 1)):
            rows = []
            for lab in cl:
                if metric == "h2.cov" and gd(G, log, lab, "h2.cov") is None:      # dense gates
                    continue
                rows.append([lab] + [gs(G, log, lab, metric, afmt(a), nd, spread=False) + (f" ({gs(G, log, lab, 'fire.hd.literal', afmt(a), nd, spread=False)})" if metric == "fire.hd" else "") for a in ALPHAS])
            P(f"    {ttl}")
            P("    " + TT(["method"] + [f"alpha = {a}" for a in ALPHAS], rows).replace("\n", "\n    "))
        P("")
        # ---- T6 H3
        P("(6) READ-OUT H3 (R6 / R25): hard negatives split by the parity of the stream position of the row's own edit (HN-Q: the edit whose question it asks); the common scale is raised (guard caps kept) until the mean")
        P("    firing on the FIT rows is at most 10 %; read on the ODD edits and the TEST rows.  Gate level: 20 draws, mean [5th, 95th]; behaviour: draws 0-2, crossed bootstrap (n draws = 3).")
        hl = list(dict.fromkeys(["rescope", "rescope_x", sel["cos_H3"], sel["cos_H3_x"], sel["retr_H3"]]))
        P(TT(["method", "unreachable draws", "raise", "edits at cap", "coverage (T+I)/2", "T-Gen cov", "I-Gen cov", "test HN-I fire", "test HN-Q fire", "test HN-X fire", "test mean", "fire hd"],
             [[lab, f"{int(sum(1 - x for x in gd(G, log, lab, 'h3.reachable')))}/20"] + [gs(G, log, lab, m, nd=n_) for m, n_ in (("h3.delta", 3), ("h3.n_at_cap", 1), ("h3.cov", 1), ("h3.tgen", 1), ("h3.igen", 1),
              ("h3.test.HN-I.fired", 1), ("h3.test.HN-Q.fired", 1), ("h3.test.HN-X.fired", 1), ("h3.test.mean", 1), ("h3.hd.fired", 2))] for lab in hl if gd(G, log, lab, "h3.reachable")]))
        P("    _x methods, TEST read-out on the subset with an unseen foreign part (same fit rows, same raise): HN-I, HN-Q; HN-X has almost no rows (count given)")
        P(TT(["method", "test HN-I fire (n)", "test HN-Q fire (n)", "HN-X rows in the subset"],
             [[lab] + [gs(G, log, lab, f"h3u.test.{k}.fired") + f" ({f((gd(G, log, lab, f'h3u.test.{k}.n') or [None])[0], 0)})" for k in ("HN-I", "HN-Q")] + [f((gd(G, log, lab, "h3u.test.HN-X.n") or [0])[0], 0)]
              for lab in hl if gd(G, log, lab, "h3u.reachable")]))
        P("    behaviour at the raised scale (draws 0-2):")
        rows = []
        for lab in hl:
            g = lambda m, pool: I.get(log, lab, m, pool)                                              # noqa: E731
            r0 = g("h3.rephrase.exact@x", "odd_edits")
            if r0 is None:
                rows.append([lab, "not run / missing"] + [""] * 8)
                continue
            rows.append([lab, str(r0.get("n_edits_at_cap")), ci(r0), ci(g("h3.tgen.exact@x", "odd_edits")), ci(g("h3.igen.exact@x", "odd_edits"))] +
                        ["/".join(f(getattr(g(f"h3.test.{k}.{m}@x", "test_hard_negatives"), "get", lambda k: None)("value")) for m in ("fired", "D1", "wrong_target")) for k in HN] +
                        [ci(g("h3.probe.fired@x", "heldout_disjoint"), 2), ci(g("h3.probe.D1@x", "heldout_disjoint"), 2)])
        P(TT(["method", "edits at cap d0,d1,d2", "(T+I)/2 exact, odd edits", "T-Gen exact", "I-Gen exact", "test HN-I fire/D1/wrong target", "test HN-Q fire/D1/wrong target", "test HN-X fire/D1/wrong target", "fire hd", "D1 hd"], rows))
        rows = []
        for lab in hl:
            if lab.endswith("_x") and I.get(log, lab, "h3u.test.HN-I.fired@x", "test_hard_negatives"):
                g = lambda m: I.get(log, lab, m, "test_hard_negatives")                              # noqa: E731
                rows.append([lab] + ["/".join(f(getattr(g(f"h3u.test.{k}.{m}@x"), "get", lambda k: None)("value")) for m in ("fired", "D1", "wrong_target")) for k in ("HN-I", "HN-Q")] +
                            [f(getattr(g("h3u.test.HN-X.n"), "get", lambda k: None)("value"), 0)])
        if rows:
            P("    _x methods, behaviour on the test rows with an unseen foreign part:")
            P(TT(["method", "test HN-I fire/D1/wrong target", "test HN-Q fire/D1/wrong target", "HN-X rows in the subset"], rows))
        P("")
        # ---- T7 endpoints
        ep = EP.get(log, {})
        P("(7) CONFIRMATORY COMPARISONS (paired crossed bootstrap over calibration draws x edits; n draws = 3 for behaviour, 20 for the gate-level coverage form; Holm within H2 and within H3 over both backbones)")
        P("    H2: (T-Gen + I-Gen) / 2 exact, all edits, ReScope minus the baseline (points)")
        rows = []
        for role, e in ep.get("H2", {}).items():
            if e.get("missing"):
                rows.append([role, e["method"], "missing"] + [""] * 8)
                continue
            g = e.get("gate_coverage_form_20_draws")
            rows.append([role, e["method"], f(e["mean_rescope"]), f(e["mean_other"]), f"{e['diff']:+.1f} [{e['ci_lo']:+.1f}, {e['ci_hi']:+.1f}]", "/".join(f"{x:+.1f}" for x in e["by_draw"]), f"{e['p']:.4f}",
                         f"{e.get('p_holm', float('nan')):.4f}", f"{g['value']:+.1f} [{g['ci_lo']:+.1f}, {g['ci_hi']:+.1f}]" if g else "-",
                         f"{e['rel_exact_diff']['value']:+.1f} [{e['rel_exact_diff']['ci_lo']:+.1f}, {e['rel_exact_diff']['ci_hi']:+.1f}]",
                         f"{e['near_scope_D1_diff']['value']:+.1f} [{e['near_scope_D1_diff']['ci_lo']:+.1f}, {e['near_scope_D1_diff']['ci_hi']:+.1f}]"])
        P(TT(["baseline", "method", "ReScope", "baseline", "difference [95 %] (3 draws)", "d0/d1/d2", "p", "p Holm", "gate coverage form (20 draws)", "secondary: Rel exact diff", "secondary: near-scope D1 diff"], rows))
        s2 = ep.get("secondary", {}).get("D1_heldout_disjoint_rescope_minus_deployed")
        if s2:
            P(f"    secondary: D1 on heldout_disjoint, ReScope minus the deployed log: {s2['value']:+.2f} [{s2['ci_lo']:+.2f}, {s2['ci_hi']:+.2f}] points (ReScope {s2['mean_a']:.2f}, deployed {s2['mean_b']:.2f}; 3 draws)")
        P("    calibrated z-score gate: gate level only (not a behavioural row of the reduced programme); exact match and dual have no calibrated form.")
        P("    H3: (T-Gen + I-Gen) / 2 exact on the odd edits under the near-scope constraint, a minus b (points)")
        rows = []
        for pair, e in ep.get("H3", {}).items():
            g = e.get("gate_coverage_form_20_draws")
            gtxt = f"{g['value']:+.1f} [{g['ci_lo']:+.1f}, {g['ci_hi']:+.1f}]" if g else "-"
            if e.get("missing"):
                rows.append([pair, e["a"], e["b"], "missing", "", "", "", "", "", gtxt, ""])
                continue
            rows.append([pair, e["a"], e["b"], f(e["mean_a"]), f(e["mean_b"]), f"{e['diff']:+.1f} [{e['ci_lo']:+.1f}, {e['ci_hi']:+.1f}]", "/".join(f"{x:+.1f}" for x in e["by_draw"]), f"{e['p']:.4f}",
                         f"{e.get('p_holm', float('nan')):.4f}", gtxt, f"{e['n_edits_at_cap_a']} / {e['n_edits_at_cap_b']}"])
        P(TT(["pair", "a", "b", "a", "b", "difference [95 %] (3 draws)", "d0/d1/d2", "p", "p Holm", "gate coverage form (20 draws)", "edits at cap a / b (d0,d1,d2)"], rows))
        P("")
        # ---- T8 hard-negative subsets
        P("(8) NEAR-SCOPE NEGATIVES BY THE SUBSETS THE CACHE FLAGS (behaviour; fire / D1 in %, mean over draws 0-2 for calibrated rows; n rows): same category as the edit or not, related or not,")
        P("    new or existing rows, degenerate pre-edit answer (empty, one character, '?', bare None / No) or not")
        hdr = ["method", "kind"] + [f"{nm}={v} fire/D1 (n)" for _, nm in (("same_category", "samecat"), ("related_flag", "related"), ("new", "new"), ("y0_deg", "y0deg")) for v in (1, 0)]
        rows = []
        for lab in ["deployed", "rescope", "rescope_x", sel["cos_H3"], sel["retr_H3"], "cosine_shared"]:
            c = lab in cal
            for k in HN:
                cells = []
                for nm in ("samecat", "related", "new", "y0deg"):
                    for v in (1, 0):
                        def val(m):
                            rs = [I.get(log, lab, f"hnsub.{k}.{nm}={v}.{m}", "hard_negatives", d_) for d_ in ((0, 1, 2) if c else (None,))]
                            rs = [r["value"] for r in rs if r is not None and r["value"] is not None]
                            return float(np.mean(rs)) if rs else None
                        n_ = val("n")
                        cells.append("-" if n_ is None else f"{f(val('fired'))}/{f(val('D1'))} ({n_:.0f})")
                rows.append([name[lab] if k == "HN-I" else "", k] + cells)
        P(TT(hdr, rows))
        P("")
        if L.backbone == "llava":
            P("(9) THE GATE AT BOTH POSITIONS (LLaVA): sequences whose teacher-forced last prompt position is one token after the decode's (lp_shift rows: "
              f"{d.get('lp_shift_rows')}); behavioural rows re-evaluated at the decode's position; metrics are printed only where a row differs")
            rows = []
            for (lg, lab, metric, pool, dr), r in I.d.items():
                if lg == log and metric == "prompt_position.n_rows_differ":
                    ns = I.get(log, lab, "prompt_position.n_shift_rows", pool, dr)
                    extra = ""
                    if r["value"]:
                        a = [(m, pl) for m, pl in (("edit.tgen.exact", "edits"), ("edit.igen.exact", "edits"), ("probe.fired", "heldout_disjoint"), ("probe.D1", "heldout_disjoint"), ("hn.HN-I.fired", "hard_negatives"),
                                                    ("hn.HN-Q.fired", "hard_negatives"), ("hn.HN-X.fired", "hard_negatives"))]
                        extra = "; ".join(f"{m} {f(getattr(I.get(log, lab, m, pl, dr), 'get', lambda k: None)('value'), 2)} -> {f(getattr(I.get(log, lab, m + '@prompt', pl, dr), 'get', lambda k: None)('value'), 2)}" for m, pl in a)
                    rows.append([lab, "-" if dr is None else dr, f(ns["value"], 0), f(r["value"], 0), extra])
            P(TT(["method", "draw", "lp_shift rows", "rows whose decision or outcome differs", "tf position -> decode's position"], rows))
            P("   gate level (20 draws, alpha = 0.05): sequences whose gate decision differs between the two positions, mean over draws (all probe, edit and hard-negative rows)")
            P(TT(["method", "rows that differ", "fire hd tf -> prompt", "T-Gen cov tf -> prompt", "I-Gen cov tf -> prompt"],
                 [[lab, gs(G, log, lab, "prompt.n_rows_differ", spread=False), gs(G, log, lab, "fire.hd", nd=2, spread=False) + " -> " + gs(G, log, lab, "fire.hd@prompt", nd=2, spread=False),
                   gs(G, log, lab, "cov.tgen", spread=False) + " -> " + gs(G, log, lab, "cov.tgen@prompt", spread=False), gs(G, log, lab, "cov.igen", spread=False) + " -> " + gs(G, log, lab, "cov.igen@prompt", spread=False)]
                  for lab in glabs if gd(G, log, lab, "prompt.n_rows_differ")]))
            P("   ('@prompt' means are over the draws in which at least one row differs.)")
            P("")
        # ---- E1b / E1c
        if FR and log in FR["logs"]:
            fr = FR["logs"][log]
            P("(10) E1c MATCHED-COVERAGE READ-OUT (gate level; one global threshold per feature, no guards): the largest threshold at which the coverage of the union of text and image rephrases reaches the target;")
            P("     there: false firing on heldout_disjoint and the release rate on each kind of near-scope negative (all rows / test rows of the H3 split).  Curves and calibrated points: out/frontier.json (E1b).")
            rows = []
            feats = ["ncoord", "cos"] + [x for x in dict.fromkeys([fx.review1.parse(sel["cos_H3"])["base"], fx.review1.parse(sel["cos_H2"])["base"]]) if x in fr["matched_coverage"]]
            for feat in feats:
                for m in fr["matched_coverage"][feat]:
                    if not m["reachable"]:
                        rows.append([feat, m["target"], f"not reachable (largest coverage {f(m.get('largest_coverage'))})"] + [""] * 7)
                    else:
                        rows.append([feat, m["target"], f(m["threshold"], 4), f(m["coverage"]), f(m["heldout_disjoint_fired"], 2)] + [f(m["hn_fired"][k]) for k in HN] + [f(m["hn_mean_fired"]), f(m["hn_test_mean_fired"])])
            P(TT(["feature", "target coverage %", "threshold", "coverage", "fire hd", "HN-I", "HN-Q", "HN-X", "mean of kinds", "mean on test rows"], rows))
            P("     E1b calibrated points on the frontiers (mean over 20 draws): fire hd | (T+I)/2 coverage | mean near-scope firing on the test hard negatives")
            rows = []
            for feat in feats:
                for key, pt in fr["features"][feat]["calibrated_points"].items():
                    rows.append([feat, pt["method"], pt["alpha"], f(pt["probe.heldout_disjoint.fired"], 2), f(pt["edit.rephrase.own_top"]), f(pt["hn.test.mean.fired"]), f(pt["n_capped"])])
            for rule, e in fr["retrieval"].items():
                for key, pt in e["calibrated_points"].items():
                    rows.append([rule, pt["method"], pt["alpha"], f(pt["probe.heldout_disjoint.fired"], 2), f(pt["edit.rephrase.own_top"]), f(pt["hn.test.mean.fired"]), f(pt["n_capped"])])
            P(TT(["feature", "calibrated method", "alpha", "fire hd", "(T+I)/2 coverage", "test near-scope fire (mean of kinds)", "capped"], rows))
            P("")
    # ---- decision
    P("#" * 200)
    P("DECISION (out/decision.json) -- " + DEC["rule"])
    for pair, dcs in DEC["pairs"].items():
        P(f"  {pair}: {dcs['verdict'].upper()}   ({dcs['claim_form']})")
        for log, c in dcs["per_log"].items():
            if c.get("missing"):
                P(f"      {log}: missing")
                continue
            P(f"      {log}: {c['a']} minus {c['b']}: behaviour {c['behaviour_diff_points']:+.1f} [{c['behaviour_interval'][0]:+.1f}, {c['behaviour_interval'][1]:+.1f}] points (3 draws: "
              f"{'/'.join(f'{x:+.1f}' for x in c['behaviour_by_draw'])}; p Holm {c['p_holm'] if c['p_holm'] is None else round(c['p_holm'], 4)}); gate-level coverage form {c['gate_coverage_diff_points']:+.1f} "
              f"[{c['gate_coverage_interval'][0]:+.1f}, {c['gate_coverage_interval'][1]:+.1f}] (20 draws); >= 10 points: {c['at_least_10_points']}; behaviour interval excludes 0: {c['behaviour_interval_excludes_0']}; "
              f"gate interval excludes 0 with the same sign: {c['gate_interval_excludes_0_same_sign']}; edits at cap a {c['n_edits_at_cap_a']} b {c['n_edits_at_cap_b']}")
    P("")
    # ---- working points
    P("#" * 200)
    P("WORKING POINTS (the same edit streams at other ridges of the editor): gate level only, alpha = 0.05, 20 draws; mean [5th, 95th percentile]")
    for log in WP:
        if log not in G:
            P(f"{log}: no gate-level cells")
            continue
        L = fx.logs.open(log)
        sel = e1c.selected_labels(L)
        glabs = [x for x in ("rescope", "rescope_shared", "rescope_null", "rescope_nullW", "rescope_pac", "rescope_x")] + list(COS_FAMILY) + [x for x in dict.fromkeys([sel["cos_H3_x"], sel["cos_H2_x"]])] + \
            list(RETR_FAMILY) + ["lookup", "ball", "ball_default", "cosine_default", "exact", "router", "router_default", "zscore", "zscore_default"]
        P("")
        P(f"{log}  ({L.describe()['editor']})")
        P(gate_table(G, log, glabs))
        P("   near scope and H3 at gate level:")
        P(TT(["method", "HN-I an edit / own", "HN-Q an edit / own", "HN-X an edit / own a", "H3 coverage", "H3 unreachable draws", "H3 edits at cap", "H3 test mean fire"],
             [[lab] + [gs(G, log, lab, f"hn.{k}.fired", spread=False) + " / " + gs(G, log, lab, f"hn.{k}.own", spread=False) for k in HN] +
              [gs(G, log, lab, "h3.cov"), f"{int(sum(1 - x for x in gd(G, log, lab, 'h3.reachable')))}/20" if gd(G, log, lab, "h3.reachable") else "-", gs(G, log, lab, "h3.n_at_cap", spread=False),
               gs(G, log, lab, "h3.test.mean", spread=False)] for lab in glabs if gd(G, log, lab, "hn.HN-I.fired")]))
    nd_ = [r for r in recs if r["metric"] == "not_defined"]
    if nd_:
        P("")
        P("NOT DEFINED (driver: fx.review1.NotDefined): " + "; ".join(sorted({f"{r['method']} on {SHORT[r['log']]}: {r.get('note', '')[:120]}" for r in nd_})))
    with open(os.path.join(OUT, "table.txt.tmp"), "w") as fh:
        fh.write("\n".join(o) + "\n")
    os.replace(os.path.join(OUT, "table.txt.tmp"), os.path.join(OUT, "table.txt"))
    print("table.txt:", len(o), "lines")


def e2_rows():
    """The rows that the main table takes from E2 (e2_curves/out/linear_rows.json, rescope_w_rows.json; read, never rebuilt): COMP-B and
    REEDIT-B at all reported complaints, ReScope-W with two-block thresholds (R24).  -> ({log: [(name, {(metric, pool): record})]}, note);
    the records used are copied to out/e2_rows_used.json."""
    out, used, notes = {}, [], []
    meta = jload(E2 + "/rows_for_e1_meta.json") if os.path.exists(E2 + "/rows_for_e1_meta.json") else None
    if meta is None:
        return {}, "e2_curves/out/rows_for_e1_meta.json does not exist at render time: the rows COMP-B, REEDIT-B and ReScope-W (two-block) are EMPTY (not rebuilt here)"
    lin = jload(E2 + "/linear_rows.json") if os.path.exists(E2 + "/linear_rows.json") else []
    cw = jload(E2 + "/rescope_w_rows.json") if os.path.exists(E2 + "/rescope_w_rows.json") else []
    for log in DEV:
        rows = []
        for name, src, flt, mean in (("COMP-B, all reported complaints (E2: rebuilt, seed 0)", lin, lambda r: r["method"] == "COMP_B" and r.get("n_label") == "ALL", False),
                                     ("REEDIT-B, all reported complaints (E2: saved weight of P12)", lin, lambda r: r["method"] == "REEDIT_B" and r.get("n_label") == "ALL", False),
                                     ("ReScope-W, two-block thresholds (E2 c'; draws 0-2)", cw, lambda r: r["method"] == "where_cal", True)):
            rs = [r for r in src if r["log"] == log and flt(r)]
            d = {}
            for r in rs:
                m = r["metric"]
                if mean:
                    if r["draw"] is None and m.endswith("@mean"):
                        d[(m[:-5], r["pool"])] = r
                    elif r["draw"] == 0 and m.startswith(("d2.", "gen.", "hngen.")):
                        d.setdefault((m, r["pool"]), r)
                else:
                    d[(m, r["pool"])] = r
            rows.append((name, d, rs[0].get("n") if rs else None))
            used += rs
            if not rs:
                notes.append(f"{SHORT[log]}: {name.split(',')[0]} missing")
        out[log] = rows
    jdump({"read_from": E2, "read_at": time.strftime("%Y-%m-%d %H:%M:%S"), "e2_meta": meta, "records": used}, os.path.join(OUT, "e2_rows_used.json"))
    note = f"read from e2_curves/out at render time (E2 rendered {meta.get('rendered')}; E2 lists as missing: {meta.get('missing')})" + ("; EMPTY rows: " + "; ".join(notes) if notes else "; all three rows present for both logs")
    return out, note


def e2_table(rows):
    def g(d, m, pool, nd=1, with_ci=True):
        r = d.get((m, pool))
        if r is None or r.get("value") is None:
            return "-"
        return ci(r, nd) if with_ci else f(r["value"], nd)
    body = []
    for name, d, n in rows:
        if not d:
            body.append([name, "EMPTY (no E2 record)"])
            continue
        body.append([name, "-" if n is None else str(n), g(d, "edit.rel.exact", "edits"), g(d, "edit.tgen.exact", "edits"), g(d, "edit.igen.exact", "edits"), g(d, "edit.rephrase.exact", "edits"),
                     g(d, "kg.tgen.kept", "edits", 0, False) + "/" + g(d, "kg.tgen.gained", "edits", 0, False), g(d, "kg.igen.kept", "edits", 0, False) + "/" + g(d, "kg.igen.gained", "edits", 0, False),
                     g(d, "probe.fired", "heldout_disjoint", 2, False), g(d, "probe.D1", "heldout_disjoint", 2), g(d, "probe.D1", "heldout_strict", 2), g(d, "probe.T-Loc*", "heldout_disjoint", 2, False),
                     g(d, "probe.I-Loc*", "heldout_disjoint", 2, False)] + [g(d, f"hn.{k}.fired", "hard_negatives", 1, False) + " / " + g(d, f"hn.{k}.D1", "hard_negatives") for k in HN] +
                    [g(d, "d2.D2", "heldout_disjoint", 2) + f" (n = {g(d, 'd2.n_pre_correct', 'heldout_disjoint', 0, False)})"])
    return TT(["method (records of E2)", "n evidence", "Rel", "T-Gen", "I-Gen", "(T-Gen + I-Gen) / 2", "T-Gen kept/gained", "I-Gen kept/gained", "fire hd", "D1 hd", "D1 strict", "T-Loc*", "I-Loc*",
               "NEAR HN-I fire / D1", "NEAR HN-Q fire / D1", "NEAR HN-X fire / D1", "D2 rate of pre-correct"], body)


def gate_table(G, log, labs):
    rows = []
    for lab in labs:
        if gd(G, log, lab, "fire.hd") is None:
            continue
        dense = gd(G, log, lab, "cov.tgen") is None
        cov = (lambda k: gs(G, log, lab, f"fired.{k}")) if dense else (lambda k: gs(G, log, lab, f"cov.{k}"))
        rows.append([lab, gs(G, log, lab, "fire.hd", nd=2), gs(G, log, lab, "fire.hd.literal", nd=2, spread=False), gs(G, log, lab, "fire.strict", nd=2, spread=False) + " (" + gs(G, log, lab, "fire.strict.literal", nd=2, spread=False) + ")",
                     gs(G, log, lab, "fire.hd.image", nd=2, spread=False) + " / " + gs(G, log, lab, "fire.hd.text", nd=2, spread=False),
                     gs(G, log, lab, "fire.strict.image", nd=2, spread=False) + " / " + gs(G, log, lab, "fire.strict.text", nd=2, spread=False),
                     gs(G, log, lab, "BEN-T.fired", nd=2), gs(G, log, lab, "MME-T.fired", nd=2) if lab not in RETR_FAMILY + ("lookup",) else "not defined", cov("tgen"), cov("igen"),
                     gs(G, log, lab, "cand.tgen", spread=False) + " / " + gs(G, log, lab, "cand.igen", spread=False), gs(G, log, lab, "other.tgen", spread=False) + " / " + gs(G, log, lab, "other.igen", spread=False),
                     gs(G, log, lab, "own.silenced_before_cap", spread=False) + " / " + gs(G, log, lab, "own.not_candidate_after_cap", spread=False), gs(G, log, lab, "n_capped"),
                     gs(G, log, lab, "lineage.hd.width_mean", nd=2, spread=False)])
    return TT(["method", "fire hd (as run)", "literal", "fire strict (literal)", "hd image / text", "strict image / text", "BEN-T fire (prompt-only keys)", "MME-T fire", "T-Gen cov", "I-Gen cov",
               "cand T / I", "other T / I", "own silenced b/a", "capped edits", "lineage"], rows)


if __name__ == "__main__":
    main()
