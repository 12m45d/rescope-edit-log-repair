"""E3 / E4 on the LLaVA logs: renders out/table.txt from out/summary.json, out/scale_series.json, out/retention_series.json (written by
e3_analyze.py).  No number is typed by hand.      python e3_render.py
"""
import os

from e3c import *  # noqa: F401,F403

S = jload(os.path.join(OUT, "summary.json"))
SC = jload(os.path.join(OUT, "scale_series.json"))["families"]
RT = jload(os.path.join(OUT, "retention_series.json"))["families"]
MISSING = jload(os.path.join(OUT, "missing.json"))
TT = fx.io.text_table
ORDER = ["deployed", "rescope", "rescope_shared", "rescope_null", "rescope_x", "rescope_nullW", "rescope_pac", "cosine", "cosine_shared", "white_1_shared", "white_0.1_shared", "white_1_x",
         "retr_sum", "lookup", "cosine_default", "cosine_published", "exact", "where_logonly"]
NAMES = {"cosine": "cosine (two-block)", "cosine_shared": "cosine (shared)", "cosine_default": "cosine 0.65 last", "cosine_published": "cosine 0.65 all pos", "exact": "exact match",
         "where_logonly": "Rescope-WHERE log-only", "rescope": "rescope (default)"}
FAMTXT = {"ae-l2_3": "AlphaEdit-style logs (T <= 500; T = 100 is the DEVELOPMENT log, 200 and 500 are unseen)", "memitw": "MEMIT-style logs, public configuration (all unseen)"}


def labs(log):
    have = [l for l in S.get(log, {}) if not l.startswith("_")]
    return [l for l in ORDER if l in have] + sorted(l for l in have if l not in ORDER)


def v(log, lab, level, nm, f="mean", ro="reg"):
    try:
        return S[log][lab][ro][level][nm][f]
    except (KeyError, TypeError):
        return None


def f1(x, nd=1):
    return "-" if x is None else (x if isinstance(x, str) else f"{x:.{nd}f}")


def ci(log, lab, level, nm, ro="reg", nd=1):
    m = v(log, lab, level, nm, ro=ro)
    lo, hi = v(log, lab, level, nm, "ci_lo", ro), v(log, lab, level, nm, "ci_hi", ro)
    return f1(m, nd) if lo is None else f"{m:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"


def nd_beh(log, lab, ro="reg"):
    return (S[log][lab].get(ro) or {}).get("n_draws_beh")


def nd_gate(log, lab, ro="reg"):
    return (S[log][lab].get(ro) or {}).get("n_draws_gate")


def state(log, lab):
    n = S[log][lab].get("reg") or {}
    if n.get("not_defined"):
        return "not defined"
    if n.get("skipped_for_cost"):
        return "skipped (cost)"
    if "beh" not in n and "gate" in n:
        return "gate level only" if lab in ("rescope_nullW", "rescope_pac") else "behaviour missing"
    if "beh" not in n:
        return "missing"
    return ""


def logs_of(fam):
    return [l for l in FAMILIES[fam] if l in S and "_meta" in S[l]]


def name(lab):
    return NAMES.get(lab, lab)


out = []
W = out.append
W("E3 (scale) and E4 (other editor) on the LLaVA-1.5-7B logs: frozen row set, alpha = 0.05, n = 200.  Rendered by e3_render.py from out/summary.json.")
W(f"driver {S['_run']['code']}; written {S['_run']['written']}; cells not yet computed: {S['_run']['n_missing']} (out/missing.json)")
W("Gate level (g): key caches, 20 calibration draws, mean over draws; [..] = crossed bootstrap over draws x units (R10; probe units = query clusters).")
W("Behaviour (b): B = 1 forwards through the driver; calibration draws 0 - 2 (T >= 1000: draw 0, R23); 'nb' = number of behavioural draws behind a cell; with 3 draws the")
W("crossed bootstrap is too narrow when draws matter (R25): the draws are listed in table 9.  Fixed rows (no calibration) have one evaluation.")
W("Selection (driver's reader of e1_main/out/best_cosine_gate.json): " + str(next(S[l]["_meta"]["selection"] for l in LOGS if l in S and "_meta" in S[l])))
W("Not run by the registered reductions: dual (R19), two-block thresholds-only repair (R24), preservation statistics are not analysed.  rescope_nullW / rescope_pac: gate level only.")
W("")

for fam in FAMILIES:
    L = logs_of(fam)
    if not L:
        continue
    W("=" * 150)
    W(f"FAMILY {fam}: {FAMTXT[fam]}")
    W("=" * 150)
    # ---------------------------------------------------------------- table 1: edits
    W("\nTABLE 1  EDITS (b; exact match in percent).  all = all T edits; first100 = the first 100 edits of the stream; later = stream position >= 100 (edits no development step has seen).")
    W("  tok = token-level; kept / gained (R7) = exact under the deployed log and the method / not exact under the deployed log but under the method, percent of all edits; cov (g) = own edit released.")
    rows = []
    for log in L:
        T = S[log]["_meta"]["T"]
        for lab in labs(log):
            st = state(log, lab)
            b = lambda nm: f1(v(log, lab, "beh", nm))
            kg = lambda m: f"{f1(v(log, lab, 'beh', f'kg.{m}.kept'))}/{f1(v(log, lab, 'beh', f'kg.{m}.gained'))}"
            rows.append([T, name(lab), nd_beh(log, lab) or "-", ci(log, lab, "beh", "edit.rel.exact"), ci(log, lab, "beh", "edit.tgen.exact"), ci(log, lab, "beh", "edit.igen.exact"),
                         f"{b('edit.rel.token')}/{b('edit.tgen.token')}/{b('edit.igen.token')}",
                         f"{b('edit.rel.exact.first100')}/{b('edit.tgen.exact.first100')}/{b('edit.igen.exact.first100')}" if T > 100 else "= all",
                         f"{b('edit.rel.exact.later')}/{b('edit.tgen.exact.later')}/{b('edit.igen.exact.later')}" if T > 100 else "none",
                         kg("rel"), kg("tgen"), kg("igen"),
                         f"{f1(v(log, lab, 'gate', 'cov.tgen'))}/{f1(v(log, lab, 'gate', 'cov.igen'))}",
                         f"{f1(v(log, lab, 'gate', 'cov.tgen.later'))}/{f1(v(log, lab, 'gate', 'cov.igen.later'))}" if T > 100 else "none", st])
    W(TT(["T", "method", "nb", "Rel", "T-Gen", "I-Gen", "tok R/T/I", "first100 R/T/I", "later R/T/I", "Rel kept/gain", "T-Gen kept/gain", "I-Gen kept/gain", "cov T/I (g)", "cov later T/I (g)", "state"], rows))
    W("  deployed log, stored metrics (vendor pass of the build, padded batches; NOT the reported numbers): " + "; ".join(
        f"T={S[l]['_meta']['T']}: Rel/T-Gen/I-Gen exact " + "/".join(f1(S[l]["deployed"]["reg"]["stored"]["edits"][m]["exact"]) for m in ("rel", "tgen", "igen")) for l in L if "stored" in S[l].get("deployed", {}).get("reg", {})))
    W("  targets that are not oracles (cannot become exact under a value-releasing method): " + "; ".join(f"T={S[l]['_meta']['T']}: {S[l]['_factor']['n_non_oracle']} {S[l]['_factor']['non_oracle_edits']}" for l in L if "_factor" in S[l]))
    # ---------------------------------------------------------------- table 2: locality, far next to near
    W("\nTABLE 2  LOCALITY, FAR NEXT TO NEAR (R6).  FF = share of rows on which an edit is released.  heldout_disjoint (hd): far = probes that share no question string with an edit of the log,")
    W("  near = probes that share one.  literal = the conformal rule without guard caps; cap = edits at their guard cap (the bound is void for them).  BEN-T: prompt-only keys, rows related to")
    W("  the family excluded (counted in table 3); MME-T: out-of-benchmark workload; both gate level only, gates calibrated on teacher-forced REPAIR keys.  HN = hard negatives of the log's edits.")
    rows = []
    for log in L:
        T = S[log]["_meta"]["T"]
        for lab in labs(log):
            g = lambda nm: f1(v(log, lab, "gate", nm))
            b = lambda nm: f1(v(log, lab, "beh", nm))
            mme = g("MME-T.fired") if v(log, lab, "gate", "MME-T.fired") is not None else ("n.d." if lab in ("retr_sum", "lookup", "retr_conj", "retr_min") else "-")
            rows.append([T, name(lab), nd_gate(log, lab) or "-", ci(log, lab, "gate", "fire.hd"), g("fire.hd.literal"), g("n_capped"), g("fire.hd_far"), g("fire.hd_near"),
                         f"{g('hn.HN-I.fired')}/{g('hn.HN-Q.fired')}/{g('hn.HN-X.fired')}", g("BEN-T.far.fired"), mme, nd_beh(log, lab) or "-",
                         ci(log, lab, "beh", "probe.D1"), b("probe.D1.far"), b("probe.D1.near"), b("probe.tok_loc"),
                         f"{b('hn.HN-I.D1')}/{b('hn.HN-Q.D1')}/{b('hn.HN-X.D1')}", f"{b('hn.HN-I.own')}/{b('hn.HN-Q.own')}/{b('hn.HN-X.own')}", state(log, lab)])
    W(TT(["T", "method", "ng", "FF hd (g)", "literal", "cap", "FF far", "FF near", "FF HN I/Q/X (g)", "BEN-T", "MME-T", "nb", "D1 hd (b)", "D1 far", "D1 near", "tok-loc", "D1 HN I/Q/X (b)",
          "own rel. HN I/Q/X", "state"], rows))
    W("  n.d. = not defined: the retrieval family and lookup have no attributes for the MME rows.  Deployed log: stored probe flags on hd: " + "; ".join(
        f"T={S[l]['_meta']['T']}: D1 {f1(S[l]['deployed']['reg']['stored']['hd']['D1'])}" for l in L if "stored" in S[l].get("deployed", {}).get("reg", {})))
    W("  rows: " + "; ".join(f"T={S[l]['_meta']['T']}: hd {S[l]['_meta']['n_hd']['hd']} (near {S[l]['_meta']['n_hd']['hd_near']}, far {S[l]['_meta']['n_hd']['hd_far']}), invalid probe rows excluded "
                             f"{S[l]['_meta']['n_probe_invalid_excluded']}, degenerate pre-edit answers among valid probes {S[l]['_meta']['degenerate_y0']}, HN "
                             f"{ {k: x for k, x in S[l]['_meta']['describe']['hard_negatives'].items()} }" for l in L))
    # _x methods: near-scope read-outs twice
    W("\nTABLE 2x  NEAR-SCOPE FIRING OF THE _x METHODS, TWICE (g, 20 draws): on all hard negatives and on those whose foreign part occurs in no recombination row of the log (HN-X: count only).")
    rows = []
    for log in L:
        u = S[log]["_meta"].get("hn_unseen_foreign_n") or {}
        for lab in labs(log):
            if not lab.endswith("_x") or state(log, lab) == "not defined":
                continue
            g = lambda nm: f1(v(log, lab, "gate", nm))
            rows.append([S[log]["_meta"]["T"], lab, g("hn.HN-I.fired"), g("hn.HN-Q.fired"), g("hn.HN-X.fired"), g("hnu.HN-I.fired"), g("hnu.HN-Q.fired"),
                         f"{(u.get('HN-I') or {}).get('kept')}/{(u.get('HN-I') or {}).get('all')}", f"{(u.get('HN-Q') or {}).get('kept')}/{(u.get('HN-Q') or {}).get('all')}",
                         f"{(u.get('HN-X') or {}).get('kept')}/{(u.get('HN-X') or {}).get('all')}", f"{f1(v(log, lab, 'beh', 'hnu.HN-I.D1'))}/{f1(v(log, lab, 'beh', 'hnu.HN-Q.D1'))}"])
    W(TT(["T", "method", "all HN-I", "all HN-Q", "all HN-X", "unseen HN-I", "unseen HN-Q", "n unseen/all HN-I", "HN-Q", "HN-X (count)", "D1 unseen I/Q (b)"], rows))
    # ---------------------------------------------------------------- table 3: calibration records
    W("\nTABLE 3  CALIBRATION RECORDS (g, mean over 20 draws): eta, bound, thresholds, own prompts silenced by the literal rule (before the cap) and not released after it, capped edits,")
    W("  own prompts released by another edit (top-1 confusions), lineage width of firing hd probes, sequences whose decision differs between the teacher-forced and the decode position.")
    rows = []
    for log in L:
        for lab in labs(log):
            if v(log, lab, "gate", "fire.hd") is None:
                continue
            g = lambda nm, nd=2: f1(v(log, lab, "gate", nm), nd)
            rows.append([S[log]["_meta"]["T"], name(lab), g("eta", 3), g("eta_infinite", 2), g("bound", 4), g("thr.median", 3), g("thr.max", 3), g("own.silenced_before_cap", 1),
                         g("own.not_released_after_cap", 1), f"{g('n_capped', 1)} ({f1(v(log, lab, 'gate', 'n_capped', 'min'), 0)}-{f1(v(log, lab, 'gate', 'n_capped', 'max'), 0)})",
                         g("own.released_by_other", 1), f"{g('other.tgen', 1)}/{g('other.igen', 1)}", g("lineage.width_mean"), g("lineage.width_max", 1), g("fire.hd@prompt", 2),
                         g("prompt.n_rows_differ", 1), g("top1.n_differs", 1), f"{f1(v(log, lab, 'gate', 'BEN-T.far.n'), 0)}+{f1(v(log, lab, 'gate', 'BEN-T.related.n'), 0)}"])
    W(TT(["T", "method", "eta", "eta=inf", "bound", "thr med", "thr max", "silenced before", "after cap", "capped (min-max)", "own by other", "other T/I %", "lineage mean", "max",
          "FF hd @decode pos", "rows differ tf/dec", "top1 differs", "BEN-T far+related n"], rows))
    # ---------------------------------------------------------------- table 4: H3
    W("\nTABLE 4  READ-OUT H3 (R6): the common scale is raised until mean firing on the FIT hard negatives (even edits; HN-Q split by the edit whose question it asks) is <= 10 percent,")
    W("  guard caps kept; read on the odd edits and the test rows.  g: 20 draws (unreachable draws count coverage 0, as in the selection file); b: behavioural draws.  at cap = edits at their cap.")
    rows = []
    for log in L:
        for lab in labs(log):
            if v(log, lab, "gate", "h3.reachable") is None or lab not in (S[log]["_meta"]["selection"]["cos_H3"], S[log]["_meta"]["selection"]["cos_H3_x"], S[log]["_meta"]["selection"]["retr_H3"], "rescope", "rescope_x"):
                continue
            g = lambda nm: f1(v(log, lab, "gate", nm))
            hb = (S[log][lab]["reg"].get("h3_beh") or {})
            h = lambda nm: f1((hb.get(nm) or {}).get("mean"))
            nr = v(log, lab, "gate", "h3.reachable")
            rows.append([S[log]["_meta"]["T"], lab, f"{round(nr * nd_gate(log, lab))}/{nd_gate(log, lab)}", g("h3.cov"), f"{g('h3.tgen')}/{g('h3.igen')}",
                         f"{g('h3.test.HN-I.fired')}/{g('h3.test.HN-Q.fired')}/{g('h3.test.HN-X.fired')}", g("h3.hd.fired"), g("h3.n_at_cap"),
                         f"{g('h3u.test.HN-I.fired')}/{g('h3u.test.HN-Q.fired')} (n {f1(v(log, lab, 'gate', 'h3u.test.HN-I.n'), 0)}/{f1(v(log, lab, 'gate', 'h3u.test.HN-Q.n'), 0)}; HN-X {f1(v(log, lab, 'gate', 'h3u.test.HN-X.n'), 0)})" if lab.endswith("_x") else "",
                         (hb.get("reachable") or {}).get("n_draws", "-"), h("exact"), f"{h('exact_tgen')}/{h('exact_igen')}", h("cov"),
                         f"{h('test.HN-I.fired')}/{h('test.HN-Q.fired')}/{h('test.HN-X.fired')}", f"{h('test.HN-I.wrong_target')}/{h('test.HN-Q.wrong_target')}/{h('test.HN-X.wrong_target')}",
                         f"{h('hd_fired')}/{h('hd_D1')}", h("n_at_cap")])
    W(TT(["T", "method", "reach (g)", "H3 cov (g)", "cov T/I", "test FF I/Q/X", "FF hd", "at cap", "unseen-foreign test FF I/Q", "nb", "H3 exact (b)", "exact T/I", "cov (b)", "test FF I/Q/X (b)",
          "wrong target I/Q/X", "hd FF/D1 (b)", "at cap (b)"], rows))
    # ---------------------------------------------------------------- table 5: factor level
    W("\nTABLE 5  FACTOR LEVEL: relative error of the released values against the cached clean-model targets with rebasing / without (logged values); self-separation margins of the solver keys.")
    rows = []
    for log in L:
        fa = S[log].get("_factor")
        if fa:
            rb = fa["rebase"] or {}
            rows.append([S[log]["_meta"]["T"], f"{rb['rebased']['max']:.2e}" if rb else "-", f"{rb['rebased']['median']:.2e}" if rb else "-", f"{rb['not_rebased']['max']:.3f}" if rb else "-",
                         f"{rb['not_rebased']['median']:.3f}" if rb else "-", f"{fa['margin']['n_positive']}/{S[log]['_meta']['T']}", f"{fa['margin']['min']:.3f}", f"{fa['margin']['median']:.3f}",
                         fa["n_non_oracle"], fa["n_targets_found"]])
    W(TT(["T", "err rebased max", "median", "err logged max", "median", "margin > 0", "margin min", "median", "non-oracle targets", "targets found"], rows))
    # ---------------------------------------------------------------- table 6: D2 and generation
    W("\nTABLE 6  DRAW 0: D2 on heldout_disjoint as a rate of the pre-correct rows (48-token decode; [..] bootstrap over query clusters, ONE calibration draw); deployed: stored probe flags.")
    rows = []
    for log in L:
        sd = S[log].get("deployed", {}).get("reg", {}).get("stored")
        if sd:
            rows.append([S[log]["_meta"]["T"], "deployed (stored flags)", f1(sd["hd"]["D2"]), f"{sd['hd']['n_D2']}/{sd['hd']['n_pre_correct']}", "-"])
        for lab in labs(log):
            d = S[log][lab]["reg"].get("d2")
            if d:
                rows.append([S[log]["_meta"]["T"], name(lab), "-" if d["D2"] is None else f"{d['D2']:.1f} [{d['ci_lo']:.1f},{d['ci_hi']:.1f}]", f"{d['n_D2']}/{d['n_pre_correct']}", f1(d["fired"])])
    W(TT(["T", "method", "D2 %", "n D2 / pre-correct", "fired % (pre-correct rows)"], rows))
    W("  Free generation (decoder-v2 greedy) of rescope at draw 0: prompts of the first 100 edits (eq = generated answer equals the target; contains) and 200 seeded hd probes.")
    rows = []
    for log in L:
        gn = S[log].get("rescope", {}).get("reg", {}).get("gen")
        if gn:
            e, p = gn["edits_first100"], gn["probes200"]
            rows.append([S[log]["_meta"]["T"]] + [f"{f1(e[m]['eq'])}/{f1(e[m]['contains'])}" for m in ("rel", "tgen", "igen")] + [sum(e[m]["n_degenerate"] for m in e), p["n"], f1(p["fired"]),
                                                                                                                         f1(p.get("agree")), f1(p.get("long_eq_pre")), p["n_degenerate"]])
    W(TT(["T", "Rel eq/contains", "T-Gen eq/contains", "I-Gen eq/contains", "degenerate (edits)", "n probes", "probes fired %", "probes: agrees with pre-edit %", "48-token answer = pre-edit %", "degenerate (probes)"], rows))
    # ---------------------------------------------------------------- table 7: scale series, two read-outs
    W("\nTABLE 7  SCALE SERIES, TWO READ-OUTS (R8): reg = registered; qd = calibration draws and probe rows restricted to rows whose question string is not a question or rephrase of any edit")
    W("  of the longest stream of the backbone.  Rel / T-Gen / I-Gen exact, D1: behaviour; cov, FF: gate level (20 draws).  Cells: reg | qd.")
    rows = []
    for lab in [l for l in ORDER if l in SC.get(fam, {})]:
        for pt in SC[fam][lab]:
            r, q = pt.get("reg") or {}, pt.get("qd") or {}
            if "not_run" in r or (r.get("Rel") is None and r.get("false_firing_hd") is None):
                continue
            two = lambda k: f"{f1(r.get(k))} | {f1(q.get(k))}"
            rows.append([name(lab), pt["T"], two("Rel"), two("T-Gen"), two("I-Gen"), two("coverage_tgen"), two("coverage_igen"), two("false_firing_hd"), two("D1_hd"), two("near_scope_D1"),
                         f"{r.get('n_draws_behaviour') or '-'}|{q.get('n_draws_behaviour') or '-'}"])
    W(TT(["method", "T", "Rel", "T-Gen", "I-Gen", "cov T", "cov I", "FF hd", "D1 hd", "near-scope D1 (HN)", "nb"], rows))
    W("  qd rows: " + "; ".join(f"T={S[l]['_meta']['T']}: hd {S[l]['_meta']['n_hd']['hd_qd']} of {S[l]['_meta']['n_hd']['hd']}" for l in L))
    # ---------------------------------------------------------------- table 8: retention
    W("\nTABLE 8  RETENTION (E9 c): Rel / T-Gen / I-Gen exact of the FIRST 100 EDITS of the stream as T grows (b).")
    rows = []
    for lab in ("deployed", "where_logonly", "rescope"):
        for pt in RT.get(fam, {}).get(lab, []):
            rows.append([name(lab), pt["T"], f1(pt["rel"]), f1(pt["tgen"]), f1(pt["igen"]), pt.get("n_draws") or "-", (pt.get("not_run") or "")[:80]])
    W(TT(["method", "T", "Rel", "T-Gen", "I-Gen", "nb", "note"], rows))
    # ---------------------------------------------------------------- table 9: the behavioural draws, listed
    W("\nTABLE 9  THE BEHAVIOURAL CALIBRATION DRAWS, LISTED (b; draw 0 / 1 / 2) and the range of the same read-out over the 20 gate-level draws where it exists (min-max).")
    rows = []
    for log in L:
        for lab in labs(log):
            n = S[log][lab].get("reg") or {}
            if "beh" not in n or (n.get("n_draws_beh") or 0) < 1 or lab in ("deployed", "exact", "lookup", "cosine_default", "cosine_published", "where_logonly"):
                continue
            pd = lambda nm: "/".join(f1(x) for x in (n["beh"].get(nm) or {}).get("per_draw", {}).values())
            rng = lambda nm: f"{f1(v(log, lab, 'gate', nm, 'min'))}-{f1(v(log, lab, 'gate', nm, 'max'))}"
            rows.append([S[log]["_meta"]["T"], name(lab), pd("edit.rel.exact"), pd("edit.tgen.exact"), pd("edit.igen.exact"), pd("probe.fired"), rng("fire.hd"), pd("probe.D1"), pd("hn.all.D1"),
                         rng("cov.tgen"), rng("cov.igen"), "/".join(str(x) for x in (n.get("n_capped_beh") or {}).values())])
    W(TT(["T", "method", "Rel", "T-Gen", "I-Gen", "FF hd", "FF hd 20 draws", "D1 hd", "D1 HN", "cov T 20 draws", "cov I 20 draws", "capped"], rows))
    # ---------------------------------------------------------------- table 10: decode position
    W("\nTABLE 10  LAST PROMPT POSITION: teacher forcing vs decoding (LLaVA).  Sequences whose position differs (cache flag lp_shift): " + "; ".join(
        f"T={S[l]['_meta']['T']}: {S[l]['_meta']['lp_shift_rows']}" for l in L))
    W("  b: these sequences evaluated with the gate at the decode's position: mean number over the behavioural draws whose firing / outcome (exact reproduction) differs from the teacher-forced position.")
    rows = []
    for log in L:
        for lab in labs(log):
            pp = (S[log][lab].get("reg") or {}).get("prompt_position")
            if pp:
                rows.append([S[log]["_meta"]["T"], name(lab), f1(pp["n_rows"], 0), f1(pp["n_fired_differs"]), f1(pp["n_outcome_differs"]), f1(v(log, lab, "gate", "fire.hd")), f1(v(log, lab, "gate", "fire.hd@prompt")),
                             f"{f1(v(log, lab, 'gate', 'cov.tgen'))}/{f1(v(log, lab, 'gate', 'cov.tgen@prompt'))}", f"{f1(v(log, lab, 'gate', 'cov.igen'))}/{f1(v(log, lab, 'gate', 'cov.igen@prompt'))}"])
    W(TT(["T", "method", "n seq", "firing differs", "outcome differs", "FF hd tf (g)", "FF hd decode pos (g)", "cov T tf/dec (g)", "cov I tf/dec (g)"], rows))
    W("")

W("=" * 150)
W("NOT DEFINED / SKIPPED / FAILED / MISSING")
for log in LOGS:
    for lab in labs(log):
        for ro in ("reg", "qd"):
            n = S[log][lab].get(ro) or {}
            if n.get("not_defined"):
                W(f"  {log} {lab} [{ro}]: not defined: {str(n['not_defined'])[:230]}")
            if n.get("skipped_for_cost"):
                W(f"  {log} {lab}: {n['skipped_for_cost']}  timing {n.get('timing')}")
    for lab in ("where_logonly",):
        t = (S.get(log, {}).get(lab, {}).get("reg") or {}).get("timing")
        if t:
            W(f"  {log} {lab}: timing {t}")
W("  retrieval family and lookup on MME-T: not defined (no retrieval attributes for the MME rows); dual: not run on unseen logs (R19).")
W(f"  failed cells (recorded twice, skipped): {S['_run']['failed_cells'] or 'none'}")
by = {}
for m in MISSING:
    by[m["log"]] = by.get(m["log"], 0) + 1
W(f"  cells not computed yet, per log: {by or 'none'}")
W("\nJOB CHECKS (every behavioural job: hook_mismatch == 0, memo conflicts == 0): " + "; ".join(f"{l}: {len(j)} job(s) clean" if j else f"{l}: no finished job" for l, j in S["_run"]["jobs_assert_clean"].items()))
dc = S.get("_dev_check")
if dc:
    W(f"DEVELOPMENT LOG AGAINST THE SANITY RUN (e0_sanity): {dc['n_cells']} cells compared, {dc['n_cells_differ_above_1e-9']} differ by more than 1e-9, largest difference {dc['largest_abs_diff']}")
    for c in dc["differing"][:12]:
        W(f"    {c}")
fx.io.Results("E3", OUT).table("\n".join(out))
print("table.txt written,", len(out), "lines")
