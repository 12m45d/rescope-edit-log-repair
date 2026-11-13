#!/usr/bin/env python3
"""Statements of Secs. 3-5 about the vision-language experiments, asserted against the experiments' JSON.

    python3 scripts/check_vlm_statements.py        exit 1 if a statement does not hold (part of `make checks`)

Every sentence of Sec. 5.1 (additions of 2026-11-12), Sec. 5.2 (development logs, H1, what complaints add, release
against change, capacity), Sec. 5.3 (H2, H3, verdict), Sec. 5.4, Sec. 5.6, Sec. 5.7 (revocation, cost, ablations), of
the captions of Fig. fig:evidence (a, b), Table tab:main, Table tab:scale (a), Table tab:maint (b, c), and of the filled
placeholders of Secs. 2-4 that states a DIRECTION, a SIZE or a VERDICT is listed below with the words it backs. The
numbers themselves are macros (scripts/vlm_macros.py -> numbers_manifest.json) or cells of generated tables
(scripts/gen_vlm.py); this script checks the WORDS around them ("not met", "most of", "fewer", "every", "at the
bound", "missed", "trade-off", ...). Sources: the paper numbers of the experiments (gen_vlm.PN_FILES), the files of
gen_vlm.OTHER, the derived facts of gen_vlm.facts, and for some statements the experiments' own records, read here.
Pattern: scripts/check_e10_statements.py (experiment E10).
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gen_vlm as g                                                          # noqa: E402

ROOT = g.find_root()
D = g.Data(ROOT)
F = {k: v["value"] for k, v in g.facts(D).items()}
P = D.p
fails, n = [], 0
ALPHA = 5.0
Q, L = "Q", "L"
BB = ("Q", "L")
TEXT = "\n".join(re.split(r"(?<!\\)%", l)[0] for l in open(os.path.join(PAPER, "sec_experiments.tex")))


def ok(cond, sentence, detail=""):
    global n
    n += 1
    print(("  ok    " if cond else "  FAIL  ") + sentence + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(sentence)


def in_text(*phrases):
    """the sentence that a check backs is still in the text (white space collapsed); a changed sentence fails here."""
    flat = " ".join(TEXT.split())
    for ph in phrases:
        ok(" ".join(ph.split()) in flat, f"text holds: '{ph}'")


def e2(b, k):
    return P("e2", f"Etwo{b}{k}")


print("Sec. 5.1 (additions)")
ok(F["hn_degenerate_pct_L"] > 25 and F["hn_degenerate_pct_Q"] < 1, "a sizeable share of the near-scope negatives of LLaVA (and none of Qwen's) has a degenerate pre-edit answer",
   f"{F['hn_degenerate_pct_Q']:.1f} | {F['hn_degenerate_pct_L']:.1f}")
proto = open(os.path.join(ROOT, g.F, "PROTOCOL.md")).read()
ok("5 % above the largest coordinate on the other logged sequences" in " ".join(proto.split()), "log-only rule: 5 % above the largest coordinate on the other logged sequences (PROTOCOL.md, Section 0)")
# 'its two editor families share requests and solver keys': rows of methods that read only these agree in both families at
# equal T. NOT exactly: the rebased values of the two families differ slightly, and on the Qwen logs with 500 edits the
# request-keyed rows differ by 0.2 point of T-Gen (one edit); the gates on solver keys agree exactly. The paper therefore
# prints both families and does not say "equal by construction".
same, exact = [], []
for b in BB:
    byT = {lt[3]: lt for lt in g.UNSEEN[b] if lt[1].endswith("ae-l2_3")}
    for lt in g.UNSEEN[b]:
        if lt[1].endswith("memitw") and lt[3] in byT:
            for m in ("retr", "lookup", "white", "cosine"):
                a, c = g.e3row(D, b, byT[lt[3]], m), g.e3row(D, b, lt, m)
                d = max(abs(a[k] - c[k]) for k in ("Rel", "TGen", "IGen", "D1"))
                (same if m in ("retr", "lookup") else exact).append(d)
ok(same and max(same) < 0.5 and max(exact) < 1e-9, "'its two editor families share requests and solver keys': gates on solver keys give equal rows in both families at equal T; "
   "request-keyed rows agree within 0.5 point", f"request-keyed: max difference {max(same):.2f}; solver-key gates: {max(exact):.2g}")
ok("by construction" not in " ".join(TEXT.split()), "the text does not say that such rows are equal 'by construction' (they differ by up to 0.2 point on one log pair)")

print("RQ1 (Sec. 5.2): development logs")
for b in BB:
    ok(e2(b, "HoneBestName") == "RB_UNION", f"{b}: 'the linear repair with the lowest D1' after 10 complaints is the union rollback (the comparator of the printed difference)")
    ok(e2(b, "HoneBestDiffHi") < 0, f"{b}: the interval of the difference to that repair excludes zero", f"{e2(b, 'HoneBestDiff'):.1f} [{e2(b, 'HoneBestDiffLo'):.1f}, {e2(b, 'HoneBestDiffHi'):.1f}]")
    ok(e2(b, "LinRbUnionAllRel") <= 5 and e2(b, "LinRbUnionAllDone") < 10, f"{b}: rollback with all complaints 'deletes the log' (Rel at most 5 %) and only then reaches a low D1")
    ok(e2(b, "LinRbUnionTenRel") < e2(b, "DepRel") - 15, f"{b}: rollback after 10 complaints loses requests", f"{e2(b, 'LinRbUnionTenRel'):.0f} against {e2(b, 'DepRel'):.0f}")
    ok(min(e2(b, "LinCompBTenDone"), e2(b, "LinCompBAllposTenDone")) > 40, f"{b}: compensation (answer positions and all positions) leaves more than 40 % changed after 10 complaints")
ok(F["h1_holm_bound"] <= 0.002, "Holm-adjusted value of the H1 differences is an upper bound at the bootstrap's resolution", str(F["h1_holm_bound"]))

print("RQ1: H1 as registered (decision E6; final revision T30: no second reading of the rule is printed)")
h1r = D.o["h1r"]["development_logs"]
two = {b: h1r[g.DEV[b]]["two_sided_reading"]["met_on_this_log"] for b in BB}
one = {b: h1r[g.DEV[b]]["one_sided_reading"]["met_on_this_log"] for b in BB}
ok(two[Q] and not two[L], "'it holds on the Qwen log and fails on the LLaVA log' (two-sided reading)")
ok(e2(L, "HoneRelMinusDeployed") > 2 and e2(L, "HoneWhereDone") <= 10, "'because the repaired log answers MORE requests than the deployed log' (D1 is within the rule; Rel is above by more than 2 points)", f"+{e2(L, 'HoneRelMinusDeployed'):.0f}")
ok(not (two[Q] and two[L]), "registered rule: NOT MET (its development-log part fails)")
cells = {k: c for k, c in D.o["h1p"]["cells"].items() if c["n"] == 10}
t2 = sorted(c["T"] for c in cells.values() if c["verdict"]["D1_at_most_10"] and c["verdict"]["rel_within_2_points"])
ok(t2 == [10, 10, 20, 20] and F["h1_prefix_two_sided"] == 4 and F["h1_prefix_n"] == 6, "'its prefix part holds (4 of 6 prefixes: those of 10 and 20 edits)'", str(t2))
ok(D.o["h1p"]["at_least_one_mild_prefix_helps"] is True, "the experiment's own flag: at least one mild prefix helps (FINAL h1_prefix.json)")
# final revision (T30): the paper no longer prints a one-sided reading; it counts the logs on which Rel of the repair is more
# than 2 points below / above the deployed log ("one" and "two" of the 8 logs), with their names
ok(F["h1_n_rel_below"] == 1 and F["h1_rel_below_logs"] == [f"{g.DEV[Q]}-p50"],
   "'more than 2 points below the deployed log on one (the 50-edit prefix of Qwen)'", str(F["h1_rel_below_logs"]))
ok(F["h1_n_rel_above"] == 2 and sorted(F["h1_rel_above_logs"]) == sorted([g.DEV[L], f"{g.DEV[L]}-p50"]),
   "'and more than 2 points above on two (the LLaVA development log and its 50-edit prefix)'", str(F["h1_rel_above_logs"]))
ok(float(F["h1_n_rel_below"]) + float(F["h1_n_rel_above"]) + 1 + F["h1_prefix_two_sided"] == F["h1_n_logs"],
   "the 8 logs: the rule holds on the Qwen development log and on 4 prefixes; Rel is outside the rule's 2 points on the other three")
cq, cl = cells[f"{g.DEV[Q]}-p50|n10"]["verdict"], cells[f"{g.DEV[L]}-p50|n10"]["verdict"]
ok(not cq["rel_within_2_points"] and not cq["rel_not_more_than_2_below_deployed"] and cq["rel_signed_diff_to_deployed"] < -2, "the 50-edit prefix of Qwen: Rel more than 2 points BELOW the deployed log (78 against 82)", str(cq["rel_signed_diff_to_deployed"]))
ok(not cl["rel_within_2_points"] and cl["rel_not_more_than_2_below_deployed"] and cl["rel_signed_diff_to_deployed"] > 2, "the 50-edit prefix of LLaVA: Rel more than 2 points ABOVE the deployed log", str(cl["rel_signed_diff_to_deployed"]))
ok(F["h1_rel_delta_min"] < 0 < F["h1_rel_delta_max"] and F["h1_n_logs"] == 8, "'Rel moves by -4 to +6 points on the 8 logs': both signs occur", f"{F['h1_rel_delta_min']} .. {F['h1_rel_delta_max']}")
ok(F["h1_d1_max"] <= 10, "'D1 after the repair is at most ... on all' (and within the 10 % of the rule)", f"{F['h1_d1_max']:.2f}")
ok(F["h1_comp_any_d1_min"] > 5 * F["h1_d1_max"], "'(compensation: ... or more)': every compensation variant leaves several times more changed answers on every one of the 8 logs", f"{F['h1_comp_any_d1_min']:.1f}")
for b in BB:
    k, ga, lo = (P("e2f", f"EtwoFx{b}HoneRel{x}") for x in ("Kept", "Gained", "Lost"))
    ok(abs((k + ga) - e2(b, "HoneWhereRel")) < 1e-6 and abs((k + lo) - e2(b, "DepRel")) < 1e-6 and lo > 0,
       f"{b}: kept + gained = Rel of the repair, kept + lost = Rel of the deployed log, and some requests are lost ('not every deployed answer')", f"{k:.0f}/{ga:.0f}/{lo:.0f}")
in_text("The registered rule is \\emph{not met}: it holds on \\lgQAcName and on \\vAPfxTwo of the \\vAPfxN", "more than the rule's \\hOneRelPts points",
        "The repair retains every recorded term, not every deployed answer")
_h1 = " ".join(TEXT.split())
_h1 = _h1[_h1.index("\\emph{H1.}"):_h1.index("\\emph{What complaints add.}")]
ok("reading" not in _h1 and "Read as" not in _h1 and "one-sided" not in _h1, "the H1 passage prints the registered rule and its outcome only: no alternative reading (final revision T30)")
ok(not re.search(r"(keeps|retains) every (request|edit|deployed edit)", " ".join(TEXT.split())), "no sentence says that the repair keeps or retains every request or edit (decisions E6, D4)")

print("RQ1: what the log and complaints add (decision E7)")
for b in BB:
    dep, lo, al, cal = e2(b, "DepDone"), P("e1", f"Eone{b}RescopeWDOne"), e2(b, "WLogAllDone"), e2(b, "WCalDone")
    ok((dep - lo) / (dep - min(al, cal)) > 0.9, f"{b}: 'thresholds derived from the log alone account for most of the reduction of D1' (more than 90 % of the largest reduction)", f"{(dep - lo) / (dep - min(al, cal)):.3f}")
    ok(al <= lo and (lo - al) < 0.1 * (dep - lo), f"{b}: 'reported complaints add a smaller, targeted refinement'", f"{lo:.2f} -> {al:.2f}")
    ok(cal < al, f"{b}: 'benign queries reduce the residual further'", f"{cal:.2f} < {al:.2f}")
    ok(P("e2f", f"EtwoFx{b}WDepTenFireHeld") > 50 and e2(b, "WDepTenDone") < 20, f"{b}: from no thresholds, after 10 complaints an edit is still released on more than half of the held-out queries although D1 is low ('a measurement, not a certificate')",
       f"released {P('e2f', f'EtwoFx{b}WDepTenFireHeld'):.0f} %, D1 {e2(b, 'WDepTenDone'):.1f} %")
ok(F["least_cells"] == F["least_cells_all_orders"] == 72 and F["least_orders"] == 20 and F["least_refused"] == 0 and F["least_own_out"] == 0,
   "'their least repair is unique (bitwise equal thresholds under 20 arrival orders in all 72 cells)'; no complaint refused, no own request out of scope")
in_text("thresholds derived from the log alone account for most of the", "reported complaints add a smaller,", "benign queries reduce the residual")
flat = " ".join(TEXT.split()).lower()
ok("complaint-driven" not in flat and "complaints are unnecessary" not in flat, "neither 'complaint-driven' nor 'complaints are unnecessary' occurs (decision E7)")

print("RQ1: release is not change (decision E4)")
for b in BB:
    t = g.DEVTAG[b]
    rl, ch, cr, nr = (g.xr(D, b, t, "WhereLeastTenHd" + x) for x in ("Released", "Changed", "ChangedReleased", "ChangedNoRelease"))
    ok(nr == 0, f"{b}: 'and on none without a release' (measured: every sequence without a release had its forward pass)")
    ok(abs(100 * ch / rl - cr) < 0.5, f"{b}: changed = changed among released x released", f"{rl:.1f} / {ch:.1f} / {cr:.1f}")
    ok(rl > 15 and cr < 15, f"{b}: 'conservative in its effect on answers, not selective in release' (released on more than 15 % of held-out queries, fewer than 15 % of them change)")
    for k in "IQX":
        ok(g.xr(D, b, t, f"WhereLeastTenHn{k}Changed") < P("e1", f"Eone{b}DeployedHN{k}DOne") - 20, f"{b}, near-scope {k}: far fewer answers change than under the deployed log")
for b in BB:
    dT, dI = P("e1", f"Eone{b}RescopeWTGen") - P("e1", f"Eone{b}DeployedTGen"), P("e1", f"Eone{b}RescopeWIGen") - P("e1", f"Eone{b}DeployedIGen")
    ok(abs(dT) <= 5 and abs(dI) <= 5, f"{b}: 'paraphrases stay at the deployed level' (T-Gen and I-Gen of the repair within 5 points of the deployed log: no gain, no loss)", f"{dT:+.0f}, {dI:+.0f}")
ok(not re.search(r"scope (detection|accuracy) (is|was) (accurate|high)", flat), "low D1 is not used to claim accurate scope detection")

print("RQ1: capacity of linear restoration (decisions E8, E12; FIXES_FOR_FINAL_PASS F4)")
ok(str(P("e5", f"capacity.{g.DEV[Q]}.prediction_hit")) == "False" and str(P("e5", f"capacity.{g.DEV[L]}.prediction_hit")) == "True", "'registered prediction ..., missed on Qwen' (and met on LLaVA)")
ok(all(str(P("e5", f"capacity.{g.DEV[b]}.nd_holds")) == "True" for b in BB), "'the hypothesis of the theorem holds on both logs' (nd_holds)")
ok(F["cap_thr_silenced_max"] == 0, "technical report: the thresholds-only repair of the same complaints silences no edit")
for b in BB:
    ok(F[f"cap_stable_{b}"] == 1, f"{b}: the numerical rank is stable over two decades of the tolerance")
# 'the repairs we run show this restoration-retention trade-off': no linear / rollback repair reaches D1 <= 10 % while Rel stays within 10 points
bad = []
for (log, part, method, variant, nlab, metric), rs in D.rec.items():
    if part == "a" and metric == "probe.D1@mean" and method not in ("deployed", "preedit") and rs[0]["pool"] == "heldout_disjoint":
        rel = D.rec[(log, part, method, variant, nlab, "edit.rel.exact@mean")][0]["value"]
        b = Q if log == g.DEV[Q] else L
        if rs[0]["value"] <= 10 and rel >= e2(b, "DepRel") - 10:
            bad.append((log, method, nlab))
for c in cells.values():
    for m, r in c["linear"].items():
        if r["hd_D1"] <= 10 and r["rel"] >= c["deployed"]["rel"] - 10:
            bad.append((c["log"], m, 10))
ok(not bad, "'on the logs we evaluate, the repairs we run show this restoration-retention trade-off': no linear or rollback repair (any n; development and prefix logs) has D1 at most 10 % with Rel within 10 points of the deployed log", str(bad[:3]))
in_text("the rollback and linear repairs we run show this restoration--retention trade-off", "scoped replay is not subject to it because",
        "We evaluate the rank formula of Thm.~\\ref{thm:lincap} on captured key", "registered prediction, rounded down to the grid")
ok("no independent confirmation" not in " ".join(TEXT.split()), "the words '(no independent confirmation)' are gone (final revision T31): the text says what was computed")
ok(all(F[f"cap_formula_holds_{b}"] == 1 for b in BB), "'and check it with the minimum-norm restoring weight on a grid': at every grid point the number of edits that this weight preserves equals the rank formula (both logs)")
ok(F["cap_hit_all_Q"] == 0 and F["cap_hit_all_L"] == 3, "like with like: the registered prediction rounded down to the grid against the grid measurement: 'missed on Qwen' (every order), met on LLaVA (every order)",
   f"Qwen {F['cap_pred_grid_list_Q']} against {F['cap_meas_list_Q']}; LLaVA {F['cap_pred_grid_list_L']} against {F['cap_meas_list_L']}")
ok(F["cap_grid_step_Q"] == F["cap_grid_step_L"] == 5, "'on a grid of step 5' (the step of the grid between the smallest predicted and the largest measured value)")
ok(F["cap_w0_miss_Q"] == 2 and F["cap_w0_hit_L"] == 3, "technical report (T39 f): the second registered prediction (from how many complaints on only the pre-edit weight restores them) is missed on Qwen in two of the three orders and met on LLaVA",
   f"Qwen {F['cap_w0_pred_grid_list_Q']} against {F['cap_fullrank_list_Q']}; LLaVA {F['cap_w0_pred_grid_list_L']} against {F['cap_fullrank_list_L']}")
ok("cannot restore" not in flat, "'cannot restore' does not occur (decision E8)")

print("RQ2 (Sec. 5.3): registered verdict, H2, H3")
ok(P("e1", "EoneDecisionPairOne") == P("e1", "EoneDecisionPairTwoX") == P("e1", "EoneDecisionRetr") == "trade-off",
   "'trade-off for both pairs of H3 and for the retrieval reference' (decision.json through the paper numbers)")
dec = D.o["dec"]
ok(all(v["verdict"] == "trade-off" for v in dec["pairs"].values()) and len(dec["pairs"]) >= 2, "decision.json: the verdict of every pair is 'trade-off' (no 'dominates')", str({k: v["verdict"] for k, v in dec["pairs"].items()}))
for b in BB:
    ok(max(P("e1", f"Eone{b}HthreePairOneDiff"), P("e1", f"Eone{b}HthreePairTwoXDiff")) < 10, f"{b}: no H3 lead of 10 points for either pair (the rule's threshold)")
    ok(P("e1", f"Eone{b}HthreeRetrDiff") < 0 and P("e1", f"Eone{b}HtwoRetrDiff") < -20, f"{b}: retrieval is ahead in H2 and H3")
ok(P("e1", "EoneQHtwoBestCosineHTwoHi") < 0 and P("e1", "EoneQHtwoBestCosineHTwoPHolm") < 0.05, "Qwen: ReScope is BELOW the selected gate in H2 (interval and Holm value)", f"{P('e1', 'EoneQHtwoBestCosineHTwoDiff'):.1f}, p = {P('e1', 'EoneQHtwoBestCosineHTwoPHolm'):.3f}")
ok(P("e1", "EoneLHtwoBestCosineHTwoLo") < 0 < P("e1", "EoneLHtwoBestCosineHTwoHi"), "LLaVA: no difference to the selected gate in H2 (interval contains zero)")
ok(F["h2_other_min"] > 20, "'against the registered cosine gate, ball, router, lookup and answer match +25 to +49': all positive", f"{F['h2_other_min']:.1f} .. {F['h2_other_max']:.1f}")
ok(P("e1f", "EoneFxHtwoNComparisons") == 18, "H2 has 18 comparisons")
for b in BB:
    rows = {r: (P("e1", f"Eone{b}{r}TGen"), P("e1", f"Eone{b}{r}IGen")) for r in ("Rescope", "RescopeX", "Cosine", "BestCos", "BestCosHtwo", "Lookup", "Dual", "Deployed", "RescopeW")}
    rt = (P("e1", f"Eone{b}RetrTGen"), P("e1", f"Eone{b}RetrIGen"))
    ok(all(rt[0] > v[0] and rt[1] > v[1] for v in rows.values()), f"{b}: 'retrieval generalises best' (T-Gen and I-Gen above every other row of the table)")
    nr = lambda r: sum(P("e1f", f"EoneFx{b}{r}HN{k}FireMean") for k in "IQX") / 3                 # noqa: E731
    ok(nr("Retr") > nr("Rescope"), f"{b}: retrieval 'is released on more near-scope queries' than ReScope (column Near: mean of the kinds)", f"{nr('Retr'):.0f} against {nr('Rescope'):.0f}")
    # figure revision of 2026-11-12 (DECISIONS_FIG_REVISION.txt, G5; FINAL_CHECK_2 S5): the advantage of a baseline of our own class
    # stays in the body. Cells of the report's Table tab:app:main, column Near (mean of the kinds, per behavioural draw).
    nd = lambda r: [sum(P("e1f", f"EoneFx{b}{r}HN{k}FireD{d}") for k in "IQX") / 3 for d in ("Zero", "One", "Two")]   # noqa: E731
    if b == Q:
        ok(all(c < r_ for c, r_ in zip(nd("Cosine"), nd("Rescope"))), "Qwen: 'the cosine on solver keys, of our class, [is released] on fewer [near-scope queries] on Qwen' than ReScope, in every draw; the row is calibrated like ReScope (block 'log, B' of the table) "
           "(report, Table tab:app:main, row 'cosine on solver keys')", f"{'/'.join(f'{v:.0f}' for v in nd('Cosine'))} against {'/'.join(f'{v:.0f}' for v in nd('Rescope'))}")
    else:
        ok(not all(c < r_ for c, r_ in zip(nd("Cosine"), nd("Rescope"))), "LLaVA: not so in every draw, hence the sentence names Qwen only",
           f"{'/'.join(f'{v:.0f}' for v in nd('Cosine'))} against {'/'.join(f'{v:.0f}' for v in nd('Rescope'))}")
    t = g.DEVTAG[b]
    ok(g.xr(D, b, t, "RescopeHdChangedReleased") == 100.0 and g.xr(D, b, t, "RescopeHdChangedNoRelease") == 0, f"{b}: 'a release of ReScope changes the answer on every released held-out query'")
ok(F["sys_changed_released_min"] > 90 and "of the released (means over draws)" in " ".join(TEXT.split()),
   "'on near-scope negatives ... on at least 95 % of the released (means over draws)': the minimum over kinds and backbones of the MEAN over the draws; single draws fall lower "
   "(figure revision G15; recomputation x_release/verify, statement a), and the text says 'means over draws'", f"{F['sys_changed_released_min']:.1f}")
ok(P("e1", "EoneQBestCosGateFire") > ALPHA and P("e1", "EoneQBestCosHtwoGateFire") > ALPHA and P("e1", "EoneQBestCosCapped") == 0 == P("e1", "EoneQBestCosHtwoCapped"),
   "'release on Held-out exceeds alpha for the whitened gates on Qwen (no capped edit)'")
ok(P("e1", "EoneLRetrGateFire") > ALPHA, "'... and for retrieval on LLaVA'", f"{P('e1', 'EoneLRetrGateFire'):.2f}")
ok(all(P("e1", f"Eone{b}RescopeGateFire") < ALPHA for b in BB), "(not said, checked) ReScope's own mean release over the 20 draws is below alpha on both logs")
for b in BB:
    ok(F[f"front_{b}_ncoord_50_ff"] < 5 < F[f"front_{b}_cos_50_ff"], f"{b}: at half of the rephrasings covered the log's coordinate fires on few held-out queries, the raw cosine on many",
       f"{F[f'front_{b}_ncoord_50_ff']:.1f} against {F[f'front_{b}_cos_50_ff']:.1f}")
mcd = [P("e1f", f"EoneFx{b}DeployedMcDOneD{d}P") for b in BB for d in ("Zero", "One", "Two")]
ok(max(mcd) < 10.0 ** F["mc_dep_exp"], "'by exact McNemar tests D1 of ReScope differs from the deployed log (p < 10^-57 in every draw)'", f"largest p {max(mcd):.1e}")
for b in BB:
    h2 = [P("e1f", f"EoneFx{b}BestCosHtwoMcDOneD{d}PHolm") for d in ("Zero", "One", "Two")]
    h3 = [P("e1f", f"EoneFx{b}BestCosMcDOneD{d}PHolm") for d in ("Zero", "One", "Two")]
    ok(sum(1 for x in h2 if x < 0.05) == 1 and min(h3) >= 0.05, f"{b}: '... and from the gate selected for H2 in one draw of 3' (Holm over all tests; the gate selected for H3: in none). "
       "The coordinator's reading said 'no difference': in one draw ReScope changes fewer answers than that gate", f"H2 gate {[round(x, 3) for x in h2]}, H3 gate {[round(x, 3) for x in h3]}")
    d1 = [(P("e1f", f"EoneFx{b}BestCosHtwoMcDOneD{d}B"), P("e1f", f"EoneFx{b}BestCosHtwoMcDOneD{d}C")) for d in ("Zero", "One", "Two")]
    sig = [bc for bc, x in zip(d1, h2) if x < 0.05][0]
    ok(sig[0] < sig[1], f"{b}: in that draw fewer held-out answers change under ReScope than under the gate (not said in the text)", str(sig))
best = D.o["best"]
ok(all(v["cosine"]["H2"] == "white_0.1_shared" and v["cosine"]["H3"] == "white_1_shared" and v["retr"]["H2"] == "retr_sum" for v in best["selected"].values()),
   "the selected reference gates are whitened coordinates under a shared scale; the retrieval rule is retr_sum (rows of Table tab:main)")
ok(F["sel_cos_family_n"] == 18 and F["sel_retr_family_n"] == 3, "'of 18 variants' of the cosine family and 'of 3' retrieval rules")
in_text("selected at gate level on the evaluation rows themselves, in the baseline's favour", "We do not claim that the coordinate detector is better",
        "the cosine on solver keys, of our class, on fewer (\\bbQs)")

print("RQ3 (Sec. 5.4): unseen logs")
for b in BB:
    ok(F[f"ret_{b}a_w_min"] == F[f"ret_{b}a_w_max"] and F[f"ret_{b}a_s_min"] == F[f"ret_{b}a_s_max"], f"{b}: 'the first 100 requests stay answered at ... for every T' (ReScope-W and ReScope: the same value at every T)")
    ok(F[f"ret_{b}a_dep_min"] < F[f"ret_{b}a_dep_max"] - 20, f"{b}: under the deployed log they are lost as T grows", f"{F[f'ret_{b}a_dep_max']:.0f} -> {F[f'ret_{b}a_dep_min']:.0f}")
    w2 = [g.e3row(D, b, lt, "where") for lt in g.UNSEEN[b] if lt[1].endswith("memitw")]
    d2 = [g.e3row(D, b, lt, "deployed") for lt in g.UNSEEN[b] if lt[1].endswith("memitw")]
    ok(min(r["D1"] for r in w2) > F[f"e3_{b}a_w_d1_max"] or min(r["D1"] for r in w2) > 4, f"{b}: 'on the second family ReScope-W is weak' (D1 above its first-family values)", f"{[round(r['D1'], 1) for r in w2]}")
    s2 = [g.e3row(D, b, lt, "rescope") for lt in g.UNSEEN[b] if lt[1].endswith("memitw")]
    ok(min(r["Rel"] for r in s2) > 95, f"{b}: 'ReScope answers the requests there'")
    ok(all(a["TGen"] > c["TGen"] and a["IGen"] > c["IGen"] for a, c in zip(s2, s2[1:])), f"{b}: 'paraphrases fall with T' (T-Gen and I-Gen, second family)")
    ok(F[f"e3_{b}m_errlog_max"] > 1.5 * F[f"e3_{b}a_errlog_max"] and F["e3_errreb_max"] < 0.05, f"{b}: 'a logged value alone is far from its target' on the second family; rebased values are within a few percent everywhere")
ok(all(w["Rel"] < d["Rel"] for w, d in zip([g.e3row(D, Q, lt, "where") for lt in g.UNSEEN[Q] if lt[1].endswith("memitw")], [g.e3row(D, Q, lt, "deployed") for lt in g.UNSEEN[Q] if lt[1].endswith("memitw")])),
   "'on Qwen [ReScope-W] answers fewer requests than the deployed log' (every log of the second family)")
ok(P("e3qf", "EthreeFxQMeMMMarginSmallest") < 0 and P("e3lf", "EthreeFxLMeThousandMarginSmallest") < 0, "'the smallest self-separation margin turns negative' at the largest T of both backbones")
ok(P("e3lf", "EthreeFxLMeThousandMarginNonPositive") == 2, "LLaVA, 1000 edits: two edits with a non-positive margin (the experiment's summary sentence '0.02' is not used)")
for b in BB:
    for lt in g.UNSEEN[b]:
        d, lo, hi = (g.fxp(D, b, lt, "HtwoCosAll" + x) for x in ("Diff", "Lo", "Hi"))
        if lt[1].endswith("ae-l2_3"):
            ok(lo < 0 < hi, f"{lt[1]}: 'no difference to the selected gate on the first family' (interval contains zero)", f"{d:.1f} [{lo:.1f}, {hi:.1f}]")
        else:
            ok(hi < 0, f"{lt[1]}: 'ReScope worse on every log of the second' family (interval below zero)", f"{d:.1f} [{lo:.1f}, {hi:.1f}]")
        ok(max(g.fxp(D, b, lt, "HthreePairOneDiff"), g.fxp(D, b, lt, "HthreePairTwoDiff")) < 10 and g.fxp(D, b, lt, "HthreeRetrDiff") < -10, f"{lt[1]}: 'in H3 nothing dominates, and retrieval leads'")
ok(P("xr", "XrelHolmHtwoRejected") == 18 and P("xr", "XrelHolmHtwoFamily") == 22, "technical report: 18 of 22 H2 comparisons rejected after Holm")
ok(max(F["e3_Qm_s_cap_max"], F["e3_Lm_s_cap_max"]) > 50, "'on average up to 60 | 149 edits sit at their guard cap' (means over the 20 draws; second family)", f"{F['e3_Qm_s_cap_max']:.1f} | {F['e3_Lm_s_cap_max']:.1f}")
ok(F["e3_near_rel_min"] > 2 and F["e3_draws_above_max"] <= 10, "'released on 3 to ... % of held-out probes that ask an edit's question'; '0 to 9 of 20 draws per log exceed alpha'")
# figure revision of 2026-11-12 (G6; FINAL_CHECK_2 S6; recomputation, statement b): the subsection is about the unseen logs, and the
# largest value is the development log of Qwen, which the text now names apart
ok(F["e3_near_rel_max"] == F["e3_Q_near_rel_dev"] > F["e3_near_rel_unseen_max"] + 5 and F["e3_near_rel_unseen_min"] <= F["e3_L_near_rel_dev"] <= F["e3_near_rel_unseen_max"],
   "'3 to 13 (Q-A100-r3: 23) %': the range is that of the 11 unseen logs, the value in parentheses is the development log of Qwen (the largest of all 13), and the development log of LLaVA lies inside the range",
   f"unseen {F['e3_near_rel_unseen_min']:.1f} .. {F['e3_near_rel_unseen_max']:.1f}; dev Q {F['e3_Q_near_rel_dev']:.1f}, dev L {F['e3_L_near_rel_dev']:.1f}")
in_text("\\sys is also released on \\vCNearRelUnseenMin to \\vCNearRelUnseenMax (\\lgQAcName: \\vCNearRelMax)\\,\\% of held-out probes")
in_text("The registered read-outs are replications: H2 is not confirmed", "nothing was tuned on them")
tab = open(os.path.join(PAPER, "tab_vlm_scale.tex")).read()
ok(tab.count(r"\\") == 6 and "memitw" not in tab, "Table tab:scale (a): one line per log family (four lines; page decision P5)")

print("RQ5 (Sec. 5.6): calibration under workload shift (decision E5)")
V = lambda b, rule, pair, leaf: P("e5", f"validity.{g.DEV[b]}.{rule}.{pair}.{leaf}")              # noqa: E731
same = ("repair_to_repair_rest", "ben_s_to_ben_s_rest", "mme_c_to_mme_c_rest")
ok(all(V(b, "two_block", p, "level") in ("met", "above, within 2 SE") and abs(V(b, "two_block", p, "ff_mean") - F["val_bound"]) < 0.25 for b in BB for p in same),
   "'within one pool the mean rate is ... at the bound' (within 0.25 point of 4.95; the experiment's own label: met, or above within 2 SE)")
ok(abs(F["val_same_above_min"] - F["val_beta_above"]) < 6 and abs(F["val_same_above_max"] - F["val_beta_above"]) < 6, "'39 to 47 % of the single draws exceed alpha, as the Beta law of the rule says (43.6 %)'")
ok(F["val_pac_exch_above_max"] < 10, "'the calibration-conditional rank lowers this share to at most 7.6 %' (below delta = 10 %)")
ok(all(P("e6", f"Esix{b}PacCovT") < P("e6", f"Esix{b}DefaultCovT") - 5 for b in BB), "'... and costs coverage'")
ok(V(Q, "two_block", "repair_to_mme_t", "ff_mean") > ALPHA and V(Q, "two_block", "repair_to_mme_t", "share_of_draws_above_alpha") > 25 and V(Q, "two_block", "ben_s_to_mme_t", "ff_mean") > ALPHA,
   "'to the out-of-benchmark workload the level is missed on Qwen' (mean above the level from both calibration sources; a third of the draws above)")
ok(V(L, "two_block", "repair_to_mme_t", "ff_mean") < ALPHA, "LLaVA: below the level there")
ok(P("e3l", "EthreeLAeTwoHundredRescopeMmeT") > ALPHA and P("e3l", "EthreeLAeFiveHundredRescopeMmeT") > ALPHA,
   "LLaVA, unseen logs of the first family: release on the out-of-benchmark workload above the level (14.9 and 6.7 %): printed with the failure on Qwen")
ok(F["h3_unreachable_max"] == 0, "technical report: the near-scope constraint of H3 is reachable in every draw")
ok(F["val_rev_min"] > 85, "'calibrated on that workload the rule fails completely' (release on 90 to 98 % of benchmark and benign queries)")
ok(all(P("e2f", f"EtwoFx{b}WCalFreshTwoHundredContainsBound") in (1, True) for b in BB), "post hoc read-out of ReScope-W with two-block thresholds: the interval contains the bound on both logs")
ok(P("e2f", "EtwoFxQWCalHeldTwoHundredMean") > F["val_bound"] and P("e2p", "PfxQFiftyWTwoBlockGateLiteralFired") > F["val_bound"], "on Held-out (a second pool) the Qwen log and its 50-edit prefix lie ABOVE the bound: printed, not explained away")
ok(V(Q, "two_block", "repair_to_heldout", "image_rows.ff_mean") > 5 > V(Q, "two_block", "repair_to_heldout", "text_rows.ff_mean"), "(technical report) Qwen: the releases are on image queries")
in_text("a failure under workload shift and not a fluctuation", "post hoc read-out", "The bound is claimed for exchangeable queries only")

print("RQ6 (Sec. 5.7): revocation, lineage, cost (decisions E10, E13, D10)")
for b in BB:
    ok(P("e9", f"Enine{b}FrameViolations") == 0 and F[f"rev_{b}_scoped_logits_max"] == 0 and F[f"rev_{b}_scoped_flags_max"] == 0,
       f"{b}: scoped log: 0 violations of the frame rule; no tested sequence without a release of a revoked edit changes (logits or answer flags), any m")
    ok(F[f"rev_{b}_linear_all_logits"] == 1, f"{b}: 'subtracting ... changes the logits of every other tested sequence' (every m)")
    ok(0 < P("e9", f"Enine{b}LinearMOneOutsideFlags") < P("e9", f"Enine{b}LinearMFiftyOutsideFlags"), f"{b}: answer flags change, more of them for larger m")
    ok(P("e9", f"Enine{b}CertSilent") == P("e9", f"Enine{b}CertIdentical"), f"{b}: (technical report) every sequence without a release has the pre-edit logits bit for bit")
ok("global" not in flat.split("revocation and cost")[-1][:1500], "'global' is not used for answers in the revocation text (decision E10)")
T100 = 100000 * P("e8", "storage.qwen25vl.numbers_per_edit") / (P("e8", "storage.qwen25vl.T_at_model_size") * P("e8", "storage.qwen25vl.numbers_per_edit")) * 100
ok(abs(T100 - P("e8", "synthetic.T100000.storage_share_of_model_pct")) < 0.05, "storage arithmetic (decision E13): numbers per edit x 10^5 / parameter count of Qwen = the printed share", f"{T100:.1f} %")
ok(abs(P("e8", "TB-Q100-ae-l2_3.storage_numbers") * 8 / 2 ** 20 - P("e8", "TB-Q100-ae-l2_3.storage_mb")) < 0.01, "MB = stored numbers x 8 bytes (double precision)")
ok(P("e8", "TB-Q100-ae-l2_3.card_utilisation_pct_median") > 80, "the latency is given with the card load at which it was measured (92 %)")
in_text("other gates are not timed", "is context, not normalised")
# final revision (T3; FINAL_CHECK M3): the serving cost of ReScope-W is printed wherever the cost of ReScope is
for lg_ in g.DEV.values():
    g1, o1 = P("e8", f"{lg_}.gate_ms_per_forward"), P("e8", f"{lg_}.overhead_end_to_end_pct")
    g2, o2 = P("e8", f"{lg_}.gate_ms_per_forward_all_positions"), P("e8", f"{lg_}.overhead_end_to_end_pct_all_positions")
    b_ = Q if lg_ == g.DEV[Q] else L
    g3, o3 = F[f"cost_{b_}_wlog_gate_ms"], F[f"cost_{b_}_wlog_over_pct"]
    # figure revision of 2026-11-12 (G17; FINAL_CHECK_2 N8): the timing is from a shared, loaded card and the experiment has a second
    # pass. The check is restricted to what BOTH passes support: the gate of ReScope is below both forms of ReScope-W (gate ms and end
    # to end), and the log-only form is below the all-positions form in gate ms. 'Log-only below calibrated' END TO END held in
    # the primary pass only (Qwen, pass 2: 15.6 against 14.5 %) and is no longer asserted; no sentence states it.
    h1, p1 = F[f"cost_{b_}_s_gate_ms_p2"], F[f"cost_{b_}_s_over_pct_p2"]
    h2, p2 = F[f"cost_{b_}_wall_gate_ms_p2"], F[f"cost_{b_}_wall_over_pct_p2"]
    h3, p3 = F[f"cost_{b_}_wlog_gate_ms_p2"], F[f"cost_{b_}_wlog_over_pct_p2"]
    ok(g1 < g3 < g2 and o1 < min(o2, o3) and h1 < h3 < h2 and p1 < min(p2, p3),
       f"{b_}: in both latency passes the gate of ReScope-W (every position) costs more than that of ReScope (gate ms and end to end), and its log-only form less gate time than its all-positions form",
       f"pass 1: {g1:.2f} < {g3:.2f} < {g2:.2f} ms; {o1:.1f} < {o3:.1f}, {o2:.1f} %; pass 2: {h1:.2f} < {h3:.2f} < {h2:.2f} ms; {p1:.1f} < {p3:.1f}, {p2:.1f} %")
ok(F["cost_Q_s_gate_ms_p2"] > 1.5 * P("e8", f"{g.DEV[Q]}.gate_ms_per_forward"), "Sec. 5.7 prints both passes because they differ (Qwen, gate of ReScope: the second pass is more than half above the primary one)",
   f"{P('e8', g.DEV[Q] + '.gate_ms_per_forward'):.2f} and {F['cost_Q_s_gate_ms_p2']:.2f} ms")
in_text("(\\cstQOverPct{}\\,\\% end to end; second pass: \\cstQGateMsTwo{}\\,ms, \\cstQOverPctTwo{}\\,\\%)",
        "(\\cstQWOverPct{}\\,\\%; second pass: \\cstQWGateMsTwo{}\\,ms, \\cstQWOverPctTwo{}\\,\\%; log-only thresholds: \\cstQWLogGateMs{}\\,ms, \\cstQWLogOverPct{}\\,\\%)", "On a shared card",
        "spread: \\trp{app:exp-repair}{app:exp}")
for _fn, _must in (("sec_intro.tex", ("gates: \\cstQGateMs{}\\,ms for \\sys", "\\cstQWGateMs{}\\,ms for \\sysW; repair: \\cstQRepMs{}\\,ms; Sec.~\\ref{sec:exp:maint})")),
                   ("sec_method.tex", ("\\sys: gate \\cstQGateMs", "\\sysW, at every position: \\cstQWGateMs", "Measured (primary pass; Sec.~\\ref{sec:exp:maint})")),
                   ("sec_conclusion.tex", ("of \\sys: \\cstQGateMs", "of \\sysW, at every position: \\cstQWGateMs", "primary pass, Sec.~\\ref{sec:exp:maint}")),
                   ("sec_experiments.tex", ("gate of \\sys takes \\cstQGateMs", "that of \\sysW, at every position, \\cstQWGateMs", "gate (of \\sys)"))):
    _t = " ".join("\n".join(re.split(r"(?<!\\)%", l)[0] for l in open(os.path.join(PAPER, _fn))).split())
    ok(all(" ".join(x.split()) in _t for x in _must), f"{_fn}: every printed gate cost names its method, the cost of ReScope-W stands next to that of ReScope (figure revision G8: also in Sec. 1), "
       "and Secs. 4, 7 say 'primary pass' (G17); Sec. 1 points to Sec. 5.7 instead (last pass L8)")

print("Ablations (Sec. 5.7)")
for b in BB:
    ok(P("e6", f"Esix{b}CalRunMaxFire") > 5 * P("e6", f"Esix{b}DefaultFire"), f"{b}: running maxima release far more than the two-block rule")
    ok(abs(P("e6", f"Esix{b}RespLinTGen") - P("e1", f"Eone{b}DeployedTGen")) <= 5 and abs(P("e6", f"Esix{b}RespLinIGen") - P("e1", f"Eone{b}DeployedIGen")) <= 5,
       f"{b}: 'the linear response falls back to the deployed level of paraphrases' (within 5 points)")
ok(F["abl_logged_maxdiff"] <= 1.0 + 1e-9, "'logged instead of rebased values move the edit scores by at most 1 point here'")

print("Figure fig:evidence (a), (b) and Table tab:main: what is drawn is what the captions say")
fig = json.load(open(os.path.join(PAPER, "figs", "evidence_ab.json")))
ok([p["log"] for p in fig["panels"]] == [g.DEV[Q], g.DEV[L]], "the figure shows the two development logs")
for p in fig["panels"]:
    b = Q if p["log"] == g.DEV[Q] else L
    ser = {s["key"]: s["points"] for s in p["series"]}
    ok(set(ser) == {"comp_b", "comp_b_allpos", "reedit_b", "rb_union", "rb_tot", "rescope_where", "rescope_where_start"}, f"{b}: all linear and rollback repairs of E2 part a and both starts of ReScope-W are drawn (decision E3)")
    ok([q["n"] for q in ser["comp_b_allpos"]] == ["1", "2", "5", "10"], f"{b}: compensation at all positions exists up to n = 10")
    w10 = [q for q in ser["rescope_where"] if q["n"] == "10"][0]
    ok(abs(w10["d1"] - e2(b, "HoneWhereDone")) < 1e-9 and abs(w10["rel"] - e2(b, "HoneWhereRel")) < 1e-9, f"{b}: the H1 point of the figure equals the H1 numbers of the text")
    ok(abs(p["reference"]["deployed"]["d1"] - e2(b, "DepDone")) < 1e-9, f"{b}: the deployed log of the figure equals the text")
mt = open(os.path.join(PAPER, "tab_vlm_main.tex")).read()
ok(mt.count(r"\\") == 10, "Table tab:main has 8 rows (five rows are in the technical report: W4, page budget, cut (2))")
mtr = open(os.path.join(PAPER, "tab_vlm_tr_main.tex")).read()
ok(mtr.count(r"\\") == 15 and "two-block thresholds" in mtr and "lookup on the requests" in mtr and "two-block thresholds" not in mt and "lookup on the requests" not in mt,
   "the report's table has all 13 rows, among them the two that left Table tab:main (ReScope-W with two-block thresholds; lookup on the requests)")

print("Secs. 2-4: filled placeholders")
ex = json.load(open(os.path.join(ROOT, g.F, "e2_curves/out/running_example.json")))
ok(ex["rollback"]["edits_broken"] == 1, "running example: one kept edit misses its target after the rollback (singular in the text)")
ok(ex["rescope"]["own_requests_firing"] == ex["T"], "running example: all own requests still fire after the least repair")
flag = json.load(open(os.path.join(ROOT, g.F, "e2_curves/out/fixup/running_example_flag.json")))
prob = open(os.path.join(PAPER, "sec_problem.tex")).read()
ok(flag["two_different_edits"] is True and "exRephrase" not in "\n".join(l for l in prob.split("\n") if not l.startswith("%")), "running example: the why-not rephrase (another edit) is not printed, so the example cannot present two edits as one")
ok(F["cap_meas_Q"] == "25--35" and F["cap_meas_L"] == "35", "Sec. 1, contribution (2): measured capacity 25-35 (Qwen) and 35 (LLaVA), 'on a grid of step 5' (Sec. 3 no longer prints the numbers: final revision T22)")

# ---------------------------------------------------------------------------------------------------------------------
# W4, 2026-11-12: the abstract (main.tex), Sec. 1 (sec_intro.tex) and Sec. 7 (sec_conclusion.tex). Their numbers are the
# macros of Sec. 5 (and \vALinMin, \vALinMax); the checks above back the statements that these texts repeat (H1 "not met
# as worded", verdict "trade-off", "retrieval ... generalises best", the second editor family, the collapsed log, the
# calibration bound under workload shift, revocation). Below: what Secs. 1 and 7 word DIFFERENTLY or state in addition,
# the wording rules of 03_analysis/READING_FOR_W4.txt (part C), and the plain-text abstract of the registration.
print("Secs. 1, 7 and the abstract (W4)")


def _body(fn):
    return " ".join("\n".join(re.split(r"(?<!\\)%", l)[0] for l in open(os.path.join(PAPER, fn))).split())


INTRO, CONCL = _body("sec_intro.tex"), _body("sec_conclusion.tex")
_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", open(os.path.join(PAPER, "main.tex")).read(), re.S)
ABSTRACT = " ".join("\n".join(re.split(r"(?<!\\)%", l)[0] for l in _m.group(1).split("\n")).split())
NUM = {}
for _l in open(os.path.join(PAPER, "numbers.tex")):
    _mm = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*(%.*)?$", _l.rstrip())
    if _mm:
        NUM[_mm.group(1)] = _mm.group(2).replace("\\xspace", "").replace("\\ensuremath{-}", "-")
lin = {(b, m): e2(b, f"Lin{m}TenDone") for b in BB for m in ("CompB", "CompBp", "CompBAllpos", "RbUnion", "RbTot")}
ok(abs(F["h1_lin_dev_d1_min"] - min(lin.values())) < 1e-9 and abs(F["h1_lin_dev_d1_max"] - max(lin.values())) < 1e-9,
   "'the rollback and compensation repairs we run: X to Y %' is the range over union / total rollback and the three compensations at n = 10, both development logs",
   f"{min(lin.values()):.1f} to {max(lin.values()):.1f}")
ok(max(e2(b, "HoneWhereDone") for b in BB) < F["h1_lin_dev_d1_min"] / 10, "the thresholds-only repair is an order of magnitude below every repair of that range")
ok(all(e2(b, "LinReeditBTenDone") >= F["h1_lin_dev_d1_min"] for b in BB), "the re-run of the editor (no repair of a frozen log; not in the range) is not below the range either")
man = json.load(open(os.path.join(HERE, "numbers_manifest.json")))
dev = {"Q": float(NUM["lgQAcCmp"]), "L": float(NUM["lgLAcCmp"])}
others = [(it["tag"], float(NUM[f"lg{it['tag']}Cmp"])) for it in man["logs"]["items"] if it.get("role") is None]
ok(len(others) == 11 and all(v > dev[t[0]] for t, v in others), "Sec. 1: the deployed log changes more held-out answers 'on every other log of Table tab:logs' (Table 1 since the figure revision) of its backbone (checked on all 11 unseen logs, of which Table tab:logs prints 6; the prefix logs are not rows of the table: final revision T11)",
   f"smallest: {min(others, key=lambda z: z[1])}")
texts = [float(NUM[f"tx{it['tag']}Cmp"]) for it in man["text_logs"]["items"]]
ok(min(texts) > max(dev.values()), "Sec. 1: ... and on the logs written by public code", f"{min(texts):.1f}")
ok(int(NUM["lgQWmmT"]) == max(int(NUM[f"lg{it['tag']}T"]) for it in man["logs"]["items"]) >= max(int(NUM[f"tx{it['tag']}T"]) for it in man["text_logs"]["items"]),
   "'T up to \\lgQWmmT' / 'at most \\lgQWmmT edits': the longest real log")
a, b_, c = (float(NUM[k]) for k in ("TextTOneDeployedDOneHeldout", "TextTOneRepairLogonlyDOneHeldout", "TextTOneRepairTwoblockDOneHeldout"))
ok(a > 3 * b_ and b_ > 3 * c, "clean public-code log: 'the share falls from A to B % from the log alone and to C % with benign queries'", f"{a} > {b_} > {c}")
ok(float(NUM["TextTOneRepairTwoblockParExact"]) < float(NUM["TextTOneDeployedParExact"]) - 5, "'does not improve paraphrases (public-code log: X against Y %)': X is below Y")
ok(F["h1_dev_two_sided"] == 1 and max(e2(b, "HoneRelMinusDeployed") for b in BB) > 2,
   "'our registered endpoint is not met as worded (on one log the repair answers more requests)'")
ok(all(P("xr", f"Xrel{b}{g.DEVTAG[b]}WhereLeastTenHdReleased") > 15 and P("xr", f"Xrel{b}{g.DEVTAG[b]}WhereLeastTenHdChanged") < 3 for b in BB),
   "'still releases an edit on ... % of held-out queries (the answer changes on ... %)': release is not change")
ok(all(P("e6", f"Esix{b}ContamTwentyCovT") < 0.5 * P("e6", f"Esix{b}DefaultCovT") and P("e6", f"Esix{b}ContamFiveCovT") < P("e6", f"Esix{b}DefaultCovT") for b in BB),
   "Sec. 7: 'calibration is sensitive to in-scope queries in the benign sample' (coverage of rephrasings falls with 5 % and halves with 20 % of in-scope queries; report, Table tab:app:ablation)")
ex = json.load(open(os.path.join(ROOT, g.F, "e2_curves/out/running_example.json")))
ok(ex["rescope"]["own_requests_firing"] == ex["T"] and ex["rollback"]["n_deleted"] > 0, "Sec. 1, contribution (1): in the running example the least threshold repair 'loses no edit' and the rollback deletes edits")
for name, txt in (("abstract", ABSTRACT), ("Sec. 1", INTRO), ("Sec. 7", CONCL)):
    low = txt.lower()
    ok(not any(w in low for w in ("complaint-driven", "keeps every", "keep every", "retains every edit", "retains every request", "retains every answer", "correct answer", "better gate", "\\tbd", "\\todo", "\\outl")),
       f"{name}: none of the banned wordings (complaint-driven; keeps / retains every edit, request, answer; correct answer; better gate) and no placeholder")
ok("retains every recorded term" in INTRO and "On the logs we evaluate" in INTRO and "On the logs we evaluate" in CONCL and "the rollback and compensation repairs we run" in INTRO
   and "the rollback and compensation repairs we run" in CONCL and "the rollback and compensation repairs we run" in ABSTRACT,
   "binding wordings are present: 'retains every recorded term', 'on the logs we evaluate', 'the rollback and compensation repairs we run'")
ok("no better" in ABSTRACT and "claim no better one" in INTRO and "no better detector" in CONCL and "does \\emph{not} show a better detector" in INTRO,
   "abstract, Sec. 1 and Sec. 7 say that no better detector is claimed")
files = ["sec_intro.tex", "sec_problem.tex", "sec_theory.tex", "sec_method.tex", "sec_experiments.tex", "sec_related.tex", "sec_conclusion.tex", "tab_positioning.tex"]
n_where = sum(len(re.findall(r"\\sql\{where\}|WHERE clause|\\textsc\{where\}", _body(f))) for f in files) + len(re.findall(r"\\sql\{where\}|WHERE clause", ABSTRACT))
ok(n_where == 1 and "\\sql{update} without a \\sql{where} clause" in INTRO, "the analogy 'an UPDATE without a WHERE clause' occurs once, in Sec. 1 (decision D1)")
ok("\\emph{construct} a relational workload" in INTRO and "constructed relational workload" in ABSTRACT and "relational workload is constructed" in CONCL,
   "the relational workload is called constructed in the abstract, Sec. 1 and Sec. 7 (decisions D12, D16)")
ok(len(re.findall(r"anonymous2026editorconfig", " ".join(_body(f) for f in files))) == 1 and "anonymous2026editorconfig" in _body("sec_experiments.tex"),
   "the related submission of the authors is cited exactly once, in Sec. 5")
# final revision (T1): the availability URL is filled; the marker of its three places must be gone (make strict checks it as well)
tok = "AVAILABILITY-URL-" + "TODO"
_url = re.search(r"(?m)^\\renewcommand\\vldbavailabilityurl\{([^}]*)\}", open(os.path.join(PAPER, "main.tex")).read()).group(1)
ok(_url.startswith("https://") and not any(tok in open(os.path.join(PAPER, f)).read() for f in ("main.tex", "sec_experiments.tex", os.path.join("bib", "local_entries.bib")))
   and _url in open(os.path.join(PAPER, "bib", "local_entries.bib")).read() and "is in the artifact (page~1)" in " ".join(TEXT.split()),   # wording pass of 2026-11-12 (CHECK_LOG.tex, part XII): was "(Artifact Availability, page~1)"; the block "PVLDB Artifact Availability" is on page 1
   "the availability URL is filled in main.tex and in the entry of the technical report, Sec. 5.1 points to the availability block ('in the artifact (page 1)'), and no marker is left", _url)
import abstract_plain as ap                                                   # noqa: E402
plain_abs = ap.plain(_m.group(1), ap.macros())
stored = open(os.path.join(PAPER, "submission", "TITLE_ABSTRACT.txt")).read() if os.path.exists(os.path.join(PAPER, "submission", "TITLE_ABSTRACT.txt")) else ""
ok(150 <= len(plain_abs.split()) <= 185 and plain_abs in stored, "the abstract has 150 to 185 words (final revision T4-T6) and submission/TITLE_ABSTRACT.txt holds its current plain text", f"{len(plain_abs.split())} words")
# ---- final revision of 2026-11-12 (TRIAGE_FINAL.txt): statements that the abstract, Sec. 1, Sec. 5 and Sec. 7 state anew
print("Final revision (TRIAGE_FINAL.txt): new statements")
ok("unique least threshold repair" in ABSTRACT, "T4: the abstract says 'a unique least threshold repair'")
ok("\\vAQLogDone" in ABSTRACT and "\\vALLogDone" in ABSTRACT and "from the log alone; the rollback and compensation repairs we run" in ABSTRACT
   and all(P("e1", f"Eone{b}RescopeWDOne") < 2 * e2(b, "HoneWhereDone") + 1 for b in BB),
   "T5: the abstract prints the log-only values next to the values after 10 complaints (the same macros as Sec. 5.2)",
   " | ".join(f"{P('e1', f'Eone{b}RescopeWDOne'):.1f} against {e2(b, 'HoneWhereDone'):.1f}" for b in BB))
ok("the best detectors that use more evidence match or exceed ours in generalisation" in ABSTRACT
   and "the best detectors that use more evidence match or exceed the coordinate detector in generalisation" in CONCL and "trade off against ours" not in ABSTRACT,
   "T6 / T37, figure revision G11: abstract and Sec. 7 say that THE BEST detectors that use more evidence match or exceed the coordinate detector in generalisation")
ok(all(P("e1", f"Eone{b}Htwo{r}Diff") > 20 for b in BB for r in ("Lookup", "Dual")),
   "G11: 'the best', not 'all': the lookup on the requests and the answer match use more evidence and are behind ReScope in H2",
   " | ".join(f"{P('e1', f'Eone{b}HtwoLookupDiff'):+.1f}, {P('e1', f'Eone{b}HtwoDualDiff'):+.1f}" for b in BB))
_more = []
for b in BB:
    _more.append(P("e1", f"Eone{b}HtwoRetrHi") < 0)                                           # retrieval (requests, encoders): ahead, interval below zero
    _more.append(P("e1", f"Eone{b}HtwoBestCosineHTwoLo") < 0)                                 # selected whitened gate (benign key statistics): ahead or no difference
    for lt in g.UNSEEN[b]:
        _more.append(g.fxp(D, b, lt, "HtwoCosAllLo") < 0 and g.fxp(D, b, lt, "HthreeRetrDiff") < 0)
ok(all(_more), "'match or exceed': in H2 no interval of ReScope minus a detector with more evidence (selected whitened gate; retrieval) lies above zero, on the development logs or on any unseen log")
# figure revision of 2026-11-12 (G13; FINAL_CHECK_2 N3): the clause on image rephrasings of the string lookup is gone (it holds by the
# construction of the lookup rule); the sentence now says what the request-keyed store does WITH encoders (the retrieval row)
_gen = []
for b in BB:
    _oth = [(P("e1", f"Eone{b}{r}TGen"), P("e1", f"Eone{b}{r}IGen")) for r in ("Rescope", "RescopeX", "Cosine", "BestCos", "BestCosHtwo", "Lookup", "Dual", "Deployed", "RescopeW")]
    _gen.append(all(P("e1", f"Eone{b}RetrTGen") > v[0] and P("e1", f"Eone{b}RetrIGen") > v[1] for v in _oth))
    _gen.append(sum(P("e1f", f"EoneFx{b}RetrHN{k}FireMean") for k in "IQX") > sum(P("e1f", f"EoneFx{b}RescopeHN{k}FireMean") for k in "IQX"))
    for lt in g.UNSEEN[b]:
        _r, _s = g.e3row(D, b, lt, "retr"), g.e3row(D, b, lt, "rescope")
        _gen.append(_r["TGen"] + _r["IGen"] > _s["TGen"] + _s["IGen"])
ok(all(_gen) and "with encoders over image and question it generalises best on the vision-language logs (Table~\\ref{tab:main}), at more near-scope releases" in INTRO
   and "LookupIGen" not in INTRO,
   "G13 (was T10): 'with encoders over image and question it generalises best on the vision-language logs (Table tab:main), at more near-scope releases': retrieval has the "
   "largest T-Gen and I-Gen of the rows of Table tab:main on both development logs and more near-scope release than ReScope there; on every unseen log its mean of T-Gen and I-Gen is above ReScope's")
ok("is the best row of our tables" in INTRO, "T10 / G13: 'where a request has a string key it is the best row of our tables' (the rows: scripts/check_e10_statements.py, blocks RQ4 and RQ6)")
ok("a detector on the solver keys with benign key statistics is ahead on the second editor family" in INTRO and "a plain cosine on the solver keys is the robust choice on the collapsed log" in INTRO,
   "T13: Sec. 1 names what each of the two detectors on the solver keys needs (second family: H2 checks of RQ3 above; collapsed log: check_e10_statements.py)")
# last pass of 2026-11-12 (DECISIONS_LAST_PASS.txt, L10; FINAL_CHECK_3 N6): 'verified independently' read as verification by other people
_v = "by a separate automated pass with its own code"
ok("each experiment and each later read-out verified " + _v + ", on" in INTRO
   and "each experiment was verified, and each later read-out recomputed from the stored rows, " + _v + " (Sec.~\\ref{sec:conclusion})" in " ".join(TEXT.split())
   and "verified independently" not in INTRO + " ".join(TEXT.split()) and "recomputed independently" not in INTRO + " ".join(TEXT.split())
   and "An independent audit" not in " ".join(TEXT.split()) and "A separate automated audit confirmed replay" in " ".join(TEXT.split())
   and "independent verifications (separate instances with their own code)" in CONCL,
   "T14 / L10 / L2: Sec. 1 and Sec. 5.1 say that experiments and later read-outs were verified 'by a separate automated pass with its own code' (Sec. 5.1 points to Sec. 7, "
   "whose statement on AI tools says 'separate instances with their own code'); neither says 'verified independently', and the audit of Sec. 5.1 is 'a separate automated audit' "
   "(the recomputation: x_release/verify under the experiments directory; its verdict file is checked below)")
_doc = json.load(open(os.path.join(ROOT, g.F, "e2_curves/out/fixup/paper_numbers_fixup_doc.json")))
ok(all("95 % interval of that mean" in _doc[f"EtwoFx{b}WCalFreshTwoHundredLo"] for b in BB) and "(95\\,\\% interval of the mean)" in " ".join(TEXT.split()),
   "T32: the two bracketed pairs of the post hoc read-out are 95 % intervals of the mean (the experiment's documentation), and the text says so")
ok("The families differ in editor, statistics and ridge; we do not compare them with each other." in " ".join(TEXT.split()), "T29: Sec. 5.1 says that the two editor families are not compared with each other")
ok("(stream positions count from 0)" in " ".join(TEXT.split()), "T36: Sec. 5.5 says that stream positions count from 0")
ok("Position}" in open(os.path.join(PAPER, "sec_intro.tex")).read() and "What we do not claim" not in INTRO, "T8: the paragraph 'Position' replaces 'What we do not claim'")
ok("on a grid of step \\vACapStep" in INTRO and "New statements:" in INTRO and "is prior work" in INTRO, "T9: contribution (2) says which statements are new, and prints the capacity 'on a grid of step 5'")
# last pass of 2026-11-12 (DECISIONS_LAST_PASS.txt, L4; FINAL_CHECK_3 S3): the sentence of Sec. 1 on the unseen logs of the first editor family
ok("on the unseen logs of the first editor family the log-only repair leaves \\vCQAWDoneMin to \\vCQAWDoneMax{}\\,|\\,\\vCLAWDoneMin to \\vCLAWDoneMax{}\\,\\%" in INTRO,
   "L4: Sec. 1 holds 'on the unseen logs of the first editor family the log-only repair leaves <min> to <max> | <min> to <max> %' through the macros \\vC<Q|L>AWDoneMin / Max")
for b in BB:
    _fam1 = [lt for lt in g.UNSEEN[b] if lt[1].endswith("ae-l2_3")]
    _d1 = [g.e3row(D, b, lt, "where")["D1"] for lt in _fam1]
    ok(len(_fam1) == 2 and all(not lt[1].endswith("ae-l2_3") or lt in _fam1 for lt in g.UNSEEN[b])
       and abs(F[f"e3_{b}a_w_d1_min"] - min(_d1)) < 1e-9 and abs(F[f"e3_{b}a_w_d1_max"] - max(_d1)) < 1e-9
       and NUM[f"vC{b}AWDoneMin"] == f"{min(_d1):.1f}" and NUM[f"vC{b}AWDoneMax"] == f"{max(_d1):.1f}",
       f"L4, {b}: the printed range is the minimum and the maximum of the held-out D1 of the log-only ReScope-W over exactly the two unseen logs of the first editor family (ae-l2_3)",
       f"{[lt[1] for lt in _fam1]}: {[round(v, 2) for v in _d1]}; printed {NUM[f'vC{b}AWDoneMin']} to {NUM[f'vC{b}AWDoneMax']}")

# ---- figure revision of 2026-11-12 (DECISIONS_FIG_REVISION.txt, G18; FINAL_CHECK_2 N9): the sentence on independent recomputation
# (Sec. 1, Sec. 5.1) is checked against the verdict file of the recomputation itself, not against its wording only
print("Figure revision (DECISIONS_FIG_REVISION.txt): the recomputation's verdicts; Figure 1; the statement on AI tools")
_vp = os.path.join(ROOT, g.F, "x_release/verify/out/verdicts.json")
ok(os.path.exists(_vp), "the verdict file of the independent recomputation exists (x_release/verify/out/verdicts.json)")
if os.path.exists(_vp):
    _vd = json.load(open(_vp))
    _lists = {k: v for k, v in _vd.items() if isinstance(v, list)}
    ok({"macros", "table_numbers"} <= set(_lists) and all(len(v) > 0 for v in _lists.values()) and all(e.get("agree") is True for v in _lists.values() for e in v),
       "every entry of every list of the verdict file (macros, table numbers) has agree = true",
       ", ".join(f"{k}: {sum(1 for e in v if e.get('agree') is True)} of {len(v)}" for k, v in _lists.items()))
    ok(_vd["other_recomputed_fixup_numbers"]["n"] == _vd["other_recomputed_fixup_numbers"]["n_agree"], "... and none of the further recomputed fix-up numbers differs",
       f"{_vd['other_recomputed_fixup_numbers']['n_agree']} of {_vd['other_recomputed_fixup_numbers']['n']}")
# Figure fig:overview (G28): one figure at the full column width, a caption without a typed number, the three references.
# Vector-figure pass of 2026-11-13 (CHECK_LOG.tex, part XV): ONE file, figs/overview_combined.pdf (the author's vector drawing, both
# parts of the figure), included once through \figOverview under one height limit; the two PNG panels are out of figs/ and nothing includes them.
_fig = re.search(r"\\begin\{figure\}\[t\](.*?)\\end\{figure\}", _body("sec_intro.tex"), re.S)
ok(bool(_fig) and _fig.group(1).count("\\includegraphics") == 1
   and _fig.group(1).count("\\includegraphics[width=\\columnwidth,height=224.3bp,keepaspectratio]{\\figOverview}") == 1
   and "\\label{fig:overview}" in _fig.group(1) and "\\Description{" in _fig.group(1),
   "Fig. fig:overview: one figure environment in Sec. 1 with ONE \\includegraphics, the file \\figOverview at \\columnwidth under the height limit 224.3bp "
   "(keepaspectratio, so that a taller file cannot make the float taller), a description and its label")
_pre = _body("preamble.tex")
ok(_pre.count("\\newcommand{\\figOverview}{figs/overview_combined}") == 1 and "\\figOverviewTop" not in _pre and "\\figOverviewBottom" not in _pre,
   "Fig. fig:overview: preamble.tex defines \\figOverview = figs/overview_combined (no extension) and no longer the two panel macros")
_cap = re.search(r"\\caption\{(.*?)\}\s*\\Description", _fig.group(1), re.S).group(1) if _fig else ""
ok(_cap and not re.search(r"\d", re.sub(r"\\ref\{[^}]*\}", "", _cap)) and all(x in _cap for x in ("\\ref{def:scoped}", "\\ref{def:repairs}", "\\ref{alg:calib}", "\\ref{thm:least}", "\\ref{prop:certificate}"))
   and "(circles: other logged keys)" in _cap and "(dots; $\\base_i+\\scaleh$" in _cap,
   "Fig. fig:overview: the caption prints no number, cites Defs. def:scoped and def:repairs, Algorithm alg:calib, Thm. thm:least, Prop. prop:certificate, and says what the circles and the dots of the bottom panel are (last pass L6)")
_ovf = os.path.join(PAPER, "figs", "overview_combined.pdf")
ok(os.path.exists(_ovf) and open(_ovf, "rb").read(5) == b"%PDF-", "the file of Fig. fig:overview is in figs/ (overview_combined.pdf, a PDF)")
_ov_old = [f for f in ("overview_a.png", "overview_b.png", "overview_a.pdf", "overview_b.pdf") if os.path.exists(os.path.join(PAPER, "figs", f))]
_ov_src = [f for f in sorted(os.listdir(PAPER)) if f.endswith(".tex") and f not in ("CHECK_LOG.tex", "NOTES_FOR_AUTHORS.tex")
           and re.search(r"overview_[ab]\b|figOverviewTop|figOverviewBottom|overview_combined\.svg", _body(f))]
ok(not _ov_old and not _ov_src, "the two former panels of Fig. fig:overview are out of figs/ and no source file names them, their macros or the SVG source outside comments",
   "in figs/: " + (", ".join(_ov_old) or "none") + "; named in: " + (", ".join(_ov_src) or "none"))
ok("Fig.~\\ref{fig:overview}, top" in INTRO and "Fig.~\\ref{fig:overview}" in _body("sec_problem.tex") and "Fig.~\\ref{fig:overview} (top)" in _body("sec_theory.tex")
   and "Fig.~\\ref{fig:overview}, bottom" in _body("sec_method.tex"), "Fig. fig:overview is referred to in Secs. 1, 2, 3 (top panel) and 4 (bottom panel)")
# statement pass of 2026-11-13 (CHECK_LOG.tex, part XIII): the paragraph was reworded (the first author's role first, the tools as tools;
# every use that the earlier paragraph disclosed stays disclosed). Earlier first sentence: "This work made extensive use of generative AI."
# second statement pass of 2026-11-13 (CHECK_LOG.tex, part XIV): final wording, the list says what the assistant was used TO DO
# ("to develop the approach, draft proofs, code and text, and run the experiments ..."); the required words ask for "to develop the approach".
_ai = ("\\paragraph{Use of AI tools} The first author initiated and supervised this work and set its direction, scope and constraints. "
       "Under this supervision, generative AI was used extensively: an AI assistant (Claude, Anthropic) to develop the approach, draft proofs, "
       "code and text, and run the experiments and their independent verifications (separate instances with their own code); a second language "
       "model as a critic; an image model for the draft of Fig.~")
# passes of 2026-11-13 (CHECK_LOG.tex, parts XV and XVI): a clause ", which the first author redrew" was added to the last item and taken out again.
# On 2026-11-13 the first author confirmed that the vector drawing of Fig. 1 is the first author's own work and decided that the paragraph
# keeps its wording without a clause on the redraw; the check below (no "redrew" in either paragraph) keeps it so.
# The tested variant with the clause is under _wvec2_orig/ (the two paragraphs and the strings of this check).
_ai_end = {"sec_conclusion.tex": "\\ref{fig:overview}. The authors reviewed the work and are responsible for its content.",
           "techreport.tex": "\\pref{fig:overview} of the paper. The authors reviewed the work and are responsible for its content."}
_ai_words = ("extensively", "Claude, Anthropic", "to develop the approach", "proofs", "code", "text", "experiments", "verifications", "separate instances",
             "second language model", "image model", "reviewed", "responsible")
_ai_par = {}
for _f in _ai_end:
    _pl = [l for l in open(os.path.join(PAPER, _f)).read().split("\n") if l.startswith("\\paragraph{Use of AI tools}")]
    _ai_par[_f] = _pl[0][len("\\paragraph{Use of AI tools}"):] if len(_pl) == 1 else ""
_tr = _body("techreport.tex")
ok(CONCL.count(_ai) == 1 and CONCL.rstrip().endswith("The authors reviewed the work and are responsible for its content.") and _tr.count(_ai) == 1
   and "denote separate automated passes of such assistants, not reviews by other people" in _tr,
   "the statement on the use of AI tools is the last paragraph of Sec. 7 and stands on the first page of the technical report, there with the sentence on 'referee', 'prover', 'verification'")
ok(all(" ".join(_ai_par[_f].split()) == " ".join((_ai + _ai_end[_f]).split())[len("\\paragraph{Use of AI tools}"):].strip() for _f in _ai_end),
   "statement pass of 2026-11-13: the paragraph 'Use of AI tools' of the paper and of the technical report is the reworded text, word for word "
   "(the report says 'Fig. 1 of the paper' for the figure)")
ok(all(w in _ai_par[_f] for _f in _ai_end for w in _ai_words),
   "statement pass of 2026-11-13: the paragraph of the paper and the paragraph of the report each contain every disclosed use and the responsibility: "
   + ", ".join("'" + w + "'" for w in _ai_words),
   "; ".join(_f + ": missing " + (", ".join(w for w in _ai_words if w not in _ai_par[_f]) or "none") for _f in _ai_end))
ok(all(_ai_par[_f] and "redrew" not in _ai_par[_f] for _f in _ai_end),
   "decision of the first author of 2026-11-13 (the vector drawing of Fig. 1 is confirmed as the first author's own work; the paragraph keeps its wording without a clause on the redraw): "
   "neither the paragraph 'Use of AI tools' of the paper nor that of the report contains 'redrew'",
   "; ".join(_f + (": contains 'redrew'" if "redrew" in _ai_par[_f] else ": paragraph not found" if not _ai_par[_f] else ": ok") for _f in _ai_end))
ok("\\label{tab:criteria}" not in _body("sec_theory.tex") and "tab_positioning" not in _body("sec_related.tex") and "\\input{tab_positioning}" in _tr,
   "G1, G2: Tables tab:criteria and tab:positioning are no longer in the body; the technical report inputs the table of closest works")

# ---- wording pass of 2026-11-12 (CHECK_LOG.tex, part XII): "registered" means written into the authors' protocol file, which carries
# the time stamps of their machine; nothing was registered with an external registry. The word is defined once (Sec. 5.1; the
# technical report, page 1); Sec. 1 says "fixed"; the other uses of the word stay.
print("Wording pass (CHECK_LOG.tex, part XII): the word 'registered'")
_flat = " ".join(TEXT.split())
ok("under a protocol fixed before the runs, each experiment and each later read-out verified " + _v + ", on" in INTRO and "protocol registered" not in INTRO,
   "P1: Sec. 1, contribution (4), says 'under a protocol fixed before the runs' (not 'registered')")
ok("Methods, pools, endpoints and decision rules were registered before the runs, that is, written into our protocol file, not an external registry; "
   "each experiment was verified, and each later read-out recomputed from the stored rows, " + _v + " (Sec.~\\ref{sec:conclusion}); "
   "the protocol with its time-stamped amendments is in the artifact (page~1)." in _flat,
   "P2: Sec. 5.1 says, in this order: registered before the runs; that is, written into our protocol file, not an external registry; verified and recomputed "
   "by a separate automated pass with its own code (Sec. 7); the protocol with its time-stamped amendments is in the artifact (page 1)")
ok("Algorithm~\\ref{alg:calib} calibrates thresholds at $\\miscov=\\pcAlpha$ on $\\ncal=\\pcNcal$ \\poolRepair queries; firing and coverage are means over \\pcDraws draws, "
   "answers over \\pcBehDraws. Intervals: paired bootstrap crossed over draws and units (\\pcBoot resamples). Only H1 to H3 are confirmatory (Holm within each); "
   "every other number is descriptive; no conclusion goes beyond the logs used. Methods, pools," in _flat,
   "P2: the other statements of the paragraph stand (calibration by Algorithm alg:calib at \\pcAlpha on \\pcNcal queries; means over \\pcDraws draws, answers over "
   "\\pcBehDraws; paired bootstrap crossed over draws and units; only H1 to H3 confirmatory, Holm within each; every other number descriptive; no conclusion beyond the logs used)")
_all = " ".join(_body(f) for f in ("sec_intro.tex", "sec_problem.tex", "sec_theory.tex", "sec_method.tex", "sec_experiments.tex", "sec_related.tex", "sec_conclusion.tex")) + " " + ABSTRACT
ok(_all.count("external registry") == 1 and not re.search(r"pre-?regist", _all, re.I),
   "the paper defines the word once ('external registry' stands once in the abstract and Secs. 1-7) and nowhere says 'pre-registered' or 'pre-registration'")
ok("not reviews by other people. ``Registered'' means written into the authors' protocol file before the run, with the time stamps of their machine; no external registry was used." in _tr,
   "P3: the technical report, first page, after the sentence on 'referee', 'prover', 'audit', 'verification': '``Registered'' means written into the authors' protocol file "
   "before the run, with the time stamps of their machine; no external registry was used.'")
_stamps = re.findall(r"^- 2026-\d\d-\d\d \d\d:\d\d UTC", proto, re.M)
ok(bool(re.search(r"^## 6\. Amendments", proto, re.M)) and len(_stamps) >= 10 and "by the clock of the machine" in proto and "frozen before the runs" in proto.split("\n")[0],
   "the protocol file is what the sentences say: its title says 'frozen before the runs', its Section 6 holds the amendments with date and time, by the clock of the authors' machine",
   f"{len(_stamps)} amendments stamped with date and time (UTC)")

print(f"\n{n} statements checked, {len(fails)} failed")
if fails:
    for s in fails:
        print("  FAILED: " + s)
    sys.exit(1)
