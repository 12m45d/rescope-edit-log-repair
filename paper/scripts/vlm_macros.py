#!/usr/bin/env python3
"""Text macros of the vision-language experiments (Secs. 3-5): the list, and its synchronisation with the manifest.

    python3 scripts/vlm_macros.py            rewrites the entries of group "vlm" in scripts/numbers_manifest.json:
                                             one entry per macro of MACROS that a .tex file of the paper uses
    python3 scripts/vlm_macros.py --all      keeps every macro of MACROS (used or not)

One line of MACROS = one macro. Its value is read
  ("pn", tag, key)     directly from the experiment's paper_numbers file (tags: gen_vlm.PN_FILES), under the experiment's
                       own key (every key is confirmed here: a key that the file does not hold stops the run), or
  ("fact", name)       from scripts/data/vlm_facts.json, a fact that scripts/gen_vlm.py derives (with source and rule).
Names: v<A|B|C|E|F|G|S><Q|L>...: A = RQ1 (Sec. 5.2), B = RQ2 (5.3), C = RQ3 (5.4), E = RQ5 (5.6), F = RQ6 (5.7),
G = ablations, S = setup; cst... = the cost figures of Sec. 4; thCap... = the capacity statement of Sec. 3.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import gen_vlm as g                                                          # noqa: E402

M = []                                                                       # (name, origin, fmt, extra)


def pn(name, tag, key, fmt="{:.1f}", **extra):
    M.append((name, ("pn", tag, key), fmt, extra))


def fc(name, fact, fmt="{:.1f}", **extra):
    M.append((name, ("fact", fact), fmt, extra))


F0, F1, F2, FD, TX = "{:.0f}", "{:.1f}", "{:.2f}", "{:d}", None
for b, lg in g.DEV.items():
    dt = g.DEVTAG[b]
    # ---------------------------------------------------------------- RQ1 (E2, prefix logs, E5 f, x_release)
    pn(f"vA{b}DepDone", "e2", f"Etwo{b}DepDone"); pn(f"vA{b}DepRel", "e2", f"Etwo{b}DepRel", F0)
    pn(f"vA{b}CompDone", "e2", f"Etwo{b}LinCompBTenDone"); pn(f"vA{b}CompRel", "e2", f"Etwo{b}LinCompBTenRel", F0)
    pn(f"vA{b}CompAllposDone", "e2", f"Etwo{b}LinCompBAllposTenDone")
    pn(f"vA{b}RbDone", "e2", f"Etwo{b}LinRbUnionTenDone"); pn(f"vA{b}RbRel", "e2", f"Etwo{b}LinRbUnionTenRel", F0)
    pn(f"vA{b}RbAllDone", "e2", f"Etwo{b}LinRbUnionAllDone"); pn(f"vA{b}RbAllRel", "e2", f"Etwo{b}LinRbUnionAllRel", F0)
    fc(f"vA{b}NAll", f"n_all_{b}", FD)
    pn(f"vA{b}WDone", "e2", f"Etwo{b}HoneWhereDone"); pn(f"vA{b}WRel", "e2", f"Etwo{b}HoneWhereRel", F0)
    pn(f"vA{b}Diff", "e2", f"Etwo{b}HoneBestDiff"); pn(f"vA{b}DiffLo", "e2", f"Etwo{b}HoneBestDiffLo"); pn(f"vA{b}DiffHi", "e2", f"Etwo{b}HoneBestDiffHi")
    pn(f"vA{b}RelDelta", "e2", f"Etwo{b}HoneRelMinusDeployed", F0)
    pn(f"vA{b}Kept", "e2f", f"EtwoFx{b}HoneRelKept", F0); pn(f"vA{b}Gained", "e2f", f"EtwoFx{b}HoneRelGained", F0); pn(f"vA{b}Lost", "e2f", f"EtwoFx{b}HoneRelLost", F0)
    pn(f"vA{b}LogDone", "e1", f"Eone{b}RescopeWDOne"); pn(f"vA{b}WAllDone", "e2", f"Etwo{b}WLogAllDone"); pn(f"vA{b}CalDone", "e2", f"Etwo{b}WCalDone", F2)
    pn(f"vA{b}StartTenDone", "e2", f"Etwo{b}WDepTenDone"); pn(f"vA{b}StartAllDone", "e2", f"Etwo{b}WDepAllDone")
    pn(f"vA{b}StartTenRls", "e2f", f"EtwoFx{b}WDepTenFireHeld", F0)
    pn(f"vA{b}Rls", "xr", f"Xrel{b}{dt}WhereLeastTenHdReleased"); pn(f"vA{b}Chg", "xr", f"Xrel{b}{dt}WhereLeastTenHdChanged")
    pn(f"vA{b}ChgRls", "xr", f"Xrel{b}{dt}WhereLeastTenHdChangedReleased"); pn(f"vA{b}ChgNoRls", "xr", f"Xrel{b}{dt}WhereLeastTenHdChangedNoRelease", F0)
    for k in "IQX":
        pn(f"vA{b}NearRls{k}", "xr", f"Xrel{b}{dt}WhereLeastTenHn{k}Released", F0)
        pn(f"vA{b}NearChg{k}", "xr", f"Xrel{b}{dt}WhereLeastTenHn{k}Changed", F0)
        pn(f"vA{b}DepNear{k}", "e1", f"Eone{b}DeployedHN{k}DOne", F0)
        pn(f"vA{b}NearCr{k}", "xr", f"Xrel{b}{dt}WhereLeastTenHn{k}ChangedReleased", F0)
        pn(f"vB{b}SysRls{k}", "xr", f"Xrel{b}{dt}RescopeHn{k}Released", F0)
        pn(f"vB{b}SysChg{k}", "xr", f"Xrel{b}{dt}RescopeHn{k}Changed", F0)
        pn(f"vB{b}XNear{k}", "e1f", f"EoneFx{b}RescopeXHN{k}FireMean", F0)
        pn(f"vB{b}SysNear{k}", "e1f", f"EoneFx{b}RescopeHN{k}FireMean", F0)
    pn(f"vA{b}WTGen", "e2", f"Etwo{b}WLogTenTGen", F0); pn(f"vA{b}WIGen", "e2", f"Etwo{b}WLogTenIGen", F0)
    pn(f"vA{b}DepTGen", "e1", f"Eone{b}DeployedTGen", F0); pn(f"vA{b}DepIGen", "e1", f"Eone{b}DeployedIGen", F0)
    fc(f"vA{b}PfxRbFiftyDone", f"pfx_{b}50_rb_d1"); fc(f"vA{b}PfxRbFiftyRel", f"pfx_{b}50_rb_rel", F0)
    fc(f"vA{b}PfxFiftyWRel", f"pfx_{b}50_w_rel", F0); fc(f"vA{b}PfxFiftyDepRel", f"pfx_{b}50_dep_rel", F0); fc(f"vA{b}PfxFiftyDelta", f"pfx_{b}50_rel_delta", F0)
    fc(f"vA{b}CapFull", f"cap_fullrank_list_{b}", TX)
    fc(f"vA{b}CapMeas", f"cap_meas_list_{b}", TX); fc(f"vA{b}CapPred", f"cap_pred_list_{b}", TX); fc(f"thCapMeas{b}", f"cap_meas_{b}", TX)
    pn(f"thCapNd{b}", "e5", f"capacity.{lg}.nd_holds", TX, map={"True": "holds", "False": "fails"})
    # final revision (T31, T39 f): the registered predictions on the grid of the measurement, the per-complaint values, the second prediction
    fc(f"vA{b}CapPredGrid", f"cap_pred_grid_list_{b}", TX); fc(f"vA{b}CapEvery", f"cap_every_list_{b}", TX)
    fc(f"vA{b}CapWzPred", f"cap_w0_pred_list_{b}", TX); fc(f"vA{b}CapWzPredGrid", f"cap_w0_pred_grid_list_{b}", TX); fc(f"vA{b}CapWzEvery", f"cap_w0_every_list_{b}", TX)
    pn(f"vB{b}LookupIGen", "e1", f"Eone{b}LookupIGen", F0)                 # T10: string lookup on the requests, image rephrasings (report, Table tab:app:main)
    # ---------------------------------------------------------------- RQ2 (E1 and its fix-up)
    pn(f"vB{b}HTwo", "e1", f"Eone{b}HtwoBestCosineHTwoDiff"); pn(f"vB{b}HTwoLo", "e1", f"Eone{b}HtwoBestCosineHTwoLo"); pn(f"vB{b}HTwoHi", "e1", f"Eone{b}HtwoBestCosineHTwoHi")
    pn(f"vB{b}HTwoP", "e1", f"Eone{b}HtwoBestCosineHTwoPHolm", "{:.3f}")
    pn(f"vB{b}HTwoRetr", "e1", f"Eone{b}HtwoRetrDiff")
    pn(f"vB{b}HThree", "e1", f"Eone{b}HthreePairOneDiff"); pn(f"vB{b}HThreeLo", "e1", f"Eone{b}HthreePairOneLo"); pn(f"vB{b}HThreeHi", "e1", f"Eone{b}HthreePairOneHi")
    pn(f"vB{b}HThreeX", "e1", f"Eone{b}HthreePairTwoXDiff"); pn(f"vB{b}HThreeXLo", "e1", f"Eone{b}HthreePairTwoXLo"); pn(f"vB{b}HThreeXHi", "e1", f"Eone{b}HthreePairTwoXHi")
    pn(f"vB{b}HThreeRetr", "e1", f"Eone{b}HthreeRetrDiff")
    pn(f"vB{b}SysDone", "e1", f"Eone{b}RescopeDOne"); pn(f"vB{b}SysRel", "e1", f"Eone{b}RescopeRel", F0)
    pn(f"vB{b}SysTGen", "e1", f"Eone{b}RescopeTGen", F0); pn(f"vB{b}XTGen", "e1", f"Eone{b}RescopeXTGen", F0)
    pn(f"vB{b}RetrTGen", "e1", f"Eone{b}RetrTGen", F0); pn(f"vB{b}RetrNearI", "e1f", f"EoneFx{b}RetrHNIFireMean", F0)
    pn(f"vB{b}WhiteGateFire", "e1", f"Eone{b}BestCosHtwoGateFire"); pn(f"vB{b}WhiteGateFireB", "e1", f"Eone{b}BestCosGateFire")
    pn(f"vB{b}WhiteCapped", "e1", f"Eone{b}BestCosHtwoCapped", F0)
    pn(f"vB{b}RetrGateFire", "e1", f"Eone{b}RetrGateFire")
    pn(f"vB{b}DepDtwo", "e1f", f"EoneFx{b}DeployedDtwo"); pn(f"vB{b}WDtwo", "e1f", f"EoneFx{b}RescopeWDtwo"); pn(f"vB{b}SysDtwo", "e1f", f"EoneFx{b}RescopeDtwo")
    pn(f"vB{b}SysWT", "e1f", f"EoneFx{b}RescopeHNAllWrongTarget"); pn(f"vB{b}RetrWT", "e1f", f"EoneFx{b}RetrHNAllWrongTarget")
    for feat, nm in (("ncoord", "Coord"), ("cos", "Cos"), ("white_0.1", "White")):
        fc(f"vB{b}Front{nm}FF", f"front_{b}_{feat}_50_ff"); fc(f"vB{b}Front{nm}Near", f"front_{b}_{feat}_50_near", F0)
    # ---------------------------------------------------------------- RQ3 (E3, fix-ups, E5 a)
    for fn, FN in (("a", "A"), ("m", "M")):
        fc(f"vC{b}{FN}T", f"e3l_{b}{fn}_T", TX)
        for meth, MN in (("deployed", "Dep"), ("where", "W"), ("rescope", "S"), ("white", "Wh"), ("retr", "Rt")):
            for met, MT in (("D1", "Done"), ("Rel", "Rel"), ("TGen", "TGen"), ("IGen", "IGen")):
                fc(f"vC{b}{FN}{MN}{MT}", f"e3l_{b}{fn}_{meth}_{met}", TX)
        fc(f"vC{b}{FN}HTwo", f"e3l_{b}{fn}_h2", TX); fc(f"vC{b}{FN}Cap", f"e3l_{b}{fn}_cap", TX)
        for short, SN in (("dep", "Dep"), ("w", "W"), ("s", "S")):
            fc(f"vC{b}{FN}Ret{SN}", f"ret_{b}{fn}_{short}", TX)
            fc(f"vC{b}{FN}Ret{SN}Min", f"ret_{b}{fn}_{short}_min", F0); fc(f"vC{b}{FN}Ret{SN}Max", f"ret_{b}{fn}_{short}_max", F0)
        fc(f"vC{b}{FN}ErrLog", f"e3_{b}{fn}_errlog_max", F0, scale=100)
        fc(f"vC{b}{FN}CapMax", f"e3_{b}{fn}_s_cap_max", F0)
        fc(f"vC{b}{FN}SRelMin", f"e3_{b}{fn}_s_rel_min", F0)
        fc(f"vC{b}{FN}SDoneMin", f"e3_{b}{fn}_s_d1_min"); fc(f"vC{b}{FN}SDoneMax", f"e3_{b}{fn}_s_d1_max")
        fc(f"vC{b}{FN}WDoneMin", f"e3_{b}{fn}_w_d1_min"); fc(f"vC{b}{FN}WDoneMax", f"e3_{b}{fn}_w_d1_max")       # T12: log-only repair on the unseen logs of a family
    last = g.UNSEEN[b][-1]
    pn(f"vC{b}Margin", "e3qf" if b == "Q" else "e3lf", f"EthreeFx{b}{last[2]}MarginSmallest", "{:.3f}")
    pn(f"vC{b}MarginN", "e3qf" if b == "Q" else "e3lf", f"EthreeFx{b}{last[2]}MarginNonPositive", FD)
    for lt in g.UNSEEN[b][:2]:
        t = "e3qf" if b == "Q" else "e3lf"
        nm = {200: "Acc", 500: "Ad"}[lt[3]]
        pn(f"vC{b}{nm}HTwo", t, f"EthreeFx{b}{lt[2]}HtwoCosAllDiff"); pn(f"vC{b}{nm}HTwoLo", t, f"EthreeFx{b}{lt[2]}HtwoCosAllLo"); pn(f"vC{b}{nm}HTwoHi", t, f"EthreeFx{b}{lt[2]}HtwoCosAllHi")
    # ---------------------------------------------------------------- RQ5 (E5 c, E6, E2 c')
    v = lambda pair, leaf, rule="two_block": f"validity.{lg}.{rule}.{pair}.{leaf}"            # noqa: E731
    for pair, nm in (("repair_to_repair_rest", "SameRep"), ("ben_s_to_ben_s_rest", "SameBen"), ("mme_c_to_mme_c_rest", "SameMme")):
        pn(f"vE{b}{nm}", "e5", v(pair, "ff_mean"), F2)
    pn(f"vE{b}Held", "e5", v("repair_to_heldout", "ff_mean"), F2); pn(f"vE{b}HeldLo", "e5", v("repair_to_heldout", "ff_p05"), F2); pn(f"vE{b}HeldHi", "e5", v("repair_to_heldout", "ff_p95"), F2)
    pn(f"vE{b}BenT", "e5", v("repair_to_ben_t", "ff_mean"), F2)
    pn(f"vE{b}Mme", "e5", v("repair_to_mme_t", "ff_mean"), F2); pn(f"vE{b}MmeLo", "e5", v("repair_to_mme_t", "ff_p05"), F2); pn(f"vE{b}MmeHi", "e5", v("repair_to_mme_t", "ff_p95"), F2)
    pn(f"vE{b}MmeAbove", "e5", v("repair_to_mme_t", "share_of_draws_above_alpha")); pn(f"vE{b}BenMme", "e5", v("ben_s_to_mme_t", "ff_mean"), F2)
    pn(f"vE{b}Img", "e5", v("repair_to_heldout", "image_rows.ff_mean")); pn(f"vE{b}Txt", "e5", v("repair_to_heldout", "text_rows.ff_mean"))
    pn(f"vE{b}PacCovT", "e6", f"Esix{b}PacCovT", F0); pn(f"vE{b}DefCovT", "e6", f"Esix{b}DefaultCovT", F0)
    pn(f"vE{b}ContamOne", "e6", f"Esix{b}ContamOneCovT", F0); pn(f"vE{b}ContamFive", "e6", f"Esix{b}ContamFiveCovT", F0); pn(f"vE{b}ContamTwenty", "e6", f"Esix{b}ContamTwentyCovT", F0)
    pn(f"vE{b}Fresh", "e2f", f"EtwoFx{b}WCalFreshTwoHundredMean", F2); pn(f"vE{b}FreshLo", "e2f", f"EtwoFx{b}WCalFreshTwoHundredLo", F2); pn(f"vE{b}FreshHi", "e2f", f"EtwoFx{b}WCalFreshTwoHundredHi", F2)
    pn(f"vE{b}HeldW", "e2f", f"EtwoFx{b}WCalHeldTwoHundredMean")
    # ---------------------------------------------------------------- RQ6 (E9, E8)
    pn(f"vF{b}Viol", "e9", f"Enine{b}FrameViolations", FD); pn(f"vF{b}FrameN", "e9", f"Enine{b}FrameSequences", "{:,d}")
    pn(f"vF{b}CertSilent", "e9", f"Enine{b}CertSilent", FD); pn(f"vF{b}CertIdentical", "e9", f"Enine{b}CertIdentical", FD)
    fc(f"vF{b}RevokedMax", f"rev_{b}_scoped_revoked_max", F0)
    pn(f"vF{b}LinFlagsOne", "e9", f"Enine{b}LinearMOneOutsideFlags", F0); pn(f"vF{b}LinFlagsFifty", "e9", f"Enine{b}LinearMFiftyOutsideFlags", F0)
    pn(f"vF{b}LineageN", "e9", f"Enine{b}LineageAllN", FD); pn(f"vF{b}OneMask", "e9", f"Enine{b}LineageAllOneMask")
    pn(f"vF{b}LineageMax", "e9", f"Enine{b}LineageAllMax", FD)
    pn(f"cst{b}GateMs", "e8", f"{lg}.gate_ms_per_forward", F2); pn(f"cst{b}OverPct", "e8", f"{lg}.overhead_end_to_end_pct")
    pn(f"cst{b}StoreMb", "e8", f"{lg}.storage_mb"); pn(f"cst{b}CalS", "e8", f"{lg}.calibration_s", F2); pn(f"cst{b}RepMs", "e8", f"{lg}.repair_ms_per_complaint", F2)
    # final revision (T3; FINAL_CHECK M3): the serving cost of ReScope-W (every position): calibrated thresholds (paper numbers) and log-only thresholds (records)
    pn(f"cst{b}WGateMs", "e8", f"{lg}.gate_ms_per_forward_all_positions", F2); pn(f"cst{b}WOverPct", "e8", f"{lg}.overhead_end_to_end_pct_all_positions")
    fc(f"cst{b}WLogGateMs", f"cost_{b}_wlog_gate_ms", F2); fc(f"cst{b}WLogOverPct", f"cost_{b}_wlog_over_pct")
    # figure revision, G17: the second latency pass of the same experiment (ReScope; ReScope-W at every position)
    fc(f"cst{b}GateMsTwo", f"cost_{b}_s_gate_ms_p2", F2); fc(f"cst{b}OverPctTwo", f"cost_{b}_s_over_pct_p2")
    fc(f"cst{b}WGateMsTwo", f"cost_{b}_wall_gate_ms_p2", F2); fc(f"cst{b}WOverPctTwo", f"cost_{b}_wall_over_pct_p2")
    pn(f"vF{b}Util", "e8", f"{lg}.card_utilisation_pct_median", F0)
    pn(f"vF{b}OverLo", "e8", f"{lg}.overhead_end_to_end_pct_p10", F0); pn(f"vF{b}OverHi", "e8", f"{lg}.overhead_end_to_end_pct_p90", F0)
    pn(f"vF{b}Numbers", "e8", f"{lg}.storage_numbers", "{:,d}")
    pn(f"vF{b}PerEdit", "e8", "storage." + {"Q": "qwen25vl", "L": "llava"}[b] + ".numbers_per_edit", "{:,d}")
    # ---------------------------------------------------------------- ablations (E6)
    pn(f"vG{b}RunMax", "e6", f"Esix{b}CalRunMaxFire"); pn(f"vG{b}DefFire", "e6", f"Esix{b}DefaultFire")
    pn(f"vG{b}LinRel", "e6", f"Esix{b}RespLinRel", F0); pn(f"vG{b}LinTGen", "e6", f"Esix{b}RespLinTGen", F0); pn(f"vG{b}LinIGen", "e6", f"Esix{b}RespLinIGen", F0)
    pn(f"vG{b}AllCovT", "e6", f"Esix{b}RoleAllCovT", F0)
    fc(f"vS{b}Degenerate", f"hn_degenerate_pct_{b}", F0)

# ---- not per backbone
fc("vAHolm", "h1_holm_bound", "{}")
fc("vAPfxN", "h1_prefix_n", FD); fc("vAPfxTwo", "h1_prefix_two_sided", FD); fc("vAPfxOne", "h1_prefix_one_sided", FD)
fc("vARelDeltaMin", "h1_rel_delta_min", F0); fc("vARelDeltaMax", "h1_rel_delta_max", F0); fc("vADoneMax", "h1_d1_max"); fc("vANLogs", "h1_n_logs", FD)
fc("vALeastCells", "least_cells", FD); fc("vALeastOrders", "least_orders", FD)
_WORD = {str(i): w for i, w in enumerate(("none", "one", "two", "three", "four", "five", "six", "seven", "eight"))}
fc("vAHOneBelow", "h1_n_rel_below", TX, map=_WORD); fc("vAHOneAbove", "h1_n_rel_above", TX, map=_WORD)   # T30: counts printed as words
fc("vAQCapWzMiss", "cap_w0_miss_Q", TX, map=_WORD)                                                    # T39 (f): orders in which the second prediction is missed on Qwen
fc("vACapStep", "cap_grid_step_Q", FD)                                                                # T31: step of the capacity grid around the transition
fc("vAPfxDepMin", "pfx_dep_d1_min"); fc("vAPfxDepMax", "pfx_dep_d1_max"); fc("vAPfxWMin", "pfx_w_d1_min"); fc("vAPfxWMax", "pfx_w_d1_max")
fc("vAPfxCompMin", "pfx_comp_d1_min"); fc("vAPfxCompMax", "pfx_comp_d1_max"); fc("vACompMin", "h1_comp_any_d1_min")
fc("vALinMin", "h1_lin_dev_d1_min"); fc("vALinMax", "h1_lin_dev_d1_max")                    # W4: abstract, Secs. 1 and 7 (range of the rollback and compensation repairs at n = 10, development logs)
fc("vACapSilenced", "cap_thr_silenced_max", FD); fc("vACapTol", "cap_tol_rank_tex_Q", "{}")        # a TeX string (power of ten): format "{}" prints it unescaped
fc("vAPfxRbSmallDone", "pfx_rb_small_d1_max"); fc("vAPfxRbSmallRel", "pfx_rb_small_rel_max", F0)
pn("vBVerdictA", "e1", "EoneDecisionPairOne", TX); pn("vBVerdictB", "e1", "EoneDecisionPairTwoX", TX); pn("vBVerdictR", "e1", "EoneDecisionRetr", TX)
fc("vBHTwoOtherMin", "h2_other_min", F0); fc("vBHTwoOtherMax", "h2_other_max", F0)
pn("vBHTwoN", "e1f", "EoneFxHtwoNComparisons", FD); pn("vBHTwoBounds", "e1f", "EoneFxHtwoNHolmBounds", FD)
fc("vBSelCos", "sel_cos_family_n", FD); fc("vBSelRetr", "sel_retr_family_n", FD)
fc("vBSysCrMin", "sys_changed_released_min", F0); pn("vEQPfxHeldW", "e2p", "PfxQFiftyWTwoBlockGateLiteralFired")
fc("vBMcExp", "mc_dep_exp", FD); fc("vBNearSpread", "near_spread_max", F0)
pn("vCHolmHTwoN", "xr", "XrelHolmHtwoFamily", FD); pn("vCHolmHTwoRej", "xr", "XrelHolmHtwoRejected", FD)
pn("vCHolmHThreeN", "xr", "XrelHolmHthreeFamily", FD); pn("vCHolmHThreeRej", "xr", "XrelHolmHthreeRejected", FD)
pn("vCHolmHThreeWorse", "xr", "XrelHolmHthreeRejectedWorse", FD); pn("vCHolmHThreeBetter", "xr", "XrelHolmHthreeRejectedBetter", FD)
fc("vCRetrHThreeMin", "e3_retr_h3_abs_min", F1); fc("vCRetrHThreeMax", "e3_retr_h3_abs_max", F1)
fc("vCCapPOneA", "h3_cap_p1_a_max", F0); fc("vCCapPOneB", "h3_cap_p1_b_max", F0); fc("vCCapPTwoA", "h3_cap_p2_a_max", F0)
fc("vCCapPTwoB", "h3_cap_p2_b_max", F0); fc("vCCapRetr", "h3_cap_retr_b_max", F0); fc("vCUnreach", "h3_unreachable_max", FD)
fc("vCNearRelMin", "e3_near_rel_min", F0); fc("vCNearRelMax", "e3_near_rel_max", F0)
fc("vCNearRelUnseenMin", "e3_near_rel_unseen_min", F0); fc("vCNearRelUnseenMax", "e3_near_rel_unseen_max", F0)   # figure revision, G6: the range over the unseen logs
fc("vCDrawsMin", "e3_draws_above_min", FD); fc("vCDrawsMax", "e3_draws_above_max", FD)
fc("vCErrRebMax", "e3_errreb_max", F1, scale=100)
pn("vCLMmeAcc", "e3l", "EthreeLAeTwoHundredRescopeMmeT"); pn("vCLMmeAd", "e3l", "EthreeLAeFiveHundredRescopeMmeT")
pn("vCQLaterHTwo", "e3qf", "EthreeFxQMeMMHtwoCosLaterDiff"); pn("vCQLaterHTwoLo", "e3qf", "EthreeFxQMeMMHtwoCosLaterLo"); pn("vCQLaterHTwoHi", "e3qf", "EthreeFxQMeMMHtwoCosLaterHi")
pn("vCLLaterHTwo", "e3lf", "EthreeFxLMeThousandHtwoCosLaterDiff"); pn("vCLLaterHTwoLo", "e3lf", "EthreeFxLMeThousandHtwoCosLaterLo"); pn("vCLLaterHTwoHi", "e3lf", "EthreeFxLMeThousandHtwoCosLaterHi")
fc("vESameMeanMin", "val_same_mean_min", F2); fc("vESameMeanMax", "val_same_mean_max", F2)
fc("vESameAboveMin", "val_same_above_min", F0); fc("vESameAboveMax", "val_same_above_max", F0)
fc("vEBeta", "val_beta_above"); fc("vEBetaN", "val_beta_n2", FD); fc("vEBound", "val_bound", F2)
fc("vEPacAboveMax", "val_pac_exch_above_max"); fc("vERevMin", "val_rev_min", F0); fc("vERevMax", "val_rev_max", F0)
fc("vERevPacMin", "val_rev_pac_min", F0); fc("vERevPacMax", "val_rev_pac_max", F0)
fc("vEContamLevels", "contam_levels", TX); pn("vEPostDraws", "e2f", "EtwoFxQWCalNDraws", FD)
fc("vFRbKMin", "rb_k_median_min", "{:g}"); fc("vFRbKMax", "rb_k_median_max", "{:g}")
pn("vFSynPct", "e8", "synthetic.T100000.storage_share_of_model_pct")
pn("cstEqualT", "e8", "storage.T_at_module_size", "{:,d}"); pn("cstNdbStore", "e8", "published.neuraldb.storage_at_1e5_pct", F0)
fc("vGLoggedDiff", "abl_logged_maxdiff", F0)


def used_names():
    names = set()
    for fn in os.listdir(PAPER):
        if fn.endswith(".tex") and fn not in ("numbers.tex", "CHECK_LOG.tex", "NOTES_FOR_AUTHORS.tex"):
            text = "\n".join(re.split(r"(?<!\\)%", l)[0] for l in open(os.path.join(PAPER, fn)))
            names.update(re.findall(r"\\([A-Za-z]+)", text))
    return names


def entries(keep_all=False):
    D = g.Data(g.find_root())
    used = used_names()
    out, seen = [], set()
    for name, origin, fmt, extra in M:
        assert name not in seen and re.fullmatch(r"[A-Za-z]+", name), name
        seen.add(name)
        if not keep_all and name not in used:
            continue
        e = {"name": name, "group": "vlm"}
        if origin[0] == "pn":
            D.p(origin[1], origin[2])                                         # the key exists (KeyError otherwise)
            e.update(file=g.PN_FILES[origin[1]], key=[origin[2]])
        else:
            e.update(source="vlm", key=["facts", origin[1], "value"])
        if fmt is None:
            e["kind"] = "text"
        else:
            e["fmt"] = fmt
        e.update(extra)
        out.append(e)
    return out


def dump_manifest(m):
    """The manifest in its hand-kept layout: one macro per line, one field of a family per line."""
    J = lambda x: json.dumps(x, ensure_ascii=False)                                              # noqa: E731
    out = ["{"]
    keys = list(m)
    for i, k in enumerate(keys):
        v, end = m[k], ("," if i < len(keys) - 1 else "")
        if k == "macros":
            out.append(f'  "{k}": [')
            out += [f"    {J(e)}" + ("," if j < len(v) - 1 else "") for j, e in enumerate(v)]
            out.append("  ]" + end)
        elif k == "families":
            out.append(f'  "{k}": [')
            for j, fam in enumerate(v):
                head = {kk: vv for kk, vv in fam.items() if kk != "fields"}
                out.append("    {" + J(head)[1:-1] + ', "fields": {')
                fl = list(fam["fields"].items())
                out += [f"      {J(fk)}: {J(fv)}" + ("," if n < len(fl) - 1 else "") for n, (fk, fv) in enumerate(fl)]
                out.append("    }}" + ("," if j < len(v) - 1 else ""))
            out.append("  ]" + end)
        elif isinstance(v, list):
            out.append(f'  "{k}": [')
            out += [f"    {J(e)}" + ("," if j < len(v) - 1 else "") for j, e in enumerate(v)]
            out.append("  ]" + end)
        elif isinstance(v, dict):
            body = json.dumps(v, indent=2, ensure_ascii=False).split("\n")
            out.append(f'  "{k}": ' + body[0])
            out += ["  " + line for line in body[1:-1]]
            out.append("  " + body[-1] + end)
        else:
            out.append(f'  "{k}": {J(v)}' + end)
    out.append("}")
    return "\n".join(out) + "\n"


def sync(keep_all=False):
    mp = os.path.join(HERE, "numbers_manifest.json")
    m = json.load(open(mp))
    ent = entries(keep_all)
    names = {e["name"] for e in ent}
    m["macros"] = [e for e in m["macros"] if e.get("group") != "vlm" and e["name"] not in names] + ent
    text = dump_manifest(m)
    assert json.loads(text) == m
    with open(mp + ".tmp", "w") as f:
        f.write(text)
    os.replace(mp + ".tmp", mp)
    print(f"numbers_manifest.json: {len(ent)} macros of group vlm ({len(M)} defined in vlm_macros.py)")


if __name__ == "__main__":
    sync("--all" in sys.argv)
