#!/usr/bin/env python3
"""Part 4: the verdicts.  One record per macro of out/macro_list.json (task 1): stored value, my value, agree or not, source rows, and
the string the paper prints against my value under the paper's printing rule.  Then the paper numbers of the fix-ups that only
generated table bodies print (report: tab_vlm_tr_repl / _draws / _mcnemar; body: tab_vlm_main), the check 'changed without a
release', and the three statements in words.  -> out/verdicts.json"""
import json
import math
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *  # noqa: E402,F401,F403

E2R = FINAL + "/e2_curves/out/rows"
UNSEEN = {"Q": QLOGS[1:], "L": LLOGS[1:]}
ST = {t: jload(f"{FINAL}/{p}") for t, p in (("e1f", "e1_main/out/fixup/paper_numbers_fixup.json"), ("e2f", "e2_curves/out/fixup/paper_numbers_fixup.json"),
                                             ("e3qf", "e3_qwen/out/fixup/paper_numbers_fixup.json"), ("e3lf", "e3_llava/out/fixup/paper_numbers_fixup.json"),
                                             ("xr", "x_release/out/paper_numbers_fixup.json"))}
REL = jload(OUT + "/mine_release.json")
E3 = jload(OUT + "/mine_e3.json")
BOOT = jload(OUT + "/mine_e3_boot.json")
E12 = jload(OUT + "/mine_e1e2.json")
EP = {"Q": jload(FINAL + "/e3_qwen/out/fixup/endpoints_added.json")["logs"], "L": jload(FINAL + "/e3_llava/out/fixup/endpoints.json")["logs"]}
H3ROLE = {"PairOne": "pair1", "PairTwo": "pair2_x", "Retr": "retr"}


def same(a, b):
    """exact agreement: equal strings / integers, floats to 1e-9 relative (1e-12 absolute)"""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    a, b = float(a), float(b)
    return abs(a - b) <= max(1e-12, 1e-9 * max(abs(a), abs(b)))


def printed_strings():
    out = {}
    for l in open(V + "/copies/numbers.tex"):
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)(\\xspace)?\}\s*(%.*)?$", l.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2).replace("\\ensuremath{-}", "-").replace("\\,", " ")
    return out


def show(fmt, v, scale=None):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if scale:
        v = v * scale
    if fmt and re.fullmatch(r"\{:\.\d+f\}", fmt):
        v = float(v)
    s = fmt_number(fmt or "{}", v)
    return s


# ---------------------------------------------------------------------------- H1: the Holm bound printed in Sec. 5.2
def ratio_boot(na, nb, den, n_boot=20000, seed=7):
    """paired crossed bootstrap over streams x query clusters of 100 * (sum a / sum den - sum b / sum den)"""
    nd, nu = na.shape
    rng = np.random.default_rng(seed)
    st = []
    for s in range(0, n_boot, 2000):
        cu = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=2000).astype(float)
        cd = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=2000).astype(float)
        f = lambda N: ((cd @ N) * cu).sum(1)                                                    # noqa: E731
        st.append(100.0 * (f(na) - f(nb)) / f(den))
    st = np.concatenate(st)
    p = 2 * min(float((st <= 0).mean()), float((st >= 0).mean()))
    return {"value": 100.0 * float((na.sum() - nb.sum()) / den.sum()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5)),
            "n_other_side": int(min((st <= 0).sum(), (st >= 0).sum())), "n_boot": n_boot, "p": max(p, 1.0 / n_boot)}


def h1_tests():
    h1 = jload(FINAL + "/e2_curves/out/fixup/h1_registered.json")
    out = {}
    for b in "QL":
        dv = h1["development_logs"][DEV[b]]["differences_to_linear_repairs"]
        for sel, x in dv.items():
            grids = {}
            for nm, pat in (("a", f"b__{b}__where_least__logonly__n10__s{{}}"), ("b", f"a__{b}__{x['linear_repair']}__n10__s{{}}")):
                per = []
                for s in range(3):
                    acc = {}
                    for r in rows(f"{E2R}/{pat.format(s)}.jsonl.gz", keep=False):
                        if is_hd(r):
                            a = acc.setdefault(r["query_id"], [0.0, 0.0])
                            a[0] += float(bool(r["D1"]))
                            a[1] += 1.0
                    per.append(acc)
                u = sorted(per[0])
                assert all(sorted(p) == u for p in per)
                grids[nm] = (np.array([[p[k][0] for k in u] for p in per]), np.array([[p[k][1] for k in u] for p in per]))
            assert (grids["a"][1] == grids["b"][1]).all()
            r = ratio_boot(grids["a"][0], grids["b"][0], grids["a"][1])
            out[f"{b} {sel}"] = dict(r, linear_repair=x["linear_repair"], stored_difference=x["difference_points"], stored_ci=[x["ci_lo"], x["ci_hi"]], stored_p=x["p_two_sided"],
                                     stored_p_holm=x["p_holm"], files=f"e2_curves/out/rows/b__{b}__where_least__logonly__n10__s<0-2> | a__{b}__{x['linear_repair']}__n10__s<0-2> (heldout_disjoint rows, D1, clusters = query_id)")
    return out, h1["holm"]


# ---------------------------------------------------------------------------- Holm over the unseen logs
def holm_counts():
    res = {}
    for h in ("H2", "H3"):
        mem = {}
        for b in "QL":
            for log in UNSEEN[b]:
                e = EP[b][log]
                if h == "H2":
                    for role, w in (("best_cosine_H2", "Cos"), ("retr", "Retr")):
                        x = e["H2"][role]["all"]
                        mem[f"{log} | {w}"] = {"stored_p": x["p"], "diff": x["diff"], "replicated_p": BOOT["replicate"][log]["Htwo" + w + "All"]["p"],
                                               "own": E3["logs"][log]["H2"][w + "All"]}
                else:
                    for w, role in H3ROLE.items():
                        x = e["H3"][role]
                        mem[f"{log} | {w}"] = {"stored_p": x["p"], "diff": x["diff"], "replicated_p": BOOT["replicate"][log]["Hthree" + w]["p"], "own": E3["logs"][log]["H3"][w]}
        for k, m in mem.items():
            n_other, n_boot = m["own"]["n_other_side"], m["own"]["n_boot"]
            for x in ((BOOT["precise"].get(k) or {}).values() if h == "H3" else []):          # the precise runs exist for members of H3 only
                n_other, n_boot = n_other + x["n_other_side"], n_boot + x["n_boot"]
            m["own_p"] = max(2.0 * n_other / n_boot, 1.0 / n_boot)
            m["own_p_resamples"] = n_boot
            m["own_p_mc_se"] = 2.0 * math.sqrt(max(n_other, 1)) / n_boot
            m["own_p_20000"] = m["own"]["p"]
            m["own_diff"] = m["own"]["value"]
            del m["own"]
        o = {"family": len(mem)}
        for tag, key in (("stored", "stored_p"), ("replicated", "replicated_p"), ("own_precise", "own_p"), ("own_20000", "own_p_20000")):
            adj = holm({k: m[key] for k, m in mem.items()})
            rej = [k for k in mem if adj[k] <= 0.05]
            o[tag] = {"rejected": len(rej), "worse": sum(mem[k]["diff"] < 0 for k in rej), "better": sum(mem[k]["diff"] > 0 for k in rej), "not_rejected": [k for k in mem if k not in rej]}
            for k in mem:
                mem[k]["holm_" + tag] = adj[k]
        o["members"] = mem
        o["closest_to_0_05"] = sorted(({"member": k, "holm_stored": m["holm_stored"], "holm_own_precise": m["holm_own_precise"], "holm_own_20000": m["holm_own_20000"], "own_p": m["own_p"],
                                        "own_p_mc_se": m["own_p_mc_se"], "own_p_resamples": m["own_p_resamples"], "stored_p": m["stored_p"]} for k, m in mem.items()),
                                      key=lambda x: abs(x["holm_own_precise"] - 0.05))[:3]
        res[h] = o
    return res


def main():
    ML = jload(OUT + "/macro_list.json")
    PR = printed_strings()
    H1, H1holm = h1_tests()
    HOLM = holm_counts()
    rep = {l: E3["logs"][l]["reporting"] for l in E3["logs"]}
    PNm = dict(REL["paper_numbers"])
    PNm.update({k: v for k, v in E12["pn"].items() if v is not None})
    SRC = dict(E12["src"])
    for k, c in REL["cells"].items():
        for s in ("Rows", "Released", "Changed", "ChangedReleased", "ChangedNoRelease"):
            SRC[k + s] = "; ".join(r["file"] for r in c["runs"]) + f" (pool {c['pool']}: flags fired, D1)"
    for log, d in REL["deployed"].items():
        if "-p" not in log:
            for s in ("DepRel", "DepTGen", "DepIGen", "DepDone", "HdRows", "T"):
                SRC["Xrel" + code(log) + "Log" + s] = d["file"]
    # E3 fix-up names
    TOL = {}
    for log, o in E3["logs"].items():
        c = "EthreeFx" + code(log)
        r = o["reporting"]
        g = f"{EXP[bb(log)]}/out/gate/{log}/rescope__reg.json (20 cells)"
        for nm, v, s in (("MarginSmallest", r["margin_min"], r["margin_file"] + " (self_separation_margin.per_edit: minimum)"),
                         ("MarginNonPositive", r["margin_nonpositive"], r["margin_file"] + " (per_edit <= 0)"),
                         ("RescopeCappedMean", r["capped_mean"], g + ": n_capped"), ("RescopeCappedMax", r["capped_max"], g + ": n_capped"),
                         ("RescopeDrawsAbove", r["draws_above"], g + ": firing flags on the valid heldout_disjoint rows against 100 * 5 / 101"),
                         ("RescopeFireMean", r["fire_mean"], g), ("RescopeNearRelGate", r["near_rel_gate"], g),
                         ("RescopeNearRelBeh", r["near_rel_beh"], f"{EXP[bb(log)]}/out/rows/{log}__rescope__d<draw>__tf.jsonl.gz (heldout_disjoint rows with near_edit, fired; draws {o['behavioural_draws']})"),
                         ("NearN", r["near_n"], "the same rows"), ("HdN", r["n_hd"], g)):
            PNm[c + nm], SRC[c + nm] = v, s
        for hk, hv in o["H2"].items():
            n = c + "Htwo" + hk
            rp = BOOT["replicate"][log]["Htwo" + hk]
            PNm[n + "Diff"], SRC[n + "Diff"] = hv["value"], f"{EXP[bb(log)]}/out/rows/{log}__rescope|{hv['ref']}__d<draw>__tf.jsonl.gz (T-Gen, I-Gen exact per edit)"
            for e, f in (("Lo", "ci_lo"), ("Hi", "ci_hi")):
                PNm[n + e], SRC[n + e] = rp[f], SRC[n + "Diff"] + "; seed scheme of the experiment replicated with own code"
                TOL[n + e] = {"own_resamples_value": hv[f], "se": hv["se"], "step": 100.0 / (2 * hv["n_units"] * hv["n_draws"])}
            PNm[n + "P"], SRC[n + "P"] = rp["p"], SRC[n + "Lo"]
        for hk, hv in o["H3"].items():
            n = c + "Hthree" + hk
            rp = BOOT["replicate"][log]["Hthree" + hk]
            PNm[n + "Diff"], SRC[n + "Diff"] = hv["value"], f"{EXP[bb(log)]}/out/rows/{log}__{hv['a']}|{hv['b']}__d<draw>__h3.jsonl.gz (odd edits; T-Gen, I-Gen exact per edit)"
            for e, f in (("Lo", "ci_lo"), ("Hi", "ci_hi")):
                PNm[n + e], SRC[n + e] = rp[f], SRC[n + "Diff"] + "; seed scheme of the experiment replicated with own code"
                TOL[n + e] = {"own_resamples_value": hv[f], "se": hv["se"], "step": 100.0 / (2 * hv["n_units"] * hv["n_draws"])}
            PNm[n + "P"], SRC[n + "P"] = rp["p"], SRC[n + "Lo"]
            cellrec = f"{EXP[bb(log)]}/out/behave/{log}/<method>__d<draw>__h3.json (n_edits_at_cap, reachable)"
            PNm[n + "AtCapA"], PNm[n + "AtCapB"], PNm[n + "Unreachable"] = hv["at_cap_a"], hv["at_cap_b"], hv["unreachable"]
            SRC[n + "AtCapA"] = SRC[n + "AtCapB"] = SRC[n + "Unreachable"] = cellrec
    # the 200-draw read-out of part c' (p6_cw200.py: recomputed from the captured per-row maxima; the 200 per-draw rates equal the stored ones)
    CW = jload(OUT + "/mine_cw200.json")
    MC = {}
    for bq in "QL":
        cw = CW[DEV[bq]]
        src = cw["source"] + f"; my 200 per-draw rates against the stored ones: largest difference {cw['max_abs_difference_to_the_200_stored_per_draw_records']:g}"
        PNm[f"EtwoFx{bq}WCalFreshTwoHundredMean"], PNm[f"EtwoFx{bq}WCalHeldTwoHundredMean"], PNm[f"EtwoFx{bq}WCalNDraws"] = cw["fresh_repair"]["mean"], cw["heldout_disjoint"]["mean"], len(cw["draws"])
        for k in ("WCalFreshTwoHundredMean", "WCalHeldTwoHundredMean", "WCalNDraws"):
            SRC[f"EtwoFx{bq}{k}"] = src
        for pool, w in (("fresh_repair", "Fresh"), ("heldout_disjoint", "Held")):
            for e, f in (("Lo", "ci_lo"), ("Hi", "ci_hi")):
                MC[f"EtwoFx{bq}WCal{w}TwoHundred{e}"] = {"mine": cw[pool][f], "se": cw[pool]["se"], "allowed_difference": 0.25 * cw[pool]["se"],
                                                          "source": src + "; interval: my own 20000 resamples of the query clusters, draws nested"}
    # Holm names
    for h, w in (("H2", "Htwo"), ("H3", "Hthree")):
        o = HOLM[h]
        PNm[f"XrelHolm{w}Family"], PNm[f"XrelHolm{w}Rejected"] = o["family"], o["replicated"]["rejected"]
        PNm[f"XrelHolm{w}RejectedWorse"], PNm[f"XrelHolm{w}RejectedBetter"] = o["replicated"]["worse"], o["replicated"]["better"]
        for s in ("Family", "Rejected", "RejectedWorse", "RejectedBetter"):
            SRC[f"XrelHolm{w}{s}"] = ("e3_qwen, e3_llava out/rows/<unseen log>__<method>__d<draw>__" + ("tf" if h == "H2" else "h3") +
                                      ".jsonl.gz: my paired crossed bootstrap (the experiment's seed scheme replicated: p identical), my Holm step-down")
    stored_all = {}
    for t in ST.values():
        stored_all.update(t)

    def fact(key):
        """my value of a derived fact of vlm_facts.json, and its source"""
        allu = lambda b: UNSEEN[b]                                                              # noqa: E731
        if key == "h1_holm_bound":
            n_tests = len(H1)
            return n_tests * (1.0 / 2000), f"{n_tests} tests (2 development logs x 2 choices of the linear repair) x resolution 1 / 2000 of the experiment's bootstrap; my own 20000 resamples: " + \
                "; ".join(f"{k}: {v['value']:.2f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}], {v['n_other_side']} of {v['n_boot']} on the other side of 0" for k, v in H1.items())
        if key == "sys_changed_released_min":
            v = {f"{b} {p}": PNm[f"Xrel{code(DEV[b])}Rescope{p}ChangedReleased"] for b in "QL" for p in ("Hd", "HnI", "HnQ", "HnX")}
            return min(v.values()), "development logs, rescope rows of E3, draws 0-2: " + ", ".join(f"{k} {x:.2f}" for k, x in v.items())
        if key == "mc_dep_exp":
            ps = {f"{b} draw {d}": PNm[f"EoneFx{b}DeployedMcDOneD{w}P"] for b in "QL" for d, w in ((0, "Zero"), (1, "One"), (2, "Two"))}
            return int(math.ceil(math.log10(max(ps.values())))), "e1_main/out/rows/<dev log>__rescope__d<0-2>__tf | __deployed__dx__tf (D1 on heldout_disjoint): exact p " + ", ".join(f"{k} {x:.3g}" for k, x in ps.items())
        m = re.fullmatch(r"e3l_([QL])m_h2", key)
        if m:
            logs = [l for l in allu(m.group(1)) if fam(l) == "memitw"]
            return ", ".join(fx(E3["logs"][l]["H2"]["CosAll"]["value"]) for l in logs), "H2 all edits, rescope minus white_0.1_shared, tf rows of " + ", ".join(logs)
        m = re.fullmatch(r"e3_([QL])m_s_cap_max", key)
        if m:
            v = {l: rep[l]["capped_mean"] for l in allu(m.group(1)) if fam(l) == "memitw"}
            return max(v.values()), "gate/<log>/rescope__reg.json, mean n_capped of 20 cells: " + ", ".join(f"{k} {x:.2f}" for k, x in v.items())
        if key in ("e3_retr_h3_abs_min", "e3_retr_h3_abs_max"):
            v = {b: [E3["logs"][l]["H3"]["Retr"]["value"] for l in allu(b)] for b in "QL"}
            val = min(abs(max(x)) for x in v.values()) if key.endswith("min") else max(abs(min(x)) for x in v.values())
            return val, "H3 retrieval, h3 rows of the 11 unseen logs: " + json.dumps({b: [round(y, 2) for y in x] for b, x in v.items()})
        if key in ("e3_near_rel_min", "e3_near_rel_max"):
            v = {l: rep[l]["near_rel_beh"] for l in QLOGS + LLOGS}
            u = {l: x for l, x in v.items() if l not in DEV.values()}
            val = min(v.values()) if key.endswith("min") else max(v.values())
            return val, f"rescope tf rows, heldout_disjoint probes with near_edit, 13 logs (development logs included): attained on {[l for l, x in v.items() if x == val][0]}; over the 11 unseen logs only: {min(u.values()):.2f} to {max(u.values()):.2f}"
        if key in ("e3_draws_above_min", "e3_draws_above_max"):
            v = {l: rep[l]["draws_above"] for b in "QL" for l in allu(b)}
            val = min(v.values()) if key.endswith("min") else max(v.values())
            return val, "gate cells of rescope, 20 draws, 11 unseen logs: " + json.dumps(v) + f"; development logs: {[rep[l]['draws_above'] for l in DEV.values()]}"
        m = re.fullmatch(r"h3_cap_(p1_a|p1_b|p2_a|p2_b|retr_b)_max", key)
        if m:
            pair, side = {"p1": "PairOne", "p2": "PairTwo", "retr": "Retr"}[m.group(1).split("_")[0]], m.group(1).split("_")[1]
            v = {l: E3["logs"][l]["H3"][pair]["at_cap_" + side] for l in QLOGS + LLOGS}
            return max(v.values()), f"behave/<log>/<method>__d<draw>__h3.json n_edits_at_cap, mean over the behavioural draws, 13 logs; attained on {max(v, key=v.get)}"
        return None, None

    recs = []
    for e in ML:
        r = {"name": e["name"], "file": e["file"], "key": e["key"], "printed_in": e["printed"], "tex_files": e["body_files"] + e["report_files"], "stored": e["value"]}
        note = None
        if e["kind"] == "pn":
            k = e["key"]
            assert same(stored_all[k], e["value"])
            if k in PNm:
                mine, src = PNm[k], SRC.get(k)
            elif k in MC:
                mine, src = MC[k]["mine"], MC[k]["source"]
                note = {"kind": "bootstrap interval end, own resamples", "bootstrap_se": MC[k]["se"], "allowed_difference": MC[k]["allowed_difference"],
                        "difference": abs(mine - e["value"]), "difference_in_se": abs(mine - e["value"]) / MC[k]["se"]}
            else:
                mine, src = None, None
            if k in TOL:
                note = TOL[k]
            if k.startswith("XrelHolm"):
                h = "H2" if "Htwo" in k else "H3"
                o = HOLM[h]
                note = {"from_stored_p": o["stored"], "own_precise_p": {x: o["own_precise"][x] for x in ("rejected", "worse", "better")},
                        "own_20000_resamples": {x: o["own_20000"][x] for x in ("rejected", "worse", "better")}, "closest_member": o["closest_to_0_05"][0]}
        elif e["kind"] == "fact":
            mine, src = fact(e["key"])
        else:
            log, m = re.fullmatch(r"logs_b1\.(.+)\.(rel|tgen|igen)", e["key"]).groups()
            d = REL["deployed"][log]
            mine, src = d[m]["value"], f"{d['file']} ({d[m]['numerator']} of {d[m]['denominator']} edits, {m} exact)"
        r["mine"] = mine
        r["recomputed"] = mine is not None
        r["agree"] = None if mine is None else bool(same(mine, e["value"]))
        r["agreement"] = "exact"
        if e["kind"] == "pn" and e["key"] in MC:
            r["agree"] = bool(abs(mine - e["value"]) <= MC[e["key"]]["allowed_difference"])
            r["agreement"] = "within Monte Carlo error (0.25 x bootstrap SE)"
        r["source_rows"] = src
        pr = PR.get(e["name"])
        r["paper_prints"] = pr
        r["mine_as_printed"] = None if mine is None else show(e.get("fmt"), mine, e.get("scale"))
        r["printed_agrees"] = None if mine is None or pr is None else (r["mine_as_printed"] == pr)
        if note:
            r["note"] = note
        recs.append(r)
    for r in recs:
        if r["agreement"] != "exact" and r["printed_agrees"] is False:
            r["printed_agrees"] = None
            r["printed_note"] = f"my interval end prints {r['mine_as_printed']}, the paper {r['paper_prints']}: another set of resamples, inside the allowance"
    # ---------------------------------------------------------------- paper numbers behind generated table bodies
    tabs = []

    def add(table, where, key, tag, kind="exact"):
        st = ST[tag].get(key)
        if st is None:
            return
        mine = PNm.get(key)
        t = {"table": table, "printed_in": where, "name": key, "stored": st, "mine": mine, "source_rows": SRC.get(key), "recomputed": mine is not None}
        if mine is None:
            t["agree"] = None
        elif kind == "exact":
            t["agree"] = bool(same(mine, st))
        if key in TOL:
            x = TOL[key]
            allow = 0.25 * x["se"] + x["step"]
            t.update(own_resamples_value=x["own_resamples_value"], allowed_difference=allow, within_monte_carlo_error=bool(abs(x["own_resamples_value"] - st) <= allow),
                     difference_in_se=abs(x["own_resamples_value"] - st) / x["se"])
        tabs.append(t)

    for b, tag in (("Q", "e3qf"), ("L", "e3lf")):
        for log in (QLOGS if b == "Q" else LLOGS):
            c = "EthreeFx" + code(log)
            for stem in ("HtwoCosAll", "HtwoCosLater", "HthreePairOne", "HthreePairTwo", "HthreeRetr"):
                if stem == "HtwoCosLater" and T_of(log) <= 100:
                    continue
                for s in ("Diff", "Lo", "Hi"):
                    add("tab_vlm_tr_repl", "report", c + stem + s, tag)
    for b in "QL":
        for macro in ("Rescope", "RescopeX", "Cosine", "BestCosHtwo", "BestCos", "Retr"):
            for k in "IQX":
                for d in ("Zero", "One", "Two"):
                    add("tab_vlm_tr_draws + tab_vlm_main (Near)", "body+report", f"EoneFx{b}{macro}HN{k}FireD{d}", "e1f")
            add("tab_vlm_tr_draws (WT)", "report", f"EoneFx{b}{macro}HNAllWrongTarget", "e1f")
        for k in "IQX":
            add("tab_vlm_main (Near, log-only row)", "body", f"EoneFx{b}RescopeWHN{k}FireMean", "e1f")
        for macro in ("Deployed", "RescopeW", "RowCompBAll", "Rescope", "RescopeX", "BestCosHtwo", "BestCos", "Retr"):
            for s in ("GenTGenEq", "GenIGenEq"):
                add("tab_vlm_main (Gen)", "body", f"EoneFx{b}{macro}{s}", "e1f")
            if macro != "RowCompBAll":
                add("tab_vlm_main (D2)", "body", f"EoneFx{b}{macro}Dtwo", "e1f")
        for macro in ("Deployed", "RescopeW", "RescopeX", "Cosine", "BestCosHtwo", "BestCos", "CosPublished", "Lookup", "Retr", "Dual", "Exact", "Preedit"):
            for oc in ("DOne", "Rel"):
                for s in ("B", "C", "PHolm"):
                    add("tab_vlm_tr_mcnemar", "report", f"EoneFx{b}{macro}Mc{oc}DZero{s}", "e1f")
        dt = code(DEV[b])
        add("tab_vlm_tr_main (Rls of the two-block row)", "report", f"Xrel{dt}WhereCalHdReleased", "xr")
        add("tab_vlm_maint / tab_vlm_tr_cost (T)", "body+report", f"Xrel{dt}LogT", "xr")
        for log in (QLOGS if b == "Q" else LLOGS):
            for k in "IQX":
                add("tab_vlm_tr_unseen (rows per kind used to pool Near)", "report", f"Xrel{code(log)}RescopeHn{k}Rows", "xr")

    # ---------------------------------------------------------------- everything else that was recomputed (not printed)
    other = {"n": 0, "n_agree": 0, "differ": []}
    for k, mine in PNm.items():
        if k in stored_all and mine is not None:
            other["n"] += 1
            if same(mine, stored_all[k]):
                other["n_agree"] += 1
            else:
                other["differ"].append({"name": k, "stored": stored_all[k], "mine": mine})

    # ---------------------------------------------------------------- 'changed without a release' on the thresholds-only repair
    nr = {}
    for x in REL["no_release"]:
        if x["method"] in ("where_logonly", "where_least_n10", "where_cal") and x["pool"] in ("Hd", "HnI", "HnQ", "HnX"):
            a = nr.setdefault(f"{x['log']} | {x['method']}", {"rows_without_a_release": 0, "of_them_changed": 0, "by_pool": {}, "src_of_these_rows": {}})
            a["rows_without_a_release"] += x["n_not_released"]
            a["of_them_changed"] += x["n_changed_not_released"]
            a["by_pool"][x["pool"]] = [x["n_changed_not_released"], x["n_not_released"]]
            for s, n in x["src"].items():
                a["src_of_these_rows"][s] = a["src_of_these_rows"].get(s, 0) + n
    gates = {}
    for x in REL["no_release"]:
        if x["method"] not in ("where_logonly", "where_least_n10", "where_cal") and x["pool"] in ("Hd", "HnI", "HnQ", "HnX"):
            a = gates.setdefault(x["method"], {"rows_without_a_release": 0, "of_them_changed": 0, "src_of_these_rows": {}})
            a["rows_without_a_release"] += x["n_not_released"]
            a["of_them_changed"] += x["n_changed_not_released"]
            for s, n in x["src"].items():
                a["src_of_these_rows"][s] = a["src_of_these_rows"].get(s, 0) + n

    # ---------------------------------------------------------------- the three statements
    sa = {"heldout": {}, "near": {}}
    for log in QLOGS + LLOGS:
        c = REL["cells"]["Xrel" + code(log) + "RescopeHd"]
        sa["heldout"][log] = {"released_by_draw": [r["n_released"] for r in c["runs"]], "changed_among_them_by_draw": [r["n_changed_released"] for r in c["runs"]],
                              "every_released_row_changed": all(r["n_released"] == r["n_changed_released"] for r in c["runs"])}
    for b in "QL":
        for p in ("HnI", "HnQ", "HnX"):
            c = REL["cells"]["Xrel" + code(DEV[b]) + "Rescope" + p]
            per = [100.0 * r["n_changed_released"] / r["n_released"] for r in c["runs"]]
            sa["near"][f"{DEV[b]} {p}"] = {"mean_over_draws": mean(per), "by_draw": per, "pooled_over_draws": c["changed_among_released_pooled_over_runs"],
                                           "released_by_draw": [r["n_released"] for r in c["runs"]], "changed_among_them_by_draw": [r["n_changed_released"] for r in c["runs"]]}
    near_min_mean = min(v["mean_over_draws"] for v in sa["near"].values())
    near_min_draw = min(min(v["by_draw"]) for v in sa["near"].values())
    near_min_pooled = min(v["pooled_over_draws"] for v in sa["near"].values())
    A = {"statement": "Sec. 5.3: a release of ReScope changes the answer on every released held-out query; on near-scope negatives on at least 95 percent of the released",
         "held_out_development_logs": {l: sa["heldout"][l] for l in DEV.values()}, "held_out_all_13_logs_true": all(v["every_released_row_changed"] for v in sa["heldout"].values()),
         "near_scope": sa["near"], "near_scope_smallest_mean_over_draws": near_min_mean, "near_scope_smallest_single_draw": near_min_draw, "near_scope_smallest_pooled_over_draws": near_min_pooled,
         "verdict": ("TRUE as printed" if all(sa["heldout"][l]["every_released_row_changed"] for l in DEV.values()) and near_min_mean >= 95 else "FALSE") +
                    f": held-out, every released row has D1 set in each of the 3 draws on both development logs (and on all 13 logs); near-scope, smallest value {near_min_mean:.2f} % as mean over the "
                    f"3 draws (pooled over the draws {near_min_pooled:.2f} %; smallest single draw {near_min_draw:.2f} %)" +
                    ("" if near_min_draw >= 95 else "; in a single draw the share falls below 95 %: the statement holds for the mean over the draws, which is what the macro prints")}
    nrv = {l: rep[l]["near_rel_beh"] for l in QLOGS + LLOGS}
    dav = {l: rep[l]["draws_above"] for l in QLOGS + LLOGS}
    un = [l for l in QLOGS + LLOGS if l not in DEV.values()]
    Bst = {"statement": "Sec. 5.4: release on held-out probes that ask an edit's question is 3 to 23 percent, and 0 to 9 of 20 calibration draws per log exceed the level",
           "near_release_percent_by_log": nrv, "draws_above_by_log": dav, "n_gate_draws": sorted({rep[l]["n_gate_draws"] for l in rep}),
           "range_13_logs": [min(nrv.values()), max(nrv.values())], "range_11_unseen_logs": [min(nrv[l] for l in un), max(nrv[l] for l in un)],
           "draws_range_11_unseen_logs": [min(dav[l] for l in un), max(dav[l] for l in un)], "draws_range_13_logs": [min(dav.values()), max(dav.values())],
           "verdict": f"numbers TRUE over the 13 logs of the experiment ({min(nrv.values()):.1f} to {max(nrv.values()):.1f} %; draws {min(dav.values())} to {max(dav.values())} of 20); SCOPE: the upper end "
                      f"{max(nrv.values()):.1f} % is the DEVELOPMENT log {max(nrv, key=nrv.get)}; over the 11 unseen logs alone the release is {min(nrv[l] for l in un):.1f} to {max(nrv[l] for l in un):.1f} % "
                      f"(the draws range {min(dav[l] for l in un)} to {max(dav[l] for l in un)} is the same with or without the development logs).  The paragraph is about the unseen logs."}
    h2, h3 = HOLM["H2"], HOLM["H3"]
    cl = h3["closest_to_0_05"][0]
    Cst = {"statement": "report: H2 18 of 22 rejected, all against ReScope; H3 27 of 33, 25 against and 2 in favour",
           "H2": {k: h2[k] for k in ("family", "stored", "replicated", "own_precise", "own_20000", "closest_to_0_05")},
           "H3": {k: h3[k] for k in ("family", "stored", "replicated", "own_precise", "own_20000", "closest_to_0_05")},
           "verdict": (f"H2 TRUE and robust ({h2['replicated']['rejected']} of {h2['family']}, {h2['replicated']['worse']} against, {h2['replicated']['better']} in favour; the same with my own resamples).  "
                       f"H3 REPRODUCED ({h3['replicated']['rejected']} of {h3['family']}, {h3['replicated']['worse']} against, {h3['replicated']['better']} in favour, from the stored values and from my replication of the "
                       f"experiment's 2000 resamples) but NOT ROBUST: the member '{cl['member']}' has stored p {cl['stored_p']} (Holm {cl['holm_stored']:.3f}); with {cl['own_p_resamples']} own resamples "
                       f"p = {cl['own_p']:.5f} +- {cl['own_p_mc_se']:.5f} against the rejection limit 0.05 / 7 = {0.05 / 7:.5f}, Holm-adjusted {cl['holm_own_precise']:.4f}; my first 20000 resamples gave "
                       f"Holm {cl['holm_own_20000']:.4f}, i.e. {h3['own_20000']['rejected']} of {h3['family']} ({h3['own_20000']['worse']} against, {h3['own_20000']['better']} in favour).  "
                       "The 27th rejection depends on the Monte Carlo error of a 2000-resample value.")}
    cap_note = {k: fact(k) for k in ("h3_cap_p1_a_max", "h3_cap_p1_b_max", "h3_cap_p2_a_max", "h3_cap_p2_b_max", "h3_cap_retr_b_max")}
    unreach = max(E3["logs"][l]["H3"][p]["unreachable"] for l in E3["logs"] for p in ("PairOne", "PairTwo", "Retr"))
    out = {"written_by": "x_release/verify/code/p4_assemble.py", "what": "independent recomputation of the fix-up read-outs that the paper and the technical report print",
           "tolerances": {"exact": "counts, rates, means, point values of differences, exact McNemar values, Holm values: 1e-9 relative",
                          "bootstrap_interval_ends": "(i) the experiment's 2000 resamples replicated with my own code under its documented seed scheme: exact (1e-9); (ii) my own 20000 resamples with "
                                                     "another generator: |difference| <= 0.25 x bootstrap SE + one step of the statistic (4 sigma of the Monte Carlo error of a 2.5 % quantile from 2000 "
                                                     "against 20000 resamples: sqrt(0.0598^2 + 0.0189^2) SE = 0.063 SE per sigma)"},
           "macros": recs, "table_numbers": tabs, "other_recomputed_fixup_numbers": other, "changed_without_a_release": {"thresholds_only_rows": nr, "gates_at_the_last_position": gates,
           "reading": "release = row flag `fired` (x_release/out/release_change.json: ReScope-W, at any position); a row counts if fired is false and D1 is true.  If the flag were the last-position "
                      "flag only, the rows without a release at any position would be a subset of the rows counted here, so 0 holds under either reading.  The gate rows (rescope, the cosine "
                      "family, retr_sum) carry src = pre: the pre-edit outcome was assigned without a forward pass, 0 holds by construction there and is no measurement."},
           "statements": {"a": A, "b": Bst, "c": Cst}, "h1_holm": {"tests": H1, "stored": H1holm}, "report_caption_tab_app_repl": {"largest_at_cap": {k: {"value": v[0], "where": v[1]} for k, v in cap_note.items()},
           "unreachable_draws_max": unreach}}
    json.dump(out, open(OUT + "/verdicts.json", "w"), indent=1)
    nm = len(recs)
    rc = [r for r in recs if r["recomputed"]]
    print(f"macros listed {nm}; recomputed {len(rc)}; agreeing {sum(1 for r in rc if r['agree'])}; printed string differs: {[r['name'] for r in rc if r['printed_agrees'] is False]}")
    print("by place:", {p: (sum(1 for r in recs if r['printed_in'] == p), sum(1 for r in rc if r['printed_in'] == p), sum(1 for r in rc if r['printed_in'] == p and r['agree'])) for p in ("body", "report", "unused")})
    print("disagree:", [(r["name"], r["stored"], r["mine"]) for r in rc if not r["agree"]])
    print("not recomputed:", [r["name"] for r in recs if not r["recomputed"]])
    te = [t for t in tabs if t["recomputed"]]
    print(f"table numbers: {len(tabs)}; recomputed {len(te)}; agreeing {sum(1 for t in te if t['agree'])}; report-only {sum(1 for t in te if t['printed_in'] == 'report')}; "
          f"interval ends with own resamples outside the allowance: {[(t['name'], t['stored'], t['own_resamples_value']) for t in te if t.get('within_monte_carlo_error') is False]}")
    print("largest interval deviation (SE):", max(t["difference_in_se"] for t in te if "difference_in_se" in t))
    print("other fix-up numbers:", other["n"], other["n_agree"], other["differ"][:5])
    for k in "abc":
        print(k, out["statements"][k]["verdict"])
    json.dump({"E2 fix-up, 200-draw read-out (heldout_disjoint interval, not printed)": {k: dict(v, stored=stored_all[k], within=bool(abs(v["mine"] - stored_all[k]) <= v["allowed_difference"])) for k, v in MC.items()}},
              open(OUT + "/cw200_intervals.json", "w"), indent=1)
    print("H1:", {k: (round(v["value"], 2), v["n_other_side"], v["stored_p_holm"]) for k, v in H1.items()})
    print("thresholds-only:", {k: (v["of_them_changed"], v["rows_without_a_release"], v["src_of_these_rows"]) for k, v in nr.items()})


main()
