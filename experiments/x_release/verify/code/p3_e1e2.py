#!/usr/bin/env python3
"""Part 3: the fix-ups of E1 and E2 from per-row files (and, for the 200-draw read-out, from the stored per-draw rates).
E1: exact McNemar tests (all 300: ReScope against 25 rows x 2 logs x 3 draws x 2 outcomes, Holm over the 150 tests of an outcome),
    near-scope release per kind and behavioural draw, generated answers (Gen), D2, wrong-target rate, number of H2 comparisons.
E2: Rel kept / gained / lost of the H1 row, release after the least repair from the deployed start, means of the 200-draw read-out.
-> out/mine_e1e2.json"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *  # noqa: E402,F401,F403

E1R, E2R = FINAL + "/e1_main/out/rows", FINAL + "/e2_curves/out/rows"
DW = {0: "Zero", 1: "One", 2: "Two"}
# macro name of the row -> (row name of E1, has calibration draws); read from e1_main/out/fixup/mcnemar.json (field macro / row)
E1ROWS = {"Deployed": ("deployed", False), "Preedit": ("preedit", False), "RescopeShared": ("rescope_shared", True), "RescopeNull": ("rescope_null", True),
          "RescopeX": ("rescope_x", True), "RescopeW": ("where_logonly", False), "RescopeLin": ("rescope_lin", True), "BestCos": ("white_1_shared", True),
          "BestCosHtwo": ("white_0.1_shared", True), "BestCosX": ("white_1_x", True), "BestCosHtwoX": ("white_0.1_x", True), "Cosine": ("cosine", True),
          "CosineShared": ("cosine_shared", True), "Retr": ("retr_sum", True), "Lookup": ("lookup", False), "CosDefault": ("cosine_default", False),
          "CosPublished": ("cosine_published", False), "Zscore": ("zscore_default", False), "Exact": ("exact", False), "Ball": ("ball", True), "Router": ("router", True),
          "Dual": ("dual", False)}
E2ROWS = {"RowCompBAll": "d__{b}__COMP_B__nALL__s0", "RowReeditBAll": "d__{b}__REEDIT_B__nALL__s0", "WCal": "cw__{b}__where_cal__d{d}"}


def path_of(b, macro, d, part="tf"):
    log = DEV[b]
    if macro == "Rescope":
        return f"{E1R}/{log}__rescope__d{d}__{part}.jsonl.gz"
    if macro in E1ROWS:
        row, cal = E1ROWS[macro]
        return f"{E1R}/{log}__{row}__d{d if cal else 'x'}__{part}.jsonl.gz"
    return f"{E2R}/{E2ROWS[macro].format(b=b, d=d)}.jsonl.gz"


def flags(path, outcome):
    if outcome == "DOne":
        return {r["csid"]: bool(r["D1"]) for r in rows(path) if is_hd(r)}
    return {r["edit"]: bool(r["exact"]) for r in rows(path) if is_edit(r, "rel")}


def main():
    out = {"mcnemar": {}, "pn": {}, "src": {}}
    PN, SRC = out["pn"], out["src"]
    # ---------------------------------------------------------------- E1: McNemar
    tests = {"DOne": {}, "Rel": {}}
    for b in "QL":
        for macro in list(E1ROWS) + list(E2ROWS):
            for d in (0, 1, 2):
                for oc in ("DOne", "Rel"):
                    fa, fo = flags(path_of(b, "Rescope", d), oc), flags(path_of(b, macro, d), oc)
                    assert set(fa) == set(fo) and fa, (b, macro, d, oc, len(fa), len(fo))
                    bb_ = sum(1 for k in fa if fa[k] and not fo[k])
                    cc = sum(1 for k in fa if fo[k] and not fa[k])
                    tests[oc][(b, macro, d)] = {"b": bb_, "c": cc, "n": len(fa), "p": mcnemar_exact(bb_, cc), "files": f"{rel(path_of(b, 'Rescope', d))} | {rel(path_of(b, macro, d))}"}
    for oc, tt in tests.items():
        adj = holm({k: v["p"] for k, v in tt.items()})
        assert len(tt) == 150
        for (b, macro, d), v in tt.items():
            n = f"EoneFx{b}{macro}Mc{oc}D{DW[d]}"
            PN[n + "B"], PN[n + "C"], PN[n + "P"], PN[n + "PHolm"] = v["b"], v["c"], v["p"], adj[(b, macro, d)]
            for s in "BCP":
                SRC[n + s] = v["files"]
            SRC[n + "PHolm"] = v["files"] + " (Holm over my 150 tests of the outcome)"
            out["mcnemar"][n] = dict(v, p_holm=adj[(b, macro, d)])
    # ---------------------------------------------------------------- E1: near-scope release per kind and draw; Gen; D2; wrong target
    for b in "QL":
        log = DEV[b]
        for macro in ("Rescope", "RescopeX", "Cosine", "BestCos", "BestCosHtwo", "Retr", "RescopeShared", "Ball", "Router", "CosineShared"):
            for k in "IQX":
                per = []
                for d in (0, 1, 2):
                    p = path_of(b, macro, d)
                    v = pct(bool(r["fired"]) for r in rows(p) if is_hn(r, "HN-" + k))
                    PN[f"EoneFx{b}{macro}HN{k}FireD{DW[d]}"] = v
                    SRC[f"EoneFx{b}{macro}HN{k}FireD{DW[d]}"] = rel(p)
                    v1 = pct(bool(r["D1"]) for r in rows(p) if is_hn(r, "HN-" + k))
                    PN[f"EoneFx{b}{macro}HN{k}DOneD{DW[d]}"] = v1
                    SRC[f"EoneFx{b}{macro}HN{k}DOneD{DW[d]}"] = rel(p)
                    per.append(v)
                PN[f"EoneFx{b}{macro}HN{k}FireMean"], PN[f"EoneFx{b}{macro}HN{k}FireSpread"] = mean(per), max(per) - min(per)
                SRC[f"EoneFx{b}{macro}HN{k}FireMean"] = SRC[f"EoneFx{b}{macro}HN{k}FireSpread"] = rel(path_of(b, macro, 0)) + " and draws 1, 2"
        for macro in ("RescopeW", "Deployed"):
            for k in "IQX":
                p = path_of(b, macro, 0)
                PN[f"EoneFx{b}{macro}HN{k}FireMean"] = pct(bool(r["fired"]) for r in rows(p) if is_hn(r, "HN-" + k)) if macro != "Deployed" else None
                SRC[f"EoneFx{b}{macro}HN{k}FireMean"] = rel(p)
        gens = {m: path_of(b, m, 0, "gen") for m in ("Deployed", "RescopeW", "Rescope", "RescopeX", "BestCosHtwo", "BestCos", "Retr", "Preedit", "Dual")}
        gens["RowCompBAll"] = f"{E2R}/gen__{b}__d__COMP_B__nALL__s0__gen.jsonl.gz"
        for macro, p in gens.items():
            if not os.path.exists(p):
                continue
            rs = rows(p)
            for m, mw in (("rel", "Rel"), ("tgen", "TGen"), ("igen", "IGen")):
                v = [bool(r["eq"]) for r in rs if r["kind"] == "edit" and r["metric"] == m and "eq" in r]
                if v:
                    PN[f"EoneFx{b}{macro}Gen{mw}Eq"] = pct(v)
                    SRC[f"EoneFx{b}{macro}Gen{mw}Eq"] = rel(p) + f" ({len(v)} edits)"
            d2 = [bool(r["D2"]) for r in rs if r["kind"] == "probe" and r.get("disjoint") and r.get("pre_correct") and r.get("D2") is not None]
            if d2 and macro != "RowCompBAll":
                PN[f"EoneFx{b}{macro}Dtwo"], PN[f"EoneFx{b}{macro}DtwoNum"], PN[f"EoneFx{b}{macro}DtwoDen"] = pct(d2), sum(d2), len(d2)
                for s in ("", "Num", "Den"):
                    SRC[f"EoneFx{b}{macro}Dtwo{s}"] = rel(p) + " (pre-correct heldout_disjoint rows)"
            hn = [r for r in rs if r["kind"] == "hn" and "wrong_target" in r]
            if hn and macro != "RowCompBAll":
                PN[f"EoneFx{b}{macro}HNAllWrongTarget"] = pct(bool(r["wrong_target"]) for r in hn)
                PN[f"EoneFx{b}{macro}HNAllWrongTargetDen"] = len(hn)
                SRC[f"EoneFx{b}{macro}HNAllWrongTarget"] = SRC[f"EoneFx{b}{macro}HNAllWrongTargetDen"] = rel(p) + " (near-scope rows of the generation part)"
                for k in "IQX":
                    sub = [r for r in hn if r["hn_type"] == "HN-" + k]
                    PN[f"EoneFx{b}{macro}HN{k}WrongTarget"] = pct(bool(r["wrong_target"]) for r in sub)
                    SRC[f"EoneFx{b}{macro}HN{k}WrongTarget"] = rel(p)
        forget()
    ep = jload(FINAL + "/e1_main/out/endpoints.json")
    PN["EoneFxHtwoNComparisons"] = sum(len(ep[l]["H2"]) for l in DEV.values())
    SRC["EoneFxHtwoNComparisons"] = "e1_main/out/endpoints.json: number of H2 entries of the two development logs (a count of comparisons, no row file)"
    # ---------------------------------------------------------------- E2: H1 row
    for b in "QL":
        kg = {"Kept": [], "Gained": [], "Lost": []}
        for s in range(3):
            p = f"{E2R}/b__{b}__where_least__logonly__n10__s{s}.jsonl.gz"
            e = [r for r in rows(p) if is_edit(r, "rel")]
            assert len(e) == 100
            kg["Kept"].append(sum(1 for r in e if r["exact"] and r["exact_dep"]))
            kg["Gained"].append(sum(1 for r in e if r["exact"] and not r["exact_dep"]))
            kg["Lost"].append(sum(1 for r in e if not r["exact"] and r["exact_dep"]))
        for k, v in kg.items():
            PN[f"EtwoFx{b}HoneRel{k}"] = mean(v)
            SRC[f"EtwoFx{b}HoneRel{k}"] = f"e2_curves/out/rows/b__{b}__where_least__logonly__n10__s<0-2>.jsonl.gz (Rel rows: exact against exact_dep); per stream {v}"
        fr = []
        for s in range(3):
            p = f"{E2R}/b__{b}__where_least__deployed__n10__s{s}.jsonl.gz"
            fr.append(pct(bool(r["fired"]) for r in rows(p) if is_hd(r)))
        PN[f"EtwoFx{b}WDepTenFireHeld"] = mean(fr)
        SRC[f"EtwoFx{b}WDepTenFireHeld"] = f"e2_curves/out/rows/b__{b}__where_least__deployed__n10__s<0-2>.jsonl.gz (heldout_disjoint rows, fired); per stream {fr}"
        cw = jload(FINAL + "/e2_curves/out/fixup/cw_check_200.json")["logs"][DEV[b]]
        PN[f"EtwoFx{b}WCalFreshTwoHundredMean"] = mean([d["fired_fresh_repair"] for d in cw["draws"]])
        PN[f"EtwoFx{b}WCalHeldTwoHundredMean"] = mean([d["fired_heldout_disjoint"] for d in cw["draws"]])
        PN[f"EtwoFx{b}WCalNDraws"] = len(cw["draws"])
        for k in ("WCalFreshTwoHundredMean", "WCalHeldTwoHundredMean", "WCalNDraws"):
            SRC[f"EtwoFx{b}{k}"] = "e2_curves/out/fixup/cw_check_200.json, field draws (200 stored per-draw rates; the rates themselves are NOT recomputed here)"
        forget()
    json.dump(out, open(OUT + "/mine_e1e2.json", "w"), indent=1)
    # ---- report
    st = jload(FINAL + "/e1_main/out/fixup/paper_numbers_fixup.json")
    st.update(jload(FINAL + "/e2_curves/out/fixup/paper_numbers_fixup.json"))
    n = bad = miss = 0
    for k, v in PN.items():
        if v is None:
            continue
        if k not in st:
            miss += 1
            continue
        n += 1
        tol = 1e-6 if k.endswith("PHolm") or k.endswith("P") else 1e-9
        if not (close(v, st[k], tol) or (isinstance(v, float) and st[k] and abs(v - st[k]) <= 1e-9 * abs(st[k]))):
            bad += 1
            print("  DIFF", k, v, st[k])
    print(f"E1 / E2 fix-up numbers compared: {n}; differing: {bad}; names of mine without a stored number: {miss}")


main()
