#!/usr/bin/env python3
"""Part 1: release against change (x_release item 5) and the deployed logs' metrics (item 6), from the per-row files.
-> out/mine_release.json  {paper-number name: my value}, per-run counts, and the check 'changed without a release'."""
import collections
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *  # noqa: E402,F401,F403

MCODE = {"where_logonly": "Where", "rescope": "Rescope", "white_1_shared": "WhiteOneShared", "white_0.1_shared": "WhiteTenthShared", "retr_sum": "RetrSum",
         "where_least_n10": "WhereLeastTen", "where_cal": "WhereCal"}
POOLS = (("Hd", lambda r: is_hd(r)), ("HdFar", lambda r: is_hd(r) and r.get("near_edit") is None), ("HdNear", lambda r: is_hd(r) and r.get("near_edit") is not None),
         ("HnI", lambda r: is_hn(r, "HN-I")), ("HnQ", lambda r: is_hn(r, "HN-Q")), ("HnX", lambda r: is_hn(r, "HN-X")))
E2R = FINAL + "/e2_curves/out/rows"


def runs_of(log, method):
    """[(tag, path)] of the runs of a (log, method)"""
    b = bb(log)
    if method == "where_logonly":
        return [("one run", e3_path(log, method, None))]
    if method == "where_least_n10":
        return [(f"stream {i}", f"{E2R}/b__{b}__where_least__logonly__n10__s{i}.jsonl.gz") for i in range(3)]
    if method == "where_cal":
        return [(f"draw {i}", f"{E2R}/cw__{b}__where_cal__d{i}.jsonl.gz") for i in range(3)]
    return [(f"draw {d}", e3_path(log, method, d)) for d in bdraws(log)]


def cell(runs, sel):
    per = []
    for tag, path in runs:
        rs = [r for r in rows(path) if sel(r)]
        if not rs:
            continue
        f = [bool(r["fired"]) for r in rs]
        d = [bool(r["D1"]) for r in rs]
        bad = [r["csid"] for r, x, y in zip(rs, f, d) if y and not x]
        per.append({"run": tag, "file": rel(path), "n": len(rs), "n_released": sum(f), "n_changed": sum(d), "n_changed_released": sum(1 for x, y in zip(f, d) if x and y),
                    "n_changed_not_released": len(bad), "rows_changed_not_released": bad, "n_y0_invalid": sum(1 for r in rs if r.get("y0_valid") is False),
                    "n_not_released_with_a_candidate": sum(1 for r, x in zip(rs, f) if not x and (r.get("top_last", -1) not in (-1, None) or r.get("n_cand_edits") not in (None, 0))),
                    "src_of_not_released": dict(collections.Counter(str(r.get("src")) for r, x in zip(rs, f) if not x))})
    return per


def main():
    PN, CELLS = {}, {}
    norel = []
    for log in QLOGS + LLOGS:
        methods = ["where_logonly", "rescope", "white_1_shared", "white_0.1_shared", "retr_sum"] + (["where_least_n10", "where_cal"] if log in DEV.values() else [])
        for method in methods:
            runs = runs_of(log, method)
            for pw, sel in POOLS:
                per = cell(runs, sel)
                nm = "Xrel" + code(log) + MCODE[method] + pw
                if not per:
                    continue
                rl = [100.0 * p["n_released"] / p["n"] for p in per]
                ch = [100.0 * p["n_changed"] / p["n"] for p in per]
                cr = [100.0 * p["n_changed_released"] / p["n_released"] for p in per if p["n_released"]]
                PN[nm + "Rows"] = per[0]["n"]
                PN[nm + "Released"] = mean(rl)
                PN[nm + "Changed"] = mean(ch)
                PN[nm + "ChangedReleased"] = mean(cr)
                PN[nm + "ChangedNoRelease"] = sum(p["n_changed_not_released"] for p in per)
                tot_r, tot_cr = sum(p["n_released"] for p in per), sum(p["n_changed_released"] for p in per)
                CELLS[nm] = {"log": log, "method": method, "pool": pw, "runs": per, "changed_among_released_pooled_over_runs": 100.0 * tot_cr / tot_r if tot_r else None,
                             "ratio_of_the_means": 100.0 * mean(ch) / mean(rl) if mean(rl) else None, "same_n_every_run": len({p["n"] for p in per}) == 1}
                norel.append({"log": log, "method": method, "pool": pw, "n_runs": len(per), "n_not_released": sum(p["n"] - p["n_released"] for p in per),
                              "n_changed_not_released": sum(p["n_changed_not_released"] for p in per),
                              "n_not_released_with_a_candidate": sum(p["n_not_released_with_a_candidate"] for p in per),
                              "src": dict(sum((collections.Counter(p["src_of_not_released"]) for p in per), collections.Counter()))})
        forget()
    # ---- deployed logs (item 6): 13 logs of E3 and 6 prefix logs
    LOGS = {}
    for log in QLOGS + LLOGS:
        LOGS[log] = e3_path(log, "deployed", None)
    for b in "QL":
        for t in (10, 20, 50):
            lg = f"{DEV[b]}-p{t}"
            LOGS[lg] = f"{FINAL}/e2_prefix/out/rows/{lg}__deployed.jsonl.gz"
    DEP = {}
    for log, path in LOGS.items():
        rs = rows(path, keep=False)
        o = {"file": rel(path)}
        for m in ("rel", "tgen", "igen"):
            v = [bool(r["exact"]) for r in rs if is_edit(r, m)]
            o[m] = {"value": pct(v), "numerator": sum(v), "denominator": len(v)}
        v = [bool(r["D1"]) for r in rs if is_hd(r)]
        o["d1"] = {"value": pct(v), "numerator": sum(v), "denominator": len(v)}
        DEP[log] = o
        if "-p" not in log:
            c = "Xrel" + code(log) + "Log"
            PN[c + "DepRel"], PN[c + "DepTGen"], PN[c + "DepIGen"], PN[c + "DepDone"], PN[c + "HdRows"], PN[c + "T"] = \
                o["rel"]["value"], o["tgen"]["value"], o["igen"]["value"], o["d1"]["value"], o["d1"]["denominator"], o["rel"]["denominator"]
    json.dump({"paper_numbers": PN, "cells": CELLS, "no_release": norel, "deployed": DEP}, open(OUT + "/mine_release.json", "w"), indent=1)
    # ---- report
    st = jload(FINAL + "/x_release/out/paper_numbers_fixup.json")
    bad = [(k, v, st.get(k)) for k, v in PN.items() if k not in st or not close(v, st[k])]
    rc = [k for k in st if any(k.endswith(s) for s in ("Released", "Changed", "ChangedReleased", "ChangedNoRelease", "Rows")) and "Holm" not in k]
    print(f"release/change + log numbers recomputed: {len(PN)}; differing from x_release/out/paper_numbers_fixup.json: {len(bad)}")
    for x in bad[:40]:
        print("  DIFF", x)
    print("stored names of the release/change kind that I did not produce:", [k for k in rc if k not in PN][:20])
    tot = collections.Counter()
    for x in norel:
        tot[x["method"], "not released"] += x["n_not_released"] if x["pool"] in ("Hd", "HnI", "HnQ", "HnX") else 0
        tot[x["method"], "changed, not released"] += x["n_changed_not_released"] if x["pool"] in ("Hd", "HnI", "HnQ", "HnX") else 0
        tot[x["method"], "not released but a candidate edit / top_last"] += x["n_not_released_with_a_candidate"] if x["pool"] in ("Hd", "HnI", "HnQ", "HnX") else 0
    for k in sorted(tot):
        print("  ", k, tot[k])
    lb = {x["log"]: x for x in jload(FINAL + "/x_release/out/logs_b1.json")["logs"]}
    nb = 0
    for log, o in DEP.items():
        d = lb[log]["deployed"]
        for mk, sk in (("rel", "rel_exact"), ("tgen", "tgen_exact"), ("igen", "igen_exact"), ("d1", "D1_heldout_disjoint")):
            if (o[mk]["numerator"], o[mk]["denominator"]) != (d[sk]["numerator"], d[sk]["denominator"]) or not close(o[mk]["value"], d[sk]["value"]):
                nb += 1
                print("  LOG DIFF", log, mk, o[mk], d[sk])
    print(f"logs_b1.json: {len(DEP)} logs x 4 metrics compared, differing: {nb}")


main()
