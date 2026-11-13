"""x_release: release against change (decision E4 of the external review's triage), the Holm correction of the replication read-outs
on unseen logs, and the paper's table of logs.  ADDITIONS ONLY; CPU; no model run; every number from stored per-row files.

    python release_compute.py      (after e3_qwen/code/fixup_compute.py and e3_llava/code/fixup_compute.py: holm.json reads their endpoint files)
    python release_render.py       (out/table_fixup.txt, out/FIXUP.txt)

Writes x_release/out/: release_change.json (item 5), holm.json (item 3, Holm), logs_b1.json (item 6), paper_numbers_fixup.json,
paper_numbers_fixup_doc.json, checks.json.

Item 5.  For every log of E3 (the two development logs are among them) the rows where_logonly, rescope, the two selected gates of the
cosine family and retr_sum come from the E3 per-row files (out/rows/<log>__<method>__d<draw|x>__tf.jsonl.gz); on the development logs
two rows of E2 are added: the thresholds-only repair after the least repair of 10 reported complaints from the log-only thresholds
(part b, the H1 row: complaint streams 0-2) and ReScope-W with two-block thresholds (part c', calibration draws 0-2).  Per pool
(heldout_disjoint: all rows / rows that share no question with an edit / rows that share one; HN-I, HN-Q, HN-X) and per draw or stream:
rows with a release (row flag `fired`: an edit is released on the sequence), rows whose pre-edit answer is not reproduced (D1), D1 among
the rows with a release, and the rows WITHOUT a release whose answer changed (the certificate says there is none).
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/10_final/x_release/code")
from xlib import *  # noqa: E402,F401,F403
import xlib  # noqa: E402

OUT = XR + "/out"
E1R, E2R, PFXR = FINAL + "/e1_main/out/rows", FINAL + "/e2_curves/out/rows", FINAL + "/e2_prefix/out/rows"
POOLS = (("hd_all", "Hd", "heldout_disjoint, all valid rows"), ("hd_far", "HdFar", "heldout_disjoint rows that share no question with an edit of the log"),
         ("hd_near", "HdNear", "heldout_disjoint rows that ask the question of an edit of the log"),
         ("HN-I", "HnI", "near-scope negatives HN-I (image of an edit, other question)"), ("HN-Q", "HnQ", "near-scope negatives HN-Q (question of an edit, other image)"),
         ("HN-X", "HnX", "near-scope negatives HN-X (image of one edit, question of another)"))
PFX_T = {10: "Ten", 20: "Twenty", 50: "Fifty"}


def is_probe(r):
    return r["kind"] == "probe" and r.get("disjoint") and r.get("lp", "tf") == "tf" and not r.get("d2_row")


def in_pool(r, pool):
    if pool == "hd_all":
        return is_probe(r)
    if pool == "hd_far":
        return is_probe(r) and r.get("near_edit") is None
    if pool == "hd_near":
        return is_probe(r) and r.get("near_edit") is not None
    return r["kind"] == "hn" and r.get("part") != "gen" and r.get("lp", "tf") == "tf" and r["hn_type"] == pool


def one(rows, pool, tag):
    sel = [r for r in rows if in_pool(r, pool)]
    if any(r.get("fired") is None or r.get("D1") is None for r in sel):
        raise RuntimeError("a row without a release flag or without D1")
    f, d = np.array([bool(r["fired"]) for r in sel], dtype=bool), np.array([bool(r["D1"]) for r in sel], dtype=bool)
    bad = [{"run": tag, "csid": r["csid"], "tok_loc": r.get("tok_loc"), "n_tok": r.get("n_tok"), "first_mismatch": r.get("first_mismatch"), "lp_shift": r.get("lp_shift"),
            "y0_deg": r.get("y0_deg"), "src": r.get("src")} for r in sel if r["D1"] and not r["fired"]]
    src = {}
    for r in sel:
        if not r["fired"]:
            src[str(r.get("src"))] = src.get(str(r.get("src")), 0) + 1
    return {"run": tag, "n": len(sel), "not_released_by_source": src, "n_released": int(f.sum()), "n_changed": int(d.sum()), "n_changed_released": int((f & d).sum()), "n_not_released": int((~f).sum()),
            "n_changed_not_released": int((d & ~f).sum()), "rows_changed_not_released": bad}


def summarise(per):
    """Mean over the draws / streams of the per-run percentages; the single runs stay listed with numerator and denominator."""
    empty = [p["run"] for p in per if p["n"] == 0]                  # E2 c': the near-scope negatives were run at calibration draw 0 only
    per = [p for p in per if p["n"] > 0] or per[:1]
    n = per[0]["n"]
    out = {"n_rows": n, "n_runs": len(per), "runs": [p["run"] for p in per], "runs_without_rows_of_the_pool": empty if n else []}
    if n == 0:
        return out | {"released_percent": None, "changed_percent": None, "changed_among_released_percent": None, "changed_not_released_count": 0, "by_run": per}
    rel = [100.0 * p["n_released"] / p["n"] for p in per]
    chg = [100.0 * p["n_changed"] / p["n"] for p in per]
    car = [100.0 * p["n_changed_released"] / p["n_released"] if p["n_released"] else None for p in per]
    out.update(released_percent=mean(rel), released_by_run=rel, changed_percent=mean(chg), changed_by_run=chg, changed_among_released_percent=mean(car), changed_among_released_by_run=car,
               n_runs_with_a_release=sum(x is not None for x in car), changed_not_released_count=int(sum(p["n_changed_not_released"] for p in per)),
               changed_not_released_by_run=[p["n_changed_not_released"] for p in per], not_released_by_run=[p["n_not_released"] for p in per], by_run=per)
    return out


def sources(B, log):
    """[(method key, {run tag: rows}, source text)] of one log."""
    out = []
    for lab in ("where_logonly", "rescope", B.sel["cos_H3"], B.sel["cos_H2"], B.sel["retr_H2"]):
        by = B.by_draw(log, lab)
        if by is None:
            out.append((lab, None, f"{B.exp}/out/rows/{log}__{lab}__*__tf.jsonl.gz"))
            continue
        out.append((lab, {("one run" if d is None else f"draw {d}"): r for d, r in by.items()}, f"{B.exp}/out/rows/{log}__{lab}__d{'x' if None in by else '<draw>'}__tf.jsonl.gz"))
    if log == B.dev:
        for key, pat, tags, what in (("where_least_n10", f"b__{B.key}__where_least__logonly__n10__s{{}}", ("stream 0", "stream 1", "stream 2"), "complaint stream"),
                                     ("where_cal", f"cw__{B.key}__where_cal__d{{}}", ("draw 0", "draw 1", "draw 2"), "calibration draw")):
            by = {}
            for i, t in enumerate(tags):
                p = os.path.join(E2R, pat.format(i) + ".jsonl.gz")
                by[t] = fx.io.load_rows(p) if os.path.exists(p) else None
            out.append((key, None if any(v is None for v in by.values()) else by, f"e2_curves/out/rows/{pat.format('<' + what + '>')}.jsonl.gz"))
    return out


def release_change(PN, CK):
    RC, missing, n_bad = {}, [], 0
    eq = []
    for key in ("Q", "L"):
        B = Backbone(key)
        old = jload(B.out + "/paper_numbers.json")
        for log in B.logs:
            node = {"backbone": B.title, "family": family_of(log), "T": B.T(log), "development_log": log == B.dev, "methods": {}}
            for lab, by, src in sources(B, log):
                if by is None:
                    missing.append({"log": log, "method": lab, "source": src})
                    continue
                m = {"title": MTITLE[lab], "source": src, "pools": {}}
                for pool, pw, ptxt in POOLS:
                    s = summarise([one(rows, pool, tag) for tag, rows in by.items()])
                    m["pools"][pool] = s
                    if not s["n_rows"]:
                        continue
                    n_bad += s["changed_not_released_count"]
                    nm = "Xrel" + B.code(log) + MCODE[lab] + pw
                    what = f"{log}, {lab}, {ptxt} ({s['n_rows']} rows; mean over {s['n_runs']} run(s): {', '.join(s['runs'])}; {src})"
                    PN.put(nm + "Rows", s["n_rows"], what + ": number of rows")
                    PN.put(nm + "Released", s["released_percent"], what + ": rows on which an edit is released, percent")
                    PN.put(nm + "Changed", s["changed_percent"], what + ": rows whose pre-edit answer is not reproduced (D1), percent")
                    PN.put(nm + "ChangedReleased", s["changed_among_released_percent"], what + ": D1 among the rows with a release, percent (mean over the runs that release on at least one row)")
                    PN.put(nm + "ChangedNoRelease", s["changed_not_released_count"], what + ": NUMBER of rows without a release whose answer changed, summed over the runs (the certificate says 0)")
                node["methods"][lab] = m
            RC[log] = node
            # equality with numbers the experiment printed
            pre = "Ethree" + B.code(log)
            for lab, mine_key, name, nd in ((("rescope", "released_percent", pre + "RescopeHdFired", 2), ("rescope", "changed_percent", pre + "RescopeHdDone", 2),
                                             ("where_logonly", "changed_percent", pre + "WhereHdDone", 2), (B.sel["cos_H2"], "changed_percent", pre + MCODE[B.sel["cos_H2"]] + "HdDone", 2))
                                            if key == "Q" else
                                            (("rescope", "changed_percent", pre + "RescopeDOne", None), ("where_logonly", "changed_percent", pre + "WhereLogonlyDOne", None),
                                             (B.sel["cos_H2"], "changed_percent", pre + camel(B.sel["cos_H2"]) + "DOne", None), ("retr_sum", "changed_percent", pre + "RetrSumDOne", None))):
                if lab not in node["methods"]:
                    continue
                v = node["methods"][lab]["pools"]["hd_all"][mine_key]
                v = round(v, nd) if nd is not None else v
                eq.append({"log": log, "name": name, "mine": v, "stored": old.get(name), "equal": name in old and abs(v - old[name]) <= 1e-9})
            B.forget()
        # development log: the same five rows from the E1 per-row files give the same counts
        dev, diff, n_cmp = B.dev, [], 0
        for lab in ("where_logonly", "rescope", B.sel["cos_H3"], B.sel["cos_H2"], B.sel["retr_H2"]):
            for d in ([None] if lab == "where_logonly" else (0, 1, 2)):
                p = os.path.join(E1R, f"{dev}__{lab}__d{'x' if d is None else d}__tf.jsonl.gz")
                if not os.path.exists(p):
                    diff.append((lab, d, "no E1 file"))
                    continue
                rows = fx.io.load_rows(p)
                tag = "one run" if d is None else f"draw {d}"
                for pool, _, _ in POOLS:
                    a = one(rows, pool, tag)
                    b = next(x for x in RC[dev]["methods"][lab]["pools"][pool]["by_run"] if x["run"] == tag)
                    n_cmp += 1
                    if any(a[k] != b[k] for k in ("n", "n_released", "n_changed", "n_changed_released", "n_changed_not_released")):
                        diff.append((lab, d, pool))
        CK.add(5, f"{dev}: the rows of E1 (e1_main/out/rows) give the same counts as the rows of E3 for the five methods, every pool and draw", not diff, f"{n_cmp} cells compared; differing: {diff}")
    CK.add(5, "no row of item 5 is missing", not missing, f"missing: {missing}")
    CK.add(5, "D1 and release rates on heldout_disjoint equal the experiments' paper numbers to the last stored digit", all(r["equal"] for r in eq),
           f"{len(eq)} numbers compared; differing: {[(r['log'], r['name'], r['mine'], r['stored']) for r in eq if not r['equal']]}")
    # the finding: rows without a release whose answer changed
    found = []
    for log, node in RC.items():
        for lab, m in node["methods"].items():
            for pool, s in m["pools"].items():
                if s.get("changed_not_released_count"):
                    found.append({"log": log, "method": lab, "pool": pool, "count": s["changed_not_released_count"], "by_run": s["changed_not_released_by_run"],
                                  "not_released_by_run": s["not_released_by_run"], "rows": [r for p in s["by_run"] for r in p["rows_changed_not_released"]]})
    # how the rows without a release were evaluated: 'lib' / 'run' / 'memo' = the sequence had a forward pass under the method; 'pre' = the driver
    # assigned the pre-edit outcome without a forward (gates decided at the last prompt position: nothing fires, so the weight is the pre-edit weight)
    ev = {}
    for log, node in RC.items():
        for lab, m in node["methods"].items():
            e = ev.setdefault(lab, {"rows_without_release_summed_over_runs": 0, "with_a_forward_pass_under_the_method": 0, "pre_edit_outcome_assigned_without_a_forward": 0, "sources": {}})
            for pool in ("hd_all",) + HN:
                for p in m["pools"][pool].get("by_run", []):
                    for k, v in p["not_released_by_source"].items():
                        e["rows_without_release_summed_over_runs"] += v
                        e["pre_edit_outcome_assigned_without_a_forward" if k == "pre" else "with_a_forward_pass_under_the_method"] += v
                        e["sources"][k] = e["sources"].get(k, 0) + v
    for lab, e in ev.items():
        e["the_count_of_changed_rows_is"] = ("a measurement: every sequence without a release had its own forward pass under the method" if e["pre_edit_outcome_assigned_without_a_forward"] == 0
                                             else "0 by construction of the evaluation: the driver assigns the pre-edit outcome to a sequence on which nothing fires (no forward pass); "
                                                  "this fix-up adds no evidence for these rows; the evidence is the independent verification of E3, whose own forward passes at "
                                                  "calibration draw 0 reproduce every stored probe flag (PROTOCOL.md, amendment of 2026-11-12 06:16 UTC: 0 of 6,804 differ)")
    return RC, found, eq, missing, ev


def explain(found, RC):
    """What the stored files say about the rows without a release whose answer changed.  For each such row: does the SAME sequence change
    under the other methods of the log on runs where they release nothing, and how large is the change (token agreement with the pre-edit
    answer)?  A method that replays logged terms (ReScope-W: W_T minus the withheld terms) does not restore the pre-edit weight bit for
    bit; a gate at the last prompt position that releases nothing runs the pre-edit weight itself."""
    by_method = {}
    for f in found:
        e = by_method.setdefault(f["method"], {"n_rows": 0, "logs": set(), "pools": set(), "tok_loc": [], "n_tok": [], "csids": set()})
        e["n_rows"] += f["count"]
        e["logs"].add(f["log"])
        e["pools"].add(f["pool"])
        for r in f["rows"]:
            e["csids"].add((f["log"], r["csid"]))
            if r.get("tok_loc") is not None:
                e["tok_loc"].append(float(r["tok_loc"]))
            if r.get("n_tok") is not None:
                e["n_tok"].append(int(r["n_tok"]))
    out = {}
    for lab, e in by_method.items():
        replay = lab in ("where_logonly", "where_least_n10", "where_cal")
        tl = np.array(e["tok_loc"]) if e["tok_loc"] else None
        # the same sequences under the gates at the last prompt position, on runs where those gates release nothing either
        same, same_changed = 0, 0
        for log, csid in e["csids"]:
            for other, m in RC[log]["methods"].items():
                if other in ("where_logonly", "where_least_n10", "where_cal"):
                    continue
                for pool, s in m["pools"].items():
                    if pool in ("hd_far", "hd_near"):
                        continue
                    for p in s.get("by_run", []):
                        if any(r["csid"] == csid for r in p["rows_changed_not_released"]):
                            same_changed += 1
        out[lab] = {"n_rows_summed_over_runs": e["n_rows"], "n_distinct_sequences": len(e["csids"]), "logs": sorted(e["logs"]), "pools": sorted(e["pools"]),
                    "replays_logged_terms": replay,
                    "token_agreement_with_the_pre_edit_answer": None if tl is None else {"min": float(tl.min()), "median": float(np.median(tl)), "max": float(tl.max())},
                    "answer_length_tokens": None if not e["n_tok"] else {"min": int(min(e["n_tok"])), "median": float(np.median(e["n_tok"])), "max": int(max(e["n_tok"]))},
                    "same_sequences_changed_without_release_under_a_last_position_gate": same_changed,
                    "reason": ("the method withholds an edit by subtracting its logged term from the deployed weight (W_T minus the withheld terms); with no edit released the weight is the "
                               "pre-edit weight up to floating-point round-off, not bit for bit, and a teacher-forced token whose two leading logits are nearly tied can flip: the round-off "
                               "floor of a method that replays logged terms" if replay else
                               "NOT explained by round-off of replayed terms: a gate at the last prompt position that releases nothing runs the pre-edit weight; see the rows")}
    return out


# ----------------------------------------------------------------------------- Holm over the replication read-outs
def holm_bounds(ps, floor):
    """Holm step-down with the information which adjusted values are upper bounds (e1_main/code/fixup_compute.py, the same rule): a
    bootstrap p at the resolution stands for 'no resample on the other side of 0'; an adjusted value whose defining term comes from
    such a p is an upper bound."""
    keys = sorted(ps, key=lambda k: ps[k])
    m, run, src, out = len(keys), 0.0, False, {}
    for rank, k in enumerate(keys):
        term = min(1.0, (m - rank) * ps[k])
        at = ps[k] <= floor * (1 + 1e-12)
        if term > run:
            run, src = term, at
        elif term == run and at:
            src = True
        out[k] = {"p": ps[k], "p_holm": run, "at_resolution": bool(at), "p_holm_is_upper_bound": bool(src), "rank": rank + 1}
    return out


def holm(PN, CK):
    EP = {"Q": jload(FINAL + "/e3_qwen/out/fixup/endpoints_added.json")["logs"], "L": jload(FINAL + "/e3_llava/out/fixup/endpoints.json")["logs"]}
    out = {"about": {"label": "replication read-outs on unseen logs (not the confirmatory family of R6)",
                     "what": "Holm's step-down correction within H2 over the unseen logs of both backbones and the two references, and within H3 over the unseen logs of both backbones "
                             "and the three comparisons, from the two-sided bootstrap values of the behavioural paired differences",
                     "p": f"two-sided value of the paired crossed bootstrap (fx.stats.crossed_bootstrap_diff, {N_BOOT} resamples): share of resamples on the other side of 0, doubled",
                     "resolution": RES, "upper_bounds": f"a value at the resolution ({RES}) is an upper bound (no resample on the other side of 0); a Holm-adjusted value whose defining "
                                                         "term comes from such a value is an upper bound too (field p_holm_is_upper_bound; print with '<=')",
                     "draws": "behavioural calibration draws: 3 on logs with at most 500 edits, 1 from 1000 edits on (R23); with 1 draw the interval holds the unit component only",
                     "sources": ["e3_qwen/out/fixup/endpoints_added.json", "e3_llava/out/fixup/endpoints.json"], "alpha": 0.05,
                     "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint()}}
    Bs = {k: Backbone(k) for k in ("Q", "L")}
    fams = {"H2": [], "H3": []}
    for k, B in Bs.items():
        for log in B.unseen:
            e = EP[k][log]
            for role in xlib.H2W:
                x = e["H2"][role]
                y = x["all"]
                fams["H2"].append({"key": f"{log} | rescope minus {x['method']}", "log": log, "backbone": B.title, "family": family_of(log), "reference": x["method"], "role": role,
                                   "diff": y["diff"], "ci_lo": y["ci_lo"], "ci_hi": y["ci_hi"], "n_draws": x["n_draws"], "p": y["p"], "code": B.code(log), "w": "Htwo" + xlib.H2W[role]})
            for pair in xlib.H3W:
                x = e["H3"][pair]
                fams["H3"].append({"key": f"{log} | {x['a']} minus {x['b']}", "log": log, "backbone": B.title, "family": family_of(log), "reference": x["b"], "role": pair, "a": x["a"],
                                   "diff": x["diff"], "ci_lo": x["ci_lo"], "ci_hi": x["ci_hi"], "n_draws": x["n_draws"], "p": x["p"], "code": B.code(log), "w": "Hthree" + xlib.H3W[pair]})
    PN.put("XrelHolmResolution", RES, f"resolution of a two-sided bootstrap value (1 / {N_BOOT}); a value at the resolution is an upper bound (x_release/out/holm.json)")
    for h, members in fams.items():
        ps = {m["key"]: m["p"] for m in members}
        hb = holm_bounds(ps, RES)
        ref = ST.holm(ps)
        worst = max(abs(hb[k]["p_holm"] - ref["adjusted"][k]) for k in ps)
        CK.add(3, f"Holm ({h}): the adjusted values computed here equal fx.stats.holm", worst <= 1e-12, f"{len(ps)} members; largest absolute difference {worst:.3g}")
        for m in members:
            m.update(p_holm=hb[m["key"]]["p_holm"], at_resolution=hb[m["key"]]["at_resolution"], p_holm_is_upper_bound=hb[m["key"]]["p_holm_is_upper_bound"],
                     rank=hb[m["key"]]["rank"], reject_at_0_05=bool(hb[m["key"]]["p_holm"] <= 0.05),
                     p_holm_print=("<= " if hb[m["key"]]["p_holm_is_upper_bound"] else "") + f"{hb[m['key']]['p_holm']:.4f}")
            nm = "Xrel" + m["code"] + "Holm" + m["w"]
            what = f"{m['log']}: {h}, {m['key'].split(' | ')[1]} (replication read-out on an unseen log; Holm family of {len(members)}; x_release/out/holm.json)"
            PN.put(nm + "P", m["p"], what + ": two-sided bootstrap value" + (" (at the resolution: an upper bound)" if m["at_resolution"] else ""))
            PN.put(nm + "Adj", m["p_holm"], what + ": Holm-adjusted value" + (" (an upper bound: print with '<=')" if m["p_holm_is_upper_bound"] else ""))
            PN.put(nm + "IsBound", int(m["p_holm_is_upper_bound"]), what + ": 1 if the Holm-adjusted value is an upper bound")
            PN.put(nm + "Reject", int(m["reject_at_0_05"]), what + ": 1 if the Holm-adjusted value is at most 0.05")
        rej = [m for m in members if m["reject_at_0_05"]]
        out[h] = {"family_size": len(members), "n_logs": len({m["log"] for m in members}), "comparisons_per_log": len(members) // len({m["log"] for m in members}),
                  "smallest_adjusted_value_that_can_be_shown": len(members) * RES, "n_at_resolution": sum(m["at_resolution"] for m in members),
                  "n_adjusted_upper_bounds": sum(m["p_holm_is_upper_bound"] for m in members), "n_rejected_at_0_05": len(rej),
                  "n_rejected_rescope_better": sum(m["diff"] > 0 for m in rej), "n_rejected_rescope_worse": sum(m["diff"] < 0 for m in rej),
                  "rejected": [m["key"] for m in rej], "not_rejected": [m["key"] for m in members if not m["reject_at_0_05"]], "members": members}
        w = "Htwo" if h == "H2" else "Hthree"
        PN.put(f"XrelHolm{w}Family", len(members), f"{h}: size of the Holm family of the replication read-outs on unseen logs ({out[h]['n_logs']} logs x {out[h]['comparisons_per_log']} comparisons; x_release/out/holm.json)")
        PN.put(f"XrelHolm{w}Rejected", len(rej), f"{h}: comparisons of the family with a Holm-adjusted value of at most 0.05 (x_release/out/holm.json)")
        PN.put(f"XrelHolm{w}RejectedWorse", out[h]["n_rejected_rescope_worse"], f"{h}: of these, comparisons in which the paper's method is the worse one (x_release/out/holm.json)")
        PN.put(f"XrelHolm{w}RejectedBetter", out[h]["n_rejected_rescope_better"], f"{h}: of these, comparisons in which the paper's method is the better one (x_release/out/holm.json)")
        PN.put(f"XrelHolm{w}Floor", len(members) * RES, f"{h}: smallest Holm-adjusted value that can be shown (family size x resolution; x_release/out/holm.json)")
    CK.add(3, "Holm families: H2 = 11 unseen logs x 2 references, H3 = 11 unseen logs x 3 comparisons", out["H2"]["family_size"] == 22 and out["H3"]["family_size"] == 33,
           f"H2 {out['H2']['family_size']}, H3 {out['H3']['family_size']}")
    return out


# ----------------------------------------------------------------------------- the paper's table of logs
def deployed_metrics(rows, src):
    e = [r for r in rows if r["kind"] == "edit"]
    p = [r for r in rows if is_probe(r)]
    o = {}
    for m, k in (("rel", "rel_exact"), ("tgen", "tgen_exact"), ("igen", "igen_exact")):
        v = [bool(r["exact"]) for r in e if r["metric"] == m]
        o[k] = {"value": pct(v), "numerator": int(sum(v)), "denominator": len(v), "source": src}
    v = [bool(r["D1"]) for r in p]
    o["D1_heldout_disjoint"] = {"value": pct(v), "numerator": int(sum(v)), "denominator": len(v), "source": src}
    return o, len(p)


def logs_table(PN, CK):
    out, eq, cross = [], [], []
    old = {"Q": jload(FINAL + "/e3_qwen/out/paper_numbers.json"), "L": jload(FINAL + "/e3_llava/out/paper_numbers.json"), "P": jload(FINAL + "/e2_prefix/out/paper_numbers.json")}
    for key in ("Q", "L"):
        B = Backbone(key)
        entries = [(log, B.row_path(log, "deployed", None), "E3") for log in B.logs]
        entries += [(f"{B.dev}-p{n}", os.path.join(PFXR, f"{B.dev}-p{n}__deployed.jsonl.gz"), "E2 prefix") for n in (10, 20, 50)]
        for log, path, exp in entries:
            info = fx.logs.info(log)
            rows = fx.io.load_rows(path)
            src = os.path.relpath(path, FINAL)
            dm, n_hd = deployed_metrics(rows, src)
            fam = family_of(info.get("prefix_of") or log)
            role = "development" if log == B.dev else ("prefix of the development stream" if exp == "E2 prefix" else "unseen")
            code = B.code(log) if exp == "E3" else key + "AePfx" + PFX_T[info["T"]]
            e = {"log": log, "backbone": B.title, "backbone_key": info["backbone"], "editor_family": fam, "editor_family_registry": info.get("editor_family"), "editor": info.get("editor"),
                 "role": role, "T": info["T"], "T_source": "common/fx/logs.py (registry; equals n_edits of the log's metrics.json)", "n_heldout_disjoint_valid": n_hd,
                 "n_heldout_disjoint_valid_source": src, "deployed": dm, "evaluated_in": exp, "forward": "one sequence per forward pass", "prefix_of": info.get("prefix_of")}
            out.append(e)
            nm = "Xrel" + code + "Log"
            PN.put(nm + "T", info["T"], f"{log}: number of edits T (registry of the driver)")
            PN.put(nm + "HdRows", n_hd, f"{log}: valid heldout_disjoint rows ({src})")
            for k, w, t in (("rel_exact", "DepRel", "Rel exact"), ("tgen_exact", "DepTGen", "T-Gen exact"), ("igen_exact", "DepIGen", "I-Gen exact"), ("D1_heldout_disjoint", "DepDone", "D1 on heldout_disjoint")):
                PN.put(nm + w, dm[k]["value"], f"{log}: deployed log, {t}, percent ({dm[k]['numerator']} of {dm[k]['denominator']}), as evaluated in {exp} with one sequence per forward pass ({src})")
            # equality with the experiments' own paper numbers
            if exp == "E3":
                pre = "Ethree" + B.code(log) + "Deployed"
                names = (("rel_exact", pre + "RelExact", 2), ("tgen_exact", pre + "TGenExact", 2), ("igen_exact", pre + "IGenExact", 2), ("D1_heldout_disjoint", pre + "HdDone", 2)) if key == "Q" else (
                    ("rel_exact", pre + "Rel", None), ("tgen_exact", pre + "TGen", None), ("igen_exact", pre + "IGen", None), ("D1_heldout_disjoint", pre + "DOne", None))
                o = old[key]
            else:
                pre = "Pfx" + key + PFX_T[info["T"]] + "Deployed"
                names = (("rel_exact", pre + "Rel", None), ("tgen_exact", pre + "TGen", None), ("igen_exact", pre + "IGen", None), ("D1_heldout_disjoint", pre + "DOne", None))
                o = old["P"]
            for k, name, nd in names:
                v = round(dm[k]["value"], nd) if nd is not None else dm[k]["value"]
                eq.append({"log": log, "name": name, "mine": v, "stored": o.get(name), "equal": name in o and abs(v - o[name]) <= 1e-9})
            # development logs: the deployed log as evaluated in E1 and in E2
            if log == B.dev:
                for other, p in (("E1", os.path.join(E1R, f"{log}__deployed__dx__tf.jsonl.gz")), ("E2", os.path.join(E2R, f"base__{key}__deployed.jsonl.gz"))):
                    dm2, n2 = deployed_metrics(fx.io.load_rows(p), os.path.relpath(p, FINAL))
                    same = n2 == n_hd and all(dm2[k]["numerator"] == dm[k]["numerator"] and dm2[k]["denominator"] == dm[k]["denominator"] for k in dm)
                    cross.append({"log": log, "other": other, "file": os.path.relpath(p, FINAL), "equal": bool(same)})
                e["same_counts_in"] = [c for c in cross if c["log"] == log]
    CK.add(6, "the deployed log's Rel / T-Gen / I-Gen exact and D1 equal the experiments' paper numbers to the last stored digit (E3: both backbones; prefix logs: e2_prefix)",
           all(r["equal"] for r in eq), f"{len(eq)} numbers compared on {len(out)} logs; differing: {[(r['log'], r['name'], r['mine'], r['stored']) for r in eq if not r['equal']]}")
    CK.add(6, "development logs: the deployed log has the same counts in the per-row files of E1 and of E2 as in E3", all(c["equal"] for c in cross), str([(c["log"], c["other"], c["equal"]) for c in cross]))
    CK.add(6, "the table holds 19 logs: 13 of E3 (the two development logs among them) and 6 prefix logs", len(out) == 19, f"{len(out)} logs")
    return out, eq


def main():
    os.makedirs(OUT, exist_ok=True)
    PN, CK = Numbers(), Checks()
    t0 = time.time()
    print("Holm", flush=True)
    PN.cur = 3
    H = holm(PN, CK)
    jdump(H, os.path.join(OUT, "holm.json"))
    print(f"release against change  [{time.time() - t0:.0f}s]", flush=True)
    PN.cur = 5
    RC, found, eq5, missing, ev = release_change(PN, CK)
    CK.add(5, "how the count of changed rows without a release was obtained, per method (reported, not a failure): a measurement where every such row had a forward pass under the method",
           True, str({k: (v["with_a_forward_pass_under_the_method"], v["pre_edit_outcome_assigned_without_a_forward"]) for k, v in ev.items()}) + " = (rows with a forward, rows assigned the pre-edit outcome)")
    ex = explain(found, RC)
    n_found = sum(f["count"] for f in found)
    last = [f for f in found if f["method"] not in ("where_logonly", "where_least_n10", "where_cal")]
    CK.add(5, "rows without a release whose answer changed: count reported; a non-zero count is a FINDING (listed with the rows in release_change.json)", n_found == 0,
           f"{n_found} rows summed over the runs, in {len(found)} (log, method, pool) cells; by method: { {k: v['n_rows_summed_over_runs'] for k, v in ex.items()} }")
    CK.add(5, "no gate at the last prompt position (rescope, the selected gates of the cosine family, retr_sum) changes an answer without a release", not last,
           f"cells: {[(f['log'], f['method'], f['pool'], f['count']) for f in last]}")
    jdump({"about": {"what": "release against change (decision E4 of 01_paper/v4_rescope/_ext_review/TRIAGE_RESULTS.txt): per log, method and pool the rows on which an edit is released, the rows "
                             "whose pre-edit answer is not reproduced (D1), D1 among the rows with a release, and the NUMBER of rows without a release whose answer changed",
                     "release": "row flag `fired` of the per-row files: an edit is released on the sequence (gates at the last prompt position: at that position; ReScope-W: at any position)",
                     "aggregation": "percentages are means over the calibration draws (or complaint streams) of the per-run percentages; every run is listed with numerator and denominator "
                                    "(by_run); 'changed among released' averages the runs that release on at least one row",
                     "pools": {p: t for p, _, t in POOLS}, "reading": "low D1 does not imply accurate scope detection: the repair is conservative in its effect on answers, not selective in "
                                                                       "release; near-scope queries are outside the guarantee",
                     "E3_draws": "behavioural calibration draws 0-2 (draw 0 only from 1000 edits on); where_logonly has no calibration (one run)",
                     "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint()},
           "logs": RC, "changed_without_release": {"n_rows_summed_over_runs": n_found, "cells": found, "by_method": ex, "evidence_by_method": ev}, "missing": missing}, os.path.join(OUT, "release_change.json"))
    print(f"table of logs  [{time.time() - t0:.0f}s]", flush=True)
    PN.cur = 6
    LB, eq6 = logs_table(PN, CK)
    jdump({"about": {"what": "the paper's table of logs: every log of E3, the two development logs (among them) and the six prefix logs; the deployed log's edit metrics and D1 as "
                             "evaluated in these experiments (one sequence per forward pass; they differ from the metrics stored at build time on 0 to 13 edits per metric, amendment of "
                             "2026-11-12 06:16 UTC, decision 7)", "written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint()}, "logs": LB}, os.path.join(OUT, "logs_b1.json"))
    PN.cur = 0
    files = [FINAL + f"/{d}/out/paper_numbers.json" for d in ("e1_main", "e2_curves", "e2_prefix", "e3_qwen", "e3_llava")] + list(xlib.EARLIER) + [
        FINAL + "/e3_qwen/out/fixup/paper_numbers_fixup.json", FINAL + "/e3_llava/out/fixup/paper_numbers_fixup.json"]
    coll = collisions(PN.num, files)
    CK.add(0, "no name of x_release/out/paper_numbers_fixup.json exists in a paper-number file of E1, E2, the prefix logs, E3 or of a fix-up (e1_main, e2_curves, e3_qwen, e3_llava)", not coll,
           f"collisions: {len(coll)}; {len(files)} files")
    CK.add(0, "every paper number has a documentation line", set(PN.num) == set(PN.doc) and all(PN.doc.values()), f"{len(PN.num)} names")
    neq = {k: sum(1 for r in eq5 + eq6 if r["log"].startswith("TB-" + k)) for k in ("Q", "L")}
    CK.add(0, "at least ten numbers per backbone that also exist in the experiments' paper numbers are equal to the last stored digit", all(r["equal"] for r in eq5 + eq6) and min(neq.values()) >= 10,
           f"compared: {neq}")
    mod = modified_since_start()
    CK.add(0, "no existing file of the experiments was modified (modification times against the start of this work; the new files of the fix-up excluded)", not mod["modified"],
           (mod["skipped"] + "; " if mod.get("skipped") else "") + f"start {mod['start_utc']} UTC; {mod['n_files_checked']} files checked in {', '.join(mod['directories'])}; modified: {mod['modified'][:5]}", skipped=bool(mod.get("skipped")))
    jdump(dict(sorted(PN.num.items())), os.path.join(OUT, "paper_numbers_fixup.json"))
    jdump(dict(sorted(PN.doc.items())), os.path.join(OUT, "paper_numbers_fixup_doc.json"))
    by_item = {str(i): sum(v == i for v in PN.item.values()) for i in sorted(set(PN.item.values()))}
    jdump({"written": time.strftime("%Y-%m-%d %H:%M:%S"), "code": fx.code_fingerprint(), "n_paper_numbers": len(PN.num), "n_paper_numbers_by_item": by_item, "n_collisions": len(coll),
           "collisions": coll, "equal_to_existing_paper_numbers": eq5 + eq6, "unmodified": mod, "n_checks": len(CK.list), "n_checks_failed": sum(c["ok"] is False for c in CK.list), "n_checks_skipped": sum(bool(c.get("skipped")) for c in CK.list),
           "checks": CK.list}, os.path.join(OUT, "checks.json"))
    print(f"x_release: paper numbers {len(PN.num)} {by_item}; collisions: {len(coll)}; checks failed: {sum(c['ok'] is False for c in CK.list)}, skipped: {sum(bool(c.get('skipped')) for c in CK.list)} of {len(CK.list)}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
