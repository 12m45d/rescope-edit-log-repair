"""Task 6: mechanics check of retr and lookup on the two development logs ONLY (calibration draw 0, alpha = 0.05, n = 200).

    python mechanics.py            -> out/mechanics.txt, out/mechanics.json

A mechanics check, not a result: it shows that the pipeline runs end to end and what the pieces do; nothing was tuned on it and
no number of it may be used for a decision.  Deterministic: no time stamp, no timing (a re-run reproduces both files byte for byte).  Row sets and the calibration draw are computed from the key cache's index and the
library (retr.row_sets / retr.default_draw), without the shared driver; tests.py (fx_compat) compares them and these numbers with
fx.gate.row_sets, fx.draws.calibration and fx.metrics.gate_stats.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402

import numpy as np  # noqa: E402

ALPHA, N, DRAW = 0.05, 200, 0


def pct(x):
    x = np.asarray(x, dtype=bool)
    return None if x.size == 0 else 100.0 * float(x.mean())


def part_stats(d, sel, with_own=False):
    sel = np.asarray(sel, dtype=bool)
    o = {"n": int(sel.sum()), "fired_pct": pct(d["fired"][sel]), "n_fired": int(d["fired"][sel].sum())}
    if with_own:
        o.update(coverage_pct=pct(d["own_top"][sel]), own_cand_pct=pct(d["own_cand"][sel]), other_edit_released_pct=pct(d["fired"][sel] & ~d["own_top"][sel]))
    return o


def run_log(cache, name):
    import retr
    E = retr.edits(name)
    st = retr.store(cache)
    rs = retr.row_sets(E)
    cal = retr.default_draw(E, N, DRAW)
    thr = retr.calibrate_log(E, cal, ALPHA, seed=DRAW)
    out = {"log": name, "cache": cache, "T": E.T, "alpha": ALPHA, "n": N, "draw": DRAW, "n_calibration_rows": len(cal),
           "calibration": {k: v for k, v in thr.rec.items() if k not in ("block1", "block2")}, "lookup_record": retr.lookup_record(E),
           "rows": {k: len(v) for k, v in rs.items()}}
    idx = retr.cache_index(cache)
    b1 = [cal[i] for i in thr.rec["block1"]]
    q1 = {st.seq[j]["question_key"] for j in b1}
    out["question_attribute"] = {
        "n_edits_whose_question_string_is_asked_by_a_block1_query": sum(C.norm_q(q) in q1 for q in E.question),
        "n_edits_with_question_baseline_ge_1-1e-6": int((thr.b_q >= 1 - 1e-6).sum()),
        "n_edits_whose_literal_question_threshold_exceeds_1": int((thr.tau_q > 1.0 + 1e-6).sum()),
        "n_calibration_queries_whose_question_is_an_edit_question_of_the_log": sum(st.seq[j]["question_key"] in {C.norm_q(q) for q in E.question} for j in cal),
        "installed_question_threshold_uncapped_edits_min_median_max": [float(x) for x in np.quantile(thr.th_q[~thr.capped_q], [0, 0.5, 1])] if (~thr.capped_q).any() else None,
        "installed_image_threshold_min_median_max": [float(x) for x in np.quantile(thr.th_img, [0, 0.5, 1])]}
    kinds = {p: np.array([st.seq[j]["kind"] for j in rs[p]]) for p in ("probe", "edit", "hn", "benign")}
    ptype = {p: np.array([st.seq[j]["ptype"] for j in rs[p]]) for p in ("probe", "benign")}
    dj = np.array([bool(idx[j].get("disjoint")) for j in rs["probe"]])
    rep = np.array([idx[j].get("pool") == "repair" for j in rs["probe"]])
    incal = np.isin(np.array(rs["probe"]), np.array(cal))
    uid_pos = {u: i for i, u in enumerate(E.uids)}
    eb = np.array([uid_pos.get(st.seq[j]["edit_b_uid"], -1) for j in rs["hn"]])
    for meth in ("retr", "lookup"):
        dec = {p: (retr.retr_decide(E, rs[p], thr) if meth == "retr" else retr.lookup_decide(E, rs[p])) for p in ("probe", "edit", "hn", "benign")}
        o = {}
        d = dec["probe"]
        o["heldout_disjoint"] = part_stats(d, dj)
        o["heldout_disjoint_image"] = part_stats(d, dj & (ptype["probe"] == "image"))
        o["heldout_disjoint_text"] = part_stats(d, dj & (ptype["probe"] == "text"))
        o["repair"] = part_stats(d, rep)
        o["repair_calibration_rows"] = part_stats(d, incal)
        if meth == "retr":
            o["heldout_disjoint_single_attribute"] = {"image_alone_pct": pct(d["n_img_pass"][dj] > 0), "question_alone_pct": pct(d["n_q_pass"][dj] > 0),
                                                      "both_some_edit_each_pct": pct((d["n_img_pass"][dj] > 0) & (d["n_q_pass"][dj] > 0)), "conjunction_pct": pct(d["fired"][dj])}
        d = dec["edit"]
        for k in C.EDIT_KINDS:
            o[k] = part_stats(d, kinds["edit"] == k, True)
        d = dec["hn"]
        for k, lab in (("hn_i", "HN-I"), ("hn_q", "HN-Q"), ("hn_x", "HN-X")):
            sel = kinds["hn"] == k
            o[lab] = part_stats(d, sel, True)
            if k == "hn_x":
                o[lab]["question_edit_released_pct"] = pct(d["fired"][sel] & (d["top_edit"][sel] == eb[sel]))
        d = dec["benign"]
        o["benign"] = part_stats(d, np.ones(len(rs["benign"]), dtype=bool))
        o["benign_image"] = part_stats(d, ptype["benign"] == "image")
        o["benign_text"] = part_stats(d, ptype["benign"] == "text")
        o["n_capped_edits"] = thr.rec["n_edits_capped"] if meth == "retr" else None
        fired_all = np.concatenate([dec[p]["fired"] for p in dec])
        ncand = np.concatenate([dec[p]["n_cand"] for p in dec])
        o["lineage"] = {"n_firing_rows": int(fired_all.sum()), "max_candidates": int(ncand.max()), "pct_firing_rows_with_more_than_one_candidate": pct(ncand[fired_all] > 1)}
        out[meth] = o
    return out


def render(res) -> str:
    f = lambda v, nd=1: "-" if v is None else f"{v:.{nd}f}"            # noqa: E731
    L = ["MECHANICS CHECK of the reference systems retr and lookup (PROTOCOL.md section 6, Review round 1, item R4)",
         "A mechanics check, NOT a result: development logs only, calibration draw 0, alpha = 0.05, n = 200; nothing was tuned, no number here",
         "selects anything.  Gate level (no model): firing = an edit is released; coverage = the sequence's own edit is the released edit.",
         "Written by feat2/retr/mechanics.py (deterministic: a re-run reproduces this file byte for byte; test repro).", ""]
    head = ["log", "method", "false firing", "T-Gen cov", "I-Gen cov", "Rel cov", "HN-I fired", "HN-Q fired", "HN-X fired", "capped edits"]
    rows = []
    for name, r in res["logs"].items():
        for m in ("retr", "lookup"):
            o = r[m]
            rows.append([name, m, f"{f(o['heldout_disjoint']['fired_pct'], 2)} % ({o['heldout_disjoint']['n_fired']}/{o['heldout_disjoint']['n']})",
                         f"{f(o['tgen']['coverage_pct'])} %", f"{f(o['igen']['coverage_pct'])} %", f"{f(o['rel']['coverage_pct'])} %",
                         f"{f(o['HN-I']['fired_pct'])} % (n {o['HN-I']['n']})", f"{f(o['HN-Q']['fired_pct'])} % (n {o['HN-Q']['n']})",
                         f"{f(o['HN-X']['fired_pct'])} % (n {o['HN-X']['n']})", "n/a (no calibration)" if o["n_capped_edits"] is None else f"{o['n_capped_edits']} of {r['T']}"])
    L += ["false firing: valid heldout_disjoint probe rows; T-Gen / I-Gen / Rel: 100 sequences each; hard negatives: valid rows of the log's edits", "",
          C.table(rows, head), ""]
    for name, r in res["logs"].items():
        c = r["calibration"]
        L += [f"--- {name} (T = {r['T']}; calibration: {r['n_calibration_rows']} valid REPAIR probes, draw {r['draw']}, blocks {c['image']['n1']} + {c['image']['n2']}, "
              f"{c['n_block1_queries_with_image']} / {c['n_block2_queries_with_image']} of them with an image)",
              f"  retr calibration   image:    eta = {c['image']['eta']:.4f}, rank {c['image']['rank']} of {c['image']['n2']}, bound {100 * c['image']['bound_false_firing']:.2f} %, "
              f"baselines min / median / max {' / '.join(f'{x:.3f}' for x in c['image']['baseline_min_median_max'])}",
              f"                     question: eta = {c['question']['eta']:.4f}, rank {c['question']['rank']} of {c['question']['n2']}, bound {100 * c['question']['bound_false_firing']:.2f} %, "
              f"baselines min / median / max {' / '.join(f'{x:.3f}' for x in c['question']['baseline_min_median_max'])}",
              f"                     bound of the conjunction {100 * c['bound_false_firing']:.2f} %; own requests the literal rule silences: {c['n_own_requests_silenced_by_literal_rule']} "
              f"-> capped edits {c['n_edits_capped']} (image threshold {c['n_edits_capped_image']}, question threshold {c['n_edits_capped_question']}); "
              f"own requests released by their own edit {c['n_own_requests_released_by_own_edit']}/{r['T']}",
              f"                     question attribute, measured: literal threshold = baseline + eta; {r['question_attribute']['n_edits_whose_literal_question_threshold_exceeds_1']} edits have a "
              f"literal threshold above 1 (unreachable for a cosine) and keep only their own question string (cap at (1 - 1e-9) x own score); "
              f"{r['question_attribute']['n_edits_with_question_baseline_ge_1-1e-6']} of them have baseline 1 (a block-1 query asks their question word for word; "
              f"{r['question_attribute']['n_calibration_queries_whose_question_is_an_edit_question_of_the_log']} of the {r['n_calibration_rows']} calibration queries ask an edit's question)",
              f"                     installed thresholds: image min / median / max {' / '.join(f'{x:.3f}' for x in r['question_attribute']['installed_image_threshold_min_median_max'])}; "
              f"question, uncapped edits: {' / '.join(f'{x:.3f}' for x in r['question_attribute']['installed_question_threshold_uncapped_edits_min_median_max'])}",
              f"                     guarantee: {c['guarantee']}",
              f"  lookup             edits without a content word (never released): {r['lookup_record']['n_edits_without_content_word']}; edits sharing their image with another "
              f"edit: {r['lookup_record']['n_edits_sharing_their_image_with_another_edit']}; stop list sha1 {r['lookup_record']['stop_list_sha1']}"]
        rows = []
        for m in ("retr", "lookup"):
            o = r[m]
            for lab, k in (("heldout_disjoint", "heldout_disjoint"), ("  image rows", "heldout_disjoint_image"), ("  text rows", "heldout_disjoint_text"), ("REPAIR pool (valid)", "repair"),
                           ("  the 200 calibration rows", "repair_calibration_rows"), ("benign pool (3000)", "benign"), ("  image rows", "benign_image"), ("  text rows", "benign_text"),
                           ("Rel", "rel"), ("T-Gen", "tgen"), ("I-Gen", "igen"), ("HN-I", "HN-I"), ("HN-Q", "HN-Q"), ("HN-X", "HN-X")):
                p = o[k]
                rows.append([m, lab, p["n"], f(p["fired_pct"], 2), f(p.get("coverage_pct"), 1) if "coverage_pct" in p else "", f(p.get("own_cand_pct"), 1) if "own_cand_pct" in p else "",
                             f(p.get("other_edit_released_pct"), 1) if "other_edit_released_pct" in p else "", f(p.get("question_edit_released_pct"), 1) if "question_edit_released_pct" in p else ""])
        L += [C.table(rows, ["method", "rows", "n", "fired %", "own edit released %", "own edit candidate %", "another edit released %", "HN-X: question's edit released %"])]
        s = r["retr"]["heldout_disjoint_single_attribute"]
        L += [f"  retr on heldout_disjoint: some edit passes the image threshold alone {f(s['image_alone_pct'], 2)} %, the question threshold alone {f(s['question_alone_pct'], 2)} %, "
              f"both (not necessarily the same edit) {f(s['both_some_edit_each_pct'], 2)} %, the conjunction (same edit) {f(s['conjunction_pct'], 2)} %",
              f"  candidates per firing row: retr max {r['retr']['lineage']['max_candidates']} ({f(r['retr']['lineage']['pct_firing_rows_with_more_than_one_candidate'])} % with more than one), "
              f"lookup max {r['lookup']['lineage']['max_candidates']}", ""]
    return "\n".join(L)


def main(out_dir: str = C.OUT, verbose: bool = True) -> dict:
    res = {"what": "mechanics check, development logs only; not a result", "alpha": ALPHA, "n": N, "draw": DRAW, "logs": {}}
    for cache, name in C.DEV_LOGS.items():
        res["logs"][name] = run_log(cache, name)
    C.jdump(res, os.path.join(out_dir, "mechanics.json"))
    txt = render(res)
    C.text_write(txt, os.path.join(out_dir, "mechanics.txt"))
    if verbose:
        print(txt)
    return res


if __name__ == "__main__":
    main()
