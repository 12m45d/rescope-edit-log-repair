"""E9: results.json (fx.io schema), table.txt, paper_numbers.json (+ doc) from out/hash/, out/lineage_<log>.json and the existing records
of the linear log.  No number is typed by hand; the linear rollback sizes are READ from the cited files, nothing is recomputed.

    python e9_render.py
"""
from __future__ import annotations
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E9 = os.path.dirname(HERE)
OUT = os.path.join(E9, "out")
sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/10_final/common")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import fx  # noqa: E402

EXP = f"{_RP.ROOT}/02_research/experiments"
P12_H2H = EXP + "/08_rescope/p12_linear_streams/out/text_h2h/rows_{log}.jsonl"
P12_MD = EXP + "/08_rescope/p12_linear_streams/results.md"
RQ1_ROWS = EXP + "/07_repair_v3/rq1/rows/rows_{log}.jsonl"
RQ1_MD = EXP + "/07_repair_v3/rq1/results.md"
E5_PN = EXP + "/10_final/e5_linear/out/paper_numbers.json"
LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
MS, N_SETS = (1, 5, 10, 20, 50), 3
WORD = {1: "One", 5: "Five", 10: "Ten", 20: "Twenty", 50: "Fifty"}
pct = fx.metrics.pct


def load(sh, tag):
    p = os.path.join(OUT, "hash", f"{sh}__{tag}.jsonl.gz")
    return fx.io.load_rows(p) if os.path.exists(p) else None


def flag(r):
    """The reported flag of a sequence: exact (edit sequences), D1 (probes)."""
    return bool(r["exact"]) if r["kind"] == "edit" else bool(r["D1"])


def pred(r):
    """What the teacher-forced answer of a sequence looks like in its row: (first mismatching token, token accuracy)."""
    return (r.get("first_mismatch"), round(float(r["token_acc"]), 6))


def revocation(before, after, pre, R, released_ref):
    """One revocation set: `before` / `after` = rows of the log before / after the revocation on the same sequences; pre = pre-edit rows;
    released_ref = {csid: edits released under the scoped log before the revocation} (the lineage of each tuple)."""
    Rs = set(R)
    pre = {r["csid"]: r for r in pre}
    o = {"m": len(R)}
    assert [r["csid"] for r in before] == [r["csid"] for r in after]
    pairs = list(zip(before, after))
    for grp, sel in (("revoked", lambda r: r["kind"] == "edit" and r["edit"] in Rs), ("kept", lambda r: r["kind"] == "edit" and r["edit"] not in Rs)):
        for mt in ("rel", "tgen", "igen"):
            sub = [(a, b) for a, b in pairs if sel(a) and a["metric"] == mt]
            o[f"{grp}.{mt}.exact_before"], o[f"{grp}.{mt}.exact_after"] = pct(a["exact"] for a, b in sub), pct(b["exact"] for a, b in sub)
            o[f"{grp}.{mt}.exact_preedit"] = pct(pre[a["csid"]]["exact"] for a, b in sub)
            o[f"{grp}.{mt}.n"] = len(sub)
        sub = [(a, b) for a, b in pairs if sel(a)]
        o[f"{grp}.n_seq"] = len(sub)
        o[f"{grp}.n_flags_changed"] = sum(flag(a) != flag(b) for a, b in sub)
        o[f"{grp}.n_logits_changed"] = sum(a["logit_sha1"] != b["logit_sha1"] for a, b in sub)
        o[f"{grp}.n_pred_changed"] = sum(pred(a) != pred(b) for a, b in sub)
        o[f"{grp}.n_logits_equal_preedit_after"] = sum(b["logit_sha1"] == pre[a["csid"]]["logit_sha1"] for a, b in sub)
    pr = [(a, b) for a, b in pairs if a["kind"] == "probe"]
    o["probe.n"] = len(pr)
    o["probe.n_flags_changed"] = sum(flag(a) != flag(b) for a, b in pr)
    o["probe.n_logits_changed"] = sum(a["logit_sha1"] != b["logit_sha1"] for a, b in pr)
    o["probe.n_pred_changed"] = sum(pred(a) != pred(b) for a, b in pr)
    o["probe.D1_before"], o["probe.D1_after"] = pct(a["D1"] for a, b in pr), pct(b["D1"] for a, b in pr)
    o["probe.n_D1_fixed"] = sum(bool(a["D1"]) and not b["D1"] for a, b in pr)
    o["probe.n_D1_lost"] = sum(not a["D1"] and bool(b["D1"]) for a, b in pr)
    # tuples outside the lineage of the deleted statements: no deleted statement is released on them under the scoped log (before);
    # the own sequences of the deleted statements are not part of this set
    outside = [(a, b) for a, b in pairs if not (set(released_ref[a["csid"]]) & Rs) and not (a["kind"] == "edit" and a["edit"] in Rs)]
    inside = [(a, b) for a, b in pairs if (set(released_ref[a["csid"]]) & Rs) and not (a["kind"] == "edit" and a["edit"] in Rs)]
    o["outside.n"] = len(outside)
    o["outside.n_logits_changed"] = sum(a["logit_sha1"] != b["logit_sha1"] for a, b in outside)
    o["outside.n_flags_changed"] = sum(flag(a) != flag(b) for a, b in outside)
    o["outside.n_pred_changed"] = sum(pred(a) != pred(b) for a, b in outside)
    o["inside.n"] = len(inside)
    o["inside.n_logits_changed"] = sum(a["logit_sha1"] != b["logit_sha1"] for a, b in inside)
    o["inside.n_flags_changed"] = sum(flag(a) != flag(b) for a, b in inside)
    return o


def frame(before, after, R):
    """Frame rule of the scoped log, on ALL sequences of the run: a sequence on which no edit of R was released before the revocation
    and whose logits changed is a violation (expected: none).  Also: sequences on which an edit of R WAS released and another edit
    is released after the masking (top-1 among several candidates), each listed."""
    Rs = set(R)
    viol, other, n_frame, n_touched, n_touched_changed = [], [], 0, 0, 0
    for a, b in zip(before, after):
        hit = set(a["released"]) & Rs
        if not hit:
            n_frame += 1
            if a["logit_sha1"] != b["logit_sha1"]:
                viol.append({"csid": a["csid"], "released_before": a["released"], "released_after": b["released"]})
        else:
            n_touched += 1
            n_touched_changed += a["logit_sha1"] != b["logit_sha1"]
            if b["released"]:
                other.append({"csid": a["csid"], "kind": a["kind"], "metric": a.get("metric"), "own": a.get("edit"), "released_before": a["released"], "released_after": b["released"]})
    return {"n_frame": n_frame, "n_violations": len(viol), "violations": viol, "n_touched": n_touched, "n_touched_logits_changed": int(n_touched_changed),
            "n_other_edit_released": len(other), "other_edit_released": other}


def linear_records(name):
    """How many edits a rollback of the LINEAR log needs for ONE complaint, read from existing records (not recomputed)."""
    import json
    out = {"sources": {}}
    for tag, path, what in (("p12_text", P12_H2H.format(log=name), "P12 text head-to-head: 30 text complaints of the REPAIR pool"),
                            ("rq1_image", RQ1_ROWS.format(log=name), "rq1 head-to-head: image complaints of the REPAIR pool")):
        if not os.path.exists(path):                                    # ARTIFACT: stop instead of dropping the paper numbers of this source
            raise SystemExit(f"e9_render: input missing: {path}\n  p12_text: linear_baselines/streams/out/text_h2h/ (in the repository); rq1_image: "
                             "linear_baselines/rollback_image_exploratory/ (in the repository; records of the exploratory phase), both mounted by "
                             "rescope_paths.py.  Restore the files (git checkout) and run again.  Nothing was written.")
        rows = [json.loads(x) for x in open(path) if x.strip()]
        out["sources"][tag] = path
        for cfg in ("RB_BEH", "RB_FO"):
            sub = [r for r in rows if r.get("config") == cfg]
            ks = [int(r["meta"]["k"]) for r in sub if "k" in (r.get("meta") or {})]
            res = [bool(r["eval"]["complaints"][0]["D1_resolved"]) for r in sub]
            kres = [k for k, ok in zip(ks, res) if ok] if len(ks) == len(res) else []
            brk = [r["eval"].get("edits_broken_total") for r in sub]
            out[f"{tag}.{cfg}"] = {"what": what, "n_complaints": len(sub), "n_resolved": int(sum(res)), "k_median": float(np.median(ks)) if ks else None,
                                   "k_mean": float(np.mean(ks)) if ks else None, "k_min": int(min(ks)) if ks else None, "k_max": int(max(ks)) if ks else None,
                                   "k_median_resolved": float(np.median(kres)) if kres else None, "n_k_is_1": int(sum(k == 1 for k in ks)),
                                   "budget": (sub[0].get("cfg") or {}).get("budget") if sub else None,
                                   "edits_broken_mean": float(np.mean([b for b in brk if b is not None])) if any(b is not None for b in brk) else None}
    pn = fx.io.jload(E5_PN)
    out["sources"]["e5"] = E5_PN
    for c in ("c0.1", "c0.5"):
        out[f"e5.{c}"] = {k: pn.get(f"rollback.{name}.{c}.{k}") for k in ("opt_n_complaints", "opt_n_certified", "opt_median_certified", "opt_median_lower_bound",
                                                                         "opt_median_upper_bound", "median_best_rule", "median_lb", "min_lb", "max_lb")}
    return out


def agg(vals):
    v = [float(x) for x in vals if x is not None]
    return None if not v else {"mean": float(np.mean(v)), "min": float(min(v)), "max": float(max(v)), "n": len(v)}


def fa(a, nd=1):
    return "-" if a is None else (f"{a['mean']:.{nd}f}" if a["min"] == a["max"] else f"{a['mean']:.{nd}f} ({a['min']:.{nd}f}-{a['max']:.{nd}f})")


def main():
    R = fx.io.Results("E9", OUT)
    paper, doc, text = {}, {}, []
    text.append(f"E9 MAINTENANCE OF A SCOPED LOG: SELECTIVE REVOCATION AND LINEAGE      (rendered {time.strftime('%Y-%m-%d %H:%M:%S')} by code/e9_render.py; no number is typed by hand)")
    text.append("Development logs; alpha = 0.05, n = 200, calibration draw 0 (R19 c); revocation sets R of m edits, 3 seeded sets per m, the SAME sets for the scoped and the linear log.")
    text.append("Scoped log = the calibrated default method with the edits of R masked (threshold +inf; nothing else changes).  Linear log = the deployed weight minus the stored rank-one")
    text.append("solver terms of R (harness rollback_weight).  Every sequence of every run is one B = 1 forward through the library's Evaluator with a hash of its logits (ses.tf_hashed);")
    text.append("'bit-identical' compares these hashes.  Sequences: Rel / T-Gen / I-Gen of all 100 edits (300) and the valid heldout_disjoint probes.  Cells: mean over the 3 sets (range).")
    text.append("flag = exact (edit sequences) / D1 (probes); the pre-edit level is the target for the revoked edits.")
    for name in LOGS:
        sh = SHORT[name]
        text += ["", "#" * 150, name, "#" * 150]
        setup_p = os.path.join(OUT, f"setup_{name}.json")
        if not os.path.exists(setup_p):
            text.append("   pending (no run yet)")
            continue
        setup = fx.io.jload(setup_p)
        base_s, base_d, base_p = load(sh, "base_scoped"), load(sh, "base_deployed"), load(sh, "base_preedit")
        if base_s is None or base_d is None or base_p is None:
            text.append("   pending (base runs)")
            continue
        main_ids = [r["csid"] for r in base_d]
        bs = {r["csid"]: r for r in base_s}
        bp = {r["csid"]: r for r in base_p}
        base_s_main = [bs[c] for c in main_ids]
        base_p_main = [bp[c] for c in main_ids]
        released_ref = {c: bs[c]["released"] for c in main_ids}
        mm = setup["method"]
        text.append(f"   scoped log before revocation: default method, draw 0, digest {mm['digest']}, capped edits {mm['n_capped']}, own prompts not released "
                    f"{(mm.get('own_prompts') or {}).get('n_not_released')}; sequences: {setup['n_edit_sequences']} edit sequences, {setup['n_probe_rows']} heldout_disjoint probes")
        pre_fire = sum(bool(r["released"]) for r in base_p)
        silent = [c for c in bs if not bs[c]["released"]]
        ncert = sum(bs[c]["logit_sha1"] == bp[c]["logit_sha1"] for c in silent)
        text.append(f"   certificate (Prop. certificate a) on the base run: {len(silent)} sequences on which nothing is released; logits bit-identical to the pre-edit model on {ncert} of them")
        R.add(name, "scoped", "certificate.n_silent", len(silent), draw=0, alpha=0.05, n=200, pool="all")
        R.add(name, "scoped", "certificate.n_silent_bit_identical_preedit", ncert, draw=0, alpha=0.05, n=200, pool="all")
        paper[f"Enine{sh}CertSilent"], paper[f"Enine{sh}CertIdentical"] = len(silent), ncert
        doc[f"Enine{sh}CertSilent"] = f"{name}: sequences (edit sequences, heldout_disjoint probes, firing near-scope negatives) on which the calibrated default method releases nothing (draw 0)"
        doc[f"Enine{sh}CertIdentical"] = f"{name}: of these, sequences whose logits are bit-identical to the pre-edit model's"

        # ------------------------------------------------------------ (a) selective revocation
        per = {"scoped": {}, "linear": {}}
        frames = {}
        for m in MS:
            for s in range(N_SETS):
                Rset = setup["sets"][str(m)][s]
                a_s, a_l = load(sh, f"scoped_m{m}_s{s}"), load(sh, f"linear_m{m}_s{s}")
                if a_s is not None:
                    per["scoped"][(m, s)] = revocation(base_s_main, a_s, base_p_main, Rset, released_ref)
                    frames[(m, s)] = frame(base_s_main, a_s, Rset)
                if a_l is not None:
                    per["linear"][(m, s)] = revocation(base_d, a_l, base_p_main, Rset, released_ref)
        for meth in ("scoped", "linear"):
            for (m, s), o in per[meth].items():
                for k, v in o.items():
                    if k != "m":
                        R.add(name, meth, "rev." + k, v, variant=f"m={m},set={s}", draw=0 if meth == "scoped" else None, alpha=0.05 if meth == "scoped" else None,
                              n=200 if meth == "scoped" else None, pool="edits" if k.startswith(("revoked", "kept")) else "heldout_disjoint" if k.startswith("probe") else "edits+heldout_disjoint")
        for (m, s), f in frames.items():
            for k in ("n_frame", "n_violations", "n_touched", "n_touched_logits_changed", "n_other_edit_released"):
                R.add(name, "scoped", "frame." + k, f[k], variant=f"m={m},set={s}", draw=0, alpha=0.05, n=200, pool="edits+heldout_disjoint")
        A = {meth: {m: {k: agg([per[meth][(m, s)][k] for s in range(N_SETS) if (m, s) in per[meth]]) for k in next(iter(per[meth].values())) if k != "m"}
                    for m in MS if any((m, s) in per[meth] for s in range(N_SETS))} for meth in ("scoped", "linear") if per[meth]}
        for meth, am in A.items():
            for m, d in am.items():
                for k, a in d.items():
                    if a:
                        R.add(name, meth, f"rev.{k}@mean", a["mean"], ci_lo=a["min"], ci_hi=a["max"], n_units=a["n"], variant=f"m={m}", pool="", interval="range over the revocation sets")
        text += ["", "(a) SELECTIVE REVOCATION -- edits: exact % of the revoked edits (after | before | pre-edit level) and of the kept edits (after | before)"]
        hdr = ["log", "m", "sets", "revoked Rel", "revoked T-Gen", "revoked I-Gen", "kept Rel", "kept T-Gen", "kept I-Gen"]
        body = []
        for meth in ("scoped", "linear"):
            for m, d in A.get(meth, {}).items():
                body.append([meth, str(m), str(d["probe.n"]["n"])]
                            + [f"{fa(d[f'revoked.{mt}.exact_after'])} | {fa(d[f'revoked.{mt}.exact_before'])} | {fa(d[f'revoked.{mt}.exact_preedit'])}" for mt in ("rel", "tgen", "igen")]
                            + [f"{fa(d[f'kept.{mt}.exact_after'])} | {fa(d[f'kept.{mt}.exact_before'])}" for mt in ("rel", "tgen", "igen")])
        text.append(fx.io.text_table(hdr, body))
        text += ["", "(a) SELECTIVE REVOCATION -- side effects: number of kept-edit sequences / heldout_disjoint probe rows whose flag changes and whose logits are not bit-identical",
                 "    to those before the revocation; D1 on the probes (% after | before; rows fixed / lost)"]
        hdr = ["log", "m", "revoked seqs: flag changed", "kept seqs", "kept: flag changed", "kept: logits changed", "probes", "probes: flag changed", "probes: logits changed", "D1 after | before", "D1 fixed", "D1 lost"]
        body = []
        for meth in ("scoped", "linear"):
            for m, d in A.get(meth, {}).items():
                body.append([meth, str(m), f"{fa(d['revoked.n_flags_changed'])} of {fa(d['revoked.n_seq'], 0)}", fa(d["kept.n_seq"], 0), fa(d["kept.n_flags_changed"]), fa(d["kept.n_logits_changed"]), fa(d["probe.n"], 0), fa(d["probe.n_flags_changed"]),
                             fa(d["probe.n_logits_changed"]), f"{fa(d['probe.D1_after'], 2)} | {fa(d['probe.D1_before'], 2)}", fa(d["probe.n_D1_fixed"]), fa(d["probe.n_D1_lost"])])
        text.append(fx.io.text_table(hdr, body))
        if frames:
            text += ["", "(a) FRAME RULE of the scoped log (Prop. certificate b, c), per revocation set, over ALL 300 edit sequences + probes: 'frame' = sequences on which no edit of R was",
                     "    released before the revocation; violation = a frame sequence whose logits changed (expected 0); 'touched' = sequences on which an edit of R was released;",
                     "    'other edit' = touched sequences on which ANOTHER edit is released after the masking (top-1 among several candidates; each case is listed below)"]
            hdr = ["m", "set", "R", "frame", "violations", "touched", "touched: logits changed", "other edit released"]
            body = []
            for (m, s), f in sorted(frames.items()):
                Rset = setup["sets"][str(m)][s]
                body.append([str(m), str(s), " ".join(map(str, Rset)) if m <= 10 else f"{' '.join(map(str, Rset[:8]))} ... ({m} edits)", str(f["n_frame"]), str(f["n_violations"]),
                             str(f["n_touched"]), str(f["n_touched_logits_changed"]), str(f["n_other_edit_released"])])
            text.append(fx.io.text_table(hdr, body))
            nv = sum(f["n_violations"] for f in frames.values())
            nf = sum(f["n_frame"] for f in frames.values())
            text.append(f"   total: {nv} violations in {nf} frame sequences over {len(frames)} revocation sets")
            for (m, s), f in sorted(frames.items()):
                for v in f["violations"]:
                    text.append(f"   VIOLATION m={m} set={s}: {v}")
                for c in f["other_edit_released"]:
                    text.append(f"   other edit released, m={m} set={s}: {c['csid']} ({c['kind']}{'/' + c['metric'] if c.get('metric') else ''}, own edit {c['own']}): "
                                f"before {c['released_before']} -> after {c['released_after']}")
            paper[f"Enine{sh}FrameViolations"], paper[f"Enine{sh}FrameSequences"] = nv, nf
            doc[f"Enine{sh}FrameViolations"] = f"{name}: frame-rule violations (sequence without a released revoked edit whose logits changed) over all revocation sets (expected 0)"
            doc[f"Enine{sh}FrameSequences"] = f"{name}: number of (sequence, revocation set) pairs checked for the frame rule"
        # ------------------------------------------------------------ database vocabulary
        if A:
            text += ["", "(a') THE SAME IN DATABASE VOCABULARY.  Statements = logged edits; tuples = query answers (the kept edits' Rel / T-Gen / I-Gen sequences and the heldout_disjoint",
                     "    probes); lineage of a tuple = the statements released on it under the scoped log before the deletion (Fire(q)); a linear log has no per-tuple lineage (every",
                     "    statement contributes to every key), so the SAME tuple set is used for both logs: tuples on which no deleted statement is released.  'changed' = logits not",
                     "    bit-identical | teacher-forced answer row changed (first mismatching token or token accuracy) | reported flag changed (exact / D1)"]
            hdr = ["log", "statements deleted", "tuples outside the lineage", "changed: logits", "changed: answer row", "changed: flag", "tuples inside the lineage", "inside: logits changed"]
            body = []
            for meth in ("scoped", "linear"):
                for m, d in A[meth].items():
                    body.append([meth, str(m), fa(d["outside.n"]), fa(d["outside.n_logits_changed"]), fa(d["outside.n_pred_changed"]), fa(d["outside.n_flags_changed"]),
                                 fa(d["inside.n"]), fa(d["inside.n_logits_changed"])])
            text.append(fx.io.text_table(hdr, body))
            for meth, am in A.items():
                for m, d in am.items():
                    cm = f"Enine{sh}{'Scoped' if meth == 'scoped' else 'Linear'}M{WORD[m]}"
                    for k, nm, what in (("outside.n_logits_changed", "OutsideLogits", "tuples outside the lineage of the deleted statements whose logits changed"),
                                        ("outside.n_flags_changed", "OutsideFlags", "tuples outside the lineage of the deleted statements whose reported flag (exact / D1) changed"),
                                        ("outside.n", "OutsideN", "tuples outside the lineage of the deleted statements"),
                                        ("kept.n_flags_changed", "KeptFlags", "kept-edit sequences whose exact flag changed"),
                                        ("probe.n_flags_changed", "ProbeFlags", "heldout_disjoint probe rows whose D1 flag changed"),
                                        ("revoked.rel.exact_after", "RevokedRel", "Rel exact % of the revoked edits after the revocation"),
                                        ("kept.rel.exact_after", "KeptRel", "Rel exact % of the kept edits after the revocation"),
                                        ("kept.tgen.exact_after", "KeptTGen", "T-Gen exact % of the kept edits after the revocation"),
                                        ("kept.igen.exact_after", "KeptIGen", "I-Gen exact % of the kept edits after the revocation"),
                                        ("probe.D1_after", "DOne", "D1 % on heldout_disjoint after the revocation")):
                        if d.get(k):
                            paper[cm + nm] = round(d[k]["mean"], 2)
                            doc[cm + nm] = f"{name}, {meth} log, {m} edits revoked: {what} (mean over 3 seeded sets)"

        # ------------------------------------------------------------ (b) lineage
        pl = os.path.join(OUT, f"lineage_{name}.json")
        text += ["", "(b) LINEAGE.  Every heldout_disjoint probe and every valid near-scope negative on which the calibrated default method (draw 0) releases an edit: that edit is masked",
                 "    and the row is run again; if another edit is released it is masked too, until nothing is released.  'pre-edit after 1' = rows whose logits are bit-identical to the",
                 "    pre-edit model after masking the ONE released edit; 'another edit' = rows on which another edit is released instead (lineage size at least 2); 'masks needed' =",
                 "    number of masked edits until nothing is released; 'gate n_cand' = number of candidate edits of the row at gate level (key cache)"]
        if os.path.exists(pl):
            lin = fx.io.jload(pl)
            hdr = ["rows", "releasing", "pre-edit after 1 (%)", "another edit (%)", "masks needed: mean", "max", "1 / 2 / 3 / 4+", "= gate n_cand", "final logits = pre-edit"]
            body = []
            for lab, sel in (("heldout_disjoint probes", lambda r: r["kind"] == "probe"), ("HN-I", lambda r: r.get("hn_type") == "HN-I"), ("HN-Q", lambda r: r.get("hn_type") == "HN-Q"),
                             ("HN-X", lambda r: r.get("hn_type") == "HN-X"), ("all near-scope negatives", lambda r: r["kind"] == "hn"), ("all", lambda r: True)):
                sub = [r for r in lin["rows"] if sel(r)]
                if not sub:
                    body.append([lab, "0"] + ["-"] * 7)
                    continue
                cnt = [len(r["masked"]) for r in sub]
                one = [len(r["chain"]) >= 2 and not r["chain"][1]["released"] and r["chain"][1]["logit_sha1"] == bp[r["csid"]]["logit_sha1"] for r in sub]
                oth = [len(r["chain"]) >= 2 and bool(r["chain"][1]["released"]) for r in sub]
                fin = [r["chain"][-1]["logit_sha1"] == bp[r["csid"]]["logit_sha1"] for r in sub]
                eqg = [len(r["masked"]) == r.get("n_cand_gate_level") for r in sub]
                body.append([lab, str(len(sub)), f"{pct(one):.1f}", f"{pct(oth):.1f}", f"{np.mean(cnt):.2f}", str(max(cnt)),
                             " / ".join(str(sum(c == k for c in cnt)) for k in (1, 2, 3)) + f" / {sum(c >= 4 for c in cnt)}", f"{sum(eqg)}/{len(sub)}", f"{sum(fin)}/{len(sub)}"])
                key = {"heldout_disjoint probes": "probe", "all near-scope negatives": "hn", "all": "all"}.get(lab, lab)
                kw = dict(draw=0, alpha=0.05, n=200, pool=key, n_units=len(sub))
                R.add(name, "scoped", "lineage.pre_edit_after_one_mask", pct(one), **kw)
                R.add(name, "scoped", "lineage.another_edit_released", pct(oth), **kw)
                R.add(name, "scoped", "lineage.masks_needed.mean", float(np.mean(cnt)), **kw)
                R.add(name, "scoped", "lineage.masks_needed.max", max(cnt), **kw)
                R.add(name, "scoped", "lineage.n_final_bit_identical_preedit", sum(fin), **kw)
                R.add(name, "scoped", "lineage.n_masks_equal_gate_n_cand", sum(eqg), **kw)
                if key in ("probe", "hn", "all"):
                    cm = f"Enine{sh}Lineage{ {'probe': 'Probe', 'hn': 'Near', 'all': 'All'}[key]}"
                    paper[cm + "N"], paper[cm + "OneMask"], paper[cm + "Mean"], paper[cm + "Max"] = len(sub), round(pct(one), 1), round(float(np.mean(cnt)), 2), max(cnt)
                    doc[cm + "N"] = f"{name}: {lab} on which the calibrated default method (draw 0) releases an edit"
                    doc[cm + "OneMask"] = f"{name}: % of these rows that are bit-identical to the pre-edit model after masking the one released edit"
                    doc[cm + "Mean"] = f"{name}: mean number of masked edits until nothing is released on these rows (lineage size)"
                    doc[cm + "Max"] = f"{name}: largest number of masked edits until nothing is released on these rows"
            text.append(fx.io.text_table(hdr, body))
            text.append(f"   {lin['n_forwards']} forwards in {lin['n_rounds']} rounds")
        else:
            text.append("   pending")
        lr = linear_records(name)
        text += ["", "    LINEAR LOG, from existing records (read, not recomputed): number of edits k that a rollback removes for ONE complaint"]
        hdr = ["source", "rule", "complaints", "resolved", "k median", "k mean", "k min-max", "k = 1", "edits broken / repair"]
        body = []
        for tag, cite in (("p12_text", "P12 text head-to-head"), ("rq1_image", "rq1 (image complaints)")):
            for cfg, rule in (("RB_BEH", "behavioural greedy (budget 20)"), ("RB_FO", "first-order greedy (budget 10)")):
                d = lr.get(f"{tag}.{cfg}")
                if not d or not d["n_complaints"]:
                    continue
                body.append([cite, rule, str(d["n_complaints"]), str(d["n_resolved"]), f"{d['k_median']:.1f}", f"{d['k_mean']:.2f}", f"{d['k_min']}-{d['k_max']}", str(d["n_k_is_1"]),
                             "-" if d["edits_broken_mean"] is None else f"{d['edits_broken_mean']:.2f}"])
                for k in ("n_complaints", "n_resolved", "k_median", "k_mean", "k_min", "k_max"):
                    R.add(name, "linear", f"lineage.rollback.{tag}.{cfg}.{k}", d[k], pool="repair complaints", source=lr["sources"][tag])
                if cfg == "RB_BEH":
                    cm = f"Enine{sh}Rollback{'Text' if tag == 'p12_text' else 'Image'}"
                    paper[cm + "KMedian"], paper[cm + "KMean"], paper[cm + "N"] = d["k_median"], round(d["k_mean"], 2), d["n_complaints"]
                    doc[cm + "KMedian"] = f"{name}: median number of edits the behavioural greedy rollback removes for one complaint ({d['what']}; read from {lr['sources'][tag]})"
                    doc[cm + "KMean"] = f"{name}: mean number of edits the behavioural greedy rollback removes for one complaint ({d['what']}; read from {lr['sources'][tag]})"
                    doc[cm + "N"] = f"{name}: number of complaints behind the rollback sizes ({d['what']})"
        text.append(fx.io.text_table(hdr, body))
        for c, tol in (("c0.1", "0.1"), ("c0.5", "0.5")):
            d = lr[f"e5.{c}"]
            text.append(f"    layer-level optimum (E5, tolerance {tol}; {d['opt_n_complaints']} complaints, {d['opt_n_certified']} certified): median optimum between {d['opt_median_lower_bound']} and "
                        f"{d['opt_median_upper_bound']} edits (certified cases: median {d['opt_median_certified']}); best greedy rule median {d['median_best_rule']}; "
                        f"lower bound of the proposition median {d['median_lb']} (min {d['min_lb']}, max {d['max_lb']})")
            for k, v in d.items():
                R.add(name, "linear", f"lineage.rollback.e5.{c}.{k}", v, pool="repair complaints", source=E5_PN)
            cm = f"Enine{sh}LayerOpt{'TolTenth' if c == 'c0.1' else 'TolHalf'}"
            paper[cm + "Lo"], paper[cm + "Hi"] = d["opt_median_lower_bound"], d["opt_median_upper_bound"]
            doc[cm + "Lo"] = f"{name}: median lower bound of the layer-level minimum rollback for one complaint at tolerance {tol} (read from {E5_PN}, key rollback.{name}.{c}.opt_median_lower_bound)"
            doc[cm + "Hi"] = f"{name}: median upper bound of the layer-level minimum rollback for one complaint at tolerance {tol} (read from {E5_PN}, key rollback.{name}.{c}.opt_median_upper_bound)"
        text.append(f"    files: {lr['sources'].get('p12_text')} (results: {P12_MD}, section 9); {lr['sources'].get('rq1_image')} (results: {RQ1_MD}); {E5_PN}")
        import glob
        jobs = sorted(glob.glob(os.path.join(OUT, f"job_{name}_*.json")))
        done = set()
        for jp in jobs:
            j = fx.io.jload(jp)
            done |= set(j["stages"])
            text.append(f"   job record, stages {' '.join(j['stages'])}: assert_clean {j['assert_clean']} (hook_mismatch and memo conflicts must be 0); hashed forwards {j['stats'].get('hashed')}")
        miss = [x for x in ["base"] + [f"m{m}" for m in MS] if x not in done]
        if miss:
            text.append(f"   stages without a finished job record: {' '.join(miss)} (the tables above hold what is complete)")
    R.save(alpha=0.05, n=200, draw=0, ms=list(MS), n_sets=N_SETS)
    if os.path.exists(R.path("paper_numbers.json")):
        os.remove(R.path("paper_numbers.json"))
    R.paper(paper)
    fx.io.jdump(dict(sorted(doc.items())), R.path("paper_numbers_doc.json"), indent=1)
    R.table("\n".join(text))
    print(f"[e9_render] {len(R.records)} records, {len(paper)} paper numbers -> {R.path('table.txt')}")


if __name__ == "__main__":
    main()
