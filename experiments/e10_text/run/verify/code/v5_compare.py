"""E10 verification (CPU): the verifier's recomputation (blocks 1-3) against the experiment's per-row files and reported values.

    python v5_compare.py  ->  out/compare.json
(A) behaviour, per log and method at calibration draw 0: row by row on the verifier's sample (same prompt): D1 (teacher-forced,
    decode), exact match of the new object, firing, first token of the decode; then the aggregates: verifier's value on its sample,
    the experiment's value on the SAME rows (from its per-row file), the experiment's value on ALL its rows at draw 0, and the value
    the experiment's summary reports (mean over draws 0-2).
(B) public scores: probabilities of the verifier's case files against the experiment's case files of the same record; ES / PS / NS
    of the public summariser on the verifier's files against the experiment's arithmetic on the same records.
(C) certificate: counts of certified prompts at draw 0 against the experiment's out/cert/<log>.json.
(D) equal evidence (T100, n = 10, seed 0): row by row and aggregates against out/rows_equal, repair records against out/equal.
(E) method records at draw 0 (eta, capped edits, own requests silenced) against the experiment's out/behave records.
"""
import os

import numpy as np

import vcommon as V

OUT = {"behaviour": {}, "public": {}, "certificate": {}, "equal": {}, "records": {}}
XTAG = {"deployed": "deployed__dx", "preedit": "preedit__dx", "where_twoblock": "where_twoblock__d0", "rescope": "rescope__d0",
        "where_twoblock@goff": "where_twoblock@goff__d0", "rescope@goff": "rescope@goff__d0"}
XFIELD = {"D1_tf": "D1_tf", "D1_dec": "D1_dec", "new_exact": "new_exact", "fired": "fired", "fired_dec": "fired_dec", "first_changed_tf": "first_changed_tf",
          "dec_first": "dec_first", "kept_tf": "tok_loc", "own_released": "own_released", "own_released_dec": "own_released_dec"}


def mean100(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) is not None]
    return 100.0 * float(np.mean(v)) if v else None


def metric_rows(rows, metric, split=None):
    """(subset, my field) of a metric name such as 'edit.request.exact.before' / 'heldout.D1_tf' / 'neighborhood.fired'."""
    p = metric.split(".")
    seg = p[-1] if p[-1] in ("before", "from") else None
    if seg:
        p = p[:-1]
    if p[0] == "edit":
        sub = [r for r in rows if r["set"] == "stream" and r["kind"] == p[1]]
        fld = {"exact": "new_exact", "own_released": "own_released", "own_released_dec": "own_released_dec"}[p[2]]
    else:
        sub = [r for r in rows if (r["set"] == "heldout" if p[0] == "heldout" else r["kind"] == "neighborhood") and r["valid"]]
        fld = p[1]
    if seg == "before":
        sub = [r for r in sub if r["pos"] < V.SPLIT]
    if seg == "from":
        sub = [r for r in sub if r["pos"] >= V.SPLIT]
    return sub, fld


for log in (V.T100, V.T500):
    collapse = log == V.T500
    xsum = V.jload(V.XOUT + f"/summary/behave_{log}.json")
    OUT["behaviour"][log] = {}
    for lab, xtag in XTAG.items():
        mp = os.path.join(V.OUT, "behave", log, lab + ".json")
        if not os.path.exists(mp):
            continue
        mine = V.load_rows(os.path.join(V.OUT, "rows", log, lab + ".jsonl.gz"))
        xall = V.load_rows(V.XOUT + f"/rows/{log}/{xtag}.jsonl.gz")
        xby = {r["prompt_id"]: r for r in xall}
        o = {"n_rows_mine": len(mine), "n_rows_found_in_experiment": sum(r["prompt_id"] in xby for r in mine), "row_level_differences": {}, "aggregates": {}}
        for k, xk in XFIELD.items():
            pairs = [(r[k], xby[r["prompt_id"]].get(xk)) for r in mine if r.get(k) is not None and r["prompt_id"] in xby and xby[r["prompt_id"]].get(xk) is not None]
            if pairs:
                nd = sum((abs(float(a) - float(b)) > 1e-9) if k == "kept_tf" else (a != b) for a, b in pairs)
                o["row_level_differences"][k] = {"n_compared": len(pairs), "n_differ": nd}
        mets = ["edit.request.exact", "edit.paraphrase.exact", "heldout.D1_tf", "heldout.kept_tf", "heldout.first_changed_tf", "neighborhood.D1_tf", "heldout.D1_dec"]
        if mine[0].get("fired") is not None:
            mets += ["heldout.fired", "neighborhood.fired", "edit.paraphrase.own_released"]
        if collapse:
            mets += [m + s for m in ("edit.request.exact", "edit.paraphrase.exact", "neighborhood.D1_tf") for s in (".before", ".from")]
        xkey = lab
        for m in mets:
            sub, fld = metric_rows(mine, m)
            xsub_same = [dict(xby[r["prompt_id"]], valid=True) for r in sub if r["prompt_id"] in xby]
            xs, xf = metric_rows(xall, m)
            xfld = XFIELD.get(fld, fld)
            rep = (xsum.get(xkey, {}).get("metrics", {}).get(m.replace("kept_tf", "kept_tf")) or {})
            pd = (xsum.get(xkey, {}).get("per_draw", {}).get(m) or {})
            e = {"mine_on_sample": mean100(sub, fld), "n_sample": len(sub), "experiment_on_same_rows": mean100(xsub_same, xfld),
                 "experiment_all_rows_draw0_from_its_rows": mean100(xs, xfld), "n_all": len(xs),
                 "experiment_summary_value_mean_over_draws": rep.get("value"), "experiment_summary_ci": [rep.get("ci_lo"), rep.get("ci_hi")],
                 "experiment_summary_draw0": pd.get("0")}
            if e["mine_on_sample"] is not None and e["experiment_on_same_rows"] is not None:
                e["abs_diff_same_rows"] = abs(e["mine_on_sample"] - e["experiment_on_same_rows"])
            o["aggregates"][m] = e
        st = V.jload(mp)
        o["my_stats_with_ci"] = {k: v for k, v in st["stats"].items() if isinstance(v, dict) and v.get("value") is not None and k in
                                 ("edit.request.exact", "edit.paraphrase.exact", "heldout.D1_tf", "neighborhood.D1_tf", "heldout.fired", "edit.request.exact.before", "edit.request.exact.from",
                                  "edit.paraphrase.exact.before", "edit.paraphrase.exact.from")}
        o["silent_prompts"] = {k: st["stats"].get(k) for k in ("silent_tf.n", "silent_tf.n_D1", "silent_dec.n", "silent_dec.n_D1")}
        o["max_abs_diff_same_rows"] = max([e.get("abs_diff_same_rows", 0.0) for e in o["aggregates"].values()], default=None)
        OUT["behaviour"][log][lab] = o
        # (E) method record
        if st["info"]["kind"] == "gate":
            xm = V.jload(V.XOUT + f"/behave/{log}/{xtag}.json")
            xc = xm["record"].get("cal", {})
            xo = xm["record"].get("own_requests", {})
            my = st["info"]["rec"]
            OUT["records"][f"{log}|{lab}"] = {"eta": [my["eta"], xc.get("eta")], "n_edits_capped": [my["n_edits_capped"], xc.get("n_edits_capped")],
                                              "own_requests_not_candidate": [my["own_requests"]["n_not_candidate"], xo.get("n_not_candidate")],
                                              "thresholds_sha1": [my["thresholds"]["sha1"], xm["record"].get("thresholds", {}).get("sha1")],
                                              "method_fingerprint": [my["fingerprint"], xm.get("key")], "calibration_in_REPAIR_pool": st["info"]["calibration_prompts_all_in_REPAIR_pool"],
                                              "alpha_n_draw": [st["info"]["alpha"], st["info"]["n"], st["info"]["draw"]], "experiment_alpha_n_draw": [xm.get("alpha"), xm.get("n"), xm.get("draw")],
                                              "same_gate": bool(my["thresholds"]["sha1"] == xm["record"].get("thresholds", {}).get("sha1"))}
    # (B) public
    OUT["public"][log] = {}
    for lab in ("deployed", "where_twoblock", "where_twoblock@goff"):
        pp = os.path.join(V.OUT, "behave", log, lab + ".public.json")
        if not os.path.exists(pp):
            continue
        mp_ = V.jload(pp)
        xdir = V.XOUT + f"/public/{log}/{XTAG[lab]}/run_000"
        mdir = mp_["case_files"]
        nrec, maxd, nflip, ndiff_rec = 0, 0.0, 0, 0
        es, ps, ns, pos = [], [], [], []
        for f in sorted(os.listdir(mdir)):
            a, b = V.jload(os.path.join(mdir, f))["post"], V.jload(os.path.join(xdir, f))["post"]
            nrec += 1
            d = 0.0
            for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs"):
                for x, y in zip(a[key], b[key]):
                    d = max(d, abs(x["target_new"] - y["target_new"]), abs(x["target_true"] - y["target_true"]))
                    nflip += int((x["target_true"] > x["target_new"]) != (y["target_true"] > y["target_new"]))
            maxd = max(maxd, d)
            ndiff_rec += int(d > 0)
            es.append(float(np.mean([x["target_true"] > x["target_new"] for x in b["rewrite_prompts_probs"]])))
            ps.append(float(np.mean([x["target_true"] > x["target_new"] for x in b["paraphrase_prompts_probs"]])))
            ns.append(float(np.mean([x["target_true"] < x["target_new"] for x in b["neighborhood_prompts_probs"]])))
            pos.append(int(f[5:-5]))
        xpub = V.jload(V.XOUT + f"/behave/{log}/{XTAG[lab]}.public.json")
        o = {"n_records": nrec, "max_abs_probability_difference": maxd, "n_records_with_any_difference": ndiff_rec, "n_comparisons_flipped": nflip,
             "public_summariser_on_my_files": mp_["public_summariser"]["all"], "my_arithmetic_on_my_files": {k: mp_["own_arithmetic"]["all"][k]["value"] for k in ("ES", "PS", "NS")},
             "experiment_files_same_records": {"ES": 100 * float(np.mean(es)), "PS": 100 * float(np.mean(ps)), "NS": 100 * float(np.mean(ns))},
             "experiment_reported_draw0_all_records": {k: xpub["all"][k] for k in ("ES", "PS", "NS")}, "experiment_n_records": xpub["all"]["n"]}
        if collapse:
            pos = np.array(pos)
            for nm, msk in (("before", pos < V.SPLIT), ("from", pos >= V.SPLIT)):
                o[nm] = {"public_summariser_on_my_files": {k: mp_["public_summariser"][nm][k] for k in ("ES", "PS", "NS", "n")},
                         "experiment_files_same_records": {"ES": 100 * float(np.mean(np.array(es)[msk])), "PS": 100 * float(np.mean(np.array(ps)[msk])), "NS": 100 * float(np.mean(np.array(ns)[msk]))},
                         "experiment_reported_draw0_all_records": {k: xpub[nm][k] for k in ("ES", "PS", "NS", "n")}}
        o["agree"] = bool(all(abs(o["public_summariser_on_my_files"][k] - o["experiment_files_same_records"][k]) < 0.006 for k in ("ES", "PS", "NS")))
        OUT["public"][log][lab] = o
    # (C) certificate
    cp = os.path.join(V.OUT, "cert", log + ".json")
    if os.path.exists(cp):
        mc, xc = V.jload(cp), V.jload(V.XOUT + f"/cert/{log}.json")
        OUT["certificate"][log] = {"experiment_total": xc["total"], "rows": {}}
        for lab, o in mc["rows"].items():
            xr = xc["rows"].get(XTAG[lab], {})
            OUT["certificate"][log]["rows"][lab] = {"n_certified": [o["n_certified"], xr.get("n_certified")], "n_checked": o["n_checked"], "n_bit_identical": o["n_logits_bit_identical"],
                                                    "n_violations": o["n_violations"], "n_checked_not_silent_by_own_arithmetic": o["n_checked_not_silent_by_own_arithmetic"],
                                                    "n_hook_fired": o["n_hook_fired"], "n_tokens_equal_stored_continuation": o["n_tokens_equal_stored_continuation"]}
        # the experiment's total recomputed from its per-row counts
        OUT["certificate"][log]["experiment_total_recomputed"] = {"n_checked_certified": sum(r["n_certified"] for r in xc["rows"].values()),
                                                                  "n_equal_certified": sum(r["n_equal"] for r in xc["rows"].values()),
                                                                  "n_violations": sum(r["n_violations_certified"] + r["n_violations_in_band"] for r in xc["rows"].values()),
                                                                  "n_rows": len(xc["rows"]), "rows_with_zero_certified": sorted(k for k, r in xc["rows"].items() if r["n_certified"] == 0)}

# (D) equal evidence
log = V.T100
ed = os.path.join(V.OUT, "equal", log)
if os.path.isdir(ed):
    OUT["equal"]["complaints"] = V.jload(os.path.join(ed, "_complaints.json"))
    xsum = V.jload(V.XOUT + f"/summary/equal_{log}.json")
    for rep in ("rb_suffix", "least_sacrificing", "least_protecting"):
        mp = os.path.join(ed, f"{rep}__n10_s0.json")
        if not os.path.exists(mp):
            continue
        st = V.jload(mp)
        mine = V.load_rows(os.path.join(V.OUT, "rows_equal", log, f"{rep}__n10_s0.jsonl.gz"))
        xall = V.load_rows(V.XOUT + f"/rows_equal/{log}/{rep}__n10_s0.jsonl.gz")
        xm = V.jload(V.XOUT + f"/equal/{log}/{rep}__n10_s0.json")
        xby = {(r["prompt_id"], r.get("part") == "extra"): r for r in xall}
        o = {"row_level_differences": {}, "aggregates": {}, "my_meta": st["meta"], "experiment_meta": {k: xm["meta"].get(k) for k in ("statements_changed", "repair_passes", "edits_lost", "complaints_refused")},
             "experiment_repair": {k: v for k, v in (xm["meta"].get("repair") or {}).items() if k in ("t_star", "n_forward", "n_thresholds_raised", "n_refused", "resolved")}}
        for k, xk in (("D1_tf", "D1_tf"), ("D1_dec", "D1_dec"), ("new_exact", "new_exact"), ("fired", "fired")):
            pairs = [(r[k], xby[(r["prompt_id"], r["part"] == "complaint")].get(xk)) for r in mine
                     if r.get(k) is not None and (r["prompt_id"], r["part"] == "complaint") in xby and xby[(r["prompt_id"], r["part"] == "complaint")].get(xk) is not None]
            o["row_level_differences"][k] = {"n_compared": len(pairs), "n_differ": sum(a != b for a, b in pairs)}
        o["n_rows_compared"] = sum((r["prompt_id"], r["part"] == "complaint") in xby for r in mine)
        o["rows_with_a_difference"] = [r["prompt_id"] + (" (reported complaint)" if r["part"] == "complaint" else "") for r in mine if (r["prompt_id"], r["part"] == "complaint") in xby and any(
            r.get(k) is not None and xby[(r["prompt_id"], r["part"] == "complaint")].get(k) is not None and r[k] != xby[(r["prompt_id"], r["part"] == "complaint")][k] for k in ("D1_tf", "D1_dec", "new_exact", "fired"))]
        xs = lambda pred, fld: mean100([r for r in xall if pred(r)], fld)      # noqa: E731
        o["aggregates"] = {
            "request.exact": {"mine": st["stats"]["request.exact"]["value"], "experiment_seed0_from_its_rows": xs(lambda r: r["kind"] == "request" and r["set"] == "stream" and r.get("part") is None, "new_exact"),
                              "experiment_summary_mean_over_seeds": xsum[f"{rep}|n10"]["metrics"]["edit.request.exact"]["value"]},
            "heldout.D1_tf": {"mine_on_sample": st["stats"]["heldout.D1_tf"]["value"], "mine_ci": [st["stats"]["heldout.D1_tf"]["ci_lo"], st["stats"]["heldout.D1_tf"]["ci_hi"]],
                              "experiment_seed0_all_rows_from_its_rows": xs(lambda r: r["set"] == "heldout" and r.get("part") is None, "D1_tf"),
                              "experiment_seed0_same_rows": mean100([xby[(r["prompt_id"], False)] for r in mine if r["set"] == "heldout" and r["part"] is None], "D1_tf"),
                              "experiment_summary_mean_over_seeds": xsum[f"{rep}|n10"]["metrics"]["heldout.D1_tf"]["value"]},
            "complaints.D1_tf": {"mine": st["stats"]["complaints.D1_tf"]["value"], "experiment_seed0": xs(lambda r: r.get("part") == "extra", "D1_tf")},
            "complaints.D1_dec": {"mine": st["stats"]["complaints.D1_dec"]["value"], "experiment_seed0": xs(lambda r: r.get("part") == "extra", "D1_dec")}}
        OUT["equal"][rep] = o
V.jdump(OUT, V.OUT + "/compare.json")

# ----------------------------------------------------------------------------- short print
for log, d in OUT["behaviour"].items():
    for lab, o in d.items():
        print(log, lab, "rows", o["n_rows_mine"], "diffs", {k: v["n_differ"] for k, v in o["row_level_differences"].items()}, "max|agg diff| same rows", o["max_abs_diff_same_rows"])
        for m, e in o["aggregates"].items():
            print(f"    {m:32s} mine {e['mine_on_sample']}  exp same rows {e['experiment_on_same_rows']}  exp all d0 {e['experiment_all_rows_draw0_from_its_rows']}  exp summary {e['experiment_summary_value_mean_over_draws']}")
for log, d in OUT["public"].items():
    for lab, o in d.items():
        print("public", log, lab, {k: o[k] for k in ("n_records", "max_abs_probability_difference", "n_comparisons_flipped", "public_summariser_on_my_files", "experiment_files_same_records", "experiment_reported_draw0_all_records", "agree")})
print("cert", OUT["certificate"])
print("records", OUT["records"])
for rep in ("rb_suffix", "least_sacrificing", "least_protecting"):
    if rep in OUT["equal"]:
        print("equal", rep, OUT["equal"][rep]["row_level_differences"], OUT["equal"][rep]["aggregates"], OUT["equal"][rep]["my_meta"], OUT["equal"][rep]["experiment_meta"], OUT["equal"][rep]["experiment_repair"])
