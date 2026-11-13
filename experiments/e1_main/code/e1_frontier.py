"""E1b (frontier tracing) and E1c (matched-coverage read-out): gate level, key caches, no model; the two development logs.

    python e1_frontier.py            -> out/frontier.json (curves + calibrated points; a figure needs no recomputation)

Global-threshold frontiers ("candidate iff feature >= t, top-1 by the feature, no guards"): the normalised solver coordinate
('ncoord'), the raw cosine ('cos') and the detector set of every variant of the cosine family (cos_centred, cos_null at calibration
draw 0, white / whitecos at kappa 0.01, 0.1, 1); 501 thresholds over the whole range.  Retrieval: the calibrated rule at alpha = 0.05,
calibration draw 0, with a common shift of its attribute scales (retr_conj: both attribute thresholds; also the selected rule).
Axes: false firing on heldout_disjoint and mean firing on the TEST hard negatives (odd edits; mean over the three kinds).
Calibrated points: mean over the 20 calibration draws, per alpha and calibration rule (gate cells of e1_gate.py + the firing on the
test hard negatives recomputed here from the same gates).
"""
import os
import time

from e1c import *  # noqa: F401,F403
import e1c

R1 = fx.review1
FEATS = [("ncoord", None, ("rescope", "rescope_shared")), ("cos", None, ("cosine", "cosine_shared"))] + \
        [(b, b, (b, b + "_shared")) for b in ("cos_centred", "cos_null", "white_0.01", "white_0.1", "white_1", "whitecos_0.01", "whitecos_0.1", "whitecos_1")]
KEEP = ("probe.heldout_disjoint.fired", "probe.heldout_disjoint_image.fired", "probe.heldout_disjoint_text.fired", "probe.heldout_strict.fired",
        "edit.tgen.own_top", "edit.igen.own_top", "edit.rephrase.own_top", "edit.rel.own_top", "edit.rephrase.other_top",
        "hn.HN-I.fired", "hn.HN-Q.fired", "hn.HN-X.fired", "hn.all.fired")


def test_axis(L, sw, ts, sp):
    pos = {j: i for i, j in enumerate(sw["parts"]["hn"]["rows"])}
    rm = sw["parts"]["hn"]["rmax"]
    out = {}
    for k in HN:
        v = np.sort(rm[[pos[j] for j in sp["test"][k]]])
        out[f"hn.test.{k}.fired"] = 100.0 * (v.size - np.searchsorted(v, ts, side="left")) / max(v.size, 1)
    out["hn.test.mean.fired"] = np.mean([out[f"hn.test.{k}.fired"] for k in HN], axis=0)
    return out


def test_fired(L, M, sp):
    per = {}
    for k in HN:
        d = fx.gate.decide(L, M, sp["test"][k], "tf")
        per[k] = 100.0 * float(np.asarray(d["fired"]).mean())
    return dict(per, mean=float(np.mean(list(per.values()))))


def mc_slim(mc):
    out = []
    for m in mc:
        o = {"target": m["target"], "reachable": m["reachable"]}
        if m["reachable"]:
            r = m["readout"]
            o.update(threshold=m["threshold"], coverage=m["coverage"], heldout_disjoint_fired=r["probe"]["heldout_disjoint"]["fired"],
                     heldout_strict_fired=r["probe"]["heldout_strict"]["fired"], tgen_coverage=r["edit"]["tgen"]["own_top"], igen_coverage=r["edit"]["igen"]["own_top"],
                     hn_fired={k: r["hn"][k]["fired"] for k in HN}, hn_own_top={k: r["hn"][k]["own_top"] for k in HN},
                     hn_mean_fired=float(np.mean([r["hn"][k]["fired"] for k in HN])))
        else:
            o["largest_coverage"] = m.get("largest_coverage")
        out.append(o)
    return out


def main():
    t0 = time.time()
    res = {"experiment": "E1b/E1c", "what": __doc__, "code": fx.code_fingerprint(), "logs": {}}
    for log in DEV:
        L = fx.logs.open(log)
        sp = fx.sweeps.h3_split(L)
        sel = e1c.selected_labels(L)
        out = {"selection": sel, "n_test_hn": sp["n"]["test"], "features": {}, "retrieval": {}, "matched_coverage": {}}
        for feat, name, cal_labels in FEATS:
            sw = fx.gate.sweep(L, feat, parts=("probe", "edit", "hn")) if name is None else \
                fx.gate.sweep(L, name, parts=("probe", "edit", "hn"), gate=R1.unthresholded(L, name, draw=0))
            fr = fx.gate.frontier(L, sw=sw, n_grid=501)
            ts = fr["thresholds"]
            curves = {k: fr["curves"][k] for k in KEEP if k in fr["curves"]}
            curves.update(test_axis(L, sw, ts, sp))
            # calibrated points: mean over the 20 draws per alpha and rule
            points = {}
            for lab in cal_labels:
                rule = "shared" if lab.endswith("_shared") else "two_block"
                for alpha in ALPHAS:
                    b = jload(os.path.join(GATE, log, f"{lab}__a{afmt(alpha)}.json"))
                    cs = [b["cells"][str(d)] for d in range(N_DRAWS)]
                    tf_ = [test_fired(L, R1.e1_rowset_review1(L, alpha, d, labels=(lab,))[0], sp) for d in range(N_DRAWS)]
                    mean = lambda f: float(np.mean([f(c) for c in cs]))                                        # noqa: E731
                    points[f"{rule}@{afmt(alpha)}"] = {
                        "method": lab, "rule": rule, "alpha": alpha, "n_draws": N_DRAWS,
                        "probe.heldout_disjoint.fired": mean(lambda c: c["st"]["probe"]["heldout_disjoint"]["fired"]),
                        "probe.heldout_disjoint.fired_literal": mean(lambda c: c["literal"]["disjoint.all"]["fired"]),
                        "edit.tgen.own_top": mean(lambda c: c["st"]["edit"]["tgen"]["own_top"]), "edit.igen.own_top": mean(lambda c: c["st"]["edit"]["igen"]["own_top"]),
                        "edit.rephrase.own_top": mean(lambda c: c["h2_cov"]),
                        **{f"hn.{k}.fired": mean(lambda c, k=k: c["st"]["hn"][k]["fired"]) for k in HN},
                        **{f"hn.test.{k}.fired": float(np.mean([t[k] for t in tf_])) for k in HN},
                        "hn.test.mean.fired": float(np.mean([t["mean"] for t in tf_])), "n_capped": mean(lambda c: c["n_capped"]),
                        "eta": mean(lambda c: c["cal"]["eta"])}
            out["features"][feat] = {"feature": feat, "detectors": "fx.methods.base_gate" if name is None else f"fx.review1.unthresholded({name}, draw=0)",
                                     "thresholds": ts, "curves": curves, "n": fr["n"], "calibrated_points": points}
            out["matched_coverage"][feat] = mc_slim(fx.gate.matched_coverage(L, sw=sw, targets=(25, 50, 75)))
            # at the matched-coverage thresholds: the mean firing on the test hard negatives as well
            for m in out["matched_coverage"][feat]:
                if m["reachable"]:
                    ta = test_axis(L, sw, np.array([m["threshold"]]), sp)
                    m["hn_test_fired"] = {k: float(ta[f"hn.test.{k}.fired"][0]) for k in HN}
                    m["hn_test_mean_fired"] = float(ta["hn.test.mean.fired"][0])
            print(f"[{time.time() - t0:6.0f}s] {log} {feat}", flush=True)
        # retrieval: common shift of the attribute scales of the calibrated rule (draw 0, alpha 0.05) + calibrated points over the draws
        deltas = np.linspace(-1.5, 1.5, 301)
        for rule in dict.fromkeys(("retr_conj", sel["retr_H3"], sel["retr_H2"])):
            M = fx.methods.build(rule, L, alpha=0.05, draw=0)
            cv = fx.sweeps.shift_curves(L, M, deltas)
            keep = {k: v for k, v in cv["curves"].items() if k in KEEP or k.startswith("hn.test.")}
            pts = {}
            for alpha in ALPHAS:
                b = jload(os.path.join(GATE, log, f"{rule}__a{afmt(alpha)}.json"))
                cs = [b["cells"][str(d)] for d in range(N_DRAWS)]
                tf_ = [test_fired(L, fx.methods.build(rule, L, alpha=alpha, draw=d), sp) for d in range(N_DRAWS)]
                mean = lambda f: float(np.mean([f(c) for c in cs]))                                            # noqa: E731
                pts[f"two_block@{afmt(alpha)}"] = {
                    "method": rule, "rule": "two_block", "alpha": alpha, "n_draws": N_DRAWS,
                    "probe.heldout_disjoint.fired": mean(lambda c: c["st"]["probe"]["heldout_disjoint"]["fired"]),
                    "probe.heldout_disjoint.fired_literal": mean(lambda c: c["literal"]["disjoint.all"]["fired"]),
                    "edit.tgen.own_top": mean(lambda c: c["st"]["edit"]["tgen"]["own_top"]), "edit.igen.own_top": mean(lambda c: c["st"]["edit"]["igen"]["own_top"]),
                    "edit.rephrase.own_top": mean(lambda c: c["h2_cov"]), **{f"hn.{k}.fired": mean(lambda c, k=k: c["st"]["hn"][k]["fired"]) for k in HN},
                    **{f"hn.test.{k}.fired": float(np.mean([t[k] for t in tf_])) for k in HN}, "hn.test.mean.fired": float(np.mean([t["mean"] for t in tf_])),
                    "n_capped": mean(lambda c: c["n_capped"])}
            out["retrieval"][rule] = {"rule": rule, "x_axis": "common shift delta of the calibrated scale(s) (alpha = 0.05, calibration draw 0; guards kept)", "deltas": deltas,
                                      "curves": keep, "n_edits_at_cap": cv["n_edits_at_cap"], "calibrated_points": pts}
            print(f"[{time.time() - t0:6.0f}s] {log} {rule}", flush=True)
        res["logs"][log] = out
    res["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(res, os.path.join(OUT, "frontier.json"))
    print("DONE")


if __name__ == "__main__":
    main()
