"""E3 / E4 gate level (CPU, key caches): one log per process.     python e3_gate.py <log> [--labels a,b] [--readouts reg,qd]

out/gate/<log>/<label>__<readout>.json = {"cells": {draw: cell}} with ALL 20 calibration draws (resumable: a finished file is skipped).
Read-out reg: the registered one (calibration draw from the REPAIR pool).  Read-out qd (R8): calibration draw from the REPAIR rows whose
question string is not a question or rephrase of any edit of the longest stream of the backbone, test rows filtered alike (probes, BEN-T).
out/gate/<log>/_factor.json: factor level (rebase error, self-separation margins, non-oracle targets, description of the log).
"""
import argparse
import dataclasses
import os
import time

from e3c import *  # noqa: F401,F403
import e3c
import torch

R1 = fx.review1
KEEP_CAL = ("op", "rule", "alpha", "n", "n1", "n2", "rank", "eta", "eta_infinite", "bound_false_firing", "n_edits_capped", "n_detectors_capped",
            "n_own_keys_silenced_by_literal_rule", "n_guards_silenced_by_literal_rule", "n_edits_silenced_by_literal_rule",
            "n_own_requests_silenced_by_literal_rule", "n_own_keys_silenced", "n_guards_silenced", "guarantee", "threshold")


def slim_cal(M):
    c = M.rec.get("cal") or {}
    out = {k: c[k] for k in KEEP_CAL if k in c and not isinstance(c[k], list)}
    if isinstance(c.get("pac"), dict):
        out["pac"] = {k: v for k, v in c["pac"].items() if not isinstance(v, (list, dict))}
    S = getattr(M, "obj", None)
    try:                                                             # thresholds summary (scoped gates: per-detector thresholds)
        th = S.theta() if hasattr(S, "theta") and callable(S.theta) else getattr(S, "det_tau", None)
        def summ(x):
            t = np.asarray(torch.as_tensor(x).double().flatten().numpy())
            f = t[np.isfinite(t)]
            return {"n": int(t.size), "n_finite": int(f.size), "min": float(f.min()) if f.size else None, "median": float(np.median(f)) if f.size else None,
                    "max": float(f.max()) if f.size else None}
        if isinstance(th, dict):
            out["thresholds"] = {str(k): summ(v) for k, v in th.items() if hasattr(v, "__len__")}
        elif th is not None:
            out["thresholds"] = summ(th)
    except Exception as e:                                           # a summary only; never fatal
        out["thresholds"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


def slim_h3(h):
    if h is None:
        return None
    o = {k: h.get(k) for k in ("reachable", "delta", "n_edits_at_cap", "n_edits_at_cap_at_0", "limit", "guards", "n_fit", "n_test")}
    o["fit"] = h.get("fit")
    if not h.get("reachable"):
        o["at_saturation"] = h.get("at_saturation")
        return o
    o["rephrase"], o["test"], o["heldout_disjoint"] = h["rephrase"], h["test"], h["heldout_disjoint"]
    o["per_edit"] = {"edits": [int(x) for x in h["per_edit"]["edits"]], "tgen": flags(h["per_edit"]["tgen"]), "igen": flags(h["per_edit"]["igen"]),
                     "rel": flags(h["per_edit"]["rel"])}
    return o


def pct(x):
    x = np.asarray(x, dtype=bool)
    return 100.0 * float(x.mean()) if x.size else None


def cell(L, lab, d, C, readout, calibrated):
    sw = {"mask": "qdisjoint"} if (readout == "qd" and calibrated) else {}
    M = R1.e1_rowset_review1(L, ALPHA, d, labels=(lab,), skip_unavailable=False, **sw)[0]
    if not M.available:
        return {"not_defined": M.rec.get("not_defined") or M.rec.get("unavailable")}
    ev = fx.gate.evaluate(L, M, positions=("tf", "prompt"), parts=("probe", "edit", "hn"), top1=True)
    st = fx.metrics.gate_stats(L, ev, "tf")
    for p in ("probe", "edit", "hn"):
        assert list(ev[p]["tf"]["rows"]) == C["rows"][p], p
    c = {"label": M.label, "kind": M.kind, "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw, "digest": M.digest, "readout": readout,
         "n_capped": fx.methods.n_capped(M) if calibrated else None, "cal": slim_cal(M) if calibrated else None, "own_prompts": M.rec.get("own_prompts"), "st": st}
    ep, ee, eh = ev["probe"]["tf"], ev["edit"]["tf"], ev["hn"]["tf"]
    f = np.asarray(ep["fired"], dtype=bool)
    c["probe_fired"] = flags(f)                                      # valid probe rows, order C["rows"]["probe"]
    c["hd"] = {k: {"n": int(m.sum()), "fired": pct(f[m])} for k, m in C["pm"].items()}
    fp = np.asarray(ev["probe"]["prompt"]["fired"], dtype=bool)      # the decode's position (prompt-only key)
    c["hd_prompt"] = {k: {"n": int(m.sum()), "fired": pct(fp[m])} for k, m in C["pm"].items()}
    c["lineage_hd"] = {"n_firing": int(f[C["pm"]["hd"]].sum()), "widths": [int(x) for x in np.asarray(ep["n_cand"])[C["pm"]["hd"] & f]]}
    # edits: own edit released / another edit released, per kind in edit order
    kinds, own = C["ekind"], np.asarray(ee["own"])
    ot, fe = np.asarray(ee["own_top"], dtype=bool), np.asarray(ee["fired"], dtype=bool)
    otp = np.asarray(ev["edit"]["prompt"]["own_top"], dtype=bool)
    c["edit"] = {}
    for k in ("rel", "tgen", "igen"):
        m = kinds == k
        o = np.argsort(own[m], kind="stable")
        assert (own[m][o] == np.arange(L.T)).all()
        c["edit"][k] = {"own_top": flags(ot[m][o]), "other_top": flags((fe & ~ot)[m][o]), "own_top_prompt_pct": pct(otp[m])}
    fh, oh = np.asarray(eh["fired"], dtype=bool), np.asarray(eh["own_top"], dtype=bool)
    c["hn_fired"], c["hn_own_top"] = flags(fh), flags(oh)
    c["hn_prompt"] = {k: pct(np.asarray(ev["hn"]["prompt"]["fired"], dtype=bool)[C["hkind"] == k]) for k in HN if (C["hkind"] == k).any()}
    c["n_rows_differ_tf_vs_prompt"] = {p: int((np.asarray(ev[p]["tf"]["fired"]) != np.asarray(ev[p]["prompt"]["fired"])).sum()) for p in ev}
    # the literal conformal rule without the guard cap (R25, item 6)
    if calibrated:
        if c["n_capped"]:
            Ml = R1.e1_rowset_review1(L, ALPHA, d, labels=(lab,), skip_unavailable=False, allow_silence=True, **sw)[0]
            fl = np.asarray(fx.gate.decide(L, Ml, C["rows"]["probe"], "tf")["fired"], dtype=bool)
            de = fx.gate.decide(L, Ml, [L.edit_rows["rel"][i] for i in range(L.T)], "tf")
            c["literal"] = {"built": True, "hd": {k: {"n": int(m.sum()), "fired": pct(fl[m])} for k, m in C["pm"].items()},
                            "rel_own_top": pct(de["own_top"]), "probe_fired": flags(fl)}
        else:
            c["literal"] = {"built": False, "hd": c["hd"], "note": "no capped edit: the literal rule is the gate as run"}
    # far columns from fx.pools.firing (BEN-T: prompt-only keys; MME-T: teacher-forced keys); never from fx.gate.evaluate's benign part
    for pool in ("BEN-T", "MME-T"):
        try:
            kw = {"mask": fx.masks.question_disjoint(L)} if readout == "qd" else {}
            b = fx.pools.firing(L, M, pool, **kw)
            c[pool] = {k: b[k] for k in ("n", "n_fired", "fired", "which", "n_masked_out") if k in b}
            for k in ("far", "related", "image", "text"):
                if k in b:
                    c[pool][k] = {x: b[k][x] for x in ("n", "n_fired", "fired")}
        except (R1.NotDefined, NotImplementedError) as e:
            c[pool] = {"not_defined": str(e)[:300]}
    # H3 (R6; every calibrated gate; _x methods in both forms) -- registered read-out only
    if calibrated and readout == "reg" and C["has_hn"]:
        try:
            if lab.endswith("_x"):
                hb = R1.h3_both(L, M)
                c["h3"], c["h3_unseen"], c["h3_subset"] = slim_h3(hb["all"]), slim_h3(hb["unseen_foreign"]), hb["subset"]
                ns = R1.near_scope(L, M)
                c["near_scope"] = {k: ns[k] for k in ("all", "unseen_foreign", "subset")}
            else:
                c["h3"] = slim_h3(fx.sweeps.h3(L, M))
        except (R1.NotDefined, NotImplementedError, ValueError) as e:
            c["h3"] = {"not_defined": f"{type(e).__name__}: {str(e)[:300]}"}
    return c


def context(L):
    rows = {"probe": list(L.rows("probe", valid=True)), "edit": None, "hn": None}
    M0 = R1.e1_rowset_review1(L, ALPHA, 0, labels=("exact",), skip_unavailable=False)[0]
    ev = fx.gate.evaluate(L, M0, positions=("tf",), parts=("probe", "edit", "hn"))
    for p in ("probe", "edit", "hn"):
        rows[p] = list(ev[p]["tf"]["rows"])
    info = fx.gate.info(L, rows["probe"])
    qd = fx.masks.question_disjoint(L)
    dis = np.array([bool(r["disjoint"]) for r in info])
    near = np.array([r["near_edit"] is not None for r in info])
    qdm = np.array([bool(qd(L.index[j])) for j in rows["probe"]])
    img = np.array([r["ptype"] == "image" for r in info])
    pm = {"hd": dis, "hd_near": dis & near, "hd_far": dis & ~near, "hd_image": dis & img, "hd_text": dis & ~img, "hd_qd": dis & qdm}
    C = {"rows": rows, "pm": pm, "ekind": np.array([L.index[j]["kind"] for j in rows["edit"]]),
         "hkind": np.array([L.index[j]["hn_type"] for j in rows["hn"]]), "has_hn": len(rows["hn"]) > 0}
    sub = R1.hn_unseen_foreign(L) if C["has_hn"] else None
    meta = {"log": L.name, "T": L.T, "describe": L.describe(),
            "probe": {"sid": [L.sid(j) for j in rows["probe"]], "query_id": [L.index[j]["query_id"] for j in rows["probe"]],
                      "masks": {k: flags(m) for k, m in pm.items()}},
            "hn": {"sid": [L.sid(j) for j in rows["hn"]], "kind": [str(x) for x in C["hkind"]], "own": [int(L.own(j)) for j in rows["hn"]],
                   "unseen_foreign": None if sub is None else flags(np.isin(np.array(rows["hn"]), np.array(sub["rows"]))),
                   "unseen_foreign_n": None if sub is None else sub["n"]},
            "later_edits": [int(x) for x in fx.masks.later_edits(L)], "n_probe_rows_all": len(L.rows("probe")), "n_probe_rows_valid": len(rows["probe"]),
            "n_probe_invalid_excluded": len(L.rows("probe")) - len(rows["probe"]),
            "degenerate_y0": {k: int(sum(bool(L.index[j].get(k)) for j in rows["probe"])) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")},
            "hn_invalid_excluded": len(L.rows("hn")) - len(rows["hn"]), "lp_shift_rows": L.describe()["lp_shift_rows"]}
    return C, meta


def factor(L):
    """Factor level: rebase error against the cached clean-model targets, self-separation margins, non-oracle targets (as e5_rescope/code/e5a_rebase.py)."""
    import rescope
    from rescope import loglib as LL
    log = L.log
    out = {"log": L.name, "T": L.T}
    per = [fx.io.json.loads(x) for x in open(os.path.join(fx.logs.info(L.name)["path"], "per_edit_metrics.jsonl"))]
    no = [int(i) for i, r in enumerate(per) if not r.get("oracle", True)]
    out["non_oracle_edits"] = no
    out["non_oracle_matches_report"] = no == [e for e in NON_ORACLE if e < L.T]
    rescope.register_logs()
    hz = rescope.harness()
    Z = hz.TB.Z
    from vendor_v7.vlm import DEFAULT_DTYPE
    cfg = hz.TB.LOGS[L.name]
    zcfg = dataclasses.replace(Z.ZConfig.for_backbone(cfg["backbone"]), dtype=Z._dtype_tag(DEFAULT_DTYPE[cfg["backbone"]]))
    tg = [Z.load_cached(cfg["backbone"], cfg["layer"], u, zcfg, None) for u in log.meta["uids"]]
    out["n_targets_found"] = int(sum(t is not None for t in tg))
    V = LL.rebase(log)
    Vd, _ = fx.methods.released_values(L, True)
    out["driver_values_vs_library_rebase_max_abs"] = float((Vd - V).abs().max())
    if out["n_targets_found"] == L.T:
        ref = torch.stack([t.z for t in tg]).double() - torch.stack([t.h for t in tg]).double()
        e = LL.rebase_error(log, ref, V)
        out["rebase"] = {"rebased": {k: e["rebased"][k] for k in ("max", "median", "mean")}, "not_rebased": {k: e["not_rebased"][k] for k in ("max", "median", "mean")},
                         "rebased_per_edit": e["rebased_per_edit"], "not_rebased_per_edit": e["not_rebased_per_edit"]}
    else:
        out["rebase"] = None
    R = log.crosstalk()                                              # R[j, i] = r_i(k_j)
    off = R.clone()
    off.fill_diagonal_(-float("inf"))
    mg = (1 - off.max(0).values).numpy()                             # margin_i = 1 - max_{j != i} r_i(k_j)
    out["self_separation_margin"] = {"n_positive": int((mg > 0).sum()), "n_nonpositive": int((mg <= 0).sum()), "min": float(mg.min()), "median": float(np.median(mg)),
                                     "argmin": int(mg.argmin()), "per_edit": mg.tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--labels", default="")
    ap.add_argument("--readouts", default="reg,qd")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    a = ap.parse_args()
    assert a.log in LOGS
    L = fx.logs.open(a.log)
    lb = e3c.labels(L)
    out_dir = os.path.join(GATE, L.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    C, meta = context(L)
    meta.update(selection=lb["sel"], labels=lb, code=fx.code_fingerprint())
    jdump(meta, os.path.join(out_dir, "_meta.json"))
    if not os.path.exists(os.path.join(out_dir, "_factor.json")):
        jdump(factor(L), os.path.join(out_dir, "_factor.json"))
        print(f"[{time.time() - t0:7.0f}s] factor level", flush=True)
    todo = [x for x in a.labels.split(",") if x] or lb["cal"] + lb["cal_gl"] + lb["fixed_cpu"]
    for lab in todo:                                                 # gate-major (feature cache of fx.gate)
        fixed = lab in lb["fixed_cpu"]
        for ro in a.readouts.split(","):
            p = os.path.join(out_dir, f"{lab}__{ro}.json")
            if os.path.exists(p):
                continue
            cells = {}
            for d in ([0] if fixed else range(a.draws)):
                cells[str(d)] = cell(L, lab, d, C, ro, not fixed)
                if "not_defined" in cells[str(d)]:
                    break
            jdump({"log": L.name, "label": lab, "readout": ro, "alpha": None if fixed else ALPHA, "fixed": fixed, "n_draws": len(cells), "cells": cells,
                   "code": fx.code_fingerprint(), "seconds": time.time() - t0, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
            print(f"[{time.time() - t0:7.0f}s] {L.name} {lab} {ro} draws={len(cells)}", flush=True)
    print("DONE", L.name, flush=True)


if __name__ == "__main__":
    main()
