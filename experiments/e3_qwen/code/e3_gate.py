"""E3 (Qwen), gate level (CPU, key caches): one log per process.

    python e3_gate.py <log> [--labels a,b] [--draws 20] [--modes reg,qd]

Per label and mode: out/gate/<log>/<label>__<mode>.json = {"cells": {draw: cell}} with ALL 20 calibration draws (resumable: a finished
file is skipped).  Modes: reg = the registered read-out; qd = the second read-out of R8 (calibration draw from the question-disjoint
REPAIR rows, driver switch mask="qdisjoint").  A cell holds: firing on heldout_disjoint (all / image / text / near-scope / far; with
the question-disjoint filter), the literal rule without the guard cap, coverage, own prompts, releases by another edit, near-scope
negatives per kind, lineage widths, capped edits, the far columns BEN-T (prompt-only keys) and MME-T from fx.pools.firing, per-row
flags (for the crossed bootstrap) and, for the H3 rows, the read-out H3 (fx.sweeps.h3; _x methods: both forms + fx.review1.near_scope).
"""
import argparse
import os
import time

from e3c import *  # noqa: F401,F403
import e3c

R1 = fx.review1


def slim_cal(M):
    c = (M.rec.get("cal") or {})
    keep = ("op", "rule", "alpha", "n", "n1", "n2", "rank", "eta", "eta_infinite", "bound_false_firing", "n_edits_capped", "n_detectors_capped",
            "n_own_keys_silenced_by_literal_rule", "n_guards_silenced_by_literal_rule", "n_edits_silenced_by_literal_rule",
            "n_own_requests_silenced_by_literal_rule", "n_own_keys_silenced", "n_guards_silenced", "guarantee", "threshold", "pac")
    out = {k: c[k] for k in keep if k in c and not isinstance(c[k], (list,))}
    if isinstance(c.get("pac"), dict):
        out["pac"] = {k: v for k, v in c["pac"].items() if not isinstance(v, (list, dict))}
    return out


def slim_own(op):
    if not isinstance(op, dict):
        return op
    return {k: v for k, v in op.items() if not isinstance(v, (list, dict))}


def thr_summary(M):
    """Summary of the per-edit thresholds of a scoped gate (finite values), if the object carries them."""
    t = getattr(M.obj, "det_tau", None)
    if t is None:
        return None
    t = np.asarray(t.detach().cpu().double().numpy() if hasattr(t, "detach") else t, dtype=np.float64).ravel()
    fin = t[np.isfinite(t)]
    if fin.size == 0:
        return {"n": int(t.size), "n_finite": 0}
    return {"n": int(t.size), "n_finite": int(fin.size), "min": float(fin.min()), "median": float(np.median(fin)), "mean": float(fin.mean()), "max": float(fin.max())}


def probe_split(d, info, keep_qd):
    f = np.asarray(d["fired"], dtype=bool)
    out = {}
    dj = np.array([bool(r["disjoint"]) for r in info])
    for mod in ("all", "image", "text"):
        m = dj & np.array([(mod == "all" or r["ptype"] == mod) for r in info])
        out[mod] = {"n": int(m.sum()), "fired": 100.0 * float(f[m].mean()) if m.any() else None}
    near = np.array([r["near_edit"] is not None for r in info])
    for nm, m in (("near", dj & near), ("far", dj & ~near), ("qd", dj & keep_qd), ("qd_far", dj & keep_qd & ~near)):
        out[nm] = {"n": int(m.sum()), "n_fired": int(f[m].sum()), "fired": 100.0 * float(f[m].mean()) if m.any() else None}
    return out


def slim_h3(h):
    if h is None:
        return None
    o = {k: h.get(k) for k in ("reachable", "delta", "n_edits_at_cap", "n_edits_at_cap_at_0", "limit", "guards", "n_fit", "n_test")}
    o["fit"] = h.get("fit")
    if not h.get("reachable"):
        o["at_saturation"] = h.get("at_saturation")
        return o
    o["rephrase"] = h["rephrase"]
    o["test"] = h["test"]
    o["heldout_disjoint"] = h["heldout_disjoint"]
    o["per_edit"] = {"edits": [int(x) for x in h["per_edit"]["edits"]], "tgen": flags(h["per_edit"]["tgen"]), "igen": flags(h["per_edit"]["igen"]),
                     "rel": flags(h["per_edit"]["rel"])}
    return o


def build(L, lab, d, **sw):
    return R1.e1_rowset_review1(L, ALPHA, d, labels=(lab,), skip_unavailable=False, **sw)[0]


def cell(L, lab, d, ctx, mode, with_h3):
    sw = {"mask": "qdisjoint"} if mode == "qd" else {}
    M = build(L, lab, d, **sw)
    if not M.available:
        return {"not_defined": M.rec.get("not_defined") or M.rec.get("unavailable")}
    calibrated = M.alpha is not None
    ev = fx.gate.evaluate(L, M, positions=("tf", "prompt"), parts=("probe", "edit", "hn"), top1=True)
    st = fx.metrics.gate_stats(L, ev, "tf")
    assert list(ev["probe"]["tf"]["rows"]) == ctx["probe_rows"] and list(ev["hn"]["tf"]["rows"]) == ctx["hn_rows"]
    c = {"label": M.label, "kind": M.kind, "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw, "digest": M.digest, "mode": mode,
         "n_capped": fx.methods.n_capped(M) if calibrated else None, "cal": slim_cal(M) if calibrated else None, "own_prompts": slim_own(M.rec.get("own_prompts")),
         "thresholds": thr_summary(M), "st": st, "split": probe_split(ev["probe"]["tf"], ctx["info_probe"], ctx["keep_qd"])}
    c["n_rows_differ_tf_vs_prompt"] = {p: int((np.asarray(ev[p]["tf"]["fired"]) != np.asarray(ev[p]["prompt"]["fired"])).sum()) for p in ev}
    # per-row flags: heldout_disjoint probes, hard negatives, per-edit coverage (for the crossed bootstraps)
    fp = np.asarray(ev["probe"]["tf"]["fired"], dtype=bool)
    c["hd_flags"] = flags(fp[ctx["hd_idx"]])
    c["hn_flags"] = flags(np.asarray(ev["hn"]["tf"]["fired"], dtype=bool))
    c["hn_own_flags"] = flags(np.asarray(ev["hn"]["tf"]["own_top"], dtype=bool))
    nc = np.asarray(ev["probe"]["tf"]["n_cand"])[ctx["hd_idx"]]
    w = nc[fp[ctx["hd_idx"]]]
    c["lineage_hd"] = {"n_firing": int(w.size), "hist": {str(int(k)): int((w == k).sum()) for k in np.unique(w)}}
    rows = ev["edit"]["tf"]["rows"]
    kinds = np.array([L.index[j]["kind"] for j in rows])
    own = np.asarray(ev["edit"]["tf"]["own"])
    ot = np.asarray(ev["edit"]["tf"]["own_top"], dtype=bool)
    fe = np.asarray(ev["edit"]["tf"]["fired"], dtype=bool)
    pe, po = {}, {}
    for k in ("rel", "tgen", "igen"):
        m = kinds == k
        o = np.argsort(own[m], kind="stable")
        assert (own[m][o] == np.arange(L.T)).all()
        pe[k] = flags(ot[m][o])
        po[k] = flags((fe[m] & ~ot[m])[o])
    c["per_edit"] = pe
    c["per_edit_other"] = po                                            # the prompt is released by ANOTHER edit (top-1 confusion)
    c["h2_cov"] = 0.5 * (st["edit"]["tgen"]["own_top"] + st["edit"]["igen"]["own_top"])
    # the literal conformal rule without the guard cap (R25 (1))
    if calibrated:
        if c["n_capped"]:
            try:
                Ml = build(L, lab, d, allow_silence=True, **sw)
                dl = fx.gate.decide(L, Ml, ev["probe"]["tf"]["rows"], "tf")
                de = fx.gate.decide(L, Ml, [L.edit_rows["rel"][i] for i in range(L.T)], "tf")
                c["literal"] = dict(probe_split(dl, ctx["info_probe"], ctx["keep_qd"]), built=True, rel_own_top=100.0 * float(np.asarray(de["own_top"]).mean()),
                                    hd_flags=flags(np.asarray(dl["fired"], dtype=bool)[ctx["hd_idx"]]))
            except Exception as e:                                     # noqa: BLE001
                c["literal"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        else:
            c["literal"] = dict(c["split"], built=False, note="no capped edit: the literal rule is the gate as run")
    # far columns from fx.pools.firing (BEN-T: prompt-only keys; MME-T: rows with a valid pre-edit answer)
    for pool in ("BEN-T", "MME-T"):
        try:
            b = fx.pools.firing(L, M, pool, mask=ctx["qd_pred"] if (mode == "qd" and pool == "BEN-T") else None)
            c[pool] = {k: b[k] for k in ("n", "n_fired", "fired", "which", "n_masked_out") if k in b}
            for k in ("image", "text", "far", "related"):
                if k in b:
                    c[pool][k] = {x: b[k][x] for x in ("n", "n_fired", "fired")}
            if mode == "reg" and pool == "BEN-T":
                bq = fx.pools.firing(L, M, pool, mask=ctx["qd_pred"])
                c["BEN-T.qd"] = {"n": bq["n"], "n_fired": bq["n_fired"], "fired": bq["fired"], "n_masked_out": bq["n_masked_out"], "far": {x: bq["far"][x] for x in ("n", "n_fired", "fired")}}
        except (R1.NotDefined, NotImplementedError) as e:
            c[pool] = {"not_defined": str(e)[:300]}
    # H3 (R6, R25 (4)); _x methods: both forms of the near-scope read-outs
    if calibrated and lab.endswith("_x") and mode == "reg":
        ns = R1.near_scope(L, M)
        c["near_scope"] = {k: ns[k] for k in ("all", "unseen_foreign", "subset")}
    if calibrated and with_h3:
        if lab.endswith("_x"):
            hb = R1.h3_both(L, M)
            c["h3"] = slim_h3(hb["all"])
            c["h3_unseen"] = slim_h3(hb["unseen_foreign"])
            c["h3_subset"] = hb["subset"]
        else:
            c["h3"] = slim_h3(fx.sweeps.h3(L, M))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--labels", default="")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--modes", default="reg,qd")
    a = ap.parse_args()
    assert a.log in LOGS, "this task opens its seven logs only"
    L = fx.logs.open(a.log)
    lb = e3c.labels(L)
    labs = [x for x in a.labels.split(",") if x] or list(lb["cal"]) + list(lb["gl"]) + list(lb["fixed"])
    probe_rows = list(L.rows("probe", valid=True))
    info_probe = fx.gate.info(L, probe_rows)
    hn_rows = list(L.rows("hn", valid=True))
    info_hn = fx.gate.info(L, hn_rows)
    hd_idx = np.array([i for i, r in enumerate(info_probe) if r["disjoint"]], dtype=np.int64)
    assert [probe_rows[i] for i in hd_idx] == list(L.pool("heldout_disjoint"))
    ctx = {"probe_rows": probe_rows, "hn_rows": hn_rows, "info_probe": info_probe, "keep_qd": fx.masks.keep(L, probe_rows), "hd_idx": hd_idx,
           "qd_pred": fx.masks.question_disjoint(L)}
    out_dir = os.path.join(GATE, L.name)
    os.makedirs(out_dir, exist_ok=True)
    jdump({"log": L.name, "T": L.T, "describe": L.describe(), "selection": lb["sel"],
           "hd": {"csid": [L.sid(probe_rows[i]) for i in hd_idx], "query_id": [L.index[probe_rows[i]]["query_id"] for i in hd_idx],
                  "near": flags(info_probe[i]["near_edit"] is not None for i in hd_idx), "qd": flags(ctx["keep_qd"][hd_idx]),
                  "ptype": [info_probe[i]["ptype"] for i in hd_idx]},
           "hn": {"csid": [L.sid(j) for j in hn_rows], "edit": [int(L.own(j)) for j in hn_rows], "hn_type": [r["hn_type"] for r in info_hn],
                  "qd": flags(fx.masks.keep(L, hn_rows))},
           "later_edits": [int(x) for x in fx.masks.later_edits(L)], "code": fx.code_fingerprint()}, os.path.join(out_dir, "_context.json"))
    t0 = time.time()
    for mode in a.modes.split(","):
        for lab in labs:                                                # gate-major: all draws of one feature, then the next
            fixed = lab in lb["fixed"]
            if mode == "qd" and fixed:
                continue                                                # no calibration: the second read-out is a filter of the same decisions (cell field split.qd)
            p = os.path.join(out_dir, f"{lab}__{mode}.json")
            if os.path.exists(p):
                continue
            cells, err = {}, None
            for d in ([0] if fixed else range(a.draws)):
                try:
                    cells[str(d)] = cell(L, lab, d, ctx, mode, with_h3=(mode == "reg" and lab in lb["h3"]))
                except (R1.NotDefined, TypeError, ValueError, NotImplementedError) as e:
                    if mode == "reg":
                        raise
                    err = f"{type(e).__name__}: {str(e)[:400]}"         # a switch the gate does not take: recorded, not replaced
                    cells = {"0": {"not_defined": err}}
                    break
                if "not_defined" in cells[str(d)]:
                    break
            jdump({"log": L.name, "label": lab, "mode": mode, "alpha": None if fixed else ALPHA, "fixed": fixed, "n_draws": len(cells), "cells": cells,
                   "code": fx.code_fingerprint(), "seconds": time.time() - t0, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
            print(f"[{time.time() - t0:7.0f}s] {L.name} {mode} {lab} draws={len(cells)} {err or ''}", flush=True)
    print("DONE", L.name, flush=True)


if __name__ == "__main__":
    main()
