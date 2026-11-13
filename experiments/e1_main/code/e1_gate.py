"""E1, gate level (CPU, key caches): one log per process.

    python e1_gate.py <log> [--alphas 0.05,...] [--labels a,b,...] [--draws 20]

Per (label, alpha): out/gate/<log>/<label>__a<alpha>.json = {"cells": {draw: cell}} with ALL calibration draws (resumable: a
finished file is skipped).  A cell holds the numbers of task 2 (false firing as run and for the literal rule without the guard cap,
split by modality; coverage; own prompts; other-edit releases; near-scope negatives; lineage; capped edits), the far columns
BEN-T (all levels) and MME-T (alpha = 0.05) from fx.pools.firing (prompt-only keys for BEN-T), the per-edit coverage flags (H2,
coverage form) and at alpha = 0.05 the read-out H3 (fx.sweeps.h3; _x methods: both forms).
"""
import argparse
import os
import time

from e1c import *  # noqa: F401,F403
import e1c

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


def probe_split(L, d, info):
    """Firing (%) on heldout_disjoint / heldout_strict, all / image / text rows, from one decision on the valid probe rows."""
    f = np.asarray(d["fired"], dtype=bool)
    out = {}
    for pool in ("disjoint", "strict"):
        for mod in ("all", "image", "text"):
            m = np.array([bool(r[pool]) and (mod == "all" or r["ptype"] == mod) for r in info])
            out[f"{pool}.{mod}"] = {"n": int(m.sum()), "fired": 100.0 * float(f[m].mean()) if m.any() else None}
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


def cell(L, lab, alpha, d, info_probe, with_h3, with_mme):
    M = R1.e1_rowset_review1(L, alpha, d, labels=(lab,), skip_unavailable=False)[0]
    if not M.available:
        return {"not_defined": M.rec.get("not_defined") or M.rec.get("unavailable")}
    calibrated = M.alpha is not None
    ev = fx.gate.evaluate(L, M, positions=("tf", "prompt"), parts=("probe", "edit", "hn"), top1=True)
    st = fx.metrics.gate_stats(L, ev, "tf")
    c = {"label": M.label, "kind": M.kind, "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw, "digest": M.digest,
         "n_capped": fx.methods.n_capped(M), "cal": slim_cal(M) if calibrated else None, "own_prompts": M.rec.get("own_prompts"), "st": st,
         "split": probe_split(L, ev["probe"]["tf"], info_probe)}
    # the decode's position (teacher-forced sequence, the decode's token): only where a decision differs
    ndiff = {p: int((np.asarray(ev[p]["tf"]["fired"]) != np.asarray(ev[p]["prompt"]["fired"])).sum()
                    + ((np.asarray(ev[p]["tf"]["top_edit"]) != np.asarray(ev[p]["prompt"]["top_edit"])) & np.asarray(ev[p]["tf"]["fired"])
                       & np.asarray(ev[p]["prompt"]["fired"])).sum()) for p in ev}
    c["n_rows_differ_tf_vs_prompt"] = ndiff
    if any(ndiff.values()):
        sp = fx.metrics.gate_stats(L, ev, "prompt")
        c["st_prompt"] = {"probe": {k: sp["probe"][k] for k in ("heldout_disjoint", "heldout_strict")}, "edit": {k: {x: sp["edit"][k][x] for x in ("fired", "own_cand", "own_top", "other_top")} for k in ("rel", "tgen", "igen")},
                          "hn": {k: sp["hn"][k] for k in HN if k in sp["hn"]}}
    # per-edit coverage flags (H2, coverage form)
    rows = ev["edit"]["tf"]["rows"]
    kinds = np.array([L.index[j]["kind"] for j in rows])
    own = np.asarray(ev["edit"]["tf"]["own"])
    ot = np.asarray(ev["edit"]["tf"]["own_top"], dtype=bool)
    pe = {}
    for k in ("tgen", "igen"):
        m = kinds == k
        o = np.argsort(own[m], kind="stable")
        assert (own[m][o] == np.arange(L.T)).all()
        pe[k] = flags(ot[m][o])
    c["per_edit"] = pe
    c["h2_cov"] = 0.5 * (st["edit"]["tgen"]["own_top"] + st["edit"]["igen"]["own_top"])
    # the literal conformal rule without the guard cap
    if calibrated:
        if c["n_capped"]:
            Ml = R1.e1_rowset_review1(L, alpha, d, labels=(lab,), skip_unavailable=False, allow_silence=True)[0]
            dl = fx.gate.decide(L, Ml, ev["probe"]["tf"]["rows"], "tf")
            de = fx.gate.decide(L, Ml, [L.edit_rows["rel"][i] for i in range(L.T)], "tf")
            c["literal"] = dict(probe_split(L, dl, info_probe), built=True, rel_own_top=100.0 * float(np.asarray(de["own_top"]).mean()),
                                rel_own_cand=100.0 * float(np.asarray(de["own_cand"]).mean()))
        else:
            c["literal"] = dict(c["split"], built=False, note="no capped edit: the literal rule is the gate as run")
    # far columns from fx.pools.firing
    for pool in ("BEN-T",) + (("MME-T",) if with_mme else ()):
        try:
            b = fx.pools.firing(L, M, pool)
            c[pool] = {k: b[k] for k in ("n", "n_fired", "fired", "which") if k in b}
            for k in ("image", "text", "far", "related"):
                if k in b:
                    c[pool][k] = {x: b[k][x] for x in ("n", "fired")}
        except R1.NotDefined as e:
            c[pool] = {"not_defined": str(e)[:300]}
        except NotImplementedError as e:
            c[pool] = {"not_defined": str(e)[:300]}
    # H3 (alpha = 0.05; every calibrated gate)
    if calibrated and with_h3:
        if lab.endswith("_x"):
            hb = R1.h3_both(L, M)
            c["h3"] = slim_h3(hb["all"])
            c["h3_unseen"] = slim_h3(hb["unseen_foreign"])
            c["h3_subset"] = hb["subset"]
            ns = R1.near_scope(L, M)
            c["near_scope"] = {k: ns[k] for k in ("all", "unseen_foreign", "subset")}
        else:
            c["h3"] = slim_h3(fx.sweeps.h3(L, M))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    ap.add_argument("--labels", default="")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    a = ap.parse_args()
    L = fx.logs.open(a.log)
    assert L.name in DEV + WP, "E1 opens the development logs and their working points only"
    alphas = [float(x) for x in a.alphas.split(",")]
    labels = [x for x in a.labels.split(",") if x] or list(CAL_CPU) + list(FIXED_CPU)
    if not a.labels:
        s = e1c.selected_labels(L)
        labels += [s[k] for k in ("cos_H3_x", "cos_H2_x") if k in s and s[k] not in labels]
    labels = list(dict.fromkeys(labels))
    info_probe = fx.gate.info(L, L.rows("probe", valid=True))
    out_dir = os.path.join(GATE, L.name)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for lab in labels:                                                  # gate-major: all levels and draws of one feature, then the next
        fixed = lab in FIXED_CPU
        for alpha in ([None] if fixed else alphas):
            p = os.path.join(out_dir, f"{lab}__a{'none' if fixed else afmt(alpha)}.json")
            if os.path.exists(p):
                continue
            cells = {}
            for d in ([0] if fixed else range(a.draws)):
                cells[str(d)] = cell(L, lab, 0.05 if fixed else alpha, d, info_probe, with_h3=(not fixed and abs(alpha - 0.05) < 1e-12),
                                     with_mme=(fixed or abs(alpha - 0.05) < 1e-12))
                if "not_defined" in cells[str(d)]:
                    break
            jdump({"log": L.name, "label": lab, "alpha": None if fixed else alpha, "fixed": fixed, "n_draws": len(cells), "cells": cells,
                   "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
            print(f"[{time.time() - t0:7.0f}s] {L.name} {lab} alpha={alpha} draws={len(cells)}", flush=True)
    print("DONE", L.name, flush=True)


if __name__ == "__main__":
    main()
