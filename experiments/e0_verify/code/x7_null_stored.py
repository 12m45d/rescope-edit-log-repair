"""Extra check X7 (CPU): two library pieces that the registered experiments use and checks 1 - 6 do not reach.

  python x7_null_stored.py        -> out/x7_null_stored.json

  A  rescope.null_repair(S, keys, ridge=0) (the detector step of the registered variant ReScope-N, PROTOCOL.md amendment of
     2026-11-11 01:35): against my own closed form of the definition (the vector closest to w_j with zero coordinate on the
     benign keys and the own coordinate kept: w' = w - C^T (C C^T)^-1 (C w - d), C = [benign keys; k_j], d = [0; <w_j, k_j>]);
     then the default two-block calibration + KeyCache.decide on the repaired gate against my naive rule on MY repaired detectors.
     n_A = 100 seeded REPAIR rows disjoint from the 200 calibration rows (draw 0).
  B  rescope.stored_metrics(name) of the two development logs against my own reading of the testbed files and my own pools.
"""
import os
import sys

import numpy as np

import vcommon as V

os.environ.setdefault("RESCOPE_CACHE", V.VER + "/work/hcache")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, V.LIB)
import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import KeyCache, add_guards, calibrate_two_block, feature_gate, from_testbed, null_repair, splits  # noqa: E402

res = {}
for tag, d in V.DEV.items():
    name, bb, fam = d["log"], d["backbone"], d["family"]
    idx = V.read_index(bb)
    Ktf = V.read_keys(bb, "tf")
    log = V.read_log(name)
    T, uids, W, K = log["T"], log["uids"], log["W"], log["K"]
    pos = {u: i for i, u in enumerate(uids)}
    in_log = lambda e: e["family"] == fam and (e.get("edit") is None or (e["edit"] < T and (e.get("edit_b") is None or e["edit_b"] < T)))  # noqa: E731
    rows = [j for j, e in enumerate(idx) if in_log(e) and e.get("row_tf") is not None]
    prow = [j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "probe"]
    ann = V.annotate([idx[j] for j in prow])
    repair = [j for j, a in zip(prow, ann) if a["pool"] == "repair" and idx[j]["y0_valid"]]
    relrows = sorted([j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "rel" and e["edit"] < T], key=lambda j: idx[j]["edit"])
    key = lambda js: Ktf[torch.tensor([idx[j]["row_tf"] for j in js])].double().numpy()      # noqa: E731
    cal_i = V.draw(len(repair), 200, 0)
    rest = np.setdiff1d(np.arange(len(repair)), cal_i)
    nA_i = rest[np.sort(np.random.default_rng(1000).choice(len(rest), 100, replace=False))]
    cal, nA = [repair[i] for i in cal_i], [repair[i] for i in nA_i]
    A = key(nA)
    # ---- mine: closed form of the definition, detector by detector
    s = np.einsum("id,id->i", W, K)
    Wn = np.empty_like(W)
    G = A @ A.T
    for j in range(T):
        C = np.vstack([A, K[j][None]])
        dvec = np.concatenate([np.zeros(A.shape[0]), [s[j]]])
        CC = np.block([[G, (A @ K[j])[:, None]], [(A @ K[j])[None, :], np.array([[K[j] @ K[j]]])]])
        lam = np.linalg.solve(CC, C @ W[j] - dvec)
        Wn[j] = W[j] - C.T @ lam
    # ---- library
    kc = KeyCache(bb)
    L = from_testbed(name)
    S0 = feature_gate(L, "ncoord")
    S1, rec = null_repair(S0, kc.keys("tf", nA), ridge=0.0)
    Wl = S1.det_w.numpy()
    rel = np.linalg.norm(Wl - Wn, axis=1) / np.linalg.norm(Wn, axis=1)
    S1g, _ = add_guards(S1, *kc.own_keys(uids))
    S2, rec2 = calibrate_two_block(S1g, coords=kc.coords(S1g, cal), alpha=0.05, seed=0)
    dec = kc.decide(S2, "tf", rows=rows, uids=uids)
    # ---- naive rule on MY repaired detectors (the scale stays s_j = <W_j, k_j> of the log: the own coordinate is kept)
    feat = lambda X: (X @ Wn.T) / s[None, :]                                              # noqa: E731
    X, Xb1 = key(rows), key(relrows)
    F = feat(X)
    i1, i2 = V.halves(200, 0)
    b, eta, k = V.naive_two_block(feat(key(cal)), i1, i2, "1/20")
    cap, _ = V.naive_caps(np.concatenate([feat(K), feat(Xb1)]), np.concatenate([np.arange(T), np.arange(T)]), b, eta)
    fired, top, cand = V.naive_decide(F, b, eta, cap)
    M = F - b[None, :]
    dist = np.minimum(np.abs(M - eta), np.abs(F - cap[None, :])).min(1)
    diff = (fired != dec["fired"]) | (top != dec["top_edit"])
    own = np.array([pos.get(idx[j].get("edit_uid"), -1) for j in rows])
    kind = np.array([idx[j]["kind"] for j in rows])
    res.setdefault("A_null_repair", {})[tag] = {
        "n_benign_keys": len(nA), "disjoint_from_calibration": len(set(nA) & set(cal)) == 0, "detectors_max_rel_diff_lib_vs_closed_form": float(rel.max()),
        "max_abs_coordinate_on_benign_keys_mine": float(np.abs(feat(A)).max()), "max_abs_coordinate_on_benign_keys_lib": float(np.abs((A @ Wl.T) / s[None, :]).max()),
        "max_own_coordinate_error_lib": float(np.abs(np.einsum("id,id->i", Wl, K) / s - 1).max()), "lib_record": {k: rec[k] for k in ("rank_benign", "infeasible", "rel_change_median", "rel_change_max", "n_guards_lost")},
        "eta_abs_diff": abs(eta - rec2["eta"]), "capped_equal": sorted(int(i) for i in np.nonzero(np.isfinite(cap))[0]) == rec2["capped_edits"], "n_capped": int(np.isfinite(cap).sum()),
        "n_rows": len(rows), "decisions_differing": int(diff.sum()), "of_them_further_than_1e-7_from_a_threshold": int((diff & (dist > 1e-7)).sum()),
        "mechanics_only_note": "development log; ReScope-N numbers are not reported here"}
    print(tag, res["A_null_repair"][tag], flush=True)
    # ---- B: stored_metrics
    sm = rescope.stored_metrics(name)
    recs = V.jsonl(f"{V.TESTBED}/{name}/probe_records_v2.jsonl")
    an = V.annotate(recs)
    met = V.jload(f"{V.TESTBED}/{name}/metrics.json")
    mine = {}
    for pl, pred in (("repair", lambda a: a["pool"] == "repair"), ("heldout", lambda a: a["pool"] == "heldout"), ("heldout_disjoint", lambda a: a["disjoint"])):
        sub = [r for r, a in zip(recs, an) if pred(a)]
        v = [r for r in sub if r["y0_valid"]]
        mine[pl] = {"n_rows": len(sub), "n_valid": len(v), "n_D1": sum(bool(r["is_complaint_D1"]) for r in v), "n_D2": sum(bool(r["is_complaint_D2"]) for r in sub)}
    eq = all(sm["probes"][pl][k] == mine[pl][k] for pl in mine for k in mine[pl])
    per = V.jsonl(f"{V.TESTBED}/{name}/per_edit_metrics.jsonl")
    ex = {m: 100.0 * np.mean([bool(r[f"post_{m}_exact"]) for r in per]) for m in ("rel", "tgen", "igen")}
    res.setdefault("B_stored_metrics", {})[tag] = {"probe_counts_equal_mine": bool(eq), "mine": mine,
                                                   "edits_exact_lib": {m: sm["edits"][m]["exact"] for m in ex}, "edits_exact_from_per_edit_file": ex,
                                                   "edits_equal": all(abs(ex[m] - sm["edits"][m]["exact"]) < 1e-9 for m in ex),
                                                   "locality_equal_metrics_json": sm["locality"]["T-Loc"] == met["TLoc_probes"] and sm["locality"]["I-Loc"] == met["ILoc_probes"],
                                                   "D1_rate_heldout_disjoint": 100.0 * mine["heldout_disjoint"]["n_D1"] / mine["heldout_disjoint"]["n_valid"]}
    print(tag, res["B_stored_metrics"][tag], flush=True)
res["pass"] = all(v["detectors_max_rel_diff_lib_vs_closed_form"] < 1e-8 and v["of_them_further_than_1e-7_from_a_threshold"] == 0 and v["capped_equal"] for v in res["A_null_repair"].values()) and \
    all(v["probe_counts_equal_mine"] and v["edits_equal"] and v["locality_equal_metrics_json"] for v in res["B_stored_metrics"].values())
V.jdump(res, f"{V.OUT}/x7_null_stored.json")
print("PASS", res["pass"])
