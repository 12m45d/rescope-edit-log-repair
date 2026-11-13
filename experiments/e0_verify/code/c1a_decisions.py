"""Check 1a: firing decisions of the library's calibration rules against my own naive implementation of PROTOCOL.md
section 6, on the cached keys of both development logs.  CPU.

  python c1a_decisions.py Q|L        -> out/c1a_<tag>.json

Library side (public API only, as docs/USAGE.md builds the default method from a KeyCache):
    KeyCache -> feature_gate -> add_guards(kc.own_keys) -> splits.sample(REPAIR valid probes, n, seed) -> calibrate_two_block /
    calibrate_shared(coords=kc.coords) (capped = default, uncapped = allow_silence) -> kc.decide(gate, which, rows, uids)
My side (vcommon.py; no library code): own read of index.jsonl / keys_*.pt / edit_factors.pt, own pools from the probe uids,
    own features in numpy fp64, the rule as the defining comparisons (no stored thresholds), guard cap, top-1 by margin.
Compared per sequence: fired, releasing edit (top-1), own edit candidate, own edit released; per calibration: the calibration
rows, eta, the rank, the capped edits, the guards silenced by the literal rule.
A differing decision is classified by the distance of the deciding quantity from its threshold in MY numbers: 'tie' if within
1e-9 (round-off between two fp64 evaluations of the same inner product), otherwise 'REAL'.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import vcommon as V

sys.path.insert(0, V.LIB)
import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import KeyCache, add_guards, calibrate_shared, calibrate_two_block, feature_gate, from_testbed, splits  # noqa: E402

ALPHAS = ("1/100", "1/20", "1/5")
SEEDS = (0, 1, 2, 3, 4)
NS = (200, 40)
TIE_BAND = 1e-9


def main(tag):
    t0 = time.time()
    d = V.DEV[tag]
    name, bb, fam = d["log"], d["backbone"], d["family"]
    # ------------------------------------------------------------------ my side: raw files
    idx = V.read_index(bb)
    Ktf, Kdec = V.read_keys(bb, "tf"), V.read_keys(bb, "decode")
    log = V.read_log(name)
    T, uids, W, K = log["T"], log["uids"], log["W"], log["K"]
    pos_of = {u: i for i, u in enumerate(uids)}
    res = {"log": name, "backbone": bb, "T": T, "structure": {}, "configs": [], "seconds": None}
    st = res["structure"]
    relrows = sorted([j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "rel" and e["edit"] < T], key=lambda j: idx[j]["edit"])
    st["family_edit_uids_equal_log_uids"] = [idx[j]["edit_uid"] for j in relrows] == uids
    in_log = lambda e: e["family"] == fam and (e.get("edit") is None or (e["edit"] < T and (e.get("edit_b") is None or e["edit_b"] < T)))  # noqa: E731
    rows_tf = [j for j, e in enumerate(idx) if in_log(e) and e.get("row_tf") is not None]
    rows_dec = [j for j, e in enumerate(idx) if in_log(e) and e.get("row_decode") is not None]
    # probe rows: the cache's rows against the development log's own probe records, and my own pools
    prow = [j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == "probe"]
    rec = V.jsonl(f"{V.TESTBED}/{name}/probe_records_v2.jsonl")
    st["n_probe_rows_cache"], st["n_probe_rows_log"] = len(prow), len(rec)
    same = [all(idx[j].get(k) == r.get(k) for k in ("probe_idx", "ptype", "uid", "prompt", "image", "y0_text", "y0_valid", "fmt")) for j, r in zip(prow, rec)]
    st["probe_rows_equal_log_records_in_order"] = len(prow) == len(rec) and all(same)
    st["n_probe_rows_differing_from_log_records"] = len(same) - sum(same)
    ann = V.annotate([idx[j] for j in prow])
    st["pool_flags_differing"] = {k: sum(a[k] != (idx[j].get(k) if k == "pool" else bool(idx[j].get(k))) for a, j in zip(ann, prow))
                                  for k in ("pool", "disjoint", "strict")}
    my_repair = [j for a, j in zip(ann, prow) if a["pool"] == "repair" and idx[j]["y0_valid"]]
    st["n_repair_valid"] = len(my_repair)
    st["n_heldout_disjoint_valid"] = sum(a["disjoint"] and bool(idx[j]["y0_valid"]) for a, j in zip(ann, prow))
    own = np.array([pos_of.get(idx[j].get("edit_uid"), -1) for j in rows_tf])
    own_d = np.array([pos_of.get(idx[j].get("edit_uid"), -1) for j in rows_dec])
    X_tf = Ktf[torch.tensor([idx[j]["row_tf"] for j in rows_tf])].double().numpy()
    X_dec = Kdec[torch.tensor([idx[j]["row_decode"] for j in rows_dec])].double().numpy()
    X_b1 = Ktf[torch.tensor([idx[j]["row_tf"] for j in relrows])].double().numpy()      # B = 1 keys of the own Rel prompts
    gedit = np.concatenate([np.arange(T), np.arange(T)])
    posr = {j: i for i, j in enumerate(rows_tf)}
    # ------------------------------------------------------------------ library side
    kc = KeyCache(bb, verify=True)
    L = from_testbed(name)
    st["lib_rows_tf_equal_mine"] = [j for j in kc.log_rows(L.meta["uids"]) if j in set(kc.having("tf"))] == rows_tf
    lib_repair = kc.log_rows(L.meta["uids"], kind="probe", pool="repair", valid=True)
    st["lib_repair_pool_equal_mine"] = lib_repair == my_repair
    V.say(f"{name}: T {T}, rows tf {len(rows_tf)}, decode {len(rows_dec)}, REPAIR valid {len(my_repair)}; structure {st}")
    tot = {"n_decisions": 0, "n_differing": 0, "n_tie": 0, "n_real": 0, "n_calibrations": 0, "n_cal_rows_differing": 0, "max_abs_eta_diff": 0.0,
           "n_rank_differing": 0, "n_capped_sets_differing": 0, "n_silenced_counts_differing": 0}
    for feature in ("ncoord", "cos"):
        F_tf, F_dec = V.FEATURES[feature](X_tf, W, K), V.FEATURES[feature](X_dec, W, K)
        G_my = np.concatenate([V.FEATURES[feature](K, W, K), V.FEATURES[feature](X_b1, W, K)])       # guards: stored keys, then B = 1 keys
        Gg, _ = add_guards(feature_gate(L, feature), *kc.own_keys(L.meta["uids"]))
        for rule in ("two_block", "shared"):
            for n in NS:
                for alpha in ALPHAS:
                    for seed in SEEDS if n == 200 else SEEDS[:2]:
                        cal = [my_repair[i] for i in V.draw(len(my_repair), n, seed)]
                        Q = F_tf[[posr[j] for j in cal]]
                        if rule == "two_block":
                            i1, i2 = V.halves(n, seed)
                            b, eta, k = V.naive_two_block(Q, i1, i2, alpha)
                        else:
                            b, eta, k = V.naive_shared(Q, alpha)
                        cap, sil = V.naive_caps(G_my, gedit, b, eta)
                        ben = splits.sample(lib_repair, n, seed)
                        coords = kc.coords(Gg, ben)
                        cfg = {"feature": feature, "rule": rule, "n": n, "alpha": alpha, "seed": seed, "eta_mine": eta, "rank_mine": k,
                               "cal_rows_equal": ben == cal, "variants": {}}
                        for capped in (True, False):
                            kw = dict(coords=coords, alpha=alpha, allow_silence=not capped)
                            S, r = calibrate_two_block(Gg, seed=seed, **kw) if rule == "two_block" else calibrate_shared(Gg, **kw)
                            which = ("tf", "decode") if (seed < 2 and n == 200) else ("tf",)
                            for wh in which:
                                F, rws, ow = (F_tf, rows_tf, own) if wh == "tf" else (F_dec, rows_dec, own_d)
                                fired, top, cand = V.naive_decide(F, b, eta, cap if capped else None)
                                has = ow >= 0
                                oc = np.zeros(len(ow), dtype=bool)
                                oc[has] = cand[has, ow[has]]
                                ot = fired & has & (top == ow)
                                dec = kc.decide(S, wh, rows=rws, uids=L.meta["uids"])
                                diff = {"fired": fired != dec["fired"], "top_edit": top != dec["top_edit"], "own_cand": oc != dec["own_cand"],
                                        "own_top": ot != dec["own_top"]}
                                anyd = np.zeros(len(rws), dtype=bool)
                                for v in diff.values():
                                    anyd |= v
                                # distance of the deciding quantities from their thresholds (my numbers)
                                M = F - b[None, :]
                                dist = np.abs(M - eta) if np.isfinite(eta) else np.full(M.shape, np.inf)
                                if capped:
                                    dist = np.minimum(dist, np.abs(F - cap[None, :]))
                                dmin = dist.min(1)
                                srt = np.sort(np.where(cand, M, -np.inf), axis=1)
                                gap = np.where(cand.sum(1) >= 2, srt[:, -1] - srt[:, -2], np.inf)      # top-1 tie between two candidates
                                tie = anyd & ((dmin <= TIE_BAND) | (gap <= TIE_BAND))
                                v = {"n_rows": len(rws), "n_differing_rows": int(anyd.sum()), "n_tie": int(tie.sum()), "n_real": int((anyd & ~tie).sum()),
                                     "by_flag": {a: int(x.sum()) for a, x in diff.items()}, "n_fired_mine": int(fired.sum()), "n_fired_lib": int(dec["fired"].sum()),
                                     "n_near_lib": int(dec["near"].sum()),
                                     "differing": [{"sid": idx[rws[i]]["sid"], "dist": float(dmin[i]), "gap": float(gap[i]), "mine": [bool(fired[i]), int(top[i])],
                                                    "lib": [bool(dec["fired"][i]), int(dec["top_edit"][i])], "lib_near": bool(dec["near"][i])}
                                                   for i in np.nonzero(anyd)[0][:8]]}
                                cfg["variants"][f"{'capped' if capped else 'uncapped'}/{wh}"] = v
                                tot["n_decisions"] += 4 * len(rws)
                                tot["n_differing"] += sum(v["by_flag"].values())
                                tot["n_tie"] += v["n_tie"]
                                tot["n_real"] += v["n_real"]
                            if capped:
                                mine_c = sorted(int(i) for i in np.nonzero(np.isfinite(cap))[0])
                                cfg.update(eta_lib=r["eta"], rank_lib=r["rank"], capped_mine=mine_c, capped_lib=r["capped_edits"],
                                           n_guards_silenced_literal_mine=int(sil.sum()), n_guards_silenced_literal_lib=r["n_guards_silenced_by_literal_rule"],
                                           bound_lib=r["bound_false_firing"], n1=r["n1"], n2=r["n2"])
                        tot["n_calibrations"] += 1
                        tot["n_cal_rows_differing"] += int(not cfg["cal_rows_equal"])
                        de = 0.0 if cfg["eta_lib"] == eta else abs(cfg["eta_lib"] - eta)
                        cfg["abs_eta_diff"] = de
                        tot["max_abs_eta_diff"] = max(tot["max_abs_eta_diff"], de)
                        tot["n_rank_differing"] += int(cfg["rank_lib"] != k)
                        tot["n_capped_sets_differing"] += int(cfg["capped_mine"] != cfg["capped_lib"])
                        tot["n_silenced_counts_differing"] += int(cfg["n_guards_silenced_literal_mine"] != cfg["n_guards_silenced_literal_lib"])
                        res["configs"].append(cfg)
            V.say(f"  {feature} {rule}: {tot} ({time.time() - t0:.0f}s)")
    res["totals"] = tot
    res["seconds"] = time.time() - t0
    res["rescope"] = rescope.__file__
    V.jdump(res, f"{V.OUT}/c1a_{tag}.json")
    V.say(f"wrote c1a_{tag}.json: {tot}")


if __name__ == "__main__":
    main(sys.argv[1])
