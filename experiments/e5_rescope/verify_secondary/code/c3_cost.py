"""Check 3 (E8): storage of the scoped log by my own count (tensors of a gate built with the library); the re-timed latency cell (g_dev.py lat) against the experiment's cell."""
from vcommon import *
from rescope import from_testbed, feature_gate, add_guards, calibrate_two_block, splits, KeyCache
P = jload(f"{FINAL}/e8_cost/out/paper_numbers.json")
res = {"storage": {}, "latency": {}}
for name, bb in ((QDEV, "qwen25vl"), (LDEV, "llava")):
    kc = KeyCache(bb); log = from_testbed(name); uids = log.meta["uids"]
    S = feature_gate(log, "ncoord"); S, _ = add_guards(S, *kc.own_keys(uids))
    G, rec = calibrate_two_block(S, coords=kc.coords(S, splits.sample(kc.log_rows(uids, kind="probe", pool="repair", valid=True), 200, 0)), alpha=0.05, seed=0)
    parts = {"detectors": G.det_w.numel(), "values": G.values.numel(), "scales": G.det_s.numel(), "thresholds": G.det_tau.numel(), "ranking_baselines": 0 if G.det_b is None else G.det_b.numel()}
    n = sum(parts.values()); module = int(log.U.shape[1] * log.W.shape[1])
    mine = {"storage_numbers": n, "storage_mb": n * 8 / 2 ** 20, "storage_mb_module_dtype": n * 2 / 2 ** 20, "storage_share_of_module_numbers_pct": 100 * n / module, "T_at_module_size": module / (n / log.T),
            "guard_keys_not_counted": int(G.guard_key.numel())}
    st = {k: P[f"{name}.{k}"] for k in ("storage_numbers", "storage_mb", "storage_mb_module_dtype", "storage_share_of_module_numbers_pct")}
    res["storage"][name] = {"parts": parts, "mine": mine, "stored": st, "ok": bool(all(abs(mine[k] - st[k]) < 1e-9 for k in st))}
    print(name, res["storage"][name])
L = jload(f"{OUT}/g_lat_{QDEV}.json")
st = {k: P[f"{QDEV}.{k}"] for k in ("plain_forward_ms", "overhead_end_to_end_pct", "overhead_end_to_end_pct_p10", "overhead_end_to_end_pct_p90", "overhead_end_to_end_pct_min_based", "gate_ms_per_forward", "gate_ms_per_forward_min_based",
                                    "gate_pct_of_forward_min_based", "card_utilisation_pct_median")}
res["latency"] = {"cell": f"{QDEV}, last_fp64 (role last, top-1, REPL, fp64 gate); 30 sequences, 1 warm-up + 9 repetitions; gate alone 50 runs per sequence",
                  "mine": {"plain_forward_ms": L["plain_ms"]["median"], "gated_ms": L["gated_ms"]["median"], "overhead_end_to_end_pct": L["overhead_pct"]["median"], "overhead_p10": L["overhead_pct"]["p10"], "overhead_p90": L["overhead_pct"]["p90"],
                           "overhead_min_based_pct": L["overhead_min_based_pct"], "gate_ms_per_forward_median": L["gate_alone_ms"]["median"], "gate_ms_per_forward_min_based": L["gate_alone_ms_mean_of_min"],
                           "gate_pct_of_forward_min_based": L["gate_alone_pct_of_plain_forward"], "card_utilisation_pct_median": float(np.median([u for u in L["utilisation_pct"] if u is not None])),
                           "n_sequences_firing": L["n_sequences_firing"], "model_parameters": L["model_parameters_largest_module"]},
                  "stored": st, "stored_model_parameters": jload(f"{FINAL}/e8_cost/out/latency/qwen25vl__model.json")["model_parameters"]}
m = res["latency"]["mine"]
res["latency"]["consistent"] = bool(m["gate_ms_per_forward_min_based"] < 2 * st["gate_ms_per_forward_min_based"] and m["gate_ms_per_forward_min_based"] > 0.3 * st["gate_ms_per_forward_min_based"]
                                    and st["overhead_end_to_end_pct_p10"] <= m["overhead_end_to_end_pct"] <= st["overhead_end_to_end_pct_p90"] and m["model_parameters"] == res["latency"]["stored_model_parameters"])
print(res["latency"])
jdump(res, OUT + "/c3_cost.json")
