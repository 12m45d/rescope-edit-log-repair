"""CPU tests of e10t (development log only; no model).  python tests/t_cpu.py [names]  -> tests/out/cpu_<name>.json

    log      conversion and orientation: own coordinate, delta() against (W_E - W_0)^T, bitwise replay of the deployed weight
    rebase   identity (c), algebraic: W_0 k_i + b + V_i against the logged module-output target; update-by-update values against
             rescope.rebase; prefix logs share the bits of their common values
    draws    calibration draw = rescope.splits.sample; null sample disjoint by text; 20 draws
    calib    thresholds from the cached per-query maxima against the library's calibration on the per-query key tensors
    null0    identity (d): the null-repair variant with n_a = 0 equals the default method bit for bit (gate tensors, decisions)
    gate     e10t.gate.decide against the library's ScopedLog.firing on every prompt of the development log and 400 pool prompts
    cache    key cache: files, verification record, float16 rounding against the decision band
    rows     the row set builds; undefined rows carry a reason; ReScope-W constructors; a prefix log
    pack     the row hook on a synthetic module: PAD masking, key-value-cache continuation, lossless edit sets of any size
    guard    guards as served (guard_margin) over 20 draws
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import rescope  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
INF = float("inf")
L = e10t.open_log(e10t.DEV_LOG)
assert L.development


def t_log():
    TX = e10t.data._tx()
    log = L.log
    W0, WE = L.weights()
    s = log.own()
    D = log.delta()
    dW = (WE.double() - W0.double()).T
    Wr = e10t.data.replay_weight(e10t.DEV_LOG)
    W50 = e10t.data.replay_weight(e10t.DEV_LOG, 50)
    f = TX.factors(e10t.DEV_LOG)
    Wc = W50.clone()
    for i in range(50, 100):
        Wc += f["left"][i].unsqueeze(1) @ f["right"][i].unsqueeze(0)
    L50 = e10t.open_log(e10t.DEV_LOG, T=50)
    d = {"own_coordinate_max_abs_dev_from_1": float((s - 1).abs().max()), "delta_shape": list(D.shape), "module_weight_shape": list(W0.shape),
         "delta_vs_WE_minus_W0_transposed_rel_fro": float((D - dW).norm() / dW.norm()), "replay_bitwise_equals_deployed": bool(torch.equal(Wr, WE)),
         "prefix_replay_continued_bitwise_equals_deployed": bool(torch.equal(Wc, WE)), "prefix_log": L50.describe(),
         "prefix_weight_equals_replay_50": bool(torch.equal(L50.weights()[1], W50)), "batch_is_arange": bool(torch.equal(log.batch, torch.arange(L.T)))}
    ok = (d["own_coordinate_max_abs_dev_from_1"] < 1e-12 and d["delta_vs_WE_minus_W0_transposed_rel_fro"] < 1e-5 and d["replay_bitwise_equals_deployed"]
          and d["prefix_replay_continued_bitwise_equals_deployed"] and d["prefix_weight_equals_replay_50"] and d["batch_is_arange"] and L50.T == 50)
    return ok, d, "own coordinate 1; delta() = (W_E - W_0)^T up to the write round-off; float32 replay = deployed weight bit for bit"


def _bias():
    from safetensors import safe_open
    import tp_common as C
    with safe_open(os.path.join(C.MODEL_DIR, "model.safetensors"), "pt") as f:
        ks = [k for k in f.keys() if k.endswith("h.17.mlp.c_proj.bias")]
        return f.get_tensor(ks[0]).double()


def t_rebase():
    log = L.log
    W0, _ = L.weights()
    b = _bias()
    V, vrec = e10t.methods.released_values(L)
    ref = rescope.rebase(log)
    base = log.meta["K_request"] @ W0.double() + b                       # module output of the pre-edit model at the solver key
    tgt = log.meta["target"]
    err_v = ((base + V - tgt).norm(dim=1) / tgt.norm(dim=1))
    err_u = ((base + log.U - tgt).norm(dim=1) / tgt.norm(dim=1))
    L50 = e10t.open_log(e10t.DEV_LOG, T=50)
    V50, _ = e10t.methods.released_values(L50)
    import types
    import fx
    Vfx, _ = fx.methods.released_values(types.SimpleNamespace(log=log, name="c_rebase:" + L.name), True)     # the driver's own function
    d = {"identity": "module output of the pre-edit model at the solver key + rebased value = logged target (k_i W_0 + b + V_i = target_i)",
         "max_rel_err_rebased": float(err_v.max()), "median_rel_err_rebased": float(err_v.median()),
         "median_rel_err_not_rebased": float(err_u[1:].median()), "max_rel_err_not_rebased": float(err_u.max()),
         "values_vs_rescope_rebase_max_rel": float(((V - ref).norm(dim=1) / ref.norm(dim=1)).max()),
         "prefix_values_bitwise_equal": bool(torch.equal(V50, V[:50])), "values_bitwise_equal_to_the_drivers_function": bool(torch.equal(V, Vfx)),
         "values_record": vrec}
    ok = (d["max_rel_err_rebased"] < 1e-5 and d["values_vs_rescope_rebase_max_rel"] < 1e-12 and d["prefix_values_bitwise_equal"] and d["median_rel_err_not_rebased"] > 1e-3
          and d["values_bitwise_equal_to_the_drivers_function"])
    return ok, d, "identity (c), algebraic: every edit attains its module-output target at its solver key with rebased values"


def t_draws():
    pool = L.benign("repair")
    same, disj, sizes, nested = 0, 0, [], 0
    for dr in range(e10t.N_DRAWS):
        c = e10t.data.calibration(L, 200, dr)
        ref = rescope.splits.sample(pool, 200, dr)
        same += int([r["prompt_id"] for r in c.rows] == [r["prompt_id"] for r in ref])
        a = e10t.data.null_sample(L, 100, dr)
        disj += int(not ({r["text_id"] for r in a} & set(c.text_ids)) and len({r["text_id"] for r in a}) == 100)
        sizes.append(len(a))
        c100 = e10t.data.calibration(L, 100, dr)
        nested += int(set(c100.text_ids) <= set(c.text_ids))
    d = {"n_draws": e10t.N_DRAWS, "n_pool": len(pool), "draw_equals_rescope_splits_sample": same, "null_sample_disjoint_by_text": disj,
         "null_sample_sizes": sorted(set(sizes)), "nested_in_n": nested, "null_sample_n0": len(e10t.data.null_sample(L, 0, 0)),
         "n_distinct_pool_texts": len({r["text_id"] for r in pool})}
    ok = same == disj == nested == e10t.N_DRAWS and sizes == [100] * e10t.N_DRAWS and d["null_sample_n0"] == 0
    return ok, d, "calibration draws are the library's seeded samples of the REPAIR pool; the null sample shares no prompt text with them"


def t_calib():
    out = {}
    ok = True
    for feat in ("ncoord", "cos"):
        S, _ = e10t.methods.base_gate(L, feat)
        dr = e10t.data.calibration(L, 200, 0)
        Q = e10t.gate.features(L, S, dr.text_ids)["Q"]
        qs = [torch.from_numpy(np.array(L.kc.keys(t))) for t in dr.text_ids]
        Qlib = S.query_max(qs)
        A, ra = rescope.calibrate_two_block(S, coords=Q, alpha=0.05, seed=0)
        Bq, rb = rescope.calibrate_two_block(S, qs, 0.05, seed=0)
        C2, rc = rescope.calibrate_shared(S, coords=Q, alpha=0.05)
        D2, rd = rescope.calibrate_shared(S, qs, 0.05)
        o = {"max_abs_diff_query_maxima": float((Q - Qlib).abs().max()), "n_query_maxima_bitwise_equal": int((Q == Qlib).sum()), "n_query_maxima": int(Q.numel()),
             "two_block": {"eta_cache": ra["eta"], "eta_library": rb["eta"], "max_abs_threshold_diff": float((A.det_tau - Bq.det_tau).abs().max()),
                           "n_thresholds_bitwise_equal": int((A.det_tau == Bq.det_tau).sum()), "n_edits_capped": ra["n_edits_capped"],
                           "bound": ra["bound_false_firing"], "all_guards_fire": bool(A.guard_ok().all()), "n_guards": int(A.guard_key.shape[0])},
             "shared": {"eta_cache": rc["eta"], "eta_library": rd["eta"], "max_abs_threshold_diff": float((C2.det_tau - D2.det_tau).abs().max()),
                        "n_edits_capped": rc["n_edits_capped"], "bound": rc["bound_false_firing"]}}
        # the calibrated event on the calibration queries themselves: at most floor(alpha (n2 + 1)) block-2 queries fire (uncapped rule)
        cal_fire = (Q >= A.det_tau[None, :]).any(1).numpy()
        b2 = np.array(ra["block2"])
        o["two_block"]["n_block2_queries_firing"] = int(cal_fire[b2].sum())
        o["two_block"]["n_allowed"] = ra["n_allowed_above"]
        o["two_block"]["n_block1_queries_firing"] = int(cal_fire[np.array(ra["block1"])].sum())
        ok &= (o["max_abs_diff_query_maxima"] < 1e-9 and o["two_block"]["max_abs_threshold_diff"] < 1e-9 and o["shared"]["max_abs_threshold_diff"] < 1e-9
               and o["two_block"]["all_guards_fire"] and o["two_block"]["n_guards"] == 2 * L.T
               and (o["two_block"]["n_edits_capped"] > 0 or o["two_block"]["n_block2_queries_firing"] <= o["two_block"]["n_allowed"])
               and o["two_block"]["n_block1_queries_firing"] == 0)
        out[feat] = o
    return ok, out, "calibration from the cached per-query maxima = the library's calibration on the per-query key tensors; guards fire; block-2 firing within the allowed count"


def _same(a, b):
    return (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))


def t_null0():
    A = e10t.methods.rescope(L, draw=0)
    Bm = e10t.methods.rescope_null(L, n_a=0, draw=0)
    C100 = e10t.methods.rescope_null(L, n_a=100, draw=0)
    fields = ("det_w", "det_s", "det_tau", "det_edit", "det_kind", "det_key", "values", "lin_w", "lin_u", "guard_key", "guard_edit", "det_b", "det_u")
    eq = {f: bool(_same(getattr(A.obj, f), getattr(Bm.obj, f))) for f in fields}
    ev_a, ev_b = e10t.gate.evaluate(L, A), e10t.gate.evaluate(L, Bm)
    dec_same = all(bool(np.array_equal(ev_a[p][k], ev_b[p][k])) for p in ev_a for k in ("fired", "near", "n_cand_edits", "own_released", "n_fire_pos")) \
        and all(ev_a[p]["rel"] == ev_b[p]["rel"] for p in ev_a)
    d = {"fields_bitwise_equal": eq, "eta_equal": A.obj.eta == Bm.obj.eta, "gate_sig_equal": A.rec["gate_sig"] == Bm.rec["gate_sig"],
         "decisions_equal_on_all_row_sets": dec_same, "n_rows_compared": int(sum(len(ev_a[p]["rows"]) for p in ev_a)),
         "null_path_taken": "null_repair" in Bm.rec and Bm.rec["null_repair"]["n_a"] == 0,
         "control_n_a_100_differs": not bool(torch.equal(A.obj.det_w, C100.obj.det_w)), "null_repair_record_n_a_100": {k: v for k, v in C100.rec["null_repair"].items() if k != "prompt_ids"}}
    ok = all(eq.values()) and d["eta_equal"] and d["gate_sig_equal"] and dec_same and d["null_path_taken"] and d["control_n_a_100_differs"]
    return ok, d, "identity (d): the null-repair variant with n_a = 0 is the default method bit for bit"


def t_gate():
    res = {}
    ok = True
    rows = L.rows["request"] + L.rows["paraphrase"] + L.rows["neighborhood"] + L.benign("heldout")[:300] + L.benign("repair")[:100]
    tids = list(dict.fromkeys(r["text_id"] for r in rows))
    for M in e10t.methods.rowset(L, draw=0, labels=("rescope", "rescope_shared", "cosine", "cosine_published", "exact")) + [e10t.methods.where(L, "two_block")]:
        S = M.obj
        d = e10t.gate.decide(L, S, tids)
        n_f = n_rel = n_band = n_cand = 0
        for i, t in enumerate(tids):
            k = torch.from_numpy(np.array(L.kc.keys(t)))
            f = S.firing(k)                                             # the library's decision on the query's keys (one sequence)
            fired = f is not None
            if d["near"][i]:
                n_band += 1
                continue
            n_f += int(fired != bool(d["fired"][i]))
            if fired and d["fired"][i]:
                if S.fire == "top1":
                    n_rel += int(sorted(zip(f.pos.tolist(), f.edit.tolist())) != sorted(d["rel"][i]))
                n_cand += int(int(S._edit_any(f.cand).any(0).sum()) != int(d["n_cand_edits"][i]))
        res[M.label] = {"n_queries": len(tids), "n_fired": int(d["fired"].sum()), "n_in_band": n_band, "fired_mismatch": n_f,
                        "released_pattern_mismatch": n_rel, "candidate_edit_count_mismatch": n_cand}
        ok &= n_f == 0 and n_rel == 0 and n_cand == 0
    return ok, res, "gate-level decisions (per query and per position) = the library's ScopedLog.firing on the cached keys"


def t_cache():
    kc = L.kc
    d = {"parts": {}}
    ok = True
    for p in kc.parts:
        m = kc.meta[p]
        v = e10t.data.jload(os.path.join(kc.dir, p, "verify.json")) if os.path.exists(os.path.join(kc.dir, p, "verify.json")) else None
        d["parts"][p] = {"n_texts": m["n_texts"], "n_positions": m["n_positions"], "dtype": m["dtype"], "gb": m["n_positions"] * 6400 * 4 / 2 ** 30,
                         "single_pass_recompute_max_rel_deviation": None if v is None else v["max_rel_deviation"], "n_verified": None if v is None else v["n_sample"]}
        ok &= v is not None and v["max_rel_deviation"] < 1e-5
    need = {r["text_id"] for r in L.all_rows()}
    d["texts_needed_by_dev_log"] = len(need)
    d["texts_missing"] = len(kc.missing(need))
    d["positions_ok"] = all(kc.keys(t).shape == (len(L.ids(t)) + e10t.N_DEC, 6400) for t in list(need)[:500])
    # float16 rounding of the cached keys against the decision band (why the cache is float32)
    S, _ = e10t.methods.base_gate(L, "ncoord")
    wn = S.det_w.norm(dim=1) / S.det_s
    tids = [r["text_id"] for r in (L.rows["request"] + L.benign("heldout")[:200])]
    mx = 0.0
    for t in tids:
        k = torch.from_numpy(np.array(kc.keys(t)))
        r32, r16 = S.scores(k), S.scores(k.half().float())
        mx = max(mx, float(((r32 - r16).abs() / (k.double().norm(dim=1, keepdim=True) * wn[None, :])).max()))
    d["float16_rounding_max_dr_over_scale"] = mx
    d["band"] = e10t.BAND
    d["float16_exceeds_band"] = mx > e10t.BAND
    ok &= d["texts_missing"] == 0 and d["positions_ok"]
    return ok, d, "key cache complete for the development log; float32 keys agree with a single-pass recomputation far inside the band; float16 would not"


def t_rows():
    rows = e10t.methods.rowset(L, draw=0)
    d = {"labels": [M.label for M in rows], "kinds": [M.kind for M in rows], "keys": [M.key for M in rows],
         "own_requests": {M.label: M.rec.get("own_requests") for M in rows if M.kind == "gate"},
         "capped_edits": {M.label: M.rec["cal"].get("n_edits_capped") for M in rows if M.kind == "gate" and "cal" in M.rec},
         "undefined": e10t.methods.UNDEFINED}
    rows1 = e10t.methods.rowset(L, draw=1)
    d["keys_change_with_draw"] = {a.label: a.key != b.key for a, b in zip(rows, rows1)}
    W = {c: e10t.methods.where(L, c) for c in ("none", "two_block", "logonly")}
    d["where"] = {c: {"label": m.label, "fire": m.obj.fire, "response": m.obj.response, "thresholds": m.rec["thresholds"], "own_requests": m.rec["own_requests"]} for c, m in W.items()}
    L50 = e10t.open_log(e10t.DEV_LOG, T=50)
    M50 = e10t.methods.rescope(L50, draw=0)
    d["prefix_T50"] = {"T": M50.obj.T, "own_requests": M50.rec["own_requests"], "n_rows": {k: len(v) for k, v in L50.rows.items()}}
    try:
        e10t.methods.rowset(L, labels=("ball",))
        refused = False
    except KeyError as e:
        refused = "GRACE" in str(e)
    d["undefined_row_refused_with_reason"] = refused
    r14 = e10t.methods.rowset(L, draw=0, labels=e10t.methods.ROWSET_R14)
    d["labels_r14"] = [M.label for M in r14]
    d["where_rows"] = {M.label: {"fire": M.obj.fire, "response": M.obj.response, "values": M.obj.meta.get("values"), "own_requests": M.rec["own_requests"],
                                 "capped": M.rec["cal"].get("n_edits_capped")} for M in r14[-2:]}
    ok = (d["labels"] == list(e10t.methods.ROWSET) and all(v["n_not_candidate"] == 0 for v in d["own_requests"].values())
          and d["labels_r14"] == list(e10t.methods.ROWSET_R14) and all(v["fire"] == "all" and v["response"] == "lin" for v in d["where_rows"].values())
          and all(d["keys_change_with_draw"][k] == (k in e10t.methods.DEPENDS_ON_DRAW) for k in d["keys_change_with_draw"])
          and W["none"].rec["thresholds"]["n_neg_inf"] == L.T and refused and M50.obj.T == 50 and M50.rec["own_requests"]["n_not_candidate"] == 0)
    return ok, d, "the eight rows build; calibrated rows depend on the draw, the others do not; own requests are candidates; undefined rows are refused with their reason"


def t_pack():
    """Edit sets of the hook's rows are lossless whatever their size (a long log can make hundreds of edits candidates on one row):
    pack_edits / edit_in round trip, and the row hook on a synthetic module where all 80 edits are candidates at every position."""
    from e10t import backbone as BB
    from rescope import ScopedLog, synthetic_log
    rng = np.random.default_rng(0)
    rt = True
    for n in (0, 1, 64, 65, 700):
        s = sorted(int(x) for x in rng.choice(2000, n, replace=False))
        p = BB.pack_edits(s)
        back = [e for e in range(2000) if BB.edit_in(p, e)]
        rt &= back == s and isinstance(p, dict) == (n > 64)
    log, truth = synthetic_log(T=80, d=32, m=8, batch=1, seed=1)
    S = ScopedLog.from_log(log, role="all", fire="all", response="lin")           # thresholds -inf: every edit is a candidate everywhere

    class Lin(torch.nn.Module):
        def __init__(self, W):
            super().__init__()
            self.weight = torch.nn.Parameter(W.clone())

        def forward(self, x):
            return x @ self.weight.T
    mod = Lin(truth["W0"])
    hk = BB.RowHook(mod, S)
    x = torch.randn(3, 5, 32, dtype=torch.float64)
    base = mod(x).detach()                                                         # the module's own output, same batch, no hook
    with hk:
        hk.start(3, torch.tensor([5, 3, 4]))                                       # a right-padded batch: rows of 5, 3, 4 real positions
        out = mod(x)
        hk.cont(None)
        mod(x[:, :1])                                                              # a key-value-cache step: one more position per row
        rows = hk.rows(own=[7, 70, None])
    lin_ok = bool(torch.allclose(out[0], x[0] @ truth["W0"].T + (x[0] @ log.W.T) @ log.U, atol=1e-9))
    d = {"pack_roundtrip": bool(rt), "n_fire_pos": [r["n_fire_pos"] for r in rows], "n_cand_edits": [r["n_cand_edits"] for r in rows],
         "cand_edits_is_mask": [isinstance(r["cand_edits"], dict) for r in rows], "own_cand": [r.get("own_cand") for r in rows],
         "own_released": [r.get("own_released") for r in rows], "other_released": [r.get("other_released") for r in rows],
         "linear_response_equals_log_terms": lin_ok, "pad_positions_bitwise_the_modules_own_output": bool(torch.equal(out[1, 3:], base[1, 3:]) and torch.equal(out[2, 4:], base[2, 4:])),
         "real_positions_changed": bool(not torch.equal(out[1, :3], base[1, :3]))}
    ok = (rt and d["n_fire_pos"] == [6, 4, 5] and d["n_cand_edits"] == [80, 80, 80] and all(d["cand_edits_is_mask"]) and d["own_cand"][:2] == [True, True]
          and d["own_released"][:2] == [True, True] and d["other_released"][:2] == [True, True] and lin_ok and d["pad_positions_bitwise_the_modules_own_output"]
          and d["real_positions_changed"])
    return ok, d, "edit sets in the hook's rows are lossless at any size; PAD positions of a padded batch never fire; key-value-cache steps continue the rows"


def t_guard():
    """Guards as served (e10t.methods.guard_margin), 20 calibration draws: only thresholds that sat at a guard cap are lowered, by a
    round-off-sized amount; afterwards no own request lies inside the decision band; nothing else changes."""
    tot = {}
    ok = True
    for d in range(e10t.N_DRAWS):
        for lab in ("rescope", "rescope_shared", "rescope_null", "cosine", "where_twoblock"):
            M, M0 = e10t.methods.build(lab, L, draw=d), e10t.methods.build(lab, L, draw=d, served_guards=False)
            g = M.rec["cal"]["guard_margin"]
            t = tot.setdefault(lab, {"n_capped": 0, "n_lowered": 0, "n_lowered_at_a_cap": 0, "max_relative_lowering": 0.0, "own_requests_in_band_before": 0,
                                     "own_requests_in_band_after": 0, "own_not_candidate_after": 0, "n_other_thresholds_changed": 0})
            t["n_capped"] += M.rec["cal"]["n_edits_capped"]
            t["n_lowered"] += g["n_thresholds_lowered"]
            t["n_lowered_at_a_cap"] += g["n_lowered_that_sat_at_a_guard_cap"]
            t["max_relative_lowering"] = max(t["max_relative_lowering"], g["max_relative_lowering"])
            t["own_requests_in_band_before"] += M0.rec["own_requests"]["n_in_band"]
            t["own_requests_in_band_after"] += M.rec["own_requests"]["n_in_band"]
            t["own_not_candidate_after"] += M.rec["own_requests"]["n_not_candidate"]
            ch = M.obj.det_tau != M0.obj.det_tau
            t["n_other_thresholds_changed"] += int(ch.sum()) - g["n_thresholds_lowered"]
            ok &= bool((M.obj.det_tau <= M0.obj.det_tau).all())
    for t in tot.values():
        ok &= (t["n_lowered"] == t["n_lowered_at_a_cap"] == t["n_capped"] and t["own_requests_in_band_after"] == 0 and t["own_not_candidate_after"] == 0
               and t["n_other_thresholds_changed"] == 0 and t["max_relative_lowering"] < 1e-3)
    M, M0 = e10t.methods.where(L, "logonly"), e10t.methods.where(L, "logonly", served_guards=False)
    tot["where_logonly"] = {"n_capped": M.rec["cal"]["n_edits_capped"], "n_lowered": M.rec["cal"]["guard_margin"]["n_thresholds_lowered"],
                            "own_requests_in_band_before": M0.rec["own_requests"]["n_in_band"], "own_requests_in_band_after": M.rec["own_requests"]["n_in_band"]}
    ok &= tot["where_logonly"]["own_requests_in_band_after"] == 0 and sum(t["n_capped"] for t in tot.values()) > 0
    return ok, tot, "guards as served: exactly the capped thresholds are lowered, by round-off-sized amounts; no own request is left inside the decision band"


TESTS = {"pack": t_pack, "log": t_log, "guard": t_guard, "rebase": t_rebase, "draws": t_draws, "calib": t_calib, "null0": t_null0, "gate": t_gate, "cache": t_cache, "rows": t_rows}
if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS)
    fails = 0
    for nm in names:
        t0 = time.time()
        try:
            ok, detail, what = TESTS[nm]()
            err = None
        except Exception as e:  # noqa: BLE001
            import traceback
            ok, detail, what, err = False, {}, "", traceback.format_exc()
        res = {"name": f"c_{nm}", "pass": bool(ok), "what": what, "detail": detail, "error": err, "seconds": time.time() - t0, "code": e10t.code_fingerprint(),
               "log": e10t.DEV_LOG, "written": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        import fx
        e10t.data.jdump(fx.io.clean(res), os.path.join(OUT, f"cpu_{nm}.json"))
        fails += int(not ok)
        print(f"{'PASS' if ok else 'FAIL'}  c_{nm}  {time.time() - t0:.1f}s  {what}" + (f"\n{err}" if err else ""), flush=True)
    sys.exit(1 if fails else 0)
