"""Tests of white.scores: the gate features of R3 as score matrices and as detectors of the library's scoped log."""
from __future__ import annotations

import json
import os
import time

import numpy as np

import naive
import tcommon as T
from tcommon import C, test
from white import scores as WS
from white import wstats as WSt

torch = C.use_library()

from rescope import (Hook, add_guards, calibrate_shared, calibrate_two_block, decision_band, feature_gate, least_repair, null_repair,  # noqa: E402
                     synthetic_log, witness_repair)

TOL_ALG = 1e-9               # algebraic identities on small / well-conditioned problems (round-off is 1e-13 or below)
TOL_NAIVE = 1e-8             # two independent float64 code paths at d = 11008: cond(C_W + rho I) <= 6e5, cancellation <= 2e3 (wstats / README)
TESTBED = C.ROOT + "/02_research/experiments/06_testbed_v3"


# ----------------------------------------------------------------------------- fixtures
def _syn(seed=0, T_=24, d=96):
    log, truth = synthetic_log(T=T_, d=d, m=20, batch=4, seed=seed)
    g = torch.Generator().manual_seed(100 + seed)
    base = torch.randn(d, generator=g, dtype=torch.float64)
    ref_small = 0.9 * base[None, :] + torch.randn(40, d, generator=g, dtype=torch.float64)       # N < d: Gram form
    ref_big = 0.9 * base[None, :] + torch.randn(150, d, generator=g, dtype=torch.float64)        # N > d: primal form
    Q = 0.7 * base[None, :] + torch.randn(17, d, generator=g, dtype=torch.float64)
    Q = torch.cat([Q, log.K[:5] + 0.05 * torch.randn(5, d, generator=g, dtype=torch.float64)])   # some queries close to an edit
    B = torch.randn(10, d, generator=g, dtype=torch.float64)                                     # keys of a null repair
    return log, truth, ref_small, ref_big, Q, B, g


def _dev():
    return list(C.DEV_LOGS.values())


def _variants(ctx):
    """[(label, un-thresholded gate without own-prompt guards)] of the 8 registered variants + the two null-repaired default gates."""
    out = [(lab, WS.feature_gate(ctx.log, feat, stats=ctx.stats, kappa=kappa, null_keys=ctx.null_keys)) for lab, feat, kappa in WS.variants()]
    base = feature_gate(ctx.log, "ncoord")
    out.append(("rescope_null", null_repair(base, ctx.null_keys, ridge=0.0)[0]))
    out.append(("rescope_nullW", WS.null_w(base, stats=ctx.stats)[0]))
    return out


def _sample_queries(ctx, n_each=(("heldout_disjoint", 15), ("tgen", 10), ("igen", 10), ("rel", 5), ("hn_i", 4), ("hn_q", 3), ("hn_x", 3))):
    """50 seeded sequences of a development log (15 held-out probes, 10 T-Gen, 10 I-Gen, 5 Rel, 10 hard negatives): positions in the index."""
    rng = np.random.default_rng(T.SEED)
    rows = []
    for nm, k in n_each:
        pool = ctx.sets[nm]
        rows += [pool[int(i)] for i in np.sort(rng.choice(len(pool), k, replace=False))]
    return rows


# ----------------------------------------------------------------------------- own-key values
@test("scores", "own-key value on synthetic logs: every score is 1 at the edit's own solver key (Gram and primal form, registered and other ridges)")
def c1_own_key_synthetic():
    log, _, ref_s, ref_b, Q, B, _ = _syn()
    worst, n, bounded = 0.0, 0, True
    for ref in (ref_s, ref_b):
        st = WSt.KeyStats.from_keys(ref)
        cases = [("cos_centred", {"stats": st}), ("cos_null", {"null_keys": B})]
        cases += [(f, {"stats": st, "kappa": k}) for f in ("white", "whitecos") for k in (0.01, 0.1, 1.0, 5.0)]
        for feat, kw in cases:
            S = WS.feature_gate(log, feat, **kw)
            one = WS.own_scores(S)                                        # every key as its own one-key query
            diag = WS.scores(log, log.K, feat, **kw).diagonal()           # the same keys as one batch
            worst = max(worst, float((one - 1).abs().max()), float((diag - 1).abs().max()))
            n += 1
            if feat in ("cos_centred", "whitecos"):                       # true cosines: no query scores above 1
                bounded &= bool((S.scores(torch.cat([Q, log.K])) <= 1 + 1e-12).all())
        nc = WS.ncoord_null(log, log.K, B).diagonal()
        worst = max(worst, float((nc - 1).abs().max()))
    return {"pass": worst <= TOL_ALG and bounded, "worst_abs_deviation_from_1": worst, "n_gates": n, "cos_centred_and_whitecos_bounded_by_1": bounded,
            "note": f"{n} gates + ncoord_null, T = 24, d = 96: max |score(k_i)_i - 1| = {worst:.1e}; cos_centred / whitecos <= 1 on all queries: {bounded}"}


@test("scores", "own-key value on the development logs: 8 registered variants, rescope_null (100 REPAIR keys) and rescope_nullW (600 BEN-W keys)")
def c2_own_key_real():
    out, worst = {}, 0.0
    for name in _dev():
        ctx = T.context(name)
        e = {}
        for lab, S in _variants(ctx):
            e[lab] = float((WS.own_scores(S) - 1).abs().max())
            worst = max(worst, e[lab])
        Sc = feature_gate(ctx.log, "cos")                                 # which key does the library's cosine gate use?
        K = WS.solver_keys(ctx.log)
        fac = torch.load(os.path.join(TESTBED, name, "edit_factors.pt"), map_location="cpu")
        e["keys_are_the_cosine_gate_keys"] = bool(torch.equal(K, Sc.det_key) and torch.equal(Sc.det_w, K / K.norm(dim=1, keepdim=True))
                                                  and torch.equal(K, fac["K1_solver"].double()))
        e["cosine_gate_key_is_not_the_B1_key"] = not bool(torch.equal(K, ctx.own[0].double()))
        out[name] = e
    ok = worst <= TOL_ALG and all(v["keys_are_the_cosine_gate_keys"] for v in out.values())
    return {"pass": ok, "per_log": out, "worst_abs_deviation_from_1": worst,
            "note": f"2 logs x 10 gates x 100 edits: max |score(k_i)_i - 1| = {worst:.1e}; k_i = EditLog.K = the file's K1_solver = det_key (and det_w |k_i|) of the library's "
                    f"cosine gate: {ok}"}


# ----------------------------------------------------------------------------- invariances and limits
@test("scores", "invariances: translation / rotation / scaling (cos_centred), linearity and joint scaling (white), query scaling and Cauchy-Schwarz (whitecos), null-key shifts and change of basis (cos_null)")
def c3_invariances():
    log, _, ref_s, ref_b, Q, B, g = _syn(seed=1)
    K, d = log.K, log.d
    res = {}
    dev = lambda a, b: float((a - b).abs().max())                         # noqa: E731
    O, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))          # a random rotation
    t = 3.0 * torch.randn(d, generator=g, dtype=torch.float64)
    for nm, ref in (("gram", ref_s), ("primal", ref_b)):
        st = WSt.KeyStats.from_keys(ref)
        e = {}
        # cos_centred
        r0 = WS.cos_centred(log, Q, st)
        e["cos_centred: translation of all keys"] = dev(WS.cos_centred(K + t, Q + t, WSt.KeyStats.from_keys(ref + t)), r0)
        e["cos_centred: rotation of all keys"] = dev(WS.cos_centred(K @ O, Q @ O, WSt.KeyStats.from_keys(ref @ O)), r0)
        e["cos_centred: query scaled about mu"] = dev(WS.cos_centred(log, st.mu + 3.7 * (Q - st.mu), st), r0)
        for kappa in (0.01, 1.0):
            tag = f"kappa {kappa:g}"
            w0, c0 = WS.white(log, Q, st, kappa), WS.whitecos(log, Q, st, kappa)
            a, b = 1.7, -0.6
            e[f"white {tag}: linear in the query"] = dev(WS.white(log, a * Q[:10] + b * Q[10:20], st, kappa), a * w0[:10] + b * w0[10:20])
            e[f"white {tag}: joint scaling of all keys"] = dev(WS.white(2.5 * K, 2.5 * Q, WSt.KeyStats.from_keys(2.5 * ref), kappa), w0)
            e[f"white {tag}: rotation of all keys"] = dev(WS.white(K @ O, Q @ O, WSt.KeyStats.from_keys(ref @ O), kappa), w0)
            e[f"whitecos {tag}: query scaling"] = dev(WS.whitecos(log, 4.2 * Q, st, kappa), c0)
            e[f"whitecos {tag}: joint scaling of all keys"] = dev(WS.whitecos(2.5 * K, 2.5 * Q, WSt.KeyStats.from_keys(2.5 * ref), kappa), c0)
            e[f"whitecos {tag}: rotation of all keys"] = dev(WS.whitecos(K @ O, Q @ O, WSt.KeyStats.from_keys(ref @ O), kappa), c0)
            wh = st.whitener(kappa=kappa)
            e[f"whitecos {tag}: = white * sqrt(k_i'Mk_i / k'Mk)"] = dev(w0 * (wh.quad(K)[None, :] / wh.quad(Q)[:, None]).sqrt(), c0)
            e[f"whitecos {tag}: |score| <= 1 (excess)"] = max(0.0, float(c0.abs().max()) - 1.0)
        res[nm] = e
    # cos_null
    S = WS.feature_gate(log, "cos_null", null_keys=B)
    r0 = S.scores(Q)
    e = {"cos_null: query scaling": dev(WS.cos_null(log, 0.3 * Q, B), r0)}
    Cm = torch.randn(Q.shape[0], B.shape[0], generator=g, dtype=torch.float64)
    Q2 = Q + Cm @ B                                                        # the query moved inside the span of the null keys
    e["cos_null: numerator unchanged by adding null keys"] = dev(S.scores(Q2) * Q2.norm(dim=1, keepdim=True), r0 * Q.norm(dim=1, keepdim=True))
    A = torch.randn(B.shape[0], B.shape[0], generator=g, dtype=torch.float64) + 3 * torch.eye(B.shape[0], dtype=torch.float64)
    e["cos_null: other basis of the same span"] = dev(WS.cos_null(log, Q, A @ B), r0)
    e["cos_null: null keys permuted and repeated"] = dev(WS.cos_null(log, Q, torch.cat([B.flip(0), B[:4]])), r0)
    n0 = WS.ncoord_null(log, Q, B)
    e["ncoord_null: other basis of the same span"] = dev(WS.ncoord_null(log, Q, A @ B), n0)
    res["null"] = e
    # limits that tie the features to the registered cosine gate
    cos = feature_gate(log, "cos").scores(Q)
    st = WSt.KeyStats.from_keys(ref_s)
    lim = {"cos_centred with mu = 0 == cos": dev(WS.cos_centred(log, Q, torch.zeros(d, dtype=torch.float64)), cos),
           "whitecos, kappa = 1e12 == cos": dev(WS.whitecos(log, Q, st, 1e12), cos),
           "white, kappa = 1e12 == <k_i, k> / |k_i|^2": dev(WS.white(log, Q, st, 1e12), (Q @ K.T) / (K * K).sum(1)[None, :]),
           "cos_null with no null key == cos": dev(WS.cos_null(log, Q, torch.zeros(0, d, dtype=torch.float64)), cos)}
    res["limits"] = lim
    worst = max(v for e in res.values() for v in e.values())
    wl = max(lim.values())
    return {"pass": worst <= 1e-7 and max(v for k, e in res.items() if k != "limits" for v in e.values()) <= TOL_ALG, "checks": res,
            "worst_abs_deviation": worst, "n_checks": sum(len(e) for e in res.values()),
            "note": f"{sum(len(e) for e in res.values())} identities on a synthetic log (Gram and primal form): worst deviation {max(v for k, e in res.items() if k != 'limits' for v in e.values()):.1e}; "
                    f"limits mu = 0 / kappa = 1e12 / no null key reproduce the cosine gate to {wl:.1e}"}


def _kkt(Wn, W, K, B):
    """Per edit: relative size of the part of (w' - w) outside span{null keys, own key} (0 <=> w' is THE minimum-change solution)."""
    Qb, _ = torch.linalg.qr(B.double().T)                                 # (d, n) orthonormal basis of the null keys
    D = Wn - W
    D1 = D - (D @ Qb) @ Qb.T
    K1 = K - (K @ Qb) @ Qb.T
    res = D1 - ((D1 * K1).sum(1) / (K1 * K1).sum(1))[:, None] * K1
    return res.norm(dim=1) / D.norm(dim=1).clamp_min(1e-300)


@test("scores", "cos_null (the 100 keys of rescope_null, draw 0) and the rescope_nullW helper (600 BEN-W keys): zero on the given keys, own coordinate unchanged, minimum change")
def c4_null_zero_and_own():
    out, ok = {}, True
    for name in _dev():
        ctx = T.context(name)
        K, kn = ctx.log.K, ctx.log.K.norm(dim=1)
        Bn = ctx.null_keys.double()
        S0 = feature_gate(ctx.log, "cos")
        S = WS.feature_gate(ctx.log, "cos_null", null_keys=ctx.null_keys)
        r_null = S.scores(ctx.null_keys)                                  # (100, T): cosine-type coordinates of the null keys
        own_raw = (S.det_w * K).sum(1)                                    # <w_i', k_i>, to be |k_i| = <k_i / |k_i|, k_i>
        from rescope.scoped import LIN
        lin = S0.replace(det_s=kn, det_kind=torch.full((ctx.T,), LIN, dtype=torch.long))
        with WSt.single_thread():                                         # like for like: feature_gate builds its detectors with ONE thread
            S_dir, _ = null_repair(lin, ctx.null_keys, ridge=0.0)         # the direct library call, WITH the guards scored
        S_free, _ = null_repair(lin, ctx.null_keys, ridge=0.0)            # the same call under the process's thread setting (information)
        e = {"n_null_keys": int(Bn.shape[0]), "null_keys_are_rescope_nulls": ctx.null_rows == list(ctx.fx.draws.null_sample(ctx.L, 100, 0)) if ctx.fx else None,
             "detectors_bitwise_equal_to_the_direct_library_call": bool(torch.equal(S_dir.det_w, S.det_w)),
             "library_call_not_pinned": {"threads": torch.get_num_threads(), "bitwise_equal": bool(torch.equal(S_free.det_w, S.det_w)),
                                         "max_relative_difference": float((S_free.det_w - S.det_w).abs().max() / S.det_w.abs().max())},
             "guards_kept": bool(torch.equal(S.guard_key, S0.guard_key) and torch.equal(S.guard_edit, S0.guard_edit)),
             "draw_is_the_library_split": bool(ctx.draw_equals_library), "null_sample_is_positions_200_299_of_the_ranking": bool(ctx.null_equals_local_rule),
             "null_keys_disjoint_from_calibration_draw": not (set(ctx.null_rows) & set(ctx.draw_rows(0))),
             "cos_null": {"max_abs_score_on_null_keys": float(r_null.abs().max()),
                          "max_abs_coordinate_on_null_keys_relative": float(((S.det_w @ Bn.T).abs() / (S.det_w.norm(dim=1)[:, None] * Bn.norm(dim=1)[None, :])).max()),
                          "own_coordinate_rel_change": float(((own_raw - kn) / kn).abs().max()),
                          "own_score_abs_dev_from_1": float((WS.own_scores(S) - 1).abs().max()),
                          "not_minimum_change_rel": float(_kkt(S.det_w, S0.det_w, K, Bn).max()),
                          "score_before_repair_on_null_keys_max": float(S0.scores(ctx.null_keys).max()),
                          "detector_norm_max": float(S.det_w.norm(dim=1).max())}}
        base = feature_gate(ctx.log, "ncoord")
        for lab, S2, Bx in (("rescope_null", null_repair(base, ctx.null_keys, ridge=0.0)[0], Bn), ("rescope_nullW", WS.null_w(base, stats=ctx.stats)[0], ctx.stats.keys.double())):
            own2 = (S2.det_w * K).sum(1)
            e[lab] = {"n_null_keys": int(Bx.shape[0]), "max_abs_score_on_null_keys": float(S2.scores(Bx).abs().max()),
                      "own_coordinate_rel_change": float(((own2 - base.det_s) / base.det_s).abs().max()),
                      "own_score_abs_dev_from_1": float((WS.own_scores(S2) - 1).abs().max()),
                      "not_minimum_change_rel": float(_kkt(S2.det_w, base.det_w, K, Bx).max()),
                      "score_before_repair_on_null_keys_max": float(base.scores(Bx).max())}
        e["null_w_keys_are_the_BEN-W_decode_keys"] = bool(torch.equal(ctx.stats.keys, ctx.kc.keys("decode", [ctx.kc.by_sid(s) for s in ctx.stats.sids])))
        Sg, _ = add_guards(base, *ctx.own)                                 # 2 T guards, as the driver's base gate
        A1, r1 = WS.null_w(Sg, stats=ctx.stats)
        A2, r2 = WS.null_w(Sg, stats=ctx.stats, score_guards=False)
        e["null_w_without_guard_scoring"] = {"detectors_bitwise_equal": bool(torch.equal(A1.det_w, A2.det_w)), "guards_kept": bool(torch.equal(A2.guard_key, Sg.guard_key)),
                                             "n_guards": int(Sg.guard_key.shape[0]), "n_guards_lost_with_scoring": r1["n_guards_lost"],
                                             "record_has_no_guard_counts": not any(k.startswith("n_guards_") for k in r2)}
        good = (all(e[k]["max_abs_score_on_null_keys"] <= TOL_ALG and e[k]["own_coordinate_rel_change"] <= 1e-12 and e[k]["own_score_abs_dev_from_1"] <= 1e-12
                    and e[k]["not_minimum_change_rel"] <= 1e-7 for k in ("cos_null", "rescope_null", "rescope_nullW"))
                and e["null_keys_disjoint_from_calibration_draw"] and e["null_w_keys_are_the_BEN-W_decode_keys"] and e["null_keys_are_rescope_nulls"] is not False
                and e["detectors_bitwise_equal_to_the_direct_library_call"] and e["guards_kept"] and e["draw_is_the_library_split"]
                and e["null_sample_is_positions_200_299_of_the_ranking"] and e["null_w_without_guard_scoring"]["detectors_bitwise_equal"]
                and e["null_w_without_guard_scoring"]["guards_kept"] and e["null_w_without_guard_scoring"]["record_has_no_guard_counts"])
        ok &= good
        out[name] = e
    q = out[C.DEV_LOGS["qwen25vl"]]
    w = lambda k: max(v[lab][k] for v in out.values() for lab in ("cos_null", "rescope_null", "rescope_nullW"))          # noqa: E731
    unp = max(v["library_call_not_pinned"]["max_relative_difference"] for v in out.values())
    return {"pass": ok, "per_log": out,
            "note": f"both logs: max |score| on the given keys {w('max_abs_score_on_null_keys'):.1e} (before the repair up to {q['cos_null']['score_before_repair_on_null_keys_max']:.2f} for the cosine), "
                    f"own coordinate changed by {w('own_coordinate_rel_change'):.1e} (relative), part of the change outside span(null keys, own key) {w('not_minimum_change_rel'):.1e}; "
                    f"cos_null detectors == the direct library null_repair with guards, bit for bit at one thread (the library's unpinned call at "
                    f"{q['library_call_not_pinned']['threads']} threads differs by {unp:.1e} relative: thread-dependent last bits)"}


# ----------------------------------------------------------------------------- second, naive code path
def _naive_log(name: str) -> dict:
    """Every score of a development log on 50 queries by the naive numpy / scipy path (tests/naive.py) against white.scores."""
    ctx = T.context(name)
    rows = _sample_queries(ctx)
    Q32 = ctx.kc.K("tf")[torch.tensor([int(ctx.kc.index[j]["row_tf"]) for j in rows])]           # teacher-forced keys of the 50 queries
    fac = torch.load(os.path.join(TESTBED, name, "edit_factors.pt"), map_location="cpu")         # the log's own file: solver keys and read-outs
    Kn, Wn, Qn = fac["K1_solver"].numpy().astype(np.float64), fac["W"].numpy().astype(np.float64), Q32.numpy().astype(np.float64)
    with open(os.path.join(C.PARTS_DIR, f"{ctx.backbone}.json")) as f:                           # BEN-W keys straight from the parts file and the key cache
        parts = json.load(f)["parts"]
    ben = [e for e in ctx.kc.index if e["kind"] == "benign" and parts[e["sid"]] == "BEN-W"]
    Kw = ctx.kc.K("decode")[torch.tensor([int(e["row_decode"]) for e in ben])].numpy().astype(np.float64)
    Bn = ctx.null_keys.numpy().astype(np.float64)
    res, t0 = {}, time.time()

    def cmp(label, mine, ref):
        res[label] = {"max_abs_deviation": T.maxabs(mine.numpy(), ref), "max_abs_score": float(np.abs(ref).max()), "shape": list(ref.shape)}

    cmp("cosine (registered; library gate)", feature_gate(ctx.log, "cos").scores(Q32), naive.cosine(Kn, Qn))
    cmp("cos_centred", WS.cos_centred(ctx.log, Q32, ctx.stats), naive.cos_centred(Kn, Qn, Kw))
    Cm, tr = naive.second_moment(Kw)                                       # the explicit d x d second moment (0.97 GB)
    res["tr(C_W) relative deviation"] = abs(tr - ctx.stats.trace) / tr
    canc = {}
    for kappa in C.KAPPAS:
        w, wc, rho = naive.whitened(Kn, Qn, Cm, tr, kappa)
        tag = WSt.kappa_tag(kappa)
        cmp(f"white_{tag}", WS.white(ctx.log, Q32, ctx.stats, kappa), w)
        cmp(f"whitecos_{tag}", WS.whitecos(ctx.log, Q32, ctx.stats, kappa), wc)
        res[f"rho_{tag} relative deviation"] = abs(rho - ctx.stats.rho(kappa)) / rho
        wh = ctx.stats.whitener(kappa=kappa)
        c = (Q32.double() ** 2).sum(1) / (wh.rho * wh.quad(Q32))
        canc[tag] = [float(c.min()), float(c.median()), float(c.max())]
    del Cm
    solves = {}                                                            # what the naive constrained solves did (rank, gap, residual)
    ref, solves["cos_null (100 REPAIR keys)"] = naive.cos_null(Kn, Qn, Bn)
    cmp("cos_null (100 REPAIR keys)", WS.cos_null(ctx.log, Q32, ctx.null_keys), ref)
    ref, solves["ncoord_null (100 REPAIR keys)"] = naive.ncoord_null(Wn, Kn, Qn, Bn)
    cmp("ncoord_null (100 REPAIR keys) = rescope_null", WS.ncoord_null(ctx.log, Q32, ctx.null_keys), ref)
    ref, solves["ncoord_null (600 BEN-W keys)"] = naive.ncoord_null(Wn, Kn, Qn, Kw)
    cmp("ncoord_null (600 BEN-W keys) = rescope_nullW", WS.ncoord_null(ctx.log, Q32, ctx.stats.keys), ref)
    cmp("null_w helper (600 BEN-W keys)", WS.null_w(feature_gate(ctx.log, "ncoord"), stats=ctx.stats)[0].scores(Q32), ref)
    lib_rank = {"cos_null (100 REPAIR keys)": WS.feature_gate(ctx.log, "cos_null", null_keys=ctx.null_keys).meta["white"]["null_repair"]["rank_benign"],
                "ncoord_null (100 REPAIR keys)": null_repair(feature_gate(ctx.log, "ncoord"), ctx.null_keys)[1]["rank_benign"],
                "ncoord_null (600 BEN-W keys)": WS.null_w(feature_gate(ctx.log, "ncoord"), stats=ctx.stats)[1]["rank_benign"]}
    for k, v in solves.items():
        v["rank_of_the_library_repair"] = int(lib_rank[k])
    # the reference must BE a solution of its defining problem (consistent system) and must have made the library's rank decision
    ref_ok = all(v["max_relative_residual"] <= TOL_NAIVE and v["rank_min"] == v["rank_max"] == v["rank_of_the_library_repair"] == v["n_distinct_keys"]
                 and v["largest_dropped_singular_value_rel"] <= 1e-3 * naive.RCOND and v["smallest_kept_singular_value_rel"] >= 1e3 * naive.RCOND for v in solves.values())
    scores_ = {k: v for k, v in res.items() if isinstance(v, dict)}
    worst = max(v["max_abs_deviation"] for v in scores_.values())
    wl = max(scores_, key=lambda k: scores_[k]["max_abs_deviation"])
    rel = max(res[k] for k in res if k.endswith("relative deviation"))
    sn = solves["cos_null (100 REPAIR keys)"]
    return {"pass": worst <= TOL_NAIVE and rel <= 1e-12 and ref_ok and len(rows) == 50 and len(set(rows)) == 50, "log": name, "n_queries": len(rows),
            "query_sids": [ctx.kc.index[j]["sid"] for j in rows], "scores": res, "tolerance": TOL_NAIVE, "worst_abs_deviation": worst, "worst_score": wl,
            "naive_constrained_solves": solves, "naive_reference_is_a_valid_solution": ref_ok,
            "cancellation_of_query_quadratic_form_min_median_max": canc, "naive_path_seconds": time.time() - t0,
            "note": f"50 queries x 100 edits, {len(scores_)} score matrices, naive numpy path (explicit d x d inverse, constrained least squares): worst |difference| {worst:.1e} "
                    f"({wl}); tolerance {TOL_NAIVE:.0e}; the 100 null keys hold {sn['n_distinct_keys']} distinct keys, rank {sn['rank_max']} in both paths; naive solves consistent "
                    f"(residual {max(v['max_relative_residual'] for v in solves.values()):.1e}); cancellation in k'Mk of the queries at kappa 0.01: median {canc['0.01'][1]:.0f}, "
                    f"max {canc['0.01'][2]:.0f}"}


@test("scores", "naive re-implementation of every score on 50 queries of TB-Q100-ae-l2_3 (second code path: numpy / scipy, explicit d x d inverse)")
def c5_naive_qwen():
    return _naive_log(C.DEV_LOGS["qwen25vl"])


@test("scores", "naive re-implementation of every score on 50 queries of TB-L100-ae-l2_3 (second code path: numpy / scipy, explicit d x d inverse)")
def c5_naive_llava():
    return _naive_log(C.DEV_LOGS["llava"])


# ----------------------------------------------------------------------------- used by the library exactly like the solver coordinate
@test("scores", "library calibration on the new features (development logs, draw 0): stored thresholds == the defining strict comparison on every calibration score; exact rank; guards fire")
def c6_calibration_contract():
    out, ok, n_gates, exact = {}, True, 0, True
    for name in _dev():
        ctx = T.context(name)
        e = {}
        for lab, S in _variants(ctx):
            S, _ = add_guards(S, *ctx.own)
            Q = ctx.kc.coords(S, ctx.draw_rows(0), "tf")
            Qn = Q.numpy()
            r = {}
            G, rec = calibrate_two_block(S, coords=Q, alpha=0.05, seed=0, allow_silence=True)      # the literal rule
            b, eta, tau = G.det_b.numpy(), rec["eta"], G.det_tau.numpy()
            i2 = np.array(rec["block2"])
            s2 = (Qn[i2] - b[None, :]).max(1)                              # the block-2 scores, from the definition
            at = s2 == eta
            r["two_block"] = {"threshold_equals_definition": bool(np.array_equal(Qn >= tau[None, :], (Qn - b[None, :]) > eta)),
                              "eta_is_the_kth_smallest_score": bool(np.sort(s2)[rec["rank"] - 1] == eta), "n2": rec["n2"], "rank": rec["rank"],
                              "queries_firing": int((Qn[i2] >= tau[None, :]).any(1).sum()), "queries_above_eta": int((s2 > eta).sum()),
                              "eta_query_silent": bool(at.any() and not (Qn[i2][at] >= tau[None, :]).any()), "bound": rec["bound_false_firing"]}
            G, rec = calibrate_shared(S, coords=Q, alpha=0.05, allow_silence=True)
            eta, tau = rec["eta"], G.det_tau.numpy()
            s2 = Qn.max(1)
            at = s2 == eta
            r["shared"] = {"threshold_equals_definition": bool(np.array_equal(Qn >= tau[None, :], Qn > eta)),
                           "eta_is_the_kth_smallest_score": bool(np.sort(s2)[rec["rank"] - 1] == eta), "n2": rec["n2"], "rank": rec["rank"],
                           "queries_firing": int((Qn >= tau[None, :]).any(1).sum()), "queries_above_eta": int((s2 > eta).sum()),
                           "eta_query_silent": bool(at.any() and not (Qn[at] >= tau[None, :]).any()), "bound": rec["bound_false_firing"]}
            for cal, fn, kw in (("two_block", calibrate_two_block, {"seed": 0}), ("shared", calibrate_shared, {})):
                Gc, rc = fn(S, coords=Q, alpha=0.05, **kw)                 # as run: guards keep firing
                r[cal].update(guards_firing=rc["n_guards_silenced"] == 0, n_guards=rc["n_guards"], n_edits_capped=rc["n_edits_capped"],
                              class_kept=type(Gc) is type(S))
            good = all(v["threshold_equals_definition"] and v["eta_is_the_kth_smallest_score"] and v["queries_firing"] == v["queries_above_eta"] <= v["n2"] - v["rank"]
                       and v["eta_query_silent"] and v["guards_firing"] and v["n_guards"] == 2 * ctx.T and v["class_kept"] for v in r.values())
            exact &= all(v["queries_firing"] == v["n2"] - v["rank"] for v in r.values())
            ok &= good
            n_gates += 1
            e[lab] = r
        out[name] = e
    return {"pass": ok, "per_log": out, "queries_firing_equal_n2_minus_rank_everywhere": exact,
            "note": f"{n_gates} gates x 2 rules: r >= tau <=> (r - b) > eta on all 200 x 100 calibration scores; eta = the rank-th smallest score; calibration queries firing "
                    f"under the literal rule = those above eta, at most n2 - rank (4 of 100 two-block, 9 of 200 shared; exactly that many everywhere: {exact}); the query "
                    f"that defines eta is silent; all 200 guards fire as run: {ok}"}


@test("scores", "library operators on the new gate classes: replace / to / add_guards keep the class; added detectors refused; null_repair / least_repair on the 'lin' gate (white)")
def c7_operator_contract():
    log, _, ref_s, _, Q, B, _ = _syn(seed=2)
    st = WSt.KeyStats.from_keys(ref_s)
    r = {}
    for feat in ("cos_centred", "whitecos"):
        S = WS.feature_gate(log, feat, stats=st, kappa=0.1)
        S2 = S.replace(det_tau=torch.full((log.T,), 0.9, dtype=torch.float64))
        S3 = S2.to("cpu")
        Sg, _ = add_guards(S2, log.K[:3] * 1.01, [0, 1, 2])
        e = {"class": type(S).__name__, "replace_keeps_class": type(S2) is WS.TCosLog and S2.q_kind == S.q_kind,
             "to_keeps_class_and_fields": type(S3) is WS.TCosLog and bool(torch.equal(S3.scores(Q), S2.scores(Q))),
             "add_guards_keeps_class": type(Sg) is WS.TCosLog and Sg.guard_key.shape[0] == log.T + 3,
             "guard_cache_kept_by_thresholds": S.guard_scores() is not None and "guard_r" in S.replace(det_tau=S2.det_tau)._cache,
             "guard_cache_reset_on_new_geometry": "guard_r" not in (S.replace(q_mu=S.q_mu + 1.0) if feat == "cos_centred" else S.replace(q_c0=S.q_c0 * 2.0))._cache,
             # a gate with another query geometry must not share its detector tensor (the driver's feature cache is keyed by its identity)
             "new_geometry_gets_new_detector_tensor": (S.replace(q_mu=S.q_mu + 1.0) if feat == "cos_centred" else S.replace(q_c0=S.q_c0 * 2.0)).det_w is not S.det_w
             and S2.det_w is S.det_w}
        try:
            witness_repair(S2, 2, Q[0], mode="detector")
            e["witness_detector_refused"] = False
        except NotImplementedError:
            e["witness_detector_refused"] = True
        _, wr = witness_repair(S2, 2, Q[0], mode="threshold")            # lowering a threshold is defined for every feature
        e["witness_threshold_applied"] = bool(wr["applied"])
        try:
            null_repair(S, B)
            e["library_null_repair_raises"] = False
        except RuntimeError:
            e["library_null_repair_raises"] = True
        r[feat] = e
    try:                                                                  # the same holds for the library's own cosine gate: a property of null_repair
        null_repair(feature_gate(log, "cos"), B)
        r["library cosine gate"] = {"library_null_repair_raises": False}
    except RuntimeError as ex:
        r["library cosine gate"] = {"library_null_repair_raises": True, "error": str(ex)[:80]}
    Sw = WS.feature_gate(log, "white", stats=st, kappa=0.1)
    Sn, nrec = null_repair(Sw, B)
    Sl, lrec = least_repair(Sw.replace(det_tau=torch.full((log.T,), 0.2, dtype=torch.float64)), Q[:1])
    r["white"] = {"class": type(Sw).__name__, "null_repair_zero_on_keys": float(Sn.scores(B).abs().max()), "null_repair_own": float((WS.own_scores(Sn) - 1).abs().max()),
                  "least_repair_resolved": bool(lrec["resolved"])}
    Sc = WS.feature_gate(log, "cos_null", null_keys=B)
    r["cos_null"] = {"class": type(Sc).__name__, "kinds_cos": bool((Sc.det_kind == 1).all()), "scale_1": bool((Sc.det_s == 1).all())}
    G = WS.gates(log, st, null_keys=B)
    r["gates()"] = {"labels": list(G), "equal_to_variants": list(G) == [lab for lab, _, _ in WS.variants()], "meta_feature": all(g.meta["feature"] == lab for lab, g in G.items())}
    ok = (all(r[f]["replace_keeps_class"] and r[f]["to_keeps_class_and_fields"] and r[f]["add_guards_keeps_class"] and r[f]["witness_detector_refused"]
              and r[f]["witness_threshold_applied"] and r[f]["guard_cache_reset_on_new_geometry"] and r[f]["guard_cache_kept_by_thresholds"]
              and r[f]["new_geometry_gets_new_detector_tensor"]
              and r[f]["library_null_repair_raises"] for f in ("cos_centred", "whitecos"))
          and r["white"]["class"] == "ScopedLog" and r["white"]["null_repair_zero_on_keys"] <= TOL_ALG and r["white"]["null_repair_own"] <= TOL_ALG
          and r["white"]["least_repair_resolved"] and r["cos_null"]["class"] == "ScopedLog" and r["cos_null"]["kinds_cos"] and r["cos_null"]["scale_1"]
          and r["gates()"]["equal_to_variants"] and r["gates()"]["meta_feature"])
    lib = r["library cosine gate"]["library_null_repair_raises"]
    return {"pass": ok, "checks": r,
            "note": f"TCosLog survives replace / to / add_guards; witness detector refused; white is a plain 'lin' gate (null_repair, least_repair apply); library null_repair on a "
                    f"gate without 'lin' detectors raises RuntimeError (its own cosine gate too: {lib}) -- documented, cos_null goes through the 'lin' reading"}


@test("scores", "the forward hook (rescope.Hook on a linear module, CPU) with every new gate: firing, releasing edit and output W_0 k + v_e as the gate's scores say; silent sequences untouched")
def c8_hook_cpu():
    log, truth, ref_s, _, _, B, g = _syn(seed=3)
    st = WSt.KeyStats.from_keys(ref_s)
    W0 = truth["W0"]
    mod = torch.nn.Linear(log.d, log.m, bias=False, dtype=torch.float64)
    with torch.no_grad():
        mod.weight.copy_(W0)
    Tn, lp = 7, 4
    res, ok = {}, True
    for lab, S in WS.gates(log, st, null_keys=B, W0=W0).items():
        S = S.replace(det_tau=torch.full((log.T,), 0.8, dtype=torch.float64))
        e = {"n_fired": 0, "n_silent": 0, "bad": 0}
        for trial in range(12):
            x = torch.randn(1, Tn, log.d, generator=g, dtype=torch.float64)
            edit = trial % log.T
            if trial % 3 != 2:                                            # the last prompt position holds a key close to the solver key of `edit`
                x[0, lp] = log.K[edit] + 0.02 * torch.randn(log.d, generator=g, dtype=torch.float64)
            x[0, lp + 1] = log.K[(edit + 1) % log.T]                      # a later position holds ANOTHER edit's key: role 'last' must ignore it
            with torch.no_grad():
                plain = mod(x)
                r = S.scores(x[0, lp][None])[0]
                want_fire = bool((r >= S.det_tau).any())
                want_top = int(torch.where(r >= S.det_tau, r, torch.full_like(r, -float("inf"))).argmax()) if want_fire else -1
                with Hook(mod, S) as hk:
                    hk.begin(torch.tensor([lp]))
                    out = mod(x)
                    flags = hk.summary(own=edit)
            good = flags["fired"] == want_fire and flags["top_last"] == want_top and not flags["fired_later"] and not flags["fired_before"]
            if want_fire:
                e["n_fired"] += 1
                target = plain[0, lp] + S.values[want_top]
                good &= bool(torch.equal(out[0, lp], target)) and bool(torch.equal(out[0, :lp], plain[0, :lp])) and bool(torch.equal(out[0, lp + 1:], plain[0, lp + 1:]))
            else:
                e["n_silent"] += 1
                good &= bool(torch.equal(out, plain))
            e["bad"] += int(not good)
        e["weight_restored"] = bool(torch.equal(mod.weight.data, W0))
        ok &= e["bad"] == 0 and e["n_fired"] > 0 and e["n_silent"] > 0 and e["weight_restored"]
        res[lab] = e
    return {"pass": ok, "per_gate": res,
            "note": f"{len(res)} gates x 12 sequences on a synthetic log: hook firing / releasing edit == the gate's scores, output == W_0 k + v_e bit for bit, silent sequences "
                    f"untouched, a later position with another edit's key ignored: {ok}"}


@test("scores", "round-off of the scores between batch shapes (one batch vs key by key) on development-log keys stays inside the library's decision band (near flag)")
def c9_decision_band():
    out, worst = {}, 0.0
    for name in _dev():
        ctx = T.context(name)
        rows = ctx.rows[::7]                                              # every 7th evaluation sequence (about 290 keys)
        Kq = ctx.kc.keys("tf", rows)
        e = {}
        for lab, S in _variants(ctx):
            S, _ = add_guards(S, *ctx.own)
            G, _ = calibrate_two_block(S, coords=ctx.kc.coords(S, ctx.draw_rows(0), "tf"), alpha=0.05, seed=0)
            rb, r1 = G.scores(Kq), G.key_scores(Kq)
            band = decision_band(G, Kq, rb)
            ratio = float(((rb - r1).abs() / band).max())
            e[lab] = {"max_abs_difference": float((rb - r1).abs().max()), "max_difference_over_band": ratio, "band_min": float(band.min())}
            worst = max(worst, ratio)
        out[name] = e
    return {"pass": worst <= 0.1, "per_log": out, "worst_difference_over_band": worst,
            "note": f"2 logs x 10 gates x about 290 keys x 100 edits: largest |batched - single| / decision band = {worst:.1e} (the band is 1e-9 relative; criterion <= 0.1)"}


@test("scores", "determinism of the detectors: built under 1 and 4 threads they are the same bits (every variant, both development logs)")
def c10_detector_bits():
    out, ok, dq = {}, True, 0.0
    n0 = torch.get_num_threads()
    try:
        for name in _dev():
            ctx = T.context(name)
            Kq = ctx.kc.keys("tf", ctx.rows[:200])
            per = {}
            for nt in (1, 4):
                torch.set_num_threads(nt)
                WSt._LOADED.clear()
                st = WSt.load(ctx.backbone)                               # a fresh statistics object: its whitening factors are recomputed
                gates = [(lab, WS.feature_gate(ctx.log, feat, stats=st, kappa=kappa, null_keys=ctx.null_keys)) for lab, feat, kappa in WS.variants()]
                gates.append(("rescope_nullW", WS.null_w(feature_gate(ctx.log, "ncoord"), stats=st)[0]))
                per[nt] = {lab: (S.det_w.clone(), S.det_s.clone(), S.scores(Kq)) for lab, S in gates}
            e = {}
            for lab in per[1]:
                e[lab] = bool(torch.equal(per[1][lab][0], per[4][lab][0]) and torch.equal(per[1][lab][1], per[4][lab][1]))
                dq = max(dq, float((per[1][lab][2] - per[4][lab][2]).abs().max()))
            ok &= all(e.values())
            out[name] = e
    finally:
        torch.set_num_threads(n0)
        WSt._LOADED.clear()
    return {"pass": ok, "per_log": out, "max_abs_score_difference_1_vs_4_threads": dq,
            "note": f"detectors (det_w, det_s) of 8 variants + rescope_nullW bitwise equal under 1 / 4 threads: {ok}; the SCORES of 200 keys (the library's batched product, "
                    f"not pinned) differ by at most {dq:.1e}"}


@test("scores", "cost (information): build time and memory of every gate, scoring time per 1000 keys; T = 100 (development log) and T = 2000 (random keys, no real log)")
def c11_cost():
    ctx = T.context(C.DEV_LOGS["qwen25vl"])
    st = ctx.stats
    g = torch.Generator().manual_seed(7)
    Kbig = 12.0 * torch.nn.functional.normalize(torch.randn(2000, st.d, generator=g, dtype=torch.float64), dim=1)      # random solver keys: cost only
    Qk = ctx.kc.keys("tf", ctx.rows[:1000])
    res = {}
    for nm, log, nk in (("T=100", ctx.log, ctx.null_keys), ("T=2000 (random keys)", Kbig, ctx.null_keys)):
        e = {}
        for lab, feat, kappa in WS.variants():
            if kappa is not None and kappa != 0.1:
                continue
            t = time.time()
            S = WS.feature_gate(log, feat, stats=st, kappa=kappa, null_keys=nk, values="stored")
            tb = time.time() - t
            t = time.time()
            S.scores(Qk)
            ts = time.time() - t
            mem = sum(getattr(S, f).numel() * 8 for f in ("det_w", "q_F") if getattr(S, f, None) is not None) / 2 ** 20
            e[lab] = {"build_s": tb, "score_1000_keys_s": ts, "gate_MiB": mem}
        t = time.time()
        Sc = feature_gate(log if not torch.is_tensor(log) else WS._as_log(log), "cos", values="stored")
        Sc.scores(Qk)
        e["cosine (library, for comparison)"] = {"build_s": 0.0, "score_1000_keys_s": time.time() - t, "gate_MiB": Sc.det_w.numel() * 8 / 2 ** 20}
        res[nm] = e
    t = time.time()
    WSt._LOADED.clear()
    WSt.load("qwen25vl")
    t_load = time.time() - t
    # the rescope_nullW helper: T = 100 with the real log; T = 2000 with random factors and 2 T random guards (cost only)
    from rescope import EditLog
    nw = {}
    Sg, _ = add_guards(feature_gate(ctx.log, "ncoord"), *ctx.own)
    for flag in (True, False):
        t = time.time()
        WS.null_w(Sg, stats=st, score_guards=flag)
        nw[f"T=100, score_guards={flag}"] = time.time() - t
    Wbig = torch.nn.functional.normalize(torch.randn(2000, st.d, generator=g, dtype=torch.float64), dim=1) + 0.2 * Kbig / 12.0
    Lbig = EditLog(torch.zeros(2000, 1, dtype=torch.float64), Wbig, Kbig, torch.zeros(2000, dtype=torch.long), name="random-T2000")
    Sb = feature_gate(Lbig, "ncoord", values="stored")
    Sb, _ = add_guards(Sb, Kbig * 1.001, list(range(2000)))            # 4000 guards, as stored keys + B = 1 keys
    for flag in (False, True):
        t = time.time()
        WS.null_w(Sb, stats=st, score_guards=flag)
        nw[f"T=2000, score_guards={flag}"] = time.time() - t
    a, b = res["T=100"], res["T=2000 (random keys)"]
    return {"pass": True, "cost": res, "null_w_seconds": nw, "load_stats_s": t_load, "threads": torch.get_num_threads(),
            "note": f"load statistics {t_load:.1f} s; build / score 1000 keys (s), T = 100: " + ", ".join(f"{k} {v['build_s']:.2f} / {v['score_1000_keys_s']:.2f}" for k, v in a.items())
                    + "; T = 2000: " + ", ".join(f"{k} {v['build_s']:.1f} / {v['score_1000_keys_s']:.1f} ({v['gate_MiB']:.0f} MiB)" for k, v in b.items())
                    + f"; null_w (600 BEN-W keys) with / without guard scoring: T = 100 {nw['T=100, score_guards=True']:.1f} / {nw['T=100, score_guards=False']:.1f} s, "
                      f"T = 2000 (4000 guards) {nw['T=2000, score_guards=True']:.0f} / {nw['T=2000, score_guards=False']:.0f} s"}
