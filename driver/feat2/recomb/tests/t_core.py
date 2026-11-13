"""CPU tests of recomb.core on synthetic logs with a fake recombination store (no real log, no model).

    python tests/t_core.py            -> tests/out/t_core.json

a1  WITHOUT recombinations the composed rule IS the library's two-block calibration: identical thresholds, baselines, eta, record
    (x = None, x = -inf; several seeds, block ratios, alphas; with guards and capped edits; allow_silence; rank 'r'; scales u; explicit blocks)
a2  WITH recombinations: thresholds = an independent plain-loop reference of the rule b = max(block 1, x); = two_block_rule_x; det_b = b
a3  the strict conformal event on block 2: exactly n_allowed block-2 queries have a candidate, the query that defines eta has none
a4  the bound: false firing of fresh exchangeable queries over many draws <= alpha, for x from independent data (small, large, mixed)
a5  the record speaks about the real queries (n, n1, block1, split, seed) and counts what x changed; inputs are validated
a6  own_scores: default feature = direct formula; gate / feature name / callable / matrix agree bit for bit; several detectors per edit
    (det_edit); kinds; only the OWNER's rows count (an X-X row does not count for the edit whose question it borrows)
a7  rows of a log: usable by position (min_T), every kind, prefix logs; a log of another stream or order is refused
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tcommon as tc  # noqa: E402
from tcommon import check, np, torch  # noqa: E402

import rescope  # noqa: E402
import recomb  # noqa: E402

ROOT = tempfile.mkdtemp(dir=tc.WORK, prefix="t_core_")
NEG = float("-inf")


def setup(T=40, d=96, seed=0, n=200, T_stream=None):
    log, uids = tc.synthetic(T=T, d=d, seed=seed)
    root = os.path.join(ROOT, f"s{seed}_{T}_{T_stream}")
    tc.fake_store(root, log, uids, seed=seed, T_stream=T_stream)
    S = tc.gate_with_guards(log, seed=seed)
    Q = S.query_max([k[None] for k in tc.benign(log, n, seed=seed + 1)])
    return log, uids, root, S, Q


def same_log(A, B):
    return (torch.equal(A.det_tau, B.det_tau) and A.eta == B.eta and ((A.det_b is None and B.det_b is None) or torch.equal(A.det_b, B.det_b))
            and ((A.det_u is None and B.det_u is None) or torch.equal(A.det_u, B.det_u)))


def a1_without_recombinations():
    n_cmp, n_capped = 0, 0
    for seed in (0, 1, 2):
        log, uids, root, S, Q = setup(seed=seed)
        D = S.D
        for alpha in (0.05, 0.1, "1/3", 0.005):
            for split in ((1, 1), (1, 3), (3, 1)):
                for kw in ({}, {"allow_silence": True}, {"rank": "r"}, {"u": np.linspace(0.5, 2.0, D)}):
                    A, ra = rescope.calibrate_two_block(S, coords=Q, alpha=alpha, split=split, seed=seed + 5, **kw)
                    for x in (None, torch.full((D,), NEG, dtype=torch.float64), np.full(D, NEG)):
                        B, rb = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=alpha, split=split, seed=seed + 5, **kw)
                        check(same_log(A, B), f"thresholds differ without recombinations (seed {seed}, alpha {alpha}, split {split}, {list(kw)})")
                        for k in ("n", "n1", "n2", "rank", "eta", "split", "seed", "block1", "block2", "n_edits_capped", "capped_edits", "bound_false_firing",
                                  "n_allowed_above", "n_own_keys_silenced_by_literal_rule", "threshold_min_median_max", "baseline_min_median_max", "guarantee"):
                            check(ra[k] == rb[k], f"record field {k}: {ra[k]} vs {rb[k]}")
                        check(rb["recomb"]["n_baselines_raised_by_x"] == 0 and rb["recomb"]["n_literal_thresholds_equal"] == D)
                        n_cmp += 1
                    n_capped += ra["n_edits_capped"]
        # explicit blocks, and queries given as keys instead of coords
        i1, i2 = rescope.split_blocks(Q.shape[0], (1, 1), 11)
        A, _ = rescope.calibrate_two_block(S, coords=Q, alpha=0.05, blocks=(i1, i2))
        B, _ = recomb.calibrate_two_block_x(S, coords=Q, x=None, alpha=0.05, blocks=(i1, i2))
        check(same_log(A, B), "explicit blocks")
        keys = [k[None] for k in tc.benign(log, 60, seed=9)]
        A, _ = rescope.calibrate_two_block(S, keys, 0.1, seed=2)
        B, _ = recomb.calibrate_two_block_x(S, keys, 0.1, x=None, seed=2)
        check(same_log(A, B), "queries as keys")
    check(n_capped > 0, "no configuration with capped edits: the guard path was not exercised")
    return {"n_comparisons_bitwise_identical": n_cmp, "capped_edits_seen_in_total": n_capped}


def a2_with_recombinations():
    out = {"n": 0, "n_raised": 0}
    for seed in (0, 1, 2, 3):
        log, uids, root, S, Q = setup(seed=seed)
        D, Qn = S.D, Q.numpy()
        g = np.random.default_rng(seed)
        xs = [recomb.own_baseline(log, S, root=root)[0].numpy(),                         # large: raises every baseline
              Qn.max(0) - g.uniform(0, 0.3, D),                                          # below the calibration maxima: raises some block-1 baselines or none
              np.where(g.random(D) < 0.5, NEG, Qn.max(0) + g.uniform(-0.1, 0.4, D))]     # half of the edits without rows
        for x in xs:
            for alpha, frac in ((0.05, (1, 20)), (0.1, (1, 10)), ("1/3", (1, 3))):
                for u in (None, np.linspace(0.5, 2.0, D)):
                    i1, i2 = rescope.split_blocks(Qn.shape[0], (1, 1), seed)
                    b, eta, tau = tc.naive_rule(Qn, x, frac[0], frac[1], i1, i2, u)
                    ref = recomb.two_block_rule_x(Qn, x, alpha, seed=seed, u=u)
                    check(np.array_equal(ref["tau"], tau) and ref["eta"] == eta and np.array_equal(ref["b"], b), "two_block_rule_x != naive reference")
                    check(np.array_equal(b, np.maximum(Qn[i1].max(0), x)), "b != max(block-1 maximum, x)")
                    B, rb = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=alpha, seed=seed, u=u, allow_silence=True)
                    check(np.array_equal(B.det_tau.numpy(), tau), "library path: uncapped thresholds != naive reference")
                    check(np.array_equal(B.det_b.numpy(), b) and rb["eta"] == eta, "library path: baselines / eta")
                    C, rc = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=alpha, seed=seed, u=u)          # guards on: only capped edits are lower
                    capped = np.zeros(D, dtype=bool)
                    capped[rc["capped_edits"]] = True
                    ct = C.det_tau.numpy()
                    check(np.array_equal(ct[~capped], tau[~capped]) and bool((ct[capped] < tau[capped]).all()) and bool(C.guard_ok().all()), "guards")
                    out["n"] += 1
                    out["n_raised"] += rb["recomb"]["n_baselines_raised_by_x"]
    return out


def a3_strict_event_on_block2():
    out = []
    for seed in (0, 1, 2):
        log, uids, root, S, Q = setup(seed=seed)
        x = recomb.own_baseline(log, S, root=root)[0]
        for xx in (None, x, x - 0.5):
            B, rb = recomb.calibrate_two_block_x(S, coords=Q, x=xx, alpha=0.05, seed=seed, allow_silence=True)
            i2 = np.array(rb["block2"])
            cand = (Q[i2] >= B.det_tau[None, :]).any(1).numpy()
            sc = recomb.two_block_rule_x(Q, xx, 0.05, seed=seed)["scores2"]
            check(len(np.unique(sc)) == len(sc), "ties in the synthetic scores")
            # eta is the 96th smallest of the 100 block-2 scores: 4 block-2 queries lie strictly above it (n2 - rank); together with a
            # fresh query that makes n_allowed = n2 + 1 - rank = 5 of n2 + 1 = 101 exchangeable scores: the bound 5 / 101
            check(rb["rank"] == 96 and rb["n_allowed_above"] == 5 and int(cand.sum()) == rb["n2"] - rb["rank"] == 4,
                  f"{int(cand.sum())} block-2 queries fire; rank {rb['rank']}, allowed {rb['n_allowed_above']}")
            check(np.array_equal(cand, sc > rb["eta"]), "a block-2 query fires iff its score is above eta")
            check(not cand[int(np.argmin(np.abs(sc - rb["eta"])))], "the query that defines eta fires")
            out.append(int(cand.sum()))
    return {"block2_queries_with_a_candidate": out, "n2_minus_rank": 4, "n_allowed_of_n2_plus_1": 5}


def a4_bound_holds():
    """Monte Carlo on one synthetic log: every draw takes n = 80 calibration queries and 400 fresh ones from the same distribution."""
    log, uids = tc.synthetic(T=30, d=64, seed=7)
    root = os.path.join(ROOT, "a4")
    tc.fake_store(root, log, uids, seed=7)
    S = rescope.feature_gate(log, "ncoord")
    own = recomb.own_baseline(log, S, root=root)[0].numpy()
    pool = S.query_max([k[None] for k in tc.benign(log, 6000, seed=123)]).numpy()
    g = np.random.default_rng(0)
    xs = {"none": None, "own_rows(large)": own, "small": own - 1.0, "mixed": np.where(np.arange(S.D) % 3 == 0, own, NEG)}
    res = {}
    alpha, n, n_draws = 0.1, 80, 400
    for name, x in xs.items():
        rates = []
        for _ in range(n_draws):
            idx = g.permutation(pool.shape[0])
            cal, fresh = pool[idx[:n]], pool[idx[n:n + 400]]
            rule = recomb.two_block_rule_x(cal, x, alpha, seed=int(g.integers(1 << 30)))
            rates.append(float((fresh >= rule["tau"][None, :]).any(1).mean()))
        m, se = float(np.mean(rates)), float(np.std(rates) / np.sqrt(n_draws))
        bound = rule["bound"]
        check(m <= bound + 4 * se, f"{name}: mean false firing {m:.4f} above the bound {bound:.4f} (se {se:.4f})")
        res[name] = {"mean_false_firing": round(m, 4), "se": round(se, 4), "bound": round(bound, 4)}
    return res


def a5_record_and_validation():
    log, uids, root, S, Q = setup(seed=4)
    x, xr = recomb.own_baseline(log, S, root=root)
    B, rb = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=0.05, seed=4)
    i1, i2 = rescope.split_blocks(Q.shape[0], (1, 1), 4)
    check(rb["n"] == 200 and rb["n1"] == 100 and rb["n2"] == 100 and rb["block1"] == [int(i) for i in i1] and rb["block2"] == [int(i) for i in i2])
    check(rb["split"] == [1, 1] and rb["seed"] == 4 and rb["op"] == "calibrate_two_block_x" and B.meta["calibration"]["n1"] == 100)
    rr = rb["recomb"]
    b1 = Q.numpy()[i1].max(0)
    check(rr["n_baselines_raised_by_x"] == int((x.numpy() > b1).sum()) and rr["n_detectors_with_x"] == S.D)
    check(rr["n_literal_thresholds_higher_with_x"] + rr["n_literal_thresholds_lower_with_x"] + rr["n_literal_thresholds_equal"] == S.D)
    check(rr["eta_with_x"] == rb["eta"] and rr["eta_without_x"] == rescope.calibrate_two_block(S, coords=Q, alpha=0.05, seed=4)[1]["eta"])
    check(xr["n_rows"] == 12 * log.T and xr["n_rows_by_kind"] == {"X-I": 4 * log.T, "X-Q": 4 * log.T, "X-X": 4 * log.T} and xr["n_detectors_without_rows"] == 0)
    bad = 0
    for wrong in (torch.zeros(S.D - 1), torch.full((S.D,), float("nan")), torch.full((S.D,), float("inf"))):
        try:
            recomb.calibrate_two_block_x(S, coords=Q, x=wrong, alpha=0.05)
        except ValueError:
            bad += 1
    try:
        recomb.calibrate_two_block_x(S, x=x, alpha=0.05)
    except ValueError:
        bad += 1
    try:
        recomb.own_scores(log, S, kinds=("X-Y",), root=root)
    except ValueError:
        bad += 1
    try:
        recomb.own_scores(log, torch.zeros(5, log.T), root=root)
    except ValueError:
        bad += 1
    check(bad == 6, f"{bad} of 6 invalid inputs were refused")
    # the input is not modified, the function is deterministic
    B2, _ = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=0.05, seed=4)
    check(same_log(B, B2) and float(S.det_tau.max()) == NEG)
    return {"record_fields_ok": True, "invalid_inputs_refused": bad}


def a6_own_scores():
    log, uids, root, S, Q = setup(seed=5)
    T = log.T
    R = recomb.log_rows(log, root=root)
    K = R.keys().double()
    r = (K @ log.W.T) / log.own()[None, :]
    own = r[torch.arange(len(R)), torch.from_numpy(R.edit)].numpy()
    x_dir = np.full(T, NEG)
    np.maximum.at(x_dir, R.edit, own)
    o = recomb.own_scores(log, None, root=root)
    check(float(np.abs(o.x.numpy() - x_dir).max()) < 1e-12 and float(np.abs(o.value - own).max()) < 1e-12, "default feature != direct formula")
    variants = {"feature name": "ncoord", "gate": S, "plain gate": rescope.feature_gate(log, "ncoord"), "callable": lambda k: S.scores(k),
                "matrix": recomb.own_scores(log, S, root=root, full=True).matrix}
    for name, sc in variants.items():
        v = recomb.own_scores(log, sc, root=root)
        check(torch.equal(v.x, o.x) and np.array_equal(v.value, o.value), f"{name}: not bit for bit the default")
    check(recomb.own_scores((log.meta["backbone"], uids), S, root=root).x.equal(o.x), "(backbone, uids) identity")
    # kinds: the baseline of all kinds is the maximum over the kinds; per_edit(kind) are the per-kind maxima
    per = {k: recomb.own_scores(log, S, kinds=(k,), root=root).x.numpy() for k in recomb.KINDS}
    check(np.array_equal(np.maximum.reduce(list(per.values())), o.x.numpy()), "x(all kinds) != max over kinds")
    for k in recomb.KINDS:
        check(np.array_equal(o.per_edit(k), per[k]), f"per_edit({k})")
    check(np.array_equal(o.per_edit(), o.x.numpy()))
    # other features of the library
    for feat in ("coord", "cos", "negdist"):
        G = rescope.feature_gate(log, feat)
        v = recomb.own_scores(log, feat, root=root)
        full = G.scores(R.keys())
        ref = torch.full((T,), NEG, dtype=torch.float64).scatter_reduce(0, torch.from_numpy(R.edit), full[torch.arange(len(R)), torch.from_numpy(R.edit)], reduce="amax")
        check(float((v.x - ref).abs().max()) < 1e-9, f"feature {feat}")
    # only the OWNER's rows count.  (i) a huge score of detector 9 on the rows owned by edit 3 moves no baseline; (ii) raising the own
    # scores of edit 3's rows by 1 moves x_3 by 1 and nothing else; (iii) an X-X row does not count for the edit whose question it borrows
    M0 = recomb.own_scores(log, S, root=root, full=True).matrix.clone()
    rows3 = np.nonzero(R.edit == 3)[0]
    M1 = M0.clone()
    M1[rows3, 9] = float(M0.max()) + 5.0
    check(torch.equal(recomb.own_scores(log, M1, root=root).x, o.x), "a row owned by edit 3 raised the baseline of edit 9")
    M3 = M0.clone()
    M3[rows3, 3] += 1.0
    v = recomb.own_scores(log, M3, root=root)
    moved = np.nonzero(v.x.numpy() != o.x.numpy())[0].tolist()
    check(moved == [3] and float(v.x[3]) == float(o.x[3] + 1.0), f"baselines that moved: {moved}")
    xx = np.nonzero(R.partner >= 0)[0]
    M4 = M0.clone()
    M4[torch.from_numpy(xx), torch.from_numpy(R.partner[xx])] = float(M0.max()) + 5.0     # the partner's detector on every X-X row
    check(torch.equal(recomb.own_scores(log, M4, root=root).x, o.x), "an X-X row raised the baseline of the edit whose question it borrows")
    # several detectors per edit: a second detector for edits 0..4 (twice the first): its baseline is twice the edit's own maximum
    de = np.concatenate([np.arange(T), np.arange(5)])
    M2 = torch.cat([M0, 2.0 * M0[:, :5]], 1)
    v2 = recomb.own_scores(log, M2, det_edit=de, root=root)
    check(torch.equal(v2.x[:T], o.x) and torch.equal(v2.x[T:], torch.where(o.x[:5] > 0, 2.0 * o.x[:5], v2.x[T:])) and np.array_equal(v2.value, o.value), "det_edit")
    try:
        recomb.own_scores(log, M2, root=root)
        check(False, "a matrix with D != T was accepted without det_edit")
    except ValueError:
        pass
    # determinism: two evaluations, other chunk size (bits may differ between chunk sizes, the values agree to round-off)
    a, b = recomb.own_scores(log, S, root=root).x, recomb.own_scores(log, S, root=root).x
    c = recomb.own_scores(log, S, root=root, chunk=7).x
    check(torch.equal(a, b) and float((a - c).abs().max()) < 1e-12)
    # decide_rows against a naive loop: thresholds from a calibration with the baseline
    B, rb = recomb.calibrate_two_block_x(S, coords=Q, x=o.x, alpha=0.05, seed=5)
    dd = recomb.decide_rows(B, log, root=root)
    th, bb = B.det_tau.numpy(), B.det_b.numpy()
    full = recomb.own_scores(log, S, root=root, full=True).matrix.numpy()
    for i in range(0, len(R), 7):
        cand = full[i] >= th
        e, pj = int(R.edit[i]), int(R.partner[i])
        top = int(np.argmax(np.where(cand, full[i] - bb, -np.inf))) if cand.any() else -1
        check(bool(dd["fired"][i]) == bool(cand.any()) and int(dd["top_edit"][i]) == top and bool(dd["own_cand"][i]) == bool(cand[e])
              and bool(dd["own_top"][i]) == (top == e) and int(dd["n_cand"][i]) == int(cand.sum())
              and bool(dd["b_cand"][i]) == (pj >= 0 and bool(cand[pj])) and bool(dd["b_top"][i]) == (pj >= 0 and top == pj), f"decide_rows, row {i}")
    capped = np.zeros(T, dtype=bool)
    capped[rb["capped_edits"]] = True
    if rb["eta"] >= 0:                                                  # r_i <= x_i <= b_i on its own rows: an uncapped edit cannot fire there
        check(not dd["own_cand"][~capped[R.edit]].any(), "an uncapped edit is a candidate on its own recombination row although eta >= 0")
    return {"n_rows": len(R), "max_abs_dev_from_direct_formula": float(np.abs(o.x.numpy() - x_dir).max()), "variants_bitwise": list(variants),
            "x_min_median_max": [float(q) for q in np.quantile(o.x.numpy(), [0, 0.5, 1])], "decide_rows_eta": rb["eta"], "decide_rows_fired": int(dd["fired"].sum())}


def a7_rows_of_a_log():
    log, uids = tc.synthetic(T=30, d=48, seed=6)
    root = os.path.join(ROOT, "a7")
    rows = tc.fake_store(root, log, uids, seed=6, T_stream=230)            # a stream of 230 edits; the log holds its first 30
    st = recomb.store("qwen25vl", root)
    check(st.T_stream == 230 and st.n == 230 * 12)
    R = recomb.log_rows(log, root=root)
    want = [i for i, r in enumerate(rows) if r["edit_pos"] < 30 and (r["edit_b_pos"] is None or r["edit_b_pos"] < 30)]
    check(R.idx.tolist() == want and int(R.edit.max()) == 29 and int(R.partner.max()) < 30, "usable rows")
    cnt = R.counts()
    check(cnt["X-I"] == cnt["X-Q"] == 120 and 60 <= cnt["X-X"] < 120, f"counts {cnt}")           # batch partners always, group partners only if < 30
    per = R.per_edit()
    check(bool((per["X-I"] == 4).all() and (per["X-Q"] == 4).all() and (per["X-X"] >= 2).all()))
    K, R2 = recomb.log_keys(log, root=root)
    check(tuple(K.shape) == (len(R), 48) and K.dtype == torch.float32 and torch.equal(K, st.K[torch.from_numpy(R.idx)]) and R2.rids == R.rids)
    check(recomb.log_rows(log, kinds="X-Q", root=root).counts() == {"X-I": 0, "X-Q": 120, "X-X": 0})
    # prefixes of the stream: every length that is a multiple of 100 has all 12 rows per edit
    for T in (10, 20, 50, 100, 200, 230):
        Rt = recomb.log_rows(("qwen25vl", st.stream[:T]), root=root)
        c = Rt.counts()
        check(c["X-I"] == c["X-Q"] == 4 * T and (c["X-X"] == 4 * T if T % 100 == 0 else 2 * T <= c["X-X"] <= 4 * T), f"T = {T}: {c}")
        check(bool((np.maximum(Rt.edit, Rt.partner) < T).all()))
    refused = 0
    for bad in (list(reversed(uids)), uids[:5] + ["other"] + uids[6:], st.stream + ["one more"], []):
        try:
            recomb.log_rows(("qwen25vl", bad), root=root)
        except LookupError:
            refused += 1
    check(refused == 4, f"{refused} of 4 foreign logs refused")
    try:
        recomb.log_rows(("llava", uids), root=root)
        check(False, "a backbone without a store was served")
    except FileNotFoundError:
        pass
    # a detector whose edit has no usable row keeps -inf and the library's own baseline
    S = rescope.feature_gate(log, "ncoord")
    sub = R.subset(R.edit >= 10)
    o = recomb.own_scores(log, S, rows=sub)
    check(bool(torch.isneginf(o.x[:10]).all()) and bool(torch.isfinite(o.x[10:]).all()) and o.record()["n_detectors_without_rows"] == 10)
    Q = S.query_max([k[None] for k in tc.benign(log, 100, seed=2)])
    A, _ = rescope.calibrate_two_block(S, coords=Q, alpha=0.1, seed=0, allow_silence=True)
    B, _ = recomb.calibrate_two_block_x(S, coords=Q, x=o.x, alpha=0.1, seed=0, allow_silence=True)
    check(torch.equal(A.det_b[:10], B.det_b[:10]), "edits without rows keep their block-1 baseline")
    return {"usable_rows_T30": cnt, "foreign_logs_refused": refused}


if __name__ == "__main__":
    R = tc.Results("t_core")
    try:
        for name, fn in (("a1_without_recombinations_identical_to_library", a1_without_recombinations), ("a2_with_recombinations_reference", a2_with_recombinations),
                         ("a3_strict_event_on_block2", a3_strict_event_on_block2), ("a4_bound_holds", a4_bound_holds),
                         ("a5_record_and_validation", a5_record_and_validation), ("a6_own_scores", a6_own_scores), ("a7_rows_of_a_log", a7_rows_of_a_log)):
            R.run(name, fn)
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    sys.exit(0 if R.save() else 1)
