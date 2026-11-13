"""E5 (c): calibration validity at gate level (CPU, key caches only), one log per process.

    python e5c_validity.py <log> [--gate-only]        development logs: everything; --gate-only (MEMIT-style logs): pool (i) only

RULES (all on the matrices of normalised coordinates r_j(k) of the log's detectors; fx.gate.features of the driver's base gate;
the arithmetic of the library: split_blocks, block_scores, conformal_rank_exact; the comparisons below are the library's stored
thresholds bit for bit -- r >= tau_j <=> fl(r - b_j) > eta -- and are checked against calibrate_two_block / calibrate_shared and
the driver's running-maxima rule on the first draws of every log):
  two_block   n queries split at random into two halves (seed = draw); b_j = block-1 maximum; eta = the k-th smallest block-2 score
              max_j (r_j - b_j), k = ceil((1 - alpha)(n2 + 1)); a query fires iff max_j (r_j - b_j) > eta.      bound floor(alpha (n2 + 1)) / (n2 + 1)
  pac         the same with k = fx.methods.pac_rank(n2, alpha, 0.1)  (rescope_pac; calibration-conditional, delta = 0.1)
  shared      eta = the k-th smallest of the n scores max_j r_j, k = ceil((1 - alpha)(n + 1)); fires iff max_j r_j > eta.   bound floor(alpha (n + 1)) / (n + 1)
  runmax      per-edit running maxima = the least repair of the n queries from thresholds -inf: fires iff r_j > max over the n queries
              of r_j for some j (no level).                                                                    bound min(1, T / (n + 1))
  literal = the rule as stated; guards = thresholds capped at (1 - 1e-9) x the smallest coordinate of the edit's guard keys (stored
  solver key and B = 1 key of the own Rel prompt; library own_caps): the method as deployed; n_capped = edits whose cap is below
  the literal threshold (the bound is void for them).
DRAWS  500 seeded draws of n calibration rows from the calibration pool (numpy default_rng([20261111, cell, n, draw]) permutation).
       Within-pool cells: test = the remaining rows of the pool.  Cross-pool cells: test = every row of the test pool.
POOLS  probes_all     valid REPAIR-plus-held-out probes of the log, teacher-forced keys (task pool (i)), within pool
       BEN-S          the calibration part of the label-free benign pool, prompt-only keys, within pool (task pool (ii): the whole pool
                      cannot be drawn from, its part BEN-T is test data only and BEN-W is the statistics part; closest admissible pool)
       repair -> heldout_disjoint at teacher-forced keys (registered) and at prompt-only keys; repair -> BEN-T, MME-T;
       BEN-S -> heldout_disjoint, BEN-T, MME-T; MME-C -> heldout_disjoint, BEN-T, MME-T: prompt-only key of every row on both sides (R25 (5)).
       BEN-T and MME-T are never on the calibration side.
SUBSETS of test rows: all; far / near (probes: the question string is the question of an edit of the log; BEN: the pool's own flags
       relate the row to an edit or probe of the family, fx.pools.related); q_edit (BEN: question string of an edit of the log); image / text rows.
COVERAGE  share of text / image rephrase sequences on which the own edit is released (top-1 by the rule's ranking), guards kept, at
       the key type of the calibration.
"""
from e5common import *
from rescope import calibrate as CAL
from rescope.calibrate import own_caps

ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.20)
NS = (20, 50, 100, 200, 500)
NDRAW = 500
PAC_DELTA = 0.1
RULES = ("two_block", "pac", "shared", "runmax")
INF = float("inf")


class Feats:
    def __init__(self, L):
        self.L = L
        self.S, _ = fx.methods.base_gate(L)
        self.cap = own_caps(self.S).numpy()
        self.T = L.T
        self._c = {}

    def R(self, rows, which, kc=None):
        return fx.gate.features(self.L, self.S, rows, which, kc=kc).numpy()

    def edits(self, which):
        if which not in self._c:
            out = {}
            for m in ("tgen", "igen"):
                rows = self.L.edit_rows[m]
                out[m] = (self.R(rows, which), np.arange(self.T))
            self._c[which] = out
        return self._c[which]


def masks_probe(L, rows):
    ix = L.index
    near = np.array([L.near_edit(j) is not None for j in rows])
    pt = np.array([ix[j].get("ptype") for j in rows])
    return {"all": np.ones(len(rows), bool), "far": ~near, "near": near, "image": pt == "image", "text": pt == "text"}


def masks_pool(L, P):
    ix = P.index
    rel = fx.pools.related(L, P)
    pt = np.array([ix[j].get("ptype") or "image" for j in P.rows])
    m = {"all": np.ones(len(P.rows), bool), "far": ~rel, "near": rel, "image": pt == "image", "text": pt == "text"}
    if P.kc is None:
        m["q_edit"] = np.array([fx.logs.norm_q(ix[j]["prompt"]) in L._q_edit for j in P.rows])
        m["not_q_edit"] = ~m["q_edit"]
    return m


def ranks(n, alpha_list):
    """per rule: list of (label, k, n_cal_for_bound) for a sample of n queries."""
    n2 = n - n // 2
    out = {"two_block": [], "pac": [], "shared": []}
    for a in alpha_list:
        k = CAL.conformal_rank_exact(n2, a)
        out["two_block"].append((a, k, n2, (0 if k > n2 else n2 + 1 - k) / (n2 + 1)))
        kp = fx.methods.pac_rank(n2, a, PAC_DELTA)
        out["pac"].append((a, kp, n2, None))
        ks = CAL.conformal_rank_exact(n, a)
        out["shared"].append((a, ks, n, (0 if ks > n else n + 1 - ks) / (n + 1)))
    return out


def kth(sorted_scores, k):
    return INF if k > len(sorted_scores) else float(sorted_scores[k - 1])


class Acc:
    """Per (rule, alpha, test, subset): realised rates per draw, literal and with guards; coverage; capped edits."""

    def __init__(self):
        self.d = {}

    def add(self, key, v):
        self.d.setdefault(key, []).append(v)


def one_draw(Q, seed, cap, tests, E, rk, acc, T):
    """Q (n, T) calibration coordinates; tests: {name: (Rt, Gt float32 (rows, T), rowmax_r, masks)}; E: {metric: (Re, own, Ge)}."""
    n = Q.shape[0]
    idx1, idx2 = CAL.split_blocks(n, (1, 1), seed)
    b = Q[idx1].max(0)
    sc2 = np.sort(CAL.block_scores(Q[idx2], b, None))
    ball = Q.max(0)
    scs = np.sort(Q.max(1))
    tm = {name: (Rt - b[None, :]).max(1) for name, (Rt, _, _, _) in tests.items()}                 # two-block margins of the test rows
    rm = {name: (Rt > ball[None, :]).any(1) for name, (Rt, _, _, _) in tests.items()}              # running maxima, literal
    em = {m: (v[0] - b[None, :]) for m, v in E.items()}
    emax = {m: v.max(1) for m, v in em.items()}
    eown = {m: v[np.arange(v.shape[0]), E[m][1]] for m, v in em.items()}

    def record(rule, a, lit, capped, margin_test, eta, Me, Memax, Meown):
        ncap = int(capped.sum())
        acc.add((rule, a, "n_capped"), ncap)
        cf = capped.astype(np.float32)
        for name, (Rt, Gt, _, masks) in tests.items():
            g = lit[name] | ((Gt @ cf) > 0) if ncap else lit[name]
            for sub, mk in masks.items():
                den = int(mk.sum())
                acc.add((rule, a, name, sub, "lit"), float(lit[name][mk].mean()) if den else np.nan)
                acc.add((rule, a, name, sub, "grd"), float(g[mk].mean()) if den else np.nan)
        for m, (Re, own, Ge, _) in E.items():
            own_lit = (Meown[m] > eta) & (Meown[m] >= Memax[m])
            cov = own_lit.copy()
            if ncap:
                cols = np.nonzero(capped)[0]
                gc = Ge[:, cols] > 0                                                               # guard candidates among the capped edits
                gm = np.where(gc, Me[m][:, cols], -INF).max(1)                                     # their best ranking key
                own_capped = capped[own]
                own_g = own_capped & (Re[np.arange(len(own)), own] >= cap[own])                    # own edit is a guard candidate
                no_lit = ~(Memax[m] > eta)
                cov = cov | (own_g & no_lit & (Meown[m] >= gm) & ~(Meown[m] > eta))
            acc.add((rule, a, "cov", m, "lit"), float(own_lit.mean()))
            acc.add((rule, a, "cov", m, "grd"), float(cov.mean()))

    for rule in ("two_block", "pac"):
        for a, k, n2, _ in rk[rule]:
            eta = kth(sc2, k)
            lit = {name: tm[name] > eta for name in tests}
            capped = (cap - b) <= eta
            record(rule, a, lit, capped, tm, eta, em, emax, eown)
    for a, k, _, _ in rk["shared"]:
        eta = kth(scs, k)
        lit = {name: tests[name][2] > eta for name in tests}
        capped = cap <= eta
        Me = {m: E[m][0] for m in E}
        record("shared", a, lit, capped, None, eta, Me, {m: E[m][3] for m in E},
               {m: Me[m][np.arange(Me[m].shape[0]), E[m][1]] for m in E})
    # running maxima: thresholds = next float above the n-query maximum; top-1 by r_j - b_j (driver's rule)
    capped = cap <= ball
    Me = {m: E[m][0] - ball[None, :] for m in E}
    record("runmax", None, rm, capped, None, 0.0, Me, {m: v.max(1) for m, v in Me.items()}, {m: v[np.arange(v.shape[0]), E[m][1]] for m, v in Me.items()})


def summarise(acc, rk, n, T, ndraw, tests_n, keep_raw):
    cells = []
    bounds = {("two_block", a): bd for a, _, _, bd in rk["two_block"]}
    bounds.update({("shared", a): bd for a, _, _, bd in rk["shared"]})
    rks = {(r, a): k for r in ("two_block", "pac", "shared") for a, k, _, _ in rk[r]}
    names = sorted({k[2] for k in acc.d if len(k) == 5 and k[2] != "cov"})
    for rule in RULES:
        for a in ([None] if rule == "runmax" else [x[0] for x in rk["two_block"]]):
            c = {"rule": rule, "alpha": a, "n": n, "n_draws": ndraw, "rank": rks.get((rule, a)),
                 "bound": min(1.0, T / (n + 1.0)) if rule == "runmax" else bounds.get((rule, a)),
                 "level": a, "n_capped_mean": float(np.mean(acc.d[(rule, a, "n_capped")])), "n_capped_max": int(np.max(acc.d[(rule, a, "n_capped")])),
                 "coverage": {m: {v: float(np.mean(acc.d[(rule, a, "cov", m, v)]) * 100) for v in ("lit", "grd")} for m in ("tgen", "igen")}, "tests": {}}
            if rule == "pac":
                c["bound"] = None
                c["pac_delta"] = PAC_DELTA
            for name in names:
                t = {}
                subs = sorted({k[3] for k in acc.d if len(k) == 5 and k[0] == rule and k[1] == a and k[2] == name})
                for sub in subs:
                    e = {}
                    for v in ("lit", "grd"):
                        x = np.array(acc.d[(rule, a, name, sub, v)]) * 100
                        if np.isnan(x).all():
                            continue
                        e[v] = {"mean": float(np.nanmean(x)), "p05": float(np.nanquantile(x, 0.05)), "p95": float(np.nanquantile(x, 0.95)),
                                "median": float(np.nanmedian(x)), "max": float(np.nanmax(x)), "sd": float(np.nanstd(x, ddof=1)),
                                "se_mean": float(np.nanstd(x, ddof=1) / np.sqrt(np.isfinite(x).sum()))}
                        if a is not None:
                            e[v]["share_of_draws_above_alpha"] = float(np.nanmean(x > a * 100) * 100)
                        if keep_raw and sub in ("all", "far"):
                            e[v]["per_draw"] = [round(float(z), 4) for z in x]
                    t[sub] = e
                t["n_rows"] = tests_n[name]
                c["tests"][name] = t
            cells.append(c)
    return cells


def prep_test(F, R, masks):
    return (R, (R >= F.cap[None, :]).astype(np.float32), R.max(1), masks)


def prep_edits(F, which):
    out = {}
    for m, (Re, own) in F.edits(which).items():
        out[m] = (Re, own, (Re >= F.cap[None, :]).astype(np.float32), Re.max(1))
    return out


def run_within(F, Rpool, masks, which, ns, cell_id, ndraw=NDRAW, extra_tests=None, keep_raw_n=(200,)):
    """Draw n rows from the pool, test = the rest (+ fixed extra test pools)."""
    N, T = Rpool.shape
    E = prep_edits(F, which)
    Gpool = (Rpool >= F.cap[None, :]).astype(np.float32)
    rmax = Rpool.max(1)
    out = []
    for n in ns:
        if n >= N - 20:
            out.append({"n": n, "skipped": f"pool of {N} rows"})
            continue
        rk = ranks(n, ALPHAS)
        acc = Acc()
        t0 = time.time()
        for d in range(ndraw):
            perm = np.random.default_rng([20261111, cell_id, n, d]).permutation(N)
            cal, rest = np.sort(perm[:n]), np.sort(perm[n:])
            tests = {"rest": (Rpool[rest], Gpool[rest], rmax[rest], {k: v[rest] for k, v in masks.items()})}
            if extra_tests:
                tests.update(extra_tests)
            one_draw(Rpool[cal], d, F.cap, tests, E, rk, acc, T)
        tn = {"rest": {"n": N - n, **{k: int(v.sum()) for k, v in masks.items()}, "note": "expected counts of the subsets = pool counts x (N - n) / N"}}
        if extra_tests:
            tn.update({k: {"n": int(v[0].shape[0]), **{s: int(mk.sum()) for s, mk in v[3].items()}} for k, v in extra_tests.items()})
        cells = summarise(acc, rk, n, T, ndraw, tn, n in keep_raw_n)
        out.append({"n": n, "n_pool": N, "which": which, "cells": cells, "seconds": time.time() - t0})
        print("   n", n, "s", round(time.time() - t0, 1), flush=True)
    return out


def check_library(F, Rpool, rows, which, kc, n=200, draws=(0, 1, 2), alphas=(0.05, 0.2)):
    """The matrix rules against the library's calibrated gates on a few draws: firing of every pool row and the capped edits."""
    from rescope import calibrate_two_block, calibrate_shared
    N = Rpool.shape[0]
    bad, cnt = 0, 0
    Rt = torch.from_numpy(Rpool)
    for d in draws:
        perm = np.random.default_rng([20261111, 999, n, d]).permutation(N)
        cal = np.sort(perm[:n])
        Q = Rpool[cal]
        for a in alphas:
            for rule, fn in (("two_block", lambda s: calibrate_two_block(F.S, coords=torch.from_numpy(Q), alpha=a, seed=d, allow_silence=s)),
                             ("shared", lambda s: calibrate_shared(F.S, coords=torch.from_numpy(Q), alpha=a, allow_silence=s))):
                for sil in (True, False):
                    S2, rec = fn(sil)
                    lib = (Rt >= S2.det_tau[None, :]).any(1).numpy()
                    acc = Acc()
                    tests = {"p": prep_test(F, Rpool, {"all": np.ones(N, bool)})}
                    rk = ranks(n, (a,))
                    # recompute with the matrix rule
                    idx1, idx2 = CAL.split_blocks(n, (1, 1), d)
                    if rule == "two_block":
                        b = Q[idx1].max(0); sc = np.sort(CAL.block_scores(Q[idx2], b, None)); eta = kth(sc, rk["two_block"][0][1])
                        lit = (Rpool - b[None, :]).max(1) > eta; capped = (F.cap - b) <= eta
                    else:
                        sc = np.sort(Q.max(1)); eta = kth(sc, rk["shared"][0][1]); lit = Rpool.max(1) > eta; capped = F.cap <= eta
                    mine = lit if sil else (lit | ((tests["p"][1] @ capped.astype(np.float32)) > 0))
                    cnt += 1
                    ok = bool((mine == lib).all()) and (sil or int(capped.sum()) == rec["n_edits_capped"])
                    bad += not ok
    return {"n_comparisons": cnt, "n_disagreements": bad, "rows_per_comparison": N}


def main():
    name = sys.argv[1]
    gate_only = "--gate-only" in sys.argv
    out_path = f"{OUT}/c/{name}.json"
    L = fx.logs.open(name)
    F = Feats(L)
    res = {"log": name, "T": L.T, "backbone": L.backbone, "role": L.role, "gate_only": gate_only, "n_draws": NDRAW, "alphas": list(ALPHAS), "ns": list(NS),
           "rules": __doc__, "within": {}, "cross": {}}
    t0 = time.time()
    # ---- pool (i): all valid probes, teacher-forced keys
    rows = L.pool("all")
    Rp = F.R(rows, "tf")
    res["library_check"] = {"probes_all": check_library(F, Rp, rows, "tf", None)}
    print("check", res["library_check"], flush=True)
    res["within"]["probes_all"] = {"pool": "valid REPAIR-plus-held-out probes, teacher-forced keys", "n_pool": len(rows), "exchangeable": True,
                                   "by_n": run_within(F, Rp, masks_probe(L, rows), "tf", NS, 1)}
    jdump(res, out_path)
    if not gate_only:
        # ---- pool (ii): BEN-S within pool (prompt-only keys), extra test: BEN-T
        Ps, Pt = fx.pools.get(L, "BEN-S", "decode"), fx.pools.get(L, "BEN-T", "decode")
        Mc, Mt = fx.pools.get(L, "MME-C", "decode"), fx.pools.get(L, "MME-T", "decode")
        Rs, Rt = F.R(Ps.rows, "decode"), F.R(Pt.rows, "decode")
        Rmc, Rmt = F.R(Mc.rows, "decode", kc=Mc.kc), F.R(Mt.rows, "decode", kc=Mt.kc)
        rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
        Rrep_tf, Rhd_tf = F.R(rep, "tf"), F.R(hd, "tf")
        Rrep_d, Rhd_d = F.R(rep, "decode"), F.R(hd, "decode")
        T_bent = prep_test(F, Rt, masks_pool(L, Pt))
        T_mmet = prep_test(F, Rmt, masks_pool(L, Mt))
        T_hd_d = prep_test(F, Rhd_d, masks_probe(L, hd))
        T_hd_tf = prep_test(F, Rhd_tf, masks_probe(L, hd))
        res["pools"] = {"repair": len(rep), "heldout_disjoint": len(hd), "BEN-S": len(Ps.rows), "BEN-T": len(Pt.rows), "MME-C": len(Mc.rows), "MME-T": len(Mt.rows),
                        "BEN-T_related": int(T_bent[3]["near"].sum()), "BEN-S_related": int(masks_pool(L, Ps)["near"].sum()),
                        "heldout_disjoint_near": int(T_hd_tf[3]["near"].sum()), "repair_near": int(masks_probe(L, rep)["near"].sum())}
        res["within"]["BEN-S"] = {"pool": "BEN-S (calibration part of the benign pool), prompt-only keys", "n_pool": len(Ps.rows), "exchangeable": True,
                                  "by_n": run_within(F, Rs, masks_pool(L, Ps), "decode", NS, 2)}
        jdump(res, out_path)
        # ---- cross-pool cells, n = 200 (and the registered pair also along n)
        cross = {
            "repair_to_heldout": dict(cal=Rrep_tf, masks=masks_probe(L, rep), which="tf", tests={"heldout_disjoint": T_hd_tf}, ns=(20, 50, 100, 200), exch="approximately (probe items split by item; held-out rows that repeat a REPAIR query removed)",
                                      keys="teacher-forced keys on both sides (registered cell)"),
            "repair_prompt_keys": dict(cal=Rrep_d, masks=masks_probe(L, rep), which="decode", tests={"heldout_disjoint": T_hd_d, "BEN-T": T_bent, "MME-T": T_mmet}, ns=(200,),
                                       exch="heldout_disjoint: approximately; BEN-T, MME-T: no (another workload)", keys="prompt-only key of every row on both sides"),
            "ben_s_prompt_keys": dict(cal=Rs, masks=masks_pool(L, Ps), which="decode", tests={"heldout_disjoint": T_hd_d, "BEN-T": T_bent, "MME-T": T_mmet}, ns=(200,),
                                      exch="BEN-T: yes (parts of one pool split by source item); heldout_disjoint, MME-T: no", keys="prompt-only key of every row on both sides"),
            "mme_c_prompt_keys": dict(cal=Rmc, masks=masks_pool(L, Mc), which="decode", tests={"heldout_disjoint": T_hd_d, "BEN-T": T_bent, "MME-T": T_mmet}, ns=(200,),
                                      exch="MME-T: yes (split by image); heldout_disjoint, BEN-T: no", keys="prompt-only key of every row on both sides"),
        }
        for ci, (cname, c) in enumerate(cross.items()):
            print(" cross", cname, flush=True)
            res["cross"][cname] = {"exchangeable": c["exch"], "keys": c["keys"], "n_cal_pool": int(c["cal"].shape[0]),
                                   "by_n": run_within(F, c["cal"], c["masks"], c["which"], c["ns"], 10 + ci, extra_tests=c["tests"], keep_raw_n=(200,))}
            jdump(res, out_path)
    res["seconds"] = time.time() - t0
    jdump(res, out_path)
    print("DONE", name, round(res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
