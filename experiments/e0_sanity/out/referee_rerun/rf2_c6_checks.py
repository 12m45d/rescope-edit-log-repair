"""rf2_c6_checks.py -- referee check of C6 (shared scale versus per-edit thresholds), independent of check_c6.py.

A  Theorem C6cal(b)(iii), exhaustive: ALL symmetric coordinatewise rules that keep both edits alive on a small universe
   (T = 2, n = 1 and n = 2): the worst exchangeable law gives P(fire) >= min(T, n+1)/(n+1).
   Theorem C6cal(a)(ii), exhaustive: ALL symmetric rules for T = 1, n = 1, 2 on a 3-point universe: a rule that can fire
   somewhere has worst-case firing probability >= 1/(n+1).
B  Theorem C6cov on random instances of H_T with exact rationals: (a), (b), (c) by brute force over all shared scales.
C  Remark C6cal(c): the max-rank rule (Timans et al., Algorithm 2: column-wise ranks among the n calibration rows, row-wise
   maximum, conformal quantile of the maxima, that order statistic in every column) is interpolating; Monte Carlo of its
   false firing for independent and comonotone coordinates.
D  A rule that is NOT covered by any lower bound of Theorem C6cal(b): the split ("hybrid") rule
       block 1 (n1 rows): a_i = max of coordinate i;   block 2 (n2 rows): eta = conformal quantile of max_i (x_i - a_i)/u_i;
       tau_i = (a_i + eta u_i)^+          (u_i > 0 fixed in advance, e.g. 1 or the own coordinate s_i).
   D1 exact validity over exchangeable laws (urn laws with ties);  D2 on H_T;  D3 on I_T;  D4 on the stored P11 keys.
E  Table C6: independent recomputation of the first rows from the stored keys.
Output: rf2_c6_checks.json
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import itertools
import json
import math
import os
import time
from fractions import Fraction as Fr

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np  # noqa: E402

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P11 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p11_scopecut/out"
rng = np.random.default_rng(31)
OUT = {}

# thresholds of the chain restricted to a universe of integers: encoded so that "fires" = x >= code
NEG, POS = -10 ** 6, 10 ** 6


def chain(universe):
    c = [NEG]
    for t in universe:
        c += [t, t + 0.5]            # t  (gate x >= t)   and   t+  (gate x > t)
    return c + [POS]


# ---------------------------------------------------------------------------------------------- A
resA = {}
U = (0, 1)
alive_thr = [t for t in chain(U) if t <= max(U)]             # thresholds that some value of the universe reaches
for n in (1, 2):
    T = 2
    N = n + 1
    samples = list(itertools.combinations_with_replacement(U, n))          # multisets of n calibration values of one coordinate
    rules_1d = list(itertools.product(alive_thr, repeat=len(samples)))     # g: multiset -> threshold (alive on every sample)
    vecs = list(itertools.product(U, repeat=T))
    pops = list(itertools.combinations_with_replacement(vecs, N))          # extreme exchangeable laws: uniform orderings of a multiset
    worst_min = 1.0
    cnt = 0
    for g1 in rules_1d:
        for g2 in rules_1d:
            best = 0.0
            for pop in pops:
                hit = 0
                for f in range(N):                                         # which element is the fresh one (uniform)
                    cal = [pop[j] for j in range(N) if j != f]
                    t1 = g1[samples.index(tuple(sorted(v[0] for v in cal)))]
                    t2 = g2[samples.index(tuple(sorted(v[1] for v in cal)))]
                    hit += (pop[f][0] >= t1) or (pop[f][1] >= t2)
                best = max(best, hit / N)
            worst_min = min(worst_min, best)
            cnt += 1
    resA[f"b_iii_T=2_n={n}"] = {"rule_pairs": cnt, "min_over_rules_of_sup_over_laws_P(fire)": worst_min, "claimed_lower_bound_min(T,n+1)/(n+1)": min(T, N) / N,
                                "ok": worst_min >= min(T, N) / N - 1e-12}
U3 = (0, 1, 2)
for n in (1, 2):
    N = n + 1
    samples = list(itertools.combinations_with_replacement(U3, n))
    thr = chain(U3)
    pops = list(itertools.combinations_with_replacement(U3, N))
    bad = cnt = 0
    for g in itertools.product(thr, repeat=len(samples)):
        can_fire = any(any(x >= g[i] for x in U3) for i in range(len(samples)))
        best = 0.0
        for pop in pops:
            hit = sum(pop[f] >= g[samples.index(tuple(sorted(pop[j] for j in range(N) if j != f)))] for f in range(N))
            best = max(best, hit / N)
        cnt += 1
        bad += can_fire and best < 1 / N - 1e-12
    resA[f"a_ii_T=1_n={n}"] = {"symmetric_rules": cnt, "rules_that_can_fire_but_have_worst_case_below_1/(n+1)": bad}
OUT["A_exhaustive_small_universe"] = resA

# ---------------------------------------------------------------------------------------------- B
resB = {"instances": 0, "viol_a": 0, "viol_b": 0, "viol_c": 0}
for trial in range(300):
    T = int(rng.integers(2, 7))
    tau0 = [Fr(int(rng.integers(1, 40)), 10) for _ in range(T)]
    xs = sorted(set(Fr(int(v), 100) for v in rng.integers(20, 95, int(rng.integers(1, 4)))))
    xlo, b = xs[0], xs[-1]
    pw = [int(v) for v in rng.integers(1, 5, len(xs))]
    nu = [(x, Fr(w, sum(pw))) for x, w in zip(xs, pw)]
    g = [None]
    for i in range(1, T):
        ub = min(Fr(1), xlo * tau0[i] / tau0[0])                          # g_i/tau0_i < xlo/tau0_1  and  g_i <= 1
        g.append(ub * Fr(int(rng.integers(1, 99)), 100))
    assert all(0 < g[i] <= 1 and g[i] / tau0[i] < xlo / tau0[0] for i in range(1, T))
    resB["instances"] += 1
    wit = [(0, Fr(1))] + [(i, g[i]) for i in range(1, T)]                 # (edit, own coordinate); the other coordinates are 0
    # (a) oracle per-edit thresholds (b+, g_2, ..., g_T)
    ff = sum(p for x, p in nu if x > b)                                    # edit 1: x > b ; edits i >= 2: 0 >= g_i never
    cov = Fr(sum((c > b) if i == 0 else (c >= g[i]) for i, c in wit), T)
    resB["viol_a"] += not (ff == 0 and cov == 1)
    # (b) least repair from n benign queries
    for n in (1, 2, 3):
        eff = Fr(0)
        okcov = True
        for idx in itertools.product(range(len(nu)), repeat=n):
            wt = Fr(1)
            for t in idx:
                wt *= nu[t][1]
            M1 = max(nu[t][0] for t in idx)
            eff += wt * sum(p for x, p in nu if x > M1)
            okcov &= all((c > M1) if i == 0 else (c > 0) for i, c in wit)
        resB["viol_b"] += not (okcov and eff <= Fr(1, n + 1) and (len(nu) > 1 or eff == 0))
    # (c) every shared scale
    sb = [(x / tau0[0], p) for x, p in nu]
    sw = [c / tau0[i] for i, c in wit]
    crit = sorted(set([s for s, _ in sb] + sw))
    cand = [crit[0] - 1, crit[-1] + 1] + crit + [(crit[i] + crit[i + 1]) / 2 for i in range(len(crit) - 1)]
    for e in cand:
        FF = sum(p for s, p in sb if s > e)
        COV = Fr(sum(v > e for v in sw), T)
        resB["viol_c"] += not (FF == 1 or COV <= Fr(1, T))
OUT["B_heterogeneous_family_exact"] = resB

# ---------------------------------------------------------------------------------------------- C
def max_rank_thresholds(S, alpha):
    """Algorithm 2 of Timans et al.: S (n, m) calibration statistics.  Returns thresholds (m,), +inf if the rank exceeds n."""
    n, m = S.shape
    R = S.argsort(0).argsort(0) + 1                         # column-wise ranks 1..n (no ties for continuous data)
    rmax = R.max(1)
    k = int(math.ceil((n + 1) * (1 - alpha) - 1e-12))
    if k > n:
        return np.full(m, np.inf), None
    r_hat = int(np.sort(rmax)[k - 1])
    return np.sort(S, axis=0)[r_hat - 1], r_hat


resC = []
for T, n, alpha, reps in ((100, 489, 0.05, 3000), (100, 489, 0.10, 3000), (100, 1999, 0.05, 600), (10, 99, 0.10, 20000), (10, 49, 0.05, 20000),
                          (500, 489, 0.05, 600), (2, 99, 0.05, 20000)):
    fire = 0
    ranks = []
    for _ in range(reps):
        X = rng.random((n + 1, T))
        thr, r_hat = max_rank_thresholds(X[:n], alpha)
        fire += bool((X[n] > thr).any())
        ranks.append(r_hat if r_hat is not None else n + 1)
    p = fire / reps
    se = math.sqrt(max(p * (1 - p), 1e-9) / reps)
    # comonotone coordinates: one uniform per row
    fire_c = 0
    for _ in range(min(reps, 3000)):
        z = rng.random(n + 1)
        X = np.repeat(z[:, None], T, axis=1) if T <= 100 else np.repeat(z[:, None], 100, axis=1)
        thr, _ = max_rank_thresholds(X[:n], alpha)
        fire_c += bool((X[n] > thr).any())
    resC.append({"T": T, "n": n, "alpha": alpha, "reps": reps, "P_fire_independent_mc": p, "mc_se": se,
                 "exceeds_alpha_by_more_than_5se": bool(p > alpha + 5 * se), "mean_selected_rank": float(np.mean(ranks)),
                 "frac_reps_with_rank_n": float(np.mean(np.array(ranks) == n)),
                 "least_repair_1-(n/(n+1))^T": 1 - (n / (n + 1)) ** T, "P_fire_comonotone_mc": fire_c / min(reps, 3000)})
OUT["C_maxrank"] = resC

# ---------------------------------------------------------------------------------------------- D
def kq(n, a):
    return int(math.ceil((n + 1) * (1 - a) - 1e-12))


# D1: exact validity of the hybrid rule over urn laws with ties (additive form, u = 1)
resD1 = {"cases": 0, "violations": 0, "max_P_minus_bound": -1.0}
for trial in range(60):
    T = int(rng.integers(1, 3))
    size = int(rng.integers(3, 7))
    pop = [tuple(int(v) for v in rng.integers(0, 4, T)) for _ in range(size)]
    for n1 in (1, 2):
        for n2 in (1, 2, 3):
            if n1 + n2 + 1 > size:
                continue
            for a in (Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(2, 3)):
                k = math.ceil((1 - a) * (n2 + 1))
                l = n2 + 1 - k
                hit = tot = 0
                for idx in itertools.permutations(range(size), n1 + n2 + 1):
                    v = [pop[i] for i in idx]
                    aref = [max(v[j][c] for j in range(n1)) for c in range(T)]
                    sc = sorted(max(v[j][c] - aref[c] for c in range(T)) for j in range(n1, n1 + n2))
                    fresh = max(v[n1 + n2][c] - aref[c] for c in range(T))
                    tot += 1
                    hit += (k <= n2) and (fresh > sc[k - 1])
                p = Fr(hit, tot)
                resD1["cases"] += 1
                resD1["violations"] += p > Fr(l, n2 + 1)
                resD1["max_P_minus_bound"] = max(resD1["max_P_minus_bound"], float(p - Fr(l, n2 + 1)))
OUT["D1_hybrid_exact_validity_urn"] = resD1


def conf_q(scores, a):
    n = scores.shape[-1]
    k = kq(n, a)
    if k > n:
        return np.full(scores.shape[:-1], np.inf)
    return np.partition(scores, k - 1, axis=-1)[..., k - 1]


# D2: H_T with T = 100, benign x ~ U[0.3, 0.5] on coordinate 1, quiet witnesses at g = 0.2, loud witness at 1, constant reference
resD2 = []
T, g, reps = 100, 0.2, 20000
for n, a in ((40, 0.05), (40, 0.10), (100, 0.05), (200, 0.05), (489, 0.05), (489, 0.01)):
    x = 0.3 + 0.2 * rng.random((reps, n + 1))
    n1 = n // 2
    n2 = n - n1
    a1 = x[:, :n1].max(1)
    sc = np.maximum(x[:, n1:n] - a1[:, None], 0.0)                         # block-2 scores: max(x - a_1, 0 - 0, ...)
    eta = conf_q(sc, a)
    ff_h = (np.maximum(x[:, n] - a1, 0.0) > eta).mean()
    cov_h = ((1.0 > a1 + eta).astype(float) + (T - 1) * (g > eta)) / T
    # (S) conformal shared scale, constant reference: score = x
    eta_s = conf_q(x[:, :n], a)
    ff_s = (x[:, n] > eta_s).mean()
    cov_s = ((1.0 > eta_s).astype(float) + (T - 1) * (g > eta_s)) / T
    # (P) least repair on all n rows
    M1 = x[:, :n].max(1)
    ff_p = (x[:, n] > M1).mean()
    resD2.append({"n": n, "alpha": a, "n1": n1, "n2": n2, "hybrid_FF": float(ff_h), "hybrid_mean_COV": float(cov_h.mean()),
                  "hybrid_P(COV=1)": float((cov_h == 1).mean()), "shared_conformal_FF": float(ff_s), "shared_conformal_mean_COV": float(cov_s.mean()),
                  "per_edit_least_repair_FF": float(ff_p), "per_edit_least_repair_COV": 1.0, "1/T": 1 / T,
                  "hybrid_bound_l2/(n2+1)": (n2 + 1 - kq(n2, a)) / (n2 + 1) if kq(n2, a) <= n2 else 0.0})
OUT["D2_hybrid_on_H_T"] = resD2

# D3: I_T (independent coordinates uniform on (0, 1/2), witnesses = own keys with coordinate 1)
resD3 = []
for T, n, a, reps in ((100, 100, 0.05, 2000), (100, 489, 0.05, 1000), (100, 489, 0.01, 1000), (500, 489, 0.05, 300)):
    ffh = ffs = ffp = 0.0
    covh = covs = covp = 0.0
    for _ in range(reps):
        X = 0.5 * rng.random((n + 1, T))
        n1 = n // 2
        aref = X[:n1].max(0)
        sc = (X[n1:n] - aref[None, :]).max(1)
        eta = float(conf_q(sc[None, :], a)[0])
        ffh += float((X[n] - aref).max() > eta)
        covh += float((1.0 > aref + eta).mean())
        S = X.max(1)
        es = float(conf_q(S[None, :n], a)[0])
        ffs += float(S[n] > es)
        covs += float(1.0 > es)
        M = X[:n].max(0)
        ffp += float((X[n] > M).any())
        covp += float((1.0 > M).mean())
    resD3.append({"T": T, "n": n, "alpha": a, "reps": reps, "hybrid_FF": ffh / reps, "hybrid_COV": covh / reps, "shared_conformal_FF": ffs / reps,
                  "shared_conformal_COV": covs / reps, "per_edit_least_repair_FF": ffp / reps, "per_edit_least_repair_COV": covp / reps,
                  "1-(n/(n+1))^T": 1 - (n / (n + 1)) ** T})
OUT["D3_hybrid_on_I_T"] = resD3

# D4 + E: stored P11 keys
real = {}
for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
    kd = os.path.join(P11, name, "keys")
    if not os.path.exists(os.path.join(kd, "K_last.npy")):
        real[name] = "data not found"
        continue
    meta = json.load(open(os.path.join(kd, "meta.json")))
    fac = np.load(os.path.join(kd, "factors.npz"))
    K = np.load(os.path.join(kd, "K_last.npy")).astype(np.float64)
    W = fac["Wfac"].astype(np.float64)
    Ks = fac["K_solver"].astype(np.float64)
    T = W.shape[0]
    C = K @ W.T                                             # (rows, T): coordinate of every edit at the last prompt position
    s = np.einsum("ij,ij->i", Ks, W)
    ar = np.arange(T)
    row = {(q["metric"], q["edit"]): q["n"] for q in meta["seqs"] if q["kind"] == "edit"}
    rel = np.array([row[("rel", i)] for i in ar])
    tg = np.array([row[("tgen", i)] for i in ar])
    ig = np.array([row[("igen", i)] for i in ar])
    fit = np.array([q["n"] for q in meta["seqs"] if q["kind"] == "probe" and q["pool"] == "repair"])
    test = np.array([q["n"] for q in meta["seqs"] if q["kind"] == "probe" and q["pool"] == "heldout"])
    n, nt = len(fit), len(test)
    own = C[rel].copy()
    own[ar, ar] = -np.inf
    Cs = Ks @ W.T                                           # coordinates of the stored solver keys
    off = Cs.copy()
    off[ar, ar] = -np.inf
    tau_self = 1.05 * off.max(0)                            # reference "SELF" of the experiment (kappa = 1.05)

    def ev(tau, strict=True):
        F = (C[test] > tau[None, :]) if strict else (C[test] >= tau[None, :])
        cmp_ = (lambda x: x > tau) if strict else (lambda x: x >= tau)
        return {"rows_firing": int(F.any(1).sum()), "frac_rows_firing": float(F.any(1).mean()), "pairs_per_row": float(F.sum() / nt),
                "cov_rel": float(cmp_(C[rel, ar]).mean()), "cov_tgen": float(cmp_(C[tg, ar]).mean()), "cov_igen": float(cmp_(C[ig, ar]).mean())}

    res = {"T": int(T), "n_fit": int(n), "n_test": int(nt), "s_positive": bool((s > 0).all()), "tau_self_positive": bool((tau_self > 0).all())}
    # E: Table C6 rows
    m_a = C[fit].max(0)
    m_b = np.maximum(m_a, own.max(0))
    res["E_per_edit_max_over_fit_rows"] = ev(m_a)
    res["E_per_edit_max_over_fit_rows_and_other_own_rows"] = ev(m_b)
    eta_fc = max((C[fit] / s[None, :]).max(), (own / s[None, :]).max())
    res["E_fit_clean_shared_scale_tau0=s"] = ev(eta_fc * s)
    res["E_bounds"] = {"T/(n+1)": T / (n + 1), "1-(n/(n+1))^T": 1 - (n / (n + 1)) ** T, "T/(N_i+1)": T / (n + T), "N_i": int(n + T - 1)}
    # D4: hybrid rule against the conformal shared scale and the per-edit least repair, random halves of the FIT rows
    d4 = []
    R_ = 300
    for a in (0.05, 0.10, 0.20):
        acc = {k_: [] for k_ in ("hyb_u=s", "hyb_u=s+own", "hyb_mult", "shared_s", "shared_self")}
        for r in range(R_):
            p = rng.permutation(fit)
            b1, b2 = p[: n // 2], p[n // 2:]
            for lab, aref in (("hyb_u=s", C[b1].max(0)), ("hyb_u=s+own", np.maximum(C[b1].max(0), own.max(0)))):
                sc = ((C[b2] - aref[None, :]) / s[None, :]).max(1)
                eta = float(conf_q(sc[None, :], a)[0])
                acc[lab].append(ev(aref + eta * s))
            aref = np.maximum(C[b1].max(0), own.max(0))
            if (aref > 0).all():                                           # multiplicative form: tau = eta * a
                sc = (C[b2] / aref[None, :]).max(1)
                eta = float(conf_q(sc[None, :], a)[0])
                acc["hyb_mult"].append(ev(eta * aref))
            for lab, t0 in (("shared_s", s), ("shared_self", tau_self)):
                sub = p[: n]                                               # all n fit rows
                sc = (C[sub] / t0[None, :]).max(1)
                eta = float(conf_q(sc[None, :], a)[0])
                acc[lab].append(ev(eta * t0))
        row_ = {"alpha": a, "resplits": R_}
        for lab, lst in acc.items():
            if lst:
                row_[lab] = {k_: float(np.mean([e[k_] for e in lst])) for k_ in ("frac_rows_firing", "cov_rel", "cov_tgen", "cov_igen")}
                row_[lab]["n_used"] = len(lst)
        d4.append(row_)
    res["D4_hybrid_vs_shared_on_heldout"] = d4
    res["D4_per_edit_least_repair_all_fit_rows"] = ev(m_a)
    real[name] = res
OUT["REAL"] = real

OUT["seconds"] = round(time.time() - T0, 1)
json.dump(OUT, open(os.path.join(HERE, "rf2_c6_checks.json"), "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
