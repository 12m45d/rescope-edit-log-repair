"""Verification, extra: the numbers of the OPTIMUM CODE (not one of the four tasks), as far as they can be certified
without re-running a branch-and-bound solver.            usage: python v_opt_lb.py [--workers 8]   -> verify/out/v_opt_lb.json

The measuring agent's optimum rows (out/opt_rows_<log>.jsonl, key 'last') give per complaint and tolerance c a bracket
opt_lb <= OPT_c <= opt_ub and a deletion set R_best.  With MY contributions (explicit vectors w_i = u_i <psi_i, k>, last
prompt key from the final key cache, G_ij = <w_i, w_j>, fp64) this script certifies, independently of that code:
  UPPER bound   R_best is a deletion set of size opt_ub with ||sum_{kept} w_i|| <= c ||D||               (direct check)
  LOWER bound   a Lagrangian certificate that needs no solver.  Let delta be a diagonal shift with M := G - Diag(delta)
                positive definite (checked on the matrix scaled to the unit diagonal of G, where the test is well
                conditioned: least eigenvalue > PD_MIN).  Every kept set with 0/1 indicator z has z^T G z = z^T M z + delta^T z,
                so for every multiplier lam >= 0 and EVERY point y of the box [0, 1]^T
                    |kept| = 1^T z <= lam c^2 ||D||^2 + h(y) + sum_i max(g_i (1 - y_i), -g_i y_i) =: U,
                    h(y) = (1 - lam delta)^T y - lam y^T M y   (concave),   g = grad h(y) = 1 - lam delta - 2 lam M y,
                because 1^T z <= 1^T z - lam (z^T G z - c^2 ||D||^2) = lam c^2 ||D||^2 + h(z) and h(z) <= h(y) + g^T (z - y).
                U is evaluated in fp64 from G, delta, lam, y alone; the certified bound is OPT_c >= T - floor(U + SLACK).
                How delta, lam and y are found affects only how tight U is, never its validity:
                  lam    bisection on the sign of y^T G y - c^2 ||D||^2 at the maximiser y of h (bounded least squares);
                  delta  (i) s gamma diag(G), s in SHIFTS (positive definite for s < 1 by the definition of gamma);
                         (ii) REFINE_ITERS rounds from 0.9 gamma diag(G): the shift is halved on the coordinates where the
                         maximiser is integral and raised on the fractional ones, along y_i (1 - y_i) G_ii, by 0.9 of the
                         largest step that keeps M positive definite (a generalised eigenvalue).
                The least U over all (delta, lam) visited is kept.  (SHIFTS and the refinement rule were fixed after trials
                on five complaints of TB-L100-ae-l2_3; they decide how tight the bound is, not whether it is valid.)
  self-test     on random instances with T <= 12 the certified bound never exceeds the brute-force optimum.
  my bracket    LB_v = max(my mu-bound lb1, my triangle bound, certified bound) <= OPT_c <= UB_v = min(my LF, FO, BI, opt_ub)
Compared with the report: its opt_lb against LB_v; its medians of rule / OPT and (bound of the proposition) / OPT over the
complaints within its budget (the first n complaints of its seeded order, n = optimum_run.n_complaints_within_budget)
against the same medians with MY bracket.  A lower bound of the report that exceeds mine is NOT refuted: it rests on the
report's branch and bound, which is not repeated here.
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import argparse
import itertools
import math
import multiprocessing as mp
import os
import time

import numpy as np
import scipy.linalg as sla
from scipy.optimize import lsq_linear

SHIFTS = (0.9, 0.99, 0.999)
REFINE_ITERS = 6
SLACK = 1e-6
PD_MIN = 1e-10
_S = {}


def _chol_scaled(M, sc):
    """Upper triangular R with M = R^T R, from the Cholesky factor of M scaled to the unit diagonal of G (sc = diag(G)^-1/2):
    the factorisation is then as well conditioned as the scaled matrix, whatever the spread of the norms."""
    return sla.cholesky(M * np.outer(sc, sc), lower=False) / sc[None, :]


def dual_bound(G, thr, delta, iters=28):
    """(least U over the multipliers visited, its multiplier, its point y); (None, None, None) if M is not positive
    definite or if the search breaks down numerically (a bound that is not found is not used; validity is not affected)."""
    T = len(G)
    sc = 1.0 / np.sqrt(np.diag(G))
    M = G - np.diag(delta)
    M = 0.5 * (M + M.T)
    if not float(sla.eigvalsh(M * np.outer(sc, sc), subset_by_index=[0, 0])[0]) > PD_MIN:
        return None, None, None
    try:
        R = _chol_scaled(M, sc)
    except sla.LinAlgError:
        return None, None, None
    one = np.ones(T)

    def U_of(lam):
        c = one - lam * delta
        b = sla.solve_triangular(R, c, trans="T", lower=False) / (2.0 * math.sqrt(lam))
        y = np.clip(lsq_linear(math.sqrt(lam) * R, b, bounds=(0.0, 1.0), method="bvls", tol=1e-13, max_iter=3000).x, 0.0, 1.0)
        My = M @ y
        g = c - 2.0 * lam * My
        U = lam * thr + float(c @ y) - lam * float(y @ My) + float(np.where(g > 0, g * (1.0 - y), -g * y).sum())
        return U, float(y @ My) + float(delta @ y), y

    lo = hi = None
    lam, best = 1.0 / thr, (np.inf, None, None)
    try:
        for _ in range(200):
            U, q, y = U_of(lam)
            if not (np.isfinite(U) and np.isfinite(q)):
                break
            if U < best[0]:
                best = (U, lam, y)
            if q > thr:
                lo, lam = lam, lam * 4.0
            else:
                hi, lam = lam, lam / 4.0
            if lo is not None and hi is not None:
                break
        if lo is not None and hi is not None:
            for _ in range(iters):
                lam = math.sqrt(lo * hi)
                U, q, y = U_of(lam)
                if not (np.isfinite(U) and np.isfinite(q)):
                    break
                if U < best[0]:
                    best = (U, lam, y)
                if q > thr:
                    lo = lam
                else:
                    hi = lam
    except (sla.LinAlgError, ValueError, FloatingPointError):
        pass
    return best if np.isfinite(best[0]) else (None, None, None)


def max_step(G, delta, d):
    """Largest tau with G - Diag(delta + tau d) positive semidefinite (d >= 0): 1 / lambda_max(R^-T Diag(d) R^-1)."""
    M = G - np.diag(delta)
    R = _chol_scaled(0.5 * (M + M.T), 1.0 / np.sqrt(np.diag(G)))
    X = sla.solve_triangular(R, np.diag(np.sqrt(d)), trans="T", lower=False)
    n = len(G)
    return 1.0 / float(sla.eigvalsh(X @ X.T, subset_by_index=[n - 1, n - 1])[0])


def certified_kept_bound(G, thr, gam):
    """Least U over the scaled shifts and the refined shifts: an upper bound on the size of every feasible kept set."""
    dg = np.diag(G).copy()
    best, how = np.inf, None
    for s in SHIFTS:
        U, _, _ = dual_bound(G, thr, s * gam * dg)
        if U is not None and U < best:
            best, how = U, f"scaled {s}"
    delta = 0.9 * gam * dg
    for it in range(REFINE_ITERS + 1):
        U, lam, y = dual_bound(G, thr, delta)
        if U is None:
            break
        if U < best:
            best, how = U, f"refined {it}"
        w = y * (1.0 - y)
        if it == REFINE_ITERS or w.max() < 1e-9:
            break
        d = (w / w.max()) * dg
        delta = np.where(w < 1e-6, 0.5 * delta, delta)
        try:
            delta = delta + 0.9 * max_step(G, delta, d) * d
        except sla.LinAlgError:
            break
    if not np.isfinite(best):                                         # no admissible shift: the trivial bound (every edit kept)
        best, how = float(len(G)), "none"
    return best, how


def tri_bound(nrm, Dn, c):
    s = np.concatenate([[0.0], np.cumsum(np.sort(nrm)[::-1])])
    need = (1.0 - c) * Dn * (1.0 - 1e-12)
    return int(np.argmax(s >= need)) if s[-1] >= need else len(nrm)


def gamma_of(G):
    sc = 1.0 / np.sqrt(np.diag(G))
    return float(sla.eigvalsh(G * np.outer(sc, sc), subset_by_index=[0, 0])[0])


def selftest(n=40, seed=0):
    """Certified bound against the brute-force optimum on small random instances (all kept sets enumerated).
    Two families: 'plain' (Gaussian contributions with a common component, norms within a factor of about 15) and 'wide'
    (contributions c_i u_i as at one key of a log: correlated write vectors u_i, signed coordinates c_i with a log-normal
    magnitude, so that the norms spread over several orders of magnitude, as on the real logs)."""
    rng = np.random.default_rng(seed)
    res = {"instances": 0, "checks": 0, "n_bound_le_opt": 0, "n_bound_eq_opt": 0, "max_opt_minus_bound": 0, "max_norm_spread": 0.0,
           "by_family": {"plain": {"checks": 0, "n_bound_le_opt": 0}, "wide": {"checks": 0, "n_bound_le_opt": 0}}}
    for it in range(2 * n):
        fam = "plain" if it < n else "wide"
        T = int(rng.integers(6, 13))
        m = int(rng.integers(T, T + 4))
        if fam == "plain":
            Wm = rng.normal(size=(T, m)) * rng.uniform(0.2, 3.0, size=(T, 1)) + rng.normal(size=(1, m)) * rng.uniform(0, 1.5)
        else:
            Um = rng.normal(size=(T, m)) + rng.normal(size=(1, m)) * rng.uniform(0.2, 0.8)
            Wm = Um * (rng.choice([-1.0, 1.0], size=(T, 1)) * np.exp(rng.normal(scale=2.5, size=(T, 1))))
        G = Wm @ Wm.T
        G = 0.5 * (G + G.T)
        gam = gamma_of(G)
        nr = np.sqrt(np.diag(G))
        res["max_norm_spread"] = max(res["max_norm_spread"], float(nr.max() / nr.min()))
        Y = np.array(list(itertools.product((0.0, 1.0), repeat=T)))
        qv = np.einsum("ij,jk,ik->i", Y, G, Y)
        for c in (0.1, 0.3, 0.5, 0.8):
            thr = c * c * float(G.sum())
            opt = int(T - Y[qv <= thr].sum(1).max())
            U, _ = certified_kept_bound(G, thr, gam)
            lb = int(T - math.floor(U + SLACK))
            res["checks"] += 1
            res["n_bound_le_opt"] += int(lb <= opt)
            res["n_bound_eq_opt"] += int(lb == opt)
            res["max_opt_minus_bound"] = max(res["max_opt_minus_bound"], opt - lb)
            res["by_family"][fam]["checks"] += 1
            res["by_family"][fam]["n_bound_le_opt"] += int(lb <= opt)
        res["instances"] += 1
    res["pass"] = bool(res["n_bound_le_opt"] == res["checks"])
    return res


def _init(name):
    F = C.factors(name)
    _S.update(U=F["U"], Psi=F["Psi"])


def one(args):
    o, k, mine = args
    U, Psi = _S["U"], _S["Psi"]
    T = U.shape[0]
    c = Psi @ np.asarray(k, dtype=np.float64)
    Wc = U * c[:, None]
    G = Wc @ Wc.T
    G = 0.5 * (G + G.T)
    nrm = np.sqrt(np.diag(G))
    gam = gamma_of(G)
    D = Wc.sum(0)
    Dn = float(np.linalg.norm(D))
    out = {"group": o["group"], "ptype": o["ptype"], "order_idx": int(o["order_idx"]), "tol": {}}
    for t in C.TOLS:
        q = o["tau"][str(t)]
        thr = t * t * Dn * Dn
        kept = np.ones(T, dtype=bool)
        kept[np.asarray(q["R_best"], dtype=int)] = False
        rel = float(np.linalg.norm(Wc[kept].sum(0)) / Dn)
        ub_ok = bool(rel <= t and len(set(q["R_best"])) == int(q["opt_ub"]))
        Umin, how = certified_kept_bound(G, thr, gam)
        m = mine["tol"][str(t)]
        lb_dual = int(T - math.floor(Umin + SLACK))
        lb_tri = tri_bound(nrm, Dn, t)
        LBv = max(int(m["lb1"]), lb_tri, lb_dual)
        UBv = min(int(m["lf"]), int(m["fo"]), int(m["bi"]), int(q["opt_ub"]) if ub_ok else T)
        out["tol"][str(t)] = {"U": Umin, "certificate": how, "lb_dual": lb_dual, "lb_tri": lb_tri,
                              "lb1": int(m["lb1"]), "lf": int(m["lf"]), "fo": int(m["fo"]), "bi": int(m["bi"]), "LBv": LBv, "UBv": UBv,
                              "ub_feasible": ub_ok, "ub_residual_over_tolerance": rel / t,
                              "report_opt_lb": int(q["opt_lb"]), "report_opt_ub": int(q["opt_ub"]), "report_how": q["how"],
                              "report_kelley_kept_bound": q.get("kelley", {}).get("kept_bound"), "report_kelley_converged": q.get("kelley", {}).get("converged")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    fh = open(os.path.join(C.LOGDIR, "v_opt_lb.log"), "a")
    PN = C.jload(os.path.join(C.REPORT_OUT, "paper_numbers.json"))
    st = selftest()
    C.log(f"self-test: certified bound <= brute-force optimum in {st['n_bound_le_opt']}/{st['checks']} checks (equal in {st['n_bound_eq_opt']}, "
          f"largest gap {st['max_opt_minus_bound']}; by family {st['by_family']}; largest norm spread {st['max_norm_spread']:.3g}) ({time.time() - t0:.0f}s)", fh)
    if not st["pass"]:
        raise SystemExit("self-test failed: the certificate is not valid")
    out = {"definition": __doc__, "shifts": list(SHIFTS), "refine_iters": REFINE_ITERS, "slack": SLACK, "pd_min": PD_MIN, "selftest": st, "logs": {}}
    for name in C.LOGS:
        rows = C.probe_rows(name)
        cand = C.candidates(rows)
        comps = {c["group"]: c for c in cand["image"] + cand["text"]}
        mine = {r["group"]: r["keydefs"]["last"] for r in C.jload(os.path.join(C.OUT, f"v_cond_rollback_{name}.json"))["rows"]}
        opt = sorted([o for o in C.jsonl(os.path.join(C.REPORT_OUT, f"opt_rows_{name}.jsonl")) if o["keydef"] == "last" and "tau" in o],
                     key=lambda o: o["order_idx"])
        Kl, _ = C.last_keys(name, [comps[o["group"]] for o in opt])
        jobs = [(o, Kl[j].numpy(), mine[o["group"]]) for j, o in enumerate(opt)]
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(name,)) as pool:
            res = pool.map(one, jobs, chunksize=1)
        n_within = int(PN[f"rollback.{name}.optimum_run.n_complaints_within_budget"])
        L = {"n_rows": len(res), "n_within_budget": n_within, "rows": res, "per_tol": {}}
        for t in C.TOLS:
            x = [r["tol"][str(t)] for r in res]
            w = [r["tol"][str(t)] for r in res if r["order_idx"] < n_within]
            d = [v["report_opt_lb"] - v["LBv"] for v in x]
            kc = [v["U"] - v["report_kelley_kept_bound"] for v in x if v["report_kelley_converged"]]
            s = {"all_rows": {"n": len(x), "n_ub_feasible": int(sum(v["ub_feasible"] for v in x)),
                              "n_report_lb_confirmed": int(sum(v["LBv"] >= v["report_opt_lb"] for v in x)),
                              "report_lb_minus_mine_hist": {str(k): int(sum(e == k for e in d)) for k in sorted(set(d))},
                              "n_my_bound_above_report_ub": int(sum(v["LBv"] > v["report_opt_ub"] for v in x)),
                              "n_report_bracket_closed": int(sum(v["report_opt_lb"] == v["report_opt_ub"] for v in x)),
                              "n_my_bracket_closed": int(sum(v["LBv"] == v["UBv"] for v in x)),
                              "n_report_closed_and_mine_closed": int(sum(v["report_opt_lb"] == v["report_opt_ub"] and v["LBv"] == v["UBv"] for v in x)),
                              "certificate_hist": {k: int(sum(v["certificate"].split()[0] == k for v in x)) for k in ("scaled", "refined")},
                              "n_kelley_converged": len(kc), "max_my_U_minus_converged_kelley": max(kc, default=None),
                              "median_my_U_minus_converged_kelley": C.med(kc)},
                 "within_budget": {"n": len(w), "n_my_bracket_closed": int(sum(v["LBv"] == v["UBv"] for v in w)),
                                   "n_report_lb_confirmed": int(sum(v["LBv"] >= v["report_opt_lb"] for v in w)),
                                   "median_LBv": C.med([v["LBv"] for v in w]), "median_UBv": C.med([v["UBv"] for v in w]),
                                   **{f"median_{r}_over_opt_lo": C.med([v[r] / v["UBv"] for v in w]) for r in ("lf", "fo", "bi")},
                                   **{f"median_{r}_over_opt_hi": C.med([v[r] / v["LBv"] for v in w]) for r in ("lf", "fo", "bi")},
                                   "median_lb_over_opt_lo": C.med([v["lb1"] / v["UBv"] for v in w]),
                                   "median_lb_over_opt_hi": C.med([v["lb1"] / v["LBv"] for v in w])},
                 "report": {k: PN.get(f"rollback.{name}.c{t}.{k}") for k in
                            ("opt_n_complaints", "opt_n_certified", "opt_median_lower_bound", "opt_median_upper_bound",
                             "opt_median_lf_over_opt_lo", "opt_median_lf_over_opt_hi", "opt_median_fo_over_opt_lo", "opt_median_fo_over_opt_hi",
                             "opt_median_bi_over_opt_lo", "opt_median_bi_over_opt_hi", "opt_median_lb_over_opt_lo", "opt_median_lb_over_opt_hi")}}
            L["per_tol"][str(t)] = s
            C.log(f"{name} c = {t}: {len(x)} rows, upper bounds feasible {s['all_rows']['n_ub_feasible']}; report's lower bound minus mine: "
                  f"{s['all_rows']['report_lb_minus_mine_hist']}; within budget ({len(w)}): my bracket closed {s['within_budget']['n_my_bracket_closed']} "
                  f"(report {s['report']['opt_n_certified']}); BI/OPT median {s['within_budget']['median_bi_over_opt_lo']:.3f}..{s['within_budget']['median_bi_over_opt_hi']:.3f} "
                  f"(report {s['report']['opt_median_bi_over_opt_lo']:.3f}..{s['report']['opt_median_bi_over_opt_hi']:.3f}); bound/OPT "
                  f"{s['within_budget']['median_lb_over_opt_lo']:.3f}..{s['within_budget']['median_lb_over_opt_hi']:.3f} "
                  f"(report {s['report']['opt_median_lb_over_opt_lo']:.3f}..{s['report']['opt_median_lb_over_opt_hi']:.3f})", fh)
        out["logs"][name] = L
    out["sec"] = time.time() - t0
    C.jdump(out, os.path.join(C.OUT, "v_opt_lb.json"))
    C.log(f"wrote v_opt_lb.json ({out['sec']:.0f}s)", fh)


if __name__ == "__main__":
    main()
