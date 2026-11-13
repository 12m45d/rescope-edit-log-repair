"""Pure functions of E5 (no file access): Gram matrix of the contributions, conditioning and cancellation statistics,
rollback by rule, the bounds of the conditioning proposition, exact vertex cover number of the negative graph.

Notation (Sec. 2 / 3.1 of the paper).  At complaint keys K (A, d_in) edit i contributes w_i = u_i (psi_i^T K^T), a
(d_out x A) matrix (A = 1: a vector); G_ij = <w_i, w_j>_F = (u_i . u_j) (c_i . c_j) with c_i = K psi_i (SIGNED).
D = sum_i w_i, ||D||^2 = sum_ij G_ij.  For a kept set with indicator y the residual is r = sum_{y_i = 1} w_i and
||r||^2 = y^T G y.  A deletion set is feasible at tolerance c iff ||r||^2 <= c^2 ||D||^2 (squared norms, fp64).
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------- Gram
def gram(UU: np.ndarray, C: np.ndarray) -> np.ndarray:
    """UU = U U^T (T, T); C (T, A) with C[i] = K psi_i.  G_ij = UU_ij <C_i, C_j> (symmetrised)."""
    G = UU * (C @ C.T)
    return 0.5 * (G + G.T)


def resid2(G: np.ndarray, y: np.ndarray) -> float:
    return float(y @ G @ y)


# ----------------------------------------------------------------------------- conditioning and cancellation
def cond_stats(G: np.ndarray) -> dict:
    """Statistics of one complaint.  Ghat = Lambda^-1/2 G Lambda^-1/2 (unit diagonal), gamma <= Gamma its extreme
    eigenvalues (numpy eigvalsh, fp64).  Pairs are the T (T - 1) / 2 unordered pairs i < j; a pair is NEGATIVE iff
    G_ij < 0 (signed inner product; exact zero counts as not negative)."""
    T = len(G)
    dg = np.diag(G).copy()
    out = {"T": int(T), "n_zero_contributions": int((dg <= 0).sum())}
    D2 = float(G.sum())
    S = float(dg.sum())
    out.update(D2=D2, sum_w2=S, energy=D2 / S if S > 0 else None)
    if out["n_zero_contributions"]:
        out.update(gamma=None, Gamma=None, ratio=None, frac_neg=None, min_cos=None, cancellation_free=None, n_neg_pairs=None,
                   norm_spread=None)
        return out
    nrm = np.sqrt(dg)
    Gh = G / np.outer(nrm, nrm)
    ev = np.linalg.eigvalsh(Gh)
    iu = np.triu_indices(T, 1)
    off = G[iu]
    neg = off < 0
    out.update(gamma=float(ev[0]), Gamma=float(ev[-1]), ratio=float(ev[-1] / ev[0]) if ev[0] > 0 else None,
               gamma_positive=bool(ev[0] > 0),
               frac_neg=float(neg.mean()), n_neg_pairs=int(neg.sum()), cancellation_free=bool(neg.sum() == 0),
               min_cos=float(Gh[iu].min()), median_cos=float(np.median(Gh[iu])), mean_abs_cos=float(np.abs(Gh[iu]).mean()),
               norm_spread=float(nrm.max() / nrm.min()),
               n_edits_for_half_w2=int(np.searchsorted(np.cumsum(np.sort(dg)[::-1]), 0.5 * S) + 1))
    return out


# ----------------------------------------------------------------------------- vertex cover of the negative graph
def max_clique_size(adj: list) -> int:
    """Clique number of a graph given as bit masks (adj[v] = neighbours of v), exact: branch and bound with a greedy
    colouring bound (Tomita-Seki style).  Python integers as bit sets."""
    n = len(adj)
    best = 0

    def expand(P: int, size: int):
        nonlocal best
        order, bound = [], []
        U, colour = P, 0
        while U:                                   # greedy colouring of the candidates: colour classes are independent sets
            colour += 1
            Q = U
            while Q:
                v = (Q & -Q).bit_length() - 1
                Q &= ~((1 << v) | adj[v])
                U &= ~(1 << v)
                order.append(v)
                bound.append(colour)
        for t in range(len(order) - 1, -1, -1):
            if size + bound[t] <= best:
                return
            v = order[t]
            Pv = P & adj[v]
            if Pv:
                expand(Pv, size + 1)
            elif size + 1 > best:
                best = size + 1
            P &= ~(1 << v)

    if n:
        expand((1 << n) - 1, 0)
    return best


def vertex_cover_number(G: np.ndarray) -> dict:
    """nu = vertex cover number of the NEGATIVE graph {i ~ j : G_ij < 0}, exact: nu = T - alpha, alpha = the largest
    set of edits whose contributions have pairwise non-negative inner products = clique number of the complement."""
    T = len(G)
    nonneg = (G >= 0)
    np.fill_diagonal(nonneg, False)
    adj = [int(sum(1 << int(j) for j in np.nonzero(nonneg[i])[0])) for i in range(T)]
    alpha = max_clique_size(adj)
    return {"nu": int(T - alpha), "alpha_nonneg": int(alpha), "nu_exact": True}


# ----------------------------------------------------------------------------- bounds
def mu(dg: np.ndarray, x: float) -> int:
    """Least number of contributions whose deletion leaves sum_{kept} ||w_i||^2 <= x (largest norms first)."""
    sq = np.sort(np.asarray(dg, dtype=np.float64))[::-1]
    kept = np.concatenate([np.cumsum(sq[::-1])[::-1], [0.0]])      # kept[k] = sum of the T - k smallest squared norms
    return int(np.argmax(kept <= x))


def lb_triangle(dg: np.ndarray, D2: float, c: float) -> int:
    """OPT_c >= min{k : sum of the k largest ||w_i|| >= (1 - c) ||D||}  (||sum_{i in R} w_i|| >= ||D|| - ||r||)."""
    nr = np.sort(np.sqrt(np.asarray(dg, dtype=np.float64)))[::-1]
    need = (1.0 - c) * np.sqrt(max(D2, 0.0))
    cs = np.concatenate([[0.0], np.cumsum(nr)])
    return int(np.argmax(cs >= need * (1 - 1e-12))) if cs[-1] >= need * (1 - 1e-12) else len(nr)


# ----------------------------------------------------------------------------- rollback by rule
def _run(G: np.ndarray, thr: float, pick) -> list:
    """Delete pick(y, Gy) until y^T G y <= thr; the criterion is re-evaluated from G at every step (fp64)."""
    n = len(G)
    y = np.ones(n)
    R = []
    while resid2(G, y) > thr and len(R) < n:
        i = int(pick(y, G @ y))
        y[i] = 0.0
        R.append(i)
    return R


def rule_lf(G, thr) -> list:
    """Largest-norm-first: delete by decreasing ||w_i|| (ties: lower index first) until the criterion holds."""
    order = np.argsort(-np.diag(G), kind="stable").tolist()
    return _run(G, thr, lambda y, Gy: order[int(len(y) - y.sum())])


def rule_fo(G, thr) -> list:
    """First-order greedy of the paper (ADAPTIVE): delete a kept edit that maximises <r, w_i> for the CURRENT residual r."""
    def pick(y, Gy):
        s = np.where(y > 0, Gy, -np.inf)
        return np.argmax(s)
    return _run(G, thr, pick)


def rule_bi(G, thr) -> list:
    """Best-improvement greedy: delete a kept edit that minimises ||r - w_i||^2 = ||r||^2 - 2 <r, w_i> + ||w_i||^2."""
    dg = np.diag(G)

    def pick(y, Gy):
        s = np.where(y > 0, -2.0 * Gy + dg, np.inf)
        return np.argmin(s)
    return _run(G, thr, pick)


def rule_fo_static(G, thr) -> list:
    """Track C's 'first-order greedy': a FIXED order by decreasing <D, w_i> (not the paper's adaptive rule)."""
    order = np.argsort(-G.sum(1), kind="stable").tolist()
    return _run(G, thr, lambda y, Gy: order[int(len(y) - y.sum())])


RULES = {"lf": rule_lf, "fo": rule_fo, "bi": rule_bi, "fo_static": rule_fo_static}


def rollback_rules(G: np.ndarray, c: float, gamma, Gamma) -> dict:
    """Deletion counts of the rules at tolerance c, the bounds of Prop. prop:cond and the triangle bound."""
    dg = np.diag(G)
    D2 = float(G.sum())
    thr = c * c * D2
    out = {"c": c}
    for k, fn in RULES.items():
        R = fn(G, thr)
        y = np.ones(len(G))
        y[R] = 0.0
        if resid2(G, y) > thr:
            raise RuntimeError(f"rule {k} returned an infeasible set")
        out[k] = len(R)
    out["best_rule"] = int(min(out[k] for k in RULES))
    ok = gamma is not None and gamma > 0
    out["lb_cond"] = mu(dg, thr / gamma) if ok else 0           # Prop. prop:cond: mu(c^2 ||D||^2 / gamma) <= OPT_c  (gamma > 0)
    out["lb_cond_valid"] = bool(ok)
    out["lf_guar"] = mu(dg, thr / Gamma) if Gamma else None     # Prop. prop:cond: LF <= mu(c^2 ||D||^2 / Gamma)
    out["lb_tri"] = lb_triangle(dg, D2, c)
    out["lb"] = int(max(out["lb_cond"], out["lb_tri"]))
    return out
