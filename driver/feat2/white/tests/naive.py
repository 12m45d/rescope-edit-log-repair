"""Second, NAIVE code path of every score of white.scores: numpy / scipy only.  No torch linear algebra, no call of the library
rescope, no Woodbury identity, no stored statistics: every quantity is recomputed from the raw keys by the textbook definition.

  cos_centred    explicit double loop over queries and edits
  white, whitecos   the d x d second-moment matrix C = K_w' K_w / N is FORMED, rho = kappa tr(C) / d, and (C + rho I) X = [k_i, k]
                 is solved by a dense Cholesky factorisation (d = 11008: 0.97 GB, about 10 s per ridge)
  cos_null, ncoord_null   the minimum-change vector of each edit straight from its definition as an equality-constrained least-squares
                 problem:   argmin |w' - w|^2  s.t.  B w' = 0,  <w', k_i> = s_i      =>     w' = w - A'(A A')^+ (A w - c),
                 A = [B; k_i'], c = (0, ..., 0, s_i)   (one SVD least-squares solve of the bordered (n + 1) x (n + 1) system per edit;
                 the keys of B may repeat, so A A' can be singular: the solve reports its rank, the gap of that decision and its residual)
All inputs: float arrays; everything is computed in float64.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def f64(x):
    return np.ascontiguousarray(np.asarray(x), dtype=np.float64)


def cos_centred(K, Q, Kw):
    """(n, T): cos(q - mu, k_i - mu), mu = mean of the rows of Kw."""
    K, Q, Kw = f64(K), f64(Q), f64(Kw)
    mu = Kw.sum(axis=0) / Kw.shape[0]
    out = np.empty((Q.shape[0], K.shape[0]))
    for a in range(Q.shape[0]):
        x = Q[a] - mu
        nx = np.sqrt(np.dot(x, x))
        for i in range(K.shape[0]):
            y = K[i] - mu
            out[a, i] = np.dot(x, y) / (nx * np.sqrt(np.dot(y, y)))
    return out


def second_moment(Kw):
    """(C (d, d), tr(C)) of the uncentred second moment C = (1/N) sum k k'."""
    Kw = f64(Kw)
    Cm = Kw.T @ Kw
    Cm /= Kw.shape[0]
    return Cm, float(np.trace(Cm))


def whitened(K, Q, Cm, trace, kappa):
    """(white (n, T), whitecos (n, T), rho) from the explicit d x d matrix: M x = solution of (C + rho I) y = x."""
    K, Q = f64(K), f64(Q)
    d = Cm.shape[0]
    rho = kappa * trace / d
    A = Cm.copy()
    A[np.diag_indices(d)] += rho
    cf = sla.cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
    X = sla.cho_solve(cf, np.concatenate([K, Q]).T.copy(), check_finite=False).T          # rows: M k_i (T of them), then M q
    MK, MQ = X[:K.shape[0]], X[K.shape[0]:]
    num = Q @ MK.T                                                      # q' M k_i
    own = np.einsum("id,id->i", K, MK)                                  # k_i' M k_i
    qq = np.einsum("nd,nd->n", Q, MQ)                                   # q' M q
    return num / own[None, :], num / np.sqrt(own[None, :] * qq[:, None]), rho


def explicit_inverse(Kw, rho):
    """(C + rho I)^-1 as a dense matrix (small problems only)."""
    Kw = f64(Kw)
    d = Kw.shape[1]
    return np.linalg.inv(Kw.T @ Kw / Kw.shape[0] + rho * np.eye(d))


RCOND = 1e-10               # relative cut-off of the singular values of the bordered Gram matrix (rank decision; the gap is reported)


def min_change(w, B, BBt, k, s):
    """argmin |w' - w|^2 s.t. B w' = 0, <w', k> = s (BBt = B B', passed in because it is the same for every edit).
    The keys in B may repeat (the REPAIR pool holds duplicate queries), so the bordered system G lam = r can be singular although the
    constraints are consistent: it is solved by an SVD least-squares solve (minimum-norm multiplier; w' itself is unique).
    -> (w', info): rank used, largest dropped and smallest kept singular value relative to the largest, relative residual of the solve
    (the system must be CONSISTENT: a large residual means w' does not satisfy the constraints and is no reference)."""
    n = B.shape[0]
    G = np.empty((n + 1, n + 1))
    G[:n, :n] = BBt
    Bk = B @ k
    G[:n, n], G[n, :n], G[n, n] = Bk, Bk, np.dot(k, k)
    r = np.empty(n + 1)
    r[:n] = B @ w
    r[n] = np.dot(k, w) - s
    lam, _, rank, sv = np.linalg.lstsq(G, r, rcond=RCOND)
    info = {"rank": int(rank), "n": n + 1, "kept_min": float(sv[rank - 1] / sv[0]), "dropped_max": float(sv[rank] / sv[0]) if rank < n + 1 else 0.0,
            "residual": float(np.linalg.norm(G @ lam - r) / max(np.linalg.norm(r), 1e-300))}
    return w - B.T @ lam[:n] - lam[n] * k, info


def _summary(infos, B):
    """What the naive solves did, over the edits: ranks, the gap of the rank decision, the worst residual, distinct constraint keys."""
    distinct = len({B[i].tobytes() for i in range(B.shape[0])})
    return {"n_keys": int(B.shape[0]), "n_distinct_keys": distinct, "rank_min": min(i["rank"] for i in infos) - 1, "rank_max": max(i["rank"] for i in infos) - 1,
            "smallest_kept_singular_value_rel": min(i["kept_min"] for i in infos), "largest_dropped_singular_value_rel": max(i["dropped_max"] for i in infos),
            "max_relative_residual": max(i["residual"] for i in infos)}


def cos_null(K, Q, B):
    """((n, T), info): <w_i', q> / |q|, w_i' = minimum change of the unit vector k_i / |k_i| with zero coordinate on the rows of B and
    unchanged coordinate <w_i', k_i> = |k_i|."""
    K, Q, B = f64(K), f64(Q), f64(B)
    BBt = B @ B.T
    qn = np.sqrt(np.einsum("nd,nd->n", Q, Q))
    out, infos = np.empty((Q.shape[0], K.shape[0])), []
    for i in range(K.shape[0]):
        kn = np.sqrt(np.dot(K[i], K[i]))
        w, info = min_change(K[i] / kn, B, BBt, K[i], kn)
        infos.append(info)
        out[:, i] = (Q @ w) / qn
    return out, _summary(infos, B)


def ncoord_null(W, K, Q, B):
    """((n, T), info): <w_i', q> / s_i, w_i' = minimum change of the read-out vector W_i with zero coordinate on the rows of B and
    unchanged own coordinate s_i = <W_i, k_i>."""
    W, K, Q, B = f64(W), f64(K), f64(Q), f64(B)
    BBt = B @ B.T
    out, infos = np.empty((Q.shape[0], K.shape[0])), []
    for i in range(K.shape[0]):
        s = np.dot(W[i], K[i])
        w, info = min_change(W[i], B, BBt, K[i], s)
        infos.append(info)
        out[:, i] = (Q @ w) / s
    return out, _summary(infos, B)


def cosine(K, Q):
    """(n, T): the raw cosine of every query to every key."""
    K, Q = f64(K), f64(Q)
    return (Q @ K.T) / (np.sqrt(np.einsum("nd,nd->n", Q, Q))[:, None] * np.sqrt(np.einsum("id,id->i", K, K))[None, :])
