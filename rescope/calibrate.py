"""Calibration of the thresholds of a scoped log.  Pure functions: (scoped log, data) -> (new scoped log, record).

calibrate_two_block(S, queries, alpha)
                              THE DEFAULT (protocol amendment of 2026-11-10): two-block per-detector conformal rule, for any
                              feature (definition: the comment block in the second half of this file).
calibrate_shared(S, queries, alpha)
                              one shared conformal threshold / scale, exact form (strict, exact rank, floating-point safe).
calibrate_dense(gate, scores, alpha)
                              the same rules for the dense baseline gates (one scalar score per query).
calibrate_log_only(S, q)      needs nothing but the log: tau = the q-quantile of the pooled normalised
                              cross-talk {r_i(k_j) : j != i} of the log's own stored solver keys.
calibrate_conformal(S, scores, alpha)
                              split-conformal global multiplier eta from n benign calibration queries (the earlier form of
                              the shared-scale rule: non-strict gate with a 1e-9 guard; kept unchanged);
                              guarantee  P(a fresh exchangeable benign query fires) <= alpha.

Own-key rule (both): an edit's guard keys (its own stored solver key, plus witnesses granted by repairs and keys
added with repair.add_guards) must remain candidates of the edit.  Unless `allow_silence=True`, a guard that the
literal rule would silence gets ONE detector of its edit capped at (1 - TIE) * its coordinate (`keep_guards`); the
record counts the own keys / guards the literal rule would have silenced.  For the conformal rule a capped
threshold is BELOW the conformal level: the guarantee is then void for that edit, and the record says so.
A guard is a MUST-FIRE key: one that does not fire under the thresholds a calibration starts from (after
allow_silence, a forced repair, a null repair, or add_guards(allow_nonfiring=True)) is made to fire as well, i.e.
its edit's threshold is LOWERED to its coordinate; the record counts these (`n_guards_not_firing_before`).

The conformal guarantee is MARGINAL over the calibration draw: the expected false-firing rate is <= alpha, a single
draw of n queries can give more (calibrate on seeded random subsets, splits.sample, and report the mean over draws).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .scoped import INF, TIE, ScopedLog

GUARD = 1e-9           # relative inflation of the conformal quantile: makes "fires" imply "score > quantile" despite round-off
ETA_MIN = 1e-200       # eta when the conformal quantile is <= 0 (firing then needs a positive coordinate)


def own_caps(S: ScopedLog, extra=None):
    """cap (D,) fp64: the largest effective threshold eta * tau_j under which every guard key of detector j's
    edit is still a candidate THROUGH detector j ((1 - TIE) * min over the edit's guard keys of r_j; +inf if
    the edit has no guard).  The exact own-key condition for logs with one detector per edit (used by
    repair.scope_cut); for several detectors per edit use keep_guards.  extra = (keys (n, d), edits (n,))."""
    ge, R = S.guard_edit, S.guard_scores()                           # (G, D), every guard key as its own query
    if extra is not None and len(extra[1]):
        ge = torch.cat([ge, torch.as_tensor(extra[1], dtype=torch.long, device=S.device).reshape(-1)])
        R = torch.cat([R, S.key_scores(torch.as_tensor(extra[0]).to(S.device, torch.float64).reshape(-1, S.d))])
    cap = torch.full((S.D,), INF, dtype=torch.float64, device=S.device)
    if ge.shape[0] == 0:
        return cap
    mine = ge[:, None] == S.det_edit[None, :]                        # guard g belongs to detector j's edit
    low = torch.where(mine, R, torch.full_like(R, INF)).min(0).values
    return torch.where(torch.isfinite(low), low - TIE * low.abs(), cap)


def keep_guards(S: ScopedLog, th: torch.Tensor):
    """Lower the proposed effective thresholds th (D,) as little as needed for every guard key to stay a candidate of
    its edit.  A guard that th would silence gets one detector of its edit capped at (1 - TIE) * its coordinate:
    the detector through which it fires under S's CURRENT thresholds if there is one, else the one on which it is
    closest to firing.  (One detector per edit: th_j <- min(th_j, own_caps_j).)
    -> (th', silenced (G,) bool under the literal th, capped (D,) bool)."""
    R, ge = S.guard_scores(), S.guard_edit
    new = th.clone()
    if ge.shape[0] == 0:
        return new, torch.zeros(0, dtype=torch.bool, device=S.device), torch.zeros_like(th, dtype=torch.bool)
    mine = ge[:, None] == S.det_edit[None, :]
    silenced = ~(mine & (R >= th[None, :])).any(1)
    cur = S.theta()
    for g in silenced.nonzero().flatten():
        js = mine[g].nonzero().flatten()
        now = js[R[g, js] >= cur[js]]
        js = now if now.numel() else js
        j = js[(R[g, js] - th[js]).argmax()] if js.numel() > 1 else js[0]
        new[j] = torch.minimum(new[j], R[g, j] - TIE * R[g, j].abs())
    return new, silenced, new < th


def pooled_crosstalk(S: ScopedLog):
    """{r_i(k_j) : j != i} over the primary detectors and their stored keys, as a 1-D numpy array (T (T-1) values)."""
    T = S.T
    R = S.scores(S.det_key[:T])[:, :T]                               # R[j, i] = r_i(k_j)
    off = ~torch.eye(T, dtype=torch.bool, device=R.device)
    return R[off].cpu().numpy(), R


def calibrate_log_only(S: ScopedLog, q: float, allow_silence: bool = False):
    """tau_i = theta_q for every primary detector, theta_q = q-quantile (numpy linear interpolation) of the pooled
    normalised cross-talk of the log's own solver keys on the OTHER edits; q in (0, 1].  eta is reset to 1.
    Further detectors (why-not repairs) keep their effective thresholds."""
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    T = S.T
    pool, R = pooled_crosstalk(S)
    theta = float(np.quantile(pool, q))
    base = S.fold()
    lit = torch.cat([torch.full((T,), theta, dtype=torch.float64, device=S.device), base.det_tau[T:]])
    tau, silenced, capped = keep_guards(base, lit)
    if allow_silence:
        tau = lit
    new = base.replace(det_tau=tau, eta=1.0)
    off = ~torch.eye(T, dtype=torch.bool, device=R.device)
    rec = {"op": "calibrate_log_only", "q": q, "theta": theta, "n_pooled": int(pool.size),
           "n_guards_not_firing_before": int((~base.guard_ok()).sum()),
           "n_own_keys_silenced_by_literal_rule": int(silenced[:T].sum()), "n_guards_silenced_by_literal_rule": int(silenced.sum()),
           "capped": (not allow_silence) and bool(capped.any()), "n_detectors_capped": 0 if allow_silence else int(capped.sum()),
           "n_own_keys_silenced": int((~new.guard_ok()[:T]).sum()),
           "log_cross_firing_pairs": int(((R >= tau[None, :T]) & off).sum()),
           "pool_quantiles": {str(p): float(np.quantile(pool, p)) for p in (0.5, 0.9, 0.99, 0.999, 1.0)}}
    return new, rec


def benign_scores(S: ScopedLog, key_sets):
    """Conformal score of each benign query: max over its eligible keys and over the detectors of r_j / tau_j.
    key_sets: list of (n_q, d) tensors (the keys at the query's eligible positions, e.g. ScopedLog.eligible_keys
    of the captured module input) or one (n, d) tensor of single-key queries.  -> numpy (n_queries,)."""
    if torch.is_tensor(key_sets):
        return S.score(key_sets.to(S.device)).cpu().numpy()
    return np.array([float(S.score(k.to(S.device)).max()) for k in key_sets], dtype=np.float64)


def conformal_rank(n: int, alpha: float) -> int:
    """ceil((1 - alpha)(n + 1)), robust to floating-point noise in the product."""
    return int(math.ceil((1 - alpha) * (n + 1) - 1e-9))


def calibrate_conformal(S: ScopedLog, scores, alpha: float, allow_silence: bool = False):
    """Global multiplier eta from n benign scores (scores_q = max over edits and eligible positions of r / tau
    under the CURRENT thresholds of S; see benign_scores).

    q = the ceil((1 - alpha)(n + 1))-th smallest score (+inf if that rank exceeds n: nothing fires);
    eta = q (1 + 1e-9).  A query fires iff some r_j >= eta tau_j (the package's non-strict rule); the tiny
    inflation makes this imply score > q despite floating-point round-off, i.e. the STRICT conformal event, so
    P(fresh exchangeable benign query fires) <= alpha holds with ties (a fresh query identical to a
    calibration query does not fire at the quantile).  If q <= 0, eta = 1e-200."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if S.eta != 1.0:
        raise ValueError("calibrate on a log with eta = 1 (the scores must be those of the thresholds being scaled)")
    sc = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = int(sc.size)
    if n == 0 or np.isnan(sc).any():
        raise ValueError("need n >= 1 finite or infinite scores, no NaN")
    k = conformal_rank(n, alpha)
    qv = INF if k > n else float(sc[k - 1])
    eta = INF if qv == INF else max(qv * (1 + GUARD), ETA_MIN)
    T = S.T
    lit = S.replace(eta=eta).theta()
    th, silenced, capped = keep_guards(S, lit)
    rec = {"op": "calibrate_conformal", "alpha": alpha, "n": n, "rank": k, "quantile": qv, "eta": eta,
           "eta_infinite": eta == INF, "expected_false_firing_rate_if_no_ties": 0.0 if k > n else 1.0 - k / (n + 1),
           "n_guards_not_firing_before": int((~S.guard_ok()).sum()),
           "n_own_keys_silenced_by_literal_rule": int(silenced[:T].sum()), "n_guards_silenced_by_literal_rule": int(silenced.sum()),
           "allow_silence": allow_silence}
    if allow_silence or not bool(capped.any()):
        new = S.replace(eta=eta)
        rec.update(capped=False, n_detectors_capped=0, guarantee="exact")
    else:
        if eta == INF:                                               # eta cannot carry finite caps: fold them into tau
            new = S.replace(det_tau=th, eta=1.0)
        else:
            new = S.replace(det_tau=torch.where(capped, th / eta, S.det_tau), eta=eta)
        rec.update(capped=True, n_detectors_capped=int(capped.sum()),
                   guarantee=f"void for {int(capped.sum())} capped detector(s): their thresholds are below the conformal level")
    rec["n_own_keys_silenced"] = int((~new.guard_ok()[:T]).sum())
    return new, rec


# ============================================================================= exact split-conformal rules for ANY per-detector feature
# (PROTOCOL.md section 6, amendment of 2026-11-10 17:10; referee of theory part C, verdicts C1 and C6)
#
# FEATURE.  A scoped log S maps a key k to one number per detector, r_j(k) = S.scores(k)[j]: the normalised solver
#   coordinate (S = ScopedLog.from_log / baselines.feature_gate(log, 'ncoord')), the raw solver coordinate ('coord'), the raw
#   cosine to the stored solver key ('cos') or minus the Euclidean distance to it ('negdist').  The rules below only use
#   these numbers, so the same calibration applies to ReScope and to the post-hoc gate baselines.
#   Q[q, j] = the largest r_j over the eligible keys of benign query q (S.query_max: one row per query).
#
# TWO-BLOCK RULE (calibrate_two_block; the default calibration of ReScope).
#   1. the n benign calibration queries are split at random (seeded) into block 1 (n1 = n // 2 by default) and block 2 (n2 = n - n1);
#   2. baseline          b_j = max over the block-1 queries of Q[q, j]
#   3. score of a query  s(q) = max over detectors j of (Q[q, j] - b_j) / u_j        u_j = 1 (option: any fixed positive vector)
#   4. eta = the k-th smallest block-2 score, k = ceil((1 - alpha)(n2 + 1)) in exact integer arithmetic; eta = +inf if k > n2
#   5. detector j is a candidate at a key k  iff  (r_j(k) - b_j) / u_j > eta           (STRICT)
#   6. fire 'top1': the candidate with the largest excess (r_j - b_j - eta u_j) / u_j releases, i.e. the largest margin
#      q_j = (r_j - b_j) / u_j (the two differ by the constant eta); an exact tie goes to the lowest detector index.
#      The gate ranks by q_j: it is the same order, it is computed without eta (the literal excess rounds eta * u_j and can
#      break an exact tie of two margins either way), and it still orders the candidates when eta = +inf (then only capped
#      detectors can be candidates and every literal excess is -inf).
#      Option rank='r': the candidate with the largest r_j (the package's default ranking).
#   Guarantee (uncapped rule): for benign queries exchangeable with the calibration queries, and whatever n1 >= 1,
#       P(a fresh benign query has a candidate) <= floor(alpha (n2 + 1)) / (n2 + 1) <= alpha,
#   marginally over the calibration draw and the split, with ties (the score function depends on block 1 only, so the
#   block-2 scores and the fresh score are exchangeable; referee C6, fix (1)).
#
# SHARED-SCALE RULE (calibrate_shared; ablation, and the calibration of gates with one global threshold): the same with NO
#   block 1: b = 0, every query is a block-2 query, s(q) = max_j Q[q, j] / u_j, candidate iff r_j / u_j > eta.
#   u = 1: one threshold eta on the feature itself (Section 0 of the protocol for the normalised coordinate; a conformal
#   cosine threshold / a conformal radius for 'cos' / 'negdist'); u = reference thresholds tau0 > 0: the scaled rule.
#
# FLOATING-POINT-SAFE STORAGE (referee C1).  In fp64 the DEFINING comparison is  fl(fl(r - b_j) / u_j) > eta  with eta itself one
#   of the block-2 scores computed by the same two operations.  The stored threshold is NOT the rounded b_j + eta u_j (for which
#   the query that defines eta can fire): it is  tau_j = succ(theta_j),  theta_j = the LARGEST fp64 number r with
#   fl(fl(r - b_j) / u_j) <= eta.  Subtracting a constant and dividing by a positive constant are monotone under correct
#   rounding, so  r >= tau_j  <=>  fl(fl(r - b_j) / u_j) > eta  for every finite r, bit for bit; the scoped log keeps eta = 1
#   and det_tau = tau, and its ordinary non-strict gate r >= tau IS the strict conformal event.  theta_j is found by bisection
#   over the fp64 lattice (float_safe_threshold), not by a guessed number of nextafter steps.
#
# GUARDS.  By default an edit's guard keys (its stored solver key and the keys added with repair.add_guards, e.g. the B = 1
#   keys of its own prompts: Evaluator.own_keys) stay candidates: a detector that the literal rule would silence on a guard is
#   capped just below its smallest guard coordinate (keep_guards).  The record counts the edits that needed the cap; for them
#   the guarantee is void.  allow_silence=True returns the uncapped (literal) rule.
from fractions import Fraction  # noqa: E402

_IMIN = np.int64(-2 ** 63)


def as_fraction(alpha) -> Fraction:
    """alpha as an exact fraction: a Fraction, a string ('1/20', '0.05'), a pair (num, den), or a float, which is read as the
    closest fraction with a denominator <= 10**9 (0.05 -> 1/20, 0.3 -> 3/10, 1/3 -> 1/3; the binary expansion of the float,
    which lies slightly off the intended decimal, is not used)."""
    if isinstance(alpha, Fraction):
        f = alpha
    elif isinstance(alpha, (tuple, list)):
        f = Fraction(int(alpha[0]), int(alpha[1]))
    elif isinstance(alpha, str):
        f = Fraction(alpha)
    else:
        f = Fraction(float(alpha)).limit_denominator(10 ** 9)
    if not 0 < f < 1:
        raise ValueError("alpha must be in (0, 1)")
    return f


def conformal_rank_exact(n: int, alpha) -> int:
    """k = ceil((1 - alpha)(n + 1)) in exact integer arithmetic (alpha: see as_fraction).  The float product can be one rank
    too high (alpha = 1/3, n = 8: 7 instead of 6; alpha = 0.3, n = 9: 8 instead of 7)."""
    a = as_fraction(alpha)
    return -((-(a.denominator - a.numerator) * (int(n) + 1)) // a.denominator)


def _ford(x):
    """fp64 -> int64 ordinal, monotone on the non-NaN floats (-0.0 and +0.0 share ordinal 0)."""
    b = np.ascontiguousarray(x, dtype=np.float64).view(np.int64)
    return np.where(b < 0, _IMIN - b, b)


def _funord(o):
    o = np.ascontiguousarray(o, dtype=np.int64)
    return np.ascontiguousarray(np.where(o < 0, _IMIN - o, o), dtype=np.int64).view(np.float64)


def float_safe_threshold(b, u, eta) -> np.ndarray:
    """tau (D,) fp64 such that for EVERY finite fp64 number r and every detector j

        r >= tau_j      <=>      fl(fl(r - b_j) / u_j) > eta          (the defining, strict comparison)

    tau_j = succ(theta_j), theta_j = the largest fp64 number r with fl(fl(r - b_j) / u_j) <= eta, found by an exponential
    bracket around the rounded b_j + eta u_j followed by bisection over the fp64 lattice (the predicate is monotone in r
    because both operations are correctly rounded with a constant second operand, u_j > 0).  b: baselines (D,) (0 for the shared rule), u: scales (D,) > 0, eta: scalar (+inf: nothing can
    be a candidate, tau = +inf; b_j = -inf: every finite r is a candidate, tau_j = the most negative finite number)."""
    b = np.atleast_1d(np.asarray(b, dtype=np.float64))
    u = np.broadcast_to(np.asarray(u, dtype=np.float64), b.shape)
    eta = float(eta)
    if np.isnan(b).any() or np.isnan(eta) or not bool(((u > 0) & np.isfinite(u)).all()):
        raise ValueError("float_safe_threshold needs baselines without NaN, finite scales > 0 and eta without NaN")
    if eta == np.inf:
        return np.full(b.shape, np.inf)

    def below(r):                                                     # the defining comparison is FALSE at r
        with np.errstate(invalid="ignore", over="ignore"):
            return (r - b) / u <= eta                                 # NaN (inf - inf) compares False
    FLO, FHI = int(_ford(-np.inf).ravel()[0]), int(_ford(np.inf).ravel()[0])
    one = np.uint64(1)
    # 1. bracket: start at the rounded b + eta u (the boundary is nearly always within a few ulps of it) and gallop outwards
    with np.errstate(invalid="ignore", over="ignore"):
        g0 = b + eta * u
    g = _ford(np.where(np.isnan(g0), 0.0, g0)).astype(np.int64)
    ok = below(_funord(g))
    lo = np.where(ok, g, FLO).astype(np.int64)                        # invariant: below(lo) (or lo is the floor -inf), not below(hi)
    hi = np.where(ok, FHI, g).astype(np.int64)
    gu = g.astype(np.uint64)                                          # (differences of ordinals are exact modulo 2^64)
    room_up, room_dn = np.uint64(FHI) - gu, gu - np.array(FLO, dtype=np.int64).astype(np.uint64)
    opn, step = np.ones(b.shape, dtype=bool), 1
    while bool(opn.any()):
        st = np.uint64(step)
        c = np.where(ok, (gu + np.minimum(st, room_up)).astype(np.int64), (gu - np.minimum(st, room_dn)).astype(np.int64))
        pc = below(_funord(c))
        lo, hi = np.where(opn & pc, c, lo), np.where(opn & ~pc, c, hi)
        opn = opn & np.where(ok, pc, ~pc) & ~np.where(ok, c == FHI, c == FLO)
        if step >= 2 ** 63:
            break
        step *= 2
    # 2. bisection over the fp64 lattice inside the bracket
    while True:
        diff = hi.astype(np.uint64) - lo.astype(np.uint64)            # exact: hi >= lo
        if not bool((diff > one).any()):
            break
        mid = (lo.astype(np.uint64) + (diff >> one)).astype(np.int64)
        okm = below(_funord(mid))
        lo, hi = np.where(okm, mid, lo), np.where(okm, hi, mid)
    with np.errstate(over="ignore"):
        return np.nextafter(_funord(lo), np.inf)                    # succ(+inf) = +inf, succ(-inf) = the most negative finite number


def block_scores(coords, b=None, u=None) -> np.ndarray:
    """(n,) conformal scores of n queries from their per-detector maxima coords (n, D): max_j fl(fl(Q - b_j) / u_j)
    (b=None: 0, u=None: 1; subtracting 0 and dividing by 1 are exact, so the shared rule's score is max_j Q itself)."""
    M = np.asarray(coords, dtype=np.float64)
    if M.ndim != 2 or M.shape[1] == 0:
        raise ValueError("coords must be an (n, D) matrix with D >= 1")
    with np.errstate(invalid="ignore", over="ignore"):
        if b is not None:
            M = M - np.asarray(b, dtype=np.float64)[None, :]
        if u is not None:
            M = M / np.asarray(u, dtype=np.float64)[None, :]
    sc = M.max(1) if M.shape[0] else np.zeros(0)
    if np.isnan(sc).any():
        raise ValueError("NaN score (NaN feature, or inf - inf)")
    return sc


def split_blocks(n: int, split=(1, 1), seed: int = 0):
    """(idx1, idx2): seeded random split of range(n) into block 1 with n1 = (n * a) // (a + b) queries, split = (a, b)
    ((1, 1): halves, n1 = n // 2; (1, 3) / (3, 1): the block ratios of the ablation), and block 2 with the rest; both sorted."""
    a, c = int(split[0]), int(split[1])
    if a < 1 or c < 1:
        raise ValueError("split = (a, b) with integers a, b >= 1")
    n1 = (int(n) * a) // (a + c)
    if n1 < 1 or n - n1 < 1:
        raise ValueError(f"cannot split {n} queries {a}:{c} into two non-empty blocks")
    perm = np.random.default_rng(seed).permutation(int(n))
    return np.sort(perm[:n1]), np.sort(perm[n1:])


def _unit(u, D):
    if u is None:
        return None
    u = np.broadcast_to(np.asarray(u, dtype=np.float64), (D,)).copy()
    if not bool(((u > 0) & np.isfinite(u)).all()):
        raise ValueError("the scales u must be finite and > 0")
    return u


def two_block_rule(coords, alpha, split=(1, 1), seed: int = 0, blocks=None, u=None) -> dict:
    """The two-block rule on a matrix of per-query maxima coords (n, D) (numbers only; no scoped log).
    blocks = (idx1, idx2) overrides the seeded split.  -> dict(b, u, eta, tau, rank, n1, n2, idx1, idx2, n_allowed, bound):
    b (D,) baselines, eta the k-th smallest block-2 score (+inf if k > n2), tau (D,) the floating-point-safe thresholds
    (r >= tau_j <=> (r - b_j) / u_j > eta), n_allowed = floor(alpha (n2 + 1)) = n2 + 1 - k (0 if eta = +inf),
    bound = n_allowed / (n2 + 1) = the guaranteed false-firing rate."""
    Q = np.asarray(coords, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError("coords must be an (n, D) matrix")
    n, D = Q.shape
    idx1, idx2 = split_blocks(n, split, seed) if blocks is None else (np.asarray(blocks[0], dtype=np.int64), np.asarray(blocks[1], dtype=np.int64))
    if len(idx1) < 1 or len(idx2) < 1 or len(np.intersect1d(idx1, idx2)):
        raise ValueError("the two blocks must be non-empty and disjoint")
    u = _unit(u, D)
    b = Q[idx1].max(0)
    if np.isnan(b).any():
        raise ValueError("NaN feature in block 1")
    return _finish(block_scores(Q[idx2], b, u), alpha, b, u, D, idx1, idx2)


def shared_rule(coords, alpha, u=None) -> dict:
    """The shared-scale rule on coords (n, D): no block 1 (b = 0), every query calibrates eta.  Same output as two_block_rule
    (b = zeros, n1 = 0)."""
    Q = np.asarray(coords, dtype=np.float64)
    if Q.ndim != 2 or Q.shape[0] < 1:
        raise ValueError("coords must be an (n, D) matrix with n >= 1")
    n, D = Q.shape
    u = _unit(u, D)
    return _finish(block_scores(Q, None, u), alpha, None, u, D, np.zeros(0, dtype=np.int64), np.arange(n))


def _finish(sc, alpha, b, u, D, idx1, idx2):
    a = as_fraction(alpha)
    n2 = int(sc.size)
    k = conformal_rank_exact(n2, a)
    eta = float("inf") if k > n2 else float(np.partition(sc, k - 1)[k - 1])          # the k-th smallest block-2 score
    b0 = np.zeros(D) if b is None else b
    tau = float_safe_threshold(b0, np.ones(D) if u is None else u, eta)
    allowed = 0 if k > n2 else n2 + 1 - k
    return {"b": b0, "has_baseline": b is not None, "u": u, "eta": eta, "tau": tau, "rank": k, "n1": int(len(idx1)), "n2": n2,
            "idx1": idx1, "idx2": idx2, "alpha": a, "n_allowed": allowed, "bound": allowed / (n2 + 1), "scores2": sc}


def _query_coords(S: ScopedLog, queries, coords):
    """(n, D) numpy matrix of per-query maxima from `queries` (a list of per-query key tensors, or one (n, d) tensor = n
    one-key queries) or from `coords` (already S.query_max of the queries)."""
    if (queries is None) == (coords is None):
        raise ValueError("give exactly one of `queries` (keys) and `coords` (their per-detector maxima, S.query_max)")
    if coords is not None:
        C = torch.as_tensor(coords).to(torch.float64).reshape(-1, S.D)
    else:
        qs = [k[None] for k in queries.reshape(-1, S.d)] if torch.is_tensor(queries) else list(queries)
        C = S.query_max(qs)
    return C.detach().cpu().numpy()


def _install(S: ScopedLog, rule: dict, op: str, rank: str, allow_silence: bool, extra: dict):
    if rank not in ("excess", "r"):
        raise ValueError("rank: 'excess' (largest margin (r - b) / u) | 'r' (largest r)")
    T, dev = S.T, S.device
    t64 = lambda v: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float64)).to(dev)          # noqa: E731
    lit = t64(rule["tau"])
    th, silenced, capped = keep_guards(S, lit)
    if allow_silence:
        th, capped = lit, torch.zeros_like(capped)
    eta, a = rule["eta"], rule["alpha"]
    kw = {"det_b": None, "det_u": None}
    if rank == "excess":
        kw = {"det_b": t64(rule["b"]) if rule["has_baseline"] else None, "det_u": None if rule["u"] is None else t64(rule["u"])}
    cal = {"op": op, "alpha": f"{a.numerator}/{a.denominator}", "eta": eta, "n1": rule["n1"], "n2": rule["n2"], "rank": rule["rank"],
           "top1_rank": rank, "capped": bool(capped.any())}
    new = S.replace(det_tau=th, eta=1.0, meta=dict(S.meta, calibration=cal, rank_eta=eta), **kw)
    ok = new.guard_ok()
    cap_edits = sorted({int(e) for e in S.det_edit[capped]})
    sil_edits = sorted({int(e) for e in S.guard_edit[silenced]})
    # information: on its own guard keys, is the guard's edit the candidate that releases under fire 'top1'?
    Rg = new.guard_scores()
    cand = Rg >= new.theta()[None, :]
    top = new.det_edit[torch.where(cand, new.rank_key(Rg), torch.full_like(Rg, -INF)).argmax(1)] if Rg.shape[0] else new.guard_edit
    own_top = cand.any(1) & (top == new.guard_edit) if Rg.shape[0] else torch.zeros(0, dtype=torch.bool)
    fin = lambda v: [float(x) for x in np.quantile(v[np.isfinite(v)], [0.0, 0.5, 1.0])] if np.isfinite(v).any() else None   # noqa: E731
    rec = {"op": op, "alpha": float(a), "alpha_exact": f"{a.numerator}/{a.denominator}", "n": rule["n1"] + rule["n2"], "n1": rule["n1"],
           "n2": rule["n2"], "rank": rule["rank"], "eta": eta, "eta_infinite": eta == INF, "strict": True,
           "n_allowed_above": rule["n_allowed"], "bound_false_firing": rule["bound"],
           "u": "1" if rule["u"] is None else "custom", "top1_rank": rank,
           "block1": [int(i) for i in rule["idx1"]], "block2": [int(i) for i in rule["idx2"]],
           "n_guards_not_firing_before": int((~S.guard_ok()).sum()),
           "n_own_keys_silenced_by_literal_rule": int(silenced[:T].sum()), "n_guards_silenced_by_literal_rule": int(silenced.sum()),
           "n_edits_silenced_by_literal_rule": len(sil_edits), "allow_silence": allow_silence,
           "capped": bool(capped.any()), "n_detectors_capped": int(capped.sum()), "n_edits_capped": len(cap_edits), "capped_edits": cap_edits,
           "guarantee": "exact" if not bool(capped.any()) else
                        f"void for {len(cap_edits)} capped edit(s): their thresholds are below the conformal level",
           "n_own_keys_silenced": int((~ok[:T]).sum()), "n_guards_silenced": int((~ok).sum()), "n_guards": int(ok.numel()),
           "n_guards_released_by_own_edit_top1": int(own_top.sum()),
           "baseline_min_median_max": fin(rule["b"]) if rule["has_baseline"] else None, "threshold_min_median_max": fin(th.cpu().numpy())}
    rec.update(extra)
    return new, rec


def calibrate_two_block(S: ScopedLog, queries=None, alpha=0.05, *, coords=None, split=(1, 1), seed: int = 0, blocks=None, u=None,
                        rank: str = "excess", allow_silence: bool = False):
    """Two-block (split) per-detector conformal calibration: the DEFAULT calibration of ReScope.  Definition: the comment
    block above (baselines from block 1, eta from block 2, strict gate, floating-point-safe thresholds).

    S        any scoped log (its detectors define the feature; current thresholds and eta are ignored and replaced);
    queries  the n benign calibration queries: a list of per-query key tensors (the keys at the query's eligible positions,
             Evaluator.eligible_keys / KeyCache.keys) or one (n, d) tensor of one-key queries; or
    coords   their (n, D) matrix of per-detector maxima (S.query_max(queries)): cheaper for many draws from one pool;
    alpha    miscoverage (float / Fraction / 'p/q'; the rank is computed in exact integer arithmetic);
    split    block ratio (a, b): n1 = (n a) // (a + b); seed: seed of the random split; blocks = (idx1, idx2): explicit blocks;
    u        None (u_j = 1) or a fixed positive (D,) vector (it must not depend on the calibration queries);
    rank     'excess': fire 'top1' releases the candidate with the largest margin (r_j - b_j) / u_j (stored in det_b / det_u);
             'r': the candidate with the largest r_j;
    allow_silence  False: guards keep firing (capped thresholds, counted in the record); True: the uncapped rule.
    -> (new scoped log with eta = 1 and det_tau = the floating-point-safe thresholds, record)."""
    Q = _query_coords(S, queries, coords)
    rule = two_block_rule(Q, alpha, split, seed, blocks, u)
    return _install(S, rule, "calibrate_two_block", rank, allow_silence,
                    {"split": None if blocks is not None else [int(split[0]), int(split[1])], "seed": None if blocks is not None else int(seed)})


def calibrate_shared(S: ScopedLog, queries=None, alpha=0.05, *, coords=None, u=None, rank: str = "r", allow_silence: bool = False):
    """Shared-scale conformal calibration in the exact form (strict gate, exact rank, floating-point-safe thresholds): one
    eta for all detectors, candidate iff r_j / u_j > eta.  u=None: u_j = 1 (one threshold on the feature); u='tau': the
    current effective thresholds of S as reference scales (they must be finite and > 0); or a positive (D,) vector.
    Arguments and output as calibrate_two_block (no block 1: every query calibrates eta, guarantee floor(alpha (n + 1)) / (n + 1)).
    calibrate_conformal is the earlier form of this rule (a multiplier eta with a 1e-9 guard); it is kept unchanged."""
    if isinstance(u, str):
        if u != "tau":
            raise ValueError("u: None | 'tau' | a positive (D,) vector")
        u = S.theta().detach().cpu().numpy()
    Q = _query_coords(S, queries, coords)
    rule = shared_rule(Q, alpha, u)
    return _install(S, rule, "calibrate_shared", rank, allow_silence, {})


def calibrate_dense(gate, scores, alpha=0.05, two_block: bool = False, split=(1, 1), seed: int = 0):
    """The same conformal rules for a gate with ONE scalar score per query (baselines.DenseGate: router = activation at the
    last prompt position, zscore = largest z-score of the sequence).  scores (n,): the benign calibration queries' scores.
    two_block=False: threshold = succ(eta), eta = the exact-rank quantile of the scores (the gate's '>=' is then the strict
    event score > eta); two_block=True: baseline b = largest block-1 score, eta from the block-2 margins, threshold tau with
    score >= tau <=> score - b > eta.  -> (copy of the gate with the new threshold, record)."""
    import copy
    sc = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    rule = two_block_rule(sc, alpha, split, seed) if two_block else shared_rule(sc, alpha)
    new = copy.copy(gate)
    new.thr = float(rule["tau"][0])
    new.name = f"{getattr(gate, 'kind', 'dense')}-conformal"
    a = rule["alpha"]
    return new, {"op": "calibrate_dense", "two_block": bool(two_block), "alpha": float(a), "alpha_exact": f"{a.numerator}/{a.denominator}",
                 "n": rule["n1"] + rule["n2"], "n1": rule["n1"], "n2": rule["n2"], "rank": rule["rank"], "eta": rule["eta"],
                 "eta_infinite": rule["eta"] == INF, "baseline": float(rule["b"][0]) if rule["has_baseline"] else None,
                 "threshold": new.thr, "bound_false_firing": rule["bound"], "strict": True}
