"""fx.stats: the statistics of PROTOCOL.md section 4.

  bootstrap(values, units)            percentile bootstrap of a mean (2000 resamples); the UNIT is resampled with its rows
                                      (fx.metrics: unit = query cluster (query_id) for probe metrics, edit for edit metrics)
  bootstrap_ratio(num, den, units)    the same for a ratio of sums (e.g. D1 among firing rows)
  bootstrap_diff(a, b, units)         paired bootstrap of mean(a) - mean(b) on the same rows (per-row differences, units resampled)
  mcnemar_exact(a, b)                 exact two-sided McNemar test of two binary outcomes on the same rows
  over_draws(values)                  summary over calibration draws: mean, 5th and 95th percentile (numpy, linear), min, max, sd
All functions are deterministic for a given seed; values may contain None (dropped, counted).
"""
from __future__ import annotations

import math

import numpy as np

N_BOOT = 2000
CI = 95.0


def _prep(values, units):
    v = np.array([np.nan if x is None else float(x) for x in values], dtype=np.float64)
    u = np.arange(v.size) if units is None else np.asarray(list(units))
    if u.shape[0] != v.size:
        raise ValueError("one unit per value")
    keep = ~np.isnan(v)
    return v[keep], u[keep], int((~keep).sum())


def _groups(u):
    _, inv = np.unique(u, return_inverse=True)
    return inv, int(inv.max()) + 1 if inv.size else 0


def _resample(n_units, n_boot, seed):
    return np.random.default_rng(seed).integers(0, n_units, size=(n_boot, n_units))


def _pct(x, ci):
    lo = (100.0 - ci) / 2.0
    return float(np.percentile(x, lo)), float(np.percentile(x, 100.0 - lo))


def bootstrap_ratio(num, den, units=None, n_boot: int = N_BOOT, seed: int = 0, ci: float = CI, scale: float = 1.0) -> dict:
    """Percentile bootstrap of scale * sum(num) / sum(den) with whole units resampled (rows of a unit stay together).
    -> {value, ci_lo, ci_hi, n_units, n_rows, n_boot, seed, n_undefined}; resamples with a zero denominator are dropped (counted)."""
    a, u, dropped = _prep(num, units)
    b = np.asarray([float(x) for x, y in zip(den, num) if y is not None], dtype=np.float64) if not np.isscalar(den) else np.full(a.size, float(den))
    if a.size == 0 or b.sum() == 0:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n_units": 0, "n_rows": int(a.size), "n_boot": n_boot, "seed": seed, "n_dropped": dropped}
    inv, nu = _groups(u)
    sa, sb = np.bincount(inv, weights=a, minlength=nu), np.bincount(inv, weights=b, minlength=nu)
    idx = _resample(nu, n_boot, seed)
    A, B = sa[idx].sum(1), sb[idx].sum(1)
    ok = B > 0
    lo, hi = _pct(scale * A[ok] / B[ok], ci)
    return {"value": float(scale * a.sum() / b.sum()), "ci_lo": lo, "ci_hi": hi, "n_units": nu, "n_rows": int(a.size), "n_boot": n_boot, "seed": seed,
            "n_dropped": dropped, "n_resamples_undefined": int((~ok).sum())}


def bootstrap(values, units=None, n_boot: int = N_BOOT, seed: int = 0, ci: float = CI, scale: float = 1.0) -> dict:
    """Percentile bootstrap of scale * mean(values) over rows, whole units resampled (units=None: every row is its own unit).
    Binary flags with scale=100 give a rate in percent."""
    v = [None if x is None else float(x) for x in values]
    return bootstrap_ratio(v, [1.0] * len(v), units, n_boot, seed, ci, scale)


def bootstrap_diff(a, b, units=None, n_boot: int = N_BOOT, seed: int = 0, ci: float = CI, scale: float = 1.0) -> dict:
    """Paired bootstrap of scale * (mean(a) - mean(b)) for two outcomes on the SAME rows (row i of a pairs with row i of b), whole
    units resampled.  -> {value, ci_lo, ci_hi, p_two_sided (share of resamples on the other side of 0, doubled, >= 1 / n_boot),
    n_units, n_rows, mean_a, mean_b}."""
    if len(a) != len(b):
        raise ValueError("paired outcomes need the same rows")
    pairs = [(None, None) if (x is None or y is None) else (float(x), float(y)) for x, y in zip(a, b)]
    d = [None if x is None else x - y for x, y in pairs]
    out = bootstrap(d, units, n_boot, seed, ci, scale)
    if out["value"] is None:
        return out
    dv, u, _ = _prep(d, units)
    inv, nu = _groups(u)
    sd, sc = np.bincount(inv, weights=dv, minlength=nu), np.bincount(inv, minlength=nu).astype(np.float64)
    idx = _resample(nu, n_boot, seed)
    D = sd[idx].sum(1) / sc[idx].sum(1)
    p = 2.0 * min(float((D <= 0).mean()), float((D >= 0).mean()))
    xa = np.array([x for x, _ in pairs if x is not None])
    xb = np.array([y for _, y in pairs if y is not None])
    out.update(p_two_sided=float(min(1.0, max(p, 1.0 / n_boot))), mean_a=float(scale * xa.mean()), mean_b=float(scale * xb.mean()))
    return out


def mcnemar_exact(a, b) -> dict:
    """Exact (binomial) two-sided McNemar test of two binary outcomes on the same rows: n10 = rows with a and not b, n01 = rows with
    b and not a; p = min(1, 2 P(Bin(n10 + n01, 1/2) <= min(n10, n01))).  -> {n10, n01, n_discordant, n, p}."""
    if len(a) != len(b):
        raise ValueError("paired outcomes need the same rows")
    pairs = [(bool(x), bool(y)) for x, y in zip(a, b) if x is not None and y is not None]
    n10 = sum(x and not y for x, y in pairs)
    n01 = sum(y and not x for x, y in pairs)
    n, k = n10 + n01, min(n10, n01)
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)
    return {"n10": n10, "n01": n01, "n_discordant": n, "n": len(pairs), "p": float(p)}


def over_draws(values) -> dict:
    """Summary of one statistic over calibration draws: mean, 5th and 95th percentile (numpy, linear interpolation), min, max, sd."""
    v = np.asarray([float(x) for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n_draws": 0, "mean": None, "p05": None, "p95": None, "min": None, "max": None, "sd": None}
    return {"n_draws": int(v.size), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}


# ----------------------------------------------------------------------------- R10: crossed bootstrap over calibration draws and units; Holm
def _counts(n: int, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n) multiplicities of a resample of n items with replacement."""
    idx = rng.integers(0, n, size=(n_boot, n))
    c = np.zeros((n_boot, n), dtype=np.float64)
    np.add.at(c, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return c


def _grid(num, den):
    N = np.asarray(num, dtype=np.float64)
    if N.ndim != 2 or N.shape[0] < 1 or N.shape[1] < 1:
        raise ValueError("a (draws, units) matrix is needed")
    Dn = np.ones_like(N) if den is None else np.broadcast_to(np.asarray(den, dtype=np.float64), N.shape).copy()
    if np.isnan(N).any() or np.isnan(Dn).any():
        raise ValueError("the crossed bootstrap needs a complete grid: every unit under every draw (no None / NaN)")
    return N, Dn


def crossed_bootstrap(num, den=None, n_boot: int = N_BOOT, seed: int = 0, ci: float = CI, scale: float = 1.0) -> dict:
    """Crossed percentile bootstrap over calibration draws AND units (amendment R10): num[d, u] = the value of unit u under
    calibration draw d (a per-edit flag, or the sum of the row flags of a query cluster); in every resample the draws and the
    units are resampled independently with replacement and the statistic is recomputed on the resampled grid:
        scale * sum(num[d*, u*]) / sum(den[d*, u*])        den=None: 1 per cell (the mean over the grid);
                                                           den (units,) or (draws, units): rows per cell (rate of a clustered pool).
    Units: query clusters (query_id) for probe metrics, edits for edit metrics (fx.metrics.grid builds the matrices).
    -> {value, ci_lo, ci_hi, se, n_draws, n_units, n_boot, seed, n_resamples_undefined}.  With one draw this is the unit bootstrap.
    The draw component is estimated from n_draws values: with 3 to 5 behavioural draws it is known to be too small by about
    (n_draws - 1) / n_draws in variance (state the number of draws next to the interval)."""
    N, Dn = _grid(num, den)
    nd, nu = N.shape
    cd = _counts(nd, n_boot, np.random.default_rng([int(seed), 101]))
    cu = _counts(nu, n_boot, np.random.default_rng([int(seed), 102]))
    A, B = ((cd @ N) * cu).sum(1), ((cd @ Dn) * cu).sum(1)
    ok = B > 0
    if Dn.sum() == 0 or not ok.any():
        return {"value": None, "ci_lo": None, "ci_hi": None, "se": None, "n_draws": nd, "n_units": nu, "n_boot": n_boot, "seed": seed, "n_resamples_undefined": int((~ok).sum())}
    st = scale * A[ok] / B[ok]
    lo, hi = _pct(st, ci)
    return {"value": float(scale * N.sum() / Dn.sum()), "ci_lo": lo, "ci_hi": hi, "se": float(st.std(ddof=1)), "n_draws": nd, "n_units": nu, "n_boot": n_boot,
            "seed": seed, "n_resamples_undefined": int((~ok).sum())}


def crossed_bootstrap_diff(num_a, num_b, den_a=None, den_b=None, n_boot: int = N_BOOT, seed: int = 0, ci: float = CI, scale: float = 1.0) -> dict:
    """PAIRED crossed bootstrap of the difference of two methods on the SAME calibration draws and units: the same resampled
    draws and units are applied to both grids; statistic = scale * (sum a / sum den_a - sum b / sum den_b).
    -> {value, ci_lo, ci_hi, se, p_two_sided (share of resamples on the other side of 0, doubled, >= 1 / n_boot), mean_a, mean_b,
    n_draws, n_units, ...}.  The grids must have the same shape (draw d and unit u mean the same in both)."""
    Na, Da = _grid(num_a, den_a)
    Nb, Db = _grid(num_b, den_b)
    if Na.shape != Nb.shape:
        raise ValueError("paired grids need the same draws and units")
    nd, nu = Na.shape
    cd = _counts(nd, n_boot, np.random.default_rng([int(seed), 101]))
    cu = _counts(nu, n_boot, np.random.default_rng([int(seed), 102]))
    f = lambda N: ((cd @ N) * cu).sum(1)                                 # noqa: E731
    Ba, Bb = f(Da), f(Db)
    ok = (Ba > 0) & (Bb > 0)
    if Da.sum() == 0 or Db.sum() == 0 or not ok.any():
        return {"value": None, "ci_lo": None, "ci_hi": None, "n_draws": nd, "n_units": nu, "n_boot": n_boot, "seed": seed}
    st = scale * (f(Na)[ok] / Ba[ok] - f(Nb)[ok] / Bb[ok])
    lo, hi = _pct(st, ci)
    p = 2.0 * min(float((st <= 0).mean()), float((st >= 0).mean()))
    ma, mb = float(scale * Na.sum() / Da.sum()), float(scale * Nb.sum() / Db.sum())
    return {"value": ma - mb, "ci_lo": lo, "ci_hi": hi, "se": float(st.std(ddof=1)), "p_two_sided": float(min(1.0, max(p, 1.0 / n_boot))), "mean_a": ma, "mean_b": mb,
            "n_draws": nd, "n_units": nu, "n_boot": n_boot, "seed": seed, "n_resamples_undefined": int((~ok).sum())}


def holm(pvalues, alpha: float = 0.05) -> dict:
    """Holm's step-down correction of a family of p-values (R6: within H1, H2, H3 over backbones, baselines and pairs).
    pvalues: a list, or {name: p}.  -> {adjusted, reject, order, m, alpha} (lists / dicts shaped like the input):
    adjusted p of the i-th smallest p = max over j <= i of min(1, (m - j + 1) p_(j)); reject iff adjusted <= alpha."""
    named = isinstance(pvalues, dict)
    keys = list(pvalues) if named else list(range(len(pvalues)))
    p = np.array([float(pvalues[k]) for k in keys], dtype=np.float64)
    if p.size == 0:
        return {"adjusted": {} if named else [], "reject": {} if named else [], "order": [], "m": 0, "alpha": alpha}
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    rej = adj <= alpha
    if named:
        return {"adjusted": {k: float(adj[i]) for i, k in enumerate(keys)}, "reject": {k: bool(rej[i]) for i, k in enumerate(keys)},
                "order": [keys[int(i)] for i in order], "m": m, "alpha": alpha}
    return {"adjusted": [float(x) for x in adj], "reject": [bool(x) for x in rej], "order": [int(i) for i in order], "m": m, "alpha": alpha}
