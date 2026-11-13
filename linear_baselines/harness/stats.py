"""Uncertainty and paired tests used by every RQ.

bootstrap_ci : percentile CI of a mean, resampling at the UID level (cluster bootstrap: a probe
               item's image and text rows, or repeated rows of one uid, move together).
mcnemar      : exact two-sided McNemar test (binomial on the discordant pairs) for paired binary
               outcomes of two methods on the same complaints.
"""
from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(values, uids=None, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    uids = list(range(len(v))) if uids is None else list(uids)
    groups = {}
    for x, u in zip(v, uids):
        groups.setdefault(u, []).append(x)
    keys = sorted(groups, key=str)
    sums = np.array([sum(groups[k]) for k in keys])
    cnts = np.array([len(groups[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    bm = sums[idx].sum(1) / cnts[idx].sum(1)
    return {"n": int(v.size), "n_uid": len(keys), "mean": float(v.mean()),
            "lo": float(np.quantile(bm, alpha / 2)), "hi": float(np.quantile(bm, 1 - alpha / 2))}


def mcnemar(a, b):
    """a, b: paired booleans. Returns counts and the exact two-sided p-value."""
    a = [bool(x) for x in a]
    b = [bool(x) for x in b]
    n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b))
    n = n10 + n01
    if n == 0:
        p = 1.0
    else:
        k = min(n10, n01)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n": len(a), "a_only": n10, "b_only": n01, "both": sum(x and y for x, y in zip(a, b)),
            "neither": sum((not x) and (not y) for x, y in zip(a, b)), "p_exact": p}
