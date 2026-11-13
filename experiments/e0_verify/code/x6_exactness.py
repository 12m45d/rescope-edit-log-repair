"""Extra check X6 (CPU, synthetic): exact rank and floating-point-safe thresholds of the library against the definitions.

  python x6_exactness.py        -> out/x6_exactness.json

  A  conformal_rank_exact(n, alpha) == ceil((1 - alpha)(n + 1)) in exact rationals, n = 1..2000, alpha given as a FLOAT
     (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 1/3, 0.15, 0.025) and as a string
  B  float_safe_threshold(b, u, eta): for random b, u > 0, eta and for every r among tau, its 3 fp64 neighbours on either side and
     the rounded b + eta u with its neighbours:   r >= tau   <=>   fl(fl(r - b) / u) > eta     (also u = 1, eta = +inf, eta < 0)
  C  two_block_rule on random matrices with heavy ties (rounded features) and a custom u: eta == the k-th smallest block-2 score
     of the naive rule; decisions on the calibration rows themselves and on fresh rows == the strict defining comparison
"""
import math
import sys
from fractions import Fraction

import numpy as np

import vcommon as V

sys.path.insert(0, V.LIB)
import rescope  # noqa: E402

res = {}
bad, n_all = 0, 0
for a_f, a_x in ((0.01, "1/100"), (0.02, "1/50"), (0.05, "1/20"), (0.1, "1/10"), (0.2, "1/5"), (0.3, "3/10"), (1 / 3, "1/3"), (0.15, "3/20"), (0.025, "1/40")):
    for n in range(1, 2001):
        want = math.ceil((1 - Fraction(a_x)) * (n + 1))
        n_all += 2
        bad += int(rescope.conformal_rank_exact(n, a_f) != want) + int(rescope.conformal_rank_exact(n, a_x) != want)
res["A_rank"] = {"n": n_all, "n_differing": bad}

rng = np.random.default_rng(20261111)
nb, nd = 0, 0
for case in range(300):
    D = 200
    b = rng.normal(0, 1, D) * 10.0 ** rng.integers(-3, 3)
    u = np.ones(D) if case % 3 == 0 else np.exp(rng.normal(0, 2, D))
    eta = [float("inf"), -0.3, 0.0][case] if case < 3 else float(rng.normal(0, 1) * 10.0 ** rng.integers(-3, 2))
    tau = rescope.float_safe_threshold(b, u, eta)
    cands = []
    for base in (tau, b + eta * u if np.isfinite(eta) else tau):
        x = base.copy()
        cands.append(x.copy())
        up, dn = x.copy(), x.copy()
        for _ in range(3):
            up, dn = np.nextafter(up, np.inf), np.nextafter(dn, -np.inf)
            cands += [up.copy(), dn.copy()]
    for r in cands:
        fin = np.isfinite(r)
        with np.errstate(invalid="ignore", over="ignore"):
            want = ((r - b) / u) > eta
        got = r >= tau
        nb += int(fin.sum())
        nd += int((got != want)[fin].sum())
res["B_float_safe"] = {"n": nb, "n_differing": nd}

nc, ndc, neta = 0, 0, 0
for case in range(200):
    n, D = int(rng.choice([40, 200, 201])), int(rng.choice([1, 5, 50]))
    dec = int(rng.choice([1, 2, 12]))
    Q = np.round(rng.normal(0, 1, (n, D)), dec)
    u = None if case % 2 == 0 else np.round(np.exp(rng.normal(0, 1, D)), 3)
    alpha = str(rng.choice(["1/100", "1/20", "1/5"]))
    seed = int(rng.integers(0, 10 ** 6))
    rule = rescope.two_block_rule(Q, alpha, seed=seed, u=u)
    i1, i2 = V.halves(n, seed)
    uu = np.ones(D) if u is None else u
    b = Q[i1].max(0)
    sc = np.sort(((Q[i2] - b) / uu).max(1))
    k = V.rank_exact(len(i2), alpha)
    eta = np.inf if k > len(i2) else sc[k - 1]
    neta += int(not (eta == rule["eta"]))
    T = np.concatenate([Q, np.round(rng.normal(0, 1, (500, D)), dec)])
    want = (((T - b) / uu) > eta).any(1)
    got = (T >= rule["tau"][None, :]).any(1)
    nc += int(T.shape[0])
    ndc += int((want != got).sum())
res["C_two_block_ties_custom_u"] = {"n_cases": 200, "n_eta_differing": neta, "n_decisions": nc, "n_differing": ndc}
res["pass"] = bad == 0 and nd == 0 and ndc == 0 and neta == 0
V.jdump(res, f"{V.OUT}/x6_exactness.json")
print(res)
