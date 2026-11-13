"""Test of the crossed bootstrap (xcommon.xboot / xboot_diff) on synthetic data -> out/tests_boot.json.

  a  D = 1 reduces to the unit bootstrap: se = sd / sqrt(U) (within 10 %)
  b  crossed design y[d, u] = a_d + b_u + e: the bootstrap variance matches the pigeonhole value
     var_a (D-1)/D^2 ... checked against a Monte-Carlo of the same resampling scheme done naively (within 10 %)
  c  coverage of the 95 % interval for the unit-only design over 300 synthetic data sets: between 90 % and 98 %
  d  units with several rows: a unit is resampled with all its rows (naive implementation, same weights: equal to 1e-12)
  e  paired difference: a constant shift is recovered exactly and its interval has zero width; for independent noise the
     paired interval is narrower than the unpaired one when the two outcomes share the unit effect
  f  missing values (NaN) are ignored in numerator and denominator
"""
import sys

import numpy as np

import xcommon as X

res = {}
rng = np.random.default_rng(1)

# a
y = rng.normal(0, 1, size=4000)
b = X.xboot(y, None, scale=1.0, seed=3)
res["a"] = {"se": b["se"], "theory": float(y.std(ddof=1) / np.sqrt(y.size)), "ok": abs(b["se"] / (y.std(ddof=1) / np.sqrt(y.size)) - 1) < 0.10}

# b: naive resampling with explicit index arrays, other random numbers
D, U = 5, 400
a_d, b_u = rng.normal(0, 1.0, size=D), rng.normal(0, 2.0, size=U)
Y = a_d[:, None] + b_u[None, :] + rng.normal(0, 1.0, size=(D, U))
xb = X.xboot(Y, None, scale=1.0, seed=5)
r2 = np.random.default_rng(99)
naive = []
for _ in range(4000):
    di, ui = r2.integers(0, D, D), r2.integers(0, U, U)
    naive.append(Y[np.ix_(di, ui)].mean())
res["b"] = {"se": xb["se"], "naive_se": float(np.std(naive, ddof=1)), "value": xb["value"], "mean": float(Y.mean()),
            "ok": abs(xb["se"] / np.std(naive, ddof=1) - 1) < 0.10 and abs(xb["value"] - Y.mean()) < 1e-12}

# c
cover = 0
for s in range(300):
    yy = np.random.default_rng(1000 + s).binomial(1, 0.3, size=300).astype(float)
    bb = X.xboot(yy, None, scale=1.0, seed=s)
    cover += int(bb["ci_lo"] <= 0.3 <= bb["ci_hi"])
res["c"] = {"coverage": cover / 300, "ok": 0.90 <= cover / 300 <= 0.98}

# d
units = np.repeat(np.arange(50), 3)
Yd = rng.normal(0, 1, size=(2, 150))
Wd, Wu = X._weights(2, 50, 500, 7)
S, N = X._unit_sums(Yd, units)
fast = X._boot_means(S, N, Wd, Wu)
slow = []
for i in range(500):
    num = den = 0.0
    for d in range(2):
        for u in range(50):
            w = Wd[i, d] * Wu[i, u]
            num += w * Yd[d, units == u].sum()
            den += w * 3
    slow.append(num / den)
res["d"] = {"max_abs_diff": float(np.abs(fast - np.array(slow)).max()), "ok": float(np.abs(fast - np.array(slow)).max()) < 1e-12}

# e
base = rng.normal(0, 1, size=(3, 600)) + rng.normal(0, 3, size=600)[None, :]
d0 = X.xboot_diff(base + 0.25, base, None, scale=1.0)
A = base + rng.normal(0, 0.5, size=base.shape) + 0.25
B = base + rng.normal(0, 0.5, size=base.shape)
dp = X.xboot_diff(A, B, None, scale=1.0)
wa, wb = X.xboot(A, None, scale=1.0), X.xboot(B, None, scale=1.0)
unpaired = float(np.hypot(wa["se"], wb["se"]))
single = X.xboot_diff(A, B[:1], None, scale=1.0)                      # a row without calibration is repeated over the draws
res["e"] = {"shift": d0["value"], "shift_width": d0["ci_hi"] - d0["ci_lo"], "paired_se": dp["se"], "unpaired_se": unpaired,
            "single_draw_value": single["value"], "single_expected": float(A.mean() - B[0].mean()),
            "ok": abs(d0["value"] - 0.25) < 1e-12 and d0["ci_hi"] - d0["ci_lo"] < 1e-9 and dp["se"] < 0.5 * unpaired
            and abs(single["value"] - (A.mean() - B[0].mean())) < 1e-12}

# f
Yf = np.array([[1.0, np.nan, 0.0, 1.0], [np.nan, np.nan, 1.0, 1.0]])
bf = X.xboot(Yf, [0, 0, 1, 2], scale=1.0)
res["f"] = {"value": bf["value"], "ok": abs(bf["value"] - 4.0 / 5.0) < 1e-12}

res["all_ok"] = all(v["ok"] for v in res.values())
X.jdump(res, X.OUT + "/tests_boot.json")
print({k: (v["ok"] if isinstance(v, dict) else v) for k, v in res.items()})
sys.exit(0 if res["all_ok"] else 1)
