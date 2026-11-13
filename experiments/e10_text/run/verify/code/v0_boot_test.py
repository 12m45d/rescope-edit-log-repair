"""Test of the verifier's crossed bootstrap on synthetic data -> out/boot_test.json.

(a) one draw, independent Bernoulli rows: the interval matches the normal approximation of a proportion;
(b) clustered rows (units of 5 identical rows): the interval is sqrt(5) wider than the row-level one;
(c) crossed: a draw effect added to every row of a draw: the standard error matches the analytic value of the two-way model
    (for D draws and U units: var = s_d^2 / D + s_u^2 / U approximately, bootstrap of D draws carries (D-1)/D of the draw term);
(d) coverage: 300 synthetic data sets, the 95 % interval contains the truth in about 95 % (units only);
(e) paired difference: two methods that share the unit effect: the paired interval is far narrower than the unpaired one.
"""
import numpy as np

import vcommon as V

rng = np.random.default_rng(7)
out = {}
# (a)
p, R = 0.3, 2000
y = (rng.random(R) < p).astype(float)
b = V.crossed_bootstrap(y)
se_th = 100 * np.sqrt(y.mean() * (1 - y.mean()) / R)
out["a_bernoulli"] = {"value": b["value"], "se_boot": b["se"], "se_normal": se_th, "ok": bool(abs(b["se"] / se_th - 1) < 0.08 and abs(b["value"] - 100 * y.mean()) < 1e-9)}
# (b)
U = 400
yu = (rng.random(U) < p).astype(float)
y = np.repeat(yu, 5)
units = np.repeat(np.arange(U), 5)
b_unit = V.crossed_bootstrap(y, units)
b_row = V.crossed_bootstrap(y)
out["b_clustered"] = {"se_units": b_unit["se"], "se_rows": b_row["se"], "ratio": b_unit["se"] / b_row["se"], "expected_ratio": float(np.sqrt(5)),
                      "ok": bool(abs(b_unit["se"] / b_row["se"] / np.sqrt(5) - 1) < 0.1 and b_unit["n_units"] == U)}
# (c)
D, U = 20, 500
sd_d, sd_u, sd_e = 0.05, 0.2, 0.1
ses, ths = [], []
for rep in range(20):
    Y = 0.5 + sd_d * rng.standard_normal((D, 1)) + sd_u * rng.standard_normal((1, U)) + sd_e * rng.standard_normal((D, U))
    b = V.crossed_bootstrap(Y, n_boot=500, seed=rep)
    ses.append(b["se"])
    # plug-in two-way variance of the grand mean under crossed resampling
    a = Y.mean(1).var(ddof=0)
    c = Y.mean(0).var(ddof=0)
    ths.append(100 * np.sqrt(a / D + c / U))
out["c_crossed"] = {"se_boot_mean": float(np.mean(ses)), "se_two_way_plug_in_mean": float(np.mean(ths)), "ratio": float(np.mean(ses) / np.mean(ths)),
                    "ok": bool(abs(np.mean(ses) / np.mean(ths) - 1) < 0.1)}
# (d)
hit, n = 0, 300
for rep in range(n):
    y = (rng.random(400) < 0.2).astype(float)
    b = V.crossed_bootstrap(y, n_boot=400, seed=rep)
    hit += int(b["ci_lo"] <= 20.0 <= b["ci_hi"])
out["d_coverage"] = {"coverage_pct": 100.0 * hit / n, "ok": bool(0.90 <= hit / n <= 0.985)}
# (e)
U = 600
base = rng.random(U)
ya = (base < 0.50).astype(float)
yb = (base < 0.46).astype(float)
bd = V.crossed_bootstrap_diff(ya, yb)
unp = 100 * np.sqrt(ya.var() / U + yb.var() / U)
out["e_paired"] = {"diff": bd["value"], "se_paired": bd["se"], "se_unpaired_normal": float(unp), "ci": [bd["ci_lo"], bd["ci_hi"]],
                   "ok": bool(bd["se"] < 0.5 * unp and bd["ci_lo"] > 0 and abs(bd["value"] - 100 * (ya.mean() - yb.mean())) < 1e-9)}
# (f) missing values and several draws with a draw-less method (repeat)
Y = rng.random((3, 50))
Y[1, :10] = np.nan
b = V.crossed_bootstrap(Y, np.arange(50) // 2)
out["f_missing"] = {"value": b["value"], "expected": float(100 * np.nanmean(Y)), "ok": bool(abs(b["value"] - 100 * np.nansum(Y) / np.sum(~np.isnan(Y))) < 1e-9 and b["n_units"] == 25)}
out["all_ok"] = all(v["ok"] for v in out.values())
V.jdump(out, V.OUT + "/boot_test.json")
print({k: (v["ok"] if isinstance(v, dict) else v) for k, v in out.items()})
