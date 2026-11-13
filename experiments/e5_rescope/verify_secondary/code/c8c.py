"""Check 8 (c), interval: my own bootstrap over query clusters with the draws nested (the fix-up's method) on the driver's 200 draws; and the same with independent draws -> Monte Carlo SE."""
from c8h import *
res = {}
F = jload(f"{FINAL}/e2_curves/out/fixup/cw_check_200.json")
for name in (QDEV, LDEV):
    L, Rr, Rh, cr, ch, rep, hd = load(name)
    N = len(rep); D = 200
    _, inv = np.unique(cr["qid"], return_inverse=True); U = inv.max() + 1
    num, den, rate = np.zeros((D, U)), np.zeros((D, U)), []
    for d in range(D):
        rows = np.array(splits.sample(list(range(N)), 200, d)); i1, i2 = split_blocks(200, (1, 1), d)
        b, eta, _ = rule(Rr[rows], i1, i2)
        f = (Rr - b[None, :]).max(1) > eta
        inq = set(cr["qid"][rows]); fresh = np.array([q not in inq for q in cr["qid"]])
        num[d] = np.bincount(inv, weights=(f & fresh).astype(float), minlength=U); den[d] = np.bincount(inv, weights=fresh.astype(float), minlength=U)
        rate.append(100 * f[fresh].mean())
    rng = np.random.default_rng(4711)
    bs = []
    for _ in range(2000):
        c = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
        bs.append(100 * np.mean((num @ c) / (den @ c)))
    st = F["logs"][name]["fresh_repair"]
    res[name] = {"mean": float(np.mean(rate)), "cluster_bootstrap_nested_draws": {"ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)), "se": float(np.std(bs, ddof=1))},
                 "stored": {k: st[k] for k in ("mean_over_draws", "ci_lo", "ci_hi", "se")}, "monte_carlo_se_over_draws": float(np.std(rate, ddof=1) / np.sqrt(D)),
                 "z_vs_bound_by_monte_carlo_se": float((np.mean(rate) - BOUND) / (np.std(rate, ddof=1) / np.sqrt(D)))}
    print(name, res[name])
jdump(res, OUT + "/c8c.json")
