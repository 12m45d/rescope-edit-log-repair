"""Check 0 (E5 f): rank K_Q(n) and cap(n) = rank[K_Q(n); K_E] - rank K_Q(n), n in {10, 20, 30}, from my own capture; float64 SVD (LAPACK, singular values only).
A singular value counts if it exceeds 1e-9 x the largest row norm of the matrix (also reported for 1e-10 and 1e-8)."""
from vcommon import *
import fx
name = QDEV
blob = torch.load(f"{OUT}/g_cap_{name}_keys.pt")
L = fx.logs.open(name)
KE = L.log.K.double()
E = jload(f"{FINAL}/e5_rescope/out/f/{name}__s0.json")
C = jload(f"{FINAL}/e5_rescope/out/capacity_curve.json")
grid = {g["n"]: g for g in E["grid"]}
res = {"log": name, "seed": 0, "d_in": int(KE.shape[1]), "T": int(KE.shape[0]), "by_n": {}}
def ranks(M):
    s = torch.linalg.svdvals(M)
    sc = float(M.norm(dim=1).max())
    return {f"{t:g}": int((s > t * sc).sum()) for t in (1e-10, 1e-9, 1e-8)}, float(s[-1] / sc), float(s[0] / sc)
for n in (10, 20, 30):
    t0 = time.time()
    KQ = torch.cat([k.double() for k in blob["K"][:n]], 0)
    rq, smin, smax = ranks(KQ)
    rs, smin2, _ = ranks(torch.cat([KQ, KE], 0))
    g = grid[n]
    cap = {t: rs[t] - rq[t] for t in rq}
    res["by_n"][str(n)] = {"n_keys": int(KQ.shape[0]), "rank_KQ": rq["1e-09"], "rank_KQ_by_tolerance": rq, "rank_stacked": rs["1e-09"], "cap": cap["1e-09"], "cap_by_tolerance": cap,
                           "min_T_d_minus_rank": int(min(KE.shape[0], KE.shape[1] - rq["1e-09"])),
                           "experiment": {"n_keys": g["n_keys"], "rank_KQ": g["rank_KQ"], "cap": g["cap"], "cap_by_tolerance": g["cap_by_tolerance"]},
                           "match": bool(g["n_keys"] == KQ.shape[0] and g["rank_KQ"] == rq["1e-09"] and g["cap"] == cap["1e-09"]), "seconds": time.time() - t0}
    print(n, res["by_n"][str(n)], flush=True)
    jdump(res, OUT + "/c0_capacity.json")
    del KQ
