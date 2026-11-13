"""Check 8 (f): which keys do BEN-T / MME-T false-firing cells use?  Reproduce the stored cells of E1 / E2 (driver draws 0-19, n = 200, alpha = 0.05, default method with guards)
with my own gate arithmetic at teacher-forced and at prompt-only keys of the pool rows; and the size of the difference over 200 own draws."""
from vgate import *
import fx
from rescope import splits, split_blocks
res = {}
for name, sh in ((QDEV, "Q"), (LDEV, "L")):
    L = fx.logs.open(name)
    cap, _ = guard_caps(L)
    rep = L.pool("repair")
    Pb = fx.pools.get(L, "BEN-T", "decode")
    Mt_tf, Mt_dec = fx.pools.get(L, "MME-T", "tf"), fx.pools.get(L, "MME-T", "decode")
    R = {"rep_tf": ncoord(L.log, keys_of(L.kc, rep, "tf")), "rep_dec": ncoord(L.log, keys_of(L.kc, rep, "decode")), "ben_dec": ncoord(L.log, keys_of(L.kc, Pb.rows, "decode")),
         "mme_tf": ncoord(L.log, keys_of(Mt_tf.kc, Mt_tf.rows, "tf")), "mme_dec": ncoord(L.log, keys_of(Mt_dec.kc, Mt_dec.rows, "decode")),
         "mme_dec_valid": ncoord(L.log, keys_of(Mt_tf.kc, Mt_tf.rows, "decode"))}
    drv = {k: [] for k in ("MME-T tf keys (cal tf)", "MME-T prompt-only keys (cal tf)", "MME-T prompt-only keys, valid rows (cal tf)", "BEN-T prompt-only keys (cal tf)")}
    for d in range(20):                                             # the driver's draws
        rows = splits.sample(list(range(len(rep))), 200, d); i1, i2 = split_blocks(200, (1, 1), d)
        Q = R["rep_tf"][rows]
        b = Q[i1].max(0); eta = float(np.sort((Q[i2] - b[None, :]).max(1))[95])
        for k, t in (("MME-T tf keys (cal tf)", "mme_tf"), ("MME-T prompt-only keys (cal tf)", "mme_dec"), ("MME-T prompt-only keys, valid rows (cal tf)", "mme_dec_valid"), ("BEN-T prompt-only keys (cal tf)", "ben_dec")):
            drv[k].append(100 * fire(R[t], b, eta, cap)[1].mean())
    own = {k: [] for k in ("MME-T tf | cal tf", "MME-T prompt | cal tf", "MME-T prompt | cal prompt", "BEN-T prompt | cal tf", "BEN-T prompt | cal prompt")}
    N = len(rep)
    for d in range(200):
        rng = np.random.default_rng([977, 6, d]); rows = rng.permutation(N)[:200]; perm = rng.permutation(200)
        for calk, tag in (("rep_tf", "cal tf"), ("rep_dec", "cal prompt")):
            b, eta, _ = two_block(R[calk][rows], perm)
            for k, t in ((f"MME-T tf | {tag}", "mme_tf"), (f"MME-T prompt | {tag}", "mme_dec"), (f"BEN-T prompt | {tag}", "ben_dec")):
                if k in own: own[k].append(100 * fire(R[t], b, eta, cap)[1].mean())
    st = lambda x: {"mean": float(np.mean(x)), "se": float(np.std(x, ddof=1) / np.sqrt(len(x))), "n_draws": len(x)}
    # stored cells
    E1 = jload(f"{FINAL}/e1_main/out/paper_numbers.json")
    E2f = jload(f"{FINAL}/e2_curves/out/fixup/paper_numbers_fixup.json")
    cg = jload(f"{FINAL}/e2_curves/out/parts/c_gate_{name}.json")
    c200 = [c for c in cg["cells"] if c["method"] == "rescope" and c["n"] == 200]
    e2 = {"mean_over_draws": float(np.mean([c["st"]["MME-T"]["fired"] for c in c200])) if c200 and "MME-T" in c200[0]["st"] else None,
          "which_field": sorted({c["st"]["MME-T"]["which"] for c in c200}) if c200 and "MME-T" in c200[0]["st"] else None, "n_rows": sorted({c["st"]["MME-T"]["n"] for c in c200}) if c200 and "MME-T" in c200[0]["st"] else None,
          "ben_t_mean": float(np.mean([c["st"]["BEN-T"]["fired"] for c in c200])) if c200 and "BEN-T" in c200[0]["st"] else None,
          "ben_t_which": sorted({c["st"]["BEN-T"]["which"] for c in c200}) if c200 and "BEN-T" in c200[0]["st"] else None, "n_draws": len(c200)}
    E5 = jload(f"{FINAL}/e5_rescope/out/paper_numbers.json")
    res[name] = {"n_rows": {"MME-T tf (valid pre-edit answer)": len(Mt_tf.rows), "MME-T prompt-only (all rows)": len(Mt_dec.rows), "BEN-T": len(Pb.rows)},
                 "driver_draws_0_19_my_arithmetic": {k: float(np.mean(v)) for k, v in drv.items()},
                 "stored": {"E1 EoneXRescopeMmeT": E1.get(f"Eone{sh}RescopeMmeT"), "E1 EoneXRescopeBenT": E1.get(f"Eone{sh}RescopeBenT"), "E2 part c, n = 200": e2,
                            "E2 fixup MmeT n = 200 mean": E2f.get(f"EtwoFx{sh}RescopeMmeTTwoHundredMean"),
                            "E5 repair_to_mme_t (prompt-only both sides, 500 draws)": E5.get(f"validity.{name}.two_block.repair_to_mme_t.ff_mean"),
                            "E5 repair_to_ben_t all rows": E5.get(f"validity.{name}.two_block.repair_to_ben_t.all_rows.ff_mean")},
                 "own_200_draws": {k: st(v) for k, v in own.items()}}
    print(name, json.dumps(res[name], indent=1))
jdump(res, OUT + "/c8f.json")
