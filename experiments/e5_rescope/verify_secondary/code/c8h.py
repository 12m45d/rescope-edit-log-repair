"""Check 8 (c), (h): literal two-block rule of ReScope-W (all positions) from the captured per-row maxima, own draws.
(i) REPAIR rows outside the calibration draw, (ii) heldout_disjoint; Monte Carlo SE over independent draws; diagnostics."""
from vgate import *
import fx
from rescope import splits, split_blocks
from fractions import Fraction

def load(name):
    L = fx.logs.open(name)
    if "-p" in name.split("l2_3")[-1]:
        c = torch.load(f"{FINAL}/e2_prefix/out/qmax/{name}.pt"); R = c["Q"].numpy()
    else:
        c = torch.load(f"{FINAL}/e2_curves/out/capture/{name}.pt"); R = c["Rmax"].numpy()
    ix = {j: i for i, j in enumerate(c["rows"])}
    rep, hd = L.pool("repair"), L.pool("heldout_disjoint")
    cov = lambda rows: {"ptype": np.array([L.index[j]["ptype"] for j in rows]), "near": np.array([L.near_edit(j) is not None for j in rows]),
                        "ntok": np.array([L.index[j]["pre_n_tok"] for j in rows]), "len": np.array([L.index[j]["len_tf"] for j in rows]),
                        "qid": np.array(fx.draws.query_ids(L, rows))}
    return L, R[[ix[j] for j in rep]], R[[ix[j] for j in hd]], cov(rep), cov(hd), rep, hd

def rule(Rcal, i1, i2, k=96):
    b = Rcal[i1].max(0)
    sc = np.sort((Rcal[i2] - b[None, :]).max(1))
    return b, float(sc[k - 1]), sc

def stat(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "mc_se": float(x.std(ddof=1) / np.sqrt(len(x))), "sd": float(x.std(ddof=1)), "n_draws": len(x), "share_above_bound": float((x > BOUND).mean())}

BOUND = 100 * 5 / 101

def analyse(name, ndraw=200):
    L, Rr, Rh, cr, ch, rep, hd = load(name)
    N = len(rep)
    out = {"log": name, "T": L.T, "n_repair": N, "n_heldout_disjoint": len(hd), "bound_pct": BOUND, "n_repair_clusters": int(len(set(cr["qid"]))),
           "composition": {p: {"repair": {"image": int((cr["ptype"] == "image").sum()), "text": int((cr["ptype"] == "text").sum()), "near": int(cr["near"].sum()),
                                           "mean_answer_tokens": float(cr["ntok"].mean()), "mean_len": float(cr["len"].mean())},
                               "heldout_disjoint": {"image": int((ch["ptype"] == "image").sum()), "text": int((ch["ptype"] == "text").sum()), "near": int(ch["near"].sum()),
                                                    "mean_answer_tokens": float(ch["ntok"].mean()), "mean_len": float(ch["len"].mean())}}[p] for p in ("repair", "heldout_disjoint")}}
    # ---- (A) the experiment's draws (library sample + split, the same seed), to check my rule against the stored numbers
    exp = {"rest": [], "fresh": [], "hd": [], "rank_block1": [], "img_block1": [], "dup_across": [], "ties": []}
    for d in range(ndraw):
        rows = splits.sample(list(range(N)), 200, d); i1, i2 = split_blocks(200, (1, 1), d)
        rows = np.array(rows)
        b, eta, sc = rule(Rr[rows], i1, i2)
        f = (Rr - b[None, :]).max(1) > eta
        inq = set(cr["qid"][rows]); fresh = np.array([q not in inq for q in cr["qid"]]); rest = np.ones(N, bool); rest[rows] = False
        exp["rest"].append(100 * f[rest].mean()); exp["fresh"].append(100 * f[fresh].mean()); exp["hd"].append(100 * ((Rh - b[None, :]).max(1) > eta).mean())
        exp["rank_block1"].append(float(np.mean(i1)))               # mean position (in pool order) of block 1 within the sample; 99.5 if the split ignores the order
        exp["img_block1"].append(float((cr["ptype"][rows[i1]] == "image").mean() - (cr["ptype"][rows[i2]] == "image").mean()))
        q1, q2 = set(cr["qid"][rows[i1]]), set(cr["qid"][rows[i2]])
        exp["dup_across"].append(len(q1 & q2)); exp["ties"].append(int(len(sc) - len(np.unique(sc))))
    out["experiment_draws"] = {"what": "seeds 0..199 of the driver's draw (rescope.splits.sample + split_blocks, same seed), my arithmetic", "rest_rows": stat(exp["rest"]), "fresh_by_query": stat(exp["fresh"]),
                               "heldout_disjoint": stat(exp["hd"]), "first20_fresh": float(np.mean(exp["fresh"][:20])), "first20_heldout_disjoint": float(np.mean(exp["hd"][:20])),
                               "block1_mean_position": stat(exp["rank_block1"]), "block1_minus_block2_image_share": stat(exp["img_block1"]),
                               "queries_on_both_sides_of_the_split_mean": float(np.mean(exp["dup_across"])), "tied_block2_scores_mean": float(np.mean(exp["ties"]))}
    # ---- (B) my own draws: one random permutation of the pool per draw (calibration = first 200: block 1 = first 100, block 2 = next 100; rest = the other rows)
    own = {k: [] for k in ("rest", "fresh", "hd", "rest_image", "rest_text", "hd_image", "hd_text", "rest_near", "rest_far", "hd_near", "hd_far", "hd_std")}
    wr = {(p, n): float(((cr["ptype"] == p) & (cr["near"] == n)).mean()) for p in ("image", "text") for n in (False, True)}
    for d in range(ndraw):
        perm = np.random.default_rng([977, 8, d]).permutation(N)
        cal, rest = perm[:200], perm[200:]
        b, eta, sc = rule(Rr[cal], np.arange(100), np.arange(100, 200))
        f = (Rr - b[None, :]).max(1) > eta; fh = (Rh - b[None, :]).max(1) > eta
        inq = set(cr["qid"][cal]); fresh = np.array([q not in inq for q in cr["qid"]])
        own["rest"].append(100 * f[rest].mean()); own["fresh"].append(100 * f[fresh].mean()); own["hd"].append(100 * fh.mean())
        for p in ("image", "text"):
            own["rest_" + p].append(100 * f[rest][cr["ptype"][rest] == p].mean()); own["hd_" + p].append(100 * fh[ch["ptype"] == p].mean())
        for nm, v in (("near", True), ("far", False)):
            m = cr["near"][rest] == v
            own["rest_" + nm].append(100 * f[rest][m].mean() if m.any() else np.nan); own["hd_" + nm].append(100 * fh[ch["near"] == v].mean())
        s = 0.0                                                     # heldout_disjoint rate standardised to the REPAIR composition (modality x near-scope)
        for (p, n), w in wr.items():
            m = (ch["ptype"] == p) & (ch["near"] == n)
            s += w * (fh[m].mean() if m.any() else 0.0)
        own["hd_std"].append(100 * s)
    out["own_draws"] = {k: stat(np.array(v)[~np.isnan(v)]) for k, v in own.items()}
    r = out["own_draws"]["rest"]
    out["own_draws"]["z_rest_vs_bound"] = (r["mean"] - BOUND) / r["mc_se"]
    r = out["own_draws"]["fresh"]
    out["own_draws"]["z_fresh_vs_bound"] = (r["mean"] - BOUND) / r["mc_se"]
    # ---- (C) one row per query cluster (no query can sit on both sides), own draws
    _, first = np.unique(cr["qid"], return_index=True); first = np.sort(first)
    Rd = Rr[first]; Nd = len(first); dd = []
    for d in range(ndraw):
        perm = np.random.default_rng([977, 9, d]).permutation(Nd)
        b, eta, sc = rule(Rd[perm[:200]], np.arange(100), np.arange(100, 200))
        dd.append(100 * ((Rd[perm[200:]] - b[None, :]).max(1) > eta).mean())
    out["own_draws_one_row_per_query"] = dict(stat(dd), n_pool=Nd)
    out["own_draws_one_row_per_query"]["z_vs_bound"] = (out["own_draws_one_row_per_query"]["mean"] - BOUND) / out["own_draws_one_row_per_query"]["mc_se"]
    # ---- covariates of the firing rows (pooled over own draws would need per-row rates): per-row firing frequency by answer length / sequence length
    return out

if __name__ == "__main__":
    res = {}
    for name in ("TB-Q100-ae-l2_3-p50", QDEV, LDEV):
        res[name] = analyse(name)
        jdump(res, OUT + "/c8h.json")
        o = res[name]
        print(name, "bound %.2f" % BOUND)
        print("  exp draws: rest", o["experiment_draws"]["rest_rows"], "\n    fresh", o["experiment_draws"]["fresh_by_query"], "\n    hd", o["experiment_draws"]["heldout_disjoint"])
        print("    first20 fresh %.3f hd %.3f | block1 pos %s | img diff %s | dup across %.2f ties %.2f" % (o["experiment_draws"]["first20_fresh"], o["experiment_draws"]["first20_heldout_disjoint"],
              o["experiment_draws"]["block1_mean_position"], o["experiment_draws"]["block1_minus_block2_image_share"], o["experiment_draws"]["queries_on_both_sides_of_the_split_mean"], o["experiment_draws"]["tied_block2_scores_mean"]))
        for k, v in o["own_draws"].items(): print("  own", k, v if not isinstance(v, dict) else {a: round(b, 3) for a, b in v.items()})
        print("  dedup", o["own_draws_one_row_per_query"], "\n  comp", o["composition"])
