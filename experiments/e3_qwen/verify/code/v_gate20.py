"""Verification of E3: the 20-draw gate-level numbers recomputed with MY OWN rule (numpy; v_gpu.my_rule / my_null / feat) on the raw key-cache
files (the cache was checked bitwise against 800 - 2400 keys captured here per log, v_compare.py).  No driver, no library calibration.

    python v_gate20.py [log ...]     -> out/gate20/<log>.json, out/gate20.json, out/gate20_table.txt

Per log, method (rescope, rescope_null, cosine_shared) and calibration draw 0 .. 19: thresholds, capped edits, firing on every valid
heldout_disjoint row, coverage of T-Gen / I-Gen (own edit released, all edits); compared flag by flag with the experiment's gate cells.
Also: are the stored solver keys of the two log families equal at equal T (the experiment's untested attribution)?
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_internal as VI  # noqa: E402

FINAL, V, TESTBED = VI.FINAL, VI.V, VI.TESTBED
ALPHA_NUM, ALPHA_DEN = 1, 20


def conf_rank(n):
    return -((-(ALPHA_DEN - ALPHA_NUM) * (n + 1)) // ALPHA_DEN)


def rule(Q, low, two_block, seed):
    """My rule on calibration features Q (n, D); low (D,) = smallest feature of each edit on its own guard keys."""
    n, D = Q.shape
    if two_block:
        perm = np.random.default_rng(seed).permutation(n)
        i2, b = np.sort(perm[n // 2:]), Q[np.sort(perm[:n // 2])].max(0)
    else:
        i2, b = np.arange(n), np.zeros(D)
    sc = (Q[i2] - b[None, :]).max(1)
    k = conf_rank(len(i2))
    eta = float(np.sort(sc)[k - 1]) if k <= len(i2) else float("inf")
    capped = ~((low - b) > eta)
    return b, eta, low - 1e-9 * np.abs(low), capped


def decide(R, b, eta, cap, capped):
    """-> fired (n,), top edit (n,) for features R (n, D): candidates by my rule (capped edits: r >= cap), top-1 by the margin r - b."""
    mar = R - b[None, :]
    cand = np.where(capped[None, :], R >= cap[None, :], mar > eta)
    return cand.any(1), np.where(cand, mar, -np.inf).argmax(1)


def null_det(W, K, s, Kb):
    Qb, _ = np.linalg.qr(Kb.T)
    P = lambda X: X - (X @ Qb) @ Qb.T                                   # noqa: E731
    Wt, Kt = P(W), P(K)
    return Wt + ((s - (Wt * K).sum(1)) / (Kt * K).sum(1))[:, None] * Kt


def one(bb, log):
    B = VI.BB[bb]
    fam, T = B["fam"], VI.T_of(log)
    ix = {r["sid"]: r for r in VI.INDEX[bb]}
    Ktf = KT[bb]
    key = lambda sids: Ktf[torch.tensor([ix[s]["row_tf"] for s in sids])].double().numpy()          # noqa: E731
    fac = torch.load(f"{TESTBED}/{log}/edit_factors.pt", map_location="cpu")
    W, K = fac["W"].double().numpy(), fac["K1_solver"].double().numpy()
    s = (W * K).sum(1)
    Kn = K / np.linalg.norm(K, axis=1, keepdims=True)
    pr = VI.jl(f"{TESTBED}/{log}/probe_records_v2.jsonl")                # the order of the library's probe items
    sid = lambda r: f"{fam}:probe:{r['ptype']}:{r['probe_idx']}"       # noqa: E731
    rep = [sid(r) for r in pr if ix[sid(r)]["pool"] == "repair" and r["y0_valid"]]
    hd = [sid(r) for r in pr if ix[sid(r)]["pool"] == "heldout" and ix[sid(r)]["disjoint"] and r["y0_valid"]]
    Krep, Khd = key(rep), key(hd)
    Kown = key([f"{fam}:edit:rel:{i}" for i in range(T)])
    Kt, Ki = key([f"{fam}:edit:tgen:{i}" for i in range(T)]), key([f"{fam}:edit:igen:{i}" for i in range(T)])
    qid = [(ix[x]["ptype"], ix[x]["prompt"], ix[x]["image_md5"]) for x in rep]
    E = B["dir"]
    if bb == "qwen":
        ctx = json.load(open(f"{E}/gate/{log}/_context.json"))
        order = [x for x in ctx["hd"]["csid"]]
    else:
        ctx = json.load(open(f"{E}/gate/{log}/_meta.json"))
        hm = VI.uf(ctx["probe"]["masks"]["hd"])
        order = [x for x, f in zip(ctx["probe"]["sid"], hm) if f]
    assert sorted(order) == sorted(hd), "heldout_disjoint rows differ"
    perm = [hd.index(x) for x in order]                                  # my rows in the experiment's order
    out = {"log": log, "T": T, "n_repair_valid": len(rep), "n_hd": len(hd), "methods": {}}

    def F(kind, detw, Kq):
        R = Kq @ detw.T
        return R / s[None, :] if kind == "ncoord" else R / np.linalg.norm(Kq, axis=1, keepdims=True)
    for m in ("rescope", "rescope_null", "cosine_shared"):
        cells = json.load(open(f"{E}/gate/{log}/{m}__reg.json"))["cells"]
        per = []
        for d in range(20):
            idx = np.sort(np.random.default_rng(d).choice(len(rep), 200, replace=False))
            kind, detw = ("cos", Kn) if m == "cosine_shared" else ("ncoord", W)
            if m == "rescope_null":
                rest = np.setdiff1d(np.arange(len(rep)), idx)
                rest = rest[np.random.default_rng([d, 2]).permutation(rest.size)]
                seen, pick = {qid[i] for i in idx}, []
                for k in rest:
                    if len(pick) == 100:
                        break
                    if qid[k] not in seen:
                        seen.add(qid[k])
                        pick.append(int(k))
                detw = null_det(W, K, s, Krep[sorted(pick)])
            stored = (detw * K).sum(1) / s if kind == "ncoord" else np.ones(T)      # guard 1: the stored solver key (cosine with itself: 1)
            own = F(kind, detw, Kown)[np.arange(T), np.arange(T)]                    # guard 2: the B = 1 key of the edit's own Rel prompt
            low = np.minimum(stored, own)
            b, eta, cap, capped = rule(F(kind, detw, Krep[idx]), low, m != "cosine_shared", d)
            fired, _ = decide(F(kind, detw, Khd), b, eta, cap, capped)
            fired = fired[perm]
            cov = {}
            for nm, Kq in (("tgen", Kt), ("igen", Ki)):
                f, top = decide(F(kind, detw, Kq), b, eta, cap, capped)
                cov[nm] = f & (top == np.arange(T))
            c = cells[str(d)]
            if bb == "qwen":
                ef, et, ei = VI.uf(c["hd_flags"]), VI.uf(c["per_edit"]["tgen"]), VI.uf(c["per_edit"]["igen"])
            else:
                ef, et, ei = VI.uf(c["probe_fired"])[hm], VI.uf(c["edit"]["tgen"]["own_top"]), VI.uf(c["edit"]["igen"]["own_top"])
            per.append({"eta_mine": eta, "eta_exp": c["cal"]["eta"], "capped_mine": int(capped.sum()), "capped_exp": c["n_capped"], "fire_mine": 100 * float(fired.mean()),
                        "fire_exp": 100 * float(ef.mean()), "n_hd_flags_differ": int((fired != ef).sum()), "cov_t_mine": 100 * float(cov["tgen"].mean()), "cov_t_exp": 100 * float(et.mean()),
                        "cov_i_mine": 100 * float(cov["igen"].mean()), "cov_i_exp": 100 * float(ei.mean()), "n_cov_flags_differ": int((cov["tgen"] != et).sum() + (cov["igen"] != ei).sum())})
        mean = lambda k: float(np.mean([p[k] for p in per]))            # noqa: E731
        out["methods"][m] = {"fire_mine": mean("fire_mine"), "fire_exp": mean("fire_exp"), "capped_mine": mean("capped_mine"), "capped_exp": mean("capped_exp"),
                             "cov_t_mine": mean("cov_t_mine"), "cov_t_exp": mean("cov_t_exp"), "cov_i_mine": mean("cov_i_mine"), "cov_i_exp": mean("cov_i_exp"),
                             "max_abs_eta_diff": float(max(abs(p["eta_mine"] - p["eta_exp"]) for p in per)), "n_draws_capped_differs": sum(p["capped_mine"] != p["capped_exp"] for p in per),
                             "n_hd_flags_differ_total": sum(p["n_hd_flags_differ"] for p in per), "n_hd_flags_total": 20 * len(hd),
                             "n_cov_flags_differ_total": sum(p["n_cov_flags_differ"] for p in per), "n_cov_flags_total": 40 * T,
                             "n_draws_above_4.95_mine": sum(p["fire_mine"] > 100 * 5 / 101 for p in per), "fire_by_draw_mine": [p["fire_mine"] for p in per]}
    return out


KT = {}


def main():
    logs = sys.argv[1:] or [g for bb in VI.BB for g in VI.BB[bb]["logs"]]
    os.makedirs(V + "/out/gate20", exist_ok=True)
    res = []
    for bb, B in VI.BB.items():
        todo = [g for g in B["logs"] if g in logs]
        if not todo:
            continue
        VI.INDEX[bb] = VI.jl(f"{FINAL}/cache/{B['cache']}/index.jsonl")
        KT[bb] = torch.load(f"{FINAL}/cache/{B['cache']}/keys_tf.pt", mmap=True)
        for log in todo:
            p = f"{V}/out/gate20/{log}.json"
            if not os.path.exists(p):
                json.dump(one(bb, log), open(p, "w"))
            res.append(json.load(open(p)))
            print(log, {m: (round(o["fire_mine"], 3), round(o["fire_exp"], 3), o["n_hd_flags_differ_total"], o["n_cov_flags_differ_total"], o["n_draws_capped_differs"])
                        for m, o in res[-1]["methods"].items()}, flush=True)
    # stored solver keys of the two families at equal T (the experiment's untested attribution)
    same = {}
    for a, b in (("TB-Q100-ae-l2_3", "TB-Q100-memitw"), ("TB-Q500-ae-l2_3", "TB-Q500-memitw"), ("TB-L100-ae-l2_3", "TB-L100-memitw"), ("TB-L500-ae-l2_3", "TB-L500-memitw")):
        fa, fb = (torch.load(f"{TESTBED}/{x}/edit_factors.pt", map_location="cpu") for x in (a, b))
        Ka, Kb = fa["K1_solver"].double(), fb["K1_solver"].double()

        def reb(f, name):
            met = json.load(open(f"{TESTBED}/{name}/metrics.json"))
            bt = torch.zeros(f["U"].shape[0], dtype=torch.long)
            for x in met["per_batch"]:
                bt[torch.tensor(x["edit_ids"])] = int(x["batch"])
            U, W_, K_ = f["U"].double(), f["W"].double(), f["K1_solver"].double()
            M = K_ @ W_.T
            return U + torch.where(bt[None, :] < bt[:, None], M, torch.zeros_like(M)) @ U
        Va, Vb = reb(fa, a), reb(fb, b)
        same[f"{a} | {b}"] = {"solver_keys_bitwise_equal": bool(torch.equal(Ka, Kb)), "max_abs_key_diff": float((Ka - Kb).abs().max()),
                              "rebased_values_rel_diff_median": float(((Va - Vb).norm(dim=1) / Va.norm(dim=1)).median()),
                              "rebased_values_rel_diff_max": float(((Va - Vb).norm(dim=1) / Va.norm(dim=1)).max())}
    json.dump({"logs": res, "families_at_equal_T": same}, open(V + "/out/gate20.json", "w"), indent=1)
    L = ["E3 VERIFICATION: 20 calibration draws, gate level, my own rule on the raw key-cache files against the experiment's gate cells",
         f"{'log':17s} {'method':14s} {'fire m/e (mean of 20)':>22s} {'hd flags differ':>16s} {'capped m/e':>14s} {'cov T m/e':>14s} {'cov I m/e':>14s} {'cov flags differ':>17s} {'draws > 4.95':>12s}"]
    for r in res:
        for m, o in r["methods"].items():
            L.append(f"{r['log']:17s} {m:14s} {o['fire_mine']:10.3f}/{o['fire_exp']:<10.3f} {o['n_hd_flags_differ_total']:>7d}/{o['n_hd_flags_total']:<8d} {o['capped_mine']:6.1f}/{o['capped_exp']:<6.1f} "
                     f"{o['cov_t_mine']:6.2f}/{o['cov_t_exp']:<6.2f} {o['cov_i_mine']:6.2f}/{o['cov_i_exp']:<6.2f} {o['n_cov_flags_differ_total']:>8d}/{o['n_cov_flags_total']:<8d} {o['n_draws_above_4.95_mine']:>6d}/20")
    L.append("")
    for k, v in same.items():
        L.append(f"{k}: {v}")
    open(V + "/out/gate20_table.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-6:]))


if __name__ == "__main__":
    main()
