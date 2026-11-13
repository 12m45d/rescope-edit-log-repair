"""Self-test of the E5 code (CPU, a minute): -> out/selftest.json.   usage: python e5_selftest.py

(a) rules and bounds against the brute-force optimum on small random instances (all 2^T kept sets, fp64):
    lb_cond <= OPT_c, lb_tri <= OPT_c, OPT_c <= every rule, lf <= lf_guar, and the last claim of Prop. prop:cond:
    lf(c) <= OPT at tolerance c sqrt(gamma / Gamma);
(b) the exact vertex cover number against networkx (clique number of the complement) and, for T <= 14, brute force;
(c) this study's best-improvement and static first-order rules against Track C's ropt functions on real Gram matrices
    of both logs (differences can only come from the incremental residual of the Track C code);
(d) the capacity formula T - s*_Z (e5_capacity.kept_pred) against the brute-force minimum sacrifice on small random
    groups in general position modulo a shared subspace (rank tests in fp64).
"""
from __future__ import annotations

import itertools
import os
import sys

import networkx as nx
import numpy as np
import torch

import e5_common as C
import e5_linalg as LA
from e5_capacity import kept_pred

sys.path.insert(0, os.path.join(C.ROPT, "code"))
import ropt  # noqa: E402


def brute_opt(G, thr):
    T = len(G)
    best = T
    for m in range(1 << T):
        y = np.array([(m >> i) & 1 for i in range(T)], dtype=float)
        if T - int(y.sum()) < best and y @ G @ y <= thr:
            best = T - int(y.sum())
    return best


def test_rules(rng, n=60):
    res = {"instances": 0, "lb_cond_le_opt": 0, "lb_tri_le_opt": 0, "opt_le_rules": 0, "lf_le_guar": 0, "prop_last_claim": 0,
           "rule_equals_opt": {k: 0 for k in LA.RULES}, "checks": 0}
    for _ in range(n):
        T = int(rng.integers(6, 12))
        m = int(rng.integers(T, T + 4))                       # T <= d_out so that gamma > 0 generically
        Wm = rng.normal(size=(T, m)) * rng.uniform(0.2, 3.0, size=(T, 1)) + rng.normal(size=(1, m)) * rng.uniform(0, 1.5)
        G = Wm @ Wm.T
        st = LA.cond_stats(G)
        for c in (0.1, 0.3, 0.5, 0.8):
            thr = c * c * G.sum()
            opt = brute_opt(G, thr)
            r = LA.rollback_rules(G, c, st["gamma"], st["Gamma"])
            c2 = c * np.sqrt(st["gamma"] / st["Gamma"])
            opt2 = brute_opt(G, c2 * c2 * G.sum())
            res["checks"] += 1
            res["lb_cond_le_opt"] += r["lb_cond"] <= opt
            res["lb_tri_le_opt"] += r["lb_tri"] <= opt
            res["opt_le_rules"] += all(opt <= r[k] for k in LA.RULES)
            res["lf_le_guar"] += r["lf"] <= r["lf_guar"]
            res["prop_last_claim"] += r["lf"] <= opt2
            for k in LA.RULES:
                res["rule_equals_opt"][k] += r[k] == opt
        res["instances"] += 1
    return res


def test_vc(rng, n=80):
    ok_nx = ok_bf = n_bf = 0
    for t in range(n):
        T = int(rng.integers(8, 15)) if t < 30 else int(rng.integers(20, 60))
        Wm = rng.normal(size=(T, 6)) + rng.normal(size=(1, 6)) * rng.uniform(0, 1.0)
        G = Wm @ Wm.T
        nu = LA.vertex_cover_number(G)["nu"]
        H = nx.Graph()
        H.add_nodes_from(range(T))
        H.add_edges_from((i, j) for i in range(T) for j in range(i + 1, T) if G[i, j] >= 0)
        ok_nx += nu == T - nx.max_weight_clique(H, weight=None)[1]
        if T <= 14:
            E = [(i, j) for i in range(T) for j in range(i + 1, T) if G[i, j] < 0]
            bf = next(k for k in range(T + 1) for S in itertools.combinations(range(T), k)
                      if all(i in S or j in S for i, j in E))
            n_bf += 1
            ok_bf += nu == bf
    return {"graphs": n, "equal_networkx": int(ok_nx), "brute_force_graphs": int(n_bf), "equal_brute_force": int(ok_bf)}


def test_ropt(n_per_log=20):
    out = {}
    for name in C.LOGS:
        L = C.load_log(name)
        U = L.U.numpy()
        UU = U @ U.T
        comps = C.complaints(name)[:: max(1, len(C.complaints(name)) // n_per_log)][:n_per_log]
        keys = C.complaint_keys(name, comps)
        eq = {"bi": 0, "fo_static": 0, "n": 0}
        for c in comps:
            for kd in C.KEYDEFS:
                K = keys[c["group"]].double()
                K = K[:1] if kd == "last" else K
                G = LA.gram(UU, (L.W @ K.T).numpy())
                for t in C.TOLS:
                    thr = t * t * G.sum()
                    eq["n"] += 1
                    eq["bi"] += len(LA.rule_bi(G, thr)) == ropt.greedy_best_improvement(G, thr)[0]
                    eq["fo_static"] += len(LA.rule_fo_static(G, thr)) == ropt.greedy_first_order(G, thr)[0]
        out[name] = eq
    return out


def test_capacity(rng, n=40):
    ok = tot = 0
    for _ in range(n):
        d, T, z, a = int(rng.integers(18, 30)), int(rng.integers(4, 8)), int(rng.integers(0, 3)), int(rng.integers(1, 3))
        ranks = [int(z + rng.integers(1, 7)) for _ in range(T)]
        Z = rng.normal(size=(d, z))
        S = [np.concatenate([Z, rng.normal(size=(d, r - z))], 1) for r in ranks]
        Kq = rng.normal(size=(d, a))
        best = T
        for k in range(T + 1):
            hit = False
            for R in itertools.combinations(range(T), k):
                kept = [S[i] for i in range(T) if i not in R]
                if kept:
                    Um, sv, _ = np.linalg.svd(np.concatenate(kept, 1), full_matrices=False)
                    Q = Um[:, : int((sv > 1e-9 * sv[0]).sum())]
                    Kp = Kq - Q @ (Q.T @ Kq)
                else:
                    Kp = Kq
                s = np.linalg.svd(Kp, compute_uv=False)
                if (s > 1e-9 * np.linalg.norm(Kq, axis=0).max()).sum() == a:
                    hit = True
                    break
            if hit:
                best = k
                break
        tot += 1
        ok += best == T - kept_pred(ranks, d, a, z)
    return {"instances": tot, "opt_equals_T_minus_kept_pred": int(ok)}


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(20261111)
    out = {"definition": __doc__, "rules": test_rules(rng), "vertex_cover": test_vc(rng), "ropt": test_ropt(),
           "capacity_law": test_capacity(rng)}
    r, v, k = out["rules"], out["vertex_cover"], out["capacity_law"]
    out["all_pass"] = bool(r["lb_cond_le_opt"] == r["lb_tri_le_opt"] == r["opt_le_rules"] == r["lf_le_guar"] == r["prop_last_claim"] == r["checks"]
                           and v["equal_networkx"] == v["graphs"] and v["equal_brute_force"] == v["brute_force_graphs"]
                           and k["opt_equals_T_minus_kept_pred"] == k["instances"])
    C.jdump(out, os.path.join(C.OUT, "selftest.json"))
    print({kk: vv for kk, vv in out.items() if kk != "definition"})


if __name__ == "__main__":
    main()
