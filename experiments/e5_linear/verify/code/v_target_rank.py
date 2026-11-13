"""Verification, extra for task (3): the rank of the compensation target.     usage: python v_target_rank.py
                                                                             -> verify/out/v_target_rank.json   (CPU, fp64)
The capacity test ('the complaint stays repairable iff (I - P) K_q keeps full column rank') decides the existence of an
admissible correction X (X K_prot = 0, X K_q = V) exactly when the target V = -(W_E - W_0) K_q has full column rank
(then null(V) = {0}, and null((I - P) K_q) must be {0}; technical report, Lemma lem:app:closed).  Checked here for every
REPAIR-pool D1 complaint with the APPLIED weight difference composite_delta.pt (not the factors):
  last key      ||V|| > 0                                  (a = rank V = 1)
  stacked keys  numerical rank of V = A and of K_q = A     (LAPACK singular values, cut max(shape) * eps * s_1)
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import os
import time

import numpy as np
import scipy.linalg as sla


def nrank(M):
    s = sla.svdvals(M)
    return int((s > max(M.shape) * C.EPS * s[0]).sum()) if s[0] > 0 else 0, float(s[-1] / s[0]) if s[0] > 0 else 0.0


def main():
    t0 = time.time()
    out = {"definition": __doc__, "logs": {}}
    for name in C.LOGS:
        CD = C.composite_delta(name)
        rows = C.probe_rows(name)
        cand = C.candidates(rows)
        samp = {c["group"] for c in C.sample60(cand)}
        comps = cand["image"] + cand["text"]
        tc = C.trackc_keys(name)
        Kl, _ = C.last_keys(name, comps, tc=tc)
        res = []
        for j, c in enumerate(comps):
            K = tc[(c["probe_idx"], c["ptype"])]["K"].double().numpy()            # (A, d_in)
            V = -(CD @ K.T)                                                       # (d_out, A)
            rv, sv = nrank(V)
            rk, sk = nrank(K.T)
            vl = float(np.linalg.norm(CD @ Kl[j].double().numpy()))
            res.append({"group": c["group"], "in_sample": c["group"] in samp, "A": int(K.shape[0]), "rank_V": rv, "rank_Kq": rk,
                        "V_sigma_min_over_max": sv, "Kq_sigma_min_over_max": sk, "V_last_norm": vl})
        L = {"n": len(res), "n_rank_V_equals_A": int(sum(r["rank_V"] == r["A"] for r in res)),
             "n_rank_Kq_equals_A": int(sum(r["rank_Kq"] == r["A"] for r in res)),
             "n_V_last_nonzero": int(sum(r["V_last_norm"] > 0 for r in res)),
             "min_V_sigma_min_over_max": float(min(r["V_sigma_min_over_max"] for r in res)),
             "min_V_last_norm": float(min(r["V_last_norm"] for r in res)),
             "sample": {"n": int(sum(r["in_sample"] for r in res)),
                        "n_rank_V_equals_A": int(sum(r["rank_V"] == r["A"] for r in res if r["in_sample"]))},
             "A_hist": {str(a): int(sum(r["A"] == a for r in res)) for a in sorted({r["A"] for r in res})}, "rows": res}
        out["logs"][name] = L
        print(name, {k: v for k, v in L.items() if k != "rows"})
    out["sec"] = time.time() - t0
    C.jdump(out, os.path.join(C.OUT, "v_target_rank.json"))


if __name__ == "__main__":
    main()
