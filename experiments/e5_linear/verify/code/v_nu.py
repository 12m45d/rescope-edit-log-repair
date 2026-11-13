"""Verification, extra: the vertex cover number nu of the graph of negative pairs at the last prompt key, EXACT, by two
methods that share no code with the measuring agent's branch and bound.  Reads NO file of e5_linear/out.

usage: python v_nu.py --log TB-Q100-ae-l2_3 [--workers 8]        -> verify/out/v_nu_<log>.json

Negative graph of a complaint: vertices = the T edits, an edge {i, j} iff <w_i, w_j> < 0 (explicit contributions
w_i = u_i <psi_i, k>, signed inner products, fp64).  nu = T - alpha, alpha = the largest set of edits with pairwise
non-negative inner products.
  (A) integer program: max sum_i x_i, x_i + x_j <= 1 for every negative pair, x binary; scipy.optimize.milp (HiGHS);
      a row counts only with status 'optimal' (0);
  (B) clique number of the complement graph by networkx.max_weight_clique (exact branch and bound).
All REPAIR-pool D1 complaints are computed (the quoted number is a MINIMUM over the complaints); the sample of 60 is flagged.
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import argparse
import multiprocessing as mp
import os
import time

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

_S = {}


def _init(name):
    F = C.factors(name)
    _S.update(U=F["U"], Psi=F["Psi"])


def one(args):
    group, ptype, k, in_sample = args
    U, Psi = _S["U"], _S["Psi"]
    c = Psi @ np.asarray(k, dtype=np.float64)
    Wc = U * c[:, None]
    G = Wc @ Wc.T
    G = 0.5 * (G + G.T)
    T = len(G)
    neg = G < 0
    np.fill_diagonal(neg, False)
    ii, jj = np.nonzero(np.triu(neg, 1))
    m = len(ii)
    A = coo_matrix((np.ones(2 * m), (np.r_[np.arange(m), np.arange(m)], np.r_[ii, jj])), shape=(m, T)).tocsr()
    r = milp(c=-np.ones(T), constraints=LinearConstraint(A, -np.inf, 1.0), integrality=np.ones(T), bounds=Bounds(0, 1),
             options={"time_limit": 120.0})
    x = np.round(r.x).astype(int) if r.x is not None else None
    ok = bool(r.status == 0 and x is not None and not neg[np.ix_(x == 1, x == 1)].any())      # optimal, and the set is independent
    g = nx.Graph()
    g.add_nodes_from(range(T))
    a, b = np.nonzero(np.triu(~neg, 1))
    g.add_edges_from(zip(a.tolist(), b.tolist()))
    _, w = nx.algorithms.clique.max_weight_clique(g, weight=None)
    return {"group": group, "ptype": ptype, "in_sample": bool(in_sample), "n_neg_pairs": int(m),
            "alpha_milp": int(round(-r.fun)) if r.fun is not None else None, "milp_status": int(r.status), "milp_ok": ok,
            "alpha_clique": int(w), "nu": int(T - w), "n_pos_coord": int((c > 0).sum()), "n_neg_coord": int((c < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    fh = open(os.path.join(C.LOGDIR, f"v_nu_{a.log}.log"), "a")
    rows = C.probe_rows(a.log)
    cand = C.candidates(rows)
    samp = {c["group"] for c in C.sample60(cand)}
    comps = cand["image"] + cand["text"]
    Kl, _ = C.last_keys(a.log, comps)
    jobs = [(c["group"], c["ptype"], Kl[j].numpy(), c["group"] in samp) for j, c in enumerate(comps)]
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.log,)) as pool:
        res = pool.map(one, jobs, chunksize=4)
    nu = [r["nu"] for r in res]
    out = {"definition": __doc__, "log": a.log, "n_complaints": len(res), "rows": res,
           "n_milp_optimal": int(sum(r["milp_ok"] for r in res)),
           "n_methods_agree": int(sum(r["alpha_milp"] == r["alpha_clique"] for r in res)),
           "nu": C.quart(nu), "nu_sample": C.quart([r["nu"] for r in res if r["in_sample"]]),
           "n_cancellation_free": int(sum(r["n_neg_pairs"] == 0 for r in res)), "sec": time.time() - t0}
    C.jdump(out, os.path.join(C.OUT, f"v_nu_{a.log}.json"))
    C.log(f"{a.log}: nu over {len(res)} complaints: min {min(nu)} median {np.median(nu):.1f} max {max(nu)}; MILP optimal {out['n_milp_optimal']}/{len(res)}, "
          f"two methods agree {out['n_methods_agree']}/{len(res)}; cancellation-free {out['n_cancellation_free']} ({out['sec']:.0f}s)", fh)


if __name__ == "__main__":
    main()
