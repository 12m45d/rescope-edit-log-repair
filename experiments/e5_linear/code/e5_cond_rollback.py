"""E5 tasks 1 and 2 (rules and bounds), one log: conditioning / cancellation of the contributions at the complaint
keys, and rollback by rule against the bounds of the conditioning proposition.

usage: python e5_cond_rollback.py --log TB-Q100-ae-l2_3 [--workers 8]      -> out/cond_rollback_<log>.json   (CPU, fp64)

Per complaint (ALL REPAIR-pool D1 complaints, e5_common.complaints) and per key definition ('last' = the last prompt
position, ONE key; 'stacked' = the A answer-predicting positions stacked, Frobenius norms):
  G_ij = <w_i, w_j> (signed), Ghat = G scaled to unit diagonal, gamma <= Gamma = extreme eigenvalues of Ghat, ratio
  Gamma / gamma, energy = ||D||^2 / sum_i ||w_i||^2 (gamma <= energy <= Gamma), frac_neg = share of the T (T - 1) / 2
  pairs with <w_i, w_j> < 0, min_cos = the most obtuse pairwise cosine, cancellation_free = no negative pair,
  nu = vertex cover number of the negative graph (EXACT: T minus the largest set of edits with pairwise non-negative
  inner products, branch and bound), norm_spread = max ||w_i|| / min ||w_i||, n_neg_coord = number of edits with
  <psi_i, k> < 0 ('last' only), fac_vs_applied = ||(W_E - W_0) K^T - D|| / ||D|| (size of the write round-off that
  the factor form omits).
  For c in {0.1, 0.3, 0.5} (criterion ||r||^2 <= c^2 ||D||^2, p = 2): deletions of
    lf         largest-norm-first,
    fo         first-order greedy of the paper: delete a kept edit maximising <r, w_i> for the CURRENT residual r,
    bi         best-improvement greedy: delete a kept edit minimising ||r - w_i||,
    fo_static  Track C's 'first-order greedy': fixed order by decreasing <D, w_i> (reported for continuity only),
  best_rule = the least of the four; lb_cond = mu(c^2 ||D||^2 / gamma) <= OPT_c (Prop. prop:cond, gamma > 0; mu(x) =
  least number of largest-norm deletions that leave sum ||w_i||^2 <= x); lf_guar = mu(c^2 ||D||^2 / Gamma) >= lf;
  lb_tri = min{k : sum of the k largest ||w_i|| >= (1 - c) ||D||} <= OPT_c (triangle inequality); lb = max of the two.
Log level: gamma, Gamma of the cosine matrix of the write vectors u_i.  At ONE key w_i = c_i u_i, so Ghat is that
matrix with rows and columns i multiplied by sign(c_i): the same spectrum for every complaint (checked: 'last_identity').
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch

import e5_common as C
import e5_linalg as LA

_W = {}


def _init(name):
    torch.set_num_threads(1)
    L = C.load_log(name)
    U = L.U.numpy()
    _W.update(U=U, W=L.W, UU=U @ U.T, CD=C.composite_delta(name))


def one(args):
    c, K = args
    U, W, UU, CD = _W["U"], _W["W"], _W["UU"], _W["CD"]
    Kd = K.double()
    out = {k: c[k] for k in ("group", "ptype", "probe_idx", "uid")}
    out["A"] = int(K.shape[0])
    out["first_mismatch_pos"] = c.get("first_mismatch_pos")
    out["keydefs"] = {}
    for kd in C.KEYDEFS:
        Kk = Kd[:1] if kd == "last" else Kd
        Cm = (W @ Kk.T).numpy()                           # (T, A): c_i = K psi_i, signed
        G = LA.gram(UU, Cm)
        st = LA.cond_stats(G)
        st["A"] = int(Kk.shape[0])
        st["min_abs_coord"] = float(np.abs(Cm).min())
        if kd == "last":
            st["n_neg_coord"] = int((Cm[:, 0] < 0).sum())
        Dfac = U.T @ Cm                                   # (d_out, A) factor-form damage
        Dapp = (CD @ Kk.T).numpy()
        st["D_norm"] = float(np.linalg.norm(Dfac))
        st["fac_vs_applied"] = float(np.linalg.norm(Dapp - Dfac) / np.linalg.norm(Dfac))
        st["D2_gram_vs_direct_rel"] = float(abs(st["D2"] - st["D_norm"] ** 2) / st["D_norm"] ** 2)
        if not st["n_zero_contributions"]:
            st.update(LA.vertex_cover_number(G))
            st["tol"] = {str(t): LA.rollback_rules(G, t, st["gamma"], st["Gamma"]) for t in C.TOLS}
        out["keydefs"][kd] = st
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    C.check_log(a.log)
    t0 = time.time()
    fh = open(os.path.join(C.LOGDIR, f"cond_rollback_{a.log}.log"), "a")
    comps = C.complaints(a.log)
    keys = C.complaint_keys(a.log, comps)
    kchk = C.keycache_check(a.log, comps, keys)
    C.log(f"{a.log}: {len(comps)} REPAIR D1 complaints ({sum(c['ptype'] == 'image' for c in comps)} image, "
          f"{sum(c['ptype'] == 'text' for c in comps)} text); last key bitwise equal to the key cache "
          f"{kchk['n_last_key_bitwise_equal']}/{len(comps)}; lp_shift rows {kchk['n_lp_shift']}", fh)
    L = C.load_log(a.log)
    U = L.U.numpy()
    UU = U @ U.T
    un = np.sqrt(np.diag(UU))
    evU = np.linalg.eigvalsh(UU / np.outer(un, un))
    iu = np.triu_indices(L.T, 1)
    cosU = (UU / np.outer(un, un))[iu]
    loglevel = {"T": int(L.T), "d_in": int(L.d), "d_out": int(L.m), "n_batches": int(L.batch.max()) + 1,
                "gamma_U": float(evU[0]), "Gamma_U": float(evU[-1]), "ratio_U": float(evU[-1] / evU[0]),
                "u_norm": C.dist(un.tolist()), "share_pairs_cos_u_negative": float((cosU < 0).mean()),
                "min_cos_u": float(cosU.min()), "median_cos_u": float(np.median(cosU)),
                "note": "cosine matrix of the write vectors u_i (signed); at one key Ghat has this spectrum for every complaint"}
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.log,)) as pool:
        rows = pool.map(one, [(c, keys[c["group"]]) for c in comps], chunksize=4)
    dev = max(max(abs(r["keydefs"]["last"]["gamma"] - loglevel["gamma_U"]) / loglevel["gamma_U"],
                  abs(r["keydefs"]["last"]["Gamma"] - loglevel["Gamma_U"]) / loglevel["Gamma_U"]) for r in rows)
    loglevel["last_identity_max_rel_dev"] = float(dev)
    out = {"definition": __doc__, "log": a.log, "backbone": C.LOGS[a.log]["backbone"], "tols": list(C.TOLS),
           "n_complaints": len(rows), "n_by_ptype": {p: sum(r["ptype"] == p for r in rows) for p in C.PTYPES},
           "keycache_check": kchk, "log_level": loglevel, "rows": rows, "sec": time.time() - t0,
           "threads": {"workers": a.workers, "blas_per_worker": os.environ.get("OPENBLAS_NUM_THREADS")}}
    C.jdump(out, os.path.join(C.OUT, f"cond_rollback_{a.log}.json"))
    for kd in C.KEYDEFS:
        v = [r["keydefs"][kd] for r in rows]
        C.log(f"  {kd}: Gamma/gamma median {np.median([x['ratio'] for x in v]):.2f}; frac_neg median "
              f"{np.median([x['frac_neg'] for x in v]):.3f}; cancellation-free {sum(x['cancellation_free'] for x in v)}/{len(v)}; "
              f"nu median {np.median([x['nu'] for x in v]):.0f}; "
              + "; ".join(f"c={t}: lf {np.median([x['tol'][str(t)]['lf'] for x in v]):.0f} fo {np.median([x['tol'][str(t)]['fo'] for x in v]):.0f} "
                          f"bi {np.median([x['tol'][str(t)]['bi'] for x in v]):.0f} lb {np.median([x['tol'][str(t)]['lb_cond'] for x in v]):.0f}"
                          for t in C.TOLS), fh)
    C.log(f"wrote cond_rollback_{a.log}.json ({out['sec']:.0f}s)", fh)


if __name__ == "__main__":
    main()
