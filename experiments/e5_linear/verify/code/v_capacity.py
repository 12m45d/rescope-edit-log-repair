"""Verification, task (3): the measured capacity of whole-sequence protection.  Own code; reads NO file of e5_linear/out.

usage: python v_capacity.py --log TB-Q100-ae-l2_3 [--no-svd]       -> verify/out/v_capacity_<log>.json      (CPU, fp64)

Definition (PROTOCOL / harness).  Protection group of edit i = the keys of EVERY position of its own request sequence
(prompt + target answer, teacher forced, B = 1).  Family: the n smallest groups, by (sequence length, edit id).  A complaint
with keys K_q stays REPAIRABLE under the protection of the n smallest groups iff (I - P_n) K_q keeps full column rank, P_n =
orthogonal projector onto the span of the protected keys; test of the harness COMP solve: every singular value of
(I - P_n) K_q exceeds 1e-9 * max_j ||k_q,j||.  Measured capacity n_cap = the largest such n.  Complaint keys: 'last' = the
last prompt key (final key cache, ONE key); 'stacked' = the A answer-predicting keys (Track C capture).

Keys of the edit sequences: 07_repair_v3/harness/cache/editkeys_*.pt (the only cache with every position; module dtype, cast
to fp64).  Cross-check with an independent capture: the key at the last prompt position of every edit sequence against the
final key cache (10_final/cache/<backbone>/keys_tf.pt, rows of kind 'rel'), bit for bit, with the sequence lengths.

Algorithm (not the incremental Gram-Schmidt of the measuring agent, and not a pivoted QR):
  1. EXACT de-duplication.  The keys of the sequences are visited smallest sequence first; a key whose float values are bit
     for bit those of an earlier key is dropped (it adds nothing to the span).  m_n = number of distinct keys among the n
     smallest sequences, N_n = number of keys.
  2. ONE Householder QR (LAPACK dgeqrf, no pivoting, fp64) of the d x M matrix A of the distinct keys in the order of first
     appearance (M = m_n at the first n with m_n >= d).  If the first m columns of A are numerically independent (every
     |R_jj|, j < m, far above round-off: reported), the first m columns of Q span exactly the keys of the n smallest sequences,
     m = m_n, and for a complaint key k:  ||(I - P_n) k|| = ||(Q^T k)[m:]||; for stacked keys the singular values of
     (I - P_n) K_q are those of (Q^T K_q)[m:, :].  Q^T K_q by LAPACK dormqr.  So ONE factorisation gives every n.
  3. Numerical rank as the harness defines it (singular values > max(shape) * eps * s_1 of the stacked keys WITH duplicates):
     LAPACK singular values of the distinct keys at n_free (the largest n with m_n < d) and at n_free + 1, with s_1 of the
     matrix with duplicates by power iteration.  sigma_i(with duplicates) >= sigma_i(distinct keys), and exact duplicates
     add exact zeros, so the numerical rank is the number of singular values of the distinct keys above the cut.
     (--no-svd skips this step.)
Law of Thm. thm:capacity (a = 1): T - s*_Z = the largest n with z + (sum of the n smallest r_i - z) <= d_in - a, r_i = numerical
rank of group i (LAPACK singular values, cut max(shape) * eps * s_1), for z = 0, z = z_bitwise (leading positions whose key is
bit for bit the same in all T sequences) and z = z_near (leading positions whose keys differ from those of edit 0 by at most
10 % in relative norm in every sequence: the definition used by the measuring agent).
Extra (same projector test, one protection set each): only the last prompt key of every edit (K_last), only the
answer-predicting keys of every edit (K_ans).
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import argparse
import hashlib
import json
import os
import time

import numpy as np
import scipy.linalg as sla
import torch
from scipy.linalg import lapack



def num_rank(s, shape):
    return int((s > max(shape) * C.EPS * s[0]).sum()) if len(s) and s[0] > 0 else 0


def kept_pred(ranks, d, a, z):
    tot, n = z, 0
    for r in sorted(x - z for x in ranks):
        if tot + r <= d - a:
            tot += r
            n += 1
        else:
            break
    return n


def power_s0(A, w, iters=400, seed=0):
    """Largest singular value of the columns of A repeated w_j times: sqrt(lambda_max(A diag(w) A^T))."""
    v = np.random.default_rng(seed).standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    lam = prev = 0.0
    for _ in range(iters):
        u = A @ (w * (A.T @ v))
        prev, lam = lam, float(np.linalg.norm(u))
        v = u / lam
    return float(np.sqrt(lam)), abs(lam - prev) / lam


def point(blocks, d, Kl, kl_norm, Kst, Ast, stmax):
    """One protection set: rank, orthonormal basis (economic QR), escapes and repairability of all complaints."""
    Kc = np.concatenate(blocks, 0).astype(np.float64)                    # (n_keys, d)
    s = sla.svdvals(Kc)
    r = num_rank(s, Kc.shape)
    if r != Kc.shape[0]:
        raise RuntimeError("point: the protected keys are not of full rank; the economic QR below would be wrong")
    Q, _ = sla.qr(Kc.T, mode="economic")

    def perp(X):
        X = X - Q @ (Q.T @ X)
        return X - Q @ (Q.T @ X)
    el = np.linalg.norm(perp(Kl.T), axis=0) / kl_norm
    rk = []
    for j in range(len(Kst)):
        sv = sla.svdvals(perp(Kst[j].T))
        rk.append(int((sv > C.TOL_REL * stmax[j]).sum()))
    return {"n_keys": int(Kc.shape[0]), "rank": int(r), "sigma_min_over_max": float(s[-1] / s[0]),
            "escape_last": el.tolist(), "rank_kept_stacked": rk,
            "n_feasible_last": int((el > C.TOL_REL).sum()), "n_feasible_stacked": int(sum(x == a for x, a in zip(rk, Ast)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--no-svd", action="store_true")
    a = ap.parse_args()
    name = a.log
    if name not in C.LOGS:
        raise SystemExit("only the two development logs")
    torch.set_num_threads(2)
    t0 = time.time()
    fh = open(os.path.join(C.LOGDIR, f"v_capacity_{name}.log"), "a")
    cfg = C.LOGS[name]
    F = C.factors(name)
    uids, T, d = F["uids"], F["U"].shape[0], F["Psi"].shape[1]
    tag = hashlib.sha1(json.dumps([cfg["backbone"], cfg["layer"], list(uids)], sort_keys=True, default=str).encode()).hexdigest()[:12]
    if tag not in cfg["editkeys"]:
        raise RuntimeError("the edit-key cache does not belong to the edit stream of the log")
    ek = torch.load(os.path.join(C.HARNESS, "cache", cfg["editkeys"]), map_location="cpu")
    lens = [int(x) for x in ek["len"]]
    last = [int(x) for x in ek["last"]]
    if not all(ek["K"][i].shape == (lens[i], d) for i in range(T)):
        raise RuntimeError("edit-key blocks with a wrong shape")
    # ---------------------------------------------------------------- cross-check with the final key cache
    eidx = C.keycache_edit_index(name, T)
    Ktf = C.keys_tf(name)
    xc = {"n_edits": T, "n_uid_len_lastpos_equal": 0, "n_last_key_bitwise_equal": 0, "n_answer_positions_consistent": 0}
    for i in range(T):
        e = eidx[i]
        xc["n_uid_len_lastpos_equal"] += int(e["edit_uid"] == uids[i] and int(e["len_tf"]) == lens[i] and int(e["lp_tf"]) == last[i])
        xc["n_last_key_bitwise_equal"] += int(torch.equal(Ktf[int(e["row_tf"])], ek["K"][i][last[i]].float()))
        pp = [int(x) for x in ek["pp"][i]]
        xc["n_answer_positions_consistent"] += int(pp[0] == last[i] and pp == list(range(last[i], last[i] + len(pp))) and pp[-1] == lens[i] - 2)
    C.log(f"{name}: T {T}, d_in {d}; edit-key cache {cfg['editkeys']} (tag ok); against the final key cache: {xc}", fh)
    # ---------------------------------------------------------------- groups: sizes, ranks, prefix
    order = sorted(range(T), key=lambda i: (lens[i], i))
    ranks, smin = [], []
    for i in range(T):
        s = sla.svdvals(ek["K"][i].double().numpy())
        ranks.append(num_rank(s, (lens[i], d)))
        smin.append(float(s[-1] / s[0]))
    minlen = min(lens)
    z_bit = 0
    while z_bit < minlen and all(torch.equal(ek["K"][0][z_bit], ek["K"][i][z_bit]) for i in range(T)):
        z_bit += 1
    near, distinct_at_pos = [], []
    for p in range(min(minlen, 40)):
        ref = ek["K"][0][p].double()
        near.append(max(float((ek["K"][i][p].double() - ref).norm() / ref.norm()) for i in range(T)))
        distinct_at_pos.append(len({(ek["K"][i][p].float() + 0.0).numpy().tobytes() for i in range(T)}))
    z_near = 0
    while z_near < len(near) and near[z_near] <= 0.1:
        z_near += 1
    law = {}
    for zn, z in (("z0", 0), ("z_bitwise", z_bit), ("z_near", z_near)):
        kp = kept_pred(ranks, d, 1, z)
        law[zn] = {"z": z, "kept_pred": kp, "sstar": T - kp}
    C.log(f"group sizes {min(lens)}..{max(lens)} (median {np.median(lens):.0f}); ranks = sizes for {sum(r == n for r, n in zip(ranks, lens))}/{T}; "
          f"z bitwise {z_bit}, near {z_near}; distinct keys at the leading positions {distinct_at_pos[:20]}; law {law} ({time.time() - t0:.0f}s)", fh)
    # ---------------------------------------------------------------- 1. exact de-duplication, smallest sequence first
    seen, cols, mult_at = {}, [], []
    m_n, N_n, classes_n = [0], [0], [0]
    first_key_class = {}
    occ = []                                                     # per sequence: indices of its keys among the distinct ones
    for i in order:
        Kf = (ek["K"][i].float() + 0.0).numpy()                  # + 0.0: -0.0 and 0.0 are the same value
        idx = []
        for p in range(lens[i]):
            h = hashlib.blake2b(Kf[p].tobytes(), digest_size=16).digest()
            j = seen.get(h)
            if j is None:
                j = seen[h] = len(cols)
                cols.append((i, p))
            idx.append(j)
        occ.append(np.asarray(idx))
        first_key_class.setdefault(idx[0], len(first_key_class))
        m_n.append(len(cols))
        N_n.append(N_n[-1] + lens[i])
        classes_n.append(len(first_key_class))
    n_free = max(n for n in range(T + 1) if m_n[n] < d)          # the largest n whose distinct keys cannot fill the key space
    n_sat = n_free + 1 if n_free < T else None
    n_top = n_sat if n_sat is not None else T
    M = m_n[n_top]
    C.log(f"distinct keys: all {T} sequences {m_n[T]} of {N_n[T]}; n_free = {n_free} (distinct {m_n[n_free]}, keys {N_n[n_free]}), "
          f"n_sat = {n_sat} (distinct {M}); first-position classes among the n_free smallest: {classes_n[n_free]}", fh)
    # ---------------------------------------------------------------- 2. one Householder QR of the distinct keys
    A = np.empty((d, M), dtype=np.float64, order="F")
    for j, (i, p) in enumerate(cols[:M]):
        A[:, j] = ek["K"][i][p].double().numpy()
    colnorm = np.linalg.norm(A, axis=0)
    w_free = np.zeros(M)
    w_sat = np.zeros(M)
    for n in range(1, n_top + 1):
        np.add.at(w_sat, occ[n - 1], 1.0)
        if n <= n_free:
            np.add.at(w_free, occ[n - 1], 1.0)
    s0_free, conv_free = power_s0(A[:, : m_n[n_free]], w_free[: m_n[n_free]])
    s0_sat, conv_sat = power_s0(A, w_sat)
    del ek["img"]
    lw, info = lapack.dgeqrf_lwork(d, M)
    tq = time.time()
    qr, tau, _, info = lapack.dgeqrf(A, lwork=int(lw.real), overwrite_a=1)
    if info != 0:
        raise RuntimeError("dgeqrf failed")
    k = min(d, M)
    rdiag = np.abs(np.diag(qr[:k, :k]).copy())
    C.log(f"QR of {d} x {M} in {time.time() - tq:.0f}s; |R_jj| / s_1: min over j < m(n_free) {rdiag[: m_n[n_free]].min() / s0_free:.3e}, "
          f"min over j < {k} {rdiag.min() / s0_sat:.3e}; s_1 with duplicates {s0_free:.4f} (n_free), {s0_sat:.4f} (n_sat)", fh)
    # ---------------------------------------------------------------- complaints
    rows = C.probe_rows(name)
    cand = C.candidates(rows)
    samp = C.sample60(cand)
    comps = cand["image"] + cand["text"]
    tc = C.trackc_keys(name)
    Kl, kinfo = C.last_keys(name, comps, Ktf=Ktf, tc=tc)
    Kl = Kl.double().numpy()                                             # (nC, d)
    Kst = [tc[(c["probe_idx"], c["ptype"])]["K"].double().numpy() for c in comps]   # (A_j, d)
    Ast = [int(x.shape[0]) for x in Kst]
    sl = np.concatenate([[0], np.cumsum(Ast)]).astype(int)
    kl_norm = np.linalg.norm(Kl, axis=1)
    stmax = [float(np.linalg.norm(x, axis=1).max()) for x in Kst]
    nC = len(comps)
    rng = np.random.default_rng(C.CAP_SEED)
    five = [samp[int(i)]["group"] for i in sorted(rng.choice(len(samp), C.N_CAP, replace=False))]
    Call = np.asfortranarray(np.concatenate([Kl] + Kst, 0).T)           # (d, nC + sum A)
    wq, info = lapack.dormqr("L", "T", qr[:, :k], tau, Call, -1)[1:]
    cq, _, info = lapack.dormqr("L", "T", qr[:, :k], tau, Call, int(wq[0].real))
    if info != 0:
        raise RuntimeError("dormqr failed")
    back = lapack.dormqr("L", "N", qr[:, :k], tau, cq, int(wq[0].real))[0]
    q_roundtrip = float(np.abs(back - Call).max() / np.abs(Call).max())  # Q Q^T K_q = K_q: Q is orthogonal to round-off
    del back
    cl = cq[:, :nC]                                                      # Q^T k (last keys)
    tail2 = np.concatenate([np.cumsum((cl * cl)[::-1], axis=0)[::-1], np.zeros((1, nC))], 0)   # tail2[m] = sum_{j >= m} c_j^2
    esc_last, rk_st = [], []
    for n in range(n_top + 1):
        m = min(m_n[n], d)
        esc_last.append((np.sqrt(tail2[m]) / kl_norm).tolist())
        r = []
        for j in range(nC):
            if m >= d:
                r.append(0)
                continue
            sv = sla.svdvals(cq[m:, nC + sl[j]: nC + sl[j + 1]])
            r.append(int((sv > C.TOL_REL * stmax[j]).sum()))
        rk_st.append(r)
    feas_l = np.array(esc_last) > C.TOL_REL                              # (n_top + 1, nC)
    feas_s = np.array(rk_st) == np.array(Ast)[None, :]
    ncap_l = [int(max(n for n in range(n_top + 1) if feas_l[n, j])) for j in range(nC)]
    ncap_s = [int(max(n for n in range(n_top + 1) if feas_s[n, j])) for j in range(nC)]
    mono_l = [bool(all(feas_l[n, j] or not feas_l[n + 1, j] for n in range(n_top))) for j in range(nC)]
    mono_s = [bool(all(feas_s[n, j] or not feas_s[n + 1, j] for n in range(n_top))) for j in range(nC)]
    C.log(f"{nC} complaints: n_cap (last) {sorted(set(ncap_l))}, (stacked) {sorted(set(ncap_s))}; monotone {sum(mono_l)}/{nC}, {sum(mono_s)}/{nC}; "
          f"escape of the last key at n_free: median {np.median(esc_last[n_free]):.4f} min {min(esc_last[n_free]):.4f}; five: {five}", fh)
    out = {"definition": __doc__, "log": name, "T": T, "d_in": d, "tol_rel": C.TOL_REL, "cap_seed": C.CAP_SEED,
           "edit_keys": {"file": cfg["editkeys"], "tag": tag, "len": lens, "group_rank": ranks, "group_sigma_min_over_max_min": min(smin),
                         "n_groups_full_rank": int(sum(r == n for r, n in zip(ranks, lens))), "order_smallest_first": order,
                         "z_bitwise": z_bit, "z_near": z_near, "near_prefix_max_rel_diff": near[: z_near + 1],
                         "distinct_keys_at_leading_positions": distinct_at_pos, "n_keys_all_groups": int(sum(lens)),
                         "n_distinct_keys_all_groups": int(m_n[T]), "cross_check_final_key_cache": xc},
           "law": law,
           "family": {"n": list(range(n_top + 1)), "edit": [None] + order[:n_top], "sum_size": N_n[: n_top + 1], "n_distinct": m_n[: n_top + 1],
                      "rank": [min(x, d) for x in m_n[: n_top + 1]], "n_first_position_classes": classes_n[: n_top + 1],
                      "n_free": n_free, "n_sat": n_sat, "qr_shape": [d, M], "q_roundtrip_max_rel": q_roundtrip,
                      "min_rdiag_over_s1_at_n_free": float(rdiag[: m_n[n_free]].min() / s0_free), "min_rdiag_over_s1_at_n_sat": float(rdiag.min() / s0_sat),
                      "s1_with_duplicates": {"n_free": s0_free, "n_sat": s0_sat, "power_iteration_last_rel_change": [conv_free, conv_sat]},
                      "min_col_norm": float(colnorm.min()), "max_col_norm": float(colnorm.max())},
           "complaints": {"group": [c["group"] for c in comps], "ptype": [c["ptype"] for c in comps], "A": Ast,
                          "in_sample": [c["group"] in {s["group"] for s in samp} for c in comps], "keys": kinfo,
                          "n_cap_last": ncap_l, "n_cap_stacked": ncap_s, "monotone_last": mono_l, "monotone_stacked": mono_s,
                          "escape_last": esc_last, "rank_kept_stacked": rk_st},
           "five": five, "sample_groups": [s["group"] for s in samp]}
    # ---------------------------------------------------------------- extra: K_last and K_ans (one protection set each)
    pts = {}
    for pname in ("K_last", "K_ans"):
        if pname == "K_last":
            blocks = [ek["K"][i][last[i]][None].double().numpy() for i in range(T)]
        else:
            blocks = [ek["K"][i][ek["pp"][i]].double().numpy() for i in range(T)]
        pts[pname] = point(blocks, d, Kl, kl_norm, Kst, Ast, stmax)
        C.log(f"point {pname}: {pts[pname]['n_keys']} keys, rank {pts[pname]['rank']}; repairable last {pts[pname]['n_feasible_last']}/{nC}, "
              f"stacked {pts[pname]['n_feasible_stacked']}/{nC}", fh)
    out["points"] = pts
    C.jdump(out, os.path.join(C.OUT, f"v_capacity_{name}.json"))
    # ---------------------------------------------------------------- 3. numerical rank at n_free and n_free + 1
    if not a.no_svd:
        spec = {}
        for lab, n, s0 in (("n_free", n_free, s0_free), ("n_sat", n_sat, s0_sat)):
            if n is None:
                continue
            m = m_n[n]
            ts = time.time()
            R = np.triu(qr[: min(d, m), :m])
            s = sla.svdvals(R, overwrite_a=True, check_finite=False)
            cut = max(d, N_n[n]) * C.EPS * s0
            spec[lab] = {"n": n, "n_keys": N_n[n], "n_distinct": m, "s1_distinct": float(s[0]), "s_min_distinct": float(s[-1]),
                         "s1_with_duplicates": s0, "cut": cut, "numerical_rank": int((s > cut).sum()),
                         "s_min_over_cut": float(s[-1] / cut), "smallest_5": s[-5:].tolist(), "sec": time.time() - ts}
            C.log(f"spectrum at n = {n}: {m} distinct keys, s_1 {s[0]:.4f} (with duplicates {s0:.4f}), s_min {s[-1]:.3e}, cut {cut:.3e}, "
                  f"numerical rank {spec[lab]['numerical_rank']} ({time.time() - ts:.0f}s)", fh)
            del R
        out["spectrum"] = spec
    out["sec"] = time.time() - t0
    out["threads"] = {"blas": os.environ.get("OPENBLAS_NUM_THREADS"), "torch": torch.get_num_threads()}
    C.jdump(out, os.path.join(C.OUT, f"v_capacity_{name}.json"))
    C.log(f"wrote v_capacity_{name}.json ({out['sec']:.0f}s)", fh)


if __name__ == "__main__":
    main()
