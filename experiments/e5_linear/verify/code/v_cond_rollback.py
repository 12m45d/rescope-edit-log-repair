"""Verification, tasks (1) and (2): contributions at the complaint key, scaled Gram matrix, gamma, Gamma, share of negative
pairs; deletions of the rules and the bounds of the conditioning proposition.  Own code; reads NO file of e5_linear/out.

usage: python v_cond_rollback.py --log TB-Q100-ae-l2_3 [--workers 8]      -> verify/out/v_cond_rollback_<log>.json

Complaints: every REPAIR-pool D1 complaint of the log is computed; the SAMPLE (flag in_sample) is the seeded sample of 60
(30 image + 30 text) that the verification reports; the rest serves the exact comparison of medians over all complaints.

Per complaint and key definition ('last' = the last prompt key from the final key cache, ONE key; 'stacked' = the A keys of
the answer-predicting positions from the Track C capture, stacked: contributions are d_out x A matrices, Frobenius norms):
  coordinates  c_i = <psi_i, k> (SIGNED; stacked: c_i = K psi_i in R^A)
  contribution w_i = u_i c_i^T, held EXPLICITLY as a vector of length d_out * A (no Gram identity is used)
  G_ij = <w_i, w_j> (explicit product of the contribution vectors, signed), Ghat = G scaled to unit diagonal
  gamma <= Gamma: extreme eigenvalues of Ghat, by two routes: (A) scipy.linalg.eigvalsh(Ghat, driver = 'evr');
               (B) squares of the extreme singular values of the row-normalised contribution matrix (scipy.linalg.svdvals)
  frac_neg     share of the T (T - 1) / 2 unordered pairs i < j with <w_i, w_j> < 0
  D = sum_i w_i (damage), residual of a kept set r = sum_{kept} w_i, criterion ||r||_2 <= c ||D||_2 (Def. def:repairs(i))
Rules (each is run once to the end; the residual is recomputed from the kept set at every step; the number of deletions at
tolerance c is the first step at which the criterion holds):
  lf  delete by decreasing ||w_i|| (ties: lower index first)                               [Prop. prop:cond]
  fo  delete a kept edit maximising <r, w_i> for the CURRENT residual r (ties: lower index) [Thm. thm:rollback-hard(c)]
  bi  delete a kept edit minimising ||r - w_i|| (ties: lower index)                        [Thm. thm:rollback-hard(c)]
Bounds (mu(x) = least number of largest-norm deletions that leave sum_{kept} ||w_i||^2 <= x):
  lb_mu     mu(c^2 ||D||^2 / gamma) <= OPT_c            (technical report, Prop. prop:app:cond, needs gamma > 0)
  lb1       max(1, lb_mu)                               (OPT_c >= 1 for c < 1 and D != 0): the report's 'Prop. bound'
  lf_guar   mu(c^2 ||D||^2 / Gamma) >= lf               (the same proposition)
  lb_stmt   lf at tolerance c sqrt(Gamma / gamma)       (what Prop. prop:cond AS DISPLAYED in sec_theory.tex gives for
            OPT_c: 'LF at tolerance c' deletes at most OPT_{c' sqrt(gamma / Gamma)}' with c' = c sqrt(Gamma / gamma))
  stmt_check  min over the rules of the deletions at tolerance c sqrt(gamma / Gamma) (an upper bound of OPT there) >= lf(c):
            the displayed proposition is not contradicted
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import scipy.linalg as sla
import torch


_S = {}


def mu(n2, x):
    s = np.sort(np.asarray(n2, dtype=np.float64))[::-1]
    for k in range(len(s) + 1):
        if float(s[k:].sum()) <= x:
            return k
    return len(s)


def trajectory(Wc, rule, nrm):
    """Deletion order of a rule and ||r_k||, k = 0 .. T (r_k = residual after k deletions, recomputed from the kept set)."""
    T = Wc.shape[0]
    kept = np.ones(T)
    order, rn = [], []
    lf_order = np.argsort(-nrm, kind="stable")
    n2 = nrm * nrm
    for k in range(T + 1):
        r = kept @ Wc
        rn.append(float(np.sqrt(r @ r)))
        if k == T:
            break
        if rule == "lf":
            i = int(lf_order[k])
        else:
            s = Wc @ r                                    # <r, w_i>
            if rule == "fo":
                i = int(np.argmax(np.where(kept > 0, s, -np.inf)))
            else:                                         # bi: ||r - w_i||^2 = ||r||^2 - 2 <r, w_i> + ||w_i||^2
                i = int(np.argmin(np.where(kept > 0, n2 - 2.0 * s, np.inf)))
        kept[i] = 0.0
        order.append(i)
    return order, np.asarray(rn)


def count(rn, thr):
    """First k with ||r_k|| <= thr, and the relative margins of the decision (stop step, step before)."""
    k = int(np.argmax(rn <= thr))                         # rn[T] = 0 <= thr always
    m_stop = float((thr - rn[k]) / thr) if thr > 0 else None
    m_prev = float((rn[k - 1] - thr) / thr) if k > 0 and thr > 0 else None
    return k, m_stop, m_prev


def one(args):
    c, kl, Kst, in_sample = args
    U, Psi, CD = _S["U"], _S["Psi"], _S["CD"]
    out = {k: c[k] for k in ("group", "ptype", "probe_idx", "uid")}
    out.update(in_sample=bool(in_sample), A=int(c["n_ans_tok"]), keydefs={})
    T = U.shape[0]
    iu = np.triu_indices(T, 1)
    for kd, K in (("last", kl[None, :]), ("stacked", Kst)):
        K = np.asarray(K, dtype=np.float64)
        A = K.shape[0]
        Cm = Psi @ K.T                                    # (T, A) signed coordinates
        Wc = (U[:, :, None] * Cm[:, None, :]).reshape(T, -1)          # explicit contributions, (T, d_out * A)
        nrm = np.sqrt((Wc * Wc).sum(1))
        st = {"A": int(A), "n_zero_contributions": int((nrm == 0).sum()), "n_neg_coord": int((Cm[:, 0] < 0).sum()) if A == 1 else None,
              "min_abs_coord": float(np.abs(Cm).min())}
        G = Wc @ Wc.T
        G = 0.5 * (G + G.T)
        D = Wc.sum(0)
        Dn = float(np.sqrt(D @ D))
        st.update(D_norm=Dn, sum_w2=float((nrm * nrm).sum()), energy=float(Dn * Dn / (nrm * nrm).sum()),
                  D2_gram_rel_dev=float(abs(G.sum() - Dn * Dn) / (Dn * Dn)))
        if kd == "last":
            Dapp = CD @ K[0]
            st["fac_vs_applied"] = float(np.linalg.norm(Dapp - D) / Dn)
        Gh = G / np.outer(nrm, nrm)
        evA = sla.eigvalsh(Gh, driver="evr")
        sv = sla.svdvals(Wc / nrm[:, None])
        gA, GA = float(evA[0]), float(evA[-1])
        gB, GB = float(sv[-1] ** 2), float(sv[0] ** 2)
        off = G[iu]
        st.update(gamma=gA, Gamma=GA, ratio=GA / gA, gamma_svd=gB, Gamma_svd=GB, ratio_svd=GB / gB,
                  routes_rel_dev=float(max(abs(gA - gB) / gB, abs(GA - GB) / GB)), tol_factor=float(np.sqrt(gA / GA)),
                  frac_neg=float((off < 0).mean()), n_neg_pairs=int((off < 0).sum()), n_zero_pairs=int((off == 0).sum()),
                  n_pairs=int(off.size), min_cos=float(Gh[iu].min()), norm_spread=float(nrm.max() / nrm.min()))
        traj = {r: trajectory(Wc, r, nrm) for r in ("lf", "fo", "bi")}
        st["tol"] = {}
        n2 = nrm * nrm
        for t in C.TOLS:
            thr = t * Dn
            q = {"c": t}
            for r in ("lf", "fo", "bi"):
                k, ms, mp_ = count(traj[r][1], thr)
                q[r] = k
                q[r + "_margin_stop"], q[r + "_margin_prev"] = ms, mp_
            q["lb_mu"] = mu(n2, thr * thr / gA) if gA > 0 else 0
            q["lb1"] = max(1, q["lb_mu"])
            q["lf_guar"] = mu(n2, thr * thr / GA)
            up, dn = t * np.sqrt(GA / gA), t * np.sqrt(gA / GA)
            q["lb_stmt"] = count(traj["lf"][1], up * Dn)[0]
            q["tol_scaled_down"] = float(dn)
            q["rules_at_scaled_down"] = {r: count(traj[r][1], dn * Dn)[0] for r in ("lf", "fo", "bi")}
            q["stmt_check"] = bool(min(q["rules_at_scaled_down"].values()) >= q["lf"])
            q["sandwich_ok"] = bool(q["lb_mu"] <= min(q["lf"], q["fo"], q["bi"]) and q["lf"] <= q["lf_guar"] and q["lb_stmt"] <= q["lb_mu"])
            st["tol"][str(t)] = q
        st["full_delete"] = {r: {str(t): bool(st["tol"][str(t)][r] == T) for t in C.TOLS} for r in ("lf", "fo", "bi")}
        out["keydefs"][kd] = st
    return out


def _init(name):
    torch.set_num_threads(2)
    F = C.factors(name)
    _S.update(U=F["U"], Psi=F["Psi"], CD=C.composite_delta(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.log not in C.LOGS:
        raise SystemExit("only the two development logs")
    t0 = time.time()
    fh = open(os.path.join(C.LOGDIR, f"v_cond_rollback_{a.log}.log"), "a")
    rows = C.probe_rows(a.log)
    cand = C.candidates(rows)
    samp = C.sample60(cand)
    hchk = C.harness_check(a.log, rows, cand, samp)
    comps = cand["image"] + cand["text"]
    in_s = {c["group"] for c in samp}
    tc = C.trackc_keys(a.log)
    Kl, kinfo = C.last_keys(a.log, comps, tc=tc)
    _, kinfo_s = C.last_keys(a.log, samp, tc=tc)
    F = C.factors(a.log)
    U, Psi = F["U"], F["Psi"]
    C.log(f"{a.log}: {len(comps)} REPAIR D1 complaints ({len(cand['image'])} image, {len(cand['text'])} text); sample {len(samp)} "
          f"({sum(c['ptype'] == 'image' for c in samp)} image); harness check {hchk}; key cache vs Track C bitwise "
          f"{kinfo['n_bitwise_equal_trackc']}/{len(comps)}; lp_shift {kinfo['n_lp_shift']}", fh)
    # log level: the factor form against the applied weight difference, the cosine matrix of the write vectors
    CD = C.composite_delta(a.log)
    Dfac = U.T @ Psi
    un = np.sqrt((U * U).sum(1))
    Uh = U / un[:, None]
    cosU = Uh @ Uh.T
    cosU = 0.5 * (cosU + cosU.T)
    evU = sla.eigvalsh(cosU, driver="evr")
    svU = sla.svdvals(Uh)
    iu = np.triu_indices(U.shape[0], 1)
    loglevel = {"T": int(U.shape[0]), "d_out": int(U.shape[1]), "d_in": int(Psi.shape[1]), "n_batches": int(F["batch"].max()) + 1,
                "module_dtype": F["dtype"], "factor_dtype": "float64",
                "delta_fac_vs_applied_fro": float(np.linalg.norm(Dfac - CD) / np.linalg.norm(CD)),
                "gamma_U": float(evU[0]), "Gamma_U": float(evU[-1]), "ratio_U": float(evU[-1] / evU[0]),
                "gamma_U_svd": float(svU[-1] ** 2), "Gamma_U_svd": float(svU[0] ** 2),
                "share_pairs_cos_u_negative": float((cosU[iu] < 0).mean()), "min_cos_u": float(cosU[iu].min())}
    del CD, Dfac
    jobs = [(c, Kl[j].numpy(), tc[(c["probe_idx"], c["ptype"])]["K"].numpy(), c["group"] in in_s) for j, c in enumerate(comps)]
    # 'spawn': a forked child of a process that has used torch's OpenMP pool hangs in its first parallel region
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.log,)) as pool:
        res = pool.map(one, jobs, chunksize=4)
    order = {c["group"]: j for j, c in enumerate(samp)}
    out = {"definition": __doc__, "log": a.log, "tols": list(C.TOLS), "split_seed": C.SPLIT_SEED, "sample_seed": C.SAMPLE_SEED,
           "n_complaints": len(comps), "n_by_ptype": {p: len(cand[p]) for p in C.PTYPES},
           "sample_groups": [c["group"] for c in samp], "sample_by_ptype": {p: sum(c["ptype"] == p for c in samp) for p in C.PTYPES},
           "harness_check": hchk, "keys_all": kinfo, "keys_sample": kinfo_s, "log_level": loglevel,
           "rows": res, "sample_order": order, "sec": time.time() - t0,
           "threads": {"workers": a.workers, "blas_per_worker": os.environ.get("OPENBLAS_NUM_THREADS")},
           "linear_algebra": "numpy / scipy fp64 (OpenBLAS LAPACK): eigvalsh driver evr, svdvals (gesdd)"}
    C.jdump(out, os.path.join(C.OUT, f"v_cond_rollback_{a.log}.json"))
    for sel, rr in (("sample", [r for r in res if r["in_sample"]]), ("all", res)):
        v = [r["keydefs"]["last"] for r in rr]
        C.log(f"  last, {sel} (n = {len(v)}): gamma {C.med([x['gamma'] for x in v]):.4f} Gamma {C.med([x['Gamma'] for x in v]):.3f} ratio "
              f"{C.med([x['ratio'] for x in v]):.3f}; frac_neg {C.med([x['frac_neg'] for x in v]):.4f}; "
              + "; ".join(f"c={t}: lf {C.med([x['tol'][str(t)]['lf'] for x in v]):.1f} fo {C.med([x['tol'][str(t)]['fo'] for x in v]):.1f} "
                          f"bi {C.med([x['tol'][str(t)]['bi'] for x in v]):.1f} lb1 {C.med([x['tol'][str(t)]['lb1'] for x in v]):.1f} "
                          f"lb_stmt {C.med([x['tol'][str(t)]['lb_stmt'] for x in v]):.1f}" for t in C.TOLS), fh)
    C.log(f"wrote v_cond_rollback_{a.log}.json ({out['sec']:.0f}s)", fh)


if __name__ == "__main__":
    main()
