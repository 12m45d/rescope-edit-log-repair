"""Independent recomputation of the headline cells of E5 with other code paths (CPU, fp64) -> out/verify_<log>.json.

usage: python e5_verify.py --log TB-Q100-ae-l2_3 [--threads 2] [--no-capacity]

(A) Conditioning and rollback WITHOUT the Gram identity: the contributions are materialised as vectors
    w_i = vec(u_i (K psi_i)^T) in R^(d_out A); gamma and Gamma are the squared extreme singular values of the matrix of
    the normalised contributions (LAPACK SVD instead of eigvalsh of the Gram matrix); negative pairs from explicit
    inner products; LF / FO / BI with explicit residual vectors and Euclidean norms; the lower bound from the sorted
    squared norms.  Key 'last': every complaint; key 'stacked': a seeded sample of N_STACKED complaints.  Compared with
    out/cond_rollback_<log>.json.
(B) Capacity WITHOUT the incremental Gram-Schmidt: the keys of the n smallest whole request sequences are stacked into
    one matrix; its numerical rank from LAPACK singular values at n = kept and n = kept + 1 (cut max(shape) eps s_1), and
    at n = kept an orthonormal basis of its column space from a QR factorisation with column pivoting, with which the
    escape ||(I - P) k_q|| / ||k_q|| of the last prompt key of every complaint is recomputed.  Compared with
    out/capacity_<log>_seq.json.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import scipy.linalg as sla
import torch

import e5_common as C

N_STACKED = 40
SEED = 1


def explicit(U, Cm, tols):
    T = U.shape[0]
    Wc = (U[:, :, None] * Cm[:, None, :]).reshape(T, -1)                 # (T, d_out * A)
    nrm = np.linalg.norm(Wc, axis=1)
    sv = np.linalg.svd(Wc / nrm[:, None], compute_uv=False)
    gamma, Gamma = float(sv[-1] ** 2), float(sv[0] ** 2)
    ip = Wc @ Wc.T
    iu = np.triu_indices(T, 1)
    D = Wc.sum(0)
    Dn = float(np.linalg.norm(D))
    sq = np.sort(nrm ** 2)[::-1]
    out = {"gamma": gamma, "Gamma": Gamma, "frac_neg": float((ip[iu] < 0).mean()),
           "min_cos": float((ip / np.outer(nrm, nrm))[iu].min()), "energy": Dn ** 2 / float((nrm ** 2).sum()), "tol": {}}
    for c in tols:
        thr = c * Dn
        order = np.argsort(-nrm, kind="stable")
        r, lf = D.copy(), 0
        while np.linalg.norm(r) > thr and lf < T:
            r = r - Wc[order[lf]]
            lf += 1
        res = {"lf": lf}
        for rule in ("fo", "bi"):
            kept = np.ones(T, bool)
            r, k = D.copy(), 0
            while np.linalg.norm(r) > thr and k < T:
                idx = np.nonzero(kept)[0]
                if rule == "fo":
                    j = idx[int(np.argmax(Wc[idx] @ r))]
                else:
                    j = idx[int(np.argmin(((r[None, :] - Wc[idx]) ** 2).sum(1)))]
                kept[j] = False
                r = Wc[kept].sum(0) if kept.any() else np.zeros_like(D)   # residual recomputed from the kept vectors
                k += 1
            res[rule] = k
        x = (thr ** 2) / gamma
        s, lb = float(sq.sum()), 0
        while s > x and lb < T:                                          # delete the largest squared norms first
            s = float(sq[lb + 1:].sum())
            lb += 1
        res["lb_cond"] = lb
        out["tol"][str(c)] = res
    return out


def part_a(name):
    L = C.load_log(name)
    U = L.U.numpy()
    comps = C.complaints(name)
    keys = C.complaint_keys(name, comps)
    ref = {r["group"]: r for r in C.jload(os.path.join(C.OUT, f"cond_rollback_{name}.json"))["rows"]}
    rng = np.random.default_rng(SEED)
    sample = {comps[int(i)]["group"] for i in rng.choice(len(comps), min(N_STACKED, len(comps)), replace=False)}
    out = {}
    for kd in C.KEYDEFS:
        cs = comps if kd == "last" else [c for c in comps if c["group"] in sample]
        st = {"n": len(cs), "max_rel_dev_gamma": 0.0, "max_rel_dev_Gamma": 0.0, "max_abs_dev_frac_neg": 0.0, "max_abs_dev_min_cos": 0.0,
              "max_rel_dev_energy": 0.0, "n_equal": {str(t): {k: 0 for k in ("lf", "fo", "bi", "lb_cond")} for t in C.TOLS}, "mismatches": []}
        med = {"ratio": [], "frac_neg": [], **{f"lf_over_lb_{t}": [] for t in C.TOLS}}
        for c in cs:
            K = keys[c["group"]].double()
            Kk = K[:1] if kd == "last" else K
            e = explicit(U, (L.W @ Kk.T).numpy(), C.TOLS)
            r = ref[c["group"]]["keydefs"][kd]
            st["max_rel_dev_gamma"] = max(st["max_rel_dev_gamma"], abs(e["gamma"] - r["gamma"]) / r["gamma"])
            st["max_rel_dev_Gamma"] = max(st["max_rel_dev_Gamma"], abs(e["Gamma"] - r["Gamma"]) / r["Gamma"])
            st["max_abs_dev_frac_neg"] = max(st["max_abs_dev_frac_neg"], abs(e["frac_neg"] - r["frac_neg"]))
            st["max_abs_dev_min_cos"] = max(st["max_abs_dev_min_cos"], abs(e["min_cos"] - r["min_cos"]))
            st["max_rel_dev_energy"] = max(st["max_rel_dev_energy"], abs(e["energy"] - r["energy"]) / r["energy"])
            med["ratio"].append(e["Gamma"] / e["gamma"])
            med["frac_neg"].append(e["frac_neg"])
            for t in C.TOLS:
                for k in ("lf", "fo", "bi", "lb_cond"):
                    eq = e["tol"][str(t)][k] == r["tol"][str(t)][k]
                    st["n_equal"][str(t)][k] += int(eq)
                    if not eq:
                        st["mismatches"].append([c["group"], t, k, e["tol"][str(t)][k], r["tol"][str(t)][k]])
                med[f"lf_over_lb_{t}"].append(e["tol"][str(t)]["lf"] / max(1, e["tol"][str(t)]["lb_cond"]))
        st["recomputed_medians"] = {k: float(np.median(v)) for k, v in med.items()}
        out[kd] = st
    return out


def part_b(name):
    ref = C.jload(os.path.join(C.OUT, f"capacity_{name}_seq.json"))
    f = ref["family"]
    kept = int(min(min(f["n_cap_last"]), min(f["n_cap_stacked"])))
    ek = torch.load(os.path.join(C.HARNESS, "cache", C.LOGS[name]["editkeys"]), map_location="cpu")
    T = len(ek["K"])
    order = sorted(range(T), key=lambda i: (int(ek["len"][i]), i))
    assert order == ref["edit_keys"]["order_smallest_first"]
    comps = C.complaints(name)
    keys = C.complaint_keys(name, comps)
    kq = np.stack([keys[c["group"]][0].double().numpy() for c in comps], 1)          # (d, n_complaints) last prompt keys
    out = {"kept": kept}
    for n in (kept, kept + 1):
        t0 = time.time()
        M = torch.cat([ek["K"][i] for i in order[:n]], 0).double().numpy().T.copy()      # (d, N)
        s = sla.svdvals(M)
        rank = int((s > max(M.shape) * C.EPS * s[0]).sum())
        rec = {"n_keys": int(M.shape[1]), "rank_svd": rank, "rank_stored": f["rank"][f["n"].index(n)], "s_max": float(s[0]),
               "s_min_kept": float(s[rank - 1]), "s_first_dropped": float(s[rank]) if rank < len(s) else None, "sec_svd": time.time() - t0}
        if n == kept:
            t1 = time.time()
            Q, R, _ = sla.qr(M, mode="economic", pivoting=True)
            dg = np.abs(np.diag(R))
            rq = int((dg > max(M.shape) * C.EPS * dg[0]).sum())
            Qr = Q[:, :rank]
            P = kq - Qr @ (Qr.T @ kq)
            P = P - Qr @ (Qr.T @ P)
            esc = np.linalg.norm(P, axis=0) / np.linalg.norm(kq, axis=0)
            st = np.array(f["escape_last"][f["n"].index(n)])
            rec.update(rank_qr_diag=rq, escape_last_median=float(np.median(esc)), escape_last_min=float(esc.min()),
                       escape_stored_median=float(np.median(st)), max_rel_dev_escape=float(np.max(np.abs(esc - st) / st)),
                       n_feasible_last=int((esc > C.TOL_REL).sum()), n_feasible_last_stored=f["n_feasible_last"][f["n"].index(n)],
                       sec_qr=time.time() - t1)
            del Q, R
        out[f"n={n}"] = rec
        del M
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-capacity", action="store_true")
    a = ap.parse_args()
    C.check_log(a.log)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    out = {"definition": __doc__, "log": a.log, "conditioning_rollback": part_a(a.log)}
    C.log(f"{a.log} (A): " + "; ".join(f"{kd}: n {v['n']}, gamma dev {v['max_rel_dev_gamma']:.1e}, Gamma dev {v['max_rel_dev_Gamma']:.1e}, "
                                       f"mismatching counts {len(v['mismatches'])}" for kd, v in out["conditioning_rollback"].items()))
    if not a.no_capacity:
        out["capacity"] = part_b(a.log)
        C.log(f"{a.log} (B): " + str({k: ({kk: vv for kk, vv in v.items() if kk in ('rank_svd', 'rank_stored', 'rank_qr_diag', 'max_rel_dev_escape', 'n_feasible_last')}
                                          if isinstance(v, dict) else v) for k, v in out["capacity"].items()}))
    out["sec"] = time.time() - t0
    C.jdump(out, os.path.join(C.OUT, f"verify_{a.log}.json"))


if __name__ == "__main__":
    main()
