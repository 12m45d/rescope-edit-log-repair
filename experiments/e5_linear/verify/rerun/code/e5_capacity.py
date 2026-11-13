"""E5 task 3, one log: the capacity of exact protection (Thm. thm:capacity) measured on the key space of the log.

usage: python e5_capacity.py --log TB-Q100-ae-l2_3 --group seq|need [--threads 2]   -> out/capacity_<log>_<group>.json

Protection group of edit i (keys of its own request sequence: prompt + target answer, teacher forced at B = 1; harness
cache of HLog.edit_keys, module dtype, cast to fp64):
  group 'seq'   every position of the sequence (the harness protection K_seq and the RQ3 files),
  group 'need'  the needed set of Thm. app:preserve: positions 0 .. len - 2 (prompt + all answer tokens but the last).
r_i = numerical rank of the group (singular values > max(shape) * eps * s_1).
Shared prefix: z_bitwise = number of leading positions whose key is bitwise the same in all T sequences; z_near = number
of leading positions whose keys differ from those of edit 0 by at most 10 % (relative norm) in every sequence (the RQ3
definitions); prefix classes = sets of edits whose first z_near keys are bitwise equal.
Family (as RQ3 / harness K_seq:<n>): the n smallest groups, by (sequence length, edit id).  The protected span is
built incrementally in fp64 (block Gram-Schmidt, re-orthogonalised; a direction counts if its singular value exceeds
max(d, N_n) * eps * s0_n, s0_n = largest singular value of the N_n stacked keys, by power iteration): rank(n).
Complaint (ALL REPAIR-pool D1 complaints): complaint keys K_q = 'last' (the last prompt key, A = 1) or 'stacked' (the A
answer-predicting keys); target V = -(W_E - W_0) K_q, a = rank V.  A complaint stays REPAIRABLE under a protection with
projector P iff K_perp = (I - P) K_q keeps full column rank, tested as in the harness COMP solve: every singular value of
K_perp > 1e-9 * max_j ||k_q,j|| (then an admissible correction exists for every target; V has rank A for every complaint
here, see 'a_rank').  n_cap = the largest n of the family at which the complaint is repairable (measured capacity);
T - n_cap edits lose their protection.
Law (Thm. thm:capacity under GP/Z): the number of protectable edits is T - s*_Z = the largest n with
z + (sum of the n smallest r_i - z) <= d_in - a; rank law: rank(n) = min(d_in, z + sum_{i <= n} (r_i - z)); both for
z in {0, z_bitwise, z_near}.
Points (group 'seq' run only): protecting only the edits' answer keys (K_ans: the answer-predicting positions of every
edit sequence) or only their last prompt keys (K_last): rank of the union, group ranks, law (z = 0), repairable complaints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

import e5_common as C

EPS = C.EPS


def sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def num_rank(M: torch.Tensor) -> int:
    s = torch.linalg.svdvals(M)
    return int((s > max(M.shape) * EPS * float(s[0])).sum()) if s.numel() and float(s[0]) > 0 else 0


def orth(Kc: torch.Tensor):
    """Harness _orth: orthonormal basis of span(Kc) (d, m): QR + SVD of R, rank cut max(shape) * eps * s0."""
    Q1, R1 = torch.linalg.qr(Kc)
    Ur, s, _ = torch.linalg.svd(R1)
    r = int((s > max(Kc.shape) * EPS * float(s[0])).sum())
    return Q1 @ Ur[:, :r], r


def spectral_top(G, iters, v=None):
    if v is None:
        g = torch.Generator().manual_seed(0)
        v = torch.randn(G.shape[0], dtype=G.dtype, generator=g)
    for _ in range(iters):
        v = G @ v
        v = v / v.norm()
    return float((v @ (G @ v)).sqrt()), v


def kept_pred(ranks, d, a, z) -> int:
    """T - s*_Z: the largest n with z + (sum of the n smallest r_i - z) <= d - a."""
    s, n = z, 0
    for r in sorted(x - z for x in ranks):
        if s + r <= d - a:
            s += r
            n += 1
        else:
            break
    return n


class Complaints:
    """Complaint keys of a log, stacked column-wise (d, m_total), with the feasibility test of both key definitions."""

    def __init__(self, name):
        self.comps = C.complaints(name)
        keys = C.complaint_keys(name, self.comps)
        self.Kq = [keys[c["group"]].double().T.contiguous() for c in self.comps]          # (d, A_c)
        self.A = [int(K.shape[1]) for K in self.Kq]
        self.sl = np.concatenate([[0], np.cumsum(self.A)]).astype(int).tolist()
        self.all = torch.cat(self.Kq, 1)
        self.kqn_max = [float(K.norm(dim=0).max()) for K in self.Kq]
        self.kq_last = [float(K[:, 0].norm()) for K in self.Kq]
        self.kq_fro = [float(K.norm()) for K in self.Kq]
        CD = C.composite_delta(name)
        self.a_rank = [num_rank(CD @ K) for K in self.Kq]                                # rank of V = -(W_E - W_0) K_q
        self.a_rank_last = [int(float((CD @ K[:, :1]).norm()) > 0) for K in self.Kq]
        self.kq_rank = [num_rank(K) for K in self.Kq]
        del CD

    def test(self, Kp_all):
        """Per complaint: rank kept of the stacked block, escape of the last key, Frobenius escape of the block."""
        rk, esc_last, esc_fro = [], [], []
        for j in range(len(self.comps)):
            Kp = Kp_all[:, self.sl[j]:self.sl[j + 1]]
            _, R1 = torch.linalg.qr(Kp)
            s = torch.linalg.svdvals(R1)
            rk.append(int((s > C.TOL_REL * self.kqn_max[j]).sum()))
            esc_last.append(float(Kp[:, 0].norm()) / self.kq_last[j])
            esc_fro.append(float(Kp.norm()) / self.kq_fro[j])
        return rk, esc_last, esc_fro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--group", default="seq", choices=("seq", "need"))
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    C.check_log(a.log)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    fh = open(os.path.join(C.LOGDIR, f"capacity_{a.log}_{a.group}.log"), "a")
    L = C.load_log(a.log)
    d, T = L.d, L.T
    cfg = C.LOGS[a.log]
    tag = sha([cfg["backbone"], cfg["layer"], list(L.meta["uids"])])
    if tag not in cfg["editkeys"]:
        raise RuntimeError(f"edit-key cache {cfg['editkeys']} does not belong to the edit stream of {a.log} (tag {tag})")
    ek = torch.load(os.path.join(C.HARNESS, "cache", cfg["editkeys"]), map_location="cpu")
    lens = [int(x) for x in ek["len"]]
    assert all(ek["K"][i].shape[0] == lens[i] for i in range(T))
    assert all(int(ek["pp"][i][0]) == int(ek["last"][i]) and int(ek["pp"][i][-1]) == lens[i] - 2 for i in range(T))

    def grp(i):
        return ek["K"][i] if a.group == "seq" else ek["K"][i][: lens[i] - 1]

    gsize = [int(grp(i).shape[0]) for i in range(T)]
    order = sorted(range(T), key=lambda i: (lens[i], i))                    # harness K_seq order
    C.log(f"{a.log} group {a.group}: d_in {d}, T {T}; group sizes min/median/max {min(gsize)}/{int(np.median(gsize))}/{max(gsize)}", fh)
    # ------------------------------------------------------------ per-edit protection ranks, shared prefix
    ranks = [num_rank(grp(i).double()) for i in range(T)]
    z_bit = 0
    while z_bit < min(gsize) and all(torch.equal(ek["K"][0][z_bit], ek["K"][i][z_bit]) for i in range(T)):
        z_bit += 1
    near = []
    for p in range(min(gsize)):
        ref = ek["K"][0][p].double()
        near.append(max(float((ek["K"][i][p].double() - ref).norm() / ref.norm()) for i in range(T)))
        if near[-1] > 0.1:
            break
    z_near = int(sum(1 for x in near if x <= 0.1))
    classes = {}
    for i in range(T):
        for cid, mem in classes.items():
            if all(torch.equal(ek["K"][i][p], ek["K"][mem[0]][p]) for p in range(z_near)):
                mem.append(i)
                break
        else:
            classes[len(classes)] = [i]
    cls_of = {i: cid for cid, mem in classes.items() for i in mem}
    prefix_rank = num_rank(ek["K"][0][:z_near].double()) if z_near else 0
    n_bit_pos0 = int(sum(torch.equal(ek["K"][0][0], ek["K"][i][0]) for i in range(T)))
    edit_keys = {"group": a.group, "len": lens, "group_size": gsize, "group_rank": ranks,
                 "group_full_rank": int(sum(r == m for r, m in zip(ranks, gsize))), "order_smallest_first": order,
                 "z_bitwise": z_bit, "z_near": z_near, "near_prefix_max_rel_diff": near[: z_near + 1],
                 "prefix_rank_of_edit0": prefix_rank, "n_seq_bitwise_equal_to_seq0_at_pos0": n_bit_pos0,
                 "n_prefix_classes": len(classes), "prefix_class_sizes": [len(m) for m in classes.values()],
                 "prefix_class_along_order": [cls_of[i] for i in order], "n_keys_all_groups": int(sum(gsize)),
                 "answer_len": [int(len(ek["pp"][i])) for i in range(T)], "editkeys_file": cfg["editkeys"], "editkeys_tag": tag}
    C.log(f"group ranks min/median/max {min(ranks)}/{int(np.median(ranks))}/{max(ranks)}, full rank {edit_keys['group_full_rank']}/{T}; "
          f"z bitwise {z_bit}, near {z_near} (rank {prefix_rank}), prefix classes {len(classes)} sizes {edit_keys['prefix_class_sizes']} "
          f"({time.time() - t0:.0f}s)", fh)
    # ------------------------------------------------------------ complaints
    cq = Complaints(a.log)
    nC = len(cq.comps)
    C.log(f"{nC} complaints; stacked keys {cq.sl[-1]}; rank V == A for {sum(x == y for x, y in zip(cq.a_rank, cq.A))}/{nC}; "
          f"rank K_q == A for {sum(x == y for x, y in zip(cq.kq_rank, cq.A))}/{nC}", fh)
    out = {"definition": __doc__, "log": a.log, "group": a.group, "d_in": d, "d_out": L.m, "T": T, "edit_keys": edit_keys,
           "complaints": {"group": [c["group"] for c in cq.comps], "ptype": [c["ptype"] for c in cq.comps], "A": cq.A,
                          "a_rank_stacked": cq.a_rank, "a_rank_last": cq.a_rank_last, "kq_rank": cq.kq_rank},
           "tol_rel": C.TOL_REL, "threads": a.threads}
    # ------------------------------------------------------------ points: K_last, K_ans (once per log)
    if a.group == "seq":
        pts = {}
        for pname in ("K_last", "K_ans"):
            if pname == "K_last":
                blocks = [ek["K"][i][ek["last"][i]][None] for i in range(T)]
            else:
                blocks = [ek["K"][i][ek["pp"][i]] for i in range(T)]
            granks = [num_rank(b.double()) for b in blocks]
            Kc = torch.cat(blocks, 0).double().T.contiguous()
            Q, r = orth(Kc)
            Kp = cq.all - Q @ (Q.T @ cq.all)
            Kp = Kp - Q @ (Q.T @ Kp)
            rk, el, ef = cq.test(Kp)
            pts[pname] = {"n_keys": int(Kc.shape[1]), "rank": int(r), "group_rank": granks, "sum_group_rank": int(sum(granks)),
                          "group_size_min_max": [min(int(b.shape[0]) for b in blocks), max(int(b.shape[0]) for b in blocks)],
                          "z": 0, "rank_pred_gp": int(min(d, sum(granks))),
                          "kept_pred": {"last": kept_pred(granks, d, 1, 0),
                                        "stacked": [kept_pred(granks, d, A, 0) for A in cq.A]},
                          "rank_kept_stacked": rk, "escape_last": el, "escape_fro": ef,
                          "feasible_last": [bool(e > C.TOL_REL) for e in el],
                          "feasible_stacked": [bool(x == A) for x, A in zip(rk, cq.A)]}
            C.log(f"point {pname}: {Kc.shape[1]} keys, rank {r}; repairable complaints last {sum(pts[pname]['feasible_last'])}/{nC}, "
                  f"stacked {sum(pts[pname]['feasible_stacked'])}/{nC}; escape of the last key median {np.median(el):.3f} min {min(el):.3f}", fh)
            del Kc, Q, Kp
        out["points"] = pts
    # ------------------------------------------------------------ the smallest-first family, incremental
    Qb = torch.zeros((d, d), dtype=torch.float64)
    rk_tot = 0
    G = torch.zeros((d, d), dtype=torch.float64)
    Kp_all = cq.all.clone()
    vpow = None
    Nn = srank = 0
    rk0, el0, ef0 = cq.test(Kp_all)
    fam = {"n": [0], "edit": [None], "rank": [0], "new_dims": [0], "sum_size": [0], "sum_rank": [0], "s0": [None],
           "rank_kept_stacked": [rk0], "escape_last": [el0], "escape_fro_median": [float(np.median(ef0))], "sec": [0.0]}
    for n, i in enumerate(order, 1):
        ts = time.time()
        B = grp(i).double().T.contiguous()                                  # (d, r_i)
        G.addmm_(B, B.T)
        Nn += B.shape[1]
        srank += ranks[i]
        s0, vpow = spectral_top(G, 20 if vpow is not None else 60, vpow)
        Qc = Qb[:, :rk_tot]
        R = B - Qc @ (Qc.T @ B)
        R = R - Qc @ (Qc.T @ R)
        Ur, s, _ = torch.linalg.svd(R, full_matrices=False)
        keep = s > max(d, Nn) * EPS * s0
        Qn = Ur[:, keep]
        if Qn.shape[1]:
            Qn = Qn - Qc @ (Qc.T @ Qn)
            Qn, _ = torch.linalg.qr(Qn)
            if rk_tot + Qn.shape[1] > d:                                    # cannot happen for an orthonormal family
                raise RuntimeError("basis larger than the key dimension")
            Qb[:, rk_tot:rk_tot + Qn.shape[1]] = Qn
            Kp_all = Kp_all - Qn @ (Qn.T @ Kp_all)
        rk_tot += int(Qn.shape[1])
        Qc = Qb[:, :rk_tot]
        Kp_all = Kp_all - Qc @ (Qc.T @ Kp_all)                              # re-orthogonalise against the whole basis
        rk, el, ef = cq.test(Kp_all)
        fam["n"].append(n)
        fam["edit"].append(int(i))
        fam["rank"].append(rk_tot)
        fam["new_dims"].append(int(Qn.shape[1]))
        fam["sum_size"].append(Nn)
        fam["sum_rank"].append(srank)
        fam["s0"].append(s0)
        fam["rank_kept_stacked"].append(rk)
        fam["escape_last"].append(el)
        fam["escape_fro_median"].append(float(np.median(ef)))
        fam["sec"].append(time.time() - ts)
        C.log(f"n={n} edit {i} size {B.shape[1]} rank {rk_tot} (+{Qn.shape[1]}; sum r {srank}) repairable last "
              f"{sum(e > C.TOL_REL for e in el)}/{nC} stacked {sum(x == A for x, A in zip(rk, cq.A))}/{nC} "
              f"escape(last) median {np.median(el):.3g} ({time.time() - ts:.0f}s)", fh)
        if rk_tot >= d:
            break
    orth_err = float((Qb[:, :rk_tot].T @ Qb[:, :rk_tot] - torch.eye(rk_tot, dtype=torch.float64)).abs().max())
    del G, Qb, Kp_all
    ns = fam["n"]
    feas_last = [[bool(fam["escape_last"][k][j] > C.TOL_REL) for k in range(len(ns))] for j in range(nC)]
    feas_stk = [[bool(fam["rank_kept_stacked"][k][j] == cq.A[j]) for k in range(len(ns))] for j in range(nC)]
    fam["n_cap_last"] = [max(ns[k] for k in range(len(ns)) if f[k]) for f in feas_last]
    fam["n_cap_stacked"] = [max(ns[k] for k in range(len(ns)) if f[k]) for f in feas_stk]
    fam["monotone_last"] = [all(f[k] or not f[k + 1] for k in range(len(ns) - 1)) for f in feas_last]
    fam["monotone_stacked"] = [all(f[k] or not f[k + 1] for k in range(len(ns) - 1)) for f in feas_stk]
    fam["rank_kept_law_holds"] = [all(fam["rank_kept_stacked"][k][j] == min(cq.A[j], d - fam["rank"][k]) for k in range(len(ns)))
                                  for j in range(nC)]
    fam["n_feasible_last"] = [int(sum(f[k] for f in feas_last)) for k in range(len(ns))]
    fam["n_feasible_stacked"] = [int(sum(f[k] for f in feas_stk)) for k in range(len(ns))]
    fam["saturated_at_n"] = ns[-1] if fam["rank"][-1] >= d else None
    fam["orth_err_basis"] = orth_err
    # ------------------------------------------------------------ the law
    rs = [ranks[i] for i in order]
    ncls = [len({cls_of[i] for i in order[:n]}) for n in ns]
    pred = {}
    for zname, z in (("z0", 0), ("z_bitwise", z_bit), ("z_near", z_near)):
        rp = [min(d, z + sum(r - z for r in rs[:n])) if n else 0 for n in ns]
        pred[zname] = {"z": z, "rank_pred": rp, "n_rank_equal": int(sum(x == y for x, y in zip(rp, fam["rank"]))),
                       "max_abs_dev": int(max(abs(x - y) for x, y in zip(rp, fam["rank"]))),
                       "kept_pred_last": kept_pred(ranks, d, 1, z),
                       "kept_pred_stacked": [kept_pred(ranks, d, A, z) for A in cq.A],
                       "sstar_last": T - kept_pred(ranks, d, 1, z)}
    rpc = [min(d, sum(rs[:n]) - z_near * (n - c)) if n else 0 for n, c in zip(ns, ncls)]
    pred["z_near_per_prefix_class"] = {"z": z_near, "rank_pred": rpc, "n_classes_among_first_n": ncls,
                                       "n_rank_equal": int(sum(x == y for x, y in zip(rpc, fam["rank"]))),
                                       "max_abs_dev": int(max(abs(x - y) for x, y in zip(rpc, fam["rank"]))),
                                       "note": "rank law if every prefix class contributes its own copy of the z_near prefix keys"}
    fam["pred"] = pred
    out["family"] = fam
    out["sec"] = time.time() - t0
    C.jdump(out, os.path.join(C.OUT, f"capacity_{a.log}_{a.group}.json"))
    C.log(f"measured capacity (n_cap): last {sorted(set(fam['n_cap_last']))}, stacked {sorted(set(fam['n_cap_stacked']))}; "
          f"law kept z_near (a = 1): {pred['z_near']['kept_pred_last']}, z0: {pred['z0']['kept_pred_last']}; saturation at n = "
          f"{fam['saturated_at_n']}; wrote capacity_{a.log}_{a.group}.json ({out['sec']:.0f}s)", fh)


if __name__ == "__main__":
    main()
