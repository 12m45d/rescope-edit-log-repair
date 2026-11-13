"""E5 (f), amendment R20: capacity curve of linear certification.  One development log per process (GPU through the queue).

    gq run --mem 12500 --prio 2 -- python e5f_capacity.py TB-Q100-ae-l2_3        (19000 for the LLaVA log: model + fp64 basis)

CAPTURE  Under the pre-edit model (the module input does not depend on the edited weight), the keys of the edited module at ALL
  positions of the pre-edit trace (prompt + pre-edit answer, teacher-forced) of the first 60 D1 complaints of the REPAIR pool in
  3 seeded orders = the nested complaint streams of P12 (fx.draws.complaints(L, 60, seed), seeds 0 - 2), through the harness's
  own function of the all-positions compensation: methods._comp_query_keys(H, [c], "all", cfg)  (07_repair_v3/harness/methods.py).
  Check: for the ten complaints of seed 0 the number of query keys and the rank after projecting off K_ans with the harness's rule
  (QR, SVD, singular values > 1e-9 x the largest key norm) against p12_linear_streams/out/<log>/COMP_B_ALLPOS__n10__s0.json.
LINEAR ALGEBRA (float64; on the model's card for Qwen, on the CPU for LLaVA)
  rank K_Q(n): an orthonormal basis Qb of the row space is grown complaint by complaint: residual of the new keys after two
  projections off Qb, SVD of the residual, right singular vectors with singular value > TOL x (largest key norm of the block)
  are appended.  TOL = 1e-9 (the harness's tol_rel).  Stability: the numbers of residual singular values in (1e-10, 1e-9] and
  (1e-9, 1e-8] x scale are counted along the stream (0 and 0 = the rank is the same for every tolerance of these two decades).
  cap(n) = rank[K_Q(n), K_E] - rank K_Q(n) = number of singular values of (I - P_Q) K_E above TOL x the largest solver-key norm
  (reported for 1e-10, 1e-9, 1e-8; the T singular values are stored).
  PRESERVED EDITS.  D = W_E - W_0 (deployed minus pre-edit weight, fp64).  Every weight W_E + X with X K_Q = -D K_Q restores the n
  complaints at all positions.  Edit i is preserved if |X k_i| <= PRES x |D k_i| at its solver key k_i (PRES = 1e-6; also 1e-9, 1e-3).
    plain    X = -D P_Q : the minimum-norm restoring weight change with no regard to the edits (the certifying update closest to D)
    keep_ls  X = -D P_Q + B E^+, B = D P_Q K_E, E = (I - P_Q) K_E : the minimum-norm restoring change that keeps the module output at
             all T solver keys where that is possible (least squares where it is not)
    keep_greedy  the same for a largest independent subset J of the solver keys modulo col K_Q (column-pivoted QR of E, the greedy
             rule of the theorem); |J| = cap(n)
  THRESHOLDS-ONLY REPAIR of the deployed scoping (ReScope-W: role all positions, logged values, linear response) from its log-only
  thresholds, for the same complaints (every position of each trace):
    sacrificing  tau_i <- max(tau_i, just above the largest coordinate of edit i on the n traces) (library margins); edit i is
             silenced at its solver key / on its own request (last prompt position of its B = 1 Rel prompt) iff its coordinate
             there is below the repaired threshold
    protecting   the library's least_repair, one complaint at a time, on_conflict = 'refuse' (a complaint whose repair would silence
             a guard key is refused): refused complaints, edits silenced (0 by construction)
ND HYPOTHESIS of the theorem: rank of the T write vectors (float64 SVD, relative tolerance 1e-10) and smallest singular value /
  condition number of the T x T matrix of coordinates <w_i, k_j>.
"""
import hashlib

from e5common import *
import rescope
from rescope import least_repair
from rescope.repair import _above, MARGIN, ABS_MARGIN

GRID = (1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60)
TOL, PRES = 1e-9, 1e-6
NMAX = 60


def proj_off(Y, Qb):
    if Qb.shape[1] == 0:
        return Y
    for _ in range(2):
        Y = Y - (Y @ Qb) @ Qb.T
    return Y


def ortho_err(Qb, k=512):
    r = Qb.shape[1]
    if r == 0:
        return 0.0
    idx = torch.from_numpy(np.sort(np.random.default_rng(0).choice(r, min(k, r), replace=False))).to(Qb.device)
    G = Qb.T @ Qb[:, idx]
    G[idx, torch.arange(len(idx), device=Qb.device)] -= 1.0
    return float(G.abs().max())


def rel_counts(XK, DKn):
    rel = (XK.norm(dim=0) / DKn).cpu().numpy()
    return {"n_preserved": int((rel <= PRES).sum()), "n_preserved_tol_1e-9": int((rel <= 1e-9).sum()), "n_preserved_tol_1e-3": int((rel <= 1e-3).sum()),
            "rel_change_median": float(np.median(rel)), "rel_change_min": float(rel.min()), "rel_change_max": float(rel.max())}


def main(name):
    out_dir = f"{OUT}/f"
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    hz = rescope.harness()
    HM = hz.HM
    dev = torch.device(H.dev)
    if L.backbone == "llava" or os.environ.get("E5F_LA") == "cpu":      # LLaVA: the model leaves no room on a 17 GB reservation for the fp64 basis:
        dev = torch.device("cpu")                                       # the linear algebra runs on the CPU (as the harness's own solve device for LLaVA)
    rowmap = {(r["probe_idx"], r["ptype"]): r for r in H.rows}
    D = (H.WE.double() - H.W0.double()).to(dev)                         # (m, d)
    KE = L.log.K.to(dev)                                                 # (T, d)
    T, d = KE.shape
    DKE = D @ KE.T
    DKn = DKE.norm(dim=0)
    scaleE = float(KE.norm(dim=1).max())
    Mw = fx.methods.build("where_logonly", L, H)
    Sg = Mw.obj.fold().to(dev)
    tau0 = Sg.det_tau.clone()
    Wd, sd = Sg.det_w, Sg.det_s
    r_solver = ((KE @ Wd.T) / sd).diagonal().clone()
    K1, ed = L.own_keys()
    r_own = Sg.key_scores(torch.as_tensor(K1).to(dev))[torch.arange(T, device=dev), torch.as_tensor(ed).to(dev)].clone()
    # ---- ND hypothesis (fp64 SVD)
    U = L.log.U
    sU = torch.linalg.svdvals(U)
    C = L.log.W @ L.log.K.T                                              # C[i, j] = <w_i, k_j>
    sC = torch.linalg.svdvals(C)
    sD = torch.linalg.svdvals(DKE.cpu())
    nd = {"nd_tolerance": 1e-10, "nd_rank_write_vectors": int((sU > 1e-10 * sU[0]).sum()), "nd_write_vectors_smallest_singular_value_rel": float(sU[-1] / sU[0]),
          "nd_coordinate_matrix_smallest_singular_value": float(sC[-1]), "nd_coordinate_matrix_largest_singular_value": float(sC[0]),
          "nd_coordinate_matrix_condition_number": float(sC[0] / sC[-1]),
          "nd_holds": bool(int((sU > 1e-10 * sU[0]).sum()) == T and float(sC[-1]) > 1e-10 * float(sC[0])),
          "nd_rank_deployed_outputs_at_solver_keys": int((sD > 1e-10 * sD[0]).sum()),
          "nd_note": "write vectors = the logged residuals U_i (T x m); coordinate matrix <w_i, k_j> of the read-out vectors against the stored solver keys; "
                     "float64 SVD, a singular value counts if it exceeds 1e-10 x the largest; last entry: rank of (W_E - W_0) K_E, the deployed changes at the solver keys",
          "T": T, "d_in": d, "m_out": int(D.shape[0])}
    jdump(dict(nd, log=name), f"{out_dir}/{name}__nd.json")
    print("ND", nd, flush=True)
    for seed in (0, 1, 2):
        path = f"{out_dir}/{name}__s{seed}.json"
        if os.path.exists(path):
            continue
        t0 = time.time()
        stream = fx.draws.complaints(L, NMAX, seed)
        Qb = torch.zeros(d, 0, dtype=torch.float64, device=dev)
        per, grid, bands = [], [], {"in_1e-10_1e-9": 0, "in_1e-9_1e-8": 0, "in_1e-8_1e-6": 0}
        mmax = torch.full((T,), -float("inf"), dtype=torch.float64, device=dev)
        Sp, n_ref, n_keys, kq10 = Sg, 0, 0, []
        min_kept, max_dropped = float("inf"), 0.0
        for t, j in enumerate(stream):
            e = L.index[j]
            c = rowmap[(e["probe_idx"], e["ptype"])]
            with torch.no_grad():
                Kc = HM._comp_query_keys(H, [c], "all", {}).to(dev).T.contiguous()      # (n_i, d) fp64
            n_keys += Kc.shape[0]
            if seed == 0 and t < 10:
                kq10.append(Kc)
            sc = float(Kc.norm(dim=1).max())
            Y = proj_off(Kc, Qb)
            S, Vh = torch.linalg.svd(Y, full_matrices=False)[1:]
            keep = S > TOL * sc
            rs = (S / sc).cpu().numpy()
            bands["in_1e-10_1e-9"] += int(((rs > 1e-10) & (rs <= 1e-9)).sum())
            bands["in_1e-9_1e-8"] += int(((rs > 1e-9) & (rs <= 1e-8)).sum())
            bands["in_1e-8_1e-6"] += int(((rs > 1e-8) & (rs <= 1e-6)).sum())
            if bool(keep.any()):
                min_kept = min(min_kept, float(rs[keep.cpu().numpy()].min()))
            if bool((~keep).any()):
                max_dropped = max(max_dropped, float(rs[(~keep).cpu().numpy()].max()))
            room = d - Qb.shape[1]
            new = Vh[keep][:room].T
            if new.shape[1]:
                new = proj_off(new.T.contiguous(), Qb).T.contiguous()
                new = torch.linalg.qr(new)[0]
                Qb = torch.cat([Qb, new], 1)
            rc = ((Kc @ Wd.T) / sd).max(0).values                        # largest coordinate of every edit on the trace
            mmax = torch.maximum(mmax, rc)
            Sn, rec = least_repair(Sp, Kc, on_conflict="refuse")
            n_ref += int(rec["refused"])
            Sp = Sp if rec["refused"] else Sn
            per.append({"t": t, "sid": e["sid"], "ptype": e["ptype"], "n_keys": int(Kc.shape[0]), "rank_after": int(Qb.shape[1]), "n_new_directions": int(new.shape[1]),
                        "keys_sha1": hashlib.sha1(Kc.cpu().numpy().tobytes()).hexdigest()[:16], "refused_by_protecting_repair": bool(rec["refused"]),
                        "n_thresholds_raised": int(rec["n_raised"]), "n_edits_candidate_on_trace_logonly": int((rc >= tau0).sum())})
            n = t + 1
            del Y, S, Vh
            if seed == 0 and n == 10:                                    # the harness's rule on the P12 complaints
                Qp, bst = HM.protection_basis(H, ("K_ans",))
                Kq = torch.cat(kq10, 0).T.contiguous()
                Qpd = Qp.to(dev)
                Kp = Kq - Qpd @ (Qpd.T @ Kq)
                R1 = torch.linalg.qr(Kp, mode="r")[1]
                s = torch.linalg.svdvals(R1)
                rk = int((s > 1e-9 * float(Kq.norm(dim=0).max())).sum())
                stored = jload(f"{fx.FINAL}/../08_rescope/p12_linear_streams/out/{name}/COMP_B_ALLPOS__n10__s0.json")
                chk = {"n_query_keys": int(Kq.shape[1]), "stored_n_query_keys": stored["rank"]["n_query_keys"], "rank_after_projecting_K_ans": rk,
                       "stored_rank": stored["rank"]["correction"], "protection_rank": int(bst["rank"]), "stored_protection_rank": stored["rank"]["protection"],
                       "rank_K_Q_10_unprojected": int(Qb.shape[1]),
                       "match": bool(int(Kq.shape[1]) == stored["rank"]["n_query_keys"] and rk == stored["rank"]["correction"])}
                jdump(dict(chk, log=name), f"{out_dir}/{name}__p12check.json")
                print("P12 check", chk, flush=True)
                del Kq, Kp, R1, Qpd, kq10
                kq10 = []
                torch.cuda.empty_cache()
            if n in GRID:
                r = int(Qb.shape[1])
                PKE = Qb @ (Qb.T @ KE.T) if r else torch.zeros(d, T, dtype=torch.float64, device=dev)
                E = KE.T - PKE
                if r:
                    E = E - Qb @ (Qb.T @ E)                              # second pass of the projection (columns are vectors here)
                Ue, sE, Vhe = torch.linalg.svd(E, full_matrices=False)
                caps = {f"{tol:g}": int((sE > tol * scaleE).sum()) for tol in (1e-10, 1e-9, 1e-8)}
                cap = caps["1e-09"]
                B = D @ PKE
                g = {"n": n, "n_keys": n_keys, "rank_KQ": r, "cap": cap, "cap_by_tolerance": caps, "min_T_d_minus_rank": int(min(T, d - r)),
                     "sing_E_rel": [float(x) for x in (sE / scaleE).cpu().numpy()],
                     "basis_orthonormality_error_512_columns": ortho_err(Qb)}
                # plain minimum-norm restoring change
                g["plain"] = rel_counts(-B, DKn)
                # keep all solver keys (least squares where infeasible)
                kc = sE > TOL * scaleE
                if bool(kc.any()):
                    Ep = (Vhe[kc].T / sE[kc]) @ Ue[:, kc].T                # E^+ (T, d)
                    XK = -B + B @ (Ep @ KE.T)
                    nx2 = float((D @ Qb).norm() ** 2 + (B @ Ep).norm() ** 2) if r else 0.0
                else:
                    XK, nx2 = -B, float((D @ Qb).norm() ** 2) if r else 0.0
                g["keep_ls"] = dict(rel_counts(XK, DKn), norm_X_over_norm_D=float(nx2 ** 0.5 / D.norm()))
                # keep a largest independent subset (column-pivoted QR = the greedy rule)
                import scipy.linalg as sla
                _, Rp, piv = sla.qr(E.cpu().numpy(), mode="economic", pivoting=True)
                J = torch.as_tensor(np.sort(piv[:cap]).copy(), device=dev) if cap else torch.zeros(0, dtype=torch.long, device=dev)
                if cap:
                    EJ = E[:, J]
                    Uj, sj, Vhj = torch.linalg.svd(EJ, full_matrices=False)
                    EJp = (Vhj.T / sj) @ Uj.T                           # (|J|, d)
                    XKJ = -B + B[:, J] @ (EJp @ KE.T)
                    g["keep_greedy"] = dict(rel_counts(XKJ, DKn), n_selected=int(cap), cond_E_J=float(sj[0] / sj[-1]),
                                            norm_X_over_norm_D=float(((D @ Qb).norm() ** 2 + (B[:, J] @ EJp).norm() ** 2) ** 0.5 / D.norm()) if r else 0.0)
                    # restoration check on the last trace: (D + X) k = 0 for a restored key
                    Xk = -(D @ (Qb @ (Qb.T @ Kc.T))) + B[:, J] @ (EJp @ Kc.T)
                    g["keep_greedy"]["restoration_rel_err_last_trace"] = float(((D @ Kc.T + Xk).norm(dim=0) / (D @ Kc.T).norm(dim=0).clamp_min(1e-300)).max())
                else:
                    g["keep_greedy"] = dict(rel_counts(-B, DKn), n_selected=0)
                    g["keep_greedy"]["restoration_rel_err_last_trace"] = float(((D @ Kc.T - D @ (Qb @ (Qb.T @ Kc.T))).norm(dim=0) / (D @ Kc.T).norm(dim=0).clamp_min(1e-300)).max())
                # thresholds-only repair
                tau_s = torch.maximum(tau0, _above(mmax, MARGIN, ABS_MARGIN))
                gok = Sp.guard_ok()
                g["thresholds"] = {"sacrificing": {"n_silenced_at_solver_key": int((r_solver < tau_s).sum()), "n_silenced_on_own_request": int((r_own < tau_s).sum()),
                                                   "n_silenced_either": int(((r_solver < tau_s) | (r_own < tau_s)).sum()), "n_thresholds_raised": int((tau_s > tau0).sum())},
                                   "protecting": {"n_complaints_refused": n_ref, "n_guards_silenced": int((~gok).sum()),
                                                  "n_silenced_at_solver_key": int((r_solver < Sp.det_tau).sum()), "n_silenced_on_own_request": int((r_own < Sp.det_tau).sum()),
                                                  "n_thresholds_raised": int((Sp.det_tau > tau0).sum())}}
                grid.append(g)
                print(f"s{seed} n={n} keys={n_keys} rank={r} cap={caps} plain={g['plain']['n_preserved']} ls={g['keep_ls']['n_preserved']} "
                      f"greedy={g['keep_greedy']['n_preserved']} thr={g['thresholds']}", flush=True)
                del PKE, E, Ue, Vhe, B
                torch.cuda.empty_cache()
                jdump({"log": name, "seed": seed, "partial": True, "grid": grid}, path + ".partial")
            del Kc
        res = {"log": name, "seed": seed, "T": T, "d_in": d, "tolerance_rank": TOL, "tolerance_preserved": PRES, "grid": grid, "per_complaint": per,
               "residual_singular_values_in_bands": bands, "smallest_kept_residual_singular_value_rel": min_kept,
               "largest_dropped_residual_singular_value_rel": max_dropped,
               "rank_stable_over_two_decades": bool(bands["in_1e-10_1e-9"] == 0 and bands["in_1e-9_1e-8"] == 0),
               "where_logonly": {"n_edits_capped": fx.methods.n_capped(Mw), "start_silenced_at_solver_key": int((r_solver < tau0).sum()),
                                 "start_silenced_on_own_request": int((r_own < tau0).sum())},
               "seconds": time.time() - t0, "gq_mem": os.environ.get("GQ_MEM"), "linear_algebra_device": str(dev), "doc": __doc__}
        jdump(res, path)
        if os.path.exists(path + ".partial"):
            os.remove(path + ".partial")
        del Qb
        torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
