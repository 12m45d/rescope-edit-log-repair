"""E5 (a): rebase lemma and retrofit proposition at factor level (CPU), one log per process.   python e5a_rebase.py <log>
Targets: the cached clean-model targets of the testbed (z*_i and the pre-edit block output h_i(W_0) of every edit; harness Z cache):
V*_i = z*_i - h_i(W_0).  All quantities fp64 from the log's factors (U, W, K, batch) and the key cache (B = 1 own Rel keys)."""
import dataclasses
from e5common import *
import rescope
from rescope import loglib as LL

def main(name):
    L = fx.logs.open(name)
    log = L.log
    rescope.register_logs()
    hz = rescope.harness()
    Z = hz.TB.Z
    from vendor_v7.vlm import DEFAULT_DTYPE
    cfg = hz.TB.LOGS[name]
    zcfg = dataclasses.replace(Z.ZConfig.for_backbone(cfg["backbone"]), dtype=Z._dtype_tag(DEFAULT_DTYPE[cfg["backbone"]]))
    tg = [Z.load_cached(cfg["backbone"], cfg["layer"], u, zcfg, None) for u in log.meta["uids"]]
    found = [t is not None for t in tg]
    out = {"log": name, "T": L.T, "backbone": L.backbone, "role": L.role, "editor": L.reg.get("editor"), "n_batches": int(log.batch.max()) + 1,
           "n_targets_found": int(sum(found))}
    V = LL.rebase(log)                                                   # factor form of the lemma's first identity
    Vd, vrec = fx.methods.released_values(L, True)                       # the driver's released values (batch by batch)
    out["driver_values_vs_library_rebase_max_abs"] = float((Vd - V).abs().max())
    if all(found):
        ref = torch.stack([t.z for t in tg]).double() - torch.stack([t.h for t in tg]).double()       # V*_i = z*_i - h_i(W_0)
        e = LL.rebase_error(log, ref, V)
        rn = ref.norm(dim=1)
        out["rebase"] = {"err_rebased": {k: e["rebased"][k] for k in ("max", "median", "mean")}, "err_logged": {k: e["not_rebased"][k] for k in ("max", "median", "mean")},
                         "err_logged_later_batches": None if e["not_rebased_later_batches"] is None else {k: e["not_rebased_later_batches"][k] for k in ("max", "median", "mean")},
                         "err_rebased_per_edit": e["rebased_per_edit"], "err_logged_per_edit": e["not_rebased_per_edit"]}
        # second identity: the deployed log at its own solver keys (factor form of W_E - W_0: sum_j u_j <w_j, k_i>) against the target
        C = log.K @ log.W.T                                              # C[i, j] = <w_j, k_i>
        dep = C @ log.U                                                  # (W_E - W_0) k_i without the write round-off
        de = ((dep - ref).norm(dim=1) / rn).numpy()
        earlier = log.batch[None, :] < log.batch[:, None]
        eye = torch.eye(L.T, dtype=torch.bool)
        rhs = (C.diagonal() - 1)[:, None] * log.U + torch.where(~earlier & ~eye, C, torch.zeros_like(C)) @ log.U
        out["own_keys"] = {"deployed_log_rel_err_vs_target": qs(de), "scoped_replay_rel_err_vs_target": qs(e["rebased_per_edit"]),
                           "second_identity_residual_max_rel": float(((dep - V - rhs).norm(dim=1) / V.norm(dim=1)).max()),
                           "deployed_rel_err_vs_rebased_value": qs(((dep - V).norm(dim=1) / V.norm(dim=1)).numpy()),
                           "note": "module output at the solver key k_i minus the target, relative to |V*_i|; deployed: W_0 k_i + sum_j u_j <w_j, k_i> (factors, no write "
                                   "round-off); scoped replay: W_0 k_i + V_i (edit i alone fires at k_i when thresholds lie in (1 - margin_i, 1])",
                           "deployed_per_edit": de.tolist()}
    else:
        out["rebase"] = None
        out["own_keys"] = None
    corr = ((V - log.U).norm(dim=1) / V.norm(dim=1)).numpy()
    first = (log.batch == int(log.batch.min())).numpy()
    out["correction_size_rel_to_value"] = dict(qs(corr), later_batches=qs(corr[~first]) if (~first).any() else None)
    R = log.crosstalk()                                                  # R[j, i] = r_i(k_j)
    off = R.clone(); off.fill_diagonal_(-float("inf"))
    margin = (1 - off.max(0).values).numpy()                             # margin_i = 1 - max_{j != i} r_i(k_j)
    out["self_separation_margin"] = dict(qs(margin), n_nonpositive=int((margin <= 0).sum()), n_below_0_1=int((margin < 0.1).sum()),
                                         quantiles={str(q): float(np.quantile(margin, q)) for q in (0.01, 0.05, 0.25, 0.75)}, per_edit=margin.tolist())
    S, grec = fx.methods.base_gate(L)
    K1, ed = L.own_keys()
    r_own = S.key_scores(torch.as_tensor(K1))                            # (T, T), each key its own query
    ed = torch.as_tensor(ed)
    ro = r_own[torch.arange(len(ed)), ed].numpy()
    o2 = r_own.clone(); o2[torch.arange(len(ed)), ed] = -float("inf")
    out["own_coordinate_b1"] = dict(qs(ro), n_below_one=int((ro < 1).sum()), n_firing_at_threshold_one=int((ro >= 1).sum()),
                                    drop_max=float(1 - ro.min()), n_own_not_top1_by_r=int((o2.max(1).values >= torch.as_tensor(ro)).sum()),
                                    n_below_own_margin=int((torch.as_tensor(ro) <= torch.as_tensor(1 - margin)[ed]).sum()), per_edit=ro.tolist(),
                                    note="normalised coordinate of edit i at the B = 1 teacher-forced key of its own Rel prompt (the key cache); the stored solver key is the B = 10 key")
    out["solver_own_coordinate_raw"] = qs(log.own().numpy())
    jdump(out, f"{OUT}/a/{name}.json")
    print(name, {k: out[k] if not isinstance(out[k], dict) else {a: b for a, b in out[k].items() if not isinstance(b, (list, dict))} for k in out if k not in ("rebase",)}, flush=True)
    if out["rebase"]:
        print(" rebase", out["rebase"]["err_rebased"], out["rebase"]["err_logged"])

if __name__ == "__main__":
    main(sys.argv[1])
