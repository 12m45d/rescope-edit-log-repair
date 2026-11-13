"""E3 (Qwen), factor level and stored metrics (CPU), one log per process.   python e3_factor.py <log>

(5) rebase lemma at factor level: relative error of the rebased values V_i and of the logged residuals U_i against the cached
clean-model targets V*_i = z*_i - h_i(W_0) (the testbed's target cache); self-separation margins of the solver keys
(margin_i = 1 - largest normalised coordinate of edit i at the solver key of another edit); the deployed log's own stored metrics
(fx.logs.stored; no model).  Writes out/factor/<log>.json.
"""
import dataclasses
import sys

from e3c import *  # noqa: F401,F403
import torch

torch.set_num_threads(4)


def qs(v):
    v = np.asarray(v, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "mean": float(v.mean()), "max": float(v.max()), "n": int(v.size)}


def main(name):
    import rescope
    from rescope import loglib as LL
    assert name in LOGS
    L = fx.logs.open(name)
    log = L.log
    out = {"log": name, "T": L.T, "role": L.role, "editor": L.reg.get("editor"), "n_batches": int(log.batch.max()) + 1, "describe": L.describe()}
    try:
        out["stored"] = fx.logs.stored(name, pools=True)
    except Exception as e:                                             # noqa: BLE001
        out["stored"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        try:
            out["stored"] = fx.logs.stored(name)
        except Exception as e2:                                        # noqa: BLE001
            out["stored"] = {"error": f"{type(e2).__name__}: {str(e2)[:300]}"}
    V = LL.rebase(log)
    Vd, _ = fx.methods.released_values(L, True)
    out["driver_values_vs_library_rebase_max_abs"] = float((Vd - V).abs().max())
    try:
        rescope.register_logs()
        hz = rescope.harness()
        Z = hz.TB.Z
        from vendor_v7.vlm import DEFAULT_DTYPE
        cfg = hz.TB.LOGS[name]
        zcfg = dataclasses.replace(Z.ZConfig.for_backbone(cfg["backbone"]), dtype=Z._dtype_tag(DEFAULT_DTYPE[cfg["backbone"]]))
        tg = [Z.load_cached(cfg["backbone"], cfg["layer"], u, zcfg, None) for u in log.meta["uids"]]
        found = [t is not None for t in tg]
        out["n_targets_found"] = int(sum(found))
        if all(found):
            ref = torch.stack([t.z for t in tg]).double() - torch.stack([t.h for t in tg]).double()
            e = LL.rebase_error(log, ref, V)
            later = np.array(fx.masks.later_edits(L), dtype=np.int64)
            first = np.arange(min(100, L.T))
            er, el = np.asarray(e["rebased_per_edit"]), np.asarray(e["not_rebased_per_edit"])
            out["rebase"] = {"err_rebased": qs(er), "err_logged": qs(el),
                             "err_rebased_first100": qs(er[first]), "err_logged_first100": qs(el[first]),
                             "err_rebased_later": qs(er[later]) if later.size else None, "err_logged_later": qs(el[later]) if later.size else None,
                             "err_logged_later_batches": None if e["not_rebased_later_batches"] is None else {k: e["not_rebased_later_batches"][k] for k in ("max", "median", "mean")},
                             "err_rebased_per_edit": er.tolist(), "err_logged_per_edit": el.tolist(),
                             "note": "|V_i - V*_i| / |V*_i| (rebased) and |U_i - V*_i| / |V*_i| (logged residual, no rebasing); V*_i = z*_i - h_i(W_0) from the testbed's target cache"}
        else:
            out["rebase"] = None
    except Exception as e:                                             # noqa: BLE001
        out["rebase"] = None
        out["rebase_error"] = f"{type(e).__name__}: {str(e)[:500]}"
    R = log.crosstalk()                                                  # R[j, i] = r_i(k_j)
    off = R.clone()
    off.fill_diagonal_(-float("inf"))
    margin = (1 - off.max(0).values).numpy()                             # margin_i = 1 - max_{j != i} r_i(k_j)
    tie = np.abs(margin) < 1e-9
    out["self_separation_margin"] = dict(qs(margin), n_positive=int((margin > 0).sum()), n_nonpositive=int((margin <= 0).sum()), n_ties=int(tie.sum()),
                                         n_negative_excluding_ties=int(((margin < 0) & ~tie).sum()), n_below_0_1=int((margin < 0.1).sum()),
                                         argmin_edit=int(margin.argmin()), per_edit=margin.tolist())
    jdump(out, f"{FAC}/{name}.json")
    print(name, "targets", out.get("n_targets_found"), "rebase", None if not out.get("rebase") else (out["rebase"]["err_rebased"]["median"], out["rebase"]["err_logged"]["median"]),
          "margin", {k: v for k, v in out["self_separation_margin"].items() if k != "per_edit"}, out.get("rebase_error"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
