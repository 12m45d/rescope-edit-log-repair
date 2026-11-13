"""Check 2 (E5 a): error of the rebased and of the logged values against the cached clean-model targets, own arithmetic, one log per family."""
from vcommon import *
import dataclasses
import fx, rescope

def targets(name, uids):
    rescope.register_logs()
    hz = rescope.harness()
    Z = hz.TB.Z
    from vendor_v7.vlm import DEFAULT_DTYPE
    cfg = hz.TB.LOGS[name]
    zcfg = dataclasses.replace(Z.ZConfig.for_backbone(cfg["backbone"]), dtype=Z._dtype_tag(DEFAULT_DTYPE[cfg["backbone"]]))
    tg = [Z.load_cached(cfg["backbone"], cfg["layer"], u, zcfg, None) for u in uids]
    assert all(t is not None for t in tg)
    return torch.stack([t.z for t in tg]).double() - torch.stack([t.h for t in tg]).double()

def run(name):
    L = fx.logs.open(name)
    U, W, K, bt = L.log.U.double(), L.log.W.double(), L.log.K.double(), L.log.batch
    T = U.shape[0]
    ref = targets(name, L.log.meta["uids"])
    C = K @ W.T                                                      # C[i, j] = <W_j, k_i>
    V = U.clone()
    for i in range(T):                                               # own loop: earlier batches only
        ear = bt < bt[i]
        if bool(ear.any()): V[i] = U[i] + C[i, ear] @ U[ear]
    rn = ref.norm(dim=1)
    er, el = ((V - ref).norm(dim=1) / rn).numpy(), ((U - ref).norm(dim=1) / rn).numpy()
    s = C.diagonal()
    Rn = C / s[None, :]                                              # r_j(k_i)
    off = Rn.clone(); off.fill_diagonal_(-float("inf"))
    margin = (1 - off.max(0).values).numpy()
    E = jload(f"{FINAL}/e5_rescope/out/a/{name}.json")
    out = {"log": name, "T": T, "mine": {"err_rebased_max_pct": 100 * float(er.max()), "err_logged_max_pct": 100 * float(el.max()), "err_rebased_median_pct": 100 * float(np.median(er)),
                                         "err_logged_median_pct": 100 * float(np.median(el)), "margin_min": float(margin.min()), "n_margin_nonpositive": int((margin <= 0).sum())},
           "experiment": {"err_rebased_max_pct": 100 * E["rebase"]["err_rebased"]["max"], "err_logged_max_pct": 100 * E["rebase"]["err_logged"]["max"],
                          "err_rebased_median_pct": 100 * E["rebase"]["err_rebased"]["median"], "err_logged_median_pct": 100 * E["rebase"]["err_logged"]["median"],
                          "margin_min": E["self_separation_margin"]["min"], "n_margin_nonpositive": E["self_separation_margin"]["n_nonpositive"]}}
    out["max_abs_diff"] = max(abs(out["mine"][k] - out["experiment"][k]) for k in out["mine"])
    return out

if __name__ == "__main__":
    res = {}
    for name in ("TB-Q500-ae-l2_3", "TB-Q2000-memitw", "TB-L500-ae-l2_3", "TB-L1000-memitw"):
        res[name] = run(name)
        print(name, json.dumps(res[name]["mine"]), "| exp", json.dumps(res[name]["experiment"]), "| maxdiff", res[name]["max_abs_diff"], flush=True)
        jdump(res, OUT + "/c2_rebase.json")
