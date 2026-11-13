"""Verification of E3, small checks (CPU): (a) the published all-positions cosine gate against its last-position form, row by row;
(b) the factor-level rebase error against the testbed's cached clean-model targets, recomputed with my own rebasing (Qwen logs).
    python v_misc.py   -> out/misc.json"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import dataclasses, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import v_internal as VI
out = {"cosine_published_vs_default": {}, "rebase": {}}
for bb, B in VI.BB.items():
    for log in B["logs"]:
        a = VI.rows(bb, log, "cosine_published", None, "tf")
        b = {r["sid"]: r for r in VI.rows(bb, log, "cosine_default", None, "tf")}
        out["cosine_published_vs_default"][log] = {"n_rows": len(a), "n_outcome_differs": sum(any(r.get(k) != b[r["sid"]].get(k) for k in ("exact", "D1")) for r in a),
                                                   "n_fired_differs": sum(bool(r.get("fired")) != bool(b[r["sid"]].get("fired")) for r in a),
                                                   "n_fired_before_last_position": sum(bool(r.get("fired_before")) for r in a) if "fired_before" in a[0] else None}
import rescope
rescope.register_logs()
hz = rescope.harness()
Z = hz.TB.Z
from vendor_v7.vlm import DEFAULT_DTYPE
for bb, B in VI.BB.items():
    for log in B["logs"]:
        cfg = hz.TB.LOGS[log]
        fac = torch.load(f"{VI.TESTBED}/{log}/edit_factors.pt", map_location="cpu")
        met = json.load(open(f"{VI.TESTBED}/{log}/metrics.json"))
        U, W, K = fac["U"].double(), fac["W"].double(), fac["K1_solver"].double()
        bt = torch.zeros(U.shape[0], dtype=torch.long)
        for x in met["per_batch"]:
            bt[torch.tensor(x["edit_ids"])] = int(x["batch"])
        zcfg = dataclasses.replace(Z.ZConfig.for_backbone(cfg["backbone"]), dtype=Z._dtype_tag(DEFAULT_DTYPE[cfg["backbone"]]))
        tg = [Z.load_cached(cfg["backbone"], cfg["layer"], u, zcfg, None) for u in fac["uids"]]
        if any(t is None for t in tg):
            out["rebase"][log] = {"targets_found": sum(t is not None for t in tg)}
            continue
        ref = torch.stack([t.z for t in tg]).double() - torch.stack([t.h for t in tg]).double()
        V = U.clone()                                                    # my rebasing: batch by batch, earlier batches only
        for b_ in torch.unique(bt).tolist():
            i, e = (bt == b_).nonzero().flatten(), (bt < b_).nonzero().flatten()
            if e.numel():
                V[i] = U[i] + (K[i] @ W[e].T) @ U[e]
        er = ((V - ref).norm(dim=1) / ref.norm(dim=1)).numpy()
        el = ((U - ref).norm(dim=1) / ref.norm(dim=1)).numpy()
        out["rebase"][log] = {"median_rebased_pct": 100 * float(np.median(er)), "max_rebased_pct": 100 * float(er.max()), "median_logged_pct": 100 * float(np.median(el)), "max_logged_pct": 100 * float(el.max())}
        print(log, out["rebase"][log], flush=True)
json.dump(out, open(VI.V + "/out/misc.json", "w"), indent=1)
print(json.dumps(out["cosine_published_vs_default"]))
