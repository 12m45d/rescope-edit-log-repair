"""Verification of E2 part c' (R24, ReScope-W with two-block thresholds) at gate level, and of the part-b bookkeeping claims.
CPU.  Input: the experiment's captured all-position vectors (out/capture/<log>.pt: Rmax / Rlast per sequence).  Everything else is
recomputed here: the 20 calibration draws (library split rule), the two-block rule, the guard caps, the firing rates.
-> out/cw_check.json"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob, json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, torch
FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
EXP, VER = FIN + "/e2_curves", FIN + "/e2_curves/verify"
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import rescope
from rescope import splits
from rescope.calibrate import split_blocks

out = {}
for name, sh in (("TB-Q100-ae-l2_3", "Q"), ("TB-L100-ae-l2_3", "L")):
    cap = torch.load(f"{EXP}/out/capture/{name}.pt")
    ix = {s.split(":", 1)[1]: i for i, s in enumerate(cap["sids"]) if s.split(":", 1)[0] in ("Q", "L")}
    Rm, Rl = cap["Rmax"].numpy(), cap["Rlast"].numpy()
    T = Rm.shape[1]
    rows, pool_of = rescope.load_rows(name)
    ann = {(a["probe_idx"], a["ptype"]): a for a in splits.annotate(rows, pool_of)}
    sid = lambda r: f"probe:{r['ptype']}:{r['probe_idx']}"
    rep = [r for r in rows if splits.in_pool(ann[(r["probe_idx"], r["ptype"])], "repair") and r["y0_valid"]]
    hd = [r for r in rows if splits.in_pool(ann[(r["probe_idx"], r["ptype"])], "heldout_disjoint") and r["y0_valid"]]
    qid = lambda r: ann[(r["probe_idx"], r["ptype"])]["query_id"]
    rep_i, hd_i = [ix[sid(r)] for r in rep], [ix[sid(r)] for r in hd]
    g_own = np.array([Rl[ix[f"edit:rel:{i}"]][i] for i in range(T)])          # own B = 1 Rel key at the last prompt position
    guards = np.stack([np.ones(T), g_own])                                     # stored solver key: coordinate 1 by the normalisation
    capv = (guards - 1e-9 * np.abs(guards)).min(0)
    theirs = json.load(open(f"{EXP}/out/parts/cw_gate_{name}.json"))["draws"]
    chk = json.load(open(f"{EXP}/out/parts/cw_check_{name}.json"))
    per = []
    for d in range(20):
        base = np.sort(np.random.default_rng(d).choice(len(rep), 200, replace=False))
        Q = Rm[[rep_i[int(i)] for i in base]]
        i1, i2 = split_blocks(200, (1, 1), d)
        b = Q[i1].max(0)
        sc = np.sort((Q[i2] - b[None, :]).max(1))
        k = -((-19 * (len(i2) + 1)) // 20)
        eta = sc[k - 1]
        lit = np.empty(T)
        for i in range(T):
            t = b[i] + eta
            while (t - b[i]) > eta:
                t = np.nextafter(t, -np.inf)
            while not (t - b[i]) > eta:
                t = np.nextafter(t, np.inf)
            lit[i] = t
        sil = (guards < lit[None, :]).any(0)
        tau = np.where(sil, np.minimum(lit, capv), lit)
        fire = lambda th, ii: 100.0 * float((Rm[ii] >= th[None, :]).any(1).mean())       # noqa: E731
        qs = {qid(rep[int(i)]) for i in base}
        fresh = [rep_i[j] for j, r in enumerate(rep) if qid(r) not in qs]
        t_ = theirs[d]
        per.append({"draw": d, "fired_as_run_mine": fire(tau, hd_i), "fired_as_run_exp": t_["gate"]["probe"]["heldout_disjoint"]["fired"],
                    "fired_literal_mine": fire(lit, hd_i), "fired_literal_exp": t_["gate_literal"]["probe"]["heldout_disjoint"]["fired"],
                    "n_capped_mine": int(sil.sum()), "n_capped_exp": t_["n_edits_capped"], "eta_mine": float(eta),
                    "fresh_repair_literal_mine": fire(lit, fresh), "n_fresh": len(fresh), "fresh_repair_literal_exp": chk["draws"][d]["fired_fresh_repair"]})
    m = lambda k: float(np.mean([p[k] for p in per]))                          # noqa: E731
    out[name] = {"draws": per, "n_heldout_disjoint": len(hd_i),
                 "mean": {k: m(k) for k in ("fired_as_run_mine", "fired_as_run_exp", "fired_literal_mine", "fired_literal_exp", "n_capped_mine", "n_capped_exp",
                                             "fresh_repair_literal_mine", "fresh_repair_literal_exp")},
                 "max_abs_diff": {a: float(max(abs(p[a + "_mine"] - p[a + "_exp"]) for p in per)) for a in ("fired_as_run", "fired_literal", "n_capped", "fresh_repair_literal")},
                 "bound_percent": 100.0 * 5 / 101}
    # part b bookkeeping claims of the summary, from the cells
    cells = [json.load(open(p)) for p in sorted(glob.glob(f"{EXP}/out/cells/b/{name}/*.json"))]
    jobs = [json.load(open(p)) for p in sorted(glob.glob(f"{EXP}/out/parts/job_{name}_*.json"))]
    out[name]["part_b"] = {"n_cells": len(cells), "n_refused_total": sum(c["n_refused"] for c in cells), "n_still_firing_total": sum(c["n_reported_still_firing"] for c in cells),
                           "n_cells_20_of_20_orders": sum(c["order_check"]["n_equal_thresholds"] == 20 for c in cells),
                           "n_cells_vector_rule_equals_driver": sum(bool(c["order_check"]["vector_rule_equals_driver_thresholds"]) for c in cells),
                           "hook_vs_vectors_rows": sum(c["hook_vs_vectors"]["n_rows"] for c in cells), "hook_vs_vectors_differ": sum(c["hook_vs_vectors"]["n_fired_differs"] + c["hook_vs_vectors"]["n_own_last_differs"] for c in cells)}
    out[name]["jobs"] = [{"blocks": j["blocks"], "assert_clean": j["assert_clean"], "hook_vs_vectors_total": j["hook_vs_vectors_total"], "smoke": j.get("smoke")} for j in jobs]
    print(name, json.dumps(out[name]["mean"]), out[name]["max_abs_diff"], out[name]["part_b"], out[name]["jobs"])
json.dump(out, open(f"{VER}/out/cw_check.json", "w"), indent=1)
