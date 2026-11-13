"""Verification of E2 a-c, CPU prep: the DRAW DEFINITIONS of the shared driver (calibration draw 0 at n = 10 / 200, reported
complaints of seed 0 at n = 10 / 100 / ALL) written to out/prep/draws_<log>.json, next to my own reconstruction of the same draws
from the library's split and the P12 records (no driver).  Draws are identities of rows, not evaluations."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
VER = FIN + "/e2_curves/verify"
P12 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p12_linear_streams"
sys.path.insert(0, FIN + "/common"); sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import fx, rescope
from rescope import splits

for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
    L = fx.logs.open(name)
    strip = lambda s: s.split(":", 1)[1]                                   # cache sid 'Q:probe:image:0' -> library sid 'probe:image:0'
    drv = {"cal10": [strip(s) for s in fx.draws.calibration(L, 10, 0).rec["sids"]],
           "cal200": [strip(s) for s in fx.draws.calibration(L, 200, 0).rec["sids"]],
           "complaints": {str(n): [L.index[j]["group"] for j in fx.draws.complaints(L, n, 0)] for n in (10, 100, "ALL")},
           "stream0": [L.index[j]["group"] for j in fx.draws.complaint_stream(L, 0)],
           "n_repair_valid": len(L.pool("repair")), "n_heldout_disjoint": len(L.pool("heldout_disjoint")), "n_heldout_strict": len(L.pool("heldout_strict")),
           "heldout_disjoint": sorted(strip(L.index[j]["sid"]) for j in L.pool("heldout_disjoint"))}
    # my reconstruction: valid REPAIR rows in the order of probe_records_v2.jsonl (the order of Evaluator.probe_items)
    rows, pool_of = rescope.load_rows(name)
    ann = {(a["probe_idx"], a["ptype"]): a for a in splits.annotate(rows, pool_of)}
    rep = [f"probe:{r['ptype']}:{r['probe_idx']}" for r in rows if splits.in_pool(ann[(r["probe_idx"], r["ptype"])], "repair") and r["y0_valid"]]
    n_pool = len(rep)
    base = np.sort(np.random.default_rng(0).choice(n_pool, 200, replace=False))
    rest = np.setdiff1d(np.arange(n_pool), base)
    rk = np.concatenate([base[np.random.default_rng([0, 1]).permutation(base.size)], rest[np.random.default_rng([0, 2]).permutation(rest.size)]])
    mine = {"cal10": [rep[int(i)] for i in np.sort(rk[:10])], "cal200": [rep[int(i)] for i in base]}
    hd = sorted(f"probe:{r['ptype']}:{r['probe_idx']}" for r in rows if splits.in_pool(ann[(r["probe_idx"], r["ptype"])], "heldout_disjoint") and r["y0_valid"])
    p12 = {str(n): json.load(open(f"{P12}/out/{name}/COMP_B__n{n}__s0.json"))["reported_groups"] for n in (10, 100, "ALL")}
    chk = {"cal10_equal": mine["cal10"] == drv["cal10"], "cal200_equal": mine["cal200"] == drv["cal200"],
           "complaints_equal_p12": {n: p12[n] == drv["complaints"][n] for n in p12}, "heldout_disjoint_equal": hd == drv["heldout_disjoint"],
           "n_repair_valid_mine": n_pool, "nested": drv["complaints"]["10"] == drv["complaints"]["ALL"][:10] and drv["complaints"]["100"] == drv["complaints"]["ALL"][:100]}
    json.dump({"log": name, "driver": drv, "mine": mine, "p12_reported_groups": p12, "check": chk}, open(f"{VER}/out/prep/draws_{name}.json", "w"), indent=1)
    print(name, chk, flush=True)
