"""E2 on prefix logs (R12, reduced by R19 d): shared constants and helpers of this experiment.  Everything method-side is the
driver fx (10_final/common) and, for the linear repairs, the functions of P12 (08_rescope/p12_linear_streams/code/run_streams.py)
called on the prefix handle of fx.prefix.load_model."""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
EXP = os.path.join(FIN, "e2_prefix")
P12 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p12_linear_streams"
sys.path.insert(0, os.path.join(FIN, "common"))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(EXP, "out")
CELLS = os.path.join(OUT, "cells")
ROWS = os.path.join(OUT, "rows")
CORR = os.path.join(OUT, "corr")
QDIR = os.path.join(OUT, "qmax")
HCACHE = os.path.join(EXP, "hcache")
P12CACHE = os.path.join(EXP, "p12cache")
LOGD = os.path.join(EXP, "logs")
EXPERIMENT = "E2P"

LOGS = tuple(fx.logs.names("prefix"))
LIN = ("COMP_B", "COMP_BP", "REEDIT_B", "RB_TOT", "RB_UNION")
LIN_LABEL = {"COMP_B": "comp_b", "COMP_BP": "comp_bp", "REEDIT_B": "reedit_b", "RB_TOT": "rb_tot", "RB_UNION": "rb_union"}
NS = (1, 10, 20, "ALL")
SEEDS = (0, 1, 2)
N_BASE = 20                    # stratified base draw of the nested complaint stream on a prefix (USAGE: fx.draws.complaints(L, n, seed, n_base=20))
ALPHA, N_CAL = 0.05, 200
DRAWS_B = (0, 1, 2)            # behavioural calibration draws
N_ORDERS = 20                  # arrival orders of the least repair (order independence)


def cells_plan():
    """(n, seed) of the complaint-driven cells: 3 seeds for n <= 20, seed 0 for ALL."""
    return [(n, s) for n in NS for s in (SEEDS if n != "ALL" else (0,))]


def cfg(method, n, seed):
    return f"{method}__n{n}__s{seed}"


def cell_path(log, name):
    return os.path.join(CELLS, log, name + ".json")


def rows_path(log, name):
    return os.path.join(ROWS, f"{log}__{name}.jsonl.gz")


def done(log, name):
    return os.path.exists(cell_path(log, name))


def put(log, name, rec):
    fx.io.jdump(fx.io.clean(rec), cell_path(log, name), indent=1)


def get(log, name):
    p = cell_path(log, name)
    return fx.io.jload(p) if os.path.exists(p) else None


def save_rows(log, name, rows):
    fx.io.save_rows(rows_path(log, name), rows)


def load_rows(log, name):
    return fx.io.load_rows(rows_path(log, name))


def where_gate_level(L, rows, Q, theta, per_row=False):
    """Gate-level decisions of a ReScope-W gate (role 'all', fire 'all') from the captured per-row maxima: Q[i, j] = the largest
    normalised coordinate of edit j over ALL positions of the teacher-forced sequence of row rows[i] (the keys of the edited module
    do not depend on the gate), so that edit j is released somewhere on the row iff Q[i, j] >= theta[j].
    -> firing on heldout_disjoint / heldout_strict (near / far), own request and rephrases released, hard negatives by kind."""
    Qn = Q.numpy() if torch.is_tensor(Q) else np.asarray(Q)
    th = theta.detach().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta, dtype=np.float64)
    if Qn.shape[1] != L.T or th.shape[0] != L.T:
        raise ValueError("one detector per edit is expected")
    pos = {int(j): i for i, j in enumerate(rows)}
    cand = Qn >= th[None, :]
    fired = cand.any(1)
    pct = lambda x: (100.0 * float(np.mean(x)) if len(x) else None)                                     # noqa: E731
    out = {"probe": {}, "edit": {}, "hn": {}}
    for pool in ("heldout_disjoint", "heldout_strict", "repair"):
        ix = [pos[j] for j in L.pool(pool)]
        out["probe"][pool] = {"n": len(ix), "fired": pct(fired[ix]), "n_fired": int(fired[ix].sum())}
    hd = L.pool("heldout_disjoint")
    for nm, sel in (("near", [j for j in hd if L.near_edit(j) is not None]), ("far", [j for j in hd if L.near_edit(j) is None])):
        ix = [pos[j] for j in sel]
        out["probe"]["heldout_disjoint_" + nm] = {"n": len(ix), "fired": pct(fired[ix])}
    for m in ("rel", "tgen", "igen"):
        js = L.edits_of(m)
        ix = [pos[j] for j in js]
        own = np.array([cand[pos[j], L.own(j)] for j in js], dtype=bool)
        out["edit"][m] = {"n": len(js), "fired": pct(fired[ix]), "own_released": pct(own), "n_own_released": int(own.sum())}
    hn = L.rows("hn", valid=True)
    for t in ("HN-I", "HN-Q", "HN-X"):
        js = [j for j in hn if L.index[j]["hn_type"] == t]
        ix = [pos[j] for j in js]
        own = np.array([cand[pos[j], L.own(j)] for j in js], dtype=bool)
        out["hn"][t] = {"n": len(js), "fired": pct(fired[ix]), "own_released": pct(own)}
    ix = [pos[j] for j in hn]
    out["hn"]["all"] = {"n": len(hn), "fired": pct(fired[ix])}
    kinds = [t for t in ("HN-I", "HN-Q", "HN-X") if out["hn"][t]["n"]]
    out["hn"]["mean_fired"] = float(np.mean([out["hn"][t]["fired"] for t in kinds])) if kinds else None
    if per_row:
        out["_fired"] = {int(j): bool(fired[i]) for j, i in pos.items()}
    return out
