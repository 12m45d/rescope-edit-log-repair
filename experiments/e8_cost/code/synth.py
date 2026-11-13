"""E8: common paths, synthetic logs, small helpers.

SYNTHETIC LOG OF T EDITS (R13; "random factors with the row norms and the coordinate distribution of a real log").
Reference log = the longest MEMIT-style log of the backbone (TB-Q2000-memitw / TB-L1000-memitw), T_ref real edits.
For synthetic edit j (seeded, numpy default_rng(SEED + T)):
    i(j)   a real edit drawn uniformly with replacement
    w'_j   = |w_i| * unit(BETA * unit(w_i) + sqrt(1 - BETA^2) * g_j),  g_j a standard Gaussian direction made orthogonal to w_i
             (a real read-out vector turned by the fixed angle arccos(BETA) in a random direction; its norm is the real one)
    s'_j   = s_i  (the real own coordinate; the synthetic solver key is DEFINED as k_i + (s_i - <w'_j, k_i>) w'_j / |w'_j|^2 and is
             never materialised: its normalised own coordinate is 1 by construction)
    v'_j   = the released (rebased) value of real edit i (real value norms)
BETA = 0.9 was fixed before any timing was looked at; costs depend on T, d and m only.  The distribution of the normalised
coordinates of the 200 calibration keys on the synthetic detectors is recorded next to the real one (quantiles).
The scoped log of a synthetic log has no guard keys (the own coordinate is 1 by construction: the guard cap is (1 - 1e-9) for
every edit and is applied explicitly), det_key / lin_w point to the detector matrix and lin_u to the values (no extra memory).
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
import subprocess
import sys
import time

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np          # noqa: E402
import torch                # noqa: E402

import fx                   # noqa: E402

torch.set_num_threads(4)
HERE = FINAL + "/e8_cost"
OUT = HERE + "/out"
BETA, SEED = 0.9, 20261111
DEV = {"qwen25vl": "TB-Q100-ae-l2_3", "llava": "TB-L100-ae-l2_3"}
REF = {"qwen25vl": "TB-Q2000-memitw", "llava": "TB-L1000-memitw"}
REAL = {"qwen25vl": ["TB-Q100-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw"],
        "llava": ["TB-L100-ae-l2_3", "TB-L100-memitw", "TB-L1000-memitw"]}
TIE = 1e-9


def jdump(obj, path):
    fx.io.jdump(obj, path, indent=1)


def jload(path):
    with open(path) as f:
        return json.load(f)


def gpu_load():
    """Utilisation / memory of the card of this process and the number of compute processes on it (other processes = n - 1)."""
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        q = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout.strip().split(",")
        uuid = subprocess.run(["nvidia-smi", "-i", idx, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.split()
        n = sum(a == uuid for a in apps)
        return {"gpu": idx, "utilization_pct": float(q[0]), "memory_used_MiB": float(q[1]), "n_compute_processes": n, "n_other_processes": max(n - 1, 0),
                "time": time.strftime("%H:%M:%S")}
    except Exception as e:                                             # context only
        return {"error": repr(e)}


def q3(v, scale=1.0):
    v = np.asarray(v, dtype=np.float64).ravel() * scale
    return {"median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9)), "mean": float(v.mean()), "n": int(v.size)}


_REFC = {}


def reference(backbone):
    """(L_ref, W (T, d), s (T,), V (T, m) released values, calibration rows of draw 0) of the reference log; cached."""
    if backbone not in _REFC:
        L = fx.logs.open(REF[backbone])
        V, _ = fx.methods.released_values(L, True)
        dr = fx.draws.calibration(L, 200, 0)
        _REFC[backbone] = (L, L.log.W, L.log.own(), V, dr)
    return _REFC[backbone]


def synth_factors(backbone, T, chunk=4000):
    """(W' (T, d) fp64, s' (T,), V' (T, m) fp64, idx (T,) the real edit behind every synthetic edit)."""
    L, W, s, V, _ = reference(backbone)
    rng = np.random.default_rng(SEED + T)
    idx = torch.from_numpy(rng.integers(0, L.T, size=T))
    gen = torch.Generator().manual_seed(SEED + T)
    Wn = W.norm(dim=1)
    out = torch.empty(T, W.shape[1], dtype=torch.float64)
    c2 = (1 - BETA ** 2) ** 0.5
    for a in range(0, T, chunk):
        ii = idx[a:a + chunk]
        wh = W[ii] / Wn[ii, None]
        g = torch.randn(len(ii), W.shape[1], generator=gen, dtype=torch.float64)
        g -= (g * wh).sum(1, keepdim=True) * wh
        g /= g.norm(dim=1, keepdim=True)
        out[a:a + chunk] = Wn[ii, None] * (BETA * wh + c2 * g)
        del g, wh
    return out, s[idx].clone(), V[idx].clone(), idx


def synth_gate(backbone, T, role="last", fire="top1", response="repl"):
    """Un-thresholded scoped log of a synthetic log (no guard keys) + a record."""
    from rescope import ScopedLog
    t0 = time.time()
    W, s, V, idx = synth_factors(backbone, T)
    ar = torch.arange(T)
    S = ScopedLog(det_w=W, det_s=s, det_tau=torch.full((T,), -float("inf"), dtype=torch.float64), det_edit=ar, det_kind=torch.zeros(T, dtype=torch.long),
                  det_key=W, values=V, lin_w=W, lin_u=V, guard_key=torch.zeros(0, W.shape[1], dtype=torch.float64), guard_edit=ar[:0],
                  role=role, fire=fire, response=response, name=f"synthetic:{backbone}:T{T}", meta={"synthetic": True, "beta": BETA, "seed": SEED + T})
    return S, {"T": T, "backbone": backbone, "reference_log": REF[backbone], "beta": BETA, "seed": SEED + T, "build_s": time.time() - t0,
               "n_distinct_real_edits": int(len(set(idx.tolist()))), "rule": __doc__.split("SYNTHETIC LOG")[1][:1200]}


def cap_own(th):
    """Guard cap of a synthetic log: own coordinate 1 -> threshold at most 1 - TIE (the library's own_caps for one guard per edit)."""
    cap = torch.full_like(th, 1.0 - TIE)
    return torch.minimum(th, cap), int((th > cap).sum())


def calibrate_slim(S, kc, rows, which="tf", alpha=0.05, seed=0, cap=None):
    """Two-block calibration of a scoped log WITHOUT guard keys (library: KeyCache.coords + calibrate_two_block), then the own caps:
    cap (D,) = the cap of every detector ((1 - TIE) x its smallest own coordinate); None: synthetic log (own coordinate 1).
    -> (calibrated log, record with the seconds of the three stages)."""
    from rescope import calibrate_two_block
    t0 = time.perf_counter()
    Q = kc.coords(S, rows, which)
    t1 = time.perf_counter()
    S1, rec = calibrate_two_block(S, coords=Q, alpha=alpha, seed=seed, allow_silence=True)
    t2 = time.perf_counter()
    if cap is None:
        th, ncap = cap_own(S1.det_tau)
    else:
        ncap = int((S1.det_tau > cap).sum())
        th = torch.minimum(S1.det_tau, cap)
    S2 = S1.replace(det_tau=th)
    t3 = time.perf_counter()
    return S2, {"coords_s": t1 - t0, "rule_s": t2 - t1, "caps_s": t3 - t2, "total_s": t3 - t0, "n": len(rows), "n_edits_capped": ncap, "eta": rec["eta"],
                "bound": rec["bound_false_firing"], "Q": Q}


def real_gate_slim(L):
    """(un-thresholded default gate of a real log without guard keys, cap (T,) from the driver's guards: the stored solver key and the
    B = 1 key of the own Rel prompt)."""
    from rescope.calibrate import own_caps
    S, _ = fx.methods.base_gate(L)
    cap = own_caps(S)
    return S.replace(guard_key=S.guard_key[:0], guard_edit=S.guard_edit[:0]), cap
