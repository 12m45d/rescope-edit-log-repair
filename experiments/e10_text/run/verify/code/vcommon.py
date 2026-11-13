"""E10 verification: shared helpers of the verifier's scripts (independent of the experiment's code under run/code).

  paths; JSON; the verifier's own crossed bootstrap of amendment R10 (calibration draws x units, 2000 resamples; paired form
  for differences; test on synthetic data: v0_boot_test.py); sampling of records.
Imports: the library (rescope), the adapter (e10t), the builder's loader (tx_public) and the public evaluation code only.  The
driver fx is the verifier's shim (code/fxshim: the three reviewed files the adapter needs, hash-checked).
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import gzip
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
E10T = FINAL + "/e10_text"
RUN = E10T + "/run"                 # the experiment under verification (read only)
XOUT = RUN + "/out"
VER = RUN + "/verify"
OUT = VER + "/out"
CODE = VER + "/code"
TXP = f"{_RP.ROOT}/02_research/experiments/09_testbed_scale/text_public"
AUDIT = TXP + "/audit"
TXOUT = TXP + "/out"
# the adapter inserts LIB / COMMON / TXC in front of sys.path only if they are absent: they are appended here, so that the
# verifier's fx shim (first on the path) wins over the live driver directory, which is never imported
for _p in (f"{_RP.ROOT}/02_research/code", FINAL + "/common", TXP + "/code"):
    if _p not in sys.path:
        sys.path.append(_p)
for _p in (E10T + "/code", CODE + "/fxshim"):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
# the verifier never reads or writes the experiment's outcome store: its own (unused: store=False everywhere) root
os.environ["E10T_STORE_ROOT"] = VER + "/store_unused"

import numpy as np  # noqa: E402

SPLIT = 173
ALPHA, N_CAL = 0.05, 200
N_BOOT = 2000
T100, T500, T2000 = "TX-G2XL-rome-T100", "TX-G2XL-rome-T500", "TX-G2XL-rome-T2000"


def say(*a):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if hasattr(x, "numel") and hasattr(x, "tolist"):
        return clean(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, indent=indent, allow_nan=False)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def save_rows(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(clean(r)) + "\n")
    os.replace(tmp, path)


def load_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(x) for x in f]


def sample_records(n_pool: int, k: int, seed: int) -> list:
    """k of range(n_pool) (all if k >= n_pool), sorted; the verifier's own seeded sample."""
    if k >= n_pool:
        return list(range(n_pool))
    return sorted(int(i) for i in np.random.default_rng([20261111, int(seed)]).choice(n_pool, size=k, replace=False))


# ----------------------------------------------------------------------------- crossed bootstrap (R10), verifier's implementation
def crossed_bootstrap(Y, units=None, n_boot: int = N_BOOT, seed: int = 12345, scale: float = 100.0, level: float = 95.0) -> dict:
    """Y: (D, R) outcomes of D calibration draws on the same R rows (None / NaN = missing), or (R,) for one draw.  units: (R,)
    unit of each row (record / edit); a unit is resampled with all its rows.  Each resample draws D draws and U units with
    replacement, independently (crossed), and takes the pooled mean over the resampled grid.  Percentile interval."""
    Y = np.array([[np.nan if v is None else float(v) for v in row] for row in (Y if np.ndim(Y) == 2 else [Y])], dtype=np.float64)
    D, R = Y.shape
    u = np.arange(R) if units is None else np.unique(np.asarray(list(units)), return_inverse=True)[1]
    U = int(u.max()) + 1 if R else 0
    ok = ~np.isnan(Y)
    num = np.zeros((D, U))
    den = np.zeros((D, U))
    for d in range(D):
        np.add.at(num[d], u[ok[d]], Y[d, ok[d]])
        np.add.at(den[d], u[ok[d]], 1.0)
    if den.sum() == 0:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n_units": 0, "n_draws": D}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        dd = rng.integers(0, D, size=D)
        uu = rng.integers(0, U, size=U)
        a = num[np.ix_(dd, uu)].sum()
        c = den[np.ix_(dd, uu)].sum()
        vals[b] = scale * a / c if c > 0 else np.nan
    vals = vals[~np.isnan(vals)]
    lo = (100.0 - level) / 2.0
    return {"value": float(scale * num.sum() / den.sum()), "ci_lo": float(np.percentile(vals, lo)), "ci_hi": float(np.percentile(vals, 100.0 - lo)),
            "n_units": U, "n_draws": D, "se": float(vals.std(ddof=1))}


def crossed_bootstrap_diff(Ya, Yb, units=None, n_boot: int = N_BOOT, seed: int = 12345, scale: float = 100.0, level: float = 95.0) -> dict:
    """Paired form: the same resampled draws and units for both methods (same rows; a method with one draw is repeated)."""
    A = np.array([[np.nan if v is None else float(v) for v in row] for row in (Ya if np.ndim(Ya) == 2 else [Ya])], dtype=np.float64)
    B = np.array([[np.nan if v is None else float(v) for v in row] for row in (Yb if np.ndim(Yb) == 2 else [Yb])], dtype=np.float64)
    D = max(A.shape[0], B.shape[0])
    A = np.repeat(A, D, 0) if A.shape[0] == 1 else A
    B = np.repeat(B, D, 0) if B.shape[0] == 1 else B
    R = A.shape[1]
    u = np.arange(R) if units is None else np.unique(np.asarray(list(units)), return_inverse=True)[1]
    U = int(u.max()) + 1

    def sums(Y):
        ok = ~np.isnan(Y)
        num, den = np.zeros((D, U)), np.zeros((D, U))
        for d in range(D):
            np.add.at(num[d], u[ok[d]], Y[d, ok[d]])
            np.add.at(den[d], u[ok[d]], 1.0)
        return num, den
    na, da = sums(A)
    nb, db = sums(B)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        dd = rng.integers(0, D, size=D)
        uu = rng.integers(0, U, size=U)
        ix = np.ix_(dd, uu)
        vals[b] = scale * (na[ix].sum() / da[ix].sum() - nb[ix].sum() / db[ix].sum())
    lo = (100.0 - level) / 2.0
    return {"value": float(scale * (na.sum() / da.sum() - nb.sum() / db.sum())), "ci_lo": float(np.percentile(vals, lo)),
            "ci_hi": float(np.percentile(vals, 100.0 - lo)), "n_units": U, "n_draws": D, "se": float(vals.std(ddof=1))}
