"""E10 (public ROME logs of GPT-2 XL): shared helpers of the experiment scripts in this directory.

  paths, JSON, flat records (experiment, log, method, variant, alpha, n, draw, pool, metric, value, ci_lo, ci_hi, n_units)
  xboot / xboot_diff      the crossed bootstrap of amendment R10: calibration draws x units (records / edits), 2000 resamples,
                          paired form for differences (tested on synthetic data: t_boot.py)
Nothing here changes the adapter (e10t), the library (rescope) or the driver (fx: only fx.draws.ranking, fx.io, fx.stats are
used, through the adapter).
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import gzip
import json
import os
import sys
import time

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
E10T = FINAL + "/e10_text"
RUN = E10T + "/run"
OUT = RUN + "/out"
CODE = RUN + "/code"
AUDIT = f"{_RP.ROOT}/02_research/experiments/09_testbed_scale/text_public/audit"
TXOUT = f"{_RP.ROOT}/02_research/experiments/09_testbed_scale/text_public/out"
# sys.path: the frozen minimal copy of the driver fx FIRST (see fx_frozen/fx/__init__.py), then the adapter and the feature layer.
# The adapter inserts LIB / COMMON / TXC only if they are not on the path yet: they are appended here so that the frozen copy wins.
for _p in (f"{_RP.ROOT}/02_research/code", FINAL + "/common",
           f"{_RP.ROOT}/02_research/experiments/09_testbed_scale/text_public/code"):
    if _p not in sys.path:
        sys.path.append(_p)
for _p in (FINAL + "/feat2", E10T + "/code", CODE + "/fx_frozen"):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.environ.setdefault("E10T_STORE_ROOT", RUN + "/store")            # behavioural outcomes of THIS experiment live under run/

import numpy as np  # noqa: E402

EXPERIMENT = "E10"
LOGS = ("TX-G2XL-rome-T100", "TX-G2XL-rome-T500", "TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000")
BEHAVE_LOGS = ("TX-G2XL-rome-T100", "TX-G2XL-rome-T500")
DEV = "TX-G2XL-rome-dev100"
SPLIT = 173                         # R21 (b): fixed by the protocol
ALPHA, N_CAL, N_DRAWS, BEHAVE_DRAWS = 0.05, 200, 20, (0, 1, 2)
N_BOOT = 2000
FIELDS = ("experiment", "log", "method", "variant", "alpha", "n", "draw", "pool", "metric", "value", "ci_lo", "ci_hi", "n_units")


def say(*a):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)


def clean(x):
    import math
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


class Records:
    """Flat records of one block of the experiment -> out/records/<block>.json (merged into out/results.json by render.py)."""

    def __init__(self, block: str):
        self.block, self.recs = block, []

    def add(self, log, method, metric, value, variant="", alpha=None, n=None, draw=None, pool="", ci_lo=None, ci_hi=None, n_units=None):
        self.recs.append({"experiment": EXPERIMENT, "log": log, "method": method, "variant": variant, "alpha": alpha, "n": n, "draw": draw,
                          "pool": pool, "metric": metric, "value": value, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_units": n_units})

    def stat(self, log, method, metric, st, **kw):
        self.add(log, method, metric, st.get("value"), ci_lo=st.get("ci_lo"), ci_hi=st.get("ci_hi"), n_units=st.get("n_units"), **kw)

    def save(self):
        jdump(self.recs, os.path.join(OUT, "records", self.block + ".json"), indent=None)


# ----------------------------------------------------------------------------- crossed bootstrap (R10)
def _unit_sums(Y, units):
    """Y (D, R) float with NaN = missing, units (R,) -> S (D, U) sums, N (D, U) counts."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    u = np.arange(Y.shape[1]) if units is None else np.asarray(list(units))
    _, inv = np.unique(u, return_inverse=True)
    U = int(inv.max()) + 1 if inv.size else 0
    ok = ~np.isnan(Y)
    S = np.stack([np.bincount(inv, weights=np.where(ok[d], Y[d], 0.0), minlength=U) for d in range(Y.shape[0])])
    N = np.stack([np.bincount(inv, weights=ok[d].astype(np.float64), minlength=U) for d in range(Y.shape[0])])
    return S, N


_WCACHE = {}


def _weights(D, U, n_boot, seed):
    """Resampling counts: draws (n_boot, D) and units (n_boot, U).  The unit resamples depend on (seed, U) only, the draw
    resamples on (seed, D) only: a row without calibration (D = 1) and a calibrated row share their unit resamples."""
    kd, ku = ("d", int(seed), D, n_boot), ("u", int(seed), U, n_boot)
    if kd not in _WCACHE:
        idx = np.random.default_rng([int(seed), 1]).integers(0, D, size=(n_boot, D))
        _WCACHE[kd] = np.stack([np.bincount(r, minlength=D) for r in idx]).astype(np.float64)
    if ku not in _WCACHE:
        ru = np.random.default_rng([int(seed), 2])
        Wu = np.zeros((n_boot, U), dtype=np.float32)
        for a in range(n_boot):
            Wu[a] = np.bincount(ru.integers(0, U, size=U), minlength=U)
        while sum(1 for k in _WCACHE if k[0] == "u") >= 16:
            _WCACHE.pop(next(k for k in _WCACHE if k[0] == "u"))
        _WCACHE[ku] = Wu
    return _WCACHE[kd], _WCACHE[ku]


def _boot_means(S, N, Wd, Wu):
    num = ((Wd @ S) * Wu).sum(1)
    den = ((Wd @ N) * Wu).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def xboot(Y, units=None, n_boot: int = N_BOOT, seed: int = 0, scale: float = 100.0, ci: float = 95.0) -> dict:
    """Crossed bootstrap of scale * mean(Y) over calibration draws (rows of Y) and units (groups of its columns): each resample
    draws D draws and U units with replacement, independently, and takes the mean over the resampled grid (a unit comes with
    all its rows).  Y (D, R) (or (R,): one draw = the plain unit bootstrap); NaN / None = missing.
    -> {value, ci_lo, ci_hi (percentile), n_units, n_draws, n_rows, se}."""
    Y = np.array([[np.nan if v is None else float(v) for v in row] for row in (Y if np.ndim(Y) == 2 else [Y])], dtype=np.float64)
    S, N = _unit_sums(Y, units)
    if N.sum() == 0:
        return {"value": None, "ci_lo": None, "ci_hi": None, "n_units": 0, "n_draws": int(Y.shape[0]), "n_rows": int(Y.shape[1])}
    Wd, Wu = _weights(S.shape[0], S.shape[1], n_boot, seed)
    b = scale * _boot_means(S, N, Wd, Wu)
    b = b[~np.isnan(b)]
    lo = (100.0 - ci) / 2.0
    return {"value": float(scale * S.sum() / N.sum()), "ci_lo": float(np.percentile(b, lo)), "ci_hi": float(np.percentile(b, 100.0 - lo)),
            "n_units": int(S.shape[1]), "n_draws": int(S.shape[0]), "n_rows": int(Y.shape[1]), "se": float(b.std(ddof=1))}


def xboot_diff(Ya, Yb, units=None, n_boot: int = N_BOOT, seed: int = 0, scale: float = 100.0, ci: float = 95.0) -> dict:
    """Paired crossed bootstrap of scale * (mean(Ya) - mean(Yb)): the SAME resampled draws and units for both (columns of Ya and
    Yb are the same rows; a method without calibration (one row) is repeated over the draws of the other)."""
    A = np.array([[np.nan if v is None else float(v) for v in row] for row in (Ya if np.ndim(Ya) == 2 else [Ya])], dtype=np.float64)
    B = np.array([[np.nan if v is None else float(v) for v in row] for row in (Yb if np.ndim(Yb) == 2 else [Yb])], dtype=np.float64)
    D = max(A.shape[0], B.shape[0])
    if A.shape[0] == 1:
        A = np.repeat(A, D, 0)
    if B.shape[0] == 1:
        B = np.repeat(B, D, 0)
    if A.shape != B.shape:
        raise ValueError("paired outcomes need the same rows and draws")
    Sa, Na = _unit_sums(A, units)
    Sb, Nb = _unit_sums(B, units)
    Wd, Wu = _weights(D, Sa.shape[1], n_boot, seed)
    d = scale * (_boot_means(Sa, Na, Wd, Wu) - _boot_means(Sb, Nb, Wd, Wu))
    d = d[~np.isnan(d)]
    lo = (100.0 - ci) / 2.0
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"value": float(scale * (Sa.sum() / Na.sum() - Sb.sum() / Nb.sum())), "ci_lo": float(np.percentile(d, lo)),
            "ci_hi": float(np.percentile(d, 100.0 - lo)), "n_units": int(Sa.shape[1]), "n_draws": int(D), "se": float(d.std(ddof=1)),
            "p_two_sided": float(min(1.0, max(p, 1.0 / n_boot)))}


def over_draws(v) -> dict:
    v = np.asarray([float(x) for x in v if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n_draws": 0, "mean": None, "p05": None, "p95": None, "min": None, "max": None}
    return {"n_draws": int(v.size), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}
