"""Independent verification of the fix-up read-outs (x_release/verify): shared helpers.  Own code throughout: rows are read with
gzip + json (the experiments' loader fx.io.load_rows does the same and filters nothing: read in common/fx/io.py); no function of
the experiments (fx.*, xlib, release_compute, fixup_*) is imported or called."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import gzip
import json
import math
import os
import re
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext
from fractions import Fraction

import numpy as np

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
ROOT = f"{_RP.ROOT}"
V = FINAL + "/x_release/verify"
OUT = V + "/out"

QLOGS = ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw")
LLOGS = ("TB-L100-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")
DEV = {"Q": QLOGS[0], "L": LLOGS[0]}
EXP = {"Q": "e3_qwen", "L": "e3_llava"}
LATER_FROM = {"Q": 500, "L": 100}
ROMAN = {100: "C", 200: "CC", 500: "D", 1000: "M", 2000: "MM"}
TW = {100: "Hundred", 200: "TwoHundred", 500: "FiveHundred", 1000: "Thousand"}
BOUND = 100.0 * 5.0 / 101.0                       # two-block rule, alpha 0.05, n 200: 5 / 101


def bb(log):
    return "Q" if log.startswith("TB-Q") else "L"


def T_of(log):
    return int(re.match(r"TB-[QL](\d+)-", log).group(1))


def fam(log):
    return "memitw" if log.endswith("-memitw") else "ae-l2_3"


def code(log):
    b, T = bb(log), T_of(log)
    return b + ("Ae" if fam(log) == "ae-l2_3" else "Me") + (ROMAN[T] if b == "Q" else TW[T])


def bdraws(log):
    return (0,) if T_of(log) >= 1000 else (0, 1, 2)


def jload(p):
    with open(p) as f:
        return json.load(f)


_cache = {}


def rows(path, keep=True):
    if path in _cache:
        return _cache[path]
    with gzip.open(path, "rt") as f:
        r = [json.loads(l) for l in f if l.strip()]
    if keep:
        _cache[path] = r
    return r


def forget():
    _cache.clear()


def e3_path(log, method, d, part="tf"):
    return f"{FINAL}/{EXP[bb(log)]}/out/rows/{log}__{method}__d{'x' if d is None else d}__{part}.jsonl.gz"


def rel(p):
    return os.path.relpath(p, FINAL)


# ---------------------------------------------------------------------------- pools (my reading of the row fields)
def is_tf(r):
    return r.get("lp", "tf") == "tf" and not r.get("d2_row")


def is_hd(r):
    """valid heldout_disjoint row of the teacher-forced part"""
    return r["kind"] == "probe" and bool(r.get("disjoint")) and is_tf(r)


def is_hn(r, t=None):
    return r["kind"] == "hn" and r.get("part") != "gen" and is_tf(r) and (t is None or r["hn_type"] == t)


def is_edit(r, metric=None):
    return r["kind"] == "edit" and is_tf(r) and (metric is None or r["metric"] == metric)


def pct(flags):
    flags = list(flags)
    return 100.0 * sum(1 for x in flags if x) / len(flags) if flags else None


def mean(v):
    v = [float(x) for x in v if x is not None]
    return sum(v) / len(v) if v else None


# ---------------------------------------------------------------------------- statistics (own implementations)
def mcnemar_exact(b, c):
    """two-sided exact binomial test on the discordant pairs, in exact rational arithmetic"""
    n, k = b + c, min(b, c)
    if n == 0 or b == c:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = Fraction(2 * tail, 2 ** n)
    return float(min(Fraction(1), p))


def holm(ps):
    """{key: p} -> {key: adjusted p}; step-down, monotone"""
    keys = sorted(ps, key=lambda k: ps[k])
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def crossed_boot_diff(A, B, n_boot=20000, seed=20261112, chunk=2000):
    """A, B: (draws, units) arrays of per-unit means in [0, 1] (the same draws and units).  Paired crossed bootstrap: draws and
    units resampled with replacement, independently, the same resample for both methods; statistic 100 * (mean A - mean B).
    One draw: only the units are resampled.  -> value, percentile 95 % interval, se, two-sided p (share on the other side of 0,
    doubled, floor 1 / n_boot), n_boot."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    assert A.shape == B.shape and A.ndim == 2
    nd, nu = A.shape
    D = A - B
    rng = np.random.default_rng(seed)
    st = []
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        cu = rng.multinomial(nu, np.full(nu, 1.0 / nu), size=m).astype(np.float64)
        cd = rng.multinomial(nd, np.full(nd, 1.0 / nd), size=m).astype(np.float64) if nd > 1 else np.ones((m, 1))
        st.append(100.0 * ((cd @ D) * cu).sum(1) / (nd * nu))
    st = np.concatenate(st)
    p = 2.0 * min(float((st <= 0).mean()), float((st >= 0).mean()))
    return {"value": 100.0 * float(D.mean()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5)), "se": float(st.std(ddof=1)),
            "p": float(min(1.0, max(p, 1.0 / n_boot))), "n_other_side": int(min((st <= 0).sum(), (st >= 0).sum())), "n_boot": int(n_boot), "n_draws": nd, "n_units": nu}


# ---------------------------------------------------------------------------- printing rule of the paper (gen_numbers.fmt_number)
def fmt_number(fmt, value):
    if fmt is None:
        return str(value)
    m = re.fullmatch(r"\{:\.(\d+)f\}", fmt)
    if m and isinstance(value, float) and value == value and abs(value) != float("inf"):
        n = int(m.group(1))
        with localcontext() as ctx:
            ctx.prec = 400
            d = Decimal(repr(value)).quantize(Decimal(1).scaleb(-(n + 9)), rounding=ROUND_HALF_EVEN)
            return format(d.quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP), "f")
    if m and isinstance(value, int) and not isinstance(value, bool):
        return fmt.format(value)
    return fmt.format(value)


def fx(v, n=1):
    s = fmt_number("{:.%df}" % n, float(v))
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def close(a, b, tol=1e-9):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
