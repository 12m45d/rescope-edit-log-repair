"""Shared helpers of the tests of white (10_final/feat2/white/tests).  Every test is a function that returns a dict with 'pass' (bool) and
'note' (one line for the table); the runner (run_tests.py) writes tests/out/<test>.json and renders tests/out/table.txt.
Temporary files live under tests/tmp/ (inside feat2) and are removed by the test that made them."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):       # at most 4 BLAS threads, set before numpy / torch load
    os.environ.setdefault(_v, "4")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"                                     # also for the child processes
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
FEAT2 = os.path.dirname(os.path.dirname(HERE))                                   # .../10_final/feat2
for _p in (HERE, FEAT2):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from white import common as C  # noqa: E402

OUT = os.path.join(HERE, "out")
TMP = os.path.join(HERE, "tmp")
TESTS = {}                                                                       # name -> (function, group, what it checks)
SEED = 20261111


def test(group: str, what: str):
    def deco(fn):
        TESTS[fn.__name__] = (fn, group, what)
        return fn
    return deco


def tmpdir(name: str) -> str:
    """A fresh empty directory tests/tmp/<name>."""
    d = os.path.join(TMP, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def rmtmp(name: str):
    shutil.rmtree(os.path.join(TMP, name), ignore_errors=True)
    if os.path.isdir(TMP) and not os.listdir(TMP):
        os.rmdir(TMP)


def child(module: str, args, env=None, timeout: int = 480):
    """Run `python -m <module> args...` in a fresh process (cwd = feat2) -> (exit code, stdout + stderr)."""
    e = dict(os.environ)
    e.update({k: str(v) for k, v in (env or {}).items()})
    r = subprocess.run([C.PYTHON, "-m", module] + [str(a) for a in args], cwd=FEAT2, env=e, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


_CTX = {}


def context(name: str):
    """white.mechanics.Context of a development log (edit log, key cache, row sets, draw 0, null sample, BEN-W statistics); cached."""
    if name not in _CTX:
        from white import mechanics
        _CTX[name] = mechanics.Context(name)
    return _CTX[name]


def maxabs(a, b) -> float:
    import numpy as np
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))) if np.asarray(a).size else 0.0
