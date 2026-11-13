"""Verifier's shim of the shared driver package fx (E10 verification).

The live driver 10_final/common/fx is being extended by other agents (its draws.py / stats.py no longer have the hashes the
adapter e10t was tested with).  The adapter needs fx.draws.ranking, fx.io and fx.stats only.  This shim exposes exactly these
three modules, as symlinks to the reviewed originals 10_final/common/review/orig_stage1/{draws,io,stats}.py, and refuses to
import if their sha1 is not the one recorded in the adapter's USAGE.txt (79bb3a2819dc / 26e6a1349917 / 6a3cf7ff2965).
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"
EXP = ROOT + "/02_research/experiments"
FINAL = EXP + "/10_final"
COMMON = FINAL + "/common"
P12 = EXP + "/08_rescope/p12_linear_streams"
PYTHON = f"{_RP.PYTHON}"
if LIB not in sys.path:
    sys.path.insert(0, LIB)
VERSION = "fx shim of the E10 verifier: draws.py, io.py, stats.py of common/review/orig_stage1"
EXPECT = {"draws.py": "79bb3a2819dc", "io.py": "26e6a1349917", "stats.py": "6a3cf7ff2965"}
HASHES = {}
_d = os.path.dirname(os.path.abspath(__file__))
for _f, _h in EXPECT.items():
    with open(os.path.join(_d, _f), "rb") as _fh:
        HASHES[_f] = hashlib.sha1(_fh.read()).hexdigest()[:12]
    if HASHES[_f] != _h:
        raise ImportError(f"fx shim: {_f} has sha1 {HASHES[_f]}, expected {_h}")


def code_fingerprint() -> dict:
    return {"fx": "shim:" + ",".join(f"{k}={v}" for k, v in sorted(HASHES.items())), "fx_version": VERSION}


from . import stats, io, draws  # noqa: E402,F401
