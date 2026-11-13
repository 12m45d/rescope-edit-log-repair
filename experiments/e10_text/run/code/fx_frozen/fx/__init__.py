"""FROZEN, MINIMAL copy of the shared driver package fx for experiment E10 (e10_text/run).

Why: the shared driver 10_final/common/fx is being extended by other agents while E10 runs (its __init__ imported a module that
did not exist yet at 2026-11-11 13:17 UTC, which stopped six E10 jobs at import).  The adapter e10t uses three things of the
driver: fx.draws.ranking, fx.io (clean / Results helpers) and fx.stats.  This package holds byte-identical copies of draws.py,
io.py and stats.py in the versions the adapter's tests and development-log results ran with (sha1: draws.py 79bb3a2819dc, io.py
26e6a1349917, stats.py 6a3cf7ff2965 = 10_final/common/review/orig_stage1) and this reduced __init__ (constants and
code_fingerprint only).  Nothing else of the driver is imported by E10.  xcommon puts this directory first on sys.path.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

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

VERSION = "fx frozen for E10: draws.py, io.py, stats.py of fx b92d0532498f (review/orig_stage1)"
FROZEN = True


def code_fingerprint() -> dict:
    import hashlib

    def fp(folder):
        h = hashlib.sha1()
        for f in sorted(os.listdir(folder)):
            if f.endswith(".py"):
                h.update(f.encode())
                with open(os.path.join(folder, f), "rb") as fh:
                    h.update(fh.read())
        return h.hexdigest()[:12]
    return {"rescope": fp(os.path.join(LIB, "rescope")), "fx": fp(os.path.dirname(os.path.abspath(__file__))), "fx_version": VERSION}


from . import stats, io, draws  # noqa: E402,F401
