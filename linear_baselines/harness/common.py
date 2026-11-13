"""Paths, imports and small helpers shared by the harness modules."""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import json
import math
import os
import sys
import time

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
EXP = f"{_RP.ROOT}/02_research/experiments"
TB_CODE = os.path.join(EXP, "06_testbed_v3", "code")
TB_ROOT = os.path.join(EXP, "06_testbed_v3")
CACHE = os.path.join(HARNESS_DIR, "cache")
if TB_CODE not in sys.path:
    sys.path.insert(0, TB_CODE)
os.makedirs(CACHE, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tb_common as TB  # noqa: E402  (re-points vendor caches to 06_testbed_v3/_shared; no writes here)
from tb_load import TestbedLog  # noqa: E402
from vendor_v7 import metrics as VM  # noqa: E402,F401


class _NotReleased:
    """Placeholder for a module of the editor that wrote the vision-language logs.  The editor is described in a related submission of
    the authors that is under review and is not part of this repository; the entry points that re-run it (REEDIT, COMP with
    solve="alphaedit") raise here.  Their per-row records and repaired-weight hashes are released (linear_baselines/streams/out)."""

    def __init__(self, what):
        object.__setattr__(self, "_what", what)

    def __getattr__(self, name):
        raise NotImplementedError(f"{self._what}.{name}: the editor that wrote the vision-language logs is not released (README, section "
                                  "\"What can be reproduced\"); REEDIT and COMP with solve=\"alphaedit\" cannot be re-run")


VE, VF, VZ = _NotReleased("editor"), _NotReleased("filters"), _NotReleased("target optimisation")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """CUDA-synchronised wall-clock."""

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.sec = time.perf_counter() - self.t0
        return False


def sha(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def clean(o):
    """JSON-safe copy (tensors / numpy / nan)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, torch.Tensor):
        return clean(o.tolist())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=1)


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()
