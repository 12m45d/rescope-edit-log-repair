"""Verification of the secondary experiments: common paths (own code; fx used for data access only)."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, os, sys, time, hashlib
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import numpy as np, torch
torch.set_num_threads(4)
HERE = FINAL + "/e5_rescope/verify_secondary"
OUT = HERE + "/out"
QDEV, LDEV = "TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"

def jload(p):
    with open(p) as f: return json.load(f)
def jdump(o, p):
    tmp = p + ".tmp"
    with open(tmp, "w") as f: json.dump(o, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    os.replace(tmp, p)
