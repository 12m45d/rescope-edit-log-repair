"""E5: common paths and helpers (driver fx, library rescope; 4 BLAS threads)."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json, os, sys, time
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np, torch
import fx
torch.set_num_threads(4)
HERE = FINAL + "/e5_rescope"
OUT = HERE + "/out"
DEVS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
LOGS_A = ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw",
          "TB-L100-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")
LOGS_C_GATE = ("TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw", "TB-L100-memitw", "TB-L1000-memitw")

def jdump(o, p): fx.io.jdump(o, p, indent=1)
def jload(p):
    with open(p) as f: return json.load(f)
def qs(v):
    v = np.asarray(v, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "mean": float(v.mean()), "max": float(v.max()), "n": int(v.size)}
