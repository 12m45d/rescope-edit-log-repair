"""Shared paths and helpers of the independent checker (feat2/verify).  Nothing here imports a builder's package.
Writes only under feat2/verify/."""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"
FINAL = ROOT + "/02_research/experiments/10_final"
CACHE = FINAL + "/cache"
CACHE2 = FINAL + "/cache2"
FEAT = FINAL + "/feat2"
VER = FEAT + "/verify"
OUT = VER + "/out"
TESTBED = ROOT + "/02_research/experiments/06_testbed_v3"
VLKEB_IMG = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"
MODELS = f"{_RP.EXT_VIEW}/data/models"
DEV = {"qwen25vl": "TB-Q100-ae-l2_3", "llava": "TB-L100-ae-l2_3"}
SEED = 9051117                      # the checker's own seed (not a seed of any builder)


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def jload(path):
    with open(path) as f:
        return json.load(f)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_d)
        f.write("\n")
    os.replace(tmp, path)


def _d(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def img_path(p):
    return p if os.path.isabs(p) else os.path.join(VLKEB_IMG, p)


def nq(s):
    return (s or "").strip().lower()


def lib():
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    import torch
    torch.set_num_threads(4)
    return torch
