"""white.common: paths, constants and model-free helpers of the data / feature layer of amendment "Review round 1", items R2 and R3
(10_final/PROTOCOL.md, section 6).

Nothing here writes outside 10_final/cache2 (data) and 10_final/feat2/white (code, tests, docs).  The library `rescope`, the shared
driver `fx` and the caches of 10_final/cache are READ-ONLY for this package: bytecode writing is switched off before they are
imported, so that no file appears in their directories.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):       # at most 4 BLAS threads per process
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"                       # the library rescope (read-only)
FINAL = ROOT + "/02_research/experiments/10_final"
DRIVER = FINAL + "/common"                             # the shared driver fx (read-only)
CACHE = FINAL + "/cache"                               # key caches, benign pool, hard negatives (read-only)
CACHE2 = FINAL + "/cache2"
PARTS_DIR = os.environ.get("WHITE_PARTS_DIR", CACHE2 + "/benign_parts")       # <backbone>.json      (benparts.py)
WHITEN_DIR = os.environ.get("WHITE_WHITEN_DIR", CACHE2 + "/whiten")           # <backbone>/...       (wstats.py)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
PYTHON = f"{_RP.PYTHON}"

BACKBONES = ("qwen25vl", "llava")                      # the backbones of the paper's main logs (the scout's blip2 pool is the same pool)
DEV_LOGS = {"qwen25vl": "TB-Q100-ae-l2_3", "llava": "TB-L100-ae-l2_3"}        # the ONLY logs on which a gate decision may be looked at
KAPPAS = (0.01, 0.1, 1.0)                              # ridge of the whitened gates: rho = kappa tr(C_W) / d_in


def use_library():
    """Put the library on sys.path (no bytecode is written) and import torch with at most 4 threads."""
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    import torch
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    return torch


def use_driver():
    """Import the shared driver fx (read-only) -> module."""
    use_library()
    if DRIVER not in sys.path:
        sys.path.insert(0, DRIVER)
    import fx
    return fx


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def jsonl_read(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def jdumps(obj, indent=1) -> str:
    """The ONE serialisation of this package's JSON files: sorted keys, fixed indent, no NaN, a final newline.  The same object
    always gives the same bytes (the determinism tests compare files byte for byte)."""
    return json.dumps(obj, indent=indent, sort_keys=True, allow_nan=False, ensure_ascii=True) + "\n"


def write_atomic(path: str, data) -> None:
    """Write bytes / text to `path` with one rename (a reader never sees a partial file)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data.encode("utf-8") if isinstance(data, str) else data)
    os.replace(tmp, path)


def norm_q(s) -> str:
    """Question-string identity of the caches (10_final/cache/code/fc_common.norm_q): stripped, lower case."""
    return (s or "").strip().lower()
