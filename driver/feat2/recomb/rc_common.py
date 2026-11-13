"""rc_common: paths, seed and model-free helpers of the BUILD scripts of the recombination layer (rc_select.py, rc_check.py,
rc_capture.py, rc_assemble.py, rc_mechanics.py).  Amendment "Review round 1", item R5, of 10_final/PROTOCOL.md.

The functions that the experiment driver uses live in recomb.py; that module does not import this one.

Nothing here writes outside 10_final/cache2/recomb (data) and 10_final/feat2/recomb (code, tests, outputs).  The key caches'
build code (10_final/cache/code: fc_common, fc_capture, fc_cats), the library `rescope` and the harness are imported
READ-ONLY: bytecode writing is off, the library's private harness cache is re-pointed to cache2/recomb/work/hcache
(RESCOPE_CACHE, set before anything imports the library), and the image-md5 memo of the caches is read, never flushed.
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
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):       # at most 4 BLAS threads per process
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

ROOT = f"{_RP.ROOT}"
FINAL = ROOT + "/02_research/experiments/10_final"
FEAT = FINAL + "/feat2/recomb"                    # code, tests, docs, out/
DATA = FINAL + "/cache2/recomb"                   # <backbone>/rows.jsonl, keys.pt, README.txt; work/; logs/
WORK = DATA + "/work"
LOGDIR = DATA + "/logs"
CACHE = FINAL + "/cache"                          # the existing key caches (read-only)
CACHE_CODE = CACHE + "/code"
GQ = ROOT + "/02_research/tools/gpuq/gq"
PYTHON = f"{_RP.PYTHON}"

os.environ.setdefault("RESCOPE_CACHE", WORK + "/hcache")       # BEFORE fc_common (which would default to 10_final/cache/work/hcache)
if CACHE_CODE not in sys.path:
    sys.path.insert(0, CACHE_CODE)
import fc_common as C  # noqa: E402  (paths of the library / harness / registry; BACKBONES; norm_q, mid, image_md5, query_key, family)

assert os.environ["RESCOPE_CACHE"].startswith(DATA), "the library's harness cache must be private to this build"

SEED = 20261111                                   # every seeded draw of the recombination selection derives from it
M = 4                                             # recombinations of each kind per edit (amendment R5)
KINDS = ("X-I", "X-Q", "X-X")
BACKBONES = ("qwen25vl", "llava")
DEV = {"qwen25vl": "TB-Q100-ae-l2_3", "llava": "TB-L100-ae-l2_3"}       # the ONLY logs on which a gate may be looked at
TEST_CACHES = ("qwen25vl", "llava", "QO", "blip2")                       # every key cache under 10_final/cache when the selection was made (rule b)
PREFIXES = {"qwen25vl": (10, 20, 50, 100, 200, 500, 1000, 2000), "llava": (10, 20, 50, 100, 200, 500, 1000)}   # registered log lengths


def say(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def useed(*parts) -> int:
    """Deterministic 32-bit seed from SEED and any labels (sha1), the form of fc_common.useed with this build's seed."""
    return int(hashlib.sha1("|".join(str(p) for p in (SEED,) + parts).encode()).hexdigest()[:8], 16)


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl_read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_write(rows, path):
    """One JSON object per line, keys in insertion order, ASCII: the bytes are a function of the rows alone."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def bb_dir(backbone: str) -> str:
    return os.path.join(DATA, backbone)


def rows_path(backbone: str) -> str:
    return os.path.join(DATA, backbone, "rows.jsonl")


def keys_path(backbone: str) -> str:
    return os.path.join(DATA, backbone, "keys.pt")


def text_table(rows, head=None) -> str:
    """Plain-text table: list of tuples of cells (strings)."""
    rows = [tuple(str(c) for c in r) for r in ([head] if head else []) + list(rows)]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    if head:
        out.insert(1, "  ".join("-" * x for x in w))
    return "\n".join(out)
