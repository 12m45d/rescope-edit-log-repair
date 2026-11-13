"""Shared paths and small helpers of the retrieval reference / lookup rule (PROTOCOL.md section 6, "Review round 1", item R4).

Writes only under 10_final/cache2/retr (data) and 10_final/feat2/retr (code, tests, outputs).  Everything else is read-only:
bytecode writing is switched off, so that importing the library, the driver or the testbed code leaves no file behind, and
the library's private harness cache is never touched (this layer does not import the harness).
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

sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):       # at most 4 BLAS threads per process
    os.environ.setdefault(_v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"
EXP = ROOT + "/02_research/experiments"
FINAL = EXP + "/10_final"
CACHE = FINAL + "/cache"                       # key caches, benign pool, hard negatives (read-only)
COMMON = FINAL + "/common"                     # the shared driver fx (read-only)
TESTBED = EXP + "/06_testbed_v3"
DATA = os.environ.get("RETR_DATA", FINAL + "/cache2/retr")      # data of this layer (RETR_DATA: a re-run into another directory)
EMB = DATA + "/emb"                            # embeddings shared by the backbones
CODE = FINAL + "/feat2/retr"
OUT = CODE + "/out"
MODELS = f"{_RP.EXT_VIEW}/data/models"
LLAVA_DIR = MODELS + "/llava-1.5-7b"           # only its CLIP vision tower and image processor are loaded
MINILM_DIR = MODELS + "/all-MiniLM-L6-v2"
GQ = ROOT + "/02_research/tools/gpuq/gq"
PYTHON = f"{_RP.PYTHON}"

BACKBONES = ("qwen25vl", "llava")              # key caches served (directory names under 10_final/cache)
DEV_LOGS = {"qwen25vl": "TB-Q100-ae-l2_3", "llava": "TB-L100-ae-l2_3"}     # the ONLY logs on which a gate decision may be looked at
SEQ_KINDS = ("rel", "tgen", "igen", "probe", "hn_i", "hn_q", "hn_x", "benign")
EDIT_KINDS = ("rel", "tgen", "igen")
HN_KINDS = ("hn_i", "hn_q", "hn_x")


def lib_path():
    """Make the library `rescope` importable (read-only)."""
    if LIB not in sys.path:
        sys.path.insert(0, LIB)


def fx_path():
    """Make the shared driver `fx` importable (read-only; used by the mechanics check and the tests, not by retr.py)."""
    lib_path()
    if COMMON not in sys.path:
        sys.path.insert(0, COMMON)


def norm_q(s) -> str:
    """Identity of a question string: stripped, lower case (the rule of 10_final/cache/code/fc_common.norm_q and fx.logs.norm_q)."""
    return (s or "").strip().lower()


def file_md5(path: str) -> str:
    """md5 of a file's bytes (the image content identifier)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def jload(path):
    with open(path) as f:
        return json.load(f)


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_default)
        f.write("\n")
    os.replace(tmp, path)


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def jsonl_read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_write(rows, path):
    """One JSON object per line, keys in the given order, UTF-8, '\\n' line ends; written with one rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def text_write(text: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, path)


def table(rows, head=None) -> str:
    """Plain-text table (left-aligned first column, right-aligned others)."""
    rows = [[("" if c is None else str(c)) for c in r] for r in ([head] if head else []) + list(rows)]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for k, r in enumerate(rows):
        out.append("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)).rstrip())
        if head and k == 0:
            out.append("  ".join("-" * x for x in w))
    return "\n".join(out)
