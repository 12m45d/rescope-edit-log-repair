"""mme_common: paths, seed and model-free helpers of the MME capture.

PROTOCOL.md of 10_final, section 6, amendment "Clarification of H3 (R6) and R17", item (2) = R17: the MME benchmark (local
copy; yes / no questions about images of other sources) is captured under the PRE-EDIT model of both backbones like the
benign pool and split by image, seed 20261111, into MME-C and MME-T.  This package captures only: no gate, no calibration
and no edited model is run on these rows.

Nothing here writes outside 10_final/cache2/mme.  The caches' build code (10_final/cache/code: fc_common, fc_capture), the
library `rescope`, the harness and the testbed are imported READ-ONLY: bytecode writing is off, the library's private
harness cache is re-pointed to cache2/mme/work/hcache (RESCOPE_CACHE, set before anything imports the library), and the
image-md5 memo of the caches is read, never flushed.

Importing this module is cheap (no torch, no testbed).  cache_code() imports the caches' fc_common on demand.
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

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):       # at most 4 BLAS threads per process
    os.environ[_v] = str(min(4, int(os.environ.get(_v, "4") or 4)))
sys.dont_write_bytecode = True

ROOT = f"{_RP.ROOT}"
FINAL = ROOT + "/02_research/experiments/10_final"
MME = FINAL + "/cache2/mme"                       # everything this package writes
CODE = MME + "/code"
IMAGES = MME + "/images"                          # <content md5>.<ext>
WORK = MME + "/work"
LOGDIR = MME + "/logs"
SELECT = MME + "/select"                          # the selection in use
ROW_TABLE = MME + "/mme_rows.jsonl"               # one line per row of the benchmark (step 1)
CACHE = FINAL + "/cache"                          # the existing key caches (read-only)
CACHE_CODE = CACHE + "/code"
CACHE2 = FINAL + "/cache2"
SRC = f"{_RP.EXT_VIEW}/data/datasets/MME"      # read-only
GQ = ROOT + "/02_research/tools/gpuq/gq"
PYTHON = f"{_RP.PYTHON}"

SEED = 20261111                                   # every seeded draw of this package derives from it (R17)
PARTS = ("MME-C", "MME-T")
BACKBONES = ("qwen25vl", "llava")
LAYER = {"qwen25vl": 20, "llava": 16}
GQ_MEM = {"qwen25vl": 10500, "llava": 17000}
SUFFIX = "Please answer yes or no."               # every MME question ends with it
QUERY_ID0 = 1_000_000                             # query ids of the MME rows start here (no collision with the caches' ids)
N_PROOF = 30                                      # rows of the existing benign pool re-captured before the long run
N_VERIFY = 50                                     # MME rows re-captured in a fresh process after the run
PART_SIZE = 50                                    # rows per capture part (resumable)
CAPTURE_WORK = WORK + "/capture"                  # fc_capture's WORK for the MME rows:   <backbone>/select/manifest.jsonl, <backbone>/parts/
PROOF_WORK = WORK + "/proof"                      # fc_capture's WORK for the proof rows (30 existing benign rows per backbone)

os.environ["RESCOPE_CACHE"] = WORK + "/hcache"   # BEFORE fc_common / rescope are imported (else: 10_final/cache/work/hcache)
_C = None


def cache_code():
    """The caches' fc_common (paths of the library / harness / registry; BACKBONES; norm_q; degenerate_flags; say; json helpers)."""
    global _C
    if _C is None:
        if CACHE_CODE not in sys.path:
            sys.path.insert(0, CACHE_CODE)
        import fc_common as C
        assert os.environ["RESCOPE_CACHE"].startswith(MME), "the library's harness cache must be private to this build"
        _C = C
    return _C


def say(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def useed(*parts) -> int:
    """Deterministic 32-bit seed from SEED and any labels (sha1): the form of fc_common.useed with this package's seed."""
    return int(hashlib.sha1("|".join(str(p) for p in (SEED,) + parts).encode()).hexdigest()[:8], 16)


def norm_q(s) -> str:
    """Question-string identity of the caches (fc_common.norm_q): stripped, lower case."""
    return (s or "").strip().lower()


def jdumps(obj, indent=1, default=None) -> str:
    """Deterministic JSON text: insertion order of keys, ASCII, no NaN, a final newline.  default: json's fallback for values
    that are not JSON types (None: such a value is an error; result files of GPU runs pass str, so that a result is never lost)."""
    return json.dumps(obj, indent=indent, allow_nan=False, ensure_ascii=True, default=default) + "\n"


def write_atomic(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data.encode("utf-8") if isinstance(data, str) else data)
    os.replace(tmp, path)


def jdump(obj, path, indent=1, default=None):
    write_atomic(path, jdumps(obj, indent, default))


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl_read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def jsonl_text(rows) -> str:
    """One JSON object per line, keys in insertion order, ASCII: the bytes are a function of the rows alone."""
    return "".join(json.dumps(r, allow_nan=False, ensure_ascii=True) + "\n" for r in rows)


def jsonl_write(rows, path):
    write_atomic(path, jsonl_text(rows))


def md5_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha1_file(path: str, n: int = 12) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def text_table(rows, head=None, right=()) -> str:
    """Plain-text table from tuples of cells; `right`: indices of right-aligned columns."""
    rows = [tuple(str(c) for c in r) for r in ([head] if head else []) + list(rows)]
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["  ".join((c.rjust(w[i]) if i in right else c.ljust(w[i])) for i, c in enumerate(r)).rstrip() for r in rows]
    if head:
        out.insert(1, "  ".join("-" * x for x in w))
    return "\n".join(out)


def pct(a, b, nd=1) -> str:
    return "-" if not b else f"{100.0 * a / b:.{nd}f}"


def code_fingerprint() -> dict:
    """sha1 (12 hex digits) of every file whose code the capture runs through."""
    C = cache_code()
    out = {}
    for name, p in (("rescope/evaluate.py", C.CODE + "/rescope/evaluate.py"), ("rescope/scoped.py", C.CODE + "/rescope/scoped.py"),
                    ("06_testbed_v3/code/tb_common.py", C.EXP + "/06_testbed_v3/code/tb_common.py"),
                    ("vendor_v7/vlm.py", C.CODE + "/vendor_v7/vlm.py"),
                    ("07_repair_v3/harness/hlog.py", C.EXP + "/07_repair_v3/harness/hlog.py"),
                    ("07_repair_v3/harness/evaluation.py", C.EXP + "/07_repair_v3/harness/evaluation.py"),
                    ("cache/code/fc_capture.py", CACHE_CODE + "/fc_capture.py"), ("cache/code/fc_common.py", CACHE_CODE + "/fc_common.py"),
                    ("cache2/mme/code/mme_capture.py", CODE + "/mme_capture.py")):
        out[name] = sha1_file(p) if os.path.exists(p) else None
    return out
