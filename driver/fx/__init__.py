"""fx: the shared driver of the final experiments E1 - E8 (10_final/PROTOCOL.md).

    import sys; sys.path.insert(0, "$RESCOPE_ROOT/02_research/experiments/10_final/common")
    import fx

    fx.logs      registry of the paper's logs and their roles; open(name) -> LogData (edit log, key cache, row mapping, pools)
    fx.draws     seeded calibration draws (REPAIR pool / benign pool / contaminated), reported-complaint sets paired with P12
    fx.methods   one constructor per method name -> Method (library object + record); e1_rowset
    fx.gate      gate-level evaluation from the key cache (CPU): per-row firing on probes, edits, hard negatives, benign pool
    fx.behave    behavioural evaluation (GPU) with the persistent memo; everything else through the library's Evaluator
    fx.memo      the on-disk memo (per-process append-only shards, merged on read)
    fx.metrics   the ONE definition of every reported number (from per-row flags)
    fx.stats     percentile bootstrap, paired bootstrap, exact McNemar, summaries over calibration draws
    fx.io        the result schema (flat records), per-row files, paper_numbers.json, the script-rendered table
  stage-1 extension (2026-11-11; what was registered after the first driver was specified):
    fx.pools     key-level pools BEN-W / BEN-S / BEN-T and MME-C / MME-T (false firing only) and the guard of the test pools
    fx.masks     later edits; question strings of the longest stream (R8)
    fx.sweeps    the raised scale of a calibrated gate: curves, coverage scale, read-out H3, the raised gate as a method
    fx.prefix    prefix logs of the development streams (R12) with a loader for the shared harness
  stage 2 (2026-11-11; the gates registered after the review, wired from the feature packages 10_final/feat2):
    fx.review1   cosine family of R3, rescope_nullW, retrieval family and lookup (R4), rescope_x / <gate>_x (R5), the row set of E1
                 after the review, the reader of the selection file, the near-scope read-outs of the _x methods

Everything method-side is the library 02_research/code/rescope; nothing here changes a method.  See ../USAGE.txt.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):       # at most 4 BLAS threads per process
    os.environ.setdefault(_v, "4")

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"
EXP = ROOT + "/02_research/experiments"
FINAL = EXP + "/10_final"
COMMON = FINAL + "/common"
MEMO_ROOT = os.environ.get("FX_MEMO_ROOT", FINAL + "/memo")
TESTBED = EXP + "/06_testbed_v3"
P12 = EXP + "/08_rescope/p12_linear_streams"
E0 = FINAL + "/e0_sanity"
GQ = ROOT + "/02_research/tools/gpuq/gq"
PYTHON = f"{_RP.PYTHON}"
if LIB not in sys.path:
    sys.path.insert(0, LIB)

import torch  # noqa: E402

torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

VERSION = "fx-1.2 stage 2 (2026-11-11)"


def code_fingerprint() -> dict:
    """sha1 (12 hex digits) of the library's .py files and of this package's .py files: which code produced a result."""
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


from . import logs, memo, stats, io, draws, gate, methods, metrics, behave  # noqa: E402,F401
from . import pools, masks, sweeps, prefix  # noqa: E402,F401   (stage-1 extension, 2026-11-11: USAGE.txt, section EXTENSION)
from . import review1  # noqa: E402,F401   (stage 2, 2026-11-11: USAGE.txt, section GATES OF THE REVIEW ROUND)
