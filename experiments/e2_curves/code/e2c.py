"""E2 a-c (evidence curves), linear rows, running example -- shared helpers.

Everything method-side goes through the shared driver fx (10_final/common) and the library rescope; this module holds only
paths, the cell files of this experiment and the statistics that the driver does not name (counts against the deployed log).

CELL = one (part, log, method, variant, n, seed / draw) evaluation.  A cell is two files:
  out/cells/<part>/<log>/<tag>.json      what determined it (method record, thresholds, counts) -- small
  out/rows/<part>__<log>__<tag>.jsonl.gz the per-row flags (fx.behave rows; part a: rows rebuilt from the P12 records)
The renderer (render.py) recomputes every reported number from these files.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
import sys
import time

FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
EXP = os.path.join(FIN, "e2_curves")
OUT = os.environ.get("E2_OUT") or os.path.join(EXP, "out")
LOGD = os.path.join(EXP, "logs")
P12 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p12_linear_streams"
sys.path.insert(0, os.path.join(FIN, "common"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import fx  # noqa: E402

LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
NS = (1, 2, 5, 10, 20, 50, 100, "ALL")
N_HN = (10, 100, "ALL")                      # cells that also receive near-scope negatives and D2
LINEAR = ("COMP_B", "COMP_BP", "COMP_B_ALLPOS", "REEDIT_B", "RB_TOT", "RB_UNION")
# restoration criterion of each linear repair, from the options of its P12 record (nothing is recomputed for this)
RESTORES = {"COMP_B": "answer positions only (query='answer'; restores, does not certify)",
            "COMP_BP": "answer positions only (query='answer'; restores, does not certify)",
            "COMP_B_ALLPOS": "ALL positions of the pre-edit traces (query='all'; records exist up to n = 10)",
            "REEDIT_B": "no module-level restoration (new targets z* installed at the last prompt position)",
            "RB_TOT": "no module-level restoration (behavioural criterion: greedy rollback until y0 is reproduced or the budget ends)",
            "RB_UNION": "no module-level restoration (behavioural criterion per complaint, union of the removal sets)"}


def seeds_of(n):
    return (0, 1, 2) if n != "ALL" and int(n) <= 20 else (0,)


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def cell_path(part, logname, tag):
    return os.path.join(OUT, "cells", part, logname, tag + ".json")


def rows_tag(part, logname, tag):
    return f"{part}__{SHORT[logname]}__{tag}"


def rows_path(part, logname, tag):
    return os.path.join(OUT, "rows", rows_tag(part, logname, tag) + ".jsonl.gz")


def cell_done(part, logname, tag):
    return os.path.exists(cell_path(part, logname, tag)) and os.path.exists(rows_path(part, logname, tag))


def write_cell(part, logname, tag, meta, rows):
    """Rows first, then the cell file (its presence marks the cell as finished)."""
    fx.io.save_rows(rows_path(part, logname, tag), rows)
    meta = dict(meta, part=part, log=logname, tag=tag, n_rows=len(rows), written=time.strftime("%Y-%m-%d %H:%M:%S"),
                code=fx.code_fingerprint())
    fx.io.jdump(meta, cell_path(part, logname, tag), indent=1)


def read_cells(part, logname):
    d = os.path.join(OUT, "cells", part, logname)
    out = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                out.append(fx.io.jload(os.path.join(d, f)))
    return out


def read_rows(part, logname, tag):
    return fx.io.load_rows(rows_path(part, logname, tag))


def n_int(L, n, seed=0):
    return len(fx.draws.complaints(L, "ALL", seed)) if n == "ALL" else int(n)


def slim_rec(rec, drop=("sids", "refused_sids", "block1", "block2", "capped_edits", "add_guards")):
    """A method record without its long lists (the lists stay in the driver's own objects; counts are kept)."""
    if isinstance(rec, dict):
        return {k: slim_rec(v, drop) for k, v in rec.items() if k not in drop}
    if isinstance(rec, (list, tuple)):
        return [slim_rec(v, drop) for v in rec][:50]
    return rec
