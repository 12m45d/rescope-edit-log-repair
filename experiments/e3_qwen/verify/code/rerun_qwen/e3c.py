"""E3 / E4 on the Qwen2.5-VL-3B logs (e3_qwen): shared constants and helpers.  Everything numerical comes from the driver fx.

Frozen row set (task text + review round 1 + R19 / R23 / R25); alpha = 0.05, n = 200; nothing is tuned on any log.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import fx  # noqa: E402
import numpy as np  # noqa: E402

ROOT = FINAL + "/e3_qwen/verify/out/rerun_qwen"
OUT = ROOT + "/out"
GATE = OUT + "/gate"            # gate-level cells: gate/<log>/<label>__<reg|qd>.json (all 20 draws)
BEH = OUT + "/behave"           # behavioural cell records: behave/<log>/<cell>.json (+ rows under out/rows)
FAC = OUT + "/factor"           # factor level + stored metrics of the deployed log
ALPHA, N_CAL = 0.05, 200
N_DRAWS = fx.draws.N_DRAWS
HN = ("HN-I", "HN-Q", "HN-X")
DEV = "TB-Q100-ae-l2_3"
# logs of this task, in the order of the task text (R19 b): at most 500 edits first, the longest last
LOGS = ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", "TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw")
FAMILIES = {"ae-l2_3": ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3"),
            "memitw": ("TB-Q100-memitw", "TB-Q500-memitw", "TB-Q1000-memitw", "TB-Q2000-memitw")}
FAMILY_TITLE = {"ae-l2_3": "AlphaEdit-style logs (T <= 500; TB-Q100-ae-l2_3 is the DEVELOPMENT log, TB-Q200 / TB-Q500 are unseen)",
                "memitw": "MEMIT-style logs, public configuration (all unseen)"}


def T_of(log):
    return fx.logs.info(log)["T"]


def bdraws(log):
    """Behavioural calibration draws: 0 - 2 on logs with at most 500 edits, draw 0 on logs with 1000 or more (R23)."""
    return (0,) if T_of(log) >= 1000 else (0, 1, 2)


def selection(L):
    """Selected rows of the log's backbone from e1_main/out/best_cosine_gate.json through the driver's reader (raises if absent)."""
    h3, h2 = fx.review1.selected(L, "H3"), fx.review1.selected(L, "H2")
    return {"cos_H3": h3["cosine"], "cos_H3_x": h3["cosine_x"], "cos_H2": h2["cosine"], "retr": h3["retr"], "retr_H2": h2["retr"]}


def labels(L):
    """The frozen row set of this task.  -> dict of label tuples."""
    s = selection(L)
    cal = tuple(dict.fromkeys(("rescope", "rescope_shared", "rescope_null", "rescope_x", "cosine", "cosine_shared", s["cos_H3"], s["cos_H2"], s["cos_H3_x"],
                               s["retr"], s["retr_H2"])))
    return {"sel": s,
            "cal": cal,                                                 # calibrated rows, memo path: gate level 20 draws + behaviour
            "gl": ("rescope_nullW", "rescope_pac"),                     # calibrated rows, gate level only (R19)
            "fixed": ("exact", "cosine_default", "lookup"),             # rows without calibration that the key cache decides
            "weights": ("deployed",),
            "lib": ("cosine_published", "where_logonly"),               # all-position rows: one forward per sequence, no gate level
            "h3": tuple(dict.fromkeys(("rescope", "rescope_x", s["cos_H3"], s["cos_H3_x"], s["retr"]))),
            "d2": None}


ORDER = ("deployed", "where_logonly", "rescope", "rescope_shared", "rescope_null", "rescope_x", "rescope_nullW", "rescope_pac", "cosine", "cosine_shared",
         "white_1_shared", "white_0.1_shared", "white_1_x", "retr_sum", "lookup", "cosine_default", "cosine_published", "exact")
H3_UNREACHABLE_RULE = ("a calibration draw on which no raise of the scale reaches 10 percent mean firing on the fit hard negatives (guard caps keep edits "
                       "in scope) contributes H3 coverage 0 for that draw, for every method alike (the rule of E1, stated there before any number); the "
                       "number of such draws is reported next to every H3 number")
QD_NOTE = ("second read-out (R8): calibration draw from the REPAIR rows whose question string is no question / text rephrase of any edit of the longest "
           "Qwen stream (driver switch mask='qdisjoint'); far test rows (heldout_disjoint, BEN-T) restricted by the same filter; edit sequences and "
           "hard negatives are in / near scope by construction and are read on all their rows under that calibration")


def flags(a):
    return "".join("1" if bool(x) else "0" for x in a)


def unflags(s):
    return np.array([c == "1" for c in s], dtype=bool)


def jload(p):
    return fx.io.jload(p)


def jdump(o, p):
    fx.io.jdump(o, p)


def order(labs):
    return sorted(labs, key=lambda x: (ORDER.index(x) if x in ORDER else len(ORDER), x))
