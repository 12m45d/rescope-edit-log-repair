"""E3 / E4 on the LLaVA-1.5-7B logs (e3_llava): shared constants and helpers.  Every number comes from the driver fx."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import fx  # noqa: E402
import numpy as np  # noqa: E402

ROOT = FINAL + "/e3_qwen/verify/out/rerun_llava"
OUT = ROOT + "/out"
GATE = OUT + "/gate"            # gate-level cells: gate/<log>/<label>__<readout>.json (all 20 draws); readout reg | qd
BEH = OUT + "/behave"           # behavioural cell records: behave/<log>/<cell>.json (+ rows under out/rows)
ALPHA, N = 0.05, 200
N_DRAWS = fx.draws.N_DRAWS
HN = ("HN-I", "HN-Q", "HN-X")
DEV = "TB-L100-ae-l2_3"
FAMILIES = {"ae-l2_3": ("TB-L100-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3"),          # AlphaEdit-style (T <= 500); T = 100 is the development log
            "memitw": ("TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")}            # MEMIT-style, public configuration
LOGS = ("TB-L100-ae-l2_3", "TB-L200-ae-l2_3", "TB-L500-ae-l2_3", "TB-L100-memitw", "TB-L500-memitw", "TB-L1000-memitw")
NON_ORACLE = (79, 84, 104, 113, 467, 475, 516, 612, 702, 994)      # 09_testbed_scale/REPORT.md; checked per log against per_edit_metrics.jsonl
for _n in LOGS:
    assert fx.logs.programme().get(_n) is not None, _n


def family_of(name):
    return next(f for f, v in FAMILIES.items() if name in v)


def bdraws(T):
    """Behavioural calibration draws (R23: draw 0 only on logs with 1000 or more edits)."""
    return (0,) if T >= 1000 else (0, 1, 2)


def selection(L):
    """Selected rows of the backbone, read with the driver's reader (file e1_main/out/best_cosine_gate.json)."""
    h3, h2 = fx.review1.selected(L, "H3"), fx.review1.selected(L, "H2")
    return {"cos_H3": h3["cosine"], "cos_H3_x": h3["cosine_x"], "cos_H2": h2["cosine"], "retr_H3": h3["retr"], "retr_H2": h2["retr"]}


def labels(L):
    s = selection(L)
    cal = list(dict.fromkeys(["rescope", "rescope_shared", "rescope_null", "rescope_x", "cosine", "cosine_shared", s["cos_H3"], s["cos_H2"], s["cos_H3_x"],
                              s["retr_H3"], s["retr_H2"]]))
    return {"cal": cal,                                              # calibrated rows with behaviour
            "cal_gl": ["rescope_nullW", "rescope_pac"],              # calibrated rows at gate level only (R19)
            "fixed_cpu": ["cosine_default", "exact", "lookup"],      # no calibration; decided on the key cache
            "fixed_gpu": ["deployed", "cosine_published", "where_logonly"],      # every sequence a forward
            "h3": list(dict.fromkeys(["rescope", "rescope_x", s["cos_H3"], s["cos_H3_x"], s["retr_H3"]])),
            "d2": list(dict.fromkeys(["rescope", "rescope_null", "cosine_shared", "rescope_x", s["cos_H3"], s["cos_H2"], s["cos_H3_x"], s["retr_H3"]])),
            "sel": s}


def flags(a):
    return "".join("1" if bool(x) else "0" for x in a)


def unflags(s):
    return np.array([c == "1" for c in s], dtype=bool)


def jload(p):
    return fx.io.jload(p)


def jdump(o, p):
    fx.io.jdump(o, p)
