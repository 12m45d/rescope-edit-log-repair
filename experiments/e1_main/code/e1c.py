"""E1 (e1_main): shared constants and helpers of the experiment's scripts.  Everything numerical comes from the driver fx."""
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
import fx  # noqa: E402
import numpy as np  # noqa: E402

ROOT = FINAL + "/e1_main"
OUT = ROOT + "/out"
GATE = OUT + "/gate"            # gate-level cells: gate/<log>/<label>__a<alpha>.json (all 20 draws)
BEH = OUT + "/behave"           # behavioural cells: behave/<log>/<tag>.json (+ rows under out/rows)
SEL = OUT + "/best_cosine_gate.json"
DEV = tuple(fx.logs.names("development"))
WP = tuple(fx.logs.names("working_point"))
ALPHAS = tuple(fx.draws.ALPHAS)
N_DRAWS = fx.draws.N_DRAWS
HN = ("HN-I", "HN-Q", "HN-X")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L", "TB-Q100-ae": "Qwp10", "TB-Q100-ae-l2_1": "Qwp1", "TB-L100-ae": "Lwp10"}

COS_FAMILY = tuple(fx.review1.COSINE_FAMILY)
RETR_FAMILY = tuple(fx.review1.RETRIEVAL_FAMILY)
# calibrated rows that the key cache decides on the CPU (they depend on alpha and on the calibration draw)
CAL_CPU = ("rescope", "rescope_shared", "rescope_null", "rescope_pac", "rescope_nullW", "rescope_x", "ball") + COS_FAMILY + RETR_FAMILY
# rows without a calibration that the key cache decides (one gate per log)
FIXED_CPU = ("cosine_default", "ball_default", "exact", "lookup")
# rows whose gate needs the model's weights (built inside a GPU job): gate level of the dense gates
DENSE = ("router", "router_default", "zscore", "zscore_default")
# H3 rule for draws on which the near-scope constraint cannot be reached (stated before any number was looked at)
H3_UNREACHABLE_RULE = ("a calibration draw on which no raise of the scale reaches 10 percent mean firing on the fit hard negatives (guard caps keep "
                       "edits in scope) contributes H3 coverage 0 for that draw, for every method alike; the number of such draws is reported next "
                       "to every H3 number")
TIE_RULE = "ties are broken by the order of fx.review1.COSINE_FAMILY / RETRIEVAL_FAMILY (the registered gate first)"


def selection():
    """The selection file (None until it exists): {backbone: {...}}."""
    if not os.path.exists(SEL):
        return None
    return fx.review1.read_selection(SEL)


def selected_labels(L):
    """Selected rows of a log's backbone: best cosine gate(s), their _x form(s), retr rule(s).  [] until the selection exists."""
    s = selection()
    if s is None:
        return {}
    b = s[L.backbone]
    return {"cos_H3": b["cosine"]["H3"], "cos_H3_x": b["cosine"]["H3_x"], "cos_H2": b["cosine"]["H2"], "cos_H2_x": b["cosine"]["H2_x"],
            "retr_H3": b["retr"]["H3"], "retr_H2": b["retr"]["H2"]}


def afmt(alpha):
    return f"{alpha:.2f}"


def flags(a):
    return "".join("1" if bool(x) else "0" for x in a)


def unflags(s):
    return np.array([c == "1" for c in s], dtype=bool)


def jload(p):
    return fx.io.jload(p)


def jdump(o, p):
    fx.io.jdump(o, p)
