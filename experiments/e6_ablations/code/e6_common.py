"""E6 ablations: the registry of rows (one switch off the default method each) and shared helpers.

Every row is a call of the shared driver fx (10_final/common): fx.methods.rescope(L, alpha=0.05, n=200, draw=d, <one switch>),
fx.methods.rescope_null(L, n_A=...), fx.methods.rescope_pac(L).  Nothing here defines a method.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(HERE)
OUT = os.path.join(E6, "out")
COMMON = f"{_RP.ROOT}/02_research/experiments/10_final/common"
sys.path.insert(0, COMMON)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import fx  # noqa: E402

ALPHA, N, N_DRAWS = 0.05, 200, 20
BEHAVE_DRAWS = (0, 1, 2)
LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}

# key, group (task item), text, constructor ("rescope" | "rescope_null" | "rescope_pac"), switches, per_draw, behaviour, multi-position role
ABL = [
    ("default",        "0 default",      "default method (role last, REPL, rebased, top-1, ncoord, two-block 1:1, guards)", "rescope", {}, True, True, False),
    ("role_lastans",   "1 role",         "role = last + answer positions",                    "rescope", {"role": "last+answer"}, True, False, True),
    ("role_all",       "1 role",         "role = all positions",                              "rescope", {"role": "all"}, True, True, True),
    ("resp_gain",      "2 response",     "response = GAIN",                                   "rescope", {"response": "gain"}, True, False, False),
    ("resp_lin",       "2 response",     "response = LIN (original log terms)",               "rescope", {"response": "lin"}, True, True, False),
    ("values_logged",  "3 values",       "values = logged write vectors (no rebase)",          "rescope", {"rebase": False}, True, True, False),
    ("fire_all",       "4 selection",    "selection = all firing edits",                      "rescope", {"fire": "all"}, True, False, False),
    ("feat_coord",     "5 feature",      "feature = raw solver coordinate",                   "rescope", {"feature": "coord"}, True, False, False),
    ("feat_cos",       "5 feature",      "feature = raw cosine to the stored solver key",     "rescope", {"feature": "cos"}, True, False, False),
    ("cal_shared",     "6 calibration",  "calibration = shared scale",                        "rescope", {"calibration": "shared"}, True, True, False),
    ("cal_logonly",    "6 calibration",  "calibration = log-only (largest pooled cross-talk; no data)", "rescope", {"calibration": "log_only"}, False, True, False),
    ("cal_runmax",     "6 calibration",  "calibration = per-edit running maxima of the n queries", "rescope", {"calibration": "running_max"}, True, True, False),
    ("split_1_3",      "7 two-block",    "block ratio 1:3",                                   "rescope", {"split": (1, 3)}, True, False, False),
    ("split_3_1",      "7 two-block",    "block ratio 3:1",                                   "rescope", {"split": (3, 1)}, True, False, False),
    ("rank_r",         "7 two-block",    "top-1 by raw coordinate (not by excess)",           "rescope", {"rank": "r"}, True, False, False),
    ("guards_off",     "8 guards",       "guards off (literal rule, no cap)",                 "rescope", {"allow_silence": True}, True, True, False),
    ("src_bens",       "9 source",       "calibration source = benign pool BEN-S (teacher-forced keys)", "rescope", {"source": "BEN-S"}, True, False, False),
    ("src_bens_dec",   "9 source",       "calibration source = benign pool BEN-S (prompt-only keys)", "rescope", {"source": "BEN-S", "which": "decode"}, True, False, False),
    ("contam_01",      "10 contamination", "contamination 1 % (rephrases of even edits)",     "rescope", {"contamination": 0.01, "contam_half": "even"}, True, False, False),
    ("contam_05",      "10 contamination", "contamination 5 %",                               "rescope", {"contamination": 0.05, "contam_half": "even"}, True, False, False),
    ("contam_20",      "10 contamination", "contamination 20 %",                              "rescope", {"contamination": 0.2, "contam_half": "even"}, True, True, False),
    ("n_40",           "11 sample size", "n = 40",                                            "rescope", {"n": 40}, True, False, False),
    ("n_100",          "11 sample size", "n = 100",                                           "rescope", {"n": 100}, True, False, False),
    ("n_400",          "11 sample size", "n = 400",                                           "rescope", {"n": 400}, True, False, False),
    ("null_50",        "12 rescope_null", "rescope_null, n_A = 50",                           "rescope_null", {"n_A": 50}, True, False, False),
    ("null_100",       "12 rescope_null", "rescope_null, n_A = 100 (registered variant)",     "rescope_null", {"n_A": 100}, True, False, False),
    ("null_200",       "12 rescope_null", "rescope_null, n_A = 200",                          "rescope_null", {"n_A": 200}, True, False, False),
    ("pac",            "13 rescope_pac", "rescope_pac (calibration-conditional rank, delta = 0.1)", "rescope_pac", {}, True, False, False),
]
KEYS = [a[0] for a in ABL]
BY = {a[0]: dict(key=a[0], group=a[1], text=a[2], ctor=a[3], sw=a[4], per_draw=a[5], behave=a[6], multi=a[7]) for a in ABL}
BEHAVE = [a[0] for a in ABL if a[6]]
MULTI = [a[0] for a in ABL if a[7]]
# rows whose switch only changes what is RELEASED (not the gate decision): at gate level they are the default method
GATE_EQUALS_DEFAULT = ("resp_gain", "resp_lin", "values_logged")


def build(L, key: str, draw: int = 0, H=None, literal: bool = False):
    """The Method of an ablation row for a calibration draw (literal=True: the same row with the literal rule, no guard cap)."""
    a = BY[key]
    sw = dict(a["sw"])
    sw.setdefault("n", N)
    if literal:
        sw["allow_silence"] = True
    kw = dict(alpha=ALPHA, draw=draw, label=key + ("@literal" if literal else ""), **sw)
    if a["multi"]:
        kw["H"] = H
    if a["ctor"] == "rescope":
        return fx.methods.rescope(L, **kw)
    if a["ctor"] == "rescope_null":
        return fx.methods.rescope_null(L, **kw)
    if a["ctor"] == "rescope_pac":
        return fx.methods.rescope_pac(L, **kw)
    raise KeyError(a["ctor"])


def draws_of(key: str, n_draws: int = N_DRAWS):
    return list(range(n_draws)) if BY[key]["per_draw"] else [0]


def cal_summary(M) -> dict:
    c = M.rec.get("cal") or {}
    op = M.rec.get("own_prompts") or {}
    eta = c.get("eta")
    ncap = fx.methods.n_capped(M)
    if ncap is None and c.get("n_detectors_capped") is not None:        # rules whose record counts detectors (log-only): one detector per edit here
        ncap = int(c["n_detectors_capped"]) if M.obj.D == M.obj.T else None
    return {"n_capped": ncap, "n_capped_driver": fx.methods.n_capped(M), "own_not_released": op.get("n_not_released"), "own_not_candidate": op.get("n_not_candidate"),
            "eta": None if eta is None else float(eta), "bound": c.get("bound_false_firing"),
            "n_own_keys_silenced_by_literal_rule": c.get("n_own_keys_silenced_by_literal_rule"),
            "n_edits_silenced_by_literal_rule": c.get("n_edits_silenced_by_literal_rule"),
            "variant": M.variant, "digest": M.digest, "n": M.n, "draw": M.draw}
