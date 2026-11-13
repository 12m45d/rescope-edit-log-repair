"""E5, linear part: real-log statistics quoted by the theorems on ROLLBACK and COMPENSATION (Sec. 3.1 / 3.2 of the paper:
Thm. thm:rollback-hard, Prop. prop:cond, Thm. thm:capacity).  Shared definitions; CPU only; fp64 linear algebra.

Logs        ONLY the two development logs TB-Q100-ae-l2_3 and TB-L100-ae-l2_3.
Complaints  ALL decoder-v2 D1 complaints of the REPAIR pool, harness conventions: rows of probe_records_v2.jsonl with
            is_complaint_D1 and y0_valid whose probe item is in the REPAIR pool (07_repair_v3/harness/sampler.py:
            candidates(H, "D1", "repair"); the split is by item, seed 20261109).  Split image / text (the probe type).
Keys        the input of the edited module (float32 images of the module-dtype values, cast to fp64), captured at B = 1 on
            the teacher-forced pre-edit answer y0:
              keydef 'last'     the last prompt position (ONE key per complaint; the primary definition),
              keydef 'stacked'  the A answer-predicting positions (last prompt position + y0[:-1]), stacked: the
                                contribution of an edit is then a (d_out x A) matrix, norms are Frobenius norms.
            Source: 07_repair_v3/rollback_opt/cache/keys_<log>_v2.pt (all v2 D1 complaints; row 0 of K is the last prompt
            position); the 'last' key is cross-checked bit for bit against the key cache 10_final/cache/<backbone>/keys_tf.pt.
Log         factor form of the log (rescope.from_testbed): U_i (d_out, fp64), W_i = psi_i (d_in, fp64); contribution of
            edit i at a key k: w_i = u_i <psi_i, k>.  D = sum_i w_i (the factor-form damage; the write round-off E of the
            low-precision in-place writes is NOT part of it; its size is reported as 'fac_vs_applied').
Signs       inner products <w_i, w_j> and the coordinates <psi_i, k> are SIGNED; norms are magnitudes.
Nothing is tuned on these numbers: tolerances, seeds, time limits and rank cuts are fixed below.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import sys

sys.dont_write_bytecode = True          # nothing is written next to the (read-only) sources that are imported by path

import importlib.util
import json
import math
import os
import time
import types

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import torch

ROOT = f"{_RP.ROOT}"
EXP = ROOT + "/02_research/experiments"
LIB = ROOT + "/02_research/code"
TB_ROOT = EXP + "/06_testbed_v3"
HARNESS = EXP + "/07_repair_v3/harness"
ROPT = EXP + "/07_repair_v3/rollback_opt"
RQ3 = EXP + "/07_repair_v3/rq3"
HERE = os.path.dirname(os.path.abspath(__file__))
E5 = os.path.dirname(HERE)
OUT = os.path.join(E5, "out")
LOGDIR = os.path.join(E5, "logs")
for _d in (OUT, LOGDIR):
    os.makedirs(_d, exist_ok=True)
if LIB not in sys.path:
    sys.path.insert(0, LIB)

LOGS = {"TB-Q100-ae-l2_3": {"backbone": "qwen25vl", "short": "Qwen", "layer": 20,
                            "editkeys": "editkeys_qwen25vl_L20_c8054748c923.pt"},
        "TB-L100-ae-l2_3": {"backbone": "llava", "short": "LLaVA", "layer": 16,
                            "editkeys": "editkeys_llava_L16_4c7ff92dd9a5.pt"}}
TOLS = (0.1, 0.3, 0.5)                  # tolerances c of MIN-ROLLBACK (given by the task)
KEYDEFS = ("last", "stacked")
PTYPES = ("image", "text")
NEED_MORE = (10, 20, 50)                # "how many complaints need more than ... deletions"
EPS = float(np.finfo(np.float64).eps)
TOL_REL = 1e-9                          # harness COMP cut: a singular value of (I - P) K_q counts if > TOL_REL * max ||k_q||
OPT_SEED = 0                            # seed of the order in which the optimum code visits the complaints
OPT_TL = 10.0                           # MILP time limit per (complaint, tolerance), seconds (fixed before any run)
OPT_BUDGET_MIN = 40.0                   # budget of the optimum code per log: minutes of CPU TIME (core-minutes), all workers
OPT_WORKERS = 8                         # single-threaded worker processes of the optimum code
OPT_INIT_S = 5.0                        # CPU seconds reserved per worker for loading the log (part of the budget)


def check_log(name: str):
    if name not in LOGS:
        raise SystemExit(f"{name}: E5 uses only the two development logs {sorted(LOGS)}")


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def clean(o):
    """JSON-safe copy (tensors / numpy scalars and arrays / nan, inf -> strings)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, torch.Tensor):
        return clean(o.tolist())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def jdump(obj, path, indent=1):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, indent=indent)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl_read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def dist(values) -> dict:
    """n, min, quartiles (numpy linear interpolation), median, max, mean of a list of numbers (None entries dropped)."""
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}
    return {"n": int(v.size), "min": float(v.min()), "q25": float(np.percentile(v, 25)), "median": float(np.median(v)),
            "q75": float(np.percentile(v, 75)), "max": float(v.max()), "mean": float(v.mean())}


def by_ptype(rows, fn) -> dict:
    """{'all': dist, 'image': dist, 'text': dist} of fn(row) over the rows (each row carries 'ptype')."""
    return {"all": dist([fn(r) for r in rows]),
            **{p: dist([fn(r) for r in rows if r["ptype"] == p]) for p in PTYPES}}


# ----------------------------------------------------------------------------- the log (library) and the complaints (harness)
def load_log(name: str):
    """rescope.EditLog of the testbed directory: U (T, d_out) fp64, W (T, d_in) fp64 read-out vectors, batch, meta['uids']."""
    check_log(name)
    import rescope
    return rescope.from_testbed(name)


def composite_delta(name: str) -> torch.Tensor:
    """W_E.float() - W_0.float() (fp32 on disk), as fp64: the weight difference actually applied (with the write round-off)."""
    return torch.load(os.path.join(TB_ROOT, name, "composite_delta.pt"), map_location="cpu").double()


_SAMPLER = None


def harness_sampler():
    """07_repair_v3/harness/sampler.py loaded by path (it needs numpy only; the harness package itself needs a model)."""
    global _SAMPLER
    if _SAMPLER is None:
        spec = importlib.util.spec_from_file_location("e5_harness_sampler", os.path.join(HARNESS, "sampler.py"))
        _SAMPLER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_SAMPLER)
    return _SAMPLER


def harness_stub(name: str):
    """The fields of a harness handle that the sampler reads: rows (probe_records_v2.jsonl), probes (uid per probe index)."""
    rows = jsonl_read(os.path.join(TB_ROOT, name, "probe_records_v2.jsonl"))
    uid = {}
    for r in rows:
        if uid.setdefault(r["probe_idx"], r["uid"]) != r["uid"]:
            raise RuntimeError("probe_idx with two uids")
    probes = [{"uid": uid[i]} for i in range(len(uid))]
    return types.SimpleNamespace(rows=rows, probes=probes, _setup={}, name=name)


def complaints(name: str, stub=None) -> list:
    """ALL D1 complaints of the REPAIR pool (harness sampler.candidates), ordered image first, then text, by probe index."""
    check_log(name)
    S = harness_sampler()
    stub = harness_stub(name) if stub is None else stub
    out = []
    for pt in PTYPES:
        out += sorted(S.candidates(stub, "D1", "repair", pt), key=lambda r: r["probe_idx"])
    return out


def complaint_keys(name: str, comps: list) -> dict:
    """{group: K (A, d_in) float32} at the answer-predicting positions (row 0 = the last prompt position), from the keys
    that Track C captured for every v2 D1 complaint; checks (uid, A, first row == last prompt position)."""
    d = torch.load(os.path.join(ROPT, "cache", f"keys_{name}_v2.pt"), map_location="cpu")
    if d["log"] != name:
        raise RuntimeError("key file of another log")
    by = {(r["probe_idx"], r["ptype"]): r for r in d["rows"]}
    out = {}
    for c in comps:
        r = by[(c["probe_idx"], c["ptype"])]
        if r["uid"] != c["uid"] or int(r["A"]) != int(c["n_ans_tok"]) or int(r["pred_pos"][0]) != int(r["last_prompt_idx"]):
            raise RuntimeError(f"key row of {c['group']} does not match the probe record")
        if r["K"].shape[0] != int(r["A"]):
            raise RuntimeError("key block with a wrong number of rows")
        out[c["group"]] = r["K"]
    return out


def keycache_check(name: str, comps: list, keys: dict) -> dict:
    """Cross-check against the final key cache (10_final/cache/<backbone>): pool / validity flags of the complaints and the
    teacher-forced last-prompt key, bit for bit."""
    import rescope
    kc = rescope.KeyCache(LOGS[name]["backbone"])
    uids = load_log(name).meta["uids"]
    rows = kc.log_rows(uids, kind="probe")
    idx = {(kc.index[j]["probe_idx"], kc.index[j]["ptype"]): j for j in rows}
    n_eq, n_pool, n_valid, shifted, missing = 0, 0, 0, [], []
    for c in comps:
        j = idx.get((c["probe_idx"], c["ptype"]))
        if j is None or kc.index[j].get("row_tf") is None:
            missing.append(c["group"])
            continue
        e = kc.index[j]
        n_pool += e["pool"] == "repair"
        n_valid += bool(e["y0_valid"]) and e["y0_text"] == c["y0_text"]
        n_eq += bool(torch.equal(kc.K("tf")[int(e["row_tf"])], keys[c["group"]][0]))
        if e.get("lp_shift"):
            shifted.append(c["group"])
    return {"n_complaints": len(comps), "n_in_cache": len(comps) - len(missing), "n_pool_repair_in_cache": int(n_pool),
            "n_valid_and_same_y0_in_cache": int(n_valid), "n_last_key_bitwise_equal": int(n_eq),
            "n_lp_shift": len(shifted), "lp_shift_groups": shifted, "missing": missing,
            "note": "lp_shift = 1: the teacher-forced last prompt position (used here, harness index) is one token after the "
                    "position of a decode (LLaVA, answers that start with a digit)"}
