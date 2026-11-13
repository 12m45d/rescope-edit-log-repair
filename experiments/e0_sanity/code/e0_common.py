"""E0 sanity comparison (precedes the final experiments): shared definitions.

Logs      ONLY the two development logs TB-Q100-ae-l2_3 and TB-L100-ae-l2_3 (every other log is an unseen test log).
Setting   alpha = 0.05, n = 200 benign calibration queries = a seeded random draw from the valid rows of the REPAIR pool
          (rescope.splits.sample(pool, 200, seed)); draw d uses seed d for the draw and for the two-block split.
Methods   M1  default method of PROTOCOL.md: role 'last', normalised solver coordinate, two-block calibration, top-1,
              response REPL with rebased values, guards on (stored solver keys + B = 1 keys of the edits' own Rel prompts)
          M2  the same with the shared-scale conformal calibration (one conformal threshold on the normalised coordinate)
          M3  cosine gate (raw cosine to the stored solver key, role 'last', top-1, rebased values), two-block calibration
          M4  cosine gate, shared-scale calibration
          M5  cosine gate at its published default threshold 0.65 (same retrofit: role 'last', top-1, rebased values)
          M6  exact-match gate (cosine >= 0.999 to the stored solver key)
          deployed = the deployed log W_E; pre-edit = W_0 (control: D1 must be 0)
Every gate is built with the library (rescope: feature_gate / add_guards / calibrate_two_block / calibrate_shared /
cosine_gate / exact_gate); nothing here changes a method.  The statistics are defined ONCE, in this file (stats_*),
and are computed by e0_render.py from the per-row flags that the experiment scripts write.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

ROOT = f"{_RP.ROOT}"
LIB = ROOT + "/02_research/code"
if LIB not in sys.path:
    sys.path.insert(0, LIB)
HERE = os.path.dirname(os.path.abspath(__file__))
E0 = os.path.dirname(HERE)
OUT = os.path.join(E0, "out")
LOGDIR = os.path.join(E0, "logs")
REFEREE = ROOT + "/03_analysis/theory/round3_rescope/C_statistics/referee"
P11 = ROOT + "/02_research/experiments/08_rescope/p11_scopecut/out"

LOGS = {"TB-Q100-ae-l2_3": {"backbone": "qwen25vl", "short": "Qwen", "gq_mem": 10500},
        "TB-L100-ae-l2_3": {"backbone": "llava", "short": "LLaVA", "gq_mem": 17000}}
ALPHA = 0.05                      # read by the library as the exact fraction 1/20
N_CAL = 200
N_DRAWS = 20                      # gate level: seeded calibration draws 0 .. 19
BEHAVIOUR_DRAWS = (0, 1, 2)       # behaviour (GPU): draws 0, 1, 2
METHODS = {
    "M1": {"feature": "ncoord", "cal": "two_block", "name": "ReScope default (ncoord, two-block)"},
    "M2": {"feature": "ncoord", "cal": "shared", "name": "ReScope, shared scale"},
    "M3": {"feature": "cos", "cal": "two_block", "name": "cosine gate, two-block"},
    "M4": {"feature": "cos", "cal": "shared", "name": "cosine gate, shared scale"},
    "M5": {"feature": "cos", "cal": "fixed", "gamma": 0.65, "name": "cosine gate, default 0.65"},
    "M6": {"feature": "cos", "cal": "fixed", "gamma": 0.999, "name": "exact-match gate (cos >= 0.999)"},
}
CALIBRATED = ("M1", "M2", "M3", "M4")
FIXED = ("M5", "M6")
HN_TYPES = ("HN-I", "HN-Q", "HN-X")


def check_log(name: str):
    if name not in LOGS:
        raise SystemExit(f"{name}: E0 touches only the two development logs {sorted(LOGS)}")


# ----------------------------------------------------------------------------- gates (library calls only)
def base_gate(log, feature: str, W0=None):
    """Un-thresholded feature gate with the protocol's fixed choices: role 'last', top-1, REPL, rebased values."""
    import rescope
    return rescope.feature_gate(log, feature, role="last", fire="top1", response="repl", values="rebased", W0=W0)


def calibrate(S, mid: str, seed: int, queries=None, coords=None, allow_silence: bool = False):
    """The calibration of method `mid` (M1 - M4) on the gate S (guards already added) -> (gate, record)."""
    import rescope
    cal = METHODS[mid]["cal"]
    if cal == "two_block":
        return rescope.calibrate_two_block(S, queries, ALPHA, coords=coords, seed=seed, allow_silence=allow_silence)
    if cal == "shared":
        return rescope.calibrate_shared(S, queries, ALPHA, coords=coords, allow_silence=allow_silence)
    raise ValueError(mid)


def fixed_gate(log, mid: str, W0=None):
    """M5 / M6: the cosine gate with a fixed threshold (library defaults: role 'last', top-1, rebased values)."""
    import rescope
    if mid == "M5":
        return rescope.cosine_gate(log, METHODS[mid]["gamma"], W0=W0)
    if mid == "M6":
        return rescope.exact_gate(log, W0=W0)
    raise ValueError(mid)


def rec_public(rec: dict) -> dict:
    """The calibration record without the long block index lists."""
    return {k: v for k, v in rec.items() if k not in ("block1", "block2")}


# ----------------------------------------------------------------------------- JSON
def _clean(x):
    """JSON-safe copy: non-finite floats become strings ('inf', '-inf', 'nan'); numpy / torch scalars become Python numbers."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if hasattr(x, "item") and not isinstance(x, (str, bytes)) and getattr(x, "ndim", 1) == 0:
        return _clean(x.item())
    return x


def jdump(obj, path: str, indent=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=indent, allow_nan=False)
    os.replace(tmp, path)


def jload(path: str):
    with open(path) as f:
        return json.load(f)


def say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bits(v) -> str:
    """A boolean vector as a string of '0' / '1' (compact per-row flags)."""
    return "".join("1" if bool(x) else "0" for x in v)


def unbits(s: str):
    import numpy as np
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def file_sha1(path: str) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def code_fingerprint() -> dict:
    """Which code produced a result: sha1 of the library's .py files and of this directory's .py files."""
    import hashlib

    def fp(folder):
        h = hashlib.sha1()
        for f in sorted(os.listdir(folder)):
            if f.endswith(".py"):
                h.update(f.encode())
                with open(os.path.join(folder, f), "rb") as fh:
                    h.update(fh.read())
        return h.hexdigest()[:12]
    return {"rescope": fp(os.path.join(LIB, "rescope")), "e0_code": {f: file_sha1(os.path.join(HERE, f)) for f in sorted(os.listdir(HERE)) if f.endswith(".py")}}


# ----------------------------------------------------------------------------- statistics (the ONE definition of every reported number)
def pct(flags) -> float | None:
    """Percentage of true flags (None for an empty list)."""
    flags = list(flags)
    return 100.0 * sum(bool(x) for x in flags) / len(flags) if flags else None


def over_draws(values) -> dict:
    """Summary of one statistic over calibration draws: mean, 5th and 95th percentile (numpy, linear interpolation), min, max."""
    import numpy as np
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n_draws": 0, "mean": None, "p05": None, "p95": None, "min": None, "max": None, "values": []}
    return {"n_draws": int(v.size), "mean": float(v.mean()), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "values": [float(x) for x in v]}


HN_SUBSETS = {"HN-I existing (P11, other category)": lambda r: r["hn_type"] == "HN-I" and not r["new"],
              "HN-I new, other category": lambda r: r["hn_type"] == "HN-I" and r["new"] and not r["same_category"],
              "HN-I new, same category": lambda r: r["hn_type"] == "HN-I" and r["new"] and r["same_category"],
              "HN-I related_flag": lambda r: r["hn_type"] == "HN-I" and r["related_flag"],
              "HN-I not related_flag": lambda r: r["hn_type"] == "HN-I" and not r["related_flag"],
              "HN-Q existing (P11)": lambda r: r["hn_type"] == "HN-Q" and not r["new"],
              "HN-Q new": lambda r: r["hn_type"] == "HN-Q" and r["new"]}


def gate_stats(info, fired, own_cand, own_top, top_edit) -> dict:
    """Gate-level statistics of ONE gate from per-row decisions (arrays aligned with `info`, the static row records).
      probe.<pool>      n, fired = % of valid probe rows of the pool on which some edit is a candidate (= an edit releases)
      edit.<metric>     n, fired, own_cand = % of sequences on which the sequence's own edit is a candidate,
                        own_top = % on which the own edit is the top-1 candidate (the edit that releases; 'coverage'),
                        other_top = % on which another edit releases
      hn.<type>         n, fired, own_cand, own_top (own = the edit whose image (HN-I, HN-X) / question (HN-Q) the row uses);
                        HN-X also b_top (the edit whose QUESTION the row uses releases) and own_or_b_top"""
    import numpy as np
    fired, own_cand, own_top = (np.asarray(x).astype(bool) for x in (fired, own_cand, own_top))
    top_edit = np.asarray(top_edit, dtype=np.int64)
    if not (len(info) == fired.size == own_cand.size == own_top.size == top_edit.size):
        raise ValueError("gate_stats: flags and row records have different lengths")

    def sel(pred):
        return np.array([i for i, r in enumerate(info) if pred(r)], dtype=np.int64)

    out = {"probe": {}, "edit": {}, "hn": {}}
    pools = {"heldout_disjoint": lambda r: bool(r.get("disjoint")), "heldout_strict": lambda r: bool(r.get("strict")),
             "heldout": lambda r: r.get("pool") == "heldout", "repair": lambda r: r.get("pool") == "repair"}
    for nm, pred in pools.items():
        idx = sel(lambda r: r["kind"] == "probe" and pred(r))
        out["probe"][nm] = {"n": int(idx.size), "fired": pct(fired[idx])}
    for pt in ("image", "text"):
        idx = sel(lambda r: r["kind"] == "probe" and bool(r.get("disjoint")) and r.get("ptype") == pt)
        out["probe"]["heldout_disjoint_" + pt] = {"n": int(idx.size), "fired": pct(fired[idx])}
    for m in ("rel", "tgen", "igen"):
        idx = sel(lambda r: r["kind"] == m)
        out["edit"][m] = {"n": int(idx.size), "fired": pct(fired[idx]), "own_cand": pct(own_cand[idx]), "own_top": pct(own_top[idx]),
                          "other_top": pct(fired[idx] & ~own_top[idx])}
    for t in HN_TYPES:
        idx = sel(lambda r: r.get("hn_type") == t)
        d = {"n": int(idx.size), "fired": pct(fired[idx]), "own_cand": pct(own_cand[idx]), "own_top": pct(own_top[idx])}
        if t == "HN-X":
            eb = np.array([info[i]["edit_b"] for i in idx], dtype=np.int64)
            bt = fired[idx] & (top_edit[idx] == eb)
            d.update(b_top=pct(bt), own_or_b_top=pct(own_top[idx] | bt))
        out["hn"][t] = d
    idx = sel(lambda r: r.get("hn_type") in HN_TYPES)
    out["hn"]["all"] = {"n": int(idx.size), "fired": pct(fired[idx])}
    out["hn_subset"] = {}
    for nm, pred in HN_SUBSETS.items():                              # subsets of the hard negatives (HN_SUBSETS)
        idx = sel(lambda r: r.get("hn_type") in HN_TYPES and pred(r))
        out["hn_subset"][nm] = {"n": int(idx.size), "fired": pct(fired[idx]), "own_top": pct(own_top[idx])}
    return out


def leaves(d, prefix=()):
    """Flatten a nested dict of numbers: {(path): value}."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(leaves(v, prefix + (k,)))
        else:
            out[prefix + (k,)] = v
    return out


def nest(flat):
    out = {}
    for path, v in flat.items():
        d = out
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = v
    return out


STATIC_COUNTS = ("n", "n_text", "n_image", "n_y0_degenerate")


def summarise_draws(per_draw: list) -> dict:
    """Per-draw nested statistics -> the same nesting with over_draws() at every leaf (the static counts stay numbers)."""
    flat = [leaves(d) for d in per_draw]
    out = {}
    for path in flat[0]:
        vals = [f.get(path) for f in flat]
        out[path] = vals[0] if path[-1] in STATIC_COUNTS else over_draws(vals)
    return nest(out)


def load_behaviour(name: str):
    """{set: {group: {method: [rows]}}} from the parts of out/behaviour/<log>/ (sets: static, draw0, draw1, draw2; groups: edits,
    probes, hn), or None if the GPU run of the log is not complete.  Every (set, group, method) must hold all its items."""
    import glob
    d = os.path.join(OUT, "behaviour", name)
    if not os.path.exists(os.path.join(d, "done.json")):
        return None
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*__*__*.json"))):
        part = jload(p)
        g = out.setdefault(part["set"], {}).setdefault(part["group"], {})
        for m, rows in part["rows"].items():
            g.setdefault(m, []).extend(rows)
    meta = jload(os.path.join(d, "meta.json"))
    for s, gs in out.items():
        for g, ms in gs.items():
            for m, rows in ms.items():
                if len(rows) != meta["n_items"][g] or len({r["sid"] for r in rows}) != len(rows):
                    raise RuntimeError(f"{name} {s} {g} {m}: {len(rows)} rows, expected {meta['n_items'][g]} distinct items")
    return out


def _pct_or_none(vals):
    """Percentage of true flags; None if the list is empty or a flag is None (methods that are not gates have no firing flags)."""
    vals = list(vals)
    if not vals or any(v is None for v in vals):
        return None
    return 100.0 * sum(bool(v) for v in vals) / len(vals)


def behaviour_stats(edits, probes, hn, degenerate=frozenset()) -> dict:
    """Behavioural statistics of ONE method from its teacher-forced rows (rescope Evaluator rows, B = 1, lp = 'tf').
      edit.<metric>            n, exact = % of the 100 edits whose target is reproduced token by token, token = mean token accuracy (%),
                               fired, own_cand, own_top = % of sequences on which an edit / the own edit is a candidate / the own edit releases
      probe.heldout_disjoint   n, D1 = % of valid rows whose pre-edit answer y0 (multi-token, decoder v2) is no longer reproduced,
                               n_D1, fired, D1_given_fired, T-Loc* / I-Loc* (rescope.locality: token-level agreement with the pre-edit
                               argmax, %, text / image rows), n_nonfinite
      probe.heldout_strict     the same on the rows whose question text does not occur in the REPAIR pool either
      hn.<type>                n, D1, fired, own_top, tok_loc (mean token-level agreement, %), n_nonfinite
    degenerate: sids of the rows whose pre-edit answer y0 is degenerate (empty, one character, '?', a bare None / No); they are
    COUNTED (n_y0_degenerate) and D1 is also given without them (D1_y0_not_degenerate) and on them (D1_y0_degenerate)."""
    from rescope import locality

    def split(sub):
        deg = [r for r in sub if r["sid"] in degenerate]
        rest = [r for r in sub if r["sid"] not in degenerate]
        return {"n_y0_degenerate": len(deg), "D1_y0_not_degenerate": _pct_or_none(r["D1"] for r in rest), "D1_y0_degenerate": _pct_or_none(r["D1"] for r in deg)}

    out = {"edit": {}, "probe": {}, "hn": {}}
    for m in ("rel", "tgen", "igen"):
        sub = [r for r in edits if r["metric"] == m]
        out["edit"][m] = {"n": len(sub), "exact": _pct_or_none(r["exact"] for r in sub), "token": 100.0 * sum(r["token_acc"] for r in sub) / len(sub),
                          "fired": _pct_or_none(r["fired"] for r in sub), "own_cand": _pct_or_none(r.get("own_fired_last") for r in sub),
                          "own_top": _pct_or_none(r.get("own_released_last") for r in sub), "n_nonfinite": sum(not r["finite"] for r in sub)}
    ex = {m: {r["edit"]: bool(r["exact"]) for r in edits if r["metric"] == m} for m in ("rel", "tgen", "igen")}
    out["edit"]["gen_mean"] = {"n": len(ex["tgen"]), "exact": (out["edit"]["tgen"]["exact"] + out["edit"]["igen"]["exact"]) / 2.0}      # (T-Gen + I-Gen) / 2
    for nm, sub in (("heldout_disjoint", list(probes)), ("heldout_strict", [r for r in probes if r["strict"]])):
        if not all(r["disjoint"] and r["y0_valid"] for r in sub):
            raise ValueError("behaviour_stats: probe rows must be valid HELD-OUT-DISJOINT rows")
        loc = locality(sub)
        fired = [r for r in sub if r["fired"]]
        out["probe"][nm] = {"n": len(sub), "D1": _pct_or_none(r["D1"] for r in sub), "n_D1": sum(bool(r["D1"]) for r in sub),
                            "fired": _pct_or_none(r["fired"] for r in sub), "n_fired": None if sub[0]["fired"] is None else len(fired),
                            "D1_given_fired": _pct_or_none(r["D1"] for r in fired) if sub[0]["fired"] is not None else None,
                            "T-Loc*": loc["T-Loc*"], "I-Loc*": loc["I-Loc*"], "n_text": loc["n_text"], "n_image": loc["n_image"],
                            "n_nonfinite": sum(not r["finite"] for r in sub), **split(sub)}
    for t in HN_TYPES + ("all",):
        sub = [r for r in hn if t == "all" or r["hn_type"] == t]
        out["hn"][t] = {"n": len(sub), "D1": _pct_or_none(r["D1"] for r in sub), "fired": _pct_or_none(r["fired"] for r in sub),
                        "own_top": _pct_or_none(r.get("own_released_last") for r in sub),
                        "tok_loc": 100.0 * sum(r["tok_loc"] for r in sub) / len(sub), "n_nonfinite": sum(not r["finite"] for r in sub), **split(sub)}
    out["hn_subset"] = {}
    for nm, pred in HN_SUBSETS.items():
        sub = [r for r in hn if pred(r)]
        out["hn_subset"][nm] = {"n": len(sub), "D1": _pct_or_none(r["D1"] for r in sub), "fired": _pct_or_none(r["fired"] for r in sub)}
    return out
