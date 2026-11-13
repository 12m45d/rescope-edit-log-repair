"""fx.review1: stage 2 of the driver -- the gates registered after the review, wired from the feature packages 10_final/feat2.

PROTOCOL.md section 6: "Review round 1" (R3 gates of the cosine family, rescope_nullW; R4 retrieval reference and lookup; R5
rescope_x / <gate>_x), "R4 extended" (the retrieval family retr_conj / retr_sum / retr_min), the clarifications of 06:34 (two
pairs of H3) and of 09:03 ((1) null-repair sample by QUERY, (2) near-scope read-outs of every _x method also on the test hard
negatives whose foreign part occurs in no recombination row), and R19 (what is NOT built: USAGE.txt).

    M = fx.methods.build(name, L, alpha=0.05, n=200, draw=0)          every name below (fx.review1.NAMES), any log of the two VLKEB streams
    rows = fx.review1.e1_rowset_review1(L, 0.05, draw, H=H)            the stage-1 row set of E1 + the new rows (R1_LABELS)
    sel = fx.review1.read_selection()                                 e1_main/out/best_cosine_gate.json -> per backbone the best gate of the
                                                                      cosine family, its _x form, the selected retrieval rule (H3; H2)

NAMES (role last, top-1, the driver's rebased values at the last prompt position, guards = stored solver keys + B = 1 keys of the own
Rel prompts, the same calibration draw and blocks as every other gate of the driver: fairness rule (i) of PROTOCOL.md section 2)
  cos_centred            cos(k - mu, k_i - mu), mu = mean BEN-W key                              (white.scores.feature_gate)
  cos_null               the cosine detector after the library's null repair (ridge 0) on the 100 null-repair keys that rescope_null
                         uses in the SAME draw (fx.draws.null_sample: no query of the calibration draw); depends on the draw
  white_<kappa>          k_i' M k / k_i' M k_i, M = (C_W + rho I)^-1, rho = kappa tr(C_W) / d_in      kappa in 0.01 | 0.1 | 1
  whitecos_<kappa>       the cosine in that whitened space                                         (also: build("white", L, kappa=0.1))
    each with the two-block calibration (the name as it stands) and with the shared-scale calibration (<name>_shared)
  rescope_nullW          rescope_null with ALL BEN-W keys (prompt-only keys of the 600 rows) as null-repair keys; then the default calibration
  retr_conj              R4 as written: image cosine AND question cosine, each calibrated by the two-block rule on the same draw at the
                         same level; top-1 by the sum of the two margins (feat2/retr: calibrate / decide)
  retr_sum               ONE per-edit score a + c (image cosine + question cosine); two-block rule
  retr_min               ONE per-edit score min(a', c'): a and c standardised PER EDIT by the mean and the (population) standard deviation
                         of that edit's block-1 scores (image attribute: over the block-1 queries that have an image); then the block-2
                         quantile as usual
    all three: rows without an image release nothing; guards keep the edit's OWN REQUEST in scope (cap at (1 - 1e-9) x its own score)
  lookup                 same image content (md5) and at least one shared content word; no calibration (feat2/retr: lookup_decide)
  <gate>_x               R5: the two-block rule with b_i = max(block-1 baseline, largest score of edit i on ITS OWN recombination rows)
                         (feat2/recomb: own_baseline; stage-1 hook baseline_x): rescope_x, rescope_null_x, rescope_nullW_x, cosine_x,
                         ball_x, cos_centred_x, cos_null_x, white_<kappa>_x, whitecos_<kappa>_x
  cosine_shared          the registered cosine gate with the shared scale (= fx.methods.cosine(L, "shared")), so that every member of
                         the cosine family (COSINE_FAMILY) is a name

NOT DEFINED (NotDefined, a LookupError that names the reason; fx.review1.why_not(L, name) returns the reason without building)
  cos_centred / white / whitecos / rescope_nullW   on logs outside the two VLKEB streams (families Q, L): the BEN-W statistics are
                         registered for them only (R2: not used with the one-hop logs; 215 benign rows are probe queries of the
                         MMKE-Bench family; no statistics for blip2)
  retr_* / lookup        on logs without an attribute table / embeddings (one-hop, BLIP2 caches); on rows of ANOTHER key cache (the MME
                         pools: no retrieval attribute was built for MME rows)
  <gate>_x               on logs whose edits are not the first T edits of a backbone's long stream (no recombination rows); for the
                         shared-scale calibration (R5 defines the construction in the two-block rule); for the retrieval family (about
                         110 of the 1200 recombination images of a T = 100 log have no embedding)
  Prefix logs use what exists for their edits (an edit without a usable recombination row keeps its block-1 baseline).

A RETRIEVAL METHOD is not a function of the module input: Method.kind == 'request', Method.obj = RequestRule.  The driver serves it
through hooks of fx.gate.decide (hence fx.gate.evaluate, fx.pools on BEN-*, fx.metrics.gate_stats), fx.behave (ses.tf / edits /
probes / hard_negatives / generate / d2: a sequence on which nothing is released IS the cached pre-edit row; a sequence on which
edit e is released runs through the memo path under the library gate that releases exactly e, feat2/retr forced_gate on the
driver's base gate: the same value, the same memo entry as any scoped gate releasing e there) and fx.sweeps (RequestRule.sweep:
the common scale of every attribute is shifted, guards kept).  ses.tf_hashed and fx.methods.mask_edits are not defined for it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys

import numpy as np
import torch

import fx
from . import draws as D, gate as G, methods as MT

FEAT2 = fx.FINAL + "/feat2"
SELECTION = fx.FINAL + "/e1_main/out/best_cosine_gate.json"
KAPPAS = (0.01, 0.1, 1.0)
STREAMS = ("Q", "L")                       # the two VLKEB streams (key-cache families)
TIE = 1e-9                                 # the library's own-key slack (rescope.scoped.TIE)
N_STEPS = G.N_STEPS
NEG = -np.inf
HN = ("HN-I", "HN-Q", "HN-X")
RETRIEVAL_FAMILY = ("retr_conj", "retr_sum", "retr_min")
_KT = {0.01: "0.01", 0.1: "0.1", 1.0: "1"}
_COS_NEW = ("cos_centred", "cos_null") + tuple(f"{f}_{_KT[k]}" for f in ("white", "whitecos") for k in KAPPAS)
COSINE_FAMILY = ("cosine", "cosine_shared") + tuple(x for b in _COS_NEW for x in (b, b + "_shared"))       # the family "the best gate" is selected from
X_BASES = ("rescope", "rescope_null", "rescope_nullW", "cosine", "ball") + _COS_NEW                         # two-block gates with a per-edit score on the key
NAMES = tuple(x for b in _COS_NEW for x in (b, b + "_shared")) + ("cosine_shared", "rescope_nullW") + RETRIEVAL_FAMILY + ("lookup",) + tuple(b + "_x" for b in X_BASES)
# rows added to the row set of E1 (e1_rowset_review1): R3 (both calibrations), R5, R4 with its extension
R1_LABELS = tuple(x for b in _COS_NEW for x in (b, b + "_shared")) + ("rescope_x",) + RETRIEVAL_FAMILY + ("lookup",)
R1_GATE_LEVEL = ("rescope_nullW",)         # gate level only (R19 a), next to fx.methods.E1_GATE_LEVEL = ("rescope_pac",)
R1_DEPENDS = {k: ("alpha", "draw") for k in R1_LABELS + R1_GATE_LEVEL if k != "lookup"}
_PK, _BASES, _BASES_MAX, _XB, _XB_MAX, _FORCED, _NULLW = {}, {}, 10, {}, 16, {}, {}


class NotDefined(LookupError):
    """A gate of the review round is not defined on a log (or on rows of another key cache); the message names the reason."""


# ----------------------------------------------------------------------------- the feature packages (read-only; imported on first use)
def package(name: str):
    """'white' -> (white.scores, white.wstats); 'recomb' -> the package; 'retr' -> the module feat2/retr/retr.py.
    Bytecode writing is switched off for the process BEFORE the import (the packages' own rule, applied from their first module
    on, so that the driver writes nothing into their directories); retr.py puts its directory FIRST on sys.path (it holds
    modules called tests, mechanics, embed): it is moved to the end here, after the import."""
    if name not in _PK:
        sys.dont_write_bytecode = True                                  # the packages are read-only: no __pycache__ may appear in their directories
        if FEAT2 not in sys.path:                                       # (white and retr set the same flag themselves, but only after their first modules)
            sys.path.append(FEAT2)
        if name == "white":
            from white import scores, wstats
            _PK[name] = (scores, wstats)
        elif name == "recomb":
            import recomb
            _PK[name] = recomb
        elif name == "retr":
            d = FEAT2 + "/retr"
            if d not in sys.path:
                sys.path.append(d)
            import retr
            while d in sys.path:
                sys.path.remove(d)
            sys.path.append(d)
            if retr.TIE != TIE or retr.N_STEPS != N_STEPS:
                raise RuntimeError("feat2/retr and fx.review1 disagree on the own-request slack or the number of decode steps")
            _PK[name] = retr
        else:
            raise KeyError(name)
    return _PK[name]


# ----------------------------------------------------------------------------- names
def parse(name: str, kappa=None, calibration=None):
    """name -> {'base', 'feature', 'kappa', 'calibration', 'x'} or None (not a name of this module).  build("white", L, kappa=0.1,
    calibration="shared") is build("white_0.1_shared", L)."""
    x = name.endswith("_x")
    core = name[:-2] if x else name
    sh = core.endswith("_shared")
    core = core[:-7] if sh else core
    feat, kap = core, None
    for f in ("whitecos", "white"):
        if core == f or core.startswith(f + "_"):
            feat = f
            try:
                kap = float(core[len(f) + 1:]) if core != f else (None if kappa is None else float(kappa))
            except ValueError:
                return None
            if kap is None:
                return {"error": f"{name!r} needs kappa (registered: 0.01, 0.1, 1)"}
            if kap not in KAPPAS:
                return {"error": f"kappa = {kap:g} is not registered (R3: 0.01, 0.1, 1)"}
            core = f"{f}_{_KT[kap]}"
            break
    cal = "shared" if sh else "two_block"
    if calibration is not None:
        if sh and calibration != "shared":
            return {"error": f"{name!r} names the shared scale; calibration={calibration!r} contradicts it"}
        cal = calibration
    if cal not in ("two_block", "shared"):
        return {"error": f"calibration {cal!r}: 'two_block' | 'shared'"}
    known = _COS_NEW + ("rescope_nullW",) + RETRIEVAL_FAMILY + ("lookup",) + (("rescope", "rescope_null", "cosine", "ball") if x else ()) + (("cosine",) if sh else ())
    if core not in known:
        return None
    return {"base": core, "feature": feat, "kappa": kap, "calibration": cal, "x": x, "name": core + ("_shared" if cal == "shared" else "") + ("_x" if x else "")}


def handles(name: str) -> bool:
    """True for the names fx.methods.build hands to this module (everything parse() understands that is not a stage-1 builder)."""
    return name not in MT.BUILDERS and parse(name) is not None


def _stream(L, name):
    if L.family not in STREAMS:
        raise NotDefined(f"{name} is not defined on {L.name}: the BEN-W statistics (cache2/whiten, cache2/benign_parts) are registered for the logs of the two "
                         f"VLKEB streams only (key-cache families Q and L; this log: family {L.family!r}, backbone {L.backbone})")


def _edits(L, name="retr"):
    R = package("retr")
    try:
        return R.edits(L)
    except (LookupError, FileNotFoundError, KeyError) as e:
        raise NotDefined(f"{name} is not defined on {L.name}: no retrieval attributes (image / question embeddings of the requests) for its edits "
                         f"({type(e).__name__}: {e})") from e


def _recomb_rows(L, name="_x"):
    try:
        return package("recomb").log_rows(L)
    except (LookupError, FileNotFoundError, KeyError) as e:
        raise NotDefined(f"{name} is not defined on {L.name}: no recombination rows for this log ({type(e).__name__}: {e})") from e


def why_not(L, name: str, **kw):
    """None if the gate `name` is defined on log L, else the reason (the message of the NotDefined that build would raise).
    Checks the data a gate needs, without building it."""
    L = MT._log(L)
    p = parse(name, kw.get("kappa"), kw.get("calibration"))
    if p is None:
        return None if name in MT.BUILDERS else f"{name!r} is not a method name"
    if "error" in p:
        return p["error"]
    try:
        if p["x"]:
            if p["base"] in RETRIEVAL_FAMILY + ("lookup",):
                raise NotDefined(f"{name} is not defined: the recombination queries have no retrieval attributes (about 110 of the 1200 recombination images "
                                 "of a T = 100 log have no embedding), and PROTOCOL.md compares retr in the first pair's form only")
            if p["calibration"] == "shared":
                raise NotDefined(f"{name} is not defined: R5 raises the BLOCK-1 baseline of the two-block rule; the shared-scale rule has none "
                                 f"(the _x form of a shared-scale gate is {p['base']}_x: read_selection)")
            _recomb_rows(L, name)
        if p["feature"] in ("cos_centred", "white", "whitecos") or p["base"] == "rescope_nullW":
            _stream(L, name)
        if p["base"] in RETRIEVAL_FAMILY + ("lookup",):
            _edits(L, name)
    except NotDefined as e:
        return str(e)
    return None


# ----------------------------------------------------------------------------- un-thresholded gates of the cosine family, rescope_nullW
def _put(cache, key, val, cap):
    while len(cache) >= cap:
        cache.pop(next(iter(cache)))
    cache[key] = val
    return val


def _values_note(S, rebase):
    return S.replace(meta=dict(S.meta, values="rebased" if rebase else "stored"))


def _guard_arg(guards):
    if guards not in (True, False, "all"):
        raise ValueError("guards: True | False | 'all'")
    return guards


def cos_base(L, feature: str, kappa=None, null_rows=None, guards=True, rebase: bool = True):
    """(S, record): the UN-THRESHOLDED gate of a cosine-family feature of R3 with its guards (stored solver keys + the B = 1 keys of
    the edits' own Rel prompts, added before any calibration, as fx.methods.base_gate does for the registered cosine gate), role
    'last', top-1, REPL, the driver's batch-by-batch rebased values.  feature: 'cos_centred' | 'cos_null' | 'white' | 'whitecos'
    (white.scores.feature_gate); null_rows: the null-repair sequences of cos_null (teacher-forced last-prompt keys).
    One object per (log, feature, kappa, null sample): the detectors are shared by every calibration."""
    from rescope import add_guards
    WS, wstats = package("white")
    L = MT._log(L)
    lab = WS.label(feature, kappa)
    key = (L.name, lab, None if null_rows is None else tuple(int(j) for j in null_rows), _guard_arg(guards), bool(rebase))
    if key in _BASES:
        return _BASES[key]
    if feature == "cos_null":                                           # one cos_null gate per log is kept (it changes with the draw)
        for k in [k for k in _BASES if k[0] == L.name and k[1] == "cos_null"]:
            _BASES.pop(k)
    V, vrec = MT.released_values(L, rebase)
    if feature == "cos_null":
        if null_rows is None:
            raise ValueError("cos_null needs the null-repair sequences (null_rows)")
        from . import pools as P
        A = [int(j) for j in null_rows]
        P.guard(L, A, "statistics")
        S = WS.feature_gate(L.log, "cos_null", null_keys=L.kc.keys("tf", A), values=V)
    else:
        _stream(L, lab)
        S = WS.feature_gate(L.log, feature, stats=wstats.load(L.backbone), kappa=kappa, values=V)
    S = _values_note(S, rebase)
    grec = None
    if guards:
        S, grec = add_guards(S, *L.own_keys(True if guards is True else "all"))
    rec = {"add_guards": dict(grec or {}, values=vrec), "white": S.meta.get("white"), "gate_class": type(S).__name__}
    return _put(_BASES, key, (S, rec), _BASES_MAX)


def nullw_base(L, guards=True, rebase: bool = True):
    """(S, record): the default method's un-thresholded gate after the library's null repair (ridge 0) on ALL BEN-W keys
    (white.scores.null_w on fx.methods.base_gate(L): the detectors do not depend on the draw)."""
    WS, wstats = package("white")
    L = MT._log(L)
    _stream(L, "rescope_nullW")
    key = (L.name, _guard_arg(guards), bool(rebase))
    if key not in _NULLW:
        S0, grec = MT.base_gate(L, "ncoord", guards=guards, rebase=rebase)
        S1, nrec = WS.null_w(S0, stats=wstats.load(L.backbone))
        _put(_NULLW, key, (S1, {"add_guards": grec, "null_repair": nrec}), 4)
    return _NULLW[key]


def _null_base(L, A, guards=True, rebase: bool = True):
    """The detectors of rescope_null for the null-repair sequences A (the library's null_repair on the base gate, as fx.methods.scoped)."""
    from rescope import null_repair
    from . import pools as P
    P.guard(L, A, "statistics")
    S0, grec = MT.base_gate(L, "ncoord", guards=guards, rebase=rebase)
    S1, r0 = null_repair(S0, L.kc.keys("tf", A), ridge=0.0)
    infl = (S1.det_w.norm(dim=1) / S0.det_w.norm(dim=1)).numpy()
    return S1, {"add_guards": grec, "null_repair": dict(r0, n_a=len(A), sids=[L.sid(j) for j in A], n_distinct_queries=len(set(D.query_ids(L, A))),
                                                       keys="teacher-forced key at the last prompt position of each benign query",
                                                       detector_norm_inflation={"median": float(np.median(infl)), "max": float(infl.max())})}


def x_baseline(L, S, keep: bool = True):
    """(x (D,) fp64, record) = recomb.own_baseline(L, S): per detector the largest score of its edit on ITS OWN recombination rows
    (-inf: the edit has no usable row); once per (log, detectors of S) (keep=False: not remembered -- detectors that change with
    the draw).  NotDefined if the log has no recombination rows."""
    RC = package("recomb")
    L = MT._log(L)
    key = (L.name, id(S.det_w), id(S.det_s), id(S.det_kind), id(S.det_key))
    if key in _XB:
        return _XB[key][1], _XB[key][2]
    _recomb_rows(L)
    x, rec = RC.own_baseline(L, S)
    if keep:
        _put(_XB, key, ((S.det_w, S.det_s, S.det_kind, S.det_key), x, rec), _XB_MAX)
    return x, rec


def _variant(now: dict, dflt: dict) -> str:
    return ",".join(f"{k}={':'.join(str(x) for x in v) if isinstance(v, tuple) else v}" for k, v in now.items() if v != dflt[k])


def _draw(L, n, draw, source, which, mask, cal, part):
    return cal if cal is not None else D.calibration(L, n, draw, D.resolve_source(source, part), which, mask=mask)


def _no_shared_query(L, A, dr, what):
    if getattr(dr, "kc", None) is None:
        shared = set(D.query_ids(L, A)) & set(D.query_ids(L, dr.rows))
        if shared:
            raise ValueError(f"the sample of the null repair of {what} shares {len(shared)} queries (same question and image content, hence the same key) "
                             "with the calibration sample: the detectors would depend on calibration queries")


def _scoped_r1(L, p, alpha, n, draw, label, guards=True, rebase=True, allow_silence=False, source="repair", which="tf", mask=None, cal=None, split=(1, 1),
               null_rows=None, baseline_x=None, part=None):
    """The gates of this module that are scoped logs: the cosine family of R3, rescope_nullW, and the _x forms that do not go through
    a stage-1 constructor.  Calibration = fx.methods.recalibrate (the library's calibrate_two_block / calibrate_shared on the draw of
    the driver; _x: the stage-1 hook baseline_x, x as one more block-1 row)."""
    base, cal_rule, x = p["base"], p["calibration"], p["x"]
    dr = _draw(L, n, draw, source, which, mask, cal, part)
    A = None
    if base in ("cos_null", "rescope_null"):
        if null_rows is not None:
            A = [int(j) for j in null_rows]
        elif cal is not None:
            raise ValueError("with an explicit calibration sample (cal=) give the null-repair sample explicitly (null_rows=)")
        else:
            A = D.null_sample(L, MT.NULL_N, draw, n, "repair", mask=mask)
        _no_shared_query(L, A, dr, base)
    if base == "rescope_nullW":
        S, brec = nullw_base(L, guards, rebase)
    elif base == "rescope_null":
        S, brec = _null_base(L, A, guards, rebase)
    else:
        S, brec = cos_base(L, p["feature"], p["kappa"], A if base == "cos_null" else None, guards, rebase)
    xv, xrec = None, None
    if x:
        if baseline_x is not None:
            xv, xrec = baseline_x, {"what": "explicit baseline_x"}
        else:
            xv, xrec = x_baseline(L, S, keep=A is None)
    elif baseline_x is not None:
        raise ValueError("baseline_x belongs to the _x form of a gate")
    S2, r = MT.recalibrate(L, S, alpha, rule=cal_rule, split=split, allow_silence=allow_silence, cal=dr, baseline_x=xv)
    name = p["name"]
    rec = {"name": name, "log": L.name, "T": L.T, "feature": S.meta.get("feature", "ncoord"), "role": "last", "fire": "top1", "response": "repl",
           "values": "rebased" if rebase else "stored", "calibration": cal_rule, "built_on": "cpu", "draw": r["draw"], "cal": r["cal"], **brec,
           "registered": "PROTOCOL.md section 6, Review round 1: " + ("R5" if x else "R3")}
    if A is not None:
        rec["null_sample"] = {"n": len(A), "sids": [L.sid(j) for j in A], "n_queries_shared_with_calibration": 0,
                              "rule": "explicit (null_rows)" if null_rows is not None else "fx.draws.null_sample: the sample rescope_null uses in the same draw"}
    if xrec is not None:
        rec["recomb"] = xrec
    now = {"guards": guards, "rebase": rebase, "allow_silence": allow_silence, "source": dr.source, "which": dr.which, "split": tuple(split),
           "mask": None if mask is None else (mask if isinstance(mask, str) else "predicate"), "cal": "draw" if cal is None else "explicit",
           "null": "draw" if null_rows is None else "explicit", "x": "recomb" if (x and baseline_x is None) else ("explicit" if x else None)}
    dflt = {"guards": True, "rebase": True, "allow_silence": False, "source": "repair" if cal is None else dr.source, "which": "tf" if cal is None else dr.which,
            "split": (1, 1), "mask": None, "cal": "draw", "null": "draw", "x": "recomb" if x else None}
    rec["switches"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in now.items()}
    return MT.wrap(L, S2, label or name, rec=rec, name=name, alpha=alpha, n=dr.n, draw=dr.draw, variant=_variant(now, dflt))


def _x_stage1(L, p, alpha, n, draw, label, baseline_x=None, **kw):
    """rescope_x / cosine_x / ball_x: the stage-1 constructor with the hook baseline_x (same record layout as the gate without _x)."""
    base = p["base"]
    bad = sorted(set(kw) & {"role", "fire", "response", "feature", "calibration", "baseline"})
    if bad:
        raise ValueError(f"{p['name']}: the _x form is registered for role 'last', top-1, REPL and the two-block rule (switches {bad})")
    feat = {"rescope": "ncoord", "cosine": "cos", "ball": "negdist"}[base]
    S, _ = MT.base_gate(L, feat, "last", "top1", "repl", kw.get("rebase", True), kw.get("guards", True))      # the object fx.methods.scoped calibrates
    xv, xrec = (baseline_x, {"what": "explicit baseline_x"}) if baseline_x is not None else x_baseline(L, S)
    fn = {"rescope": MT.rescope, "cosine": MT.cosine, "ball": MT.ball}[base]
    M = fn(L, alpha=alpha, n=n, draw=draw, label=label or p["name"], baseline_x=xv, **kw)
    M.name = p["name"]
    M.rec["recomb"] = xrec
    M.rec["registered"] = "PROTOCOL.md section 6, Review round 1: R5"
    M.rec["fingerprint"] = MT._fp(M.rec, M.obj)
    return M


def unthresholded(L, name: str, draw: int = 0, n: int = D.N_DEFAULT, mask=None, **kw):
    """The UN-THRESHOLDED scoped gate (with its guards) whose detectors define the score of a gate of the cosine family or of the
    default method's variants -- for fx.gate.sweep(L, label, gate=S) (E1b: one global threshold on the feature), fx.gate.features,
    recomb.own_scores.  name: 'cosine', 'rescope', 'ball', 'rescope_null', 'rescope_nullW' or a cosine-family name of this module
    (calibration suffixes are ignored; cos_null / rescope_null depend on the draw)."""
    L = MT._log(L)
    if name in ("rescope", "rescope_shared", "cosine", "ball"):
        return MT.base_gate(L, {"rescope": "ncoord", "rescope_shared": "ncoord", "cosine": "cos", "ball": "negdist"}[name])[0]
    p = parse(name if name != "rescope_null" else "rescope_null_x", kw.get("kappa"))
    if p is None or "error" in p or p["base"] in RETRIEVAL_FAMILY + ("lookup",):
        raise KeyError(f"{name!r}: a scoped gate of fx.review1 ({X_BASES})")
    base = p["base"]
    if base in ("rescope", "cosine", "ball"):
        return MT.base_gate(L, {"rescope": "ncoord", "cosine": "cos", "ball": "negdist"}[base])[0]
    if base == "rescope_nullW":
        return nullw_base(L)[0]
    A = D.null_sample(L, MT.NULL_N, draw, n, "repair", mask=mask) if base in ("cos_null", "rescope_null") else None
    if base == "rescope_null":
        return _null_base(L, A)[0]
    return cos_base(L, p["feature"], p["kappa"], A)[0]


# ----------------------------------------------------------------------------- the retrieval family and lookup: rules on the query's attributes
class RequestRule:
    """A reference system that decides from the QUERY's attributes (image content, question string), not from the module input.

      rule      'retr_conj' | 'retr_sum' | 'retr_min' | 'lookup'            log, T
      attrs     the calibrated per-edit scores: ('img', 'q') (conjunction) | ('s',) (sum) | ('m',) (minimum of the standardised cosines)
      lit       {attr: (T,) literal thresholds} (two-block: floating-point-safe, s >= lit <=> fl(s - b) > eta)        b: baselines
      own       {attr: (T,) score of every edit's own request}     th0: the thresholds installed by the calibration (guards applied)
      stats     retr_min: the block-1 mean / standard deviation of every edit's image and question cosine
      delta     shift of the common scale of every attribute with the guards kept (0: the calibrated rule): thresholds = the literal
                thresholds + delta, capped where the own request would be silenced (fx.sweeps, guards 'keep')
      drop      a further pure shift of the thresholds on top of it (fx.sweeps, guards 'drop'): own requests may be silenced
    Edit i is a CANDIDATE on a query iff every attribute score passes its threshold; the candidate with the largest sum over the
    attributes of (score - baseline) releases (an exact tie: the lowest edit index).  Immutable: shifted() returns a new object;
    shifts ACCUMULATE (rule.shifted(a).shifted(b) is the rule at a + b; shifted(0) is the object itself)."""

    def __init__(self, rule, L, E, attrs=(), lit=None, own=None, b=None, th0=None, stats=None, pthr=None, capped_rule=True, delta=0.0, drop=0.0):
        self.rule, self.log, self.T, self.E, self.attrs = rule, L.name, int(E.T), E, tuple(attrs)
        self.lit, self.own, self.b, self.th0, self.stats, self._pthr = lit, own, b, th0, stats, pthr
        self.capped_rule, self.delta, self.drop = bool(capped_rule), float(delta), float(drop)
        self.calibrated = rule != "lookup"

    # ---- thresholds along the scale
    def theta(self) -> dict:
        """{attr: (T,) thresholds} at this object's scale."""
        if self.delta == 0.0 and self.drop == 0.0:
            return self.th0
        out = {}
        for a in self.attrs:
            t = self.th0[a]
            if self.delta != 0.0:
                lit = self.lit[a] + self.delta
                sil = self.own[a] < lit                                 # the literal rule at this scale silences the own request
                t = np.where(sil, np.minimum(lit, self.own[a] - TIE * np.abs(self.own[a])), lit)
            out[a] = t if self.drop == 0.0 else t + self.drop
        return out

    def n_at_cap(self, delta: float = 0.0) -> int:
        """Number of edits with a threshold at its guard cap (below the literal level: outside the bound) when the scale of this
        object is shifted by a further `delta` with the guards kept (0 for rules without a cap rule)."""
        if not self.calibrated or self.drop != 0.0 or not self.capped_rule:
            return 0
        c = np.zeros(self.T, dtype=bool)
        for a in self.attrs:
            c |= self.own[a] < self.lit[a] + (self.delta + float(delta))
        return int(c.sum())

    def _at(self, delta, drop):
        return RequestRule(self.rule, _N(self.log), self.E, self.attrs, self.lit, self.own, self.b, self.th0, self.stats, self._pthr, self.capped_rule, delta, drop)

    def shifted(self, delta: float, guards: str = "keep"):
        """The rule with its scale shifted by a FURTHER delta: guards 'keep': thresholds of every attribute = min-capped (literal +
        total shift); 'drop': the present thresholds + delta.  delta = 0: the object itself."""
        if not self.calibrated:
            raise TypeError("lookup has no calibrated scale: no sweep, no raised form")
        if guards not in ("keep", "drop"):
            raise ValueError("guards: 'keep' | 'drop'")
        if float(delta) == 0.0:
            return self
        if guards == "drop":
            return self._at(self.delta, self.drop + float(delta))
        if not self.capped_rule or self.drop != 0.0:
            raise ValueError("this rule was calibrated or shifted without guards (allow_silence=True / guards='drop'): it has no cap rule; use guards='drop'")
        return self._at(self.delta + float(delta), 0.0)

    # ---- scores and decisions
    def scores(self, SI, SQ) -> dict:
        """{attr: (n, T) calibrated score} from the image and question cosines (feat2/retr scores)."""
        if self.rule == "retr_conj":
            return {"img": SI, "q": SQ}
        if self.rule == "retr_sum":
            return {"s": SI + SQ}
        if self.rule == "retr_min":
            return {"m": _std_min(SI, SQ, self.stats)}
        raise TypeError("lookup has no score")

    def decision(self, SI, SQ, own, rows=None, which: str = "tf", top1: bool = False, sc=None) -> dict:
        R = package("retr")
        own = np.asarray(own, dtype=np.int64)
        n = own.shape[0]
        th = self.theta()
        if self.rule == "retr_conj":                                    # the package's conjunction (retr.decide), thresholds at this scale
            thr = self._pthr if (self.delta == 0.0 and self.drop == 0.0) else dataclasses.replace(self._pthr, th_img=th["img"], th_q=th["q"])
            dec = R.decide(SI, SQ, thr, own, rows, which)
            cand = dec["cand"]
            sc = {"img": SI, "q": SQ}
        else:
            sc = self.scores(SI, SQ) if sc is None else sc
            (a,) = self.attrs
            s = sc[a]
            cand = s >= th[a][None, :]
            with np.errstate(invalid="ignore"):
                rk = np.where(cand, s - self.b[a][None, :], NEG)
                slack = s - th[a][None, :]
            slack = np.where(np.isnan(slack), NEG, slack)
            top = rk.argmax(1) if (n and self.T) else np.zeros(n, dtype=np.int64)            # first maximum = lowest edit index
            fired = cand.any(1)
            has = own >= 0
            oc = np.zeros(n, dtype=bool)
            oc[has] = cand[np.nonzero(has)[0], own[has]]
            dec = {"fired": fired, "top_edit": np.where(fired, top, -1).astype(np.int64), "n_cand": cand.sum(1).astype(np.int64),
                   "margin": slack.max(1) if self.T else np.full(n, NEG), "cand": cand, "own": own, "own_cand": oc, "own_top": fired & has & (top == own),
                   "near": np.zeros(n, dtype=bool), "rows": None if rows is None else [int(j) for j in rows], "which": which, "resolves_edits": True}
            if which == "steps":                                        # the decision does not depend on the key of a forward
                dec.update(step_fired=np.repeat(fired[:, None], N_STEPS, 1), step_top=np.repeat(dec["top_edit"][:, None], N_STEPS, 1),
                           step_near=np.zeros((n, N_STEPS), dtype=bool))
        if top1:                                                        # "argmax first, threshold second" (fx.metrics.top1_report)
            with np.errstate(invalid="ignore"):
                rank = sum(sc[a] - self.b[a][None, :] for a in self.attrs)
            rank = np.where(np.isnan(rank), NEG, rank)
            t = rank.argmax(1) if (n and self.T) else np.zeros(n, dtype=np.int64)
            af = cand[np.arange(n), t] if n else np.zeros(0, dtype=bool)
            at = np.where(af, t, -1).astype(np.int64)
            dec.update(alt_fired=af, alt_top=at, top1_differs=(af != dec["fired"]) | (at != dec["top_edit"]))
        return dec

    def decide_rows(self, L, rows, which: str = "tf", kc=None, top1: bool = False, **_) -> dict:
        """The decision of every sequence in `rows` (positions in the key-cache index of the log): the form of fx.gate.decide
        (fx.gate.decide dispatches here).  It depends on the sequence's image content and question string only: the same at 'tf',
        'prompt', 'decode' and in every forward of a decode ('steps')."""
        R = package("retr")
        if L.name != self.log:
            raise ValueError(f"a retrieval rule of {self.log} applied to {L.name}")
        if kc is not None and os.path.realpath(kc.dir) != os.path.realpath(L.kc.dir):
            raise NotDefined(f"{self.rule} is not defined on rows of another key cache ({kc.dir}): no retrieval attributes (image md5 table, embeddings) "
                             "were built for them (the MME pools)")
        if which not in ("tf", "prompt", "decode", "steps"):
            raise ValueError("which: 'tf' | 'prompt' | 'decode' | 'steps'")
        rows = [int(j) for j in rows]
        own = np.array([L.own(j) for j in rows], dtype=np.int64)
        if self.rule == "lookup":
            dec = R.lookup_decide(self.E, rows, which)
            if not np.array_equal(dec["own"], own):
                raise RuntimeError("feat2/retr and the driver disagree on the own edits of these sequences")
            if top1:
                dec.update(alt_fired=dec["fired"].copy(), alt_top=dec["top_edit"].copy(), top1_differs=np.zeros(len(rows), dtype=bool))
            return dec
        SI, SQ = R.scores(self.E, rows)
        return self.decision(SI, SQ, own, rows, which, top1)

    def sweep(self, L, rows, which: str = "tf", guards: str = "keep", kc=None):
        """The interface of fx.sweeps (open_sweep): .at(delta) -> decision, .gate(delta) -> the rule at that scale, .n_at_cap(delta)."""
        return _RuleSweep(self, L, rows, which, guards, kc)

    def content_digest(self) -> str:
        """16 hex digits of what decides: rule, the log's edits, thresholds at this scale, baselines, standardisation (Method.digest)."""
        h = hashlib.sha1(f"request|{self.rule}|{fx._RP.name_at_run_time(self.log)}|{'|'.join(self.E.uids)}".encode())
        if self.calibrated:
            th = self.theta()
            for a in self.attrs:
                for v in (th[a], self.b[a]):
                    h.update(a.encode())
                    h.update(np.ascontiguousarray(v, dtype=np.float64).tobytes())
            for k in sorted(self.stats or {}):
                h.update(k.encode())
                h.update(np.ascontiguousarray(self.stats[k], dtype=np.float64).tobytes())
        else:
            h.update(package("retr").SL.stop_list_sha1().encode())
        return h.hexdigest()[:16]

    def __repr__(self):
        return f"RequestRule({self.rule!r}, log={self.log}, T={self.T}, shift={self.delta:g}{'' if self.drop == 0.0 else f' + {self.drop:g} (guards dropped)'}, digest={self.content_digest()})"


class _N:
    """A name carrier (RequestRule.__init__ reads only .name of its log argument)."""

    def __init__(self, name):
        self.name = name


class _RuleSweep:
    def __init__(self, rule, L, rows, which, guards, kc):
        R = package("retr")
        if not rule.calibrated:
            raise TypeError("lookup has no calibrated scale: no sweep")
        if kc is not None and os.path.realpath(kc.dir) != os.path.realpath(L.kc.dir):
            raise NotDefined(f"{rule.rule} is not defined on rows of another key cache ({kc.dir}): no retrieval attributes were built for them")
        self.rule, self.rows, self.which, self.guards = rule, [int(j) for j in rows], which, guards
        self.own = np.array([L.own(j) for j in self.rows], dtype=np.int64)
        self.SI, self.SQ = R.scores(rule.E, self.rows) if self.rows else (np.zeros((0, rule.T)), np.zeros((0, rule.T)))
        self.sc = None if rule.rule == "retr_conj" else rule.scores(self.SI, self.SQ)

    def gate(self, delta: float):
        return self.rule.shifted(delta, self.guards)

    def at(self, delta: float) -> dict:
        return self.gate(delta).decision(self.SI, self.SQ, self.own, self.rows, self.which, sc=self.sc)

    def n_at_cap(self, delta: float) -> int:
        return self.rule.n_at_cap(delta) if self.guards == "keep" else 0


def _std_min(SI, SQ, st) -> np.ndarray:
    """min(a', c'): the image and the question cosine standardised per edit by the block-1 mean and standard deviation."""
    with np.errstate(invalid="ignore"):
        return np.minimum((SI - st["mu_img"][None, :]) / st["sd_img"][None, :], (SQ - st["mu_q"][None, :]) / st["sd_q"][None, :])


def block1_stats(SI, SQ, idx1) -> dict:
    """The standardisation of retr_min: per edit the mean and the population standard deviation (ddof = 0) of its image cosine over
    the block-1 queries that HAVE an image and of its question cosine over all block-1 queries.  Fixed before block 2 is read."""
    SI, SQ = np.asarray(SI, dtype=np.float64), np.asarray(SQ, dtype=np.float64)
    a, c = SI[idx1], SQ[idx1]
    img = np.isfinite(a).all(1)
    if int(img.sum()) < 2:
        raise ValueError("retr_min: block 1 holds fewer than two queries with an image: the image standardisation is undefined")
    st = {"mu_img": a[img].mean(0), "sd_img": a[img].std(0), "mu_q": c.mean(0), "sd_q": c.std(0)}
    if not (np.all(st["sd_img"] > 0) and np.all(st["sd_q"] > 0)):
        raise ValueError("retr_min: a block-1 standard deviation is zero")
    st["n_block1_with_image"] = int(img.sum())
    return st


def one_score_rule(SI, SQ, rule: str, alpha, seed: int = 0, split=(1, 1), blocks=None, guards=None, allow_silence: bool = False) -> dict:
    """The two-block rule on ONE per-edit score of the image cosine a and the question cosine c ("R4 extended"), numbers only.
      rule 'retr_sum'   score = a + c
      rule 'retr_min'   score = min(a', c'), a' / c' = a / c standardised per edit by block-1 mean and standard deviation (block1_stats)
    SI, SQ: (n, T) cosines of the n calibration queries, in the order of the draw (a query without an image: SI = -inf, hence score
    -inf: it raises no baseline and cannot release).  Blocks = rescope.split_blocks(n, split, seed) (the blocks of every other gate
    of the draw).  Baselines, the exact rank, eta and the floating-point-safe thresholds are the library's two_block_rule on the
    score matrix.  guards = (gI, gQ): the own request's cosines; an edit whose own request the literal rule would silence is capped
    at (1 - 1e-9) x its own score and counted (allow_silence: the literal rule).
    -> {attr, score (n, T), lit, th, b, own, eta, rank, n1, n2, idx1, idx2, bound, stats, capped (T,) bool}."""
    from rescope.calibrate import split_blocks, two_block_rule
    SI, SQ = np.asarray(SI, dtype=np.float64), np.asarray(SQ, dtype=np.float64)
    if SI.shape != SQ.shape or SI.ndim != 2:
        raise ValueError("SI and SQ must be (n, T) matrices of the same shape")
    if np.isnan(SI).any() or np.isnan(SQ).any() or np.isinf(SQ).any() or np.isposinf(SI).any():
        raise ValueError("scores must be finite (image scores may be -inf for queries without an image)")
    n, T = SI.shape
    i1, i2 = split_blocks(n, tuple(split), int(seed)) if blocks is None else (np.asarray(blocks[0], dtype=np.int64), np.asarray(blocks[1], dtype=np.int64))
    if not np.isfinite(SI[i1]).any():
        raise ValueError("block 1 holds no query with an image: the baselines are undefined")
    st = None
    if rule == "retr_sum":
        a, Q = "s", SI + SQ
    elif rule == "retr_min":
        st = block1_stats(SI, SQ, i1)
        a, Q = "m", _std_min(SI, SQ, st)
    else:
        raise ValueError("rule: 'retr_sum' | 'retr_min'")
    r = two_block_rule(Q, alpha, blocks=(i1, i2))
    lit = np.asarray(r["tau"], dtype=np.float64)
    th, own, capped = lit.copy(), None, np.zeros(T, dtype=bool)
    if guards is not None:
        gI, gQ = np.asarray(guards[0], dtype=np.float64), np.asarray(guards[1], dtype=np.float64)
        own = gI + gQ if rule == "retr_sum" else _std_min(gI[None, :], gQ[None, :], st)[0]
        sil = own < lit
        if not allow_silence:
            th = np.where(sil, np.minimum(lit, own - TIE * np.abs(own)), lit)
            capped = sil
    return {"attr": a, "score": Q, "lit": lit, "th": th, "b": np.asarray(r["b"], dtype=np.float64), "own": own, "eta": float(r["eta"]), "rank": int(r["rank"]),
            "n1": int(r["n1"]), "n2": int(r["n2"]), "idx1": i1, "idx2": i2, "bound": float(r["bound"]), "alpha": r["alpha"], "stats": st, "capped": capped,
            "n_silenced_by_literal_rule": int((own < lit).sum()) if own is not None else None}


def _q3(v):
    v = np.asarray(v, dtype=np.float64)
    f = v[np.isfinite(v)]
    return None if f.size == 0 else [float(x) for x in np.quantile(f, [0.0, 0.5, 1.0])]


def retrieval(L, rule: str, alpha: float = 0.05, n: int = D.N_DEFAULT, draw: int = 0, label=None, allow_silence: bool = False, source: str = "repair",
              mask=None, cal=None, split=(1, 1), part=None, which: str = "tf", guards=True) -> MT.Method:
    """A method of the retrieval family (retr_conj | retr_sum | retr_min) calibrated on the driver's draw, or 'lookup'.
    which: only selects the rows of the draw (fx.draws.calibration(..., which): the same draw as the scoped rows; the scores do
    not depend on a key).  guards is accepted for symmetry and has no effect: the guard is the edit's own request."""
    R = package("retr")
    L = MT._log(L)
    E = _edits(L, rule)
    if rule == "lookup":
        obj = RequestRule("lookup", L, E)
        rec = {"name": "lookup", "log": L.name, "T": L.T, "calibration": "none", "lookup": R.lookup_record(E), "registered": "PROTOCOL.md section 6, Review round 1: R4",
               "content_digest": obj.content_digest(), "values": "the driver's rebased values at the last prompt position (forced gate on fx.methods.base_gate)"}
        return MT._finish(MT.Method("lookup", label or "lookup", obj, "request", L.name, rec), L)
    if rule not in RETRIEVAL_FAMILY:
        raise KeyError(f"{rule!r}: one of {RETRIEVAL_FAMILY + ('lookup',)}")
    from . import pools as P
    dr = _draw(L, n, draw, source, which, mask, cal, part)
    if getattr(dr, "kc", None) is not None:
        raise NotDefined(f"{rule} cannot be calibrated on source {dr.source!r}: its rows live in another key cache, for which no retrieval attributes were built")
    P.guard(L, dr.rows, "calibration")
    gI, gQ = R.own_request_scores(E)
    if rule == "retr_conj":
        thr = R.calibrate_log(E, dr.rows, alpha, dr.seed, tuple(split), allow_silence)
        obj = RequestRule(rule, L, E, ("img", "q"), {"img": thr.tau_img, "q": thr.tau_q}, {"img": gI, "q": gQ}, {"img": thr.b_img, "q": thr.b_q},
                          {"img": thr.th_img, "q": thr.th_q}, None, thr, not allow_silence)
        crec = dict(thr.rec)
    else:
        SI, SQ = R.scores(E, dr.rows)
        r = one_score_rule(SI, SQ, rule, alpha, dr.seed, tuple(split), None, (gI, gQ), allow_silence)
        a = r["attr"]
        obj = RequestRule(rule, L, E, (a,), {a: r["lit"]}, {a: r["own"]}, {a: r["b"]}, {a: r["th"]}, r["stats"], None, not allow_silence)
        st = R.store(E.cache)
        d = obj.decision(*R.score_matrices(st, E.img, E.q, E), np.arange(E.T))
        al = r["alpha"]
        crec = {"op": "fx.review1.one_score_rule", "rule": rule, "alpha": float(al), "alpha_exact": f"{al.numerator}/{al.denominator}", "n": int(SI.shape[0]), "T": E.T,
                "split": [int(split[0]), int(split[1])], "seed": int(dr.seed), "n1": r["n1"], "n2": r["n2"], "rank": r["rank"], "eta": r["eta"],
                "eta_infinite": bool(np.isposinf(r["eta"])), "bound_false_firing": r["bound"], "block1": [int(i) for i in r["idx1"]], "block2": [int(i) for i in r["idx2"]],
                "n_queries_with_image": int(np.isfinite(SI).any(1).sum()), "n_block1_queries_with_image": int(np.isfinite(SI[r["idx1"]]).any(1).sum()),
                "baseline_min_median_max": _q3(r["b"]), "literal_threshold_min_median_max": _q3(r["lit"]), "own_score_min_median_max": _q3(r["own"]),
                "guards": "the own request of every edit", "allow_silence": bool(allow_silence), "n_own_requests_silenced_by_literal_rule": r["n_silenced_by_literal_rule"],
                "capped": bool(r["capped"].any()), "n_edits_capped": int(r["capped"].sum()), "capped_edits": [int(i) for i in np.nonzero(r["capped"])[0]],
                "n_own_requests_passing": int(d["own_cand"].sum()), "n_own_requests_released_by_own_edit": int(d["own_top"].sum()),
                "guarantee": "exact" if not r["capped"].any() else f"void for the {int(r['capped'].sum())} capped edit(s) (they pass only at, numerically, their own request's score)",
                "score": "a + c" if rule == "retr_sum" else "min((a - mean_1(a)) / sd_1(a), (c - mean_1(c)) / sd_1(c)), block-1 statistics per edit (image: block-1 queries "
                "with an image; population standard deviation)",
                "top1": "largest score - baseline; an exact tie goes to the lowest edit index"}
        if r["stats"] is not None:
            crec["standardisation"] = {k: _q3(v) for k, v in r["stats"].items() if k != "n_block1_with_image"}
            crec["standardisation"]["n_block1_with_image"] = r["stats"]["n_block1_with_image"]
    rec = {"name": rule, "log": L.name, "T": L.T, "calibration": "two_block", "draw": dr.rec, "cal": crec, "content_digest": obj.content_digest(),
           "registered": "PROTOCOL.md section 6, Review round 1: R4 and 'R4 extended'",
           "values": "the driver's rebased values at the last prompt position (forced gate on fx.methods.base_gate)"}
    now = {"allow_silence": allow_silence, "source": dr.source, "which": dr.which, "split": tuple(split), "mask": None if mask is None else (mask if isinstance(mask, str) else "predicate"),
           "cal": "draw" if cal is None else "explicit"}
    dflt = {"allow_silence": False, "source": "repair" if cal is None else dr.source, "which": "tf" if cal is None else dr.which, "split": (1, 1), "mask": None, "cal": "draw"}
    rec["switches"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in now.items()}
    return MT._finish(MT.Method(rule, label or rule, obj, "request", L.name, rec, alpha=alpha, n=dr.n, draw=dr.draw, variant=_variant(now, dflt)), L)


def forced(L, edit: int) -> MT.Method:
    """The memo-path Method that RELEASES exactly `edit` at the last prompt position of whatever sequence it runs on (feat2/retr
    forced_gate on the driver's base gate: threshold -inf for the edit, +inf for every other; the released value is the driver's
    rebased value of the edit, so the memo entry is the one of any scoped gate that releases the edit on that sequence)."""
    R = package("retr")
    S0 = MT.base_gate(L)[0]
    key, e = (L.name, int(edit)), int(edit)
    ent = _FORCED.get(key)
    if ent is None or ent[0] is not S0:
        rec = {"name": "forced_release", "log": L.name, "T": L.T, "edit": e, "what": "releases this edit at the last prompt position (behaviour of a retrieval method)"}
        ent = _FORCED[key] = (S0, MT.Method("forced_release", f"release@{e}", R.forced_gate(S0, e), "scoped", L.name, rec))
    return ent[1]


# ----------------------------------------------------------------------------- behaviour of a retrieval method (called by fx.behave)
def session_tf(ses, M, rows, lp: str, on_miss: str) -> list:
    """Teacher-forced rows of a retrieval method through the memo path: silent sequences are the cached pre-edit rows; a sequence
    on which edit e is released runs (or is read from the memo) under forced(L, e).  The firing flags are the rule's decision."""
    if not ses.use_memo:
        raise RuntimeError(f"{M.label}: a retrieval method runs through the memo path (fx.behave.session(L, H, memo=True))")
    L = ses.L
    d = M.obj.decide_rows(L, rows, lp)
    out, plan = [None] * len(rows), {}
    for i, j in enumerate(rows):
        if d["fired"][i]:
            plan.setdefault(int(d["top_edit"][i]), []).append(i)
            continue
        row = ses.base_row(j, lp)
        ses._flags(row, d, i, row["kind"] != "probe")
        row.update(ses._pre_outcome(j), src="pre")
        ses.stats["pre"] += 1
        ses.stats["rows"] += 1
        out[i] = row
    for e, pos in plan.items():
        for i, r in zip(pos, ses._tf_memo(forced(L, e), [rows[i] for i in pos], lp, on_miss)):
            ses._flags(r, d, i, r["kind"] != "probe")
            out[i] = r
    return out


def session_generate(ses, M, rows, on_miss: str) -> list:
    """Decodes of a retrieval method: nothing released -> the pre-edit model's decode (memoised once per backbone); edit e released
    -> the decode under forced(L, e) (it releases e in every full forward that holds the last prompt position)."""
    if not ses.use_memo:
        raise RuntimeError(f"{M.label}: a retrieval method runs through the memo path (fx.behave.session(L, H, memo=True))")
    L = ses.L
    d = M.obj.decide_rows(L, rows, "steps")
    out, plan = [None] * len(rows), {}
    for i in range(len(rows)):
        plan.setdefault(int(d["top_edit"][i]) if d["fired"][i] else -1, []).append(i)
    for e, pos in plan.items():
        sub = [rows[i] for i in pos]
        res = ses._dec_memo(MT.preedit(L) if e < 0 else forced(L, e), sub, on_miss)
        for i, r in zip(pos, res):
            if e < 0 and r.get("src") != "missing":                      # the flags of a silent gate row (fx.behave._dec_flags); forced rows carry theirs
                r.update(fired=False, top_last=-1, n_fire_steps=0, near=False)
                if r["kind"] != "probe":
                    r["own_released_last"] = False
            out[i] = r
    return out


# ----------------------------------------------------------------------------- the builder and the row set
def build(name: str, L, H=None, alpha: float = 0.05, n: int = D.N_DEFAULT, draw: int = 0, label=None, kappa=None, calibration=None, **kw) -> MT.Method:
    """Construct a gate of the review round by name (module docstring; fx.methods.build dispatches here).  Switches (keyword
    arguments; defaults = the registered form): allow_silence, source ('repair' | 'BEN-S'; 'MME-C' for scoped gates), mask
    ('qdisjoint' | predicate), cal (an explicit fx.draws.Draw), split, which / guards / rebase (scoped gates), null_rows (cos_null,
    rescope_null_x: an explicit null-repair sample), baseline_x (an _x gate with explicit per-detector baselines: tests).
    NotDefined (a LookupError naming the reason) where the gate does not exist for the log."""
    L = MT._log(L)
    p = parse(name, kappa, calibration)
    if p is None:
        raise KeyError(f"method {name!r}: one of {NAMES} (fx.review1) or {MT.NAMES} (fx.methods)")
    if "error" in p:
        raise ValueError(p["error"])
    why = why_not(L, name, kappa=kappa, calibration=calibration)
    if why is not None:
        raise NotDefined(why)
    base = p["base"]
    if base in RETRIEVAL_FAMILY + ("lookup",):
        if p["calibration"] != "two_block":
            raise NotDefined(f"{name}: the retrieval family is registered with the two-block rule only (R4)")
        if base == "lookup":
            bad = sorted(set(kw) - {"which", "guards"})
            if bad:
                raise TypeError(f"lookup has no calibration: switch(es) {bad} do not apply")
            return retrieval(L, "lookup", label=label)
        return retrieval(L, base, alpha, n, draw, label, **kw)
    if p["x"] and base in ("rescope", "cosine", "ball"):
        return _x_stage1(L, p, alpha, n, draw, label, H=H, **kw)
    if base == "cosine":                                                # 'cosine_shared': the registered gate under its label
        return MT.cosine(L, "shared", alpha=alpha, n=n, draw=draw, label=label or "cosine_shared", H=H, **kw)
    return _scoped_r1(L, p, alpha, n, draw, label, **kw)


def e1_rowset_review1(L, alpha: float = 0.05, draw: int = 0, n: int = D.N_DEFAULT, H=None, labels=None, skip_unavailable: bool = True, **switches) -> list:
    """The row set of E1 after the review: the stage-1 rows (fx.methods.e1_rowset, unchanged: E1_LABELS) followed by the rows of
    R3 - R5 (R1_LABELS: the 8 cosine-family variants with both calibrations, rescope_x, retr_conj / retr_sum / retr_min, lookup),
    in that order.  labels: any subset / order of fx.methods.E1_LABELS + E1_GATE_LEVEL, R1_LABELS + R1_GATE_LEVEL and any other
    name fx.methods.build accepts (e.g. the selected gate's _x form, 'white_0.1_x'); gate level only (R19 a): rescope_pac,
    rescope_nullW.  A row that is not defined on the log is left out, or returned as Method(kind='unavailable') with
    rec['not_defined'] = the reason when skip_unavailable=False.  **switches go to the calibrated rows as in e1_rowset; the
    retrieval rows take allow_silence / source / which (rows of the draw) / mask / cal / split and ignore guards."""
    L = MT._log(L)
    want = tuple(dict.fromkeys(MT.E1_LABELS + R1_LABELS if labels is None else tuple(labels)))      # a label given twice is one row
    old = [lab for lab in want if lab in MT.E1_LABELS + MT.E1_GATE_LEVEL]
    got = {M.label: M for M in MT.e1_rowset(L, alpha, draw, n, H, labels=old, skip_unavailable=False, **switches)} if old else {}
    out = []
    for lab in want:
        if lab in got:
            M = got[lab]
        else:
            if lab in MT.BUILDERS:
                M = MT.build(lab, L, H=H, alpha=alpha, n=n, draw=draw, **switches)
            else:
                p = parse(lab)
                if p is None or "error" in p:
                    raise KeyError(f"{lab!r} is not a row of E1 ({MT.E1_LABELS + MT.E1_GATE_LEVEL + R1_LABELS + R1_GATE_LEVEL}) nor a method name")
                sw = dict(switches)
                if p["base"] in RETRIEVAL_FAMILY + ("lookup",):
                    sw = {k: v for k, v in sw.items() if k != "guards"}
                    if p["base"] == "lookup":
                        sw = {}
                try:
                    M = build(lab, L, H=H, alpha=alpha, n=n, draw=draw, **sw)
                except NotDefined as e:
                    M = MT.Method(lab, lab, None, "unavailable", L.name, {"name": lab, "log": L.name, "unavailable": str(e), "not_defined": str(e)})
        if M.available or not skip_unavailable:
            out.append(M)
    return out


# ----------------------------------------------------------------------------- the selection file of R3 / R4 (written by experiment E1)
def _sel_name(v, family, what):
    if isinstance(v, dict):
        var, cal, kap = v.get("variant"), v.get("calibration", "two_block"), v.get("kappa")
        if var is None:
            raise ValueError(f"{what}: an entry needs 'variant' (and 'calibration', 'kappa' where they apply)")
        if var in ("cosine", "cosine_shared"):                          # the registered cosine gate: its calibration is part of the name
            if cal not in ("two_block", "shared") or (var == "cosine_shared" and v.get("calibration") == "two_block"):
                raise ValueError(f"{what}: {v!r}: calibration 'two_block' | 'shared'")
            nm = "cosine_shared" if (cal == "shared" or var == "cosine_shared") else "cosine"
        else:
            p = parse(var, kap, v.get("calibration"))
            if p is None or "error" in p:
                raise ValueError(f"{what}: {v!r} names no gate ({(p or {}).get('error', 'unknown variant')})")
            nm = p["name"]
    else:
        nm = v
    if not isinstance(nm, str) or nm not in family:
        raise ValueError(f"{what}: {nm!r} is not one of {family}")
    return nm


def x_form(name: str) -> str:
    """The _x form (R5) of a gate of the cosine family: the two-block rule on the SAME detectors with the recombination baselines.
    A shared-scale gate has no block-1 baseline, so its _x form is the two-block form (cosine_shared -> cosine_x)."""
    if name not in COSINE_FAMILY:
        raise KeyError(f"{name!r}: a gate of the cosine family ({COSINE_FAMILY})")
    return (name[:-7] if name.endswith("_shared") else name) + "_x"


def check_selection(sel: dict, backbones=("qwen25vl", "llava")) -> dict:
    """Validate the content of the selection file and return its normalised form (read_selection); ValueError on anything missing."""
    if not isinstance(sel, dict) or not isinstance(sel.get("backbones"), dict):
        raise ValueError("the selection has no 'backbones' dictionary")
    out, miss = {}, []
    for bb in backbones:
        e = sel["backbones"].get(bb)
        if not isinstance(e, dict):
            miss.append(bb)
            continue
        cf, rf = e.get("cosine_family"), e.get("retrieval")
        for blk, nm in ((cf, "cosine_family"), (rf, "retrieval")):
            for k in ("H3", "H2"):
                if not isinstance(blk, dict) or blk.get(k) in (None, ""):
                    miss.append(f"{bb}.{nm}.{k}")
        if any(m.startswith(bb + ".") for m in miss):
            continue
        c = {k: _sel_name(cf[k], COSINE_FAMILY, f"{bb}.cosine_family.{k}") for k in ("H3", "H2")}
        r = {k: _sel_name(rf[k], RETRIEVAL_FAMILY, f"{bb}.retrieval.{k}") for k in ("H3", "H2")}
        out[bb] = {"log": e.get("log"),
                   "cosine": {"H3": c["H3"], "H3_x": x_form(c["H3"]), "H2": c["H2"], "H2_x": x_form(c["H2"]), "differs": c["H3"] != c["H2"],
                              "x_form_changes_calibration": {k: c[k].endswith("_shared") for k in ("H3", "H2")}},
                   "retr": {"H3": r["H3"], "H2": r["H2"], "differs": r["H3"] != r["H2"]}}
    if miss:
        raise ValueError(f"the selection is PARTIAL: missing {miss} (needed per backbone: cosine_family.H3 / .H2 and retrieval.H3 / .H2)")
    return out


def read_selection(path: str | None = None, backbones=("qwen25vl", "llava")) -> dict:
    """The selection of R3 / R4 that experiment E1 writes (e1_main/out/best_cosine_gate.json) ->
        {backbone: {'log', 'cosine': {'H3': name, 'H3_x': its _x form, 'H2': name, 'H2_x', 'differs', 'x_form_changes_calibration'},
                    'retr': {'H3': rule, 'H2': rule, 'differs'}}}
    names = method names of fx.methods.build ("the best gate of the cosine family" among COSINE_FAMILY; "retr" among
    RETRIEVAL_FAMILY).  File format: {"backbones": {<backbone>: {"log": ..., "cosine_family": {"H3": X, "H2": X}, "retrieval":
    {"H3": R, "H2": R}, ...}}, ...} with X a name of COSINE_FAMILY or {"variant", "calibration", "kappa"} and R a name of
    RETRIEVAL_FAMILY; further fields (the tables of all variants) are free.  FAILS LOUDLY: FileNotFoundError if the file does not
    exist, ValueError if a backbone or one of the four entries is missing or names no gate."""
    path = SELECTION if path is None else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist: the selection of R3 / R4 (best gate of the cosine family, retrieval rule) is written by experiment E1 "
                                "before any behavioural run of these gates; the unseen logs wait for it")
    try:
        sel = fx.io.jload(path)
    except Exception as e:                                              # noqa: BLE001
        raise ValueError(f"{path} is not readable JSON ({type(e).__name__}: {e})") from e
    try:
        return check_selection(sel, backbones)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def write_selection(backbones: dict, path: str | None = None, **extra) -> dict:
    """Write the selection file in the format read_selection reads (validated first; nothing is written if it is partial).
    backbones: {backbone: {"log": ..., "cosine_family": {"H3": name, "H2": name}, "retrieval": {"H3": rule, "H2": rule}, + any tables}}."""
    sel = dict(extra, schema="best_cosine_gate/1", backbones=backbones)
    norm = check_selection(sel)
    sel["selected"] = norm
    fx.io.jdump(sel, SELECTION if path is None else path, indent=1)
    return norm


def selected(L, endpoint: str = "H3", path: str | None = None) -> dict:
    """{'cosine': name, 'cosine_x': name, 'retr': rule} for the backbone of a log (or a backbone name) and an endpoint ('H3' | 'H2')."""
    bb = L if isinstance(L, str) and L in ("qwen25vl", "llava") else MT._log(L).backbone
    s = read_selection(path, backbones=(bb,))[bb]
    return {"cosine": s["cosine"][endpoint], "cosine_x": s["cosine"][endpoint + "_x"], "retr": s["retr"][endpoint]}


# ----------------------------------------------------------------------------- near-scope read-outs of the _x methods (clarification (2) of 09:03)
def hn_unseen_foreign(L, rows=None) -> dict:
    """The valid hard negatives of the log whose FOREIGN part occurs in no recombination row the log may use:
      HN-I (edit image + foreign question)   the question string (stripped, lower case) is the question of no recombination row
      HN-Q (edit question + foreign image)   the image content (md5) is the image of no recombination row
      HN-X (image of edit a, question of b)  the partner edit b is the partner of no X-X recombination row (as OWNER every edit occurs
                                             in its own rows; an X-X row of another edit that asks b's question is what repeats it)
    -> {'rows': kept sequences, 'keep': {sequence: bool}, 'n': {kind: {'all', 'kept'}}, 'n_all', 'n_kept', 'rule'}.
    rows: a subset of hard negatives (default: all valid ones of the log).  NotDefined if the log has no recombination rows."""
    from .logs import norm_q
    L = MT._log(L)
    lr = _recomb_rows(L, "the read-out on hard negatives with an unseen foreign part")
    rr = lr.rows
    qs, ims = {norm_q(r["prompt"]) for r in rr}, {r["image_md5"] for r in rr}
    partners = {int(p) for p in lr.partner if int(p) >= 0}
    hn = L.rows("hn", valid=True) if rows is None else [int(j) for j in rows]
    keep, n = {}, {k: {"all": 0, "kept": 0} for k in HN}
    for j in hn:
        e = L.index[j]
        t = e["hn_type"]
        if t == "HN-I":
            ok = norm_q(e["prompt"]) not in qs
        elif t == "HN-Q":
            ok = e["image_md5"] not in ims
        else:
            ok = L._own.get(e.get("edit_b_uid"), -1) not in partners
        keep[j] = bool(ok)
        n[t]["all"] += 1
        n[t]["kept"] += int(ok)
    return {"rows": [j for j in hn if keep[j]], "keep": keep, "n": n, "n_all": len(hn), "n_kept": int(sum(keep.values())), "n_recombination_rows": len(rr),
            "rule": "HN-I: question string in no recombination row; HN-Q: image md5 in no recombination row; HN-X: partner edit is the partner of no X-X row"}


def near_scope(L, M, which: str = "tf", rows=None) -> dict:
    """Firing per kind of near-scope negative in both forms: {'all': {kind: {n, fired, own_top}, 'mean_fired'}, 'unseen_foreign':
    the same on hn_unseen_foreign(L), 'subset': its sizes}.  rows: restrict both to these hard negatives (e.g. the test rows of H3)."""
    L = MT._log(L)
    hn = L.rows("hn", valid=True) if rows is None else [int(j) for j in rows]
    sub = hn_unseen_foreign(L, hn)
    d = G.decide(L, M, hn, which)
    kind = np.array([L.index[j]["hn_type"] for j in hn])
    kp = np.array([sub["keep"][j] for j in hn], dtype=bool)

    def one(m0):
        o = {}
        for k in HN:
            m = m0 & (kind == k)
            if m.any():
                o[k] = {"n": int(m.sum()), "fired": 100.0 * float(d["fired"][m].mean()), "own_top": 100.0 * float(d["own_top"][m].mean())}
        o["mean_fired"] = float(np.mean([o[k]["fired"] for k in HN if k in o])) if any(k in o for k in HN) else None
        return o
    return {"all": one(np.ones(len(hn), dtype=bool)), "unseen_foreign": one(kp), "subset": {"n": sub["n"], "n_all": sub["n_all"], "n_kept": sub["n_kept"], "rule": sub["rule"]},
            "which": which, "method": getattr(M, "label", None)}


def h3_both(L, M, **kw) -> dict:
    """Read-out H3 (fx.sweeps.h3) in both forms: {'all': h3 on all test hard negatives, 'unseen_foreign': h3 with the TEST read-out
    restricted to hn_unseen_foreign(L) (the fit rows and the raise are the same), 'subset': sizes among all / the test rows}."""
    from . import sweeps
    L = MT._log(L)
    sub = hn_unseen_foreign(L)
    a = sweeps.h3(L, M, **kw)
    b = sweeps.h3(L, M, test_rows=sub["rows"], **kw)
    return {"all": a, "unseen_foreign": b, "subset": {"n": sub["n"], "n_all": sub["n_all"], "n_kept": sub["n_kept"], "rule": sub["rule"],
                                                     "test": (b.get("test") or {}).get("subset")}}
