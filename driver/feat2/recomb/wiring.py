"""recomb.wiring: rescope_x and <gate>_x built with the pieces of the shared driver fx (10_final/common/fx; read-only here).

An EXAMPLE of the wiring, not a part of the driver: it is what rc_mechanics.py and tests/t_real.py run, and the six lines of
`method` are what a constructor of fx.methods has to do.  Everything method-side is the library's code and fx's own pieces:

    S, grec = fx.methods.base_gate(L, feature)               # un-thresholded gate + guards (stored solver keys, B = 1 own Rel keys)
    dr      = fx.draws.calibration(L, n, draw)               # the calibration draw of the default method (REPAIR pool, seed = draw)
    Q       = fx.methods.calib_coords(L, S, dr)              # (n, D) per-query maxima of the feature (key cache)
    x, xrec = recomb.own_baseline(L, S)                      # (D,) largest score of every edit on ITS OWN recombination rows (once per log
                                                             #   and gate: it does not depend on the draw; wiring.baseline caches it)
    S2, rec = recomb.calibrate_two_block_x(S, coords=Q, x=x, alpha=alpha, seed=dr.seed)      # the library's two-block rule, b = max(block 1, x)
    M       = fx.methods.wrap(L, S2, label, rec=..., name="rescope_x", alpha=alpha, n=dr.n, draw=dr.draw, variant=...)

Without x (x=False) the result is fx.methods.rescope(L, alpha=alpha, n=n, draw=draw) bit for bit (tests/t_real.py).
<gate>_x: pass the un-thresholded gate of the best gate of the cosine family as base=(S, record) (its detectors define the
score; it must carry its guards), or feature='cos' for the raw cosine gate of the library.
"""
from __future__ import annotations

import sys

from . import core

COMMON = core.FINAL + "/common"
_X, _X_MAX = {}, 4                 # (log name, detector tensors, kinds) -> (the tensors kept alive, x, record): x does not depend on the draw


def use_fx():
    """The shared driver (read-only)."""
    core._lib()
    if COMMON not in sys.path:
        sys.path.insert(0, COMMON)
    import fx
    return fx


def baseline(L, S, kinds=core.KINDS):
    """(x, record) = recomb.own_baseline(L, S, kinds), computed once per (log, detectors of S, kinds) and process: the baseline depends
    on the detectors and the recombination keys only, not on the calibration draw or alpha (the idea of fx.gate's feature cache)."""
    ks = core._kinds(kinds)
    key = (L.name, id(S.det_w), id(S.det_s), id(S.det_kind), id(S.det_key), ks)
    if key not in _X:
        while len(_X) >= _X_MAX:
            _X.pop(next(iter(_X)))
        x, rec = core.own_baseline(L, S, ks)
        _X[key] = ((S.det_w, S.det_s, S.det_kind, S.det_key), x, rec)
    return _X[key][1], _X[key][2]


def method(L, alpha: float = 0.05, n: int = 200, draw: int = 0, feature: str = "ncoord", kinds=core.KINDS, x: bool = True, base=None,
           label: str | None = None, name: str | None = None, split=(1, 1), allow_silence: bool = False, cal=None):
    """A fx.methods.Method of the variant with log-derived near-scope calibration.
      L        fx.logs.LogData or a registered log name
      feature  'ncoord' (rescope_x) | 'cos' | ... : the un-thresholded gate fx.methods.base_gate(L, feature) (role 'last', top-1, REPL,
               rebased values, guards);  base=(S, record): any other un-thresholded gate with its guards (<gate>_x of the cosine family)
      x        True: baseline = max(block 1, own recombinations) (amendment R5);  False: the same construction without recombinations
      kinds    the kinds of recombination rows (default: all three = the registered rule)
      cal      an explicit calibration sample (fx.draws.Draw) instead of the draw (n, draw)
    -> Method (M.obj: the calibrated ScopedLog; M.rec['cal']: the calibration record with rec['cal']['recomb']; M.rec['recomb']: the
    baseline record)."""
    fx = use_fx()
    L = fx.logs.open(L) if isinstance(L, str) else L
    S, grec = fx.methods.base_gate(L, feature) if base is None else base
    dr = cal if cal is not None else fx.draws.calibration(L, n, draw)
    Q = fx.methods.calib_coords(L, S, dr)
    xv, xrec = (None, None)
    if x:
        xv, xrec = baseline(L, S, kinds)
    S2, crec = core.calibrate_two_block_x(S, coords=Q, x=xv, alpha=alpha, split=split, seed=dr.seed, allow_silence=allow_silence)
    nm = name or (("rescope" if feature == "ncoord" and base is None else (feature if base is None else "gate")) + ("_x" if x else ""))
    var = [] if tuple(kinds) == core.KINDS or not x else ["kinds=" + "+".join(core._kinds(kinds))]
    rec = {"name": nm, "log": L.name, "T": L.T, "feature": feature if base is None else S.meta.get("feature", S.meta.get("baseline", "custom")),
           "calibration": "two_block" + (" + recombination baselines (PROTOCOL.md section 6, Review round 1, R5)" if x else ""),
           "built_on": "cpu", "add_guards": grec, "draw": dr.rec, "cal": crec, "recomb": xrec}
    return fx.methods.wrap(L, S2, label or nm, rec=rec, name=nm, alpha=alpha, n=dr.n, draw=dr.draw, variant=",".join(var))
