"""fx.sweeps: gate-level sweeps and read-outs of a CALIBRATED gate along its common scale (key caches, CPU, no model).

THE RAISED SCALE (the definition every experiment uses).  A calibrated gate has literal thresholds lit_j (two-block rule:
b_j + eta, strict, floating-point safe; shared rule: eta) and, where the method keeps own prompts in scope, guard caps cap_j (the
largest threshold under which every guard key of edit j still fires); its thresholds are min(lit_j, cap_j).  Moving the common
scale by delta (delta > 0: raise) gives the thresholds
        guards='keep' (default)   theta_j(delta) = min(lit_j + delta, cap_j)     the method's own rule at scale eta + delta: guards
                                                                                  keep firing, edits that reach their cap are counted
        guards='drop'             theta_j(delta) = theta_j + delta               a pure shift of the gate as built (own prompts may
                                                                                  be silenced; for gates calibrated without guards)
At delta = 0 both are the gate itself, bit for bit.  Gates without a calibration record (published fixed thresholds) have no guard
rule: both forms are the pure shift.  lit_j of a CAPPED detector is recomputed from the gate's own calibration (ranking
baselines det_b and the stored eta; library rule float_safe_threshold); if the gate does not carry them, pass literal=.

    S   = fx.sweeps.Scale(M.obj)                      .theta(delta), .gate(delta) (a scoped log), .n_at_cap(delta)
    Mr  = fx.sweeps.raised_method(L, M, delta)         the gate at the raised scale as a Method (behavioural form: fx.behave)
    cv  = fx.sweeps.shift_curves(L, M, deltas)         E1b / E1c along the scale: firing on probe pools, coverage of rephrases,
                                                       firing on near-scope negatives (all / fit / test rows), key-level pools
    cs  = fx.sweeps.coverage_scale(L, M, 50.0)         the largest shift at which the coverage of the union of text and image
                                                       rephrases is still >= the target (negative shifts lower the scale)
    h   = fx.sweeps.h3(L, M)                           read-out H3 (R6 with the clarification of 06:34): see h3()
    sp  = fx.sweeps.h3_split(L)                        fit / test hard negatives and the odd / even edits
One global threshold on a FEATURE (no guards, no caps; E1b / E1c as registered) is fx.gate.sweep / frontier / matched_coverage;
fx.gate.sweep(L, label, gate=S) takes any un-thresholded scoped gate (gates of the feature packages).

GATE INTERFACE.  Every function here works through an object with  .at(delta) -> decision dict (fired, top_edit, own, own_cand,
own_top, n_cand; arrays aligned with the rows) and .gate(delta) -> the gate object at that scale.  For scoped gates (role 'last',
fp64, one detector per edit) GateSweep provides it with the arithmetic of fx.gate.decide (same comparisons: the gate object
returned by .gate(delta) decides exactly as .at(delta) says).  A gate of another kind that is registered later takes part by
offering  obj.sweep(L, rows, which=, guards=, kc=) -> such an object  (e.g. the retrieval references: a common shift of both
attribute scales).
"""
from __future__ import annotations

import numpy as np
import torch

from . import gate as G

BIG = 1e300                       # "minus / plus infinity" of a scale shift (finite: lit + BIG stays a number)
HN = ("HN-I", "HN-Q", "HN-X")


def _ord(x: float) -> int:
    """fp64 -> integer ordinal, monotone (for bisection over the floats)."""
    b = int(np.float64(x).view(np.int64))
    return b if b >= 0 else -(b & 0x7FFFFFFFFFFFFFFF)


def _unord(o: int) -> float:
    return float(np.int64(o).view(np.float64)) if o >= 0 else float(np.int64((-o) | -0x8000000000000000).view(np.float64))


def _literal_from_calibration(g):
    """Literal thresholds of the library's calibrations from what the gate carries (None: not recoverable)."""
    from rescope import float_safe_threshold
    cal, eta = (g.meta or {}).get("calibration") or {}, (g.meta or {}).get("rank_eta")
    op = cal.get("op")
    if eta is None or op not in ("calibrate_two_block", "calibrate_shared"):
        return None
    if op == "calibrate_two_block" and g.det_b is None:                 # ranked by r: the baselines are not stored in the gate
        return None
    b = np.zeros(g.D) if g.det_b is None else g.det_b.detach().cpu().numpy()
    u = np.ones(g.D) if g.det_u is None else g.det_u.detach().cpu().numpy()
    return torch.from_numpy(float_safe_threshold(b, u, float(eta)))


class Scale:
    """The thresholds of a calibrated scoped gate along its common scale (module docstring)."""

    def __init__(self, gate, guards: str = "keep", literal=None):
        from rescope import own_caps
        g = getattr(gate, "obj", gate)
        if not hasattr(g, "det_tau"):
            raise TypeError("Scale: a scoped gate")
        g = G._cpu(g)
        if guards not in ("keep", "drop"):
            raise ValueError("guards: 'keep' | 'drop'")
        th = g.theta()
        self.g, self.guards, self.th, self.cap, self.note = g, guards, th, None, ""
        self.calibrated = bool((g.meta or {}).get("calibration"))
        if guards == "drop" or not self.calibrated:
            self.lit = th
            self.note = "pure shift of the thresholds" + ("" if self.calibrated else " (no calibration record: a gate with fixed thresholds has no guard rule)")
            return
        if g.D != g.T:
            raise NotImplementedError("guards='keep' is defined for one detector per edit (the library's own_caps); use guards='drop'")
        cap = own_caps(g)
        capped = (th == cap) & torch.isfinite(cap)
        lit = th.clone()
        if literal is not None:
            lit = torch.as_tensor(literal, dtype=torch.float64).reshape(-1).clone()
        elif bool(capped.any()):
            rec = _literal_from_calibration(g)
            if rec is None:
                raise ValueError(f"{int(capped.sum())} thresholds of this gate sit at their guard cap and the gate does not carry its literal thresholds "
                                 "(ranking baselines + eta of a library calibration): pass literal=, or guards='drop'")
            lit[capped] = torch.maximum(th[capped], rec[capped])
        if not torch.equal(torch.minimum(lit, cap), th):
            raise ValueError("the thresholds of this gate are not min(literal threshold, guard cap) (calibrated without guards, or changed since): "
                             "use guards='drop', or pass literal=")
        self.lit, self.cap, self.capped0 = lit, cap, capped

    def theta(self, delta: float) -> torch.Tensor:
        t = self.lit + float(delta)
        return t if self.cap is None else torch.minimum(t, self.cap)

    def n_at_cap(self, delta: float) -> int:
        """Number of edits whose threshold at this scale is its guard cap (below the literal level: outside the bound)."""
        if self.cap is None:
            return 0
        return int((((self.lit + float(delta)) > self.cap) & torch.isfinite(self.cap)).sum())

    def gate(self, delta: float):
        """The scoped log at the raised scale (same detectors, values, ranking; thresholds theta(delta), eta = 1)."""
        if float(delta) == 0.0:
            return self.g
        return self.g.replace(det_tau=self.theta(delta), eta=1.0, meta=dict(self.g.meta, raised={"delta": float(delta), "guards": self.guards}))


class GateSweep:
    """Decisions of a scoped gate (role 'last', fp64) on a fixed row set at any scale shift: features once, then comparisons."""

    def __init__(self, L, M, rows, which: str = "tf", guards: str = "keep", kc=None, literal=None):
        g = getattr(M, "obj", M)
        if not G.is_scoped(g):
            raise NotImplementedError("sweeps: a scoped gate that decides at the last prompt position (role 'last', fp64), or an object with .sweep(...)")
        self.scale = Scale(g, guards, literal)
        self.g = self.scale.g
        self.rows, self.which = [int(j) for j in rows], which
        n = len(self.rows)
        self.R = G.features(L, self.g, self.rows, which, kc=kc) if n else torch.zeros(0, self.g.D, dtype=torch.float64)
        self.key = self.g.rank_key(self.R)
        self.own = np.array([L.own(j) for j in self.rows], dtype=np.int64) if kc is None else np.full(n, -1, dtype=np.int64)
        self._o = torch.as_tensor(self.own, dtype=torch.long)
        self._ar = torch.arange(n)

    def at(self, delta: float) -> dict:
        from rescope.scoped import INF
        g, R, o = self.g, self.R, self._o
        th = self.scale.theta(delta)
        cand = R >= th[None, :]                                          # the comparisons of fx.gate._decide_keys
        fired = cand.any(1)
        top = g.det_edit[torch.where(cand, self.key, torch.full_like(R, -INF)).argmax(1)] if R.shape[0] else torch.zeros(0, dtype=torch.long)
        ce = g._edit_any(cand)
        oc = ce[self._ar, o.clamp(min=0)] & (o >= 0)
        return {"fired": fired.numpy(), "top_edit": torch.where(fired, top, torch.full_like(top, -1)).numpy().astype(np.int64), "own": self.own,
                "own_cand": oc.numpy(), "own_top": (fired & (o >= 0) & (top == o)).numpy(), "n_cand": ce.sum(1).numpy().astype(np.int64)}

    def gate(self, delta: float):
        return self.scale.gate(delta)

    def n_at_cap(self, delta: float) -> int:
        return self.scale.n_at_cap(delta)


def open_sweep(L, M, rows, which: str = "tf", guards: str = "keep", kc=None, literal=None):
    """The sweep object of a method on a row set (module docstring: gate interface)."""
    obj = getattr(M, "obj", M)
    if callable(getattr(obj, "sweep", None)):
        return obj.sweep(L, rows, which=which, guards=guards, kc=kc)
    return GateSweep(L, M, rows, which, guards, kc, literal)


def raised_method(L, M, delta: float, guards: str = "keep", label: str | None = None, literal=None):
    """The gate of a calibrated Method at the raised scale as a Method (the behavioural form of a scale read-out: it runs through
    fx.gate / fx.behave / the memo like the method itself; released values are unchanged, so its memo entries are shared)."""
    from . import methods as MT
    L = MT._log(L)
    sw = open_sweep(L, M, [], guards=guards, literal=literal)
    g2 = sw.gate(delta)
    rec = {k: v for k, v in M.rec.items() if k not in ("fingerprint", "thresholds", "own_prompts", "gate", "baselines")}
    rec["raised"] = {"delta": float(delta), "guards": guards, "of": M.key, "of_digest": M.digest, "n_edits_at_cap": int(sw.n_at_cap(delta)) if hasattr(sw, "n_at_cap") else None,
                     "rule": "thresholds = min(literal threshold + delta, guard cap)" if guards == "keep" else "thresholds = thresholds + delta"}
    M2 = MT.Method(M.name, label or f"{M.label}@{float(delta):+.6g}", g2, M.kind, L.name, rec, alpha=M.alpha, n=M.n, draw=M.draw,
                   variant=",".join(v for v in (M.variant, f"raise={float(delta):.6g}/{guards}") if v))
    return MT._finish(M2, L)


# ----------------------------------------------------------------------------- hard negatives: fit / test by parity (H3)
def h3_split(L, hn_rows=None) -> dict:
    """The split of read-out H3: the valid hard negatives of the log by the parity of the STREAM POSITION of the row's own edit --
    the edit whose image the row shows (HN-I, HN-X); an HN-Q row shows a foreign image and belongs to the edit whose question it
    asks -- even = fit, odd = test.  -> {'fit': {kind: rows}, 'test': {kind: rows}, 'even_edits', 'odd_edits' (edit ids of the log
    by stream position), 'n'}.  hn_rows: a subset of the hard negatives to split (default: all valid ones)."""
    from . import masks as MK
    pos = MK.stream_pos(L)
    rows = L.rows("hn", valid=True) if hn_rows is None else [int(j) for j in hn_rows]
    out = {"fit": {k: [] for k in HN}, "test": {k: [] for k in HN}}
    for j in rows:
        e = L.own(j)
        if e < 0:
            raise ValueError(f"{L.sid(j)}: a hard negative without an own edit in this log")
        out["fit" if pos[e] % 2 == 0 else "test"][L.index[j]["hn_type"]].append(j)
    out["even_edits"] = [i for i, p in enumerate(pos) if p % 2 == 0]
    out["odd_edits"] = [i for i, p in enumerate(pos) if p % 2 == 1]
    out["n"] = {s: {k: len(v) for k, v in out[s].items()} for s in ("fit", "test")}
    return out


def _rate(flags) -> float | None:
    flags = np.asarray(flags, dtype=bool)
    return 100.0 * float(flags.mean()) if flags.size else None


# ----------------------------------------------------------------------------- curves along the scale (E1b / E1c for calibrated gates)
def shift_curves(L, M, deltas, which: str = "tf", guards: str = "keep", parts=("probe", "edit", "hn"), pools=(), literal=None) -> dict:
    """Gate-level numbers of a calibrated gate for a grid of common scale shifts `deltas` (0 = the gate itself).
    -> {'deltas', 'curves': {name: (n_deltas,) percent}, 'n': {group: rows}, 'n_edits_at_cap': (n_deltas,), 'guards'}.
    Groups (those of fx.gate.frontier): probe.heldout_disjoint (+ _image, _text, _near, _far), probe.heldout_strict, probe.repair;
    edit.rel / tgen / igen / rephrase (the union of text and image rephrases); hn.HN-I / HN-Q / HN-X / all; and, by the split of
    H3, hn.fit.<kind>, hn.test.<kind>, hn.fit.mean / hn.test.mean (mean over the kinds), edit.odd.rephrase / edit.even.rephrase.
    Curves: '<group>.fired' (an edit is released), and where rows have an own edit '.own_top' (coverage), '.own_cand', '.other_top'.
    pools: names of key-level pools (fx.pools: 'BEN-T', 'MME-T', ...) -> 'pool.<name>.fired'."""
    from . import pools as P
    ds = np.asarray(list(deltas), dtype=np.float64)
    sets = G.row_sets(L, benign=False)
    curves, ns, at_cap = {}, {}, None
    sp = h3_split(L) if "hn" in parts and sets.get("hn") else None

    def put(name, i, val):
        curves.setdefault(name, np.full(ds.size, np.nan))[i] = np.nan if val is None else val

    for p in parts:
        rows = sets.get(p) or []
        if not rows:
            continue
        sw = open_sweep(L, M, rows, which, guards, None, literal)
        groups = {k: m for k, (pp, m) in G._groups(L, {"parts": {p: {"rows": rows}}}).items()}
        if p == "hn":
            pos = {j: i for i, j in enumerate(rows)}
            for side in ("fit", "test"):
                for k in HN:
                    m = np.zeros(len(rows), dtype=bool)
                    m[[pos[j] for j in sp[side][k]]] = True
                    groups[f"hn.{side}.{k}"] = m
        if p == "edit":
            own = np.array([L.own(j) for j in rows])
            reph = groups["edit.rephrase"]
            odd = np.isin(own, _odd(L))
            groups["edit.odd.rephrase"], groups["edit.even.rephrase"] = reph & odd, reph & ~odd
        for k, m in groups.items():
            ns[k] = int(m.sum())
        if at_cap is None and hasattr(sw, "n_at_cap"):
            at_cap = [int(sw.n_at_cap(float(d))) for d in ds]
        for i, d in enumerate(ds):
            dec = sw.at(float(d))
            f, has = dec["fired"], dec["own"] >= 0
            for k, m in groups.items():
                if not m.any():
                    continue
                put(k + ".fired", i, _rate(f[m]))
                if (has & m).any():
                    put(k + ".own_top", i, _rate(dec["own_top"][m]))
                    put(k + ".own_cand", i, _rate(dec["own_cand"][m]))
                    put(k + ".other_top", i, _rate(f[m] & ~dec["own_top"][m]))
            if p == "hn":
                for side in ("fit", "test"):
                    v = [curves[f"hn.{side}.{k}.fired"][i] for k in HN if f"hn.{side}.{k}.fired" in curves]
                    put(f"hn.{side}.mean.fired", i, float(np.mean(v)) if v else None)
    for name in pools:
        Pl = P.get(L, name)
        sw = open_sweep(L, M, Pl.rows, Pl.which, guards, Pl.kc, literal)
        ns[f"pool.{name}"] = len(Pl.rows)
        for i, d in enumerate(ds):
            put(f"pool.{name}.fired", i, _rate(sw.at(float(d))["fired"]))
    return {"log": L.name, "method": getattr(M, "label", None), "which": which, "guards": guards, "deltas": ds, "curves": curves, "n": ns, "n_edits_at_cap": at_cap}


def _odd(L) -> list:
    from . import masks as MK
    return [i for i, p in enumerate(MK.stream_pos(L)) if p % 2 == 1]


# ----------------------------------------------------------------------------- the scale that reaches a coverage of the rephrases
def coverage_scale(L, M, target: float, which: str = "tf", guards: str = "keep", edits=None, kinds=("tgen", "igen"), literal=None) -> dict:
    """The scale shift at which a calibrated gate reaches a given coverage of the union of text and image rephrases (own edit
    released; T-Gen and I-Gen sequences of `edits`, default all, together): the LARGEST shift delta (negative: the scale is
    lowered) with coverage(delta) >= target percent, found by bisection over the floats with the exact gate decision.
    -> {target, reachable, delta, coverage (at delta), coverage_next (one float above: < target), n, n_edits_at_cap, guards}.
    reachable False: even with every threshold at minus infinity the coverage stays below the target (largest_coverage; the
    top-ranked edit of a rephrase is then not its own edit).  delta = +inf: the coverage never falls below the target (the guard
    caps hold it).  Coverage is monotone in the shift when no threshold is capped; with capped edits it can fail to be (another
    edit can out-rank the own edit only while it is a candidate): the value returned is then A boundary of the level set
    (coverage >= target at delta, < target one float above), stated in 'monotone'."""
    ed = list(range(L.T)) if edits is None else [int(i) for i in edits]
    rows = [L.edit_rows[k][i] for k in kinds for i in ed]
    sw = open_sweep(L, M, rows, which, guards, None, literal)
    cov = lambda d: _rate(sw.at(d)["own_top"])                           # noqa: E731
    out = {"target": float(target), "n": len(rows), "kinds": list(kinds), "guards": guards, "coverage_at_0": cov(0.0)}
    c_lo, c_hi = cov(-BIG), cov(BIG)
    if c_lo < target:
        return dict(out, reachable=False, delta=None, largest_coverage=c_lo)
    if c_hi >= target:
        return dict(out, reachable=True, delta=float("inf"), coverage=c_hi, note="the coverage never falls below the target: the guard caps hold it")
    lo, hi = _ord(-BIG), _ord(BIG)                                       # invariant: cov(lo) >= target > cov(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cov(_unord(mid)) >= target:
            lo = mid
        else:
            hi = mid
    d = _unord(lo)
    grid = [cov(x) for x in np.linspace(d - 1.0, d + 1.0, 41)]
    return dict(out, reachable=True, delta=d, coverage=cov(d), coverage_next=cov(_unord(hi)), n_edits_at_cap=int(sw.n_at_cap(d)) if hasattr(sw, "n_at_cap") else None,
                monotone=bool(all(a >= b for a, b in zip(grid, grid[1:]))))


# ----------------------------------------------------------------------------- read-out H3
def h3(L, M, which: str = "tf", limit: float = 10.0, guards: str = "keep", test_rows=None, literal=None) -> dict:
    """Read-out H3 (amendment "Review round 1", R6, with the clarification of 2026-11-11 06:34), gate level.
    The valid hard negatives of the log are split by the parity of the stream position of the row's own edit (h3_split: even =
    fit, odd = test).  The common scale of the calibrated gate M is RAISED by the smallest amount delta >= 0 at which the firing
    rate on the fit rows, averaged over the kinds HN-I, HN-Q, HN-X (those with fit rows), is at most `limit` percent (exact:
    bisection over the floats with the gate's own decision; raise 0 if the gate is already there).  Read at that scale:
      rephrase     coverage (own edit released) of the T-Gen and I-Gen sequences of the ODD edits: tgen, igen, mean = (T-Gen +
                   I-Gen) / 2; own_cand; other_top (another edit is released: a wrong target on a rephrase); rel (own prompts)
      test         per kind on the TEST rows: n, fired (an edit is released = the gate-level wrong-target event), own_top (the
                   row's own edit is released), other_top; HN-X also b_top (the edit whose question the row asks); mean_fired
      heldout_disjoint   false firing on the valid heldout_disjoint probes (n, fired; near / far)
      fit          per kind and mean at raise 0 and at the raise; mean one float below the raise (> limit: the raise is minimal)
      per_edit     {'edits': odd edits, 'tgen': flags, 'igen': flags}: per-edit coverage flags (units of the crossed bootstrap)
      test_rows    {'rows', 'kind', 'own', 'fired', 'top_edit'}: per-row flags of the test rows
      method       the gate at the raised scale as a Method (fx.sweeps.raised_method): THE BEHAVIOURAL FORM -- exact on the odd
                   edits: ses.edits(h['method'], edits=h['odd_edits']); firing, D1 and wrong-target flags on the test rows:
                   ses.hard_negatives(h['method'], rows=h['test_rows']['rows'], generate=True)
      reachable    False if no raise reaches the limit (guard caps hold edits in scope on fit rows): delta None, read-outs at the
                   largest raise are given under 'at_saturation' and the method is None
    test_rows: restrict the TEST read-out to these hard negatives (e.g. the rows whose foreign part occurs in no recombination
    row: clarification (2) of 09:03); the fit rows and the raise are unchanged."""
    sp = h3_split(L)
    kinds = [k for k in HN if sp["fit"][k]]
    if not kinds:
        raise ValueError(f"{L.name}: no hard negatives of edits with an even stream position (H3 is not defined)")
    fit_rows = [j for k in kinds for j in sp["fit"][k]]
    kidx = np.array([i for i, k in enumerate(kinds) for _ in sp["fit"][k]])
    swf = open_sweep(L, M, fit_rows, which, guards, None, literal)

    def fit(d):
        f = swf.at(d)["fired"]
        per = {k: _rate(f[kidx == i]) for i, k in enumerate(kinds)}
        return float(np.mean(list(per.values()))), per

    f0, per0 = fit(0.0)
    out = {"log": L.name, "method_label": getattr(M, "label", None), "which": which, "limit": float(limit), "guards": guards, "kinds_fit": kinds,
           "n_fit": {k: len(sp["fit"][k]) for k in HN}, "n_test": {k: len(sp["test"][k]) for k in HN}, "odd_edits": sp["odd_edits"], "even_edits": sp["even_edits"],
           "fit": {"at_0": {"mean": f0, **per0}}}
    if f0 <= limit:
        delta = 0.0
    else:
        fb, perb = fit(BIG)
        if fb > limit:
            out.update(reachable=False, delta=None, method=None, at_saturation={"mean": fb, **perb, "n_edits_at_cap": int(swf.n_at_cap(BIG)) if hasattr(swf, "n_at_cap") else None})
            return out
        lo, hi = _ord(0.0), _ord(BIG)                                    # invariant: fit(lo) > limit >= fit(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fit(_unord(mid))[0] <= limit:
                hi = mid
            else:
                lo = mid
        delta = _unord(hi)
        out["fit"]["one_float_below"] = {"mean": fit(_unord(lo))[0]}
    fd, perd = fit(delta)
    out["fit"]["at_raise"] = {"mean": fd, **perd}
    out.update(reachable=True, delta=float(delta), n_edits_at_cap=int(swf.n_at_cap(delta)) if hasattr(swf, "n_at_cap") else None,
               n_edits_at_cap_at_0=int(swf.n_at_cap(0.0)) if hasattr(swf, "n_at_cap") else None)
    # ---- rephrases and own prompts of the odd edits
    odd = sp["odd_edits"]
    cov = {}
    for k in ("tgen", "igen", "rel"):
        d = open_sweep(L, M, [L.edit_rows[k][i] for i in odd], which, guards, None, literal).at(delta)
        cov[k] = {"n": len(odd), "own_top": _rate(d["own_top"]), "own_cand": _rate(d["own_cand"]), "other_top": _rate(d["fired"] & ~d["own_top"]), "flags": d["own_top"]}
    out["rephrase"] = {"tgen": cov["tgen"]["own_top"], "igen": cov["igen"]["own_top"], "mean": 0.5 * (cov["tgen"]["own_top"] + cov["igen"]["own_top"]) if odd else None,
                       "own_cand": {k: cov[k]["own_cand"] for k in ("tgen", "igen")}, "other_top": {k: cov[k]["other_top"] for k in ("tgen", "igen")},
                       "rel": cov["rel"]["own_top"], "n_edits": len(odd)}
    out["per_edit"] = {"edits": odd, "tgen": cov["tgen"]["flags"], "igen": cov["igen"]["flags"], "rel": cov["rel"]["flags"]}
    # ---- test hard negatives
    keep = None if test_rows is None else {int(j) for j in test_rows}
    trow = [j for k in HN for j in sp["test"][k] if keep is None or j in keep]
    tk = np.array([L.index[j]["hn_type"] for j in trow])
    d = open_sweep(L, M, trow, which, guards, None, literal).at(delta)
    test = {}
    for k in HN:
        m = tk == k
        if not m.any():
            continue
        t = {"n": int(m.sum()), "fired": _rate(d["fired"][m]), "own_top": _rate(d["own_top"][m]), "other_top": _rate(d["fired"][m] & ~d["own_top"][m])}
        if k == "HN-X":
            eb = np.array([L._own.get(L.index[j].get("edit_b_uid"), -1) for j, mm in zip(trow, m) if mm], dtype=np.int64)
            t["b_top"] = _rate(d["fired"][m] & (d["top_edit"][m] == eb))
        test[k] = t
    test["mean_fired"] = float(np.mean([test[k]["fired"] for k in HN if k in test])) if any(k in test for k in HN) else None
    test["subset"] = None if keep is None else {"n_rows": len(trow), "of": sum(len(sp["test"][k]) for k in HN)}
    out["test"] = test
    out["test_rows"] = {"rows": trow, "kind": tk.tolist(), "own": d["own"], "fired": d["fired"], "top_edit": d["top_edit"], "own_top": d["own_top"]}
    # ---- far locality at the raised scale
    hd = L.pool("heldout_disjoint")
    dd = open_sweep(L, M, hd, which, guards, None, literal).at(delta)
    near = np.array([L.near_edit(j) is not None for j in hd])
    out["heldout_disjoint"] = {"n": len(hd), "fired": _rate(dd["fired"]), "near": {"n": int(near.sum()), "fired": _rate(dd["fired"][near])},
                               "far": {"n": int((~near).sum()), "fired": _rate(dd["fired"][~near])}}
    out["method"] = raised_method(L, M, delta, guards, label=f"{getattr(M, 'label', 'gate')}@h3", literal=literal) if hasattr(M, "rec") else None
    return out
