"""fx.gate: gate-level evaluation from the key cache, on the CPU (no model).

    dec = fx.gate.decide(L, M, rows, which="tf")        per-row decision of a method on sequences of the key cache
    R   = fx.gate.features(L, M, rows, which="tf")      (n, D) fp64 feature matrix of the gate's detectors (for simulations over many draws)
    fr  = fx.gate.frontier(L, "ncoord" | "cos")          E1b: one global threshold swept over a grid (firing, coverage, near-scope firing)
    mc  = fx.gate.matched_coverage(L, feature, (25, 50, 75))   E1c: the threshold reaching a coverage of rephrases + the numbers there
    ev  = fx.gate.evaluate(L, M, positions=("tf",))     the standard row sets of a log: probes, edits, hard negatives, benign pool
    st  = fx.metrics.gate_stats(L, ev)                  the numbers (firing rates, coverage, lineage widths)

METHODS.  Scoped gates that decide at the last prompt position (rescope.ScopedLog with role 'last' and the fp64 gate: ReScope,
cosine / ball / exact gates) and the router (needs the model's weights: build it with a handle).  Roles 'last+answer' / 'all', the
z-score gate, weights and DUAL need every position or the model: use fx.behave.

POSITIONS (which)   'tf'      the teacher-forced key (what the teacher-forced evaluation presents to the gate)
                    'prompt'  the teacher-forced sequence at the decode's token (differs from 'tf' only where lp_shift != 0: LLaVA)
                    'decode'  the prompt-only key = the first forward of a decode (the only key of benign rows without an answer)
                    'steps'   the 11 full forwards of a decode: per-row OR, plus the per-step decisions (step_fired, step_top)
DECISION OF A ROW (arrays aligned with `rows`)
    fired      some detector is a candidate            top_edit   the edit that releases under fire 'top1' (-1: nothing fires)
    n_cand     number of candidate EDITS (lineage width of the query)      margin   largest r_j - threshold_j (>= 0 <=> fired)
    own        the row's own edit in this log (-1: none); own_cand: it is a candidate; own_top: it is the edit that releases
    near       a coordinate lies within 1e-9 (relative) of its threshold: only there may the GPU hook decide otherwise
  (fire 'all', an ablation: every candidate edit releases, so its coverage is own_cand, not own_top; top_edit is the first-ranked candidate)
The arithmetic is rescope's: ScopedLog.scores in chunks of 1024 keys (as KeyCache.decide), thresholds / ranking / decision band by
the library's own functions; test t_cpu.gate_equals_library compares every array with KeyCache.decide.  The feature matrix of a
log's sequences is computed once per detector set and reused by every calibration draw (T = 2000: seconds instead of minutes).
"""
from __future__ import annotations

import os

import numpy as np
import torch

CHUNK = 1024
SLACK = 1e-9
N_STEPS = 11                     # full forwards of a decoder-v2 decode that contain the last prompt position (10 no-cache + 1)
_FILE = {"tf": "tf", "prompt": "tf", "decode": "decode", "steps": "steps"}
_FEAT = {}                       # (log name, detector ids, file) -> {"keep": tensors, "pos": {key row: position}, "R": (n, D) fp64, "kn": (n,) fp64}
_FEAT_MAX = 6


def is_scoped(obj) -> bool:
    """A gate whose decision the key cache determines: ScopedLog, role 'last', fp64."""
    from rescope import ScopedLog
    return isinstance(obj, ScopedLog) and obj.role == "last" and obj.precision == "fp64"


def is_router(obj) -> bool:
    return getattr(obj, "kind", None) == "router"


def _cpu(g):
    return g if g.device.type == "cpu" else g.to("cpu")


def _features(L, g, file: str, krows, kc=None):
    """(R (n, D) fp64, key norms (n,)) of the gate's detectors on rows `krows` of a key file; cached per detector set.
    kc: another key cache than the log's (a key-level pool: fx.pools); None = L.kc (the present behaviour)."""
    ids = (L.name, id(g.det_w), id(g.det_s), id(g.det_kind), id(g.det_key), file)
    if kc is not None:
        ids = ids + (os.path.realpath(kc.dir),)
    ent = _FEAT.get(ids)
    if ent is None:
        if len(_FEAT) >= _FEAT_MAX:
            _FEAT.pop(next(iter(_FEAT)))
        ent = _FEAT[ids] = {"keep": (g.det_w, g.det_s, g.det_kind, g.det_key, kc), "pos": {}, "R": torch.zeros(0, g.D, dtype=torch.float64),
                            "kn": torch.zeros(0, dtype=torch.float64)}
    new = sorted({int(r) for r in krows} - ent["pos"].keys())
    if new:
        K = (L.kc if kc is None else kc).K(file)
        Rs, kn = [], []
        for a in range(0, len(new), CHUNK):
            k = K[torch.tensor(new[a:a + CHUNK], dtype=torch.long)]
            Rs.append(g.scores(k))
            kn.append(k.double().norm(dim=1))
        base = ent["R"].shape[0]
        ent["R"], ent["kn"] = torch.cat([ent["R"]] + Rs), torch.cat([ent["kn"]] + kn)
        ent["pos"].update({r: base + i for i, r in enumerate(new)})
    p = torch.tensor([ent["pos"][int(r)] for r in krows], dtype=torch.long)
    return ent["R"][p], ent["kn"][p]


def clear_cache():
    _FEAT.clear()


def features(L, M, rows, which: str = "tf", kc=None) -> torch.Tensor:
    """(n, D) fp64: the feature of every detector of a scoped gate (normalised coordinate, cosine, minus distance) at the ONE key that
    each sequence presents at position `which` ('tf' | 'prompt' | 'decode').  It depends on the detectors only, not on thresholds:
    simulations over many calibration draws (E5: conformal validity) take it once and apply the library's rules on matrices
    (rescope.two_block_rule / shared_rule(coords[idx], alpha) -> tau; a row fires iff (features >= tau).any(1)).  Same arithmetic
    and cache as decide().  kc: the key cache `rows` are positions of (a key-level pool, fx.pools); None = the log's."""
    g = getattr(M, "obj", M)
    if not is_scoped(g):
        raise NotImplementedError("features: a scoped gate with role 'last' and the fp64 gate")
    if which not in ("tf", "prompt", "decode"):
        raise ValueError("which: 'tf' | 'prompt' | 'decode'")
    rows = [int(j) for j in rows]
    return _features(L, _cpu(g), _FILE[which], (L.kc if kc is None else kc).key_rows(which, rows), kc)[0]


def _decide_keys(L, g, file: str, krows, own, slack: float = SLACK, kc=None, top1: bool = False) -> dict:
    """Decision at ONE key per entry (the arithmetic of KeyCache.decide).  top1=True adds the read-out of the OTHER order of
    selection, "argmax first, threshold second" (alt_fired, alt_top, top1_differs: module docstring of fx.metrics.top1_report)."""
    from rescope.keycache import near_threshold
    from rescope.scoped import INF
    n = len(krows)
    if n == 0:
        z = np.zeros(0, dtype=bool)
        out = {"fired": z, "near": z.copy(), "n_cand": np.zeros(0, dtype=np.int64), "top_edit": np.zeros(0, dtype=np.int64), "margin": np.zeros(0),
               "own_cand": z.copy(), "own_top": z.copy()}
        if top1:
            out.update(alt_fired=z.copy(), alt_top=np.zeros(0, dtype=np.int64), top1_differs=z.copy())
        return out
    R, kn = _features(L, g, file, krows, kc)
    th = g.theta()
    cand = R >= th[None, :]
    fired = cand.any(1)
    d = R - th[None, :]
    margin = torch.where(torch.isnan(d), torch.full_like(d, -INF), d).max(1).values
    near = near_threshold(g, kn[:, None], R, slack)                      # the band only needs |k|: a (n, 1) surrogate has that norm
    top = g.det_edit[torch.where(cand, g.rank_key(R), torch.full_like(R, -INF)).argmax(1)]
    ce = g._edit_any(cand)
    o = torch.as_tensor(np.asarray(own), dtype=torch.long)
    oc = ce[torch.arange(n), o.clamp(min=0)] & (o >= 0)
    out = {"fired": fired.numpy(), "near": near.numpy(), "n_cand": ce.sum(1).numpy().astype(np.int64),
           "top_edit": torch.where(fired, top, torch.full_like(top, -1)).numpy().astype(np.int64), "margin": margin.numpy(),
           "own_cand": oc.numpy(), "own_top": (fired & (o >= 0) & (top == o)).numpy()}
    if top1:                                                            # "argmax first, threshold second": the first-ranked detector of ALL, if it is a candidate
        a = g.rank_key(R).argmax(1)
        af = cand[torch.arange(n), a]
        at = torch.where(af, g.det_edit[a], torch.full_like(top, -1))
        out.update(alt_fired=af.numpy(), alt_top=at.numpy().astype(np.int64),
                   top1_differs=((af != fired) | (at != torch.where(fired, top, torch.full_like(top, -1)))).numpy())
    return out


def decide(L, M, rows, which: str = "tf", slack: float = SLACK, kc=None, top1: bool = False) -> dict:
    """Decision of method M (a fx.methods.Method, or a gate object) on the sequences `rows` (positions in the key-cache index).
    -> dict of arrays aligned with `rows` (module docstring) + 'rows', 'which'.
    kc: `rows` are positions in ANOTHER key cache (a key-level pool of fx.pools, e.g. the MME capture): no row has an own edit.
    top1=True: + alt_fired, alt_top, top1_differs = what "argmax first, threshold second" would decide and whether it differs
    from the library's rule "candidates after the caps first, the best candidate second" (scoped gates; 'steps': any step)."""
    g = getattr(M, "obj", M)
    rows = [int(j) for j in rows]
    if callable(getattr(g, "decide_rows", None)):                       # stage 2 (fx.review1): a reference system decides from the query's attributes
        return g.decide_rows(L, rows, which, kc=kc, top1=top1)
    ix = L.index if kc is None else kc.index
    own = np.array([L.own(j) for j in rows], dtype=np.int64) if kc is None else np.full(len(rows), -1, dtype=np.int64)
    if is_router(g):
        d = (L.kc if kc is None else kc).decide(g, which, rows=rows, slack=slack)
        n = len(rows)
        return {"rows": rows, "which": which, "fired": np.asarray(d["fired"]), "near": np.asarray(d["near"]), "margin": np.asarray(d["margin"]),
                "n_cand": np.zeros(n, dtype=np.int64), "top_edit": np.full(n, -1, dtype=np.int64), "own": own,
                "own_cand": np.zeros(n, dtype=bool), "own_top": np.zeros(n, dtype=bool), "resolves_edits": False}
    if not is_scoped(g):
        raise NotImplementedError(f"{getattr(M, 'name', type(g).__name__)}: only gates that decide at the last prompt position (scoped logs with role "
                                  "'last' and the fp64 gate; the router) have a gate-level evaluation from the key cache; use fx.behave")
    g = _cpu(g)
    if which in ("tf", "prompt", "decode"):
        krows = (L.kc if kc is None else kc).key_rows(which, rows)
        out = _decide_keys(L, g, _FILE[which], krows, own, slack, kc, top1)
    elif which == "steps":
        st = np.array([ix[j]["rows_decode_steps"] for j in rows], dtype=np.int64).reshape(len(rows), -1 if rows else N_STEPS)
        k = st.shape[1]
        flat = _decide_keys(L, g, "steps", st.reshape(-1).tolist(), np.repeat(own, k), slack, kc, top1)
        f = flat["fired"].reshape(len(rows), k)
        first = np.where(f.any(1), f.argmax(1), 0)
        ar = np.arange(len(rows))
        out = {"fired": f.any(1), "near": flat["near"].reshape(len(rows), k).any(1), "n_cand": flat["n_cand"].reshape(len(rows), k).max(1),
               "top_edit": np.where(f.any(1), flat["top_edit"].reshape(len(rows), k)[ar, first], -1), "margin": flat["margin"].reshape(len(rows), k).max(1),
               "own_cand": flat["own_cand"].reshape(len(rows), k).any(1), "own_top": flat["own_top"].reshape(len(rows), k)[ar, first] & f.any(1),
               "step_fired": f, "step_top": flat["top_edit"].reshape(len(rows), k), "step_near": flat["near"].reshape(len(rows), k)}
        if top1:
            out.update(alt_fired=flat["alt_fired"].reshape(len(rows), k).any(1), top1_differs=flat["top1_differs"].reshape(len(rows), k).any(1),
                       alt_top=np.where(flat["alt_fired"].reshape(len(rows), k).any(1),
                                        flat["alt_top"].reshape(len(rows), k)[ar, np.where(flat["alt_fired"].reshape(len(rows), k).any(1), flat["alt_fired"].reshape(len(rows), k).argmax(1), 0)], -1))
    else:
        raise ValueError("which: 'tf' | 'prompt' | 'decode' | 'steps'")
    out.update(rows=rows, which=which, own=own, resolves_edits=True)
    return out


# ----------------------------------------------------------------------------- one global threshold on a feature (E1b, E1c)
def sweep(L, feature: str = "ncoord", which: str = "tf", parts=("probe", "edit", "hn", "benign"), gate=None) -> dict:
    """What a gate with ONE global threshold t on a feature needs to know of every sequence (candidate iff feature >= t, top-1 by the
    feature, no guards, no caps): rmax = the largest feature over the detectors, amax = its edit, rown = the feature of the row's
    own edit.  feature: 'ncoord' (normalised solver coordinate) | 'coord' | 'cos' (raw cosine to the stored key) | 'negdist'.
    -> the input of frontier / readout / matched_coverage (amendment of 2026-11-11 00:55: frontier tracing and the matched-coverage
    read-out; gate level, key cache, no model).
    gate: ANY scoped gate (role 'last', fp64) whose detectors define the feature, e.g. a gate of a feature package (its
    thresholds are ignored); `feature` is then only the label of the sweep.  None: fx.methods.base_gate(L, feature)."""
    from . import methods as MT
    if gate is not None:
        S = getattr(gate, "obj", gate)
        if not is_scoped(S):
            raise NotImplementedError("sweep(gate=): a scoped gate with role 'last' and the fp64 gate")
        S = _cpu(S)
    else:
        S = MT.base_gate(L, feature)[0]                                 # un-thresholded; shares the feature cache with the calibrated gates
    sets, out = row_sets(L), {}
    for p in parts:
        rows = sets.get(p) or []
        if not rows:
            continue
        w = "decode" if p == "benign" else which
        R = features(L, S, rows, w)
        rmax, amax = R.max(1)
        own = np.array([L.own(j) for j in rows], dtype=np.int64)
        rown = R[torch.arange(len(rows)), torch.as_tensor(own).clamp(min=0)].numpy()
        out[p] = {"rows": rows, "which": w, "rmax": rmax.numpy(), "amax": S.det_edit[amax].numpy(), "own": own, "rown": np.where(own >= 0, rown, -np.inf)}
    return {"log": L.name, "feature": feature, "which": which, "parts": out, "gate": S}


def readout(L, sw: dict, t: float) -> dict:
    """The gate-level numbers (fx.metrics.gate_stats: firing per pool, coverage, hard negatives, benign pool, lineage) of the gate
    'candidate iff feature >= t, top-1 by the feature' at ONE threshold t, from sweep()."""
    from . import metrics
    ev = {}
    for p, d in sw["parts"].items():
        fired = d["rmax"] >= t
        has = d["own"] >= 0
        R = features(L, sw["gate"], d["rows"], d["which"])
        ev[p] = {d["which"]: {"rows": d["rows"], "which": d["which"], "fired": fired, "top_edit": np.where(fired, d["amax"], -1), "own": d["own"],
                              "own_top": fired & has & (d["amax"] == d["own"]), "own_cand": has & (d["rown"] >= t), "n_cand": (R >= t).sum(1).numpy()}}
    return metrics.gate_stats(L, ev, sw["which"])


def _groups(L, sw: dict) -> dict:
    """Row groups of the frontier: name -> (part, boolean mask)."""
    g = {}
    for p, d in sw["parts"].items():
        inf = info(L, d["rows"])
        m = lambda f: np.array([bool(f(r)) for r in inf])              # noqa: E731
        if p == "probe":
            dj = m(lambda r: r["disjoint"])
            g.update({"probe.heldout_disjoint": (p, dj), "probe.heldout_strict": (p, m(lambda r: r["strict"])), "probe.repair": (p, m(lambda r: r["pool"] == "repair")),
                      "probe.heldout_disjoint_image": (p, dj & m(lambda r: r["ptype"] == "image")), "probe.heldout_disjoint_text": (p, dj & m(lambda r: r["ptype"] == "text")),
                      "probe.heldout_disjoint_near": (p, dj & m(lambda r: r["near_edit"] is not None)), "probe.heldout_disjoint_far": (p, dj & m(lambda r: r["near_edit"] is None))})
        elif p == "edit":
            for k in ("rel", "tgen", "igen"):
                g[f"edit.{k}"] = (p, m(lambda r: r["kind"] == k))
            g["edit.rephrase"] = (p, m(lambda r: r["kind"] in ("tgen", "igen")))          # the union of text and image rephrases
        elif p == "hn":
            for k in ("HN-I", "HN-Q", "HN-X"):
                g[f"hn.{k}"] = (p, m(lambda r: r["hn_type"] == k))
            g["hn.all"] = (p, np.ones(len(inf), dtype=bool))
        elif p == "benign":
            g.update({"benign": (p, np.ones(len(inf), dtype=bool)), "benign.near": (p, m(lambda r: r["near_edit"] is not None)),
                      "benign.far": (p, m(lambda r: r["near_edit"] is None))})
    return g


def frontier(L, feature: str = "ncoord", thresholds=None, n_grid: int = 501, which: str = "tf", sw: dict | None = None) -> dict:
    """E1b, frontier tracing: one global threshold on a feature swept over a grid.  -> {'thresholds' (n,), 'curves': {name: (n,) percent},
    'n': {group: rows}}: '<group>.fired' = % of the group's rows on which an edit fires, and for groups with an own edit
    '<group>.own_top' (the own edit releases = coverage), '.own_cand', '.other_top'.  Groups: probe.heldout_disjoint (+ _image, _text,
    _near, _far), probe.heldout_strict, probe.repair, edit.rel / tgen / igen / rephrase (text and image rephrases together),
    hn.HN-I / HN-Q / HN-X / all, benign (+ .near, .far; prompt-only key).  thresholds: default = n_grid evenly spaced values
    between the smallest and the largest rmax of the log's sequences."""
    sw = sweep(L, feature, which) if sw is None else sw
    if thresholds is None:
        lo = min(float(d["rmax"].min()) for d in sw["parts"].values())
        hi = max(float(d["rmax"].max()) for d in sw["parts"].values())
        thresholds = np.linspace(lo, hi, int(n_grid))
    ts = np.asarray(thresholds, dtype=np.float64)

    def ge(v):                                                           # number of values >= t, for every t
        v = np.sort(v)
        return v.size - np.searchsorted(v, ts, side="left")
    curves, ns = {}, {}
    for name, (p, mask) in _groups(L, sw).items():
        d = sw["parts"][p]
        n = int(mask.sum())
        ns[name] = n
        if n == 0:
            continue
        curves[name + ".fired"] = 100.0 * ge(d["rmax"][mask]) / n
        has = mask & (d["own"] >= 0)
        if has.any():
            top = has & (d["amax"] == d["own"])
            curves[name + ".own_top"] = 100.0 * ge(d["rmax"][top]) / n
            curves[name + ".own_cand"] = 100.0 * ge(d["rown"][has]) / n
            curves[name + ".other_top"] = curves[name + ".fired"] - curves[name + ".own_top"]
    return {"log": L.name, "feature": feature, "which": which, "thresholds": ts, "curves": curves, "n": ns}


def matched_coverage(L, feature: str = "ncoord", targets=(25, 50, 75), which: str = "tf", sw: dict | None = None) -> list:
    """E1c, matched-coverage read-out: for each target c (percent) the LARGEST global threshold of the feature at which the coverage
    of the union of text and image rephrases (own edit releases; T-Gen and I-Gen sequences together) is at least c -- exact, from the
    order statistics, not from a grid -- and the gate-level numbers at that threshold (readout).  -> [{target, reachable, threshold,
    coverage, readout}] ('where reachable': a target above the coverage at threshold -inf is reported as not reachable)."""
    sw = sweep(L, feature, which) if sw is None else sw
    d = sw["parts"]["edit"]
    sel = np.array([L.index[j]["kind"] in ("tgen", "igen") for j in d["rows"]])
    top = sel & (d["amax"] == d["own"])
    v = np.sort(d["rmax"][top])[::-1]                                   # descending: coverage at threshold v[k - 1] is at least k / N
    N = int(sel.sum())
    out = []
    for c in targets:
        k = int(np.ceil(c / 100.0 * N - 1e-9))
        if k < 1 or k > v.size:
            out.append({"target": c, "reachable": False, "largest_coverage": 100.0 * v.size / N})
            continue
        t = float(v[k - 1])
        out.append({"target": c, "reachable": True, "threshold": t, "coverage": 100.0 * float((d["rmax"][top] >= t).sum()) / N, "readout": readout(L, sw, t)})
    return out


def library_decide(L, M, rows, which: str = "tf") -> dict:
    """The same decision by rescope.KeyCache.decide (reference; slower: the features are recomputed)."""
    g = getattr(M, "obj", M)
    return L.kc.decide(g, which, rows=[int(j) for j in rows], uids=L.uids)


def row_sets(L, valid: bool = True, benign: bool = True) -> dict:
    """The standard row sets of a log for the gate-level evaluation: 'probe' (valid probe rows of every pool), 'edit' (Rel, T-Gen,
    I-Gen of every edit), 'hn' (valid hard negatives of the log's edits), 'benign' (the label-free benign pool; prompt-only keys)."""
    out = {"probe": L.rows("probe", valid=True if valid else None), "edit": L.rows("edit"), "hn": L.rows("hn", valid=True if valid else None)}
    if benign:
        out["benign"] = L.rows("benign")
    return out


def evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn", "benign"), rows: dict | None = None, top1: bool = False) -> dict:
    """Gate-level pass of one method over the row sets of a log -> {part: {which: decision}}.
    positions: any of 'tf', 'prompt', 'decode', 'steps' ('prompt' / 'decode' are only stored where asked for; the benign pool has
    no teacher-forced sequence for most rows and is always evaluated at 'decode').
    top1=True: every decision also carries top1_differs (see decide); fx.metrics.gate_stats then reports, per part, the number of
    sequences on which "argmax first, threshold second" would decide otherwise (block 'top1')."""
    sets = row_sets(L) if rows is None else rows
    out = {}
    for p in parts:
        if p not in sets:
            continue
        if p == "benign":
            out[p] = {"decode": decide(L, M, sets[p], "decode", top1=top1)}
            continue
        out[p] = {w: decide(L, M, sets[p], w, top1=top1) for w in positions}
    return out


def info(L, rows) -> list:
    """Static per-row records (what the statistics group by): kind, edit, pool flags, near-scope flag, hard-negative fields."""
    out = []
    for j in rows:
        e = L.index[j]
        r = {"sid": e["sid"], "lib_sid": e.get("lib_sid"), "kind": e["kind"], "own": L.own(j), "lp_shift": e.get("lp_shift")}
        if e["kind"] == "probe":
            r.update(ptype=e["ptype"], pool=e["pool"], disjoint=bool(e.get("disjoint")), strict=bool(e.get("strict")), uid=e["uid"],
                     y0_valid=bool(e["y0_valid"]), near_edit=L.near_edit(j), near_reph=L.near_edit(j, True))
        elif e["kind"].startswith("hn"):
            b = e.get("edit_b")
            r.update(hn_type=e["hn_type"], edit_b=None if b is None else int(b), group=e.get("hn_group"), y0_valid=bool(e["y0_valid"]),
                     new=bool(e.get("new")), related_flag=bool(e.get("related_flag")), same_category=bool(e.get("same_category")))
        elif e["kind"] == "benign":
            r.update(ptype=e["ptype"], uid=e.get("uid"), near_edit=L.near_edit(j), near_reph=L.near_edit(j, True))
        out.append(r)
    return out
