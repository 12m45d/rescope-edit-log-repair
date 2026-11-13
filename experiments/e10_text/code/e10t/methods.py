"""Methods for the public text logs: one constructor per row -> Method (a library object + the record of what determined it).

    rows = e10t.methods.rowset(L, alpha=0.05, draw=0)       # the fixed rows of E10 that are defined on these logs (ROWSET)
    M = e10t.methods.build("rescope", L, draw=3)

ROWS (ROWSET; mirror of the driver's fx.methods.e1_rowset, role set = ALL positions instead of the last prompt position)
  deployed          the deployed log: module weight W_E
  preedit           the pre-edit model: module weight W_0 (control)
  rescope           THE DEFAULT METHOD on a text log: detector = the logged right factor normalised to 1 at the solver key (the
                    library's 'ncoord' feature: <W_i, k> / <W_i, k_i>, k_i = k_request), role 'all', top-1 per position, response
                    REPL with rebased values, guards = the stored solver keys + the key of every request AS SERVED (its own
                    request prompt at the solver position, from the key cache), two-block conformal calibration on the n = 200
                    benign prompts of draw d of the REPAIR pool at alpha = 0.05; the score of a prompt is its maximum over the
                    positions of the prompt and of the pre-edit 10-token continuation.  Guards hold AS SERVED: a threshold whose
                    guard lies inside the decision band is lowered until the guard is clear of it (guard_margin; switch
                    served_guards=False = the library's thresholds exactly)
  rescope_shared    the same with the shared-scale conformal rule (one threshold on the normalised coordinate)
  rescope_null      the registered null-repair variant: the detectors are first cleared (library null_repair, ridge 0) of the keys
                    at ALL positions (prompt + pre-edit continuation: the role set) of n_a = 100 further benign REPAIR prompts that
                    share no text with the calibration draw; then the default calibration runs unchanged.  n_a = 0 is the default
                    method bit for bit (test c_null0 / g_null0)
  cosine            cosine gate (top-1 raw cosine to the stored solver keys, all positions, REPL, rebased values, same guards)
                    with the same two-block calibration
  cosine_published  cosine gate at 0.65 at all positions (the published form of NeuralDB), rebased values, no guards
  exact             exact-match gate: cosine >= 0.999 at all positions (library exact_gate), rebased values, no guards
FURTHER CONSTRUCTORS (defined on text logs; not rows of ROWSET; for the additions R14 of the protocol and for the tests)
  where             ReScope-W: role 'all', fire 'all', response LIN = the original log terms <W_i, k> U_i (logged values, no
                    round-off term); calibration 'none' (all thresholds -inf: the deployed log up to round-off), 'two_block'
                    (the default calibration), 'logonly' (tau_i = 5 % above the largest coordinate of edit i over every prompt
                    position of the OTHER requests' prompts and the non-solver positions of its own request prompt).  Registered
                    for E10 by amendment R14: rows where_logonly and where_twoblock (ROWSET_R14)
  cosine(calibration='shared'), rescope(response=..., fire=..., rebase=False, allow_silence=True)      switches as in the driver
  calibrate(L, S, rule, alpha, n, draw)   the same calibration for ANY scoped log with role 'all' (-> wrap(L, S2, label))
  weight(L, W, label)                     any replacement weight (6400, 1600), e.g. e10t.data.replay_weight(name, T) for a rollback
Rows of the driver's E1 row set that are undefined / not registered / not built here: e10t.methods.UNDEFINED (label -> reason).

Every gate is built on the CPU from the log's files and the key cache (the same bits in every process); released values are
the library's rebased values computed update by update (the driver's formula and order: logs with the same first requests
share their bits).  All arithmetic is the library's (feature_gate, add_guards, calibrate_two_block, calibrate_shared, null_repair,
cosine_gate, exact_gate, keep_guards).
"""
from __future__ import annotations

import hashlib
import json
import time
import types
from dataclasses import dataclass, field

import numpy as np
import torch

import e10t
from . import data as D, gate as G

SELF_MARGIN = 0.05
_BASE, _VALUES, _BASE_MAX = {}, {}, 6


@dataclass
class Method:
    name: str
    label: str
    obj: object                 # None (deployed) | 'W0' (pre-edit) | rescope.ScopedLog | tensor (a replacement weight)
    kind: str                   # 'deployed' | 'preedit' | 'gate' | 'weight'
    log: str
    rec: dict = field(default_factory=dict)
    alpha: float | None = None
    n: int | None = None
    draw: int | None = None
    variant: str = ""

    @property
    def key(self) -> str:
        return self.rec.get("fingerprint", "")

    def __repr__(self):
        return f"Method({self.label!r}, {self.kind}, log={self.log}, alpha={self.alpha}, n={self.n}, draw={self.draw}, key={self.key})"


def _clean(x):
    import fx
    return fx.io.clean(x)


def _quant(v) -> dict:
    v = np.asarray(v, dtype=np.float64).ravel()
    f = v[np.isfinite(v)]
    return {"n": int(v.size), "n_finite": int(f.size), "n_pos_inf": int(np.isposinf(v).sum()), "n_neg_inf": int(np.isneginf(v).sum()),
            "min": float(f.min()) if f.size else None, "median": float(np.median(f)) if f.size else None, "max": float(f.max()) if f.size else None}


def gate_sig(S) -> str:
    """Identity of a gate as a served model: detectors, thresholds, ranking, values, log terms, role / fire / response."""
    h = hashlib.sha1(f"{G.fkey(S)}|{S.role}|{S.fire}|{S.response}|{float(S.eta)!r}|{S.precision}".encode())
    for t in (S.det_tau, S.det_edit, S.det_b, S.det_u, S.values) + ((S.lin_w, S.lin_u) if S.response == "lin" else ()):
        h.update(b"-" if t is None else np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def _finish(M: Method, L) -> Method:
    g = M.obj
    if hasattr(g, "det_tau"):
        th = g.theta().detach().cpu().numpy()
        M.rec["thresholds"] = dict(_quant(th), sha1=hashlib.sha1(np.ascontiguousarray(th).tobytes()).hexdigest()[:12])
        if g.det_b is not None:
            M.rec["baselines"] = _quant(g.det_b.detach().cpu().numpy())
        M.rec["gate"] = {"role": g.role, "fire": g.fire, "response": g.response, "n_detectors": int(g.D), "n_edits": int(g.T),
                         "n_guards": int(g.guard_key.shape[0]), "values": g.meta.get("values"), "feature": g.meta.get("feature", g.meta.get("baseline"))}
        rq = L.rows["request"]
        d = G.decide(L, g, [r["text_id"] for r in rq], [r["pos"] for r in rq])
        M.rec["own_requests"] = {"n": L.T, "n_not_candidate": int((~d["own_cand"]).sum()), "n_not_released": int((~d["own_released"]).sum()),
                                 "n_other_released": int(d["other_released"].sum()), "n_in_band": int(d["near"].sum()),
                                 "not_candidate_edits": [int(i) for i in np.nonzero(~d["own_cand"])[0][:50]]}
        M.rec["gate_sig"] = gate_sig(g)
    M.rec["fingerprint"] = hashlib.sha1(json.dumps(_clean({k: v for k, v in M.rec.items() if k not in ("fingerprint", "seconds")}), sort_keys=True).encode()).hexdigest()[:12]
    return M


def released_values(L, rebase: bool = True):
    """(T, m) fp64 values a scoped gate releases + record.  rebase=True: the library's rebased values
        V_i = U_i + sum over the earlier updates j < i of U_j <W_j, k_i>          (rescope.rebase; log quantities only)
    (one request per update: every earlier update leaks at the solver key), computed UPDATE BY UPDATE exactly as the driver's
    fx.methods.released_values does (same formula, same order: the same bits; test c_rebase), so that logs with the same first
    requests share the bits of their common values.  The deviation from rescope.rebase (one product over the whole log) is
    recorded; the driver refuses a deviation above 1e-12, here a deviation above 1e-9 is a WARNING and a record field (a long
    log with very large updates sums terms far larger than the result), and only a non-finite value is an error.
    rebase=False: the logged U_i."""
    from rescope import rebase as lib_rebase
    log = L.log
    if not rebase:
        return log.U, {"values": "logged residuals U_i (rebase off)"}
    if L.name not in _VALUES:
        while len(_VALUES) >= _BASE_MAX:
            _VALUES.pop(next(iter(_VALUES)))
        V = log.U.clone()
        for b in torch.unique(log.batch, sorted=True).tolist():
            idx = (log.batch == b).nonzero().flatten()
            ear = (log.batch < b).nonzero().flatten()
            if ear.numel():
                V[idx] = log.U[idx] + (log.K[idx] @ log.W[ear].T) @ log.U[ear]
        ref = lib_rebase(log)
        dev = float(((V - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-300)).max())
        if not bool(torch.isfinite(V).all()):
            raise RuntimeError(f"{L.name}: a rebased value is not finite")
        if not dev < 1e-9:                                              # cancellation among very large terms: reported, not hidden
            import sys
            print(f"e10t WARNING {L.name}: update-by-update rebased values deviate from rescope.rebase by {dev:.2e} (relative)", file=sys.stderr, flush=True)
        _VALUES[L.name] = (V, {"values": "rebased (rescope.rebase's formula, update by update)", "max_rel_deviation_from_rescope_rebase": dev,
                               "deviation_above_1e-9": not dev < 1e-9,
                               "n_bitwise_equal_to_rescope_rebase": int((V == ref).all(1).sum()), "sha1": hashlib.sha1(V.numpy().tobytes()).hexdigest()[:12],
                               "norm_min_median_max": [float(x) for x in np.quantile(V.norm(dim=1).numpy(), [0.0, 0.5, 1.0])]})
    return _VALUES[L.name]


def _named(g, note: str):
    return g.replace(meta=dict(g.meta, values=note))


def base_gate(L, feature: str = "ncoord", fire: str = "top1", response: str = "repl", rebase: bool = True, guards: bool = True):
    """The un-thresholded gate of a text log (library feature_gate, role 'all') with its guards: the stored solver keys and
    the key of every request as served.  One object per (log, switches): every calibration shares its detectors."""
    from rescope import add_guards, feature_gate
    key = (L.name, feature, fire, response, bool(rebase), bool(guards))
    if key not in _BASE:
        while len(_BASE) >= _BASE_MAX:
            _BASE.pop(next(iter(_BASE)))
        V, vrec = released_values(L, rebase)
        S = _named(feature_gate(L.log, feature, role="all", fire=fire, response=response, values=V), "rebased" if rebase else "logged")
        rec = {"values": vrec}
        if guards:
            S, grec = add_guards(S, *L.own_keys())
            rec.update(grec, guard_keys="stored solver keys (k_request) + the cached key of every request prompt at its solver position")
        _BASE[key] = (S, rec)
    return _BASE[key]


def _install_thresholds(S, lit: torch.Tensor, allow_silence: bool, op: str):
    """Thresholds `lit` (D,) with the library's own-key rule (keep_guards), as the driver's fx.methods._install_thresholds."""
    from rescope import keep_guards
    base = S.fold()
    th, silenced, capped = keep_guards(base, lit)
    if allow_silence:
        th, capped = lit, torch.zeros_like(capped)
    new = base.replace(det_tau=th, eta=1.0, det_b=None, det_u=None, meta=dict(base.meta, calibration={"op": op}))
    ok, T = new.guard_ok(), S.T
    cap_edits = sorted({int(e) for e in S.det_edit[capped]})
    rec = {"op": op, "allow_silence": allow_silence, "n_guards": int(ok.numel()), "n_own_keys_silenced_by_literal_rule": int(silenced[:T].sum()),
           "n_guards_silenced_by_literal_rule": int(silenced.sum()), "capped": bool(capped.any()), "n_edits_capped": len(cap_edits),
           "capped_edits": cap_edits, "n_guards_silenced": int((~ok).sum())}
    return new, rec


def self_negatives(L, S) -> np.ndarray:
    """m_i = the largest feature of edit i on the OTHER logged sequences (the log-only rule of ReScope-W): every prompt position of
    the other requests' prompts and the non-solver positions of its own request prompt (cached keys; the pre-edit continuation
    is not part of the log and is not used).  The request prompts ALONE are used: the editor's 20 template-prefixed copies of
    each request (the context templates in the log's meta) are not in the prompt table and are not scored."""
    T = S.T
    m = torch.full((T,), -float("inf"), dtype=torch.float64)
    for r in L.rows["request"]:
        k = torch.from_numpy(np.array(L.kc.keys(r["text_id"])[:L.kc.n_prompt(r["text_id"])]))
        R = S.scores(k)[:, :T].clone()
        R[L.solver_pos[r["pos"]], r["pos"]] = -float("inf")
        m = torch.maximum(m, R.max(0).values)
    return m.numpy()


def guard_margin(S, slack: float = e10t.BAND, key_norm=None):
    """GUARDS HOLD AS SERVED.  The library keeps a guard firing by capping its edit's threshold at (1 - 1e-9) x the guard's coordinate:
    enough where the served key reproduces the guard key bit for bit (the vision-language testbed at batch size 1), not here, where
    the key of a prompt served in a padded batch differs from the cached key by float32 round-off (a few 1e-7 of the coordinate's
    scale; tests c_cache / g_cache): about half of the capped edits would not fire on their own request as served.  Every threshold
    whose guard fires but lies INSIDE the decision band is therefore lowered until the guard is clear of the band:
        tau_j <- min(tau_j, min over the firing guards g of edit j of  r_j(g) - slack (|tau_j| + scale_j(g))),
    scale = |w_j||g|/s_j (solver coordinate), 1 (cosine), the distance (ball); slack = e10t.BAND (times 1.001).  key_norm (T,): the
    largest key norm of every edit's own request prompt; it replaces |g| where larger, so that the request is also outside the
    band that e10t.gate.decide draws around the threshold (which uses the largest key norm of the query).  Thresholds are only
    lowered, by at most a few 1e-5 of their scale; guards that do not fire (allow_silence) are left alone; one detector per edit.
    -> (scoped log, record: detectors lowered, how many of them the library had capped, largest lowering)."""
    from rescope.scoped import BALL, COS
    Sf = S.fold()
    G_ = int(Sf.guard_key.shape[0])
    if G_ == 0:
        return S, {"op": "guard_margin", "slack": slack, "n_guards": 0, "n_thresholds_lowered": 0}
    if Sf.D != Sf.T:
        raise NotImplementedError("guard_margin: one detector per edit")
    ge = Sf.guard_edit
    own = Sf.guard_scores()[torch.arange(G_), ge]                       # coordinate of every guard on its edit's detector (each guard its own query)
    wn = Sf.det_w.norm(dim=1) / Sf.det_s
    kind = Sf.det_kind[ge]
    gn = Sf.guard_key.norm(dim=1)
    if key_norm is not None:
        gn = torch.maximum(gn, torch.as_tensor(key_norm, dtype=torch.float64, device=gn.device)[ge])
    scale = gn * wn[ge]
    scale = torch.where(kind == COS, wn[ge], scale)
    scale = torch.where(kind == BALL, own.abs(), scale)
    th = Sf.det_tau
    thg = th[ge]
    fires = own >= thg
    need = own - 1.001 * slack * (torch.where(torch.isfinite(thg), thg.abs(), torch.zeros_like(thg)) + scale)
    low = torch.full_like(th, float("inf"))
    if bool(fires.any()):
        low = low.scatter_reduce(0, ge[fires], need[fires], reduce="amin", include_self=True)
    new = torch.minimum(th, low)
    ch = (new < th).nonzero().flatten()
    capped = (th[ch] >= (1 - 1e-6) * torch.full_like(th, float("inf")).scatter_reduce(0, ge[fires], own[fires], reduce="amin", include_self=True)[ch]) if ch.numel() else ch
    rec = {"op": "guard_margin", "slack": slack, "n_guards": G_, "n_guards_firing": int(fires.sum()), "n_thresholds_lowered": int(ch.numel()),
           "n_lowered_that_sat_at_a_guard_cap": int(capped.sum()) if ch.numel() else 0, "lowered_edits": [int(j) for j in ch[:50]],
           "max_lowering": float((th[ch] - new[ch]).max()) if ch.numel() else 0.0,
           "max_relative_lowering": float(((th[ch] - new[ch]) / th[ch].abs().clamp_min(1e-300)).max()) if ch.numel() else 0.0}
    return (Sf.replace(det_tau=new) if ch.numel() else S), rec


def _request_key_norm(L, S):
    """(T,) the largest key norm over the cached positions of every edit's own request prompt."""
    return G.features(L, S, [r["text_id"] for r in L.rows["request"]])["kmax"]


def calibrate(L, S, rule: str = "two_block", alpha: float = e10t.ALPHA, n: int = e10t.N_CAL, draw: int = 0, cal=None, split=(1, 1),
              allow_silence: bool = False, device: str = "cpu", served_guards: bool = True):
    """The library's conformal calibration of ANY scoped log S of this log at all positions (role 'all'): the score of a benign
    prompt is the maximum of S's feature over the cached positions of the prompt and of its pre-edit continuation.
    rule 'two_block' (default: calibrate_two_block, split seed = the draw, top-1 by margin) | 'shared' (calibrate_shared).
    cal: an explicit e10t.data.Draw instead of (n, draw).  -> (calibrated scoped log, the library's record).  Guards of S keep
    firing unless allow_silence (capped edits are counted in the record); served_guards (default): the thresholds of guards that
    lie inside the decision band are then lowered until the guard fires AS SERVED (guard_margin; record field 'guard_margin').
    This is the calibration of every row; use it with e10t.methods.wrap for gates built elsewhere (other features, repaired
    detectors)."""
    from rescope import calibrate_shared, calibrate_two_block
    if S.role != "all":
        raise ValueError("text gates decide at all positions (role 'all')")
    dr = cal if cal is not None else D.calibration(L, n, draw)
    Q = G.features(L, S, dr.text_ids, device)["Q"]
    if rule == "two_block":
        S2, rec = calibrate_two_block(S, coords=Q, alpha=alpha, split=tuple(split), seed=dr.seed, rank="excess", allow_silence=allow_silence)
    elif rule == "shared":
        S2, rec = calibrate_shared(S, coords=Q, alpha=alpha, rank="r", allow_silence=allow_silence)
    else:
        raise ValueError("rule: 'two_block' | 'shared'")
    if served_guards and not allow_silence:
        S2, rec["guard_margin"] = guard_margin(S2, key_norm=_request_key_norm(L, S2))
    return S2, rec


def scoped(L, name: str = "rescope", label: str | None = None, *, feature: str = "ncoord", fire: str = "top1", response: str = "repl",
           rebase: bool = True, guards: bool = True, calibration: str = "two_block", alpha: float = e10t.ALPHA, n: int = e10t.N_CAL,
           draw: int = 0, split=(1, 1), allow_silence: bool = False, null_n: int | None = None, null_ridge: float = 0.0,
           device: str = "cpu", served_guards: bool = True) -> Method:
    """A calibrated scoped gate at all positions (the builder behind rescope / rescope_shared / rescope_null / cosine / where).
    null_n: None = no null repair; an integer (also 0) = the null-repair path with that many benign prompts."""
    t0 = time.time()
    S, grec = base_gate(L, feature, fire, response, rebase, guards)
    rec = {"name": name, "log": L.name, "T": L.T, "feature": feature, "role": "all", "fire": fire, "response": response,
           "values": "the log's terms <W_i, k> U_i (response LIN: logged values, no round-off term)" if response == "lin" else ("rebased" if rebase else "logged"),
           "calibration": calibration, "built_on": "cpu", "add_guards": grec,
           "positions": f"prompt + {e10t.N_DEC}-token pre-edit continuation (key cache)", "band": e10t.BAND}
    if null_n is not None:
        from rescope import null_repair
        if feature not in ("ncoord", "coord"):
            raise ValueError("the null repair changes linear detectors (feature 'ncoord' | 'coord')")
        A = D.null_sample(L, int(null_n), draw, n)
        Kb, _ = L.kc.stack([r["text_id"] for r in A])
        S0 = S
        S, r0 = null_repair(S0, Kb, ridge=float(null_ridge))
        infl = (S.det_w.norm(dim=1) / S0.det_w.norm(dim=1)).numpy()
        rec["null_repair"] = dict(r0, n_a=len(A), prompt_ids=[r["prompt_id"] for r in A], n_keys=int(Kb.shape[0]),
                                  keys="every cached position (prompt + pre-edit continuation) of the n_a benign prompts",
                                  sample="the next n_a rows of the draw's ranking of the REPAIR pool whose text is no calibration text (e10t.data.null_sample)",
                                  detector_norm_inflation={"median": float(np.median(infl)), "max": float(infl.max())})
    dr = None
    if calibration in ("two_block", "shared"):
        dr = D.calibration(L, n, draw)
        if null_n:
            if set(dr.text_ids) & {r["text_id"] for r in A}:
                raise ValueError("the null-repair sample shares a prompt text with the calibration draw")
        S2, crec = calibrate(L, S, calibration, alpha, cal=dr, split=split, allow_silence=allow_silence, device=device, served_guards=served_guards)
        rec["draw"] = dr.rec
    elif calibration == "logonly":
        mneg = self_negatives(L, S)
        lit = torch.cat([torch.from_numpy(mneg + SELF_MARGIN * np.abs(mneg)), S.fold().det_tau[S.T:]])
        S2, crec = _install_thresholds(S, lit, allow_silence, "self (log-only, per edit)")
        crec.update(margin=SELF_MARGIN, negatives=_quant(mneg))
        if served_guards and not allow_silence:
            S2, crec["guard_margin"] = guard_margin(S2, key_norm=_request_key_norm(L, S2))
    elif calibration == "none":
        S2, crec = S, {"op": "none (all thresholds -inf)"}
    else:
        raise ValueError(f"calibration {calibration!r}: 'two_block' | 'shared' | 'logonly' | 'none'")
    rec["cal"] = crec
    rec["seconds"] = time.time() - t0
    cal = calibration in ("two_block", "shared")
    M = Method(name=name, label=label or name, obj=S2, kind="gate", log=L.name, rec=rec, alpha=alpha if cal else None,
               n=dr.n if dr is not None else None, draw=dr.draw if dr is not None else None,
               variant=",".join(f"{k}={v}" for k, v in (("fire", fire), ("response", response), ("rebase", rebase), ("guards", guards),
                                                        ("allow_silence", allow_silence), ("split", tuple(split)), ("served_guards", served_guards))
                                if v != {"fire": "top1", "response": "repl", "rebase": True, "guards": True, "allow_silence": False, "split": (1, 1),
                                         "served_guards": True}[k]))
    return _finish(M, L)


def rescope(L, **kw) -> Method:
    kw.setdefault("label", "rescope")
    return scoped(L, "rescope", **kw)


def rescope_shared(L, **kw) -> Method:
    kw.setdefault("label", "rescope_shared")
    return scoped(L, "rescope_shared", calibration="shared", **kw)


def rescope_null(L, n_a: int = e10t.N_NULL, ridge: float = 0.0, **kw) -> Method:
    kw.setdefault("label", "rescope_null")
    return scoped(L, "rescope_null", null_n=int(n_a), null_ridge=ridge, **kw)


def cosine(L, calibration: str = "two_block", gamma: float = e10t.COS_PUBLISHED, rebase: bool = True, label: str | None = None, **kw) -> Method:
    """Cosine gate at all positions.  calibration='two_block' (default) | 'shared': calibrated like the default method, with its
    guards; 'published': the fixed threshold gamma = 0.65 (NeuralDB), no guards."""
    if calibration in ("published", "default", "fixed"):
        from rescope import cosine_gate
        V, vrec = released_values(L, rebase)
        g = _named(cosine_gate(L.log, float(gamma), role="all", values=V), "rebased" if rebase else "logged")
        rec = {"name": "cosine", "log": L.name, "T": L.T, "feature": "cos", "role": "all", "fire": "top1", "response": "repl", "values": vrec,
               "calibration": "published", "gamma": float(gamma), "source_of_default": "NeuralDB (arXiv 2507.18028): gamma = 0.65",
               "guards": "none (fixed threshold)", "built_on": "cpu", "band": e10t.BAND}
        return _finish(Method("cosine", label or "cosine_published", g, "gate", L.name, rec, variant=f"gamma{gamma:g}/all"), L)
    lab = label or ("cosine" if calibration == "two_block" else f"cosine_{calibration}")
    return scoped(L, "cosine", label=lab, feature="cos", calibration=calibration, rebase=rebase, **kw)


def exact(L, rebase: bool = True, label: str = "exact") -> Method:
    from rescope import exact_gate
    V, vrec = released_values(L, rebase)
    g = _named(exact_gate(L.log, role="all", values=V), "rebased" if rebase else "logged")
    rec = {"name": "exact", "log": L.name, "T": L.T, "feature": "cos", "role": "all", "fire": "top1", "response": "repl", "values": vrec,
           "calibration": "fixed", "gamma": 0.999, "guards": "none (fixed threshold)", "built_on": "cpu", "band": e10t.BAND}
    return _finish(Method("exact", label, g, "gate", L.name, rec), L)


def where(L, calibration: str = "two_block", label: str | None = None, **kw) -> Method:
    """ReScope-W (thresholds as the only repair variable): role 'all', fire 'all', response LIN with the logged values."""
    lab = label or {"none": "where_all", "two_block": "where_twoblock", "logonly": "where_logonly", "shared": "where_shared"}[calibration]
    return scoped(L, "where", label=lab, fire="all", response="lin", calibration=calibration, **kw)


def deployed(L, label: str = "deployed") -> Method:
    rec = {"name": "deployed", "log": L.name, "T": L.T, "what": "the deployed log: module weight W_E" + (" (prefix replay)" if L.is_prefix else "")}
    return _finish(Method("deployed", label, None, "deployed", L.name, rec), L)


def preedit(L, label: str = "preedit") -> Method:
    rec = {"name": "preedit", "log": L.name, "T": L.T, "what": "the pre-edit model: module weight W_0 (control)"}
    return _finish(Method("preedit", label, "W0", "preedit", L.name, rec), L)


def weight(L, W, label: str, rec: dict | None = None) -> Method:
    """Any full replacement weight of the edited module, in the module's orientation (6400, 1600): rollbacks, corrections."""
    if tuple(W.shape) != (D.D_IN, D.D_OUT):
        raise ValueError("a replacement weight has the module's orientation (6400, 1600)")
    r = dict(rec or {}, name="weight", log=L.name, T=L.T, weight_sha1=hashlib.sha1(W.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16])
    return _finish(Method("weight", label, W, "weight", L.name, r), L)


def wrap(L, S, label: str, rec: dict | None = None, alpha=None, n=None, draw=None) -> Method:
    """A Method around any scoped log of this log (e.g. after the library's least_repair / null_repair / calibrations)."""
    if S.role != "all":
        raise ValueError("text gates decide at all positions (role 'all')")
    return _finish(Method("custom", label, S, "gate", L.name, dict(rec or {}, name="custom", log=L.name, T=L.T, wrapped=True), alpha, n, draw), L)


BUILDERS = {"deployed": deployed, "preedit": preedit, "rescope": rescope, "rescope_shared": rescope_shared, "rescope_null": rescope_null,
            "cosine": cosine, "cosine_published": lambda L, **kw: cosine(L, "published", **kw), "exact": exact, "where": where,
            "where_logonly": lambda L, **kw: where(L, "logonly", **kw), "where_twoblock": lambda L, **kw: where(L, "two_block", **kw)}
ROWSET = ("deployed", "preedit", "rescope", "rescope_shared", "rescope_null", "cosine", "cosine_published", "exact")
ROWSET_R14 = ROWSET + ("where_logonly", "where_twoblock")              # + the two ReScope-W rows that amendment R14 registers for E10
DEPENDS_ON_DRAW = ("rescope", "rescope_shared", "rescope_null", "cosine", "where_twoblock")
UNDEFINED = {
    "where_logonly": "DEFINED, outside the eight rows: ReScope-W is registered for E10 by amendment R14 with log-only and with two-block thresholds; "
                     "built here as the rows where_logonly / where_twoblock (e10t.methods.ROWSET_R14).",
    "cosine_shared": "DEFINED, not registered for E10: e10t.methods.cosine(L, 'shared').",
    "cosine_default": "UNDEFINED: the retrofit of the cosine gate to the write position (0.65 at role 'last') needs a marked position; ROME writes at the last "
                      "subject token, which a served prompt does not mark.  Its all-positions form is the row cosine_published.",
    "ball": "NOT REGISTERED for E10: the GRACE-style gate queries the final prompt token by definition, where a subject-token key of these logs is never "
            "presented, so its published form cannot release anything here; an all-positions retrofit is constructible (e10t.methods.scoped(L, 'ball', "
            "feature='negdist')) but is not in the registered row list and was not built or run.",
    "ball_default": "UNDEFINED in its published form (radius 1.0 at the final prompt token): as ball.",
    "router": "UNDEFINED as defined: the WISE-style router decides at the last prompt position and serves the deployed output from there on (role "
              "'last+answer'); these logs write at the last subject token, before that position, so a routed query would not receive its edit.",
    "router_default": "UNDEFINED as defined: as router.",
    "zscore": "DEFINABLE (the z-score gate looks at every position and needs the deployed weight), not in the registered row list of E10; not built.",
    "zscore_default": "DEFINABLE, not in the registered row list of E10; not built.",
    "dual": "DEFINABLE (answer match: two decodes per query); cut on unseen logs by amendment R19 (6); not built.",
    "p12_comp_b_nall": "UNDEFINED: the linear repairs of P12 (COMP-B, REEDIT-B, RB-UNION, RB-TOT) are corrections that the vision-language harness "
                       "(07_repair_v3) built from its complaint sets; neither the harness nor the saved corrections exist for a text log.  Their text "
                       "counterparts (comp_min, rb_suffix, rb_lineage, blocklist) are registered separately by R14 and enter through e10t.methods.weight.",
    "p12_reedit_b_nall": "UNDEFINED: as p12_comp_b_nall.", "p12_rb_union_nall": "UNDEFINED: as p12_comp_b_nall.", "p12_rb_tot_nall": "UNDEFINED: as p12_comp_b_nall.",
}


def build(name: str, L, **kw) -> Method:
    if name not in BUILDERS:
        raise KeyError(f"method {name!r}: one of {tuple(BUILDERS)}")
    return BUILDERS[name](L, **kw)


def rowset(L, alpha: float = e10t.ALPHA, draw: int = 0, n: int = e10t.N_CAL, labels=None, **switches) -> list:
    """The rows of E10 that are defined on a text log, in table order (default labels: ROWSET, the eight rows; ROWSET_R14 adds the
    two ReScope-W rows of amendment R14).  Only DEPENDS_ON_DRAW rows depend on (alpha, draw); the others have the same
    Method.key for every draw.  switches go to the calibrated rows."""
    mk = {"deployed": lambda: deployed(L), "preedit": lambda: preedit(L),
          "rescope": lambda: rescope(L, alpha=alpha, n=n, draw=draw, **switches),
          "rescope_shared": lambda: rescope_shared(L, alpha=alpha, n=n, draw=draw, **switches),
          "rescope_null": lambda: rescope_null(L, alpha=alpha, n=n, draw=draw, **switches),
          "cosine": lambda: cosine(L, "two_block", alpha=alpha, n=n, draw=draw, **switches),
          "cosine_published": lambda: cosine(L, "published"), "exact": lambda: exact(L),
          "where_logonly": lambda: where(L, "logonly"), "where_twoblock": lambda: where(L, "two_block", alpha=alpha, n=n, draw=draw, **switches)}
    out = []
    for lab in (ROWSET if labels is None else labels):
        if lab not in mk:
            raise KeyError(f"{lab!r} is not a row of the E10 row set {ROWSET_R14}" + (f": {UNDEFINED[lab]}" if lab in UNDEFINED else ""))
        out.append(mk[lab]())
    return out
