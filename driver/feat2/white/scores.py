"""white.scores: the gate features added by PROTOCOL.md section 6, amendment "Review round 1", item R3, as score matrices and as
detectors of the library's scoped log.

EDIT KEYS.  k_i = the log's STORED SOLVER KEY of edit i, `EditLog.K[i]` (testbed: edit_factors.pt 'K1_solver', the B = 10 batched key at
the last prompt position of the edit's own sequence; float64 copy).  This is the key the library's cosine gate uses
(rescope.baselines._key_gate: det_key = log.K, det_w = log.K / |log.K|), and every feature below uses the same.

FEATURES (a query key k -> one score per edit; LARGER = CLOSER; value at the edit's own solver key k = k_i: 1 for all four)
  cos_centred     cos(k - mu, k_i - mu)                                    mu = mean reference key (BEN-W)            in [-1, 1]
  white(kappa)    k_i' M k / k_i' M k_i                                    M = (C_W + rho I)^-1, rho = kappa tr(C_W) / d;  a coordinate
                                                                           normalised to 1 at k_i (unbounded, linear in k)
  whitecos(kappa) k_i' M k / sqrt(k_i' M k_i * k' M k)                     the cosine in the whitened space             in [-1, 1]
  cos_null        <w_i', k> / |k|,  w_i' = the cosine detector k_i / |k_i| after the library's null repair (ridge 0) on a given set of
                  keys B: the minimum-change vector with <w_i', b> = 0 for every b in B and <w_i', k_i> = <k_i / |k_i|, k_i> (own
                  coordinate unchanged);  closed form w_i' = |k_i| P k_i / |P k_i|^2, P = projector on the complement of span(B).
                  0 on every b in B (to round-off), 1 at k_i.  The caller passes the 100 keys that rescope_null uses in the same draw.
  helper of rescope_nullW:  null_w(S) = the default method's detectors (normalised solver coordinate) after the library's null repair
                  (ridge 0) on ALL BEN-W keys (the prompt-only key of every BEN-W row); ncoord_null(log, queries, keys) = its scores.
Limits that tie them to the registered cosine gate (tested): mu = 0 -> cos_centred = cos;  kappa -> infinity -> whitecos = cos and
white = <k_i, k> / |k_i|^2;  B empty -> cos_null = cos.

TWO FORMS OF EVERY FEATURE, ONE ARITHMETIC
  scores(log, queries, feature, ...)          -> (n, T) float64 score matrix [queries x edits]     (cos_centred(...), white(...), ...)
  feature_gate(log, feature, ...)             -> an UN-THRESHOLDED rescope.ScopedLog (tau = -inf) whose detectors compute the feature:
                                                 S.scores(keys) IS the score matrix (the score functions call it).
  The gate is the registered cosine gate in everything but its detectors: it is built from rescope.feature_gate(log, 'cos', ...)
  (role 'last', top-1, REPL, rebased values, guards = the stored solver keys) and only det_w / det_s / det_kind are replaced.  So it is
  used exactly like the solver coordinate:  add_guards -> KeyCache.coords -> calibrate_two_block / calibrate_shared -> KeyCache.decide
  / fx.gate / the forward hook.
    white      plain ScopedLog, 'lin' detectors  w_i = M k_i,  s_i = <w_i, k_i>                    (every library operator applies)
    cos_null   plain ScopedLog, 'cos' detectors  w_i' (not unit vectors), s_i = 1
    cos_centred, whitecos   TCosLog, a subclass of ScopedLog that overrides ONE method (_scores: how a key is scored); the library's
               'cos' kind divides by the Euclidean norm of the raw key, these two divide by |k - mu| / by sqrt(k' M k).  replace / to /
               add_guards / calibration / key cache / hook work unchanged; operators that add detectors (witness_repair mode
               'detector') are refused (NotImplementedError).
  The library's null_repair changes 'lin' detectors only and RAISES (RuntimeError, an empty reduction) on any gate that has none --
  its own cosine gate included (tests/t_scores.py, c7).  cos_null therefore hands the cosine detector to null_repair as the linear
  coordinate it is (w = k_i / |k_i|, own coordinate s_i = |k_i|) and reads the repaired vector as a cosine again; nothing of the
  repair is re-implemented here.  Do not call null_repair on a cos_centred / whitecos / cos_null gate.

PRECISION AND COST.  float64 throughout (the library's gate is float64 as well).  Detectors are built inside wstats.single_thread()
(the same bits in every process); scoring is the library's batched float64 product (last bits depend on batch shape and device, as for
every gate).  whitecos needs float64 for k' M k = (|k|^2 - |k R'|^2) / rho, a difference of two nearly equal numbers (the measured
cancellation factors and the measured deviations from a naive explicit-inverse implementation are in tests/out/table.txt).  Cost per
query key: T d flops for the detectors (as the cosine gate) + N d for whitecos (N = 600 reference keys); memory: T x d float64
detectors + N x d float64 for whitecos (53 MB).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from . import common as C
from . import wstats

torch = C.use_library()

from rescope import EditLog, ScopedLog, null_repair  # noqa: E402
from rescope import feature_gate as lib_feature_gate  # noqa: E402
from rescope.scoped import COS, LIN  # noqa: E402

FEATURES = ("cos_centred", "cos_null", "white", "whitecos")
NEEDS_STATS = ("cos_centred", "white", "whitecos")
NEEDS_KAPPA = ("white", "whitecos")
KAPPAS = C.KAPPAS


def label(feature: str, kappa=None) -> str:
    """Row label of a variant: 'cos_centred', 'cos_null', 'white_0.01', 'whitecos_1', ..."""
    return feature if feature not in NEEDS_KAPPA else f"{feature}_{wstats.kappa_tag(kappa)}"


def variants() -> list:
    """The registered variants of R3 in table order: [(label, feature, kappa or None), ...]."""
    out = [("cos_centred", "cos_centred", None), ("cos_null", "cos_null", None)]
    for f in NEEDS_KAPPA:
        out += [(label(f, k), f, float(k)) for k in KAPPAS]
    return out


def solver_keys(log):
    """(T, d) float64: the keys k_i of every feature = the log's stored solver keys (EditLog.K), the keys of the library's cosine gate."""
    return _as_log(log).K


def _as_log(log):
    """An EditLog, or a (T, d) tensor of solver keys (wrapped into a log without factors: enough for scores and detectors)."""
    if isinstance(log, EditLog):
        return log
    K = torch.as_tensor(log).detach().cpu().double()
    if K.dim() != 2:
        raise ValueError("log: a rescope.EditLog or a (T, d) tensor of solver keys")
    T = K.shape[0]
    return EditLog(torch.zeros(T, 1, dtype=torch.float64), K, K, torch.zeros(T, dtype=torch.long), name="keys")


@dataclass(eq=False)
class TCosLog(ScopedLog):
    """A scoped log with cosine-type detectors whose QUERY is normalised in another geometry than the raw key (one detector per edit):

        q_kind 'centred'   r_j(k) = <det_w_j, k - q_mu> / (det_s_j |k - q_mu|)              det_w_j = (k_j - mu) / |k_j - mu|
        q_kind 'mnorm'     r_j(k) = <det_w_j, k> / (det_s_j sqrt(k' M k))                   det_w_j = M k_j / sqrt(k_j' M k_j)
                           k' M k = q_c0 |k|^2 + q_c1 |k q_F'|^2                             (wstats.Whitener: F, c0, c1)

    det_s = 1, det_kind = 'cos', det_key = the raw solver keys (r_j = 1 there).  Everything else is ScopedLog; only _scores differs.
    float64 gate only (precision 'fp64')."""
    q_kind: str = "centred"
    q_mu: torch.Tensor | None = None       # (d,) float64 centre                      ('centred')
    q_F: torch.Tensor | None = None        # (N, d) or (d, d) float64 whitening factor ('mnorm')
    q_c0: float = 0.0
    q_c1: float = 0.0

    _TENSORS = ScopedLog._TENSORS + ("q_mu", "q_F")

    def __post_init__(self):
        super().__post_init__()
        if self.q_kind not in ("centred", "mnorm"):
            raise ValueError(f"q_kind {self.q_kind!r}")
        if self.precision != "fp64":
            raise NotImplementedError("TCosLog: the float64 gate only (precision 'fp64')")
        if self.D != self.T or not bool((self.det_kind == COS).all()):
            raise NotImplementedError("TCosLog holds exactly one cosine-type detector per edit (operators that add detectors are not defined for it)")
        if self.q_kind == "centred" and (self.q_mu is None or tuple(self.q_mu.shape) != (self.d,)):
            raise ValueError("q_kind 'centred' needs the centre q_mu (d,)")
        if self.q_kind == "mnorm" and (self.q_F is None or self.q_F.dim() != 2 or self.q_F.shape[1] != self.d):
            raise ValueError("q_kind 'mnorm' needs the whitening factor q_F (*, d)")
        self.q_c0, self.q_c1 = float(self.q_c0), float(self.q_c1)

    def replace(self, **kw):
        geo = bool(set(kw) & {"q_kind", "q_mu", "q_F", "q_c0", "q_c1"})
        if geo and "det_w" not in kw:
            # the scores depend on the query geometry as well as on det_w: caches keyed by the identity of the detector tensors (the shared
            # driver's feature cache, fx.gate) must not serve the old scores, so a gate with another geometry gets its own det_w tensor
            kw = dict(kw, det_w=self.det_w.clone())
        new = super().replace(**kw)
        if geo:                                                          # cached guard scores are stale
            new._cache = {}
        return new

    def _scores(self, k):
        kd = k.double()
        if self.q_kind == "centred":
            x = kd - self.q_mu
            c = x @ self.det_w.T
            nrm = x.norm(dim=1, keepdim=True)
        else:
            c = kd @ self.det_w.T
            p = kd @ self.q_F.T
            n2 = self.q_c0 * (kd * kd).sum(1, keepdim=True) + self.q_c1 * (p * p).sum(1, keepdim=True)
            nrm = n2.clamp_min(0.0).sqrt()
        return (c / self.det_s) / nrm.clamp_min(1e-300)


def _subclass(S0: ScopedLog, **kw) -> TCosLog:
    """TCosLog with every field of S0 (the registered cosine gate) and the given replacements."""
    f = {x.name: getattr(S0, x.name) for x in dataclasses.fields(S0)}
    f.update(kw)
    return TCosLog(**f)


def _stats_of(stats, feature):
    if stats is None:
        raise ValueError(f"feature {feature!r} needs the reference statistics: stats = white.wstats.load(backbone) (or a KeyStats / Whitener)")
    return stats


def feature_gate(log, feature: str, *, stats=None, kappa=None, null_keys=None, ridge: float = 0.0, role: str = "last", fire: str = "top1",
                 response: str = "repl", values="rebased", W0=None) -> ScopedLog:
    """Un-thresholded scoped log (tau = -inf) whose detectors compute `feature` (module docstring), to be calibrated by
    rescope.calibrate_two_block / calibrate_shared like the library's feature_gate(log, 'ncoord' | 'cos').

    log        rescope.EditLog (keys = log.K, the stored solver keys), or a (T, d) tensor of solver keys (scores only: no values)
    feature    'cos_centred' | 'cos_null' | 'white' | 'whitecos'
    stats      white.wstats.KeyStats (wstats.load(backbone): the BEN-W statistics) for cos_centred / white / whitecos; for white /
               whitecos a ready wstats.Whitener is accepted too (kappa then comes from it)
    kappa      ridge of white / whitecos: rho = kappa tr(C_W) / d       (registered: 0.01, 0.1, 1)
    null_keys  (n, d) keys of cos_null (registered: the teacher-forced last-prompt keys of the 100 benign REPAIR queries that
               rescope_null uses in the same calibration draw: L.kc.keys('tf', fx.draws.null_sample(L, 100, draw)))
    ridge      ridge of the null repair of cos_null (registered: 0)
    role, fire, response, values, W0   as rescope.feature_gate (defaults: the registered cosine gate)
    The record of the construction is in S.meta['white']; S.meta['feature'] = the variant's label.
    Guards are the stored solver keys; add the B = 1 own-prompt keys with rescope.add_guards BEFORE calibrating, as for every gate."""
    if feature not in FEATURES:
        raise ValueError(f"feature: one of {FEATURES}")
    L = _as_log(log)
    S0 = lib_feature_gate(L, "cos", role=role, fire=fire, response=response, values=values, W0=W0)      # the registered cosine gate
    K, T = L.K, L.T
    one = torch.ones(T, dtype=torch.float64)
    rec = {"feature": feature, "keys": "stored solver keys (EditLog.K), as the library's cosine gate", "T": T, "d": L.d}
    with wstats.single_thread():
        if feature == "cos_centred":
            st = _stats_of(stats, feature)
            mu = st.mu if hasattr(st, "mu") else torch.as_tensor(st).double()
            if tuple(mu.shape) != (L.d,):
                raise ValueError("stats: the mean key must have the width of the log's keys")
            Cn = K - mu
            nrm = Cn.norm(dim=1)
            if not bool((nrm > 0).all()):
                raise ValueError("a solver key equals the mean key: its centred cosine is undefined")
            S = _subclass(S0, det_w=Cn / nrm[:, None], det_s=one, det_key=K, q_kind="centred", q_mu=mu.clone())
            rec.update(stats=getattr(st, "name", "custom"), n_reference=getattr(st, "n", None), norm_mu=float(mu.norm()),
                       centred_key_norm_min_median_max=[float(nrm.min()), float(nrm.median()), float(nrm.max())],
                       raw_key_norm_min_median_max=[float(x) for x in (K.norm(dim=1).min(), K.norm(dim=1).median(), K.norm(dim=1).max())])
        elif feature in NEEDS_KAPPA:
            st = _stats_of(stats, feature)
            if isinstance(st, wstats.Whitener):
                wh = st
                if kappa is not None and wh.kappa is not None and float(kappa) != float(wh.kappa):
                    raise ValueError("kappa differs from the whitener's")
            else:
                if kappa is None:
                    raise ValueError(f"feature {feature!r} needs kappa (registered: {KAPPAS})")
                wh = st.whitener(kappa=float(kappa))
            if wh.d != L.d:
                raise ValueError("stats: reference keys of another width than the log's keys")
            Wm = wh.apply(K)                                             # rows k_i' M
            if feature == "white":
                s = (Wm * K).sum(1)                                      # s_i = <w_i, k_i> = k_i' M k_i (the library's convention for 'lin')
                if not bool((s > 0).all()):
                    raise FloatingPointError("k_i' M k_i is not positive for some edit")
                S = S0.replace(det_w=Wm, det_s=s, det_kind=torch.full((T,), LIN, dtype=torch.long), det_key=K)
            else:
                s = wh.quad(K)                                           # k_i' M k_i by the formula that scores the query
                S = _subclass(S0, det_w=Wm / s.sqrt()[:, None], det_s=one, det_key=K, q_kind="mnorm", q_F=wh.F, q_c0=wh.c0, q_c1=wh.c1)
            kk = (K * K).sum(1)
            rec.update(stats=wh.rec.get("stats", "custom"), n_reference=wh.n, kappa=wh.kappa, rho=wh.rho, form=wh.form, linv=wh.rec.get("linv"),
                       own_quadratic_form_min_median_max=[float(s.min()), float(s.median()), float(s.max())],
                       # share of |k_i|^2 that M keeps relative to the isotropic 1 / rho: rho k_i' M k_i / |k_i|^2 (1 / cancellation factor)
                       rho_kMk_over_kk_min_median_max=[float(x) for x in ((wh.rho * s / kk).min(), (wh.rho * s / kk).median(), (wh.rho * s / kk).max())])
        else:                                                            # cos_null: the library's null repair on the cosine detectors
            if null_keys is None:
                raise ValueError("feature 'cos_null' needs null_keys (the keys rescope_null uses in the same calibration draw)")
            Kb = torch.as_tensor(null_keys).detach().cpu().reshape(-1, L.d)
            kn = K.norm(dim=1)
            # the cosine detector k_i / |k_i| as a linear coordinate with its own coordinate s_i = <k_i / |k_i|, k_i> = |k_i| ...
            # (without its guards during the repair: the library scores every guard key by key before and after a repair, 2 T one-key
            #  products that say nothing for an un-thresholded gate -- every guard fires at tau = -inf -- and cost 40 s at T = 2000;
            #  the repaired detectors do not depend on the guards: tests/t_scores.py, c4)
            S_lin = S0.replace(det_s=kn, det_kind=torch.full((T,), LIN, dtype=torch.long), guard_key=S0.guard_key[:0], guard_edit=S0.guard_edit[:0])
            S_rep, nrec = null_repair(S_lin, Kb, ridge=float(ridge))    # ... repaired by the LIBRARY: <w', b> = 0 (ridge 0), <w', k_i> = |k_i| ...
            S = S_rep.replace(det_s=one, det_kind=torch.full((T,), COS, dtype=torch.long),      # ... and read as a cosine again: <w', k> / |k|
                              guard_key=S0.guard_key, guard_edit=S0.guard_edit)
            infl = S.det_w.norm(dim=1) / S0.det_w.norm(dim=1)
            nrec = {k: v for k, v in nrec.items() if not k.startswith("n_guards_")}
            rec.update(null_repair=nrec, n_null_keys=int(Kb.shape[0]), ridge=float(ridge),
                       detector_norm_inflation={"median": float(infl.median()), "max": float(infl.max())})
    lab = label(feature, rec.get("kappa"))
    rec["label"] = lab
    return S.replace(name=f"{lab}:{L.name}", meta=dict(S0.meta, feature=lab, baseline=lab, white=rec))


def gates(log, stats, null_keys=None, kappas=KAPPAS, **kw) -> dict:
    """{label: un-thresholded gate} of the registered variants of R3 in table order: cos_centred, cos_null (only if null_keys is
    given: it depends on the calibration draw), white_<kappa>, whitecos_<kappa> for kappa in `kappas`.  kw: role / fire / response /
    values / W0 as feature_gate.  Memory: T x d float64 detectors per gate (T = 2000, d = 11008: 176 MB each)."""
    out = {"cos_centred": feature_gate(log, "cos_centred", stats=stats, **kw)}
    if null_keys is not None:
        out["cos_null"] = feature_gate(log, "cos_null", null_keys=null_keys, **kw)
    for f in NEEDS_KAPPA:
        for k in kappas:
            out[label(f, k)] = feature_gate(log, f, stats=stats, kappa=float(k), **kw)
    return out


def null_w(S: ScopedLog, stats=None, keys=None, ridge: float = 0.0, score_guards: bool = True):
    """Helper of rescope_nullW (R3: "rescope_null with all BEN-W keys in place of its 100 REPAIR keys"): the library's null repair
    (ridge 0) of the detectors of S on ALL BEN-W keys.

    S       the default method's un-thresholded gate (rescope.feature_gate(log, 'ncoord') with its guards added, i.e. fx.methods.
            base_gate(L)[0]): every 'lin' detector is replaced by its minimum-change version with zero coordinate on the keys and
            unchanged coordinate on the edit's own stored key;
    stats   white.wstats.load(backbone): its `keys` are the prompt-only keys at the last prompt position of the 600 BEN-W rows; or
    keys    an explicit (n, d) key matrix;
    score_guards   True (default): the library scores every guard of S before and after the repair, as in the driver's rescope_null
            (record fields n_guards_firing_before / _after / n_guards_lost; one one-key product per guard, twice: about 80 s for the
            4000 guards of a T = 2000 log).  False: the repair runs on S without its guards and they are put back: the same
            detectors bit for bit (tests/t_scores.py, c4), no guard counts in the record (at tau = -inf every guard fires anyway).
    -> (repaired scoped log, record) with the fields of fx.methods' rescope_null record (the library's null_repair record + n_a, sids,
    keys, sample, detector_norm_inflation).  Then run the default two-block calibration unchanged.
    The detectors are built with one BLAS thread (the same bits in every process); the library's own call is not pinned."""
    if (stats is None) == (keys is None):
        raise ValueError("give exactly one of stats (white.wstats.load(backbone)) and keys")
    Kb = stats.keys if keys is None else torch.as_tensor(keys).detach().cpu()
    with wstats.single_thread():
        if score_guards:
            S2, r0 = null_repair(S, Kb, ridge=float(ridge))
        else:
            S2, r0 = null_repair(S.replace(guard_key=S.guard_key[:0], guard_edit=S.guard_edit[:0]), Kb, ridge=float(ridge))
            S2 = S2.replace(guard_key=S.guard_key, guard_edit=S.guard_edit)
            r0 = {k: v for k, v in r0.items() if not k.startswith("n_guards_")}
        infl = S2.det_w.norm(dim=1) / S.det_w.norm(dim=1)
    rec = dict(r0, n_a=int(Kb.shape[0]), sids=None if stats is None or stats.sids is None else list(stats.sids),
               keys="prompt-only key at the last prompt position (the key cache's decode key) of every BEN-W row" if keys is None else "explicit keys",
               sample=("all rows of BEN-W (10_final/cache2/benign_parts): no BEN-W query is a probe query of the backbone's VLKEB family, so the sample is "
                       "disjoint from the REPAIR pool and from every calibration draw of those logs (NOT of the MMKE-Bench logs: tests/t_parts.py, p9)")
               if keys is None else "explicit",
               detector_norm_inflation={"median": float(infl.median()), "max": float(infl.max())})
    return S2.replace(meta=dict(S2.meta, null_w=rec)), rec


# ----------------------------------------------------------------------------- score matrices [queries x edits]
def scores(log, queries, feature: str, *, stats=None, kappa=None, null_keys=None, ridge: float = 0.0):
    """(n, T) float64: score of every query key (rows of `queries`, (n, d), any float dtype) on every edit of the log, for
    feature in FEATURES (arguments as feature_gate).  = feature_gate(...).scores(queries): the arithmetic the calibrated gate runs."""
    S = feature_gate(log, feature, stats=stats, kappa=kappa, null_keys=null_keys, ridge=ridge, values="stored")
    return S.scores(torch.as_tensor(queries))


def cos_centred(log, queries, stats):
    """cos(k - mu, k_i - mu); stats: wstats.KeyStats (mu = its mean key).  1 at k = k_i."""
    return scores(log, queries, "cos_centred", stats=stats)


def white(log, queries, stats, kappa: float):
    """k_i' M k / k_i' M k_i with M = (C + rho I)^-1, rho = kappa tr(C) / d.  1 at k = k_i."""
    return scores(log, queries, "white", stats=stats, kappa=kappa)


def whitecos(log, queries, stats, kappa: float):
    """k_i' M k / sqrt(k_i' M k_i * k' M k).  1 at k = k_i."""
    return scores(log, queries, "whitecos", stats=stats, kappa=kappa)


def cos_null(log, queries, null_keys, ridge: float = 0.0):
    """The library's cosine detector of every edit after the library's null repair (ridge 0) on `null_keys`: <w_i', k> / |k|.
    0 on the null keys, 1 at k = k_i."""
    return scores(log, queries, "cos_null", null_keys=null_keys, ridge=ridge)


def ncoord_null(log, queries, null_keys, ridge: float = 0.0):
    """Scores of the default method's detectors (normalised solver coordinate <W_i, k> / <W_i, k_i>) after the library's null
    repair on `null_keys` (rescope_null: its 100 REPAIR keys; rescope_nullW: stats.keys, all BEN-W keys).  0 on the null keys, 1 at k_i."""
    if not isinstance(log, EditLog):
        raise TypeError("ncoord_null needs the edit log (its read-out vectors W_i), not only the solver keys")
    S = lib_feature_gate(log, "ncoord", values="stored")
    S2, _ = null_w(S, keys=null_keys, ridge=ridge)
    return S2.scores(torch.as_tensor(queries))


def own_scores(S: ScopedLog):
    """(T,) float64: the score of every edit's own solver key on its own detector (each key as its own one-key query); 1 by definition."""
    T = S.T
    R = S.key_scores(S.det_key[:T])
    return R[torch.arange(T), torch.arange(T)]
