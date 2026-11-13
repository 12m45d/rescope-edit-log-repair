"""fx.methods: one constructor per method name -> Method (an object the evaluation accepts + a record of what determined it).

    M = fx.methods.build("rescope", L, alpha=0.05, n=200, draw=0)          # L = fx.logs.open(name); H = fx.logs.load_model(name) where needed
    rows = fx.methods.e1_rowset(L, alpha=0.05, draw=0, H=H)                 # the fixed list of methods of experiment E1 (E3 / E4: subsets by label)

NAMES (build(name, L, H=None, **kw))
  deployed          the deployed log W_E                                            preedit    the pre-edit model W_0 (control)
  rescope           THE DEFAULT METHOD (PROTOCOL.md section 0 + two-block calibration of section 6): role 'last', normalised solver
                    coordinate, top-1, response REPL with rebased values, guards = stored solver keys + B = 1 keys of the edits' own
                    Rel prompts, two-block conformal calibration on n benign REPAIR queries (draw d: seed d for the draw and the split)
  rescope_shared    the same with the shared-scale conformal rule (one conformal threshold on the normalised coordinate)
  rescope_null      ReScope-N (registered variant, amendment of 2026-11-11 01:35): the detectors are first cleared of the last-prompt keys
                    of n_a = 100 further benign REPAIR queries (library null_repair, ridge 0), then the default calibration runs unchanged
                    (the 100 queries share no query with the calibration draw: fx.draws.null_sample, amendment of 09:03)
  rescope_logonly   the log-only start of the evidence curve: thresholds = the largest pooled normalised cross-talk among the log's own
                    solver keys (calibrate_log_only(q = 1)); least=<sequences>: + least repair of these received benign queries
  where_logonly     Rescope-WHERE (role 'all', fire 'all', response LIN = the original log terms <W_i, k> U_i, no round-off term) with the
                    log-only rule: tau_i = 5 % above the largest coordinate of edit i on the other logged sequences (all positions of the
                    other edits' own Rel sequences and the non-solver positions of its own; needs the model once per log, then cached)
  where_least       Rescope-WHERE from the deployed log (all thresholds -inf) + least repair of a complaint set (complaints=<sequences>,
                    e.g. fx.draws.complaints(L, n, seed)); needs the model (every position of the complaints' sequences)
  cosine            NeuralDB-style gate (top-1 raw cosine to the stored solver keys, REPL, rebased values):
                    calibration='two_block' (default) | 'shared' | 'default' (published threshold 0.65); role='last' (default) | 'all'
  ball              GRACE-style gate (minus the Euclidean distance to the stored key): calibration='two_block' | 'shared' | 'default'
                    (published eps_init = 1.0, one global radius) | 'half_min' (half the smallest distance between two stored keys)
  exact             exact-match gate (cosine >= 0.999)
  router            WISE-style activation router (needs the model's weights): calibration='conformal' (default: one conformal threshold
                    on the activation, all n queries) | 'two_block' | 'default' (eps = the smallest activation of the edits' own keys)
  zscore            CoachHooK-style z-score gate (needs the model): calibration='conformal' | 'two_block' | 'default' (alpha_z = 2.2) |
                    'dynamic' (the paper's rule Eq. 19 on the edits' own prompts)
  dual              DUAL (answer match; free generation only): variant='eq' (default) | 'contains'
  p12               a linear repair of P12 saved on disk (method='COMP_B', n='ALL', seed=0): harness Correction (needs the model)
  wrap(L, obj, label, rec)   any other object of the library (a scoped log after least_repair / null_repair / witness_repair, a weight)
  recalibrate(L, S, alpha, n, draw, exclude=complaints)      the two-block rule re-run on any scoped log (after detector repairs)
ABLATION SWITCHES of rescope / cosine / ball (keyword arguments; defaults = the protocol's method)
  role='last' | 'last+answer' | 'all'        response='repl' | 'gain' | 'lin'        rebase=True | False        fire='top1' | 'all'
  feature='ncoord' | 'coord' | 'cos' | 'negdist'       calibration='two_block' | 'shared' | 'log_only' | 'self' | 'running_max' | 'none'
  split=(1, 1) | (1, 3) | (3, 1) (block ratio)      baseline='max' | 0.98 (block-1 baseline: maximum or that quantile)
  source='repair' | 'benign', which='tf' | 'prompt' | 'decode' | 'steps' | 'both' | 'all' (calibration queries and their keys)
  contamination=0 | 0.01 | 0.05 | 0.2
  cal=<fx.draws.Draw> (an explicit calibration sample, e.g. a part of fx.draws.partition)      null_n / null_rows / null_ridge (ReScope-N)
  guards=True | False | 'all' (B = 1 own-prompt keys as guards; 'all': also the keys of a decode of the own prompt)
  which='all' + guards='all': calibration and guards on every key a query presents on either path (decode-proof variant)
  allow_silence=False | True (True: the literal, uncapped rule)
A Method's `variant` names the switches that are off their default (e.g. 'feature=coord,split=1:3'): use it in the result records.
Every constructor takes a LogData (fx.logs.open) or the name of a registered log.
Every gate is built ON THE CPU from the log's files and the key cache (the same bits in every process); the evaluation moves it to
the model's device.  Released values: the library's rebased values, computed batch by batch (released_values: same numbers as
rescope.rebase to round-off, checked at every construction; logs with the same first edits share the bits and hence the memo).
Nothing here changes a method: all arithmetic is the library's (feature_gate, add_guards, calibrate_*, least_repair, cosine_gate,
...); the three rules that the library does not name ('self', 'running_max', baseline quantile) are assembled from its public
pieces (keep_guards, float-safe thresholds) and described in their records.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

import fx
from . import draws as D, gate as G

NULL_N = 100                  # ReScope-N: size of the benign sample that the detectors are cleared of
COS_DEFAULT = 0.65            # NeuralDB: gamma = 0.65 (GPT2-XL, GPT-J, Llama-3)
EXACT_GAMMA = 0.999
BALL_DEFAULT = 1.0            # GRACE: eps_init = 1.0 (SCOTUS, Hallucination; 0.5 for zsRE); one global radius here
ZSCORE_DEFAULT = 2.2          # CoachHooK: alpha_z = 2.2
ROUTER_DEFAULT_Q = 0.0        # WISE: eps = the smallest activation indicator within the edits
SELF_MARGIN = 0.05            # log-only rule of section 0: 5 % above the largest coordinate on the other logged sequences
N_SWITCH = 40                 # evidence curve: two-block rule from n = 40 on (amendment of 22:10, item 2)
_BASE, _VALUES, _BASE_MAX = {}, {}, 6


@dataclass
class Method:
    name: str                               # constructor name
    label: str                              # label inside a row set (unique there)
    obj: object                             # None (deployed) | 'W0' (pre-edit) | ScopedLog | DenseGate | Dual | harness Correction | tensor
    kind: str                               # 'deployed' | 'preedit' | 'scoped' | 'gate' | 'dense' | 'dual' | 'correction' | 'weight' | 'unavailable'
    log: str
    rec: dict = field(default_factory=dict)
    alpha: float | None = None
    n: int | None = None
    draw: int | None = None
    variant: str = ""

    @property
    def memo(self) -> bool:
        """Scoped replay that decides at the last prompt position, top-1, response REPL: the model's output on a sequence depends
        only on the sequence and on the value released (fx.behave memoises it)."""
        return self.kind == "scoped" and G.is_scoped(self.obj) and self.obj.fire == "top1" and self.obj.response == "repl"

    @property
    def available(self) -> bool:
        return self.kind != "unavailable"

    @property
    def key(self) -> str:
        """12 hex digits identifying the method (its record without run-time notes)."""
        return self.rec.get("fingerprint", "")

    @property
    def digest(self) -> str:
        """16 hex digits of the CONTENT of the method object (content_digest): detectors, thresholds, eta, ranking baselines,
        released values, role / fire / response (scoped gates); the bytes of a replacement weight; kind and threshold of a dense
        gate; factors of a correction.  Two methods with the same digest decide and answer identically, whatever their labels
        and records; `key` identifies the RECORD (and misses the content of wrapped objects: review finding 2).  Use the digest
        wherever methods are compared, deduplicated or cached.  Recomputed at every access (objects are immutable by
        convention, but nothing is assumed): about 0.02 s at T = 100 and 0.5 s at T = 2000."""
        if self.kind == "unavailable":                                  # no object: the constructor name stands for it
            return hashlib.sha1(f"unavailable|{self.name}|{self.label}|{fx._RP.name_at_run_time(self.log)}".encode()).hexdigest()[:16]
        return content_digest(self.obj, self.log, self.kind)

    def __repr__(self):
        return f"Method({self.label!r}, {self.kind}, log={self.log}, alpha={self.alpha}, n={self.n}, draw={self.draw}, key={self.key})"


def _hb(h, t):
    """Feed a tensor's dtype, shape and bytes to a hash (None: a marker)."""
    if t is None:
        h.update(b"<none>")
        return
    a = t.detach().cpu() if torch.is_tensor(t) else torch.as_tensor(np.asarray(t))
    h.update(f"{a.dtype}{tuple(a.shape)}".encode())
    if a.dtype == torch.bfloat16:
        a = a.view(torch.int16)
    h.update(np.ascontiguousarray(a.numpy()).tobytes())


def content_digest(obj, log: str = "", kind: str = "") -> str:
    """Content hash of a method object (Method.digest).  Scoped logs: role, fire, response, precision, eta and the bytes of
    det_w, det_s, det_kind, det_key, det_edit, det_tau, det_b, det_u, values (+ lin_w, lin_u for response 'lin').  Guard keys are
    not part of it (they do not enter a decision or an answer).  Weights: dtype, shape, bytes.  Dense gates: kind, threshold,
    KV rule and the log (their detectors are the log's weight difference).  DUAL: targets and rule.  Corrections: kind, factors,
    gate, weight.  Deployed / pre-edit: the log."""
    if callable(getattr(obj, "content_digest", None)):                  # stage 2 (fx.review1): a retrieval rule hashes its own content
        return obj.content_digest()
    h = hashlib.sha1()
    log = fx._RP.name_at_run_time(log)                                  # digests were computed under the working names (LOG_NAMES.json, name_at_run_time)
    if obj is None or isinstance(obj, str):
        h.update(f"{'deployed' if obj is None else obj}|{log}".encode())
    elif hasattr(obj, "det_tau"):
        h.update(f"scoped|{obj.role}|{obj.fire}|{obj.response}|{obj.precision}|{float(obj.eta)!r}".encode())
        for nm in ("det_w", "det_s", "det_kind", "det_key", "det_edit", "det_tau", "det_b", "det_u", "values") + (("lin_w", "lin_u") if obj.response == "lin" else ()):
            h.update(nm.encode())
            _hb(h, getattr(obj, nm))
    elif torch.is_tensor(obj):
        h.update(b"weight|")
        _hb(h, obj)
    elif hasattr(obj, "targets") and hasattr(obj, "variant"):
        h.update(("dual|" + json.dumps([list(obj.targets), obj.variant, getattr(obj, "on", None)], sort_keys=True)).encode())
    elif hasattr(obj, "thr") and hasattr(obj, "firing"):
        h.update(f"dense|{getattr(obj, 'kind', '')}|{float(obj.thr)!r}|{getattr(obj, 'kv', '')}|{log}".encode())
    elif hasattr(obj, "kind") and hasattr(obj, "meta"):
        h.update(f"correction|{obj.kind}|{getattr(obj, 'gate', None)!r}".encode())
        for nm in ("Lf", "Rf", "W"):
            h.update(nm.encode())
            _hb(h, getattr(obj, nm, None))
    else:
        raise TypeError(f"no content digest for {type(obj).__name__}")
    return h.hexdigest()[:16]


def n_capped(M):
    """Number of capped edits of a calibrated gate (edits whose threshold was lowered so that their own keys keep firing: the
    conformal bound is void for them) -- to be reported for every calibrated gate; None for methods without a calibration."""
    return ((getattr(M, "rec", None) or {}).get("cal") or {}).get("n_edits_capped")


def dedupe(methods) -> dict:
    """{digest: [methods with that content]} in the order given: evaluate one method per digest."""
    out = {}
    for M in methods:
        out.setdefault(M.digest, []).append(M)
    return out


def _fp(rec: dict, obj=None) -> str:
    h = hashlib.sha1(fx._RP.name_at_run_time(json.dumps(fx.io.clean({k: v for k, v in rec.items() if k not in ("fingerprint", "seconds")}), sort_keys=True)).encode())
    if G.is_scoped(obj) or hasattr(obj, "det_tau"):
        h.update(np.ascontiguousarray(obj.det_tau.detach().cpu().numpy()).tobytes())
        h.update(repr(float(obj.eta)).encode())
    return h.hexdigest()[:12]


def _quant(v) -> dict:
    v = np.asarray(v, dtype=np.float64).ravel()
    f = v[np.isfinite(v)]
    return {"n": int(v.size), "n_finite": int(f.size), "n_pos_inf": int(np.isposinf(v).sum()), "n_neg_inf": int(np.isneginf(v).sum()),
            "min": float(f.min()) if f.size else None, "median": float(np.median(f)) if f.size else None, "max": float(f.max()) if f.size else None}


def _finish(M: Method, L) -> Method:
    """Complete the record: thresholds summary, own prompts, fingerprint."""
    g = M.obj
    if hasattr(g, "det_tau"):
        th = g.theta().detach().cpu().numpy()
        M.rec["thresholds"] = dict(_quant(th), sha1=hashlib.sha1(np.ascontiguousarray(th).tobytes()).hexdigest()[:12])
        if g.det_b is not None:
            M.rec["baselines"] = _quant(g.det_b.detach().cpu().numpy())
        M.rec["gate"] = {"role": g.role, "fire": g.fire, "response": g.response, "n_detectors": int(g.D), "n_edits": int(g.T), "n_guards": int(g.guard_key.shape[0]),
                         "values": g.meta.get("values"), "feature": g.meta.get("feature", g.meta.get("baseline"))}
        if G.is_scoped(g):                                              # own B = 1 Rel prompts under the final thresholds (key cache)
            d = G.decide(L, g, L.edit_rows["rel"], "tf")
            M.rec["own_prompts"] = {"n": L.T, "n_not_candidate": int((~d["own_cand"]).sum()), "n_not_released": int((~d["own_top"]).sum()),
                                    "not_candidate_edits": [int(i) for i in np.nonzero(~d["own_cand"])[0][:50]]}
    elif hasattr(g, "thr"):
        M.rec["thresholds"] = {"n": 1, "value": float(g.thr)}
    M.rec["fingerprint"] = _fp(M.rec, g)
    return M


def released_values(L, rebase: bool = True):
    """(T, m) fp64 values a scoped gate releases, and a record.  rebase=True: the rebased values of the library,
        V_i = U_i + sum over edits j of strictly earlier batches of U_j <W_j, k_i>          (rescope.rebase; log quantities only),
    computed BATCH BY BATCH: the rows of a batch come from one product over the edits written before it, so two logs that start with
    the same edits (a family T = 100 .. 2000) get the same bits for them, and the memo of fx.behave is shared along the family.
    rescope.rebase computes the same numbers in one product over the whole log, whose last bits depend on T; the two are compared
    here every time (the largest relative deviation must be round-off: below 1e-12).  rebase=False: the stored residuals U_i."""
    from rescope import rebase as lib_rebase
    log = L.log
    if not rebase:
        return log.U, {"values": "stored residuals U_i (rebase off)"}
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
        if not dev < 1e-12:
            raise RuntimeError(f"{L.name}: batch-by-batch rebased values deviate from rescope.rebase by {dev:.2e} (relative)")
        _VALUES[L.name] = (V, {"values": "rebased (rescope.rebase's formula, batch by batch)", "max_rel_deviation_from_rescope_rebase": dev,
                               "n_bitwise_equal_to_rescope_rebase": int((V == ref).all(1).sum()), "sha1": hashlib.sha1(V.numpy().tobytes()).hexdigest()[:12]})
    return _VALUES[L.name]


def _named(g, note: str):
    """The gate with a readable note about its values in its meta (the library writes 'custom' for an explicit tensor)."""
    return g.replace(meta=dict(g.meta, values=note))


def base_gate(L, feature: str = "ncoord", role: str = "last", fire: str = "top1", response: str = "repl", rebase: bool = True, guards: bool = True):
    """The un-thresholded gate of a log (library: feature_gate; guards = stored solver keys + the B = 1 keys of the edits' own Rel
    prompts, added BEFORE any calibration).  One object per (log, switches): its detectors are shared by every calibration."""
    from rescope import add_guards, feature_gate
    key = (L.name, feature, role, fire, response, bool(rebase), guards)
    if key not in _BASE:
        while len(_BASE) >= _BASE_MAX:                                   # T = 2000: a base gate with its guards is about 0.5 GB
            _BASE.pop(next(iter(_BASE)))
        V, vrec = released_values(L, rebase)
        S = _named(feature_gate(L.log, feature, role=role, fire=fire, response=response, values=V), "rebased" if rebase else "stored")
        rec = None
        if guards not in (True, False, "all"):
            raise ValueError("guards: True | False | 'all'")
        if guards == "all" and role != "last":
            raise ValueError("guards='all' (keys of a decode) is defined for role 'last'")
        if guards:
            S, rec = add_guards(S, *L.own_keys(True if guards is True else "all"))
        rec = dict(rec or {}, values=vrec)
        _BASE[key] = (S, rec)
    return _BASE[key]


def _slim(S):
    """The gate without its guard keys (a forward and a coordinate do not need them): what is moved to the model's device."""
    return S.replace(guard_key=S.guard_key[:0], guard_edit=S.guard_edit[:0])


def _session(L, H):
    if H is None:
        raise RuntimeError("this method needs the model: pass H = fx.logs.load_model(name)")
    from . import behave
    return behave.session(L, H)


def calib_coords(L, S, draw: D.Draw, H=None):
    """(n, D) fp64 per-query maxima of the gate's feature on the calibration queries: from the key cache for role 'last'
    (KeyCache.coords, as docs/USAGE.md); for the roles that look at every position, from the model (one truncated forward each)."""
    from . import pools as P
    kc = getattr(draw, "kc", None)
    P.guard(L, draw.rows, "calibration", kc)                            # test-only pool rows (BEN-T, MME-T) never calibrate
    if S.role == "last":
        return (L.kc if kc is None else kc).coords(S, draw.rows, draw.which)
    if kc is not None:
        raise ValueError("a gate that looks at every position is calibrated on sequences of the log's own evaluation (not on a key-level pool)")
    if draw.which != "tf":
        raise ValueError("calibration of a gate that looks at every position uses the teacher-forced sequences")
    ses = _session(L, H)
    Sg = _slim(S).to(H.dev)
    out = []
    with torch.no_grad(), ses.ev.hz.HE.vision_cache(H):
        for j in draw.rows:
            ses.ev.trim_vcache()
            out.append(Sg.query_max(ses.ev.eligible_keys(ses.item(j), Sg)).cpu())
    del Sg
    torch.cuda.empty_cache()
    return torch.cat(out)


# ----------------------------------------------------------------------------- calibration rules
def _cal_two_block(S, Q, alpha, seed, split, baseline, allow_silence, rank):
    from rescope import calibrate_two_block, split_blocks
    if baseline == "max":
        S2, rec = calibrate_two_block(S, coords=Q, alpha=alpha, split=tuple(split), seed=int(seed), rank=rank, allow_silence=allow_silence)
        rec["baseline_rule"] = "max of block 1"
        return S2, rec
    q = float(baseline)
    if not 0 < q <= 1:
        raise ValueError("baseline: 'max' or a quantile in (0, 1]")
    # block-1 baseline = the q-quantile (numpy, linear) of the block-1 coordinates of every detector.  The library takes the MAXIMUM of
    # block 1: hand it one synthetic block-1 row holding the quantiles; block 2, eta, thresholds, guards and record are the library's.
    Qn = Q.detach().cpu().numpy()
    i1, i2 = split_blocks(Qn.shape[0], tuple(split), int(seed))
    b = np.quantile(Qn[i1], q, axis=0)
    Q2 = torch.from_numpy(np.concatenate([b[None, :], Qn[i2]]))
    S2, rec = calibrate_two_block(S, coords=Q2, alpha=alpha, blocks=(np.array([0]), np.arange(1, 1 + len(i2))), rank=rank, allow_silence=allow_silence)
    rec.update(baseline_rule=f"{q:g}-quantile of block 1 (numpy linear)", n1=int(len(i1)), n=int(len(i1) + len(i2)), split=[int(split[0]), int(split[1])],
               seed=int(seed), block1=[int(i) for i in i1], block2=[int(i) for i in i2])
    return S2, rec


def pac_rank(n2: int, alpha, delta=0.1) -> int:
    """Rank of the block-2 quantile of the calibration-conditional variant (amendment R9, 'rescope_pac'): the smallest k with
        P[Beta(n2 + 1 - k, k) <= alpha] >= 1 - delta.
    With eta = the k-th smallest of n2 exchangeable block-2 scores, the false-firing probability of a fresh query, CONDITIONAL
    on the calibration sample, is Beta(n2 + 1 - k, k) distributed (continuous scores); the rule keeps it at most alpha with
    probability at least 1 - delta over calibration samples.  Computed exactly: P[Beta(n2 + 1 - k, k) <= alpha] =
    P[Bin(n2, alpha) >= n2 + 1 - k], so k = n2 - m with m the largest integer with P[Bin(n2, alpha) <= m] <= delta, in integer
    arithmetic (alpha, delta: floats are read as the closest fractions, as rescope.as_fraction).  k = n2 + 1 (nothing may fire:
    eta = +inf) if even m = 0 fails, i.e. (1 - alpha)^n2 > delta."""
    from math import comb
    from rescope import as_fraction
    a, d = as_fraction(alpha), as_fraction(delta)
    n2 = int(n2)
    if n2 < 1:
        raise ValueError("n2 >= 1")
    p, q = a.numerator, a.denominator
    lim = d.numerator * q ** n2                                          # cdf(m) <= delta  <=>  d.den * sum_{i <= m} C(n2, i) p^i (q - p)^(n2 - i) <= d.num * q^n2
    acc, m = 0, -1
    for i in range(n2 + 1):
        acc += comb(n2, i) * p ** i * (q - p) ** (n2 - i)
        if d.denominator * acc > lim:
            break
        m = i
    return n2 + 1 if m < 0 else n2 - m


def _cal_two_block_ext(S, Q, alpha, seed, split, allow_silence, rank, x=None, pac_delta=None):
    """The library's two-block calibration with (a) per-detector EXTERNAL baselines x (R5: b_j = max(block-1 maximum, x_j); x enters
    as one more block-1 row, so block 2, the exact rank, eta, the strict floating-point-safe thresholds, the ranking by the
    margin and the guards are the library's own code) and / or (b) the calibration-conditional rank of R9 (pac_rank; the library
    is called with alpha' = (n2 + 1 - k) / (n2 + 1), for which its exact rank ceil((1 - alpha')(n2 + 1)) is k).  With x = None and
    pac_delta = None this is calibrate_two_block(split, seed) itself (the caller uses the plain path then)."""
    from fractions import Fraction
    from rescope import as_fraction, calibrate_two_block, split_blocks
    Qn = Q.detach().cpu().numpy()
    n, Dd = Qn.shape
    i1, i2 = split_blocks(n, tuple(split), int(seed))
    blocks, Q2, xrec = (i1, i2), Qn, None
    if x is not None:
        xv = np.asarray(x.detach().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float64).reshape(-1)
        if xv.shape[0] == S.T and Dd > S.T:
            xv = np.concatenate([xv, np.full(Dd - S.T, -np.inf)])
        if xv.shape[0] != Dd or np.isnan(xv).any():
            raise ValueError(f"baseline_x: one value per detector ({Dd}; -inf = none), without NaN")
        b1 = Qn[i1].max(0)
        Q2 = np.concatenate([Qn, xv[None, :]])
        blocks = (np.append(i1, n), i2)
        xrec = {"n_given": int(np.isfinite(xv).sum()), "n_baselines_raised": int((xv > b1).sum()), "sha1": hashlib.sha1(np.ascontiguousarray(xv).tobytes()).hexdigest()[:12],
                "rule": "b_j = max(largest coordinate of detector j over block 1, x_j); block 2, rank, eta, thresholds and guards unchanged (R5)"}
    a = as_fraction(alpha)
    a_lib, prec = a, None
    if pac_delta is not None:
        n2 = len(i2)
        k = pac_rank(n2, a, pac_delta)
        a_lib = Fraction(n2 + 1 - k, n2 + 1) if k <= n2 else Fraction(1, 2 * (n2 + 1))
        from rescope import conformal_rank_exact
        prec = {"delta": float(as_fraction(pac_delta)), "alpha": float(a), "rank": int(k), "rank_of_the_default_rule": int(conformal_rank_exact(n2, a)), "n2": n2,
                "library_called_with_alpha": f"{a_lib.numerator}/{a_lib.denominator}",
                "guarantee": "P[false-firing probability of a fresh exchangeable query <= alpha | calibration sample] >= 1 - delta (uncapped edits)",
                "rule": "smallest k with P[Beta(n2 + 1 - k, k) <= alpha] >= 1 - delta (R9); eta = +inf if k = n2 + 1"}
    S2, rec = calibrate_two_block(S, coords=torch.from_numpy(np.ascontiguousarray(Q2)), alpha=a_lib, blocks=blocks, rank=rank, allow_silence=allow_silence)
    rec.update(baseline_rule="max of block 1" + ("" if x is None else " and the external per-detector baselines (baseline_x)"), n1=int(len(i1)), n=int(n),
               split=[int(split[0]), int(split[1])], seed=int(seed), block1=[int(i) for i in i1], block2=[int(i) for i in i2])
    if xrec is not None:
        rec["baseline_x"] = xrec
    if prec is not None:
        if prec["rank"] != rec["rank"] and not (prec["rank"] == prec["n2"] + 1 and rec["eta_infinite"]):
            raise RuntimeError(f"the library selected rank {rec['rank']}, the calibration-conditional rule asks for {prec['rank']}")
        rec.update(pac=prec, alpha=float(a), alpha_exact=f"{a.numerator}/{a.denominator}", bound_false_firing_marginal=rec["bound_false_firing"])
    return S2, rec


def _install_thresholds(S, lit: torch.Tensor, allow_silence: bool, op: str, b=None):
    """Thresholds `lit` (D,) with the library's own-key rule (keep_guards: an edit whose guard would be silenced is capped just below
    its smallest guard coordinate, and counted).  b: ranking baselines of fire 'top1' (None: rank by r)."""
    from rescope import keep_guards
    base = S.fold()
    th, silenced, capped = keep_guards(base, lit)
    if allow_silence:
        th, capped = lit, torch.zeros_like(capped)
    new = base.replace(det_tau=th, eta=1.0, det_b=b, det_u=None, meta=dict(base.meta, calibration={"op": op}))
    ok, T = new.guard_ok(), S.T
    cap_edits = sorted({int(e) for e in S.det_edit[capped]})
    rec = {"op": op, "allow_silence": allow_silence, "n_guards": int(ok.numel()), "n_own_keys_silenced_by_literal_rule": int(silenced[:T].sum()),
           "n_guards_silenced_by_literal_rule": int(silenced.sum()), "n_edits_silenced_by_literal_rule": len({int(e) for e in S.guard_edit[silenced]}),
           "capped": bool(capped.any()), "n_detectors_capped": int(capped.sum()), "n_edits_capped": len(cap_edits), "capped_edits": cap_edits,
           "n_guards_silenced": int((~ok).sum()), "n_own_keys_silenced": int((~ok[:T]).sum())}
    return new, rec


def self_negatives(L, S, H=None) -> np.ndarray:
    """m_i = the largest coordinate of edit i on the OTHER logged sequences (log-only rule of section 0; P10's HARD-SELF negatives).
    role 'last': the other edits' stored solver keys (the log alone).  Other roles: every position of the other edits' own B = 1 Rel
    sequences and the non-solver positions of its own Rel sequence: T truncated forwards with the model, cached under
    <memo>/aux/<log>/ (so that later processes need no model)."""
    from rescope import pooled_crosstalk
    T = S.T
    if S.role == "last":
        _, R = pooled_crosstalk(S)                                      # R[j, i] = r_i(k_j)
        R = R.clone()
        R[torch.arange(T), torch.arange(T)] = -float("inf")
        return R.max(0).values.cpu().numpy()
    feat = S.meta.get("feature", "ncoord")
    p = os.path.join(fx.MEMO_ROOT, "aux", L.name, f"selfneg_{feat}_{S.role.replace('+', '_')}.json")
    if os.path.exists(p):
        blob = fx.io.jload(p)
        if blob["T"] == T and blob["rescope"] == fx.code_fingerprint()["rescope"]:
            return np.array([float(x) for x in blob["m"]], dtype=np.float64)
    ses = _session(L, H)
    Sg = _slim(S).to(H.dev)
    m = torch.full((T,), -float("inf"), dtype=torch.float64, device=H.dev)
    npos = 0
    with torch.no_grad(), ses.ev.hz.HE.vision_cache(H):
        for i, j in enumerate(L.edit_rows["rel"]):
            ses.ev.trim_vcache()
            x, lp = ses.ev.capture(ses.item(j))
            R = Sg.scores(x)[:, :T]                                     # (positions, T)
            own = R[:, i].clone()
            own[lp] = -float("inf")                                     # its own solver position is not a negative
            R[:, i] = own
            if S.role == "last+answer":
                R = R[lp:]
            m = torch.maximum(m, R.max(0).values)
            npos += int(R.shape[0])
    out = m.cpu().numpy()
    del Sg, m
    torch.cuda.empty_cache()
    fx.io.jdump({"log": L.name, "T": T, "feature": feat, "role": S.role, "m": [float(x) for x in out], "n_positions": npos,
                 "rescope": fx.code_fingerprint()["rescope"], "what": "per-edit largest coordinate on the other logged sequences (B = 1 Rel sequences)"}, p)
    return out


def _cal_self(L, S, margin, allow_silence, H=None):
    m = self_negatives(L, S, H)
    lit = torch.from_numpy(m + margin * np.abs(m)).to(S.device)
    lit = torch.cat([lit, S.fold().det_tau[S.T:]])
    new, rec = _install_thresholds(S, lit, allow_silence, "self (log-only, per edit)")
    rec.update(margin=margin, negatives=_quant(m), rule="tau_i = m_i + margin |m_i|, m_i = largest coordinate of edit i on the other logged sequences",
               negatives_from="stored solver keys of the other edits" if S.role == "last" else "every position of the B = 1 Rel sequences (model)")
    return new, rec


def _cal_running_max(S, Q, allow_silence):
    """Per-edit running maxima: edit i fires iff r_i > the largest r_i over the n calibration queries (no alpha; bound T / (n + 1))."""
    b = Q.max(0).values.to(S.device)
    lit = torch.nextafter(b, torch.full_like(b, float("inf")))
    new, rec = _install_thresholds(S, lit, allow_silence, "running_max (per edit)", b=b)
    n, T = int(Q.shape[0]), S.T
    rec.update(n=n, rule="tau_i = next float above the largest coordinate of edit i over the n calibration queries; top-1 by r_i - b_i",
               bound_false_firing=min(1.0, T / (n + 1.0)), bound="T / (n + 1) (union bound over edits)")
    return new, rec


# ----------------------------------------------------------------------------- scoped gates
def scoped(L, name: str = "rescope", label: str | None = None, *, feature: str = "ncoord", role: str = "last", fire: str = "top1", response: str = "repl",
           rebase: bool = True, guards: bool = True, calibration: str = "two_block", alpha: float = 0.05, n: int = D.N_DEFAULT, draw: int = 0,
           split=(1, 1), baseline="max", source: str = "repair", which: str = "tf", contamination: float = 0.0, allow_silence: bool = False,
           rank: str | None = None, least=None, cal=None, null_n: int = 0, null_rows=None, null_ridge: float = 0.0, H=None,
           null_keys=None, baseline_x=None, delta: float = 0.1, part=None, mask=None, contam_half: str = "first") -> Method:
    """A calibrated scoped gate on feature `feature` (the builder behind rescope / rescope_shared / rescope_null / rescope_logonly /
    cosine / ball).  cal: an explicit calibration sample (fx.draws.Draw, e.g. a part of fx.draws.partition) instead of the draw
    (n, draw).  null_n / null_rows: clear the detectors of the last-prompt keys of a benign sample before calibrating (ReScope-N).
    Added with the stage-1 extension (defaults = the behaviour before it, bit for bit):
      null_keys     the null-repair keys given EXPLICITLY, (n_A, d) (e.g. fx.methods.null_keys(L, draw), or the BEN-W keys of
                    rescope_nullW); a key that is bitwise a teacher-forced calibration key is refused
      baseline_x    (D,) external per-detector baselines of the two-block rule (R5: b_j = max(block-1 maximum, x_j); -inf = none)
      calibration='pac', delta   the calibration-conditional rank of R9 (pac_rank) in the two-block rule
      source='BEN-S' | 'MME-C' (or source='benign', part='BEN-S'), mask='qdisjoint' | predicate, contam_half='even'   fx.draws.calibration"""
    L = _log(L)
    source = D.resolve_source(source, part)
    S, grec = base_gate(L, feature, role, fire, response, rebase, guards)
    nrec, A = None, None
    if null_keys is not None and (null_rows is not None or null_n):
        raise ValueError("give the null-repair sample once: null_keys (explicit keys), null_rows (sequences) or null_n (the draw's seeded sample)")
    if null_rows is not None:
        A = [int(j) for j in null_rows]
    elif null_n:
        if cal is not None:
            raise ValueError("with an explicit calibration sample (cal=) give the benign sample of the null repair explicitly (null_rows=)")
        A = D.null_sample(L, int(null_n), draw, n, "repair", mask=mask)
    if A is not None:                                                   # ReScope-N: detectors cleared of benign directions BEFORE calibration
        from rescope import null_repair
        from . import pools as _P
        if feature not in ("ncoord", "coord"):
            raise ValueError("the null repair changes linear detectors (feature 'ncoord' | 'coord')")
        _P.guard(L, A, "statistics")                                    # BEN-T rows never shape a detector
        S0 = S
        S, r0 = null_repair(S0, L.kc.keys("tf", A), ridge=float(null_ridge))
        infl = (S.det_w.norm(dim=1) / S0.det_w.norm(dim=1)).numpy()
        nrec = dict(r0, n_a=len(A), sids=[L.sid(j) for j in A], keys="teacher-forced key at the last prompt position of each benign query",
                    n_distinct_queries=len(set(D.query_ids(L, A))),
                    sample="explicit (null_rows)" if null_rows is not None else f"the first {len(A)} rows after position {n} of the draw's ranking of the {'repair'} "
                    "pool whose query is no calibration query and not yet in the sample (fx.draws.null_sample): disjoint from the calibration draw BY QUERY",
                    detector_norm_inflation={"median": float(np.median(infl)), "max": float(infl.max())})
    Kn = None
    if null_keys is not None:                                           # the null-repair keys themselves (same operator, same order of steps)
        from rescope import null_repair
        if feature not in ("ncoord", "coord"):
            raise ValueError("the null repair changes linear detectors (feature 'ncoord' | 'coord')")
        Kn = torch.as_tensor(null_keys)
        if Kn.dim() != 2 or Kn.shape[1] != S.d:
            raise ValueError(f"null_keys: (n_A, {S.d})")
        S0 = S
        S, r0 = null_repair(S0, Kn, ridge=float(null_ridge))
        infl = (S.det_w.norm(dim=1) / S0.det_w.norm(dim=1)).numpy()
        nrec = dict(r0, n_a=int(Kn.shape[0]), sids=None, keys="explicit keys (null_keys)", sample="explicit (null_keys)",
                    keys_sha1=hashlib.sha1(np.ascontiguousarray(Kn.detach().cpu().numpy()).tobytes()).hexdigest()[:12],
                    detector_norm_inflation={"median": float(np.median(infl)), "max": float(infl.max())})
    rec = {"name": name, "log": L.name, "T": L.T, "feature": feature, "role": role, "fire": fire, "response": response, "values": "rebased" if rebase else "stored",
           "guards": {True: "stored solver keys + B = 1 keys of the own Rel prompts", False: "stored solver keys only",
                      "all": "stored solver keys + B = 1 keys of the own Rel prompts (teacher-forced and the 11 forwards of a decode)"}[guards],
           "calibration": calibration,
           "built_on": "cpu", "add_guards": grec}
    if nrec is not None:
        rec["null_repair"] = nrec
    dr = None
    if calibration in ("two_block", "shared", "running_max", "pac"):
        dr = cal if cal is not None else D.calibration(L, n, draw, source, which, contamination, part=None, mask=mask, contam_half=contam_half)
        dkc = getattr(dr, "kc", None)
        if A is not None:                                               # by QUERY: a repeated query has the same key (amendment of 2026-11-11 09:03)
            shared = set() if dkc is not None else set(D.query_ids(L, A)) & set(D.query_ids(L, dr.rows))     # another key cache: no common query ids
            if shared:
                raise ValueError(f"the sample of the null repair shares {len(shared)} queries (same question and image content, hence the same key) with "
                                 "the calibration sample: the detectors would depend on calibration queries (drop these rows from one of the two)")
            nrec["n_queries_shared_with_calibration"] = 0
        if Kn is not None:                                              # explicit keys: no key may be a (teacher-forced) calibration key, bit for bit
            sha = lambda K: {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in K.detach().cpu().float().numpy()}      # noqa: E731
            ck = (L.kc if dkc is None else dkc)
            have = [j for j in dr.rows if ck.index[j].get("row_tf") is not None]
            both = sha(Kn) & sha(ck.keys("tf", have)) if have else set()
            if both:
                raise ValueError(f"{len(both)} of the explicit null-repair keys are calibration keys, bit for bit: the detectors would depend on "
                                 "calibration queries")
            nrec["n_keys_equal_to_a_calibration_key"] = 0
        Q = calib_coords(L, S, dr, H)
        if calibration in ("two_block", "pac") and (baseline_x is not None or calibration == "pac"):
            if baseline != "max":
                raise ValueError("baseline_x / calibration='pac' take the block-1 maximum as baseline (baseline='max')")
            S2, crec = _cal_two_block_ext(S, Q, alpha, dr.seed, split, allow_silence, rank or "excess", baseline_x, delta if calibration == "pac" else None)
        elif calibration == "two_block":
            S2, crec = _cal_two_block(S, Q, alpha, dr.seed, split, baseline, allow_silence, rank or "excess")
        elif calibration == "shared":
            from rescope import calibrate_shared
            S2, crec = calibrate_shared(S, coords=Q, alpha=alpha, rank=rank or "r", allow_silence=allow_silence)
        else:
            S2, crec = _cal_running_max(S, Q, allow_silence)
        rec["draw"] = dr.rec
    elif calibration == "log_only":
        from rescope import calibrate_log_only
        S2, crec = calibrate_log_only(S, q=1.0, allow_silence=allow_silence)
    elif calibration == "self":
        S2, crec = _cal_self(L, S, SELF_MARGIN, allow_silence, H)
    elif calibration == "none":
        S2, crec = S, {"op": "none (all thresholds -inf)"}
    else:
        raise ValueError(f"calibration {calibration!r}")
    rec["cal"] = crec
    if baseline_x is not None and calibration not in ("two_block", "pac"):
        raise ValueError("baseline_x raises the block-1 baselines of the two-block rule (calibration 'two_block' | 'pac')")
    if least is not None and len(least):                                # least repair of received benign queries (role 'last': key cache)
        from rescope import least_repair
        from . import pools as _P
        if role != "last":
            raise ValueError("least= takes sequences of the key cache (role 'last'); for Rescope-WHERE use where_least")
        _P.guard(L, least, "stream")
        K = L.kc.keys("tf", [int(j) for j in least])
        recs = []
        for i in range(K.shape[0]):                                     # one query at a time (the library's Session.complain, keys not kept)
            new, r = least_repair(S2, K[i:i + 1])
            recs.append(r)
            S2 = S2 if r["refused"] else new
        rec["least_repair"] = {"n_queries": len(recs), "n_refused": sum(r["refused"] for r in recs), "n_raising": sum(r["n_raised"] > 0 for r in recs),
                               "n_thresholds_raised_total": sum(r["n_raised"] for r in recs), "sids": [L.sid(int(j)) for j in least],
                               "margin": recs[0]["margin"], "abs_margin": recs[0]["abs_margin"], "on_conflict": "refuse (library default)"}
    cal_alpha = alpha if calibration in ("two_block", "shared", "pac") else None
    dflt = {"feature": {"cosine": "cos", "ball": "negdist"}.get(name, "ncoord"), "role": "last", "fire": "top1", "response": "repl", "rebase": True, "guards": True,
            "calibration": {"rescope_shared": "shared", "rescope_logonly": "log_only", "rescope_pac": "pac"}.get(name, "two_block"), "split": (1, 1), "baseline": "max", "source": "repair",
            "which": "tf", "contamination": 0.0, "allow_silence": False, "null": NULL_N if name == "rescope_null" else 0, "null_ridge": 0.0, "cal": "draw"}
    now = {"feature": feature, "role": role, "fire": fire, "response": response, "rebase": rebase, "guards": guards, "calibration": calibration, "split": tuple(split),
           "baseline": baseline, "source": dr.source if cal is not None else source, "which": dr.which if cal is not None else which, "contamination": contamination,
           "allow_silence": allow_silence, "null": 0 if A is None else len(A), "null_ridge": float(null_ridge), "cal": "draw" if cal is None else "explicit"}
    if cal is not None:
        dflt.update(source=now["source"], which=now["which"])
    variant = ",".join(f"{k}={':'.join(str(x) for x in v) if isinstance(v, tuple) else v}" for k, v in now.items() if v != dflt[k])   # the switches off their default
    rec["switches"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in now.items()}
    ext = {}                                                            # switches of the stage-1 extension: recorded only where they are off their default
    if mask is not None:
        ext["mask"] = mask if isinstance(mask, str) else "predicate"
    if contam_half != "first" and contamination:
        ext["contam_half"] = contam_half
    if baseline_x is not None:
        ext["x"] = "external"
    if calibration == "pac" and float(delta) != 0.1:
        ext["delta"] = float(delta)
    if Kn is not None:
        ext["null_keys"] = int(Kn.shape[0])
    if ext:
        variant = ",".join([v for v in (variant,) if v] + [f"{k}={v}" for k, v in ext.items()])
        rec["switches"].update(ext)
    M = Method(name=name, label=label or name, obj=S2, kind="scoped" if (role == "last") else "gate", log=L.name, rec=rec, alpha=cal_alpha,
               n=None if dr is None else dr.n, draw=None if dr is None else dr.draw, variant=variant)
    return _finish(M, L)


def rescope(L, **kw) -> Method:
    """The default method of the protocol (see the module docstring); keyword arguments are the ablation switches."""
    kw.setdefault("label", "rescope")
    return scoped(L, "rescope", **kw)


def rescope_shared(L, **kw) -> Method:
    kw.setdefault("label", "rescope_shared")
    return scoped(L, "rescope_shared", calibration="shared", **kw)


def rescope_null(L, n_a: int = 100, ridge: float = 0.0, n_A=None, **kw) -> Method:
    """ReScope-N, the registered variant of the amendment of 2026-11-11 01:35: a seeded draw of n_a = 100 benign queries of the REPAIR
    pool that shares no QUERY with the n calibration queries (fx.draws.null_sample; clarification of 2026-11-11 09:03); every
    detector is replaced by its minimum-change version with zero coordinate on the last-prompt keys of these queries and unchanged
    coordinate on the edit's own stored key (the library's null_repair, ridge 0); then the default two-block calibration runs
    unchanged on the n calibration queries.  Everything else is the default method (role, top-1, REPL, rebased values, guards).
    n_A (alias of n_a): the number of benign queries; n_A = 0 is the default method, bit for bit.  null_keys=<(n_A, d) keys>: the
    null-repair keys given explicitly (fx.methods.null_keys(L, draw) returns the keys this constructor uses for a draw);
    null_rows=<sequences>: an explicit sample of sequences."""
    kw.setdefault("label", "rescope_null")
    if n_A is not None:
        n_a = int(n_A)
    if kw.get("null_keys") is not None or kw.get("null_rows") is not None:
        n_a = 0
    return scoped(L, "rescope_null", null_n=n_a, null_ridge=ridge, **kw)


def null_rows(L, draw: int = 0, n_A: int = NULL_N, n: int = D.N_DEFAULT, mask=None) -> list:
    """The n_A null-repair sequences of ReScope-N for a calibration draw (fx.draws.null_sample: seeded, from the valid REPAIR rows,
    no QUERY in common with the n calibration queries of the same draw index)."""
    return D.null_sample(_log(L), int(n_A), int(draw), int(n), "repair", mask=mask)


def null_keys(L, draw: int = 0, n_A: int = NULL_N, n: int = D.N_DEFAULT, mask=None):
    """(keys (n_A, d) float32, rows): the teacher-forced last-prompt keys of the null-repair sample of a draw -- what rescope_null
    clears from the detectors, and what a gate of another family needs for the same construction (cos_null of R3)."""
    L = _log(L)
    rows = null_rows(L, draw, n_A, n, mask)
    return L.kc.keys("tf", rows), rows


def rescope_pac(L, delta: float = 0.1, **kw) -> Method:
    """Registered variant rescope_pac (amendment R9): the default method with the calibration-conditional rank of the block-2
    quantile, the smallest k with P[Beta(n2 + 1 - k, k) <= alpha] >= 1 - delta (pac_rank; delta = 0.1).  Gate level on all logs
    (R19: no behaviour beyond the development logs' draws 0 - 2 was registered, and R19 (a) keeps it at gate level)."""
    kw.setdefault("label", "rescope_pac")
    return scoped(L, "rescope_pac", calibration="pac", delta=delta, **kw)


def rescope_logonly(L, least=None, **kw) -> Method:
    """Log-only start of the evidence curve (POOL(q = 1) of P11); least = sequences of received benign queries to repair on top."""
    kw.setdefault("label", "rescope_logonly")
    M = scoped(L, "rescope_logonly", calibration="log_only", least=least, **kw)
    if least is not None:
        M.n = len(least)
    return M


def evidence(L, n: int, draw: int = 0, alpha: float = 0.05, **kw) -> Method:
    """ReScope on n benign queries along the evidence curve of E2(c) (amendment of 22:10, item 2): n = 0 the log-only rule; 0 < n < 40
    the log-only rule + least repair of the n received queries (the nested draw); n >= 40 the two-block rule at alpha."""
    L = _log(L)
    if n >= N_SWITCH:
        return rescope(L, alpha=alpha, n=n, draw=draw, label=f"rescope_n{n}", **kw)
    rows = D.calibration(L, n, draw).rows if n > 0 else None
    M = rescope_logonly(L, least=rows, label=f"rescope_n{n}", **kw)
    M.n, M.draw = int(n), int(draw)
    M.rec["evidence"] = {"n": int(n), "draw": int(draw), "rule": "log-only + least repair (n < 40)"}
    M.rec["fingerprint"] = _fp(M.rec, M.obj)
    return M


def cosine(L, calibration: str = "two_block", role: str = "last", gamma: float = COS_DEFAULT, rebase: bool = True, label: str | None = None, **kw) -> Method:
    """NeuralDB-style cosine gate.  calibration='default': the published threshold gamma = 0.65 (library cosine_gate; no own-prompt
    guards, as published); role='all': the published form (the gate acts at every token position)."""
    L = _log(L)
    if calibration in ("default", "fixed"):
        from rescope import cosine_gate
        V, vrec = released_values(L, rebase)
        g = _named(cosine_gate(L.log, float(gamma), role=role, values=V), "rebased" if rebase else "stored")
        rec = {"name": "cosine", "log": L.name, "T": L.T, "feature": "cos", "role": role, "fire": "top1", "response": "repl",
               "values": vrec, "calibration": "default", "gamma": float(gamma),
               "source_of_default": "NeuralDB (arXiv 2507.18028): gamma = 0.65", "guards": "none (fixed threshold)", "built_on": "cpu"}
        lab = label or ("cosine_default" if role == "last" else "cosine_published")
        return _finish(Method("cosine", lab, g, "scoped" if role == "last" else "gate", L.name, rec, variant=f"gamma{gamma:g}/{role}"), L)
    lab = label or ("cosine" if (calibration == "two_block" and role == "last") else f"cosine_{calibration}" + ("" if role == "last" else f"_{role}"))
    return scoped(L, "cosine", label=lab, feature="cos", role=role, calibration=calibration, rebase=rebase, **kw)


def exact(L, role: str = "last", rebase: bool = True, label: str = "exact") -> Method:
    L = _log(L)
    from rescope import exact_gate
    V, vrec = released_values(L, rebase)
    g = _named(exact_gate(L.log, role=role, values=V), "rebased" if rebase else "stored")
    rec = {"name": "exact", "log": L.name, "T": L.T, "feature": "cos", "role": role, "fire": "top1", "response": "repl", "calibration": "default",
           "gamma": EXACT_GAMMA, "values": vrec, "guards": "none (fixed threshold)", "built_on": "cpu"}
    return _finish(Method("exact", label, g, "scoped" if role == "last" else "gate", L.name, rec), L)


def ball(L, calibration: str = "two_block", radius: float | None = None, role: str = "last", rebase: bool = True, label: str | None = None, **kw) -> Method:
    """GRACE-style gate.  calibration='default': one global radius eps_init = 1.0 (published); 'half_min': half the smallest distance
    between two stored keys (GRACE's split rule applied to the log's closest pair); radius=<number>: that radius."""
    L = _log(L)
    if calibration in ("default", "half_min", "fixed") or radius is not None:
        from rescope import ball_gate, key_distances
        if radius is None:
            radius = BALL_DEFAULT if calibration in ("default", "fixed") else 0.5 * float(key_distances(L.log).min())
        V, vrec = released_values(L, rebase)
        g = _named(ball_gate(L.log, float(radius), role=role, values=V), "rebased" if rebase else "stored")
        rec = {"name": "ball", "log": L.name, "T": L.T, "feature": "negdist", "role": role, "fire": "top1", "response": "repl", "calibration": calibration,
               "radius": float(radius), "source_of_default": "GRACE (arXiv 2211.11031): eps_init = 1.0 (SCOTUS, Hallucination), 0.5 (zsRE)",
               "smallest_distance_between_stored_keys": float(key_distances(L.log).min()), "values": vrec,
               "guards": "none (fixed threshold)", "built_on": "cpu"}
        lab = label or ("ball_default" if calibration in ("default", "fixed") else f"ball_{calibration}")
        return _finish(Method("ball", lab, g, "scoped" if role == "last" else "gate", L.name, rec, variant=f"radius{radius:g}"), L)
    lab = label or ("ball" if calibration == "two_block" else f"ball_{calibration}")
    return scoped(L, "ball", label=lab, feature="negdist", role=role, calibration=calibration, rebase=rebase, **kw)


# ----------------------------------------------------------------------------- Rescope-WHERE
def _where_base(L):
    S, grec = base_gate(L, "ncoord", "all", "all", "lin", True, True)
    return S, grec


def where_logonly(L, H=None, margin: float = SELF_MARGIN, allow_silence: bool = False, label: str = "where_logonly") -> Method:
    """Rescope-WHERE with the log-only rule (see the module docstring).  Needs the model the first time (cached per log afterwards)."""
    L = _log(L)
    S, grec = _where_base(L)
    S2, crec = _cal_self(L, S, margin, allow_silence, H)
    rec = {"name": "where_logonly", "log": L.name, "T": L.T, "feature": "ncoord", "role": "all", "fire": "all", "response": "lin",
           "values": "the log's stored residuals U_i (response LIN: <W_i, k> U_i), no round-off term E", "calibration": "self", "cal": crec,
           "guards": "stored solver keys + B = 1 keys of the own Rel prompts", "add_guards": grec, "built_on": "cpu"}
    return _finish(Method("where_logonly", label, S2, "gate", L.name, rec), L)


def where_least(L, complaints, H=None, on_conflict: str = "refuse", start: str = "deployed", label: str | None = None, seed=None) -> Method:
    """Rescope-WHERE + least repair of reported complaints.  complaints: sequences of the key cache (fx.draws.complaints(L, n, seed));
    start='deployed' (all thresholds -inf: the deployed log up to its round-off term) | 'logonly'.  Each complaint is one call of the
    library's least_repair on the keys at EVERY position of its teacher-forced sequence (model: one truncated forward per complaint);
    on_conflict='refuse' (library default: a complaint whose repair would silence an own key is refused, counted) | 'partial' | 'force'."""
    from rescope import least_repair
    L = _log(L)
    ses_m = _session(L, H)
    if start == "deployed":
        S, grec = _where_base(L)
        crec = {"op": "none (all thresholds -inf: the deployed log without its round-off term)"}
    elif start == "logonly":
        S, grec = _where_base(L)
        S, crec = _cal_self(L, S, SELF_MARGIN, False, H)
    else:
        raise ValueError("start: 'deployed' | 'logonly'")
    Sg = S.to(H.dev)                                                     # complaints are scored where the hook will score them
    recs = []
    with torch.no_grad(), ses_m.ev.hz.HE.vision_cache(H):
        for j in complaints:                                            # one complaint at a time (the library's Session.complain, keys not kept)
            ses_m.ev.trim_vcache()
            keys = ses_m.ev.eligible_keys(ses_m.item(int(j)), Sg)
            new, r = least_repair(Sg, keys, on_conflict=on_conflict)
            recs.append({k: v for k, v in r.items() if k != "raised"})
            Sg = Sg if r["refused"] else new
            del keys
    S2 = Sg.to("cpu")
    n = len(recs)
    rec = {"name": "where_least", "log": L.name, "T": L.T, "feature": "ncoord", "role": "all", "fire": "all", "response": "lin", "start": start, "cal": crec,
           "guards": "stored solver keys + B = 1 keys of the own Rel prompts", "add_guards": grec, "built_on": "cpu, complaints scored on the model's device",
           "least_repair": {"n_complaints": n, "seed": seed, "on_conflict": on_conflict, "n_refused": sum(r["refused"] for r in recs),
                            "n_with_conflict": sum(bool(r["conflicts"]) for r in recs), "n_resolved": sum(bool(r.get("resolved")) for r in recs),
                            "n_keys": sum(r["n_keys"] for r in recs), "n_thresholds_raised_total": sum(r["n_raised"] for r in recs),
                            "margin": recs[0]["margin"] if recs else None, "abs_margin": recs[0]["abs_margin"] if recs else None,
                            "sids": [L.sid(int(j)) for j in complaints],
                            "refused_sids": [L.sid(int(j)) for j, r in zip(complaints, recs) if r["refused"]]}}
    M = Method("where_least", label or f"where_least_n{n}", S2, "gate", L.name, rec, n=n, draw=seed, variant=f"{start}/{on_conflict}")
    return _finish(M, L)


# ----------------------------------------------------------------------------- dense gates (need the model's weights)
def _dense_scores(L, H, gate, dr: D.Draw):
    """One scalar score per calibration query: router = activation at the last prompt position (teacher-forced key); z-score = the
    largest z-score of the teacher-forced sequence (one truncated forward per query, cached per log: fx.behave Session.zmax)."""
    if gate.kind == "router":
        return gate.activation((L.kc if getattr(dr, "kc", None) is None else dr.kc).keys(dr.which if dr.which != "steps" else "decode", dr.rows)).cpu().numpy()
    sc = _session(L, H).zmax(dr.rows)                                   # cached per log and sequence (it does not depend on the threshold)
    if np.isnan(sc).any():
        raise ValueError("a calibration query has no finite z-score")
    return sc


def _dense(L, H, kind: str, calibration: str, alpha, n, draw, label, source="repair", which="tf", **default_kw) -> Method:
    from rescope import calibrate_dense, coachhook_alpha, router_gate, zscore_gate
    L = _log(L)
    label = label or (kind if calibration == "conformal" else f"{kind}_{calibration}")
    if H is None:
        return Method(kind, label, None, "unavailable", L.name, {"name": kind, "log": L.name, "unavailable": "needs the model's weights (W_E - W_0): pass H"})
    rec = {"name": kind, "log": L.name, "T": L.T, "calibration": calibration, "role": "last+answer" if kind == "router" else "all",
           "response": "the deployed output W_E k at the selected positions", "built_on": str(H.dev)}
    if kind == "router":
        g = router_gate(L.log.to(H.dev), H.WE, H.W0, q=default_kw.get("q", ROUTER_DEFAULT_Q))
        rec["default"] = {"rule": "eps = q-quantile of the edits' own activation on the stored solver keys", "q": default_kw.get("q", ROUTER_DEFAULT_Q),
                          "eps": float(g.thr), "source": "WISE (arXiv 2405.14768): eps = the smallest activation within the edits"}
    else:
        a = float(default_kw.get("alpha_z", ZSCORE_DEFAULT))
        g = zscore_gate(H.WE, H.W0, alpha=a)
        rec["default"] = {"alpha_z": a, "source": "CoachHooK (arXiv 2403.05330): alpha_z = 2.2"}
        if calibration == "dynamic":
            mz = [float(z) for z in _session(L, H).zmax(L.edit_rows["rel"])]
            a_dyn = coachhook_alpha(mz, step=[int(b) for b in L.log.batch], alpha_z=a)
            g = zscore_gate(H.WE, H.W0, alpha=a_dyn)
            rec["dynamic"] = {"alpha": a_dyn, "rule": "Eq. 19 on the edits' own Rel prompts (teacher-forced, B = 1), steps = the log's batches",
                              "max_z": _quant(mz)}
    M = Method(kind, label, g, "dense", L.name, rec)
    if calibration in ("conformal", "two_block"):
        dr = D.calibration(L, n, draw, source, which)
        sc = _dense_scores(L, H, g, dr)
        g2, crec = calibrate_dense(g, sc, alpha, two_block=(calibration == "two_block"), seed=dr.seed)
        M.obj, M.alpha, M.n, M.draw = g2, alpha, dr.n, dr.draw
        rec.update(cal=crec, draw=dr.rec, scores=_quant(sc))
    elif calibration not in ("default", "dynamic"):
        raise ValueError("calibration: 'conformal' | 'two_block' | 'default'" + (" | 'dynamic'" if kind == "zscore" else ""))
    return _finish(M, L)


def router(L, H=None, calibration: str = "conformal", alpha: float = 0.05, n: int = D.N_DEFAULT, draw: int = 0, label: str | None = None, **kw) -> Method:
    return _dense(L, H, "router", calibration, alpha, n, draw, label, **kw)


def zscore(L, H=None, calibration: str = "conformal", alpha: float = 0.05, n: int = D.N_DEFAULT, draw: int = 0, label: str | None = None, **kw) -> Method:
    return _dense(L, H, "zscore", calibration, alpha, n, draw, label, **kw)


# ----------------------------------------------------------------------------- weights, DUAL, linear repairs
def deployed(L, label: str = "deployed") -> Method:
    L = _log(L)
    rec = {"name": "deployed", "log": L.name, "T": L.T, "what": "the deployed log: module weight W_E"}
    return _finish(Method("deployed", label, None, "deployed", L.name, rec), L)


def preedit(L, label: str = "preedit") -> Method:
    L = _log(L)
    rec = {"name": "preedit", "log": L.name, "T": L.T, "what": "the pre-edit model: module weight W_0 (control)"}
    return _finish(Method("preedit", label, "W0", "preedit", L.name, rec), L)


def dual(L, variant: str = "eq", on: str = "line", label: str | None = None) -> Method:
    """DUAL (answer match): the deployed answer iff it equals (variant 'eq') / contains ('contains') the target of a logged edit, else
    the pre-edit answer.  Free generation only (fx.behave.generate / d2 / hard_negatives(generate=True))."""
    from rescope import Dual
    L = _log(L)
    g = Dual(L.targets, variant, on)
    rec = {"name": "dual", "log": L.name, "T": L.T, "variant": variant, "on": on, "n_targets": L.T, "evaluated_by": "generation (two decodes per query)"}
    return _finish(Method("dual", label or ("dual" if variant == "eq" else f"dual_{variant}"), g, "dual", L.name, rec, variant=variant), L)


def p12(L, H=None, method: str = "COMP_B", n="ALL", seed: int = 0, label: str | None = None) -> Method:
    """A linear repair that P12 saved (out/<log>/corr/<METHOD>__n<n>__s<seed>.pt): the harness Correction, evaluated by the library."""
    L = _log(L)
    name = f"{method}__n{n}__s{seed}"
    p = os.path.join(fx.P12, "out", L.name, "corr", name + ".pt")
    lab = label or f"p12_{method.lower()}_n{str(n).lower()}"
    rec = {"name": "p12", "log": L.name, "config": name, "file": p, "what": "linear repair of P12 (saved correction)"}
    if not os.path.exists(p) or H is None:
        rec["unavailable"] = "no saved correction for this log" if not os.path.exists(p) else "needs the model: pass H"
        return Method("p12", lab, None, "unavailable", L.name, rec)
    from rescope import harness
    Correction = harness().HL.Correction
    b = torch.load(p, map_location="cpu")
    kind = "correction"
    if b["kind"] == "adapter":
        c = Correction("adapter", Lf=b["Lf"].to(H.dev), Rf=b["Rf"].to(H.dev), gate=b["gate"], meta={"p12": name})
    elif b["kind"] == "weight":                                         # a full replacement weight: a plain tensor is a method (memoised by its signature)
        if b["W"].dtype != H.W0.dtype or b["W"].shape != H.W0.shape:
            raise RuntimeError(f"{p}: saved weight {b['W'].dtype} {tuple(b['W'].shape)}, module {H.W0.dtype} {tuple(H.W0.shape)}")
        c, kind = b["W"].to(H.dev), "weight"
    else:
        c = Correction("none", meta={"p12": name})
    rj = os.path.join(fx.P12, "out", L.name, name + ".json")
    if os.path.exists(rj):
        r = fx.io.jload(rj)
        rec.update(n_reported=r.get("n_reported"), options=r.get("options"), reported_groups_sha1=hashlib.sha1(json.dumps(r.get("reported_groups")).encode()).hexdigest()[:12])
    rec["correction_kind"] = b["kind"]
    return _finish(Method("p12", lab, c, kind, L.name, rec, n=None if n == "ALL" else int(n), draw=seed, variant=name), L)


def recalibrate(L, S, alpha: float = 0.05, n: int = D.N_DEFAULT, draw: int = 0, exclude=None, rule: str = "two_block", split=(1, 1), baseline="max",
                allow_silence: bool = False, source: str = "repair", which: str = "tf", cal=None, baseline_x=None, pac_delta=None, mask=None):
    """Calibrate ANY scoped log S of this log (role 'last'; e.g. after null_repair / witness_repair changed its detectors) on the
    calibration draw (n, draw) of the REPAIR pool, leaving out the sequences `exclude` (the reported complaints: the calibration sample
    stays disjoint from them, amendment of 22:10, item 3).  rule: 'two_block' (default) | 'shared'.  Thresholds, eta and ranking of S
    are replaced (the library's calibrate_two_block / calibrate_shared; guards of S keep firing).  -> (scoped log, record); pass the
    result to fx.methods.wrap.  With the un-thresholded base gate and no exclusion this is the 'rescope' method bit for bit.
    cal: an explicit sample instead (a fx.draws.Draw).  After detector repairs driven by complaints the bound needs a sample that was
    never looked at before (amendment of 2026-11-11 01:50): C0, S_, C1 = fx.draws.partition(L, (160, 160, 160), draw); calibrate the
    starting thresholds on C0 (fx.methods.rescope(L, cal=C0)), take complaints from S_, recalibrate with cal=C1.
    baseline_x / pac_delta: the external baselines of R5 / the calibration-conditional rank of R9 in the two-block rule (as in
    scoped); mask: the mask of the draw.  This is how a gate of a feature package is calibrated like the registered gates:
    S = its un-thresholded scoped log AFTER add_guards, then fx.methods.wrap."""
    L = _log(L)
    dr = cal if cal is not None else D.calibration(L, n, draw, source, which, exclude=exclude, mask=mask)
    Q = calib_coords(L, S, dr)
    if rule == "two_block" and (baseline_x is not None or pac_delta is not None):
        if baseline != "max":
            raise ValueError("baseline_x / pac_delta take the block-1 maximum as baseline")
        S2, rec = _cal_two_block_ext(S, Q, alpha, dr.seed, split, allow_silence, "excess", baseline_x, pac_delta)
    elif rule == "two_block":
        S2, rec = _cal_two_block(S, Q, alpha, dr.seed, split, baseline, allow_silence, "excess")
    elif rule == "shared":
        from rescope import calibrate_shared
        S2, rec = calibrate_shared(S, coords=Q, alpha=alpha, allow_silence=allow_silence)
    else:
        raise ValueError("rule: 'two_block' | 'shared'")
    return S2, {"op": "recalibrate", "rule": rule, "alpha": alpha, "draw": dr.rec, "cal": rec}


def wrap(L, obj, label: str, rec: dict | None = None, name: str = "custom", alpha=None, n=None, draw=None, variant: str = "") -> Method:
    """A Method around any object the library's Evaluator accepts (e.g. a scoped log after library operators: least_repair, null_repair,
    witness_repair, scope_cut applied to another Method's .obj), so that it runs through fx.gate / fx.behave / the memo like the named
    methods.  rec: what determined it (the operators' records); the thresholds summary and the fingerprint are added here."""
    from rescope import Dual, ScopedLog
    L = _log(L)
    if obj is None:
        kind = "deployed"
    elif isinstance(obj, ScopedLog):
        kind = "scoped" if obj.role == "last" else "gate"
    elif isinstance(obj, Dual):
        kind = "dual"
    elif torch.is_tensor(obj):
        kind = "weight"
    elif hasattr(obj, "firing"):
        kind = "dense"
    elif hasattr(obj, "kind") and hasattr(obj, "meta"):
        kind = "correction"
    else:
        raise TypeError(f"not a method of the library's Evaluator: {type(obj).__name__}")
    r = dict(rec or {}, name=name, log=L.name, T=L.T, wrapped=True)
    return _finish(Method(name, label, obj, kind, L.name, r, alpha=alpha, n=n, draw=draw, variant=variant), L)


# ----------------------------------------------------------------------------- registry
BUILDERS = {"deployed": lambda L, H=None, **kw: deployed(L, **kw), "preedit": lambda L, H=None, **kw: preedit(L, **kw),
            "rescope": lambda L, H=None, **kw: rescope(L, H=H, **kw), "rescope_shared": lambda L, H=None, **kw: rescope_shared(L, H=H, **kw),
            "rescope_null": lambda L, H=None, **kw: rescope_null(L, H=H, **kw),
            "rescope_pac": lambda L, H=None, **kw: rescope_pac(L, H=H, **kw),
            "rescope_logonly": lambda L, H=None, **kw: rescope_logonly(L, H=H, **kw), "where_logonly": where_logonly, "where_least": where_least,
            "cosine": lambda L, H=None, **kw: cosine(L, H=H, **kw) if kw.get("calibration", "two_block") not in ("default", "fixed") else cosine(L, **kw),
            "ball": lambda L, H=None, **kw: ball(L, H=H, **kw) if (kw.get("calibration", "two_block") not in ("default", "half_min", "fixed") and kw.get("radius") is None) else ball(L, **kw),
            "exact": lambda L, H=None, **kw: exact(L, **kw), "router": router, "zscore": zscore, "dual": lambda L, H=None, **kw: dual(L, **kw), "p12": p12}
NAMES = tuple(BUILDERS)


def build(name: str, L, H=None, **kw) -> Method:
    """Construct a method by name (module docstring).  L: fx.logs.LogData; H: the harness handle where the method needs the model."""
    if name not in BUILDERS:
        from . import review1                                           # stage 2: the gates registered after the review (fx.review1.NAMES)
        if review1.handles(name):
            return review1.build(name, _log(L), H=H, **kw)
        raise KeyError(f"method {name!r}: one of {NAMES} or of fx.review1.NAMES")
    return BUILDERS[name](_log(L), H=H, **kw)


def _log(L):
    """A LogData, or the name of a registered log."""
    from . import logs
    return logs.open(L) if isinstance(L, str) else L


E1_LABELS = ("deployed", "preedit", "rescope", "rescope_shared", "rescope_null", "where_logonly", "cosine", "cosine_shared", "cosine_default", "cosine_published",
             "ball", "ball_default", "exact", "router", "router_default", "zscore", "zscore_default", "dual",
             "p12_comp_b_nall", "p12_reedit_b_nall", "p12_rb_union_nall", "p12_rb_tot_nall")
# E3 (scale, unseen logs): LINEAR, ReScope(0.05) with its two registered companions, COSINE(0.05), COSINE(default), Rescope-WHERE(log-only).
# The answer-match baseline DUAL was part of this row set until amendment R19 (2026-11-11 08:49) cut it on unseen logs: E3_LABELS + ("dual",).
E3_LABELS = ("deployed", "rescope", "rescope_shared", "rescope_null", "cosine", "cosine_default", "where_logonly")
E1_DEPENDS = {k: ("alpha", "draw") for k in ("rescope", "rescope_shared", "rescope_null", "cosine", "cosine_shared", "ball", "router", "zscore", "rescope_pac")}
E1_GATE_LEVEL = ("rescope_pac",)          # registered variants reported at gate level only (R19 a): e1_rowset(labels=E1_LABELS + E1_GATE_LEVEL)


def e1_rowset(L, alpha: float = 0.05, draw: int = 0, n: int = D.N_DEFAULT, H=None, labels=None, skip_unavailable: bool = True, **switches) -> list:
    """The fixed list of methods of experiment E1 for (log, alpha, calibration draw), in table order (labels: E1_LABELS):
      deployed, preedit (control); rescope = ReScope(alpha), the default method; rescope_shared = the shared-scale calibration (kept next
      to the default in every table: amendment of 2026-11-11 00:55); rescope_null = the registered variant ReScope-N (amendment of
      2026-11-11 01:35; reported next to the default on every log); where_logonly = Rescope-WHERE(log-only);
      cosine = COSINE(alpha) (two-block, role 'last'), cosine_shared (shared scale), cosine_default = 0.65 at the last prompt position (the
      retrofit), cosine_published = 0.65 at every position (the published form); ball = BALL(alpha) (two-block), ball_default = radius 1.0; exact;
      router = ROUTER(alpha), router_default; zscore = ZSCORE(alpha), zscore_default (one conformal threshold on their scalar score);
      dual; p12_comp_b_nall / p12_reedit_b_nall / p12_rb_union_nall / p12_rb_tot_nall = the linear repairs of P12 on ALL reported complaints
      (seed 0; P12's saved corrections, evaluated here; they exist for the development logs only and are skipped elsewhere).
    Only the rows of E1_DEPENDS (the calibrated ones) depend on (alpha, draw): the other rows are the same object for every
    alpha and draw (same Method.key).  H=None: the methods that need the model at construction (where_logonly unless cached, router,
    zscore, p12) are returned as unavailable, or left out (skip_unavailable).  labels: a subset (E3: fx.methods.E3_LABELS).
    **switches: passed to the calibrated scoped rows (rescope, rescope_shared, rescope_null, cosine, cosine_shared, ball), e.g.
    which="all", guards="all" for the decode-proof calibration, or source="benign"; they appear in Method.variant."""
    L = _log(L)
    want = E1_LABELS if labels is None else tuple(labels)
    mk = {"deployed": lambda: deployed(L), "preedit": lambda: preedit(L), "rescope": lambda: rescope(L, alpha=alpha, n=n, draw=draw, **switches),
          "rescope_shared": lambda: rescope_shared(L, alpha=alpha, n=n, draw=draw, **switches),
          "rescope_null": lambda: rescope_null(L, alpha=alpha, n=n, draw=draw, **switches),
          "rescope_pac": lambda: rescope_pac(L, alpha=alpha, n=n, draw=draw, **switches),
          "cosine_shared": lambda: cosine(L, "shared", alpha=alpha, n=n, draw=draw, **switches),
          "where_logonly": lambda: _try(lambda: where_logonly(L, H), L, "where_logonly"),
          "cosine": lambda: cosine(L, "two_block", alpha=alpha, n=n, draw=draw, **switches), "cosine_default": lambda: cosine(L, "default"),
          "cosine_published": lambda: cosine(L, "default", role="all"), "ball": lambda: ball(L, "two_block", alpha=alpha, n=n, draw=draw, **switches),
          "ball_default": lambda: ball(L, "default"), "exact": lambda: exact(L),
          "router": lambda: router(L, H, "conformal", alpha, n, draw), "router_default": lambda: router(L, H, "default"),
          "zscore": lambda: zscore(L, H, "conformal", alpha, n, draw), "zscore_default": lambda: zscore(L, H, "default"),
          "dual": lambda: dual(L), "p12_comp_b_nall": lambda: p12(L, H, "COMP_B", "ALL", 0), "p12_reedit_b_nall": lambda: p12(L, H, "REEDIT_B", "ALL", 0),
          "p12_rb_union_nall": lambda: p12(L, H, "RB_UNION", "ALL", 0), "p12_rb_tot_nall": lambda: p12(L, H, "RB_TOT", "ALL", 0)}
    out = []
    for lab in want:
        if lab not in mk:
            raise KeyError(f"{lab!r} is not a row of E1 ({E1_LABELS})")
        M = mk[lab]()
        if M.available or not skip_unavailable:
            out.append(M)
    return out


def _try(fn, L, label):
    try:
        return fn()
    except RuntimeError as e:
        if "needs the model" not in str(e):
            raise
        return Method(label, label, None, "unavailable", L.name, {"name": label, "log": L.name, "unavailable": str(e)})


# ----------------------------------------------------------------------------- maintenance of a scoped log (E9)
def mask_edits(L, M, edits, label: str | None = None) -> Method:
    """Selective revocation in a scoped log (E9 a): the method M with the edits `edits` MASKED -- every detector of a masked edit
    (its primary detector and any detector added for it) gets the threshold +infinity, and the masked edits' guards are dropped,
    so that no later guard rule can re-open them; detectors, all other thresholds, eta, ranking baselines and released values are
    the same tensors / bits as in M (raise-only: the false-firing bound of M still holds).  A masked edit is never a candidate,
    hence never released, on any key.  Masking the empty set returns M's own object (same digest).
    M: a Method around a scoped log (any role: the default method, a gate of the cosine family, Rescope-WHERE)."""
    L = _log(L)
    g = M.obj
    if not hasattr(g, "det_tau"):
        raise TypeError("mask_edits: a method around a scoped log")
    R = sorted({int(i) for i in edits})
    if R and (R[0] < 0 or R[-1] >= g.T):
        raise ValueError(f"edits out of range (T = {g.T})")
    if R:
        sel = torch.isin(g.det_edit, torch.tensor(R, dtype=torch.long, device=g.det_edit.device))
        tau = g.det_tau.clone()
        tau[sel] = float("inf")
        keep = ~torch.isin(g.guard_edit, torch.tensor(R, dtype=torch.long, device=g.guard_edit.device))
        g2 = g.replace(det_tau=tau, guard_key=g.guard_key[keep], guard_edit=g.guard_edit[keep], meta=dict(g.meta, masked_edits=R))
    else:
        g2 = g
    rec = {k: v for k, v in M.rec.items() if k not in ("fingerprint", "thresholds", "own_prompts", "gate", "baselines")}
    rec.update(masked={"edits": R, "n": len(R), "of": M.key, "of_digest": M.digest, "rule": "thresholds of every detector of a masked edit = +inf; their guards dropped; "
                       "everything else unchanged (raise-only)"})
    lab = label or (M.label if not R else f"{M.label}-mask{len(R)}")
    M2 = Method(M.name, lab, g2, M.kind, L.name, rec, alpha=M.alpha, n=M.n, draw=M.draw, variant=",".join(v for v in (M.variant, f"mask={len(R)}" if R else "") if v))
    return _finish(M2, L)


class _Deltas:
    """The applied batch deltas of a testbed log read from disk on demand (H.A of a handle loaded without them)."""

    def __init__(self, tl, n):
        self.tl, self.n = tl, n

    def __len__(self):
        return self.n

    def __getitem__(self, b):
        return self.tl.batch_delta(int(b))


def rollback(L, H, edits, label: str | None = None) -> Method:
    """The ROLLBACK weight of the linear log for a set of edits, by the shared harness's convention (07_repair_v3/harness/hlog.py,
    HLog.rollback_weight): W(R) = (W_0.float() + (C - X(R))).to(dtype), C = W_E.float() - W_0.float(), X(R) = the exact applied
    batch deltas of the batches that lie entirely in R + the rank-one solver terms outer(U_i, W_i) (fp64 -> fp32) of the other
    edits of R; no re-solve.  W(empty) is W_E bitwise, W(all edits) is W_0.  -> a Method of kind 'weight': it runs through the same
    evaluation functions as the deployed weight (fx.behave: memoised under the sha1 of the weight; ses.tf_hashed for logit
    hashes).  H: the handle of the log (fx.logs.load_model); the applied batch deltas are read from disk when the handle was
    loaded without them."""
    L = _log(L)
    if H is None:
        raise RuntimeError("rollback needs the model handle: H = fx.logs.load_model(name)")
    R = sorted({int(i) for i in edits})
    if R and (R[0] < 0 or R[-1] >= L.T):
        raise ValueError(f"edits out of range (T = {L.T})")
    if getattr(H, "A", None) is None:
        H.A = _Deltas(H.tl, len(H.batches))
    W = H.rollback_weight(R)
    whole = [b for b, ids in enumerate(H.batches) if ids and all(i in set(R) for i in ids)]
    rec = {"name": "rollback", "log": L.name, "T": L.T, "edits": R, "n": len(R), "whole_batches": whole,
           "rule": "harness HLog.rollback_weight: applied batch deltas of whole batches + rank-one solver terms of the other edits, subtracted from W_E - W_0 (fp32)",
           "equals_deployed_weight": bool(torch.equal(W, H.WE)), "equals_pre_edit_weight": bool(torch.equal(W, H.W0))}
    return _finish(Method("rollback", label or f"rollback{len(R)}", W, "weight", L.name, rec, n=len(R), variant=f"R={len(R)}"), L)
