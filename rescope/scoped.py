"""Scoped replay of an edit log: the scoped log, its gate, and the forward hook (the ONE code path).

SCOPED LOG  S = (W_0; detectors (w_j, s_j, tau_j) -> edit e(j); values v_i; eta; role; fire; response)

  normalised coordinate   r_j(k) = <w_j, k> / s_j                    (kind 'lin'; s_j > 0, r_j = 1 at the detector's key)
                          r_j(k) = <w_j, k> / (s_j |k|)              (kind 'cos': w_j a unit vector, s_j = 1 -> raw cosine)
                          r_j(k) = -|k - key_j|                      (kind 'ball': minus the Euclidean distance)
  eligible positions      role 'last'        : the last prompt position of a sequence only
                          role 'last+answer' : the last prompt position and every later position
                          role 'all'         : every position
  candidates at k         detectors with r_j(k) >= eta * tau_j (non-strict; fp64; nothing can fire if eta = +inf).
                          An edit is a candidate if any of its detectors is (the first T detectors are the
                          edits' primary detectors, further detectors are added by why-not repairs).
  fire 'top1'             only the candidate detector with the largest r_j releases (its edit e(j)); an exact tie
                          goes to the LOWEST detector index (the earlier edit).  If the log carries ranking
                          baselines b_j / scales u_j (`det_b`, `det_u`; set by calibrate.calibrate_two_block) the
                          candidates are ranked by the margin q_j = (r_j - b_j) / u_j instead (= the excess
                          (r_j - b_j - eta u_j) / u_j over the calibrated level up to the constant eta); the
                          default b = 0, u = 1 is the ranking by r_j;
       'all'              every candidate edit releases.
  response 'repl'         out = W_0 k + v_e                          (fire 'all': sum over the releasing edits)
           'gain'         out = W_0 k + min(1, r_j) v_e              (r_j of the releasing detector; 'all': per edit the
                                                                      largest r among its candidate detectors)
           'lin'          out = W_0 k + sum_{releasing e} <W_e, k> U_e   (the ORIGINAL log term: read-out vector W_e
                          and stored residual U_e of the log, whatever the detectors are; with tau = -inf,
                          role 'all', fire 'all' this is the linear log without the write round-off E)
  no candidate            the module's own output tensor is returned UNTOUCHED (branch, no zero is added): a
                          sequence on which nothing fires is bitwise the pre-edit model.

Arithmetic: detectors, thresholds, values and the response are fp64; at a releasing position the module's own
W_0 output (module dtype) is cast to fp64, the value is added, and the sum is cast ONCE to the module dtype.
`precision='fp32'` / `'module'` compute the inner products <w_j, k> in fp32 / in the module dtype instead (cheap
gates, NOT the reference: they can change decisions near a threshold; bf16 / fp16 products are only accurate to
about 1e-2 relative).
Batches: role 'last' uses the last prompt index of every row (ONE index per row is required: a vector of another
length is an error).  Roles 'last+answer' and 'all' also evaluate the PAD positions of a padded batch (harmless
for a causal model with right padding, but they count in the firing flags) unless the hook is told the row
lengths (Hook.begin(lp, seq_len=...): positions >= seq_len never fire).  The evaluation code uses B = 1 everywhere.

"The last prompt position" is whatever index the caller passes: the evaluation passes the harness index of the
sequence it runs (teacher-forced sequence: answer_start - 1; decode: the last token of the prompt alone).  These
are the same token except where the tokenizer puts a bare space between prompt and answer (LLaVA, answers that
start with a digit; see evaluate.py, `lp_shift`).

The object is immutable by convention: every operator returns a new one (`replace`).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch

from .loglib import EditLog, rebase

ROLES = ("last", "last+answer", "all")
FIRE = ("top1", "all")
RESPONSES = ("repl", "gain", "lin")
LIN, COS, BALL = 0, 1, 2
KINDS = {"lin": LIN, "cos": COS, "ball": BALL}
TIE = 1e-9                     # relative slack of "the own key must fire": tau_j <= (1 - TIE) * r_j(own key)
INF = float("inf")
CHUNK = 4096


@dataclass
class Firing:
    """What fired in ONE call of the module (positions with at least one candidate)."""
    idx: torch.Tensor                      # (n,) flat index row * Tn + position into the call's (B * Tn) positions
    row: torch.Tensor                      # (n,) batch row
    pos: torch.Tensor                      # (n,) absolute sequence position
    edit: torch.Tensor                     # (n,) releasing edit of fire 'top1' (-1: no single edit)
    cand: torch.Tensor | None = None       # (n, D) bool candidate detectors          (None for dense gates)
    r: torch.Tensor | None = None          # (n, D) fp64 normalised coordinates
    top: torch.Tensor | None = None        # (n,) candidate detector with the largest r
    aux: dict = field(default_factory=dict)


@dataclass(eq=False)
class ScopedLog:
    det_w: torch.Tensor                    # (D, d) fp64 detectors
    det_s: torch.Tensor                    # (D,) fp64 scales > 0
    det_tau: torch.Tensor                  # (D,) fp64 thresholds on r (-inf: always a candidate, +inf: never)
    det_edit: torch.Tensor                 # (D,) long owning edit; det_edit[:T] == arange(T)
    det_kind: torch.Tensor                 # (D,) long LIN | COS | BALL
    det_key: torch.Tensor                  # (D, d) fp64 key of the detector (r = 1 there; centre of a ball)
    values: torch.Tensor                   # (T, m) fp64 released values v_i
    lin_w: torch.Tensor                    # (T, d) fp64 the log's read-out vectors W_i   (response 'lin')
    lin_u: torch.Tensor                    # (T, m) fp64 the log's stored residuals U_i   (response 'lin')
    guard_key: torch.Tensor                # (G, d) fp64 keys on which edit guard_edit[g] must stay a candidate
    guard_edit: torch.Tensor               # (G,) long   (initially: every edit's own stored solver key)
    eta: float = 1.0
    role: str = "last"
    fire: str = "top1"
    response: str = "repl"
    precision: str = "fp64"
    W0: torch.Tensor | None = None         # reference to the pre-edit module weight (module dtype); not copied
    name: str = ""
    meta: dict = field(default_factory=dict)
    det_b: torch.Tensor | None = None      # (D,) fp64 ranking baselines of fire 'top1' (None: 0)        } candidates are ranked by
    det_u: torch.Tensor | None = None      # (D,) fp64 ranking scales > 0 of fire 'top1' (None: 1)       } (r - det_b) / det_u

    def __post_init__(self):
        if self.role not in ROLES or self.fire not in FIRE or self.response not in RESPONSES:
            raise ValueError(f"role / fire / response: {self.role!r} / {self.fire!r} / {self.response!r}")
        for nm in ("det_b", "det_u"):
            t = getattr(self, nm)
            if t is not None and (t.shape != self.det_tau.shape or bool(torch.isnan(t).any())):
                raise ValueError(f"{nm} must be a (D,) tensor without NaN")
        if self.det_u is not None and not bool(((self.det_u > 0) & torch.isfinite(self.det_u)).all()):
            raise ValueError("ranking scales det_u must be finite and > 0")
        if self.precision not in ("fp64", "fp32", "module"):
            raise ValueError(self.precision)
        self.eta = float(self.eta)
        if not self.eta > 0:
            raise ValueError("eta must be > 0 (may be +inf)")
        D, T = self.det_w.shape[0], self.values.shape[0]
        ok = (self.det_s.shape == (D,) and self.det_tau.shape == (D,) and self.det_edit.shape == (D,) and self.det_kind.shape == (D,)
              and self.det_key.shape == self.det_w.shape and self.lin_w.shape == (T, self.det_w.shape[1])
              and self.lin_u.shape == self.values.shape and self.guard_key.shape[0] == self.guard_edit.shape[0] and D >= T)
        if not ok:
            raise ValueError("inconsistent scoped-log shapes")
        if not bool((self.det_s > 0).all()):
            raise ValueError("detector scales must be > 0")
        if not bool((self.det_edit[:T] == torch.arange(T, device=self.det_edit.device)).all()) or int(self.det_edit.max()) >= T:
            raise ValueError("the first T detectors must be the edits' primary detectors, in edit order")
        if self.response == "gain" and bool((self.det_kind == BALL).any()):
            raise ValueError("response 'gain' is undefined for 'ball' detectors (r = minus a distance)")
        self._cache = {}

    # ------------------------------------------------------------------ construction
    @staticmethod
    def from_log(log: EditLog, role="last", fire="top1", response="repl", tau=-INF, values="rebased", W0=None, name=None):
        """Detectors = the log's read-out vectors W_i, s_i = <W_i, k_i>; values = rebased residuals (default),
        'stored' (U_i) or an explicit (T, m) tensor; tau: scalar or (T,) thresholds on r (default: no threshold)."""
        s = log.own()
        if not bool((s > 0).all()):
            raise ValueError("an edit has a non-positive own coordinate s_i = <W_i, k_i>")
        if isinstance(values, str) and values not in ("rebased", "stored"):
            raise ValueError("values: 'rebased' | 'stored' | a (T, m) tensor")
        V = rebase(log) if isinstance(values, str) and values == "rebased" else (log.U if isinstance(values, str) else values)
        T = log.T
        t = torch.as_tensor(tau, dtype=torch.float64, device=s.device)
        ar = torch.arange(T, device=s.device)
        return ScopedLog(det_w=log.W, det_s=s, det_tau=t.expand(T).clone() if t.dim() == 0 else t.clone(), det_edit=ar,
                         det_kind=torch.zeros(T, dtype=torch.long, device=s.device), det_key=log.K,
                         values=torch.as_tensor(V).to(s.device, torch.float64), lin_w=log.W, lin_u=log.U,
                         guard_key=log.K, guard_edit=ar.clone(), role=role, fire=fire, response=response, W0=W0,
                         name=name or log.name, meta={"log": log.name, "values": values if isinstance(values, str) else "custom"})

    def replace(self, **kw):
        new = dataclasses.replace(self, **kw)
        if not set(kw) & {"det_w", "det_s", "det_kind", "det_key", "guard_key", "precision"}:
            new._cache = dict(self._cache)          # cached scores of the guard keys do not depend on thresholds / eta / role
        return new

    _TENSORS = ("det_w", "det_s", "det_tau", "det_edit", "det_kind", "det_key", "values", "lin_w", "lin_u", "guard_key", "guard_edit",
                "det_b", "det_u")

    def to(self, device):
        """Copy with every fp64 / index tensor on `device` (W0 stays where the model is)."""
        memo = {}

        def mv(t):
            if t is None:                           # optional fields (ranking baselines / scales)
                return None
            if id(t) not in memo:
                memo[id(t)] = t.to(device)
            return memo[id(t)]                      # tensors shared between fields stay shared
        return self.replace(**{k: mv(getattr(self, k)) for k in self._TENSORS})

    def fold(self):
        """Equivalent log with eta folded into the thresholds (tau <- eta * tau, eta <- 1); fires identically."""
        return self if self.eta == 1.0 else self.replace(det_tau=self.theta(), eta=1.0)

    # ------------------------------------------------------------------ sizes
    @property
    def T(self):
        return self.values.shape[0]

    @property
    def D(self):
        return self.det_w.shape[0]

    @property
    def d(self):
        return self.det_w.shape[1]

    @property
    def device(self):
        return self.det_w.device

    # ------------------------------------------------------------------ the gate (the only place decisions are made)
    def theta(self):
        """Effective thresholds eta * tau, (D,) fp64 (+inf everywhere if eta = +inf)."""
        if self.eta == INF:
            return torch.full_like(self.det_tau, INF)
        return self.eta * self.det_tau

    def scores(self, keys):
        """Normalised coordinates r (n, D) fp64 of keys (n, d) (any float dtype) on every detector.
        The keys of ONE query are evaluated as one batch (in chunks of CHUNK rows), here and in the hook: floating-
        point matrix products depend on the batch shape in the last bits, so thresholds derived from the scores of
        a query reproduce exactly only if the query is passed the same way (see repair.py)."""
        k = keys.reshape(-1, self.d)
        return torch.cat([self._scores(k[a:a + CHUNK]) for a in range(0, max(k.shape[0], 1), CHUNK)])

    def _scores(self, k):
        kd = None
        if self.precision != "fp64" and k.dtype != torch.float64:
            dt = torch.float32 if self.precision == "fp32" else k.dtype
            w = self._cache.get(("w", dt))
            if w is None:
                w = self._cache.setdefault(("w", dt), self.det_w.to(dt))
            c = (k.to(dt) @ w.T).double()
        else:
            kd = k.double()
            c = kd @ self.det_w.T
        r = c / self.det_s
        cos, ball = self.det_kind == COS, self.det_kind == BALL
        if bool(cos.any()) or bool(ball.any()):
            kd = k.double() if kd is None else kd
            if bool(cos.any()):
                r[:, cos] = r[:, cos] / kd.norm(dim=1, keepdim=True).clamp_min(1e-300)
            if bool(ball.any()):
                r[:, ball] = -torch.cdist(kd, self.det_key[ball], compute_mode="donot_use_mm_for_euclid_dist")
        return r

    def rank_key(self, r):
        """(n, D) fp64 quantity that orders the candidates of fire 'top1': r itself, or the margin (r - det_b) / det_u if
        the log carries ranking baselines / scales (two-block calibration: the per-edit conformal score, i.e. the excess
        over the calibrated level up to the constant eta).  r: scores (n, D)."""
        if self.det_b is None and self.det_u is None:
            return r
        q = r if self.det_b is None else r - self.det_b[None, :]
        return q if self.det_u is None else q / self.det_u[None, :]

    def key_scores(self, keys):
        """(n, D) fp64: every key scored as its OWN one-key query (the way the hook scores the single eligible
        position of a role-'last' sequence).  Used for guard keys, so that whether a guard fires never depends on
        how many other keys are evaluated with it."""
        k = keys.reshape(-1, self.d)
        if k.shape[0] == 0:
            return torch.zeros(0, self.D, dtype=torch.float64, device=self.device)
        return torch.cat([self._scores(k[i:i + 1]) for i in range(k.shape[0])])

    def guard_scores(self):
        """(G, D) fp64 scores of the guard keys, each as its own one-key query (cached; independent of the thresholds)."""
        if "guard_r" not in self._cache:
            self._cache["guard_r"] = self.key_scores(self.guard_key)
        return self._cache["guard_r"]

    def query_max(self, key_sets):
        """(N, D) fp64: largest coordinate of every detector over the eligible keys of each query
        (key_sets: list of (n_q, d) tensors, one per query, or one (n, d) tensor = ONE query)."""
        qs = [key_sets] if torch.is_tensor(key_sets) else list(key_sets)
        return torch.stack([self.scores(q.to(self.device)).max(0).values for q in qs]) if qs else torch.zeros(0, self.D, dtype=torch.float64, device=self.device)

    def fires_on(self, keys):
        """(n, D) bool: detector j is a candidate on key n (keys are taken as eligible)."""
        return self.scores(keys) >= self.theta()[None, :]

    def edit_fires(self, keys):
        """(n, T) bool: edit e is a candidate on key n (through any of its detectors)."""
        return self._edit_any(self.fires_on(keys))

    def score(self, keys):
        """Conformal score of each key: max_j r_j / tau_j (eta is NOT applied); needs every finite tau > 0.
        Detectors with tau = +inf can never fire and do not count (-inf if there is no other detector)."""
        fin = torch.isfinite(self.det_tau)
        if bool((self.det_tau[fin] <= 0).any()) or bool((self.det_tau == -INF).any()):
            raise ValueError("score needs thresholds tau > 0 (calibrate first)")
        ratio = self.scores(keys) / self.det_tau[None, :]
        return torch.where(fin[None, :], ratio, torch.full_like(ratio, -INF)).max(1).values

    def guard_ok(self):
        """(G,) bool: edit guard_edit[g] is a candidate on guard_key[g]."""
        if self.guard_key.shape[0] == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)
        ef = self._edit_any(self.guard_scores() >= self.theta()[None, :])
        return ef[torch.arange(ef.shape[0], device=ef.device), self.guard_edit]

    def _edit_any(self, cand):
        if self.D == self.T:
            return cand
        out = torch.zeros(cand.shape[0], self.T, dtype=torch.float64, device=cand.device)
        return out.index_add_(1, self.det_edit, cand.double()) > 0

    def _edit_max(self, cand, r):
        """(n, T) largest r among the candidate detectors of every edit (-inf where the edit is no candidate)."""
        rc = torch.where(cand, r, torch.full_like(r, -INF))
        if self.D == self.T:
            return rc
        out = torch.full((cand.shape[0], self.T), -INF, dtype=torch.float64, device=cand.device)
        return out.scatter_reduce(1, self.det_edit[None, :].expand_as(rc), rc, reduce="amax", include_self=True)

    def eligible(self, B, Tn, lp=None, pos0=0):
        """Flat indices (row * Tn + position) of the eligible positions of a call; None = every position."""
        if self.role == "all":
            return None
        if lp is None:
            raise ValueError(f"role {self.role!r} needs the last prompt index of every row (Hook.begin(lp))")
        lp = torch.as_tensor(lp, device=self.device).reshape(-1)
        if lp.numel() != B:                             # a shorter vector would silently make only the first rows eligible
            raise ValueError(f"need one last prompt index per row: got {lp.numel()} for a batch of {B}")
        ar = torch.arange(Tn, device=self.device) + pos0
        mask = (ar[None, :] == lp[:, None]) if self.role == "last" else (ar[None, :] >= lp[:, None])
        return mask.reshape(-1).nonzero().flatten()

    def eligible_keys(self, x, lp=None):
        """Keys at the eligible positions of ONE captured sequence x (Tn, d)."""
        idx = self.eligible(1, x.shape[-2], None if lp is None else torch.as_tensor(lp).reshape(-1)[:1])
        x2 = x.reshape(-1, x.shape[-1])
        return x2 if idx is None else x2[idx.to(x2.device)]

    def firing(self, x, lp=None, pos0=0, state=None):
        """Decision for one call of the module: x (B, Tn, d) or (Tn, d) module input.  None if nothing fires."""
        B, Tn = (x.shape[0], x.shape[1]) if x.dim() == 3 else (1, x.shape[0])
        idx = self.eligible(B, Tn, lp, pos0)
        if idx is not None and idx.numel() == 0:
            return None
        x2 = x.reshape(-1, x.shape[-1])
        r = self.scores(x2 if idx is None else x2[idx])
        cand = r >= self.theta()[None, :]
        sel = cand.any(1).nonzero().flatten()
        if sel.numel() == 0:
            return None
        cand, r = cand[sel], r[sel]
        flat = sel if idx is None else idx[sel]
        top = torch.where(cand, self.rank_key(r), torch.full_like(r, -INF)).argmax(1)
        return Firing(idx=flat, row=flat // Tn, pos=flat % Tn + pos0, cand=cand, r=r, top=top,
                      edit=self.det_edit[top] if self.fire == "top1" else torch.full_like(top, -1))

    def respond(self, x, out, f: Firing):
        """New module output: `out` with the rows of the firing positions replaced (see the module docstring)."""
        m = out.shape[-1]
        if self.fire == "top1":
            rel = torch.zeros(f.idx.numel(), self.T, dtype=torch.bool, device=out.device)
            rel[torch.arange(f.idx.numel(), device=out.device), f.edit] = True
            rtop = f.r.gather(1, f.top[:, None])                                   # (n, 1) r of the releasing detector
        else:
            rel = self._edit_any(f.cand)
        if self.response == "lin":
            c = x.reshape(-1, x.shape[-1])[f.idx].double() @ self.lin_w.T          # coordinates on the log's read-outs
            delta = torch.where(rel, c, torch.zeros_like(c)) @ self.lin_u
        elif self.fire == "top1":
            delta = self.values[f.edit]
            if self.response == "gain":
                delta = rtop.clamp(max=1.0) * delta
        else:
            coef = rel.double() if self.response == "repl" else torch.where(rel, self._edit_max(f.cand, f.r).clamp(max=1.0),
                                                                             torch.zeros(1, dtype=torch.float64, device=out.device))
            delta = coef @ self.values
        new = out.reshape(-1, m).clone()
        new[f.idx] = (new[f.idx].double() + delta).to(out.dtype)                   # ONE cast to the module dtype
        return new.view(out.shape)

    def apply(self, x, out, lp=None, pos0=0):
        """out' for module input x and module output out = F.linear(x, W_0); returns `out` itself if nothing fires."""
        f = self.firing(x, lp, pos0)
        return out if f is None else self.respond(x, out, f)


# ----------------------------------------------------------------------------- the forward hook
def _keep(f: Firing, keep: torch.Tensor):
    """The firing restricted to the positions keep (n,) bool; None if nothing is left."""
    if bool(keep.all()):
        return f
    if not bool(keep.any()):
        return None
    sub = lambda t: None if t is None else t[keep]                                           # noqa: E731
    return Firing(idx=f.idx[keep], row=f.row[keep], pos=f.pos[keep], edit=f.edit[keep], cand=sub(f.cand), r=sub(f.r), top=sub(f.top), aux=f.aux)


class Hook:
    """Context manager that installs a gate on the edited linear module: THE code path of every gate
    (scoped logs and the baseline gates of baselines.py; a gate is any object with
    firing(x, lp, pos0, state) -> Firing | None and respond(x, out, firing) -> tensor).

        with Hook(module, gate) as hk:
            hk.begin(lp)                 # lp (B,) last prompt index of every row of the next forward(s)
            logits = model(...)
            flags = hk.summary()

    The module weight is set to gate.W0 for the duration (and restored), so the base term is the pre-edit
    module output.  A gate without W0 is refused unless set_weight=False (the caller then guarantees that the
    module already holds W_0; with a W0 the hook verifies it).  begin(lp, gen=True) keeps the position
    bookkeeping through a decode (B = 1): a forward
    whose input is longer than lp is a full-sequence forward (position 0 first); a shorter one is a KV-cache
    step that continues after the positions seen so far.  The module input must be (B, Tn, d) or (Tn, d); lp
    must hold one index per row.  begin(lp, seq_len=...) gives the length of every row of a padded batch:
    PAD positions (>= seq_len) then never fire, whatever the role.
    record=False skips the bookkeeping behind summary() (deployment cost; `last` is still set)."""

    def __init__(self, module, gate, set_weight=True, record=True):
        self.module, self.gate, self.set_weight, self.record = module, gate, set_weight, record
        self._saved = None
        self.begin(None)

    def __enter__(self):
        W0 = getattr(self.gate, "W0", None)
        if W0 is None and self.set_weight:              # would silently run the gate on whatever weight the module holds (W_E)
            raise RuntimeError("the gate has no W0: pass W0= when building it (or set_weight=False if the module already holds W_0)")
        if W0 is not None:
            if self.set_weight:
                self._saved = self.module.weight.data.clone()
                self.module.weight.data.copy_(W0)
            if not torch.equal(self.module.weight.data, W0.to(self.module.weight.device)):
                raise RuntimeError("the module weight is not the gate's pre-edit weight W_0")
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        if self._saved is not None:
            self.module.weight.data.copy_(self._saved)
            self._saved = None
        return False

    def begin(self, lp=None, gen=False, seq_len=None):
        """Start a new prompt (resets the firing record and the per-prompt state of the gate).
        lp: last prompt index of every row; gen: decode bookkeeping (B = 1); seq_len: length of every row of a
        padded batch (PAD positions never fire; not with gen=True, where the sequence grows)."""
        self.lp = None if lp is None else torch.as_tensor(lp).reshape(-1)
        if gen and self.lp is not None and int(self.lp.min()) < 1:
            raise ValueError("decode bookkeeping needs a prompt of at least two tokens (last prompt index >= 1)")
        if gen and seq_len is not None:
            raise ValueError("seq_len masks the PAD positions of a padded batch: not available with gen=True")
        self.seq_len = None if seq_len is None else torch.as_tensor(seq_len).reshape(-1)
        self.gen, self.seen, self.state = gen, 0, {}
        self.calls = self.fire_calls = self.calls_last = self.fire_calls_last = 0
        self.last = None
        self._pos = set()                               # (row, absolute position) that fired in some call
        self._cand_any = self._cand_last = None         # (T,) bool: candidate edits anywhere / at the last prompt position
        self._top_last = -1
        self._n_det = 0

    def _hook(self, mod, args, out):
        x = args[0]
        B = x.shape[0] if x.dim() == 3 else 1
        for nm, v in (("last prompt index", self.lp), ("sequence length", self.seq_len)):
            if v is not None and v.numel() != B:        # any gate: never broadcast / truncate silently
                raise ValueError(f"need one {nm} per row: got {v.numel()} for a batch of {B}")
        pos0 = 0
        if self.gen and self.lp is not None:
            if x.shape[-2] > int(self.lp.max()):
                self.seen = x.shape[-2]
            else:
                pos0 = self.seen
                self.seen += x.shape[-2]
        self.calls += 1
        if self.lp is not None and bool(((self.lp >= pos0) & (self.lp < pos0 + x.shape[-2])).any()):
            self.calls_last += 1                        # this call contains a last prompt position
        f = self.gate.firing(x, self.lp, pos0, self.state)
        if f is not None and self.seq_len is not None:
            f = _keep(f, f.pos < self.seq_len.to(f.pos.device)[f.row])       # drop the PAD positions of a padded batch
        self.last = f
        if f is None:
            return out                                  # the module's own output object, untouched
        self.fire_calls += 1
        if self.record:
            self._record(f)
        return self.gate.respond(x, out, f)

    def _record(self, f):
        rows, pos = f.row.tolist(), f.pos.tolist()
        self._pos.update(zip(rows, pos))
        at = None
        if self.lp is not None:
            at = f.pos == self.lp.to(f.pos.device)[f.row]
            self.fire_calls_last += int(bool(at.any()))
            if bool(at.any()) and self._top_last < 0 and f.top is not None:
                self._top_last = int(self.gate.det_edit[f.top[at][0]])       # edit of the first-ranked candidate (largest r / margin)
        if f.cand is not None:
            ce = self.gate._edit_any(f.cand)
            self._n_det = max(self._n_det, int(f.cand.sum(1).max()))
            self._cand_any = ce.any(0) if self._cand_any is None else (self._cand_any | ce.any(0))
            if at is not None and bool(at.any()):
                cl = ce[at].any(0)
                self._cand_last = cl if self._cand_last is None else (self._cand_last | cl)

    def summary(self, own=None):
        """Firing flags of the prompt since begin() (JSON-serialisable).  own: an edit index -> own_* flags."""
        if not self.record:
            raise RuntimeError("this hook does not record (record=False)")
        lp = None if self.lp is None else self.lp.tolist()
        last = [(r, p) for (r, p) in self._pos if lp is not None and p == lp[r]]
        s = {"fired": bool(self._pos), "n_fire_pos": len(self._pos), "n_calls": self.calls, "n_fire_calls": self.fire_calls,
             "n_calls_last": self.calls_last, "n_fire_calls_last": self.fire_calls_last,   # a decode re-evaluates the last prompt
             "fired_last": bool(last),                                                      # position in every no-cache forward
             "fired_before": any(lp is not None and p < lp[r] for (r, p) in self._pos),
             "fired_later": any(lp is not None and p > lp[r] for (r, p) in self._pos),
             "top_last": self._top_last,                 # edit of the first-ranked candidate at the last prompt position (-1: none / dense gate)
             "n_cand_edits": None if self._cand_any is None else int(self._cand_any.sum()),
             "max_cand_detectors": self._n_det}
        if own is not None:                             # per-edit flags: None for gates that do not resolve edits (dense gates)
            res = hasattr(self.gate, "det_edit")
            s["own_fired"] = bool(self._cand_any is not None and self._cand_any[own]) if res else None
            s["own_fired_last"] = bool(self._cand_last is not None and self._cand_last[own]) if res else None
            s["own_released_last"] = (bool(self._top_last == own) if self.gate.fire == "top1" else s["own_fired_last"]) if res else None
        return s
