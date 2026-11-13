"""Post-hoc gate baselines on the same log.  Every function returns an object with the interface of a scoped
log (firing / respond / apply, role, W0), so scoped.Hook and evaluate.Evaluator treat all gates alike.

  cosine_gate   NeuralDB-style (arXiv 2507.18028, Eq. 11-12): j = argmax_i cos(k, k_i) over the stored solver
                keys; if cos >= gamma the module output is W_0 k + v_j.  A ScopedLog with 'cos' detectors.
                Deviations from the paper (forced by "same log", as in P11a): keys are the log's stored solver
                keys (one context) instead of prefix-averaged subject keys; the released value is the log's
                rebased residual v_j (values='stored' gives the literal stored residual U_j); one layer; fp64 side
                path; '>=' instead of the paper's strict '>' (differs only at an exact tie); DEFAULT ROLE 'last'
                (the published gate acts at every token position: role='all').
  exact_gate    cosine_gate with gamma = 0.999 (memorisation of the edit prompts).  The stored key is the
                solver's B = 10 key; the B = 1 key of the same prompt is not bitwise that key (cosine >= 0.9988
                on Qwen, >= 0.9996 on LLaVA: test b10), so 'exact' can miss an edit's own prompt.
  ball_gate     GRACE-style (arXiv 2211.11031, Eq. 1; checked against the paper and the official adaptor code):
                the nearest stored key fires iff the Euclidean distance |k - k_i| < radius (strict, as Eq. 1;
                the official code tests '<='), ONE global radius; releases v_i.  Query = the final prompt token
                (role 'last'), the paper's choice for autoregressive models.  Deviations: GRACE REPLACES the
                layer output by a learned codebook value and adapts one radius per key while editing (eps_init,
                split / expand); here the value is the log's rebased residual added to W_0 k and the radius is
                global.
  router_gate   WISE-style activation router as in P11a: a = |(W_E - W_0) k| at the last prompt position; if
                a >= eps the deployed output W_E k is used at the last prompt position and all later positions,
                otherwise W_0 everywhere.  The decision of the first forward that contains the last prompt
                position is kept for the whole prompt (decode).  eps: a number, or the q-quantile of the edits'
                own activations on the stored solver keys.
  zscore_gate   CoachHooK-style (arXiv 2403.05330, Sec. 3.1.3, Eq. 16-18; official hook: validated_hook in
                CoachHooK_main.py): M_p = |(W_E - W_0) k_p| for every
                token position p of the input, Z = (M - mean) / std over the positions of the same input
                (unbiased std, torch.std, as the official code); the deployed output W_E k_p is used where
                Z_p >= alpha, W_0 k_p otherwise.  Deviations: M is computed in fp64 from the exact weight
                difference instead of the difference of the two low-precision layer outputs; rows of a padded
                batch are not masked (neither does the official hook; use B = 1).  KV-cache decoding steps
                (the module input holds only the new token): kv='stale' (default) scores the new token with
                the mean and std of the last full-sequence forward of the prompt; kv='call' is the literal
                official hook, which standardises over the positions of each call and therefore NEVER swaps
                in a one-token step (the std of one value is NaN).  Full-sequence forwards (teacher forcing,
                the first 10 tokens of decoder v2) are the same under both.  coachhook_alpha implements the
                paper's threshold rule (Eq. 19).

Dense gates (router, zscore) need the deployed weight W_E; cosine / ball gates need only the log.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as Fnn

from .loglib import EditLog, rebase
from .scoped import BALL, COS, INF, LIN, Firing, ScopedLog


def _key_gate(log: EditLog, kind, tau, role, values, W0, name):
    T = log.T
    dev = log.K.device
    if isinstance(values, str) and values not in ("rebased", "stored"):
        raise ValueError("values: 'rebased' | 'stored' | a (T, m) tensor")
    V = rebase(log) if isinstance(values, str) and values == "rebased" else (log.U if isinstance(values, str) else values)
    ar = torch.arange(T, device=dev)
    w = log.K / log.K.norm(dim=1, keepdim=True) if kind == COS else log.K
    return ScopedLog(det_w=w, det_s=torch.ones(T, dtype=torch.float64, device=dev),
                     det_tau=torch.full((T,), float(tau), dtype=torch.float64, device=dev), det_edit=ar,
                     det_kind=torch.full((T,), kind, dtype=torch.long, device=dev), det_key=log.K,
                     values=torch.as_tensor(V).to(dev, torch.float64), lin_w=log.W, lin_u=log.U, guard_key=log.K,
                     guard_edit=ar.clone(), role=role, fire="top1", response="repl", W0=W0, name=name,
                     meta={"log": log.name, "baseline": name, "values": values if isinstance(values, str) else "custom"})


def cosine_gate(log: EditLog, gamma: float, role: str = "last", values="rebased", W0=None) -> ScopedLog:
    return _key_gate(log, COS, gamma, role, values, W0, f"cosine-{gamma:g}")


def exact_gate(log: EditLog, role: str = "last", values="rebased", W0=None) -> ScopedLog:
    return _key_gate(log, COS, 0.999, role, values, W0, "exact")


def ball_gate(log: EditLog, radius: float, role: str = "last", values="rebased", W0=None) -> ScopedLog:
    """Fires iff |k - k_i| < radius: threshold on r = -distance is the next float above -radius."""
    tau = float(np.nextafter(-float(radius), np.inf))
    return _key_gate(log, BALL, tau, role, values, W0, f"ball-{radius:g}")


FEATURES = ("ncoord", "coord", "cos", "negdist")


def feature_gate(log: EditLog, feature: str = "ncoord", role: str = "last", fire: str = "top1", response: str = "repl", values="rebased",
                 W0=None) -> ScopedLog:
    """Un-thresholded scoped log (tau = -inf) whose detectors compute one per-edit score FEATURE of a key k, to be calibrated
    by calibrate.calibrate_two_block / calibrate_shared (the same rule for ReScope and for the post-hoc gates):
      'ncoord'   normalised solver coordinate  <W_i, k> / <W_i, k_i>      (ReScope; = ScopedLog.from_log)
      'coord'    raw solver coordinate         <W_i, k>                   (margin not divided by s_i)
      'cos'      raw cosine to the stored solver key  cos(k, k_i)         (NeuralDB-style gate)
      'negdist'  minus the Euclidean distance to the stored key  -|k - k_i|   (GRACE-style gate)
    Guards are the stored solver keys (add the B = 1 own-prompt keys with repair.add_guards before calibrating)."""
    if feature not in FEATURES:
        raise ValueError(f"feature: one of {FEATURES}")
    if feature != "ncoord" and response == "gain":
        raise ValueError("response 'gain' needs a coordinate normalised to 1 at the edit's key (feature 'ncoord')")
    if feature == "ncoord":
        S = ScopedLog.from_log(log, role=role, fire=fire, response=response, values=values, W0=W0, name=f"{feature}:{log.name}")
    else:
        S = _key_gate(log, {"coord": LIN, "cos": COS, "negdist": BALL}[feature], -INF, role, values, W0, f"{feature}:{log.name}")
        if feature == "coord":                                          # detectors = the read-out vectors, scale 1 (no s_i > 0 needed)
            S = S.replace(det_w=log.W)
        S = S.replace(fire=fire, response=response)
    return S.replace(meta=dict(S.meta, feature=feature))


def key_distances(log: EditLog) -> torch.Tensor:
    """(T, T) Euclidean distances between the stored solver keys (+inf on the diagonal): for choosing a radius."""
    D = torch.cdist(log.K, log.K, compute_mode="donot_use_mm_for_euclid_dist")
    return D + torch.diag(torch.full((log.T,), INF, dtype=torch.float64, device=D.device))


_DIFF = {}          # one cached fp64 weight difference (W_E - W_0), shared by the dense gates of the same weights


def weight_diff(WE, W0):
    """W_E - W_0 in fp64 (exact for half-precision weights), cached for the most recent (W_E, W_0) pair."""
    if _DIFF.get("key") is None or _DIFF["key"][0] is not WE or _DIFF["key"][1] is not W0:
        _DIFF.clear()
        _DIFF.update(key=(WE, W0), C=WE.double() - W0.double())
    return _DIFF["C"]


def release():
    """Drop the cached fp64 weight difference (0.2 - 0.4 GB on the model's device)."""
    _DIFF.clear()


class DenseGate:
    """router / zscore: at the selected positions the module output is the DEPLOYED output F.linear(x, W_E)
    (module dtype, computed on the whole input: the very operation of the W_E forward)."""

    def __init__(self, kind, WE, W0, thr, name, kv="stale"):
        if kind not in ("router", "zscore"):
            raise ValueError(kind)
        if kv not in ("stale", "call"):
            raise ValueError("kv: 'stale' | 'call'")
        self.kind, self.WE, self.W0, self.thr, self.name, self.kv = kind, WE, W0, float(thr), name, kv
        self.role = "last+answer" if kind == "router" else "all"
        self.fire, self.response = "dense", "deployed"

    def to(self, device):
        return self

    def activation(self, keys):
        """|(W_E - W_0) k| per key, fp64 (the weight difference is exact in fp64)."""
        C = weight_diff(self.WE, self.W0)
        k = keys.reshape(-1, keys.shape[-1])
        return torch.cat([(k[a:a + 1024].to(C.device).double() @ C.T).norm(dim=1) for a in range(0, k.shape[0], 1024)])

    def zscores(self, x):
        """(Tn,) z-scores of ONE full sequence x (Tn, d): CoachHooK's Z^l."""
        M = self.activation(x)
        return (M - M.mean()) / M.std()

    def firing(self, x, lp=None, pos0=0, state=None):
        state = {} if state is None else state
        x3 = x if x.dim() == 3 else x[None]
        B, Tn, d = x3.shape
        dev = x3.device
        ar = torch.arange(Tn, device=dev) + pos0
        if self.kind == "router":
            if lp is None:
                raise ValueError("the router needs the last prompt index (Hook.begin(lp))")
            lp = torch.as_tensor(lp, device=dev).reshape(-1)
            if lp.numel() != B:                                      # a single index would be broadcast to every row
                raise ValueError(f"need one last prompt index per row: got {lp.numel()} for a batch of {B}")
            route = state.get("route")
            if route is None:
                rel = lp - pos0
                has = (rel >= 0) & (rel < Tn)
                if not bool(has.any()):
                    return None
                a = self.activation(x3[torch.arange(B, device=dev), rel.clamp(0, Tn - 1)])
                route = has & (a >= self.thr)
                state["route"], state["a"] = route, a
            mask = route[:, None] & (ar[None, :] >= lp[:, None])
        else:
            if self.kv == "call" and Tn < 2:                         # official hook: the std of one position is NaN -> no swap
                return None
            M = self.activation(x3.reshape(-1, d)).view(B, Tn)
            if pos0 == 0 or self.kv == "call":                       # a full-sequence forward (kv 'call': every call): statistics
                state["mu"], state["sd"] = M.mean(1, keepdim=True), M.std(1, keepdim=True)       # of this input
            if "mu" not in state:
                return None
            mask = (M - state["mu"]) / state["sd"] >= self.thr       # NaN (one position) compares False
        idx = mask.reshape(-1).nonzero().flatten()
        if idx.numel() == 0:
            return None
        return Firing(idx=idx, row=idx // Tn, pos=idx % Tn + pos0, edit=torch.full_like(idx, -1))

    def respond(self, x, out, f):
        m = out.shape[-1]
        new = out.reshape(-1, m).clone()
        new[f.idx] = Fnn.linear(x, self.WE).reshape(-1, m)[f.idx]
        return new.view(out.shape)

    def apply(self, x, out, lp=None, pos0=0, state=None):
        f = self.firing(x, lp, pos0, state)
        return out if f is None else self.respond(x, out, f)


def router_gate(log: EditLog, WE, W0, eps: float | None = None, q: float | None = None) -> DenseGate:
    """eps given, or eps = q-quantile (numpy linear) of the edits' own activation |(W_E - W_0) k_i| (stored keys)."""
    if (eps is None) == (q is None):
        raise ValueError("give exactly one of eps and q")
    g = DenseGate("router", WE, W0, 0.0, "router")
    if eps is None:
        a_own = g.activation(log.K.to(WE.device)).cpu().numpy()
        eps = float(np.quantile(a_own, q))
        g.name = f"router-q{q:g}"
    else:
        g.name = f"router-{eps:g}"
    g.thr = float(eps)
    return g


def zscore_gate(WE, W0, alpha: float = 2.2, kv: str = "stale") -> DenseGate:
    """kv: statistics used in KV-cache decoding steps ('stale' = those of the last full-sequence forward; 'call' =
    the literal official hook: per call, so a one-token step never swaps).  See the module docstring."""
    return DenseGate("zscore", WE, W0, alpha, f"zscore-{alpha:g}" + ("" if kv == "stale" else f"-kv_{kv}"), kv=kv)


def coachhook_alpha(max_z, step=None, alpha_z: float = 2.2) -> float:
    """CoachHooK Eq. 19: alpha_1 = alpha_z; alpha_s = min(alpha_c(s), alpha_{s-1}), alpha_c(s) = the smallest, over
    the edit instances of consecutive step s, of the instance's largest z-score (DenseGate.zscores(x).max() of the
    edit prompt).  max_z: per-instance maxima; step: their step (batch) index, first step = min(step) (literal
    reading: the candidates of the first step are not used); step=None: every instance counts."""
    z = np.asarray(max_z, dtype=np.float64)
    if step is not None:
        st = np.asarray(step)
        z = z[st != st.min()]
    return float(min(alpha_z, z.min())) if z.size else float(alpha_z)
