"""E10: the repairs of amendments R14 / R21 (d) / R22 (e) on a text log, as functions of the log and of the complaints' keys.

  replay(L, keep)                weight after replaying the logged float32 writes of the kept updates, in order (rollbacks)
  rb_suffix(L, B, resolved)      point-in-time recovery: the LONGEST prefix of the log under which every reported complaint is
                                 resolved; every prefix 1..T is replayed and tested (the complaint test stops at the first
                                 unresolved batch); forward passes are counted
  lineage(S, key_sets)           edits that the thresholds of S (ReScope-W, log-only) release somewhere on the complaints
  comp_min(L, KQ)                minimum-norm correction C of the module: k (W_E + C) = k W_0 at every complaint key, k_i C = 0 at
                                 every solver key; infeasible if the constraints exceed the input dimension
  least_sacrificing / least_protecting      the library's least_repair of ReScope-W thresholds in the two forms of R21 (d)
Orientation: the edited module is a Conv1D, y = x @ W + b, W (6400, 1600); keys are rows.
"""
from __future__ import annotations

import time

import numpy as np
import torch

import e10t
from e10t import data as D

D_IN = 6400


def factors(L):
    f = D._tx().factors(L.base)
    return f["left"][:L.T], f["right"][:L.T]


def replay(L, keep=None) -> torch.Tensor:
    """W_0 plus the public code's float32 in-place writes of the updates i with keep[i] (default: all), in log order."""
    left, right = factors(L)
    W = L.weights()[0].clone()
    for i in range(L.T):
        if keep is None or bool(keep[i]):
            W += left[i].unsqueeze(1) @ right[i].unsqueeze(0)
    return W


def rb_suffix(L, B, resolved, stop_after_fail: int | None = None) -> dict:
    """resolved() -> (bool, n_sequences_run) is called with the model served at each prefix weight.  -> {t_star, n_forward,
    resolved_prefixes, replay_equals_deployed, seconds}.  Every prefix is tested (the set of resolving prefixes need not be an
    interval); t = 0 (the pre-edit model) resolves every complaint by definition."""
    t0 = time.time()
    left, right = factors(L)
    W = L.weights()[0].clone()
    ok_at, nfw = [0], 0
    for t in range(1, L.T + 1):
        W += left[t - 1].unsqueeze(1) @ right[t - 1].unsqueeze(0)
        with B.served(W):
            ok, n = resolved()
        nfw += n
        if ok:
            ok_at.append(t)
    return {"t_star": int(max(ok_at)), "n_forward": int(nfw), "n_resolving_prefixes": len(ok_at), "resolving_prefixes_tail": ok_at[-10:],
            "resolving_set_is_an_interval": ok_at == list(range(len(ok_at))), "replay_equals_deployed": bool(torch.equal(W, L.weights()[1])),
            "seconds": time.time() - t0}


def lineage(S, key_sets) -> list:
    """Edits that are candidates under S's thresholds at some key of some complaint (key_sets: list of (n_i, d) tensors)."""
    S1 = S.fold()
    if not key_sets:
        return []
    Q = S1.query_max([torch.as_tensor(k) for k in key_sets])
    cand = S1._edit_any(Q >= S1.theta()[None, :]).any(0)
    return [int(i) for i in cand.nonzero().flatten()]


def comp_min(L, KQ, rcond: float = 1e-10) -> dict:
    """KQ: (n, 6400) keys of the reported complaints (all their positions).  -> {feasible, W (float32 module weight) | None, ...}."""
    t0 = time.time()
    W0, WE = L.weights()
    KQ = torch.unique(torch.as_tensor(KQ).float(), dim=0).double()
    KE = L.log.K.double()
    n_q, n_e = int(KQ.shape[0]), int(KE.shape[0])
    rec = {"n_complaint_keys_distinct": n_q, "n_solver_keys": n_e, "n_constraints": n_q + n_e, "d_in": D_IN}
    if n_q + n_e > D_IN:
        rec.update(feasible=False, reason="constraints exceed the input dimension", seconds=time.time() - t0)
        return rec
    Delta = WE.double() - W0.double()
    A = torch.cat([KQ, KE])
    Bm = torch.cat([-(KQ @ Delta), torch.zeros(n_e, Delta.shape[1], dtype=torch.float64)])
    ev, V = torch.linalg.eigh(A @ A.T)
    keep = ev > rcond * ev[-1]
    C = A.T @ ((V[:, keep] / ev[keep]) @ (V[:, keep].T @ Bm))
    R = A @ C - Bm
    ref = (KQ @ Delta).norm(dim=1)
    res_q = (R[:n_q].norm(dim=1) / ref.clamp_min(1e-300)).max() if n_q else torch.tensor(0.0)
    res_e = (R[n_q:].norm(dim=1) / (KE @ Delta).norm(dim=1).clamp_min(1e-300)).max()
    Wn = (WE.double() + C).float()
    rec.update(feasible=True, rank=int(keep.sum()), n_dropped_directions=int((~keep).sum()), correction_fro=float(C.norm()), delta_fro=float(Delta.norm()),
               max_rel_residual_complaint_keys=float(res_q), max_rel_residual_solver_keys=float(res_e),
               consistent=bool(res_q < 1e-6 and res_e < 1e-6), seconds=time.time() - t0)
    rec["W"] = Wn
    return rec


def least_sacrificing(S0, key_sets):
    """The form of the theory: no must-cover pair; every complaint is certified; guards that the raised thresholds silence
    are counted (conflicts).  S0: ReScope-W with the log-only thresholds, guards off."""
    from rescope import least_repair
    t0 = time.time()
    S, rec = least_repair(S0, [torch.as_tensor(k) for k in key_sets], on_conflict="force")
    return S, {"form": "sacrificing", "n_complaints": len(key_sets), "n_refused": 0, "n_thresholds_raised": rec["n_raised"], "resolved": rec["resolved"],
               "n_guard_conflicts": len(rec["conflicts"]), "edits_with_guard_conflict": sorted({c["edit"] for c in rec["conflicts"]})[:64],
               "n_edits_with_guard_conflict": len({c["edit"] for c in rec["conflicts"]}), "seconds": time.time() - t0}


def least_protecting(S0, key_sets):
    """The library's rule: a complaint whose repair would silence a guard (own request) is refused; complaints are handled one
    by one (a coordinatewise maximum: the order plays no role).  S0: ReScope-W with the log-only thresholds, guards on."""
    from rescope import least_repair
    t0 = time.time()
    S, refused, raised = S0, [], set()
    for j, k in enumerate(key_sets):
        S, rec = least_repair(S, torch.as_tensor(k), on_conflict="refuse")
        if rec["refused"]:
            refused.append(j)
        else:
            raised |= {r["detector"] for r in rec["raised"]}
    return S, {"form": "protecting", "n_complaints": len(key_sets), "n_refused": len(refused), "refused": refused[:200], "n_thresholds_raised": len(raised),
               "seconds": time.time() - t0}


def self_margin(L) -> np.ndarray:
    """Self-separation margin of every edit, from the log alone (R21 c): 1 minus the largest normalised coordinate of edit i at
    the solver key of ANOTHER edit, r_i(k_j) = <W_i, k_j> / <W_i, k_i>.  (T,) fp64."""
    log = L.log
    C = (log.K @ log.W.T) / log.own()[None, :]                       # C[j, i] = r_i(k_j)
    C.fill_diagonal_(-float("inf"))
    return (1.0 - C.max(0).values).numpy()
