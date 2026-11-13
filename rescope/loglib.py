"""The edit log of ONE edited linear module, as a sequential (batched) AlphaEdit run leaves it.

    W_E = W_0 + sum_i outer(U_i, W_i) + E

  U_i (m)  stored residual of edit i at solve time, U_i = z*_i - h_i(W_before its batch)
           (h = block output at the last prompt token; the module has no bias, h = mid + W k)
  W_i (d)  read-out vector of edit i (row i of the solver's factor: batch delta = sum_i outer(U_i, W_i))
  k_i (d)  stored solver key (module input at the last prompt token of the edit's own sequence)
  batch_i  index of the batch that wrote edit i (every edit of a batch sees the pre-batch weight)
  E        dense round-off of the low-precision in-place writes; NOT part of this object.

Coordinates are SIGNED: c_i(k) = <W_i, k>, s_i = c_i(k_i).

REBASED VALUE (derivation: [redacted: describes a related submission under double-blind review]).
Targets z*_i are computed once on the clean model.  Batch b captures, in one forward at the current
weight W_{b-1}, key k_i and block output h_i = mid_i + W_{b-1} k_i of each of its edits, stores
U_i = z*_i - h_i, solves once, writes once.  Hence the residual relative to the PRE-EDIT weight is

    V_i := z*_i - mid_i - W_0 k_i = U_i + (W_{b-1} - W_0) k_i
         = U_i + sum_{j : batch(j) < batch(i)} U_j <W_j, k_i>   (+ E_{<b} k_i, dropped: "factor form")

Batch-mates do not enter each other's residual.  `rebase` is the factor form (log quantities only);
`rebase_exact` adds the round-off term from the stored applied batch deltas.

Everything here is CPU-friendly fp64 torch; no model, no harness.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

TESTBED = f"{_RP.ROOT}/02_research/experiments/06_testbed_v3"


@dataclass
class EditLog:
    U: torch.Tensor                 # (T, m) fp64 stored residuals
    W: torch.Tensor                 # (T, d) fp64 read-out vectors
    K: torch.Tensor                 # (T, d) fp64 stored solver keys
    batch: torch.Tensor             # (T,) long, batch index of every edit
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.U, self.W, self.K = self.U.double(), self.W.double(), self.K.double()
        self.batch = torch.as_tensor(self.batch, dtype=torch.long)
        T = self.U.shape[0]
        if not (self.W.shape[0] == self.K.shape[0] == self.batch.shape[0] == T and self.W.shape == self.K.shape):
            raise ValueError("inconsistent log shapes")

    @property
    def T(self):
        return self.U.shape[0]

    @property
    def m(self):
        return self.U.shape[1]

    @property
    def d(self):
        return self.W.shape[1]

    def to(self, device):
        return EditLog(self.U.to(device), self.W.to(device), self.K.to(device), self.batch.to(device), self.name, dict(self.meta))

    def coords(self, keys):
        """c[n, i] = <W_i, key_n> (fp64).  keys (n, d) or (d,)."""
        k = torch.as_tensor(keys).to(self.W.device, torch.float64)
        return k.reshape(-1, self.d) @ self.W.T

    def own(self):
        """s_i = <W_i, k_i>, the coordinate of every edit's own stored solver key."""
        return (self.W * self.K).sum(1)

    def crosstalk(self):
        """R[j, i] = c_i(k_j) / s_i: normalised coordinate of solver key j on edit i (diagonal = 1)."""
        return (self.K @ self.W.T) / self.own()[None, :]

    def delta(self):
        """sum_i outer(U_i, W_i), (m, d) fp64: the log's linear term WITHOUT the write round-off E."""
        return self.U.T @ self.W


# ----------------------------------------------------------------------------- rebased values
def rebase(log: EditLog) -> torch.Tensor:
    """Factor form: V_i = U_i + sum_{j: batch(j) < batch(i)} U_j <W_j, k_i>.  (T, m) fp64."""
    M = log.K @ log.W.T                                              # M[i, j] = c_j(k_i)
    earlier = log.batch[None, :] < log.batch[:, None]                # [i, j]: j written in an earlier batch than i
    return log.U + torch.where(earlier, M, torch.zeros_like(M)) @ log.U


def rebase_exact(log: EditLog, batch_deltas) -> torch.Tensor:
    """Exact form: V_i = U_i + (sum of the applied deltas of the earlier batches) k_i.
    batch_deltas: iterable of (m, d) tensors in batch order (e.g. the testbed's batch_deltas/*.pt)."""
    S = torch.zeros(log.m, log.d, dtype=torch.float64, device=log.U.device)
    V = log.U.clone()
    for b, A in enumerate(batch_deltas):
        idx = (log.batch == b).nonzero().flatten()
        V[idx] = log.U[idx] + log.K[idx] @ S.T
        S += torch.as_tensor(A).to(S.device, torch.float64)
    return V


def rebase_error(log: EditLog, target: torch.Tensor, V: torch.Tensor | None = None) -> dict:
    """Relative error per edit, ||X_i - target_i|| / ||target_i||, of the rebased value and of the stored
    residual against a reference pre-edit residual `target` (T, m) = z*_i - h_i(W_0)."""
    tgt = torch.as_tensor(target).to(log.U.device, torch.float64)
    V = rebase(log) if V is None else V

    def rel(X):
        return ((X - tgt).norm(dim=1) / tgt.norm(dim=1)).cpu().numpy()

    def q(v):
        return {"max": float(v.max()), "median": float(np.median(v)), "mean": float(v.mean()), "argmax": int(v.argmax())}
    ev, eu = rel(V), rel(log.U)
    first = (log.batch == int(log.batch.min())).cpu().numpy()
    return {"rebased": q(ev), "not_rebased": q(eu), "n": int(log.T),
            "not_rebased_later_batches": q(eu[~first]) if (~first).any() else None,
            "rebased_per_edit": ev.tolist(), "not_rebased_per_edit": eu.tolist()}


# ----------------------------------------------------------------------------- loaders
def from_testbed(name_or_path: str) -> EditLog:
    """Log of a 06_testbed_v3 directory (edit_factors.pt + metrics.json); CPU, no model."""
    p = name_or_path if os.path.isdir(name_or_path) else os.path.join(TESTBED, name_or_path)
    fac = torch.load(os.path.join(p, "edit_factors.pt"), map_location="cpu")
    with open(os.path.join(p, "metrics.json")) as f:
        met = json.load(f)
    T = fac["U"].shape[0]
    batch = torch.full((T,), -1, dtype=torch.long)
    for b in met["per_batch"]:
        batch[torch.tensor(b["edit_ids"])] = int(b["batch"])
    if int(batch.min()) < 0:
        raise ValueError("an edit has no batch")
    return EditLog(fac["U"], fac["W"], fac["K1_solver"], batch, name=os.path.basename(p.rstrip("/")),
                   meta={"path": p, "uids": list(fac["uids"]), "l2": met["config"].get("l2"), "dtype": met.get("dtype"),
                         "layer": met["config"].get("layer"), "backbone": met["config"].get("backbone")})


def batch_deltas(name_or_path: str):
    """Generator over the stored exact applied batch deltas of a testbed log (for `rebase_exact`)."""
    p = name_or_path if os.path.isdir(name_or_path) else os.path.join(TESTBED, name_or_path)
    n = len([f for f in os.listdir(os.path.join(p, "batch_deltas")) if f.startswith("batch_")])
    for b in range(n):
        yield torch.load(os.path.join(p, "batch_deltas", f"batch_{b:03d}.pt"), map_location="cpu")


def from_harness(H) -> EditLog:
    """Log of a loaded harness handle (07_repair_v3/harness.load): same tensors, on the CPU."""
    batch = torch.tensor([H.batch_of[i] for i in range(H.T)])
    return EditLog(H.U.cpu(), H.Wfac.cpu(), H.Kp_log.T.contiguous().cpu(), batch, name=H.name,
                   meta={"uids": [d["uid"] for d in H.items], "l2": H.cfg.get("l2"), "dtype": str(H.dtype), "layer": H.L,
                         "backbone": H.backbone})


# ----------------------------------------------------------------------------- synthetic logs (unit tests)
def synthetic_log(T=24, d=96, m=20, batch=4, seed=0, l2=0.5, n_protect=16, common=1.0, write_dtype=None):
    """A small log written by a sequential residual-form solver with batches (fp64).

    W_0 random; keys = common direction + noise (correlated, like real keys); targets z*_i random.
    Batch b: U_i = z*_i - W_{b-1} k_i for its edits (mid = 0), AlphaEdit closed form
        D_b = R K1^T P (Kp Kp^T P + K1 K1^T P + l2 I)^-1 = sum_i outer(U_i, W_i),
    P = projector onto the complement of `n_protect` random directions, Kp = keys of the earlier batches;
    write W_b = W_{b-1} + D_b (exactly; or rounded to `write_dtype` twice like the real editor, which
    creates a round-off term E).  Returns (log, truth) with truth = dict(W0, WE, Z, P)."""
    g = torch.Generator().manual_seed(seed)
    W0 = torch.randn(m, d, generator=g, dtype=torch.float64) / d ** 0.5
    base = torch.randn(d, generator=g, dtype=torch.float64)
    K = common * base[None, :] + torch.randn(T, d, generator=g, dtype=torch.float64)
    Z = torch.randn(T, m, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(torch.randn(d, n_protect, generator=g, dtype=torch.float64))
    P = torch.eye(d, dtype=torch.float64) - Q @ Q.T
    W = W0.clone() if write_dtype is None else W0.to(write_dtype).double()
    W0 = W.clone()
    U = torch.zeros(T, m, dtype=torch.float64)
    Wf = torch.zeros(T, d, dtype=torch.float64)
    bidx = torch.arange(T) // batch
    Kp = torch.zeros(d, 0, dtype=torch.float64)
    for b in range(int(bidx.max()) + 1):
        ids = (bidx == b).nonzero().flatten()
        K1 = K[ids].T                                                # (d, B)
        R = (Z[ids] - K[ids] @ W.T).T                                # (m, B) residuals at the pre-batch weight
        A = Kp @ Kp.T @ P + K1 @ K1.T @ P + l2 * torch.eye(d, dtype=torch.float64)
        F = torch.linalg.solve(A.T, (K1.T @ P).T).T                  # (B, d) = K1^T P A^-1: rows are the read-outs
        U[ids], Wf[ids] = R.T, F
        D = R @ F
        if write_dtype is None:
            W = W + D
        else:
            W = (W.to(write_dtype) + D.to(write_dtype)).double()     # the editor's in-place low-precision write
        Kp = torch.cat([Kp, K1], 1)
    log = EditLog(U, Wf, K, bidx, name=f"synthetic-T{T}-s{seed}", meta={"l2": l2})
    return log, {"W0": W0, "WE": W, "Z": Z, "P": P}
