"""Loader for the 06_testbed_v3 edit logs (CPU only; no model needed).

Files in each log directory:
  edit_batch.jsonl          edit stream (edit_id, uid, subject, alt, image, ...) in edit order
  batch_deltas/batch_XXX.pt EXACT applied update of batch XXX: W_after.float() - W_before.float()
                            (fp32, (d_out, d_in)); B=1 logs have one file per edit
  composite_delta.pt        W_E.float() - W0.float() (fp32), W_E = final weights after all edits
  edit_factors.pt           per edit j: U[j] (d_out, fp64) and W[j] (d_in, fp64) with
                            batch solver delta (fp64) = sum_{j in batch} outer(U[j], W[j]);
                            K1_solver[j] (key the solver used), k_t_single[j] (single-item key)
  editlog.pkl               editlog.EditLog; B=1: entry.delta_W = exact applied delta;
                            B>1: entry.delta_W = None, factors + batch file in entry.meta
  metrics.json              build + evaluation summary (per_batch has bf16_write_rel_dev etc.)

Exactness:
  * W_E == (W0.float() + sum_b batch_delta_b).to(bf16/fp16)  bit-for-bit (checked at build time,
    metrics.json build.recon_bitwise_W_E_from_batch_deltas).
  * the solver's per-edit rank-1 terms sum to each batch's fp64 solver delta to ~1e-12; the
    applied (bf16/fp16-rounded) batch delta differs from the solver delta by the in-place
    low-precision write (per_batch[].bf16_write_rel_dev, ~0.5-2%).  Use batch deltas to rebuild
    W_E exactly; use the factors for per-edit attribution inside a batch.
"""
from __future__ import annotations

import json
import os

import torch


class TestbedLog:
    def __init__(self, path: str):
        self.path = path
        self.edits = [json.loads(l) for l in open(os.path.join(path, "edit_batch.jsonl"))]
        self.metrics = json.load(open(os.path.join(path, "metrics.json")))
        self.cfg = self.metrics["config"]
        self.per_batch = self.metrics["per_batch"]
        self._fac = None

    @property
    def factors(self):
        if self._fac is None:
            self._fac = torch.load(os.path.join(self.path, "edit_factors.pt"))
        return self._fac

    def batch_of(self, edit_id: int) -> int:
        for b in self.per_batch:
            if edit_id in b["edit_ids"]:
                return b["batch"]
        raise KeyError(edit_id)

    def batch_delta(self, b: int) -> torch.Tensor:
        return torch.load(os.path.join(self.path, "batch_deltas", f"batch_{b:03d}.pt"))

    def composite_delta(self) -> torch.Tensor:
        return torch.load(os.path.join(self.path, "composite_delta.pt"))

    def edit_delta_solver(self, j: int, dtype=torch.float32) -> torch.Tensor:
        """Exact per-edit term of the batch solve (fp64 outer product, cast to dtype)."""
        f = self.factors
        return torch.outer(f["U"][j], f["W"][j]).to(dtype)

    def edit_delta_applied(self, j: int) -> torch.Tensor:
        """Exact applied delta; only defined per edit when the batch size is 1."""
        if self.cfg["batch"] != 1:
            raise ValueError("B>1: applied deltas exist per batch only; use batch_delta / edit_delta_solver")
        return self.batch_delta(j)

    def rebuild_WE(self, W0: torch.Tensor) -> torch.Tensor:
        S = torch.zeros(W0.shape, dtype=torch.float32, device=W0.device)
        for b in range(len(self.per_batch)):
            S += self.batch_delta(b).to(W0.device)
        return (W0.float() + S).to(W0.dtype)


if __name__ == "__main__":
    import sys
    lg = TestbedLog(sys.argv[1])
    out = []
    for b in lg.per_batch[:3]:
        A = lg.batch_delta(b["batch"])
        D = sum(lg.edit_delta_solver(j, torch.float64) for j in b["edit_ids"]).float()
        out.append({"batch": b["batch"], "rel_dev_applied_vs_sum_factors": float((A - D).norm() / D.norm()),
                    "recorded_bf16_write_rel_dev": b["bf16_write_rel_dev"]})
    print(json.dumps(out, indent=1))
