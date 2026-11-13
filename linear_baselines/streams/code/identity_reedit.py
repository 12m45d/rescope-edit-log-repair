"""P12 -- REEDIT-S against REEDIT-B on the saved weights (CPU only).

Sequential AlphaEdit with accumulating Kp and residuals taken at the current weights, with the targets z* fixed
before the stream (the testbed build's convention), is recursive least squares: with
  A_b = P (Kp Kp^T + sum_{j<=b} K_j K_j^T) + l2 I   and   D_tot,b A_b = D_tot,b-1 A_b-1 + R_b^0 K_b^T
the batch-b update D_b = (R_b^0 - D_tot,b-1 K_b) K_b^T A_b^-1 is exactly the editor's solve on the advanced log, so
the stream ends at the joint solve of all batches (exact arithmetic; the half-precision write after every batch
and the half-precision residuals are the only differences).  This script measures the difference on the
corrections saved by run_streams.py (n in {10, 100, ALL}, seed 0):
    rel = ||W_S - W_B||_F / ||W_B - W_E||_F          (||W_B - W_E||_F = the record's eval.fro)
-> out/<log>/_reedit_identity.json (read by aggregate.py).

usage: CUDA_VISIBLE_DEVICES="" python identity_reedit.py
"""
from __future__ import annotations

import json
import os

import torch

P12 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(P12, "out")
for log in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"):
    od = os.path.join(OUT, log)
    res = {}
    for n in ("10", "100", "ALL"):
        nb, ns = f"REEDIT_B__n{n}__s0", f"REEDIT_S__n{n}__s0"
        pb, ps, pa = (os.path.join(od, "corr", nb + ".pt"), os.path.join(od, "corr", ns + ".pt"),
                      os.path.join(od, "corr", ns + ".alias"))
        rb = os.path.join(od, nb + ".json")
        if os.path.exists(pa):
            res[n] = {"alias": True, "note": "one batch: the same computation"}
            continue
        if not (os.path.exists(pb) and os.path.exists(ps) and os.path.exists(rb)):
            continue
        b, s = torch.load(pb, map_location="cpu"), torch.load(ps, map_location="cpu")
        if b["kind"] != "weight" or s["kind"] != "weight":
            continue
        fro_b = json.load(open(rb))["eval"]["fro"]
        d = (s["W"].double() - b["W"].double())
        # size of the entry-wise differences in steps of the weight format (ULP at the larger of the two entries;
        # explicit mantissa bits: bfloat16 7, float16 10).  Rounding of the per-batch writes alone moves an entry by
        # about one step per batch; much larger multiples on small entries come from the updates themselves
        # (residuals taken from half-precision forwards at the rounded weights).
        mb = 7 if b["W"].dtype == torch.bfloat16 else 10
        wmax = torch.maximum(b["W"].double().abs(), s["W"].double().abs()).clamp_min(1e-300)
        r = d.abs() / torch.pow(2.0, torch.floor(torch.log2(wmax)) - mb)
        nz = r[r > 0]
        ulp = None if nz.numel() == 0 else {"median": float(nz.median()), "share_within_2": float((nz <= 2.0 + 1e-9).double().mean()),
                                            "share_above_16": float((nz > 16.0).double().mean()), "max": float(nz.max()),
                                            "mantissa_bits": mb}
        del wmax, r, nz
        res[n] = {"alias": False, "rel_fro_diff": float(d.norm()) / max(fro_b, 1e-30), "fro_B": fro_b, "dtype": str(b["W"].dtype), "ulp": ulp,
                  "entries_differing": int((s["W"] != b["W"]).sum()), "entries": int(b["W"].numel()),
                  "max_abs_diff": float(d.abs().max())}
    if res:
        with open(os.path.join(od, "_reedit_identity.json"), "w") as f:
            json.dump(res, f, indent=1)
    print(log, res)
