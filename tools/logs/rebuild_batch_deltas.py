#!/usr/bin/env python
"""Rebuild the applied batch updates of the released vision-language logs from the public checkpoint and the logged factors.

    python tools/logs/rebuild_batch_deltas.py [paper names ...]        (default: every log that is unpacked; CPU; needs the checkpoints)
    python tools/logs/rebuild_batch_deltas.py --check-only Q-A100-r3   (no file is written: only the check against the deployed update)

WHY.  Every released log holds, per edit, the write vector U_i, the read-out vector W_i, the solver key and the batch, and the
deployed update composite_delta.pt = float(W_E) - float(W_0).  The working tree also kept the EXACT update of every batch,
A_b = float(W after batch b) - float(W before batch b) (one dense matrix per batch: 18 GB for the longest log); the harness of the
linear repairs (rollbacks) and the loaders of the driver read them from <log>/batch_deltas/batch_XXX.pt.  They are not in the
archives because they are a function of released data: the editor wrote each batch as
        W <- W + cast(sum_{i in batch} outer(U_i, W_i))            (sum in float64, cast to the weight's type, add in that type)
so A_b follows from W_0 (public checkpoint: tools/logs/extract_w0.py) and the factors, bit for bit.  This script replays that rule
and CHECKS the result: for every log, cast(float(W_0) + sum_b A_b) - float(W_0) must equal the released composite_delta.pt in
every entry (the definition of the deployed weight W_E of a log; metrics.json, build.W_E_definition).  A log whose check fails is
reported and its files are not written.

WHERE.  $RESCOPE_LOGS/batch_deltas/<paper name of the longest log of the run>/batch_XXX.pt; logs that are prefixes of one
sequential run share the files (LOG_NAMES.json, field batch_deltas); rescope_paths.py links them into each log's directory.
Disk: 90 MB per batch for Qwen2.5-VL-3B, 180 MB for LLaVA-1.5-7B (Q-M2000: 18 GB; L-M1000: 18 GB; a 100-edit log: 0.9 / 1.8 GB).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import rescope_paths as RP  # noqa: E402
import torch  # noqa: E402
from extract_w0 import extract, tensor_sha256, expected  # noqa: E402

CKPT = {"qwen25vl": "Qwen2.5-VL-3B", "llava": "llava-1.5-7b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--check-only", action="store_true", help="write nothing")
    ap.add_argument("--compare-with", help="a directory <dir>/<working-tree log name>/batch_deltas with existing batch updates: compare every rebuilt batch with it, entry by entry")
    a = ap.parse_args()
    N = RP.log_names()["vl"]
    want = a.names or [p for p in N if os.path.exists(os.path.join(RP.log_dir(p), "edit_factors.pt"))]
    owners = {}
    for p in want:
        owners.setdefault(N[p]["batch_deltas"], []).append(p)
    W0s, ok_all = {}, True
    for owner, members in owners.items():
        e = N[owner]
        bb = e["backbone"]
        if bb not in W0s:
            name, W0, f = extract(bb, os.path.join(RP.EXT, "models", CKPT[bb]))
            h, ex = tensor_sha256(W0), expected(bb)
            if ex is not None and ex["sha256"] != h:
                raise SystemExit(f"{bb}: the pre-edit weight of the checkpoint ({h}) is not the one the logs were written on ({ex['sha256']})")
            W0s[bb] = W0
        W0 = W0s[bb]
        dt = W0.dtype
        d = RP.log_dir(owner)
        if not os.path.exists(os.path.join(d, "edit_factors.pt")):          # the longest log of the run must be unpacked
            print(f"{owner}: not unpacked (needed for {members}); skipped")
            ok_all = False
            continue
        fac = torch.load(os.path.join(d, "edit_factors.pt"), map_location="cpu")
        with open(os.path.join(d, "metrics.json")) as fh:
            per_batch = json.load(fh)["per_batch"]
        out = os.path.join(RP.LOGS, "batch_deltas", owner)
        ends = {N[p]["n_batches"]: p for p in set(members) | {owner}}
        W, S = W0.clone(), torch.zeros(W0.shape, dtype=torch.float32)
        pending, bad, n_cmp_bad = [], [], 0
        for b in per_batch:
            ids = b["edit_ids"]
            D = fac["U"][ids].double().T @ fac["W"][ids].double()            # float64 sum of the batch's rank-one terms
            Wn = W + D.to(dt)                                                # the write: cast, then add in the weight's type
            A = Wn.float() - W.float()
            if a.compare_with:
                ref = torch.load(os.path.join(a.compare_with, e["dir"], "batch_deltas", f"batch_{b['batch']:03d}.pt"), map_location="cpu")
                n_cmp_bad += int((ref != A).sum())
            S += A
            W = Wn
            pending.append((b["batch"], A))
            nb = b["batch"] + 1
            if nb in ends:
                comp = torch.load(os.path.join(RP.log_dir(ends[nb]), "composite_delta.pt"), map_location="cpu")
                WE = (W0.float() + S).to(dt)
                same = bool(torch.equal(WE.float() - W0.float(), comp))
                print(f"  {ends[nb]:12s} T = {N[ends[nb]]['T']:5d}: rebuilt W_E - W_0 equals the released deployed update in every entry: {same}", flush=True)
                if not same:
                    bad.append(ends[nb])
            if not a.check_only and (len(pending) >= 8 or nb == len(per_batch)):
                os.makedirs(out, exist_ok=True)
                for k, Ak in pending:
                    tmp = os.path.join(out, f"batch_{k:03d}.pt.tmp")
                    torch.save(Ak, tmp)
                    os.replace(tmp, os.path.join(out, f"batch_{k:03d}.pt"))
                pending = []
            elif a.check_only:
                pending = []
        if a.compare_with:
            print(f"  {owner}: entries that differ from the compared batch updates, all batches: {n_cmp_bad}")
            ok_all = ok_all and n_cmp_bad == 0
        if bad:
            ok_all = False
            print(f"{owner}: CHECK FAILED for {bad}")
        else:
            print(f"{owner}: {len(per_batch)} batches" + ("" if a.check_only else f" written to {out}"))
    RP.build_tree()
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
