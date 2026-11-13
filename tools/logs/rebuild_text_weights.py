#!/usr/bin/env python
"""Rebuild the weights of the public-code text logs (GPT-2 XL, ROME) from the public checkpoint and the logged factors.

    python tools/logs/rebuild_text_weights.py            (CPU, about one minute; needs $RESCOPE_EXT/models/gpt2-xl)

The archive logs_text.tar holds the factors of every update (left and right vector as the public ROME code returned them)
but no weight.  The public code applies update i as  W <- W + outer(left_i, right_i)  in float32, in the order of the stream.
This script replays that rule from the pre-edit weight of the public checkpoint and writes the files that the registry of the
logs names (logs_edit/registry.json):
    logs_edit/W0_transformer.h.17.mlp.c_proj.weight.pt        the pre-edit weight
    logs_edit/stream/snapshots/W_T{100,500,1000,2000}.pt      the deployed weights of G-R100 ... G-R2000
    logs_edit/dev/snapshots/W_T100.pt                         the deployed weight of the development log
Every file is written only if its SHA-256 equals the hash that the registry recorded when the logs were built.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import rescope_paths as RP  # noqa: E402
import torch  # noqa: E402
from extract_w0 import extract  # noqa: E402

NAME = "transformer.h.17.mlp.c_proj.weight"


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    base = os.path.join(RP.LOGS, "logs_text", "logs_edit")
    with open(os.path.join(base, "registry.json")) as f:
        reg = json.load(f)
    _, W0, _ = extract("gpt2-xl", os.path.join(RP.EXT, "models", "gpt2-xl"))
    W0 = W0.float().clone()
    h0 = sha(W0)
    want0 = next(iter(reg.values()))["pre_edit_sha256"]
    print(f"pre-edit weight {tuple(W0.shape)} sha256 {h0}: {'== registry' if h0 == want0 else 'DIFFERS from the registry ' + want0}")
    if h0 != want0:
        raise SystemExit(1)
    torch.save({"weight": W0, "sha256": {"sha256": h0, "dtype": "float32", "shape": list(W0.shape)}, "name": NAME,
                "note": "pre-edit Conv1D weight (d_in, d_out) float32; y = x @ W + b"}, os.path.join(base, "W0_" + NAME + ".pt"))
    ok = True
    for src in ("dev", "stream"):
        fac = torch.load(os.path.join(base, src, "factors.pt"), map_location="cpu")
        ends = {e["T"]: (n, e["deployed_sha256"]) for n, e in reg.items() if e["source"] == src}
        W = W0.clone()
        os.makedirs(os.path.join(base, src, "snapshots"), exist_ok=True)
        for i in range(max(ends)):
            W += fac["left"][i].unsqueeze(1) @ fac["right"][i].unsqueeze(0)          # the public code's update, float32
            if i + 1 in ends:
                n, want = ends[i + 1]
                h = sha(W)
                print(f"  {n}: deployed weight after {i + 1} updates sha256 {h}: {'== registry' if h == want else 'DIFFERS from the registry'}", flush=True)
                if h == want:
                    torch.save({"weight": W.clone(), "T": i + 1, "sha256": {"sha256": h, "dtype": "float32", "shape": list(W.shape)}, "name": NAME, "log": src,
                                "at": "rebuilt by tools/logs/rebuild_text_weights.py"}, os.path.join(base, src, "snapshots", f"W_T{i + 1}.pt"))
                ok = ok and h == want
    RP.build_tree()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
