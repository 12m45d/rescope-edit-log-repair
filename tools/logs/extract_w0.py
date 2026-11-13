#!/usr/bin/env python
"""Extract the pre-edit weight W_0 of the edited module from a PUBLIC checkpoint and print its hash.

    python extract_w0.py --backbone qwen25vl --checkpoint <dir of Qwen2.5-VL-3B>   [--out W0.pt]
    python extract_w0.py --backbone llava    --checkpoint <dir of llava-1.5-7b>    [--out W0.pt]
    python extract_w0.py --backbone gpt2-xl  --checkpoint <dir of gpt2-xl>         [--out W0.pt]

The edited module is one MLP output projection (paper, Sec. 5.1):
    qwen25vl  layer 20   tensor  model.layers.20.mlp.down_proj.weight                      bfloat16 (2048, 11008)
    llava     layer 16   tensor  language_model.model.layers.16.mlp.down_proj.weight       float16  (4096, 11008)
    gpt2-xl   layer 17   tensor  h.17.mlp.c_proj.weight (transformer.h.17.mlp.c_proj)      float32  (6400, 1600)  [Conv1D: (d_in, d_out)]
The hash is the SHA-256 of the tensor's bytes in its stored type and row-major order (little endian).  Expected values:
W0_SHA256.json next to this script (and in every data archive).  Needs torch and safetensors; no GPU; nothing is downloaded.
This script stands alone (it is also shipped inside the data archives).
"""
import argparse
import glob
import hashlib
import json
import os

SUFFIX = {"qwen25vl": "layers.20.mlp.down_proj.weight", "llava": "layers.16.mlp.down_proj.weight", "gpt2-xl": "h.17.mlp.c_proj.weight"}


def tensor_sha256(t) -> str:
    import torch
    if t.dtype == torch.bfloat16:                        # numpy has no bfloat16: hash the raw 16-bit words
        raw = t.contiguous().view(torch.int16).numpy().tobytes()
    else:
        raw = t.contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def extract(backbone: str, checkpoint: str):
    """-> (tensor name, tensor in its stored type, file)."""
    from safetensors import safe_open
    suf = SUFFIX[backbone]
    hits = []
    for f in sorted(glob.glob(os.path.join(checkpoint, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.endswith(suf) and "visual" not in k and "vision" not in k:
                    hits.append((k, s.get_tensor(k), os.path.basename(f)))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one tensor ending in {suf!r} under {checkpoint}, found {[h[0] for h in hits]}")
    return hits[0]


def expected(backbone: str):
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "W0_SHA256.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get(backbone)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(SUFFIX))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    name, W0, f = extract(a.backbone, a.checkpoint)
    h = tensor_sha256(W0)
    e = expected(a.backbone)
    print(f"{a.backbone}: {name} in {f}: {W0.dtype} {tuple(W0.shape)}")
    print(f"sha256 {h}" + ("" if e is None else ("   == expected" if e["sha256"] == h else f"   DIFFERS from the expected {e['sha256']}")))
    if a.out:
        import torch
        torch.save({"weight": W0, "name": name, "sha256": h, "backbone": a.backbone}, a.out)
        print("written:", a.out)
    if e is not None and e["sha256"] != h:
        raise SystemExit(1)
