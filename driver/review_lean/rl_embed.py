"""Lean review, item 10: 100 image and 100 question embeddings of the retrieval layer re-embedded with the reviewer's own loader.

    gq run --mem 2500 --prio 5 --name rl_embed -- python rl_embed.py            -> out/embed.json
Images: the CLIP vision tower of the local LLaVA-1.5 checkpoint, read by this file from the safetensors shards into a CLIPVisionModel
(float32), preprocessing by this file's OWN code (PIL bicubic resize of the shortest edge to 336, centre crop, 1/255, CLIP mean / std)
and, next to it, by the checkpoint's image processor; pooled output; compared with cache2/retr/emb/image_emb.npy (cosine, largest
component difference, equality after the package's storage rounding).  Questions: all-MiniLM-L6-v2 by the model card's transformers
recipe written here (mean pooling over the attention mask, L2 normalisation), string = question.strip().lower().
Also: md5 of the 100 image files against the key list.  Label-free.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import hashlib
import json
import os

import rl_common as C
import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
MODELS = f"{_RP.EXT_VIEW}/data/models"
LLAVA, MINILM = MODELS + "/llava-1.5-7b", MODELS + "/all-MiniLM-L6-v2"
IMG = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"
EMB = C.FIN + "/cache2/retr/emb"
SEED = 77032
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    C.cap_gpu()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
rng = np.random.default_rng(SEED)
out = {"device": str(dev)}


def grid(x):
    x = np.asarray(x, dtype=np.float64)
    x = x / np.sqrt((x * x).sum(-1, keepdims=True))
    return (np.rint(x * 2 ** 24) / 2 ** 24).astype(np.float32)


# ---------------------------------------------------------------- images
ik = [json.loads(x) for x in open(EMB + "/image_keys.jsonl") if x.strip()]
E = np.load(EMB + "/image_emb.npy", mmap_mode="r")
assert E.shape[0] == len(ik)
pick = sorted(int(i) for i in rng.choice(len(ik), 100, replace=False))
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

cfg = json.load(open(LLAVA + "/config.json"))["vision_config"]
vc = CLIPVisionConfig(**{k: v for k, v in cfg.items() if k not in ("model_type", "vocab_size")})
model = CLIPVisionModel(vc)
wm = json.load(open(LLAVA + "/model.safetensors.index.json"))["weight_map"]
want = set(model.state_dict().keys())
sd = {}
for fn in sorted({v for k, v in wm.items() if "vision_tower" in k}):
    with safe_open(os.path.join(LLAVA, fn), framework="pt", device="cpu") as f:
        for k in f.keys():
            if "vision_tower" in k:
                kk = k[k.index("vision_model."):]
                if kk not in want:
                    kk = kk[len("vision_model."):]
                if kk in want:
                    sd[kk] = f.get_tensor(k).float()
missing = sorted(want - set(sd))
model.load_state_dict(sd, strict=True)
model = model.eval().requires_grad_(False).to(dev)
proc = CLIPImageProcessor.from_pretrained(LLAVA)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def my_pixels(img):
    w, h = img.size
    s = 336 / min(w, h)
    nw, nh = (336, int(h * s)) if w <= h else (int(w * s), 336)
    im = img.resize((nw, nh), resample=Image.BICUBIC)
    l, t = (nw - 336) // 2, (nh - 336) // 2
    im = im.crop((l, t, l + 336, t + 336))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return torch.from_numpy(a.transpose(2, 0, 1)[None].copy())


recs = []
for i in pick:
    r = ik[i]
    p = r["image"] if os.path.isabs(r["image"]) else os.path.join(IMG, r["image"])
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()
    with Image.open(p) as im:
        img = im.convert("RGB")
    with torch.no_grad():
        e_proc = model(pixel_values=proc(images=[img], return_tensors="pt")["pixel_values"].float().to(dev)).pooler_output[0].float().cpu().numpy()
        e_mine = model(pixel_values=my_pixels(img).to(dev)).pooler_output[0].float().cpu().numpy()
    st = np.asarray(E[i], dtype=np.float64)
    g = grid(e_proc)
    cos = lambda a: float((a / np.linalg.norm(a)) @ st / np.linalg.norm(st))          # noqa: E731
    recs.append({"i": i, "md5_ok": md5 == r["md5"], "bitwise_after_rounding": bool((g == np.asarray(E[i])).all()), "max_abs_diff": float(np.abs(g.astype(np.float64) - st).max()),
                 "cos_processor": cos(e_proc.astype(np.float64)), "cos_own_preprocessing": cos(e_mine.astype(np.float64))})
out["images"] = {"n": len(recs), "missing_tensors": missing, "n_tower_tensors": len(sd), "md5_ok": sum(r["md5_ok"] for r in recs),
                 "bitwise_after_rounding": sum(r["bitwise_after_rounding"] for r in recs), "max_abs_diff": max(r["max_abs_diff"] for r in recs),
                 "min_cos_processor": min(r["cos_processor"] for r in recs), "min_cos_own_preprocessing": min(r["cos_own_preprocessing"] for r in recs),
                 "stored_unit_norm_max_dev": float(max(abs(np.linalg.norm(np.asarray(E[i], dtype=np.float64)) - 1) for i in pick))}
del model
# ---------------------------------------------------------------- questions (CPU, the model card's recipe)
qk = [json.loads(x) for x in open(EMB + "/question_keys.jsonl") if x.strip()]
Q = np.load(EMB + "/question_emb.npy", mmap_mode="r")
assert Q.shape[0] == len(qk)
pq = sorted(int(i) for i in rng.choice(len(qk), 100, replace=False))
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MINILM)
mm = AutoModel.from_pretrained(MINILM).eval().requires_grad_(False)
qr = []
for i in pq:
    s = qk[i]["question_key"]
    enc = tok([s], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        te = mm(**enc)[0]
    m = enc["attention_mask"].unsqueeze(-1).float()
    e = ((te * m).sum(1) / m.sum(1).clamp(min=1e-9))[0].double().numpy()
    g = grid(e)
    st = np.asarray(Q[i], dtype=np.float64)
    qr.append({"i": i, "key_is_stripped_lower": s == s.strip().lower(), "bitwise_after_rounding": bool((g == np.asarray(Q[i])).all()),
               "max_abs_diff": float(np.abs(g.astype(np.float64) - st).max()), "cos": float((e / np.linalg.norm(e)) @ st / np.linalg.norm(st))})
out["questions"] = {"n": len(qr), "key_is_stripped_lower": sum(r["key_is_stripped_lower"] for r in qr), "bitwise_after_rounding": sum(r["bitwise_after_rounding"] for r in qr),
                    "max_abs_diff": max(r["max_abs_diff"] for r in qr), "min_cos": min(r["cos"] for r in qr)}
out["ok"] = bool(out["images"]["md5_ok"] == 100 and out["images"]["min_cos_processor"] > 1 - 1e-6 and out["images"]["min_cos_own_preprocessing"] > 0.995
                 and out["questions"]["min_cos"] > 1 - 1e-6 and not missing)
out["rows_images"], out["rows_questions"] = recs, qr
C.dump("embed.json", out)
print({k: v for k, v in out.items() if not k.startswith("rows")}, flush=True)
