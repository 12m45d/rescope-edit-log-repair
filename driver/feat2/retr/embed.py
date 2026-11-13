"""Task 2: embeddings of every distinct image content and of every distinct question string of the attribute tables.

    python embed.py keys [--out DIR]          CPU, deterministic: the key lists (image md5 -> one file; question keys) from the attribute tables
    python embed.py questions                 CPU: all-MiniLM-L6-v2, resumable parts
    $GQ run --mem 2600 --prio 4 --name retr_img -- python embed.py images        GPU: CLIP vision tower of LLaVA-1.5, resumable parts
    python embed.py assemble                  parts -> image_emb.npy, question_emb.npy, meta.json
    python embed.py status

Files under <DATA>/emb (DATA = 10_final/cache2/retr; shared by the backbones: an image content or a question string that occurs in
both caches has ONE row):
  image_keys.jsonl       one line per distinct image content, sorted by md5: {"md5", "image" (the representative file: the smallest
                         benchmark path with that content), "n_paths", "backbones"}
  question_keys.jsonl    one line per distinct question key (question.strip().lower()), sorted (code points): {"question_key", "backbones"}
  keys_meta.json         counts, md5 of the two key files, the attribute tables they were built from
  image_emb.npy          float32 (n_images, 1024): row i = the embedding of the image content image_keys[i]
  question_emb.npy       float32 (n_questions, 384): row i = the embedding of question_keys[i]
  meta.json              encoders, exact preprocessing, versions, md5 of the matrices, norm statistics, GPU memory measured
  parts/<tag>/           resumable parts (256 keys each; tag = md5 of the key file they belong to)
Encoders, preprocessing and the storage form (unit norm, float32, 2^-24 grid): encoders.py.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import argparse
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402

import numpy as np  # noqa: E402

PART = 256


def image_path(img: str) -> str:
    vlkeb = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"       # vendor_v7.vlm.VLKEB_IMG_DIR
    return img if os.path.isabs(img) else os.path.join(vlkeb, img)


# ----------------------------------------------------------------------------- key lists
def build_keys(out_dir: str, backbones=C.BACKBONES, data_dir: str | None = None) -> dict:
    data_dir = C.DATA if data_dir is None else data_dir
    img, qs, src = {}, {}, {}
    for bb in backbones:
        p = os.path.join(data_dir, bb, "attributes.jsonl")
        src[bb] = {"file": p, "md5": C.file_md5(p)}
        for r in C.jsonl_read(p):
            if r["image_md5"] is not None:
                e = img.setdefault(r["image_md5"], {"paths": set(), "backbones": set()})
                e["paths"].add(r["image"])
                e["backbones"].add(bb)
            qs.setdefault(r["question_key"], set()).add(bb)
    irows = [{"md5": m, "image": min(img[m]["paths"]), "n_paths": len(img[m]["paths"]), "backbones": sorted(img[m]["backbones"])} for m in sorted(img)]
    qrows = [{"question_key": q, "backbones": sorted(qs[q])} for q in sorted(qs)]
    os.makedirs(out_dir, exist_ok=True)
    pi, pq = os.path.join(out_dir, "image_keys.jsonl"), os.path.join(out_dir, "question_keys.jsonl")
    C.jsonl_write(irows, pi)
    C.jsonl_write(qrows, pq)
    meta = {"n_images": len(irows), "n_questions": len(qrows), "image_keys_md5": C.file_md5(pi), "question_keys_md5": C.file_md5(pq),
            "backbones": list(backbones), "attribute_tables": src,
            "n_images_shared_by_backbones": sum(len(r["backbones"]) > 1 for r in irows),
            "n_questions_shared_by_backbones": sum(len(r["backbones"]) > 1 for r in qrows),
            "n_image_contents_with_several_paths": sum(r["n_paths"] > 1 for r in irows),
            "image_root_of_relative_paths": image_path("x")[:-2], "question_key_rule": "question.strip().lower()"}
    C.jdump(meta, os.path.join(out_dir, "keys_meta.json"))
    return meta


def load_keys(kind: str, emb_dir: str | None = None) -> list:
    emb_dir = C.EMB if emb_dir is None else emb_dir
    return C.jsonl_read(os.path.join(emb_dir, f"{kind}_keys.jsonl"))


def part_dir(kind: str) -> str:
    tag = C.file_md5(os.path.join(C.EMB, f"{kind}_keys.jsonl"))[:12]
    return os.path.join(C.EMB, "parts", f"{kind}_{tag}")


def parts(kind: str) -> list:
    n = len(load_keys(kind))
    d = part_dir(kind)
    return [(p, a, min(a + PART, n), os.path.join(d, f"{p:05d}.npy")) for p, a in enumerate(range(0, n, PART))]


def save_part(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- questions (CPU)
def embed_questions(limit=None):
    import torch
    torch.set_num_threads(4)
    from encoders import QuestionEncoder
    keys = [r["question_key"] for r in load_keys("question")]
    enc = QuestionEncoder("sentence-transformers", "cpu")
    todo = [t for t in parts("question") if not os.path.exists(t[3])]
    print(f"questions: {len(keys)} keys, {len(todo)} parts to do", flush=True)
    t0, nt = time.time(), []
    for k, (p, a, b, path) in enumerate(todo[:limit]):
        sub = keys[a:b]
        nt += [enc.n_tokens(s) for s in sub]
        if max(nt) > enc.max_tokens:
            raise RuntimeError("a question is longer than the encoder's 256 tokens (it would be truncated)")
        save_part(path, enc.embed(sub))
        if (k + 1) % 5 == 0 or k + 1 == len(todo):
            print(f"  part {p} ({k + 1}/{len(todo)}) {time.time() - t0:.0f}s", flush=True)
    C.jdump({"info": enc.info, "seconds": time.time() - t0, "n_parts_done_now": len(todo[:limit]), "max_tokens_seen": max(nt) if nt else None},
            os.path.join(part_dir("question"), f"run_{int(time.time())}.json"))


# ----------------------------------------------------------------------------- images (GPU)
def gpu_cap():
    """Cap the process at GQ_MEM - 600 MiB (README of the queue); -> the cap in MiB (None outside the queue)."""
    import torch
    if "GQ_MEM" not in os.environ or not torch.cuda.is_available():
        return None
    cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    return cap // 2 ** 20


def gpu_used_mib():
    """Memory of THIS process as nvidia-smi counts it (MiB), or None."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem)
    except Exception:                                                   # noqa: BLE001
        pass
    return None


def embed_images(limit=None, device="cuda:0"):
    import torch
    torch.set_num_threads(1)                                            # preprocessing runs in 4 Python threads
    cap = gpu_cap() if device.startswith("cuda") else None
    from encoders import ImageEncoder
    keys = load_keys("image")
    todo = [t for t in parts("image") if not os.path.exists(t[3])]
    print(f"images: {len(keys)} keys, {len(todo)} parts to do, device {device}, cap {cap} MiB", flush=True)
    if not todo:
        return
    enc = ImageEncoder(device)
    t0, peak_smi = time.time(), None

    def pv(r):
        with open(image_path(r["image"]), "rb") as f:
            raw = f.read()
        if C.bytes_md5(raw) != r["md5"]:
            raise RuntimeError(f"{r['image']}: the file's md5 is not {r['md5']}")
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(io.BytesIO(raw)) as im:
            rgb = im.convert("RGB")
        return enc.proc(images=[rgb], return_tensors="pt")["pixel_values"].to(torch.float32)

    from concurrent.futures import ThreadPoolExecutor
    from encoders import finalize, IMG_DIM
    with ThreadPoolExecutor(4) as ex:
        for k, (p, a, b, path) in enumerate(todo[:limit]):
            out = np.zeros((b - a, IMG_DIM), dtype=np.float32)
            for i, x in enumerate(ex.map(pv, keys[a:b])):
                out[i] = finalize(enc.raw(x).numpy())[0]
            save_part(path, out)
            if k == 0 or (k + 1) % 4 == 0 or k + 1 == len(todo):
                smi = gpu_used_mib()
                peak_smi = smi if peak_smi is None else max(peak_smi, smi or 0)
                print(f"  part {p} ({k + 1}/{len(todo)}) {time.time() - t0:.0f}s  nvidia-smi {smi} MiB", flush=True)
    rec = {"info": enc.info, "seconds": time.time() - t0, "n_parts_done_now": len(todo[:limit]), "gq_mem": os.environ.get("GQ_MEM"), "cap_mib": cap}
    if device.startswith("cuda"):
        rec["gpu"] = {"name": torch.cuda.get_device_name(0), "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
                      "max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2 ** 20, "nvidia_smi_process_mib_peak_seen": peak_smi}
    C.jdump(rec, os.path.join(part_dir("image"), f"run_{int(time.time())}.json"))
    print("done", rec.get("gpu"), flush=True)


# ----------------------------------------------------------------------------- assemble
def assemble():
    from encoders import on_grid
    meta = {"keys": C.jload(os.path.join(C.EMB, "keys_meta.json"))}
    for kind, dim in (("image", 1024), ("question", 384)):
        ps = parts(kind)
        miss = [p for p, a, b, path in ps if not os.path.exists(path)]
        if miss:
            raise RuntimeError(f"{kind}: {len(miss)} of {len(ps)} parts are missing (e.g. {miss[:5]}); run `embed.py {kind}s`")
        E = np.concatenate([np.load(path) for _, _, _, path in ps])
        n = len(load_keys(kind))
        if E.shape != (n, dim) or E.dtype != np.float32:
            raise RuntimeError(f"{kind}: assembled {E.shape} {E.dtype}, expected ({n}, {dim}) float32")
        nrm = np.sqrt((E.astype(np.float64) ** 2).sum(1))
        if not on_grid(E) or not bool(np.isfinite(E).all()) or float(np.abs(nrm - 1).max()) > 1e-6:
            raise RuntimeError(f"{kind}: embeddings are not unit-norm vectors on the 2^-24 grid")
        out = os.path.join(C.EMB, f"{kind}_emb.npy")
        tmp = out + f".tmp{os.getpid()}.npy"
        np.save(tmp, E)
        os.replace(tmp, out)
        runs = sorted(f for f in os.listdir(part_dir(kind)) if f.startswith("run_"))
        recs = [C.jload(os.path.join(part_dir(kind), f)) for f in runs]
        meta[kind] = {"file": f"{kind}_emb.npy", "keys_file": f"{kind}_keys.jsonl", "shape": list(E.shape), "dtype": "float32", "md5": C.file_md5(out),
                      "keys_md5": C.file_md5(os.path.join(C.EMB, f"{kind}_keys.jsonl")),
                      "norm_minus_1": {"max_abs": float(np.abs(nrm - 1).max()), "mean_abs": float(np.abs(nrm - 1).mean())},
                      "on_2^-24_grid": True, "n_parts": len(ps), "part_size": PART,
                      "encoder": recs[-1]["info"] if recs else None,
                      "runs": [{k: v for k, v in r.items() if k != "info"} for r in recs],
                      "encoder_same_in_every_run": all({k: v for k, v in r["info"].items() if k != "device"} == {k: v for k, v in recs[-1]["info"].items() if k != "device"}
                                                       for r in recs) if recs else None}
        print(f"{kind}: {E.shape} -> {out}  md5 {meta[kind]['md5']}  max | |e| - 1 | = {meta[kind]['norm_minus_1']['max_abs']:.2e}", flush=True)
    meta["storage"] = ("unit-norm float32 rows on the 2^-24 grid: the float64 dot product of two rows is exact (encoders.py); "
                       "score = that dot product (no further normalisation)")
    meta["built_by"] = "feat2/retr/embed.py (encoders.py)"
    C.jdump(meta, os.path.join(C.EMB, "meta.json"))


def status():
    for kind in ("image", "question"):
        ps = parts(kind)
        done = sum(os.path.exists(t[3]) for t in ps)
        print(f"{kind}: {done}/{len(ps)} parts in {part_dir(kind)}; assembled: {os.path.exists(os.path.join(C.EMB, kind + '_emb.npy'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("keys", "questions", "images", "assemble", "status"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="number of parts to do in this run")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.cmd == "keys":
        print(build_keys(a.out or C.EMB))
    elif a.cmd == "questions":
        embed_questions(a.limit)
    elif a.cmd == "images":
        embed_images(a.limit, a.device)
    elif a.cmd == "assemble":
        assemble()
    else:
        status()
