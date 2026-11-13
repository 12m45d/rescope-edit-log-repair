"""Shared paths, registries and model-free helpers of the key caches of the final experiments (10_final/cache).

Nothing here writes outside 10_final/cache.  The library `rescope` and the harness are imported READ-ONLY:
bytecode writing is switched off and the library's private harness cache is re-pointed to work/hcache
(RESCOPE_CACHE), so that no file appears inside the package or inside an existing experiment directory.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True

ROOT = f"{_RP.ROOT}"
CODE = ROOT + "/02_research/code"
EXP = ROOT + "/02_research/experiments"
OUT = EXP + "/10_final/cache"
WORK = OUT + "/work"
LOGDIR = OUT + "/logs"
HN_OLD = EXP + "/08_rescope/p11_retrofit/out/hard_negatives"
HN_NEW = OUT + "/hard_negatives"
BENIGN = OUT + "/benign"
os.environ.setdefault("RESCOPE_CACHE", WORK + "/hcache")
for _p in (EXP + "/09_testbed_scale/code", EXP + "/07_repair_v3", CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 20261110                       # every seeded draw of this build derives from it
N_DECODE = 11                         # full forwards of a decoder-v2 decode that contain the prompt's last token
# backbone -> edited layer, families (stream family -> master log = the longest log of the stream), development log
BACKBONES = {
    # [changed for the public artifact] master logs of the VLKEB families: the released logs with the SAME edit streams (Q-M2000, L-M1000; the files
    # edit_batch.jsonl are byte-identical to those of the internal logs TB-Q2000-ae-l2_3 / TB-L1000-ae-l2_3 that the build used and that are not
    # released).  The capture runs the PRE-EDIT model, so the keys do not depend on which log of the stream is loaded.  Family QK (an MMKE-Bench
    # stress log that the paper does not report) is not released: its 2000 sequences cannot be captured (fc_assemble.py --partial).
    "qwen25vl": {"layer": 20, "families": {"Q": "TB-Q2000-memitw", "QK": "TB-Q500-mmke-l2_3"}, "vlkeb": "Q",
                 "dev": "TB-Q100-ae-l2_3", "mem_gb": 10.3, "reserve_mib": 10500},
    "llava": {"layer": 16, "families": {"L": "TB-L1000-memitw"}, "vlkeb": "L",
              "dev": "TB-L100-ae-l2_3", "mem_gb": 16.6, "reserve_mib": 17000},
}
DATASET = {"Q": "VLKEB", "L": "VLKEB", "QK": "MMKE-Bench"}
KINDS = ("rel", "tgen", "igen", "probe", "hn_i", "hn_q", "hn_x", "benign")
HN_KIND = {"HN-I": "hn_i", "HN-Q": "hn_q", "HN-X": "hn_x"}


def say(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl_read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def jsonl_write(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)


def jsonl_append(row, path):
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def useed(*parts) -> int:
    """Deterministic 32-bit seed from SEED and any labels (sha1)."""
    return int(hashlib.sha1("|".join(str(p) for p in (SEED,) + parts).encode()).hexdigest()[:8], 16)


def norm_q(s: str) -> str:
    """Question-string identity used by every 'same question' test: stripped, lower case."""
    return (s or "").strip().lower()


def mid(path: str) -> str:
    """Freebase MID (entity directory) of a VLKEB image path."""
    return path.split("/")[0]


# ----------------------------------------------------------------------------- testbed / library (model-free)
_TB = None


def tb():
    """tb_common of the testbed with the 09_testbed_scale registry installed (new log names, MMKE loaders)."""
    global _TB
    if _TB is None:
        import scale_registry  # noqa: F401  (registers the long logs in tb_common.LOGS, in-process)
        import tb_common as TB
        _TB = TB
    return _TB


def import_rescope(tries: int = 10, wait: float = 60.0):
    """The library, read-only.  Another agent edits it at the same time: retry if an import fails."""
    last = None
    for _ in range(tries):
        try:
            import rescope
            return rescope
        except Exception as e:                      # a file is being edited: wait and retry
            last = e
            for k in [k for k in sys.modules if k == "rescope" or k.startswith("rescope.")]:
                del sys.modules[k]
            print(f"import rescope failed ({type(e).__name__}: {e}); retry in {wait:.0f} s", flush=True)
            time.sleep(wait)
    raise last


def image_path(img: str) -> str:
    from vendor_v7.vlm import VLKEB_IMG_DIR
    return img if os.path.isabs(img) else os.path.join(VLKEB_IMG_DIR, img)


_MD5 = None
_MD5_PATH = WORK + "/image_md5.json"


def image_md5(img):
    """md5 of the image FILE (identity of the pixels the model sees); None for text rows.  Cached on disk."""
    global _MD5
    if img is None:
        return None
    if _MD5 is None:
        _MD5 = jload(_MD5_PATH) if os.path.exists(_MD5_PATH) else {}
    if img not in _MD5:
        h = hashlib.md5()
        with open(image_path(img), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _MD5[img] = h.hexdigest()
    return _MD5[img]


def image_md5_flush():
    if _MD5 is not None:
        jdump(_MD5, _MD5_PATH, indent=None)


def query_key(ptype: str, prompt: str, image) -> str:
    """Identity of a query = the model input up to the last prompt token: (ptype, question text, image pixels)."""
    return json.dumps([ptype, prompt, image_md5(image)], ensure_ascii=False)


# ----------------------------------------------------------------------------- streams, probes, pools
_FAM = {}


def family(fam: str) -> dict:
    """Model-free data of a stream family: items (edit order), probes (500 items), probe rows (1000 rows of the
    master log's probe_records_v2.jsonl: the pre-edit fields are the same in every log of the family), the
    harness REPAIR / HELD-OUT pool of every probe item and the library's row annotation (splits.annotate)."""
    if fam not in _FAM:
        TB = tb()
        bb = [b for b, c in BACKBONES.items() if fam in c["families"]][0]
        name = BACKBONES[bb]["families"][fam]
        cfg = TB.LOGS[name]
        items, recs = TB.load_stream(cfg["stream"])
        probes, pinfo = TB.select_probes(items, key_exclude=(bb == "llava"))
        rows = TB.jsonl_read(os.path.join(TB.OUT_ROOT, name, "probe_records_v2.jsonl"))
        assert all(probes[r["probe_idx"]]["uid"] == r["uid"] for r in rows), fam
        import types
        from harness import sampler as HS
        pool = HS.split_pools(types.SimpleNamespace(probes=probes))
        rs = import_rescope()
        ann = rs.splits.annotate(rows, pool)
        _FAM[fam] = {"family": fam, "backbone": bb, "master": name, "cfg": cfg, "items": items, "recs": recs,
                     "probes": probes, "probe_info": pinfo, "rows": rows, "pool": pool, "ann": ann,
                     "dataset": DATASET[fam]}
    return _FAM[fam]


def degenerate_flags(y0: str) -> dict:
    """Degenerate-answer flags of a decoder-v2 pre-edit answer (counted in the tables)."""
    s = (y0 or "").strip()
    bare = s.rstrip(".!").strip().lower()
    return {"deg_empty": not s, "deg_single_char": len(s) == 1, "deg_qmark": s == "?",
            "deg_none_no": bare in ("none", "no")}
