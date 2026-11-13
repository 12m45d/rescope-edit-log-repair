"""P12 verifier -- split leak, refined: HELD-OUT rows whose model input is identical to a REPAIR-pool row.
The experiment's twin test compares (ptype, prompt, image FILE NAME).  Here the image is compared by content
(md5 of the file bytes, and md5 of the decoded RGB pixels + size), so the same picture stored under two file names
is also found.  CPU only.  -> out/leak_pixels.json"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import hashlib
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402

IMG = f"{_RP.EXT_VIEW}/data/datasets/VLKEB/mmkb_images"
if not os.path.isdir(IMG):
    sys.path.insert(0, os.path.join(VC.ROOT, "02_research", "code"))
    from vendor_v7.vlm import VLKEB_IMG_DIR as IMG  # noqa: E402
_h = {}


def hashes(fn):
    if fn not in _h:
        p = os.path.join(IMG, fn)
        b = open(p, "rb").read()
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            px = hashlib.md5(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
        _h[fn] = (hashlib.md5(b).hexdigest(), px)
    return _h[fn]


out = {}
for log in VC.LOGS:
    rows = VC.records(log)
    pool = VC.my_split(rows)
    img = [r for r in rows if r["ptype"] == "image"]
    kname = lambda r: (r["prompt"], r["image"])                     # noqa: E731  the experiment's key
    kpix = lambda r: (r["prompt"], hashes(r["image"])[1])           # noqa: E731  same pixels
    kbyte = lambda r: (r["prompt"], hashes(r["image"])[0])          # noqa: E731  same file bytes
    res = {}
    for nm, key in (("file_name", kname), ("file_bytes", kbyte), ("pixels", kpix)):
        rep = {}
        for r in img:
            if pool[r["probe_idx"]] == "repair":
                rep.setdefault(key(r), []).append(r)
        tw = [r for r in img if pool[r["probe_idx"]] == "heldout" and r["y0_valid"] and key(r) in rep]
        rc = [r for r in img if pool[r["probe_idx"]] == "repair" and r["y0_valid"] and r["is_complaint_D1"]]
        res[nm] = {"n_valid_heldout_twins": len(tw), "twin_complaints": int(sum(bool(r["is_complaint_D1"]) for r in tw)),
                   "twin_complaints_of_repair_complaint": int(sum(bool(r["is_complaint_D1"]) and any(x["is_complaint_D1"] for x in rep[key(r)]) for r in tw)),
                   "twin_groups": sorted(r["group"] for r in tw),
                   "same_y0_as_repair_twin": int(sum(any(x["y0_text"] == r["y0_text"] for x in rep[key(r)]) for r in tw)),
                   "repair_complaints_distinct_queries": len({key(r) for r in rc}), "repair_complaints_image": len(rc),
                   "distinct_image_queries": len({key(r) for r in img})}
    # text rows: identical prompt strings across pools
    txt = [r for r in rows if r["ptype"] == "text"]
    rp = {r["prompt"] for r in txt if pool[r["probe_idx"]] == "repair"}
    res["text_twins_valid_heldout"] = int(sum(pool[r["probe_idx"]] == "heldout" and r["y0_valid"] and r["prompt"] in rp for r in txt))
    extra = sorted(set(res["pixels"]["twin_groups"]) - set(res["file_name"]["twin_groups"]))
    byg = {r["group"]: r for r in rows}
    res["extra_twins_found_by_pixels"] = [{"group": g, "image": byg[g]["image"], "is_complaint_D1": bool(byg[g]["is_complaint_D1"])} for g in extra]
    res["n_valid_heldout"] = int(sum(pool[r["probe_idx"]] == "heldout" and r["y0_valid"] for r in rows))
    res["n_valid_heldout_complaints"] = int(sum(pool[r["probe_idx"]] == "heldout" and r["y0_valid"] and bool(r["is_complaint_D1"]) for r in rows))
    out[log] = res
    print(log, {k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "twin_groups"}) for k, v in res.items()})
VC.jdump(out, os.path.join(VC.VOUT, "leak_pixels.json"))
