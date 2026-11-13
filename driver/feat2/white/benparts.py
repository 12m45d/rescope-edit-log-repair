"""white.benparts: the three parts of the label-free benign pool (PROTOCOL.md section 6, amendment "Review round 1", item R2).

    BEN-W   20 percent of the source items   statistics of baselines (mean key, second moment), extra null-repair keys
    BEN-S   45 percent                       complaint streams, second calibration source
    BEN-T   35 percent                       fresh test rows, used for nothing else

SPLIT RULE (deterministic; the unit is the SOURCE ITEM = field `uid` of 10_final/cache/benign/<backbone>/rows.jsonl)
  1. items = the distinct uids of the pool, sorted as strings (the order of the file plays no role);
  2. perm  = numpy.random.default_rng(20261111).permutation(len(items));
  3. the items perm[0 : n_W] are BEN-W, perm[n_W : n_W + n_S] are BEN-S, the rest are BEN-T, with
     n_W = round(0.20 n), n_S = round(0.45 n) (integer arithmetic, halves up), n_T = n - n_W - n_S;
  4. a row belongs to the part of its item: an item's image row and text row are never separated.
The pool holds the same rows for every backbone (same sid, uid, ptype, prompt, image, answer subset; only the text format and
the flags against the backbone's edit stream differ); `build` checks this and refuses to write otherwise, so the item split is
the same for all backbones by construction (same list of uids, same seed).

FILE  10_final/cache2/benign_parts/<backbone>.json   (written by `python -m white.benparts build`; no time stamp inside: a re-run
      reproduces it byte for byte, tests/t_parts.py)
  parts     {sid: 'BEN-W' | 'BEN-S' | 'BEN-T'}           every row of the pool
  items     {uid: part}                                   every source item
  n_items, n_rows, percent_of_items, seed, rule, source (path, md5 and identity hash of the pool file the split was made from)
  summary   per part: rows by ptype; rows of the answer subset (decoder-v2 pre-edit answer cached), of them valid and with a
            teacher-forced key; rows whose question is the question or rephrase of an edit of the backbone's long stream
            (`q_in_edit`; also per stream prefix T); every boolean flag of the pool's own fields, also per family of `overlap`;
            question strings and image contents that occur in two parts (items are disjoint, strings need not be)

READER
    P = white.benparts.load("qwen25vl")            # verifies that the pool file is the one the split was made from
    P.part(sid), P.sids("BEN-W"), P.rows("BEN-T", kc)   (positions in the index of rescope.KeyCache: fx sequences), P.summary
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import common as C

SEED = 20261111
PARTS = ("BEN-W", "BEN-S", "BEN-T")
PERCENT = {"BEN-W": 20, "BEN-S": 45, "BEN-T": 35}
USE = {"BEN-W": "statistics of baselines (mean key, second moment), extra null-repair keys",
       "BEN-S": "complaint streams, second calibration source",
       "BEN-T": "fresh test rows, used for nothing else (D1 on its rows with a cached pre-edit answer, firing on all its rows)"}
IDENTITY_FIELDS = ("sid", "uid", "ptype", "prompt", "image", "answer_subset")      # what "the same pool" means across backbones
FLAGS = ("q_in_edit", "q_in_edit_exact", "img_in_edit", "q_in_probe", "q_in_probe_exact", "img_in_probe", "query_in_probe",
         "img_entity_in_edit", "img_entity_in_probe")
PREFIXES = (100, 200, 500, 1000, 2000)                 # stream prefixes T for which the near-scope rows are counted


def pool_path(backbone: str) -> str:
    return os.path.join(C.CACHE, "benign", backbone, "rows.jsonl")


def parts_path(backbone: str, directory: str | None = None) -> str:
    return os.path.join(C.PARTS_DIR if directory is None else directory, f"{backbone}.json")


def pool_backbones() -> list:
    """Backbones that have a benign pool under 10_final/cache/benign (the paper's two first, then any other, sorted)."""
    d = os.path.join(C.CACHE, "benign")
    have = sorted(b for b in os.listdir(d) if os.path.exists(os.path.join(d, b, "rows.jsonl")))
    return [b for b in C.BACKBONES if b in have] + [b for b in have if b not in C.BACKBONES]


def item_split(uids) -> dict:
    """{uid: part} for the distinct source items in `uids` (rule: module docstring).  Depends on the SET of uids and the seed only."""
    items = sorted({str(u) for u in uids})
    n = len(items)
    n_w = (PERCENT["BEN-W"] * n + 50) // 100
    n_s = (PERCENT["BEN-S"] * n + 50) // 100
    if n_w < 1 or n_s < 1 or n - n_w - n_s < 1:
        raise ValueError(f"cannot split {n} items into three non-empty parts")
    perm = np.random.default_rng(SEED).permutation(n)
    out = {}
    for rank, i in enumerate(perm.tolist()):
        out[items[i]] = PARTS[0] if rank < n_w else (PARTS[1] if rank < n_w + n_s else PARTS[2])
    return out


def pool_identity(rows) -> str:
    """sha1 of what a pool row IS, independent of the backbone: sid, uid, ptype, prompt, image, answer subset, in file order."""
    h = hashlib.sha1()
    for r in rows:
        h.update(C.jdumps([r.get(f) for f in IDENTITY_FIELDS], indent=None).encode())
    return h.hexdigest()


def _count(rows, pred) -> dict:
    """{'all': n, 'image': n, 'text': n} of the rows that satisfy pred."""
    sel = [r for r in rows if pred(r)]
    return {"all": len(sel), "image": sum(r["ptype"] == "image" for r in sel), "text": sum(r["ptype"] == "text" for r in sel)}


def _image_id(r, index_row=None):
    """Identity of an image CONTENT: the md5 the key cache stores for the row, else the path."""
    if r.get("image") is None:
        return None
    return (index_row or {}).get("image_md5") or r["image"]


def summarise(rows, part_of_sid, index_rows=None) -> dict:
    """The summary block of the parts file (counts only; every field comes from the pool's own rows)."""
    idx = {} if index_rows is None else {e["sid"]: e for e in index_rows}
    by = {p: [r for r in rows if part_of_sid[r["sid"]] == p] for p in PARTS}
    fams = sorted({f for r in rows for f in (r.get("overlap") or {})})
    out = {"parts": {}, "all": {}}
    for p, sub in list(by.items()) + [("all", rows)]:
        ans = [r for r in sub if r.get("answer_subset")]
        fp = [r.get("q_in_edit_first_pos") for r in sub]
        s = {"use": USE.get(p, "the whole pool"),
             "n_items": len({r["uid"] for r in sub}),
             "items_with_2_rows": sum(v == 2 for v in _per_item(sub).values()),
             "rows": _count(sub, lambda r: True),
             "answer_subset": {"rows": _count(ans, lambda r: True), "y0_valid": _count(ans, lambda r: bool(r.get("y0_valid"))),
                               "with_teacher_forced_key": _count(ans, lambda r: r.get("row_tf") is not None)},
             "q_in_edit": {"rows": _count(sub, lambda r: bool(r.get("q_in_edit"))),
                           "what": "the row's question string (stripped, lower case) is the question or the text rephrase of an edit "
                                   "of the backbone's long stream (pool field q_in_edit)",
                           "first_edit_below_T": {str(T): sum(x is not None and int(x) < T for x in fp) for T in PREFIXES}},
             "flags": {f: _count(sub, lambda r, f=f: bool(r.get(f))) for f in FLAGS},
             "flags_by_family": {fam: {f: sum(bool((r.get("overlap") or {}).get(fam, {}).get(f)) for r in sub)
                                       for f in FLAGS if any(f in (r.get("overlap") or {}).get(fam, {}) for r in rows)} for fam in fams}}
        (out["parts"] if p != "all" else out)[p] = s
    q = {p: {C.norm_q(r["prompt"]) for r in by[p]} for p in PARTS}
    im = {p: {_image_id(r, idx.get(r["sid"])) for r in by[p] if r.get("image") is not None} for p in PARTS}
    share = {}
    for a in PARTS:
        for b in PARTS:
            if a != b:
                share[f"{a} rows whose question string occurs in {b}"] = _count(by[a], lambda r, b=b: C.norm_q(r["prompt"]) in q[b])
                share[f"{a} rows whose image content occurs in {b}"] = sum(
                    r.get("image") is not None and _image_id(r, idx.get(r["sid"])) in im[b] for r in by[a])
    out["shared_between_parts"] = dict(share, note="items are disjoint between parts; a question STRING can occur in two parts (the "
                                       "benchmark asks the same image-locality question about several images) and so can an image")
    return out


def _per_item(rows) -> dict:
    c = {}
    for r in rows:
        c[r["uid"]] = c.get(r["uid"], 0) + 1
    return c


def build(backbone: str, check_same_pool: bool = True) -> dict:
    """The parts object of one backbone (what `write` serialises).  check_same_pool: every backbone's pool must hold the same rows."""
    rows = C.jsonl_read(pool_path(backbone))
    sids = [r["sid"] for r in rows]
    if len(set(sids)) != len(sids):
        raise RuntimeError(f"{backbone}: duplicate sids in the benign pool")
    ident = pool_identity(rows)
    same = {}
    if check_same_pool:
        for other in pool_backbones():
            same[other] = pool_identity(C.jsonl_read(pool_path(other)))
        bad = [b for b, h in same.items() if h != ident]
        if bad:
            raise RuntimeError(f"the benign pool of {backbone} differs from that of {bad}: the item split would not be the same")
    part_of_item = item_split(r["uid"] for r in rows)
    part_of_sid = {r["sid"]: part_of_item[r["uid"]] for r in rows}
    index_rows, index_md5 = None, None
    ip = os.path.join(C.CACHE, backbone, "index.jsonl")
    if os.path.exists(ip):                                             # image identity by content (md5 stored in the key-cache index)
        index_rows = [e for e in C.jsonl_read(ip) if e.get("kind") == "benign"]
        if [e["sid"] for e in index_rows] != sids:
            raise RuntimeError(f"{backbone}: the benign rows of the key-cache index are not the rows of the pool file")
        index_md5 = C.md5_file(ip)
    n = len(part_of_item)
    return {"what": "parts of the label-free benign pool: sid -> BEN-W | BEN-S | BEN-T, split by source item",
            "protocol": "10_final/PROTOCOL.md, section 6, amendment 'Review round 1', item R2",
            "backbone": backbone, "seed": SEED, "percent_of_items": dict(PERCENT), "parts_order": list(PARTS),
            "rule": "items = sorted distinct uids; perm = numpy.random.default_rng(seed).permutation(n); first round(0.20 n) items of perm "
                    "= BEN-W, next round(0.45 n) = BEN-S, rest = BEN-T; a row takes the part of its item (code: feat2/white/benparts.py)",
            "source": {"rows": os.path.relpath(pool_path(backbone), C.FINAL), "md5": C.md5_file(pool_path(backbone)), "n_rows": len(rows),
                       "n_items": n, "pool_identity_sha1": ident, "identity_fields": list(IDENTITY_FIELDS), "key_cache_index_md5": index_md5},
            "same_pool": {"checked": check_same_pool, "backbones": sorted(same), "identical": bool(same) and all(h == ident for h in same.values())},
            "n_items": {p: sum(v == p for v in part_of_item.values()) for p in PARTS},
            "n_rows": {p: sum(v == p for v in part_of_sid.values()) for p in PARTS},
            "items": part_of_item, "parts": part_of_sid,
            "summary": summarise(rows, part_of_sid, index_rows)}


def render(backbone: str) -> str:
    """The bytes of <backbone>.json (deterministic: sorted keys, no time stamp)."""
    return C.jdumps(build(backbone))


def write(backbone: str, directory: str | None = None) -> str:
    p = parts_path(backbone, directory)
    C.write_atomic(p, render(backbone))
    return p


class BenignParts:
    """The parts of one backbone's benign pool (read from <backbone>.json).

      backbone, path, seed, summary (dict), n_items / n_rows ({part: count})
      part(sid) -> 'BEN-W' | 'BEN-S' | 'BEN-T'           part_of_item(uid) -> part
      sids(part) -> [sid, ...] in pool order              (part: one name or a tuple of names)
      rows(part, kc) -> positions in the index of the backbone's key cache (rescope.KeyCache; = fx sequences), pool order
      mask(sids) -> {part: [bool, ...]} for a list of sids
    """

    def __init__(self, obj: dict, path: str):
        self.obj, self.path = obj, path
        self.backbone, self.seed, self.summary = obj["backbone"], obj["seed"], obj["summary"]
        self._part, self._item = obj["parts"], obj["items"]
        self.n_items, self.n_rows = obj["n_items"], obj["n_rows"]
        self._order = None

    def part(self, sid: str) -> str:
        return self._part[sid]

    def part_of_item(self, uid: str) -> str:
        return self._item[uid]

    def _pool_order(self) -> list:
        if self._order is None:                                         # 'benign:<ptype>:<n>': image rows first, then text rows, by n = file order
            self._order = sorted(self._part, key=lambda s: (s.split(":")[1] != "image", int(s.split(":")[2])))
        return self._order

    def sids(self, part) -> list:
        want = {part} if isinstance(part, str) else set(part)
        if not want <= set(PARTS):
            raise ValueError(f"part: one of {PARTS}")
        return [s for s in self._pool_order() if self._part[s] in want]

    def rows(self, part, kc) -> list:
        """Positions of the part's rows in the key-cache index (kc: rescope.KeyCache of the same backbone, or fx LogData.kc)."""
        if getattr(kc, "backbone", self.backbone) != self.backbone:
            raise ValueError(f"key cache of {kc.backbone}, parts of {self.backbone}")
        out = []
        for s in self.sids(part):
            j = kc.by_sid(s)
            if j is None or kc.index[j].get("kind") != "benign":
                raise LookupError(f"benign row {s} is not in the key cache")
            out.append(j)
        return out

    def mask(self, sids) -> dict:
        return {p: [self._part.get(s) == p for s in sids] for p in PARTS}


def load(backbone: str, verify: bool = True, directory: str | None = None) -> BenignParts:
    """Read 10_final/cache2/benign_parts/<backbone>.json.  verify: the pool file must still be the file the split was made from
    (md5), every row of the pool must have a part, and no item may lie in two parts; raises otherwise."""
    import json
    p = parts_path(backbone, directory)
    with open(p) as f:
        obj = json.load(f)
    P = BenignParts(obj, p)
    if verify:
        if C.md5_file(pool_path(backbone)) != obj["source"]["md5"]:
            raise RuntimeError(f"{pool_path(backbone)} is not the pool file the parts of {backbone} were made from (rebuild: python -m white.benparts build)")
        rows = C.jsonl_read(pool_path(backbone))
        seen = {}
        for r in rows:
            if r["sid"] not in obj["parts"] or obj["parts"][r["sid"]] != obj["items"][r["uid"]]:
                raise RuntimeError(f"{backbone}: row {r['sid']} has no part or not the part of its item")
            if seen.setdefault(r["uid"], obj["parts"][r["sid"]]) != obj["parts"][r["sid"]]:
                raise RuntimeError(f"{backbone}: item {r['uid']} lies in two parts")
        if len(rows) != len(obj["parts"]):
            raise RuntimeError(f"{backbone}: {len(obj['parts'])} sids in the parts file, {len(rows)} rows in the pool")
    return P


def table(objs: dict) -> str:
    """Plain-text table of the parts of several backbones ({backbone: parts object})."""
    lines = []
    for bb, o in objs.items():
        lines.append(f"{bb}: {o['source']['n_rows']} rows of {o['source']['n_items']} source items; pool md5 {o['source']['md5']}; "
                     f"same pool as {', '.join(b for b in o['same_pool']['backbones'] if b != bb) or '-'}: {o['same_pool']['identical']}")
        head = ("part", "items", "rows", "image", "text", "answer subset", "valid y0", "tf key", "q_in_edit", "of them image", "edit < 100", "< 500",
                "q_in_probe", "QK probe query")
        rows = [head]
        for p in PARTS + ("all",):
            s = o["summary"]["parts"][p] if p != "all" else o["summary"]["all"]
            qk = s["flags_by_family"].get("QK", {}).get("query_in_probe")
            rows.append((p, str(s["n_items"]), str(s["rows"]["all"]), str(s["rows"]["image"]), str(s["rows"]["text"]), str(s["answer_subset"]["rows"]["all"]),
                         str(s["answer_subset"]["y0_valid"]["all"]), str(s["answer_subset"]["with_teacher_forced_key"]["all"]), str(s["q_in_edit"]["rows"]["all"]),
                         str(s["q_in_edit"]["rows"]["image"]), str(s["q_in_edit"]["first_edit_below_T"]["100"]), str(s["q_in_edit"]["first_edit_below_T"]["500"]),
                         str(s["flags"]["q_in_probe"]["all"]), "-" if qk is None else str(qk)))
        w = [max(len(r[i]) for r in rows) for i in range(len(head))]
        lines += ["  " + "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
        sh = o["summary"]["shared_between_parts"]
        lines.append("  BEN-T rows whose question string also occurs in BEN-W / BEN-S: "
                     f"{sh['BEN-T rows whose question string occurs in BEN-W']['all']} / {sh['BEN-T rows whose question string occurs in BEN-S']['all']} "
                     f"(image rows only: {sh['BEN-T rows whose question string occurs in BEN-W']['text'] + sh['BEN-T rows whose question string occurs in BEN-S']['text'] == 0}); "
                     f"whose image content occurs in BEN-W / BEN-S: {sh['BEN-T rows whose image content occurs in BEN-W']} / {sh['BEN-T rows whose image content occurs in BEN-S']}")
        lines.append("")
    return "\n".join(lines)


README = """10_final/cache2/benign_parts -- parts of the label-free benign pool (PROTOCOL.md section 6, amendment "Review round 1", item R2)
Written by `python -m white.benparts build` (code: 10_final/feat2/white/benparts.py; its module docstring is the reference).

  <backbone>.json   parts {sid: BEN-W | BEN-S | BEN-T}, items {uid: part}, summary; split by SOURCE ITEM (uid), seed 20261111:
                    BEN-W 20 % of the items (statistics of baselines, extra null-repair keys), BEN-S 45 % (complaint streams, second
                    calibration source), BEN-T 35 % (fresh test rows, used for nothing else).  The item split is the same for every
                    backbone (the pool holds the same rows for all of them; checked at build time).
  Reader: white.benparts.load(backbone) -> .part(sid), .sids(part), .rows(part, kc), .summary     (see feat2/white/USAGE.txt)
  The files carry no time stamp: a re-run reproduces them byte for byte (feat2/white/tests/t_parts.py).
  blip2.json is the same item split for the scout's backbone (same pool); the amendment names no use for it.

columns: answer subset = rows with a cached decoder-v2 pre-edit answer; valid y0 = of them reproduced by the pre-edit model (the rows on
which D1 is defined); tf key = rows with a teacher-forced key; q_in_edit = the row's question string is the question or text rephrase of
an edit of the backbone's long stream (near-scope benign rows; 'edit < T' = of one of the first T edits); q_in_probe = the question string
is also asked by a probe row; QK probe query = the whole query is a probe query of the MMKE-Bench family (Qwen only).

"""


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="parts of the benign pool (R2)")
    ap.add_argument("cmd", choices=("build", "check", "show"))
    ap.add_argument("backbones", nargs="*")
    ap.add_argument("--dir", default=None, help="write / read another directory (tests)")
    a = ap.parse_args(argv)
    bbs = a.backbones or pool_backbones()
    if a.cmd == "build":
        objs = {}
        for bb in bbs:
            p = write(bb, a.dir)
            with open(p) as f:
                objs[bb] = json.load(f)
            print(f"wrote {p} (md5 {C.md5_file(p)})")
        txt = README + table(objs)
        C.write_atomic(os.path.join(C.PARTS_DIR if a.dir is None else a.dir, "README.txt"), txt)
        print(table(objs))
    elif a.cmd == "check":                                             # byte-for-byte reproduction of the stored files + reader checks
        ok = True
        for bb in bbs:
            with open(parts_path(bb, a.dir), "rb") as f:
                same = f.read() == render(bb).encode("utf-8")
            load(bb, True, a.dir)
            ok &= same
            print(f"{bb}: stored file {'reproduced byte for byte' if same else 'DIFFERS from a re-run'}; reader checks passed")
        raise SystemExit(0 if ok else 1)
    else:
        objs = {}
        for bb in bbs:
            with open(parts_path(bb, a.dir)) as f:
                objs[bb] = json.load(f)
        print(table(objs))


if __name__ == "__main__":
    main()
