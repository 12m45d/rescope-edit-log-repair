"""Model-free selection (CPU, deterministic) for one backbone: the benign pool, the hard negatives (existing rows
annotated + new rows) and the manifest of every sequence of the backbone's key cache.

    python fc_select.py <qwen25vl | llava>
    FC_SELECT_DIR=<dir> python fc_select.py <backbone>      the same selection into another directory: fc_check_select.sh
                                                            compares it with the selection in use (it must be byte-identical)

Writes work/<backbone>/select/: benign.jsonl, hn.jsonl, hn_edits.json, manifest.jsonl, select_summary.json.
No model is loaded.  Every draw is seeded (fc_common.SEED, sha1 of labels / edit uids).

BENIGN POOL (task 2).  Label-free queries of the same kind as the probe rows: the image-locality row (m_loc image +
m_loc_q) and the text-locality row (loc) of VLKEB eval items that are neither one of the 500 probe items nor an
edit of the backbone's long stream.  The probe items are a uniform sample of the eval split, so the remaining eval
ITEMS are exchangeable with them (train items are counted in the summary but not used).  The ROWS are deduplicated
and exclude every query that a probe row also asks (the benchmark repeats image-locality rows across items), so the
image rows are distinct queries and not a like-for-like sample of the probe image rows.  Items are visited in one
seeded random order; an item contributes its image row if that query (image, question) is not a probe query and was
not taken before, and its text row likewise, until 1500 image rows and 1500 text rows exist.  A seeded subset of
300 image + 300 text rows gets a decoder-v2 pre-edit answer (fc_capture.py).

HARD NEGATIVES (task 3).  Rules of 08_rescope/p11_retrofit/code/hn_build.py, WITHOUT its category restriction:
  source pool   VLKEB eval items that are not probe items and share no (image, src, alt), no Freebase MID (image
                directory of image / m_loc against the edits' image / image_rephrase) and no entity name
                (pred_text / alt, normalised) with ANY edit of the project's VLKEB edit streams (now including
                the long streams of both backbones);
  HN-I          the edit's image + the image-locality question (m_loc_q) of a source item drawn uniformly
                (seeded permutation of the source items per edit uid; a question equal to the edit's own question
                or rephrase, or already used for this edit, is skipped); same_category is RECORDED, not used;
  HN-Q          the edit's question (src) + the entity image (`image`) of a source item drawn uniformly (other
                MID, other names; an image already used for this edit is skipped);
  HN-X          the image of edit a + the question (src) of edit b, a != b: within a group of 100 edits one seeded
                cyclic order, so that every edit gives its image once and its question once; a pair is admissible
                if b's question differs from a's question, from a's rephrase and from the questions of a's HN-I
                rows, and the two edits share no MID and no name.
  (a) first 100 edits of the stream: HN-I k = 2..5 and HN-Q k = 2..5 (k = 0, 1 are the existing rows);
  (b) 100 edits sampled with a fixed seed from stream positions >= 100: HN-I k = 0, 1 and HN-Q k = 0, 1;
  (c) HN-X: 100 pairs among the first 100 edits, 100 pairs among the sampled later edits.
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_common as C  # noqa: E402
from fc_cats import cats  # noqa: E402

import numpy as np  # noqa: E402

BB = sys.argv[1]
CFG = C.BACKBONES[BB]
VF = CFG["vlkeb"]                               # the VLKEB family of this backbone (Q | L)
SD = os.environ.get("FC_SELECT_DIR") or os.path.join(C.WORK, BB, "select")     # FC_SELECT_DIR: re-run into another directory (determinism check)
os.makedirs(SD, exist_ok=True)
TB = C.tb()
N_BENIGN, N_SUBSET = 1500, 300                  # per row type (image / text)
summ = {"backbone": BB, "seed": C.SEED, "vlkeb_family": VF}

fams = {f: C.family(f) for f in CFG["families"]}
F = fams[VF]
items, probes = F["items"], F["probes"]
T = len(items)
ev = TB.load_vlkeb("eval")
tr = TB.load_vlkeb("train")
ev_by_uid = {d["uid"]: d for d in ev}
C.say(f"{BB}: family {VF} T = {T}, probes {len(probes)}; families {list(fams)}")


# ----------------------------------------------------------------------------- image identities (md5 of the files)
def all_images():
    out = []
    for f in fams.values():
        for d in f["items"]:
            out += [d["image"], d["image_rephrase"]]
        out += [p["m_loc"] for p in f["probes"]]
    out += [d["m_loc"] for d in ev] + [d["image"] for d in ev]
    return list(dict.fromkeys(out))


imgs = all_images()
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
C.image_md5(imgs[0])
todo = [i for i in imgs if i not in C._MD5]
if todo:
    def _one(p):
        import hashlib
        h = hashlib.md5()
        with open(C.image_path(p), "rb") as fh_:
            for chunk in iter(lambda: fh_.read(1 << 20), b""):
                h.update(chunk)
        return p, h.hexdigest()
    with ThreadPoolExecutor(4) as ex:
        for p, m in ex.map(_one, todo):
            C._MD5[p] = m
    C.image_md5_flush()
C.say(f"  image md5: {len(imgs)} files ({len(todo)} new), {len({C.image_md5(i) for i in imgs})} distinct contents")
summ["images"] = {"n_paths": len(imgs), "n_distinct_md5": len({C.image_md5(i) for i in imgs})}


# ----------------------------------------------------------------------------- what occurs among edits / probes
class Occ:
    """Question strings and images of the edits and of the probe rows of one family."""

    def __init__(self, f):
        self.fam = f["family"]
        self.vlkeb = f["dataset"] == "VLKEB"
        self.q_edit, self.qx_edit, self.img_edit, self.mid_edit = {}, {}, {}, {}
        for i, d in enumerate(f["items"]):
            for q in (d["src"], d["rephrase"]):
                self.q_edit.setdefault(C.norm_q(q), i)
                self.qx_edit.setdefault(q, i)
            for im in (d["image"], d["image_rephrase"]):
                self.img_edit.setdefault(C.image_md5(im), i)
                if self.vlkeb:
                    self.mid_edit.setdefault(C.mid(im), i)
        self.q_probe, self.qx_probe, self.img_probe, self.mid_probe, self.query_probe = set(), set(), set(), set(), set()
        for r in f["rows"]:
            self.q_probe.add(C.norm_q(r["prompt"]))
            self.qx_probe.add(r["prompt"])
            self.query_probe.add(C.query_key(r["ptype"], r["prompt"], r["image"]))
            if r["image"] is not None:
                self.img_probe.add(C.image_md5(r["image"]))
                if self.vlkeb:
                    self.mid_probe.add(C.mid(r["image"]))

    def flags(self, ptype, prompt, image):
        nq, m = C.norm_q(prompt), C.image_md5(image)
        o = {"q_in_edit": nq in self.q_edit, "q_in_edit_first_pos": self.q_edit.get(nq),
             "q_in_edit_exact": prompt in self.qx_edit,
             "img_in_edit": m in self.img_edit if m else False, "img_in_edit_first_pos": self.img_edit.get(m) if m else None,
             "q_in_probe": nq in self.q_probe, "q_in_probe_exact": prompt in self.qx_probe,
             "img_in_probe": m in self.img_probe if m else False,
             "query_in_probe": C.query_key(ptype, prompt, image) in self.query_probe}
        if self.vlkeb and image is not None and not os.path.isabs(image):
            o["img_entity_in_edit"] = C.mid(image) in self.mid_edit
            o["img_entity_in_edit_first_pos"] = self.mid_edit.get(C.mid(image))
            o["img_entity_in_probe"] = C.mid(image) in self.mid_probe
        return o


occ = {f: Occ(fams[f]) for f in fams}

# ============================================================================= benign pool
probe_uids = {p["uid"] for p in probes}
edit_uids = {d["uid"] for d in items}
edit_keys = {(d["image"], d["src"], d["alt"]) for d in items}
probe_img_q = {C.query_key("image", p["m_loc_q"], p["m_loc"]) for p in probes}
probe_txt_q = {C.query_key("text", p["loc"], None) for p in probes}
elig = [d for d in ev if d["uid"] not in probe_uids and d["uid"] not in edit_uids
        and (d["image"], d["src"], d["alt"]) not in edit_keys and d.get("m_loc") and d.get("loc")]


def distinct_rows(pool):
    qi, qt = set(), set()
    for d in pool:
        k = C.query_key("image", d["m_loc_q"], d["m_loc"])
        if k not in probe_img_q:
            qi.add(k)
        k = C.query_key("text", d["loc"], None)
        if k not in probe_txt_q:
            qt.add(k)
    return len(qi), len(qt)


tr_elig = [d for d in tr if d["uid"] not in edit_uids and d.get("m_loc") and d.get("loc")]
ni, nt = distinct_rows(elig)
tni, tnt = distinct_rows(tr_elig)
order = np.random.default_rng(C.useed("benign_order")).permutation(len(elig))
seen_i, seen_t, ben_i, ben_t = set(probe_img_q), set(probe_txt_q), [], []
n_img_probe_dup = n_img_dup = 0
for rank, j in enumerate(order):
    d = elig[int(j)]
    ki = C.query_key("image", d["m_loc_q"], d["m_loc"])
    if ki in probe_img_q:
        n_img_probe_dup += 1
    elif ki in seen_i:
        n_img_dup += 1
    elif len(ben_i) < N_BENIGN:
        seen_i.add(ki)
        ben_i.append({"ptype": "image", "uid": d["uid"], "item_rank": rank, "prompt": d["m_loc_q"], "image": d["m_loc"], "fmt": "vqa",
                      "ref_answer": d["m_loc_a"], "ref_field": "m_loc_a"})
    kt = C.query_key("text", d["loc"], None)
    if kt not in seen_t and len(ben_t) < N_BENIGN:
        seen_t.add(kt)
        ben_t.append({"ptype": "text", "uid": d["uid"], "item_rank": rank, "prompt": d["loc"], "image": None,
                      "fmt": TB.TEXT_FMT[BB], "ref_answer": d["loc_ans"], "ref_field": "loc_ans"})
benign = []
for pt, rows_ in (("image", ben_i), ("text", ben_t)):
    sub = set(np.random.default_rng(C.useed("benign_subset", pt)).choice(len(rows_), min(N_SUBSET, len(rows_)), replace=False).tolist())
    for j, r in enumerate(rows_):
        r = dict(r, benign_idx=j, sid=f"benign:{pt}:{j}", answer_subset=j in sub)
        r.update(occ[VF].flags(pt, r["prompt"], r["image"]))
        r["overlap"] = {f: occ[f].flags(pt, r["prompt"], r["image"]) for f in fams}
        benign.append(r)
C.jsonl_write(benign, os.path.join(SD, "benign.jsonl"))


def cnt(rows_, key):
    return {pt: sum(bool(r[key]) for r in rows_ if r["ptype"] == pt) for pt in ("image", "text")}


summ["benign"] = {
    "source": "VLKEB eval items that are neither one of the 500 probe items nor an edit of the long stream: image-locality row "
              "(m_loc, m_loc_q) and text-locality row (loc); distinct queries, none equal to a probe query",
    "exists": {"eval_items_eligible": len(elig), "eval_distinct_image_rows_not_probe_queries": ni, "eval_distinct_text_rows_not_probe_queries": nt,
               "eval_items_whose_image_row_is_a_probe_query": n_img_probe_dup,
               "eval_distinct_image_rows_that_are_probe_queries(excluded)": len({C.query_key("image", d["m_loc_q"], d["m_loc"]) for d in elig}
                                                                                & probe_img_q),
               "eval_items_whose_image_row_repeats_an_earlier_benign_row": n_img_dup,
               "train_items_not_edited(not used)": len(tr_elig), "train_distinct_image_rows_not_probe_queries(not used)": tni,
               "train_distinct_text_rows_not_probe_queries(not used)": tnt},
    "n_rows": {"image": len(ben_i), "text": len(ben_t)}, "n_source_items": len({r["uid"] for r in benign}),
    "n_answer_subset": cnt(benign, "answer_subset"),
    "flags_vs_family_" + VF: {k: cnt(benign, k) for k in ("q_in_edit", "q_in_edit_exact", "img_in_edit", "q_in_probe", "q_in_probe_exact",
                                                           "img_in_probe", "query_in_probe")},
    "flags_vs_other_families": {f: {k: {pt: sum(bool(r["overlap"][f][k]) for r in benign if r["ptype"] == pt) for pt in ("image", "text")}
                                    for k in ("q_in_edit", "img_in_edit", "q_in_probe", "img_in_probe", "query_in_probe")}
                                for f in fams if f != VF},
    "image_rows_entity_flags": {k: sum(bool(r.get(k)) for r in benign if r["ptype"] == "image")
                                for k in ("img_entity_in_edit", "img_entity_in_probe")}}
C.say(f"  benign: {summ['benign']['n_rows']} rows from {summ['benign']['n_source_items']} items; exists {summ['benign']['exists']}")

# ============================================================================= hard negatives
# --- every VLKEB edit stream of the project (exclusion set of the source pool)
edit_all = {}
streams_used = []
for name, cfg in sorted(TB.LOGS.items()):
    p = cfg["stream"]
    if not os.path.exists(p):
        continue
    recs = TB.jsonl_read(p)
    if recs and recs[0].get("dataset") == "MMKE-Bench":
        continue
    its, _ = TB.load_stream(p)
    streams_used.append((name, len(its)))
    for d in its:
        edit_all.setdefault(d["uid"], d)
for fam_, bb_ in (("Q", "qwen25vl"), ("L", "llava")):          # the long streams of BOTH backbones, whatever the registry holds
    for d in (fams[fam_]["items"] if fam_ in fams else C.family(fam_)["items"]):
        edit_all.setdefault(d["uid"], d)
all_probe_uids = probe_uids | {p["uid"] for p in C.family("Q" if VF == "L" else "L")["probes"]}
ek_all = {(d["image"], d["src"], d["alt"]) for d in edit_all.values()}
em_all, en_all = set(), set()
for d in edit_all.values():
    em_all |= {C.mid(d["image"]), C.mid(d["image_rephrase"])}
    en_all |= {TB.norm_answer(d["alt"]), TB.norm_answer(d["pred_text"])}
pool, excl = [], collections.Counter()
for d in ev:
    if d["uid"] in all_probe_uids:
        excl["probe"] += 1
    elif (d["image"], d["src"], d["alt"]) in ek_all:
        excl["edit_key"] += 1
    elif C.mid(d["image"]) in em_all or C.mid(d["m_loc"]) in em_all:
        excl["edit_mid"] += 1
    elif TB.norm_answer(d["pred_text"]) in en_all or TB.norm_answer(d["alt"]) in en_all:
        excl["edit_name"] += 1
    else:
        pool.append(d)
pool_uids = {d["uid"] for d in pool}
C.say(f"  HN source pool: {len(pool)} of {len(ev)} eval items (excluded {dict(excl)}; {len(edit_all)} edit items of {len(streams_used)} registered VLKEB logs)")

# --- relation of ANY eval item to the edits of THIS backbone's long stream (first stream position)
key_pos, mid_pos, name_pos = {}, {}, {}
for i, d in enumerate(items):
    key_pos.setdefault((d["image"], d["src"], d["alt"]), i)
    for m in (C.mid(d["image"]), C.mid(d["image_rephrase"])):
        mid_pos.setdefault(m, i)
    for n in (TB.norm_answer(d["alt"]), TB.norm_answer(d["pred_text"])):
        name_pos.setdefault(n, i)


def first_related_pos(p):
    """Smallest stream position of an edit that the eval item p is related to by hn_build's rule (None: unrelated)."""
    c = [key_pos.get((p["image"], p["src"], p["alt"])), mid_pos.get(C.mid(p["image"])), mid_pos.get(C.mid(p["m_loc"])),
         name_pos.get(TB.norm_answer(p["pred_text"])), name_pos.get(TB.norm_answer(p["alt"]))]
    c = [x for x in c if x is not None]
    return min(c) if c else None


pos_of = {d["uid"]: i for i, d in enumerate(items)}

# --- existing rows (first 100 edits)
old = C.jsonl_read(os.path.join(C.HN_OLD, BB, "rows.jsonl"))
old_by_edit = collections.defaultdict(lambda: {"HN-I": [], "HN-Q": []})
bad_cats = 0
for r in old:
    assert pos_of[r["edit_uid"]] < 100, r["hn_id"]
    old_by_edit[r["edit_uid"]][r["hn_type"]].append(r)
    d = items[pos_of[r["edit_uid"]]]
    ec = sorted(set(cats(d["src"])) | set(cats(d["rephrase"])))
    qc = [c for c in cats(r["prompt"])] if r["hn_type"] == "HN-I" else cats(d["src"])
    bad_cats += int(ec != r["edit_cats"]) + int(qc != r["q_cats"])
assert all(len(v["HN-I"]) == 2 and len(v["HN-Q"]) == 2 for v in old_by_edit.values()) and len(old_by_edit) == 100
assert bad_cats == 0, f"the copied category lexicon does not reproduce the stored categories ({bad_cats})"
later = sorted(np.random.default_rng(C.useed("hn_later_edits", VF)).choice(np.arange(100, T), 100, replace=False).tolist())
groups = {"first100": list(range(100)), "later100": later}
C.jdump({"family": VF, "T": T, "first100": groups["first100"], "later100": later,
         "later100_uids": [items[i]["uid"] for i in later],
         "later100_per_prefix": {str(t): sum(i < t for i in later) for t in (200, 500, 1000, 2000) if t <= T}},
        os.path.join(SD, "hn_edits.json"))

from fc_cats import useed_p11  # noqa: E402

hn = []
for r in old:                                   # existing rows, annotated
    p = ev_by_uid[r["src_uid"]]
    hn.append(dict(r, rule="p11_category_restricted", group="first100", edit_pos=pos_of[r["edit_uid"]], new=False,
                   src_in_final_pool=r["src_uid"] in pool_uids, src_related_edit_first_pos=first_related_pos(p)))
n_naive_same = []
for g, ids in groups.items():
    k0, n_new = (2, 4) if g == "first100" else (0, 2)
    for i in ids:
        d = items[i]
        ec = sorted(set(cats(d["src"])) | set(cats(d["rephrase"])))
        base = {"edit_uid": d["uid"], "fmt": "vqa", "edit_cats": ec, "edit_src": d["src"], "edit_rephrase": d["rephrase"],
                "edit_alt": d["alt"], "edit_pred": d["pred_text"], "rule": "final_unrestricted", "group": g, "edit_pos": i, "new": True,
                "src_in_final_pool": True, "src_related_edit_first_pos": None}
        # HN-I: the edit's image + an unrelated question (uniform over the source items)
        own = (C.norm_q(d["src"]), C.norm_q(d["rephrase"]))
        cand = [p for p in pool if C.norm_q(p["m_loc_q"]) not in own]
        n_naive_same.append(float(np.mean([bool(set(cats(p["m_loc_q"])) & set(ec)) for p in cand])))
        rng = np.random.default_rng(useed_p11(d["uid"], "HN-I|10_final"))
        seen = {C.norm_q(r["prompt"]) for r in old_by_edit[d["uid"]]["HN-I"]} if g == "first100" else set()
        k = k0
        for j in rng.permutation(len(cand)):
            p = cand[int(j)]
            qn = C.norm_q(p["m_loc_q"])
            if qn in seen:
                continue
            seen.add(qn)
            qc = cats(p["m_loc_q"])
            hn.append(dict(base, hn_id=f"HN-I:{d['uid']}:{k}", hn_type="HN-I", k=k, image=d["image"], prompt=p["m_loc_q"],
                           src_uid=p["uid"], src_field="m_loc_q", q_cats=qc, same_category=bool(set(qc) & set(ec))))
            k += 1
            if k == k0 + n_new:
                break
        assert k == k0 + n_new, d["uid"]
        # HN-Q: the edit's question + an unrelated image (uniform over the source items)
        dm = {C.mid(d["image"]), C.mid(d["image_rephrase"])}
        dn = {TB.norm_answer(d["alt"]), TB.norm_answer(d["pred_text"])}
        cand = [p for p in pool if C.mid(p["image"]) not in dm and TB.norm_answer(p["pred_text"]) not in dn
                and TB.norm_answer(p["alt"]) not in dn]
        rng = np.random.default_rng(useed_p11(d["uid"], "HN-Q|10_final"))
        seen = {r["image"] for r in old_by_edit[d["uid"]]["HN-Q"]} if g == "first100" else set()
        k = k0
        for j in rng.permutation(len(cand)):
            p = cand[int(j)]
            if p["image"] in seen:
                continue
            seen.add(p["image"])
            hn.append(dict(base, hn_id=f"HN-Q:{d['uid']}:{k}", hn_type="HN-Q", k=k, image=p["image"], prompt=d["src"],
                           src_uid=p["uid"], src_field="image", q_cats=cats(d["src"]),
                           same_category=bool(set(cats(p["src"])) & set(cats(d["src"]))),
                           src_item_question=p["src"], src_item_pred=p["pred_text"]))
            k += 1
            if k == k0 + n_new:
                break
        assert k == k0 + n_new, d["uid"]


hn_i_q = collections.defaultdict(set)          # questions already paired with an edit's image (HN-I rows, existing and new)
for r in hn:
    if r["hn_type"] == "HN-I":
        hn_i_q[r["edit_uid"]].add(C.norm_q(r["prompt"]))


def admissible(a, b):
    """HN-X pair (image of a, question of b): another question (also none of a's HN-I questions), another entity."""
    if C.norm_q(b["src"]) in (C.norm_q(a["src"]), C.norm_q(a["rephrase"])) or C.norm_q(b["src"]) in hn_i_q[a["uid"]]:
        return False
    if {C.mid(a["image"]), C.mid(a["image_rephrase"])} & {C.mid(b["image"]), C.mid(b["image_rephrase"])}:
        return False
    return not ({TB.norm_answer(a["alt"]), TB.norm_answer(a["pred_text"])} & {TB.norm_answer(b["alt"]), TB.norm_answer(b["pred_text"])})


x_attempts = {}
for g, ids in groups.items():
    for att in range(10000):
        cyc = [ids[int(j)] for j in np.random.default_rng(C.useed("hn_x", VF, g, att)).permutation(len(ids))]
        pairs = [(cyc[j], cyc[(j + 1) % len(cyc)]) for j in range(len(cyc))]
        if all(admissible(items[a], items[b]) for a, b in pairs):
            break
    else:
        raise RuntimeError("no admissible cyclic order")
    x_attempts[g] = att + 1
    for a, b in sorted(pairs):
        da, db = items[a], items[b]
        ec = sorted(set(cats(da["src"])) | set(cats(da["rephrase"])))
        qc = cats(db["src"])
        hn.append({"hn_id": f"HN-X:{da['uid']}:{db['uid']}", "hn_type": "HN-X", "edit_uid": da["uid"], "k": 0, "image": da["image"],
                   "prompt": db["src"], "fmt": "vqa", "src_uid": db["uid"], "src_field": "edit_src", "q_cats": qc, "edit_cats": ec,
                   "same_category": bool(set(qc) & set(ec)), "edit_src": da["src"], "edit_rephrase": da["rephrase"],
                   "edit_alt": da["alt"], "edit_pred": da["pred_text"], "edit_b_uid": db["uid"], "edit_b_pos": b,
                   "edit_b_src": db["src"], "edit_b_alt": db["alt"], "edit_b_pred": db["pred_text"],
                   "rule": "final_unrestricted", "group": g, "edit_pos": a, "new": True,
                   "src_in_final_pool": None, "src_related_edit_first_pos": None})
ids_ = [r["hn_id"] for r in hn]
assert len(set(ids_)) == len(ids_)
for r in hn:
    r.update({"q_in_edit": occ[VF].flags("image", r["prompt"], r["image"])["q_in_edit"],
              "q_in_edit_first_pos": occ[VF].flags("image", r["prompt"], r["image"])["q_in_edit_first_pos"],
              "q_in_probe": occ[VF].flags("image", r["prompt"], r["image"])["q_in_probe"],
              "img_in_probe": occ[VF].flags("image", r["prompt"], r["image"])["img_in_probe"],
              "query_in_probe": occ[VF].flags("image", r["prompt"], r["image"])["query_in_probe"]})
C.jsonl_write(hn, os.path.join(SD, "hn.jsonl"))
per = collections.Counter((r["hn_type"], r["group"], r["new"]) for r in hn)
summ["hard_negatives"] = {
    "source_pool": {"n_items": len(pool), "excluded": dict(excl), "n_edit_items_excluded_against": len(edit_all),
                    "registered_vlkeb_logs": streams_used, "n_distinct_questions": len({C.norm_q(p["m_loc_q"]) for p in pool})},
    "n_rows": len(hn), "n_by_type_group_new": {f"{a}|{b}|{'new' if c else 'existing'}": n for (a, b, c), n in sorted(per.items())},
    "rows_per_edit_first100": {t: sorted(collections.Counter(collections.Counter(r["edit_uid"] for r in hn if r["hn_type"] == t and r["group"] == "first100").values()).items()) for t in ("HN-I", "HN-Q", "HN-X")},
    "rows_per_edit_later100": {t: sorted(collections.Counter(collections.Counter(r["edit_uid"] for r in hn if r["hn_type"] == t and r["group"] == "later100").values()).items()) for t in ("HN-I", "HN-Q", "HN-X")},
    "same_category": {f"{t}|{'new' if nw else 'existing'}": [sum(r["same_category"] for r in hn if r["hn_type"] == t and r["new"] == nw),
                                                              sum(1 for r in hn if r["hn_type"] == t and r["new"] == nw)]
                      for t in ("HN-I", "HN-Q", "HN-X") for nw in (False, True) if any(r["hn_type"] == t and r["new"] == nw for r in hn)},
    "naive_same_category_share_mean_HN-I_candidates": float(np.mean(n_naive_same)),
    "existing_rows_whose_source_item_is_related_to_a_long_stream_edit": {
        t: {"n": sum(1 for r in hn if not r["new"] and r["hn_type"] == t and r["src_related_edit_first_pos"] is not None),
            "first_pos_lt": {str(tt): sum(1 for r in hn if not r["new"] and r["hn_type"] == t and r["src_related_edit_first_pos"] is not None
                                          and r["src_related_edit_first_pos"] < tt) for tt in (100, 200, 500, 1000, 2000) if tt <= T}}
        for t in ("HN-I", "HN-Q")},
    "question_string_also_an_edit_question": {t: sum(bool(r["q_in_edit"]) for r in hn if r["hn_type"] == t) for t in ("HN-I", "HN-Q", "HN-X")},
    "hn_x_cyclic_order_attempts": x_attempts, "later100_positions": later,
    "later100_per_prefix": {str(t): sum(i < t for i in later) for t in (200, 500, 1000, 2000) if t <= T}}
C.say(f"  hard negatives: {summ['hard_negatives']['n_by_type_group_new']}; same category {summ['hard_negatives']['same_category']}")

# ============================================================================= manifest
man = []
for f, fd in fams.items():
    its = fd["items"]
    for metric in ("rel", "tgen", "igen"):
        for i, d in enumerate(its):
            man.append({"sid": f"{f}:edit:{metric}:{i}", "kind": metric, "family": f, "dataset": fd["dataset"], "lib_kind": "edit",
                        "lib_sid": f"edit:{metric}:{i}", "ptype": "image", "fmt": "vqa",
                        "prompt": d["rephrase"] if metric == "tgen" else d["src"],
                        "image": d["image_rephrase"] if metric == "igen" else d["image"],
                        "edit": i, "edit_uid": d["uid"], "metric": metric, "answer": d["alt"], "answer_from": "edit_target",
                        "decode": False})
    ann = {(a["probe_idx"], a["ptype"]): a for a in fd["ann"]}
    for r in fd["rows"]:
        a = ann[(r["probe_idx"], r["ptype"])]
        man.append({"sid": f"{f}:probe:{r['ptype']}:{r['probe_idx']}", "kind": "probe", "family": f, "dataset": fd["dataset"],
                    "lib_kind": "probe", "lib_sid": f"probe:{r['ptype']}:{r['probe_idx']}", "ptype": r["ptype"], "fmt": r["fmt"],
                    "prompt": r["prompt"], "image": r["image"], "probe_idx": r["probe_idx"], "uid": r["uid"], "group": r["group"],
                    "pool": a["pool"], "disjoint": a["disjoint"], "strict": a["strict"], "question_in_repair": a["question_in_repair"],
                    "twin_of": a["twin_of"], "probe_query_id": a["query_id"], "probe_n_same_query": a["n_same_query"],
                    "answer": r["y0_text"], "answer_from": "probe_records_v2", "y0_text": r["y0_text"], "y0_valid": bool(r["y0_valid"]),
                    "y0_invalid_reason": r["y0_invalid_reason"], "w0_reproduced": r["w0_reproduced"],
                    "pre_correct_long": bool(r["pre_correct_long"]), "decode": False})
for r in hn:
    man.append({"sid": r["hn_id"], "kind": C.HN_KIND[r["hn_type"]], "family": VF, "dataset": "VLKEB", "lib_kind": "hn", "lib_sid": r["hn_id"],
                "ptype": "image", "fmt": "vqa", "prompt": r["prompt"], "image": r["image"], "hn_id": r["hn_id"], "hn_type": r["hn_type"],
                "edit": r["edit_pos"], "edit_uid": r["edit_uid"], "edit_b": r.get("edit_b_pos"), "edit_b_uid": r.get("edit_b_uid"),
                "src_uid": r["src_uid"], "k": r["k"], "group": r["group"], "rule": r["rule"], "new": r["new"],
                "same_category": r["same_category"],
                "answer": None if r["new"] else r["y0_text"], "answer_from": "decoder_v2_here" if r["new"] else "p11_rows",
                "y0_text": None if r["new"] else r["y0_text"], "y0_valid": None if r["new"] else bool(r["y0_valid"]),
                "y0_invalid_reason": None if r["new"] else r["y0_invalid_reason"], "decode": bool(r["new"])})
for r in benign:
    man.append({"sid": r["sid"], "kind": "benign", "family": VF, "dataset": "VLKEB", "lib_kind": "benign", "lib_sid": r["sid"],
                "ptype": r["ptype"], "fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"], "benign_idx": r["benign_idx"],
                "uid": r["uid"], "pool": "benign", "answer_subset": r["answer_subset"], "answer": None,
                "answer_from": "decoder_v2_here" if r["answer_subset"] else None, "y0_text": None, "y0_valid": None,
                "y0_invalid_reason": None, "decode": bool(r["answer_subset"])})
qid = {}
for m in man:
    m["query_id"] = qid.setdefault(C.query_key(m["ptype"], m["prompt"], m["image"]), len(qid))
    m["image_md5"] = C.image_md5(m["image"])
cq = collections.Counter(m["query_id"] for m in man)
for m in man:
    m["n_same_query"] = cq[m["query_id"]]
assert len({m["sid"] for m in man}) == len(man)
C.jsonl_write(man, os.path.join(SD, "manifest.jsonl"))
C.image_md5_flush()
kinds = collections.Counter((m["family"], m["kind"]) for m in man)
summ["manifest"] = {"n_sequences": len(man), "n_distinct_queries": len(qid), "n_to_decode": sum(m["decode"] for m in man),
                    "by_family_kind": {f"{a}|{b}": n for (a, b), n in sorted(kinds.items())},
                    "sequences_in_duplicate_query_groups": {k: sum(1 for m in man if m["kind"] == k and cq[m["query_id"]] > 1) for k in C.KINDS}}
C.jdump(summ, os.path.join(SD, "select_summary.json"))
C.say(f"  manifest: {summ['manifest']}")
