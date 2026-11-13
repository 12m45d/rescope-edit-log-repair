"""Task 1: the attribute table of a backbone (CPU, deterministic, no model weights).

    python build_attributes.py <qwen25vl | llava> [--out DIR]

Writes <DATA>/<backbone>/attributes.jsonl and attributes_meta.json (DATA = 10_final/cache2/retr; --out: the same table into
another directory, used by the byte-for-byte reproducibility test).

THE TABLE.  One JSON object per line, keys in a fixed order.
  lines 0 .. n - 1     table = "sequence": line j describes the sequence at position j of the key cache index
                       10_final/cache/<backbone>/index.jsonl (edit prompts rel / tgen / igen, probe rows, hard negatives, benign rows)
  the following lines  table = "edit": one line per edit REQUEST of every edit stream of the backbone's cache (the long stream of
                       the backbone: family Q = TB-Q2000-ae-l2_3, L = TB-L1000-ae-l2_3; qwen25vl also holds the MMKE-Bench
                       stream QK = TB-Q500-mmke-l2_3), in stream order
  fields   table, row (cache position; null for edit lines), sid (edit lines: '<family>:request:<position>'), kind (rel | tgen | igen |
           probe | hn_i | hn_q | hn_x | benign | request), family, ptype (image | text), image (path as in the benchmark; null: text
           row), image_md5 (md5 of the image FILE; null: no image), question (the question string as the user asks it: no chat
           template, no answer), question_key (question.strip().lower(): the identity under which questions are embedded),
           edit, edit_uid (edit sequences / requests: the edit; hard negatives: the edit whose image (HN-I, HN-X) or question (HN-Q)
           the row uses), edit_b, edit_b_uid (HN-X: the edit whose question the row uses)

HOW THE TWO ATTRIBUTES ARE DERIVED AND CHECKED (every check is counted in attributes_meta.json; any failure raises)
  1. from the DATA LOADERS the caches were built with (testbed tb_common + scale registry: load_stream, select_probes, load_vlkeb;
     the row files of the hard negatives and of the benign pool):
       rel   (image, src)              tgen  (image, rephrase)             igen  (image_rephrase, src)        of the stream item
       probe image row (m_loc, m_loc_q), text row (loc) of the probe item
       hn    the row of hard_negatives/<backbone>/rows.jsonl, itself re-derived: HN-I = edit image + m_loc_q of its source item,
             HN-Q = edit question + entity image of its source item, HN-X = image of edit a + question of edit b
       benign  image row (m_loc, m_loc_q) / text row (loc) of the source eval item (benign/<backbone>/rows.jsonl)
       request (image, src) of the stream item == (image, subject) of the testbed's edit_batch.jsonl == the cached rel sequence
  2. compared with the fields `prompt`, `image` of the cache index (must be equal for every sequence);
  3. image_md5 is RECOMPUTED from the image file and compared with the index (must be equal);
  4. the prompt is RENDERED for every sequence with the backbone's tokenizer and prompt code (vendor_v7.vlm.VLM without weights):
     the rendered text must be the backbone's template around exactly the question string, and its token count (+ image slots)
     must equal the captured prompt length of the cache (len_prompt, lp_decode); for Qwen the number of image slots is recomputed
     from the image file's size.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402

FIELDS = ("table", "row", "sid", "kind", "family", "ptype", "image", "image_md5", "question", "question_key", "edit", "edit_uid", "edit_b", "edit_b_uid")


def testbed():
    """tb_common with the scale registry installed (the loaders the caches were built with); read-only."""
    for p in (C.EXP + "/09_testbed_scale/code", C.TESTBED + "/code", C.LIB):
        if p not in sys.path:
            sys.path.insert(0, p)
    import scale_registry  # noqa: F401
    import tb_common as TB
    return TB


def image_path(img: str) -> str:
    from vendor_v7.vlm import VLKEB_IMG_DIR
    return img if os.path.isabs(img) else os.path.join(VLKEB_IMG_DIR, img)


def row(table, j, sid, kind, family, ptype, image, md5, question, edit=None, edit_uid=None, edit_b=None, edit_b_uid=None) -> dict:
    v = (table, j, sid, kind, family, ptype, image, md5, question, C.norm_q(question), edit, edit_uid, edit_b, edit_b_uid)
    return dict(zip(FIELDS, v))


class Renderer:
    """The prompt of a query as the backbone receives it (tokenizer and prompt code only, no weights)."""

    def __init__(self, backbone: str):
        from vendor_v7 import vlm as V
        self.V, self.bb = V, backbone
        self.v = V.VLM(backbone, load_model=False)
        self._ntok = {}

    def template(self, fmt: str, with_image: bool):
        """(text before the question, text after it) of the rendered prompt text."""
        if self.bb == "qwen25vl":
            a, b = self.v._qwen_tmpl[with_image].split(self.V._Q_SENTINEL)
            return a, b
        if fmt == "vqa":
            a, b = self.V.QA_TEMPLATE.split("{}")
            return (self.v.llava_image_sep if (self.bb == "llava" and with_image) else "") + a, b
        return (self.v.llava_image_sep if (self.bb == "llava" and with_image) else ""), ""

    def n_image_tokens(self, image: str) -> int:
        if image not in self._ntok:
            self._ntok[image] = int(self.v.estimate_image_tokens(image_path(image)))
        return self._ntok[image]

    def check(self, e: dict, question: str, image) -> dict:
        """Render the prompt of index row e from (question, image) -> flags."""
        with_image = image is not None
        prefix, text, sep = self.v.text_parts(question, e["fmt"], with_image)
        a, b = self.template(e["fmt"], with_image)
        ok_text = text == a + question + b and text.startswith(a) and text.endswith(b) and text[len(a):len(text) - len(b)] == question
        n = len(prefix) + len(self.v._tok(text))
        n_img = None
        if with_image:
            n_img = self.n_image_tokens(image)
            if self.bb == "qwen25vl":
                n += n_img - 1                                           # the single <|image_pad|> of the template is expanded
        ok_len = n == e["len_prompt"] and e["lp_decode"] == n - 1
        ok_img = (not with_image and e["n_image_tokens"] == 0) or (with_image and n_img == e["n_image_tokens"])
        return {"text": ok_text, "len": ok_len, "image_slots": ok_img, "rendered": text}


def build(backbone: str, out_dir: str, verbose: bool = True) -> dict:
    TB = testbed()
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    cdir = os.path.join(C.CACHE, backbone)
    meta = C.jload(os.path.join(cdir, "meta.json"))
    with open(os.path.join(cdir, "index.jsonl"), "rb") as f:
        raw = f.read()
    if C.bytes_md5(raw) != meta["index_md5"]:
        raise RuntimeError(f"{backbone}: index.jsonl is not the index that meta.json describes")
    index = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    say(f"[{backbone}] {len(index)} sequences, families {list(meta['families'])}")

    # ---- the loaders the cache was built with
    fam, streams = {}, {}
    for f_, info in meta["families"].items():
        cfg = TB.LOGS[info["master_log"]]
        items, recs = TB.load_stream(cfg["stream"])
        probes, _ = TB.select_probes(items, key_exclude=(backbone == "llava"))
        fam[f_] = {"items": items, "recs": recs, "probes": probes, "master": info["master_log"], "dataset": info["dataset"]}
        streams[f_] = cfg["stream"]
        say(f"  family {f_}: {info['master_log']}, {len(items)} edits, {len(probes)} probe items ({info['dataset']})")
    ev = {d["uid"]: d for d in TB.load_vlkeb("eval")}
    hn = {r["hn_id"]: r for r in C.jsonl_read(os.path.join(C.CACHE, "hard_negatives", backbone, "rows.jsonl"))}
    ben = {r["sid"]: r for r in C.jsonl_read(os.path.join(C.CACHE, "benign", backbone, "rows.jsonl"))}
    vlkeb = [f_ for f_, v in fam.items() if v["dataset"] == "VLKEB"]
    assert len(vlkeb) == 1, vlkeb
    VF = vlkeb[0]

    # ---- 1. derive (question, image) of every sequence from the loaders
    derived, bad = [], collections.Counter()
    for j, e in enumerate(index):
        k, F = e["kind"], fam[e["family"]]
        if k in C.EDIT_KINDS:
            d = F["items"][e["edit"]]
            bad["edit_uid"] += d["uid"] != e["edit_uid"]
            q, im = (d["rephrase"] if k == "tgen" else d["src"]), (d["image_rephrase"] if k == "igen" else d["image"])
        elif k == "probe":
            p = F["probes"][e["probe_idx"]]
            bad["probe_uid"] += p["uid"] != e["uid"]
            q, im = (p["m_loc_q"], p["m_loc"]) if e["ptype"] == "image" else (p["loc"], None)
        elif k in C.HN_KINDS:
            r = hn[e["hn_id"]]
            q, im = r["prompt"], r["image"]
            d = F["items"][e["edit"]]
            bad["hn_edit_uid"] += d["uid"] != e["edit_uid"] or r["edit_uid"] != e["edit_uid"]
            if k == "hn_i":                                               # the edit's image + the image-locality question of the source item
                bad["hn_rule"] += im != d["image"] or q != ev[r["src_uid"]]["m_loc_q"]
            elif k == "hn_q":                                             # the edit's question + the entity image of the source item
                bad["hn_rule"] += q != d["src"] or im != ev[r["src_uid"]]["image"]
            else:                                                         # the image of edit a + the question of edit b
                db = F["items"][e["edit_b"]]
                bad["hn_rule"] += im != d["image"] or q != db["src"] or db["uid"] != e["edit_b_uid"]
        elif k == "benign":
            r = ben[e["sid"]]
            d = ev[r["uid"]]
            bad["benign_uid"] += r["uid"] != e["uid"]
            q, im = (d["m_loc_q"], d["m_loc"]) if e["ptype"] == "image" else (d["loc"], None)
            bad["benign_row"] += q != r["prompt"] or im != r["image"]
        else:
            raise ValueError(f"unknown kind {k!r}")
        bad["prompt_vs_index"] += q != e["prompt"]
        bad["image_vs_index"] += im != e["image"]
        bad["ptype_vs_image"] += (im is None) != (e["ptype"] == "text")
        derived.append((q, im))
    if any(bad.values()):
        raise RuntimeError(f"{backbone}: the attributes derived from the data loaders differ from the cache index: {dict(bad)}")
    say(f"  1./2. (question, image) from the loaders == index fields for {len(index)} sequences")

    # ---- 3. md5 of every image file, recomputed
    paths = sorted({im for _, im in derived if im is not None} | {d["image"] for F in fam.values() for d in F["items"]})
    with ThreadPoolExecutor(4) as ex:
        md5 = dict(zip(paths, ex.map(lambda p: C.file_md5(image_path(p)), paths)))
    n_md5_bad = sum(md5[im] != e["image_md5"] for (q, im), e in zip(derived, index) if im is not None)
    n_md5_bad += sum(e["image_md5"] is not None for (q, im), e in zip(derived, index) if im is None)
    if n_md5_bad:
        raise RuntimeError(f"{backbone}: {n_md5_bad} image md5 sums of the cache index differ from the files")
    say(f"  3. md5 recomputed for {len(paths)} image files ({len(set(md5.values()))} distinct contents) == index")

    # ---- 4. render every prompt
    R = Renderer(backbone)
    rc = collections.Counter()
    samples = []
    for j, ((q, im), e) in enumerate(zip(derived, index)):
        c = R.check(e, q, im)
        rc["text_ok"] += c["text"]
        rc["len_ok"] += c["len"]
        rc["image_slots_ok"] += c["image_slots"]
        if not (c["text"] and c["len"] and c["image_slots"]):
            rc["failed"] += 1
            if len(samples) < 5:
                samples.append({"sid": e["sid"], "question": q, "rendered": c["rendered"], "flags": {k: c[k] for k in ("text", "len", "image_slots")}})
    if rc["failed"]:
        raise RuntimeError(f"{backbone}: {rc['failed']} prompts do not render to the cached sequence: {samples}")
    say(f"  4. rendered {len(index)} prompts: template around the question and token count == cache for all")
    ex_render = {}
    for j, ((q, im), e) in enumerate(zip(derived, index)):                # one rendered example per (family, ptype, fmt), for the record
        key = f"{e['family']}|{e['ptype']}|{e['fmt']}"
        if key not in ex_render:
            ex_render[key] = {"sid": e["sid"], "question": q, "rendered_prompt_text": R.check(e, q, im)["rendered"], "len_prompt": e["len_prompt"],
                              "n_image_tokens": e["n_image_tokens"]}

    # ---- the table
    rows = []
    for j, ((q, im), e) in enumerate(zip(derived, index)):
        rows.append(row("sequence", j, e["sid"], e["kind"], e["family"], e["ptype"], im, None if im is None else md5[im], q,
                        e.get("edit"), e.get("edit_uid"), e.get("edit_b"), e.get("edit_b_uid")))
    n_seq = len(rows)
    rel = {(e["family"], e["edit"]): j for j, e in enumerate(index) if e["kind"] == "rel"}
    ebad = collections.Counter()
    for f_, F in fam.items():
        for i, (d, r) in enumerate(zip(F["items"], F["recs"])):
            ebad["stream_file"] += r["edit_id"] != i or r["uid"] != d["uid"] or r["subject"] != d["src"] or r["image"] != d["image"]
            c = index[rel[(f_, i)]]
            ebad["rel_sequence"] += c["prompt"] != d["src"] or c["image"] != d["image"] or c["edit_uid"] != d["uid"]
            rows.append(row("edit", None, f"{f_}:request:{i}", "request", f_, "image", d["image"], md5[d["image"]], d["src"], i, d["uid"]))
    if any(ebad.values()):
        raise RuntimeError(f"{backbone}: edit requests differ between the loader, the stream file and the cached rel sequence: {dict(ebad)}")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "attributes.jsonl")
    C.jsonl_write(rows, p)
    say(f"  wrote {p}: {n_seq} sequence lines + {len(rows) - n_seq} edit lines")

    kinds = collections.Counter((r["family"], r["kind"]) for r in rows)
    info = {"backbone": backbone, "file": "attributes.jsonl", "md5": C.file_md5(p), "n_lines": len(rows), "n_sequences": n_seq, "n_edit_requests": len(rows) - n_seq,
            "fields": list(FIELDS), "cache_index_md5": meta["index_md5"], "families": {f_: {"master_log": F["master"], "dataset": F["dataset"],
                                                                                           "n_edits": len(F["items"]), "stream": streams[f_]} for f_, F in fam.items()},
            "vlkeb_family": VF, "by_family_kind": {f"{a}|{b}": n for (a, b), n in sorted(kinds.items())},
            "n_text_rows": sum(r["image"] is None for r in rows[:n_seq]),
            "n_distinct_image_paths": len(paths), "n_distinct_image_md5": len(set(md5.values())),
            "n_distinct_question_strings": len({r["question"] for r in rows}), "n_distinct_question_keys": len({r["question_key"] for r in rows}),
            "checks": {"sequences_derived_from_loaders_equal_index": n_seq, "mismatches": {k: int(v) for k, v in sorted(bad.items())},
                       "md5_recomputed_files": len(paths), "md5_mismatches": int(n_md5_bad),
                       "prompts_rendered": n_seq, "rendered_text_is_template_around_question": int(rc["text_ok"]),
                       "rendered_token_count_equals_cached_prompt_length": int(rc["len_ok"]), "image_slots_equal_cache": int(rc["image_slots_ok"]),
                       "edit_requests_equal_stream_file_and_rel_sequence": len(rows) - n_seq, "edit_mismatches": {k: int(v) for k, v in sorted(ebad.items())}},
            "rendered_examples": ex_render,
            "image_root_of_relative_paths": image_path("x")[:-2], "question_key_rule": "question.strip().lower()",
            "built_by": "feat2/retr/build_attributes.py"}
    C.jdump(info, os.path.join(out_dir, "attributes_meta.json"))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=C.BACKBONES)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    info = build(a.backbone, a.out or os.path.join(C.DATA, a.backbone))
    print({k: info[k] for k in ("md5", "n_sequences", "n_edit_requests", "n_distinct_image_md5", "n_distinct_question_keys")})
