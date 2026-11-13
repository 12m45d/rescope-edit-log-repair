"""Step 2: the selection of the MME rows (CPU, deterministic, seed 20261111; no model).

    python -I -B mme_select.py build [--out DIR]     writes select/ (or DIR): selection.jsonl, manifest_<backbone>.jsonl, select_summary.json
    python -I -B mme_select.py check                 runs `build` again in a fresh process into work/select_recheck and compares the
                                                     files byte for byte with the selection in use -> select/check_select.json

ROWS      all rows of the benchmark (mme_rows.jsonl of mme_extract.py), in benchmark order; sid = 'mme:<row number>'.
SPLIT     BY IMAGE (unit = image content, md5): MME-C (calibration source) and MME-T (test rows), 50 percent of the images each,
          stratified by category.  Rule:
            1. stratum of an image = the sorted set of the categories of its rows, joined by '+' (ten images are asked in two
               categories: their strata are the four pairs that occur);
            2. per stratum: the image md5s sorted as strings; perm = numpy.random.default_rng(useed('split', stratum)).permutation(n),
               useed = first 32 bits of sha1('20261111|split|<stratum>');
            3. the first n_C images of perm are MME-C, the rest MME-T; n_C = n // 2, plus 1 in every second stratum of odd size
               (odd strata in name order: the 1st, 3rd, 5th get the extra image in MME-C, the 2nd, 4th in MME-T);
            4. a row belongs to the part of its image: the rows of an image are never separated.
FLAGS     category; img_in_cache: the image content (md5 of the file) is an image of the project's caches = the image of any
          sequence of 10_final/cache/<dir>/index.jsonl (the named directories qwen25vl, llava, blip2, QO: edits, image rephrases, probes, hard
          negatives, benign rows), of any row of cache/hard_negatives/<backbone>/rows.jsonl or cache/benign/<backbone>/rows.jsonl
          (files hashed again here), or of any image in the caches' md5 memo (every benchmark image their selection read);
          expected: never.  [The recombination rows of cache2/recomb, a directory that is still being built, are compared in
          mme_verify.py cpu, not here.]  q_in_edit / q_in_probe: the question string (stripped, lower case:
          fc_common.norm_q) is the question or text rephrase of an edit / the question of a probe row, per cache directory and
          stream family (field `overlap`), computed from the index files exactly as fc_select.Occ does from the streams;
          *_exact: without the normalisation; query_in_probe: question AND image content are a probe query.
MANIFEST  manifest_<backbone>.jsonl: the rows as manifest sequences of 10_final/cache/code/fc_capture.py (kind 'benign', free image
          query, fmt 'vqa', decode = true for EVERY row: decoder-v2 pre-edit answer, teacher-forced key and decode-step keys).
No time stamp is written by `build`: a re-run reproduces every file byte for byte.
"""
from __future__ import annotations

import argparse
import collections
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402

FILES = ("selection.jsonl", "select_summary.json") + tuple(f"manifest_{b}.jsonl" for b in M.BACKBONES)
# The caches the flags are computed against are NAMED here (the directories of 10_final/cache that hold a key cache / hard-negative
# and benign rows on 2026-11-11), not discovered: a cache that another build adds later cannot change the selection silently.
# mme_verify.py cpu compares the images with every cache index that exists when it runs.
REF_CACHES = ("QO", "blip2", "llava", "qwen25vl")
REF_ROW_BACKBONES = ("blip2", "llava", "qwen25vl")
QFLAGS = ("q_in_edit", "q_in_edit_exact", "q_in_probe", "q_in_probe_exact", "query_in_probe", "img_in_edit", "img_in_probe")


# ----------------------------------------------------------------------------- what the caches hold
class Occ:
    """Question strings and image contents of the edits and of the probe rows of one stream family, read from a key-cache index
    (the model-free content of 10_final/cache/code/fc_select.Occ: src / rephrase, image / image_rephrase, probe rows)."""

    def __init__(self, index_rows):
        self.q_edit, self.qx_edit, self.img_edit = {}, {}, {}
        self.q_probe, self.qx_probe, self.img_probe, self.query_probe = set(), set(), set(), set()
        self.n_edit_seq = self.n_probe = 0
        for e in sorted((e for e in index_rows if e["kind"] in ("rel", "tgen", "igen")), key=lambda e: int(e["edit"])):
            self.n_edit_seq += 1
            i = int(e["edit"])
            self.q_edit.setdefault(M.norm_q(e["prompt"]), i)           # rel / igen: the edit's question; tgen: its text rephrase
            self.qx_edit.setdefault(e["prompt"], i)
            if e.get("image_md5"):
                self.img_edit.setdefault(e["image_md5"], i)            # rel / tgen: the edit's image; igen: its image rephrase
        for e in index_rows:
            if e["kind"] == "probe":
                self.n_probe += 1
                self.q_probe.add(M.norm_q(e["prompt"]))
                self.qx_probe.add(e["prompt"])
                self.query_probe.add((e["ptype"], e["prompt"], e.get("image_md5")))
                if e.get("image_md5"):
                    self.img_probe.add(e["image_md5"])

    def flags(self, prompt, md5) -> dict:
        nq = M.norm_q(prompt)
        return {"q_in_edit": nq in self.q_edit, "q_in_edit_first_pos": self.q_edit.get(nq), "q_in_edit_exact": prompt in self.qx_edit,
                "img_in_edit": md5 in self.img_edit, "img_in_edit_first_pos": self.img_edit.get(md5),
                "q_in_probe": nq in self.q_probe, "q_in_probe_exact": prompt in self.qx_probe, "img_in_probe": md5 in self.img_probe,
                "query_in_probe": ("image", prompt, md5) in self.query_probe}


def hash_files(paths) -> dict:
    """{path: md5 of the file} with 4 threads."""
    from concurrent.futures import ThreadPoolExecutor
    paths = sorted(set(paths))
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(paths, ex.map(M.md5_file, paths)))


def load_refs() -> dict:
    """Everything of the project's caches that the flags are computed against (read-only)."""
    C = M.cache_code()
    refs = {"caches": {}, "img": collections.defaultdict(set), "q_other": set(), "sources": {}, "inputs": {}}
    for d in REF_CACHES:
        idx = M.jsonl_read(os.path.join(M.CACHE, d, "index.jsonl"))
        meta = M.jload(os.path.join(M.CACHE, d, "meta.json"))
        fams = list(dict.fromkeys(e["family"] for e in idx))
        refs["caches"][d] = {"backbone": meta["backbone"], "families": {f: Occ([e for e in idx if e["family"] == f]) for f in fams},
                             "index_md5": M.md5_file(os.path.join(M.CACHE, d, "index.jsonl")), "n": len(idx),
                             "md5_of_sid": {e["sid"]: e.get("image_md5") for e in idx}}
        refs["img"][f"index:{d}"] |= {e["image_md5"] for e in idx if e.get("image_md5")}
        refs["q_other"] |= {M.norm_q(e["prompt"]) for e in idx if e["kind"] in ("hn_i", "hn_q", "hn_x", "benign")}
    # hard-negative and benign rows of every backbone: the image FILES are hashed again here (not taken from the index)
    todo, where = [], []
    for kind in ("hard_negatives", "benign"):
        for bb in REF_ROW_BACKBONES:
            p = os.path.join(M.CACHE, kind, bb, "rows.jsonl")
            refs["inputs"][f"cache/{kind}/{bb}/rows.jsonl"] = M.md5_file(p)
            for r in M.jsonl_read(p):
                if r.get("image"):
                    todo.append(C.image_path(r["image"]))
                    where.append((kind, bb, r.get("hn_id") or r.get("sid"), todo[-1]))
    h = hash_files(todo)
    n_same = n_cmp = 0
    for kind, bb, sid, p in where:
        refs["img"][f"{kind}:{bb}"].add(h[p])
        ref = refs["caches"].get(bb, {}).get("md5_of_sid", {}).get(sid)
        if ref is not None:
            n_cmp += 1
            n_same += int(ref == h[p])
    refs["sources"]["hn_and_benign_files_hashed_here"] = {"n_rows": len(where), "n_files": len(h), "n_rows_compared_with_index_md5": n_cmp,
                                                          "n_equal": n_same}
    memo = M.jload(os.path.join(M.CACHE, "work", "image_md5.json"))          # read, never flushed
    refs["inputs"]["cache/work/image_md5.json"] = M.md5_file(os.path.join(M.CACHE, "work", "image_md5.json"))
    refs["img"]["md5_memo_of_the_caches"] = set(memo.values())
    refs["img"] = dict(sorted(refs["img"].items()))                         # only files of 10_final/cache (finished builds): the selection
    return refs                                                             # does not depend on directories that are still being written


# ----------------------------------------------------------------------------- split
def split_images(rows) -> tuple:
    """({md5: part}, {md5: stratum}, per-stratum record) by the rule of the module docstring."""
    import numpy as np
    cats = collections.defaultdict(set)
    for r in rows:
        cats[r["image_md5"]].add(r["category"])
    stratum = {m: "+".join(sorted(c)) for m, c in cats.items()}
    by = collections.defaultdict(list)
    for m, s in stratum.items():
        by[s].append(m)
    odd = [s for s in sorted(by) if len(by[s]) % 2 == 1]
    part, rec = {}, {}
    for s in sorted(by):
        ims = sorted(by[s])
        n = len(ims)
        extra = 1 if (n % 2 == 1 and odd.index(s) % 2 == 0) else 0
        n_c = n // 2 + extra
        perm = np.random.default_rng(M.useed("split", s)).permutation(n)
        for rank, i in enumerate(perm.tolist()):
            part[ims[i]] = M.PARTS[0] if rank < n_c else M.PARTS[1]
        rec[s] = {"n_images": n, M.PARTS[0]: n_c, M.PARTS[1]: n - n_c, "seed": M.useed("split", s),
                  "odd": n % 2 == 1, "extra_image_in": None if n % 2 == 0 else (M.PARTS[0] if extra else M.PARTS[1])}
    return part, stratum, rec


# ----------------------------------------------------------------------------- build
def build(out: str) -> dict:
    C = M.cache_code()
    table = M.jsonl_read(M.ROW_TABLE)
    assert [r["mme_idx"] for r in table] == list(range(len(table)))
    bad = [r["image_file"] for r in table if not os.path.exists(os.path.join(M.IMAGES, r["image_file"]))]
    if bad:
        raise SystemExit(f"{len(bad)} image files are missing under {M.IMAGES} (run mme_extract.py)")
    h = hash_files(os.path.join(M.IMAGES, f) for f in {r["image_file"] for r in table})
    wrong = [p for p, m in h.items() if os.path.basename(p).split(".")[0] != m]
    if wrong:
        raise SystemExit(f"{len(wrong)} image files do not have their content md5 as name (e.g. {wrong[:3]})")
    refs = load_refs()
    part, stratum, strata = split_images(table)
    per_img = collections.Counter(r["image_md5"] for r in table)
    qid, sel = {}, []
    for r in table:
        md5, q = r["image_md5"], r["question"]
        core = q[:-len(M.SUFFIX)].strip() if q.endswith(M.SUFFIX) else q
        where = [k for k, s in refs["img"].items() if md5 in s]
        ov = {d: {f: o.flags(q, md5) for f, o in c["families"].items()} for d, c in refs["caches"].items()}
        allf = [fl for d in ov.values() for fl in d.values()]
        core_hit = any(M.norm_q(core) in o.q_edit or M.norm_q(core) in o.q_probe for c in refs["caches"].values() for o in c["families"].values())
        e = {"sid": f"mme:{r['mme_idx']}", "mme_idx": r["mme_idx"], "part": part[md5], "category": r["category"], "stratum": stratum[md5],
             "uid": f"mme_img:{md5}", "question_id": r["question_id"], "ptype": "image", "fmt": "vqa", "prompt": q,
             "image": os.path.join(M.IMAGES, r["image_file"]), "image_md5": md5, "ref_answer": r["ref_answer"], "ref_field": "answer",
             "rows_of_image": per_img[md5],
             "query_id": M.QUERY_ID0 + qid.setdefault(("image", q, md5), len(qid)),
             "img_in_cache": bool(where), "img_in_cache_where": where}
        e.update({k: any(fl[k] for fl in allf) for k in QFLAGS})
        e.update(q_without_suffix_in_edit_or_probe=core_hit, q_in_hn_or_benign_rows=M.norm_q(q) in refs["q_other"], overlap=ov)
        sel.append(e)
    cq = collections.Counter(e["query_id"] for e in sel)
    for e in sel:
        e["n_same_query"] = cq[e["query_id"]]
    os.makedirs(out, exist_ok=True)
    M.jsonl_write(sel, os.path.join(out, "selection.jsonl"))
    man_md5 = {}
    for bb in M.BACKBONES:
        vf = C.BACKBONES[bb]["vlkeb"]
        man = [{"sid": e["sid"], "kind": "benign", "family": vf, "dataset": "MME", "lib_kind": "benign", "lib_sid": e["sid"], "ptype": "image",
                "fmt": "vqa", "prompt": e["prompt"], "image": e["image"], "benign_idx": e["mme_idx"], "uid": e["uid"], "pool": "benign",
                "answer_subset": True, "answer": None, "answer_from": "decoder_v2_here", "y0_text": None, "y0_valid": None,
                "y0_invalid_reason": None, "decode": True, "query_id": e["query_id"], "image_md5": e["image_md5"],
                "n_same_query": e["n_same_query"], "part": e["part"], "category": e["category"]} for e in sel]
        M.jsonl_write(man, os.path.join(out, f"manifest_{bb}.jsonl"))
        man_md5[bb] = M.md5_file(os.path.join(out, f"manifest_{bb}.jsonl"))

    cats = list(dict.fromkeys(e["category"] for e in sel))

    def count(sub):
        return {"rows": len(sub), "images": len({e["image_md5"] for e in sub}), "ref_yes": sum(e["ref_answer"] == "Yes" for e in sub),
                "ref_no": sum(e["ref_answer"] == "No" for e in sub)}
    by_part = {p: [e for e in sel if e["part"] == p] for p in M.PARTS}
    img_parts = collections.defaultdict(set)
    for e in sel:
        img_parts[e["image_md5"]].add(e["part"])
    qs = {p: {M.norm_q(e["prompt"]) for e in by_part[p]} for p in M.PARTS}
    summ = {
        "what": "selection of the MME rows: all rows; split by image into MME-C and MME-T, stratified by category; flags against the project's caches",
        "protocol": "10_final/PROTOCOL.md, section 6, amendment 'Clarification of H3 (R6) and R17', item (2)",
        "seed": M.SEED, "rule": __doc__.split("SPLIT")[1].split("FLAGS")[0].strip(),
        "row_table_md5": M.md5_file(M.ROW_TABLE), "n_rows": len(sel), "n_images": len(per_img), "n_categories": len(cats),
        "n_distinct_queries": len(qid), "query_id_range": [M.QUERY_ID0, M.QUERY_ID0 + len(qid) - 1],
        "parts": {p: count(by_part[p]) for p in M.PARTS},
        "share_of_images": {p: round(len({e["image_md5"] for e in by_part[p]}) / len(per_img), 6) for p in M.PARTS},
        "images_in_both_parts": sum(len(v) > 1 for v in img_parts.values()),
        "by_category": {c: {p: count([e for e in by_part[p] if e["category"] == c]) for p in M.PARTS} for c in cats},
        "strata": strata,
        "question_strings_in_both_parts": len(qs[M.PARTS[0]] & qs[M.PARTS[1]]),
        "rows_whose_question_string_also_occurs_in_the_other_part": {p: sum(M.norm_q(e["prompt"]) in qs[o] for e in by_part[p])
                                                                     for p, o in zip(M.PARTS, M.PARTS[::-1])},
        "flags": {
            "img_in_cache": {"n_rows": sum(e["img_in_cache"] for e in sel), "n_images": len({e["image_md5"] for e in sel if e["img_in_cache"]}),
                             "by_reference_set": {k: len({e["image_md5"] for e in sel} & s) for k, s in refs["img"].items()},
                             "reference_set_sizes(distinct image contents)": {k: len(s) for k, s in refs["img"].items()},
                             "hn_and_benign_files_hashed_here": refs["sources"]["hn_and_benign_files_hashed_here"]},
            "question": {k: sum(bool(e[k]) for e in sel) for k in QFLAGS + ("q_without_suffix_in_edit_or_probe", "q_in_hn_or_benign_rows")},
            "by_cache_and_family": {d: {f: {k: sum(bool(e["overlap"][d][f][k]) for e in sel) for k in QFLAGS} for f in c["families"]}
                                    for d, c in refs["caches"].items()},
        },
        "reference_caches": {d: {"backbone": c["backbone"], "n_sequences": c["n"], "index_md5": c["index_md5"],
                                 "families": {f: {"n_edit_sequences": o.n_edit_seq, "n_probe_rows": o.n_probe, "n_edit_question_strings": len(o.q_edit),
                                                  "n_edit_images": len(o.img_edit), "n_probe_question_strings": len(o.q_probe),
                                                  "n_probe_images": len(o.img_probe)} for f, o in c["families"].items()}}
                             for d, c in refs["caches"].items()},
        "inputs_md5": {"cache2/mme/mme_rows.jsonl": M.md5_file(M.ROW_TABLE),
                       **{f"cache/{d}/index.jsonl": c["index_md5"] for d, c in refs["caches"].items()}, **refs["inputs"]},
        "determinism": "the files of this directory are a function of the input files listed under inputs_md5 (and of the image files, "
                       "whose names are their md5 sums); no time stamp, no dependence on directory listings or hash order",
        "files": {"selection.jsonl": M.md5_file(os.path.join(out, "selection.jsonl")),
                  **{f"manifest_{bb}.jsonl": man_md5[bb] for bb in M.BACKBONES}},
    }
    M.jdump(summ, os.path.join(out, "select_summary.json"))
    return summ


def check() -> dict:
    """Determinism: `build` once more in a fresh process (its own hash randomisation) into work/select_recheck; byte comparison."""
    d = os.path.join(M.WORK, "select_recheck")
    os.makedirs(d, exist_ok=True)
    for f in FILES:
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    r = subprocess.run([M.PYTHON, "-I", "-B", os.path.abspath(__file__), "build", "--out", d], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("the second selection run failed:\n" + r.stderr[-2000:])
    res = {"what": "the selection was run a second time in a fresh process into work/select_recheck; files compared byte for byte",
           "files": {}, "hash_randomisation": "python -I ignores PYTHONHASHSEED: every process has its own random string-hash seed, so a "
                                              "result that depended on set or dict-of-set order would differ between the two runs"}
    for f in FILES:
        a, b = os.path.join(M.SELECT, f), os.path.join(d, f)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            same = fa.read() == fb.read()
        res["files"][f] = {"identical": same, "md5": M.md5_file(a), "md5_second_run": M.md5_file(b), "bytes": os.path.getsize(a)}
    res["ok"] = all(v["identical"] for v in res["files"].values())
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    M.jdump(res, os.path.join(M.SELECT, "check_select.json"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "check"))
    ap.add_argument("--out", default=M.SELECT)
    a = ap.parse_args()
    if a.cmd == "build":
        s = build(a.out)
        print(f"selection: {s['n_rows']} rows, {s['n_images']} images, {s['n_categories']} categories; parts {s['parts']}; "
              f"images in both parts {s['images_in_both_parts']}; images also in the caches: {s['flags']['img_in_cache']['n_images']}; "
              f"question flags {s['flags']['question']}")
        print(M.text_table([(c,) + tuple(v[p][k] for p in M.PARTS for k in ("images", "rows", "ref_yes", "ref_no")) for c, v in s["by_category"].items()],
                           head=("category",) + tuple(f"{p} {k}" for p in M.PARTS for k in ("images", "rows", "yes", "no")), right=tuple(range(1, 9))))
    else:
        r = check()
        print("selection reproduced byte for byte:", r["ok"], {f: v["identical"] for f, v in r["files"].items()})
        return 0 if r["ok"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
