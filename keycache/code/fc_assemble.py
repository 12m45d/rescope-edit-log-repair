"""Assemble the key cache of one backbone from the capture parts.

    python fc_assemble.py <backbone> [--partial] [--tag smoke]

Writes 10_final/cache/<backbone>/ : index.jsonl, keys_tf.pt, keys_decode.pt, keys_decode_steps.pt,
pre_edit_flags.jsonl, FORMAT.txt, meta.json; 10_final/cache/hard_negatives/<backbone>/ (all hard negatives, existing
and new, in the file format of 08_rescope/p11_retrofit/out/hard_negatives) and 10_final/cache/benign/<backbone>/.
With --tag the output goes to work/<backbone>/assembled_<tag>/ (smoke tests).  --partial assembles the sequences
whose parts exist (meta.json: complete = false); without it a missing part is an error.
CPU only.  Keys are stored as float32 (exact images of the module-dtype values).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_common as C  # noqa: E402
import fc_capture as CAP  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("backbone")
ap.add_argument("--partial", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--part", type=int, default=100)
ap.add_argument("--out", default="", help="write under work/<backbone>/assembled_<out>/ instead of the final location (trial assemblies)")
args = ap.parse_args()
BB = args.backbone
CFG = C.BACKBONES[BB]
VF = CFG["vlkeb"]
TB = C.tb()
SD = os.path.join(C.WORK, BB, "select")
PD = os.path.join(C.WORK, BB, "parts" + (f"_{args.tag}" if args.tag else ""))
if args.tag or args.out:
    base = os.path.join(C.WORK, BB, f"assembled_{args.out or args.tag}")
    FINAL = [os.path.join(base, "cache"), os.path.join(base, "hard_negatives"), os.path.join(base, "benign")]
else:
    FINAL = [os.path.join(C.OUT, BB), os.path.join(C.HN_NEW, BB), os.path.join(C.BENIGN, BB)]
# Everything is written into '<dir>.building' and renamed at the end, so that a reader (rescope.KeyCache loads the index
# first and the key files lazily) never sees files of two assemblies; a previous version is moved to work/<backbone>/old.
OD, HD, BD = (p + ".building" for p in FINAL)
import shutil  # noqa: E402
for d_ in (OD, HD, BD):
    if os.path.isdir(d_):
        shutil.rmtree(d_)
    os.makedirs(d_)
man = C.jsonl_read(os.path.join(SD, "manifest.jsonl"))
hn_sel = {r["hn_id"]: r for r in C.jsonl_read(os.path.join(SD, "hn.jsonl"))}
ben_sel = {r["sid"]: r for r in C.jsonl_read(os.path.join(SD, "benign.jsonl"))}
hn_edits = C.jload(os.path.join(SD, "hn_edits.json"))
t0 = time.time()

# ----------------------------------------------------------------------------- parts
recs, loc, blobs, part_meta, editpre = {}, {}, {}, [], {}
files = sorted(f for f in os.listdir(PD) if f.endswith(".pt"))
for f in files:
    b = torch.load(os.path.join(PD, f), map_location="cpu", weights_only=False)        # our own part files
    blobs[f] = b
    part_meta.append({"part": b["part"], "n": len(b["rows"]), "seconds": round(b["seconds"], 1), "gpu": b["gpu"], "torch": b["torch"],
                      "cuda_visible": b["cuda_visible"], "time": b["time"], "master": b["master"]})
    for r in b["rows"]:
        if r["sid"].startswith("editpre:"):
            editpre[r["edit_uid"]] = r
        else:
            assert r["sid"] not in recs, r["sid"]
            recs[r["sid"]] = r
            loc[r["sid"]] = f
missing = [m["sid"] for m in man if m["sid"] not in recs]
if missing and not args.partial:
    raise SystemExit(f"{len(missing)} sequences without a captured part (e.g. {missing[:3]}); use --partial")
man = [m for m in man if m["sid"] in recs]
dtype = blobs[files[0]]["dtype"] if files else None
C.say(f"{BB}: {len(files)} parts, {len(man)} sequences ({len(missing)} missing), dtype {dtype} ({time.time() - t0:.0f} s)")

# ----------------------------------------------------------------------------- hard-negative flags (hn_build's rules)
pre_rel = {}
for r in C.jsonl_read(os.path.join(C.HN_OLD, BB, "edit_pre.jsonl")):
    pre_rel[r["edit_uid"]] = r
for u, r in editpre.items():
    pre_rel[u] = r


def hn_flags(sel, y0, long):
    """Relation flags of a hard negative to its edit(s), as 08_rescope/p11_retrofit/code/hn_build.py (HN-X: both edits)."""
    fl = {"y0_contains_edit_pred": TB.contains_ref(y0, sel["edit_pred"]), "y0_contains_edit_alt": TB.contains_ref(y0, sel["edit_alt"]),
          "long_contains_edit_pred": TB.contains_ref(long, sel["edit_pred"]), "long_contains_edit_alt": TB.contains_ref(long, sel["edit_alt"])}
    pa = pre_rel.get(sel["edit_uid"])
    fl["y0_equals_pre_rel_y0"] = None if pa is None else TB.norm_answer(y0) == TB.norm_answer(pa["rel"]["y0_text"])
    rel = bool(fl["long_contains_edit_pred"] or fl["long_contains_edit_alt"] or (sel["hn_type"] in ("HN-I", "HN-X") and fl["y0_equals_pre_rel_y0"]))
    if sel["hn_type"] == "HN-X":
        pb = pre_rel.get(sel["edit_b_uid"])
        fl.update({"long_contains_edit_b_pred": TB.contains_ref(long, sel["edit_b_pred"]),
                   "long_contains_edit_b_alt": TB.contains_ref(long, sel["edit_b_alt"]),
                   "y0_equals_pre_rel_y0_b": None if pb is None else TB.norm_answer(y0) == TB.norm_answer(pb["rel"]["y0_text"])})
        rel = rel or bool(fl["long_contains_edit_b_pred"] or fl["long_contains_edit_b_alt"])
    fl["related_flag"] = rel
    return fl


# ----------------------------------------------------------------------------- index + keys
N = len(man)
n_tf = sum(bool(recs[m["sid"]]["has_tf"]) for m in man)
n_tfp = sum(bool(recs[m["sid"]].get("has_tf_prompt")) for m in man)
n_steps = sum(int(recs[m["sid"]]["n_decode_distinct"]) for m in man)
d = int(blobs[files[0]]["K_dec"].shape[1])
K_tf = torch.empty(n_tf + n_tfp, d, dtype=torch.float32)
K_dec = torch.empty(N, d, dtype=torch.float32)
K_steps = torch.empty(n_steps, d, dtype=torch.float32)
index, hn_rows, ben_rows, flag_rows = [], [], [], []
i_tf = i_tfp = i_st = 0
for j, m in enumerate(man):
    r, b = recs[m["sid"]], blobs[loc[m["sid"]]]
    nd, r0 = int(r["n_decode_distinct"]), int(r["dec_row0"])
    K_steps[i_st:i_st + nd] = b["K_dec"][r0:r0 + nd].float()
    assert r["decode_map"][0] == 0 and len(r["decode_map"]) == C.N_DECODE
    K_dec[j] = K_steps[i_st]
    e = {k: m[k] for k in ("sid", "kind", "family", "dataset", "lib_sid", "ptype", "fmt", "prompt", "image", "image_md5", "query_id", "n_same_query")}
    e["master_log"] = CFG["families"][m["family"]]
    # identifiers
    for k in ("edit", "edit_uid", "metric", "edit_b", "edit_b_uid", "probe_idx", "uid", "group", "pool", "disjoint", "strict",
              "question_in_repair", "twin_of", "probe_query_id", "probe_n_same_query", "pre_correct_long", "hn_id", "hn_type",
              "src_uid", "k", "rule", "new", "same_category", "benign_idx", "answer_subset"):
        if k in m:
            e[k] = m[k]
    # aliases (the same values under the other obvious names)
    if "edit" in e:
        e["edit_pos"] = e["edit"]
    if m["kind"] == "probe":
        e["probe_uid"] = m["uid"]
    if m["lib_kind"] == "hn":
        e["hn_group"] = m["group"]
    e["item_uid"] = m.get("edit_uid") if m["lib_kind"] in ("edit", "hn") else m.get("uid")
    # answer (what is teacher-forced) and the pre-edit answer where one exists
    if m["decode"]:
        e.update(answer=r["y0_text"], answer_from=m["answer_from"], y0_text=r["y0_text"], y0_valid=bool(r["y0_valid"]),
                 y0_invalid_reason=r["y0_invalid_reason"])
    else:
        e.update(answer=m["answer"], answer_from=m["answer_from"], y0_text=m.get("y0_text"), y0_valid=m.get("y0_valid"),
                 y0_invalid_reason=m.get("y0_invalid_reason"))
    if e["y0_text"] is not None:
        e.update(C.degenerate_flags(e["y0_text"]))
    # pre-edit teacher-forced flags of the answer (edits: is the target already produced; others: is y0 reproduced)
    for k in ("pre_exact", "pre_token_acc", "pre_n_tok", "pre_finite", "pre_first_mismatch"):
        e[k] = r.get(k)
    # positions and key rows
    e.update(lp_tf=r["lp_tf"], lp_decode=r["lp_decode"], lp_prompt=r["lp_decode"], lp_shift=r["lp_shift"], len_tf=r["len_tf"],
             len_prompt=r["len_prompt"], n_image_tokens=r["n_image_tokens"], tf_note=r.get("tf_note"))
    e["row_tf"] = e["row_tf_prompt"] = None
    if r["has_tf"]:
        K_tf[i_tf] = b["K_tf"][int(r["tf_row"])].float()
        e["row_tf"] = i_tf
        i_tf += 1
    if r.get("has_tf_prompt"):
        K_tf[n_tf + i_tfp] = b["K_tfp"][int(r["tfp_row"])].float()
        e["row_tf_prompt"] = n_tf + i_tfp
        i_tfp += 1
    e["row_decode"] = j
    e["rows_decode_steps"] = [i_st + int(u) for u in r["decode_map"]]
    e["n_decode_distinct"] = nd
    i_st += nd
    # kind-specific rows
    if m["lib_kind"] == "hn":
        sel = dict(hn_sel[m["hn_id"]])
        if m["decode"]:
            sel.update({k: r[k] for k in ("y0_text", "y0_ids", "y0_gen_ids", "y0_stop", "y0_raw", "y0_roundtrip_ids_equal", "w0_reproduced",
                                          "w0_n_ans_tok", "w0_first_mismatch", "y0_valid", "y0_invalid_reason", "y0_note", "long_text",
                                          "long_ids", "long_stop")})
            sel.update(hn_flags(sel, sel["y0_text"], sel["long_text"]))
            sel.update(last_prompt_idx=r["lp_tf"] if r["has_tf"] else r["lp_decode"], n_image_tokens=r["n_image_tokens"],
                       seq_len=r["len_tf"] if r["has_tf"] else r["len_prompt"])
        sel.update(C.degenerate_flags(sel["y0_text"]))
        sel.update(lp_tf=r["lp_tf"], lp_decode=r["lp_decode"], lp_shift=r["lp_shift"], pre_exact=r.get("pre_exact"),
                   pre_token_acc=r.get("pre_token_acc"), query_id=m["query_id"], n_same_query=m["n_same_query"],
                   edit=sel["edit_pos"], edit_b=sel.get("edit_b_pos"))
        e.update(related_flag=bool(sel.get("related_flag")), src_related_edit_first_pos=sel.get("src_related_edit_first_pos"),
                 q_in_edit=sel.get("q_in_edit"), q_in_edit_first_pos=sel.get("q_in_edit_first_pos"))
        hn_rows.append(sel)
    elif m["lib_kind"] == "benign":
        sel = dict(ben_sel[m["sid"]])
        for k in ("q_in_edit", "q_in_edit_first_pos", "q_in_edit_exact", "img_in_edit", "img_in_edit_first_pos", "q_in_probe",
                  "q_in_probe_exact", "img_in_probe", "query_in_probe", "img_entity_in_edit", "img_entity_in_edit_first_pos",
                  "img_entity_in_probe", "overlap", "ref_answer", "ref_field"):
            if k in sel:
                e[k] = sel[k]
        if m["decode"]:
            sel.update({k: r[k] for k in ("y0_text", "y0_ids", "y0_gen_ids", "y0_stop", "y0_raw", "y0_roundtrip_ids_equal", "w0_reproduced",
                                          "w0_n_ans_tok", "w0_first_mismatch", "y0_valid", "y0_invalid_reason", "y0_note", "long_text",
                                          "long_ids", "long_stop")})
            sel.update(C.degenerate_flags(sel["y0_text"]))
            sel["pre_correct_at10"] = bool(TB.contains_ref(sel["y0_text"], sel["ref_answer"]))
            sel["pre_correct_long"] = bool(TB.contains_ref(sel["long_text"], sel["ref_answer"]))
            e["pre_correct_long"] = sel["pre_correct_long"]
        sel.update(lp_tf=r["lp_tf"], lp_decode=r["lp_decode"], lp_shift=r["lp_shift"], pre_exact=r.get("pre_exact"),
                   query_id=m["query_id"], n_same_query=m["n_same_query"], row_tf=e["row_tf"], row_decode=e["row_decode"])
        ben_rows.append(sel)
    elif m["lib_kind"] == "edit":
        pa = pre_rel.get(m["edit_uid"])                                 # decoder-v2 pre-edit answer of the edit sequence's own prompt
        if pa is not None and m["family"] == VF:                        # (exists for the 200 edits that have hard negatives)
            e.update(prompt_y0_text=pa[m["metric"]]["y0_text"], prompt_y0_valid=bool(pa[m["metric"]]["y0_valid"]))
        flag_rows.append({"sid": m["sid"], "family": m["family"], "edit": m["edit"], "edit_uid": m["edit_uid"], "metric": m["metric"],
                          "pre_exact": r.get("pre_exact"), "pre_token_acc": r.get("pre_token_acc"), "pre_n_tok": r.get("pre_n_tok"),
                          "pre_finite": r.get("pre_finite"), "pre_first_mismatch": r.get("pre_first_mismatch"),
                          "lp_shift": r["lp_shift"]})
    index.append(e)
assert i_tf == n_tf and i_tfp == n_tfp and i_st == n_steps
del blobs


def save(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


save(K_tf, os.path.join(OD, "keys_tf.pt"))
save(K_dec, os.path.join(OD, "keys_decode.pt"))
save(K_steps, os.path.join(OD, "keys_decode_steps.pt"))
C.jsonl_write(index, os.path.join(OD, "index.jsonl"))
C.jsonl_write(flag_rows, os.path.join(OD, "pre_edit_flags.jsonl"))
C.say(f"  index {len(index)} lines; keys_tf {tuple(K_tf.shape)} ({n_tf} + {n_tfp} shifted), keys_decode {tuple(K_dec.shape)}, "
      f"keys_decode_steps {tuple(K_steps.shape)} ({time.time() - t0:.0f} s)")

# ----------------------------------------------------------------------------- hard negatives in the P11 file format
if hn_rows:
    C.jsonl_write(hn_rows, os.path.join(HD, "rows.jsonl"))
    by = {e["sid"]: e for e in index}
    Kl = torch.stack([K_tf[by[r["hn_id"]]["row_tf"]] if by[r["hn_id"]]["row_tf"] is not None else K_dec[by[r["hn_id"]]["row_decode"]]
                      for r in hn_rows])
    save({"hn_ids": [r["hn_id"] for r in hn_rows], "K_last": Kl, "last_prompt_idx": [r["last_prompt_idx"] for r in hn_rows],
          "layer": CFG["layer"], "backbone": BB,
          "doc": "K_last[r] = input of the edited module at the last prompt token of row hn_ids[r] (teacher-forced sequence prompt + y0, "
                 "harness index; prompt-only sequence where y0 is empty), B = 1, float32 image of the module-dtype values; the same "
                 "values as keys_tf.pt[row_tf] of the backbone's key cache"}, os.path.join(HD, "keys.pt"))
    C.jsonl_write([pre_rel[u] for u in dict.fromkeys(r["edit_uid"] for r in hn_rows) if u in pre_rel], os.path.join(HD, "edit_pre.jsonl"))
if ben_rows:
    C.jsonl_write(ben_rows, os.path.join(BD, "rows.jsonl"))

# ----------------------------------------------------------------------------- meta + format description
def lib_fingerprint():
    """sha1 of the library module the capture runs through (evaluate.py) and whether it is older than the first capture
    part (the capture processes then imported exactly this file)."""
    p = C.CODE + "/rescope/evaluate.py"
    with open(p, "rb") as fh_:
        h = hashlib.sha1(fh_.read()).hexdigest()[:12]
    mt = os.path.getmtime(p)
    first = min(time.mktime(time.strptime(x["time"], "%Y-%m-%d %H:%M:%S")) - x["seconds"] for x in part_meta)
    return {"evaluate_py_sha1": h, "evaluate_py_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mt)),
            "evaluate_py_older_than_first_capture_part": bool(mt < first)}


kinds = collections.Counter((e["family"], e["kind"]) for e in index)


def same_partition(rows_, a, b):
    """Do two id fields induce the same grouping of the rows?"""
    return len({(e[a], e[b]) for e in rows_}) == len({e[a] for e in rows_}) == len({e[b] for e in rows_})


probe_part = {f: same_partition([e for e in index if e["kind"] == "probe" and e["family"] == f], "query_id", "probe_query_id")
              for f in CFG["families"] if any(e["kind"] == "probe" and e["family"] == f for e in index)}
meta = {"backbone": BB, "layer": CFG["layer"], "module": f"input of model.language_model.layers.{CFG['layer']}.mlp.down_proj",
        "module_dtype": dtype, "d_in": d, "families": {f: {"master_log": n, "dataset": C.DATASET[f]} for f, n in CFG["families"].items()},
        "complete": not missing, "n_missing_sequences": len(missing), "n_sequences": len(index), "seed": C.SEED,
        "by_family_kind": {f"{a}|{b}": n for (a, b), n in sorted(kinds.items())},
        "keys_tf": {"shape": list(K_tf.shape), "n_main_rows": n_tf, "n_shifted_extra_rows": n_tfp, "md5": md5(os.path.join(OD, "keys_tf.pt"))},
        "keys_decode": {"shape": list(K_dec.shape), "md5": md5(os.path.join(OD, "keys_decode.pt"))},
        "keys_decode_steps": {"shape": list(K_steps.shape), "md5": md5(os.path.join(OD, "keys_decode_steps.pt")),
                              "distinct_keys_per_sequence_hist": {str(k): int(v) for k, v in sorted(collections.Counter(e["n_decode_distinct"] for e in index).items())}},
        "index_md5": md5(os.path.join(OD, "index.jsonl")),
        "probe_rows_query_id_groups_equal_library_groups(splits.annotate)": probe_part,
        "capture": {"n_parts": len(part_meta), "gpu_seconds": round(sum(p["seconds"] for p in part_meta), 1),
                    "gpus": sorted({p["gpu"] for p in part_meta}), "torch": sorted({p["torch"] for p in part_meta}),
                    "cuda_visible_devices": sorted({str(p["cuda_visible"]) for p in part_meta}),
                    "first": min(p["time"] for p in part_meta), "last": max(p["time"] for p in part_meta)},
        "rescope_at_capture": lib_fingerprint(), "assembled": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": "02_research/experiments/10_final/cache/code/{fc_select,fc_capture,fc_assemble,fc_verify,fc_table}.py"}
C.jdump(meta, os.path.join(OD, "meta.json"))
C.jdump({"parts": part_meta}, os.path.join(OD, "capture_parts.json"))
import fc_format  # noqa: E402
with open(os.path.join(OD, "FORMAT.txt"), "w") as f:
    f.write(fc_format.cache_text(BB, meta))
if hn_rows:
    with open(os.path.join(HD, "README.txt"), "w") as f:
        f.write(fc_format.hn_text(BB, hn_rows, hn_edits))
if ben_rows:
    with open(os.path.join(BD, "README.txt"), "w") as f:
        f.write(fc_format.benign_text(BB, ben_rows))
# publish: one rename per directory (a previous version is moved aside first)
old = os.path.join(C.WORK, BB, "old", time.strftime("%Y%m%d_%H%M%S") + (f"_{args.out or args.tag}" if (args.out or args.tag) else ""))
for built, final in zip((OD, HD, BD), FINAL):
    if not os.listdir(built):
        os.rmdir(built)
        continue
    if os.path.isdir(final):
        os.makedirs(old, exist_ok=True)
        os.rename(final, os.path.join(old, os.path.basename(os.path.dirname(final)) + "_" + os.path.basename(final)))
    os.makedirs(os.path.dirname(final), exist_ok=True)
    os.rename(built, final)
C.say(f"  published {FINAL[0]} ({time.time() - t0:.0f} s); complete = {not missing}")
