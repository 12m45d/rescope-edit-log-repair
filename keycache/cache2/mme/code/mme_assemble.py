"""Step 4a: assemble the MME capture of one backbone from the capture parts (CPU).

    python -I -B mme_assemble.py <qwen25vl | llava> [--partial] [--trial]

Writes 10_final/cache2/mme/<backbone>/ in the file format of the backbone key caches (10_final/cache/<backbone>/, FORMAT.txt there),
which is the format the benign pool's keys are stored in and rescope.keycache.KeyCache reads:
    index.jsonl, keys_tf.pt, keys_decode.pt, keys_decode_steps.pt, meta.json, capture_parts.json
and rows.jsonl in the format of 10_final/cache/benign/<backbone>/rows.jsonl (+ part, category, ...).
The bookkeeping follows 10_final/cache/code/fc_assemble.py line by line for its benign rows (index fields, key rows, float32
images of the module-dtype keys, degenerate-answer flags fc_common.degenerate_flags, pre_correct_* = tb_common.contains_ref).
Everything is written into '<backbone>.building' and published with one rename (a previous version is moved to work/old/);
README.txt and verify.json are added by mme_finish.py.  --partial assembles the rows whose parts exist (meta: complete = false).
--trial writes to work/trial/<backbone>/ instead of the published location (a test of this script while the capture runs).
"""
from __future__ import annotations

import argparse
import collections
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402

DECODE_FIELDS = ("y0_text", "y0_ids", "y0_gen_ids", "y0_stop", "y0_raw", "y0_roundtrip_ids_equal", "w0_reproduced", "w0_n_ans_tok", "w0_first_mismatch",
                 "y0_valid", "y0_invalid_reason", "y0_note", "long_text", "long_ids", "long_stop")        # fc_assemble: the fields copied from a capture record
FLAG_FIELDS = ("q_in_edit", "q_in_edit_first_pos", "q_in_edit_exact", "img_in_edit", "img_in_edit_first_pos", "q_in_probe", "q_in_probe_exact",
               "img_in_probe", "query_in_probe")


def yes_no(TB, text) -> str:
    """'yes' | 'no' | 'other': the answer after the testbed's normalisation (tb_common.norm_answer: lower case, punctuation and
    articles removed) IS the word yes / no."""
    s = TB.norm_answer(text or "")
    return s if s in ("yes", "no") else "other"


def yes_no_first(TB, text) -> str:
    """'yes' | 'no' | 'other' by the FIRST word of the normalised answer ('Yes, it is' -> yes)."""
    w = TB.norm_answer(text or "").split()
    return w[0] if w and w[0] in ("yes", "no") else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=M.BACKBONES)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--trial", action="store_true")
    args = ap.parse_args()
    args.partial = args.partial or args.trial
    bb = args.backbone
    C = M.cache_code()
    TB = C.tb()
    import torch
    torch.set_num_threads(4)
    vf = C.BACKBONES[bb]["vlkeb"]
    t0 = time.time()
    sel = {r["sid"]: r for r in M.jsonl_read(os.path.join(M.SELECT, "selection.jsonl"))}
    src = os.path.join(M.SELECT, f"manifest_{bb}.jsonl")
    used = os.path.join(M.CAPTURE_WORK, bb, "select", "manifest.jsonl")
    if M.md5_file(src) != M.md5_file(used):
        raise SystemExit(f"the manifest the capture read ({used}) is not the selection in use ({src})")
    man = M.jsonl_read(src)
    pd = os.path.join(M.CAPTURE_WORK, bb, "parts")
    files = sorted(f for f in os.listdir(pd) if f.endswith(".pt"))
    recs, loc, blobs, part_meta = {}, {}, {}, []
    for f in files:
        b = torch.load(os.path.join(pd, f), map_location="cpu", weights_only=False)        # our own part files
        blobs[f] = b
        part_meta.append({"part": b["part"], "n": len(b["rows"]), "seconds": round(b["seconds"], 1), "gpu": b["gpu"], "torch": b["torch"],
                          "cuda_visible": b["cuda_visible"], "time": b["time"], "master": b["master"], "dtype": b["dtype"]})
        assert [r["sid"] for r in b["rows"]] == b["sids"], f
        for r in b["rows"]:
            assert r["sid"] not in recs, r["sid"]
            recs[r["sid"]] = r
            loc[r["sid"]] = f
    missing = [m["sid"] for m in man if m["sid"] not in recs]
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} rows without a captured part (e.g. {missing[:3]}); use --partial")
    extra = sorted(set(recs) - {m["sid"] for m in man})
    if extra:
        raise SystemExit(f"{len(extra)} captured rows are not in the manifest (e.g. {extra[:3]})")
    man = [m for m in man if m["sid"] in recs]
    dtypes = sorted({p["dtype"] for p in part_meta})
    masters = sorted({p["master"] for p in part_meta})
    assert len(dtypes) == 1 and len(masters) == 1, (dtypes, masters)
    M.say(f"{bb}: {len(files)} parts, {len(man)} rows ({len(missing)} missing), dtype {dtypes[0]}, master {masters[0]} ({time.time() - t0:.0f} s)")

    # ----------------------------------------------------------------------------- index + keys (fc_assemble, benign rows)
    N = len(man)
    n_tf = sum(bool(recs[m["sid"]]["has_tf"]) for m in man)
    n_tfp = sum(bool(recs[m["sid"]].get("has_tf_prompt")) for m in man)
    n_steps = sum(int(recs[m["sid"]]["n_decode_distinct"]) for m in man)
    d = int(blobs[files[0]]["K_dec"].shape[1])
    K_tf = torch.empty(n_tf + n_tfp, d, dtype=torch.float32)
    K_dec = torch.empty(N, d, dtype=torch.float32)
    K_steps = torch.empty(n_steps, d, dtype=torch.float32)
    index, rows = [], []
    i_tf = i_tfp = i_st = 0
    fams = [f for dname, c in sel[man[0]["sid"]]["overlap"].items() for f in c if M.jload(os.path.join(M.CACHE, dname, "meta.json"))["backbone"] == bb]
    for j, m in enumerate(man):
        r, b, s = recs[m["sid"]], blobs[loc[m["sid"]]], sel[m["sid"]]
        assert m["decode"] and s["prompt"] == m["prompt"] and s["image"] == m["image"]
        nd, r0 = int(r["n_decode_distinct"]), int(r["dec_row0"])
        K_steps[i_st:i_st + nd] = b["K_dec"][r0:r0 + nd].float()
        assert r["decode_map"][0] == 0 and len(r["decode_map"]) == C.N_DECODE
        K_dec[j] = K_steps[i_st]
        ov = {f: fl for dname, c in s["overlap"].items() for f, fl in c.items() if f in fams}        # the families of THIS backbone's caches
        e = {k: m[k] for k in ("sid", "kind")}
        e.update(family="MME", dataset="MME", lib_sid=m["lib_sid"], ptype=m["ptype"], fmt=m["fmt"], prompt=m["prompt"], image=m["image"],
                 image_md5=m["image_md5"], query_id=m["query_id"], n_same_query=m["n_same_query"], master_log=b["master"], capture_family=vf,
                 uid=m["uid"], pool=m["pool"], benign_idx=m["benign_idx"], answer_subset=m["answer_subset"], item_uid=m["uid"],
                 part=s["part"], category=s["category"], stratum=s["stratum"], question_id=s["question_id"], mme_idx=s["mme_idx"])
        e.update(answer=r["y0_text"], answer_from=m["answer_from"], y0_text=r["y0_text"], y0_valid=bool(r["y0_valid"]), y0_invalid_reason=r["y0_invalid_reason"],
                 long_text=r["long_text"])                              # long_text: not in the caches' index; kept here so that the index alone gives the items
        e.update(C.degenerate_flags(e["y0_text"]))
        for k in ("pre_exact", "pre_token_acc", "pre_n_tok", "pre_finite", "pre_first_mismatch"):
            e[k] = r.get(k)
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
        e.update({k: ov[vf][k] for k in FLAG_FIELDS})                   # top level: against the backbone's VLKEB family, as in the benign pool
        e.update(img_in_cache=s["img_in_cache"], overlap=ov, ref_answer=s["ref_answer"], ref_field=s["ref_field"])
        yn = {"y0_yn": yes_no(TB, r["y0_text"]), "y0_yn_first": yes_no_first(TB, r["y0_text"]), "long_yn_first": yes_no_first(TB, r["long_text"]),
              "ref_yn": yes_no(TB, s["ref_answer"])}
        pc10, pcl = bool(TB.contains_ref(r["y0_text"], s["ref_answer"])), bool(TB.contains_ref(r["long_text"], s["ref_answer"]))
        e.update(pre_correct_long=pcl, **yn)
        index.append(e)
        # the row in the format of benign/<backbone>/rows.jsonl
        row = {"ptype": m["ptype"], "uid": m["uid"], "prompt": m["prompt"], "image": m["image"], "fmt": m["fmt"], "ref_answer": s["ref_answer"],
               "ref_field": s["ref_field"], "benign_idx": m["benign_idx"], "sid": m["sid"], "answer_subset": m["answer_subset"],
               "part": s["part"], "category": s["category"], "stratum": s["stratum"], "question_id": s["question_id"], "mme_idx": s["mme_idx"],
               "image_md5": m["image_md5"]}
        row.update({k: ov[vf][k] for k in FLAG_FIELDS})
        row.update(img_in_cache=s["img_in_cache"], overlap=ov)
        row.update({k: r[k] for k in DECODE_FIELDS})
        row["y0_valid"] = bool(r["y0_valid"])
        row.update(C.degenerate_flags(row["y0_text"]))
        row.update(pre_correct_at10=pc10, pre_correct_long=pcl, **yn)
        row.update(lp_tf=r["lp_tf"], lp_decode=r["lp_decode"], lp_shift=r["lp_shift"], pre_exact=r.get("pre_exact"), pre_token_acc=r.get("pre_token_acc"),
                   pre_n_tok=r.get("pre_n_tok"), len_prompt=r["len_prompt"], len_tf=r["len_tf"], n_image_tokens=r["n_image_tokens"],
                   n_decode_distinct=nd, query_id=m["query_id"], n_same_query=m["n_same_query"], row_tf=e["row_tf"], row_decode=e["row_decode"])
        rows.append(row)
    assert i_tf == n_tf and i_tfp == n_tfp and i_st == n_steps
    del blobs

    final = os.path.join(M.WORK, "trial", bb) if args.trial else os.path.join(M.MME, bb)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    od = final + ".building"
    if os.path.isdir(od):
        shutil.rmtree(od)
    os.makedirs(od)

    def save(obj, name):
        tmp = os.path.join(od, name + f".tmp{os.getpid()}")
        torch.save(obj, tmp)
        os.replace(tmp, os.path.join(od, name))

    save(K_tf, "keys_tf.pt")
    save(K_dec, "keys_decode.pt")
    save(K_steps, "keys_decode_steps.pt")
    M.jsonl_write(index, os.path.join(od, "index.jsonl"))
    M.jsonl_write(rows, os.path.join(od, "rows.jsonl"))
    first = min(time.mktime(time.strptime(x["time"], "%Y-%m-%d %H:%M:%S")) - x["seconds"] for x in part_meta)
    lib = C.CODE + "/rescope/evaluate.py"
    meta = {"backbone": bb, "layer": C.BACKBONES[bb]["layer"], "module": f"input of model.language_model.layers.{C.BACKBONES[bb]['layer']}.mlp.down_proj",
            "module_dtype": dtypes[0], "d_in": d, "families": {"MME": {"master_log": masters[0], "dataset": "MME", "capture_family": vf}},
            "what": "MME benchmark, out-of-benchmark benign workload (PROTOCOL.md section 6, R17), captured under the PRE-EDIT model like the benign pool",
            "model_state": "pre-edit weight W_0 of the edited module for the whole capture (Evaluator.applied(H.W0) of the handle of master_log)",
            "complete": not missing, "n_missing_sequences": len(missing), "n_sequences": len(index), "seed": M.SEED,
            "by_family_kind": {"MME|benign": len(index)},
            "by_part": {p: sum(e["part"] == p for e in index) for p in M.PARTS},
            "keys_tf": {"shape": list(K_tf.shape), "n_main_rows": n_tf, "n_shifted_extra_rows": n_tfp, "md5": M.md5_file(os.path.join(od, "keys_tf.pt"))},
            "keys_decode": {"shape": list(K_dec.shape), "md5": M.md5_file(os.path.join(od, "keys_decode.pt"))},
            "keys_decode_steps": {"shape": list(K_steps.shape), "md5": M.md5_file(os.path.join(od, "keys_decode_steps.pt")),
                                  "distinct_keys_per_sequence_hist": {str(k): int(v) for k, v in sorted(collections.Counter(e["n_decode_distinct"] for e in index).items())}},
            "index_md5": M.md5_file(os.path.join(od, "index.jsonl")), "rows_md5": M.md5_file(os.path.join(od, "rows.jsonl")),
            "selection": {"selection.jsonl": M.md5_file(os.path.join(M.SELECT, "selection.jsonl")), f"manifest_{bb}.jsonl": M.md5_file(src)},
            "capture": {"n_parts": len(part_meta), "gpu_seconds": round(sum(p["seconds"] for p in part_meta), 1), "gpus": sorted({p["gpu"] for p in part_meta}),
                        "torch": sorted({p["torch"] for p in part_meta}), "cuda_visible_devices": sorted({str(p["cuda_visible"]) for p in part_meta}),
                        "first": min(p["time"] for p in part_meta), "last": max(p["time"] for p in part_meta), "batch_size": 1},
            "rescope_at_capture": {"evaluate_py_sha1": M.sha1_file(lib), "evaluate_py_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(lib))),
                                   "evaluate_py_older_than_first_capture_part": bool(os.path.getmtime(lib) < first)},
            "code_now": M.code_fingerprint(), "assembled": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code": "02_research/experiments/10_final/cache2/mme/code/{mme_extract,mme_select,mme_capture,mme_assemble,mme_verify,mme_finish}.py; "
                    "capture = 10_final/cache/code/fc_capture.main()"}
    M.jdump(meta, os.path.join(od, "meta.json"))                       # written last: the reader checks index and key files against it
    M.jdump({"parts": part_meta}, os.path.join(od, "capture_parts.json"))
    if os.path.isdir(final):                                            # a previous version is moved aside (never deleted)
        old = os.path.join(M.WORK, "old", time.strftime("%Y%m%d_%H%M%S") + "_" + bb)
        os.makedirs(os.path.dirname(old), exist_ok=True)
        os.rename(final, old)
        for f in ("README.txt", "verify.json"):                         # they describe the capture, not the assembly: kept until mme_finish.py renews them
            if os.path.exists(os.path.join(old, f)):
                shutil.copyfile(os.path.join(old, f), os.path.join(od, f + ".previous"))
    os.rename(od, final)
    M.say(f"  published {final}: index {len(index)} lines; keys_tf {tuple(K_tf.shape)} ({n_tf} + {n_tfp} shifted), keys_decode {tuple(K_dec.shape)}, "
          f"keys_decode_steps {tuple(K_steps.shape)}; valid y0 {sum(e['y0_valid'] for e in index)}/{len(index)}; "
          f"answers {dict(collections.Counter(e['y0_yn'] for e in index))}; complete = {not missing} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
