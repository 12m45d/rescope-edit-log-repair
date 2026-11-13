"""Step 5: README.txt and verify.json of each backbone's MME capture, the directory README.txt and summary.json (CPU; no model).

    python -I -B mme_finish.py [qwen25vl] [llava]

Every number of the texts is computed here from the published files (index.jsonl, rows.jsonl, meta.json), the selection
(select/), the extraction summary and the verification results (work/proof/<backbone>/proof.json, work/verify/<backbone>_gpu.json,
work/verify/<backbone>_cpu.json, select/check_select.json, test_mme_pool.json); nothing is typed by hand.  A verification result
that does not belong to the published files (md5 of index, rows and key files) is refused.
"""
from __future__ import annotations

import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402

T = M.text_table


def pc(x) -> str:
    """A percentage with two decimals ('-' if undefined)."""
    return "-" if x is None else f"{x:.2f}"


def wrap(items, width=118, indent="    ", sep="; ") -> list:
    """Lines of at most `width` characters holding the items in order."""
    lines, cur = [], indent
    for i, it in enumerate(items):
        piece = it + (sep if i < len(items) - 1 else "")
        if len(cur) + len(piece) > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += piece
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


def quant(v):
    import numpy as np
    v = np.asarray(v)
    return {"min": int(v.min()), "p05": int(np.quantile(v, 0.05)), "median": int(np.median(v)), "p95": int(np.quantile(v, 0.95)), "max": int(v.max())}


def stats(bb) -> dict:
    """Counts of one backbone's capture (from its index and rows)."""
    od = os.path.join(M.MME, bb)
    index, rows = M.jsonl_read(os.path.join(od, "index.jsonl")), M.jsonl_read(os.path.join(od, "rows.jsonl"))
    cats = list(dict.fromkeys(e["category"] for e in index))

    def block(sub):
        n, v = len(sub), [e for e in sub if e["y0_valid"]]
        yn = collections.Counter(e["y0_yn"] for e in sub)
        return {"rows": n, "images": len({e["image_md5"] for e in sub}), "valid": len(v), "valid_percent": round(100.0 * len(v) / n, 2) if n else None,
                "yes": yn["yes"], "no": yn["no"], "other": yn["other"],
                "valid_yes": sum(e["y0_yn"] == "yes" for e in v), "valid_no": sum(e["y0_yn"] == "no" for e in v), "valid_other": sum(e["y0_yn"] == "other" for e in v),
                "equal_to_reference": sum(e["y0_yn"] == e["ref_yn"] for e in sub),
                "equal_to_reference_percent": round(100.0 * sum(e["y0_yn"] == e["ref_yn"] for e in sub) / n, 2) if n else None,
                "ref_yes": sum(e["ref_yn"] == "yes" for e in sub), "ref_no": sum(e["ref_yn"] == "no" for e in sub)}
    ben = [e for e in M.jsonl_read(os.path.join(M.CACHE, bb, "index.jsonl")) if e["kind"] == "benign" and e["ptype"] == "image"]
    mxb = max(e["len_prompt"] for e in ben)
    by_sid = {r["sid"]: r for r in rows}
    return {
        "backbone": bb, "n_rows": len(index), "n_images": len({e["image_md5"] for e in index}), "n_categories": len(cats),
        "all": block(index), "by_part": {p: block([e for e in index if e["part"] == p]) for p in M.PARTS},
        "by_category": {c: dict(block([e for e in index if e["category"] == c]),
                                **{p: {"rows": sum(e["category"] == c and e["part"] == p for e in index),
                                       "images": len({e["image_md5"] for e in index if e["category"] == c and e["part"] == p}),
                                       "valid": sum(e["category"] == c and e["part"] == p and e["y0_valid"] for e in index)} for p in M.PARTS}) for c in cats},
        "answer_strings": dict(collections.Counter(e["y0_text"] for e in index).most_common()),
        "other_answers": [{"sid": e["sid"], "part": e["part"], "category": e["category"], "y0_text": e["y0_text"], "long_text": by_sid[e["sid"]]["long_text"][:120],
                           "y0_valid": e["y0_valid"], "first_word": e["y0_yn_first"]} for e in index if e["y0_yn"] == "other"],
        "other_by_first_word": dict(collections.Counter(e["y0_yn_first"] for e in index if e["y0_yn"] == "other")),
        "invalid_rows": [{"sid": e["sid"], "part": e["part"], "category": e["category"], "y0_text": e["y0_text"], "reason": e["y0_invalid_reason"],
                          "pre_first_mismatch": e["pre_first_mismatch"], "pre_n_tok": e["pre_n_tok"], "prompt": e["prompt"]} for e in index if not e["y0_valid"]],
        "invalid_reasons": dict(collections.Counter(e["y0_invalid_reason"] for e in index if not e["y0_valid"])),
        "n_answer_tokens": {str(k): v for k, v in sorted(collections.Counter(e["pre_n_tok"] for e in index).items(), key=lambda t: (t[0] is None, t[0]))},
        "y0_stop": dict(collections.Counter(r["y0_stop"] for r in rows)), "long_stop": dict(collections.Counter(r["long_stop"] for r in rows)),
        "long_answer_equals_short_answer": sum(r["long_text"] == r["y0_text"] for r in rows),
        "y0_ids_equal_generated_ids": sum(bool(r["y0_roundtrip_ids_equal"]) for r in rows),
        "degenerate": {k: sum(bool(e[k]) for e in index) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")},
        "deg_none_no_among_answers_no": [sum(bool(e["deg_none_no"]) for e in index if e["y0_yn"] == "no"), sum(e["y0_yn"] == "no" for e in index)],
        "with_teacher_forced_key": sum(e["row_tf"] is not None for e in index),
        "lp_shift": {str(k): v for k, v in collections.Counter(e["lp_shift"] for e in index).items()},
        "n_decode_distinct": {str(k): v for k, v in sorted(collections.Counter(e["n_decode_distinct"] for e in index).items())},
        "len_prompt": quant([e["len_prompt"] for e in index]), "n_image_tokens": quant([e["n_image_tokens"] for e in index]),
        "n_text_tokens": quant([e["len_prompt"] - e["n_image_tokens"] for e in index]),
        "benign_pool_image_rows": {"n": len(ben), "len_prompt": quant([e["len_prompt"] for e in ben]), "n_image_tokens": quant([e["n_image_tokens"] for e in ben]),
                                   "n_text_tokens": quant([e["len_prompt"] - e["n_image_tokens"] for e in ben])},
        "n_rows_longer_than_the_longest_benign_image_row": sum(e["len_prompt"] > mxb for e in index),
        "flags": {k: sum(bool(e[k]) for e in index) for k in ("img_in_cache", "q_in_edit", "q_in_probe", "img_in_edit", "img_in_probe", "query_in_probe")},
        "fields": {"index": list(index[0].keys()), "rows": list(rows[0].keys()),
                   "benign_pool_rows_fields_absent_here": sorted(set().union(*(r.keys() for r in M.jsonl_read(os.path.join(M.CACHE, "benign", bb, "rows.jsonl"))))
                                                                 - set(rows[0].keys())),
                   "rows_fields_not_in_the_benign_pool": sorted(set(rows[0].keys()) - set().union(*(r.keys() for r in M.jsonl_read(os.path.join(M.CACHE, "benign", bb, "rows.jsonl"))))),
                   "cache_index_benign_fields_absent_here": sorted(set().union(*(e.keys() for e in ben)) - set(index[0].keys())),
                   "index_fields_not_in_the_cache_index": sorted(set(index[0].keys()) - set().union(*(e.keys() for e in ben)))},
    }


def verify_json(bb, st) -> dict:
    od = os.path.join(M.MME, bb)
    meta = M.jload(os.path.join(od, "meta.json"))
    ident = {"index_md5": M.md5_file(os.path.join(od, "index.jsonl")), "rows_md5": M.md5_file(os.path.join(od, "rows.jsonl")),
             "keys_tf_md5": meta["keys_tf"]["md5"], "keys_decode_md5": meta["keys_decode"]["md5"], "keys_decode_steps_md5": meta["keys_decode_steps"]["md5"]}
    proof = M.jload(os.path.join(M.PROOF_WORK, bb, "proof.json"))
    stp = os.path.join(M.PROOF_WORK, bb, "selftest_compare.json")
    selftest = M.jload(stp) if os.path.exists(stp) else None
    exr = {f: os.path.exists(os.path.join(M.WORK, "extract_recheck", f)) and M.md5_file(os.path.join(M.WORK, "extract_recheck", f)) == M.md5_file(os.path.join(M.MME, f))
           for f in ("mme_rows.jsonl", "extract_summary.json")}
    gpu = M.jload(os.path.join(M.WORK, "verify", f"{bb}_gpu.json"))
    cpu = M.jload(os.path.join(M.WORK, "verify", f"{bb}_cpu.json"))
    chk = M.jload(os.path.join(M.SELECT, "check_select.json"))
    summ = M.jload(os.path.join(M.SELECT, "select_summary.json"))
    for name, v in (("gpu", gpu), ("cpu", cpu)):
        if v["files"] != ident:
            raise SystemExit(f"{bb}: the {name} verification belongs to other files than the published ones (run mme_verify.py {bb} {name} again)")
    sel_ok = (chk["ok"] and chk["files"]["selection.jsonl"]["md5"] == meta["selection"]["selection.jsonl"] == M.md5_file(os.path.join(M.SELECT, "selection.jsonl"))
              and chk["files"][f"manifest_{bb}.jsonl"]["md5"] == meta["selection"][f"manifest_{bb}.jsonl"]
              == M.md5_file(os.path.join(M.CAPTURE_WORK, bb, "select", "manifest.jsonl")))
    tp = os.path.join(M.MME, "test_mme_pool.json")
    test = M.jload(tp) if os.path.exists(tp) else None
    glue_ok = None
    if test is not None and bb in test["backbones"]:
        tb_ = test["backbones"][bb]
        glue_ok = bool(tb_["A_reader"]["ok"] and tb_["C_example"]["ok"] and tb_["capture"]["index_md5"] == ident["index_md5"])
    s = gpu["sample"]
    out = {
        "backbone": bb, "what": "verification of the MME capture (PROTOCOL.md section 6, R17); written by code/mme_finish.py from the results of "
                                "code/mme_capture.py (proof), code/mme_verify.py (gpu, cpu), code/mme_select.py (check) and test_mme_pool.py",
        "files": ident,
        "summary": {
            "bitwise_recapture_of_50_random_mme_rows": {
                "ok": gpu["ok"], "n_rows": s["n"], "n_rows_all_keys_bitwise": s["n_rows_all_keys_bitwise"],
                "prompt_only_key": s["decode_key_equal"], "decode_step_keys(11 per row)": s["decode_steps_equal"], "teacher_forced_key": s["tf_key_equal"],
                "teacher_forced_key_at_decode_token": s["tf_prompt_key_equal"], "eligible_keys_all": s["eligible_all_equal"],
                "n_rows_answer_and_validity_reproduced": s["n_rows_answer_and_validity_reproduced"],
                "extra_rows(invalid answer / answer not yes-no / shifted position)": {k: (None if v is None else [v["n_rows_all_keys_bitwise"], v["n"]]) for k, v in gpu["extra"].items()},
                "fresh_process": True, "vision_memo": False, "seed": gpu["seed"], "time": gpu["time"]},
            "code_path_proof_30_rows_of_the_existing_benign_pool": {
                "ok": proof["ok"], "n_rows": proof["n_rows"], "n_rows_ok": proof["n_rows_ok"], "prompt_only_key_bitwise": proof["prompt_only_key_bitwise"],
                "decode_step_keys_bitwise": proof["decode_step_keys_bitwise(rows; 11 keys each)"], "teacher_forced_key_bitwise": proof["teacher_forced_key_bitwise"],
                "answers_ids_validity_positions_equal": [proof["n_rows_answer_ids_validity_positions_equal"], proof["n_rows"]],
                "before_the_capture": proof["time"] <= meta["capture"]["first"],
                "same_code_at_capture": "code/mme_capture.py refuses to capture unless the proof passed with the fingerprint of the code it is about to run",
                "same_code_at_assembly": proof["code"] == meta["code_now"], "code_unchanged_until_now": proof["code"] == M.code_fingerprint(),
                "code_fingerprint": proof["code"], "time": proof["time"],
                "comparison_selftest(a copy of the cache passes; one flipped bit of a key or a changed answer is detected)": selftest},
            "selection_deterministic": {"ok": sel_ok, "files_identical_on_a_second_run": {f: v["identical"] for f, v in chk["files"].items()},
                                        "row_table_identical_on_a_second_extraction(work/extract_recheck)": exr,
                                        "selection_md5": chk["files"]["selection.jsonl"]["md5"], "the_capture_read_this_manifest": True if sel_ok else False,
                                        "time": chk["time"]},
            "no_image_shared_with_the_caches": {
                "ok": cpu["images"]["n_images_shared_with_any_cache"] == 0 and summ["flags"]["img_in_cache"]["n_images"] == 0,
                "n_images": cpu["images"]["n_files"], "n_shared_by_reference_set(selection)": summ["flags"]["img_in_cache"]["by_reference_set"],
                "reference_set_sizes": summ["flags"]["img_in_cache"]["reference_set_sizes(distinct image contents)"],
                "n_shared_with_cache_index(files hashed again)": cpu["images"]["n_images_shared_with_cache_index"],
                "n_question_strings_equal_to_an_edit_or_probe_question": summ["flags"]["question"]},
            "internal_consistency": {"ok": cpu["ok"], "tf_key_vs_decode_step_of_same_length": cpu["consistency"]["tf_key_vs_decode_step_of_same_length"],
                                     "n_images_in_both_parts": cpu["selection"]["n_images_in_both_parts"],
                                     "cpu_dry_run_of_prompt_construction": cpu["selection"]["cpu_dry_run_of_prompt_construction"],
                                     "pixel_identity": cpu.get("pixel_identity"), "n_distinct_prompt_only_keys": cpu["key_statistics"]["n_distinct_keys_bitwise"],
                                     "library_reader_ok": cpu["library_reader"]["ok"]},
            "pre_edit_model_only": {"n_forwards_with_a_weight_other_than_W0_in_the_verification": gpu["edited_module_weight"]["n_forwards_with_a_weight_other_than_W0"],
                                    "n_forwards_of_the_edited_module_in_the_verification": gpu["edited_module_weight"]["n_forwards_of_the_edited_module"],
                                    "capture": "fc_capture.main keeps the handle at W_0 for every part (Evaluator.applied(H.W0), asserted at the start of each part; "
                                               "a failed assertion would have ended the job, and every part exists)",
                                    "gates_and_calibration": "not measured, by construction: no script of cache2/mme builds, calibrates or evaluates a gate on MME "
                                                             "keys; test_mme_pool.py runs the library's gate functions on RANDOM keys and a synthetic detector set only"},
            "glue_mme_pool": {"gpu(pre-edit weight as the method)": gpu["glue_mme_pool"], "cpu_test_ok": glue_ok},
        },
        "proof": {k: v for k, v in proof.items() if k != "rows"}, "proof_rows": proof["rows"],
        "gpu": gpu, "cpu": cpu, "selection_check": chk, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    sm = out["summary"]
    out["ok"] = bool(sm["bitwise_recapture_of_50_random_mme_rows"]["ok"] and s["n"] >= M.N_VERIFY and s["n_rows_all_keys_bitwise"] == s["n"]
                     and sm["code_path_proof_30_rows_of_the_existing_benign_pool"]["ok"] and sm["code_path_proof_30_rows_of_the_existing_benign_pool"]["before_the_capture"]
                     and sm["code_path_proof_30_rows_of_the_existing_benign_pool"]["same_code_at_assembly"]
                     and sm["selection_deterministic"]["ok"] and sm["no_image_shared_with_the_caches"]["ok"] and sm["internal_consistency"]["ok"]
                     and sm["pre_edit_model_only"]["n_forwards_with_a_weight_other_than_W0_in_the_verification"] == 0 and glue_ok is not False)
    return {"ok": out["ok"], **{k: v for k, v in out.items() if k != "ok"}}


def readme(bb, st, vj, ex, summ) -> str:
    od = os.path.join(M.MME, bb)
    meta = M.jload(os.path.join(od, "meta.json"))
    C = M.cache_code()
    a, sm = st["all"], vj["summary"]
    cap = meta["capture"]
    L = []
    w = L.append
    w(f"MME WORKLOAD of backbone {bb} (edited module: input of layer {meta['layer']}, {meta['module'].split('input of ')[1]}, d = {meta['d_in']}): the out-of-benchmark")
    w("benign queries of 10_final/PROTOCOL.md, section 6, amendment \"Clarification of H3 (R6) and R17\", item (2) = R17, captured under the")
    w("PRE-EDIT model like the benign pool and split by image into MME-C (calibration source) and MME-T (test rows).")
    w("This directory is a capture and nothing else: no gate, no calibration and no edited model was run on these rows.")
    w("Written by cache2/mme/code/mme_finish.py; every number below is computed from the files of this directory.")
    w("Reader: ../mme_pool.py (ten-line example in its docstring; test ../test_mme_pool.py).")
    w("")
    w("CONSTRUCTION")
    w(f"  1. data       {M.SRC} (read-only; {len(ex['parquet_files'])} parquet files): {ex['n_rows']} rows = image + question + reference")
    w(f"                answer (Yes | No) + category; {ex['n_questions_ending_with_suffix']} of {ex['n_rows']} questions end with \"{M.SUFFIX}\".  {ex['n_images']} distinct images")
    w(f"                (content md5), {ex['n_categories']} categories, {ex['n_question_ids']} question ids (two rows each: one reference Yes, one No); "
      f"{ex['rows_per_image'].get('4', 0)} images are asked")
    w("                in two categories (4 rows).  Images -> ../images/<md5>.<jpg|png> (the benchmark's bytes, unchanged); row table")
    w("                ../mme_rows.jsonl (code/mme_extract.py).")
    w(f"  2. selection  ALL rows.  Split BY IMAGE (content md5), stratified by category, seed {M.SEED}: MME-C {summ['parts']['MME-C']['images']} images / "
      f"{summ['parts']['MME-C']['rows']} rows, MME-T {summ['parts']['MME-T']['images']}")
    w(f"                images / {summ['parts']['MME-T']['rows']} rows; the rows of an image are never separated ({summ['images_in_both_parts']} images in both parts).  Rule (code/mme_select.py):")
    for ln in summ["rule"].splitlines():
        w("                  " + ln.strip())
    w("                Deterministic: a second run in a fresh process reproduces selection.jsonl and both manifests byte for byte")
    w(f"                (../select/check_select.json: {sm['selection_deterministic']['ok']}).  Flags: see FLAGS below.")
    w("  3. capture    code/mme_capture.py calls 10_final/cache/code/fc_capture.main() -- the function that captured the benign pool -- on a")
    w("                manifest in which every MME row is a benign IMAGE row (kind 'benign', free image query, prompt format 'vqa') with")
    w("                decode = true.  Nothing about prompts, decoding, validity or keys is implemented in cache2/mme.  Per row, B = 1:")
    w("                  decoder v2 (tb_common.Prober.decode_v2): 10 greedy ids from full no-cache forwards, cut at the first newline =")
    w("                    y0_text; the same ids + KV-cache continuation to 48 tokens = long_text;")
    w("                  validity: y0 non-empty AND reproduced by the pre-edit model under teacher forcing (Prober.tf_state);")
    w("                  the key of the edited module at the last prompt position of the PROMPT-ONLY sequence, in the 11 full forwards of a")
    w("                    decode (Evaluator.decode_keys: prompt + k filler ids, k = 0..10; k = 0 is the prompt alone);")
    w("                  the key at the last prompt position of the TEACHER-FORCED sequence prompt + y0 (Evaluator.capture), and the")
    w("                    pre-edit flags of that sequence (Evaluator.tf).")
    w(f"                Model: rescope.load('{meta['families']['MME']['master_log']}') with the edited module at its PRE-EDIT weight W_0 for the whole job")
    w("                (Evaluator.applied(H.W0), asserted at the start of every part); keys do not depend on the edit log.")
    w(f"                {cap['n_parts']} resumable parts of {M.PART_SIZE} rows, {cap['gpu_seconds']:.0f} GPU-seconds, {', '.join(cap['gpus'])}, torch {', '.join(cap['torch'])},")
    w(f"                {cap['first']} .. {cap['last']} (clock of the machine).")
    p = sm["code_path_proof_30_rows_of_the_existing_benign_pool"]
    w(f"                BEFORE the run ({p['time']}): {p['n_rows']} rows of the existing benign pool were captured again through the same call and")
    w(f"                compared with 10_final/cache/{bb}/: prompt-only key {p['prompt_only_key_bitwise'][0]}/{p['prompt_only_key_bitwise'][1]}, decode-step keys "
      f"{p['decode_step_keys_bitwise'][0]}/{p['decode_step_keys_bitwise'][1]} rows, teacher-forced key")
    w(f"                {p['teacher_forced_key_bitwise'][0]}/{p['teacher_forced_key_bitwise'][1]} bit for bit; answers, ids, validity, positions equal on "
      f"{p['answers_ids_validity_positions_equal'][0]}/{p['answers_ids_validity_positions_equal'][1]} rows.")
    w("  4. assembly   code/mme_assemble.py: the bookkeeping of 10_final/cache/code/fc_assemble.py for benign rows (index fields, key rows,")
    w("                float32 images of the module-dtype keys, degenerate-answer flags, pre_correct_*).")
    w("  5. checks     code/mme_verify.py, test_mme_pool.py -> verify.json (VERIFICATION below).")
    w("")
    w("COUNTS")
    w(f"  {st['n_rows']} rows, {st['n_images']} images, {st['n_categories']} categories; teacher-forced key for {st['with_teacher_forced_key']} rows; "
      f"valid pre-edit answer: {a['valid']} of {a['rows']} rows ({pc(a['valid_percent'])} %).")
    w(T([(p_, b["rows"], b["images"], b["valid"], pc(b["valid_percent"]), b["yes"], b["no"], b["other"], b["ref_yes"], b["ref_no"], pc(b["equal_to_reference_percent"]))
         for p_, b in list(st["by_part"].items()) + [("all", a)]],
        head=("part", "rows", "images", "valid y0", "valid %", "y0 yes", "y0 no", "y0 other", "ref yes", "ref no", "y0 = ref %"), right=tuple(range(1, 11))).replace("\n", "\n  ").join(["  ", ""]))
    w("")
    w("  per category (valid = pre-edit answer reproduced by the pre-edit model; yes / no / other = the pre-edit answer after the testbed's")
    w("  normalisation IS the word yes / no, or anything else; y0 = ref: it equals the benchmark's reference answer):")
    w(T([(c, b["MME-C"]["images"], b["MME-C"]["rows"], b["MME-C"]["valid"], b["MME-T"]["images"], b["MME-T"]["rows"], b["MME-T"]["valid"], b["rows"], b["valid"], pc(b["valid_percent"]),
          b["yes"], b["no"], b["other"], pc(b["equal_to_reference_percent"])) for c, b in st["by_category"].items()],
        head=("category", "C img", "C rows", "C valid", "T img", "T rows", "T valid", "rows", "valid", "valid %", "yes", "no", "other", "y0 = ref %"),
        right=tuple(range(1, 14))).replace("\n", "\n  ").join(["  ", ""]))
    two = [k for k, v in summ["strata"].items() if "+" in k]
    simg = sum(b["MME-C"]["images"] + b["MME-T"]["images"] for b in st["by_category"].values())
    w(f"  An image that is asked in two categories counts in both rows of this table (sum of the image columns {simg}, images {st['n_images']}); the split keeps")
    w(f"  such an image whole, in its own stratum ({', '.join(two)}),")
    w("  so a category that shares images is not split exactly in half:")
    w("  " + "; ".join(f"{c} {b['MME-C']['images']} / {b['MME-T']['images']} images" for c, b in st["by_category"].items()
                       if abs(b["MME-C"]["images"] - b["MME-T"]["images"]) > 1) + " (MME-C / MME-T).")
    w("")
    w("PRE-EDIT ANSWERS (decoder v2, pre-edit model)")
    w(f"  yes {a['yes']}, no {a['no']}, other {a['other']} of {a['rows']} rows; among the valid rows: yes {a['valid_yes']}, no {a['valid_no']}, other {a['valid_other']}.")
    w("  answer strings:")
    for ln in wrap([f"{k!r} x {v}" for k, v in list(st["answer_strings"].items())[:12]] + (["..."] if len(st["answer_strings"]) > 12 else []), sep=", "):
        w(ln)
    if st["other_answers"]:
        w(f"  the {len(st['other_answers'])} answers that are not yes / no (first word: {st['other_by_first_word']}):")
        for o in st["other_answers"][:25]:
            w(f"    {o['sid']} [{o['part']}, {o['category']}, valid = {o['y0_valid']}] {o['y0_text']!r}")
        if len(st["other_answers"]) > 25:
            w(f"    ... ({len(st['other_answers']) - 25} more: rows.jsonl, y0_yn == 'other')")
    else:
        w("  every pre-edit answer is yes or no.")
    w(f"  answer length in scored tokens (teacher-forced sequence): {st['n_answer_tokens']};")
    w(f"  the 10-token decode stopped by: {st['y0_stop']}; the 48-token answer equals the 10-token answer on {st['long_answer_equals_short_answer']} of {a['rows']} rows")
    w(f"  (stop of the long decode: {st['long_stop']}).")
    if st["invalid_rows"]:
        w(f"  invalid rows ({len(st['invalid_rows'])}; reasons {st['invalid_reasons']}): kept in the files, to be excluded from D1 statistics:")
        for o in st["invalid_rows"][:20]:
            w(f"    {o['sid']} [{o['part']}, {o['category']}] y0 = {o['y0_text']!r}: answer token {o['pre_first_mismatch']} (counted from 0) of {o['pre_n_tok']} is not the argmax under")
            w(f"      teacher forcing; question {o['prompt'][:100]!r}")
        if len(st["invalid_rows"]) > 20:
            w(f"    ... ({len(st['invalid_rows']) - 20} more: rows.jsonl, y0_valid == false)")
    else:
        w("  no invalid row.")
    d = st["degenerate"]
    w(f"  degenerate-answer flags (fc_common.degenerate_flags): empty {d['deg_empty']}, one character {d['deg_single_char']}, '?' {d['deg_qmark']}, bare None / No {d['deg_none_no']}.")
    w(f"  NOTE  deg_none_no is true for {st['deg_none_no_among_answers_no'][0]} of the {st['deg_none_no_among_answers_no'][1]} answers \"No\": on this yes / no workload it marks a legitimate answer,")
    w(f"        not a degenerate one.  A read-out that sets deg_none_no rows apart (as for the VLKEB rows) would set apart {pc(100.0 * d['deg_none_no'] / a['rows'])} % of this pool.")
    n12 = sum(v for k, v in st["n_answer_tokens"].items() if k in ("1", "2"))
    w(f"  NOTE  {n12} of {a['rows']} answers are one or two tokens long: D1 on such a row is the flip of the first answer token(s); the 48-token")
    w(f"        answer differs from the 10-token answer on {a['rows'] - st['long_answer_equals_short_answer']} rows.")
    w("")
    w("PROMPTS AND KEYS")
    bp = st["benign_pool_image_rows"]
    w("  prompt format 'vqa' of the backbone (the format of the benign pool's image rows); question string as in the benchmark, including")
    w(f"  \"{M.SUFFIX}\".  Sequence lengths in tokens (min / 5 % / median / 95 % / max):")
    w(T([("MME prompt-only sequence", *st["len_prompt"].values()), ("  image tokens", *st["n_image_tokens"].values()), ("  other tokens", *st["n_text_tokens"].values()),
         (f"benign pool, {bp['n']} image rows", *bp["len_prompt"].values()), ("  image tokens", *bp["n_image_tokens"].values()), ("  other tokens", *bp["n_text_tokens"].values())],
        head=("", "min", "p05", "median", "p95", "max"), right=(1, 2, 3, 4, 5)).replace("\n", "\n  ").join(["  ", ""]))
    w(f"  {st['n_rows_longer_than_the_longest_benign_image_row']} MME rows are longer than the longest image row of the benign pool ({bp['len_prompt']['max']} tokens).")
    w(f"  last prompt position: teacher-forced index minus prompt-only index (lp_shift) = {st['lp_shift']} (0 = the same token on both paths);")
    w(f"  distinct keys among the 11 decode forwards of a row (the low-precision forward changes the last bits with the length): {st['n_decode_distinct']}.")
    ks = vj["cpu"].get("key_statistics")
    if ks:
        cz, nm = ks["cosine_between_two_keys"], ks["norm"]
        w(f"  keys (prompt-only, keys_decode.pt; descriptive, no gate and no edit log involved): {ks['n_distinct_keys_bitwise']} distinct keys for {ks['n_rows']} rows.")
        w("  Euclidean norm and raw cosine between two keys (min / 5 % / median / 95 % / max):")
        def f4(dct):
            return [f"{dct[k]:.4f}" for k in ("min", "p05", "median", "p95", "max")]
        w(T([("norm, MME", *f4(nm["mme"])),
             (f"norm, benign pool ({ks['n_benign_pool_image_rows']} image rows)", *f4(nm["benign_pool_image_rows"])),
             ("cosine, two MME rows (all pairs)", *f4(cz["mme_all_pairs"])),
             (f"cosine, two MME rows of one image ({cz['mme_same_image_pairs']['n']} pairs)", *f4(cz["mme_same_image_pairs"])),
             ("cosine, two benign-pool image rows", *f4(cz["benign_pool_image_rows_all_pairs"])),
             ("cosine, an MME row and a benign-pool image row", *f4(cz["mme_x_benign_pool_image_rows"]))],
            head=("", "min", "p05", "median", "p95", "max"), right=(1, 2, 3, 4, 5)).replace("\n", "\n  ").join(["  ", ""]))
        w(f"  cosine of the mean keys: MME-C and MME-T {ks['cosine_of_mme_c_mean_key_and_mme_t_mean_key']}; MME and the benign pool's image rows {ks['cosine_of_mme_mean_key_and_benign_image_mean_key']}.")
        a_, b_, c_ = cz["mme_all_pairs"]["median"], cz["benign_pool_image_rows_all_pairs"]["median"], cz["mme_x_benign_pool_image_rows"]["median"]
        rel = "closer to each other than" if a_ - b_ > 0.1 else ("farther from each other than" if b_ - a_ > 0.1 else "about as close to each other as")
        w(f"  NOTE  (medians above; wording by rule: a difference of more than 0.1 in the median cosine) two MME rows are")
        w(f"        {rel} two image rows of the benign pool ({a_} against {b_}); an MME row and a")
        w("        benign-pool row are " + ("less similar than rows within either pool" if c_ < min(a_, b_) else "not less similar than rows within the pools") + f" ({c_}).")
        w("        Every MME question ends with the same sentence.  MME-C and MME-T are exchangeable with each other by construction (seeded")
        w("        split by image); these numbers describe the keys only and say nothing about any gate.")
    w("")
    w("FILES (the file format of 10_final/cache/<backbone>/, described in FORMAT.txt there; rescope.keycache.KeyCache reads this directory)")
    w(f"  index.jsonl           {st['n_rows']} lines, one per row, benchmark order (= row order of keys_decode.pt); kind = 'benign', pool = 'benign',")
    w("                        family = 'MME', dataset = 'MME', sid = 'mme:<row number in the benchmark>', uid = 'mme_img:<image md5>'.")
    w(f"  keys_tf.pt            float32 {tuple(meta['keys_tf']['shape'])}: key at the last prompt position of the teacher-forced sequence prompt + y0; row = row_tf")
    w(f"                        ({meta['keys_tf']['n_main_rows']} rows + {meta['keys_tf']['n_shifted_extra_rows']} rows for the decode's token where lp_shift != 0, row_tf_prompt).")
    w(f"  keys_decode.pt        float32 {tuple(meta['keys_decode']['shape'])}: key at the last token of the prompt-only sequence; row = row_decode = line number.")
    w(f"  keys_decode_steps.pt  float32 {tuple(meta['keys_decode_steps']['shape'])}: the DISTINCT keys of the 11 full forwards of a decode; rows_decode_steps (11 row numbers).")
    w(f"                        All keys are float32 images of the module-dtype values ({meta['module_dtype']}): exact.")
    w("  rows.jsonl            the rows in the format of 10_final/cache/benign/<backbone>/rows.jsonl (below).")
    w("  meta.json             shapes, md5 sums, capture provenance (the reader checks index and key files against it).")
    w("  capture_parts.json    the capture parts.      verify.json   verification results.      README.txt   this file.")
    w("")
    f = st["fields"]
    w("ROW FIELDS (rows.jsonl)")
    w("  as in the benign pool: sid, ptype ('image'), uid (source item: here the IMAGE), prompt, image (absolute path), fmt ('vqa'), ref_answer")
    w("    (benchmark reference Yes | No; never used for keys), ref_field, benign_idx (row number), answer_subset (true for EVERY row: all rows")
    w("    have a decoder-v2 answer); y0_text, y0_ids, y0_gen_ids, y0_stop, y0_raw, y0_roundtrip_ids_equal, w0_reproduced, w0_n_ans_tok,")
    w("    w0_first_mismatch, y0_valid, y0_invalid_reason, y0_note, long_text, long_ids, long_stop, deg_empty, deg_single_char, deg_qmark,")
    w("    deg_none_no, pre_correct_at10 / pre_correct_long (the 10- / 48-token answer contains the reference), lp_tf, lp_decode, lp_shift,")
    w("    pre_exact, query_id, n_same_query, row_tf, row_decode; flags q_in_edit, q_in_edit_first_pos, q_in_edit_exact, img_in_edit,")
    w("    img_in_edit_first_pos, q_in_probe, q_in_probe_exact, img_in_probe, query_in_probe (against the backbone's VLKEB family) and `overlap`")
    w("    (the same flags per stream family of the backbone's caches).")
    w("  added: part (MME-C | MME-T), category, stratum (categories of the image, the unit of the stratified split), question_id, mme_idx,")
    w("    image_md5, img_in_cache, y0_yn (yes | no | other: the normalised answer is that word), y0_yn_first (by its first word), long_yn_first,")
    w("    ref_yn, pre_token_acc, pre_n_tok, len_prompt, len_tf, n_image_tokens, n_decode_distinct.")
    w(f"  fields of the benign pool's rows that do not apply and are absent: {', '.join(f['benign_pool_rows_fields_absent_here']) or '-'}")
    w("    (item_rank: seeded visiting order of the VLKEB items; img_entity_*: Freebase MID of a VLKEB image path).")
    w(f"  index.jsonl: the fields of the caches' index for a benign row (absent: {', '.join(f['cache_index_benign_fields_absent_here']) or '-'}) plus")
    w(f"    {', '.join(f['index_fields_not_in_the_cache_index'])}.")
    w("    master_log = the log whose handle carried the model (its PRE-EDIT weight); capture_family = that log's stream family.")
    w("")
    w("FLAGS (selection; expected and found: none)")
    fl = summ["flags"]
    w(f"  image content (md5) also an image of the project's caches: {fl['img_in_cache']['n_images']} of {summ['n_images']} images.  Compared with (distinct image contents):")
    for ln in wrap([f"{k} {v}" for k, v in fl["img_in_cache"]["reference_set_sizes(distinct image contents)"].items()]):
        w(ln)
    w("    (index:<dir> = every sequence of 10_final/cache/<dir>/index.jsonl: edits, image rephrases, probes, hard negatives, benign rows;")
    w("    hard_negatives / benign:<backbone> = the rows files, image files hashed again; md5_memo = every image the caches' selection read.)")
    rc = vj["cpu"].get("images_vs_recombination_rows") or {}
    if rc:
        w("    Also compared at verification time (cache2/recomb, recombination rows): "
          + "; ".join(f"{k}: {v.get('n_shared', '?')} shared of {v.get('n_image_contents', '?')} contents" for k, v in rc.items()) + ".")
    w(f"  question string (stripped, lower case) equal to the question or text rephrase of an edit: {fl['question']['q_in_edit']} rows; to a probe question: {fl['question']['q_in_probe']};")
    w(f"  also without the suffix \"{M.SUFFIX}\": {fl['question']['q_without_suffix_in_edit_or_probe']}; equal to a hard-negative or benign-pool question: {fl['question']['q_in_hn_or_benign_rows']}.")
    w(f"  {summ['question_strings_in_both_parts']} question strings occur in both parts (the benchmark asks the same question about several images): "
      f"{summ['rows_whose_question_string_also_occurs_in_the_other_part']['MME-C']} rows of MME-C and")
    w(f"  {summ['rows_whose_question_string_also_occurs_in_the_other_part']['MME-T']} rows of MME-T; the parts are disjoint by image, not by question string.")
    w("")
    w("VERIFICATION (verify.json; ok = " + str(vj["ok"]) + ")")
    g = sm["bitwise_recapture_of_50_random_mme_rows"]
    w(f"  bitwise re-capture   {g['n_rows']} seeded random rows captured again in a fresh process with the plain library calls, WITHOUT the vision-feature")
    w(f"                       memo, pre-edit weight: prompt-only key {g['prompt_only_key']['n_true']}/{g['prompt_only_key']['n']}, the 11 decode-step keys "
      f"{g['decode_step_keys(11 per row)']['n_true']}/{g['decode_step_keys(11 per row)']['n']} rows,")
    w(f"                       teacher-forced key {g['teacher_forced_key']['n_true']}/{g['teacher_forced_key']['n']} bit for bit; decoder-v2 answer, ids, 48-token answer, validity and pre-edit")
    w(f"                       flags reproduced on {g['n_rows_answer_and_validity_reproduced']}/{g['n_rows']} rows.  Rows outside the sample, [bitwise, n]:")
    w(f"                       {g['extra_rows(invalid answer / answer not yes-no / shifted position)']} (None: no such row exists).")
    ic = sm["internal_consistency"]
    w(f"  two forwards agree   teacher-forced key (prompt + answer) = decode-step key of the same sequence length (prompt + filler ids), two")
    w(f"                       separate forward passes of the capture: {ic['tf_key_vs_decode_step_of_same_length']['n_bitwise_equal']}/{ic['tf_key_vs_decode_step_of_same_length']['n_checked']} rows bit for bit.")
    w(f"  selection            reproduced byte for byte by a second run: {sm['selection_deterministic']['ok']}; images in both parts: {ic['n_images_in_both_parts']}.")
    w(f"  images               shared with the caches: {vj['cpu']['images']['n_images_shared_with_any_cache']} (image files hashed again; md5 sets of every cache index).")
    px = vj["cpu"].get("pixel_identity")
    if px:
        w(f"                       {px['n_image_files']} files decode to {px['n_distinct_decoded_rgb_contents']} distinct RGB contents; groups of files with identical pixels: "
          f"{px['n_groups_of_files_with_identical_pixels']} (in both parts: {px['n_such_groups_in_both_parts']}).")
    pw = sm["pre_edit_model_only"]
    w(f"  pre-edit model only  forwards of the edited module in the verification: {pw['n_forwards_of_the_edited_module_in_the_verification']}, of them with a weight other than W_0: "
      f"{pw['n_forwards_with_a_weight_other_than_W0_in_the_verification']}.")
    gl = sm["glue_mme_pool"]["gpu(pre-edit weight as the method)"]
    w(f"  glue (mme_pool.py)   {gl['n_items']} items found in this key cache by content: {gl['n_found_in_key_cache_by_content']}; Evaluator.tf_rows with the pre-edit weight as the method returns")
    w(f"                       the cached pre-edit outcome for {gl['n_tf_rows_equal_to_cached_pre_edit_outcome']} of them; CPU test (../test_mme_pool.json): {sm['glue_mme_pool']['cpu_test_ok']}.")
    w("")
    w("USE (R17) AND CAVEATS")
    w("  Read-outs registered by R17: calibrated on REPAIR, false firing and D1 on MME-T; calibrated on MME-C, false firing on")
    w("  heldout_disjoint, BEN-T and MME-T.  \"The bound of the calibration theorem is claimed only where calibration and test queries are")
    w("  exchangeable; the other cells measure what a shift of the workload costs, for every gate alike.  MME images may show entities")
    w("  that an edit concerns; such queries are not benign, cannot be identified from the data, and are part of the measured rate.\"")
    w("  * keys: which = 'decode' (prompt-only) and 'steps' exist for every row; 'tf' for the rows with a non-empty answer.  D1 needs y0_valid.")
    w("  * every row has kind = 'benign' and pool = 'benign', like the benign pool: after a concatenation with the backbone's cache, tell the")
    w("    rows apart by family == 'MME' (or the sid prefix 'mme:').  query_id starts at " + str(M.QUERY_ID0) + " (no collision with the caches' ids).")
    w("  * the unit of resampling is the image (uid); MME-C and MME-T are the same for both backbones.")
    w("  * image paths are absolute (../images/); the keys were captured from these files.")
    return "\n".join(L) + "\n"


def top_readme(stats_, ex, summ, vjs) -> str:
    L = []
    w = L.append
    w("10_final/cache2/mme -- the MME benchmark as an out-of-benchmark benign workload (PROTOCOL.md section 6, amendment \"Clarification of H3 (R6)")
    w("and R17\", item (2)): captured under the PRE-EDIT model of both backbones like the benign pool, split by image into MME-C and MME-T.")
    w("Capture only: no gate, no calibration and no edited model was run on these rows.  Written by code/mme_finish.py.")
    w("")
    w("  mme_pool.py           the reader (glue): mme_pool.load(backbone) -> rescope KeyCache + rows of a part + library items; ten-line example in")
    w("                        its docstring.  test_mme_pool.py / test_mme_pool.json: its test.")
    w("  <backbone>/           qwen25vl (layer 20), llava (layer 16): index.jsonl, keys_tf.pt, keys_decode.pt, keys_decode_steps.pt, rows.jsonl, meta.json,")
    w("                        capture_parts.json, README.txt (construction, counts, format, validity and answers per category), verify.json")
    w(f"  images/               {ex['n_images']} image files <content md5>.<jpg|png> (the benchmark's bytes); mme_rows.jsonl: the {ex['n_rows']} benchmark rows; extract_summary.json")
    w("  select/               selection.jsonl (rows, parts, flags), manifest_<backbone>.jsonl (what the capture read), select_summary.json, check_select.json")
    w("  summary.json          the counts below as JSON")
    w("  code/                 mme_extract.py -> mme_select.py -> mme_capture.py (calls 10_final/cache/code/fc_capture.main) -> mme_assemble.py ->")
    w("                        mme_verify.py -> mme_finish.py; run_bb.sh (the GPU chain of a backbone, through the queue); mme_precheck.py (CPU dry run")
    w("                        of the prompt construction); mme_common.py")
    w("  logs/, work/          logs; work/capture/<backbone>/parts (the raw capture parts), work/proof (the proof on the existing benign pool),")
    w("                        work/verify (verification results), work/select_recheck and work/extract_recheck (second runs), work/test_pool")
    w("                        (the synthetic copy of the test), work/hcache (private copy of the harness' cache directory)")
    w("")
    w(f"{ex['n_rows']} rows, {ex['n_images']} images, {ex['n_categories']} categories; split by image, stratified by category, seed {M.SEED}: "
      + "; ".join(f"{p} {summ['parts'][p]['images']} images / {summ['parts'][p]['rows']} rows" for p in M.PARTS) + ".")
    w("")
    rows = []
    for bb, st in stats_.items():
        for p, b in list(st["by_part"].items()) + [("all", st["all"])]:
            rows.append((bb, p, b["rows"], b["images"], b["valid"], pc(b["valid_percent"]), b["yes"], b["no"], b["other"], pc(b["equal_to_reference_percent"])))
    w(T(rows, head=("backbone", "part", "rows", "images", "valid y0", "valid %", "y0 yes", "y0 no", "y0 other", "y0 = ref %"), right=tuple(range(2, 10))))
    w("")
    w("valid y0 = decoder-v2 pre-edit answer reproduced by the pre-edit model under teacher forcing; yes / no / other = the pre-edit answer; y0 = ref =")
    w("it equals the benchmark's reference answer.  Verification (per backbone, <backbone>/verify.json): "
      + "; ".join(f"{bb} ok = {v['ok']}" for bb, v in vjs.items()) + ".")
    return "\n".join(L) + "\n"


def main():
    M.cache_code()
    bbs = sys.argv[1:] or list(M.BACKBONES)
    ex = M.jload(os.path.join(M.MME, "extract_summary.json"))
    summ = M.jload(os.path.join(M.SELECT, "select_summary.json"))
    stats_, vjs = {}, {}
    for bb in bbs:
        st = stats(bb)
        vj = verify_json(bb, st)
        od = os.path.join(M.MME, bb)
        M.jdump(vj, os.path.join(od, "verify.json"))
        M.write_atomic(os.path.join(od, "README.txt"), readme(bb, st, vj, ex, summ))
        for f in ("README.txt.previous", "verify.json.previous"):
            if os.path.exists(os.path.join(od, f)):
                os.remove(os.path.join(od, f))
        stats_[bb], vjs[bb] = st, vj
        print(f"{bb}: verify.json ok = {vj['ok']}; README.txt written ({st['n_rows']} rows; valid {st['all']['valid']}; yes / no / other {st['all']['yes']} / {st['all']['no']} / {st['all']['other']})")
    M.jdump({"what": "counts of the MME capture per backbone (written by code/mme_finish.py)", "n_rows": ex["n_rows"], "n_images": ex["n_images"],
             "n_categories": ex["n_categories"], "seed": M.SEED, "selection": {"parts": summ["parts"], "by_category": summ["by_category"]},
             "backbones": {bb: {k: v for k, v in st.items() if k != "fields"} for bb, st in stats_.items()},
             "verify_ok": {bb: v["ok"] for bb, v in vjs.items()}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(M.MME, "summary.json"))
    M.write_atomic(os.path.join(M.MME, "README.txt"), top_readme(stats_, ex, summ, vjs))
    return 0 if all(v["ok"] for v in vjs.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
