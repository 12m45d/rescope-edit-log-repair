"""Step 4b: verification of the assembled MME capture of one backbone.

    python -I -B mme_verify.py <qwen25vl | llava> gpu [--n 50]     fresh capture of a seeded sample (through the GPU queue: run_bb.sh)
    python -I -B mme_verify.py <backbone> cpu                      model-free checks of the published files

gpu   A seeded uniform sample of n rows of <backbone>/index.jsonl (plus, reported separately, up to 10 rows each of: invalid
      pre-edit answer, answer that is not yes / no, shifted last prompt position, if such rows exist outside the sample) is
      captured AGAIN in a fresh process with the PLAIN library calls -- rescope.Evaluator.decode_keys / capture / lp_shift /
      eligible_keys(answer='all') / tf and tb_common.Prober.decode_v2 / tf_state -- WITHOUT the harness' vision-feature memo
      (the capture used it), and compared with the published files: keys bit for bit (keys_decode.pt, the 11 rows of
      keys_decode_steps.pt, keys_tf.pt), positions, the decoder-v2 answer, its ids, the 48-token answer, validity, pre-edit flags.
      Model state: the PRE-EDIT weight W_0 for every forward (Evaluator.applied(H.W0), asserted).  [10_final/cache/code/
      fc_verify.py leaves its handle at the edited weight; here no edited model is run on MME rows: a hook on the edited module
      counts its forwards and those with a weight other than W_0, which must be 0.]  The gate object that eligible_keys asks for
      is a stub that only carries role = 'last': no gate is built or evaluated.  Last, the items of the glue (mme_pool.py) are
      looked up in the published key cache by content and run through Evaluator.tf_rows with the pre-edit weight as the method.
      -> work/verify/<backbone>_gpu.json
cpu   (1) internal consistency of index and key files (as fc_verify.py cpu, part 4): positions, step 0 = prompt-only key, finite,
      no zero row, row numbering, and the identity "teacher-forced key = decode-step key of the same sequence length" (two
      independent captures of the same thing; bitwise, where lp_shift = 0 and the answer has at most 10 tokens);
      (2) the library's reader rescope.keycache.KeyCache on the directory (md5 of every key file, same tensors);
      (3) the rows against the selection (order, parts, no image in two parts) and against the CPU dry run of the prompt
      construction (mme_precheck.py: sequence length, image tokens, last prompt index);
      (4) images: every image file hashed again; none is an image of the project's caches (md5 sets of the caches' indexes);
      reported in addition: images shared with the recombination rows of cache2/recomb (a build in progress);
      (5) descriptive statistics of the prompt-only keys (norms, raw cosines; no gate, no edit log): one distinct key per query;
      (6) no picture is stored under two md5 sums with identical decoded pixels in different parts.
      -> work/verify/<backbone>_cpu.json
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402

KEYS = ("decode_key_equal", "decode_steps_equal", "lp_decode_equal", "tf_key_equal", "lp_tf_equal", "lp_shift_equal", "tf_prompt_key_equal",
        "eligible_all_equal")
ANSWER = ("y0_text_equal", "y0_gen_ids_equal", "long_text_equal", "long_ids_equal", "y0_valid_equal", "pre_flags_equal")
GLUE = ("glue_item_equals_capture_item", "glue_item_tf_row_equal")      # the item of mme_pool.py through the library under the pre-edit model


def out_path(bb, mode):
    return os.path.join(M.WORK, "verify", f"{bb}_{mode}{'_trial' if ROOT != M.MME else ''}.json")


ROOT = M.MME                                      # --trial: work/trial (an assembly made while the capture runs)


def load_published(bb):
    import torch
    od = os.path.join(ROOT, bb)
    index = M.jsonl_read(os.path.join(od, "index.jsonl"))
    rows = M.jsonl_read(os.path.join(od, "rows.jsonl"))
    meta = M.jload(os.path.join(od, "meta.json"))
    K = {n: torch.load(os.path.join(od, f), map_location="cpu") for n, f in (("tf", "keys_tf.pt"), ("decode", "keys_decode.pt"), ("steps", "keys_decode_steps.pt"))}
    return od, index, rows, meta, K


def identity(od, meta):
    """What a verification result belongs to: the md5 sums of the files it compared with."""
    return {"index_md5": M.md5_file(os.path.join(od, "index.jsonl")), "rows_md5": M.md5_file(os.path.join(od, "rows.jsonl")),
            "keys_tf_md5": meta["keys_tf"]["md5"], "keys_decode_md5": meta["keys_decode"]["md5"], "keys_decode_steps_md5": meta["keys_decode_steps"]["md5"]}


# ============================================================================= gpu
def run_gpu(bb, n):
    import numpy as np
    import torch
    C = M.cache_code()
    import fc_capture as CAP
    if not os.environ.get("GQ_MEM"):
        raise SystemExit("GPU steps run only through the queue (gq run sets GQ_MEM): see run_bb.sh")
    t0 = time.time()
    od, index, rows, meta, K = load_published(bb)
    rowof = {r["sid"]: r for r in rows}
    N = len(index)
    pick = np.sort(np.random.default_rng(M.useed("verify", bb)).choice(N, min(n, N), replace=False))
    sample = [index[int(i)] for i in pick]
    got = {e["sid"] for e in sample}
    extra = {"invalid_y0": [e for e in index if not e["y0_valid"] and e["sid"] not in got][:10],
             "answer_not_yes_no": [e for e in index if e["y0_yn"] == "other" and e["y0_valid"] and e["sid"] not in got][:10],
             "shifted_position": [e for e in index if e["lp_shift"] not in (0, None) and e["sid"] not in got][:10]}
    C.tb()
    if M.MME not in sys.path:
        sys.path.insert(0, M.MME)
    import mme_pool as POOL                                             # the glue: its item form is run through the library below
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20                  # the queue's cap (gpuq/README.txt)
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    rs = C.import_rescope()
    master = meta["families"]["MME"]["master_log"]
    H = rs.load(master, device="cuda:0", mem_gb=cap / 2 ** 30)
    assert H.backbone == bb and H.L == meta["layer"], (H.backbone, H.L)
    ev = rs.Evaluator(H, hard_negatives=False)                          # the plain library object (image rows need no subclass)
    S = types.SimpleNamespace(role="last")                              # eligible_keys only reads the role: no gate exists in this process
    M.say(f"{bb} verify gpu: loaded {master} in {time.time() - t0:.0f} s; {len(sample)} sampled rows, extra {dict((k, len(v)) for k, v in extra.items())}")

    def compare(e):
        r = rowof[e["sid"]]
        it = CAP.make_item(dict(e, lib_kind="benign"))                 # as fc_verify.item_of: the library item of an index line
        out = {"sid": e["sid"], "part": e["part"], "category": e["category"], "y0_text": e["y0_text"], "y0_valid": e["y0_valid"], "lp_shift": e["lp_shift"]}
        st = K["steps"][torch.tensor(e["rows_decode_steps"], dtype=torch.long)]
        dk = ev.decode_keys(it).float().cpu()
        out["decode_steps_equal"] = bool(torch.equal(dk, st))
        out["decode_steps_n_rows_differing"] = int((dk != st).any(1).sum())
        x, p = ev.capture(it, False)
        out["decode_key_equal"] = bool(torch.equal(x[p].float().cpu(), K["decode"][e["row_decode"]]))
        out["lp_decode_equal"] = int(p) == e["lp_decode"] and int(x.shape[0]) == e["len_prompt"]
        if e["row_tf"] is not None:
            x, p = ev.capture(it, True)
            out["tf_key_equal"] = bool(torch.equal(x[p].float().cpu(), K["tf"][e["row_tf"]]))
            out["tf_max_abs_diff"] = float((x[p].float().cpu() - K["tf"][e["row_tf"]]).abs().max())
            out["lp_tf_equal"] = int(p) == e["lp_tf"] and int(x.shape[0]) == e["len_tf"]
            sh = ev.lp_shift(it)
            out["lp_shift_equal"] = sh == e["lp_shift"]
            if e["row_tf_prompt"] is not None:
                x2, p2 = ev.capture(it, True, lp="prompt")
                out["tf_prompt_key_equal"] = bool(torch.equal(x2[p2].float().cpu(), K["tf"][e["row_tf_prompt"]])) and int(p2) == e["lp_tf"] - e["lp_shift"]
            allk = ev.eligible_keys(it, S, answer="all").float().cpu()
            mine = torch.cat([K["tf"][e["row_tf"]][None]] + ([K["tf"][e["row_tf_prompt"]][None]] if e["row_tf_prompt"] is not None else []) + [st])
            out["eligible_all_equal"] = bool(allk.shape == mine.shape and torch.equal(allk, mine))
            t = ev.tf(it)                                               # pre-edit flags of the teacher-forced answer (the model is at W_0)
            out["pre_flags_equal"] = (bool(t["exact"]) == e["pre_exact"] and float(t["token_acc"]) == e["pre_token_acc"] and int(t["n_tok"]) == e["pre_n_tok"]
                                      and bool(t["finite"]) == e["pre_finite"] and t["first_mismatch"] == e["pre_first_mismatch"])
        q, ptype, _ = ev._query(it)                                     # the decoder again: tb_common.Prober.decode_v2 + tf_state
        y0, lt, info = H.prober.decode_v2(q, ptype)
        out["y0_text_equal"] = y0 == e["y0_text"]
        out["y0_gen_ids_equal"] = info["gen_ids"] == r["y0_gen_ids"] and info["stop"] == r["y0_stop"]
        out["long_text_equal"] = lt == r["long_text"]
        out["long_ids_equal"] = info["long_ids"] == r["long_ids"] and info["long_stop"] == r["long_stop"]
        try:
            ok, _, _ = H.prober.tf_state(q, ptype, y0)
        except RuntimeError:
            ok = None
        out["y0_valid_equal"] = (bool(y0) and ok is True) == e["y0_valid"]
        if e["row_tf"] is not None:                                     # the glue's item (mme_pool.item_of): same query, and its teacher-forced
            it2 = POOL.item_of(e)                                       # row under the pre-edit model carries the cached pre-edit outcome
            out["glue_item_equals_capture_item"] = all(it2["_row"][k] == v for k, v in it["_row"].items()) and it2["kind"] == it["kind"] and it2["sid"] == it["sid"]
            t2 = ev.tf(it2)
            out["glue_item_tf_row_equal"] = (bool(t2["exact"]) == e["pre_exact"] and t2["D1"] == (not e["pre_exact"]) and float(t2["token_acc"]) == e["pre_token_acc"]
                                             and t2["tok_loc"] == (e["pre_token_acc"] if e["y0_valid"] else None) and t2["fired"] is None)
        out["all_equal"] = all(out[k] for k in KEYS + ANSWER + GLUE if k in out)
        return out

    guard = {"calls": 0, "not_w0": 0}

    def _guard(mod, args):                                              # every forward of the edited module in this process: is its weight W_0?
        guard["calls"] += 1
        if not torch.equal(mod.weight.data, H.W0):
            guard["not_w0"] += 1
    hg = H.mod.register_forward_pre_hook(_guard)

    res_rows, res_extra = [], {k: [] for k in extra}
    with torch.no_grad(), ev.applied(H.W0):                             # the PRE-EDIT model; no vision-feature memo
        assert torch.equal(H.mod.weight.data, H.W0)
        assert "get_image_features" not in H.vlm.m.model.__dict__      # the harness memo is not installed
        for i, e in enumerate(sample):
            res_rows.append(compare(e))
            if (i + 1) % 10 == 0:
                M.say(f"  {i + 1}/{len(sample)} rows, all equal so far: {all(r['all_equal'] for r in res_rows)} ({time.time() - t0:.0f} s)")
        for k, sub in extra.items():
            for e in sub:
                res_extra[k].append(compare(e))
        assert torch.equal(H.mod.weight.data, H.W0)
    n_calls_main = guard["calls"]

    # the glue (mme_pool.py) through the library's public row functions, pre-edit model only: the items are found in the key
    # cache by content (the lookup of rescope.SparseEvaluator; no forward), and Evaluator.tf_rows with the method "replacement
    # weight W_0" returns for every item the cached pre-edit outcome (D1 = the pre-edit answer is not reproduced).  [These
    # forwards use the harness' vision-feature memo, as every row function of the library does.]
    pos = {e["sid"]: j for j, e in enumerate(index)}
    tfs = [e for e in sample + [x for v in extra.values() for x in v] if e["row_tf"] is not None]
    P = POOL.load(bb)
    evs = rs.SparseEvaluator(H, cache=P.kc, hard_negatives=False)
    items = P.items([pos[e["sid"]] for e in tfs])
    found = [evs.cache_row(it) for it in items]
    w0_rows = evs.tf_rows(H.W0, items)
    glue = {"n_items": len(items), "n_found_in_key_cache_by_content": sum(f == pos[e["sid"]] for f, e in zip(found, tfs)),
            "n_tf_rows_equal_to_cached_pre_edit_outcome": sum(bool(r["exact"]) == e["pre_exact"] and r["D1"] == (not e["pre_exact"])
                                                              and float(r["token_acc"]) == e["pre_token_acc"] and r["fired"] is None
                                                              and r["sid"] == e["sid"] and r["part"] == e["part"] for r, e in zip(w0_rows, tfs)),
            "n_D1_false_on_valid_rows": sum((not r["D1"]) for r, e in zip(w0_rows, tfs) if e["y0_valid"]), "n_valid_rows": sum(bool(e["y0_valid"]) for e in tfs),
            "method": "torch.Tensor H.W0 (the pre-edit weight as a replacement weight); no gate"}
    gen = evs.generate(H.W0, items[:20])                                # free generation of the same items under the pre-edit model
    glue["generate_under_W0"] = {"n": len(gen), "n_agree_with_cached_pre_edit_answer": sum(bool(g["agree"]) for g in gen),
                                 "n_long_answer_equal": sum(bool(g["long_eq_pre"]) for g in gen), "n_leak": sum(bool(g["leak"]) for g in gen)}
    glue["ok"] = (glue["n_found_in_key_cache_by_content"] == len(items) and glue["n_tf_rows_equal_to_cached_pre_edit_outcome"] == len(items)
                  and glue["generate_under_W0"]["n_agree_with_cached_pre_edit_answer"] == len(gen) and glue["generate_under_W0"]["n_long_answer_equal"] == len(gen)
                  and glue["generate_under_W0"]["n_leak"] == 0)
    hg.remove()
    weight = {"n_forwards_of_the_edited_module": guard["calls"], "n_in_the_bitwise_comparison": n_calls_main,
              "n_forwards_with_a_weight_other_than_W0": guard["not_w0"]}

    def summarise(rs_):
        s = {"n": len(rs_), "by_part": dict(collections.Counter(r["part"] for r in rs_)), "n_valid_y0": sum(bool(r["y0_valid"]) for r in rs_)}
        for k in KEYS + ANSWER + GLUE:
            v = [r[k] for r in rs_ if k in r]
            s[k] = {"n": len(v), "n_true": int(sum(v))}
        s["all_bitwise"] = all(r[k] for r in rs_ for k in KEYS if k in r)
        s["n_rows_all_keys_bitwise"] = sum(all(r[k] for k in KEYS if k in r) for r in rs_)
        s["n_rows_answer_and_validity_reproduced"] = sum(all(r[k] for k in ANSWER if k in r) for r in rs_)
        s["n_rows_glue_item_ok"] = sum(all(r[k] for k in GLUE if k in r) for r in rs_)
        s["failures"] = [r for r in rs_ if not r["all_equal"]][:20]
        return s

    res = {"backbone": bb, "mode": "gpu",
           "ok": all(r["all_equal"] for r in res_rows) and all(r["all_equal"] for v in res_extra.values() for r in v) and glue["ok"] and guard["not_w0"] == 0,
           "glue_mme_pool": glue, "edited_module_weight": weight,
           "what": "fresh process; plain library calls (rescope.Evaluator, tb_common.Prober); no vision-feature memo; pre-edit weight W_0; compared with the "
                   "published key files bit for bit (float32 images) and with index.jsonl / rows.jsonl",
           "seed": [M.SEED, "verify", bb], "n_candidates": N, "sample": summarise(res_rows),
           "extra": {k: (summarise(v) if v else None) for k, v in res_extra.items()},
           "n_extra_available": {k: len(v) for k, v in extra.items()},
           "handle": {"log": master, "T": int(H.T), "model_weight": "W_0 (pre-edit) for every forward", "vision_memo": False,
                      "gate_for_eligible_keys": "stub with role 'last' (no gate built)"},
           "files": identity(od, meta), "gpu": torch.cuda.get_device_name(0), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": str(torch.__version__),
           "capture_gpus": meta["capture"]["gpus"], "capture_cuda_visible_devices": meta["capture"]["cuda_visible_devices"], "pid": os.getpid(),
           "seconds": round(time.time() - t0, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": res_rows, "rows_extra": res_extra}
    M.jdump(res, out_path(bb, "gpu"), default=str)
    M.say(f"VERIFY {bb}: ok = {res['ok']}; sample: keys bitwise {res['sample']['n_rows_all_keys_bitwise']}/{len(res_rows)}, answers and validity reproduced "
          f"{res['sample']['n_rows_answer_and_validity_reproduced']}/{len(res_rows)}; extra rows equal "
          f"{sum(r['all_equal'] for v in res_extra.values() for r in v)}/{sum(len(v) for v in res_extra.values())}; glue {glue}; weight {weight}")
    return 0 if res["ok"] else 2


# ============================================================================= cpu
def run_cpu(bb):
    import torch
    C = M.cache_code()
    torch.set_num_threads(4)
    t0 = time.time()
    od, index, rows, meta, K = load_published(bb)
    K_tf, K_dec, K_st = K["tf"], K["decode"], K["steps"]
    res = {"backbone": bb, "mode": "cpu", "files": identity(od, meta)}
    # (1) internal consistency
    with_tf = [e for e in index if e["row_tf"] is not None]
    same_len = [(e, e["len_tf"] - e["len_prompt"]) for e in with_tf if e["lp_shift"] == 0]
    chk = [(e, n_) for e, n_ in same_len if 0 < n_ <= C.N_DECODE - 1]
    eq_n = [bool(torch.equal(K_tf[e["row_tf"]], K_st[e["rows_decode_steps"][n_]])) for e, n_ in chk]
    res["consistency"] = {
        "n_index": len(index), "sids_unique": len({e["sid"] for e in index}) == len(index),
        "row_decode_is_line_number": all(e["row_decode"] == j for j, e in enumerate(index)),
        "row_tf_unique_and_dense": sorted(e["row_tf"] for e in with_tf) == list(range(len(with_tf))),
        "n_position_identity_violations(lp_tf - lp_decode != lp_shift)": sum(1 for e in with_tf if e["lp_tf"] - e["lp_decode"] != e["lp_shift"]),
        "n_lp_shift_none_with_tf": sum(1 for e in with_tf if e["lp_shift"] is None),
        "lp_decode_is_last_position_of_the_prompt": all(e["lp_decode"] == e["len_prompt"] - 1 for e in index),
        "decode_step0_equals_keys_decode": bool(torch.equal(K_st[[e["rows_decode_steps"][0] for e in index]], K_dec)),
        "decode_step_rows_cover_keys_decode_steps": sorted({x for e in index for x in e["rows_decode_steps"]}) == list(range(K_st.shape[0])),
        "all_finite": bool(torch.isfinite(K_tf).all() and torch.isfinite(K_dec).all() and torch.isfinite(K_st).all()),
        "n_zero_rows": int((K_tf.abs().sum(1) == 0).sum() + (K_dec.abs().sum(1) == 0).sum() + (K_st.abs().sum(1) == 0).sum()),
        "keys_are_images_of_the_module_dtype": bool(all(torch.equal(k, k.to(getattr(torch, meta["module_dtype"].split(".")[1])).float()) for k in (K_tf, K_dec, K_st))),
        "n_rows_without_teacher_forced_key": len(index) - len(with_tf),
        "n_answers_without_teacher_forced_sequence": sum(1 for e in index if e["answer"] and e["row_tf"] is None),
        "n_empty_answers": sum(1 for e in index if not e["answer"]),
        "n_teacher_forced_sequences_with_nonfinite_pre_edit_logits": sum(1 for e in with_tf if e["pre_finite"] is False),
        "n_pre_exact_differs_from_y0_valid": sum(1 for e in with_tf if bool(e["pre_exact"]) != bool(e["y0_valid"])),
        "n_valid_without_teacher_forced_key": sum(1 for e in index if e["y0_valid"] and e["row_tf"] is None),
        "tf_key_vs_decode_step_of_same_length": {"n_checked": len(chk), "n_bitwise_equal": int(sum(eq_n)),
                                                 "n_longer_than_10_tokens(not checkable)": len(same_len) - len(chk),
                                                 "failures": [e["sid"] for (e, _), q in zip(chk, eq_n) if not q][:20]},
        "tf_key_equals_prompt_only_key(same token, other length)": {"n": len(same_len), "n_bitwise_equal": int(sum(bool(torch.equal(K_tf[e["row_tf"]], K_dec[e["row_decode"]])) for e, _ in same_len))},
        "distinct_decode_keys_per_row": {str(k): v for k, v in sorted(collections.Counter(e["n_decode_distinct"] for e in index).items())},
        "duplicate_queries": len(index) - len({e["query_id"] for e in index}),
    }
    # (2) the library's reader
    try:
        rs = C.import_rescope()
        from rescope import keycache as KC
        kc = KC.KeyCache(od, verify=True)                               # md5 of every key file against meta.json
        allr = list(range(len(index)))
        tfr = [j for j, e in enumerate(index) if e["row_tf"] is not None]
        e0 = index[tfr[0]]
        res["library_reader"] = {
            "ok": True, "summary": kc.summary(), "n_equal": kc.n == len(index), "d": kc.d, "backbone": kc.backbone, "layer": kc.layer,
            "md5_of_key_files_checked": True,
            "n_rows_kind_benign": len(kc.rows(kind="benign")), "n_rows_pool_benign": len(kc.rows(pool="benign")),
            "n_rows_benign_valid": len(kc.rows(kind="benign", valid=True)),
            "keys_decode_equal": bool(torch.equal(kc.keys("decode", allr), K_dec)),
            "keys_tf_equal": bool(torch.equal(kc.keys("tf", tfr), K_tf[[index[j]["row_tf"] for j in tfr]])),
            "keys_steps_equal": bool(torch.equal(kc.keys("steps", allr[:64]), K_st[torch.tensor([index[j]["rows_decode_steps"] for j in allr[:64]])])),
            "having": {w: len(kc.having(w)) for w in ("tf", "decode", "steps", "all")},
            "find_by_content": kc.find(e0["fmt"], e0["prompt"], e0["image"], e0["answer"]) == tfr[0],
            "by_sid": all(kc.by_sid(e["sid"]) == j for j, e in enumerate(index)),
            "family_of_a_log_is_none": kc.family_of(["x"]) is None, "is_final": kc.is_final}
        del rs
    except Exception as ex:
        res["library_reader"] = {"ok": False, "error": f"{type(ex).__name__}: {ex}"[:500]}
    # (3) selection and CPU dry run of the prompt construction
    sel = M.jsonl_read(os.path.join(M.SELECT, "selection.jsonl"))
    img_parts = collections.defaultdict(set)
    for e in index:
        img_parts[e["image_md5"]].add(e["part"])
    pre_p = os.path.join(M.WORK, f"precheck_{bb}.json")
    pre = {r["sid"]: r for r in M.jload(pre_p)["rows"]} if os.path.exists(pre_p) else None
    res["selection"] = {
        "rows_in_selection_order": [e["sid"] for e in index] == [s["sid"] for s in sel] == [r["sid"] for r in rows],
        "parts_equal": all(e["part"] == s["part"] and r["part"] == s["part"] for e, s, r in zip(index, sel, rows)),
        "prompts_images_equal": all(e["prompt"] == s["prompt"] and e["image"] == s["image"] and e["image_md5"] == s["image_md5"] for e, s in zip(index, sel)),
        "n_images": len(img_parts), "n_images_in_both_parts": sum(len(v) > 1 for v in img_parts.values()),
        "by_part": {p: {"rows": sum(e["part"] == p for e in index), "images": len({e["image_md5"] for e in index if e["part"] == p})} for p in M.PARTS},
        "selection_md5_equals_meta": M.md5_file(os.path.join(M.SELECT, "selection.jsonl")) == meta["selection"]["selection.jsonl"],
        "cpu_dry_run_of_prompt_construction": None if pre is None else {
            "n": len(pre), "len_prompt_equal": sum(pre[e["sid"]]["len_prompt"] == e["len_prompt"] for e in index),
            "n_image_tokens_equal": sum(pre[e["sid"]]["n_image_tokens"] == e["n_image_tokens"] for e in index),
            "lp_decode_equal": sum(pre[e["sid"]]["lp_decode"] == e["lp_decode"] for e in index)}}
    # (4) images
    files = {e["image"] for e in index}
    h = {p: M.md5_file(p) for p in sorted(files)}
    ref = {}
    for dname in sorted(dn for dn in os.listdir(M.CACHE) if os.path.exists(os.path.join(M.CACHE, dn, "index.jsonl"))):
        ref[dname] = {x.get("image_md5") for x in M.jsonl_read(os.path.join(M.CACHE, dname, "index.jsonl")) if x.get("image_md5")}
    mine = set(h.values())
    recomb = {}
    for b2 in M.BACKBONES:                                              # recombination rows (cache2/recomb; a build in progress: reported, not part of ok)
        rp = os.path.join(M.CACHE2, "recomb", b2, "rows.jsonl")
        if os.path.exists(rp):
            try:
                rm = {x.get("image_md5") for x in M.jsonl_read(rp) if x.get("image_md5")}
                recomb[b2] = {"n_image_contents": len(rm), "n_shared": len(mine & rm), "rows_md5": M.md5_file(rp)}
            except ValueError as ex_:                                   # the file is being rewritten
                recomb[b2] = {"error": f"{type(ex_).__name__}"}
    res["images_vs_recombination_rows"] = recomb
    res["images"] = {"n_files": len(h), "n_files_whose_name_is_their_md5": sum(os.path.basename(p).split(".")[0] == m for p, m in h.items()),
                     "n_index_md5_equal_to_file_md5": sum(h[e["image"]] == e["image_md5"] for e in index),
                     "n_images_shared_with_cache_index": {dname: len(mine & s) for dname, s in ref.items()},
                     "n_images_shared_with_any_cache": len(mine & set().union(*ref.values())),
                     "n_rows_flagged_img_in_cache": sum(bool(e["img_in_cache"]) for e in index),
                     "selection_summary_img_in_cache": M.jload(os.path.join(M.SELECT, "select_summary.json"))["flags"]["img_in_cache"]["by_reference_set"]}
    # (5) descriptive statistics of the prompt-only keys (NO gate, NO edit log: norms and raw cosines of the keys themselves), next to
    #     the image rows of the backbone's benign pool.  They show that the capture holds one distinct key per query of the same
    #     module, and how concentrated the keys of this workload are (every question ends with the same sentence).
    def _q(v):
        v = v.double().flatten()
        qs = torch.quantile(v, torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64)) if v.numel() <= 16_000_000 else None
        if qs is None:                                                   # torch.quantile has an input-size limit: a seeded subsample
            g_ = torch.Generator().manual_seed(M.useed("keystats"))
            qs = torch.quantile(v[torch.randint(0, v.numel(), (4_000_000,), generator=g_)], torch.tensor([0.05, 0.5, 0.95], dtype=torch.float64))
        return {"min": round(float(v.min()), 4), "p05": round(float(qs[0]), 4), "median": round(float(qs[1]), 4), "p95": round(float(qs[2]), 4),
                "max": round(float(v.max()), 4), "mean": round(float(v.mean()), 4)}

    def _offdiag(Gm):
        iu = torch.triu_indices(Gm.shape[0], Gm.shape[1], 1)
        return Gm[iu[0], iu[1]]
    bidx = [x for x in M.jsonl_read(os.path.join(M.CACHE, bb, "index.jsonl")) if x["kind"] == "benign" and x["ptype"] == "image"]
    KB = torch.load(os.path.join(M.CACHE, bb, "keys_decode.pt"), map_location="cpu", mmap=True)[torch.tensor([x["row_decode"] for x in bidx])]
    Kn, Bn = torch.nn.functional.normalize(K_dec.double(), dim=1), torch.nn.functional.normalize(KB.double(), dim=1)
    G = Kn @ Kn.T
    img_id = {m_: i for i, m_ in enumerate(dict.fromkeys(e["image_md5"] for e in index))}
    ii = torch.tensor([img_id[e["image_md5"]] for e in index])
    same_img = (ii[:, None] == ii[None, :])
    iu = torch.triu_indices(G.shape[0], G.shape[1], 1)
    si = same_img[iu[0], iu[1]]
    cw = G[iu[0], iu[1]]
    res["key_statistics"] = {
        "what": "prompt-only keys (keys_decode.pt): Euclidean norms and raw cosines between keys; no gate, no edit log",
        "n_distinct_keys_bitwise": len({K_dec[i].numpy().tobytes() for i in range(K_dec.shape[0])}), "n_rows": int(K_dec.shape[0]),
        "norm": {"mme": _q(K_dec.double().norm(dim=1)), "benign_pool_image_rows": _q(KB.double().norm(dim=1))},
        "cosine_between_two_keys": {"mme_all_pairs": _q(cw), "mme_same_image_pairs": dict(_q(cw[si]), n=int(si.sum())),
                                    "mme_pairs_of_different_images": _q(cw[~si]),
                                    "benign_pool_image_rows_all_pairs": _q(_offdiag(Bn @ Bn.T)), "mme_x_benign_pool_image_rows": _q(Kn @ Bn.T)},
        "cosine_of_mme_c_mean_key_and_mme_t_mean_key": round(float(torch.nn.functional.cosine_similarity(
            K_dec[[j for j, e in enumerate(index) if e["part"] == "MME-C"]].double().mean(0), K_dec[[j for j, e in enumerate(index) if e["part"] == "MME-T"]].double().mean(0), dim=0)), 6),
        "cosine_of_mme_mean_key_and_benign_image_mean_key": round(float(torch.nn.functional.cosine_similarity(K_dec.double().mean(0), KB.double().mean(0), dim=0)), 6),
        "n_benign_pool_image_rows": int(KB.shape[0])}
    # (6) the unit of the split is the image FILE content (md5).  Is any picture stored twice, under two md5 sums, with identical
    #     decoded RGB pixels (the conversion the model wrapper applies)?  Such a pair in two parts would cross the split.
    #     (Exact pixel identity only: a re-encoded or resized copy of a picture is not detected.)
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    part_of_img = {e["image"]: e["part"] for e in index}

    def _pix(path):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            return path, hashlib.sha1(repr(rgb.size).encode() + rgb.tobytes()).hexdigest()
    with ThreadPoolExecutor(4) as ex:
        ph = dict(ex.map(_pix, sorted(part_of_img)))
    grp = collections.defaultdict(list)
    for path, hx in ph.items():
        grp[hx].append(path)
    dupg = [v for v in grp.values() if len(v) > 1]
    res["pixel_identity"] = {"n_image_files": len(ph), "n_distinct_decoded_rgb_contents": len(grp), "n_groups_of_files_with_identical_pixels": len(dupg),
                             "n_such_groups_in_both_parts": sum(len({part_of_img[p_] for p_ in v}) > 1 for v in dupg),
                             "groups": [[os.path.basename(p_) for p_ in v] for v in dupg][:20]}
    c = res["consistency"]
    res["ok"] = bool(c["sids_unique"] and c["row_decode_is_line_number"] and c["row_tf_unique_and_dense"] and c["decode_step0_equals_keys_decode"]
                     and c["n_position_identity_violations(lp_tf - lp_decode != lp_shift)"] == 0 and c["all_finite"] and c["n_zero_rows"] == 0
                     and c["keys_are_images_of_the_module_dtype"] and c["decode_step_rows_cover_keys_decode_steps"]
                     and c["tf_key_vs_decode_step_of_same_length"]["n_bitwise_equal"] == c["tf_key_vs_decode_step_of_same_length"]["n_checked"]
                     and c["n_pre_exact_differs_from_y0_valid"] == 0 and c["duplicate_queries"] == 0
                     and res["key_statistics"]["n_distinct_keys_bitwise"] == len(index)
                     and res["pixel_identity"]["n_such_groups_in_both_parts"] == 0
                     and res["library_reader"]["ok"] and all(v is True for k, v in res["library_reader"].items() if k.endswith("_equal") or k in ("find_by_content", "by_sid"))
                     and res["selection"]["rows_in_selection_order"] and res["selection"]["parts_equal"] and res["selection"]["prompts_images_equal"]
                     and res["selection"]["n_images_in_both_parts"] == 0 and res["selection"]["selection_md5_equals_meta"]
                     and (pre is None or all(v == len(index) for k, v in res["selection"]["cpu_dry_run_of_prompt_construction"].items() if k != "n"))
                     and res["images"]["n_images_shared_with_any_cache"] == 0 and res["images"]["n_rows_flagged_img_in_cache"] == 0
                     and res["images"]["n_files_whose_name_is_their_md5"] == len(h) and res["images"]["n_index_md5_equal_to_file_md5"] == len(index))
    res["seconds"] = round(time.time() - t0, 1)
    res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    M.jdump(res, out_path(bb, "cpu"))
    print(M.jdumps({k: res[k] for k in ("ok", "consistency", "library_reader", "selection", "images", "key_statistics", "pixel_identity")})[:9500])
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=M.BACKBONES)
    ap.add_argument("mode", choices=("gpu", "cpu"))
    ap.add_argument("--n", type=int, default=M.N_VERIFY)
    ap.add_argument("--trial", action="store_true", help="verify work/trial/<backbone> (cpu mode only)")
    a = ap.parse_args()
    if a.trial:
        if a.mode != "cpu":
            raise SystemExit("--trial: cpu mode only")
        ROOT = os.path.join(M.WORK, "trial")
    os.makedirs(os.path.join(M.WORK, "verify"), exist_ok=True)
    import traceback
    try:
        sys.exit(run_gpu(a.backbone, a.n) if a.mode == "gpu" else run_cpu(a.backbone))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
