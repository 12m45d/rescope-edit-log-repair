"""Step 3: GPU capture of the MME rows of one backbone under the PRE-EDIT model, through the caches' own capture code.

    python -I -B mme_capture.py <qwen25vl | llava> plan        CPU: writes the two manifests, lists the parts to do
    python -I -B mme_capture.py <backbone> proof               GPU: 30 rows of the EXISTING benign pool, compared bit for bit with the cache
    python -I -B mme_capture.py <backbone> capture             GPU: the MME rows (refuses to start without a passed proof of the same code)
    python -I -B mme_capture.py <backbone> selftest            CPU: the comparison code of `proof` on a part rebuilt from the cache
    (GPU steps only through the queue:  gq run --mem 10500 | 17000 --prio 3 --name mme_<backbone> -- bash run_bb.sh <backbone>)

NOTHING ABOUT PROMPTS, DECODING, VALIDITY OR KEYS IS IMPLEMENTED HERE.  Both GPU steps call 10_final/cache/code/fc_capture.main()
itself, the function that captured the benign pool, with fc_common.WORK / LOGDIR re-pointed to this package's work directory:
  manifest   work/<step>/<backbone>/select/manifest.jsonl, the file fc_capture.main reads.  proof: 30 manifest rows of the caches'
             own selection (cache/work/<backbone>/select/manifest.jsonl: benign image rows of the answer subset, seeded choice),
             copied unchanged.  capture: select/manifest_<backbone>.jsonl of mme_select.py (kind 'benign', free image query, fmt
             'vqa', decode = true for every row), copied byte for byte.
  call       fc_capture.main() with argv  <backbone> <VLKEB family> --groups benign --part N --mem_gb (GQ_MEM - 600) / 1024
             [--tag proof].  Inside: rescope.load(master log of the family) -> the PRE-EDIT model for the whole job
             (Evaluator.applied(H.W0), asserted), harness vision-feature memo; per row fc_capture.one(): decoder v2
             (tb_common.Prober.decode_v2: 10 greedy ids from full no-cache forwards cut at the first newline; 48-token long
             answer), validity (non-empty AND reproduced by the pre-edit model under teacher forcing: Prober.tf_state), the
             prompt-only batch, Evaluator.decode_keys (the prompt's last token in the 11 full forwards of a decode),
             Evaluator.capture (teacher-forced key at the harness index, and at the decode's token where lp_shift != 0),
             Evaluator.tf (pre-edit flags of the teacher-forced answer).
  parts      work/<step>/<backbone>/parts[_proof]/<family>_benign_<nnnn>.pt, one file per N rows; a finished part is skipped on
             restart (resumable); several workers may run (fc_capture's claim files).
  cap        the first GPU lines are the queue's (README.txt of gpuq): (GQ_MEM - 600) MiB per process; rescope.load sets the same cap.
proof        after fc_capture.main returns, the part is compared with 10_final/cache/<backbone>/: every decode-step key with
             keys_decode_steps.pt[rows_decode_steps], the prompt-only key with keys_decode.pt[row_decode], the teacher-forced key
             with keys_tf.pt[row_tf] (bitwise on the float32 images), positions and lengths, the decoder-v2 answer, its ids, its
             validity and the pre-edit flags with index.jsonl and benign/<backbone>/rows.jsonl.  Result: work/proof/<backbone>/
             proof.json; exit code 2 if anything differs (the chain then stops before the long run).
"""
from __future__ import annotations

import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402

KEY_FIELDS = ("lp_decode", "len_prompt", "n_image_tokens", "n_decode_distinct", "lp_tf", "lp_shift", "len_tf")
ANSWER_FIELDS_INDEX = ("y0_text", "y0_valid", "y0_invalid_reason", "pre_exact", "pre_token_acc", "pre_n_tok", "pre_finite", "pre_first_mismatch")
ANSWER_FIELDS_ROWS = ("y0_ids", "y0_gen_ids", "y0_stop", "y0_raw", "y0_roundtrip_ids_equal", "w0_reproduced", "w0_n_ans_tok", "w0_first_mismatch",
                      "y0_note", "long_text", "long_ids", "long_stop")


def family(bb):
    return M.cache_code().BACKBONES[bb]["vlkeb"]


def proof_dir(bb):
    return os.path.join(M.PROOF_WORK, bb)


def capture_dir(bb):
    return os.path.join(M.CAPTURE_WORK, bb)


def write_stub(sd):
    """fc_capture.main loads select/hn_edits.json whatever the groups are; it is only used for the group 'editpre' (not run here)."""
    M.jdump({"later100": [], "note": "stub: fc_capture.main loads this file; the group 'editpre' that uses it is not run by the MME capture"},
            os.path.join(sd, "hn_edits.json"))


def proof_manifest(bb) -> list:
    """30 manifest rows of the caches' own selection: benign IMAGE rows of the answer subset (decoder v2 + teacher-forced key +
    decode-step keys = the whole path an MME row takes), seeded choice, unchanged."""
    import numpy as np
    man = M.jsonl_read(os.path.join(M.CACHE, "work", bb, "select", "manifest.jsonl"))
    cand = [m for m in man if m["kind"] == "benign" and m["ptype"] == "image" and m["decode"]]
    pick = np.sort(np.random.default_rng(M.useed("proof", bb)).choice(len(cand), M.N_PROOF, replace=False))
    return [cand[int(i)] for i in pick]


def setup(bb, step) -> dict:
    """Write the manifest fc_capture.main will read for `step`; -> {'work', 'tag', 'part', 'manifest', 'parts', 'parts_dir'}."""
    import fc_capture as CAP
    fam = family(bb)
    if step == "proof":
        work, tag, part = M.PROOF_WORK, "proof", M.N_PROOF
        sd = os.path.join(work, bb, "select")
        os.makedirs(sd, exist_ok=True)
        rows = proof_manifest(bb)
        M.jsonl_write(rows, os.path.join(sd, "manifest.jsonl"))
    else:
        work, tag, part = M.CAPTURE_WORK, "", M.PART_SIZE
        sd = os.path.join(work, bb, "select")
        os.makedirs(sd, exist_ok=True)
        src, dst = os.path.join(M.SELECT, f"manifest_{bb}.jsonl"), os.path.join(sd, "manifest.jsonl")
        pd = os.path.join(work, bb, "parts")
        if os.path.exists(dst) and M.md5_file(dst) != M.md5_file(src) and os.path.isdir(pd) and any(f.endswith(".pt") for f in os.listdir(pd)):
            raise SystemExit(f"{dst} differs from the selection in use and capture parts exist: move work/capture/{bb} away first")
        shutil.copyfile(src, dst + f".tmp{os.getpid()}")
        os.replace(dst + f".tmp{os.getpid()}", dst)
        rows = M.jsonl_read(dst)
        assert M.md5_file(dst) == M.jload(os.path.join(M.SELECT, "select_summary.json"))["files"][f"manifest_{bb}.jsonl"]
    write_stub(sd)
    pd = os.path.join(work, bb, "parts" + (f"_{tag}" if tag else ""))
    return {"work": work, "tag": tag, "part": part, "manifest": rows, "parts_dir": pd, "family": fam,
            "parts": [p for p, _ in CAP.plan_parts(rows, fam, part, ["benign"])]}


def mem_gb() -> float:
    if not os.environ.get("GQ_MEM"):
        raise SystemExit("GPU steps run only through the queue (gq run sets GQ_MEM): see run_bb.sh")
    return (int(os.environ["GQ_MEM"]) - 600) / 1024.0


def run_fc_capture(bb, st) -> int:
    """fc_capture.main() of the caches on the manifest of `st` (see the module docstring)."""
    C = M.cache_code()
    import fc_capture as CAP
    C.tb()                                                              # the testbed's environment (allocator setting) before CUDA starts, as in fc_capture
    import torch
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20                  # the queue's cap (gpuq/README.txt): GQ_MEM = the --mem of this job
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)
    C.WORK, C.LOGDIR = st["work"], M.LOGDIR                             # fc_capture.main reads C.WORK/<bb>/select, writes C.WORK/<bb>/parts*, logs to C.LOGDIR
    argv = ["fc_capture.py", bb, st["family"], "--groups", "benign", "--part", str(st["part"]), "--mem_gb", repr(mem_gb())]
    if st["tag"]:
        argv += ["--tag", st["tag"]]
    old = sys.argv
    sys.argv = argv
    try:
        M.say(f"fc_capture.main {argv[1:]} with WORK = {C.WORK} (GQ_MEM {os.environ.get('GQ_MEM')}, CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')})")
        return int(CAP.main() or 0)
    finally:
        sys.argv = old


# ----------------------------------------------------------------------------- proof: comparison with the cache
def compare_with_cache(bb, blob) -> dict:
    """Bitwise comparison of one fc_capture part (rows of the existing benign pool) with the key cache of the backbone."""
    import torch
    od = os.path.join(M.CACHE, bb)
    index = {e["sid"]: e for e in M.jsonl_read(os.path.join(od, "index.jsonl")) if e["kind"] == "benign"}
    ben = {r["sid"]: r for r in M.jsonl_read(os.path.join(M.CACHE, "benign", bb, "rows.jsonl"))}
    K_tf = torch.load(os.path.join(od, "keys_tf.pt"), map_location="cpu", mmap=True)
    K_dec = torch.load(os.path.join(od, "keys_decode.pt"), map_location="cpu", mmap=True)
    K_st = torch.load(os.path.join(od, "keys_decode_steps.pt"), map_location="cpu", mmap=True)
    recs = []
    for r in blob["rows"]:
        e, b = index[r["sid"]], ben[r["sid"]]
        r0, nd = int(r["dec_row0"]), int(r["n_decode_distinct"])
        steps = blob["K_dec"][r0:r0 + nd].float()[torch.tensor(r["decode_map"], dtype=torch.long)]       # the 11 keys of the decode's full forwards
        ref = K_st[torch.tensor(e["rows_decode_steps"], dtype=torch.long)]
        rec = {"sid": r["sid"], "ptype": e["ptype"], "answer_subset": bool(e.get("answer_subset")), "y0_text": r.get("y0_text"), "lp_shift": r.get("lp_shift"),
               "decode_steps_bitwise": bool(torch.equal(steps, ref)), "n_decode_steps_differing": int((steps != ref).any(1).sum()),
               "prompt_key_bitwise": bool(torch.equal(steps[0], K_dec[int(e["row_decode"])])),
               "decode_map_equal": [int(x) for x in r["decode_map"]] == [int(x) - int(e["rows_decode_steps"][0]) for x in e["rows_decode_steps"]],
               "has_tf": bool(r["has_tf"]), "cache_has_tf": e["row_tf"] is not None, "tf_key_bitwise": None, "tf_prompt_key_bitwise": None}
        if r["has_tf"] and e["row_tf"] is not None:
            k = blob["K_tf"][int(r["tf_row"])].float()
            rec["tf_key_bitwise"] = bool(torch.equal(k, K_tf[int(e["row_tf"])]))
            rec["tf_key_max_abs_diff"] = float((k - K_tf[int(e["row_tf"])]).abs().max())
            if r.get("has_tf_prompt"):
                rec["tf_prompt_key_bitwise"] = e["row_tf_prompt"] is not None and bool(torch.equal(blob["K_tfp"][int(r["tfp_row"])].float(), K_tf[int(e["row_tf_prompt"])]))
        rec["fields_differing"] = ([k for k in KEY_FIELDS + ANSWER_FIELDS_INDEX if r.get(k) != e.get(k)]
                                   + [k for k in ANSWER_FIELDS_ROWS if r.get(k) != b.get(k)])
        rec["ok"] = bool(rec["decode_steps_bitwise"] and rec["prompt_key_bitwise"] and rec["decode_map_equal"] and rec["has_tf"] == rec["cache_has_tf"]
                         and rec["tf_key_bitwise"] is not False and rec["tf_prompt_key_bitwise"] is not False and not rec["fields_differing"])
        recs.append(rec)

    def cnt(key):
        v = [x[key] for x in recs if x[key] is not None]
        return [int(sum(v)), len(v)]
    return {"ok": bool(recs) and all(x["ok"] for x in recs), "n_rows": len(recs), "n_rows_ok": sum(x["ok"] for x in recs),
            "prompt_only_key_bitwise": cnt("prompt_key_bitwise"), "decode_step_keys_bitwise(rows; 11 keys each)": cnt("decode_steps_bitwise"),
            "teacher_forced_key_bitwise": cnt("tf_key_bitwise"), "teacher_forced_key_at_decode_token_bitwise": cnt("tf_prompt_key_bitwise"),
            "n_rows_with_decoder_v2_answer": sum(x["y0_text"] is not None for x in recs),
            "n_rows_answer_ids_validity_positions_equal": sum(not x["fields_differing"] for x in recs),
            "fields_compared": {"with index.jsonl": list(KEY_FIELDS + ANSWER_FIELDS_INDEX), "with benign/<backbone>/rows.jsonl": list(ANSWER_FIELDS_ROWS)},
            "rows": recs}


def proof_record(bb, st, blob, extra=None) -> dict:
    res = compare_with_cache(bb, blob)
    out = {"backbone": bb, "ok": res["ok"],
           "what": f"{M.N_PROOF} rows of the existing benign pool (image rows of the answer subset, seeded choice) captured again by "
                   "fc_capture.main() through mme_capture.py and compared with 10_final/cache/<backbone>/ : keys bit for bit on the float32 "
                   "images, positions, decoder-v2 answer and ids, validity, pre-edit flags",
           "seed": [M.SEED, "proof", bb], "sids": [m["sid"] for m in st["manifest"]], "batch_size": 1,
           "part": {k: blob.get(k) for k in ("part", "master", "dtype", "seconds", "gpu", "torch", "cuda_visible", "time")},
           "cache_index_md5": M.md5_file(os.path.join(M.CACHE, bb, "index.jsonl")), "code": M.code_fingerprint(),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    out.update({k: v for k, v in res.items() if k != "ok"})
    out.update(extra or {})
    return out


def fake_part(bb, st) -> dict:
    """A part rebuilt from the CACHE for the proof rows (CPU self-test of compare_with_cache: it must pass on it, and fail after a bit flip)."""
    import torch
    od = os.path.join(M.CACHE, bb)
    index = {e["sid"]: e for e in M.jsonl_read(os.path.join(od, "index.jsonl")) if e["kind"] == "benign"}
    ben = {r["sid"]: r for r in M.jsonl_read(os.path.join(M.CACHE, "benign", bb, "rows.jsonl"))}
    K_tf, K_st = torch.load(os.path.join(od, "keys_tf.pt"), mmap=True), torch.load(os.path.join(od, "keys_decode_steps.pt"), mmap=True)
    rows, ktf, kdec = [], [], []
    for m in st["manifest"]:
        e, b = index[m["sid"]], ben[m["sid"]]
        rs_ = e["rows_decode_steps"]
        rec = {"sid": m["sid"], "dec_row0": sum(k.shape[0] for k in kdec), "n_decode_distinct": e["n_decode_distinct"], "decode_map": [x - rs_[0] for x in rs_],
               "has_tf": e["row_tf"] is not None, "has_tf_prompt": False}
        rec.update({k: e.get(k) for k in KEY_FIELDS + ANSWER_FIELDS_INDEX})
        rec.update({k: b.get(k) for k in ANSWER_FIELDS_ROWS})
        kdec.append(K_st[rs_[0]:rs_[0] + e["n_decode_distinct"]].clone())
        if e["row_tf"] is not None:
            rec["tf_row"] = len(ktf)
            ktf.append(K_tf[e["row_tf"]].clone())
        rows.append(rec)
    return {"part": "fake", "rows": rows, "K_dec": torch.cat(kdec), "K_tf": torch.stack(ktf), "K_tfp": torch.zeros(0, K_tf.shape[1]), "master": "none",
            "dtype": "float32 (rebuilt from the cache)", "seconds": 0.0, "gpu": "none", "torch": str(torch.__version__), "cuda_visible": None, "time": ""}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=M.BACKBONES)
    ap.add_argument("step", choices=("plan", "proof", "capture", "selftest"))
    a = ap.parse_args()
    bb = a.backbone
    M.cache_code()
    os.makedirs(M.LOGDIR, exist_ok=True)
    pj = os.path.join(proof_dir(bb), "proof.json")
    fp = M.code_fingerprint()

    if a.step == "plan":
        for step in ("proof", "capture"):
            st = setup(bb, step)
            done = [p for p in st["parts"] if os.path.exists(os.path.join(st["parts_dir"], p + ".pt"))]
            print(f"{bb} {step}: {len(st['manifest'])} rows, {len(st['parts'])} parts of {st['part']} ({len(done)} done) in {st['parts_dir']}; "
                  f"kinds {sorted({m['kind'] for m in st['manifest']})}, decode {sorted({m['decode'] for m in st['manifest']})}, family {st['family']}")
        print("proof passed with this code:", os.path.exists(pj) and M.jload(pj).get("ok") and M.jload(pj).get("code") == fp)
        return 0

    if a.step == "selftest":
        import torch
        st = setup(bb, "proof")
        blob = fake_part(bb, st)
        good = compare_with_cache(bb, blob)
        blob["K_tf"][3, 17] = torch.nextafter(blob["K_tf"][3, 17], torch.tensor(float("inf")))          # one bit of one teacher-forced key
        bad_tf = compare_with_cache(bb, blob)
        blob = fake_part(bb, st)
        blob["K_dec"][5, 100] = torch.nextafter(blob["K_dec"][5, 100], torch.tensor(float("inf")))      # one bit of one decode-step key
        bad_dec = compare_with_cache(bb, blob)
        blob = fake_part(bb, st)
        blob["rows"][7]["y0_text"] = (blob["rows"][7]["y0_text"] or "") + "x"
        bad_txt = compare_with_cache(bb, blob)
        res = {"backbone": bb, "pass_on_cache_copy": good["ok"], "n_rows": good["n_rows"], "tf": good["teacher_forced_key_bitwise"],
               "steps": good["decode_step_keys_bitwise(rows; 11 keys each)"],
               "detects_one_bit_in_tf_key": (not bad_tf["ok"]) and bad_tf["n_rows_ok"] == good["n_rows"] - 1,
               "detects_one_bit_in_decode_key": (not bad_dec["ok"]) and bad_dec["n_rows_ok"] == good["n_rows"] - 1,
               "detects_changed_answer": (not bad_txt["ok"]) and bad_txt["n_rows_ok"] == good["n_rows"] - 1}
        res["ok"] = all(v for k, v in res.items() if k.startswith(("pass", "detects")))
        M.jdump(res, os.path.join(proof_dir(bb), "selftest_compare.json"))
        print(res)
        return 0 if res["ok"] else 1

    if a.step == "proof":
        if os.path.exists(pj) and M.jload(pj).get("ok") and M.jload(pj).get("code") == fp:
            M.say(f"{bb}: proof already passed with this code ({pj})")
            return 0
        st = setup(bb, "proof")
        for p in st["parts"]:                                           # a part of an earlier code version is captured again
            for ext in (".pt", ".claim"):
                if os.path.exists(os.path.join(st["parts_dir"], p + ext)):
                    os.remove(os.path.join(st["parts_dir"], p + ext))
        t0 = time.time()
        rc = run_fc_capture(bb, st)
        import torch
        blob = torch.load(os.path.join(st["parts_dir"], st["parts"][0] + ".pt"), map_location="cpu", weights_only=False)       # our own part file
        assert blob["sids"] == [m["sid"] for m in st["manifest"]]
        out = proof_record(bb, st, blob, {"fc_capture_exit_code": rc, "seconds_total": round(time.time() - t0, 1),
                                          "GQ_MEM": os.environ.get("GQ_MEM"), "cap_gb": round(mem_gb(), 3)})
        M.jdump(out, pj)
        M.say(f"PROOF {bb}: ok = {out['ok']}; rows {out['n_rows_ok']}/{out['n_rows']}; prompt-only key bitwise {out['prompt_only_key_bitwise']}, "
              f"decode-step keys {out['decode_step_keys_bitwise(rows; 11 keys each)']}, teacher-forced key {out['teacher_forced_key_bitwise']}, "
              f"answers / validity / positions equal {out['n_rows_answer_ids_validity_positions_equal']}/{out['n_rows']}")
        return 0 if out["ok"] else 2

    # ------------------------------------------------------------------ capture of the MME rows
    if not (os.path.exists(pj) and M.jload(pj).get("ok")):
        M.say(f"STOP: no passed proof for {bb} ({pj}); run the step 'proof' first")
        return 2
    if M.jload(pj).get("code") != fp:
        M.say(f"STOP: the proof of {bb} was made with another version of the code; run the step 'proof' again")
        return 2
    st = setup(bb, "capture")
    rc = run_fc_capture(bb, st)
    missing = [p for p in st["parts"] if not os.path.exists(os.path.join(st["parts_dir"], p + ".pt"))]
    M.say(f"CAPTURE {bb}: fc_capture.main returned {rc}; {len(st['parts']) - len(missing)}/{len(st['parts'])} parts exist" + (f"; missing {missing[:5]}" if missing else ""))
    return 0 if (rc == 0 and not missing) else 3


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
