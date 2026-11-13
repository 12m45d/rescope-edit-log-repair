"""GPU capture of the recombination keys of one backbone (amendment R5): for every row of cache2/recomb/<backbone>/rows.jsonl the
input of the edited module at the last prompt position of the PROMPT-ONLY sequence (no answer tokens) under the PRE-EDIT model, B = 1.

    python rc_capture.py <qwen25vl | llava> [--steps verify,capture,recheck] [--part 200] [--shard i/n] [--reverse]
                         [--limit N] [--tag smoke] [--mem_gb X] [--fake]
    (through the queue:  gq run --mem 10500 | 17000 --prio 4 --name <tag> -- python rc_capture.py <backbone>)

THE CODE PATH IS THE CACHES' (10_final/cache/code/fc_capture.py); nothing about prompts or positions is implemented here:
  handle       rescope.load(<master log of the backbone's VLKEB family>), as fc_capture.main; the pre-edit model for the whole job
               (Evaluator.applied(H.W0)); the harness' vision-feature memo is active and trimmed per row (as fc_capture.main)
  evaluator    fc_capture.make_evaluator(rs, H)
  item         fc_capture.make_item(<manifest sequence>): a recombination row is handed over as a free image query without an
               answer, the form of a benign image row ({'kind', 'sid', '_row': {image, prompt, y0_text: '', ptype: 'image'}})
  key          fc_capture.one's decode-path lines:  bq = ev.batch(it, answer=False);  dk = ev.decode_keys(it, n)   with n = 0:
               decode_keys(it, 0)[0] is the module input at the prompt's own last token in the prompt-only forward, i.e. row 0 of
               the 11 decode keys that fc_capture stores, i.e. keys_decode.pt[row_decode] of the key cache (fc_assemble: K_dec[j] =
               the first decode key).  last prompt index / sequence length / image tokens come from bq, as in fc_capture.one.

STEPS (default: all three, in this order, in ONE process)
  verify    BEFORE anything is captured: a seeded sample of 50 existing hard-negative rows and 50 probe rows of the key cache is
            re-captured with the function above and compared BIT FOR BIT with keys_decode.pt[row_decode]; in addition their
            teacher-forced key (Evaluator.capture(it, True), fc_capture.one's teacher-forced lines) is compared with
            keys_tf.pt[row_tf] and, for hard negatives, with hard_negatives/<backbone>/keys.pt.  Result:
            <backbone>/verify_capture.json.  Any difference stops the job (exit code 2) before the long run.
  capture   the rows in file order, in parts of --part rows: work/<backbone>/parts/part_<nnnn>.pt (keys in the module dtype, row
            ids, last prompt index, ...).  Resumable: a finished part is skipped; a part is claimed by a live process (several
            workers may run: --shard i/n, --reverse).  A part records the sha256 of rows.jsonl it was captured for.
  recheck   after the last part: a seeded sample of 32 rows per kind is captured again WITHOUT the vision-feature memo and
            compared bit for bit with the stored parts: <backbone>/recheck_capture.json.
--fake: no model; pseudo keys from a hash of the query (CPU test of the part / claim / assemble mechanics; tag 'fake').
Then: python rc_assemble.py <backbone>  ->  <backbone>/keys.pt, README.txt, capture_meta.json.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as R  # noqa: E402
from rc_common import C  # noqa: E402
import fc_capture as CAP  # noqa: E402  (make_item, make_evaluator: the caches' capture)

N_VERIFY = 50            # rows per kind (hard negatives, probes) of the bitwise proof of the code path
N_RECHECK = 32           # rows per recombination kind captured twice


def item_of(r):
    """A recombination row as fc_capture's library item (free image query without an answer; the form of a benign image row)."""
    return CAP.make_item({"lib_kind": "recomb", "sid": r["rid"], "image": r["image"], "prompt": r["prompt"], "ptype": "image", "answer": None})


def prompt_key(ev, it):
    """(key (d,) module dtype, last prompt index, sequence length, image tokens) of the PROMPT-ONLY sequence of an item:
    fc_capture.one's decode-path lines with n = 0 (the first of the decode keys = keys_decode.pt[row_decode] of the key cache)."""
    bq = ev.batch(it, answer=False)
    dk = ev.decode_keys(it, n=0)
    return dk[0], int(bq["last_prompt_idx"][0]), int(bq["seq_len"][0]), int(bq["n_image_tokens"][0])


class FakeEvaluator:
    """No model: deterministic pseudo keys from a hash of the query (tests of the bookkeeping only)."""
    d = 64

    def batch(self, it, answer=False):
        n = 20 + len(it["_row"]["prompt"])
        return {"last_prompt_idx": [n - 1], "seq_len": [n], "n_image_tokens": [16]}

    def decode_keys(self, it, n=0):
        import numpy as np
        import torch
        r = it["_row"]
        seed = int(hashlib.sha1(f"{r['image']}|{r['prompt']}".encode()).hexdigest()[:8], 16)
        return torch.from_numpy(np.random.default_rng(seed).standard_normal((n + 1, self.d)).astype("float32")).to(torch.bfloat16)

    def trim_vcache(self):
        pass


def claim(PD, pname):
    """fc_capture.main's claim: True if this process may compute the part (not finished, no LIVE process holds it)."""
    if os.path.exists(os.path.join(PD, pname + ".pt")):
        return False
    cp = os.path.join(PD, pname + ".claim")
    for _ in range(2):
        try:
            fd = os.open(cp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(cp).read().strip() or 0)
                os.kill(pid, 0)
                return False                                        # a live worker holds it
            except (ValueError, ProcessLookupError, FileNotFoundError):
                try:
                    os.remove(cp)                                   # stale claim: take over
                except FileNotFoundError:
                    pass
            except PermissionError:
                return False
    return False


def lib_fingerprint():
    out = {}
    for f in ("evaluate.py", "scoped.py"):
        with open(os.path.join(C.CODE, "rescope", f), "rb") as fh:
            out[f] = hashlib.sha1(fh.read()).hexdigest()[:12]
    with open(os.path.join(R.CACHE_CODE, "fc_capture.py"), "rb") as fh:
        out["fc_capture.py"] = hashlib.sha1(fh.read()).hexdigest()[:12]
    with open(os.path.abspath(__file__), "rb") as fh:
        out["rc_capture.py"] = hashlib.sha1(fh.read()).hexdigest()[:12]
    return out


def verify(BB, ev, H, hz, fh):
    """The bitwise proof of the code path on rows that the key cache already holds.  -> record (record['ok'])."""
    import numpy as np
    import torch
    fam = C.BACKBONES[BB]["vlkeb"]
    man = C.jsonl_read(os.path.join(R.CACHE, "work", BB, "select", "manifest.jsonl"))
    index = {e["sid"]: e for e in C.jsonl_read(os.path.join(R.CACHE, BB, "index.jsonl"))}
    pools = {"hn": [m for m in man if m["lib_kind"] == "hn"], "probe": [m for m in man if m["lib_kind"] == "probe" and m["family"] == fam]}
    sample = []
    for tag, rows in pools.items():
        pick = np.sort(np.random.default_rng(R.useed("verify", BB, tag)).choice(len(rows), N_VERIFY, replace=False))
        sample += [rows[int(i)] for i in pick]
    K_dec = torch.load(os.path.join(R.CACHE, BB, "keys_decode.pt"), map_location="cpu", mmap=True)
    K_tf = torch.load(os.path.join(R.CACHE, BB, "keys_tf.pt"), map_location="cpu", mmap=True)
    hk = torch.load(os.path.join(C.HN_NEW, BB, "keys.pt"), map_location="cpu")
    hpos = {h: i for i, h in enumerate(hk["hn_ids"])}
    recs, t0 = [], time.time()
    with torch.no_grad(), ev.applied(H.W0), hz.HE.vision_cache(H):
        assert torch.equal(H.mod.weight.data, H.W0)
        for m in sample:
            ev.trim_vcache()
            e = index[m["sid"]]
            it = CAP.make_item(m, y0=e.get("answer")) if m["lib_kind"] == "hn" else CAP.make_item(m)
            key, lp, sl, nimg = prompt_key(ev, it)
            k32, ref = key.float().cpu(), K_dec[int(e["row_decode"])]
            rec = {"sid": m["sid"], "kind": m["kind"], "ptype": m["ptype"], "new": m.get("new"), "lp": lp, "lp_cached": e["lp_decode"],
                   "len_prompt": sl, "len_prompt_cached": e["len_prompt"], "n_image_tokens": nimg, "module_dtype": str(key.dtype),
                   "prompt_key_bitwise": bool(torch.equal(k32, ref)), "prompt_key_max_abs_diff": float((k32 - ref).abs().max()),
                   "tf_key_bitwise": None, "tf_key_equals_hn_keys_pt": None}
            if e.get("row_tf") is not None:                             # the teacher-forced key of the same row (fc_capture.one)
                b = ev.batch(it, True)
                x, p = ev.capture(it, True, batch=b)
                tf = x[p].float().cpu()
                rec.update(lp_tf=int(p), lp_tf_cached=e["lp_tf"], lp_shift=e.get("lp_shift"),
                           tf_key_bitwise=bool(torch.equal(tf, K_tf[int(e["row_tf"])])),
                           tf_key_max_abs_diff=float((tf - K_tf[int(e["row_tf"])]).abs().max()))
                if m["lib_kind"] == "hn":
                    rec["tf_key_equals_hn_keys_pt"] = bool(torch.equal(tf, hk["K_last"][hpos[m["hn_id"]]]))
            recs.append(rec)

    def cnt(key, sub):
        v = [r[key] for r in sub if r[key] is not None]
        return [int(sum(v)), len(v)]
    by = {}
    for tag, kinds in (("hard_negatives", ("hn_i", "hn_q", "hn_x")), ("probes", ("probe",))):
        sub = [r for r in recs if r["kind"] in kinds]
        by[tag] = {"n": len(sub), "by_kind": {k: sum(r["kind"] == k for r in sub) for k in kinds},
                   "by_ptype": {p: sum(r["ptype"] == p for r in sub) for p in ("image", "text")},
                   "existing_new": [sum(r["new"] is False for r in sub), sum(r["new"] is True for r in sub)] if tag == "hard_negatives" else None,
                   "prompt_key_bitwise": cnt("prompt_key_bitwise", sub), "last_prompt_index_equal": [sum(r["lp"] == r["lp_cached"] for r in sub), len(sub)],
                   "tf_key_bitwise": cnt("tf_key_bitwise", sub), "tf_key_equals_hn_keys_pt": cnt("tf_key_equals_hn_keys_pt", sub),
                   "n_lp_shift": sum(bool(r.get("lp_shift")) for r in sub)}
    ok_prompt = all(r["prompt_key_bitwise"] and r["lp"] == r["lp_cached"] and r["len_prompt"] == r["len_prompt_cached"] for r in recs)
    ok_tf = all(r[k] is not False for r in recs for k in ("tf_key_bitwise", "tf_key_equals_hn_keys_pt"))
    out = {"backbone": BB, "ok": bool(ok_prompt and ok_tf), "ok_prompt_only_key": bool(ok_prompt), "ok_teacher_forced_key": bool(ok_tf),
           "what": "seeded sample of existing hard-negative rows and probe rows of the key cache, re-captured with rc_capture.prompt_key (prompt-only "
                   "key; compared with 10_final/cache/<backbone>/keys_decode.pt[row_decode]) and with Evaluator.capture(it, True) (teacher-forced key; "
                   "compared with keys_tf.pt[row_tf] and hard_negatives/<backbone>/keys.pt); bit for bit on the float32 images",
           "batch_size": 1, "n_rows": len(recs), "summary": by, "master_log": H.name, "module_dtype": str(H.dtype), "seconds": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "time": time.strftime("%Y-%m-%d %H:%M:%S"), "code": lib_fingerprint(), "seed": [R.SEED, "verify", BB, "hn | probe"], "rows": recs}
    R.jdump(out, os.path.join(R.bb_dir(BB), "verify_capture.json"))
    R.say(f"  VERIFY {BB}: ok = {out['ok']}; " + "; ".join(
        f"{t}: prompt-only key bitwise {v['prompt_key_bitwise'][0]}/{v['prompt_key_bitwise'][1]}, teacher-forced key bitwise "
        f"{v['tf_key_bitwise'][0]}/{v['tf_key_bitwise'][1]}" for t, v in by.items()) + f" ({out['seconds']} s)", fh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", choices=R.BACKBONES)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_gb", type=float, default=None)
    ap.add_argument("--part", type=int, default=200)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--reverse", action="store_true", help="take the parts from the end (a second worker)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N rows")
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", default="verify,capture,recheck")
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    BB = args.backbone
    cfg = C.BACKBONES[BB]
    master = cfg["families"][cfg["vlkeb"]]
    steps = [s for s in args.steps.split(",") if s]
    tag = "fake" if args.fake else args.tag
    if args.fake:
        steps = [s for s in steps if s == "capture"]
    PD = os.path.join(R.WORK, BB, "parts" + (f"_{tag}" if tag else ""))
    os.makedirs(PD, exist_ok=True)
    os.makedirs(R.LOGDIR, exist_ok=True)
    fh = open(os.path.join(R.LOGDIR, f"capture_{BB}{'_' + tag if tag else ''}.log"), "a")
    R.say(f"==== rc_capture {vars(args)} pid {os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} GQ_MEM={os.environ.get('GQ_MEM')}", fh)
    rows = R.jsonl_read(R.rows_path(BB))
    sel_sha = R.file_sha256(R.rows_path(BB))
    if args.limit:
        rows = rows[:args.limit]
    parts = [(f"part_{a // args.part:04d}", rows[a:a + args.part]) for a in range(0, len(rows), args.part)]
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = [(n, p, rs_) for n, (p, rs_) in enumerate(parts) if n % sn == si and not os.path.exists(os.path.join(PD, p + ".pt"))]
    if args.reverse:
        todo = todo[::-1]
    vp = os.path.join(R.bb_dir(BB), "verify_capture.json")
    need_verify = "verify" in steps and not (os.path.exists(vp) and R.jload(vp).get("ok") and R.jload(vp).get("code") == lib_fingerprint())
    need_recheck = "recheck" in steps and not args.limit
    R.say(f"  {len(rows)} rows, {len(parts)} parts of {args.part}, {len(todo)} to do in shard {args.shard}; verify needed: {need_verify}", fh)
    if not todo and not need_verify and not (need_recheck and not os.path.exists(os.path.join(R.bb_dir(BB), "recheck_capture.json"))):
        R.say("DONE (nothing to do)", fh)
        return 0

    import torch
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    if args.fake:
        ev, H, hz, gpu, dtype, d_in = FakeEvaluator(), None, None, "none", "torch.bfloat16", FakeEvaluator.d
    else:
        C.tb()                                                          # registers the long logs before the harness loads one (as fc_capture)
        rs = C.import_rescope()
        mem_gb = args.mem_gb or ((int(os.environ["GQ_MEM"]) - 600) / 1024.0 if os.environ.get("GQ_MEM") else cfg["mem_gb"])
        t0 = time.time()
        H = rs.load(master, device=args.device, mem_gb=mem_gb)
        ev = CAP.make_evaluator(rs, H)
        hz = rs.harness()
        assert H.backbone == BB and H.L == cfg["layer"], (H.backbone, H.L)
        assert all(H.checks[k] for k in H.checks if k != "collapsed_working_point"), H.checks
        gpu, dtype, d_in = torch.cuda.get_device_name(0), str(H.dtype), H.d_in
        R.say(f"  loaded {master} in {time.time() - t0:.0f} s (cap {mem_gb:.2f} GB): T = {H.T}, layer {H.L}, dtype {H.dtype}, d_in {H.d_in}; "
              f"GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", fh)
        if need_verify:
            v = verify(BB, ev, H, hz, fh)
            if not v["ok"]:
                R.say("STOP: the capture path does not reproduce the cached keys bit for bit (see verify_capture.json); nothing was captured", fh)
                return 2
        elif "capture" in steps and not os.path.exists(vp):
            R.say("STOP: no verify_capture.json (run the step 'verify' first)", fh)
            return 2

    import contextlib

    def model_state(memo=True):
        if args.fake:
            return contextlib.nullcontext()
        st = contextlib.ExitStack()
        st.enter_context(torch.no_grad())
        st.enter_context(ev.applied(H.W0))
        if memo:
            st.enter_context(hz.HE.vision_cache(H))
        return st

    n_done = 0
    if "capture" in steps:
        for n, pname, prow in todo:
            if not claim(PD, pname):
                continue
            t1 = time.time()
            K, lp, sl, ni = [], [], [], []
            with model_state():
                if not args.fake:
                    assert torch.equal(H.mod.weight.data, H.W0)
                for r in prow:
                    ev.trim_vcache()
                    key, p, s, m = prompt_key(ev, item_of(r))
                    K.append(key.cpu())
                    lp.append(p)
                    sl.append(s)
                    ni.append(m)
            blob = {"part": pname, "rids": [r["rid"] for r in prow], "K": torch.stack(K), "last_prompt_idx": lp, "seq_len": sl, "n_image_tokens": ni,
                    "rows_sha256": sel_sha, "master": master, "dtype": dtype, "seconds": time.time() - t1, "gpu": gpu, "torch": str(torch.__version__),
                    "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid()}
            tmp = os.path.join(PD, pname + f".tmp{os.getpid()}")
            torch.save(blob, tmp)
            os.replace(tmp, os.path.join(PD, pname + ".pt"))
            try:
                os.remove(os.path.join(PD, pname + ".claim"))
            except FileNotFoundError:
                pass
            n_done += 1
            dt = time.time() - t1
            peak = 0.0 if args.fake else torch.cuda.max_memory_allocated() / 2 ** 30
            R.say(f"  {pname} ({len(prow)} rows, {dt:.0f} s, {dt / max(1, len(prow)):.3f} s/row; {n_done}/{len(todo)} parts of this worker; GPU peak {peak:.2f} GB)", fh)

    # ------------------------------------------------------------------ recheck: captured twice, the second time without the vision memo
    all_done = all(os.path.exists(os.path.join(PD, p + ".pt")) for p, _ in parts)
    if need_recheck and all_done and not args.fake:
        import numpy as np
        where = {r["rid"]: (p, j) for p, prow in parts for j, r in enumerate(prow)}
        sample = []
        for kind in R.KINDS:
            sub = [r for r in rows if r["kind"] == kind]
            sample += [sub[int(i)] for i in np.sort(np.random.default_rng(R.useed("recheck", BB, kind)).choice(len(sub), N_RECHECK, replace=False))]
        blobs, recs, t1 = {}, [], time.time()
        with model_state(memo=False):
            assert torch.equal(H.mod.weight.data, H.W0)
            for r in sample:
                p, j = where[r["rid"]]
                if p not in blobs:
                    blobs[p] = torch.load(os.path.join(PD, p + ".pt"), map_location="cpu", weights_only=False)
                key, lp_, _, _ = prompt_key(ev, item_of(r))
                assert blobs[p]["rids"][j] == r["rid"]
                recs.append({"rid": r["rid"], "kind": r["kind"], "bitwise": bool(torch.equal(key.cpu(), blobs[p]["K"][j])),
                             "lp_equal": lp_ == blobs[p]["last_prompt_idx"][j], "first_capture": {k: blobs[p][k] for k in ("gpu", "cuda_visible", "pid", "time")}})
        out = {"backbone": BB, "ok": all(r["bitwise"] and r["lp_equal"] for r in recs), "n": len(recs),
               "n_bitwise": sum(r["bitwise"] for r in recs), "by_kind": {k: [sum(r["bitwise"] for r in recs if r["kind"] == k), sum(r["kind"] == k for r in recs)] for k in R.KINDS},
               "what": "seeded sample of recombination rows captured a second time WITHOUT the vision-feature memo and compared bit for bit with the stored parts",
               "second_capture": {"gpu": gpu, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S")},
               "same_process_as_first_capture": [sum(r["first_capture"]["pid"] == os.getpid() for r in recs), len(recs)],
               "seconds": round(time.time() - t1, 1), "rows": recs}
        R.jdump(out, os.path.join(R.bb_dir(BB), "recheck_capture.json"))
        R.say(f"  RECHECK {BB}: {out['n_bitwise']}/{out['n']} rows bitwise identical on a second capture without the vision memo ({out['seconds']} s)", fh)
    R.say("DONE" + ("" if all_done else " (parts are missing: run again)"), fh)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
