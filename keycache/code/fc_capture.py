"""GPU capture of one stream family of a backbone: keys at the last prompt position (teacher forcing and decoding),
pre-edit teacher-forced flags, decoder-v2 pre-edit answers of the new hard negatives and of the benign subset.

    python fc_capture.py <backbone> <family> [--device cuda:0] [--mem_gb 10.3] [--shard i/n] [--part 100]
                         [--groups rel,probe,...] [--limit N] [--tag smoke]

Resumable: the sequences of the family (work/<backbone>/select/manifest.jsonl) are cut into parts of --part
sequences per group; a finished part is one file work/<backbone>/parts[_tag]/<family>_<group>_<nnnn>.pt and is
skipped on restart.  --shard i/n processes the parts with (running number % n) == i (two cards in parallel).

Everything goes through the library (rescope.Evaluator of the family's MASTER log, loaded with rescope.load):
  model state      the PRE-EDIT model for the whole job (Evaluator.applied(H.W0)); the keys do not depend on the
                   edited weight, the flags and the decoder-v2 answers are pre-edit quantities;
  teacher forcing  Evaluator.batch(it) / capture(it, True): the item's own B = 1 sequence with its answer (edit
                   target, or the decoder-v2 pre-edit answer y0), key at the harness index answer_start - 1;
                   where lp_shift != 0 also the key of the same sequence at the decode's token (lp = 'prompt');
  decoding         Evaluator.decode_keys(it): the key at the prompt's own last token in the 11 full forwards a
                   decoder-v2 decode can run (prompt + k ids, k = 0..10); k = 0 is the prompt-only sequence;
  flags            Evaluator.tf(it): exact / token accuracy of the answer under the pre-edit model, B = 1;
  decoder v2       tb_common.Prober.decode_v2 + tf_state, as 08_rescope/p11_retrofit/code/hn_build.py:
                   y0 = 10 greedy ids from full no-cache forwards cut at the first newline, 48-token long answer,
                   valid = non-empty AND reproduced by the pre-edit model under teacher forcing.
The harness' vision-feature memo is active (as in Evaluator._each): the forwards of one item share the image
features of its first forward.  fc_verify.py re-captures a seeded sample WITHOUT the memo in a fresh process.
Text rows of the benign pool need one addition to the library: Evaluator._query has no text-only item besides probe
rows, so a 4-line subclass maps an item {'_row': {'ptype': 'text', 'prompt': ...}} to the Prober's text batch.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_common as C  # noqa: E402

GROUPS = ("rel", "probe", "benign", "hn_i", "hn_q", "hn_x", "tgen", "igen", "editpre")


def make_item(m, y0=None):
    """The library item (Evaluator conventions) of a manifest sequence."""
    if m["lib_kind"] == "edit":
        return {"kind": "edit", "sid": m["lib_sid"], "metric": m["metric"], "edit": int(m["edit"]), "uid": m["edit_uid"]}
    if m["lib_kind"] == "probe":
        return {"kind": "probe", "sid": m["lib_sid"], "probe_idx": int(m["probe_idx"]), "ptype": m["ptype"], "uid": m["uid"]}
    ans = m["answer"] if y0 is None else y0
    it = {"kind": m["lib_kind"], "sid": m["sid"], "_row": {"image": m["image"], "prompt": m["prompt"], "y0_text": ans or "",
                                                           "ptype": m["ptype"]}}
    if m["lib_kind"] == "hn":
        it.update(hn_id=m["hn_id"], hn_type=m["hn_type"], edit=int(m["edit"]))
    return it


def make_evaluator(rs, H):
    class CacheEvaluator(rs.Evaluator):
        """rescope.Evaluator + text-only free queries (the library's `_row` items are image rows)."""

        def _query(self, it):
            r = it.get("_row")
            if r is not None and r.get("ptype") == "text":
                return {"loc": r["prompt"]}, "text", r["y0_text"]
            return super()._query(it)
    return CacheEvaluator(H, hard_negatives=False)


def plan_parts(man, family, part, groups):
    """[(part name, [manifest rows])] of a family in the fixed processing order."""
    out = []
    for g in groups:
        sub = [m for m in man if m["family"] == family and m["kind"] == g]
        for j in range(0, len(sub), part):
            out.append((f"{family}_{g}_{j // part:04d}", sub[j:j + part]))
    return out


def unique_rows(dk):
    """(distinct rows of dk in order of first occurrence, map (n,) row -> index of its distinct row), bitwise."""
    import torch
    keep, mp = [], []
    for k in range(dk.shape[0]):
        for u, j in enumerate(keep):
            if torch.equal(dk[k], dk[j]):
                mp.append(u)
                break
        else:
            mp.append(len(keep))
            keep.append(k)
    return dk[keep], mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone")
    ap.add_argument("family")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_gb", type=float, default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--part", type=int, default=100)
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N sequences of every group")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reverse", action="store_true", help="take the parts from the end (a second worker on the same family)")
    args = ap.parse_args()
    BB, FAM = args.backbone, args.family
    cfg = C.BACKBONES[BB]
    master = cfg["families"][FAM]
    PD = os.path.join(C.WORK, BB, "parts" + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(PD, exist_ok=True)
    os.makedirs(C.LOGDIR, exist_ok=True)
    fh = open(os.path.join(C.LOGDIR, f"capture_{BB}_{FAM}{'_' + args.tag if args.tag else ''}.log"), "a")
    C.say(f"==== fc_capture {vars(args)} pid {os.getpid()} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", fh)
    man = C.jsonl_read(os.path.join(C.WORK, BB, "select", "manifest.jsonl"))
    groups = [g for g in args.groups.split(",") if g]
    if args.limit:
        keep, seen = [], {}
        for m in man:
            k = (m["family"], m["kind"])
            seen[k] = seen.get(k, 0) + 1
            if seen[k] <= args.limit:
                keep.append(m)
        man = keep
    parts = plan_parts(man, FAM, args.part, [g for g in groups if g != "editpre"])
    hn_edits = C.jload(os.path.join(C.WORK, BB, "select", "hn_edits.json"))
    if "editpre" in groups and FAM == cfg["vlkeb"]:                 # decoder-v2 answers of the later edits' own prompts
        ids = hn_edits["later100"][:args.limit] if args.limit else hn_edits["later100"]
        for j in range(0, len(ids), 25):
            parts.append((f"{FAM}_editpre_{j // 25:04d}", [{"sid": f"editpre:{i}", "kind": "editpre", "edit": i} for i in ids[j:j + 25]]))
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = [(n, p, rows) for n, (p, rows) in enumerate(parts) if n % sn == si and not os.path.exists(os.path.join(PD, p + ".pt"))]
    if args.reverse:
        todo = todo[::-1]
    C.say(f"  {len(parts)} parts, {len(todo)} to do in shard {args.shard} ({sum(len(r) for _, _, r in todo)} sequences)", fh)
    if not todo:
        C.say("DONE (nothing to do)", fh)
        return 0

    import torch
    C.tb()                                                              # registers the long logs (scale_registry) before the harness loads one
    rs = C.import_rescope()
    t0 = time.time()
    H = rs.load(master, device=args.device, mem_gb=args.mem_gb or cfg["mem_gb"])
    ev = make_evaluator(rs, H)
    hz = rs.harness()
    assert H.backbone == BB and H.L == cfg["layer"], (H.backbone, H.L)
    assert all(H.checks[k] for k in H.checks if k != "collapsed_working_point"), H.checks
    fam = C.family(FAM)
    assert [d["uid"] for d in H.items] == [d["uid"] for d in fam["items"]] and [p["uid"] for p in H.probes] == [p["uid"] for p in fam["probes"]]
    C.say(f"  loaded {master} in {time.time() - t0:.0f} s: T = {H.T}, layer {H.L}, dtype {H.dtype}, d_in {H.d_in}, checks {H.checks}; "
          f"GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", fh)
    prober = H.prober

    def answer_ids(q, ptype, text):
        if not text:
            return []
        b = prober.batch(q, ptype, answer=text)
        return b["answer_ids"][0][b["answer_mask"][0]].tolist()

    def decode_v2(q, ptype):
        """hn_build.decode: decoder-v2 pre-edit answer + its teacher-forced self-check (pre-edit model)."""
        y0, lt, info = prober.decode_v2(q, ptype)
        note = None
        try:
            aid = answer_ids(q, ptype, y0)
            ok, n, mism = prober.tf_state(q, ptype, y0)
        except RuntimeError as e:                                       # y0 without a scorable tokenisation: kept, invalid
            aid, ok, n, mism, note = [], None, 0, None, f"{type(e).__name__}: {e}"[:200]
        valid = bool(y0) and ok is True
        return {"y0_text": y0, "y0_ids": aid, "y0_gen_ids": info["gen_ids"], "y0_stop": info["stop"], "y0_raw": info["raw"],
                "y0_roundtrip_ids_equal": aid == info["gen_ids"][:len(aid)], "w0_reproduced": ok, "w0_n_ans_tok": n,
                "w0_first_mismatch": mism, "y0_valid": valid,
                "y0_invalid_reason": None if valid else ("empty" if not y0 else ("untokenisable" if note else
                                                                                 ("w0_nonfinite" if ok is None else "w0_not_reproduced"))),
                "y0_note": note, "long_text": lt, "long_ids": info["long_ids"], "long_stop": info["long_stop"]}

    def one(m):
        rec = {"sid": m["sid"]}
        y0 = None
        it = make_item(m)
        if m["decode"]:
            q, ptype, _ = ev._query(it)
            rec.update(decode_v2(q, ptype))
            y0 = rec["y0_text"]
            it = make_item(m, y0=y0)
        ans = y0 if m["decode"] else m["answer"]
        # decode path: the prompt's own last token in the 11 full forwards of a decoder-v2 decode
        bq = ev.batch(it, answer=False)
        dk = ev.decode_keys(it)
        du, dmap = unique_rows(dk)
        rec.update(lp_decode=int(bq["last_prompt_idx"][0]), len_prompt=int(bq["seq_len"][0]),
                   n_image_tokens=int(bq["n_image_tokens"][0]), decode_map=dmap, n_decode_distinct=int(du.shape[0]))
        out = {"dec": du.cpu(), "tf": None, "tfp": None}
        rec.update(has_tf=False, has_tf_prompt=False, lp_tf=None, lp_shift=None, len_tf=None, tf_note=None)
        if ans:
            try:
                b = ev.batch(it, True)
            except RuntimeError as e:                                   # an answer without a scorable tokenisation
                b, rec["tf_note"] = None, f"{type(e).__name__}: {e}"[:200]
            if b is not None and int(b["answer_mask"][0].sum()) == 0:
                b, rec["tf_note"] = None, "no answer tokens"
            if b is not None:
                x, p = ev.capture(it, True, batch=b)
                sh = ev.lp_shift(it)
                out["tf"] = x[p].cpu()
                if sh:
                    out["tfp"] = x[p - sh].cpu()
                row = ev.tf(it, b)                                      # pre-edit model: is the answer already produced
                rec.update(has_tf=True, lp_tf=int(p), lp_shift=sh, len_tf=int(b["seq_len"][0]), has_tf_prompt=bool(sh),
                           pre_exact=bool(row["exact"]), pre_token_acc=float(row["token_acc"]), pre_n_tok=int(row["n_tok"]),
                           pre_finite=bool(row["finite"]), pre_first_mismatch=row["first_mismatch"])
        return rec, out

    def editpre(i):
        """Decoder-v2 pre-edit answers of an edit's own prompts (rel / tgen / igen), as hn_build's edit_pre.jsonl."""
        d = H.items[i]
        rec = {"sid": f"editpre:{i}", "edit": i, "edit_uid": d["uid"], "alt": d["alt"], "pred_text": d["pred_text"]}
        for mname, (img, q) in (("rel", (d["image"], d["src"])), ("tgen", (d["image"], d["rephrase"])), ("igen", (d["image_rephrase"], d["src"]))):
            r = decode_v2({"m_loc": img, "m_loc_q": q}, "image")
            rec[mname] = {k: r[k] for k in ("y0_text", "long_text", "y0_valid", "y0_stop", "long_stop")}
        return rec

    def claim(pname):
        """True if this process may compute the part: it is not finished and no LIVE process has claimed it.  Several
        workers can therefore run on one family at any time (a claim of a dead process is taken over)."""
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

    n_done = 0
    for n, pname, rows in todo:
        if not claim(pname):
            continue
        t1 = time.time()
        recs, K_tf, K_tfp, K_dec = [], [], [], []
        with torch.no_grad(), ev.applied(H.W0), hz.HE.vision_cache(H):
            assert torch.equal(H.mod.weight.data, H.W0)
            for m in rows:
                ev.trim_vcache()
                if m["kind"] == "editpre":
                    recs.append(editpre(m["edit"]))
                    continue
                rec, out = one(m)
                rec["dec_row0"] = sum(k.shape[0] for k in K_dec)
                K_dec.append(out["dec"])
                if out["tf"] is not None:
                    rec["tf_row"] = len(K_tf)
                    K_tf.append(out["tf"])
                if out["tfp"] is not None:
                    rec["tfp_row"] = len(K_tfp)
                    K_tfp.append(out["tfp"])
                recs.append(rec)
        d_in = H.d_in
        blob = {"part": pname, "sids": [m["sid"] for m in rows], "rows": recs, "master": master, "dtype": str(H.dtype),
                "K_tf": torch.stack(K_tf) if K_tf else torch.zeros(0, d_in, dtype=H.dtype),
                "K_tfp": torch.stack(K_tfp) if K_tfp else torch.zeros(0, d_in, dtype=H.dtype),
                "K_dec": torch.cat(K_dec) if K_dec else torch.zeros(0, d_in, dtype=H.dtype),
                "seconds": time.time() - t1, "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = os.path.join(PD, pname + f".tmp{os.getpid()}")
        torch.save(blob, tmp)
        os.replace(tmp, os.path.join(PD, pname + ".pt"))
        try:
            os.remove(os.path.join(PD, pname + ".claim"))
        except FileNotFoundError:
            pass
        n_done += 1
        C.say(f"  part {pname} ({len(rows)} seq, {time.time() - t1:.0f} s, {(time.time() - t1) / max(1, len(rows)):.2f} s/seq; "
              f"{n_done}/{len(todo)} parts; GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB)", fh)
    C.say("DONE", fh)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
