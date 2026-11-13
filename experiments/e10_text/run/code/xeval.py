"""E10: one behavioural evaluation of a method instance, written to files (the format of run_behave.py).

    evaluate(ses, M, tag, sub, meta, public=True)
      out/<sub>/<log>/<tag>.json            method record, cost, session counters, `meta`
      out/rows_<sub>/<log>/<tag>.jsonl.gz   one row per prompt (edit rows, HELD-OUT benign, neighbourhood[, extra rows])
      out/public_<sub>/<log>/<tag>/...      public case files;  out/<sub>/<log>/<tag>.public.json  public summariser (all / before / from)
Resumable: finished files are not written again.
"""
import os
import time

import torch

import xcommon as X
import e10t

KEEP = ("prompt_id", "text_id", "set", "kind", "pos", "k", "case_id", "record", "clean", "near_subject", "near_mention", "near_text", "own", "valid",
        "D1_tf", "D1_dec", "tok_loc", "dec_loc", "finite_tf", "finite_dec", "src", "src_dec", "fired", "fired_dec", "n_fire_pos", "n_cand_edits",
        "fired_last_only", "near", "own_cand", "own_released", "other_released", "own_cand_dec", "own_released_dec", "other_released_dec",
        "new_exact", "new_tok", "true_exact", "true_tok", "n_target_tok")


def slim_row(L, r, part=None):
    o = {k: r[k] for k in KEEP if k in r}
    y0 = L.y0(r["text_id"])
    o["first_changed_tf"] = (not r["finite_tf"]) or r["tf_tok"][0] != y0[0]
    o["first_changed_dec"] = (not r["finite_dec"]) or r["dec_tok"][0] != y0[0]
    o["dec_first"] = r["dec_tok"][0]
    if part:
        o["part"] = part
    return o


def slim_rec(rec):
    out = {}
    for k, v in rec.items():
        if k.startswith("_") or torch.is_tensor(v):
            continue
        if isinstance(v, dict):
            v = slim_rec(v)
        if isinstance(v, list) and len(v) > 64:
            v = {"n": len(v), "first": v[:64]}
        out[k] = v
    return out


def subset_public(run_dir, records, name):
    """The public summariser on a subset of the case files of a run (symlinks, as the adapter's retention read-out)."""
    data = e10t.data.counterfact()
    src = os.path.normpath(os.path.join(os.path.dirname(run_dir), "..", os.path.basename(os.path.dirname(run_dir)) + "__" + name))
    sub = os.path.join(src, "run_000")
    os.makedirs(sub, exist_ok=True)
    for f in os.listdir(sub):
        os.remove(os.path.join(sub, f))
    for ri in records:
        f = f"case_{data[ri]['case_id']}.json"
        os.symlink(os.path.join(run_dir, f), os.path.join(sub, f))
    return e10t.public._summarise(src)


def evaluate(ses, M, tag, sub, meta=None, public=True, extra_rows=None, first=None, parts=("edits", "heldout", "neighborhood")):
    L = ses.L
    mp = os.path.join(X.OUT, sub, L.name, tag + ".json")
    if not os.path.exists(mp):
        t0 = time.time()
        before = {k: v for k, v in ses.stats.items() if k != "mismatches"}
        rows = []
        if "edits" in parts:
            rows += [slim_row(L, r) for r in ses.edits(M, first=first)]
        if "heldout" in parts:
            rows += [slim_row(L, r) for r in ses.probes(M, "heldout")]
        if "neighborhood" in parts:
            rows += [slim_row(L, r) for r in ses.probes(M, "neighborhood", first=first)]
        if extra_rows:
            rows += [slim_row(L, r, "extra") for r in ses.rows(M, extra_rows)]
        cost = {k: ses.stats[k] - before[k] for k in before}
        cost.update(seconds=time.time() - t0, n_prompts=len(rows))
        X.save_rows(os.path.join(X.OUT, "rows_" + sub, L.name, tag + ".jsonl.gz"), rows)
        X.jdump({"log": L.name, "label": M.label, "tag": tag, "draw": M.draw, "kind": M.kind, "key": M.key, "variant": M.variant, "alpha": M.alpha, "n": M.n,
                 "record": slim_rec(M.rec), "cost": cost, "mismatches": ses.stats["mismatches"][-20:] if cost["hook_mismatch"] else [], "meta": meta or {},
                 "code": e10t.code_fingerprint(), "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2 ** 20}, mp)
        X.say(f"{sub} {L.name} {tag}: {cost['sequences_run']} run, {cost['silent']} silent, {cost['from_store']} store, hook!=cache tf {cost['hook_mismatch']} dec {cost['hook_mismatch_dec']}, {cost['seconds']:.0f}s")
    pp = os.path.join(X.OUT, sub, L.name, tag + ".public.json")
    if public and not os.path.exists(pp):
        sc = e10t.public.scores(ses, M, os.path.join(X.OUT, "public_" + sub), tag=tag, first=first)
        out = {"all": sc}
        if L.T > X.SPLIT and first is None:
            out["before"] = subset_public(sc["case_files"], L.records[:X.SPLIT], f"before{X.SPLIT}")
            out["from"] = subset_public(sc["case_files"], L.records[X.SPLIT:], f"from{X.SPLIT}")
        X.jdump(out, pp)
        X.say(f"public {sub} {L.name} {tag}: ES {sc['ES']:.1f} PS {sc['PS']:.1f} NS {sc['NS']:.1f} ({sc['seconds']:.0f}s)")
