"""E10 behaviour on one reported log: metrics (i), (ii), (iii) for every row and calibration draw.

    $GQ run --mem 13000 --prio -2 --name e10b_T100 -- $PY run_behave.py --log TX-G2XL-rome-T100
Rows (R19 f / R21 / R22): deployed, preedit, where_logonly, where_twoblock, rescope, cosine, cosine_published, exact, the ONE
gate of the R3 family named in out/best_gate_text.json, lookup_subject.  Calibrated rows: draws 0, 1, 2.  On the collapse log
(T > 173) every calibrated row and where_logonly also with guards off (label '<row>@goff').
Per method instance (tag = <label>__d<draw>), resumable (a finished instance is skipped):
    out/behave/<log>/<tag>.json           method record (thresholds, capped edits, own requests), cost, session counters
    out/rows/<log>/<tag>.jsonl.gz         one row per prompt: edit rows (request, paraphrase), HELD-OUT benign, neighbourhood
    out/public/<log>/<tag>/run_000/       public-format case files (public probability test on the served model)
    out/behave/<log>/<tag>.public.json    public summariser on all records (and, for T > 173, on the records before / from 173)
Numbers are computed from these files by summarise.py; nothing is selected here.
"""
import argparse
import os
import time

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--draws", default="0,1,2")
ap.add_argument("--rows", default="")
ap.add_argument("--no-public", action="store_true")
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xmeth as XM  # noqa: E402

best = X.jload(X.OUT + "/best_gate_text.json")["selected"]
L = e10t.open_log(args.log)
ses = XM.XSession(L, B)
collapse = L.T > X.SPLIT
draws = [int(d) for d in args.draws.split(",")]
PLAIN = ["deployed", "preedit", "where_logonly", "cosine_published", "exact", "lookup_subject"]
CALIB = ["where_twoblock", "rescope", "cosine", best]
plan = [(lab, None, True) for lab in PLAIN] + [(lab, d, True) for lab in CALIB for d in draws]
if collapse:
    plan += [("where_logonly", None, False)] + [(lab, d, False) for lab in CALIB for d in draws]
if args.rows:
    want = set(args.rows.split(","))
    plan = [p for p in plan if p[0] in want]
KEEP = ("prompt_id", "text_id", "set", "kind", "pos", "k", "case_id", "record", "clean", "near_subject", "near_mention", "near_text", "own", "valid",
        "D1_tf", "D1_dec", "tok_loc", "dec_loc", "finite_tf", "finite_dec", "src", "src_dec", "fired", "fired_dec", "n_fire_pos", "n_cand_edits",
        "fired_last_only", "near", "own_cand", "own_released", "other_released", "own_cand_dec", "own_released_dec", "other_released_dec",
        "new_exact", "new_tok", "true_exact", "true_tok", "n_target_tok")


def slim_row(r):
    o = {k: r[k] for k in KEEP if k in r}
    y0 = L.y0(r["text_id"])
    o["first_changed_tf"] = (not r["finite_tf"]) or r["tf_tok"][0] != y0[0]
    o["first_changed_dec"] = (not r["finite_dec"]) or r["dec_tok"][0] != y0[0]
    o["dec_first"] = r["dec_tok"][0]
    return o


def slim_rec(rec):
    out = {}
    for k, v in rec.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            v = slim_rec(v)
        if isinstance(v, list) and len(v) > 64:
            v = {"n": len(v), "first": v[:64]}
        out[k] = v
    return out


def subset_public(run_dir, records, name):
    """The public summariser on a subset of the case files of a run (symlinks, as the adapter does for its retention read-out)."""
    data = e10t.data.counterfact()
    src = os.path.join(os.path.dirname(run_dir), "..", os.path.basename(os.path.dirname(run_dir)) + "__" + name)
    src = os.path.normpath(src)
    sub = os.path.join(src, "run_000")
    os.makedirs(sub, exist_ok=True)
    for f in os.listdir(sub):
        os.remove(os.path.join(sub, f))
    for ri in records:
        f = f"case_{data[ri]['case_id']}.json"
        os.symlink(os.path.join(run_dir, f), os.path.join(sub, f))
    return e10t.public._summarise(src)


for lab, d, guards in plan:
    M = XM.method(L, lab, 0 if d is None else d, guards)
    tag = f"{M.label}__d{'x' if d is None else d}"
    mp = os.path.join(X.OUT, "behave", L.name, tag + ".json")
    if not os.path.exists(mp):
        t0 = time.time()
        before = {k: v for k, v in ses.stats.items() if k != "mismatches"}
        e = ses.edits(M)
        p = ses.probes(M, "heldout")
        nb = ses.probes(M, "neighborhood")
        cost = {k: ses.stats[k] - before[k] for k in before}
        cost.update(seconds=time.time() - t0, n_prompts=len(e) + len(p) + len(nb))
        X.save_rows(os.path.join(X.OUT, "rows", L.name, tag + ".jsonl.gz"), [slim_row(r) for r in e + p + nb])
        X.jdump({"log": L.name, "label": M.label, "base_label": lab, "draw": d, "guards": guards, "kind": M.kind, "key": M.key, "variant": M.variant,
                 "alpha": M.alpha, "n": M.n, "record": slim_rec(M.rec), "cost": cost, "mismatches": ses.stats["mismatches"][-20:] if cost["hook_mismatch"] else [],
                 "code": e10t.code_fingerprint(), "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2 ** 20}, mp)
        X.say(f"behave {L.name} {tag}: {cost['sequences_run']} run, {cost['silent']} silent, {cost['from_store']} store, hook!=cache tf {cost['hook_mismatch']} dec {cost['hook_mismatch_dec']}, {cost['seconds']:.0f}s")
    pp = os.path.join(X.OUT, "behave", L.name, tag + ".public.json")
    if not args.no_public and not os.path.exists(pp):
        sc = e10t.public.scores(ses, M, os.path.join(X.OUT, "public"), tag=tag)
        out = {"all": sc}
        if collapse:
            out["before"] = subset_public(sc["case_files"], L.records[:X.SPLIT], f"before{X.SPLIT}")
            out["from"] = subset_public(sc["case_files"], L.records[X.SPLIT:], f"from{X.SPLIT}")
        X.jdump(out, pp)
        X.say(f"public {L.name} {tag}: ES {sc['ES']:.1f} PS {sc['PS']:.1f} NS {sc['NS']:.1f} ({sc['seconds']:.0f}s)")
B.check_w0()
X.jdump({k: v for k, v in ses.stats.items()}, os.path.join(X.OUT, "behave", L.name, "_session.json"))
X.say("done", L.name, {k: v for k, v in ses.stats.items() if k != "mismatches"})
