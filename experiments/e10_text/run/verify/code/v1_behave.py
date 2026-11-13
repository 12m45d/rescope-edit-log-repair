"""E10 verification, block 1 (GPU): behaviour, public scores and certificate, recomputed with the verifier's own evaluation loop.

    $GQ run --mem 13000 --prio -2 --name e10v_b100 -- $PY v1_behave.py --log TX-G2XL-rome-T100
For one reported log and calibration draw 0:
  methods     deployed (weight), preedit (control of the stored pre-edit continuations), the REPAIR (ReScope-W, two-block thresholds:
              e10t.methods.where(L, 'two_block', draw=0)) and the MIGRATION (e10t.methods.rescope(L, draw=0)), rebuilt from the log
              with the adapter; on the collapse log also with guards off (allow_silence=True)
  sample      300 random logged records (all if the log has fewer) and 300 random HELD-OUT records (seeded by the verifier)
  evaluation  EVERY sampled prompt runs through the model under the method (no sparse shortcut, no outcome store): teacher forcing
              of the stored pre-edit continuation and a 10-token greedy decode; firing flags from the library's hook
  numbers     exact match of the new object on request / paraphrase prompts (greedy decode, multi-token), D1 on HELD-OUT benign and on
              neighbourhood prompts (teacher-forced; decode variant), tokens kept, changed first tokens, false firing (hook)
  public      deployed and the repair: the public compute_rewrite_quality_counterfact on the served model -> public-format case files
              -> the public summarize.py; the verifier's own arithmetic on the same case files next to it
  certificate 200 random HELD-OUT prompts on whose cached pre-edit trace the row releases nothing (outside the decision band): logits
              of the 10-token decode under the row against the pre-edit model's, bit for bit, in the verifier's batches
Files (resumable per method): out/behave/<log>/<label>.json, out/rows/<log>/<label>.jsonl.gz, out/public/<log>/<label>/run_000/,
out/cert/<log>.json.  Nothing of the experiment's code (run/code) is imported; its outcome store is neither read nor written.
"""
import argparse
import os
import time
from pathlib import Path

import vcommon as V

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from e10t import gate as G, methods as MT  # noqa: E402
from e10t.backbone import edit_in  # noqa: E402

L = e10t.open_log(args.log)
T = L.T
collapse = T > V.SPLIT
cf = B.C.load_counterfact()
N_DEC = e10t.N_DEC

# ----------------------------------------------------------------------------- the sample
pos_sample = V.sample_records(T, 300, 1)
pos_set = set(pos_sample)
ho_all = sorted({r["record"] for r in L.benign("heldout")})
ho_sample = [ho_all[i] for i in V.sample_records(len(ho_all), 300, 2)]
ho_set = set(ho_sample)
edit_rows = [r for k in ("request", "paraphrase") for r in L.rows[k] if r["pos"] in pos_set]
nb_rows = [r for r in L.rows["neighborhood"] if r["pos"] in pos_set]
ho_rows = [r for r in L.benign("heldout") if r["record"] in ho_set]
ALL_ROWS = edit_rows + ho_rows + nb_rows
TIDS = list(dict.fromkeys(r["text_id"] for r in ALL_ROWS))
# the build's token ids are the tokenizer's ids of the prompt text; the logged request is the data set's record
n_tok_bad = sum(B.tok(r["text"])["input_ids"] != L.ids(r["text_id"]) for r in ALL_ROWS)
TGT = {}
for p in pos_sample:
    rw = cf[L.records[p]]["requested_rewrite"]
    if rw["target_new"]["str"] != L.requests[p]["target_new"]["str"] or rw["prompt"].format(rw["subject"]) != L.rows["request"][p]["text"]:
        raise RuntimeError(f"request {p} of the log is not record {L.records[p]} of the data set")
    TGT[p] = B.tok(" " + rw["target_new"]["str"].strip())["input_ids"]
sample_info = {"log": L.name, "T": T, "n_logged_records": len(pos_sample), "n_heldout_records": len(ho_sample), "n_edit_rows": len(edit_rows),
               "n_neighborhood_rows": len(nb_rows), "n_heldout_rows": len(ho_rows), "n_texts": len(TIDS), "rows_whose_token_ids_differ_from_the_tokenizer": n_tok_bad,
               "positions": pos_sample, "heldout_records": ho_sample, "heldout_rows_not_clean": sum(not r["clean"] for r in ho_rows)}
V.jdump(sample_info, os.path.join(V.OUT, "behave", L.name, "_sample.json"))
V.say(L.name, {k: v for k, v in sample_info.items() if k not in ("positions", "heldout_records")})


# ----------------------------------------------------------------------------- the evaluation loop (every prompt through the model)
@torch.no_grad()
def run(obj, tids):
    out = {}
    lens = [len(L.ids(t)) for t in tids]
    with B.served(obj) as srv:
        for b in B.batches(lens):
            bt = [tids[i] for i in b]
            seqs = [L.ids(t) for t in bt]
            tok, fin = B.tf(seqs, [L.y0(t) for t in bt])
            f_tf = srv.rows()
            dtok, dfin = B.decode(seqs)
            f_dec = srv.rows()
            for i, t in enumerate(bt):
                o = {"tf": tok[i].tolist(), "tf_fin": bool(fin[i]), "dec": dtok[i].tolist(), "dec_fin": bool(dfin[i])}
                if f_tf is not None:
                    o.update(fired=bool(f_tf[i]["fired"]), rel_edits=f_tf[i]["rel_edits"], n_fire_pos=int(f_tf[i]["n_fire_pos"]),
                             fired_dec=bool(f_dec[i]["fired"]), rel_edits_dec=f_dec[i]["rel_edits"])
                out[t] = o
    return out


def rows_of(res):
    rows = []
    for r in ALL_ROWS:
        t, o = r["text_id"], res[r["text_id"]]
        y0 = L.y0(t)
        same_tf = [a == b for a, b in zip(o["tf"], y0)]
        same_dec = [a == b for a, b in zip(o["dec"], y0)]
        row = {"prompt_id": r["prompt_id"], "text_id": t, "set": r["set"], "kind": r["kind"], "pos": r["pos"], "record": r["record"], "case_id": r["case_id"],
               "clean": r.get("clean"), "valid": L.valid(t),
               "D1_tf": (not o["tf_fin"]) or not all(same_tf), "D1_dec": (not o["dec_fin"]) or not all(same_dec),
               "kept_tf": float(np.mean(same_tf)) if o["tf_fin"] else 0.0, "kept_dec": float(np.mean(same_dec)) if o["dec_fin"] else 0.0,
               "first_changed_tf": (not o["tf_fin"]) or o["tf"][0] != y0[0], "first_changed_dec": (not o["dec_fin"]) or o["dec"][0] != y0[0],
               "dec_first": o["dec"][0], "fired": o.get("fired"), "fired_dec": o.get("fired_dec")}
        if r["set"] == "stream":
            own = r["pos"]
            if o.get("fired") is not None:
                row.update(own_released=bool(edit_in(o["rel_edits"], own)), own_released_dec=bool(edit_in(o["rel_edits_dec"], own)))
            if r["kind"] in ("request", "paraphrase"):
                tg = TGT[own]
                row["new_exact"] = bool(o["dec_fin"] and len(tg) <= N_DEC and o["dec"][:len(tg)] == tg)
                row["n_target_tok"] = len(tg)
        rows.append(row)
    return rows


def pct(rows, key, units):
    v = [None if r.get(key) is None else float(r[key]) for r in rows]
    if not rows or all(x is None for x in v):
        return {"value": None, "n": len(rows)}
    b = V.crossed_bootstrap(v, [r[units] for r in rows])
    return {"value": b["value"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "n": len(rows), "n_units": b["n_units"]}


def stats_of(rows):
    out = {}
    segs = [("", lambda r: True)] + ([(".before", lambda r: r["pos"] < V.SPLIT), (".from", lambda r: r["pos"] >= V.SPLIT)] if collapse else [])
    for kind in ("request", "paraphrase"):
        for sfx, pred in segs:
            sub = [r for r in rows if r["set"] == "stream" and r["kind"] == kind and pred(r)]
            out[f"edit.{kind}.exact{sfx}"] = pct(sub, "new_exact", "pos")
            out[f"edit.{kind}.own_released{sfx}"] = pct(sub, "own_released", "pos")
            out[f"edit.{kind}.own_released_dec{sfx}"] = pct(sub, "own_released_dec", "pos")
    for part, pred0 in (("heldout", lambda r: r["set"] == "heldout"), ("neighborhood", lambda r: r["kind"] == "neighborhood")):
        subsets = [("", lambda r: True), (".clean", lambda r: bool(r["clean"])), (".near_scope", lambda r: not r["clean"])]
        if collapse and part == "neighborhood":
            subsets += segs[1:]
        for sfx, pred in subsets:
            sub = [r for r in rows if pred0(r) and r["valid"] and pred(r)]
            for k in ("D1_tf", "D1_dec", "kept_tf", "first_changed_tf", "fired", "fired_dec"):
                out[f"{part}{sfx}.{k}"] = pct(sub, k, "case_id")
    # behavioural certificate on the sample: a prompt on which the hook released nothing in a pass reproduces the pre-edit tokens
    gate = [r for r in rows if r.get("fired") is not None]
    if gate:
        out["silent_tf.n"] = sum(not r["fired"] for r in gate)
        out["silent_tf.n_D1"] = sum((not r["fired"]) and r["D1_tf"] for r in gate)
        out["silent_dec.n"] = sum(not r["fired_dec"] for r in gate)
        out["silent_dec.n_D1"] = sum((not r["fired_dec"]) and r["D1_dec"] for r in gate)
    return out


def slim(rec):
    cal = rec.get("cal") or {}
    own = rec.get("own_requests") or {}
    ce = cal.get("capped_edits") or []
    return {"eta": cal.get("eta"), "n_edits_capped": cal.get("n_edits_capped"), "capped_edits_first": list(ce)[:30], "guard_margin": cal.get("guard_margin"),
            "allow_silence": cal.get("allow_silence"), "own_requests": {k: own.get(k) for k in ("n", "n_not_candidate", "n_not_released", "n_other_released", "n_in_band")},
            "own_not_candidate_before_split": (sum(i < V.SPLIT for i in own.get("not_candidate_edits", [])) if collapse else None),
            "thresholds": rec.get("thresholds"), "draw": rec.get("draw"), "gate": rec.get("gate"), "fingerprint": rec.get("fingerprint")}


# ----------------------------------------------------------------------------- the methods, rebuilt from the log with the adapter
def build(label):
    if label == "deployed":
        return L.weights()[1], {"kind": "weight", "what": "deployed weight W_E of the log"}
    if label == "preedit":
        return None, {"kind": "weight", "what": "pre-edit weight W_0"}
    base, goff = (label[:-5], True) if label.endswith("@goff") else (label, False)
    kw = dict(alpha=V.ALPHA, n=V.N_CAL, draw=0, **({"allow_silence": True} if goff else {}))
    M = MT.where(L, "two_block", **kw) if base == "where_twoblock" else MT.rescope(L, **kw)
    d = M.rec["draw"]
    pool_ids = {r["prompt_id"] for r in L.benign("repair")}
    info = {"kind": "gate", "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw, "rec": slim(M.rec),
            "calibration_prompts_all_in_REPAIR_pool": all(p in pool_ids and p.startswith("repair:") for p in d["prompt_ids"]), "n_calibration_prompts": len(d["prompt_ids"])}
    return M.obj, info


LABELS = ["deployed", "preedit", "where_twoblock", "rescope"] + (["where_twoblock@goff", "rescope@goff"] if collapse else [])
GATES = {}
for lab in LABELS:
    mp = os.path.join(V.OUT, "behave", L.name, lab + ".json")
    obj, info = build(lab)
    if info["kind"] == "gate":
        GATES[lab] = obj
    if os.path.exists(mp):
        continue
    t0 = time.time()
    res = run(obj, TIDS)
    rows = rows_of(res)
    V.save_rows(os.path.join(V.OUT, "rows", L.name, lab + ".jsonl.gz"), rows)
    V.jdump({"log": L.name, "label": lab, "info": info, "stats": stats_of(rows), "n_rows": len(rows), "seconds": time.time() - t0,
             "code": e10t.code_fingerprint(), "evaluation": "every sampled prompt through the model; verifier's batches; no store"}, mp)
    V.say(f"behave {L.name} {lab}: {len(TIDS)} texts, {time.time() - t0:.0f}s")

# ----------------------------------------------------------------------------- public scores: deployed and the repair
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact  # noqa: E402
from experiments.summarize import main as public_summarize  # noqa: E402

PUB_LABELS = ["deployed", "where_twoblock"] + (["where_twoblock@goff"] if collapse else [])
recs = [L.records[p] for p in pos_sample]


def own_arithmetic(run_dir, case_ids):
    es, ps, ns, nonfin = [], [], [], 0
    for c in case_ids:
        post = V.jload(os.path.join(run_dir, f"case_{c}.json"))["post"]
        es.append(float(np.mean([x["target_true"] > x["target_new"] for x in post["rewrite_prompts_probs"]])))
        ps.append(float(np.mean([x["target_true"] > x["target_new"] for x in post["paraphrase_prompts_probs"]])))
        ns.append(float(np.mean([x["target_true"] < x["target_new"] for x in post["neighborhood_prompts_probs"]])))
        vals = [x[k] for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs") for x in post[key] for k in ("target_new", "target_true")]
        nonfin += int(not np.all(np.isfinite(vals)))
    o = {"n": len(case_ids), "n_records_with_nonfinite_prob": nonfin}
    for k, v in (("ES", es), ("PS", ps), ("NS", ns)):
        b = V.crossed_bootstrap(v, case_ids)
        o[k] = {"value": b["value"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "n_units": b["n_units"]}
    return o


def summarise(parent):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        s = public_summarize(Path(parent), None, abs_path=True)
    if len(s) != 1:
        raise RuntimeError(f"{parent}: {len(s)} runs")
    s = s[0]
    return {"n": int(s["num_cases"]), "ES": float(s["post_rewrite_success"][0]), "PS": float(s["post_paraphrase_success"][0]), "NS": float(s["post_neighborhood_success"][0]),
            "EM": float(s["post_rewrite_diff"][0]), "PM": float(s["post_paraphrase_diff"][0]), "NM": float(s["post_neighborhood_diff"][0])}


def subset_dir(parent, name, case_ids):
    src = parent + "__" + name
    sub = os.path.join(src, "run_000")
    os.makedirs(sub, exist_ok=True)
    for f in os.listdir(sub):
        os.remove(os.path.join(sub, f))
    for c in case_ids:
        os.symlink(os.path.join(parent, "run_000", f"case_{c}.json"), os.path.join(sub, f"case_{c}.json"))
    return src


for lab in PUB_LABELS:
    pp = os.path.join(V.OUT, "behave", L.name, lab + ".public.json")
    if os.path.exists(pp):
        continue
    t0 = time.time()
    parent = os.path.join(V.OUT, "public", L.name, lab)
    run_dir = os.path.join(parent, "run_000")
    os.makedirs(run_dir, exist_ok=True)
    obj = GATES[lab] if lab in GATES else L.weights()[1]
    n_fb = 0
    with B.served(obj):
        for ri in recs:
            cp = os.path.join(run_dir, f"case_{cf[ri]['case_id']}.json")
            if os.path.exists(cp):
                continue
            try:
                post = compute_rewrite_quality_counterfact(B.model, B.tok, cf[ri], None, None)
            except torch.cuda.OutOfMemoryError:
                post, _ = B.C.public_prob_test(B.model, B.tok, cf[ri])     # the builder's wrapper: the same public call with use_cache=False
                n_fb += 1
            V.jdump({"case_id": cf[ri]["case_id"], "requested_rewrite": cf[ri]["requested_rewrite"], "time": 0.0, "post": post}, cp)
    cids = [cf[ri]["case_id"] for ri in recs]
    out = {"label": lab, "n_records": len(recs), "case_files": run_dir, "n_use_cache_false": n_fb, "public_summariser": {"all": summarise(parent)}, "own_arithmetic": {"all": own_arithmetic(run_dir, cids)}}
    if collapse:
        for nm, sel in (("before", [c for c, p in zip(cids, pos_sample) if p < V.SPLIT]), ("from", [c for c, p in zip(cids, pos_sample) if p >= V.SPLIT])):
            out["public_summariser"][nm] = summarise(subset_dir(parent, nm, sel))
            out["own_arithmetic"][nm] = own_arithmetic(run_dir, sel)
    out["seconds"] = time.time() - t0
    V.jdump(out, pp)
    V.say(f"public {L.name} {lab}: {out['public_summariser']['all']} {time.time() - t0:.0f}s")

# ----------------------------------------------------------------------------- certificate check on 200 certified prompts per row
cp = os.path.join(V.OUT, "cert", L.name + ".json")
if not os.path.exists(cp):
    ho_tids = list(dict.fromkeys(r["text_id"] for r in L.benign("heldout")))
    cert = {"log": L.name, "n_heldout_texts": len(ho_tids), "rows": {}}
    for lab, S in GATES.items():
        d = G.decide(L, S, ho_tids, with_positions=False)
        certified = [t for t, f, nr in zip(ho_tids, d["fired"], d["near"]) if not f and not nr]
        o = {"n_fired_on_cache": int(d["fired"].sum()), "n_in_band": int(d["near"].sum()), "n_certified": len(certified)}
        pick = [certified[i] for i in V.sample_records(len(certified), 200, 3)]
        # the verifier's own reading of 'nothing is released on the cached pre-edit trace': library scores of the cached keys against the thresholds
        Sf = S
        th = Sf.theta()
        n_own_bad, worst = 0, -float("inf")
        for t in pick:
            R = Sf.scores(torch.from_numpy(np.array(L.kc.keys(t))))
            m = float((R - th[None, :]).max())
            worst = max(worst, m)
            n_own_bad += int(m >= 0)
        o.update(n_checked=len(pick), n_checked_not_silent_by_own_arithmetic=n_own_bad, largest_margin_over_threshold=worst if pick else None)
        n_eq, n_tok_eq, n_hook, viol = 0, 0, 0, []
        lens = [len(L.ids(t)) for t in pick]
        for b in B.batches(lens):
            bt = [pick[i] for i in b]
            seqs = [L.ids(t) for t in bt]
            with B.served(None):
                tok0, fin0, lg0 = B.decode(seqs, return_logits=True)
            with B.served(S) as srv:
                tok1, fin1, lg1 = B.decode(seqs, return_logits=True)
                fl = srv.rows()
            for j, t in enumerate(bt):
                eq = bool(torch.equal(lg1[j], lg0[j]))
                n_eq += int(eq)
                n_tok_eq += int(tok1[j].tolist() == L.y0(t))
                n_hook += int(fl[j]["fired"])
                if not eq:
                    viol.append({"text_id": t, "hook_fired": bool(fl[j]["fired"]), "max_abs_logit_diff": float((lg1[j] - lg0[j]).abs().max())})
            del lg0, lg1
        o.update(n_logits_bit_identical=n_eq, n_violations=len(viol), violations=viol[:20], n_tokens_equal_stored_continuation=n_tok_eq, n_hook_fired=n_hook)
        cert["rows"][lab] = o
        V.say(f"cert {L.name} {lab}: {o['n_certified']} certified, checked {o['n_checked']}, bit-identical {n_eq}, violations {len(viol)}")
    V.jdump(cert, cp)
B.check_w0()
V.say("done", L.name)
