"""E10 verification, block 3 (GPU): ONE equal-evidence cell (n = 10 reported D1 complaints, seed 0) on TX-G2XL-rome-T100, for
rb_suffix and for the least repair of ReScope-W (sacrificing and protecting), with the verifier's own code.

    $GQ run --mem 13000 --prio -2 --name e10v_eq -- $PY v2_equal.py
Complaints.  The verifier runs the deployed weight on ALL benign prompts of the REPAIR pool (teacher forcing of the stored pre-edit
continuation); the D1 complaints are the valid prompts whose continuation is not reproduced, in pool order; the cell's 10 complaints
are the first 10 of numpy default_rng([seed, 3]).permutation (the experiment's stated seeding rule, mirrored so that the SAME cell is
recomputed) and are compared with the experiment's out/equal/<log>/_complaints.json.
rb_suffix.  Every prefix 1..T of the log is replayed (float32 in-place writes, as the public code) and tested on the 10 complaints;
t* = the longest resolving prefix; forward passes = sequences run by the scan.  The repaired model = the replay of the first t*.
Least repair.  Library least_repair from the adapter's log-only thresholds of ReScope-W: sacrificing = guards off + on_conflict
'force'; protecting = guards on + on_conflict 'refuse', complaint by complaint.  Complaint keys: cached keys at all positions.
Evaluation of every repaired model: all requests and paraphrases of the log, the HELD-OUT sample of block 1 (300 records), the 10
complaints; every prompt through the model.  -> out/equal/<log>/<repair>__n10_s0.json (+ rows).
"""
import os
import time

import vcommon as V

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from e10t import methods as MT  # noqa: E402
from e10t.backbone import edit_in  # noqa: E402
from rescope import least_repair  # noqa: E402
import tx_public as TX  # noqa: E402

N, SEED = 10, 0
L = e10t.open_log(V.T100)
T = L.T
cf = B.C.load_counterfact()
ODIR = os.path.join(V.OUT, "equal", L.name)
W0, WE = L.weights()
TGT = {p: B.tok(" " + cf[L.records[p]]["requested_rewrite"]["target_new"]["str"].strip())["input_ids"] for p in range(T)}


@torch.no_grad()
def tf_pass(obj, tids):
    out = {}
    with B.served(obj):
        for b in B.batches([len(L.ids(t)) for t in tids]):
            bt = [tids[i] for i in b]
            tok, fin = B.tf([L.ids(t) for t in bt], [L.y0(t) for t in bt])
            for i, t in enumerate(bt):
                out[t] = (tok[i].tolist(), bool(fin[i]))
    return out


@torch.no_grad()
def full_pass(obj, tids):
    out = {}
    with B.served(obj) as srv:
        for b in B.batches([len(L.ids(t)) for t in tids]):
            bt = [tids[i] for i in b]
            seqs = [L.ids(t) for t in bt]
            tok, fin = B.tf(seqs, [L.y0(t) for t in bt])
            f1 = srv.rows()
            dtok, dfin = B.decode(seqs)
            f2 = srv.rows()
            for i, t in enumerate(bt):
                o = {"tf": tok[i].tolist(), "tf_fin": bool(fin[i]), "dec": dtok[i].tolist(), "dec_fin": bool(dfin[i])}
                if f1 is not None:
                    o.update(fired=bool(f1[i]["fired"]), rel_edits=f1[i]["rel_edits"], fired_dec=bool(f2[i]["fired"]), rel_edits_dec=f2[i]["rel_edits"])
                out[t] = o
    return out


# ----------------------------------------------------------------------------- the complaints (REPAIR pool, deployed log)
rp = [r for r in L.benign("repair")]
cpath = os.path.join(ODIR, "_complaints.json")
if os.path.exists(cpath):
    cinfo = V.jload(cpath)
else:
    res = tf_pass(WE, list(dict.fromkeys(r["text_id"] for r in rp)))
    comp_all = [r for r in rp if L.valid(r["text_id"]) and ((not res[r["text_id"]][1]) or res[r["text_id"]][0] != L.y0(r["text_id"]))]
    perm = np.random.default_rng([SEED, 3]).permutation(len(comp_all))
    mine = [comp_all[int(i)]["prompt_id"] for i in perm[:N]]
    theirs_all = V.jload(V.XOUT + f"/equal/{L.name}/_complaints.json")
    theirs = theirs_all["sets"][f"n{N}_s{SEED}"]["prompt_ids"]
    d1 = {r["prompt_id"] for r in comp_all}
    cinfo = {"n_repair_prompts": len(rp), "n_D1_complaints_under_deployed": len(comp_all), "experiment_n_D1_complaints": theirs_all["n_D1_complaints_under_deployed"],
             "mine": mine, "experiment": theirs, "same_complaints": mine == theirs, "experiment_complaints_all_in_REPAIR_pool": all(p.startswith("repair:") and p in {r["prompt_id"] for r in rp} for p in theirs),
             "experiment_complaints_are_D1_under_deployed_by_my_pass": all(p in d1 for p in theirs),
             "used": theirs if mine != theirs else mine, "note": "the experiment's set is used if the two differ (same cell), and the difference is reported"}
    V.jdump(cinfo, cpath)
V.say("complaints", {k: cinfo[k] for k in ("n_D1_complaints_under_deployed", "experiment_n_D1_complaints", "same_complaints", "experiment_complaints_are_D1_under_deployed_by_my_pass")})
by_id = {r["prompt_id"]: r for r in rp}
crows = [by_id[p] for p in cinfo["used"]]
ctids = [r["text_id"] for r in crows]
ckeys = [torch.from_numpy(np.array(L.kc.keys(t))) for t in ctids]

# ----------------------------------------------------------------------------- evaluation rows
ho_all = sorted({r["record"] for r in L.benign("heldout")})
ho_set = {ho_all[i] for i in V.sample_records(len(ho_all), 300, 2)}
ho_rows = [r for r in L.benign("heldout") if r["record"] in ho_set]
edit_rows = [r for k in ("request", "paraphrase") for r in L.rows[k]]
EVAL = edit_rows + ho_rows + [dict(r, part="complaint") for r in crows]
TIDS = list(dict.fromkeys(r["text_id"] for r in EVAL))


def evaluate(obj, label, meta):
    res = full_pass(obj, TIDS)
    rows = []
    for r in EVAL:
        t, o = r["text_id"], res[r["text_id"]]
        y0 = L.y0(t)
        row = {"prompt_id": r["prompt_id"], "set": r["set"], "kind": r["kind"], "pos": r["pos"], "case_id": r["case_id"], "part": r.get("part"),
               "D1_tf": (not o["tf_fin"]) or o["tf"] != y0, "D1_dec": (not o["dec_fin"]) or o["dec"] != y0,
               "kept_tf": float(np.mean([a == b for a, b in zip(o["tf"], y0)])) if o["tf_fin"] else 0.0, "fired": o.get("fired"), "fired_dec": o.get("fired_dec")}
        if r["set"] == "stream":
            tg = TGT[r["pos"]]
            row["new_exact"] = bool(o["dec_fin"] and o["dec"][:len(tg)] == tg)
            if o.get("fired") is not None:
                row.update(own_released=bool(edit_in(o["rel_edits"], r["pos"])), own_released_dec=bool(edit_in(o["rel_edits_dec"], r["pos"])))
        rows.append(row)
    V.save_rows(os.path.join(V.OUT, "rows_equal", L.name, f"{label}__n{N}_s{SEED}.jsonl.gz"), rows)

    def pct(sub, key, unit):
        v = [None if r.get(key) is None else float(r[key]) for r in sub]
        if not sub or all(x is None for x in v):
            return None
        b = V.crossed_bootstrap(v, [r[unit] for r in sub])
        return {"value": b["value"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "n": len(sub)}
    req = [r for r in rows if r["kind"] == "request" and r["set"] == "stream"]
    par = [r for r in rows if r["kind"] == "paraphrase" and r["set"] == "stream"]
    ho = [r for r in rows if r["set"] == "heldout"]
    co = [r for r in rows if r["part"] == "complaint"]
    st = {"request.exact": pct(req, "new_exact", "pos"), "paraphrase.exact": pct(par, "new_exact", "pos"), "heldout.D1_tf": pct(ho, "D1_tf", "case_id"),
          "heldout.kept_tf": pct(ho, "kept_tf", "case_id"), "heldout.fired": pct(ho, "fired", "case_id"),
          "complaints.D1_tf": pct(co, "D1_tf", "case_id"), "complaints.D1_dec": pct(co, "D1_dec", "case_id"), "complaints.fired": pct(co, "fired", "case_id"),
          "request.own_released_dec": pct(req, "own_released_dec", "pos"),
          "requests_whose_own_edit_is_not_released_as_served": (sum(not r["own_released_dec"] for r in req) if req and "own_released_dec" in req[0] else None)}
    V.jdump({"log": L.name, "repair": label, "n": N, "seed": SEED, "meta": meta, "stats": st, "n_rows": len(rows)}, os.path.join(ODIR, f"{label}__n{N}_s{SEED}.json"))
    V.say(label, {k: (round(v["value"], 2) if isinstance(v, dict) else v) for k, v in st.items()})


# ----------------------------------------------------------------------------- rb_suffix
if not os.path.exists(os.path.join(ODIR, f"rb_suffix__n{N}_s{SEED}.json")):
    t0 = time.time()
    f = TX.factors(L.base)
    left, right = f["left"][:T], f["right"][:T]
    W = W0.clone()
    ok_at, nfw = [0], 0
    seqs, conts = [L.ids(t) for t in ctids], [L.y0(t) for t in ctids]
    bl = B.batches([len(s) for s in seqs])
    for t in range(1, T + 1):
        W += left[t - 1].unsqueeze(1) @ right[t - 1].unsqueeze(0)
        ok = True
        with B.served(W):
            for b in bl:
                tok, fin = B.tf([seqs[i] for i in b], [conts[i] for i in b])
                nfw += len(b)
                if any((not bool(fin[j])) or tok[j].tolist() != conts[i] for j, i in enumerate(b)):
                    ok = False
                    break
        if ok:
            ok_at.append(t)
    replay_ok = bool(torch.equal(W, WE))
    ts = max(ok_at)
    Wr = W0.clone()
    for i in range(ts):
        Wr += left[i].unsqueeze(1) @ right[i].unsqueeze(0)
    meta = {"t_star": ts, "resolving_prefixes": ok_at, "n_forward": nfw, "edits_lost": T - ts, "full_replay_equals_deployed_bitwise": replay_ok, "repair_seconds": time.time() - t0}
    evaluate(Wr, "rb_suffix", meta)

# ----------------------------------------------------------------------------- least repair of ReScope-W, two forms
for form, goff in (("least_sacrificing", True), ("least_protecting", False)):
    if os.path.exists(os.path.join(ODIR, f"{form}__n{N}_s{SEED}.json")):
        continue
    M0 = MT.where(L, "logonly", allow_silence=goff)
    S0 = M0.obj
    th0 = S0.theta().clone()
    t0 = time.time()
    if goff:
        S, rec = least_repair(S0, ckeys, on_conflict="force")
        refused = 0
    else:
        S, refused = S0, 0
        for k in ckeys:
            S, rec = least_repair(S, k, on_conflict="refuse")
            refused += int(bool(rec.get("refused")))
    th1 = S.theta()
    raised = int((th1 > th0).sum())
    lowered = int((th1 < th0).sum())
    # the complaints on the cached trace after the repair: largest margin of a coordinate over its threshold
    worst = max(float((S.scores(k) - th1[None, :]).max()) for k in ckeys)
    meta = {"form": form, "guards_off_start": goff, "n_thresholds_raised": raised, "n_thresholds_lowered": lowered, "complaints_refused": refused,
            "largest_complaint_margin_over_threshold_after_repair": worst, "repair_seconds": time.time() - t0, "repair_passes": 0,
            "logonly_record": {k: M0.rec["cal"].get(k) for k in ("n_edits_capped", "n_own_keys_silenced_by_literal_rule", "allow_silence")}}
    evaluate(S, form, meta)
B.check_w0()
V.say("done")
