"""E10 task 3 (R14, R21 d): repairs at EQUAL EVIDENCE on one reported log.

    $GQ run --mem 13000 --prio -2 --name e10q_T100 -- $PY run_equal.py --log TX-G2XL-rome-T100
Evidence: n in {1, 10, 100} reported D1 complaints = benign prompts of the REPAIR pool whose pre-edit 10-token continuation the
deployed log does not reproduce (teacher-forced), in a seeded order (seeds 0, 1, 2 for n <= 10; seed 0 for n = 100; nested in n).
A complaint comes with its keys at all positions of [prompt, pre-edit continuation] (key cache).  Repairs:
  comp_min          minimum-norm correction restoring the pre-edit module output at every complaint key, unchanged at all solver keys
  rb_suffix         the longest prefix of the log under which every complaint is resolved (every prefix replayed and tested)
  rb_lineage        deletion of the edits that the log-only thresholds release on the complaints
  blocklist         deployed model + exact-match list of the reported prompts (answered by the pre-edit model)
  least_sacrificing ReScope-W, least repair from the log-only thresholds without guards (every complaint certified; edits whose own
                    request is silenced are lost)
  least_protecting  ReScope-W, the library's least repair from the log-only thresholds with guards (a complaint that cannot be
                    separated from an own request is refused)
Each repaired model: edit rows, HELD-OUT benign rows, neighbourhood rows, the reported complaints themselves, public scores.
Files: out/equal/<log>/<repair>__n<n>_s<seed>.json (+ rows, public); out/equal/<log>/_complaints.json.
"""
import argparse
import os
import time

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--sets", default="1:0,1:1,1:2,10:0,10:1,10:2,100:0")
ap.add_argument("--repairs", default="least_sacrificing,least_protecting,rb_suffix,rb_lineage,comp_min,blocklist")
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from e10t import methods as MT  # noqa: E402
import xmeth as XM  # noqa: E402
import xrepair as XR  # noqa: E402
import xeval as XE  # noqa: E402

L = e10t.open_log(args.log)
T = L.T
ses = XM.XSession(L, B)
dep = MT.deployed(L)
rp = ses.probes(dep, "repair", modes=("tf",))
comp_all = [r for r in rp if r["valid"] and r["D1_tf"]]
S0 = {True: XM.method(L, "where_logonly", guards=True).obj, False: XM.method(L, "where_logonly", guards=False).obj}
rows_by_tid = {r["text_id"]: r for r in L.benign("repair")}
cinfo = {"n_repair_prompts": len(rp), "n_D1_complaints_under_deployed": len(comp_all), "n_complaints_near_scope": sum(not r["clean"] for r in comp_all), "sets": {}}
SUB = "equal"


def lost(deleted):
    o = {"n": len(deleted)}
    if T > X.SPLIT:
        o.update(before=sum(i < X.SPLIT for i in deleted), **{"from": sum(i >= X.SPLIT for i in deleted)})
    return o


for spec in args.sets.split(","):
    n, seed = (int(x) for x in spec.split(":"))
    perm = np.random.default_rng([seed, 3]).permutation(len(comp_all))
    cs = [comp_all[int(i)] for i in perm[:n]]
    tids = [c["text_id"] for c in cs]
    crow = [rows_by_tid[t] for t in tids]
    ks = [torch.from_numpy(np.array(L.kc.keys(t))) for t in tids]
    tag0 = f"n{n}_s{seed}"
    cinfo["sets"][tag0] = {"n": n, "seed": seed, "prompt_ids": [c["prompt_id"] for c in cs], "n_near_scope": sum(not c["clean"] for c in cs),
                           "n_keys": int(sum(k.shape[0] for k in ks))}
    X.jdump(cinfo, os.path.join(X.OUT, SUB, L.name, "_complaints.json"))
    meta0 = {"n_complaints": n, "seed": seed, "evidence": f"{n} reported D1 complaints (REPAIR pool) with their keys"}
    for rep in args.repairs.split(","):
        tag = f"{rep}__{tag0}"
        if os.path.exists(os.path.join(X.OUT, SUB, L.name, tag + ".public.json")) or os.path.exists(os.path.join(X.OUT, SUB, L.name, tag + ".infeasible.json")):
            continue
        t0 = time.time()
        public = True
        if rep in ("least_sacrificing", "least_protecting"):
            g = rep == "least_protecting"
            S, rec = (XR.least_protecting if g else XR.least_sacrificing)(S0[g], ks)
            M = MT.wrap(L, S, rep, rec={"repair": rec})
            own = M.rec["own_requests"]
            nl = [int(i) for i in own["not_candidate_edits"]]
            meta = dict(meta0, repair=rec, statements_changed=rec["n_thresholds_raised"], repair_passes=0, complaints_refused=rec["n_refused"],
                        edits_lost={"n": own["n_not_candidate"], "first": nl}, repair_seconds=rec["seconds"])
            if T > X.SPLIT:
                d = e10t.gate.decide(L, S, [r["text_id"] for r in L.rows["request"]], [r["pos"] for r in L.rows["request"]], with_positions=False)
                nc = ~d["own_cand"]
                meta["edits_lost"].update(before=int(nc[:X.SPLIT].sum()), **{"from": int(nc[X.SPLIT:].sum())})
        elif rep == "rb_suffix":
            seqs, conts = [L.ids(t) for t in tids], [L.y0(t) for t in tids]
            bl = B.batches([len(s) for s in seqs])

            def resolved():
                m = 0
                for b in bl:
                    tok, fin = B.tf([seqs[i] for i in b], [conts[i] for i in b])
                    m += len(b)
                    if any((not bool(fin[j])) or tok[j].tolist() != conts[i] for j, i in enumerate(b)):
                        return False, m
                return True, m
            rec = XR.rb_suffix(L, B, resolved)
            ts = rec["t_star"]
            M = MT.weight(L, XR.replay(L, [i < ts for i in range(T)]), rep, rec={"repair": rec})
            meta = dict(meta0, repair=rec, statements_changed=T - ts, repair_passes=rec["n_forward"], edits_lost=lost(list(range(ts, T))), repair_seconds=time.time() - t0)
        elif rep == "rb_lineage":
            dele = XR.lineage(S0[True], ks)
            M = MT.weight(L, XR.replay(L, [i not in set(dele) for i in range(T)]), rep, rec={"deleted": dele})
            meta = dict(meta0, statements_changed=len(dele), repair_passes=0, edits_lost=lost(dele), repair_seconds=time.time() - t0)
        elif rep == "comp_min":
            rec = XR.comp_min(L, torch.cat(ks))
            W = rec.pop("W", None)
            if W is None:
                X.jdump(dict(meta0, repair=rec, infeasible=True, repair_seconds=rec["seconds"]), os.path.join(X.OUT, SUB, L.name, tag + ".infeasible.json"))
                X.say(f"{L.name} {tag}: infeasible ({rec['n_constraints']} constraints)")
                continue
            M = MT.weight(L, W, rep, rec={"repair": rec})
            meta = dict(meta0, repair=rec, statements_changed=1, repair_passes=0, edits_lost={"n": 0}, repair_seconds=rec["seconds"])
        elif rep == "blocklist":
            M = XM.blocklist_method(L, tids)
            meta = dict(meta0, statements_changed=n, repair_passes=0, edits_lost={"n": 0}, repair_seconds=0.0)
            public = False
        else:
            raise KeyError(rep)
        XE.evaluate(ses, M, tag, SUB, meta=meta, public=public, extra_rows=crow)
        if not public:
            X.jdump({"note": "the public test prompts are never a listed query: the public scores of the blocklist are those of the deployed log"},
                    os.path.join(X.OUT, SUB, L.name, tag + ".public.json"))
B.check_w0()
X.say("done", L.name, {k: v for k, v in ses.stats.items() if k != "mismatches"})
