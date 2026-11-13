"""E10 / R22: relational maintenance workload on one text log (TX-G2XL-rome-T100 / -T500).

    $GQ run --mem 13000 --prio -2 --name e10r_T100 -- $PY run_r22.py --log TX-G2XL-rome-T100
Relation Facts over the log's requests (edited tuples) and the VIEW pool (xview.view_pool); value of a tuple under a served
model = first token of the greedy continuation of its prompt.  Every state and every repaired model is a set of rows of the
table val of out/r22/<log>.sqlite; every read-out is a statement of r22.sql.  Output: out/r22/<log>.json (rows with their
evidence, scores, costs), out/r22/<log>_flags.json (per-tuple flags for intervals), out/r22/<log>.sqlite.
Resumable: the values of every model are kept in out/r22/<log>_values/<model>.json.

Evidence contract (R22 d).  A complaint is a report-half query (selection view or aggregate of a relation) whose result on the
deployed model differs from its result on R*; the complainant supplies the query only.  Diagnosis (charged: two forward passes
per tuple of the complained view: pre-edit and deployed): the tuples whose value changed and whose prompt is not the own request
of a logged edit (a prompt position carries a logged solver key: |k - k_i| <= 1e-4 |k_i|) become the tuple-level complaints.
Every repair receives exactly these tuples (prompt, pre-edit value, and the keys of the prompt positions, which the
diagnosis' pre-edit pass yields at no further pass).  Further passes of a repair (prefix replays of rb_suffix) are counted.
"""
import argparse
import os
import re
import sqlite3
import time

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--draws", default="0,1,2")
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from e10t import gate as G, methods as MT  # noqa: E402
import xmeth as XM  # noqa: E402
import xrepair as XR  # noqa: E402
import xview as XV  # noqa: E402

t_start = time.time()
L = e10t.open_log(args.log)
T = L.T
collapse = T > X.SPLIT
pool = XV.view_pool()
XV.build_store(B)
VS = XV.ViewStore()
VL = XV.VLog(L, VS)
cf = e10t.data.counterfact()
ODIR = os.path.join(X.OUT, "r22")
VDIR = os.path.join(ODIR, L.name + "_values")
os.makedirs(VDIR, exist_ok=True)
edited = list(L.records)
report, test = list(pool["report"]), list(pool["test"])
ALL = edited + report + test
EVAL = edited + test                                                    # tuples on which a repaired model is materialised
ids = lambda r: VS.ids[XV.VBASE + r]                                    # noqa: E731
vt = lambda rs: [XV.VBASE + r for r in rs]                              # noqa: E731
draws = [int(d) for d in args.draws.split(",")]
CHUNK = 32


def batches(rs):
    out = []
    for b in B.batches([len(ids(r)) for r in rs], extra=1):
        out += [b[a:a + CHUNK] for a in range(0, len(b), CHUNK)]
    return out


@torch.no_grad()
def serve(obj, rs, check=None):
    """First tokens of the prompts rs under the served object (every prompt through the model).  check: {rid: predicted firing}."""
    vals, mism = {}, 0
    if not rs:
        return vals, mism
    with B.served(obj) as srv:
        for b in batches(rs):
            tok, fin = B.decode([ids(rs[i]) for i in b], n=1, feed_last=False)
            flags = srv.rows()
            for j, i in enumerate(b):
                vals[rs[i]] = int(tok[j, 0]) if bool(fin[j]) else -1
                if check is not None and flags is not None and check.get(rs[i]) is not None and bool(flags[j]["fired"]) != check[rs[i]]:
                    mism += 1
    return vals, mism


def values(name, kind, obj, rs, pre=None):
    """{rid: value} of a model on the tuples rs (cached on disk).  kind: 'weight' | 'gate' | 'lookup'.  A gate / the keyed store
    runs the model only where something is released on the prompt's pre-edit trace; elsewhere the value is the pre-edit value."""
    p = os.path.join(VDIR, re.sub(r"[^A-Za-z0-9_.@=-]", "_", name) + ".json")
    if os.path.exists(p):
        c = X.jload(p)
        if c["rids"] == rs:
            return dict(zip(c["rids"], c["values"])), c["info"]
    t0 = time.time()
    info = {"n": len(rs)}
    if kind == "weight":
        vals, _ = serve(obj, rs)
        info.update(n_forward=len(rs))
    elif kind == "gate":
        d = G.decide(VL, obj, vt(rs), with_positions=False)
        run = [r for r, f, nr in zip(rs, d["fired"], d["near"]) if f or nr]
        chk = {r: (None if nr else bool(f)) for r, f, nr in zip(rs, d["fired"], d["near"])}
        got, mism = serve(obj, run, chk)
        vals = {r: got[r] if r in got else pre[r] for r in rs}
        info.update(n_forward=len(run), n_fired=int(d["fired"].sum()), n_in_band=int(d["near"].sum()), hook_mismatch=mism)
    elif kind == "lookup":
        XM.install_capture(B)
        plan = {r: XM.lookup_plan(obj.tok, ids(r), obj.subjects) for r in rs}
        run = [r for r in rs if plan[r]]
        got, mism = serve(obj, run, {r: True for r in run})
        vals = {r: got[r] if r in got else pre[r] for r in rs}
        info.update(n_forward=len(run), n_fired=len(run), hook_mismatch=mism)
    else:
        raise KeyError(kind)
    info["seconds"] = time.time() - t0
    X.jdump({"rids": rs, "values": [vals[r] for r in rs], "info": info}, p, indent=None)
    return vals, info


# ----------------------------------------------------------------------------- states
W0, WE = L.weights()
pre, _ = values("preedit", "weight", None, ALL)
dep, _ = values("deployed", "weight", WE, ALL)
X.say(f"{L.name}: pre-edit and deployed states on {len(ALL)} tuples")

# own requests, recognised from the log: a prompt position carries a logged solver key
KE = L.log.K.double()
ke2 = (KE * KE).sum(1)
own_edit = {}
for a in range(0, len(ALL), 512):
    rs = ALL[a:a + 512]
    K, off = VS.stack(vt(rs))
    Kd = K.double()
    d2 = (Kd * Kd).sum(1)[:, None] + ke2[None, :] - 2.0 * (Kd @ KE.T)
    rel = d2.clamp_min(0).sqrt() / ke2.sqrt()[None, :]
    for i, r in enumerate(rs):
        blk = rel[int(off[i]):int(off[i + 1])]
        m = blk.min()
        own_edit[r] = int(blk.min(0).values.argmin()) if float(m) <= 1e-4 else -1

SQL = {}
with open(os.path.join(X.CODE, "r22.sql")) as f:
    for blk in re.split(r"^-- name: ", f.read(), flags=re.M)[1:]:
        nm, body = blk.split("\n", 1)
        SQL[nm.strip()] = body
dbp = os.path.join(ODIR, L.name + ".sqlite")
if os.path.exists(dbp):
    os.remove(dbp)
db = sqlite3.connect(dbp)
db.row_factory = sqlite3.Row
db.executescript(SQL["schema"])


def q(name, **kw):
    return [dict(r) for r in db.execute(SQL[name], kw).fetchall()]


def put(model, vals, rs):
    db.executemany("INSERT INTO val (model, rid, value) VALUES (?, ?, ?)", [(model, r, int(vals[r])) for r in rs])
    db.commit()


new_first = {L.records[i]: int(L.target_ids(i, "new")[0]) for i in range(T)}
rows = []
for i, r in enumerate(edited):
    rr = cf[r]["requested_rewrite"]
    rows.append((r, rr["subject"], rr["relation_id"], XV.prompt_of(cf[r]), 1, "both", i, own_edit[r], None, new_first[r]))
for half, rs in (("report", report), ("test", test)):
    for r in rs:
        rr = cf[r]["requested_rewrite"]
        rows.append((r, rr["subject"], rr["relation_id"], XV.prompt_of(cf[r]), 0, half, None, own_edit[r], None, None))
db.executemany("INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
put("preedit", pre, ALL)
put("deployed", dep, ALL)
for nm in ("set_pre", "set_intended_unedited", "materialise_intended", "fill_member", "fill_valid_rel"):
    db.execute(SQL[nm])
db.commit()
counts = q("counts")[0]
dropped = q("dropped_relations")
X.say(f"{L.name}: {counts}")

FLAGS = {}


def readout(model):
    """The scores of R22 (c) of a materialised model, by SQL."""
    o = {}
    r = q("reference_preservation", model=model)[0]
    o["reference_preservation"] = {"n": r["n"], "n_ok": r["n_ok"], "pct": 100.0 * r["n_ok"] / r["n"]}
    segs = [("all", 0, T)] + ([("before", 0, X.SPLIT), ("from", X.SPLIT, T)] if collapse else [])
    for nm, lo, hi in segs:
        r = q("update_retention", model=model, lo=lo, hi=hi)[0]
        o["retention_" + nm] = {"n": r["n"], "n_ok": r["n_ok"], "pct": 100.0 * r["n_ok"] / r["n"]}
    vr, ar = q("view_results", model=model, half="test"), q("aggregate_results", model=model, half="test")
    o["n_views"], o["n_aggregates"] = len(vr), len(ar)
    o["views_exact_pct"] = 100.0 * sum(x["n_wrong"] == 0 for x in vr) / len(vr)
    o["aggregates_exact_pct"] = 100.0 * sum(x["tv"] == 0 for x in ar) / len(ar)
    o["queries_exact_pct"] = 100.0 * (sum(x["n_wrong"] == 0 for x in vr) + sum(x["tv"] == 0 for x in ar)) / (len(vr) + len(ar))
    o["aggregate_tv_mean"] = float(np.mean([x["tv"] for x in ar]))
    o["view_wrong_share_mean_pct"] = 100.0 * float(np.mean([x["n_wrong"] / x["n"] for x in vr]))
    fl = q("tuple_flags", model=model)
    FLAGS[model] = {"rid": [x["rid"] for x in fl], "edited": [x["edited"] for x in fl], "pos": [x["pos"] for x in fl],
                    "same_as_pre": [x["same_as_pre"] for x in fl], "as_intended": [x["as_intended"] for x in fl]}
    return o


OUT = {"meta": {"log": L.name, "T": T, "split": X.SPLIT if collapse else None, "view_pool": {k: pool[k] for k in ("n", "n_excluded_shared_subject", "n_candidates_scanned", "seed")},
                "counts": counts, "dropped_relations": dropped, "code": e10t.code_fingerprint(), "sql_file": "code/r22.sql", "database": dbp,
                "value": "first token id of the greedy continuation of the record's request prompt; -1 if a logit is not finite"}, "rows": []}


def add_row(label, model, evidence, cls, **kw):
    row = {"label": label, "model": model, "evidence": evidence, "class": cls, **kw, "scores": readout(model)}
    OUT["rows"].append(row)
    X.jdump(OUT, os.path.join(ODIR, L.name + ".json"))
    X.jdump(FLAGS, os.path.join(ODIR, L.name + "_flags.json"), indent=None)
    s = row["scores"]
    X.say(f"  {model}: ref {s['reference_preservation']['pct']:.1f} ret {s['retention_all']['pct']:.1f} queries exact {s['queries_exact_pct']:.1f} TV {s['aggregate_tv_mean']:.3f}")
    return row


def own_lost(S):
    """Edits whose own request no longer releases them (prompt positions of the request, view keys)."""
    Q = G.features(VL, S, vt(edited))["Q"]
    th = S.theta()
    lost = ~(Q[torch.arange(T), torch.arange(T)] >= th[:T])
    o = {"n": int(lost.sum())}
    if collapse:
        o.update(before=int(lost[:X.SPLIT].sum()), **{"from": int(lost[X.SPLIT:].sum())})
    return o


def lost_of_deleted(deleted):
    o = {"n": len(deleted)}
    if collapse:
        o.update(before=sum(i < X.SPLIT for i in deleted), **{"from": sum(i >= X.SPLIT for i in deleted)})
    return o


ZERO = {"diagnosis_passes": 0, "repair_passes": 0, "repair_seconds": 0.0, "statements_changed": 0}
# ----------------------------------------------------------------------------- rows without complaints
put("restore_base", pre, EVAL)
add_row("restore-base", "restore_base", "none (the log is discarded)", "reference", edits_lost=lost_of_deleted(list(range(T))), cost=dict(ZERO, statements_changed=T))
put("deployed_eval", dep, EVAL)
add_row("deployed", "deployed_eval", "none", "reference", edits_lost={"n": 0}, cost=ZERO)
GATES = {}
for g in ((True, False) if collapse else (True,)):
    M = XM.method(L, "where_logonly", guards=g)
    GATES[M.label] = M.obj
    v, info = values(M.label, "gate", M.obj, EVAL, pre)
    put(M.label, v, EVAL)
    add_row("ReScope-W log-only" + ("" if g else " (guards off)"), M.label, "log only", "repair of the log (thresholds)", guards=g, edits_lost=own_lost(M.obj), serve=info,
            cost=dict(ZERO, statements_changed=T), capped=M.rec["cal"].get("n_edits_capped"))
    for lab, nice in (("where_twoblock", "ReScope-W two-block"), ("rescope", "migration (ReScope)")):
        for d in draws:
            M = XM.method(L, lab, d, g)
            name = f"{M.label}__d{d}"
            v, info = values(name, "gate", M.obj, EVAL, pre)
            put(name, v, EVAL)
            add_row(nice + ("" if g else " (guards off)"), name, "log + 200 benign calibration prompts (REPAIR pool)",
                    "repair of the log (thresholds)" if lab == "where_twoblock" else "migration of the log", guards=g, draw=d, edits_lost=own_lost(M.obj), serve=info,
                    cost=dict(ZERO, statements_changed=T), capped=M.rec["cal"].get("n_edits_capped"))
            if d == 0 and g:
                GATES[lab] = M.obj
margin = XR.self_margin(L)
flagged = [int(i) for i in np.nonzero(margin <= 0)[0]]
t0 = time.time()
Wd = XR.replay(L, [i not in set(flagged) for i in range(T)])
sec = time.time() - t0
v, _ = values("rb_screen", "weight", Wd, EVAL)
put("rb_screen", v, EVAL)
add_row("rb_screen", "rb_screen", "log only (screening: self-separation margin <= 0)", "rollback", edits_lost=lost_of_deleted(flagged), deleted=flagged[:64],
        cost=dict(ZERO, repair_seconds=sec, statements_changed=len(flagged)))
if collapse:
    t0 = time.time()
    Wd = XR.replay(L, [i != X.SPLIT for i in range(T)])
    sec = time.time() - t0
    v, _ = values("oracle_delete_split", "weight", Wd, EVAL)
    put("oracle_delete_split", v, EVAL)
    add_row(f"oracle: delete update {X.SPLIT}", "oracle_delete_split", "oracle (the split position)", "rollback (oracle)", edits_lost=lost_of_deleted([X.SPLIT]),
            cost=dict(ZERO, repair_seconds=sec, statements_changed=1))
Mk = XM.lookup_method(L)
v, info = values("lookup_subject", "lookup", Mk.obj, EVAL, pre)
put("lookup_subject", v, EVAL)
add_row("lookup_subject", "lookup_subject", "the REQUESTS (subject strings) + log values", "OTHER DEPLOYMENT CLASS: keyed store over the requests", edits_lost={"n": 0}, serve=info, cost=ZERO)

# ----------------------------------------------------------------------------- serving overhead per forward pass
sample = test[:512]
tim = {}
objs = {"pre-edit weight": None, "deployed weight": WE, "ReScope-W (hook, all positions)": GATES["where_logonly"], "migration (hook, all positions)": GATES["rescope"], "lookup_subject": Mk.obj}
for rep in range(3):
    for nm, obj in objs.items():
        torch.cuda.synchronize()
        t0 = time.time()
        serve(obj, sample)
        torch.cuda.synchronize()
        tim.setdefault(nm, []).append(1000.0 * (time.time() - t0) / len(sample))
OUT["serving_overhead"] = {nm: {"ms_per_forward_min": min(v), "ms_per_forward_all": v, "ratio_to_pre_edit": min(v) / min(tim["pre-edit weight"])} for nm, v in tim.items()}
OUT["serving_overhead"]["_note"] = "512 test prompts, every prompt through the model (no sparse shortcut), batches of at most 32, 3 repeats, minimum; shared GPU"

# ----------------------------------------------------------------------------- query complaints
vr = {x["relation"]: x for x in q("view_results", model="deployed", half="report")}
ar = {x["relation"]: x for x in q("aggregate_results", model="deployed", half="report")}
cands = sorted([(r, "view") for r, x in vr.items() if x["n_wrong"] > 0] + [(r, "aggregate") for r, x in ar.items() if x["tv"] > 0])
OUT["complaints"] = {"n_report_queries": 2 * len(vr), "n_complaint_candidates": len(cands), "sets": {}}
S0 = {True: XM.method(L, "where_logonly", guards=True).obj, False: XM.method(L, "where_logonly", guards=False).obj}
keyset = lambda rs: [torch.from_numpy(np.array(VS.keys(XV.VBASE + r))) for r in rs]      # noqa: E731
for seed in (0, 1, 2):
    perm = np.random.default_rng([seed, 22]).permutation(len(cands))
    for m in (1, 3, 10):
        chosen = [cands[int(i)] for i in perm[:m]]
        rels = sorted({r for r, _ in chosen})
        sizes = [q("view_size", half="report", relation=r)[0] for r in rels]
        comp = sorted({x["rid"] for r in rels for x in q("diagnosis", relation=r)})
        n_diag = 2 * sum(s["n"] for s in sizes)
        tag = f"m{m}_s{seed}"
        OUT["complaints"]["sets"][tag] = {"m": m, "seed": seed, "queries": chosen, "relations": rels, "n_tuples_in_complained_views": sum(s["n"] for s in sizes),
                                          "n_own_requests_in_views": sum(s["n_own_requests"] for s in sizes), "diagnosis_passes": n_diag, "n_tuple_complaints": len(comp)}
        ev = f"log + {m} complained quer{'y' if m == 1 else 'ies'} (diagnosis: {n_diag} passes -> {len(comp)} tuples)"
        base = {"diagnosis_passes": n_diag, "repair_passes": 0}
        kw = dict(m=m, seed=seed, n_tuple_complaints=len(comp))
        ks = keyset(comp)
        X.say(f"{L.name} {tag}: {len(rels)} relations, {len(comp)} tuple complaints")
        # least threshold repair of ReScope-W, two forms (R21 d)
        for form, fn, g in (("sacrificing", XR.least_sacrificing, False), ("protecting", XR.least_protecting, True)):
            S, rec = fn(S0[g], ks) if ks else (S0[g], {"form": form, "n_complaints": 0, "n_refused": 0, "n_thresholds_raised": 0, "seconds": 0.0})
            name = f"least_{form}__{tag}"
            v, info = values(name, "gate", S, EVAL, pre)
            put(name, v, EVAL)
            d = G.decide(VL, S, vt(comp), with_positions=False) if comp else {"fired": np.zeros(0, bool), "near": np.zeros(0, bool)}
            add_row(f"ReScope-W least repair ({form})", name, ev, "repair of the log (thresholds)", **kw, edits_lost=own_lost(S), serve=info, repair=rec,
                    complaints_refused=rec["n_refused"], complaints_still_released=int(d["fired"].sum()), complaints_in_band=int(d["near"].sum()),
                    cost=dict(base, repair_seconds=rec["seconds"], statements_changed=rec["n_thresholds_raised"]))
        # rb_suffix
        seqs, exp = [ids(r) for r in comp], [pre[r] for r in comp]
        bl = batches(comp)

        def resolved():
            n = 0
            for b in bl:
                tok, fin = B.decode([seqs[i] for i in b], n=1, feed_last=False)
                n += len(b)
                if any((not bool(fin[j])) or int(tok[j, 0]) != exp[i] for j, i in enumerate(b)):
                    return False, n
            return True, n
        rec = XR.rb_suffix(L, B, resolved)
        ts = rec["t_star"]
        name = f"rb_suffix__{tag}"
        Wd = XR.replay(L, [i < ts for i in range(T)])
        v, _ = values(name, "weight", Wd, EVAL)
        put(name, v, EVAL)
        add_row("rb_suffix", name, ev, "rollback", **kw, edits_lost=lost_of_deleted(list(range(ts, T))), repair=rec,
                cost=dict(base, repair_passes=rec["n_forward"], repair_seconds=rec["seconds"], statements_changed=T - ts))
        # rb_lineage
        t0 = time.time()
        dele = XR.lineage(S0[True], ks)
        Wd = XR.replay(L, [i not in set(dele) for i in range(T)])
        sec = time.time() - t0
        name = f"rb_lineage__{tag}"
        v, _ = values(name, "weight", Wd, EVAL)
        put(name, v, EVAL)
        add_row("rb_lineage", name, ev, "rollback", **kw, edits_lost=lost_of_deleted(dele), cost=dict(base, repair_seconds=sec, statements_changed=len(dele)))
        # comp_min
        rec = XR.comp_min(L, torch.cat(ks) if ks else torch.zeros(0, 6400))
        Wc = rec.pop("W", None)
        name = f"comp_min__{tag}"
        if Wc is not None:
            v, _ = values(name, "weight", Wc, EVAL)
            put(name, v, EVAL)
            add_row("comp_min", name, ev, "compensation", **kw, edits_lost={"n": 0}, repair=rec, cost=dict(base, repair_seconds=rec["seconds"], statements_changed=1))
        else:
            OUT["rows"].append({"label": "comp_min", "model": name, "evidence": ev, "class": "compensation", **kw, "infeasible": True, "repair": rec, "scores": None,
                                "cost": dict(base, repair_seconds=rec["seconds"], statements_changed=0)})
        # blocklist
        name = f"blocklist__{tag}"
        db.executemany("INSERT INTO blocklist (setname, rid) VALUES (?, ?)", [(tag, r) for r in comp])
        db.execute(SQL["materialise_blocklist"], {"model": name, "setname": tag})
        db.commit()
        add_row("blocklist", name, ev, "exact-match list", **kw, edits_lost={"n": 0}, cost=dict(base, repair_seconds=0.0, statements_changed=len(comp)))
B.check_w0()
OUT["meta"]["seconds"] = time.time() - t_start
X.jdump(OUT, os.path.join(ODIR, L.name + ".json"))
X.jdump(FLAGS, os.path.join(ODIR, L.name + "_flags.json"), indent=None)
db.close()
X.say("done", L.name, f"{time.time() - t_start:.0f}s")
