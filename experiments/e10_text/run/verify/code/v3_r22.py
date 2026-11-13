"""E10 verification, block 2 (GPU): the relational maintenance workload of amendment R22 on TX-G2XL-rome-T100, recomputed with the
verifier's own tables (sqlite3) and SQL (v3_r22.sql).

    $GQ run --mem 13000 --prio -2 --name e10v_r22 -- $PY v3_r22.py
VIEW pool: built here from the protocol's rule and the builder's pool rule (records in no log, numpy default_rng(20261109)
permutation of the eligible records in case-id order; the first 2000 are the REPAIR and HELD-OUT pools (checked against the
builder's pools.json); the permutation is continued; records whose subject string is the subject of a request of the reported
stream are excluded and counted; the first 5000 remaining records; report half = the first 2500, test half = the others (the
experiment's stated split, mirrored).  Compared with the experiment's out/r22/view_pool.json.
States: value of a tuple = first token of the greedy continuation of its request prompt (one forward per tuple): R0 (pre-edit
weight), deployed (W_E), intended (first token of the new object for edited tuples, the R0 value otherwise).
Rows recomputed: deployed, restore-base, and the least threshold repair of ReScope-W (sacrificing and protecting) for seed 0 at
m = 3 complained queries; charged forward passes.  Complaint selection: the experiment's stated seeding rule (candidates sorted by
(relation, kind), numpy default_rng([seed, 22]).permutation), mirrored so that the same cell is recomputed.  Diagnosis and
own-request recognition (a prompt position carries a logged solver key: relative distance <= 1e-4) by the verifier.  Every
tuple on which a repaired model is materialised runs through the model under the library's hook (no sparse shortcut).
-> out/r22/TX-G2XL-rome-T100.json, out/r22/TX-G2XL-rome-T100.sqlite, per-state value files; compared with the experiment's
out/r22/TX-G2XL-rome-T100.json and its per-tuple value files.
"""
import os
import re
import sqlite3
import time

import vcommon as V

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from e10t import methods as MT  # noqa: E402
from e10t.backbone import edit_in  # noqa: E402
from rescope import least_repair  # noqa: E402

t_start = time.time()
M_Q, SEED = 3, 0
L = e10t.open_log(V.T100)
T = L.T
cf = B.C.load_counterfact()
ODIR = os.path.join(V.OUT, "r22")
os.makedirs(ODIR, exist_ok=True)
XR = V.jload(V.XOUT + f"/r22/{L.name}.json")                         # the experiment's result (read for the comparison only)

# ----------------------------------------------------------------------------- the VIEW pool
pools = V.jload(V.TXOUT + "/pools.json")
s0, s1 = pools["stream_records"]
d0, d1 = pools["dev_records"]
logged = set(range(s0, s1 + 1)) | set(range(d0, d1 + 1))
eligible = [i for i in range(len(cf)) if i not in logged]
perm = np.random.default_rng(20261109).permutation(len(eligible))
rep_pool = sorted(eligible[int(j)] for j in perm[:1000])
ho_pool = sorted(eligible[int(j)] for j in perm[1000:2000])
stream_subjects = {cf[i]["requested_rewrite"]["subject"] for i in range(s0, s1 + 1)}
view, excluded = [], []
for j in perm[2000:]:
    i = eligible[int(j)]
    if cf[i]["requested_rewrite"]["subject"] in stream_subjects:
        excluded.append(i)
        continue
    view.append(i)
    if len(view) == 5000:
        break
report, test = view[:2500], view[2500:]
xpool = V.jload(V.XOUT + "/r22/view_pool.json")
pool_info = {"n_view": len(view), "n_excluded_shared_subject": len(excluded), "pools_of_the_build_reproduced": rep_pool == sorted(pools["repair_records"]) and ho_pool == sorted(pools["heldout_records"]),
             "view_disjoint_from_logs_and_pools": not (set(view) & (logged | set(rep_pool) | set(ho_pool))),
             "same_records_as_experiment": view == xpool["records"], "same_report_half": report == xpool["report"], "same_test_half": test == xpool["test"],
             "experiment_n_excluded": xpool["n_excluded_shared_subject"],
             "view_records_sharing_a_subject_with_a_T100_request": sum(cf[i]["requested_rewrite"]["subject"] in {cf[r]["requested_rewrite"]["subject"] for r in L.records} for i in view)}
V.say("view pool", pool_info)

edited = list(L.records)
ALL = edited + report + test
EVAL = edited + test
prompt = {r: cf[r]["requested_rewrite"]["prompt"].format(cf[r]["requested_rewrite"]["subject"]) for r in ALL}
ids = {r: B.tok(prompt[r])["input_ids"] for r in ALL}
for i, r in enumerate(edited):
    if ids[r] != L.ids(L.rows["request"][i]["text_id"]):
        raise RuntimeError("request prompt of an edited tuple differs from the build's")
new_first = {L.records[i]: int(B.tok(" " + cf[L.records[i]]["requested_rewrite"]["target_new"]["str"].strip())["input_ids"][0]) for i in range(T)}


@torch.no_grad()
def first_tokens(obj, rs, own=None):
    """{rid: first token of the greedy continuation} under the served object; every prompt through the model.  For gates also
    {rid: hook fired}, {rid: own edit released} (own: {rid: edit})."""
    vals, fired, ownrel = {}, {}, {}
    with B.served(obj) as srv:
        for b in B.batches([len(ids[r]) for r in rs], extra=1):
            br = [rs[i] for i in b]
            tok, fin = B.decode([ids[r] for r in br], n=1, feed_last=False)
            fl = srv.rows()
            for j, r in enumerate(br):
                vals[r] = int(tok[j, 0]) if bool(fin[j]) else -1
                if fl is not None:
                    fired[r] = bool(fl[j]["fired"])
                    if own is not None and r in own:
                        ownrel[r] = bool(edit_in(fl[j]["rel_edits"], own[r]))
    return vals, fired, ownrel


def cached(name, fn):
    p = os.path.join(ODIR, "values", name + ".json")
    if os.path.exists(p):
        c = V.jload(p)
        return {int(k): v for k, v in c["values"].items()}, c.get("extra", {})
    vals, extra = fn()
    V.jdump({"values": vals, "extra": extra}, p, indent=None)
    return vals, extra


W0, WE = L.weights()
pre, _ = cached("R0", lambda: (first_tokens(None, ALL)[0], {}))
dep, _ = cached("deployed", lambda: (first_tokens(WE, ALL)[0], {}))
V.say("states R0 and deployed on", len(ALL), "tuples")

# ----------------------------------------------------------------------------- tables
SQL = {}
with open(os.path.join(V.CODE, "v3_r22.sql")) as f:
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


def put(name, vals, rs):
    db.executemany("INSERT INTO state (name, rid, v) VALUES (?, ?, ?)", [(name, r, int(vals[r])) for r in rs])
    db.commit()


rows = []
for i, r in enumerate(edited):
    rr = cf[r]["requested_rewrite"]
    rows.append((r, rr["subject"], rr["relation_id"], prompt[r], 1, None, i, new_first[r]))
for half, rs in (("report", report), ("test", test)):
    for r in rs:
        rr = cf[r]["requested_rewrite"]
        rows.append((r, rr["subject"], rr["relation_id"], prompt[r], 0, half, None, None))
db.executemany("INSERT INTO tuple VALUES (?,?,?,?,?,?,?,?)", rows)
put("R0", pre, ALL)
put("deployed", dep, ALL)
db.execute(SQL["fill_inview"])
db.execute(SQL["intended_state"])
db.commit()
put("restore_base", pre, EVAL)                                           # restore-base: the pre-edit model on every tuple

OUT = {"log": L.name, "pool": pool_info, "sql_file": "code/v3_r22.sql", "database": dbp, "states": {}, "rows": {}, "compare": {}}
OUT["states"]["sizes"] = q("state_sizes")
OUT["states"]["queries_test"] = len(q("queries_of_half", half="test"))
OUT["states"]["queries_report"] = len(q("queries_of_half", half="report"))
OUT["states"]["dropped_test"] = q("dropped_of_half", half="test")
OUT["states"]["dropped_report"] = q("dropped_of_half", half="report")
for nm, a, b_, ed, half in (("deployed_vs_R0_unedited_test", "deployed", "R0", 0, "test"), ("deployed_vs_R0_unedited_report", "deployed", "R0", 0, "report"),
                            ("intended_vs_R0_unedited_test", "intended", "R0", 0, "test"), ("intended_vs_R0_edited", "intended", "R0", 1, None),
                            ("deployed_vs_intended_edited", "deployed", "intended", 1, None), ("deployed_vs_R0_edited", "deployed", "R0", 1, None)):
    OUT["states"][nm] = q("states_differ", a=a, b=b_, edited=ed, half=half)[0]
OUT["states"]["n_nonfinite_values"] = {"R0": sum(v == -1 for v in pre.values()), "deployed": sum(v == -1 for v in dep.values())}


def scores(model):
    rp = q("reference_preservation", model=model)[0]
    rt = q("update_retention", model=model)[0]
    sv = q("selection_views", model=model, half="test")
    ag = q("aggregates", model=model, half="test")
    nq = len(sv) + len(ag)
    ex_v, ex_a = sum(x["n_wrong"] == 0 for x in sv), sum(x["half_l1"] == 0 for x in ag)
    return {"reference_preservation": {"n": rp["n"], "n_ok": rp["n_ok"], "pct": 100.0 * rp["n_ok"] / rp["n"]},
            "retention": {"n": rt["n"], "n_ok": rt["n_ok"], "pct": 100.0 * rt["n_ok"] / rt["n"]},
            "n_views": len(sv), "n_aggregates": len(ag), "views_exact": ex_v, "aggregates_exact": ex_a, "queries_exact_pct": 100.0 * (ex_v + ex_a) / nq,
            "aggregate_tv_mean": float(np.mean([x["tv"] for x in ag]))}


def compare(label, sc, xmodel, vals=None):
    """Against the experiment's row of the same model (scores) and its per-tuple values."""
    xr = [r for r in XR["rows"] if r["model"] == xmodel]
    o = {"experiment_row_found": bool(xr)}
    if xr:
        xs = xr[0]["scores"]
        o.update(reference_preservation=[sc["reference_preservation"]["n_ok"], xs["reference_preservation"]["n_ok"]], retention=[sc["retention"]["n_ok"], xs["retention_all"]["n_ok"]],
                 queries_exact_pct=[sc["queries_exact_pct"], xs["queries_exact_pct"]], aggregate_tv_mean=[sc["aggregate_tv_mean"], xs["aggregate_tv_mean"]],
                 cost=xr[0]["cost"], edits_lost=xr[0].get("edits_lost"), complaints_refused=xr[0].get("complaints_refused"))
        o["scores_equal"] = bool(o["reference_preservation"][0] == o["reference_preservation"][1] and o["retention"][0] == o["retention"][1]
                                 and abs(o["queries_exact_pct"][0] - o["queries_exact_pct"][1]) < 1e-9 and abs(o["aggregate_tv_mean"][0] - o["aggregate_tv_mean"][1]) < 1e-12)
    vp = V.XOUT + f"/r22/{L.name}_values/{xmodel if xmodel not in ('deployed_eval', 'restore_base') else {'deployed_eval': 'deployed', 'restore_base': 'preedit'}[xmodel]}.json"
    if vals is not None and os.path.exists(vp):
        c = V.jload(vp)
        xv = dict(zip(c["rids"], c["values"]))
        common = [r for r in vals if r in xv]
        o["per_tuple"] = {"n_common": len(common), "n_values_differ": sum(int(vals[r]) != int(xv[r]) for r in common)}
    OUT["compare"][label] = o
    return o


for label, model, xmodel, vals in (("deployed", "deployed", "deployed_eval", dep), ("restore-base", "restore_base", "restore_base", pre)):
    sc = scores(model)
    OUT["rows"][label] = {"scores": sc, "diagnosis_passes": 0, "repair_passes": 0}
    V.say(label, sc, compare(label, sc, xmodel, vals).get("scores_equal"))

# ----------------------------------------------------------------------------- query complaints (m = 3, seed 0), diagnosis
sv = {x["relation"]: x for x in q("selection_views", model="deployed", half="report")}
ag = {x["relation"]: x for x in q("aggregates", model="deployed", half="report")}
cands = sorted([(r, "view") for r, x in sv.items() if x["n_wrong"] > 0] + [(r, "aggregate") for r, x in ag.items() if x["half_l1"] > 0])
permc = np.random.default_rng([SEED, 22]).permutation(len(cands))
chosen = [list(cands[int(i)]) for i in permc[:M_Q]]
rels = sorted({r for r, _ in chosen})
xset = XR["complaints"]["sets"][f"m{M_Q}_s{SEED}"]
view_rids = {r: [x["rid"] for x in q("view_tuples", relation=r)] for r in rels}
diag_rids = sorted({rid for r in rels for rid in view_rids[r]})
n_view_tuples = sum(len(v) for v in view_rids.values())
# own requests, recognised from the log: a position of the prompt carries a logged solver key (keys of the diagnosis' pre-edit pass)
KE = L.log.K.double()
keys = {}
with torch.no_grad():
    B.check_w0()
    for b in B.batches([len(ids[r]) for r in diag_rids], extra=0):
        br = [diag_rids[i] for i in b]
        for r, x in zip(br, B.keys([ids[r] for r in br])):
            keys[r] = x.float().cpu()
own = {}
for r in diag_rids:
    k = keys[r].double()
    rel = torch.cdist(k, KE) / KE.norm(dim=1)[None, :]
    m = rel.min()
    if float(m) <= 1e-4:
        own[r] = int(rel.min(0).values.argmin())
db.executemany("INSERT INTO ownreq (rid, edit) VALUES (?, ?)", list(own.items()))
db.commit()
comp = sorted({x["rid"] for r in rels for x in q("diagnosis", relation=r)})
diag = {"m": M_Q, "seed": SEED, "n_report_queries": 2 * len(sv), "n_complaint_candidates": len(cands), "queries": chosen, "relations": rels,
        "n_tuples_in_complained_views": n_view_tuples, "diagnosis_passes": 2 * n_view_tuples, "n_own_requests_in_views": len(own),
        "own_requests_are_exactly_the_edited_tuples_of_the_views": sorted(own) == sorted(r for r in diag_rids if r in set(edited)),
        "n_tuple_complaints": len(comp),
        "experiment": {k: xset[k] for k in ("queries", "relations", "n_tuples_in_complained_views", "n_own_requests_in_views", "diagnosis_passes", "n_tuple_complaints")},
        "experiment_n_report_queries": XR["complaints"]["n_report_queries"], "experiment_n_candidates": XR["complaints"]["n_complaint_candidates"]}
diag["same_as_experiment"] = bool(chosen == [list(x) for x in xset["queries"]] and diag["diagnosis_passes"] == xset["diagnosis_passes"] and len(comp) == xset["n_tuple_complaints"])
diag["complaints_all_in_report_half_and_unedited"] = all(r in set(report) for r in comp)
OUT["diagnosis"] = diag
V.say("diagnosis", {k: v for k, v in diag.items() if k != "experiment"})
ckeys = [keys[r] for r in comp]

# ----------------------------------------------------------------------------- least threshold repair of ReScope-W (two forms)
own_edit = {r: i for i, r in enumerate(edited)}
for form, goff in (("least_sacrificing", True), ("least_protecting", False)):
    name = f"{form}__m{M_Q}_s{SEED}"
    S0 = MT.where(L, "logonly", allow_silence=goff).obj
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
    sec = time.time() - t0
    th1 = S.theta()

    def serve():
        vals, fired, ownrel = first_tokens(S, EVAL, own_edit)
        return vals, {"n_fired": sum(fired.values()), "n_fired_test": sum(fired[r] for r in test), "edits_lost": sum(not ownrel[r] for r in edited),
                      "silent_tuples_with_a_value_other_than_R0": sum((not fired[r]) and vals[r] != pre[r] for r in EVAL)}
    vals, extra = cached(name, serve)
    put(name, vals, EVAL)
    # the complaints after the repair, as served: first token under the repaired model against the pre-edit value
    cv, cf_, _ = first_tokens(S, comp)
    sc = scores(name)
    row = {"scores": sc, "diagnosis_passes": diag["diagnosis_passes"], "repair_passes": 0, "repair_seconds": sec, "n_thresholds_raised": int((th1 > th0).sum()),
           "complaints_refused": refused, "serve": extra, "complaints_resolved_as_served": sum(cv[r] == pre[r] for r in comp), "complaints_still_released_as_served": sum(cf_.values()),
           "n_tuple_complaints": len(comp)}
    OUT["rows"][form] = row
    c = compare(form, sc, name, vals)
    V.say(form, sc, {k: row[k] for k in ("n_thresholds_raised", "complaints_refused", "serve", "complaints_resolved_as_served")}, c.get("scores_equal"), c.get("per_tuple"))
B.check_w0()
OUT["seconds"] = time.time() - t_start
V.jdump(OUT, os.path.join(ODIR, L.name + ".json"))
db.close()
V.say("done")
