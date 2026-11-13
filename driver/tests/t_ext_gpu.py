"""GPU tests of the stage-1 extension of the fx driver (development logs and their prefixes only).

  $GQ run --mem 10500 --prio 5 -- python t_ext_gpu.py TB-Q100-ae-l2_3 h m r s k        (17000 for TB-L100-ae-l2_3)
  $GQ run --mem 10500 --prio 5 -- python t_ext_gpu.py TB-Q100-ae-l2_3-p10 p            (one log per process)
  -> tests/out/ext_gpu_<log>_<test>.json          (python render_ext.py renders tests/out/table_ext.txt)

  h   (10 ii) ses.tf_hashed: logit hashes and released edits; silent sequences == the pre-edit model bit for bit; deterministic;
      outcome fields == the memo path; Rescope-WHERE (fire 'all') released sets
  m   (10 i)  masked edits are never released by the hook; frame rule on real forwards: a sequence on which no masked edit was
      released has bit-identical logits before and after masking
  r   (10 iii) rollback weights: harness convention (empty set = W_E, all = W_0, one edit, a whole batch, mixed) and the same
      evaluation functions as the deployed weight (memo by content, == the library's rows)
  s   (3, 4, 5) the behavioural form of H3 (raised gate through the memo path == gate level), rescope_pac, the dense gates'
      calibration rows == the default method's, the router on key-level pools
  k   (6) kept / gained and the D2 rate with interval on real rows
  p   (9) a prefix log: driver session, harness linear repairs (COMP, ROLLBACK_FO, REEDIT) through fx.prefix.load_model, rollback
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(HERE, "out")
FIELDS = ("exact", "token_acc", "n_tok", "finite", "first_mismatch", "D1", "tok_loc", "fired", "top_last")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_rows(L, n_edit=20, n_probe=20, n_hn=20):
    hd, hn = L.pool("heldout_disjoint"), L.rows("hn", valid=True)
    ed = list(range(0, L.T, max(1, L.T // n_edit)))[:n_edit]
    return ([L.edit_rows["rel"][i] for i in ed] + [L.edit_rows["tgen"][i] for i in ed] + [L.edit_rows["igen"][i] for i in ed]
            + hd[::max(1, len(hd) // n_probe)][:n_probe] + hn[::max(1, len(hn) // n_hn)][:n_hn])


def same_fields(a, b, fields=FIELDS) -> int:
    return sum(any(x.get(k) != y.get(k) for k in fields) for x, y in zip(a, b))


def finish(ses, res):
    res["hook_mismatch"] = int(ses.stats["hook_mismatch"])
    res["memo_conflicts"] = int(ses.memo.stats.get("conflicts", 0))
    res["ok"] = bool(res["ok"] and res["hook_mismatch"] == 0 and res["memo_conflicts"] == 0)
    return res


# ----------------------------------------------------------------------------- h: logit hashes
def t_h(L, H, ses):
    M, pre, dep = fx.methods.rescope(L, draw=0), fx.methods.preedit(L), fx.methods.deployed(L)
    rows = sample_rows(L)
    t0 = time.time()
    a = ses.tf_hashed(M, rows)
    sec = time.time() - t0
    a2 = ses.tf_hashed(M, rows)
    p = ses.tf_hashed(pre, rows)
    d = ses.tf_hashed(dep, rows[:20])
    memo = ses.tf(M, rows)
    dmemo = ses.tf(dep, rows[:20])
    g = fx.gate.decide(L, M, rows)
    silent = [i for i, r in enumerate(a) if not r["released"]]
    firing = [i for i, r in enumerate(a) if r["released"]]
    res = {"n_rows": len(rows), "seconds_per_forward": sec / len(rows), "n_firing": len(firing), "n_silent": len(silent),
           "deterministic": sum(x["logit_sha1"] != y["logit_sha1"] for x, y in zip(a, a2)),
           "silent_rows_not_bitwise_pre_edit": sum(a[i]["logit_sha1"] != p[i]["logit_sha1"] for i in silent),
           "firing_rows_bitwise_pre_edit": sum(a[i]["logit_sha1"] == p[i]["logit_sha1"] for i in firing),
           "released_differs_from_gate_level": sum(r["released"] != ([int(g["top_edit"][i])] if g["fired"][i] else []) for i, r in enumerate(a)),
           "outcome_fields_differ_from_memo_path": same_fields(a, memo), "deployed_fields_differ_from_weights_path": same_fields(d, dmemo, FIELDS[:7]),
           "deployed_released_is_none": all(r["released"] is None for r in d), "preedit_released_is_none": all(r["released"] is None for r in p),
           "deployed_rows_bitwise_pre_edit": sum(x["logit_sha1"] == y["logit_sha1"] for x, y in zip(d, p)), "sha_length": len(a[0]["logit_sha1"])}
    # Rescope-WHERE: every position, every candidate edit releases
    Mw = fx.methods.where_logonly(L, H)
    w = ses.tf_hashed(Mw, rows[:12])
    lib = ses.tf(Mw, rows[:12])
    res["where"] = {"n": len(w), "released_sets": [len(r["released"]) for r in w], "fired_consistent": all(bool(r["released"]) == bool(r["fired"]) for r in w),
                    "n_cand_consistent": all((r["n_cand_edits"] or 0) == len(r["released"]) for r in w), "fields_differ_from_library_path": same_fields(w, lib)}
    res["ok"] = (res["deterministic"] == 0 and res["silent_rows_not_bitwise_pre_edit"] == 0 and res["released_differs_from_gate_level"] == 0
                 and res["outcome_fields_differ_from_memo_path"] == 0 and res["deployed_fields_differ_from_weights_path"] == 0 and res["deployed_released_is_none"]
                 and res["n_firing"] > 0 and res["n_silent"] > 0 and res["where"]["fired_consistent"] and res["where"]["n_cand_consistent"]
                 and res["where"]["fields_differ_from_library_path"] == 0)
    return finish(ses, res)


# ----------------------------------------------------------------------------- m: masking on real forwards
def t_m(L, H, ses):
    M, pre = fx.methods.rescope(L, draw=0), fx.methods.preedit(L)
    R = [0, 3, 17, 18, 40]
    Mm = fx.methods.mask_edits(L, M, R)
    rows = sorted(set([L.edit_rows[k][i] for k in ("rel", "tgen", "igen") for i in R] + sample_rows(L, 12, 15, 25)
                      + [j for j in L.rows("hn", valid=True) if L.own(j) in R][:30]))
    a, b, p = ses.tf_hashed(M, rows), ses.tf_hashed(Mm, rows), ses.tf_hashed(pre, rows)
    memo = ses.tf(Mm, rows)
    hit = [bool(set(r["released"]) & set(R)) for r in a]
    res = {"n_rows": len(rows), "masked_edits": R, "n_rows_with_a_masked_release_before": sum(hit),
           "n_releases_of_a_masked_edit_after": sum(bool(set(r["released"]) & set(R)) for r in b),
           "frame_rule_violations": sum((not h) and x["logit_sha1"] != y["logit_sha1"] for h, x, y in zip(hit, a, b)),
           "n_rows_checked_by_the_frame_rule": sum(not h for h in hit),
           "masked_rows_now_pre_edit_bitwise": sum(h and not y["released"] and y["logit_sha1"] == z["logit_sha1"] for h, y, z in zip(hit, b, p)),
           "masked_rows_now_silent": sum(h and not y["released"] for h, y in zip(hit, b)),
           "masked_rows_other_edit_released": sum(h and bool(y["released"]) for h, y in zip(hit, b)),
           "outcome_fields_differ_from_memo_path": same_fields(b, memo), "empty_mask_digest_equal": fx.methods.mask_edits(L, M, []).digest == M.digest}
    res["ok"] = (res["n_releases_of_a_masked_edit_after"] == 0 and res["frame_rule_violations"] == 0 and res["n_rows_with_a_masked_release_before"] >= 10
                 and res["masked_rows_now_pre_edit_bitwise"] == res["masked_rows_now_silent"] and res["outcome_fields_differ_from_memo_path"] == 0 and res["empty_mask_digest_equal"])
    return finish(ses, res)


# ----------------------------------------------------------------------------- r: rollback weights
def t_r(L, H, ses):
    W0f, C = H.W0.float(), H.WE.float() - H.W0.float()
    U, Wf = H.U, H.Wfac
    Me, Ma = fx.methods.rollback(L, H, []), fx.methods.rollback(L, H, range(L.T))
    i = 7
    M1 = fx.methods.rollback(L, H, [i])
    ref1 = (W0f + (C - torch.outer(U[i], Wf[i]).float())).to(H.dtype)
    batch1 = list(H.batches[1])
    Mb = fx.methods.rollback(L, H, batch1)
    A1 = H.tl.batch_delta(1).to(H.dev)
    refb = (W0f + (C - A1)).to(H.dtype)
    mixed = batch1 + [25]
    Mx = fx.methods.rollback(L, H, mixed)
    refx = (W0f + (C - (A1 + torch.outer(U[25], Wf[25]).float()))).to(H.dtype)
    rows = sample_rows(L, 6, 12, 6)
    n0 = dict(ses.stats)
    a = ses.tf(Mb, rows)
    lib = ses.ev_plain.tf_rows(Mb.obj, [ses.item(j) for j in rows])
    f1 = ses.stats["forwards"]
    a2 = ses.tf(fx.methods.rollback(L, H, list(reversed(batch1))), rows)          # the same content again: memo, no forward
    f2 = ses.stats["forwards"]
    hsh = ses.tf_hashed(Mb, rows[:8])
    e_rows = ses.tf(Me, rows)
    d_rows = ses.tf(fx.methods.deployed(L), rows)
    p_rows = ses.tf(Ma, rows[:8])
    pre = ses.tf(fx.methods.preedit(L), rows[:8])
    res = {"empty_set_is_WE_bitwise": bool(torch.equal(Me.obj, H.WE)), "all_edits_is_W0_bitwise": bool(torch.equal(Ma.obj, H.W0)),
           "one_edit_equals_formula": bool(torch.equal(M1.obj, ref1)), "whole_batch_equals_formula": bool(torch.equal(Mb.obj, refb)), "mixed_equals_formula": bool(torch.equal(Mx.obj, refx)),
           "one_edit_changes_the_weight": not bool(torch.equal(M1.obj, H.WE)), "kinds": [Me.kind, Mb.kind], "whole_batches_recorded": Mb.rec["whole_batches"],
           "digests_distinct": len({Me.digest, Ma.digest, M1.digest, Mb.digest, Mx.digest}) == 5, "order_irrelevant": fx.methods.rollback(L, H, list(reversed(mixed))).digest == Mx.digest,
           "rows_differ_from_library": same_fields(a, lib, FIELDS[:7]), "second_evaluation_forwards": f2 - f1, "rows_equal_second": same_fields(a, a2, FIELDS[:7]),
           "hashed_fields_differ": same_fields(hsh, a[:8], FIELDS[:7]), "empty_set_rows_differ_from_deployed": same_fields(e_rows, d_rows, FIELDS[:7]),
           "all_edits_rows_differ_from_pre_edit": same_fields(p_rows, pre, FIELDS[:5]), "n_rows": len(rows), "forwards_first": f1 - n0["forwards"]}
    res["ok"] = (res["empty_set_is_WE_bitwise"] and res["all_edits_is_W0_bitwise"] and res["one_edit_equals_formula"] and res["whole_batch_equals_formula"] and res["mixed_equals_formula"]
                 and res["one_edit_changes_the_weight"] and res["kinds"] == ["weight", "weight"] and res["digests_distinct"] and res["order_irrelevant"] and res["rows_differ_from_library"] == 0
                 and res["second_evaluation_forwards"] == 0 and res["rows_equal_second"] == 0 and res["hashed_fields_differ"] == 0 and res["empty_set_rows_differ_from_deployed"] == 0
                 and res["all_edits_rows_differ_from_pre_edit"] == 0)
    return finish(ses, res)


# ----------------------------------------------------------------------------- s: H3 behavioural form, rescope_pac, dense gates, pools
def t_s(L, H, ses):
    res = {"h3": {}}
    ok = True
    for lab, M in (("rescope", fx.methods.rescope(L, draw=0)), ("cosine", fx.methods.cosine(L, draw=0))):
        h = fx.sweeps.h3(L, M)
        Mh = h["method"]
        odd = h["odd_edits"][:12]
        ed = ses.edits(Mh, edits=odd)
        tr = h["test_rows"]["rows"][::9]
        hn = ses.hard_negatives(Mh, rows=tr)
        gl = {j: (bool(f), int(t)) for j, f, t in zip(h["test_rows"]["rows"], h["test_rows"]["fired"], h["test_rows"]["top_edit"])}
        pe = {e: (bool(a), bool(b)) for e, a, b in zip(h["per_edit"]["edits"], h["per_edit"]["tgen"], h["per_edit"]["igen"])}
        bad_e = sum(bool(r["own_released_last"]) != pe[r["edit"]][0 if r["metric"] == "tgen" else 1] for r in ed if r["metric"] in ("tgen", "igen"))
        bad_h = sum((bool(r["fired"]), int(r["top_last"])) != gl[j] for j, r in zip(tr, hn))
        res["h3"][lab] = {"delta": h["delta"], "n_edits": len(odd), "n_test_rows": len(tr), "edit_flags_differ_from_gate_level": bad_e, "hard_negative_flags_differ_from_gate_level": bad_h,
                          "memo_path": Mh.memo, "exact_on_sample": fx.metrics.edit_stats(ed)["rephrase"]["exact"], "n_fired_hn": sum(bool(r["fired"]) for r in hn), "src": sorted({r["src"] for r in ed + hn})}
        ok &= bad_e == 0 and bad_h == 0 and Mh.memo
    Mp = fx.methods.build("rescope_pac", L, draw=0)
    rows = sample_rows(L, 8, 10, 10)
    a = ses.tf(Mp, rows)
    g = fx.gate.decide(L, Mp, rows)
    res["rescope_pac"] = {"n_rows": len(rows), "flags_differ_from_gate_level": sum((bool(r["fired"]), int(r["top_last"])) != (bool(f), int(t)) for r, f, t in zip(a, g["fired"], g["top_edit"])),
                          "rank": Mp.rec["cal"]["rank"], "n_edits_capped": Mp.rec["cal"]["n_edits_capped"]}
    ok &= res["rescope_pac"]["flags_differ_from_gate_level"] == 0
    # the SAME calibration rows for every calibrated row of the E1 row set (dense gates need the model)
    rs = {M.label: M for M in fx.methods.e1_rowset(L, 0.05, 1, H=H, labels=("rescope", "rescope_shared", "rescope_null", "cosine", "cosine_shared", "ball", "router", "zscore"))}
    sids = {k: tuple(M.rec["draw"]["sids"]) for k, M in rs.items()}
    res["rowset"] = {"calibrated_rows": list(rs), "same_calibration_rows": len(set(sids.values())) == 1, "n_rows": len(next(iter(sids.values()))),
                     "rules": {k: M.rec["cal"].get("op") for k, M in rs.items()}, "digests_distinct": len({M.digest for M in rs.values()}) == len(rs),
                     "all_rows_built": [M.label for M in fx.methods.e1_rowset(L, 0.05, 1, H=H)] == list(fx.methods.E1_LABELS)}
    ok &= res["rowset"]["same_calibration_rows"] and res["rowset"]["digests_distinct"] and res["rowset"]["all_rows_built"]
    # the router on key-level pools (its activation needs the weights); the z-score gate needs whole sequences
    pr = {}
    for pn in ("BEN-T", "MME-T"):
        st = fx.pools.firing(L, rs["router"], pn)
        P = fx.pools.get(L, pn)
        ref = (L.kc if P.kc is None else P.kc).decide(rs["router"].obj, P.which, rows=P.rows)
        pr[pn] = {"n": st["n"], "fired": st["fired"], "equals_library": bool(np.array_equal(np.asarray(ref["fired"]), np.asarray(fx.pools.decide(L, rs["router"], P)["fired"])))}
        ok &= pr[pn]["equals_library"] and st["n"] == len(P.rows)
    try:
        fx.pools.decide(L, rs["zscore"], "BEN-T")
        pr["zscore_refused"] = False
    except NotImplementedError:
        pr["zscore_refused"] = True
    Mrm = fx.methods.router(L, H, "conformal", 0.05, 200, 0, source="MME-C")
    pr["router_calibrated_on_MME-C"] = {"n": Mrm.n, "thr_differs": Mrm.obj.thr != rs["router"].obj.thr}
    ok &= pr["zscore_refused"] and Mrm.n == 200
    res["pools"] = pr
    ses.release()
    res["ok"] = bool(ok)
    return finish(ses, res)


# ----------------------------------------------------------------------------- k: R7 read-outs on real rows
def t_k(L, H, ses):
    M, dep = fx.methods.rescope(L, draw=0), fx.methods.deployed(L)
    rm, rd = ses.edits(M), ses.edits(dep)
    kg = fx.metrics.kept_gained(rm, rd, ci=True)
    es_m, es_d = fx.metrics.edit_stats(rm), fx.metrics.edit_stats(rd)
    d2d, d2m = ses.d2(dep), ses.d2(M)
    sd, sm = fx.metrics.d2_stats(d2d, ci=True), fx.metrics.d2_stats(d2m, ci=True)
    stored = [L.probe_rec(j) for j in L.rows("probe", pool="heldout_disjoint", where=lambda e: bool(e.get("pre_correct_long")))]
    gr = fx.metrics.grid({0: d2m, 1: d2m}, "D2", where=lambda r: r.get("pre_correct"))
    cb = fx.stats.crossed_bootstrap(gr["num"], gr["den"], scale=100.0)
    res = {"kept_gained": {m: {k: (v["value"] if isinstance(v, dict) else v) for k, v in kg[m].items()} for m in kg},
           "kept_ci": {m: [kg[m]["kept"]["ci_lo"], kg[m]["kept"]["ci_hi"], kg[m]["kept"]["n_units"]] for m in kg},
           "consistent": all(abs(kg[m]["kept"]["value"] + kg[m]["gained"]["value"] - es_m[m]["exact"]) < 1e-9 and abs(kg[m]["kept"]["value"] + kg[m]["lost"]["value"] - es_d[m]["exact"]) < 1e-9 for m in kg),
           "d2_deployed": {"n_pre_correct": sd["n_pre_correct"], "n_D2": sd["n_D2"], "rate": sd["D2"], "n_D2_stored_records": sum(bool(r["is_complaint_D2"]) for r in stored), "n_rows_stored": len(stored)},
           "d2_method": {"n_pre_correct": sm["n_pre_correct"], "n_D2": sm["n_D2"], "rate": sm["D2"]},
           "crossed_grid": {"shape": list(gr["num"].shape), "value": cb["value"], "ci": [cb["ci_lo"], cb["ci_hi"]]}}
    res["ok"] = (res["consistent"] and sd["D2"]["ci_lo"] <= sd["D2"]["value"] <= sd["D2"]["ci_hi"] and sd["n_pre_correct"] == len(stored) and sd["D2"]["n_units"] <= sd["n_pre_correct"]
                 and abs(cb["value"] - sm["D2"]["value"]) < 1e-9 and sd["n_D2"] == res["d2_deployed"]["n_D2_stored_records"])
    return finish(ses, res)


# ----------------------------------------------------------------------------- p: a prefix log through the driver and the harness
def t_p(name):
    import rescope
    parent, k = fx.prefix.spec(name)
    cdir = os.path.join(OUT, "hcache_test")
    t0 = time.time()
    H = fx.prefix.load_model(name, keep_batch_deltas=True, cache_dir=cdir)
    L = fx.logs.open(name)
    hz = rescope.harness()
    res = {"log": name, "T": L.T, "seconds_load": time.time() - t0, "harness_cache": hz.HC.CACHE, "checks": {c: bool(v) for c, v in H.checks.items()},
           "applied_updates_kept": H.A is not None and len(H.A) == k // 10, "handle_T": int(H.T)}
    ok = res["applied_updates_kept"]
    # ---- 1. the harness alone on the prefix handle (the calls of P12's run_streams.py), before any driver session takes memory
    cs = hz.HS.sample_complaints(H, 4, 0)
    drv = fx.draws._repair_complaints(L)
    res["complaints"] = {"n_sampled": len(cs), "n_repair_D1": len(drv), "sampled_are_repair_D1": all((c["probe_idx"], c["ptype"]) in {(t[0], t[1]) for t in drv} for c in cs)}
    st20 = fx.draws.complaint_stream(L, 0, n_base=20)
    h20 = hz.HS.sample_complaints(H, 20, 0)
    res["complaints"].update(stream_n_base_20={"n": len(st20), "first_20_are_harness_sample": set(st20[:20]) == {L.by_lib_sid(f"probe:{c['ptype']}:{c['probe_idx']}") for c in h20},
                                               "nested": fx.draws.complaints(L, 10, 0, n_base=20) == st20[:10] and fx.draws.complaints(L, "ALL", 0, n_base=20) == st20,
                                               "is_all_repair_complaints": len(st20) == len(drv)})
    ok &= res["complaints"]["sampled_are_repair_D1"] and all(v for k_, v in res["complaints"]["stream_n_base_20"].items() if k_ != "n")
    lin = {}
    t1 = time.time()                                                    # COMP: joint closed-form compensation, K_ans protected (the options of P12's COMP_B)
    corr, cost = hz.HM.repair("COMP", H, cs, protect=("K_ans",), query="answer", form="adapter")
    lin["COMP"] = {"seconds": time.time() - t1, "correction_kind": corr.kind, "n_complaints": len(cs)}
    t1 = time.time()                                                    # the harness's own evaluation record (builds the W_E baseline of the prefix in the private cache)
    try:
        rec = hz.HE.evaluate(H, corr, cs, d2=False)
        lin["harness_evaluate"] = {"seconds": time.time() - t1, "ran": True, "record": {k_: v for k_, v in rec.items() if isinstance(v, (int, float, str, bool))},
                                   "baseline_file_in_private_cache": any(f.startswith(f"baseline_{name}_") for f in os.listdir(cdir))}
        ok &= lin["harness_evaluate"]["baseline_file_in_private_cache"]
    except Exception as e:                                              # noqa: BLE001
        lin["harness_evaluate"] = {"seconds": time.time() - t1, "ran": False, "error": f"{type(e).__name__}: {e}"[:300]}
        ok = False
    torch.cuda.empty_cache()
    t1 = time.time()                                                    # ROLLBACK_FO: greedy first-order rollback (applied batch updates + rank-one terms of the prefix)
    corr2, cost2 = hz.HM.repair("ROLLBACK_FO", H, cs[:1], budget=2)
    R = [int(i) for i in corr2.meta.get("R", [])]
    lin["ROLLBACK_FO"] = {"seconds": time.time() - t1, "correction_kind": corr2.kind, "R": R}
    t1 = time.time()                                                    # REEDIT: the faithful re-edit of one complaint (z* optimisation + AlphaEdit install, the prefix's keys as earlier keys)
    corr3 = None
    try:
        corr3, cost3 = hz.HM.repair("REEDIT", H, cs[:1], variant="k0+kp")
        lin["REEDIT"] = {"seconds": time.time() - t1, "correction_kind": corr3.kind, "ran": True}
    except Exception as e:                                              # noqa: BLE001
        lin["REEDIT"] = {"seconds": time.time() - t1, "ran": False, "error": f"{type(e).__name__}: {e}"[:300]}
        ok = False
    for st in ("evalcache", "baseline"):                                # the harness's cached states are not needed any more
        H._setup.pop(st, None)
    torch.cuda.empty_cache()
    # ---- 2. the driver on the prefix: session, deployed weight from the memo, the default method, the corrections as methods
    ses = fx.behave.session(L, H)
    dep = fx.methods.deployed(L)
    d = fx.prefix.describe(name)
    ed = ses.edits(dep)
    res["deployed_from_memo"] = {"rel": fx.metrics.edit_stats(ed)["rel"]["exact"], "described": d["deployed"]["rel"]["exact"], "src": sorted({r["src"] for r in ed})}
    ok &= res["deployed_from_memo"]["rel"] == res["deployed_from_memo"]["described"]
    M = fx.methods.rescope(L, draw=0)
    rows = sample_rows(L, 10, 8, 8)
    a = ses.tf(M, rows)
    g = fx.gate.decide(L, M, rows)
    res["default_method"] = {"n_rows": len(rows), "n_fired": sum(bool(r["fired"]) for r in a), "n_edits_capped": M.rec["cal"]["n_edits_capped"],
                             "flags_differ_from_gate_level": sum((bool(r["fired"]), int(r["top_last"])) != (bool(f), int(t)) for r, f, t in zip(a, g["fired"], g["top_edit"]))}
    ok &= res["default_method"]["flags_differ_from_gate_level"] == 0
    cj = [L.by_lib_sid(f"probe:{c['ptype']}:{c['probe_idx']}") for c in cs]
    probe = L.pool("heldout_disjoint")[::40]
    Mc = fx.methods.wrap(L, corr, "comp_b_n4", rec={"harness": {k_: v for k_, v in corr.meta.items() if not k_.startswith("_") and isinstance(v, (int, float, str, bool))}})
    rc = ses.tf(Mc, cj + probe)
    lib = ses.ev_plain.tf_rows(corr, [ses.item(j) for j in cj + probe])
    lin["COMP"].update(kind=Mc.kind, complaints_still_D1=sum(bool(r["D1"]) for r in rc[:len(cj)]), n_probe_rows=len(probe), rows_differ_from_plain_library=same_fields(rc, lib, FIELDS[:7]),
                       digest=Mc.digest)
    ok &= Mc.kind == "correction" and lin["COMP"]["rows_differ_from_plain_library"] == 0
    if corr2.kind == "weight":
        Mr = fx.methods.wrap(L, corr2.W, "rb_fo")
        rr = ses.tf(Mr, cj[:1] + probe)
        mine = fx.methods.rollback(L, H, R)
        lin["ROLLBACK_FO"].update(kind=Mr.kind, n_rows=len(rr), equals_driver_rollback_weight=bool(torch.equal(mine.obj, corr2.W)), all_edits_in_prefix=all(0 <= i < k for i in R),
                                  complaint_still_D1=bool(rr[0]["D1"]))
        ok &= lin["ROLLBACK_FO"]["equals_driver_rollback_weight"] and lin["ROLLBACK_FO"]["all_edits_in_prefix"]
    else:
        lin["ROLLBACK_FO"]["note"] = "no rollback was applied (the complaint was reproduced without one)"
    res["rollback_all_is_W0"] = bool(torch.equal(fx.methods.rollback(L, H, list(range(k))).obj, H.W0))
    ok &= res["rollback_all_is_W0"]
    if corr3 is not None and corr3.kind == "weight":
        r3 = ses.tf(fx.methods.wrap(L, corr3.W, "reedit_n1"), cj[:1] + probe[:4])
        lin["REEDIT"].update(n_rows=len(r3), complaint_still_D1=bool(r3[0]["D1"]), weight_differs_from_WE=not bool(torch.equal(corr3.W, H.WE)))
    lib_cache = os.path.join(fx.LIB, "rescope", "_hcache")
    res["no_prefix_file_in_library_or_harness_cache"] = not any(name in f for f in os.listdir(lib_cache)) and not any(name in f for f in os.listdir(os.path.join(rescope.evaluate.HARNESS_PARENT, "harness", "cache")))
    ok &= res["no_prefix_file_in_library_or_harness_cache"]
    res["linear"] = lin
    res["ok"] = bool(ok)
    return finish(ses, res)


TESTS = {"h": ("hashes", t_h), "m": ("mask", t_m), "r": ("rollback", t_r), "s": ("h3_pac_rowset", t_s), "k": ("readouts", t_k)}

if __name__ == "__main__":
    name, which = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    os.makedirs(OUT, exist_ok=True)
    if which == ["p"]:
        t0 = time.time()
        res = t_p(name)
        res.update(test="p", seconds=time.time() - t0, code=fx.code_fingerprint(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        fx.io.jdump(res, os.path.join(OUT, f"ext_gpu_{name}_p_prefix.json"), indent=1)
        say(f"p {name}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds']:.0f}s)")
        sys.exit(0)
    if name not in fx.logs.names("development"):
        raise SystemExit("tests h m r s k run on the development logs")
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    ses = fx.behave.session(L, H)
    for t in which:
        tag, fn = TESTS[t]
        t0 = time.time()
        res = fn(L, H, ses)
        res.update(test=t, log=name, seconds=time.time() - t0, code=fx.code_fingerprint(), time=time.strftime("%Y-%m-%d %H:%M:%S"), stats={k: v for k, v in ses.stats.items()})
        fx.io.jdump(res, os.path.join(OUT, f"ext_gpu_{name}_{t}_{tag}.json"), indent=1)
        say(f"{t} {name}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds']:.0f}s)")
