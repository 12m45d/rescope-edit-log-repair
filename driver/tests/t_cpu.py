"""CPU tests of the fx driver (development logs only for every number; other logs: mechanics only, no metric).

  python t_cpu.py [names...]          -> tests/out/cpu_<name>.json            (python render.py renders tests/out/table.txt)

  registry      every registered log that exists opens (edit log, key cache rows, pools); shapes only for the unseen logs; released values:
                batch-by-batch rebase == rescope.rebase to round-off, and bit for bit the same on the common prefix of logs of one family
  draws         n = 200 draw == rescope.splits.sample (the draw of docs/USAGE.md and of the sanity agent); draws nested in n;
                contaminated and benign-pool draws; complaint sets == the records of P12 (reported_groups of every record)
  gate          fx.gate.decide == rescope.KeyCache.decide on every array, for several gates and all positions
  regress_gate  (a) the constructors reproduce the sanity agent's M1 - M6 at the gate level: thresholds bit for bit and the firing
                decision of every sequence, 20 draws, capped and uncapped rule, positions tf / prompt / decode
  stats         bootstrap against a direct computation, exact McNemar against scipy, summaries over draws
  io_memo       result schema round trip; the memo under 6 concurrent writer processes (no lost key, no torn line, no conflict)
  methods       every constructor that needs no model builds on both development logs with every ablation switch; e1_rowset; wrap; recalibrate
  frontier      E1b / E1c: a fixed-threshold gate is a point of the sweep (read-out == gate statistics of the cosine 0.65 / exact gates); grid
                points of the frontier == the read-out at that threshold; matched-coverage thresholds reach their target and are the largest
  scale         time and peak RAM of the gate-level pass at T = 2000 (20 draws of three gates); mechanics only, no metric
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(HERE, "out")
DEV = fx.logs.names("development")
E0 = os.path.join(fx.E0, "out")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unbits(s):
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


# ----------------------------------------------------------------------------- registry
def registry():
    out = {"logs": {}, "n_registered": len(fx.logs.REGISTRY)}
    vals, pre_sig = {}, {}
    for name, e in fx.logs.REGISTRY.items():
        r = {"role": e["role"], "exists": fx.logs.exists(name)}
        if r["exists"]:
            t0 = time.time()
            L = fx.logs.open(name)
            r.update(opened=True, seconds=time.time() - t0, T=L.T, family=L.family, U=list(L.log.U.shape), W=list(L.log.W.shape), K=list(L.log.K.shape),
                     n_batches=int(L.log.batch.max()) + 1, n_edit_rows={m: len(v) for m, v in L.edit_rows.items()}, n_probe_rows=len(L.probe_rows),
                     n_hn_rows=len(L.hn_rows), n_benign_rows=len(L.benign_rows), probe_rows_equal_to_cache=L.checks["n_probe_rows_equal_to_cache"],
                     own_keys=list(L.own_keys()[0].shape), finite=bool(torch.isfinite(L.log.U).all() and torch.isfinite(L.log.W).all() and torch.isfinite(L.log.K).all()))
            # near-scope flag against the flags the cache builder stored for hard negatives and benign rows (first stream position of an edit with that question)
            bad = n = 0
            for j in L.hn_rows + L.benign_rows:
                e = L.index[j]
                ov = (e.get("overlap") or {}).get(L.family)
                pos = ov["q_in_edit_first_pos"] if ov else e.get("q_in_edit_first_pos")
                n += 1
                bad += int(L.near_edit(j, rephrase=True) != (int(pos) if pos is not None and int(pos) < L.T else None))
            r.update(near_scope_flag_checked=n, near_scope_flag_differing=bad, n_edits_asking_a_probe_question=L.describe()["n_edits_asking_a_probe_question"])
            # the log's own probe records: pre-edit fields must be the same for every log of a cache family (decode outcomes are shared through the memo)
            h = hashlib.sha1()
            for j in L.probe_rows:
                pr = L.probe_rec(j)
                h.update(repr((pr["probe_idx"], pr["ptype"], pr["y0_text"], pr["y0_valid"], pr["pre_long_text"], pr["ref_answer"], pr["pre_correct_long"])).encode())
            pre_sig.setdefault((L.reg["cache"], L.family), {})[name] = h.hexdigest()
            # released values: batch-by-batch rebase against the library's rebase; their sha1 per prefix, for the family comparison below
            V, vrec = fx.methods.released_values(L)
            r.update(values_max_rel_deviation_from_library=vrec["max_rel_deviation_from_rescope_rebase"])
            reg = fx.logs.REGISTRY[name]
            vals[name] = (L.family, reg["editor_family"], reg.get("l2"), {t: hashlib.sha1(V[:t].numpy().tobytes()).hexdigest() for t in (100, 200, 500, 1000, 2000) if t <= L.T})
            fx.methods._VALUES.pop(name, None)
            fx.logs._OPEN.pop(name, None)                               # keep the memory flat
        out["logs"][name] = r
    # logs of one family (same stream, editor and ridge) share their first edits: the released values of the common prefix are the same bits
    pairs = same = 0
    names = list(vals)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if vals[a][:3] == vals[b][:3]:
                for t in set(vals[a][3]) & set(vals[b][3]):
                    pairs += 1
                    same += int(vals[a][3][t] == vals[b][3][t])
    out["values_prefix"] = {"n_prefix_pairs": pairs, "n_bitwise_equal": same,
                            "max_rel_deviation_from_library_rebase": max(r["values_max_rel_deviation_from_library"] for r in out["logs"].values() if r["exists"])}
    out["pre_edit_probe_records"] = {f"{c}/{f}": {"n_logs": len(v), "identical": len(set(v.values())) == 1} for (c, f), v in pre_sig.items()}
    # a data property stated in the protocol (amendment of 22:10, item 5): 677 of the 2000 edits of the long Qwen stream ask a probe question
    q2000 = out["logs"].get("TB-Q2000-ae-l2_3", {})
    out["protocol_677"] = {"n_edits_asking_a_probe_question_TB-Q2000-ae-l2_3": q2000.get("n_edits_asking_a_probe_question"), "protocol": 677,
                           "ok": (not q2000.get("exists")) or q2000.get("n_edits_asking_a_probe_question") == 677}
    out["n_existing"] = sum(r["exists"] for r in out["logs"].values())
    out["missing"] = [n for n, r in out["logs"].items() if not r["exists"]]
    out["register"] = _register_elsewhere()
    R = fx.logs.REGISTRY                                                # roles as registered (R18 withdrawn); what R19 cut is flagged, nothing else
    cut = set(fx.logs.names(("internal", "stress", "unseen_onehop"))) | {"TB-Q200-memitw", "TB-L200-memitw"}
    out["roles"] = {"by_role": {r: len(fx.logs.names(r)) for r in fx.logs.ROLES}, "never_reported": sorted(n for n, e in R.items() if e["reported"] is False),
                    "cut_by_R19": sorted(n for n, e in R.items() if e["cut_by_R19"]),
                    "ok": {n for n, e in R.items() if e["reported"] is False} == set(fx.logs.names("internal")) and {n for n, e in R.items() if e["cut_by_R19"]} == cut
                    and not any("pilot" in e for e in R.values()) and fx.logs.names("development") == ["TB-Q100-ae-l2_3", "TB-L100-ae-l2_3"]}
    out["ok"] = all(r.get("opened") and r["finite"] and r["probe_rows_equal_to_cache"] == 1000 and r["near_scope_flag_differing"] == 0
                    for r in out["logs"].values() if r["exists"]) and all(fx.logs.exists(n) for n in DEV) \
        and out["values_prefix"]["n_bitwise_equal"] == out["values_prefix"]["n_prefix_pairs"] and out["values_prefix"]["max_rel_deviation_from_library_rebase"] < 1e-12 \
        and out["protocol_677"]["ok"] and all(v["identical"] for v in out["pre_edit_probe_records"].values()) and out["register"]["ok"] and out["roles"]["ok"]
    return out


def _null_by_query(L):
    """The null-repair sample against the calibration draw BY QUERY (amendment of 2026-11-11 09:03), for the 20 draws: a naive
    re-implementation of the rule; no query shared; 100 distinct queries; how many calibration queries the rule used until then
    (positions 200 .. 299 of the ranking) shared with the null sample (the amendment names 5 (Qwen) and 3 (LLaVA) at draw 0); an
    explicit null sample that holds a calibration query is refused."""
    pool, ix = L.pool("repair"), L.index
    q = lambda rows: [ix[j]["query_id"] for j in rows]                                             # noqa: E731
    res = {"n_draws": fx.draws.N_DRAWS, "naive_equal": 0, "shared_queries_max": 0, "distinct_queries_min": 10 ** 9, "old_rule_shared_calibration_rows": [],
           "rows_changed_vs_old_rule": []}
    twin = None
    for d in range(fx.draws.N_DRAWS):
        A, cal = fx.draws.null_sample(L, 100, d), fx.draws.calibration(L, 200, d).rows
        rk = fx.draws.ranking(len(pool), d)
        seen, naive = set(q([pool[int(k)] for k in rk[:200]])), []
        for k in rk[200:]:
            j = pool[int(k)]
            if len(naive) < 100 and ix[j]["query_id"] not in seen:
                seen.add(ix[j]["query_id"])
                naive.append(j)
        old = sorted(pool[int(k)] for k in rk[200:300])
        oq = set(q(old))
        res["naive_equal"] += int(sorted(naive) == A and set(cal) == {pool[int(k)] for k in rk[:200]})
        res["shared_queries_max"] = max(res["shared_queries_max"], len(set(q(A)) & set(q(cal))))
        res["distinct_queries_min"] = min(res["distinct_queries_min"], len(set(q(A))))
        res["old_rule_shared_calibration_rows"].append(sum(x in oq for x in q(cal)))
        res["rows_changed_vs_old_rule"].append(len(set(old) - set(A)))
        if twin is None:
            tw = [j for j in old if ix[j]["query_id"] in set(q(cal))]
            twin = (d, tw[0]) if tw else None
    refused = None
    if twin is not None:                                                # an explicit null sample holding a calibration query must be refused
        try:
            fx.methods.rescope(L, draw=twin[0], null_rows=[twin[1]])
            refused = False
        except ValueError as e:
            refused = "shares 1 queries" in str(e)
    res.update(explicit_sample_with_a_calibration_query_refused=refused, old_rule_shared_at_draw_0=res["old_rule_shared_calibration_rows"][0],
               protocol_says_at_draw_0={"Q": 5, "L": 3}[L.family])
    res["ok"] = bool(res["naive_equal"] == fx.draws.N_DRAWS and res["shared_queries_max"] == 0 and res["distinct_queries_min"] == 100 and refused is not False
                     and res["old_rule_shared_at_draw_0"] == res["protocol_says_at_draw_0"] and fx.draws.null_sample(L, 0, 0) == [])
    return res


def _register_elsewhere():
    """A log registered at run time that is NOT under 06_testbed_v3: register(path=, setup=<callable>) on a link to a development
    log in another directory; the log opens from the registered directory and gives that log's factors, thresholds and released
    values bit for bit."""
    import tempfile
    tmp = tempfile.mkdtemp(prefix="fx_regtest_", dir=OUT)
    name, src, called = "TW-TEST100", DEV[0], []
    try:
        os.symlink(os.path.join(fx.TESTBED, src), os.path.join(tmp, name))
        e = fx.logs.register(name, "primary", fx.logs.REGISTRY[src]["backbone"], fx.logs.REGISTRY[src]["layer"], fx.logs.REGISTRY[src]["family"], fx.logs.REGISTRY[src]["T"],
                             path=os.path.join(tmp, name), setup=lambda: called.append(1), reported=True)
        La, Lb = fx.logs.open(src), fx.logs.open(name)
        Ma, Mb = fx.methods.rescope(La, draw=0), fx.methods.rescope(Lb, draw=0)
        fx.logs.setup(name)
        r = {"exists": fx.logs.exists(name), "role": e["role"], "cut_by_R19": e["cut_by_R19"], "in_names": name in fx.logs.names("primary"),
             "factors_equal": bool(torch.equal(La.log.U, Lb.log.U) and torch.equal(La.log.W, Lb.log.W) and torch.equal(La.log.K, Lb.log.K)),
             "thresholds_equal": bool(torch.equal(Ma.obj.theta(), Mb.obj.theta())), "values_equal": bool(torch.equal(Ma.obj.values, Mb.obj.values)),
             "setup_called": called == [1], "entry_serialisable": bool(json.dumps(fx.io.clean(fx.logs.info(name))))}
    finally:
        fx.logs.REGISTRY.pop(name, None)
        fx.logs.ROLES.pop("primary", None)
        fx.logs._OPEN.pop(name, None)
        fx.methods._VALUES.pop(name, None)
        for k in [k for k in fx.methods._BASE if k[0] == name]:
            fx.methods._BASE.pop(k)
        if os.path.islink(os.path.join(tmp, name)):
            os.remove(os.path.join(tmp, name))
        os.rmdir(tmp)
    r["ok"] = bool(r["exists"] and r["role"] == "primary" and r["cut_by_R19"] is False and r["in_names"] and r["factors_equal"] and r["thresholds_equal"]
                   and r["values_equal"] and r["setup_called"] and r["entry_serialisable"])
    return r


# ----------------------------------------------------------------------------- draws
def draws():
    from rescope import splits
    out = {"logs": {}}
    ok = True
    for name in DEV:
        L = fx.logs.open(name)
        pool = L.kc.log_rows(L.uids, kind="probe", pool="repair", valid=True)
        r = {"n_pool": len(pool), "default_equals_library": 0, "nested": True, "sorted_in_pool_order": True}
        for d in range(fx.draws.N_DRAWS):
            dr = fx.draws.calibration(L, 200, d)
            r["default_equals_library"] += int(dr.rows == splits.sample(pool, 200, d))
            prev = set()
            for n in (1, 2, 5, 10, 20, 40, 50, 100, 200, 300, len(pool)):
                rows = fx.draws.calibration(L, n, d).rows
                r["nested"] &= prev <= set(rows) and len(set(rows)) == n
                r["sorted_in_pool_order"] &= rows == sorted(rows)
                prev = set(rows)
        c = {}
        for p in fx.draws.CONTAMINATION:
            dr = fx.draws.calibration(L, 200, 0, contamination=p)
            k = int(round(p * 200))
            ins = [j for j in dr.rows if L.index[j]["kind"] in ("tgen", "igen")]
            c[str(p)] = {"n": len(dr.rows), "n_in_scope": len(ins), "expected": k, "all_half_A": all(L.own(j) < L.T // 2 for j in ins),
                         "benign_part_nested_in_clean_draw": set(dr.rows[:200 - k]) <= set(fx.draws.calibration(L, 200, 0).rows)}
            ok &= len(dr.rows) == 200 and len(ins) == k and c[str(p)]["all_half_A"] and c[str(p)]["benign_part_nested_in_clean_draw"]
        c5, c20 = fx.draws.calibration(L, 200, 0, contamination=0.05), fx.draws.calibration(L, 200, 0, contamination=0.2)
        r["contamination"] = c
        r["contaminating_queries_nested"] = set(c5.rec["contaminating_sids"]) <= set(c20.rec["contaminating_sids"])
        b = {}
        for which in ("tf", "decode", "steps"):
            dr = fx.draws.calibration(L, 200, 0, source="BEN-S", which=which)      # stage-1 extension: the part is named (review finding 3)
            b[which] = {"n_pool": dr.rec["n_pool"], "n": len(dr.rows), "all_benign": all(L.index[j]["kind"] == "benign" for j in dr.rows),
                        "have_key": all(L.index[j].get("row_tf" if which == "tf" else "row_decode") is not None for j in dr.rows)}
            ok &= b[which]["all_benign"] and b[which]["have_key"] and b[which]["n"] == 200
        r["benign"] = b
        v = fx.draws.verify_p12(L)
        r["p12"] = {k: v[k] for k in ("available", "n_records", "n_equal", "seeds", "ns", "differing")}
        s0 = fx.draws.complaint_stream(L, 0)
        r["complaints"] = {"n_stream": len(s0), "nested": fx.draws.complaints(L, 10, 0) == s0[:10] and fx.draws.complaints(L, "ALL", 0) == s0,
                           "all_repair_valid_D1": all(L.index[j]["pool"] == "repair" and L.index[j]["y0_valid"] and L.probe_rec(j)["is_complaint_D1"] for j in s0),
                           "first_100_stratified": sum(L.index[j]["ptype"] == "image" for j in s0[:100]) == 50}
        ok &= (r["default_equals_library"] == fx.draws.N_DRAWS and r["nested"] and r["sorted_in_pool_order"] and r["contaminating_queries_nested"]
               and v.get("ok", False) and all(r["complaints"][k] for k in ("nested", "all_repair_valid_D1", "first_100_stratified")))
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- gate == library
def gate():
    out = {"logs": {}}
    ok = True
    for name in DEV:
        L = fx.logs.open(name)
        sets = fx.gate.row_sets(L, benign=False)
        rows = sets["probe"] + sets["edit"] + sets["hn"]
        ms = {"rescope_d0": fx.methods.rescope(L, draw=0), "rescope_d7_a0.2": fx.methods.rescope(L, draw=7, alpha=0.2), "shared_d3": fx.methods.rescope_shared(L, draw=3),
              "cosine_d1": fx.methods.cosine(L, draw=1), "cosine_default": fx.methods.cosine(L, "default"), "ball_d2": fx.methods.ball(L, draw=2),
              "ball_default": fx.methods.ball(L, "default"), "exact": fx.methods.exact(L), "logonly": fx.methods.rescope_logonly(L)}
        r = {"n_rows": len(rows), "n_compared": 0, "n_differing": 0, "differing": [], "seconds_fx": 0.0, "seconds_library": 0.0}
        for lab, M in ms.items():
            for which in ("tf", "prompt", "decode", "steps"):
                t0 = time.time()
                a = fx.gate.decide(L, M, rows, which)
                t1 = time.time()
                b = fx.gate.library_decide(L, M, rows, which)
                r["seconds_fx"] += t1 - t0
                r["seconds_library"] += time.time() - t1
                for k in ("fired", "near", "top_edit", "own", "own_cand", "own_top", "n_cand"):
                    x, y = np.asarray(a[k]), np.asarray(b[k])
                    r["n_compared"] += int(x.size)
                    nd = int((x != y).sum())
                    if nd:
                        r["n_differing"] += nd
                        r["differing"].append({"method": lab, "which": which, "field": k, "n": nd})
                dm = np.abs(np.asarray(a["margin"]) - np.asarray(b["margin"]))
                dm = dm[np.isfinite(dm)]
                r["n_compared"] += int(dm.size)
                if dm.size and float(dm.max()) > 0:
                    r["n_differing"] += int((dm > 0).sum())
                    r["differing"].append({"method": lab, "which": which, "field": "margin", "max_abs": float(dm.max())})
        Rf = fx.gate.features(L, ms["rescope_d0"], rows[:200])
        r["features_equal_library_coords"] = bool(torch.equal(Rf, L.kc.coords(ms["rescope_d0"].obj, rows[:200])))
        ok &= r["features_equal_library_coords"]
        bn = L.rows("benign")
        a, b = fx.gate.decide(L, ms["rescope_d0"], bn, "decode"), fx.gate.library_decide(L, ms["rescope_d0"], bn, "decode")
        r["benign_equal"] = bool((np.asarray(a["fired"]) == np.asarray(b["fired"])).all() and (np.asarray(a["top_edit"]) == np.asarray(b["top_edit"])).all())
        ok &= r["n_differing"] == 0 and r["benign_equal"]
        out["logs"][name] = r
        say(f"  gate {name}: {r['n_compared']} values, {r['n_differing']} differing; fx {r['seconds_fx']:.1f}s, library {r['seconds_library']:.1f}s")
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- (a) regression against the sanity agent, gate level
E0_METHODS = {"M1": ("rescope", {}), "M2": ("rescope_shared", {}), "M3": ("cosine", {"calibration": "two_block"}), "M4": ("cosine", {"calibration": "shared"}),
              "M5": ("cosine", {"calibration": "default"}), "M6": ("exact", {})}


def e0_method(L, mid, draw=None, **kw):
    name, k = E0_METHODS[mid]
    k = dict(k, **kw)
    if draw is not None:
        k.update(alpha=0.05, n=200, draw=draw)
    return fx.methods.build(name, L, **k)


def regress_gate():
    out = {"logs": {}, "source": E0}
    ok = True
    for name in DEV:
        p = os.path.join(E0, f"gate_{name}.json")
        if not os.path.exists(p):
            out["logs"][name] = {"available": False}
            ok = False
            continue
        Gj = fx.io.jload(p)
        L = fx.logs.open(name)
        rows = [L.kc.by_sid(r["sid"]) for r in Gj["rows"]]
        r = {"available": True, "n_rows": len(rows), "n_draws": Gj["n_draws"], "methods": {}, "draws_equal": True, "e0_code": Gj["code"]["rescope"],
             "fx_rescope": fx.code_fingerprint()["rescope"]}
        for d in range(Gj["n_draws"]):
            mine = [L.lib_sid(j) for j in fx.draws.calibration(L, 200, d).rows]
            r["draws_equal"] &= mine == Gj["draws"][d]["calibration_sids"]
        for mid in E0_METHODS:
            m = {"n_gates": 0, "n_decisions": 0, "n_differing": 0, "tau_bitwise": 0, "max_abs_eta_diff": 0.0, "capped_equal": True, "differing": []}
            if "fixed" in Gj["methods"][mid]:
                cases = [(None, {}, Gj["methods"][mid]["fixed"], ("tf", "prompt", "decode"), "tau")]
            else:
                cases = []
                for e in Gj["methods"][mid]["draws"]:
                    cases.append((e["seed"], {}, e, ("tf", "prompt", "decode"), "tau"))
                    cases.append((e["seed"], {"allow_silence": True}, {"tf": e["tf_uncapped"], "tau_uncapped": e["tau_uncapped"], "rec": e["rec_uncapped"]},
                                  ("tf",), "tau_uncapped"))
            for seed, kw, e, variants, tk in cases:
                M = e0_method(L, mid, seed, **kw)
                m["n_gates"] += 1
                mine_tau = [float(x) for x in M.obj.theta()]
                theirs = [float(x) if not isinstance(x, str) else float(x) for x in e[tk]]
                m["tau_bitwise"] += int(mine_tau == theirs)
                if "rec" in e:
                    m["max_abs_eta_diff"] = max(m["max_abs_eta_diff"], abs(float(M.rec["cal"]["eta"]) - float(e["rec"]["eta"])))
                    m["capped_equal"] &= M.rec["cal"]["n_edits_capped"] == e["rec"]["n_edits_capped"]
                for v in variants:
                    dd = fx.gate.decide(L, M, rows, v)
                    for k in ("fired", "own_cand", "own_top"):
                        diff = np.asarray(dd[k]) != unbits(e[v][k])
                        m["n_decisions"] += int(diff.size)
                        m["n_differing"] += int(diff.sum())
                        if diff.any() and len(m["differing"]) < 5:
                            m["differing"].append({"seed": seed, "variant": v, "field": k, "sids": [L.sid(rows[i]) for i in np.nonzero(diff)[0][:3]], **kw})
                    dt = np.asarray(dd["top_edit"]) != np.asarray(e[v]["top_edit"])
                    m["n_decisions"] += int(dt.size)
                    m["n_differing"] += int(dt.sum())
            r["methods"][mid] = m
            say(f"  regress_gate {name} {mid}: {m['n_gates']} gates, thresholds bitwise {m['tau_bitwise']}, {m['n_decisions']} decisions, {m['n_differing']} differing")
        r["n_decisions"] = sum(m["n_decisions"] for m in r["methods"].values())
        r["n_differing"] = sum(m["n_differing"] for m in r["methods"].values())
        r["n_gates"] = sum(m["n_gates"] for m in r["methods"].values())
        r["n_tau_bitwise"] = sum(m["tau_bitwise"] for m in r["methods"].values())
        # the sanity agent's table numbers (heldout_disjoint firing, coverage) recomputed through fx.metrics for the 20 draws
        tab = {}
        e0res = os.path.join(E0, "results.json")
        if os.path.exists(e0res):
            R0 = fx.io.jload(e0res)["logs"][name]["gate"]["methods"]
            for mid in ("M1", "M2", "M3", "M4"):
                per = [fx.metrics.gate_stats(L, fx.gate.evaluate(L, e0_method(L, mid, d), parts=("probe", "edit", "hn"))) for d in range(Gj["n_draws"])]
                mine = {"fire": np.mean([s["probe"]["heldout_disjoint"]["fired"] for s in per]), "tgen": np.mean([s["edit"]["tgen"]["own_top"] for s in per]),
                        "igen": np.mean([s["edit"]["igen"]["own_top"] for s in per]), "hn_i": np.mean([s["hn"]["HN-I"]["fired"] for s in per])}
                t = R0[mid]["tf"]
                ref = {"fire": t["probe"]["heldout_disjoint"]["fired"]["mean"], "tgen": t["edit"]["tgen"]["own_top"]["mean"],
                       "igen": t["edit"]["igen"]["own_top"]["mean"], "hn_i": t["hn"]["HN-I"]["fired"]["mean"]}
                tab[mid] = {"fx": {k: float(v) for k, v in mine.items()}, "e0": ref, "max_abs_diff": max(abs(float(mine[k]) - ref[k]) for k in ref)}
        r["table_means"] = tab
        ok &= r["draws_equal"] and r["n_differing"] == 0 and r["n_tau_bitwise"] == r["n_gates"] and all(v["max_abs_diff"] < 1e-9 for v in tab.values())
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- statistics
def stats():
    from scipy.stats import binomtest
    S = fx.stats
    rng = np.random.default_rng(1)
    out = {}
    v = rng.random(300) < 0.3
    u = rng.integers(0, 120, 300)
    b = S.bootstrap(v, u, scale=100.0)
    inv = np.unique(u, return_inverse=True)[1]
    nu = inv.max() + 1
    idx = np.random.default_rng(0).integers(0, nu, size=(2000, nu))    # the documented resampling, written out directly
    direct = []
    for row in idx:
        cnt = np.bincount(row, minlength=nu)[inv]
        direct.append(100.0 * (cnt * v).sum() / cnt.sum())
    out["bootstrap"] = {"value": b["value"], "ci": [b["ci_lo"], b["ci_hi"]], "direct_ci": [float(np.percentile(direct, 2.5)), float(np.percentile(direct, 97.5))],
                        "n_units": b["n_units"], "value_ok": abs(b["value"] - 100.0 * v.mean()) < 1e-12}
    out["bootstrap"]["ok"] = bool(out["bootstrap"]["value_ok"] and np.allclose(out["bootstrap"]["ci"], out["bootstrap"]["direct_ci"], atol=1e-9) and b["n_units"] == nu)
    a2, b2 = rng.random(200) < 0.6, rng.random(200) < 0.4
    d = S.bootstrap_diff(a2, b2, np.arange(200), scale=100.0)
    out["paired"] = {"value": d["value"], "ci": [d["ci_lo"], d["ci_hi"]], "p": d["p_two_sided"], "ok": bool(abs(d["value"] - 100.0 * (a2.mean() - b2.mean())) < 1e-9
                                                                                                         and d["ci_lo"] < d["value"] < d["ci_hi"] and d["p_two_sided"] < 0.01)}
    same = S.bootstrap_diff(a2, a2, np.arange(200))
    out["paired"]["ok"] &= same["value"] == 0.0 and same["ci_lo"] == 0.0 and same["ci_hi"] == 0.0 and same["p_two_sided"] == 1.0
    cases = []
    for _ in range(30):
        x, y = rng.random(80) < rng.random(), rng.random(80) < rng.random()
        m = S.mcnemar_exact(x, y)
        n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
        ref = 1.0 if n10 + n01 == 0 else binomtest(min(n10, n01), n10 + n01, 0.5, alternative="two-sided").pvalue
        cases.append(abs(m["p"] - ref) < 1e-12 and m["n10"] == n10 and m["n01"] == n01)
    out["mcnemar"] = {"n_cases": len(cases), "n_equal_scipy": int(sum(cases)), "ok": all(cases)}
    od = S.over_draws([1.0, 2.0, 3.0, 4.0, None])
    out["over_draws"] = {"res": od, "ok": od["n_draws"] == 4 and od["mean"] == 2.5 and abs(od["p05"] - 1.15) < 1e-12 and abs(od["p95"] - 3.85) < 1e-12}
    r = S.bootstrap_ratio([1, 0, 1, None], [1, 1, 1, 1], [0, 0, 1, 1])
    out["ratio_none"] = {"res": r, "ok": r["n_dropped"] == 1 and abs(r["value"] - 2 / 3) < 1e-12}
    out["ok"] = all(v["ok"] for v in out.values())
    return out


# ----------------------------------------------------------------------------- io + memo
NKEY = 3000


def _writer(args):
    root, w, n, start_at = args
    import fx as _fx
    m = _fx.memo.Memo("testbb", root=root)
    while time.time() < start_at:                                       # all writers (and the reader) start together
        time.sleep(0.01)
    t0 = time.time()
    for i in range(n):
        k = (i * 7 + w * 13) % NKEY                                     # every writer writes every key: identical outcomes
        m.put(f"seq{k}", "tf", f"v:{k % 5}", {"exact": bool(k % 2), "token_acc": k / float(NKEY), "n_tok": k % 11, "finite": True, "first_mismatch": None if k % 2 else k % 3})
        if i % 200 == 199:
            m.refresh()
    return {"writes": m.stats["writes"], "start": t0, "end": time.time(), "conflicts": m.stats["conflicts"]}


def _reader(args):
    root, start_at, seconds = args
    import fx as _fx
    m = _fx.memo.Memo("testbb", root=root)
    while time.time() < start_at:
        time.sleep(0.01)
    n = 0
    while time.time() < start_at + seconds:                             # reads while the writers append: complete lines only
        m.refresh()
        n += 1
    return {"refreshes": n, "entries": len(m.data), "conflicts": m.stats["conflicts"],
            "values_ok": all(v["token_acc"] == int(k.split("|")[0][3:]) / float(NKEY) for k, v in m.data.items())}


def io_memo():
    out = {}
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        R = fx.io.Results("T0", os.path.join(tmp, "out"))
        R.add("TB-Q100-ae-l2_3", "rescope", "tgen_exact", 62.0, ci_lo=52.0, ci_hi=71.0, n_units=100, alpha=0.05, n=200, draw=0, pool="edits")
        R.add_draws("TB-Q100-ae-l2_3", "rescope", "fire", {0: 4.0, 1: 6.0, 2: 5.0}, alpha=0.05, n=200, pool="heldout_disjoint")
        R.add_nested("TB-Q100-ae-l2_3", "cosine", {"probe": {"D1": 1.5, "n": 465, "fired": {"value": 3.0, "ci_lo": 2.0, "ci_hi": 4.0, "n_units": 240}}}, alpha=0.05)
        R.rows("rescope_draw0", [{"sid": "probe:image:0", "D1": False, "x": float("inf")}])
        p = R.save(note="test")
        R.paper({"TestQwenTGen": 62.0})
        R.paper({"TestQwenFire": 5.0})
        R.table(fx.io.pivot(R.records, row=("method",), col=("metric",), where=lambda r: r["draw"] in (0, None) and "@p" not in r["metric"]))
        back = fx.io.jload(p)
        rows = fx.io.load_rows(os.path.join(tmp, "out", "rows", "rescope_draw0.jsonl.gz"))
        out["io"] = {"n_records": len(back), "fields_ok": all(all(f in r for f in fx.io.FIELDS) for r in back), "is_list": isinstance(back, list),
                     "summary_records": sorted(r["metric"] for r in back if "@" in r["metric"]), "rows_back": rows,
                     "paper": fx.io.jload(os.path.join(tmp, "out", "paper_numbers.json")), "table": open(os.path.join(tmp, "out", "table.txt")).read()}
        bad = None
        try:
            R.paper({"bad_name1": 1.0})
        except ValueError as e:
            bad = str(e)
        out["io"]["ok"] = bool(out["io"]["fields_ok"] and out["io"]["is_list"] and len(back) == 1 + 6 + 3 and rows[0]["x"] == "inf" and bad
                               and out["io"]["paper"] == {"TestQwenFire": 5.0, "TestQwenTGen": 62.0})
        # the memo under concurrent writers
        root = os.path.join(tmp, "memo")
        t0 = time.time()
        start_at = time.time() + 12.0                                   # after every process has imported fx
        with mp.get_context("spawn").Pool(7) as pool:
            rd = pool.apply_async(_reader, ((root, start_at, 3.0),))
            wr = pool.map(_writer, [(root, w, 2 * NKEY, start_at) for w in range(6)])
            rd = rd.get()
        writes = [x["writes"] for x in wr]
        overlap = min(x["end"] for x in wr) - max(x["start"] for x in wr)       # > 0: all six were writing at the same time
        m = fx.memo.Memo("testbb", root=root)
        keys = {f"seq{k}|tf|v:{k % 5}" for k in range(NKEY)}
        vals_ok = all(m.data[k]["token_acc"] == int(k.split("|")[0][3:]) / float(NKEY) for k in keys if k in m.data)
        m2 = fx.memo.Memo("testbb", root=root)
        m2.put("seq1", "tf", "v:1", {"exact": False, "token_acc": 0.123, "n_tok": 1, "finite": True, "first_mismatch": 0})       # another outcome: a conflict
        old = fx.memo.Memo("testbb", root=root)
        old.accept = {"another-library-version"}                        # a reader running another version of the library: nothing is accepted
        old.data.clear(), old._off.clear(), old._hdr.clear()
        old.refresh()
        os.environ["FX_MEMO_ACCEPT"] = m.code                           # ... unless that version is named in FX_MEMO_ACCEPT
        try:
            adm = fx.memo.Memo("testbb", root=root)
            adm.code = "another-library-version"
            n_adm = len(adm.data)
        finally:
            del os.environ["FX_MEMO_ACCEPT"]
        out["memo"] = {"writers": 6, "writes_per_writer": writes, "seconds_all_writing_together": overlap, "n_duplicate_writes": sum(writes) - len(keys),
                       "conflicts_seen_by_writers": sum(x["conflicts"] for x in wr), "concurrent_reader": rd,
                       "n_expected_keys": len(keys), "n_keys_read": len(m.data), "all_present": keys <= set(m.data),
                       "values_ok": bool(vals_ok), "conflicts_between_writers": m.stats["conflicts"], "shards": m.stats["shards"], "seconds": time.time() - t0,
                       "conflict_detected": m2.stats["conflicts"] == 1 and m2.data["seq1|tf|v:1"]["token_acc"] == 1 / float(NKEY),
                       "other_library_version_ignored": len(old.data) == 0 and old.stats["shards_other_code"] >= 1,
                       "accept_list_admits_named_version": n_adm == len(keys)}
        out["memo"]["ok"] = bool(out["memo"]["all_present"] and vals_ok and len(m.data) == len(keys) and m.stats["conflicts"] == 0 and out["memo"]["conflict_detected"]
                                 and overlap > 0 and all(w > 0 for w in writes) and m.stats["shards"] == 6 and rd["values_ok"] and rd["conflicts"] == 0
                                 and out["memo"]["conflicts_seen_by_writers"] == 0
                                 and out["memo"]["other_library_version_ignored"] and out["memo"]["accept_list_admits_named_version"])
    out["ok"] = out["io"]["ok"] and out["memo"]["ok"]
    return out


# ----------------------------------------------------------------------------- constructors
def methods():
    out = {"logs": {}}
    ok = True
    variants = [("rescope", {}), ("rescope", {"alpha": 0.01}), ("rescope", {"alpha": 0.2, "n": 100}), ("rescope", {"split": (1, 3)}), ("rescope", {"split": (3, 1)}),
                ("rescope", {"baseline": 0.98}), ("rescope", {"feature": "coord"}), ("rescope", {"rebase": False}), ("rescope", {"fire": "all"}),
                ("rescope", {"response": "gain"}), ("rescope", {"response": "lin"}), ("rescope", {"role": "last+answer", "calibration": "log_only"}),
                ("rescope", {"calibration": "log_only"}), ("rescope", {"calibration": "self"}), ("rescope", {"calibration": "running_max"}),
                ("rescope", {"source": "BEN-S"}), ("rescope", {"source": "BEN-S", "which": "decode"}), ("rescope", {"contamination": 0.05}),
                ("rescope", {"guards": False}), ("rescope", {"allow_silence": True}), ("rescope", {"which": "all", "guards": "all"}), ("rescope", {"which": "both"}),
                ("rescope_shared", {}), ("rescope_null", {}), ("rescope_null", {"ridge": 0.1}), ("rescope_logonly", {}),
                ("cosine", {}), ("cosine", {"calibration": "shared"}), ("cosine", {"calibration": "default"}), ("cosine", {"calibration": "default", "role": "all"}),
                ("ball", {}), ("ball", {"calibration": "shared"}), ("ball", {"calibration": "default"}), ("ball", {"calibration": "half_min"}), ("exact", {}),
                ("deployed", {}), ("preedit", {}), ("dual", {}), ("dual", {"variant": "contains"})]
    for name in DEV:
        L = fx.logs.open(name)
        r = {"n_variants": len(variants), "built": 0, "keys": {}, "records_complete": True, "notes": {}}
        for nm, kw in variants:
            M = fx.methods.build(nm, L, **kw)
            tag = nm + "".join(f"/{k}={v}" for k, v in kw.items())
            r["built"] += 1
            r["keys"][tag] = M.key
            r["records_complete"] &= bool(M.key) and M.rec.get("log") == name and (not hasattr(M.obj, "det_tau") or ("thresholds" in M.rec and "gate" in M.rec))
            if nm == "rescope" and kw.get("alpha") == 0.01:
                r["notes"]["alpha_0.01_eta_finite"] = not M.rec["cal"]["eta_infinite"]
            if kw.get("baseline") == 0.98:
                base = fx.methods.rescope(L)
                r["notes"]["baseline_q98_below_max"] = bool((M.obj.det_b <= base.obj.det_b).all() and (M.obj.det_b < base.obj.det_b).any()) and M.rec["cal"]["n1"] == 100
            if kw.get("which") == "all":                                 # decode-proof variant: the calibration queries are silent on every key of both paths
                dr = fx.draws.calibration(L, 200, 0, which="all")
                blk2 = [dr.rows[i] for i in M.rec["cal"]["block2"]]
                fire = {w: int(fx.gate.decide(L, M, blk2, w)["fired"].sum()) for w in ("tf", "prompt", "steps")}
                own = fx.gate.decide(L, M, L.edit_rows["rel"], "steps")
                r["notes"]["which_all"] = {"block2_firing": fire, "n_allowed": M.rec["cal"]["n_allowed_above"], "n_guards": M.rec["gate"]["n_guards"],
                                           "own_prompts_fire_in_every_decode_forward": int(own["step_fired"].all(1).sum()), "variant": M.variant}
                r["notes"]["which_all_bound_holds_on_block2"] = all(v <= M.rec["cal"]["n_allowed_above"] + M.rec["cal"]["n_edits_capped"] * 0 for v in fire.values()) or M.rec["cal"]["capped"]
            if kw.get("calibration") == "running_max":
                Q = L.kc.coords(M.obj, fx.draws.calibration(L, 200, 0).rows)
                r["notes"]["running_max_silent_on_calibration"] = not bool((Q >= M.obj.theta()[None, :]).any()) or M.rec["cal"]["capped"]
        ev5 = fx.methods.evidence(L, 5, 0)
        ev40 = fx.methods.evidence(L, 40, 0)
        r["notes"]["evidence_n5"] = {"rule": ev5.rec["calibration"], "least": ev5.rec["least_repair"]["n_queries"], "n": ev5.n}
        r["notes"]["evidence_n40"] = {"rule": ev40.rec["calibration"], "n": ev40.n, "eta_finite": not ev40.rec["cal"]["eta_infinite"]}
        from rescope import least_repair, null_repair
        base = fx.methods.rescope(L)
        fire = [j for j, f in zip(L.pool("repair"), fx.gate.decide(L, base, L.pool("repair"))["fired"]) if f][:3]
        S2 = least_repair(base.obj, [L.kc.keys("tf", [j]) for j in fire])[0]
        Wl = fx.methods.wrap(L, S2, "rescope+least", rec={"from": base.key, "op": "least_repair", "sids": [L.sid(j) for j in fire]})
        Wn = fx.methods.wrap(L, null_repair(base.obj, L.kc.keys("tf", fire))[0], "rescope+null", rec={"from": base.key, "op": "null_repair"})
        r["wrap"] = {"kinds": [Wl.kind, Wn.kind], "memo": [Wl.memo, Wn.memo], "keys_differ": len({Wl.key, Wn.key, base.key}) == 3,
                     "least_silences_complaints": not fx.gate.decide(L, Wl, fire)["fired"].any() and len(fire) == 3,
                     "null_repair_changes_features": bool((fx.gate.decide(L, Wn, fire)["margin"] != fx.gate.decide(L, base, fire)["margin"]).all())}
        S0 = fx.methods.base_gate(L)[0]
        Sr, rr = fx.methods.recalibrate(L, S0, 0.05, 200, 4)
        Sx, rx = fx.methods.recalibrate(L, Wn.obj, 0.05, 200, 4, exclude=fire)
        r["recalibrate"] = {"equals_rescope_bitwise": bool(torch.equal(Sr.det_tau, fx.methods.rescope(L, draw=4).obj.det_tau)),
                            "excluded_rows_absent": not (set(L.sid(j) for j in fire) & set(rx["draw"]["sids"])) and len(rx["draw"]["sids"]) == 200,
                            "excluded_queries_absent": not (set(fx.draws.query_ids(L, fire)) & set(fx.draws.query_ids(L, [L.kc.by_sid(x) for x in rx["draw"]["sids"]]))),
                            "n_excluded": rx["draw"]["n_excluded"], "n_pool_rows_with_an_excluded_query": sum(x in set(fx.draws.query_ids(L, fire)) for x in fx.draws.query_ids(L, L.pool("repair"))),
                            "eta_after_null_repair_finite": not rx["cal"]["eta_infinite"]}
        # ReScope-N (registered variant): against an independent assembly from the library's functions; samples disjoint; A keys cleared
        from rescope import add_guards, calibrate_two_block, feature_gate
        Mn = fx.methods.rescope_null(L, alpha=0.05, draw=2)
        A, cal = fx.draws.null_sample(L, 100, 2), fx.draws.calibration(L, 200, 2).rows
        Sa = add_guards(null_repair(feature_gate(L.log, "ncoord"), L.kc.keys("tf", A), ridge=0.0)[0], *L.kc.own_keys(L.uids))[0]
        Sb = calibrate_two_block(Sa, coords=L.kc.coords(Sa, cal), alpha=0.05, seed=2)[0]
        r["rescope_null"] = {"thresholds_bitwise_equal_independent_assembly": bool(torch.equal(Sb.det_tau, Mn.obj.det_tau) and torch.equal(Sb.det_w, Mn.obj.det_w)),
                             "samples_disjoint": not (set(A) & set(cal)) and len(set(A)) == 100, "A_in_repair_pool": set(A) <= set(L.pool("repair")),
                             "by_query": _null_by_query(L),
                             "max_abs_feature_on_A": float(fx.gate.features(L, Mn, A).abs().max()), "n_edits_capped": Mn.rec["cal"]["n_edits_capped"],
                             "norm_inflation_max": Mn.rec["null_repair"]["detector_norm_inflation"]["max"], "memo_path": Mn.memo, "variant": Mn.variant,
                             "values_unchanged": bool(torch.equal(Mn.obj.values, base.obj.values))}
        # disjoint parts C0 / S / C1 (independence of the recalibration sample) and explicit calibration samples
        C0, S_, C1 = fx.draws.partition(L, (160, 160, 160), 5)
        Mc = fx.methods.rescope(L, cal=C0)
        Me = fx.methods.rescope(L, cal=C1, null_rows=L.rows("hn", valid=True)[:30], null_ridge=0.1, label="rescope+null(hn)")
        r["partition"] = {"sizes": [len(x.rows) for x in (C0, S_, C1)], "disjoint": len(set(C0.rows) | set(S_.rows) | set(C1.rows)) == 480,
                          "first_part_is_the_plain_draw": C0.rows == fx.draws.calibration(L, 160, 5).rows, "cal_explicit_n": Mc.n, "variant": Mc.variant,
                          "recalibrated_on_C1": fx.methods.recalibrate(L, Mn.obj, cal=C1)[1]["draw"]["part"] == 2, "e2e_variant": Me.variant,
                          "e2e_record": {k: Me.rec["null_repair"][k] for k in ("n_a", "ridge", "detector_norm_inflation")}, "e2e_capped": Me.rec["cal"]["n_edits_capped"]}
        r["same_key_when_rebuilt"] = fx.methods.rescope(L, draw=3).key == fx.methods.rescope(L, draw=3).key
        r["keys_distinct"] = len(set(r["keys"].values())) == len(r["keys"])
        rs = fx.methods.e1_rowset(L, 0.05, 0, skip_unavailable=False)
        r["e1_rowset_no_model"] = {M.label: M.kind for M in rs}
        r["e1_labels_ok"] = [M.label for M in rs] == list(fx.methods.E1_LABELS)
        a, b = fx.methods.e1_rowset(L, 0.05, 0), fx.methods.e1_rowset(L, 0.10, 1)
        ka, kb = {M.label: M.key for M in a}, {M.label: M.key for M in b}
        r["e1_only_calibrated_rows_change"] = sorted(k for k in ka if ka[k] != kb[k]) == sorted(k for k in ka if k in fx.methods.E1_DEPENDS)
        ok &= (r["built"] == len(variants) and r["records_complete"] and r["same_key_when_rebuilt"] and r["keys_distinct"] and r["e1_labels_ok"]
               and r["e1_only_calibrated_rows_change"] and all(v for k, v in r["notes"].items() if isinstance(v, bool)) and ev5.rec["calibration"] == "log_only"
               and r["rescope_null"]["thresholds_bitwise_equal_independent_assembly"] and r["rescope_null"]["samples_disjoint"] and r["rescope_null"]["A_in_repair_pool"]
               and r["rescope_null"]["by_query"]["ok"]
               and r["rescope_null"]["max_abs_feature_on_A"] < 1e-9 and r["rescope_null"]["memo_path"] and r["rescope_null"]["values_unchanged"]
               and r["partition"]["disjoint"] and r["partition"]["first_part_is_the_plain_draw"] and r["partition"]["recalibrated_on_C1"] and r["partition"]["cal_explicit_n"] == 160
               and r["recalibrate"]["equals_rescope_bitwise"] and r["recalibrate"]["excluded_rows_absent"] and r["recalibrate"]["excluded_queries_absent"] and 3 <= r["recalibrate"]["n_excluded"] == r["recalibrate"]["n_pool_rows_with_an_excluded_query"]
               and r["wrap"]["memo"] == [True, True] and r["wrap"]["keys_differ"] and r["wrap"]["least_silences_complaints"] and r["wrap"]["null_repair_changes_features"]
               and ev40.rec["calibration"] == "two_block")
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- frontier tracing and matched-coverage read-out (E1b, E1c)
def frontier():
    out = {"logs": {}}
    ok = True
    for name in DEV:
        L = fx.logs.open(name)
        r = {}
        for feat, fixed, t in (("cos", fx.methods.cosine(L, "default"), 0.65), ("cos", fx.methods.exact(L), 0.999)):
            sw = fx.gate.sweep(L, feat)
            same = fx.gate.readout(L, sw, t) == fx.metrics.gate_stats(L, fx.gate.evaluate(L, fixed))      # a fixed-threshold gate IS a point of the sweep
            r[f"readout_{feat}_{t}_equals_fixed_gate"] = bool(same)
            ok &= bool(same)
        for feat in ("ncoord", "cos"):
            t0 = time.time()
            sw = fx.gate.sweep(L, feat)
            fr = fx.gate.frontier(L, feat, sw=sw)
            sec = time.time() - t0
            ts, cv = fr["thresholds"], fr["curves"]
            bad = 0
            for i in (0, 100, 250, 400, 500):                           # grid points against the full read-out at that threshold
                ro = fx.gate.readout(L, sw, float(ts[i]))
                pairs = [(cv["probe.heldout_disjoint.fired"][i], ro["probe"]["heldout_disjoint"]["fired"]), (cv["edit.tgen.own_top"][i], ro["edit"]["tgen"]["own_top"]),
                         (cv["edit.igen.own_cand"][i], ro["edit"]["igen"]["own_cand"]), (cv["hn.HN-I.fired"][i], ro["hn"]["HN-I"]["fired"]),
                         (cv["hn.HN-Q.own_top"][i], ro["hn"]["HN-Q"]["own_top"]), (cv["benign.fired"][i], ro["benign"]["fired"]),
                         (cv["probe.heldout_disjoint_near.fired"][i], ro["probe"]["heldout_disjoint_near"]["fired"])]
                bad += sum(abs(float(a) - float(b)) > 1e-9 for a, b in pairs)
            mono = all(bool((np.diff(v) <= 1e-12).all()) for k, v in cv.items() if k.endswith((".fired", ".own_cand")))
            mc = fx.gate.matched_coverage(L, feat, sw=sw)
            cov_ok = all((not m["reachable"]) or (m["coverage"] >= m["target"] and abs(0.5 * (m["readout"]["edit"]["tgen"]["own_top"] + m["readout"]["edit"]["igen"]["own_top"]) - m["coverage"]) < 1e-9)
                         for m in mc)
            # the threshold is the LARGEST with that coverage: one float above it the coverage is below the target
            tight = all((not m["reachable"]) or 0.5 * sum(fx.gate.readout(L, sw, float(np.nextafter(m["threshold"], np.inf)))["edit"][k]["own_top"] for k in ("tgen", "igen")) < m["target"]
                        for m in mc)
            r[feat] = {"n_grid": int(len(ts)), "n_curves": len(cv), "seconds": sec, "grid_points_differing_from_readout": bad, "curves_monotone": mono,
                       "matched_coverage": [{k: v for k, v in m.items() if k != "readout"} for m in mc], "matched_coverage_consistent": cov_ok, "threshold_is_largest": tight}
            ok &= bad == 0 and mono and cov_ok and tight
        out["logs"][name] = r
    out["ok"] = bool(ok)
    return out


# ----------------------------------------------------------------------------- gate level at scale (mechanics: time and memory only)
def scale():
    """Time and peak RAM of the gate-level pass on the longest log (T = 2000, internal): 20 calibration draws of rescope / cosine / ball
    over every cached sequence of the log, positions tf + prompt.  NO metric is computed, stored or printed."""
    import resource
    name = "TB-Q2000-ae-l2_3"
    if not fx.logs.exists(name):
        return {"available": False, "ok": True}
    t0 = time.time()
    L = fx.logs.open(name)
    sets = fx.gate.row_sets(L)
    out = {"available": True, "log": name, "T": L.T, "n_rows": {k: len(v) for k, v in sets.items()}, "seconds_open": time.time() - t0, "per_method": {}}
    for lab, fn in (("rescope", fx.methods.rescope), ("cosine", fx.methods.cosine), ("ball", fx.methods.ball)):
        t0 = time.time()
        fx.gate.evaluate(L, fn(L, draw=0), positions=("tf", "prompt"), rows=sets)
        first = time.time() - t0
        t0 = time.time()
        for d in range(1, 20):
            fx.gate.evaluate(L, fn(L, draw=d), positions=("tf", "prompt"), rows=sets)
        out["per_method"][lab] = {"first_draw_s": first, "next_19_draws_s": time.time() - t0}
        say(f"  scale {lab}: first draw {first:.0f}s, next 19 draws {out['per_method'][lab]['next_19_draws_s']:.0f}s")
    R = fx.gate.features(L, fx.methods.rescope(L), sets["probe"][:50])
    out.update(features_shape=list(R.shape), peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
               note="mechanics only (time and memory); no metric computed or printed", ok=list(R.shape) == [50, L.T])
    return out


TESTS = {"registry": registry, "draws": draws, "gate": gate, "regress_gate": regress_gate, "stats": stats, "io_memo": io_memo, "methods": methods, "frontier": frontier,
         "scale": scale}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(TESTS)
    for nm in names:
        t0 = time.time()
        res = TESTS[nm]()
        res.update(test=nm, seconds=time.time() - t0, code=fx.code_fingerprint(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        fx.io.jdump(res, os.path.join(OUT, f"cpu_{nm}.json"), indent=1)
        say(f"{nm}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds']:.0f}s)")
