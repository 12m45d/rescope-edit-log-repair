"""CPU unit tests, part 4: scale-safe registries / stored metrics (no model) and the key cache.

  a21_scale_registry   register_logs: the scale registry, the part-2 registry (absent / incomplete / complete-but-not-yet-
                       appended: simulated in a scratch directory) and the text registry; stored_metrics of every complete
                       log against the testbed files; the harness guards (T >= 500); the T = 2000 log's factors on the CPU;
                       the final hard-negative set (if it exists) through load_hard_negatives / Evaluator.hn_items.
  a22_keycache         KeyCache on a synthetic cache: decide / coords / rows / find against a naive per-key evaluation through
                       the gate's own firing(), for scoped gates (coordinate, cosine, ball; two-block thresholds) and the
                       router; the decision band; the silent-row flags of SparseEvaluator against a real hook; and, if they
                       exist, the per-backbone caches under the cache root (10_final/cache, or RESCOPE_KEYCACHE_ROOT: e.g. a
                       snapshot of a partial cache) are opened and their format is checked.
Each test returns a dict with 'pass' and the key numbers; run_tests.py writes them to tests/out/."""
from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
import types

import numpy as np
import torch

from rescope import baselines as BL, calibrate as CA, evaluate as EV, keycache as KC, loglib as LL
from rescope.scoped import Hook, ScopedLog
from rescope.sparse import SparseEvaluator
from t_core import linear

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")      # scratch directories live under the project root, never in /tmp


# ----------------------------------------------------------------------------- a21
def test_scale_registry():
    hz = EV.harness()
    TB = hz.TB
    out = {}
    rss0 = EV.peak_rss_gb()
    reg = copy.deepcopy(EV.register_logs())                             # a snapshot: the simulations below change the live record
    src = reg["sources"]
    by = {}
    for n, v in reg["logs"].items():
        by.setdefault(v["source"], []).append(n)
    out["sources"] = {k: {a: b for a, b in v.items() if a != "names"} for k, v in src.items()}
    out["logs_by_source"] = by
    complete = [n for n, v in reg["logs"].items() if v["complete"]]
    out["n_logs"], out["n_complete"] = len(reg["logs"]), len(complete)
    # ---- 1. every name of the three registries is in tb_common.LOGS; the registry files agree with what was registered
    with open(os.path.join(EV.SCALE, "registry.json")) as f:
        rj = json.load(f)
    with open(os.path.join(EV.SCALE, "registry_text.json")) as f:
        rt = json.load(f)
    out["registered"] = {"scale_names_in_LOGS": all(TB.LOGS.get(n) == c for n, c in rj.items()), "n_scale": len(rj),
                         "text_names_in_LOGS": all(TB.LOGS.get(n) == c for n, c in rt.items()), "n_text": len(rt)}
    plan_p = os.path.join(EV.SCALE, "out_part2", "plan.json")
    if os.path.exists(plan_p):
        with open(plan_p) as f:
            pe = json.load(f)["registry_entries"]
        out["part2_now"] = {"plan_exists": True, "n_planned": len(pe), "registered": [n for n in pe if n in TB.LOGS],
                            "not_registered_incomplete": [n for n in pe if n not in TB.LOGS],
                            "every_complete_planned_log_is_registered": all(n in TB.LOGS for n in pe if EV._complete(TB, n)),
                            "no_incomplete_planned_log_is_registered": all(EV._complete(TB, n) or n in rj for n in pe if n in TB.LOGS)}
    else:
        out["part2_now"] = {"plan_exists": False, "every_complete_planned_log_is_registered": True, "no_incomplete_planned_log_is_registered": True}
    # ---- 2. the part-2 registry: absent, then present with one complete and one incomplete log (scratch copy of the layout)
    saved_scale, saved_logs, saved_reg = EV.SCALE, dict(TB.LOGS), {"logs": dict(EV._REG["logs"]), "sources": dict(EV._REG["sources"])}
    tmp = tempfile.mkdtemp(prefix="scratch_a21_", dir=SCRATCH)
    sim = {}
    try:
        probe = "TB-Q100-ae-l2_1"                                        # a complete log directory, used under its own name
        EV.SCALE = tmp
        TB.LOGS.pop(probe)
        EV._register(TB)                                                 # nothing exists under tmp: every source must be tolerated
        s = EV._REG["sources"]
        sim["absent"] = {"no_exception": True, "scale_ok": s["scale"]["ok"], "part2_ok": s["part2"]["ok"], "text_ok": s["text"]["ok"],
                         "probe_registered": probe in TB.LOGS, "part2_error": s["part2"].get("error")}
        try:
            EV.ensure_registered(probe)
            sim["absent"]["unknown_log_raises"] = False
        except KeyError:
            sim["absent"]["unknown_log_raises"] = True
        os.makedirs(os.path.join(tmp, "out_part2"))
        with open(os.path.join(tmp, "out_part2", "plan.json"), "w") as f:
            json.dump({"registry_entries": {probe: saved_logs[probe], "TB-NOT-BUILT-YET": dict(saved_logs[probe], stream="/nonexistent")}}, f)
        new = EV._register(TB)
        s = EV._REG["sources"]["part2"]
        sim["present"] = {"part2_ok": s["ok"], "pending_registered": s["pending"], "incomplete": s["incomplete"], "newly_registered": new,
                          "probe_registered_with_its_config": TB.LOGS.get(probe) == saved_logs[probe],
                          "source": EV._REG["logs"].get(probe, {}).get("source")}
        EV.ensure_registered(probe)
        with open(os.path.join(tmp, "registry.json"), "w") as f:         # the builder appends the name: the file is read again
            json.dump({"TB-APPENDED-LATER": dict(saved_logs[probe])}, f)
        new = EV._register(TB)
        sim["appended"] = {"picked_up": "TB-APPENDED-LATER" in TB.LOGS, "appended": EV._REG["sources"]["part2"]["appended"]}
    finally:
        EV.SCALE = saved_scale
        TB.LOGS.clear()
        TB.LOGS.update(saved_logs)
        EV._REG["logs"], EV._REG["sources"] = saved_reg["logs"], saved_reg["sources"]
        shutil.rmtree(tmp, ignore_errors=True)
    out["simulated_part2"] = sim
    out["registry_restored"] = dict(TB.LOGS) == saved_logs
    # ---- 3. stored metrics of every complete log against the testbed files
    st = {}
    ok_st = True
    for n in complete:
        d = os.path.join(EV.EXP, "06_testbed_v3", n)
        try:
            m = EV.stored_metrics(n)
        except Exception as e:                                           # reported per log (e.g. the legacy B = 1 log)
            st[n] = {"error": f"{type(e).__name__}: {e}"}
            ok_st = ok_st and n in ("[redacted: describes a related submission under double-blind review]",)
            continue
        with open(os.path.join(d, "metrics.json")) as f:
            met = json.load(f)
        with open(os.path.join(d, "per_edit_metrics.jsonl")) as f:
            pe_ = [json.loads(x) for x in f if x.strip()]
        with open(os.path.join(d, "probe_records_v2.jsonl")) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        v = [r for r in rows if r["y0_valid"]]
        ok = (m["T"] == met["n_edits"] == len(pe_) == len(m["per_edit"]) and m["edits"]["rel"]["exact"] == met["Rel_exact"]
              and abs(met["Rel_exact"] - 100.0 * np.mean([bool(r["post_rel_exact"]) for r in pe_])) < 1e-9
              and m["locality"]["T-Loc"] == met["TLoc_probes"] and m["locality"]["I-Loc"] == met["ILoc_probes"]
              and m["probes"]["all"]["n_valid"] == len(v) and m["probes"]["all"]["n_D1"] == sum(bool(r["is_complaint_D1"]) for r in v)
              and m["probes"]["all"]["n_D2"] == sum(bool(r["is_complaint_D2"]) for r in rows)
              and m["probes"]["repair"]["n_rows"] + m["probes"]["heldout"]["n_rows"] == len(rows)
              and m["probes"]["heldout_disjoint"]["n_rows"] <= m["probes"]["heldout"]["n_rows"])
        st[n] = {"T": m["T"], "Rel_exact": m["edits"]["rel"]["exact"], "TGen_exact": m["edits"]["tgen"]["exact"], "IGen_exact": m["edits"]["igen"]["exact"],
                 "T-Loc": m["locality"]["T-Loc"], "I-Loc": m["locality"]["I-Loc"], "n_valid": m["probes"]["all"]["n_valid"],
                 "D1_rate_heldout_disjoint": m["probes"]["heldout_disjoint"]["D1_rate"], "n_D2": m["probes"]["all"]["n_D2"], "ok": bool(ok)}
        ok_st = ok_st and ok
    out["stored_metrics"] = st
    # ---- 4. the guards: refused from T = 500 on, passed through below; lifted by the switch
    g = {}
    for nm, fn in (("baseline", hz.HE.baseline), ("pkg.baseline", hz.pkg.baseline), ("evalcache", hz.HE.evalcache), ("edit_keys", hz.HL.HLog.edit_keys)):
        res = {}
        for T in (100, 499, 500, 2000):
            try:
                fn(types.SimpleNamespace(T=T))                           # a stub: a call that passes the guard fails on the stub itself
                res[T] = "ran"
            except RuntimeError as e:
                res[T] = "refused" if "is refused for a log with T" in str(e) else "passed"
            except Exception:
                res[T] = "passed"
        EV.ALLOW_HARNESS_AT_SCALE = True
        try:
            fn(types.SimpleNamespace(T=2000))
            res["lifted"] = "ran"
        except RuntimeError as e:
            res["lifted"] = "refused" if "is refused for a log with T" in str(e) else "passed"
        except Exception:
            res["lifted"] = "passed"
        finally:
            EV.ALLOW_HARNESS_AT_SCALE = False
        g[nm] = res
    out["guards"] = g
    ok_g = all(r[100] == r[499] == "passed" and r[500] == r[2000] == "refused" and r["lifted"] == "passed" for r in g.values())
    # ---- 5. the longest log on the CPU: factors, rebased values, a calibrated gate; lean-load default by T
    name = "TB-Q2000-ae-l2_3"
    log = LL.from_testbed(name)
    S = BL.feature_gate(log, "ncoord")
    rng = torch.Generator().manual_seed(21)
    q = torch.randn(80, log.d, generator=rng, dtype=torch.float64)
    S2, rec = CA.calibrate_two_block(S, q, 0.05, seed=0)
    out["longest_log_cpu"] = {"name": name, "T": log.T, "n_batches": int(log.batch.max()) + 1, "values_finite": bool(torch.isfinite(S.values).all()),
                              "own_coordinates_positive": bool((log.own() > 0).all()), "n_edits_capped_on_random_keys": rec["n_edits_capped"],
                              "guards_fire": bool(S2.guard_ok().all()), "n_edits_from_metrics": EV.n_edits(name),
                              "lean_by_default": {n: EV.n_edits(n) >= EV.SCALE_T for n in ("TB-Q100-ae-l2_3", "TB-Q200-ae-l2_3", "TB-Q500-ae-l2_3", name)}}
    # ---- 6. the final hard-negative set (10_final/cache/hard_negatives/<backbone>/, if it exists): loader and item fields
    fh = {}
    for bb, short, long_ in (("qwen25vl", "TB-Q100-ae-l2_3", "TB-Q2000-ae-l2_3"), ("llava", "TB-L100-ae-l2_3", "TB-L1000-ae-l2_3")):
        if not os.path.exists(os.path.join(EV.HN_DIR_FINAL, bb, "rows.jsonl")):
            fh[bb] = {"available": False}
            continue
        r = {"available": True}
        for lg in (short, long_):
            items, _ = TB.load_stream(TB.LOGS[lg]["stream"])
            stub = types.SimpleNamespace(backbone=bb, items=items, name=lg, T=len(items))
            old, new = EV.load_hard_negatives(stub), EV.load_hard_negatives(stub, EV.HN_DIR_FINAL)
            ev = EV.Evaluator.__new__(EV.Evaluator)                       # hn_items needs the rows and T only
            ev.H, ev.hn, ev._hn_error = stub, new, None
            its = ev.hn_items()
            ty = {}
            for h in its:
                ty[h["hn_type"]] = ty.get(h["hn_type"], 0) + 1
            r[lg] = {"T": len(items), "n_p11_rows": len(old), "n_final_rows": len(new), "types": ty,
                       "p11_rows_contained": {x["hn_id"] for x in old} <= {x["hn_id"] for x in new},
                       "edits_in_log": all(0 <= h["edit"] < len(items) for h in its),
                       "cross_rows_have_both_edits": all(0 <= h["edit_b"] < len(items) for h in its if "edit_b" in h), "n_cross": sum("edit_b" in h for h in its),
                       "fields": all(all(k in h for k in ("rule", "group", "new", "src_related_to_log", "q_edit")) for h in its),
                       "n_src_related_to_log": sum(h["src_related_to_log"] for h in its), "n_q_edit": sum(h["q_edit"] is not None for h in its),
                       "all_valid": all(h["y0_valid"] is True and h["_row"]["y0_valid"] for h in its)}
            r[lg]["ok"] = bool(r[lg]["p11_rows_contained"] and r[lg]["edits_in_log"] and r[lg]["cross_rows_have_both_edits"] and r[lg]["fields"]
                                 and r[lg]["all_valid"] and r[lg]["n_final_rows"] > r[lg]["n_p11_rows"] > 0 and r[lg]["n_cross"] > 0)
        fh[bb] = r
    out["final_hard_negatives"] = fh
    ok_fh = all(v[k]["ok"] for v in fh.values() if v["available"] for k in v if k != "available")
    out["peak_rss_gb"] = {"before": rss0, "after": EV.peak_rss_gb()}
    lo = out["longest_log_cpu"]
    out["pass"] = bool(src["scale"]["ok"] and src["text"]["ok"] and out["registered"]["scale_names_in_LOGS"] and out["registered"]["text_names_in_LOGS"]
                       and out["part2_now"]["every_complete_planned_log_is_registered"] and out["part2_now"]["no_incomplete_planned_log_is_registered"]
                       and sim["absent"]["no_exception"] and not sim["absent"]["scale_ok"] and not sim["absent"]["part2_ok"] and not sim["absent"]["text_ok"]
                       and not sim["absent"]["probe_registered"] and sim["absent"]["unknown_log_raises"]
                       and sim["present"]["part2_ok"] and sim["present"]["pending_registered"] == ["TB-Q100-ae-l2_1"]
                       and sim["present"]["incomplete"] == ["TB-NOT-BUILT-YET"] and sim["present"]["probe_registered_with_its_config"]
                       and sim["present"]["source"] == "part2-pending" and sim["appended"]["picked_up"] and out["registry_restored"]
                       and ok_st and ok_g and ok_fh and lo["T"] == 2000 == lo["n_edits_from_metrics"] and lo["values_finite"] and lo["guards_fire"]
                       and lo["lean_by_default"] == {"TB-Q100-ae-l2_3": False, "TB-Q200-ae-l2_3": False, "TB-Q500-ae-l2_3": True, name: True}
                       and out["peak_rss_gb"]["after"] < 40.0)
    return out


# ----------------------------------------------------------------------------- a22
def _synthetic_cache(directory, log, g, n_seq=90):
    """A key cache in the documented format around a synthetic log: edits (rel / tgen / igen), probes, hard negatives, benign."""
    T, d = log.T, log.d
    uids = [f"u{i}" for i in range(T)]
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)                             # noqa: E731
    index, tf, dec, steps = [], [], [], []
    for j in range(n_seq):
        kind = ("rel", "tgen", "igen", "probe", "probe", "hn_i", "hn_q", "hn_x", "benign")[j % 9]
        e = j % T
        base = log.K[e] * (0.75 + 0.5 * float(torch.rand(1, generator=g))) + 0.35 * rnd(d) if kind in ("rel", "tgen", "igen", "hn_i", "hn_x") else rnd(d) + 0.3 * log.K[e]
        row = {"sid": f"s{j}", "kind": kind, "family": "S", "fmt": "vqa", "prompt": f"q{j}", "image": f"i{j}.jpg" if kind != "benign" else None,
               "query_id": j, "lp_decode": 5, "row_decode": j}
        dec.append(base)
        nd = 1 + j % 3                                                   # 1 - 3 distinct keys in the 11 forwards of a decode
        row["rows_decode_steps"] = [len(steps) + (k * nd) // 11 for k in range(11)]
        steps += [base] + [base + 0.02 * rnd(d) for _ in range(nd - 1)]
        if kind in ("rel", "tgen", "igen"):
            row.update(edit=e, edit_uid=uids[e], metric=kind, answer=f"target{e}")
        elif kind == "probe":
            row.update(probe_idx=j, uid=f"p{j}", ptype="image", pool="repair" if j % 2 else "heldout", disjoint=bool(j % 2 == 0 and j % 4 == 0),
                       strict=False, y0_text=f"y{j}", y0_valid=bool(j % 5), answer=f"y{j}")
        elif kind.startswith("hn"):
            row.update(edit=e, edit_uid=uids[e], y0_text=f"y{j}", y0_valid=True, answer=f"y{j}")
            if kind == "hn_x":
                row.update(edit_b=(e + 3) % T, edit_b_uid=uids[(e + 3) % T])
        else:
            row.update(pool="benign", answer=None)
        if row.get("answer"):
            sh = 1 if j % 7 == 3 else 0
            row.update(lp_tf=5 + sh, lp_shift=sh, row_tf=len(tf), row_tf_prompt=None, pre_exact=bool(j % 2), pre_token_acc=0.5 + 0.5 * (j % 2), pre_n_tok=2,
                       pre_finite=True, pre_first_mismatch=None if j % 2 else 0)
            tf.append(base + 0.01 * rnd(d))
        else:
            row.update(lp_tf=None, lp_shift=None, row_tf=None, row_tf_prompt=None)
        index.append(row)
    n_main = len(tf)
    for row in index:                                                    # the extra rows: the teacher-forced sequence at the decode's token
        if row.get("lp_shift"):
            row["row_tf_prompt"] = len(tf)
            tf.append(torch.stack(dec)[row["row_decode"]] + 0.5 * rnd(d))
    KC.write_cache(directory, index, torch.stack(tf).float(), torch.stack(dec).float(), torch.stack(steps).float(),
                   meta={"backbone": "synthetic", "complete": True, "n_main_rows": n_main})
    return uids


def _naive_decide(gate, kc, j, which, own):
    """The decision of ONE cached sequence, key by key, through the gate's own firing() (the hook's code path)."""
    e = kc.index[j]
    rows = {"tf": [("tf", e.get("row_tf"))], "prompt": [("tf", e.get("row_tf_prompt") if e.get("lp_shift") else e.get("row_tf"))],
            "decode": [("decode", e["row_decode"])], "steps": [("steps", r) for r in e["rows_decode_steps"]]}
    rows["both"] = rows["tf"] + rows["decode"]
    rows["all"] = rows["tf"] + ([("tf", e["row_tf_prompt"])] if e.get("lp_shift") else []) + rows["steps"]
    fired, top, ncand, oc, ot, qmax = False, -1, 0, False, False, None
    for name, r in rows[which]:
        k = kc.K(name)[int(r)]
        f = gate.firing(k[None, None], torch.tensor([0]))
        sc = gate.scores(k[None])[0]
        qmax = sc if qmax is None else torch.maximum(qmax, sc)
        if f is None:
            continue
        ce = gate._edit_any(f.cand)[0]
        if not fired:
            top = int(f.edit[0])
            ot = own >= 0 and top == own
        fired = True
        ncand = max(ncand, int(f.cand.sum()))
        oc = oc or (own >= 0 and bool(ce[own]))
    return fired, top, ncand, oc, ot, qmax


def test_keycache():
    g = torch.Generator().manual_seed(22)
    log, tr = LL.synthetic_log(T=12, d=48, m=10, batch=4, seed=3, l2=4.0)
    tmp = tempfile.mkdtemp(prefix="scratch_a22_", dir=SCRATCH)
    out = {"gates": {}, "n_decisions": 0, "n_equal": 0}
    try:
        uids = _synthetic_cache(tmp, log, g)
        log.meta["uids"] = uids
        kc = KC.KeyCache(tmp)
        out["summary"] = kc.summary()
        ben = kc.rows(kind="probe", pool="repair")
        gates = {}
        for feat in BL.FEATURES:                                         # two-block thresholds from the cache's own benign rows
            S0 = BL.feature_gate(log, feat)
            gates[feat], _ = CA.calibrate_two_block(S0, coords=kc.coords(S0, ben, "tf"), alpha=0.2, seed=1, allow_silence=(feat == "negdist"))
        gates["ncoord_rank_r"], _ = CA.calibrate_two_block(BL.feature_gate(log, "ncoord"), coords=kc.coords(gates["ncoord"], ben, "tf"), alpha=0.2, seed=1, rank="r")
        gates["tau0.3_fire_all"] = ScopedLog.from_log(log, tau=0.3, fire="all")
        for name, gate in gates.items():
            r = {}
            for which in KC.WHICH:
                dec = kc.decide(gate, which, uids=uids)
                n = eq = 0
                near_n, near_diff, cmax = out.get("n_in_band", 0), out.get("n_in_band_decided_differently_key_by_key", 0), out.get("coords_max_rel_diff", 0.0)
                C = kc.coords(gate, dec["rows"], which)
                for i, j in enumerate(dec["rows"]):
                    own = int(dec["own"][i])
                    fired, top, ncand, oc, ot, qmax = _naive_decide(gate, kc, j, which, own)
                    same = (bool(dec["fired"][i]) == fired and int(dec["n_cand"][i]) == ncand and bool(dec["own_cand"][i]) == oc
                            and (gate.fire != "top1" or (int(dec["top_edit"][i]) == top and bool(dec["own_top"][i]) == ot))
                            and (dec["margin"][i] >= 0) == bool(dec["fired"][i]))
                    # a sequence inside the decision band (e.g. the calibration query that defines eta: its coordinate is ON the
                    # boundary) may be decided differently key by key than in a batch: that is what the flag is for
                    near_n += int(dec["near"][i])
                    near_diff += int(dec["near"][i] and not same)
                    n += 1
                    eq += int(same or bool(dec["near"][i]))
                    cmax = max(cmax, float((C[i] - qmax).abs().max() / qmax.abs().max().clamp_min(1e-300)))
                r[which] = {"n": n, "equal": eq, "fired": int(dec["fired"].sum()), "near": int(dec["near"].sum())}
                out["n_decisions"] += n
                out["n_equal"] += eq
                out["n_in_band"], out["n_in_band_decided_differently_key_by_key"], out["coords_max_rel_diff"] = near_n, near_diff, cmax
            out["gates"][name] = r
        # router (dense gate): activation at the last prompt position
        WE = (tr["WE"]).float()
        W0 = tr["W0"].float()
        router = BL.router_gate(log, WE, W0, q=0.25)
        dr = kc.decide(router, "tf")
        act = router.activation(kc.keys("tf", dr["rows"]))
        out["router"] = {"n": len(dr["rows"]), "equal": int((torch.from_numpy(dr["fired"]) == (act >= router.thr)).sum()), "fired": int(dr["fired"].sum())}
        # selection helpers
        e0 = kc.index[0]
        sel = {"log_rows_T6_edits": sorted({kc.index[j]["edit"] for j in kc.log_rows(uids[:6], kind=("rel", "tgen", "igen"))}) == list(range(6)),
               "log_rows_T6_cross_rows_need_both_edits": all(kc.index[j]["edit"] < 6 and kc.index[j]["edit_b"] < 6 for j in kc.log_rows(uids[:6], kind="hn_x")),
               "family_of": kc.family_of(uids) == "S" and kc.family_of(uids[:5]) == "S" and kc.family_of(["x"] + uids[1:]) is None,
               "find_by_content": kc.find(e0["fmt"], e0["prompt"], e0["image"], e0["answer"]) == 0 and kc.find(e0["fmt"], e0["prompt"], e0["image"], "other") is None
               and kc.find(e0["fmt"], e0["prompt"], e0["image"]) == 0,
               "pool_filters": all(kc.index[j]["pool"] == "repair" for j in ben) and all(kc.index[j]["disjoint"] for j in kc.rows(pool="heldout_disjoint")),
               "valid_filter": all(kc.index[j]["y0_valid"] for j in kc.rows(kind="probe", valid=True)),
               "keys_shapes": tuple(kc.keys("steps", [0, 1]).shape) == (2, 11, log.d) and tuple(kc.keys("tf", kc.having("tf")[:3]).shape) == (3, log.d),
               "prompt_key_is_extra_row_where_shifted": all((kc.key_rows("prompt", [j])[0] == kc.index[j]["row_tf_prompt"]) == bool(kc.index[j]["lp_shift"])
                                                           for j in kc.having("tf")),
               "decide_log_shorthand": kc.decide(gates["ncoord"], "tf", log=log)["rows"] == [j for j in kc.log_rows(uids) if kc.index[j].get("row_tf") is not None]}
        out["selection"] = sel
        # refused: roles that need every position, cheap precisions, wrong dtypes
        ref = {}
        for nm, fn in (("role_all", lambda: kc.decide(gates["ncoord"].replace(role="all"))), ("precision_fp32", lambda: kc.decide(gates["ncoord"].replace(precision="fp32"))),
                       ("zscore", lambda: kc.decide(BL.zscore_gate(WE, W0)))):
            try:
                fn()
                ref[nm] = False
            except NotImplementedError:
                ref[nm] = True
        try:
            KC.write_cache(os.path.join(tmp, "bad"), [], torch.zeros(2, 4, dtype=torch.float64))
            ref["float64_keys"] = False
        except ValueError:
            ref["float64_keys"] = True
        try:
            KC.KeyCache(os.path.join(tmp, "nowhere"))
            ref["missing_cache"] = False
        except FileNotFoundError:
            ref["missing_cache"] = True
        # a cache whose files do not belong together must raise at construction (a directory that is being rewritten)
        import hashlib
        cons = {}
        ok0 = KC.KeyCache(tmp, verify=True)
        cons["consistent_cache_opens_with_verify"] = ok0.n == kc.n and ok0.layer is None
        mixed = os.path.join(tmp, "mixed")
        shutil.copytree(tmp, mixed, ignore=shutil.ignore_patterns("bad", "mixed"))
        t_ok = torch.load(os.path.join(mixed, "keys_tf.pt"))
        with open(os.path.join(mixed, "meta.json")) as f:
            meta_ok = json.load(f)

        def raises(fn):
            try:
                fn()
                return False
            except RuntimeError as e:
                return "inconsistent key cache" in str(e)
        torch.save(t_ok[:-3].clone(), os.path.join(mixed, "keys_tf.pt"))            # 1. a shorter key file (older build) under the same index
        cons["shorter_key_file_raises"] = raises(lambda: KC.KeyCache(mixed))
        os.remove(os.path.join(mixed, "meta.json"))                                 #    ... also without meta.json (the index addresses missing rows)
        cons["shorter_key_file_without_meta_raises"] = raises(lambda: KC.KeyCache(mixed))
        torch.save(t_ok, os.path.join(mixed, "keys_tf.pt"))
        with open(os.path.join(mixed, "meta.json"), "w") as f:
            json.dump(meta_ok, f)
        with open(os.path.join(mixed, "index.jsonl"), "a") as f:                    # 2. an index that meta.json does not describe
            f.write(json.dumps({"sid": "late", "kind": "benign", "row_decode": 0}) + "\n")
        cons["other_index_raises"] = raises(lambda: KC.KeyCache(mixed))
        shutil.copy2(os.path.join(tmp, "index.jsonl"), os.path.join(mixed, "index.jsonl"))
        t2 = t_ok.clone()
        t2[0, 0] += 1.0
        torch.save(t2, os.path.join(mixed, "keys_tf.pt"))                           # 3. same shape, other content: found by verify=True
        with open(os.path.join(mixed, "keys_tf.pt"), "rb") as f:
            cons["md5_really_differs"] = hashlib.md5(f.read()).hexdigest() != hashlib.md5(open(os.path.join(tmp, "keys_tf.pt"), "rb").read()).hexdigest()
        meta_md5 = dict(meta_ok, keys_tf=dict(meta_ok["keys_tf"], md5=hashlib.md5(open(os.path.join(tmp, "keys_tf.pt"), "rb").read()).hexdigest()))
        with open(os.path.join(mixed, "meta.json"), "w") as f:
            json.dump(meta_md5, f)
        cons["other_content_opens_without_verify"] = KC.KeyCache(mixed).n == kc.n
        cons["other_content_raises_with_verify"] = raises(lambda: KC.KeyCache(mixed, verify=True))
        out["consistency"] = cons
        out["refused"] = ref
        # decision band: a threshold placed exactly ON a cached key's coordinate, and one ulp-scale above / far above it
        S = gates["ncoord"]
        k = kc.keys("tf", [kc.having("tf")[0]])
        r = S.scores(k)[0]
        j0 = int(r.argmax())
        band = {}
        for nm, th in (("at_coordinate", float(r[j0])), ("1e-12_above", float(r[j0]) * (1 + 1e-12) + 1e-13), ("1e-6_above", float(r[j0]) * (1 + 1e-6) + 1e-6)):
            tau = torch.full_like(S.det_tau, float("inf"))
            tau[j0] = th
            Sx = S.replace(det_tau=tau)
            band[nm] = {"fired": bool((S.scores(k) >= Sx.theta()[None, :]).any()), "near": bool(KC.near_threshold(Sx, k, S.scores(k))[0])}
        out["decision_band"] = band
        # the flags of a silent row == the summary of a real hook that was called once and did not fire
        lin = linear(tr["W0"], torch.float32)
        fl = {"n": 0, "equal": 0}
        for fire in ("top1", "all"):
            for own in (None, 3):
                Sx = ScopedLog.from_log(log, tau=float("inf"), fire=fire, W0=lin.weight.data.clone())
                with Hook(lin, Sx) as hk:
                    hk.begin(torch.tensor([2]))
                    lin(torch.randn(1, 6, log.d, generator=g, dtype=torch.float64).float())
                    real = hk.summary(own=own)
                fl["n"] += 1
                fl["equal"] += int(real == SparseEvaluator._silent_flags(Sx, own))
        out["silent_flags"] = fl
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # the per-backbone caches of the final experiments (if they exist now): opened, counted, format checked
    real = {}
    for bb in ("qwen25vl", "llava"):
        if not KC.KeyCache.available(bb):
            real[bb] = {"available": False}
            continue
        kc = KC.KeyCache(bb)
        need = ("sid", "kind", "family", "fmt", "prompt", "answer", "lp_tf", "lp_decode", "lp_shift", "row_tf", "row_decode", "query_id")
        tfk = kc.K("tf")
        rows_tf = [e["row_tf"] for e in kc.index if e.get("row_tf") is not None] + [e["row_tf_prompt"] for e in kc.index if e.get("row_tf_prompt") is not None]
        real[bb] = {"available": True, "is_final": kc.is_final, "summary": kc.summary(), "fields_present": all(all(k in e for k in need) for e in kc.index),
                    "kinds_known": all(e["kind"] in KC.KINDS for e in kc.index), "keys_float32": tfk.dtype == torch.float32 and kc.K("decode").dtype == torch.float32,
                    "row_tf_is_a_bijection": sorted(rows_tf) == list(range(tfk.shape[0])), "n_decode_rows_equal_index": kc.K("decode").shape[0] == kc.n,
                    "n_with_pre_edit_outcome": sum(e.get("pre_exact") is not None for e in kc.index), "n_shifted": sum(bool(e.get("lp_shift")) for e in kc.index)}
        real[bb]["ok"] = bool(real[bb]["fields_present"] and real[bb]["kinds_known"] and real[bb]["keys_float32"] and real[bb]["row_tf_is_a_bijection"]
                              and real[bb]["n_decode_rows_equal_index"])
    out["final_caches"] = real
    out["cache_root"], out["cache_root_is_final_location"] = KC.CACHE_ROOT, KC.CACHE_ROOT == KC.DEFAULT_ROOT
    b = out["decision_band"]
    out["pass"] = bool(out["n_equal"] == out["n_decisions"] > 1000 and all(v["fired"] > 0 for r in out["gates"].values() for v in r.values())
                       and out["router"]["equal"] == out["router"]["n"] > 0 and out["coords_max_rel_diff"] < 1e-12
                       and out["n_in_band"] < 0.02 * out["n_decisions"]
                       and all(out["selection"].values()) and all(out["refused"].values()) and all(out["consistency"].values())
                       and b["at_coordinate"] == {"fired": True, "near": True} and b["1e-12_above"] == {"fired": False, "near": True}
                       and b["1e-6_above"] == {"fired": False, "near": False}
                       and out["silent_flags"]["equal"] == out["silent_flags"]["n"] == 4
                       and all(v.get("ok", True) for v in real.values()))
    return out


TESTS = {"a21_scale_registry": test_scale_registry, "a22_keycache": test_keycache}
