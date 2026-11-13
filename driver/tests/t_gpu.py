"""GPU tests of the fx driver.  One process per log, through the shared queue (resumable: finished tests are skipped, the memo persists):

  GQ=$RESCOPE_ROOT/02_research/tools/gpuq/gq
  nohup $GQ run --mem 10500 --name fx_t_Q -- python t_gpu.py TB-Q100-ae-l2_3 a b c d > out/gpu_TB-Q100-ae-l2_3.log 2>&1 &
  nohup $GQ run --mem 17000 --name fx_t_L -- python t_gpu.py TB-L100-ae-l2_3 a b c d > out/gpu_TB-L100-ae-l2_3.log 2>&1 &
  nohup $GQ run --mem 10500 --name fx_t_e1 -- python t_gpu.py TB-Q2000-ae-l2_3 e > out/gpu_TB-Q2000-ae-l2_3.log 2>&1 &     (mechanics only)

  a  regress   (development logs) the constructors reproduce the sanity agent's M1 - M6, deployed and pre-edit rows for calibration draws
               0 - 2: every teacher-forced row of e0_sanity/out/behaviour/<log>/ against fx.behave, field by field; differences are
               re-run with a gate built the sanity agent's way (on the GPU) to establish their cause.
  b  memo      300 random (sequence, firing edit) pairs: the memoised outcome == a fresh forward through the library's Evaluator with the
               real gate; 40 decodes likewise; the deployed weight on 60 sequences; DUAL composed from memoised decodes == the library's
               DUAL; a second pass of a whole evaluation runs the model on nothing and returns the same rows.
  c  hook      one behavioural pass WITHOUT memo of ReScope and of the cosine gate (two-block, draw 0) over every sequence (edits, valid
               probes of all pools, valid hard negatives; + lp='prompt' where the position differs): the firing observed in the hook ==
               fx.gate on every sequence, and every row == the memo path's row.
  d  e1        e1_rowset end to end on one draw (all methods; hard negatives: every 4th row; DUAL and the other decode metrics on a
               subsample): seconds per method, rows through fx.io (results.json, rows, paper_numbers.json, table.txt).
  f  where     (development logs) Rescope-WHERE + least repair of 10 / 50 reported complaints of the P12 stream: builds, repaired complaints are
               silent afterwards, order independence; where_logonly cache; the evidence-curve constructor at n = 0, 10, 39, 40.
  g  ablations (development logs) every ablation switch of E6 (roles, responses, fire, rebase, feature, block ratio, baseline quantile, calibration
               source / keys / contamination, dense-gate calibrations) builds on the GPU and gives the plain Evaluator's rows on a sample.
     --smoke   tests c, d, f on every 8th sequence (a quick run of every code path; written to *_smoke.json, never rendered)
  e  mechanics (any log; the only test for logs that are not development logs) open, build 'rescope' for one draw, gate-level pass, 50
               behavioural sequences (own Rel prompts); peak RAM and time; NO metric is computed, stored or printed.
"""
from __future__ import annotations

import glob
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fx  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from t_cpu import E0_METHODS, e0_method  # noqa: E402

OUT = os.path.join(HERE, "out")
DEV = fx.logs.names("development")
SMOKE = "--smoke" in sys.argv          # every 8th sequence in tests c, d, f (a quick run of every code path; never rendered)
STEP = 8 if SMOKE else 1
CMP_OUT = ("exact", "token_acc", "n_tok", "finite", "first_mismatch", "D1", "tok_loc")
CMP_FLAGS = ("fired", "top_last", "n_cand_edits", "own_fired_last", "own_released_last")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def gpu_gb():
    return torch.cuda.max_memory_allocated() / 2 ** 30


def diff_fields(mine, theirs, fields):
    return [k for k in fields if k in theirs and mine.get(k) != theirs[k]]


# ----------------------------------------------------------------------------- a: regression against the sanity agent (behaviour)
def e0_gpu_gate(ses, mid, seed):
    """The gate exactly as e0_behaviour.py built it: on the model's device, calibration queries scored one by one."""
    from rescope import add_guards, calibrate_shared, calibrate_two_block, feature_gate, from_harness, splits
    H, ev = ses.H, ses.ev
    name, kw = E0_METHODS[mid]
    log = from_harness(H).to(H.dev)
    S = feature_gate(log, "ncoord" if name.startswith("rescope") else "cos", role="last", fire="top1", response="repl", values="rebased", W0=H.W0)
    S, _ = add_guards(S, *ev.own_keys("rel"))
    benign = splits.sample(ev.probe_items("repair"), 200, seed)
    q = [ev.eligible_keys(it, S) for it in benign]
    if mid in ("M1", "M3"):
        return calibrate_two_block(S, q, 0.05, seed=seed)[0]
    return calibrate_shared(S, q, 0.05)[0]


def a_regress(L, H, ses_global):
    """Every row is compared twice: (1) through a PRIVATE EMPTY memo store, so that each firing row comes from a fresh forward of this
    code; (2) through the shared memo store 10_final/memo (what the experiments will read: entries written by earlier runs)."""
    import shutil
    d = os.path.join(fx.E0, "out", "behaviour", L.name)
    if not os.path.exists(os.path.join(d, "done.json")):
        return {"available": False, "ok": False, "note": "the sanity agent's behavioural run of this log is not complete"}
    cold = os.path.join(OUT, f"memo_cold_{L.name}_a")
    shutil.rmtree(cold, ignore_errors=True)
    old_root, old_dsig = fx.MEMO_ROOT, fx.behave._DSIG.pop(L.name, None)
    fx.MEMO_ROOT = cold
    try:
        res = _a_regress(L, H, fx.behave.Session(L, H), d)
    finally:
        fx.MEMO_ROOT = old_root
        fx.behave._DSIG.pop(L.name, None)
        if old_dsig is not None:
            fx.behave._DSIG[L.name] = old_dsig
        shutil.rmtree(cold, ignore_errors=True)
    f0 = ses_global.stats["forwards"]
    shared = _a_regress(L, H, ses_global, d)
    res["shared_memo"] = {"n_rows": shared["n_rows"], "n_rows_differing": shared["n_rows_differing"], "n_unexplained": shared["n_unexplained"],
                          "numbers_differing": shared["numbers"]["n_differing"], "forwards": ses_global.stats["forwards"] - f0, "ok": shared["ok"]}
    res["ok"] = bool(res["ok"] and shared["ok"])
    return res


def _a_regress(L, H, ses, d):
    B = {}
    for p in sorted(glob.glob(os.path.join(d, "*__*__*.json"))):
        part = fx.io.jload(p)
        g = B.setdefault(part["set"], {}).setdefault(part["group"], {})
        for m, rows in part["rows"].items():
            g.setdefault(m, []).extend(rows)
    res = {"available": True, "sets": sorted(B), "n_rows": 0, "n_fields": 0, "n_rows_differing": 0, "by_method": {}, "differing": [], "headline": {},
           "numbers": {"source": os.path.join(fx.E0, "out", "results.json"), "n_compared": 0, "n_differing": 0, "max_abs_diff": 0.0, "differing": []}}
    e0res = fx.io.jload(res["numbers"]["source"])["logs"][L.name].get("behaviour") if os.path.exists(res["numbers"]["source"]) else None
    res["numbers"]["available"] = e0res is not None
    # the row sets of fx (key cache) against the item sets of the library's Evaluator for this log
    ev, rs = ses.ev, {}
    pairs = {"edits": (L.rows("edit"), ev.edit_items()), "hard negatives (valid)": (L.rows("hn", valid=True), ev.hn_items())}
    for pool in ("repair", "heldout", "heldout_disjoint", "heldout_strict", "all"):
        pairs[f"probes {pool} (valid)"] = (L.pool(pool), ev.probe_items(pool))
        pairs[f"probes {pool} (pre-correct: D2)"] = (L.rows("probe", pool=pool, where=lambda e: bool(e.get("pre_correct_long"))), ev.probe_items(pool, valid_only=False, pre_correct=True))
    for nm, (mine, lib) in pairs.items():
        a_, b_ = [L.lib_sid(j) for j in mine], [it["sid"] for it in lib]
        rs[nm] = {"n_fx": len(a_), "n_library": len(b_), "same_set": set(a_) == set(b_) and len(a_) == len(set(a_)), "same_order": a_ == b_}
    res["row_sets"] = rs
    res["row_sets_equal"] = all(v["same_set"] for v in rs.values())
    f0 = ses.stats["forwards"]
    t0 = time.time()
    gpu_gates = {}
    for sname in sorted(B):
        draw = None if sname == "static" else int(sname[4:])
        for mid in sorted(next(iter(B[sname].values()))):
            if mid == "deployed":
                M = fx.methods.deployed(L)
            elif mid == "pre-edit":
                M = fx.methods.preedit(L)
            else:
                M = e0_method(L, mid, draw)
            acc = res["by_method"].setdefault(mid, {"n_rows": 0, "n_differing": 0, "n_decision_differing": 0, "n_outcome_differing": 0, "explained_by_gpu_built_gate": 0,
                                                    "unexplained": 0})
            mine_all, theirs_all = {}, {}
            for gname, ms in B[sname].items():
                theirs = ms[mid]
                rows = [L.by_lib_sid(r["sid"]) for r in theirs]
                mine = ses.tf(M, rows)
                mine_all[gname], theirs_all[gname] = mine, theirs
                for j, a, b in zip(rows, mine, theirs):
                    acc["n_rows"] += 1
                    res["n_rows"] += 1
                    res["n_fields"] += len(CMP_OUT) + len(CMP_FLAGS)
                    do, df = diff_fields(a, b, CMP_OUT), diff_fields(a, b, CMP_FLAGS)
                    if not do and not df:
                        continue
                    acc["n_differing"] += 1
                    res["n_rows_differing"] += 1
                    acc["n_decision_differing"] += int(bool(df))
                    acc["n_outcome_differing"] += int(bool(do))
                    rec = {"set": sname, "method": mid, "sid": b["sid"], "fields": do + df, "fx": {k: a.get(k) for k in do + df}, "e0": {k: b.get(k) for k in do + df},
                           "near": a.get("near"), "src": a.get("src")}
                    if mid in E0_METHODS and draw is not None:          # cause: re-run with a gate built the sanity agent's way
                        if (mid, draw) not in gpu_gates:
                            gpu_gates.clear()
                            gpu_gates[(mid, draw)] = e0_gpu_gate(ses, mid, draw)
                        r2 = ses.ev_plain.tf_rows(gpu_gates[(mid, draw)], [ses.item(j)])[0]
                        same = not diff_fields(r2, b, CMP_OUT + CMP_FLAGS)
                        rec["gpu_built_gate_reproduces_e0"] = same
                        acc["explained_by_gpu_built_gate" if same else "unexplained"] += 1
                    else:
                        acc["unexplained"] += 1
                    if len(res["differing"]) < 60:
                        res["differing"].append(rec)
            if e0res is not None:                                       # the sanity agent's rendered numbers against fx.metrics on fx rows
                ref = e0res["methods"][mid]
                ref = ref["single"] if draw is None else ref["draws"][str(draw)]
                pm = mine_all["probes"]
                st = {"edit": fx.metrics.edit_stats(mine_all["edits"]), "hn": fx.metrics.hn_stats(mine_all["hn"]),
                      "probe": {"heldout_disjoint": fx.metrics.probe_stats(pm), "heldout_strict": fx.metrics.probe_stats([r for r in pm if r["strict"]])}}
                pairs = [(f"edit.{m}.{k}", st["edit"][m][k], ref["edit"][m][k]) for m in ("rel", "tgen", "igen") for k in ("exact", "token", "fired", "own_cand", "own_top")]
                if "gen_mean" in ref["edit"]:
                    pairs.append(("edit.rephrase.exact", st["edit"]["rephrase"]["exact"], ref["edit"]["gen_mean"]["exact"]))
                pairs += [(f"probe.{p}.{k}", st["probe"][p][k], ref["probe"][p][k]) for p in ("heldout_disjoint", "heldout_strict")
                          for k in ("D1", "n_D1", "fired", "n_fired", "D1_given_fired", "T-Loc*", "I-Loc*", "n_y0_degenerate", "D1_y0_not_degenerate") if k in ref["probe"][p]]
                pairs += [(f"hn.{t}.{k}", st["hn"][t][k], ref["hn"][t][k]) for t in ("HN-I", "HN-Q", "HN-X")
                          for k in ("D1", "fired", "own_top", "tok_loc", "n_y0_degenerate", "D1_y0_not_degenerate") if k in ref["hn"][t]]
                for nm, x, y in pairs:
                    res["numbers"]["n_compared"] += 1
                    dv = 0.0 if (x is None and y is None) else (float("inf") if (x is None or y is None) else abs(float(x) - float(y)))
                    res["numbers"]["max_abs_diff"] = max(res["numbers"]["max_abs_diff"], dv)
                    if dv > 1e-9:
                        res["numbers"]["n_differing"] += 1
                        if len(res["numbers"]["differing"]) < 20:
                            res["numbers"]["differing"].append({"set": sname, "method": mid, "number": nm, "fx": x, "e0": y})
            if mid in ("M1", "M3", "deployed") and sname in ("static", "draw0"):      # headline numbers both ways (development log)
                hd = {}
                for nm, rr in (("fx", mine_all), ("e0", theirs_all)):
                    e, p, h = fx.metrics.edit_stats(rr["edits"]), fx.metrics.probe_stats(rr["probes"]), fx.metrics.hn_stats(rr["hn"])
                    hd[nm] = {"rel": e["rel"]["exact"], "tgen": e["tgen"]["exact"], "igen": e["igen"]["exact"], "D1": p["D1"], "fired": p["fired"],
                              "T-Loc*": p["T-Loc*"], "I-Loc*": p["I-Loc*"], "hn_D1": h["all"]["D1"], "hn_fired": h["all"]["fired"]}
                res["headline"][f"{mid}/{sname}"] = hd
            say(f"  a {sname} {mid}: {acc}")
    res.update(seconds=time.time() - t0, forwards=ses.stats["forwards"] - f0, hook_mismatch=ses.stats["hook_mismatch"],
               n_unexplained=sum(v["unexplained"] for v in res["by_method"].values()),
               n_explained=sum(v["explained_by_gpu_built_gate"] for v in res["by_method"].values()))
    res["cause"] = ("rows that differ are re-run with a gate built as the sanity agent built it (log factors, rebased values and thresholds computed on the "
                    "GPU, calibration queries scored one by one); 'explained' = that gate reproduces the sanity agent's row, i.e. the difference is the last-bit "
                    "difference between CPU-built (fx) and GPU-built values / thresholds")
    res["ok"] = bool(res["n_unexplained"] == 0 and ses.stats["hook_mismatch"] == 0 and (res["n_rows_differing"] > 0 or res["numbers"]["n_differing"] == 0)
                     and res["row_sets_equal"])
    return res


# ----------------------------------------------------------------------------- b: memo correctness
def b_memo(L, H, ses):
    rng = np.random.default_rng(20261111)
    ms = [fx.methods.rescope(L, draw=0), fx.methods.cosine(L, draw=0), fx.methods.cosine(L, "default"), fx.methods.rescope(L, draw=1, alpha=0.2),
          fx.methods.ball(L, draw=0), fx.methods.rescope_shared(L, draw=2)]
    seqs = L.rows("edit") + L.pool("all") + L.rows("hn", valid=True)
    pairs = {}
    for M in ms:                                                        # fill the memo (first pass) and collect the (sequence, firing edit) pairs
        rows = ses.tf(M, seqs)
        for j, r in zip(seqs, rows):
            if r["fired"]:
                pairs.setdefault((j, ses.vsig(M.obj, r["top_last"])), (M, r))
    keys = sorted(pairs, key=lambda k: (k[0], k[1]))
    pick = [keys[int(i)] for i in sorted(rng.choice(len(keys), min(300, len(keys)), replace=False))]
    res = {"n_pairs_available": len(keys), "n_checked": len(pick), "n_equal": 0, "n_decision_equal": 0, "differing": []}
    by = {}
    for k in pick:
        by.setdefault(id(pairs[k][0]), (pairs[k][0], []))[1].append(k)
    t0 = time.time()
    for M, ks in by.values():                                           # fresh forwards through the library's Evaluator, the real gate, no memo
        fresh = ses.ev_plain.tf_rows(M.obj.replace(W0=H.W0), [ses.item(k[0]) for k in ks])
        for k, r in zip(ks, fresh):
            o = ses.memo.get(L.sid(k[0]), "tf", k[1])
            same = o is not None and all(o.get(f) == r.get(f) for f in CMP_OUT if f in r)
            dec = bool(r["fired"]) and int(r["top_last"]) == int(pairs[k][1]["top_last"])
            res["n_equal"] += int(same)
            res["n_decision_equal"] += int(dec)
            if not (same and dec) and len(res["differing"]) < 20:
                res["differing"].append({"csid": L.sid(k[0]), "memo": o, "fresh": {f: r.get(f) for f in CMP_OUT}, "method": M.label})
    res["seconds_fresh"] = time.time() - t0
    # the deployed weight: memoised rows against fresh forwards
    Md = fx.methods.deployed(L)
    sub = [seqs[int(i)] for i in sorted(rng.choice(len(seqs), 60, replace=False))]
    mem = ses.tf(Md, sub)
    fresh = ses.ev_plain.tf_rows(None, [ses.item(j) for j in sub])
    res["deployed"] = {"n": len(sub), "n_equal": sum(all(a.get(f) == b.get(f) for f in CMP_OUT if f in b) for a, b in zip(mem, fresh))}
    # decodes: memo against fresh
    M0 = ms[0]
    hn = L.rows("hn", valid=True)
    dsteps = ses.decide(M0, hn, "steps")
    fire = [j for j, f in zip(hn, dsteps["fired"]) if f]
    dsub = [fire[int(i)] for i in sorted(rng.choice(len(fire), min(25, len(fire)), replace=False))] + L.edits_of("tgen", range(0, L.T, 12)) + L.pool("heldout_disjoint")[::70]
    g1 = ses.generate(M0, dsub)
    fresh = ses.ev_plain.generate(M0.obj.replace(W0=H.W0), [ses.item(j) for j in dsub])
    fld = ("y", "long", "line", "stop", "long_stop", "n_long", "eq", "contains", "agree", "D2", "leak", "fired", "top_last")
    res["decode"] = {"n": len(dsub), "n_equal": sum(all(a.get(f) == b.get(f) for f in fld if f in b) for a, b in zip(g1, fresh)),
                     "n_fired": sum(bool(r["fired"]) for r in g1), "n_from_pre_edit_decode": sum(r["fired"] is False for r in g1)}
    gd = ses.generate(Md, dsub[:12])
    fresh = ses.ev_plain.generate(None, [ses.item(j) for j in dsub[:12]])
    res["decode_deployed"] = {"n": 12, "n_equal": sum(all(a.get(f) == b.get(f) for f in fld[:11] if f in b) for a, b in zip(gd, fresh))}
    # hard negatives with generated answers: every row decoded (decode_all) against the library's decodes; wrong-target flags with / without decode_all
    hsub = hn[::60]
    _, ga = ses.hard_negatives(M0, rows=hsub, generate=True, decode_all=True)
    _, gf = ses.hard_negatives(M0, rows=hsub, generate=True)
    libh = ses.ev_plain.generate(M0.obj.replace(W0=H.W0), [ses.item(j) for j in hsub])
    res["hn_decode_all"] = {"n": len(hsub), "n_equal_library": sum(a["line"] == b["line"] and a["fired"] == bool(b["fired"]) and a.get("leak") == b.get("leak") for a, b in zip(ga, libh)),
                            "n_decoded_without_decode_all": sum("line" in r for r in gf),
                            "wrong_target_same_with_and_without_decode_all": [a["wrong_target"] for a in ga] == [b["wrong_target"] for b in gf]}
    # DUAL composed from the memoised deployed / pre-edit decodes against the library's own DUAL (two fresh decodes per query)
    Mdual = fx.methods.dual(L)
    dl = dsub[::3][:16]
    mine, lib = ses.generate(Mdual, dl), ses.ev_plain.generate(Mdual.obj, [ses.item(j) for j in dl])
    fld_d = fld[:11] + ("fired", "dual", "n_generations", "n_new_tokens")
    res["dual"] = {"n": len(dl), "n_equal": sum(all(a.get(f) == b.get(f) for f in fld_d if f in b) for a, b in zip(mine, lib)),
                   "n_released_deployed": sum(bool(r["fired"]) for r in mine)}
    tfm, tfl = ses.dual_tf(Mdual, dl[:8]), ses.ev_plain.dual_tf_rows(Mdual.obj, [ses.item(j) for j in dl[:8]])
    res["dual_tf"] = {"n": len(tfl), "n_equal": sum(all(a.get(f) == b.get(f) for f in CMP_OUT + ("fired", "dual") if f in b) for a, b in zip(tfm, tfl))}
    # second pass: the model runs on nothing, the rows are the same
    f0, d0, s0 = ses.stats["forwards"], ses.stats["decodes"], ses.stats["seconds_model"]
    first = {M.label + str(i): ses.tf(M, seqs) for i, M in enumerate(ms)}
    f1 = ses.stats["forwards"]
    t0 = time.time()
    second = {M.label + str(i): ses.tf(M, seqs) for i, M in enumerate(ms)}
    g2 = ses.generate(M0, dsub)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "src"} for r in rows]                # noqa: E731
    res["second_pass"] = {"n_rows": sum(len(v) for v in second.values()) + len(g2), "forwards": ses.stats["forwards"] - f1, "decodes": ses.stats["decodes"] - d0,
                          "forwards_between": f1 - f0, "seconds": time.time() - t0, "seconds_model": ses.stats["seconds_model"] - s0,
                          "rows_equal": all(strip(first[k]) == strip(second[k]) for k in first) and strip(g1) == strip(g2),
                          "sources": {s: sum(r["src"] == s for v in second.values() for r in v) for s in ("pre", "memo", "run")}}
    res["hook_mismatch"] = ses.stats["hook_mismatch"]
    res["forced"] = _forced_misprediction(L, ses, M0, seqs, first[M0.label + "0"], dsub, g1)
    res["memo"] = ses.memo.summary()
    res["ok"] = bool(res["n_equal"] == res["n_checked"] == res["n_decision_equal"] and res["n_checked"] >= min(300, len(keys)) and res["deployed"]["n_equal"] == 60
                     and res["forced"]["ok"]
                     and res["decode"]["n_equal"] == len(dsub) and res["decode_deployed"]["n_equal"] == 12 and res["dual"]["n_equal"] == len(dl)
                     and res["dual_tf"]["n_equal"] == len(tfl) and res["hn_decode_all"]["n_equal_library"] == len(hsub)
                     and res["hn_decode_all"]["wrong_target_same_with_and_without_decode_all"] and res["second_pass"]["forwards"] == 0
                     and res["second_pass"]["decodes"] == 0 and res["second_pass"]["rows_equal"] and ses.memo.stats["conflicts"] == 0
                     and ses.stats["hook_mismatch"] == 0)
    return res


def _forced_misprediction(L, ses, M, seqs, true_tf, dsub, true_dec):
    """The must-not-happen path, forced: the decision handed to the memo path is FALSIFIED for a few firing sequences (it names
    another edit).  The forward then runs, the hook's own decision must win, the rows must be the rows of the true decision, every
    case must be counted, and nothing may be stored under the falsified signature."""
    g, T = M.obj, L.T
    wrong = lambda e: (np.asarray(e) + 1) % T                                                      # noqa: E731
    fire = [i for i, r in enumerate(true_tf) if r["fired"] and not ses.memo.has(L.sid(seqs[i]), "tf", ses.vsig(g, int(wrong(r["top_last"]))))]
    fire = fire[::max(1, len(fire) // 6)][:6]
    dfire = [i for i, r in enumerate(true_dec) if r["fired"]][:3]
    hm0, nm0, real = ses.stats["hook_mismatch"], len(ses.mismatches), ses.decide

    def fake(M_, rows, which="tf"):
        d = real(M_, rows, which)
        if which == "steps":
            d["step_top"] = np.where(d["step_fired"], wrong(d["step_top"]), d["step_top"])
        d["top_edit"] = np.where(d["fired"], wrong(d["top_edit"]), d["top_edit"])
        return d
    ses.decide = fake
    try:
        sub, dsb = [seqs[i] for i in fire], [dsub[i] for i in dfire]
        w0 = ses.memo.stats["writes"]
        ftf = ses.tf(M, sub)
        n_tf = ses.stats["hook_mismatch"] - hm0
        fdec = ses.generate(M, dsb)
        n_dec = ses.stats["hook_mismatch"] - hm0 - n_tf
        wsig = [ses._dec_sig(g, fake(M, dsb, "steps"), k) for k in range(len(dsb))]
    finally:
        del ses.decide                                                  # the method of the class again
        ses.stats["hook_mismatch"] = hm0
        del ses.mismatches[nm0:]
    f_tf = CMP_OUT + ("fired", "top_last", "own_fired_last", "own_released_last", "n_cand_edits")
    f_dec = ("y", "long", "line", "stop", "long_stop", "n_long", "eq", "contains", "agree", "D2", "leak", "fired", "top_last", "n_fire_steps", "own_released_last")
    out = {"tf": {"n": len(sub), "counted": n_tf, "rows_equal_true_rows": sum(all(a.get(f) == true_tf[i].get(f) for f in f_tf) for a, i in zip(ftf, fire)),
                  "stored_under_falsified_signature": sum(ses.memo.has(L.sid(seqs[i]), "tf", ses.vsig(g, int(wrong(true_tf[i]["top_last"])))) for i in fire)},
           "decode": {"n": len(dsb), "counted": n_dec, "rows_equal_true_rows": sum(all(a.get(f) == true_dec[i].get(f) for f in f_dec) for a, i in zip(fdec, dfire)),
                      "stored_under_falsified_signature": sum(ses.memo.has(L.sid(j), "decode", sg) for j, sg in zip(dsb, wsig))},
           "memo_writes": ses.memo.stats["writes"] - w0}
    out["ok"] = bool(len(sub) > 0 and len(dsb) > 0 and all(v["counted"] == v["n"] == v["rows_equal_true_rows"] and v["stored_under_falsified_signature"] == 0
                                                           for v in (out["tf"], out["decode"])) and out["memo_writes"] == 0)
    return out


# ----------------------------------------------------------------------------- c: fx.gate == the hook, on every sequence of one pass
def c_hook(L, H, ses):
    seqs = (L.rows("edit") + L.pool("all") + L.rows("hn", valid=True))[::STEP]
    res = {"n_sequences": len(seqs), "methods": {}}
    ok = True
    for M in (fx.methods.rescope(L, draw=0), fx.methods.cosine(L, draw=0)):
        r = {"capped_edits": M.rec["cal"]["n_edits_capped"]}
        for lp in ("tf", "prompt"):
            rows = seqs if lp == "tf" else [j for j in seqs if L.index[j].get("lp_shift")]
            if not rows:
                r[lp] = {"n": 0}
                continue
            t0 = time.time()
            lib = ses.ev_plain.tf_rows(M.obj.replace(W0=H.W0), [ses.item(j) for j in rows], lp=lp)     # every sequence through the hook, no memo
            sec = time.time() - t0
            cpu = fx.gate.decide(L, M, rows, lp)
            dev = ses.decide(M, rows, lp)
            mine = ses.tf(M, rows, lp=lp)
            c = {"n": len(rows), "seconds_full_pass": sec, "n_fired_hook": sum(bool(x["fired"]) for x in lib), "n_near": int(cpu["near"].sum()),
                 "cpu_vs_hook": 0, "device_vs_hook": 0, "rows_vs_hook": 0, "differing": []}
            for i, (j, x) in enumerate(zip(rows, lib)):
                own = "own_fired_last" in x
                hook = (bool(x["fired"]), int(x["top_last"]), x["n_cand_edits"] if x["fired"] else None, x.get("own_fired_last"), x.get("own_released_last"))
                for nm, d in (("cpu_vs_hook", cpu), ("device_vs_hook", dev)):
                    got = (bool(d["fired"][i]), int(d["top_edit"][i]), int(d["n_cand"][i]) if d["fired"][i] else None, bool(d["own_cand"][i]) if own else None,
                           bool(d["own_top"][i]) if own else None)
                    if got != hook:
                        c[nm] += 1
                        if len(c["differing"]) < 10:
                            c["differing"].append({"what": nm, "csid": L.sid(j), "fx": got, "hook": hook, "near": bool(cpu["near"][i])})
                if diff_fields(mine[i], x, CMP_OUT + CMP_FLAGS):
                    c["rows_vs_hook"] += 1
                    if len(c["differing"]) < 10:
                        c["differing"].append({"what": "row", "csid": L.sid(j), "fields": diff_fields(mine[i], x, CMP_OUT + CMP_FLAGS)})
            r[lp] = c
            ok &= c["device_vs_hook"] == 0 and c["rows_vs_hook"] == 0
            say(f"  c {M.label} lp={lp}: {({k: v for k, v in c.items() if k != 'differing'})}")
        res["methods"][M.label] = r
    # dense gates: the fast paths (router: memoised routed outcome; z-score: sequences below the threshold = pre-edit) against a full pass
    # through the hook on every 3rd sequence
    res["dense"] = {}
    dseq = seqs[::3]
    for M in (fx.methods.router(L, H, "conformal", 0.05, 200, 0), fx.methods.zscore(L, H, "conformal", 0.05, 200, 0), fx.methods.zscore(L, H, "default")):
        st0 = dict(ses.stats)
        t0 = time.time()
        mine = ses.tf(M, dseq)
        sec = time.time() - t0
        src = {k: ses.stats.get(k, 0) - st0.get(k, 0) for k in ("pre", "memo", "run", "lib", "forwards", "captures")}
        t0 = time.time()
        lib = ses.ev_plain.tf_rows(M.obj, [ses.item(j) for j in dseq])
        bad = [L.sid(j) for j, a, b in zip(dseq, mine, lib) if diff_fields(a, b, CMP_OUT + ("fired", "top_last", "n_cand_edits", "own_fired_last", "own_released_last"))]
        c = {"n": len(dseq), "threshold": float(M.obj.thr), "n_fired_hook": sum(bool(x["fired"]) for x in lib), "rows_vs_hook": len(bad), "differing": bad[:10],
             "seconds_fast": sec, "seconds_full_pass": time.time() - t0, "sources": src}
        res["dense"][M.label] = c
        ok &= not bad
        say(f"  c {M.label}: {({k: v for k, v in c.items() if k != 'differing'})}")
        ses.release()
    res["hook_mismatch"] = ses.stats["hook_mismatch"]
    res["note"] = ("cpu_vs_hook = fx.gate.decide (CPU, key cache) against the hook; device_vs_hook = fx.behave's decision (CPU + the rows inside the decision band "
                   "re-decided on the model's device); rows_vs_hook = the memo path's row against the row of the full pass")
    res["ok"] = bool(ok and ses.stats["hook_mismatch"] == 0)
    return res


# ----------------------------------------------------------------------------- d: the E1 row set end to end
def d_e1(L, H, ses_global):
    """The E1 row set end to end for one calibration draw, with a COLD private memo store (tests/out/memo_cold_<log>/, emptied first), so
    that the seconds per method are those of a first evaluation; then the calibrated rows once more for the next draw (warm memo)."""
    import shutil
    alpha, draw = 0.05, 0
    t0 = time.time()
    cold = os.path.join(OUT, f"memo_cold_{L.name}" + ("_smoke" if SMOKE else ""))
    shutil.rmtree(cold, ignore_errors=True)
    key = (L.name, id(H), True)
    old_root, old_dsig = fx.MEMO_ROOT, fx.behave._DSIG.pop(L.name, None)
    fx.MEMO_ROOT = cold                                                 # memo AND per-log auxiliary caches (z-scores, log-only negatives) start empty
    ses = fx.behave.Session(L, H)
    fx.behave._SES[key] = ses                                           # the constructors' session
    try:
        res = _d_e1(L, H, ses, alpha, draw)
    finally:
        fx.MEMO_ROOT = old_root
        fx.behave._SES[key] = ses_global
        fx.behave._DSIG.pop(L.name, None)
        if old_dsig is not None:
            fx.behave._DSIG[L.name] = old_dsig
        shutil.rmtree(cold, ignore_errors=True)
    res.update(seconds=time.time() - t0, memo_store="a private store, empty at the start, removed afterwards", hook_mismatch=ses.stats["hook_mismatch"], peak_gpu_gb=gpu_gb(), peak_rss_gb=rss_gb(),
               session_stats_cold=dict(ses.stats), memo_cold=ses.memo.summary())
    res["ok"] = bool(len(res["methods"]) == len(fx.methods.E1_LABELS) and not res["unavailable"] and ses.stats["hook_mismatch"] == 0)
    return res


def _d_e1(L, H, ses, alpha, draw):
    R = fx.io.Results("E1demo", os.path.join(OUT, f"e1_demo_{L.name}" + ("_smoke" if SMOKE else "")))
    ed, pr = L.rows("edit")[::STEP], L.pool("heldout_disjoint")[::STEP]
    hn = L.rows("hn", valid=True)[::4 * STEP]
    gen_e = L.edits_of("rel", range(0, L.T, 10)) + L.edits_of("tgen", range(0, L.T, 10))
    gen_p = L.rows("probe", pool="heldout_disjoint", where=lambda e: bool(e.get("pre_correct_long")))[::8]
    gen_h = hn[::16] if not SMOKE else hn[:3]
    if SMOKE:
        gen_e, gen_p = gen_e[::5], gen_p[::5]
    n_tf = len(ed) + len(pr) + len(hn)
    res = {"alpha": alpha, "draw": draw, "n_sequences": {"edits": len(ed), "probes": len(pr), "hn": len(hn), "gen_edits": len(gen_e), "d2": len(gen_p), "gen_hn": len(gen_h)},
           "methods": {}, "unavailable": {}}
    rows_by = {}
    cnt = ("forwards", "decodes", "pre", "memo", "run", "lib", "captures")
    for lab in fx.methods.E1_LABELS:                                    # one method at a time (the device holds one gate / correction)
        t1 = time.time()
        M = fx.methods.e1_rowset(L, alpha, draw, H=H, labels=(lab,), skip_unavailable=False)[0]
        sec_build = time.time() - t1
        if not M.available:
            res["unavailable"][M.label] = M.rec.get("unavailable")
            continue
        st0 = dict(ses.stats)
        m = {"kind": M.kind, "memo_path": M.memo, "key": M.key, "seconds_build": sec_build}
        kw = dict(alpha=M.alpha, n=M.n, draw=M.draw, variant=M.variant)
        if M.kind != "dual":
            t1 = time.time()
            re_, rp, rh = ses.tf(M, ed), ses.tf(M, pr), ses.tf(M, hn)
            m["seconds_tf"] = time.time() - t1
            m["seconds_per_sequence"] = m["seconds_tf"] / n_tf
            rows_by[M.label] = re_
            R.add_nested(L.name, M.label, fx.metrics.edit_stats(re_, ci=True), prefix="edit.", pool="edits", **kw)
            R.add_nested(L.name, M.label, fx.metrics.probe_stats(rp, ci=True), prefix="probe.", pool="heldout_disjoint", **kw)
            R.add_nested(L.name, M.label, fx.metrics.hn_stats(rh), prefix="hn.", pool="hard_negatives/4", **kw)
            R.rows(f"{M.label}__tf", re_ + rp + rh)
            if hasattr(M.obj, "det_tau") and fx.gate.is_scoped(M.obj):
                t1 = time.time()
                gs = fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf", "prompt", "decode")))
                m["seconds_gate_level"] = time.time() - t1
                R.add_nested(L.name, M.label, gs, prefix="gate.", pool="gate_level", **kw)
        m.update({k: ses.stats.get(k, 0) - st0.get(k, 0) for k in cnt})
        if M.label in ("deployed", "rescope", "cosine", "dual", "p12_comp_b_nall"):       # decode metrics on a subsample (the methods of the protocol + DUAL)
            st1 = dict(ses.stats)
            t1 = time.time()
            ge, gp = ses.generate(M, gen_e), ses.d2(M, rows=gen_p)
            _, gh = ses.hard_negatives(M, rows=gen_h, generate=True)
            m["seconds_decode"] = time.time() - t1
            m["n_decode_items"] = len(gen_e) + len(gen_p) + len(gen_h)
            m["decodes_run"] = ses.stats["decodes"] - st1["decodes"]
            R.add_nested(L.name, M.label, fx.metrics.gen_stats(ge), prefix="gen.", pool="edit_prompts/10", **kw)
            R.add_nested(L.name, M.label, fx.metrics.d2_stats(gp), prefix="d2.", pool="heldout_disjoint/8", **kw)
            R.add_nested(L.name, M.label, {t: {k: v for k, v in s.items() if k in ("wrong_target", "eq_own", "eq_any_edit", "n_gen", "n_decoded", "fired_decode")}
                                           for t, s in fx.metrics.hn_stats(None, gh).items()}, prefix="hn_gen.", pool="hard_negatives/64", **kw)
            R.rows(f"{M.label}__gen", ge + gp + gh)
        if M.label in fx.methods.E1_DEPENDS:                             # the same row for the NEXT calibration draw: what the memo saves
            M2 = fx.methods.e1_rowset(L, alpha, draw + 1, H=H, labels=(lab,))[0]
            st1 = dict(ses.stats)
            t1 = time.time()
            ses.tf(M2, ed), ses.tf(M2, pr), ses.tf(M2, hn)
            m["seconds_tf_next_draw"] = time.time() - t1
            m["next_draw"] = {k: ses.stats.get(k, 0) - st1.get(k, 0) for k in cnt}
            del M2
        m["record_calibration"] = {k: M.rec.get("cal", {}).get(k) for k in ("eta", "n_edits_capped", "bound_false_firing", "threshold")} if "cal" in M.rec else None
        res["methods"][M.label] = m
        say(f"  d {M.label}: {({k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items() if k not in ('record_calibration',)})}")
        del M
        ses.release()
    if "rescope" in rows_by:                                            # the primary endpoint, paired over edits
        for other in ("cosine", "ball", "deployed"):
            if other in rows_by:
                e = fx.metrics.endpoint(rows_by["rescope"], rows_by[other])
                R.add_stat(L.name, "rescope", "endpoint.rephrase_exact_diff", e["rephrase"], variant=f"minus {other}", alpha=alpha, n=200, draw=draw, pool="edits",
                           p_two_sided=e["rephrase"].get("p_two_sided"))
    R.save(test="d_e1", log=L.name)
    short = "Q" if L.backbone == "qwen25vl" else "L"
    num = {r["metric"]: r["value"] for r in R.records if r["method"] == "rescope"}
    R.paper({f"Demo{short}RescopeTGen": num.get("edit.tgen.exact"), f"Demo{short}RescopeFire": num.get("probe.fired")})
    cols = [("edit.rel.exact",), ("edit.tgen.exact",), ("edit.igen.exact",), ("probe.fired",), ("probe.D1",), ("probe.T-Loc*",), ("probe.I-Loc*",), ("hn.all.fired",),
            ("hn.all.D1",)]
    R.table(f"E1 row set on {L.name}, alpha = {alpha}, draw {draw} (test d of the fx driver; hard negatives: every 4th row; cells: value [95 % bootstrap interval])\n"
            + fx.io.pivot(R.records, row=("method",), col=("metric",), where=lambda r: (r["metric"],) in cols, col_order=cols))
    res.update(out=R.out, n_records=len(R.records), n_tf=n_tf)
    return res


# ----------------------------------------------------------------------------- f: complaint-driven constructors
def f_where(L, H, ses):
    """Rescope-WHERE + least repair of reported complaints (the P12 stream, seed 0) and the evidence-curve constructor: they build, the
    repaired complaints / queries are silent afterwards, and the least repair does not depend on the order of the complaints."""
    ns = (4, 8) if SMOKE else (10, 50)
    res = {"ns": list(ns)}
    for n in ns:
        cs = fx.draws.complaints(L, n, 0)
        t1 = time.time()
        Mw = fx.methods.where_least(L, cs, H=H, seed=0)
        sec = time.time() - t1
        wl = Mw.rec["least_repair"]
        Mr = fx.methods.where_least(L, cs[::-1], H=H, seed=0)             # the same complaints in reverse order
        rw, dep = ses.tf(Mw, cs), ses.tf(fx.methods.deployed(L), cs)
        res[f"where_least_n{n}"] = {"n": len(cs), "seconds_build": sec, "n_refused": wl["n_refused"], "n_with_conflict": wl["n_with_conflict"],
                                    "n_resolved": wl["n_resolved"], "complaints_firing_after": sum(bool(r["fired"]) for r in rw),
                                    "complaints_D1_after": sum(bool(r["D1"]) for r in rw), "complaints_D1_deployed": sum(bool(r["D1"]) for r in dep),
                                    "resolved_complaints_silent": all(not r["fired"] for r, sid in zip(rw, wl["sids"]) if sid not in wl["refused_sids"]),
                                    "order_independent": bool(torch.equal(Mw.obj.det_tau, Mr.obj.det_tau)),
                                    "refused_set_order_independent": sorted(wl["refused_sids"]) == sorted(Mr.rec["least_repair"]["refused_sids"])}
        say(f"  f where_least n = {n}: {res[f'where_least_n{n}']}")
    Mp = fx.methods.where_least(L, fx.draws.complaints(L, 10, 0), H=H, seed=0, on_conflict="partial")
    res["where_least_n10_partial"] = {"n_with_conflict": Mp.rec["least_repair"]["n_with_conflict"], "n_refused": Mp.rec["least_repair"]["n_refused"]}
    Ml = fx.methods.where_logonly(L, H)
    res["where_logonly"] = {"capped_edits": Ml.rec["cal"]["n_edits_capped"], "own_keys_silenced_by_literal_rule": Ml.rec["cal"]["n_own_keys_silenced_by_literal_rule"],
                            "cached": os.path.exists(os.path.join(fx.MEMO_ROOT, "aux", L.name, "selfneg_ncoord_all.json"))}
    for n in (0, 10, 39, 40):
        Me = fx.methods.evidence(L, n, 0)
        q = fx.draws.calibration(L, n, 0).rows if n else []
        r = ses.tf(Me, q) if q else []
        res[f"evidence_n{n}"] = {"rule": Me.rec["calibration"], "n_queries": len(q), "queries_firing_after": sum(bool(x["fired"]) for x in r),
                                 "n_refused": Me.rec.get("least_repair", {}).get("n_refused"), "eta_infinite": Me.rec["cal"].get("eta_infinite")}
    res["ok"] = bool(all(res[f"where_least_n{n}"][k] for n in ns for k in ("resolved_complaints_silent", "order_independent", "refused_set_order_independent"))
                     and res["where_logonly"]["cached"] and res["evidence_n0"]["rule"] == "log_only" and res["evidence_n40"]["rule"] == "two_block"
                     and all(res[f"evidence_n{n}"]["queries_firing_after"] <= (res[f"evidence_n{n}"]["n_refused"] or 0) for n in (10, 39))
                     and ses.stats["hook_mismatch"] == 0)
    return res


# ----------------------------------------------------------------------------- g: ablation switches on the GPU
def g_ablations(L, H, ses):
    """The ablation switches of E6: each variant builds (the roles that look at every position calibrate on captured keys: one
    truncated forward per calibration query), runs through fx.behave on a sample of sequences, and gives the rows of the plain
    Evaluator with the same gate (memo-path variants: an independent path; library-path variants: the same path, i.e. a plumbing check)."""
    seqs = L.rows("edit")[::10] + L.pool("heldout_disjoint")[::12] + L.rows("hn", valid=True)[::40]
    res = {"n_sequences": len(seqs), "variants": {}}
    ok = True
    variants = [("role=last+answer", "rescope", {"role": "last+answer"}), ("role=all", "rescope", {"role": "all"}), ("response=gain", "rescope", {"response": "gain"}),
                ("response=lin", "rescope", {"response": "lin"}), ("fire=all", "rescope", {"fire": "all"}), ("rebase=False", "rescope", {"rebase": False}),
                ("feature=coord", "rescope", {"feature": "coord"}), ("split=1:3", "rescope", {"split": (1, 3)}), ("baseline=0.98", "rescope", {"baseline": 0.98}),
                ("which=all,guards=all", "rescope", {"which": "all", "guards": "all"}), ("source=BEN-S", "rescope", {"source": "BEN-S"}),
                ("contamination=0.05", "rescope", {"contamination": 0.05}), ("calibration=shared", "rescope_shared", {}), ("log-only", "rescope_logonly", {}),
                ("rescope_null", "rescope_null", {}), ("rescope_null ridge 0.1", "rescope_null", {"ridge": 0.1}),
                ("running_max", "rescope", {"calibration": "running_max"}), ("cosine role=all two-block", "cosine", {"role": "all"}),
                ("zscore two_block", "zscore", {"calibration": "two_block"}), ("zscore dynamic", "zscore", {"calibration": "dynamic"}),
                ("router two_block", "router", {"calibration": "two_block"}), ("ball half_min", "ball", {"calibration": "half_min"})]
    for tag, name, kw in variants:
        t0 = time.time()
        M = fx.methods.build(name, L, H=H, **kw)
        build = time.time() - t0
        st0 = dict(ses.stats)
        mine = ses.tf(M, seqs)
        ref = ses.ev_plain.tf_rows(ses._lib_obj(M), [ses.item(j) for j in seqs])
        bad = [L.sid(j) for j, a, b in zip(seqs, mine, ref) if diff_fields(a, b, CMP_OUT + CMP_FLAGS)]
        cal = M.rec.get("cal") or {}
        res["variants"][tag] = {"kind": M.kind, "memo_path": M.memo, "variant": M.variant, "key": M.key, "seconds_build": build, "n_fired": sum(bool(r["fired"]) for r in ref),
                                "rows_differing": len(bad), "differing": bad[:5], "eta_finite": None if "eta" not in cal else bool(np.isfinite(cal["eta"])),
                                "n_edits_capped": cal.get("n_edits_capped"), "forwards": ses.stats["forwards"] - st0["forwards"], "lib": ses.stats["lib"] - st0["lib"]}
        ok &= not bad
        say(f"  g {tag}: {({k: v for k, v in res['variants'][tag].items() if k != 'differing'})}")
        del M
        ses.release()
    res["keys_distinct"] = len({v["key"] for v in res["variants"].values()}) == len(variants)
    res["ok"] = bool(ok and res["keys_distinct"] and ses.stats["hook_mismatch"] == 0)
    return res


# ----------------------------------------------------------------------------- e: mechanics on a long log (no metric)
def e_mechanics(name):
    t = {}
    t0 = time.time()
    L = fx.logs.open(name)
    t["open"] = time.time() - t0
    t0 = time.time()
    M = fx.methods.rescope(L, alpha=0.05, n=200, draw=0)
    t["build_rescope"] = time.time() - t0
    t0 = time.time()
    ev = fx.gate.evaluate(L, M, positions=("tf",))
    t["gate_level_pass"] = time.time() - t0
    n_gate = {p: len(v[next(iter(v))]["rows"]) for p, v in ev.items()}
    t0 = time.time()
    M2 = fx.methods.rescope(L, alpha=0.05, n=200, draw=1)
    fx.gate.evaluate(L, M2, positions=("tf",))
    t["second_draw_build_and_gate_pass"] = time.time() - t0
    rss_cpu = rss_gb()
    t0 = time.time()
    H = fx.logs.load_model(name)
    t["load_model"] = time.time() - t0
    import shutil
    cold = os.path.join(OUT, f"memo_cold_{name}_e")                     # a private, empty memo store: the 50 sequences really run
    shutil.rmtree(cold, ignore_errors=True)
    old_root, fx.MEMO_ROOT = fx.MEMO_ROOT, cold
    try:
        ses = fx.behave.Session(L, H)
        seqs = L.edits_of("rel", range(L.T - 50, L.T))                  # the LAST 50 edits' own prompts: they fire by construction (guards)
        t0 = time.time()
        rows = ses.tf(M, seqs)
        t["behaviour_50_sequences"] = time.time() - t0
        t0 = time.time()
        rows2 = ses.tf(M, seqs)
        t["behaviour_50_sequences_second_pass"] = time.time() - t0
    finally:
        fx.MEMO_ROOT = old_root
        shutil.rmtree(cold, ignore_errors=True)
    n_fwd = ses.stats["forwards"]
    # cost per forward / decode on this log, for projections (the outcomes are discarded: nothing is computed from them)
    it50 = [ses.item(j) for j in L.edits_of("tgen", range(50))]
    per = {}
    t0 = time.time()
    ses.ev_plain.tf_rows(None, it50)
    per["forward_deployed_weight"] = (time.time() - t0) / 50
    gg = ses.hook_gate(M)
    t0 = time.time()
    ses.ev_plain.tf_rows(gg, it50)
    per["forward_gate_role_last"] = (time.time() - t0) / 50
    ga = fx.methods.cosine(L, "default", role="all")
    t0 = time.time()
    ses.ev_plain.tf_rows(ses._lib_obj(ga), it50[:20])
    per["forward_gate_role_all"] = (time.time() - t0) / 20
    ses.release()
    t0 = time.time()
    ses.ev_plain.generate(None, it50[:10])
    per["decode_deployed_weight"] = (time.time() - t0) / 10
    del gg, ga
    return {"log": name, "role": L.role, "T": L.T, "seconds": t, "seconds_per_item": per, "n_sequences": {"edit": 3 * L.T, "probe_heldout_disjoint": len(L.pool("heldout_disjoint")),
                                                                                                         "hn_valid": len(L.rows("hn", valid=True))},
            "n_gate_rows": n_gate, "n_behaviour_rows": len(rows), "rows_complete": all("exact" in r for r in rows),
            "second_pass_equal": [{k: v for k, v in r.items() if k != "src"} for r in rows] == [{k: v for k, v in r.items() if k != "src"} for r in rows2],
            "forwards": n_fwd, "sources_first_pass": {x: sum(r["src"] == x for r in rows) for x in ("pre", "memo", "run")},
            "sources_second_pass": {x: sum(r["src"] == x for r in rows2) for x in ("pre", "memo", "run")},
            "hook_mismatch": ses.stats["hook_mismatch"], "peak_rss_gb_before_model": rss_cpu, "peak_rss_gb": rss_gb(),
            "peak_gpu_gb": gpu_gb(), "load_info": getattr(H, "load_info", None), "lean": bool(getattr(H, "load_info", {}).get("lean")),
            "note": "mechanics only: no metric of this log is computed, stored or printed",
            "ok": bool(len(rows) == 50 and all("exact" in r for r in rows) and ses.stats["hook_mismatch"] == 0 and n_fwd == 50
                       and all(r["src"] == "memo" for r in rows2))}


TESTS = {"a": ("regress", a_regress), "b": ("memo", b_memo), "c": ("hook", c_hook), "d": ("e1", d_e1), "f": ("where", f_where), "g": ("ablations", g_ablations)}

if __name__ == "__main__":
    name, which = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    force = "--force" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    code = fx.code_fingerprint()
    say(f"{name}: tests {which}, GQ_MEM {os.environ.get('GQ_MEM')}, CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')}, code {code}")
    if which == ["e"]:
        res = e_mechanics(name)
        res.update(test="mechanics", code=code, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        fx.io.jdump(res, os.path.join(OUT, f"gpu_{name}_e_mechanics.json"), indent=1)
        say(f"e mechanics {name}: {'PASS' if res['ok'] else 'FAIL'}; seconds {({k: round(v, 1) for k, v in res['seconds'].items()})}, per item "
            f"{({k: round(v, 3) for k, v in res['seconds_per_item'].items()})}, peak RAM {res['peak_rss_gb']:.1f} GB, peak GPU {res['peak_gpu_gb']:.2f} GB")
        sys.exit(0)
    if name not in DEV:
        raise SystemExit(f"{name} is not a development log: only test e (mechanics) may touch it")
    todo = []
    for k in which:
        p = os.path.join(OUT, f"gpu_{name}_{k}_{TESTS[k][0]}" + ("_smoke" if SMOKE else "") + ".json")
        if os.path.exists(p) and not force and fx.io.jload(p).get("code") == code and fx.io.jload(p).get("ok"):
            say(f"{k}: already passed with this code, skipped")
            continue
        todo.append((k, p))
    if todo:
        t0 = time.time()
        L = fx.logs.open(name)
        H = fx.logs.load_model(name)
        ses = fx.behave.session(L, H)
        say(f"loaded in {time.time() - t0:.0f}s; checks {H.checks}; memo {ses.memo.summary()}")
        for k, p in todo:
            t0 = time.time()
            res = TESTS[k][1](L, H, ses)
            res.update(test=TESTS[k][0], log=name, seconds_test=time.time() - t0, code=code, time=time.strftime("%Y-%m-%d %H:%M:%S"), session_stats=dict(ses.stats),
                       peak_gpu_gb=gpu_gb(), peak_rss_gb=rss_gb())
            fx.io.jdump(res, p, indent=1)
            say(f"{k} {TESTS[k][0]}: {'PASS' if res['ok'] else 'FAIL'} ({res['seconds_test']:.0f}s); session {ses.stats}")
        H.reset()
        say(f"module weight is W_E after the tests: {bool(torch.equal(H.mod.weight.data, H.WE))}")
