"""Check 4: baseline DUAL and token-level locality, recomputed with my own code and compared with the library.  GPU.

  gq run --mem <MiB> -- python c4_dual_loc.py Q|L        -> out/c4_<tag>/*.json (stages, resumable), out/c4_<tag>.json

DUAL (PROTOCOL.md section 6, amendment "Added baseline DUAL"): run the pre-edit and the deployed model on the query and return
the deployed model's answer only if it equals the target of a logged edit, otherwise the pre-edit answer.
  library   ev.generate({"deployed": None, "pre-edit": W_0, "dual": ev.dual("eq"), "dual_contains": ev.dual("contains")}, items)
  mine      my own decoder v2 (vgpu.my_decode) on the model with MY weights (W_0 read from the checkpoint file, W_E rebuilt from
            the stored batch deltas), my own answer normalisation and matching rule (vgpu.dual_match), items from the raw data
  items     100 valid HELD-OUT-DISJOINT probe rows (50 image, 50 text) and 50 edit prompts (Rel / T-Gen / I-Gen), seeded
Token-level locality of the deployed log on the valid HELD-OUT-DISJOINT rows:
  library   ev.token_locality(None, "heldout_disjoint")  (T-Loc* / I-Loc*, stored benchmark locality, signed difference)
  mine      teacher forcing of the pre-edit answer at W_0 and at W_E (B = 1), share of answer positions with the same argmax
  and, to explain the gap to the stored benchmark locality, the BENCHMARK quantity itself at B = 1 with my code: reference
  answers (loc_ans / m_loc_a), benchmark formats (T-Loc: 'plain'), all 500 probe items.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import vcommon as V
import vgpu as G

os.environ.setdefault("RESCOPE_CACHE", V.VER + "/work/hcache")
import torch  # noqa: E402
import rescope  # noqa: E402
from rescope import Evaluator, load, summarise  # noqa: E402

SEED = 20261111


def main(tag):
    d = V.DEV[tag]
    name, bb = d["log"], d["backbone"]
    odir = f"{V.OUT}/c4_{tag}"
    os.makedirs(odir, exist_ok=True)
    mem = int(os.environ.get("GQ_MEM", d["mem"]))
    cap = (mem - 600) / 1024.0
    t0 = time.time()
    H = load(name, device="cuda:0", mem_gb=cap)
    ev = Evaluator(H, hard_negatives=False)
    vlm, mod = H.vlm, H.mod
    res = {"log": name, "load_checks": H.checks}
    # ------------------------------------------------------------------ my weights
    Wc, wkey, _ = G.checkpoint_weight(bb, d["layer"])
    W0 = Wc.to(H.dev, H.dtype)
    WE, comp_ok = G.deployed_weight(name, W0)
    res["weights"] = {"checkpoint_key": wkey, "H.W0_equals_checkpoint_file": bool(torch.equal(H.W0, W0)), "H.WE_equals_my_rebuild": bool(torch.equal(H.WE, WE)),
                      "my_WE_minus_W0_equals_stored_composite": comp_ok, "module_name": H.module_name == vlm.edit_module_name(d["layer"])}
    V.say(f"{name}: loaded ({time.time() - t0:.0f}s); weights {res['weights']}")

    def at(W):
        mod.weight.data.copy_(W)
        assert torch.equal(mod.weight.data, W)

    # ------------------------------------------------------------------ items from the raw data
    recs = V.jsonl(f"{V.TESTBED}/{name}/probe_records_v2.jsonl")
    ann = V.annotate(recs)
    train = {x["uid"]: x for x in V.vlkeb("train")}
    evl = {x["uid"]: x for x in V.vlkeb("eval")}
    stream = [train[r["uid"]] for r in V.stream_items(name)]
    targets = [x["alt"] for x in stream]
    hd = [r for r, a in zip(recs, ann) if a["disjoint"] and r["y0_valid"]]
    rng = np.random.default_rng([SEED, 4])
    pick = []
    for pt in ("image", "text"):
        sub = [r for r in hd if r["ptype"] == pt]
        pick += [sub[int(i)] for i in sorted(rng.choice(len(sub), 50, replace=False))]
    mine_items = [{"sid": f"probe:{r['ptype']}:{r['probe_idx']}", "kind": "probe", "q": {"fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"]}} for r in pick]
    em = [(m, i) for m in ("rel", "tgen", "igen") for i in range(len(stream))]
    for j in sorted(rng.choice(len(em), 50, replace=False)):
        m, i = em[int(j)]
        x = stream[i]
        mine_items.append({"sid": f"edit:{m}:{i}", "kind": "edit", "edit": i, "metric": m,
                           "q": {"fmt": "vqa", "prompt": x["rephrase"] if m == "tgen" else x["src"], "image": x["image_rephrase"] if m == "igen" else x["image"]}})
    lib_all = {it["sid"]: it for it in ev.probe_items("heldout_disjoint") + ev.edit_items()}
    lib_items = [lib_all[it["sid"]] for it in mine_items]
    # ------------------------------------------------------------------ stage 1: library generation
    p = f"{odir}/lib_gen.json"
    if not os.path.exists(p):
        t1 = time.time()
        out = ev.generate({"deployed": None, "pre-edit": H.W0, "dual": ev.dual("eq"), "dual_contains": ev.dual("contains")}, lib_items)
        V.jdump({"rows": out, "seconds": time.time() - t1}, p, indent=None)
        V.say(f"library generation: {time.time() - t1:.0f}s")
    lib = V.jload(p)["rows"]
    # ------------------------------------------------------------------ stage 2: my generation
    p = f"{odir}/my_gen.json"
    if not os.path.exists(p):
        t1 = time.time()
        mine = {}
        for wname, Wt in (("deployed", WE), ("pre-edit", W0)):
            at(Wt)
            mine[wname] = [G.my_decode(vlm, G.build(vlm, it["q"])) for it in mine_items]
        H.reset()
        V.jdump({"rows": mine, "seconds": time.time() - t1}, p, indent=None)
        V.say(f"my generation: {time.time() - t1:.0f}s")
    mine = V.jload(p)["rows"]
    cmp = {"n_items": len(mine_items), "n_probe": 100, "n_edit": 50}
    for w in ("deployed", "pre-edit"):
        cmp[f"{w}_long_equal"] = sum(a["long"] == b["long"] for a, b in zip(mine[w], lib[w]))
        cmp[f"{w}_y_equal"] = sum(a["y"] == b["y"] for a, b in zip(mine[w], lib[w]))
        cmp[f"{w}_line_equal"] = sum(a["line"] == b["line"] for a, b in zip(mine[w], lib[w]))
    dual = {}
    for variant, lname in (("eq", "dual"), ("contains", "dual_contains")):
        rows = []
        for it, e, o, l in zip(mine_items, mine["deployed"], mine["pre-edit"], lib[lname]):
            m = G.dual_match(targets, e["line"], e["long"], variant)
            ret = e if m else o
            own_eq = None
            if it["kind"] == "edit":                                   # the returned answer equals the item's own target (row field `eq`)
                own_eq = G.norm_answer(ret["line"]) == G.norm_answer(targets[it["edit"]])
            rows.append({"sid": it["sid"], "kind": it["kind"], "released": bool(m), "edits": m[:8], "n_match": len(m), "returned_long": ret["long"], "returned_y": ret["y"],
                         "own_eq": own_eq, "lib_released": bool(l["fired"]), "lib_edits": l["dual"]["edits"], "lib_n_match": l["dual"]["n_matching_edits"],
                         "lib_long": l["long"], "lib_y": l["y"], "lib_eq": l.get("eq"), "lib_source": l["dual"]["source"],
                         "lib_n_new_tokens": l["n_new_tokens"], "my_n_new_tokens": [e["n_long"], o["n_long"]]})
        k = lambda f: sum(bool(f(r)) for r in rows)                   # noqa: E731
        dual[variant] = {"n": len(rows), "released_equal": k(lambda r: r["released"] == r["lib_released"]), "matching_edits_equal": k(lambda r: r["edits"] == r["lib_edits"] and r["n_match"] == r["lib_n_match"]),
                         "returned_long_equal": k(lambda r: r["returned_long"] == r["lib_long"]), "returned_y_equal": k(lambda r: r["returned_y"] == r["lib_y"]),
                         "own_eq_equal": [sum(r["own_eq"] == r["lib_eq"] for r in rows if r["kind"] == "edit"), sum(r["kind"] == "edit" for r in rows)],
                         "n_new_tokens_equal": k(lambda r: r["my_n_new_tokens"] == r["lib_n_new_tokens"]),
                         "mine": {"probe_released": [sum(r["released"] for r in rows if r["kind"] == "probe"), 100],
                                  "edit_released": [sum(r["released"] for r in rows if r["kind"] == "edit"), 50],
                                  "edit_own_target_returned": [sum(bool(r["own_eq"]) for r in rows if r["kind"] == "edit"), 50]},
                         "library": {"probe_released": [sum(r["lib_released"] for r in rows if r["kind"] == "probe"), 100],
                                     "edit_released": [sum(r["lib_released"] for r in rows if r["kind"] == "edit"), 50],
                                     "edit_own_target_returned": [sum(bool(r["lib_eq"]) for r in rows if r["kind"] == "edit"), 50]},
                         "differing": [r for r in rows if not (r["released"] == r["lib_released"] and r["returned_long"] == r["lib_long"] and r["edits"] == r["lib_edits"])][:10]}
    cmp["dual"] = dual
    try:
        cmp["library_summarise_dual"] = summarise(lib["dual"]).get("dual")
    except Exception as e:                                      # convenience only
        cmp["library_summarise_dual"] = f"{type(e).__name__}: {e}"
    cmp["degenerate_library_rows"] = {w: sum(any(r["degenerate"].values()) for r in lib[w]) for w in ("deployed", "pre-edit", "dual")}
    res["dual"] = cmp
    V.say(f"DUAL: { {k: v for k, v in cmp.items() if k != 'dual'} }")
    for v_, x in dual.items():
        V.say(f"  {v_}: { {k: x[k] for k in x if k != 'differing'} }")
    # ------------------------------------------------------------------ stage 3: library token locality of the deployed log
    p = f"{odir}/lib_loc.json"
    if not os.path.exists(p):
        t1 = time.time()
        V.jdump(dict(ev.token_locality(None, "heldout_disjoint"), seconds=time.time() - t1), p, indent=None)
    ll = V.jload(p)
    # ------------------------------------------------------------------ stage 4: my token locality + the benchmark quantity at B = 1
    p = f"{odir}/my_loc.json"
    if not os.path.exists(p):
        t1 = time.time()
        rows = [{"sid": f"probe:{r['ptype']}:{r['probe_idx']}", "ptype": r["ptype"], "q": {"fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"]}, "ans": r["y0_text"],
                 "D1_stored": bool(r["is_complaint_D1"])} for r in hd]
        uids = []
        for r in recs:
            if r["uid"] not in uids:
                uids.append(r["uid"])
        bench = []
        for u in uids:
            x = evl[u]
            bench.append({"uid": u, "ptype": "text", "q": {"fmt": "plain", "prompt": x["loc"], "image": None}, "ans": x["loc_ans"]})
            bench.append({"uid": u, "ptype": "image", "q": {"fmt": "vqa", "prompt": x["m_loc_q"], "image": x["m_loc"]}, "ans": x["m_loc_a"]})
        for wname, Wt in (("pre", W0), ("post", WE)):
            at(Wt)
            for r in rows + bench:
                pred, ans, fin = G.tf_forward(vlm, G.build(vlm, r["q"], r["ans"]))
                r[wname] = pred.tolist()
                r[wname + "_finite"] = fin
                r["ans_ids"] = ans.tolist()
        H.reset()
        V.jdump({"rows": rows, "bench": bench, "seconds": time.time() - t1}, p, indent=None)
        V.say(f"my locality passes: {time.time() - t1:.0f}s")
    ml = V.jload(p)

    def agree(r):
        if not (r["pre_finite"] and r["post_finite"]):
            return 0.0
        return float(np.mean(np.asarray(r["pre"]) == np.asarray(r["post"])))
    loc = {}
    rows = ml["rows"]
    lib_row = {r["sid"]: r for r in ll["per_row"]}
    for nm, pt in (("T-Loc*", "text"), ("I-Loc*", "image")):
        sub = [r for r in rows if r["ptype"] == pt]
        loc[nm] = {"n": len(sub), "mine": 100.0 * float(np.mean([agree(r) for r in sub])), "library": ll[nm], "n_library": ll[f"n_{pt}"]}
    loc["n_rows"] = len(rows)
    loc["rows_equal_library_rows"] = [r["sid"] for r in rows] == [r["sid"] for r in ll["per_row"]]
    # the library stores float32 means (1/3 -> 0.33333334): compare to 1e-6
    loc["per_row_tok_loc_differing"] = sum(abs(agree(r) - lib_row[r["sid"]]["tok_loc"]) > 1e-6 for r in rows if r["sid"] in lib_row)
    loc["per_row_max_abs_diff"] = max(abs(agree(r) - lib_row[r["sid"]]["tok_loc"]) for r in rows if r["sid"] in lib_row)
    loc["y0_reproduced_by_my_W0"] = [sum(r["pre_finite"] and r["pre"] == r["ans_ids"] for r in rows), len(rows)]
    loc["D1_mine"] = 100.0 * float(np.mean([not (r["post_finite"] and r["post"] == r["ans_ids"]) for r in rows]))
    loc["D1_flags_equal_stored_complaints"] = [sum((not (r["post_finite"] and r["post"] == r["ans_ids"])) == r["D1_stored"] for r in rows), len(rows)]
    loc["n_nonfinite"] = sum(not (r["pre_finite"] and r["post_finite"]) for r in rows)
    st = V.jload(f"{V.TESTBED}/{name}/metrics.json")
    loc["stored_benchmark"] = {"T-Loc": st["TLoc_probes"], "I-Loc": st["ILoc_probes"], "nonfinite": st.get("loc_nonfinite")}
    loc["library_stored"] = ll.get("stored")
    loc["library_diff"] = ll.get("diff")
    loc["my_diff"] = {"T-Loc": loc["T-Loc*"]["mine"] - st["TLoc_probes"], "I-Loc": loc["I-Loc*"]["mine"] - st["ILoc_probes"]}
    bm = {}
    for nm, pt in (("T-Loc", "text"), ("I-Loc", "image")):
        sub = [r for r in ml["bench"] if r["ptype"] == pt]
        bm[nm] = {"n": len(sub), "mine_B1": 100.0 * float(np.mean([agree(r) for r in sub])), "stored_B8": st[f"{nm[0]}Loc_probes"],
                  "n_nonfinite": sum(not (r["pre_finite"] and r["post_finite"]) for r in sub)}
        bm[nm]["mine_minus_stored"] = bm[nm]["mine_B1"] - bm[nm]["stored_B8"]
    loc["benchmark_quantity_recomputed_at_B1"] = bm
    # ---------------------------------------------------------------- stage 5: the same on ALL valid probe rows (the pool of the library's own test b12)
    p = f"{odir}/my_loc_rest.json"
    if not os.path.exists(p):
        t1 = time.time()
        rest = [{"sid": f"probe:{r['ptype']}:{r['probe_idx']}", "ptype": r["ptype"], "q": {"fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"]}, "ans": r["y0_text"]}
                for r, a in zip(recs, ann) if r["y0_valid"] and not a["disjoint"]]
        for wname, Wt in (("pre", W0), ("post", WE)):
            at(Wt)
            for r in rest:
                pred, ans, fin = G.tf_forward(vlm, G.build(vlm, r["q"], r["ans"]))
                r[wname] = pred.tolist()
                r[wname + "_finite"] = fin
        H.reset()
        V.jdump({"rows": rest, "seconds": time.time() - t1}, p, indent=None)
    allrows = rows + V.jload(p)["rows"]
    b12 = V.jload(f"{V.LIB}/rescope/tests/out/gpu_{name}_b12_tokloc.json")["deployed_vs_stored"]["pool_all"]
    pa = {"n_rows": len(allrows)}
    for nm, pt in (("T-Loc", "text"), ("I-Loc", "image")):
        sub = [r for r in allrows if r["ptype"] == pt]
        mine_ = 100.0 * float(np.mean([agree(r) for r in sub]))
        pa[nm] = {"n": len(sub), "mine*": mine_, "library_test_b12*": b12[nm + "*"], "mine_minus_stored": mine_ - st[f"{nm[0]}Loc_probes"], "library_test_b12_diff": b12["diff"][nm]}
    loc["pool_all_valid_rows"] = pa
    res["locality"] = loc
    V.say(f"locality: {loc}")
    H.reset()
    res.update(seconds=time.time() - t0, peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, module_weight_is_WE_after=bool(torch.equal(H.mod.weight.data, H.WE)),
               rescope=os.path.dirname(rescope.__file__))
    V.jdump(res, f"{V.OUT}/c4_{tag}.json")
    V.say(f"DONE ({res['seconds']:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1])
