"""Test of mme_pool.py (CPU; no model; reads the published captures, writes test_mme_pool.json next to this file).

    python -I -B test_mme_pool.py [qwen25vl] [llava]

A  The reader on the PUBLISHED capture of each backbone (reading only): md5-checked KeyCache, parts against the selection,
   images never in two parts, valid rows have a teacher-forced key, key tensors by `which`, item form (= the form of the caches'
   capture code for a benign row), content lookup (what rescope.SparseEvaluator uses to find an item in a key cache).
B  The two backbones hold the same rows and the same parts (the split is by image; it does not depend on the backbone).
C  The ten-line example of the module docstring.  Its lines 6 and 7 hand the rows to the library's gate functions
   (KeyCache.coords / KeyCache.decide).  THIS TEST DOES NOT RUN A GATE ON MME ROWS (the capture task forbids it): the example
   is executed on a SYNTHETIC copy -- the first 240 index rows of the capture with RANDOM keys of width 96 (rescope.write_cache
   into work/test_pool/<backbone>/) -- and a SYNTHETIC detector set (rescope.synthetic_log: random factors, no edit log of the
   project).  No key of the capture and no logged edit enters these lines; what is tested is that a directory in the MME
   layout is accepted by the library's calibration-score and decision functions, with the row selections of the glue.
   Lines 8 and 9 of the example need a model: the same calls (SparseEvaluator(H, cache=P.kc, ...), tf_rows(<method>,
   P.items(rows))) are run with the PRE-EDIT weight as the method in the GPU verification (<backbone>/verify.json, 'glue').
   With a gate as the method they have NOT been run on MME rows by this package (that is the experiment, not the capture).
"""
from __future__ import annotations

import inspect
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
sys.path.insert(0, HERE)
import mme_common as M  # noqa: E402
import mme_pool  # noqa: E402

N_SYN, D_SYN = 240, 96


def test_reader(bb, sel) -> dict:
    import torch
    import fc_capture as CAP
    P = mme_pool.load(bb)                                               # verify=True: md5 of index and key files, completeness, parts
    ix, kc = P.index, P.kc
    allr = P.rows()
    c, t = P.rows("MME-C"), P.rows("MME-T")
    vc, vt = P.rows("MME-C", valid=True), P.rows("MME-T", valid=True)
    img = {p: {ix[j]["uid"] for j in P.rows(p)} for p in mme_pool.PARTS}
    tfr = [j for j in allr if ix[j]["row_tf"] is not None]
    res = {
        "n_rows": kc.n, "layer": kc.layer, "d": kc.d,
        "rows_all_is_range": allr == list(range(kc.n)),
        "parts_partition_the_rows": sorted(c + t) == allr and not set(c) & set(t),
        "parts_tuple_equals_all": P.rows(("MME-C", "MME-T")) == allr,
        "parts_equal_selection": [P.part(e["sid"]) for e in sel] == [e["part"] for e in sel] and [ix[j]["sid"] for j in allr] == [e["sid"] for e in sel],
        "sids_of_part": P.sids("MME-C") == [e["sid"] for e in sel if e["part"] == "MME-C"],
        "n_images_in_both_parts": len(img["MME-C"] & img["MME-T"]),
        "by_part": {p: {"rows": len(P.rows(p)), "images": len(img[p]), "valid": len(P.rows(p, valid=True))} for p in mme_pool.PARTS},
        "valid_rows_have_tf_key": all(ix[j]["row_tf"] is not None for j in vc + vt),
        "valid_equals_keycache_rows": sorted(vc + vt) == kc.rows(kind="benign", pool="benign", valid=True),
        "category_filter": sorted(j for cat in {e["category"] for e in ix} for j in P.rows(category=cat)) == allr,
        "where_filter": P.rows("MME-T", where=lambda e: e["y0_yn"] == "yes") == [j for j in t if ix[j]["y0_yn"] == "yes"],
        "field": P.field(t[:5], "uid") == [ix[j]["uid"] for j in t[:5]],
        "summary": P.summary,
    }
    # keys by `which`
    kd = kc.keys("decode", c[:16])
    ks = kc.keys("steps", c[:16])
    ktf = kc.keys("tf", vc[:16])
    res["keys"] = {"decode_shape": list(kd.shape), "steps_shape": list(ks.shape), "tf_shape": list(ktf.shape),
                   "step0_equals_decode": bool(torch.equal(ks[:, 0], kd)), "dtype": str(kd.dtype),
                   "prompt_equals_tf_where_no_shift": bool(torch.equal(kc.keys("prompt", [j for j in vc[:16] if not ix[j]["lp_shift"]]),
                                                                       kc.keys("tf", [j for j in vc[:16] if not ix[j]["lp_shift"]]))),
                   "having": {w: len(kc.having(w)) for w in ("tf", "decode", "steps", "all")}}
    # items
    its = P.items(tfr)
    def same_item(it, j):                                               # the capture code's item of the row; the glue adds two fields for generate
        ref = CAP.make_item(dict(ix[j], lib_kind="benign"))
        return (all(it["_row"][k] == v for k, v in ref["_row"].items()) and set(it["_row"]) - set(ref["_row"]) == {"long_text", "edit_alt"}
                and it["_row"]["edit_alt"] is None and it["_row"]["long_text"] == ix[j]["long_text"] and it["kind"] == ref["kind"] == "benign"
                and it["sid"] == ref["sid"] and it["y0_valid"] == bool(ix[j]["y0_valid"]) and it["part"] == ix[j]["part"])
    same = all(same_item(it, j) for it, j in zip(its, tfr))
    try:                                                                # P.rows(part, kc) by analogy with the benign parts' reader must not pass silently
        P.rows("MME-T", kc)
        positional_refused = False
    except TypeError:
        positional_refused = True
    found = sum(kc.find(ix[j]["fmt"], it["_row"]["prompt"], it["_row"]["image"], it["_row"]["y0_text"]) == j for it, j in zip(its, tfr))
    no_tf = [j for j in allr if ix[j]["row_tf"] is None]
    try:                                                                # a row without a teacher-forced sequence must be refused
        mme_pool.item_of(dict(ix[0], row_tf=None))
        refused = False
    except ValueError:
        refused = True
    res["items"] = {"n": len(its), "equal_to_the_capture_code_item": same, "n_found_by_content_in_the_key_cache": found,
                    "n_rows_without_tf": len(no_tf), "row_without_tf_is_refused": refused, "rows_second_positional_argument_is_refused": positional_refused,
                    "images_exist": all(os.path.exists(it["_row"]["image"]) for it in its[:50])}
    res["ok"] = bool(res["rows_all_is_range"] and res["parts_partition_the_rows"] and res["parts_tuple_equals_all"] and res["parts_equal_selection"]
                     and res["sids_of_part"] and res["n_images_in_both_parts"] == 0 and res["valid_rows_have_tf_key"] and res["valid_equals_keycache_rows"]
                     and res["category_filter"] and res["where_filter"] and res["field"] and res["keys"]["step0_equals_decode"]
                     and res["keys"]["prompt_equals_tf_where_no_shift"] and res["keys"]["decode_shape"] == [16, kc.d] and res["keys"]["steps_shape"] == [16, 11, kc.d]
                     and same and found == len(its) and refused and positional_refused and res["items"]["images_exist"])
    return res, P


def synthetic_copy(P, bb, root) -> str:
    """work/test_pool/<bb>/: the first N_SYN index rows of the capture with RANDOM float32 keys of width D_SYN (row addresses renumbered)."""
    import torch
    import rescope
    new, n_tf, n_st = [], 0, 0
    for j, e in enumerate(P.index[:N_SYN]):
        e = dict(e)
        e["row_decode"] = j
        if e["row_tf"] is not None:
            e["row_tf"] = n_tf
            n_tf += 1
        base = min(e["rows_decode_steps"])
        e["rows_decode_steps"] = [n_st + (x - base) for x in e["rows_decode_steps"]]
        n_st += e["n_decode_distinct"]
        new.append(e)
    k = n_tf
    for e in new:
        if e["row_tf_prompt"] is not None:
            e["row_tf_prompt"] = k
            k += 1
    g = torch.Generator().manual_seed(M.useed("test_pool", bb))
    K_st = torch.randn(n_st, D_SYN, generator=g)
    K_tf = torch.randn(k, D_SYN, generator=g)
    K_dec = K_st[[e["rows_decode_steps"][0] for e in new]].clone()
    d = os.path.join(root, bb)
    if os.path.isdir(d):
        shutil.rmtree(d)
    rescope.write_cache(d, new, K_tf, K_dec, K_st, meta={"backbone": bb, "layer": mme_pool.LAYER[bb], "complete": True,
                                                         "synthetic": "RANDOM keys (test of the file layout); not a capture"})
    return d


def test_example(P_real, bb) -> dict:
    """The example of the module docstring on the synthetic copy with a synthetic detector set (see the module docstring, C)."""
    import numpy as np
    import torch
    import rescope
    root = os.path.join(M.WORK, "test_pool")
    synthetic_copy(P_real, bb, root)
    log, _ = rescope.synthetic_log(T=12, d=D_SYN, m=20, batch=4, seed=3)
    P0 = mme_pool.load(bb, root=root)
    r0 = rescope.ScopedLog.from_log(log, role="last", tau=1.0).scores(P0.kc.keys("decode")).max(1).values
    tau = float(r0.median())                                            # random factors; a threshold at the median of the synthetic scores, so
    gate = rescope.ScopedLog.from_log(log, role="last", tau=tau)        # that about half of the synthetic rows are released and half are not
    doc = mme_pool.__doc__.split("TEN-LINE EXAMPLE")[1].split("\n\n")[0].splitlines()[1:]
    lines = [ln.strip() for ln in doc if ln.strip()]
    assert len(lines) == 10, len(lines)

    class Shim:                                                         # mme_pool.load bound to the synthetic directory
        @staticmethod
        def load(backbone):
            return mme_pool.load(backbone, root=root)
    ns = {"mme_pool": Shim, "gate": gate, "rescope": rescope}
    ran, skipped, ran_imports, imp = [], [], [], {}
    for n, ln in enumerate(lines, 1):
        code = ln.replace('"qwen25vl"', repr(bb))
        compile(code, f"<example line {n}>", "exec")                    # every line is valid Python
        if n in (1, 2):                                                 # the imports: executed in a namespace of their own (the example's
            exec(code, imp)                                             # mme_pool would read the real capture; below it is bound to the copy)
            ran_imports.append(n)
        elif n in (8, 9):
            skipped.append([n, "needs a model: run with the pre-edit weight as the method in the GPU verification (verify.json, glue_mme_pool)"])
        else:
            exec(code, ns)
            ran.append(n)
    P, cal, test, coords, fire, units = (ns[k] for k in ("P", "cal", "test", "coords", "fire", "units"))
    ref = gate.scores(P.kc.keys("tf", cal))
    th = gate.theta()
    exp_fired = (gate.scores(P.kc.keys("tf", test)) >= th[None, :]).any(1).numpy()
    other = {w: {"coords_shape": list(P.kc.coords(gate, P.rows("MME-C"), which=w).shape),
                 "n_fired": int(P.kc.decide(gate, w, P.rows("MME-T"))["fired"].sum())} for w in ("decode", "steps")}
    allk = P.kc.coords(gate, cal, which="all")
    res = {"synthetic_dir": os.path.relpath(os.path.join(root, bb), HERE), "n_rows": P.kc.n, "key_width": P.kc.d, "n_detectors": gate.D,
           "lines_executed": ran, "lines_not_executed": skipped, "import_lines_executed": ran_imports,
           "imports_give_the_modules": imp.get("mme_pool") is mme_pool and imp.get("rescope") is rescope, "n_cal": len(cal), "n_test": len(test),
           "coords_shape": list(coords.shape), "coords_equal_gate_scores": bool(torch.allclose(coords, ref, rtol=1e-12, atol=0)),
           "decide_fields": sorted(fire), "decide_fired_equals_scores_vs_thresholds": bool(np.array_equal(fire["fired"], exp_fired)),
           "decide_rows_are_the_test_rows": list(fire["rows"]) == test, "n_fired_synthetic": int(fire["fired"].sum()),
           "n_not_fired_synthetic": int((~fire["fired"]).sum()), "synthetic_threshold": tau,
           "units_are_images": units == [P.index[j]["uid"] for j in test], "which_decode_steps": other,
           "which_all_at_least_tf": bool((allk >= coords - 1e-12).all()),
           "signatures": {"mme_pool.load": str(inspect.signature(mme_pool.load)), "MMEPool.rows": str(inspect.signature(mme_pool.MMEPool.rows)),
                          "MMEPool.items": str(inspect.signature(mme_pool.MMEPool.items)), "KeyCache.coords": str(inspect.signature(rescope.KeyCache.coords)),
                          "KeyCache.decide": str(inspect.signature(rescope.KeyCache.decide)),
                          "SparseEvaluator.__init__": str(inspect.signature(rescope.SparseEvaluator.__init__)),
                          "Evaluator.tf_rows": str(inspect.signature(rescope.Evaluator.tf_rows))}}
    res["ok"] = bool(res["coords_shape"] == [len(cal), gate.D] and res["coords_equal_gate_scores"] and res["decide_fired_equals_scores_vs_thresholds"]
                     and res["decide_rows_are_the_test_rows"] and res["units_are_images"] and res["which_all_at_least_tf"]
                     and ran == [3, 4, 5, 6, 7, 10] and ran_imports == [1, 2] and res["imports_give_the_modules"]
                     and len(cal) > 0 and len(test) > 0 and 0 < res["n_fired_synthetic"] < len(test)
                     and all(k in fire for k in ("fired", "top_edit", "near", "margin", "n_cand")))
    return res


def main():
    C = M.cache_code()
    if C.CODE not in sys.path:
        sys.path.insert(0, C.CODE)
    import torch
    torch.set_num_threads(4)
    bbs = sys.argv[1:] or list(mme_pool.BACKBONES)
    sel = M.jsonl_read(os.path.join(M.SELECT, "selection.jsonl"))
    out = {"what": "test of mme_pool.py: reader on the published captures (A), same rows and parts for both backbones (B), the ten-line example on a "
                   "synthetic copy with random keys and a synthetic detector set (C); no gate is run on MME rows",
           "backbones": {}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    pools = {}
    for bb in bbs:
        a, P = test_reader(bb, sel)
        pools[bb] = P
        out["backbones"][bb] = {"A_reader": a, "C_example": test_example(P, bb),
                                "capture": {"index_md5": P.meta["index_md5"], "keys_decode_md5": P.meta["keys_decode"]["md5"]}}
    if len(pools) > 1:
        a, b = (pools[k].index for k in list(pools)[:2])
        out["B_same_rows_for_both_backbones"] = {"sids": [e["sid"] for e in a] == [e["sid"] for e in b], "parts": [e["part"] for e in a] == [e["part"] for e in b],
                                                 "queries": [(e["prompt"], e["image"], e["category"]) for e in a] == [(e["prompt"], e["image"], e["category"]) for e in b]}
        out["B_same_rows_for_both_backbones"]["ok"] = all(out["B_same_rows_for_both_backbones"].values())
    out["ok"] = all(v["A_reader"]["ok"] and v["C_example"]["ok"] for v in out["backbones"].values()) and out.get("B_same_rows_for_both_backbones", {"ok": True})["ok"]
    M.jdump(out, os.path.join(HERE, "test_mme_pool.json"))
    for bb, v in out["backbones"].items():
        print(f"{bb}: A reader ok = {v['A_reader']['ok']} ({v['A_reader']['by_part']}); C example ok = {v['C_example']['ok']} "
              f"(lines {v['C_example']['lines_executed']} executed on the synthetic copy)")
    print("B same rows for both backbones:", out.get("B_same_rows_for_both_backbones"))
    print("ALL OK" if out["ok"] else "FAILED")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
