"""Extra GPU check (not part of the fingerprinted suite): one log of every registry family loads through rescope.load and
evaluates at B = 1, with ONE model in memory (load(reuse=...)).

    $GQ run --mem 10500 -- python tests/extra/load_families.py [--mem_gb 9.8]

Families (Qwen2.5-VL-3B, the backbone on which every family exists): the scale registry (long VLKEB log, MEMIT-style log,
MMKE-Bench log), the part-2 registry (public-default MEMIT family; the longest log that is complete now) and the text
registry (CounterFact).  Per log: registry source, lean / standard loader, harness load-time checks, seconds, peak RAM; on a
seeded sample: the deployed log's B = 1 D1 flags against the stored probe records, the pre-edit weight (D1 must be 0), Rel at
B = 1 next to the stored vendor-pass flags, the default gate (two-block rule, own Rel prompts as guards), token locality of
the deployed log next to the stored benchmark locality; text log: hard negatives from the text registry's directory.
Pre-specified criterion: PASS iff for EVERY log the harness checks hold, the deployed D1 flags equal the stored ones on all
sampled rows, the pre-edit weight gives D1 = 0, every guarded own Rel prompt fires its edit, and the peak resident memory of
the process stays below 40 GB.  Writes tests/out/extra_load_families.json (stamped with the package fingerprint).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # .../02_research/code
sys.path.insert(0, os.path.dirname(HERE))                                            # tests/ (run_tests: fingerprint)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
import run_tests  # noqa: E402
from rescope import baselines as BL, calibrate as CA, evaluate as EV, loglib as LL, repair as RP, splits as SP  # noqa: E402

WANT = [("scale", "TB-Q2000-ae-l2_3"), ("scale", "TB-Q500-memit"), ("scale", "TB-Q500-mmke-l2_3"),
        ("part2", ["TB-Q2000-memitw", "TB-Q1000-memitw", "TB-Q500-memitw", "TB-Q200-memitw", "TB-Q100-memitw"]),   # the longest complete one
        ("text", "TB-Q500-cf-l2_3"), ("testbed", "TB-Q100-ae-l2_3")]


def one(name, H, device, mem_gb, reg, n=12):
    out = {"name": name, "source": reg["logs"][name]["source"]}
    t0 = time.time()
    H = EV.load(name, device=device, mem_gb=mem_gb, reuse=H)
    out.update(T=H.T, lean=H.load_info["lean"], seconds_load=H.load_info["seconds"], reused_model_of=H.load_info["reused_model_of"],
               checks=dict(H.checks), checks_ok=all(v for k, v in H.checks.items() if k != "collapsed_working_point"),
               peak_rss_gb_after_load=EV.peak_rss_gb(), n_probe_rows=len(H.rows), ptypes=sorted({r["ptype"] for r in H.rows}))
    hn_dir = None
    if out["source"] == "text":
        import text_registry
        hn_dir = text_registry.HN_DIR
    ev = EV.Evaluator(H, hn_dir=hn_dir)
    sm = EV.stored_metrics(name)
    rng = np.random.default_rng(15)
    pr = ev.probe_items("heldout_disjoint")
    sample = [pr[int(i)] for i in sorted(rng.choice(len(pr), min(n, len(pr)), replace=False))]
    rows = ev.tf_rows({"deployed": None, "pre-edit": H.W0}, sample)
    eds = [int(i) for i in sorted(rng.choice(H.T, n, replace=False))]
    er = ev.edits(None, ("rel",), eds)
    out["sample"] = {"n_probe": len(sample), "deployed_D1_equals_stored": sum(r["D1"] == r["D1_WE"] for r in rows["deployed"]),
                     "deployed_D1": sum(r["D1"] for r in rows["deployed"]), "pre_edit_D1": sum(r["D1"] for r in rows["pre-edit"]),
                     "n_edits": len(eds), "rel_B1_exact": sum(r["exact"] for r in er),
                     "rel_stored_exact_same_edits": sum(bool(sm["per_edit"][i]["rel_exact"]) for i in eds),
                     "rel_B1_flags_equal_stored_vendor_pass": sum(r["exact"] == bool(sm["per_edit"][r["edit"]]["rel_exact"]) for r in er)}
    loc = EV.locality(rows["deployed"])
    out["stored"] = {"Rel_exact": sm["edits"]["rel"]["exact"], "T-Loc": sm["locality"]["T-Loc"], "I-Loc": sm["locality"]["I-Loc"],
                     "D1_rate_heldout_disjoint": sm["probes"]["heldout_disjoint"]["D1_rate"],
                     "sample_T-Loc*": loc["T-Loc*"], "sample_I-Loc*": loc["I-Loc*"], "sample_n_text": loc["n_text"], "sample_n_image": loc["n_image"]}
    log = LL.from_harness(H)
    S = BL.feature_gate(log, "ncoord", W0=H.W0)
    S, _ = RP.add_guards(S, *ev.own_keys("rel", edits=eds))
    pool = ev.probe_items("repair")
    benign = SP.sample(pool, min(60, len(pool)), 0)
    S, rec = CA.calibrate_two_block(S, [ev.eligible_keys(it, S) for it in benign], 0.05, seed=0)
    gr = ev.edits(S, ("rel", "tgen"), eds)
    gp = ev.tf_rows(S, sample)
    out["gate"] = {"eta": rec["eta"], "n2": rec["n2"], "n_edits_capped": rec["n_edits_capped"], "n_rel": len(eds),
                   "rel_own_fired_last": sum(bool(r["own_fired_last"]) for r in gr if r["metric"] == "rel"),
                   "rel_exact": sum(r["exact"] for r in gr if r["metric"] == "rel"), "tgen_exact": sum(r["exact"] for r in gr if r["metric"] == "tgen"),
                   "tgen_own_fired_last": sum(bool(r["own_fired_last"]) for r in gr if r["metric"] == "tgen"),
                   "probe_fired": sum(bool(r["fired"]) for r in gp), "probe_D1": sum(r["D1"] for r in gp), "n_probe": len(gp)}
    if ev._hn_error is None:
        hn = ev.hn_items()
        hs = [hn[int(i)] for i in sorted(rng.choice(len(hn), min(n, len(hn)), replace=False))]
        hr = ev.tf_rows({"pre-edit": H.W0, "gate": S}, hs)
        out["hard_negatives"] = {"dir": ev.hn_dir, "n_valid_rows": len(hn), "types": sorted({h["hn_type"] for h in hn}), "n_sample": len(hs),
                                 "pre_edit_D1": sum(r["D1"] for r in hr["pre-edit"]), "gate_fired": sum(bool(r["fired"]) for r in hr["gate"]),
                                 "gate_D1": sum(r["D1"] for r in hr["gate"])}
    else:
        out["hard_negatives"] = {"dir": ev.hn_dir, "note": f"{type(ev._hn_error).__name__}: {ev._hn_error}"[:200]}
    out["seconds"] = time.time() - t0
    out["peak_rss_gb"] = EV.peak_rss_gb()
    out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    s, g = out["sample"], out["gate"]
    out["ok"] = bool(out["checks_ok"] and s["deployed_D1_equals_stored"] == s["n_probe"] and s["pre_edit_D1"] == 0
                     and g["rel_own_fired_last"] == g["n_rel"] and out["peak_rss_gb"] < 40.0
                     and ("pre_edit_D1" not in out["hard_negatives"] or out["hard_negatives"]["pre_edit_D1"] == 0))
    del ev, S, log
    gc.collect()
    torch.cuda.empty_cache()
    return H, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_gb", type=float, default=9.8)
    a = ap.parse_args()
    path = os.path.join(run_tests.OUT, "extra_load_families.json")
    res = {"test": "extra_load_families", "code": run_tests.CODE, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "logs": [], "error": None}
    try:
        reg = EV.register_logs()
        res["registry_sources"] = {k: {x: y for x, y in v.items() if x != "names"} for k, v in reg["sources"].items()}
        H = None
        for src, names in WANT:
            name = next((n for n in ([names] if isinstance(names, str) else names) if reg["logs"].get(n, {}).get("complete")), None)
            if name is None:
                res["logs"].append({"source": src, "name": None, "ok": False, "note": f"no complete log among {names}"})
                continue
            H, out = one(name, H, a.device, a.mem_gb, reg)
            res["logs"].append(out)
            run_tests.say(f"{name}: " + json.dumps({k: v for k, v in out.items() if k not in ("checks",)}, default=str)[:900])
        res["pass"] = bool(res["logs"]) and all(x["ok"] for x in res["logs"])
    except Exception:
        res["pass"], res["error"] = False, traceback.format_exc()
        print(res["error"], flush=True)
    res.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), peak_rss_gb=EV.peak_rss_gb())
    run_tests.jdump(res, path)
    run_tests.say(f"extra_load_families: {'PASS' if res['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
