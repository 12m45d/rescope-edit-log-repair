"""Information, not a pass / fail test: is the harness REEDIT correction reproducible?

Observed while running test b6 twice: two processes built the REEDIT correction of the same 10 complaints and the
corrections differed in one Rel flag (Qwen) / one HELD-OUT flag (LLaVA), although the z* optimisation is seeded.
This script measures it directly.  In ONE process it builds the correction twice, with the two steps of
rescope.evaluate.reedit (z* per complaint at B = 1 with chunk=1, then one joint install), and compares
  the z* targets of the two builds (per complaint: bitwise equal? largest |z_1 - z_0| / |z_0 - h_0|),
  the two correction weights (bitwise equal? entries that differ, ||W_1 - W_0||_F / ||W_0 - W_E||_F),
  the Rel exact flags (all edits) and the HELD-OUT D1 flags under the two corrections (Evaluator, B = 1),
and puts the rates of this process next to those of the b6 run of the test-suite (another process).
Writes tests/out/extra_reedit_repro_<log>.json.   usage: reedit_repro.py --log TB-Q100-ae-l2_3 --mem_gb 11
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "out")
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))        # .../02_research/code

import torch  # noqa: E402

torch.set_num_threads(4)
from rescope import evaluate as EV  # noqa: E402


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build(hz, H, cs):
    """The two steps of evaluate.reedit, keeping the targets."""
    tg = []
    for c in cs:
        one, _ = hz.pkg.repair("REEDIT", H, [c], variant="k0+kp", return_targets=True, chunk=1)
        tg.append(one.meta.pop("_targets")[0])
    corr, _ = hz.pkg.repair("REEDIT", H, cs, variant="k0+kp", targets=tg, chunk=1)
    return corr, tg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="TB-Q100-ae-l2_3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_gb", type=float, default=11.0)
    a = ap.parse_args()
    t0 = time.time()
    H = EV.load(a.log, device=a.device, mem_gb=a.mem_gb)
    hz = EV.harness()
    ev = EV.Evaluator(H, hard_negatives=False)
    cs = hz.pkg.sample_complaints(H, n=10, seed=0)                       # the complaints of test b6
    builds = []
    for k in range(2):
        corr, tg = build(hz, H, cs)
        builds.append((corr, tg))
        say(f"build {k}: installed {len(corr.meta.get('installed_rows', []))}/{len(cs)}, oracle {[bool(t.oracle) for t in tg]}")
    for key in [key for key in H._setup if isinstance(key, tuple) and key[:1] == ("reedit",)]:
        H._setup.pop(key)                                                # the solve state is not needed any more
    gc.collect()
    torch.cuda.empty_cache()
    (c0, t0s), (c1, t1s) = builds
    rel = [float((y.z.double() - x.z.double()).norm() / (x.z.double() - x.h.double()).norm()) for x, y in zip(t0s, t1s)]
    W0c, W1c, WE = c0.W.double(), c1.W.double(), H.WE.double()
    out = {"log": a.log, "n_complaints": len(cs), "complaints": [c["group"] for c in cs],
           "targets": {"n_bitwise_equal": sum(bool(torch.equal(x.z, y.z)) for x, y in zip(t0s, t1s)), "n": len(cs),
                       "max_rel_diff_z": max(rel), "rel_diff_z": rel, "h_bitwise_equal": sum(bool(torch.equal(x.h, y.h)) for x, y in zip(t0s, t1s)),
                       "steps": [[int(t.steps_used) for t in t0s], [int(t.steps_used) for t in t1s]],
                       "oracle": [[bool(t.oracle) for t in t0s], [bool(t.oracle) for t in t1s]]},
           "weight": {"bitwise_equal": bool(torch.equal(c0.W, c1.W)), "n_entries_differ": int((c0.W != c1.W).sum()), "n_entries": int(c0.W.numel()),
                      "rel_fro_diff": float((W1c - W0c).norm() / (W0c - WE).norm())}}
    say("targets", {k: v for k, v in out["targets"].items() if k != "rel_diff_z"}, "weight", out["weight"])
    del W0c, W1c, WE
    r = ev.edits({"b0": c0.W, "b1": c1.W}, ("rel",))
    p = ev.probes({"b0": c0.W, "b1": c1.W}, pool="heldout")
    out["flags"] = {"rel": {"n": len(r["b0"]), "exact": [sum(x["exact"] for x in r[b]) for b in ("b0", "b1")],
                            "n_differ": sum(x["exact"] != y["exact"] for x, y in zip(r["b0"], r["b1"]))},
                    "heldout": {"n": len(p["b0"]), "D1": [sum(x["D1"] for x in p[b]) for b in ("b0", "b1")],
                                "n_differ": sum(x["D1"] != y["D1"] for x, y in zip(p["b0"], p["b1"]))}}
    out["this_process"] = {"rel_exact_pct": [100.0 * v / len(r["b0"]) for v in out["flags"]["rel"]["exact"]],
                           "heldout_D1_pct": [100.0 * v / len(p["b0"]) for v in out["flags"]["heldout"]["D1"]]}
    b6 = os.path.join(OUT, f"gpu_{a.log}_b6_harness.json")
    if os.path.exists(b6):
        with open(b6) as f:
            m = json.load(f)["methods"]["REEDIT"]["corr"]
        out["suite_b6_other_process"] = {"rel_exact_pct": m["rel_exact"], "heldout_D1_pct": m["heldout_D1_rate"]}
    H.reset()
    out["seconds"] = time.time() - t0
    out["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = os.path.join(OUT, f"extra_reedit_repro_{a.log}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(OUT, f"extra_reedit_repro_{a.log}.json"))
    say("DONE", json.dumps({k: v for k, v in out.items() if k not in ("targets", "complaints")}))


if __name__ == "__main__":
    main()
