#!/usr/bin/env python
"""One gate-level and behavioural computation of the main comparison on a development log, from released files (GPU, about 3 minutes).

    gq run --mem 10500 -- python examples/gate_smoke.py [Q-A100-r3] [--n 20]        (tools/gpuq/gq, or any single GPU: python examples/gate_smoke.py)

Needs: the log (logs_vl_QA.tar, pools_vl.tar), its batch updates (tools/logs/rebuild_batch_deltas.py Q-A100-r3), the public checkpoint and
the VLKEB benchmark under $RESCOPE_EXT (README, "Checkpoints and benchmarks").  No key cache is needed: the keys of the queries are
computed here by the model.  It builds the thresholds-only repair of the paper from the log and 100 benign calibration queries of
the REPAIR pool (two-block rule, alpha = 0.05, draw 0), and evaluates n held-out probe rows under the deployed log, the pre-edit
model and the repair: whether the gate fires, and whether the pre-edit answer is reproduced (D1).  The deployed rows are compared
with the flags stored in the released log (probe_records_v2.jsonl).  The registered experiment is experiments/e1_main (all rows,
20 draws, from the key cache); this script is a small end-to-end check of the released stack, not a number of the paper.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rescope_paths as RP  # noqa: E402

sys.path.insert(0, os.path.join(RP.ROOT, "02_research", "code"))
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("log", nargs="?", default="Q-A100-r3")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--out", default="")
a = ap.parse_args()
if "GQ_MEM" in os.environ:                                   # under the queue: cap the process (tools/gpuq/README.txt)
    cap = (int(os.environ["GQ_MEM"]) - 600) * 2**20
    torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)

from rescope import *  # noqa: E402,F401,F403

name = RP.working_tree_name(a.log)
H = load(name, device="cuda:0")                              # harness handle: public checkpoint + released log; exact W_0 / W_E
ev = Evaluator(H)
log = from_harness(H).to(H.dev)
S = feature_gate(log, "ncoord", W0=H.W0)                     # ReScope: normalised coordinates on the read-out vectors
S, _ = add_guards(S, *ev.own_keys("rel"))
benign = splits.sample(ev.probe_items("repair"), 100, 0)     # benign calibration queries: a seeded draw from the REPAIR pool
S, rec = calibrate_two_block(S, [ev.eligible_keys(it, S) for it in benign], 0.05, seed=0)    # the default calibration (rescope/docs/USAGE.md)
items = ev.probe_items("heldout_disjoint")[: a.n]
rows = {k: ev.tf_rows(m, items) for k, m in (("deployed", None), ("pre-edit", H.W0), ("rescope", S))}
with open(os.path.join(RP.log_dir(a.log), "probe_records_v2.jsonl")) as f:
    stored = {(r["probe_idx"], r["ptype"]): r for r in map(json.loads, f)}
same = sum(bool(r["D1"]) == bool(stored[(it["probe_idx"], it["ptype"])]["is_complaint_D1"]) for it, r in zip(items, rows["deployed"]))
out = {"log": a.log, "internal_name": name, "n_rows": len(items), "checks": H.checks if hasattr(H, "checks") else None,
       "calibration": {k: rec.get(k) for k in ("op", "alpha", "eta", "n_edits_capped", "bound_false_firing", "guarantee") if k in rec},
       "D1_changed_answers": {k: sum(bool(r["D1"]) for r in v) for k, v in rows.items()},
       "fired": sum(bool(r.get("fired")) for r in rows["rescope"]),
       "deployed_D1_equal_to_stored_flags": f"{same}/{len(items)}"}
print(json.dumps(out, indent=1, default=str))
if a.out:
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
