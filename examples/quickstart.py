#!/usr/bin/env python
"""QUICK START, steps 2 to 4 of the README (CPU, about one minute, no checkpoint, no benchmark).

    python examples/quickstart.py [paper name of a log, default Q-A100-r3]

  2. load one released log through the library (rescope.loglib.from_testbed)
  3. check that its deployed update equals the sum of its recorded terms, up to the round-off of the low-precision write that
     the log's metadata records
  4. rebuild the thresholds of the repair from the log alone (log-only calibration, q = 0.999) and print them
Step 1 (checksums) and step 5 (regenerate one table of the paper) are shell commands: see the README.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rescope_paths as RP  # noqa: E402

sys.path.insert(0, os.path.join(RP.ROOT, "02_research", "code"))
import torch  # noqa: E402
from rescope import loglib  # noqa: E402
from rescope.scoped import ScopedLog  # noqa: E402
from rescope.calibrate import calibrate_log_only  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "Q-A100-r3"
d = RP.log_dir(name)
if not os.path.exists(os.path.join(d, "edit_factors.pt")):
    raise SystemExit(f"{name}: not found under {RP.LOGS} (unpack the release archives there; `python rescope_paths.py` prints what is present)")

# 2. the log
log = loglib.from_testbed(d)
print(f"[2] {name} (internal name {RP.working_tree_name(name)}): T = {log.T} edits, write vectors U {tuple(log.U.shape)}, read-out vectors W {tuple(log.W.shape)}, "
      f"solver keys K {tuple(log.K.shape)}, {int(log.batch.max()) + 1} batches")

# 3. deployed update against the sum of the recorded rank-one terms
comp = torch.load(os.path.join(d, "composite_delta.pt"), map_location="cpu").double()        # float(W_E) - float(W_0)
terms = log.delta()                                                                          # sum_i outer(U_i, W_i), float64
dev = float((comp - terms).norm() / terms.norm())
with open(os.path.join(d, "metrics.json")) as f:
    met = json.load(f)
rec = None if met.get("prefix") else (met.get("build") or {}).get("composite_rel_dev_solver_factors_vs_applied")   # (a prefix log carries its parent's record)
worst = max(b["bf16_write_rel_dev"] for b in met["per_batch"])
print(f"[3] ||deployed update - sum of terms|| / ||sum of terms|| = {dev:.6f}"
      + (f"   (recorded when the log was written: {rec:.6f})" if rec is not None else "")
      + f"; round-off of the low-precision write, largest batch: {worst:.6f}")
# the recorded value was computed in float32 when the log was written, this one in float64: they agree to about three digits
ok3 = (abs(dev - rec) <= 0.01 * rec) if rec is not None else dev <= 0.05
print("    -> the deployed update is the sum of the recorded terms up to the recorded write round-off "
      "(criterion: within 1 percent of the recorded deviation; without a recorded value: at most 5 percent of the update's norm):", ok3)

# 4. thresholds of the repair from the log alone
S = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
S, r = calibrate_log_only(S, q=0.999)
tau = S.det_tau.flatten()
print(f"[4] log-only thresholds (q = 0.999): theta = {r['theta']:.6f} = the 0.999 quantile of the {r['n_pooled']} normalised cross-talk values of the log's own solver keys; "
      f"{tau.numel()} thresholds, min {float(tau.min()):.6f}, max {float(tau.max()):.6f}; own solver keys silenced: {r['n_own_keys_silenced']}; "
      f"pairs (key of edit j, detector of edit i != j) that still fire: {r['log_cross_firing_pairs']}")
raise SystemExit(0 if ok3 else 1)
