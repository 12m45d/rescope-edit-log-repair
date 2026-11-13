"""The ONE gate of the R3 family that enters the behavioural rows (R19 f / R21): the variant with the largest paraphrase coverage at
gate level on the DEVELOPMENT log (mean over the 20 calibration draws; two-block calibration, guards = the frozen default).  Ties:
the lower false firing on HELD-OUT prompts, then the table order.  Written to out/best_gate_text.json BEFORE any behavioural run on
a reported log; refuses to overwrite."""
import os
import sys
import time

import numpy as np

import xcommon as X

R3 = ("cos_centred", "cos_null", "white_0.01", "white_0.1", "white_1", "whitecos_0.01", "whitecos_0.1", "whitecos_1")
p = X.OUT + "/best_gate_text.json"
if os.path.exists(p):
    print("exists:", X.jload(p)["selected"])
    sys.exit(0)
if os.path.isdir(X.OUT + "/behave"):
    raise SystemExit("a behavioural run exists: the selection must precede it")
g = X.jload(X.OUT + f"/gate/{X.DEV}.json")
cand = {}
for lab in R3 + ("cosine", "rescope"):
    pd = g["rows"][lab]["gon"]["per_draw"]
    cand[lab] = {"coverage_paraphrase": float(np.mean([pd[d]["paraphrase"]["own_released"] for d in pd])),
                 "false_firing_heldout": float(np.mean([pd[d]["heldout"]["fired"] for d in pd])),
                 "firing_neighborhood": float(np.mean([pd[d]["neighborhood"]["fired"] for d in pd])),
                 "n_draws": len(pd), "capped_mean": float(np.mean([g["rows"][lab]["gon"]["cal"][d]["n_edits_capped"] for d in pd]))}
order = sorted(R3, key=lambda x: (-cand[x]["coverage_paraphrase"], cand[x]["false_firing_heldout"], R3.index(x)))
X.jdump({"selected": order[0], "rule": __doc__, "log": X.DEV, "ranking": order, "candidates": {k: cand[k] for k in R3},
         "for_reference_not_candidates": {k: cand[k] for k in ("cosine", "rescope")}, "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, p)
print("selected", order[0], {k: round(cand[k]["coverage_paraphrase"], 2) for k in order})
