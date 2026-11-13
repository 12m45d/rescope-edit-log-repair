"""P12 verifier -- last CPU readings. -> out/cpu_final.json
  1. z* reproducibility, restricted to rows where BOTH targets come from the first (vendor) attempt; and the oracle
     flips documented in the experiment's own files (same code path, different run).
  2. near-twins: valid HELD-OUT image rows that share (question, entity directory of the image) with a REPAIR-pool row
     without being an identical query; their share among the held-out complaints the best point fixes.
"""
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vcommon as VC  # noqa: E402

out = {}
# ---------------------------------------------------------------- 1. z*
log = "TB-Q100-ae-l2_3"
mine = torch.load(os.path.join(VC.VER, "cache", f"targets_{log}.pt"), map_location="cpu", weights_only=False)
theirs = torch.load(os.path.join(VC.P12, "cache", f"targets_{log}.pt"), map_location="cpu", weights_only=False)
rows = VC.records(log)
byg = {r["group"]: r for r in rows}
z = {"image": [], "text": []}
for g, a in mine.items():
    b = theirs.get(g)
    if b is None or a["attempts"] != 1 or b["attempts"] != 1 or b.get("zproc"):
        continue                                     # both must be the plain first vendor attempt
    da, db = a["blob"]["delta"].double(), b["blob"]["delta"].double()
    z[byg[g]["ptype"]].append({"group": g, "rel": float((da - db).norm() / db.norm()),
                               "h_equal": bool(torch.equal(a["blob"]["h"], b["blob"]["h"])),
                               "steps": [int(a["blob"]["steps_used"]), int(b["blob"]["steps_used"])],
                               "oracle": [bool(a["oracle"]), bool(b["oracle"])]})
zs = {}
for pt, v in z.items():
    r = np.array([x["rel"] for x in v])
    zs[pt] = {"n": len(v), "bitwise_equal": int((r == 0).sum()), "median_rel_diff": float(np.median(r)), "max_rel_diff": float(r.max()),
              "h_bitwise_equal": int(sum(x["h_equal"] for x in v)), "steps_differ": int(sum(x["steps"][0] != x["steps"][1] for x in v)),
              "oracle_differ": int(sum(x["oracle"][0] != x["oracle"][1] for x in v))}
# flips documented by the experiment itself
flips = []
for line in open(os.path.join(VC.P12, "logs", f"streams_{log}.log")):
    m = re.search(r"z\* of (\S+) recomputed with the oracle fallback ladder: oracle (\w+) \(attempts (\d+)\) -> (\w+) \(attempts (\d+)\)", line)
    if m:
        flips.append({"source": f"p12 logs/streams_{log}.log", "group": m.group(1), "first_run_oracle": m.group(2) == "True",
                      "second_run_oracle": m.group(4) == "True", "second_run_attempts": int(m.group(5)), "ptype": byg[m.group(1)]["ptype"]})
import json
rq1 = [json.loads(l) for l in open(os.path.join(VC.V3, "rq1", "rows", f"rows_{log}.jsonl")) if l.strip()]
ni = sorted({r["group"] for r in rq1 if r["method"] == "REEDIT" and (r.get("meta") or {}).get("status") == "zopt_no_oracle"})
for g in ni:
    e = theirs.get(g)
    flips.append({"source": "07_repair_v3/rq1 rows (default call, one attempt) vs p12 cache/targets", "group": g, "first_run_oracle": False,
                  "second_run_oracle": None if e is None else bool(e["oracle"]), "second_run_attempts": None if e is None else int(e["attempts"]),
                  "ptype": byg[g]["ptype"]})
out["zstar"] = {"log": log, "first_attempt_pairs": zs, "oracle_flips_in_experiment_files": flips,
                "my_first_attempt_not_oracle": sorted(g for g, a in mine.items() if not a["vendor_attempt_oracle"]),
                "my_oracle_after_ladder": int(sum(a["oracle"] for a in mine.values())), "n_mine": len(mine)}
print("z*:", out["zstar"])

# ---------------------------------------------------------------- 2. near-twins
nt = {}
for log in VC.LOGS:
    rows = VC.records(log)
    pool = VC.my_split(rows)
    img = [r for r in rows if r["ptype"] == "image"]
    ent = lambda r: r["image"].split("/")[0]            # noqa: E731
    exact = {(r["prompt"], r["image"]) for r in img if pool[r["probe_idx"]] == "repair"}
    near = {}
    for r in img:
        if pool[r["probe_idx"]] == "repair":
            near.setdefault((r["prompt"], ent(r)), []).append(r)
    ho = [r for r in rows if pool[r["probe_idx"]] == "heldout" and r["y0_valid"]]
    is_exact = {r["group"]: (r["ptype"] == "image" and (r["prompt"], r["image"]) in exact) for r in ho}
    is_near = {r["group"]: (r["ptype"] == "image" and not is_exact[r["group"]] and (r["prompt"], ent(r)) in near) for r in ho}
    near_c = {r["group"]: (is_near[r["group"]] and any(x["is_complaint_D1"] and x["y0_valid"] for x in near[(r["prompt"], ent(r))])) for r in ho}
    d = VC.jload(os.path.join(VC.P12, "out", log, "COMP_B__nALL__s0.json"))["eval"]["heldout_rows"]
    assert [x["group"] for x in d] == [r["group"] for r in ho]
    aft = {x["group"]: bool(x["reproduced_after"]) for x in d}
    we = {r["group"]: bool(r["reproduced"]) for r in ho}
    comp = [r["group"] for r in ho if not we[r["group"]]]
    def rate(gs):
        return [int(sum(aft[g] for g in gs)), len(gs)]
    rest = [g for g in comp if not is_exact[g] and not near_c[g]]
    strict = [g for g in we if not is_exact[g] and not is_near[g]]
    p, b, c = VC.mcnemar_exact([aft[g] for g in strict], [we[g] for g in strict])
    nt[log] = {"n_valid_heldout": len(ho), "exact_twins": int(sum(is_exact.values())), "near_twins": int(sum(is_near.values())),
               "near_twins_that_are_complaints": int(sum(is_near[g] and not we[g] for g in we)),
               "near_twins_of_a_repair_complaint": int(sum(near_c.values())),
               "best_point_fixed_over_complaints": {"exact_twins": rate([g for g in comp if is_exact[g]]),
                                                    "near_twins_of_a_repair_complaint": rate([g for g in comp if near_c[g]]),
                                                    "all_other_heldout_complaints": rate(rest)},
               "strictly_disjoint_rows": len(strict),
               "best_point_D1_strict": 100.0 * sum(not aft[g] for g in strict) / len(strict),
               "none_D1_strict": 100.0 * sum(not we[g] for g in strict) / len(strict), "mcnemar_p_strict": p, "fixed_lost_strict": [b, c]}
    print(log, nt[log])
out["near_twins"] = nt
VC.jdump(out, os.path.join(VC.VOUT, "cpu_final.json"))
