"""Verification of E1c (matched-coverage read-out), CPU: own computation with the LIBRARY only (rescope.KeyCache, from_testbed, feature_gate; no driver).
One global threshold t on a feature ('ncoord' = normalised solver coordinate, 'cos' = raw cosine to the stored key), top-1 by the feature, no guards:
an edit is released iff the largest feature of the key is >= t.  For each target (25, 50, 75 percent) the largest t at which the coverage of the union of
text and image rephrases (the own edit is the top-1 edit and is released) reaches the target; there: firing on heldout_disjoint and on each kind of
near-scope negative.  Compared with e1_main/out/frontier.json.

    python v_e1c.py            -> verify/out/e1c_check.json
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
import rescope  # noqa: E402

FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
EO, VO = FINAL + "/e1_main/out", FINAL + "/e1_main/verify/out"
DEV = {"TB-Q100-ae-l2_3": "qwen25vl", "TB-L100-ae-l2_3": "llava"}
HN = {"hn_i": "HN-I", "hn_q": "HN-Q", "hn_x": "HN-X"}


def main():
    FR = json.load(open(EO + "/frontier.json"))["logs"]
    out, worst = {}, 0.0
    for log, bb in DEV.items():
        kc = rescope.KeyCache(bb)
        L = rescope.from_testbed(log)
        uids = L.meta["uids"]
        T = len(uids)
        rows = kc.log_rows(uids)
        ix = [kc.index[j] for j in rows]
        K = kc.keys("tf", [j for j, e in zip(rows, ix) if e.get("row_tf") is not None])
        have = [e for e in ix if e.get("row_tf") is not None]
        out[log] = {}
        for feat in ("ncoord", "cos"):
            G = rescope.feature_gate(L, feat)
            R = G.scores(K).numpy()                                     # (n, D) one detector per edit
            de = G.det_edit.numpy() if hasattr(G.det_edit, "numpy") else np.asarray(G.det_edit)
            assert R.shape[1] == T and (de == np.arange(T)).all()
            rmax, amax = R.max(1), R.argmax(1)
            kind = np.array([e["kind"] for e in have])
            reph = np.isin(kind, ("tgen", "igen"))
            own = np.array([e.get("edit") if e["kind"] in ("rel", "tgen", "igen") else -1 for e in have])
            hd = np.array([e["kind"] == "probe" and bool(e.get("y0_valid")) and bool(e.get("disjoint")) for e in have])
            hn = {k: np.array([e["kind"] == k and bool(e.get("y0_valid")) and (k != "hn_x" or all(int(x) < T for x in ([e["edit"]] if not isinstance(e["edit"], list) else e["edit"]))) for e in have]) for k in HN}
            top = reph & (amax == own)
            v = np.sort(rmax[top])[::-1]
            N = int(reph.sum())
            cells = []
            ref = {m["target"]: m for m in FR[log]["matched_coverage"][feat]}
            for c in (25, 50, 75):
                k = int(np.ceil(c / 100.0 * N - 1e-9))
                e = ref[c]
                if k < 1 or k > v.size:
                    cells.append({"target": c, "reachable": False, "experiment_reachable": e["reachable"]})
                    continue
                t = float(v[k - 1])
                mine = {"threshold": t, "coverage": 100.0 * float((rmax[top] >= t).sum()) / N, "heldout_disjoint_fired": 100.0 * float((rmax[hd] >= t).mean()),
                        **{f"hn.{HN[q]}": 100.0 * float((rmax[m] >= t).mean()) for q, m in hn.items()}}
                theirs = {"threshold": e["threshold"], "coverage": e["coverage"], "heldout_disjoint_fired": e["heldout_disjoint_fired"], **{f"hn.{q}": e["hn_fired"][q] for q in HN.values()}}
                df = max(abs(mine[x] - theirs[x]) for x in mine)
                worst = max(worst, df)
                cells.append({"target": c, "reachable": True, "experiment_reachable": e["reachable"], "mine": mine, "experiment": theirs, "max_abs_diff": df})
            out[log][feat] = {"n_rephrase_rows": N, "n_heldout_disjoint": int(hd.sum()), "n_hn": {HN[q]: int(m.sum()) for q, m in hn.items()}, "cells": cells}
            print(log, feat, [(x["target"], x.get("max_abs_diff")) for x in cells], flush=True)
    res = {"what": __doc__, "logs": out, "largest_abs_diff": worst, "ok": bool(worst < 1e-6)}
    with open(VO + "/e1c_check.json", "w") as f:
        json.dump(res, f)
    print("DONE", worst)


if __name__ == "__main__":
    main()
