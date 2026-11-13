"""E2 part c, gate level (CPU, key cache): ReScope on label-free evidence.

  n = 0                      rescope_logonly (log-only start)
  n in {1, 2, 5, 10, 20}     + least repair of the n benign queries of the nested draw (3 seeds = draws 0 - 2)
  n in {40, 50, 100, 200, all valid REPAIR rows}   the default two-block rule at alpha = 0.05 (20 draws; at n = all the sample is the
                             whole pool and a draw only changes the random split into the two blocks), and rescope_shared
Per cell: fx.gate.evaluate + fx.metrics.gate_stats (heldout_disjoint / strict firing, near / far, coverage of the own requests and
rephrases, firing on the near-scope negatives), fx.methods.n_capped, the literal rule without guards (allow_silence=True),
false firing on BEN-T and MME-T through fx.pools.firing (prompt-only keys; test side only).
-> out/parts/c_gate_<log>.json (all draws kept).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2c  # noqa: E402
from e2c import fx  # noqa: E402


def read(L, M, literal=None):
    st = fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn"), top1=True))
    out = {"probe": {k: st["probe"][k] for k in ("heldout_disjoint", "heldout_strict", "repair", "heldout_disjoint_near", "heldout_disjoint_far",
                                                "heldout_disjoint_image", "heldout_disjoint_text")},
           "lineage": st["probe"]["lineage"],
           "edit": {m: {k: st["edit"][m][k] for k in ("n", "fired", "own_cand", "own_top", "other_top")} for m in ("rel", "tgen", "igen")},
           "hn": {t: {k: st["hn"][t][k] for k in ("n", "fired", "own_top")} for t in ("HN-I", "HN-Q", "HN-X")},
           "top1_n_differs": st.get("top1", {}).get("all", {}).get("n_differs"),
           "n_edits_capped": fx.methods.n_capped(M), "digest": M.digest}
    out["hn"]["mean_fired"] = sum(out["hn"][t]["fired"] for t in ("HN-I", "HN-Q", "HN-X")) / 3.0
    cal = M.rec.get("cal", {})
    out["cal"] = {k: cal.get(k) for k in ("op", "eta", "bound_false_firing", "n_own_keys_silenced_by_literal_rule", "n_edits_capped", "guarantee") if k in cal}
    if "least_repair" in M.rec:
        out["least_repair"] = {k: v for k, v in M.rec["least_repair"].items() if k != "sids"}
    for pool in ("BEN-T", "MME-T"):
        try:
            f = fx.pools.firing(L, M, pool)
            out[pool] = {"n": f["n"], "fired": f["fired"], "which": f["which"], "far": f["far"]["fired"], "n_far": f["far"]["n"]}
        except Exception as e:      # noqa: BLE001  (a pool that is not defined for the method is recorded, not substituted)
            out[pool] = {"not_defined": f"{type(e).__name__}: {e}"[:200]}
    if literal is not None:
        sl = fx.metrics.gate_stats(L, fx.gate.evaluate(L, literal, positions=("tf",), parts=("probe", "edit"), top1=False))
        out["literal"] = {"heldout_disjoint_fired": sl["probe"]["heldout_disjoint"]["fired"], "heldout_strict_fired": sl["probe"]["heldout_strict"]["fired"],
                          "rel_own_top": sl["edit"]["rel"]["own_top"]}
    return out


def main(name):
    L = fx.logs.open(name)
    n_all = len(L.pool("repair"))
    cells, t0 = [], time.time()
    cells.append({"method": "rescope", "n": 0, "n_label": "0", "draw": 0, "rule": "log-only", "st": read(L, fx.methods.evidence(L, 0, 0))})
    for n in (1, 2, 5, 10, 20):
        for s in (0, 1, 2):
            cells.append({"method": "rescope", "n": n, "n_label": str(n), "draw": s, "rule": "log-only + least repair", "st": read(L, fx.methods.evidence(L, n, s))})
    for n in (40, 50, 100, 200, n_all):
        for d in range(fx.draws.N_DRAWS):
            M = fx.methods.evidence(L, n, d, 0.05)
            cells.append({"method": "rescope", "n": n, "n_label": "ALL" if n == n_all else str(n), "draw": d, "rule": "two-block", "alpha": 0.05,
                          "st": read(L, M, fx.methods.evidence(L, n, d, 0.05, allow_silence=True))})
            Ms = fx.methods.rescope_shared(L, alpha=0.05, n=n, draw=d)
            cells.append({"method": "rescope_shared", "n": n, "n_label": "ALL" if n == n_all else str(n), "draw": d, "rule": "shared scale", "alpha": 0.05,
                          "st": read(L, Ms, fx.methods.rescope_shared(L, alpha=0.05, n=n, draw=d, allow_silence=True))})
        print(name, "n", n, f"{time.time() - t0:.0f}s", flush=True)
    fx.io.jdump({"log": name, "n_all_valid_repair": n_all, "cells": cells, "code": fx.code_fingerprint(),
                 "pools": "BEN-T / MME-T: fx.pools.firing, prompt-only keys, gates calibrated at teacher-forced REPAIR keys"},
                os.path.join(e2c.OUT, "parts", f"c_gate_{name}.json"))


if __name__ == "__main__":
    main(sys.argv[1])
