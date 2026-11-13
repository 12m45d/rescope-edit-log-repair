"""Verification of E1, gate level (CPU, one log per process): a spot check that the experiment's gate-level cells are what the driver gives.
Rebuilds, for calibration draws 0, 7 and 19 at alpha = 0.05, the gates that decide the selection and the confirmatory comparisons and compares with
the stored cells: content digest, false firing on heldout_disjoint, H2 coverage, the H3 read-out (raise, coverage, edits at cap), capped edits, and
the firing on BEN-T / MME-T from fx.pools.firing.

    python v_gatecheck.py <development log>      -> verify/out/gatecheck_<log>.json
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
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
sys.path.insert(0, FINAL + "/common")
import fx  # noqa: E402
import numpy as np  # noqa: E402

EO, VO = FINAL + "/e1_main/out", FINAL + "/e1_main/verify/out"
NAMES = ("rescope", "rescope_x", "white_1_shared", "whitecos_1_shared", "white_0.1_shared", "white_0.01_shared", "white_1_x", "cos_centred_shared", "cosine_shared", "cosine",
         "retr_sum", "retr_min", "retr_conj")
DRAWS = (0, 7, 19)


def main():
    log = sys.argv[1]
    assert log in fx.logs.names("development") + fx.logs.names("working_point")
    L = fx.logs.open(log)
    out, worst, n_digest_differs = [], 0.0, 0
    for nm in NAMES:
        cells = json.load(open(f"{EO}/gate/{log}/{nm}__a0.05.json"))["cells"]
        for d in DRAWS:
            c = cells[str(d)]
            M = fx.methods.build(nm, L, alpha=0.05, n=200, draw=d)
            ev = fx.gate.evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn"))
            st = fx.metrics.gate_stats(L, ev, "tf")
            h = fx.sweeps.h3(L, M)
            mine = {"fire.hd": st["probe"]["heldout_disjoint"]["fired"], "h2.cov": 0.5 * (st["edit"]["tgen"]["own_top"] + st["edit"]["igen"]["own_top"]),
                    "hn.HN-I.fired": st["hn"]["HN-I"]["fired"], "hn.HN-Q.fired": st["hn"]["HN-Q"]["fired"], "hn.HN-X.fired": st["hn"]["HN-X"]["fired"],
                    "n_capped": float(fx.methods.n_capped(M)), "h3.reachable": float(bool(h["reachable"])), "h3.delta": h.get("delta") or 0.0,
                    "h3.cov": h["rephrase"]["mean"] if h["reachable"] else 0.0, "h3.n_at_cap": float(h.get("n_edits_at_cap") or 0), "BEN-T.fired": fx.pools.firing(L, M, "BEN-T")["fired"]}
            theirs = {"fire.hd": c["st"]["probe"]["heldout_disjoint"]["fired"], "h2.cov": c["h2_cov"], "hn.HN-I.fired": c["st"]["hn"]["HN-I"]["fired"], "hn.HN-Q.fired": c["st"]["hn"]["HN-Q"]["fired"],
                      "hn.HN-X.fired": c["st"]["hn"]["HN-X"]["fired"], "n_capped": float(c["n_capped"]), "h3.reachable": float(bool(c["h3"]["reachable"])), "h3.delta": c["h3"].get("delta") or 0.0,
                      "h3.cov": c["h3"]["rephrase"]["mean"] if c["h3"]["reachable"] else 0.0, "h3.n_at_cap": float(c["h3"].get("n_edits_at_cap") or 0), "BEN-T.fired": c["BEN-T"]["fired"]}
            if not nm.startswith("retr"):
                mine["MME-T.fired"], theirs["MME-T.fired"] = fx.pools.firing(L, M, "MME-T")["fired"], c["MME-T"]["fired"]
            df = max(abs(mine[k] - theirs[k]) for k in mine)
            worst = max(worst, df)
            n_digest_differs += int(M.digest != c["digest"])
            out.append({"method": nm, "draw": d, "digest_equal": M.digest == c["digest"], "max_abs_diff": df, "mine": mine, "experiment": theirs})
            print(nm, d, M.digest == c["digest"], f"{df:.2e}", flush=True)
    fx.io.jdump({"log": log, "names": NAMES, "draws": DRAWS, "n_cells": len(out), "n_digest_differs": n_digest_differs, "largest_abs_diff": worst, "cells": out,
                 "ok": bool(worst < 1e-9 and n_digest_differs == 0)}, f"{VO}/gatecheck_{log}.json")
    print("DONE", log, len(out), n_digest_differs, worst)


if __name__ == "__main__":
    main()
