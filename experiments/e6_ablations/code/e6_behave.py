"""E6, behaviour (GPU): the ablation rows registered for behaviour by R19, calibration draws 0 - 2, on one log.

    gq run --mem <MiB> --prio 1 -- python e6_behave.py <log> [key ...]

Rows: default, resp_lin (first: the main table needs it), values_logged, cal_shared, cal_logonly, cal_runmax, guards_off, contam_20,
role_all; reference: the deployed log (edits kept / gained of R7).  Per (row, draw): edits (Rel / T-Gen / I-Gen), heldout_disjoint
probes (heldout_strict = the rows flagged strict), valid hard negatives; at draw 0: the 48-token decode of the pre-correct
heldout_disjoint rows (D2).  Every row through fx.behave (memo path / library path).  Resumable per cell (out/cells/, out/rows/).
One log per process; ends with ses.assert_clean().
"""
from __future__ import annotations

import os
import sys
import time

import e6_common as C
from e6_common import fx

ORDER = ["default", "resp_lin", "values_logged", "cal_shared", "cal_logonly", "cal_runmax", "guards_off", "contam_20", "role_all"]


def main():
    name = sys.argv[1]
    keys = sys.argv[2:] or ORDER
    assert all(C.BY[k]["behave"] for k in keys)
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    ses = fx.behave.session(L, H)
    ses.verbose = True
    cells = os.path.join(C.OUT, "cells")
    os.makedirs(cells, exist_ok=True)
    strict, disj = set(L.pool("heldout_strict")), set(L.pool("heldout_disjoint"))
    assert strict <= disj, "heldout_strict is read as the strict-flagged rows of heldout_disjoint"
    R = fx.io.Results("E6", C.OUT)                                       # only for the row files (results.json is written by e6_render.py)
    tag0 = f"{C.SHORT[name]}__deployed"
    p0 = os.path.join(cells, tag0 + ".json")
    if not os.path.exists(p0):
        Md = fx.methods.deployed(L)
        t0 = time.time()
        e, p, h = ses.edits(Md), ses.probes(Md), ses.hard_negatives(Md)
        R.rows(tag0, e + p + h)
        fx.io.jdump({"log": name, "key": "deployed", "n_rows": len(e + p + h), "seconds": time.time() - t0, "stats": dict(ses.stats)}, p0, indent=1)
        print(f"[e6_behave] {name} deployed {time.time() - t0:.0f}s", flush=True)
    for key in keys:
        for d in (C.BEHAVE_DRAWS if C.BY[key]["per_draw"] else (0,)):
            tag = f"{C.SHORT[name]}__{key}__d{d}"
            pc = os.path.join(cells, tag + ".json")
            if os.path.exists(pc):
                continue
            t0 = time.time()
            s0 = dict(ses.stats)
            M = C.build(L, key, d, H=H)
            e = ses.edits(M)
            p = ses.probes(M)
            h = ses.hard_negatives(M)
            R.rows(tag, e + p + h)
            rec = {"log": name, "key": key, "draw": d, "variant": M.variant, "digest": M.digest, "memo_path": bool(M.memo), "cal": C.cal_summary(M),
                   "n_edit_rows": len(e), "n_probe_rows": len(p), "n_hn_rows": len(h), "seconds_tf": time.time() - t0}
            if d == 0:                                                  # D2 at draw 0 (R19 a: generated answers and D2 at draw 0)
                t1 = time.time()
                g = ses.d2(M)
                R.rows(tag + "__d2", g)
                rec.update(n_d2_rows=len(g), seconds_d2=time.time() - t1)
            rec["session"] = {k: ses.stats[k] - s0.get(k, 0) for k in ("rows", "pre", "memo", "run", "lib", "forwards", "decodes", "hook_mismatch")}
            clean = ses.assert_clean()
            rec["assert_clean"] = clean
            fx.io.jdump(rec, pc, indent=1)
            print(f"[e6_behave] {name} {key} d{d} {time.time() - t0:.0f}s {rec['session']}", flush=True)
    rec = ses.assert_clean()
    fx.io.jdump({"log": name, "assert_clean": rec, "memo": ses.memo.summary() if hasattr(ses.memo, "summary") else None, "stats": dict(ses.stats),
                 "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S"), "keys": keys},
                os.path.join(C.OUT, f"behave_{name}_job_{int(time.time())}.json"), indent=1)
    print("[e6_behave] done", name, rec, flush=True)


if __name__ == "__main__":
    main()
