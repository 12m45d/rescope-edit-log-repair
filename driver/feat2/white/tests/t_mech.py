"""Tests of white.mechanics: the stored mechanics check of the two development logs."""
from __future__ import annotations

import json
import os

import tcommon as T
from tcommon import C, test
from white import mechanics as MC


def _stored(name):
    with open(os.path.join(C.OUT, f"mechanics_{name}.json")) as f:
        return MC._unclean(json.load(f))


@test("mechanics", "the registered rows of the mechanics check agree with the sanity experiment e0 (per-row file, table.txt) and with the shared driver's rows")
def m1_agrees_with_e0_and_driver():
    out, ok = {}, True
    for name in C.DEV_LOGS.values():
        ck = _stored(name)["checks"]
        er, td, t20 = ck["e0_rows"], ck["e0_table_draw0"], ck["e0_table_20_draws"]
        e = {"same_library_version_as_e0": er["e0_rescope_fingerprint"] == er["rescope_fingerprint_now"],
             "calibration_queries_equal": er["calibration_sids_equal"], "rows_compared": er["n_rows_also_in_e0"], "rows_here": er["n_rows_here"],
             "thresholds_bitwise_equal": all(m["thresholds_bitwise_equal"] for m in er["methods"].values()),
             "rows_with_other_decision": sum(m["n_rows_fired_differs"] + m["n_rows_own_top_differs"] for m in er["methods"].values()),
             "table_draw0_equal": all(m["equal"] for m in td["methods"].values()),
             "table_20_draw_cells_equal": f"{t20['n_equal']} of {t20['n_cells']}",
             "driver_imported": "driver_rows" in ck,
             "driver_thresholds_bitwise_equal": all(v.get("thresholds_bitwise_equal") is True for v in ck["driver_rows"]["rows"].values()) if "driver_rows" in ck else None,
             "driver_gate_level_identical": (f"{sum(all(v.values()) for v in ck['driver_gate_level']['rows'].values() if 'error' not in v)} of "
                                             f"{len(ck['driver_gate_level']['rows'])}") if "driver_gate_level" in ck else None}
        good = (e["calibration_queries_equal"] and e["rows_compared"] == e["rows_here"] and e["thresholds_bitwise_equal"] and e["rows_with_other_decision"] == 0
                and e["table_draw0_equal"] and t20["n_equal"] == t20["n_cells"] and e["driver_imported"] and e["driver_thresholds_bitwise_equal"]
                and e["driver_gate_level_identical"].split(" of ")[0] == e["driver_gate_level_identical"].split(" of ")[1])
        ok &= bool(good)
        out[name] = e
    q = out[C.DEV_LOGS["qwen25vl"]]
    return {"pass": ok, "per_log": out,
            "note": f"both logs, draw 0: thresholds of rescope / cosine x two-block / shared bitwise equal to e0's, 0 of {q['rows_compared']} (Qwen) rows decided otherwise, "
                    f"eta / capped of e0's table equal, 20-draw cells {q['table_20_draw_cells_equal']} (Qwen) and {out[C.DEV_LOGS['llava']]['table_20_draw_cells_equal']} (LLaVA) "
                    f"reproduced; driver rows bitwise equal; fx.gate.decide == KeyCache.decide for the new gates: {ok}"}


@test("mechanics", "a re-run in a fresh process reproduces out/mechanics_<log>.json byte for byte, and out/mechanics.txt is the rendering of these files")
def m2_reproducible():
    rc, log = T.child("white.mechanics", ["check"], timeout=500)
    lines = [ln for ln in log.splitlines() if "reproduced" in ln or "DIFFERS" in ln or "rendering" in ln]
    ok = rc == 0 and len(lines) == len(C.DEV_LOGS) + 1 and not any("DIFFERS" in ln for ln in lines)
    return {"pass": ok, "exit": rc, "lines": lines, "tail": log.splitlines()[-3:] if not ok else [],
            "note": f"mechanics_<log>.json of both development logs re-computed in a fresh process (4 threads): byte-identical, mechanics.txt == render(): {ok}"}


@test("mechanics", "the mechanics check reads development logs only, and every variant row is present for both calibrations")
def m3_scope():
    labels = [v for v, _ in MC.variant_list()]
    ok, out = True, {}
    for name in C.DEV_LOGS.values():
        r = _stored(name)
        good = (r["log"] == name and list(r["variants"]) == labels and all(set(e["cal"]) == set(MC.CALS) for e in r["variants"].values())
                and r["draw"] == 0 and r["alpha"] == 0.05 and r["n_cal"] == 200 and len(r["calibration_sids"]) == 200 and len(r["null_sids"]) == 100
                and not set(r["calibration_sids"]) & set(r["null_sids"]))
        ok &= good
        out[name] = {"variants": len(r["variants"]), "draw_source": r["draw_source"]}
    refused = None
    try:
        MC.Context("TB-Q200-ae-l2_3")
    except SystemExit as e:
        refused = str(e)[:80]
    ns = {n: _stored(n)["null_sample"] for n in C.DEV_LOGS.values()}
    twins_ok = all(v["heldout_disjoint"]["same_query"] == 0 and v["calibration"]["same_query"] == v["calibration"]["same_query_key_bitwise_equal"]
                   and all((v["after_repair"][k]["max_abs_score_of_same_query_calibration_rows"] or 0.0) <= 1e-9 for k in ("rescope_null", "cos_null"))
                   for v in ns.values())
    ok &= twins_ok
    files = sorted(f for f in os.listdir(C.OUT) if f.startswith("mechanics_"))
    only_dev = files == sorted(f"mechanics_{n}.json" for n in C.DEV_LOGS.values())
    tw = "; ".join(f"{n.split('-')[1]} {v['calibration']['same_query']} of 200" for n, v in ns.items())
    return {"pass": ok and refused is not None and only_dev, "per_log": out, "other_log_refused": refused, "files": files,
            "calibration_queries_identical_to_a_null_sample_query": {n: v["calibration"] for n, v in ns.items()},
            "note": f"{len(labels)} rows x 2 calibrations x 2 development logs; draw 0, 200 calibration + 100 null queries, disjoint as rows; another log is refused: "
                    f"{refused is not None}; draw function: the shared driver's ({out[C.DEV_LOGS['qwen25vl']]['draw_source'][:50]}...); calibration queries that ARE a null-sample query "
                    f"(same key bit for bit, score 0 after the repair): {tw}; none in HELD-OUT-DISJOINT: {twins_ok}"}
