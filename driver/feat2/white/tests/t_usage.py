"""Tests of the documentation and of the hand-over to the shared driver: the example of USAGE.txt runs as printed; a new gate goes through
the driver's public calibration / wrapping functions and gives the thresholds of the direct library calls."""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np

import tcommon as T
from tcommon import C, test
from white import scores as WS
from white import wstats as WSt

torch = C.use_library()

USAGE = os.path.join(C.HERE, "USAGE.txt")
BEGIN, END = "    # ---- example (executed by tests/t_usage.py) ----", "    # ---- end of the example ----"


def example_lines() -> list:
    with open(USAGE) as f:
        lines = f.read().splitlines()
    a, b = lines.index(BEGIN), lines.index(END)
    return [ln[4:] if ln.startswith("    ") else ln for ln in lines[a + 1:b]]


@test("usage", "the ten-line example of USAGE.txt is executed as printed (development log TB-Q100-ae-l2_3)")
def u1_usage_example():
    code = example_lines()
    ns, buf = {}, io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile("\n".join(code), "USAGE.txt example", "exec"), ns)
    printed = buf.getvalue().strip().splitlines()
    labels = [ln.split()[0] for ln in printed]
    from rescope import ScopedLog
    ok = (len(code) == 10 and labels == [lab for lab, _, _ in WS.variants()] and isinstance(ns["Sn"], ScopedLog) and tuple(ns["R"].shape) == (8, ns["L"].T)
          and ns["L"].name in C.DEV_LOGS.values() and bool(torch.isfinite(ns["R"]).all()) and "n_guards_lost" in ns["nrec"])
    return {"pass": ok, "n_lines": len(code), "printed": printed, "R_shape": list(ns["R"].shape),
            "note": f"{len(code)} lines executed; printed one line per variant ({', '.join(labels)}); null_w returns a ScopedLog; score matrix {tuple(ns['R'].shape)}: {ok}"}


@test("usage", "hand-over to the shared driver (development logs): a new gate through fx.methods.recalibrate + fx.methods.wrap == the direct library calibration, bit for bit")
def u2_driver_wiring():
    from rescope import add_guards, calibrate_shared, calibrate_two_block
    out, ok, n = {}, True, 0
    for name in C.DEV_LOGS.values():
        ctx = T.context(name)
        if ctx.fx is None:
            return {"pass": False, "note": "the shared driver did not import: " + getattr(ctx, "driver_error", "")}
        fx, L = ctx.fx, ctx.L
        V, _ = fx.methods.released_values(L)                              # the driver's released values (shared memo bits along a log family)
        S0, _ = fx.methods.base_gate(L)                                   # the driver's default gate with its guards
        gates = {"cos_centred": WS.feature_gate(L.log, "cos_centred", stats=ctx.stats, values=V),
                 "cos_null": WS.feature_gate(L.log, "cos_null", null_keys=L.kc.keys("tf", fx.draws.null_sample(L, 100, 0)), values=V),
                 "white_0.1": WS.feature_gate(L.log, "white", stats=ctx.stats, kappa=0.1, values=V),
                 "whitecos_0.1": WS.feature_gate(L.log, "whitecos", stats=ctx.stats, kappa=0.1, values=V)}
        gates = {k: add_guards(g, *L.own_keys())[0] for k, g in gates.items()}
        gates["rescope_nullW"] = WS.null_w(S0, stats=ctx.stats)[0]       # guards already on (base gate), repaired detectors
        e = {}
        for lab, S in gates.items():
            r = {"values_are_the_drivers": bool(torch.equal(S.values, V)), "n_guards": int(S.guard_key.shape[0])}
            for rule, fn, kw in (("two_block", calibrate_two_block, {"seed": 0}), ("shared", calibrate_shared, {})):
                G1, rec1 = fx.methods.recalibrate(L, S, 0.05, 200, 0, rule=rule)                      # the driver's path
                M = fx.methods.wrap(L, G1, f"{lab}_{rule}", rec=rec1, name=lab, alpha=0.05, n=200, draw=0)
                G2, _ = fn(S, coords=L.kc.coords(S, ctx.draw_rows(0), "tf"), alpha=0.05, **kw)       # the direct library call (as white.mechanics)
                d = fx.gate.decide(L, M, L.pool("heldout_disjoint"), "tf")
                d2 = L.kc.decide(G2, "tf", rows=L.pool("heldout_disjoint"), uids=L.uids)
                r[rule] = {"thresholds_bitwise_equal": bool(torch.equal(G1.det_tau, G2.det_tau)),
                           "ranking_equal": (G1.det_b is None and G2.det_b is None) or bool(torch.equal(G1.det_b, G2.det_b)),
                           "method_kind": M.kind, "memo_path": bool(M.memo), "class": type(M.obj).__name__, "fingerprint": M.key,
                           "decisions_equal": bool(np.array_equal(d["fired"], d2["fired"]) and np.array_equal(d["top_edit"], d2["top_edit"])),
                           "own_prompts_not_released": M.rec["own_prompts"]["n_not_released"]}
                ok &= (r[rule]["thresholds_bitwise_equal"] and r[rule]["ranking_equal"] and M.kind == "scoped" and M.memo and r[rule]["decisions_equal"]
                       and len(M.key) == 12)
                n += 1
            ok &= r["values_are_the_drivers"] and r["n_guards"] == 2 * L.T
            e[lab] = r
        keys = {v[rule]["fingerprint"] for v in e.values() for rule in ("two_block", "shared")}
        ok &= len(keys) == 2 * len(e)                                      # every row has its own method key
        out[name] = e
    return {"pass": ok, "per_log": out,
            "note": f"{n} rows (5 gates x 2 rules x 2 logs): fx.methods.recalibrate thresholds == direct library calibration bit for bit; fx.methods.wrap gives kind 'scoped' on "
                    f"the memo path with a distinct method key; fx.gate.decide == KeyCache.decide; values = fx.methods.released_values: {ok}"}
