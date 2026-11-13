"""The driver calls that USAGE.txt (sections of stage 2) prints, executed once on a development log (CPU; nothing is stored)."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import os, sys
sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/10_final/common")
import fx
import numpy as np
R1, MT = fx.review1, fx.methods
L = fx.logs.open(sys.argv[1] if len(sys.argv) > 1 else "TB-Q100-ae-l2_3")
ok = {}
# E1b / E1c for a cosine-family variant and the test-row axis
for name in ("whitecos_0.1", "cos_centred", "cos_null", "white_1_shared"):
    sw = fx.gate.sweep(L, name, gate=R1.unthresholded(L, name, draw=0))
    fr = fx.gate.frontier(L, sw=sw)
    mc = fx.gate.matched_coverage(L, sw=sw)
    sp = fx.sweeps.h3_split(L)
    pos = {j: i for i, j in enumerate(sw["parts"]["hn"]["rows"])}
    t = float(np.median(fr["thresholds"]))
    per = [float((sw["parts"]["hn"]["rmax"][[pos[j] for j in sp["test"][k]]] >= t).mean()) for k in ("HN-I", "HN-Q", "HN-X")]
    ok["frontier " + name] = len(fr["thresholds"]) == 501 and "hn.HN-I.fired" in fr["curves"] and len(mc) == 3 and len(per) == 3
# the frontier of a calibrated gate at shift 0 == its gate statistics (retrieval conjunction and a scoped gate)
for name in ("retr_conj", "retr_min", "whitecos_0.1"):
    M = MT.build(name, L)
    cv = fx.sweeps.shift_curves(L, M, [0.0, 0.05, -0.05], pools=("BEN-T",))
    st = fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn")))
    ok["shift_curves " + name] = (abs(cv["curves"]["probe.heldout_disjoint.fired"][0] - st["probe"]["heldout_disjoint"]["fired"]) < 1e-9
                                  and abs(cv["curves"]["hn.HN-I.fired"][0] - st["hn"]["HN-I"]["fired"]) < 1e-9 and "hn.test.mean.fired" in cv["curves"]
                                  and abs(cv["curves"]["pool.BEN-T.fired"][0] - fx.pools.firing(L, M, "BEN-T")["fired"]) < 1e-9)
    cs = fx.sweeps.coverage_scale(L, M, 25.0)
    ok["coverage_scale " + name] = "reachable" in cs
# the E3 row set as printed (a selection file written to a scratch path)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sel.json")
R1.write_selection({bb: {"log": "x", "cosine_family": {"H3": "cosine", "H2": "whitecos_0.1_shared"}, "retrieval": {"H3": "retr_sum", "H2": "retr_sum"}} for bb in ("qwen25vl", "llava")}, p)
sel = R1.selected(L, path=p)
ALL = MT.E3_LABELS + (sel["cosine"], sel["cosine_x"], sel["retr"], "lookup", "rescope_x")
GL = ("rescope_nullW", "rescope_pac")
rs = R1.e1_rowset_review1(L, 0.05, 0, labels=tuple(dict.fromkeys(ALL + GL)), skip_unavailable=False)
ok["e3 rowset"] = [M.label for M in rs] == list(dict.fromkeys(ALL + GL)) and sel == {"cosine": "cosine", "cosine_x": "cosine_x", "retr": "retr_sum"}
rq = R1.e1_rowset_review1(L, 0.05, 0, labels=("rescope", "cosine_x", "retr_sum", "cos_null", "rescope_x"), mask="qdisjoint")
ok["qdisjoint rowset"] = all("mask=qdisjoint" in M.variant for M in rq)
rw = R1.e1_rowset_review1(L, 0.05, 0, labels=("rescope", "whitecos_1", "retr_conj", "rescope_x", "lookup"), which="all", guards="all")
ok["decode-proof switches"] = len(rw) == 5 and len({tuple(M.rec["draw"]["sids"]) for M in rw if "draw" in M.rec}) == 1
os.remove(p)
# near-scope read-outs, masks, pools as printed
M = MT.build("rescope_x", L)
hb, ns = R1.h3_both(L, M), R1.near_scope(L, M)
ok["h3_both / near_scope"] = hb["unseen_foreign"]["test"]["subset"]["n_rows"] <= hb["subset"]["n_kept"] and ns["subset"]["n_kept"] == hb["subset"]["n_kept"]
ok["pools mask"] = fx.pools.firing(L, M, "BEN-T", mask=fx.masks.question_disjoint(L))["n_masked_out"] > 0 and fx.pools.firing(L, MT.build("retr_sum", L), "BEN-T")["n"] == 1025
ok["why_not"] = R1.why_not(L, "whitecos_0.1_x") is None and bool(R1.why_not(L, "retr_sum_x"))
print({k: bool(v) for k, v in ok.items()}); print("ALL OK" if all(ok.values()) else "FAILED")
