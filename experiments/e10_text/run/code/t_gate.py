"""Test of the fast gate-level evaluator (xgate) on the DEVELOPMENT log -> out/tests_gate.json.
  a  every number of out/gate/<dev>.json for the adapter's rows (rescope, cosine, where_twoblock over 20 draws; cosine_published,
     exact, where_logonly) equals the adapter's own gate-level run of the development log (e10_text/out/dev/parts/gate.json)
  b  an R3 gate with a query-normalised cosine (cos_centred, whitecos_0.1; draw 0) and white_0.1, cos_null: xgate (cpu) equals
     e10t.gate.evaluate + gate_stats with the run-time patch of the per-position refinement
  c  thresholds of xmeth.calibrated('rescope' / 'cosine', draw 0) are bitwise those of e10t.methods.rowset
  d  lookup_subject: the plan of every request prompt holds (solver position, own edit) unless a later edit has the same subject
"""
import sys

import xcommon as X
import numpy as np
import torch
import e10t
from e10t import gate as G, methods as MT
import xmeth as XM
import xgate as XG

res = {}
L = e10t.open_log(X.DEV)
mine = X.jload(X.OUT + f"/gate/{X.DEV}.json")["rows"]
ref = X.jload(X.E10T + "/out/dev/parts/gate.json")["per_draw"]
KEYS = ("n", "fired", "fired_prompt", "fired_cont_only", "fired_last_only", "lineage_width_mean_if_fired", "lineage_width_max", "n_clean", "fired_clean",
        "n_near_scope", "fired_near_scope", "own_cand", "own_released", "other_released", "only_other_released")
worst, n_cmp, bad = 0.0, 0, []
for lab in ("rescope", "cosine", "where_twoblock", "cosine_published", "exact", "where_logonly"):
    for d, st in ref[lab].items():
        for part in ("heldout", "paraphrase", "neighborhood", "request"):
            for k in KEYS:
                if k in st[part]:
                    a, b = mine[lab]["gon"]["per_draw"][d][part][k], st[part][k]
                    n_cmp += 1
                    if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                        bad.append([lab, d, part, k, a, b])
        c = mine[lab]["gon"]["cal"][d]
        if "eta" in (c or {}) and st["cal"]["eta"] is not None:
            n_cmp += 1
            if c["eta"] != st["cal"]["eta"] or c["n_edits_capped"] != st["cal"]["n_edits_capped"]:
                bad.append([lab, d, "cal", c["eta"], st["cal"]["eta"]])
res["a"] = {"n_compared": n_cmp, "n_different": len(bad), "examples": bad[:10], "ok": not bad and n_cmp > 1000}

lay = XG.Layout(L)
bad, n_cmp = [], 0
for lab in ("cos_centred", "whitecos_0.1", "white_0.1", "cos_null"):
    S2, rec = XM.calibrated(L, lab, 0, True)
    r = XG.evaluate(L, lay, [(XG.light(S2), [XG.variant_of("x", S2)])], "cpu")["x"]
    st = XG.stats(L, lay, r)
    st0 = G.gate_stats(L, G.evaluate(L, S2, parts=lay.parts))
    for part in lay.parts:
        for k in KEYS:
            if k in st0[part]:
                n_cmp += 1
                a, b = st[part][k], st0[part][k]
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-9):
                    bad.append([lab, part, k, a, b])
res["b"] = {"n_compared": n_cmp, "n_different": len(bad), "examples": bad[:10], "ok": not bad}

ok = True
for lab in ("rescope", "cosine"):
    S2, _ = XM.calibrated(L, lab, 0, True)
    M = MT.rowset(L, draw=0, labels=[lab])[0]
    ok &= bool(torch.equal(S2.theta(), M.obj.theta())) and bool(torch.equal(S2.det_b, M.obj.det_b))
    S3, _ = XM.calibrated(L, lab, 0, False)
    M3 = MT.rowset(L, draw=0, labels=[lab], allow_silence=True)[0]
    ok &= bool(torch.equal(S3.theta(), M3.obj.theta()))
res["c"] = {"ok": ok}

Mk = XM.lookup_method(L)
miss = []
for i in range(L.T):
    pl = XM.lookup_text_plan(L, Mk, L.rows["request"][i]["text_id"])
    later = any(L.requests[j]["subject"] == L.requests[i]["subject"] for j in range(i + 1, L.T))
    if (L.solver_pos[i], i) not in pl and not later:
        miss.append(i)
res["d"] = {"n_requests": L.T, "n_missing": len(miss), "missing": miss[:10], "ok": not miss}
res["all_ok"] = all(v["ok"] for v in res.values())
X.jdump(res, X.OUT + "/tests_gate.json")
print({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "examples"}) for k, v in res.items()})
print(res["a"]["examples"][:5], res["b"]["examples"][:5])
sys.exit(0 if res["all_ok"] else 1)
