"""E6, gate level (CPU, key cache): every role-'last' ablation row x 20 calibration draws on one log.

    python e6_gate.py <log> [n_draws]

Writes out/gate_<log>.json: {key: {draw: {stats...}}} with the driver's gate-level statistics (fx.gate.evaluate + fx.metrics.gate_stats,
teacher-forced keys), false firing on BEN-T (fx.pools.firing: prompt-only keys), the same row under the literal rule without the guard
cap (allow_silence=True), the number of capped edits (fx.methods.n_capped) and the own prompts not released.
One log per process.  Resumable per ablation row (a finished row is skipped).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import e6_common as C
from e6_common import fx


def edit_parity(L, d, even):
    """Coverage of the rephrases separately for the edits with even and odd stream index (item 10)."""
    rows = d["rows"]
    kind = np.array([L.index[j]["kind"] for j in rows])
    own = np.array([L.own(j) for j in rows])
    ev = np.isin(own, even)
    out = {}
    for par, m in (("even", ev), ("odd", ~ev)):
        o = {}
        for k in ("tgen", "igen"):
            s = m & (kind == k)
            o[k] = {"n": int(s.sum()), "own_top": fx.metrics.pct(np.asarray(d["own_top"])[s]), "own_cand": fx.metrics.pct(np.asarray(d["own_cand"])[s])}
        out[par] = o
    return out


def one(L, key, draw, even):
    M = C.build(L, key, draw)
    ev = fx.gate.evaluate(L, M, positions=("tf",), parts=("probe", "edit", "hn"), top1=True)
    st = fx.metrics.gate_stats(L, ev, "tf")
    out = {"stats": st, "cal": C.cal_summary(M), "parity": edit_parity(L, ev["edit"]["tf"], even)}
    bt = fx.pools.firing(L, M, "BEN-T")
    out["BEN-T"] = {k: bt[k] for k in ("n", "n_fired", "fired", "which")}
    out["BEN-T"].update(image=bt.get("image"), text=bt.get("text"), far=bt.get("far"), related=bt.get("related"))
    # the literal rule without the guard cap (the event the bound covers)
    if (M.rec.get("cal") or {}).get("allow_silence"):
        lit = st["probe"]
        out["literal"] = {"same_as_row": True, "heldout_disjoint": lit["heldout_disjoint"]["fired"], "heldout_strict": lit["heldout_strict"]["fired"], "BEN-T": bt["fired"]}
    else:
        Ml = C.build(L, key, draw, literal=True)
        evl = fx.gate.evaluate(L, Ml, positions=("tf",), parts=("probe", "edit"))
        sl = fx.metrics.gate_stats(L, evl, "tf")
        out["literal"] = {"same_as_row": Ml.digest == M.digest, "heldout_disjoint": sl["probe"]["heldout_disjoint"]["fired"],
                          "heldout_strict": sl["probe"]["heldout_strict"]["fired"], "BEN-T": fx.pools.firing(L, Ml, "BEN-T")["fired"],
                          "rel_own_top": sl["edit"]["rel"]["own_top"], "own_not_released": (Ml.rec.get("own_prompts") or {}).get("n_not_released")}
    return out


def main():
    name = sys.argv[1]
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else C.N_DRAWS
    L = fx.logs.open(name)
    p = os.path.join(C.OUT, f"gate_{name}.json")
    res = fx.io.jload(p) if os.path.exists(p) else {}
    even = np.array(fx.draws.even_edits(L))
    for key in C.KEYS:
        a = C.BY[key]
        if a["multi"]:
            continue                                                    # roles that look at every position: e6_multi.py (model)
        ds = C.draws_of(key, nd)
        if key in res and all(str(d) in res[key] for d in ds):
            continue
        t0 = time.time()
        res[key] = {str(d): one(L, key, d, even) for d in ds}
        fx.io.jdump(res, p)
        print(f"[e6_gate] {name} {key}: {len(ds)} draws, {time.time() - t0:.0f}s", flush=True)
    meta = {"log": name, "code": fx.code_fingerprint(), "n_draws": nd, "alpha": C.ALPHA, "n": C.N, "describe": L.describe(),
            "even_edits": [int(e) for e in even], "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    # identity checks recorded for the table: null n_A = 0 is the default; feature 'cos' is the registered cosine gate
    chk = {}
    for d in (0, 1, 2):
        Md = C.build(L, "default", d)
        chk[f"null0_equals_default_d{d}"] = fx.methods.rescope_null(L, n_A=0, alpha=C.ALPHA, n=C.N, draw=d).digest == Md.digest
        chk[f"featcos_equals_cosine_d{d}"] = C.build(L, "feat_cos", d).digest == fx.methods.cosine(L, alpha=C.ALPHA, n=C.N, draw=d).digest
        for k in C.GATE_EQUALS_DEFAULT:
            a, b = C.build(L, k, d).obj, Md.obj
            chk[f"{k}_same_thresholds_d{d}"] = bool((a.theta() == b.theta()).all()) and bool((a.det_w == b.det_w).all())
    meta["checks"] = chk
    fx.io.jdump(meta, os.path.join(C.OUT, f"gate_{name}_meta.json"), indent=1)
    print("[e6_gate] done", name, chk, flush=True)


if __name__ == "__main__":
    main()
