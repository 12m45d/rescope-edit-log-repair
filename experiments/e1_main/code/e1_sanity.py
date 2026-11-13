"""E1: comparison with the sanity run e0 (10_final/e0_sanity/out/results.json) for the cells both hold: the default method (M1), the
shared scale (M2), the cosine gate with both calibrations (M3, M4), cosine 0.65 (M5), exact match (M6), deployed, pre-edit; alpha = 0.05.
Gate level: every one of the 20 draws; behaviour: draws 0 - 2.  Writes out/sanity_compare.json (largest absolute difference per cell)."""
import os

from e1c import *  # noqa: F401,F403

E0 = jload(FINAL + "/e0_sanity/out/results.json")
MAP = {"M1": "rescope", "M2": "rescope_shared", "M3": "cosine", "M4": "cosine_shared", "M5": "cosine_default", "M6": "exact", "deployed": "deployed", "pre-edit": "preedit"}


def main():
    recs = jload(os.path.join(OUT, "results.json"))
    G = jload(os.path.join(OUT, "gate_draws.json"))["draws"]
    idx = {(r["log"], r["method"], r["metric"], r["pool"], r["draw"]): r["value"] for r in recs}
    out = {"what": __doc__, "logs": {}}
    worst = 0.0
    for log in DEV:
        e0 = E0["logs"][log]
        cells = []
        # gate level, 20 draws
        for m0, lab in MAP.items():
            if m0 not in e0["gate"]["methods"]:
                continue
            g = e0["gate"]["methods"][m0]["tf"]
            a = "0.05" if "0.05" in G[log][lab] else "none"
            for path, mine in ((("probe", "heldout_disjoint", "fired"), "fire.hd"), (("probe", "heldout_strict", "fired"), "fire.strict"), (("edit", "tgen", "own_top"), "cov.tgen"),
                               (("edit", "igen", "own_top"), "cov.igen"), (("hn", "HN-I", "fired"), "hn.HN-I.fired"), (("hn", "HN-Q", "fired"), "hn.HN-Q.fired"), (("hn", "HN-X", "fired"), "hn.HN-X.fired")):
                v = g
                for k in path:
                    v = v[k]
                theirs = v["values"] if isinstance(v, dict) else [v]
                ours = G[log][lab][a][mine]
                d = float(np.max(np.abs(np.array(theirs, dtype=float) - np.array(ours, dtype=float))))
                cells.append({"level": "gate", "method": lab, "e0": m0, "metric": mine, "n_draws": len(theirs), "max_abs_diff": d, "e0_mean": float(np.mean(theirs)), "e1_mean": float(np.mean(ours))})
                worst = max(worst, d)
        # behaviour, draws 0 - 2
        for m0, lab in MAP.items():
            b = e0["behaviour"]["methods"].get(m0)
            if b is None:
                continue
            per = {None: b["single"]} if "single" in b else {int(k): v for k, v in b["draws"].items()}
            for path, metric, pool in ((("edit", "rel", "exact"), "edit.rel.exact", "edits"), (("edit", "tgen", "exact"), "edit.tgen.exact", "edits"), (("edit", "igen", "exact"), "edit.igen.exact", "edits"),
                                       (("probe", "heldout_disjoint", "D1"), "probe.D1", "heldout_disjoint"), (("probe", "heldout_strict", "D1"), "probe.D1", "heldout_strict"),
                                       (("probe", "heldout_disjoint", "T-Loc*"), "probe.T-Loc*", "heldout_disjoint"), (("probe", "heldout_disjoint", "I-Loc*"), "probe.I-Loc*", "heldout_disjoint"),
                                       (("hn", "HN-I", "D1"), "hn.HN-I.D1", "hard_negatives"), (("hn", "HN-Q", "D1"), "hn.HN-Q.D1", "hard_negatives"), (("hn", "HN-X", "D1"), "hn.HN-X.D1", "hard_negatives")):
                th, ou = [], []
                for d, v in per.items():
                    for k in path:
                        v = v[k]
                    o = idx.get((log, lab, metric, pool, d))
                    if o is None:
                        continue
                    th.append(float(v))
                    ou.append(float(o))
                if th:
                    d = float(np.max(np.abs(np.array(th) - np.array(ou))))
                    cells.append({"level": "behaviour", "method": lab, "e0": m0, "metric": f"{metric} ({pool})", "n_draws": len(th), "max_abs_diff": d, "e0_mean": float(np.mean(th)), "e1_mean": float(np.mean(ou))})
                    worst = max(worst, d)
        out["logs"][log] = {"cells": cells, "n_cells": len(cells), "n_cells_differ_above_1e-9": sum(c["max_abs_diff"] > 1e-9 for c in cells),
                            "largest_abs_diff": max(c["max_abs_diff"] for c in cells) if cells else None,
                            "differing": [c for c in cells if c["max_abs_diff"] > 1e-9]}
    out["largest_abs_diff"] = worst
    jdump(out, os.path.join(OUT, "sanity_compare.json"))
    for log, e in out["logs"].items():
        print(log, "cells", e["n_cells"], "differ", e["n_cells_differ_above_1e-9"], "largest", e["largest_abs_diff"])
        for c in e["differing"][:10]:
            print("   ", c)


if __name__ == "__main__":
    main()
