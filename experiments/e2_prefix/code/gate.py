"""E2 on prefix logs: gate level of the default method on ONE prefix log (CPU; key cache; 20 calibration draws).

    python gate.py <prefix log>      -> out/cells/<log>/gate__rescope.json

Per draw: firing on heldout_disjoint / heldout_strict (near / far), coverage of the own requests and of the rephrases, firing on the
hard negatives of the prefix's edits by kind, capped edits (fx.methods.n_capped), the literal rule without guards (allow_silence=True),
firing on BEN-T (fx.pools.firing: prompt-only keys; the gate is calibrated at teacher-forced REPAIR keys)."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcommon import *  # noqa: F401,F403,E402
from pcommon import fx  # noqa: E402


def main():
    name = sys.argv[1]
    if name not in LOGS:
        raise SystemExit(f"{name} is not a prefix log")
    os.makedirs(os.path.join(CELLS, name), exist_ok=True)
    t0 = time.time()
    L = fx.logs.open(name)
    per = {}
    for d in range(fx.draws.N_DRAWS):
        M = fx.methods.rescope(L, alpha=ALPHA, n=N_CAL, draw=d)
        st = fx.metrics.gate_stats(L, fx.gate.evaluate(L, M, positions=("tf",), top1=True))
        st.pop("benign", None)                                          # the whole benign pool is not a reported column (R25 (2))
        Ml = fx.methods.rescope(L, alpha=ALPHA, n=N_CAL, draw=d, allow_silence=True)
        sl = fx.metrics.gate_stats(L, fx.gate.evaluate(L, Ml, positions=("tf",)))
        bt = fx.pools.firing(L, M, "BEN-T")
        kinds = [t for t in ("HN-I", "HN-Q", "HN-X") if st["hn"][t]["n"]]
        per[d] = {"gate_stats": st, "n_capped": fx.methods.n_capped(M), "capped_edits": M.rec["cal"].get("capped_edits"), "eta": M.rec["cal"].get("eta"),
                  "bound_false_firing": M.rec["cal"].get("bound_false_firing"), "own_prompts": M.rec.get("own_prompts"), "digest": M.digest,
                  "hn_mean_fired": sum(st["hn"][t]["fired"] for t in kinds) / len(kinds) if kinds else None,
                  "literal_no_guards": {"heldout_disjoint_fired": sl["probe"]["heldout_disjoint"]["fired"], "heldout_strict_fired": sl["probe"]["heldout_strict"]["fired"],
                                        "rel_own_top": sl["edit"]["rel"]["own_top"], "n_own_keys_silenced": Ml.rec["cal"].get("n_own_keys_silenced")},
                  "BEN-T": {k: bt[k] for k in ("pool", "which", "n", "n_fired", "fired", "far", "related", "image", "text")}}
    put(name, "gate__rescope", {"log": name, "method": "rescope", "alpha": ALPHA, "n": N_CAL, "draws": per, "seconds": time.time() - t0, "code": fx.code_fingerprint(),
                                "what": "gate level from the key cache (teacher-forced keys); BEN-T: prompt-only keys (fx.pools.firing)"})
    print(name, "gate level done", f"{time.time() - t0:.0f}s", {d: round(per[d]["gate_stats"]["probe"]["heldout_disjoint"]["fired"], 2) for d in per})


if __name__ == "__main__":
    main()
