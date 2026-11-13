"""E10: what the LOG ALONE shows (no model, no method): R21 (c) screening and the description of the stream (task 5, part 1).

    python run_logdesc.py   ->  out/screening.json, out/stream/description.json
Screening: self-separation margin of every edit = 1 - max over the other edits j of <W_i, k_j> / <W_i, k_i> (the largest normalised
coordinate of edit i at the solver key of another edit).  Flagged: margin <= 0.  Compared with (a) the updates of exploding norm
(the audit's list: audit/out/recon.json, stream_updates.exploding: update norm > |W_0|_F) and (b) the requests whose subject is
the first token of the prompt (public lookup index 0): precision and recall, per reported log (prefix of the stream).
Description: per update the norm of the update, the norm of W - W_0 after it, the division factor (the coordinate of the solver key
the public code divides by), the own coordinate in the library's normalisation, the norm of the read-out vector.
"""
import os

import numpy as np

import xcommon as X
import e10t
import xrepair as XR

rec = X.jload(X.AUDIT + "/out/recon.json")["stream_updates"]
EXPL = sorted(int(x["index"]) for x in rec["exploding"])
out = {"definition": __doc__, "audit_exploding_updates": EXPL, "logs": {}}
desc = {}
R = X.Records("screening")
for lg in X.LOGS:
    L = e10t.open_log(lg)
    T = L.T
    mg = XR.self_margin(L)
    flagged = [int(i) for i in np.nonzero(mg <= 0)[0]]
    first_tok = [i for i in range(T) if L.solver_pos[i] == 0]
    C = ((L.log.K @ L.log.W.T) / L.log.own()[None, :]).numpy()
    np.fill_diagonal(C, -np.inf)
    partner = C.argmax(0)
    o = {"T": T, "n_flagged": len(flagged), "flagged": flagged, "margin_quantiles": {q: float(np.quantile(mg, float(q))) for q in ("0", "0.01", "0.05", "0.5", "0.95", "1")},
         "n_margin_below_0.5": int((mg < 0.5).sum()), "against": {}}
    for nm, ref in (("exploding_norm", [i for i in EXPL if i < T]), ("first_token_subject", first_tok)):
        tp = sorted(set(flagged) & set(ref))
        o["against"][nm] = {"n_reference": len(ref), "reference": ref, "true_positives": tp, "false_positives": sorted(set(flagged) - set(ref)),
                            "false_negatives": sorted(set(ref) - set(flagged)), "precision": 100.0 * len(tp) / len(flagged) if flagged else None,
                            "recall": 100.0 * len(tp) / len(ref) if ref else None,
                            "margins_of_reference": {str(i): float(mg[i]) for i in ref}}
        R.add(lg, "screening", f"screen.{nm}.precision", o["against"][nm]["precision"], pool="log", n_units=len(flagged))
        R.add(lg, "screening", f"screen.{nm}.recall", o["against"][nm]["recall"], pool="log", n_units=len(ref))
    R.add(lg, "screening", "screen.n_flagged", len(flagged), pool="log", n_units=T)
    o["flagged_detail"] = [{"edit": i, "margin": float(mg[i]), "largest_coordinate_at_solver_key_of_edit": int(partner[i]), "solver_pos": L.solver_pos[i],
                            "exploding": i in EXPL, "request": L.requests[i]["prompt"].format(L.requests[i]["subject"]) + " -> " + L.requests[i]["target_new"]["str"]} for i in flagged]
    out["logs"][lg] = o
    if lg == X.LOGS[-1]:
        f = e10t.data._tx().factors(lg)
        sc = f["scalars"]
        wn = L.log.W.norm(dim=1).numpy()
        un = L.log.U.norm(dim=1).numpy()
        own = L.log.own().numpy()
        desc = {"log": lg, "T": T, "W0_fro": float(L.weights()[0].double().norm()),
                "per_update": {"update_fro": [float(s["update_fro"]) for s in sc], "W_minus_W0_fro_after": [float(s["W_minus_W0_fro_after"]) for s in sc],
                               "div_factor": [float(s["div_factor"]) for s in sc], "own_coordinate": [float(x) for x in own], "readout_norm": [float(x) for x in wn],
                               "residual_norm": [float(x) for x in un], "solver_pos": list(L.solver_pos), "p_new_editor": [float(s["v_prob_new"]) for s in sc],
                               "self_margin": [float(x) for x in mg]},
                "exploding_updates": [{"index": i, "update_fro": float(sc[i]["update_fro"]), "div_factor": float(sc[i]["div_factor"]), "readout_norm": float(wn[i]), "solver_pos": L.solver_pos[i],
                                       "self_margin": float(mg[i]), "W_minus_W0_fro_after": float(sc[i]["W_minus_W0_fro_after"]),
                                       "request": L.requests[i]["prompt"].format(L.requests[i]["subject"]) + " -> " + L.requests[i]["target_new"]["str"]} for i in EXPL],
                "summary": {}}
        uf, wf = np.array(desc["per_update"]["update_fro"]), np.array(desc["per_update"]["W_minus_W0_fro_after"])
        for Tp in (100, 173, 174, 500, 1000, 2000):
            desc["summary"][str(Tp)] = {"update_fro_median": float(np.median(uf[:Tp])), "update_fro_max": float(uf[:Tp].max()), "W_minus_W0_fro": float(wf[Tp - 1]),
                                        "n_exploding": int(sum(i < Tp for i in EXPL)), "share_of_squared_norm_in_exploding": float(sum(uf[i] ** 2 for i in EXPL if i < Tp) / (uf[:Tp] ** 2).sum()),
                                        "own_coordinate_min_max": [float(own[:Tp].min()), float(own[:Tp].max())], "readout_norm_median": float(np.median(wn[:Tp])),
                                        "readout_norm_max": float(wn[:Tp].max()), "n_first_token_subject": int(sum(p == 0 for p in L.solver_pos[:Tp])),
                                        "editor_p_new_median": float(np.median(desc["per_update"]["p_new_editor"][:Tp]))}
R.save()
X.jdump(out, X.OUT + "/screening.json")
X.jdump(desc, X.OUT + "/stream/description.json")
for lg, o in out["logs"].items():
    print(lg, "flagged", o["n_flagged"], {k: (v["precision"], v["recall"], v["false_negatives"][:8]) for k, v in o["against"].items()})
