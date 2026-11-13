"""E10 verification, block 4 (CPU): screening from the log alone (amendment R21 c) on TX-G2XL-rome-T2000, recomputed from the public
code's factors (not through the library's normalisation).

    python v4_screen.py  ->  out/screening_T2000.json
Per update i the public code stores the read-out vector w_i (its 'left' factor, d_in = 6400) and the solver key k_i (k_request: the
key of the request prompt at the last subject token).  Self-separation margin of edit i = 1 - max over j != i of <w_i, k_j> /
<w_i, k_i>.  Flagged: margin <= 0.  Compared with (a) the audit's list of updates of exploding norm and (b) the requests whose
subject is the first prompt token (the public lookup index of the request prompt is 0; cross-checked on the prompt text):
precision and recall; then against the experiment's out/screening.json (flag sets, margins).  Prefix logs T100 / T500 / T1000 are
read off the same matrix (the shorter logs are prefixes of the stream).
"""
import numpy as np

import vcommon as V

import e10t  # noqa: E402
import torch  # noqa: E402
import tx_public as TX  # noqa: E402

f = TX.factors(V.T2000)
Wl = f["left"].double()                     # (T, 6400) read-out vectors
K = f["k_request"].double()                 # (T, 6400) solver keys
T = Wl.shape[0]
own = (Wl * K).sum(1)
C = (K @ Wl.T) / own[None, :]               # C[j, i] = <w_i, k_j> / <w_i, k_i>
# the library's log holds the same quantities up to normalisation: cross-check
L = e10t.open_log(V.T2000)
Clib = (L.log.K @ L.log.W.T) / L.log.own()[None, :]
lib_dev = float((C - Clib).abs().max() / C.abs().max())
audit = V.jload(V.AUDIT + "/out/recon.json")["stream_updates"]
EXPL = sorted(int(x["index"]) for x in audit["exploding"])
# first-token subjects: by the public lookup index and by the text (the prompt starts with the subject)
first_idx = [i for i in range(T) if int(f["scalars"][i]["lookup_idxs"][0]) == 0]
first_txt = [i for i in range(T) if f["requests"][i]["prompt"].startswith("{}") and len(e10t.data.tokenizer()(f["requests"][i]["subject"])["input_ids"]) == 1]
XS = V.jload(V.XOUT + "/screening.json")
out = {"T": T, "own_coordinate_min_max": [float(own.min()), float(own.max())], "max_rel_deviation_from_library_log": lib_dev, "audit_exploding": EXPL,
       "first_token_subject_by_lookup_index": first_idx, "first_token_subject_by_text": first_txt, "logs": {}}
for name, Tp in (("TX-G2XL-rome-T100", 100), ("TX-G2XL-rome-T500", 500), ("TX-G2XL-rome-T1000", 1000), ("TX-G2XL-rome-T2000", 2000)):
    Cp = C[:Tp, :Tp].clone()
    Cp.fill_diagonal_(-float("inf"))
    mg = (1.0 - Cp.max(0).values).numpy()
    flagged = [int(i) for i in np.nonzero(mg <= 0)[0]]
    o = {"n_flagged": len(flagged), "flagged": flagged, "margin_min": float(mg.min()), "margin_median": float(np.median(mg)), "against": {}}
    for nm, ref in (("exploding_norm", [i for i in EXPL if i < Tp]), ("first_token_subject", [i for i in first_idx if i < Tp])):
        tp = sorted(set(flagged) & set(ref))
        o["against"][nm] = {"reference": ref, "true_positives": tp, "false_negatives": sorted(set(ref) - set(flagged)), "false_positives": sorted(set(flagged) - set(ref)),
                            "precision": 100.0 * len(tp) / len(flagged) if flagged else None, "recall": 100.0 * len(tp) / len(ref) if ref else None,
                            "margins_of_reference": {str(i): float(mg[i]) for i in ref}}
    x = XS["logs"][name]
    o["experiment"] = {"n_flagged": x["n_flagged"], "same_flagged_set": x["flagged"] == flagged,
                       "precision_recall": {nm: [x["against"][nm]["precision"], x["against"][nm]["recall"]] for nm in ("exploding_norm", "first_token_subject")},
                       "same_precision_recall": all(x["against"][nm]["precision"] == o["against"][nm]["precision"] and x["against"][nm]["recall"] == o["against"][nm]["recall"]
                                                    for nm in ("exploding_norm", "first_token_subject")),
                       "same_references": all(x["against"][nm]["reference"] == o["against"][nm]["reference"] for nm in ("exploding_norm", "first_token_subject")),
                       "max_abs_margin_difference_on_flagged": max([abs(d["margin"] - float(mg[d["edit"]])) for d in x["flagged_detail"]], default=0.0)}
    out["logs"][name] = o
    print(name, "flagged", flagged, {k: (v["precision"], v["recall"]) for k, v in o["against"].items()}, o["experiment"]["same_flagged_set"], o["experiment"]["same_precision_recall"])
# exact ties: requests with bitwise identical solver keys (repeated request prompts) have margin 0 in exact arithmetic, so the rule
# "margin <= 0" is decided by the last bit of the arithmetic
same = {}
for i in range(T):
    same.setdefault(K[i].numpy().tobytes(), []).append(i)
groups = [g for g in same.values() if len(g) > 1]
Cf = C.clone()
Cf.fill_diagonal_(-float("inf"))
mg_pub = (1.0 - Cf.max(0).values).numpy()
Cl = Clib.clone()
Cl.fill_diagonal_(-float("inf"))
mg_lib = (1.0 - Cl.max(0).values).numpy()
tie = sorted(i for g in groups for i in g)
xf = set(XS["logs"][V.T2000]["flagged"])
mf = set(out["logs"][V.T2000]["flagged"])
strict = sorted(i for i in range(T) if mg_pub[i] < -1e-9)
allt = sorted(set(strict) | set(tie))
ties = {"groups_of_identical_solver_keys": groups, "tie_edits": tie, "n_tie_edits": len(tie),
        "margins": {str(i): {"public_factor_arithmetic": float(mg_pub[i]), "library_normalised_arithmetic": float(mg_lib[i]), "flagged_by_experiment": i in xf, "flagged_by_verifier": i in mf,
                             "request": f["requests"][i]["prompt"].format(f["requests"][i]["subject"]) + " -> " + f["requests"][i]["target_new"]["str"]} for i in tie},
        "max_abs_margin_of_tie_edits": float(max(abs(mg_pub[i]) for i in tie)) if tie else None,
        "n_flagged_with_margin_below_minus_1e-9": len(strict), "n_tie_edits_flagged_by_experiment": len(xf & set(tie)), "n_tie_edits_flagged_by_verifier": len(mf & set(tie)),
        "flag_sets_differ_only_on_tie_edits": (xf ^ mf) <= set(tie),
        "n_flagged_if_every_tie_is_flagged": len(allt),
        "precision_range_exploding": [100.0 * len(set(EXPL) & set(allt)) / len(allt), 100.0 * len(set(EXPL) & set(strict)) / len(strict)],
        "precision_range_first_token": [100.0 * len(set(first_idx) & set(allt)) / len(allt), 100.0 * len(set(first_idx) & set(strict)) / len(strict)],
        "tie_groups_inside_T1000": [g for g in groups if max(g) < 1000], "tie_groups_inside_T500": [g for g in groups if max(g) < 500]}
out["ties_T2000"] = ties
print("ties", {k: v for k, v in ties.items() if k != "margins"})
V.jdump(out, V.OUT + "/screening_T2000.json")
