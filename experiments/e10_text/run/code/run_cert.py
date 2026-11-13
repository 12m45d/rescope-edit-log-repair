"""E10 task 6: certificate check.  For every HELD-OUT benign prompt on whose pre-edit trace a calibrated row releases nothing
(cached keys of [prompt, pre-edit continuation]; outside the decision band = 'certified'; inside it = 'in band', listed apart),
the served logits of the 10-token greedy decode are compared BIT FOR BIT with the pre-edit model's logits of the same batch.

    $GQ run --mem 13000 --prio -2 --name e10c -- $PY run_cert.py --log TX-G2XL-rome-T100
Calibrated rows: where_twoblock, rescope, cosine, the R3 gate of out/best_gate_text.json; draws 0-2; on the collapse log also with
guards off.  Every violation is listed with its cause.  -> out/cert/<log>.json
"""
import argparse
import os
import time

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
args = ap.parse_args()

import e10t  # noqa: E402

B = e10t.Backbone()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from e10t import gate as G  # noqa: E402
import xmeth as XM  # noqa: E402

t0 = time.time()
best = X.jload(X.OUT + "/best_gate_text.json")["selected"]
L = e10t.open_log(args.log)
rows = L.benign("heldout")
tids = list(dict.fromkeys(r["text_id"] for r in rows))
gates = []
for guards in (True, False) if L.T > X.SPLIT else (True,):
    for lab in ("where_twoblock", "rescope", "cosine", best):
        for d in (0, 1, 2):
            M = XM.method(L, lab, d, guards)
            dec = G.decide(L, M.obj, tids, with_positions=False)
            gates.append({"tag": f"{M.label}__d{d}", "M": M, "silent": {t: not f for t, f in zip(tids, dec["fired"])}, "near": {t: bool(x) for t, x in zip(tids, dec["near"])},
                          "n_certified": 0, "n_in_band": 0, "n_equal": 0, "n_in_band_equal": 0, "violations": [], "tokens_equal_y0": 0})
    X.say(f"{L.name}: gates built (guards {guards})")
lens = [len(L.ids(t)) for t in tids]
pre_ok = 0
for bi, b in enumerate(B.batches(lens)):
    bt = [tids[i] for i in b]
    seqs = [L.ids(t) for t in bt]
    with B.served(None):
        tok0, fin0, lg0 = B.decode(seqs, return_logits=True)
    pre_ok += sum(tok0[j].tolist() == L.y0(t) for j, t in enumerate(bt))
    for g in gates:
        js = [j for j, t in enumerate(bt) if g["silent"][t]]
        if not js:
            continue
        with B.served(g["M"].obj) as srv:
            tok, fin, lg = B.decode(seqs, return_logits=True)
            flags = srv.rows()
        for j in js:
            t = bt[j]
            eq = bool(torch.equal(lg[j], lg0[j]))
            band = g["near"][t]
            g["n_in_band" if band else "n_certified"] += 1
            g["n_in_band_equal" if band else "n_equal"] += int(eq)
            g["tokens_equal_y0"] += int(tok[j].tolist() == L.y0(t))
            if not eq:
                cause = ("the hook released an edit in the live decode (key of this batch layout differs from the cached key" + (", inside the decision band)" if band else ", OUTSIDE the decision band)")
                         if flags[j]["fired"] else "logits differ although the hook released nothing")
                g["violations"].append({"text_id": t, "in_band": band, "hook_fired": bool(flags[j]["fired"]), "rel": flags[j]["rel"][:5] if flags[j]["rel"] else None,
                                        "decode_left_y0": tok0[j].tolist() != L.y0(t), "max_abs_logit_diff": float((lg[j] - lg0[j]).abs().max()), "cause": cause})
        del lg
    del lg0
    if bi % 10 == 0:
        X.say(f"{L.name}: batch {bi}")
B.check_w0()
out = {"log": L.name, "n_heldout_prompts": len(tids), "pre_edit_decode_in_these_batches_equals_stored_continuation": pre_ok, "seconds": time.time() - t0,
       "what": __doc__, "code": e10t.code_fingerprint(), "rows": {}}
for g in gates:
    out["rows"][g["tag"]] = {k: g[k] for k in ("n_certified", "n_equal", "n_in_band", "n_in_band_equal", "tokens_equal_y0")}
    out["rows"][g["tag"]].update(n_violations_certified=sum(not v["in_band"] for v in g["violations"]), n_violations_in_band=sum(v["in_band"] for v in g["violations"]),
                                 violations=g["violations"][:50])
out["total"] = {"n_checked_certified": sum(g["n_certified"] for g in gates), "n_equal_certified": sum(g["n_equal"] for g in gates),
                "n_checked_in_band": sum(g["n_in_band"] for g in gates), "n_equal_in_band": sum(g["n_in_band_equal"] for g in gates),
                "n_violations_certified": sum(sum(not v["in_band"] for v in g["violations"]) for g in gates),
                "n_violations_in_band": sum(sum(v["in_band"] for v in g["violations"]) for g in gates)}
X.jdump(out, os.path.join(X.OUT, "cert", L.name + ".json"))
X.say("done", out["total"])
