"""E10 metric (iv): gate level of every row on one log over the calibration draws (key cache only; products on --device).

    $GQ run --mem 9000 --prio -2 --name e10g_T500 -- $PY run_gate.py --log TX-G2XL-rome-T500
    -> out/gate/<log>.json : rows[label][mode]{per_draw: {draw: stats per part}, cal: {draw: record}}; mode 'gon' = guards (the frozen
       default), 'goff' = guards off (allow_silence; R21 b).  Rows without calibration have the single draw '0'.
Calibrated rows (two-block, alpha = 0.05, n = 200, draws 0..19): rescope, where_twoblock (same thresholds as rescope, fire 'all'),
cosine, and the R3 gates cos_centred, cos_null, white_k, whitecos_k.  Without calibration: cosine_published, exact, where_logonly
(guards on / off), lookup_subject (string rule; no key involved).
Thresholds are built on the CPU exactly as the behavioural rows build them (e10t.methods.calibrate); only the evaluation of the
parts runs on the device.
"""
import argparse
import os
import time

import xcommon as X

ap = argparse.ArgumentParser()
ap.add_argument("--log", required=True)
ap.add_argument("--draws", type=int, default=X.N_DRAWS)
ap.add_argument("--device", default="cuda")
ap.add_argument("--labels", default="")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import e10t  # noqa: E402
from e10t import methods as MT  # noqa: E402
import xmeth as XM  # noqa: E402
import xgate as XG  # noqa: E402

if args.device != "cpu" and "GQ_MEM" in os.environ:
    torch.cuda.set_per_process_memory_fraction((int(os.environ["GQ_MEM"]) - 600) * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory, 0)

out_path = args.out or os.path.join(X.OUT, "gate", args.log + ".json")
t00 = time.time()
L = e10t.open_log(args.log)
lay = XG.Layout(L)
CAL = ("rescope", "cosine") + XM.R3_LABELS
labels = [x for x in args.labels.split(",") if x] or list(CAL)
MODES = (("gon", True), ("goff", False))
groups, cal = [], {}
keep = ("eta", "n_edits_capped", "capped_edits", "bound_false_firing", "n_own_keys_silenced", "n_guards_silenced", "eta_infinite")


def crec(rec, S2):
    c = {k: rec.get(k) for k in keep}
    th = S2.theta().numpy()
    f = th[np.isfinite(th)]
    c.update(threshold_median=float(np.median(f)) if f.size else None, threshold_max=float(f.max()) if f.size else None,
             n_threshold_inf=int(np.isposinf(th).sum()), guard_margin_lowered=(rec.get("guard_margin") or {}).get("n_thresholds_lowered"))
    return c


for lab in labels:
    t0 = time.time()
    if lab == "cos_null":
        for d in range(args.draws):
            vs = []
            Sb = XM.r3_base(L, lab, d)                                     # detectors of this draw (null sample of the draw)
            for mode, g in MODES:
                S2, rec = MT.calibrate(L, Sb, "two_block", X.ALPHA, X.N_CAL, d, allow_silence=not g)
                vs.append(XG.variant_of(f"{lab}|{mode}|{d}", S2))
                cal[f"{lab}|{mode}|{d}"] = crec(rec, S2)
            groups.append((XG.light(Sb), vs))
            del Sb
    else:
        vs = []
        for d in range(args.draws):
            for mode, g in MODES:
                S2, rec = XM.calibrated(L, lab, d, g)
                vs.append(XG.variant_of(f"{lab}|{mode}|{d}", S2))
                cal[f"{lab}|{mode}|{d}"] = crec(rec, S2)
                if lab == "rescope":                                       # ReScope-W with two-block thresholds: the same thresholds, every candidate releases
                    vs.append(XG.variant_of(f"where_twoblock|{mode}|{d}", S2, fire="all"))
                    cal[f"where_twoblock|{mode}|{d}"] = cal[f"{lab}|{mode}|{d}"]
        if lab == "rescope":
            for mode, g in MODES:
                M = XM.method(L, "where_logonly", guards=g)
                vs.append(XG.variant_of(f"where_logonly|{mode}|0", M.obj, fire="all"))
                c = M.rec["cal"]
                cal[f"where_logonly|{mode}|0"] = {"n_edits_capped": c.get("n_edits_capped"), "capped_edits": c.get("capped_edits"),
                                                  "n_own_keys_silenced_by_literal_rule": c.get("n_own_keys_silenced_by_literal_rule"),
                                                  "threshold_median": M.rec["thresholds"]["median"], "threshold_max": M.rec["thresholds"]["max"]}
        if lab == "cosine":
            for nm in ("cosine_published", "exact"):
                M = XM.method(L, nm)
                vs.append(XG.variant_of(f"{nm}|gon|0", M.obj))
                cal[f"{nm}|gon|0"] = {"threshold_median": M.rec["thresholds"]["median"]}
        groups.append((XG.light(XM.base_of(L, lab)), vs))
    X.say(f"{args.log}: thresholds of {lab} built ({time.time() - t0:.0f}s)")

t1 = time.time()
res = XG.evaluate(L, lay, groups, args.device, progress=10)
X.say(f"{args.log}: evaluation {time.time() - t1:.0f}s")
split = X.SPLIT if L.T > X.SPLIT else None
rows = {}
for tag, r in res.items():
    lab, mode, d = tag.split("|")
    e = rows.setdefault(lab, {}).setdefault(mode, {"per_draw": {}, "cal": {}})
    e["per_draw"][d] = XG.stats(L, lay, r, split)
    e["cal"][d] = cal.get(tag)

# per-row flags of the three gate-level rates (for the crossed bootstrap over draws and records), bit-packed
FL = {}


def keep_flags(tag, r):
    FL[f"{tag}|heldout"] = np.packbits(r["fired"][lay.text["heldout"]])
    FL[f"{tag}|neighborhood"] = np.packbits(r["fired"][lay.text["neighborhood"]])
    FL[f"{tag}|paraphrase"] = np.packbits(r["paraphrase"]["own_rel"])
    FL[f"{tag}|request"] = np.packbits(r["request"]["own_rel"])


for tag, r in res.items():
    keep_flags(tag, r)
for p in lay.parts:
    FL[f"units|{p}"] = np.array([int(row["case_id"]) for row in lay.rows[p]], dtype=np.int64)
    FL[f"own|{p}"] = lay.own[p]
    FL[f"clean|{p}"] = lay.clean[p]

# lookup_subject: the string rule on the prompt (no key)
Mk = XM.lookup_method(L)
r = {"fired": np.zeros(len(lay.tids), bool), "fired_prompt": np.zeros(len(lay.tids), bool), "last_only": np.zeros(len(lay.tids), bool),
     "n_cand": np.zeros(len(lay.tids), np.int32), **{p: {k: np.zeros(len(lay.rows[p]), bool) for k in ("own_cand", "own_rel", "oth_rel")} for p in XG.OWN_PARTS}}
plans = [XM.lookup_text_plan(L, Mk, t) for t in lay.tids]
for i, pl in enumerate(plans):
    r["fired"][i] = r["fired_prompt"][i] = bool(pl)
    r["n_cand"][i] = len({e for _, e in pl})
for p in XG.OWN_PARTS:
    for j, (ti, own) in enumerate(zip(lay.text[p], lay.own[p])):
        eds = {e for _, e in plans[ti]}
        r[p]["own_rel"][j] = r[p]["own_cand"][j] = own in eds
        r[p]["oth_rel"][j] = bool(eds - {own})
rows["lookup_subject"] = {"gon": {"per_draw": {"0": XG.stats(L, lay, r, split)}, "cal": {"0": None}}}
keep_flags("lookup_subject|gon|0", r)
np.savez_compressed(out_path.replace(".json", "_flags.npz"), **FL)
own_at_solver = sum((L.solver_pos[i], i) in XM.lookup_text_plan(L, Mk, L.rows["request"][i]["text_id"]) for i in range(L.T))

X.jdump({"meta": {"log": L.name, "T": L.T, "describe": L.describe(), "alpha": X.ALPHA, "n": X.N_CAL, "n_draws": args.draws, "device": args.device,
                  "code": e10t.code_fingerprint(), "split": split, "seconds": time.time() - t00, "r3_stats": XM.r3_stats(L)[1] if any(x in XM.R3_LABELS for x in labels) else None,
                  "lookup_own_request_released_at_solver_position": own_at_solver,
                  "n": {p: len(lay.rows[p]) for p in lay.parts}}, "rows": rows}, out_path)
X.say(f"{args.log}: gate level written to {out_path} ({time.time() - t00:.0f}s)")
