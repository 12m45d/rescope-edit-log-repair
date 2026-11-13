"""E10: numbers from the files of the runs -> out/summary/*.json (statistics), out/records/*.json (flat records).

    python summarise.py [gate] [behave] [equal] [r22]            (default: every block whose files exist)
Statistics: crossed bootstrap over calibration draws (or complaint seeds) and units (records / edits), 2000 resamples
(xcommon.xboot); gate level over the 20 draws: mean, 5th / 95th percentile over draws, and the crossed-bootstrap interval from
the per-row flags.  Nothing is selected here: every instance found is summarised.
"""
import os as _go, sys as _gs  # ARTIFACT: a failed run leaves the stored output files untouched (rescope_paths.guard_outputs)
_gd = _go.environ.get("RESCOPE_HOME") or _go.path.dirname(_go.path.abspath(__file__))
while not _go.path.exists(_go.path.join(_gd, "rescope_paths.py")) and _go.path.dirname(_gd) != _gd: _gd = _go.path.dirname(_gd)
_gs.path.insert(0, _gd); import rescope_paths as _RPG; _RPG.guard_outputs(__file__, __name__)
import os
import sys

import numpy as np

import xcommon as X

S = X.SPLIT
SUM = os.path.join(X.OUT, "summary")


# ----------------------------------------------------------------------------- gate level
def gate_block(log):
    p = os.path.join(X.OUT, "gate", log + ".json")
    if not os.path.exists(p):
        return None
    g = X.jload(p)
    fp = p.replace(".json", "_flags.npz")
    sp = os.path.join(SUM, "gate_" + log + ".json")
    if not os.path.exists(fp) and os.path.exists(sp) and '"xboot.' in open(sp).read():
        # ARTIFACT: the stored summary of this log has intervals from the flags: stop instead of dropping them (a log without a flags file, as the development log, has none)
        raise SystemExit(f"summarise: input missing: {fp}\n  The per-record gate flags are in results_rows.tar.gz of the release (unpack it into the clone: "
                         "tar -xzf results_rows.tar.gz -C <clone>).  Nothing was written.")
    fl = np.load(fp) if os.path.exists(fp) else None
    rec = X.Records("gate_" + log)
    out = {"meta": g["meta"], "rows": {}}
    n = g["meta"]["n"]
    for lab, modes in g["rows"].items():
        for mode, e in modes.items():
            pd, cal = e["per_draw"], e["cal"]
            ds = sorted(pd, key=int)
            o = {"n_draws": len(ds)}
            variant = "" if mode == "gon" else "guards off"

            def series(name, fn):
                v = [fn(pd[d]) for d in ds]
                s = X.over_draws(v)
                o[name] = s
                return s
            keys = [("heldout.fired", lambda x: x["heldout"]["fired"]), ("heldout.fired_clean", lambda x: x["heldout"]["fired_clean"]),
                    ("heldout.fired_near_scope", lambda x: x["heldout"]["fired_near_scope"]), ("heldout.fired_last_only", lambda x: x["heldout"]["fired_last_only"]),
                    ("heldout.lineage_width", lambda x: x["heldout"]["lineage_width_mean_if_fired"]),
                    ("paraphrase.own_released", lambda x: x["paraphrase"]["own_released"]), ("paraphrase.fired", lambda x: x["paraphrase"]["fired"]),
                    ("paraphrase.only_other_released", lambda x: x["paraphrase"]["only_other_released"]),
                    ("neighborhood.fired", lambda x: x["neighborhood"]["fired"]), ("neighborhood.fired_clean", lambda x: x["neighborhood"]["fired_clean"]),
                    ("neighborhood.own_released", lambda x: x["neighborhood"]["own_released"]),
                    ("request.own_released", lambda x: x["request"]["own_released"]), ("request.other_released", lambda x: x["request"]["other_released"])]
            if g["meta"].get("split") is not None:
                for part in ("paraphrase", "neighborhood", "request"):
                    for seg in ("before", "from"):
                        keys.append((f"{part}.{seg}.own_released", lambda x, part=part, seg=seg: x[part][seg]["own_released"]))
                        keys.append((f"{part}.{seg}.fired", lambda x, part=part, seg=seg: x[part][seg]["fired"]))
            for name, fn in keys:
                s = series(name, fn)
                rec.add(log, lab, "gate." + name + "@mean_over_draws", s["mean"], variant=variant, alpha=X.ALPHA if len(ds) > 1 else None, n=X.N_CAL if len(ds) > 1 else None,
                        pool=name.split(".")[0], ci_lo=s["p05"], ci_hi=s["p95"], n_units=s["n_draws"])
            for ck in ("eta", "n_edits_capped", "threshold_median", "threshold_max", "n_threshold_inf", "n_own_keys_silenced"):
                v = [(cal[d] or {}).get(ck) for d in ds]
                v = [None if isinstance(x, str) else x for x in v]
                if any(x is not None for x in v):
                    o["cal." + ck] = X.over_draws(v)
                    rec.add(log, lab, "gate.cal." + ck + "@mean_over_draws", o["cal." + ck]["mean"], variant=variant, ci_lo=o["cal." + ck]["p05"], ci_hi=o["cal." + ck]["p95"], n_units=len(ds))
            ce = [(cal[d] or {}).get("capped_edits") for d in ds]
            if any(x is not None for x in ce):
                allc = sorted({i for x in ce if x for i in x})
                o["capped_edits_union"] = allc[:100]
                o["n_capped_edits_union"] = len(allc)
            if fl is not None:                                           # crossed bootstrap over draws and records
                for part, nm in (("heldout", "heldout.fired"), ("paraphrase", "paraphrase.own_released"), ("neighborhood", "neighborhood.fired"), ("request", "request.own_released")):
                    Y = np.stack([np.unpackbits(fl[f"{lab}|{mode}|{d}|{part}"])[:n[part]] for d in ds]).astype(np.float64)
                    b = X.xboot(Y, fl[f"units|{part}"])
                    o["xboot." + nm] = b
                    rec.stat(log, lab, "gate." + nm, b, variant=variant, alpha=X.ALPHA if len(ds) > 1 else None, n=X.N_CAL if len(ds) > 1 else None, pool=part)
            out["rows"][lab + ("" if mode == "gon" else X_GOFF)] = o
    rec.save()
    X.jdump(out, os.path.join(SUM, "gate_" + log + ".json"))
    return out


X_GOFF = "@goff"


# ----------------------------------------------------------------------------- behaviour (rows of run_behave / run_equal)
def instances(log, sub):
    d = os.path.join(X.OUT, sub, log)
    if not os.path.isdir(d):
        return {}
    inst = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f.startswith("_") or f.endswith(".public.json") or f.endswith(".infeasible.json"):
            continue
        tag = f[:-5]
        rp = os.path.join(X.OUT, "rows" if sub == "behave" else "rows_" + sub, log, tag + ".jsonl.gz")
        if not os.path.exists(rp):
            continue
        meta = X.jload(os.path.join(d, f))
        if sub in ("behave", "stream"):
            key, di = meta["label"], meta["draw"]
        else:
            a, b = tag.split("__")
            key, di = a + "|" + b.split("_s")[0], int(b.split("_s")[1])
        pp = os.path.join(d, tag + ".public.json")
        inst.setdefault(key, {})[-1 if di is None else di] = {"tag": tag, "meta": meta, "rows": rp, "public": X.jload(pp) if os.path.exists(pp) else None}
    return inst


_PUB = {}


def public_per_record(case_dir, records_case_ids):
    """Per-record success values of the public summariser's arithmetic (for intervals): rewrite, paraphrase, neighbourhood."""
    key = case_dir
    if key not in _PUB:
        out = {}
        for f in os.listdir(case_dir):
            if f.startswith("case_"):
                c = X.jload(os.path.join(case_dir, f))["post"]
                out[int(f[5:-5])] = (float(np.mean([x["target_true"] > x["target_new"] for x in c["rewrite_prompts_probs"]])),
                                     float(np.mean([x["target_true"] > x["target_new"] for x in c["paraphrase_prompts_probs"]])),
                                     float(np.mean([x["target_true"] < x["target_new"] for x in c["neighborhood_prompts_probs"]])))
        _PUB[key] = out
    return np.array([_PUB[key][c] for c in records_case_ids])


def behave_block(log, sub="behave"):
    inst = instances(log, sub)
    if not inst:
        return None
    collapse = int(log.split("-T")[-1]) > S
    dep_rows = None
    dp = os.path.join(X.OUT, "rows", log, "deployed__dx.jsonl.gz")
    if os.path.exists(dp):
        dep_rows = {r["prompt_id"]: r for r in X.load_rows(dp)}
    rec = X.Records(f"{sub}_{log}")
    out = {}
    for key in sorted(inst):
        ds = sorted(inst[key])
        rows = [X.load_rows(inst[key][d]["rows"]) for d in ds]

        def rid_(rs):
            # identity of a row across draws / seeds: the prompt; the reported complaints of a seed ('extra' rows) differ between
            # seeds and are paired by their index in the seed's complaint set (their unit is that index)
            out_, j = [], 0
            for r in rs:
                if r.get("part") == "extra":
                    out_.append(f"extra|{j}")
                    r["case_id"] = j
                    j += 1
                else:
                    out_.append(r["prompt_id"])
            return out_
        idl = [rid_(rs) for rs in rows]
        ids0 = idl[0]
        for k in range(1, len(rows)):
            if idl[k] != ids0:
                m = dict(zip(idl[k], rows[k]))
                rows[k] = [m[i] for i in ids0]
        R0 = rows[0]
        meta0 = inst[key][ds[0]]["meta"]
        base = key.split("@")[0].split("|")[0]
        variant = ("guards off" if key.endswith(X_GOFF) else "") if sub == "behave" else key.split("|")[1]
        cal = sub == "behave" and ds != [-1]
        o = {"n_draws": len(ds), "draws": ds, "kind": meta0["kind"], "metrics": {}, "public": {}, "per_draw": {}}

        def stat(name, sel, fn, unit, pool):
            if not sel:
                return
            Y = [[fn(r[i]) for i in sel] for r in rows]
            b = X.xboot(Y, [R0[i][unit] for i in sel])
            o["metrics"][name] = b
            rec.stat(log, base, name, b, variant=variant, alpha=X.ALPHA if cal else None, n=X.N_CAL if cal else None, pool=pool)
            if len(ds) > 1:
                for d, y in zip(ds, Y):
                    v = [x for x in y if x is not None]
                    o["per_draw"].setdefault(name, {})[d] = 100.0 * float(np.mean(v)) if v else None
        fb = lambda k: (lambda r: None if r.get(k) is None else float(r[k]))        # noqa: E731
        # (i) edits
        for kind in ("request", "paraphrase"):
            segs = [("", lambda r: True)] + ([(".before", lambda r: r["pos"] < S), (".from", lambda r: r["pos"] >= S)] if collapse else [])
            for sfx, pred in segs:
                sel = [i for i, r in enumerate(R0) if r["kind"] == kind and "new_exact" in r and r.get("part") is None and pred(r)]
                pool = f"edits{sfx}"
                stat(f"edit.{kind}.exact{sfx}", sel, fb("new_exact"), "pos", pool)
                stat(f"edit.{kind}.token{sfx}", sel, fb("new_tok"), "pos", pool)
                stat(f"edit.{kind}.true_exact{sfx}", sel, fb("true_exact"), "pos", pool)
                if R0[sel[0]].get("fired_dec") is not None if sel else False:
                    stat(f"edit.{kind}.own_released{sfx}", sel, fb("own_released_dec"), "pos", pool)
                    stat(f"edit.{kind}.other_released{sfx}", sel, fb("other_released_dec"), "pos", pool)
                    stat(f"edit.{kind}.fired{sfx}", sel, fb("fired_dec"), "pos", pool)
                if dep_rows is not None and sel:
                    de = lambda r: bool(dep_rows[r["prompt_id"]]["new_exact"])      # noqa: E731
                    stat(f"edit.{kind}.kept{sfx}", sel, lambda r: float(de(r) and r["new_exact"]), "pos", pool)
                    stat(f"edit.{kind}.gained{sfx}", sel, lambda r: float((not de(r)) and r["new_exact"]), "pos", pool)
                    stat(f"edit.{kind}.lost{sfx}", sel, lambda r: float(de(r) and not r["new_exact"]), "pos", pool)
        # (iii) D1
        probes = [("heldout", lambda r: r["set"] == "heldout" and r.get("part") is None), ("neighborhood", lambda r: r["kind"] == "neighborhood" and r.get("part") is None),
                  ("complaints", lambda r: r.get("part") == "extra")]
        for part, pred in probes:
            subsets = [("", lambda r: True), (".clean", lambda r: bool(r.get("clean", True))), (".near_scope", lambda r: not r.get("clean", True))]
            if collapse and part == "neighborhood":
                subsets += [(".before", lambda r: r["pos"] < S), (".from", lambda r: r["pos"] >= S)]
            for sfx, p2 in subsets:
                if part == "complaints" and sfx:
                    continue
                sel = [i for i, r in enumerate(R0) if pred(r) and r["valid"] and p2(r)]
                if not sel:
                    continue
                pool = part + sfx
                for nm, fld in (("D1_tf", "D1_tf"), ("D1_dec", "D1_dec"), ("kept_tf", "tok_loc"), ("kept_dec", "dec_loc"), ("first_changed_tf", "first_changed_tf"),
                                ("first_changed_dec", "first_changed_dec")):
                    stat(f"{part}{sfx}.{nm}", sel, fb(fld), "case_id", pool)
                if R0[sel[0]].get("fired") is not None:
                    stat(f"{part}{sfx}.fired", sel, fb("fired"), "case_id", pool)
                    if part == "neighborhood":
                        stat(f"{part}{sfx}.own_released", sel, fb("own_released"), "case_id", pool)
                o["metrics"][f"{part}{sfx}.n"] = {"value": len(sel)}
        # (ii) public scores
        pubs = [inst[key][d]["public"] for d in ds]
        if all(p is not None and "all" in p for p in pubs):
            for seg in ("all", "before", "from"):
                if seg not in pubs[0]:
                    continue
                for k in ("ES", "PS", "NS", "EM", "PM", "NM", "S"):
                    v = [p[seg][k] for p in pubs]
                    o["public"][f"{seg}.{k}"] = {"value": float(np.mean(v)), "per_draw": v, "n_records": pubs[0][seg]["n"]}
            import rescope_paths as _RPX                                # ARTIFACT: released records hold ${RESCOPE_ROOT}/... for the directory of the case files
            cds = [_RPX.expand(p["all"]["case_files"]) for p in pubs]
            ok = all(os.path.isdir(c) for c in cds)
            if not ok:                                                  # ARTIFACT: stop instead of dropping the intervals of ES / PS / NS
                raise SystemExit("summarise: input missing: the per-record case files " + ", ".join(c for c in cds if not os.path.isdir(c))
                                 + "\n  They are in results_rows.tar.gz of the release (unpack it into the clone: tar -xzf results_rows.tar.gz -C <clone>).  Nothing was written.")
            if ok:
                first = pubs[0]["all"]["n"]
                cids = list(range(first))                               # stream records 0..n-1 (case id = record index)
                A = np.stack([public_per_record(c, cids) for c in cds])           # (draws, records, 3)
                for seg, lo, hi in [("all", 0, first)] + ([("before", 0, S), ("from", S, first)] if collapse and first > S else []):
                    for j, k in enumerate(("ES", "PS", "NS")):
                        b = X.xboot(A[:, lo:hi, j], cids[lo:hi])
                        e = o["public"][f"{seg}.{k}"]
                        e.update(ci_lo=b["ci_lo"], ci_hi=b["ci_hi"], n_units=b["n_units"], recomputed=b["value"], agrees_with_public_summariser=bool(abs(b["value"] - e["value"]) < 0.0051 * max(1, 1)))      # the public summariser rounds to 2 decimals
                        rec.add(log, base, f"public.{k}" + ("" if seg == "all" else "." + seg), e["value"], variant=variant, alpha=X.ALPHA if cal else None, n=X.N_CAL if cal else None,
                                pool="records" if seg == "all" else "records." + seg, ci_lo=b["ci_lo"], ci_hi=b["ci_hi"], n_units=b["n_units"])
            for seg in ("all", "before", "from"):
                for k in ("EM", "PM", "NM", "S"):
                    if f"{seg}.{k}" in o["public"]:
                        rec.add(log, base, f"public.{k}" + ("" if seg == "all" else "." + seg), o["public"][f"{seg}.{k}"]["value"], variant=variant, pool="records", n_units=o["public"][f"{seg}.{k}"]["n_records"])
        # method records and cost
        mrec = [inst[key][d]["meta"] for d in ds]
        c0 = [m["record"].get("cal", {}) if isinstance(m.get("record"), dict) else {} for m in mrec]
        o["cal"] = {"eta": [c.get("eta") for c in c0], "n_edits_capped": [c.get("n_edits_capped") for c in c0],
                    "capped_edits": [(c.get("capped_edits") if not isinstance(c.get("capped_edits"), dict) else c["capped_edits"]["first"]) for c in c0],
                    "own_not_candidate": [(m["record"].get("own_requests") or {}).get("n_not_candidate") for m in mrec],
                    "own_not_released": [(m["record"].get("own_requests") or {}).get("n_not_released") for m in mrec]}
        o["cost"] = {k: float(np.sum([m["cost"][k] for m in mrec])) for k in ("sequences_run", "silent", "from_store", "seconds", "hook_mismatch", "hook_mismatch_dec", "near")}
        o["meta"] = [m.get("meta") for m in mrec] if sub != "behave" else None
        out[key] = o
    rec.save()
    X.jdump(out, os.path.join(SUM, f"{sub}_{log}.json"))
    return out


if __name__ == "__main__":
    want = sys.argv[1:] or ["gate", "behave", "equal"]
    if "gate" in want:
        for log in (X.DEV,) + X.LOGS:
            if gate_block(log) is not None:
                X.say("gate", log)
    for sub in ("behave", "equal"):
        if sub in want:
            for log in X.BEHAVE_LOGS:
                if behave_block(log, sub) is not None:
                    X.say(sub, log)


# ----------------------------------------------------------------------------- R22: relational maintenance workload
def r22_block(log):
    p = os.path.join(X.OUT, "r22", log + ".json")
    if not os.path.exists(p):
        return None
    o = X.jload(p)
    fl = X.jload(os.path.join(X.OUT, "r22", log + "_flags.json"))
    collapse = o["meta"]["split"] is not None
    rec = X.Records("r22_" + log)
    groups = {}
    for r in o["rows"]:
        key = (r["label"], r.get("m"))
        groups.setdefault(key, []).append(r)
    out = {"meta": o["meta"], "serving_overhead": o.get("serving_overhead"), "complaints": o.get("complaints"), "rows": []}
    for (label, mm), rs in groups.items():
        feas = [r for r in rs if r.get("scores") is not None]
        g = {"label": label, "m": mm, "class": rs[0]["class"], "evidence": rs[0]["evidence"] if mm is None else f"log + {mm} complained quer{'y' if mm == 1 else 'ies'} (diagnosis charged)",
             "n_instances": len(rs), "n_infeasible": len(rs) - len(feas), "instances": [r.get("draw") if r.get("draw") is not None else r.get("seed") for r in rs],
             "instance_kind": "calibration draws" if rs[0].get("draw") is not None else ("complaint seeds" if mm is not None else "-")}
        variant = "" if mm is None else f"m={mm}"

        def add(name, st):
            g[name] = st
            rec.stat(log, label, "r22." + name, st, variant=variant, pool="test half" if "retention" not in name else "edited tuples")
        if feas:
            F = [fl[r["model"]] for r in feas]
            rid = np.array(F[0]["rid"])
            ed = np.array(F[0]["edited"]).astype(bool)
            pos = np.array([-1 if x is None else x for x in F[0]["pos"]])
            sp = np.stack([np.array(f["same_as_pre"], dtype=np.float64) for f in F])
            ai = np.stack([np.array(f["as_intended"], dtype=np.float64) for f in F])
            add("reference_preservation", X.xboot(sp[:, ~ed], rid[~ed]))
            add("retention", X.xboot(ai[:, ed], rid[ed]))
            if collapse:
                add("retention_before", X.xboot(ai[:, ed & (pos < S)], rid[ed & (pos < S)]))
                add("retention_from", X.xboot(ai[:, ed & (pos >= S)], rid[ed & (pos >= S)]))
            for k in ("views_exact_pct", "aggregates_exact_pct", "queries_exact_pct", "aggregate_tv_mean", "view_wrong_share_mean_pct"):
                v = [r["scores"][k] for r in feas]
                g[k] = {"value": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
                rec.add(log, label, "r22." + k, g[k]["value"], variant=variant, pool="test half", ci_lo=g[k]["min"], ci_hi=g[k]["max"], n_units=feas[0]["scores"]["n_views"])
        for k in ("diagnosis_passes", "repair_passes", "repair_seconds", "statements_changed"):
            v = [r["cost"][k] for r in rs]
            g[k] = {"value": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
            rec.add(log, label, "r22.cost." + k, g[k]["value"], variant=variant, pool="cost", ci_lo=g[k]["min"], ci_hi=g[k]["max"], n_units=len(rs))
        el = [r["edits_lost"] for r in rs if r.get("edits_lost") is not None]
        if el:
            g["edits_lost"] = {k: float(np.mean([e.get(k, 0) for e in el])) for k in ("n", "before", "from") if any(k in e for e in el)}
            g["edits_lost_range"] = [int(min(e["n"] for e in el)), int(max(e["n"] for e in el))]
            rec.add(log, label, "r22.edits_lost", g["edits_lost"]["n"], variant=variant, pool="log", ci_lo=g["edits_lost_range"][0], ci_hi=g["edits_lost_range"][1], n_units=len(el))
        for k in ("complaints_refused", "complaints_still_released", "complaints_in_band", "n_tuple_complaints", "capped"):
            v = [r[k] for r in rs if r.get(k) is not None]
            if v:
                g[k] = float(np.mean(v))
        ts = [r["repair"]["t_star"] for r in rs if isinstance(r.get("repair"), dict) and "t_star" in r["repair"]]
        if ts:
            g["t_star"] = ts
        hm = [r["serve"].get("hook_mismatch", 0) for r in rs if isinstance(r.get("serve"), dict)]
        g["hook_mismatch"] = int(sum(hm)) if hm else None
        g["guards"] = rs[0].get("guards")
        out["rows"].append(g)
    rec.save()
    X.jdump(out, os.path.join(SUM, f"r22_{log}.json"))
    return out


if __name__ == "__main__" and (not sys.argv[1:] or "r22" in sys.argv[1:]):
    for log in X.BEHAVE_LOGS:
        if r22_block(log) is not None:
            X.say("r22", log)
