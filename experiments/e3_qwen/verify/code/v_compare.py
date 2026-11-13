"""Verification of E3, check 2 (CPU): my own rows (v_gpu.py: plain Evaluator, own captures, own thresholds) against the experiment's files.

    python v_compare.py [--sub gpu|smoke]

Writes out/compare.json (every number of the comparison) and out/compare_table.txt.  Reads, of the experiment: out/rows/<log>__<label>__d0__tf
(.tfp) .jsonl.gz, out/gate/<log>/<label>__reg.json (draw 0).  Coverage of rephrases on ALL edits (Qwen): my thresholds on the cached keys
(raw files of 10_final/cache/qwen25vl), after a bitwise check of the cache against the keys captured here.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import argparse
import gzip
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
FINAL = f"{_RP.ROOT}/02_research/experiments/10_final"
ROOT = FINAL + "/e3_qwen/verify"
TESTBED = f"{_RP.ROOT}/02_research/experiments/06_testbed_v3"
LOGS = ("TB-Q500-ae-l2_3", "TB-Q500-memitw", "TB-Q2000-memitw", "TB-L500-ae-l2_3", "TB-L500-memitw", "TB-L1000-memitw")
METHODS = ("rescope", "rescope_null", "cosine_shared")


def exp_dir(log):
    return FINAL + ("/e3_qwen" if log.startswith("TB-Q") else "/e3_llava") + "/out"


def jl(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as f:
        return [json.loads(x) for x in f]


def pct(a):
    a = list(a)
    return 100.0 * sum(bool(x) for x in a) / len(a) if a else None


_CACHE = {}


def cache(bb):
    if bb not in _CACHE:
        d = f"{FINAL}/cache/{bb}"
        ix = jl(d + "/index.jsonl")
        _CACHE[bb] = ({r["sid"]: r for r in ix}, torch.load(d + "/keys_tf.pt", mmap=True))
    return _CACHE[bb]


def csid(fam, sid):
    return f"{fam}:{sid}" if sid.startswith(("edit:", "probe:")) else sid


def flagdiff(mine, theirs, keys):
    """Row-level agreement on the rows of `mine` (joined by sid)."""
    t = {r["sid"]: r for r in theirs}
    out = {"n": len(mine), "n_missing_in_experiment": sum(r["sid"] not in t for r in mine)}
    for k in keys:
        out["n_differ_" + k] = sum(bool(r.get(k)) != bool(t[r["sid"]].get(k)) for r in mine if r["sid"] in t)
    return out, t


def one_log(log, sub):
    d = f"{ROOT}/out/{sub}/{log}"
    if not os.path.exists(d + "/edits.json"):
        return {"log": log, "missing": True}
    bb, fam = ("qwen25vl", "Q") if log.startswith("TB-Q") else ("llava", "L")
    E = exp_dir(log)
    g = json.load(open(d + "/gates.json"))
    out = {"log": log, "T": g["T"], "methods": {}, "n_repair_valid": g["n_repair_valid"]}
    KS = torch.load(d + "/keys.pt")
    sidx, Ktf = cache(bb)
    # ---- cache check: every key captured here against the cached teacher-forced key of the same sequence (bitwise)
    cc = {}
    for nm, sids in (("own_rel", [f"edit:rel:{i}" for i in KS["own_edit"]]), ("calibration", KS["cal_sid"]), ("null_sample", KS["null_sid"]), ("rephrase_sample", KS["cov_sid"])):
        K = KS[{"own_rel": "own", "calibration": "cal", "null_sample": "null", "rephrase_sample": "cov"}[nm]]
        if not len(sids):
            continue
        rows = [sidx[csid(fam, s)]["row_tf"] for s in sids]
        C = Ktf[torch.tensor(rows)]
        eq = (C == K).all(1)
        cc[nm] = {"n": len(sids), "n_bitwise_equal": int(eq.sum()), "max_abs_diff": float((C - K).abs().max())}
    out["cache_check"] = cc
    # ---- thresholds: mine against the experiment's draw-0 gate cell
    for m in METHODS:
        c = json.load(open(f"{E}/gate/{log}/{m}__reg.json"))["cells"]["0"]
        th = c.get("thresholds") or c["cal"].get("thresholds")
        cm = g["cmp"][m]
        o = {"eta_mine": cm["eta_mine"], "eta_library_on_my_keys": cm["eta_lib"], "eta_experiment": c["cal"]["eta"], "n_capped_mine": cm["n_capped_mine"],
             "n_capped_experiment": c["n_capped"], "thr_min_med_max_mine": cm["thr_min_med_max"], "thr_min_med_max_experiment": [th["min"], th["median"], th["max"]],
             "max_abs_thr_diff_mine_vs_library": cm["max_abs_thr_diff"], "bound": cm["bound"]}
        o["eta_abs_diff"] = abs(o["eta_mine"] - o["eta_experiment"])
        o["thr_summary_max_abs_diff"] = float(np.abs(np.array(o["thr_min_med_max_mine"]) - np.array(o["thr_min_med_max_experiment"])).max())
        out["methods"][m] = {"thresholds": o}
    # ---- behaviour: my plain-Evaluator rows against the experiment's rows
    for m in METHODS:
        O = out["methods"][m]
        theirs = jl(f"{E}/rows/{log}__{m}__d0__tf.jsonl.gz")
        tp = [r for r in theirs if r["kind"] == "probe"]
        te = [r for r in theirs if r["kind"] == "edit"]
        mp = jl(f"{d}/rows__{m}__probe_tf.jsonl.gz")
        me = jl(f"{d}/rows__{m}__edit_tf.jsonl.gz")
        fd, t = flagdiff(mp, tp, ("fired", "D1", "exact"))
        on = [t[r["sid"]] for r in mp if r["sid"] in t]
        O["probes"] = dict(fd, n_experiment_all=len(tp), fired_mine=pct(r["fired"] for r in mp), fired_experiment_same_rows=pct(r["fired"] for r in on),
                           D1_mine=pct(r["D1"] for r in mp), D1_experiment_same_rows=pct(r["D1"] for r in on),
                           fired_experiment_all_rows=pct(r["fired"] for r in tp), D1_experiment_all_rows=pct(r["D1"] for r in tp),
                           n_D1_without_firing_mine=sum(bool(r["D1"]) and not r["fired"] for r in mp))
        fd, t = flagdiff(me, te, ("exact", "fired", "own_released_last"))
        O["edits"] = dict(fd, n_edits=len(me) // 3)
        for k in ("rel", "tgen", "igen"):
            a = [r for r in me if r["metric"] == k]
            O["edits"][k] = {"exact_mine": pct(r["exact"] for r in a), "exact_experiment_same_edits": pct(t[r["sid"]]["exact"] for r in a),
                             "released_own_mine": pct(r["own_released_last"] for r in a),
                             "exact_experiment_all_edits": pct(r["exact"] for r in te if r["metric"] == k)}
        pp = f"{d}/rows__{m}__probe_prompt.jsonl.gz"
        if os.path.exists(pp):                                         # LLaVA: the decode's position on the shifted rows
            mq = jl(pp)
            tq = [r for r in jl(f"{E}/rows/{log}__{m}__d0__tfp.jsonl.gz") if r["kind"] == "probe"]
            fd, t = flagdiff(mq, tq, ("fired", "D1"))
            tf_of = {r["sid"]: r for r in mp}
            O["probes_prompt_position"] = dict(fd, n_shifted=len(mq), n_fired_differs_from_tf_mine=sum(bool(r["fired"]) != bool(tf_of[r["sid"]]["fired"]) for r in mq),
                                               n_D1_differs_from_tf_mine=sum(bool(r["D1"]) != bool(tf_of[r["sid"]]["D1"]) for r in mq))
            rep = {r["sid"]: r for r in mq}
            allp = [rep.get(r["sid"], r) for r in mp]
            tfx = {r["sid"]: r for r in tp}
            tq_ = {r["sid"]: r for r in tq}
            alle = [tq_.get(r["sid"], tfx[r["sid"]]) for r in mp if r["sid"] in tfx]
            O["probes_prompt_position"].update(fired_mine=pct(r["fired"] for r in allp), D1_mine=pct(r["D1"] for r in allp),
                                               fired_experiment=pct(r["fired"] for r in alle), D1_experiment=pct(r["D1"] for r in alle))
    # ---- coverage of rephrases on ALL edits, gate level: my thresholds on the cached keys
    G = torch.load(d + "/gates.pt")
    fac = torch.load(f"{TESTBED}/{log}/edit_factors.pt", map_location="cpu")
    W, Ks = fac["W"].double().numpy(), fac["K1_solver"].double().numpy()
    s = (W * Ks).sum(1)
    T = W.shape[0]
    Kn = Ks / np.linalg.norm(Ks, axis=1, keepdims=True)
    for m in METHODS:
        mi = G["mine"][m]
        b, eta, cap, capped = (np.asarray(mi[k]) for k in ("b", "eta", "cap", "capped"))
        detw = {"rescope": W, "rescope_null": G["Wnull_mine"].numpy(), "cosine_shared": Kn}[m]
        c = json.load(open(f"{E}/gate/{log}/{m}__reg.json"))["cells"]["0"]
        cov = {}
        for k in ("tgen", "igen", "rel"):
            rows = [sidx[f"{fam}:edit:{k}:{i}"]["row_tf"] for i in range(T)]
            own = np.zeros(T, dtype=bool)
            for a in range(0, T, 250):
                K = Ktf[torch.tensor(rows[a:a + 250])].double().numpy()
                R = K @ detw.T
                R = R / s[None, :] if m != "cosine_shared" else R / np.linalg.norm(K, axis=1, keepdims=True)
                mar = R - b[None, :]
                cand = np.where(capped[None, :], R >= cap[None, :], mar > eta)
                top = np.where(cand, mar, -np.inf).argmax(1)
                own[a:a + len(K)] = cand.any(1) & (top == np.arange(a, a + len(K)))
            ex = c["per_edit"][k] if "per_edit" in c else c["edit"][k]["own_top"]
            exf = np.array([x == "1" for x in ex])
            cov[k] = {"coverage_mine": 100.0 * float(own.mean()), "coverage_experiment": 100.0 * float(exf.mean()), "n_flags_differ": int((own != exf).sum()), "n": T}
        out["methods"][m]["coverage_all_edits_gate_level"] = cov
    return out


def table(res):
    L = ["E3 VERIFICATION, CHECK 2: my plain-Evaluator rows and my own thresholds (calibration draw 0) against the experiment's files",
         "mine = rescope.Evaluator, no driver / memo / key cache; thresholds from keys captured here by my own numpy rule.  exp = experiment (same rows).", ""]
    for r in res:
        if r.get("missing"):
            L.append(f"{r['log']}: NOT FINISHED")
            continue
        L.append(f"== {r['log']}  (T = {r['T']}; cache check, bitwise equal keys: " + ", ".join(f"{k} {v['n_bitwise_equal']}/{v['n']}" for k, v in r["cache_check"].items()) + ")")
        L.append(f"{'method':14s} {'eta mine':>12s} {'eta exp':>12s} {'cap m/e':>8s} {'thr diff':>9s} | {'hd n':>5s} {'fired m/e':>11s} {'D1 m/e':>11s} {'flags diff f/D1':>15s} | "
                 f"{'edits':>5s} {'Rel m/e':>13s} {'T-Gen m/e':>11s} {'I-Gen m/e':>11s} {'diff ex/own':>11s} | {'cov TG m/e':>11s} {'cov IG m/e':>11s} {'diff':>7s}")
        for m in METHODS:
            o = r["methods"][m]
            t, p, e, c = o["thresholds"], o["probes"], o["edits"], o["coverage_all_edits_gate_level"]
            L.append(f"{m:14s} {t['eta_mine']:12.8f} {t['eta_experiment']:12.8f} {t['n_capped_mine']:>3d}/{t['n_capped_experiment']:<4d} {t['thr_summary_max_abs_diff']:9.1e} | "
                     f"{p['n']:5d} {p['fired_mine']:5.2f}/{p['fired_experiment_same_rows']:<5.2f} {p['D1_mine']:5.2f}/{p['D1_experiment_same_rows']:<5.2f} "
                     f"{p['n_differ_fired']:>7d}/{p['n_differ_D1']:<7d} | {e['n_edits']:5d} {e['rel']['exact_mine']:6.2f}/{e['rel']['exact_experiment_same_edits']:<6.2f} "
                     f"{e['tgen']['exact_mine']:5.2f}/{e['tgen']['exact_experiment_same_edits']:<5.2f} {e['igen']['exact_mine']:5.2f}/{e['igen']['exact_experiment_same_edits']:<5.2f} "
                     f"{e['n_differ_exact']:>5d}/{e['n_differ_own_released_last']:<5d} | {c['tgen']['coverage_mine']:5.2f}/{c['tgen']['coverage_experiment']:<5.2f} "
                     f"{c['igen']['coverage_mine']:5.2f}/{c['igen']['coverage_experiment']:<5.2f} {c['tgen']['n_flags_differ']:>3d}/{c['igen']['n_flags_differ']:<3d}")
            if "probes_prompt_position" in o:
                q = o["probes_prompt_position"]
                L.append(f"{'':14s} decode's position: {q['n_shifted']} shifted rows, fired differs from tf on {q['n_fired_differs_from_tf_mine']}, D1 on {q['n_D1_differs_from_tf_mine']}; "
                         f"hd fired m/e {q['fired_mine']:.2f}/{q['fired_experiment']:.2f}, D1 m/e {q['D1_mine']:.2f}/{q['D1_experiment']:.2f}; flags differing from the experiment f/D1 {q['n_differ_fired']}/{q['n_differ_D1']}")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="gpu")
    ap.add_argument("--logs", default="")
    a = ap.parse_args()
    res = [one_log(g, a.sub) for g in (a.logs.split(",") if a.logs else LOGS)]
    sfx = "" if a.sub == "gpu" else "_" + a.sub
    with open(f"{ROOT}/out/compare{sfx}.json", "w") as f:
        json.dump(res, f, indent=1)
    t = table(res)
    with open(f"{ROOT}/out/compare_table{sfx}.txt", "w") as f:
        f.write(t + "\n")
    print(t)
