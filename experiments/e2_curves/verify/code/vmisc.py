"""Verification of E2 a-c, CPU: claims of the summary recomputed from the experiment's per-row files and the P12 records
(round-off floor, own prompts out of scope, part-d flags against the records, single-seed values, near-scope and far read-outs
that the files hold).  -> out/misc_check.json"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import glob, gzip, json, os
import numpy as np
FIN = f"{_RP.ROOT}/02_research/experiments/10_final"
EXP, VER = FIN + "/e2_curves", FIN + "/e2_curves/verify"
P12 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p12_linear_streams"
jl = lambda p: [json.loads(x) for x in gzip.open(p, "rt")]                    # noqa: E731
tf = lambda rows: [r for r in rows if "line" not in r and not r.get("d2_row")]   # noqa: E731
res = json.load(open(EXP + "/out/results.json"))
out = {}
for name, sh in (("TB-Q100-ae-l2_3", "Q"), ("TB-L100-ae-l2_3", "L")):
    o = out.setdefault(name, {})
    dep = {r["sid"]: r for r in tf(jl(f"{EXP}/out/rows/base__{sh}__deployed.jsonl.gz"))}
    flo = tf(jl(f"{EXP}/out/rows/base__{sh}__where_floor.jsonl.gz"))
    o["floor"] = {"edits": sum(bool(r["exact"]) != bool(dep[r["sid"]]["exact"]) for r in flo if r["kind"] == "edit"), "n_edits": sum(r["kind"] == "edit" for r in flo),
                  "heldout_disjoint": sum(bool(r["exact"]) != bool(dep[r["sid"]]["exact"]) for r in flo if r["kind"] == "probe" and r.get("disjoint")),
                  "n_heldout_disjoint": sum(r["kind"] == "probe" and bool(r.get("disjoint")) for r in flo),
                  "hard_negatives": sum(bool(r["exact"]) != bool(dep[r["sid"]]["exact"]) for r in flo if r["kind"] == "hn"), "n_hard_negatives": sum(r["kind"] == "hn" for r in flo)}
    o["floor"]["experiment_states"] = {pool: next(r["value"] for r in res if r["log"] == name and r["metric"] == "floor.n_flags_differ_from_deployed" and r["pool"] == pool)
                                       for pool in ("edits", "heldout_disjoint", "hard_negatives")}
    # part b: own Rel prompts out of scope in every cell; D1 / Rel per seed at n = 1 and n = 10
    own_out, per = 0, {}
    for p in sorted(glob.glob(f"{EXP}/out/rows/b__{sh}__where_least__*.jsonl.gz")):
        rows = tf(jl(p))
        tag = os.path.basename(p)[len(f"b__{sh}__"):-len(".jsonl.gz")]
        own_out += sum(not r["own_fired_last"] for r in rows if r["kind"] == "edit" and r["metric"] == "rel")
        hd = [r for r in rows if r["kind"] == "probe" and r.get("disjoint")]
        ed = lambda m: [r for r in rows if r["kind"] == "edit" and r["metric"] == m]    # noqa: E731
        per[tag] = {"D1": 100.0 * float(np.mean([bool(r["D1"]) for r in hd])), "fired": 100.0 * float(np.mean([bool(r["fired"]) for r in hd])),
                    "rel": 100.0 * float(np.mean([bool(r["exact"]) for r in ed("rel")])), "tgen": 100.0 * float(np.mean([bool(r["exact"]) for r in ed("tgen")])),
                    "igen": 100.0 * float(np.mean([bool(r["exact"]) for r in ed("igen")])),
                    "rel_kept": sum(bool(r["exact"]) and bool(dep[r["sid"]]["exact"]) for r in ed("rel")), "rel_gained": sum(bool(r["exact"]) and not dep[r["sid"]]["exact"] for r in ed("rel")),
                    "rel_lost": sum((not r["exact"]) and bool(dep[r["sid"]]["exact"]) for r in ed("rel")),
                    "checks_lost_rel_tgen_igen": sum((not r["exact"]) and bool(dep[r["sid"]]["exact"]) for r in rows if r["kind"] == "edit"),
                    "checks_gained_rel_tgen_igen": sum(bool(r["exact"]) and not dep[r["sid"]]["exact"] for r in rows if r["kind"] == "edit"),
                    "hn_D1": {t: 100.0 * float(np.mean([bool(r["D1"]) for r in rows if r["kind"] == "hn" and r["hn_type"] == t])) for t in ("HN-I", "HN-Q", "HN-X")
                              if any(r["kind"] == "hn" for r in rows)},
                    "hn_fired_hook": {t: 100.0 * float(np.mean([bool(r["fired"]) for r in rows if r["kind"] == "hn" and r["hn_type"] == t])) for t in ("HN-I", "HN-Q", "HN-X")
                                      if any(r["kind"] == "hn" for r in rows)}}
    o["part_b"] = {"n_own_rel_prompts_out_of_scope_over_all_cells": own_out, "n_cells": len(per),
                   "deployed_start_n1": {s: per[f"where_least__deployed__n1__s{s}"]["D1"] for s in (0, 1, 2)},
                   "h1_cells_logonly_n10": {s: per[f"where_least__logonly__n10__s{s}"] for s in (0, 1, 2)},
                   "deployed_start_n10": {s: per[f"where_least__deployed__n10__s{s}"] for s in (0, 1, 2)}}
    # part d: flags of the rebuilt rows against the ORIGINAL records (recomputed here from the rows)
    d = {}
    for tag in ("COMP_B__nALL__s0", "RB_UNION__n10__s0", "RB_UNION__nALL__s0", "RB_TOT__n10__s0", "REEDIT_B__n10__s0", "REEDIT_B__nALL__s0"):
        rows = tf(jl(f"{EXP}/out/rows/d__{sh}__{tag}.jsonl.gz"))
        rec = json.load(open(f"{P12}/out/{name}/{tag}.json"))
        ho = {h["group"]: bool(h["reproduced_after"]) for h in rec["eval"]["heldout_rows"]}
        pr = [r for r in rows if r["kind"] == "probe"]
        d[tag] = {"n_heldout": len(pr), "heldout_differ": sum(bool(r["exact"]) != ho[r["group"]] for r in pr),
                  "edit_differ": sum(bool(r["exact"]) != bool(rec["eval"]["edits"][r["metric"]]["em_list"][r["edit"]]) for r in rows if r["kind"] == "edit"),
                  "D1_disjoint": 100.0 * float(np.mean([bool(r["D1"]) for r in pr if r.get("disjoint")])),
                  "experiment_states": [json.load(open(f"{EXP}/out/cells/d/{name}/{tag}.json"))["flags_vs_record"][k] for k in ("n_heldout_differ", "n_edit_differ")]}
    o["part_d_flags_vs_record"] = d
    # read-outs in the files next to the far rates (MME-T / BEN-T firing of the label-free curve, near-scope of the H1 row)
    g = lambda part, m, v, nl, metric, pool: next((r["value"] for r in res if r["log"] == name and r.get("part") == part and r["method"] == m and r["variant"] == v   # noqa: E731
                                                  and str(r.get("n_label")) == nl and r["draw"] is None and r["metric"] == metric and r["pool"] == pool), None)
    o["files_hold"] = {"rescope_MME-T_fired_mean": {nl: g("c", "rescope", "", nl, "gate.MME-T.fired@mean", "MME-T") for nl in ("40", "50", "100", "200", "ALL")},
                       "rescope_BEN-T_fired_mean": {nl: g("c", "rescope", "", nl, "gate.BEN-T.fired@mean", "BEN-T") for nl in ("40", "50", "100", "200", "ALL")},
                       "rescope_n200_hn_fired_mean": {t: g("c", "rescope", "", "200", f"gate.hn.{t}.fired@mean", "hard_negatives") for t in ("HN-I", "HN-Q", "HN-X")},
                       "h1_row_hn_fired_gate_mean": {t: g("b", "where_least", "logonly", "10", f"gate.hn.{t}.fired@mean", "hard_negatives") for t in ("HN-I", "HN-Q", "HN-X")},
                       "h1_row_fired_heldout_mean": g("b", "where_least", "logonly", "10", "probe.fired@mean", "heldout_disjoint"),
                       "deployed_start_n10_fired_heldout_mean": g("b", "where_least", "deployed", "10", "probe.fired@mean", "heldout_disjoint"),
                       "deployed_start_n10_hn_fired_gate_mean": {t: g("b", "where_least", "deployed", "10", f"gate.hn.{t}.fired@mean", "hard_negatives") for t in ("HN-I", "HN-Q", "HN-X")}}
    # the experiment's sanity claim: its cells shared with e0 (deployed; rescope / rescope_shared at n 200, draws 0 - 2), recomputed here from ITS rows
    e0 = json.load(open(FIN + "/e0_sanity/out/results.json"))["logs"][name]["behaviour"]["methods"]
    v = lambda x: x["value"] if isinstance(x, dict) else x                # noqa: E731
    pairs = [("deployed", e0["deployed"]["single"], tf(jl(f"{EXP}/out/rows/base__{sh}__deployed.jsonl.gz")))]
    for dd in (0, 1, 2):
        for mine_, theirs_ in (("rescope", "M1"), ("rescope_shared", "M2")):
            pairs.append((f"{mine_} draw {dd}", e0[theirs_]["draws"][str(dd)], tf(jl(f"{EXP}/out/rows/c__{sh}__{mine_}_n200__d{dd}.jsonl.gz"))))
    nnum, worst = 0, 0.0
    for lab, a_, rows in pairs:
        hd = [r for r in rows if r["kind"] == "probe" and r.get("disjoint")]
        st = [r for r in hd if r.get("strict")]
        cmp_ = [(a_["probe"]["heldout_disjoint"]["D1"], 100.0 * np.mean([bool(r["D1"]) for r in hd])), (a_["probe"]["heldout_strict"]["D1"], 100.0 * np.mean([bool(r["D1"]) for r in st]))]
        if a_["probe"]["heldout_disjoint"].get("fired") is not None:
            cmp_.append((a_["probe"]["heldout_disjoint"]["fired"], 100.0 * np.mean([bool(r["fired"]) for r in hd])))
        cmp_ += [(a_["edit"][m]["exact"], 100.0 * np.mean([bool(r["exact"]) for r in rows if r["kind"] == "edit" and r["metric"] == m])) for m in ("rel", "tgen", "igen")]
        hn = [r for r in rows if r["kind"] == "hn"]
        if hn and "hn" in a_:
            cmp_ += [(a_["hn"][t]["D1"], 100.0 * np.mean([bool(r["D1"]) for r in hn if r["hn_type"] == t])) for t in ("HN-I", "HN-Q", "HN-X") if t in a_["hn"]]
        for x, y in cmp_:
            if x is not None:
                nnum += 1
                worst = max(worst, abs(float(v(x)) - float(y)))
    o["sanity_claim_from_the_experiments_rows"] = {"n_cells": len(pairs), "n_numbers": nnum, "max_abs_diff_points": worst}
    print(name, json.dumps(o)[:300], o["sanity_claim_from_the_experiments_rows"])
import time
mt = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(os.path.getmtime(p)))      # noqa: E731  (the machine's clock is UTC)
out["_timeline"] = {"code_last_modified": {f: mt(f"{EXP}/code/{f}") for f in ("tables.py", "example.py", "render.py", "job.py", "cw_check.py", "e2c.py", "a_import.py", "c_gate.py")},
                    "h1_cells_written": {sh: [json.load(open(f"{EXP}/out/cells/b/{name}/where_least__logonly__n10__s{s}.json"))["written"] for s in (0, 1, 2)]
                                         for name, sh in (("TB-Q100-ae-l2_3", "Q"), ("TB-L100-ae-l2_3", "L"))},
                    "running_example_written": json.load(open(FIN + "/out/running_example.json"))["written"]}
tl = out["_timeline"]
import calendar
tl["job_process_started"] = {}
for name, sh in (("TB-Q100-ae-l2_3", "Q"), ("TB-L100-ae-l2_3", "L")):
    j = json.load(open(sorted(glob.glob(f"{EXP}/out/parts/job_{name}_*.json"))[-1]))
    t_end = calendar.timegm(time.strptime(j["time"], "%Y-%m-%d %H:%M:%S"))
    tl["job_process_started"][sh] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t_end - j["seconds"] - 15))      # 15 s: model load before the job's own clock starts (lower bound)
tl["job_py_modified_after_the_job_started"] = {sh: tl["code_last_modified"]["job.py"] > t for sh, t in tl["job_process_started"].items()}
qk = {os.path.basename(p_).replace("489", "N"): set(json.load(open(p_))) for p_ in glob.glob(f"{EXP}/out/cells/*/TB-Q100-ae-l2_3/*.json")}
lk = {os.path.basename(p_).replace("494", "N"): set(json.load(open(p_))) for p_ in glob.glob(f"{EXP}/out/cells/*/TB-L100-ae-l2_3/*.json")}
tl["cells_of_the_two_logs_have_the_same_tags_and_fields"] = set(qk) - {"running_example.json"} == set(lk) and all(qk[k] == lk[k] for k in lk)
tl["n_cells"] = {"Q": len(qk), "L": len(lk)}
tl["tables_py_modified_after_the_last_h1_cell"] = tl["code_last_modified"]["tables.py"] > max(max(v) for v in tl["h1_cells_written"].values())
tl["example_py_modified_before_the_example_ran"] = tl["code_last_modified"]["example.py"] < tl["running_example_written"]
print(json.dumps(tl))
json.dump(out, open(f"{VER}/out/misc_check.json", "w"), indent=1)
