"""E10 verification (CPU): discipline checks on the experiment's files, code and logs.

    python v7_discipline.py  ->  out/discipline.json
 1. audited logs      every log named in the experiment's outputs is declared safe by the independent audit (verdicts.json)
 2. pools             calibration draws of every behavioural record: REPAIR pool, n = 200, alpha = 0.05, draws 0-2; the R3 statistics
                      prompts recomputed from the stated rule (REPAIR prompts in none of the 20 draws); equal-evidence complaints: REPAIR
                      pool; locality rows: HELD-OUT pool and neighbourhood prompts; near-scope rows reported separately
 3. nothing tuned     the R3 gate: selection recomputed from the development log's gate-level file, written before the first outcome
                      of a reported log; the split; one queue entry per log file, exit codes; every gate / weight in the experiment's
                      outcome store is a method instance recorded under out/ (no served method without a record); code files
                      modified after the first job started
 4. public scores     code path (the adapter's e10t.public -> the public functions); case files; summariser
 5. R21 expectations  gate-level false firing recounted from the per-row flags; eta >= 1 counted from the per-draw records and, for T500,
                      recomputed by rebuilding the 20 calibrations from the log with the adapter
"""
import hashlib
import os
import re
import time

import numpy as np

import vcommon as V

import e10t  # noqa: E402
import torch  # noqa: E402
from e10t import data as D, methods as MT  # noqa: E402

OUT = {}
X = V.XOUT
LOGS4 = ("TX-G2XL-rome-T100", "TX-G2XL-rome-T500", "TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000")
DEV = "TX-G2XL-rome-dev100"

# ----------------------------------------------------------------------------- 1. audited logs
aud = V.jload(V.AUDIT + "/verdicts.json")
per = {k: v.get("verdict") for k, v in aud["per_log"].items()}
used = set()
for sub in ("behave", "equal", "stream", "rows", "public", "gen"):
    d = os.path.join(X, sub)
    if os.path.isdir(d):
        used |= {x.split("@")[0].replace("_summary.json", "") for x in os.listdir(d) if x.startswith("TX-")}
for sub in ("gate", "cert", "r22", "summary"):
    for f in os.listdir(os.path.join(X, sub)):
        m = re.search(r"TX-G2XL-rome-[A-Za-z0-9]+", f)
        if m:
            used.add(m.group(0))
used = sorted(u for u in used if u.startswith("TX-"))
OUT["audited_logs"] = {"audit_file": V.AUDIT + "/verdicts.json", "audit_verdicts": per, "audit_checks": {k: (v.get("verdict") if isinstance(v, dict) else v) for k, v in aud.get("checks", {}).items()},
                       "logs_named_in_the_experiment_outputs": used, "all_safe": all(per.get(u) == "safe" for u in used),
                       "prefix_logs_of_the_stream": sorted(x for x in os.listdir(os.path.join(X, "stream")) if "@" in x)}

# ----------------------------------------------------------------------------- 2. pools
pool = {"behaviour_records": {}, "equal_complaints": {}, "rows": {}}
L100 = e10t.open_log(V.T100)
rep_ids = {r["prompt_id"] for r in L100.benign("repair")}
ho_ids = {r["prompt_id"] for r in L100.benign("heldout")}
n_rec = n_cal = bad_pool = bad_an = 0
draws_seen, alphas, ns = set(), set(), set()
mism = {"hook_mismatch": 0, "hook_mismatch_dec": 0}
for sub in ("behave", "equal", "stream"):
    for log in sorted(os.listdir(os.path.join(X, sub))):
        dd = os.path.join(X, sub, log)
        if not os.path.isdir(dd):
            continue
        for f in sorted(os.listdir(dd)):
            if not f.endswith(".json") or f.startswith("_") or f.endswith(".public.json") or f.endswith(".infeasible.json"):
                continue
            m = V.jload(os.path.join(dd, f))
            n_rec += 1
            for k in mism:
                mism[k] += int(m.get("cost", {}).get(k, 0))
            dr = (m.get("record") or {}).get("draw")
            if isinstance(dr, dict):
                n_cal += 1
                ids = dr["prompt_ids"]["first"] if isinstance(dr["prompt_ids"], dict) else dr["prompt_ids"]
                bad_pool += int(dr.get("pool") != "repair" or any(p not in rep_ids for p in ids))
                bad_an += int(dr.get("n") != 200 or m.get("alpha") != 0.05)
                draws_seen.add((sub, dr.get("draw")))
                alphas.add(m.get("alpha"))
                ns.add(dr.get("n"))
pool["behaviour_records"] = {"n_method_records": n_rec, "n_calibrated": n_cal, "n_calibrated_not_from_REPAIR_pool": bad_pool, "n_calibrated_with_other_alpha_or_n": bad_an,
                             "alphas": sorted(alphas), "n_values": sorted(ns), "draws": sorted({d for _, d in draws_seen}), "draws_by_block": {s: sorted(d for ss, d in draws_seen if ss == s) for s in ("behave", "equal", "stream")},
                             "hook_mismatch_total": mism}
# R3 statistics prompts, from the stated rule
rows = L100.benign("repair")
usedt = set()
for d in range(20):
    usedt |= set(D.calibration(L100, 200, d).text_ids)
rest = [r for r in rows if r["text_id"] not in usedt]
idx = np.sort(np.random.default_rng(20261111).permutation(len(rest))[:500])
sha = hashlib.sha1(",".join(rest[int(i)]["prompt_id"] for i in idx).encode()).hexdigest()[:12]
gsha = {lg: V.jload(X + f"/gate/{lg}.json")["meta"]["r3_stats"]["prompt_ids_sha1"] for lg in LOGS4 + (DEV,)}
pool["r3_statistics"] = {"rule": "REPAIR-pool prompts in none of the 20 calibration draws; seeded sample of 500 (experiment's seed 20261111)", "n_repair_prompts_in_no_draw": len(rest),
                         "sha1_recomputed": sha, "sha1_in_gate_files": gsha, "same": all(v == sha for v in gsha.values())}
for log in (V.T100, V.T500):
    c = V.jload(X + f"/equal/{log}/_complaints.json")
    ids = [p for s in c["sets"].values() for p in s["prompt_ids"]]
    pool["equal_complaints"][log] = {"n_sets": len(c["sets"]), "sets": sorted(c["sets"]), "n_complaints": len(ids), "n_not_in_REPAIR_pool": sum(p not in rep_ids for p in ids),
                                     "n_D1_complaints_available": c["n_D1_complaints_under_deployed"]}
    r = V.load_rows(X + f"/rows/{log}/rescope__d0.jsonl.gz")
    ho = [x for x in r if x["set"] == "heldout"]
    nb = [x for x in r if x["kind"] == "neighborhood"]
    L = e10t.open_log(log)
    pool["rows"][log] = {"n_heldout_rows": len(ho), "heldout_rows_not_in_HELDOUT_pool": sum(x["prompt_id"] not in ho_ids for x in ho), "heldout_rows_in_REPAIR_pool": sum(x["prompt_id"] in rep_ids for x in ho),
                         "n_neighborhood_rows": len(nb), "neighborhood_rows_not_of_logged_requests": sum(not (x["set"] == "stream" and x["pos"] < L.T) for x in nb),
                         "n_heldout_near_scope": sum(not x["clean"] for x in ho), "n_neighborhood_near_scope": sum(not x["clean"] for x in nb)}
    s = V.jload(X + f"/summary/behave_{log}.json")["rescope"]["metrics"]
    pool["rows"][log]["near_scope_reported_separately_in_summary"] = all(k in s for k in ("heldout.clean.D1_tf", "heldout.near_scope.D1_tf", "neighborhood.clean.D1_tf", "neighborhood.near_scope.D1_tf"))
table = open(X + "/table.txt").read()
pool["table_has_near_scope_columns"] = bool(re.search(r"near.scope|FF near|near-scope", table))
OUT["pools"] = pool

# ----------------------------------------------------------------------------- 3. nothing tuned
tun = {}
bg = V.jload(X + "/best_gate_text.json")
g = V.jload(X + f"/gate/{DEV}.json")
R3 = ("cos_centred", "cos_null", "white_0.01", "white_0.1", "white_1", "whitecos_0.01", "whitecos_0.1", "whitecos_1")
cov = {lab: float(np.mean([v["paraphrase"]["own_released"] for v in g["rows"][lab]["gon"]["per_draw"].values()])) for lab in R3}
best = max(R3, key=lambda x: cov[x])
shards = []
for root, _, files in os.walk(V.RUN + "/store/gpt2-xl"):
    shards += [os.path.getmtime(os.path.join(root, f)) for f in files if f.startswith("shard_")]
first_rows = min(os.path.getmtime(os.path.join(root, f)) for sub in ("rows", "public", "behave") for root, _, files in os.walk(os.path.join(X, sub)) for f in files)
first_gate_rep = min(os.path.getmtime(X + f"/gate/{lg}.json") for lg in LOGS4)
utc = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))      # noqa: E731
tun["best_gate"] = {"selected": bg["selected"], "recomputed_argmax_paraphrase_coverage_on_dev": best, "coverage_dev": cov, "same": best == bg["selected"], "selection_log": bg["log"],
                    "file_written": utc(os.path.getmtime(X + "/best_gate_text.json")), "dev_gate_file_written": utc(os.path.getmtime(X + f"/gate/{DEV}.json")),
                    "first_outcome_shard_of_the_store": utc(min(shards)), "first_behavioural_output_file": utc(first_rows), "first_gate_level_file_of_a_reported_log": utc(first_gate_rep),
                    "written_before_any_outcome_on_a_reported_log": bool(os.path.getmtime(X + "/best_gate_text.json") < min(min(shards), first_rows, first_gate_rep))}
sp = V.jload(X + "/split.json")
tun["split"] = {"split_position": sp["split_position"], "is_173": sp["split_position"] == 173, "written": sp.get("written_utc")}
# queue entries and exit codes of the experiment's logs
ql = {}
for f in sorted(os.listdir(V.RUN + "/logs")):
    txt = open(os.path.join(V.RUN, "logs", f)).read()
    ql[f] = {"queued": len(re.findall(r"\[gq [0-9:]+\] queued", txt)), "exit_codes": re.findall(r"exit code (\d+)", txt), "tracebacks": txt.count("Traceback")}
tun["logs"] = {"files": ql, "n_files": len(ql), "n_with_more_than_one_queue_entry": sum(v["queued"] > 1 for v in ql.values()),
               "n_nonzero_exit": sum(any(c != "0" for c in v["exit_codes"]) for v in ql.values()), "n_tracebacks": sum(v["tracebacks"] for v in ql.values())}
# every served gate / weight of the outcome store has a record under out/
sigs = set()
for sub in ("behave", "equal", "stream"):
    for log in os.listdir(os.path.join(X, sub)):
        if not os.path.isdir(os.path.join(X, sub, log)):
            continue
        for f in os.listdir(os.path.join(X, sub, log)):
            if f.endswith(".json") and not f.startswith("_") and not f.endswith(".public.json") and not f.endswith(".infeasible.json"):
                rec = V.jload(os.path.join(X, sub, log, f)).get("record") or {}
                if rec.get("gate_sig"):
                    sigs.add("G" + rec["gate_sig"])
                if rec.get("weight_sha1"):
                    sigs.add("W" + rec["weight_sha1"])
                if rec.get("sig"):
                    sigs.add(rec["sig"])
# deployed weights of the registered logs and of the stream prefixes, and the pre-edit weight
import tx_public as TX  # noqa: E402
f2 = TX.factors(V.T2000)
W = torch.load(TX.registry()[V.T2000]["pre_edit_weight"], map_location="cpu")["weight"].clone()
wsha = lambda w: "W" + hashlib.sha1(w.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]      # noqa: E731
wnames = {wsha(W): "pre-edit"}
want = {100, 150, 170, 172, 173, 174, 175, 180, 200, 250, 300, 400, 500}
for i in range(500):
    W += f2["left"][i].unsqueeze(1) @ f2["right"][i].unsqueeze(0)
    if i + 1 in want:
        wnames[wsha(W)] = f"deployed prefix {i + 1}"
for lg in ("TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000"):
    wnames[wsha(TX.weights(lg)[1])] = "deployed " + lg
ns_ = sorted({d.split("__", 1)[1] for d in os.listdir(V.RUN + "/store/gpt2-xl")})
unmatched = [s for s in ns_ if s not in sigs and s not in wnames]
tun["store"] = {"n_signatures_in_store": len(ns_), "n_gates": sum(s.startswith("G") for s in ns_), "n_weights": sum(s.startswith("W") for s in ns_), "n_lookup": sum(s.startswith("L") for s in ns_),
                "n_recorded_method_signatures": len(sigs), "n_matched_to_a_method_record": sum(s in sigs for s in ns_), "n_matched_to_a_deployed_or_pre_edit_weight": sum(s in wnames for s in ns_),
                "unmatched": unmatched, "n_unmatched": len(unmatched)}
# code files modified after the first job of the experiment started (13:21:44 UTC queue time of the behavioural jobs)
t_first = min(shards)
code = {f: utc(os.path.getmtime(os.path.join(V.RUN, "code", f))) for f in sorted(os.listdir(V.RUN + "/code")) if f.endswith((".py", ".sql"))}
tun["code_files_modified_after_the_first_outcome"] = {f: t for f, t in code.items() if os.path.getmtime(os.path.join(V.RUN, "code", f)) > t_first}
tun["tests_boot_written"] = utc(os.path.getmtime(X + "/tests_boot.json"))
# per job: the experiment's code files it imports (transitively) that were modified after the job had started
srcs = {f: open(os.path.join(V.RUN, "code", f)).read() for f in os.listdir(V.RUN + "/code") if f.endswith(".py")}


def imports(f, seen=None):
    seen = set() if seen is None else seen
    for m_ in re.findall(r"^\s*(?:import|from)\s+(x\w+|summarise|render\w*)\b", srcs[f], flags=re.M):
        g_ = m_ + ".py"
        if g_ in srcs and g_ not in seen:
            seen.add(g_)
            imports(g_, seen)
    return seen


day = time.strftime("%Y-%m-%d", time.gmtime(t_first))
jobs = {}
for f in sorted(os.listdir(V.RUN + "/logs")):
    txt = open(os.path.join(V.RUN, "logs", f)).read()
    m_ = re.search(r"\[gq (\d\d:\d\d:\d\d)\] started .*?(\w+\.py)", txt)
    if not m_ or m_.group(2) not in srcs:
        continue
    start = day + " " + m_.group(1)
    used_ = sorted({m_.group(2)} | imports(m_.group(2)))
    jobs[f] = {"script": m_.group(2), "started": start, "imports": used_, "modified_after_the_start": {g_: code[g_] for g_ in used_ if code[g_] > start}}
tun["jobs"] = jobs
tun["files_modified_after_a_job_that_imports_them_started"] = sorted({g_ for j in jobs.values() for g_ in j["modified_after_the_start"]})
tun["jobs_started_before_such_a_modification"] = sorted(f for f, j in jobs.items() if j["modified_after_the_start"])
tun["method_and_behaviour_code_last_modified"] = {f: code[f] for f in ("xmeth.py", "run_behave.py", "xgate.py", "select_best.py")}
tun["first_outcome"] = utc(t_first)
# grep: settings in the experiment's code
src = {f: open(os.path.join(V.RUN, "code", f)).read() for f in os.listdir(V.RUN + "/code") if f.endswith(".py")}
tun["code_constants"] = {"ALPHA, N_CAL, N_DRAWS, BEHAVE_DRAWS line": re.search(r"ALPHA, N_CAL, N_DRAWS, BEHAVE_DRAWS = .*", src["xcommon.py"]).group(0),
                         "files_that_mention_another_alpha": sorted(f for f, s in src.items() if re.search(r"alpha\s*=\s*0\.(?!05\b)\d+", s))}
# tags present against the registered plan
tags = {log: sorted(x[:-5] for x in os.listdir(os.path.join(X, "behave", log)) if x.endswith(".json") and not x.startswith("_") and not x.endswith(".public.json")) for log in (V.T100, V.T500)}
PLAIN = ["deployed", "preedit", "where_logonly", "cosine_published", "exact", "lookup_subject"]
CAL = ["where_twoblock", "rescope", "cosine", bg["selected"]]
exp100 = sorted([f"{x}__dx" for x in PLAIN] + [f"{x}__d{d}" for x in CAL for d in (0, 1, 2)])
exp500 = sorted(exp100 + ["where_logonly@goff__dx"] + [f"{x}@goff__d{d}" for x in CAL for d in (0, 1, 2)])
tun["behaviour_instances"] = {"T100_as_planned": tags[V.T100] == exp100, "T500_as_planned": tags[V.T500] == exp500, "n_T100": len(tags[V.T100]), "n_T500": len(tags[V.T500]),
                              "extra_T100": sorted(set(tags[V.T100]) - set(exp100)), "extra_T500": sorted(set(tags[V.T500]) - set(exp500))}
OUT["nothing_tuned"] = tun

# ----------------------------------------------------------------------------- 4. public scores: code path
ps = {"experiment_calls": {f: sorted(set(re.findall(r"e10t\.public\.\w+", s))) for f, s in src.items() if "e10t.public" in s},
      "experiment_defines_its_own_probability_test": any(re.search(r"def .*(prob_test|rewrite_quality)", s) for s in src.values()),
      "adapter_calls_public_function": "compute_rewrite_quality_counterfact" in open(V.E10T + "/code/e10t/public.py").read() or "public_prob_test" in open(V.E10T + "/code/e10t/public.py").read(),
      "adapter_uses_public_summariser": "experiments.summarize" in open(V.E10T + "/code/e10t/public.py").read(),
      "builder_wrapper_calls_public_function": "from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact" in open(V.TXP + "/code/tp_common.py").read()}
OUT["public_code_path"] = ps

# ----------------------------------------------------------------------------- 5. R21 expectations, recounted
exp = {"false_firing_heldout_mean_over_20_draws": {}, "eta": {}}
for lg in ("TX-G2XL-rome-T500", "TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000"):
    fl = np.load(X + f"/gate/{lg}_flags.npz")
    gj = V.jload(X + f"/gate/{lg}.json")
    n = gj["meta"]["n"]["heldout"]
    o = {}
    for lab in ("rescope", "where_twoblock", "cosine", "whitecos_0.1"):
        for mode in ("gon", "goff"):
            v = [100.0 * np.unpackbits(fl[f"{lab}|{mode}|{d}|heldout"])[:n].mean() for d in range(20)]
            o[f"{lab}|{mode}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
    exp["false_firing_heldout_mean_over_20_draws"][lg] = o
    et = [gj["rows"]["rescope"]["gon"]["cal"][str(d)]["eta"] for d in range(20)]
    exp["eta"][lg] = {"n_draws_eta_ge_1_from_gate_file": int(sum(e >= 1.0 for e in et)), "eta_draw0": et[0]}
# T500: the 20 calibrations rebuilt from the log with the adapter (guards off: thresholds as calibrated)
L = e10t.open_log(V.T500)
etas, lost = [], []
for d in range(20):
    M = MT.rescope(L, alpha=V.ALPHA, n=V.N_CAL, draw=d, allow_silence=True)
    etas.append(float(M.rec["cal"]["eta"]))
    lost.append(int(M.rec["own_requests"]["n_not_candidate"]))
exp["eta"]["TX-G2XL-rome-T500"].update(rebuilt_etas=etas, n_draws_eta_ge_1_rebuilt=int(sum(e >= 1.0 for e in etas)), own_requests_silenced_guards_off_per_draw=lost,
                                       own_requests_silenced_draws_0_1_2=lost[:3])
OUT["r21_expectations"] = exp
V.jdump(OUT, V.OUT + "/discipline.json")
print({k: v for k, v in OUT["audited_logs"].items() if k != "audit_checks"})
print(pool["behaviour_records"], pool["r3_statistics"], pool["equal_complaints"], pool["rows"], pool["table_has_near_scope_columns"])
print(tun["best_gate"], tun["split"], {k: v for k, v in tun["logs"].items() if k != "files"}, tun["store"], tun["tests_boot_written"], tun["code_constants"], tun["behaviour_instances"])
print(tun["files_modified_after_a_job_that_imports_them_started"], tun["jobs_started_before_such_a_modification"], tun["method_and_behaviour_code_last_modified"], tun["first_outcome"])
print(ps)
print(exp)
