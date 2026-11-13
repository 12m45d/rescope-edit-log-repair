"""Audit: remaining checks on stored files (CPU).  Output: out/extra.json

 1. Statistics file: sha256 of the copy the audit downloaded itself from rome.baulab.info (tmp/dl/stats_fresh.npz).
 2. Hash chain: every weight hash the builder recorded while the runs were writing (progress.jsonl, every 25 updates and
    at the snapshots) against the audit's replay of the stored factors.
 3. The public code's own printouts: statistics loaded from the cached file (never recomputed / downloaded by the public
    loader), layer and loss layer, number of optimisation steps; the request that was computed twice (the stream stopped
    in the write-time test of update 1531): are both computations identical?
 4. Code on disk vs the runs: modification times of the builder's driver files against the start / end of the runs they
    are said to have produced.
 5. Collapsed logs: exact-match hits of the deployed model whose target is one of the tokens the collapsed model emits
    everywhere (new objects of the first-token-subject updates).
 6. Single-edit files: public format; 'pre' of the single-edit run vs the pre-edit description pass; exploding single edits.
 7. Stale fields of the development log's meta.json.
 8. The builder's driver against the public experiments/evaluate.py: the calls of the public loop, compared as syntax trees.
"""
import ast
import calendar
import collections
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

out_path = os.path.join(A.AOUT, "extra.json")
out = A.jload(out_path) if os.path.exists(out_path) else {}
out["written"] = A.now()
data = A.load_counterfact()
B_LOGS = os.path.join(A.TP, "logs")

# ---- 1. statistics, fresh download
fresh = os.path.join(A.ATMP, "dl", "stats_fresh.npz")
if os.path.exists(fresh):
    h = A.file_sha256(fresh)
    out["statistics_fresh_download"] = {"url": "https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/" + os.path.basename(A.STATS_FILE),
                                        "bytes": os.path.getsize(fresh), "sha256": h, "local_sha256": A.file_sha256(A.STATS_FILE),
                                        "equal_to_local": h == A.file_sha256(A.STATS_FILE), "at": A.now(),
                                        "note": "downloaded by the audit in resumed pieces (the first attempt of cpu_provenance.py timed out)"}
    os.remove(fresh)
A.say("statistics:", out.get("statistics_fresh_download"))

# ---- 2. hash chain
W0 = A.load_w0()
chain = {}
for src in ("dev", "stream"):
    fac = A.load_factors(src)
    rec = {}
    for line in open(os.path.join(A.B_LOGS_EDIT, src, "progress.jsonl")):
        r = json.loads(line)
        if r.get("w_sha256"):
            rec[r["n_applied"]] = r["w_sha256"]
    W = W0.clone()
    ok, bad = 0, []
    for i in range(fac["left"].shape[0]):
        W += fac["left"][i].unsqueeze(1) @ fac["right"][i].unsqueeze(0)
        if (i + 1) in rec:
            if A.tensor_sha256(W) == rec[i + 1]:
                ok += 1
            else:
                bad.append(i + 1)
    chain[src] = {"n_recorded_hashes": len(rec), "n_reproduced_by_audit_replay": ok, "not_reproduced_at": bad, "recorded_at": sorted(rec)[:6] + ["..."] + sorted(rec)[-2:]}
out["hash_chain"] = chain
A.say("hash chain:", chain)
A.jdump(out, out_path)

# ---- 3. public printouts
pp = {}
for name, logf in (("dev", "dev_public_stdout.log"), ("stream", "stream_public_stdout.log"), ("single", "single_run_000_public_stdout.log")):
    txt = open(os.path.join(B_LOGS, logf), errors="replace").read()
    blocks = txt.split("Executing ROME algorithm for the update: ")[1:]
    steps = collections.Counter(len(re.findall(r"^loss ", b, flags=re.M)) for b in blocks if "Deltas successfully computed" in b)
    pp[name] = {"n_blocks": len(blocks), "n_complete_blocks": int(sum("Deltas successfully computed" in b for b in blocks)),
                "n_'Loading cached' (statistics read from the file)": len(re.findall(r"Loading cached (\S+)", txt)),
                "statistics_files_loaded": sorted(set(re.findall(r"Loading cached (\S+)", txt))),
                "n_'Attempting to download'": txt.count("Attempting to download"), "n_'Computing locally'": txt.count("Computing locally"),
                "n_'Retrieving inverse covariance' (once per process)": txt.count("Retrieving inverse covariance statistics"),
                "n_'Cached context templates' (templates sampled; once per process that sampled)": txt.count("Cached context templates"),
                "rewrite_layers": sorted(set(re.findall(r"Rewrite layer is (\d+)", txt))), "loss_layers": sorted(set(re.findall(r"Tying optimization objective to (\d+)", txt))),
                "optimisation_steps_per_update": dict(sorted(steps.items()))}
    if name == "stream":
        heads = collections.Counter(b.split("\n", 1)[0] for b in blocks if "Deltas successfully computed" in b)
        dup = [h_ for h_, c in heads.items() if c > 1]
        d = []
        for h_ in dup:
            bs = [b for b in blocks if b.split("\n", 1)[0] == h_ and "Deltas successfully computed" in b]

            def nums(b):
                return re.findall(r"^loss .*$|Division Factor: .*$|Right vector norm: .*$|Delta norm: .*$", b, flags=re.M)
            idx = [i for i in range(2000) if f"[{data[i]['requested_rewrite']['prompt'].format(data[i]['requested_rewrite']['subject'])}]" in h_]
            d.append({"request": h_, "stream_index": idx, "n_computations": len(bs), "all_printed_numbers_identical": all(nums(b) == nums(bs[0]) for b in bs),
                      "n_printed_lines_compared": len(nums(bs[0]))})
        pp[name]["requests_computed_more_than_once"] = d
out["public_printouts"] = pp
A.say("public printouts:", json.dumps(pp)[:2500])
A.jdump(out, out_path)


# ---- 4. code on disk vs the runs
def ts(s):
    return calendar.timegm(time.strptime(s.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S"))


def mt(f):
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(os.path.getmtime(os.path.join(A.B_CODE, f))))


runs = {}
for src in ("dev", "stream"):
    P = [json.loads(l) for l in open(os.path.join(A.B_LOGS_EDIT, src, "progress.jsonl"))]
    m = A.jload(os.path.join(A.B_LOGS_EDIT, src, "meta.json"))
    gaps = [(a["n_applied"], a["at"], b["at"]) for a, b in zip(P, P[1:]) if ts(b["at"]) - ts(a["at"]) > 120]
    runs[src] = {"started": m["started"], "first_update_written": P[0]["at"], "last_update_written": P[-1]["at"], "resumed": m.get("resumed", []),
                 "pauses_longer_than_2_min(after update, from, to)": gaps}
sm = A.jload(os.path.join(A.B_OUT, "public_results", "ROME_single", "run_000", "run_meta.json"))
runs["single"] = {"started": sm["started"], "finished": sm["finished"]}
drift = {"mtime": {f: mt(f) for f in ("tp_common.py", "run_stream.py", "run_single.py")}, "runs": runs}
t_common, t_stream, t_single = (os.path.getmtime(os.path.join(A.B_CODE, f)) for f in ("tp_common.py", "run_stream.py", "run_single.py"))
P = [json.loads(l) for l in open(os.path.join(A.B_LOGS_EDIT, "stream", "progress.jsonl"))]
n_before = sum(ts(p["at"]) < t_stream for p in P)
resumed_at = [r["n_have"] for r in runs["stream"]["resumed"]]
first_proc_end = resumed_at[0] if resumed_at else len(P)
drift["tp_common.py_modified_after_dev_log_complete"] = t_common > ts(runs["dev"]["last_update_written"])
drift["tp_common.py_modified_after_single_edit_run_finished"] = t_common > ts(runs["single"]["finished"])
drift["run_single.py_modified_after_single_edit_run_started"] = t_single > ts(runs["single"]["started"])
drift["run_stream.py_modified_while_first_stream_process_was_running"] = ts(runs["stream"]["started"]) < t_stream < ts(P[first_proc_end - 1]["at"])
drift["stream_updates_written_by_the_process_started_before_the_last_modification"] = first_proc_end
drift["stream_updates_written_before_the_modification_time"] = n_before
drift["earlier_versions_kept"] = sorted(glob.glob(os.path.join(A.B_CODE, "*.bak*")) + glob.glob(os.path.join(A.B_CODE, "*.orig")) + glob.glob(os.path.join(A.B_CODE, "*~")))
drift["statement"] = ("tp_common.py and run_stream.py on disk were last modified during the first process of the reported stream; that process (updates 0 .. %d), "
                      "the development log and the single-edit run had loaded earlier versions, of which no copy is kept" % (first_proc_end - 1))
out["code_on_disk_vs_runs"] = drift
A.say("code drift:", json.dumps(drift)[:1800])

# ---- 5. collapsed logs: spurious exact-match hits
fac = A.load_factors("stream")
s = (fac["left"].double() * fac["k_request"].double()).sum(1)
fro = fac["left"].double().norm(dim=1) * fac["right"].double().norm(dim=1)
prom = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
sp = {}
for T in (100, 500, 1000, 2000):
    name = f"TX-G2XL-rome-T{T}"
    ftok = sorted({data[i]["requested_rewrite"]["target_new"]["str"] for i in range(T) if float(s[i]) < 0.5})          # first-token-subject updates
    expl = sorted({data[i]["requested_rewrite"]["target_new"]["str"] for i in range(T) if float(fro[i]) > 117.13})
    rw = A.jload(os.path.join(A.B_OUT, "text_logs_rows", f"{name}__deployed.json"))["edits"]
    pre = {x["prompt_id"]: x for x in A.jload(os.path.join(A.B_OUT, "text_logs_rows", f"{name}__pre_edit.json"))["edits"]}
    res = {"objects_of_first_token_subject_updates": ftok, "n_objects_of_exploding_updates": len(expl)}
    for kind in ("request", "paraphrase"):
        rows = [x for x in rw if x["kind"] == kind]
        hits = [x for x in rows if x["new_dec_exact"]]
        tgt = lambda x: data[x["case_id"]]["requested_rewrite"]["target_new"]["str"]   # noqa: E731
        res[kind] = {"n": len(rows), "n_exact": len(hits), "n_exact_whose_target_is_an_object_of_a_first_token_subject_update": int(sum(tgt(x) in ftok for x in hits)),
                     "n_exact_whose_target_is_an_object_of_an_exploding_update": int(sum(tgt(x) in expl for x in hits)),
                     "n_exact_that_the_pre_edit_model_also_has": int(sum(pre[x["prompt_id"]]["new_dec_exact"] for x in hits)),
                     "n_rows_whose_target_is_an_object_of_an_exploding_update": int(sum(tgt(x) in expl for x in rows)),
                     "n_exact_pre_edit": int(sum(pre[x["prompt_id"]]["new_dec_exact"] for x in rows))}
    sp[name] = res
out["collapsed_logs_exact_match_hits"] = sp
A.say("exact-match hits on the stream logs:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("request", "paraphrase")} for k, v in sp.items()}))
A.jdump(out, out_path)

# ---- 6. single-edit files
B_RUN = os.path.join(A.B_OUT, "public_results", "ROME_single", "run_000")
keys = collections.Counter()
same_pre, n_cmp = 0, 0
for i in range(1000):
    d = A.jload(os.path.join(B_RUN, f"case_{i}.json"))
    keys[tuple(sorted(d.keys()))] += 1
    w = os.path.join(A.B_OUT, "public_results", "ROME_seq", "run_W0", f"case_{i}.json")
    if os.path.exists(w):
        n_cmp += 1
        same_pre += d["pre"] == A.jload(w)["post"]
sing = {"keys_of_case_files": {",".join(k): v for k, v in keys.items()}, "public_format_keys": "case_id,post,pre,requested_rewrite,time",
        "pre_of_single_run_equals_pre_edit_description_pass": [int(same_pre), n_cmp], "exploding_single_edits": []}
W0n = float(W0.double().norm())
for i in range(1000):
    t = torch.load(os.path.join(B_RUN, "cases_factors", f"case_{i}.pt"), map_location="cpu")["tensors"]
    f = float(t["left"].double().norm() * t["right"].double().norm())
    if f > W0n:
        d = A.jload(os.path.join(B_RUN, f"case_{i}.json"))["post"]
        sing["exploding_single_edits"].append({"record": i, "update_fro": f, "division_factor": float(t["left"].double() @ t["k_request"].double()),
                                               "ES": float(np.mean([x["target_true"] > x["target_new"] for x in d["rewrite_prompts_probs"]])),
                                               "PS": float(np.mean([x["target_true"] > x["target_new"] for x in d["paraphrase_prompts_probs"]])),
                                               "NS": float(np.mean([x["target_true"] < x["target_new"] for x in d["neighborhood_prompts_probs"]]))})
out["single_edit_files"] = sing
A.say("single-edit files:", json.dumps(sing)[:1200])

# ---- 7. stale meta of the development log
m = A.jload(os.path.join(A.B_LOGS_EDIT, "dev", "meta.json"))
out["dev_meta_stale_fields"] = {"what": m["what"], "T_planned": m["T_planned"], "T_done": m.get("T_done"), "n_update_files": len(glob.glob(os.path.join(A.B_LOGS_EDIT, "dev", "updates", "u_*.pt"))),
                                "stale": m["T_planned"] != m.get("T_done")}
A.say("dev meta:", out["dev_meta_stale_fields"])


# ---- 8. the builder's single-edit driver vs the public experiments/evaluate.py (calls of the loop, as syntax trees)
def calls(path, names):
    tree = ast.parse(open(path).read())
    res = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = ast.unparse(node.func)
            if fn in names:
                res.setdefault(fn, []).append(ast.unparse(node))
    return res


pub = calls(os.path.join(A.ROME_REPO, "experiments", "evaluate.py"), {"apply_algo", "ds_eval_method", "nethook.get_parameter", "ds_class", "AutoModelForCausalLM.from_pretrained",
                                                                       "AutoTokenizer.from_pretrained", "params_class.from_json", "shutil.copyfile", "json.dump"})
bld = calls(os.path.join(A.B_CODE, "run_single.py"), {"apply_rome_to_model", "compute_rewrite_quality_counterfact", "nethook.get_parameter", "torch.manual_seed",
                                                      "C.load_model", "C.load_hparams", "C.load_counterfact", "C.jdump", "shutil.copyfile"})
strm = calls(os.path.join(A.B_CODE, "run_stream.py"), {"apply_rome_to_model", "C.public_prob_test", "torch.manual_seed", "upd_matrix_match_shape", "compute_u"})
pubm = open(os.path.join(A.ROME_REPO, "rome", "rome_main.py")).read()
cmp8 = {"public_evaluate.py": pub, "builder_run_single.py": bld, "builder_run_stream.py": strm,
        "edit_call_equal_up_to_names": bld["apply_rome_to_model"] == ["apply_rome_to_model(model, tok, [record['requested_rewrite']], hparams, copy=False, return_orig_weights=True)"]
        and pub["apply_algo"] == ["apply_algo(model, tok, [record['requested_rewrite']], hparams, copy=False, return_orig_weights=True, **args_conserve_memory)"],
        "post_and_pre_evaluation_calls_equal_up_to_names": sorted(bld["compute_rewrite_quality_counterfact"]) == sorted(
            c.replace("ds_eval_method", "compute_rewrite_quality_counterfact") for c in pub["ds_eval_method"]),
        "restore_statement_equal": ("nethook.get_parameter(model, k)[...] = v.to(\"cuda\")" in open(os.path.join(A.ROME_REPO, "experiments", "evaluate.py")).read()
                                    and "nethook.get_parameter(model, k)[...] = v.to(\"cuda\")" in open(os.path.join(A.B_CODE, "run_single.py")).read()),
        "sequential_call": strm["apply_rome_to_model"],
        "replay_expression_in_builder_resume": "u[\"tensors\"][\"left\"].unsqueeze(1) @ u[\"tensors\"][\"right\"].unsqueeze(0)" in open(os.path.join(A.B_CODE, "run_stream.py")).read(),
        "public_write_expression": "upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)" in pubm and "w[...] += upd_matrix" in pubm}
out["driver_vs_public_calls"] = cmp8
A.say("driver vs public:", {k: v for k, v in cmp8.items() if isinstance(v, bool)}, "sequential call:", cmp8["sequential_call"])
out["finished"] = A.now()
A.jdump(out, out_path)
