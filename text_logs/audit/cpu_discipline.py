"""Audit check 6: discipline.  No gating / repair method may have been run on (or looked at for) a reported log
(TX-G2XL-rome-T100 / T500 / T1000 / T2000, records from 0); the development log (records from 15000) is the only log of the
family for that.  CPU; reads code, outputs and the queue's event log; runs nothing.

 a. Every GPU job of the build (queue event log, job names tx_*), and the script each ran (from the builder's run logs).
 b. Static scan of the builder's code: which scripts import the repair library (rescope) or use a gate object, and on
    which log names they operate.
 c. Scan of the builder's outputs for gate-level content (thresholds, firing, released edits, calibration) and the log
    it belongs to.
 d. Detector-related quantities that the build DID compute on the reported stream (no threshold, no decision): listed
    exactly, so that the reader can judge.
 e. Other agents: anything outside text_public that touches the reported logs with a gate or a repair (10_final/e10_text,
    10_final/cache/text) at the time of the scan.
Output: out/discipline.json
"""
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import torch  # noqa: E402

out = {"written": A.now()}
REPORTED = ["TX-G2XL-rome-T100", "TX-G2XL-rome-T500", "TX-G2XL-rome-T1000", "TX-G2XL-rome-T2000"]
GATE_WORDS = r"ScopedLog|Hook\(|rescope\.|from rescope|rebase|calibrat|threshold|\btau\b|null_repair|blocklist|rb_suffix|comp_min|cosine|\.scores\(|fire="

# ---- a. GPU jobs of the build
ev = os.path.join(A.ROOT, "tools", "gpuq", "state", "events.log")
jobs = {}
for line in open(ev):
    try:
        e = json.loads(line)
    except Exception:  # noqa: BLE001
        continue
    n = e.get("name", "")
    if e.get("event") == "start" and (n.startswith("tx_") or n.startswith("e10")):
        jobs.setdefault(n, []).append(e["time"])
out["gpu_jobs"] = {"builder(tx_*)": {k: v for k, v in jobs.items() if k.startswith("tx_")},
                   "audit(e10aud_*)": {k: v for k, v in jobs.items() if k.startswith("e10aud")},
                   "other_agents(e10t_*)": {k: v for k, v in jobs.items() if k.startswith("e10t")}}
# the script behind each builder job, from the 'started ...' line the queue writes into the builder's run logs
started = {}
for f in sorted(glob.glob(os.path.join(A.TP, "logs", "*.out"))):
    for line in open(f, errors="replace"):
        m = re.match(r"\[gq [0-9:]+\] started (\S+) on card \d+ after \d+ s in the queue: (.*)$", line.rstrip())
        if m:
            cmd = m.group(2)
            started.setdefault(m.group(1), set()).add(re.sub(r"^\S*python\S*\s+", "", cmd).replace(A.B_CODE + "/", ""))
out["builder_job_commands"] = {k: sorted(v) for k, v in sorted(started.items())}
out["builder_jobs_without_a_logged_command"] = sorted(set(out["gpu_jobs"]["builder(tx_*)"]) - set(started))

# ---- b. static scan of the builder's code
def code_only(src):
    """Python source without comments and without string literals (docstrings, messages, descriptions): what is executed."""
    import io
    import tokenize
    keep = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            keep.append(t.string)
    except tokenize.TokenError:
        return src
    return " ".join(keep)


code = {}
for f in sorted(glob.glob(os.path.join(A.B_CODE, "*.py")) + glob.glob(os.path.join(A.B_CODE, "*.sh"))):
    src = open(f, errors="replace").read()
    body = code_only(src) if f.endswith(".py") else "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    hits = sorted(set(re.findall(GATE_WORDS, body)))
    if hits:
        code[os.path.basename(f)] = {
            "gate_words_in_executable_text": hits,
            "imports_rescope": bool(re.search(r"^\s*(import rescope|from rescope)", body, flags=re.M)),
            "uses_ScopedLog_or_Hook": bool(re.search(r"ScopedLog|\bHook\b", body)),
            "log_selection": sorted(set(re.findall(r"NAME = TX \. \w+|TX \. DEV_LOG|TX \. available \( \)|TX \. NAMES|TX \. REPORTED", body))),
            "asserts_dev_log": bool(re.search(r'assert NAME == "TX-G2XL-rome-dev100"', src))}
out["builder_code"] = code
out["builder_scripts_using_a_gate_object"] = sorted(k for k, v in code.items() if v["uses_ScopedLog_or_Hook"])
out["builder_scripts_using_a_gate_object_all_restricted_to_dev_log"] = all(code[k]["asserts_dev_log"] for k in out["builder_scripts_using_a_gate_object"])

# ---- c. gate-level content in the builder's outputs
found = {}
for f in sorted(glob.glob(os.path.join(A.B_OUT, "*.json")) + glob.glob(os.path.join(A.B_OUT, "*.txt")) + glob.glob(os.path.join(A.TP, "logs", "*.out"))
                + glob.glob(os.path.join(A.B_LOGS_EDIT, "*.json")) + glob.glob(os.path.join(A.B_LOGS_EDIT, "*", "meta.json"))):
    if os.path.getsize(f) > 60e6:
        continue
    txt = open(f, errors="replace").read()
    hits = sorted(set(m.lower() for m in re.findall(r"threshold|\"tau\"|released|firing|fires|false firing|coverage|calibrat|ScopedLog|null repair", txt)))
    if hits:
        ctx = [re.sub(r"\s+", " ", txt[max(0, m.start() - 70):m.end() + 50]) for m in list(re.finditer(r"threshold|\"tau\"|released|firing|fires|calibrat|ScopedLog", txt))[:3]]
        found[os.path.relpath(f, A.TP)] = {"words": hits, "reported_log_names_in_file": [n for n in REPORTED if n in txt], "first_contexts": ctx}
out["builder_outputs_with_gate_words"] = found
lc = A.jload(os.path.join(A.B_OUT, "library_check.json"))
out["library_check"] = {"log": lc["log"], "is_development_log": lc["log"] == "TX-G2XL-rome-dev100",
                        "sections": [k for k in lc if k[:2] in ("a_", "b_", "c_", "d_", "e_", "f_")]}
# text_logs_table.txt: the only gate words are in section 7 (development log)?
tbl = open(os.path.join(A.B_OUT, "text_logs_table.txt")).read().split("\n")
gate_lines = [(i + 1, l.strip()[:150]) for i, l in enumerate(tbl) if re.search(r"threshold|released|hook with|linear replay|rebase", l)]
sec7 = next(i for i, l in enumerate(tbl) if l.startswith("7. LIBRARY FORMAT CHECK"))
sec8 = next(i for i, l in enumerate(tbl) if l.startswith("8. "))
out["table_gate_lines"] = {"lines": gate_lines, "all_inside_section_7(development log only)": all(sec7 < i <= sec8 for i, _ in gate_lines)}

# ---- d. detector-related quantities computed on the reported stream by the build (no threshold, no decision)
fac = A.load_factors("stream")
sc = fac["scalars"][0]
out["detector_quantities_on_the_reported_stream"] = {
    "per_update_scalars_in_logs_edit/stream/factors.pt": [k for k in sc if k.startswith("coord_") or k.startswith("w_dot")],
    "n_updates": len(fac["scalars"]), "n_values_per_update(coord_templates)": len(sc["coord_templates"]),
    "what": "normalised coordinate <left, k> / <left, k_request> of the edit's OWN subject token under the editor's own 21 context templates "
            "(the request prompt behind 20 sampled prefixes), recorded while the edit was written; their mean is the 'template-averaged key' "
            "coordinate printed for all five logs in section 1 of out/text_logs_table.txt (q05 / median / q95 / max, share within 1 % / 10 %)",
    "is_a_gate_decision": False, "threshold_applied": False, "model_run_under_a_gate": False,
    "library_constructor_on_reported_logs": "export_library_format.py: rescope.loglib.from_testbed(<dir>) for all five logs, then own() (= 1) and "
                                            "delta() (= W_E - W_0): format check of the stored factors; no ScopedLog, no Hook, no calibration"}

# ---- e. other agents (at the time of this scan)
oth = {"scan_time": A.now()}
e10 = os.path.join(A.ROOT, "experiments", "10_final", "e10_text")
if os.path.isdir(e10):
    files = [os.path.relpath(os.path.join(r, f), e10) for r, _, fs in os.walk(e10) for f in fs if "__pycache__" not in r]
    oth["10_final/e10_text"] = {"n_files": len(files), "out_dir_files": sorted(f for f in files if f.startswith("out/"))[:40],
                                "store_files": sorted(f for f in files if f.startswith("store/"))[:40],
                                "logs": sorted(f for f in files if f.startswith("logs/")),
                                "tests": sorted(f for f in files if f.startswith("tests/"))[:60]}
    rep = {}
    for f in files:
        p = os.path.join(e10, f)
        if os.path.getsize(p) > 20e6 or not f.endswith((".json", ".out", ".txt", ".jsonl", ".py")):
            continue
        txt = open(p, errors="replace").read()
        names = [n for n in REPORTED if n in txt]
        if names and not f.endswith(".py"):
            rep[f] = names
    oth["10_final/e10_text"]["non_code_files_naming_a_reported_log"] = rep
    opens = {}
    for f in files:
        if f.endswith(".py") and (f.startswith("tests/") or f.startswith("scripts/")):
            src = open(os.path.join(e10, f), errors="replace").read()
            opens[f] = sorted(set(re.findall(r"open_log\(([^)]*)\)|from_public\(([^)]*)\)|--log (\S+)", src)))
            opens[f] = sorted({x for t in opens[f] for x in t if x})
    oth["10_final/e10_text"]["logs_opened_by_test_and_script_files"] = opens
    oth["10_final/e10_text"]["method_modules_are_placeholders"] = {m: os.path.getsize(os.path.join(e10, "code", "e10t", m)) < 200
                                                                   for m in ("gate.py", "methods.py", "behave.py") if os.path.exists(os.path.join(e10, "code", "e10t", m))}
cache = os.path.join(A.ROOT, "experiments", "10_final", "cache", "text", "gpt2-xl")
oth["10_final/cache/text/gpt2-xl parts"] = sorted(os.listdir(cache)) if os.path.isdir(cache) else None
oth["key_cache_parts_of_the_reported_stream"] = [p for p in (oth["10_final/cache/text/gpt2-xl parts"] or []) if p.startswith("stream")]
out["other_agents"] = oth
out["verdict_inputs"] = {
    "gate_or_repair_on_reported_log_by_builder": False if out["builder_scripts_using_a_gate_object_all_restricted_to_dev_log"]
    and out["library_check"]["is_development_log"] and out["table_gate_lines"]["all_inside_section_7(development log only)"] else None}
A.jdump(out, os.path.join(A.AOUT, "discipline.json"))
for k in ("gpu_jobs", "builder_job_commands", "builder_jobs_without_a_logged_command", "builder_code", "builder_scripts_using_a_gate_object",
          "builder_scripts_using_a_gate_object_all_restricted_to_dev_log", "builder_outputs_with_gate_words", "library_check", "table_gate_lines",
          "other_agents", "verdict_inputs"):
    A.say(k, json.dumps(out[k])[:1500])
