"""Audit: (1) integrity of the pip --target stacks the build used (every installed file against the sha256 of the wheel's
RECORD), (2) repeated subjects / repeated request prompts inside the logs, (3) how the squared update norm of each log is
distributed over its updates, (4) the project environment and [redacted: related project] untouched since the build started.
CPU; stored files only.  Output: out/extra3.json
"""
import base64
import collections
import csv
import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

out = {"written": A.now()}
out_path = os.path.join(A.AOUT, "extra3.json")

# ---- 1. installed packages against their RECORD
pk = {}
for root in (A.PYDEPS_TF4, A.PYDEPS):
    for d in sorted(os.listdir(root)):
        if not d.endswith(".dist-info"):
            continue
        rec = os.path.join(root, d, "RECORD")
        n = ok = missing = nohash = 0
        bad = []
        for row in csv.reader(open(rec, newline="")):
            path, h = row[0], row[1]
            p = os.path.normpath(os.path.join(root, path))
            if path.endswith(".pyc") or "__pycache__" in path or not h:
                nohash += 1
                continue
            n += 1
            if not os.path.isfile(p):
                missing += 1
                continue
            algo, b64 = h.split("=", 1)
            dig = base64.urlsafe_b64encode(hashlib.new(algo, open(p, "rb").read()).digest()).rstrip(b"=").decode()
            if dig == b64:
                ok += 1
            else:
                bad.append(path)
        pk[d] = {"root": root, "n_files_with_hash": n, "n_equal_to_wheel_record": ok, "n_missing": missing, "modified_files": bad[:20],
                 "missing_are_only_bin_scripts": None}
        if missing:
            miss = [row[0] for row in csv.reader(open(rec, newline="")) if row[1] and not os.path.isfile(os.path.normpath(os.path.join(root, row[0])))]
            pk[d]["missing_files"] = miss[:10]
out["pip_target_packages"] = pk
out["pip_target_packages_unmodified"] = all(v["n_equal_to_wheel_record"] + v["n_missing"] == v["n_files_with_hash"] and not v["modified_files"] for v in pk.values())
A.say("pip --target packages:", {k: {a: b for a, b in v.items() if a != "root"} for k, v in pk.items()})

# ---- 2. repeated subjects / request prompts
data = A.load_counterfact()
rep = {}
for name, (src, T, first) in A.LOGS.items():
    subj = [data[first + i]["requested_rewrite"]["subject"] for i in range(T)]
    text = [data[first + i]["requested_rewrite"]["prompt"].format(subj[i]) for i in range(T)]
    cs, ct = collections.Counter(subj), collections.Counter(text)
    rep[name] = {"T": T, "n_distinct_subjects": len(cs), "n_requests_whose_subject_occurred_earlier": T - len(cs),
                 "n_requests_whose_request_prompt_occurred_earlier": T - len(ct),
                 "examples": [(s, [i for i in range(T) if subj[i] == s]) for s, c in cs.items() if c > 1][:4]}
out["repeated_subjects"] = rep
A.say("repeated subjects:", {k: {a: b for a, b in v.items() if a != "examples"} for k, v in rep.items()})

# ---- 3. squared update norms
tok_idx = A.jload(os.path.join(A.AOUT, "recon.json"))["logs"]
nrm = {}
for name, (src, T, first) in A.LOGS.items():
    fac = A.load_factors(src)
    fro = (fac["left"][:T].double().norm(dim=1) * fac["right"][:T].double().norm(dim=1)).numpy()
    ft = tok_idx[name]["updates"]["subject_is_first_token"]
    sq = fro ** 2
    order = np.argsort(-sq)
    nrm[name] = {"T": T, "sum_of_squared_update_norms": float(sq.sum()), "sqrt_of_it": float(np.sqrt(sq.sum())),
                 "first_token_subject_updates": ft, "their_share_of_the_summed_squared_norm_pct": float(100 * sq[ft].sum() / sq.sum()) if ft else 0.0,
                 "largest_update_share_pct": float(100 * sq[order[0]] / sq.sum()), "largest_update_index": int(order[0]),
                 "median_update_norm": float(np.median(fro)), "max_update_norm": float(fro.max()),
                 "deployed_minus_pre_edit_fro": tok_idx[name]["reconstruction_fp64"]["update_fro"]}
out["update_norm_concentration"] = nrm
A.say("update norms:", nrm)

# ---- 4. nothing of [redacted: related project] (models, data, environment) changed since the build started (04:40 UTC)
r = subprocess.run(["find", "[redacted: local path]", "-newermt", "2026-11-11 04:40:00 UTC", "-type", "f"], capture_output=True, text=True)
files = [l for l in r.stdout.splitlines() if l]
now = __import__("time").time()
out["[redacted: related project]_files_modified_since_build_start"] = {"n": len(files), "files": files[:10],
                                                     "with_mtime_in_the_future(not written today)": [f for f in files if os.path.getmtime(f) > now + 86400]}
home = os.path.expanduser("~")
r = subprocess.run(["find", os.path.join(home, ".cache"), "-newermt", "2026-11-11 04:40:00 UTC", "-type", "f"], capture_output=True, text=True)
out["home_cache_files_newer_than_build_start"] = {"home": home, "n": len([l for l in r.stdout.splitlines() if l]), "files": r.stdout.splitlines()[:10]}
out["finished"] = A.now()
A.jdump(out, out_path)
A.say("[redacted: related project] modified:", out["[redacted: related project]_files_modified_since_build_start"], "| home cache:", out["home_cache_files_newer_than_build_start"])
