"""Audit check 1: provenance of the public code, the hyper-parameter file, the model, the data and the statistics.
CPU + network (github.com, codeload.github.com, hf-mirror.com, rome.baulab.info).  Every remote step is recorded with
its outcome; a remote failure is reported as "not verified", never as a pass.  Output: out/provenance.json
"""
import hashlib
import json
import os
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

DL = os.path.join(A.ATMP, "dl")
os.makedirs(DL, exist_ok=True)
COMMIT = "0874014cd9837e4365f3e6f3c71400ef11509e04"       # the builder's statement
out_path = os.path.join(A.AOUT, "provenance.json")
ONLY_CODE = "--only-code" in sys.argv          # recompute section 1 only and merge it into the existing record
out = A.jload(out_path) if ONLY_CODE and os.path.exists(out_path) else {"written": A.now(), "builder_stated_commit": COMMIT}


def run(cmd, timeout=120, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def git(*a, timeout=120):
    return run(["git", "-c", "safe.directory=*", "-C", A.ROME_REPO, *a], timeout=timeout)


def curl(url, dest, timeout=900):
    rc, so, se = run(["curl", "-sS", "-L", "--retry", "3", "--retry-delay", "5", "-m", str(timeout), "-o", dest, url], timeout=timeout + 60)
    return rc == 0 and os.path.exists(dest), se.strip()[:300]


def git_blob_sha1(path):
    h = hashlib.sha1()
    h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- 1. public code
c = {}
c["HEAD"] = git("rev-parse", "HEAD")[1].strip()
c["HEAD_equals_stated_commit"] = c["HEAD"] == COMMIT
c["commit_date"] = git("log", "-1", "--format=%ci")[1].strip()
c["status_porcelain"] = git("status", "--porcelain")[1]
c["status_ignored"] = [l for l in git("status", "--porcelain", "--ignored")[1].splitlines()]
c["diff_HEAD_names"] = git("diff", "HEAD", "--name-only")[1].split()
c["fsck_rc"] = git("fsck", "--no-dangling")[0]
# tracked files: blob id of the commit against a hash of the working-tree file computed here (not git's stat cache)
rc, so, _ = git("ls-tree", "-r", "HEAD")
tracked, bad, missing, n_links = {}, [], [], 0
for line in so.splitlines():
    meta, path = line.split("\t", 1)
    mode, typ, oid = meta.split()
    if typ != "blob":
        continue
    tracked[path] = oid
    p = os.path.join(A.ROME_REPO, path)
    if mode == "120000":                               # a symbolic link: the blob is the link target
        tgt = os.readlink(p).encode() if os.path.islink(p) else None
        if tgt is None:
            missing.append(path)
        elif hashlib.sha1(b"blob %d\0" % len(tgt) + tgt).hexdigest() != oid:
            bad.append(path)
        n_links += 1
    elif os.path.islink(p) or not os.path.isfile(p):
        missing.append(path)
    elif git_blob_sha1(p) != oid:
        bad.append(path)
c["n_tracked_files"] = len(tracked)
c["n_tracked_symbolic_links"] = n_links
c["tracked_files_whose_content_differs_from_commit"] = bad
c["tracked_files_missing"] = missing
untracked = []
for root, dirs, files in os.walk(A.ROME_REPO):
    if ".git" in dirs:
        dirs.remove(".git")
    for fn in files:
        rel = os.path.relpath(os.path.join(root, fn), A.ROME_REPO)
        if rel not in tracked:
            untracked.append(rel)
c["untracked_files"] = {"n": len(untracked), "n_pyc_in___pycache__": sum("__pycache__" in u and u.endswith(".pyc") for u in untracked),
                        "others": [u for u in untracked if not ("__pycache__" in u and u.endswith(".pyc"))]}
# the remote: which commit do the public branches point to today
for _try in range(4):                                   # github.com is slow from this machine: a few attempts
    rc, so, se = run(["git", "ls-remote", "https://github.com/kmeng01/rome.git"], timeout=60)
    if rc == 0 and so.strip():
        break
c["ls_remote"] = {"ok": rc == 0, "refs": dict((l.split("\t")[1], l.split("\t")[0]) for l in so.splitlines()) if rc == 0 else None, "err": se[:200]}
if rc == 0:
    c["remote_HEAD_equals_clone_HEAD"] = c["ls_remote"]["refs"].get("HEAD") == c["HEAD"]
# the commit archive, fetched by the audit itself, against the working tree
tb = os.path.join(DL, f"rome-{COMMIT}.tar.gz")
ok, err = (True, "") if os.path.exists(tb) and os.path.getsize(tb) > 10000 else curl(f"https://codeload.github.com/kmeng01/rome/tar.gz/{COMMIT}", tb, 180)
c["archive_fetched_by_audit"] = {"ok": ok, "err": err}
if ok:
    c["archive_fetched_by_audit"]["sha256"] = A.file_sha256(tb)
    b_tb = os.path.join(A.EXT, "third_party", "_tarballs", f"rome-{COMMIT}.tar.gz")
    c["archive_fetched_by_audit"]["builder_archive_sha256"] = A.file_sha256(b_tb) if os.path.exists(b_tb) else None
    n_cmp, n_lnk, diff, absent, tar_names = 0, 0, [], [], set()
    with tarfile.open(tb, "r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() or m.issym()):
                continue
            rel = m.name.split("/", 1)[1]
            tar_names.add(rel)
            p = os.path.join(A.ROME_REPO, rel)
            if m.issym():
                n_lnk += 1
                if not os.path.islink(p):
                    absent.append(rel)
                elif os.readlink(p) != m.linkname:
                    diff.append(rel)
                continue
            data = tf.extractfile(m).read()             # read only; nothing is extracted or executed
            if os.path.islink(p) or not os.path.isfile(p):
                absent.append(rel)
                continue
            n_cmp += 1
            if hashlib.sha256(data).hexdigest() != A.file_sha256(p):
                diff.append(rel)
    c["archive_fetched_by_audit"].update({"n_files_compared": n_cmp, "n_links_compared": n_lnk, "files_differing_from_clone": diff,
                                          "files_absent_in_clone": absent, "tracked_not_in_archive": sorted(set(tracked) - tar_names),
                                          "archive_not_tracked": sorted(tar_names - set(tracked))})
arch = c["archive_fetched_by_audit"]
c["verdict_unmodified"] = bool(c["HEAD_equals_stated_commit"] and not bad and not missing and c["status_porcelain"] == "" and not c["diff_HEAD_names"]
                               and not c["untracked_files"]["others"]
                               and (not arch["ok"] or (not arch["files_differing_from_clone"] and not arch["files_absent_in_clone"]
                                                       and not arch["tracked_not_in_archive"] and not arch["archive_not_tracked"])))
c["note"] = ("untracked files are compiled byte code (__pycache__/*.pyc, ignored by the repository's .gitignore) written when the public "
             "modules were imported; they are not source files")
out["public_code"] = c
A.jdump(out, out_path)
A.say("public code:", {k: c[k] for k in ("HEAD", "HEAD_equals_stated_commit", "n_tracked_files", "tracked_files_whose_content_differs_from_commit",
                                         "verdict_unmodified")}, c["untracked_files"], c.get("remote_HEAD_equals_clone_HEAD"))
if ONLY_CODE:
    out["public_code_recomputed"] = A.now()
    A.jdump(out, out_path)
    sys.exit(0)

# ----------------------------------------------------------------------------- 2. hyper-parameters
h = {"file": A.HPARAMS_FILE, "sha256": A.file_sha256(A.HPARAMS_FILE), "git_blob": git_blob_sha1(A.HPARAMS_FILE),
     "git_blob_of_commit": tracked.get("hparams/ROME/gpt2-xl.json"), "content": A.jload(A.HPARAMS_FILE)}
h["equals_commit"] = h["git_blob"] == h["git_blob_of_commit"]
h["builder_stated_sha256_prefix_matches"] = h["sha256"].startswith("81a98b788b8f")
# the copies the builder's runs made / recorded
copies = {}
p = os.path.join(A.B_OUT, "public_results", "ROME_single", "run_000", "params.json")
copies["single_run_params.json_sha256_equal"] = os.path.exists(p) and A.file_sha256(p) == h["sha256"]
for src in ("dev", "stream"):
    copies[f"{src}_meta_hparams_equal"] = A.jload(os.path.join(A.B_LOGS_EDIT, src, "meta.json"))["hparams"] == h["content"]
h["copies_recorded_by_runs"] = copies
out["hparams"] = h
A.jdump(out, out_path)
A.say("hparams:", {k: v for k, v in h.items() if k != "content"})

# ----------------------------------------------------------------------------- 3. model files against the mirror's listing
m = {"dir": A.MODEL_DIR}
api = os.path.join(DL, "gpt2xl_api.json")
tree = os.path.join(DL, "gpt2xl_tree.json")
ok1, e1 = curl("https://hf-mirror.com/api/models/openai-community/gpt2-xl", api, 60)
ok2, e2 = curl("https://hf-mirror.com/api/models/openai-community/gpt2-xl/tree/main", tree, 60)
m["mirror_api_ok"] = [ok1, ok2, e1, e2]
listing = {}
if ok1:
    try:
        m["mirror_revision_main"] = json.load(open(api)).get("sha")
    except Exception as ex:  # noqa: BLE001
        m["mirror_api_parse_error"] = str(ex)[:200]
if ok2:
    try:
        for e in json.load(open(tree)):
            listing[e["path"]] = {"size": e.get("size"), "blob_oid": e.get("oid"), "lfs_sha256": (e.get("lfs") or {}).get("oid"),
                                  "lfs_size": (e.get("lfs") or {}).get("size")}
    except Exception as ex:  # noqa: BLE001
        m["mirror_tree_parse_error"] = str(ex)[:200]
rec = A.jload(os.path.join(A.MODEL_DIR, "DOWNLOAD_RECORD.json"))
m["builder_download_record_revision"] = rec["revision"]
m["revision_equals_mirror_main"] = (m.get("mirror_revision_main") == rec["revision"]) if ok1 else None
files = {}
for fn in sorted(os.listdir(A.MODEL_DIR)):
    p = os.path.join(A.MODEL_DIR, fn)
    if not os.path.isfile(p) or fn == "DOWNLOAD_RECORD.json":
        continue
    d = {"bytes": os.path.getsize(p)}
    L = listing.get(fn)
    if fn in ("model.safetensors", "pytorch_model.bin"):
        d["sha256"] = A.file_sha256(p)
        d["builder_recorded_lfs_sha256"] = rec["files"][fn]["lfs_sha256"]
        d["sha256_equals_builder_record"] = d["sha256"] == d["builder_recorded_lfs_sha256"]
        if L:
            d["mirror_lfs_sha256"] = L["lfs_sha256"]
            d["sha256_equals_mirror_listing"] = d["sha256"] == L["lfs_sha256"]
            d["size_equals_mirror_listing"] = d["bytes"] == (L["lfs_size"] or L["size"])
    else:
        d["git_blob"] = git_blob_sha1(p)
        if L:
            d["mirror_blob_oid"] = L["blob_oid"]
            d["blob_equals_mirror_listing"] = d["git_blob"] == L["blob_oid"]
            d["size_equals_mirror_listing"] = d["bytes"] == L["size"]
    files[fn] = d
m["files"] = files
m["files_in_mirror_listing_not_on_disk"] = sorted(set(listing) - set(files))
out["model"] = m
A.jdump(out, out_path)
A.say("model:", {fn: {k: v for k, v in d.items() if k.endswith("listing") or k.endswith("record")} for fn, d in files.items()},
      m.get("revision_equals_mirror_main"))

# ----------------------------------------------------------------------------- 4. data
d = {"file": A.COUNTERFACT, "bytes": os.path.getsize(A.COUNTERFACT), "sha256": A.file_sha256(A.COUNTERFACT)}
data = A.load_counterfact()
d["n_records"] = len(data)
d["case_ids_are_0_to_n_minus_1"] = [r["case_id"] for r in data] == list(range(len(data)))
d["first_record"] = {"case_id": data[0]["case_id"], "requested_rewrite": data[0]["requested_rewrite"]}
d["last_record"] = {"case_id": data[-1]["case_id"], "requested_rewrite": data[-1]["requested_rewrite"]}
d["keys"] = sorted(data[0].keys())
d["prompt_counts"] = {k: sorted({len(r[k]) for r in data}) for k in ("paraphrase_prompts", "neighborhood_prompts", "attribute_prompts", "generation_prompts")}
d["builder_stated_sha256_prefix_matches"] = d["sha256"].startswith("d01705612517")
fresh = os.path.join(DL, "counterfact_fresh.json")
ok, err = curl("https://rome.baulab.info/data/dsets/counterfact.json", fresh, 600)
d["fresh_download"] = {"ok": ok, "err": err}
if ok:
    d["fresh_download"].update({"bytes": os.path.getsize(fresh), "sha256": A.file_sha256(fresh)})
    d["fresh_download"]["equal_to_local"] = d["fresh_download"]["sha256"] == d["sha256"]
    os.remove(fresh)
d["rome_data_link_resolves_to_file"] = os.path.realpath(os.path.join(A.DATA_DIR, "counterfact.json")) == os.path.realpath(A.COUNTERFACT)
out["data"] = d
A.jdump(out, out_path)
A.say("data:", {k: v for k, v in d.items() if k not in ("first_record", "last_record", "keys")})

# ----------------------------------------------------------------------------- 5. statistics
import numpy as np  # noqa: E402

s = {"file": A.STATS_FILE, "bytes": os.path.getsize(A.STATS_FILE), "sha256": A.file_sha256(A.STATS_FILE)}
s["builder_stated_sha256_matches"] = s["sha256"] == "376f29d29b05fed1f5c48daa47b40edb87a239ec817a51bfe4cf68659749bc78"
link = os.path.join(A.DATA_DIR, "stats", "gpt2-xl", "wikipedia_stats", os.path.basename(A.STATS_FILE))
s["path_the_public_code_reads"] = link
s["path_resolves_to_file"] = os.path.realpath(link) == os.path.realpath(A.STATS_FILE)
fresh = os.path.join(DL, "stats_fresh.npz")
url = "https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/" + os.path.basename(A.STATS_FILE)
ok, err = curl(url, fresh, 900)
s["fresh_download"] = {"url": url, "ok": ok, "err": err}
if ok:
    s["fresh_download"].update({"bytes": os.path.getsize(fresh), "sha256": A.file_sha256(fresh)})
    s["fresh_download"]["equal_to_local"] = s["fresh_download"]["sha256"] == s["sha256"]
    os.remove(fresh)
z = np.load(A.STATS_FILE)
s["npz_keys"] = sorted(z.files)
M = z["mom2.mom2"]
cnt = float(z["mom2.count"])
s.update({"mom2_shape": list(M.shape), "mom2_dtype": str(M.dtype), "token_count": cnt, "sample_size": int(z["sample_size"]),
          "constructor": str(z["mom2.constructor"]), "finite": bool(np.isfinite(M).all()),
          "max_abs_asymmetry": float(np.abs(M - M.T).max()), "max_abs": float(np.abs(M).max())})
C = M.astype(np.float64) / cnt                    # runningstats.SecondMoment.moment(): mom2 / count
ev = np.linalg.eigvalsh((C + C.T) / 2)
s["second_moment"] = {"trace": float(np.trace(C)), "eig_min": float(ev.min()), "eig_max": float(ev.max()),
                      "condition_number": float(ev.max() / ev.min()) if ev.min() > 0 else None, "n_nonpositive_eigenvalues": int((ev <= 0).sum())}
s["recomputed_by_builder"] = False
out["statistics"] = s
A.jdump(out, out_path)
A.say("statistics:", {k: v for k, v in s.items() if k not in ("npz_keys",)})

# ----------------------------------------------------------------------------- 6. GPT-J (no GPU work; file check only)
g = {}
gd = os.path.join(A.EXT, "models", "gpt-j-6b")
if os.path.isdir(gd):
    g["bytes_files"] = {fn: os.path.getsize(os.path.join(gd, fn)) for fn in sorted(os.listdir(gd)) if os.path.isfile(os.path.join(gd, fn))}
    g["bytes_total"] = sum(g["bytes_files"].values())
    g["bytes_total_without_download_record"] = g["bytes_total"] - g["bytes_files"].get("DOWNLOAD_RECORD.json", 0)
    g["builder_stated_total_bytes"] = 24210465421
    rec = A.jload(os.path.join(gd, "DOWNLOAD_RECORD.json"))
    g["repo"], g["revision"] = rec.get("repo"), rec.get("revision")
    tj = os.path.join(DL, "gptj_tree.json")
    ok, err = curl(f"https://hf-mirror.com/api/models/{rec.get('repo')}/tree/{rec.get('revision')}", tj, 60)
    g["mirror_listing_ok"] = ok
    if ok:
        try:
            L = {e["path"]: e for e in json.load(open(tj))}
            e = L.get("pytorch_model.bin", {})
            g["mirror_lfs_sha256"] = (e.get("lfs") or {}).get("oid")
            g["mirror_lfs_size"] = (e.get("lfs") or {}).get("size")
            g["size_equals_mirror_listing"] = g["bytes_files"].get("pytorch_model.bin") == g["mirror_lfs_size"]
        except Exception as ex:  # noqa: BLE001
            g["mirror_parse_error"] = str(ex)[:200]
    g["pytorch_model.bin_sha256"] = A.file_sha256(os.path.join(gd, "pytorch_model.bin"))
    g["sha256_equals_builder_record"] = g["pytorch_model.bin_sha256"] == rec["files"]["pytorch_model.bin"]["lfs_sha256"]
    if g.get("mirror_lfs_sha256"):
        g["sha256_equals_mirror_listing"] = g["pytorch_model.bin_sha256"] == g["mirror_lfs_sha256"]
out["gpt_j"] = g
out["finished"] = A.now()
A.jdump(out, out_path)
A.say("gpt-j:", {k: v for k, v in g.items() if k != "bytes_files"})
