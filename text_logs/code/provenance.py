"""Provenance of the E10 build: public code, model, data, statistics, stack.  Writes out/provenance.json (CPU)."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

TPARTY = os.path.join(C.EXT, "third_party")
SAME_FILES = ["rome/rome_main.py", "rome/compute_u.py", "rome/compute_v.py", "rome/repr_tools.py", "rome/layer_stats.py",
              "rome/tok_dataset.py", "rome/rome_hparams.py", "util/generate.py", "util/nethook.py", "util/runningstats.py",
              "util/globals.py", "util/hparams.py", "hparams/ROME/gpt2-xl.json", "dsets/counterfact.py",
              "experiments/py/eval_utils_counterfact.py", "experiments/summarize.py", "globals.yml"]


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


def repo_info(name):
    p = os.path.join(TPARTY, name)
    tb = [f for f in os.listdir(os.path.join(TPARTY, "_tarballs")) if f.startswith(name + "-")]
    return {"url": f"https://github.com/kmeng01/{name}", "path": p, "commit": git(p, "rev-parse", "HEAD"),
            "commit_date": git(p, "log", "-1", "--format=%ad", "--date=iso"), "commit_subject": git(p, "log", "-1", "--format=%s"),
            "shallow_clone": git(p, "rev-parse", "--is-shallow-repository") == "true",
            "working_tree_clean": git(p, "status", "--porcelain") == "",
            "status_porcelain": git(p, "status", "--porcelain"),
            "commit_archive": {"file": os.path.join(TPARTY, "_tarballs", tb[0]), "sha256": C.sha256_file(os.path.join(TPARTY, "_tarballs", tb[0])),
                               "note": "archive of the same commit from codeload.github.com; `diff -r` against the clone: identical"} if tb else None}


def main():
    out = {"written": C.now()}
    out["rome_repo"] = repo_info("rome")
    out["memit_repo"] = repo_info("memit")
    cmp_ = {}
    for f in SAME_FILES:
        a, b = os.path.join(TPARTY, "rome", f), os.path.join(TPARTY, "memit", f)
        cmp_[f] = (C.sha256_file(a) == C.sha256_file(b)) if os.path.exists(a) and os.path.exists(b) else None
    out["rome_files_identical_in_memit_repo"] = cmp_
    out["hparams"] = {"file": C.HPARAMS_FILE, "sha256": C.sha256_file(C.HPARAMS_FILE), "content": C.jload(C.HPARAMS_FILE),
                      "tracked_file_unmodified": git(os.path.join(TPARTY, "rome"), "status", "--porcelain", "hparams/ROME/gpt2-xl.json") == ""}
    rec = C.jload(os.path.join(C.MODEL_DIR, "DOWNLOAD_RECORD.json"))
    st = os.path.join(C.MODEL_DIR, "model.safetensors")
    sha = C.sha256_file(st)
    out["model"] = {"repo": rec["repo"], "revision": rec["revision"], "endpoint": rec["endpoint"], "path": C.MODEL_DIR,
                    "files": rec["files"], "complete": rec["complete"], "model.safetensors_sha256": sha,
                    "model.safetensors_sha256_matches_hub_lfs": sha == rec["files"]["model.safetensors"]["lfs_sha256"],
                    "dtype": "float32", "config": C.jload(os.path.join(C.MODEL_DIR, "config.json"))}
    gj = os.path.join(C.EXT, "models", "gpt-j-6b", "DOWNLOAD_RECORD.json")
    if os.path.exists(gj):
        r = C.jload(gj)
        d = os.path.dirname(gj)
        out["gpt_j"] = {"repo": r["repo"], "revision": r["revision"], "path": d, "complete": r["complete"], "files": r["files"],
                        "bytes_on_disk": sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))),
                        "note": "downloaded for a later task; no GPU work was done with it"}
    else:
        out["gpt_j"] = {"complete": False, "note": "download not finished"}
    cf = os.path.realpath(C.COUNTERFACT)
    data = C.load_counterfact()
    out["counterfact"] = {"file": cf, "url": "https://rome.baulab.info/data/dsets/counterfact.json (the URL of the public loader dsets/counterfact.py)",
                          "sha256": C.sha256_file(cf), "bytes": os.path.getsize(cf), "n_records": len(data),
                          "keys": sorted(data[0].keys()), "case_ids_are_0_to_n_minus_1": [r["case_id"] for r in data] == list(range(len(data))),
                          "n_paraphrase_prompts": sorted({len(r["paraphrase_prompts"]) for r in data}),
                          "n_neighborhood_prompts": sorted({len(r["neighborhood_prompts"]) for r in data}),
                          "n_generation_prompts": sorted({len(r["generation_prompts"]) for r in data}),
                          "first_record": {"case_id": data[0]["case_id"], "requested_rewrite": data[0]["requested_rewrite"]},
                          "targets_with_leading_or_trailing_space": sum(
                              r["requested_rewrite"][k]["str"] != r["requested_rewrite"][k]["str"].strip() for r in data for k in ("target_new", "target_true"))}
    sf = os.path.realpath(os.path.join(C.DATA_DIR, "stats", "gpt2-xl", "wikipedia_stats", "transformer.h.17.mlp.c_proj_float32_mom2_100000.npz"))
    import numpy as np
    z = np.load(sf)
    out["stats"] = {"source": "authors' precomputed file (rome.baulab.info), not recomputed", "file": sf,
                    "url": "https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/transformer.h.17.mlp.c_proj_float32_mom2_100000.npz",
                    "sha256": C.sha256_file(sf), "bytes": os.path.getsize(sf), "mom2_shape": list(z["mom2.mom2"].shape),
                    "mom2_dtype": str(z["mom2.mom2"].dtype), "token_count": int(z["mom2.count"]), "sample_size": int(z["sample_size"]),
                    "constructor": str(z["mom2.constructor"])}
    aux = os.path.join(C.EXT, "data", "eval_aux")
    out["eval_aux"] = {f: {"bytes": os.path.getsize(os.path.join(aux, f))} for f in sorted(os.listdir(aux))}
    out["eval_aux"]["note"] = ("attribute snippets / tf-idf files serve the consistency and essence scores only, which are not "
                               "computed (attribute_snippets.json: 926285719 bytes on the server)")
    paper = os.path.join(C.EXT, "data", "paper_rome")
    out["paper"] = {"arxiv": "2202.05262v5", "e_print": os.path.join(paper, "2202.05262_eprint"),
                    "e_print_sha256": C.sha256_file(os.path.join(paper, "2202.05262_eprint")),
                    "table_file": os.path.join(paper, "src", "tables", "cf-results.tex")}
    C.setup_public()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_DIR)
    import collections
    cn = collections.Counter(len(tok(" " + r["requested_rewrite"]["target_new"]["str"])["input_ids"]) for r in data)
    ct = collections.Counter(len(tok(" " + r["requested_rewrite"]["target_true"]["str"])["input_ids"]) for r in data)
    out["counterfact"]["object_token_counts"] = {"target_new": {str(k): v for k, v in sorted(cn.items())},
                                                 "target_true": {str(k): v for k, v in sorted(ct.items())},
                                                 "note": "number of GPT-2 tokens of ' ' + object, over all records"}
    gs = os.path.join(C.LOGS, "setup", "gptj_sha256.txt")
    if os.path.exists(gs) and out["gpt_j"].get("complete"):
        got = open(gs).read().split()
        if got:
            out["gpt_j"]["pytorch_model.bin_sha256"] = got[0]
            out["gpt_j"]["pytorch_model.bin_sha256_matches_hub_lfs"] = got[0] == out["gpt_j"]["files"]["pytorch_model.bin"]["lfs_sha256"]
    out["versions"] = C.versions()
    out["python_env"] = sys.executable
    out["pydeps"] = {"nltk": C.PYDEPS, "transformers_stack": C.PYDEPS_TF4,
                     "public_env_pins": {"rome.yml": "python 3.9.7, pytorch 1.10.2, transformers 4.15.0, tokenizers 0.10.3, nltk 3.6.5",
                                         "memit.yml": "python 3.9.7, pytorch 1.12.1, transformers 4.23.1, nltk 3.7"}}
    nd = os.path.join(C.NLTK_DATA, "tokenizers")
    out["nltk_data"] = sorted(os.listdir(nd)) if os.path.isdir(nd) else None
    for src in ("dev", "stream"):
        mp = os.path.join(C.LOGS_EDIT, src, "meta.json")
        if os.path.exists(mp):
            m = C.jload(mp)
            out.setdefault("runs", {})[src] = {k: m.get(k) for k in ("started", "finished", "T_planned", "T_done", "seed", "resumed",
                                                                     "context_templates", "peak_alloc_MiB", "versions")}
    sm = os.path.join(C.OUT, "public_results", "ROME_single", "run_000", "run_meta.json")
    if os.path.exists(sm):
        m = C.jload(sm)
        out.setdefault("runs", {})["single"] = {k: m.get(k) for k in ("started", "finished", "seed", "resumed", "context_templates",
                                                                      "n_case_files", "peak_alloc_MiB", "versions")}
    C.jdump(out, os.path.join(C.OUT, "provenance.json"))
    print(json.dumps({k: out[k] for k in ("rome_repo", "memit_repo", "rome_files_identical_in_memit_repo")}, indent=1)[:3000])
    print("model sha matches:", out["model"]["model.safetensors_sha256_matches_hub_lfs"], "| gpt-j:", {k: v for k, v in out["gpt_j"].items() if k != "files"})
    print("counterfact:", {k: v for k, v in out["counterfact"].items() if k not in ("first_record", "keys")})
    print("stats:", out["stats"])


if __name__ == "__main__":
    main()
