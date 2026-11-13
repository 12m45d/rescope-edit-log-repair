"""Audit: further checks on stored files (CPU).  Output: out/extra2.json

 1. Direction of the updates from the statistics file: left_i = C^-1 k_i / ||C^-1 k_i|| with C = mom2 / count of the
    authors' file, computed by the audit in float64 on the CPU from the stored keys, against the stored left vectors
    (cosine).  With k = the template-averaged key (what the public compute_u uses) and, for contrast, with the key of the
    request prompt alone.  Ties the stored factors to the statistics file whose hash was checked.
 2. Prompt texts shared between the development log and the reported stream (records are disjoint; texts are not).
 3. DEVIATIONS.txt: the sentence on the use_cache check; items; what the list does not say.
 4. Driver versions: which stored files carry the field that the driver version on disk writes.
 5. The MEMIT clone: which ROME files are byte-identical there (argument for the transformers 4.23.1 stack).
 6. n-gram entropy: the audit's formula against the public formula on the same token lists (why differences are 0).
 7. Sizes: bytes and tensors of the five logs as stored.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
out = {"written": A.now()}
out_path = os.path.join(A.AOUT, "extra2.json")
data = A.load_counterfact()

# ---- 1. direction from the statistics file
z = np.load(A.STATS_FILE)
C = torch.from_numpy(z["mom2.mom2"].astype(np.float64)) / float(z["mom2.count"])
Cinv = torch.linalg.inv(C)
resid = float((Cinv @ C - torch.eye(C.shape[0], dtype=torch.float64)).abs().max())
rows = []
for src, idxs in (("dev", list(range(0, 100, 5))), ("stream", [0, 1, 50, 99, 100, 172, 173, 174, 240, 266, 500, 946, 999, 1000, 1416, 1530, 1531, 1947, 1999])):
    fac = A.load_factors(src)
    for i in idxs:
        left = fac["left"][i].double()
        for key_name in ("k_template_mean", "k_request"):
            u = Cinv @ fac[key_name][i].double()
            u = u / u.norm()
            rows.append({"source": src, "index": i, "key": key_name, "cosine_to_stored_left": float(u @ left / left.norm()),
                         "rel_l2_diff": float((u - left).norm() / left.norm())})
tm = [r for r in rows if r["key"] == "k_template_mean"]
rq = [r for r in rows if r["key"] == "k_request"]
out["left_vector_from_statistics"] = {
    "n_updates": len(tm), "inverse_residual_max_abs(float64)": resid,
    "with_template_mean_key": {"min_cosine": min(r["cosine_to_stored_left"] for r in tm), "median_cosine": float(np.median([r["cosine_to_stored_left"] for r in tm])),
                               "max_rel_l2_diff": max(r["rel_l2_diff"] for r in tm), "median_rel_l2_diff": float(np.median([r["rel_l2_diff"] for r in tm]))},
    "with_request_key(for contrast)": {"min_cosine": min(r["cosine_to_stored_left"] for r in rq), "median_cosine": float(np.median([r["cosine_to_stored_left"] for r in rq])),
                                       "max_cosine": max(r["cosine_to_stored_left"] for r in rq)},
    "first_token_subject_updates_with_request_key": {str(r["index"]): r["cosine_to_stored_left"] for r in rq if r["source"] == "stream" and r["index"] in (173, 240, 266, 946, 1416, 1947)},
    "note": "the public code inverts the float32 second moment on the GPU in float32 (torch.inverse(...).float()); the audit inverts it in float64 on "
            "the CPU: the difference between the two directions is the round-off of that float32 inversion (condition number about 6e4)"}
A.say("left vector from the statistics file:", out["left_vector_from_statistics"])
A.jdump(out, out_path)
del C, Cinv

# ---- 2. texts shared between the development log and the reported stream
P = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
by_text = collections.defaultdict(set)
for r in P:
    by_text[r["text"]].add((r["set"], r["kind"]))
dev_rows = [r for r in P if r["set"] == "dev"]
st_rows = [r for r in P if r["set"] == "stream"]
sh = {"n_dev_rows": len(dev_rows), "n_dev_distinct_texts": len({r["text"] for r in dev_rows})}
for kind in ("request", "paraphrase", "neighborhood"):
    rr = [r for r in dev_rows if r["kind"] == kind]
    sh[f"dev_{kind}_rows_whose_text_is_a_stream_prompt"] = {
        "n_rows": len(rr), "any_kind": int(sum(any(s == "stream" for s, _ in by_text[r["text"]]) for r in rr)),
        **{f"stream_{k2}": int(sum(("stream", k2) in by_text[r["text"]] for r in rr)) for k2 in ("request", "paraphrase", "neighborhood")}}
for T in (100, 500, 1000, 2000):
    rr = [r for r in st_rows if r["pos"] < T]
    sh[f"T{T}_rows_whose_text_is_a_dev_prompt"] = {"n_rows": len(rr), "n": int(sum(any(s == "dev" for s, _ in by_text[r["text"]]) for r in rr)),
                                                   "neighborhood": int(sum(r["kind"] == "neighborhood" and any(s == "dev" for s, _ in by_text[r["text"]]) for r in rr))}
sh["n_distinct_texts_in_both"] = int(sum(1 for t, o in by_text.items() if any(s == "dev" for s, _ in o) and any(s == "stream" for s, _ in o)))
dev_subj = {data[i]["requested_rewrite"]["subject"] for i in range(15000, 15100)}
sh["stream_requests_with_a_dev_subject"] = [i for i in range(2000) if data[i]["requested_rewrite"]["subject"] in dev_subj]
sh["relations"] = {"dev": len({data[i]["requested_rewrite"]["relation_id"] for i in range(15000, 15100)}),
                   "stream": len({data[i]["requested_rewrite"]["relation_id"] for i in range(2000)}),
                   "dataset": len({r["requested_rewrite"]["relation_id"] for r in data})}
# the same for the pools: benign prompt texts of HELD-OUT that are prompts of the development log
for pool in ("repair", "heldout"):
    rr = [r for r in P if r["set"] == pool]
    sh[f"{pool}_rows_whose_text_is_a_dev_prompt"] = int(sum(any(s == "dev" for s, _ in by_text[r["text"]]) for r in rr))
    sh[f"{pool}_rows_whose_text_is_a_stream_prompt"] = int(sum(any(s == "stream" for s, _ in by_text[r["text"]]) for r in rr))
out["dev_vs_stream_shared_prompt_texts"] = sh
A.say("dev vs stream shared texts:", sh)

# ---- 3. DEVIATIONS.txt
dv = re.sub(r"\s+", " ", open(os.path.join(A.B_OUT, "DEVIATIONS.txt")).read())
m = re.search(r"Equality of the public function's output[^.]*\.", dv)
items = re.findall(r"(?:^| )(\d+)\. ([A-Z][^:]{3,90}?) what:", dv)
out["deviations_file"] = {"n_items": len(items), "titles": [t for _, t in items], "use_cache_sentence": m.group(0) if m else None,
                          "use_cache_check_stated_as_not_run": bool(m and "not run yet" in m.group(0)),
                          "mentions": {k: (k in dv) for k in ("safetensors", "pytorch_model.bin", "overwritten", "earlier version", "version of the driver", "TF32", "deterministic")}}
infos = {}
for f in sorted(glob.glob(os.path.join(A.B_OUT, "describe", "*", "info.json"))):
    infos[os.path.basename(os.path.dirname(f))] = A.jload(f).get("use_cache_equivalence_at_W0")
out["deviations_file"]["use_cache_equivalence_recorded_in_describe_info"] = infos
A.say("deviations file:", out["deviations_file"])

# ---- 4. driver versions
dvv = {}
for src, first in (("dev", 15000), ("stream", 0)):
    n = len(glob.glob(os.path.join(A.B_LOGS_EDIT, src, "updates", "u_*.pt")))
    has = [("prob_test_use_cache_false" in A.jload(os.path.join(A.B_LOGS_EDIT, src, "write_time", f"case_{data[first + i]['case_id']}.json"))) for i in range(n)]
    dvv[src] = {"n_updates": n, "n_write_time_files_with_field_of_current_driver": int(sum(has)),
                "first_index_with_field": has.index(True) if any(has) else None, "contiguous_from_there": all(has[has.index(True):]) if any(has) else None}
out["driver_versions"] = {"field": "prob_test_use_cache_false (written by run_stream.py as it is on disk)", **dvv,
                          "reading": "write-time files without the field were written by an earlier version of the driver (before the memory fallback was added)"}
A.say("driver versions:", out["driver_versions"])

# ---- 5. MEMIT clone
memit = os.path.join(A.EXT, "third_party", "memit")
same = {}
for f in ("rome/compute_u.py", "rome/compute_v.py", "rome/rome_hparams.py", "rome/tok_dataset.py", "util/generate.py", "util/nethook.py", "util/runningstats.py",
          "hparams/ROME/gpt2-xl.json", "rome/rome_main.py", "rome/repr_tools.py", "rome/layer_stats.py", "experiments/py/eval_utils_counterfact.py", "experiments/summarize.py"):
    a, b = os.path.join(A.ROME_REPO, f), os.path.join(memit, f)
    same[f] = (A.file_sha256(a) == A.file_sha256(b)) if os.path.exists(a) and os.path.exists(b) else None
pins = {}
for name, p in (("rome", os.path.join(A.ROME_REPO, "scripts", "rome.yml")), ("memit", os.path.join(memit, "scripts", "memit.yml"))):
    if os.path.exists(p):
        t = open(p).read()
        pins[name] = sorted(set(re.findall(r"(?:transformers|pytorch|python|nltk|tokenizers)=+[0-9][0-9.]*", t)))
out["memit_clone"] = {"rome_files_byte_identical_in_memit": same, "environment_pins": pins}
A.say("memit clone:", out["memit_clone"])

# ---- 6. n-gram entropy formula
sys.path.append(A.PYDEPS)
import nltk  # noqa: E402

g = A.jload(os.path.join(A.B_OUT, "describe", "T1000", "gen_stream_first100.json"))
txt = g["rows"][3]["text"][2]
toks = nltk.word_tokenize(txt)


def ent(n_, log2):
    grams = [tuple(toks[i:i + n_]) for i in range(len(toks) - n_ + 1)]
    c = np.array(list(collections.Counter(grams).values()), dtype=np.float64)
    p = c / c.sum()
    return float(np.sum(-p * np.log2(p))) if log2 else float(np.sum(-p * np.log(p) / np.log(2)))


out["ngram_entropy_formula"] = {"text": txt[:160], "n_word_tokens": len(toks), "n_distinct_word_tokens": len(set(toks)),
                                "bigram_bits": [ent(2, True), ent(2, False)], "trigram_bits": [ent(3, True), ent(3, False)],
                                "audit(log2)": float(np.mean([2 / 3 * ent(2, True), 4 / 3 * ent(3, True)])),
                                "public_formula(log/log 2)": float(np.mean([2 / 3 * ent(2, False), 4 / 3 * ent(3, False)])),
                                "stored": g["rows"][3]["ngram_entropy_per_text"][2],
                                "upper_bound_for_a_uniformly_random_sequence_over_5_symbols": float(np.mean([2 / 3 * np.log2(25), 4 / 3 * np.log2(125)]))}
A.say("n-gram entropy:", out["ngram_entropy_formula"])

# ---- 7. sizes
sz = {}
for name, (src, T, first) in A.LOGS.items():
    d = os.path.join(A.B_LOGS_EDIT, "library_format", name)
    sz[name] = {"T": T, "library_format_bytes": sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)),
                "deployed_weight_file_bytes": os.path.getsize(os.path.join(A.B_LOGS_EDIT, src, "snapshots", f"W_T{T}.pt"))}
out["sizes"] = sz
out["finished"] = A.now()
A.jdump(out, out_path)
