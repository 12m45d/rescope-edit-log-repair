"""Audit, condition (2), CPU part.

 a. Record by record: the audit's re-run of the single-edit protocol on 150 random records of the first 1000 (public
    experiments/evaluate.py main(), seed 0; gpu_single.py) against the builder's stored case files and factors.
 b. The three aggregate scores recomputed from the builder's 1000 per-record files with the audit's own arithmetic
    (definitions read from the public experiments/summarize.py, re-implemented here; the public summariser is not called).
 c. The published numbers, read by the audit from the arXiv source it downloaded itself (tmp/dl/paper/eprint_v5).
 d. Sensitivity: the same 150 records with other context templates (seed 1), and the four first-token-subject records.
Output: out/single_cmp.json
"""
import glob
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

B_RUN = os.path.join(A.B_OUT, "public_results", "ROME_single", "run_000")
KEYS = ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs", "attribute_prompts_probs")
out = {"written": A.now()}
out_path = os.path.join(A.AOUT, "single_cmp.json")


def flat(d):
    return np.array([x[k] for key in KEYS for x in d[key] for k in ("target_new", "target_true")], dtype=np.float64)


def scores(d):
    """Per-record public scores (the audit's arithmetic).  Stored values are NEGATIVE log-probabilities:
    efficacy / paraphrase success: P(new) > P(true)  <=>  nll_true > nll_new;  neighbourhood success: P(true) > P(new)."""
    r = {}
    for key, short in (("rewrite_prompts_probs", "ES"), ("paraphrase_prompts_probs", "PS")):
        r[short] = float(np.mean([x["target_true"] > x["target_new"] for x in d[key]]))
        r[short[0] + "M"] = float(np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in d[key]]))
    x_ = d["neighborhood_prompts_probs"]
    r["NS"] = float(np.mean([x["target_true"] < x["target_new"] for x in x_]))
    r["NM"] = float(np.mean([np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in x_]))
    return r


def agg(rows):
    n = len(rows)
    res = {"n": n}
    for k in ("ES", "PS", "NS", "EM", "PM", "NM"):
        v = np.array([r[k] for r in rows]) * 100
        res[k] = float(v.mean())
        res[k + "_se"] = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    res["S_harmonic_mean_of_ES_PS_NS"] = float(3 / (1 / res["ES"] + 1 / res["PS"] + 1 / res["NS"])) if min(res["ES"], res["PS"], res["NS"]) > 0 else 0.0
    return res


# ----------------------------------------------------------------------------- c. published numbers (audit's own download)
ep = os.path.join(A.ATMP, "dl", "paper", "eprint_v5")
pub = {"source": "https://arxiv.org/e-print/2202.05262v5 downloaded by the audit", "eprint_sha256": A.file_sha256(ep),
       "builder_eprint_sha256": A.file_sha256(os.path.join(A.EXT, "data", "paper_rome", "2202.05262_eprint"))}
pub["eprint_equal_to_builder_copy"] = pub["eprint_sha256"] == pub["builder_eprint_sha256"]
absf = os.path.join(A.ATMP, "dl", "paper", "abs.html")
if os.path.exists(absf):
    h = open(absf, errors="replace").read()
    pub["arxiv_abs_page_versions_mentioned"] = sorted(set(re.findall(r"arXiv:2202\.05262(v\d+)", h)))
    m = re.search(r"last revised ([^<(]*)\(this version, (v\d+)\)", h)
    pub["arxiv_abs_latest"] = [m.group(1).strip(), m.group(2)] if m else None
with tarfile.open(ep, "r:gz") as tf:
    names = [m.name for m in tf.getmembers() if m.isfile()]
    tex = tf.extractfile("tables/cf-results.tex").read().decode("utf8")          # read only
    exp = tf.extractfile("text/experiments.tex").read().decode("utf8")
    mini = tf.extractfile("tables/cf-results-mini.tex").read().decode("utf8") if "tables/cf-results-mini.tex" in names else ""
block = tex.split("\\midrule\\midrule")[0]                                       # the GPT-2 XL block comes first
COLS = ["S", "ES", "EM", "PS", "PM", "NS", "NM", "GE", "RS"]
rows = {}
for line in block.split("\\\\"):
    line = line.replace("\\midrule", "").strip()
    cells = [c.strip() for c in line.split("&")]
    if len(cells) != 10:
        continue
    name = cells[0].split("\n")[-1].strip()
    vals = {}
    for col, cell in zip(COLS, cells[1:]):
        cell = re.sub(r"\\(?:goodmetric|badmetric)\{([^}]*)\}", r"\1", cell).strip()
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:\s*\((\d+(?:\.\d+)?)\))?", cell)
        if not m:
            vals = None
            break
        vals[col] = [float(m.group(1)), float(m.group(2)) if m.group(2) else None]
    if vals:
        rows[name] = vals
pub["table_caption_has"] = "Quantitative Editing Results" in tex
pub["gpt2xl_block_rows"] = list(rows)
pub["ROME"] = rows.get("ROME")
pub["GPT-2 XL (unedited)"] = rows.get("GPT-2 XL")
m = re.search(r"over ([\d,]+) and ([\d,]+)-record test sets", exp)
pub["n_records_gpt2xl"] = int(m.group(1).replace(",", "")) if m else None
pub["sentence_on_test_sets"] = m.group(0) if m else None
out["published"] = pub
P = {k: pub["ROME"][k][0] for k in ("ES", "PS", "NS")}
A.say("published ROME row (GPT-2 XL):", pub["ROME"], "| n records:", pub["n_records_gpt2xl"], "| e-print = builder's copy:", pub["eprint_equal_to_builder_copy"])

# ----------------------------------------------------------------------------- b. aggregates from the builder's 1000 files
files = sorted(glob.glob(os.path.join(B_RUN, "case_*.json")), key=lambda p: int(os.path.basename(p)[5:-5]))
ids = [int(os.path.basename(p)[5:-5]) for p in files]
data = A.load_counterfact()
B = {}
bad_req = 0
for p in files:
    d = json.load(open(p))
    B[d["case_id"]] = d
    bad_req += d["requested_rewrite"] != data[d["case_id"]]["requested_rewrite"]
post = [scores(B[i]["post"]) for i in range(1000)]
pre = [scores(B[i]["pre"]) for i in range(1000)]
nonfinite = int(sum(not np.isfinite(flat(B[i]["post"])).all() or not np.isfinite(flat(B[i]["pre"])).all() for i in range(1000)))
ag_post, ag_pre = agg(post), agg(pre)
b_json = A.jload(os.path.join(A.B_OUT, "single_edit_check.json"))
out["builder_1000"] = {
    "n_case_files": len(files), "case_ids_are_0_999": ids == list(range(1000)), "n_requests_not_equal_dataset": int(bad_req),
    "n_records_with_nonfinite_probability": nonfinite,
    "n_prompts_per_record": sorted({(len(B[i]["post"]["rewrite_prompts_probs"]), len(B[i]["post"]["paraphrase_prompts_probs"]),
                                     len(B[i]["post"]["neighborhood_prompts_probs"])) for i in range(1000)}),
    "edited_recomputed_by_audit": ag_post, "unedited_recomputed_by_audit": ag_pre,
    "builder_reported": {k: b_json["comparison"]["post"][k]["ours"] for k in ("ES", "PS", "NS", "EM", "PM", "NM", "S")},
    "builder_reported_unedited": {k: b_json["comparison"]["pre"][k]["ours"] for k in ("ES", "PS", "NS")},
    "diff_to_published": {k: ag_post[k] - P[k] for k in ("ES", "PS", "NS")},
    "within_3_points": {k: bool(abs(ag_post[k] - P[k]) <= 3.0) for k in ("ES", "PS", "NS")},
    "diff_unedited_to_published": {k: ag_pre[k] - pub["GPT-2 XL (unedited)"][k][0] for k in ("ES", "PS", "NS")},
    # how the score reacts to the definition of the aggregate (per-record mean of per-prompt flags vs pooled prompts)
    "pooled_over_prompts": {"PS": float(100 * np.mean([x["target_true"] > x["target_new"] for i in range(1000) for x in B[i]["post"]["paraphrase_prompts_probs"]])),
                            "NS": float(100 * np.mean([x["target_true"] < x["target_new"] for i in range(1000) for x in B[i]["post"]["neighborhood_prompts_probs"]]))},
    "ties": int(sum(x["target_true"] == x["target_new"] for i in range(1000) for k in KEYS[:3] for x in B[i]["post"][k])),
}
out["builder_1000"]["audit_equals_builder_reported_to_0.01"] = {k: bool(abs(ag_post[k] - out["builder_1000"]["builder_reported"][k]) < 0.006)
                                                                for k in ("ES", "PS", "NS", "EM", "PM", "NM")}
# sub-blocks of the 1000 records, to see how much a 3-point tolerance has to absorb
out["builder_1000"]["by_block_of_200"] = [{"records": [a, a + 199], **{k: agg(post[a:a + 200])[k] for k in ("ES", "PS", "NS")}} for a in range(0, 1000, 200)]
ft = [173, 240, 266, 946]
out["builder_1000"]["first_token_subject_records"] = {str(i): {**{k: post[i][k] for k in ("ES", "PS", "NS")}} for i in ft}
out["builder_1000"]["without_first_token_subject_records"] = {k: agg([post[i] for i in range(1000) if i not in ft])[k] for k in ("ES", "PS", "NS")}
A.say("builder's 1000 records, recomputed:", {k: round(ag_post[k], 3) for k in ("ES", "PS", "NS")}, "published:", P,
      "within 3:", out["builder_1000"]["within_3_points"])
A.jdump(out, out_path)


# ----------------------------------------------------------------------------- a. record by record (seed 0) and d. seed 1
def compare(tag):
    meta_p = os.path.join(A.AOUT, f"single_{tag}.json")
    if not os.path.exists(meta_p):
        return None
    meta = A.jload(meta_p)
    if "finished" not in meta:
        return {"finished": False}
    run = os.path.join(A.AOUT, "public_results", f"ROME_audit_{tag}", "run_000")
    b_meta = A.jload(os.path.join(B_RUN, "run_meta.json"))
    res = {"seed": meta["seed"], "n_sample": len(meta["sample"]), "extra_records": meta["extra_records"],
           "public_driver": meta["public_driver"], "baselines_replaced_by_placeholders": meta["baselines_replaced_by_placeholders"],
           "context_templates_equal_builder": meta["context_templates"] == b_meta["context_templates"],
           "n_templates_equal_positionwise": int(sum(a == b for a, b in zip(meta["context_templates"], b_meta["context_templates"]))),
           "context_templates": meta["context_templates"], "weight_restored_bitwise": meta["weight_restored_bitwise"]}
    per = []
    for i in sorted(set(meta["sample"]) | set(meta["extra_records"])):
        a = json.load(open(os.path.join(run, f"case_{i}.json")))
        b = B[i]
        fa = torch.load(os.path.join(A.AOUT, f"single_{tag}", "factors", f"case_{i}.pt"), map_location="cpu")
        fb = torch.load(os.path.join(B_RUN, "cases_factors", f"case_{i}.pt"), map_location="cpu")["tensors"]
        sa, sb = scores(a["post"]), scores(b["post"])
        fl_a = [x["target_true"] > x["target_new"] for k in KEYS[:3] for x in a["post"][k]]
        fl_b = [x["target_true"] > x["target_new"] for k in KEYS[:3] for x in b["post"][k]]
        per.append({"record": i, "in_sample": i in meta["sample"],
                    "post_probabilities_bitwise_equal": bool(np.array_equal(flat(a["post"]), flat(b["post"]))),
                    "pre_probabilities_bitwise_equal": bool(np.array_equal(flat(a["pre"]), flat(b["pre"]))),
                    "post_max_abs_diff_nll": float(np.abs(flat(a["post"]) - flat(b["post"])).max()),
                    "left_bitwise_equal": bool(torch.equal(fa["left"], fb["left"])), "right_bitwise_equal": bool(torch.equal(fa["right"], fb["right"])),
                    "left_rel_l2_diff": float((fa["left"].double() - fb["left"].double()).norm() / fb["left"].double().norm()),
                    "right_rel_l2_diff": float((fa["right"].double() - fb["right"].double()).norm() / fb["right"].double().norm()),
                    "update_fro_audit": float(fa["left"].double().norm() * fa["right"].double().norm()),
                    "update_fro_builder": float(fb["left"].double().norm() * fb["right"].double().norm()),
                    "n_prompt_flags": len(fl_a), "n_prompt_flags_differ": int(sum(x != y for x, y in zip(fl_a, fl_b))),
                    "audit": {k: sa[k] for k in ("ES", "PS", "NS")}, "builder": {k: sb[k] for k in ("ES", "PS", "NS")},
                    "scores_equal": all(sa[k] == sb[k] for k in ("ES", "PS", "NS", "EM", "PM", "NM")),
                    "requested_rewrite_equal": a["requested_rewrite"] == b["requested_rewrite"]})
    S = [p for p in per if p["in_sample"]]
    a_rows = [scores(json.load(open(os.path.join(run, f"case_{p['record']}.json")))["post"]) for p in S]
    b_rows = [post[p["record"]] for p in S]
    res.update({
        "n_records_compared": len(S),
        "n_post_probabilities_bitwise_equal": int(sum(p["post_probabilities_bitwise_equal"] for p in S)),
        "n_pre_probabilities_bitwise_equal": int(sum(p["pre_probabilities_bitwise_equal"] for p in S)),
        "max_abs_diff_nll": float(max(p["post_max_abs_diff_nll"] for p in S)),
        "n_left_bitwise_equal": int(sum(p["left_bitwise_equal"] for p in S)), "n_right_bitwise_equal": int(sum(p["right_bitwise_equal"] for p in S)),
        "max_left_rel_l2_diff": float(max(p["left_rel_l2_diff"] for p in S)), "max_right_rel_l2_diff": float(max(p["right_rel_l2_diff"] for p in S)),
        "median_right_rel_l2_diff": float(np.median([p["right_rel_l2_diff"] for p in S])),
        "n_records_all_scores_equal": int(sum(p["scores_equal"] for p in S)),
        "n_prompt_flags": int(sum(p["n_prompt_flags"] for p in S)), "n_prompt_flags_differ": int(sum(p["n_prompt_flags_differ"] for p in S)),
        "n_records_with_a_flag_differing": int(sum(p["n_prompt_flags_differ"] > 0 for p in S)),
        "audit_sample_scores": agg(a_rows), "builder_same_records_scores": agg(b_rows),
        "diff_audit_minus_builder": {k: agg(a_rows)[k] - agg(b_rows)[k] for k in ("ES", "PS", "NS")},
        "extra_first_token_records": [p for p in per if not p["in_sample"]],
        "sample_records_with_update_fro_above_117": [(p["record"], round(p["update_fro_audit"], 1)) for p in S if p["update_fro_audit"] > 117.13],
        "differing_records": [p for p in S if not p["scores_equal"]][:20],
    })
    return res


for tag in ("seed0", "seed1"):
    r = compare(tag)
    if r is not None:
        out[f"rerun_{tag}"] = r
        A.say(tag, {k: v for k, v in r.items() if k not in ("context_templates", "extra_first_token_records", "differing_records", "public_driver",
                                                            "baselines_replaced_by_placeholders", "extra_records")})
out["condition_2"] = {"published": P, "builder_first_1000_recomputed": {k: ag_post[k] for k in ("ES", "PS", "NS")},
                      "abs_diff": {k: abs(ag_post[k] - P[k]) for k in ("ES", "PS", "NS")}, "tolerance": 3.0,
                      "pass": bool(all(abs(ag_post[k] - P[k]) <= 3.0 for k in ("ES", "PS", "NS")))}
out["finished"] = A.now()
A.jdump(out, out_path)
A.say("condition 2:", out["condition_2"])
