"""Audit checks 4 and 7, CPU part: data traps and the builder's table, recomputed from the stored row / case files with
the audit's own arithmetic (the GPU part, gpu_eval.py, checks samples of those rows against the audit's own decode).

 1. Tokenisation of the objects: number of GPT-2 tokens of ' ' + object over all 21,919 records (new and true object);
    leading / trailing spaces in object strings; what the rule would do WITHOUT the leading space; prompts that end in
    white space (the object would then be tokenised without a leading space); stored token ids of the 27,878 prompt texts
    vs the tokenizer of the project environment.
 2. Pre-edit continuations (all 27,878): degenerate ones by the audit's own definitions; EOS; validity; duplicates.
 3. The table of the deployed logs from the builder's per-row files: exact-match columns, D1, tokens kept, retention;
    teacher-forced vs decode D1 on all deployed rows; first-token rule vs the stored 10-token decode (the audit applies
    the exact-match rule to the stored decode itself, and a TEXT rule next to the token rule).
 4. Public scores of the deployed / pre-edit model from the stored public-format case files (audit's arithmetic).
 5. Generation entropy: the audit's own n-gram entropy of the stored generations (definition re-implemented from the
    public eval_utils_counterfact; nltk.word_tokenize), per log; what the collapsed models generate.
 6. Ground truth vs evaluation: which stored quantity defines y0 and which model is scored against it.
Output: out/traps.json
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402   (project environment's transformers; tokenizer only)

import transformers  # noqa: E402

tok = AutoTokenizer.from_pretrained(A.MODEL_DIR)
data = A.load_counterfact()
out = {"written": A.now(), "tokenizer_stack": f"transformers {transformers.__version__} (project environment)"}
out_path = os.path.join(A.AOUT, "traps.json")
NAMES = {"dev100": "TX-G2XL-rome-dev100", "T100": "TX-G2XL-rome-T100", "T500": "TX-G2XL-rome-T500", "T1000": "TX-G2XL-rome-T1000",
         "T2000": "TX-G2XL-rome-T2000"}
REPORT = {"dev100": {"ES": 100.0, "PS": 98.0, "NS": 72.9, "req": 98.0, "para": 75.5, "D1": 95.4, "tok": 64.7, "GE": 587.2},
          "T100": {"ES": 100.0, "PS": 98.5, "NS": 66.8, "req": 97.0, "para": 75.0, "D1": 93.5, "tok": 69.8, "GE": 594.2},
          "T500": {"ES": 57.0, "PS": 52.2, "NS": 52.1, "req": 2.8, "para": 0.7, "D1": 100.0, "tok": 0.6, "GE": 578.6},
          "T1000": {"ES": 53.4, "PS": 52.4, "NS": 52.0, "req": 1.4, "para": 0.9, "D1": 100.0, "tok": 0.3, "GE": 587.3},
          "T2000": {"ES": 50.4, "PS": 47.3, "NS": 51.9, "req": 0.9, "para": 0.4, "D1": 100.0, "tok": 0.3, "GE": 503.1}}   # the builder's report

# ----------------------------------------------------------------------------- 1. tokenisation
new = [r["requested_rewrite"]["target_new"]["str"] for r in data]
true = [r["requested_rewrite"]["target_true"]["str"] for r in data]
ids_new = tok([" " + s for s in new])["input_ids"]
ids_true = tok([" " + s for s in true])["input_ids"]
ids_new_nosp = tok(new)["input_ids"]
t1 = {"n_records": len(data),
      "n_tokens_of_space_plus_new_object": dict(sorted(collections.Counter(len(x) for x in ids_new).items())),
      "n_tokens_of_space_plus_true_object": dict(sorted(collections.Counter(len(x) for x in ids_true).items())),
      "n_tokens_of_new_object_WITHOUT_leading_space": dict(sorted(collections.Counter(len(x) for x in ids_new_nosp).items())),
      "n_records_where_token_without_space_equals_token_with_space": int(sum(a == b for a, b in zip(ids_new, ids_new_nosp))),
      "objects_with_leading_or_trailing_whitespace": int(sum(s != s.strip() for s in new + true)),
      "first_token_of_space_plus_object_starts_with_space": int(sum(tok.decode(x[:1]).startswith(" ") for x in ids_new)),
      "new_object_equals_true_object": int(sum(a == b for a, b in zip(new, true))),
      "distinct_new_objects": len(set(new))}
# first-token collisions: another object whose first token is the same (a first-token test cannot tell them apart) -- none if all are 1 token
t1["max_tokens_any_object"] = int(max(max(len(x) for x in ids_new), max(len(x) for x in ids_true)))
PT = A.jload(os.path.join(A.B_OUT, "prompt_texts.json"))
TEXTS, IDS = PT["texts"], PT["input_ids"]
mine = tok(TEXTS)["input_ids"]
t1["prompt_texts"] = {"n": len(TEXTS), "n_stored_token_ids_equal_environment_tokenizer": int(sum(a == b for a, b in zip(mine, IDS))),
                      "n_ending_in_whitespace": int(sum(t != t.rstrip() for t in TEXTS)), "n_starting_with_whitespace": int(sum(t != t.lstrip() for t in TEXTS)),
                      "n_empty": int(sum(len(x) == 0 for x in IDS)), "n_with_bos_token_first": int(sum(x[0] == tok.bos_token_id for x in IDS if x)),
                      "token_length_min_max": [int(min(map(len, IDS))), int(max(map(len, IDS)))]}
rows = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
t1["request_prompts_whose_template_ends_with_the_subject"] = int(sum(data[r["record"]]["requested_rewrite"]["prompt"].rstrip().endswith("{}")
                                                                    for r in rows if r["kind"] == "request"))
out["tokenisation"] = t1
A.say("tokenisation:", t1)

# ----------------------------------------------------------------------------- 2. pre-edit continuations
PRE = torch.load(os.path.join(A.B_OUT, "pre_continuations.pt"), map_location="cpu")
Y0 = PRE["y0"].tolist()
eos = tok.eos_token_id


def degen(c):
    text = tok.decode(c)
    run = best = 1
    for a, b in zip(c, c[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return {"empty": text.strip() == "", "newline_first": text[:1] == "\n", "eos_first": c[0] == eos, "has_eos": eos in c,
            "le3_distinct": len(set(c)) <= 3, "le5_distinct": len(set(c)) <= 5, "run_ge4": best >= 4, "run_ge3": best >= 3,
            "period2": all(c[i] == c[i % 2] for i in range(len(c))), "le2_words": len(text.split()) <= 2,
            "no_letter": not any(ch.isalpha() for ch in text)}


dg = [degen(c) for c in Y0]
t2 = {"n_texts": len(Y0), "n_tokens_per_continuation": sorted({len(c) for c in Y0}), "n_valid": int(PRE["valid"].sum()), "n_finite": int(PRE["finite"].sum()),
      "n_y0_equals_pre_tf": int((PRE["y0"] == PRE["pre_tf"]).all(1).sum()),
      "counts": {k: int(sum(d[k] for d in dg)) for k in dg[0]},
      "builder_summary": {k: v for k, v in A.jload(os.path.join(A.B_OUT, "pre_continuations_summary.json")).items()
                          if k in ("n_texts", "n_valid", "n_invalid", "n_nonfinite", "n_empty", "n_repetitive", "n_few_distinct", "n_long_run", "n_has_eos")}}
t2["n_repetitive_builder_definition(le3 distinct or run>=4)"] = int(sum(d["le3_distinct"] or d["run_ge4"] for d in dg))
ho_t = sorted({r["text_id"] for r in rows if r["set"] == "heldout"})
t2["heldout"] = {"n_texts": len(ho_t), "has_eos": int(sum(dg[t]["has_eos"] for t in ho_t)), "newline_first": int(sum(dg[t]["newline_first"] for t in ho_t)),
                 "repetitive": int(sum(dg[t]["le3_distinct"] or dg[t]["run_ge4"] for t in ho_t)), "le5_distinct": int(sum(dg[t]["le5_distinct"] for t in ho_t))}
cnt = collections.Counter(tok.decode(c) for c in Y0)
t2["most_common_continuations"] = [[k, v] for k, v in cnt.most_common(5)]
t2["n_distinct_continuations"] = len(cnt)
first = collections.Counter(c[0] for c in Y0)
t2["most_common_first_tokens"] = [[tok.decode([k]), v] for k, v in first.most_common(6)]
out["pre_edit_continuations"] = t2
A.say("pre-edit continuations:", t2)
A.jdump(out, out_path)

# ----------------------------------------------------------------------------- 3. the table from the per-row files


def pct(x):
    x = list(x)
    return float(100 * np.mean(x)) if x else None


tab, n_dis, n_rows_dep = {}, 0, 0
for tag, name in NAMES.items():
    src, T, first_rec = A.LOGS[name]
    R = {}
    for label in ("deployed", "pre_edit"):
        rw = A.jload(os.path.join(A.B_OUT, "text_logs_rows", f"{name}__{label}.json"))
        dec = torch.load(os.path.join(A.B_OUT, "describe", tag if label == "deployed" else "W0", "dec.pt"), map_location="cpu")
        pos = {int(t): i for i, t in enumerate(dec["text_ids"].tolist())}
        e, h, nb = rw["edits"], rw["heldout"], rw["neighborhood"]
        hv, nv = [x for x in h if x["valid"]], [x for x in nb if x["valid"]]
        # the exact-match rule applied by the audit to the stored 10-token decode: token rule and text rule
        prow = {r["prompt_id"]: r for r in rows if r["set"] == src and r["kind"] in ("request", "paraphrase") and r["pos"] < T}
        tok_rule, text_rule, word_rule, kinds = [], [], [], []
        for x in e:
            r = prow[x["prompt_id"]]
            tgt_s = data[r["record"]]["requested_rewrite"]["target_new"]["str"]
            tgt = tok(" " + tgt_s)["input_ids"]
            cont = dec["dec"][pos[r["text_id"]]].tolist()
            txt = tok.decode(cont)
            tok_rule.append(cont[:len(tgt)] == tgt)
            text_rule.append(txt.lstrip().startswith(tgt_s))
            rest = txt.lstrip()[len(tgt_s):]
            word_rule.append(txt.lstrip().startswith(tgt_s) and (rest == "" or not (rest[0].isalnum())))
            kinds.append(x["kind"])
        M = {"n_request": int(sum(x["kind"] == "request" for x in e)), "n_paraphrase": int(sum(x["kind"] == "paraphrase" for x in e)),
             "request_exact": pct(x["new_dec_exact"] for x in e if x["kind"] == "request"),
             "paraphrase_exact": pct(x["new_dec_exact"] for x in e if x["kind"] == "paraphrase"),
             "request_exact_tf": pct(x["new_tf_exact"] for x in e if x["kind"] == "request"),
             "request_token_level": pct(x["new_dec_tok"] for x in e if x["kind"] == "request"),
             "n_rows_dec_exact_differs_from_tf_exact": int(sum(x["new_dec_exact"] != x["new_tf_exact"] for x in e)),
             "n_rows_exact_differs_from_token_level": int(sum(float(x["new_dec_exact"]) != x["new_dec_tok"] for x in e)),
             "audit_token_rule_on_stored_decode": {"request": pct(a for a, k in zip(tok_rule, kinds) if k == "request"),
                                                   "paraphrase": pct(a for a, k in zip(tok_rule, kinds) if k == "paraphrase"),
                                                   "n_rows_differ_from_builder_flag": int(sum(a != x["new_dec_exact"] for a, x in zip(tok_rule, e)))},
             "audit_text_rule_on_stored_decode(starts with the object string)": {
                 "request": pct(a for a, k in zip(text_rule, kinds) if k == "request"), "paraphrase": pct(a for a, k in zip(text_rule, kinds) if k == "paraphrase"),
                 "n_rows_differ_from_token_rule": int(sum(a != b for a, b in zip(text_rule, tok_rule)))},
             "audit_word_rule_on_stored_decode(object string followed by a non-alphanumeric character)": {
                 "request": pct(a for a, k in zip(word_rule, kinds) if k == "request"), "paraphrase": pct(a for a, k in zip(word_rule, kinds) if k == "paraphrase"),
                 "n_rows_differ_from_token_rule": int(sum(a != b for a, b in zip(word_rule, tok_rule)))},
             "request_true_object_exact": pct(x["true_dec_exact"] for x in e if x["kind"] == "request"),
             "heldout": {"n": len(h), "n_valid": len(hv), "D1_tf": pct(x["D1_tf"] for x in hv), "D1_dec": pct(x["D1_dec"] for x in hv),
                         "tok_loc": pct(x["tok_loc"] for x in hv), "n_tf_differs_from_dec": int(sum(x["D1_tf"] != x["D1_dec"] for x in hv)),
                         "repetitive": pct(x["repetitive"] for x in h), "n_nonfinite": int(sum(not x["finite"] for x in h)),
                         "D1_tf_clean": pct(x["D1_tf"] for x in hv if x["clean"]), "n_clean": int(sum(x["clean"] for x in hv)),
                         "D1_tf_near": pct(x["D1_tf"] for x in hv if not x["clean"]), "n_near": int(sum(not x["clean"] for x in hv))},
             "neighborhood": {"n": len(nb), "n_valid": len(nv), "D1_tf": pct(x["D1_tf"] for x in nv), "D1_dec": pct(x["D1_dec"] for x in nv),
                              "tok_loc": pct(x["tok_loc"] for x in nv), "n_tf_differs_from_dec": int(sum(x["D1_tf"] != x["D1_dec"] for x in nv))}}
        # D1 recomputed by the audit from the stored tensors (dec.pt vs pre_continuations.pt), not from the builder's flags
        y = PRE["y0"]
        d1_dec = [bool((dec["dec"][pos[x["text_id"]]] != y[x["text_id"]]).any()) for x in hv]
        d1_tf = [bool((dec["tf_pred"][pos[x["text_id"]]] != y[x["text_id"]]).any()) for x in hv]
        tl = [float((dec["tf_pred"][pos[x["text_id"]]] == y[x["text_id"]]).float().mean()) for x in hv]
        M["heldout"]["audit_from_tensors"] = {"D1_dec": pct(d1_dec), "D1_tf": pct(d1_tf), "tok_loc": pct(tl),
                                              "first_token_changed": pct(bool(dec["tf_pred"][pos[x["text_id"]]][0] != y[x["text_id"]][0]) for x in hv),
                                              "n_rows_differ_from_builder_flags": int(sum(a != x["D1_tf"] or b != x["D1_dec"] for a, b, x in zip(d1_tf, d1_dec, hv)))}
        if src == "stream":
            M["first100"] = {"request_exact": pct(x["new_dec_exact"] for x in e if x["kind"] == "request" and x["pos"] < 100),
                             "paraphrase_exact": pct(x["new_dec_exact"] for x in e if x["kind"] == "paraphrase" and x["pos"] < 100)}
        if label == "deployed":
            n_dis += M["heldout"]["n_tf_differs_from_dec"] + M["neighborhood"]["n_tf_differs_from_dec"]
            n_rows_dep += len(hv) + len(nv)
        R[label] = M
    tab[tag] = R
    rep = REPORT[tag]
    R["vs_report"] = {"req": [R["deployed"]["request_exact"], rep["req"]], "para": [R["deployed"]["paraphrase_exact"], rep["para"]],
                      "D1": [R["deployed"]["heldout"]["D1_tf"], rep["D1"]], "tok": [R["deployed"]["heldout"]["tok_loc"], rep["tok"]]}
    R["vs_report"]["all_equal_to_reported_digits"] = all(abs(a - b) < 0.05 for a, b in R["vs_report"].values())
    A.say(tag, "deployed:", {k: v for k, v in R["deployed"].items() if k in ("request_exact", "paraphrase_exact", "first100")},
          {k: round(v, 2) if isinstance(v, float) else v for k, v in R["deployed"]["heldout"].items() if k != "audit_from_tensors"},
          "| pre-edit D1:", R["pre_edit"]["heldout"]["D1_tf"], "req:", R["pre_edit"]["request_exact"], "| report ok:", R["vs_report"]["all_equal_to_reported_digits"])
out["table_from_row_files"] = tab
out["D1_tf_vs_dec_on_deployed_rows"] = {"n_rows": n_rows_dep, "n_differ": n_dis}
out["retention_first100_request_exact"] = {t: tab[t]["deployed"]["first100"]["request_exact"] for t in ("T100", "T500", "T1000", "T2000")}
A.say("D1 tf vs dec:", out["D1_tf_vs_dec_on_deployed_rows"], "retention:", out["retention_first100_request_exact"])
A.jdump(out, out_path)

# ----------------------------------------------------------------------------- 4. public scores from the stored case files


def pub_scores(d):
    es = np.mean([x["target_true"] > x["target_new"] for x in d["rewrite_prompts_probs"]])
    ps = np.mean([x["target_true"] > x["target_new"] for x in d["paraphrase_prompts_probs"]])
    ns = np.mean([x["target_true"] < x["target_new"] for x in d["neighborhood_prompts_probs"]])
    return es, ps, ns


pubt = {}
for tag, name in NAMES.items():
    src, T, first_rec = A.LOGS[name]
    res = {}
    for label, run in (("deployed", f"run_{tag}"), ("pre_edit", "run_W0")):
        S, nf, fb = [], 0, []
        for ri in range(first_rec, first_rec + T):
            d = A.jload(os.path.join(A.B_OUT, "public_results", "ROME_seq", run, f"case_{data[ri]['case_id']}.json"))
            S.append(pub_scores(d["post"]))
            vals = [x[k] for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs") for x in d["post"][key] for k in ("target_new", "target_true")]
            nf += not np.all(np.isfinite(vals))
            if d.get("prob_test_use_cache_false"):
                fb.append(ri)
        S = np.array(S) * 100
        res[label] = {"n": len(S), "ES": float(S[:, 0].mean()), "PS": float(S[:, 1].mean()), "NS": float(S[:, 2].mean()), "n_nonfinite_records": int(nf),
                      "records_with_use_cache_false": fb}
        if src == "stream":
            res[label]["first100"] = {"ES": float(S[:100, 0].mean()), "PS": float(S[:100, 1].mean()), "NS": float(S[:100, 2].mean())}
    rep = REPORT[tag]
    res["vs_report"] = {k: [res["deployed"][k], rep[k]] for k in ("ES", "PS", "NS")}
    res["all_equal_to_reported_digits"] = all(abs(a - b) < 0.05 for a, b in res["vs_report"].values())
    pubt[tag] = res
    A.say(tag, "public scores deployed:", {k: round(res["deployed"][k], 2) for k in ("ES", "PS", "NS")}, "pre-edit:",
          {k: round(res["pre_edit"][k], 2) for k in ("ES", "PS", "NS")}, "report ok:", res["all_equal_to_reported_digits"])
out["public_scores_from_case_files"] = pubt
A.jdump(out, out_path)

# ----------------------------------------------------------------------------- 5. generation entropy from the stored texts
sys.path.append(A.PYDEPS)
import nltk  # noqa: E402


def my_entropy(text):
    """n-gram entropy as defined in the public eval_utils_counterfact: bigram and trigram entropies (bits), weights 2/3 and
    4/3, arithmetic mean.  Re-implemented."""
    toks = nltk.word_tokenize(text)
    es = []
    for n_, w in ((2, 2 / 3), (3, 4 / 3)):
        grams = [tuple(toks[i:i + n_]) for i in range(len(toks) - n_ + 1)]
        c = np.array(list(collections.Counter(grams).values()), dtype=np.float64)
        p = c / c.sum()
        es.append(w * float(-(p * np.log2(p)).sum()))
    return float(np.mean(es))


gen = {}
expl_objs = None
for tag in ("W0", "dev100", "T100", "T500", "T1000", "T2000"):
    for f in sorted(glob.glob(os.path.join(A.B_OUT, "describe", tag, "gen_*.json"))):
        g = A.jload(f)
        sname = os.path.basename(f)[4:-5]
        ok = [r for r in g["rows"] if "failed" not in r]
        mine_rec = [float(np.mean([my_entropy(t) for t in r["text"]])) for r in ok]
        stored_rec = [r["ngram_entropy"] for r in ok]
        texts = [t for r in ok for t in r["text"]]
        words = [w for t in texts for w in t.split()]
        top = collections.Counter(words).most_common(8)
        distinct_ratio = float(np.mean([len(set(t.split())) / max(1, len(t.split())) for t in texts]))
        gen[f"{tag}:{sname}"] = {"n_records": len(ok), "n_failed": len(g["rows"]) - len(ok), "n_texts": len(texts),
                                 "GE_audit_x100": float(100 * np.mean(mine_rec)), "GE_stored_x100": float(100 * np.mean(stored_rec)),
                                 "max_abs_diff_per_record": float(np.max(np.abs(np.array(mine_rec) - np.array(stored_rec)))),
                                 "mean_words_per_text": float(np.mean([len(t.split()) for t in texts])),
                                 "mean_share_of_distinct_words_per_text": distinct_ratio,
                                 "share_of_all_words_taken_by_the_8_most_frequent": float(sum(c for _, c in top) / max(1, len(words))),
                                 "top_words": top, "example": texts[7][:220]}
out["generation_entropy"] = gen
for k, v in gen.items():
    A.say("GE", k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ("example",)})
out["ground_truth_vs_evaluation"] = {
    "D1": "ground truth y0 = greedy continuation of the PRE-EDIT model (pre_continuations.pt, weight sha256 = W_0); evaluated: the DEPLOYED "
          "model's argmax (teacher-forced on y0, and free decode).  'valid' uses the pre-edit model only.  The pre-edit row (D1 = 0) is a re-run "
          "of the pre-edit model, i.e. a reproducibility figure, not an evaluation.",
    "exact_match": "target = dataset string of the new object (tok(' ' + object)); evaluated: argmax of the deployed model.  Independent of ES/PS/NS.",
    "public_scores": "P(new) vs P(true) from the public probability test; never used to define a ground truth of (i) or (iii).",
    "pre_continuations_weight_sha256": PRE["w0"]["sha256"],
    "pre_continuations_weight_is_W0": PRE["w0"]["sha256"] == A.tensor_sha256(A.load_w0())}
out["finished"] = A.now()
A.jdump(out, out_path)
A.say("done")
