"""Audit: tests of the audit's OWN interpretive statements (stored files; CPU).  Each block states the prediction that
would refute the statement before computing it.  Output: out/selftest.json

 1. "The exact-match hits of the collapsed logs are coincidences with the tokens the collapsed model emits everywhere."
    Test: let S = new objects of the first-token-subject updates of the log.  If hits are coincidences, the model emits
    X in S as first token on requests whose target is X at the SAME rate as on requests whose target is not X.
    If hits are retained edits, the rate on target-X requests is far higher.  Observed hits vs hits expected under
    independence (sum over X of n_X * rate of X on the other requests), request and paraphrase prompts, with the
    first-token-subject requests themselves counted separately.
 2. "The public n-gram entropy cannot show this collapse because sampling among a handful of tokens already gives a high
    value."  Test: the public formula on uniformly random sequences over k symbols of the length of the stored texts.
    If the statement holds, k = 5 gives values near those of the collapsed logs (about 580); if such sequences scored far
    lower, the explanation would be wrong.  (An asymptotic formula would give 619; it does not apply to 100-token texts.)
 3. Public magnitude scores (EM, PM, NM) of the deployed logs from the stored case files, next to ES / PS / NS: is "about
    50" on the collapsed logs accompanied by probability differences near zero?
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(A.MODEL_DIR)
data = A.load_counterfact()
out = {"written": A.now()}
out_path = os.path.join(A.AOUT, "selftest.json")
P = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
fac = A.load_factors("stream")
s = (fac["left"].double() * fac["k_request"].double()).sum(1).numpy()

# ---- 1. coincidence test
res1 = {}
for T in (500, 1000, 2000):
    tag = f"T{T}"
    d = torch.load(os.path.join(A.B_OUT, "describe", tag, "dec.pt"), map_location="cpu")
    pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
    ft_idx = [i for i in range(T) if s[i] < 0.5]
    S = sorted({data[i]["requested_rewrite"]["target_new"]["str"] for i in ft_idx})
    S_tok = {o: tok(" " + o)["input_ids"][0] for o in S}
    r_ = {"first_token_subject_requests": ft_idx, "their_new_objects": S}
    for kind in ("request", "paraphrase"):
        rows = [r for r in P if r["set"] == "stream" and r["kind"] == kind and r["pos"] < T]
        first = np.array([int(d["dec"][pos[r["text_id"]]][0]) for r in rows])
        tgt = np.array([tok(" " + data[r["record"]]["requested_rewrite"]["target_new"]["str"])["input_ids"][0] for r in rows])
        own = np.array([r["pos"] in ft_idx for r in rows])
        obs = exp = 0.0
        n_target_in_S = 0
        per = {}
        for o, x in S_tok.items():
            a = (tgt == x) & ~own                           # requests of OTHER edits whose target is X
            b = (tgt != x)                                  # requests whose target is not X
            rate_other = float((first[b] == x).mean())
            h = int((first[a] == x).sum())
            per[o] = {"n_requests_with_this_target": int(a.sum()), "hits": h, "rate_on_them": float((first[a] == x).mean()) if a.sum() else None,
                      "rate_of_this_token_on_requests_with_another_target": rate_other}
            obs += h
            exp += float(a.sum()) * rate_other
            n_target_in_S += int(a.sum())
        hits_all = int((first == tgt).sum())
        hits_own = int(((first == tgt) & own).sum())
        hits_outside_S = int(((first == tgt) & ~np.isin(tgt, list(S_tok.values()))).sum())
        r_[kind] = {"n_rows": len(rows), "hits_total": hits_all, "hits_on_the_first_token_subject_requests_themselves": hits_own,
                    "rows_of_other_edits_whose_target_is_in_S": n_target_in_S, "hits_among_them_observed": int(obs),
                    "hits_among_them_expected_if_output_is_independent_of_the_target": exp,
                    "poisson_sd_of_expected": float(np.sqrt(exp)), "observed_minus_expected_in_sd": float((obs - exp) / np.sqrt(exp)) if exp > 0 else None,
                    "hits_whose_target_is_not_in_S": hits_outside_S, "rows_whose_target_is_not_in_S": int((~np.isin(tgt, list(S_tok.values()))).sum()),
                    "per_object": per}
    res1[tag] = r_
    A.say(tag, {k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items() if a != "per_object"} for k, v in r_.items() if k in ("request", "paraphrase")})
out["1_exact_match_hits_of_collapsed_logs"] = res1
A.jdump(out, out_path)

# ---- 2. n-gram entropy of random sequences
sys.path.append(A.PYDEPS)
import nltk  # noqa: E402


def public_formula(toks):
    es = []
    for n_, w in ((2, 2 / 3), (3, 4 / 3)):
        grams = [tuple(toks[i:i + n_]) for i in range(len(toks) - n_ + 1)]
        c = np.array(list(collections.Counter(grams).values()), dtype=np.float64)
        p = c / c.sum()
        es.append(w * float(np.sum(-p * np.log(p) / np.log(2))))
    return float(np.mean(es))


lens = {}
for tag, f in (("W0", "gen_stream_first100.json"), ("T500", "gen_stream_first100.json"), ("T1000", "gen_stream_first100.json"), ("T2000", "gen_stream_first100.json")):
    g = A.jload(os.path.join(A.B_OUT, "describe", tag, f))
    L_ = [len(nltk.word_tokenize(t)) for r in g["rows"] for t in r["text"]]
    stored = float(100 * np.mean([r["ngram_entropy"] for r in g["rows"]]))
    lens[tag] = {"median_word_tokens_per_text": float(np.median(L_)), "stored_GE": stored}
rng = np.random.default_rng(A.AUDIT_SEED + 200)
sim = {}
for n_tok in (90, 100):
    for k in (2, 3, 5, 10, 50, 1000):
        vals = [public_formula([str(x) for x in rng.integers(0, k, n_tok)]) for _ in range(400)]
        sim[f"{n_tok} tokens, {k} symbols (uniform)"] = float(100 * np.mean(vals))
# skewed sampling: 5 symbols with the probabilities 0.5, 0.25, 0.15, 0.07, 0.03 (one dominant token)
for n_tok in (90, 100):
    vals = [public_formula([str(x) for x in rng.choice(5, n_tok, p=[0.5, 0.25, 0.15, 0.07, 0.03])]) for _ in range(400)]
    sim[f"{n_tok} tokens, 5 symbols (p = .5 .25 .15 .07 .03)"] = float(100 * np.mean(vals))
out["2_ngram_entropy_of_random_sequences"] = {"stored_texts": lens, "public_formula_on_random_sequences(x100, mean of 400)": sim,
                                              "asymptotic_value_for_5_uniform_symbols(not attainable at this length)": float(100 * np.mean([2 / 3 * np.log2(25), 4 / 3 * np.log2(125)]))}
A.say("n-gram entropy:", out["2_ngram_entropy_of_random_sequences"])
A.jdump(out, out_path)

# ---- 3. magnitude scores of the deployed logs
res3 = {}
for name, (src, T, first) in A.LOGS.items():
    tag = name.split("-")[-1]
    rows = {"deployed": f"run_{tag}", "pre_edit": "run_W0"}
    r_ = {}
    for label, run in rows.items():
        es, ps, ns, em, pm, nm = [], [], [], [], [], []
        for ri in range(first, first + T):
            dd = A.jload(os.path.join(A.B_OUT, "public_results", "ROME_seq", run, f"case_{data[ri]['case_id']}.json"))["post"]
            es.append(np.mean([x["target_true"] > x["target_new"] for x in dd["rewrite_prompts_probs"]]))
            ps.append(np.mean([x["target_true"] > x["target_new"] for x in dd["paraphrase_prompts_probs"]]))
            ns.append(np.mean([x["target_true"] < x["target_new"] for x in dd["neighborhood_prompts_probs"]]))
            em.append(np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in dd["rewrite_prompts_probs"]]))
            pm.append(np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in dd["paraphrase_prompts_probs"]]))
            nm.append(np.mean([np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in dd["neighborhood_prompts_probs"]]))
        # how large are the two probabilities themselves on the request prompt
        pn = [np.exp(-A.jload(os.path.join(A.B_OUT, "public_results", "ROME_seq", run, f"case_{data[ri]['case_id']}.json"))["post"]["rewrite_prompts_probs"][0]["target_new"])
              for ri in range(first, first + T)]
        r_[label] = {"ES": float(100 * np.mean(es)), "PS": float(100 * np.mean(ps)), "NS": float(100 * np.mean(ns)), "EM": float(100 * np.mean(em)),
                     "PM": float(100 * np.mean(pm)), "NM": float(100 * np.mean(nm)), "median_P(new object | request prompt)": float(np.median(pn)),
                     "share_P(new object | request prompt) > 0.5": float(100 * np.mean(np.array(pn) > 0.5))}
    res3[tag] = r_
    A.say(tag, {k: {a: round(b, 3) for a, b in v.items()} for k, v in r_.items()})
out["3_magnitude_scores"] = res3
out["finished"] = A.now()
A.jdump(out, out_path)
