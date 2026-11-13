"""Audit: statements of the builder's report that are not covered by a condition, checked against stored files with the
audit's own arithmetic.  CPU.  Output: out/claims.json

 1. p_new, division factor and right-vector norm of every update: the scalars the builder stored (read from the frame of
    the public compute_v) against what the PUBLIC code printed itself (logs/*_public_stdout.log, parsed by the audit).
 2. "From 173 on, the editor's own optimisation fails on 99.0 % of updates"; write-time public scores before / from 173.
 3. Single edit vs stream: direction (left vector) and both keys of records 0..999.
 4. D1 curve of the development log (builder's file) next to the audit's 300-prompt sample.
 5. Number of deviations listed; item 15's unfinished sentence.
 6. Per-update key identity of the stream with the template-averaged key (the key the task text specified).
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

out = {"written": A.now()}
out_path = os.path.join(A.AOUT, "claims.json")
data = A.load_counterfact()
B_LOGS = os.path.join(A.TP, "logs")


# ----------------------------------------------------------------------------- 1. public printouts vs stored scalars
def parse_public_log(path):
    """Blocks of the public code's stdout, one per execute_rome call."""
    txt = open(path, errors="replace").read()
    blocks = txt.split("Executing ROME algorithm for the update: ")[1:]
    res = []
    for b in blocks:
        head = b.split("\n", 1)[0]
        probs = re.findall(r"avg prob of \[.*?\] ([-+0-9.eE]+|nan|inf)\s*\n", b)
        m_div = re.search(r"Division Factor: ([-+0-9.eE]+)", b)
        m_rv = re.search(r"Right vector norm: ([-+0-9.eE]+)", b)
        m_dn = re.search(r"Delta norm: ([-+0-9.eE]+)", b)
        m_lk = re.search(r"Lookup index found: (-?\d+) \| Sentence: (.*?) \| Token:", b)
        res.append({"head": head, "n_steps": len(probs), "p_last": float(probs[-1]) if probs else None, "p_first": float(probs[0]) if probs else None,
                    "div": float(m_div.group(1)) if m_div else None, "right_norm": float(m_rv.group(1)) if m_rv else None,
                    "delta_norm": float(m_dn.group(1)) if m_dn else None, "lookup": int(m_lk.group(1)) if m_lk else None,
                    "complete": "Deltas successfully computed" in b})
    return res


pl = {}
for src, logf in (("dev", "dev_public_stdout.log"), ("stream", "stream_public_stdout.log")):
    fac = A.load_factors(src)
    n = fac["left"].shape[0]
    first = fac["record_indices"][0]
    blocks = [b for b in parse_public_log(os.path.join(B_LOGS, logf)) if b["complete"]]
    # the log is appended across processes; a request may appear more than once only if a process died after printing
    by_head = {}
    for b in blocks:
        by_head.setdefault(b["head"], []).append(b)
    n_found = n_p = n_div = n_rn = n_lk = 0
    worst_p = worst_div = 0.0
    multi = 0
    for i in range(n):
        rw = data[first + i]["requested_rewrite"]
        head = f"[{rw['prompt'].format(rw['subject'])}] -> [ {rw['target_new']['str']}]"
        cand = by_head.get(head, [])
        if not cand:
            continue
        n_found += 1
        multi += len(cand) > 1
        sc = fac["scalars"][i]
        # with several blocks under the same head (a repeated request text), take the one whose division factor matches
        b = min(cand, key=lambda x: abs((x["div"] or 0) - sc["div_factor"]))
        dp = abs(b["p_last"] - sc["v_prob_new"])
        dd = abs(b["div"] / sc["div_factor"] - 1)
        n_p += dp <= 1e-6 + 1e-5 * abs(sc["v_prob_new"])
        n_div += dd <= 1e-5
        n_rn += abs(b["right_norm"] / sc["right_norm"] - 1) <= 1e-4
        n_lk += b["lookup"] == sc["lookup_idxs"][0]
        worst_p, worst_div = max(worst_p, dp), max(worst_div, dd)
    pl[src] = {"n_updates": n, "n_complete_blocks_in_public_log": len(blocks), "n_updates_found_in_public_log": n_found,
               "n_requests_with_several_blocks": int(multi), "n_p_new_equal_to_printed": int(n_p), "max_abs_diff_p_new": worst_p,
               "n_division_factor_equal_to_printed": int(n_div), "max_rel_diff_division_factor": worst_div,
               "n_right_vector_norm_equal_to_printed": int(n_rn), "n_lookup_index_equal_to_printed": int(n_lk)}
    A.say("public printouts vs stored scalars,", src, pl[src])
out["public_printouts_vs_stored_scalars"] = pl

# ----------------------------------------------------------------------------- 2. the editor before / from update 173
fac = A.load_factors("stream")
p_new = np.array([s["v_prob_new"] for s in fac["scalars"]])
tn = np.array([s["target_norm"] for s in fac["scalars"]])
ti = np.array([s["target_init_norm"] for s in fac["scalars"]])


def wt_scores(lo, hi):
    es, ps, ns, ex = [], [], [], []
    for i in range(lo, hi):
        d = A.jload(os.path.join(A.B_LOGS_EDIT, "stream", "write_time", f"case_{data[i]['case_id']}.json"))["post"]
        es.append(np.mean([x["target_true"] > x["target_new"] for x in d["rewrite_prompts_probs"]]))
        ps.append(np.mean([x["target_true"] > x["target_new"] for x in d["paraphrase_prompts_probs"]]))
        ns.append(np.mean([x["target_true"] < x["target_new"] for x in d["neighborhood_prompts_probs"]]))
        # probability of the new object on the request prompt right after the request's own update
        ex.append(float(np.exp(-d["rewrite_prompts_probs"][0]["target_new"])))
    return {"n": hi - lo, "ES": float(100 * np.mean(es)), "PS": float(100 * np.mean(ps)), "NS": float(100 * np.mean(ns)),
            "share_P(new|request)>0.5_right_after_own_update": float(100 * np.mean(np.array(ex) > 0.5)),
            "median_P(new|request)_right_after_own_update": float(np.median(ex))}


out["editor_before_and_from_173"] = {
    "share_p_new_below_0.5": {"0..172": float(100 * (p_new[:173] < 0.5).mean()), "173..499": float(100 * (p_new[173:500] < 0.5).mean()),
                              "173..999": float(100 * (p_new[173:1000] < 0.5).mean()), "173..1999": float(100 * (p_new[173:] < 0.5).mean())},
    "n_updates_with_p_new>=0.5_from_173": int((p_new[173:] >= 0.5).sum()),
    "indices_with_p_new>=0.5_from_173": [int(i) + 173 for i in np.nonzero(p_new[173:] >= 0.5)[0]],
    "median_target_norm": {"0..172": float(np.median(tn[:173])), "173..1999": float(np.median(tn[173:]))},
    "median_target_init_norm(module output at the subject token before the update)": {"0..172": float(np.median(ti[:173])), "173..1999": float(np.median(ti[173:]))},
    "write_time_public_scores": {"0..172": wt_scores(0, 173), "173..499": wt_scores(173, 500), "173..1999": wt_scores(173, 2000)},
    "n_requests_installed_at_write_time(P(new|request) > 0.5)": {f"T{T}": int(sum(
        np.exp(-A.jload(os.path.join(A.B_LOGS_EDIT, "stream", "write_time", f"case_{data[i]['case_id']}.json"))["post"]["rewrite_prompts_probs"][0]["target_new"]) > 0.5
        for i in range(T))) for T in (100, 500, 1000, 2000)}}
A.say("editor before / from 173:", json.dumps(out["editor_before_and_from_173"])[:1800])
A.jdump(out, out_path)

# ----------------------------------------------------------------------------- 3. single edit vs stream (records 0..999)
sdir = os.path.join(A.B_OUT, "public_results", "ROME_single", "run_000", "cases_factors")
n_left = n_kreq = n_kavg = n_right = 0
rel_right = []
for i in range(1000):
    a = torch.load(os.path.join(sdir, f"case_{i}.pt"), map_location="cpu")["tensors"]
    n_left += bool(torch.equal(a["left"], fac["left"][i]))
    n_kreq += bool(torch.equal(a["k_request"], fac["k_request"][i]))
    n_kavg += bool(torch.equal(a["k_avg"], fac["k_template_mean"][i]))
    n_right += bool(torch.equal(a["right"], fac["right"][i]))
    rel_right.append(float((a["right"].double() - fac["right"][i].double()).norm() / a["right"].double().norm()))
out["single_vs_stream_records_0_999"] = {"n": 1000, "n_left_bitwise_equal": int(n_left), "n_k_request_bitwise_equal": int(n_kreq),
                                         "n_k_template_mean_bitwise_equal": int(n_kavg), "n_right_bitwise_equal": int(n_right),
                                         "right_rel_l2_diff_median": float(np.median(rel_right)),
                                         "right_rel_l2_diff_median_first_100": float(np.median(rel_right[1:100])),
                                         "note": "left vector and keys do not depend on the edited weight (upstream of it); the right vector does"}
A.say("single vs stream:", out["single_vs_stream_records_0_999"])

# ----------------------------------------------------------------------------- 4. D1 curve of the development log
bc = A.jload(os.path.join(A.B_OUT, "d1_curve_dev.json"))
out["d1_curve_dev_builder"] = [{k: p[k] for k in ("n_updates", "D1_tf", "tok_loc") if k in p} for p in bc["points"]]
ev = os.path.join(A.AOUT, "eval_tf4.json")
if os.path.exists(ev):
    e = A.jload(ev).get("B", {}).get("TX-G2XL-rome-dev100", {})
    out["d1_curve_dev_audit_300_prompts"] = {**e.get("heldout_prefix_curve", {}), "100": {"audit_D1_tf_pct": e.get("heldout", {}).get("audit_D1_tf_pct"),
                                                                                         "audit_tok_loc_pct": e.get("heldout", {}).get("audit_tok_loc_pct")}}

# ----------------------------------------------------------------------------- 5. deviations
dv = open(os.path.join(A.B_OUT, "DEVIATIONS.txt")).read()
items = re.findall(r"^(\d+)\. (.+)$", dv, flags=re.M)
flat_dv = re.sub(r"\s+", " ", dv)
m_ = re.search(r"Equality of the public function's output[^.]*\.", flat_dv)
out["deviations"] = {"n_items": len(items), "titles": [t for _, t in items], "contains_'not run yet'": "not run yet" in flat_dv,
                     "sentence_with_'not run yet'": m_.group(0) if m_ else None}
A.say("deviations:", out["deviations"]["n_items"], out["deviations"]["sentence_with_'not run yet'"])

# ----------------------------------------------------------------------------- 6. template-mean coordinate, stream
F64 = torch.float64
left, kreq, kavg = fac["left"].to(F64), fac["k_request"].to(F64), fac["k_template_mean"].to(F64)
s = (left * kreq).sum(1)
c = ((left * kavg).sum(1) / s).numpy()
out["template_mean_coordinate_T2000"] = {"median": float(np.median(c)), "q95": float(np.quantile(c, 0.95)), "max": float(c.max()), "min": float(c.min()),
                                         "share_within_1pct": float(100 * np.mean(np.abs(c - 1) < 0.01)),
                                         "share_within_10pct": float(100 * np.mean(np.abs(c - 1) < 0.10)),
                                         "n_above_2": int((c > 2).sum()), "n_above_100": int((c > 100).sum()),
                                         "indices_above_100": [int(i) for i in np.nonzero(c > 100)[0]],
                                         "max_abs_request_key_coordinate_minus_1": float(np.abs((left * kreq).sum(1) / s - 1).max())}
A.say("template-mean coordinate (stream, 2000 updates):", out["template_mean_coordinate_T2000"])
# ----------------------------------------------------------------------------- 7. what the collapsed models decode
# share of the greedy tokens on the HELD-OUT prompts that are new objects of (a) the exploding updates, (b) the
# first-token-subject updates of the log -- from the stored decodes, with the environment's tokenizer
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(A.MODEL_DIR)
P = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
ho_t = sorted({r["text_id"] for r in P if r["set"] == "heldout"})
fro = (fac["left"].double().norm(dim=1) * fac["right"].double().norm(dim=1)).numpy()
sv = s.numpy()
W0n = float(A.load_w0().double().norm())
tk = {}
for tag, T in (("T100", 100), ("T500", 500), ("T1000", 1000), ("T2000", 2000)):
    res = {}
    for label, idx in (("exploding", [i for i in range(T) if fro[i] > W0n]), ("first_token_subject", [i for i in range(T) if sv[i] < 0.5])):
        objs = sorted({data[i]["requested_rewrite"]["target_new"]["str"] for i in idx})
        ids = torch.tensor(sorted({t for o in objs for t in tok(" " + o)["input_ids"]}), dtype=torch.long)
        row = {"n_updates": len(idx), "objects": objs}
        for model_tag in (tag, "W0"):
            d = torch.load(os.path.join(A.B_OUT, "describe", model_tag, "dec.pt"), map_location="cpu")
            pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
            dd = torch.stack([d["dec"][pos[t]] for t in ho_t])
            hit = torch.isin(dd, ids) if len(ids) else torch.zeros_like(dd, dtype=torch.bool)
            row["deployed" if model_tag == tag else "pre_edit"] = {"share_of_tokens_pct": float(100 * hit.float().mean()),
                                                                 "share_of_prompts_with_one_pct": float(100 * hit.any(1).float().mean()),
                                                                 "share_of_first_tokens_pct": float(100 * hit[:, 0].float().mean())}
        res[label] = row
    d = torch.load(os.path.join(A.B_OUT, "describe", tag, "dec.pt"), map_location="cpu")
    pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
    dd = torch.stack([d["dec"][pos[t]] for t in ho_t])
    vals, counts = torch.unique(dd, return_counts=True)
    top = torch.argsort(counts, descending=True)[:5]
    res["five_most_frequent_greedy_tokens"] = [[tok.decode([int(vals[i])]), float(100 * counts[i] / dd.numel())] for i in top]
    res["n_distinct_first_tokens_over_heldout"] = int(torch.unique(dd[:, 0]).numel())
    tk[tag] = res
out["heldout_greedy_tokens_of_deployed_models"] = tk
A.say("greedy tokens:", json.dumps({k: {"expl": v["exploding"]["deployed"], "top": v["five_most_frequent_greedy_tokens"], "first": v["n_distinct_first_tokens_over_heldout"]} for k, v in tk.items()})[:1800])
out["finished"] = A.now()
A.jdump(out, out_path)
