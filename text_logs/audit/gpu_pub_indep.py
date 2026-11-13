"""Audit: an INDEPENDENT computation of the quantities behind the public scores, and the numerical robustness of the
deployed models.  Only the pre-edit and the deployed model are run.

 1. Probabilities of the new / true object, the audit's own code: one sequence at a time, NO padding, no attention mask
    ("{prompt} {object}" tokenised as a whole; mean negative log-probability of the object's tokens, tok(' ' + object),
    read at the positions that predict them).  Against the builder's stored case files (which come from the public
    test_batch_prediction: one right-padded batch of 46 sequences per record), for W_0 (30 records), dev100 (50), T100 (30),
    T1000 (50): max abs difference, number of prompts whose success flag (P(new) vs P(true)) differs, ES / PS / NS on the
    sample from both.
 2. Precision: last-position logits of 40 HELD-OUT prompts under the deployed weights of dev100, T100 and T1000 and under
    W_0, in float32 on the GPU (single prompt, full forward) and in float64 on the CPU: argmax equal?  max abs logit
    difference, float64 top-1 margin.  (Are the outputs of the collapsed model numerically reproducible?)
Output: out/pub_indep.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

tfv = A.setup_stack("tf4", public=False)
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

cap = A.cap_gpu()
model, tok = A.load_model()
mod = A.get_module(model)
W0 = mod.weight.detach().clone()
data = A.load_counterfact()
out_path = os.path.join(A.AOUT, "pub_indep.json")
RES = {"written": A.now(), "transformers": tfv}
FAC = {}


def deployed(src, n):
    if src not in FAC:
        FAC[src] = A.load_factors(src)
    return A.replay(W0, FAC[src]["left"], FAC[src]["right"], n, device="cuda")


def set_weight(Wn):
    with torch.no_grad():
        mod.weight.copy_(Wn)


@torch.no_grad()
def nll(prompt, obj):
    """Mean negative log-probability of the tokens of ' ' + obj after `prompt` (single sequence, no padding)."""
    p_ids = tok(prompt)["input_ids"]
    o_ids = tok(" " + obj)["input_ids"]
    full = tok(f"{prompt} {obj}")["input_ids"]
    assert full == p_ids + o_ids, (prompt, obj)                  # the object starts a new token after the prompt
    lg = model(input_ids=torch.tensor([full], device="cuda"), use_cache=False).logits[0]
    lp = torch.log_softmax(lg[len(p_ids) - 1:len(full) - 1].double(), dim=-1)
    return float(-lp[torch.arange(len(o_ids)), torch.tensor(o_ids, device="cuda")].mean())


def scores(rows):
    es = np.mean([np.mean([n_ < t_ for n_, t_ in r["rewrite"]]) for r in rows])
    ps = np.mean([np.mean([n_ < t_ for n_, t_ in r["paraphrase"]]) for r in rows])
    ns = np.mean([np.mean([t_ < n_ for n_, t_ in r["neighborhood"]]) for r in rows])
    return {"ES": float(100 * es), "PS": float(100 * ps), "NS": float(100 * ns)}


RES["probabilities"] = {}
PLAN = [("W0", None, [("stream", 2000, 0, 20), ("dev", 100, 15000, 10)]), ("dev100", ("dev", 100), [("dev", 100, 15000, 50)]),
        ("T100", ("stream", 100), [("stream", 100, 0, 30)]), ("T1000", ("stream", 1000), [("stream", 1000, 0, 50)])]
for ti, (tag, dep, groups) in enumerate(PLAN):
    set_weight(W0 if dep is None else deployed(*dep))
    rng = np.random.default_rng(A.AUDIT_SEED + 90 + ti)
    recs = []
    for src, T, first, n in groups:
        recs += [first + int(i) for i in sorted(rng.choice(T, n, replace=False))]
    mine, theirs, mx, n_flag, n_prompts, worst = [], [], 0.0, 0, 0, None
    for ri in recs:
        rec = data[ri]
        rw = rec["requested_rewrite"]
        st = A.jload(os.path.join(A.B_OUT, "public_results", "ROME_seq", f"run_{tag}", f"case_{rec['case_id']}.json"))["post"]
        groups_p = {"rewrite": [rw["prompt"].format(rw["subject"])], "paraphrase": rec["paraphrase_prompts"], "neighborhood": rec["neighborhood_prompts"]}
        m, t = {}, {}
        for g, prompts in groups_p.items():
            m[g] = [(nll(p, rw["target_new"]["str"]), nll(p, rw["target_true"]["str"])) for p in prompts]
            t[g] = [(x["target_new"], x["target_true"]) for x in st[f"{g}_prompts_probs"]]
            for (a1, a2), (b1, b2) in zip(m[g], t[g]):
                d = max(abs(a1 - b1), abs(a2 - b2))
                if d > mx:
                    mx, worst = d, ri
                n_flag += (a1 < a2) != (b1 < b2)
                n_prompts += 1
        mine.append(m)
        theirs.append(t)
    RES["probabilities"][tag] = {"n_records": len(recs), "n_prompts": n_prompts, "max_abs_diff_of_mean_neg_log_prob": mx, "record_of_max_diff": worst,
                                 "n_prompts_success_flag_differs": int(n_flag), "audit_unpadded": scores(mine), "builder_stored_public_padded": scores(theirs)}
    A.jdump(RES, out_path)
    A.say("probabilities", tag, RES["probabilities"][tag])
set_weight(W0)

# ---- 2. precision: float32 (GPU) vs float64 (CPU) last-position logits
PROMPTS = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
ho = [r for r in PROMPTS if r["set"] == "heldout"]
rng = np.random.default_rng(A.AUDIT_SEED + 99)
sample = [ho[int(i)] for i in sorted(rng.choice(len(ho), 40, replace=False))]
ids = [tok(r["text"])["input_ids"] for r in sample]
weights = {"W0": W0, "dev100": deployed("dev", 100), "T100": deployed("stream", 100), "T1000": deployed("stream", 1000)}
f32 = {}
for tag, Wn in weights.items():
    set_weight(Wn)
    with torch.no_grad():
        f32[tag] = [model(input_ids=torch.tensor([x], device="cuda"), use_cache=False).logits[0, -1].double().cpu() for x in ids]
set_weight(W0)
weights = {k: v.cpu() for k, v in weights.items()}
del model
torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM  # noqa: E402

m64 = AutoModelForCausalLM.from_pretrained(A.MODEL_DIR).double()
mod64 = A.get_module(m64)
RES["precision"] = {}
for tag, Wn in weights.items():
    with torch.no_grad():
        mod64.weight.copy_(Wn.double())
        rows = []
        for x, a in zip(ids, f32[tag]):
            b = m64(input_ids=torch.tensor([x]), use_cache=False).logits[0, -1]
            top = torch.topk(b, 2).values
            rows.append({"argmax_equal": bool(a.argmax() == b.argmax()), "max_abs_logit_diff": float((a - b).abs().max()),
                         "float64_top1_margin": float(top[0] - top[1]), "float64_logit_range": float(b.max() - b.min())})
    RES["precision"][tag] = {"n_prompts": len(rows), "n_argmax_equal_float32_vs_float64": int(sum(r["argmax_equal"] for r in rows)),
                             "max_abs_logit_diff": max(r["max_abs_logit_diff"] for r in rows),
                             "median_abs_logit_diff": float(np.median([r["max_abs_logit_diff"] for r in rows])),
                             "median_float64_top1_margin": float(np.median([r["float64_top1_margin"] for r in rows])),
                             "median_float64_logit_range": float(np.median([r["float64_logit_range"] for r in rows]))}
    A.jdump(RES, out_path)
    A.say("precision", tag, RES["precision"][tag])
RES["finished"] = A.now()
A.jdump(RES, out_path)
A.say("done")
