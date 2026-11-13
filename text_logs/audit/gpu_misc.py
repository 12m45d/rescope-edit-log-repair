"""Audit, two remaining GPU checks (public code + the audit's own decode).  Only the public editor on the DEVELOPMENT log
and the deployed / pre-edit model of the reported stream are run.

 T  Context templates and determinism (development log).  The public code samples its 21 context templates once per
    process, from the model as it is at the first request.  Re-running update 97 of the development log from the saved
    state before it:
      (a) templates sampled from the EDITED model at that state (seed 0)  -> are they the builder's?  factors vs stored
      (b) templates sampled from the PRE-EDIT model (seed 0), then the weight set to the state -> factors vs stored
    (b) is what an uninterrupted sequential process does and what the builder's resume restores from its meta file.
 K  The reported stream around update 173, deployed prefixes n = 100, 150, 173, 174, 200 (weights rebuilt by the audit):
      - 200 random HELD-OUT prompts: D1 (teacher-forced), share of pre-edit tokens kept, first token changed (the audit's
        single-prompt decode; y0 recomputed by the audit) against the builder's collapse_point.json (all 3000 prompts);
      - the first 100 request prompts: greedy first token = new object (audit's decode);
      - n = 173 and 174: the public probability test on records 0..99 against the builder's stored case files.
Output: out/misc.json
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

tfv = A.setup_stack("tf4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

cap = A.cap_gpu()
import rome.rome_main as rm  # noqa: E402
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact  # noqa: E402
from rome import ROMEHyperParams  # noqa: E402
from rome.rome_main import execute_rome, get_context_templates  # noqa: E402

model, tok = A.load_model()
mod = A.get_module(model)
W = mod.weight
W0 = W.detach().clone()
hparams = ROMEHyperParams.from_json(A.HPARAMS_FILE)
data = A.load_counterfact()
out_path = os.path.join(A.AOUT, "misc.json")
RES = A.jload(out_path) if os.path.exists(out_path) else {}
RES.update({"written": A.now(), "transformers": tfv})
sink = open(os.path.join(A.ALOGS, "misc_public_stdout.log"), "a", buffering=1)


def set_weight(Wn):
    with torch.no_grad():
        W.copy_(Wn)


def rel(a, b):
    a, b = a.detach().double().cpu(), b.detach().double().cpu()
    return float((a - b).norm() / b.norm())


# ----------------------------------------------------------------------------------------------------------- T
if "T" not in RES:
    fac = A.load_factors("dev")
    b_templates = A.jload(os.path.join(A.B_LOGS_EDIT, "dev", "meta.json"))["context_templates"]
    J = 97
    req = data[15000 + J]["requested_rewrite"]
    state = A.replay(W0, fac["left"], fac["right"], J, device="cuda")
    T = {"update": J, "request": req["prompt"].format(req["subject"]) + " -> " + req["target_new"]["str"]}
    for label, sample_at_state in (("a_templates_from_edited_model_at_state", True), ("b_templates_from_pre_edit_model", False)):
        rm.CONTEXT_TEMPLATES_CACHE = None
        set_weight(state if sample_at_state else W0)
        torch.manual_seed(0)
        with contextlib.redirect_stdout(sink):
            templ = list(get_context_templates(model, tok, hparams.context_template_length_params))
        set_weight(state)
        with contextlib.redirect_stdout(sink):
            d = execute_rome(model, tok, req, hparams)
        left, right = d[A.WEIGHT_NAME]
        T[label] = {"templates_equal_builder": templ == b_templates, "n_templates_equal_positionwise": int(sum(x == y for x, y in zip(templ, b_templates))),
                    "templates": templ, "left_bitwise_equal_stored": bool(torch.equal(left.cpu(), fac["left"][J])),
                    "right_bitwise_equal_stored": bool(torch.equal(right.cpu(), fac["right"][J])),
                    "left_rel_l2_diff": rel(left, fac["left"][J]), "right_rel_l2_diff": rel(right, fac["right"][J]),
                    "update_fro": float(left.double().norm() * right.double().norm()),
                    "outer_product_rel_fro_diff": float((torch.outer(left.double().cpu(), right.double().cpu())
                                                         - torch.outer(fac["left"][J].double(), fac["right"][J].double())).norm()
                                                        / (fac["left"][J].double().norm() * fac["right"][J].double().norm()))}
        A.say("T", label, {k: v for k, v in T[label].items() if k != "templates"})
    set_weight(W0)
    rm.CONTEXT_TEMPLATES_CACHE = None
    RES["T"] = T
    A.jdump(RES, out_path)

# ----------------------------------------------------------------------------------------------------------- K
if "K" not in RES:
    fac = A.load_factors("stream")
    PROMPTS = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
    ho = [r for r in PROMPTS if r["set"] == "heldout"]
    rng = np.random.default_rng(A.AUDIT_SEED + 120)
    sample = [ho[int(i)] for i in sorted(rng.choice(len(ho), 200, replace=False))]
    req_rows = [r for r in PROMPTS if r["set"] == "stream" and r["kind"] == "request" and r["pos"] < 100]
    bc = {p["n_requests"]: p for p in A.jload(os.path.join(A.B_OUT, "collapse_point.json"))["points"]}

    @torch.no_grad()
    def greedy(ids, n):
        x = torch.tensor([ids], device="cuda")
        res = []
        for _ in range(n):
            nxt = int(model(input_ids=x, use_cache=False).logits[0, -1].argmax())
            res.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], 1)
        return res

    @torch.no_grad()
    def tf(ids, cont):
        x = torch.tensor([ids + cont[:-1]], device="cuda")
        return model(input_ids=x, use_cache=False).logits[0, len(ids) - 1:].argmax(-1).tolist()

    set_weight(W0)
    ids = {r["prompt_id"]: tok(r["text"])["input_ids"] for r in sample + req_rows}
    y0 = {r["prompt_id"]: greedy(ids[r["prompt_id"]], 10) for r in sample}
    K = {"n_heldout_prompts": len(sample), "points": []}
    for n in (100, 150, 173, 174, 200):
        Wn = A.replay(W0, fac["left"], fac["right"], n, device="cuda")
        set_weight(Wn)
        d1, tl, ft = [], [], []
        for r in sample:
            p = tf(ids[r["prompt_id"]], y0[r["prompt_id"]])
            y = y0[r["prompt_id"]]
            d1.append(p != y)
            tl.append(float(np.mean([a == b for a, b in zip(p, y)])))
            ft.append(p[0] != y[0])
        ex = []
        for r in req_rows:
            tgt = tok(" " + data[r["record"]]["requested_rewrite"]["target_new"]["str"])["input_ids"]
            ex.append(greedy(ids[r["prompt_id"]], len(tgt)) == tgt)
        pt = {"n_requests": n, "weight_sha256": A.tensor_sha256(Wn), "weight_sha256_builder": bc[n]["weight_sha256"],
              "weight_equals_builder": A.tensor_sha256(Wn) == bc[n]["weight_sha256"],
              "W_minus_W0_fro": float((Wn.double() - W0.double()).norm()),
              "audit_200_prompts": {"D1_tf": float(100 * np.mean(d1)), "tok_loc": float(100 * np.mean(tl)), "first_token_changed": float(100 * np.mean(ft))},
              "builder_3000_prompts": {"D1_tf": bc[n]["D1_tf"], "tok_loc": bc[n]["tok_loc"], "first_token_changed": bc[n]["first_token_changed"]},
              "audit_first_100_requests_exact_pct": float(100 * np.mean(ex)),
              "builder_first100_public": {"ES": bc[n]["first100_ES"], "PS": bc[n]["first100_PS"], "NS": bc[n]["first100_NS"]}}
        if n in (173, 174):
            n_eq, es, ps, ns = 0, [], [], []
            for ri in range(100):
                with contextlib.redirect_stdout(io.StringIO()):
                    got = compute_rewrite_quality_counterfact(model, tok, data[ri], None, None)
                st = A.jload(os.path.join(A.B_OUT, "public_results", "ROME_seq_collapse", f"run_n{n:04d}", f"case_{data[ri]['case_id']}.json"))["post"]
                n_eq += got == st
                es.append(np.mean([x["target_true"] > x["target_new"] for x in got["rewrite_prompts_probs"]]))
                ps.append(np.mean([x["target_true"] > x["target_new"] for x in got["paraphrase_prompts_probs"]]))
                ns.append(np.mean([x["target_true"] < x["target_new"] for x in got["neighborhood_prompts_probs"]]))
            pt["public_test_records_0_99"] = {"n_records_identical_to_builder_case_files": int(n_eq),
                                              "audit": {"ES": float(100 * np.mean(es)), "PS": float(100 * np.mean(ps)), "NS": float(100 * np.mean(ns))}}
        K["points"].append(pt)
        A.say("K", json.dumps(pt)[:900])
    set_weight(W0)
    RES["K"] = K
    A.jdump(RES, out_path)

RES["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
RES["finished"] = A.now()
A.jdump(RES, out_path)
A.say("done")
