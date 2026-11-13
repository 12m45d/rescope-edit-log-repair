"""Audit, condition (1) determinism and condition-(1) identities, with the PUBLIC ROME code and a driver of the audit's own.

    python gpu_rome.py --tag det  --plan dev:0:3,dev:40:3,dev:97:3,stream:100:3,stream:173:3,stream:1530:3
    python gpu_rome.py --tag devfull --plan dev:0:100

For every item src:j:n of the plan
  * the weight is set to the state after j logged updates (W_0 + the first j logged outer products, written in float32 in
    place and in order, exactly as the public apply_rome_to_model writes them; compared with the builder's recorded
    sha256 of that state where one exists),
  * the public execute_rome(model, tok, request, hparams) is called for the next n requests of the data set (records
    first + j ...), each returned pair is applied with the public expression, and the pair is compared with the
    builder's stored factors (bitwise, max abs, relative L2),
  * the audit captures, with hooks of its own, the input and the output of transformer.h.17.mlp.c_proj at the last
    subject token of the request prompt alone, BEFORE and AFTER the update, and checks
        <left / s, k> = 1 with s = <left, k>,  right * s = target - output_before,  output_after = target,
        k (audit) = stored solver key, output_before (audit) = stored cur_output.
The context templates are sampled ONCE, at the start, by the public get_context_templates from the PRE-EDIT model after
torch.manual_seed(0) (what the public code does in an uninterrupted sequential process) and compared with the builder's.
All parameters other than the edited weight are hashed before and after.
No gate, no repair: only the public editor is run.  Output: out/rome_<tag>.json
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--plan", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
PLAN = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in args.plan.split(",")]

tfv = A.setup_stack("tf4")
import hashlib  # noqa: E402

import torch  # noqa: E402

cap = A.cap_gpu()
import rome.rome_main as rm  # noqa: E402
from rome import ROMEHyperParams  # noqa: E402
from rome.rome_main import execute_rome, get_context_templates, upd_matrix_match_shape  # noqa: E402
from util import nethook  # noqa: E402

out_path = os.path.join(A.AOUT, f"rome_{args.tag}.json")
pub_log = open(os.path.join(A.ALOGS, f"rome_{args.tag}_public_stdout.log"), "a", buffering=1)
model, tok = A.load_model()
hparams = ROMEHyperParams.from_json(A.HPARAMS_FILE)
data = A.load_counterfact()
W = nethook.get_parameter(model, A.WEIGHT_NAME)
mod = A.get_module(model)
bias0 = mod.bias.detach().clone()
W0_file = A.load_w0()
W0 = W.detach().clone()
F64 = torch.float64


def others_hash():
    h = hashlib.sha256()
    n = 0
    for name, p in sorted(model.state_dict().items()):
        if name == A.WEIGHT_NAME:
            continue
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
        n += 1
    return h.hexdigest(), n


res = {"written": A.now(), "tag": args.tag, "plan": args.plan, "seed": args.seed, "transformers": tfv, "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0), "cap_bytes": cap,
       "pre_edit_weight_of_loaded_model_equals_builder_W0_file_bitwise": bool(torch.equal(W0.cpu(), W0_file)),
       "pre_edit_weight_sha256": A.tensor_sha256(W0), "items": []}
h_before, n_other = others_hash()
res["other_parameters"] = {"n_tensors": n_other, "sha256_before": h_before}

# ---- context templates: sampled once from the pre-edit model, as the public code does in one sequential process
torch.manual_seed(args.seed)
with contextlib.redirect_stdout(pub_log):
    templates = list(get_context_templates(model, tok, hparams.context_template_length_params))
b_templates = {src: A.jload(os.path.join(A.B_LOGS_EDIT, src, "meta.json"))["context_templates"] for src in ("dev", "stream")}
b_templates["single"] = A.jload(os.path.join(A.B_OUT, "public_results", "ROME_single", "run_000", "run_meta.json"))["context_templates"]
res["context_templates"] = {"audit": templates, "n": len(templates),
                            "equal_to_builder": {k: templates == v for k, v in b_templates.items()},
                            "n_equal_positionwise": {k: sum(a == b for a, b in zip(templates, v)) for k, v in b_templates.items()}}
A.say("templates equal to the builder's:", res["context_templates"]["equal_to_builder"])
A.jdump(res, out_path)


def subject_last_index(prompt, subject):
    """Index of the last token of the subject in tok(prompt.format(subject)), from character offsets (the audit's own
    rule; the public code counts tokens of prefix and subject instead)."""
    text = prompt.format(subject)
    c0 = prompt.index("{}")
    c1 = c0 + len(subject)                              # exclusive
    enc = tok(text, return_offsets_mapping=True)
    idx = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c1 and b > c0]
    return text, enc["input_ids"], max(idx)


@torch.no_grad()
def key_and_output(ids, pos):
    got = {}
    h1 = mod.register_forward_pre_hook(lambda m, a: got.__setitem__("in", a[0].detach()[0, pos].clone()))
    h2 = mod.register_forward_hook(lambda m, a, o: got.__setitem__("out", o.detach()[0, pos].clone()))
    try:
        model(input_ids=torch.tensor([ids], device="cuda"))
    finally:
        h1.remove()
        h2.remove()
    return got["in"], got["out"]


def cmp(a, b):
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    d = (a.to(F64) - b.to(F64))
    return {"bitwise_equal": bool(torch.equal(a, b)), "max_abs_diff": float(d.abs().max()),
            "rel_l2_diff": float(d.norm() / b.to(F64).norm())}


FAC, PROG = {}, {}
for src, j, n in PLAN:
    if src not in FAC:
        FAC[src] = A.load_factors(src)
        PROG[src] = {}
        for line in open(os.path.join(A.B_LOGS_EDIT, src, "progress.jsonl")):
            r = json.loads(line)
            if r.get("w_sha256"):
                PROG[src][r["n_applied"]] = r["w_sha256"]
    fac = FAC[src]
    first = fac["record_indices"][0]
    with torch.no_grad():
        W[...] = A.replay(W0, fac["left"], fac["right"], j, device="cuda")
    item = {"source": src, "state_after_updates": j, "n_updates_rerun": n, "state_sha256": A.tensor_sha256(W),
            "state_sha256_recorded_by_builder": PROG[src].get(j), "updates": []}
    if j in PROG[src]:
        item["state_equals_builder_recorded_hash"] = item["state_sha256"] == PROG[src][j]
    for i in range(j, j + n):
        rec = data[first + i]
        req = rec["requested_rewrite"]
        assert fac["case_ids"][i] == rec["case_id"] and fac["requests"][i] == req, (src, i)
        text, ids, pos = subject_last_index(req["prompt"], req["subject"])
        k_before, o_before = key_and_output(ids, pos)
        with contextlib.redirect_stdout(pub_log):
            deltas = execute_rome(model, tok, req, hparams)                    # PUBLIC
        assert list(deltas) == [A.WEIGHT_NAME]
        left, right = deltas[A.WEIGHT_NAME]
        with torch.no_grad():
            W[...] += upd_matrix_match_shape(left.unsqueeze(1) @ right.unsqueeze(0), W.shape)   # the public write
        k_after, o_after = key_and_output(ids, pos)
        ld, rd, kd = left.to(F64), right.to(F64), k_before.to(F64)
        s = float(ld @ kd)
        tgt = fac["target"][i].to(F64).cuda()
        resid = tgt - o_before.to(F64)
        u = {"index": i, "record": first + i, "case_id": rec["case_id"], "request": f"{text} -> {req['target_new']['str']}",
             "subject_last_token_index_audit": pos, "subject_last_token_index_builder": fac["scalars"][i]["lookup_idxs"][0],
             "left_vs_stored": cmp(left, fac["left"][i]), "right_vs_stored": cmp(right, fac["right"][i]),
             "update_fro": float(ld.norm() * rd.norm()), "update_fro_stored": float(fac["left"][i].double().norm() * fac["right"][i].double().norm()),
             "division_factor_audit_key": s, "division_factor_stored": fac["scalars"][i]["div_factor"],
             "readout_dot_key_minus_1": float((ld / s) @ kd - 1.0),
             "key_audit_vs_stored_k_request": cmp(k_before, fac["k_request"][i]),
             "key_audit_vs_stored_k_template_mean_rel": float((kd.cpu() - fac["k_template_mean"][i].double()).norm() / kd.norm().cpu()),
             "output_before_audit_vs_stored_cur_output": cmp(o_before, fac["cur_output"][i]),
             "right_times_s_vs_target_minus_output_rel": float((rd * s - resid).norm() / resid.norm()),
             "output_after_vs_stored_target_rel": float((o_after.to(F64) - tgt).norm() / tgt.norm()),
             "key_unchanged_by_update_bitwise": bool(torch.equal(k_before, k_after)),
             "module_output_formula_rel": float((kd @ (W.to(F64)) + bias0.to(F64) - o_after.to(F64)).norm() / o_after.to(F64).norm()),
             "weight_after_equals_replay_bitwise": bool(torch.equal(W, A.replay(W0, fac["left"], fac["right"], i + 1, device="cuda")))
             if n <= 5 else None}
        item["updates"].append(u)
        A.say(f"{src} update {i}: left bitwise {u['left_vs_stored']['bitwise_equal']} ({u['left_vs_stored']['rel_l2_diff']:.2e}) "
              f"right bitwise {u['right_vs_stored']['bitwise_equal']} ({u['right_vs_stored']['rel_l2_diff']:.2e}) |dW| {u['update_fro']:.2f} "
              f"s {s:.4f} <w,k>-1 {u['readout_dot_key_minus_1']:.1e} out_after-target {u['output_after_vs_stored_target_rel']:.1e}")
    Wend = A.replay(W0, fac["left"], fac["right"], j + n, device="cuda")
    item["final_weight_equals_replay_of_stored_factors_bitwise"] = bool(torch.equal(W, Wend))
    item["final_weight_sha256"] = A.tensor_sha256(W)
    item["final_weight_sha256_recorded_by_builder"] = PROG[src].get(j + n)
    item["n_left_bitwise"] = sum(u["left_vs_stored"]["bitwise_equal"] for u in item["updates"])
    item["n_right_bitwise"] = sum(u["right_vs_stored"]["bitwise_equal"] for u in item["updates"])
    item["max_left_rel_l2_diff"] = max(u["left_vs_stored"]["rel_l2_diff"] for u in item["updates"])
    item["max_right_rel_l2_diff"] = max(u["right_vs_stored"]["rel_l2_diff"] for u in item["updates"])
    res["items"].append(item)
    A.jdump(res, out_path)
    A.say(f"== {src}:{j}:{n} left bitwise {item['n_left_bitwise']}/{n} right bitwise {item['n_right_bitwise']}/{n} "
          f"final weight = replay: {item['final_weight_equals_replay_of_stored_factors_bitwise']}")

with torch.no_grad():
    W[...] = W0
h_after, _ = others_hash()
res["other_parameters"]["sha256_after"] = h_after
res["other_parameters"]["unchanged"] = h_after == h_before
res["templates_cache_unchanged"] = list(rm.CONTEXT_TEMPLATES_CACHE) == templates
res["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
res["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2 ** 20
res["finished"] = A.now()
A.jdump(res, out_path)
A.say("done", out_path, f"peak {res['peak_alloc_MiB']:.0f} MiB (reserved {res['peak_reserved_MiB']:.0f})")
