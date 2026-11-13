"""Audit, cross-stack checks under the PROJECT ENVIRONMENT's transformers (5.x) -- the stack the logs were NOT built with.

 1. Keys: the input of transformer.h.17.mlp.c_proj at the last subject token of the request prompt alone (pre-edit model,
    the audit's own hook) vs the stored solver key k_request, and the division factor <left, k> with the stored left
    vector, for the eight first-token-subject requests of the stream, 12 other stream requests and 5 dev requests.
    (Is the tiny division factor of the exploding updates a property of the model, or of the old transformers stack?)
 2. The builder's reason for the old stack (deviation 6): the public util.generate.generate_fast under transformers 5.x.
    The same call as the public get_context_templates makes (seed 0); the texts are stored for inspection next to the
    templates the builder obtained under 4.23.1.
Only the pre-edit model is run.  Output: out/env_extra.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

tfv = A.setup_stack("env", public=True)
import torch  # noqa: E402

cap = A.cap_gpu()
model, tok = A.load_model()
mod = A.get_module(model)
data = A.load_counterfact()
F64 = torch.float64
out = {"written": A.now(), "transformers": tfv, "torch": torch.__version__}


def subject_last_index(prompt, subject):
    text = prompt.format(subject)
    c0 = prompt.index("{}")
    c1 = c0 + len(subject)
    enc = tok(text, return_offsets_mapping=True)
    return enc["input_ids"], max(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < c1 and b > c0)


@torch.no_grad()
def key_at(ids, pos):
    got = {}
    h = mod.register_forward_pre_hook(lambda m, a: got.__setitem__("in", a[0].detach()[0, pos].clone()))
    try:
        model(input_ids=torch.tensor([ids], device="cuda"))
    finally:
        h.remove()
    return got["in"]


rows = []
for src, idxs in (("stream", [173, 240, 266, 946, 1416, 1420, 1508, 1947, 0, 1, 50, 99, 100, 172, 174, 193, 341, 500, 999, 1999]), ("dev", [0, 25, 50, 75, 99])):
    fac = A.load_factors(src)
    first = fac["record_indices"][0]
    for i in idxs:
        req = data[first + i]["requested_rewrite"]
        ids, pos = subject_last_index(req["prompt"], req["subject"])
        k = key_at(ids, pos).double().cpu()
        ks = fac["k_request"][i].double()
        left = fac["left"][i].double()
        rows.append({"source": src, "index": i, "request": req["prompt"].format(req["subject"]), "subject_last_index": pos,
                     "key_rel_l2_diff_to_stored": float((k - ks).norm() / ks.norm()), "key_bitwise_equal": bool(torch.equal(k.float(), fac["k_request"][i])),
                     "division_factor_env_key": float(left @ k), "division_factor_stored_key": float(left @ ks),
                     "key_norm": float(k.norm()), "cos_key_vs_template_mean": float((k @ fac["k_template_mean"][i].double()) / (k.norm() * fac["k_template_mean"][i].double().norm()))})
out["keys"] = {"rows": rows, "max_key_rel_l2_diff": max(r["key_rel_l2_diff_to_stored"] for r in rows),
               "max_rel_diff_of_division_factor": max(abs(r["division_factor_env_key"] / r["division_factor_stored_key"] - 1) for r in rows)}
A.say("keys under", tfv, {k: v for k, v in out["keys"].items() if k != "rows"})
for r in rows[:9]:
    A.say("  ", r["source"], r["index"], r["request"][:40], "pos", r["subject_last_index"], "s", round(r["division_factor_env_key"], 4),
          "stored", round(r["division_factor_stored_key"], 4), "cos(k, k_avg)", round(r["cos_key_vs_template_mean"], 3))
A.jdump(out, os.path.join(A.AOUT, "env_extra.json"))

from util.generate import generate_fast  # noqa: E402   PUBLIC

torch.manual_seed(0)
try:
    g5 = generate_fast(model, tok, ["<|endoftext|>"], n_gen_per_prompt=10, max_out_len=5)
    g10 = generate_fast(model, tok, ["<|endoftext|>"], n_gen_per_prompt=10, max_out_len=10)
    out["generate_fast_env"] = {"len5": g5, "len10": g10}
except Exception as ex:  # noqa: BLE001
    out["generate_fast_env"] = {"error": f"{type(ex).__name__}: {str(ex)[:300]}"}
out["builder_templates_tf4"] = A.jload(os.path.join(A.B_LOGS_EDIT, "dev", "meta.json"))["context_templates"]
out["finished"] = A.now()
A.jdump(out, os.path.join(A.AOUT, "env_extra.json"))
A.say("generate_fast under", tfv, ":", out["generate_fast_env"])
