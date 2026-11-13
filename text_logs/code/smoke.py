"""Smoke test: the public ROME code under this environment, on record 0 (edited alone, weight restored).
Measures peak GPU memory and timings; prints what the capture records.  Writes out/smoke.json."""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import sys
import time

sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/09_testbed_scale/text_public/code")
import tp_common as C  # noqa: E402

C.setup_public()
import torch  # noqa: E402

cap_bytes = C.cap_gpu()
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact, n_gram_entropy  # noqa: E402
from rome import apply_rome_to_model  # noqa: E402
import rome.rome_main as rome_main  # noqa: E402
from util import nethook  # noqa: E402
from util.generate import generate_fast  # noqa: E402

out = {"versions": C.versions(), "cap_bytes": cap_bytes}
t0 = time.time()
model, tok = C.load_model()
out["load_s"] = time.time() - t0
hp = C.load_hparams()
print(hp)
data = C.load_counterfact()
cap = C.Capture()
cap.install()
W = C.get_weight(model)
print("weight", W.shape, W.dtype, C.sha256_tensor(W))
out["mem_after_load_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20

torch.manual_seed(C.SEED)
recs = [data[0], data[1], data[2]]
for rec in recs:
    torch.cuda.reset_peak_memory_stats()
    cap.begin()
    t0 = time.time()
    edited, wc = apply_rome_to_model(model, tok, [rec["requested_rewrite"]], hp, copy=False, return_orig_weights=True)
    t_edit = time.time() - t0
    cur = cap.end()
    ten, sc = C.update_record(cur)
    print({k: v for k, v in sc.items() if k != "coord_templates"})
    print("coord_templates", [round(x, 4) for x in sc["coord_templates"]])
    # the weight really moved by outer(left, right)
    d = (W.detach().double() - wc[C.WEIGHT_NAME].double()).cpu()
    rec1 = torch.outer(ten["left"].double(), ten["right"].double())
    print("applied - outer(left,right): max abs", float((d - rec1).abs().max()), "rel fro", float((d - rec1).norm() / rec1.norm()))
    t0 = time.time()
    post = compute_rewrite_quality_counterfact(edited, tok, rec, None, None)
    t_eval = time.time() - t0
    with torch.no_grad():
        for k, v in wc.items():
            nethook.get_parameter(model, k)[...] = v.to("cuda")
    pre = compute_rewrite_quality_counterfact(model, tok, rec, None, None)
    print("case", rec["case_id"], "edit s", round(t_edit, 2), "eval s", round(t_eval, 2), "peak MiB", round(torch.cuda.max_memory_allocated() / 2 ** 20))
    print(" post rewrite", post["rewrite_prompts_probs"], "\n pre rewrite", pre["rewrite_prompts_probs"])
    print(" post para", post["paraphrase_prompts_probs"])
    print(" post nbr[:3]", post["neighborhood_prompts_probs"][:3], "\n pre nbr[:3]", pre["neighborhood_prompts_probs"][:3])
    out.setdefault("cases", []).append({"case_id": rec["case_id"], "edit_s": t_edit, "eval_s": t_eval,
                                        "peak_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
                                        "scalars": {k: v for k, v in sc.items() if k != "coord_templates"}})
print("templates:", rome_main.CONTEXT_TEMPLATES_CACHE)
out["templates"] = rome_main.CONTEXT_TEMPLATES_CACHE
print("weight restored:", C.sha256_tensor(W))

# generation path of the public evaluation (top-k sampling, 100 tokens) + n-gram entropy
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
torch.manual_seed(0)
gen = generate_fast(model, tok, recs[0]["generation_prompts"], n_gen_per_prompt=1, max_out_len=100)
out["gen_s"] = time.time() - t0
out["gen_peak_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
print("generation s", round(out["gen_s"], 2), "peak MiB", round(out["gen_peak_MiB"]))
for g in gen[:3]:
    print("   ", repr(g[:300]))
out["ngram_entropy"] = n_gram_entropy(gen)
print("ngram entropy", out["ngram_entropy"])
out["nvidia_smi_note"] = "peak torch allocation; add ~500 MiB of CUDA context"
out["max_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2 ** 20
print("max reserved MiB", out["max_reserved_MiB"])
C.jdump(out, C.OUT + "/smoke.json")
