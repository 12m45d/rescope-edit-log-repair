"""Audit: the PUBLIC probability test (experiments/py/eval_utils_counterfact.compute_rewrite_quality_counterfact, imported
from the unmodified repository) re-run by the audit at the pre-edit and at deployed weights, against the builder's stored
public-format case files.  Only the pre-edit and the deployed model are evaluated.

    python gpu_pub.py --parts cmp      # samples: W0 (50 stream + 20 dev records), dev100 (50), T100 (30), T1000 (50), T2000 (30)
    python gpu_pub.py --parts uc       # use_cache True vs False on the records for which the builder used its memory fallback

 cmp  the weight is rebuilt by the audit (W_0 of the checkpoint + logged outer products, float32 in place) and compared
      bitwise with the snapshot; per record: are the stored probabilities reproduced (exact / max abs difference of the
      per-prompt negative log-probabilities), and the per-prompt success flags (new vs true).
 uc   deviation 15 of the builder (model.config.use_cache = False on out-of-memory): the public function's output with
      use_cache True and False on the flagged records [147, 427, 1531, 1878] at W_0, and with use_cache False against the
      stored case files of the weights where the fallback was used.
Output: out/pub_<parts>.json
"""
import argparse
import contextlib
import gc
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parts", default="cmp")
args = ap.parse_args()

tfv = A.setup_stack("tf4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

cap = A.cap_gpu()
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact  # noqa: E402  PUBLIC

model, tok = A.load_model()
mod = A.get_module(model)
W0 = mod.weight.detach().clone()
data = A.load_counterfact()
FAC = {}
out_path = os.path.join(A.AOUT, f"pub_{args.parts}.json")
RES = {"written": A.now(), "transformers": tfv, "cap_bytes": cap, "parts": args.parts}
KEYS = ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs", "attribute_prompts_probs")


def set_weight(Wn):
    with torch.no_grad():
        mod.weight.copy_(Wn)


def deployed(src, n):
    if src not in FAC:
        FAC[src] = A.load_factors(src)
    return A.replay(W0, FAC[src]["left"], FAC[src]["right"], n, device="cuda")


def flat(d):
    return np.array([x[k] for key in KEYS for x in d[key] for k in ("target_new", "target_true")], dtype=np.float64)


def flags(d):
    return [x["target_new"] < x["target_true"] for key in KEYS[:3] for x in d[key]]


def pub(record, use_cache=None):
    old = model.config.use_cache
    if use_cache is not None:
        model.config.use_cache = use_cache
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return compute_rewrite_quality_counterfact(model, tok, record, None, None)
    finally:
        model.config.use_cache = old


def stored(tag, cid):
    return A.jload(os.path.join(A.B_OUT, "public_results", "ROME_seq", f"run_{tag}", f"case_{cid}.json"))


if "cmp" in args.parts:
    PLAN = [("W0", None, 0, [("stream", 2000, 0, 50), ("dev", 100, 15000, 20)]),
            ("dev100", ("dev", 100), 0, [("dev", 100, 15000, 50)]),
            ("T100", ("stream", 100), 0, [("stream", 100, 0, 30)]),
            ("T1000", ("stream", 1000), 0, [("stream", 1000, 0, 50)]),
            ("T2000", ("stream", 2000), 0, [("stream", 2000, 0, 30)])]
    RES["cmp"] = {}
    for ti, (tag, dep, _, groups) in enumerate(PLAN):
        if dep is None:
            Wn, eq = W0, bool(torch.equal(W0.cpu(), A.load_w0()))
        else:
            Wn = deployed(*dep)
            eq = bool(torch.equal(Wn.cpu(), A.load_snapshot(*dep)))
        set_weight(Wn)
        rng = np.random.default_rng(A.AUDIT_SEED + 70 + ti)
        recs = []
        for src, T, first, n in groups:
            recs += [first + int(i) for i in sorted(rng.choice(T, n, replace=False))]
        n_exact, mx, n_flags, n_oom, fb = 0, 0.0, 0, 0, 0
        worst = None
        for ri in recs:
            st = stored(tag, data[ri]["case_id"])
            try:
                got = pub(data[ri], use_cache=False if st.get("prob_test_use_cache_false") else None)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                got = pub(data[ri], use_cache=False)
                n_oom += 1
            fb += bool(st.get("prob_test_use_cache_false"))
            a, b = flat(got), flat(st["post"])
            d = float(np.abs(a - b).max())
            n_exact += bool(np.array_equal(a, b))
            if d > mx:
                mx, worst = d, ri
            n_flags += flags(got) == flags(st["post"])
        RES["cmp"][tag] = {"n_records": len(recs), "weight_equals_builder_file_bitwise": eq, "n_records_all_probabilities_bitwise_equal": int(n_exact),
                           "max_abs_diff_of_neg_log_prob": mx, "record_of_max_diff": worst, "n_records_all_success_flags_equal": int(n_flags),
                           "n_records_recomputed_with_use_cache_false_after_oom": n_oom, "n_records_flagged_fallback_by_builder": fb,
                           "records": recs}
        A.jdump(RES, out_path)
        A.say("cmp", tag, {k: v for k, v in RES["cmp"][tag].items() if k != "records"})
    set_weight(W0)

if "uc" in args.parts:
    RES["uc"] = {"at_W0": {}, "against_stored": {}}
    set_weight(W0)
    for ri in (147, 427, 1531, 1878, 0, 1, 15000):
        r = {"n_prompt_tokens_max": int(max(len(tok(p)["input_ids"]) for p in data[ri]["paraphrase_prompts"] + data[ri]["neighborhood_prompts"]
                                              + data[ri]["attribute_prompts"]))}
        b = pub(data[ri], use_cache=False)
        try:
            torch.cuda.reset_peak_memory_stats()
            a = pub(data[ri], use_cache=True)
            r["peak_alloc_MiB_use_cache_true"] = torch.cuda.max_memory_allocated() / 2 ** 20
            r["identical_output_true_vs_false"] = bool(np.array_equal(flat(a), flat(b)))
            r["max_abs_diff_true_vs_false"] = float(np.abs(flat(a) - flat(b)).max())
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            r["use_cache_true"] = "out of memory under the audit's cap"
        st = stored("W0", data[ri]["case_id"])
        r["stored_W0_flagged_fallback"] = bool(st.get("prob_test_use_cache_false"))
        r["use_cache_false_equals_stored_W0_bitwise"] = bool(np.array_equal(flat(b), flat(st["post"])))
        RES["uc"]["at_W0"][str(ri)] = r
        A.jdump(RES, out_path)
        A.say("uc W0", ri, r)
    for tag, dep, recs in (("T500", ("stream", 500), (147, 427)), ("T1000", ("stream", 1000), (147, 427)), ("T2000", ("stream", 2000), (147, 427, 1531, 1878))):
        set_weight(deployed(*dep))
        for ri in recs:
            st = stored(tag, data[ri]["case_id"])
            b = pub(data[ri], use_cache=False)
            r = {"stored_flagged_fallback": bool(st.get("prob_test_use_cache_false")),
                 "use_cache_false_equals_stored_bitwise": bool(np.array_equal(flat(b), flat(st["post"]))),
                 "max_abs_diff": float(np.abs(flat(b) - flat(st["post"])).max()), "finite": bool(np.isfinite(flat(b)).all())}
            try:
                a = pub(data[ri], use_cache=True)
                r["identical_output_true_vs_false"] = bool(np.array_equal(flat(a), flat(b)))
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                r["use_cache_true"] = "out of memory under the audit's cap"
            RES["uc"]["against_stored"][f"{tag}:{ri}"] = r
            A.jdump(RES, out_path)
            A.say("uc", tag, ri, r)
    set_weight(W0)

RES["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
RES["finished"] = A.now()
A.jdump(RES, out_path)
A.say("done", out_path)
