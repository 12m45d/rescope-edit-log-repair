"""Audit: the builder's WRITE-TIME case files (public probability test of each request right after its own update), re-derived
on a sample.  They carry two statements of the audit ("only 221 of 2000 requests had P(new object) > 0.5 on their own prompt
right after their own update"; the builder's 'at write' rows), so they are checked here and not taken on trust.

For 49 updates of the reported stream (40 random + 0, 172, 173, 174, 240, 500, 1530, 1531, 1999) and 10 random updates of the
development log: weight = W_0 + the first i+1 logged outer products (float32, in order); the PUBLIC
compute_rewrite_quality_counterfact on record i; compared with logs_edit/<run>/write_time/case_<id>.json.
Only deployed prefixes are evaluated.  Output: out/writetime.json
"""
import contextlib
import gc
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

tfv = A.setup_stack("tf4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

cap = A.cap_gpu()
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact  # noqa: E402

model, tok = A.load_model()
mod = A.get_module(model)
W0 = mod.weight.detach().clone()
data = A.load_counterfact()
KEYS = ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs", "attribute_prompts_probs")
out = {"written": A.now(), "transformers": tfv, "runs": {}}


def flat(d):
    return np.array([x[k] for key in KEYS for x in d[key] for k in ("target_new", "target_true")], dtype=np.float64)


def pub(record, use_cache=None):
    old = model.config.use_cache
    if use_cache is not None:
        model.config.use_cache = use_cache
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return compute_rewrite_quality_counterfact(model, tok, record, None, None)
    finally:
        model.config.use_cache = old


rng = np.random.default_rng(A.AUDIT_SEED + 300)
PLAN = {"stream": sorted(set(int(x) for x in rng.choice(2000, 40, replace=False)) | {0, 172, 173, 174, 240, 500, 1530, 1531, 1999}),
        "dev": sorted(int(x) for x in rng.choice(100, 10, replace=False))}
for src, idxs in PLAN.items():
    fac = A.load_factors(src)
    first = fac["record_indices"][0]
    W = W0.clone()
    n_applied = 0
    rows = []
    for i in idxs:
        with torch.no_grad():
            while n_applied < i + 1:
                W += fac["left"][n_applied].cuda().unsqueeze(1) @ fac["right"][n_applied].cuda().unsqueeze(0)
                n_applied += 1
            mod.weight.copy_(W)
        rec = data[first + i]
        st = A.jload(os.path.join(A.B_LOGS_EDIT, src, "write_time", f"case_{rec['case_id']}.json"))
        try:
            got = pub(rec, use_cache=False if st.get("prob_test_use_cache_false") else None)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            got = pub(rec, use_cache=False)
        a, b = flat(got), flat(st["post"])
        rows.append({"index": i, "identical": bool(np.array_equal(a, b)), "max_abs_diff": float(np.abs(a - b).max()),
                     "P_new_on_request_audit": float(np.exp(-got["rewrite_prompts_probs"][0]["target_new"])),
                     "P_new_on_request_stored": float(np.exp(-st["post"]["rewrite_prompts_probs"][0]["target_new"]))})
    with torch.no_grad():
        mod.weight.copy_(W0)
    out["runs"][src] = {"n_updates_checked": len(rows), "n_identical_to_stored_write_time_file": int(sum(r["identical"] for r in rows)),
                        "max_abs_diff": max(r["max_abs_diff"] for r in rows), "n_after_173": int(sum(r["index"] >= 173 for r in rows)) if src == "stream" else None,
                        "n_installed_flag_equal(P(new|request) > 0.5)": int(sum((r["P_new_on_request_audit"] > 0.5) == (r["P_new_on_request_stored"] > 0.5) for r in rows)),
                        "n_installed_in_sample_audit": int(sum(r["P_new_on_request_audit"] > 0.5 for r in rows)),
                        "n_installed_in_sample_after_173_audit": int(sum(r["P_new_on_request_audit"] > 0.5 for r in rows if r["index"] >= 173)) if src == "stream" else None,
                        "rows": rows}
    A.say(src, {k: v for k, v in out["runs"][src].items() if k != "rows"})
    A.jdump(out, os.path.join(A.AOUT, "writetime.json"))
out["finished"] = A.now()
A.jdump(out, os.path.join(A.AOUT, "writetime.json"))
