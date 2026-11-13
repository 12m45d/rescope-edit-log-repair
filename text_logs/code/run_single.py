"""E10 step 5: single-edit check (audit condition 2).  Records 0 .. N-1, each edited ALONE from the pre-edit weight.

This is the loop of the public driver experiments/evaluate.py (main, lines 105-146), re-typed here because importing
that module needs the dependencies of the other baselines (hydra, higher, allennlp, knowledge-neurons):
    edited_model, weights_copy = apply_rome_to_model(model, tok, [record["requested_rewrite"]], hparams,
                                                     copy=False, return_orig_weights=True)
    post = compute_rewrite_quality_counterfact(edited_model, tok, record, snips, vec)
    restore the weight from weights_copy
    pre  = compute_rewrite_quality_counterfact(model, tok, record, snips, vec)
    -> case_<id>.json  {case_id, requested_rewrite, time, post, pre}        (the public file format)
Generation tests are skipped (the public flag --skip_generation_tests): snips = vec = None.
The public experiments/summarize.py then scores the directory (score_single.py).

Added, without changing a computed value: torch.manual_seed before the first request; capture of the factors and
keys of every edit (cases_factors/); a check that the restored weight is the pre-edit weight bit for bit.
Resumable: existing case files are skipped; the context templates of the first process are restored.
"""
# ---------------------------------------------------------------------------------------------------------------------
# NOTICE.  The loop of main() below re-types lines 105-146 of experiments/evaluate.py of the public ROME code
# (https://github.com/kmeng01/rome, commit 0874014cd9837e4365f3e6f3c71400ef11509e04).  That code is distributed under
# the following licence, which applies to the re-typed lines:
#
#   MIT License
#
#   Copyright (c) 2022 Kevin Meng
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
# ---------------------------------------------------------------------------------------------------------------------
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=C.SINGLE_N)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--run", default="run_000")
args = ap.parse_args()

C.setup_public()
import torch  # noqa: E402

cap_bytes = C.cap_gpu()
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact  # noqa: E402
from rome import apply_rome_to_model  # noqa: E402
import rome.rome_main as rome_main  # noqa: E402
from util import nethook  # noqa: E402
from util.globals import RESULTS_DIR  # noqa: E402

run_dir = os.path.join(str(RESULTS_DIR), "ROME_single", args.run)       # public layout: results/<dir_name>/run_xxx
fac_dir = os.path.join(run_dir, "cases_factors")
os.makedirs(fac_dir, exist_ok=True)
meta_path = os.path.join(run_dir, "run_meta.json")
pub_log = os.path.join(C.LOGS, f"single_{args.run}_public_stdout.log")

model, tok = C.load_model()
hparams = C.load_hparams()
if not os.path.exists(os.path.join(run_dir, "params.json")):
    import shutil
    shutil.copyfile(C.HPARAMS_FILE, os.path.join(run_dir, "params.json"))   # as the public driver does
data = C.load_counterfact()
W = C.get_weight(model)
W0 = W.detach().clone()
w0_hash = C.sha256_tensor(W0)
capt = C.Capture()
capt.install()

if os.path.exists(meta_path):
    meta = C.jload(meta_path)
    assert meta["w0"] == w0_hash, "pre-edit weight differs from the first process of this run"
    assert meta["seed"] == C.SEED
    rome_main.CONTEXT_TEMPLATES_CACHE = list(meta["context_templates"])     # the first process's cache
    meta["resumed"] = meta.get("resumed", []) + [C.now()]
    C.say("resumed; templates restored from", meta_path)
else:
    meta = {"what": "single-edit check, public protocol (weight restored after each record)",
            "seed": C.SEED, "w0": w0_hash, "weight_name": C.WEIGHT_NAME, "hparams_file": C.HPARAMS_FILE,
            "hparams": C.jload(C.HPARAMS_FILE), "versions": C.versions(), "started": C.now(),
            "context_templates": None, "cap_bytes": cap_bytes, "skip_generation_tests": True}
snips = vec = None

tee = C.Tee(pub_log)
torch.manual_seed(C.SEED)
t_all = time.time()
n_done = 0
for idx in range(args.start, args.start + args.n):
    record = data[idx]
    case_id = record["case_id"]
    assert case_id == idx
    case_path = os.path.join(run_dir, f"case_{case_id}.json")
    if os.path.exists(case_path):
        continue
    sys.stdout = tee
    try:
        capt.begin()
        start = time.time()
        edited_model, weights_copy = apply_rome_to_model(model, tok, [record["requested_rewrite"]], hparams,
                                                         copy=False, return_orig_weights=True)
        exec_time = time.time() - start
        cur = capt.end()
        if meta["context_templates"] is None:
            meta["context_templates"] = list(rome_main.CONTEXT_TEMPLATES_CACHE)
            C.jdump(meta, meta_path)
        start = time.time()
        metrics = {"case_id": case_id, "requested_rewrite": record["requested_rewrite"], "time": exec_time,
                   "post": compute_rewrite_quality_counterfact(edited_model, tok, record, snips, vec)}
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
        metrics["pre"] = compute_rewrite_quality_counterfact(model, tok, record, snips, vec)
        eval_time = time.time() - start
    finally:
        sys.stdout = sys.__stdout__
    assert torch.equal(W, W0), "weight not restored"
    assert list(rome_main.CONTEXT_TEMPLATES_CACHE) == meta["context_templates"]
    ten, sc = C.update_record(cur)
    sc.update({"case_id": case_id, "exec_time": exec_time, "eval_time": eval_time})
    C.tsave({"tensors": ten, "scalars": sc}, os.path.join(fac_dir, f"case_{case_id}.pt"))
    C.jdump(metrics, case_path)                       # last: its existence marks the case as complete
    n_done += 1
    if n_done % 20 == 0 or n_done == 1:
        el = time.time() - t_all
        C.say(f"case {case_id}: {n_done} done in {el:.0f}s ({el / n_done:.2f} s/case); edit {exec_time:.2f}s eval {eval_time:.2f}s; "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB (reserved {torch.cuda.max_memory_reserved() / 2 ** 20:.0f})")

meta["finished"] = C.now()
meta["n_case_files"] = len([f for f in os.listdir(run_dir) if f.startswith("case_") and f.endswith(".json")])
meta["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
meta["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2 ** 20
C.jdump(meta, meta_path)
C.say("done", meta["n_case_files"], "case files in", run_dir)
