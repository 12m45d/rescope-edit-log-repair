"""E10 steps 6-7: sequential ROME logs (the weight is NOT restored between requests), one request per update.

    dev     records 15000 .. 15099   (development log, T = 100)
    stream  records 0 .. 1999        (reported stream; snapshots after 100, 500, 1000, 2000 requests)

Each request goes through the public apply_rome_to_model(model, tok, [request], hparams, copy=False,
return_orig_weights=False): execute_rome computes the two factors on the CURRENT weight, the public code adds their
outer product to the Conv1D weight in place (float32).  The only difference from the public evaluation driver is that
the weight is not restored afterwards (DEVIATIONS.txt).

Stored per update (updates/u_<i>.pt): what execute_rome returns (left_vector d_in, right_vector d_out), the module
input at the last subject token averaged over the context templates (the vector compute_u multiplies by the inverse
second moment), the module input / output of the request prompt alone (the key compute_v divides by, and the output
it subtracts), the target value, target_init, delta, the request, the record index, the seed, norms and checks.
Stored per run: the context templates in force (sampled once by the public code from the pre-edit model at the
first request, cached in a process global; restored from meta.json when a run is resumed), the pre-edit weight,
the deployed weight at every snapshot, sha256 of the weight every 25 updates.
Right after each update the public probability test (compute_rewrite_quality_counterfact, no generation) is run on
the request's own record with the deployed weight of that moment (write_time/case_<id>.json, key "post").

Resume: the weight is rebuilt from the pre-edit weight and the stored factors with the public expression
(left[:, None] @ right[None, :], added in place in float32: bit for bit the original writes; checked against the
recorded sha256), then the stream continues.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, choices=["dev", "stream"])
ap.add_argument("--limit", type=int, default=None, help="stop after this many updates in total (testing)")
args = ap.parse_args()
START, T, SNAPS = {"dev": (C.DEV_START, C.DEV_T, (C.DEV_T,)), "stream": (C.STREAM_START, C.STREAM_T, C.SNAPSHOTS)}[args.name]
if args.limit:
    T = min(T, args.limit)

C.setup_public()
import torch  # noqa: E402

cap_bytes = C.cap_gpu()
from rome import apply_rome_to_model  # noqa: E402
import rome.rome_main as rome_main  # noqa: E402
from rome.rome_main import upd_matrix_match_shape  # noqa: E402

D = os.path.join(C.LOGS_EDIT, args.name)
for sub in ("updates", "write_time", "snapshots"):
    os.makedirs(os.path.join(D, sub), exist_ok=True)
meta_path = os.path.join(D, "meta.json")
prog_path = os.path.join(D, "progress.jsonl")
pub_log = os.path.join(C.LOGS, f"{args.name}_public_stdout.log")
HASH_EVERY = 25

model, tok = C.load_model()
hparams = C.load_hparams()
data = C.load_counterfact()
W = C.get_weight(model)
W0 = W.detach().clone()
w0_hash = C.sha256_tensor(W0)
w0_path = os.path.join(C.LOGS_EDIT, "W0_transformer.h.17.mlp.c_proj.weight.pt")
if not os.path.exists(w0_path):
    C.tsave({"weight": W0.cpu(), "sha256": w0_hash, "name": C.WEIGHT_NAME,
             "note": "pre-edit Conv1D weight (d_in, d_out) float32; y = x @ W + b"}, w0_path)
from util import nethook  # noqa: E402
b0 = nethook.get_parameter(model, C.MODULE + ".bias").detach().clone()
capt = C.Capture()
capt.install()


def upath(i):
    return os.path.join(D, "updates", f"u_{i:05d}.pt")


n_have = 0
while os.path.exists(upath(n_have)):
    n_have += 1

if os.path.exists(meta_path):
    meta = C.jload(meta_path)
    assert meta["w0"] == w0_hash and meta["seed"] == C.SEED and meta["start"] == START
    assert meta["context_templates"] is not None or n_have == 0
    if meta["context_templates"] is not None:
        rome_main.CONTEXT_TEMPLATES_CACHE = list(meta["context_templates"])
    meta.setdefault("resumed", []).append({"at": C.now(), "n_have": n_have})
    meta["T_planned"] = T                                   # a test run with --limit may have written a shorter plan
    meta["snapshots"] = list(SNAPS)
    meta["what"] = (f"sequential ROME log '{args.name}': records {START} .. {START + T - 1}, one request per update, "
                    "weight not restored between requests")
else:
    assert n_have == 0
    meta = {"what": f"sequential ROME log '{args.name}': records {START} .. {START + T - 1}, one request per update, "
                    "weight not restored between requests",
            "name": args.name, "start": START, "T_planned": T, "snapshots": list(SNAPS), "seed": C.SEED,
            "w0": w0_hash, "bias_sha256": C.sha256_tensor(b0), "weight_name": C.WEIGHT_NAME, "module": C.MODULE,
            "hparams_file": C.HPARAMS_FILE, "hparams": C.jload(C.HPARAMS_FILE), "versions": C.versions(),
            "started": C.now(), "context_templates": None, "cap_bytes": cap_bytes}
    C.jdump(meta, meta_path)

# ----------------------------------------------------------------------------- resume: replay the stored factors
hashes = {}
if os.path.exists(prog_path):
    for line in open(prog_path):
        r = json.loads(line)
        if r.get("w_sha256"):
            hashes[r["n_applied"]] = r["w_sha256"]
if n_have:
    t0 = time.time()
    n_checked = 0
    with torch.no_grad():
        for i in range(n_have):
            u = torch.load(upath(i), map_location="cuda")
            assert u["index"] == i and u["case_id"] == data[START + i]["case_id"]
            upd = u["tensors"]["left"].unsqueeze(1) @ u["tensors"]["right"].unsqueeze(0)
            W[...] += upd_matrix_match_shape(upd, W.shape)
            if (i + 1) in hashes:
                got = C.sha256_tensor(W)["sha256"]
                assert got == hashes[i + 1], f"replayed weight differs from the recorded one after {i + 1} updates"
                n_checked += 1
    C.say(f"resumed '{args.name}': replayed {n_have} updates in {time.time() - t0:.1f}s; {n_checked} recorded hashes reproduced")
    # Does this process compute what the interrupted one computed?  The direction (left vector) of an update depends on
    # the keys (upstream of the edited module, hence independent of the edited weight) and on the inverse second moment
    # only: recompute it with the public compute_u for the last stored update and compare with the stored vector.
    from copy import deepcopy
    from rome.compute_u import compute_u
    from rome.rome_main import get_context_templates
    last = torch.load(upath(n_have - 1), map_location="cuda")
    sys.stdout = C.Tee(pub_log)
    try:
        with torch.no_grad():
            left_again = compute_u(model, tok, deepcopy(last["request"]), hparams, hparams.layers[0],
                                   get_context_templates(model, tok, hparams.context_template_length_params))
    finally:
        sys.stdout = sys.__stdout__
    chk = {"at": C.now(), "n_have": n_have, "update_index": n_have - 1, "n_recorded_hashes_reproduced": n_checked,
           "left_vector_bitwise_equal": bool(torch.equal(left_again, last["tensors"]["left"])),
           "left_vector_max_abs_diff": float((left_again - last["tensors"]["left"]).abs().max()),
           "gpu": torch.cuda.get_device_name(0)}
    meta.setdefault("resume_checks", []).append(chk)
    C.say("resume check:", chk)
    C.jdump(meta, meta_path)

tee = C.Tee(pub_log)
torch.manual_seed(C.SEED)
t_all = time.time()
n_new = 0
for i in range(n_have, T):
    record = data[START + i]
    case_id = record["case_id"]
    request = record["requested_rewrite"]
    sys.stdout = tee
    try:
        capt.begin()
        t0 = time.time()
        apply_rome_to_model(model, tok, [request], hparams, copy=False, return_orig_weights=False)
        exec_time = time.time() - t0
        cur = capt.end()
        if meta["context_templates"] is None:
            meta["context_templates"] = list(rome_main.CONTEXT_TEMPLATES_CACHE)
            C.jdump(meta, meta_path)
        t0 = time.time()
        post, fb = C.public_prob_test(model, tok, record)       # the public test; memory fallback: see tp_common
        eval_time = time.time() - t0
    finally:
        sys.stdout = sys.__stdout__
    assert list(rome_main.CONTEXT_TEMPLATES_CACHE) == meta["context_templates"]
    ten, sc = C.update_record(cur)
    with torch.no_grad():
        sc["W_fro_after"] = float(W.double().norm())
        sc["W_minus_W0_fro_after"] = float((W.double() - W0.double()).norm())
        sc["W_finite_after"] = bool(torch.isfinite(W).all())
    sc.update({"exec_time": exec_time, "eval_time": eval_time})
    C.jdump({"case_id": case_id, "index": i, "requested_rewrite": request, "time": exec_time, "post": post,
             "prob_test_use_cache_false": bool(fb)},
            os.path.join(D, "write_time", f"case_{case_id}.json"))
    C.tsave({"index": i, "record_index": START + i, "case_id": case_id, "request": request, "seed": C.SEED,
             "weight_name": cur["weight_name"], "tensors": ten, "scalars": sc}, upath(i))      # marks the update as done
    n_applied = i + 1
    n_new += 1
    prog = {"n_applied": n_applied, "case_id": case_id, "update_fro": sc["update_fro"], "W_fro_after": sc["W_fro_after"],
            "W_minus_W0_fro_after": sc["W_minus_W0_fro_after"], "exec_time": exec_time, "at": C.now(), "w_sha256": None}
    if n_applied % HASH_EVERY == 0 or n_applied in SNAPS or n_applied == T:
        prog["w_sha256"] = C.sha256_tensor(W)["sha256"]
    if n_applied in SNAPS:
        C.tsave({"weight": W.detach().cpu().clone(), "T": n_applied, "sha256": C.sha256_tensor(W), "name": C.WEIGHT_NAME,
                 "log": args.name, "at": C.now()}, os.path.join(D, "snapshots", f"W_T{n_applied}.pt"))
        C.say(f"snapshot T = {n_applied} written")
    with open(prog_path, "a") as f:
        f.write(json.dumps(prog) + "\n")
    if n_new % 10 == 0 or n_new == 1:
        el = time.time() - t_all
        C.say(f"{args.name} update {n_applied}/{T} (case {case_id}): {el / n_new:.2f} s/update; |dW| {sc['update_fro']:.2f} "
              f"|W-W0| {sc['W_minus_W0_fro_after']:.1f} <w,k_avg> {sc['w_dot_k_avg']:.4f} p_new {sc.get('v_prob_new', float('nan')):.3f}; "
              f"peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB (reserved {torch.cuda.max_memory_reserved() / 2 ** 20:.0f})")

meta["finished"] = C.now()
meta["T_done"] = T
meta["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
meta["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2 ** 20
C.jdump(meta, meta_path)
C.say("done:", args.name, T, "updates in", D)
