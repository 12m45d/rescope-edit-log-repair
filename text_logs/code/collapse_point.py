"""Where the reported stream breaks: the DEPLOYED model at prefixes around the first exploding update (deployed weights
only; no gate, no repair).

The audit of the stream shows that update 173 (0-based; the 174th request, 'Chicago is a twin city of') has a Frobenius
norm 50 times that of the pre-edit weight.  This script measures the deployed model just before and just after it:
weights after n = 150, 173, 174 and 200 requests of the stream, rebuilt by the exact float32 replay of the logged factors
(n = 100 is rebuilt too and must equal the T100 snapshot bit for bit).
    HELD-OUT benign prompts   D1 (teacher-forced), share of y0 tokens kept, first token changed, repetitive greedy
                              continuations                                    (same code path as describe.py)
    first 100 requests        the public probability test (compute_rewrite_quality_counterfact, no generation), scored by
                              the public experiments/summarize.py: ES / PS / NS
Writes out/collapse_point.json and out/collapse_point.txt.
"""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

NAME = "TX-G2XL-rome-T500"                     # any prefix log that contains the updates needed (factors are shared)
NS = (100, 150, 173, 174, 200)
model, tok = TX.load_model()
import torch  # noqa: E402

C.cap_gpu()
import tp_eval as E  # noqa: E402
from experiments.summarize import main as public_summarize  # noqa: E402

data = C.load_counterfact()
PT = TX.prompt_texts()
PRE = TX.pre_continuations()
rows = TX.prompts("heldout")
tids = sorted({r["text_id"] for r in rows})
batches = E.make_batches([len(PT["input_ids"][t]) for t in tids])
fac = TX.factors(NAME)
W0, _ = TX.weights(NAME)
_, W100 = TX.weights("TX-G2XL-rome-T100")
W = W0.clone().cuda()
out = {"written": C.now(), "stream": "records 0 ..", "n_heldout_rows": len(rows), "points": []}
n_applied = 0
for n in NS:
    with torch.no_grad():
        while n_applied < n:
            W += fac["left"][n_applied].cuda().unsqueeze(1) @ fac["right"][n_applied].cuda().unsqueeze(0)   # the public write
            n_applied += 1
    if n == 100:
        assert torch.equal(W.cpu(), W100), "replay does not give the T100 snapshot"
    TX.set_weight(model, W)
    res = {}
    for b in batches:
        x = torch.tensor([PT["input_ids"][tids[i]] for i in b], device="cuda")
        y = torch.stack([PRE["y0"][tids[i]] for i in b]).cuda()
        dec, f1 = E.greedy(model, x)
        tf, f2 = E.teacher_forced(model, x, y)
        for j, i in enumerate(b):
            cont = dec[j].tolist()
            res[tids[i]] = (bool((tf[j] != y[j]).any()) or not bool(f2[j]), float((tf[j] == y[j]).float().mean()),
                            bool(tf[j][0] != y[j][0]), E.degenerate_flags(cont, tok.decode(cont))["repetitive"], bool(f1[j] and f2[j]))
    v = [r for r in rows if bool(PRE["valid"][r["text_id"]])]
    pt = {"n_requests": n, "n_valid": len(v),
          "D1_tf": 100 * sum(res[r["text_id"]][0] for r in v) / len(v), "tok_loc": 100 * sum(res[r["text_id"]][1] for r in v) / len(v),
          "first_token_changed": 100 * sum(res[r["text_id"]][2] for r in v) / len(v),
          "repetitive": 100 * sum(res[r["text_id"]][3] for r in rows) / len(rows),
          "n_nonfinite": sum(not res[r["text_id"]][4] for r in rows),
          "W_minus_W0_fro": float((W.double() - W0.cuda().double()).norm()), "weight_sha256": C.sha256_tensor(W)["sha256"]}
    parent = os.path.join(C.OUT, "public_results", "ROME_seq_collapse")
    run = os.path.join(parent, f"run_n{n:04d}")
    os.makedirs(run, exist_ok=True)
    for ri in range(100):
        record = data[ri]
        post, fb = C.public_prob_test(model, tok, record)
        C.jdump({"case_id": record["case_id"], "requested_rewrite": record["requested_rewrite"], "time": 0.0, "post": post,
                 "prob_test_use_cache_false": bool(fb), "n_requests_applied": n}, os.path.join(run, f"case_{record['case_id']}.json"))
    with redirect_stdout(io.StringIO()):
        s = public_summarize(Path(parent), [f"run_n{n:04d}"], abs_path=True)[0]
    pt.update({"first100_ES": float(s["post_rewrite_success"][0]), "first100_PS": float(s["post_paraphrase_success"][0]),
               "first100_NS": float(s["post_neighborhood_success"][0]), "first100_EM": float(s["post_rewrite_diff"][0])})
    out["points"].append(pt)
    C.say(pt)
TX.set_weight(model, W0)
u = fac["requests"][173]
out["update_173"] = {"request": u["prompt"].format(u["subject"]) + " -> " + u["target_new"]["str"], "case_id": fac["case_ids"][173],
                     "update_fro": fac["scalars"][173]["update_fro"], "division_factor": fac["scalars"][173]["div_factor"]}
C.jdump(out, os.path.join(C.OUT, "collapse_point.json"))
L = ["The deployed model around the first exploding update of the reported stream (update index 173 = the 174th request:",
     f"'{out['update_173']['request']}', ||update||_F = {out['update_173']['update_fro']:.0f}, division factor {out['update_173']['division_factor']:.4f})",
     f"{'requests':>9s}{'|W-W0|_F':>10s} | {'D1 tf':>7s}{'tok-loc':>9s}{'1st tok':>9s}{'rep':>7s} | {'ES':>7s}{'PS':>7s}{'NS':>7s}{'EM':>7s}"]
for p in out["points"]:
    L.append(f"{p['n_requests']:9d}{p['W_minus_W0_fro']:10.1f} | {p['D1_tf']:7.1f}{p['tok_loc']:9.1f}{p['first_token_changed']:9.1f}{p['repetitive']:7.1f} | "
             f"{p['first100_ES']:7.1f}{p['first100_PS']:7.1f}{p['first100_NS']:7.1f}{p['first100_EM']:7.1f}")
L.append(f"left block: HELD-OUT benign prompts ({out['points'][0]['n_valid']} valid rows): D1 teacher-forced, share of pre-edit tokens kept, first token")
L.append("changed, repetitive greedy continuations.  right block: public scores of the first 100 requests (public summariser).  Percent.")
open(os.path.join(C.OUT, "collapse_point.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
