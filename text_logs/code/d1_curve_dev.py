"""Check behind the D1 numbers -- DEVELOPMENT LOG ONLY (TX-G2XL-rome-dev100): D1 on the HELD-OUT benign prompts as a
function of the number of updates applied.

The description finds that after 100 sequential public ROME updates about 95 % of the pre-edit 10-token continuations of
unrelated prompts are no longer reproduced.  Before that number is used, this script measures the same quantity for the
deployed PREFIXES of the development log (weights after n = 0, 1, 2, 5, 10, 20, 50, 100 updates, rebuilt by the exact
float32 replay of the logged factors): if the measurement is sound, D1 is 0 at n = 0, small after one update and grows
with n.  Only deployed weights are evaluated; no gate, no repair.  Same code path as describe.py (tp_eval).
Writes out/d1_curve_dev.json and out/d1_curve_dev.txt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

NAME = TX.DEV_LOG
assert NAME == "TX-G2XL-rome-dev100"
model, tok = TX.load_model()
import torch  # noqa: E402

C.cap_gpu()
import tp_eval as E  # noqa: E402

NS = (0, 1, 2, 5, 10, 20, 50, 100)
PT = TX.prompt_texts()
PRE = TX.pre_continuations()
rows = TX.benign("heldout", NAME)
tids = sorted({r["text_id"] for r in rows})
lengths = [len(PT["input_ids"][t]) for t in tids]
batches = E.make_batches(lengths)
fac = TX.factors(NAME)
W0, WE = TX.weights(NAME)
W = W0.clone().cuda()
out = {"log": NAME, "written": C.now(), "n_prompt_rows": len(rows), "n_distinct_texts": len(tids), "points": []}
n_applied = 0
for n in NS:
    with torch.no_grad():
        while n_applied < n:
            W += fac["left"][n_applied].cuda().unsqueeze(1) @ fac["right"][n_applied].cuda().unsqueeze(0)   # the public write
            n_applied += 1
    if n == 100:
        assert torch.equal(W.cpu(), WE), "replay does not give the deployed weight"
    TX.set_weight(model, W)
    res = {}
    for b in batches:
        x = torch.tensor([PT["input_ids"][tids[i]] for i in b], device="cuda")
        y = torch.stack([PRE["y0"][tids[i]] for i in b]).cuda()
        dec, f1 = E.greedy(model, x)
        tf, f2 = E.teacher_forced(model, x, y)
        for j, i in enumerate(b):
            res[tids[i]] = (bool((tf[j] != y[j]).any()) or not bool(f2[j]), bool((dec[j] != y[j]).any()) or not bool(f1[j]),
                            float((tf[j] == y[j]).float().mean()), int((tf[j] != y[j]).float().argmax()) if bool((tf[j] != y[j]).any()) else E.N_DEC)
    v = [r for r in rows if bool(PRE["valid"][r["text_id"]])]
    d1tf = 100 * sum(res[r["text_id"]][0] for r in v) / len(v)
    d1dec = 100 * sum(res[r["text_id"]][1] for r in v) / len(v)
    tl = 100 * sum(res[r["text_id"]][2] for r in v) / len(v)
    first = 100 * sum(res[r["text_id"]][3] == 0 for r in v) / len(v)
    out["points"].append({"n_updates": n, "n_valid": len(v), "D1_tf": d1tf, "D1_dec": d1dec, "tok_loc": tl,
                          "first_token_changed": first, "W_minus_W0_fro": float((W.double() - W0.cuda().double()).norm())})
    C.say(out["points"][-1])
TX.set_weight(model, W0)
C.jdump(out, os.path.join(C.OUT, "d1_curve_dev.json"))
L = [f"D1 on the HELD-OUT benign prompts along the DEVELOPMENT log {NAME} (deployed prefixes; {out['points'][0]['n_valid']} valid rows)",
     f"{'updates':>8s}{'D1 tf':>9s}{'D1 dec':>9s}{'tok-loc':>9s}{'1st tok':>9s}{'|W-W0|_F':>10s}"]
for p in out["points"]:
    L.append(f"{p['n_updates']:8d}{p['D1_tf']:9.1f}{p['D1_dec']:9.1f}{p['tok_loc']:9.1f}{p['first_token_changed']:9.1f}{p['W_minus_W0_fro']:10.1f}")
L.append("D1 tf / dec: the pre-edit 10-token continuation is not reproduced (teacher-forced / decode); tok-loc: share of its tokens kept;")
L.append("1st tok: share of prompts whose FIRST continuation token changes.  All in percent.")
open(os.path.join(C.OUT, "d1_curve_dev.txt"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
