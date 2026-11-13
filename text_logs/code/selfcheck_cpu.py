"""Device-independent self-check (CPU, float32, 4 threads): a small seeded sample of description rows is recomputed
on the CPU, one prompt at a time, with full forwards and no key-value cache, and compared with the stored GPU results.

    tags W0, T100, T2000 (as far as described): 8 prompt rows each
        greedy 10-token continuation (10 full forwards)        vs out/describe/<tag>/dec.pt
        argmax under teacher forcing of the stored y0          vs out/describe/<tag>/dec.pt (tf_pred)
Near-ties may differ between devices; differences are counted and listed.  Writes out/selfcheck_cpu.json.
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

model, tok = TX.load_model(device="cpu")
import torch  # noqa: E402

torch.set_num_threads(4)
TAG_LOG = {"T100": "TX-G2XL-rome-T100", "T2000": "TX-G2XL-rome-T2000"}
PT = TX.prompt_texts()
PRE = TX.pre_continuations()
ROWS = TX.prompts()
W0, _ = TX.weights(TX.DEV_LOG)
DESC = os.path.join(C.OUT, "describe")
out = {"written": C.now(), "device": "cpu", "threads": torch.get_num_threads(), "tags": {}}


@torch.no_grad()
def greedy_nocache(ids, n=10):
    x = torch.tensor([ids])
    res = []
    for _ in range(n):
        nxt = model(input_ids=x, use_cache=False).logits[0, -1].argmax()
        res.append(int(nxt))
        x = torch.cat([x, nxt.view(1, 1)], 1)
    return res


@torch.no_grad()
def tf_single(ids, cont):
    x = torch.tensor([ids + cont[:-1]])
    return model(input_ids=x, use_cache=False).logits[0, len(ids) - 1:].argmax(-1).tolist()


t0 = time.time()
for tag in ["W0"] + list(TAG_LOG):
    D = os.path.join(DESC, tag)
    if not os.path.exists(os.path.join(D, "dec.pt")):
        continue
    if tag == "W0":
        W, scope = W0, [r for r in ROWS if r["set"] in ("stream", "dev", "heldout")]
    else:
        _, W = TX.weights(TAG_LOG[tag])
        src, T = TX.PLAN[TAG_LOG[tag]]
        scope = [r for r in ROWS if r["set"] == "heldout" or (r["set"] == src and r["pos"] < T)]
    scope = [r for r in scope if r["n_tok"] <= 24]                 # short prompts: CPU time
    TX.set_weight(model, W)
    d = torch.load(os.path.join(D, "dec.pt"), map_location="cpu")
    pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
    sample = random.Random(20261112).sample(scope, 8)
    n_dec = n_tf = 0
    diffs = []
    for r in sample:
        tid = r["text_id"]
        ids = PT["input_ids"][tid]
        g = greedy_nocache(ids)
        t = tf_single(ids, PRE["y0"][tid].tolist())
        a, b = g == d["dec"][pos[tid]].tolist(), t == d["tf_pred"][pos[tid]].tolist()
        n_dec += a
        n_tf += b
        if not (a and b):
            diffs.append({"prompt_id": r["prompt_id"], "dec_equal": a, "tf_equal": b})
    out["tags"][tag] = {"n_rows": len(sample), "dec_equal": int(n_dec), "tf_equal": int(n_tf), "differences": diffs,
                        "seconds": time.time() - t0}
    C.say(tag, out["tags"][tag])
    C.jdump(out, os.path.join(C.OUT, "selfcheck_cpu.json"))
TX.set_weight(model, W0)
C.jdump(out, os.path.join(C.OUT, "selfcheck_cpu.json"))
