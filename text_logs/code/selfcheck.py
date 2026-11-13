"""Self-check of the description's own measurements (not the independent audit): a seeded sample of rows is recomputed
ONE PROMPT AT A TIME with full forwards and no key-value cache, and compared with the stored batched results.

For each available weight tag (W0, dev100, T100, ...): 40 prompt rows of its scope (seeded) ->
    greedy 10-token continuation by 10 full forwards (no cache)   vs  dec.pt
    argmax under teacher forcing of y0 (one forward)              vs  dec.pt (tf_pred)
and 20 request rows -> argmax under teacher forcing of the new object vs edit_tf.pt.
Differences can only come from floating-point effects of batching / caching at near-ties; they are counted.
Writes out/selfcheck.json.
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

model, tok = TX.load_model()
import torch  # noqa: E402

C.cap_gpu()
TAG_LOG = {"dev100": "TX-G2XL-rome-dev100", "T100": "TX-G2XL-rome-T100", "T500": "TX-G2XL-rome-T500",
           "T1000": "TX-G2XL-rome-T1000", "T2000": "TX-G2XL-rome-T2000"}
PT = TX.prompt_texts()
PRE = TX.pre_continuations()
ROWS = TX.prompts()
data = C.load_counterfact()
W0, _ = TX.weights(TX.DEV_LOG)
DESC = os.path.join(C.OUT, "describe")
out = {"written": C.now(), "tags": {}}


@torch.no_grad()
def greedy_nocache(ids, n=10):
    x = torch.tensor([ids], device="cuda")
    res = []
    for _ in range(n):
        nxt = model(input_ids=x, use_cache=False).logits[0, -1].argmax()
        res.append(int(nxt))
        x = torch.cat([x, nxt.view(1, 1)], 1)
    return res


@torch.no_grad()
def tf_single(ids, cont):
    x = torch.tensor([ids + cont[:-1]], device="cuda")
    return model(input_ids=x, use_cache=False).logits[0, len(ids) - 1:].argmax(-1).tolist()


for tag in ["W0"] + list(TAG_LOG):
    D = os.path.join(DESC, tag)
    if not (os.path.exists(os.path.join(D, "dec.pt")) and os.path.exists(os.path.join(D, "edit_tf.pt"))):
        continue
    if tag == "W0":
        W, scope = W0, [r for r in ROWS if r["set"] in ("stream", "dev", "heldout")]
    else:
        _, W = TX.weights(TAG_LOG[tag])
        src, T = TX.PLAN[TAG_LOG[tag]]
        scope = [r for r in ROWS if r["set"] == "heldout" or (r["set"] == src and r["pos"] < T)]
    TX.set_weight(model, W)
    d = torch.load(os.path.join(D, "dec.pt"), map_location="cpu")
    pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
    et = torch.load(os.path.join(D, "edit_tf.pt"), map_location="cpu")
    rng = random.Random(20261111)
    sample = rng.sample(scope, 40)
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
    req = rng.sample([r for r in scope if r["set"] in ("stream", "dev") and r["kind"] in ("request", "paraphrase")], 20)
    n_et = 0
    for r in req:
        e = et["new"][r["prompt_id"]]
        tgt = tok(f" {data[r['record']]['requested_rewrite']['target_new']['str']}")["input_ids"]
        assert tgt == e["target_ids"]
        n_et += tf_single(PT["input_ids"][r["text_id"]], tgt) == e["pred"]
    out["tags"][tag] = {"n_rows": len(sample), "dec_equal": int(n_dec), "tf_equal": int(n_tf), "n_edit_rows": len(req),
                        "edit_tf_equal": int(n_et), "differences": diffs}
    C.say(tag, out["tags"][tag])
TX.set_weight(model, W0)
C.jdump(out, os.path.join(C.OUT, "selfcheck.json"))
