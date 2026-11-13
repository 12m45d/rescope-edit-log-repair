"""Check 4 (GPU part): re-capture of 100 recombination keys per backbone with the checker's OWN capture code, compared bit for bit
with cache2/recomb/<backbone>/keys.pt, and the last prompt index.

    gq run --mem 10500|17000 --prio 4 --name ver_rc_<bb> -- python v4_recapture.py <qwen25vl|llava>     -> out/v4_recapture_<bb>.json

What is NOT reused from the builders: rc_capture.py / fc_capture.py (items, Evaluator.batch, Evaluator.decode_keys), the harness'
capture_inputs / vlm.capture, the truncated forward, the vision-feature memo.
What IS used: the library's loader (rescope.load of the DEVELOPMENT log of the backbone: the model object every experiment uses; the
recombination keys were captured through the handle of the long master log), and the vendor's prompt builder VLM.build_batch (the
definition of the model input of an image query, fmt 'vqa').
Own path: PIL image opened here -> VLM.build_batch(prompt only) -> the FULL forward of the HF model (use_cache False) with a
forward-PRE-hook of this file on the edited module -> its input at the last token of the sequence.  Module weight = W_0 (pre-edit); ten
rows are repeated at W_E (the key must not depend on the edited weight)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcommon as V

BB = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
torch = V.lib()
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
cap_gb = (int(os.environ["GQ_MEM"]) - 600) / 1024.0
os.environ.setdefault("RESCOPE_CACHE", V.VER + "/work/hcache")      # the library's private harness cache: a copy under verify/
os.makedirs(os.environ["RESCOPE_CACHE"], exist_ok=True)
import rescope  # noqa: E402

rows = V.jl(f"{V.CACHE2}/recomb/{BB}/rows.jsonl")
blob = torch.load(f"{V.CACHE2}/recomb/{BB}/keys.pt", map_location="cpu", mmap=True)
K, rids, lp_st = blob["K_last"], list(blob["rids"]), [int(x) for x in blob["last_prompt_idx"]]
assert rids == [r["rid"] for r in rows] and K.shape[0] == len(rows)
print("keys.pt fields:", {k: (tuple(v.shape) if torch.is_tensor(v) else (type(v).__name__ if isinstance(v, (list, dict)) else v)) for k, v in blob.items()}, flush=True)

# the checker's sample: seeded, stratified by kind, spread over the whole stream
rng = np.random.default_rng([V.SEED, 4, 0 if BB == "qwen25vl" else 1])
pick = []
for kind, n in (("X-I", N - 2 * (N // 3)), ("X-Q", N // 3), ("X-X", N // 3)):
    sub = [i for i, r in enumerate(rows) if r["kind"] == kind]
    pick += sorted(int(sub[j]) for j in rng.choice(len(sub), n, replace=False))
print(f"{BB}: {len(pick)} rows sampled of {len(rows)}", flush=True)

t0 = time.time()
H = rescope.load(V.DEV[BB], device="cuda:0", mem_gb=cap_gb)
v = H.vlm
mod = v.edit_module(H.L)
assert mod is H.mod
name = [n for n, m in v.m.named_modules() if m is mod]
print(f"loaded {H.name} in {time.time() - t0:.0f} s; edited module {name}, layer {H.L}, dtype {H.dtype}, d_in {H.d_in}", flush=True)
store = {}


def pre_hook(m, args):
    store["x"] = args[0].detach()


def my_key(r):
    """(key float32 (d,), last index, sequence length, last token id, n image tokens) of the prompt-only sequence of a row."""
    with Image.open(V.img_path(r["image"])) as im:
        img = im.convert("RGB")
    b = v.build_batch([img], [r["prompt"]], answers=None, fmt="vqa")
    ids = b["inputs"]["input_ids"]
    assert ids.shape[0] == 1 and int(b["inputs"]["attention_mask"].sum()) == ids.shape[1]
    last = ids.shape[1] - 1                                             # the last token of the prompt-only sequence
    store.clear()
    h = mod.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            v.m(**b["inputs"], use_cache=False)
    finally:
        h.remove()
    x = store["x"]
    x = x[0] if x.dim() == 3 else x
    assert x.shape[0] == ids.shape[1], (x.shape, ids.shape)
    return x[last].float().cpu(), last, int(ids.shape[1]), int(ids[0, last]), int((ids[0] == v.image_token_id).sum()), int(b["last_prompt_idx"][0])


recs = []
H.set_weight(H.W0)
assert torch.equal(mod.weight.data, H.W0)
t1 = time.time()
for n_, i in enumerate(pick):
    r = rows[i]
    k, last, sl, tok, nimg, lp_vendor = my_key(r)
    ref = K[i]
    recs.append({"row": i, "rid": r["rid"], "kind": r["kind"], "edit_pos": r["edit_pos"], "bitwise": bool(torch.equal(k, ref)),
                 "max_abs_diff": float((k - ref).abs().max()), "cos": float(torch.nn.functional.cosine_similarity(k.double(), ref.double(), dim=0)),
                 "lp_mine": last, "lp_stored": lp_st[i], "lp_vendor": lp_vendor, "seq_len": sl, "last_token": v.tok.decode([tok]), "last_token_id": tok,
                 "n_image_tokens": nimg})
    if (n_ + 1) % 20 == 0:
        print(f"  {n_ + 1}/{len(pick)} rows, {time.time() - t1:.0f} s, bitwise so far {sum(x['bitwise'] for x in recs)}", flush=True)
# ten rows again at the deployed weight W_E: the module input must not depend on the module's weight
H.set_weight(H.WE)
at_we = []
for i in pick[::10]:
    k, *_ = my_key(rows[i])
    at_we.append(bool(torch.equal(k, K[i])))
H.set_weight(H.W0)
# the stored last-prompt index of ALL rows against the tokenizer alone (no forward): length of the prompt-only id sequence - 1
tok_chk = {"n": 0, "equal": 0}
if BB == "llava":                                                       # LLaVA: the image slots are a constant; every row can be recounted from the tokenizer
    for i, r in enumerate(rows):
        e = v.encode_row(r["prompt"], None, "vqa", v.n_llava_img)
        tok_chk["n"] += 1
        tok_chk["equal"] += int(len(e["ids"]) - 1 == lp_st[i])
else:                                                                   # Qwen: the image slots depend on the image size (header read only)
    for i, r in enumerate(rows):
        e = v.encode_row(r["prompt"], None, "vqa", v.estimate_image_tokens(r["image"]))
        tok_chk["n"] += 1
        tok_chk["equal"] += int(len(e["ids"]) - 1 == lp_st[i])
out = {"backbone": BB, "log_handle": H.name, "edited_module": name, "n": len(recs), "n_bitwise": sum(r["bitwise"] for r in recs),
       "by_kind": {k: [sum(r["bitwise"] for r in recs if r["kind"] == k), sum(r["kind"] == k for r in recs)] for k in ("X-I", "X-Q", "X-X")},
       "max_abs_diff": max(r["max_abs_diff"] for r in recs), "min_cos": min(r["cos"] for r in recs),
       "lp_equal_stored": sum(r["lp_mine"] == r["lp_stored"] for r in recs), "lp_equal_vendor_last_prompt_idx": sum(r["lp_mine"] == r["lp_vendor"] for r in recs),
       "last_tokens": sorted({(r["last_token_id"], r["last_token"]) for r in recs}),
       "at_deployed_weight_bitwise": [sum(at_we), len(at_we)], "last_prompt_index_all_rows_tokenizer_only": tok_chk,
       "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "seconds": round(time.time() - t0, 1),
       "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "rows": recs,
       "ok": all(r["bitwise"] and r["lp_mine"] == r["lp_stored"] for r in recs) and all(at_we) and tok_chk["n"] == tok_chk["equal"]}
V.jdump(out, f"{V.OUT}/v4_recapture_{BB}.json")
print({k: v_ for k, v_ in out.items() if k != "rows"}, flush=True)
