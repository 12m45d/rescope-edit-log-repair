"""E10 step 9 (second half): the pre-edit greedy continuation of every prompt, stored once.

For every distinct prompt text of out/prompts.jsonl (benign prompts of the REPAIR and HELD-OUT pools; request,
paraphrase and neighbourhood prompts of the logged records) with the PRE-EDIT model:
    y0        greedy continuation of 10 tokens (token ids and text)
    pre_tf    argmax under teacher forcing of y0 (one full forward): y0 is VALID iff pre_tf == y0 (the decode uses the
              key-value cache, teacher forcing one full forward: a near-tie can differ; such rows are counted and
              excluded from teacher-forced D1)
    flags     empty / repetitive / contains <|endoftext|>, non-finite logits
Output: out/pre_continuations.pt (tensors, indexed by text_id) and out/pre_continuations.jsonl (readable).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

C.setup_public()
import torch  # noqa: E402

cap_bytes = C.cap_gpu()
import tp_eval as E  # noqa: E402

model, tok = C.load_model()
W = C.get_weight(model)
w0_hash = C.sha256_tensor(W)
pt = C.jload(os.path.join(C.OUT, "prompt_texts.json"))
texts, ids = pt["texts"], pt["input_ids"]
n = len(texts)
lengths = [len(x) for x in ids]
batches = E.make_batches(lengths)
C.say(f"{n} distinct prompts, {len(batches)} batches, weight {w0_hash['sha256'][:16]}")

y0 = torch.zeros(n, E.N_DEC, dtype=torch.long)
pre_tf = torch.zeros(n, E.N_DEC, dtype=torch.long)
finite = torch.zeros(n, dtype=torch.bool)
done = torch.zeros(len(batches), dtype=torch.bool)
part = os.path.join(C.OUT, "pre_continuations.partial.pt")
sig = {"n": n, "n_batches": len(batches), "batch": E.BATCH, "max_tokens": E.MAX_TOKENS, "w0": w0_hash["sha256"]}
if os.path.exists(part):                                    # resume: same texts, same batches, same weight
    st = torch.load(part, map_location="cpu")
    if st["sig"] == sig:
        y0, pre_tf, finite, done = st["y0"], st["pre_tf"], st["finite"], st["done"]
        C.say(f"resumed: {int(done.sum())} of {len(batches)} batches done")
t0 = time.time()
for bi, b in enumerate(batches):
    if bool(done[bi]):
        continue
    x = torch.tensor([ids[i] for i in b], device="cuda")
    cont, f1 = E.greedy(model, x)
    pred, f2 = E.teacher_forced(model, x, cont)
    idx = torch.tensor(b)
    y0[idx], pre_tf[idx], finite[idx] = cont.cpu(), pred.cpu(), (f1 & f2).cpu()
    done[bi] = True
    if bi % 100 == 0:
        C.say(f"batch {bi}/{len(batches)} len {lengths[b[0]]} x {len(b)} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
        C.tsave({"sig": sig, "y0": y0, "pre_tf": pre_tf, "finite": finite, "done": done}, part)
assert bool(done.all())

valid = (y0 == pre_tf).all(1) & finite
eos = tok.eos_token_id
rows = []
for i in range(n):
    t = y0[i].tolist()
    txt = tok.decode(t)
    fl = E.degenerate_flags(t, txt)
    rows.append({"text_id": i, "text": texts[i], "n_tok": lengths[i], "y0_ids": t, "y0_text": txt,
                 "valid": bool(valid[i]), "finite": bool(finite[i]), "has_eos": eos in t,
                 "pre_tf_token_acc": float((y0[i] == pre_tf[i]).float().mean()), **fl})
with open(os.path.join(C.OUT, "pre_continuations.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
C.tsave({"y0": y0, "pre_tf": pre_tf, "valid": valid, "finite": finite, "n_dec": E.N_DEC, "w0": w0_hash,
         "batch": E.BATCH, "max_tokens": E.MAX_TOKENS}, os.path.join(C.OUT, "pre_continuations.pt"))
summ = {"n_texts": n, "n_dec": E.N_DEC, "w0": w0_hash, "n_valid": int(valid.sum()), "n_invalid": int((~valid).sum()),
        "n_nonfinite": int((~finite).sum()), "n_empty": sum(r["empty"] for r in rows),
        "n_repetitive": sum(r["repetitive"] for r in rows), "n_few_distinct": sum(r["few_distinct"] for r in rows),
        "n_long_run": sum(r["long_run"] for r in rows), "n_has_eos": sum(r["has_eos"] for r in rows),
        "seconds": time.time() - t0, "versions": C.versions(), "written": C.now(),
        "peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2 ** 20}
C.jdump(summ, os.path.join(C.OUT, "pre_continuations_summary.json"))
if os.path.exists(part):
    os.remove(part)
C.say(json.dumps(summ))
