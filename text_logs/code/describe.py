"""E10 step 10: raw measurements of one WEIGHT of the family (the pre-edit weight or the deployed weight of a log).
Only the pre-edit and the deployed model are run here; no gate, no repair, nothing is fitted.

    python describe.py --tags W0 dev100 T100 T500 T1000 T2000 [--parts dec,edit,public,gen]

Weight tags: W0 (pre-edit) | dev100 | T100 | T500 | T1000 | T2000 (deployed weight of the log).
Prompt scope of a tag: the prompts of its logged records (request, paraphrase, neighbourhood) and the HELD-OUT benign
prompts; W0 covers the union (stream 0 .. 1999, dev, HELD-OUT).

Parts (each resumable, written under out/describe/<tag>/):
  dec     per distinct prompt text: greedy 10-token continuation (decode variant) and the argmax under teacher
          forcing of the stored pre-edit continuation y0 (out/pre_continuations.pt); finite flags.   -> dec.pt
  edit    per request / paraphrase prompt of a logged record: argmax under teacher forcing of the tokens of the new
          object, tok(' ' + target_new)  (the public tokenisation).                                  -> edit_tf.pt
  public  per logged record: the public probability test compute_rewrite_quality_counterfact(model, tok, record,
          None, None) (rewrite / paraphrase / neighbourhood / attribute prompts; no generation), written as public
          case files  out/public_results/ROME_seq/run_<tag>/case_<id>.json  (key 'post' = this weight).
  gen     the public generation test without the tf-idf consistency score: for the records of a generation set,
          generate_fast(model, tok, record['generation_prompts'], n_gen_per_prompt=1, max_out_len=100) (top-5
          sampling, as in the public test_generation) and the public n_gram_entropy of the 10 texts of each record.
          The prompts of 2 records go into one call (see part_gen); the sampler is seeded per call
          (torch.manual_seed(77000 + index of the first record)), the same for every weight.          -> gen_<set>.json
          Generation sets: first100 (the first 100 requests of the log) and, for the full stream (T = 2000) only,
          last100 (its last 100 requests).
No padding in dec / edit (batches of one token length); the public parts pad as the public code does.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", required=True)
ap.add_argument("--parts", default="dec,edit,public,gen")
args = ap.parse_args()
PARTS = args.parts.split(",")
TAG_LOG = {"dev100": "TX-G2XL-rome-dev100", "T100": "TX-G2XL-rome-T100", "T500": "TX-G2XL-rome-T500",
           "T1000": "TX-G2XL-rome-T1000", "T2000": "TX-G2XL-rome-T2000"}
GEN_SEED = 77000
GEN_CHUNK = 2                 # records per call of the public generate_fast (20 prompts: about 8.8 GB; 40 prompts exceeded a 9.7 GiB cap)

C.setup_public()
import torch  # noqa: E402

cap_bytes = C.cap_gpu()
import tp_eval as E  # noqa: E402
from experiments.py.eval_utils_counterfact import compute_n_gram_entropy, n_gram_entropy  # noqa: E402
from util.generate import generate_fast  # noqa: E402

model, tok = C.load_model()
data = C.load_counterfact()
PT = TX.prompt_texts()
TEXTS, IDS = PT["texts"], PT["input_ids"]
PRE = TX.pre_continuations()
Y0 = PRE["y0"]
ROWS = TX.prompts()
W0 = torch.load(os.path.join(C.LOGS_EDIT, "W0_transformer.h.17.mlp.c_proj.weight.pt"), map_location="cpu")["weight"]
assert torch.equal(C.get_weight(model).detach().cpu(), W0), "the loaded model is not at the pre-edit weight"
assert PRE["w0"]["sha256"] == C.sha256_tensor(W0)["sha256"]


def scope_rows(tag):
    """Prompt rows in the scope of a weight tag."""
    if tag == "W0":
        return [r for r in ROWS if r["set"] in ("stream", "dev", "heldout")]
    src, T = TX.PLAN[TAG_LOG[tag]]
    return [r for r in ROWS if r["set"] == "heldout" or (r["set"] == src and r["pos"] < T)]


def gen_sets(tag):
    """{set name: [record indices]} of the generation test for a weight tag."""
    def sets_of(src, T):
        first = TX.FIRST_RECORD[src]
        out = {f"{src}_first100": list(range(first, first + 100))}
        if T == C.STREAM_T:                                   # the last 100 requests: for the full stream only (GPU time)
            out[f"{src}_last100_T{T}"] = list(range(first + T - 100, first + T))
        return out
    if tag == "W0":
        out = {}
        for name in TX.NAMES:
            out.update(sets_of(*TX.PLAN[name]))
        return out
    return sets_of(*TX.PLAN[TAG_LOG[tag]])


def part_dec(tag, D):
    path = os.path.join(D, "dec.pt")
    if os.path.exists(path):
        return
    tids = sorted({r["text_id"] for r in scope_rows(tag)})
    lengths = [len(IDS[t]) for t in tids]
    batches = E.make_batches(lengths)
    n = len(tids)
    dec = torch.zeros(n, E.N_DEC, dtype=torch.long)
    tfp = torch.zeros(n, E.N_DEC, dtype=torch.long)
    fin_d = torch.zeros(n, dtype=torch.bool)
    fin_t = torch.zeros(n, dtype=torch.bool)
    t0 = time.time()
    for bi, b in enumerate(batches):
        x = torch.tensor([IDS[tids[i]] for i in b], device="cuda")
        y = torch.stack([Y0[tids[i]] for i in b]).cuda()
        cont, f1 = E.greedy(model, x)
        pred, f2 = E.teacher_forced(model, x, y)
        idx = torch.tensor(b)
        dec[idx], tfp[idx], fin_d[idx], fin_t[idx] = cont.cpu(), pred.cpu(), f1.cpu(), f2.cpu()
        if bi % 150 == 0:
            C.say(f"  {tag} dec batch {bi}/{len(batches)} {time.time() - t0:.0f}s")
    C.tsave({"text_ids": torch.tensor(tids), "dec": dec, "tf_pred": tfp, "dec_finite": fin_d, "tf_finite": fin_t,
             "n_dec": E.N_DEC, "seconds": time.time() - t0}, path)
    C.say(f"{tag}: dec done, {n} texts, {time.time() - t0:.0f}s")


def part_edit(tag, D):
    path = os.path.join(D, "edit_tf.pt")
    if os.path.exists(path):
        return
    rows = [r for r in scope_rows(tag) if r["set"] in ("stream", "dev") and r["kind"] in ("request", "paraphrase")]
    items = []
    for r in rows:
        rw = data[r["record"]]["requested_rewrite"]
        items.append((r["prompt_id"], IDS[r["text_id"]], tok(f" {rw['target_new']['str']}")["input_ids"],
                      tok(f" {rw['target_true']['str']}")["input_ids"]))
    out = {}
    for which, col in (("new", 2), ("true", 3)):
        order = sorted(range(len(items)), key=lambda i: (len(items[i][1]), len(items[i][col]), i))
        res = {}
        i = 0
        while i < len(order):
            L, n = len(items[order[i]][1]), len(items[order[i]][col])
            j = i
            while j < len(order) and j - i < E.BATCH and len(items[order[j]][1]) == L and len(items[order[j]][col]) == n:
                j += 1
            b = order[i:j]
            x = torch.tensor([items[k][1] for k in b], device="cuda")
            y = torch.tensor([items[k][col] for k in b], device="cuda")
            pred, fin = E.teacher_forced(model, x, y)
            for k, p, f in zip(b, pred.cpu().tolist(), fin.cpu().tolist()):
                res[items[k][0]] = {"target_ids": items[k][col], "pred": p, "finite": bool(f)}
            i = j
        out[which] = res
    C.tsave(out, path)
    C.say(f"{tag}: edit teacher forcing done, {len(items)} rows")


def part_public(tag, D):
    run_dir = os.path.join(C.OUT, "public_results", "ROME_seq", f"run_{tag}")
    os.makedirs(run_dir, exist_ok=True)
    recs = sorted({r["record"] for r in scope_rows(tag) if r["set"] in ("stream", "dev")})
    t0 = time.time()
    n_new = 0
    for ri in recs:
        record = data[ri]
        p = os.path.join(run_dir, f"case_{record['case_id']}.json")
        if os.path.exists(p):
            continue
        post, fb = C.public_prob_test(model, tok, record)      # the public test; memory fallback: see tp_common
        C.jdump({"case_id": record["case_id"], "requested_rewrite": record["requested_rewrite"], "time": 0.0,
                 "weight_tag": tag, "post": post, "prob_test_use_cache_false": bool(fb)}, p)
        n_new += 1
        if n_new % 250 == 0:
            C.say(f"  {tag} public {n_new} new of {len(recs)} {time.time() - t0:.0f}s")
    C.say(f"{tag}: public probability test done, {len(recs)} records ({n_new} new), {time.time() - t0:.0f}s")


def part_gen(tag, D):
    """One call of the public generate_fast per chunk of GEN_CHUNK records (their 10 generation prompts each, in record
    order), then the public n_gram_entropy per record.  The public test_generation makes one call per record; passing
    several records' prompts in one call changes only which random numbers the sampler draws (rows of a batch do not
    interact: no pad position is ever fed), and makes the pass about three times cheaper.  Seed per chunk; the same
    chunks and seeds for every weight.  Resumable per chunk."""
    for sname, recs in gen_sets(tag).items():
        path = os.path.join(D, f"gen_{sname}.json")
        if os.path.exists(path):
            continue
        part = path + ".partial"
        out = C.jload(part)["rows"] if os.path.exists(part) else []
        t0 = time.time()
        for c0 in range(len(out), len(recs), GEN_CHUNK):
            chunk = recs[c0:c0 + GEN_CHUNK]
            prompts = [p for ri in chunk for p in data[ri]["generation_prompts"]]
            torch.manual_seed(GEN_SEED + chunk[0])
            try:
                texts = generate_fast(model, tok, prompts, n_gen_per_prompt=1, max_out_len=100)
            except RuntimeError as ex:                         # e.g. the sampler refuses non-finite probabilities
                if isinstance(ex, torch.cuda.OutOfMemoryError):
                    raise
                out += [{"record": ri, "case_id": data[ri]["case_id"], "failed": f"{type(ex).__name__}: {str(ex)[:300]}"} for ri in chunk]
                C.jdump({"rows": out}, part)
                try:                                           # a device-side assert leaves the CUDA context unusable
                    float((torch.zeros(1, device="cuda") + 1).sum())
                except RuntimeError:
                    C.say(f"{tag} gen {sname}: CUDA context lost after a failed generation call at record {chunk[0]}; "
                          "this process stops (run it again: the set resumes after the failed records)")
                    os._exit(3)
                continue
            k = 0
            for ri in chunk:
                n = len(data[ri]["generation_prompts"])
                tx = texts[k:k + n]
                k += n
                out.append({"record": ri, "case_id": data[ri]["case_id"], "ngram_entropy": n_gram_entropy(tx),
                            "ngram_entropy_per_text": [float(compute_n_gram_entropy(t)) for t in tx], "text": tx})
            C.jdump({"rows": out}, part)
            if (c0 // GEN_CHUNK) % 10 == 0:
                C.say(f"  {tag} gen {sname}: {len(out)}/{len(recs)} records {time.time() - t0:.0f}s "
                      f"peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
        C.jdump({"tag": tag, "set": sname, "records": recs, "chunk": GEN_CHUNK,
                 "seed_rule": f"torch.manual_seed({GEN_SEED} + index of the first record of the chunk of {GEN_CHUNK} records)",
                 "seconds": time.time() - t0, "rows": out}, path)
        if os.path.exists(part):
            os.remove(part)
        C.say(f"{tag}: generation set {sname} done, {time.time() - t0:.0f}s")


UC = C.use_cache_equivalence(model, tok, [data[0], data[1], data[15000]]) if "public" in PARTS else None
if UC is not None:
    C.say("public probability test, use_cache True vs False at the pre-edit weight:", UC)

for tag in args.tags:
    if tag == "W0":
        W = W0
        info = {"tag": tag, "weight_sha256": C.sha256_tensor(W0)["sha256"], "log": None}
    else:
        name = TAG_LOG[tag]
        if name not in TX.available():
            C.say(f"{tag}: log {name} is not built yet, skipped")
            continue
        _, W = TX.weights(name)
        info = {"tag": tag, "weight_sha256": C.sha256_tensor(W)["sha256"], "log": name}
    D = os.path.join(C.OUT, "describe", tag)
    os.makedirs(D, exist_ok=True)
    ipath = os.path.join(D, "info.json")
    if os.path.exists(ipath):
        old = C.jload(ipath)
        assert old["weight_sha256"] == info["weight_sha256"], f"{tag}: stored results belong to another weight"
    TX.set_weight(model, W)
    assert C.sha256_tensor(C.get_weight(model))["sha256"] == info["weight_sha256"]
    info.update({"versions": C.versions(), "started": C.now(), "n_dec": E.N_DEC, "batch": E.BATCH})
    if UC is not None:
        info["use_cache_equivalence_at_W0"] = UC
    C.jdump(info, ipath)
    C.say(f"== {tag} ({info['log']}) weight {info['weight_sha256'][:16]}")
    for part, fn in (("dec", part_dec), ("edit", part_edit), ("public", part_public), ("gen", part_gen)):
        if part in PARTS:
            fn(tag, D)
    info["finished"] = C.now()
    info["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
    C.jdump(info, ipath)
TX.set_weight(model, W0)
C.say("describe done:", args.tags, f"peak {torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB (reserved {torch.cuda.max_memory_reserved() / 2 ** 20:.0f})")
