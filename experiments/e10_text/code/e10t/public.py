"""The PUBLIC evaluation code's own functions on a served model: public scores (ii) and n-gram entropy of generations (v).

    sc = e10t.public.scores(ses, M, out_dir)             # ES / PS / NS (+ EM / PM / NM / S) by the public summariser
    ge = e10t.public.generation(ses, M, out_path)        # GE: n-gram entropy of the public generate_fast texts

(ii)  For every logged record the public experiments/py/eval_utils_counterfact.compute_rewrite_quality_counterfact(model,
      tok, record, None, None) is called on the model UNDER THE METHOD (Backbone.served: a weight, or W_0 + the library's
      hook; the hook reads the attention mask of the public code's padded batch, so PAD positions never fire).  The result
      is written as a public-format case file (key 'post') and summarised by the public experiments/summarize.py.  No
      shortcut: every record runs through the public function (through the build's wrapper tp_common.public_prob_test,
      which only repeats the same call with use_cache=False after a CUDA out-of-memory error).  Scores are probabilities
      of the new vs the true object; they are never mixed with the decode-based metric (i).
(v)   As the build (describe.py, part gen): the 10 generation prompts of each record, the public util.generate.generate_fast
      (top-5 sampling, 100 tokens), 2 records per call, sampler seeded per call with 77000 + index of the first record; the
      public n_gram_entropy of the 10 texts of a record; GE = 100 x mean over records.  With the same seeds the pre-edit and
      the deployed model reproduce the build's texts (test g_public).
Both are resumable (case files / a partial file).  A gate's firing flags are stored next to the public output (key 'e10t').
"""
from __future__ import annotations

import contextlib
import io
import os
import time
from pathlib import Path

import numpy as np
import torch

import e10t
from . import data as D

GEN_SEED, GEN_CHUNK = 77000, 2
GROUPS = (("rewrite", 1), ("paraphrase", 2), ("neighborhood", 10), ("attribute", 10))


def _summarise(parent: str) -> dict:
    from experiments.summarize import main as public_summarize
    with contextlib.redirect_stdout(io.StringIO()):
        s = public_summarize(Path(parent), None, abs_path=True)
    if len(s) != 1:
        raise RuntimeError(f"{parent}: the public summariser found {len(s)} runs")
    s = s[0]
    return {"n": int(s["num_cases"]), "ES": float(s["post_rewrite_success"][0]), "PS": float(s["post_paraphrase_success"][0]),
            "NS": float(s["post_neighborhood_success"][0]), "EM": float(s["post_rewrite_diff"][0]), "PM": float(s["post_paraphrase_diff"][0]),
            "NM": float(s["post_neighborhood_diff"][0]), "S": float(s["post_score"][0]),
            "ES_sd": float(s["post_rewrite_success"][1]), "PS_sd": float(s["post_paraphrase_success"][1]), "NS_sd": float(s["post_neighborhood_success"][1])}


def scores(ses, M, out_dir: str, first: int | None = None, tag: str | None = None, verbose: int = 0) -> dict:
    """Public scores of a method on the records of the session's log (first: only its first `first` requests, e.g. 100 for the
    retention read-out).  Case files: <out_dir>/<log>/<tag or label>/run_000/case_<id>.json.  -> summary dict (percent)."""
    L, B = ses.L, ses.B
    if B is None:
        raise RuntimeError("the public scores need the model")
    C = B.C
    data = D.counterfact()
    tag = tag or M.label
    parent = os.path.join(out_dir, L.name, tag)
    run = os.path.join(parent, "run_000")
    os.makedirs(run, exist_ok=True)
    recs = L.records if first is None else L.records[:first]
    todo = [ri for ri in recs if not os.path.exists(os.path.join(run, f"case_{data[ri]['case_id']}.json"))]
    sig = ses.sig(M)
    t0 = time.time()
    if todo:
        with B.served(ses.served_obj(M)) as srv:
            for n, ri in enumerate(todo):
                record = data[ri]
                post, fb = C.public_prob_test(B.model, B.tok, record)
                extra = {"method": M.label, "sig": sig, "log": L.name, "use_cache_false": bool(fb)}
                flags = srv.rows()
                if flags is not None:                                   # rows of the public batch: prompts in group order, (new, true) each
                    a = 0
                    for g, _ in GROUPS:
                        k = len(record[f"{g}_prompts"]) if g != "rewrite" else 1
                        fl = flags[2 * a:2 * (a + k)]
                        extra[f"{g}_rows_fired"] = int(sum(f["fired"] for f in fl))
                        extra[f"{g}_rows"] = len(fl)
                        a += k
                    extra["n_rows"] = len(flags)
                D.jdump({"case_id": record["case_id"], "requested_rewrite": record["requested_rewrite"], "time": 0.0, "post": post, "e10t": extra},
                        os.path.join(run, f"case_{record['case_id']}.json"))
                if verbose and (n + 1) % verbose == 0:
                    print(f"    public {M.label}: {n + 1}/{len(todo)} records {time.time() - t0:.0f}s", flush=True)
    want = {f"case_{data[ri]['case_id']}.json" for ri in recs}
    have = {f for f in os.listdir(run) if f.startswith("case_")}
    src = parent
    if have != want:                                                    # a subset (retention): summarise links to exactly these files
        src = os.path.join(out_dir, L.name, f"{tag}__first{len(recs)}")
        sub = os.path.join(src, "run_000")
        os.makedirs(sub, exist_ok=True)
        for f in os.listdir(sub):
            os.remove(os.path.join(sub, f))
        for f in sorted(want):
            os.symlink(os.path.join(run, f), os.path.join(sub, f))
    out = _summarise(src)
    nonfin, fired = 0, {g: [0, 0] for g, _ in GROUPS}
    for f in sorted(want):
        c = D.jload(os.path.join(run, f))
        vals = [x[k] for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs") for x in c["post"][key] for k in ("target_new", "target_true")]
        nonfin += int(not np.all(np.isfinite(vals)))
        for g, _ in GROUPS:
            if f"{g}_rows" in c.get("e10t", {}):
                fired[g][0] += c["e10t"][f"{g}_rows_fired"]
                fired[g][1] += c["e10t"][f"{g}_rows"]
        if c.get("e10t", {}).get("sig") != sig:
            raise RuntimeError(f"{run}/{f} was written for another method ({c.get('e10t', {}).get('sig')} != {sig}): use another tag or directory")
    out.update(n_records_with_nonfinite_prob=nonfin, case_files=run, seconds=time.time() - t0, n_new=len(todo),
               rows_fired_pct={g: (100.0 * a / b if b else None) for g, (a, b) in fired.items()})
    return out


def generation(ses, M, out_path: str, records=None, verbose: int = 0) -> dict:
    """n-gram entropy of the public generations of a method on the generation prompts of `records` (default: the first 100
    requests of the log).  out_path: JSON file with the texts (resumable through <out_path>.partial).  -> summary."""
    L, B = ses.L, ses.B
    if B is None:
        raise RuntimeError("generation needs the model")
    from experiments.py.eval_utils_counterfact import compute_n_gram_entropy, n_gram_entropy
    from util.generate import generate_fast
    data = D.counterfact()
    recs = list(L.records[:100] if records is None else records)
    sig = ses.sig(M)
    if os.path.exists(out_path):
        g = D.jload(out_path)
        if g["sig"] != sig or g["records"] != recs:
            raise RuntimeError(f"{out_path} holds the generations of another method or record set")
        rows = g["rows"]
    else:
        part = out_path + ".partial"
        rows = D.jload(part)["rows"] if os.path.exists(part) and D.jload(part).get("sig") == sig else []
        t0 = time.time()
        with B.served(ses.served_obj(M)) as srv:
            for c0 in range(len(rows), len(recs), GEN_CHUNK):
                chunk = recs[c0:c0 + GEN_CHUNK]
                prompts = [p for ri in chunk for p in data[ri]["generation_prompts"]]
                torch.manual_seed(GEN_SEED + chunk[0])
                try:
                    texts = generate_fast(B.model, B.tok, prompts, n_gen_per_prompt=1, max_out_len=100)
                except RuntimeError as ex:                              # e.g. the sampler refuses non-finite probabilities
                    if isinstance(ex, torch.cuda.OutOfMemoryError):
                        raise
                    rows += [{"record": ri, "case_id": data[ri]["case_id"], "failed": f"{type(ex).__name__}: {str(ex)[:300]}"} for ri in chunk]
                    D.jdump({"sig": sig, "rows": rows}, part)
                    continue
                flags = srv.rows()
                k = 0
                for ri in chunk:
                    n = len(data[ri]["generation_prompts"])
                    tx = texts[k:k + n]
                    row = {"record": ri, "case_id": data[ri]["case_id"], "ngram_entropy": n_gram_entropy(tx),
                           "ngram_entropy_per_text": [float(compute_n_gram_entropy(t)) for t in tx], "text": tx}
                    if flags is not None:
                        row.update(n_prompts_fired=int(sum(f["fired"] for f in flags[k:k + n])),
                                   n_fire_pos=[int(f["n_fire_pos"]) for f in flags[k:k + n]])
                    rows.append(row)
                    k += n
                D.jdump({"sig": sig, "rows": rows}, part)
                if verbose and (c0 // GEN_CHUNK + 1) % verbose == 0:
                    print(f"    generation {M.label}: {len(rows)}/{len(recs)} records {time.time() - t0:.0f}s", flush=True)
        D.jdump({"sig": sig, "method": M.label, "log": L.name, "records": recs, "chunk": GEN_CHUNK,
                 "seed_rule": f"torch.manual_seed({GEN_SEED} + index of the first record of the chunk of {GEN_CHUNK} records)",
                 "seconds": time.time() - t0, "rows": rows}, out_path)
        if os.path.exists(part):
            os.remove(part)
    return generation_summary(rows)


def generation_summary(rows, ref_per_text=None) -> dict:
    """GE (x100, mean over records), its standard error, per-text entropies; with the pre-edit model's per-text entropies as
    reference: the share of texts below their 1st percentile."""
    ok = [r for r in rows if "failed" not in r]
    if not ok:
        return {"n_records": 0, "n_failed": len(rows), "GE": None}
    per_rec = np.array([r["ngram_entropy"] for r in ok])
    per_text = np.array([x for r in ok for x in r["ngram_entropy_per_text"]])
    out = {"n_records": len(ok), "n_failed": len(rows) - len(ok), "n_texts": int(per_text.size), "GE": float(100 * per_rec.mean()),
           "GE_se": float(100 * per_rec.std(ddof=1) / np.sqrt(per_rec.size)) if per_rec.size > 1 else None,
           "per_text_q05": float(100 * np.quantile(per_text, 0.05)), "per_text_min": float(100 * per_text.min()), "per_text": per_text.tolist()}
    if "n_prompts_fired" in ok[0]:
        out["prompts_fired_pct"] = float(100.0 * sum(r["n_prompts_fired"] for r in ok) / per_text.size)
        out["mean_fire_positions_per_text"] = float(np.mean([x for r in ok for x in r["n_fire_pos"]]))
    if ref_per_text is not None:
        out["pct_below_ref_q01"] = float(100.0 * (per_text < np.quantile(np.asarray(ref_per_text), 0.01)).mean())
    return out
