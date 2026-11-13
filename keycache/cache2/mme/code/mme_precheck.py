"""CPU dry run of the prompt construction of every MME row, per backbone (tokenizer + image processor only; NO model, no GPU).

    python -I -B mme_precheck.py <qwen25vl | llava>

Every row is built with the testbed's own Prober (tb_common.Prober.batch(item, 'image') on a model-free vendor VLM), i.e. the
function fc_capture's decode path calls for a free image query: the image is decoded and preprocessed, the prompt tokenised.
Records per row the prompt-only sequence length, the number of image tokens and the last prompt index, and compares the lengths
with those of the benign pool's image rows.  Purpose: no row can fail in the GPU run for a reason that the CPU can see, and the
question "do MME prompts exceed the usual length" is answered before the run.  -> work/precheck_<backbone>.json
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mme_common as M  # noqa: E402


def quant(v):
    import numpy as np
    v = np.asarray(v)
    return {"min": int(v.min()), "p05": int(np.quantile(v, 0.05)), "median": int(np.median(v)), "p95": int(np.quantile(v, 0.95)), "max": int(v.max()),
            "mean": round(float(v.mean()), 1)}


def main():
    bb = sys.argv[1]
    C = M.cache_code()
    TB = C.tb()
    import torch
    torch.set_num_threads(4)
    t0 = time.time()
    vlm = TB.VLM(bb, device="cpu", load_model=False)
    pr = TB.Prober(vlm)
    rows = M.jsonl_read(os.path.join(M.SELECT, f"manifest_{bb}.jsonl"))
    out, bad = [], []
    for n, m in enumerate(rows):
        try:
            b = pr.batch({"m_loc": m["image"], "m_loc_q": m["prompt"]}, "image")           # == Evaluator._query of a free image query
            ids = b["inputs"]["input_ids"][0]
            out.append({"sid": m["sid"], "len_prompt": int(b["seq_len"][0]), "n_image_tokens": int(b["n_image_tokens"][0]),
                        "lp_decode": int(b["last_prompt_idx"][0]), "last_token_id": int(ids[int(b["last_prompt_idx"][0])]),
                        "n_text_tokens": int(b["seq_len"][0]) - int(b["n_image_tokens"][0])})
        except Exception as e:                                          # reported, never hidden
            bad.append({"sid": m["sid"], "error": f"{type(e).__name__}: {e}"[:300]})
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)
    idx = [e for e in M.jsonl_read(os.path.join(M.CACHE, bb, "index.jsonl")) if e["kind"] == "benign" and e["ptype"] == "image"]
    ben = {"len_prompt": quant([e["len_prompt"] for e in idx]), "n_image_tokens": quant([e["n_image_tokens"] for e in idx]),
           "n_text_tokens": quant([e["len_prompt"] - e["n_image_tokens"] for e in idx])}
    mx = ben["len_prompt"]["max"]
    res = {"backbone": bb, "n_rows": len(rows), "n_built": len(out), "n_failed": len(bad), "failed": bad,
           "mme": {k: quant([r[k] for r in out]) for k in ("len_prompt", "n_image_tokens", "n_text_tokens")},
           "benign_pool_image_rows": ben,
           "n_mme_rows_longer_than_the_longest_benign_image_row": sum(r["len_prompt"] > mx for r in out),
           "last_prompt_token_ids": sorted({r["last_token_id"] for r in out}),
           "last_prompt_token": sorted({vlm.tok.decode([r["last_token_id"]]) for r in out}),
           "last_prompt_index_is_last_position": all(r["lp_decode"] == r["len_prompt"] - 1 for r in out),
           "seconds": round(time.time() - t0, 1), "rows": out}
    M.jdump(res, os.path.join(M.WORK, f"precheck_{bb}.json"), indent=None)
    print({k: v for k, v in res.items() if k not in ("rows",)})
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
