"""Check 2, GPU part: the cached keys re-captured with my own hook (own model instance at the checkpoint weight, no rescope).

  gq run --mem <MiB> -- python c2_gpu.py qwen25vl|llava        -> out/c2_gpu_<backbone>.json

  A  60 seeded sequences per backbone, every kind (and both families of Qwen): teacher-forced key at lp_tf, the key at the
     decode's token (shifted sequences), the prompt-only key, the 11 keys of prompt + k filler ids (k = 0..10): bit for bit
     against keys_tf.pt / keys_decode.pt / keys_decode_steps.pt; lp_tf / lp_decode / n_image_tokens against the index; the
     pre-edit teacher-forced outcome against pre_exact / pre_token_acc / pre_n_tok.
  B  LLaVA: ALL shifted sequences (lp_shift = 1), the same comparisons (key at lp_tf and at lp_tf - 1).
  C  the 30 sampled hard negatives (c2_cpu.py): the pre-edit answer re-decoded with my own decoder v2 (y0_text, 48-token text),
     its validity (teacher forcing at W_0), and the keys of the REAL decode (every full forward that contains the prompt's
     last token) against rows_decode_steps.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import vcommon as V
import vgpu as G

import torch  # noqa: E402

SEED = 20261111
PLAN = {"qwen25vl": {("Q", "rel"): 4, ("Q", "tgen"): 4, ("Q", "igen"): 4, ("Q", "probe"): 6, ("Q", "hn_i"): 5, ("Q", "hn_q"): 5, ("Q", "hn_x"): 5,
                     ("Q", "benign"): 8, ("QK", "rel"): 4, ("QK", "tgen"): 4, ("QK", "igen"): 4, ("QK", "probe"): 7},
        "llava": {("L", "rel"): 7, ("L", "tgen"): 7, ("L", "igen"): 7, ("L", "probe"): 8, ("L", "hn_i"): 7, ("L", "hn_q"): 7, ("L", "hn_x"): 7, ("L", "benign"): 10}}
LAYER = {"qwen25vl": 20, "llava": 16}
MEM = {"qwen25vl": 10500, "llava": 17000}


def main(bb):
    path = f"{V.OUT}/c2_gpu_{bb}.json"
    if os.path.exists(path):
        V.say(f"skip {path}")
        return
    t0 = time.time()
    mem, cap = G.cap_gpu(MEM[bb])
    idx = V.read_index(bb)
    Ktf, Kdec, Kst = V.read_keys(bb, "tf"), V.read_keys(bb, "decode"), V.read_keys(bb, "decode_steps")
    vlm = G.own_vlm(bb)
    mod = vlm.edit_module(LAYER[bb])
    Wc, wkey, wfile = G.checkpoint_weight(bb, LAYER[bb])
    res = {"backbone": bb, "gq_mem": mem, "cap_gb": cap, "gpu": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "module": vlm.edit_module_name(LAYER[bb]), "module_dtype": str(mod.weight.dtype),
           "module_weight_equals_checkpoint_file": bool(torch.equal(mod.weight.data.cpu(), Wc.to(mod.weight.dtype))), "checkpoint_key": wkey, "checkpoint_file": wfile}
    V.say(f"{bb}: model loaded ({time.time() - t0:.0f}s), {res}")
    rng = np.random.default_rng([SEED, 2])
    pick = []
    for (fam, kind), k in PLAN[bb].items():
        sub = [j for j, e in enumerate(idx) if e["family"] == fam and e["kind"] == kind]
        if kind == "benign":                                              # half with an answer (teacher-forced key), half without
            a = [j for j in sub if idx[j]["row_tf"] is not None]
            b = [j for j in sub if idx[j]["row_tf"] is None]
            pick += [a[int(i)] for i in sorted(rng.choice(len(a), k // 2, replace=False))] + [b[int(i)] for i in sorted(rng.choice(len(b), k - k // 2, replace=False))]
        else:
            pick += [sub[int(i)] for i in sorted(rng.choice(len(sub), k, replace=False))]
    shifted = [j for j, e in enumerate(idx) if e.get("lp_shift")]

    def recapture(j):
        e = idx[j]
        q = {"fmt": e["fmt"], "prompt": e["prompt"], "image": e["image"]}
        r = {"sid": e["sid"], "kind": e["kind"], "family": e["family"], "lp_shift": e.get("lp_shift")}
        with torch.no_grad(), G.KeyTap(mod, stop=True) as tap:
            bp = G.build(vlm, q)
            lq = int(bp["last_prompt_idx"][0])
            r["lp_decode_ok"] = lq == e["lp_decode"] and int(bp["seq_len"][0]) == e["len_prompt"] and int(bp["n_image_tokens"][0]) == (e["n_image_tokens"] or 0)
            fill = int(bp["inputs"]["input_ids"][0, lq])
            ks = [G.run_truncated(vlm, G.extend(bp["inputs"], [fill] * k), tap)[0, lq].float().cpu() for k in range(11)]
            r["decode_bitwise"] = bool(torch.equal(ks[0], Kdec[e["row_decode"]]))
            st = [bool(torch.equal(ks[k], Kst[e["rows_decode_steps"][k]])) for k in range(11)]
            r["steps_bitwise"] = all(st)
            r["n_steps_bitwise"] = sum(st)
            r["n_distinct_step_keys"] = len({tuple(k.tolist()[:64]) for k in ks})
            if e["row_tf"] is not None:
                bt = G.build(vlm, q, e["answer"])
                lp = int(bt["last_prompt_idx"][0])
                r["lp_tf_ok"] = lp == e["lp_tf"] and int(bt["seq_len"][0]) == e["len_tf"] and lp - lq == e["lp_shift"]
                x = G.run_truncated(vlm, bt["inputs"], tap)
                kt = x[0, lp].float().cpu()
                r["tf_bitwise"] = bool(torch.equal(kt, Ktf[e["row_tf"]]))
                if not r["tf_bitwise"]:
                    r["tf_max_abs_diff"] = float((kt - Ktf[e["row_tf"]]).abs().max())
                if e["lp_shift"]:
                    r["tf_prompt_bitwise"] = bool(torch.equal(x[0, lp - e["lp_shift"]].float().cpu(), Ktf[e["row_tf_prompt"]]))
                    r["token_at_lp_tf"] = vlm.tok.convert_ids_to_tokens([int(bt["inputs"]["input_ids"][0, lp])])[0]
                    r["token_at_lp_decode"] = vlm.tok.convert_ids_to_tokens([int(bt["inputs"]["input_ids"][0, lp - 1])])[0]
                    r["tf_key_cos_decode_token_key"] = float(torch.nn.functional.cosine_similarity(kt.double(), x[0, lp - 1].double().cpu(), dim=0))
        if e["row_tf"] is not None:                                        # pre-edit teacher-forced outcome (full forward at W_0)
            pred, ans, fin = G.tf_forward(vlm, bt)
            s = G.score(pred, ans, fin)
            mism = None if (s["exact"] or not fin) else int((pred != ans).nonzero()[0, 0])
            r["pre_flags_equal"] = (s["exact"], s["n_tok"], fin, mism) == (e["pre_exact"], e["pre_n_tok"], e["pre_finite"], e["pre_first_mismatch"]) and \
                abs(s["token_acc"] - e["pre_token_acc"]) < 1e-9
        r["ok"] = all(v for k, v in r.items() if isinstance(v, bool))
        return r

    def summ(rows):
        out = {"n": len(rows), "n_ok": sum(r["ok"] for r in rows)}
        for k in ("decode_bitwise", "steps_bitwise", "tf_bitwise", "tf_prompt_bitwise", "lp_decode_ok", "lp_tf_ok", "pre_flags_equal"):
            sub = [r[k] for r in rows if k in r]
            out[k] = [sum(sub), len(sub)]
        out["by_kind"] = {}
        for r in rows:
            c = out["by_kind"].setdefault(f"{r['family']}|{r['kind']}", [0, 0])
            c[0] += int(r["ok"])
            c[1] += 1
        out["not_ok"] = [r for r in rows if not r["ok"]][:10]
        return out

    rows = [recapture(j) for j in pick]
    res["A_sample"] = dict(summ(rows), rows=rows)
    V.say(f"A: { {k: v for k, v in res['A_sample'].items() if k != 'rows'} } ({time.time() - t0:.0f}s)")
    if shifted:
        rows = [recapture(j) for j in shifted]
        res["B_shifted"] = dict(summ(rows), rows=rows, tokens_at_lp_tf=sorted({r.get("token_at_lp_tf") for r in rows}),
                                tokens_at_lp_decode=sorted({r.get("token_at_lp_decode") for r in rows}),
                                tf_key_cos_decode_token_key_min_max=[min(r["tf_key_cos_decode_token_key"] for r in rows), max(r["tf_key_cos_decode_token_key"] for r in rows)])
        V.say(f"B: { {k: v for k, v in res['B_shifted'].items() if k != 'rows'} } ({time.time() - t0:.0f}s)")
    # ------------------------------------------------------------------ C: hard negatives, pre-edit answer reproduced
    hn = V.jload(f"{V.OUT}/c2_hn_sample_{bb}.json")
    by_sid = {e["sid"]: e for e in idx}
    out = []
    for r in hn:
        q = {"fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"]}
        e = by_sid[r["hn_id"]]
        with torch.no_grad():
            bp = G.build(vlm, q)
            lq = int(bp["last_prompt_idx"][0])
            with G.KeyTap(mod) as tap:
                d = G.my_decode(vlm, bp)
            full = [x[0, lq].float().cpu() for x in tap.x if x.shape[1] > lq]
            real = [bool(torch.equal(k, Kst[e["rows_decode_steps"][i]])) for i, k in enumerate(full[:11])]
            c = {"hn_id": r["hn_id"], "y0_equal": d["y"] == r["y0_text"], "long_equal": d["long"] == r["long_text"], "n_full_forwards": len(full),
                 "real_decode_keys_bitwise": all(real), "n_real_decode_keys_bitwise": sum(real), "y0_mine": d["y"], "y0_file": r["y0_text"]}
            valid = False
            if d["y"]:
                try:
                    pred, ans, fin = G.tf_forward(vlm, G.build(vlm, q, d["y"]))
                    valid = bool(fin and (pred == ans).all())
                except (RuntimeError, ValueError, AssertionError):
                    valid = False
            c["valid_mine"], c["valid_file"] = valid, bool(r["y0_valid"])
            c["ok"] = c["y0_equal"] and c["long_equal"] and c["valid_mine"] == c["valid_file"] and c["real_decode_keys_bitwise"]
        out.append(c)
    res["C_hard_negatives"] = {"n": len(out), "n_ok": sum(c["ok"] for c in out), "y0_equal": sum(c["y0_equal"] for c in out), "long_equal": sum(c["long_equal"] for c in out),
                               "validity_equal": sum(c["valid_mine"] == c["valid_file"] for c in out), "n_valid_file": sum(c["valid_file"] for c in out),
                               "real_decode_keys_bitwise": sum(c["real_decode_keys_bitwise"] for c in out),
                               "n_real_decode_forwards": sum(c["n_full_forwards"] for c in out), "rows": out}
    V.say(f"C: { {k: v for k, v in res['C_hard_negatives'].items() if k != 'rows'} }")
    res["seconds"] = time.time() - t0
    res["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    V.jdump(res, path)
    V.say(f"wrote {path} ({res['seconds']:.0f}s, peak {res['peak_gpu_gb']:.2f} GB)")


if __name__ == "__main__":
    main(sys.argv[1])
