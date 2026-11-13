"""E10 step 10: aggregate the raw measurements (describe.py) into out/text_logs.json and render
out/text_logs_table.txt.  CPU only; every number of the table comes from this script.

Rows: for each log the DEPLOYED model (the log's snapshot weight) and the PRE-EDIT model, on the same prompts.
  (i)   new object on the request / paraphrase prompts of the logged records.
          dec  exact: the greedy 10-token continuation starts with the tokens of the new object (tok(' ' + object));
               tok:   share of the object's tokens that the free-running decode has at the same position
          tf   exact / tok: the same under teacher forcing of the object's tokens
  (ii)  public scores, by the public experiments/summarize.py on public-format case files (post = deployed weight,
        pre = pre-edit weight): ES / PS / NS (success), EM / PM / NM (magnitude), S (harmonic mean).  Never mixed
        with (i): (ii) compares the probabilities of the new and the true object, (i) asks for the argmax.
  (iii) D1 = the pre-edit greedy 10-token continuation y0 is not reproduced, on rows with a valid y0
        (valid: the pre-edit model reproduces y0 under teacher forcing).
          tf   some token of y0 is no longer the argmax under teacher forcing of y0 (or a logit is not finite)
          dec  the greedy decode differs from y0
        on the HELD-OUT benign prompts (all / clean / near: sharing a subject string, mentioning a logged subject, or
        equal to a logged prompt) and on the neighbourhood prompts of the logged records (all / clean: not equal to a
        logged request and not mentioning another logged subject).
  (iv)  degeneration: the public n-gram entropy (x100, mean over records of the public n_gram_entropy of the 10
        generations of a record), share of empty / repetitive 10-token greedy continuations (repetitive: at most 3
        distinct tokens, or one token 4 times in a row), weight and update norms, exploding updates.
  (v)   retention: (i) and (ii) on the first 100 requests of the stream at T = 100, 500, 1000, 2000.
"""
import glob
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402
import tx_public as TX  # noqa: E402

C.setup_public()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from experiments.summarize import main as public_summarize  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import tp_eval as E  # noqa: E402

TAG = {"TX-G2XL-rome-dev100": "dev100", "TX-G2XL-rome-T100": "T100", "TX-G2XL-rome-T500": "T500",
       "TX-G2XL-rome-T1000": "T1000", "TX-G2XL-rome-T2000": "T2000"}
RUNNAME = {"dev100": "run_dev100", "T100": "run_T0100", "T500": "run_T0500", "T1000": "run_T1000", "T2000": "run_T2000"}
DESC = os.path.join(C.OUT, "describe")
ROWDIR = os.path.join(C.OUT, "text_logs_rows")
os.makedirs(ROWDIR, exist_ok=True)
tok = AutoTokenizer.from_pretrained(C.MODEL_DIR)
data = C.load_counterfact()
PRE = TX.pre_continuations()
Y0, VALID = PRE["y0"], PRE["valid"]


def pct(x):
    x = np.asarray(x, dtype=np.float64)
    return float(100 * x.mean()) if x.size else None


class Dec:
    """dec.pt of a weight tag, indexed by text_id."""

    def __init__(self, tag):
        d = torch.load(os.path.join(DESC, tag, "dec.pt"), map_location="cpu")
        self.pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
        self.dec, self.tf, self.fd, self.ft = d["dec"], d["tf_pred"], d["dec_finite"], d["tf_finite"]

    def row(self, tid):
        i = self.pos[tid]
        return self.dec[i], self.tf[i], bool(self.fd[i]), bool(self.ft[i])


def have(tag):
    need = ["dec.pt", "edit_tf.pt"]
    return all(os.path.exists(os.path.join(DESC, tag, f)) for f in need)


def edit_rows(name, tag, dec, etf):
    """Per request / paraphrase row of the log: match of the new (and of the true) object."""
    out = []
    for r in TX.log_prompts(name):
        if r["kind"] not in ("request", "paraphrase"):
            continue
        d, _, fd, _ = dec.row(r["text_id"])
        row = {"prompt_id": r["prompt_id"], "pos": r["pos"], "kind": r["kind"], "k": r["k"], "case_id": r["case_id"]}
        for which in ("new", "true"):
            e = etf[which][r["prompt_id"]]
            tgt, pred = e["target_ids"], e["pred"]
            n = len(tgt)
            m = min(n, E.N_DEC)
            dm = [int(d[j]) == tgt[j] for j in range(m)]
            tm = [p == t for p, t in zip(pred, tgt)]
            row.update({f"{which}_n_tok": n,
                        f"{which}_dec_exact": bool(all(dm)) and fd and n <= E.N_DEC, f"{which}_dec_tok": float(np.mean(dm)) if fd else 0.0,
                        f"{which}_tf_exact": bool(all(tm)) and e["finite"], f"{which}_tf_tok": float(np.mean(tm)) if e["finite"] else 0.0,
                        f"{which}_finite": bool(e["finite"] and fd)})
        out.append(row)
    return out


def edit_summary(rows, first=None):
    out = {}
    for kind in ("request", "paraphrase"):
        rr = [r for r in rows if r["kind"] == kind and (first is None or r["pos"] < first)]
        out[kind] = {"n": len(rr), "dec_exact": pct([r["new_dec_exact"] for r in rr]), "dec_tok": pct([r["new_dec_tok"] for r in rr]),
                     "tf_exact": pct([r["new_tf_exact"] for r in rr]), "tf_tok": pct([r["new_tf_tok"] for r in rr]),
                     "true_dec_exact": pct([r["true_dec_exact"] for r in rr]), "true_tf_exact": pct([r["true_tf_exact"] for r in rr]),
                     "n_nonfinite": int(sum(not r["new_finite"] for r in rr)),
                     "pct_multi_token_objects": pct([r["new_n_tok"] > 1 for r in rr]),
                     "max_object_tokens": int(max(r["new_n_tok"] for r in rr)) if rr else None,
                     "n_object_longer_than_decode": int(sum(r["new_n_tok"] > E.N_DEC for r in rr))}
    return out


def probe_rows(rows, dec):
    """Per prompt row: D1 flags against the stored pre-edit continuation."""
    out = []
    for r in rows:
        tid = r["text_id"]
        d, t, fd, ft = dec.row(tid)
        y = Y0[tid]
        cont = d.tolist()
        fl = E.degenerate_flags(cont, tok.decode(cont))
        out.append({"prompt_id": r["prompt_id"], "text_id": tid, "set": r["set"], "kind": r["kind"], "pos": r["pos"],
                    "valid": bool(VALID[tid]), "D1_tf": (not ft) or bool((t != y).any()), "D1_dec": (not fd) or bool((d != y).any()),
                    "tok_loc": float((t == y).float().mean()) if ft else 0.0, "finite": bool(fd and ft),
                    "empty": fl["empty"], "repetitive": fl["repetitive"], "few_distinct": fl["few_distinct"], "long_run": fl["long_run"],
                    "n_distinct": fl["n_distinct"],
                    **{k: r[k] for k in ("near_subject", "near_mention", "near_text", "clean") if k in r}})
    return out


def probe_summary(rows):
    v = [r for r in rows if r["valid"]]
    return {"n_rows": len(rows), "n_valid": len(v), "n_distinct_texts_valid": len({r["text_id"] for r in v}),
            "D1_tf": pct([r["D1_tf"] for r in v]), "D1_dec": pct([r["D1_dec"] for r in v]),
            "tok_loc": pct([r["tok_loc"] for r in v]), "n_nonfinite": int(sum(not r["finite"] for r in rows)),
            "n_D1_tf_differs_from_D1_dec": int(sum(r["D1_tf"] != r["D1_dec"] for r in v)),
            "empty": pct([r["empty"] for r in rows]), "repetitive": pct([r["repetitive"] for r in rows]),
            "few_distinct": pct([r["few_distinct"] for r in rows]), "long_run": pct([r["long_run"] for r in rows]),
            "mean_n_distinct_tokens": float(np.mean([r["n_distinct"] for r in rows])) if rows else None}


def nbr_flags(name, r):
    src, T = TX.PLAN[name]
    if src == "stream":
        eq = r.get("text_first_stream_request_pos") is not None and r["text_first_stream_request_pos"] < T
        men = r["mentions_stream_subject_pos"] is not None and r["mentions_stream_subject_pos"] < T
    else:
        eq = bool(r.get("text_is_dev_request"))
        men = r["mentions_dev_subject_pos"] is not None
    return {"near_text": bool(eq), "near_mention": bool(men), "clean": not (eq or men)}


def public_scores(name, tag):
    """Public summariser on merged public-format case files (post = deployed, pre = pre-edit)."""
    src, T = TX.PLAN[name]
    first = TX.FIRST_RECORD[src]
    merged = os.path.join(C.OUT, "public_results", "ROME_seq_merged", RUNNAME[tag])
    os.makedirs(merged, exist_ok=True)
    n_nonfinite = {"post": 0, "pre": 0}
    for ri in range(first, first + T):
        cid = data[ri]["case_id"]
        post = C.jload(os.path.join(C.OUT, "public_results", "ROME_seq", f"run_{tag}", f"case_{cid}.json"))
        pre = C.jload(os.path.join(C.OUT, "public_results", "ROME_seq", "run_W0", f"case_{cid}.json"))
        for which, d in (("post", post["post"]), ("pre", pre["post"])):
            vals = [x[k] for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs", "neighborhood_prompts_probs") for x in d[key]
                    for k in ("target_new", "target_true")]
            n_nonfinite[which] += int(not np.all(np.isfinite(vals)))
        C.jdump({"case_id": cid, "requested_rewrite": post["requested_rewrite"], "time": 0.0, "post": post["post"], "pre": pre["post"]},
                os.path.join(merged, f"case_{cid}.json"))
    stale = [p for p in glob.glob(os.path.join(merged, "case_*.json"))
             if not (first <= int(os.path.basename(p)[5:-5]) < first + T)]
    assert not stale, stale

    def run(first_n=None):
        with redirect_stdout(io.StringIO()):
            s = public_summarize("ROME_seq_merged", [RUNNAME[tag]], first_n_cases=first_n)
        assert len(s) == 1
        s = s[0]
        out = {"n": int(s["num_cases"])}
        for which, lab in (("post", "deployed"), ("pre", "pre_edit")):
            out[lab] = {"ES": float(s[f"{which}_rewrite_success"][0]), "PS": float(s[f"{which}_paraphrase_success"][0]),
                        "NS": float(s[f"{which}_neighborhood_success"][0]), "EM": float(s[f"{which}_rewrite_diff"][0]),
                        "PM": float(s[f"{which}_paraphrase_diff"][0]), "NM": float(s[f"{which}_neighborhood_diff"][0]),
                        "S": float(s[f"{which}_score"][0])}
        return out
    res = {"all": run(), "n_records_with_nonfinite_prob": n_nonfinite, "case_files": merged}
    if src == "stream":
        res["first100"] = run(100)
    return res


def write_time_scores(name):
    """Public summariser on the case files written right after each update (the request's own record, deployed
    weight of that moment)."""
    src, T = TX.PLAN[name]
    with redirect_stdout(io.StringIO()):
        s = public_summarize(Path(os.path.join(C.LOGS_EDIT, src)), ["write_time"], first_n_cases=(T if src == "stream" else None), abs_path=True)
    s = s[0]
    return {"n": int(s["num_cases"]), "ES": float(s["post_rewrite_success"][0]), "PS": float(s["post_paraphrase_success"][0]),
            "NS": float(s["post_neighborhood_success"][0]), "EM": float(s["post_rewrite_diff"][0]),
            "PM": float(s["post_paraphrase_diff"][0]), "NM": float(s["post_neighborhood_diff"][0]), "S": float(s["post_score"][0])}


def write_time_segment(name, lo, hi):
    """Public summariser on the write-time case files of the updates lo .. hi-1 of a log (links in a scratch run dir)."""
    src, T = TX.PLAN[name]
    first = TX.FIRST_RECORD[src]
    parent = os.path.join(C.OUT, "public_results", "ROME_seq_writetime", f"{TAG[name]}_{lo}_{hi}")
    run = os.path.join(parent, "run_wt")
    os.makedirs(run, exist_ok=True)
    want = {f"case_{data[first + i]['case_id']}.json" for i in range(lo, hi)}
    for fn in os.listdir(run):
        if fn not in want:
            os.remove(os.path.join(run, fn))
    for fn in want:
        dst = os.path.join(run, fn)
        if not os.path.exists(dst):
            os.symlink(os.path.join(C.LOGS_EDIT, src, "write_time", fn), dst)
    with redirect_stdout(io.StringIO()):
        s = public_summarize(Path(parent), ["run_wt"], abs_path=True)
    s = s[0]
    return {"range": [lo, hi], "n": int(s["num_cases"]), "ES": float(s["post_rewrite_success"][0]), "PS": float(s["post_paraphrase_success"][0]),
            "NS": float(s["post_neighborhood_success"][0]), "EM": float(s["post_rewrite_diff"][0]),
            "PM": float(s["post_paraphrase_diff"][0]), "NM": float(s["post_neighborhood_diff"][0])}


def gen_summary(tag, sname):
    p = os.path.join(DESC, tag, f"gen_{sname}.json")
    if not os.path.exists(p):
        return None
    g = C.jload(p)
    ok = [r for r in g["rows"] if "failed" not in r]
    if not ok:
        return {"n_records": 0, "n_failed": len(g["rows"]), "n_texts": 0, "GE": None, "GE_se": None, "per_text": [],
                "per_text_q05": None, "per_text_min": None, "mean_words": None,
                "failure": g["rows"][0]["failed"] if g["rows"] else None}
    per_rec = np.array([r["ngram_entropy"] for r in ok])
    per_text = np.array([x for r in ok for x in r["ngram_entropy_per_text"]])
    words = np.array([len(t.split()) for r in ok for t in r["text"]])
    return {"n_records": len(ok), "n_failed": len(g["rows"]) - len(ok), "n_texts": int(per_text.size), "GE": float(100 * per_rec.mean()),
            "GE_se": float(100 * per_rec.std(ddof=1) / np.sqrt(per_rec.size)), "per_text": per_text.tolist(),
            "per_text_q05": float(100 * np.quantile(per_text, 0.05)), "per_text_min": float(100 * per_text.min()),
            "mean_words": float(words.mean())}


def updates_summary(per, W0_fro):
    fro = np.array([p["update_fro"] for p in per])
    med = float(np.median(fro))
    expl = [{k: p[k] for k in ("index", "case_id", "subject", "prompt", "target_new", "update_fro", "division_factor", "att_template_mean",
                               "lookup_idx_request", "residual_norm", "v_prob_new", "W_minus_W0_fro_after")}
            for p in per if p["update_fro"] > W0_fro]
    pn = np.array([p["v_prob_new"] for p in per], dtype=np.float64)
    return {"n": len(per), "update_fro": {"min": float(fro.min()), "median": med, "q95": float(np.quantile(fro, 0.95)), "max": float(fro.max())},
            "n_exploding": len(expl), "exploding_rule": "||update||_F > ||W_0||_F", "exploding": expl,
            "n_large": int((fro > 5 * med).sum()), "large_rule": "||update||_F > 5 x median of the log",
            "n_subject_at_position_0": int(sum(p["lookup_idx_request"] == 0 for p in per)),
            "n_negative_division_factor": int(sum(p["division_factor"] < 0 for p in per)),
            "min_abs_division_factor": float(min(abs(p["division_factor"]) for p in per)),
            "editor_p_new_median": float(np.median(pn)), "editor_p_new_share_below_0.5": float((pn < 0.5).mean()),
            "first_exploding_index": expl[0]["index"] if expl else None}


def main():
    reg = TX.registry()
    audit = C.jload(os.path.join(C.OUT, "log_audit.json"))
    pools = C.jload(os.path.join(C.OUT, "pools.json"))
    res = {"written": C.now(), "family": "TX-G2XL-rome", "n_dec": E.N_DEC, "logs": {}, "pending": [],
           "pre_continuations": C.jload(os.path.join(C.OUT, "pre_continuations_summary.json")),
           "pools": {k: v for k, v in pools.items() if k not in ("repair_records", "heldout_records")}}
    if not have("W0"):
        raise SystemExit("the pre-edit description (tag W0) is missing")
    dec0 = Dec("W0")
    etf0 = torch.load(os.path.join(DESC, "W0", "edit_tf.pt"), map_location="cpu")
    for name in TX.NAMES:
        tag = TAG[name]
        if name not in reg or not have(tag):
            res["pending"].append(name)
            continue
        src, T = TX.PLAN[name]
        e = reg[name]
        dec = Dec(tag)
        etf = torch.load(os.path.join(DESC, tag, "edit_tf.pt"), map_location="cpu")
        L = {"name": name, "tag": tag, "source": src, "T": T, "records": e["records"], "development_log": e["development_log"],
             "pre_edit_sha256": e["pre_edit_sha256"], "deployed_sha256": e["deployed_sha256"],
             "exact_float32_replay_equals_deployed": e["exact_float32_replay_equals_deployed"],
             "reconstruction": e["reconstruction"], "key_identity": e["key_identity"]}
        per = audit["sources"][src]["per_update"][:T]
        L["updates"] = updates_summary(per, e["reconstruction"]["W0_fro"])
        L["weight_norm_along_stream"] = [{"n_applied": p["index"] + 1, "W_fro": p["W_fro_after"], "W_minus_W0_fro": p["W_minus_W0_fro_after"],
                                          "update_fro": p["update_fro"]} for p in per]
        L["models"] = {}
        ben = TX.benign("heldout", name)
        nbr = [dict(r, **nbr_flags(name, r)) for r in TX.log_prompts(name, "neighborhood")]
        req = TX.log_prompts(name, "request")
        for lab, d_, et_ in (("deployed", dec, etf), ("pre_edit", dec0, etf0)):
            er = edit_rows(name, tag, d_, et_)
            pb, pn, pr = probe_rows(ben, d_), probe_rows(nbr, d_), probe_rows(req, d_)
            M = {"edits": edit_summary(er),
                 "heldout": {"all": probe_summary(pb), "clean": probe_summary([r for r in pb if r["clean"]]),
                             "near": probe_summary([r for r in pb if not r["clean"]]),
                             "near_subject": probe_summary([r for r in pb if r["near_subject"]]),
                             "near_mention": probe_summary([r for r in pb if r["near_mention"]]),
                             "near_text": probe_summary([r for r in pb if r["near_text"]])},
                 "neighborhood": {"all": probe_summary(pn), "clean": probe_summary([r for r in pn if r["clean"]]),
                                  "near": probe_summary([r for r in pn if not r["clean"]])},
                 "request_continuations": probe_summary(pr)}
            # how much of the greedy output consists of the new objects of the exploding updates (collapse signature)
            xs = sorted({t for x in L["updates"]["exploding"] for t in tok(" " + x["target_new"])["input_ids"]})
            if xs:
                xt = torch.tensor(xs)
                dd = torch.stack([d_.row(r["text_id"])[0] for r in ben])
                hit = torch.isin(dd, xt)
                M["heldout_tokens_that_are_objects_of_exploding_updates"] = {
                    "n_object_tokens": len(xs), "objects": sorted({x["target_new"] for x in L["updates"]["exploding"]}),
                    "share_of_tokens": float(100 * hit.float().mean()), "share_of_prompts_with_one": float(100 * hit.any(1).float().mean()),
                    "share_of_first_tokens": float(100 * hit[:, 0].float().mean())}
            if src == "stream":
                M["edits_first100"] = edit_summary(er, first=100)
                M["neighborhood_first100"] = probe_summary([r for r in pn if r["pos"] < 100])
            gtag = tag if lab == "deployed" else "W0"
            M["generation"] = {}
            for sname in ([f"{src}_first100"] + ([f"{src}_last100_T{T}"] if T == C.STREAM_T else [])):
                g = gen_summary(gtag, sname)
                if g is not None:
                    M["generation"][sname.replace(f"{src}_", "").split("_T")[0]] = g
            L["models"][lab] = M
            C.jdump({"log": name, "model": lab, "edits": er, "heldout": pb, "neighborhood": pn, "request": pr},
                    os.path.join(ROWDIR, f"{name}__{lab}.json"), indent=None)
        # generation: share of deployed texts below the 1st percentile of the pre-edit texts of the same set
        for sname, g in L["models"]["deployed"]["generation"].items():
            g0 = L["models"]["pre_edit"]["generation"].get(sname)
            if g0 and g0["per_text"] and g["per_text"]:
                thr = float(np.quantile(g0["per_text"], 0.01))
                g["share_texts_below_pre_edit_q01"] = float(100 * np.mean(np.array(g["per_text"]) < thr))
                g0["share_texts_below_pre_edit_q01"] = float(100 * np.mean(np.array(g0["per_text"]) < thr))
        for lab in ("deployed", "pre_edit"):
            for g in L["models"][lab]["generation"].values():
                g.pop("per_text", None)
        if os.path.isdir(os.path.join(C.OUT, "public_results", "ROME_seq", f"run_{tag}")) and \
                os.path.isdir(os.path.join(C.OUT, "public_results", "ROME_seq", "run_W0")):
            try:
                L["public"] = public_scores(name, tag)
            except FileNotFoundError as ex:
                L["public"] = None
                L["public_missing"] = str(ex)
        L["public_write_time"] = write_time_scores(name)
        e0 = L["updates"]["first_exploding_index"]
        if e0 is not None and 0 < e0 < T:
            segs = {"before_first_exploding": (0, e0), "from_first_exploding": (e0, T)}
            L["write_time_segments"] = {}
            for sname, (lo, hi) in segs.items():
                w = write_time_segment(name, lo, hi)
                pn = np.array([p_["v_prob_new"] for p_ in per[lo:hi]], dtype=np.float64)
                ti = np.array([p_["target_norm"] for p_ in per[lo:hi]], dtype=np.float64)
                w.update({"editor_p_new_median": float(np.median(pn)), "editor_p_new_share_below_0.5": float(100 * (pn < 0.5).mean()),
                          "target_norm_median": float(np.median(ti))})
                L["write_time_segments"][sname] = w
        res["logs"][name] = L
        C.say("aggregated", name)
    C.jdump(res, os.path.join(C.OUT, "text_logs.json"))
    import render_table
    render_table.render()


if __name__ == "__main__":
    main()
