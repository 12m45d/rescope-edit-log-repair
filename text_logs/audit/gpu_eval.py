"""Audit checks 4 and 7 (GPU): the audit's OWN greedy decode against the builder's stored continuations and table.

    python gpu_eval.py [--stack tf4|env] [--parts A,B,C,R]

Nothing of the builder's evaluation code is imported.  The audit's decode is the simplest possible one: ONE prompt at a
time, NO padding, NO attention mask, NO key-value cache -- ten full forward passes, argmax of the last position.
Deployed weights are rebuilt by the audit: W_0 of the loaded checkpoint plus the logged outer products (float32, in place,
in order) and compared bitwise with the builder's snapshot before use.  Only the pre-edit and the deployed model are run.

 A  pre-edit continuations: 100 random prompt texts (of all texts in out/prompt_texts.json): the audit's 10-token greedy
    continuation vs the stored y0; degenerate flags by the audit's own definitions.
 B  the table, for TX-G2XL-rome-dev100 and TX-G2XL-rome-T1000, on 300 random prompts each:
      edits     300 request / paraphrase prompts of the log (dev100: all 300; T1000: 300 of 3000): the deployed greedy
                continuation starts with the token(s) of the new object, tok(' ' + object) (token rule), and the decoded
                text starts with the object string (text rule); vs the builder's per-row flag.
      heldout   300 HELD-OUT benign prompts: y0 = the audit's pre-edit continuation; D1 decode (deployed continuation
                != y0), D1 teacher-forced (argmax under teacher forcing of y0 != y0 somewhere), share of y0 tokens kept;
                vs the builder's per-row flags.
      nbr       300 neighbourhood prompts of the log: the same.
    plus, on the SAME 300 held-out prompts of dev100, the deployed prefixes after 1 and 10 updates (builder: 7.0 / 26.2).
 C  batching / padding: 50 prompts of different lengths.  (i) the builder's stored batched decode vs the audit's single
    decode (pre-edit and dev100 deployed); (ii) the audit's own PADDED batched decode (batches of 10 prompts of unequal
    length) vs single decode: left padding + attention mask + position ids from the mask (correct), and right padding with
    the continuation appended after the pad tokens (the trap), to show what the trap would do to these prompts.
 R  retention: the first 100 request prompts of the stream under the deployed weights T100, T500, T1000, T2000.
Output: out/eval_<stack>.json (+ out/eval_<stack>_rows.json), resumable per part.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stack", default="tf4", choices=["tf4", "env"])
ap.add_argument("--parts", default="A,B,C,R")
args = ap.parse_args()
PARTS = args.parts.split(",")

tfv = A.setup_stack(args.stack, public=False)
import numpy as np  # noqa: E402
import torch  # noqa: E402

cap = A.cap_gpu()
N_DEC = 10
out_path = os.path.join(A.AOUT, f"eval_{args.stack}.json")
rows_path = os.path.join(A.AOUT, f"eval_{args.stack}_rows.json")
RES = A.jload(out_path) if os.path.exists(out_path) else {}
ROWS = A.jload(rows_path) if os.path.exists(rows_path) else {}
RES.update({"stack": args.stack, "transformers": tfv, "torch": torch.__version__, "n_dec": N_DEC})

model, tok = A.load_model()
mod = A.get_module(model)
W0 = mod.weight.detach().clone()
RES["gpu"] = torch.cuda.get_device_name(0)
RES["tokenizer"] = {"padding_side": tok.padding_side, "bos_added_by_tok()": tok("a")["input_ids"][0] == tok.bos_token_id,
                    "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id}
RES["pre_edit_weight_equals_builder_W0_file"] = bool(torch.equal(W0.cpu(), A.load_w0()))
data = A.load_counterfact()
PT = A.jload(os.path.join(A.B_OUT, "prompt_texts.json"))
TEXTS, IDS = PT["texts"], PT["input_ids"]
PRE = torch.load(os.path.join(A.B_OUT, "pre_continuations.pt"), map_location="cpu")
PROMPTS = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
FAC = {}


def save():
    A.jdump(RES, out_path)
    A.jdump(ROWS, rows_path, indent=None)


def set_weight(Wn):
    with torch.no_grad():
        mod.weight.copy_(Wn)


def deployed(src, n):
    """W_0 (of the loaded checkpoint) + the first n logged outer products, written as the public code writes them."""
    if src not in FAC:
        FAC[src] = A.load_factors(src)
    return A.replay(W0, FAC[src]["left"], FAC[src]["right"], n, device="cuda")


@torch.no_grad()
def greedy_single(ids, n=N_DEC):
    x = torch.tensor([ids], device="cuda")
    res, finite = [], True
    for _ in range(n):
        lg = model(input_ids=x, use_cache=False).logits[0, -1]
        finite = finite and bool(torch.isfinite(lg).all())
        nxt = int(lg.argmax())
        res.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], 1)
    return res, finite


@torch.no_grad()
def tf_single(ids, cont):
    x = torch.tensor([ids + cont[:-1]], device="cuda")
    lg = model(input_ids=x, use_cache=False).logits[0, len(ids) - 1:]
    return lg.argmax(-1).tolist(), bool(torch.isfinite(lg).all())


def my_ids(text):
    return tok(text)["input_ids"]


def degenerate(cont):
    """The audit's own definitions for a 10-token continuation."""
    text = tok.decode(cont)
    runs, best = 1, 1
    for a_, b_ in zip(cont, cont[1:]):
        runs = runs + 1 if a_ == b_ else 1
        best = max(best, runs)
    words = text.split()
    return {"empty_or_whitespace": text.strip() == "", "starts_with_newline": text[:1] == "\n", "has_eos": tok.eos_token_id in cont,
            "eos_first": cont[0] == tok.eos_token_id, "n_distinct": len(set(cont)), "max_run": best,
            "le3_distinct": len(set(cont)) <= 3, "run_ge4": best >= 4, "le2_words": len(words) <= 2,
            "one_token_cycle2": all(cont[i] == cont[i % 2] for i in range(len(cont)))}


def bdec(tag):
    d = torch.load(os.path.join(A.B_OUT, "describe", tag, "dec.pt"), map_location="cpu")
    pos = {int(t): i for i, t in enumerate(d["text_ids"].tolist())}
    return d, pos


def brows(name, model_label):
    r = A.jload(os.path.join(A.B_OUT, "text_logs_rows", f"{name}__{model_label}.json"))
    return {k: {x["prompt_id"]: x for x in r[k]} for k in ("edits", "heldout", "neighborhood", "request")}


def pct(x):
    x = list(x)
    return float(100 * np.mean(x)) if x else None


# ----------------------------------------------------------------------------------------------------------- A
if "A" in PARTS and "A" not in RES:
    rng = np.random.default_rng(A.AUDIT_SEED + 1)
    tids = sorted(int(x) for x in rng.choice(len(TEXTS), 100, replace=False))
    set_weight(W0)
    rows, n_ids, n_y0, n_tf, n_valid_equal = [], 0, 0, 0, 0
    for t in tids:
        ids = my_ids(TEXTS[t])
        n_ids += ids == IDS[t]
        y, fin = greedy_single(ids)
        tfp, fin2 = tf_single(ids, y)
        st = PRE["y0"][t].tolist()
        n_y0 += y == st
        n_tf += tfp == y
        n_valid_equal += (tfp == y and fin and fin2) == bool(PRE["valid"][t])
        rows.append({"text_id": t, "text": TEXTS[t], "audit_y0": y, "stored_y0": st, "equal": y == st, "audit_y0_text": tok.decode(y),
                     "n_tokens_equal": int(sum(a_ == b_ for a_, b_ in zip(y, st))), "finite": fin and fin2, **degenerate(y)})
    RES["A"] = {"n": len(tids), "sample_rule": f"default_rng({A.AUDIT_SEED + 1}).choice(n_texts, 100, replace=False)",
                "n_token_ids_equal_stored": int(n_ids), "n_y0_equal_stored": int(n_y0),
                "n_tokens_equal_of_1000": int(sum(r["n_tokens_equal"] for r in rows)),
                "n_audit_y0_reproduced_under_teacher_forcing": int(n_tf), "n_valid_flag_equal": int(n_valid_equal),
                "degenerate": {k: int(sum(r[k] for r in rows)) for k in ("empty_or_whitespace", "starts_with_newline", "has_eos", "eos_first",
                                                                         "le3_distinct", "run_ge4", "le2_words", "one_token_cycle2")},
                "differences": [r for r in rows if not r["equal"]]}
    ROWS["A"] = rows
    save()
    A.say("A:", {k: v for k, v in RES["A"].items() if k not in ("differences",)})

# ----------------------------------------------------------------------------------------------------------- B
LOGS_B = {"TX-G2XL-rome-dev100": ("dev", 100, "dev100"), "TX-G2XL-rome-T1000": ("stream", 1000, "T1000")}
if "B" in PARTS:
    RES.setdefault("B", {})
    for li, (name, (src, T, tag)) in enumerate(LOGS_B.items()):
        if name in RES["B"] and RES["B"][name].get("done"):
            continue
        rng = np.random.default_rng(A.AUDIT_SEED + 10 + li)
        log_rows = [r for r in PROMPTS if r["set"] == src and r["pos"] < T]
        ed = [r for r in log_rows if r["kind"] in ("request", "paraphrase")]
        nb = [r for r in log_rows if r["kind"] == "neighborhood"]
        ho = [r for r in PROMPTS if r["set"] == "heldout"]
        s_ed = ed if len(ed) <= 300 else [ed[int(i)] for i in sorted(rng.choice(len(ed), 300, replace=False))]
        s_ho = [ho[int(i)] for i in sorted(rng.choice(len(ho), 300, replace=False))]
        s_nb = [nb[int(i)] for i in sorted(rng.choice(len(nb), 300, replace=False))]
        WE = deployed(src, T)
        snap = A.load_snapshot(src, T)
        eq_snap = bool(torch.equal(WE.cpu(), snap))
        B_dep, B_pre = brows(name, "deployed"), brows(name, "pre_edit")
        d_dep, pos_dep = bdec(tag)
        R = {"log": name, "T": T, "audit_replay_equals_builder_snapshot_bitwise": eq_snap, "deployed_sha256": A.tensor_sha256(WE),
             "n_edit_rows_population": len(ed), "n_heldout_rows_population": len(ho), "n_neighborhood_rows_population": len(nb)}
        # --- pre-edit continuations of the probe rows (the audit's own y0)
        set_weight(W0)
        y0, y0_ok = {}, {}
        for r in s_ho + s_nb:
            if r["text_id"] in y0:
                continue
            ids = my_ids(r["text"])
            y, fin = greedy_single(ids)
            tfp, fin2 = tf_single(ids, y)
            y0[r["text_id"]] = y
            y0_ok[r["text_id"]] = bool(tfp == y and fin and fin2)
        pre_ed = {}
        for r in s_ed:
            if r["text_id"] not in pre_ed:
                pre_ed[r["text_id"]] = greedy_single(my_ids(r["text"]))[0]
        # --- deployed
        set_weight(WE)
        dep, dep_tf = {}, {}
        for r in s_ho + s_nb:
            t = r["text_id"]
            if t in dep:
                continue
            ids = my_ids(r["text"])
            dep[t] = greedy_single(ids)
            dep_tf[t] = tf_single(ids, y0[t])
        dep_ed = {}
        for r in s_ed:
            if r["text_id"] not in dep_ed:
                dep_ed[r["text_id"]] = greedy_single(my_ids(r["text"]))
        set_weight(W0)
        # --- edits
        rows_e = []
        for r in s_ed:
            rw = data[r["record"]]["requested_rewrite"]
            tgt = tok(" " + rw["target_new"]["str"])["input_ids"]
            true = tok(" " + rw["target_true"]["str"])["input_ids"]
            cont, fin = dep_ed[r["text_id"]]
            txt = tok.decode(cont)
            b = B_dep["edits"][r["prompt_id"]]
            bp = B_pre["edits"][r["prompt_id"]]
            pre_cont = pre_ed[r["text_id"]]
            rows_e.append({"prompt_id": r["prompt_id"], "kind": r["kind"], "n_target_tokens": len(tgt),
                           "audit_exact_token": bool(cont[:len(tgt)] == tgt and fin),
                           "audit_exact_text": txt.lstrip().startswith(rw["target_new"]["str"]),
                           "audit_true_token": bool(cont[:len(true)] == true and fin),
                           "audit_pre_exact_token": bool(pre_cont[:len(tgt)] == tgt),
                           "builder_dec_exact": bool(b["new_dec_exact"]), "builder_tf_exact": bool(b["new_tf_exact"]),
                           "builder_pre_dec_exact": bool(bp["new_dec_exact"]), "finite": fin,
                           "stored_ids_equal": my_ids(r["text"]) == IDS[r["text_id"]], "prompt_ends_with_space": r["text"] != r["text"].rstrip(),
                           "audit_text": txt, "target": rw["target_new"]["str"]})
        R["edits"] = {"n": len(rows_e), "n_rows_equal_builder_dec_exact": int(sum(x["audit_exact_token"] == x["builder_dec_exact"] for x in rows_e)),
                      "n_rows_pre_edit_equal_builder": int(sum(x["audit_pre_exact_token"] == x["builder_pre_dec_exact"] for x in rows_e)),
                      "n_token_rule_differs_from_text_rule": int(sum(x["audit_exact_token"] != x["audit_exact_text"] for x in rows_e)),
                      "n_nonfinite": int(sum(not x["finite"] for x in rows_e)), "max_target_tokens": max(x["n_target_tokens"] for x in rows_e)}
        for kind in ("request", "paraphrase"):
            kk = [x for x in rows_e if x["kind"] == kind]
            full = [x for x in B_dep["edits"].values() if x["kind"] == kind]
            R["edits"][kind] = {"n": len(kk), "audit_exact_token_pct": pct(x["audit_exact_token"] for x in kk),
                                "audit_exact_text_pct": pct(x["audit_exact_text"] for x in kk),
                                "builder_same_rows_pct": pct(x["builder_dec_exact"] for x in kk),
                                "builder_all_rows_pct": pct(x["new_dec_exact"] for x in full), "n_all_rows": len(full),
                                "audit_pre_edit_exact_pct": pct(x["audit_pre_exact_token"] for x in kk),
                                "audit_true_object_pct": pct(x["audit_true_token"] for x in kk)}
        # --- probes
        for label, sample, key in (("heldout", s_ho, "heldout"), ("neighborhood", s_nb, "neighborhood")):
            rows_p = []
            for r in sample:
                t = r["text_id"]
                y = y0[t]
                cont, fin = dep[t]
                tfp, fin2 = dep_tf[t]
                b = B_dep[key][r["prompt_id"]]
                rows_p.append({"prompt_id": r["prompt_id"], "text_id": t, "audit_y0_equals_stored": y == PRE["y0"][t].tolist(),
                               "audit_valid": y0_ok[t], "builder_valid": bool(b["valid"]),
                               "audit_D1_dec": bool(cont != y or not fin), "audit_D1_tf": bool(tfp != y or not fin2),
                               "audit_tok_loc": float(np.mean([a_ == b_ for a_, b_ in zip(tfp, y)])) if fin2 else 0.0,
                               "audit_first_token_changed": bool(tfp[0] != y[0]),
                               "builder_D1_dec": bool(b["D1_dec"]), "builder_D1_tf": bool(b["D1_tf"]), "builder_tok_loc": float(b["tok_loc"]),
                               "audit_dep_equals_builder_stored_dec": cont == d_dep["dec"][pos_dep[t]].tolist(),
                               "finite": fin and fin2, "has_eos_y0": tok.eos_token_id in y, **{"dep_" + k: v for k, v in degenerate(cont).items()
                                                                                              if k in ("le3_distinct", "run_ge4", "empty_or_whitespace")}})
            v = [x for x in rows_p if x["audit_valid"]]
            full = [x for x in B_dep[key].values() if x["valid"]]
            R[label] = {"n": len(rows_p), "n_valid": len(v), "n_audit_y0_equal_stored": int(sum(x["audit_y0_equals_stored"] for x in rows_p)),
                        "audit_D1_tf_pct": pct(x["audit_D1_tf"] for x in v), "audit_D1_dec_pct": pct(x["audit_D1_dec"] for x in v),
                        "audit_tok_loc_pct": pct(x["audit_tok_loc"] for x in v), "audit_first_token_changed_pct": pct(x["audit_first_token_changed"] for x in v),
                        "builder_same_rows_D1_tf_pct": pct(x["builder_D1_tf"] for x in v), "builder_same_rows_D1_dec_pct": pct(x["builder_D1_dec"] for x in v),
                        "builder_same_rows_tok_loc_pct": pct(x["builder_tok_loc"] for x in v),
                        "builder_all_rows_D1_tf_pct": pct(x["D1_tf"] for x in full), "builder_all_rows_tok_loc_pct": pct(x["tok_loc"] for x in full),
                        "n_all_rows_valid": len(full),
                        "n_rows_D1_tf_equal_builder": int(sum(x["audit_D1_tf"] == x["builder_D1_tf"] for x in rows_p)),
                        "n_rows_D1_dec_equal_builder": int(sum(x["audit_D1_dec"] == x["builder_D1_dec"] for x in rows_p)),
                        "n_rows_tok_loc_equal_builder": int(sum(abs(x["audit_tok_loc"] - x["builder_tok_loc"]) < 1e-6 for x in rows_p)),
                        "n_D1_tf_differs_from_D1_dec": int(sum(x["audit_D1_tf"] != x["audit_D1_dec"] for x in rows_p)),
                        "n_deployed_continuation_equal_builder_stored": int(sum(x["audit_dep_equals_builder_stored_dec"] for x in rows_p)),
                        "audit_D1_tf_pct_without_eos_rows": pct(x["audit_D1_tf"] for x in v if not x["has_eos_y0"]),
                        "n_rows_with_eos_in_y0": int(sum(x["has_eos_y0"] for x in rows_p)),
                        "deployed_repetitive_pct(le3 distinct or run>=4)": pct(x["dep_le3_distinct"] or x["dep_run_ge4"] for x in rows_p),
                        "n_nonfinite": int(sum(not x["finite"] for x in rows_p))}
            ROWS[f"B:{name}:{label}"] = rows_p
        ROWS[f"B:{name}:edits"] = rows_e
        # --- dev100 only: D1 on the same held-out sample after 1 and 10 updates
        if src == "dev":
            R["heldout_prefix_curve"] = {}
            for n in (1, 10):
                set_weight(deployed(src, n))
                fl, tl = [], []
                for r in s_ho:
                    t = r["text_id"]
                    if not y0_ok[t]:
                        continue
                    tfp, fin2 = tf_single(my_ids(r["text"]), y0[t])
                    fl.append(tfp != y0[t] or not fin2)
                    tl.append(float(np.mean([a_ == b_ for a_, b_ in zip(tfp, y0[t])])))
                R["heldout_prefix_curve"][str(n)] = {"n": len(fl), "audit_D1_tf_pct": pct(fl), "audit_tok_loc_pct": pct(tl)}
            set_weight(W0)
        R["done"] = True
        RES["B"][name] = R
        save()
        A.say("B", name, json.dumps({k: v for k, v in R.items() if k in ("edits", "heldout", "neighborhood", "heldout_prefix_curve")})[:3000])

# ----------------------------------------------------------------------------------------------------------- C
if "C" in PARTS and "C" not in RES:
    rng = np.random.default_rng(A.AUDIT_SEED + 30)
    ho = [r for r in PROMPTS if r["set"] == "heldout"]
    dv = [r for r in PROMPTS if r["set"] == "dev"]
    sample = [ho[int(i)] for i in rng.choice(len(ho), 25, replace=False)] + [dv[int(i)] for i in rng.choice(len(dv), 25, replace=False)]
    tids = []
    for r in sample:
        if r["text_id"] not in tids:
            tids.append(r["text_id"])
    WE = deployed("dev", 100)
    C_ = {"n_prompts": len(tids), "token_lengths": sorted({len(IDS[t]) for t in tids})}

    @torch.no_grad()
    def batched_left(tids_b):
        """Left padding, attention mask, position ids counted from the first real token: the correct padded batch."""
        L = max(len(IDS[t]) for t in tids_b)
        x = torch.full((len(tids_b), L), tok.pad_token_id, dtype=torch.long, device="cuda")
        m = torch.zeros((len(tids_b), L), dtype=torch.long, device="cuda")
        for i, t in enumerate(tids_b):
            x[i, L - len(IDS[t]):] = torch.tensor(IDS[t], device="cuda")
            m[i, L - len(IDS[t]):] = 1
        res = []
        for _ in range(N_DEC):
            pos = (m.cumsum(1) - 1).clamp(min=0)
            lg = model(input_ids=x, attention_mask=m, position_ids=pos, use_cache=False).logits[:, -1]
            nxt = lg.argmax(-1)
            res.append(nxt)
            x = torch.cat([x, nxt[:, None]], 1)
            m = torch.cat([m, torch.ones_like(nxt)[:, None]], 1)
        return torch.stack(res, 1).tolist()

    @torch.no_grad()
    def batched_right_naive(tids_b):
        """Right padding with an attention mask, the continuation appended after the pad tokens: the classic trap."""
        L = max(len(IDS[t]) for t in tids_b)
        x = torch.full((len(tids_b), L), tok.pad_token_id, dtype=torch.long, device="cuda")
        m = torch.zeros((len(tids_b), L), dtype=torch.long, device="cuda")
        for i, t in enumerate(tids_b):
            x[i, :len(IDS[t])] = torch.tensor(IDS[t], device="cuda")
            m[i, :len(IDS[t])] = 1
        res = []
        for _ in range(N_DEC):
            lg = model(input_ids=x, attention_mask=m, use_cache=False).logits[:, -1]
            nxt = lg.argmax(-1)
            res.append(nxt)
            x = torch.cat([x, nxt[:, None]], 1)
            m = torch.cat([m, torch.ones_like(nxt)[:, None]], 1)
        return torch.stack(res, 1).tolist()

    for label, Wn, tag in (("pre_edit", W0, "W0"), ("dev100_deployed", WE, "dev100")):
        set_weight(Wn)
        d, pos = bdec(tag)
        single = {t: greedy_single(IDS[t])[0] for t in tids}
        n_stored = sum(single[t] == d["dec"][pos[t]].tolist() for t in tids)
        left, right = {}, {}
        for i in range(0, len(tids), 10):
            b = tids[i:i + 10]
            for t, a_, b_ in zip(b, batched_left(b), batched_right_naive(b)):
                left[t], right[t] = a_, b_
        longest = {t for i in range(0, len(tids), 10) for t in tids[i:i + 10] if len(IDS[t]) == max(len(IDS[u]) for u in tids[i:i + 10])}
        C_[label] = {"n_builder_stored_batched_equal_audit_single": int(n_stored),
                     "n_audit_left_padded_batched_equal_single": int(sum(left[t] == single[t] for t in tids)),
                     "n_audit_right_padded_naive_equal_single": int(sum(right[t] == single[t] for t in tids)),
                     "n_prompts_that_needed_no_padding": len(longest),
                     "n_right_padded_naive_equal_single_among_padded": int(sum(right[t] == single[t] for t in tids if t not in longest)),
                     "left_padded_differences": [{"text_id": t, "single": tok.decode(single[t]), "batched": tok.decode(left[t])} for t in tids if left[t] != single[t]][:5]}
    set_weight(W0)
    RES["C"] = C_
    save()
    A.say("C:", json.dumps(C_)[:1500])

# ----------------------------------------------------------------------------------------------------------- R
if "R" in PARTS and "R" not in RES:
    req = [r for r in PROMPTS if r["set"] == "stream" and r["kind"] == "request" and r["pos"] < 100]
    par = [r for r in PROMPTS if r["set"] == "stream" and r["kind"] == "paraphrase" and r["pos"] < 100]
    R_ = {}
    for T in (100, 500, 1000, 2000):
        WE = deployed("stream", T)
        eq = bool(torch.equal(WE.cpu(), A.load_snapshot("stream", T)))
        set_weight(WE)
        hit = {"request": [], "paraphrase": []}
        for r in req + par:
            tgt = tok(" " + data[r["record"]]["requested_rewrite"]["target_new"]["str"])["input_ids"]
            cont, fin = greedy_single(my_ids(r["text"]), n=len(tgt))
            hit[r["kind"]].append(bool(cont == tgt and fin))
        name = f"TX-G2XL-rome-T{T}"
        B_dep = brows(name, "deployed")["edits"]
        R_[f"T{T}"] = {"audit_replay_equals_snapshot": eq, "n_request": len(req), "audit_request_exact_pct": pct(hit["request"]),
                       "audit_paraphrase_exact_pct": pct(hit["paraphrase"]),
                       "builder_request_exact_pct": pct(B_dep[r["prompt_id"]]["new_dec_exact"] for r in req),
                       "builder_paraphrase_exact_pct": pct(B_dep[r["prompt_id"]]["new_dec_exact"] for r in par),
                       "n_rows_equal_builder": int(sum(h == bool(B_dep[r["prompt_id"]]["new_dec_exact"]) for h, r in zip(hit["request"] + hit["paraphrase"], req + par)))}
    set_weight(W0)
    RES["R"] = R_
    save()
    A.say("R:", json.dumps(R_))

RES["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2 ** 20
RES["finished"] = A.now()
save()
A.say("done", out_path, f"peak {RES['peak_alloc_MiB']:.0f} MiB")
