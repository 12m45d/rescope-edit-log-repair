"""Verification of a backbone's key cache.

    python fc_verify.py <backbone> gpu [--device cuda:0] [--mem_gb G] [--n 100] [--tag T]   fresh capture, seeded sample
    python fc_verify.py <backbone> dev [--device cuda:0] [--n 30]    the same on the DEVELOPMENT log's handle (another log)
    python fc_verify.py <backbone> cpu [--tag T]                     model-free checks against stored files

gpu   A seeded uniform sample of n sequences of index.jsonl is captured again in a fresh process with the PLAIN library
      calls (rescope.Evaluator.capture / decode_keys / lp_shift / eligible_keys(answer='all'); no vision-feature memo;
      the handle of the family's master log stays at its edited weight W_E, whereas the cache was captured under W_0)
      and compared bit for bit with keys_tf.pt, keys_decode.pt and keys_decode_steps.pt.  Sequences with lp_shift != 0
      that are not in the sample are checked in addition, at most 60 of them in index order (reported separately;
      the table states how many shifted sequences were NOT re-captured).
dev   The same comparison for a seeded sample of sequences that belong to the development log, captured through the
      development log's own handle (another edit log of the same backbone).
cpu   (1) stored solver keys (B = 10 batched forward, edit_factors.pt K1_solver) against the cached Rel keys: cosine
      range; stored single-item keys (k_t_single, B = 1) against the cached Rel keys: bitwise; (2) the keys.pt of the
      existing hard negatives against the cached keys: bitwise; (3) pre-edit flags against the stored ones; (4) internal
      consistency (positions, duplicate queries give identical decode keys, finite values).
Results: <cache>/<backbone>/verify_<mode>.json.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_common as C  # noqa: E402
import fc_capture as CAP  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("backbone")
ap.add_argument("mode", choices=("gpu", "dev", "cpu"))
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--mem_gb", type=float, default=None)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--tag", default="")
args = ap.parse_args()
BB = args.backbone
CFG = C.BACKBONES[BB]
VF = CFG["vlkeb"]
OD = os.path.join(C.WORK, BB, f"assembled_{args.tag}", "cache") if args.tag else os.path.join(C.OUT, BB)
index = C.jsonl_read(os.path.join(OD, "index.jsonl"))
K_tf = torch.load(os.path.join(OD, "keys_tf.pt"))
K_dec = torch.load(os.path.join(OD, "keys_decode.pt"))
K_st = torch.load(os.path.join(OD, "keys_decode_steps.pt"))
by_sid = {e["sid"]: e for e in index}
TB = C.tb()
t0 = time.time()


def item_of(e):
    """Library item of an index line (for a handle of the line's family)."""
    m = dict(e, lib_kind={"rel": "edit", "tgen": "edit", "igen": "edit", "probe": "probe", "hn_i": "hn", "hn_q": "hn", "hn_x": "hn",
                           "benign": "benign"}[e["kind"]])
    return CAP.make_item(m)


def compare(ev, S, e):
    """Fresh capture of one sequence through the library -> dict of bitwise comparisons with the cache."""
    it = item_of(e)
    out = {"sid": e["sid"], "kind": e["kind"], "family": e["family"], "lp_shift": e["lp_shift"]}
    dk = ev.decode_keys(it).float().cpu()
    out["decode_steps_equal"] = bool(torch.equal(dk, K_st[e["rows_decode_steps"]]))
    out["decode_steps_n_rows_differing"] = int((dk != K_st[e["rows_decode_steps"]]).any(1).sum())
    x, p = ev.capture(it, False)
    out["decode_key_equal"] = bool(torch.equal(x[p].float().cpu(), K_dec[e["row_decode"]]))
    out["lp_decode_equal"] = int(p) == e["lp_decode"]
    if e["row_tf"] is not None:
        x, p = ev.capture(it, True)
        out["tf_key_equal"] = bool(torch.equal(x[p].float().cpu(), K_tf[e["row_tf"]]))
        out["tf_max_abs_diff"] = float((x[p].float().cpu() - K_tf[e["row_tf"]]).abs().max())
        out["lp_tf_equal"] = int(p) == e["lp_tf"]
        sh = ev.lp_shift(it)
        out["lp_shift_equal"] = sh == e["lp_shift"]
        if e["row_tf_prompt"] is not None:
            x2, p2 = ev.capture(it, True, lp="prompt")
            out["tf_prompt_key_equal"] = bool(torch.equal(x2[p2].float().cpu(), K_tf[e["row_tf_prompt"]])) and int(p2) == e["lp_tf"] - e["lp_shift"]
        # the library's 'every key the query can present' against the cache's composition of it
        allk = ev.eligible_keys(it, S, answer="all").float().cpu()
        rows = [K_tf[e["row_tf"]][None]] + ([K_tf[e["row_tf_prompt"]][None]] if e["row_tf_prompt"] is not None else []) + [K_st[e["rows_decode_steps"]]]
        mine = torch.cat(rows)
        out["eligible_all_equal"] = bool(allk.shape == mine.shape and torch.equal(allk, mine))
    return out


def summarise(rows):
    keys = ("decode_key_equal", "decode_steps_equal", "lp_decode_equal", "tf_key_equal", "lp_tf_equal", "lp_shift_equal",
            "tf_prompt_key_equal", "eligible_all_equal")
    s = {"n": len(rows), "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
         "by_family": dict(collections.Counter(r["family"] for r in rows))}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        s[k] = {"n": len(v), "n_true": int(sum(v))}
    s["all_bitwise"] = all(r[k] for r in rows for k in keys if k in r)
    s["n_sequences_all_bitwise"] = sum(all(r[k] for k in keys if k in r) for r in rows)
    s["failures"] = [r for r in rows if not all(r[k] for k in keys if k in r)][:20]
    return s


def run_gpu(sample, extra, dev_log=None):
    import gc
    import types
    rs = C.import_rescope()
    C.tb()
    rows, rows_x = [], []
    fams = list(dict.fromkeys(e["family"] for e in sample + extra))
    info = {}
    for f in fams:
        name = dev_log or CFG["families"][f]
        t1 = time.time()
        H = rs.load(name, device=args.device, mem_gb=args.mem_gb or CFG["mem_gb"])
        ev_lib = rs.Evaluator(H, hard_negatives=False)          # the plain library object: every item it can express
        ev_txt = CAP.make_evaluator(rs, H)                      # + text-only free queries (benign text rows only)
        try:                                                    # eligible_keys only reads the gate's role
            S = rs.ScopedLog.from_log(rs.from_harness(H), role="last", W0=H.W0)
            gate = "ScopedLog.from_log(role='last')"
        except Exception as ex:                                 # e.g. an edit with a non-positive own coordinate
            S, gate = types.SimpleNamespace(role="last"), f"stub with role 'last' ({type(ex).__name__}: {ex})"[:200]
        info[f] = {"log": name, "T": H.T, "gate_for_eligible_keys": gate, "model_weight": "W_E of the log (the cache was captured under W_0)",
                   "weight_is_WE": bool(torch.equal(H.mod.weight.data, H.WE)), "weight_is_W0": bool(torch.equal(H.mod.weight.data, H.W0)),
                   "n_through_plain_library_evaluator": 0, "n_through_text_subclass": 0}
        with torch.no_grad():
            for dst, src in ((rows, sample), (rows_x, extra)):
                for e in [e for e in src if e["family"] == f]:
                    txt = e["kind"] == "benign" and e["ptype"] == "text"
                    info[f]["n_through_text_subclass" if txt else "n_through_plain_library_evaluator"] += 1
                    dst.append(compare(ev_txt if txt else ev_lib, S, e))
        C.say(f"  {f} ({name}): {sum(r['family'] == f for r in rows)} + {sum(r['family'] == f for r in rows_x)} sequences in {time.time() - t1:.0f} s")
        del H, ev_lib, ev_txt, S
        gc.collect()
        torch.cuda.empty_cache()
    return rows, rows_x, info


if args.mode in ("gpu", "dev"):
    n = args.n or (100 if args.mode == "gpu" else 30)
    if args.mode == "gpu":
        cand = index
        dev = None
    else:                                           # sequences that belong to the development log
        dev = CFG["dev"]
        cand = [e for e in index if e["family"] == VF and e["kind"] != "benign"
                and (e["kind"] == "probe" or (e.get("edit") is not None and e["edit"] < 100 and (e.get("edit_b") is None or e["edit_b"] < 100)))]
    pick = np.sort(np.random.default_rng(C.useed("verify", BB, args.mode)).choice(len(cand), min(n, len(cand)), replace=False))
    sample = [cand[int(i)] for i in pick]
    got = {e["sid"] for e in sample}
    extra = [e for e in cand if e["lp_shift"] not in (0, None) and e["sid"] not in got] if args.mode == "gpu" else []
    extra = extra[:60]
    C.say(f"{BB} verify {args.mode}: {len(sample)} sampled sequences {dict(collections.Counter(e['kind'] for e in sample))}, {len(extra)} extra shifted ones")
    rows, rows_x, info = run_gpu(sample, extra, dev)
    res = {"backbone": BB, "mode": args.mode, "seed": C.useed("verify", BB, args.mode), "n_candidates": len(cand), "handles": info,
           "sample": summarise(rows), "extra_shifted": summarise(rows_x) if rows_x else None,
           "n_shifted_in_sample": sum(e["lp_shift"] not in (0, None) for e in sample),
           "gpu": torch.cuda.get_device_name(0), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
           "vision_memo": False, "seconds": round(time.time() - t0, 1), "rows": rows, "rows_extra": rows_x,
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.jdump(res, os.path.join(OD, f"verify_{args.mode}.json"))
    C.say(f"  sample: all bitwise = {res['sample']['all_bitwise']} ({res['sample']['n_sequences_all_bitwise']}/{len(rows)}); "
          f"extra shifted: {None if not rows_x else res['extra_shifted']['n_sequences_all_bitwise']}/{len(rows_x)}")
    sys.exit(0)

# ============================================================================= cpu
rs = C.import_rescope()
res = {"backbone": BB, "mode": "cpu"}


def q(v):
    v = np.asarray(v, dtype=np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()), "n": int(v.size)}


# (1) stored solver keys / single-item keys against the cached Rel keys
sol = {}
for f, master in CFG["families"].items():
    rel = {e["edit"]: e for e in index if e["family"] == f and e["kind"] == "rel"}
    logs = [master] + ([CFG["dev"]] if f == VF else [])
    for name in logs:
        fac = torch.load(os.path.join(TB.OUT_ROOT, name, "edit_factors.pt"), map_location="cpu")
        T = fac["U"].shape[0]
        ids = [i for i in range(T) if i in rel]
        if not ids:
            continue
        Kc = K_tf[[rel[i]["row_tf"] for i in ids]].double()                  # cached B = 1 Rel keys
        Ks = fac["K1_solver"][ids].double()                                 # stored solver keys (B = 10 batched forward)
        K1 = fac["k_t_single"][ids].float()                                 # stored single-item keys (B = 1, build time)
        W = fac["W"][ids].double()
        cos = (Kc * Ks).sum(1) / (Kc.norm(dim=1) * Ks.norm(dim=1))
        r = (W * Kc).sum(1) / (W * Ks).sum(1)                               # normalised own coordinate of the B = 1 key
        eq1 = (K1 == K_tf[[rel[i]["row_tf"] for i in ids]]).all(1)
        cos1 = (Kc * K1.double()).sum(1) / (Kc.norm(dim=1) * K1.double().norm(dim=1))
        sh = [rel[i]["lp_shift"] for i in ids]
        sol[name] = {"family": f, "n_edits": len(ids), "T_log": T,
                     "cosine_solver_key_vs_cached_rel_key": q(cos.numpy()), "argmin_edit": int(ids[int(cos.argmin())]),
                     "rel_l2_diff": q(((Kc - Ks).norm(dim=1) / Ks.norm(dim=1)).numpy()),
                     "own_coordinate_r_of_cached_key": dict(q(r.numpy()), n_below_1=int((r < 1).sum())),
                     "n_solver_key_bitwise_equal": int((fac["K1_solver"][ids].float() == K_tf[[rel[i]["row_tf"] for i in ids]]).all(1).sum()),
                     "single_item_key_bitwise_equal": {"n": len(ids), "n_equal": int(eq1.sum()),
                                                       "edits_differing": [int(ids[j]) for j in (~eq1).nonzero().flatten().tolist()][:30]},
                     "cosine_single_item_key_vs_cached": q(cos1.numpy()),
                     "n_edits_lp_shift_nonzero": int(sum(s not in (0, None) for s in sh)),
                     "cosine_where_lp_shift_nonzero": [float(cos[j]) for j, s in enumerate(sh) if s not in (0, None)][:20]}
        del fac
res["solver_keys"] = sol

# (2) keys of the existing hard negatives
old = torch.load(os.path.join(C.HN_OLD, BB, "keys.pt"), map_location="cpu")
eq, miss, diffs = 0, 0, []
for j, h in enumerate(old["hn_ids"]):
    e = by_sid.get(h)
    if e is None:
        miss += 1
        continue
    mine = K_tf[e["row_tf"]] if e["row_tf"] is not None else K_dec[e["row_decode"]]
    same = bool(torch.equal(old["K_last"][j].float(), mine))
    eq += int(same)
    if not same:
        diffs.append({"hn_id": h, "max_abs_diff": float((old["K_last"][j].float() - mine).abs().max()),
                      "lp_old": int(old["last_prompt_idx"][j]), "lp_tf": e["lp_tf"]})
res["existing_hn_keys"] = {"n": len(old["hn_ids"]), "n_in_cache": len(old["hn_ids"]) - miss, "n_bitwise_equal": eq, "differing": diffs[:20]}

# (3) pre-edit flags against stored ones
pre = {}
for f, master in CFG["families"].items():
    pem = {r["edit_id"]: r for r in TB.jsonl_read(os.path.join(TB.OUT_ROOT, master, "per_edit_metrics.jsonl"))}
    rel = [e for e in index if e["family"] == f and e["kind"] == "rel" and e["edit"] in pem]
    pre[f] = {"n_rel": len(rel),
              "pre_exact_B1_here": int(sum(bool(e["pre_exact"]) for e in rel)),
              "pre_exact_batch8_stored": int(sum(bool(pem[e["edit"]]["pre_rel_exact"]) for e in rel)),
              "n_exact_flag_differs": int(sum(bool(e["pre_exact"]) != bool(pem[e["edit"]]["pre_rel_exact"]) for e in rel)),
              "n_token_acc_differs_gt_1e-6": int(sum(abs(e["pre_token_acc"] - pem[e["edit"]]["pre_rel_token"]) > 1e-6 for e in rel)),
              "mean_pre_token_acc_here": float(np.mean([e["pre_token_acc"] for e in rel])) if rel else None,
              "mean_pre_token_acc_stored": float(np.mean([pem[e["edit"]]["pre_rel_token"] for e in rel])) if rel else None,
              "n_nonfinite": int(sum(not e["pre_finite"] for e in rel)),
              "by_metric_pre_exact": {m: [int(sum(bool(e["pre_exact"]) for e in index if e["family"] == f and e["kind"] == m)),
                                          sum(1 for e in index if e["family"] == f and e["kind"] == m)] for m in ("rel", "tgen", "igen")}}
res["pre_edit_flags_vs_stored"] = pre
pr = [e for e in index if e["kind"] == "probe"]
res["probe_y0"] = {"n": len(pr), "n_with_tf": sum(e["row_tf"] is not None for e in pr), "n_valid": sum(bool(e["y0_valid"]) for e in pr),
                   "n_pre_exact_differs_from_y0_valid": sum(e["row_tf"] is not None and bool(e["pre_exact"]) != bool(e["y0_valid"]) for e in pr),
                   "n_empty_y0": sum(not e["y0_text"] for e in pr)}
dc = [e for e in index if e["answer_from"] == "decoder_v2_here"]
res["decoded_here"] = {"n": len(dc), "n_valid": sum(bool(e["y0_valid"]) for e in dc),
                       "n_pre_exact_differs_from_y0_valid": sum(e["row_tf"] is not None and bool(e["pre_exact"]) != bool(e["y0_valid"]) for e in dc),
                       "n_without_tf_sequence": sum(e["row_tf"] is None for e in dc)}
oh = [e for e in index if e["answer_from"] == "p11_rows"]
res["existing_hn_y0"] = {"n": len(oh), "n_valid": sum(bool(e["y0_valid"]) for e in oh),
                         "n_pre_exact_differs_from_y0_valid": sum(e["row_tf"] is not None and bool(e["pre_exact"]) != bool(e["y0_valid"]) for e in oh)}

# (4) internal consistency
bad_pos = sum(1 for e in index if e["lp_tf"] is not None and e["lp_shift"] is not None and e["lp_tf"] - e["lp_decode"] != e["lp_shift"])
k0_eq = bool(torch.equal(K_st[[e["rows_decode_steps"][0] for e in index]], K_dec[[e["row_decode"] for e in index]]))
groups = collections.defaultdict(list)
for e in index:
    groups[e["query_id"]].append(e)
dup = [g for g in groups.values() if len(g) > 1]
dup_eq = sum(all(torch.equal(K_dec[g[0]["row_decode"]], K_dec[x["row_decode"]]) for x in g[1:]) for g in dup)
dup_steps_eq = sum(all(torch.equal(K_st[g[0]["rows_decode_steps"]], K_st[x["rows_decode_steps"]]) for x in g[1:]) for g in dup)
dup_lp = sum(all(g[0]["lp_decode"] == x["lp_decode"] for x in g[1:]) for g in dup)
res["consistency"] = {
    "n_index": len(index), "sids_unique": len(by_sid) == len(index),
    "row_decode_is_line_number": all(e["row_decode"] == j for j, e in enumerate(index)),
    "row_tf_unique": len({e["row_tf"] for e in index if e["row_tf"] is not None}) == sum(e["row_tf"] is not None for e in index),
    "n_position_identity_violations(lp_tf - lp_decode != lp_shift)": bad_pos,
    "n_lp_shift_none_with_tf": sum(1 for e in index if e["row_tf"] is not None and e["lp_shift"] is None),
    "decode_step0_equals_keys_decode": k0_eq,
    "all_finite": bool(torch.isfinite(K_tf).all() and torch.isfinite(K_dec).all() and torch.isfinite(K_st).all()),
    "n_teacher_forced_sequences_with_nonfinite_pre_edit_logits": sum(1 for e in index if e["row_tf"] is not None and e["pre_finite"] is False),
    "n_answers_without_teacher_forced_sequence": {k: sum(1 for e in index if e["kind"] == k and e["answer"] and e["row_tf"] is None) for k in C.KINDS},
    "n_empty_answers": {k: sum(1 for e in index if e["kind"] == k and e["answer"] is not None and not e["answer"]) for k in C.KINDS},
    "n_zero_rows": int((K_tf.abs().sum(1) == 0).sum() + (K_dec.abs().sum(1) == 0).sum() + (K_st.abs().sum(1) == 0).sum()),
    "duplicate_query_groups": {"n_groups": len(dup), "n_sequences": sum(len(g) for g in dup),
                               "n_groups_decode_key_bitwise_identical": dup_eq, "n_groups_all_11_steps_identical": dup_steps_eq,
                               "n_groups_same_lp_decode": dup_lp},
    "tf_key_equals_decode_key(same token, other length)": {
        "n_with_tf_and_shift_0": sum(1 for e in index if e["row_tf"] is not None and e["lp_shift"] == 0),
        "n_bitwise_equal": int(sum(bool(torch.equal(K_tf[e["row_tf"]], K_dec[e["row_decode"]])) for e in index
                                   if e["row_tf"] is not None and e["lp_shift"] == 0))}}
# (4a) probe pools by CONTENT (query_id = ptype + question string + image md5; the library's ids use the image path)
ppg = {}
ben_q = {e["query_id"] for e in index if e["kind"] == "benign"}
for f in CFG["families"]:
    pr = [e for e in index if e["kind"] == "probe" and e["family"] == f]
    if not pr:
        continue
    lib_of = collections.defaultdict(set)
    for e in pr:
        lib_of[e["query_id"]].add(e["probe_query_id"])
    rep = {e["query_id"] for e in pr if e["pool"] == "repair"}
    rep_other = {e["query_id"] for e in index if e["kind"] == "probe" and e["family"] != f and e["pool"] == "repair"}
    dj = [e for e in pr if e["disjoint"]]
    merged = [k for k, v in lib_of.items() if len(v) > 1]
    ppg[f] = {"n_rows": len(pr), "n_distinct_queries_by_content": len(lib_of), "n_distinct_queries_by_path(library)": len({e["probe_query_id"] for e in pr}),
              "content_groups_spanning_several_library_groups": [[(e["sid"], e["pool"], e["image"]) for e in pr if e["query_id"] == k] for k in merged][:10],
              "n_heldout_disjoint_rows": len(dj),
              "n_heldout_disjoint_rows_equal_by_content_to_a_REPAIR_row_of_the_family": sum(e["query_id"] in rep for e in dj),
              "n_heldout_disjoint_rows_equal_to_a_REPAIR_row_of_another_family": sum(e["query_id"] in rep_other for e in dj),
              "n_probe_rows_equal_to_a_benign_row": {"all": sum(e["query_id"] in ben_q for e in pr), "repair": sum(e["query_id"] in ben_q and e["pool"] == "repair" for e in pr),
                                                     "heldout_disjoint": sum(e["query_id"] in ben_q and bool(e["disjoint"]) for e in pr)}}
res["probe_pools_by_content"] = ppg

# (4b) two independent captures of the same thing: the teacher-forced sequence (prompt + answer, n tokens appended) and the
#      decode forward "prompt + n filler ids" have the same length and, the model being causal, the same key at the prompt's
#      last token -- bit for bit, if both captures are right (same token only: lp_shift == 0; n <= 10)
same_len = [(e, e["len_tf"] - e["len_prompt"]) for e in index if e["row_tf"] is not None and e["lp_shift"] == 0]
chk = [(e, n) for e, n in same_len if 0 < n <= C.N_DECODE - 1]
eq_n = [bool(torch.equal(K_tf[e["row_tf"]], K_st[e["rows_decode_steps"][n]])) for e, n in chk]
res["tf_key_vs_decode_step_of_same_length"] = {
    "n_checked": len(chk), "n_bitwise_equal": int(sum(eq_n)), "n_longer_than_10_tokens(not checkable)": len(same_len) - len(chk),
    "by_kind": {k: [int(sum(q_ for (e, n), q_ in zip(chk, eq_n) if e["kind"] == k)), sum(1 for e, n in chk if e["kind"] == k)] for k in C.KINDS
                if any(e["kind"] == k for e, n in chk)},
    "failures": [e["sid"] for (e, n), q_ in zip(chk, eq_n) if not q_][:20]}

# (5) shifted positions: tokenizer-only recount in this process, and the count on the development log's sequences
vt = TB.VLM(BB, load_model=False)


def shift_of(e):
    if not e["answer"]:
        return None
    img = e["ptype"] == "image"
    try:
        _, text, sep = vt.text_parts(e["prompt"], "vqa" if img else TB.TEXT_FMT[BB], img)
        a, joint = vt.answer_and_joint(text, e["answer"], sep)
        head, alone = joint[:len(joint) - len(a)], vt._tok(text)
        return len(head) - len(alone) if head[:len(alone)] == alone else None
    except RuntimeError:
        return None


with_tf = [e for e in index if e["row_tf"] is not None]
rec_shift = [shift_of(e) for e in with_tf]
dev_seq = [e for e in index if e["family"] == VF and ((e["kind"] in ("rel", "tgen", "igen") and e["edit"] < 100)
                                                      or (e["kind"] == "probe" and e["y0_valid"])
                                                      or (e["kind"] in ("hn_i", "hn_q") and not e["new"] and e["y0_valid"]))]
res["shifted_positions"] = {
    "n_with_tf": len(with_tf), "n_recount_differs_from_cached_lp_shift": sum(a != e["lp_shift"] for a, e in zip(rec_shift, with_tf)),
    "n_shifted": sum(e["lp_shift"] not in (0, None) for e in with_tf),
    "shift_values": {str(k): v for k, v in collections.Counter(e["lp_shift"] for e in with_tf).items()},
    "shifted_by_kind": dict(collections.Counter(e["kind"] for e in with_tf if e["lp_shift"] not in (0, None))),
    "shifted_answers_start_with_digit": sum(e["answer"].strip()[:1].isdigit() for e in with_tf if e["lp_shift"] not in (0, None)),
    "answers_starting_with_digit_not_shifted": sum(e["answer"].strip()[:1].isdigit() and e["lp_shift"] == 0 for e in with_tf),
    "development_log_sequences(300 edit sequences + valid probes + valid existing hard negatives)": {
        "n": len(dev_seq), "n_shifted": sum(e["lp_shift"] not in (0, None) for e in dev_seq),
        "shifted_rel_edits": [e["edit"] for e in dev_seq if e["kind"] == "rel" and e["lp_shift"] not in (0, None)]},
    "shifted_rel_edits_long_stream": [e["edit"] for e in index if e["family"] == VF and e["kind"] == "rel" and e["lp_shift"] not in (0, None)][:60],
    "tf_key_vs_tf_prompt_key_cosine_on_shifted": q([float(torch.nn.functional.cosine_similarity(K_tf[e["row_tf"]].double(), K_tf[e["row_tf_prompt"]].double(), dim=0))
                                                    for e in with_tf if e["row_tf_prompt"] is not None]) if any(e["row_tf_prompt"] is not None for e in with_tf) else None,
    "tf_prompt_key_equals_decode_key_on_shifted": [int(sum(bool(torch.equal(K_tf[e["row_tf_prompt"]], K_dec[e["row_decode"]])) for e in with_tf if e["row_tf_prompt"] is not None)),
                                                   sum(e["row_tf_prompt"] is not None for e in with_tf)]}
# (6) the library's reader (rescope.keycache.KeyCache) on this cache: same counts, same keys
try:
    from rescope import keycache as KC
    kc = KC.KeyCache(OD)
    tf_rows = [j for j, e in enumerate(index) if e["row_tf"] is not None]
    dev_uids = rs.from_testbed(CFG["dev"]).meta["uids"]
    samp = [int(i) for i in np.random.default_rng(C.useed("reader", BB)).choice(len(tf_rows), min(300, len(tf_rows)), replace=False)]
    found = [kc.find(index[tf_rows[i]]["fmt"], index[tf_rows[i]]["prompt"], index[tf_rows[i]]["image"], index[tf_rows[i]]["answer"]) for i in samp]
    st = kc.keys("steps", tf_rows[:64])
    res["library_reader"] = {
        "ok": True, "summary": kc.summary(), "n_equal": kc.n == len(index), "d": kc.d,
        "family_of_development_log_uids": kc.family_of(dev_uids), "family_expected": VF,
        "keys_tf_equal": bool(torch.equal(kc.keys("tf", tf_rows), K_tf[[index[j]["row_tf"] for j in tf_rows]])),
        "keys_prompt_equal": bool(torch.equal(kc.keys("prompt", tf_rows), K_tf[[index[j]["row_tf_prompt"] if index[j]["lp_shift"] else index[j]["row_tf"] for j in tf_rows]])),
        "keys_decode_equal": bool(torch.equal(kc.keys("decode", list(range(len(index)))), K_dec)),
        "keys_steps_shape_ok": list(st.shape) == [min(64, len(tf_rows)), C.N_DECODE, int(K_st.shape[1])],
        "keys_steps_equal": bool(torch.equal(st, K_st[torch.tensor([index[j]["rows_decode_steps"] for j in tf_rows[:64]])])),
        "n_rows_rel_T100": len(kc.rows(kind="rel", family=VF, T=100)),
        "n_rows_hn_T100": len(kc.rows(kind=("hn_i", "hn_q", "hn_x"), family=VF, T=100)),
        "n_rows_probe_repair_valid": len(kc.rows(kind="probe", family=VF, pool="repair", valid=True)),
        "n_rows_probe_heldout_disjoint_valid": len(kc.rows(kind="probe", family=VF, pool="heldout_disjoint", valid=True)),
        "n_rows_benign": len(kc.rows(pool="benign")),
        "find_by_content": {"n": len(samp), "n_found": sum(f is not None for f in found),
                            "n_same_sequence_or_duplicate": sum(f is not None and (f == tf_rows[i] or index[f]["query_id"] == index[tf_rows[i]]["query_id"])
                                                                for f, i in zip(found, samp))}}
except Exception as ex:                                      # the library is being extended: report, do not fail
    res["library_reader"] = {"ok": False, "error": f"{type(ex).__name__}: {ex}"[:500]}
# (7) which logs this cache serves: every log directory of the backbone whose edit stream is a prefix of a cached family;
#     its probe rows (prompt, image, y0, validity) must equal the cached probe rows of that family
fam_uids = {f: [e["edit_uid"] for e in sorted((e for e in index if e["family"] == f and e["kind"] == "rel"), key=lambda e: e["edit"])]
            for f in CFG["families"]}
fam_probe = {f: {(e["probe_idx"], e["ptype"]): e for e in index if e["family"] == f and e["kind"] == "probe"} for f in CFG["families"]}
served, not_served = {}, {}
for name in sorted(os.listdir(TB.OUT_ROOT)):
    p = os.path.join(TB.OUT_ROOT, name, "edit_batch.jsonl")
    if not (name.startswith("TB-") and os.path.exists(p)):
        continue
    bbn = (TB.LOGS.get(name) or {}).get("backbone") or ("llava" if name.startswith("TB-L") else "qwen25vl")
    if bbn != BB:
        continue
    uids = [r["uid"] for r in TB.jsonl_read(p)]
    fam = next((f for f, u in fam_uids.items() if u[:len(uids)] == uids and len(uids) <= len(u)), None)
    if fam is None:
        sub = next((f for f, u in fam_uids.items() if set(uids) <= set(u)), None)
        not_served[name] = {"T": len(uids), "items_are_a_subset_of_family": sub,
                            "reason": ("the same edit items as family " + sub + " in another order: its sequences are cached and found by content "
                                       "(KeyCache.find), but the cached edit positions are not its positions") if sub else
                                      "its edit items are not in the cache (another edit stream)"}
        continue
    rec = {"family": fam, "T": len(uids), "registered": name in TB.LOGS}
    pr = os.path.join(TB.OUT_ROOT, name, "probe_records_v2.jsonl")
    if os.path.exists(pr) and fam_probe[fam]:
        rows_ = TB.jsonl_read(pr)
        diff = sum(1 for r in rows_ if (lambda e: e is None or e["prompt"] != r["prompt"] or e["image"] != r["image"] or e["y0_text"] != r["y0_text"]
                                        or bool(e["y0_valid"]) != bool(r["y0_valid"]) or e["uid"] != r["uid"])(fam_probe[fam].get((r["probe_idx"], r["ptype"]))))
        rec.update(n_probe_rows=len(rows_), n_probe_rows_differing_from_cache=diff)
    else:
        rec.update(n_probe_rows=None, n_probe_rows_differing_from_cache=None)
    served[name] = rec
res["served_logs"] = {"served": served, "not_served": not_served}
res["seconds"] = round(time.time() - t0, 1)
res["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
C.jdump(res, os.path.join(OD, "verify_cpu.json"))
import json  # noqa: E402
print(json.dumps({k: res[k] for k in ("existing_hn_keys", "probe_y0", "decoded_here", "existing_hn_y0", "consistency")}, indent=1)[:5000])
for name, s in sol.items():
    print(name, "cos(solver key, cached Rel key)", s["cosine_solver_key_vs_cached_rel_key"], "| single-item key bitwise", s["single_item_key_bitwise_equal"]["n_equal"], "/", s["n_edits"])
