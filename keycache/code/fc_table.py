"""Table of what the key caches contain: cache_table.json and cache_table.txt in 10_final/cache/ (rendered from the
index files and the verification JSON; no hand-typed number).

    python fc_table.py [--tag smoke]
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fc_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
args = ap.parse_args()


def cache_dir(bb):
    return os.path.join(C.WORK, bb, f"assembled_{args.tag}", "cache") if args.tag else os.path.join(C.OUT, bb)


def count(rows):
    """Counts of a set of index lines."""
    y = [e for e in rows if e.get("y0_text") is not None]
    reasons = collections.Counter(e["y0_invalid_reason"] for e in y if not e["y0_valid"])
    return {"n": len(rows), "n_tf_key": sum(e["row_tf"] is not None for e in rows), "n_decode_key": sum(e["row_decode"] is not None for e in rows),
            "n_with_y0": len(y), "n_valid_y0": sum(bool(e["y0_valid"]) for e in y), "invalid_reasons": dict(reasons),
            "degenerate_y0": {k: sum(bool(e.get(k)) for e in y) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")},
            "degenerate_valid_y0": {k: sum(bool(e.get(k)) for e in y if e["y0_valid"]) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")},
            "n_any_degenerate_y0": sum(any(e.get(k) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")) for e in y),
            "n_in_duplicate_query_group": sum(e["n_same_query"] > 1 for e in rows), "n_distinct_queries": len({e["query_id"] for e in rows}),
            "n_in_duplicate_group_within_family_and_kind": sum(v for v in collections.Counter((e["family"], e["kind"], e["query_id"]) for e in rows).values() if v > 1),
            "n_shifted": sum(e["lp_shift"] not in (0, None) for e in rows),
            "n_shift_undefined_with_answer": sum(e["row_tf"] is not None and e["lp_shift"] is None for e in rows),
            "n_decode_keys_not_all_identical": sum(e["n_decode_distinct"] > 1 for e in rows),
            "max_distinct_decode_keys": max([e["n_decode_distinct"] for e in rows], default=0)}


out = {"generated_by": "10_final/cache/code/fc_table.py", "seed": C.SEED, "backbones": {}}
for bb, cfg in C.BACKBONES.items():
    d = cache_dir(bb)
    if not os.path.exists(os.path.join(d, "index.jsonl")):
        continue
    idx = C.jsonl_read(os.path.join(d, "index.jsonl"))
    meta = C.jload(os.path.join(d, "meta.json"))
    B = {"meta": {k: meta[k] for k in ("layer", "d_in", "module_dtype", "complete", "n_missing_sequences", "n_sequences", "families")},
         "files": {k: meta[k]["shape"] for k in ("keys_tf", "keys_decode", "keys_decode_steps")},
         "keys_tf_shifted_extra_rows": meta["keys_tf"]["n_shifted_extra_rows"],
         "capture": meta["capture"], "by_kind": {}, "by_family_kind": {}}
    for k in C.KINDS:
        sub = [e for e in idx if e["kind"] == k]
        if sub:
            B["by_kind"][k] = count(sub)
    B["by_kind"]["all"] = count(idx)
    for f in cfg["families"]:
        for k in C.KINDS:
            sub = [e for e in idx if e["kind"] == k and e["family"] == f]
            if sub:
                B["by_family_kind"][f"{f}|{k}"] = count(sub)
    # edits: pre-edit teacher-forced flags
    B["pre_edit_flags"] = {}
    for f in cfg["families"]:
        for k in ("rel", "tgen", "igen"):
            sub = [e for e in idx if e["kind"] == k and e["family"] == f]
            if sub:
                B["pre_edit_flags"][f"{f}|{k}"] = {
                    "n": len(sub), "n_pre_exact": sum(bool(e["pre_exact"]) for e in sub),
                    "mean_pre_token_acc_pct": 100.0 * sum(e["pre_token_acc"] for e in sub) / len(sub),
                    "n_nonfinite": sum(not e["pre_finite"] for e in sub),
                    "first100_n_pre_exact": sum(bool(e["pre_exact"]) for e in sub if e["edit"] < 100)}
    # probes: pools
    B["probe_pools"] = {}
    for f in cfg["families"]:
        pr = [e for e in idx if e["kind"] == "probe" and e["family"] == f]
        if pr:
            def n_(sel):
                s = [e for e in pr if sel(e)]
                return {"rows": len(s), "valid": sum(bool(e["y0_valid"]) for e in s),
                        "image_valid": sum(bool(e["y0_valid"]) for e in s if e["ptype"] == "image"),
                        "text_valid": sum(bool(e["y0_valid"]) for e in s if e["ptype"] == "text"),
                        "distinct_queries_valid": len({e["query_id"] for e in s if e["y0_valid"]})}
            B["probe_pools"][f] = {"all": n_(lambda e: True), "repair": n_(lambda e: e["pool"] == "repair"),
                                   "heldout": n_(lambda e: e["pool"] == "heldout"), "heldout_disjoint": n_(lambda e: e["disjoint"]),
                                   "heldout_strict": n_(lambda e: e["strict"])}
    # hard negatives
    hn = [e for e in idx if e["kind"] in ("hn_i", "hn_q", "hn_x")]
    B["hard_negatives"] = {}
    for t in ("hn_i", "hn_q", "hn_x"):
        for g in ("first100", "later100"):
            for nw in (False, True):
                s = [e for e in hn if e["kind"] == t and e["group"] == g and bool(e["new"]) == nw]
                if s:
                    B["hard_negatives"][f"{t}|{g}|{'new' if nw else 'existing'}"] = {
                        "n": len(s), "n_edits": len({e["edit"] for e in s}), "valid": sum(bool(e["y0_valid"]) for e in s),
                        "same_category": sum(bool(e["same_category"]) for e in s), "related_flag": sum(bool(e["related_flag"]) for e in s),
                        "related_flag_valid": sum(bool(e["related_flag"]) and bool(e["y0_valid"]) for e in s),
                        "degenerate": sum(any(e.get(k) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")) for e in s),
                        "deg_none_no": sum(bool(e.get("deg_none_no")) for e in s), "shifted": sum(e["lp_shift"] not in (0, None) for e in s),
                        "source_related_to_a_stream_edit": sum(e.get("src_related_edit_first_pos") is not None for e in s),
                        "question_is_an_edit_question": sum(bool(e.get("q_in_edit")) for e in s)}
    B["hard_negatives_per_edit_valid_hist"] = {}
    for t in ("hn_i", "hn_q"):
        for g in ("first100", "later100"):
            per = collections.Counter(e["edit"] for e in hn if e["kind"] == t and e["group"] == g and e["y0_valid"])
            eds = {e["edit"] for e in hn if e["kind"] == t and e["group"] == g}
            B["hard_negatives_per_edit_valid_hist"][f"{t}|{g}"] = {str(k): v for k, v in sorted(collections.Counter(per.get(x, 0) for x in eds).items())}
    # new HN-I rows: does the unrestricted draw return questions the pre-edit model answers with the edit's own entity?
    nhi = [e for e in hn if e["kind"] == "hn_i" and e["new"]]
    B["hn_i_new_related_by_category"] = {("same_category" if sc else "other_category"): {
        "n": sum(bool(e["same_category"]) == sc for e in nhi), "related": sum(bool(e["same_category"]) == sc and bool(e["related_flag"]) for e in nhi)}
        for sc in (True, False)}
    ohi = [e for e in hn if e["kind"] == "hn_i" and not e["new"]]
    B["hn_i_existing_related"] = {"n": len(ohi), "related": sum(bool(e["related_flag"]) for e in ohi)}
    # edit sequences that are the same model input as another sequence of the SAME edit (e.g. I-Gen image == Rel image)
    byq = collections.defaultdict(list)
    for e in idx:
        if e["kind"] in ("rel", "tgen", "igen"):
            byq[(e["family"], e["edit"], e["query_id"])].append(e["kind"])
    same = collections.defaultdict(list)
    for (f, ed, _), ks in byq.items():
        if len(ks) > 1:
            same[f"{f}|{'='.join(sorted(ks))}"].append(ed)
    B["edits_whose_sequences_coincide"] = {k: sorted(v) for k, v in sorted(same.items())}
    # edit sequences that are the same query as a sequence of ANOTHER edit of the family
    oth = collections.defaultdict(set)
    for e in idx:
        if e["kind"] in ("rel", "tgen", "igen"):
            oth[(e["family"], e["kind"], e["query_id"])].add(e["edit"])
    B["edit_sequences_shared_between_edits"] = {}
    for (f, k, _), eds in oth.items():
        if len(eds) > 1:
            B["edit_sequences_shared_between_edits"].setdefault(f"{f}|{k}", []).append(sorted(eds))
    # benign pool
    be = [e for e in idx if e["kind"] == "benign"]
    B["benign"] = {}
    for pt in ("image", "text"):
        s = [e for e in be if e["ptype"] == pt]
        if s:
            a = [e for e in s if e["answer_subset"]]
            B["benign"][pt] = {"n": len(s), "n_source_items": len({e["uid"] for e in s}), "subset": len(a), "subset_valid": sum(bool(e["y0_valid"]) for e in a),
                               "subset_degenerate": sum(any(e.get(k) for k in ("deg_empty", "deg_single_char", "deg_qmark", "deg_none_no")) for e in a),
                               "subset_deg_none_no": sum(bool(e.get("deg_none_no")) for e in a),
                               "subset_pre_correct_long": sum(bool(e.get("pre_correct_long")) for e in a),
                               "subset_shifted": sum(e["lp_shift"] not in (0, None) for e in a),
                               "q_in_edit": sum(bool(e["q_in_edit"]) for e in s), "img_in_edit": sum(bool(e["img_in_edit"]) for e in s),
                               "q_in_probe": sum(bool(e["q_in_probe"]) for e in s), "img_in_probe": sum(bool(e["img_in_probe"]) for e in s),
                               "query_in_probe": sum(bool(e["query_in_probe"]) for e in s),
                               "query_in_probe_of_another_family": sum(any(v["query_in_probe"] for f, v in e["overlap"].items() if f != e["family"]) for e in s),
                               "q_in_edit_first_pos_lt": {str(t): sum(e["q_in_edit_first_pos"] is not None and e["q_in_edit_first_pos"] < t for e in s)
                                                          for t in (100, 200, 500, 1000, 2000)}}
    sel = os.path.join(C.WORK, bb, "select", "select_summary.json")
    if os.path.exists(sel):
        ss = C.jload(sel)
        B["benign_exists"] = ss["benign"]["exists"]
        B["hn_source_pool"] = {k: ss["hard_negatives"]["source_pool"][k] for k in ("n_items", "excluded", "n_edit_items_excluded_against", "n_distinct_questions")}
        B["hn_later100_per_prefix"] = ss["hard_negatives"]["later100_per_prefix"]
        B["hn_naive_same_category_share"] = ss["hard_negatives"]["naive_same_category_share_mean_HN-I_candidates"]
    for mode in ("gpu", "dev", "cpu"):
        p = os.path.join(d, f"verify_{mode}.json")
        if os.path.exists(p):
            v = C.jload(p)
            if mode == "cpu":
                B["verify_cpu"] = {k: v[k] for k in ("solver_keys", "existing_hn_keys", "pre_edit_flags_vs_stored", "probe_y0", "decoded_here",
                                                      "existing_hn_y0", "consistency", "shifted_positions", "library_reader", "served_logs",
                                                      "tf_key_vs_decode_step_of_same_length", "probe_pools_by_content") if k in v}
            else:
                B[f"verify_{mode}"] = {k: v[k] for k in ("n_candidates", "handles", "sample", "extra_shifted", "n_shifted_in_sample", "gpu",
                                                         "cuda_visible", "vision_memo", "seconds", "time")}
    out["backbones"][bb] = B
dst = os.path.join(C.WORK, f"table_{args.tag}") if args.tag else C.OUT
os.makedirs(dst, exist_ok=True)
C.jdump(out, os.path.join(dst, "cache_table.json"))

# ----------------------------------------------------------------------------- plain text
L = []


def table(head, rows):
    w = [max(len(str(r[i])) for r in [head] + rows) for i in range(len(head))]
    L.append("  ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
    L.append("  ".join("-" * w[i] for i in range(len(head))))
    for r in rows:
        L.append("  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r)))
    L.append("")


L.append("KEY CACHES OF THE FINAL EXPERIMENTS (10_final/cache) -- rendered by code/fc_table.py from index.jsonl / verify_*.json")
L.append("")
for bb, B in out["backbones"].items():
    m = B["meta"]
    fams_txt = ", ".join("{} = {} ({})".format(f, v["dataset"], v["master_log"]) for f, v in m["families"].items())
    L.append(f"=== {bb}: layer {m['layer']}, d = {m['d_in']}, module dtype {m['module_dtype']}, complete = {m['complete']} "
             f"({m['n_missing_sequences']} sequences missing); families {fams_txt}")
    L.append(f"    keys_tf {B['files']['keys_tf']} (last {B['keys_tf_shifted_extra_rows']} rows: shifted sequences at the decode's token), "
             f"keys_decode {B['files']['keys_decode']}, keys_decode_steps {B['files']['keys_decode_steps']}; capture {B['capture']['gpu_seconds']:.0f} GPU s, {B['capture']['n_parts']} parts")
    L.append("")
    L.append("Sequences per kind (y0 = decoder-v2 pre-edit answer; valid = reproduced by the pre-edit model under teacher forcing; degenerate = empty / "
             "single character / '?' / bare None or No, counted among rows with a y0; dup = sequences whose query (ptype, question string, image "
             "content) also occurs in another sequence of the same family and kind / anywhere in this backbone's cache; shifted = lp_shift != 0)")
    rows = []
    for k, c in list(B["by_family_kind"].items()) + [("all", B["by_kind"]["all"])]:
        dg = c["degenerate_y0"]
        rows.append([k, c["n"], c["n_tf_key"], c["n_decode_key"], c["n_with_y0"], c["n_valid_y0"],
                     f"{dg['deg_empty']}/{dg['deg_single_char']}/{dg['deg_qmark']}/{dg['deg_none_no']}", c["n_in_duplicate_group_within_family_and_kind"],
                     c["n_in_duplicate_query_group"], c["n_distinct_queries"], c["n_shifted"], c["n_decode_keys_not_all_identical"], c["max_distinct_decode_keys"]])
    table(["family|kind", "rows", "tf key", "decode key", "with y0", "valid y0", "degen e/1/?/None-No", "dup (same family+kind)", "dup (whole cache)",
           "distinct queries", "shifted", "decode keys vary", "max distinct"], rows)
    if B["pre_edit_flags"]:
        L.append("Pre-edit teacher-forced flags of the edit sequences (target already produced by the pre-edit model, B = 1)")
        table(["family|metric", "edits", "pre exact", "pre token %", "non-finite", "pre exact, first 100"],
              [[k, v["n"], v["n_pre_exact"], f"{v['mean_pre_token_acc_pct']:.2f}", v["n_nonfinite"], v["first100_n_pre_exact"]] for k, v in B["pre_edit_flags"].items()])
    if B["probe_pools"]:
        L.append("Probe rows per pool (valid rows; harness split by item, seed 20261109)")
        rows = []
        for f, P in B["probe_pools"].items():
            for pool, v in P.items():
                rows.append([f"{f}|{pool}", v["rows"], v["valid"], v["image_valid"], v["text_valid"], v["distinct_queries_valid"]])
        table(["family|pool", "rows", "valid", "image valid", "text valid", "distinct valid queries"], rows)
    if B["hard_negatives"]:
        L.append("Hard negatives (same category = the question shares a category with the edit's question / rephrase; related = the pre-edit answer names "
                 "an entity of the edit(s) or equals the edit prompt's own pre-edit answer; src related = the source item of an existing row is related "
                 "to some edit of the long stream)")
        table(["type|group|origin", "rows", "edits", "valid", "same category", "related", "degenerate", "None/No", "shifted", "src related", "q is an edit q"],
              [[k, v["n"], v["n_edits"], v["valid"], v["same_category"], v["related_flag"], v["degenerate"], v["deg_none_no"], v["shifted"],
                v["source_related_to_a_stream_edit"], v["question_is_an_edit_question"]] for k, v in B["hard_negatives"].items()])
        L.append("Valid hard negatives per edit (histogram: number of valid rows -> edits): " +
                 "; ".join(f"{k}: {v}" for k, v in B["hard_negatives_per_edit_valid_hist"].items()))
        if "hn_source_pool" in B:
            L.append(f"Source pool: {B['hn_source_pool']}; later100 edits per prefix: {B['hn_later100_per_prefix']}; "
                     f"share of unrestricted HN-I candidates in the edit's category: {B['hn_naive_same_category_share']:.3f}")
        L.append("")
    if B["benign"]:
        L.append("Benign pool (label-free; subset = rows with a decoder-v2 pre-edit answer)")
        table(["row type", "rows", "source items", "subset", "subset valid", "subset degenerate", "subset None/No", "subset shifted",
               "q in edits", "image in edits", "q in probes", "image in probes", "query in probes", "query in probes (other family)"],
              [[pt, v["n"], v["n_source_items"], v["subset"], v["subset_valid"], v["subset_degenerate"], v["subset_deg_none_no"], v["subset_shifted"],
                v["q_in_edit"], v["img_in_edit"], v["q_in_probe"], v["img_in_probe"], v["query_in_probe"], v["query_in_probe_of_another_family"]]
               for pt, v in B["benign"].items()])
        if "benign_exists" in B:
            L.append(f"What exists: {B['benign_exists']}")
            L.append("")
    for mode, label in (("verify_gpu", "fresh capture, seeded sample, master-log handles"), ("verify_dev", "fresh capture through the development log's handle")):
        if mode in B:
            v = B[mode]
            s = v["sample"]
            L.append(f"Verification ({label}; GPU {v['gpu']}, no vision memo): {s['n_sequences_all_bitwise']} of {s['n']} sequences bitwise identical "
                     f"in every key {s['by_kind']}; teacher-forced key {s['tf_key_equal']['n_true']}/{s['tf_key_equal']['n']}, prompt-only key "
                     f"{s['decode_key_equal']['n_true']}/{s['decode_key_equal']['n']}, 11 decode forwards {s['decode_steps_equal']['n_true']}/{s['decode_steps_equal']['n']}, "
                     f"shifted-token key {s['tf_prompt_key_equal']['n_true']}/{s['tf_prompt_key_equal']['n']}, eligible_keys(answer='all') "
                     f"{s['eligible_all_equal']['n_true']}/{s['eligible_all_equal']['n']}, positions {s['lp_tf_equal']['n_true']}/{s['lp_tf_equal']['n']} + "
                     f"{s['lp_decode_equal']['n_true']}/{s['lp_decode_equal']['n']}; shifted sequences in the sample {v['n_shifted_in_sample']}"
                     + ("" if not v["extra_shifted"] else f"; further shifted sequences checked: {v['extra_shifted']['n_sequences_all_bitwise']} of {v['extra_shifted']['n']} bitwise")
                     + (f"; shifted sequences of the cache NOT re-captured: {B['by_kind']['all']['n_shifted'] - v['n_shifted_in_sample'] - (v['extra_shifted']['n'] if v['extra_shifted'] else 0)}"
                        if mode == "verify_gpu" else ""))
    if "verify_cpu" in B:
        v = B["verify_cpu"]
        for name, s in v["solver_keys"].items():
            c = s["cosine_solver_key_vs_cached_rel_key"]
            L.append(f"Solver keys of {name} ({s['n_edits']} edits): cosine(stored B = 10 solver key, cached B = 1 Rel key) min {c['min']:.6f} / median {c['median']:.6f} / "
                     f"max {c['max']:.6f}; own coordinate r of the cached key {s['own_coordinate_r_of_cached_key']['min']:.3f} .. {s['own_coordinate_r_of_cached_key']['max']:.3f} "
                     f"({s['own_coordinate_r_of_cached_key']['n_below_1']} below 1); stored single-item key (B = 1, build time) bitwise equal to the cached key: "
                     f"{s['single_item_key_bitwise_equal']['n_equal']} of {s['n_edits']}; edits with lp_shift != 0: {s['n_edits_lp_shift_nonzero']}")
        h = v["existing_hn_keys"]
        L.append(f"Existing hard-negative keys (P11 keys.pt): {h['n_bitwise_equal']} of the {h['n_in_cache']} cached rows ({h['n']} existing) bitwise equal to the cached teacher-forced keys")
        for f, s in v["pre_edit_flags_vs_stored"].items():
            L.append(f"Pre-edit Rel flags, family {f}: exact here (B = 1) {s['pre_exact_B1_here']} vs stored (batch 8) {s['pre_exact_batch8_stored']} of {s['n_rel']}; "
                     f"exact flags differing {s['n_exact_flag_differs']}, token accuracies differing {s['n_token_acc_differs_gt_1e-6']}")
        c = v["consistency"]
        L.append(f"Consistency: probe y0 validity vs library TF at W_0 differing {v['probe_y0']['n_pre_exact_differs_from_y0_valid']} of {v['probe_y0']['n_with_tf']}; "
                 f"answers decoded here: {v['decoded_here']['n_pre_exact_differs_from_y0_valid']} differing of {v['decoded_here']['n']}; existing hard negatives: "
                 f"{v['existing_hn_y0']['n_pre_exact_differs_from_y0_valid']} of {v['existing_hn_y0']['n']}; duplicate-query groups with bitwise identical decode keys "
                 f"{c['duplicate_query_groups']['n_groups_decode_key_bitwise_identical']} of {c['duplicate_query_groups']['n_groups']}; position identity violations "
                 f"{c['n_position_identity_violations(lp_tf - lp_decode != lp_shift)']}; all keys finite {c['all_finite']}; pre-edit logits non-finite on "
                 f"{c.get('n_teacher_forced_sequences_with_nonfinite_pre_edit_logits')} sequences; empty answers {c.get('n_empty_answers')}; teacher-forced key bitwise equal to the prompt-only key "
                 f"(same token) {c['tf_key_equals_decode_key(same token, other length)']['n_bitwise_equal']} of {c['tf_key_equals_decode_key(same token, other length)']['n_with_tf_and_shift_0']}")
        if "tf_key_vs_decode_step_of_same_length" in v:
            s = v["tf_key_vs_decode_step_of_same_length"]
            L.append(f"Two independent captures agree: the teacher-forced key (prompt + n answer tokens) is bitwise the decode key of the forward 'prompt + n ids' on "
                     f"{s['n_bitwise_equal']} of {s['n_checked']} sequences with the same last prompt token and n <= 10 {s['by_kind']} "
                     f"({s['n_longer_than_10_tokens(not checkable)']} longer answers not checkable)")
        if "shifted_positions" in v:
            s = v["shifted_positions"]
            dv = s["development_log_sequences(300 edit sequences + valid probes + valid existing hard negatives)"]
            cs = s["tf_key_vs_tf_prompt_key_cosine_on_shifted"]
            L.append(f"Shifted positions: {s['n_shifted']} of {s['n_with_tf']} teacher-forced sequences {s['shifted_by_kind']} (answers starting with a digit: "
                     f"{s['shifted_answers_start_with_digit']}; digit-leading answers NOT shifted: {s['answers_starting_with_digit_not_shifted']}); tokenizer recount differs on "
                     f"{s['n_recount_differs_from_cached_lp_shift']}; on the development log's sequences: {dv['n_shifted']} of {dv['n']} (shifted Rel edits {dv['shifted_rel_edits']})"
                     + ("" if cs is None else f"; cosine(key at the teacher-forced position, key at the decode's token) on shifted sequences {cs['min']:.3f} .. {cs['max']:.3f}; "
                        f"shifted-token key bitwise equal to the prompt-only key: {s['tf_prompt_key_equals_decode_key_on_shifted'][0]} of {s['tf_prompt_key_equals_decode_key_on_shifted'][1]}"))
        if "library_reader" in v:
            s = v["library_reader"]
            if s.get("ok"):
                fb = s["find_by_content"]
                L.append(f"Library reader (rescope.keycache.KeyCache on this directory): rows {s['n_equal']}, keys tf / prompt / decode / steps equal "
                         f"{s['keys_tf_equal']} / {s['keys_prompt_equal']} / {s['keys_decode_equal']} / {s['keys_steps_equal']}; family of the development log's "
                         f"uids {s['family_of_development_log_uids']} (expected {s['family_expected']}); rows(rel, T=100) {s['n_rows_rel_T100']}, hard negatives "
                         f"T=100 {s['n_rows_hn_T100']}, valid REPAIR probes {s['n_rows_probe_repair_valid']}, valid HELD-OUT-DISJOINT probes "
                         f"{s['n_rows_probe_heldout_disjoint_valid']}, benign {s['n_rows_benign']}; find by content {fb['n_same_sequence_or_duplicate']} of {fb['n']}")
            else:
                L.append(f"Library reader (rescope.keycache.KeyCache): FAILED {s.get('error')}")
        if "served_logs" in v:
            sv = v["served_logs"]
            L.append("Logs served by this cache (edit stream = prefix of a cached family; probe rows equal to the cached ones): " +
                     "; ".join(f"{n} ({r['family']}, T = {r['T']}" + ("" if r["n_probe_rows"] is None else f", probe rows differing {r['n_probe_rows_differing_from_cache']}") +
                               ("" if r["registered"] else ", not registered yet") + ")" for n, r in sv["served"].items()))
            if sv["not_served"]:
                L.append("Logs of this backbone NOT served by position: " + "; ".join(f"{n} (T = {r['T']}: {r['reason']})" for n, r in sv["not_served"].items()))
    L.append("")
with open(os.path.join(dst, "cache_table.txt"), "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))

# ----------------------------------------------------------------------------- short table (final message)
S = []
bbs = list(out["backbones"])
short = {"qwen25vl": "Q", "llava": "L"}
cols = ("rows", "valid/y0", "dup", "shift")
head = ["kind"] + [f"{short.get(bb, bb)} {c}" for bb in bbs for c in cols]
rows = []
for k in list(C.KINDS) + ["all"]:
    r = [k]
    for bb in bbs:
        B = out["backbones"][bb]
        c = B["by_kind"].get(k)
        if c is None:
            r += ["-"] * len(cols)
            continue
        fam = [B["by_family_kind"][f"{f}|{k}"]["n"] for f in B["meta"]["families"] if f"{f}|{k}" in B["by_family_kind"]]
        r += ["+".join(str(x) for x in fam) if len(fam) > 1 else c["n"],
              f"{c['n_valid_y0']}/{c['n_with_y0']}" if c["n_with_y0"] else "-", c["n_in_duplicate_query_group"], c["n_shifted"]]
    rows.append(r)
w = [max(len(str(x[i])) for x in [head] + rows) for i in range(len(head))]
S.append(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(head)))
for r in rows:
    S.append(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
S.append("Q = qwen25vl (a+b: VLKEB + MMKE-Bench family), L = llava; valid/y0 = decoder-v2 pre-edit answers reproduced by the pre-edit model / rows with "
         "an answer; dup = sequences whose query occurs more than once in the backbone's cache; shift = last prompt position differs between teacher "
         "forcing and decoding")
for bb in bbs:
    B = out["backbones"][bb]
    S.append(f"{short.get(bb, bb)} degenerate answers (empty / 1 char / ? / bare None-No): " + "; ".join(
        f"{k} {c['degenerate_y0']['deg_empty']}/{c['degenerate_y0']['deg_single_char']}/{c['degenerate_y0']['deg_qmark']}/{c['degenerate_y0']['deg_none_no']}"
        for k, c in B["by_kind"].items() if k != "all" and c["n_with_y0"]))
with open(os.path.join(dst, "cache_table_short.txt"), "w") as f:
    f.write("\n".join(S) + "\n")
print("\n".join(S))

# ----------------------------------------------------------------------------- directory guide (plain text)
if not args.tag:
    G = ["10_final/cache -- key caches, benign pool and hard negatives of the final experiments (written by code/fc_table.py)", "",
         "  <backbone>/                 qwen25vl (layer 20), llava (layer 16): index.jsonl, keys_tf.pt, keys_decode.pt, keys_decode_steps.pt,",
         "                              pre_edit_flags.jsonl, FORMAT.txt (the format), meta.json, verify_{gpu,dev,cpu}.json",
         "  hard_negatives/<backbone>/  rows.jsonl (existing + new rows: HN-I, HN-Q, HN-X), keys.pt, edit_pre.jsonl, README.txt",
         "                              (the file format of 08_rescope's hard negatives; rescope.evaluate.HN_DIR_FINAL)",
         "  benign/<backbone>/          rows.jsonl (label-free benign pool; decoder-v2 answers of the seeded subset), README.txt",
         "  cache_table.json / .txt     what the caches contain, verification results (cache_table_short.txt: one compact table)",
         "  code/                       fc_select.py (CPU selection) -> fc_capture.py (GPU, resumable parts) -> fc_assemble.py -> fc_verify.py -> fc_table.py",
         "  work/<backbone>/select/     the selection in use: manifest.jsonl (every sequence), hn.jsonl, benign.jsonl, hn_edits.json, select_summary.json",
         "  work/<backbone>/parts/      capture parts (one file per 100 sequences; resumable)",
         "  logs/                       capture and verification logs", "",
         "Reader: rescope.keycache.KeyCache('<backbone>').  One cache per backbone serves every log whose edit stream is a prefix of a cached",
         "family (list: cache_table.txt, 'Logs served').  Rebuild: fc_select.py is deterministic (code/fc_check_select.sh compares a re-run byte",
         "for byte); fc_capture.py skips finished parts; fc_assemble.py publishes each directory with one rename.", ""]
    for bb, B in out["backbones"].items():
        G.append(f"  {bb}: {B['meta']['n_sequences']} sequences, complete = {B['meta']['complete']}; keys_tf {B['files']['keys_tf']}, keys_decode {B['files']['keys_decode']}, "
                 f"keys_decode_steps {B['files']['keys_decode_steps']}")
    G += ["", "NOT in these caches: the text family (CounterFact logs TB-Q{100,500}-cf-l2_3).  Its keys already exist next to the logs and were not",
          "captured again: 06_testbed_v3/<log>/scope_keys.pt + scope_rows.jsonl (edit prompts, paraphrase witnesses, neighbourhood negatives) and",
          "probe_keys.pt.  They hold the TEACHER-FORCED key at the last prompt position only (no prompt-only key, no decode-step keys)."]
    for name in ("TB-Q100-cf-l2_3", "TB-Q500-cf-l2_3"):
        pth = os.path.join(C.EXP, "06_testbed_v3", name, "scope_rows.jsonl")
        if os.path.exists(pth):
            kc_ = collections.Counter(r["kind"] for r in C.jsonl_read(pth))
            G.append(f"  {name}: scope rows {dict(sorted(kc_.items()))}")
    with open(os.path.join(dst, "README.txt"), "w") as f:
        f.write("\n".join(G) + "\n")

# ----------------------------------------------------------------------------- summary sentences (numbers for the final message)
M = []
for bb, B in out["backbones"].items():
    tag = short.get(bb, bb)
    be, hn_ = B.get("benign", {}), B.get("hard_negatives", {})
    if be:
        ex = B.get("benign_exists", {})
        M.append(f"[{tag}] benign: " + ", ".join(f"{v['n']} {pt} rows ({v['subset_valid']}/{v['subset']} subset answers valid, {v['subset_degenerate']} degenerate)" for pt, v in be.items())
                 + f"; exists: {ex.get('eval_items_eligible')} eligible eval items, {ex.get('eval_distinct_image_rows_not_probe_queries')} distinct image rows that are not probe queries "
                   f"({ex.get('eval_distinct_image_rows_that_are_probe_queries(excluded)')} more are probe queries, excluded), {ex.get('eval_distinct_text_rows_not_probe_queries')} text rows; "
                 + "; ".join(f"{pt}: question string also an edit question {v['q_in_edit']}, also a probe question {v['q_in_probe']}, image among edits {v['img_in_edit']} / probes {v['img_in_probe']}, "
                             f"same query as a probe of another family {v['query_in_probe_of_another_family']}" for pt, v in be.items()))
    if hn_:
        def tot(t, key, new=None):
            return sum(v[key] for k, v in hn_.items() if k.startswith(t + "|") and (new is None or k.endswith("new" if new else "existing")))
        M.append(f"[{tag}] hard negatives: " + "; ".join(
            f"{t}: {tot(t, 'n', False)} existing + {tot(t, 'n', True)} new, valid {tot(t, 'valid')}/{tot(t, 'n')}, degenerate {tot(t, 'degenerate')} (None/No {tot(t, 'deg_none_no')}), "
            f"same category among new {tot(t, 'same_category', True)}/{tot(t, 'n', True)}, related {tot(t, 'related_flag')}, shifted {tot(t, 'shifted')}"
            for t in ("hn_i", "hn_q", "hn_x"))
            + f"; existing rows whose source item is related to a long-stream edit: {tot('hn_i', 'source_related_to_a_stream_edit', False)} HN-I, "
              f"{tot('hn_q', 'source_related_to_a_stream_edit', False)} HN-Q; source pool {B.get('hn_source_pool', {}).get('n_items')} eval items")
    if hn_:
        r_ = B["hn_i_new_related_by_category"]
        M.append(f"[{tag}] new HN-I rows whose pre-edit answer is 'related' to the edit: {r_['same_category']['related']}/{r_['same_category']['n']} of the same-category rows, "
                 f"{r_['other_category']['related']}/{r_['other_category']['n']} of the other-category rows (existing, category-restricted rows: "
                 f"{B['hn_i_existing_related']['related']}/{B['hn_i_existing_related']['n']})")
    if B["edits_whose_sequences_coincide"] or B["edit_sequences_shared_between_edits"]:
        M.append(f"[{tag}] edit sequences that are the same model input: within one edit {B['edits_whose_sequences_coincide']}; shared between edits "
                 + str({k: (v if len(v) <= 6 else f'{len(v)} groups, e.g. {v[:3]}') for k, v in B['edit_sequences_shared_between_edits'].items()}))
    for mode in ("verify_gpu", "verify_dev"):
        if mode in B:
            s = B[mode]["sample"]
            xs = B[mode]["extra_shifted"]
            M.append(f"[{tag}] {mode}: {s['n_sequences_all_bitwise']}/{s['n']} sampled sequences bitwise identical (tf {s['tf_key_equal']['n_true']}/{s['tf_key_equal']['n']}, "
                     f"decode {s['decode_key_equal']['n_true']}/{s['decode_key_equal']['n']}, 11 steps {s['decode_steps_equal']['n_true']}/{s['decode_steps_equal']['n']}, "
                     f"eligible_keys('all') {s['eligible_all_equal']['n_true']}/{s['eligible_all_equal']['n']})"
                     + ("" if not xs else f"; + {xs['n_sequences_all_bitwise']}/{xs['n']} further shifted sequences")
                     + (f" ({B['by_kind']['all']['n_shifted'] - B[mode]['n_shifted_in_sample'] - (xs['n'] if xs else 0)} shifted sequences not re-captured)" if mode == "verify_gpu" else "")
                     + f"; handles {', '.join(h['log'] for h in B[mode]['handles'].values())}")
    v = B.get("verify_cpu")
    if v:
        for name, s in v["solver_keys"].items():
            c, r = s["cosine_solver_key_vs_cached_rel_key"], s["own_coordinate_r_of_cached_key"]
            M.append(f"[{tag}] solver keys {name}: cosine {c['min']:.5f} .. {c['max']:.5f} (median {c['median']:.5f}, n = {s['n_edits']}); r of the B = 1 key {r['min']:.3f} .. {r['max']:.3f}, "
                     f"{r['n_below_1']} below 1; build-time B = 1 keys bitwise equal {s['single_item_key_bitwise_equal']['n_equal']}/{s['n_edits']}")
        sl, sp, c = v.get("tf_key_vs_decode_step_of_same_length"), v.get("shifted_positions"), v["consistency"]
        M.append(f"[{tag}] other checks: existing hard-negative keys bitwise {v['existing_hn_keys']['n_bitwise_equal']}/{v['existing_hn_keys']['n_in_cache']}; "
                 + ("" if not sl else f"tf key == decode key of the same length {sl['n_bitwise_equal']}/{sl['n_checked']}; ")
                 + f"y0 validity vs TF at W_0 differing: probes {v['probe_y0']['n_pre_exact_differs_from_y0_valid']}/{v['probe_y0']['n_with_tf']}, existing hard negatives "
                   f"{v['existing_hn_y0']['n_pre_exact_differs_from_y0_valid']}/{v['existing_hn_y0']['n']}, decoded here {v['decoded_here']['n_pre_exact_differs_from_y0_valid']}/{v['decoded_here']['n']}; "
                   f"duplicate-query groups with identical decode keys {c['duplicate_query_groups']['n_groups_decode_key_bitwise_identical']}/{c['duplicate_query_groups']['n_groups']}"
                 + ("" if not sp else f"; shifted {sp['n_shifted']}/{sp['n_with_tf']} {sp['shifted_by_kind']}, shifted Rel edits {sp['shifted_rel_edits_long_stream']}"))
        for f, s in v["pre_edit_flags_vs_stored"].items():
            M.append(f"[{tag}] pre-edit flags {f}: " + ", ".join(f"{m} exact {a}/{b}" for m, (a, b) in s["by_metric_pre_exact"].items())
                     + f"; Rel exact B = 1 vs stored batch 8: {s['pre_exact_B1_here']} vs {s['pre_exact_batch8_stored']}, flags differing {s['n_exact_flag_differs']}, "
                       f"token accuracies differing {s['n_token_acc_differs_gt_1e-6']} (mean {100 * s['mean_pre_token_acc_here']:.2f} % here vs "
                       f"{100 * s['mean_pre_token_acc_stored']:.2f} % stored), non-finite {s['n_nonfinite']}")
        for f, s in (v.get("probe_pools_by_content") or {}).items():
            bq = s["n_probe_rows_equal_to_a_benign_row"]
            M.append(f"[{tag}] probe pools {f} by content (question + image md5): HELD-OUT-DISJOINT rows equal to a REPAIR row of the family "
                     f"{s['n_heldout_disjoint_rows_equal_by_content_to_a_REPAIR_row_of_the_family']}/{s['n_heldout_disjoint_rows']}, to a REPAIR row of another family "
                     f"{s['n_heldout_disjoint_rows_equal_to_a_REPAIR_row_of_another_family']}; probe rows equal to a benign row {bq['all']} (REPAIR {bq['repair']}, "
                     f"HELD-OUT-DISJOINT {bq['heldout_disjoint']}); distinct queries by content {s['n_distinct_queries_by_content']} vs by path (library) "
                     f"{s['n_distinct_queries_by_path(library)']}: {s['content_groups_spanning_several_library_groups']}")
        if "served_logs" in v:
            M.append(f"[{tag}] serves: " + ", ".join(v["served_logs"]["served"]) + ("" if not v["served_logs"]["not_served"] else
                                                                                   "; not served by position: " + ", ".join(
                                                                                       f"{n} ({'same items, other order' if r.get('items_are_a_subset_of_family') else 'other stream'})"
                                                                                       for n, r in v["served_logs"]["not_served"].items())))
with open(os.path.join(dst, "cache_summary.txt"), "w") as f:
    f.write("\n".join(M) + "\n")
print("\n".join(M))
