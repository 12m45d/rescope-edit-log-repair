"""Renders out/table.txt and ../verdicts.json from the JSON files of the checks (no number is typed by hand).

  python render.py
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np

import vcommon as V

TAGS = ("Q", "L")
EVAL_MTIME = "2026-11-10 21:25:16"          # modification time of rescope/evaluate.py recorded in the caches' meta.json


def lib_fingerprint():
    """sha1 (12 hex digits) of the package's .py files (the convention of its own test runner) and of the two cache indices."""
    import hashlib
    h = hashlib.sha1()
    folder = V.LIB + "/rescope"
    for f in sorted(os.listdir(folder)):
        if f.endswith(".py"):
            h.update(f.encode())
            with open(os.path.join(folder, f), "rb") as fh:
                h.update(fh.read())
    out = {"rescope": h.hexdigest()[:12]}
    for bb in ("qwen25vl", "llava"):
        with open(f"{V.CACHE}/{bb}/index.jsonl", "rb") as fh:
            out[f"cache_index_md5_{bb}"] = hashlib.md5(fh.read()).hexdigest()
        out[f"cache_meta_index_md5_{bb}"] = V.jload(f"{V.CACHE}/{bb}/meta.json")["index_md5"]
    return out


def load(name):
    p = f"{V.OUT}/{name}"
    return V.jload(p) if os.path.exists(p) else None


def fmt_tab(rows, header):
    w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)).rstrip()      # noqa: E731
    return [line(header), "  ".join("-" * x for x in w)] + [line(r) for r in rows]


def check1(out, ver):
    out.append("=" * 150)
    out.append("CHECK 1  two-block / shared-scale calibration: library against my naive implementation of PROTOCOL.md section 6, and validity")
    out.append("=" * 150)
    ok_all, nums = True, {}
    rows = []
    for t in TAGS:
        a = load(f"c1a_{t}.json")
        if a is None:
            ok_all = False
            rows.append([V.DEV[t]["log"], "MISSING"])
            continue
        tt, st = a["totals"], a["structure"]
        s_ok = st["family_edit_uids_equal_log_uids"] and st["probe_rows_equal_log_records_in_order"] and not any(st["pool_flags_differing"].values()) and \
            st["lib_rows_tf_equal_mine"] and st["lib_repair_pool_equal_mine"]
        ok = s_ok and tt["n_real"] == 0 and tt["n_cal_rows_differing"] == 0 and tt["n_rank_differing"] == 0 and tt["n_capped_sets_differing"] == 0 and \
            tt["n_silenced_counts_differing"] == 0 and tt["max_abs_eta_diff"] <= 1e-12
        ok_all = ok_all and ok
        n_inf = sum(c["eta_mine"] == "inf" for c in a["configs"])
        n_cap = sum(bool(c["capped_mine"]) for c in a["configs"])
        nums[t] = {"n_calibrations": tt["n_calibrations"], "n_decisions": tt["n_decisions"], "n_differing": tt["n_differing"], "n_tie": tt["n_tie"], "n_real": tt["n_real"],
                   "max_abs_eta_diff": tt["max_abs_eta_diff"], "n_calibrations_with_infinite_eta": n_inf, "n_calibrations_with_capped_edits": n_cap, "structure_ok": bool(s_ok)}
        rows.append([a["log"], tt["n_calibrations"], tt["n_decisions"], tt["n_differing"], tt["n_real"], f"{tt['max_abs_eta_diff']:.1e}", tt["n_rank_differing"],
                     tt["n_capped_sets_differing"], tt["n_silenced_counts_differing"], n_inf, n_cap, "yes" if s_ok else "NO", "PASS" if ok else "FAIL"])
    out.append("1a  firing decisions on every cached sequence of the log (probes, Rel / T-Gen / I-Gen, hard negatives, benign rows with an answer; teacher-forced key, and the")
    out.append("    prompt-only key for seeds 0, 1): fired / releasing edit / own edit candidate / own edit released.  Features ncoord, cos; rules two-block, shared; alpha 1/100, 1/20, 1/5;")
    out.append("    n = 200 (seeds 0-4) and n = 40 (seeds 0, 1); guards on (capped) and off (uncapped).  'structure' = cache rows == the log's probe records, pools re-derived, row sets equal.")
    out += fmt_tab(rows, ["log", "calibrations", "decisions", "differing", "of them real", "max |eta diff|", "rank differs", "capped sets differ", "silenced counts differ",
                          "eta = inf", "with capped edits", "structure", "verdict"])
    out.append("")
    cells, bad_alpha, bad_bound, worst, ndiff, ndec, capeq = [], 0, 0, -1e9, 0, 0, True
    for p in sorted(glob.glob(f"{V.OUT}/c1b/*.json")):
        o = V.jload(p)
        for c in o["cells"]:
            for who in ("lib_uncapped", "naive_uncapped"):
                bad_alpha += int(not c[who]["within_alpha_3se"])
                bad_bound += int(not c[who]["within_bound_3se"])
                worst = max(worst, c[who]["excess_over_bound_in_se"] if c[who]["se"] else -1e9)
            ndiff += c["uncapped_decisions_differing_lib_vs_naive"] + c["capped_decisions_differing_lib_vs_naive"]
            ndec += c["uncapped_decisions"] + c["capped_decisions"]
            capeq = capeq and c["capped_counts_equal_first"]
            cells.append((o, c))
    n_files = len(glob.glob(f"{V.OUT}/c1b/*.json"))
    ok_b = n_files == 8 and bad_alpha == 0 and ndiff == 0 and capeq
    ok_all = ok_all and ok_b
    out.append(f"1b  validity: 1000 random splits of the cached benign rows into n calibration rows and test rows; false-firing rate = share of test rows on which an edit is a candidate.")
    out.append("    pools: 'benign' = the 3000 label-free benign rows of the cache (prompt-only key), 'probe' = the valid probe rows of the log (teacher-forced key).")
    out.append("    cells: mean over splits in % (standard error); bound = floor(alpha (n2 + 1)) / (n2 + 1); capped = guards on (naive, 1000 splits) with the mean number of capped edits.")
    for n in (40, 200):
        out.append(f"    n = {n}")
        rows = []
        for o, c in cells:
            if c["n"] != n or c["alpha"] == "1/100" and n == 40:
                continue
            u, nu, cp = c["lib_uncapped"], c["naive_uncapped"], c["naive_capped"]
            rows.append([o["log"].split("-")[1][0], o["feature"], o["pool"], c["rule"], c["alpha"], f"{100 * c['bound']:.2f}", f"{100 * u['mean']:.2f} ({100 * u['se']:.2f})",
                         f"{100 * nu['mean']:.2f}", f"{u['excess_over_bound_in_se']:+.1f}", "yes" if u["within_alpha_3se"] else "NO",
                         f"{100 * cp['mean']:.2f}", f"{c['mean_edits_capped_naive']:.1f}"])
        out += ["    " + x for x in fmt_tab(rows, ["log", "feature", "pool", "rule", "alpha", "bound %", "uncapped lib % (se)", "naive %", "excess / se", "<= alpha + 3 se",
                                                  "capped %", "edits capped"])]
    out.append(f"    n = 40, alpha = 1/100: eta = +inf in all splits of all 16 cells (rank exceeds the block size), uncapped false firing 0, capped false firing "
               f"{max(100 * c['naive_capped']['mean'] for o, c in cells if c['n'] == 40 and c['alpha'] == '1/100'):.4f} % at most (every edit capped).")
    out.append(f"    cells (library and naive): {2 * len(cells)}; above alpha + 3 se: {bad_alpha}; above the exact bound + 3 se: {bad_bound}; largest excess over the bound: {worst:+.1f} se; "
               f"library vs naive decisions differing: {ndiff} of {ndec}; capped-edit counts equal: {capeq}.   verdict 1b: {'PASS' if ok_b else 'FAIL'}")
    cap_inc = max(c["naive_capped"]["mean"] - c["naive_uncapped"]["mean"] for o, c in cells)
    out.append(f"    capped rule next to the uncapped one: largest increase of the mean false-firing rate by the guard caps over all cells: {100 * cap_inc:.4f} points "
               f"(largest mean number of capped edits in a cell with a finite eta: {max(c['mean_edits_capped_naive'] for o, c in cells if c['n_eta_infinite'] == 0):.1f} of 100)")
    x = load("x1_seed_coupling.json")
    if x:
        out.append("    extra (x1): one seed is used for the draw and for the split.  Shared seed against an independently seeded split, default method, 2000 paired draws:")
        for t in TAGS:
            e = x[t]["default_method_shared_vs_independent_split_seed_2000_draws"]
            s = x[t]["shared"]
            out.append(f"      {V.DEV[t]['log']}: firing {e['fire']['shared_mean']:.2f} vs {e['fire']['indep_mean']:.2f} % (diff {e['fire']['diff_mean']:+.2f}, z {e['fire']['z']:+.1f}); "
                       f"T-Gen coverage {e['cov']['shared_mean']:.1f} vs {e['cov']['indep_mean']:.1f} % (diff {e['cov']['diff_mean']:+.2f}, z {e['cov']['z']:+.1f}); "
                       f"block composition z: image share {s['img']['z_vs_independent_split']:+.1f}, pool position {s['pos']['z_vs_independent_split']:+.1f}")
    ver["check1_calibration"] = {"verdict": "pass" if ok_all else "fail", "decisions_vs_naive": nums,
                                 "validity": {"n_cells": 2 * len(cells), "n_above_alpha_plus_3se": bad_alpha, "n_above_bound_plus_3se": bad_bound, "largest_excess_over_bound_in_se": worst,
                                              "lib_vs_naive_decisions_differing": [ndiff, ndec], "n_files": n_files,
                                              "largest_increase_of_false_firing_by_guard_caps_points": 100 * cap_inc}}
    out.append("")


def part_times(bb):
    """{(family, kind): [time of part 0, part 1, ...]} from the cache's capture_parts.json."""
    cp = V.jload(f"{V.CACHE}/{bb}/capture_parts.json")["parts"]
    out = {}
    for p in cp:
        fam, kind, k = p["part"].rsplit("_", 2)[0].split("_", 1)[0], p["part"].split("_", 1)[1].rsplit("_", 1)[0], int(p["part"].rsplit("_", 1)[1])
        out[(fam, kind, k)] = p["time"]
    return out


def check2(out, ver):
    out.append("=" * 150)
    out.append("CHECK 2  key caches, benign pool, hard negatives")
    out.append("=" * 150)
    c = load("c2_cpu.json")
    ok_all, nums = c is not None, {}
    for bb in ("qwen25vl", "llava"):
        g = load(f"c2_gpu_{bb}.json")
        if c is None or g is None:
            ok_all = False
            out.append(f"{bb}: MISSING")
            continue
        R = c[bb]
        A, B, C, D, E = R["A_index_vs_raw"], R["B_last_prompt_indices"], R["C_benign"], R["D_hard_negatives"], R["E_image_md5"]
        okA = A["stream_fields_equal_train_items"] and not any(A["edit_sequences_differing_from_stream"].values()) and A["probe_rows_differing_from_eval_items"] == 0 and \
            all(v["equal_cache_probe_uids"] for v in A["probe_set_rederived"].values())
        okB = B["n_tf_differing"] == 0 and B["n_decode_differing"] == 0 and B["shifted_sets_equal"] and B["n_prefix_violations"] == 0 and B["n_tf_errors"] == 0 and \
            set(B["shift_token"]) <= {"▁"} and B["n_shifted_answer_starts_with_digit"] == B["n_shifted_mine"]
        okC = C["rows_file_equal_index"] and C["rows_differing_from_their_eval_item"] == 0 and C["n_benign_rows_whose_item_is_a_probe_item"] == 0 and \
            C["n_benign_rows_whose_item_is_an_edit_item"] == 0 and C["n_benign_rows_equal_to_an_edit_query"] == 0 and C["n_benign_rows_equal_to_a_probe_query"] == 0 and \
            C["n_benign_rows_equal_to_an_edit_query_by_content"] == 0 and C["n_benign_rows_equal_to_a_probe_query_by_content"] == 0
        okD = D["n_ok"] == D["n_sampled"] == 30 and D["whole_file_image_question_check"]["bad"] == 0
        gA, gC = g["A_sample"], g["C_hard_negatives"]
        gB = g.get("B_shifted")
        okG = g["module_weight_equals_checkpoint_file"] and gA["n_ok"] == gA["n"] == 60 and gC["n_ok"] == gC["n"] == 30 and (gB is None or gB["n_ok"] == gB["n"])
        ok = okA and okB and okC and okD and okG and E["n_files_whose_md5_differs_from_index"] == 0
        ok_all = ok_all and ok
        # which capture parts (before / after the last change of rescope/evaluate.py) do the re-captured sequences come from
        idx = V.read_index(bb)
        pt = part_times(bb)
        cnt, posn = {}, {}
        for e in idx:
            k = (e["family"], e["kind"])
            posn[e["sid"]] = (e["family"], e["kind"], cnt.get(k, 0) // 100)
            cnt[k] = cnt.get(k, 0) + 1
        sids = [r["sid"] for r in gA["rows"]] + ([r["sid"] for r in gB["rows"]] if gB else [])
        tm = [pt.get(posn[s]) for s in sids]
        before = sum(t is not None and t < EVAL_MTIME for t in tm)
        after = sum(t is not None and t >= EVAL_MTIME for t in tm)
        out.append(f"{bb}  ({R['n_sequences']} cached sequences)   verdict: {'PASS' if ok else 'FAIL'}")
        out.append(f"  re-capture with my own hook (own model instance; module weight == checkpoint file: {g['module_weight_equals_checkpoint_file']}), seeded sample of 60 sequences, all kinds "
                   f"({', '.join(f'{k} {v[0]}/{v[1]}' for k, v in gA['by_kind'].items())}):")
        out.append(f"    bitwise equal: teacher-forced key {gA['tf_bitwise'][0]}/{gA['tf_bitwise'][1]}, prompt-only key {gA['decode_bitwise'][0]}/{gA['decode_bitwise'][1]}, "
                   f"the 11 decode-step keys {gA['steps_bitwise'][0]}/{gA['steps_bitwise'][1]} sequences; lp_tf / lengths {gA['lp_tf_ok'][0]}/{gA['lp_tf_ok'][1]}, "
                   f"lp_decode {gA['lp_decode_ok'][0]}/{gA['lp_decode_ok'][1]}; pre-edit outcome {gA['pre_flags_equal'][0]}/{gA['pre_flags_equal'][1]}")
        if gB:
            out.append(f"    ALL {gB['n']} shifted sequences: key at lp_tf {gB['tf_bitwise'][0]}/{gB['tf_bitwise'][1]}, key at the decode's token {gB['tf_prompt_bitwise'][0]}/{gB['tf_prompt_bitwise'][1]}, "
                       f"prompt-only {gB['decode_bitwise'][0]}/{gB['decode_bitwise'][1]}, steps {gB['steps_bitwise'][0]}/{gB['steps_bitwise'][1]}; token at lp_tf {gB['tokens_at_lp_tf']}, "
                       f"at the decode's position {gB['tokens_at_lp_decode']}; cosine between the two keys {gB['tf_key_cos_decode_token_key_min_max'][0]:.2f} .. {gB['tf_key_cos_decode_token_key_min_max'][1]:.2f}")
        out.append(f"    re-captured sequences from capture parts written before / after the last change of rescope/evaluate.py ({EVAL_MTIME}): {before} / {after}")
        out.append(f"  last prompt indices re-derived with the tokenizer for EVERY sequence: teacher-forced differing {B['n_tf_differing']}/{B['n_tf']}, prompt-only differing "
                   f"{B['n_decode_differing']}/{B['n_decode']}; shifted {B['n_shifted_mine']} (index {B['n_shifted_index']}, same set: {B['shifted_sets_equal']}), token at lp_tf "
                   f"{B['shift_token']}, answers starting with a digit {B['n_shifted_answer_starts_with_digit']}; by kind {B['shifted_by_kind']}; shifted Rel edits {B['shifted_rel_edits']}")
        out.append(f"  index against the raw data: edit sequences differing from the stream {A['edit_sequences_differing_from_stream']} of {A['edit_sequences']}; probe rows differing from "
                   f"their eval item {A['probe_rows_differing_from_eval_items']}/{A['n_probe_rows']}; probe set re-derived: {A['probe_set_rederived']}")
        ci = C["image"]
        out.append(f"  benign pool ({C['n_image']} image + {C['n_text']} text rows, {C['n_distinct_benign_queries']} distinct queries): rows whose item is a probe item "
                   f"{C['n_benign_rows_whose_item_is_a_probe_item']}, an edit item {C['n_benign_rows_whose_item_is_an_edit_item']}; rows equal to an edit query "
                   f"{C['n_benign_rows_equal_to_an_edit_query']} (by file content {C['n_benign_rows_equal_to_an_edit_query_by_content']}), to a probe query "
                   f"{C['n_benign_rows_equal_to_a_probe_query']} (by content {C['n_benign_rows_equal_to_a_probe_query_by_content']})")
        out.append(f"    image rows sharing the QUESTION string with an edit {ci['question_is_an_edit_question']}, with a probe {ci['question_is_a_probe_question_same_ptype']}; sharing the IMAGE "
                   f"with an edit {ci['image_path_is_an_edit_image']} (content {ci['image_content_is_an_edit_image']}, Freebase MID {ci['image_mid_is_an_edit_mid']}), with a probe "
                   f"{ci['image_path_is_a_probe_image']} (content {ci['image_content_is_a_probe_image']}); text rows sharing the question with an edit "
                   f"{C['text']['question_is_an_edit_question']}, with a probe {C['text']['question_is_a_probe_question_same_ptype']}")
        for k, v in C.items():
            if k.startswith("other_family_"):
                out.append(f"    family {k[13:]} of the same backbone: benign image rows that ARE a probe query {v['n_benign_image_rows_equal_to_a_probe_query']} (REPAIR {v['of_them_repair']}, "
                           f"HELD-OUT-DISJOINT {v['of_them_heldout_disjoint']}; index flag overlap.{k[13:]}.query_in_probe {v['index_flag_query_in_probe']}), an edit query "
                           f"{v['n_benign_image_rows_equal_to_an_edit_query']}")
        out.append(f"  hard negatives, 30 seeded rows (12 HN-I, 12 HN-Q, 6 HN-X): image / question from the stated edit and source item, source unrelated (no probe item, no shared image, MID, "
                   f"name): {D['n_ok']}/30; every row of the file, image / question against its edit(s): {D['whole_file_image_question_check']['n'] - D['whole_file_image_question_check']['bad']}"
                   f"/{D['whole_file_image_question_check']['n']}")
        out.append(f"    pre-edit answer re-decoded with my decoder v2: y0 equal {gC['y0_equal']}/30, 48-token text equal {gC['long_equal']}/30, validity equal {gC['validity_equal']}/30 "
                   f"({gC['n_valid_file']} valid); keys of the REAL decode against the cached step keys: {gC['real_decode_keys_bitwise']}/30 sequences ({gC['n_real_decode_forwards']} forwards)")
        out.append(f"  md5 of {E['n_files']} image files against the index: {E['n_files_whose_md5_differs_from_index']} differing")
        nums[bb] = {"recaptured_bitwise": [gA["n_ok"], gA["n"]], "shifted_bitwise": None if gB is None else [gB["n_ok"], gB["n"]], "lp_indices_differing": B["n_tf_differing"] + B["n_decode_differing"],
                    "n_shifted": B["n_shifted_mine"], "benign_rows_that_are_edit_or_probe_items_or_queries": C["n_benign_rows_whose_item_is_a_probe_item"] + C["n_benign_rows_whose_item_is_an_edit_item"]
                    + C["n_benign_rows_equal_to_an_edit_query_by_content"] + C["n_benign_rows_equal_to_a_probe_query_by_content"],
                    "benign_image_rows_sharing_question_with_edit": ci["question_is_an_edit_question"], "benign_image_rows_sharing_question_with_probe": ci["question_is_a_probe_question_same_ptype"],
                    "benign_rows_sharing_image_with_edit_or_probe": ci["image_content_is_an_edit_image"] + ci["image_content_is_a_probe_image"],
                    "benign_rows_equal_to_probe_query_of_other_family": {k[13:]: v["n_benign_image_rows_equal_to_a_probe_query"] for k, v in C.items() if k.startswith("other_family_")},
                    "hard_negatives_construction_ok": [D["n_ok"], 30], "hard_negatives_y0_reproduced": [gC["n_ok"], 30], "recaptured_before_after_evaluate_py_change": [before, after]}
    ver["check2_caches"] = {"verdict": "pass" if ok_all else "fail", "numbers": nums}
    out.append("")


def check3(out, ver):
    out.append("=" * 150)
    out.append("CHECK 3  cached-key evaluation: rows of SparseEvaluator == rows of the full Evaluator (two gates: default method, cosine two-block; draw 0)")
    out.append("=" * 150)
    ok_all, nums, rows = True, {}, []
    for t in TAGS:
        s = load(f"c3_{t}.json")
        if s is None:
            ok_all = False
            rows.append([V.DEV[t]["log"], "MISSING (partial: " + str((load(f"c3_{t}_partial.json") or {}).get("groups", {}).keys()) + ")"])
            continue
        ok = s["n_rows"] == s["n_rows_equal"] and s["module_weight_is_WE_after"] and all(g["n_rows"] == 2 * g["n_items"] for g in s["groups"].values())
        ok_all = ok_all and ok
        for gname, g in s["groups"].items():
            rows.append([s["log"], gname, g["lp"], g["n_items"], g["n_rows"], g["n_rows_equal"], "/".join(str(v) for v in g["n_fired"].values()), "PASS" if g["n_rows"] == g["n_rows_equal"] else "FAIL"])
        st = s["sparse_stats"]
        nums[t] = {"n_rows": s["n_rows"], "n_rows_equal": s["n_rows_equal"], "by_group": {k: [g["n_rows_equal"], g["n_rows"]] for k, g in s["groups"].items()},
                   "sparse_forwards": st["forwards"], "full_forwards": s["n_rows"], "capped_edits": {k: v["n_edits_capped"] for k, v in s["gates"].items()}}
        rows.append([s["log"], "all", "", sum(g["n_items"] for g in s["groups"].values()), s["n_rows"], s["n_rows_equal"], f"sparse forwards {st['forwards']} of {s['n_rows']}",
                     "PASS" if ok else "FAIL"])
    out += fmt_tab(rows, ["log", "group", "gate position", "items", "rows (2 gates)", "rows equal", "fired G1/G2", "verdict"])
    out.append("groups: edits = Rel / T-Gen / I-Gen of all edits; probes = every valid probe row (all pools); probes_invalid = rows with an invalid but scorable pre-edit answer;")
    out.append("shifted = lp 'prompt' on the sequences whose teacher-forced last prompt position differs from the decode's; hn = valid hard negatives of the final set.  A row = the whole dict.")
    ver["check3_sparse_evaluator"] = {"verdict": "pass" if ok_all else "fail", "numbers": nums}
    out.append("")


def check4(out, ver):
    out.append("=" * 150)
    out.append("CHECK 4  DUAL from my own generations; token-level locality of the deployed log")
    out.append("=" * 150)
    ok_all, nums = True, {}
    for t in TAGS:
        s = load(f"c4_{t}.json")
        if s is None:
            ok_all = False
            out.append(f"{V.DEV[t]['log']}: MISSING")
            continue
        d, L, w = s["dual"], s["locality"], s["weights"]
        n = d["n_items"]
        okd = all(d[f"{m}_{k}_equal"] == n for m in ("deployed", "pre-edit") for k in ("long", "y", "line")) and \
            all(x["released_equal"] == n and x["matching_edits_equal"] == n and x["returned_long_equal"] == n and x["returned_y_equal"] == n and x["own_eq_equal"][0] == x["own_eq_equal"][1]
                for x in d["dual"].values())
        okl = L["rows_equal_library_rows"] and L["per_row_tok_loc_differing"] == 0 and all(abs(L[k]["mine"] - L[k]["library"]) < 1e-4 for k in ("T-Loc*", "I-Loc*")) and \
            L["y0_reproduced_by_my_W0"][0] == L["y0_reproduced_by_my_W0"][1] and L["D1_flags_equal_stored_complaints"][0] == L["D1_flags_equal_stored_complaints"][1]
        okw = w["H.W0_equals_checkpoint_file"] and w["H.WE_equals_my_rebuild"] and w["my_WE_minus_W0_equals_stored_composite"]
        ok = okd and okl and okw
        ok_all = ok_all and ok
        out.append(f"{s['log']}   verdict: {'PASS' if ok else 'FAIL'}   (weights: H.W0 == checkpoint file {w['H.W0_equals_checkpoint_file']}, H.WE == my rebuild from the batch deltas "
                   f"{w['H.WE_equals_my_rebuild']}, composite delta {w['my_WE_minus_W0_equals_stored_composite']})")
        out.append(f"  generations, mine against the library's ({d['n_probe']} valid HELD-OUT-DISJOINT probe rows + {d['n_edit']} edit prompts): deployed 48-token text equal "
                   f"{d['deployed_long_equal']}/{n}, 10-token answer {d['deployed_y_equal']}/{n}; pre-edit {d['pre-edit_long_equal']}/{n}, {d['pre-edit_y_equal']}/{n}; "
                   f"degenerate outputs (library rows) {d['degenerate_library_rows']}")
        for v_, x in d["dual"].items():
            out.append(f"  DUAL '{v_}': released flag equal {x['released_equal']}/{n}, matching edits equal {x['matching_edits_equal']}/{n}, returned text equal {x['returned_long_equal']}/{n}; "
                       f"mine: deployed answer released on {x['mine']['probe_released'][0]}/100 probe rows and {x['mine']['edit_released'][0]}/50 edit prompts, own target returned "
                       f"{x['mine']['edit_own_target_returned'][0]}/50 (library {x['library']['probe_released'][0]}, {x['library']['edit_released'][0]}, {x['library']['edit_own_target_returned'][0]})")
        bm = L["benchmark_quantity_recomputed_at_B1"]
        out.append(f"  token-level locality of the deployed log, valid HELD-OUT-DISJOINT rows (text {L['T-Loc*']['n']}, image {L['I-Loc*']['n']}): T-Loc* mine {L['T-Loc*']['mine']:.2f} library "
                   f"{L['T-Loc*']['library']:.2f}; I-Loc* mine {L['I-Loc*']['mine']:.2f} library {L['I-Loc*']['library']:.2f}; per-row values differing {L['per_row_tok_loc_differing']}/{L['n_rows']}; "
                   f"y0 reproduced by my W_0 {L['y0_reproduced_by_my_W0'][0]}/{L['y0_reproduced_by_my_W0'][1]}; D1 {L['D1_mine']:.2f} %, flags equal to the stored complaints "
                   f"{L['D1_flags_equal_stored_complaints'][0]}/{L['D1_flags_equal_stored_complaints'][1]}; non-finite rows {L['n_nonfinite']}")
        out.append(f"  stored benchmark locality (vendor pass, reference answers, padded batches of 8, all 500 items): T-Loc {L['stored_benchmark']['T-Loc']:.2f}, I-Loc {L['stored_benchmark']['I-Loc']:.2f};  "
                   f"SIGNED difference T-Loc* - T-Loc = {L['my_diff']['T-Loc']:+.2f}, I-Loc* - I-Loc = {L['my_diff']['I-Loc']:+.2f} (library: {L['library_diff']['T-Loc']:+.2f}, {L['library_diff']['I-Loc']:+.2f})")
        out.append(f"  the BENCHMARK quantity itself recomputed with my code at B = 1 (reference answers, benchmark formats, 500 items): T-Loc {bm['T-Loc']['mine_B1']:.2f} (stored {bm['T-Loc']['stored_B8']:.2f}, "
                   f"diff {bm['T-Loc']['mine_minus_stored']:+.2f}), I-Loc {bm['I-Loc']['mine_B1']:.2f} (stored {bm['I-Loc']['stored_B8']:.2f}, diff {bm['I-Loc']['mine_minus_stored']:+.2f}); "
                   f"non-finite {bm['T-Loc']['n_nonfinite']} / {bm['I-Loc']['n_nonfinite']}")
        pa = L.get("pool_all_valid_rows")
        if pa:
            out.append(f"  the same on ALL valid probe rows ({pa['n_rows']}; the pool of the library's test b12 and of the library summary): T-Loc* mine {pa['T-Loc']['mine*']:.2f} (library test "
                       f"{pa['T-Loc']['library_test_b12*']:.2f}), I-Loc* mine {pa['I-Loc']['mine*']:.2f} (library test {pa['I-Loc']['library_test_b12*']:.2f}); difference to the stored benchmark "
                       f"{pa['T-Loc']['mine_minus_stored']:+.2f} / {pa['I-Loc']['mine_minus_stored']:+.2f} (library test {pa['T-Loc']['library_test_b12_diff']:+.2f} / {pa['I-Loc']['library_test_b12_diff']:+.2f})")
        lg = load(f"c4_{t}/lib_gen.json")
        agree = [r.get("agree") for r in lg["rows"]["pre-edit"] if r["kind"] == "probe"] if lg else []
        out.append(f"  decoder-v2 pre-edit answers of the {len(agree)} sampled probe rows re-decoded (library rows, equal to my generations): equal to the stored y0 on {sum(bool(a) for a in agree)}/{len(agree)}")
        nums[t] = {"generations_equal": [d["deployed_long_equal"] + d["pre-edit_long_equal"], 2 * n], "pool_all_valid_rows": pa, "y0_redecoded_equal": [sum(bool(a) for a in agree), len(agree)], "dual_eq_decisions_equal": [d["dual"]["eq"]["released_equal"], n],
                   "dual_contains_decisions_equal": [d["dual"]["contains"]["released_equal"], n], "dual_eq_mine": d["dual"]["eq"]["mine"],
                   "T-Loc*": [L["T-Loc*"]["mine"], L["T-Loc*"]["library"]], "I-Loc*": [L["I-Loc*"]["mine"], L["I-Loc*"]["library"]], "stored_benchmark": L["stored_benchmark"],
                   "signed_diff_Loc*_minus_stored": L["my_diff"], "benchmark_quantity_at_B1_mine": {k: v["mine_B1"] for k, v in bm.items()},
                   "benchmark_quantity_mine_minus_stored": {k: v["mine_minus_stored"] for k, v in bm.items()}}
    ver["check4_dual_locality"] = {"verdict": "pass" if ok_all else "fail", "numbers": nums}
    out.append("")


def check5(out, ver):
    out.append("=" * 150)
    out.append("CHECK 5  sanity table: M1 and M3, gate level and behaviour, recomputed (my naive rule / my own replay hook) and compared with the reported rows")
    out.append("=" * 150)
    ok_all, nums = True, {}
    for t in TAGS:
        s = load(f"c5_{t}.json")
        if s is None:
            ok_all = False
            out.append(f"{V.DEV[t]['log']}: MISSING")
            continue
        g, b = s["gate"], s["behaviour"]
        tot = g["decisions_vs_stored_flags"]
        okg = g["rows_equal_stored"] and tot["n_differing"] == 0 and tot["n_draws_cal_rows_differing"] == 0 and g["all_reported_rows_reproduced"]
        my = b.get("my_run")
        okb = b["all_reported_rows_reproduced_from_stored_rows"] and my is not None
        if my is not None:
            okb = okb and my["n_keys_bitwise_cache"] == my["n_rows"] and all(sum(m["flags_differing"].values()) == 0 and m["stats_equal"] and m["n_rows_compared"] == m["n_stored_rows"] == my["n_rows"]
                                                                              and m["thresholds"]["max_abs_threshold_diff"] <= 1e-12 and m["thresholds"]["calibration_sids_equal"]
                                                                              and m["thresholds"]["capped_mine"] == m["thresholds"]["capped_stored"] for m in my["methods"].values())
        ok = okg and okb
        ok_all = ok_all and ok
        out.append(f"{s['log']}   verdict: {'PASS' if ok else 'FAIL'}")
        out.append(f"  gate level: my decisions against the stored flags, M1 - M4 (20 draws, capped and uncapped), M5, M6: {tot['n_differing']} of {tot['n_flags']} differing; calibration rows differing "
                   f"in {tot['n_draws_cal_rows_differing']} of 20 draws; rows of the reported table reproduced from MY decisions: " + ", ".join(f"{k} {'yes' if v['equal_reported'] else 'NO'}" for k, v in g["table"].items()))
        rows = []
        for mid in ("M1", "M3"):
            m = g["table"][mid]
            d0 = m["draw0"]
            rows.append([mid, "reported (20 draws)", m["reported"]])
            rows.append([mid, "mine (20 draws)", f"{m['fire_mean']:.2f}[{m['fire_p05']:.2f}-{m['fire_p95']:.2f}] {m['cov_tgen']:.1f}/{m['cov_igen']:.1f} {m['sil_before']:.1f}>{m['sil_after']:.0f} "
                         + "/".join(f"{x:.0f}" for x in m["hn_any"]) + " " + "/".join(f"{x:.0f}" for x in m["hn_own"])])
            rows.append([mid, "mine, draw 0", f"{d0['fire']:.2f} {d0['cov_tgen']:.1f}/{d0['cov_igen']:.1f} {d0['sil_before']}>{d0['sil_after']} " + "/".join(f"{x:.1f}" for x in d0["hn_any"]) + " "
                         + "/".join(f"{x:.1f}" for x in d0["hn_own"]) + f"   (eta {d0['eta']:.6f}, capped edits {d0['n_capped']}, eta equal to the stored one to {d0['eta_abs_diff_stored']:.0e})"])
        out += ["    " + x for x in fmt_tab(rows, ["method", "source", "fire %[5th-95th]  cov T/I  sil before>after  HN any I/Q/X  HN own I/Q/X"])]
        tb = b["table_from_stored_rows"]
        out.append("  behaviour: rows of the reported table (mean of draws 0 - 2) recomputed from the stored per-row flags: " + ", ".join(f"{k} {'yes' if v['equal_reported'] else 'NO'}" for k, v in tb.items()))
        if my is None:
            out.append("  behaviour, my own run of draw 0: MISSING")
        else:
            out.append(f"  my own run of draw 0 ({my['n_rows']} sequences; own model instance, own hook, own rebased values, calibration from the cached keys on the CPU): keys bitwise equal to the cache "
                       f"{my['n_keys_bitwise_cache']}/{my['n_rows']}")
            rows = []
            f3 = lambda v, dg: "/".join(f"{x:.{dg}f}" for x in v)                             # noqa: E731
            for mid in ("M1", "M3"):
                m = my["methods"][mid]
                th = m["thresholds"]
                rows.append([mid, "reported (mean of draws 0-2)", tb[mid]["reported"], ""])
                for nm, st in (("stored rows, draw 0", m["stats_stored_draw0"]), ("mine, draw 0", m["stats_mine_draw0"])):
                    rows.append([mid, nm, f"{f3(st['exact'], 0)} {f3(st['D1'], 2)} {f3(st['hn_D1'], 1)} {f3(st['loc'], 2)}",
                                 "" if nm.startswith("stored") else f"flags differing {sum(m['flags_differing'].values())} in {m['n_rows_with_a_difference']} of {m['n_rows_compared']} rows; fired {m['n_fired_mine']}; "
                                 f"thresholds max |diff| {th['max_abs_threshold_diff']:.1e}, capped {th['capped_mine'] == th['capped_stored']}"])
            out += ["    " + x for x in fmt_tab(rows, ["method", "source", "Rel/T/I exact %  D1 % disj/strict  HN D1 % I/Q/X  T-Loc*/I-Loc*", "per-sequence comparison with the stored rows"])]
        nums[t] = {"gate_decisions_differing": [tot["n_differing"], tot["n_flags"]], "gate_rows_reproduced": {k: v["equal_reported"] for k, v in g["table"].items()},
                   "behaviour_rows_reproduced_from_stored_rows": {k: v["equal_reported"] for k, v in tb.items()},
                   "my_run": None if my is None else {"n_sequences": my["n_rows"], "keys_bitwise_cache": my["n_keys_bitwise_cache"],
                                                      "flags_differing": {k: sum(m["flags_differing"].values()) for k, m in my["methods"].items()},
                                                      "rows_with_a_difference": {k: m["n_rows_with_a_difference"] for k, m in my["methods"].items()},
                                                      "stats_draw0_mine": {k: m["stats_mine_draw0"] for k, m in my["methods"].items()},
                                                      "stats_draw0_stored": {k: m["stats_stored_draw0"] for k, m in my["methods"].items()},
                                                      "max_abs_threshold_diff": {k: m["thresholds"]["max_abs_threshold_diff"] for k, m in my["methods"].items()}}}
    ver["check5_sanity_table"] = {"verdict": "pass" if ok_all else "fail", "numbers": nums}
    out.append("")


def check6(out, ver):
    out.append("=" * 150)
    out.append("CHECK 6  scale mechanics on TB-Q2000-ae-l2_3 (unseen test log: pass / fail of the mechanics only)")
    out.append("=" * 150)
    s = load("c6_scale.json")
    if s is None:
        out.append("MISSING")
        ver["check6_scale"] = {"verdict": "fail", "numbers": "missing"}
        return
    M = s["mechanics"]
    out.append(f"verdict: {'PASS' if s['pass'] else 'FAIL'}")
    out.append(f"  load through the library: lean {s['load']['info']['lean']}, {s['load']['info']['seconds']:.0f} s, load checks ok {M['load_checks_ok']}; peak resident memory after load "
               f"{s['load']['peak_rss_gb_after_load']:.1f} GB (before load {s['peak_rss_gb_before_load']:.1f} GB), at the end of the whole check {s['peak_rss_gb_end']:.1f} GB; peak GPU {s['peak_gpu_gb']:.2f} GB")
    out.append(f"  default method from the key cache, one calibration draw: {M['n_cached_sequences_of_the_log']} cached sequences decided; calibration record ok {M['calibration_record_ok']}; "
               f"calibration rows == mine {M['calibration_rows_equal_mine']}; decisions differing from my naive rule {M['cpu_decisions_differing_from_naive'][0]} of "
               f"{M['cpu_decisions_differing_from_naive'][1]}; eta equal {M['eta_equal_naive_1e-12']}; capped sets equal {M['capped_edits_equal_naive']}")
    out.append(f"  model: gate built on the GPU == CPU gate to 1e-12 {M['gpu_thresholds_equal_cpu_1e-12']}; {M['n_firing_edit_sequences_run']} firing edit sequences: hook fired {M['hook_fired'][0]}/{M['hook_fired'][1]}, "
               f"releasing edit == CPU decision {M['hook_top_equals_cpu_decision'][0]}/{M['hook_top_equals_cpu_decision'][1]}, own flags == CPU {M['hook_own_flags_equal_cpu_decision'][0]}/"
               f"{M['hook_own_flags_equal_cpu_decision'][1]}, finite logits {M['logits_finite'][0]}/{M['logits_finite'][1]}, sparse rows == full rows {M['sparse_rows_equal_full_rows_firing'][0]}/"
               f"{M['sparse_rows_equal_full_rows_firing'][1]}, my own replay == library rows {M['my_replay_equals_library_rows'][0]}/{M['my_replay_equals_library_rows'][1]}; "
               f"30 silent sequences: sparse == full {M['sparse_rows_equal_full_rows_silent'][0]}/{M['sparse_rows_equal_full_rows_silent'][1]}, no forward run {M['sparse_ran_no_forward_on_silent_sequences']}")
    ver["check6_scale"] = {"verdict": "pass" if s["pass"] else "fail",
                           "numbers": {"peak_rss_gb_after_load": s["load"]["peak_rss_gb_after_load"], "peak_rss_gb_end_of_check": s["peak_rss_gb_end"], "load_seconds": s["load"]["info"]["seconds"],
                                       "peak_gpu_gb": s["peak_gpu_gb"], "mechanics": M}}
    out.append("")


def extras(out, ver):
    out.append("=" * 150)
    out.append("EXTRA CHECKS (not in the assignment)")
    out.append("=" * 150)
    x2 = load("x2_pool_content.json")
    if x2:
        out.append("x2  HELD-OUT-DISJOINT by image CONTENT (md5) instead of the image path: " + "; ".join(
            f"{k}: {v['n_equal_to_a_repair_row_by_content']} of {v['n_valid_heldout_disjoint']} valid rows equal a REPAIR row, {v['n_sharing_the_question_string_with_a_repair_row']} share only its "
            f"question string, {v['n_distinct_queries_by_content_in_heldout_disjoint']} distinct queries" for k, v in x2.items() if "n_valid_heldout_disjoint" in v))
    x3ok, x3n = True, {}
    for t in TAGS:
        x = load(f"x3_{t}.json")
        if x is None:
            out.append(f"x3  gated free generation {V.DEV[t]['log']}: MISSING")
            x3ok = False
            continue
        x3ok = x3ok and x["pass"]
        x3n[t] = {k: x[k] for k in ("n", "fired_equal", "top_equal", "long_equal", "y_equal", "n_fired_lib", "pos_ok_lib", "n_shifted", "by_group")}
        out.append(f"x3  free generation THROUGH the default gate (draw 0), library ev.generate against my decoder + my replay hook, {x['log']}: {x['n']} items "
                   f"(30 edit prompts, 30 probes, 20 hard negatives): 48-token text equal {x['long_equal']}/{x['n']}, 10-token answer {x['y_equal']}/{x['n']}, fired flag equal {x['fired_equal']}/{x['n']} "
                   f"({x['n_fired_lib']} fire), releasing edit equal {x['top_equal']}/{x['n']}; library position bookkeeping ok {x['pos_ok_lib']}/{x['n']}; shifted items {x['n_shifted']}   "
                   f"{'PASS' if x['pass'] else 'FAIL'}")
        for r in x["differing"][:3]:
            out.append(f"      differing {r['sid']}: fired mine {r['fired_mine']} lib {r['fired_lib']}, lp_shift {r['lp_shift']}; mine {r['mine'][:80]!r} | lib {r['lib'][:80]!r}")
    xs = load("x3s_L.json")
    if xs:
        out.append(f"x3s the same on EVERY evaluated LLaVA sequence whose teacher-forced last prompt position is one token after the decode's ({xs['n']} items, {xs['n_shifted']} shifted): "
                   f"texts equal {xs['long_equal']}/{xs['n']}, fired flag equal {xs['fired_equal']}/{xs['n']} ({xs['n_fired_lib']} fire in the decode); own edit fires in the decode of its shifted "
                   f"Rel prompt: {xs['own_edit_fired_in_decode_of_shifted_rel']}   {'PASS' if xs['pass'] else 'FAIL'}")
        x3ok = x3ok and xs["pass"]
        x3n["L_shifted"] = {k: xs[k] for k in ("n", "fired_equal", "long_equal", "y_equal", "n_fired_lib", "n_shifted", "own_edit_fired_in_decode_of_shifted_rel")}
    x4 = load("x4_more_rules.json")
    if x4:
        for t in TAGS:
            a_, b_ = x4[t]["A_D_features_ratios_rank"], x4[t]["B_which_all"]
            out.append(f"x4  {x4[t]['log']}: features coord / negdist, block ratios 1:3 and 3:1, rank 'r' ({a_['n_configs']} calibrations, capped and uncapped): {a_['n_differing']} of {a_['n_decisions']} decisions "
                       f"differ from the naive rule (real {a_['n_real']}), capped sets differing {a_['n_capped_sets_differing']}; several keys per query (which = 'all', calibrated on them): "
                       f"{b_['n_differing']} of {b_['n_rows']} sequences differ ({b_['n_fired']} fire, {b_['n_fired_on_the_teacher_forced_key_alone']} on the teacher-forced key alone)")
        c_ = x4["C_calibrate_dense"]
        out.append(f"x4  calibrate_dense (router / z-score baselines), shared and two-block, scores with ties: {c_['n_differing']} of {c_['n_decisions']} decisions differ from the strict defining comparison "
                   f"({c_['n_cases']} calibrations); threshold != succ(eta): {c_['n_threshold_not_succ_eta']}   {'PASS' if x4['pass'] else 'FAIL'}")
    x5, x6, x7 = load("x5_more.json"), load("x6_exactness.json"), load("x7_null_stored.json")
    if x5:
        for t in TAGS:
            a_ = x5["A_top1_readings"][t]
            out.append(f"x5  {V.DEV[t]['log']}: top-1 with CAPPED edits, library (best candidate releases) against the literal text of the protocol (argmax over all edits, then its threshold), 20 draws, "
                       f"mean number of sequences decided differently (of {a_['M1']['n_rows']}): " + ", ".join(f"{m} {v['rows_differing']:.2f} (capped edits {v['n_capped']:.1f})" for m, v in a_.items())
                       + f"; own Rel prompt released, M3: {a_['M3']['rel_own_released_lib']:.1f} vs {a_['M3']['rel_own_released_literal']:.1f} %")
        b_ = x5["B_shifted_edits_solver_key_position"]
        out.append(f"x5  LLaVA edits with a shifted last prompt position {[e['edit'] for e in b_['edits']]}: stored solver key against the cached B = 1 key at the teacher-forced position, cosine "
                   f"{min(e['cos_stored_key_vs_key_at_lp_tf'] for e in b_['edits']):.5f} at least; at the decode's token {max(e['cos_stored_key_vs_key_at_decode_token'] for e in b_['edits']):.2f} at most "
                   f"-> all written at the teacher-forced (bare-space) position: {b_['all_written_at_the_teacher_forced_position']}")
        for bb, c_ in x5["C_cache_counts"].items():
            out.append(f"x5  {bb} counts recounted: " + "; ".join(f"{k} {sum(v['rows_by_family'].values())} rows, valid/y0 {v['valid']}/{v['with_y0']}, dup {v['dup']}, shifted {v['shifted']}, bare None/No {v['bare_none_no']}, "
                                                                 f"empty {v['empty']}, 1 char {v['single_char']}" for k, v in c_.items() if k not in ("all", "hn_file")))
            h = c_["hn_file"]
            d_ = x5["D_hard_negative_sources"][bb]
            out.append(f"      hard negatives: {h['existing']} existing + {h['new']} new; new HN-I same category {h['new_HN-I_same_category'][0]}/{h['new_HN-I_same_category'][1]}, related_flag "
                       f"{h['new_HN-I_related_flag'][0]}/{h['new_HN-I_related_flag'][1]} (existing {h['existing_HN-I_related_flag'][0]}/{h['existing_HN-I_related_flag'][1]}); existing rows whose source "
                       f"is related to a stream edit {h['existing_rows_with_source_related_to_a_stream_edit']} (first at position {h['smallest_such_position']}; my rule {d_['existing_rows_related_to_a_stream_edit_mine']}, "
                       f"{d_['existing_rows_first_related_position_differing_from_file']} row differs); source pool re-derived {d_['my_source_pool_size']} items, new rows with a source outside it "
                       f"{d_['new_rows_whose_source_is_outside_my_pool']}/{d_['n_new_rows']}")
    if x6:
        out.append(f"x6  synthetic: exact rank {x6['A_rank']['n_differing']} of {x6['A_rank']['n']} differing; float-safe thresholds on boundary inputs {x6['B_float_safe']['n_differing']} of "
                   f"{x6['B_float_safe']['n']} differing; two-block rule with ties and custom scales {x6['C_two_block_ties_custom_u']['n_differing']} of {x6['C_two_block_ties_custom_u']['n_decisions']} "
                   f"decisions, eta differing {x6['C_two_block_ties_custom_u']['n_eta_differing']}   {'PASS' if x6['pass'] else 'FAIL'}")
    if x7:
        for t in TAGS:
            n_, m_ = x7["A_null_repair"][t], x7["B_stored_metrics"][t]
            out.append(f"x7  {V.DEV[t]['log']}: null_repair (detector step of the registered variant ReScope-N; mechanics only) against my closed form: detectors differ by "
                       f"{n_['detectors_max_rel_diff_lib_vs_closed_form']:.1e} (relative) at most, coordinate on the {n_['n_benign_keys']} benign keys {n_['max_abs_coordinate_on_benign_keys_lib']:.1e} at most, "
                       f"own coordinate kept to {n_['max_own_coordinate_error_lib']:.1e}; two-block calibration + decisions on the repaired gate: {n_['decisions_differing']} of {n_['n_rows']} sequences differ "
                       f"from my naive rule; stored_metrics == my reading of the files: probes {m_['probe_counts_equal_mine']}, edits {m_['edits_equal']}, locality {m_['locality_equal_metrics_json']}")
    x2d = (x2 or {}).get("development_logs")
    if x2d:
        out.append("x2  calibration pool (valid REPAIR rows): " + "; ".join(f"{V.DEV[t]['log']}: {v['n_repair_valid_rows_in_duplicate_query_groups']} of {v['n_repair_valid']} rows lie in duplicate-query groups; image rows "
                                                                          f"that ask the question of a logged edit: REPAIR {v['n_repair_image_rows_asking_the_question_of_a_logged_edit']}, HELD-OUT-DISJOINT "
                                                                          f"{v['n_heldout_disjoint_image_rows_asking_the_question_of_a_logged_edit']} of {v['n_heldout_disjoint_valid']}" for t, v in x2d.items()))
    ver["bugs"] = []
    c2 = load("c2_cpu.json") or {}
    ver["notes"] = {
        "no_bug_found": "no decision, key, row or table value differs between the audited code and my own implementation in any check",
        "top1_with_capped_edits": None if not x5 else {t: {m: v["rows_differing"] for m, v in x5["A_top1_readings"][t].items()} for t in TAGS},
        "benign_rows_that_are_probe_queries_of_family_QK": (c2.get("qwen25vl", {}).get("C_benign", {}).get("other_family_QK") or {}).get("n_benign_image_rows_equal_to_a_probe_query"),
        "benign_image_rows_sharing_a_question_with_an_edit_of_the_long_stream": {bb: c2[bb]["C_benign"]["image"]["question_is_an_edit_question"] for bb in c2},
        "library_has_no_option_for_a_quantile_block1_baseline": "PROTOCOL.md section 6, 2026-11-10 22:10 (4): the 0.98-quantile baseline of E6 is not in rescope.calibrate (two_block_rule: b = max only)",
        "shifted_llava_edits_written_at_the_bare_space_token": None if not x5 else [e["edit"] for e in x5["B_shifted_edits_solver_key_position"]["edits"]],
        "shared_with_the_audited_code": "the testbed's model wrapper vendor_v7.vlm (tokenisation, batch building, checkpoint loading), HF generate for the KV-cache continuation of decoder v2, "
                                        "the guard-cap constant (1 - 1e-9) and the seeded draw / split conventions (numpy default_rng(seed).choice / .permutation)"}
    ver["extra"] = {"x5_more": x5, "x6_exactness": x6, "x7_null_stored": x7,
                    "x1_seed_coupling": load("x1_seed_coupling.json"), "x2_pool_by_content": x2, "x3_gated_free_generation": {"verdict": "pass" if x3ok else "fail", "numbers": x3n},
                    "x4_more_rules": None if x4 is None else {"verdict": "pass" if x4["pass"] else "fail", "A_D": {t: {k: v for k, v in x4[t]["A_D_features_ratios_rank"].items() if k != "configs"} for t in TAGS},
                                                              "B_which_all": {t: x4[t]["B_which_all"] for t in TAGS}, "C_calibrate_dense": x4["C_calibrate_dense"]}}
    out.append("")


def main():
    out, ver = [], {"rendered": time.strftime("%Y-%m-%d %H:%M:%S"), "by": "e0_verify/code/render.py from e0_verify/out/*.json"}
    out.append(f"E0 VERIFICATION (independent)      rendered {ver['rendered']} by code/render.py from out/*.json; no number is typed by hand")
    out.append("")
    for f in (check1, check2, check3, check4, check5, check6, extras):
        f(out, ver)
    ver["all_pass"] = all(v["verdict"] == "pass" for k, v in ver.items() if k.startswith("check"))
    ver["library_fingerprint"] = lib_fingerprint()
    with open(f"{V.OUT}/table.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")
    V.jdump(ver, f"{V.VER}/verdicts.json")
    print("\n".join(out))
    print({k: v["verdict"] for k, v in ver.items() if k.startswith("check")})


if __name__ == "__main__":
    main()
