"""Check 2, CPU part: structure of the key caches, the benign pool and the hard negatives, from the raw data.

  python c2_cpu.py        -> out/c2_cpu.json, out/c2_hn_sample_<backbone>.json (the 30 sampled hard negatives, for the GPU part)

  A  index vs raw data        edit sequences against the edit stream + VLKEB train items; probe rows against VLKEB eval items;
                              the probe set re-derived (eval items with m_loc and loc, seed 42, 500; LLaVA: key exclusion)
  B  last prompt indices      lp_tf / lp_decode / lp_shift of EVERY cached sequence re-derived with the tokenizer; for the
                              shifted sequences the token at lp_tf (must be the bare space that the joint tokenisation leaves)
  C  benign pool              no benign row is an edit item, a probe item, an edit query or a probe query (of its own VLKEB
                              family); rows that share a question string or an image (path and file content) with one: counted
  D  hard negatives           30 seeded rows per backbone: image and question come from the stated edit and from the stated
                              source item; the source item is unrelated (no probe item, no shared image / MID / name)
  E  image md5                md5 of the image files of the benign and probe rows against the index
"""
from __future__ import annotations

import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import vcommon as V

sys.path.insert(0, V.LIB)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from vendor_v7.vlm import VLM  # noqa: E402  (tokenizer only)

FAMILIES = {"qwen25vl": {"Q": "TB-Q2000-ae-l2_3", "QK": "TB-Q500-mmke-l2_3"}, "llava": {"L": "TB-L1000-ae-l2_3"}}
VLKEB_FAM = {"qwen25vl": "Q", "llava": "L"}
SEED = 20261111


def mid(path):
    return path.split("/")[0] if path else None


def low(s):
    return " ".join((s or "").lower().split())


def md5_file(path):
    h = hashlib.md5()
    with open(path if os.path.isabs(path) else os.path.join(V.VLKEB_IMG, path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    res = {}
    train = {d["uid"]: d for d in V.vlkeb("train")}
    evl = {d["uid"]: d for d in V.vlkeb("eval")}
    for bb in ("qwen25vl", "llava"):
        idx = V.read_index(bb)
        fam = VLKEB_FAM[bb]
        R = res[bb] = {"n_sequences": len(idx)}
        # ------------------------------------------------------------------ A: index vs raw data
        stream = V.stream_items(FAMILIES[bb][fam])
        items = [train[r["uid"]] for r in stream]
        A = {"n_edits_stream": len(items), "stream_fields_equal_train_items": all(d["src"] == r["subject"] and d["alt"] == r["alt"] and d["image"] == r["image"]
                                                                                 for d, r in zip(items, stream))}
        bad = {"rel": 0, "tgen": 0, "igen": 0}
        n = {"rel": 0, "tgen": 0, "igen": 0}
        for e in idx:
            if e["family"] == fam and e["kind"] in bad:
                d = items[e["edit"]]
                want = {"rel": (d["src"], d["image"]), "tgen": (d["rephrase"], d["image"]), "igen": (d["src"], d["image_rephrase"])}[e["kind"]]
                n[e["kind"]] += 1
                bad[e["kind"]] += int(not ((e["prompt"], e["image"]) == want and e["answer"] == d["alt"] and e["edit_uid"] == d["uid"] and e["fmt"] == "vqa"))
        A["edit_sequences"] = n
        A["edit_sequences_differing_from_stream"] = bad
        pr = [e for e in idx if e["family"] == fam and e["kind"] == "probe"]
        pbad = 0
        for e in pr:
            d = evl[e["uid"]]
            want = (d["m_loc_q"], d["m_loc"]) if e["ptype"] == "image" else (d["loc"], None)
            pbad += int((e["prompt"], e["image"]) != want)
        A["n_probe_rows"], A["probe_rows_differing_from_eval_items"] = len(pr), pbad
        probe_uids = []
        for e in pr:
            if e["uid"] not in probe_uids:
                probe_uids.append(e["uid"])
        derived = {}
        for tag, its in (("long_stream", items), ("first_100", items[:100])):          # the probe set re-derived (tb_common.select_probes)
            keys = {(d["image"], d["src"], d["alt"]) for d in its}
            elig = [d for d in V.vlkeb("eval") if d.get("m_loc") and d.get("loc") and (bb != "llava" or (d["image"], d["src"], d["alt"]) not in keys)]
            pick = sorted(np.random.default_rng(42).choice(len(elig), 500, replace=False))
            derived[tag] = {"n_eligible": len(elig), "equal_cache_probe_uids": [elig[i]["uid"] for i in pick] == probe_uids}
        A["probe_set_rederived"] = derived
        A["probe_idx_order_ok"] = [e["probe_idx"] for e in pr if e["ptype"] == "image"] == list(range(500))
        R["A_index_vs_raw"] = A
        # ------------------------------------------------------------------ B: last prompt indices from the tokenizer
        v = VLM(bb, device="cpu", load_model=False)
        B = {"n_tf": 0, "n_tf_differing": 0, "n_decode": 0, "n_decode_differing": 0, "n_shifted_index": 0, "n_shifted_mine": 0, "shifted_sets_equal": None,
             "n_prefix_violations": 0, "n_tf_errors": 0, "shift_token": {}, "n_shifted_answer_starts_with_digit": 0, "differing": []}
        sh_idx, sh_mine = set(), set()
        for e in idx:
            nimg = e.get("n_image_tokens") if e.get("image") else None
            p = v.encode_row(e["prompt"], None, e["fmt"], nimg)
            B["n_decode"] += 1
            okd = (len(p["ids"]) - 1 == e["lp_decode"]) and (len(p["ids"]) == e["len_prompt"])
            B["n_decode_differing"] += int(not okd)
            if e.get("lp_shift"):
                sh_idx.add(e["sid"])
            if e.get("row_tf") is None:
                continue
            try:
                t = v.encode_row(e["prompt"], e["answer"], e["fmt"], nimg)
            except (RuntimeError, ValueError):
                B["n_tf_errors"] += 1
                continue
            lp_tf = t["answer_start"] - 1
            shift = lp_tf - (len(p["ids"]) - 1)
            B["n_tf"] += 1
            ok = lp_tf == e["lp_tf"] and len(t["ids"]) == e["len_tf"] and shift == e["lp_shift"]
            B["n_tf_differing"] += int(not ok)
            if not (ok and okd) and len(B["differing"]) < 5:
                B["differing"].append({"sid": e["sid"], "mine": [lp_tf, len(p["ids"]) - 1, shift], "index": [e["lp_tf"], e["lp_decode"], e["lp_shift"]]})
            B["n_prefix_violations"] += int(t["ids"][:len(p["ids"])] != p["ids"])
            if shift:
                sh_mine.add(e["sid"])
                tk = v.tok.convert_ids_to_tokens([t["ids"][lp_tf]])[0]
                B["shift_token"][tk] = B["shift_token"].get(tk, 0) + 1
                B["n_shifted_answer_starts_with_digit"] += int(e["answer"].lstrip()[:1].isdigit())
        B.update(n_shifted_index=len(sh_idx), n_shifted_mine=len(sh_mine), shifted_sets_equal=sh_idx == sh_mine)
        B["shifted_by_kind"] = {}
        for e in idx:
            if e["sid"] in sh_mine:
                B["shifted_by_kind"][e["kind"]] = B["shifted_by_kind"].get(e["kind"], 0) + 1
        B["shifted_rel_edits"] = sorted(e["edit"] for e in idx if e["sid"] in sh_mine and e["kind"] == "rel")
        R["B_last_prompt_indices"] = B
        # ------------------------------------------------------------------ C: benign pool
        ben = [e for e in idx if e["kind"] == "benign"]
        brows = V.jsonl(f"{V.CACHE}/benign/{bb}/rows.jsonl")
        C = {"n_rows": len(ben), "n_image": sum(e["ptype"] == "image" for e in ben), "n_text": sum(e["ptype"] == "text" for e in ben),
             "rows_file_equal_index": [(r["sid"], r["prompt"], r["image"], r["uid"]) for r in brows] == [(e["sid"], e["prompt"], e["image"], e["uid"]) for e in ben]}
        cbad = 0
        for e in ben:
            d = evl[e["uid"]]
            want = (d["m_loc_q"], d["m_loc"]) if e["ptype"] == "image" else (d["loc"], None)
            cbad += int((e["prompt"], e["image"]) != want)
        C["rows_differing_from_their_eval_item"] = cbad
        edit_uids = {d["uid"] for d in items}
        edit_keys = {(d["image"], d["src"], d["alt"]) for d in items}
        edit_q = {(d["src"], d["image"]) for d in items} | {(d["rephrase"], d["image"]) for d in items} | {(d["src"], d["image_rephrase"]) for d in items}
        edit_quest = {low(d["src"]) for d in items} | {low(d["rephrase"]) for d in items}
        edit_img = {d["image"] for d in items} | {d["image_rephrase"] for d in items}
        probe_q = {(e["ptype"], e["prompt"], e["image"]) for e in pr}
        probe_quest = {(e["ptype"], low(e["prompt"])) for e in pr}
        probe_img = {e["image"] for e in pr if e["image"]}
        probe_mloc_item = {evl[u]["image"] for u in probe_uids}
        C["n_benign_rows_whose_item_is_a_probe_item"] = sum(e["uid"] in set(probe_uids) for e in ben)
        C["n_benign_rows_whose_item_is_an_edit_item"] = sum(e["uid"] in edit_uids or (evl[e["uid"]]["image"], evl[e["uid"]]["src"], evl[e["uid"]]["alt"]) in edit_keys for e in ben)
        C["n_benign_rows_equal_to_an_edit_query"] = sum(e["ptype"] == "image" and (e["prompt"], e["image"]) in edit_q for e in ben)
        C["n_benign_rows_equal_to_a_probe_query"] = sum((e["ptype"], e["prompt"], e["image"]) in probe_q for e in ben)
        C["n_distinct_benign_queries"] = len({(e["ptype"], e["prompt"], e["image"]) for e in ben})
        for pt in ("image", "text"):
            sub = [e for e in ben if e["ptype"] == pt]
            C[pt] = {"n": len(sub),
                     "question_is_an_edit_question": sum(low(e["prompt"]) in edit_quest for e in sub),
                     "question_is_a_probe_question_same_ptype": sum((pt, low(e["prompt"])) in probe_quest for e in sub),
                     "image_path_is_an_edit_image": sum(bool(e["image"]) and e["image"] in edit_img for e in sub),
                     "image_path_is_a_probe_image": sum(bool(e["image"]) and e["image"] in probe_img for e in sub),
                     "image_mid_is_an_edit_mid": sum(bool(e["image"]) and mid(e["image"]) in {mid(x) for x in edit_img} for e in sub),
                     "index_flags": {k: sum(bool(e.get(k)) for e in sub) for k in ("q_in_edit", "q_in_probe", "img_in_edit", "img_in_probe", "query_in_probe")}}
        # ------------------------------------------------------------------ E: md5 of the image files (benign + probe rows of every family)
        paths = sorted({e["image"] for e in idx if e["kind"] in ("benign", "probe") and e["image"]} | {e["image"] for e in idx if e["kind"] in ("rel", "igen") and e["image"]})
        with ThreadPoolExecutor(8) as ex:
            md5 = dict(zip(paths, ex.map(md5_file, paths)))
        want = {}
        for e in idx:
            if e["image"] in md5:
                want.setdefault(e["image"], set()).add(e["image_md5"])
        R["E_image_md5"] = {"n_files": len(paths), "n_files_whose_md5_differs_from_index": sum(want[p] != {md5[p]} for p in paths)}
        edit_md5 = {md5[x] for x in edit_img}
        probe_md5 = {md5[x] for x in probe_img}
        bi = [e for e in ben if e["ptype"] == "image"]
        C["image"]["image_content_is_an_edit_image"] = sum(md5[e["image"]] in edit_md5 for e in bi)
        C["image"]["image_content_is_a_probe_image"] = sum(md5[e["image"]] in probe_md5 for e in bi)
        C["n_benign_rows_equal_to_an_edit_query_by_content"] = sum((e["prompt"], md5[e["image"]]) in {(q, md5[i]) for q, i in edit_q} for e in bi)
        C["n_benign_rows_equal_to_a_probe_query_by_content"] = sum((e["prompt"], md5[e["image"]]) in {(x["prompt"], md5[x["image"]]) for x in pr if x["image"]} for e in bi)
        for ofam in FAMILIES[bb]:                                                        # other families of the backbone (MMKE-Bench for Qwen)
            if ofam == fam:
                continue
            opr = [e for e in idx if e["family"] == ofam and e["kind"] == "probe"]
            oq = {(e["prompt"], md5[e["image"]]): e for e in opr if e["image"]}
            ot = {e["prompt"] for e in opr if not e["image"]}
            hit = [oq[(e["prompt"], md5[e["image"]])] for e in bi if (e["prompt"], md5[e["image"]]) in oq]
            oed = {(e["prompt"], md5[e["image"]]) for e in idx if e["family"] == ofam and e["kind"] in ("rel", "tgen", "igen")}
            C[f"other_family_{ofam}"] = {"n_benign_image_rows_equal_to_a_probe_query": len(hit), "of_them_repair": sum(x["pool"] == "repair" for x in hit),
                                         "of_them_heldout_disjoint": sum(bool(x["disjoint"]) for x in hit),
                                         "n_benign_text_rows_equal_to_a_probe_query": sum(e["prompt"] in ot for e in ben if e["ptype"] == "text"),
                                         "n_benign_image_rows_equal_to_an_edit_query": sum((e["prompt"], md5[e["image"]]) in oed for e in bi),
                                         "index_flag_query_in_probe": sum(bool((e.get("overlap") or {}).get(ofam, {}).get("query_in_probe")) for e in ben)}
        R["C_benign"] = C
        # ------------------------------------------------------------------ D: hard negatives, 30 seeded rows
        hn = V.jsonl(f"{V.CACHE}/hard_negatives/{bb}/rows.jsonl")
        rng = np.random.default_rng([SEED, 0 if bb == "qwen25vl" else 1])
        pick = []
        for t, k in (("HN-I", 12), ("HN-Q", 12), ("HN-X", 6)):
            sub = [i for i, r in enumerate(hn) if r["hn_type"] == t]
            pick += [sub[int(i)] for i in sorted(rng.choice(len(sub), k, replace=False))]
        all_mids = {}
        all_names = {}
        for pos, d in enumerate(items):
            for m in {mid(d["image"]), mid(d["image_rephrase"])}:
                all_mids.setdefault(m, pos)
            for nm in {low(d["alt"]), low(d["pred"].replace("_", " "))}:
                all_names.setdefault(nm, pos)
        D = {"n_rows_file": len(hn), "n_sampled": len(pick), "rows": [], "n_ok": 0}
        sample = []
        hn_index = {e["sid"]: e for e in idx if e["kind"].startswith("hn")}
        for i in pick:
            r = hn[i]
            a = items[r["edit_pos"]]
            chk = {"hn_id": r["hn_id"], "type": r["hn_type"], "new": bool(r.get("new")), "edit_pos": r["edit_pos"],
                   "edit_uid_is_stream_uid": a["uid"] == r["edit_uid"], "edit_fields_equal_item": (r["edit_src"], r["edit_alt"], r["edit_rephrase"]) == (a["src"], a["alt"], a["rephrase"])}
            src = None
            if r["hn_type"] == "HN-I":
                src = evl[r["src_uid"]]
                chk["image_is_edit_image"] = r["image"] == a["image"]
                chk["question_is_source_m_loc_q"] = r["prompt"] == src["m_loc_q"] and r["src_field"] == "m_loc_q"
                chk["question_differs_from_edit_questions"] = low(r["prompt"]) not in (low(a["src"]), low(a["rephrase"]))
            elif r["hn_type"] == "HN-Q":
                src = evl[r["src_uid"]]
                chk["question_is_edit_question"] = r["prompt"] == a["src"]
                chk["image_is_source_image"] = r["image"] == src["image"] and r["src_field"] == "image"
                chk["image_differs_from_edit_images"] = r["image"] not in (a["image"], a["image_rephrase"]) and mid(r["image"]) not in (mid(a["image"]), mid(a["image_rephrase"]))
            else:
                b = items[r["edit_b_pos"]]
                chk["image_is_edit_a_image"] = r["image"] == a["image"]
                chk["question_is_edit_b_question"] = r["prompt"] == b["src"] and b["uid"] == r["edit_b_uid"] and r["edit_b_pos"] != r["edit_pos"]
                chk["question_differs_from_edit_a_questions"] = low(r["prompt"]) not in (low(a["src"]), low(a["rephrase"]))
                chk["edits_share_no_mid_or_name"] = not ({mid(a["image"]), mid(a["image_rephrase"])} & {mid(b["image"]), mid(b["image_rephrase"])}) and \
                    not ({low(a["alt"]), low(a["pred"].replace("_", " "))} & {low(b["alt"]), low(b["pred"].replace("_", " "))})
            if src is not None:
                smids = {mid(src["image"]), mid(src["image_rephrase"])}
                snames = {low(src["alt"]), low(src["pred"].replace("_", " "))}
                chk["source_is_no_probe_item"] = src["uid"] not in set(probe_uids)
                chk["source_shares_no_mid_or_name_with_the_edit"] = not (smids & {mid(a["image"]), mid(a["image_rephrase"])}) and \
                    not (snames & {low(a["alt"]), low(a["pred"].replace("_", " "))})
                chk["source_item_key_is_no_edit_key"] = (src["image"], src["src"], src["alt"]) not in edit_keys
                first = [all_mids[m] for m in smids if m in all_mids] + [all_names[x] for x in snames if x in all_names]
                chk["info_first_stream_edit_related_to_source"] = min(first) if first else None
                chk["info_file_src_related_edit_first_pos"] = r.get("src_related_edit_first_pos")
            e = hn_index.get(r["hn_id"])
            chk["index_row_equal"] = e is not None and (e["prompt"], e["image"], e["answer"], e["y0_valid"], e["edit"]) == (r["prompt"], r["image"], r["y0_text"] or None, r["y0_valid"], r["edit_pos"])
            chk["ok"] = all(v for k, v in chk.items() if isinstance(v, bool) and not k.startswith("info") and k != "new")
            D["n_ok"] += int(chk["ok"])
            D["rows"].append(chk)
            sample.append({"hn_id": r["hn_id"], "hn_type": r["hn_type"], "fmt": r["fmt"], "prompt": r["prompt"], "image": r["image"], "y0_text": r["y0_text"],
                           "y0_valid": r["y0_valid"], "long_text": r.get("long_text"), "edit_pos": r["edit_pos"]})
        # the whole file: every row's image / question against its edit(s)
        whole = {"n": len(hn), "bad": 0}
        for r in hn:
            a = items[r["edit_pos"]]
            if r["hn_type"] == "HN-I":
                ok = r["image"] == a["image"] and r["prompt"] == evl[r["src_uid"]]["m_loc_q"]
            elif r["hn_type"] == "HN-Q":
                ok = r["prompt"] == a["src"] and r["image"] == evl[r["src_uid"]]["image"]
            else:
                ok = r["image"] == a["image"] and r["prompt"] == items[r["edit_b_pos"]]["src"]
            whole["bad"] += int(not (ok and a["uid"] == r["edit_uid"]))
        D["whole_file_image_question_check"] = whole
        R["D_hard_negatives"] = D
        V.jdump(sample, f"{V.OUT}/c2_hn_sample_{bb}.json")
        V.say(f"{bb}: A {A}")
        V.say(f"{bb}: B { {k: v for k, v in B.items() if k != 'differing'} }")
        V.say(f"{bb}: C {C}")
        V.say(f"{bb}: D n_ok {D['n_ok']}/{len(pick)}, whole {whole}; E {R['E_image_md5']}")
    V.jdump(res, f"{V.OUT}/c2_cpu.json")


if __name__ == "__main__":
    main()
