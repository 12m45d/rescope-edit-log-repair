"""Assemble the recombination keys of one backbone from the capture parts.  CPU.

    python rc_assemble.py <qwen25vl | llava> [--tag fake] [--out <dir>]

Reads cache2/recomb/<backbone>/rows.jsonl and work/<backbone>/parts[_tag]/part_*.pt (rc_capture.py); writes, each with one rename,
    <backbone>/keys.pt            {'rids' (row ids, the order of rows.jsonl), 'K_last' FloatTensor (n, d) float32, 'last_prompt_idx',
                                   'seq_len', 'n_image_tokens', 'layer', 'backbone', 'module_dtype', 'rows_sha256', 'doc'}
    <backbone>/capture_meta.json  md5 of keys.pt, the parts, the bitwise verification of the capture path, the re-capture check,
                                  label-free consistency checks against the key cache
    <backbone>/README.txt         what the directory holds (the style of the hard negatives' README)
keys.pt holds no time stamp: assembling the same parts twice gives the same bytes (tests/t_store.py).  A missing part, a part of
another rows.jsonl, or a missing / failed verify_capture.json is an error.  --out / --tag: write elsewhere (tests).
Reader: recomb.store(backbone) (core.py).
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as R  # noqa: E402
from rc_common import C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("backbone", choices=R.BACKBONES)
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
ap.add_argument("--limit", type=int, default=0, help="assemble only the first N rows (smoke / fake tests)")
args = ap.parse_args()
BB = args.backbone
CFG = C.BACKBONES[BB]
VF = CFG["vlkeb"]
DIR = R.bb_dir(BB)
if (args.tag or args.limit) and not args.out:
    raise SystemExit("--tag / --limit assemble a test store: give --out <dir> (the published directory holds the real keys only)")
OUT = args.out or DIR
PD = os.path.join(R.WORK, BB, "parts" + (f"_{args.tag}" if args.tag else ""))
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
rows = R.jsonl_read(R.rows_path(BB))
sel_sha = R.file_sha256(R.rows_path(BB))
if args.limit:
    rows = rows[:args.limit]
real = not args.tag and not args.limit

# ----------------------------------------------------------------------------- parts
files = sorted(f for f in os.listdir(PD) if f.startswith("part_") and f.endswith(".pt"))
pos = {r["rid"]: i for i, r in enumerate(rows)}
K, lp, sl, ni, part_meta, seen = None, None, None, None, [], np.zeros(len(rows), dtype=bool)
dtype = None
for f in files:
    b = torch.load(os.path.join(PD, f), map_location="cpu", weights_only=False)        # our own part files
    if b["rows_sha256"] != sel_sha:
        raise SystemExit(f"{f} was captured for another rows.jsonl (sha256 {b['rows_sha256'][:12]} != {sel_sha[:12]}): delete the stale parts")
    idx = np.array([pos.get(r, -1) for r in b["rids"]], dtype=np.int64)
    if args.limit:
        keep = idx >= 0
        idx, b["K"] = idx[keep], b["K"][torch.from_numpy(keep)]
        for k in ("last_prompt_idx", "seq_len", "n_image_tokens"):
            b[k] = [v for v, m in zip(b[k], keep) if m]
    if (idx < 0).any() or seen[idx].any():
        raise SystemExit(f"{f}: unknown or repeated row ids")
    if K is None:
        dtype = b["dtype"]
        K = torch.empty(len(rows), b["K"].shape[1], dtype=torch.float32)
        lp, sl, ni = (np.full(len(rows), -1, dtype=np.int64) for _ in range(3))
    if b["dtype"] != dtype or str(b["K"].dtype) != dtype:
        raise SystemExit(f"{f}: module dtype {b['dtype']} / {b['K'].dtype}, expected {dtype}")
    ti = torch.from_numpy(idx)
    K[ti] = b["K"].float()                                              # exact image of the module-dtype values
    lp[idx], sl[idx], ni[idx] = b["last_prompt_idx"], b["seq_len"], b["n_image_tokens"]
    seen[idx] = True
    part_meta.append({"part": b["part"], "n": int(len(idx)), "seconds": round(float(b["seconds"]), 1), "gpu": b["gpu"], "torch": b["torch"],
                      "cuda_visible": b["cuda_visible"], "time": b["time"], "master": b["master"], "pid": b.get("pid")})
if K is None or not seen.all():
    raise SystemExit(f"{int((~seen).sum()) if K is not None else len(rows)} rows without a captured part under {PD}: run rc_capture.py again")
if not bool(torch.isfinite(K).all()):
    raise SystemExit("non-finite key")
R.say(f"{BB}: {len(files)} parts, {len(rows)} rows, module dtype {dtype}, d = {K.shape[1]} ({time.time() - t0:.0f} s)")

# ----------------------------------------------------------------------------- checks (label-free)
checks = {}
checks["last_prompt_index_is_last_token"] = bool((lp == sl - 1).all())
groups = collections.defaultdict(list)
for i, r in enumerate(rows):
    groups[r["query_id"]].append(i)
dup = [g for g in groups.values() if len(g) > 1]
same = sum(all(torch.equal(K[g[0]], K[j]) and lp[g[0]] == lp[j] for j in g[1:]) for g in dup)
checks["duplicate_query_groups_with_identical_keys"] = [int(same), len(dup)]
if not args.tag:
    ix = {(e["family"], e["kind"], e.get("edit")): e for e in C.jsonl_read(os.path.join(R.CACHE, BB, "index.jsonl")) if e["kind"] == "rel"}
    img_ok = txt_ok = img_n = txt_n = 0
    for i, r in enumerate(rows):
        e = ix[(VF, "rel", r["edit_pos"])]
        if r["kind"] in ("X-I", "X-X"):                                # the edit's image: as many image tokens as its own Rel sequence
            img_n += 1
            img_ok += int(ni[i] == e["n_image_tokens"])
        else:                                                           # the edit's question: as many text tokens as its own Rel prompt
            txt_n += 1
            txt_ok += int(sl[i] - ni[i] == e["len_prompt"] - e["n_image_tokens"])
    checks["X-I_X-X_rows_with_the_image_token_count_of_the_edit's_own_sequence"] = [img_ok, img_n]
    checks["X-Q_rows_with_the_text_token_count_of_the_edit's_own_prompt"] = [txt_ok, txt_n]
norms = K.double().norm(dim=1).numpy()
checks["key_norm_min_median_max"] = [float(v) for v in np.quantile(norms, [0.0, 0.5, 1.0])]
ok = checks["last_prompt_index_is_last_token"] and same == len(dup) and all(v[0] == v[1] for k, v in checks.items() if k.startswith("X-"))

# ----------------------------------------------------------------------------- verification records of the capture
ver = rec = None
if not args.tag:
    vp, cp = os.path.join(DIR, "verify_capture.json"), os.path.join(DIR, "recheck_capture.json")
    if not os.path.exists(vp) or not R.jload(vp).get("ok"):
        raise SystemExit("verify_capture.json is missing or failed: the capture path is not proven")
    v = R.jload(vp)
    ver = {k: v[k] for k in ("ok", "ok_prompt_only_key", "ok_teacher_forced_key", "batch_size", "n_rows", "summary", "master_log", "module_dtype", "gpu",
                             "torch", "time", "code", "what")}
    if os.path.exists(cp):
        c = R.jload(cp)
        rec = {k: c[k] for k in ("ok", "n", "n_bitwise", "by_kind", "what", "second_capture", "same_process_as_first_capture")}

# ----------------------------------------------------------------------------- keys.pt
kp = os.path.join(OUT, "keys.pt")
blob = {"rids": [r["rid"] for r in rows], "K_last": K, "last_prompt_idx": [int(v) for v in lp], "seq_len": [int(v) for v in sl],
        "n_image_tokens": [int(v) for v in ni], "layer": CFG["layer"], "backbone": BB, "module_dtype": dtype, "rows_sha256": sel_sha,
        "doc": "K_last[r] = input of the edited module (layer " + str(CFG["layer"]) + ", mlp.down_proj) at the LAST PROMPT TOKEN of the PROMPT-ONLY sequence "
               "(no answer tokens) of recombination row rids[r] = line r of rows.jsonl, under the PRE-EDIT model, B = 1; float32 image of the "
               "module-dtype values (exact).  The quantity of keys_decode.pt[row_decode] of the backbone's key cache (10_final/cache), captured "
               "through the same functions (rc_capture.py; verify_capture.json: bit for bit on existing hard-negative and probe rows)."}
tmp = kp + f".tmp{os.getpid()}"
torch.save(blob, tmp)
os.replace(tmp, kp)
kmd5 = R.file_md5(kp)
by_kind = {k: sum(r["kind"] == k for r in rows) for k in R.KINDS}
meta = {"backbone": BB, "family": VF, "layer": CFG["layer"], "module": f"input of model.language_model.layers.{CFG['layer']}.mlp.down_proj",
        "module_dtype": dtype, "d_in": int(K.shape[1]), "master_log": CFG["families"][VF], "n_rows": len(rows), "n_by_kind": by_kind,
        "keys_md5": kmd5, "keys_bytes": os.path.getsize(kp), "keys_shape": list(K.shape), "rows_sha256": sel_sha, "complete": bool(real),
        "position": "last prompt token of the prompt-only sequence (no answer tokens); pre-edit model; B = 1",
        "checks": checks, "checks_ok": bool(ok), "verify": ver, "recheck": rec,
        "last_prompt_idx_min_median_max": [int(v) for v in np.quantile(lp, [0.0, 0.5, 1.0])],
        "n_image_tokens_min_median_max": [int(v) for v in np.quantile(ni, [0.0, 0.5, 1.0])],
        "capture": {"n_parts": len(part_meta), "gpu_seconds": round(sum(p["seconds"] for p in part_meta), 1), "gpus": sorted({p["gpu"] for p in part_meta}),
                    "torch": sorted({p["torch"] for p in part_meta}), "cuda_visible_devices": sorted({str(p["cuda_visible"]) for p in part_meta}),
                    "first": min(p["time"] for p in part_meta), "last": max(p["time"] for p in part_meta),
                    "seconds_per_row": round(sum(p["seconds"] for p in part_meta) / len(rows), 4), "parts": part_meta},
        "assembled": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code": "02_research/experiments/10_final/feat2/recomb/{rc_select,rc_check,rc_capture,rc_assemble}.py"}
R.jdump(meta, os.path.join(OUT, "capture_meta.json"))
if not ok:
    raise SystemExit(f"consistency checks failed: {checks}")


# ----------------------------------------------------------------------------- README.txt (rendered from the files)
def readme() -> str:
    s = R.jload(os.path.join(DIR, "select_summary.json"))
    ck = R.jload(os.path.join(DIR, "check_select.json")) if os.path.exists(os.path.join(DIR, "check_select.json")) else None
    red = s["redraws"]
    T = s["T"]
    L = []
    L.append(f"RECOMBINATIONS of backbone {BB}: log-derived near-scope calibration rows of the variant rescope_x / <gate>_x (PROTOCOL.md section 6,")
    L.append("amendment \"Review round 1\", item R5).  They are CALIBRATION INPUT (per-edit baselines), never test rows: no label, no answer, no complaint;")
    L.append("their queries are disjoint from every test query (check_select.json).  Built by 10_final/feat2/recomb: rc_select.py (selection, CPU,")
    L.append(f"seed {s['seed']}) -> rc_check.py -> rc_capture.py (keys, GPU) -> rc_assemble.py.  Reader: recomb.store('{BB}') / recomb.log_rows(log)")
    L.append("(feat2/recomb/core.py, USAGE.txt).")
    L.append("")
    L.append(f"Rows: {s['n_rows']} = {s['m']} of each kind for each of the {T} edits of the long stream of family {VF} ({s['master_log']}): {s['n_by_kind']}")
    L.append("Usable by a log with the first T edits (owner position < T; X-X partner position < T = row field min_T <= T):")
    for t, v in s["rows_usable_by_log_length_T"].items():
        L.append(f"    T = {t:>4}: X-I {v['X-I']:>5}  X-Q {v['X-Q']:>5}  X-X {v['X-X']:>5}" + (f"   ({v['edits_with_fewer_than_m_cross_rows']} edits with fewer than {s['m']} X-X rows)"
                                                                                          if v["edits_with_fewer_than_m_cross_rows"] else ""))
    L.append("")
    L.append("CONSTRUCTION (modelled on the hard negatives: 10_final/cache/hard_negatives/" + BB + "/README.txt)")
    L.append("  source pool   the source pool of the hard negatives, recomputed and checked against its record: VLKEB eval items that are not probe")
    L.append(f"                items and share no (image, src, alt), no Freebase MID and no entity name with ANY edit of the project's VLKEB edit streams")
    L.append(f"                ({s['source_pool']['n_items']} items, {s['source_pool']['n_distinct_questions']} distinct image-locality questions).")
    L.append("  X-I           the edit's image + the image-locality question (m_loc_q) of a pool item (seeded permutation of the pool per edit).")
    L.append("  X-Q           the edit's question (src) + the entity image (`image`) of a pool item (seeded permutation per edit).")
    L.append("  X-X           the edit's image + the question (src) of another edit j of the same stream (edit_b_pos = j, edit_b_uid):")
    L.append("                stream position < 100: 2 partners from the edit's batch of 10 (scope 'batch'), then 2 from its group of 100 (scope 'group');")
    L.append(f"                later positions: 4 from its group of 100.  Counts: {s['x_x']['scope']}; batches that could not supply 2 partners: "
             f"{s['x_x']['batch_shortfall']['n_edits']}.")
    L.append("  discarded and redrawn (the next candidate of the permutation), in this order:")
    L.append("    own_question  the question string (stripped, lower case) is the question or the rephrase of the edit            [X-I, X-X]")
    L.append("    test_query    (image content = md5 of the file, question string) is the query of ANY row of a key cache under 10_final/cache")
    L.append(f"                  ({', '.join(c for c in s['test_query_sources'] if not c.startswith('n_'))}: edit prompts and rephrases, probe rows, hard negatives of all kinds,")
    L.append("                  benign rows)")
    L.append("    related       X-X: the two edits share an entity (Freebase MID), a name or an answer (normalised pred / alt)")
    L.append("    repeat        (image content, question string) repeats an earlier recombination of the same edit (any kind)")
    L.append("  redraws       " + "; ".join(f"{k}: {red[k]['n_redraws_total']} discarded candidates for {red[k]['n_rows']} rows ({red[k]['by_reason']}; "
                                            f"{red[k]['n_rows_with_a_redraw']} rows with a redraw, at most {red[k]['max_per_row']})" for k in R.KINDS))
    L.append("")
    L.append("FLAGS (as the hard negatives' rows carry them)")
    L.append("  q_in_edit / q_in_edit_first_pos   the row's question string is the question or rephrase of an edit of the stream; first such stream position")
    L.append("                (a row is 'cross-like' for a log with T edits iff that position < T).  Rows with the flag: "
             f"{s['flags']['question_string_also_an_edit_question_or_rephrase']} (X-Q and X-X by construction);")
    L.append(f"                X-I rows that are cross-like by log length: {s['flags']['X-I_rows_cross_like_by_log_length_T(q_in_edit_first_pos < T)']}")
    L.append(f"  same_category the question shares a category of hn_build.py's lexicon with the edit's question or rephrase (X-Q: the partner item's own")
    L.append(f"                question with the edit's question): {s['flags']['same_category']}")
    L.append(f"  q_in_probe {s['flags']['q_in_probe']}, img_in_probe {s['flags']['img_in_probe']}, query_in_probe {s['flags']['query_in_probe']} (against the probe rows of the family)")
    L.append("  n_redraws / redraws   candidates discarded before the row was accepted, by reason")
    sh = s["shared_with_test_hard_negatives_of_the_backbone(same generator, other query)"]
    L.append("  NOT a flag, but to be known: partner items come from the SAME source pool as the test hard negatives.  "
             f"{sh['X-I_rows_whose_question_string_is_the_question_of_a_test_HN-I_row']} X-I rows ask a question string")
    L.append(f"  that a test HN-I row asks about another image ({sh['n_distinct_question_strings_of_test_HN-I_rows']} distinct strings), "
             f"{sh['X-Q_rows_whose_image_is_the_image_of_a_test_HN-Q_row']} X-Q rows show an image that a test HN-Q row shows with another question")
    L.append(f"  ({sh['n_distinct_images_of_test_HN-Q_rows']} distinct images).  No recombination QUERY (image content + question) is a test query.")
    L.append("")
    L.append("OTHER FIELDS of rows.jsonl: rid ('<kind>:<edit uid>:<k>'), kind, family, edit_uid, edit_pos (= edit: the OWNING edit's stream position), k,")
    L.append("image, prompt, fmt ('vqa'), ptype ('image'), image_md5, src_uid, src_field, edit_b_uid, edit_b_pos (= edit_b), scope, min_T, q_cats,")
    L.append("edit_cats, edit_src, edit_rephrase, edit_alt, edit_pred, src_item_question / src_item_pred (X-Q), edit_b_src / edit_b_alt / edit_b_pred (X-X),")
    L.append(f"query_id / n_same_query (rows of this file with the same query: {s['queries']['n_distinct(image content, prompt)']} distinct queries; "
             f"rows in duplicate groups {s['queries']['rows_in_duplicate_query_groups']}: two edits with the same question drew the same pool image).")
    L.append("There is NO pre-edit answer, no teacher-forced sequence and no validity flag: only the last-prompt key is captured.")
    L.append("")
    L.append(f"keys.pt: rids (the order of rows.jsonl), K_last FloatTensor {list(K.shape)} float32, last_prompt_idx, seq_len, n_image_tokens, layer, backbone,")
    L.append(f"module_dtype ({dtype}), rows_sha256, doc.  K_last[r] = input of the edited module (layer {CFG['layer']}, mlp.down_proj) at the last prompt token of the")
    L.append("PROMPT-ONLY sequence (no answer tokens) of row r, pre-edit model, B = 1 (float32 image of the module-dtype values: exact).  This is the quantity")
    L.append("of keys_decode.pt[row_decode] of the key cache, NOT the teacher-forced key that hard_negatives/<backbone>/keys.pt holds (a recombination has")
    L.append("no answer).  md5 " + kmd5 + f"; rows.jsonl sha256 {sel_sha}.")
    if ver is not None:
        hs, ps = ver["summary"]["hard_negatives"], ver["summary"]["probes"]
        L.append("")
        L.append("CAPTURE PATH (rc_capture.py: fc_capture.make_evaluator / make_item, Evaluator.batch(it, answer=False), Evaluator.decode_keys(it, 0))")
        L.append(f"  proof before the run (verify_capture.json): {hs['n']} existing hard-negative rows ({hs['by_kind']}) and {ps['n']} probe rows ({ps['by_ptype']})")
        L.append(f"  re-captured at B = {ver['batch_size']}: prompt-only key bit for bit equal to keys_decode.pt {hs['prompt_key_bitwise'][0]}/{hs['prompt_key_bitwise'][1]} and "
                 f"{ps['prompt_key_bitwise'][0]}/{ps['prompt_key_bitwise'][1]}; teacher-forced key equal to keys_tf.pt")
        L.append(f"  {hs['tf_key_bitwise'][0]}/{hs['tf_key_bitwise'][1]} and {ps['tf_key_bitwise'][0]}/{ps['tf_key_bitwise'][1]} (hard negatives also equal to hard_negatives/{BB}/keys.pt: "
                 f"{hs['tf_key_equals_hn_keys_pt'][0]}/{hs['tf_key_equals_hn_keys_pt'][1]}).")
        if rec is not None:
            L.append(f"  after the run (recheck_capture.json): {rec['n_bitwise']}/{rec['n']} sampled recombination rows bit for bit identical on a second capture without")
            L.append(f"  the vision-feature memo ({rec['same_process_as_first_capture'][0]} of them in the process of their first capture; the proof across processes is the")
            L.append("  comparison with the cached keys above).")
        L.append(f"  label-free consistency (capture_meta.json): last prompt index = last token of the sequence: {checks['last_prompt_index_is_last_token']}; duplicate-query groups")
        L.append(f"  with identical keys {checks['duplicate_query_groups_with_identical_keys'][0]}/{checks['duplicate_query_groups_with_identical_keys'][1]}; "
                 + "; ".join(f"{k.replace('_', ' ')} {v[0]}/{v[1]}" for k, v in checks.items() if k.startswith("X-")) + ".")
        L.append(f"  {meta['capture']['n_parts']} parts, {meta['capture']['gpu_seconds']} GPU-seconds ({meta['capture']['seconds_per_row']} s per row), {meta['capture']['gpus']}, "
                 f"torch {meta['capture']['torch']}, master log {meta['master_log']}.")
    if ck is not None:
        c = ck["checks"]
        d = c["disjoint_from_every_test_query"]
        L.append("")
        L.append(f"SELECTION CHECK (rc_check.py -> check_select.json; ok = {ck['ok']}): two re-runs in fresh interpreters reproduce rows.jsonl and")
        how = "image md5 recomputed" if c["image_md5_recomputed_from_files"]["recomputed"] else "image md5 as stored, not recomputed,"
        L.append(f"select_summary.json byte for byte; {d['n_recombination_rows']} rows against {d['n_test_rows_with_an_image']} test rows with an image "
                 f"({d['n_distinct_test_queries_normalised']} distinct queries; {how}")
        L.append(f"from {c['image_md5_recomputed_from_files']['n_image_files']} files): {d['n_rows_equal_to_a_test_query_exact']} equal exactly, "
                 f"{d['n_rows_equal_to_a_test_query_normalised']} equal after normalising the question; rule violations {c['rules']['violations'] or 0}.")
        L.append("Test rows compared: every image sequence of the key caches " + ", ".join(c["caches_compared(information)"]["caches_present"])
                 + ", their hard-negative and benign row files, the raw edit stream and probe records of the family, the one-hop portability queries.")
    L.append("")
    L.append("FILES  rows.jsonl, select_summary.json, check_select.json, keys.pt, capture_meta.json, verify_capture.json, recheck_capture.json, README.txt;")
    L.append("       ../work/<backbone>/parts (capture parts, resumable), ../logs (capture logs).")
    return "\n".join(L) + "\n"


if not args.tag and not args.limit:
    txt = readme()
    tmp = os.path.join(OUT, f"README.txt.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(txt)
    os.replace(tmp, os.path.join(OUT, "README.txt"))
R.say(f"  wrote {kp} ({os.path.getsize(kp)} bytes, md5 {kmd5}); checks {checks} ({time.time() - t0:.0f} s)")
