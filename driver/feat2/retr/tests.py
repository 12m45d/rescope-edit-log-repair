"""Tests of the retrieval reference / lookup layer.

    python tests.py                       every CPU test, then the table
    python tests.py <name> [...]          selected tests (names: TESTS below)
    $GQ run --mem 2000 --prio 4 --name retr_t -- python tests.py reembed_images_gpu          GPU test 1 (50 images, 1814 MiB)
    $GQ run --mem 10500 --prio 4 --name retr_b -- python tests.py behaviour_smoke            GPU test 2 (Qwen development log through rescope.Evaluator)
    RETR_SMOKE_LOG=TB-L100-ae-l2_3 $GQ run --mem 17000 --prio 4 --name retr_bL -- python tests.py behaviour_smoke      the same on the LLaVA development log
    $GQ run --mem 10500 --prio 4 --name retr_fx -- python tests.py behaviour_fx              GPU test 3 (the wiring through fx.behave, private temporary memo)
    python tests.py table                 out/test_table.txt from out/tests/*.json

Every test writes out/tests/<name>.json = {ok, summary, ...}.  Gate decisions are looked at on the two development logs only
(rcommon.DEV_LOGS); the label-free checks (attributes, embeddings, own requests) cover every edit of the long streams.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rcommon as C  # noqa: E402

import numpy as np  # noqa: E402

TDIR = os.path.join(C.OUT, "tests")
SEED = 20261111
N_RE = 50


def _q(v, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    v = np.asarray(v, dtype=np.float64)
    return {f"q{int(round(100 * p)):02d}" if p not in (0.0, 1.0) else ("min" if p == 0.0 else "max"): float(np.quantile(v, p)) for p in qs}


# ----------------------------------------------------------------------------- 1. selections reproduce byte for byte
def t_repro():
    """A re-run of the attribute tables and of the key lists reproduces the files byte for byte."""
    import build_attributes as BA
    import embed as EM
    tmp = os.path.join(C.OUT, "tmp_repro")
    shutil.rmtree(tmp, ignore_errors=True)
    res, ok = {}, True
    for bb in C.BACKBONES:
        BA.build(bb, os.path.join(tmp, bb), verbose=False)
        same = filecmp.cmp(os.path.join(tmp, bb, "attributes.jsonl"), os.path.join(C.DATA, bb, "attributes.jsonl"), shallow=False)
        res[f"{bb}/attributes.jsonl"] = {"byte_identical": same, "md5": C.file_md5(os.path.join(C.DATA, bb, "attributes.jsonl"))}
        ok &= same
    EM.build_keys(os.path.join(tmp, "emb"), data_dir=tmp)                # from the re-built tables
    for f in ("image_keys.jsonl", "question_keys.jsonl"):
        same = filecmp.cmp(os.path.join(tmp, "emb", f), os.path.join(C.EMB, f), shallow=False)
        res[f"emb/{f}"] = {"byte_identical": same, "md5": C.file_md5(os.path.join(C.EMB, f))}
        ok &= same
    import make_readme
    import mechanics
    mechanics.main(tmp, verbose=False)                                   # the mechanics check into the temporary directory
    for f in ("mechanics.txt", "mechanics.json"):
        same = os.path.exists(os.path.join(C.OUT, f)) and filecmp.cmp(os.path.join(tmp, f), os.path.join(C.OUT, f), shallow=False)
        res[f"out/{f}"] = {"byte_identical": bool(same)}
        ok &= bool(same)
    with open(os.path.join(C.DATA, "README.txt"), encoding="utf-8") as fh:
        same = fh.read() == make_readme.main(write=False, verbose=False) + "\n"
    res["README.txt"] = {"byte_identical": bool(same)}
    ok &= bool(same)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"ok": bool(ok), "summary": f"{sum(v['byte_identical'] for v in res.values())}/{len(res)} files byte-identical on a re-run (attribute tables, key lists, "
                                       "mechanics.txt / .json, README.txt)", "files": res}


# ----------------------------------------------------------------------------- 2. attributes
def t_attributes():
    """The checks made while building the tables (every sequence), and an independent re-check of a seeded sample."""
    import build_attributes as BA
    out, ok = {}, True
    for bb in C.BACKBONES:
        m = C.jload(os.path.join(C.DATA, bb, "attributes_meta.json"))
        ch = m["checks"]
        n = m["n_sequences"]
        full = (ch["sequences_derived_from_loaders_equal_index"] == n and not any(ch["mismatches"].values()) and ch["md5_mismatches"] == 0
                and ch["prompts_rendered"] == n and ch["rendered_text_is_template_around_question"] == n
                and ch["rendered_token_count_equals_cached_prompt_length"] == n and ch["image_slots_equal_cache"] == n
                and ch["edit_requests_equal_stream_file_and_rel_sequence"] == m["n_edit_requests"] and not any(ch["edit_mismatches"].values()))
        # independent sample: render with the tokenizer from the TABLE's fields (not from the loaders) and compare with the cache index
        rows = C.jsonl_read(os.path.join(C.DATA, bb, "attributes.jsonl"))[:n]
        idx = C.jsonl_read(os.path.join(C.CACHE, bb, "index.jsonl"))
        BA.testbed()
        R = BA.Renderer(bb)
        pick = np.random.default_rng(SEED).choice(n, 300, replace=False)
        good = 0
        for j in pick:
            r, e = rows[int(j)], idx[int(j)]
            c = R.check(e, r["question"], r["image"])
            md5_ok = (r["image"] is None and r["image_md5"] is None) or C.file_md5(BA.image_path(r["image"])) == r["image_md5"]
            good += int(c["text"] and c["len"] and c["image_slots"] and md5_ok and r["sid"] == e["sid"] and r["question_key"] == r["question"].strip().lower())
        out[bb] = {"n_sequences": n, "n_edit_requests": m["n_edit_requests"], "all_sequences_checked_at_build": bool(full), "sample_rendered_ok": f"{good}/300",
                   "n_distinct_image_md5": m["n_distinct_image_md5"], "n_distinct_question_keys": m["n_distinct_question_keys"], "n_text_rows": m["n_text_rows"]}
        ok &= full and good == 300
    return {"ok": bool(ok), "summary": "; ".join(f"{bb}: {v['n_sequences']} sequences + {v['n_edit_requests']} requests derived from the loaders == index, md5 recomputed, "
                                                  f"all prompts rendered; sample {v['sample_rendered_ok']}" for bb, v in out.items()), "backbones": out}


# ----------------------------------------------------------------------------- 3. embeddings: unit norm, grid
def t_unit_norm():
    from encoders import on_grid
    meta = C.jload(os.path.join(C.EMB, "meta.json"))
    out, ok = {}, True
    for kind in ("image", "question"):
        E = np.load(os.path.join(C.EMB, f"{kind}_emb.npy"))
        keys = C.jsonl_read(os.path.join(C.EMB, f"{kind}_keys.jsonl"))
        nrm = np.sqrt((E.astype(np.float64) ** 2).sum(1))
        o = {"shape": list(E.shape), "dtype": str(E.dtype), "n_keys": len(keys), "max_abs_norm_minus_1": float(np.abs(nrm - 1).max()), "finite": bool(np.isfinite(E).all()),
             "on_grid_2^-24": on_grid(E), "md5_equals_meta": C.file_md5(os.path.join(C.EMB, f"{kind}_emb.npy")) == meta[kind]["md5"],
             "keys_sorted_unique": [r["md5" if kind == "image" else "question_key"] for r in keys] == sorted({r["md5" if kind == "image" else "question_key"] for r in keys})}
        o["ok"] = bool(E.dtype == np.float32 and E.shape[0] == len(keys) and o["max_abs_norm_minus_1"] <= 1e-6 and o["finite"] and o["on_grid_2^-24"]
                       and o["md5_equals_meta"] and o["keys_sorted_unique"])
        out[kind] = o
        ok &= o["ok"]
    return {"ok": bool(ok), "summary": "; ".join(f"{k}: {v['shape']} float32, max | |e| - 1 | = {v['max_abs_norm_minus_1']:.1e}, on the 2^-24 grid" for k, v in out.items()), **out}


# ----------------------------------------------------------------------------- 4. / 5. re-embedding
def _sample(kind):
    keys = C.jsonl_read(os.path.join(C.EMB, f"{kind}_keys.jsonl"))
    pick = np.sort(np.random.default_rng([SEED, 0 if kind == "image" else 1]).choice(len(keys), N_RE, replace=False))
    return keys, pick


def _cmp(new, old):
    d = np.abs(new.astype(np.float64) - old.astype(np.float64))
    return {"max_abs_diff": float(d.max()), "n_rows_bitwise_equal": int((new == old).all(1).sum()), "n": int(new.shape[0]),
            "min_cosine": float((new.astype(np.float64) * old.astype(np.float64)).sum(1).min())}


def t_reembed_questions():
    """50 questions re-embedded: sentence-transformers (the build's path) and the model card's transformers recipe."""
    import torch
    torch.set_num_threads(4)
    from encoders import QuestionEncoder
    keys, pick = _sample("question")
    old = np.load(os.path.join(C.EMB, "question_emb.npy"))[pick]
    strs = [keys[int(i)]["question_key"] for i in pick]
    a = _cmp(QuestionEncoder("sentence-transformers").embed(strs), old)
    enc_t = QuestionEncoder("transformers")
    b = _cmp(enc_t.embed(strs), old)
    # the same strings in another order (the build embeds one string per forward: order cannot matter)
    perm = np.random.default_rng(SEED).permutation(N_RE)
    c = _cmp(QuestionEncoder("sentence-transformers").embed([strs[int(i)] for i in perm]), old[perm])
    ok = max(a["max_abs_diff"], b["max_abs_diff"], c["max_abs_diff"]) <= 1e-6
    return {"ok": bool(ok), "summary": f"{N_RE} questions re-embedded: max |diff| {a['max_abs_diff']:.1e} (sentence-transformers, {a['n_rows_bitwise_equal']}/{N_RE} bitwise), "
                                       f"{b['max_abs_diff']:.1e} (model card recipe through transformers), {c['max_abs_diff']:.1e} (other order)",
            "sentence_transformers": a, "transformers_model_card_recipe": b, "other_order": c, "max_tokens_in_sample": max(enc_t.n_tokens(s) for s in strs)}


def t_reembed_questions_all():
    """A complete re-run of the question embeddings (6424 strings, CPU) reproduces the stored matrix bit for bit."""
    import torch
    torch.set_num_threads(4)
    from encoders import QuestionEncoder
    keys = [r["question_key"] for r in C.jsonl_read(os.path.join(C.EMB, "question_keys.jsonl"))]
    old = np.load(os.path.join(C.EMB, "question_emb.npy"))
    new = QuestionEncoder("sentence-transformers").embed(keys)
    same = bool(np.array_equal(new, old))
    return {"ok": same, "summary": f"all {len(keys)} questions re-embedded: matrix bitwise identical = {same} (max |diff| {float(np.abs(new.astype(np.float64) - old).max()):.1e})",
            "n": len(keys), "bitwise_identical": same, "n_rows_bitwise_equal": int((new == old).all(1).sum())}


def _reembed_images(device):
    import torch
    torch.set_num_threads(4)
    import embed as EM
    from encoders import ImageEncoder
    if device.startswith("cuda"):
        EM.gpu_cap()
    keys, pick = _sample("image")
    old = np.load(os.path.join(C.EMB, "image_emb.npy"))[pick]
    paths = [EM.image_path(keys[int(i)]["image"]) for i in pick]
    ok_md5 = all(C.file_md5(p) == keys[int(i)]["md5"] for p, i in zip(paths, pick))
    enc = ImageEncoder(device)
    t0 = time.time()
    a = _cmp(enc.embed_paths(paths), old)
    perm = np.random.default_rng(SEED).permutation(N_RE)
    c = _cmp(enc.embed_paths([paths[int(i)] for i in perm]), old[perm])
    new = enc.embed_paths(paths).astype(np.float64)
    dcos = float(np.abs(new @ new.T - old.astype(np.float64) @ old.astype(np.float64).T).max())      # effect on a score: the 50 x 50 cosines
    out = {"device": device, "files_md5_ok": bool(ok_md5), "same_order": a, "other_order": c, "max_abs_change_of_a_cosine_among_the_sample": dcos,
           "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
           "nvidia_smi_mib": EM.gpu_used_mib() if device.startswith("cuda") else None, "weights_sha1": enc.info["weights_sha1"]}
    out["ok"] = bool(ok_md5 and max(a["max_abs_diff"], c["max_abs_diff"]) <= 1e-6)
    out["summary"] = (f"{N_RE} images re-embedded on {device}: max |diff| {a['max_abs_diff']:.1e} ({a['n_rows_bitwise_equal']}/{N_RE} bitwise), "
                      f"other order {c['max_abs_diff']:.1e}; largest change of a cosine among them {dcos:.1e}")
    return out


def t_reembed_images_gpu():
    return _reembed_images("cuda:0")


def t_reembed_images_cpu():
    """The same 50 images on the CPU (another device than the build's): how far the embedding moves across devices.  The 1e-6
    criterion of the re-embedding test is applied unchanged; a larger deviation is reported as such (status), not hidden."""
    r = _reembed_images("cpu")
    r["note"] = "cross-device check: the stored embeddings were computed on the GPU in float32; this run is the CPU in float32"
    if not r["ok"] and r["files_md5_ok"]:
        r["status"] = "exceeds 1e-6"
        r["summary"] += "  -> NOT within 1e-6 across devices (float32 forward); the stored files are the reference"
    return r


# ----------------------------------------------------------------------------- facts about the attributes (label-free, every stream)
def t_facts():
    """Descriptive facts that matter for reading retr / lookup: questions that repeat across edits, images shared between items.
    Attributes only (no score, no gate); every edit of every stream of both caches."""
    import collections
    out = {}
    for cache in C.BACKBONES:
        rows = C.jsonl_read(os.path.join(C.DATA, cache, "attributes.jsonl"))
        seq = [r for r in rows if r["table"] == "sequence"]
        idx = C.jsonl_read(os.path.join(C.CACHE, cache, "index.jsonl"))
        for fam in sorted({r["family"] for r in rows if r["table"] == "edit"}):
            req = [r for r in rows if r["table"] == "edit" and r["family"] == fam]
            T = len(req)
            by = {k: {r["edit"]: r for r in seq if r["family"] == fam and r["kind"] == k} for k in C.EDIT_KINDS}
            qc = collections.Counter(r["question_key"] for r in req)
            top = qc.most_common(3)
            eq = {r["question_key"] for r in req}
            eimg = collections.Counter(r["image_md5"] for r in req)
            rimg = collections.Counter(by["igen"][i]["image_md5"] for i in range(T))
            o = {"T": T, "n_distinct_edit_questions": len(qc), "n_edits_whose_question_is_asked_by_another_edit": sum(qc[r["question_key"]] > 1 for r in req),
                 "most_frequent_edit_questions": [[q, n] for q, n in top],
                 "n_edits_whose_text_rephrase_equals_their_question": sum(by["tgen"][i]["question_key"] == req[i]["question_key"] for i in range(T)),
                 "n_edits_whose_text_rephrase_is_the_question_of_another_edit": sum(by["tgen"][i]["question_key"] != req[i]["question_key"] and by["tgen"][i]["question_key"] in eq
                                                                                    for i in range(T)),
                 "n_distinct_edit_images": len(eimg), "n_edits_whose_image_is_the_image_of_another_edit": sum(eimg[r["image_md5"]] > 1 for r in req),
                 "n_edits_whose_image_rephrase_is_their_own_image": sum(by["igen"][i]["image_md5"] == req[i]["image_md5"] for i in range(T)),
                 "n_edits_whose_image_rephrase_is_shared_with_another_edit": sum(rimg[by["igen"][i]["image_md5"]] > 1 for i in range(T)),
                 "n_edits_whose_image_rephrase_is_the_image_of_another_edit": sum(by["igen"][i]["image_md5"] != req[i]["image_md5"] and by["igen"][i]["image_md5"] in eimg
                                                                                  for i in range(T))}
            allimg = set(eimg) | set(rimg)
            for kind, sel in (("probe", lambda r, e: r["kind"] == "probe" and r["family"] == fam), ("benign", lambda r, e: r["kind"] == "benign"),
                              ("hn", lambda r, e: r["kind"] in C.HN_KINDS and r["family"] == fam)):
                sub = [(r, e) for r, e in zip(seq, idx) if sel(r, e)]
                if not sub:
                    continue
                o[kind] = {"n": len(sub), "n_with_image": sum(r["image_md5"] is not None for r, _ in sub),
                           "n_whose_question_is_an_edit_question": sum(r["question_key"] in eq for r, _ in sub),
                           "n_whose_image_is_an_edit_image": sum(r["image_md5"] in eimg for r, _ in sub if r["image_md5"] is not None),
                           "n_whose_image_is_an_edit_image_or_image_rephrase": sum(r["image_md5"] in allimg for r, _ in sub if r["image_md5"] is not None)}
                if kind == "probe":
                    o[kind]["n_valid_repair_rows_whose_question_is_an_edit_question"] = sum(r["question_key"] in eq and e.get("pool") == "repair" and bool(e.get("y0_valid")) for r, e in sub)
                    o[kind]["n_valid_repair_rows"] = sum(e.get("pool") == "repair" and bool(e.get("y0_valid")) for _, e in sub)
                    pq = {r["question_key"] for r, e in sub}
                    o["n_edits_whose_question_is_a_probe_question"] = sum(r["question_key"] in pq for r in req)
            out[f"{cache}/{fam}"] = o
    km = C.jload(os.path.join(C.EMB, "keys_meta.json"))
    out["embedding_keys"] = {k: km[k] for k in ("n_images", "n_questions", "n_images_shared_by_backbones", "n_questions_shared_by_backbones", "n_image_contents_with_several_paths")}
    f = out
    summ = "; ".join(f"{k}: {v['n_edits_whose_question_is_asked_by_another_edit']}/{v['T']} edits share their question with another edit ({v['n_distinct_edit_questions']} distinct "
                     f"questions), {v['n_edits_whose_question_is_a_probe_question']} ask a probe question; edit images all distinct: {v['n_distinct_edit_images'] == v['T']}; "
                     f"image rephrase = own image for {v['n_edits_whose_image_rephrase_is_their_own_image']}, shared with another edit for "
                     f"{v['n_edits_whose_image_rephrase_is_shared_with_another_edit']}" for k, v in f.items() if k != "embedding_keys")
    return {"ok": True, "summary": summ, "families": {k: v for k, v in out.items() if k != "embedding_keys"}, "embedding_keys": out["embedding_keys"]}


# ----------------------------------------------------------------------------- 6. exact scores
def t_exact_scores():
    """The float64 BLAS dot product of stored embeddings equals exact integer arithmetic; a score does not depend on the batch."""
    import retr
    rng = np.random.default_rng(SEED)
    out, ok = {}, True
    for kind, M in (("image", np.load(os.path.join(C.EMB, "image_emb.npy"))), ("question", np.load(os.path.join(C.EMB, "question_emb.npy")))):
        a, b = rng.choice(M.shape[0], 300, replace=False), rng.choice(M.shape[0], 200, replace=False)
        G = M[a].astype(np.float64) @ M[b].astype(np.float64).T
        ia, ib = np.rint(M[a].astype(np.float64) * 2 ** 24).astype(np.int64), np.rint(M[b].astype(np.float64) * 2 ** 24).astype(np.int64)
        exact = np.array([[int(np.dot(ia[i], ib[j])) for j in range(len(b))] for i in range(len(a))], dtype=object)       # int64 products < 2^49: exact
        Gi = np.array([[float(exact[i, j]) / float(2 ** 48) for j in range(len(b))] for i in range(len(a))])
        out[kind] = {"n_pairs": int(G.size), "n_bitwise_equal_to_integer_arithmetic": int((G == Gi).sum()), "max_abs_integer_sum_log2": float(np.log2(max(abs(int(x)) for x in exact.ravel())))}
        ok &= bool((G == Gi).all())
    shape = {}
    for cache, name in C.DEV_LOGS.items():
        E = retr.edits(name)
        rs = retr.row_sets(E)
        rows = [int(j) for j in rng.choice(rs["probe"] + rs["edit"] + rs["hn"] + rs["benign"], 200, replace=False)]
        SI, SQ = retr.scores(E, rows)
        one = [retr.scores(E, [j]) for j in rows]
        SI1, SQ1 = np.concatenate([o[0] for o in one]), np.concatenate([o[1] for o in one])
        rev = retr.scores(E, rows[::-1])
        same = bool(np.array_equal(SI, SI1) and np.array_equal(SQ, SQ1) and np.array_equal(SI, rev[0][::-1]) and np.array_equal(SQ, rev[1][::-1]))
        shape[name] = {"n_sequences": len(rows), "batch_of_200_equals_one_by_one_and_reversed_bitwise": same}
        ok &= same
    return {"ok": bool(ok), "summary": "; ".join(f"{k}: {v['n_bitwise_equal_to_integer_arithmetic']}/{v['n_pairs']} dot products == exact integer arithmetic" for k, v in out.items())
            + "; scores independent of the batch (bitwise)", "dot": out, "batch_independence": shape}


# ----------------------------------------------------------------------------- 7. own requests
def t_own_request():
    """The edit's own request scores 1 on both attributes: every edit of every stream of both caches.  Label-free: only the score of
    a request against ITSELF is computed here (and of the cached Rel sequence against its own edit); no edit is compared with another."""
    import retr
    out, ok = {}, True
    for cache in C.BACKBONES:
        st = retr.store(cache)
        idx = retr.cache_index(cache)
        for fam, req in st.requests.items():
            E = retr.edits((cache, [r["edit_uid"] for r in req]))
            gI, gQ = retr.own_request_scores(E)
            rel = {int(e["edit"]): j for j, e in enumerate(idx) if e["kind"] == "rel" and e["family"] == fam}
            rows = [rel[i] for i in range(E.T)]
            same_attr = all(st.seq[j]["image_md5"] == r["image_md5"] and st.seq[j]["question"] == r["question"] for j, r in zip(rows, req))
            pI = (np.asarray(st.EI[st.seq_img[rows]], dtype=np.float64) * np.asarray(st.EI[E.img], dtype=np.float64)).sum(1)      # Rel sequence i x edit i
            pQ = (np.asarray(st.EQ[st.seq_q[rows]], dtype=np.float64) * np.asarray(st.EQ[E.q], dtype=np.float64)).sum(1)
            o = {"T": E.T, "image_score_min_max": [float(gI.min()), float(gI.max())], "question_score_min_max": [float(gQ.min()), float(gQ.max())],
                 "max_abs_dev_from_1": float(max(np.abs(gI - 1).max(), np.abs(gQ - 1).max())), "rel_sequence_has_the_request_attributes": bool(same_attr),
                 "rel_sequence_scores_bitwise_equal": bool(np.array_equal(pI, gI) and np.array_equal(pQ, gQ)), "edits_are_a_prefix_of_the_family": bool(E.prefix)}
            o["ok"] = bool(o["max_abs_dev_from_1"] <= 1e-6 and same_attr and o["rel_sequence_scores_bitwise_equal"])
            out[f"{cache}/{fam}"] = o
            ok &= o["ok"]
    for cache, name in C.DEV_LOGS.items():                              # through the public function, on the development logs
        E = retr.edits(name)
        SI, SQ = retr.scores(E, retr.row_sets(E)["edit"][:E.T])
        gI, gQ = retr.own_request_scores(E)
        o = {"diag_of_scores_equals_own_request_scores": bool(np.array_equal(np.diag(SI), gI) and np.array_equal(np.diag(SQ), gQ)),
             "max_abs_dev_from_1": float(max(np.abs(np.diag(SI) - 1).max(), np.abs(np.diag(SQ) - 1).max()))}
        out[name] = o
        ok &= o["diag_of_scores_equals_own_request_scores"] and o["max_abs_dev_from_1"] <= 1e-6
    return {"ok": bool(ok), "summary": "; ".join(f"{k}: {v['T']} edits, | score - 1 | <= {v['max_abs_dev_from_1']:.1e}" for k, v in out.items() if "T" in v)
            + "; retr.scores on the Rel sequences of the development logs: diagonal == 1 (bitwise the own-request scores)", "families": out}


# ----------------------------------------------------------------------------- 8. rephrases (development logs only)
def t_rephrase_scores():
    """Distribution of the two scores of the text / image rephrases against their own edit (development logs)."""
    import retr
    out = {}
    for cache, name in C.DEV_LOGS.items():
        E = retr.edits(name)
        rs = retr.row_sets(E)
        ed = rs["edit"]
        o = {}
        for k, kind in enumerate(C.EDIT_KINDS):
            rows = ed[k * E.T:(k + 1) * E.T]
            SI, SQ = retr.scores(E, rows)
            di, dq = np.diag(SI), np.diag(SQ)
            off = ~np.eye(E.T, dtype=bool)
            o[kind] = {"image_cosine_own_edit": _q(di), "question_cosine_own_edit": _q(dq),
                       "n_image_cosine_ge_1-1e-6": int((di >= 1 - 1e-6).sum()), "n_question_cosine_ge_1-1e-6": int((dq >= 1 - 1e-6).sum()),
                       "image_cosine_other_edits_max": _q(np.where(off, SI, -np.inf).max(1)), "question_cosine_other_edits_max": _q(np.where(off, SQ, -np.inf).max(1)),
                       "n_own_edit_is_argmax_image": int((SI.argmax(1) == np.arange(E.T)).sum()), "n_own_edit_is_argmax_question": int((SQ.argmax(1) == np.arange(E.T)).sum()),
                       "n_own_edit_is_argmax_of_the_sum": int(((SI + SQ).argmax(1) == np.arange(E.T)).sum())}
        out[name] = o
    f = lambda d: f"{d['min']:.3f} / {d['q25']:.3f} / {d['q50']:.3f} / {d['q75']:.3f} / {d['max']:.3f}"      # noqa: E731
    return {"ok": True, "summary": "; ".join(f"{n}: T-Gen question cos (min / q25 / med / q75 / max) {f(o['tgen']['question_cosine_own_edit'])}, image cos = 1 on "
                                             f"{o['tgen']['n_image_cosine_ge_1-1e-6']}/100; I-Gen image cos {f(o['igen']['image_cosine_own_edit'])}, question cos = 1 on "
                                             f"{o['igen']['n_question_cosine_ge_1-1e-6']}/100" for n, o in out.items()), "logs": out}


# ----------------------------------------------------------------------------- 9. naive second implementation (100 sequences)
class Naive:
    """A second implementation of retr and lookup that shares no code with retr.py: it reads the files itself and works in plain
    Python floats, with the DEFINING comparisons (score - baseline > eta), not the stored thresholds."""

    def __init__(self, cache, name):
        import re
        self.re = re
        rows = C.jsonl_read(os.path.join(C.DATA, cache, "attributes.jsonl"))
        self.seq = [r for r in rows if r["table"] == "sequence"]
        uids = [r["uid"] for r in C.jsonl_read(os.path.join(C.TESTBED, name, "edit_batch.jsonl"))]
        req = {r["edit_uid"]: r for r in rows if r["table"] == "edit"}
        self.req = [req[u] for u in uids]
        self.uid_pos = {u: i for i, u in enumerate(uids)}
        self.T = len(uids)
        ik = {r["md5"]: i for i, r in enumerate(C.jsonl_read(os.path.join(C.EMB, "image_keys.jsonl")))}
        qk = {r["question_key"]: i for i, r in enumerate(C.jsonl_read(os.path.join(C.EMB, "question_keys.jsonl")))}
        EI, EQ = np.load(os.path.join(C.EMB, "image_emb.npy")), np.load(os.path.join(C.EMB, "question_emb.npy"))
        self.vi = lambda r: None if r["image_md5"] is None else [float(x) for x in EI[ik[r["image_md5"]]]]          # noqa: E731
        self.vq = lambda r: [float(x) for x in EQ[qk[r["question"].strip().lower()]]]                               # noqa: E731
        self.ei, self.eq = [self.vi(r) for r in self.req], [self.vq(r) for r in self.req]

    @staticmethod
    def dot(a, b):
        import math
        return math.fsum(x * y for x, y in zip(a, b))                   # exactly rounded sum of exact products

    def score(self, r):
        vi, vq = self.vi(r), self.vq(r)
        si = [float("-inf")] * self.T if vi is None else [self.dot(vi, e) for e in self.ei]
        return si, [self.dot(vq, e) for e in self.eq]

    def calibrate(self, cal_rows, alpha, seed):
        n = len(cal_rows)
        perm = np.random.default_rng(seed).permutation(n)
        b1, b2 = sorted(int(i) for i in perm[:n // 2]), sorted(int(i) for i in perm[n // 2:])
        S = [self.score(self.seq[j]) for j in cal_rows]
        a = Fraction(alpha).limit_denominator(10 ** 9)
        k = -((-(a.denominator - a.numerator) * (len(b2) + 1)) // a.denominator)
        self.b, self.eta = [], []
        for t in (0, 1):
            b = [max(S[q][t][i] for q in b1) for i in range(self.T)]
            sc = sorted(max(S[q][t][i] - b[i] for i in range(self.T)) for q in b2)
            self.b.append(b)
            self.eta.append(float("inf") if k > len(b2) else sc[k - 1])
        self.g = [[self.dot(e, e) for e in self.ei], [self.dot(e, e) for e in self.eq]]

    def decide(self, j):
        r = self.seq[j]
        s = self.score(r)
        cand = []
        for i in range(self.T):
            ok = True
            for t in (0, 1):
                lit = s[t][i] - self.b[t][i] > self.eta[t]
                own_silenced = not (self.g[t][i] - self.b[t][i] > self.eta[t])
                ok &= lit or (own_silenced and s[t][i] >= self.g[t][i] - 1e-9 * abs(self.g[t][i]))
            if ok:
                cand.append(i)
        top, best = -1, None
        for i in cand:
            v = (s[0][i] - self.b[0][i]) + (s[1][i] - self.b[1][i])
            if best is None or v > best:
                top, best = i, v
        o = self.uid_pos.get(r["edit_uid"], -1)
        return {"fired": bool(cand), "top_edit": top, "n_cand": len(cand), "own": o, "own_cand": o in cand, "own_top": o >= 0 and top == o}

    def words(self, q):
        import stoplist
        stop = set(stoplist.STOP)
        return {w for w in self.re.findall(r"[^\W\d_]+", q.lower()) if len(w) >= 2 and w not in stop}

    def lookup(self, j):
        r = self.seq[j]
        best, top, n = 0, -1, 0
        if r["image_md5"] is not None:
            w = self.words(r["question"])
            for i, e in enumerate(self.req):
                if e["image_md5"] == r["image_md5"]:
                    c = len(w & self.words(e["question"]))
                    n += c > 0
                    if c > best:
                        best, top = c, i
        o = self.uid_pos.get(r["edit_uid"], -1)
        return {"fired": top >= 0, "top_edit": top, "n_cand": n, "own": o, "own_top": o >= 0 and top == o}


def t_naive():
    """Naive second implementation of the conjunction (and of the lookup rule) on 100 sequences of each development log."""
    import retr
    out, ok = {}, True
    for cache, name in C.DEV_LOGS.items():
        E = retr.edits(name)
        rs = retr.row_sets(E)
        cal = retr.default_draw(E, 200, 0)
        thr = retr.calibrate_log(E, cal, 0.05, seed=0)
        rng = np.random.default_rng([SEED, 9])
        pick = []
        for part, k in (("heldout_disjoint", 25), ("edit", 35), ("hn", 30), ("benign", 10)):
            pick += [int(j) for j in rng.choice(rs[part], k, replace=False)]
        d = retr.retr_decide(E, pick, thr)
        dl = retr.lookup_decide(E, pick)
        N = Naive(cache, name)
        N.calibrate(cal, 0.05, 0)
        bad, badl = [], []
        for i, j in enumerate(pick):
            nv = N.decide(j)
            if any(nv[k] != (bool(d[k][i]) if k in ("fired", "own_cand", "own_top") else int(d[k][i])) for k in ("fired", "top_edit", "n_cand", "own", "own_cand", "own_top")):
                bad.append(j)
            nl = N.lookup(j)
            if any(nl[k] != (bool(dl[k][i]) if k in ("fired", "own_top") else int(dl[k][i])) for k in ("fired", "top_edit", "n_cand", "own", "own_top")):
                badl.append(j)
        eta_same = (N.eta[0] == thr.eta_img and N.eta[1] == thr.eta_q and np.array_equal(np.array(N.b[0]), thr.b_img) and np.array_equal(np.array(N.b[1]), thr.b_q))
        out[name] = {"n_sequences": len(pick), "n_retr_decisions_equal": len(pick) - len(bad), "n_lookup_decisions_equal": len(pick) - len(badl),
                     "retr_differing": bad[:10], "lookup_differing": badl[:10], "baselines_and_eta_equal_bitwise": bool(eta_same),
                     "n_fired_retr": int(d["fired"].sum()), "n_fired_lookup": int(dl["fired"].sum()), "n_retr_candidates_total": int(d["n_cand"].sum())}
        ok &= not bad and not badl and eta_same
    return {"ok": bool(ok), "summary": "; ".join(f"{n}: retr {v['n_retr_decisions_equal']}/{v['n_sequences']}, lookup {v['n_lookup_decisions_equal']}/{v['n_sequences']} "
                                                  f"decisions equal to the naive implementation ({v['n_fired_retr']} / {v['n_fired_lookup']} firing), baselines and eta bitwise equal"
                                                  for n, v in out.items()), "logs": out}


# ----------------------------------------------------------------------------- 10. properties of the decision
def t_properties():
    """Conjunction <= each attribute; text rows silent; tie -> lowest index; eta = +inf; 'steps'; own requests; library conventions."""
    import retr
    C.lib_path()
    from rescope import calibrate as LC
    out, ok = {}, True
    for cache, name in C.DEV_LOGS.items():
        E = retr.edits(name)
        st = retr.store(cache)
        rs = retr.row_sets(E)
        cal = retr.default_draw(E, 200, 0)
        thr = retr.calibrate_log(E, cal, 0.05, seed=0)
        rows = rs["probe"] + rs["edit"] + rs["hn"] + rs["benign"]
        SI, SQ = retr.scores(E, rows)
        d = retr.decide(SI, SQ, thr, retr.own(E, rows), rows, "steps")
        noimg = np.array([st.seq[j]["image_md5"] is None for j in rows])
        o = {"n_rows": len(rows), "fired_implies_some_edit_passes_image": bool((~d["fired"] | (d["n_img_pass"] > 0)).all()),
             "fired_implies_some_edit_passes_question": bool((~d["fired"] | (d["n_q_pass"] > 0)).all()),
             "n_cand_le_min_of_single_attribute_counts": bool((d["n_cand"] <= np.minimum(d["n_img_pass"], d["n_q_pass"])).all()),
             "n_rows_without_image": int(noimg.sum()), "n_rows_without_image_firing": int(d["fired"][noimg].sum()),
             "margin_ge_0_iff_fired": bool(((d["margin"] >= 0) == d["fired"]).all()), "top_edit_minus1_iff_not_fired": bool(((d["top_edit"] < 0) == ~d["fired"]).all()),
             "steps_arrays": bool(d["step_fired"].shape == (len(rows), 11) and (d["step_fired"] == d["fired"][:, None]).all() and (d["step_top"] == d["top_edit"][:, None]).all()),
             "near_all_false": bool(not d["near"].any())}
        # single attribute, literal rule == the library's two_block_rule called directly on the same matrix and split
        SIc, SQc = retr.scores(E, cal)
        lit = retr.calibrate(SIc, SQc, 0.05, seed=0)
        r_q, r_i = LC.two_block_rule(SQc, 0.05, seed=0), LC.two_block_rule(SIc, 0.05, seed=0)
        o["literal_thresholds_equal_library_rule"] = bool(np.array_equal(lit.th_q, r_q["tau"]) and np.array_equal(lit.th_img, r_i["tau"]) and lit.eta_q == r_q["eta"])
        # the same thresholds through the library's scoped-log calibration of a cosine gate on the question embeddings (literal rule)
        import torch
        from rescope import EditLog, calibrate_two_block, feature_gate
        K = torch.from_numpy(np.asarray(st.EQ[E.q], dtype=np.float64))
        S = feature_gate(EditLog(torch.zeros(E.T, 2), K.clone(), K, torch.zeros(E.T, dtype=torch.long), name="q"), "cos")
        S2, rec = calibrate_two_block(S, coords=torch.from_numpy(SQc), alpha=0.05, seed=0, allow_silence=True)
        S3, rec3 = calibrate_two_block(S, coords=torch.from_numpy(SQc), alpha=0.05, seed=0)
        o["library_calibrate_two_block_same_question_thresholds"] = bool(np.array_equal(S2.det_tau.numpy(), lit.th_q) and rec["block1"] == lit.rec["block1"])
        o["library_capped_edits_equal_question_attribute"] = bool(rec3["capped_edits"] == [int(i) for i in np.nonzero(thr.capped_q)[0]])
        # own requests
        o["own_requests_passing"] = f"{thr.rec['n_own_requests_passing']}/{E.T}"
        o["own_requests_released_by_own_edit"] = f"{thr.rec['n_own_requests_released_by_own_edit']}/{E.T}"
        # allow_silence: no caps; eta = +inf with too few calibration queries: nothing but own requests can pass
        gI, gQ = retr.own_request_scores(E)
        few = [j for j in cal if st.seq[j]["image_md5"] is not None][:6] + [j for j in cal if st.seq[j]["image_md5"] is None][:4]
        t_inf = retr.calibrate(*retr.scores(E, few), 0.05, seed=0, guards=(gI, gQ))
        d_inf = retr.decide(SI, SQ, t_inf, retr.own(E, rows))
        own_req = np.array([st.seq[j]["kind"] == "rel" for j in rows])
        o["eta_infinite_with_10_queries"] = bool(np.isposinf(t_inf.eta_img) and np.isposinf(t_inf.eta_q) and t_inf.rec["n_edits_capped"] == E.T)
        o["eta_infinite_fires_only_on_exact_requests"] = bool((d_inf["rank"][d_inf["fired"]] > -np.inf).all() and d_inf["fired"][own_req].all()
                                                              and int(d_inf["fired"].sum()) == int(((SI >= t_inf.th_img) & (SQ >= t_inf.th_q)).any(1).sum()))
        t_sil = retr.calibrate(*retr.scores(E, few), 0.05, seed=0, guards=(gI, gQ), allow_silence=True)
        o["allow_silence_nothing_fires_at_eta_infinite"] = bool(not retr.decide(SI, SQ, t_sil)["fired"].any() and t_sil.rec["n_edits_capped"] == 0)
        # shared rule (not registered for retr; provided): one conformal threshold per attribute, candidate iff score > eta
        t_sh = retr.calibrate(SIc, SQc, 0.05, rule="shared")
        d_sh = retr.decide(SI, SQ, t_sh)
        o["shared_rule_is_score_gt_eta"] = bool(np.array_equal(d_sh["cand"], (SI > t_sh.eta_img) & (SQ > t_sh.eta_q)) and not t_sh.b_img.any() and not t_sh.b_q.any())
        import json as _json
        o["record_is_json"] = bool(_json.loads(_json.dumps(thr.rec))["fingerprint"] == thr.rec["fingerprint"] and len(_json.dumps(retr.lookup_record(E))) > 0)
        o["thresholds_reproducible"] = bool(retr.calibrate_log(E, cal, 0.05, seed=0).rec["fingerprint"] == thr.rec["fingerprint"])
        o["ok"] = all(v is True for k, v in o.items() if isinstance(v, bool)) and o["n_rows_without_image_firing"] == 0
        out[name] = o
        ok &= o["ok"]
    # edge: nearly no calibration query has an image -> eta_img = -inf: the image attribute passes every query WITH an image, text rows stay silent
    rng = np.random.default_rng(SEED)
    Tn, nn = 5, 60
    SQs, SIs = rng.uniform(0, 0.8, (nn, Tn)), rng.uniform(0, 0.8, (nn, Tn))
    SIs[4:] = -np.inf
    te = retr.calibrate(SIs, SQs, 0.05, seed=0, guards=(np.ones(Tn), np.ones(Tn)))
    de = retr.decide(np.array([[0.1] * Tn, [-np.inf] * Tn]), np.array([[0.99] * Tn, [0.99] * Tn]), te)
    edge = {"eta_img": float(te.eta_img), "note": te.rec.get("note_image_scale") is not None, "fired": de["fired"].tolist()}
    edge["ok"] = bool(np.isneginf(te.eta_img) and edge["note"] and edge["fired"] == [True, False])
    ok &= edge["ok"]
    try:                                                                # no image at all in block 1: refused
        retr.calibrate(np.full((nn, Tn), -np.inf), SQs, 0.05, seed=0)
        refused = False
    except ValueError:
        refused = True
    ok &= refused
    # tie -> lowest edit index; ranking by the sum of the margins (synthetic)
    thr = retr.Thresholds(T=3, th_img=np.array([0.5, 0.5, 0.5]), th_q=np.array([0.5, 0.5, 0.5]), b_img=np.zeros(3), b_q=np.array([0.0, 0.0, 0.2]), eta_img=0.5, eta_q=0.5,
                          tau_img=np.full(3, 0.5), tau_q=np.full(3, 0.5), capped_img=np.zeros(3, bool), capped_q=np.zeros(3, bool))
    SI = np.array([[0.9, 0.9, 0.9], [0.6, 0.9, 0.4], [-np.inf] * 3, [0.9, 0.7, 0.95]])
    SQ = np.array([[0.8, 0.8, 0.9], [0.9, 0.7, 0.99], [0.9, 0.9, 0.9], [0.4, 0.8, 0.99]])
    d = retr.decide(SI, SQ, thr, own=np.array([1, 1, 0, 2]))
    syn = {"top_edit": d["top_edit"].tolist(), "n_cand": d["n_cand"].tolist(), "own_cand": d["own_cand"].tolist(), "own_top": d["own_top"].tolist()}
    syn["ok"] = syn["top_edit"] == [0, 1, -1, 2] and syn["n_cand"] == [3, 2, 0, 2] and syn["own_cand"] == [True, True, False, True] and syn["own_top"] == [False, True, False, True]
    ok &= syn["ok"]
    return {"ok": bool(ok), "summary": "conjunction never releases where one attribute alone has no candidate; rows without an image never fire; tie -> lowest index; "
                                       "eta = +inf releases own requests only; literal thresholds == library rule (also through calibrate_two_block of a cosine gate): "
                                       + ("all hold" if ok else "FAILED"), "logs": out, "synthetic": syn, "edge_no_images": edge, "no_image_in_block1_refused": refused}


# ----------------------------------------------------------------------------- 11. the library gate of a decision
def t_forced_gate():
    """forced_gate: the library's ScopedLog releases exactly the given edit at the last prompt position (key cache + Hook, CPU)."""
    import torch
    import retr
    C.lib_path()
    from rescope import Hook, KeyCache, feature_gate, from_testbed
    out, ok = {}, True
    for cache, name in C.DEV_LOGS.items():
        log = from_testbed(name)
        E = retr.edits(log)                                             # a rescope EditLog as the "log"
        same_uids = E.uids == retr.edits(name).uids
        kc = KeyCache(cache)
        S = feature_gate(log, "ncoord")
        rs = retr.row_sets(E)
        rng = np.random.default_rng([SEED, 11])
        rows = [int(j) for j in rng.choice(rs["probe"] + rs["edit"] + rs["hn"], 60, replace=False)]
        good = 0
        for j, e in zip(rows, rng.integers(0, E.T, len(rows))):
            d = kc.decide(retr.forced_gate(S, int(e)), "tf", rows=[j], uids=log.meta["uids"])
            good += int(bool(d["fired"][0]) and int(d["top_edit"][0]) == int(e) and int(d["n_cand"][0]) == 1)
        # through the library's Hook on a linear module of the right shape: output = W0 k + V_e at the last prompt position only
        m, dd = S.values.shape[1], S.d
        lin = torch.nn.Linear(dd, m, bias=False).to(torch.float32)
        W0 = lin.weight.data.clone()
        x = kc.keys("tf", rows[:7])[None].to(torch.float32)             # (1, 7, d): a 'sequence' of 7 positions
        e, lp = 5, 3
        g = retr.forced_gate(S, e).replace(W0=W0)
        with torch.no_grad(), Hook(lin, g) as hk:
            hk.begin(torch.tensor([lp]))
            y = lin(x)
            flags = hk.summary(own=e)
        base = torch.nn.functional.linear(x, W0)
        want = base.clone()
        want[0, lp] = (base[0, lp].double() + S.values[e]).to(base.dtype)
        hook_ok = bool(torch.equal(y, want) and flags["fired"] and flags["fired_last"] and flags["top_last"] == e and flags["n_fire_pos"] == 1 and flags["own_released_last"])
        out[name] = {"uids_from_edit_batch_equal_edit_factors": bool(same_uids), "keycache_decide_forced": f"{good}/{len(rows)}", "hook_releases_value_at_lp_only": hook_ok}
        ok &= same_uids and good == len(rows) and hook_ok
    return {"ok": bool(ok), "summary": "; ".join(f"{n}: forced gate releases the given edit on {v['keycache_decide_forced']} cached sequences; Hook output = W0 k + V_e at the "
                                                  f"last prompt position only: {v['hook_releases_value_at_lp_only']}" for n, v in out.items()), "logs": out}


# ----------------------------------------------------------------------------- 11b. the gate object (library interface) on the CPU
def t_request_gate():
    """RequestGate through the library's Hook on a linear module (CPU): Firing, flags, output; forced gates agree; decode bookkeeping."""
    import torch
    import retr
    C.lib_path()
    from rescope import Hook, KeyCache, feature_gate, from_testbed
    out, ok = {}, True
    for cache, name in C.DEV_LOGS.items():
        log = from_testbed(name)
        E = retr.edits(log)
        kc = KeyCache(cache)
        rs = retr.row_sets(E)
        thr = retr.calibrate_log(E, retr.default_draw(E, 200, 0), 0.05, seed=0)
        rng = np.random.default_rng([SEED, 12])
        rows = [int(j) for j in rng.choice(rs["edit"], 40, replace=False)] + [int(j) for j in rng.choice(rs["heldout_disjoint"], 10, replace=False)] \
            + [int(j) for j in rng.choice(rs["hn"], 20, replace=False)]
        S = feature_gate(log, "ncoord")
        m, d = S.values.shape[1], S.d
        lin = torch.nn.Linear(d, m, bias=False).to(torch.float32)
        W0 = lin.weight.data.clone()
        S = S.replace(W0=W0)
        idx = retr.cache_index(cache)
        res = {}
        for meth in ("retr", "lookup"):
            dec = retr.retr_decide(E, rows, thr) if meth == "retr" else retr.lookup_decide(E, rows)
            g = retr.request_gate(S, E, dec)
            gs = retr.gates(S, dec)
            n_ok = n_fire = n_dec = 0
            for i, j in enumerate(rows):
                x = torch.cat([kc.keys("decode", [rows[(i + 1) % len(rows)], rows[(i + 2) % len(rows)]]), kc.keys("tf", [j]),
                               kc.keys("decode", [rows[(i + 3) % len(rows)]])])[None].to(torch.float32)          # 4 positions, the last prompt position is 2
                lp, o = 2, int(dec["own"][i])
                base = torch.nn.functional.linear(x, W0)
                want = base.clone()
                if dec["fired"][i]:
                    want[0, lp] = (base[0, lp].double() + S.values[int(dec["top_edit"][i])]).to(base.dtype)
                g.select(idx[j]["lib_sid"])
                with torch.no_grad(), Hook(lin, g) as hk:
                    hk.begin(torch.tensor([lp]))
                    y = lin(x)
                    fl = hk.summary(own=o if o >= 0 else None)
                good = torch.equal(y, want) and fl["fired"] == bool(dec["fired"][i]) and fl["top_last"] == int(dec["top_edit"][i]) \
                    and (not dec["fired"][i] or (fl["n_cand_edits"] == int(dec["n_cand"][i]) and fl["n_fire_pos"] == 1 and fl["fired_last"]))
                if o >= 0:
                    good = good and fl["own_fired_last"] == bool(dec["own_cand"][i]) and fl["own_released_last"] == bool(dec["own_top"][i])
                if not dec["fired"][i]:
                    good = good and y is not None and torch.equal(y, base) and gs[i] is None
                else:                                                   # the forced gate of the same decision gives the same output
                    with torch.no_grad(), Hook(lin, gs[i]) as hk2:
                        hk2.begin(torch.tensor([lp]))
                        y2 = lin(x)
                    good = good and torch.equal(y2, y)
                    n_fire += 1
                    if n_dec < 5:                                       # decode bookkeeping: a full forward, then a longer one, then a one-token step
                        with torch.no_grad(), Hook(lin, g) as hk3:
                            hk3.begin(torch.tensor([lp]), gen=True)
                            y_a, y_b, y_c = lin(x), lin(torch.cat([x, x[:, :1]], 1)), lin(x[:, :1])
                            f3 = hk3.summary()
                        good = good and torch.equal(y_a, want) and torch.equal(y_b[:, :4], want) and torch.equal(y_c, torch.nn.functional.linear(x[:, :1], W0)) \
                            and f3["n_calls_last"] == 2 and f3["n_fire_calls_last"] == 2 and f3["n_fire_calls"] == 2
                        n_dec += 1
                n_ok += int(bool(good))
            try:
                g.select("no-such-sequence")
                raised = False
            except KeyError:
                raised = True
            res[meth] = {"n_sequences": len(rows), "n_firing": n_fire, "n_hook_output_and_flags_equal_decision": n_ok, "n_decode_checks": n_dec,
                         "unknown_sequence_raises": raised, "release_plan_edits": len(retr.release_plan(dec))}
            ok &= n_ok == len(rows) and raised and n_fire > 0
        out[name] = res
    return {"ok": bool(ok), "summary": "; ".join(f"{n}: " + ", ".join(f"{m} {v['n_hook_output_and_flags_equal_decision']}/{v['n_sequences']} ({v['n_firing']} firing)" for m, v in o.items())
                                                  for n, o in out.items()) + " sequences: Hook output and flags == decision, forced gate == RequestGate, decode bookkeeping", "logs": out}


# ----------------------------------------------------------------------------- 13. behaviour smoke test on the GPU (development log)
def _behaviour_smoke_one():
    """The gate objects through the REAL evaluation (rescope.load + rescope.Evaluator, B = 1) on a development log: a mechanics check.
    The log: RETR_SMOKE_LOG (default: the Qwen development log)."""
    import torch
    import retr
    C.lib_path()
    name = os.environ.get("RETR_SMOKE_LOG", C.DEV_LOGS["qwen25vl"])
    cache = [c for c, n in C.DEV_LOGS.items() if n == name][0]
    hc = os.path.join(C.OUT, "hcache")                                  # the library's private harness cache, re-pointed to this layer's output
    os.environ["RESCOPE_CACHE"] = hc
    import rescope
    from rescope import evaluate as RE
    RE.CACHE = hc
    mem = (int(os.environ["GQ_MEM"]) - 600) / 1024.0 if "GQ_MEM" in os.environ else None
    H = rescope.load(name, device="cuda:0", mem_gb=mem)
    ev = rescope.Evaluator(H, hn_dir=rescope.HN_DIR_FINAL)
    log = rescope.from_harness(H)
    E = retr.edits(log)
    S = rescope.feature_gate(log.to(H.dev), "ncoord", W0=H.W0)          # on the model's device: the released values are the library's rebased values
    thr = retr.calibrate_log(E, retr.default_draw(E, 200, 0), 0.05, seed=0)
    rs = retr.row_sets(E)
    idx = retr.cache_index(cache)
    T = E.T
    eds = list(range(8))
    rows = [rs["edit"][k * T + i] for k in range(3) for i in eds] + rs["heldout_disjoint"][:6] + rs["hn"][:6]
    items = {it["sid"]: it for it in ev.edit_items() + ev.probe_items("all") + ev.hn_items()}
    its = [items[idx[j]["lib_sid"]] for j in rows]
    out, ok = {"log": name, "n_sequences": len(rows), "load": H.load_info}, True
    for meth in ("retr", "lookup"):
        dec = retr.retr_decide(E, rows, thr) if meth == "retr" else retr.lookup_decide(E, rows)
        g = retr.request_gate(S, E, dec)
        r_gate = retr.gate_rows(ev, g, its)
        gs = retr.gates(S, dec)
        n_flags = n_forced = n_silent = n_fired = 0
        rel_exact = []
        for i, (j, it, r) in enumerate(zip(rows, its, r_gate)):
            f = bool(dec["fired"][i])
            good = r["fired"] == f and r["top_last"] == int(dec["top_edit"][i]) and (not f or r["n_cand_edits"] == int(dec["n_cand"][i]))
            if it["kind"] != "probe":
                good = good and r["own_fired_last"] == bool(dec["own_cand"][i]) and r["own_released_last"] == bool(dec["own_top"][i])
            n_flags += int(bool(good))
            keys = ("exact", "token_acc", "n_tok", "finite", "first_mismatch")
            if f:
                n_fired += 1
                r2 = ev.tf_rows(gs[i], [it])[0]                         # the forced gate of the same decision: the same outcome
                n_forced += int(all(r2[k] == r[k] for k in keys) and r2["fired"] and r2["top_last"] == int(dec["top_edit"][i]))
                if idx[j]["kind"] == "rel":
                    rel_exact.append(bool(r["exact"]))
            else:                                                       # nothing released: the pre-edit model = the cached pre-edit outcome
                e = idx[j]
                r0 = ev.tf_rows(H.W0, [it])[0]
                n_silent += int(all(r0[k] == r[k] for k in keys) and bool(e["pre_exact"]) == r["exact"] and abs(float(e["pre_token_acc"]) - r["token_acc"]) < 1e-12)
        fi = [i for i in range(len(rows)) if dec["fired"][i] and idx[rows[i]]["kind"] == "rel"][:2]
        gen = retr.gate_rows(ev, {"g": g}, [its[i] for i in fi], fn="generate")["g"]          # also the dict form
        dec_ok = all(r["fired"] and r["fired_last"] and r["n_fire_calls_last"] == r["n_calls_last"] and r.get("pos_ok", True) and r["top_last"] == int(dec["top_edit"][i])
                     for i, r in zip(fi, gen))
        o = {"n_firing": n_fired, "n_flags_equal_decision": n_flags, "n_forced_gate_rows_equal": n_forced, "n_silent_rows_equal_pre_edit_model_and_cache": n_silent,
             "n_silent": len(rows) - n_fired, "decodes": len(gen), "decode_fires_in_every_full_forward": bool(dec_ok),
             "rel_exact_when_own_request_released": f"{sum(rel_exact)}/{len(rel_exact)}", "generated": [r["y"] for r in gen], "targets": [idx[rows[i]]["answer"] for i in fi]}
        o["ok"] = bool(n_flags == len(rows) and n_forced == n_fired and n_silent == len(rows) - n_fired and dec_ok and n_fired > 0)
        out[meth] = o
        ok &= o["ok"]
    out["gpu"] = {"name": torch.cuda.get_device_name(0), "max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2 ** 20}
    import shutil as _sh
    _sh.rmtree(hc, ignore_errors=True)                                  # copies of the harness' baseline files and symbolic links only
    out["ok"] = bool(ok)
    out["summary"] = f"{name}, {len(rows)} sequences through rescope.Evaluator: " + "; ".join(
        f"{m}: flags == decision {out[m]['n_flags_equal_decision']}/{len(rows)}, forced gate == RequestGate {out[m]['n_forced_gate_rows_equal']}/{out[m]['n_firing']}, "
        f"silent == pre-edit {out[m]['n_silent_rows_equal_pre_edit_model_and_cache']}/{out[m]['n_silent']}, decode fires in every full forward {out[m]['decode_fires_in_every_full_forward']}, "
        f"Rel exact {out[m]['rel_exact_when_own_request_released']}" for m in ("retr", "lookup"))
    return out


def t_behaviour_smoke():
    """One development log per call (RETR_SMOKE_LOG); the results of both logs are kept in out/tests/behaviour_smoke.json."""
    p = os.path.join(TDIR, "behaviour_smoke.json")
    logs = C.jload(p).get("logs", {}) if os.path.exists(p) else {}
    r = _behaviour_smoke_one()
    logs[r["log"]] = r
    return {"ok": all(v["ok"] for v in logs.values()), "logs": logs, "not_run": [n for n in C.DEV_LOGS.values() if n not in logs],
            "summary": " | ".join(v["summary"] for v in logs.values()) + "".join(f" | {n}: NOT RUN" for n in C.DEV_LOGS.values() if n not in logs)}


# ----------------------------------------------------------------------------- 14. the wiring through the driver's memo path (GPU)
def t_behaviour_fx():
    """The wiring recommended in USAGE.txt, end to end on a development log (GPU; a mechanics check): ONE forced gate per released
    edit -> fx.methods.wrap -> fx.behave Session.tf (the driver's memo path), compared with the library route (RequestGate through
    rescope.Evaluator) and with the cached pre-edit outcome on silent sequences.  The driver's memo and the library's private harness
    cache are re-pointed to a temporary directory of this layer: the real memo is never read or written."""
    import shutil as _sh
    import torch
    import retr
    name = os.environ.get("RETR_SMOKE_LOG", C.DEV_LOGS["qwen25vl"])
    tmp = os.path.join(C.OUT, "fx_tmp")
    _sh.rmtree(tmp, ignore_errors=True)
    os.environ["FX_MEMO_ROOT"], os.environ["RESCOPE_CACHE"] = os.path.join(tmp, "memo"), os.path.join(tmp, "hcache")
    try:
        C.fx_path()
        import fx
    except Exception as e:                                              # noqa: BLE001
        return {"ok": True, "skipped": True, "summary": f"fx could not be imported ({type(e).__name__}: {e}); not checked"}
    from rescope import evaluate as RE
    old_roots = (fx.MEMO_ROOT, RE.CACHE)
    fx.MEMO_ROOT, RE.CACHE = os.path.join(tmp, "memo"), os.path.join(tmp, "hcache")
    try:
        L = fx.logs.open(name)
        H = fx.logs.load_model(name)
        ses = fx.behave.session(L, H)
        if not ses.memo.root.startswith(tmp):
            raise RuntimeError(f"the driver's memo is not the temporary one ({ses.memo.root})")
        dr = fx.draws.calibration(L, 200, 0)
        refs = {"retr": retr.reference("retr", L, dr.rows, 0.05, seed=dr.seed), "lookup": retr.reference("lookup", L)}
        S0 = fx.methods.base_gate(L)[0]                                 # the driver's un-thresholded gate: its values are the released values
        Sg = S0.replace(guard_key=S0.guard_key[:0], guard_edit=S0.guard_edit[:0], W0=H.W0).to(torch.device(H.dev))
        rows = [L.edit_rows[m][i] for m in ("rel", "tgen", "igen") for i in range(8)] + L.pool("heldout_disjoint")[:6] + fx.gate.row_sets(L)["hn"][:6]
        keys = ("exact", "token_acc", "n_tok", "finite", "first_mismatch")
        out, ok = {"log": name, "n_sequences": len(rows), "fx_version": getattr(fx, "VERSION", None)}, True
        for meth, R in refs.items():
            dec = R.decide(rows)
            plan = retr.release_plan(dec)
            first, src1, src2, same2 = {}, [], [], True
            for pass_ in (1, 2):
                for e, pos in plan.items():
                    M = fx.methods.wrap(L, retr.forced_gate(S0, e), f"{meth}@{e}", rec={"reference": meth, "released_edit": int(e), "fingerprint_of_reference": R.fingerprint})
                    for i, r in zip(pos, ses.tf(M, [rows[i] for i in pos])):
                        if pass_ == 1:
                            first[i] = r
                            src1.append(r["src"])
                        else:
                            src2.append(r["src"])
                            same2 &= all(r[k] == first[i][k] for k in keys)
            fired = sorted(first)
            g = R.gate(Sg, dec)
            lib = retr.gate_rows(ses.ev_plain, g, [ses.item(rows[i]) for i in fired])
            n_eq = sum(all(a[k] == first[i][k] for k in keys) and first[i]["fired"] and first[i]["top_last"] == int(dec["top_edit"][i]) == a["top_last"] for a, i in zip(lib, fired))
            sil = [i for i in range(len(rows)) if not dec["fired"][i]][:6]
            lib0 = retr.gate_rows(ses.ev_plain, g, [ses.item(rows[i]) for i in sil])
            n_sil = sum(all(ses._pre_outcome(rows[i])[k] == a[k] for k in keys) and a["fired"] is False for a, i in zip(lib0, sil))
            # a RequestGate built on the CPU: the Evaluator moves it to the model's device (RequestGate.to; the selection is shared)
            g_cpu = R.gate(S0.replace(guard_key=S0.guard_key[:0], guard_edit=S0.guard_edit[:0], W0=H.W0), dec)
            lib_cpu = retr.gate_rows(ses.ev_plain, g_cpu, [ses.item(rows[i]) for i in fired[:3]])
            n_cpu = sum(all(a[k] == first[i][k] for k in keys) and a["fired"] and a["top_last"] == int(dec["top_edit"][i]) for a, i in zip(lib_cpu, fired[:3]))
            # decodes: the forced gate through the driver's generation (memo path) == the RequestGate through the library's generation
            gen_i = [i for i in fired if L.index[rows[i]]["kind"] in ("rel", "tgen", "igen")][:2]
            n_gen = 0
            for i in gen_i:
                e = int(dec["top_edit"][i])
                M = fx.methods.wrap(L, retr.forced_gate(S0, e), f"{meth}@{e}", rec={"reference": meth, "released_edit": e})
                a = ses.generate(M, [rows[i]])[0]
                b = retr.gate_rows(ses.ev_plain, g, [ses.item(rows[i])], fn="generate")[0]
                n_gen += int(a["y"] == b["y"] and a["long"] == b["long"] and bool(a["fired"]) and bool(b["fired"]) and int(a["top_last"]) == e == int(b["top_last"])
                             and int(b["n_fire_calls_last"]) == int(b["n_calls_last"]))
            o = {"n_firing": len(fired), "n_forced_gate_methods": len(plan), "first_pass_sources": {k: src1.count(k) for k in sorted(set(src1))},
                 "second_pass_sources": {k: src2.count(k) for k in sorted(set(src2))}, "second_pass_equal": bool(same2),
                 "n_driver_rows_equal_library_route": n_eq, "n_silent_checked": len(sil), "n_silent_cached_pre_edit_outcome_equal_library_route": n_sil,
                 "n_decodes": len(gen_i), "n_decodes_driver_equal_library_route": n_gen, "n_cpu_built_request_gate_rows_equal": f"{n_cpu}/{len(fired[:3])}"}
            o["ok"] = bool(n_eq == len(fired) and len(fired) > 0 and same2 and set(src2) == {"memo"} and n_sil == len(sil) and n_gen == len(gen_i) and len(gen_i) > 0
                           and n_cpu == len(fired[:3]))
            out[meth] = o
            ok &= o["ok"]
        out["hook_mismatch"] = int(ses.stats["hook_mismatch"])
        out["memo_conflicts"] = int(ses.memo.stats.get("conflicts", 0)) if hasattr(ses.memo, "stats") else None
        ok &= out["hook_mismatch"] == 0 and not out["memo_conflicts"]
        out["ok"] = bool(ok)
        out["summary"] = f"{name}: " + "; ".join(
            f"{m}: {out[m]['n_driver_rows_equal_library_route']}/{out[m]['n_firing']} firing rows of fx.behave (forced gates, memo path; first pass {out[m]['first_pass_sources']}, "
            f"second pass {out[m]['second_pass_sources']}) == library route, silent {out[m]['n_silent_cached_pre_edit_outcome_equal_library_route']}/{out[m]['n_silent_checked']} == cached pre-edit outcome, "
            f"decodes {out[m]['n_decodes_driver_equal_library_route']}/{out[m]['n_decodes']} equal"
            for m in refs) + f"; hook mismatches {out['hook_mismatch']}"
        return out
    finally:
        fx.MEMO_ROOT, RE.CACHE = old_roots
        _sh.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- 12. the driver accepts the decisions
def t_fx_compat():
    """Row sets, calibration draw, own edits and the decision dicts against the shared driver fx (read-only import)."""
    import retr
    try:
        C.fx_path()
        import fx
    except Exception as e:                                              # noqa: BLE001  (the driver is under review: its absence is reported, not a failure of this layer)
        return {"ok": True, "skipped": True, "summary": f"fx could not be imported ({type(e).__name__}: {e}); not checked"}
    out, ok = {}, True
    for cache, name in C.DEV_LOGS.items():
        L = fx.logs.open(name)
        E = retr.edits(L)                                               # a fx LogData as the "log"
        rs, fs = retr.row_sets(E), fx.gate.row_sets(L)
        dr = fx.draws.calibration(L, 200, 0)
        thr = retr.calibrate_log(L, dr.rows, 0.05, seed=dr.seed)
        o = {"row_sets_equal": all(rs[k] == [int(j) for j in fs[k]] for k in ("probe", "edit", "hn", "benign")),
             "heldout_disjoint_equal": rs["heldout_disjoint"] == [int(j) for j in L.pool("heldout_disjoint")],
             "default_draw_equals_fx_draw": all(retr.default_draw(E, 200, k) == [int(j) for j in fx.draws.calibration(L, 200, k).rows] for k in (0, 1, 2, 19)),
             "own_equals_fx_own": all(int(a) == int(L.own(j)) for part in ("edit", "hn", "probe") for j, a in zip(rs[part], retr.own(E, rs[part])))}
        st = {}
        for meth, th in (("retr", thr), ("lookup", None)):
            ev = retr.evaluate(L, th, positions=("tf", "steps"))        # the form of fx.gate.evaluate, from this layer's own row sets
            g = fx.metrics.gate_stats(L, ev)
            o[f"{meth}_steps_equal_tf"] = bool(all(np.array_equal(ev[p]["steps"]["fired"], ev[p]["tf"]["fired"]) for p in ("probe", "edit", "hn")))
            st[meth] = {"heldout_disjoint_fired": g["probe"]["heldout_disjoint"]["fired"], "tgen_own_top": g["edit"]["tgen"]["own_top"], "igen_own_top": g["edit"]["igen"]["own_top"],
                        "rel_own_top": g["edit"]["rel"]["own_top"], "hn": {t: g["hn"][t]["fired"] for t in ("HN-I", "HN-Q", "HN-X")}, "benign_fired": g["benign"]["fired"]}
            ref = fx.gate.decide(L, fx.methods.cosine(L, "default"), fs["probe"][:5], "tf")
            o[f"{meth}_decision_has_every_key_of_fx_gate_decide"] = set(ref) <= set(ev["probe"]["tf"])
        # the forced gate of a decision through the driver: fx.methods.wrap -> a memo-path Method that releases exactly that edit
        S0 = fx.methods.base_gate(L)[0]
        dec = retr.retr_decide(L, fs["edit"], thr)
        plan = retr.release_plan(dec)
        e0 = sorted(plan)[0]
        Mf = fx.methods.wrap(L, retr.forced_gate(S0, e0), f"retr@{e0}", rec={"reference": "retr", "released_edit": e0, "cal": {k: v for k, v in thr.rec.items() if k not in ("block1", "block2")}})
        rows_e = [fs["edit"][i] for i in plan[e0]]
        df = fx.gate.decide(L, Mf, rows_e, "tf")
        o["forced_gate_is_a_memo_method_of_the_driver"] = bool(Mf.memo and Mf.kind == "scoped")
        o["forced_gate_through_fx_gate_decide"] = bool(df["fired"].all() and (df["top_edit"] == e0).all() and not df["near"].any() and (df["n_cand"] == 1).all())
        o["forced_gate_releases_the_drivers_values"] = bool(retr.forced_gate(S0, e0).values is S0.values)
        R, Rl = retr.reference("retr", L, dr.rows, 0.05, seed=dr.seed), retr.reference("lookup", L)       # the one-handle form
        o["reference_handle"] = bool(R.fingerprint == thr.rec["fingerprint"] and np.array_equal(R.decide(fs["edit"])["top_edit"], dec["top_edit"])
                                     and np.array_equal(Rl.decide(fs["hn"])["fired"], retr.lookup_decide(L, fs["hn"])["fired"]) and len(Rl.fingerprint) == 12
                                     and isinstance(R.gate(S0, dec), retr.RequestGate))
        mech = os.path.join(C.OUT, "mechanics.json")
        if os.path.exists(mech):                                        # the mechanics numbers were computed without fx: they must agree
            mm = C.jload(mech)["logs"][name]
            for meth in ("retr", "lookup"):
                a, b = mm[meth], st[meth]
                o[f"{meth}_gate_stats_equal_mechanics"] = bool(
                    abs(a["heldout_disjoint"]["fired_pct"] - b["heldout_disjoint_fired"]) < 1e-9 and abs(a["tgen"]["coverage_pct"] - b["tgen_own_top"]) < 1e-9
                    and abs(a["igen"]["coverage_pct"] - b["igen_own_top"]) < 1e-9 and all(abs(a[t]["fired_pct"] - b["hn"][t]) < 1e-9 for t in ("HN-I", "HN-Q", "HN-X")))
        o["gate_stats"] = st
        o["ok"] = all(v for k, v in o.items() if isinstance(v, bool))
        out[name] = o
        ok &= o["ok"]
    return {"ok": bool(ok), "skipped": False, "fx_version": getattr(fx, "VERSION", None),
            "summary": "row sets, draw 0, own edits equal the driver's; fx.metrics.gate_stats accepts the retr / lookup decisions and gives the mechanics numbers: "
                       + ("all hold" if ok else "FAILED"), "logs": out}


TESTS = {"repro": t_repro, "attributes": t_attributes, "facts": t_facts, "unit_norm": t_unit_norm, "reembed_questions": t_reembed_questions, "reembed_questions_all": t_reembed_questions_all,
         "reembed_images_gpu": t_reembed_images_gpu,
         "reembed_images_cpu": t_reembed_images_cpu, "exact_scores": t_exact_scores, "own_request": t_own_request, "rephrase_scores": t_rephrase_scores,
         "naive": t_naive, "properties": t_properties, "forced_gate": t_forced_gate, "request_gate": t_request_gate, "fx_compat": t_fx_compat,
         "behaviour_smoke": t_behaviour_smoke, "behaviour_fx": t_behaviour_fx}
GPU = ("reembed_images_gpu", "behaviour_smoke", "behaviour_fx")
CPU = [k for k in TESTS if k not in GPU]
LABEL = {"repro": "re-runs reproduce byte for byte (attribute tables, key lists, mechanics, README)", "attributes": "attributes derived from the loaders == cache; prompts rendered",
         "facts": "repeated questions / shared images (all streams; descriptive)",
         "unit_norm": "embeddings: unit norm, float32, 2^-24 grid", "reembed_questions": "50 questions re-embedded (<= 1e-6)", "reembed_questions_all": "all questions re-embedded: matrix bit for bit",
         "reembed_images_gpu": "50 images re-embedded, GPU (<= 1e-6)", "reembed_images_cpu": "50 images re-embedded, CPU (other device; <= 1e-6)",
         "exact_scores": "scores exact and batch-independent", "own_request": "own request scores 1 on both attributes (all edits)",
         "rephrase_scores": "rephrases: distribution of the two scores (dev logs; descriptive)", "naive": "naive second implementation, 100 sequences per dev log",
         "properties": "properties of calibration and decision", "forced_gate": "forced_gate through the library's key cache and Hook",
         "request_gate": "RequestGate (library gate interface) through the library's Hook, CPU",
         "fx_compat": "decisions accepted by the driver (fx.metrics.gate_stats)",
         "behaviour_smoke": "gate objects through rescope.Evaluator on the model (GPU, both dev logs; mechanics)",
         "behaviour_fx": "forced gates through the driver's memo path == library route (GPU, Qwen dev log; mechanics)"}


def run(names):
    os.makedirs(TDIR, exist_ok=True)
    for k in names:
        t0 = time.time()
        try:
            r = TESTS[k]()
        except Exception as e:                                          # noqa: BLE001
            import traceback
            r = {"ok": False, "summary": f"ERROR {type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        r.update(test=k, seconds=round(time.time() - t0, 1), when=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC")
        C.jdump(r, os.path.join(TDIR, k + ".json"))
        print(f"[{'ok' if r['ok'] else 'FAIL'}] {k} ({r['seconds']} s): {r['summary']}", flush=True)


def table():
    rows, n_ok, n, bad = [], 0, 0, []
    for k in TESTS:
        p = os.path.join(TDIR, k + ".json")
        if not os.path.exists(p):
            rows.append([k, "NOT RUN", "", LABEL[k]])
            continue
        r = C.jload(p)
        n += 1
        n_ok += bool(r["ok"])
        rows.append([k, "skipped" if r.get("skipped") else ("pass" if r["ok"] else r.get("status", "FAIL")), f"{r['seconds']} s", LABEL[k]])
        if not r["ok"]:
            bad.append(k)
    txt = ["TESTS of feat2/retr (python tests.py; details: out/tests/<test>.json)", "", C.table(rows, ["test", "result", "time", "what"]), "",
           f"{n_ok} of {n} tests that ran pass." + (f"  NOT passing: {', '.join(bad)}." if bad else ""), ""]
    for k in TESTS:
        p = os.path.join(TDIR, k + ".json")
        if os.path.exists(p):
            txt.append(f"{k}: {C.jload(p)['summary']}")
    C.text_write("\n".join(txt), os.path.join(C.OUT, "test_table.txt"))
    print("\n".join(txt))


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["table"]:
        table()
    else:
        run(args or CPU)
        table()
