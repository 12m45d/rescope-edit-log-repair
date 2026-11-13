"""Selection of the log-derived recombinations of one backbone (amendment R5 of 10_final/PROTOCOL.md): CPU, seeded, deterministic.

    python rc_select.py <qwen25vl | llava>
    RC_SELECT_DIR=<dir> python rc_select.py <backbone>      the same selection into another directory (rc_check.py compares
                                                            it byte for byte with the selection in use)

Writes cache2/recomb/<backbone>/rows.jsonl and select_summary.json.  No model is loaded.  Every draw is seeded
(rc_common.SEED = 20261111, sha1 of kind / family / edit uid / scope); neither file holds a time stamp or a path that changes.

For EVERY edit i of the backbone's long VLKEB stream (qwen25vl: 2000 edits of TB-Q2000-ae-l2_3, llava: 1000 edits of
TB-L1000-ae-l2_3) m = 4 recombinations of each kind, modelled on the hard negatives (10_final/cache/code/fc_select.py):

  X-I   the image of edit i + the image-locality question (m_loc_q) of a partner item of the SOURCE POOL of the hard negatives
        (seeded permutation of the pool per edit; first 4 admissible candidates)
  X-Q   the question (src) of edit i + the entity image (`image`) of a partner item of the source pool (same form)
  X-X   the image of edit i + the question (src) of another edit j of the same stream:
          stream position i < 100:   2 partners from i's batch of 10 (positions 10 * (i // 10) .. + 9), then 2 from i's group of 100
          stream position i >= 100:  4 partners from i's group of 100 (positions 100 * (i // 100) .. + 99)
        (seeded permutation of the candidate positions per edit and scope).  The partner's position is stored (edit_b_pos,
        min_T = max(i, j) + 1): a log with T edits uses a row iff min_T <= T.  Every registered log length that is a multiple of
        100 contains both edits of every row of its edits; the prefix logs of 10 / 20 / 50 edits contain the batch partners and
        drop the group partners beyond their end by position.  If a batch cannot supply 2 admissible partners the group supplies
        the rest (counted in the summary: x_x.batch_shortfall).

SOURCE POOL = the pool of fc_select.py, recomputed from the list of edit streams recorded in the caches' select summary
(VLKEB eval items that are not probe items and share no (image, src, alt), no Freebase MID and no entity name with ANY edit of
the project's VLKEB streams); the script stops unless size, exclusion counts and the sources of the existing new hard negatives
agree with the record.

A candidate is DISCARDED AND REDRAWN (the next candidate of the permutation is taken) if
  (a) own_question   its question string (stripped, lower case) is the question or the rephrase of edit i     [X-I, X-X]
  (b) test_query     its (image content = md5 of the file, question string stripped lower case) is the query of ANY row of ANY key
                     cache under 10_final/cache (index.jsonl of qwen25vl, llava, QO, blip2: edit prompts and rephrases (rel / tgen /
                     igen) of every cached family, probe rows, hard negatives of all kinds, benign rows; the caches of the other
                     backbones are included although their rows are never test rows of this backbone's logs)
  (c) related        X-X: the two edits share an entity (Freebase MID of image / image_rephrase), a name or an answer
                     (normalised pred_text / alt); X-Q: the partner item shares one with the edit (cannot happen: pool rule)
  (d) repeat         its (image content, question string) repeats an earlier recombination of the same edit (any kind)
checked in this order (X-Q: 'related' first; it never occurs); a row records the number of candidates discarded before it was
accepted (n_redraws) and why (redraws).

ROW FIELDS
  rid ('X-I:<edit uid>:<k>' ...), kind, family, edit_uid, edit_pos (= edit: stream position of the edit that OWNS the row), k,
  image, prompt, fmt ('vqa'), ptype ('image'), image_md5, src_uid, src_field (m_loc_q | image | edit_src),
  edit_b_uid, edit_b_pos (= edit_b; X-X: the edit whose question is used), scope (X-X: batch | group), min_T,
  q_cats, edit_cats, same_category (as the hard negatives: X-I / X-X the question shares a category of the lexicon with the edit's
  question or rephrase; X-Q the partner item's own question shares one with the edit's question),
  q_in_edit / q_in_edit_first_pos (the question string is the question or rephrase of an edit of the stream; first such stream
  position: a row is "cross-like" for a log with T edits iff that position < T), q_in_probe, img_in_probe, query_in_probe,
  edit_src, edit_rephrase, edit_alt, edit_pred; X-Q: src_item_question, src_item_pred; X-X: edit_b_src, edit_b_alt, edit_b_pred,
  n_redraws, redraws, query_id (rows of this file with the same (image content, prompt)), n_same_query.
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as R  # noqa: E402
from rc_common import C  # noqa: E402
from fc_cats import cats  # noqa: E402

import numpy as np  # noqa: E402

BB = sys.argv[1]
CFG = C.BACKBONES[BB]
VF = CFG["vlkeb"]                                  # the VLKEB stream family of the backbone (Q | L)
OUT = os.environ.get("RC_SELECT_DIR") or R.bb_dir(BB)
os.makedirs(OUT, exist_ok=True)
TB = C.tb()
M = R.M
F = C.family(VF)
items, prows = F["items"], F["rows"]
T = len(items)
ev = TB.load_vlkeb("eval")
norm = C.norm_q
R.say(f"{BB}: family {VF}, master {F['master']}, T = {T}, {len(prows)} probe rows")
summ = {"backbone": BB, "family": VF, "master_log": F["master"], "seed": R.SEED, "T": T, "m": M, "kinds": list(R.KINDS),
        "amendment": "PROTOCOL.md section 6, Review round 1, item R5"}


# ----------------------------------------------------------------------------- the source pool of the hard negatives
def source_pool():
    ref = R.jload(os.path.join(R.CACHE, "work", BB, "select", "select_summary.json"))["hard_negatives"]["source_pool"]
    edit_all = {}
    for name, n in ref["registered_vlkeb_logs"]:                 # the streams the hard negatives' pool was built against
        its, _ = TB.load_stream(TB.LOGS[name]["stream"])
        assert len(its) == n, (name, len(its), n)
        for d in its:
            edit_all.setdefault(d["uid"], d)
    for fam in ("Q", "L"):                                       # the long streams of BOTH backbones
        for d in C.family(fam)["items"]:
            edit_all.setdefault(d["uid"], d)
    probe_uids = {p["uid"] for fam in ("Q", "L") for p in C.family(fam)["probes"]}
    ek = {(d["image"], d["src"], d["alt"]) for d in edit_all.values()}
    em, en = set(), set()
    for d in edit_all.values():
        em |= {C.mid(d["image"]), C.mid(d["image_rephrase"])}
        en |= {TB.norm_answer(d["alt"]), TB.norm_answer(d["pred_text"])}
    pool, excl = [], collections.Counter()
    for d in ev:
        if d["uid"] in probe_uids:
            excl["probe"] += 1
        elif (d["image"], d["src"], d["alt"]) in ek:
            excl["edit_key"] += 1
        elif C.mid(d["image"]) in em or C.mid(d["m_loc"]) in em:
            excl["edit_mid"] += 1
        elif TB.norm_answer(d["pred_text"]) in en or TB.norm_answer(d["alt"]) in en:
            excl["edit_name"] += 1
        else:
            pool.append(d)
    got = {"n_items": len(pool), "excluded": dict(excl), "n_edit_items_excluded_against": len(edit_all),
           "n_distinct_questions": len({norm(p["m_loc_q"]) for p in pool})}
    want = {k: ref[k] for k in got}
    if got != want:
        raise RuntimeError(f"the source pool of the hard negatives is not reproduced: {got} vs recorded {want}")
    hn = R.jsonl_read(os.path.join(C.HN_NEW, BB, "rows.jsonl"))
    pu = {d["uid"] for d in pool}
    new = [r for r in hn if r["new"] and r["hn_type"] in ("HN-I", "HN-Q")]
    if not new or any(r["src_uid"] not in pu for r in new):
        raise RuntimeError("a new hard negative has a source item outside the recomputed pool")
    return pool, dict(got, n_new_hard_negatives_with_source_in_pool=len(new),
                      streams=[n for n, _ in ref["registered_vlkeb_logs"]] + ["long streams of families Q and L"])


pool, pool_rec = source_pool()
summ["source_pool"] = pool_rec
R.say(f"  source pool reproduced: {pool_rec['n_items']} items, {pool_rec['n_distinct_questions']} distinct questions")

# ----------------------------------------------------------------------------- every query of every key cache (rule b)
CACHES = [BB] + [c for c in R.TEST_CACHES if c != BB]          # a FIXED list (rc_check.py also compares with any cache that appears later)
test_q, index = {}, None                           # (image md5, question stripped lower case) -> 'cache|family|kind' of the first row with it
summ["test_query_sources"] = {}
for c in CACHES:                                   # the backbone's own cache first, then the other caches under 10_final/cache
    p = os.path.join(R.CACHE, c, "index.jsonl")
    ix = R.jsonl_read(p)
    if c == BB:
        index = ix
    n0 = len(test_q)
    for e in ix:
        if e.get("image") is not None:
            test_q.setdefault((e["image_md5"], norm(e["prompt"])), f"{c}|{e['family']}|{e['kind']}")
    kinds_ = collections.Counter(f"{e['family']}|{e['kind']}" for e in ix if e.get("image") is not None)
    rec = {"index": f"10_final/cache/{c}/index.jsonl", "md5": R.file_md5(p), "n_sequences": len(ix),
           "n_image_sequences_by_family_kind": dict(sorted(kinds_.items())), "n_distinct_image_queries_added": len(test_q) - n0}
    for name, path in (("hard_negatives", os.path.join(C.HN_NEW, c, "rows.jsonl")), ("benign", os.path.join(C.BENIGN, c, "rows.jsonl"))):
        if os.path.exists(path):                   # the published row files themselves: every row must be in the index
            rows_ = R.jsonl_read(path)
            md5_of = {e["image"]: e["image_md5"] for e in ix if e.get("image") is not None}
            miss = [r for r in rows_ if r.get("image") is not None and (md5_of.get(r["image"]), norm(r["prompt"])) not in test_q]
            if miss:
                raise RuntimeError(f"{len(miss)} {name} rows of {path} are not in the index of cache {c}")
            rec[name] = {"n_rows": len(rows_), "md5": R.file_md5(path)}
    summ["test_query_sources"][c] = rec
bad = [e["sid"] for e in index if e["kind"] == "rel" and e["family"] == VF and e["image_md5"] != C.image_md5(items[int(e["edit"])]["image"])]
if bad:
    raise RuntimeError(f"image md5 of the index and of the md5 memo differ for {len(bad)} edit rows")
summ["test_query_sources"]["n_distinct_image_queries(md5, question stripped lower case)"] = len(test_q)
test_hits = collections.Counter()                  # which test rows caused the discards of rule (b)
hn_q_strings = {norm(e["prompt"]) for e in index if e["kind"] == "hn_i"}       # question strings / images of the backbone's test hard negatives
hn_images = {e["image_md5"] for e in index if e["kind"] == "hn_q"}
R.say(f"  test queries: {len(test_q)} distinct (image content, question) of the image sequences of the caches {CACHES}")

# ----------------------------------------------------------------------------- flags against the edits / probes of the stream
q_edit = {}                                        # question string (src or rephrase) -> first stream position
for i, d in enumerate(items):
    for q in (d["src"], d["rephrase"]):
        q_edit.setdefault(norm(q), i)
q_probe = {norm(r["prompt"]) for r in prows}
img_probe = {C.image_md5(r["image"]) for r in prows if r["image"] is not None}
query_probe = {C.query_key(r["ptype"], r["prompt"], r["image"]) for r in prows}


def flags(prompt, image):
    nq = norm(prompt)
    return {"q_in_edit": nq in q_edit, "q_in_edit_first_pos": q_edit.get(nq), "q_in_probe": nq in q_probe,
            "img_in_probe": C.image_md5(image) in img_probe, "query_in_probe": C.query_key("image", prompt, image) in query_probe}


def own_q(d):
    return (norm(d["src"]), norm(d["rephrase"]))


def mids(d):
    return {C.mid(d["image"]), C.mid(d["image_rephrase"])}


def names(d):
    return {TB.norm_answer(d["alt"]), TB.norm_answer(d["pred_text"])}


# ----------------------------------------------------------------------------- the selection
rows, shortfall = [], []
for i, d in enumerate(items):
    uid = d["uid"]
    ec = sorted(set(cats(d["src"])) | set(cats(d["rephrase"])))
    img = C.image_md5(d["image"])
    oq, dm, dn = own_q(d), mids(d), names(d)
    seen = set()                                   # (image md5, question) of this edit's recombinations so far (rule d)
    base = {"family": VF, "edit_uid": uid, "edit_pos": i, "edit": i}
    tail = {"edit_src": d["src"], "edit_rephrase": d["rephrase"], "edit_alt": d["alt"], "edit_pred": d["pred_text"]}

    def add(kind, k, image, prompt, key, red, src_uid, src_field, q_cats, same, extra=None, b=None, scope=None):
        seen.add(key)
        r = dict(rid=f"{kind}:{uid}:{k}", kind=kind, **base, k=k, image=image, prompt=prompt, fmt="vqa", ptype="image", image_md5=key[0],
                 src_uid=src_uid, src_field=src_field, edit_b_uid=None if b is None else items[b]["uid"], edit_b_pos=b, edit_b=b, scope=scope,
                 min_T=(i if b is None else max(i, b)) + 1, q_cats=q_cats, edit_cats=ec, same_category=same, **flags(prompt, image), **tail)
        r.update(extra or {})
        r.update(n_redraws=int(sum(red.values())), redraws=dict(sorted(red.items())))
        rows.append(r)

    # --- X-I: the edit's image + the image-locality question of a pool item
    red, k = collections.Counter(), 0
    for j in np.random.default_rng(R.useed("X-I", VF, uid)).permutation(len(pool)):
        p = pool[int(j)]
        qn = norm(p["m_loc_q"])
        key = (img, qn)
        if qn in oq:
            red["own_question"] += 1
        elif key in test_q:
            red["test_query"] += 1
            test_hits["X-I|" + test_q[key]] += 1
        elif key in seen:
            red["repeat"] += 1
        else:
            qc = cats(p["m_loc_q"])
            add("X-I", k, d["image"], p["m_loc_q"], key, red, p["uid"], "m_loc_q", qc, bool(set(qc) & set(ec)))
            red, k = collections.Counter(), k + 1
            if k == M:
                break
    assert k == M, ("X-I", uid, k)
    # --- X-Q: the edit's question + the entity image of a pool item
    red, k = collections.Counter(), 0
    for j in np.random.default_rng(R.useed("X-Q", VF, uid)).permutation(len(pool)):
        p = pool[int(j)]
        key = (C.image_md5(p["image"]), oq[0])
        if C.mid(p["image"]) in dm or TB.norm_answer(p["pred_text"]) in dn or TB.norm_answer(p["alt"]) in dn:
            red["related"] += 1
        elif key in test_q:
            red["test_query"] += 1
            test_hits["X-Q|" + test_q[key]] += 1
        elif key in seen:
            red["repeat"] += 1
        else:
            add("X-Q", k, p["image"], d["src"], key, red, p["uid"], "image", cats(d["src"]), bool(set(cats(p["src"])) & set(cats(d["src"]))),
                extra={"src_item_question": p["src"], "src_item_pred": p["pred_text"]})
            red, k = collections.Counter(), k + 1
            if k == M:
                break
    assert k == M, ("X-Q", uid, k)
    # --- X-X: the edit's image + the question of another edit of the stream
    red, k = collections.Counter(), 0
    plan = [("batch", 2), ("group", M)] if i < 100 else [("group", M)]       # numbers = rows wanted once the scope is done
    for scope, upto in plan:
        lo, n = (10 * (i // 10), 10) if scope == "batch" else (100 * (i // 100), 100)
        cand = [j for j in range(lo, min(lo + n, T)) if j != i]
        for jj in np.random.default_rng(R.useed("X-X", VF, uid, scope)).permutation(len(cand)):
            if k == upto:
                break
            j = cand[int(jj)]
            e = items[j]
            qn = norm(e["src"])
            key = (img, qn)
            if qn in oq:
                red["own_question"] += 1
            elif key in test_q:
                red["test_query"] += 1
                test_hits["X-X|" + test_q[key]] += 1
            elif dm & mids(e) or dn & names(e):
                red["related"] += 1
            elif key in seen:
                red["repeat"] += 1
            else:
                qc = cats(e["src"])
                add("X-X", k, d["image"], e["src"], key, red, e["uid"], "edit_src", qc, bool(set(qc) & set(ec)), b=j, scope=scope,
                    extra={"edit_b_src": e["src"], "edit_b_alt": e["alt"], "edit_b_pred": e["pred_text"]})
                red, k = collections.Counter(), k + 1
        if scope == "batch" and k < 2:
            shortfall.append({"edit_pos": i, "edit_uid": uid, "n_batch_partners": k})
    assert k == M, ("X-X", uid, k)

assert len(rows) == 3 * M * T and len({r["rid"] for r in rows}) == len(rows)
qid = {}
for r in rows:
    r["query_id"] = qid.setdefault((r["image_md5"], r["prompt"]), len(qid))
cq = collections.Counter(r["query_id"] for r in rows)
for r in rows:
    r["n_same_query"] = cq[r["query_id"]]
PATH = os.path.join(OUT, "rows.jsonl")
R.jsonl_write(rows, PATH)


# ----------------------------------------------------------------------------- summary (no time stamp: it reproduces as well)
def by_kind(fn):
    return {k: fn([r for r in rows if r["kind"] == k]) for k in R.KINDS}


def redraw_stats(sub):
    n = np.array([r["n_redraws"] for r in sub])
    why = collections.Counter()
    for r in sub:
        why.update(r["redraws"])
    return {"n_rows": len(sub), "n_redraws_total": int(n.sum()), "by_reason": dict(sorted(why.items())), "n_rows_with_a_redraw": int((n > 0).sum()),
            "mean_per_row": round(float(n.mean()), 4), "max_per_row": int(n.max()),
            "histogram": {str(v): int(c) for v, c in sorted(collections.Counter(n.tolist()).items())}}


prefixes = [t for t in R.PREFIXES[BB] if t <= T]
summ["n_rows"] = len(rows)
summ["n_by_kind"] = by_kind(len)
summ["rows_per_edit"] = by_kind(lambda s: sorted(collections.Counter(collections.Counter(r["edit_pos"] for r in s).values()).items()))
summ["redraws"] = dict(by_kind(redraw_stats), all=redraw_stats(rows))
summ["test_query_discards_by(kind | cache | family | kind of the first test row with that query)"] = dict(sorted(test_hits.items()))
xx = [r for r in rows if r["kind"] == "X-X"]
summ["x_x"] = {"scope": dict(sorted(collections.Counter(r["scope"] for r in xx).items())),
               "scope_first100": dict(sorted(collections.Counter(r["scope"] for r in xx if r["edit_pos"] < 100).items())),
               "batch_shortfall": {"n_edits": len(shortfall), "edits": shortfall[:50]},
               "group_partners_of_first100_in_own_batch": sum(1 for r in xx if r["edit_pos"] < 100 and r["scope"] == "group"
                                                              and r["edit_b_pos"] // 10 == r["edit_pos"] // 10),
               "partner_same_group_of_100": sum(1 for r in xx if r["edit_b_pos"] // 100 == r["edit_pos"] // 100),
               "partner_later_than_owner": sum(1 for r in xx if r["edit_b_pos"] > r["edit_pos"])}
summ["rows_usable_by_log_length_T"] = {str(t): dict(by_kind(lambda s, t=t: sum(1 for r in s if r["min_T"] <= t)),
                                                    edits_with_fewer_than_m_cross_rows=sum(1 for i in range(t) if sum(
                                                        1 for r in rows[3 * M * i + 2 * M:3 * M * (i + 1)] if r["min_T"] <= t) < M))
                                       for t in prefixes}
summ["flags"] = {"same_category": by_kind(lambda s: sum(bool(r["same_category"]) for r in s)),
                 "question_string_also_an_edit_question_or_rephrase": by_kind(lambda s: sum(bool(r["q_in_edit"]) for r in s)),
                 "X-I_rows_cross_like_by_log_length_T(q_in_edit_first_pos < T)": {
                     str(t): sum(1 for r in rows if r["kind"] == "X-I" and r["edit_pos"] < t and r["q_in_edit"] and r["q_in_edit_first_pos"] < t)
                     for t in prefixes},
                 "q_in_probe": by_kind(lambda s: sum(bool(r["q_in_probe"]) for r in s)),
                 "img_in_probe": by_kind(lambda s: sum(bool(r["img_in_probe"]) for r in s)),
                 "query_in_probe": by_kind(lambda s: sum(bool(r["query_in_probe"]) for r in s))}
summ["shared_with_test_hard_negatives_of_the_backbone(same generator, other query)"] = {
    "X-I_rows_whose_question_string_is_the_question_of_a_test_HN-I_row": sum(1 for r in rows if r["kind"] == "X-I" and norm(r["prompt"]) in hn_q_strings),
    "X-Q_rows_whose_image_is_the_image_of_a_test_HN-Q_row": sum(1 for r in rows if r["kind"] == "X-Q" and r["image_md5"] in hn_images),
    "n_distinct_question_strings_of_test_HN-I_rows": len(hn_q_strings), "n_distinct_images_of_test_HN-Q_rows": len(hn_images)}
summ["queries"] = {"n_distinct(image content, prompt)": len(qid),
                   "rows_in_duplicate_query_groups": by_kind(lambda s: sum(1 for r in s if r["n_same_query"] > 1)),
                   "n_distinct_partner_items": by_kind(lambda s: len({r["src_uid"] for r in s})),
                   "n_distinct_question_strings": by_kind(lambda s: len({norm(r["prompt"]) for r in s})),
                   "n_distinct_images": by_kind(lambda s: len({r["image_md5"] for r in s}))}
summ["rows_file"] = {"name": "rows.jsonl", "n_lines": len(rows), "sha256": R.file_sha256(PATH), "bytes": os.path.getsize(PATH)}
R.jdump(summ, os.path.join(OUT, "select_summary.json"))
R.say(f"  rows: {summ['n_by_kind']}; redraws {({k: (v['n_redraws_total'], v['by_reason']) for k, v in summ['redraws'].items()})}")
R.say(f"  X-X: {summ['x_x']['scope']}, batch shortfall {len(shortfall)} edits; distinct queries {len(qid)} of {len(rows)} rows")
R.say(f"  wrote {PATH} ({summ['rows_file']['bytes']} bytes, sha256 {summ['rows_file']['sha256'][:16]})")
