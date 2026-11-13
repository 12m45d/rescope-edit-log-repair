"""Check of the recombination selection of one backbone: it is reproduced byte for byte, and it is disjoint from every test query.

    python rc_check.py <qwen25vl | llava> [--fast]            -> cache2/recomb/<backbone>/check_select.json, exit code 0 iff ok
                                                              (--fast: check_select_fast.json; the full check stays the record)

1. REPRODUCTION.  rc_select.py is run twice in fresh interpreters with two different PYTHONHASHSEED values into
   work/<backbone>/select_recheck_<n>; rows.jsonl and select_summary.json must equal the files in use byte for byte.
2. DISJOINTNESS, with code that shares nothing with rc_select.py but the question normalisation (strip, lower case):
   the queries (image content, question) of rows.jsonl are compared with
     - every sequence of every key cache that exists under 10_final/cache (index.jsonl; a cache that appeared after the selection
       is listed in 'caches_compared' and compared like the others),
     - every row of 10_final/cache/hard_negatives/*/rows.jsonl and 10_final/cache/benign/*/rows.jsonl,
     - the raw sources of the backbone's family: (image, src), (image, rephrase), (image_rephrase, src) of every edit of the
       long stream, and (image, prompt) of every row of the master log's probe_records_v2.jsonl,
     - the portability queries of the one-hop streams (09_testbed_scale/streams_part3: image of a one-hop edit + its one-hop
       question; they are in no key cache and were not part of the selection's rule (b): compared here all the same),
   as exact strings and after the normalisation.  Image content = md5 of the image FILE, recomputed here from the files (all
   images of rows.jsonl and of the test rows; --fast trusts the md5 values stored in the cache indices and in rows.jsonl).
3. RULES, re-derived from the edit stream and the VLKEB eval split: 4 rows per kind and edit, in stream order; images / questions
   / partners as defined; partner items in the (independently recomputed) source pool of the hard negatives; rule (a) own
   question, rule (c) no shared entity / name / answer for X-X, rule (d) no repeat within an edit; batch / group scope and min_T
   of X-X; flags (q_in_edit and its first position, q_in_probe, img_in_probe, query_in_probe, same_category); redraw counts are
   consistent with the recorded reasons; the solver batches of the master log are the position decades.
"""
from __future__ import annotations

import collections
import hashlib
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc_common as R  # noqa: E402
from rc_common import C  # noqa: E402

BB = sys.argv[1]
FAST = "--fast" in sys.argv[2:]
CFG = C.BACKBONES[BB]
VF = CFG["vlkeb"]
DIR = R.bb_dir(BB)
import numpy  # noqa: E402
out = {"backbone": BB, "fast": FAST, "checks": {}, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
       "environment": {"python": sys.version.split()[0], "numpy": numpy.__version__,
                       "note": "the seeded permutations are numpy.random.default_rng(seed).permutation: reproduction is checked in this environment"}}
ok_all = True


def check(name, ok, **info):
    global ok_all
    ok_all = ok_all and bool(ok)
    out["checks"][name] = dict(ok=bool(ok), **info)
    R.say(f"  [{'ok' if ok else 'FAIL'}] {name}: {info if len(str(info)) < 400 else str(info)[:400] + ' ...'}")


def nq(s):
    return (s or "").strip().lower()


# ============================================================================= 1. reproduction
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
rep = []
for n, hs in enumerate(("1", "20261111")):
    d = os.path.join(R.WORK, BB, f"select_recheck_{n}")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    p = subprocess.run([sys.executable, os.path.join(R.FEAT, "rc_select.py"), BB], env=dict(env, RC_SELECT_DIR=d, PYTHONHASHSEED=hs),
                       capture_output=True, text=True)
    with open(os.path.join(d, "log.txt"), "w") as f:
        f.write(p.stdout + p.stderr)
    same = {f: p.returncode == 0 and os.path.exists(os.path.join(d, f)) and open(os.path.join(d, f), "rb").read() == open(os.path.join(DIR, f), "rb").read()
            for f in ("rows.jsonl", "select_summary.json")}
    rep.append({"PYTHONHASHSEED": hs, "returncode": p.returncode, "identical": same})
summ = R.jload(os.path.join(DIR, "select_summary.json"))
sha = R.file_sha256(R.rows_path(BB))
check("reproduced_byte_for_byte", all(all(r["identical"].values()) for r in rep) and sha == summ["rows_file"]["sha256"],
      runs=rep, rows_sha256=sha, bytes=os.path.getsize(R.rows_path(BB)))

# ============================================================================= data (independent of rc_select.py)
rows = R.jsonl_read(R.rows_path(BB))
TB = C.tb()
F = C.family(VF)
items, prows = F["items"], F["rows"]
T = len(items)
ev = {d["uid"]: d for d in TB.load_vlkeb("eval")}
from vendor_v7.vlm import VLKEB_IMG_DIR  # noqa: E402


def fpath(img):
    return img if os.path.isabs(img) else os.path.join(VLKEB_IMG_DIR, img)


def md5_files(paths):
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        h = hashlib.md5()
        with open(fpath(p), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return p, h.hexdigest()
    with ThreadPoolExecutor(4) as ex:
        return dict(ex.map(one, paths))


# ============================================================================= 2. disjointness from every test query
caches = sorted(c for c in os.listdir(R.CACHE) if os.path.exists(os.path.join(R.CACHE, c, "index.jsonl")))
check("caches_compared(information)", True, caches_present=caches, caches_of_the_selection=list(R.TEST_CACHES),
      caches_that_appeared_after_the_selection=sorted(set(caches) - set(R.TEST_CACHES)),
      note="rule (b) of the selection used the caches of the selection; a cache that appeared later is compared below all the same, and a "
           "row that equals one of its queries fails 'disjoint_from_every_test_query'")
test_rows = []                                     # (source, kind, image path, prompt, stored md5 or None)
for c in caches:
    for e in R.jsonl_read(os.path.join(R.CACHE, c, "index.jsonl")):
        if e.get("image") is not None:
            test_rows.append((f"index:{c}", f"{e.get('family')}|{e['kind']}", e["image"], e["prompt"], e.get("image_md5")))
for sub in ("hard_negatives", "benign"):
    for c in sorted(os.listdir(os.path.join(R.CACHE, sub))):
        p = os.path.join(R.CACHE, sub, c, "rows.jsonl")
        if os.path.exists(p):
            for r in R.jsonl_read(p):
                if r.get("image") is not None:
                    test_rows.append((f"{sub}:{c}", r.get("hn_type") or r.get("ptype"), r["image"], r["prompt"], None))
for i, d in enumerate(items):                      # the raw edit stream of the family
    test_rows += [(f"stream:{VF}", "rel", d["image"], d["src"], None), (f"stream:{VF}", "tgen", d["image"], d["rephrase"], None),
                  (f"stream:{VF}", "igen", d["image_rephrase"], d["src"], None)]
for r in prows:                                    # the master log's probe records
    if r["image"] is not None:
        test_rows.append((f"probe_records:{F['master']}", "probe", r["image"], r["prompt"], None))
import glob  # noqa: E402
PORT = sorted(glob.glob(os.path.join(C.EXP, "09_testbed_scale", "streams_part3", "*.jsonl")))
for p in PORT:                                     # portability queries of the one-hop streams (image of a one-hop edit + its one-hop question):
    for r in R.jsonl_read(p):                      # test queries of the one-hop logs, which are not sequences of any key cache
        if r.get("port_q") and r.get("image"):
            test_rows.append(("onehop_portability", "port", r["image"], r["port_q"], None))
t0 = time.time()
paths = sorted({t[2] for t in test_rows} | {r["image"] for r in rows})
if FAST:
    md5 = {t[2]: t[4] for t in test_rows if t[4]}
    md5.update({r["image"]: r["image_md5"] for r in rows})
    for p in paths:
        if p not in md5:
            md5[p] = C.image_md5(p)
else:
    md5 = md5_files(paths)
bad_md5 = [t[2] for t in test_rows if t[4] and md5[t[2]] != t[4]] + [r["rid"] for r in rows if md5[r["image"]] != r["image_md5"]]
check("image_md5_recomputed_from_files", not bad_md5, n_image_files=len(paths), n_distinct_contents=len(set(md5.values())), n_differing=len(bad_md5),
      recomputed=not FAST, seconds=round(time.time() - t0, 1))
exact, normd = collections.defaultdict(set), collections.defaultdict(set)
for src, kind, img, prompt, _ in test_rows:
    exact[(md5[img], prompt)].add(f"{src}|{kind}")
    normd[(md5[img], nq(prompt))].add(f"{src}|{kind}")
hit_e = [(r["rid"], sorted(exact[(md5[r["image"]], r["prompt"])])) for r in rows if (md5[r["image"]], r["prompt"]) in exact]
hit_n = [(r["rid"], sorted(normd[(md5[r["image"]], nq(r["prompt"]))])) for r in rows if (md5[r["image"]], nq(r["prompt"])) in normd]
per_src = collections.Counter(t[0] for t in test_rows)
check("disjoint_from_every_test_query", not hit_e and not hit_n, n_recombination_rows=len(rows), n_test_rows_with_an_image=len(test_rows),
      n_test_rows_by_source=dict(sorted(per_src.items())), n_distinct_test_queries_exact=len(exact), n_distinct_test_queries_normalised=len(normd),
      n_rows_equal_to_a_test_query_exact=len(hit_e), n_rows_equal_to_a_test_query_normalised=len(hit_n), examples=(hit_e + hit_n)[:5])
txt = {nq(e["prompt"]) for c in caches for e in R.jsonl_read(os.path.join(R.CACHE, c, "index.jsonl")) if e.get("image") is None}
check("text_queries_of_the_caches_have_no_image(information)", True, n_distinct_text_queries=len(txt),
      note="text rows (probe text rows, benign text rows) have no image; every recombination has one, so no recombination is a text query")

# ============================================================================= 3. the rules, re-derived
sys.path.insert(0, R.CACHE_CODE)
from fc_cats import cats  # noqa: E402
err = collections.Counter()
ex = {}


def bad(name, r):
    err[name] += 1
    ex.setdefault(name, r["rid"])


# the source pool of the hard negatives, recomputed (rule of 10_final/cache/code/fc_select.py)
names = [n for n, _ in R.jload(os.path.join(R.CACHE, "work", BB, "select", "select_summary.json"))["hard_negatives"]["source_pool"]["registered_vlkeb_logs"]]
edits = {}
for n in names:
    for d in TB.load_stream(TB.LOGS[n]["stream"])[0]:
        edits[d["uid"]] = d
for fam in ("Q", "L"):
    for d in C.family(fam)["items"]:
        edits[d["uid"]] = d
probe_uids = {p["uid"] for fam in ("Q", "L") for p in C.family(fam)["probes"]}
E_key = {(d["image"], d["src"], d["alt"]) for d in edits.values()}
E_mid = {x.split("/")[0] for d in edits.values() for x in (d["image"], d["image_rephrase"])}
E_name = {TB.norm_answer(x) for d in edits.values() for x in (d["alt"], d["pred_text"])}
pool = {u for u, d in ev.items() if u not in probe_uids and (d["image"], d["src"], d["alt"]) not in E_key
        and d["image"].split("/")[0] not in E_mid and d["m_loc"].split("/")[0] not in E_mid
        and TB.norm_answer(d["pred_text"]) not in E_name and TB.norm_answer(d["alt"]) not in E_name}
hn_src = {r["src_uid"] for r in R.jsonl_read(os.path.join(C.HN_NEW, BB, "rows.jsonl")) if r["new"] and r["hn_type"] in ("HN-I", "HN-Q")}
check("source_pool_recomputed", len(pool) == summ["source_pool"]["n_items"] and hn_src <= pool, n_items=len(pool),
      n_sources_of_new_hard_negatives=len(hn_src), all_in_pool=hn_src <= pool)

q_first = {}
for i, d in enumerate(items):
    for q in (d["src"], d["rephrase"]):
        q_first.setdefault(nq(q), i)
p_q = {nq(r["prompt"]) for r in prows}
p_img = {md5[r["image"]] for r in prows if r["image"] is not None}
p_query = {(md5[r["image"]], r["prompt"]) for r in prows if r["image"] is not None}


def ent(d):
    return ({d["image"].split("/")[0], d["image_rephrase"].split("/")[0]}, {TB.norm_answer(d["alt"]), TB.norm_answer(d["pred_text"])})


M = R.M
if len(rows) != 3 * M * T or len({r["rid"] for r in rows}) != len(rows):
    err["row_count"] += 1
for i, d in enumerate(items):
    sub = rows[3 * M * i:3 * M * (i + 1)]
    own = {nq(d["src"]), nq(d["rephrase"])}
    seen = set()
    em, en = ent(d)
    ec = sorted(set(cats(d["src"])) | set(cats(d["rephrase"])))
    for n, r in enumerate(sub):
        kind, k = R.KINDS[n // M], n % M
        if (r["kind"], r["k"], r["edit_pos"], r["edit"], r["edit_uid"], r["rid"], r["family"]) != (kind, k, i, i, d["uid"], f"{kind}:{d['uid']}:{k}", VF):
            bad("order_or_identity", r)
            continue
        key = (md5[r["image"]], nq(r["prompt"]))
        if key in seen:
            bad("rule_d_repeat_within_edit", r)
        seen.add(key)
        if r["fmt"] != "vqa" or r["ptype"] != "image":
            bad("format", r)
        j = r["edit_b_pos"]
        if kind == "X-I":
            p = ev.get(r["src_uid"])
            if p is None or r["src_uid"] not in pool or r["image"] != d["image"] or r["prompt"] != p["m_loc_q"] or r["src_field"] != "m_loc_q" or j is not None:
                bad("X-I_definition", r)
            qc = cats(r["prompt"])
            same = bool(set(qc) & set(ec))
        elif kind == "X-Q":
            p = ev.get(r["src_uid"])
            if p is None or r["src_uid"] not in pool or r["prompt"] != d["src"] or r["image"] != p["image"] or r["src_field"] != "image" or j is not None:
                bad("X-Q_definition", r)
            elif p["image"].split("/")[0] in em or {TB.norm_answer(p["alt"]), TB.norm_answer(p["pred_text"])} & en:
                bad("X-Q_partner_related", r)
            qc = cats(d["src"])
            same = bool(p is not None and set(cats(p["src"])) & set(cats(d["src"])))
        else:
            if j is None or j == i or not 0 <= j < T or items[j]["uid"] != r["edit_b_uid"] or r["src_uid"] != r["edit_b_uid"] or r["edit_b"] != j \
                    or r["image"] != d["image"] or r["prompt"] != items[j]["src"] or r["src_field"] != "edit_src":
                bad("X-X_definition", r)
                continue
            jm, jn = ent(items[j])
            if em & jm or en & jn:
                bad("rule_c_shared_entity_name_answer", r)
            want = "batch" if (i < 100 and k < 2) else "group"
            inside = (j // 10 == i // 10) if want == "batch" else (j // 100 == i // 100)
            if r["scope"] != want or not inside:
                bad("X-X_scope", r)
            qc = cats(r["prompt"])
            same = bool(set(qc) & set(ec))
        if kind != "X-Q" and nq(r["prompt"]) in own:
            bad("rule_a_own_question", r)
        if r["min_T"] != (i if j is None else max(i, j)) + 1:
            bad("min_T", r)
        fl = (nq(r["prompt"]) in q_first, q_first.get(nq(r["prompt"])), nq(r["prompt"]) in p_q, md5[r["image"]] in p_img, (md5[r["image"]], r["prompt"]) in p_query)
        if fl != (r["q_in_edit"], r["q_in_edit_first_pos"], r["q_in_probe"], r["img_in_probe"], r["query_in_probe"]):
            bad("flags", r)
        if r["same_category"] != same or r["q_cats"] != qc or r["edit_cats"] != ec:
            bad("same_category", r)
        if r["n_redraws"] != sum(r["redraws"].values()) or set(r["redraws"]) - {"own_question", "test_query", "related", "repeat"} or r["n_redraws"] < 0:
            bad("redraws", r)
        if (r["edit_src"], r["edit_rephrase"], r["edit_alt"], r["edit_pred"]) != (d["src"], d["rephrase"], d["alt"], d["pred_text"]):
            bad("edit_fields", r)
check("rules", not err, n_rows=len(rows), n_edits=T, violations=dict(err), first_examples=ex,
      rules=["4 rows per kind and edit in stream order", "definitions of X-I / X-Q / X-X", "partner items in the source pool", "(a) own question",
             "(c) X-X: no shared entity / name / answer", "(d) no repeat within an edit", "X-X scope: batch | group", "min_T", "flags", "same_category",
             "redraw bookkeeping"])
qid, groups = {}, collections.defaultdict(list)
for r in rows:
    groups[qid.setdefault((md5[r["image"]], r["prompt"]), len(qid))].append(r)
check("query_ids", all(r["query_id"] == qid[(md5[r["image"]], r["prompt"])] and r["n_same_query"] == len(groups[r["query_id"]]) for r in rows),
      n_distinct_queries=len(qid), n_rows_in_duplicate_groups=sum(len(g) for g in groups.values() if len(g) > 1),
      duplicate_groups_by_kinds=dict(sorted(collections.Counter("+".join(sorted({x["kind"] for x in g})) for g in groups.values() if len(g) > 1).items())),
      duplicates_are_between_different_edits_only=all(len({x["edit_pos"] for x in g}) == len(g) for g in groups.values()))
import json  # noqa: E402
pb = json.load(open(os.path.join(TB.OUT_ROOT, F["master"], "metrics.json")))["per_batch"]
check("solver_batches_are_position_decades", all(b["edit_ids"] == list(range(10 * b["batch"], 10 * b["batch"] + 10)) for b in pb),
      master_log=F["master"], n_batches=len(pb))
pre = {str(t): {k: sum(1 for r in rows if r["kind"] == k and r["min_T"] <= t) for k in R.KINDS} for t in R.PREFIXES[BB] if t <= T}
check("rows_usable_by_log_length", all(pre[t][k] == M * int(t) for t in pre for k in ("X-I", "X-Q")) and
      all(pre[t]["X-X"] == M * int(t) for t in pre if int(t) % 100 == 0) and all(pre[t]["X-X"] >= 2 * int(t) for t in pre), usable=pre,
      note="X-I / X-Q: 4 per edit at every length; X-X: 4 per edit at multiples of 100, at least the 2 batch partners per edit at 10 / 20 / 50")
red = {k: {"n_redraws": sum(r["n_redraws"] for r in rows if r["kind"] == k), "rows_with_redraw": sum(r["n_redraws"] > 0 for r in rows if r["kind"] == k),
           "max": max(r["n_redraws"] for r in rows if r["kind"] == k)} for k in R.KINDS}
out.update(ok=bool(ok_all), n_rows=len(rows), n_by_kind={k: sum(r["kind"] == k for r in rows) for k in R.KINDS}, redraws=red, rows_sha256=sha)
OUTFILE = os.path.join(DIR, "check_select_fast.json" if FAST else "check_select.json")       # the full check is the record of the store
R.jdump(out, OUTFILE)
for n in range(2):
    shutil.rmtree(os.path.join(R.WORK, BB, f"select_recheck_{n}"), ignore_errors=True)
R.say(f"{BB}: check_select ok = {ok_all} ({len(out['checks'])} checks) -> {OUTFILE}")
sys.exit(0 if ok_all else 1)
