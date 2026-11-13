"""E10 step 9: pools and prompt tables (CPU, no model).

Pools.  Records that are in no log (not in the reported stream 0 .. 1999, not in the development log
15000 .. 15099), in case-id order, are permuted with numpy default_rng(20261109) -- the convention of the harness
split (07_repair_v3/harness/sampler.py) -- and the first 1000 records of the permutation are the REPAIR pool, the
next 1000 the HELD-OUT pool.  Benign prompts of a pool record = its request prompt and its two paraphrase prompts.

Flags per pool record: the subject string occurs as the subject of a logged request (dev, stream, and the first
stream index at which it occurs, so that the flag can be read for every prefix T).  Flags per benign prompt: the
prompt string equals a prompt of a logged record (request / paraphrase / neighbourhood), or a benign prompt of the
other pool.

Prompt table (out/prompts.jsonl): one row per (record, kind, k); rows with the same text share `text_id`.
    set      'repair' | 'heldout' | 'stream' | 'dev'
    kind     'request' | 'paraphrase' | 'neighborhood'
Token ids are those of tok(prompt) (no BOS), as the public evaluation feeds prompts.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

C.setup_public()
import numpy as np  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

N_POOL = 1000
data = C.load_counterfact()
n = len(data)
stream_idx = list(range(C.STREAM_START, C.STREAM_START + C.STREAM_T))
dev_idx = list(range(C.DEV_START, C.DEV_START + C.DEV_T))
logged = set(stream_idx) | set(dev_idx)
eligible = [i for i in range(n) if i not in logged]
perm = np.random.default_rng(C.POOL_SEED).permutation(len(eligible))
repair = sorted(eligible[int(j)] for j in perm[:N_POOL])
heldout = sorted(eligible[int(j)] for j in perm[N_POOL:2 * N_POOL])
assert not (set(repair) & set(heldout)) and not ((set(repair) | set(heldout)) & logged)


def subj(i):
    return data[i]["requested_rewrite"]["subject"]


def request_prompt(i):
    rw = data[i]["requested_rewrite"]
    return rw["prompt"].format(rw["subject"])


first_stream = {}
for pos, i in enumerate(stream_idx):
    first_stream.setdefault(subj(i), pos)
dev_subjects = {subj(i) for i in dev_idx}
first_stream_cf = {}
for pos, i in enumerate(stream_idx):
    first_stream_cf.setdefault(subj(i).casefold().strip(), pos)

rows = []


def add(set_name, i, pos):
    rec = data[i]
    base = {"set": set_name, "record": i, "case_id": rec["case_id"], "pos": pos, "subject": subj(i)}
    rows.append(dict(base, kind="request", k=0, text=request_prompt(i)))
    for k, p in enumerate(rec["paraphrase_prompts"]):
        rows.append(dict(base, kind="paraphrase", k=k, text=p))
    if set_name in ("stream", "dev"):
        for k, p in enumerate(rec["neighborhood_prompts"]):
            rows.append(dict(base, kind="neighborhood", k=k, text=p))


for pos, i in enumerate(stream_idx):
    add("stream", i, pos)
for pos, i in enumerate(dev_idx):
    add("dev", i, pos)
for pos, i in enumerate(repair):
    add("repair", i, pos)
for pos, i in enumerate(heldout):
    add("heldout", i, pos)

# text ids and token ids
tok = AutoTokenizer.from_pretrained(C.MODEL_DIR)
text_id = {}
for r in rows:
    r["text_id"] = text_id.setdefault(r["text"], len(text_id))
texts = [None] * len(text_id)
for t, j in text_id.items():
    texts[j] = t
ids = tok(texts)["input_ids"]
for r in rows:
    r["prompt_id"] = f"{r['set']}:{r['case_id']}:{r['kind']}:{r['k']}"
    r["n_tok"] = len(ids[r["text_id"]])
assert len({r["prompt_id"] for r in rows}) == len(rows)

# where does every text occur
occ = collections.defaultdict(set)
for r in rows:
    occ[r["text_id"]].add((r["set"], r["kind"]))
min_stream_pos = {}
for r in rows:
    if r["set"] == "stream":
        key = (r["text_id"], r["kind"])
        min_stream_pos[key] = min(min_stream_pos.get(key, 10 ** 9), r["pos"])

# prompts that MENTION the subject of a logged request (whole-word match of the subject string).  ROME writes at
# the last subject token, so such a prompt presents (nearly) the key of that edit whatever it asks.
import re  # noqa: E402


def mention_index(subject_pos):
    """subject -> first position; returns f(text, own_subject) -> smallest position of a mentioned subject or None."""
    subs = sorted(subject_pos, key=len, reverse=True)
    rx = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in subs) + r")(?!\w)")

    def f(text, own=None):
        best = None
        for m in rx.finditer(text):
            if m.group(0) == own:
                continue
            q = subject_pos[m.group(0)]
            best = q if best is None or q < best else best
        if best is None and own is not None:
            # the regex returns the longest alternative at a position: look for other subjects nested in the own one
            for x in subs:
                if x != own and x in text and re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", text.replace(own, " ")):
                    q = subject_pos[x]
                    best = q if best is None or q < best else best
        return best
    return f


dev_first = {}
for pos, i in enumerate(dev_idx):
    dev_first.setdefault(subj(i), pos)
m_stream, m_dev = mention_index(first_stream), mention_index(dev_first)
cache_s, cache_d = {}, {}
for r in rows:
    own_s = r["subject"] if r["set"] == "stream" else None
    own_d = r["subject"] if r["set"] == "dev" else None
    ks, kd = (r["text_id"], own_s), (r["text_id"], own_d)
    if ks not in cache_s:
        cache_s[ks] = m_stream(r["text"], own_s)
    if kd not in cache_d:
        cache_d[kd] = m_dev(r["text"], own_d)
    # smallest stream / dev position of a logged subject mentioned in the prompt (for logged records: another
    # request's subject; the record's own subject is not counted)
    r["mentions_stream_subject_pos"] = cache_s[ks]
    r["mentions_dev_subject_pos"] = cache_d[kd]
    if r["set"] in ("stream", "dev") and r["kind"] == "neighborhood":
        o = occ[r["text_id"]]
        r["text_first_stream_request_pos"] = min_stream_pos.get((r["text_id"], "request"))
        r["text_is_dev_request"] = ("dev", "request") in o

pool_records = []
for set_name, lst in (("repair", repair), ("heldout", heldout)):
    for pos, i in enumerate(lst):
        s = subj(i)
        pool_records.append({"set": set_name, "pos": pos, "record": i, "case_id": data[i]["case_id"], "subject": s,
                             "relation_id": data[i]["requested_rewrite"]["relation_id"],
                             "subject_in_stream": s in first_stream, "subject_first_stream_pos": first_stream.get(s),
                             "subject_in_dev": s in dev_subjects,
                             "subject_casefold_first_stream_pos": first_stream_cf.get(s.casefold().strip())})
by_rec = {(p["set"], p["record"]): p for p in pool_records}
for r in rows:
    if r["set"] in ("repair", "heldout"):
        p = by_rec[(r["set"], r["record"])]
        o = occ[r["text_id"]]
        other = "heldout" if r["set"] == "repair" else "repair"
        r.update({
            "subject_in_stream": p["subject_in_stream"], "subject_first_stream_pos": p["subject_first_stream_pos"],
            "subject_in_dev": p["subject_in_dev"],
            "text_in_other_pool": any(s == other for s, _ in o),
            "text_is_stream_request": ("stream", "request") in o, "text_is_stream_paraphrase": ("stream", "paraphrase") in o,
            "text_is_stream_neighborhood": ("stream", "neighborhood") in o,
            "text_first_stream_neighborhood_pos": min_stream_pos.get((r["text_id"], "neighborhood")),
            "text_first_stream_request_pos": min_stream_pos.get((r["text_id"], "request")),
            "text_is_dev_request": ("dev", "request") in o, "text_is_dev_paraphrase": ("dev", "paraphrase") in o,
            "text_is_dev_neighborhood": ("dev", "neighborhood") in o,
        })

os.makedirs(C.OUT, exist_ok=True)
with open(os.path.join(C.OUT, "prompts.jsonl"), "w") as f:
    import json
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
C.jdump({"texts": texts, "input_ids": ids}, os.path.join(C.OUT, "prompt_texts.json"), indent=None)


def cnt(pred):
    return sum(1 for r in rows if pred(r))


def pool_summary(set_name):
    pr = [p for p in pool_records if p["set"] == set_name]
    rr = [r for r in rows if r["set"] == set_name]
    out = {"n_records": len(pr), "n_benign_prompts": len(rr), "n_distinct_texts": len({r["text_id"] for r in rr}),
           "records_subject_in_stream": sum(p["subject_in_stream"] for p in pr),
           "records_subject_in_dev": sum(p["subject_in_dev"] for p in pr),
           "records_subject_in_stream_casefold": sum(p["subject_casefold_first_stream_pos"] is not None for p in pr)}
    for T in C.SNAPSHOTS:
        out[f"records_subject_in_T{T}"] = sum(p["subject_first_stream_pos"] is not None and p["subject_first_stream_pos"] < T for p in pr)
        out[f"prompts_text_is_neighborhood_of_T{T}"] = sum(
            r["text_first_stream_neighborhood_pos"] is not None and r["text_first_stream_neighborhood_pos"] < T for r in rr)
    for T in C.SNAPSHOTS:
        out[f"prompts_mention_subject_of_T{T}"] = sum(
            r["mentions_stream_subject_pos"] is not None and r["mentions_stream_subject_pos"] < T for r in rr)
    out["prompts_mention_subject_of_dev"] = sum(r["mentions_dev_subject_pos"] is not None for r in rr)
    for k in ("text_in_other_pool", "text_is_stream_request", "text_is_stream_paraphrase", "text_is_stream_neighborhood",
              "text_is_dev_request", "text_is_dev_paraphrase", "text_is_dev_neighborhood"):
        out["prompts_" + k] = sum(bool(r[k]) for r in rr)
    return out


summary = {
    "seed": C.POOL_SEED, "rule": "eligible = records in no log, case-id order; perm = numpy.random.default_rng(seed)."
                                 "permutation(len(eligible)); REPAIR = first 1000 of perm, HELD-OUT = next 1000",
    "n_records_total": n, "n_logged": len(logged), "n_eligible": len(eligible),
    "stream_records": [C.STREAM_START, C.STREAM_START + C.STREAM_T - 1], "dev_records": [C.DEV_START, C.DEV_START + C.DEV_T - 1],
    "repair_records": repair, "heldout_records": heldout,
    "pools": {s: pool_summary(s) for s in ("repair", "heldout")},
    "n_prompt_rows": len(rows), "n_distinct_texts": len(texts),
    "rows_by_set_kind": {f"{s}/{k}": c for (s, k), c in sorted(collections.Counter((r["set"], r["kind"]) for r in rows).items())},
    "token_length": {"min": int(min(map(len, ids))), "max": int(max(map(len, ids))), "mean": float(np.mean(list(map(len, ids))))},
    "stream_duplicate_subjects": len(stream_idx) - len(first_stream),
    "stream_dev_shared_subjects": len(set(first_stream) & dev_subjects),
    "logged_neighborhood_texts_that_are_logged_requests": {
        "stream": len({r["text_id"] for r in rows if r["set"] == "stream" and r["kind"] == "neighborhood"}
                      & {r["text_id"] for r in rows if r["set"] == "stream" and r["kind"] == "request"}),
        "dev": len({r["text_id"] for r in rows if r["set"] == "dev" and r["kind"] == "neighborhood"}
                   & {r["text_id"] for r in rows if r["set"] == "dev" and r["kind"] == "request"})},
    "neighborhood_prompts_mentioning_another_logged_subject": {
        **{f"T{T}": sum(1 for r in rows if r["set"] == "stream" and r["kind"] == "neighborhood" and r["pos"] < T
                        and r["mentions_stream_subject_pos"] is not None and r["mentions_stream_subject_pos"] < T)
           for T in C.SNAPSHOTS},
        "dev": sum(1 for r in rows if r["set"] == "dev" and r["kind"] == "neighborhood" and r["mentions_dev_subject_pos"] is not None)},
    "neighborhood_prompts_equal_to_a_logged_request": {
        **{f"T{T}": sum(1 for r in rows if r["set"] == "stream" and r["kind"] == "neighborhood" and r["pos"] < T
                        and r.get("text_first_stream_request_pos") is not None and r["text_first_stream_request_pos"] < T)
           for T in C.SNAPSHOTS},
        "dev": sum(1 for r in rows if r["set"] == "dev" and r["kind"] == "neighborhood" and r.get("text_is_dev_request"))},
    "written": C.now(),
}
C.jdump(pool_records, os.path.join(C.OUT, "pool_records.json"))
C.jdump(summary, os.path.join(C.OUT, "pools.json"))
print({k: v for k, v in summary.items() if k not in ("repair_records", "heldout_records")})
