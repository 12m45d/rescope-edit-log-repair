"""Audit check 5: pools.  CPU, no model; the builder's code is not imported.

 a. Re-derivation of the split from the stated rule and seed (records in no log, index order;
    numpy default_rng(20261109).permutation; first 1000 -> REPAIR, next 1000 -> HELD-OUT) vs out/pools.json and the rows of
    out/prompts.jsonl.
 b. Pairwise disjointness BY RECORD of REPAIR / HELD-OUT / stream (0..1999) / dev (15000..15099); also by case_id.
 c. Subject-overlap flags, recomputed: pool records whose subject string is the subject of a logged request (per T), and
    the per-row flags of prompts.jsonl; the same after case folding.
 d. Beyond the registered flags (reported, not conditions): duplicate prompt TEXTS across sets, subjects shared between
    REPAIR and HELD-OUT, benign prompts that contain the subject of a logged request (the audit's own matching rule),
    neighbourhood prompts equal to a logged request / containing another logged subject, dev vs stream overlap.
Output: out/pools.json
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a_common as A  # noqa: E402

import numpy as np  # noqa: E402

SEED = 20261109
data = A.load_counterfact()
n = len(data)
stream = list(range(0, 2000))
dev = list(range(15000, 15100))
logged = set(stream) | set(dev)
eligible = [i for i in range(n) if i not in logged]
perm = np.random.default_rng(SEED).permutation(len(eligible))
my_repair = sorted(eligible[int(j)] for j in perm[:1000])
my_heldout = sorted(eligible[int(j)] for j in perm[1000:2000])
bp = A.jload(os.path.join(A.B_OUT, "pools.json"))
rows = [json.loads(l) for l in open(os.path.join(A.B_OUT, "prompts.jsonl"))]
out = {"written": A.now(), "seed_protocol": SEED, "seed_in_builder_pools_json": bp["seed"], "rule_in_builder_pools_json": bp["rule"]}


def subj(i):
    return data[i]["requested_rewrite"]["subject"]


def req_text(i):
    rw = data[i]["requested_rewrite"]
    return rw["prompt"].format(rw["subject"])


# ---- a. split
by_set = collections.defaultdict(set)
for r in rows:
    by_set[r["set"]].add(r["record"])
out["split"] = {
    "n_records": n, "n_logged": len(logged), "n_eligible": len(eligible),
    "repair_equals_rederivation": bp["repair_records"] == my_repair, "heldout_equals_rederivation": bp["heldout_records"] == my_heldout,
    "n_repair": len(bp["repair_records"]), "n_heldout": len(bp["heldout_records"]),
    "prompt_rows_records_equal": {s: sorted(by_set[s]) == {"repair": my_repair, "heldout": my_heldout, "stream": stream, "dev": dev}[s]
                                  for s in ("repair", "heldout", "stream", "dev")},
    "other_seed_would_differ": sorted(eligible[int(j)] for j in np.random.default_rng(SEED + 1).permutation(len(eligible))[:1000]) != my_repair,
    "first_repair_records": my_repair[:5], "first_heldout_records": my_heldout[:5]}
# ---- b. disjointness
sets = {"repair": set(bp["repair_records"]), "heldout": set(bp["heldout_records"]), "stream": set(stream), "dev": set(dev)}
names = list(sets)
out["disjoint_by_record"] = {f"{a}&{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]}
cids = {k: {data[i]["case_id"] for i in v} for k, v in sets.items()}
out["disjoint_by_case_id"] = {f"{a}&{b}": len(cids[a] & cids[b]) for i, a in enumerate(names) for b in names[i + 1:]}
out["all_pairwise_disjoint"] = all(v == 0 for v in out["disjoint_by_record"].values()) and all(v == 0 for v in out["disjoint_by_case_id"].values())
# rows of the prompt table: one request + two paraphrases per pool record; + ten neighbourhood prompts per logged record
cnt = collections.Counter((r["set"], r["kind"]) for r in rows)
out["prompt_rows"] = {f"{s}/{k}": c for (s, k), c in sorted(cnt.items())}
bad_text = 0
for r in rows:
    rec = data[r["record"]]
    t = req_text(r["record"]) if r["kind"] == "request" else rec[{"paraphrase": "paraphrase_prompts", "neighborhood": "neighborhood_prompts"}[r["kind"]]][r["k"]]
    bad_text += t != r["text"]
out["prompt_rows_text_not_equal_dataset"] = bad_text

# ---- c. subject flags
first_pos = {}
for pos, i in enumerate(stream):
    first_pos.setdefault(subj(i), pos)
dev_subj = {subj(i) for i in dev}
first_pos_cf = {}
for pos, i in enumerate(stream):
    first_pos_cf.setdefault(subj(i).casefold().strip(), pos)
dev_subj_cf = {s.casefold().strip() for s in dev_subj}
pr = {(p["set"], p["record"]): p for p in A.jload(os.path.join(A.B_OUT, "pool_records.json"))}
fl = {}
for name in ("repair", "heldout"):
    recs = bp[f"{name}_records"]
    f = {"n_records": len(recs)}
    for T in (100, 500, 1000, 2000):
        f[f"share_subject_with_T{T}"] = int(sum(first_pos.get(subj(i), 10 ** 9) < T for i in recs))
        f[f"share_subject_with_T{T}_casefold"] = int(sum(first_pos_cf.get(subj(i).casefold().strip(), 10 ** 9) < T for i in recs))
    f["share_subject_with_dev"] = int(sum(subj(i) in dev_subj for i in recs))
    f["share_subject_with_dev_casefold"] = int(sum(subj(i).casefold().strip() in dev_subj_cf for i in recs))
    f["builder_pools_json"] = {k: v for k, v in bp["pools"][name].items() if k.startswith("records_subject")}
    # per-record flags of the builder
    f["n_pool_records_flag_differs"] = int(sum((pr[(name, i)]["subject_first_stream_pos"] != first_pos.get(subj(i))) or
                                               (pr[(name, i)]["subject_in_dev"] != (subj(i) in dev_subj)) for i in recs))
    rr = [r for r in rows if r["set"] == name]
    f["n_prompt_rows"] = len(rr)
    f["n_prompt_rows_flag_differs"] = int(sum((r["subject_first_stream_pos"] != first_pos.get(subj(r["record"]))) or
                                              (bool(r["subject_in_dev"]) != (subj(r["record"]) in dev_subj)) for r in rr))
    f["flagged_records_examples"] = [(i, subj(i), first_pos[subj(i)]) for i in recs if subj(i) in first_pos][:6]
    fl[name] = f
out["subject_flags"] = fl
out["subject_flags_equal_builder_report"] = {
    "repair": [fl["repair"][f"share_subject_with_T{T}"] for T in (100, 500, 1000, 2000)] == [0, 6, 14, 22] and fl["repair"]["share_subject_with_dev"] == 0,
    "heldout": [fl["heldout"][f"share_subject_with_T{T}"] for T in (100, 500, 1000, 2000)] == [2, 4, 13, 19] and fl["heldout"]["share_subject_with_dev"] == 1}

# ---- d. beyond the registered flags
texts = collections.defaultdict(set)                        # text -> {(set, kind)}
for r in rows:
    texts[r["text"]].add((r["set"], r["kind"]))
d = {}
for name in ("repair", "heldout"):
    rr = [r for r in rows if r["set"] == name]
    other = "heldout" if name == "repair" else "repair"
    d[name] = {"n_prompts": len(rr), "n_distinct_texts": len({r["text"] for r in rr}),
               "text_is_in_other_pool": int(sum(any(s == other for s, _ in texts[r["text"]]) for r in rr)),
               "text_is_logged_stream_request": int(sum(("stream", "request") in texts[r["text"]] for r in rr)),
               "text_is_logged_stream_paraphrase": int(sum(("stream", "paraphrase") in texts[r["text"]] for r in rr)),
               "text_is_logged_stream_neighborhood": int(sum(("stream", "neighborhood") in texts[r["text"]] for r in rr)),
               "text_is_logged_dev_prompt": int(sum(any(s == "dev" for s, _ in texts[r["text"]]) for r in rr))}
ps = {name: collections.Counter(subj(i) for i in bp[f"{name}_records"]) for name in ("repair", "heldout")}
d["subjects_shared_between_repair_and_heldout"] = len(set(ps["repair"]) & set(ps["heldout"]))
d["subjects_repeated_within"] = {name: int(sum(c > 1 for c in ps[name].values())) for name in ps}
d["stream_duplicate_subjects"] = 2000 - len(first_pos)
d["stream_and_dev_shared_subjects"] = len(set(first_pos) & dev_subj)
d["stream_and_dev_shared_prompt_texts"] = int(sum(1 for t, o in texts.items() if any(s == "stream" for s, _ in o) and any(s == "dev" for s, _ in o)))


def mention_counter(subject_pos):
    """The audit's rule: the subject string occurs in the prompt delimited by non-word characters (or string ends)."""
    subs = sorted(subject_pos, key=len, reverse=True)
    rx = re.compile(r"(?<![0-9A-Za-z_])(?:" + "|".join(re.escape(s) for s in subs) + r")(?![0-9A-Za-z_])")

    def f(text, own=None):
        best = None
        for m in rx.finditer(text):
            if m.group(0) == own:
                continue
            q = subject_pos[m.group(0)]
            best = q if best is None or q < best else best
        return best
    return f


ms = mention_counter(first_pos)
dev_first = {}
for pos, i in enumerate(dev):
    dev_first.setdefault(subj(i), pos)
md = mention_counter(dev_first)
for name in ("repair", "heldout"):
    rr = [r for r in rows if r["set"] == name]
    got = [ms(r["text"]) for r in rr]
    d[name]["prompts_mentioning_logged_subject_audit_rule"] = {f"T{T}": int(sum(g is not None and g < T for g in got)) for T in (100, 500, 1000, 2000)}
    d[name]["prompts_mentioning_logged_subject_audit_rule"]["dev"] = int(sum(md(r["text"]) is not None for r in rr))
    d[name]["prompts_mentioning_logged_subject_builder"] = {**{f"T{T}": bp["pools"][name][f"prompts_mention_subject_of_T{T}"] for T in (100, 500, 1000, 2000)},
                                                            "dev": bp["pools"][name]["prompts_mention_subject_of_dev"]}
    d[name]["n_rows_mention_flag_differs_from_builder"] = int(sum((g is None) != (r["mentions_stream_subject_pos"] is None) for g, r in zip(got, rr)))
    # plain substring rule (no word boundary): an upper bound
    d[name]["prompts_containing_logged_subject_as_substring_T2000"] = int(sum(any(s in r["text"] for s in first_pos) for r in rr))
nb = [r for r in rows if r["set"] == "stream" and r["kind"] == "neighborhood"]
req_pos = {}
for pos, i in enumerate(stream):
    req_pos.setdefault(req_text(i), pos)
d["neighborhood_equal_to_a_logged_request"] = {f"T{T}": int(sum(r["pos"] < T and req_pos.get(r["text"], 10 ** 9) < T for r in nb)) for T in (100, 500, 1000, 2000)}
d["neighborhood_mentioning_another_logged_subject_audit_rule"] = {
    f"T{T}": int(sum(r["pos"] < T and (lambda g: g is not None and g < T)(ms(r["text"], r["subject"])) for r in nb)) for T in (100, 500, 1000, 2000)}
d["builder_neighborhood_equal"] = bp["neighborhood_prompts_equal_to_a_logged_request"]
d["builder_neighborhood_mention"] = bp["neighborhood_prompts_mentioning_another_logged_subject"]
# a neighbourhood prompt of record i has another subject but the TRUE object of record i; how many logged records share
# their relation prompt template with a pool record (benign prompts are other subjects under the same templates)
tmpl_logged = {data[i]["requested_rewrite"]["prompt"] for i in stream}
d["heldout_request_templates_also_used_by_a_logged_request"] = int(sum(data[i]["requested_rewrite"]["prompt"] in tmpl_logged for i in bp["heldout_records"]))
out["beyond_registered_flags"] = d
out["finished"] = A.now()
A.jdump(out, os.path.join(A.AOUT, "pools.json"))
for k in ("split", "disjoint_by_record", "all_pairwise_disjoint", "prompt_rows", "prompt_rows_text_not_equal_dataset", "subject_flags_equal_builder_report"):
    A.say(k, out[k])
A.say("flags", {k: {a: b for a, b in v.items() if a not in ("flagged_records_examples",)} for k, v in fl.items()})
A.say("beyond", json.dumps(d)[:3000])
