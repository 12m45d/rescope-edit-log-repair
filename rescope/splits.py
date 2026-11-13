"""Splits of the evaluation data.

Probe rows (probe_records_v2.jsonl; one image row and one text row per VLKEB eval probe item):
  REPAIR             rows of the items in the harness REPAIR pool (07_repair_v3/harness/sampler.py: 500 items split
                     once by item uid, seed 20261109): the only rows a method may use (complaints, calibration).
  HELD-OUT           rows of the other items: reported.
  HELD-OUT-DISJOINT  the HELD-OUT rows whose QUERY (ptype, prompt, image) does not occur in the REPAIR pool.
                     VLKEB eval items share image-locality probes, so the item-level split leaves 25 valid
                     HELD-OUT rows per log that are the same input as a REPAIR row ("twins", P12 Anomaly 0);
                     a method fitted on REPAIR rows has seen them.  Report both, select on REPAIR only.
  HELD-OUT-STRICT    the HELD-OUT-DISJOINT rows whose QUESTION TEXT (ptype, prompt) does not occur in the REPAIR
                     pool either.  About 90 valid disjoint image rows per log ask a question that a REPAIR row
                     asks about another image (same prompt, other image: not the same input, but not unrelated);
                     this pool is for sensitivity analyses.
Duplicates: some HELD-OUT-DISJOINT rows are the same query as another HELD-OUT row (29 valid rows per log lie in
such groups).  Every annotated row carries `query_id` (rows with the same id are the same model input): bootstrap
and count by query_id clusters, not by row, when independence matters.
Edits: half A = the first T // 2 edits (selection), half B = the rest (report), as in P11a.

Model-free: works on the row dicts and an item -> pool mapping.
"""
from __future__ import annotations

import numpy as np

POOLS = ("repair", "heldout", "heldout_disjoint", "heldout_strict", "all")


def query_key(row) -> tuple:
    """Identity of a probe query: the same key means the same model input."""
    return (row["ptype"], row["prompt"], row["image"])


def question_key(row) -> tuple:
    """Identity of the question text alone (the image may differ)."""
    return (row["ptype"], row["prompt"])


def annotate(rows, pool_of_item) -> list:
    """One dict per probe row (same order): probe_idx, ptype, group, uid, pool, y0_valid, twin_of (groups of the
    REPAIR rows with the same query, valid or not), disjoint (HELD-OUT row without a REPAIR twin), question_in_repair
    (the row's question text occurs in the REPAIR pool, with any image), strict (disjoint and not question_in_repair),
    query_id (index of the row's distinct query: equal ids = the same model input) and n_same_query (rows with it)."""
    rep, repq, qid, cnt = {}, set(), {}, {}
    for r in rows:
        k = query_key(r)
        qid.setdefault(k, len(qid))
        cnt[k] = cnt.get(k, 0) + 1
        if pool_of_item[r["probe_idx"]] == "repair":
            rep.setdefault(k, []).append(r["group"])
            repq.add(question_key(r))
    out = []
    for r in rows:
        pool = pool_of_item[r["probe_idx"]]
        k = query_key(r)
        twins = rep.get(k, []) if pool == "heldout" else []
        disjoint = pool == "heldout" and not twins
        qin = question_key(r) in repq
        out.append({"probe_idx": r["probe_idx"], "ptype": r["ptype"], "group": r["group"], "uid": r["uid"], "pool": pool,
                    "y0_valid": bool(r["y0_valid"]), "twin_of": list(twins), "disjoint": disjoint, "question_in_repair": qin,
                    "strict": disjoint and not qin, "query_id": qid[k], "n_same_query": cnt[k]})
    return out


def in_pool(info: dict, pool: str) -> bool:
    """Membership of an annotated row in 'repair' | 'heldout' | 'heldout_disjoint' | 'heldout_strict' | 'all'."""
    if pool not in POOLS:
        raise ValueError(pool)
    if pool == "all":
        return True
    if pool == "heldout_disjoint":
        return info["disjoint"]
    if pool == "heldout_strict":
        return info["strict"]
    return info["pool"] == pool


def select(rows, pool_of_item, pool: str, valid_only: bool = True) -> list:
    """The rows of a pool (the original row dicts), by default only those with a valid pre-edit answer y0."""
    return [r for r, a in zip(rows, annotate(rows, pool_of_item)) if in_pool(a, pool) and (a["y0_valid"] or not valid_only)]


def summary(rows, pool_of_item) -> dict:
    """Counts of the split (valid rows unless stated), for reporting next to every held-out number."""
    an = annotate(rows, pool_of_item)
    v = [a for a in an if a["y0_valid"]]
    d1 = {r["group"]: bool(r["is_complaint_D1"]) for r in rows}
    held = [a for a in v if a["pool"] == "heldout"]
    tw = [a for a in held if not a["disjoint"]]
    dj = [a for a in held if a["disjoint"]]
    per_q = {}
    for a in dj:
        per_q[a["query_id"]] = per_q.get(a["query_id"], 0) + 1
    return {"n_rows": len(an), "n_valid": len(v), "n_repair": sum(a["pool"] == "repair" for a in v), "n_heldout": len(held),
            "n_heldout_disjoint": len(dj), "n_twins": len(tw),
            "n_twins_by_ptype": {p: sum(a["ptype"] == p for a in tw) for p in ("image", "text")},
            "n_twins_D1_at_WE": sum(d1[a["group"]] for a in tw),
            "n_heldout_strict": sum(a["strict"] for a in dj),
            "n_disjoint_question_in_repair": {p: sum(a["question_in_repair"] and a["ptype"] == p for a in dj) for p in ("image", "text")},
            "n_disjoint_distinct_queries": len(per_q),
            "n_disjoint_rows_in_duplicate_groups": sum(c for c in per_q.values() if c > 1),
            "n_distinct_queries": {p: len({query_key(r) for r in rows if r["ptype"] == p}) for p in ("image", "text")}}


def sample(items, n: int, seed: int) -> list:
    """Seeded random subset of n items without replacement, in the original order (calibration draws: the conformal
    guarantee is marginal over the draw, so use several seeds and report the mean; items[:n] is NOT a random subset)."""
    if not 0 <= n <= len(items):
        raise ValueError(f"cannot draw {n} of {len(items)} items")
    idx = np.sort(np.random.default_rng(seed).choice(len(items), n, replace=False))
    return [items[int(i)] for i in idx]


def edit_halves(T: int):
    """(A, B): edit ids of the selection half and of the report half."""
    return list(range(T // 2)), list(range(T // 2, T))
