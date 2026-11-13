"""fx.masks: the masks of amendment R8 (unseen logs).

  later_edits(L)                 the edits of a log that no development step has seen: stream position >= 500 on the Qwen stream,
                                 >= 100 on the LLaVA stream (LATER[family]); [] for a log that ends before
  question_disjoint(L)           predicate on an index row (of ANY key cache: probes, benign parts, MME): its question string
                                 (stripped, lower case) is NOT the question or the text rephrase of any edit of the LONGEST stream
                                 of the backbone (the whole cached family: 2000 edits on Qwen, 1000 on LLaVA), whatever the log
  keep(L, rows, kc=None)         (n,) bool of that predicate on rows
  Use: fx.draws.calibration(L, ..., mask="qdisjoint") (the draw is then taken from the filtered pool);
       L.rows("probe", pool="heldout_disjoint", valid=True, where=fx.masks.question_disjoint(L)); fx.pools.firing(L, M, P, mask=...);
       behavioural rows: select the sequences first (ses.tf(M, rows)), or filter fx.behave rows by their sequence:
       [r for r, k in zip(rows, fx.masks.keep(L, [L.kc.by_sid(r["csid"]) for r in rows])) if k].
"""
from __future__ import annotations

import numpy as np

LATER = {"Q": 500, "L": 100}                # first stream position of the "later edits" per stream family (R8)
_QS = {}


def later_from(L) -> int:
    if L.family not in LATER:
        raise KeyError(f"{L.name}: R8 names the later edits of the Qwen and LLaVA streams only (family {L.family!r})")
    return LATER[L.family]


def stream_pos(L, edits=None) -> list:
    """Stream position of the log's edits (index field edit_pos of the key cache: the position in the family's long stream)."""
    ed = range(L.T) if edits is None else edits
    return [int(L.index[L.edit_rows["rel"][int(i)]].get("edit_pos", i)) for i in ed]


def later_edits(L) -> list:
    """Edit ids of the log with stream position >= later_from(L)."""
    t0 = later_from(L)
    return [i for i, p in enumerate(stream_pos(L)) if p >= t0]


def stream_questions(L) -> frozenset:
    """Question strings (stripped, lower case) of the questions and text rephrases of EVERY edit of the longest stream of the
    log's stream family (all 'rel' and 'tgen' sequences of the family in the key cache; the I-Gen sequence asks the Rel question)."""
    from .logs import norm_q
    k = (L.reg["cache"], L.family)
    if k not in _QS:
        _QS[k] = frozenset(norm_q(L.index[j]["prompt"]) for j in L.kc.rows(kind=("rel", "tgen"), family=L.family))
    return _QS[k]


def n_stream_edits(L) -> int:
    return len(L.kc.rows(kind="rel", family=L.family))


def question_disjoint(L):
    """Predicate on an index row: True = keep (the row's question string is no question / rephrase of the longest stream)."""
    from .logs import norm_q
    qs = stream_questions(L)
    return lambda e: norm_q(e.get("prompt")) not in qs


def keep(L, rows, kc=None) -> np.ndarray:
    f = question_disjoint(L)
    ix = L.index if kc is None else kc.index
    return np.array([f(ix[int(j)]) for j in rows], dtype=bool)
