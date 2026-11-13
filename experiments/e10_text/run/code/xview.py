"""E10 / R22: the VIEW pool of the relational workload and the keys of its prompts.

VIEW pool (R22 a): 5000 records that are in no log and in neither the REPAIR nor the HELD-OUT pool, by record, seed 20261109:
the build's permutation of the eligible records (pools.json: REPAIR = its first 1000, HELD-OUT = the next 1000; reproduced and
checked here) is continued; records whose subject string is the subject of a request of the reported stream (any of its 2000
requests, so that one pool serves every reported log) are excluded and counted; the first 5000 remaining records are the pool,
its first 2500 the REPORT half, the other 2500 the TEST half.
A tuple's prompt is the record's request prompt; its key trace is the module input at the positions of the prompt (the value of
a tuple is the first token of the greedy continuation, so only prompt positions are presented).
ViewStore: float32 keys of the prompts of the VIEW records and of the stream's first 500 requests under the pre-edit model
(run/store/view), with the interface of the adapter's key cache; text ids are VBASE + record index (disjoint from the build's
text ids, so that the adapter's feature cache cannot mix the two).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

import xcommon as X
import e10t
from e10t import data as D

VBASE = 10_000_000
N_VIEW, SEED = 5000, 20261109
STORE = X.RUN + "/store/view"
POOL_FILE = X.OUT + "/r22/view_pool.json"


def view_pool() -> dict:
    if os.path.exists(POOL_FILE):
        return X.jload(POOL_FILE)
    cf = D.counterfact()
    pools = X.jload(X.TXOUT + "/pools.json")
    logged = set(range(pools["stream_records"][0], pools["stream_records"][1] + 1)) | set(range(pools["dev_records"][0], pools["dev_records"][1] + 1))
    if any(cf[i]["case_id"] != i for i in range(len(cf))):
        raise RuntimeError("record index != case id")
    eligible = [i for i in range(len(cf)) if i not in logged]
    perm = np.random.default_rng(SEED).permutation(len(eligible))
    rep, ho = [eligible[int(j)] for j in perm[:1000]], [eligible[int(j)] for j in perm[1000:2000]]
    if sorted(rep) != sorted(pools["repair_records"]) or sorted(ho) != sorted(pools["heldout_records"]):
        raise RuntimeError("the build's pools are not reproduced by its stated rule")
    subj = {cf[i]["requested_rewrite"]["subject"] for i in range(pools["stream_records"][0], pools["stream_records"][1] + 1)}
    subj_cf = {s.casefold() for s in subj}
    view, excl, excl_cf_only = [], [], 0
    for j in perm[2000:]:
        i = eligible[int(j)]
        s = cf[i]["requested_rewrite"]["subject"]
        if s in subj:
            excl.append(i)
            continue
        excl_cf_only += int(s.casefold() in subj_cf)
        view.append(i)
        if len(view) == N_VIEW:
            break
    out = {"seed": SEED, "rule": __doc__.split("ViewStore")[0].strip(), "n": len(view), "records": view, "report": view[:N_VIEW // 2], "test": view[N_VIEW // 2:],
           "n_excluded_shared_subject": len(excl), "excluded": excl, "n_kept_whose_subject_matches_only_case_insensitively": excl_cf_only,
           "n_candidates_scanned": len(view) + len(excl), "pools_reproduced": True}
    X.jdump(out, POOL_FILE)
    return out


def prompt_of(rec: dict) -> str:
    rr = rec["requested_rewrite"]
    return rr["prompt"].format(rr["subject"])


class ViewStore:
    """Key-cache interface (missing, n_prompt, n_pos, keys, stack) over the prompt keys of the view tuples."""
    parts = ["view"]

    def __init__(self):
        ix = X.jload(os.path.join(STORE, "index.json"))
        self.arr = np.load(os.path.join(STORE, "keys.npy"), mmap_mode="r")
        self.where = {VBASE + int(r): (int(o), int(n)) for r, o, n in zip(ix["records"], ix["offset"], ix["n_prompt"])}
        self.ids = {VBASE + int(r): t for r, t in zip(ix["records"], ix["input_ids"])}
        self.meta = ix["meta"]

    def missing(self, tids):
        return [int(t) for t in tids if int(t) not in self.where]

    def n_prompt(self, tid):
        return self.where[int(tid)][1]

    n_pos = n_prompt

    def keys(self, tid):
        o, n = self.where[int(tid)]
        return self.arr[o:o + n]

    def stack(self, tids):
        ks = [np.asarray(self.keys(t)) for t in tids]
        off = np.concatenate([[0], np.cumsum([k.shape[0] for k in ks])]).astype(np.int64)
        return (torch.from_numpy(np.concatenate(ks)) if ks else torch.zeros(0, 6400)), torch.from_numpy(off)


class VLog:
    """What e10t.gate.features / decide need of a log, over the view keys."""

    def __init__(self, L, store: ViewStore):
        self.name, self.T, self.kc = L.name + ":view", L.T, store


def build_store(B, n_stream: int = 500) -> dict:
    """Keys of the prompts of the VIEW records and of the first n_stream requests of the reported stream (pre-edit model)."""
    if os.path.exists(os.path.join(STORE, "index.json")):
        return X.jload(os.path.join(STORE, "index.json"))["meta"]
    os.makedirs(STORE, exist_ok=True)
    cf, tok = D.counterfact(), B.tok
    recs = list(range(n_stream)) + view_pool()["records"]
    ids = [tok(prompt_of(cf[r]))["input_ids"] for r in recs]
    L = e10t.open_log("TX-G2XL-rome-T500")
    for i in range(n_stream):                                              # the request prompt of an edited tuple is the build's request prompt
        if ids[i] != L.ids(L.rows["request"][i]["text_id"]):
            raise RuntimeError("request prompt tokens differ from the build's")
    lens = [len(s) for s in ids]
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    arr = np.lib.format.open_memmap(os.path.join(STORE, "keys.npy.tmp"), mode="w+", dtype=np.float32, shape=(int(off[-1]), 6400))
    B.check_w0()
    for b in B.batches(lens, extra=0):
        xs = B.keys([ids[i] for i in b])
        for i, x in zip(b, xs):
            arr[off[i]:off[i + 1]] = x.float().cpu().numpy()
    arr.flush()
    del arr
    os.replace(os.path.join(STORE, "keys.npy.tmp"), os.path.join(STORE, "keys.npy"))
    meta = {"n_records": len(recs), "n_positions": int(off[-1]), "n_stream": n_stream, "what": "module inputs of the edited module under the pre-edit model at the prompt positions",
            "layout": "right-padded batches (e10t.backbone.batches, extra=0)", "code": e10t.code_fingerprint()}
    X.jdump({"records": recs, "offset": [int(x) for x in off[:-1]], "n_prompt": lens, "input_ids": ids, "meta": meta}, os.path.join(STORE, "index.json"), indent=None)
    return meta
