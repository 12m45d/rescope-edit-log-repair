"""Dataset-aware stream / probe loaders for the 09_testbed_scale logs (in-process patch of
tb_common, no file of 06_testbed_v3 is modified).

    install(TB)  replaces TB.load_stream and TB.select_probes by wrappers that
      * delegate to the ORIGINAL functions for VLKEB streams (bit-identical behaviour), and
      * handle MMKE-Bench streams (edit_batch.jsonl records with "dataset": "MMKE-Bench"):
          load_stream   -> the MMKE train items (vendor_data_extra.mmke_items, VLKEB-like schema,
                           image paths made absolute), every field checked against the stream file
          select_probes -> the testbed's P8 rule on MMKE eval: eligible = eval items (all three
                           edit types, loader order) that are not edit items [uid and (image, src,
                           alt)], have m_loc and loc, and are the FIRST eval item carrying their
                           image-probe row (m_loc, m_loc_q) and their text-probe row (loc) -- MMKE
                           eval repeats locality rows across items (1060 distinct image rows and
                           1451 distinct text rows over 1759 items), and a repeated row would be the
                           same probe twice; then rng(42).choice(eligible, 500), sorted.
Used by the build scripts (sc_common) and by scale_registry (harness users).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MMKE_TAG = "MMKE-Bench"
_MM = {}


def mmke_pool(split):
    """All MMKE items of a split (entity, visual, user in loader order), absolute image paths."""
    if split not in _MM:
        import vendor_data_extra as X
        items = []
        for et in X.EDIT_TYPES:
            for d in X.mmke_items(et, split):
                d = dict(d)
                for f in ("image", "image_rephrase", "m_loc"):      # [redacted: describes a related submission under double-blind review]
                    v = d.get(f)
                    if v is not None and not os.path.isabs(v):
                        d[f] = os.path.join(X.MMKE_IMAGE_DIR, v)
                items.append(d)
        _MM[split] = items
    return [dict(d) for d in _MM[split]]


def is_mmke_items(items):
    return bool(items) and str(items[0].get("uid", "")).startswith("mmke-")


def mmke_select_probes(TB, edit_items):
    ev = mmke_pool("eval")
    uids = {d["uid"] for d in edit_items}
    keys = {(d["image"], d["src"], d["alt"]) for d in edit_items}
    base = [d for d in ev if d["uid"] not in uids and d.get("m_loc") and d.get("loc")
            and (d["image"], d["src"], d["alt"]) not in keys]
    seen_i, seen_t, elig = set(), set(), []
    for d in base:
        ki, kt = (d["m_loc"], d["m_loc_q"]), d["loc"]
        if ki in seen_i or kt in seen_t:
            continue
        seen_i.add(ki)
        seen_t.add(kt)
        elig.append(d)
    rng = np.random.default_rng(TB.SEED_SEL)
    idx = sorted(rng.choice(len(elig), TB.N_PROBES, replace=False))
    probes = [elig[i] for i in idx]
    return probes, {"n_eval": len(ev), "n_eligible": len(elig), "n_eligible_uid_only": len(base),
                    "key_exclude": True, "probe_uids_sha": TB._sha([p["uid"] for p in probes]),
                    "dataset": MMKE_TAG,
                    "rule": "eval items not edited, with m_loc and loc, first occurrence of their "
                            "(m_loc, m_loc_q) and loc rows; rng(42) choice of 500, sorted"}


def install(TB):
    if getattr(TB, "_scale_data_installed", False):
        return
    orig_load, orig_sel = TB.load_stream, TB.select_probes

    def load_stream(path):
        recs = TB.jsonl_read(path)
        if not (recs and recs[0].get("dataset") == MMKE_TAG):
            return orig_load(path)
        tr = {d["uid"]: d for d in mmke_pool("train")}
        items = []
        for r in recs:
            d = tr[r["uid"]]
            assert d["src"] == r["subject"] and d["alt"] == r["alt"] and d["image"] == r["image"], r
            items.append(d)
        assert [r["edit_id"] for r in recs] == list(range(len(recs)))
        return items, recs

    def select_probes(edit_items, key_exclude=False):
        if is_mmke_items(edit_items):
            return mmke_select_probes(TB, edit_items)
        return orig_sel(edit_items, key_exclude=key_exclude)

    TB.load_stream, TB.select_probes = load_stream, select_probes
    TB._scale_data_orig = (orig_load, orig_sel)
    TB._scale_data_installed = True
