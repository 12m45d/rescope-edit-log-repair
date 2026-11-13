"""VLKEB loaders, seeded sampling of edit streams / evaluation sets, and the
non-finite pre-edit filter ([redacted: related project] §2.2).

Item conventions (added by :func:`load_vlkeb`, original fields untouched):
  uid        "{split}:{index in file}" (stable id)
  pred_text  ``pred`` with "_" -> " " (VLKEB stores wiki ids such as "Laredo,_Texas")
  port_q / port_a   from ``port_new[0]``, on every split that can supply them (see below)
  eval_idx   onehop only: index of the same record in eval.json (onehop is a subset of eval)

Port fields: ``port_new`` is a one-hop-only *field* in the shipped data -- all 1278 records of
``eval_edit_onehop.json`` carry exactly one entry, and none of the 3174 ``eval.json`` records do.
Every editing experiment [redacted: describes a related submission under double-blind review] evaluates ``load_vlkeb("eval")``, so without a backfill the
Port probe silently matched nothing and every run logged ``Port = null`` (746/746 rows).  The
one-hop split is a subset of eval by ``(image, src, alt)`` (1278/1278 matched), so eval items are
backfilled from it via the reverse of the ``eval_idx`` map.  Only the read path is changed: the
raw ``port_new`` structure is left alone, and ``port_q``/``port_a`` are absent (not None) on eval
records with no one-hop counterpart, which is what :func:`metrics.build_probes` tests for.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import hashlib
import json
import logging
import os
import time
from typing import Iterable, Sequence

import numpy as np

ROOT = f"{_RP.EXT_VIEW}"
VLKEB_DIR = os.path.join(ROOT, "data", "datasets", "VLKEB")
SPLIT_FILES = {"eval": "eval.json", "onehop": "eval_edit_onehop.json", "train": "train.json"}
# [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
log = logging.getLogger("v7.data")

_CACHE: dict[str, list[dict]] = {}


def _key(d: dict) -> tuple:
    return (d["image"], d["src"], d["alt"])


_ONEHOP_PORTS: dict[tuple, tuple[str, str]] | None = None


def _onehop_ports() -> dict[tuple, tuple[str, str]]:
    """{``(image, src, alt)``: (port_q, port_a)} read straight from the one-hop file.

    Read from the raw file rather than via :func:`load_vlkeb` to avoid recursing back into the
    eval branch that calls this helper.  Entries with a malformed ``port_new`` are skipped
    rather than raising: a missing Port probe degrades one column, a crash loses the whole run.
    Memoised, because the eval branch of :func:`load_vlkeb` asks for it once per item.
    """
    global _ONEHOP_PORTS
    if _ONEHOP_PORTS is not None:
        return _ONEHOP_PORTS
    path = os.path.join(VLKEB_DIR, SPLIT_FILES["onehop"])
    out: dict[tuple, tuple[str, str]] = {}
    for d in json.load(open(path)):
        pn = d.get("port_new")
        if not isinstance(pn, (list, tuple)) or not pn:
            continue
        e = pn[0]
        if not isinstance(e, dict) or "port_q" not in e or "port_a" not in e:
            continue
        out[_key(d)] = (e["port_q"], e["port_a"])
    _ONEHOP_PORTS = out
    return out


def load_vlkeb(split: str) -> list[dict]:
    """Load a VLKEB split ("eval" 3174, "onehop" 1278, "train" 5000) as a list of dicts.

    Returns fresh shallow copies each call (safe to mutate the list, not the nested port_new)."""
    if split not in SPLIT_FILES:
        raise ValueError(f"split must be one of {list(SPLIT_FILES)}")
    if split not in _CACHE:
        raw = json.load(open(os.path.join(VLKEB_DIR, SPLIT_FILES[split])))
        items = []
        eval_index = None
        if split == "onehop":
            eval_index = {_key(d): i for i, d in enumerate(load_vlkeb("eval"))}
        for i, d in enumerate(raw):
            d = dict(d)
            d["uid"] = f"{split}:{i}"
            d["split"] = split
            d["pred_text"] = d["pred"].replace("_", " ")
            if "port_new" in d and d["port_new"]:
                d["port_q"] = d["port_new"][0]["port_q"]
                d["port_a"] = d["port_new"][0]["port_a"]
            elif split == "eval":
                # eval.json carries no port_new; backfill from the one-hop subset so the
                # Port probe has something to score (see the module docstring).
                qa = _onehop_ports().get(_key(d))
                if qa is not None:
                    d["port_q"], d["port_a"] = qa
            if eval_index is not None:
                d["eval_idx"] = eval_index.get(_key(d), -1)
            items.append(d)
        _CACHE[split] = items
    return [dict(d) for d in _CACHE[split]]


# [not released: code that no released code calls; it served the related project (README, "What can be reproduced")]
