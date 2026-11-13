"""fx.pools: key-level pools (false firing only; NO behavioural path) and the guard that keeps test pools out of calibration.

POOLS (PROTOCOL.md section 6: R2, R17, "Conventions for the MME pools", R19 (e): gate level only, on cached keys)
  BEN-W / BEN-S / BEN-T   parts of the label-free benign pool of the backbone's key cache, AS DEFINED BY
                          10_final/cache2/benign_parts/<backbone>.json (split by source item, seed 20261111; this module only reads it)
        BEN-W   statistics part: used only by the feature packages (10_final/feat2); never a calibration source or a test pool here
        BEN-S   second calibration SOURCE (fx.draws.calibration(L, source="BEN-S"))
        BEN-T   TEST rows only: firing on all its rows at the prompt-only key
  MME-C / MME-T           the out-of-benchmark workload 10_final/cache2/mme/<backbone>/ (its own key cache; reader mme_pool.py)
        MME-C   calibration SOURCE (fx.draws.calibration(L, source="MME-C"))
        MME-T   TEST rows only.  The degenerate-answer filter is NOT applied to MME rows: every row with a valid pre-edit
                answer counts (its answers are Yes / No by construction)
  DEFAULT KEYS   BEN-*: which='decode' = the prompt-only key, every row of the part (R2: "firing on all its rows");
                        which='tf' = the rows of the answer subset with a valid pre-edit answer, teacher-forced key
                 MME-*: which='tf' = the rows with a valid pre-edit answer, teacher-forced key; which='decode' = every row
  A pool of another key cache (MME) addresses its rows by position in THAT cache's index: Pool.kc; never mix them with the
  log's sequences.  Benign-pool and MME rows have no item in the behavioural evaluation: fx.behave refuses them (R19 (e)).

    P   = fx.pools.get(L, "BEN-T")                    Pool: .name .rows .which .kc (None: the log's cache) .index .test_only
    dec = fx.pools.decide(L, M, P)                    decision of any gate fx.gate decides (scoped role-'last' gates, the router)
    st  = fx.pools.firing(L, M, "MME-T")              {n, fired (%), image / text (BEN), by category (MME), far / related, capped edits}
    fx.pools.guard(L, rows, use, kc=None)             raises PoolGuardError if a row is test-only (BEN-T, MME-T): every calibration
                                                      draw, statistics sample (null repair) and stream sample of the driver passes it
"""
from __future__ import annotations

import os
import sys

import numpy as np

import fx

BEN = ("BEN-W", "BEN-S", "BEN-T")
MME = ("MME-C", "MME-T")
NAMES = BEN + MME
TEST_ONLY = ("BEN-T", "MME-T")
SOURCES = ("BEN-S", "MME-C")                       # the parts that may calibrate
STATISTICS_ONLY = ("BEN-W",)
PARTS_DIR = fx.FINAL + "/cache2/benign_parts"
MME_DIR = fx.FINAL + "/cache2/mme"
RELATED = ("q_in_edit", "img_in_edit", "img_entity_in_edit", "q_in_probe", "img_in_probe", "query_in_probe", "img_entity_in_probe")
_PARTS, _MME, _SETS = {}, {}, {}


class PoolGuardError(ValueError):
    pass


class Pool:
    def __init__(self, name, rows, which, kc, index, backbone):
        self.name, self.rows, self.which, self.kc, self.index, self.backbone = name, list(rows), which, kc, index, backbone
        self.test_only = name in TEST_ONLY
        self.source = name in SOURCES

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return f"Pool({self.name}, {self.backbone}, {len(self.rows)} rows at '{self.which}', {'TEST ONLY' if self.test_only else 'source' if self.source else 'statistics'})"


def available(backbone: str) -> dict:
    """{pool name: bool}: the files of the pool exist for this backbone."""
    b = os.path.exists(os.path.join(PARTS_DIR, f"{backbone}.json"))
    m = os.path.exists(os.path.join(MME_DIR, backbone, "meta.json"))
    return {**{n: b for n in BEN}, **{n: m for n in MME}}


def benign_parts(L) -> dict:
    """{cache sid: 'BEN-W' | 'BEN-S' | 'BEN-T'} of the benign pool of the log's key cache, from cache2/benign_parts/<backbone>.json
    (the split is NOT computed here).  Raises if the file was made from another key cache or does not cover the pool."""
    bb = L.reg["cache"]
    if bb not in _PARTS:
        p = os.path.join(PARTS_DIR, f"{bb}.json")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no parts of the benign pool for the key cache {bb!r}: {p}")
        blob = fx.io.jload(p)
        md5 = (blob.get("source") or {}).get("key_cache_index_md5")
        if md5 is not None and L.kc.meta.get("index_md5") is not None and md5 != L.kc.meta["index_md5"]:
            raise RuntimeError(f"{p} was made from another key cache (index md5 {md5}, cache {L.kc.meta['index_md5']})")
        parts = blob["parts"]
        sids = [L.index[j]["sid"] for j in L.benign_rows]
        if set(sids) != set(parts) or set(parts.values()) - set(BEN):
            raise RuntimeError(f"{p} does not hold exactly the {len(sids)} rows of the benign pool of {bb}")
        _PARTS[bb] = {"parts": parts, "rule": blob.get("rule"), "seed": blob.get("seed"), "n_rows": blob.get("n_rows"), "file": p}
    return _PARTS[bb]["parts"]


def mme(backbone: str):
    """The MME capture of a backbone through its own reader (cache2/mme/mme_pool.py; md5-checked once per process)."""
    if backbone not in _MME:
        if MME_DIR not in sys.path:
            sys.path.insert(0, MME_DIR)
        import mme_pool
        _MME[backbone] = mme_pool.load(backbone, verify=True)
    return _MME[backbone]


def _sets(L) -> dict:
    """{part: set of positions in the log's key cache} for the benign parts."""
    k = L.reg["cache"]
    if k not in _SETS:
        parts = benign_parts(L)
        s = {n: set() for n in BEN}
        for j in L.benign_rows:
            s[parts[L.index[j]["sid"]]].add(j)
        _SETS[k] = s
    return _SETS[k]


def source_kc(L, name: str):
    """The key cache that holds the rows of a pool: None (the log's own cache: BEN-*) or the MME capture."""
    return mme(L.backbone).kc if name in MME else None


def rows(L, name: str, which: str | None = None) -> tuple:
    """(rows, which): the rows of a pool that carry the key `which` (None: the pool's default, module docstring)."""
    if name in BEN:
        which = "decode" if which is None else which
        sub = sorted(_sets(L)[name])
        if which == "tf":
            sub = [j for j in sub if L.index[j].get("row_tf") is not None and L.index[j].get("y0_valid")]
        elif which not in ("decode", "steps"):
            raise ValueError("benign parts: which = 'tf' | 'decode' | 'steps'")
        return sub, which
    if name in MME:
        which = "tf" if which is None else which
        P = mme(L.backbone)
        if which == "tf":
            return P.rows(name, valid=True), which
        if which in ("decode", "steps"):
            return P.rows(name), which
        raise ValueError("MME pools: which = 'tf' | 'decode' | 'steps'")
    raise KeyError(f"pool {name!r}: one of {NAMES}")


def get(L, name: str, which: str | None = None) -> Pool:
    """A key-level pool of the log's backbone.  One-hop / BLIP2 / MMKE-Bench logs have none (their caches hold no benign parts)."""
    r, w = rows(L, name, which)
    kc = source_kc(L, name)
    return Pool(name, r, w, kc, L.index if kc is None else kc.index, L.backbone)


def part_of(L, rows_, kc=None) -> list:
    """The pool part of each row (None: not a row of a key-level pool, e.g. a probe or an edit sequence)."""
    if kc is not None:
        return [kc.index[int(j)].get("part") for j in rows_]
    try:
        s = _sets(L)
    except (FileNotFoundError, KeyError):
        return [None] * len(list(rows_))
    lut = {j: n for n, v in s.items() for j in v}
    return [lut.get(int(j)) for j in rows_]


def guard(L, rows_, use: str = "calibration", kc=None, allow_statistics: bool = False):
    """Raise PoolGuardError if a row is a TEST-ONLY row (BEN-T, MME-T) -- `use`: 'calibration' | 'statistics' | 'stream' -- or,
    for use 'calibration' / 'stream', a row of the statistics part BEN-W (allow_statistics=True: BEN-W rows may be used, e.g.
    as null-repair keys of rescope_nullW).  kc: the cache the rows are positions of (None: the log's)."""
    rows_ = [int(j) for j in rows_]
    if not rows_:
        return
    if kc is not None and os.path.realpath(kc.dir) == os.path.realpath(L.kc.dir):
        kc = None
    if kc is None and not any(L.index[j].get("kind") == "benign" for j in rows_):
        return
    parts = part_of(L, rows_, kc)
    bad = [p for p in parts if p in TEST_ONLY or (p in STATISTICS_ONLY and not allow_statistics and use != "statistics")]
    if bad:
        raise PoolGuardError(f"{len(bad)} row(s) of {sorted(set(bad))} used as {use} data: BEN-T and MME-T are test rows and are used for nothing else; "
                             "BEN-W is the statistics part of the feature packages (PROTOCOL.md R2 / R17; calibration sources: REPAIR, BEN-S, MME-C)")


def related(L, pool: Pool) -> np.ndarray:
    """(n,) bool: the row is related to the edit stream or the probes of the log's family by the pool's OWN flags (question string
    or image of an edit / probe of the backbone's long stream, image entity of an edit / probe): such rows are near-scope, not
    benign; reports give the firing rate with and without them (counted)."""
    out = []
    for j in pool.rows:
        e = pool.index[j]
        ov = (e.get("overlap") or {}).get(L.family)
        src = ov if ov else e
        out.append(any(bool(src.get(k)) for k in RELATED))
    return np.array(out, dtype=bool)


def decide(L, M, pool, which: str | None = None, top1: bool = False) -> dict:
    """fx.gate.decide of a method on the rows of a pool (a Pool or a pool name) at the pool's key.  Gates that the key cache
    decides: scoped gates with role 'last' and the router.  The z-score gate and the roles that look at every position need whole
    sequences, which a key-level pool does not hold (NotImplementedError)."""
    from . import gate as G
    P = get(L, pool, which) if isinstance(pool, str) else pool
    d = G.decide(L, M, P.rows, P.which if which is None else which, kc=P.kc, top1=top1)
    d["pool"] = P.name
    return d


def firing(L, M, pool, which: str | None = None, mask=None, rows_out: bool = False) -> dict:
    """False firing of a method on a pool: n, n_fired, fired (% of rows on which an edit is released), split by modality (BEN:
    image / text rows), by category (MME), far / related (rows that the pool's own flags do not / do relate to the family's edits
    or probes), n_edits_capped of the method.  mask: (n,) bool or a predicate on the index row = rows to KEEP (e.g.
    fx.masks.question_disjoint(L)); the number of rows left out is reported.  rows_out=True adds the per-row arrays (rows, fired,
    unit = the source item / image, for bootstraps: rows of one item are one unit)."""
    from .metrics import pct
    P = get(L, pool, which) if isinstance(pool, str) else pool
    d = decide(L, M, P, which)
    f = np.asarray(d["fired"], dtype=bool)
    keep = np.ones(len(P.rows), dtype=bool)
    if mask is not None:
        keep = np.array([bool(mask(P.index[j])) for j in P.rows]) if callable(mask) else np.asarray(mask, dtype=bool)
        if keep.shape != f.shape:
            raise ValueError("mask: one flag per row of the pool")
    rel = related(L, P)
    ix = P.index
    sub = lambda m: {"n": int((keep & m).sum()), "n_fired": int(f[keep & m].sum()), "fired": pct(f[keep & m])}           # noqa: E731
    out = {"pool": P.name, "which": d["which"], **sub(np.ones_like(keep)), "n_masked_out": int((~keep).sum()),
           "far": sub(~rel), "related": sub(rel)}
    pt = np.array([ix[j].get("ptype") for j in P.rows])
    for t in ("image", "text"):
        if (pt == t).any():
            out[t] = sub(pt == t)
    if P.name in MME:
        cat = np.array([ix[j].get("category") for j in P.rows])
        out["by_category"] = {c: sub(cat == c) for c in sorted(set(cat.tolist()))}
    cal = (getattr(M, "rec", None) or {}).get("cal") or {}
    out["n_edits_capped"] = cal.get("n_edits_capped")
    if rows_out:
        out["rows"] = {"rows": list(P.rows), "sid": [ix[j]["sid"] for j in P.rows], "fired": f, "keep": keep, "related": rel,
                       "unit": [ix[j].get("uid") for j in P.rows], "top_edit": np.asarray(d.get("top_edit", np.full(len(f), -1)))}
    return out
