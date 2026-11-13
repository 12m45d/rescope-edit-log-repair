"""Seeded probe-pool split and complaint sampler (random, NOT difficulty-filtered).

Pools: the 500 VLKEB eval probe items of a log are split once, by item, with a fixed seed
(SPLIT_SEED) into a REPAIR pool (complaints are sampled here; COMP's known-good probe keys come
from here) and a HELD-OUT pool (only used to measure collateral).  Both probe rows of an item
(image and text) go to the same pool.  The split depends only on the probe uids, so it is the
same for every log of a backbone family.

Complaints: rows of probe_records_v2.jsonl (decoder v2) with
  kind "D1": is_complaint_D1 (y0 valid: non-empty and reproduced by W_0; not reproduced by W_E)
  kind "D2": is_complaint_D2 (pre-edit 48-token answer contains the reference, post-edit not)
stratified image/text (n/2 each), drawn with numpy default_rng(seed) without replacement.
"""
from __future__ import annotations

import numpy as np

SPLIT_SEED = 20261109


def split_pools(H, seed: int = SPLIT_SEED, frac: float = 0.5):
    """frac = fraction of probe ITEMS in the REPAIR pool (default 0.5: the harness split, unchanged).
    For a given seed the REPAIR pools are nested: the first round(frac * n) items of one permutation."""
    uids = sorted({p["uid"] for p in H.probes})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uids))
    n_rep = len(uids) // 2 if frac == 0.5 else int(round(len(uids) * frac))
    rep = {uids[i] for i in perm[:n_rep]}
    pool = {}
    for pi, p in enumerate(H.probes):
        pool[pi] = "repair" if p["uid"] in rep else "heldout"
    return pool


def pool_of(H):
    if "pool" not in H._setup:
        H._setup["pool"] = split_pools(H)
    return H._setup["pool"]


def split_tag(H):
    """None for the default split; otherwise (frac, seed) (used in cache tags)."""
    sp = H._setup.get("split")
    if sp is None or (sp["frac"] == 0.5 and sp["seed"] == SPLIT_SEED):
        return None
    return (float(sp["frac"]), int(sp["seed"]))


def set_split(H, frac: float = 0.5, seed: int = SPLIT_SEED):
    """(RQ4 option, backwards compatible) re-split the probe items of H into REPAIR / HELD-OUT and
    drop every cached object that depends on the pools (held-out eval cache, W_E baseline,
    protection bases).  set_split(H) restores the default harness split."""
    H._setup["pool"] = split_pools(H, seed, frac)
    H._setup["split"] = {"frac": float(frac), "seed": int(seed)}
    for k in list(H._setup):
        if k in ("evalcache", "baseline") or (isinstance(k, tuple) and k and k[0] == "basis"):
            del H._setup[k]
    return H._setup["pool"]


def candidates(H, kind="D1", pool="repair", ptype=None):
    pl = pool_of(H)
    key = {"D1": "is_complaint_D1", "D2": "is_complaint_D2"}[kind]
    return [r for r in H.rows if r[key] and (pool is None or pl[r["probe_idx"]] == pool)
            and (ptype is None or r["ptype"] == ptype)
            and r["y0_valid"]]


def sample_complaints(H, n: int, seed: int = 0, kind: str = "D1", pool: str = "repair",
                      stratify: bool = True):
    """n complaints (n/2 image + n/2 text if stratify), deterministic given (log, n, seed, kind)."""
    rng = np.random.default_rng(seed)
    out = []
    if stratify:
        k = {"image": n // 2, "text": n - n // 2}
        for pt in ("image", "text"):
            pool_rows = sorted(candidates(H, kind, pool, pt), key=lambda r: r["probe_idx"])
            if k[pt] > len(pool_rows):
                raise ValueError(f"only {len(pool_rows)} {kind} {pt} complaints in pool {pool}")
            idx = rng.choice(len(pool_rows), k[pt], replace=False)
            out += [pool_rows[int(i)] for i in idx]
    else:
        rows = sorted(candidates(H, kind, pool), key=lambda r: (r["probe_idx"], r["ptype"]))
        out = [rows[int(i)] for i in rng.choice(len(rows), n, replace=False)]
    return [dict(r, complaint_kind=kind, sample_seed=seed) for r in out]
