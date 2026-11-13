"""e0_verify: helpers of the independent verification.  NOTHING here imports the package under test (rescope).

Own readers of the cache files and of the testbed logs, and a NAIVE implementation of the calibration rules as PROTOCOL.md
section 6 defines them (plain numpy fp64, no threshold storage: every decision is the defining comparison itself).
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import math
import os
import time
from fractions import Fraction

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

ROOT = f"{_RP.ROOT}"
EXP = ROOT + "/02_research/experiments"
FINAL = EXP + "/10_final"
CACHE = FINAL + "/cache"
VER = FINAL + "/e0_verify"
OUT = VER + "/out"
TESTBED = EXP + "/06_testbed_v3"
LIB = ROOT + "/02_research/code"
SANITY = FINAL + "/e0_sanity/out"
EXT_VIEW = f"{_RP.EXT_VIEW}"      # [renamed: the variable was named after a related project]
VLKEB = EXT_VIEW + "/data/datasets/VLKEB"
VLKEB_IMG = VLKEB + "/mmkb_images"
DEV = {"Q": {"log": "TB-Q100-ae-l2_3", "backbone": "qwen25vl", "family": "Q", "layer": 20, "mem": 10500},
       "L": {"log": "TB-L100-ae-l2_3", "backbone": "llava", "family": "L", "layer": 16, "mem": 17000}}
SPLIT_SEED = 20261109          # harness split of the probe ITEMS into REPAIR / HELD-OUT
TIE = 1e-9                     # relative slack of the guard cap (library convention, docs/USAGE.md section 1, "Guards")
INF = float("inf")


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        x = float(x)
    if isinstance(x, Fraction):
        return f"{x.numerator}/{x.denominator}"
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    if hasattr(x, "item") and not isinstance(x, (str, bytes)) and getattr(x, "ndim", 1) == 0:
        return _clean(x.item())
    return x


def jdump(obj, path, indent=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=indent, allow_nan=False)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------- own readers
def read_index(backbone):
    return jsonl(f"{CACHE}/{backbone}/index.jsonl")


def read_keys(backbone, which):
    """float32 (n, d) tensor of keys_tf.pt | keys_decode.pt | keys_decode_steps.pt (memory-mapped)."""
    import torch
    return torch.load(f"{CACHE}/{backbone}/keys_{which}.pt", map_location="cpu", mmap=True)


def read_log(name):
    """Factors of a testbed log, fp64 numpy: U (T, m), W (T, d), K (T, d) stored solver keys, batch (T,), uids."""
    import torch
    fac = torch.load(f"{TESTBED}/{name}/edit_factors.pt", map_location="cpu")
    met = jload(f"{TESTBED}/{name}/metrics.json")
    T = fac["U"].shape[0]
    batch = np.full(T, -1, dtype=np.int64)
    for b in met["per_batch"]:
        batch[np.asarray(b["edit_ids"])] = int(b["batch"])
    assert batch.min() >= 0
    return {"U": fac["U"].double().numpy(), "W": fac["W"].double().numpy(), "K": fac["K1_solver"].double().numpy(), "batch": batch,
            "uids": list(fac["uids"]), "T": T, "metrics": met}


def stream_items(name):
    """The edit stream of a log as full VLKEB train items (own read of train.json; uid = 'train:<index>')."""
    met = jload(f"{TESTBED}/{name}/metrics.json")
    recs = jsonl(met["config"]["stream"])
    return recs


_VL = {}


def vlkeb(split):
    if split not in _VL:
        raw = jload(f"{VLKEB}/{ {'eval': 'eval.json', 'train': 'train.json'}[split] }")
        _VL[split] = [dict(d, uid=f"{split}:{i}") for i, d in enumerate(raw)]
    return _VL[split]


# ----------------------------------------------------------------------------- naive features (fp64)
def feat_ncoord(keys, W, K):
    """r[n, i] = <W_i, k_n> / <W_i, k_i>  (normalised solver coordinate; SIGNED)."""
    k = np.asarray(keys, dtype=np.float64)
    s = np.einsum("id,id->i", W, K)
    return (k @ W.T) / s[None, :]


def feat_cos(keys, W, K):
    """r[n, i] = cos(k_n, k_i): raw cosine to the stored solver key (SIGNED)."""
    k = np.asarray(keys, dtype=np.float64)
    return (k @ K.T) / (np.linalg.norm(k, axis=1)[:, None] * np.linalg.norm(K, axis=1)[None, :])


FEATURES = {"ncoord": feat_ncoord, "cos": feat_cos}


# ----------------------------------------------------------------------------- naive rules (PROTOCOL.md section 6)
def as_fraction(alpha):
    return alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))


def rank_exact(n, alpha):
    """ceil((1 - alpha)(n + 1)) with exact rationals."""
    return math.ceil((1 - as_fraction(alpha)) * (n + 1))


def draw(n_pool, n, seed):
    """Seeded subset of n of n_pool positions, sorted (the convention of the experiments: numpy default_rng(seed).choice)."""
    return np.sort(np.random.default_rng(seed).choice(n_pool, n, replace=False))


def halves(n, seed, split=(1, 1)):
    """Seeded random split of range(n) into block 1 (n * a // (a + b) queries) and block 2 (the rest)."""
    a, c = split
    n1 = (n * a) // (a + c)
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[:n1]), np.sort(perm[n1:])


def naive_two_block(Q, idx1, idx2, alpha):
    """Q (n, T) features of the calibration queries.  b_i = largest r_i over block 1; eta = the k-th smallest block-2 score
    max_i (r_i - b_i), k = ceil((1 - alpha)(n2 + 1)); +inf if k > n2.  -> (b, eta, k)."""
    b = Q[idx1].max(0)
    sc = np.sort((Q[idx2] - b[None, :]).max(1))
    k = rank_exact(len(idx2), alpha)
    return b, (INF if k > len(idx2) else float(sc[k - 1])), k


def naive_shared(Q, alpha):
    """One shared threshold on the feature: eta = the k-th smallest of max_i r_i over ALL n queries.  -> (b = 0, eta, k)."""
    sc = np.sort(Q.max(1))
    k = rank_exact(Q.shape[0], alpha)
    return np.zeros(Q.shape[1]), (INF if k > Q.shape[0] else float(sc[k - 1])), k


def naive_caps(G, gedit, b, eta):
    """Guard rule: guard key g of edit e must stay a candidate of e.  A guard that the literal rule silences
    (not r_e(g) - b_e > eta) caps e's threshold at r_e(g) - 1e-9 |r_e(g)|.  G (n_g, T) features of the guard keys.
    -> (cap (T,) = +inf for uncapped edits, silenced (n_g,) bool)."""
    own = G[np.arange(G.shape[0]), gedit]
    silenced = ~((own - b[gedit]) > eta)
    cap = np.full(G.shape[1], INF)
    for g in np.nonzero(silenced)[0]:
        cap[gedit[g]] = min(cap[gedit[g]], own[g] - TIE * abs(own[g]))
    return cap, silenced


def naive_decide(F, b, eta, cap=None):
    """F (n, T) features of n keys.  Edit i is a candidate iff r_i - b_i > eta (STRICT), or (capped edit) r_i >= cap_i.
    The key fires iff some edit is a candidate; the releasing edit is the candidate with the largest margin r_i - b_i
    (lowest index on an exact tie).  -> (fired (n,), top (n,) = -1 where nothing fires, cand (n, T))."""
    M = F - b[None, :]
    cand = M > eta
    if cap is not None:
        cand = cand | (F >= cap[None, :])
    fired = cand.any(1)
    top = np.where(cand, M, -INF).argmax(1)
    return fired, np.where(fired, top, -1), cand


# ----------------------------------------------------------------------------- pools from the raw data (own code)
def split_pools(uids, seed=SPLIT_SEED):
    """Harness split (07_repair_v3/harness/sampler.py, re-implemented): the sorted distinct probe uids are permuted with
    numpy default_rng(seed); the first half is the REPAIR pool."""
    u = sorted(set(uids))
    perm = np.random.default_rng(seed).permutation(len(u))
    rep = {u[i] for i in perm[:len(u) // 2]}
    return {x: ("repair" if x in rep else "heldout") for x in u}


def annotate(rows):
    """rows: dicts with uid, ptype, prompt, image (probe rows in file order).  Adds my_pool, my_disjoint, my_strict:
    disjoint = HELD-OUT row whose query (ptype, prompt, image) is not a REPAIR query; strict = disjoint and the question
    text (ptype, prompt) does not occur in the REPAIR pool."""
    pool = split_pools([r["uid"] for r in rows])
    repq = {(r["ptype"], r["prompt"], r["image"]) for r in rows if pool[r["uid"]] == "repair"}
    repquest = {(r["ptype"], r["prompt"]) for r in rows if pool[r["uid"]] == "repair"}
    out = []
    for r in rows:
        p = pool[r["uid"]]
        dj = p == "heldout" and (r["ptype"], r["prompt"], r["image"]) not in repq
        out.append({"pool": p, "disjoint": dj, "strict": dj and (r["ptype"], r["prompt"]) not in repquest})
    return out
