"""Shared helpers of the tests of the recombination layer: paths, a synthetic log with a fake recombination store, a naive reference
of the two-block rule, result bookkeeping.  No test writes outside feat2/recomb/tests/out and (temporary) cache2/recomb/work."""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True
os.environ.setdefault("RESCOPE_CACHE", f"{_RP.ROOT}/02_research/experiments/10_final/cache2/recomb/work/hcache")
HERE = os.path.dirname(os.path.abspath(__file__))
FEAT = os.path.dirname(HERE)                       # .../feat2/recomb
FEAT2 = os.path.dirname(FEAT)                      # .../feat2
FINAL = os.path.dirname(FEAT2)
OUT = os.path.join(HERE, "out")
WORK = os.path.join(FINAL, "cache2", "recomb", "work")
LIB = f"{_RP.ROOT}/02_research/code"
for p in (LIB, FEAT2):
    if p not in sys.path:
        sys.path.insert(0, p)
os.makedirs(OUT, exist_ok=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
DEV = {"qwen25vl": "TB-Q100-ae-l2_3", "llava": "TB-L100-ae-l2_3"}       # the ONLY real logs a test may run a gate on


class Results:
    """One JSON file per test group: tests/out/<group>.json = {test: {ok, seconds, info | error}}."""

    def __init__(self, group: str):
        self.group, self.res = group, {}

    def run(self, name, fn, *a, **k):
        t0 = time.time()
        try:
            info = fn(*a, **k)
            self.res[name] = {"ok": True, "seconds": round(time.time() - t0, 2), "info": info}
            print(f"  [ok]   {self.group}.{name} ({time.time() - t0:.1f} s): {json.dumps(info, default=str)[:300]}", flush=True)
        except Exception as e:                      # a failing test is recorded, the others still run
            self.res[name] = {"ok": False, "seconds": round(time.time() - t0, 2), "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            print(f"  [FAIL] {self.group}.{name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    def save(self) -> bool:
        with open(os.path.join(OUT, f"{self.group}.json"), "w") as f:
            json.dump({"group": self.group, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "tests": self.res}, f, indent=1, default=str)
        return all(r["ok"] for r in self.res.values())


def check(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "check failed")


# ----------------------------------------------------------------------------- a synthetic log + fake recombination store
def synthetic(T=40, d=96, seed=0, backbone="qwen25vl"):
    """(EditLog with meta uids / backbone, uids) of rescope.synthetic_log."""
    import rescope
    log, _ = rescope.synthetic_log(T=T, d=d, m=20, batch=10 if T >= 10 else T, seed=seed)
    uids = [f"syn:{seed}:{i}" for i in range(T)]
    log.meta.update(uids=uids, backbone=backbone)
    return log, uids


def fake_store(root, log, uids, backbone="qwen25vl", seed=0, T_stream=None, noise=0.6):
    """Write <root>/<backbone>/{rows.jsonl, keys.pt} for a synthetic log: 4 rows of every kind per edit of a stream that starts with the
    log's edits (T_stream >= T: further stream positions get further uids), X-X partners by the rule (batch of 10 below position 100, group
    of 100).  Keys = the owner's solver key + noise (so that own coordinates are large) as float32.  -> the rows."""
    import recomb
    T, d = log.T, log.d
    Ts = T if T_stream is None else int(T_stream)
    stream = list(uids) + [f"syn:later:{i}" for i in range(T, Ts)]
    g = np.random.default_rng(seed)
    rows, keys = [], []
    for i in range(Ts):
        base = log.K[i % T].numpy()
        for kind in recomb.KINDS:
            for k in range(recomb.M):
                j = None
                if kind == "X-X":
                    lo, n = (10 * (i // 10), 10) if (i < 100 and k < 2) else (100 * (i // 100), 100)
                    cand = [c for c in range(lo, min(lo + n, Ts)) if c != i]
                    j = int(cand[int(g.integers(len(cand)))])
                rows.append({"rid": f"{kind}:{stream[i]}:{k}", "kind": kind, "family": "S", "edit_uid": stream[i], "edit_pos": i, "edit": i, "k": k,
                             "edit_b_uid": None if j is None else stream[j], "edit_b_pos": j, "edit_b": j, "min_T": (i if j is None else max(i, j)) + 1})
                keys.append((base + noise * g.standard_normal(d) * np.abs(base).mean()).astype(np.float32))
    ddir = os.path.join(root, backbone)
    os.makedirs(ddir, exist_ok=True)
    with open(os.path.join(ddir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    torch.save({"rids": [r["rid"] for r in rows], "K_last": torch.from_numpy(np.stack(keys)), "last_prompt_idx": [7] * len(rows), "layer": 0,
                "backbone": backbone, "doc": "fake"}, os.path.join(ddir, "keys.pt"))
    return rows


def gate_with_guards(log, feature="ncoord", seed=0, n_extra=1):
    """Un-thresholded gate of the library with additional guard keys (noisy own keys, as the B = 1 own-prompt keys)."""
    import rescope
    S = rescope.feature_gate(log, feature)
    g = torch.Generator().manual_seed(seed)
    K = log.K + 0.02 * log.K.abs().mean() * torch.randn(log.K.shape, generator=g, dtype=torch.float64)
    S, _ = rescope.add_guards(S, K, list(range(log.T)))
    return S


def benign(log, n=200, seed=1, scale=1.0):
    """n benign keys (n, d): the common direction of the synthetic keys + independent noise."""
    g = torch.Generator().manual_seed(seed)
    mu = log.K.mean(0)
    return mu[None, :] * scale + torch.randn(n, log.d, generator=g, dtype=torch.float64) * log.K.std(0)[None, :]


# ----------------------------------------------------------------------------- naive reference of the rule (plain loops; no library rule)
def naive_rule(Q, x, alpha_num, alpha_den, idx1, idx2, u=None):
    """Thresholds of the two-block rule with baseline max(block-1 maximum, x), written out with plain Python loops:
    -> (b (D,), eta, tau (D,)) with tau_j = the smallest fp64 number r such that fl(fl(r - b_j) / u_j) > eta."""
    Q = np.asarray(Q, dtype=np.float64)
    D = Q.shape[1]
    b = np.empty(D)
    for j in range(D):
        m = max(Q[q, j] for q in idx1)
        b[j] = m if (x is None or not x[j] > m) else x[j]
    uu = np.ones(D) if u is None else np.asarray(u, dtype=np.float64)
    sc = sorted(max((Q[q, j] - b[j]) / uu[j] for j in range(D)) for q in idx2)
    n2 = len(sc)
    k = -((-(alpha_den - alpha_num) * (n2 + 1)) // alpha_den)           # ceil((1 - alpha)(n2 + 1)), integers
    eta = float("inf") if k > n2 else sc[k - 1]
    tau = np.full(D, np.inf)
    if eta != float("inf"):
        for j in range(D):
            fires = lambda r: (r - b[j]) / uu[j] > eta                  # noqa: E731
            r = np.float64(b[j] + eta * uu[j])
            for _ in range(10000):                                      # walk down while the predecessor still fires
                p = np.nextafter(r, -np.inf)
                if not fires(p):
                    break
                r = p
            for _ in range(10000):                                      # walk up until r itself fires
                if fires(r):
                    break
                r = np.nextafter(r, np.inf)
            assert fires(r) and not fires(np.nextafter(r, -np.inf))
            tau[j] = r
    return b, eta, tau
