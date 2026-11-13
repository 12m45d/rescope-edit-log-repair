"""Extra CPU check (not part of the fingerprinted suite): the per-backbone key cache against the keys that the exploratory
experiment P11b captured independently (08_rescope/p11_scopecut/out/<log>/keys: last-prompt key of every B = 1 teacher-forced
sequence of the two development logs, float32).

    python tests/extra/cache_vs_p11.py          (cache root: 10_final/cache, or RESCOPE_KEYCACHE_ROOT)

Per backbone: how many P11 sequences the cache holds (matched through the cache's `lib_sid` among KeyCache.log_rows of the
log), whether the cached teacher-forced keys are bitwise the P11 keys, whether the last prompt index agrees, and whether the
CPU decision of a reference gate (thresholds 0.35) from the cache equals the decision from P11's keys; and the CPU decisions of
three gates (coordinate threshold 0.35, log-only POOL(1), cosine 0.95) against the firing that P11a OBSERVED in its forwards
(edit sequences, valid probe rows, hard negatives).
Writes tests/out/extra_cache_vs_p11.json.  PASS iff every cache that exists holds at least one P11 sequence and all of its
overlapping keys are bitwise equal with equal decisions."""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
import run_tests  # noqa: E402
from rescope import baselines as BL, calibrate as CA, keycache as KC, loglib as LL  # noqa: E402
from rescope.scoped import ScopedLog  # noqa: E402

P11 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p11_scopecut/out"
res = {"test": "extra_cache_vs_p11", "code": run_tests.CODE, "cache_root": KC.CACHE_ROOT, "cache_root_is_final_location": KC.CACHE_ROOT == KC.DEFAULT_ROOT,
       "time": time.strftime("%Y-%m-%d %H:%M:%S"), "backbones": {}}
for bb, name in (("qwen25vl", "TB-Q100-ae-l2_3"), ("llava", "TB-L100-ae-l2_3")):
    if not KC.KeyCache.available(bb):
        res["backbones"][bb] = {"available": False}
        continue
    kc = KC.KeyCache(bb)
    with open(f"{P11}/{name}/keys/meta.json") as f:
        seqs = json.load(f)["seqs"]
    K = torch.from_numpy(np.load(f"{P11}/{name}/keys/K_last.npy"))
    log = LL.from_testbed(name)
    by = {kc.index[j]["lib_sid"]: j for j in kc.log_rows(log.meta["uids"]) if kc.index[j].get("lib_sid") and kc.index[j].get("row_tf") is not None}
    both = [q for q in seqs if q["sid"] in by]
    rows = [by[q["sid"]] for q in both]
    kk = kc.keys("tf", rows) if rows else torch.zeros(0, K.shape[1])
    ref = K[[q["n"] for q in both]]
    S = ScopedLog.from_log(log, tau=0.35)
    dec = kc.decide(S, "tf", rows=rows)
    fire_ref = (S.key_scores(ref.double()) >= S.theta()[None, :]).any(1).numpy() if rows else np.zeros(0, dtype=bool)
    kinds = {}
    for q in both:
        k = q["kind"] + ":" + str(q.get("metric", q.get("ptype")))
        kinds[k] = kinds.get(k, 0) + 1
    r = {"available": True, "is_final": kc.is_final, "dir": kc.dir, "log": name, "cache_complete": kc.meta.get("complete"), "n_cache_sequences": kc.n, "n_p11_sequences": len(seqs), "n_in_both": len(both),
         "by_kind": kinds, "keys_bitwise_equal": int(sum(bool(torch.equal(a, b)) for a, b in zip(kk, ref))),
         "max_abs_diff": float((kk - ref).abs().max()) if rows else None,
         "last_prompt_index_equal": int(sum(kc.index[j]["lp_tf"] == q["last"] for j, q in zip(rows, both))),
         "decisions_equal": int((dec["fired"] == fire_ref).sum()), "n_fired": int(fire_ref.sum()), "n_near": int(dec["near"].sum())}
    # CPU decisions from the cache against the firing OBSERVED in real forwards by the exploratory experiment P11a (three gates:
    # coordinate threshold 0.35, log-only POOL(1) thresholds, cosine 0.95; edit sequences, probe rows, hard negatives)
    R11 = f"{_RP.ROOT}/02_research/experiments/08_rescope/p11_retrofit/out"
    with open(f"{R11}/{name}/C/_meta_main.json") as f:
        m11 = json.load(f)["meta"]
    cid = {(kc.index[j].get("lib_sid") or kc.index[j]["sid"]): j for j in kc.log_rows(log.meta["uids"]) if kc.index[j].get("row_tf") is not None}
    for j in kc.log_rows(log.meta["uids"]):                              # hard negatives are addressed by their id
        if kc.index[j].get("hn_id") and kc.index[j].get("row_tf") is not None:
            cid[kc.index[j]["hn_id"]] = j
    gates = {"R1-REL0.35-REPL": S, "R1-POOL1-REPL": CA.calibrate_log_only(ScopedLog.from_log(log), 1.0)[0], "NDB-R1-0.95": BL.cosine_gate(log, 0.95)}
    obs = {}
    for cfg, g in gates.items():
        with open(f"{R11}/{name}/C/{cfg}.json") as f:
            rows11 = json.load(f)["rows"]
        pairs = [(cid[m["sid"]], rows11[k]["nfp"] > 0) for k, m in enumerate(m11) if m["sid"] in cid]
        d2 = kc.decide(g, "tf", rows=[a for a, _ in pairs])
        o = np.array([b for _, b in pairs])
        obs[cfg] = {"n_p11_rows": len(m11), "n_in_cache": len(pairs), "equal": int((d2["fired"] == o).sum()), "n_fired_observed": int(o.sum()), "n_near": int(d2["near"].sum())}
    r["observed_in_forwards"] = obs
    r["ok"] = bool(r["n_in_both"] > 0 and r["keys_bitwise_equal"] == r["n_in_both"] == r["last_prompt_index_equal"] == r["decisions_equal"]
                   and all(v["equal"] == v["n_in_cache"] > 0 for v in obs.values()))
    res["backbones"][bb] = r
res["pass"] = bool(any(v["available"] for v in res["backbones"].values()) and all(v.get("ok", True) for v in res["backbones"].values()))
run_tests.jdump(res, os.path.join(run_tests.OUT, "extra_cache_vs_p11.json"))
print(json.dumps(res, indent=1))
