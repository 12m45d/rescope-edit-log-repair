"""Independent verification of E5 (linear part): shared readers.  CPU only, fp64 linear algebra.

Nothing of e5_linear/code is imported here.  The inputs are read from their primary files:
  factors      06_testbed_v3/<log>/edit_factors.pt        U (T, d_out) fp64 write vectors, W (T, d_in) fp64 read-out vectors psi_i
  complaints   06_testbed_v3/<log>/probe_records_v2.jsonl rows with is_complaint_D1 and y0_valid whose probe item is in the REPAIR
               pool; the pool split is re-implemented here (items = probe uids, numpy default_rng(20261109).permutation of the
               sorted uids, first half = REPAIR) and compared with 07_repair_v3/harness/sampler.py (loaded by path, read only)
  sample       60 complaints, 30 image + 30 text, numpy default_rng(SAMPLE_SEED): per probe type, choice without replacement
               among the candidates sorted by probe index (the harness rule sample_complaints(H, 60, seed = 0), re-implemented)
  last key     10_final/cache/<backbone>/keys_tf.pt, row `row_tf` of the probe row in index.jsonl: the module input at the last
               prompt position of the teacher-forced pre-edit answer (float32 image of the module-dtype value).  This is NOT the
               file that the measuring agent used (07_repair_v3/rollback_opt/cache/keys_<log>_v2.pt, row 0 of K); the two
               captures are compared bit for bit.
  answer keys  (only for the 'stacked' extra and the stacked test of the capacity) the Track C file above: the only cache
               that holds the keys of the answer-predicting positions.
Signs: coordinates <psi_i, k> and inner products <w_i, w_j> are SIGNED; norms are magnitudes.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.abspath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys

sys.dont_write_bytecode = True                      # nothing is written next to read-only sources
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import hashlib
import importlib.util
import json
import math
import time
import types

import numpy as np
import torch

ROOT = f"{_RP.ROOT}"
EXP = ROOT + "/02_research/experiments"
TB = EXP + "/06_testbed_v3"
KEYCACHE = EXP + "/10_final/cache"
TRACKC = EXP + "/07_repair_v3/rollback_opt/cache"
HARNESS = EXP + "/07_repair_v3/harness"
E5 = EXP + "/10_final/e5_linear"
REPORT_OUT = E5 + "/out"                            # the measuring agent's files (read only)
VER = E5 + "/verify"
OUT = VER + "/out"
LOGDIR = VER + "/logs"
for _d in (OUT, LOGDIR):
    os.makedirs(_d, exist_ok=True)

LOGS = {"TB-Q100-ae-l2_3": {"backbone": "qwen25vl", "family": "Q", "short": "Q", "layer": 20,
                            "editkeys": "editkeys_qwen25vl_L20_c8054748c923.pt"},
        "TB-L100-ae-l2_3": {"backbone": "llava", "family": "L", "short": "L", "layer": 16,
                            "editkeys": "editkeys_llava_L16_4c7ff92dd9a5.pt"}}
SPLIT_SEED = 20261109                               # harness split of the probe items
SAMPLE_SEED = 0                                     # harness default seed of sample_complaints
N_SAMPLE = 60
TOLS = (0.1, 0.3, 0.5)                              # 0.1 and 0.5 are the task; 0.3 is reported as an extra
PTYPES = ("image", "text")
EPS = float(np.finfo(np.float64).eps)
CAP_SEED = 5                                        # seed of the five complaints of the capacity check (drawn from the 60)
N_CAP = 5
TOL_REL = 1e-9                                      # harness COMP cut: singular value of (I - P) K_q > TOL_REL * max ||k_q||


def log(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, torch.Tensor):
        return clean(o.tolist())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return o


def jdump(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(clean(obj), f, indent=1)
    os.replace(tmp, path)


def jload(path):
    with open(path) as f:
        return json.load(f)


def jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(np.asarray(v, dtype=np.float64))) if len(v) else None


def quart(v):
    a = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    return {"n": int(a.size), "min": float(a.min()), "q25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "q75": float(np.percentile(a, 75)), "max": float(a.max())}


# ----------------------------------------------------------------------------- the log
def factors(name):
    """U (T, d_out), Psi (T, d_in) as fp64 numpy arrays, uids, and the batch of every edit (metrics.json)."""
    fac = torch.load(os.path.join(TB, name, "edit_factors.pt"), map_location="cpu")
    if fac["U"].dtype != torch.float64 or fac["W"].dtype != torch.float64:
        raise RuntimeError("factors are not stored in fp64")
    met = jload(os.path.join(TB, name, "metrics.json"))
    T = fac["U"].shape[0]
    batch = np.full(T, -1)
    for b in met["per_batch"]:
        batch[np.asarray(b["edit_ids"])] = int(b["batch"])
    if batch.min() < 0:
        raise RuntimeError("an edit without a batch")
    return {"U": fac["U"].numpy().copy(), "Psi": fac["W"].numpy().copy(), "uids": list(fac["uids"]), "batch": batch,
            "dtype": met.get("dtype"), "config": met["config"]}


def composite_delta(name):
    """W_E - W_0 as stored (fp32), cast to fp64: the applied weight difference, write round-off included."""
    return torch.load(os.path.join(TB, name, "composite_delta.pt"), map_location="cpu").double().numpy()


# ----------------------------------------------------------------------------- complaints
def probe_rows(name):
    return jsonl(os.path.join(TB, name, "probe_records_v2.jsonl"))


def repair_uids(rows):
    """My own implementation of the harness split: REPAIR pool = the first half of a seeded permutation of the sorted uids."""
    uids = sorted({r["uid"] for r in rows})
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(uids))
    return {uids[int(i)] for i in perm[: len(uids) // 2]}


def candidates(rows, rep=None):
    """{ptype: [row, ...]} of the REPAIR-pool D1 complaints (decoder v2, y0 valid), sorted by probe index."""
    rep = repair_uids(rows) if rep is None else rep
    out = {}
    for pt in PTYPES:
        out[pt] = sorted([r for r in rows if r["is_complaint_D1"] and r["y0_valid"] and r["uid"] in rep and r["ptype"] == pt],
                         key=lambda r: r["probe_idx"])
    return out


def sample60(cand, n=N_SAMPLE, seed=SAMPLE_SEED):
    """n / 2 image + n / 2 text complaints; one generator, image first (the harness rule)."""
    rng = np.random.default_rng(seed)
    k = {"image": n // 2, "text": n - n // 2}
    out = []
    for pt in PTYPES:
        idx = rng.choice(len(cand[pt]), k[pt], replace=False)
        out += [cand[pt][int(i)] for i in idx]
    return out


def harness_check(name, rows, cand, samp):
    """The same pool and the same sample from the harness sampler itself (07_repair_v3/harness/sampler.py, by path)."""
    spec = importlib.util.spec_from_file_location("v_harness_sampler", os.path.join(HARNESS, "sampler.py"))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    uid = {}
    for r in rows:
        uid.setdefault(r["probe_idx"], r["uid"])
    H = types.SimpleNamespace(rows=rows, probes=[{"uid": uid[i]} for i in range(len(uid))], _setup={}, name=name)
    hc = {pt: sorted(S.candidates(H, "D1", "repair", pt), key=lambda r: r["probe_idx"]) for pt in PTYPES}
    hs = S.sample_complaints(H, N_SAMPLE, seed=SAMPLE_SEED, kind="D1", pool="repair")
    return {"candidates_equal": all([r["group"] for r in hc[pt]] == [r["group"] for r in cand[pt]] for pt in PTYPES),
            "sample_equal": [r["group"] for r in hs] == [r["group"] for r in samp],
            "harness_split_seed": int(S.SPLIT_SEED)}


# ----------------------------------------------------------------------------- keys
def keycache_index(name):
    """{(probe_idx, ptype): index entry} of the probe rows of the log's family in the final key cache."""
    cfg = LOGS[name]
    out = {}
    with open(os.path.join(KEYCACHE, cfg["backbone"], "index.jsonl")) as f:
        for line in f:
            if '"kind": "probe"' not in line:
                continue
            e = json.loads(line)
            if e["kind"] == "probe" and e["family"] == cfg["family"]:
                out[(e["probe_idx"], e["ptype"])] = e
    return out


def keycache_edit_index(name, T):
    """{edit position: index entry} of the 'rel' sequences (the edit's own request) of the first T edits of the family."""
    cfg = LOGS[name]
    out = {}
    with open(os.path.join(KEYCACHE, cfg["backbone"], "index.jsonl")) as f:
        for line in f:
            if '"kind": "rel"' not in line:
                continue
            e = json.loads(line)
            if e["kind"] == "rel" and e["family"] == cfg["family"] and e["edit"] < T:
                out[int(e["edit"])] = e
    return out


def keys_tf(name):
    return torch.load(os.path.join(KEYCACHE, LOGS[name]["backbone"], "keys_tf.pt"), map_location="cpu")


def trackc_keys(name):
    """{(probe_idx, ptype): row} of the Track C capture (K (A, d_in) float32, row 0 = the last prompt position)."""
    d = torch.load(os.path.join(TRACKC, f"keys_{name}_v2.pt"), map_location="cpu")
    if d["log"] != name:
        raise RuntimeError("key file of another log")
    return {(r["probe_idx"], r["ptype"]): r for r in d["rows"]}


def last_keys(name, comps, idx=None, Ktf=None, tc=None):
    """Last prompt keys of the complaints from the final key cache, (n, d_in) float32, with the checks against the
    probe record (uid, group, answer text, pool) and, bit for bit, against the Track C capture."""
    idx = keycache_index(name) if idx is None else idx
    Ktf = keys_tf(name) if Ktf is None else Ktf
    tc = trackc_keys(name) if tc is None else tc
    K, info = [], {"n": len(comps), "n_same_uid_group_answer": 0, "n_pool_repair": 0, "n_y0_valid": 0, "n_bitwise_equal_trackc": 0,
                   "n_lp_shift": 0, "lp_shift_groups": [], "n_A_equal": 0, "max_abs_diff_trackc": 0.0}
    for c in comps:
        e = idx[(c["probe_idx"], c["ptype"])]
        if e["row_tf"] is None:
            raise RuntimeError(f"{c['group']}: no teacher-forced key in the cache")
        k = Ktf[int(e["row_tf"])]
        info["n_same_uid_group_answer"] += int(e["uid"] == c["uid"] and e["group"] == c["group"] and e["y0_text"] == c["y0_text"]
                                               and e["answer"] == c["y0_text"] and e["prompt"] == c["prompt"])
        info["n_pool_repair"] += int(e["pool"] == "repair")
        info["n_y0_valid"] += int(bool(e["y0_valid"]))
        r = tc[(c["probe_idx"], c["ptype"])]
        info["n_bitwise_equal_trackc"] += int(torch.equal(k, r["K"][0]))
        info["max_abs_diff_trackc"] = max(info["max_abs_diff_trackc"], float((k.double() - r["K"][0].double()).abs().max()))
        info["n_A_equal"] += int(int(r["A"]) == int(c["n_ans_tok"]) == int(r["K"].shape[0]) and r["uid"] == c["uid"]
                                 and int(r["pred_pos"][0]) == int(r["last_prompt_idx"]))
        if e.get("lp_shift"):
            info["n_lp_shift"] += 1
            info["lp_shift_groups"].append(c["group"])
        K.append(k)
    return torch.stack(K), info
