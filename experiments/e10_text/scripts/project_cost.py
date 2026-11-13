"""Cost of the E10 row set on a T = 2000 log: measurements on the development log and on SYNTHETIC logs of the same shape
(the development log's factors tiled with 2 % noise: realistic sizes and arithmetic, no reported log is touched), and the
projection.  GPU job:

    $GQ run --mem 14000 --prio -8 --name e10t_cost -- $PY scripts/project_cost.py        -> out/cost_projection.json
    python scripts/make_usage.py                                                          -> USAGE.txt (renders the projection)

Measured: seconds per sequence pass (teacher-forced and decode) for a weight, and for a gate with T detectors where nothing
fires (the hook's scoring overhead) and where it fires; construction time of every calibrated row (guards, features of the
calibration draw, calibration, null repair); gate-level evaluation time per position x detector (CPU and GPU); public
probability test per record; public generation per record; peak GPU memory.
"""
import copy
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import e10t  # noqa: E402

B = e10t.Backbone()
import fx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rescope.loglib import EditLog  # noqa: E402

INF = float("inf")
L = e10t.open_log(e10t.DEV_LOG)
OUTP = os.path.join(e10t.OUT, "cost_projection.json")
say = lambda *a: print(f"[{time.strftime('%H:%M:%S', time.gmtime())}]", *a, flush=True)   # noqa: E731


def synthetic(T, seed=0):
    """A text log of T updates with the shape of a real one: the development log's factors tiled, copies perturbed by 2 % noise."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(T) % L.T
    log = L.log
    noise = lambda X: X + (torch.arange(T) >= L.T)[:, None] * 0.02 * X.norm(dim=1, keepdim=True) / X.shape[1] ** 0.5 * torch.randn(X.shape, generator=g, dtype=torch.float64)  # noqa: E731
    K, W = noise(log.K[idx]), noise(log.W[idx])
    W = W / (W * K).sum(1, keepdim=True)
    S = copy.copy(L)
    S.log = EditLog(log.U[idx], W, K, torch.arange(T), name=f"SYN-T{T}", meta={})
    S.T, S.name = T, f"SYN-T{T}"
    S.rows = {k: [dict(r, pos=r["pos"] + c * L.T) for c in range(T // L.T) for r in v] for k, v in L.rows.items()}
    S.solver_pos = L.solver_pos * (T // L.T)
    return S


def sync():
    torch.cuda.synchronize()


def time_passes(obj, tids, reps=1):
    """Seconds per sequence pass (teacher-forced, decode) of a served object on fixed padded batches."""
    lens = [len(L.ids(t)) for t in tids]
    bl = B.batches(lens)
    out = {}
    with B.served(obj):
        for mode in ("tf", "dec"):
            for b in bl[:1]:                                            # warm-up
                B.tf([L.ids(tids[i]) for i in b], [L.y0(tids[i]) for i in b]) if mode == "tf" else B.decode([L.ids(tids[i]) for i in b])
            sync()
            t0 = time.time()
            for _ in range(reps):
                for b in bl:
                    seqs = [L.ids(tids[i]) for i in b]
                    B.tf(seqs, [L.y0(tids[i]) for i in b]) if mode == "tf" else B.decode(seqs)
            sync()
            out[mode] = (time.time() - t0) / (reps * len(tids))
    out["n_sequences"], out["n_batches"] = len(tids), len(bl)
    return out


res = {"code": e10t.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()), "gpu": torch.cuda.get_device_name(0),
       "note": "timings on a shared GPU are load-dependent; synthetic logs = the development log's factors tiled with 2 % noise"}
rng = np.random.default_rng(0)
ho = L.benign("heldout")
tids_b = [ho[int(i)]["text_id"] for i in np.sort(rng.choice(len(ho), 600, replace=False))]
tids_r = [r["text_id"] for r in L.rows["request"]] + [r["text_id"] for r in L.rows["paraphrase"]]
res["passes"] = {"weight": {"benign": time_passes(L.weights()[1], tids_b), "request_paraphrase": time_passes(L.weights()[1], tids_r)}}
say("weight", res["passes"]["weight"])
res["build"], res["gate_level"], res["mem"] = {}, {}, {}
for T in (100, 500, 2000):
    SL = L if T == L.T else synthetic(T)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rows = {}
    for lab in ("rescope", "rescope_shared", "rescope_null", "cosine", "cosine_published", "exact"):
        t1 = time.time()
        M = e10t.methods.build(lab, SL) if lab in ("cosine_published", "exact") else e10t.methods.build(lab, SL, draw=0)
        rows[lab] = M
        res["build"].setdefault(f"T{T}", {})[lab] = time.time() - t1
    # a second draw: the base gate (values, guards) is reused
    t1 = time.time()
    e10t.methods.rescope(SL, draw=1)
    res["build"][f"T{T}"]["rescope_next_draw"] = time.time() - t1
    t1 = time.time()
    e10t.methods.rescope_null(SL, draw=1)
    res["build"][f"T{T}"]["rescope_null_next_draw"] = time.time() - t1
    say(f"T = {T}: rows built", {k: round(v, 1) for k, v in res["build"][f"T{T}"].items()})
    # gate level: features of a fresh detector set over 300 pool texts, CPU and GPU
    S0 = rows["rescope_null"].obj
    tids = tids_b[:300]
    npos = int(sum(L.kc.n_pos(t) for t in tids))
    gl = {"n_texts": len(tids), "n_positions": npos}
    for devn in ("cpu", "cuda"):
        e10t.gate.clear_cache()
        if devn == "cuda":
            e10t.gate.features(SL, S0, tids[:8], devn)
            e10t.gate.clear_cache()
            sync()
        t1 = time.time()
        e10t.gate.features(SL, S0, tids, devn)
        sync()
        gl[f"seconds_{devn}"] = time.time() - t1
        gl[f"ns_per_position_detector_{devn}"] = 1e9 * gl[f"seconds_{devn}"] / (npos * T)
    t1 = time.time()
    d = e10t.gate.decide(SL, rows["rescope"].obj, [r["text_id"] for r in L.rows["request"]], [r["pos"] for r in L.rows["request"]])
    gl["decide_100_request_prompts_seconds"] = time.time() - t1
    res["gate_level"][f"T{T}"] = gl
    say(f"T = {T}: gate level", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in gl.items()})
    # passes through the hook
    g = rows["rescope"].obj
    off = g.replace(det_tau=torch.full_like(g.det_tau, INF))
    cg = rows["cosine"].obj
    P = {"silent_hook": {"benign": time_passes(off, tids_b)}, "rescope": {"benign": time_passes(g, tids_b), "request_paraphrase": time_passes(g, tids_r)},
         "cosine_silent_hook": {"benign": time_passes(cg.replace(det_tau=torch.full_like(cg.det_tau, INF)), tids_b)},
         "where_all_fire": {"request_paraphrase": time_passes(e10t.methods.where(SL, "none").obj, tids_r[:150])}}
    res["passes"][f"gate_T{T}"] = P
    say(f"T = {T}: passes", {k: {a: {m: round(1000 * x[m], 2) for m in ("tf", "dec")} for a, x in v.items()} for k, v in P.items()})
    # public probability test and generation under the gate (10 records / 4 records)
    data = e10t.data.counterfact()
    pub = {}
    for lab, obj in (("weight", L.weights()[1]), ("gate", g)):
        with B.served(obj):
            B.C.public_prob_test(B.model, B.tok, data[L.records[0]])
            sync()
            t1 = time.time()
            for ri in L.records[:10]:
                B.C.public_prob_test(B.model, B.tok, data[ri])
            sync()
            pub[f"prob_test_seconds_per_record_{lab}"] = (time.time() - t1) / 10
            from util.generate import generate_fast
            t1 = time.time()
            for c0 in (0, 2):
                torch.manual_seed(77000 + c0)
                generate_fast(B.model, B.tok, [p for ri in L.records[c0:c0 + 2] for p in data[ri]["generation_prompts"]], n_gen_per_prompt=1, max_out_len=100)
            sync()
            pub[f"generation_seconds_per_record_{lab}"] = (time.time() - t1) / 4
    res["passes"][f"public_T{T}"] = pub
    say(f"T = {T}: public", {k: round(v, 3) for k, v in pub.items()})
    res["mem"][f"T{T}"] = {"peak_alloc_MiB": torch.cuda.max_memory_allocated() / 2 ** 20, "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2 ** 20,
                           "seconds_block": time.time() - t0}
    e10t.gate.clear_cache()
    e10t.methods._BASE.clear()
    del rows, g, off, cg, S0
    torch.cuda.empty_cache()
    e10t.data.jdump(fx.io.clean(res), OUTP)
B.check_w0()
say("written", OUTP)
