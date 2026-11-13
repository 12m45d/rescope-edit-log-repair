"""Independent verification of E3 (check 2): one log per process, the library's PLAIN Evaluator.

    gq run --mem 10500|17000 -- python v_gpu.py <log> [--smoke]

No driver (fx), no memo, no key cache, no SparseEvaluator: every key is captured here with the model, every row is a
forward through rescope.Evaluator.  Calibration draw 0, alpha = 0.05, n = 200.  Methods: rescope (two-block default),
rescope_null (null repair on 100 REPAIR queries, then two-block), cosine_shared (shared-scale cosine gate).

Stages (each saved; a finished stage is skipped on restart):
  keys    own B = 1 Rel keys of all T edits, the 200 calibration keys, the 100 null-repair keys, 100 seeded rephrase keys (Qwen)
  gates   thresholds by MY OWN numpy rule (two-block / shared, guard caps) and by the library's calibrate_* on the same keys
  probes  heldout_disjoint (all valid rows; a seeded sample of 200 on logs with 1000 or more edits); LLaVA: lp='prompt' on shifted rows
  edits   Rel / T-Gen / I-Gen of a seeded sample of 150 edits (100 on logs with 1000 or more edits)
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import argparse
import gzip
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
sys.path.insert(0, f"{_RP.ROOT}/02_research/code")
from rescope import Evaluator, add_guards, calibrate_shared, calibrate_two_block, feature_gate, from_harness, load, null_repair  # noqa: E402

ROOT = f"{_RP.ROOT}/02_research/experiments/10_final/e3_qwen/verify"
ALPHA_NUM, ALPHA_DEN = 1, 20
N_CAL, N_NULL, DRAW = 200, 100, 0
SEED_EDITS, SEED_PROBES, SEED_COV = 20261112, 20261113, 20261114
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def jdump(o, p):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(o, f)
    os.replace(tmp, p)


def wrows(rows, p):
    tmp = p + ".tmp"
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, p)


# ----------------------------------------------------------------------------- my own rules (numpy, fp64)
def feat(kind, detw, s, K):
    """(n, D) features of keys K (n, d): 'ncoord' <w_j, k> / s_j ; 'cos' cosine(k, key_j) with detw = normalised stored keys."""
    K = np.asarray(K, dtype=np.float64)
    out = np.empty((K.shape[0], detw.shape[0]))
    for i in range(K.shape[0]):                                         # one key at a time, as the hook scores a one-key query
        c = detw @ K[i]
        out[i] = c / s if kind == "ncoord" else c / max(np.linalg.norm(K[i]), 1e-300)
    return out


def conf_rank(n):
    return -((-(ALPHA_DEN - ALPHA_NUM) * (n + 1)) // ALPHA_DEN)        # ceil((1 - alpha)(n + 1)), integers


def my_rule(Q, Rg, ge, T, two_block, seed):
    """Q (n, D) calibration features; Rg (G, D) features of the guard keys, ge (G,) their edits.
    -> b, eta, literal level b + eta (candidate iff r - b > eta), cap, capped (bool), n_capped."""
    n, D = Q.shape
    if two_block:
        perm = np.random.default_rng(seed).permutation(n)
        i1, i2 = np.sort(perm[:n // 2]), np.sort(perm[n // 2:])
        b = Q[i1].max(0)
    else:
        i2, b = np.arange(n), np.zeros(D)
    sc = (Q[i2] - b[None, :]).max(1)
    k = conf_rank(len(i2))
    eta = float(np.sort(sc)[k - 1]) if k <= len(i2) else float("inf")
    low = np.full(D, np.inf)
    for j in range(T):                                                  # one detector per edit: smallest own-guard feature
        low[j] = Rg[ge == j, j].min()
    capped = ~((low - b) > eta)                                         # the literal (strict) rule silences a guard of the edit
    cap = low - 1e-9 * np.abs(low)
    return {"b": b, "eta": eta, "lit": b + eta, "cap": cap, "capped": capped, "n_capped": int(capped.sum()), "rank": k, "n2": int(len(i2)),
            "bound": (len(i2) + 1 - k) / (len(i2) + 1) if k <= len(i2) else 0.0}


def my_null(W, K, s, Kb):
    """Minimum-change detectors with zero coordinate on the benign keys Kb and own coordinate s kept:
    w' = P w + nu P k, P = projector onto the orthogonal complement of span(Kb), nu = (s - <P w, k>) / <P k, k>."""
    Qb, _ = np.linalg.qr(np.asarray(Kb, dtype=np.float64).T)            # (d, n_b) orthonormal basis of the benign span
    P = lambda X: X - (X @ Qb) @ Qb.T                                   # noqa: E731
    Wt, Kt = P(W), P(K)
    nu = (s - (Wt * K).sum(1)) / (Kt * K).sum(1)
    return Wt + nu[:, None] * Kt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    name = a.log
    out = os.path.join(ROOT, "out", "smoke" if a.smoke else "gpu", name)
    os.makedirs(out, exist_ok=True)
    mem_gb = (int(os.environ["GQ_MEM"]) - 600) / 1024.0
    H = load(name, device="cuda:0", mem_gb=mem_gb)
    ev = Evaluator(H, hard_negatives=False)
    T = int(H.T)
    llava = H.backbone != "qwen25vl"
    say("loaded", name, "T", T, "backbone", H.backbone, H.load_info)
    log = from_harness(H)                                               # CPU, fp64

    # ------------------------------------------------------------------ the draw (documented rule: seeded subset of the valid REPAIR rows)
    rep = ev.probe_items("repair")
    idx = np.sort(np.random.default_rng(DRAW).choice(len(rep), N_CAL, replace=False))
    cal = [rep[int(i)] for i in idx]
    # null-repair sample: first 100 rows after the calibration draw in the draw's seeded ranking whose QUERY (ptype, question,
    # image content) is no calibration query and not yet in the sample (PROTOCOL 09:03; ranking of the driver's USAGE)
    from vendor_v7.vlm import VLKEB_IMG_DIR
    md5 = {}

    def qid(it):
        r = H.row[(it["probe_idx"], it["ptype"])]
        im = r.get("image")
        if im is not None and im not in md5:
            h = hashlib.md5()
            with open(im if os.path.isabs(im) else os.path.join(VLKEB_IMG_DIR, im), "rb") as f:
                h.update(f.read())
            md5[im] = h.hexdigest()
        return (r["ptype"], r["prompt"], md5.get(im))
    rest = np.setdiff1d(np.arange(len(rep)), idx)
    rest = rest[np.random.default_rng([DRAW, 2]).permutation(rest.size)]
    seen, pick = {qid(it) for it in cal}, []
    for k in rest:
        if len(pick) == N_NULL:
            break
        q = qid(rep[int(k)])
        if q not in seen:
            seen.add(q)
            pick.append(int(k))
    assert len(pick) == N_NULL
    nul = [rep[k] for k in sorted(pick)]
    n_shared_path = len({(H.row[(it["probe_idx"], it["ptype"])]["prompt"], H.row[(it["probe_idx"], it["ptype"])].get("image"), it["ptype"]) for it in cal}
                        & {(H.row[(it["probe_idx"], it["ptype"])]["prompt"], H.row[(it["probe_idx"], it["ptype"])].get("image"), it["ptype"]) for it in nul})
    assert not ({qid(it) for it in cal} & {qid(it) for it in nul}) and not n_shared_path

    # ------------------------------------------------------------------ stage keys
    pk = os.path.join(out, "keys.pt")
    n_e = (5 if a.smoke else (100 if T >= 1000 else 150))
    e_s = sorted(int(i) for i in np.random.default_rng(SEED_EDITS).choice(T, n_e, replace=False))
    cov_pick = sorted(int(i) for i in np.random.default_rng(SEED_COV).choice(2 * T, 10 if a.smoke else 100, replace=False))
    cov_items = [ev.edit_items((("tgen", "igen")[i // T],), [i % T])[0] for i in cov_pick] if not llava else []
    if os.path.exists(pk):
        KS = torch.load(pk)
    else:
        t0 = time.time()
        with torch.no_grad():                                           # no vision cache: every sequence is captured once

            def last_key(it):
                x, p = ev.capture(it, True)
                return x[p].float().cpu()
            Kown = torch.stack([last_key(it) for it in ev.edit_items(("rel",))])       # = Evaluator.own_keys('rel'): B = 1 own Rel sequences
            eown = list(range(T))
            say("own keys", tuple(Kown.shape), f"{time.time() - t0:.0f}s")
            Kcal = torch.stack([last_key(it) for it in cal])
            Knul = torch.stack([last_key(it) for it in nul])
            Kcov = torch.stack([last_key(it) for it in cov_items]) if cov_items else torch.zeros(0, Kcal.shape[1])
        KS = {"own": Kown.float().cpu(), "own_edit": [int(e) for e in eown], "cal": Kcal, "cal_sid": [it["sid"] for it in cal], "null": Knul,
              "null_sid": [it["sid"] for it in nul], "cov": Kcov, "cov_sid": [it["sid"] for it in cov_items], "dtype": str(H.dtype)}
        torch.save(KS, pk)
        say("keys captured", f"{time.time() - t0:.0f}s", "module dtype", KS["dtype"])
    assert KS["cal_sid"] == [it["sid"] for it in cal] and KS["null_sid"] == [it["sid"] for it in nul] and KS["own_edit"] == list(range(T))

    # ------------------------------------------------------------------ stage gates
    t0 = time.time()
    Kown, Kcal, Knul = KS["own"], KS["cal"], KS["null"]
    W, Ks = log.W.numpy(), log.K.numpy()
    s = (W * Ks).sum(1)
    Kn = Ks / np.linalg.norm(Ks, axis=1, keepdims=True)
    guards = np.concatenate([Ks, Kown.double().numpy()])                # stored solver keys + B = 1 own Rel keys
    ge = np.concatenate([np.arange(T), np.arange(T)])
    Wnull = my_null(W, Ks, s, Knul.double().numpy())
    mine = {}
    for lab, kind, detw, tb in (("rescope", "ncoord", W, True), ("rescope_null", "ncoord", Wnull, True), ("cosine_shared", "cos", Kn, False)):
        ss = s if kind == "ncoord" else np.ones(T)
        Q = feat(kind, detw, ss, Kcal.double().numpy())
        Rg = feat(kind, detw, ss, guards)
        mine[lab] = dict(my_rule(Q, Rg, ge, T, tb, DRAW), kind=kind)
    say("my rules", {k: (v["eta"], v["n_capped"]) for k, v in mine.items()}, f"{time.time() - t0:.0f}s")

    # the library's gates on the same captured keys (what the Evaluator runs)
    def base(feature):
        S = feature_gate(log, feature)
        S, _ = add_guards(S, Kown, list(range(T)))
        return S
    qs = [k[None] for k in Kcal]
    Sn = base("ncoord")
    G = {}
    G["rescope"], rec_r = calibrate_two_block(Sn, qs, alpha=0.05, seed=DRAW)
    Snull, rec_null = null_repair(Sn, Knul, ridge=0.0)
    G["rescope_null"], rec_n = calibrate_two_block(Snull, qs, alpha=0.05, seed=DRAW)
    G["cosine_shared"], rec_c = calibrate_shared(base("cos"), qs, alpha=0.05)
    lib_rec = {"rescope": rec_r, "rescope_null": rec_n, "cosine_shared": rec_c}
    cmp = {}
    for lab, g in G.items():
        m = mine[lab]
        th = g.theta().numpy()
        my_th = np.where(m["capped"], m["cap"], m["lit"])
        un = ~m["capped"]
        cmp[lab] = {"eta_mine": m["eta"], "eta_lib": lib_rec[lab]["eta"], "n_capped_mine": m["n_capped"], "n_capped_lib": lib_rec[lab]["n_edits_capped"],
                    "capped_sets_equal": sorted(np.nonzero(m["capped"])[0].tolist()) == lib_rec[lab]["capped_edits"],
                    "max_abs_thr_diff": float(np.abs(th - my_th).max()), "max_abs_thr_diff_uncapped": float(np.abs(th - my_th)[un].max()) if un.any() else None,
                    "rank": m["rank"], "n2": m["n2"], "bound": m["bound"], "bound_lib": lib_rec[lab]["bound_false_firing"],
                    "thr_min_med_max": [float(x) for x in np.quantile(my_th, [0, 0.5, 1])]}
    cmp["null_repair"] = {"max_rel_det_diff_mine_vs_lib": float((np.linalg.norm(Wnull - Snull.det_w.numpy(), axis=1) / np.linalg.norm(Wnull, axis=1)).max()),
                          "max_abs_benign_coord_mine": float(np.abs((Wnull @ Knul.double().numpy().T) / s[:, None]).max()),
                          "max_own_rel_err_mine": float(np.abs((Wnull * Ks).sum(1) / s - 1).max()), "lib": {k: v for k, v in rec_null.items() if not isinstance(v, list)}}
    say("gates", json.dumps(cmp)[:1500])
    torch.save({"mine": {k: {x: (torch.from_numpy(np.asarray(y)) if isinstance(y, np.ndarray) else y) for x, y in v.items()} for k, v in mine.items()},
                "lib_theta": {k: g.theta() for k, g in G.items()}, "Wnull_mine": torch.from_numpy(Wnull), "cmp": cmp}, os.path.join(out, "gates.pt"))
    jdump({"log": name, "T": T, "cmp": cmp, "cal_sid": KS["cal_sid"], "null_sid": KS["null_sid"], "n_repair_valid": len(rep), "edit_sample": e_s,
           "lib_rec": {k: {x: y for x, y in v.items() if not isinstance(y, (list, dict))} for k, v in lib_rec.items()},
           "capped_edits": {k: v["capped_edits"] for k, v in lib_rec.items()}}, os.path.join(out, "gates.json"))
    slim = {k: g.replace(guard_key=g.guard_key[:0], guard_edit=g.guard_edit[:0]) for k, g in G.items()}
    del Sn, Snull, G

    def run(items, lp="tf"):
        """Plain Evaluator rows of the three methods; one method at a time on the long logs (GPU memory)."""
        if T >= 1000:
            o = {}
            for k, g in slim.items():
                o[k] = ev.tf_rows(g, items, lp=lp)
                torch.cuda.empty_cache()
            return o
        return ev.tf_rows(slim, items, lp=lp)

    # ------------------------------------------------------------------ stage probes
    hd = ev.probe_items("heldout_disjoint")
    n_hd = len(hd)
    if a.smoke:
        hd = hd[:10]
    elif T >= 1000:
        pi = np.sort(np.random.default_rng(SEED_PROBES).choice(n_hd, 200, replace=False))
        hd = [hd[int(i)] for i in pi]
    pp = os.path.join(out, "probes.json")
    if not os.path.exists(pp):
        t0 = time.time()
        rows = run(hd)
        for k, v in rows.items():
            wrows(v, os.path.join(out, f"rows__{k}__probe_tf.jsonl.gz"))
        info = {"n_heldout_disjoint_valid": n_hd, "n_evaluated": len(hd), "seconds": time.time() - t0}
        if llava:
            sh = [it for it in hd if ev.lp_shift(it)]
            info["n_shifted"] = len(sh)
            info["n_shift_undefined"] = sum(ev.lp_shift(it) is None for it in hd)
            if sh:
                rp = run(sh, lp="prompt")
                for k, v in rp.items():
                    wrows(v, os.path.join(out, f"rows__{k}__probe_prompt.jsonl.gz"))
        info["summary"] = {k: {"n": len(v), "fired": sum(bool(r["fired"]) for r in v), "D1": sum(bool(r["D1"]) for r in v)} for k, v in rows.items()}
        jdump(info, pp)
        say("probes", info)

    # ------------------------------------------------------------------ stage edits
    pe = os.path.join(out, "edits.json")
    if not os.path.exists(pe):
        t0 = time.time()
        items = ev.edit_items(("rel", "tgen", "igen"), e_s)
        rows = run(items)
        for k, v in rows.items():
            wrows(v, os.path.join(out, f"rows__{k}__edit_tf.jsonl.gz"))
        info = {"edits": e_s, "seconds": time.time() - t0,
                "summary": {k: {m: {"n": sum(r["metric"] == m for r in v), "exact": sum(bool(r["exact"]) for r in v if r["metric"] == m),
                                    "own_released": sum(bool(r["own_released_last"]) for r in v if r["metric"] == m)} for m in ("rel", "tgen", "igen")}
                            for k, v in rows.items()}}
        jdump(info, pe)
        say("edits", info["summary"])
    jdump({"log": name, "done": True, "seconds": time.time() - T0, "peak_rss_gb": H.load_info.get("peak_rss_gb"), "written": time.strftime("%Y-%m-%d %H:%M:%S")},
          os.path.join(out, "DONE.json"))
    say("JOB DONE")


if __name__ == "__main__":
    main()
