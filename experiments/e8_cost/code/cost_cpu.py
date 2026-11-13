"""E8 tasks 2 and 3 on the CPU (4 BLAS threads), from stored keys; real logs and synthetic logs.

    python cost_cpu.py real <log>            one real log per process
    python cost_cpu.py synth <backbone> <T>  one synthetic log per process (T = 1000 | 10000 | 100000)

REAL LOG (default method: two-block, alpha = 0.05, n = 200, draw 0; the driver's gate with its guards = stored solver key and
B = 1 key of the own Rel prompt)
  storage           numbers and megabytes of the scoped log: serving part = detectors (T x d_in), released values (T x d_out), scales,
                    thresholds, ranking baselines (3 T): in fp64 as evaluated and in the module's dtype; guard keys (2 T x d_in; used
                    by calibration and repair only) separately; the module (d_out x d_in, module dtype) next to it
  calibration_s     KeyCache.coords of the 200 calibration queries + calibrate_two_block with guards on a gate whose guard scores are
                    not cached (the whole calibration from stored keys); 5 repetitions, median.  'slim' = the same rule on the gate
                    without guard keys, caps applied from precomputed own coordinates (the form used for the synthetic logs; the
                    thresholds are the driver's bit for bit)
  least repair      library least_repair of ONE complaint (the teacher-forced last-prompt key of a REPAIR query) on the calibrated gate:
                    30 complaints, median; 'slim' = on the gate without guard keys + the cap check on the T own coordinates
  null repair       library null_repair(ridge 0) of A in {1, 10, 100} REPAIR keys (removal of A keys from all detectors): 3 repetitions
  minimum cut       library scope_cut: rows = all valid REPAIR rows, witnesses = the text rephrases of all edits, lambda = 1 (one cut)
  revocation        fx.methods.mask_edits of one edit
  gate on CPU       decision + response for one sequence (one eligible key), fp64 and float32 gate: 30 keys x 5 runs, median
SYNTHETIC LOG (synth.py): storage, slim calibration, slim least repair, revocation (threshold +inf), gate on CPU.
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from synth import *          # noqa: F401,F403,E402

from rescope import calibrate_two_block, least_repair, null_repair, scope_cut   # noqa: E402

INF = float("inf")


def med(f, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), ts, r


def storage(T, d, m, module_bytes_per_number, model_bytes=None):
    serving = T * (d + m + 3)
    guards = 2 * T * d
    module = m * d
    out = {"T": T, "d_in": d, "d_out": m, "numbers_serving": serving, "numbers_per_edit": d + m + 3, "numbers_guard_keys": guards, "numbers_module": module,
           "mb_serving_fp64": serving * 8 / 2 ** 20, "mb_serving_module_dtype": serving * module_bytes_per_number / 2 ** 20,
           "mb_guard_keys_fp64": guards * 8 / 2 ** 20, "mb_guard_keys_float32_as_cached": guards * 4 / 2 ** 20, "mb_module": module * module_bytes_per_number / 2 ** 20,
           "mb_per_edit_fp64": (d + m + 3) * 8 / 2 ** 20, "mb_per_edit_module_dtype": (d + m + 3) * module_bytes_per_number / 2 ** 20,
           "share_of_module_numbers_pct": serving / module * 100, "share_of_module_bytes_fp64_pct": serving * 8 / (module * module_bytes_per_number) * 100,
           "T_at_module_size_numbers": module / (d + m + 3), "T_at_module_size_bytes_fp64_store": module * module_bytes_per_number / ((d + m + 3) * 8)}
    if model_bytes:
        out.update(model_bytes=model_bytes, share_of_model_bytes_fp64_pct=serving * 8 / model_bytes * 100,
                   share_of_model_bytes_module_dtype_pct=serving * module_bytes_per_number / model_bytes * 100,
                   T_at_model_size_bytes_module_dtype=model_bytes / ((d + m + 3) * module_bytes_per_number))
    return out


def model_info(bb):
    p = f"{OUT}/latency/{bb}__model.json"
    return jload(p) if os.path.exists(p) else None


def cpu_gate(S, keys, m_out):
    """ms per sequence of decision + response on the CPU for a role-last gate (one eligible key per sequence)."""
    out = {}
    for prec in ("fp64", "fp32"):
        g = S if prec == "fp64" else S.replace(precision="fp32")
        ts = []
        o = torch.zeros(1, 1, m_out, dtype=torch.float32)
        lp = torch.tensor([0])
        g.firing(keys[0][None, None], lp, 0, {})
        for k in keys:
            x = k[None, None]
            for _ in range(5):
                t0 = time.perf_counter()
                f = g.firing(x, lp, 0, {})
                if f is not None:
                    g.respond(x, o, f)
                ts.append(time.perf_counter() - t0)
        out[prec] = q3(ts, 1e3)
    return out


def real(name):
    L = fx.logs.open(name)
    T, d, m = L.T, L.log.d, L.log.m
    mi = model_info(L.backbone)
    bpn = 2
    res = {"log": name, "kind": "real", "T": T, "backbone": L.backbone, "threads": torch.get_num_threads(),
           "storage": storage(T, d, m, bpn, mi["model_bytes"] if mi else None)}
    dr = fx.draws.calibration(L, 200, 0)
    S, _ = fx.methods.base_gate(L)
    # ---- calibration from stored keys (library path, guards scored inside)
    def cal_full():
        S1 = S.replace(guard_key=S.guard_key.clone())                    # drops the cached guard scores
        Q = L.kc.coords(S1, dr.rows, dr.which)
        return calibrate_two_block(S1, coords=Q, alpha=0.05, seed=dr.seed)
    t, ts, (Sc, rec) = med(cal_full, 5)
    M = fx.methods.build("rescope", L, alpha=0.05, n=200, draw=0)
    res["calibration"] = {"library_with_guards_s": t, "runs_s": ts, "n": 200, "n_edits_capped": rec["n_edits_capped"],
                          "thresholds_equal_driver": bool(torch.equal(Sc.det_tau, M.obj.det_tau))}
    Ss, cap = real_gate_slim(L)
    t, ts, (S2, crec) = med(lambda: calibrate_slim(Ss, L.kc, dr.rows, dr.which, cap=cap), 5)
    crec.pop("Q")
    res["calibration"].update(slim_s=t, slim_stages=crec, slim_thresholds_equal_driver=bool(torch.equal(S2.det_tau, M.obj.det_tau)))
    # ---- least repair of one complaint
    rep = L.pool("repair")
    rows = [rep[int(i)] for i in np.random.default_rng(20261111).choice(len(rep), 30, replace=False)]
    K = L.kc.keys("tf", rows)
    G = M.obj
    G.guard_scores()
    ts, nr = [], 0
    for k in K:
        t0 = time.perf_counter()
        new, r = least_repair(G, k[None])
        ts.append(time.perf_counter() - t0)
        nr += r["n_raised"] > 0
    res["least_repair"] = {"library_with_guards_ms": q3(ts, 1e3), "n_complaints": len(K), "n_with_a_raised_threshold": int(nr)}
    ts = []
    for k in K:
        t0 = time.perf_counter()
        new, r = least_repair(S2, k[None])
        bad = int((new.det_tau > cap).sum())
        ts.append(time.perf_counter() - t0)
    res["least_repair"]["slim_ms"] = q3(ts, 1e3)
    # ---- removal of A keys from the detectors
    res["null_repair"] = {}
    Ka = L.kc.keys("tf", rep[:100])
    for A in (1, 10, 100):
        t, ts, _ = med(lambda: null_repair(G, Ka[:A], ridge=0.0), 3)
        res["null_repair"][str(A)] = {"s": t, "runs_s": ts}
    # ---- one minimum cut
    Qr = L.kc.coords(S, rep, "tf")
    wr = L.edit_rows["tgen"]
    Rw = fx.gate.features(L, S, wr, "tf")
    wx = Rw[torch.arange(T), torch.arange(T)]
    t, ts, (Scut, crec) = med(lambda: scope_cut(G, Qr, (list(range(T)), wx), lam=1.0), 3)
    res["minimum_cut"] = {"s": t, "runs_s": ts, "n_rows": int(Qr.shape[0]), "n_witnesses": T, "lam": 1.0,
                          "n_clean": crec["n_clean"], "n_uncovered": crec["n_uncovered"], "graph": {k: v for k, v in crec["flow"].items() if isinstance(v, (int, float))}}
    # ---- revocation of one edit
    t, ts, _ = med(lambda: fx.methods.mask_edits(L, M, [T // 2]), 5)
    res["revocation"] = {"ms": t * 1e3, "runs_ms": [x * 1e3 for x in ts], "rule": "fx.methods.mask_edits: threshold of the edit = +inf, its guards dropped"}
    t, ts, _ = med(lambda: G.replace(det_tau=torch.where(torch.arange(T) == T // 2, torch.full_like(G.det_tau, INF), G.det_tau)), 5)
    res["revocation"]["threshold_only_ms"] = t * 1e3
    res["gate_cpu_ms"] = cpu_gate(G.replace(guard_key=G.guard_key[:0], guard_edit=G.guard_edit[:0]), K, m)
    jdump(res, f"{OUT}/cpu/{name}.json")
    print(name, {k: res[k] for k in ("calibration", "revocation")}, res["least_repair"]["library_with_guards_ms"]["median"], res["gate_cpu_ms"]["fp64"]["median"], flush=True)


def synth(bb, T):
    t0 = time.time()
    S, srec = synth_gate(bb, T)
    Lr, Wr, sr, Vr, dr = reference(bb)
    d, m = S.d, S.values.shape[1]
    mi = model_info(bb)
    res = {"log": f"synthetic-T{T}", "kind": "synthetic", "T": T, "backbone": bb, "threads": torch.get_num_threads(), "synthetic": srec,
           "storage": storage(T, d, m, 2, mi["model_bytes"] if mi else None)}
    reps = 3 if T >= 100000 else 5
    t, ts, (S2, crec) = med(lambda: calibrate_slim(S, Lr.kc, dr.rows, dr.which), reps)
    Q = crec.pop("Q")
    Sref, _ = real_gate_slim(Lr)
    Qref = Lr.kc.coords(Sref, dr.rows, dr.which)
    qq = (0.01, 0.5, 0.99, 0.999)
    res["calibration"] = {"slim_s": t, "runs_s": ts, "slim_stages": crec, "n": 200,
                          "coordinate_quantiles_synthetic": {str(q): float(np.quantile(Q.numpy(), q)) for q in qq},
                          "coordinate_quantiles_reference_log": {str(q): float(np.quantile(Qref.numpy(), q)) for q in qq},
                          "query_max_median_synthetic": float(Q.max(1).values.median()), "query_max_median_reference_log": float(Qref.max(1).values.median())}
    del Q, Qref, Sref
    rep = Lr.pool("repair")
    rows = [rep[int(i)] for i in np.random.default_rng(20261111).choice(len(rep), 30 if T < 100000 else 10, replace=False)]
    K = Lr.kc.keys("tf", rows)
    ts, nr = [], 0
    for k in K:
        t1 = time.perf_counter()
        new, r = least_repair(S2, k[None])
        th, nc = cap_own(new.det_tau)
        ts.append(time.perf_counter() - t1)
        nr += r["n_raised"] > 0
    res["least_repair"] = {"slim_ms": q3(ts, 1e3), "n_complaints": len(K), "n_with_a_raised_threshold": int(nr)}
    t, ts, _ = med(lambda: S2.replace(det_tau=torch.where(torch.arange(T) == T // 2, torch.full_like(S2.det_tau, INF), S2.det_tau)), 5)
    res["revocation"] = {"threshold_only_ms": t * 1e3, "rule": "threshold of the edit = +inf (one write; a new threshold vector is allocated)"}
    res["gate_cpu_ms"] = cpu_gate(S2, K, m)
    res["seconds_total"] = time.time() - t0
    jdump(res, f"{OUT}/cpu/{bb}__synthetic-T{T}.json")
    print(bb, T, res["calibration"]["slim_s"], res["least_repair"]["slim_ms"]["median"], res["gate_cpu_ms"]["fp64"]["median"], flush=True)


if __name__ == "__main__":
    os.makedirs(OUT + "/cpu", exist_ok=True)
    if sys.argv[1] == "real":
        real(sys.argv[2])
    else:
        synth(sys.argv[2], int(sys.argv[3]))
