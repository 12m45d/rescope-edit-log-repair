"""E8 task 1: gate latency per forward pass on the GPU (one process per backbone, one model handle = the development log's).

    gq run --mem 10500 --prio 2 -- python lat.py qwen25vl [--pass 2]       (17000 for llava)
A second pass (--pass 2 -> out/latency_pass2/; optionally for a subset of gate sets) repeats the measurement at another time; the rule, fixed before the
second pass was run: per gate set, the pass with the LOWER median card utilisation is the primary one (ties: pass 1); both are kept and printed.

Gate sets: the default method (two-block, alpha = 0.05, n = 200, draw 0) of the development log and of the MEMIT-style logs
(their detectors, thresholds and released values; the pre-edit weight is the backbone's), and synthetic logs (synth.py) with
T = 1000 and 10000.  Modes per gate set, timed on 30 sequences (10 edit sequences, 10 image probes, 10 text probes; seeded),
1 warm-up + 9 repetitions, the modes rotating within every (repetition, sequence) so that each gated forward has a plain
pre-edit forward next to it:
    plain_W0        the pre-edit model
    last_fp64       role last, top-1, REPL, gate in fp64 (as evaluated)
    last_module     the same gate in the module's dtype
    all_fp64        role all positions, fire all, response LIN (the form of ReScope-W) with the SAME calibrated thresholds
    all_nothr_fp64  the same with all thresholds -inf (every edit released at every position: the deployed scoping replayed; real logs only)
    where_logonly   development log only: ReScope-W with its log-only thresholds (the driver's method)
Reported per mode: ms per forward and paired ratio to the plain forward of the same (repetition, sequence): median, 10th, 90th
percentile; the gate alone (decision + response) on the captured module input of each sequence, 10 runs per sequence (median /
percentiles over the runs; mean of the per-sequence minima = the library's convention for the gate alone, nearly free of contention by other
processes on the card; role-last gates: 50 runs per sequence, all-position gates: 10).  Card utilisation and the number of other processes are sampled
before and after every repetition.  No driver session and no memo are used (nothing behavioural is evaluated).
"""
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from synth import *          # noqa: F401,F403,E402

import rescope               # noqa: E402
from rescope.scoped import Hook   # noqa: E402

REPS, NRUN, NRUN_LAST = 9, 10, 50          # gate alone: 50 runs per sequence for the role-last gates (cheap), 10 for the all-position gates
INF = float("inf")


def seq30(ev):
    rng = np.random.default_rng(20261110)
    pr = ev.probe_items("all")
    pick = lambda xs, n: [xs[int(i)] for i in sorted(rng.choice(len(xs), n, replace=False))]          # noqa: E731
    return pick(ev.edit_items(), 10) + pick([p for p in pr if p["ptype"] == "image"], 10) + pick([p for p in pr if p["ptype"] == "text"], 10)


def model_bytes(H):
    best = None
    for v in vars(H.vlm).values():
        if isinstance(v, torch.nn.Module):
            n = sum(p.numel() for p in v.parameters())
            b = sum(p.numel() * p.element_size() for p in v.parameters())
            if best is None or n > best[0]:
                best = (n, b)
    return best if best is not None else (None, None)


def time_set(H, ev, items, batches, gates, tag, meta):
    names = ["plain_W0"] + list(gates)
    modes = [None] + list(gates.values())
    T = np.zeros((REPS + 1, len(modes), len(batches)))
    fired = {k: [0] * len(batches) for k in names}
    load = [gpu_load()]
    with torch.no_grad(), H.at(H.W0):
        for rep in range(REPS + 1):
            for bi, b in enumerate(batches):
                for o in range(len(modes)):
                    mi = (o + rep + bi) % len(modes)
                    g = modes[mi]
                    hk = None if g is None else Hook(H.mod, g, set_weight=False, record=False)
                    if hk is not None:
                        hk.__enter__()
                        hk.begin(b["last_prompt_idx"])
                    try:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        H.vlm.forward(b, logits_to_keep="answer")
                        torch.cuda.synchronize()
                        T[rep, mi, bi] = time.perf_counter() - t0
                    finally:
                        if hk is not None:
                            hk.__exit__(None, None, None)
                    if hk is not None and rep == REPS:
                        fired[names[mi]][bi] = 0 if hk.last is None else int(hk.last.idx.numel())
            load.append(gpu_load())
    ntok = np.array([int(b["seq_len"][0]) for b in batches])
    G = np.full((len(modes), len(batches), NRUN_LAST), np.nan)
    nrun = [NRUN_LAST if (g is None or g.role == "last") else NRUN for g in modes]
    with torch.no_grad():
        for bi, (it, b) in enumerate(zip(items, batches)):
            x, lp = ev.capture(it, batch=b)
            x, lpt = x[None], torch.tensor([lp], device=x.device)
            out0 = torch.nn.functional.linear(x, H.W0)
            for gi, g in enumerate(modes):
                for r in range(nrun[gi]):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if g is None:
                        torch.nn.functional.linear(x, H.W0)
                    else:
                        f = g.firing(x, lpt, 0, {})
                        if f is not None:
                            g.respond(x, out0, f)
                    torch.cuda.synchronize()
                    G[gi, bi, r] = time.perf_counter() - t0
            del x, out0
    load.append(gpu_load())
    out = dict(meta, tag=tag, n_sequences=len(batches), reps=REPS, mean_seq_len=float(ntok.mean()), device=torch.cuda.get_device_name(0),
               load_samples=load, load_utilization_pct=q3([s.get("utilization_pct", np.nan) for s in load]),
               load_other_processes_max=max(s.get("n_other_processes", 0) for s in load),
               load_other_processes_median=float(np.median([s.get("n_other_processes", 0) for s in load])),
               module_linear_ms=q3(G[0], 1e3), gate_only_runs_per_sequence=dict(zip(names, nrun)), modes={}, raw_seconds=T[1:].tolist(), mode_names=names)
    P = T[1:]
    for mi, k in enumerate(names):
        ratio = P[:, mi] / P[:, 0]
        d = {"ms_per_forward": q3(P[:, mi], 1e3), "n_positions_firing_last_rep": int(sum(fired[k])), "n_sequences_firing": int(sum(v > 0 for v in fired[k]))}
        if mi:
            d.update(overhead_end_to_end_pct=q3((ratio - 1) * 100), overhead_ms=q3(P[:, mi] - P[:, 0], 1e3),
                     gate_only_ms=q3(G[mi][:, :nrun[mi]], 1e3), gate_only_ms_mean_of_min=float(np.nanmin(G[mi], 1).mean() * 1e3),
                     gate_only_ms_median_of_min=float(np.median(np.nanmin(G[mi], 1)) * 1e3),
                     gate_only_pct_of_plain_forward=float(np.nanmin(G[mi], 1).mean() / P[:, 0].min(0).mean() * 100))
        out["modes"][k] = d
    return out


def gate_modes(g_last, real=True):
    ga = g_last.replace(role="all", fire="all", response="lin", det_b=None, det_u=None)
    m = {"last_fp64": g_last, "last_module": g_last.replace(precision="module"), "all_fp64": ga}
    if real:
        m["all_nothr_fp64"] = ga.replace(det_tau=torch.full_like(ga.det_tau, -INF))
    return m


def main():
    bb = sys.argv[1]
    args = sys.argv[2:]
    npass = 1
    if "--pass" in args:                                                # a further pass of the same measurement at another time (another load): out/latency_pass<k>/
        i = args.index("--pass")
        npass = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    only = args                                                         # optional subset of tags
    out_dir = OUT + ("/latency" if npass == 1 else f"/latency_pass{npass}")
    os.makedirs(out_dir, exist_ok=True)
    dev_log = DEV[bb]
    H = fx.logs.load_model(dev_log)
    ev = rescope.Evaluator(H)
    items = seq30(ev)
    batches = [ev.batch(it) for it in items]
    nparam, nbytes = model_bytes(H)
    base = {"backbone": bb, "model_handle_of": dev_log, "model_parameters": nparam, "model_bytes": nbytes, "module_shape": list(H.W0.shape),
            "module_dtype": str(H.W0.dtype), "gq_mem": os.environ.get("GQ_MEM")}
    jdump(base, f"{out_dir}/{bb}__model.json")
    dev = torch.device(H.dev)
    tags = [("real", n) for n in REAL[bb]] + [("synthetic", T) for T in (1000, 10000)]
    for kind, x in tags:
        tag = x if kind == "real" else f"synthetic-T{x}"
        path = f"{out_dir}/{bb}__{tag}.json"
        if (only and tag not in only) or os.path.exists(path):
            continue
        t0 = time.time()
        if kind == "real":
            L = fx.logs.open(x)
            M = fx.methods.build("rescope", L, alpha=0.05, n=200, draw=0)
            g = M.obj.replace(guard_key=M.obj.guard_key[:0], guard_edit=M.obj.guard_edit[:0], W0=H.W0).to(dev)
            meta = dict(base, kind="real", log=x, T=L.T, method="rescope (two-block, alpha 0.05, n 200, draw 0)", n_edits_capped=fx.methods.n_capped(M))
            gates = gate_modes(g, True)
            if x == dev_log:
                Mw = fx.methods.build("where_logonly", L, H)
                gates["where_logonly"] = Mw.obj.replace(guard_key=Mw.obj.guard_key[:0], guard_edit=Mw.obj.guard_edit[:0], W0=H.W0).to(dev)
        else:
            S, srec = synth_gate(bb, x)
            Lr, _, _, _, dr = reference(bb)
            S2, crec = calibrate_slim(S, Lr.kc, dr.rows, dr.which)
            crec.pop("Q")
            g = S2.replace(W0=H.W0).to(dev)
            del S, S2
            meta = dict(base, kind="synthetic", log=tag, T=x, synthetic=srec, calibration=crec, n_edits_capped=crec["n_edits_capped"])
            gates = gate_modes(g, False)
        res = time_set(H, ev, items, batches, gates, tag, meta)
        res["seconds_total"] = time.time() - t0
        jdump(res, path)
        print(tag, {k: (round(v["ms_per_forward"]["median"], 2), round(v.get("gate_only_ms", {}).get("median", 0), 3),
                        round(v.get("overhead_end_to_end_pct", {}).get("median", 0), 2)) for k, v in res["modes"].items()}, flush=True)
        del gates, g
        fx.gate.clear_cache()
        torch.cuda.empty_cache()
    # ---- task 4: the answer-match baseline, 50 queries (two generations per query) next to one generation
    path = f"{out_dir}/{bb}__dual.json"
    if not only and npass == 1 and not os.path.exists(path):
        its = ev.probe_items("heldout_disjoint")
        rng = np.random.default_rng(20261111)
        its = [its[int(i)] for i in sorted(rng.choice(len(its), 50, replace=False))]
        res = {"backbone": bb, "log": dev_log, "n_queries": len(its), "load_before": gpu_load()}
        for name, meth in (("preedit_one_generation", {"preedit": H.W0}), ("dual_two_generations", {"dual": ev.dual("eq")}),
                           ("deployed_one_generation", {"deployed": None})):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rows = ev.generate(meth, its)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r = list(rows.values())[0]
            res[name] = {"seconds_total": dt, "ms_per_query": dt / len(its) * 1e3, "n_rows": len(r),
                         "n_generations_per_query": r[0].get("n_generations", 1) if r else None}
        res["load_after"] = gpu_load()
        jdump(res, path)
        print("dual", {k: v["ms_per_query"] for k, v in res.items() if isinstance(v, dict) and "ms_per_query" in v}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
