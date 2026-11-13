"""E2 on ONE prefix log (GPU job; one log per process; resumable: a finished cell is skipped).

    $GQ run --mem 10500|17000 --prio 1 --name e2p_<log> -- python job.py <prefix log> [--phases A,B,C,D,E,F,G] [--limit K]

  A  linear repairs BUILT through the shared harness with the calls of P12 (run_streams.make: COMP_B, COMP_BP, REEDIT_B, RB_TOT,
     RB_UNION) on the nested complaint sets fx.draws.complaints(L, n, seed, n_base=20); every correction is saved ONCE (out/corr/).
  B  every saved correction re-scored through the driver (fx.behave, B = 1): the prefix's edits, heldout_disjoint, the reported rows.
  C  capture: per row (valid probes, edit sequences, valid hard negatives of the prefix's edits) the vector over edits of the largest
     normalised coordinate over all positions of the teacher-forced sequence (out/qmax/<log>.pt).
  D  thresholds-only repair: where_logonly, then fx.methods.where_least(start='logonly') on the same complaint sets; behaviour;
     gate level from C; hook == gate level on the evaluated rows; N_ORDERS arrival orders give the same thresholds.
  E  blocklist line (ses.blocklist) on the same complaint sets.
  F  default method ReScope(0.05), n = 200, calibration draws 0 - 2: behaviour.
  G  ReScope-W with two-block thresholds from the same benign queries (R24): library rule on the vectors of C; gate level for 20
     draws (literal rule next to it), behaviour for draws 0 - 2; draw 0 == the driver's own constructor.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcommon import *  # noqa: F401,F403,E402
from pcommon import fx, np, torch  # noqa: E402

T00 = time.time()


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T00:6.0f}s] {msg}", flush=True)


def attempt(log, name, fn):
    """Run one cell; a failure is recorded (cells/<log>/<name>.failed.json, with the attempt count); after two failures the cell
    is skipped for good (the cause stays on file)."""
    fp = cell_path(log, name).replace(".json", ".failed.json")
    prev = fx.io.jload(fp) if os.path.exists(fp) else None
    if prev is not None and prev.get("attempts", 0) >= 2:
        return False
    try:
        fn()
        if os.path.exists(fp):
            os.replace(fp, fp.replace(".failed.json", ".failed_then_ok.json"))
        return True
    except Exception as e:  # noqa: BLE001
        say(f"  {name}: FAILED {type(e).__name__}: {str(e)[:300]}")
        fx.io.jdump({"log": log, "cell": name, "error": f"{type(e).__name__}: {e}"[:2000], "traceback": traceback.format_exc()[-4000:],
                     "attempts": (prev or {}).get("attempts", 0) + 1, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, fp, indent=1)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--phases", default="A,B,C,D,E,F,G")      # H (D2 decodes) is run as a second pass, after every log has its cells A - G
    ap.add_argument("--limit", type=int, default=None, help="smoke: at most this many cells per phase")
    a = ap.parse_args()
    name, phases = a.log, a.phases.split(",")
    if name not in LOGS:
        raise SystemExit(f"{name} is not a prefix log")
    for d in (os.path.join(CELLS, name), ROWS, os.path.join(CORR, name), QDIR, HCACHE, P12CACHE, LOGD):
        os.makedirs(d, exist_ok=True)
    import rescope
    say(f"==== job {name} phases {phases} pid {os.getpid()} GQ_MEM {os.environ.get('GQ_MEM')} fx {fx.code_fingerprint()}")
    H = fx.prefix.load_model(name, keep_batch_deltas=True, cache_dir=HCACHE)
    H._D_act = None                                                     # (a cache of W_E - W_0 on the solve device; rebuilt on first use)
    for key in [key for key in H._setup if isinstance(key, tuple) and key and key[0] in ("basis", "reedit")]:
        H._setup.pop(key)
    H.solve_dev = torch.device("cpu")                                   # P12: load(name, solve_device="cpu") (fp64 solves on the CPU, both backbones)
    L = fx.logs.open(name)
    hz = rescope.harness()
    say(f"loaded: T {H.T}, checks {H.checks}, harness cache {hz.HC.CACHE}, solve device {H.solve_dev}")
    limit = a.limit

    # ---- the nested complaint sets (sequences of the key cache) and their harness rows
    hrow = {(r["probe_idx"], r["ptype"]): r for r in H.rows}
    streams, hstream = {}, {}
    for s in SEEDS:
        st = fx.draws.complaint_stream(L, s, n_base=N_BASE)
        streams[s] = st
        hstream[s] = [dict(hrow[(L.index[j]["probe_idx"], L.index[j]["ptype"])], complaint_kind="D1", sample_seed=s) for j in st]
        h20 = hz.HS.sample_complaints(H, N_BASE, s)
        if {(c["probe_idx"], c["ptype"]) for c in h20} != {(c["probe_idx"], c["ptype"]) for c in hstream[s][:N_BASE]}:
            raise RuntimeError("the first complaints of the stream are not the harness sample")
        if not all(c["is_complaint_D1"] and c["y0_valid"] for c in hstream[s]):
            raise RuntimeError("a reported row is not a D1 complaint with a valid pre-edit answer")
    nval = lambda n, s: len(streams[s]) if n == "ALL" else int(n)       # noqa: E731
    seqs = lambda n, s: streams[s][:nval(n, s)]                         # noqa: E731
    if not done(name, "setup"):
        put(name, "setup", {"log": name, "T": L.T, "n_repair_complaints": len(streams[0]), "n_base": N_BASE,
                            "streams": {str(s): [L.sid(j) for j in streams[s]] for s in SEEDS},
                            "first_n_base_equal_harness_sample": True, "load_checks": H.checks, "solve_device": str(H.solve_dev),
                            "pools": {p: len(L.pool(p)) for p in ("repair", "heldout", "heldout_disjoint", "heldout_strict")},
                            "hn_valid": len(L.rows("hn", valid=True)), "code": fx.code_fingerprint(), "prefix": fx.prefix.describe(name)["deployed"]})

    timing = get(name, "timing") or {}

    def clock(ph, t0, n_cells):
        timing[ph] = {"seconds": timing.get(ph, {}).get("seconds", 0.0) + time.time() - t0, "cells_run": timing.get(ph, {}).get("cells_run", 0) + n_cells}
        put(name, "timing", timing)

    # =================================================================== A: build the linear repairs (harness alone on the card)
    if "A" in phases:
        t0, k = time.time(), 0
        sys.path.insert(0, os.path.join(P12, "code"))
        import run_streams as RS                                        # P12's own functions; nothing of P12 is written
        while os.path.join(P12, "code") in sys.path:                    # P12's directory holds modules called render / aggregate / status: off the path again
            sys.path.remove(os.path.join(P12, "code"))
        if RS.repair is not hz.pkg.repair or RS.HM is not hz.HM:
            raise RuntimeError("P12's code is bound to another instance of the harness")
        RS.MYCACHE = P12CACHE                                           # per-complaint z* targets and rollback sets of THIS experiment
        fh = open(os.path.join(LOGD, f"harness_{name}.log"), "a")
        tc, rc = RS.TargetCache(H, fh), RS.RollbackCache(H, fh)
        for method in LIN:
            for n, s in cells_plan():
                c = cfg(method, n, s)
                if done(name, "build__" + c) or (limit is not None and k >= limit):
                    continue

                def build(method=method, n=n, s=s, c=c):
                    cs = hstream[s][:nval(n, s)]
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    t1 = time.time()
                    corr, cost, extra, rank = RS.make(H, method, cs, tc, rc, fh)
                    wall = time.time() - t1
                    RS.save_corr(corr, os.path.join(CORR, name, c + ".pt"))
                    fro = corr.fro(H)
                    put(name, "build__" + c, {"log": name, "config": c, "method": method, "n": n, "n_reported": len(cs), "seed": s, "options": RS.OPTIONS[method],
                                              "reported_groups": [x["group"] for x in cs], "reported_sids": [L.sid(j) for j in seqs(n, s)],
                                              "correction_kind": corr.kind, "fro": fro, "fro_rel_to_log": fro / max(float(H.D_act.norm()), 1e-30),
                                              "cost": cost, "meta": RS.meta_clean(corr.meta), "extra": extra, "rank": rank, "repair_wall_sec": wall,
                                              "peak_gpu_bytes_repair": torch.cuda.max_memory_allocated(), "solve_device": str(H.solve_dev),
                                              "time": time.strftime("%Y-%m-%d %H:%M:%S")})
                    say(f"  A {c}: kind {corr.kind} fro/log {fro / max(float(H.D_act.norm()), 1e-30):.3f} rank {rank.get('correction')} wall {wall:.0f}s "
                        f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB")
                    del corr
                if attempt(name, "build__" + c, build):
                    k += 1
        for key in [key for key in H._setup if key in ("evalcache", "baseline") or (isinstance(key, tuple) and key and key[0] in ("reedit", "basis"))]:
            H._setup.pop(key, None)                                     # the harness's cached solver states are not needed any more
        H.reset()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        fh.close()
        clock("A", t0, k)
        say(f"phase A done: {k} corrections built")

    # =================================================================== the driver session
    ses = fx.behave.session(L, H)
    edit_rows = [j for m in ("rel", "tgen", "igen") for j in L.edits_of(m)]
    hd_rows = L.pool("heldout_disjoint")
    rp_rows = L.pool("repair")
    base_rows = edit_rows + hd_rows
    dep = fx.methods.deployed(L)
    if not done(name, "eval__deployed"):
        rows = ses.tf(dep, base_rows + rp_rows)
        save_rows(name, "deployed", rows)
        save_rows(name, "preedit", ses.tf(fx.methods.preedit(L), base_rows + rp_rows))
        nc = sum(bool(r["D1"]) for r in rows if r["kind"] == "probe" and r["pool"] == "repair")
        if nc != len(streams[0]):
            raise RuntimeError(f"REPAIR complaints under the deployed weight: {nc} evaluated, {len(streams[0])} in the stream")
        put(name, "eval__deployed", {"log": name, "n_rows": len(rows), "weight_sig": dep.rec.get("weight_sig"), "n_repair_complaints": nc,
                                     "src": sorted({r["src"] for r in rows})})
        say(f"deployed rows: {len(rows)} ({sorted({r['src'] for r in rows})})")

    # =================================================================== B: re-score the saved corrections through the driver
    if "B" in phases:
        t0, k = time.time(), 0
        Correction = hz.HL.Correction
        for method in LIN:
            for n, s in cells_plan():
                c = cfg(method, n, s)
                if done(name, "eval__" + c) or not done(name, "build__" + c) or (limit is not None and k >= limit):
                    continue

                def score(method=method, n=n, s=s, c=c):
                    t1 = time.time()
                    b = torch.load(os.path.join(CORR, name, c + ".pt"), map_location="cpu")
                    rows_eval = base_rows + seqs(n, s)
                    rec = {"log": name, "config": c, "method": method, "n": n, "seed": s, "correction_kind": b["kind"]}
                    if b["kind"] == "adapter":
                        M = fx.methods.wrap(L, Correction("adapter", Lf=b["Lf"].to(H.dev), Rf=b["Rf"].to(H.dev), gate=b["gate"], meta={"e2p": c}), LIN_LABEL[method],
                                            rec={"config": c}, name="linear", n=nval(n, s), draw=s, variant=f"n{n}")
                    elif b["kind"] == "weight":
                        if b["W"].dtype != H.W0.dtype or b["W"].shape != H.W0.shape:
                            raise RuntimeError("saved weight of another dtype / shape")
                        M = fx.methods.wrap(L, b["W"].to(H.dev), LIN_LABEL[method], rec={"config": c}, name="linear", n=nval(n, s), draw=s, variant=f"n{n}")
                    else:                                               # no correction was applied (e.g. no rollback needed): the deployed log
                        M = dep
                        rec["note"] = "the harness returned no correction: the repaired model is the deployed log"
                    rows = ses.tf(M, rows_eval)
                    save_rows(name, c, rows)
                    rec.update(kind=M.kind, digest=M.digest, weight_sig=M.rec.get("weight_sig"), n_rows=len(rows), seconds=time.time() - t1,
                               src=sorted({str(r.get("src")) for r in rows}))
                    put(name, "eval__" + c, rec)
                    hdr = [r for r in rows if r["kind"] == "probe" and r["pool"] != "repair"]
                    say(f"  B {c}: {M.kind} D1 hd {100 * np.mean([r['D1'] for r in hdr]):.1f} Rel {100 * np.mean([r['exact'] for r in rows if r.get('metric') == 'rel']):.0f} "
                        f"({time.time() - t1:.0f}s)")
                    del M, b
                    ses.release()
                    torch.cuda.empty_cache()
                if attempt(name, "eval__" + c, score):
                    k += 1
        clock("B", t0, k)
        say(f"phase B done: {k} corrections re-scored")

    # =================================================================== C: per-row maxima over all positions (ReScope-W coordinates)
    qp = os.path.join(QDIR, name + ".pt")
    if "C" in phases and not os.path.exists(qp):
        t0 = time.time()
        S, _ = fx.methods._where_base(L)
        if S.D != L.T:
            raise RuntimeError("one detector per edit is expected")
        Sg = fx.methods._slim(S).to(H.dev)
        rows_all = L.rows("probe", valid=True) + edit_rows + L.rows("hn", valid=True)
        out = []
        with torch.no_grad(), ses.ev.hz.HE.vision_cache(H):
            for i, j in enumerate(rows_all):
                ses.ev.trim_vcache()
                out.append(Sg.query_max(ses.ev.eligible_keys(ses.item(j), Sg)).cpu())
                if i % 500 == 0:
                    say(f"  C capture {i}/{len(rows_all)}")
        Q = torch.cat(out)
        tmp = qp + f".tmp{os.getpid()}"
        torch.save({"log": name, "rows": [int(j) for j in rows_all], "sids": [L.sid(j) for j in rows_all], "Q": Q,
                    "what": "Q[i, j] = largest normalised coordinate of edit j over ALL positions of the teacher-forced sequence of row i "
                            "(prompt and pre-edit answer; edit sequences: prompt and target), fp64, gate on the model's device"}, tmp)
        os.replace(tmp, qp)
        del Sg
        torch.cuda.empty_cache()
        clock("C", t0, 1)
        say(f"phase C done: {tuple(Q.shape)} in {time.time() - t0:.0f}s")
    QD = torch.load(qp, map_location="cpu") if os.path.exists(qp) else None
    qpos = {int(j): i for i, j in enumerate(QD["rows"])} if QD is not None else None

    def where_cell(M, cname, rows_eval, extra):
        """Behaviour of a ReScope-W gate (library path, every row a forward) + gate level from the captured maxima + hook check."""
        t1 = time.time()
        rows = ses.tf(M, rows_eval)
        save_rows(name, cname, rows)
        gl = where_gate_level(L, QD["rows"], QD["Q"], M.obj.theta(), per_row=True)
        fr = gl.pop("_fired")
        diff = [r["csid"] for r, j in zip(rows, rows_eval) if bool(r["fired"]) != fr[int(j)]]
        rec = dict(extra, log=name, cell=cname, kind=M.kind, digest=M.digest, n_rows=len(rows), seconds=time.time() - t1, gate_level=gl,
                   theta=[float(x) for x in M.obj.theta()], n_capped=fx.methods.n_capped(M), cal={k: v for k, v in (M.rec.get("cal") or {}).items() if k not in ("block1", "block2")},
                   hook_vs_gate_level={"n_rows": len(rows), "n_differ": len(diff), "differing": diff[:20]})
        put(name, "eval__" + cname, rec)
        hdr = [r for r in rows if r["kind"] == "probe" and r["pool"] != "repair"]
        say(f"  {cname}: D1 hd {100 * np.mean([r['D1'] for r in hdr]):.1f} fired hd {100 * np.mean([bool(r['fired']) for r in hdr]):.1f} "
            f"Rel {100 * np.mean([r['exact'] for r in rows if r.get('metric') == 'rel']):.0f} hook!=gate {len(diff)} ({time.time() - t1:.0f}s)")
        ses.release()

    # =================================================================== D: thresholds-only repair (log-only start + least repair)
    if "D" in phases and QD is not None:
        t0, k = time.time(), 0
        Mlo = fx.methods.where_logonly(L, H)
        if not done(name, "eval__where_logonly"):
            if attempt(name, "eval__where_logonly", lambda: where_cell(Mlo, "where_logonly", base_rows, {"method": "where_logonly", "n": 0, "seed": None,
                                                                                                       "own_prompts": Mlo.rec.get("own_prompts")})):
                k += 1
        for n, s in cells_plan():
            c = cfg("WHERE_LEAST", n, s)
            if done(name, "eval__" + c) or (limit is not None and k >= limit):
                continue

            def least(n=n, s=s, c=c):
                cs = seqs(n, s)
                M = fx.methods.where_least(L, cs, H, start="logonly", seed=s, label="where_least")
                ref = M.obj.theta().clone()
                refused = set(M.rec["least_repair"]["refused_sids"])
                rng = np.random.default_rng([int(s), 77, nval(n, s)])
                oc = {"n_orders": 0, "n_equal_thresholds": 0, "n_equal_refused_sets": 0, "max_abs_diff": 0.0}
                for _ in range(N_ORDERS if len(cs) > 1 else 0):
                    Mo = fx.methods.where_least(L, [cs[int(i)] for i in rng.permutation(len(cs))], H, start="logonly", seed=s, label="where_least_order")
                    th = Mo.obj.theta()
                    eq = bool(torch.equal(th, ref))
                    fin = torch.isfinite(th) & torch.isfinite(ref)
                    oc["n_orders"] += 1
                    oc["n_equal_thresholds"] += int(eq)
                    oc["n_equal_refused_sets"] += int(set(Mo.rec["least_repair"]["refused_sids"]) == refused)
                    if not eq and bool(fin.any()):
                        oc["max_abs_diff"] = max(oc["max_abs_diff"], float((th[fin] - ref[fin]).abs().max()))
                    del Mo
                where_cell(M, c, base_rows + cs, {"method": "where_least", "start": "logonly", "n": n, "n_reported": len(cs), "seed": s,
                                                  "least_repair": M.rec["least_repair"], "own_prompts": M.rec.get("own_prompts"), "order_check": oc,
                                                  "reported_sids": [L.sid(j) for j in cs]})
            if attempt(name, "eval__" + c, least):
                k += 1
        clock("D", t0, k)
        say(f"phase D done: {k} cells")

    # =================================================================== E: blocklist line
    if "E" in phases:
        t0, k = time.time(), 0
        for n, s in cells_plan():
            c = cfg("BLOCKLIST", n, s)
            if done(name, "eval__" + c):
                continue
            cs = seqs(n, s)
            rows = ses.blocklist(cs, base_rows + cs)
            save_rows(name, c, rows)
            put(name, "eval__" + c, {"log": name, "cell": c, "method": "blocklist", "n": n, "n_reported": len(cs), "seed": s, "n_rows": len(rows),
                                     "n_blocked_rows": sum(bool(r["blocked"]) for r in rows),
                                     "n_blocked_heldout_disjoint": sum(bool(r["blocked"]) for r in rows if r["kind"] == "probe" and r["pool"] != "repair")})
            k += 1
        clock("E", t0, k)
        say(f"phase E done: {k} cells")

    # =================================================================== F: the default method, draws 0 - 2
    if "F" in phases:
        t0, k = time.time(), 0
        for d in DRAWS_B:
            c = f"RESCOPE__d{d}"
            if done(name, "eval__" + c):
                continue
            M = fx.methods.rescope(L, alpha=ALPHA, n=N_CAL, draw=d)
            t1 = time.time()
            rows = ses.tf(M, base_rows)
            save_rows(name, c, rows)
            put(name, "eval__" + c, {"log": name, "cell": c, "method": "rescope", "alpha": ALPHA, "n": N_CAL, "draw": d, "digest": M.digest, "n_capped": fx.methods.n_capped(M),
                                     "own_prompts": M.rec.get("own_prompts"), "cal": {k_: v for k_, v in M.rec["cal"].items() if k_ not in ("block1", "block2")},
                                     "n_rows": len(rows), "seconds": time.time() - t1, "src": sorted({str(r.get("src")) for r in rows})})
            k += 1
            say(f"  F {c}: capped {fx.methods.n_capped(M)} fired hd {100 * np.mean([bool(r['fired']) for r in rows if r['kind'] == 'probe']):.1f} ({time.time() - t1:.0f}s)")
        clock("F", t0, k)

    # =================================================================== G: ReScope-W with two-block thresholds (R24)
    if "G" in phases and QD is not None:
        t0, k = time.time(), 0
        S, grec = fx.methods._where_base(L)

        def two_block(d, allow_silence=False):
            dr = fx.draws.calibration(L, N_CAL, d)                       # the driver's calibration draw of the REPAIR pool (the default method's)
            fx.pools.guard(L, dr.rows, "calibration", None)
            Qc = QD["Q"][[qpos[int(j)] for j in dr.rows]]
            S2, crec = fx.methods._cal_two_block(S, Qc, ALPHA, dr.seed, (1, 1), "max", allow_silence, "excess")   # = rescope.calibrate_two_block(S, coords=Qc, ...)
            return S2, crec, dr

        if not done(name, "gate__where_two_block"):
            per = {}
            for d in range(fx.draws.N_DRAWS):
                S2, crec, dr = two_block(d)
                S3, lrec, _ = two_block(d, allow_silence=True)
                gl = where_gate_level(L, QD["rows"], QD["Q"], S2.theta())
                gll = where_gate_level(L, QD["rows"], QD["Q"], S3.theta())
                per[d] = {"gate_level": gl, "n_capped": crec.get("n_edits_capped"), "capped_edits": crec.get("capped_edits"), "eta": crec.get("eta"),
                          "bound_false_firing": crec.get("bound_false_firing"), "literal_no_guards": {"heldout_disjoint_fired": gll["probe"]["heldout_disjoint"]["fired"],
                                                                                                     "heldout_strict_fired": gll["probe"]["heldout_strict"]["fired"],
                                                                                                     "rel_own_released": gll["edit"]["rel"]["own_released"],
                                                                                                     "n_own_keys_silenced": lrec.get("n_own_keys_silenced")},
                          "calibration_sids_sha1": __import__("hashlib").sha1(" ".join(L.sid(j) for j in dr.rows).encode()).hexdigest()[:12]}
            put(name, "gate__where_two_block", {"log": name, "method": "where_two_block", "alpha": ALPHA, "n": N_CAL, "draws": per, "add_guards": grec,
                                                "what": "gate level from the captured per-row maxima (teacher-forced sequences, all positions)"})
        for d in DRAWS_B:
            c = f"WHERE_TB__d{d}"
            if done(name, "eval__" + c):
                continue

            def tb(d=d, c=c):
                S2, crec, dr = two_block(d)
                M = fx.methods.wrap(L, S2, "where_two_block", rec={"cal": crec, "draw": dr.rec, "construction": "library two-block rule on the per-row maxima over all positions"},
                                    name="where_two_block", alpha=ALPHA, n=N_CAL, draw=d)
                extra = {"method": "where_two_block", "alpha": ALPHA, "n": N_CAL, "draw": d, "own_prompts": M.rec.get("own_prompts")}
                if d == 0:                                              # the driver's own constructor (calibration coordinates captured by fx.methods.calib_coords)
                    Md = fx.methods.rescope(L, role="all", fire="all", response="lin", H=H, alpha=ALPHA, n=N_CAL, draw=0, label="where_two_block_driver")
                    extra["equals_driver_constructor"] = {"thresholds_bitwise": bool(torch.equal(Md.obj.theta(), M.obj.theta())), "digest_equal": Md.digest == M.digest,
                                                         "max_abs_diff": float((Md.obj.theta() - M.obj.theta()).abs().nan_to_num(0.0, posinf=0.0, neginf=0.0).max())}
                    del Md
                where_cell(M, c, base_rows, extra)
            if attempt(name, "eval__" + c, tb):
                k += 1
        clock("G", t0, k)

    # =================================================================== H: D2 (rate of the pre-correct heldout_disjoint rows; R7), seed 0 / draw 0, n = 10
    if "H" in phases:
        t0, k = time.time(), 0
        Correction = hz.HL.Correction
        # the round-off floor of ReScope-W (E2 b): all thresholds at minus infinity = the deployed log without its write round-off term
        if not done(name, "eval__where_floor"):
            def floor():
                t1 = time.time()
                M = fx.methods.where_least(L, [], H, start="deployed", label="where_floor")
                rows = ses.tf(M, base_rows)
                save_rows(name, "where_floor", rows)
                put(name, "eval__where_floor", {"log": name, "cell": "where_floor", "n_rows": len(rows), "seconds": time.time() - t1,
                                                "n_thresholds_minus_inf": int(torch.isneginf(M.obj.theta()).sum()), "n_thresholds": int(M.obj.theta().numel()),
                                                "what": "ReScope-W with every threshold at minus infinity (linear response, logged values, no round-off term)"})
                ses.release()
            attempt(name, "eval__where_floor", floor)

        def d2_cell(cname, make):
            nonlocal k
            if done(name, "d2__" + cname) or (limit is not None and k >= limit):
                return

            def run():
                t1 = time.time()
                M = make()
                if M is None:
                    return
                rows = ses.d2(M)
                save_rows(name, "d2__" + cname, rows)
                put(name, "d2__" + cname, {"log": name, "cell": cname, "n_rows": len(rows), "n_D2": sum(bool(r.get("D2")) for r in rows), "seconds": time.time() - t1,
                                           "what": "48-token decode (decoder v2) of the pre-correct heldout_disjoint rows; D2 = the answer no longer contains the reference"})
                say(f"  H {cname}: D2 {sum(bool(r.get('D2')) for r in rows)}/{len(rows)} ({time.time() - t1:.0f}s)")
                ses.release()
                torch.cuda.empty_cache()
            if attempt(name, "d2__" + cname, run):
                k += 1

        def lin_method(c):
            if not done(name, "build__" + c):
                return None
            b = torch.load(os.path.join(CORR, name, c + ".pt"), map_location="cpu")
            if b["kind"] == "adapter":
                return fx.methods.wrap(L, Correction("adapter", Lf=b["Lf"].to(H.dev), Rf=b["Rf"].to(H.dev), gate=b["gate"], meta={"e2p": c}), "linear", rec={"config": c}, name="linear")
            if b["kind"] == "weight":
                return fx.methods.wrap(L, b["W"].to(H.dev), "linear", rec={"config": c}, name="linear")
            return dep
        d2_cell("deployed", lambda: dep)
        for method in LIN:
            d2_cell(cfg(method, 10, 0), lambda method=method: lin_method(cfg(method, 10, 0)))
        if QD is not None:
            d2_cell("where_logonly", lambda: fx.methods.where_logonly(L, H))
            d2_cell(cfg("WHERE_LEAST", 10, 0), lambda: fx.methods.where_least(L, seqs(10, 0), H, start="logonly", seed=0, label="where_least"))

            def tb0():
                dr = fx.draws.calibration(L, N_CAL, 0)
                S0, _ = fx.methods._where_base(L)
                S2, crec = fx.methods._cal_two_block(S0, QD["Q"][[qpos[int(j)] for j in dr.rows]], ALPHA, dr.seed, (1, 1), "max", False, "excess")
                return fx.methods.wrap(L, S2, "where_two_block", rec={"cal": crec}, name="where_two_block", alpha=ALPHA, n=N_CAL, draw=0)
            d2_cell("WHERE_TB__d0", tb0)
        d2_cell("RESCOPE__d0", lambda: fx.methods.rescope(L, alpha=ALPHA, n=N_CAL, draw=0))
        clock("H", t0, k)
        say(f"phase H done: {k} cells")

    rec = ses.assert_clean()
    put(name, "clean__" + time.strftime("%Y%m%d_%H%M%S"), dict(rec, log=name, phases=phases, stats=ses.stats, memo=ses.memo.summary() if hasattr(ses.memo, "summary") else None,
                                                             peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.time() - T00, code=fx.code_fingerprint()))
    say(f"job finished clean: {rec}; {time.time() - T00:.0f}s")


if __name__ == "__main__":
    main()
