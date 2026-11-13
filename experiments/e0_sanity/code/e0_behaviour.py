"""E0 (2): behaviour of the methods for calibration draws 0, 1, 2 on ONE development log.  GPU, through the shared queue:

  GQ=$RESCOPE_ROOT/02_research/tools/gpuq/gq
  nohup $GQ run --mem 10500 --name e0_beh_Q -- python e0_behaviour.py TB-Q100-ae-l2_3 > ../logs/behaviour_TB-Q100-ae-l2_3.log 2>&1 &
  nohup $GQ run --mem 17000 --name e0_beh_L -- python e0_behaviour.py TB-L100-ae-l2_3 > ../logs/behaviour_TB-L100-ae-l2_3.log 2>&1 &

The gates are built exactly as docs/USAGE.md builds the default method (library calls only): from_harness(H).to(H.dev),
feature_gate, add_guards(own B = 1 Rel keys), a seeded draw of n = 200 valid REPAIR probes, calibrate_two_block / calibrate_shared
on their teacher-forced keys; evaluation by rescope.SparseEvaluator (teacher forcing, B = 1, lp = 'tf').
Method sets: 'static' = deployed log, pre-edit model (control), M5, M6;  'draw<d>' = M1 .. M4 calibrated on draw d.
Item groups: edits (Rel / T-Gen / I-Gen of all 100 edits), probes (valid rows of HELD-OUT-DISJOINT; HELD-OUT-STRICT is the subset
with strict = True), hn (valid rows of the final hard-negative set for the log's edits: HN-I, HN-Q, HN-X of the first 100 edits).
Resumable: one JSON per (method set, group, chunk of items) under out/behaviour/<log>/; finished parts are skipped on restart.
The thresholds of every gate are stored (gates_<set>.json).  A restart re-builds every gate and compares it with the stored one:
bitwise equal, or (another card: last-bit differences of the GPU products) equal to 1e-12, in which case the STORED thresholds are
installed so that all parts of a set come from one gate; anything else stops the run.
A restart of a COMPLETE run (done.json exists) runs no part again: it re-evaluates the first 40 items of every group and method
set and compares the rows with the stored ones (restart_check.json: reproducibility of the gates and of the rows).
"""
from __future__ import annotations

import os
import sys
import time

import e0_common as C

import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import HN_DIR_FINAL, SparseEvaluator, add_guards, from_harness, from_testbed, load, splits  # noqa: E402

CHUNK = 250
SPOT = 40                      # items per group re-evaluated by a restart of a complete run


def main(name: str, limit: int | None = None):
    C.check_log(name)
    t0 = time.time()
    mem = int(os.environ.get("GQ_MEM", C.LOGS[name]["gq_mem"]))
    cap_gb = (mem - 600) / 1024.0                                    # README of the queue: cap the process at GQ_MEM - 600 MiB
    odir = os.path.join(C.OUT, "behaviour" if limit is None else "behaviour_smoke", name)
    os.makedirs(odir, exist_ok=True)
    complete = os.path.exists(os.path.join(odir, "done.json"))        # a restart of a complete run: checks only
    C.say(f"{name}: GQ_MEM {mem} MiB, cap {cap_gb:.2f} GB, CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')}, out {odir}")
    H = load(name, device="cuda:0", mem_gb=cap_gb)
    ev = SparseEvaluator(H, cache="auto", hn_dir=HN_DIR_FINAL)
    if ev.cache is None or not ev.cache.is_final:
        raise SystemExit("the final key cache of this backbone is required")
    log = from_harness(H).to(H.dev)
    Lt = from_testbed(name)
    same_log = {k: bool(torch.equal(getattr(log, k).cpu(), getattr(Lt, k))) for k in ("U", "W", "K", "batch")}
    groups = {"edits": ev.edit_items(), "probes": ev.probe_items("heldout_disjoint"), "hn": ev.hn_items()}
    if limit is not None:                                             # smoke run: every `limit`-th item (never rendered)
        groups = {k: v[::limit] for k, v in groups.items()}
    repair = ev.probe_items("repair")
    own = ev.own_keys("rel")                                          # B = 1 keys of the edits' own Rel prompts (from the key cache)
    C.say(f"loaded in {time.time() - t0:.0f}s: T = {H.T}, checks {H.checks}, items {({k: len(v) for k, v in groups.items()})}, REPAIR pool "
          f"{len(repair)}, own keys {tuple(own[0].shape)}, from_harness == from_testbed {same_log}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    ver = ev.verify_cache(groups["edits"] + groups["probes"] + groups["hn"], n=24, seed=0)      # fresh capture of 24 sequences vs the cache
    C.say(f"verify_cache: {ver}")
    meta = {"log": name, "T": int(H.T), "alpha": C.ALPHA, "n_cal": C.N_CAL, "draws": list(C.BEHAVIOUR_DRAWS), "gq_mem": mem, "cap_gb": cap_gb,
            "load_checks": H.checks, "from_harness_equals_from_testbed": same_log, "n_items": {k: len(v) for k, v in groups.items()},
            "n_repair_pool": len(repair), "verify_cache": ver, "hn_dir": HN_DIR_FINAL, "cache_dir": ev.cache.dir, "code": C.code_fingerprint(),
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "limit": limit, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.jdump(meta, os.path.join(odir, "restart_meta.json" if complete else "meta.json"), indent=1)
    check = {"gates": {}, "rows": {}, "n_rows": 0, "n_rows_equal": 0, "differing": []}

    def calibrated(mid, seed):
        S = C.base_gate(log, C.METHODS[mid]["feature"], W0=H.W0)
        S, _ = add_guards(S, *own)
        benign = splits.sample(repair, C.N_CAL, seed)
        S, rec = C.calibrate(S, mid, seed, queries=[ev.eligible_keys(it, S) for it in benign])
        rec["calibration_sids"] = [it["sid"] for it in benign]
        return S, rec

    sets = [("static", None)] + [(f"draw{d}", d) for d in C.BEHAVIOUR_DRAWS]
    for sname, seed in sets:
        if seed is None:
            methods = {"deployed": None, "pre-edit": H.W0, "M5": C.fixed_gate(log, "M5", W0=H.W0), "M6": C.fixed_gate(log, "M6", W0=H.W0)}
            grec = {m: {"tau": [float(x) for x in methods[m].theta()]} for m in C.FIXED}
        else:
            methods, grec = {}, {}
            for mid in C.CALIBRATED:
                methods[mid], rec = calibrated(mid, seed)
                g = methods[mid]
                grec[mid] = {"rec": rec, "tau": [float(x) for x in g.det_tau], "b": None if g.det_b is None else [float(x) for x in g.det_b]}
                C.say(f"  {sname} {mid}: eta {rec['eta']:.6f}, n1 {rec['n1']}, n2 {rec['n2']}, rank {rec['rank']}, capped edits {rec['n_edits_capped']}, "
                      f"guards silenced by the literal rule {rec['n_guards_silenced_by_literal_rule']}, bound {rec['bound_false_firing']:.4f}")
        gpath = os.path.join(odir, f"gates_{sname}.json")
        if os.path.exists(gpath):                                     # a restart must evaluate the very same gates
            old = C.jload(gpath)
            for m in grec:
                bitwise = C._clean(grec[m]["tau"]) == old[m]["tau"] and C._clean(grec[m].get("b")) == old[m].get("b")
                a, b = (torch.tensor([float(x) for x in v], dtype=torch.float64) for v in (grec[m]["tau"], old[m]["tau"]))
                fin = torch.isfinite(b)
                rel = float(((a[fin] - b[fin]).abs() / b[fin].abs().clamp_min(1e-300)).max()) if bool(fin.any()) and bool((torch.isfinite(a) == fin).all()) else float("inf")
                check["gates"][f"{sname}/{m}"] = {"bitwise": bool(bitwise), "max_rel_diff": 0.0 if bitwise else rel}
                if not bitwise:
                    if not rel <= 1e-12:
                        raise SystemExit(f"{sname} {m}: thresholds differ from the stored gate of the earlier run by {rel:.1e} ({gpath}); remove {odir} to start again")
                    kw = {"det_tau": b.to(methods[m].det_tau.device)}  # last-bit difference: evaluate the STORED gate
                    if old[m].get("b") is not None:
                        kw["det_b"] = torch.tensor([float(x) for x in old[m]["b"]], dtype=torch.float64, device=methods[m].det_tau.device)
                    methods[m] = methods[m].replace(**kw)
                    C.say(f"  {sname} {m}: thresholds rebuilt to {rel:.1e} of the stored gate (not bitwise): the stored thresholds are installed")
        else:
            C.jdump(grec, gpath)
        if complete:                                                  # reproducibility of the rows: the first SPOT items of every group
            for gname, items in groups.items():
                part = C.jload(os.path.join(odir, f"{sname}__{gname}__000.json"))
                new = C._clean(ev.tf_rows(methods, items[:SPOT]))
                for m in methods:
                    eq = [x == y for x, y in zip(new[m], part["rows"][m][:SPOT])]
                    check["n_rows"] += len(eq)
                    check["n_rows_equal"] += sum(eq)
                    check["rows"][f"{sname}/{gname}/{m}"] = [sum(eq), len(eq)]
                    check["differing"] += [f"{sname}/{m}/{r['sid']}" for r, e in zip(new[m], eq) if not e][:5]
            C.say(f"  {sname}: rows re-evaluated {check['n_rows_equal']}/{check['n_rows']} equal to the stored rows so far")
            continue
        for gname, items in groups.items():
            for ci, a in enumerate(range(0, len(items), CHUNK)):
                path = os.path.join(odir, f"{sname}__{gname}__{ci:03d}.json")
                if os.path.exists(path):
                    continue
                t1 = time.time()
                st0 = dict(ev.stats)
                rows = ev.tf_rows(methods, items[a:a + CHUNK])
                C.jdump({"set": sname, "group": gname, "chunk": ci, "first": a, "n": len(items[a:a + CHUNK]), "rows": rows,
                         "sparse_stats": {k: ev.stats[k] - st0[k] for k in st0}, "seconds": time.time() - t1}, path)
                C.say(f"  {sname} {gname} chunk {ci} ({len(items[a:a + CHUNK])} items, {len(methods)} methods): {time.time() - t1:.0f}s, "
                      f"forwards {ev.stats['forwards'] - st0['forwards']}, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    H.reset()
    done = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds_this_run": time.time() - t0, "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
            "module_weight_is_WE_after": bool(torch.equal(H.mod.weight.data, H.WE)), "sparse_stats_this_run": dict(ev.stats),
            "rescope": os.path.dirname(rescope.__file__)}
    if complete:
        check.update(done, n_gates=len(check["gates"]), n_gates_bitwise=sum(v["bitwise"] for v in check["gates"].values()), spot=SPOT,
                     cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), code=C.code_fingerprint(),
                     ok=bool(check["n_rows_equal"] == check["n_rows"] and all(v["bitwise"] or v["max_rel_diff"] <= 1e-12 for v in check["gates"].values())))
        C.jdump(check, os.path.join(odir, "restart_check.json"), indent=1)
        C.say(f"RESTART CHECK: gates bitwise {check['n_gates_bitwise']}/{check['n_gates']}, rows equal {check['n_rows_equal']}/{check['n_rows']}, ok {check['ok']}")
        return
    C.jdump(done, os.path.join(odir, "done.json"), indent=1)
    C.say(f"DONE {done}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
