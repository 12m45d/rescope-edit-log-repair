"""E5 task 2 (optimum), one log: certified optimum of the linear surrogate of MIN-ROLLBACK with the EXISTING Track C code
(07_repair_v3/rollback_opt/code: run_ropt.solve_one -> ropt.exhaustive_min_resid (all |R| <= 4), ropt.qcr_delta,
ropt.kelley_root, ropt.milp_max_kept (HiGHS, outer approximation), ropt.local_search), unchanged; only the tolerances
(run_ropt.TAUS = 0.1, 0.3, 0.5), the complaint set, the key definition and the MILP time limit are set here.

usage: python e5_opt.py --log TB-Q100-ae-l2_3 [--workers 8] [--budget-cpu-min 40] [--tl 10]
       -> out/opt_rows_<log>.jsonl (one line per (complaint, key definition); resumable), out/opt_meta_<log>.json

Budget    at most `budget-cpu-min` minutes of CPU TIME (core-minutes) per log, summed over all worker processes and over
          every earlier run of this log (out/opt_meta_<log>.json, out/opt_stop_<log>.json).  A job is submitted only if
          the CPU seconds already spent plus the worst case of every running job and of the new one (WORST_JOB_S each)
          stay inside the budget; INIT_S per worker is reserved for loading the log.
Order     seeded (e5_common.OPT_SEED): the image complaints and the text complaints are permuted separately and
          interleaved (image, text, image, ...).  Jobs: every complaint with the key definition 'last' in that order,
          then, with what is left of the budget, the same order with the key definition 'stacked'.  The complaints that
          are reached are therefore a seeded random subsample, stratified by probe type.
Output    per job the dictionary of run_ropt.solve_one (per tolerance: fo = Track C's static first-order greedy, bi,
          ls = local search, exhaustive_opt, kelley, milp, opt_lb <= OPT <= opt_ub, certified, how) plus group, keydef,
          order index, wall and CPU seconds.  A complaint with ONE answer position has the same instance under both
          key definitions: its 'stacked' row is a copy of its 'last' row (copied_from_last = true, no CPU).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import signal
import sys
import time
from collections import deque

import numpy as np
import torch

import e5_common as C

sys.path.insert(0, os.path.join(C.ROPT, "code"))
import run_ropt  # noqa: E402  (read-only Track C driver; nothing is written by importing or calling it)

WORST_JOB_S = 3 * (C.OPT_TL + 14.0) + 8.0        # 3 tolerances x (MILP limit + Kelley root of 600 LPs) + exhaustive / QCR
INIT_S = C.OPT_INIT_S                            # CPU seconds reserved per worker for loading the log


def opt_order(comps, seed=C.OPT_SEED):
    rng = np.random.default_rng(seed)
    perm = {}
    for p in C.PTYPES:
        rows = [c for c in comps if c["ptype"] == p]
        perm[p] = [rows[int(i)] for i in rng.permutation(len(rows))]
    out = []
    for k in range(max(len(v) for v in perm.values())):
        for p in C.PTYPES:
            if k < len(perm[p]):
                out.append(perm[p][k])
    return out


def _init(name):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    torch.set_num_threads(1)
    run_ropt.TAUS = C.TOLS
    run_ropt._init(name)


def job(args):
    idx, group, kd, row, tl = args
    t0, c0 = time.time(), time.process_time()
    try:
        o = run_ropt.solve_one((row, tl))
    except Exception as e:                                   # an assertion of the Track C code: recorded, not hidden
        o = {"error": f"{type(e).__name__}: {e}"}
    o.update(group=group, keydef=kd, order_idx=idx, A_keydef=int(row["K"].shape[0]), wall=time.time() - t0,
             cpu=time.process_time() - c0, tl=tl, copied_from_last=False)
    return o


class Stop(Exception):
    pass


def _term(signum, frame):
    raise Stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--workers", type=int, default=C.OPT_WORKERS)
    ap.add_argument("--budget-cpu-min", type=float, default=C.OPT_BUDGET_MIN)
    ap.add_argument("--tl", type=float, default=C.OPT_TL)
    ap.add_argument("--limit", type=int, default=0, help="debug: at most this many jobs")
    ap.add_argument("--tag", default="", help="debug: suffix of the output files")
    a = ap.parse_args()
    C.check_log(a.log)
    run_ropt.TAUS = C.TOLS
    fh = open(os.path.join(C.LOGDIR, f"opt_{a.log}{a.tag}.log"), "a")
    outp = os.path.join(C.OUT, f"opt_rows_{a.log}{a.tag}.jsonl")
    metap = os.path.join(C.OUT, f"opt_meta_{a.log}{a.tag}.json")
    stopp = os.path.join(C.OUT, f"opt_stop_{a.log}{a.tag}.json")
    comps = C.complaints(a.log)
    order = opt_order(comps)
    d = torch.load(os.path.join(C.ROPT, "cache", f"keys_{a.log}_v2.pt"), map_location="cpu")
    by = {(r["probe_idx"], r["ptype"]): r for r in d["rows"]}
    done = {}
    if os.path.exists(outp):
        for o in C.jsonl_read(outp):
            done[(o["group"], o["keydef"])] = o
    meta = C.jload(metap) if os.path.exists(metap) else {"runs": []}
    spent = sum(r["cpu_s_children"] for r in meta["runs"]) + (C.jload(stopp)["cpu_s_total"] if os.path.exists(stopp) else 0.0)
    budget_s = a.budget_cpu_min * 60.0 - spent - INIT_S * a.workers
    jobs = deque()
    for kd in C.KEYDEFS:
        for idx, c in enumerate(order):
            if (c["group"], kd) in done:
                continue
            row = dict(by[(c["probe_idx"], c["ptype"])])
            if kd == "last":
                row["K"] = row["K"][:1]
            jobs.append((idx, c["group"], kd, row, a.tl))
    if a.limit:
        jobs = deque(list(jobs)[: a.limit])
    C.log(f"{a.log}: {len(comps)} complaints, {len(done)} jobs done, {len(jobs)} to do; workers {a.workers}, MILP limit {a.tl}s, "
          f"CPU budget {a.budget_cpu_min:.0f} min, already spent {spent / 60:.1f} min, left for jobs {budget_s / 60:.1f} min "
          f"(worst job {WORST_JOB_S:.0f}s), tolerances {run_ropt.TAUS}", fh)
    meta.update(log=a.log, seed=C.OPT_SEED, tols=list(C.TOLS), definition=__doc__, order=[c["group"] for c in order],
                n_complaints=len(comps), worst_job_s=WORST_JOB_S, init_s_per_worker=INIT_S,
                versions={"numpy": np.__version__, "scipy": __import__("scipy").__version__, "torch": torch.__version__})
    if budget_s < WORST_JOB_S:
        C.log(f"{a.log}: no CPU budget left; nothing is run", fh)
        meta["n_done"] = {kd: sum(1 for k in done if k[1] == kd) for kd in C.KEYDEFS}
        C.jdump(meta, metap)
        return
    t0 = time.time()
    running, n_new, cpu_jobs, n_copied, interrupted = {}, 0, 0.0, 0, False
    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_init, initargs=(a.log,))
    signal.signal(signal.SIGTERM, _term)
    try:
        with open(outp, "a") as fo:
            while True:
                while jobs and len(running) < a.workers and cpu_jobs + (len(running) + 1) * WORST_JOB_S <= budget_s:
                    j = jobs.popleft()
                    if j[2] == "stacked" and j[3]["K"].shape[0] == 1 and (j[1], "last") in done:
                        o = dict(done[(j[1], "last")], keydef="stacked", copied_from_last=True, wall=0.0, cpu=0.0)
                        fo.write(json.dumps(C.clean(o)) + "\n")
                        fo.flush()
                        done[(j[1], "stacked")] = o
                        n_copied += 1
                        continue
                    running[(j[1], j[2])] = pool.apply_async(job, (j,))
                for key, res in list(running.items()):
                    if res.ready():
                        o = res.get()
                        fo.write(json.dumps(C.clean(o)) + "\n")
                        fo.flush()
                        done[key] = o
                        del running[key]
                        n_new += 1
                        cpu_jobs += o["cpu"]
                        if n_new % 10 == 1 and "tau" in o:
                            s = o["tau"]
                            C.log(f"  [{n_new}] {time.time() - t0:.0f}s {o['group']} {o['keydef']} ({o['cpu']:.0f} CPU-s; total {cpu_jobs:.0f}): "
                                  + "  ".join(f"c{t}: bi{s[t]['bi']} OPT[{s[t]['opt_lb']},{s[t]['opt_ub']}]" for t in s), fh)
                        if "error" in o:
                            C.log(f"  ERROR {o['group']} {o['keydef']}: {o['error']}", fh)
                if not running and (not jobs or cpu_jobs + WORST_JOB_S > budget_s):
                    break
                time.sleep(0.25)
        pool.close()
    except Stop:
        interrupted = True
        pool.terminate()
    pool.join()
    wall = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta["runs"].append({"start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "wall_s": wall, "workers": a.workers,
                         "tl": a.tl, "budget_cpu_min": a.budget_cpu_min, "cpu_s_spent_before": spent, "n_jobs_run": n_new,
                         "n_jobs_copied": n_copied, "n_jobs_not_reached": len(jobs), "cpu_s_jobs": cpu_jobs,
                         "cpu_s_children": ru.ru_utime + ru.ru_stime, "interrupted": interrupted,
                         "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")})
    meta["n_done"] = {kd: sum(1 for k in done if k[1] == kd) for kd in C.KEYDEFS}
    C.jdump(meta, metap)
    C.log(f"{a.log}: {n_new} jobs run (+{n_copied} copied) in {wall:.0f}s wall, {cpu_jobs:.0f} CPU-s in jobs "
          f"({ru.ru_utime + ru.ru_stime:.0f} CPU-s of all workers incl. start); done {meta['n_done']}; not reached {len(jobs)}"
          + ("; INTERRUPTED" if interrupted else ""), fh)


if __name__ == "__main__":
    main()
