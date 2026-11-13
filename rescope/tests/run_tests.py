"""Test runner of the rescope package.  Every test writes tests/out/<test>.json; the table is printed from them.

  python run_tests.py cpu                      CPU unit tests (synthetic logs + CPU checks on the real logs)
  python run_tests.py gpu --log TB-Q100-ae-l2_3 [--device cuda:0] [--mem_gb 11] [--only b1,b2,...] [--force] [--smoke]
                                               GPU tests on one log (resumable: finished tests are skipped);
                                               --smoke = reduced sizes, written to out/smoke_* (never shown in the table)
  python run_tests.py scale [--log TB-Q2000-ae-l2_3] [--device cuda:0] [--mem_gb 11] [--force]
                                               scale-safe use on the longest log in a fresh process (peak RAM, guards, lean loader)
  python run_tests.py table                    pass / fail table from the JSON files (also written to out/table.md)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))          # .../02_research/code
sys.path.insert(0, HERE)
os.makedirs(OUT, exist_ok=True)
LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SCALE_LOG = "TB-Q2000-ae-l2_3"


def fingerprint(folder, skip=()):
    """sha1 (12 hex digits) of the .py files of a folder: which code produced a result."""
    import hashlib
    h = hashlib.sha1()
    for f in sorted(os.listdir(folder)):
        if f.endswith(".py") and f not in skip:
            h.update(f.encode())
            with open(os.path.join(folder, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:12]


CODE = fingerprint(os.path.dirname(HERE))                            # the package, as loaded by this process
TESTS = {"cpu": fingerprint(HERE, skip=("t_table.py", "run_tests.py", "t_gpu.py", "t_gpu2.py")),      # t_core.py + t_ops.py + t_cal.py + t_final.py
         "gpu": fingerprint(HERE, skip=("t_table.py", "run_tests.py", "t_core.py", "t_ops.py", "t_cal.py", "t_final.py"))}  # t_gpu.py + t_gpu2.py


def jdump(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def say(msg, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def run_one(name, fn, path, fh=None, kind="cpu"):
    t0 = time.time()
    try:
        res = fn()
        res["error"] = None
    except (FileNotFoundError, NotImplementedError) as e:
        # [added for the public artifact] an input of this test is not part of the release (README, "What can be reproduced") or is not
        # unpacked / rebuilt yet: the test is reported as NOT RUN with the missing input, never as PASS
        res = {"pass": None, "error": None, "not_run": f"{type(e).__name__}: {e}"}
    except Exception:
        res = {"pass": False, "error": traceback.format_exc()}
        say(res["error"], fh)
    res.update(test=name, seconds=time.time() - t0, finished=time.strftime("%Y-%m-%d %H:%M:%S"), code=CODE, tests=TESTS[kind])
    jdump(res, path)
    say(f"{name}: {'NOT RUN (input not available: ' + res['not_run'] + ')' if res.get('not_run') else 'PASS' if res['pass'] else 'FAIL'} ({res['seconds']:.0f}s)", fh)
    return res


def cmd_cpu(args):
    import torch
    torch.set_num_threads(4)
    import t_cal
    import t_core
    import t_final
    import t_ops
    tests = dict(t_core.TESTS, **t_ops.TESTS, **t_cal.TESTS, **t_final.TESTS)
    for name, fn in tests.items():
        if args.only and not any(name.startswith(o) for o in args.only.split(",")):
            continue
        run_one(name, fn, os.path.join(OUT, f"cpu_{name}.json"))


def cmd_gpu(args):
    import torch
    torch.set_num_threads(4)
    import t_gpu
    import t_gpu2
    from rescope import evaluate as EV, loglib as LL
    pre = "smoke" if args.smoke else "gpu"
    fh = open(os.path.join(OUT, f"{pre}_{args.log}.log"), "a")
    say(f"==== gpu tests {vars(args)}", fh)
    names = ["b1_linear", "b2_off", "b345_pass", "b6_harness", "b7_decode", "b9_paths", "b10_own", "c_timing", "b8_usage", "b11_dual", "b12_tokloc", "b13_sparse", "b14_usage2"]
    todo = [n for n in names if (not args.only or any(n.startswith(o) for o in args.only.split(",")))
            and (args.force or args.smoke or not os.path.exists(os.path.join(OUT, f"gpu_{args.log}_{n}.json")))]
    say(f"to run: {todo}", fh)
    if not todo:
        return
    part = os.path.join(OUT, f"_partial_pass_{args.log}{'_lim60' if args.smoke else ''}.json")
    if (args.force or args.smoke) and os.path.exists(part):
        os.remove(part)                                              # a forced run starts the big pass again
    H = EV.load(args.log, device=args.device, mem_gb=args.mem_gb)
    ev = EV.Evaluator(H)
    log = LL.from_harness(H)
    say(f"loaded {args.log}: checks {H.checks}; hard negatives {len(ev.hn)}; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", fh)
    lf = lambda m: say(m, fh)                                                                    # noqa: E731
    sm = args.smoke
    fns = {"b1_linear": lambda: t_gpu.b1_linear(ev, log), "b2_off": lambda: t_gpu.b2_off(ev, log),
           "b345_pass": lambda: t_gpu.b345_pass(ev, log, OUT, 60 if sm else 0, lf, tag=f"{CODE}/{TESTS['gpu']}"), "b6_harness": lambda: t_gpu.b6_harness(ev, log, lf, smoke=sm),
           "b7_decode": lambda: t_gpu.b7_decode(ev, log, log_fn=lf, **(dict(n=6, n_edit=4) if sm else {})),
           "b9_paths": lambda: t_gpu.b9_paths(ev, log, lf, smoke=sm), "b10_own": lambda: t_gpu.b10_own(ev, log, lf),
           "c_timing": lambda: t_gpu.c_timing(ev, log, **(dict(reps=2, n_seq=10) if sm else {})),
           "b8_usage": lambda: t_gpu.b8_usage(ev, log, lf), "b11_dual": lambda: t_gpu2.b11_dual(ev, log, lf, smoke=sm),
           "b12_tokloc": lambda: t_gpu2.b12_tokloc(ev, log, lf, smoke=sm), "b13_sparse": lambda: t_gpu2.b13_sparse(ev, log, lf, smoke=sm),
           "b14_usage2": lambda: t_gpu2.b14_usage2(ev, log, lf)}
    for n in todo:
        res = run_one(n, fns[n], os.path.join(OUT, f"{pre}_{args.log}_{n}.json"), fh, kind="gpu")
        say("   " + json.dumps({k: v for k, v in res.items() if k not in ("rows", "regression", "iii", "iv", "error", "methods", "info", "criterion", "what")},
                               default=str)[:1800], fh)
        H.reset()
        say(f"   peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", fh)
    say("DONE", fh)


def cmd_scale(args):
    """Scale-safe use on the longest log, in THIS fresh process (peak resident memory is a property of the process)."""
    import torch
    torch.set_num_threads(4)
    import t_gpu2
    name = args.log if args.log != LOGS[0] else SCALE_LOG
    fh = open(os.path.join(OUT, f"scale_{name}.log"), "a")
    say(f"==== scale test {vars(args)}", fh)
    path = os.path.join(OUT, f"scale_{name}_d1_scale.json")
    if os.path.exists(path) and not args.force:
        say("already done (use --force)", fh)
        return
    res = run_one("d1_scale", lambda: t_gpu2.d1_scale(name, args.device, args.mem_gb or 11.0, lambda m: say(m, fh)), path, fh, kind="gpu")
    say("   " + json.dumps({k: v for k, v in res.items() if k not in ("criterion", "error", "registry", "stored")}, default=str)[:3000], fh)
    say("DONE", fh)


def cmd_table(args):
    import t_table
    txt = t_table.render(OUT, CODE, TESTS)
    print(txt)
    with open(os.path.join(OUT, "table.md"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cpu", "gpu", "scale", "table"))
    ap.add_argument("--log", default=LOGS[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--mem_gb", type=float, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"cpu": cmd_cpu, "gpu": cmd_gpu, "scale": cmd_scale, "table": cmd_table}[a.cmd](a)
    if a.cmd != "table" and not a.smoke:
        cmd_table(a)                                                 # every run ends with the table of ALL results on disk
