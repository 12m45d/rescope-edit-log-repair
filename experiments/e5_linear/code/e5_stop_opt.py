"""Stop a running e5_opt.py of one log and record what it had spent:  python e5_stop_opt.py --log <log> --reason "..."
-> out/opt_stop_<log>.json (CPU seconds of the main process and of every worker, read from ps before the signal).

Only processes of the calling user whose command line is `e5_opt.py --log <log>` (and the run_opt.sh launcher, so that
it does not start the next log) receive SIGTERM.  The rows that finished stay in out/opt_rows_<log>.jsonl.
"""
from __future__ import annotations

import argparse
import getpass
import os
import signal
import subprocess
import time

import e5_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--reason", required=True)
    a = ap.parse_args()
    C.check_log(a.log)
    ps = subprocess.run(["ps", "-u", getpass.getuser(), "-o", "pid=,ppid=,cputimes=,etimes=,args="], capture_output=True, text=True).stdout
    procs, launch = [], []
    for line in ps.splitlines():
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        pid, ppid, cpu, et, args = int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4]
        try:                                           # only processes started from THIS experiment's code directory
            if os.path.realpath(os.readlink(f"/proc/{pid}/cwd")) != os.path.realpath(C.HERE):
                continue
        except OSError:
            continue
        if "e5_opt.py" in args and f"--log {a.log}" in args:
            procs.append({"pid": pid, "ppid": ppid, "cpu_s": cpu, "elapsed_s": et, "args": args})
        elif "run_opt.sh" in args and "bash" in args:
            launch.append({"pid": pid, "args": args})
    pids = {p["pid"] for p in procs}
    main_p = [p for p in procs if p["ppid"] not in pids]
    workers = [p for p in procs if p["ppid"] in pids]
    rows = C.jsonl_read(os.path.join(C.OUT, f"opt_rows_{a.log}.jsonl")) if os.path.exists(os.path.join(C.OUT, f"opt_rows_{a.log}.jsonl")) else []
    rec = {"log": a.log, "stopped": time.strftime("%Y-%m-%d %H:%M:%S"), "reason": a.reason,
           "n_processes": len(procs), "n_workers": len(workers),
           "wall_s": max([p["elapsed_s"] for p in main_p], default=None),
           "cpu_s_workers": int(sum(p["cpu_s"] for p in workers)), "cpu_s_main": int(sum(p["cpu_s"] for p in main_p)),
           "cpu_s_total": int(sum(p["cpu_s"] for p in procs)),
           "n_rows_finished": len(rows), "cpu_s_rows_finished": float(sum(r["cpu"] for r in rows)),
           "launcher_stopped": [p["pid"] for p in launch], "processes": procs}
    for p in launch:                                   # first the launcher: it must not start the next log
        os.kill(p["pid"], signal.SIGTERM)
    for p in main_p + workers:
        try:
            os.kill(p["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    C.jdump(rec, os.path.join(C.OUT, f"opt_stop_{a.log}.json"))
    print({k: v for k, v in rec.items() if k != "processes"})


if __name__ == "__main__":
    main()
