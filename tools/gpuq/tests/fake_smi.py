#!/usr/bin/env python3
"""Fake nvidia-smi for the gq tests (GQ_SMI="python fake_smi.py WORLD.json"): answers gq's two queries.

WORLD.json: {"gpus": [[index, "uuid", total_MiB], ...],
             "apps": [[index, pid, MiB], ...],          compute processes that are simply there (foreign ones)
             "hidden": {"index": MiB},                  memory that no listed process accounts for
             "ramp": {"index": [MiB_per_s, t0, t1]},    ... and that grows from time t0 to time t1
             "fail": false}                             true: fail like a broken nvidia-smi
WORLD.json.d/<pid>: '{"gpu": index, "mib": MiB}'        written by fake_job.py for itself; counts while <pid> lives
Like the real driver, 497 MiB of every card are neither used nor free.
"""
import json
import os
import sys
import time

RESERVED = 497
with open(sys.argv[1]) as f:
    world = json.load(f)
if world.get("fail"):
    sys.stderr.write("fake nvidia-smi: unable to determine the device handle\n")
    sys.exit(9)
apps = [tuple(a) for a in world.get("apps", [])]
if os.path.isdir(sys.argv[1] + ".d"):
    for name in sorted(os.listdir(sys.argv[1] + ".d")):
        if name.isdigit() and os.path.exists(f"/proc/{name}"):
            try:
                with open(os.path.join(sys.argv[1] + ".d", name)) as f:
                    p = json.load(f)
                apps.append((p["gpu"], int(name), p["mib"]))
            except (OSError, ValueError):
                pass
if sys.argv[2].startswith("--query-gpu="):
    for i, uuid, total in world["gpus"]:
        rate, t0, t1 = world.get("ramp", {}).get(str(i), (0, 0, 0))
        used = sum(m for g, _, m in apps if g == i) + world.get("hidden", {}).get(str(i), 0)
        used += int(rate * (min(max(time.time(), t0), t1) - t0))
        print(f"{i}, {uuid}, {total}, {used}, {total - used - RESERVED}")
elif sys.argv[2].startswith("--query-compute-apps="):
    uuid_of = {i: uuid for i, uuid, _ in world["gpus"]}
    for g, pid, mib in apps:
        print(f"{uuid_of[g]}, {pid}, {mib}")
else:
    sys.exit(2)
