#!/usr/bin/env python3
"""A job that only pretends to use a GPU (for the gq tests with the fake nvidia-smi; no GPU, no torch).

It registers --mib MiB for its own pid on card $CUDA_VISIBLE_DEVICES in $GQ_FAKE_WORLD.d/, sleeps, and exits.
  --secs S         run time                       --code N        exit code
  --oom-once FILE  first call (FILE missing): create FILE, print a CUDA out-of-memory error, exit 1
  --oom            always fail with a CUDA out-of-memory error
  --ignore-term    ignore SIGTERM                 --say TEXT      print TEXT on stdout and on stderr
"""
import argparse
import json
import os
import signal
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--mib", type=int, default=0)
p.add_argument("--secs", type=float, default=1.0)
p.add_argument("--code", type=int, default=0)
p.add_argument("--oom-once")
p.add_argument("--oom", action="store_true")
p.add_argument("--ignore-term", action="store_true")
p.add_argument("--say")
p.add_argument("--cap_gb")                           # accepted and ignored: the command line of tests/gpu_job.py
p.add_argument("--tag")
a = p.parse_args()

if a.ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
card = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if a.say:
    print(f"{a.say} stdout card={card} mem={os.environ.get('GQ_MEM')}", flush=True)
    print(f"{a.say} stderr", file=sys.stderr, flush=True)
if a.oom or (a.oom_once and not os.path.exists(a.oom_once)):
    if a.oom_once:
        open(a.oom_once, "w").close()
    time.sleep(min(a.secs, 0.3))
    print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of "
          "31.84 GiB of which 1.10 GiB is free.", file=sys.stderr, flush=True)
    sys.exit(1)
reg = None
if a.mib and os.environ.get("GQ_FAKE_WORLD"):
    d = os.environ["GQ_FAKE_WORLD"] + ".d"
    os.makedirs(d, exist_ok=True)
    reg = os.path.join(d, str(os.getpid()))
    with open(reg + ".tmp", "w") as f:
        json.dump({"gpu": int(card), "mib": a.mib}, f)
    os.replace(reg + ".tmp", reg)
try:
    time.sleep(a.secs)
finally:
    if reg:
        os.remove(reg)
sys.exit(a.code)
