#!/usr/bin/env python3
"""Dummy GPU job for tests/test_gpu.py: takes about --mib MiB on its card, as nvidia-smi counts them (CUDA context
+ one torch.empty tensor), holds them for --secs seconds and exits.  It caps itself the way README.txt tells real
jobs to, and it looks before it allocates: it does not start if that would leave less than --keep-free MiB free
on the card (exit code 75).

    gq run --mem 3000 -- python gpu_job.py --mib 2850 --secs 8 [--cap_gb 2.3] [--tag TEXT] [--oom-once FILE]

--cap_gb: the cap it sets on itself (GiB of torch memory; default: what its tensor needs).  It is given on the
command line because the older launchers of this project read other jobs' caps from there.
--oom-once FILE: the first call (FILE does not exist yet) creates FILE and then asks torch for 1 GiB more than
its own cap.  torch refuses that by itself (torch.OutOfMemoryError, exit code 1) without asking the card for the
memory: a real CUDA out-of-memory failure that cannot hurt a neighbour.
"""
import argparse
import os
import subprocess
import sys
import time

CONTEXT = 500        # MiB that a torch process takes on the card besides its tensors (measured here: 498)

p = argparse.ArgumentParser()
p.add_argument("--mib", type=int, required=True)
p.add_argument("--secs", type=float, default=5.0)
p.add_argument("--keep-free", type=int, default=1500)
p.add_argument("--cap_gb", type=float)
p.add_argument("--oom-once")
p.add_argument("--tag", default="")                  # only so that the test finds its processes in /proc
a = p.parse_args()
card = os.environ["CUDA_VISIBLE_DEVICES"]            # set by gq run


def free_mib():
    out = subprocess.run(["nvidia-smi", "-i", card, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout
    return int(out.strip())


free = free_mib()
if free - a.mib < a.keep_free:
    print(f"gpu_job: card {card} has {free} MiB free; {a.mib} MiB more would leave less than {a.keep_free}: not started",
          file=sys.stderr, flush=True)
    sys.exit(75)

import torch  # noqa: E402

tensor = a.mib - CONTEXT
total = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
cap = a.cap_gb * 1024 if a.cap_gb else tensor + 16
torch.cuda.set_per_process_memory_fraction(cap / total, 0)                # the cap: torch cannot take more than this
if a.oom_once and not os.path.exists(a.oom_once):
    open(a.oom_once, "w").close()
    tensor += 1024                                   # beyond the cap: torch raises OutOfMemoryError
x = torch.empty(tensor * 2 ** 20, dtype=torch.uint8, device="cuda")
print(f"gpu_job card={card} pid={os.getpid()} tensor={tensor} free_before={free} free_after={free_mib()}", flush=True)
time.sleep(a.secs)
