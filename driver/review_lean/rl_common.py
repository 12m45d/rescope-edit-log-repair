"""Common header of the lean review (review_lean): paths, environment, the driver import.  Nothing here computes a metric."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.dirname(HERE)
FIN = os.path.dirname(COMMON)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(FIN)))
LIB = os.path.join(ROOT, "02_research", "code", "rescope")
OUT = os.path.join(HERE, "out")
DEV = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
sys.path.insert(0, COMMON)


def cap_gpu():
    import torch
    if torch.cuda.is_available() and os.environ.get("GQ_MEM"):
        cap = (int(os.environ["GQ_MEM"]) - 600) * 2 ** 20
        torch.cuda.set_per_process_memory_fraction(cap / torch.cuda.get_device_properties(0).total_memory, 0)


def dump(name, obj):
    import json
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    return p
