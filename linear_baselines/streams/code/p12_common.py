"""P12 (linear repair streams) -- shared helpers.

Everything method-side goes through the shared repair harness of 07_repair_v3 (imported read-only):
load / sample_complaints / repair / evaluate.  Nothing under 01_* ... 07_* is written: the harness cache
directory is re-pointed to p12_linear_streams/cache (existing harness cache files are symlinked read-only,
the baseline file -- which the harness may re-save -- is copied).

Stream of reported complaints (one per log and seed), nested by construction:
  base  = harness.sample_complaints(H, 100, seed)      (D1 v2, REPAIR pool, 50 image + 50 text, draw order)
  stream[k] = text[k // 2] if k is even else image[k // 2]          k = 0..99
  then (n = ALL) the remaining REPAIR-pool D1 complaints in a seeded random order (default_rng([seed, 12])).
The reported set of size n is the first n complaints of the stream: ceil(n/2) text + floor(n/2) image, the
stratification of the harness sampler (n // 2 image, n - n // 2 text).  Sequential methods process the
complaints in stream order (a seeded random order that is the same for every n).
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import gpuq  # noqa: E402
except ImportError:
    # ARTIFACT: code/gpuq.py is not released.  It watched the launcher scripts of the other jobs on the authors' machine and made
    # a stream worker give way to them (reservations, head room of running neighbours).  Without it these courtesy checks are
    # skipped: the worker assumes that the card is its own.  Start GPU runs through the released queue (tools/gpuq/gq run --mem ...),
    # which does the admission.  Nothing that is computed depends on this module.
    class _NoNeighbours:
        BLOCK = 10 ** 9          # reserve() value that keeps a launcher from firing
        CTX_MIB = 560            # CUDA context on top of a process's torch cap (MiB)

        def reserve(self, g, *a, **k):
            return 0

        def free_mib(self):
            import collections
            return collections.defaultdict(lambda: 10 ** 9)

        def proc_mib(self, pid):
            return 0

        def blocked_by(self, *a, **k):
            return []

        def headroom(self, *a, **k):
            return 0

        def cards(self):
            raise RuntimeError("the queue helper code/gpuq.py is not released")

    gpuq = _NoNeighbours()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("[p12] code/gpuq.py is not released: the checks that made a worker give way to other jobs of the authors' machine are "
              "skipped (use tools/gpuq/gq run for admission)", file=sys.stderr, flush=True)

ROOT = f"{_RP.ROOT}"
EXP = os.path.join(ROOT, "02_research", "experiments")
P12 = os.path.join(EXP, "08_rescope", "p12_linear_streams")
V3 = os.path.join(EXP, "07_repair_v3")
OUT = os.path.join(P12, "out")
LOGD = os.path.join(P12, "logs")
FIG = os.path.join(P12, "figures")
MYCACHE = os.path.join(P12, "cache")
if V3 not in sys.path:
    sys.path.insert(0, V3)

from harness import baseline, evaluate, load, repair, sample_complaints  # noqa: E402,F401
from harness import evaluation as HE, hlog as HL, methods as HM, sampler as SM  # noqa: E402,F401
from harness.common import CACHE as HARNESS_CACHE, TB, VZ, Timer, clean, jdump, log, sha, sync  # noqa: E402,F401
from harness.hlog import Correction  # noqa: E402,F401
from harness.methods import _orth, protection_basis  # noqa: E402,F401

LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
N_GRID = (1, 2, 5, 10, 20, 50, 100, "ALL")
N_SMALL = (1, 2, 5, 10, 20)
N_ALLPOS = (1, 2, 5, 10)
SEEDS = (0, 1, 2)
N_BASE = 100
METHODS = ("COMP_B", "COMP_BP", "COMP_S", "COMP_SP", "COMP_B_ALLPOS", "REEDIT_B", "REEDIT_S", "RB_TOT",
           "RB_UNION")
# exploratory variants OUTSIDE the protocol (seed 0 only; reported separately, never in the protocol's
# operating-point selection): REEDIT_SC for n >= 20; COMP_BR at n = ALL
EXTRA_METHODS = ("REEDIT_SC", "COMP_BR")
PROT = ("K_ans",)
PROT_PLUS = ("K_ans", "probe:100000")               # probe:<n>, n >= pool size: every REPAIR-pool non-complaint
RB_TOT_BUDGET = 30
RB_UNION_BUDGET = 10
REEDIT_BATCH = 10
REEDIT_CHUNK = 1                                    # install capture: B=1 forwards (the z* compute path)
D2_NS = (10, 100, "ALL")
D2_SEED = 0
CAP_GB = {"qwen25vl": 10.0, "llava": 15.5}          # GiB of torch memory (+ ~0.5 GiB CUDA context); measured peaks 9.64 / 15.17 GiB


# --------------------------------------------------------------------------- harness cache redirect
def redirect_cache():
    """All harness cache reads/writes of this process go to p12_linear_streams/cache."""
    os.makedirs(MYCACHE, exist_ok=True)
    for fn in sorted(os.listdir(HARNESS_CACHE)):
        src, dst = os.path.join(HARNESS_CACHE, fn), os.path.join(MYCACHE, fn)
        if os.path.lexists(dst):
            continue
        try:
            if fn.startswith("baseline_"):
                tmp = dst + f".tmp{os.getpid()}"
                shutil.copy2(src, tmp)              # private copy: baseline() may re-save its file
                os.replace(tmp, dst)
            elif fn.startswith(("editkeys_", "probekeys_")):
                os.symlink(src, dst)                # only ever torch.load()-ed when present
        except FileExistsError:
            pass
    HM.CACHE = HE.CACHE = HL.CACHE = MYCACHE


def setup_process(backbone, cap_gb=None, threads=None):
    # CPU threads of this process (torch intra-op = OpenMP / MKL BLAS): $P12_THREADS, default 4 (resume of
    # 2026-11-10 17:10: at most 4 BLAS threads per process; the first run used 8)
    torch.set_num_threads(int(threads or os.environ.get("P12_THREADS", "4")))
    HL.GPU_DELTA_BUDGET = 0                         # exact batch deltas stay on the CPU (shared GPUs)
    cap = float(cap_gb or CAP_GB[backbone])
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2 ** 30 / tot), 0)
    redirect_cache()
    return cap


EXIT_RETRY = 75                                     # "could not start now, queue me again" (chain.sh)


def load_log(name, cap_gb=None):
    bb = TB.LOGS[name]["backbone"]
    cap = setup_process(bb, cap_gb)
    # last look before allocating: another job may have taken the card between gpu_wait.sh and here.
    # Leave without touching the memory rather than race a higher-priority job for it.
    try:
        g = _card()
        res = gpuq.reserve(g)                       # BLOCK if another agent's launcher is waiting, else headroom
        free = gpuq.free_mib().get(g, 0)
    except Exception:          # noqa: BLE001
        res, free = gpuq.BLOCK, 0
    mine = cap * 1024 + gpuq.CTX_MIB                # my largest footprint (MiB)
    if res >= gpuq.BLOCK or free < mine + 256 + res:
        print(f"[p12] not starting: free {free} MiB, my footprint {mine:.0f} MiB, "
              + ("another agent's launcher is waiting for the card" if res >= gpuq.BLOCK else f"neighbours' declared headroom {res} MiB"),
              flush=True)
        raise SystemExit(EXIT_RETRY)
    try:
        H = load(name, solve_device="cpu")          # COMP and REEDIT solves on the CPU (fp64), both logs
    except RuntimeError as e:                       # torch.OutOfMemoryError / AcceleratorError: the card filled up
        if "out of memory" not in str(e).lower():
            raise
        print(f"[p12] CUDA out of memory while loading the model: {str(e)[:200]}", flush=True)
        os._exit(EXIT_RETRY)                        # leave at once, give everything back
    return H, cap


def oom_kind(msg):
    """'own_cap' if this process's memory fraction was the binding limit (the card itself still had room),
    else 'card_full'.  torch: '... has a total capacity of X of which Y is free. ... Z allowed; ...'."""
    import re
    m = re.search(r"of which ([0-9.]+) (MiB|GiB) is free", msg)
    free_gib = None if m is None else float(m.group(1)) / (1024.0 if m.group(2) == "MiB" else 1.0)
    return "own_cap" if ("allowed;" in msg and free_gib is not None and free_gib >= 1.5) else "card_full"


def is_oom(e):
    """CUDA ran out of memory, whatever the exception class (OutOfMemoryError, AcceleratorError, cuBLAS / cuDNN
    allocation failures)."""
    t = str(e)
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in t.lower() or "ALLOC_FAILED" in t


def leave_card(fh, why):
    """The card is full: this lowest-priority process gives its memory back at once and is queued again."""
    try:
        log(f"  leaving the card ({why}); queued again by chain.sh", fh)
    finally:
        os._exit(EXIT_RETRY)


def gpu_procs():
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.run(["nvidia-smi", "-i", idx, "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        return len([l for l in out.splitlines() if l.strip()])
    except Exception:          # noqa: BLE001
        return None


TIGHT_MIB = 700                                     # card almost full while I hold memory: a neighbour grew


def _card():
    return int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])


def blocked_waiters():
    """Other agents' queued launchers that do not fit on THIS card now but would if this process left
    (gpuq.blocked_by; every launcher kind the other agents use).  They have priority."""
    try:
        return gpuq.blocked_by(_card(), gpuq.proc_mib(os.getpid()))
    except Exception:          # noqa: BLE001
        return []


def card_free_mib():
    try:
        return gpuq.free_mib().get(_card())
    except Exception:          # noqa: BLE001
        return None


def in_the_way():
    """(reason or None) -- is this process in another agent's way on its card right now?
      * a queued launcher of another agent does not fit, but would if this process left, or
      * the card cannot hold what the other agents' running processes may still take within their declared caps
        (free < their headroom + 256 MiB), or it is almost full (< TIGHT_MIB free)."""
    try:
        g = _card()
        mine = gpuq.proc_mib(os.getpid())
        cs = gpuq.cards()
        free = cs[g]["free"]
        b = gpuq.blocked_by(g, mine, {k: v["free"] for k, v in cs.items()})
        hr = gpuq.headroom(g, cs)
    except Exception:          # noqa: BLE001
        return None
    if b:
        return f"another agent's queued job is blocked by this process: {b}"
    if mine > 0 and free < max(TIGHT_MIB, hr + 256):
        return f"{free} MiB free on the card, the running neighbours may still take {hr} MiB within their declared caps"
    return None


def yield_if_blocking(fh, what=""):
    """Config-boundary check: leave the card (exit EXIT_RETRY; chain.sh queues this worker again)."""
    why = in_the_way()
    if why:
        log(f"  yielding the card ({why}) {what}", fh)
        raise SystemExit(EXIT_RETRY)


def start_watchdog(fh, period=20):
    """Background thread of every GPU worker: every `period` s ask in_the_way(); on two consecutive positive
    looks leave the card at once (os._exit(EXIT_RETRY); outputs are written atomically and every unit is
    resumable).  This process has the lowest priority on the card."""
    import threading

    def loop():
        strikes = 0
        while True:
            time.sleep(period)
            why = in_the_way()
            strikes = strikes + 1 if why else 0
            if strikes >= 2:
                try:
                    log(f"  [watchdog] leaving the card now ({why}); the current config is redone later", fh)
                finally:
                    os._exit(EXIT_RETRY)
    threading.Thread(target=loop, daemon=True, name="p12-watchdog").start()


# --------------------------------------------------------------------------- pools / streams
def repair_rows(H):
    """(valid rows, D1 complaints, non-complaints) of the REPAIR pool, sorted."""
    pl = SM.pool_of(H)
    rows = sorted([r for r in H.rows if pl[r["probe_idx"]] == "repair" and r["y0_valid"]],
                  key=lambda r: (r["probe_idx"], r["ptype"]))
    comp = [r for r in rows if r["is_complaint_D1"]]
    nonc = [r for r in rows if r["reproduced"] is True]
    return rows, comp, nonc


def build_stream(H, seed):
    base = sample_complaints(H, N_BASE, seed=seed)
    img = [c for c in base if c["ptype"] == "image"]
    txt = [c for c in base if c["ptype"] == "text"]
    assert len(img) == N_BASE // 2 and len(txt) == N_BASE - N_BASE // 2
    stream = [txt[k // 2] if k % 2 == 0 else img[k // 2] for k in range(N_BASE)]
    used = {c["group"] for c in stream}
    assert len(used) == N_BASE
    _, comp, _ = repair_rows(H)
    rest = [dict(r, complaint_kind="D1", sample_seed=seed) for r in comp if r["group"] not in used]
    perm = np.random.default_rng([seed, 12]).permutation(len(rest))
    stream += [rest[int(i)] for i in perm]
    assert len(stream) == len(comp) and len({c["group"] for c in stream}) == len(comp)
    pl = SM.pool_of(H)
    assert all(pl[c["probe_idx"]] == "repair" and c["is_complaint_D1"] and c["y0_valid"] for c in stream)
    return stream


def n_value(stream, n):
    return len(stream) if n == "ALL" else int(n)


def reported(stream, n):
    return stream[:n_value(stream, n)]


def config_name(method, n, seed):
    return f"{method}__n{n}__s{seed}"


def phase_plan(phase):
    """Work units of chain.sh (each one resumable):
      tab       seed 0 of the protocol at n in {1, 10, 100, ALL} (the summary table)
      br        the one exploratory COMP_BR config (outside the protocol; after the protocol's D2 core)
      s0        seed 0 of the protocol, all n
      rest      seeds 1 and 2 of the protocol (n <= 20)
      extra     exploratory REEDIT_SC (seed 0)
      protocol  every protocol config (status check)"""
    if phase == "tab":                               # protocol, seed 0, the n of the summary table
        return [x for x in plan((0,), extra=False) if x[1] in (1, 10, 100, "ALL")]
    if phase == "br":                                # the one exploratory "largest subspace" config
        return [("COMP_BR", "ALL", 0)]
    if phase == "s0":                                # protocol, seed 0, every n
        return plan((0,), extra=False)
    if phase == "rest":
        return plan((1, 2), extra=False)
    if phase == "extra":
        return [("REEDIT_SC", n, 0) for n in (100, "ALL", 20, 50)]
    if phase == "protocol":
        return plan(extra=False)
    raise ValueError(phase)


def plan(seeds=SEEDS, extra=True):
    """Every (method, n, seed) of the protocol, in priority order."""
    out = []
    order0 = (1, 10, 100, "ALL", 2, 5, 20, 50)          # the n of the summary table first
    for seed in seeds:
        ns = order0 if seed == 0 else N_SMALL
        for n in ns:
            for m in METHODS:
                if m == "COMP_B_ALLPOS" and n not in N_ALLPOS:
                    continue
                out.append((m, n, seed))
    if extra and 0 in seeds:
        out.append(("COMP_BR", "ALL", 0))
        for n in (20, 50, 100, "ALL"):
            out.append(("REEDIT_SC", n, 0))
    return out


def alias_of(method, n):
    """Configs that are the same computation as another config (not run twice)."""
    if method == "REEDIT_S" and n != "ALL" and int(n) <= REEDIT_BATCH:
        return "REEDIT_B"                               # one batch = one joint solve of the same targets
    if method == "COMP_S" and n == 1:
        return "COMP_B"
    if method == "COMP_SP" and n == 1:
        return "COMP_BP"
    return None


# --------------------------------------------------------------------------- file helpers
def jload(p):
    with open(p) as f:
        return json.load(f)


def atomic_jdump(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    jdump(obj, tmp)
    os.replace(tmp, path)


@contextlib.contextmanager
def flock(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def claim(lockpath):
    """Claim a config for this process (several workers may share one out directory)."""
    try:
        fd = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = open(lockpath).read().strip()
        except OSError:
            return False
        if pid and pid_alive(pid) and int(pid) != os.getpid():
            return False
        try:                                            # stale lock of a dead worker
            os.remove(lockpath)
        except OSError:
            return False
        return claim(lockpath)


def release(lockpath):
    with contextlib.suppress(OSError):
        os.remove(lockpath)


def meta_clean(meta):
    return {k: v for k, v in meta.items() if not k.startswith("_")}


def cost_small(c):
    """JSON-size-bounded copy of a harness cost record."""
    keep = ("method", "cfg", "passes", "tokens", "flops_total", "flops_solve", "sec", "n_backward", "n_forward",
            "n_complaints", "correction_fro", "solve_device", "setup_sec_excluded", "passes_zopt", "flops_zopt",
            "n_backward_zopt", "zopt_attempts_max", "device", "torch_threads")
    return {k: c[k] for k in keep if k in c}


def add_costs(costs):
    out = {"passes": {}, "flops_total": 0.0, "flops_solve": 0.0, "sec_total": 0.0, "n_backward": 0, "n_forward": 0}
    for c in costs:
        for k, v in c["passes"].items():
            out["passes"][k] = out["passes"].get(k, 0) + v
        out["flops_total"] += c["flops_total"]
        out["flops_solve"] += c["flops_solve"]
        out["sec_total"] += c["sec"]["total"]
        out["n_backward"] += c["n_backward"]
        out["n_forward"] += c["n_forward"]
    out["n_calls"] = len(costs)
    return out


def save_corr(corr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if corr.kind == "adapter":
        blob = {"kind": "adapter", "Lf": corr.Lf.cpu(), "Rf": corr.Rf.cpu(), "gate": corr.gate}
    elif corr.kind == "weight":
        blob = {"kind": "weight", "W": corr.W.cpu()}
    else:
        blob = {"kind": "none"}
    tmp = path + f".tmp{os.getpid()}"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_corr(path, dev):
    b = torch.load(path, map_location="cpu")
    if b["kind"] == "adapter":
        return Correction("adapter", Lf=b["Lf"].to(dev), Rf=b["Rf"].to(dev), gate=b["gate"], meta={})
    if b["kind"] == "weight":
        return Correction("weight", W=b["W"].to(dev), meta={})
    return Correction("none", meta={})
