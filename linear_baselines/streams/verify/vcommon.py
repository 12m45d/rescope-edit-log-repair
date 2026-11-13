"""P12 verifier -- shared helpers (CPU only; nothing here imports the experiment's code except gpuq, read-only).

Independent re-implementations (numpy only) of
  * the harness REPAIR / HELD-OUT split (sampler.split_pools: sorted uids, default_rng(20261109) permutation,
    first half = REPAIR),
  * the harness complaint sampler (sampler.sample_complaints, stratified image / text) and
  * P12's nested stream (p12_common.build_stream),
all from probe_records_v2.jsonl alone.
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = f"{_RP.ROOT}"
EXP = os.path.join(ROOT, "02_research", "experiments")
P12 = os.path.join(EXP, "08_rescope", "p12_linear_streams")
VER = os.path.join(P12, "verify")
VOUT = os.path.join(VER, "out")
V3 = os.path.join(EXP, "07_repair_v3")
TB_ROOT = os.path.join(EXP, "06_testbed_v3")
LOGS = ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")
SPLIT_SEED = 20261109
N_BASE = 100


def jload(p):
    with open(p) as f:
        return json.load(f)


def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(o, f, indent=1)
    os.replace(tmp, p)


def records(log):
    return [json.loads(l) for l in open(os.path.join(TB_ROOT, log, "probe_records_v2.jsonl")) if l.strip()]


def my_split(rows):
    """{probe_idx: 'repair' | 'heldout'} -- own implementation of the harness split."""
    uids = sorted({r["uid"] for r in rows})
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(uids))
    rep = {uids[int(i)] for i in perm[:len(uids) // 2]}
    pool = {}
    for r in rows:
        p = "repair" if r["uid"] in rep else "heldout"
        assert pool.setdefault(r["probe_idx"], p) == p
    return pool


def my_stream(rows, pool, seed):
    """Own implementation of harness.sample_complaints(H, 100, seed) + p12_common.build_stream."""
    rng = np.random.default_rng(seed)
    drawn = {}
    for pt, k in (("image", N_BASE // 2), ("text", N_BASE - N_BASE // 2)):          # the sampler's order: image, text
        cand = sorted([r for r in rows if r["is_complaint_D1"] and r["y0_valid"] and pool[r["probe_idx"]] == "repair"
                       and r["ptype"] == pt], key=lambda r: r["probe_idx"])
        idx = rng.choice(len(cand), k, replace=False)
        drawn[pt] = [cand[int(i)] for i in idx]
    stream = [drawn["text"][k // 2] if k % 2 == 0 else drawn["image"][k // 2] for k in range(N_BASE)]
    used = {c["group"] for c in stream}
    comp = sorted([r for r in rows if pool[r["probe_idx"]] == "repair" and r["y0_valid"] and r["is_complaint_D1"]],
                  key=lambda r: (r["probe_idx"], r["ptype"]))
    rest = [r for r in comp if r["group"] not in used]
    perm = np.random.default_rng([seed, 12]).permutation(len(rest))
    stream += [rest[int(i)] for i in perm]
    return [dict(r, complaint_kind="D1", sample_seed=seed) for r in stream]


def mcnemar_exact(after, before):
    """Exact two-sided McNemar p on paired booleans (own implementation)."""
    b = sum(1 for x, y in zip(after, before) if x and not y)
    c = sum(1 for x, y in zip(after, before) if y and not x)
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n), b, c


def cluster_boot_ci(values, uids, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of a mean, resampling uids (own implementation, same convention as harness.stats)."""
    g = {}
    for v, u in zip(values, uids):
        g.setdefault(u, []).append(float(v))
    keys = sorted(g, key=str)
    s = np.array([sum(g[k]) for k in keys])
    c = np.array([len(g[k]) for k in keys])
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    bm = s[idx].sum(1) / c[idx].sum(1)
    return float(np.quantile(bm, alpha / 2)), float(np.quantile(bm, 1 - alpha / 2))


# --------------------------------------------------------------------------- shared-GPU manners
# This verifier has the LOWEST priority on the two shared cards.  Everything here only READS /proc and nvidia-smi.
#   start rule (can_start): on the card, free >= my need + 1 GiB + the thresholds of every other agent's WAITING
#       launcher that fits on the card now (it may fire at any second) + half of what the running neighbours may still
#       take inside the caps they declared; and no other agent's job is being loaded right now (a python process of a
#       launcher younger than 100 s).  The caller looks twice (>= 20 s apart, no memory growth in between).
#   while running (Watchdog): leave the card (exit 75; the queue script starts me again later, finished conditions
#       are kept) when another agent's waiting launcher would fit without me, or when the card is nearly full
#       (a neighbour grew): first give back the unused part of my pool, then leave.
LAUNCHERS = {"gpu_wait.sh": "gn", "wait_gpu.sh": "gn", "gpu_run.sh": "gn", "gpu_any.sh": "gn",
             "gpu_fill.sh": "fill", "gpu_big.sh": "big", "gpu_launch.sh": "launch"}
CTX_MIB = 560
UNDECLARED_MIB = 2600
CAP_FLAGS = {"--mem_frac": "frac", "--frac": "frac", "--need_gb": "need_gb", "--cap_gb": "gib", "--lowmem": "gib", "--mem_gb": "gib"}


def _smi(args):
    return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=30).stdout


def cards():
    out = {}
    for line in _smi(["--query-gpu=index,uuid,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]).splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) == 5 and a[0].isdigit():
            out[int(a[0])] = {"uuid": a[1], "total": int(a[2]), "free": int(a[3]), "used": int(a[4])}
    return out


def free_mib(g):
    return cards().get(g, {}).get("free", 0)


def _procs():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
            with open(f"/proc/{pid}/stat") as f:
                st = f.read().rsplit(")", 1)[1].split()
            ppid, start = int(st[1]), int(st[19])
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except (OSError, ValueError, IndexError):
            continue
        out[int(pid)] = {"ppid": ppid, "argv": argv, "cwd": cwd, "start": start}
    return out


def _uptime():
    with open("/proc/uptime") as f:
        return float(f.read().split()[0])


def _is_python(argv):
    return bool(argv) and "python" in os.path.basename(argv[0])


def _descendants(pid, procs):
    kids = {}
    for p, d in procs.items():
        kids.setdefault(d["ppid"], []).append(p)
    out, stack = [], list(kids.get(pid, []))
    while stack:
        p = stack.pop()
        out.append(p)
        stack += kids.get(p, [])
    return out


def _env(pid, var):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            for e in f.read().split(b"\0"):
                if e.startswith(var.encode() + b"="):
                    return e.split(b"=", 1)[1].decode()
    except OSError:
        pass
    return None


LAUNCHER_RE = __import__("re").compile(r"(gpu|wait|launch|queue)", __import__("re").I)
THR_UNKNOWN = 17524          # a launcher whose arguments I cannot read is assumed to need a LLaVA-sized window


def launchers(procs=None):
    """Other agents' launcher processes: waiting (no python descendant yet) or with a job.

    A launcher = a process that IS a shell script (argv[0..2]) whose name contains gpu / wait / launch / queue, whatever
    its argument layout (2026-11-10 15:54: `p11_retrofit/verify/run_gpu.sh` and the comma-list form of
    `code/rescope/review/gpu_wait.sh` were not recognised by the first version of this function; my job then started
    in the same second as one of them and delayed two others).  Layouts read:
        <gpu|any> <need> cmd...            gpu_wait.sh / wait_gpu.sh / gpu_run.sh / gpu_any.sh
        <g,g> <need> cmd...                code/rescope/{tests/gpu_launch.sh, review/gpu_wait.sh}   (need + 1024)
        <need> cmd...                      09_testbed_scale gpu_fill.sh ($FILL_GPUS) / gpu_big.sh ($BIG_GPUS) (need + 1024),
                                           07_repair_v3/rq4 gpu_wait.sh
        <log> <need> <frac> "<g g>" ...    p11_retrofit/verify/run_gpu.sh
    anything else: cards 0 and 1, threshold THR_UNKNOWN, flagged unknown."""
    procs = _procs() if procs is None else procs
    hz = os.sysconf("SC_CLK_TCK")
    up = _uptime()
    out = []
    for pid, d in procs.items():
        argv = d["argv"]
        k = next((i for i, a in enumerate(argv[:3]) if a.endswith(".sh") and LAUNCHER_RE.search(os.path.basename(a))), None)
        if k is None or d["cwd"].startswith(VER) or (argv and _is_python(argv)):
            continue
        name, rest = os.path.basename(argv[k]), argv[k + 1:]
        cs, need, unknown = [0, 1], None, False
        try:
            if name == "run_gpu.sh":
                need = int(rest[1])
                cs = [int(x) for x in rest[3].replace(",", " ").split() if x.isdigit()] or [0, 1]
            elif name in ("gpu_fill.sh", "gpu_big.sh"):
                need = int(rest[0])
                var, dflt = ("FILL_GPUS", "0") if name == "gpu_fill.sh" else ("BIG_GPUS", "0 1")
                cs = [int(x) for x in (_env(pid, var) or dflt).split() if x.isdigit()] or [0, 1]
            elif rest and rest[0] in ("0", "1", "any"):
                cs = [0, 1] if rest[0] == "any" else [int(rest[0])]
                need = int(rest[1])
            elif rest and "," in rest[0] and all(x.isdigit() for x in rest[0].split(",")):
                cs = [int(x) for x in rest[0].split(",")]
                need = int(rest[1])
            else:
                need = int(rest[0])
        except (IndexError, ValueError):
            need = None
        if need is None or not (500 <= need <= 40000):
            need, unknown = THR_UNKNOWN, True
        plus = 1024 if (not unknown and ("09_testbed_scale" in d["cwd"] or "/code/rescope/" in d["cwd"])) else 0
        kids = [p for p in _descendants(pid, procs) if _is_python(procs[p]["argv"])]
        out.append({"pid": pid, "kind": name, "need": need, "thr": need + plus, "cards": cs, "unknown_layout": unknown,
                    "cwd": d["cwd"].split("/project_3/")[-1], "waiting": not kids,
                    "job_age_s": min((up - procs[p]["start"] / hz for p in kids), default=None),
                    "job_pids": kids})
    return out


def declared_cap_mib(argv, total):
    toks = []
    for a in argv:
        toks += a.split("=", 1) if a.startswith("--") and "=" in a else [a]
    cap = None
    for i, t in enumerate(toks[:-1]):
        kind = CAP_FLAGS.get(t)
        if kind is None:
            continue
        try:
            x = float(toks[i + 1])
        except ValueError:
            continue
        if kind == "frac":
            v = x * total if 0.0 < x <= 1.0 else None
        elif kind == "need_gb":
            v = (x + 1.0) * 1024.0 if x > 0 else None
        else:
            v = x * 1024.0 if x > 0 else None
        if v is not None:
            cap = v if cap is None else min(cap, v)
    return cap


def neighbours(g, cs=None, procs=None):
    """Other processes on card g: used MiB, declared cap, headroom (what they may still take)."""
    cs = cards() if cs is None else cs
    procs = _procs() if procs is None else procs
    out = []
    for line in _smi(["--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"]).splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) != 3 or a[2] != cs[g]["uuid"] or not a[0].isdigit() or int(a[0]) == os.getpid():
            continue
        pid, used = int(a[0]), int(a[1])
        d = procs.get(pid, {"argv": [], "cwd": ""})
        cap = declared_cap_mib(d["argv"], cs[g]["total"])
        hr = UNDECLARED_MIB if cap is None else max(0, int(cap + CTX_MIB - used))
        out.append({"pid": pid, "used": used, "cap": None if cap is None else int(cap), "headroom": hr,
                    "cwd": d["cwd"].split("/project_3/")[-1]})
    return out


def my_mib():
    for line in _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        a = [x.strip() for x in line.split(",")]
        if len(a) == 2 and a[0] == str(os.getpid()):
            return int(a[1])
    return 0


def can_start(g, need_mib, margin=1024):
    cs = cards()
    free = cs[g]["free"]
    try:
        procs = _procs()
        ls = [w for w in launchers(procs) if g in w["cards"]]
        nb = neighbours(g, cs, procs)
    except Exception as e:      # noqa: BLE001
        return False, {"free": free, "error": repr(e)}
    fit = [w for w in ls if w["waiting"] and w["thr"] <= free]
    waiting = [w for w in ls if w["waiting"]]
    loading = [w for w in ls if (not w["waiting"]) and w["job_age_s"] is not None and w["job_age_s"] < 100]
    hr = int(sum(n["headroom"] for n in nb))
    want = need_mib + margin + sum(w["thr"] for w in fit) + hr // 2
    info = {"free": free, "used": cs[g]["used"], "need": need_mib, "want": want, "neighbour_headroom": hr,
            "waiters_fitting_now": [(w["kind"], w["thr"], w["cwd"]) for w in fit],
            "waiters_not_fitting": [(w["kind"], w["thr"], w["cwd"]) for w in ls if w["waiting"] and w["thr"] > free],
            "jobs_loading": [(w["kind"], round(w["job_age_s"]), w["cwd"]) for w in loading]}
    # strict rule (since 16:02): this lowest-priority job does not start on a card while ANY other agent's launcher is
    # waiting for that card, whatever its distance from fitting
    return (free >= want and not loading and not waiting), info


def blocked_by_me(g, mine=None):
    """Other agents' waiting launchers that do not fit on card g now (nor on another card they may use) but would
    fit on g if this process left."""
    try:
        cs = cards()
        mine = my_mib() if mine is None else mine
        fg = cs[g]["free"]
        return [(w["kind"], w["thr"], w["cwd"]) for w in launchers() if w["waiting"] and g in w["cards"]
                and fg < w["thr"] <= fg + mine - 200 and all(cs[c]["free"] < w["thr"] for c in w["cards"] if c != g)]
    except Exception:           # noqa: BLE001
        return []


class Watchdog:
    """Background thread of the GPU worker (see the rules above)."""

    def __init__(self, g, say, period=12, tight=1300, exit_code=75):
        import threading
        self.g, self.say, self.period, self.tight, self.exit_code = g, say, period, tight, exit_code
        self.events = []
        threading.Thread(target=self._loop, daemon=True, name="p12v-watchdog").start()

    def _loop(self):
        import time
        import torch
        strikes = 0
        while True:
            time.sleep(self.period)
            try:
                free = free_mib(self.g)
                if free < self.tight:                   # a neighbour grew: give back what I do not use, look again
                    torch.cuda.empty_cache()
                    time.sleep(2)
                    free2 = free_mib(self.g)
                    self.say(f"  [watchdog] card nearly full ({free} MiB free): released my unused pool -> {free2} MiB free")
                    if free2 < self.tight:
                        self.say("  [watchdog] still nearly full: leaving the card (exit 75); finished conditions are kept")
                        os._exit(self.exit_code)
                b = blocked_by_me(self.g)
                strikes = strikes + 1 if b else 0
                if strikes >= 2:
                    self.say(f"  [watchdog] another agent's waiting launcher is blocked only by this process: {b}; leaving the card (exit 75)")
                    os._exit(self.exit_code)
            except Exception as e:      # noqa: BLE001
                self.say(f"  [watchdog] error {e!r}")
