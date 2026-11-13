#!/usr/bin/env python3
"""gq -- the one GPU job queue shared by all agents on this machine (usage and rules: README.txt).

    gq run --mem MiB [--gpu any|N] [--name TAG] [--prio INT] [--no-retry] -- command ...
    gq status
    gq wait-idle [--timeout S]

Every job is one small file state/job-<id>.json, written only by its own `gq run` process (the owner).  While
"gpu" is null the file is a ticket: the job waits.  When a card is written into it, it is a reservation of "need"
MiB on that card until the command has exited.  A file whose owner process is dead is ignored and removed.  The
decision "choose a card + write the reservation" is taken under an exclusive flock on state/lock, on a fresh
nvidia-smi reading.  There is no daemon: every waiting `gq run` decides for itself, and only about itself.
gq never signals or changes a process, with one exception: its own command gets the termination signal that the
`gq run` process receives (SIGTERM if that process is killed outright).  Standard library only; Linux only.
Other tools read the job files (fields id, name, pid, pstart, need, want, gpu): keep them when changing this file.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import argparse
import contextlib
import ctypes
import fcntl
import json
import os
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time


def _env(name, default):
    return type(default)(os.environ.get("GQ_" + name, default))


# The GQ_* variables exist for the tests; real jobs run with the defaults.
STATE = _env("STATE", f"{_RP.ROOT}/02_research/tools/gpuq/state")
SMI = _env("SMI", "nvidia-smi")           # the tests put a fake reading here
MARGIN = _env("MARGIN", 700)              # MiB that must stay free on top of a job's need
POLL = _env("POLL", 15.0)                 # s between two decisions of a waiting job
LOOK = _env("LOOK", 10.0)                 # s between the two looks of one decision
YOUNG = _env("YOUNG", 120.0)              # a foreign process younger than this (s) may still be loading
GROW = _env("GROW", 300)                  # MiB of growth between the two looks that makes a card busy
STARVE = _env("STARVE", 900.0)            # a job that has waited longer (s) keeps cards from later tickets
OOM_WINDOW = _env("OOM_WINDOW", 120.0)    # a CUDA out-of-memory failure within this run time (s) is retried once
OOM_WAIT = _env("OOM_WAIT", 60.0)         # ... after this many seconds
BEAT = _env("BEAT", 120.0)                # a waiting job whose process has not looked for this long is passed over
CAPS = {c: int(m) for c, m in (x.split(":") for x in _env("CAP", "").split(",") if x)}
#        ^ GQ_CAP="1:4000" (the GPU test): the jobs submitted with it reserve at most 4000 MiB of card 1 together,
#          so that the test stays small next to the real jobs in the same queue
KEYS = ("id", "name", "pid", "pstart", "need", "want", "prio", "t", "beat", "not_before", "gpu", "t_start", "reason", "cmd")

STOP = None       # termination signal received by this `gq run` process
CHILD = None      # the command it runs (subprocess.Popen)
LIBC = ctypes.CDLL(None)


class Cancelled(BaseException):
    """A termination signal arrived while the job was waiting."""


# ----------------------------------------------------------------------------- processes (read-only: /proc)
def pstat(pid):
    """(parent pid, start time in clock ticks since boot, state letter) of a process; None if there is none."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            a = f.read().rsplit(")", 1)[1].split()
        return int(a[1]), int(a[19]), a[0]
    except (OSError, IndexError, ValueError):
        return None


def alive(job):
    """Is the owner of the job file still running?  (pid + start time: a recycled pid is not the owner)"""
    s = pstat(job["pid"])
    return s is not None and s[1] == job["pstart"] and s[2] not in "ZX"


def age_of(pid):
    """Seconds since the process started; None if it is gone."""
    s = pstat(pid)
    if s is None:
        return None
    with open("/proc/uptime") as f:
        return float(f.read().split()[0]) - s[1] / os.sysconf("SC_CLK_TCK")


def owner_of(pid, owners, card):
    """The running queue job on `card` that process `pid` descends from ({owner pid: job}); None: a foreign process.
    (The nearest such ancestor: a `gq run` inside a `gq run` on the other card holds a reservation on both.)"""
    for _ in range(100):
        if pid in owners and owners[pid]["gpu"] == card:
            return owners[pid]
        s = pstat(pid)
        if s is None or s[0] <= 0:
            return None
        pid = s[0]
    return None


# ----------------------------------------------------------------------------- state directory
def path_of(jid, kind="job", ext="json"):
    return os.path.join(STATE, f"{kind}-{jid}.{ext}")


@contextlib.contextmanager
def locked(patience=60.0):
    """Exclusive lock on state/lock: one decision at a time.  The kernel drops it when the holder dies.
    TimeoutError if another process keeps it for `patience` s (a stopped gq?): the caller tries again later."""
    path, end = os.path.join(STATE, "lock"), time.time() + patience
    while True:
        os.makedirs(STATE, exist_ok=True)
        with open(path, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                mine = os.fstat(f.fileno()).st_ino == os.stat(path).st_ino       # False: the file was replaced
            except OSError:               # held by another process
                mine = False
            if mine:
                yield
                return
        if time.time() > end:
            raise TimeoutError(f"state/lock has been held by another process for {patience:.0f} s")
        time.sleep(0.05)


def save(job):
    os.makedirs(STATE, exist_ok=True)
    with open(path_of(job["id"], ext="tmp"), "w") as f:
        json.dump(job, f)
    os.replace(path_of(job["id"], ext="tmp"), path_of(job["id"]))     # atomic: readers see the old or the new file


def forget(jid):
    for p in (path_of(jid), path_of(jid, ext="tmp"), path_of(jid, "err", "log")):
        with contextlib.suppress(OSError):
            os.remove(p)


def load_jobs(clean=False):
    """The live jobs.  A file whose owner is dead is ignored, and removed if `clean` (callers hold the lock)."""
    jobs = []
    for name in sorted(os.listdir(STATE)):
        if name.startswith("job-") and name.endswith(".json"):
            try:
                with open(os.path.join(STATE, name)) as f:
                    job = json.load(f)
                live = alive(job)
            except (OSError, ValueError, KeyError, TypeError):
                continue                  # its owner has just removed it, or it is not a job file
            if live and all(k in job for k in KEYS):
                jobs.append(job)
            elif not live and clean:
                forget(name[4:-5])
                log("stale", job)
    return jobs


def log(event, job, **more):
    """One line per event in state/events.log: who was queued, started (and on which reading), ended."""
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), t=round(time.time(), 2), event=event,
               **{k: job.get(k) for k in ("id", "name", "pid", "need", "gpu")}, **more)
    with contextlib.suppress(Exception), open(os.path.join(STATE, "events.log"), "a") as f:
        f.write(json.dumps(rec) + "\n")               # the log must never be in the way of a job


def say(text):
    with contextlib.suppress(OSError, ValueError):       # a closed stderr must not stop a job that holds a card
        print(f"[gq {time.strftime('%H:%M:%S')}] {text}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------- nvidia-smi
def smi(query):
    """The rows of one nvidia-smi csv query; None when nvidia-smi fails."""
    try:
        r = subprocess.run(shlex.split(SMI) + [query, "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def look(jobs):
    """One reading: {card: {"total", "free", "used", "foreign", "own": {job id: MiB}, "young", ...}} in MiB, or
    None when nvidia-smi failed or printed something unexpected (the caller waits and looks again).
    used = total - free: what is really taken, with the ~0.5 GiB that the driver keeps for itself (nvidia-smi's
    own "used" column leaves that out).  own = memory of the processes that descend from a running queue job, on
    the card of that job.  foreign = memory of all other compute processes, or, if that is more, whatever
    nvidia-smi counts as used beyond the queue's own (memory that it shows for no process is foreign as well).
    young = a foreign process that started less than YOUNG s ago, if there is one."""
    gpus = smi("--query-gpu=index,uuid,memory.total,memory.used,memory.free")
    apps = smi("--query-compute-apps=gpu_uuid,pid,used_memory")
    if not gpus or apps is None:
        return None
    owners = {j["pid"]: j for j in jobs if j["gpu"] is not None}
    try:
        cards = {int(i): dict(uuid=u, total=int(t), smi_used=int(m), free=int(f), used=int(t) - int(f), foreign=0,
                              own={}, young="", pids=[]) for i, u, t, m, f in gpus}
        by_uuid = {card["uuid"]: (c, card) for c, card in cards.items()}
        for uuid, pid, mib in apps:
            (c, card), pid, mib = by_uuid[uuid], int(pid), int(mib) if mib.isdigit() else 0
            job = owner_of(pid, owners, c)
            if job is not None:
                card["own"][job["id"]] = card["own"].get(job["id"], 0) + mib
            else:
                card["foreign"] += mib
                card["pids"].append(f"{pid}:{mib}")
                age = age_of(pid)
                if age is not None and age < YOUNG:
                    card["young"] = f"foreign pid {pid} started {age:.0f} s ago"
        for card in cards.values():
            card["foreign"] = max(card["foreign"], card["smi_used"] - sum(card["own"].values()))
    except (ValueError, KeyError):
        return None
    return cards


def claimed(card, running):
    """MiB of a card that are spoken for: the foreign memory + the reservations of the queue's jobs on the card
    (a job that takes more than it reserved counts with what it takes)."""
    return card["foreign"] + sum(max(j["need"], card["own"].get(j["id"], 0)) for j in running)


def taken(card, running):
    """MiB of a card that count as taken: what is used now, or what is spoken for, whichever is more."""
    return max(card["used"], claimed(card, running))


def mark_busy(before, cards, jobs):
    """Set cards[c]["busy"]: why card c is left alone for this look ("" = it is not).  A foreign process may still
    be growing: one that started less than YOUNG s ago, or memory that grew by more than GROW MiB since the look
    LOOK s earlier.  What the queue's own jobs take inside their reservations is not growth: it has been counted
    in full since they were admitted.  What they take beyond their reservations is."""
    for c, card in cards.items():
        on = [j for j in jobs if j["gpu"] == c]
        if c not in before:
            card["busy"] = "no earlier look at this card"
            continue
        grew = claimed(card, on) - claimed(before[c], on)
        card["busy"] = card["young"] or (f"memory grew by {grew} MiB in {LOOK:.0f} s" if grew > GROW else "")


# ----------------------------------------------------------------------------- the admission rule
def ticket(job):
    return (job["t"], job["id"])


def schedule(jobs, cards, now):
    """The admission rule as a pure function.  jobs: the live jobs; cards: a look() with "busy" set.
    Returns {job id: (card or None, reason why not)} for the waiting jobs.

    Jobs are served by (priority descending, ticket ascending).  One pass in that order plans a card for every
    waiting job that can start now, so a later job never takes the room of an earlier one; a later job may start
    if it fits now (backfilling), except on a card that is kept for a job which has waited for more than STARVE s."""
    held = {c: [j for j in jobs if j["gpu"] == c] for c in cards}     # running jobs, plus the jobs planned in this pass

    def room(c, j, alone=False):
        """MiB that job j may take on card c.  alone: as if the queue's jobs with later tickets were not there."""
        card, keep = cards[c], [h for h in held[c] if not alone or ticket(h) < ticket(j)]
        gone = sum(card["own"].get(h["id"], 0) for h in held[c] if h not in keep)
        free = card["total"] - taken(dict(card, used=card["used"] - gone), keep)
        if str(c) in j.get("cap", {}):    # GPU test: the jobs that carry a budget share that much of the card
            free = min(free, j["cap"][str(c)] - sum(h["need"] for h in keep if h.get("cap")) + MARGIN)
        return free

    out, old = {}, []
    for j in sorted((j for j in jobs if j["gpu"] is None), key=lambda j: (-j["prio"], ticket(j))):
        if now - j["beat"] > BEAT:        # its process is stopped or stuck: it must not hold up the others
            out[j["id"]] = (None, f"its gq process has not looked for {now - j['beat']:.0f} s")
            continue
        if now < j["not_before"]:
            out[j["id"]] = (None, f"queued again after CUDA out of memory: next try in {j['not_before'] - now:.0f} s")
            continue
        fits, why = [], []
        for c in sorted(cards):
            if j["want"] not in (None, c):
                continue
            keeper = next((w["name"] for w in old if ticket(w) < ticket(j) and w["want"] in (None, c)
                           and room(c, w, True) >= w["need"] + MARGIN), None)      # (w's priority is j's or higher)
            ahead = [h["name"] for h in held[c] if h["gpu"] is None]
            if room(c, j) < j["need"] + MARGIN:
                why.append(f"card {c}: room {max(room(c, j), 0)} MiB < {j['need']}+{MARGIN}"
                           + (f" (after {', '.join(ahead)}, next in line)" if ahead else ""))
            elif cards[c]["busy"]:
                why.append(f"card {c} busy: {cards[c]['busy']}")
            elif keeper:
                why.append(f"card {c} is kept for the long-waiting job {keeper}")
            else:
                fits.append((room(c, j), c))
        if fits:
            c = min(fits)[1]              # best fit: the card with LESS free memory (keeps room for big jobs)
            held[c].append(j)
        out[j["id"]] = (c, "") if fits else (None, "; ".join(why) or f"there is no card {j['want']}")
        if now - j["t"] > STARVE:
            old.append(j)                 # while it waits, later tickets stay off the cards that it could use
    return out


# ----------------------------------------------------------------------------- gq run
def on_signal(sig, frame):
    """Pass a termination signal on to our own command (its process group); while the job waits, stop waiting."""
    global STOP
    STOP = sig
    if CHILD is not None and CHILD.returncode is None:
        with contextlib.suppress(OSError):
            os.killpg(CHILD.pid, sig)


def nap(seconds):
    end = time.time() + seconds
    while not STOP and time.time() < end:
        time.sleep(min(0.25, max(0.0, end - time.time())))
    if STOP:
        raise Cancelled()


def decide(job, before):
    """One decision, under the lock: may the job start now?  If so its reservation is written before returning."""
    jobs = [j for j in load_jobs(clean=True) if j["id"] != job["id"]] + [job]
    cards = look(jobs) if before else None
    now = job["beat"] = time.time()
    gpu, job["reason"] = None, "nvidia-smi failed, looking again"
    if cards:
        mark_busy(before, cards, jobs)
        gpu, job["reason"] = schedule(jobs, cards, now)[job["id"]]
    if gpu is None:
        save(job)                         # the ticket, with the time of this look and the reason for gq status
        return False
    on = [j for j in jobs if j["gpu"] == gpu]
    save(dict(job, gpu=gpu, t_start=now))                # the reservation
    job.update(gpu=gpu, t_start=now)
    log("start", job, reserved_before=sum(j["need"] for j in on), room=cards[gpu]["total"] - taken(cards[gpu], on),
        **{k: cards[gpu][k] for k in ("total", "used", "foreign")})
    return True


def admit(job):
    """Wait until the job may start.  On return its reservation (job["gpu"]) is on disk."""
    trouble = None
    while True:
        try:
            before = look(load_jobs())
            most = {c: k["total"] - (k["used"] - k["smi_used"]) for c, k in (before or {}).items()}     # empty card
            if before and all(most[c] < job["need"] + MARGIN for c in most if job["want"] in (None, c)):
                sizes = ", ".join(f"card {c}: {most[c]} MiB when empty" for c in sorted(most))
                say(f"{job['name']}: {job['need']}+{MARGIN} MiB can never start on --gpu "
                    f"{'any' if job['want'] is None else job['want']} ({sizes})")
                sys.exit(2)
            nap(LOOK)
            with locked():
                if decide(job, before):
                    return
        except Exception as e:            # the disk is full, the lock is stuck, ...: a waiting job does not die of it
            if repr(e) != trouble:
                trouble = repr(e)
                say(f"{job['name']}: {trouble}; still waiting, looking again every {POLL:.0f} s")
            job["reason"] = f"gq cannot decide: {trouble}"
            with contextlib.suppress(Exception):
                save(job)
        nap(POLL - LOOK)


def attend(job, t0):
    """Stay with the running command until it has exited.  Meanwhile pass its stderr on to ours, keep what arrives
    in the first OOM_WINDOW s in a file as well, and write the reservation again every BEAT / 2 s (it comes back
    if somebody has emptied the state directory)."""
    fds, copy, saved, gone = ([CHILD.stderr.fileno()] if CHILD.stderr else []), None, t0, None
    if fds:
        with contextlib.suppress(OSError):
            copy = open(path_of(job["id"], "err", "log"), "wb")
    while True:
        ready = select.select(fds, [], [], 0.5 if fds else 0.2)[0]
        data = os.read(fds[0], 65536) if ready else b""
        if ready and not data:
            fds = []                      # end of its stderr; the command itself may live on
        with contextlib.suppress(OSError):               # our stderr is gone or the disk is full: go on reading
            rest = data
            while rest:
                rest = rest[os.write(2, rest):]
            if data and copy and time.time() < t0 + OOM_WINDOW:
                copy.write(data)
                copy.flush()
        if CHILD.poll() is not None:      # exited; something it started may still hold its stderr: 2 s for the rest
            gone = gone or time.time()
            if not data or time.time() - gone > 2:
                break
        elif time.time() - saved > BEAT / 2:
            saved = time.time()
            with contextlib.suppress(OSError):
                save(job)
    if copy:
        copy.close()


def execute(job, cmd, tee):
    """Run the command on the reserved card.  Returns (exit code, died of CUDA out of memory early?, seconds)."""
    global CHILD
    me, t0 = os.getpid(), time.time()

    def tie():                            # runs in the new process, just before the command replaces it
        LIBC.prctl(1, signal.SIGTERM)     # PR_SET_PDEATHSIG: the command gets SIGTERM if this gq process is killed
        if os.getppid() != me:
            os._exit(143)

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), CUDA_DEVICE_ORDER="PCI_BUS_ID", GQ_MEM=str(job["need"]))
    try:                                  # own session: a signal for the job reaches all processes of the command
        CHILD = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE if tee else None, start_new_session=True,
                                 preexec_fn=tie)
    except OSError as e:
        say(f"{job['name']}: cannot start the command: {e}")
        return 126, False, 0.0
    if STOP:
        on_signal(STOP, None)             # a signal that arrived while the command was being started
    attend(job, t0)                       # signals are passed on by on_signal meanwhile
    code, ran, oom = CHILD.wait(), time.time() - t0, False
    if CHILD.stderr:
        CHILD.stderr.close()
    if tee and code != 0 and ran < OOM_WINDOW:
        with contextlib.suppress(OSError), open(path_of(job["id"], "err", "log"), "rb") as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - 4096))
            oom = b"out of memory" in f.read().lower()
    return (128 - code if code < 0 else code), oom, ran


def cmd_run(a, cmd):
    now = time.time()
    name = "_".join((a.name or f"{os.path.basename(cmd[0])}.{os.getpid()}").split())
    with contextlib.suppress(Exception):  # central priority policy: state/prio_override.json = {"name prefix": priority}
        with open(os.path.join(STATE, "prio_override.json")) as f:
            for pre, pr in sorted(json.load(f).items(), key=lambda kv: -len(kv[0])):
                if name.startswith(pre):
                    a.prio = int(pr)
                    break
    job = dict(id=f"{int(now * 1000)}-{os.getpid()}", name=name, pid=os.getpid(), pstart=pstat(os.getpid())[1],
               need=a.mem, want=None if a.gpu == "any" else int(a.gpu), prio=a.prio, t=now, beat=now, not_before=0,
               gpu=None, t_start=None, reason="new: first look pending", cmd=" ".join(cmd)[:200], cap=CAPS)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        if signal.getsignal(s) != signal.SIG_IGN:        # under nohup SIGHUP stays ignored
            signal.signal(s, on_signal)
    retry, since = not a.no_retry, now
    with contextlib.suppress(OSError):    # (a full disk: the ticket is written again at every look)
        save(job)
    log("queued", job, prio=a.prio, want=job["want"])
    say(f"queued {name}: needs {a.mem} MiB, card {a.gpu}, priority {a.prio}")
    try:
        while True:
            admit(job)
            if STOP:
                raise Cancelled()
            say(f"started {name} on card {job['gpu']} after {job['t_start'] - since:.0f} s in the queue: {job['cmd']}")
            code, oom, ran = execute(job, cmd, retry)
            card = job["gpu"]
            job.update(gpu=None, t_start=None)           # the reservation ends here (written or removed below)
            if oom and retry and not STOP:
                retry, since = False, time.time()
                job.update(not_before=since + OOM_WAIT, beat=since, reason="queued again after CUDA out of memory")
                with contextlib.suppress(OSError):
                    save(job)
                log("requeued", job, code=code, card=card, ran=round(ran, 1))
                say(f"{name} ran out of CUDA memory after {ran:.0f} s on card {card} (exit code {code}): "
                    f"queued once more, next try in {OOM_WAIT:.0f} s")
                continue
            log("exit", job, code=code, card=card, ran=round(ran, 1))
            say(f"{name} exit code {code} after {ran:.0f} s on card {card}")
            return code
    except Cancelled:
        log("cancelled", job)
        say(f"{name} cancelled by signal {STOP} after {time.time() - since:.0f} s in the queue")
        return 128 + STOP
    finally:
        if CHILD is not None and CHILD.returncode is None:
            CHILD.wait()                  # never give the card away while the command is still there
        forget(job["id"])


# ----------------------------------------------------------------------------- gq status, gq wait-idle
def hms(seconds):
    s = int(max(seconds, 0))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def table(rows):
    """Print rows as aligned columns: first column left, the others right, the last one as it is."""
    width = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]) - 1)]
    for r in rows:
        cells = [str(r[0]).ljust(width[0])] + [str(x).rjust(w) for x, w in zip(r[1:-1], width[1:])] + [str(r[-1])]
        print("  ".join(cells).rstrip())


def cmd_status():
    try:
        with locked(5):
            jobs, stuck = load_jobs(clean=True), ""
    except TimeoutError as e:
        jobs, stuck = load_jobs(), f"\nWARNING: {e}; no job can start meanwhile (is a gq process stopped?)"
    cards, now = look(jobs), time.time()
    running = sorted((j for j in jobs if j["gpu"] is not None), key=lambda j: j["t_start"])
    waiting = sorted((j for j in jobs if j["gpu"] is None), key=lambda j: (-j["prio"], ticket(j)))
    print(f"gq status {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(running)} running, {len(waiting)} waiting "
          f"(MiB; FITS = largest --mem that could start on the card now){stuck}")
    rows = [("CARD", "TOTAL", "USED", "FREE", "RESERVED", "FOREIGN", "FITS", "NOTE")]
    for c in sorted(cards or {}):
        card, on = cards[c], [j for j in running if j["gpu"] == c]
        note = ([f"busy: {card['young']}"] if card["young"] else []) \
            + (["foreign pid:MiB " + " ".join(card["pids"])] if card["pids"] else [])
        rows.append((c, card["total"], card["smi_used"], card["free"], sum(j["need"] for j in on), card["foreign"],
                     max(card["total"] - taken(card, on) - MARGIN, 0), "; ".join(note)))
    table(rows if cards else rows + [("?", "", "", "", "", "", "", "nvidia-smi failed")])
    rows = [("RUNNING", "PID", "CARD", "RESERVED", "ACTUAL", "RUNTIME", "PRIO", "COMMAND")]
    for j in running:
        actual = cards[j["gpu"]]["own"].get(j["id"], "-") if cards and j["gpu"] in cards else "?"
        over = "!" if isinstance(actual, int) and actual > j["need"] else ""
        short = " ".join(os.path.basename(x) or x for x in j["cmd"].split(" "))    # paths without their directories
        rows.append((j["name"], j["pid"], j["gpu"], j["need"], f"{actual}{over}", hms(now - j["t_start"]), j["prio"],
                     short[:70]))
    table(rows)
    rows = [("WAITING", "PID", "NEED", "CARD", "PRIO", "WAITED", "WHY IT HAS NOT STARTED")]
    for j in waiting:
        late = f" [its last look was {now - j['beat']:.0f} s ago]" if now - j["beat"] > 2 * POLL + LOOK else ""
        rows.append((j["name"], j["pid"], j["need"], "any" if j["want"] is None else j["want"], j["prio"],
                     hms(now - j["t"]), j["reason"] + late))
    table(rows)
    return 0 if cards else 1


def cmd_wait_idle(timeout):
    """Return 0 when no queue job is running or waiting, 1 if there still is one after `timeout` seconds.
    The queue counts as idle when it is empty on two looks 1 s apart: a `gq run ... &` that a script has just
    started, or the next step of a chain, needs a moment to write its ticket."""
    t0, empty = time.time(), 0
    while True:
        jobs = load_jobs()
        empty = 0 if jobs else empty + 1
        if empty >= 2:
            return 0
        pause = 2.0 if jobs else 1.0
        if timeout is not None:
            left = t0 + timeout - time.time()
            if left <= 0:
                if jobs:
                    print(f"gq wait-idle: still {len(jobs)} job(s) after {timeout:.0f} s", file=sys.stderr)
                return 1 if jobs else 0
            pause = min(pause, max(left, 0.05))
        time.sleep(pause)


def main(argv):
    cmd = []
    if "--" in argv:
        argv, cmd = argv[:argv.index("--")], argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser(prog="gq", description="Shared GPU job queue; see README.txt next to gq.py.")
    sub = p.add_subparsers(dest="what", required=True)
    r = sub.add_parser("run", usage="gq run --mem MiB [--gpu any|N] [--name TAG] [--prio INT] [--no-retry] -- command ...")
    r.add_argument("--mem", type=int, required=True, help="MiB the job may take on its card, as nvidia-smi counts")
    r.add_argument("--gpu", default="any", help="any (default) or a card index")
    r.add_argument("--name", help="tag shown by gq status")
    r.add_argument("--prio", type=int, default=0, help="higher is served first (default 0)")
    r.add_argument("--no-retry", action="store_true", help="do not queue again after an early CUDA out of memory")
    sub.add_parser("status")
    w = sub.add_parser("wait-idle")
    w.add_argument("--timeout", type=float, help="seconds; exit code 1 if the queue is still not empty then")
    a = p.parse_args(argv)
    os.makedirs(STATE, exist_ok=True)
    if a.what == "run":
        if not cmd or not (a.gpu == "any" or a.gpu.isdigit()):
            r.error("needed: --mem MiB, --gpu any|N, and the command after --")
        if a.mem < 256:
            r.error(f"--mem {a.mem}: the unit is MiB (a CUDA context alone takes about 500 MiB)")
        if shutil.which(cmd[0]) is None:
            say(f"command not found or not executable: {cmd[0]}")
            return 127
        return cmd_run(a, cmd)
    return cmd_status() if a.what == "status" else cmd_wait_idle(a.timeout)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
