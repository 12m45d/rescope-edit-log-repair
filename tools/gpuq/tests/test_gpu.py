#!/usr/bin/env python3
"""GPU test of gq: dummy torch jobs (tests/gpu_job.py) through the real queue on a real card.  About 6 minutes.

    python tests/test_gpu.py [--budget MiB] [--card N]        exit code 0: all checks passed, 1: a check failed,
    python tests/test_gpu.py --dry [--slow]                                 2: not started (not enough room now)

The test jobs go through the live queue like every other job (same state directory, 15 s polling, two looks 10 s
apart), so the queue itself keeps them and the real jobs apart.  They are small and polite:
  * one card only, the one with the most room.  All test jobs together reserve at most --budget MiB of it
    (GQ_CAP; default 4000, i.e. jobs of 1000 to 3000 MiB), and the test starts only if the card then still has
    2500 MiB of room by the queue's rule (else exit code 2: try later);
  * every job looks at nvidia-smi before it allocates and leaves if less than 1500 MiB would stay free (exit 75);
  * priority -1000 and below: every real job is served first.  With GQ_STARVE = 60 s the test jobs also stop
    starting as soon as a real job that has waited for a minute could use the card without them;
  * every job caps itself (torch.cuda.set_per_process_memory_fraction) and says so on its command line.
Checks: the budget is never exceeded (reservations: replay of the event log; memory: nvidia-smi every 0.5 s),
FIFO, priority, backfill, anti-starvation (age limit 20 s instead of 15 min), a real torch out-of-memory failure
is queued again once, SIGTERM / SIGKILL of the wrapper free the reservation, gq status against an independent
reading of nvidia-smi, /proc and the job files.
--dry: the same scenario without a GPU (tests/fake_smi.py, tests/fake_job.py, a private state directory), five
times faster unless --slow.  GQ_DIR chooses the directory of the gq under test (default: the parent of tests/).
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True       # no __pycache__ in tests/
sys.path.insert(0, HERE)
from test_queue import parse_status  # noqa: E402

GQ_DIR = os.environ.get("GQ_DIR", os.path.dirname(HERE))
GQ, PY = os.path.join(GQ_DIR, "gq"), sys.executable
MARGIN, KEEP_FREE, ROOM_LEFT, BASE_PRIO, TIME_LIMIT = 700, 1500, 2500, -1000, 540


class Stalled(Exception):
    """Something did not happen in time (the scenario is given up; the reason is reported)."""


def proc(pid):
    """(parent pid, start time) of a process, None if it is gone"""
    try:
        with open(f"/proc/{pid}/stat") as f:
            a = f.read().rsplit(")", 1)[1].split()
        return (int(a[1]), int(a[19])) if a[0] not in "ZX" else None
    except (OSError, IndexError, ValueError):
        return None


def ancestors(pid):
    out = []
    while pid > 1 and len(out) < 100 and proc(pid):
        out.append(pid)
        pid = proc(pid)[0]
    return out


class Test:
    def __init__(self, a):
        self.dry, self.S = a.dry, (1.0 if a.slow or not a.dry else 0.2)      # S: time scale of the scenario
        os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="gpu_", dir=os.path.join(HERE, "tmp"))
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GQ_")}
        self.job = [PY, os.path.join(HERE, "gpu_job.py")]
        if self.dry:
            world = os.path.join(self.dir, "world.json")
            with open(world, "w") as f:
                json.dump({"gpus": [[0, "GPU-0", 32607], [1, "GPU-1", 32607]], "apps": [[0, 1, 20000], [1, 1, 12000]]}, f)
            self.env.update(GQ_STATE=os.path.join(self.dir, "state"), GQ_FAKE_WORLD=world,
                            GQ_SMI=f"{PY} {os.path.join(HERE, 'fake_smi.py')} {world}",
                            GQ_POLL=str(15 * self.S), GQ_LOOK=str(10 * self.S))
            self.job = [PY, os.path.join(HERE, "fake_job.py")]
        self.state = self.env.get("GQ_STATE", f"{_RP.ROOT}/02_research/tools/gpuq/state")
        os.makedirs(self.state, exist_ok=True)
        self.prefix = f"gqtest{os.getpid()}_"
        self.procs, self.spec, self.results, self.reasons = {}, {}, [], {}
        self.max_mem = self.max_res = self.elsewhere = self.sample_errors = 0
        self.min_free, self.stop, self.t0 = 10 ** 9, False, time.time()
        try:
            self.offset = os.path.getsize(os.path.join(self.state, "events.log"))
        except OSError:
            self.offset = 0
        self.choose_card(a)
        self.u = self.B // 4

    # ------------------------------------------------------------------ readings (independent of gq)
    def smi(self, query):
        out = subprocess.run(shlex.split(self.env.get("GQ_SMI", "nvidia-smi")) + [query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]

    def reading(self):
        """({card: {total, used, free}}, [(card, pid, MiB)]) from nvidia-smi"""
        rows = self.smi("--query-gpu=index,uuid,memory.total,memory.used,memory.free")
        gpus = {int(i): dict(total=int(t), used=int(m), free=int(f)) for i, _, t, m, f in rows}
        card_of = {u: int(i) for i, u, _, _, _ in rows}
        return gpus, [(card_of[u], int(p), int(m)) for u, p, m in self.smi("--query-compute-apps=gpu_uuid,pid,used_memory")]

    def job_files(self):
        """{name: job} of the job files whose owner is alive"""
        out = {}
        for n in os.listdir(self.state):
            if n.startswith("job-") and n.endswith(".json"):
                try:
                    with open(os.path.join(self.state, n)) as f:
                        j = json.load(f)
                    if (proc(j["pid"]) or (0, 0))[1] == j["pstart"]:
                        out[j["name"]] = j
                except (OSError, ValueError, KeyError):
                    pass
        return out

    def gq(self, *args):
        return subprocess.run([GQ, *args], env=self.env, capture_output=True, text=True, timeout=120)

    def events(self, kind=None, label=None, mine=True):
        out = []
        try:
            with open(os.path.join(self.state, "events.log")) as f:
                f.seek(self.offset)
                for line in f:
                    e = json.loads(line)
                    if e["name"].startswith(self.prefix) == mine and kind in (None, e["event"]) \
                            and label in (None, e["name"][len(self.prefix):]):
                        out.append(e)
        except (OSError, ValueError):
            pass
        return out

    def index(self, kind, label):
        """position of the (last) event of this kind in the queue's log, None if there is none"""
        hits = [i for i, e in enumerate(self.events()) if e["event"] == kind and e["name"] == self.prefix + label]
        return hits[-1] if hits else None

    def when(self, kind, label):
        ev = self.events(kind, label)
        return ev[-1]["t"] if ev else None

    def mem(self, label):
        """MiB that the processes of a test job hold on the test card now, and their pids"""
        mine = [(p, m) for c, p, m in self.reading()[1] if c == self.card and self.procs[label].pid in ancestors(p)]
        return sum(m for _, m in mine), [p for p, _ in mine]

    # ------------------------------------------------------------------ start: look first, choose the card
    def choose_card(self, a):
        r = self.gq("status")
        print(r.stdout.rstrip(), flush=True)
        if r.returncode != 0:
            sys.exit("gq status failed: " + r.stderr)
        if any(n.startswith("gqtest") for n in self.job_files()):
            print("another test_gpu.py seems to be running (gqtest jobs in the queue): not started")
            sys.exit(2)
        cards = parse_status(r.stdout)[0]
        self.B = a.budget
        ok = {int(c): int(row["FITS"]) for c, row in cards.items() if a.card in (None, int(c))
              and int(row["FITS"]) >= self.B + ROOM_LEFT and int(row["FREE"]) >= self.B + KEEP_FREE + 1000}
        if not ok:
            print(f"not started: no card has room for a test budget of {self.B} MiB next to the real jobs now (needed: FITS "
                  f">= {self.B + ROOM_LEFT} and FREE >= {self.B + KEEP_FREE + 1000}).  Try later, or a smaller --budget.")
            sys.exit(2)
        self.card = max(ok, key=ok.get)
        print(f"\ntest card {self.card}: FITS {ok[self.card]} MiB; test budget {self.B} MiB, test jobs of {self.B // 4} to "
              f"{3 * self.B // 4} MiB; time scale {self.S}; job names {self.prefix}*\n", flush=True)

    # ------------------------------------------------------------------ test jobs
    def submit(self, label, need, secs, prio=0, starve=60, extra=()):
        """gq run a dummy job that takes `need` - 150 MiB for `secs` seconds (times are scaled in --dry)"""
        mib = need - 150
        cmd = [GQ, "run", "--mem", str(need), "--gpu", str(self.card), "--prio", str(BASE_PRIO + prio), "--name",
               self.prefix + label, "--", *self.job, "--mib", str(mib), "--secs", str(secs * self.S),
               "--cap_gb", f"{(mib - 480) / 1024:.3f}", "--tag", self.prefix, *extra]
        env = dict(self.env, GQ_CAP=f"{self.card}:{self.B}", GQ_STARVE=str(starve * self.S), GQ_OOM_WAIT=str(20 * self.S))
        with open(os.path.join(self.dir, label + ".out"), "w") as out, open(os.path.join(self.dir, label + ".err"), "w") as err:
            self.procs[label] = subprocess.Popen(cmd, env=env, stdout=out, stderr=err)
        self.spec[label] = dict(need=need, mib=mib, prio=BASE_PRIO + prio)
        self.say(f"submitted {label}: --mem {need}, priority {BASE_PRIO + prio}, runs {secs * self.S:.0f} s")
        time.sleep(0.1)

    def say(self, text):
        print(f"[{time.time() - self.t0:5.0f} s] {text}", flush=True)

    def check(self, name, ok, detail=""):
        self.results.append((name, bool(ok), str(detail)))
        self.say(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))

    def wait(self, what, cond, timeout):
        end = min(time.time() + timeout * max(self.S, 0.5), self.t0 + TIME_LIMIT * self.S)
        while time.time() < end:
            got = cond()
            if got:
                return got
            time.sleep(0.2)
        why = "; ".join(f"{n[len(self.prefix):]}: {j['reason']}" for n, j in self.job_files().items()
                        if n.startswith(self.prefix) and j["gpu"] is None)
        raise Stalled(f"{what}: not within {timeout * max(self.S, 0.5):.0f} s (or the time limit of the test is reached)"
                      + (f".  Waiting test jobs: {why}" if why else ""))

    def wait_started(self, *labels, timeout=75):
        self.wait(f"start of {', '.join(labels)}", lambda: all(self.when("start", x) for x in labels), timeout)

    def wait_memory(self, *labels, timeout=40):
        """until the dummy jobs have allocated (nvidia-smi shows their memory)"""
        self.wait(f"GPU memory of {', '.join(labels)}", lambda: all(self.mem(x)[0] >= self.spec[x]["mib"] - 150 for x in labels),
                  timeout)

    def wait_done(self, *labels, timeout=200):
        self.wait(f"end of {', '.join(labels)}", lambda: all(self.procs[x].poll() is not None for x in labels), timeout)
        codes = {x: self.procs[x].returncode for x in labels}
        self.check(f"exit code 0: {', '.join(labels)}", set(codes.values()) == {0}, "" if set(codes.values()) == {0} else codes)

    def tickets(self, *labels):
        """the labels in ticket order (the job id starts with the ticket time in ms)"""
        self.wait("tickets", lambda: all(self.events("queued", x) for x in labels), 20)
        return sorted(labels, key=lambda x: self.events("queued", x)[0]["id"])

    def sample(self):
        """background: what the test jobs hold on the cards, every 0.5 s"""
        tagged = {}
        while not self.stop:
            try:
                gpus, apps = self.reading()
                mine = dict.fromkeys(gpus, 0)
                for c, pid, mib in apps:
                    if pid not in tagged:
                        try:
                            with open(f"/proc/{pid}/cmdline", "rb") as f:
                                tagged[pid] = self.prefix.encode() in f.read()
                        except OSError:
                            continue                     # gone already
                    mine[c] += mib if tagged[pid] else 0
                files = {n: j for n, j in self.job_files().items() if n.startswith(self.prefix)}
                self.max_mem = max(self.max_mem, mine[self.card])
                self.max_res = max(self.max_res, sum(j["need"] for j in files.values() if j["gpu"] is not None))
                self.elsewhere = max([self.elsewhere] + [m for c, m in mine.items() if c != self.card]
                                     + [j["need"] for j in files.values() if j["gpu"] not in (None, self.card)])
                if mine[self.card]:
                    self.min_free = min(self.min_free, gpus[self.card]["free"])
                for n, j in files.items():
                    self.reasons.setdefault(n[len(self.prefix):], set()).add(j["reason"])
            except (OSError, ValueError, KeyError, subprocess.SubprocessError):
                self.sample_errors += 1
            time.sleep(0.5 * max(self.S, 0.4))

    def seen(self, label, text):
        return any(text in r for r in self.reasons.get(label, ()))

    # ------------------------------------------------------------------ gq status against an independent reading
    def check_status(self, title, running=(), waiting=None):
        """running: labels expected in the RUNNING table; waiting: {label: regex of the reason}"""
        problems = ["no stable reading"]
        for _ in range(5):                               # the real jobs on the cards change while we look
            g1, a1 = self.reading()
            files = self.job_files()
            now, r = time.time(), self.gq("status")
            g2, a2 = self.reading()
            if files.keys() != self.job_files().keys() or {(c, p) for c, p, _ in a1} != {(c, p) for c, p, _ in a2}:
                continue
            cards, run, wait = parse_status(r.stdout)
            problems, tol = [], 60

            def near(shown, x, y, what):
                if not min(x, y) - tol <= int(shown) <= max(x, y) + tol:
                    problems.append(f"{what}: status {shown}, nvidia-smi {x}..{y}")

            for c in g1:
                row, on = cards[str(c)], [j for j in files.values() if j["gpu"] == c]
                owners = {j["pid"] for j in on}
                foreign = [sum(m for cc, p, m in apps if cc == c and not owners & set(ancestors(p))) for apps in (a1, a2)]
                near(row["USED"], g1[c]["used"], g2[c]["used"], f"card {c} USED")
                near(row["FREE"], g1[c]["free"], g2[c]["free"], f"card {c} FREE")
                near(row["FOREIGN"], *foreign, f"card {c} FOREIGN")
                if int(row["TOTAL"]) != g1[c]["total"] or int(row["RESERVED"]) != sum(j["need"] for j in on):
                    problems.append(f"card {c}: TOTAL / RESERVED {row['TOTAL']} / {row['RESERVED']}, expected "
                                    f"{g1[c]['total']} / {sum(j['need'] for j in on)}")
                own = sum(max(int(x["RESERVED"]), int(x["ACTUAL"].strip("!-?") or 0)) for x in run.values() if x["CARD"] == str(c))
                fits = max(0, int(row["TOTAL"]) - max(int(row["TOTAL"]) - int(row["FREE"]), int(row["FOREIGN"]) + own) - MARGIN)
                if int(row["FITS"]) != fits:
                    problems.append(f"card {c} FITS {row['FITS']}, by the rule {fits}")
            for x in running:
                row, s = run.get(self.prefix + x), self.spec[x]
                if row is None:
                    problems.append(f"{x} is not in the RUNNING table")
                    continue
                got = (row["PID"], row["CARD"], row["RESERVED"], row["PRIO"])
                if got != (str(self.procs[x].pid), str(self.card), str(s["need"]), str(s["prio"])):
                    problems.append(f"{x}: PID / CARD / RESERVED / PRIO {got}")
                if not row["ACTUAL"].isdigit() or abs(int(row["ACTUAL"]) - s["mib"]) > 150:
                    problems.append(f"{x}: ACTUAL {row['ACTUAL']}, allocated about {s['mib']}")
                h, m, sec = map(int, row["RUNTIME"].split(":"))
                if abs(h * 3600 + m * 60 + sec - (now - self.when("start", x))) > 3:
                    problems.append(f"{x}: RUNTIME {row['RUNTIME']}, started {now - self.when('start', x):.0f} s ago")
                if not row["COMMAND"].startswith(f"python {os.path.basename(self.job[1])} --mib {s['mib']} "):
                    problems.append(f"{x}: COMMAND {row['COMMAND']}")
            for x, why in (waiting or {}).items():
                row, s = wait.get(self.prefix + x), self.spec[x]
                if row is None:
                    problems.append(f"{x} is not in the WAITING table")
                    continue
                got = (row["PID"], row["NEED"], row["CARD"], row["PRIO"])
                if got != (str(self.procs[x].pid), str(s["need"]), str(self.card), str(s["prio"])):
                    problems.append(f"{x}: PID / NEED / CARD / PRIO {got}")
                h, m, sec = map(int, row["WAITED"].split(":"))
                if abs(h * 3600 + m * 60 + sec - (now - self.when("queued", x))) > 3:
                    problems.append(f"{x}: WAITED {row['WAITED']}, queued {now - self.when('queued', x):.0f} s ago")
                if not re.search(why, row["WHY"]):
                    problems.append(f"{x}: reason '{row['WHY']}', expected /{why}/")
            if not problems:
                break
        self.check(f"gq status is correct ({title})", not problems, "; ".join(problems))
        if problems:
            print(r.stdout, flush=True)

    # ------------------------------------------------------------------ the scenarios (budget B = 4 u)
    def burst(self):
        """six launches at once, each 1.5 u: only two fit the budget at a time -> no overcommit"""
        need = 3 * self.u // 2
        labels = [f"b{i}" for i in range(1, 7)]
        for x in labels:
            self.submit(x, need, 8)
        order = self.tickets(*labels)
        self.wait_started(*order[:2])
        self.wait_memory(*order[:2])
        late = [x for x in order[2:] if self.when("start", x)]
        self.check("burst: of six jobs launched together only two start: the two with the earliest tickets", not late, late)
        self.check_status("two running, four waiting", order[:2],
                          {x: rf"^card {self.card}: room {self.B - 2 * need + MARGIN} MiB < {need}\+{MARGIN}" for x in order[2:]})
        self.wait_done(*labels)

    def order(self):
        """three launches at once, each 3 u: one fits at a time -> they run by (priority, ticket)"""
        for x, p in (("o1", -1), ("o2", -1), ("ohigh", 0)):
            self.submit(x, 3 * self.u, 3, prio=p)
        tickets = self.tickets("o1", "o2", "ohigh")
        self.wait_done("o1", "o2", "ohigh")
        idx = [self.index("start", x) for x in ("ohigh", "o1", "o2")]
        ends = [self.index("exit", x) for x in ("ohigh", "o1")]
        self.check("priority: the job with the latest ticket and the highest priority runs first", tickets[2] == "ohigh"
                   and idx[0] < min(idx[1:]), f"ticket order {tickets}")
        self.check("FIFO: jobs of equal priority run in ticket order, each after the one before has ended",
                   tickets[:2] == ["o1", "o2"] and idx[0] < ends[0] < idx[1] < ends[1] < idx[2], f"ticket order {tickets}")

    def backfill(self):
        """a blocker of 2 u runs; big (3 u) does not fit, small (u, later ticket) does and starts"""
        u = self.u
        self.submit("fx", 2 * u, 22)
        self.wait_started("fx")
        self.submit("fbig", 3 * u, 4)
        self.submit("fsmall", u, 6)
        order = self.tickets("fbig", "fsmall")
        self.wait_started("fsmall")
        self.check("backfill: the later, smaller job starts while the earlier, bigger one waits",
                   order == ["fbig", "fsmall"] and not self.when("start", "fbig") and not self.when("exit", "fx"),
                   f"ticket order {order}")
        self.wait_memory("fx", "fsmall", timeout=20)
        self.check_status("backfill", ["fx", "fsmall"], {"fbig": rf"^card {self.card}: room ({2 * u + MARGIN}|{u + MARGIN}) MiB < {3 * u}\+"})
        self.wait_done("fx", "fbig", "fsmall")
        self.check("backfill: the big job starts when the blocker has ended",
                   self.index("exit", "fx") < self.index("start", "fbig"))

    def starvation(self):
        """age limit 20 s.  old (3 u) waits; s1 (later ticket) is backfilled while old is young; when old is older
        than the limit, s2 (later ticket) fits but must not start; old starts as soon as s1 has ended, then s2"""
        u, lim = self.u, 20
        self.submit("sx", 2 * u, 12, starve=lim)
        self.wait_started("sx")
        self.submit("sold", 3 * u, 6, starve=lim)
        self.submit("s1", 3 * u // 2, 45, starve=lim)
        self.wait_started("s1")
        age = self.when("start", "s1") - self.when("queued", "sold")
        self.check("starvation rule: before the age limit a later job is still backfilled", age < lim * self.S and
                   not self.when("start", "sold"), f"the waiting job was {age:.0f} s old")
        self.wait("end of sx", lambda: self.when("exit", "sx"), 60)
        self.wait("age limit", lambda: time.time() - self.when("queued", "sold") > (lim + 2) * self.S, 60)
        self.submit("s2", u, 4, starve=lim)
        kept = f"card {self.card} is kept for the long-waiting job {self.prefix}sold"
        self.wait("s2 to look", lambda: self.seen("s2", "kept for") or self.when("start", "s2"), 60)
        self.check("starvation rule: after the age limit a later job that fits is not started", self.seen("s2", kept)
                   and not self.when("start", "s2") and not self.when("exit", "s1"), sorted(self.reasons.get("s2", ())))
        self.check_status("card kept for the old job", ["s1"],
                          {"s2": "^" + re.escape(kept) + "$", "sold": rf"^card {self.card}: room \d+ MiB < {3 * u}\+{MARGIN}$"})
        self.wait_done("sx", "sold", "s1", "s2", timeout=240)
        self.check("starvation rule: the old job gets the card when the job in its way ends, the held job after it",
                   self.index("exit", "s1") < self.index("start", "sold") < self.index("start", "s2"))

    def oom(self):
        """a job dies of a real torch out-of-memory error in its first seconds (it asks for more than its own cap;
        the card is not filled): it is queued again once, 20 s later (60 s by default), and then runs"""
        self.submit("oom", 2 * self.u, 3, extra=["--oom-once", os.path.join(self.dir, "oom.marker")])
        self.wait_done("oom", timeout=150)
        ev = self.events(label="oom")
        with open(os.path.join(self.dir, "oom.err")) as f:
            err = f.read()
        kinds = [e["event"] for e in ev]
        self.check("out of memory: the job is queued again once instead of failing, then runs",
                   kinds == ["queued", "start", "requeued", "start", "exit"] and ev[2]["code"] == 1 and ev[4]["code"] == 0
                   and "out of memory" in err.lower() and "queued once more" in err, kinds)
        gap = ev[3]["t"] - ev[2]["t"] if len(ev) > 3 else -1
        self.check("out of memory: the second try starts after the pause", gap >= 20 * self.S, f"{gap:.0f} s later")

    def kills(self):
        """SIGTERM, then SIGKILL of a wrapper whose job holds the room that the next job is waiting for"""
        u = self.u
        for victim, waiter, sig in (("v1", "w1", signal.SIGTERM), ("w1", "w2", signal.SIGKILL)):
            if victim == "v1":
                self.submit("v1", 3 * u, 100)
            self.wait_started(victim)
            self.wait_memory(victim)
            self.submit(waiter, 2 * u if waiter == "w1" else 3 * u, 100 if waiter == "w1" else 3)
            self.wait(f"{waiter} to look", lambda: self.seen(waiter, "room"), 45)
            self.check_status(f"before {sig.name}", [victim], {waiter: rf"^card {self.card}: room \d+ MiB < "})
            jid, child = self.events("queued", victim)[0]["id"], self.mem(victim)[1]
            t_kill = time.time()
            self.procs[victim].send_signal(sig)
            code = self.procs[victim].wait(30)
            path = os.path.join(self.state, f"job-{jid}.json")
            if sig == signal.SIGTERM:
                self.check("SIGTERM: the wrapper passes it on, exits with 143 and removes its reservation",
                           code == 143 and not os.path.exists(path), f"exit code {code}")
            else:
                listed = self.prefix + victim in parse_status(self.gq("status").stdout)[1]
                self.check("SIGKILL: the dead wrapper's reservation is ignored and removed by the next look",
                           code == -9 and not listed and not os.path.exists(path) and self.events("stale", victim),
                           f"exit code {code}, listed by gq status: {listed}")
            self.wait(f"the command of {victim} to end", lambda: not any(proc(p) for p in child), 20)
            self.check(f"{sig.name}: the command is gone and its GPU memory is free",
                       child and not any(p in child for _, p, _ in self.reading()[1]), f"its processes: {child}")
            self.wait_started(waiter, timeout=60)
            self.check(f"{sig.name}: the waiting job gets the freed room", self.when("start", waiter) > t_kill,
                       f"{self.when('start', waiter) - t_kill:.0f} s after the signal")
        self.wait_done("w2")

    # ------------------------------------------------------------------ run
    def cleanup(self):
        """end what is left of the test's own jobs"""
        alive = [p for p in self.procs.values() if p.poll() is None]
        for p in alive:
            p.terminate()
        end = time.time() + 20
        for p in alive:
            try:
                p.wait(max(0.1, end - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        return len(alive)

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))            # a stopped test still ends its own jobs
        threading.Thread(target=self.sample, daemon=True).start()
        try:
            for scenario in (self.burst, self.order, self.backfill, self.starvation, self.oom, self.kills):
                self.say(f"--- {scenario.__name__}: {' '.join(scenario.__doc__.split())}")
                try:
                    scenario()
                except Stalled as e:
                    self.check(f"{scenario.__name__}: completed", False, e)
                    self.cleanup()
                if time.time() > self.t0 + TIME_LIMIT * self.S:
                    self.check("time limit of the test", False, "the remaining scenarios were not run")
                    break
        finally:
            left = self.cleanup()
            time.sleep(1.0)
            self.stop = True
        self.final(left)

    def final(self, left):
        held, top = {}, 0                                # replay of the queue's own log
        for e in self.events():
            if e["event"] == "start":
                held[e["id"]] = e["need"]
                top = max(top, sum(held.values()))
            elif e["event"] in ("exit", "requeued", "cancelled", "stale"):
                held.pop(e["id"], None)
        ran = [x for x in self.procs if self.when("start", x)]
        cards = set()
        for x in ran:
            with open(os.path.join(self.dir, x + ".out")) as f:
                cards |= set(re.findall(r"card=(\d+)", f.read()))
        codes = {x: p.returncode for x, p in self.procs.items() if p.returncode not in (0, 143, -9)}
        mine = [n for n in os.listdir(self.state) if any(e["id"] in n for e in self.events("queued"))]
        self.check(f"budget: the test jobs never reserved more than {self.B} MiB together", top <= self.B and self.max_res <= self.B,
                   f"most reserved: {top} MiB by the event log, {self.max_res} MiB in the samples")
        self.check(f"budget: the test jobs never held more than {self.B} MiB of GPU memory together",
                   0 < self.max_mem <= self.B, f"most held: {self.max_mem} MiB")
        self.check(f"the test never left less than {KEEP_FREE} MiB free on its card", self.min_free >= KEEP_FREE,
                   f"least free while test jobs held memory: {self.min_free} MiB")
        self.check("no test job touched the other card", self.elsewhere == 0 and (self.dry or cards == {str(self.card)}), cards)
        unplanned = [e["name"] for e in self.events("requeued") if not e["name"].endswith("_oom")]
        self.check("no other test job ran out of memory or refused to start", not unplanned and not codes, unplanned or codes)
        self.check("nothing of the test is left in the queue", not mine and not left, mine or f"{left} jobs had to be ended")
        real = [f"{e['time'][11:]} {e['event']} {e['name']}" + (f" card {e['gpu']}" if e["event"] == "start" else "")
                for e in self.events(mine=False)]
        bad = [r for r in self.results if not r[1]]
        print(f"\nGPU TEST {'(dry) ' if self.dry else ''}{'FAILED' if bad else 'PASSED'}: {len(self.results) - len(bad)} of "
              f"{len(self.results)} checks passed in {time.time() - self.t0:.0f} s; card {self.card}, budget {self.B} MiB, "
              f"{len(ran)} test jobs ran, most reserved {top} MiB, most held {self.max_mem} MiB, least free {self.min_free} MiB"
              + (f"; {self.sample_errors} unreadable samples" if self.sample_errors else ""))
        for name, _, detail in bad:
            print(f"  FAIL  {name}  [{detail}]")
        if real:
            print(f"Events of real jobs in the queue during the test ({len(real)}): " + "; ".join(real[:12])
                  + ("" if not bad else "\n(a failed timing check can be due to them: run the test again)"))
        if not bad:
            shutil.rmtree(self.dir, ignore_errors=True)
        else:
            print(f"logs of the test jobs: {self.dir}")
        return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=4000, help="MiB that all test jobs together may reserve (default 4000)")
    ap.add_argument("--card", type=int, help="use this card (default: the one with the most room)")
    ap.add_argument("--dry", action="store_true", help="no GPU: fake nvidia-smi and fake jobs in a private state directory")
    ap.add_argument("--slow", action="store_true", help="with --dry: real intervals (15 s polling)")
    args = ap.parse_args()
    if args.budget < 2000 or args.budget % 400:
        ap.error("--budget: a multiple of 400, at least 2000")
    test = Test(args)
    test.run()
    sys.exit(1 if any(not r[1] for r in test.results) else 0)
