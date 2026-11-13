#!/usr/bin/env python3
"""End-to-end tests of gq with a fake nvidia-smi (tests/fake_smi.py) and jobs that only pretend to use a GPU
(tests/fake_job.py).  No GPU is touched; everything runs in a private state directory under tests/tmp/, with
short intervals (decision every 0.5 s instead of 15 s).  Run: python tests/test_queue.py [-v] [TestClass.test]
GQ_DIR chooses the directory of the gq under test (default: the parent of tests/).
"""
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GQ_DIR = os.environ.get("GQ_DIR", os.path.dirname(HERE))
GQ = os.path.join(GQ_DIR, "gq")
PY = sys.executable
TOTAL, DRV, MARGIN = 32607, 497, 700
QWEN, LLAVA = 10500, 17000
FAKE_JOB = [PY, os.path.join(HERE, "fake_job.py")]


def until(cond, timeout=20.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        got = cond()
        if got:
            return got
        time.sleep(step)
    return cond()


def parse_status(text):
    """The output of gq status as ({card: row}, {running job: row}, {waiting job: row}); a row is {column: text}."""
    parts, key, head = {"CARD": {}, "RUNNING": {}, "WAITING": {}}, None, None
    for line in text.splitlines()[1:]:
        if line.startswith("WARNING"):
            continue
        if line.split()[0] in parts:
            key, head = line.split()[0], line.split()
            continue
        if key == "WAITING":                             # the last column (the reason) contains spaces
            cells = line.split(None, 6)
            row = dict(zip(["WAITING", "PID", "NEED", "CARD", "PRIO", "WAITED", "WHY"], cells + [""]))
        else:
            cells = line.split(None, len(head) - 1)
            row = dict(zip(head, cells + [""] * (len(head) - len(cells))))
        parts[key][cells[0]] = row
    return parts["CARD"], parts["RUNNING"], parts["WAITING"]


class Box:
    """A private queue: its own state directory, fake cards (fake_smi.py), and the gq processes started in it."""

    def __init__(self, cards=(TOTAL, TOTAL), apps=(), **knobs):
        os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)
        self.dir = tempfile.mkdtemp(prefix="e2e_", dir=os.path.join(HERE, "tmp"))
        self.state, self.path = os.path.join(self.dir, "state"), os.path.join(self.dir, "world.json")
        os.makedirs(self.state)
        self.world = {"gpus": [[i, f"GPU-{i}", t] for i, t in enumerate(cards)], "apps": [list(a) for a in apps],
                      "hidden": {}, "fail": False}
        self.put()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GQ_")}
        self.env.update(GQ_STATE=self.state, GQ_SMI=f"{PY} {os.path.join(HERE, 'fake_smi.py')} {self.path}",
                        GQ_FAKE_WORLD=self.path, GQ_POLL="0.5", GQ_LOOK="0.25", GQ_OOM_WAIT="1.5")
        self.env.update({f"GQ_{k}": str(v) for k, v in knobs.items()})
        self.procs, self.extra = {}, []

    def put(self, **change):
        """(re)write the fake world, e.g. put(fail=True) or put(hidden={"0": 4000})"""
        self.world.update(change)
        with open(self.path + ".tmp", "w") as f:
            json.dump(self.world, f)
        os.replace(self.path + ".tmp", self.path)

    def gq(self, *args, timeout=60):
        return subprocess.run([GQ, *map(str, args)], env=self.env, capture_output=True, text=True, timeout=timeout)

    def run(self, name, mem, *job, gpu=None, prio=None, flags=(), cmd=None, ignore=(), env=None):
        """gq run ... -- fake_job <job args> in the background (or the command `cmd`); returns the Popen of gq"""
        args = [GQ, "run", "--mem", str(mem), "--name", name]
        args += ["--gpu", str(gpu)] if gpu is not None else []
        args += ["--prio", str(prio)] if prio is not None else []
        args += [*flags, "--", *(cmd or FAKE_JOB + [str(x) for x in job])]
        def signals():                    # the same in every environment the tests are started from (nohup, &)
            for s in (signal.SIGINT, signal.SIGQUIT, signal.SIGHUP):
                signal.signal(s, signal.SIG_IGN if s in ignore else signal.SIG_DFL)

        with open(os.path.join(self.dir, name + ".out"), "w") as out, open(os.path.join(self.dir, name + ".err"), "w") as err:
            self.procs[name] = subprocess.Popen(args, env=dict(self.env, **(env or {})), stdout=out, stderr=err,
                                                preexec_fn=signals)
        return self.procs[name]

    def pids(self):
        """pids of the fake jobs that hold (fake) GPU memory now"""
        try:
            return [int(n) for n in os.listdir(self.path + ".d") if n.isdigit() and os.path.exists(f"/proc/{n}")]
        except OSError:
            return []

    def text(self, name, ext="err"):
        with open(os.path.join(self.dir, f"{name}.{ext}")) as f:
            return f.read()

    def wait(self, *names, timeout=60):
        return [self.procs[n].wait(timeout=timeout) for n in (names or list(self.procs))]

    def events(self, event=None, name=None):
        try:
            with open(os.path.join(self.state, "events.log")) as f:
                ev = [json.loads(line) for line in f]
        except OSError:
            return []
        return [e for e in ev if event in (None, e["event"]) and name in (None, e["name"])]

    def started(self, name):
        """time of the (last) start of a job, None if it has not started"""
        ev = self.events("start", name)
        return ev[-1]["t"] if ev else None

    def ended(self, name):
        ev = self.events("exit", name)
        return ev[-1]["t"] if ev else None

    def jobs(self):
        out = {}
        for n in os.listdir(self.state):
            if n.startswith("job-") and n.endswith(".json"):
                try:
                    with open(os.path.join(self.state, n)) as f:
                        j = json.load(f)
                    out[j["name"]] = j
                except (OSError, ValueError):
                    pass
        return out

    def status(self):
        """gq status parsed: ({card: row}, {running name: row}, {waiting name: row}, raw text, exit code)"""
        r = self.gq("status")
        return (*parse_status(r.stdout), r.stdout, r.returncode)

    def close(self):
        for p in list(self.procs.values()) + self.extra:
            if p.poll() is None:
                p.kill()
                p.wait()
        shutil.rmtree(self.dir, ignore_errors=True)


class Case(unittest.TestCase):
    def box(self, *a, **k):
        b = Box(*a, **k)
        self.addCleanup(b.close)
        return b

    def assertClean(self, b):
        """no ticket, no reservation, no stderr copy is left behind"""
        left = [n for n in os.listdir(b.state) if n not in ("lock", "events.log")]
        self.assertEqual(left, [])


class Basics(Case):
    def test_output_exit_code_and_the_three_lines(self):
        b = self.box()
        b.run("basic", QWEN, "--mib", 9000, "--secs", 0.6, "--code", 7, "--say", "hello")
        self.assertEqual(b.wait("basic"), [7])                               # the exit code of the command
        self.assertEqual(b.text("basic", "out"), "hello stdout card=0 mem=10500\n")       # CUDA_VISIBLE_DEVICES, GQ_MEM
        lines = b.text("basic").splitlines()
        self.assertEqual(len(lines), 4, lines)
        self.assertRegex(lines[0], r"^\[gq \d\d:\d\d:\d\d\] queued basic: needs 10500 MiB, card any, priority 0$")
        self.assertRegex(lines[1], r"^\[gq [\d:]+\] started basic on card 0 after \d+ s in the queue: .*fake_job.py --mib 9000")
        self.assertEqual(lines[2], "hello stderr")
        self.assertRegex(lines[3], r"^\[gq [\d:]+\] basic exit code 7 after \d+ s on card 0$")
        self.assertEqual([e["event"] for e in b.events()], ["queued", "start", "exit"])
        self.assertClean(b)

    def test_usage_errors_do_not_queue(self):
        b = self.box()
        for args, code, text in ((["run", "--mem", TOTAL - DRV - MARGIN + 1, "--", "true"], 2,
                                  "31411+700 MiB can never start on --gpu any (card 0: 32110 MiB when empty, card 1"),
                                 (["run", "--mem", 1000, "--gpu", 5, "--", "true"], 2, "can never start on --gpu 5"),
                                 (["run", "--mem", 1000, "--", "no_such_command_xyz"], 127, "command not found"),
                                 (["run", "--mem", 1000], 2, "needed:"), (["run", "--", "true"], 2, "--mem"),
                                 (["run", "--mem", 17, "--", "true"], 2, "the unit is MiB"),
                                 (["run", "--mem", 1000, "--gpu", "x", "--", "true"], 2, "needed:")):
            r = b.gq(*args)
            self.assertEqual(r.returncode, code, r.stderr)
            self.assertIn(text, r.stderr)
        self.assertClean(b)

    def test_death_by_signal_gives_128_plus_signal(self):
        b = self.box()
        b.run("selfkill", QWEN, cmd=["bash", "-c", "kill -KILL $$"])
        self.assertEqual(b.wait("selfkill"), [137])
        self.assertIn("selfkill exit code 137", b.text("selfkill"))
        self.assertClean(b)

    def test_memory_of_grandchildren_belongs_to_the_job(self):
        b = self.box()
        inner = " ".join(FAKE_JOB) + " --mib 9000 --secs 3"
        b.run("tree", QWEN, cmd=["bash", "-c", f"{inner} & {inner} & wait"])
        self.assertTrue(until(lambda: len(b.pids()) == 2))
        cards, running, _, text, _ = b.status()
        self.assertEqual(running["tree"]["ACTUAL"], "18000!", text)          # both processes; above the reservation
        self.assertEqual((cards["0"]["RESERVED"], cards["0"]["FOREIGN"], cards["0"]["USED"]), ("10500", "0", "18000"))
        self.assertEqual(b.wait("tree"), [0])

    def test_no_retry_leaves_stderr_alone(self):
        b = self.box()
        b.run("plain", QWEN, "--secs", 0.3, "--oom", flags=["--no-retry"])
        self.assertEqual(b.wait("plain"), [1])
        self.assertIn("CUDA out of memory", b.text("plain"))
        self.assertEqual([e["event"] for e in b.events()], ["queued", "start", "exit"])
        self.assertClean(b)


class Admission(Case):
    def test_burst_never_overcommits_a_card(self):
        """20 launches in the same instant on two cards (the collision of the old launchers)"""
        b = self.box(apps=[(0, 1, 5000)])                                    # a foreign 5000 MiB process on card 0
        rng = random.Random(7)
        needs = [rng.choice([QWEN, QWEN, LLAVA, 6000, 3000]) for _ in range(20)]
        for i, need in enumerate(needs):
            b.run(f"j{i}", need, "--mib", need - 400, "--secs", round(rng.uniform(0.6, 1.4), 2))
        peak = {0: 0, 1: 0}
        while any(p.poll() is None for p in b.procs.values()):               # what the fake cards really hold
            used = {0: 5000, 1: 0}
            for n in os.listdir(b.path + ".d") if os.path.isdir(b.path + ".d") else []:
                try:
                    with open(os.path.join(b.path + ".d", n)) as f:
                        p = json.load(f)
                    used[p["gpu"]] += p["mib"]
                except (OSError, ValueError):
                    pass
            peak = {c: max(peak[c], used[c]) for c in peak}
            time.sleep(0.02)
        self.assertEqual(b.wait(), [0] * 20)
        held, top = {0: {}, 1: {}}, {0: 0, 1: 0}                             # replay the event log
        for e in b.events():
            if e["event"] == "start":
                c = e["gpu"]
                self.assertGreaterEqual(e["reserved_before"], sum(held[c].values()))  # it saw every live reservation
                held[c][e["id"]] = e["need"]
                top[c] = max(top[c], sum(held[c].values()))
                self.assertLessEqual(e["reserved_before"] + e["need"] + e["foreign"] + MARGIN, TOTAL, e)
                self.assertEqual(e["foreign"], 5000 if c == 0 else 0)
            elif e["event"] == "exit":
                held[e["card"]].pop(e["id"])
        self.assertEqual(len(b.events("start")), 20)
        self.assertLessEqual(top[0], TOTAL - 5000 - MARGIN)
        self.assertLessEqual(top[1], TOTAL - MARGIN)
        self.assertGreater(top[0] + top[1], 45000)                           # ... and the cards were really filled
        self.assertLessEqual(max(peak.values()), TOTAL - DRV)
        self.assertClean(b)

    def test_fifo(self):
        b = self.box(cards=[TOTAL])
        for i in range(5):                                                   # one LLaVA-size job at a time fits
            b.run(f"f{i}", LLAVA, "--secs", 0.4)
            time.sleep(0.12)
        self.assertEqual(b.wait(), [0] * 5)
        self.assertEqual([e["name"] for e in b.events("start")], [f"f{i}" for i in range(5)])
        for i in range(4):
            self.assertLessEqual(b.ended(f"f{i}"), b.started(f"f{i + 1}"))

    def test_priority(self):
        b = self.box(cards=[TOTAL])
        b.run("first", LLAVA, "--secs", 1.5)
        until(lambda: b.started("first"))
        for name, prio in (("p0", None), ("p5", 5), ("p-1", -1), ("p5b", 5), ("p1", 1)):
            b.run(name, LLAVA, "--secs", 0.3, prio=prio)
            time.sleep(0.1)
        self.assertEqual(b.wait(), [0] * 6)
        self.assertEqual([e["name"] for e in b.events("start")], ["first", "p5", "p5b", "p1", "p0", "p-1"])

    def test_backfill(self):
        b = self.box(cards=[TOTAL])
        b.run("blocker", 20000, "--secs", 2.0)
        until(lambda: b.started("blocker"))
        b.run("big", LLAVA, "--secs", 0.3)                                   # 12607 MiB left: does not fit
        time.sleep(0.2)
        b.run("small", QWEN, "--secs", 0.3)                                  # later ticket, fits now
        self.assertEqual(b.wait(), [0] * 3)
        self.assertEqual([e["name"] for e in b.events("start")], ["blocker", "small", "big"])
        self.assertLess(b.started("small"), b.ended("blocker"))
        self.assertGreaterEqual(b.started("big"), b.ended("blocker"))

    def test_best_fit_and_gpu_option(self):
        b = self.box(apps=[(1, 1, 15000)])                                   # card 1 holds a foreign 15000 MiB job
        b.run("q_any", QWEN, "--secs", 3.0)                                  # -> card 1 (less free, still fits)
        until(lambda: b.started("q_any"))
        b.run("l_any", LLAVA, "--secs", 1.5)                                 # -> card 0 (does not fit on card 1)
        b.run("q_on0", 3000, "--secs", 1.5, gpu=0)                           # -> card 0 as asked (best fit: card 1)
        b.run("q_on1", QWEN, "--secs", 0.3, gpu=1)                           # waits for card 1, never moves to card 0
        self.assertTrue(until(lambda: b.started("l_any") and b.started("q_on0")))
        time.sleep(0.6)
        _, _, waiting, text, _ = b.status()
        self.assertIn("q_on1", waiting, text)
        self.assertEqual(waiting["q_on1"]["WHY"], "card 1: room 7107 MiB < 10500+700", text)
        self.assertEqual(b.wait(), [0] * 4)
        self.assertEqual({e["name"]: e["gpu"] for e in b.events("start")}, {"q_any": 1, "l_any": 0, "q_on0": 0, "q_on1": 1})
        self.assertGreaterEqual(b.started("q_on1"), b.ended("q_any"))

    def stream(self, starve):
        """Qwen jobs keep arriving: three fill the card and end one second apart, then one more arrives every
        second (3 s each), so that the card never holds fewer than two of them.  A LLaVA job queued at the start
        needs all but one of them gone.  Returns (box, seconds until the LLaVA job started, or None if it had not
        started when the last Qwen job arrived)."""
        b = self.box(cards=[TOTAL], STARVE=starve)
        for i in range(3):
            b.run(f"q{i}", QWEN, "--secs", 2 + i)
        until(lambda: all(b.started(f"q{i}") for i in range(3)))
        t0 = time.time()
        b.run("llava", LLAVA, "--secs", 0.5)
        for i in range(3, 14):
            time.sleep(1.0)
            b.run(f"q{i}", QWEN, "--secs", 3)
        start = b.started("llava")
        return b, (None if start is None else start - t0)

    def test_starvation_rule(self):
        b, waited = self.stream(starve=4)
        self.assertIsNotNone(waited, "the LLaVA job starved although the age limit (4 s) had passed")
        self.assertGreater(waited, 4)
        self.assertLess(waited, 4 + 3 + 2.5)             # the limit, then the running jobs end (3 s), then a look
        self.assertLess(b.started("q3"), b.started("llava"))               # before the limit: later jobs backfilled
        self.assertEqual(b.wait(timeout=90), [0] * 15)
        late = [n for n in b.procs if n != "llava" and b.started(n) > b.started("llava")]
        self.assertGreaterEqual(len(late), 6, late)      # the rest of the stream ran after (or next to) it
        self.assertClean(b)

    def test_without_the_rule_the_big_job_starves(self):
        b, waited = self.stream(starve=1000)
        self.assertIsNone(waited, "control: backfilling alone should keep the LLaVA job waiting")
        self.assertEqual(b.wait(timeout=90), [0] * 15)                       # it runs when the stream has dried up

    def test_starving_job_holds_the_card_against_later_tickets(self):
        b = self.box(cards=[TOTAL], STARVE=2)
        b.run("earlier", 15000, "--secs", 2.0)                               # ticket before llava's
        until(lambda: b.started("earlier"))
        b.run("llava", LLAVA, "--secs", 0.5)                                 # 17607 MiB left: 93 MiB short
        time.sleep(0.2)
        small = [f"s{i}" for i in range(3)]
        for n in small:                                                      # later tickets: backfilled next to it
            b.run(n, 5000, "--secs", 7)
        self.assertTrue(until(lambda: b.ended("earlier")))
        self.assertTrue(all(b.started(n) for n in small))
        time.sleep(1.0)                                                      # llava is older than the limit now, and
        b.run("late", 5000, "--secs", 0.5)                                   # only later tickets are in its way: late
        b.run("late_hi", 300, "--secs", 0.5, prio=1)                         # fits but must not start; higher priority may
        self.assertTrue(until(lambda: "kept for" in b.status()[2].get("late", {}).get("WHY", ""), 5), b.status()[3])
        _, _, waiting, text, _ = b.status()
        self.assertEqual(waiting["late"]["WHY"], "card 0 is kept for the long-waiting job llava", text)
        self.assertRegex(waiting["llava"]["WHY"], r"^card 0: room 17[36]07 MiB < 17000\+700", text)   # 93 MiB short
        self.assertEqual(b.wait(), [0] * 7)
        first_end = min(b.ended(n) for n in small)
        self.assertLess(b.started("late_hi"), first_end)
        self.assertGreater(b.started("late"), first_end)                     # 3 s after it was queued, although it fitted
        self.assertLess(b.started("llava"), first_end + 1.5)                 # the old job got the card as soon as it fitted

    def test_young_foreign_process_makes_the_card_busy(self):
        b = self.box(cards=[TOTAL], YOUNG=3)
        young = subprocess.Popen(["sleep", "30"])
        b.extra.append(young)
        b.put(apps=[[0, young.pid, 4000]])
        t0 = time.time()
        b.run("q", QWEN, "--secs", 0.3)
        self.assertTrue(until(lambda: "busy" in b.status()[2].get("q", {}).get("WHY", ""), 3), b.status()[3])
        cards, _, waiting, text, _ = b.status()
        self.assertRegex(waiting["q"]["WHY"], rf"^card 0 busy: foreign pid {young.pid} started \d s ago$")
        self.assertIn(f"busy: foreign pid {young.pid} started", cards["0"]["NOTE"], text)
        self.assertEqual(b.wait("q"), [0])
        self.assertGreaterEqual(b.started("q") - t0, 2.5)                    # only when the process was 3 s old
        self.assertLess(b.started("q") - t0, 5.0)

    def test_queue_job_filling_its_reservation_does_not_make_the_card_busy(self):
        b = self.box(cards=[TOTAL], POLL=4, LOOK=1)
        b.run("a", QWEN, "--mib", QWEN - 50, "--secs", 3)                    # allocates at once when it starts (t = 1 s)
        time.sleep(0.5)
        b.run("b", 3000, "--secs", 0.3)                                      # its two looks: before and after that
        self.assertEqual(b.wait(), [0, 0])
        self.assertLess(b.started("b") - b.started("a"), 2.0)                # at its first decision, not 4 s later

    def test_growing_memory_makes_the_card_busy(self):
        b = self.box(cards=[TOTAL])
        t0, why = time.time(), set()
        b.put(hidden={"0": 4000}, ramp={"0": [2000, t0, t0 + 2.5]})          # 2000 MiB/s more, of unknown origin
        b.run("q", QWEN, "--secs", 0.3)
        while time.time() - t0 < 2.4:
            why.add(b.jobs().get("q", {}).get("reason", ""))
            time.sleep(0.05)
        self.assertIsNone(b.started("q"), why)
        self.assertTrue(any(re.match(r"card 0 busy: memory grew by \d+ MiB in 0 s", w) for w in why), why)
        self.assertEqual(b.wait("q"), [0])                                   # it stopped growing: the job starts
        self.assertLess(b.started("q") - t0, 4.5)

    def test_nvidia_smi_failure_is_waited_out(self):
        b = self.box(cards=[TOTAL])
        b.put(fail=True)
        b.run("q", QWEN, "--secs", 0.3)
        self.assertTrue(until(lambda: b.jobs().get("q", {}).get("reason") == "nvidia-smi failed, looking again", 5))
        _, _, waiting, text, code = b.status()
        self.assertEqual(code, 1)                                            # gq status says so, too
        self.assertIn("nvidia-smi failed", text)
        self.assertIn("q", waiting)
        time.sleep(1.0)
        self.assertIsNone(b.started("q"))
        b.put(fail=False)
        self.assertEqual(b.wait("q"), [0])
        self.assertClean(b)

    def test_stopped_waiter_is_passed_over(self):
        b = self.box(cards=[TOTAL], BEAT=1.5)
        b.run("first", LLAVA, "--secs", 3.5)
        until(lambda: b.started("first"))
        stuck = b.run("stuck", LLAVA, "--secs", 0.3)
        time.sleep(0.8)
        b.run("next", LLAVA, "--secs", 0.3)
        os.kill(stuck.pid, signal.SIGSTOP)                                   # my own test process
        try:
            self.assertEqual(b.wait("first", "next"), [0, 0])                # next did not wait for the stopped one
            self.assertIsNone(b.started("stuck"))
            self.assertIn("its last look was", b.status()[2]["stuck"]["WHY"])
        finally:
            os.kill(stuck.pid, signal.SIGCONT)
        self.assertEqual(b.wait("stuck"), [0])
        self.assertClean(b)

    def test_budget_of_the_gpu_test(self):
        """GQ_CAP limits what the jobs submitted with it reserve on a card together; other jobs are not limited"""
        b = self.box()
        cap = {"GQ_CAP": "0:6000,1:0"}
        for n in ("t1", "t2", "t3"):
            b.run(n, 2500, "--secs", 1.5, env=cap)
            time.sleep(0.1)
        b.run("real", LLAVA, "--secs", 0.3)
        self.assertTrue(until(lambda: b.started("t1") and b.started("t2") and b.started("real")))
        time.sleep(0.6)
        _, running, waiting, text, _ = b.status()
        self.assertEqual(waiting["t3"]["WHY"], "card 0: room 1700 MiB < 2500+700; card 1: room 700 MiB < 2500+700", text)
        self.assertEqual(b.wait(), [0] * 4)
        self.assertEqual({e["name"]: e["gpu"] for e in b.events("start")}, {"t1": 0, "t2": 0, "t3": 0, "real": 0})
        self.assertGreaterEqual(b.started("t3"), min(b.ended("t1"), b.ended("t2")))

    def test_largest_job_that_can_ever_start(self):
        b = self.box(cards=[TOTAL])
        b.run("huge", TOTAL - DRV - MARGIN, "--secs", 0.3)                   # 31410: all of an empty card but 700 MiB
        self.assertEqual(b.wait("huge"), [0])

    def test_held_lock_is_reported_and_waited_out(self):
        import fcntl
        b = self.box(cards=[TOTAL])
        with open(os.path.join(b.state, "lock"), "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)                                    # as a gq process stopped in mid-decision
            b.run("q", QWEN, "--secs", 0.3)
            t0 = time.time()
            _, _, waiting, text, code = b.status()
            self.assertGreaterEqual(time.time() - t0, 4.5)
            self.assertLess(time.time() - t0, 8)
            self.assertIn("WARNING: state/lock has been held by another process for 5 s", text)
            self.assertIn("q", waiting, text)                                # the table is still shown
            self.assertIsNone(b.started("q"))
        self.assertEqual(b.wait("q"), [0])                                   # released: the job starts
        self.assertClean(b)

    def test_emptied_state_directory_heals(self):
        b = self.box(cards=[TOTAL], BEAT=1.0)
        b.run("run", 20000, "--mib", 9000, "--secs", 5)
        self.assertTrue(until(b.pids))
        b.run("wait", 20000, "--secs", 0.3)
        self.assertTrue(until(lambda: "wait" in b.jobs()))
        for n in os.listdir(b.state):                                        # somebody "cleans up"
            os.remove(os.path.join(b.state, n))
        self.assertTrue(until(lambda: sorted(b.jobs()) == ["run", "wait"], 5))       # ticket and reservation are back
        cards, running, waiting, text, _ = b.status()
        self.assertEqual((cards["0"]["RESERVED"], running["run"]["ACTUAL"]), ("20000", "9000"), text)
        self.assertEqual(b.wait(), [0, 0])
        self.assertGreaterEqual(b.started("wait"), b.ended("run"))
        self.assertClean(b)

    def test_job_above_its_reservation_is_counted_by_what_it_takes(self):
        b = self.box(cards=[TOTAL])
        b.run("greedy", QWEN, "--mib", 20000, "--secs", 2.0)                 # reserved 10500, takes 20000
        self.assertTrue(until(b.pids))
        b.run("q", 12000, "--secs", 0.3)                                     # 32607 - 20497 = 12110 < 12700
        time.sleep(1.0)
        cards, running, waiting, text, _ = b.status()
        self.assertEqual(running["greedy"]["ACTUAL"], "20000!", text)
        self.assertEqual(cards["0"]["FITS"], "11410", text)
        self.assertEqual(waiting["q"]["WHY"], "card 0: room 12110 MiB < 12000+700", text)
        self.assertEqual(b.wait(), [0, 0])
        self.assertGreaterEqual(b.started("q"), b.ended("greedy"))


class Kills(Case):
    """a killed job frees its reservation"""

    def running(self, b, name, *job, mem=20000, ignore=()):
        p = b.run(name, mem, "--mib", 9000, "--secs", 60, *job, ignore=ignore)
        self.assertTrue(until(b.pids))
        child = b.pids()[0]
        self.assertEqual(b.status()[1][name]["ACTUAL"], "9000")
        return p, child

    def test_sigterm_of_the_wrapper(self):
        b = self.box(cards=[TOTAL])
        p, child = self.running(b, "victim")
        b.run("next", 20000, "--secs", 0.3)                                  # needs the victim's room
        time.sleep(0.8)
        self.assertIsNone(b.started("next"))
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=10), 143)
        self.assertFalse(os.path.exists(f"/proc/{child}"))                   # the command got the signal and is gone
        self.assertNotIn("victim", b.jobs())                                 # reservation removed by the wrapper itself
        self.assertIn("victim exit code 143", b.text("victim"))
        self.assertEqual(b.wait("next"), [0])
        self.assertClean(b)

    def test_sigint_and_sighup_are_passed_on_too(self):
        for sig in (signal.SIGINT, signal.SIGHUP):
            b = self.box(cards=[TOTAL])
            p, child = self.running(b, "victim")
            p.send_signal(sig)
            self.assertEqual(p.wait(timeout=10), 128 + sig)
            self.assertFalse(os.path.exists(f"/proc/{child}"))
            self.assertClean(b)

    def test_sigkill_of_the_wrapper(self):
        b = self.box(cards=[TOTAL])
        p, child = self.running(b, "victim")
        b.run("next", 20000, "--secs", 0.3)
        time.sleep(0.8)
        self.assertIsNone(b.started("next"))
        p.kill()
        self.assertEqual(p.wait(timeout=10), -9)
        cards, running, waiting, text, _ = b.status()                        # the reservation no longer counts ...
        self.assertNotIn("victim", running, text)
        self.assertEqual(cards["0"]["RESERVED"], "0" if "next" in waiting else "20000", text)
        self.assertNotIn("victim", b.jobs())                                 # ... and its file has been removed
        self.assertTrue(until(lambda: not os.path.exists(f"/proc/{child}"), 5))      # the command got SIGTERM
        self.assertEqual(b.wait("next"), [0])
        self.assertEqual([e["name"] for e in b.events("stale")], ["victim"])
        self.assertClean(b)

    def test_under_nohup_sighup_is_ignored(self):
        b = self.box(cards=[TOTAL])
        p = b.run("nohup", QWEN, "--mib", 9000, "--secs", 2.0, ignore=[signal.SIGHUP])
        self.assertTrue(until(b.pids))
        p.send_signal(signal.SIGHUP)
        time.sleep(0.5)
        self.assertEqual(b.status()[1]["nohup"]["ACTUAL"], "9000")           # still running
        self.assertEqual(p.wait(timeout=10), 0)
        self.assertClean(b)

    def test_sigterm_of_a_waiting_job(self):
        b = self.box(cards=[TOTAL])
        self.running(b, "holder")
        p = b.run("waiter", 20000, "--secs", 0.3)
        self.assertTrue(until(lambda: "room" in b.jobs().get("waiter", {}).get("reason", "")))
        p.send_signal(signal.SIGTERM)
        self.assertEqual(p.wait(timeout=10), 143)
        self.assertIn("waiter cancelled by signal 15", b.text("waiter"))
        self.assertEqual(sorted(b.jobs()), ["holder"])
        self.assertIsNone(b.started("waiter"))

    def test_reservation_lasts_as_long_as_the_command(self):
        """a command that ignores SIGTERM keeps its card: the reservation is not given away under it"""
        b = self.box(cards=[TOTAL])
        p, child = self.running(b, "stubborn", "--ignore-term")
        b.run("next", 20000, "--secs", 0.3)
        p.send_signal(signal.SIGTERM)
        time.sleep(1.5)
        self.assertIsNone(p.poll())                                          # gq still waits for its command
        self.assertEqual(b.status()[0]["0"]["RESERVED"], "20000")
        self.assertIsNone(b.started("next"))
        p.kill()                                                             # now the wrapper dies; the command lives on
        p.wait(timeout=10)
        time.sleep(1.0)
        self.assertTrue(os.path.exists(f"/proc/{child}"))
        cards, running, waiting, text, _ = b.status()
        self.assertEqual((cards["0"]["RESERVED"], cards["0"]["FOREIGN"]), ("0", "9000"), text)    # foreign memory now
        self.assertEqual(running, {})
        self.assertIn("next", waiting, text)                                 # 32607 - 9497 = 23110 >= 20700, but ...
        os.kill(child, signal.SIGKILL)                                       # my own test process
        self.assertEqual(b.wait("next"), [0])


class OutOfMemory(Case):
    def test_requeued_once_after_the_wait(self):
        b = self.box(cards=[TOTAL])
        b.run("oom", QWEN, "--mib", 9000, "--secs", 0.4, "--code", 5, "--oom-once", os.path.join(b.dir, "marker"))
        self.assertEqual(b.wait("oom"), [5])                                 # the second run's own exit code
        ev = b.events()
        self.assertEqual([e["event"] for e in ev], ["queued", "start", "requeued", "start", "exit"])
        self.assertGreaterEqual(ev[3]["t"] - ev[2]["t"], 1.5)                # GQ_OOM_WAIT (60 s by default)
        text = b.text("oom")
        self.assertIn("CUDA out of memory", text)                            # the error was passed on as well
        self.assertRegex(text, r"oom ran out of CUDA memory after \d+ s on card 0 \(exit code 1\): queued once more, next try in 2 s")
        self.assertEqual(len(re.findall(r"started oom on card 0", text)), 2)
        self.assertClean(b)

    def test_while_it_waits_it_holds_nothing(self):
        b = self.box(cards=[TOTAL], OOM_WAIT=3)
        b.run("oom", LLAVA, "--secs", 0.3, "--oom-once", os.path.join(b.dir, "marker"))
        self.assertTrue(until(lambda: b.events("requeued")))
        b.run("other", LLAVA, "--secs", 0.3)                                 # later ticket, takes the card meanwhile
        _, running, waiting, text, _ = b.status()
        self.assertRegex(waiting["oom"]["WHY"], r"^queued again after CUDA out of memory", text)
        self.assertEqual(b.wait(), [0, 0])
        self.assertLess(b.started("other"), b.started("oom"))

    def test_second_failure_is_returned(self):
        b = self.box(cards=[TOTAL])
        b.run("oom", QWEN, "--secs", 0.3, "--oom")
        self.assertEqual(b.wait("oom"), [1])
        self.assertEqual([e["event"] for e in b.events()], ["queued", "start", "requeued", "start", "exit"])
        self.assertEqual(b.text("oom").count("CUDA out of memory. Tried"), 2)
        self.assertClean(b)

    def test_other_failures_and_late_failures_are_not_retried(self):
        b = self.box(cards=[TOTAL], OOM_WINDOW=0.5)
        b.run("fail", QWEN, "--secs", 0.3, "--code", 3)                      # fails, but not out of memory
        b.run("late", QWEN, "--secs", 0.3, cmd=["bash", "-c", "sleep 1; echo CUDA out of memory >&2; exit 1"])
        b.run("fine", QWEN, cmd=["bash", "-c", "echo CUDA out of memory >&2; exit 0"])       # the text alone: no
        self.assertEqual(b.wait(), [3, 1, 0])
        self.assertEqual(b.events("requeued"), [])
        self.assertClean(b)


class StatusAndIdle(Case):
    def test_status_table(self):
        b = self.box(apps=[(1, 1, 15000)])
        b.run("run_a", LLAVA, "--mib", 16000, "--secs", 4, prio=2)           # -> card 0 (card 1 has 17110 < 17700)
        self.assertTrue(until(b.pids))
        b.run("wait_b", LLAVA, "--secs", 0.3)
        b.run("wait_c", 20000, "--secs", 0.3, gpu=1, prio=-3)
        self.assertTrue(until(lambda: all("room" in b.jobs().get(n, {}).get("reason", "") for n in ("wait_b", "wait_c"))))
        time.sleep(1.2)
        cards, running, waiting, text, code = b.status()
        self.assertEqual(code, 0)
        self.assertRegex(text.splitlines()[0], r"^gq status \d{4}-\d\d-\d\d \d\d:\d\d:\d\d: 1 running, 2 waiting ")
        self.assertEqual([l.split()[0] for l in text.splitlines()[1:]],
                         ["CARD", "0", "1", "RUNNING", "run_a", "WAITING", "wait_b", "wait_c"])
        c0 = {k: cards["0"][k] for k in ("TOTAL", "USED", "FREE", "RESERVED", "FOREIGN", "FITS")}
        self.assertEqual(c0, dict(TOTAL="32607", USED="16000", FREE="16110", RESERVED="17000", FOREIGN="0", FITS="14907"), text)
        c1 = {k: cards["1"][k] for k in ("TOTAL", "USED", "FREE", "RESERVED", "FOREIGN", "FITS")}
        self.assertEqual(c1, dict(TOTAL="32607", USED="15000", FREE="17110", RESERVED="0", FOREIGN="15000", FITS="16410"), text)
        self.assertEqual(cards["1"]["NOTE"], "foreign pid:MiB 1:15000")
        r = running["run_a"]
        self.assertEqual((r["PID"], r["CARD"], r["RESERVED"], r["ACTUAL"], r["PRIO"]),
                         (str(b.procs["run_a"].pid), "0", "17000", "16000", "2"), text)
        self.assertRegex(r["RUNTIME"], r"^0:00:0[1-9]$")
        self.assertEqual(r["COMMAND"], "python fake_job.py --mib 16000 --secs 4")
        w = waiting["wait_b"]
        self.assertEqual((w["PID"], w["NEED"], w["CARD"], w["PRIO"]), (str(b.procs["wait_b"].pid), "17000", "any", "0"))
        self.assertRegex(w["WAITED"], r"^0:00:0[1-9]$")
        self.assertEqual(w["WHY"], "card 0: room 15607 MiB < 17000+700; card 1: room 17110 MiB < 17000+700", text)
        w = waiting["wait_c"]
        self.assertEqual((w["NEED"], w["CARD"], w["PRIO"], w["WHY"]), ("20000", "1", "-3", "card 1: room 17110 MiB < 20000+700"))
        b.procs["wait_c"].terminate()
        self.assertEqual(b.wait("run_a", "wait_b"), [0, 0])
        _, running, waiting, text, _ = b.status()
        self.assertEqual((running, waiting), ({}, {}), text)

    def test_wait_idle(self):
        b = self.box(cards=[TOTAL])
        t0 = time.time()
        self.assertEqual(b.gq("wait-idle").returncode, 0)                    # empty queue: after its second look
        self.assertGreaterEqual(time.time() - t0, 0.95)
        self.assertLess(time.time() - t0, 2.5)
        self.assertEqual(b.gq("wait-idle", "--timeout", 0.2).returncode, 0)  # empty when the time is up: idle
        b.run("a", LLAVA, "--secs", 2.0)
        time.sleep(0.15)                                                     # a's ticket is the earlier one
        b.run("b", LLAVA, "--secs", 1.0)                                     # waits behind a
        until(lambda: b.started("a"))
        t0 = time.time()
        r = b.gq("wait-idle", "--timeout", 0.6)
        self.assertEqual(r.returncode, 1)
        self.assertIn("still 2 job(s) after 1 s", r.stderr)
        self.assertLess(time.time() - t0, 1.5)
        self.assertEqual(b.gq("wait-idle").returncode, 0)                    # returns when both are done
        self.assertIsNotNone(b.ended("a"))
        self.assertIsNotNone(b.ended("b"))
        self.assertLess(time.time() - b.ended("b"), 3.5)
        self.assertClean(b)

    def test_wait_idle_sees_a_job_that_was_started_this_instant(self):
        """the script pattern `gq run ... &` + `gq wait-idle`, and a chain `gq run a && gq run b`"""
        b = self.box(cards=[TOTAL])
        b.run("bg", QWEN, "--secs", 1.0)                                     # no pause: its ticket is not written yet
        self.assertEqual(b.gq("wait-idle").returncode, 0)
        self.assertIsNotNone(b.ended("bg"), "wait-idle returned before the job that had just been started")
        chain = f"{GQ} run --mem 3000 --name c1 -- sleep 0.5 && {GQ} run --mem 3000 --name c2 -- sleep 0.5"
        p = subprocess.Popen(["bash", "-c", chain], env=b.env, stderr=subprocess.DEVNULL)
        b.extra.append(p)
        self.assertTrue(until(lambda: b.started("c1")))
        self.assertEqual(b.gq("wait-idle").returncode, 0)                    # not in the gap between c1 and c2
        self.assertIsNotNone(b.ended("c2"))
        self.assertEqual(p.wait(10), 0)
        self.assertClean(b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
