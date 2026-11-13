#!/usr/bin/env python3
"""Unit tests of gq's admission logic.  No GPU, no nvidia-smi, no processes: readings and the process table are
injected (gq.smi, gq.pstat, gq.age_of are replaced).  Run: python tests/test_admission.py [-v]
GQ_DIR chooses the directory of the gq.py under test (default: the parent of tests/).
"""
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True       # no __pycache__ next to gq.py
HERE = os.path.dirname(os.path.abspath(__file__))
GQ_DIR = os.environ.get("GQ_DIR", os.path.dirname(HERE))
for k in [k for k in os.environ if k.startswith("GQ_") and k != "GQ_DIR"]:
    del os.environ[k]                                    # the defaults are what is tested
os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)
TMP = tempfile.mkdtemp(prefix="unit_", dir=os.path.join(HERE, "tmp"))
os.environ["GQ_STATE"] = TMP
spec = importlib.util.spec_from_file_location("gq", os.path.join(GQ_DIR, "gq.py"))
gq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gq)
REAL = (gq.smi, gq.pstat, gq.age_of)

TOTAL, DRV = 32607, 497          # the cards of this machine; MiB the driver keeps (total - used - free)
QWEN, LLAVA = 10500, 17000
NOW = 1_000_000.0


def J(name, need, t, gpu=None, prio=0, want=None, beat=None, not_before=0, pid=None):
    """A job as gq stores it; t = ticket time relative to NOW - 100 s; running if gpu is given."""
    return dict(id=f"id-{name}", name=name, pid=pid or (hash(name) % 30000 + 40000), pstart=1, need=need, want=want,
                prio=prio, t=NOW - 100 + t, beat=NOW if beat is None else beat, not_before=not_before, gpu=gpu,
                t_start=None if gpu is None else NOW - 50, reason="", cmd="x")


def C(used_by_procs=0, foreign=0, own=None, busy="", total=TOTAL):
    """A card as gq.look() returns it.  used_by_procs: memory of all processes on the card (nvidia-smi 'used')."""
    return dict(total=total, used=used_by_procs + DRV, free=total - used_by_procs - DRV, smi_used=used_by_procs,
                foreign=foreign, own=dict(own or {}), busy=busy, young="", pids=[])


def plan(jobs, cards, now=NOW):
    return {k[3:]: v for k, v in gq.schedule(jobs, cards, now).items()}       # {job name: (card or None, reason)}


class FitRule(unittest.TestCase):
    """fits iff total - max(used, reservations + foreign) >= need + 700"""

    def test_empty_card_boundary(self):
        self.assertEqual(plan([J("a", TOTAL - DRV - 700, 0)], {0: C()})["a"][0], 0)
        got = plan([J("a", TOTAL - DRV - 700 + 1, 0)], {0: C()})["a"]
        self.assertIsNone(got[0])
        self.assertIn("card 0: room 32110 MiB < 31411+700", got[1])

    def test_reservation_counts_before_the_memory_is_taken(self):
        run = J("run", LLAVA, 0, gpu=0)                  # admitted, still loading: only 3000 MiB taken so far
        cards = {0: C(3000, own={"id-run": 3000})}
        self.assertEqual(plan([run, J("a", TOTAL - LLAVA - 700, 1)], cards)["a"][0], 0)
        self.assertIsNone(plan([run, J("a", TOTAL - LLAVA - 700 + 1, 1)], cards)["a"][0])

    def test_two_launches_in_the_same_second_cannot_share_the_same_memory(self):
        """the failure of the old launchers: both see the same free memory"""
        a, b = J("a", LLAVA, 0), J("b", LLAVA, 1)
        got = plan([a, b], {0: C()})
        self.assertEqual(got["a"][0], 0)
        self.assertIsNone(got["b"][0])
        a["gpu"] = 0                                     # a has written its reservation, nothing allocated yet
        self.assertIsNone(plan([a, b], {0: C()})["b"][0])

    def test_actual_use_above_the_reservation_counts(self):
        run = J("run", 10000, 0, gpu=0)                  # reserved 10000, takes 12000
        cards = {0: C(12000, own={"id-run": 12000})}
        room = TOTAL - 12000 - DRV
        self.assertEqual(plan([run, J("a", room - 700, 1)], cards)["a"][0], 0)
        self.assertIsNone(plan([run, J("a", room - 700 + 1, 1)], cards)["a"][0])

    def test_overshoot_of_one_job_does_not_eat_the_reservation_of_another(self):
        over, grow = J("over", QWEN, 0, gpu=0), J("grow", QWEN, 1, gpu=0)   # over takes 13000, grow 8000 so far
        cards = {0: C(21000, own={"id-over": 13000, "id-grow": 8000})}
        room = TOTAL - (13000 + QWEN)                    # not TOTAL - used = 11110: grow may still take 2500 more
        self.assertEqual(plan([over, grow, J("a", room - 700, 2)], cards)["a"][0], 0)
        got = plan([over, grow, J("a", room - 700 + 1, 2)], cards)["a"]
        self.assertIsNone(got[0])
        self.assertIn(f"card 0: room {room} MiB", got[1])

    def test_foreign_memory_is_added_to_the_reservations(self):
        run = J("run", 10000, 0, gpu=0)                  # nothing allocated yet, next to a foreign 8000 MiB process
        cards = {0: C(8000, foreign=8000)}
        self.assertEqual(plan([run, J("a", TOTAL - 18000 - 700, 1)], cards)["a"][0], 0)
        self.assertIsNone(plan([run, J("a", TOTAL - 18000 - 700 + 1, 1)], cards)["a"][0])

    def test_memory_that_no_process_accounts_for_counts(self):
        cards = {0: C(22000, foreign=12000)}             # 10000 MiB used that nvidia-smi shows for no process
        self.assertIsNone(plan([J("a", QWEN, 0)], cards)["a"][0])
        self.assertEqual(plan([J("a", QWEN, 0)], {0: C(12000, foreign=12000)})["a"][0], 0)

    def test_standard_jobs(self):
        """three Qwen jobs share a card; LLaVA + Qwen share a card; two LLaVA jobs do not"""
        q = [J(f"q{i}", QWEN, i, gpu=0) for i in range(3)]
        self.assertEqual(plan(q[:2] + [J("a", QWEN, 9)], {0: C()})["a"][0], 0)
        self.assertIsNone(plan(q + [J("a", QWEN, 9)], {0: C()})["a"][0])
        self.assertEqual(plan([J("l", LLAVA, 0, gpu=0), J("a", QWEN, 9)], {0: C()})["a"][0], 0)
        self.assertIsNone(plan([J("l", LLAVA, 0, gpu=0), J("a", LLAVA, 9)], {0: C()})["a"][0])
        self.assertIsNone(plan([J("l", LLAVA, 0, gpu=0), q[0], J("a", QWEN, 9)], {0: C()})["a"][0])


class Order(unittest.TestCase):
    def test_fifo(self):
        got = plan([J("late", LLAVA, 5), J("early", LLAVA, 1)], {0: C()})
        self.assertEqual(got["early"][0], 0)
        self.assertIsNone(got["late"][0])
        self.assertIn("after early", got["late"][1])     # the reason names the job that goes first

    def test_priority_beats_ticket_time(self):
        got = plan([J("early", LLAVA, 1), J("late_hi", LLAVA, 5, prio=3)], {0: C()})
        self.assertEqual(got["late_hi"][0], 0)
        self.assertIsNone(got["early"][0])
        got = plan([J("early", LLAVA, 1), J("late_lo", LLAVA, 5, prio=-1)], {0: C()})
        self.assertEqual(got["early"][0], 0)

    def test_backfill(self):
        """a later job starts if it fits now, while an earlier, bigger one waits"""
        cards = {0: C(16000, foreign=16000)}
        got = plan([J("big", LLAVA, 1), J("small", QWEN, 5)], cards)
        self.assertIsNone(got["big"][0])
        self.assertEqual(got["small"][0], 0)

    def test_backfill_never_takes_what_an_earlier_job_can_use_now(self):
        cards = {0: C(5000, foreign=5000)}               # big (17000) fits; then 10607 MiB are left: small does not
        got = plan([J("big", LLAVA, 1), J("small", QWEN, 5)], cards)
        self.assertEqual(got["big"][0], 0)
        self.assertIsNone(got["small"][0])
        self.assertIn("after big", got["small"][1])
        self.assertEqual(plan([J("small", QWEN, 5)], cards)["small"][0], 0)           # alone it would fit
        self.assertEqual(plan([J("big", LLAVA, 1), J("small", QWEN - 600, 5)], cards)["small"][0], 0)   # both fit

    def test_every_process_computes_the_same_plan(self):
        jobs = [J("a", LLAVA, 1), J("b", QWEN, 2), J("c", LLAVA, 3), J("d", QWEN, 4), J("e", QWEN, 5), J("f", 3000, 6)]
        cards = {0: C(), 1: C(9000, foreign=9000)}
        got = plan(jobs, cards)
        self.assertEqual(got, plan(list(reversed(jobs)), cards))
        # a -> card 1 (best fit), b -> card 0, c -> card 0, d: no room, e: no room, f -> card 0 (backfill)
        self.assertEqual([got[n][0] for n in "abcdef"], [1, 0, 0, None, None, 0])
        for c in cards:                                  # the plan itself respects the rule on every card
            need = sum(j["need"] for j in jobs if got[j["name"]][0] == c)
            self.assertLessEqual(need + cards[c]["foreign"] + 700, TOTAL)

    def test_budget_of_the_gpu_test(self):
        """GQ_CAP: the jobs that carry a budget for a card reserve at most that much of it together"""
        def T(name, need, t, **kw):
            return dict(J(name, need, t, **kw), cap={"0": 6000, "1": 0})
        cards = {0: C(), 1: C()}
        got = plan([T("a", 2500, 1), T("b", 2500, 2), T("c", 2500, 3), J("real", LLAVA, 4)], cards)
        self.assertEqual([got[n][0] for n in "abc"], [0, 0, None])         # 7500 > 6000; and never card 1
        self.assertEqual(got["c"][1], "card 0: room 1700 MiB < 2500+700 (after a, b, next in line); "
                                      "card 1: room 700 MiB < 2500+700")
        self.assertEqual(got["real"][0], 0)              # a job without a budget is not limited by it
        got = plan([T("a", 2500, 1, gpu=0), J("real", LLAVA, 2, gpu=0), T("b", 3500, 3), T("c", 3501, 4)], cards)
        self.assertEqual((got["b"][0], got["c"][0]), (0, None))            # 2500 + 3500 = 6000: the real job is not counted
        full = {0: C(28000, foreign=28000), 1: C()}      # the budget never overrides the rule itself
        self.assertIsNone(plan([T("a", 4000, 1)], full)["a"][0])
        self.assertEqual(gq.CAPS, {})                    # real jobs carry none

    def test_a_job_whose_process_stopped_looking_is_passed_over(self):
        got = plan([J("stuck", LLAVA, 1, beat=NOW - gq.BEAT - 1), J("next", LLAVA, 5)], {0: C()})
        self.assertIsNone(got["stuck"][0])
        self.assertIn("has not looked", got["stuck"][1])
        self.assertEqual(got["next"][0], 0)
        got = plan([J("slow", LLAVA, 1, beat=NOW - gq.BEAT + 5), J("next", LLAVA, 5)], {0: C()})
        self.assertEqual((got["slow"][0], got["next"][0]), (0, None))

    def test_a_job_queued_again_after_oom_waits_its_60_seconds(self):
        got = plan([J("again", LLAVA, 1, not_before=NOW + 30), J("next", LLAVA, 5)], {0: C()})
        self.assertIsNone(got["again"][0])
        self.assertIn("out of memory", got["again"][1])
        self.assertEqual(got["next"][0], 0)
        self.assertEqual(plan([J("again", LLAVA, 1, not_before=NOW - 1), J("next", LLAVA, 5)], {0: C()})["again"][0], 0)


class CardChoice(unittest.TestCase):
    def test_best_fit_takes_the_card_with_less_free_memory(self):
        cards = {0: C(), 1: C(15000, foreign=15000)}     # room 32110 and 17110
        self.assertEqual(plan([J("q", QWEN, 0)], cards)["q"][0], 1)
        self.assertEqual(plan([J("l", LLAVA, 0)], cards)["l"][0], 0)       # 17000 + 700 > 17110
        self.assertEqual(plan([J("q", QWEN, 0)], {0: C(15000, foreign=15000), 1: C()})["q"][0], 0)
        self.assertEqual(plan([J("q", QWEN, 0)], {0: C(), 1: C()})["q"][0], 0)        # a tie: the lower index

    def test_gpu_option(self):
        cards = {0: C(), 1: C(15000, foreign=15000)}
        self.assertEqual(plan([J("q", QWEN, 0, want=0)], cards)["q"][0], 0)
        got = plan([J("l", LLAVA, 0, want=1)], cards)["l"]                 # never moved to the other card
        self.assertIsNone(got[0])
        self.assertIn("card 1", got[1])
        self.assertNotIn("card 0", got[1])

    def test_busy_card_is_left_alone(self):
        cards = {0: C(), 1: C(15000, foreign=15000, busy="foreign pid 7 started 30 s ago")}
        self.assertEqual(plan([J("q", QWEN, 0)], cards)["q"][0], 0)        # best fit would be card 1
        got = plan([J("q", QWEN, 0, want=1)], cards)["q"]
        self.assertIsNone(got[0])
        self.assertIn("card 1 busy: foreign pid 7 started 30 s ago", got[1])


class Starvation(unittest.TestCase):
    """A job that has waited > 15 min: no later ticket of the same or lower priority starts on a card where it
    could fit if the queue's own jobs with later tickets were not there."""

    def setUp(self):
        self.old = gq.STARVE + 60                        # seconds before NOW - 100
        self.q = [J("q1", QWEN, 100, gpu=0), J("q2", QWEN, 110, gpu=0)]    # running, later tickets than `big`
        self.cards = {0: C(18000, own={"id-q1": 9000, "id-q2": 9000})}

    def test_later_small_job_may_not_start(self):
        got = plan(self.q + [J("big", LLAVA, -self.old), J("small", QWEN, 200)], self.cards)
        self.assertIsNone(got["big"][0])
        self.assertIsNone(got["small"][0])
        self.assertIn("card 0 is kept for the long-waiting job big", got["small"][1])

    def test_before_15_minutes_backfilling_goes_on(self):
        got = plan(self.q + [J("big", LLAVA, -gq.STARVE + 160), J("small", QWEN, 200)], self.cards)
        self.assertEqual(got["small"][0], 0)

    def test_higher_priority_and_earlier_tickets_are_not_held_back(self):
        big = J("big", LLAVA, -self.old)
        self.assertEqual(plan(self.q + [big, J("small_hi", QWEN, 200, prio=1)], self.cards)["small_hi"][0], 0)
        self.assertEqual(plan(self.q + [big, J("older_lo", QWEN, -self.old - 5, prio=-1)], self.cards)["older_lo"][0], 0)
        self.assertIsNone(plan(self.q + [big, J("small_lo", QWEN, 200, prio=-1)], self.cards)["small_lo"][0])

    def test_only_cards_that_the_old_job_could_use(self):
        big, small = J("big", LLAVA, -self.old), J("small", QWEN, 200)
        cards = {0: C(20000, foreign=20000), 1: self.cards[0]}             # card 0: foreign jobs are in big's way
        self.assertEqual(plan([dict(q, gpu=1) for q in self.q] + [big, small], cards)["small"][0], 0)
        cards = {0: self.cards[0], 1: C(18000, own={"id-q3": 9000, "id-q4": 9000})}
        more = [J("q3", QWEN, 120, gpu=1), J("q4", QWEN, 130, gpu=1)]
        self.assertIsNone(plan(self.q + more + [big, small], cards)["small"][0])          # both cards are kept
        big1 = J("big", LLAVA, -self.old, want=1)                          # big asked for card 1: card 0 stays open
        self.assertEqual(plan(self.q + more + [big1, small], cards)["small"][0], 0)

    def test_jobs_with_earlier_tickets_are_not_imagined_away(self):
        earlier = [J("q1", QWEN, -self.old - 20, gpu=0), J("q2", QWEN, -self.old - 10, gpu=0)]
        got = plan(earlier + [J("big", LLAVA, -self.old), J("small", QWEN, 200)], self.cards)
        self.assertEqual(got["small"][0], 0)             # big would not fit even without the later jobs

    def test_while_the_old_job_waits_nothing_later_starts_on_its_cards(self):
        """also in the look in which the old job can start at last: it is still waiting until it has done so"""
        big, small = J("big", LLAVA, -self.old), J("small", 3000, 200)
        got = plan([big, small], {0: C(), 1: C()})
        self.assertEqual(got["big"][0], 0)
        self.assertEqual(got["small"], (None, "card 0 is kept for the long-waiting job big; card 1 is kept for the "
                                              "long-waiting job big"))
        self.assertEqual(plan([dict(big, gpu=0), small], {0: C(), 1: C()})["small"][0], 0)      # big runs: go

    def test_old_jobs_that_cannot_look_or_sleep_after_oom_keep_nothing(self):
        small = J("small", QWEN, 200)
        for big in (J("big", LLAVA, -self.old, beat=NOW - gq.BEAT - 1), J("big", LLAVA, -self.old, not_before=NOW + 30)):
            self.assertEqual(plan(self.q + [big, small], self.cards)["small"][0], 0)

    def test_the_old_job_starts_when_the_card_has_drained(self):
        cards = {0: C(9000, own={"id-q1": 9000})}
        got = plan(self.q[:1] + [J("big", LLAVA, -self.old), J("small", QWEN, 200)], cards)
        self.assertEqual(got["big"][0], 0)
        self.assertIsNone(got["small"][0])

    def simulate(self, starve, ncards, minutes=240):
        """Qwen jobs (20 min each) arrive every 3 min per card, twice what the cards can run; one LLaVA job arrives
        at minute 10, when every card holds three of them.  Returns the minute at which the LLaVA job starts
        (None: not within `minutes`)."""
        gq.STARVE, jobs, n = starve, [], 0
        for step in range(minutes * 4):                  # one decision every 15 s
            now = NOW + step * 15
            jobs = [j for j in jobs if j["gpu"] is None or j["end"] > now]
            if step % (3 * 4 // ncards) == 0:
                n += 1
                jobs.append(dict(J(f"q{n}", QWEN, 0), t=now, beat=now, end=None))
            if step == 10 * 4:
                jobs.append(dict(J("llava", LLAVA, 0), t=now, beat=now, end=None))
            for j in jobs:
                j["beat"] = now
            cards = {c: C(sum(j["need"] - 500 for j in jobs if j["gpu"] == c),
                          own={j["id"]: j["need"] - 500 for j in jobs if j["gpu"] == c}) for c in range(ncards)}
            for jid, (gpu, _) in gq.schedule(jobs, cards, now).items():
                if gpu is not None:
                    j = next(j for j in jobs if j["id"] == jid)
                    j.update(gpu=gpu, end=now + 20 * 60)
                    if j["name"] == "llava":
                        return step / 4
            for c in range(ncards):                      # the rule holds at every step
                self.assertLessEqual(sum(j["need"] for j in jobs if j["gpu"] == c) + 700, TOTAL)
        return None

    def test_llava_job_gets_a_card_although_qwen_jobs_keep_arriving(self):
        default = gq.STARVE
        try:
            for ncards in (1, 2):
                self.assertIsNone(self.simulate(1e12, ncards), "without the rule the LLaVA job starves (4 h)")
                start = self.simulate(default, ncards)
                self.assertIsNotNone(start)
                self.assertGreater(start, 10 + 15)       # not before it has waited 15 min ...
                self.assertLessEqual(start, 10 + 15 + 20 + 0.5)        # ... then as soon as the running jobs end
        finally:
            gq.STARVE = default


class RandomPlans(unittest.TestCase):
    """4000 random queues on random cards: what must hold for every plan of gq.schedule()."""

    def case(self, rng):
        ncards, cards, jobs = rng.choice([1, 2, 2, 3]), {}, []
        for c in range(ncards):
            while True:
                run = [J(f"r{c}{i}", rng.choice([3000, QWEN, LLAVA]), rng.uniform(-2000, 0), gpu=c, prio=rng.choice([0, 0, 1]))
                       for i in range(rng.randint(0, 2))]
                own = {j["id"]: rng.choice([0, j["need"] // 2, j["need"] - 300, j["need"] + 1500]) for j in run}
                foreign = rng.choice([0, 0, 5000, 12000, 20000]) + rng.choice([0, 0, 300])
                if foreign + sum(own.values()) + DRV <= TOTAL:
                    break
            cards[c] = C(foreign + sum(own.values()), foreign=foreign, own=own, busy=rng.choice(["", "", "", "loading"]))
            jobs += run
        for i in range(rng.randint(1, 7)):
            jobs.append(J(f"w{i}", rng.choice([1000, 3000, 6000, QWEN, LLAVA, 25000]), rng.uniform(-2000, 0),
                          prio=rng.choice([0, 0, 0, 1, -1]), want=rng.choice([None, None, 0, ncards - 1]),
                          beat=NOW - rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 500]),
                          not_before=NOW + rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 30])))
        return cards, jobs

    def test_what_holds_for_every_plan(self):
        rng, tk = random.Random(20261110), gq.ticket
        seen = dict(planned=0, room=0, busy=0, kept=0, old=0)
        for _ in range(4000):
            cards, jobs = self.case(rng)
            got = gq.schedule(jobs, cards, NOW)
            self.assertEqual(got, gq.schedule(rng.sample(jobs, len(jobs)), cards, NOW))      # whoever computes it
            waiting = [j for j in jobs if j["gpu"] is None]
            ok = {j["id"]: NOW - j["beat"] <= gq.BEAT and NOW >= j["not_before"] for j in waiting}
            plan_on = {c: [j for j in waiting if got[j["id"]][0] == c] for c in cards}
            running = {c: [j for j in jobs if j["gpu"] == c] for c in cards}

            def claimed(c, js, planned):
                return cards[c]["foreign"] + sum(max(j["need"], cards[c]["own"][j["id"]]) for j in js) + sum(j["need"] for j in planned)

            for c, card in cards.items():        # 1. the rule: the reservations + foreign memory leave 700 MiB
                for j in plan_on[c]:
                    seen["planned"] += 1
                    self.assertTrue(ok[j["id"]] and j["want"] in (None, c) and not card["busy"])
                    self.assertGreaterEqual(card["total"] - card["used"], j["need"] + 700)
                if plan_on[c]:
                    self.assertLessEqual(claimed(c, running[c], plan_on[c]) + 700, card["total"])
            room = {c: card["total"] - max(card["used"], claimed(c, running[c], plan_on[c])) for c, card in cards.items()}
            for j in waiting:                    # 2. no job waits without a reason: no card idles next to a job that fits
                card_of_j, why = got[j["id"]]
                if card_of_j is not None or not ok[j["id"]]:
                    continue
                allowed = [c for c in sorted(cards) if j["want"] in (None, c)]
                self.assertEqual(len(why.split("; ")), len(allowed), why)
                for c, part in zip(allowed, why.split("; ")):
                    if part.startswith(f"card {c}: room "):
                        seen["room"] += 1
                        self.assertLess(room[c], j["need"] + 700, part)
                    elif part.startswith(f"card {c} busy"):
                        seen["busy"] += 1
                        self.assertTrue(cards[c]["busy"])
                    else:
                        seen["kept"] += 1
                        w = next(w for w in waiting if part == f"card {c} is kept for the long-waiting job {w['name']}")
                        self.assertTrue(ok[w["id"]] and NOW - w["t"] > gq.STARVE and w["prio"] >= j["prio"]
                                        and tk(w) < tk(j) and w["want"] in (None, c), part)
            for w in waiting:                    # 3. the starvation rule, computed independently
                if not (ok[w["id"]] and NOW - w["t"] > gq.STARVE):
                    continue
                seen["old"] += 1
                for c, card in cards.items():
                    if w["want"] not in (None, c):
                        continue
                    later = [j for j in plan_on[c] if tk(j) > tk(w) and j["prio"] <= w["prio"]]
                    if later:                    # then w could not use card c even without the jobs with later tickets
                        early = [r for r in running[c] if tk(r) < tk(w)]
                        used = card["used"] - sum(card["own"][r["id"]] for r in running[c] if r not in early)
                        alone = card["total"] - max(used, claimed(c, early, [k for k in plan_on[c] if tk(k) < tk(w)]))
                        self.assertLess(alone, w["need"] + 700, (w["name"], [j["name"] for j in later]))
        self.assertTrue(all(n > 300 for n in seen.values()), seen)          # every branch was exercised


class Reading(unittest.TestCase):
    """gq.look() / gq.mark_busy() on an injected nvidia-smi reading and process table."""

    def setUp(self):
        self.gpus = [["0", "GPU-a", str(TOTAL), "0", str(TOTAL - DRV)], ["1", "GPU-b", str(TOTAL), "0", str(TOTAL - DRV)]]
        self.apps, self.procs, self.ages = [], {}, {}
        gq.smi = lambda q: self.gpus if "query-gpu" in q else self.apps
        gq.pstat = lambda pid: self.procs.get(pid)
        gq.age_of = lambda pid: self.ages.get(pid, 5000.0)

    def tearDown(self):
        gq.smi, gq.pstat, gq.age_of = REAL

    def use(self, card, pid, mib, ppid=1, age=5000.0):
        """a compute process on a card"""
        row = self.gpus[card]
        row[3], row[4] = str(int(row[3]) + mib), str(int(row[4]) - mib)
        self.apps.append([row[1], str(pid), str(mib)])
        self.procs[pid], self.ages[pid] = (ppid, 1, "S"), age

    def test_own_and_foreign_processes(self):
        run = J("run", QWEN, 0, gpu=0, pid=100)
        self.procs.update({100: (1, 1, "S"), 101: (100, 1, "S")})          # 100 = gq, 101 = bash, 102 = python
        self.use(0, 102, 9000, ppid=101)                 # descendant of the running job, on its card
        self.use(0, 200, 4000)                           # someone else's
        self.use(1, 103, 2500, ppid=101)                 # the job's process on a card it did not reserve
        cards = gq.look([run, J("wait", QWEN, 1, pid=300)])
        self.assertEqual(cards[0]["own"], {"id-run": 9000})
        self.assertEqual((cards[0]["foreign"], cards[1]["foreign"]), (4000, 2500))
        self.assertEqual((cards[0]["total"], cards[0]["used"], cards[0]["free"]), (TOTAL, 13000 + DRV, TOTAL - 13000 - DRV))
        self.assertEqual(cards[0]["pids"], ["200:4000"])
        self.assertEqual(cards[0]["young"], "")

    def test_gq_run_inside_gq_run_holds_both_cards(self):
        outer, inner = J("outer", 5000, 0, gpu=0, pid=100), J("inner", 4000, 1, gpu=1, pid=101)
        self.procs.update({100: (1, 1, "S"), 101: (100, 1, "S")})
        self.use(0, 102, 3000, ppid=101)                 # under inner (card 1) and outer (card 0): outer's, on card 0
        self.use(1, 103, 2000, ppid=101)
        cards = gq.look([outer, inner])
        self.assertEqual((cards[0]["own"], cards[1]["own"]), ({"id-outer": 3000}, {"id-inner": 2000}))
        self.assertEqual((cards[0]["foreign"], cards[1]["foreign"]), (0, 0))

    def test_processes_of_a_waiting_or_dead_job_are_foreign(self):
        self.procs[100] = (1, 1, "S")
        self.use(0, 102, 9000, ppid=100)
        self.assertEqual(gq.look([J("wait", QWEN, 0, pid=100)])[0]["foreign"], 9000)
        self.assertEqual(gq.look([])[0]["foreign"], 9000)

    def test_young_foreign_process(self):
        self.use(0, 200, 4000, age=gq.YOUNG - 1)
        self.use(1, 201, 4000, age=gq.YOUNG + 1)
        cards = gq.look([])
        self.assertIn("foreign pid 200 started 119 s ago", cards[0]["young"])
        self.assertEqual(cards[1]["young"], "")
        gq.mark_busy(gq.look([]), cards, [])
        self.assertTrue(cards[0]["busy"])
        self.assertFalse(cards[1]["busy"])
        self.assertEqual(plan([J("q", QWEN, 0)], cards)["q"][0], 1)

    def test_young_process_of_a_queue_job_does_not_make_the_card_busy(self):
        run = J("run", QWEN, 0, gpu=0, pid=100)
        self.procs[100] = (1, 1, "S")
        self.use(0, 102, 2000, ppid=100, age=3)
        cards = gq.look([run])
        gq.mark_busy(gq.look([run]), cards, [run])
        self.assertEqual(cards[0]["busy"], "")

    def test_unreadable_readings(self):
        self.use(0, 200, 4000)
        self.assertIsNotNone(gq.look([]))
        for bad in (None, [], [["0", "GPU-a", "x", "0", "1"]], [["0", "GPU-a", str(TOTAL)]],
                    [["Unable to determine the device handle for GPU0000:01:00.0: Unknown Error"]]):
            good, self.gpus = self.gpus, bad
            self.assertIsNone(gq.look([]), bad)
            self.gpus = good
        good, self.apps = self.apps, None                # the second query failed
        self.assertIsNone(gq.look([]))
        self.apps = [["GPU-unknown", "200", "4000"]]
        self.assertIsNone(gq.look([]))
        self.apps = [["GPU-a", "200", "[N/A]"]]          # memory not shown for this process: the card's "used" counts
        self.assertEqual(gq.look([])[0]["foreign"], 4000)
        self.apps = []                                   # no compute process is listed at all: the same
        self.assertEqual(gq.look([])[0]["foreign"], 4000)
        self.gpus[0][3], self.gpus[0][4] = "0", str(TOTAL - DRV)
        self.assertEqual(gq.look([])[0]["foreign"], 0)   # an empty card is a valid reading

    def test_reservation_is_not_hidden_under_memory_that_no_process_shows(self):
        """10000 MiB are used that nvidia-smi lists for no process; a job is admitted and has allocated nothing
        yet: its 10500 MiB come on top of them (max(used, reservations + listed processes) would say 10497)"""
        self.gpus[0][3], self.gpus[0][4] = "10000", str(TOTAL - DRV - 10000)
        run = J("run", QWEN, 0, gpu=0, pid=100)
        self.procs[100] = (1, 1, "S")
        cards = gq.look([run])
        self.assertEqual((cards[0]["foreign"], cards[0]["own"]), (10000, {}))
        gq.mark_busy(gq.look([run]), cards, [run])
        self.assertEqual(plan([run, J("a", TOTAL - 20500 - 700, 1, want=0)], cards)["a"][0], 0)
        self.assertIsNone(plan([run, J("a", TOTAL - 20500 - 700 + 1, 1, want=0)], cards)["a"][0])

    def growth(self, before_mib, after_mib, jobs=(), ppid=1):
        """busy text of card 0 after its only process went from before_mib to after_mib between the two looks"""
        self.use(0, 102, before_mib, ppid=ppid)
        before = gq.look(list(jobs))
        self.gpus[0][3], self.gpus[0][4] = str(after_mib), str(TOTAL - DRV - after_mib)
        self.apps[0][2] = str(after_mib)
        cards = gq.look(list(jobs))
        gq.mark_busy(before, cards, list(jobs))
        return cards[0]["busy"]

    def test_foreign_growth_makes_the_card_busy(self):
        self.assertEqual(self.growth(5000, 5000 + gq.GROW + 1), "memory grew by 301 MiB in 10 s")

    def test_small_growth_and_shrinking_do_not(self):
        self.assertEqual(self.growth(5000, 5000 + gq.GROW), "")
        self.setUp()
        self.assertEqual(self.growth(5000, 1000), "")

    def test_queue_job_filling_its_reservation_is_not_growth(self):
        self.procs[100] = (1, 1, "S")
        self.assertEqual(self.growth(1000, 9000, [J("run", QWEN, 0, gpu=0, pid=100)], ppid=100), "")

    def test_queue_job_filling_its_reservation_to_the_brim_is_not_growth(self):
        """nvidia-smi then shows more memory taken than was reserved (the driver's 497 MiB): still no growth"""
        self.procs[100] = (1, 1, "S")
        self.assertEqual(self.growth(0, QWEN - 50, [J("run", QWEN, 0, gpu=0, pid=100)], ppid=100), "")

    def test_queue_job_growing_beyond_its_reservation_is_growth(self):
        self.procs[100] = (1, 1, "S")
        self.assertIn("memory grew", self.growth(10200, 11200, [J("run", QWEN, 0, gpu=0, pid=100)], ppid=100))

    def test_growth_that_no_process_shows(self):
        before = gq.look([])
        self.gpus[0][3], self.gpus[0][4] = "2000", str(TOTAL - DRV - 2000)
        cards = gq.look([])
        gq.mark_busy(before, cards, [])
        self.assertIn("memory grew by 2000 MiB", cards[0]["busy"])
        self.assertEqual(cards[1]["busy"], "")

    def test_card_without_an_earlier_look_is_busy(self):
        cards = gq.look([])
        gq.mark_busy({1: cards[1]}, cards, [])
        self.assertTrue(cards[0]["busy"])
        self.assertEqual(cards[1]["busy"], "")


class StateFiles(unittest.TestCase):
    """job files: a file whose owner is dead is ignored and removed"""

    def setUp(self):
        for n in os.listdir(TMP):
            os.remove(os.path.join(TMP, n))

    def test_dead_and_live_owners(self):
        me = dict(J("me", QWEN, 0, gpu=0, pid=os.getpid()), pstart=gq.pstat(os.getpid())[1])
        dead = subprocess.Popen(["true"])
        dead.wait()
        gone = J("gone", QWEN, 1, gpu=0, pid=dead.pid)                     # this pid no longer exists
        recycled = dict(J("recycled", QWEN, 2, gpu=1, pid=os.getpid()), pstart=gq.pstat(os.getpid())[1] - 1)
        for j in (me, gone, recycled):
            gq.save(j)
            open(gq.path_of(j["id"], "err", "log"), "w").close()
        with open(os.path.join(TMP, "job-broken.json"), "w") as f:
            f.write("{not json")
        with open(os.path.join(TMP, "job-other.json"), "w") as f:         # live owner, fields of another version
            json.dump({"pid": os.getpid(), "pstart": me["pstart"]}, f)
        self.assertEqual([j["name"] for j in gq.load_jobs()], ["me"])      # ignored ...
        self.assertTrue(os.path.exists(gq.path_of("id-gone")))            # ... but only removed under the lock
        with gq.locked():
            self.assertEqual([j["name"] for j in gq.load_jobs(clean=True)], ["me"])
        left = sorted(os.listdir(TMP))
        self.assertEqual(left, ["err-id-me.log", "events.log", "job-broken.json", "job-id-me.json", "job-other.json", "lock"])
        with open(os.path.join(TMP, "events.log")) as f:
            self.assertEqual(sorted(json.loads(l)["name"] for l in f), ["gone", "recycled"])

    def test_a_zombie_owner_is_dead(self):
        z = subprocess.Popen(["true"])
        try:
            while gq.pstat(z.pid)[2] != "Z":
                pass
            self.assertFalse(gq.alive(dict(pid=z.pid, pstart=gq.pstat(z.pid)[1])))
        finally:
            z.wait()

    def test_defaults(self):
        """the constants of the task statement"""
        self.assertEqual((gq.MARGIN, gq.POLL, gq.LOOK, gq.YOUNG, gq.GROW, gq.STARVE, gq.OOM_WINDOW, gq.OOM_WAIT),
                         (700, 15.0, 10.0, 120.0, 300, 900.0, 120.0, 60.0))
        self.assertEqual(gq.CAPS, {})


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
