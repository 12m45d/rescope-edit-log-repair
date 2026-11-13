"""E3 (Qwen), behaviour (GPU through the queue; ONE log per process; resumable: a finished cell is skipped, the memo holds every
forward that ran).

    gq run --mem 10500 --prio 2 -- python e3_behave.py <log> [--blocks fixed,cal,h3,d2gen,calq,lib]

Row files out/rows/<log>__<label>__d<draw|x>__<part>.jsonl.gz, cell records out/behave/<log>/<cell>.json (method record: digest, variant,
capped edits; session counters; hook_mismatch and memo conflicts asserted 0 in every cell and at the end of the job).  All statistics
are computed from the row files by e3_analyze.py.  Draws: 0 - 2 on logs with at most 500 edits, draw 0 on logs with 1000 or more (R23).
Parts: tf = teacher forcing on all edit sequences (Rel, T-Gen, I-Gen of all T edits), heldout_disjoint, the valid hard negatives of the
log's edits; tfq = the same under the calibration of the second read-out (mask='qdisjoint'); h3 = read-out H3 at the raised scale;
d2 = 48-token decode of the pre-correct heldout_disjoint rows (draw 0); gen = free generation (rescope, draw 0: prompts of the first
100 edits + 200 seeded heldout_disjoint probes).
"""
import argparse
import os
import time

from e3c import *  # noqa: F401,F403
import e3c

R1, MT = fx.review1, fx.methods
T0 = time.time()
WHERE_BUDGET_S = 40 * 60                 # task text / R23: Rescope-WHERE on logs with 1000 or more edits only if one pass fits in 40 minutes


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


class Job:
    def __init__(self, name):
        assert name in LOGS, "this task opens its seven logs only"
        self.L = L = fx.logs.open(name)
        self.lb = e3c.labels(L)                                         # raises if the selection file does not exist
        self.H = fx.logs.load_model(name)
        self.ses = fx.behave.session(L, self.H)
        self.R = fx.io.Results("E3", OUT)
        self.dir = os.path.join(BEH, L.name)
        os.makedirs(self.dir, exist_ok=True)
        self.draws = bdraws(name)
        self.ed = L.rows("edit")
        self.hd = L.pool("heldout_disjoint")
        self.hn = L.rows("hn", valid=True)
        self.all = list(self.ed) + list(self.hd) + list(self.hn)

    def cell_path(self, cell):
        return os.path.join(self.dir, cell + ".json")

    def done(self, cell):
        return os.path.exists(self.cell_path(cell))

    def finish(self, cell, M, extra=None, st0=None, t0=None):
        rec = {"cell": cell, "log": self.L.name, "label": M.label, "kind": M.kind, "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw,
               "digest": M.digest, "n_capped": MT.n_capped(M) if M.alpha is not None else None,
               "cal": {k: v for k, v in (M.rec.get("cal") or {}).items() if not isinstance(v, (list, dict))},
               "hook_mismatch": int(self.ses.stats["hook_mismatch"]), "memo_conflicts": int(self.ses.memo.stats.get("conflicts", 0)),
               "seconds_cell": None if t0 is None else time.time() - t0, "seconds_job": time.time() - T0,
               "session_delta": {k: self.ses.stats.get(k, 0) - (st0 or {}).get(k, 0) for k in ("rows", "pre", "memo", "run", "lib", "forwards", "decodes")},
               "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        rec.update(extra or {})
        assert rec["hook_mismatch"] == 0 and rec["memo_conflicts"] == 0, rec
        jdump(rec, self.cell_path(cell))
        say("done", cell, rec["session_delta"])
        self.ses.release()

    def tag(self, label, d, part):
        return f"{self.L.name}__{label}__d{'x' if d is None else d}__{part}"

    def build(self, label, d, **sw):
        M = R1.e1_rowset_review1(self.L, ALPHA, 0 if d is None else d, H=self.H, labels=(label,), skip_unavailable=False, **sw)[0]
        if not M.available:
            raise RuntimeError(f"{label}: {M.rec.get('not_defined') or M.rec.get('unavailable')}")
        return M

    def guarded(self, cell, fn):
        """Run a cell; a failure is recorded (cell file <cell>.FAILED<k>.json) and the job moves on; after two failures the cell is skipped."""
        if self.done(cell):
            return
        k = sum(os.path.exists(self.cell_path(f"{cell}.FAILED{i}")) for i in (1, 2))
        if k >= 2:
            say("skip (failed twice)", cell)
            return
        try:
            fn()
        except AssertionError:
            raise
        except Exception as e:                                         # noqa: BLE001
            import traceback
            jdump({"cell": cell, "error": f"{type(e).__name__}: {str(e)[:1000]}", "trace": traceback.format_exc()[-3000:], "written": time.strftime("%Y-%m-%d %H:%M:%S")},
                  self.cell_path(f"{cell}.FAILED{k + 1}"))
            say("FAILED", cell, type(e).__name__, str(e)[:300])
            self.ses.release()

    # ------------------------------------------------------------------ teacher forcing: all edits, heldout_disjoint, hard negatives
    def tf(self, label, d, masked=False):
        part = "tfq" if masked else "tf"
        cell = f"{label}__d{'x' if d is None else d}__{part}"

        def run():
            t0 = time.time()
            M, st0 = self.build(label, d, **({"mask": "qdisjoint"} if masked else {})), dict(self.ses.stats)
            rows = self.ses.tf(M, self.ed) + self.ses.tf(M, self.hd) + self.ses.tf(M, self.hn)
            self.R.rows(self.tag(label, d, part), rows)
            self.finish(cell, M, {"n_rows": len(rows), "masked": masked}, st0, t0)
        self.guarded(cell, run)

    # ------------------------------------------------------------------ D2 (draw 0 / fixed rows): 48-token decode of the pre-correct heldout_disjoint rows
    def d2(self, label, d):
        cell = f"{label}__d{'x' if d is None else d}__d2"

        def run():
            t0 = time.time()
            M, st0 = self.build(label, d), dict(self.ses.stats)
            rows = self.ses.d2(M)
            self.R.rows(self.tag(label, d, "d2"), rows)
            self.finish(cell, M, {"n_rows": len(rows)}, st0, t0)
        self.guarded(cell, run)

    # ------------------------------------------------------------------ free generation (rescope, draw 0)
    def gen(self, label, d):
        cell = f"{label}__d{d}__gen"

        def run():
            t0 = time.time()
            L = self.L
            M, st0 = self.build(label, d), dict(self.ses.stats)
            e100 = [L.edit_rows[m][i] for m in ("rel", "tgen", "igen") for i in range(100)]
            hd = sorted(self.hd)
            pick = sorted(np.random.RandomState(20261112).choice(len(hd), size=200, replace=False).tolist())
            prow = [hd[i] for i in pick]
            ge = self.ses.generate(M, e100)
            gp = self.ses.generate(M, prow)
            self.R.rows(self.tag(label, d, "gen"), ge + gp)
            self.finish(cell, M, {"n_rows": len(ge) + len(gp), "probe_sample": "200 heldout_disjoint rows, numpy RandomState(20261112).choice without replacement on the sorted pool",
                                  "edit_prompts": "Rel, T-Gen, I-Gen prompts of the first 100 edits of the stream"}, st0, t0)
        self.guarded(cell, run)

    # ------------------------------------------------------------------ behavioural H3
    def h3(self, label, d):
        cell = f"{label}__d{d}__h3"

        def run():
            t0 = time.time()
            M, st0 = self.build(label, d), dict(self.ses.stats)
            h = fx.sweeps.h3(self.L, M)
            ex = {"reachable": bool(h["reachable"]), "delta": h.get("delta"), "n_edits_at_cap": h.get("n_edits_at_cap"), "odd_edits": [int(x) for x in h["odd_edits"]]}
            if h["reachable"]:
                Mh = h["method"]
                e = self.ses.edits(Mh, edits=h["odd_edits"])
                t, g = self.ses.hard_negatives(Mh, rows=h["test_rows"]["rows"], generate=True)
                p = self.ses.tf(Mh, self.hd)
                for r in g:
                    r["part"] = "gen"
                self.R.rows(self.tag(label, d, "h3"), e + t + p + g)
                ex.update(gate_level={"rephrase": h["rephrase"], "test": h["test"], "heldout_disjoint": h["heldout_disjoint"]}, raised_digest=Mh.digest,
                          n_rows=len(e) + len(t) + len(p) + len(g))
                if label.endswith("_x"):
                    sub = R1.hn_unseen_foreign(self.L)
                    ex["unseen_foreign"] = {"n": sub["n"], "csids": [self.L.sid(j) for j in sub["rows"]]}
            self.finish(cell, M, ex, st0, t0)
        self.guarded(cell, run)

    # ------------------------------------------------------------------ all-position rows (library path: one forward per sequence)
    def lib(self, label):
        cell = f"{label}__dx__tf"
        skip = self.cell_path(f"{label}__dx__SKIPPED")
        if self.done(cell) or os.path.exists(skip):
            return

        def run():
            t0 = time.time()
            M, st0 = self.build(label, None), dict(self.ses.stats)
            t_build = time.time() - t0
            timing = {"build_s": t_build}
            if label == "where_logonly":
                samp = [self.all[i] for i in sorted(np.random.RandomState(0).choice(len(self.all), size=min(100, len(self.all)), replace=False).tolist())]
                t1 = time.time()
                self.ses.tf(M, samp)
                per = (time.time() - t1) / len(samp)
                timing.update(per_sequence_s=per, n_sequences=len(self.all), projected_pass_s=t_build + per * len(self.all), budget_s=WHERE_BUDGET_S,
                              rule="logs with 1000 or more edits: run only if build + one pass over all sequences is projected (100 seeded sequences) to fit in 40 minutes")
                say("where_logonly timing", timing)
                if self.L.T >= 1000 and timing["projected_pass_s"] > WHERE_BUDGET_S:
                    jdump({"cell": cell, "log": self.L.name, "label": label, "skipped_for_cost": True, "timing": timing, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, skip)
                    say("where_logonly SKIPPED FOR COST")
                    self.ses.release()
                    return
            rows = self.ses.tf(M, self.ed) + self.ses.tf(M, self.hd) + self.ses.tf(M, self.hn)
            self.R.rows(self.tag(label, None, "tf"), rows)
            self.finish(cell, M, {"n_rows": len(rows), "timing": timing, "where_cal": {k: v for k, v in (M.rec.get("cal") or {}).items() if not isinstance(v, (list, dict))}}, st0, t0)
        self.guarded(cell, run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--blocks", default="fixed,cal,h3,d2gen,calq,lib")
    a = ap.parse_args()
    J = Job(a.log)
    lb = J.lb
    say("log", J.L.name, "T", J.L.T, "draws", J.draws, "selection", lb["sel"], "blocks", a.blocks, "sequences", len(J.all))
    for blk in a.blocks.split(","):
        if blk == "fixed":
            for lab in lb["weights"] + lb["fixed"]:
                J.tf(lab, None)
        elif blk == "cal":
            for d in J.draws:
                for lab in lb["cal"]:
                    J.tf(lab, d)
        elif blk == "h3":
            for d in J.draws:
                for lab in lb["h3"]:
                    J.h3(lab, d)
        elif blk == "d2gen":
            for lab in lb["weights"] + lb["fixed"]:
                J.d2(lab, None)
            for lab in lb["cal"]:
                J.d2(lab, 0)
            J.gen("rescope", 0)
        elif blk == "calq":
            for d in J.draws:
                for lab in lb["cal"]:
                    J.tf(lab, d, masked=True)
        elif blk == "lib":
            for lab in lb["lib"]:
                J.lib(lab)
                if J.done(f"{lab}__dx__tf"):
                    J.d2(lab, None)
        say("block done", blk)
    rec = J.ses.assert_clean()
    jdump({"log": J.L.name, "assert_clean": rec, "stats": dict(J.ses.stats), "memo": J.ses.memo.summary(), "blocks": a.blocks, "seconds": time.time() - T0,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(J.dir, f"_job_{int(time.time())}.json"))
    say("JOB DONE", rec)


if __name__ == "__main__":
    main()
