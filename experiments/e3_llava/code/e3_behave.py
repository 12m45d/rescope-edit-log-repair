"""E3 / E4 behaviour (GPU through the queue; ONE log per process; resumable: a finished cell is skipped, the memo holds every forward).

    gq run --mem 17000 --prio 2 -- python e3_behave.py <log> [--blocks fixed,cal,h3,gen,prompt,qd,pub,where]

Per-row files out/rows/<log>__<label>__d<draw|x>__<part>.jsonl.gz and cell records out/behave/<log>/<cell>.json (method record, digest,
capped edits, session counters).  All statistics are computed from the row files by e3_analyze.py.  Frozen rows, alpha = 0.05, n = 200;
calibration draws 0 - 2 (logs with 1000 or more edits: draw 0, R23).  Parts: tf = edits + heldout_disjoint + hard negatives of the log.
"""
import argparse
import os
import time

from e3c import *  # noqa: F401,F403
import e3c

R1, MT = fx.review1, fx.methods
T0 = time.time()
WHERE_BUDGET_S = 40 * 60


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


class Job:
    def __init__(self, name):
        assert name in LOGS
        self.L = L = fx.logs.open(name)
        self.lb = e3c.labels(L)
        self.H = fx.logs.load_model(name)
        self.ses = fx.behave.session(L, self.H)
        self.R = fx.io.Results("E3", OUT)
        self.dir = os.path.join(BEH, L.name)
        os.makedirs(self.dir, exist_ok=True)
        self.ed = L.rows("edit")
        self.hd = L.pool("heldout_disjoint")
        self.hn = L.rows("hn", valid=True)
        self.draws = bdraws(L.T)

    def cell_path(self, cell):
        return os.path.join(self.dir, cell + ".json")

    def done(self, cell):
        return os.path.exists(self.cell_path(cell))

    def finish(self, cell, M, extra=None, st0=None):
        st0 = st0 or {}
        rec = {"cell": cell, "log": self.L.name, "label": getattr(M, "label", None), "kind": getattr(M, "kind", None), "variant": getattr(M, "variant", None),
               "alpha": getattr(M, "alpha", None), "n": getattr(M, "n", None), "draw": getattr(M, "draw", None), "digest": getattr(M, "digest", None),
               "n_capped": MT.n_capped(M) if M is not None and getattr(M, "alpha", None) is not None else None,
               "cal": {k: v for k, v in ((M.rec.get("cal") or {}) if M is not None else {}).items() if not isinstance(v, (list, dict))},
               "hook_mismatch": int(self.ses.stats["hook_mismatch"]), "memo_conflicts": int(self.ses.memo.stats.get("conflicts", 0)), "seconds": time.time() - T0,
               "session_delta": {k: self.ses.stats.get(k, 0) - st0.get(k, 0) for k in ("rows", "pre", "memo", "run", "lib", "forwards", "decodes")},
               "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        rec.update(extra or {})
        assert rec["hook_mismatch"] == 0 and rec["memo_conflicts"] == 0, rec
        jdump(rec, self.cell_path(cell))
        say("done", cell, rec["session_delta"])
        self.ses.release()

    def tag(self, label, d, part):
        return f"{self.L.name}__{label}__d{'x' if d is None else d}__{part}"

    def build(self, label, d, qd=False):
        sw = {"mask": "qdisjoint"} if qd else {}
        M = R1.e1_rowset_review1(self.L, ALPHA, 0 if d is None else d, H=self.H, labels=(label,), skip_unavailable=False, **sw)[0]
        if not M.available:
            raise R1.NotDefined(f"{label}: {M.rec.get('not_defined') or M.rec.get('unavailable')}")
        return M

    def guarded(self, cell, fn):
        """Run a cell; a method that is not defined on the log is recorded as such; another failure is recorded and the job goes on."""
        if self.done(cell) or os.path.exists(self.cell_path(cell + "__FAILED2")):
            return
        try:
            fn(cell)
        except R1.NotDefined as e:
            jdump({"cell": cell, "log": self.L.name, "not_defined": str(e)[:500]}, self.cell_path(cell))
            say("not defined", cell, str(e)[:200])
        except AssertionError:
            raise
        except Exception as e:                                       # recorded; a second failure for the same cell skips it (cost discipline)
            import traceback
            p1 = self.cell_path(cell + "__FAILED1")
            jdump({"cell": cell, "error": f"{type(e).__name__}: {str(e)[:800]}", "trace": traceback.format_exc()[-3000:]},
                  self.cell_path(cell + "__FAILED2") if os.path.exists(p1) else p1)
            say("FAILED", cell, type(e).__name__, str(e)[:300])
            self.ses.release()

    # ---------------------------------------------------------------- teacher forcing: edits, heldout_disjoint, hard negatives
    def tf(self, label, d, part="tf"):
        def run(cell):
            qd = part == "tfq"
            M, st0 = self.build(label, d, qd), dict(self.ses.stats)
            if part == "tfp":                                        # the decode's position: the sequences on which the two positions differ
                sh = [j for j in self.ed + self.hd + self.hn if self.L.index[j].get("lp_shift")]
                rows = self.ses.tf(M, sh, "prompt") if sh else []
                self.R.rows(self.tag(label, d, part), rows)
                self.finish(cell, M, {"n_rows": len(rows), "what": "sequences with lp_shift != 0 at the decode's position"}, st0)
                return
            rows = self.ses.tf(M, self.ed, "tf") + self.ses.tf(M, self.hd, "tf") + self.ses.tf(M, self.hn, "tf")
            self.R.rows(self.tag(label, d, part), rows)
            self.finish(cell, M, {"n_rows": len(rows), "readout": "qd" if qd else "reg"}, st0)
        self.guarded(f"{label}__d{'x' if d is None else d}__{part}", run)

    # ---------------------------------------------------------------- Rescope-WHERE with the log-only thresholds (timed first at T >= 1000)
    def where(self):
        def run(cell):
            L, t0, st0 = self.L, time.time(), dict(self.ses.stats)
            M = self.build("where_logonly", None)
            t_build = time.time() - t0
            allrows = self.ed + self.hd + self.hn
            extra = {"seconds_build": t_build, "n_sequences": len(allrows)}
            if L.T >= 1000:
                smp = [allrows[i] for i in np.linspace(0, len(allrows) - 1, 60).astype(int)]
                t1 = time.time()
                self.ses.tf(M, smp, "tf")
                per = (time.time() - t1) / len(smp)
                est = t_build + per * len(allrows)
                extra.update(seconds_per_sequence=per, seconds_one_pass_estimated=est, budget_seconds=WHERE_BUDGET_S)
                if est > WHERE_BUDGET_S:
                    extra["skipped_for_cost"] = f"one pass is estimated at {est / 60:.0f} minutes (> 40 minutes): not run (PROTOCOL.md, R23)"
                    self.finish(cell, M, extra, st0)
                    return
            rows = self.ses.tf(M, self.ed, "tf") + self.ses.tf(M, self.hd, "tf") + self.ses.tf(M, self.hn, "tf")
            self.R.rows(self.tag("where_logonly", None, "tf"), rows)
            extra.update(n_rows=len(rows), seconds_one_pass=time.time() - t0)
            self.finish(cell, M, extra, st0)
        self.guarded("where_logonly__dx__tf", run)

    # ---------------------------------------------------------------- draw 0: D2 on heldout_disjoint; free generation for rescope
    def d2(self, label):
        def run(cell):
            M, st0 = self.build(label, 0), dict(self.ses.stats)
            rows = self.ses.d2(M, pool="heldout_disjoint")
            self.R.rows(self.tag(label, 0, "d2"), rows)
            self.finish(cell, M, {"n_rows": len(rows)}, st0)
        self.guarded(f"{label}__d0__d2", run)

    def gen(self, label="rescope"):
        def run(cell):
            L = self.L
            M, st0 = self.build(label, 0), dict(self.ses.stats)
            e100 = [L.edit_rows[m][i] for m in ("rel", "tgen", "igen") for i in range(100)]
            hd = sorted(self.hd, key=lambda j: L.sid(j))
            pick = sorted(np.random.default_rng(20261112).choice(len(hd), 200, replace=False).tolist())
            p200 = [hd[i] for i in pick]
            ge, gp = self.ses.generate(M, e100), self.ses.generate(M, p200)
            self.R.rows(self.tag(label, 0, "gen"), ge + gp)
            self.finish(cell, M, {"n_rows": len(ge) + len(gp), "probe_sample": "200 heldout_disjoint rows (sorted by sid), numpy default_rng(20261112).choice without replacement",
                                  "edits": "Rel / T-Gen / I-Gen prompts of the first 100 edits of the stream"}, st0)
        self.guarded(f"{label}__d0__gen", run)

    # ---------------------------------------------------------------- behavioural H3
    def h3(self, label, d):
        def run(cell):
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
            self.finish(cell, M, ex, st0)
        self.guarded(f"{label}__d{d}__h3", run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--blocks", default="fixed,cal,h3,gen,prompt,qd,pub,where")
    a = ap.parse_args()
    J = Job(a.log)
    lb = J.lb
    say("log", J.L.name, "T", J.L.T, "draws", J.draws, "selection", lb["sel"], "blocks", a.blocks)
    for blk in a.blocks.split(","):
        if blk == "fixed":
            for lab in ["deployed"] + lb["fixed_cpu"]:
                J.tf(lab, None)
        elif blk == "cal":
            for lab in lb["cal"]:
                for d in J.draws:
                    J.tf(lab, d)
        elif blk == "h3" and J.hn:
            for lab in lb["h3"]:
                for d in J.draws:
                    J.h3(lab, d)
        elif blk == "gen":
            for lab in lb["d2"]:
                J.d2(lab)
            J.gen("rescope")
        elif blk == "prompt":
            for lab in lb["fixed_cpu"]:
                J.tf(lab, None, "tfp")
            for lab in lb["cal"]:
                for d in J.draws:
                    J.tf(lab, d, "tfp")
        elif blk == "qd":
            for lab in lb["cal"]:
                for d in J.draws:
                    J.tf(lab, d, "tfq")
        elif blk == "pub":
            J.tf("cosine_published", None)
        elif blk == "where":
            J.where()
    rec = J.ses.assert_clean()
    jdump({"log": J.L.name, "assert_clean": rec, "stats": dict(J.ses.stats), "memo": J.ses.memo.summary(), "blocks": a.blocks, "seconds": time.time() - T0,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(J.dir, f"_job_{int(time.time())}.json"))
    say("JOB DONE", rec)


if __name__ == "__main__":
    main()
