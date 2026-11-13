"""E1, behaviour (GPU through the queue; one development log per process; resumable: a finished cell is skipped, the memo holds
every forward that ran).

    gq run --mem 10500|17000 --prio 3 -- python e1_behave.py <development log> [--blocks fixed,cal,dual,gen,h3,dense,prompt]

Writes per-row files out/rows/<log>__<label>__d<draw|x>__<part>.jsonl.gz and the cell records out/behave/<log>/<cell>.json (method
record: digest, variant, capped edits; session counters).  All statistics are computed from the row files by e1_analyze.py.
Reduced programme (R19 a): alpha = 0.05, calibration draws 0 - 2; generation and D2 at draw 0 for the rows of the main table.
"""
import argparse
import os
import time

from e1c import *  # noqa: F401,F403
import e1c

R1, MT = fx.review1, fx.methods
ALPHA, DRAWS = 0.05, (0, 1, 2)
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


class Job:
    def __init__(self, name, dense_only=False):
        self.L = fx.logs.open(name)
        assert self.L.name in DEV or (dense_only and self.L.name in WP), "behaviour of E1 runs on the development logs only (R19 a)"
        self.dense_only = dense_only
        self.H = fx.logs.load_model(name)
        self.ses = fx.behave.session(self.L, self.H)
        self.R = fx.io.Results("E1", OUT)
        self.dir = os.path.join(BEH, self.L.name)
        os.makedirs(self.dir, exist_ok=True)
        self.sel = e1c.selected_labels(self.L)
        assert self.sel or dense_only, "the selection file must exist before any behavioural run of the selected gates"
        L = self.L
        self.ed = L.rows("edit")
        self.hd = L.pool("heldout_disjoint")
        self.hs = [j for j in L.pool("heldout_strict") if j not in set(self.hd)]          # strict rows outside heldout_disjoint (expected: none)
        self.hn = L.rows("hn", valid=True)

    # ------------------------------------------------------------------ cells
    def cell_path(self, cell):
        return os.path.join(self.dir, cell + ".json")

    def done(self, cell):
        return os.path.exists(self.cell_path(cell))

    def finish(self, cell, M, extra=None, st0=None):
        rec = {"cell": cell, "log": self.L.name, "label": M.label, "kind": M.kind, "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw,
               "digest": M.digest, "key": M.key, "n_capped": MT.n_capped(M), "cal": {k: v for k, v in (M.rec.get("cal") or {}).items() if not isinstance(v, (list, dict))},
               "raised": M.rec.get("raised"), "hook_mismatch": int(self.ses.stats["hook_mismatch"]),
               "memo_conflicts": int(self.ses.memo.stats.get("conflicts", 0)), "seconds": time.time() - T0,
               "session_delta": {k: self.ses.stats.get(k, 0) - (st0 or {}).get(k, 0) for k in ("rows", "pre", "memo", "run", "lib", "forwards", "decodes")},
               "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        rec.update(extra or {})
        assert rec["hook_mismatch"] == 0 and rec["memo_conflicts"] == 0, rec
        jdump(rec, self.cell_path(cell))
        say("done", cell, rec["session_delta"])
        self.ses.release()

    def tag(self, label, d, part):
        return f"{self.L.name}__{label}__d{'x' if d is None else d}__{part}"

    def build(self, label, d):
        if label == "rescope_lin":
            return MT.rescope(self.L, response="lin", alpha=ALPHA, draw=d, label="rescope_lin")
        M = R1.e1_rowset_review1(self.L, ALPHA, 0 if d is None else d, H=self.H, labels=(label,), skip_unavailable=False)[0]
        if not M.available:
            raise RuntimeError(f"{label}: {M.rec.get('not_defined') or M.rec.get('unavailable')}")
        return M

    # ------------------------------------------------------------------ teacher forcing: edits, heldout_disjoint (+ strict), hard negatives
    def tf(self, label, d, lp="tf"):
        cell = f"{label}__d{'x' if d is None else d}__{'tf' if lp == 'tf' else 'tfp'}"
        if self.done(cell):
            return
        M, st0 = self.build(label, d), dict(self.ses.stats)
        if lp == "prompt":                                              # the decode's position: only the sequences where the two positions differ
            sh = [j for j in self.ed + self.hd + self.hs + self.hn if self.L.index[j].get("lp_shift")]
            rows = self.ses.tf(M, sh, "prompt") if sh else []
            self.R.rows(self.tag(label, d, "tfp"), rows)
            self.finish(cell, M, {"n_rows": len(rows), "what": "sequences with lp_shift != 0 at the decode's position"}, st0)
            return
        rows = self.ses.tf(M, self.ed, lp) + self.ses.tf(M, self.hd + self.hs, lp) + self.ses.tf(M, self.hn, lp)
        self.R.rows(self.tag(label, d, "tf"), rows)
        self.finish(cell, M, {"n_rows": len(rows)}, st0)

    # ------------------------------------------------------------------ generation and D2 (draw 0; rows of the main table)
    def gen(self, label, d):
        cell = f"{label}__d{'x' if d is None else d}__gen"
        if self.done(cell):
            return
        M, st0 = self.build(label, d), dict(self.ses.stats)
        L = self.L
        d2rows = L.rows("probe", pool="heldout_disjoint", where=lambda e: bool(e.get("pre_correct_long")))
        prow = list(dict.fromkeys(self.hd + d2rows))
        ge = self.ses.generate(M, self.ed)
        gp = self.ses.generate(M, prow)
        _, gh = self.ses.hard_negatives(M, generate=True, rows=self.hn)
        for r in gp:
            r["in_d2_set"] = r.get("csid") in {L.sid(j) for j in d2rows}
        self.R.rows(self.tag(label, d, "gen"), ge + gp + gh)
        self.finish(cell, M, {"n_rows": len(ge) + len(gp) + len(gh), "n_d2_rows": len(d2rows)}, st0)

    # ------------------------------------------------------------------ DUAL (answer match): generation on everything + the labelled teacher-forced composition
    def dual(self):
        cell = "dual__dx__gen"
        if self.done(cell):
            return
        M, st0 = MT.dual(self.L), dict(self.ses.stats)
        L = self.L
        d2rows = L.rows("probe", pool="heldout_disjoint", where=lambda e: bool(e.get("pre_correct_long")))
        prow = list(dict.fromkeys(self.hd + d2rows))
        ge = self.ses.generate(M, self.ed)
        say("dual: edit prompts decoded")
        gp = self.ses.generate(M, prow)
        say("dual: probes decoded")
        _, gh = self.ses.hard_negatives(M, generate=True, rows=self.hn)
        say("dual: hard negatives decoded")
        for r in gp:
            r["in_d2_set"] = r.get("csid") in {L.sid(j) for j in d2rows}
        self.R.rows(self.tag("dual", None, "gen"), ge + gp + gh)
        tf = self.ses.dual_tf(M, self.ed) + self.ses.dual_tf(M, self.hd + self.hs) + self.ses.dual_tf(M, self.hn)
        self.R.rows(self.tag("dual", None, "tf"), tf)
        self.finish(cell, M, {"n_rows": len(ge) + len(gp) + len(gh), "tf": "ses.dual_tf: teacher-forced rows of the model DUAL selects (labelled composition)"}, st0)

    # ------------------------------------------------------------------ behavioural H3 (draws 0 - 2)
    def h3(self, label, d):
        cell = f"{label}__d{d}__h3"
        if self.done(cell):
            return
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
            ex.update(gate_level={"rephrase": h["rephrase"], "test": h["test"], "heldout_disjoint": h["heldout_disjoint"]}, raised_digest=Mh.digest, n_rows=len(e) + len(t) + len(p) + len(g))
            if label.endswith("_x"):
                sub = R1.hn_unseen_foreign(self.L)
                ex["unseen_foreign"] = {"n": sub["n"], "csids": [self.L.sid(j) for j in sub["rows"]]}
        self.finish(cell, M, ex, st0)

    # ------------------------------------------------------------------ gate level of the dense gates (they need the model's weights)
    def dense(self):
        L = self.L
        info = fx.gate.info(L, L.rows("probe", valid=True))
        prow = L.rows("probe", valid=True)
        ekind = np.array([L.index[j]["kind"] for j in self.ed])
        hkind = np.array([L.index[j]["hn_type"] for j in self.hn])
        for lab in DENSE:
            fixed = lab.endswith("_default")
            for alpha in ([None] if fixed else (ALPHAS if L.name in DEV else (ALPHA,))):      # working points: alpha = 0.05 (task 7)
                p = os.path.join(GATE, L.name, f"{lab}__a{'none' if fixed else afmt(alpha)}.json")
                if os.path.exists(p):
                    continue
                cells = {}
                for d in ([0] if fixed else range(N_DRAWS)):
                    M = MT.e1_rowset(L, 0.05 if fixed else alpha, d, H=self.H, labels=(lab,))[0]
                    dp, de, dh = (self.ses.decide_dense(M, r) for r in (prow, self.ed, self.hn))
                    f = np.asarray(dp["fired"], dtype=bool)
                    split = {}
                    for pool in ("disjoint", "strict"):
                        for mod in ("all", "image", "text"):
                            m = np.array([bool(r[pool]) and (mod == "all" or r["ptype"] == mod) for r in info])
                            split[f"{pool}.{mod}"] = {"n": int(m.sum()), "fired": 100.0 * float(f[m].mean()) if m.any() else None}
                    c = {"label": lab, "kind": M.kind, "variant": M.variant, "alpha": M.alpha, "n": M.n, "draw": M.draw, "digest": M.digest,
                         "threshold": float(M.obj.thr), "cal": {k: v for k, v in (M.rec.get("cal") or {}).items() if not isinstance(v, (list, dict))},
                         "n_capped": MT.n_capped(M), "split": split, "n_near": int(np.asarray(dp["near"]).sum() + np.asarray(de["near"]).sum() + np.asarray(dh["near"]).sum()),
                         "edit_fired": {k: 100.0 * float(np.asarray(de["fired"])[ekind == k].mean()) for k in ("rel", "tgen", "igen")},
                         "hn_fired": {k: 100.0 * float(np.asarray(dh["fired"])[hkind == k].mean()) for k in HN},
                         "note": "a dense gate releases the deployed output and has no per-edit decision: 'fired' on a rephrase is its coverage"}
                    if M.obj.kind == "router" and (fixed or abs(alpha - 0.05) < 1e-12 or True):
                        for pool in ("BEN-T", "MME-T"):
                            try:
                                b = fx.pools.firing(L, M, pool)
                                c[pool] = {k: b[k] for k in ("n", "n_fired", "fired", "which")}
                            except (NotImplementedError, R1.NotDefined) as e:
                                c[pool] = {"not_defined": str(e)[:300]}
                    else:
                        c["BEN-T"] = c["MME-T"] = {"not_defined": "the z-score gate needs whole sequences; key-level pools hold the last-prompt key only"}
                    cells[str(d)] = c
                jdump({"log": L.name, "label": lab, "alpha": None if fixed else alpha, "fixed": fixed, "n_draws": len(cells), "cells": cells, "dense": True,
                       "code": fx.code_fingerprint(), "written": time.strftime("%Y-%m-%d %H:%M:%S")}, p)
                say("dense gate level", lab, alpha)
        self.ses.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--blocks", default="fixed,cal,h3,dense,prompt,gen,dual")
    ap.add_argument("--dense-only", action="store_true", help="gate level of the dense gates only (working points: no behavioural evaluation)")
    a = ap.parse_args()
    if a.dense_only:
        J = Job(a.log, dense_only=True)
        J.dense()
        rec = J.ses.assert_clean()
        jdump({"log": J.L.name, "assert_clean": rec, "stats": dict(J.ses.stats), "what": "gate level of the dense gates (router: last-prompt keys; z-score: one truncated forward per sequence)",
               "seconds": time.time() - T0, "written": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(GATE, J.L.name, "_dense_job.json"))
        say("DENSE DONE", rec)
        return
    J = Job(a.log)
    s = J.sel
    best = list(dict.fromkeys([s["cos_H3"], s["cos_H2"]]))
    best_x = list(dict.fromkeys([s["cos_H3_x"], s["cos_H2_x"]]))
    retr = list(dict.fromkeys([s["retr_H3"], s["retr_H2"]]))
    fixed = ["deployed", "preedit", "exact", "cosine_default", "lookup", "zscore_default", "cosine_published", "where_logonly"]
    cal = list(dict.fromkeys(["rescope", "rescope_shared", "rescope_null", "rescope_x", "rescope_lin"] + best + best_x + ["cosine", "cosine_shared"] + retr + ["ball", "router"]))
    main_gen = [("deployed", None), ("preedit", None), ("rescope", 0), ("rescope_x", 0)] + [(b, 0) for b in best] + [(r, 0) for r in retr] + [("where_logonly", None)]
    h3 = list(dict.fromkeys(["rescope", "rescope_x", s["cos_H3"], s["cos_H3_x"], s["retr_H3"]]))
    say("log", J.L.name, "selection", s, "blocks", a.blocks)
    for blk in a.blocks.split(","):
        if blk == "fixed":
            for lab in fixed:
                J.tf(lab, None)
        elif blk == "cal":
            for lab in cal:
                for d in DRAWS:
                    J.tf(lab, d)
        elif blk == "h3":
            for lab in h3:
                for d in DRAWS:
                    J.h3(lab, d)
        elif blk == "dense":
            J.dense()
        elif blk == "prompt" and J.L.backbone == "llava":
            memo_fixed = ["exact", "cosine_default", "lookup"]
            memo_cal = [x for x in cal if x not in ("rescope_lin", "router")]
            for lab in memo_fixed:
                J.tf(lab, None, "prompt")
            for lab in memo_cal:
                for d in DRAWS:
                    J.tf(lab, d, "prompt")
        elif blk == "gen":
            for lab, d in main_gen:
                J.gen(lab, d)
        elif blk == "dual":
            J.dual()
    rec = J.ses.assert_clean()
    jdump({"log": J.L.name, "assert_clean": rec, "stats": dict(J.ses.stats), "memo": J.ses.memo.summary(), "blocks": a.blocks, "seconds": time.time() - T0,
           "written": time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(J.dir, f"_job_{int(time.time())}.json"))
    say("JOB DONE", rec)


if __name__ == "__main__":
    main()
