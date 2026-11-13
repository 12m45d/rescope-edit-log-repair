"""fx.behave: behavioural evaluation (the model's answers) with the persistent memo.

    ses = fx.behave.session(L, H)                       # L = fx.logs.open(name), H = fx.logs.load_model(name) (None: memo / cache only)
    rows = ses.edits(M)                                 # Rel / T-Gen / I-Gen, teacher forced, B = 1: exact, token_acc, firing flags
    rows = ses.probes(M, pool="heldout_disjoint")       # D1, firing, tok_loc (token-level locality)
    rows = ses.hard_negatives(M)                        # D1, firing; generate=True: + wrong-target rate from generated answers
    rows = ses.d2(M, pool="heldout_disjoint")           # 48-token decode of the pre-correct rows: D2
    rows = ses.generate(M, seqs)                        # free generation (decoder v2) of any sequences
    {label: rows} = ses.edits({label: M, ...})          # every function also takes a dict of methods

THREE PATHS (chosen per method; every forward of every path runs through the library's Evaluator and its hook)
  memo        scoped replay with role 'last', top-1, response REPL (Method.memo: ReScope, the cosine / ball / exact gates, any draw):
              1. the gate is decided on the CPU from the key cache (fx.gate.decide); a row inside the 1e-9 decision band is decided
                 on the model's device, one key at a time = the arithmetic of the hook;
              2. a sequence on which nothing fires IS the pre-edit model: its row = the cached pre-edit outcome of the key cache;
              3. a sequence on which edit j fires: memo[(sequence, mode, signature of the released value V_j)]; a miss runs ONE
                 forward through rescope.Evaluator with the real gate and stores the outcome; the hook's own decision is compared
                 with the predicted one (session.stats['hook_mismatch'] must stay 0; the hook's decision wins).
              Decodes (d2, generate): the signature lists the value released in each of the 11 full forwards of the decode.
  weights     deployed (W_E), preedit (W_0), any replacement weight: memo[(sequence, mode, signature of the weight)]; the pre-edit
              teacher-forced outcome comes from the key cache (no model at all).
  dense       router: decided from the key at the last prompt position; a routed sequence has one outcome whatever the threshold
              (memoised under the deployed weight's signature), an unrouted one is the pre-edit model.  z-score: a sequence whose
              largest z-score (cached per log: Session.zmax) is below the threshold is the pre-edit model; the others run through
              the library.  Teacher forcing only (their decodes take the library path); session.dense_fast = False disables both.
  library     everything else (other roles / responses / fire 'all', harness corrections): rescope.SparseEvaluator (= the Evaluator's
              rows; for scoped gates it skips sequences that are surely silent), no memo.
  DUAL        generation only.  Its two decodes per query (deployed, pre-edit) are the memoised decodes of the weights path; the rule is
              the library's Dual.match and the row is that of rescope.Evaluator.generate (session.dual_from_memo = False runs the
              library's own DUAL instead).  dual_tf: the teacher-forced rows of the model DUAL selects (a labelled composition).
ROWS are plain dicts in the library's vocabulary (kind 'edit' | 'probe' | 'hn', sid = the library's sid, metric / ptype / hn_type, exact,
token_acc, n_tok, finite, first_mismatch, D1, tok_loc, fired, top_last, own_fired_last, own_released_last, n_cand_edits, lp, lp_shift) plus
csid (the sequence id of the key cache), near, near_edit (probe rows: an edit of the log asks the same question; near_reph: or has
it as its text rephrase), y0_deg (the pre-edit answer is degenerate: empty / one character / '?' / bare None-No), query_id (probe
rows with the same id are the same query: the key cache's content id = question string + image content, as fx.draws.query_ids;
probe_query_id: the library's id, by image path; the two partitions are the same on the held-out pools), hard negatives: rule,
new, same_category, related_flag, src_related_to_log, q_edit; src (where the outcome came from: 'pre' | 'memo' | 'run' | 'lib')
-- rescope.summarise / rescope.locality accept them.
Without a model (H=None) the memo and weights paths still answer whatever the memo holds; on_miss='raise' (default) | 'skip'.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch

import fx
from . import gate as G, memo as MM

OUT_TF = ("exact", "token_acc", "n_tok", "finite", "first_mismatch", "D1", "tok_loc")
OUT_DEC = ("y", "long", "line", "stop", "long_stop", "n_long", "degenerate", "eq", "eq10", "contains", "starts", "agree", "long_eq_pre", "pre_correct",
           "post_correct", "D2", "leak", "n_calls_last")
PRE = ("exact", "token_acc", "n_tok", "finite", "first_mismatch")
_SES = {}
CHUNK_TF, CHUNK_DEC = 128, 16         # forwards / decodes per call of the library's Evaluator (the memo is written after each)
SLACK_Z = 1e-9                        # relative band around a z-score threshold inside which a sequence is run, not decided from its cached score


class MemoMiss(RuntimeError):
    pass


def session(L, H=None, memo: bool = True):
    """The evaluation session of a log (one per process, log and handle)."""
    key = (L.name, id(H), bool(memo))
    if key not in _SES:
        _SES[key] = Session(L, H, memo)
    return _SES[key]


def _tb():
    from rescope import harness
    return harness().TB


def y0_degenerate(e: dict) -> bool:
    """The pre-edit answer of a cached row is degenerate: empty, one character, '?', or a bare None / No (flags of the key cache)."""
    return any(bool(e.get("deg_" + k)) for k in ("empty", "single_char", "qmark", "none_no"))


class Session:
    def __init__(self, L, H=None, memo: bool = True):
        self.L, self.H = L, H
        self.memo = MM.open(L.backbone, layer=L.layer) if memo else None
        self.use_memo = bool(memo)
        self.ev = self.ev_plain = None
        self._items, self._vsig, self._hook_gate, self._wsig, self._norm = None, {}, None, {}, None
        self.stats = {"rows": 0, "pre": 0, "memo": 0, "run": 0, "lib": 0, "missing": 0, "forwards": 0, "decodes": 0, "hook_mismatch": 0, "near": 0,
                      "near_changed_on_device": 0, "seconds_model": 0.0}
        self.mismatches = []
        self.verbose = False                # True: one line per chunk of forwards (long jobs)
        self.dense_fast = True              # router: memoised routed outcome; z-score: sequences below the threshold are the pre-edit model
        self._z, self._z_path = None, os.path.join(fx.MEMO_ROOT, "aux", L.name, "zscore_max.json")
        self.dual_from_memo = True          # DUAL composed from the memoised deployed / pre-edit decodes (False: rescope.Evaluator.generate)
        if H is not None:
            from rescope import HN_DIR_FINAL, Evaluator, SparseEvaluator
            if H.name != L.name:
                raise ValueError(f"handle of {H.name} for log {L.name}")
            self.ev = SparseEvaluator(H, cache=L.kc, hn_dir=HN_DIR_FINAL)       # the library path (rows == Evaluator's rows)
            self.ev_plain = Evaluator(H, hn_dir=HN_DIR_FINAL)                   # memo misses: every forward through the hook
            if [d["uid"] for d in H.items] != L.uids:
                raise RuntimeError("the handle's edit stream is not the log's")
            if [d["alt"] for d in H.items] != L.targets:
                raise RuntimeError("the handle's edit targets differ from the key cache's")

    # ------------------------------------------------------------------ items
    def item(self, j: int) -> dict:
        """The library's item of a sequence (edit / probe / hard-negative row)."""
        if self.ev is None:
            raise RuntimeError("no model: this needs H = fx.logs.load_model(name)")
        if self._items is None:
            its = self.ev.edit_items() + self.ev.probe_items("all", valid_only=False)
            if self.ev._hn_error is None:
                its += self.ev.hn_items()
            self._items = {it["sid"]: it for it in its}
        sid = self.L.index[j].get("lib_sid")
        if sid not in self._items:
            raise KeyError(f"sequence {self.L.sid(j)} has no item in the library's evaluation (benign rows and hard negatives with an invalid "
                           "pre-edit answer are gate-level only)")
        return self._items[sid]

    def base_row(self, j: int, lp: str) -> dict:
        e = self.L.index[j]
        k = e["kind"]
        if k == "benign":
            raise KeyError(f"{e['sid']}: the benign pool is gate-level only (fx.gate); it has no row in the behavioural evaluation")
        r = {"kind": "edit" if k in ("rel", "tgen", "igen") else ("probe" if k == "probe" else "hn"), "sid": e.get("lib_sid"), "csid": e["sid"], "lp": lp,
             "lp_shift": e.get("lp_shift")}
        if r["kind"] == "edit":
            r.update(metric=k, edit=int(e["edit"]), uid=e["edit_uid"])
        elif r["kind"] == "probe":
            r.update(probe_idx=e["probe_idx"], ptype=e["ptype"], uid=e["uid"], group=e["group"], pool=e["pool"], disjoint=bool(e.get("disjoint")),
                     strict=bool(e.get("strict")), question_in_repair=bool(e.get("question_in_repair")), query_id=e["query_id"],
                     probe_query_id=e.get("probe_query_id"),
                     y0_valid=bool(e["y0_valid"]), y0_deg=y0_degenerate(e), near_edit=self.L.near_edit(j), near_reph=self.L.near_edit(j, True))
        else:
            b, T = e.get("edit_b"), self.L.T
            rel, qe = e.get("src_related_edit_first_pos"), e.get("q_in_edit_first_pos")
            r.update(hn_id=e["hn_id"], hn_type=e["hn_type"], edit=self.L.own(j), uid=e["edit_uid"], hn_group=e.get("hn_group"), y0_valid=bool(e["y0_valid"]),
                     y0_deg=y0_degenerate(e), related_flag=bool(e.get("related_flag")), same_category=bool(e.get("same_category")), new=bool(e.get("new")),
                     rule=e.get("rule"), src_related_to_log=rel is not None and int(rel) < T,       # the row's source item is related to an edit of THIS log
                     q_edit=int(qe) if qe is not None and int(qe) < T else None)                    # an edit of this log asks the row's question
            if b is not None:
                r["edit_b"] = int(b)
        return r

    def _pre_outcome(self, j: int) -> dict:
        """Teacher-forced outcome of the PRE-EDIT model on the sequence, from the key cache (as SparseEvaluator's silent row)."""
        e = self.L.index[j]
        if e.get("pre_exact") is None:
            raise LookupError(f"{e['sid']}: the key cache holds no pre-edit outcome (no teacher-forced sequence)")
        o = {k: e["pre_" + k] for k in PRE}
        o["exact"], o["token_acc"], o["n_tok"], o["finite"] = bool(o["exact"]), float(o["token_acc"]), int(o["n_tok"]), bool(o["finite"])
        if e["kind"] not in ("rel", "tgen", "igen"):
            o["D1"] = not o["exact"]
            o["tok_loc"] = o["token_acc"] if e.get("y0_valid") else (1.0 if o["finite"] else None)
        return o

    # ------------------------------------------------------------------ signatures and device copies
    def vsig(self, g, edit: int) -> str:
        # Reviewer patch P2 (review/patches.txt): the signature of the value the gate holds NOW, hashed at every call (about 20 microseconds).
        # The cache that stood here was keyed by id(g.values): a values tensor changed in place kept the signatures of its old contents.
        return MM.value_sig(g.values[int(edit)])

    def wsig(self, W) -> str:
        # Reviewer patch P1 (review/patches.txt): hashed at every call (0.07 s Qwen, 0.13 s LLaVA).  The cache that stood here was keyed by
        # id(W): a weight buffer changed in place (e.g. a rollback accumulated in one tensor) kept the signature of its FIRST contents, so
        # that the memo answered with the first weight's rows and stored the new weight's outcomes under the first weight's signature.
        return MM.weight_sig(W)

    def hook_gate(self, M):
        """The gate on the model's device for the hook (same detectors, thresholds and values, bit for bit; guards are not needed
        by a forward and are left on the CPU).  One device copy at a time."""
        from . import methods as MT
        g = M.obj
        # patch P2 of the review: the copy holds THESE values; stage-1 extension (review finding 2): ... and these detectors, thresholds,
        # baselines, role and response -- the content digest of the gate (0.02 s at T = 100, 0.5 s at T = 2000, per chunk of forwards)
        vh = MT.content_digest(g)
        if self._hook_gate is None or self._hook_gate[0] is not g or self._hook_gate[2] != vh:
            self._hook_gate = None
            torch.cuda.empty_cache()
            slim = g.replace(guard_key=g.guard_key[:0], guard_edit=g.guard_edit[:0], W0=self.H.W0)
            self._hook_gate = (g, slim.to(torch.device(self.H.dev)), vh)
        return self._hook_gate[1]

    @staticmethod
    def _release_dense():
        """Drop the library's cached fp64 weight difference W_E - W_0 (0.2 - 0.4 GB on the model's device; rebuilt on demand)."""
        from rescope import baselines
        baselines.release()
        torch.cuda.empty_cache()

    def release(self):
        """Drop the device copy of the last gate and the library's cached weight difference."""
        from rescope import baselines
        self._hook_gate = None
        baselines.release()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ gate decision (CPU + the hook's arithmetic where near)
    def decide(self, M, rows, which: str = "tf") -> dict:
        """fx.gate.decide + the rows inside the decision band re-decided on the model's device, one key at a time (bit for bit the
        hook's computation).  Without a model the CPU decision stands (near rows stay flagged)."""
        d = G.decide(self.L, M, rows, which)
        nn = np.nonzero(d["step_near"] if which == "steps" else d["near"])
        self.stats["near"] += int(len(nn[0]))
        if self.H is None or len(nn[0]) == 0:
            return d
        from rescope.scoped import INF
        gg = self.hook_gate(M)
        th = gg.theta()
        kc, ix = self.L.kc, self.L.index
        for t in zip(*nn):
            i = int(t[0])
            j = d["rows"][i]
            if which == "steps":
                key = kc.K("steps")[int(ix[j]["rows_decode_steps"][int(t[1])])]
            else:
                key = kc.K(G._FILE[which])[kc.key_rows(which, [j])[0]]
            r = gg.scores(key[None].to(gg.device))
            cand = r >= th[None, :]
            f = bool(cand.any())
            top = int(gg.det_edit[torch.where(cand, gg.rank_key(r), torch.full_like(r, -INF)).argmax(1)[0]]) if f else -1
            ce = gg._edit_any(cand)[0]
            o = int(d["own"][i])
            if which == "steps":
                s = int(t[1])
                self.stats["near_changed_on_device"] += int(bool(d["step_fired"][i, s]) != f or int(d["step_top"][i, s]) != top)
                d["step_fired"][i, s], d["step_top"][i, s] = f, top
            else:
                self.stats["near_changed_on_device"] += int(bool(d["fired"][i]) != f or int(d["top_edit"][i]) != top)
                d["fired"][i], d["top_edit"][i], d["n_cand"][i] = f, top, int(ce.sum())
                d["own_cand"][i] = bool(o >= 0 and ce[o])
                d["own_top"][i] = bool(f and o >= 0 and top == o)
        if which == "steps":                                            # per-row summaries from the (possibly corrected) steps
            f = d["step_fired"]
            first = np.where(f.any(1), f.argmax(1), 0)
            ar = np.arange(f.shape[0])
            d["fired"] = f.any(1)
            d["top_edit"] = np.where(d["fired"], d["step_top"][ar, first], -1)
            d["own_top"] = d["fired"] & (d["own"] >= 0) & (d["top_edit"] == d["own"])
        return d

    # ------------------------------------------------------------------ teacher forcing
    def tf(self, methods, rows, lp: str = "tf", on_miss: str = "raise", verbose: int = 0):
        """Teacher-forced rows of sequences `rows` (positions in the key-cache index) under one method or a dict {label: Method}."""
        many = isinstance(methods, dict)
        ms = methods if many else {"_": methods}
        rows = [int(j) for j in rows]
        if not rows:
            return {lab: [] for lab in ms} if many else []
        out, lib = {}, {}
        for lab, M in ms.items():
            if not M.available:
                raise RuntimeError(f"{lab}: {M.rec.get('unavailable')}")
            if M.kind == "dual":
                raise TypeError("DUAL is defined on generated answers: use generate / d2 / hard_negatives(generate=True)")
            if self.use_memo and M.memo:
                out[lab] = self._tf_memo(M, rows, lp, on_miss)
            elif M.kind == "request":                                   # stage 2 (fx.review1): retrieval methods, through the memo path
                from . import review1
                out[lab] = review1.session_tf(self, M, rows, lp, on_miss)
            elif M.kind in ("deployed", "preedit", "weight"):
                out[lab] = self._tf_weight(M, rows, lp, on_miss)
            elif self.use_memo and self.dense_fast and M.kind == "dense" and self.H is not None:
                out[lab] = self._tf_router(M, rows, lp) if M.obj.kind == "router" else self._tf_zscore(M, rows, lp)
                self._release_dense()
            else:
                lib[lab] = M
        if lib:
            if self.ev is None:
                raise RuntimeError(f"{list(lib)}: these methods run through the library's Evaluator: a model is needed")
            items = [self.item(j) for j in rows]
            for part in self._lib_parts(lib):                           # long logs: one gate on the device at a time
                t0 = time.time()
                res = self.ev.tf_rows({lab: self._lib_obj(M) for lab, M in part.items()}, items, verbose=verbose, lp=lp)
                self.stats["seconds_model"] += time.time() - t0
                for lab in part:
                    out[lab] = [self._from_lib(j, r, lp) for j, r in zip(rows, res[lab])]
                    self.stats["lib"] += len(rows)
                    self.stats["rows"] += len(rows)
                del res
                self.release()
        return {lab: out[lab] for lab in ms} if many else out["_"]

    def _lib_obj(self, M):
        """What the library's Evaluator receives for a method of the library path (gates: with W_0 for the hook's check, without
        the guard keys, which a forward does not need)."""
        if M.kind == "preedit":
            return self.H.W0
        g = M.obj
        if hasattr(g, "det_tau"):
            return g.replace(guard_key=g.guard_key[:0], guard_edit=g.guard_edit[:0], W0=self.H.W0)
        return g

    def _lib_parts(self, lib: dict) -> list:
        if self.L.T < 500:
            return [lib]
        return [{lab: M} for lab, M in lib.items()]

    def _from_lib(self, j, r, lp, src="lib") -> dict:
        row = self.base_row(j, lp)
        for k in OUT_TF + OUT_DEC:
            if k in r:
                row[k] = r[k]
        row["fired"] = r.get("fired")
        for k in ("top_last", "own_fired_last", "own_released_last", "n_cand_edits", "dual", "n_generations", "n_new_tokens"):
            if k in r:
                row[k] = r[k]
        row["near"], row["src"] = None, src
        return row

    def _flags(self, row, d, i, has_own):
        f = bool(d["fired"][i])
        row.update(fired=f, top_last=int(d["top_edit"][i]), n_cand_edits=int(d["n_cand"][i]) if f else None, near=bool(d["near"][i]))
        if has_own:
            row.update(own_fired_last=bool(d["own_cand"][i]), own_released_last=bool(d["own_top"][i]))

    def _tf_memo(self, M, rows, lp, on_miss):
        L, g = self.L, M.obj
        d = self.decide(M, rows, lp)
        out, miss = [None] * len(rows), {}
        for i, j in enumerate(rows):
            e = L.index[j]
            mode = "prompt" if (lp == "prompt" and e.get("lp_shift")) else "tf"
            row = self.base_row(j, lp)
            self._flags(row, d, i, row["kind"] != "probe")
            if not d["fired"][i]:
                row.update(self._pre_outcome(j), src="pre")
                self.stats["pre"] += 1
            else:
                sig = self.vsig(g, int(d["top_edit"][i]))
                o = self.memo.get(e["sid"], mode, sig)
                if o is None:
                    miss[i] = (mode, sig)
                else:
                    row.update(o, src="memo")
                    self.stats["memo"] += 1
            out[i] = row
        self.stats["rows"] += len(rows)
        if miss:
            if self.H is None:
                if on_miss == "raise":
                    raise MemoMiss(f"{M.label}: {len(miss)} of {len(rows)} sequences are not in the memo and no model is loaded")
                for i in miss:
                    out[i]["src"] = "missing"
                self.stats["missing"] += len(miss)
                return out
            self.memo.refresh()                                         # another process may have computed them meanwhile
            todo = []
            for i, (mode, sig) in miss.items():
                o = self.memo.get(L.index[rows[i]]["sid"], mode, sig)
                if o is None:
                    todo.append(i)
                else:
                    out[i].update(o, src="memo")
                    self.stats["memo"] += 1
            for c0 in range(0, len(todo), CHUNK_TF):                    # results reach the disk chunk by chunk
                part = todo[c0:c0 + CHUNK_TF]
                t0 = time.time()
                gg = self.hook_gate(M)
                res = self.ev_plain.tf_rows(gg, [self.item(rows[i]) for i in part], lp=lp)
                self.stats["seconds_model"] += time.time() - t0
                self.stats["forwards"] += len(part)
                if self.verbose:
                    print(f"    [fx.behave] {L.name} {M.label} {lp}: {min(c0 + CHUNK_TF, len(todo))}/{len(todo)} forwards ({time.time() - t0:.0f}s)", flush=True)
                for i, r in zip(part, res):
                    j = rows[i]
                    mode = miss[i][0]
                    o = {k: r[k] for k in OUT_TF if k in r}
                    pred = (True, int(d["top_edit"][i]))
                    got = (bool(r["fired"]), int(r["top_last"]))
                    if got != pred:                                     # must not happen: the hook's decision wins and is recorded
                        self.stats["hook_mismatch"] += 1
                        self.mismatches.append({"method": M.label, "csid": L.sid(j), "lp": lp, "predicted": pred, "hook": got})
                        out[i].update(fired=got[0], top_last=got[1], n_cand_edits=r.get("n_cand_edits") if got[0] else None)
                        if "own_fired_last" in r:
                            out[i].update(own_fired_last=r["own_fired_last"], own_released_last=r["own_released_last"])
                    if got[0]:
                        self.memo.put(L.sid(j), mode, self.vsig(g, got[1]), o, by={"log": L.name, "method": M.label})
                    out[i].update(o, src="run")
                    self.stats["run"] += 1
        return out

    # ------------------------------------------------------------------ dense gates (router, z-score)
    def zmax(self, rows) -> np.ndarray:
        """Largest z-score (rescope DenseGate.zscores: CoachHooK's Z over the positions of the input) of the teacher-forced sequence of
        each row: the one scalar score of a query under the z-score gate.  It depends on the log's weight difference and on the
        sequence, not on a threshold: computed once per log and sequence (one truncated forward) and kept under <memo>/aux/<log>/.
        NaN where the score is not finite."""
        L = self.L
        code = fx.code_fingerprint()["rescope"]

        def read():
            if os.path.exists(self._z_path):
                blob = fx.io.jload(self._z_path)
                if blob.get("rescope") == code and (self.H is None or blob.get("weight_sig") == self.wsig(self.H.WE)):
                    return blob["scores"]
            return {}
        if self._z is None:
            self._z = read()
        miss = [int(j) for j in rows if L.sid(int(j)) not in self._z]
        if miss:
            if self.H is None:
                raise MemoMiss(f"z-scores of {len(miss)} sequences of {L.name} are not cached and no model is loaded")
            from rescope import zscore_gate
            g = zscore_gate(self.H.WE, self.H.W0)
            t0 = time.time()
            with torch.no_grad(), self.ev.hz.HE.vision_cache(self.H):
                for j in miss:
                    self.ev.trim_vcache()
                    x, _ = self.ev.capture(self.item(j))
                    z = float(g.zscores(x).max())
                    self._z[L.sid(j)] = z if np.isfinite(z) else None
            self.stats["seconds_model"] += time.time() - t0
            self.stats["captures"] = self.stats.get("captures", 0) + len(miss)
            for k, v in read().items():                                 # another process may have added sequences meanwhile
                self._z.setdefault(k, v)
            fx.io.jdump({"log": L.name, "rescope": code, "weight_sig": self.wsig(self.H.WE), "scores": self._z,
                         "what": "largest z-score over the positions of the teacher-forced sequence (rescope DenseGate.zscores), by cache sid"}, self._z_path)
        return np.array([np.nan if self._z[L.sid(int(j))] is None else float(self._z[L.sid(int(j))]) for j in rows], dtype=np.float64)

    def decide_dense(self, M, rows, lp: str = "tf") -> dict:
        """Firing of a dense gate on teacher-forced sequences WITHOUT a forward of the whole model: router = activation of the key at
        the last prompt position (key cache; rows inside the decision band are re-decided as the hook computes them, one key at a
        time); z-score = the cached largest z-score of the sequence against the threshold (`near`: within 1e-9 relative, undecided
        here).  -> {'fired', 'near', 'score'} aligned with rows."""
        L, g = self.L, M.obj
        rows = [int(j) for j in rows]
        if g.kind == "router":
            d = L.kc.decide(g, lp, rows=rows)
            fired, near = np.asarray(d["fired"]).copy(), np.asarray(d["near"]).copy()
            self.stats["near"] += int(near.sum())
            for i in np.nonzero(near)[0]:
                key = L.kc.K("tf")[L.kc.key_rows(lp, [rows[int(i)]])[0]]
                f = bool(g.activation(key[None])[0] >= g.thr)
                self.stats["near_changed_on_device"] += int(f != bool(fired[i]))
                fired[i], near[i] = f, False
            return {"rows": rows, "fired": fired, "near": near, "score": np.asarray(d["margin"]) + g.thr}
        sc = self.zmax(rows)
        near = (np.abs(sc - g.thr) <= SLACK_Z * (abs(g.thr) + np.abs(sc))) | np.isnan(sc)
        self.stats["near"] += int(near.sum())
        with np.errstate(invalid="ignore"):
            return {"rows": rows, "fired": sc >= g.thr, "near": near, "score": sc}

    def _dense_row(self, j, lp, fired, near):
        row = self.base_row(j, lp)
        row.update(fired=bool(fired), top_last=-1, n_cand_edits=None, near=bool(near))
        if row["kind"] != "probe":                                      # dense gates do not resolve edits (as the hook's summary)
            row.update(own_fired_last=None, own_released_last=None)
        return row

    def _tf_router(self, M, rows, lp):
        """The router decides at the last prompt position and then uses the deployed output there and at every later position: a
        routed sequence has ONE outcome whatever the threshold (memoised under the signature of the deployed weight), a sequence
        that is not routed is the pre-edit model."""
        L, g = self.L, M.obj
        d = self.decide_dense(M, rows, lp)
        sig = "R" + self.wsig(self.H.WE)
        out, todo = [], []
        for i, j in enumerate(rows):
            row = self._dense_row(j, lp, d["fired"][i], False)
            if not d["fired"][i]:
                row.update(self._pre_outcome(j), src="pre")
                self.stats["pre"] += 1
            else:
                mode = "prompt" if (lp == "prompt" and L.index[j].get("lp_shift")) else "tf"
                o = self.memo.get(L.sid(j), mode, sig)
                if o is None:
                    todo.append((i, mode))
                else:
                    row.update(o, src="memo")
                    self.stats["memo"] += 1
            out.append(row)
        self.stats["rows"] += len(rows)
        if todo and self.memo.refresh():                                # another process may have routed the same sequences meanwhile
            still = []
            for i, mode in todo:
                o = self.memo.get(L.sid(rows[i]), mode, sig)
                if o is None:
                    still.append((i, mode))
                else:
                    out[i].update(o, src="memo")
                    self.stats["memo"] += 1
            todo = still
        for c0 in range(0, len(todo), CHUNK_TF):
            part = todo[c0:c0 + CHUNK_TF]
            t0 = time.time()
            res = self.ev_plain.tf_rows(g, [self.item(rows[i]) for i, _ in part], lp=lp)
            self.stats["seconds_model"] += time.time() - t0
            self.stats["forwards"] += len(part)
            for (i, mode), r in zip(part, res):
                o = {k: r[k] for k in OUT_TF if k in r}
                if not r["fired"]:                                      # must not happen: the hook's decision wins
                    self.stats["hook_mismatch"] += 1
                    self.mismatches.append({"method": M.label, "csid": L.sid(rows[i]), "lp": lp, "predicted": True, "hook": False})
                    out[i]["fired"] = False
                else:
                    self.memo.put(L.sid(rows[i]), mode, sig, o, by={"log": L.name, "method": M.label})
                out[i].update(o, src="run")
                self.stats["run"] += 1
        return out

    def _tf_zscore(self, M, rows, lp):
        """A sequence whose largest z-score is below the threshold has no position at which the gate swaps the deployed output in: it
        is the pre-edit model.  Every other sequence (and every sequence inside the decision band) runs through the library."""
        L, g = self.L, M.obj
        d = self.decide_dense(M, rows, lp)
        out, todo = [], []
        for i, j in enumerate(rows):
            row = self._dense_row(j, lp, d["fired"][i], d["near"][i])
            if not d["fired"][i] and not d["near"][i]:
                row.update(self._pre_outcome(j), src="pre")
                self.stats["pre"] += 1
            else:
                todo.append(i)
            out.append(row)
        self.stats["rows"] += len(rows)
        for c0 in range(0, len(todo), CHUNK_TF):
            part = todo[c0:c0 + CHUNK_TF]
            t0 = time.time()
            res = self.ev_plain.tf_rows(g, [self.item(rows[i]) for i in part], lp=lp)
            self.stats["seconds_model"] += time.time() - t0
            self.stats["forwards"] += len(part)
            for i, r in zip(part, res):
                if bool(r["fired"]) != bool(d["fired"][i]) and not d["near"][i]:
                    self.stats["hook_mismatch"] += 1
                    self.mismatches.append({"method": M.label, "csid": L.sid(rows[i]), "lp": lp, "predicted": bool(d["fired"][i]), "hook": bool(r["fired"])})
                out[i].update({k: r[k] for k in OUT_TF if k in r}, fired=bool(r["fired"]), src="lib")
                self.stats["lib"] += 1
        return out

    def _tf_weight(self, M, rows, lp, on_miss):
        L = self.L
        out = []
        if M.kind == "preedit":                                         # the pre-edit model: the key cache holds its outcome
            for j in rows:
                row = self.base_row(j, lp)
                row.update(self._pre_outcome(j), fired=None, near=None, src="pre")
                out.append(row)
            self.stats["pre"] += len(rows)
            self.stats["rows"] += len(rows)
            return out
        W = None if self.H is None else (self.H.WE if M.kind == "deployed" else M.obj)
        sig = M.rec.get("weight_sig")
        if W is not None:
            sig = self.wsig(W)
            if M.rec.get("weight_sig") not in (None, sig):
                raise RuntimeError(f"{M.label}: the weight in memory is not the weight this method was recorded with")
            M.rec["weight_sig"] = sig
        if sig is None and M.kind == "deployed":
            sig = _deployed_sig(L)
        if sig is None:
            raise MemoMiss(f"{M.label}: the signature of the weight is unknown without a model")
        todo = []
        for i, j in enumerate(rows):
            row = self.base_row(j, lp)
            o = self.memo.get(L.sid(j), "tf", sig) if self.use_memo else None
            if o is None:
                todo.append(i)
            else:
                row.update(o, src="memo")
                self.stats["memo"] += 1
            row.update(fired=None, near=None)
            out.append(row)
        self.stats["rows"] += len(rows)
        if todo:
            if self.H is None:
                if on_miss == "raise":
                    raise MemoMiss(f"{M.label}: {len(todo)} of {len(rows)} sequences are not in the memo and no model is loaded")
                for i in todo:
                    out[i]["src"] = "missing"
                self.stats["missing"] += len(todo)
                return out
            if self.use_memo and self.memo.refresh():                   # another process may have computed them meanwhile
                still = []
                for i in todo:
                    o = self.memo.get(L.sid(rows[i]), "tf", sig)
                    if o is None:
                        still.append(i)
                    else:
                        out[i].update(o, src="memo")
                        self.stats["memo"] += 1
                todo = still
            for c0 in range(0, len(todo), CHUNK_TF):
                part = todo[c0:c0 + CHUNK_TF]
                t0 = time.time()
                res = self.ev_plain.tf_rows(None if M.kind == "deployed" else W, [self.item(rows[i]) for i in part], lp=lp)
                self.stats["seconds_model"] += time.time() - t0
                self.stats["forwards"] += len(part)
                for i, r in zip(part, res):
                    o = {k: r[k] for k in OUT_TF if k in r}
                    if self.use_memo:
                        self.memo.put(L.sid(rows[i]), "tf", sig, o, by={"log": L.name, "method": M.label})
                    out[i].update(o, src="run")
                    self.stats["run"] += 1
            if M.kind == "deployed":
                _deployed_sig(L, sig)
        return out

    # ------------------------------------------------------------------ decodes
    def generate(self, methods, rows, on_miss: str = "raise", verbose: int = 0):
        """Free generation (decoder v2: 10 ids from full forwards + KV-cache continuation to 48 tokens) of sequences `rows`.
        Row fields: y (10-token answer), long (48 tokens), line (its first line), stop, long_stop, n_long, degenerate; edit prompts:
        eq / eq10 / contains / starts against the edit target; probes: agree (y == y0), pre_correct, post_correct, D2; hard
        negatives: agree, leak; gates: fired (in a forward that ran), top_last, n_fire_steps; DUAL: fired (= deployed answer released), dual."""
        many = isinstance(methods, dict)
        ms = methods if many else {"_": methods}
        rows = [int(j) for j in rows]
        if not rows:
            return {lab: [] for lab in ms} if many else []
        out, lib = {}, {}
        for lab, M in ms.items():
            if not M.available:
                raise RuntimeError(f"{lab}: {M.rec.get('unavailable')}")
            if self.use_memo and M.kind == "dual" and self.dual_from_memo:
                out[lab] = self._dual_memo(M, rows, on_miss)
            elif M.kind == "request":                                   # stage 2 (fx.review1): retrieval methods, through the memo path
                from . import review1
                out[lab] = review1.session_generate(self, M, rows, on_miss)
            elif self.use_memo and (M.memo or M.kind in ("deployed", "preedit", "weight")):
                out[lab] = self._dec_memo(M, rows, on_miss)
            else:
                lib[lab] = M
        if lib:
            if self.ev is None:
                raise RuntimeError(f"{list(lib)}: these methods run through the library's Evaluator: a model is needed")
            items = [self.item(j) for j in rows]
            for part in self._lib_parts(lib):
                t0 = time.time()
                res = self.ev.generate({lab: self._lib_obj(M) for lab, M in part.items()}, items, verbose=verbose)
                self.stats["seconds_model"] += time.time() - t0
                for lab, M in part.items():
                    out[lab] = [self._from_lib(j, r, "prompt") for j, r in zip(rows, res[lab])]
                    self.stats["lib"] += len(rows)
                    self.stats["rows"] += len(rows)
                    self.stats["decodes"] += len(rows) * (2 if M.kind == "dual" else 1)
                del res
                self.release()
        return {lab: out[lab] for lab in ms} if many else out["_"]

    def _dual_memo(self, M, rows, on_miss):
        """DUAL from the decodes of the deployed and of the pre-edit model (both memoised, shared with the 'deployed' / 'preedit'
        rows and with every gate that stays silent): the rule is the library's Dual.match, the row is the row of
        rescope.Evaluator._dual_row (test b compares them)."""
        from . import methods as MT
        TB = _tb()
        rE = self._dec_memo(MT.deployed(self.L), rows, on_miss)
        r0 = self._dec_memo(MT.preedit(self.L), rows, on_miss)
        out = []
        for a, b in zip(rE, r0):
            if "line" not in a or "line" not in b:
                out.append(dict(a, src="missing"))
                continue
            m = M.obj.match(TB, a["line"], a["long"], a["y"])
            row = dict(a if m else b)
            own = row.get("edit") if row["kind"] == "edit" else None
            row.update(fired=bool(m), near=None, n_generations=2, n_new_tokens=[a["n_long"], b["n_long"]], src=f"dual({a['src']},{b['src']})",
                       dual={"variant": M.obj.variant, "match": bool(m), "n_matching_edits": len(m), "edits": m[:8], "own": None if own is None else own in m,
                             "source": "deployed" if m else "pre-edit", "deployed_equals_pre_edit": a["long"] == b["long"]})
            out.append(row)
        return out

    def dual_tf(self, M, rows, lp: str = "tf", **kw):
        """Teacher-forced rows of the DUAL COMPOSITION (rescope.Evaluator.dual_tf_rows; NOT part of the baseline's definition): per
        sequence the teacher-forced row of the model that DUAL selects from the two decodes (deployed if its answer matches a
        logged target, else pre-edit) + fired (= the deployed model was selected), dual, n_generations."""
        from . import methods as MT
        rows = [int(j) for j in rows]
        g = self.generate(M, rows, **kw)
        dep, pre = self.tf(MT.deployed(self.L), rows, lp, **kw), self.tf(MT.preedit(self.L), rows, lp, **kw)
        out = []
        for d, a, b in zip(g, dep, pre):
            row = dict(a if d.get("fired") else b)
            row.update(fired=d.get("fired"), dual=d.get("dual"), n_generations=2, n_new_tokens=d.get("n_new_tokens"))
            out.append(row)
        return out

    def _dec_sig(self, g, d, i):
        ks = d["step_fired"].shape[1]
        if not d["step_fired"][i].any():
            return "none"
        parts = [self.vsig(g, int(d["step_top"][i, s])) if d["step_fired"][i, s] else "-" for s in range(ks)]
        return "d:" + hashlib.sha1(",".join(parts).encode()).hexdigest()[:20]

    @staticmethod
    def _dec_flags(row, d, i, o, has_own):
        """Firing flags of a decode from the per-step decisions and the number of full forwards that ran (n_calls_last)."""
        if d is None:
            row.update(fired=None, near=None)
            return
        k = o.get("n_calls_last")
        sf = d["step_fired"][i][:k] if k else d["step_fired"][i]
        f = bool(sf.any())
        top = int(d["step_top"][i][int(np.argmax(sf))]) if f else -1
        row.update(fired=f, top_last=top, n_fire_steps=int(sf.sum()), near=bool(d["step_near"][i].any()))
        if has_own:
            row.update(own_released_last=bool(f and top == int(d["own"][i])))

    def _dec_memo(self, M, rows, on_miss):
        L = self.L
        gate = M.memo
        d = self.decide(M, rows, "steps") if gate else None
        if gate:
            sigs = [self._dec_sig(M.obj, d, i) for i in range(len(rows))]
        elif M.kind == "preedit":
            sigs = ["none"] * len(rows)
        else:
            W = None if self.H is None else (self.H.WE if M.kind == "deployed" else M.obj)
            s = self.wsig(W) if W is not None else (M.rec.get("weight_sig") or (_deployed_sig(L) if M.kind == "deployed" else None))
            if s is None:
                raise MemoMiss(f"{M.label}: the signature of the weight is unknown without a model")
            M.rec["weight_sig"] = s
            sigs = [s] * len(rows)
        out, todo = [], []
        for i, j in enumerate(rows):
            row = self.base_row(j, "prompt")
            o = self.memo.get(L.sid(j), "decode", sigs[i])
            if o is None:
                todo.append(i)
            else:
                row.update(o, src="memo")
                self._dec_flags(row, d, i, o, row["kind"] != "probe")
                self.stats["memo"] += 1
            out.append(row)
        self.stats["rows"] += len(rows)
        if not todo:
            return out
        if self.H is None:
            if on_miss == "raise":
                raise MemoMiss(f"{M.label}: {len(todo)} of {len(rows)} decodes are not in the memo and no model is loaded")
            for i in todo:
                out[i]["src"] = "missing"
            self.stats["missing"] += len(todo)
            return out
        self.memo.refresh()
        t0 = time.time()
        mism = {}                                                       # decodes in which the hook decided otherwise than predicted
        groups = {"none": [i for i in todo if sigs[i] == "none"], "own": [i for i in todo if sigs[i] != "none"]}
        for kind, idx in groups.items():
            idx = [i for i in idx if not self.memo.has(L.sid(rows[i]), "decode", sigs[i])] if idx else idx
            if not idx:
                continue
            if kind == "none":
                obj = self.H.W0                                         # nothing is released in any forward: the pre-edit model
            elif gate:
                obj = self.hook_gate(M)
            else:
                obj = None if M.kind == "deployed" else M.obj
            res = []
            for c0 in range(0, len(idx), CHUNK_DEC):
                res += self.ev_plain.generate(obj, [self.item(rows[i]) for i in idx[c0:c0 + CHUNK_DEC]])
                self.stats["decodes"] += len(idx[c0:c0 + CHUNK_DEC])
                if not (gate and kind == "own"):                        # weights: store chunk by chunk (gates: after the hook check below)
                    for i, r in zip(idx[c0:c0 + CHUNK_DEC], res[c0:]):
                        self.memo.put(L.sid(rows[i]), "decode", sigs[i], {k: r[k] for k in OUT_DEC if k in r}, by={"log": L.name, "method": M.label})
            for i, r in zip(idx, res):
                o = {k: r[k] for k in OUT_DEC if k in r}
                if gate and kind == "own":                              # the hook's decisions against the predicted steps
                    k = int(r["n_calls_last"])
                    sf = d["step_fired"][i][:k]
                    pred = (bool(sf.any()), int(d["step_top"][i][int(np.argmax(sf))]) if sf.any() else -1, int(sf.sum()))
                    got = (bool(r["fired"]), int(r["top_last"]), int(r["n_fire_calls_last"]))
                    if got != pred:
                        self.stats["hook_mismatch"] += 1
                        self.mismatches.append({"method": M.label, "csid": L.sid(rows[i]), "lp": "decode", "predicted": pred, "hook": got})
                        mism[i] = (o, got)
                        continue                                        # not stored (the signature names the PREDICTED releases)
                self.memo.put(L.sid(rows[i]), "decode", sigs[i], o, by={"log": L.name, "method": M.label})
        self.stats["seconds_model"] += time.time() - t0
        for i in todo:
            if i in mism:                                               # must not happen: the row is the hook's own run and decision
                o, got = mism[i]
                out[i].update(o, src="run", fired=got[0], top_last=got[1], n_fire_steps=got[2], near=bool(d["step_near"][i].any()))
                if out[i]["kind"] != "probe":
                    out[i]["own_released_last"] = bool(got[0] and got[1] == int(d["own"][i]))
                self.stats["run"] += 1
                continue
            o = self.memo.get(L.sid(rows[i]), "decode", sigs[i])
            if o is None:
                raise RuntimeError(f"{M.label}: the decode of {L.sid(rows[i])} ran but its outcome is not in the memo")
            out[i].update(o, src="run")
            self._dec_flags(out[i], d, i, o, out[i]["kind"] != "probe")
            self.stats["run"] += 1
        if M.kind == "deployed":
            _deployed_sig(L, sigs[0])
        return out

    # ------------------------------------------------------------------ one function per item type
    def edits(self, methods, metrics=("rel", "tgen", "igen"), edits=None, lp: str = "tf", **kw):
        """Rel / T-Gen / I-Gen of the logged edits (optional subset `edits`), teacher forced at B = 1: exact, token_acc + firing."""
        rows = [j for m in metrics for j in self.L.edits_of(m, edits)]
        return self.tf(methods, rows, lp, **kw)

    def probes(self, methods, pool: str = "heldout_disjoint", lp: str = "tf", **kw):
        """Valid probe rows of a pool: D1 (the pre-edit answer is no longer reproduced), firing, tok_loc (token-level locality)."""
        return self.tf(methods, self.L.pool(pool), lp, **kw)

    def hard_negatives(self, methods, lp: str = "tf", types=("hn_i", "hn_q", "hn_x"), rows=None, generate: bool = False, gen_rows=None, decode_all: bool = False,
                       **kw):
        """Valid hard negatives of the log's edits (HN-I, HN-Q, HN-X): D1 and firing (teacher forced).  generate=True returns
        (tf rows, generation rows): the generation rows carry wrong_target = the generated answer (first line, normalised) equals
        the target of the edit that fires (gates; rows on which no edit fires in the decode are False without a decode), eq_own /
        eq_b (the target of the row's own edit / of edit b for HN-X) and eq_any_edit.  gen_rows: the sequences to decode for the
        methods that are not memoised gates (default: all valid rows; two decodes per row for DUAL).  DUAL: wrong_target = the
        deployed answer was released (it equals a logged target by the rule); methods without firing flags (weights, corrections):
        wrong_target is None, eq_own / eq_any_edit are the comparable quantities.  decode_all=True: memoised gates decode EVERY row
        (a row on which nothing fires is the pre-edit model's decode, memoised once per backbone and shared by all gates and logs),
        so that eq_own / eq_any_edit / leak are defined on the same rows for every method."""
        rows = self.L.rows(types, valid=True) if rows is None else [int(j) for j in rows]
        many = isinstance(methods, dict)
        ms = methods if many else {"_": methods}
        tf = {lab: self.tf(M, rows, lp, **kw) for lab, M in ms.items() if M.kind != "dual"}
        if not generate:
            return tf if many else tf["_"]
        gen = {}
        for lab, M in ms.items():
            sub = rows if gen_rows is None else [int(j) for j in gen_rows]
            if self.use_memo and (M.memo or M.kind == "request") and not decode_all:             # only the rows on which the gate fires in a decode need one
                d = self.decide(M, sub, "steps")
                fire = [j for j, f in zip(sub, d["fired"]) if f]
                got = {r["csid"]: r for r in self.generate(M, fire, **kw)}
                g = []
                for j in sub:
                    r = got.get(self.L.sid(j))
                    if r is None:
                        r = self.base_row(j, "prompt")
                        r.update(fired=False, top_last=-1, src="pre", decoded=False)
                    g.append(r)
            else:
                g = self.generate(M, sub, **kw)
            gen[lab] = [self.wrong_target(r) for r in g]
        return (tf, gen) if many else (tf.get("_"), gen["_"])

    def wrong_target(self, r: dict) -> dict:
        """Add the wrong-target flags to a generation row of a hard negative (testbed normalisation tb_common.norm_answer)."""
        if self._norm is None:
            TB = _tb()
            nt = [TB.norm_answer(t) for t in self.L.targets]
            by = {}
            for i, t in enumerate(nt):
                if t:
                    by.setdefault(t, []).append(i)
            self._norm = (TB.norm_answer, nt, by)
        norm, nt, by = self._norm
        if "line" not in r:                                             # no decode: nothing fired, the answer is the pre-edit answer
            r.update(wrong_target=False if r.get("fired") is False else None, eq_own=None, eq_any_edit=None)
            return r
        nl = norm(r["line"])
        m = by.get(nl, []) if nl else []
        top = r.get("top_last", -1)
        r.update(eq_own=bool(nl) and r.get("edit", -1) >= 0 and nt[r["edit"]] == nl, eq_any_edit=bool(m), n_matching_edits=len(m))
        if "edit_b" in r:
            r["eq_b"] = bool(nl) and nt[r["edit_b"]] == nl
        if "dual" in r:                                                 # DUAL releases the deployed answer iff it equals a logged target
            r["wrong_target"] = bool(r["dual"]["match"])
        else:                                                           # gates: the answer equals the target of the edit that fires
            r["wrong_target"] = (bool(nl) and top is not None and top >= 0 and nt[top] == nl) if r.get("fired") is not None and "top_last" in r else None
        return r

    def assert_clean(self) -> dict:
        """The two assertions every job makes at its end (rules of the driver's review): every forward agreed with the predicted
        firing (stats['hook_mismatch'] == 0) and the memo holds no key with two outcomes (memo.stats['conflicts'] == 0).
        -> {'hook_mismatch': 0, 'memo_conflicts': 0, 'forwards', 'rows'} for the job's record; AssertionError otherwise."""
        hm, mc = int(self.stats["hook_mismatch"]), int(self.memo.stats.get("conflicts", 0)) if self.memo is not None else 0
        assert hm == 0, f"{self.L.name}: {hm} forwards disagreed with the predicted firing: {self.mismatches[:3]}"
        assert mc == 0, f"{self.L.name}: {mc} memo keys with two outcomes: {self.memo.conflicts[:3]}"
        return {"hook_mismatch": hm, "memo_conflicts": mc, "forwards": int(self.stats["forwards"]), "rows": int(self.stats["rows"])}

    # ------------------------------------------------------------------ logit hashes and released edits (E9), blocklist (R11 h)
    def tf_hashed(self, M, rows, lp: str = "tf", verbose: int = 0) -> list:
        """Teacher-forced rows WITH a hash of the logits and the set of edits released (E9: revocation, frame rule, lineage).
        EVERY sequence runs one forward through the library's Evaluator under the method (gate through the hook, replacement
        weight, deployed / pre-edit weight, correction): nothing is read from or written to the memo and no row is taken from the
        key cache, so that "the logits are bit for bit those of ..." is a statement about forwards that ran.
        Row = the usual teacher-forced row (exact, token_acc, D1, tok_loc, firing flags) plus
          logit_sha1   sha1 of the logits at the answer positions that the row scores ((n_tok, vocabulary), the model's dtype,
                       hashed as float32, which holds bf16 / fp16 exactly), 20 hex digits
          released     sorted list of the edits released in the forward: fire 'top1' = the releasing edit of every firing
                       position (role 'last': at most one edit); fire 'all' = every candidate edit; [] if nothing fired;
                       None for methods that are not gates over edits (weights, corrections, dense gates)
          src          'hash'
        Two rows with the same logit_sha1 had the same logits (up to a sha1 collision); compare the hashes of two methods on the
        same sequences, or with the pre-edit model (fx.methods.preedit(L)) for "equals the pre-edit model bit for bit"."""
        if self.H is None:
            raise RuntimeError("tf_hashed runs every forward: a model is needed (H = fx.logs.load_model(name))")
        if not M.available:
            raise RuntimeError(f"{M.label}: {M.rec.get('unavailable')}")
        if M.kind == "dual":
            raise TypeError("DUAL has no teacher-forced forward")
        if M.kind == "request":                                         # stage 2 (fx.review1)
            raise TypeError("tf_hashed is not defined for a retrieval method (its silent sequences are the pre-edit model without a forward)")
        rows = [int(j) for j in rows]
        scoped = hasattr(M.obj, "det_tau")
        if M.kind == "deployed":
            obj = None
        elif M.kind == "preedit":
            obj = self.H.W0
        elif scoped:
            obj = self.hook_gate(M) if M.memo else self.ev_plain._on_device(self._lib_obj(M))
        else:
            obj = M.obj
        ev, out = self.ev_plain, []
        t0 = time.time()
        with torch.no_grad(), ev.hz.HE.vision_cache(self.H):
            for n, j in enumerate(rows):
                ev.trim_vcache()
                it = self.item(j)
                b = ev.batch(it)
                pre = ev.pre_ids(it, b)
                with ev.applied(obj) as hk:
                    r, lg = ev.tf(it, b, logits=True, lp=lp, pre_ids=pre)
                    rel = None
                    if scoped:
                        f = hk.last
                        if f is None:
                            rel = []
                        elif obj.fire == "top1":
                            rel = sorted({int(e) for e in f.edit.tolist() if int(e) >= 0})
                        else:
                            rel = [int(e) for e in obj._edit_any(f.cand).any(0).nonzero().flatten().tolist()]
                row = self._from_lib(j, r, lp, src="hash")
                row["logit_sha1"] = hashlib.sha1(np.ascontiguousarray(lg.detach().float().cpu().numpy()).tobytes()).hexdigest()[:20]
                row["released"] = rel
                out.append(row)
                if verbose and (n + 1) % verbose == 0:
                    print(f"    [fx.behave] {self.L.name} {M.label} hashed {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        self.stats["forwards"] += len(rows)
        self.stats["rows"] += len(rows)
        self.stats["hashed"] = self.stats.get("hashed", 0) + len(rows)
        self.stats["seconds_model"] += time.time() - t0
        if scoped and M.memo:                                           # the hook's decision against the CPU prediction, as on the memo path
            d = self.decide(M, rows, lp)
            for i, row in enumerate(out):
                if (bool(row["fired"]), int(row["top_last"])) != (bool(d["fired"][i]), int(d["top_edit"][i])):
                    self.stats["hook_mismatch"] += 1
                    self.mismatches.append({"method": M.label, "csid": row["csid"], "lp": lp, "predicted": (bool(d["fired"][i]), int(d["top_edit"][i])),
                                            "hook": (bool(row["fired"]), int(row["top_last"]))})
        return out

    def blocklist(self, reported, rows, lp: str = "tf", **kw) -> list:
        """The BLOCKLIST line of the evidence curves (R11 h): the deployed model plus an exact-match list of the reported queries,
        which are answered by the pre-edit model.  reported: sequences (the reported complaints); rows: the sequences to
        evaluate.  A sequence whose QUERY (query_id: same question string and image content) is a reported query takes the
        pre-edit model's row, every other sequence the deployed log's row; nothing is assumed: both rows are the driver's
        own (deployed: forwards / memo; pre-edit: the key cache).  Row field blocked (= the query is on the list) and
        fired = blocked."""
        from . import methods as MT
        rows = [int(j) for j in rows]
        listed = {self.L.index[int(j)]["query_id"] for j in reported}
        dep = self.tf(MT.deployed(self.L), rows, lp, **kw)
        pre = self.tf(MT.preedit(self.L), rows, lp, **kw)
        out = []
        for j, a, b in zip(rows, dep, pre):
            hit = self.L.index[j]["query_id"] in listed
            r = dict(b if hit else a)
            r.update(blocked=hit, fired=hit, src=f"blocklist({'pre' if hit else a.get('src')})")
            out.append(r)
        return out

    def d2(self, methods, pool: str = "heldout_disjoint", rows=None, **kw):
        """48-token decode of the pre-correct probe rows of a pool (valid pre-edit answer or not): D2 = the 48-token answer no
        longer contains the reference."""
        rows = self.L.rows("probe", pool=pool, where=lambda e: bool(e.get("pre_correct_long"))) if rows is None else rows
        return self.generate(methods, rows, **kw)


_DSIG = {}


def _deployed_sig(L, sig=None):
    """Signature of a log's deployed weight, remembered under <memo>/aux/<log>/deployed_sig.json once a process with the model has
    computed it (so that later processes can read the deployed log's memoised rows without a model)."""
    import os
    p = os.path.join(fx.MEMO_ROOT, "aux", L.name, "deployed_sig.json")
    if sig is not None:
        if _DSIG.get(L.name) != sig:
            _DSIG[L.name] = sig
            if not os.path.exists(p) or fx.io.jload(p).get("sig") != sig:
                fx.io.jdump({"log": L.name, "sig": sig, "what": "sha1 signature of the deployed module weight W_E (fx.memo.weight_sig)"}, p)
        return sig
    if L.name not in _DSIG and os.path.exists(p):
        _DSIG[L.name] = fx.io.jload(p)["sig"]
    return _DSIG.get(L.name)
