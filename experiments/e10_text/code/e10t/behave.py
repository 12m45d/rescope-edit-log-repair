"""Behavioural evaluation on text logs: the metrics (i), (iii), (vi) of the E10 amendment (the public scores (ii) and the
n-gram entropy (v) are in e10t.public).

    ses = e10t.behave.Session(L, B)                      # L = e10t.open_log(name), B = e10t.Backbone() (None: store / cache only);
                                                         # store=False: no on-disk store; fresh=True: earlier outcomes are not read
    e = ses.edits(M)                                     # request + paraphrase prompts of the logged records
    p = ses.probes(M, "heldout")                         # HELD-OUT benign prompts;  ses.probes(M, "neighborhood")
    e10t.behave.edit_stats(e), e10t.behave.probe_stats(p)

ROWS (one per prompt row; plain dicts)
  tf_tok / dec_tok     the 10 argmax tokens under teacher forcing of the pre-edit continuation y0 / of the greedy decode
  D1_tf, D1_dec        y0 is not reproduced (or a logit is not finite): teacher-forced / decode variant;  tok_loc = share of y0 kept
  fired, n_fire_pos, rel_edits, n_cand_edits, own_cand, own_released, other_released   firing flags of the hook (teacher-forced
                       pass; fired_dec ... of the decode); None for weights.  own = the edit of the row's record.  rel_edits is a
                       list, or a bit mask {'hex': ...} if it has more than 64 members (e10t.backbone.edit_in reads both)
  fired_last_only      the only release is at the tenth continuation token (it cannot change the ten measured tokens; top-1 gates)
  near                 the cached decision lies inside the decision band (the row ran through the hook, which decided)
  src                  'silent' (nothing fires: the cached pre-edit continuation, no forward) | 'run' | 'store'
  request / paraphrase rows: new_exact, new_tok (greedy decode against the tokens of the new object, multi-token), true_exact
METRICS  (i)   edit_stats(rows): exact and token-level match of the new object, request and paraphrase prompts (greedy decode)
         (iii) probe_stats(rows): D1 (teacher-forced and decode), token locality, firing; clean / near-scope prompts apart
         (vi)  retention: edit_stats of ses.edits(M, first=100) on every log of a stream (the first 100 requests as T grows)

SPARSE EVALUATION.  The module input does not depend on the edited weight, so for a gate at all positions the cached keys of
[prompt, y0] decide what the hook will do on the teacher-forced pass AND on the decode as long as it follows y0: a prompt on
which nothing fires (and that is not inside the decision band) is served bit for bit by the pre-edit model in both, and its
row is the cached pre-edit continuation, with no forward.  Every other prompt runs through the library's hook in padded
batches; the hook's own firing flags are recorded and compared with the cached decision (stats['hook_mismatch'] must be 0).
stats['hook_mismatch']: teacher-forced passes whose firing differs from the cached decision outside the decision band; the keys of
that pass ARE the cached keys up to round-off, so it must be 0 (assert it).  stats['hook_mismatch_dec']: decodes whose firing differs
from the cached decision; possible only if the decode left the cached continuation before the predicted release (a near-tie token in
another batch layout); it is reported, not asserted (0 on the development log).
Weights (deployed, pre-edit, replacement weights) run every prompt.  Outcomes are kept in an on-disk store
(e10_text/store/<backbone>/<mode>__<signature>/shard_*.jsonl; signature = the gate's identity or the weight's sha1), written
by this code version only, so a restarted job skips what was computed.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time

import numpy as np
import torch

import e10t
from . import gate as G, methods as MT
from .backbone import edit_in

MODES = ("tf", "dec")


def _store_code() -> str:
    h = hashlib.sha1(f"{e10t.N_DEC}|{e10t.FEED_LAST}|{e10t.BAND}".encode())
    d = os.path.dirname(os.path.abspath(__file__))
    for f in ("backbone.py", "behave.py", "gate.py"):
        with open(os.path.join(d, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


class Store:
    """Append-only store of outcomes per namespace (a directory of per-process shards).  Only shards written by the same
    code (sha1 of backbone.py, behave.py, gate.py and the constants) are read.  fresh=True: earlier shards are not read
    (everything is computed again in this process and written as new shards)."""

    def __init__(self, root: str | None = None, enabled: bool = True, fresh: bool = False):
        self.root = os.path.join(root or e10t.STORE_ROOT, e10t.BACKBONE)
        self.enabled, self.fresh, self.code = enabled, fresh, _store_code()
        self.data, self._fh = {}, {}
        self.stats = {"hits": 0, "writes": 0, "shards_other_code": 0}

    def _load(self, ns):
        if ns in self.data:
            return self.data[ns]
        d, out = os.path.join(self.root, ns), {}
        if self.enabled and not self.fresh and os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if not f.endswith(".jsonl"):
                    continue
                ok = False
                with open(os.path.join(d, f)) as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue                                    # a torn last line
                        if "header" in rec:
                            ok = rec["header"].get("code") == self.code
                            self.stats["shards_other_code"] += int(not ok)
                        elif ok:
                            out.setdefault(int(rec["t"]), rec["o"])
        self.data[ns] = out
        return out

    def get(self, ns, tid):
        o = self._load(ns).get(int(tid))
        self.stats["hits"] += int(o is not None)
        return o

    def put(self, ns, tid, outcome):
        self._load(ns)[int(tid)] = outcome
        if not self.enabled:
            return
        if ns not in self._fh:
            d = os.path.join(self.root, ns)
            os.makedirs(d, exist_ok=True)
            self._fh[ns] = open(os.path.join(d, f"shard_{socket.gethostname()}_{os.getpid()}_{int(time.time() * 1000)}.jsonl"), "a", buffering=1)
            self._fh[ns].write(json.dumps({"header": {"code": self.code, "e10t": e10t.VERSION, "ns": ns, "started": time.strftime("%Y-%m-%d %H:%M:%S")}}) + "\n")
        self._fh[ns].write(json.dumps({"t": int(tid), "o": outcome}) + "\n")
        self.stats["writes"] += 1


class Session:
    def __init__(self, L, B=None, store: bool = True, store_root: str | None = None, device: str = "cpu", fresh: bool = False):
        self.L, self.B, self.device = L, B, device
        self.store = Store(store_root, enabled=store, fresh=fresh)
        self.stats = {"sequences_run": 0, "batches": 0, "silent": 0, "from_store": 0, "near": 0, "hook_mismatch": 0, "hook_mismatch_dec": 0,
                      "seconds_model": 0.0, "mismatches": []}
        self._wsig = {}

    # ------------------------------------------------------------------ what a method is on the model
    def served_obj(self, M):
        """The object Backbone.served takes for a Method: None (pre-edit), the deployed / replacement weight, or the gate."""
        if M.kind == "preedit":
            return None
        if M.kind == "deployed":
            return self.L.weights()[1]
        return M.obj

    def sig(self, M) -> str:
        """Store signature of a method, computed from the OBJECT that is served (never from its label or record):
        'G' + identity of the gate (detectors, thresholds, ranking, values, role / fire / response), or 'W' + sha1 of the weight."""
        if M.kind == "gate":
            return "G" + MT.gate_sig(M.obj)
        if M.kind in ("preedit", "deployed"):
            if M.kind not in self._wsig:
                W = self.L.weights()[0 if M.kind == "preedit" else 1]
                self._wsig[M.kind] = "W" + hashlib.sha1(W.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
            return self._wsig[M.kind]
        if not torch.is_tensor(M.obj):
            raise TypeError(f"{M.label}: a method is a gate, 'preedit', 'deployed' or a replacement weight")
        return "W" + hashlib.sha1(M.obj.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()).hexdigest()[:16]

    # ------------------------------------------------------------------ outcomes per prompt text
    def outcomes(self, M, tids, mode: str = "tf", force_run: bool = False) -> dict:
        """{text_id: outcome} of a method on prompt texts: mode 'tf' (teacher forcing of y0) | 'dec' (greedy decode).
        outcome: tok (10 ids), fin, src, and for gates fired, n_fire_pos, rel, rel_edits, n_cand_edits, cand_edits, near.
        force_run: no sparse shortcut and no store (every prompt through the model; tests)."""
        if mode not in MODES:
            raise ValueError(f"mode: one of {MODES}")
        L = self.L
        tids = list(dict.fromkeys(int(t) for t in tids))
        ns = f"{mode}__{self.sig(M)}"
        out, todo = {}, []
        for t in tids:
            o = None if force_run else self.store.get(ns, t)
            if o is None:
                todo.append(t)
            else:
                out[t] = dict(o, src="store")
                self.stats["from_store"] += 1
        if not todo:
            return out
        pred = None
        run = todo
        if M.kind == "gate":
            d = G.decide(L, M.obj, todo, device=self.device, with_positions=False)
            pred = {t: (bool(f), bool(nr)) for t, f, nr in zip(todo, d["fired"], d["near"])}
            if not force_run:
                run = []
                for t in todo:
                    if pred[t][0] or pred[t][1]:
                        run.append(t)
                    else:                                               # nothing fires: bit for bit the pre-edit model
                        out[t] = {"tok": L.y0(t), "fin": True, "fired": False, "n_fire_pos": 0, "rel": [], "rel_edits": [], "n_rel_edits": 0,
                                  "n_cand_edits": 0, "cand_edits": [], "near": False, "src": "silent"}
                        self.stats["silent"] += 1
        if run:
            if self.B is None:
                raise RuntimeError(f"{len(run)} prompts of {M.label} need the model: pass a Backbone to the Session")
            B = self.B
            lens = [len(L.ids(t)) for t in run]
            t0 = time.time()
            with B.served(self.served_obj(M)) as srv:
                for b in B.batches(lens):
                    bt = [run[i] for i in b]
                    seqs = [L.ids(t) for t in bt]
                    if mode == "tf":
                        tok, fin = B.tf(seqs, [L.y0(t) for t in bt])
                    else:
                        tok, fin = B.decode(seqs)
                    flags = srv.rows()
                    self.stats["batches"] += 1
                    for i, t in enumerate(bt):
                        o = {"tok": tok[i].tolist(), "fin": bool(fin[i])}
                        if flags is not None:
                            f = flags[i]
                            o.update(fired=f["fired"], n_fire_pos=f["n_fire_pos"], rel=f["rel"], rel_edits=f["rel_edits"], n_rel_edits=f["n_rel_edits"],
                                     n_cand_edits=f["n_cand_edits"], cand_edits=f["cand_edits"], near=pred[t][1])
                            self.stats["near"] += int(pred[t][1])
                            if not pred[t][1] and f["fired"] != pred[t][0]:
                                self.stats["hook_mismatch" if mode == "tf" else "hook_mismatch_dec"] += 1
                                if len(self.stats["mismatches"]) < 50:
                                    self.stats["mismatches"].append({"text_id": t, "mode": mode, "method": M.label, "cache": pred[t][0], "hook": f["fired"]})
                        else:
                            o["fired"] = None
                        if not force_run:
                            self.store.put(ns, t, o)
                        out[t] = dict(o, src="run")
            self.stats["sequences_run"] += len(run)
            self.stats["seconds_model"] += time.time() - t0
        return out

    # ------------------------------------------------------------------ rows
    def rows(self, M, prompt_rows, modes=MODES, force_run: bool = False) -> list:
        """One row per prompt row (see the module docstring)."""
        L = self.L
        tids = [r["text_id"] for r in prompt_rows]
        res = {m: self.outcomes(M, tids, m, force_run) for m in modes}
        out = []
        for r in prompt_rows:
            t = r["text_id"]
            y0 = L.y0(t)
            own = r["pos"] if r["set"] in ("dev", "stream") else None
            row = {k: r[k] for k in ("prompt_id", "text_id", "set", "kind", "pos", "k", "case_id", "record") if k in r}
            row.update({k: r[k] for k in ("clean", "near_subject", "near_mention", "near_text") if k in r})
            row.update(method=M.label, own=own, valid=L.valid(t))
            for m in modes:
                o = res[m][t]
                same = [a == b for a, b in zip(o["tok"], y0)]
                sfx = "" if m == "tf" else "_dec"
                if m == "tf":
                    row.update(tf_tok=o["tok"], D1_tf=(not o["fin"]) or not all(same), tok_loc=float(np.mean(same)) if o["fin"] else 0.0,
                               finite_tf=o["fin"], src=o["src"])
                else:
                    row.update(dec_tok=o["tok"], D1_dec=(not o["fin"]) or not all(same), dec_loc=float(np.mean(same)) if o["fin"] else 0.0,
                               finite_dec=o["fin"], src_dec=o["src"])
                row["fired" + sfx] = o.get("fired")
                if o.get("fired") is not None:
                    re_, ce = o.get("rel_edits"), o.get("cand_edits")
                    last = len(L.ids(t)) + e10t.N_DEC - 1                # position of the tenth continuation token
                    row.update({"n_fire_pos" + sfx: o["n_fire_pos"], "rel_edits" + sfx: re_, "n_cand_edits" + sfx: o["n_cand_edits"],
                                "near" + sfx: o.get("near", False),
                                "fired_last_only" + sfx: bool(o["fired"]) and o.get("rel") is not None and all(q == last for q, _ in o["rel"])})
                    if own is not None:
                        has = edit_in(re_, own)
                        row.update({"own_cand" + sfx: edit_in(ce, own), "own_released" + sfx: has,
                                    "other_released" + sfx: o["n_rel_edits"] - int(has) > 0})
            if r["kind"] in ("request", "paraphrase") and own is not None and "dec" in modes:
                d = res["dec"][t]
                for which in ("new", "true"):
                    tgt = L.target_ids(own, which)
                    hit = [a == b for a, b in zip(d["tok"], tgt)]
                    ok = d["fin"] and len(tgt) <= e10t.N_DEC
                    row[f"{which}_exact"] = bool(ok and all(hit))
                    row[f"{which}_tok"] = float(np.mean(hit + [False] * (len(tgt) - len(hit)))) if d["fin"] else 0.0
                row["n_target_tok"] = len(L.target_ids(own, "new"))
            out.append(row)
        return out

    def edits(self, M, kinds=("request", "paraphrase"), first: int | None = None, modes=MODES, **kw) -> list:
        """Metric (i): request and paraphrase prompts of the logged records (first: only the first `first` requests of the log,
        e.g. 100 for the retention read-out (vi))."""
        pr = [r for k in kinds for r in self.L.rows[k] if first is None or r["pos"] < first]
        return self.rows(M, pr, modes, **kw)

    def probes(self, M, part: str = "heldout", first: int | None = None, modes=MODES, **kw) -> list:
        """Metric (iii): D1 rows of a part: 'heldout' | 'repair' (benign prompts) | 'neighborhood' | 'request' | 'paraphrase'."""
        pr = [r for r in G.part_rows(self.L, part) if first is None or r["set"] not in ("dev", "stream") or r["pos"] < first]
        return self.rows(M, pr, modes, **kw)


# ----------------------------------------------------------------------------- numbers from rows
def _stat(values, units, ci, scale=100.0):
    import fx
    v = [None if x is None else float(x) for x in values]
    if not v or all(x is None for x in v):
        return {"value": None, "n": len(v)}
    if not ci:
        vv = [x for x in v if x is not None]
        return {"value": float(scale * np.mean(vv)), "n": len(vv)}
    b = fx.stats.bootstrap(v, units, scale=scale)
    return {"value": b["value"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"], "n": b["n_rows"], "n_units": b["n_units"]}


def edit_stats(rows, ci: bool = False) -> dict:
    """Metric (i) per kind (request, paraphrase): exact and token-level match of the new object in the greedy decode (percent;
    bootstrap unit = edit), share whose decode starts with the TRUE object, firing and own-edit flags of the decode."""
    out = {}
    for kind in ("request", "paraphrase"):
        sub = [r for r in rows if r["kind"] == kind and "new_exact" in r]
        if not sub:
            continue
        u = [r["pos"] for r in sub]
        gate = sub[0].get("fired_dec") is not None
        out[kind] = {"n": len(sub), "n_edits": len(set(u)), "exact": _stat([r["new_exact"] for r in sub], u, ci), "token": _stat([r["new_tok"] for r in sub], u, ci),
                     "true_exact": _stat([r["true_exact"] for r in sub], u, False), "n_nonfinite": sum(not r["finite_dec"] for r in sub),
                     "n_multi_token_objects": sum(r["n_target_tok"] > 1 for r in sub),
                     "fired": _stat([r["fired_dec"] for r in sub], u, False) if gate else None,
                     "own_released": _stat([r.get("own_released_dec") for r in sub], u, ci) if gate else None,
                     "other_released": _stat([r.get("other_released_dec") for r in sub], u, False) if gate else None,
                     "exact_given_own_released": _stat([r["new_exact"] for r in sub if r.get("own_released_dec")], None, False) if gate else None}
    return out


def probe_stats(rows, ci: bool = False) -> dict:
    """Metric (iii) of ONE part (valid rows): D1 teacher-forced / decode, token locality, firing (percent; bootstrap unit =
    record), the same for clean and near-scope prompts, D1 among firing rows, rows whose two D1 variants differ."""
    def one(sub):
        if not sub:
            return {"n": 0}
        u = [r.get("case_id", i) for i, r in enumerate(sub)]
        gate = sub[0].get("fired") is not None
        fired = [r for r in sub if r.get("fired")]
        o = {"n": len(sub), "D1_tf": _stat([r["D1_tf"] for r in sub], u, ci), "tok_loc": _stat([r["tok_loc"] for r in sub], u, ci),
             "n_nonfinite": sum(not r["finite_tf"] for r in sub), "fired": _stat([r["fired"] for r in sub], u, ci) if gate else None,
             "n_fired": len(fired) if gate else None, "D1_tf_given_fired": _stat([r["D1_tf"] for r in fired], None, False) if gate and fired else None,
             "n_fired_last_only": sum(bool(r.get("fired_last_only")) for r in sub) if gate else None,
             "n_from_cache": sum(r["src"] == "silent" for r in sub)}
        if "D1_dec" in sub[0]:
            o.update(D1_dec=_stat([r["D1_dec"] for r in sub], u, ci), n_D1_tf_differs_from_dec=sum(r["D1_tf"] != r["D1_dec"] for r in sub))
        if sub[0].get("own") is not None and gate:
            o.update(own_released=_stat([r.get("own_released") for r in sub], u, False), other_released=_stat([r.get("other_released") for r in sub], u, False))
        return o
    v = [r for r in rows if r["valid"]]
    out = one(v)
    out["n_invalid"] = len(rows) - len(v)
    if v and "clean" in v[0]:
        out["clean"], out["near_scope"] = one([r for r in v if r["clean"]]), one([r for r in v if not r["clean"]])
    return out
