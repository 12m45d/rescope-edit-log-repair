"""E10: construction of every row that the adapter does not build itself, on top of the adapter (e10t) and the library.

  R3 gates on a text log (amendment R14): cos_centred, cos_null, white(kappa), whitecos(kappa); detectors by the feature layer
     10_final/feat2/white (white.scores.feature_gate, role 'all', rebased values), statistics (mean key, uncentred second moment)
     from the keys at ALL cached positions of 500 benign REPAIR-pool prompts that are in none of the 20 calibration draws
     (stat_prompts); cos_null: the library's null repair on the keys (all positions) of the 100 prompts that the adapter's
     rescope_null would use in the same draw (e10t.data.null_sample).  Guards and calibration exactly as the adapter's cosine row:
     add_guards(stored solver keys + keys as served), e10t.methods.calibrate (two-block, guard_margin).
  guards off (R21 b) = the adapter's allow_silence=True.
  lookup_subject (R22 e): a keyed store that needs the REQUESTS (another deployment class): an edit is released iff the prompt
     contains its subject string, at the last token of the occurrence, with the migration's (rebased) value; the latest edit wins
     a position.  LookupGate reads the token ids of each model call (capture hook), so that the public evaluation code is served too.
  XSession: the adapter's behavioural Session extended by the kinds 'lookup' and 'blocklist' (the adapter's code path for
     everything else).
PATCH (run time only, no file of the adapter is changed): e10t.gate._scores_sub re-types the library's score formulas for the
  kinds lin / cos / ball; the query-normalised cosine gates of the feature layer (TCosLog: cos_centred, whitecos) score a key in
  another geometry, so the per-position refinement of e10t.gate.decide would use the raw cosine for them.  The patch routes such
  gates through their own ScopedLog.scores; all other gates take the adapter's path unchanged.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch

import xcommon as X
import e10t
from e10t import data as D, gate as G, methods as MT

R3_LABELS = ("cos_centred", "cos_null", "white_0.01", "white_0.1", "white_1", "whitecos_0.01", "whitecos_0.1", "whitecos_1")
STAT_N, STAT_SEED = 500, 20261111
WH_DIR = X.RUN + "/store/wstats"

_orig_scores_sub = G._scores_sub


def _scores_sub(S, k, J):
    if type(S).__name__ == "TCosLog":
        return S.scores(k)[:, J]
    return _orig_scores_sub(S, k, J)


G._scores_sub = _scores_sub


# ----------------------------------------------------------------------------- R3 statistics
def stat_prompts(L) -> list:
    """The 500 benign REPAIR-pool prompts of the R3 statistics: rows of the pool that are in none of the N_DRAWS calibration
    draws (n = 200), a seeded sample of 500 of them, in pool order.  The pool rows are the same for every log."""
    rows = L.benign("repair")
    used = set()
    for d in range(X.N_DRAWS):
        used |= set(D.calibration(L, X.N_CAL, d).text_ids)
    rest = [r for r in rows if r["text_id"] not in used]
    idx = np.sort(np.random.default_rng(STAT_SEED).permutation(len(rest))[:STAT_N])
    return [rest[int(i)] for i in idx]


_ST = {}


def r3_stats(L):
    """(white.wstats.KeyStats of the keys at all cached positions of the 500 prompts, record)."""
    if "st" not in _ST:
        from white import wstats
        rows = stat_prompts(L)
        tids = [r["text_id"] for r in rows]
        K, _ = L.kc.stack(tids)
        st = wstats.KeyStats.from_keys(K, name="REPAIR-500-all-positions")
        rec = {"n_prompts": len(rows), "n_keys": int(K.shape[0]), "d": int(K.shape[1]), "trace": st.trace, "norm_mu": float(st.mu.norm()),
               "prompt_ids_sha1": hashlib.sha1(",".join(r["prompt_id"] for r in rows).encode()).hexdigest()[:12],
               "n_near_scope_T2000": None, "seed": STAT_SEED,
               "rule": "REPAIR-pool benign prompts in none of the 20 calibration draws (n = 200); seeded sample of 500; keys at all "
                       "cached positions (prompt + 10-token pre-edit continuation)"}
        _ST["st"] = (st, rec)
    return _ST["st"]


def whitener(L, kappa: float):
    """white.wstats.Whitener of the R3 statistics at ridge kappa (primal form: more keys than dimensions); the factor is kept
    on disk (run/store/wstats) so that every process uses the same bits."""
    from white import wstats
    key = ("wh", float(kappa))
    if key not in _ST:
        st, rec = r3_stats(L)
        p = os.path.join(WH_DIR, f"F_{wstats.kappa_tag(kappa)}_{rec['prompt_ids_sha1']}.pt")
        rho = st.rho(kappa)
        if os.path.exists(p):
            F = torch.load(p, map_location="cpu")
            wh = wstats.Whitener("primal", F, rho, st.n, st.d, float(kappa), {"form": "primal", "N": st.n, "d": st.d, "rho": rho, "stats": st.name, "linv": "stored"})
        else:
            wh = st.whitener(kappa=float(kappa))
            if wh.form != "primal":
                raise RuntimeError("expected the primal form (more keys than dimensions)")
            os.makedirs(WH_DIR, exist_ok=True)
            tmp = p + f".tmp{os.getpid()}"
            torch.save(wh.F, tmp)
            os.replace(tmp, p)
        _ST[key] = wh
    return _ST[key]


def r3_base(L, label: str, draw: int | None = None):
    """Un-thresholded gate of an R3 variant on a text log (role 'all', top-1, REPL, rebased values) with the adapter's guards."""
    from rescope import add_guards
    from white import scores as WS
    V, _ = MT.released_values(L, True)
    kw = dict(role="all", fire="top1", response="repl", values=V)
    if label == "cos_centred":
        S = WS.feature_gate(L.log, "cos_centred", stats=r3_stats(L)[0], **kw)
    elif label == "cos_null":
        if draw is None:
            raise ValueError("cos_null depends on the calibration draw")
        A = D.null_sample(L, e10t.N_NULL, draw, X.N_CAL)
        if set(D.calibration(L, X.N_CAL, draw).text_ids) & {r["text_id"] for r in A}:
            raise RuntimeError("null sample and calibration draw share a prompt text")
        Kb, _ = L.kc.stack([r["text_id"] for r in A])
        S = WS.feature_gate(L.log, "cos_null", null_keys=Kb, **kw)
    elif label.startswith("white_") or label.startswith("whitecos_"):
        f, k = label.split("_")
        S = WS.feature_gate(L.log, f, stats=whitener(L, float(k)), **kw)
    else:
        raise KeyError(label)
    S = MT._named(S, "rebased")
    S, _ = add_guards(S, *L.own_keys())
    return S


_BASES = {}


def base_of(L, label: str, draw: int | None = None):
    """The un-thresholded gate (with guards) behind a calibrated row: 'rescope' / 'where_twoblock' (solver coordinate), 'cosine',
    or an R3 label.  One object per (log, feature) (cos_null: per draw, not kept)."""
    if label == "cos_null":
        return r3_base(L, label, draw)
    key = (L.name, label)
    if key not in _BASES:
        if label == "rescope":
            _BASES[key] = MT.base_gate(L, "ncoord", "top1", "repl", True, True)[0]
        elif label == "where_twoblock":
            _BASES[key] = MT.base_gate(L, "ncoord", "all", "lin", True, True)[0]
        elif label == "cosine":
            _BASES[key] = MT.base_gate(L, "cos", "top1", "repl", True, True)[0]
        else:
            _BASES[key] = r3_base(L, label)
    return _BASES[key]


def calibrated(L, label: str, draw: int, guards: bool = True):
    """(calibrated scoped log, calibration record) of a calibrated row at (alpha, n) = (0.05, 200), draw `draw`; guards=False is
    the adapter's allow_silence=True (R21 b: guards off)."""
    S = base_of(L, label, draw)
    S2, rec = MT.calibrate(L, S, "two_block", X.ALPHA, X.N_CAL, draw, allow_silence=not guards)
    return S2, rec


GOFF = "@goff"


def method(L, label: str, draw: int = 0, guards: bool = True):
    """A behavioural Method (adapter's dataclass) for any row label.  Rows of the adapter come from e10t.methods.rowset (switch
    allow_silence for guards off); R3 rows are wrapped (e10t.methods.wrap).  Label of a guards-off row: '<label>@goff'."""
    sfx = "" if guards else GOFF
    if label in ("deployed", "preedit", "cosine_published", "exact"):
        return MT.rowset(L, X.ALPHA, draw, X.N_CAL, labels=[label])[0]
    if label == "where_logonly":
        M = MT.where(L, "logonly", allow_silence=not guards)
        M.label = label + sfx
        return M
    if label in ("rescope", "cosine", "where_twoblock"):
        M = MT.rowset(L, X.ALPHA, draw, X.N_CAL, labels=[label], **({} if guards else {"allow_silence": True}))[0]
        M.label = label + sfx
        return M
    if label in R3_LABELS:
        t0 = time.time()
        S2, rec = calibrated(L, label, draw, guards)
        M = MT.wrap(L, S2, label + sfx, rec={"cal": rec, "feature": label, "calibration": "two_block", "seconds": time.time() - t0,
                                             "stats": r3_stats(L)[1] if label != "cos_null" else None}, alpha=X.ALPHA, n=X.N_CAL, draw=draw)
        M.variant = "" if guards else "allow_silence=True"
        return M
    if label == "lookup_subject":
        return lookup_method(L)
    raise KeyError(label)


# ----------------------------------------------------------------------------- lookup_subject
def subject_matches(text: str, subjects: list) -> list:
    """[(char index of the last character of an occurrence, edit)] for every occurrence of every subject string in text."""
    out = []
    for i, s in enumerate(subjects):
        a = text.find(s)
        while a >= 0:
            out.append((a + len(s) - 1, i))
            a = text.find(s, a + 1)
    return out


_PLAN_CACHE = {}


def lookup_plan(tok, ids, subjects, limit=None) -> list:
    """[(token position, edit)] of the releases of the keyed store on ONE row of token ids (limit: only the first `limit`
    tokens are searched = the prompt).  The row is decoded to text (no clean-up of spaces); position = the first token at which
    the decoded prefix holds the last character of the subject occurrence; the latest edit wins a position."""
    ids = tuple(ids if limit is None else ids[:limit])
    key = (id(subjects), ids)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key]
    text = tok.decode(list(ids), clean_up_tokenization_spaces=False)
    m = subject_matches(text, subjects)
    plan = {}
    if m:
        ends = np.array([len(tok.decode(list(ids[:j + 1]), clean_up_tokenization_spaces=False)) for j in range(len(ids))])
        for c, e in m:
            p = int(np.searchsorted(ends, c, side="right"))
            plan[p] = max(plan.get(p, -1), e)
    out = sorted(plan.items())
    if len(_PLAN_CACHE) < 2000000:
        _PLAN_CACHE[key] = out
    return out


class LookupGate:
    """Gate object for e10t.Backbone.served: releases the rebased value of edit e at the planned positions of a FULL forward
    (a call without past_key_values); key-value-cache steps release nothing (the subject is searched in the prompt).  The
    plan of a call comes from the token ids captured by install_capture(B).  limits: per-row prompt lengths for the next full
    forward (teacher-forced passes feed [prompt, y0]; None = the whole row)."""
    fire, response, role = "top1", "repl", "all"

    def __init__(self, L, tok):
        self.tok, self.subjects = tok, [rq["subject"] for rq in L.requests]
        self.values = MT.released_values(L, True)[0]
        self.T, self.limits, self.device = L.T, None, None
        self.sig = "L" + hashlib.sha1(("|".join(self.subjects)).encode() + self.values.numpy().tobytes()).hexdigest()[:16]
        self._vals = None

    def to(self, device):
        return self

    def firing(self, x, lp=None, pos0=0, state=None):
        from rescope.scoped import Firing
        cap = _CAPTURE.get("last")
        if cap is None or cap["past"]:
            return None
        ids, am = cap["ids"], cap["mask"]
        if x.dim() != 3 or ids.shape != x.shape[:2]:
            raise RuntimeError("LookupGate: the captured token ids do not match the module input")
        rows, pos, ed = [], [], []
        for r in range(ids.shape[0]):
            n = int(am[r].sum()) if am is not None else ids.shape[1]
            lim = None if self.limits is None else int(self.limits[r])
            for p, e in lookup_plan(self.tok, ids[r, :n].tolist(), self.subjects, lim):
                rows.append(r); pos.append(p); ed.append(e)
        self.limits = None
        if not rows:
            return None
        dev = x.device
        row, pos, ed = (torch.tensor(v, dtype=torch.long, device=dev) for v in (rows, pos, ed))
        return Firing(idx=row * x.shape[1] + pos, row=row, pos=pos, edit=ed)

    def respond(self, x, out, f):
        if self._vals is None or self._vals.device != out.device:
            self._vals = self.values.to(out.device)
        m = out.shape[-1]
        new = out.reshape(-1, m).clone()
        new[f.idx] = (new[f.idx].double() + self._vals[f.edit]).to(out.dtype)
        return new.view(out.shape)


_CAPTURE = {}


def install_capture(B):
    """A forward pre-hook on the model that records the token ids of every call (read by LookupGate).  Installed once."""
    if _CAPTURE.get("installed") is B:
        return

    def pre(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        _CAPTURE["last"] = {"ids": ids.detach().cpu(), "mask": None if kwargs.get("attention_mask") is None or kwargs.get("past_key_values") is not None
                            else kwargs["attention_mask"].detach().cpu(), "past": kwargs.get("past_key_values") is not None}
        return None
    B.model.register_forward_pre_hook(pre, with_kwargs=True)
    _CAPTURE["installed"] = B


def lookup_method(L):
    tok = D.tokenizer()
    g = LookupGate(L, tok)
    rec = {"name": "lookup_subject", "log": L.name, "T": L.T, "class": "keyed store over the REQUESTS (not a repair of the log): needs the subject strings",
           "rule": "release iff the prompt contains the edit's subject string, at the last token of the occurrence; the latest edit wins a position; "
                   "value = the migration's rebased value; response REPL", "sig": g.sig}
    return MT.Method("lookup_subject", "lookup_subject", g, "lookup", L.name, dict(rec, fingerprint=g.sig[1:13]))


def lookup_text_plan(L, M, tid) -> list:
    """Releases of the keyed store on the PROMPT of a cached text: [(position, edit)]."""
    c = M.rec.setdefault("_plan", {})
    if tid not in c:
        c[tid] = lookup_plan(M.obj.tok, L.ids(tid), M.obj.subjects)
    return c[tid]


# ----------------------------------------------------------------------------- session with the kinds 'lookup' and 'blocklist'
class XSession(e10t.behave.Session):
    """The adapter's Session; adds kind 'lookup' (LookupGate; a prompt without a subject match is the pre-edit model: cached
    continuation, no forward) and kind 'blocklist' (deployed model + exact-match list of prompt texts answered by the pre-edit
    model: rows are taken from the stored outcomes of the two weights)."""

    def served_obj(self, M):
        if M.kind == "lookup":
            install_capture(self.B)
            return M.obj
        return super().served_obj(M)

    def sig(self, M):
        if M.kind == "lookup":
            return M.obj.sig
        if M.kind == "blocklist":
            return "B" + M.rec["fingerprint"]
        return super().sig(M)

    def outcomes(self, M, tids, mode="tf", force_run=False):
        if M.kind == "blocklist":
            dep, pre = MT.deployed(self.L), MT.preedit(self.L)
            tids = list(dict.fromkeys(int(t) for t in tids))
            listed = [t for t in tids if t in M.obj]
            out = super().outcomes(dep, [t for t in tids if t not in M.obj], mode) if len(listed) < len(tids) else {}
            if listed:
                out.update(super().outcomes(pre, listed, mode))
            return out
        if M.kind != "lookup":
            return super().outcomes(M, tids, mode, force_run)
        L, B = self.L, self.B
        tids = list(dict.fromkeys(int(t) for t in tids))
        ns = f"{mode}__{self.sig(M)}"
        out, run = {}, []
        for t in tids:
            o = self.store.get(ns, t)
            if o is not None:
                out[t] = dict(o, src="store")
                self.stats["from_store"] += 1
            elif not lookup_text_plan(L, M, t):
                out[t] = {"tok": L.y0(t), "fin": True, "fired": False, "n_fire_pos": 0, "rel": [], "rel_edits": [], "n_rel_edits": 0,
                          "n_cand_edits": 0, "cand_edits": [], "near": False, "src": "silent"}
                self.stats["silent"] += 1
            else:
                run.append(t)
        if run:
            lens = [len(L.ids(t)) for t in run]
            t0 = time.time()
            with B.served(self.served_obj(M)) as srv:
                for b in B.batches(lens):
                    bt = [run[i] for i in b]
                    seqs = [L.ids(t) for t in bt]
                    if mode == "tf":
                        M.obj.limits = [len(s) for s in seqs]
                        tok, fin = B.tf(seqs, [L.y0(t) for t in bt])
                    else:
                        tok, fin = B.decode(seqs)
                    flags = srv.rows()
                    self.stats["batches"] += 1
                    for i, t in enumerate(bt):
                        f = flags[i]
                        plan = lookup_text_plan(L, M, t)
                        if sorted((p, e) for p, e in f["rel"]) != sorted(plan):
                            self.stats["hook_mismatch" if mode == "tf" else "hook_mismatch_dec"] += 1
                            self.stats["mismatches"].append({"text_id": t, "mode": mode, "method": M.label, "plan": plan, "hook": f["rel"]})
                        o = {"tok": tok[i].tolist(), "fin": bool(fin[i]), "fired": f["fired"], "n_fire_pos": f["n_fire_pos"], "rel": f["rel"],
                             "rel_edits": f["rel_edits"], "n_rel_edits": f["n_rel_edits"], "n_cand_edits": f["n_rel_edits"], "cand_edits": f["rel_edits"], "near": False}
                        self.store.put(ns, t, o)
                        out[t] = dict(o, src="run")
            self.stats["sequences_run"] += len(run)
            self.stats["seconds_model"] += time.time() - t0
        return out


def blocklist_method(L, tids, label="blocklist"):
    s = set(int(t) for t in tids)
    fp = hashlib.sha1((L.name + ":" + ",".join(str(t) for t in sorted(s))).encode()).hexdigest()[:12]
    return MT.Method("blocklist", label, s, "blocklist", L.name, {"name": "blocklist", "log": L.name, "T": L.T, "n_listed": len(s), "fingerprint": fp,
                                                                  "rule": "deployed model; the listed prompt texts are answered by the pre-edit model"})
