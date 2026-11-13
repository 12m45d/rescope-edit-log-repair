"""Complaint-driven repair operators on scoped logs.  Pure functions: (scoped log, complaint) -> (new scoped
log, record of what changed).  All arithmetic fp64.  "keys" are always the module inputs at the ELIGIBLE
positions of the complaint's query (ScopedLog.eligible_keys of the captured input; role 'last': one key).

least_repair    wrong-answer complaint: raise the threshold of every detector that is a candidate on the
                complaint to just above its largest coordinate there.  A coordinatewise maximum => idempotent
                and order independent (bitwise); with margins 0 the unique least repair.
null_repair     detector repair: minimum-change detectors with zero coordinate on the given benign keys and
                unchanged own coordinate (closed form; ridge variant).
witness_repair  why-not complaint (edit i should fire on a key): lower a threshold to the witness coordinate,
                or add a detector centred on the witness key.
scope_cut       exact two-sided optimum of the per-edit thresholds by one minimum cut.
add_guards      further must-fire keys of an edit (protected by every calibration / repair; a guard that does not
                fire is MADE to fire by the next calibration, so non-firing keys are refused by default).
Session         bookkeeping sugar: the benign queries recorded so far + the operator records.

Thresholds are compared in the normalised domain after folding eta (ScopedLog.fold).  "Just above" a
coordinate m is the next float after m (1 + margin) + abs_margin; the defaults (1e-6, 1e-12; P10's bump) absorb
the round-off of re-evaluating a key on another device or in another batch, margins 0 give the exact lattice
statements (next float).  Guard keys are always scored one by one (ScopedLog.key_scores).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .calibrate import own_caps
from .scoped import COS, INF, LIN, ScopedLog


def _f64(S, x):
    return torch.as_tensor(x).to(S.device, torch.float64).reshape(-1, S.d)


def _queries(S, keys):
    """keys -> list of fp64 (n_q, d) tensors, one per QUERY (a tensor is one query; a list is one query per entry).
    The coordinates of a query are evaluated as one batch, exactly as the hook evaluates its eligible positions."""
    return [_f64(S, keys)] if torch.is_tensor(keys) else [_f64(S, k) for k in keys]


MARGIN, ABS_MARGIN = 1e-6, 1e-12     # default "just above / just below" (P10's bump): far above the round-off of a recomputed
                                     # inner product (other device, other batch shape), far below any difference that matters
DECODE_MARGIN = 0.05                 # relative margin for repairs that must also hold on the DECODE path (D2, free generation) when
                                     # only the keys of the two sequences are used (Evaluator.eligible_keys(answer="both")): a decode
                                     # re-evaluates the last prompt position in up to 11 full forwards of growing length and the low-
                                     # precision forward gives each a slightly different key (GPU test b9 measures the margin needed).
                                     # A protocol constant, not a guarantee; the exact alternative for role 'last' is answer="all"
                                     # (every key of a decode) with the default margin.


def _above(m, margin=0.0, abs_margin=0.0):
    """Threshold just above a coordinate m (elementwise): the next float after m + margin |m| + abs_margin
    (-inf stays -inf: nothing to silence).  margin = abs_margin = 0: the least threshold that silences m."""
    base = m if margin == 0 and abs_margin == 0 else torch.where(torch.isinf(m), m, m + margin * m.abs() + abs_margin)
    return torch.where(m == -INF, m, torch.nextafter(base, torch.full_like(m, INF)))


def _below(x, margin=0.0, abs_margin=0.0):
    """Threshold at / just below a coordinate x that must fire: x - margin |x| - abs_margin."""
    return x if margin == 0 and abs_margin == 0 else torch.where(torch.isinf(x), x, x - margin * x.abs() - abs_margin)


def _guards(S, must_fire=None):
    """(edits (G,), scores (G, D)) of the guard keys (+ extra must-fire keys), every key as its own query."""
    ge, Rg = S.guard_edit, S.guard_scores()
    if must_fire is not None and len(must_fire[1]):
        ge = torch.cat([ge, torch.as_tensor(must_fire[1], dtype=torch.long, device=S.device).reshape(-1)])
        Rg = torch.cat([Rg, S.key_scores(_f64(S, must_fire[0]))])
    return ge, Rg


def _guard_fires(S, tau, Rg, ge):
    """(G,) bool: guard g's edit is a candidate on guard key g under thresholds tau (Rg = scores of the guard keys)."""
    c = S._edit_any(Rg >= tau[None, :])
    return c[torch.arange(c.shape[0], device=c.device), ge]


def add_guards(S: ScopedLog, keys, edits, allow_nonfiring: bool = False):
    """Scoped log with further must-fire keys: edit edits[n] has to stay a candidate on keys[n] (one key per row), e.g.
    the B = 1 key of every edit's own prompt next to its stored (batched) solver key (Evaluator.own_keys).
    A guard is a MUST-FIRE constraint, not a wish: least_repair refuses to silence a guard that fires, and
    calibrate_* / scope_cut LOWER the edit's threshold until every guard of the edit fires (calibration records:
    `n_guards_not_firing_before`).  A key on which the edit does not fire now would therefore pull the edit's
    threshold down to its coordinate at the next calibration: such keys are refused (ValueError) unless
    allow_nonfiring=True.  Add guards BEFORE calibrating (at tau = -inf every key fires), or use witness_repair,
    which checks the recorded benign queries first.  The record counts the keys that do not fire now."""
    k = _f64(S, keys)
    e = torch.as_tensor(edits, dtype=torch.long, device=S.device).reshape(-1)
    if k.shape[0] != e.shape[0]:
        raise ValueError("one edit per key")
    new = S.replace(guard_key=torch.cat([S.guard_key, k]), guard_edit=torch.cat([S.guard_edit, e]))
    ok = new.guard_ok()[S.guard_key.shape[0]:]
    if not allow_nonfiring and not bool(ok.all()):
        raise ValueError(f"{int((~ok).sum())} of {int(ok.numel())} key(s) do not fire their edit now: as guards they would force the next "
                         "calibration to lower those thresholds to the keys' coordinates (add guards before calibrating, use "
                         "witness_repair, or pass allow_nonfiring=True)")
    return new, {"op": "add_guards", "n_added": int(k.shape[0]), "n_firing_now": int(ok.sum()), "not_firing_now": [int(i) for i in (~ok).nonzero().flatten()]}


# ----------------------------------------------------------------------------- 3. least repair
def least_repair(S: ScopedLog, keys, margin: float = MARGIN, abs_margin: float = ABS_MARGIN, on_conflict: str = "refuse",
                 must_fire=None):
    """tau_j <- max(tau_j, above(m_j)) for every detector j, m_j = its largest coordinate on the complaint's eligible
    keys (keys: one tensor = one complaint, or a list of tensors = several complaints at once).

    A coordinatewise maximum: the least thresholds >= tau under which every complaint stays `margin` clear of
    firing; idempotent and independent of the order in which complaints are repaired (bitwise).
    margin = abs_margin = 0 is the textbook least repair: only the candidate detectors change, each to the NEXT
    FLOAT above its coordinate.  It silences the complaint only if the hook evaluates the complaint on the same
    device and in the same batch as here; the default (1e-6 relative + 1e-12) absorbs that round-off, and also
    lifts a detector whose coordinate is within the margin below its threshold.  Neither covers the round-off of
    the FORWARD PASS (the same query in another batch / context moves coordinates at the 1e-2 level in bf16), nor a
    complaint whose decode sits on another token than its teacher-forced sequence (evaluate.py, `lp_shift`).
    A repair that must also hold for D2 / free generation takes every key the complaint can present to the gate:
    Evaluator.eligible_keys(it, S, answer="all") with the default margin (exact; role 'last'), or the keys of the two
    sequences (answer="both") with margin=DECODE_MARGIN (GPU test b9).
    A repair that would stop an edit's guard key (own solver key, granted witnesses, `must_fire` =
    (keys, edits)) from firing is a CONFLICT: on_conflict 'refuse' (default) returns S unchanged, 'partial'
    leaves the detectors of the conflicting edits untouched, 'force' silences them."""
    if on_conflict not in ("refuse", "partial", "force"):
        raise ValueError(on_conflict)
    S1 = S.fold()
    qs = _queries(S, keys)
    Q = S1.query_max(qs)                                             # (n_queries, D)
    m = Q.max(0).values
    tau = S1.det_tau
    new = torch.maximum(tau, _above(m, margin, abs_margin))
    ge, Rg = _guards(S1, must_fire)
    before, after = _guard_fires(S1, tau, Rg, ge), _guard_fires(S1, new, Rg, ge)
    lost = (before & ~after).nonzero().flatten()
    conflicts = [{"guard": int(g), "edit": int(ge[g])} for g in lost]
    rec = {"op": "least_repair", "n_queries": len(qs), "n_keys": int(sum(q.shape[0] for q in qs)), "margin": margin,
           "abs_margin": abs_margin, "eta_folded": S.eta, "n_candidates_before": int((m >= tau).sum()), "conflicts": conflicts,
           "refused": False}
    if conflicts and on_conflict == "refuse":
        rec.update(refused=True, n_raised=0, raised=[], resolved=False)
        return S, rec
    if conflicts and on_conflict == "partial":
        bad = torch.zeros(S.T, dtype=torch.bool, device=S.device)
        bad[ge[lost]] = True
        new = torch.where(bad[S.det_edit], tau, new)
    ch = (new > tau).nonzero().flatten()
    rec.update(n_raised=int(ch.numel()), resolved=not bool((Q >= new[None, :]).any()),
               raised=[{"detector": int(j), "edit": int(S.det_edit[j]), "tau_old": float(tau[j]), "tau_new": float(new[j]),
                        "coordinate": float(m[j])} for j in ch])
    return S1.replace(det_tau=new), rec


# ----------------------------------------------------------------------------- 4. null repair
def null_repair(S: ScopedLog, keys, ridge: float = 0.0, zero_force_log: bool = False, rcond: float = 1e-12):
    """Replace every 'lin' detector w_j (key k_j, scale s_j) by the minimum-change vector w_j' with
         <w_j', b> = 0 for every benign key b in `keys`,    <w_j', k_j> = s_j.
    Closed form:  w' = M w + nu M k,  nu = (s - <M w, k>) / <M k, k>,
      M = I - Kb^T (Kb Kb^T)^+ Kb               (ridge = 0: the projector onto the complement of the benign keys)
      M = I - Kb^T (Kb Kb^T + rho I)^-1 Kb      (ridge > 0, rho = ridge * mean ||b||^2: the minimiser of
                                                 ||w' - w||^2 + ||Kb w'||^2 / rho  s.t. <w', k> = s; the benign
                                                 coordinates shrink instead of vanishing).
    zero_force_log=True (one detector per edit only) adds <w_i', k_j> = 0 for the OTHER edits' keys
    (P11b's NULL(n): joint zero-forcing on the other solver keys and the reported keys):
         W' = M W - (M W K^T - diag(s)) (K M K^T)^-1 M K.
    Thresholds, values, guards are unchanged.  A detector whose key lies in the benign span cannot keep its
    own coordinate: it is left unchanged and reported (`infeasible`)."""
    Kb = _f64(S, keys)
    if Kb.shape[0] == 0:
        return S, {"op": "null_repair", "n_keys": 0, "n_detectors": 0, "infeasible": []}
    lin = (S.det_kind == LIN).nonzero().flatten()
    W, A, s = S.det_w[lin], S.det_key[lin], S.det_s[lin]
    G = Kb @ Kb.T
    n = G.shape[0]
    rho = ridge * float(G.diagonal().mean()) if ridge > 0 else 0.0
    ev, V = torch.linalg.eigh(G)
    keep = ev > rcond * ev[-1]                                       # numerically null directions of Kb contribute exactly 0
    Q = (V[:, keep] / (ev[keep] + rho)) @ V[:, keep].T               # (Kb Kb^T + rho I)^-1 on the row space of Kb
    rank = int(keep.sum())

    def M(X):
        return X - ((X @ Kb.T) @ Q) @ Kb
    Wt, At = M(W), M(A)
    den = (At * A).sum(1)
    feas = den > 1e-9 * (A * A).sum(1)
    if zero_force_log:
        if lin.numel() != S.T or S.D != S.T:
            raise NotImplementedError("zero_force_log needs exactly one 'lin' detector per edit")
        if not bool(feas.all()):
            raise ValueError("a solver key lies in the span of the benign keys: joint zero-forcing is infeasible")
        C = Wt @ A.T - torch.diag(s)
        Wn = Wt - torch.linalg.solve(At @ A.T, C.T).T @ At
    else:
        nu = (s - (Wt * A).sum(1)) / torch.where(feas, den, torch.ones_like(den))
        Wn = torch.where(feas[:, None], Wt + nu[:, None] * At, W)
    det_w = S.det_w.clone()
    det_w[lin] = Wn
    new = S.replace(det_w=det_w)
    chg = ((Wn - W).norm(dim=1) / W.norm(dim=1))
    ok0, ok1 = S.guard_ok(), new.guard_ok()
    rec = {"op": "null_repair", "n_keys": n, "rank_benign": rank, "ridge": ridge, "rho": rho, "zero_force_log": zero_force_log,
           "n_detectors": int(lin.numel()), "infeasible": [int(j) for j in lin[~feas]],
           "rel_change_median": float(chg.median()), "rel_change_max": float(chg.max()),
           "max_abs_benign_r": float(((Wn[feas] @ Kb.T) / s[feas, None]).abs().max()) if bool(feas.any()) and n else 0.0,
           "max_own_rel_err": float((((Wn * A).sum(1) - s) / s).abs().max()),
           "n_guards_firing_before": int(ok0.sum()), "n_guards_firing_after": int(ok1.sum()),
           "n_guards_lost": int((ok0 & ~ok1).sum())}
    return new, rec


# ----------------------------------------------------------------------------- 5. witness repair
def _rank_ext(S, t_new):
    """Ranking baseline / scale (ScopedLog.det_b / det_u) of a detector that is ADDED to a log whose top-1 candidates are
    ranked by margins (two-block calibration): b = its threshold minus the calibrated eta (meta['rank_eta']; 0 if unknown
    or infinite), u = 1, so that its margin exceeds eta by exactly its excess over its own threshold.  {} for logs that
    rank by r."""
    if S.det_b is None and S.det_u is None:
        return {}
    eta = float(S.meta.get("rank_eta", 0.0))
    b_new = float(t_new) - (eta if np.isfinite(eta) else 0.0)
    b = S.det_b if S.det_b is not None else torch.zeros_like(S.det_tau)
    u = S.det_u if S.det_u is not None else torch.ones_like(S.det_tau)
    one = lambda v: torch.tensor([v], dtype=torch.float64, device=S.device)                       # noqa: E731
    return {"det_b": torch.cat([b, one(b_new)]), "det_u": torch.cat([u, one(1.0)])}


def witness_repair(S: ScopedLog, edit: int, key, mode: str = "threshold", benign_keys=None, margin: float = MARGIN,
                   abs_margin: float = ABS_MARGIN, gamma: float = 0.98, detector=None, tau=None, force: bool = False):
    """Why-not complaint: edit `edit` should fire on `key` (one eligible key).

    mode 'threshold': lower the threshold of the edit's detector that needs the smallest change to the witness
        coordinate r_j(key) (minus the margins, see least_repair; 0 / 0 = exactly the coordinate).  If a recorded
        benign query (`benign_keys`: a list of per-query key tensors, e.g. Session.benign, or one (n, d) tensor =
        n one-key queries) that is silent on that detector now would fire through it afterwards, this is a
        CONFLICT: S is returned unchanged with the conflicting benign queries in the record (force=True applies
        it anyway).
    mode 'detector': add a detector for the edit centred on the witness key.  Default = the NORMALISED WITNESS
        KEY WITH A COSINE THRESHOLD gamma (kind 'cos': r = cos(k, key)).  The log-style read-out vector of a new
        key, (l2 I + P (K K^T + Kp Kp^T))^-1 P k, needs the editor's protected-subspace projector P, which is not
        a stored log quantity; a caller that has it passes the vector as `detector` (kind 'lin', s = <detector,
        key>, threshold `tau`, default the threshold of the edit's primary detector).  Same conflict rule.
    On success the witness becomes a guard of the edit (later repairs must keep it firing).  The repair makes the
    edit a CANDIDATE on the witness; under fire 'top1' another edit with a larger coordinate may still be the one
    that releases (record: `edit_is_top1_on_witness`)."""
    if mode not in ("threshold", "detector"):
        raise ValueError(mode)
    S1 = S.fold()
    k = _f64(S, key)
    if k.shape[0] != 1:
        raise ValueError("one witness key per call")
    js = (S1.det_edit == edit).nonzero().flatten()
    r = S1.scores(k)[0]
    rec = {"op": "witness_repair", "mode": mode, "edit": int(edit), "conflict": False, "conflicts": [], "applied": False,
           "already_fired": bool((r[js] >= S1.det_tau[js]).any()), "eta_folded": S.eta}
    guard = dict(guard_key=torch.cat([S1.guard_key, k]),
                 guard_edit=torch.cat([S1.guard_edit, torch.tensor([edit], dtype=torch.long, device=S.device)]))
    if rec["already_fired"]:
        return S1.replace(**guard), rec
    B = None                                                         # per-query maxima are evaluated like least_repair does
    if benign_keys is not None and len(benign_keys):
        B = [q[None] for q in _f64(S, benign_keys)] if torch.is_tensor(benign_keys) else [_f64(S, q) for q in benign_keys]
    if mode == "threshold":
        j = int(js[(r[js] - S1.det_tau[js]).argmax()])
        t_new = _below(r[j], margin, abs_margin)
        rec.update(detector=j, tau_old=float(S1.det_tau[j]), tau_new=float(t_new), coordinate=float(r[j]),
                   tau_nonpositive=bool(t_new <= 0))          # a threshold <= 0 fires on a half-space: reported, not refused
        if B is not None:
            rb = S1.query_max(B)[:, j]
            hit = ((rb >= t_new) & (rb < S1.det_tau[j])).nonzero().flatten()
            rec["conflicts"] = [{"benign": int(b), "r": float(rb[b])} for b in hit]
        new_tau = S1.det_tau.clone()
        new_tau[j] = t_new
        out = S1.replace(det_tau=new_tau, **guard)
    else:
        if detector is None:
            w, s_new, kind = k / k.norm(), 1.0, COS
            t_new = float(gamma)
            rec["detector_kind"] = "normalised witness key, cosine threshold"
        else:
            w = _f64(S, detector)
            s_new, kind = float((w * k).sum()), LIN
            if not s_new > 0:
                raise ValueError("the supplied detector has a non-positive coordinate on the witness key")
            t_new = float(S1.det_tau[int(edit)] if tau is None else tau)
            rec["detector_kind"] = "supplied vector, linear coordinate"
        one = lambda v, dt: torch.tensor([v], dtype=dt, device=S.device)                       # noqa: E731
        out = S1.replace(det_w=torch.cat([S1.det_w, w]), det_s=torch.cat([S1.det_s, one(s_new, torch.float64)]),
                         det_tau=torch.cat([S1.det_tau, one(t_new, torch.float64)]),
                         det_edit=torch.cat([S1.det_edit, one(int(edit), torch.long)]),
                         det_kind=torch.cat([S1.det_kind, one(kind, torch.long)]), det_key=torch.cat([S1.det_key, k]),
                         **_rank_ext(S1, t_new), **guard)
        rec.update(detector=out.D - 1, tau_new=t_new, coordinate=float(out.scores(k)[0, -1]))
        if B is not None:
            rb = out.query_max(B)[:, -1]
            silent = ~(S1.query_max(B) >= S1.det_tau[None, :]).any(1)
            hit = ((rb >= t_new) & silent).nonzero().flatten()
            rec["conflicts"] = [{"benign": int(b), "r": float(rb[b])} for b in hit]
    rec["conflict"] = bool(rec["conflicts"])
    if rec["conflict"] and not force:
        return S, rec
    ro = out.key_scores(k)[0]                                       # who releases on the witness under fire 'top1'?
    if not bool(((ro >= out.det_tau) & (out.det_edit == int(edit))).any()):
        # the witness becomes a guard = a must-fire key: a repair that does not make it fire would leave a non-firing
        # guard behind, which the next calibration would enforce by lowering a threshold of the edit
        raise ValueError("the repair does not make the edit fire on the witness (cosine detector: gamma must be below 1; supplied "
                         "detector: tau must be below 1, the coordinate of the witness on it)")
    top = int(out.det_edit[torch.where(ro >= out.det_tau, out.rank_key(ro[None])[0], torch.full_like(ro, -INF)).argmax()])
    rec.update(applied=True, top1_edit_on_witness=top, edit_is_top1_on_witness=top == int(edit))
    return out, rec


# ----------------------------------------------------------------------------- 6. scope cut
class CutProblem:
    """max over thresholds tau <= cap of   sum_{clean rows} a_r  -  lam * sum_{uncovered witnesses} b_w.

    C (N, T): coordinate of edit i on benign row r (-inf: edit i is exempt on that row).
    row r is CLEAN iff C[r, i] < tau_i for every i;  witness w = (edit i, coordinate x) is COVERED iff x >= tau_i.
    Max-weight closure solved by one minimum cut (ported from 08_rescope/p11_scopecut/code/scopelib.py):
    source -> feasible row (a_r), witness -> sink (lam b_w), row -> the highest witness of each edit that is not
    above the row's coordinate (inf), witness -> next lower witness of the same edit (inf).  Capacities are
    integers round(weight * scale) (scipy maximum_flow, int32): the optimum is exact for those integer weights
    (pass integer weights and scale=1, or a scale that makes lam * b_w * scale integral, for an exact optimum of
    the stated objective)."""

    def __init__(self, C, cap, wit_edit, wit_x, row_w=None, wit_w=None, scale=None):
        self.C = np.asarray(C, dtype=np.float64)
        self.N, self.T = self.C.shape
        self.cap = np.asarray(cap, dtype=np.float64)
        self.we = np.asarray(wit_edit, dtype=np.int64).reshape(-1)
        self.wx = np.asarray(wit_x, dtype=np.float64).reshape(-1)
        self.M = len(self.we)
        self.row_w = np.ones(self.N) if row_w is None else np.asarray(row_w, dtype=np.float64)
        self.wit_w = np.ones(self.M) if wit_w is None else np.asarray(wit_w, dtype=np.float64)
        self.scale = float(2 ** 29 / max(self.row_w.sum(), 1e-300)) if scale is None else float(scale)
        self.a_int = np.maximum(np.rint(self.row_w * self.scale), 0).astype(np.int64)
        self.exempt = np.isneginf(self.C)
        self.feasible = (self.C < self.cap[None, :]).all(1)          # rows that can be clean at all
        self.wact = self.wx < self.cap[self.we]                      # witnesses that can be uncovered at all
        fr, wa = np.nonzero(self.feasible)[0], np.nonzero(self.wact)[0]
        node = {int(w): 2 + len(fr) + n for n, w in enumerate(wa)}
        src, dst = [], []
        for i in (np.unique(self.we[wa]) if len(wa) else []):
            ws = wa[self.we[wa] == i]
            ws = ws[np.argsort(self.wx[ws], kind="stable")]
            for j in range(1, len(ws)):                              # a higher uncovered witness => the lower ones too
                src.append(node[int(ws[j])])
                dst.append(node[int(ws[j - 1])])
            cnt = np.searchsorted(self.wx[ws], self.C[fr, i], side="right")       # witnesses of i not above the row
            for r in np.nonzero(cnt > 0)[0]:
                src.append(2 + int(r))
                dst.append(node[int(ws[cnt[r] - 1])])
        self.fr, self.wa = fr, wa
        self.e_src, self.e_dst = np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)
        self.n_nodes = 2 + len(fr) + len(wa)
        self.total = int(self.a_int[fr].sum())
        self.BIG = self.total + 1
        if self.BIG >= 2 ** 31 - 1:
            raise ValueError("row weights * scale overflow int32")

    def wit_cap(self, lam):
        return np.minimum(np.rint(lam * self.wit_w * self.scale), self.BIG).astype(np.int64)

    def solve(self, lam):
        """-> (clean (N,) bool: the inclusion-minimal optimal clean set, info)."""
        from scipy.sparse import csr_array
        from scipy.sparse.csgraph import breadth_first_order, maximum_flow
        nf, wa = len(self.fr), self.wa
        clean = np.zeros(self.N, dtype=bool)
        if nf == 0:
            return clean, {"flow": 0, "value_int": 0}
        rows = np.concatenate([np.zeros(nf, dtype=np.int64), self.e_src, 2 + nf + np.arange(len(wa))])
        cols = np.concatenate([2 + np.arange(nf), self.e_dst, np.ones(len(wa), dtype=np.int64)])
        vals = np.concatenate([self.a_int[self.fr], np.full(len(self.e_src), self.BIG, dtype=np.int64), self.wit_cap(lam)[wa]])
        keep = vals > 0
        Am = csr_array((vals[keep].astype(np.int32), (rows[keep], cols[keep])), shape=(self.n_nodes, self.n_nodes))
        res = maximum_flow(Am, 0, 1)
        Rm = (Am - res.flow).tocsr()
        Rm.data[Rm.data < 0] = 0
        Rm.eliminate_zeros()
        side = np.zeros(self.n_nodes, dtype=bool)
        side[breadth_first_order(Rm, 0, directed=True, return_predecessors=False)] = True
        clean[self.fr] = side[2:2 + nf]
        return clean, {"flow": int(res.flow_value), "value_int": self.total - int(res.flow_value),
                       "n_nodes": int(self.n_nodes), "n_arcs": int(keep.sum())}

    def box(self, clean):
        """(m, hi): the optimal thresholds of a clean set are exactly the box m_i < tau_i <= hi_i, with
        m_i = largest coordinate of edit i over the clean rows (-inf if none), hi_i = min(cap_i, lowest witness above m_i)."""
        m = self.C[clean].max(0) if clean.any() else np.full(self.T, -np.inf)
        hi = self.cap.copy()
        for w in range(self.M):
            i = self.we[w]
            if m[i] < self.wx[w] < hi[i]:
                hi[i] = self.wx[w]
        return m, hi

    def objective(self, tau, lam):
        """(integer objective, clean, uncovered) of a threshold vector; None if tau exceeds a cap."""
        tau = np.asarray(tau, dtype=np.float64)
        if (tau > self.cap).any():
            return None
        clean = ((self.C < tau[None, :]) | self.exempt).all(1)
        unc = self.wx < tau[self.we]
        return int(self.a_int[clean].sum()) - int(self.wit_cap(lam)[unc].sum()), clean, unc


def brute_force(P: CutProblem, lam):
    """Exhaustive optimum of the integer objective over every threshold pattern (tests; small instances)."""
    cands = []
    for i in range(P.T):
        v = np.concatenate([P.C[:, i][np.isfinite(P.C[:, i])], P.wx[P.we == i], [P.cap[i]]])
        cands.append(np.concatenate([[-np.inf], np.unique(v[v <= P.cap[i]])]))
    return max(P.objective(np.array(c), lam)[0] for c in itertools.product(*cands))


def scope_cut(S: ScopedLog, rows, witnesses, lam: float, row_w=None, wit_w=None, rule: str = "nearest",
              margin: float = MARGIN, abs_margin: float = ABS_MARGIN, scale=None):
    """Per-edit thresholds maximising  w(clean benign rows) - lam * w(uncovered witnesses)  exactly (one min cut).

    rows: the benign queries, as a list of per-query key tensors (eligible keys of each query) or as their (N, T)
    matrix of normalised coordinates (ScopedLog.query_max; -inf = edit exempt on that row);
    witnesses: the keys that should fire, as a list of (edit, key) pairs or as a tuple (edits (M,), coordinates
    (M,)).  Own keys (guards) are hard constraints: tau_i <= cap_i (a guard that does not fire now is made to
    fire: `n_guards_not_firing_before`).  The optimum is a box m_i < tau_i <= x_i (m = largest clean-row coordinate,
    x = lowest covered witness or the cap); the returned thresholds lie in lo_i <= tau_i <= hi_i with lo = just
    above m and hi = just below x (margins as in least_repair; where the margins do not fit the exact box is
    used and counted in `n_tight`).  `rule` picks the point: 'nearest' (default: the point of the box closest to
    the current thresholds, i.e. the minimal change), 'lo', 'mid', 'hi'.  One detector per edit only; eta is
    folded.  Weights default to 1 per row / witness."""
    if S.D != S.T:
        raise NotImplementedError("scope_cut optimises one threshold per edit (logs without added detectors)")
    if rule not in ("nearest", "lo", "mid", "hi"):
        raise ValueError(rule)
    S1 = S.fold()
    if isinstance(rows, (list, tuple)):                              # per-query key tensors -> coordinates
        rows = S1.query_max([_f64(S, q) for q in rows])
    C = torch.as_tensor(rows).to(torch.float64).cpu().numpy().reshape(-1, S.T)
    if isinstance(witnesses, list):                                  # (edit, key) pairs -> coordinates, each key as its own query
        we = [int(e) for e, _ in witnesses]
        wx = torch.stack([S1.key_scores(_f64(S, k))[0, e] for e, k in witnesses]) if witnesses else torch.zeros(0, dtype=torch.float64)
    else:
        we, wx = witnesses
    P = CutProblem(C, own_caps(S1).cpu().numpy(), np.asarray(we), torch.as_tensor(wx).to(torch.float64).cpu().numpy(),
                   row_w, wit_w, scale)
    clean, info = P.solve(lam)
    m, hi = P.box(clean)
    mt, ht = torch.from_numpy(m), torch.from_numpy(hi)
    lo0 = np.minimum(_above(mt).numpy(), hi)                          # exact box [next float after m, x]
    lo = _above(mt, margin, abs_margin).numpy()
    hi2 = np.where(hi < P.cap, _below(ht, margin, abs_margin).numpy(), hi)      # the caps already carry their slack
    tight = lo > hi2
    lo, hi2 = np.where(tight, lo0, lo), np.where(tight, hi, hi2)
    cur = S1.det_tau.cpu().numpy()
    with np.errstate(invalid="ignore"):
        mid = np.where(np.isneginf(lo), hi2, 0.5 * (lo + hi2))
    tau = {"nearest": np.clip(cur, lo, hi2), "lo": lo, "hi": hi2, "mid": mid}[rule]
    val, clean2, unc = P.objective(tau, lam)
    if val != info["value_int"]:
        raise RuntimeError("thresholds do not reproduce the min-cut optimum")
    rec = {"op": "scope_cut", "lam": lam, "rule": rule, "margin": margin, "n_rows": int(P.N), "n_witnesses": int(P.M),
           "n_rows_never_clean": int((~P.feasible).sum()), "n_witnesses_always_covered": int((~P.wact).sum()),
           "n_clean": int(clean2.sum()), "clean_weight": float(P.row_w[clean2].sum()),
           "n_uncovered": int(unc.sum()), "uncovered_weight": float(P.wit_w[unc].sum()),
           "objective": float(P.row_w[clean2].sum() - lam * P.wit_w[unc].sum()), "value_int": info["value_int"], "scale": P.scale,
           "abs_margin": abs_margin, "n_guards_not_firing_before": int((~S1.guard_ok()).sum()),      # the caps make them fire
           "n_thresholds_changed": int((tau != cur).sum()), "n_raised": int((tau > cur).sum()), "n_lowered": int((tau < cur).sum()),
           "flow": info, "tau_lo": lo.tolist(), "tau_hi": hi2.tolist(), "n_tight": int(tight.sum()), "eta_folded": S.eta}
    return S1.replace(det_tau=torch.from_numpy(np.asarray(tau, dtype=np.float64)).to(S.device)), rec


# ----------------------------------------------------------------------------- session sugar
class Session:
    """A repair session: the current scoped log, the benign queries recorded so far, and every operator record.
    complain() = least_repair (the complaint's keys become a recorded benign query);
    why_not() = witness_repair checked against the recorded benign queries."""

    def __init__(self, S: ScopedLog):
        self.S = S
        self.benign = []                                             # one (n_q, d) fp64 tensor per recorded benign query
        self.history = []

    def complain(self, keys, **kw):
        new, rec = least_repair(self.S, keys, **kw)
        self.history.append(rec)
        if not rec["refused"]:
            self.S = new
            self.benign += _queries(new, keys)
        return rec

    def why_not(self, edit, key, mode="threshold", **kw):
        new, rec = witness_repair(self.S, edit, key, mode, benign_keys=self.benign, **kw)
        self.history.append(rec)
        self.S = new
        return rec
