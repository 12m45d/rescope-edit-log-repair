#!/usr/bin/env python3
"""Check that the pseudo-code and the statements of Sec. 4 (sec_method.tex, alg_calibrate.tex) and of Algorithm 2
(alg_complaints.tex) describe what the library 02_research/code/rescope does.

Revision of 2026-11-11 (review round 1): Algorithm 2 (the complaint operators beyond the least repair) and its text
moved to the technical report (app_rescope.tex, Sec. app:operators, label alg:app:repair); it is still checked here,
line by line. Its line 12 now recalibrates on a FRESH calibration sample (PROTOCOL.md, amendment "Independence of the
recalibration sample"): part A2l draws a new sample for every batch. The least repair of Sec. 4 (paragraph "The
repair") is the operator Raise of part A2r.

Revision of 2026-11-11, evening (external referee read of Secs. 2-4; CHECK_LOG.tex, part VI): Algorithm alg:calib now
takes GUARD KEYS as input and has one index per edit; its line 7 is unchanged in meaning (part A1 re-implements it as
before: every guard key of an edit at which the edit does not fire caps the threshold just below its coordinate). The
sentence on the successor of the rounded sum (part S) moved to the technical report (Remark rem:app:format). New part
A3 checks the sentences that the revision added to Sec. 4 about the implementation: top-1 selection (largest margin
zeta - b, ties to the earliest edit; the selected edit is not monotone in the thresholds), the log-only rule of the
migration (one shared threshold = the largest coordinate of an edit on the stored solver key of another edit; undefined
for T = 1), m >= 2, and the refusal condition zeta_i(q) + xi >= zeta_i(k) for a guard key k at which edit i fires.

    $RESCOPE_PYTHON scripts/check_method_pseudocode.py

Every part re-implements the paper's lines LITERALLY (plain Python floats, own bisection over the doubles, own closed
forms in numpy) and compares the result with the library operator, on synthetic logs written by a sequential
residual-form solver with batches (loglib.synthetic_log). CPU only, at most 4 threads, nothing is written outside this
directory (sys.dont_write_bytecode; the library is only imported). Output: scripts/check_method_pseudocode.json.

  L   Lemma lem:rebase: both identities against the ground truth of the solver; negative control (earlier EDITS instead
      of earlier batches); the library's rebased values = the first identity.
  P   Prop. prop:retrofit: thresholds in (1 - sep_i, 1 - eps_i]: exactly the own edit fires at every solver key and
      the module returns the target; part (c).
  A1  Algorithm alg:calib, lines 1-7, against calibrate_two_block: baselines, eta (exact rank), stored thresholds
      (bitwise), capped edits; the stored comparison equals the calibrated one on every calibration coordinate; at most
      floor(alpha (m_2 + 1)) queries of block 2 fire under the uncapped rule.
  S   The two sentences on the stored threshold (line 6), on random doubles, without the library: under the successor
      of the ROUNDED SUM b + eta the query that defines eta can fire; under the successor of the largest a with
      a (-) b <= eta it never does, and the stored comparison equals the calibrated one on the neighbouring doubles;
      rounded subtraction of a constant is monotone.
  A2r Raise (Algorithm alg:repair, line 3 and last line) against least_repair: thresholds (bitwise), refusals, order
      independence.
  A2w Why-not branch (lines 5-10) against witness_repair through Session: lowered thresholds (bitwise), added cosine
      detectors, refusals, must-cover pairs.
  A2n Line 11 against null_repair: the paper's closed form (projection, corrected along the projected own key), an
      independent formula (minimum-change solution of the linear system, technical report, Prop. prop:app:halfreal(b)),
      and the library; repeating the repair with more keys = repairing once with all keys.
  A2l The loop (lines 1-12, composed from the library's operators as the pseudo-code prescribes): after every batch all
      must-cover pairs are covered, all accepted complaints are certified, and no uncapped threshold is below its
      calibrated value.
  info  (not pass / fail) line 11 when a reported key approaches the own key of an edit: norm of the repaired detector, and
      the scale eta and the number of capped edits after the next calibration.
The guard band (epsilon of the paper) is read from the library: scoped.TIE for the caps of Algorithm alg:calib,
repair.MARGIN / ABS_MARGIN for Raise and for lowered thresholds.
"""
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP
import json
import math
import os
import struct
import sys
from fractions import Fraction

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _RP.ROOT
sys.path.insert(0, os.path.join(ROOT, "02_research", "code"))

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402

torch.set_num_threads(4)
from rescope import calibrate as CA, loglib as LL, repair as RP            # noqa: E402
from rescope.scoped import TIE, ScopedLog                                  # noqa: E402

INF = math.inf


# ----------------------------------------------------------------------------------------------- doubles as a chain
def f2o(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)


def o2f(o):
    return struct.unpack("<d", struct.pack("<Q", o if o >= 0 else ((-o) | (1 << 63))))[0]


def largest_with(pred):
    """the largest double a (possibly -inf) with pred(a), for a predicate that is true at -inf, false at +inf and monotone"""
    lo, hi = f2o(-INF), f2o(INF)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pred(o2f(mid)):
            lo = mid
        else:
            hi = mid
    return o2f(lo)


def succ(x):
    return math.nextafter(x, INF)


def above(x):
    """(x + eps)^+ of Raise: eps = repair.MARGIN |x| + repair.ABS_MARGIN, added in the library's order of operations"""
    return x if x == -INF else succ(x + RP.MARGIN * abs(x) + RP.ABS_MARGIN)


def below(x):
    """x - eps of a lowered threshold (same band, the library's order of operations)"""
    return x - RP.MARGIN * abs(x) - RP.ABS_MARGIN


def cap(x):
    """x - eps of the guard of Algorithm alg:calib: eps = scoped.TIE |x|"""
    return x - TIE * abs(x)


# ----------------------------------------------------------------------------------------------- data
def make(seed, T=24, d=96, n_out=20, batch=4):
    log, truth = LL.synthetic_log(T=T, d=d, m=n_out, batch=batch, seed=seed)
    g = torch.Generator().manual_seed(1000 + seed)
    base = log.K.mean(0)
    return log, truth, g, base


def benign_queries(log, g, base, n, spread=1.0):
    """n queries of 1-3 keys each: the common direction of the log's keys plus noise (like the log's own keys)"""
    out = []
    for _ in range(n):
        nk = int(torch.randint(1, 4, (1,), generator=g))
        out.append(base[None, :] + spread * torch.randn(nk, log.d, generator=g, dtype=torch.float64))
    return out


def served_keys(log, g, noise=0.03):
    """one 'served' key per edit: the solver key plus a small perturbation (own coordinate no longer exactly 1)"""
    return log.K + noise * torch.randn(log.K.shape, generator=g, dtype=torch.float64)


# ----------------------------------------------------------------------------------------------- L, P
def check_lemma(seed):
    log, truth, _, _ = make(seed)
    U, Psi, K, bt = log.U.numpy(), log.W.numpy(), log.K.numpy(), log.batch.numpy()
    W0, WE, Z = truth["W0"].numpy(), truth["WE"].numpy(), truth["Z"].numpy()
    T = log.T
    s = (Psi * K).sum(1)
    err1 = err2 = errlib = 0.0
    neg = 0.0
    lib = LL.rebase(log).numpy()
    for i in range(T):
        vrb = U[i] + sum((Psi[j] @ K[i]) * U[j] for j in range(T) if bt[j] < bt[i])
        tgt = Z[i] - W0 @ K[i]
        sc = np.linalg.norm(tgt)
        err1 = max(err1, np.linalg.norm(vrb - tgt) / sc)
        errlib = max(errlib, np.linalg.norm(lib[i] - vrb) / sc)
        rhs = (s[i] - 1) * U[i] + sum((Psi[j] @ K[i]) * U[j] for j in range(T) if bt[j] >= bt[i] and j != i)
        err2 = max(err2, np.linalg.norm(WE @ K[i] - Z[i] - rhs) / np.linalg.norm(Z[i]))
        wrong = U[i] + sum((Psi[j] @ K[i]) * U[j] for j in range(i))                 # earlier EDITS: the wrong sum
        neg = max(neg, np.linalg.norm(wrong - tgt) / sc)
    return {"seed": seed, "first_identity_max_rel_err": err1, "second_identity_max_rel_err": err2,
            "library_rebase_vs_first_identity": errlib, "negative_control_earlier_edits_max_rel_err": neg,
            "ok": bool(err1 < 1e-9 and err2 < 1e-9 and errlib < 1e-9 and neg > 1e-3)}


def check_retrofit(seed):
    log, truth, g, _ = make(seed)
    T = log.T
    R = log.crosstalk().numpy()                                    # R[j, i] = zeta_i(k_j)
    off = R - np.diag(np.diag(R)) - np.eye(T) * 1e300
    sep = 1 - off.max(0)                                           # sep_i = 1 - max_{j != i} zeta_i(k_j)
    if not (sep > 0).all():
        return {"seed": seed, "skipped": "an edit is not self-separated", "ok": True}
    eps = sep / 2
    u = torch.rand(T, generator=g, dtype=torch.float64).numpy()
    thr = (1 - sep) + (sep - eps) * (0.05 + 0.95 * u)             # in (1 - sep_i, 1 - eps_i]
    S = ScopedLog.from_log(log, role="last", fire="all", response="repl", tau=torch.from_numpy(thr), values="rebased")
    fires = S.fires_on(log.K).numpy()                              # [key of edit i, detector j]
    W0 = truth["W0"]
    x = log.K[:, None, :]
    out = S.apply(x, x @ W0.T, lp=torch.zeros(T, dtype=torch.long))
    err = float(((out[:, 0, :] - truth["Z"]).norm(dim=1) / truth["Z"].norm(dim=1)).max())
    # (c): keys with zeta_i >= 1 - eps_i fire edit i
    kk = log.K + 0.02 * torch.randn(log.K.shape, generator=g, dtype=torch.float64)
    z = S.scores(kk).numpy()
    c_ok = all((not z[i, i] >= 1 - eps[i]) or bool(S.fires_on(kk[i:i + 1])[0, i]) for i in range(T))
    return {"seed": seed, "min_separation_margin": float(sep.min()), "exactly_own_edit_fires_at_solver_keys": bool((fires == np.eye(T, dtype=bool)).all()),
            "target_attained_max_rel_err": err, "part_c": bool(c_ok),
            "ok": bool((fires == np.eye(T, dtype=bool)).all() and err < 1e-9 and c_ok)}


# ----------------------------------------------------------------------------------------------- A1
def paper_calibrate(S, queries, alpha, seed, guards):
    """Algorithm alg:calib, lines 1-7, literally. guards: list of (edit, key tensor (1, d)). -> dict"""
    Zq = S.query_max(queries).numpy()                                               # line 1 (the gate's own arithmetic)
    m, D = Zq.shape
    idx1, idx2 = CA.split_blocks(m, (1, 1), seed)                                   # line 2 (the library's random draw)
    assert len(idx1) == m // 2 and len(idx2) == m - m // 2
    b = [max(float(Zq[q, j]) for q in idx1) for j in range(D)]                      # line 3
    sc = sorted(max(float(Zq[q, j]) - b[j] for j in range(D)) for q in idx2)        # line 4
    m2 = len(idx2)
    rank = math.ceil((1 - Fraction(alpha).limit_denominator(10 ** 9)) * (m2 + 1))   # line 5: exact rank
    eta = sc[rank - 1] if rank <= m2 else INF
    if eta == INF:
        tau = [INF] * D
    else:
        tau = [succ(largest_with(lambda a, j=j: a - b[j] <= eta)) for j in range(D)]   # line 6
    literal = list(tau)
    det_edit = S.det_edit.tolist()
    capped = set()
    for e, k in guards:                                                             # line 7
        r = S.key_scores(k)[0].tolist()
        js = [j for j in range(D) if det_edit[j] == e]
        if not any(r[j] >= tau[j] for j in js):
            j = min(js, key=lambda j: tau[j] - r[j])
            tau[j] = cap(r[j])
            capped.add(e)
    return {"Zq": Zq, "b": b, "eta": eta, "rank": rank, "tau": tau, "literal": literal, "capped": sorted(capped), "idx2": idx2}


def check_calibration(seed, alpha, m=60, noise=0.03):
    log, _, g, base = make(seed)
    S0 = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
    sk = served_keys(log, g, noise)                                                 # large noise: some own requests need the guard
    S0, _ = RP.add_guards(S0, sk, list(range(log.T)))                               # own requests as served: must-cover
    queries = benign_queries(log, g, base, m)
    S1, rec = CA.calibrate_two_block(S0, queries, alpha, seed=seed)
    guards = [(int(e), S0.guard_key[i:i + 1]) for i, e in enumerate(S0.guard_edit.tolist())]
    P = paper_calibrate(S0, queries, alpha, seed, guards)
    tau_lib = S1.det_tau.tolist()
    same_tau = all((a == b_) for a, b_ in zip(P["tau"], tau_lib))
    same_b = S1.det_b is not None and all(a == b_ for a, b_ in zip(P["b"], S1.det_b.tolist()))
    # the stored comparison is the calibrated one, on every calibration coordinate (uncapped thresholds)
    Zq, b, eta, lit = P["Zq"], P["b"], P["eta"], P["literal"]
    stored_is_calibrated = all((float(Zq[q, j]) >= lit[j]) == (float(Zq[q, j]) - b[j] > eta)
                               for q in range(Zq.shape[0]) for j in range(Zq.shape[1]))
    m2 = len(P["idx2"])
    n_fire2 = sum(any(float(Zq[q, j]) >= lit[j] for j in range(Zq.shape[1])) for q in P["idx2"])
    allowed = math.floor(Fraction(alpha).limit_denominator(10 ** 9) * (m2 + 1))
    ok = bool(same_tau and same_b and P["eta"] == rec["eta"] and P["rank"] == rec["rank"] and P["capped"] == rec["capped_edits"]
              and stored_is_calibrated and n_fire2 <= allowed and S1.eta == 1.0 and bool(S1.guard_ok().all()))
    return {"seed": seed, "alpha": alpha, "m": m, "served_key_noise": noise, "m2": m2, "eta": P["eta"], "eta_infinite": P["eta"] == INF, "rank": P["rank"],
            "thresholds_bitwise_equal": same_tau, "baselines_equal": bool(same_b), "capped_edits_equal": P["capped"] == rec["capped_edits"],
            "n_capped_edits": len(P["capped"]), "stored_comparison_is_calibrated": stored_is_calibrated,
            "block2_queries_firing_uncapped": n_fire2, "allowed": allowed, "all_must_cover_pairs_covered": bool(S1.guard_ok().all()), "ok": ok}


def check_stored(seed, n=20000):
    rng = np.random.default_rng(seed)
    r = rng.uniform(-0.2, 1.2, n).tolist()                       # coordinate of the query that defines eta
    b = rng.uniform(-0.2, 1.2, n).tolist()                       # baseline of its detector
    naive = bad_def = bad_eq = 0
    for x, y in zip(r, b):
        eta = x - y                                              # the computed margin
        naive += int(x >= succ(y + eta))                         # threshold = successor of the rounded sum: fires?
        thr = succ(largest_with(lambda a: a - y <= eta))         # line 6
        bad_def += int(x >= thr)
        z = thr
        for _ in range(4):
            z = math.nextafter(z, -INF)
        for _ in range(9):                                       # four doubles below the threshold to four above
            bad_eq += int((z >= thr) != (z - y > eta))
            z = succ(z)
    xs = np.sort(rng.uniform(-2, 2, 200000))
    mono = bool(np.all(np.diff(xs - 0.3712345678901234) >= 0))
    return {"seed": seed, "n": n, "defining_query_fires_under_successor_of_rounded_sum": naive,
            "defining_query_fires_under_line_6": bad_def, "stored_differs_from_calibrated_on_neighbours": bad_eq,
            "rounded_subtraction_monotone": mono, "ok": bool(naive > 0 and bad_def == 0 and bad_eq == 0 and mono)}


# ----------------------------------------------------------------------------------------------- A2: Raise, why-not
class PaperState:
    """Thresholds of Algorithm alg:repair in plain Python floats, next to the library's scoped log (which is used here only
    to compute the coordinates of the detectors: the gate's own arithmetic)."""

    def __init__(self, S):
        self.S = S.fold()
        self.tau = self.S.det_tau.tolist()
        self.det_edit = self.S.det_edit.tolist()
        self.must = [(int(e), self.S.guard_key[i:i + 1]) for i, e in enumerate(self.S.guard_edit.tolist())]
        self.must_r = [self.S.key_scores(k)[0].tolist() for _, k in self.must]

    def coords(self, keys):
        """zeta_j(q) for every detector j: the largest coordinate over the keys of the query"""
        return self.S.query_max([keys])[0].tolist()

    def covered(self, tau):
        return [any(r[j] >= tau[j] for j in range(len(tau)) if self.det_edit[j] == e) for (e, _), r in zip(self.must, self.must_r)]

    def raise_(self, keys):
        """Raise(q): tau_j <- max(tau_j, (zeta_j(q) + eps)^+); refused (nothing changes) if a must-cover pair loses coverage"""
        z = self.coords(keys)
        new = [max(t, above(x)) for t, x in zip(self.tau, z)]
        if any(b_ and not a for b_, a in zip(self.covered(self.tau), self.covered(new))):
            return False
        self.tau = new
        return True

    def certified(self, keys):
        return all(x < t for x, t in zip(self.coords(keys), self.tau))


def check_raise(seed, alpha=0.1, m=60, n_complaints=40):
    log, _, g, base = make(seed)
    S0 = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
    S0, _ = RP.add_guards(S0, served_keys(log, g), list(range(log.T)))
    S1, _ = CA.calibrate_two_block(S0, benign_queries(log, g, base, m), alpha, seed=seed)
    # complaints: benign-like queries, some of them next to an own key (no threshold separates these: they must be refused)
    comp = benign_queries(log, g, base, n_complaints, spread=1.3)
    for i in range(0, n_complaints, 8):
        e = int(torch.randint(0, log.T, (1,), generator=g))
        comp[i] = log.K[e:e + 1] + 0.01 * torch.randn(1, log.d, generator=g, dtype=torch.float64)
    P, ses = PaperState(S1), RP.Session(S1)
    same, acc_fwd, n_raised = True, [], 0
    for q in comp:
        acc = P.raise_(q)
        rec = ses.complain(q)
        same = same and (acc == (not rec["refused"])) and P.tau == ses.S.det_tau.tolist()
        acc_fwd.append(acc)
        n_raised += rec["n_raised"]
    P2 = PaperState(S1)                                          # the same complaints in reversed order
    acc_rev = [P2.raise_(q) for q in reversed(comp)][::-1]
    order_ok = P2.tau == P.tau and acc_rev == acc_fwd
    cert = all(P.certified(q) for q, a in zip(comp, acc_fwd) if a)
    cov = all(P.covered(P.tau))
    n_ref = len(comp) - sum(acc_fwd)
    return {"seed": seed, "n_complaints": n_complaints, "n_refused": n_ref, "n_thresholds_raised": n_raised,
            "thresholds_and_refusals_equal_to_least_repair": bool(same), "order_independent_thresholds_and_refusals": bool(order_ok),
            "accepted_complaints_certified": bool(cert), "must_cover_pairs_covered": bool(cov),
            "ok": bool(same and order_ok and cert and cov and n_ref > 0 and n_raised > 0)}


def check_whynot(seed, alpha=0.1, m=60, tau_new=0.95):
    log, _, g, base = make(seed)
    T = log.T
    S0 = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
    S0, _ = RP.add_guards(S0, served_keys(log, g), list(range(T)))
    S1, _ = CA.calibrate_two_block(S0, benign_queries(log, g, base, m), alpha, seed=seed)
    ses = RP.Session(S1)
    tau = S1.det_tau.tolist()                                   # the paper's state (plain floats)
    det_edit = S1.det_edit.tolist()
    cneg = []                                                   # accepted (certified) complaints
    for q in benign_queries(log, g, base, 25, spread=1.3):      # a few wrong-answer complaints first (one key each)
        q = q[:1]
        rec = ses.complain(q)
        if not rec["refused"]:
            cneg.append(q)
    tau = ses.S.det_tau.tolist()
    stats = {"already": 0, "lowered": 0, "added": 0, "refused": 0}
    same = True
    for t in range(3 * T):
        i = t % T
        # a rephrasing: the own key moved towards a random certified complaint or towards noise
        mix = float(torch.rand(1, generator=g)) * 0.9
        other = cneg[int(torch.randint(0, len(cneg), (1,), generator=g))] if cneg and t % 2 == 0 else \
            base[None, :] + torch.randn(1, log.d, generator=g, dtype=torch.float64)
        key = (1 - mix) * log.K[i:i + 1] + mix * other
        S = ses.S
        D = S.D
        r = S.key_scores(key)[0].tolist()
        js = [j for j in range(D) if det_edit[j] == i]
        # ---- the paper's lines 5-10
        if any(r[j] >= tau[j] for j in js):
            outcome = "already"
        else:
            j = min(js, key=lambda j: tau[j] - r[j])                              # line 6
            a = below(r[j])
            zc = [S.query_max([q])[0].tolist() for q in cneg]
            if all(z[j] < a for z in zc):                                         # line 7
                outcome, new_tau = "lowered", (j, a)
            else:
                kn = (key / key.norm()).numpy()[0]
                cosq = [max(float(kk @ kn) / float(np.linalg.norm(kk)) for kk in q.numpy()) for q in cneg]
                if not any(c >= tau_new for c in cosq):                           # line 8
                    outcome = "added"
                else:
                    outcome = "refused"                                           # line 9
        # ---- the library, as docs/USAGE.md composes it
        rec = ses.why_not(i, key, mode="threshold")
        if rec["conflict"]:
            rec = ses.why_not(i, key, mode="detector", gamma=tau_new)
        lib = "already" if rec["already_fired"] else ("refused" if rec["conflict"] else ("lowered" if rec["mode"] == "threshold" else "added"))
        same = same and (lib == outcome)
        if outcome == "lowered":
            tau[new_tau[0]] = new_tau[1]
        elif outcome == "added":
            tau.append(tau_new)
            det_edit.append(i)
        same = same and tau == ses.S.det_tau.tolist() and det_edit == ses.S.det_edit.tolist()
        stats[outcome] += 1
    S = ses.S
    cert = all(not bool((S.query_max([q]) >= S.det_tau[None, :]).any()) for q in cneg)
    return {"seed": seed, **{"n_" + k: v for k, v in stats.items()}, "outcomes_thresholds_detectors_equal_to_witness_repair": bool(same),
            "certified_complaints_stay_certified": bool(cert), "must_cover_pairs_covered": bool(S.guard_ok().all()),
            "n_must_cover_pairs": int(S.guard_key.shape[0]),
            "ok": bool(same and cert and bool(S.guard_ok().all()) and stats["lowered"] > 0 and stats["added"] + stats["refused"] > 0)}


# ----------------------------------------------------------------------------------------------- A2: line 11
def check_null(seed, n_keys=12):
    log, _, g, base = make(seed)
    S = ScopedLog.from_log(log, role="last", fire="top1", response="repl", tau=0.5)
    Kb = (base[None, :] + torch.randn(n_keys, log.d, generator=g, dtype=torch.float64))
    Sn, rec = RP.null_repair(S, Kb, ridge=0.0)
    Psi, K, s = log.W.numpy(), log.K.numpy(), log.own().numpy()
    B = Kb.numpy()
    Pn = np.eye(log.d) - np.linalg.pinv(B) @ B                    # projector onto the complement of the reported keys
    e1 = e2 = e3 = e4 = 0.0
    for j in range(log.T):
        pw, pk = Pn @ Psi[j], Pn @ K[j]
        new = pw + (s[j] - pw @ K[j]) / (pk @ K[j]) * pk          # the paper's closed form
        Y = np.concatenate([K[j:j + 1], B], 0).T                  # independent formula: closest solution of Y^T psi = pi
        pi = np.zeros(Y.shape[1])
        pi[0] = s[j]
        alt = Psi[j] + Y @ np.linalg.pinv(Y.T @ Y) @ (pi - Y.T @ Psi[j])
        nrm = np.linalg.norm(Psi[j])
        e1 = max(e1, np.linalg.norm(new - Sn.det_w[j].numpy()) / nrm)
        e2 = max(e2, np.linalg.norm(new - alt) / nrm)
        e3 = max(e3, float(np.abs(B @ new).max()) / (np.linalg.norm(B, axis=1).max() * nrm))
        e4 = max(e4, abs(new @ K[j] - s[j]) / abs(s[j]))
    half, _ = RP.null_repair(S, Kb[: n_keys // 2], ridge=0.0)     # nested: first half, then all keys == all keys at once
    twice, _ = RP.null_repair(half, Kb, ridge=0.0)
    e5 = float(((twice.det_w - Sn.det_w).norm(dim=1) / log.W.norm(dim=1)).max())
    # a detector whose own key lies in the span of the reported keys is kept
    Sk, rk = RP.null_repair(S, torch.cat([Kb, log.K[3:4]]), ridge=0.0)
    kept = bool(torch.equal(Sk.det_w[3], S.det_w[3])) and rk["infeasible"] == [3]
    ok = bool(max(e1, e2, e3, e4, e5) < 1e-8 and kept and rec["infeasible"] == [])
    return {"seed": seed, "paper_closed_form_vs_library": e1, "paper_closed_form_vs_minimum_change_solution": e2,
            "max_coordinate_on_reported_keys": e3, "own_coordinate_rel_err": e4, "nested_repairs_equal_one_repair": e5,
            "detector_with_own_key_in_span_is_kept": kept, "ok": ok}


# ----------------------------------------------------------------------------------------------- A2: the loop
def check_loop(seed, alpha=0.1, m=80, n_batches=3, per_batch=32):
    log, _, g, base = make(seed)
    T = log.T
    S = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
    S, _ = RP.add_guards(S, served_keys(log, g), list(range(T)))
    calib = benign_queries(log, g, base, m)
    S, rec = CA.calibrate_two_block(S, calib, alpha, seed=seed)
    cneg, inv, n_w, n_added = [], [], 0, 0

    def silent(Sx, q):
        return not bool((Sx.query_max([q]) >= Sx.fold().det_tau[None, :]).any())
    for bno in range(n_batches):
        ses = RP.Session(S)
        ses.benign = [q for q in cneg if silent(S, q)]              # the certified complaints of the earlier batches
        for t in range(per_batch):
            if t % 4 == 0:                                          # wrong-answer complaint next to an own key: to be refused
                e = int(torch.randint(0, T, (1,), generator=g))
                q = log.K[e:e + 1] + 0.01 * torch.randn(1, log.d, generator=g, dtype=torch.float64)
                cneg.append(q)
                ses.complain(q)
            elif t % 4 in (1, 2):                                   # ordinary wrong-answer complaint
                q = benign_queries(log, g, base, 1, spread=1.3)[0][:1]
                cneg.append(q)
                ses.complain(q)
            else:                                                   # why-not complaint: a rephrasing that leans on a certified complaint
                i = int(torch.randint(0, T, (1,), generator=g))
                other = ses.benign[int(torch.randint(0, len(ses.benign), (1,), generator=g))] if ses.benign else base[None, :]
                w = 0.3 + 0.6 * float(torch.rand(1, generator=g))  # strong mixtures force an added detector or a refusal
                key = (1 - w) * log.K[i:i + 1] + w * other
                r = ses.why_not(i, key, mode="threshold")
                if r["conflict"]:
                    r = ses.why_not(i, key, mode="detector", gamma=0.95)
                    n_added += int(r["applied"])
                n_w += int(r["applied"] or r["already_fired"])
        S = ses.S
        S, _ = RP.null_repair(S, torch.cat(cneg), ridge=0.0)                         # line 11: keys of ALL reported complaints
        fresh = benign_queries(log, g, base, m)                                     # line 12: a FRESH calibration sample
        S, rec = CA.calibrate_two_block(S, fresh, alpha, seed=seed)                  # line 12
        cal_tau = S.det_tau.clone()
        capped = set(rec["capped_edits"])
        accepted = []
        for q in cneg:                                                              # line 12: Raise again
            S, r = RP.least_repair(S, q)
            accepted.append(not r["refused"])
        cert = all(silent(S, q) for q, a in zip(cneg, accepted) if a)
        unc = [j for j in range(S.D) if int(S.det_edit[j]) not in capped]
        inv.append({"batch": bno, "n_recorded_complaints": len(cneg), "n_accepted": int(sum(accepted)), "n_detectors": int(S.D),
                    "n_capped_edits": len(capped), "must_cover_pairs_covered": bool(S.guard_ok().all()),
                    "accepted_complaints_certified": bool(cert),
                    "uncapped_thresholds_not_below_calibrated": bool((S.det_tau[unc] >= cal_tau[unc]).all())})
    ok = all(v["must_cover_pairs_covered"] and v["accepted_complaints_certified"] and v["uncapped_thresholds_not_below_calibrated"] for v in inv)
    return {"seed": seed, "batches": inv, "n_why_not_granted": n_w, "n_detectors_added": n_added, "ok": bool(ok)}


def info_near_own_key(seed, alpha=0.1, m=80):
    """Not a pass / fail check: what line 11 does to a detector when a reported key is close to its own key, and what the
    next calibration (line 12) then does. Reported: norm of the repaired detectors relative to the log's, eta, capped edits."""
    log, _, g, base = make(seed)
    T = log.T
    S = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
    S, _ = RP.add_guards(S, served_keys(log, g), list(range(T)))
    calib = benign_queries(log, g, base, m)
    S, rec0 = CA.calibrate_two_block(S, calib, alpha, seed=seed)
    generic = [benign_queries(log, g, base, 1, spread=1.3)[0][:1] for _ in range(8)]
    out = {"seed": seed, "before": {"eta": rec0["eta"], "n_capped_edits": rec0["n_edits_capped"]}}
    for rel in (1.0, 0.3, 0.1, 0.03, 0.01):
        noise = torch.randn(1, log.d, generator=g, dtype=torch.float64)
        near = log.K[5:6] + rel * noise                                             # a complaint next to the own key of edit 5
        Sn, rn = RP.null_repair(S, torch.cat(generic + [near]), ridge=0.0)
        S2, r2 = CA.calibrate_two_block(Sn, calib, alpha, seed=seed)
        ratio = (Sn.det_w.norm(dim=1) / S.det_w.norm(dim=1))
        cos = float(torch.nn.functional.cosine_similarity(near, log.K[5:6]))
        out[f"complaint_at_noise_{rel:g}"] = {"cosine_of_complaint_key_and_own_key": cos, "detector_norm_ratio_of_that_edit": float(ratio[5]),
                                              "detector_norm_ratio_median_of_the_others": float(ratio[torch.arange(T) != 5].median()),
                                              "eta_after_recalibration": r2["eta"], "n_capped_edits_after_recalibration": r2["n_edits_capped"]}
    Sn, rn = RP.null_repair(S, torch.cat(generic), ridge=0.0)
    S2, r2 = CA.calibrate_two_block(Sn, calib, alpha, seed=seed)
    out["generic_complaints_only"] = {"detector_norm_ratio_max": float((Sn.det_w.norm(dim=1) / S.det_w.norm(dim=1)).max()),
                                      "eta_after_recalibration": r2["eta"], "n_capped_edits_after_recalibration": r2["n_edits_capped"]}
    return out


# ----------------------------------------------------------------------------------------------- A3
def check_added_sentences(seed):
    """The sentences that the revision of 2026-11-11 (evening) added to Sec. 4, against the library (see the docstring)."""
    log, _, g, base = make(seed)
    T = log.T
    S = ScopedLog.from_log(log, role="last", fire="top1", response="repl", tau=0.0)
    S = S.replace(det_b=0.1 * torch.randn(T, generator=g, dtype=torch.float64))
    keys = torch.cat(benign_queries(log, g, base, 40)) + 0.5 * log.K[torch.randint(0, T, (1,), generator=g)]
    lp = torch.zeros(keys.shape[0], dtype=torch.long)
    r, b, tau = S.scores(keys).tolist(), S.det_b.tolist(), S.det_tau.tolist()

    def selected(row, thr):                                                  # the paper's rule, literally
        cand = [i for i in range(T) if row[i] >= thr[i]]
        return None if not cand else max(cand, key=lambda i: (row[i] - b[i], -i))
    f = S.firing(keys[:, None, :], lp=lp)
    lib = {int(n): int(e) for n, e in zip(f.idx.tolist(), f.edit.tolist())} if f is not None else {}
    top1_ok = all(lib.get(n) == selected(r[n], tau) for n in range(len(r)))
    # ties: all margins equal (b_i := the coordinate itself) -> the earliest candidate is selected
    k0 = keys[:1]
    St = S.replace(det_b=S.scores(k0)[0].clone(), det_tau=torch.full((T,), -INF, dtype=torch.float64))
    tie_ok = int(St.firing(k0[:, None, :], lp=lp[:1]).edit[0]) == 0
    # not monotone: raise the threshold of the selected edit above its coordinate -> another candidate is selected
    n0 = next((n for n in range(len(r)) if sum(r[n][i] >= tau[i] for i in range(T)) >= 2), None)
    mono = None
    if n0 is not None:                                                       # one key = one execution configuration: its own scores
        r0 = S.scores(keys[n0:n0 + 1])[0].tolist()
        i0 = selected(r0, tau)
        t2 = S.det_tau.clone()
        t2[i0] = succ(r0[i0])
        f2 = S.replace(det_tau=t2).firing(keys[n0:n0 + 1][:, None, :], lp=lp[:1])
        mono = f2 is not None and int(f2.edit[0]) != i0 and int(f2.edit[0]) == selected(r0, t2.tolist())
    # log-only rule of the migration: one shared threshold, the largest coordinate of an edit on another edit's solver key
    S0 = ScopedLog.from_log(log, role="last", fire="top1", response="repl")
    R = S0.scores(log.K).tolist()                                            # R[j][i] = zeta_i(k_j)
    theta = max(R[j][i] for i in range(T) for j in range(T) if i != j)
    Sl, _ = CA.calibrate_log_only(S0, q=1.0, allow_silence=True)
    logonly_ok = all(t == theta for t in Sl.det_tau.tolist())
    one, _ = LL.synthetic_log(T=1, d=log.d, m=log.U.shape[1], batch=1, seed=seed)
    try:
        CA.calibrate_log_only(ScopedLog.from_log(one, role="last"), q=1.0)
        t1_undefined = False
    except Exception:
        t1_undefined = True
    try:
        CA.two_block_rule(np.zeros((1, T)), 0.05)
        m1_refused = False
    except ValueError:
        m1_refused = True
    # refusal: zeta_i(q) + xi >= zeta_i(k) for a guard key k of an edit i that fires at k (library: least_repair, 'refuse')
    S1, _ = RP.add_guards(S0, served_keys(log, g), list(range(T)))
    S1, _ = CA.calibrate_two_block(S1, benign_queries(log, g, base, 60), 0.1, seed=seed)
    th, ge, gr = S1.det_tau.tolist(), S1.guard_edit.tolist(), S1.guard_scores().tolist()
    refusal_ok, n_ref = True, 0
    for n in range(24):
        q = benign_queries(log, g, base, 1, spread=1.3)[0] if n % 3 else log.K[n % T:n % T + 1] + 0.01 * torch.randn(1, log.d, generator=g, dtype=torch.float64)
        z = S1.query_max([q])[0].tolist()
        paper = any(th[i] <= gr[k][i] < above(z[i]) for k, i in enumerate(ge))          # zeta_i(k) <= zeta_i(q) + xi, on the doubles
        _, rec = RP.least_repair(S1, q)
        refusal_ok = refusal_ok and (paper == rec["refused"])
        n_ref += int(rec["refused"])
    ok = bool(top1_ok and tie_ok and mono is not False and logonly_ok and t1_undefined and m1_refused and refusal_ok and n_ref > 0)
    return {"seed": seed, "top1_is_largest_margin_among_candidates": bool(top1_ok), "ties_go_to_the_earliest_edit": bool(tie_ok),
            "selected_edit_not_monotone_in_thresholds": mono, "log_only_rule_of_the_migration_is_largest_cross_coordinate": bool(logonly_ok),
            "log_only_rule_undefined_for_T_1": t1_undefined, "two_block_rule_refuses_m_1": m1_refused,
            "refusal_condition_equals_library": bool(refusal_ok), "n_refused": n_ref, "ok": ok}


def code_md5():
    import hashlib
    lib = os.path.join(ROOT, "02_research", "code", "rescope")
    return {f: hashlib.md5(open(os.path.join(lib, f), "rb").read()).hexdigest() for f in ("loglib.py", "scoped.py", "calibrate.py", "repair.py")}


def main():
    out = {"_about": "Sec. 4 against the library rescope: see the docstring of scripts/check_method_pseudocode.py",
           "library": os.path.join(ROOT, "02_research", "code", "rescope"),
           "guard_band": {"caps_of_Algorithm_1_relative": TIE, "raise_and_lowered_thresholds_relative": RP.MARGIN,
                          "raise_and_lowered_thresholds_absolute": RP.ABS_MARGIN, "keys_of_uncaptured_passes_relative": RP.DECODE_MARGIN},
           "L_rebase": [check_lemma(s) for s in range(6)],
           "P_retrofit": [check_retrofit(s) for s in range(6)],
           "A1_calibration": [check_calibration(s, a, m, nz) for s in range(6)
                              for a, m, nz in ((0.1, 60, 0.03), (0.05, 200, 0.03), (0.05, 200, 0.6), (0.2, 31, 0.6), (0.05, 30, 0.03))],
           "S_stored_threshold": [check_stored(s) for s in range(3)],
           "A2_raise": [check_raise(s) for s in range(6)],
           "A2_why_not": [check_whynot(s) for s in range(6)],
           "A2_null": [check_null(s) for s in range(6)],
           "A2_loop": [check_loop(s) for s in range(4)],
           "A3_added_sentences": [check_added_sentences(s) for s in range(4)],
           "info_line_11_near_own_key": [info_near_own_key(s) for s in range(3)],
           "library_md5": code_md5()}
    parts = [k for k in out if k[0] in "LPAS"]
    out["summary"] = {k: f"{sum(r['ok'] for r in out[k])}/{len(out[k])}" for k in parts}
    out["all_pass"] = all(r["ok"] for k in parts for r in out[k]) and sum(r["n_detectors_added"] for r in out["A2_loop"]) > 0
    json.dump(out, open(os.path.join(HERE, "check_method_pseudocode.json"), "w"), indent=1)
    for k in parts:
        print(f"{k:16s} {out['summary'][k]}")
    print("ALL PASS" if out["all_pass"] else "FAILURES: see scripts/check_method_pseudocode.json")
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
