"""CPU unit tests, part 3: the exact split-conformal rules of calibrate.py (two-block rule, shared-scale rule, the
floating-point-safe thresholds, the exact rank), for every score feature.

  a17_two_block_naive     equality with a naive re-implementation (plain Python floats, the defining comparisons) on random
                          data including ties: baselines, eta, rank, thresholds, candidates, top-1, guard caps; exact
                          enumeration against the referee's exact rational false-firing probabilities.
  a18_two_block_validity  exchangeable synthetic scores, 2000 trials per cell: false firing <= floor(alpha (n2 + 1)) / (n2 + 1)
                          within 3 standard errors (continuous, heavy-tailed, discrete with ties, urn = exchangeable non-i.i.d.).
  a19_float_safe          stored decision == defining comparison bit for bit (the referee's check C1 re-run through the
                          library function, plus baselines b != 0, scales u != 1, extreme magnitudes); exact rank on a grid.
  a20_two_block_referee   the referee's scripts re-run in a scratch copy on the stored keys of the development logs: their
                          numbers reproduced, and the library's rule on the very same random halves.
Each test returns a dict with 'pass' and the key numbers; run_tests.py writes them to tests/out/."""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
from fractions import Fraction

import numpy as np
import torch

from rescope import baselines as BL, calibrate as CA, loglib as LL
from rescope.scoped import TIE, ScopedLog

REFEREE = f"{_RP.ROOT}/03_analysis/theory/round3_rescope/C_statistics/referee"
P11B = f"{_RP.ROOT}/02_research/experiments/08_rescope/p11_scopecut/out"
HERE = os.path.dirname(os.path.abspath(__file__))


def ident_log(D, m=3):
    """A log whose 'ncoord' feature is the key itself: W = K = identity (scores(x) == x exactly), so any synthetic score
    matrix can be fed to the package's gate as keys."""
    eye = torch.eye(D, dtype=torch.float64)
    return LL.EditLog(U=torch.zeros(D, m, dtype=torch.float64), W=eye, K=eye.clone(), batch=torch.zeros(D, dtype=torch.long), name=f"identity-{D}")


# ----------------------------------------------------------------------------- naive reference (plain Python floats)
def naive_rule(Q, alpha: Fraction, idx1, idx2, u):
    """The definition, literally: baselines from block 1, sorted block-2 scores, exact rank.  Q: list of lists of floats."""
    D = len(Q[0])
    b = [max(Q[q][j] for q in idx1) for j in range(D)] if len(idx1) else [0.0] * D
    sc = sorted(max((Q[q][j] - b[j]) / u[j] for j in range(D)) for q in idx2)
    n2 = len(idx2)
    k = math.ceil((1 - alpha) * (n2 + 1))                              # Fraction arithmetic: exact
    return b, (sc[k - 1] if k <= n2 else math.inf), k


def tau_is_threshold(tau, b, u, eta):
    """tau is THE stored threshold of the defining comparison (which is monotone in r): the comparison holds AT tau and fails
    at the float just below it.  (A stepping search from b + eta u is not a usable reference: on grid data the boundary can
    lie ~1e18 floats away from it, e.g. b = -1.5, eta = 1.5: every r < 1.1e-16 has fl(r - b) = 1.5 = eta.)"""
    if eta == math.inf:
        return tau == math.inf
    return (tau - b) / u > eta and not (math.nextafter(tau, -math.inf) - b) / u > eta


def naive_fire(r, b, u, eta, rank):
    """(candidates, top-1) of one key under the DEFINING comparison; None if there is no candidate."""
    cand = [j for j in range(len(r)) if (r[j] - b[j]) / u[j] > eta]
    if not cand:
        return None
    key = (lambda j: (r[j] - b[j]) / u[j]) if rank == "excess" else (lambda j: r[j])
    top = cand[0]
    for j in cand[1:]:
        if key(j) > key(top):                                           # strict: an exact tie stays with the lowest index
            top = j
    return cand, top


def lib_fire(S, X):
    """{row: (candidates, top-1)} of the package gate on keys X (n, D): one one-key sequence per row (role 'last')."""
    f = S.firing(X[:, None, :], torch.zeros(X.shape[0], dtype=torch.long))
    if f is None:
        return {}
    return {int(r): (f.cand[i].nonzero().flatten().tolist(), int(f.top[i])) for i, r in enumerate(f.row)}


# ----------------------------------------------------------------------------- a17
def test_two_block_naive():
    rng = np.random.default_rng(20261110)
    out = {"n_trials": 0, "rule_equal": 0, "tau_is_threshold": 0, "decisions_equal": 0, "n_keys": 0, "n_keys_firing": 0, "n_keys_two_or_more_candidates": 0,
           "n_keys_exact_tie_in_rank": 0, "n_keys_tied_with_eta": 0, "tied_with_eta_firing": 0, "caps_equal": 0, "n_cap_trials": 0, "n_edits_capped_total": 0,
           "shared_rule_equal": 0, "n_shared_trials": 0, "eta_infinite_trials": 0, "by_data": {}, "n_top1_literal_excess_checked": 0,
           "n_top1_literal_excess_equal": 0, "n_top1_literal_excess_differs_at_exact_margin_tie": 0}
    alphas = [Fraction(1, 20), Fraction(1, 10), Fraction(1, 5), Fraction(1, 3), Fraction(3, 10), Fraction(1, 2), Fraction(1, 100)]
    for trial in range(400):
        D = int(rng.choice([1, 2, 3, 5, 8]))
        n = int(rng.choice([6, 9, 14, 21, 40, 60]))
        kind = ("continuous", "grid_quarter", "small_integers", "few_atoms")[trial % 4]
        if kind == "continuous":
            Q = rng.normal(size=(n, D)) + 0.5 * rng.normal(size=(n, 1))
            F = rng.normal(size=(30, D)) + 0.5 * rng.normal(size=(30, 1))
        elif kind == "grid_quarter":                                    # multiples of 1/4: every difference is exact, ties everywhere
            Q, F = rng.integers(-6, 7, size=(n, D)) / 4.0, rng.integers(-6, 9, size=(30, D)) / 4.0
        elif kind == "small_integers":
            Q, F = rng.integers(0, 4, size=(n, D)).astype(float), rng.integers(0, 6, size=(30, D)).astype(float)
        else:
            atoms = np.array([0.3, 0.4, 0.5, 0.9])
            Q, F = atoms[rng.integers(0, 4, size=(n, D))], atoms[rng.integers(0, 4, size=(30, D))]
        split = [(1, 1), (1, 3), (3, 1), (1, 1)][trial % 4]
        a = alphas[trial % len(alphas)]
        u = None if trial % 3 else np.array([0.5, 1.0, 2.0, 4.0])[rng.integers(0, 4, size=D)]      # powers of two keep grid ties exact
        if trial % 7 == 3:
            u = rng.uniform(0.2, 3.0, size=D)
        rank = ("excess", "r")[(trial // 2) % 2]
        seed = int(rng.integers(0, 10 ** 6))
        S0 = ScopedLog.from_log(ident_log(D))
        Qt = torch.from_numpy(Q)
        S, rec = CA.calibrate_two_block(S0, list(Qt[:, None, :]) if trial % 5 == 0 else None, a if trial % 2 else float(a),
                                        coords=None if trial % 5 == 0 else Qt, split=split, seed=seed, u=u, rank=rank, allow_silence=True)
        idx1, idx2 = CA.split_blocks(n, split, seed)
        ul = [1.0] * D if u is None else [float(x) for x in u]
        b, eta, k = naive_rule(Q.tolist(), a, idx1.tolist(), idx2.tolist(), ul)
        tau = S.det_tau.tolist()                                        # the library's thresholds, verified by their defining property
        ok_rule = (rec["block1"] == idx1.tolist() and rec["block2"] == idx2.tolist() and rec["rank"] == k and rec["n1"] == len(idx1) and rec["n2"] == len(idx2)
                   and (rec["eta"] == eta) and rec["eta_infinite"] == (eta == math.inf)
                   and rec["n_allowed_above"] == (0 if k > len(idx2) else len(idx2) + 1 - k) and S.eta == 1.0
                   and (rank == "r" or S.det_b.tolist() == b) and (rank == "excess" or (S.det_b is None and S.det_u is None)))
        ok_tau = all(tau_is_threshold(tau[j], b[j], ul[j], eta) for j in range(D))
        # keys: fresh rows, every calibration row (ties with eta), and one row per detector AT / just below / just above its threshold
        X = [F, Q]
        if eta != math.inf:
            base = np.array(b) - 3.0 * np.array(ul)                     # far below every threshold
            for j in range(D):
                for t in (tau[j], math.nextafter(tau[j], -math.inf), math.nextafter(tau[j], math.inf)):
                    row = base.copy()
                    row[j] = t
                    X.append(row[None])
        X = np.concatenate(X)
        got = lib_fire(S, torch.from_numpy(X))
        same, n_fire, n_multi, n_tie = True, 0, 0, 0
        for i, r in enumerate(X.tolist()):
            ref = naive_fire(r, b, ul, eta, rank)
            if ref is None:
                same = same and i not in got
                continue
            n_fire += 1
            n_multi += int(len(ref[0]) > 1)
            keyv = [((r[j] - b[j]) / ul[j]) if rank == "excess" else r[j] for j in ref[0]]
            n_tie += int(len(ref[0]) > 1 and sorted(keyv)[-1] == sorted(keyv)[-2])
            same = same and got.get(i) == (ref[0], ref[1])
            if rank == "excess" and len(ref[0]) > 1:                    # the LITERAL excess (r - b - eta u) / u orders the candidates the same way
                ex_ = [(r[j] - b[j] - eta * ul[j]) / ul[j] for j in ref[0]]
                lit = ref[0][max(range(len(ex_)), key=lambda q: (ex_[q], -q))]
                out["n_top1_literal_excess_checked"] += 1
                out["n_top1_literal_excess_equal"] += int(lit == ref[1])
                out["n_top1_literal_excess_differs_at_exact_margin_tie"] += int(lit != ref[1] and sorted(keyv)[-1] == sorted(keyv)[-2])
        # calibration rows whose score equals eta (the eta-defining query and its ties) must not fire: the strict event
        sc_all = [max((r[j] - b[j]) / ul[j] for j in range(D)) for r in Q.tolist()]
        tied = [i for i, v in enumerate(sc_all) if v == eta]
        out["n_keys_tied_with_eta"] += len(tied)
        out["tied_with_eta_firing"] += sum((F.shape[0] + i) in got for i in tied)
        out["n_trials"] += 1
        out["rule_equal"] += int(ok_rule)
        out["tau_is_threshold"] += int(ok_tau)
        out["decisions_equal"] += int(same)
        out["n_keys"] += len(X)
        out["n_keys_firing"] += n_fire
        out["n_keys_two_or_more_candidates"] += n_multi
        out["n_keys_exact_tie_in_rank"] += n_tie
        out["eta_infinite_trials"] += int(eta == math.inf)
        d = out["by_data"].setdefault(kind, {"trials": 0, "keys_firing": 0, "exact_rank_ties": 0, "all_equal": 0})
        d["trials"] += 1
        d["keys_firing"] += n_fire
        d["exact_rank_ties"] += n_tie
        d["all_equal"] += int(ok_rule and ok_tau and same)
        # ---- guard caps: random guard keys; the literal thresholds silence some of them
        if trial % 4 == 0:
            G = rng.normal(size=(2 * D, D)) + 0.5 if kind == "continuous" else rng.integers(-6, 7, size=(2 * D, D)) / 4.0
            ge = np.arange(2 * D) % D
            Sg = S0.replace(guard_key=torch.from_numpy(G), guard_edit=torch.from_numpy(ge))
            Sc, rc = CA.calibrate_two_block(Sg, coords=Qt, alpha=a, split=split, seed=seed, u=u, rank=rank)
            cap = list(tau)
            capped = set()
            for g in range(2 * D):
                e = int(ge[g])
                if not (G[g, e] - b[e]) / ul[e] > eta:                  # silenced by the literal rule (the defining comparison)
                    c = G[g, e] - TIE * abs(G[g, e])
                    if c < cap[e]:
                        cap[e] = c
                    capped.add(e)
            ok_cap = (Sc.det_tau.tolist() == cap and rc["n_edits_capped"] == len(capped) and rc["capped_edits"] == sorted(capped)
                      and rc["n_guards_silenced_by_literal_rule"] == sum(not (G[g, int(ge[g])] - b[int(ge[g])]) / ul[int(ge[g])] > eta for g in range(2 * D))
                      and rc["n_guards_silenced"] == 0 and bool(Sc.guard_ok().all()) and rc["capped"] == bool(capped)
                      and (rc["guarantee"] == "exact") == (not capped))
            out["n_cap_trials"] += 1
            out["caps_equal"] += int(ok_cap)
            out["n_edits_capped_total"] += len(capped)
        # ---- shared-scale rule: no block 1
        if trial % 4 == 1:
            Ss, rs = CA.calibrate_shared(S0, coords=Qt, alpha=a, u=u, allow_silence=True)
            b0, eta0, k0 = naive_rule(Q.tolist(), a, [], list(range(n)), ul)
            tau0 = Ss.det_tau.tolist()
            gots = lib_fire(Ss, torch.from_numpy(X))
            sames = all((gots.get(i) == (ref[0], ref[1])) if ref is not None else (i not in gots)
                        for i, ref in ((i, naive_fire(r, b0, ul, eta0, "r")) for i, r in enumerate(X.tolist())))
            out["n_shared_trials"] += 1
            out["shared_rule_equal"] += int(rs["eta"] == eta0 and rs["rank"] == k0 and all(tau_is_threshold(tau0[j], 0.0, ul[j], eta0) for j in range(D))
                                            and sames and rs["n1"] == 0 and rs["n2"] == n)
    # ---- exact enumeration against the referee's exact rational numbers (H_T family: coordinate 1 = the atoms, coordinate 2 = 0)
    ex = {"cases": 0, "equal": 0, "violations_of_bound": 0, "sources": {}}

    def enumerate_ff(atoms, probs, n1, n2, a):
        S0 = ScopedLog.from_log(ident_log(2))
        ff = Fraction(0)
        A = len(atoms)
        fa = [float(x) for x in atoms]
        blocks = (np.arange(n1), np.arange(n1, n1 + n2))
        import itertools
        for idx in itertools.product(range(A), repeat=n1 + n2):
            w = Fraction(1)
            for t in idx:
                w *= probs[t]
            Q = np.array([[fa[t], 0.0] for t in idx])
            S, _ = CA.calibrate_two_block(S0, coords=Q, alpha=a, blocks=blocks, allow_silence=True)
            fire = S.fires_on(torch.tensor([[x, 0.0] for x in fa], dtype=torch.float64)).any(1).tolist()
            ff += w * sum(probs[t] for t in range(A) if fire[t])
        return ff

    fr = lambda s: Fraction(s)                                                                    # noqa: E731
    with open(os.path.join(REFEREE, "rf2_c6_more.json")) as f:
        more = json.load(f)["1a_hybrid_on_H_T_exact"]
    with open(os.path.join(REFEREE, "rf2_c6_split_rule.json")) as f:
        split_ex = json.load(f)["1_split_rule_on_H_T_exact"]["examples"]
    two_point = [Fraction(1, 2), Fraction(1, 2)]                         # the law of the stored examples (rf2_c6_split_rule.py, laws[1])
    for src, cases in (("rf2_c6_more.json/1a", [(c["atoms"], c["probs"], c["n1"], c["n2"], c["alpha"], c["hybrid_FF_exact"], c["bound_l2/(n2+1)"]) for c in more]),
                       ("rf2_c6_split_rule.json/1", [(c["atoms"], [str(p) for p in two_point], c["n1"], c["n2"], c["alpha"], c["FF"], c["FF_bound"]) for c in split_ex])):
        s = ex["sources"].setdefault(src, {"cases": 0, "equal": 0})
        for atoms, probs, n1, n2, a, ff_ref, bound in cases:
            if len(atoms) ** (n1 + n2) > 7000:
                continue
            ff = enumerate_ff([fr(x) for x in atoms], [fr(p) for p in probs], n1, n2, fr(a))
            s["cases"] += 1
            s["equal"] += int(ff == fr(ff_ref))
            ex["cases"] += 1
            ex["equal"] += int(ff == fr(ff_ref))
            ex["violations_of_bound"] += int(ff > fr(bound))
    out["exact_enumeration_vs_referee"] = ex
    # ---- the same rules for a gate with one scalar score per query (calibrate_dense: router / z-score gates)
    dn = {"n": 0, "equal": 0}
    gate = types.SimpleNamespace(kind="router", thr=0.0, name="router")
    for trial in range(200):
        n = int(rng.choice([8, 20, 41]))
        sc = rng.integers(0, 6, size=n).astype(float) if trial % 2 else rng.normal(size=n)
        fresh = np.concatenate([sc, rng.integers(0, 8, size=30).astype(float) if trial % 2 else rng.normal(size=30)])
        a = alphas[trial % len(alphas)]
        seed = int(rng.integers(0, 10 ** 6))
        g1, r1 = CA.calibrate_dense(gate, sc, a)
        b0, eta0, k0 = naive_rule([[x] for x in sc.tolist()], a, [], list(range(n)), [1.0])
        ok = r1["eta"] == eta0 and r1["rank"] == k0 and np.array_equal(fresh >= g1.thr, fresh > eta0) and gate.thr == 0.0
        g2, r2 = CA.calibrate_dense(gate, sc, a, two_block=True, seed=seed)
        i1, i2 = CA.split_blocks(n, (1, 1), seed)
        b, eta, k = naive_rule([[x] for x in sc.tolist()], a, i1.tolist(), i2.tolist(), [1.0])
        ok = ok and r2["eta"] == eta and r2["baseline"] == b[0] and np.array_equal(fresh >= g2.thr, (fresh - b[0]) > eta)
        dn["n"] += 1
        dn["equal"] += int(ok)
    out["dense"] = dn
    n = out["n_trials"]
    out["pass"] = bool(out["rule_equal"] == n and out["tau_is_threshold"] == n and out["decisions_equal"] == n and out["tied_with_eta_firing"] == 0
                       and out["n_keys_tied_with_eta"] > 0 and out["n_keys_two_or_more_candidates"] > 100 and out["n_keys_exact_tie_in_rank"] > 20
                       and out["caps_equal"] == out["n_cap_trials"] > 0 and out["n_edits_capped_total"] > 0
                       and out["shared_rule_equal"] == out["n_shared_trials"] > 0 and out["eta_infinite_trials"] > 0
                       and ex["equal"] == ex["cases"] > 10 and ex["violations_of_bound"] == 0 and dn["equal"] == dn["n"]
                       # the literal excess may only disagree with the margin where two margins tie exactly (it then breaks the tie by the
                       # rounding of eta * u; the margin keeps the documented rule: lowest index)
                       and out["n_top1_literal_excess_checked"] - out["n_top1_literal_excess_equal"] == out["n_top1_literal_excess_differs_at_exact_margin_tie"])
    return out


# ----------------------------------------------------------------------------- a18
def _law(name, rng, N, T):
    if name == "gaussian, common factor":
        return rng.normal(size=(N, T)) + rng.normal(size=(N, 1))
    if name == "independent Student-t(2)":
        return rng.standard_t(2, size=(N, T))
    if name == "type mixture (near the union bound)":
        X = rng.random((N, T))
        X[np.arange(N), rng.integers(0, T, N)] += 1.0
        return X
    if name == "discrete with a common factor (ties)":
        return (rng.integers(0, 4, size=(N, 1)) + rng.integers(0, 3, size=(N, T))).astype(float)
    if name == "three atoms (heavy ties)":
        return np.array([0.3, 0.4, 0.5])[rng.integers(0, 3, size=(N, T))]
    if name == "Poisson(3) counts (ties)":
        return rng.poisson(3.0, size=(N, T)).astype(float)
    if name == "urn without replacement (exchangeable, not i.i.d.)":
        pool = (rng.integers(0, 5, size=(N // 3 + 1, 1)) + rng.integers(0, 3, size=(N // 3 + 1, T))).astype(float)    # every row three times
        return np.concatenate([pool, pool, pool])[rng.permutation(3 * (N // 3 + 1))[:N]]
    raise ValueError(name)


def test_two_block_validity():
    """Pre-specified criterion: in EVERY cell (2000 trials, 50 fresh queries per trial, drawn with the calibration queries
    from one exchangeable law) the mean false-firing rate of the uncapped rule is at most floor(alpha (n2 + 1)) / (n2 + 1)
    + 3 standard errors; where eta = +inf nothing fires; in every trial the package gate (fires_on) equals the strict
    conformal event 'fresh score > eta'.  The same for every score feature (normalised coordinate, raw coordinate, cosine, minus
    distance) on random keys through the package's own gate (2000 trials per feature, calibration from keys).
    Reported: z against the bound (for continuous laws the expectation EQUALS it)."""
    cells = [("gaussian, common factor", 20, 200, "1/20", (1, 1), None, "two_block"),
             ("gaussian, common factor", 20, 40, "1/20", (1, 1), None, "two_block"),
             ("gaussian, common factor", 20, 30, "1/20", (1, 1), None, "two_block"),             # n2 = 15 < 19: eta = +inf
             ("gaussian, common factor", 20, 200, "1/20", (1, 1), "random", "two_block"),
             ("independent Student-t(2)", 50, 200, "1/10", (1, 1), None, "two_block"),
             ("type mixture (near the union bound)", 50, 200, "1/20", (1, 1), None, "two_block"),
             ("type mixture (near the union bound)", 50, 100, "1/5", (1, 3), None, "two_block"),
             ("discrete with a common factor (ties)", 50, 200, "1/20", (1, 1), None, "two_block"),
             ("discrete with a common factor (ties)", 20, 60, "1/10", (3, 1), None, "two_block"),
             ("three atoms (heavy ties)", 3, 50, "1/5", (1, 1), None, "two_block"),
             ("urn without replacement (exchangeable, not i.i.d.)", 20, 100, "1/10", (1, 1), None, "two_block"),
             ("Poisson(3) counts (ties)", 5, 40, "1/5", (1, 1), None, "two_block"),
             ("Poisson(3) counts (ties)", 2, 20, "1/20", (1, 1), None, "two_block"),                # n2 = 10 < 19: eta = +inf
             ("Poisson(3) counts (ties)", 3, 60, "1/10", None, None, "shared"),
             ("gaussian, common factor", 20, 99, "1/20", None, None, "shared"),
             ("discrete with a common factor (ties)", 20, 100, "1/10", None, None, "shared"),
             ("independent Student-t(2)", 20, 100, "1/3", None, "random", "shared")]
    n_trials, m = 2000, 50
    out = {"cells": [], "n_trials": n_trials, "n_fresh_per_trial": m, "pass": True, "criterion": test_two_block_validity.__doc__}
    for ci, (law, T, n, alpha, split, umode, rule) in enumerate(cells):
        rng = np.random.default_rng([20261110, ci])
        S0 = ScopedLog.from_log(ident_log(T))
        a = Fraction(alpha)
        u = None if umode is None else rng.uniform(0.3, 3.0, size=T)       # fixed in advance (independent of the data)
        rates, same, n_inf, etas = np.zeros(n_trials), 0, 0, []
        bound = None
        for t in range(n_trials):
            X = _law(law, rng, n + m, T)
            Q, Fr = torch.from_numpy(X[:n]), torch.from_numpy(X[n:])
            if rule == "two_block":
                S, rec = CA.calibrate_two_block(S0, coords=Q, alpha=a, split=split, seed=int(rng.integers(0, 2 ** 31)), u=u, allow_silence=True)
            else:
                S, rec = CA.calibrate_shared(S0, coords=Q, alpha=a, u=u, allow_silence=True)
            fire = S.fires_on(Fr).any(1).numpy()                         # the package's gate on the fresh keys
            b = None if S.det_b is None else S.det_b.numpy()
            ref = CA.block_scores(X[n:], Q[rec["block1"]].max(0).values.numpy() if rule == "two_block" else None, u) > rec["eta"]
            same += int(np.array_equal(fire, ref) and (rule != "two_block" or np.array_equal(b, Q[rec["block1"]].max(0).values.numpy())))
            rates[t] = fire.mean()
            n_inf += int(rec["eta_infinite"])
            bound = rec["bound_false_firing"]
        se = float(rates.std(ddof=1) / math.sqrt(n_trials))
        cell = {"law": law, "T": T, "n": n, "alpha": alpha, "rule": rule, "split": None if split is None else list(split), "u": "1" if umode is None else "random fixed",
                "n1": rec["n1"], "n2": rec["n2"], "rank": rec["rank"], "bound": bound, "mean_false_firing_rate": float(rates.mean()), "se": se,
                "z_vs_bound": None if se == 0 else float((rates.mean() - bound) / se), "eta_infinite_trials": n_inf,
                "gate_equals_strict_event": [same, n_trials], "continuous": law in ("gaussian, common factor", "independent Student-t(2)",
                                                                                   "type mixture (near the union bound)")}
        cell["ok"] = bool(rates.mean() <= bound + 3 * se + 1e-15 and same == n_trials and (n_inf == 0 or (n_inf == n_trials and rates.max() == 0.0)))
        out["cells"].append(cell)
        out["pass"] = out["pass"] and cell["ok"]
    # ---- every score feature through the package's own gate: random keys in d dimensions, the calibration from KEYS (one key per
    #      query, as the hook scores a role-'last' sequence), the fresh keys scored one by one as well
    log, _ = LL.synthetic_log(T=8, d=24, m=4, batch=4, seed=5, l2=4.0)
    out["feature_cells"] = []
    for fi, feat in enumerate(BL.FEATURES):
        rng = np.random.default_rng([20261110, 100 + fi])
        S0 = BL.feature_gate(log, feat)
        n, a, n_tr, mf = 60, Fraction(1, 5), 2000, 20
        rates, same = np.zeros(n_tr), 0
        Kn = log.K.numpy()
        for t in range(n_tr):
            X = torch.from_numpy(0.6 * rng.normal(size=(n + mf, log.d)) + 0.5 * Kn[rng.integers(0, log.T, n + mf)])
            S, rec = CA.calibrate_two_block(S0, X[:n], a, seed=int(rng.integers(0, 2 ** 31)), allow_silence=True)
            R = S.key_scores(X[n:])                                      # one key per query: the hook's evaluation
            fire = (R >= S.theta()[None, :]).any(1).numpy()
            b = S0.query_max([X[i][None] for i in rec["block1"]]).max(0).values.numpy()
            same += int(np.array_equal(fire, CA.block_scores(R.numpy(), b, None) > rec["eta"]) and np.array_equal(b, S.det_b.numpy()))
            rates[t] = fire.mean()
        se = float(rates.std(ddof=1) / math.sqrt(n_tr))
        cell = {"feature": feat, "T": log.T, "d": log.d, "n": n, "alpha": "1/5", "n2": rec["n2"], "bound": rec["bound_false_firing"],
                "mean_false_firing_rate": float(rates.mean()), "se": se, "z_vs_bound": float((rates.mean() - rec["bound_false_firing"]) / se),
                "gate_equals_strict_event": [same, n_tr]}
        cell["ok"] = bool(rates.mean() <= rec["bound_false_firing"] + 3 * se and same == n_tr)
        out["feature_cells"].append(cell)
        out["pass"] = out["pass"] and cell["ok"]
    zc = [c["z_vs_bound"] for c in out["cells"] if c["continuous"] and c["z_vs_bound"] is not None]
    out["z_vs_bound_continuous"] = {"n_cells": len(zc), "mean": float(np.mean(zc)), "min": float(np.min(zc)), "max": float(np.max(zc))}
    out["max_excess_over_bound"] = max(c["mean_false_firing_rate"] - c["bound"] for c in out["cells"])
    out["n_cells_with_ties"] = sum(not c["continuous"] for c in out["cells"])
    return out


# ----------------------------------------------------------------------------- a19
def test_float_safe():
    out = {}
    # ---- 1. the referee's check (rf2_c1_float_safe_threshold.py, float64 loop, same seed and the same draws), with the LIBRARY's
    #         threshold in place of his theta_safe: stored gate 'r >= tau' against the ratio gate 'fl(r / tau0) > eta'
    rng = np.random.default_rng(113)
    dt = np.float64
    mism_safe = mism_naive = naive_fires = 0
    n_cases = 20000
    for _ in range(n_cases):
        t0 = dt(rng.uniform(0.05, 3.0))
        r_def = dt(rng.uniform(0.01, 3.0)) if rng.random() < 0.5 else dt(round(rng.uniform(0.01, 3.0), 2))
        eta = dt(r_def / t0)
        tau = CA.float_safe_threshold(np.zeros(1), np.array([t0]), eta)[0]
        naive = dt(eta * t0)
        cand = [r_def]
        x = r_def
        for _k in range(4):
            x = np.nextafter(x, dt(np.inf))
            cand.append(x)
        x = r_def
        for _k in range(4):
            x = np.nextafter(x, dt(-np.inf))
            cand.append(x)
        cand += [dt(v) for v in rng.uniform(0.0, 3.0, 4)]
        for r in cand:
            ratio_gate = bool(dt(r / t0) > eta)
            mism_safe += ratio_gate != bool(r >= tau)
            mism_naive += ratio_gate != bool(r > naive)
        naive_fires += bool(r_def > naive)
    with open(os.path.join(REFEREE, "rf2_c1_float_safe_threshold.json")) as f:
        ref = json.load(f)["float64"]
    out["referee_c1"] = {"cases": n_cases, "inputs_per_case": 13, "mismatches_library_threshold_vs_ratio_gate": int(mism_safe),
                         "mismatches_naive_product_threshold_vs_ratio_gate": int(mism_naive), "referee_naive_mismatches": ref["mismatches_naive_product_threshold_vs_ratio_gate"],
                         "referee_safe_mismatches": ref["mismatches_safe_threshold_vs_ratio_gate"],
                         "frac_defining_point_fires_under_naive": naive_fires / n_cases,
                         "referee_frac": ref["frac_cases_where_the_defining_calibration_point_fires_under_the_naive_threshold"]}
    # ---- 2. general form: baselines b != 0 and scales u != 1 (the two-block rule), numpy and torch arithmetic
    rng = np.random.default_rng(7)
    g = {"n_inputs": 0, "mismatch_numpy": 0, "mismatch_torch": 0, "mismatch_naive_product": 0, "defining_point_fires": 0, "defining_point_fires_naive_product": 0, "n_cases": 0}
    for trial in range(400):
        D = 64
        b = rng.normal(size=D) * rng.choice([1e-3, 1.0, 1e3, 1e6], size=D)
        u = np.ones(D) if trial % 3 == 0 else rng.uniform(0.05, 5.0, size=D)
        if trial % 4 == 1:                                              # two-decimal data, like the referee's counterexample
            b, u = np.round(b, 2), np.round(u, 2) + 0.01
        r_def = b + rng.normal(size=D) * rng.choice([1e-6, 1e-2, 1.0, 30.0], size=D)
        if trial % 4 == 1:
            r_def = np.round(r_def, 2)
        j = int(rng.integers(D))
        eta = float((r_def[j] - b[j]) / u[j])                           # eta is the score of a calibration point
        tau = CA.float_safe_threshold(b, u, eta)
        naive = np.nextafter(b + eta * u, np.inf)                       # succ(fl(b + eta u)): the product form
        R = [r_def + 0.0]
        for d in range(1, 5):
            up, dn = tau.copy(), tau.copy()
            for _ in range(d):
                up, dn = np.nextafter(up, np.inf), np.nextafter(dn, -np.inf)
            R += [up, dn]
        R += [tau, b + rng.normal(size=D) * 5, rng.normal(size=D)]
        R = np.stack(R)
        ref_np = (R - b[None]) / u[None] > eta
        Rt, bt, ut = torch.from_numpy(R), torch.from_numpy(b), torch.from_numpy(u)
        ref_t = ((Rt - bt[None]) / ut[None] > eta).numpy()
        g["n_inputs"] += R.size
        g["mismatch_numpy"] += int(((R >= tau[None]) != ref_np).sum())
        g["mismatch_torch"] += int(((Rt >= torch.from_numpy(tau)[None]).numpy() != ref_t).sum())
        g["mismatch_naive_product"] += int(((R >= naive[None]) != ref_np).sum())
        g["defining_point_fires"] += int(r_def[j] >= tau[j])
        g["defining_point_fires_naive_product"] += int(r_def[j] >= naive[j])
        g["n_cases"] += 1
    out["general"] = g
    # ---- 3. extreme magnitudes: |b| >> |r| (a stepping search would need ~1e14 nextafter steps), infinite baselines / eta
    ext = {"n": 0, "ok": 0, "seconds": 0.0}
    t0 = time.time()
    for b, u, r_def in ((-1e10, 1.0, 1e-5), (1e12, 3.0, 1e12 + 7.0), (-3e15, 0.25, 2.5), (0.0, 1e-300, 1e-290), (5e-320, 1.0, 9e-320), (-1e300, 1e-5, 1e299)):
        eta = float((np.float64(r_def) - b) / u)
        tau = CA.float_safe_threshold(np.array([b]), np.array([u]), eta)[0]
        ok = not (r_def >= tau)
        for r in (tau, np.nextafter(tau, -np.inf), np.nextafter(tau, np.inf), tau * 2 if tau > 0 else tau / 2, r_def):
            ok = ok and bool((r >= tau) == ((np.float64(r) - b) / u > eta))
        ext["n"] += 1
        ext["ok"] += int(ok)
    ext["seconds"] = time.time() - t0
    with np.errstate(invalid="ignore"):
        tinf = CA.float_safe_threshold(np.array([0.0, 1.0, -np.inf, np.inf]), np.ones(4), 0.5)
    ext["eta_inf_gives_inf"] = bool(np.isinf(CA.float_safe_threshold(np.zeros(3), np.ones(3), np.inf)).all())
    ext["b_minus_inf_every_finite_r_is_candidate"] = bool(tinf[2] == -np.finfo(np.float64).max)
    ext["b_plus_inf_no_finite_r_is_candidate"] = bool(tinf[3] == np.inf)
    try:
        CA.float_safe_threshold(np.array([np.nan]), np.ones(1), 0.0)
        ext["nan_refused"] = False
    except ValueError:
        ext["nan_refused"] = True
    out["extreme"] = ext
    # ---- 4. the exact rank against Fraction arithmetic on a grid, and how often the float formula is one rank off
    rk = {"n": 0, "equal_exact": 0, "float_formula_differs": 0, "float_input_equal": 0, "examples": []}
    for q in range(2, 41):
        for p in range(1, q):
            if math.gcd(p, q) != 1:
                continue
            a = Fraction(p, q)
            for n in list(range(1, 60)) + [99, 100, 199, 200, 244, 245, 489, 500, 999, 1000]:
                k = math.ceil((1 - a) * (n + 1))
                rk["n"] += 1
                rk["equal_exact"] += int(CA.conformal_rank_exact(n, a) == k == CA.conformal_rank_exact(n, f"{p}/{q}") == CA.conformal_rank_exact(n, (p, q)))
                rk["float_input_equal"] += int(CA.conformal_rank_exact(n, p / q) == k)       # a float alpha is read as the closest simple fraction
                kf = int(math.ceil((1 - p / q) * (n + 1)))
                if kf != k:
                    rk["float_formula_differs"] += 1
                    if len(rk["examples"]) < 4:
                        rk["examples"].append({"alpha": f"{p}/{q}", "n": n, "exact": k, "float": kf})
    rk["one_third_n8"] = CA.conformal_rank_exact(8, 1 / 3)
    out["rank"] = rk
    c1 = out["referee_c1"]
    out["pass"] = bool(c1["mismatches_library_threshold_vs_ratio_gate"] == 0 == c1["referee_safe_mismatches"]
                       and c1["mismatches_naive_product_threshold_vs_ratio_gate"] == c1["referee_naive_mismatches"] > 0
                       and abs(c1["frac_defining_point_fires_under_naive"] - c1["referee_frac"]) < 1e-12
                       and g["mismatch_numpy"] == 0 == g["mismatch_torch"] and g["defining_point_fires"] == 0 and g["mismatch_naive_product"] > 0
                       and ext["ok"] == ext["n"] and ext["eta_inf_gives_inf"] and ext["b_minus_inf_every_finite_r_is_candidate"]
                       and ext["b_plus_inf_no_finite_r_is_candidate"] and ext["nan_refused"] and ext["seconds"] < 5
                       and rk["equal_exact"] == rk["n"] == rk["float_input_equal"] and rk["one_third_n8"] == 6)
    return out


# ----------------------------------------------------------------------------- a20: the referee's reference numbers
_RUNNER = r'''
import os, runpy, sys
import numpy as np
_real, _rec = np.random.default_rng, []
class _Proxy:                                   # the referee's generator, unchanged; permutations of >= 300 rows are recorded
    def __init__(self, g):
        self._g = g
    def __getattr__(self, name):
        attr = getattr(self._g, name)
        if name != "permutation":
            return attr
        def permutation(x, *a, **k):
            out = attr(x, *a, **k)
            if hasattr(out, "__len__") and len(out) >= 300:
                _rec.append(np.asarray(out).copy())
            return out
        return permutation
np.random.default_rng = lambda *a, **k: _Proxy(_real(*a, **k))
sys.argv = [sys.argv[1]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    np.savez(sys.argv[0] + ".permutations.npz", **{f"p{i:05d}": p for i, p in enumerate(_rec)})
'''


def _rerun(script, scratch):
    """Run a COPY of a referee script in `scratch` (it writes its JSON next to itself) -> (its fresh JSON, recorded permutations)."""
    os.makedirs(scratch, exist_ok=True)
    dst = os.path.join(scratch, script)
    shutil.copy2(os.path.join(REFEREE, script), dst)
    runner = os.path.join(scratch, "_runner.py")
    with open(runner, "w") as f:
        f.write(_RUNNER)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    t0 = time.time()
    r = subprocess.run([sys.executable, runner, dst], capture_output=True, text=True, env=env, cwd=scratch)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed: {r.stderr[-2000:]}")
    with open(dst.replace(".py", ".json")) as f:
        fresh = json.load(f)
    with np.load(dst + ".permutations.npz") as z:
        perms = [z[k] for k in sorted(z.files)]
    os.remove(dst + ".permutations.npz")                               # reproducible; 20 MB
    return fresh, perms, time.time() - t0


def _p11_keys(name):
    kd = os.path.join(P11B, name, "keys")
    with open(os.path.join(kd, "meta.json")) as f:
        meta = json.load(f)
    fac = np.load(os.path.join(kd, "factors.npz"))
    K = np.load(os.path.join(kd, "K_last.npy")).astype(np.float64)
    W, Ks = fac["Wfac"].astype(np.float64), fac["K_solver"].astype(np.float64)
    T = W.shape[0]
    row = {(q["metric"], q["edit"]): q["n"] for q in meta["seqs"] if q["kind"] == "edit"}
    pick = lambda m: np.array([row[(m, i)] for i in range(T)]) if (m, 0) in row else None           # noqa: E731
    return {"C": K @ W.T, "s": np.einsum("ij,ij->i", Ks, W), "T": T, "K": K, "rel": pick("rel"), "tgen": pick("tgen"), "igen": pick("igen"),
            "fit": np.array([q["n"] for q in meta["seqs"] if q["kind"] == "probe" and q["pool"] == "repair"]),
            "test": np.array([q["n"] for q in meta["seqs"] if q["kind"] == "probe" and q["pool"] == "heldout"])}


def _same_numbers(a, b, tol=0.0):
    """Recursive equality of two JSON values (numbers within tol)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_numbers(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_numbers(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b) <= tol
    return a == b


def test_two_block_referee():
    """Pre-specified criterion.  The referee's scripts rf2_c6_more.py (200 random halves, alpha = 5 % and 20 %, three logs) and
    rf2_c6_checks.py (300 random halves, alpha = 5 / 10 / 20 %, the two development logs; the numbers quoted in PROTOCOL.md)
    are re-run UNCHANGED in a scratch copy; their generator is wrapped only to record the permutations it returns.  PASS iff
    (1) the re-run reproduces the stored numbers of the REAL section of both scripts exactly for the two development logs;
    (2) a literal transcription of the referee's rule (thresholds a + eta s, gate c > threshold) on the recorded
        permutations gives his means to 1e-12 (the permutations and the data are the ones he used);
    (3) the library's two-block rule (feature = raw coordinate, u = s, uncapped, the floating-point-safe thresholds) on the
        same permutations: every (resplit, row, edit) decision that differs from the transcription lies within 4 ulp of the
        product threshold, and the mean rows-firing / coverage numbers differ from the referee's by at most 1e-3;
    (4) likewise the library's shared-scale rule (u = s) against his 'shared conformal, tau0 = s';
    (5) the protocol's default form (feature = normalised coordinate, u = 1; the same rule in exact arithmetic) on the same
        permutations at alpha = 5 %: mean rows-firing / coverage within 1e-3 of the referee's (decisions that differ are counted).
    Reported: the numbers of both, the decisions that differ, the feature computed by the package (S.scores) against C."""
    scratch = os.path.join(HERE, "out", "referee_rerun")
    out = {"scripts": {}, "logs": {}, "criterion": test_two_block_referee.__doc__, "scratch": scratch}
    res = {}
    for script in ("rf2_c6_more.py", "rf2_c6_checks.py"):
        fresh, perms, sec = _rerun(script, scratch)
        with open(os.path.join(REFEREE, script.replace(".py", ".json"))) as f:
            stored = json.load(f)
        res[script] = (fresh, perms, stored)
        out["scripts"][script] = {"seconds": sec, "n_permutations_recorded": len(perms)}
    ok_all = True
    for name in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3", "TB-Q500-ae"):
        L = _p11_keys(name)
        C, s, T, fit, test = L["C"], L["s"], L["T"], L["fit"], L["test"]
        n, ar = len(fit), np.arange(T)
        log = LL.from_testbed(name)
        S0 = BL.feature_gate(log, "coord")
        Ct = S0.scores(torch.from_numpy(L["K"][np.concatenate([fit[:50], test[:50]])])).numpy()
        r = {"T": T, "n_fit": n, "n_test": len(test), "development_log": name != "TB-Q500-ae",
             "feature_max_abs_diff_package_vs_referee": float(np.abs(Ct - C[np.concatenate([fit[:50], test[:50]])]).max()),
             "s_max_rel_diff_package_vs_referee": float(np.abs(log.own().numpy() - s).max() / np.abs(s).max())}
        fitset = np.sort(fit)

        def ev(tau, strict):
            F = (C[test] > tau[None, :]) if strict else (C[test] >= tau[None, :])
            cmp_ = (lambda x: x > tau) if strict else (lambda x: x >= tau)
            return F, {"frac_rows_firing": float(F.any(1).mean()), "pairs_per_row": float(F.sum() / len(test)),
                       "cov_rel": float(cmp_(C[L["rel"], ar]).mean()), "cov_tgen": float(cmp_(C[L["tgen"], ar]).mean()), "cov_igen": float(cmp_(C[L["igen"], ar]).mean())}

        def run(perms_fit, alpha, shared_all_rows):
            acc = {k: [] for k in ("lit_hyb", "lib_hyb", "lit_sh", "lib_sh")}
            diff = {"hyb_decisions_differing": 0, "hyb_differing_within_4ulp": 0, "sh_decisions_differing": 0, "sh_differing_within_4ulp": 0,
                    "hyb_resplits_with_a_difference": 0, "n_decisions": 0}
            kq = lambda m: int(math.ceil((m + 1) * (1 - alpha) - 1e-12))                         # noqa: E731  (the referee's rank)
            for p in perms_fit:
                b1, b2 = p[: n // 2], p[n // 2:]
                aref = C[b1].max(0)
                sc = ((C[b2] - aref[None, :]) / s[None, :]).max(1)
                k = kq(len(b2))
                eta = np.partition(sc, k - 1)[k - 1] if k <= len(b2) else np.inf
                tau_lit = aref + eta * s
                F_lit, e_lit = ev(tau_lit, True)
                S, rec = CA.calibrate_two_block(S0, coords=C[p], alpha=alpha, blocks=(np.arange(n // 2), np.arange(n // 2, n)), u=s, allow_silence=True)
                tau_lib = S.det_tau.numpy()
                F_lib, e_lib = ev(tau_lib, False)
                d = F_lit != F_lib
                diff["n_decisions"] += d.size
                diff["hyb_decisions_differing"] += int(d.sum())
                diff["hyb_resplits_with_a_difference"] += int(d.any())
                if d.any():
                    rr, ee = np.nonzero(d)
                    diff["hyb_differing_within_4ulp"] += int((np.abs(C[test][rr, ee] - tau_lit[ee]) <= 4 * np.spacing(np.abs(tau_lit[ee]))).sum())
                acc["lit_hyb"].append(e_lit)
                acc["lib_hyb"].append(e_lib)
                assert rec["eta"] == eta and rec["rank"] == k, (rec["eta"], eta, rec["rank"], k)
                sub = fit if shared_all_rows else p[:n]
                sc = (C[sub] / s[None, :]).max(1)
                k = kq(n)
                eta = np.partition(sc, k - 1)[k - 1] if k <= n else np.inf
                F_lit, e_lit = ev(eta * s, True)
                Ss, _ = CA.calibrate_shared(S0, coords=C[sub], alpha=alpha, u=s, allow_silence=True)
                F_lib, e_lib = ev(Ss.det_tau.numpy(), False)
                d = F_lit != F_lib
                diff["sh_decisions_differing"] += int(d.sum())
                if d.any():
                    rr, ee = np.nonzero(d)
                    diff["sh_differing_within_4ulp"] += int((np.abs(C[test][rr, ee] - (eta * s)[ee]) <= 4 * np.spacing(np.abs((eta * s)[ee]))).sum())
                acc["lit_sh"].append(e_lit)
                acc["lib_sh"].append(e_lib)
            mean = {k: {kk: float(np.mean([e[kk] for e in v])) for kk in v[0]} for k, v in acc.items()}
            return mean, diff

        checks = []
        for script, alphas, reps, labels in (("rf2_c6_more.py", (0.05, 0.2), 200, ("hybrid", "shared_conformal_tau0=s")),
                                             ("rf2_c6_checks.py", (0.05, 0.10, 0.20), 300, ("hyb_u=s", "shared_s"))):
            fresh, perms, stored = res[script]
            if name not in fresh["REAL"] or isinstance(fresh["REAL"][name], str):
                continue
            mine = [p for p in perms if len(p) == n and np.array_equal(np.sort(p), fitset)]
            keyf = "rules" if script == "rf2_c6_more.py" else "D4_hybrid_vs_shared_on_heldout"
            ok_rerun = _same_numbers(fresh["REAL"][name], stored["REAL"][name])
            sc = {"script": script, "rerun_reproduces_stored_numbers": bool(ok_rerun), "n_permutations_of_the_fit_rows": len(mine), "alphas": []}
            if len(mine) != reps * len(alphas):
                sc["error"] = f"expected {reps * len(alphas)} recorded permutations of the fit rows"
                checks.append(sc)
                ok_all = False
                continue
            if "default_form" not in r:                                   # the protocol's DEFAULT form on the same halves (alpha = 5 %):
                Sn = BL.feature_gate(log, "ncoord")                       # normalised coordinate r = c / s, u = 1 (in exact arithmetic the same
                Cn = Sn.scores(torch.from_numpy(L["K"])).numpy()          # rule as the referee's '(c - a) / s'; the round-off differs)
                acc, nd, ntot = [], 0, 0
                for pm in mine[:reps]:
                    b1, b2 = pm[: n // 2], pm[n // 2:]
                    aref = C[b1].max(0)
                    scd = ((C[b2] - aref[None, :]) / s[None, :]).max(1)
                    k = int(math.ceil((len(b2) + 1) * (1 - alphas[0]) - 1e-12))
                    eta = np.partition(scd, k - 1)[k - 1] if k <= len(b2) else np.inf
                    F_ref = C[test] > (aref + eta * s)[None, :]
                    Sd, _ = CA.calibrate_two_block(Sn, coords=Cn[pm], alpha=alphas[0], blocks=(np.arange(n // 2), np.arange(n // 2, n)), allow_silence=True)
                    td = Sd.det_tau.numpy()
                    F_def = Cn[test] >= td[None, :]
                    nd += int((F_def != F_ref).sum())
                    ntot += F_def.size
                    acc.append({"frac_rows_firing": float(F_def.any(1).mean()), "cov_rel": float((Cn[L["rel"], ar] >= td).mean()),
                                "cov_tgen": float((Cn[L["tgen"], ar] >= td).mean()), "cov_igen": float((Cn[L["igen"], ar] >= td).mean())})
                th = fresh["REAL"][name][keyf][0][labels[0]]
                dm = {kk: float(np.mean([e[kk] for e in acc])) for kk in acc[0]}
                r["default_form"] = dict(dm, alpha=alphas[0], resplits=reps, script=script, decisions_differing_from_referee_form=nd, n_decisions=ntot,
                                         max_abs_diff_to_referee=float(max(abs(dm[kk] - th[kk]) for kk in dm if kk in th)))
                ok_all = ok_all and (r["default_form"]["max_abs_diff_to_referee"] <= 1e-3 or not r["development_log"])
            for ai, alpha in enumerate(alphas):
                mean, diff = run(mine[ai * reps:(ai + 1) * reps], alpha, shared_all_rows=True)
                theirs = fresh["REAL"][name][keyf][ai]
                th_h, th_s = theirs[labels[0]], theirs[labels[1]]
                keys = [k for k in ("frac_rows_firing", "cov_rel", "cov_tgen", "cov_igen") if k in th_h]
                rec = {"alpha": alpha, "resplits": reps,
                       "referee": {"two_block": {k: th_h[k] for k in keys}, "shared": {k: th_s[k] for k in keys}},
                       "transcription": {"two_block": {k: mean["lit_hyb"][k] for k in keys}, "shared": {k: mean["lit_sh"][k] for k in keys}},
                       "library": {"two_block": {k: mean["lib_hyb"][k] for k in keys}, "shared": {k: mean["lib_sh"][k] for k in keys}},
                       "decisions": diff}
                rec["transcription_equals_referee"] = bool(all(abs(mean["lit_hyb"][k] - th_h[k]) < 1e-12 and abs(mean["lit_sh"][k] - th_s[k]) < 1e-12 for k in keys))
                rec["library_max_abs_diff_to_referee"] = float(max(max(abs(mean["lib_hyb"][k] - th_h[k]), abs(mean["lib_sh"][k] - th_s[k])) for k in keys))
                rec["ok"] = bool(rec["transcription_equals_referee"] and rec["library_max_abs_diff_to_referee"] <= 1e-3
                                 and diff["hyb_decisions_differing"] == diff["hyb_differing_within_4ulp"]
                                 and diff["sh_decisions_differing"] == diff["sh_differing_within_4ulp"])
                sc["alphas"].append(rec)
            sc["ok"] = bool(ok_rerun and all(x["ok"] for x in sc["alphas"]))
            ok_all = ok_all and (sc["ok"] or not r["development_log"])
            checks.append(sc)
        r["checks"] = checks
        out["logs"][name] = r
    out["pass"] = bool(ok_all and all(len(v["checks"]) >= 1 for v in out["logs"].values())
                       and all(len(out["logs"][nm]["checks"]) == 2 for nm in ("TB-Q100-ae-l2_3", "TB-L100-ae-l2_3")))
    return out


TESTS = {"a17_two_block_naive": test_two_block_naive, "a18_two_block_validity": test_two_block_validity, "a19_float_safe": test_float_safe,
         "a20_two_block_referee": test_two_block_referee}
