#!/usr/bin/env python3
"""Brute-force check of the sentence that follows Thm. thm:least in sec_theory.tex:

    "Beyond them least repairs are lost: a role set chosen together with a threshold admits two incomparable
     repairs, and so does the tolerance criterion."        ("them": threshold scopes on one detector)

The two examples behind it are consequences (ii) and (iii) after Prop. prop:app:union of the technical report
(app_rescope.tex). Their arguments were written during the transcription of the proofs and no referee examined
them (marker [GAP:conseq] of proof_gaps.txt). This script checks both examples by exhaustive enumeration, in exact
rational arithmetic, with its own implementation of the definitions of Sec. 2 of the paper (Def. def:scoped: an
edit fires at a position iff the role of the position is in its role set and the normalised coordinate
<psi,k>/s is >= its threshold; thresholds live in the chain {-inf} u {a, a^+} u {+inf}; Def. def:fire: a query is
certified iff no edit fires at a position of its pre-edit trace; Def. def:repairs(i): tolerance criterion
||r(R)||_p <= c ||D||_p). Nothing is imported from the provers' or the referees' scripts.

    python3 checks/check_lost_least_repairs.py        writes checks/check_lost_least_repairs.json; exit 1 on failure

PART A  role set chosen together with one threshold (one edit; a repair may shrink the role set and raise the
        threshold; "least repair" = a certifying repair whose scope contains the scope of every certifying repair).
   A1   the example of the report: all roles, a complaint key at the role `last' whose coordinate is at or above
        the threshold. Expected: exactly two maximal certifying scopes (remove the role; raise the threshold just
        above the complaint), incomparable, and their union is not a scope of the class.
   A2   the same with the complaint at the role `ans' and the own request (role `last', coordinate 1) REQUIRED to
        stay covered: the two repairs still exist and are incomparable, so the loss is not an artefact of giving
        up the own request.
   A3   control: with the role set fixed (thresholds only) there is exactly one maximal certifying scope, the one
        of the threshold max(tau0, c^+) of Thm. thm:least(a).
   A4   random instances (several complaint positions, random role sets and thresholds): thresholds only -> always
        a greatest certifying scope, equal to the closed form; role set + threshold -> how often there are two or
        more maximal certifying scopes; whenever there is a greatest one it is the union of all certifying scopes
        (Prop. prop:app:union).
PART B  tolerance criterion (three edits with the scalar contributions +1, -1, +1 at one complaint key).
   B1   Def. def:repairs(i) on an edit log (rollback sets R, weight W_R): the sets that meet the criterion are
        exactly {1}, {3}, {1,2,3}, for every tolerance c in [0,1) that is tried and p in {1, 2, inf}.
   B2   the same as a threshold repair of a scoped log (own requests at the normalised coordinate 1 stay covered,
        the complaint has the coordinate 1/2 under every edit): the threshold vectors tau >= tau0 that meet the
        criterion have exactly two minimal elements, which are incomparable, so there is no least repair.
   B3   control: under the certificate criterion (no edit fires on the complaint) the same instance has the least
        repair of Thm. thm:least(a).
"""
import itertools
import json
import os
import random
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
NEG, POS = "-inf", "+inf"
ROLES = ("pre", "last", "ans")


# ----------------------------------------------------------------------------------------------- the chain Theta
def th(a, plus=0):
    """threshold a (plus = 0) or 'just above a' (plus = 1)"""
    return (Fr(a), plus)


def th_key(t):
    if t == NEG:
        return (0, 0, 0)
    if t == POS:
        return (2, 0, 0)
    return (1, t[0], t[1])


def th_le(t1, t2):
    return th_key(t1) <= th_key(t2)


def meets(coord, t):
    """coord >= t in the chain: zeta >= a iff zeta >= a; zeta >= a^+ iff zeta > a."""
    if t == NEG:
        return True
    if t == POS:
        return False
    return coord > t[0] if t[1] else coord >= t[0]


def ip(u, v):
    return sum(Fr(a) * Fr(b) for a, b in zip(u, v))


# ----------------------------------------------------------------------------------------------- scoped log (Def. def:scoped)
class Edit:
    def __init__(self, psi, own_key, roles, thr, val, resp="identity"):
        self.psi, self.roles, self.thr, self.val, self.resp = tuple(psi), frozenset(roles), thr, tuple(val), resp
        self.own = ip(psi, own_key)
        assert self.own > 0, "own coordinate must be positive (Def. def:log)"

    def coord(self, key):
        return ip(self.psi, key) / self.own

    def fires(self, role, key):
        return role in self.roles and meets(self.coord(key), self.thr)

    def with_(self, roles=None, thr=None):
        e = Edit.__new__(Edit)
        e.__dict__.update(self.__dict__)
        if roles is not None:
            e.roles = frozenset(roles)
        if thr is not None:
            e.thr = thr
        return e


def module_delta(edits, role, key):
    """module output minus W_0 k at one position (eq. (scoped))"""
    n = len(edits[0].val)
    out = [Fr(0)] * n
    for e in edits:
        if e.fires(role, key):
            g = e.coord(key) if e.resp == "identity" else Fr(1)
            out = [o + g * Fr(v) for o, v in zip(out, e.val)]
    return out


def certified(edits, trace):
    return not any(e.fires(role, key) for e in edits for role, key in trace)


def norm(vec, p):
    a = [abs(x) for x in vec]
    if p == "inf":
        return max(a)
    if p == 1:
        return sum(a)
    return sum(x * x for x in a)          # p = 2: squared norm (compare with c^2 times the squared norm)


def tol_ok(res, dmg, c, p):
    if p == 2:
        return norm(res, 2) <= c * c * norm(dmg, 2)
    return norm(res, p) <= c * norm(dmg, p)


# =============================================================================================== PART A
GRID = [Fr(x) for x in (-1, 0, Fr(1, 4), Fr(1, 2), Fr(3, 4), 1, Fr(5, 4))]
CHAIN = [NEG] + [t for g in GRID for t in (th(g), th(g, 1))] + [POS]
SUBSETS = [frozenset(s) for r in range(4) for s in itertools.combinations(ROLES, r)]
PSI, OWNKEY = (1, 0), (1, 0)                       # detector and own key in R^2: zeta(k) = k[0]
UNIVERSE = [(role, (g, 7)) for role in ROLES for g in GRID]      # test keys (second entry irrelevant to psi)


def scope_of(edit):
    return frozenset((role, key) for role, key in UNIVERSE if edit.fires(role, key))


def analyse_role_class(roles0, thr0, trace, must=()):
    """All repairs (roles' subset of roles0, thr' >= thr0) of ONE edit that certify the trace (and keep the
    positions of `must' firing): their scopes, the maximal ones, and whether a greatest one exists."""
    base = Edit(PSI, OWNKEY, roles0, thr0, (1,))
    cert = {}
    for roles in SUBSETS:
        if not roles <= frozenset(roles0):
            continue
        for t in CHAIN:
            if not th_le(thr0, t):
                continue
            e = base.with_(roles=roles, thr=t)
            if certified([e], trace) and all(e.fires(r, k) for r, k in must):
                cert.setdefault(scope_of(e), []).append((sorted(roles), t))
    scopes = list(cert)
    maximal = [s for s in scopes if not any(s < s2 for s2 in scopes)]
    union = frozenset().union(*scopes) if scopes else frozenset()
    greatest = [s for s in scopes if all(s2 <= s for s2 in scopes)]
    return cert, maximal, greatest, union


def all_scopes_of_class():
    base = Edit(PSI, OWNKEY, ROLES, NEG, (1,))
    return {scope_of(base.with_(roles=r, thr=t)) for r in SUBSETS for t in CHAIN}


def show(t):
    return t if isinstance(t, str) else (f"{t[0]}" + ("^+" if t[1] else ""))


def part_a():
    out, ok = {}, True
    class_scopes = all_scopes_of_class()
    # ---- A1: the example of the report
    thr0, c = th(Fr(1, 4)), Fr(1, 2)
    trace = [("last", (c, 3))]
    cert, maximal, greatest, union = analyse_role_class(ROLES, thr0, trace)
    descr = sorted((sorted({(tuple(r), show(t)) for r, t in cert[s]}), len(s)) for s in maximal)
    incomparable = len(maximal) == 2 and not (maximal[0] <= maximal[1]) and not (maximal[1] <= maximal[0])
    exp1 = scope_of(Edit(PSI, OWNKEY, ("pre", "ans"), thr0, (1,)))            # remove the role `last'
    exp2 = scope_of(Edit(PSI, OWNKEY, ROLES, th(c, 1), (1,)))                 # raise the threshold just above c
    a1 = dict(n_certifying_scopes=len(cert), n_maximal=len(maximal), incomparable=incomparable,
              maximal_are_the_two_repairs_of_the_report=set(maximal) == {exp1, exp2},
              greatest_exists=bool(greatest), union_is_a_scope_of_the_class=(maximal[0] | maximal[1]) in class_scopes,
              union_certifies=("last", (c, 7)) not in (maximal[0] | maximal[1]),
              maximal_repairs=[[[list(r), t] for r, t in d] for d, _ in descr])
    a1["pass"] = (a1["n_maximal"] == 2 and incomparable and a1["maximal_are_the_two_repairs_of_the_report"]
                  and not a1["greatest_exists"] and not a1["union_is_a_scope_of_the_class"] and a1["union_certifies"])
    out["A1_report_example_complaint_at_role_last"] = a1
    ok &= a1["pass"]
    # ---- A2: complaint at the role `ans', own request (role last, coordinate 1) must stay covered
    trace2 = [("ans", (c, 3))]
    own_pos = [("last", OWNKEY)]
    cert2, max2, great2, _ = analyse_role_class(ROLES, thr0, trace2, must=own_pos)
    inc2 = len(max2) == 2 and not (max2[0] <= max2[1]) and not (max2[1] <= max2[0])
    e1 = scope_of(Edit(PSI, OWNKEY, ("pre", "last"), thr0, (1,)))
    e2 = scope_of(Edit(PSI, OWNKEY, ROLES, th(c, 1), (1,)))
    a2 = dict(n_certifying_scopes=len(cert2), n_maximal=len(max2), incomparable=inc2,
              maximal_are_remove_role_and_raise_threshold=set(max2) == {e1, e2}, greatest_exists=bool(great2),
              union_is_a_scope_of_the_class=(max2[0] | max2[1]) in class_scopes if len(max2) == 2 else None)
    a2["pass"] = (a2["n_maximal"] == 2 and inc2 and a2["maximal_are_remove_role_and_raise_threshold"]
                  and not a2["greatest_exists"] and a2["union_is_a_scope_of_the_class"] is False)
    out["A2_own_request_must_stay_covered"] = a2
    ok &= a2["pass"]
    # ---- A3: control, thresholds only
    base = Edit(PSI, OWNKEY, ROLES, thr0, (1,))
    certs = {}
    for t in CHAIN:
        if th_le(thr0, t) and certified([base.with_(thr=t)], trace):
            certs.setdefault(scope_of(base.with_(thr=t)), []).append(t)
    max3 = [s for s in certs if not any(s < s2 for s2 in certs)]
    star = th(c, 1) if th_le(thr0, th(c, 1)) else thr0
    a3 = dict(n_maximal=len(max3), greatest_is_closed_form=max3 == [scope_of(base.with_(thr=star))],
              least_threshold=show(min(certs[max3[0]], key=th_key)) if max3 else None, closed_form=show(star))
    a3["pass"] = a3["n_maximal"] == 1 and a3["greatest_is_closed_form"] and a3["least_threshold"] == a3["closed_form"]
    out["A3_control_thresholds_only"] = a3
    ok &= a3["pass"]
    # ---- A4: random instances
    rng = random.Random(20261110)
    n, hist, thr_only_ok, union_ok, lost = 3000, {}, 0, 0, 0
    for _ in range(n):
        roles0 = rng.choice([s for s in SUBSETS if s])
        thr0 = rng.choice(CHAIN[:9])
        trace = [(rng.choice(ROLES), (rng.choice(GRID), rng.randint(-3, 3))) for _ in range(rng.randint(1, 3))]
        cert, maximal, greatest, union = analyse_role_class(roles0, thr0, trace)
        hist[len(maximal)] = hist.get(len(maximal), 0) + 1
        lost += len(maximal) >= 2
        union_ok += (not greatest) or (greatest[0] == union)
        if greatest:
            assert len(maximal) == 1
        else:
            assert len(maximal) >= 2 and union not in cert
        base = Edit(PSI, OWNKEY, roles0, thr0, (1,))            # thresholds only
        cs = [c_ for r, c_ in ((r, k[0]) for r, k in trace) if r in roles0]
        star = thr0
        for c_ in cs:
            if th_le(star, th(c_, 1)):
                star = th(c_, 1)
        tc = {}
        for t in CHAIN:
            if th_le(thr0, t) and certified([base.with_(thr=t)], trace):
                tc.setdefault(scope_of(base.with_(thr=t)), []).append(t)
        mx = [s for s in tc if not any(s < s2 for s2 in tc)]
        thr_only_ok += (len(mx) == 1 and mx[0] == scope_of(base.with_(thr=star))
                        and min(tc[mx[0]], key=th_key) == star and certified([base.with_(thr=star)], trace))
    a4 = dict(instances=n, thresholds_only_unique_least_equal_to_closed_form=f"{thr_only_ok}/{n}",
              role_set_and_threshold_number_of_maximal_certifying_scopes={str(k): v for k, v in sorted(hist.items())},
              instances_without_a_least_repair=lost, greatest_scope_is_union_whenever_it_exists=f"{union_ok}/{n}")
    a4["pass"] = thr_only_ok == n and union_ok == n and lost > 0
    out["A4_random_instances"] = a4
    ok &= a4["pass"]
    return out, ok


# =============================================================================================== PART B
def part_b():
    out, ok = {}, True
    contrib = [Fr(1), Fr(-1), Fr(1)]
    T = 3
    tolerances = [Fr(0), Fr(1, 4), Fr(1, 2), Fr(3, 4), Fr(99, 100)]
    expected = [frozenset({0}), frozenset({2}), frozenset({0, 1, 2})]
    # ---- B1: edit log of Lemma lem:app:realise: k_q = e_0, k_i = e_i, psi_i = e_0 + e_i, u_i = w_i; W_0 = 0
    e = lambda j: tuple(1 if x == j else 0 for x in range(T + 1))
    kq = e(0)
    psi = [tuple(a + b for a, b in zip(e(0), e(i + 1))) for i in range(T)]
    u = contrib
    w = [u[i] * ip(psi[i], kq) for i in range(T)]                 # contribution Delta_i k_q = u_i <psi_i, k_q>
    assert w == contrib and all(ip(psi[i], e(i + 1)) == 1 for i in range(T))
    D = sum(w)
    rows, good = [], True
    for c in tolerances:
        for p in (1, 2, "inf"):
            feas = []
            for r in range(T + 1):
                for R in itertools.combinations(range(T), r):
                    WRkq = sum(u[i] * ip(psi[i], kq) for i in range(T) if i not in R)       # (W_R - W_0) k_q
                    if tol_ok([WRkq], [D], c, p):
                        feas.append(frozenset(R))
            minimal = [R for R in feas if not any(R2 < R for R2 in feas)]
            least = [R for R in feas if all(R <= R2 for R2 in feas)]
            good_here = (sorted(map(sorted, feas)) == sorted(map(sorted, expected)) and len(minimal) == 2
                         and not least and not (minimal[0] <= minimal[1]) and not (minimal[1] <= minimal[0]))
            good &= good_here
            rows.append(dict(c=str(c), p=p, feasible_sets_1_based=sorted([i + 1 for i in sorted(R)] for R in feas),
                             minimal_sets_1_based=sorted([i + 1 for i in sorted(R)] for R in minimal),
                             least_set_exists=bool(least), as_expected=good_here))
    out["B1_rollback_sets"] = dict(contributions=[str(x) for x in contrib], damage=str(D), cases=rows, **{"pass": good})
    ok &= good
    # ---- B2: scoped log; own request at normalised coordinate 1 (role last), complaint coordinate 1/2 (role last)
    psi2 = [tuple(Fr(1, 2) if x == 0 else (1 if x == i + 1 else 0) for x in range(T + 1)) for i in range(T)]
    vals = [2 * x for x in contrib]                               # contribution = zeta_i(k_q) * v_i = v_i / 2
    own = [e(i + 1) for i in range(T)]
    chain = [NEG, th(Fr(1, 2)), th(Fr(1, 2), 1), th(1), th(1, 1), POS]
    thr0 = [NEG] * T
    base = [Edit(psi2[i], own[i], ROLES, thr0[i], (vals[i],)) for i in range(T)]
    assert all(b.coord(kq) == Fr(1, 2) and b.coord(own[i]) == 1 for i, b in enumerate(base))
    dmg0 = module_delta(base, "last", kq)
    assert dmg0 == [D]
    rows2, good2 = [], True
    for c in tolerances:
        for p in (1, 2, "inf"):
            feas = []
            for tau in itertools.product(chain, repeat=T):
                ed = [base[i].with_(thr=tau[i]) for i in range(T)]
                if not all(ed[i].fires("last", own[i]) for i in range(T)):          # own requests stay covered
                    continue
                if tol_ok(module_delta(ed, "last", kq), dmg0, c, p):
                    feas.append(tau)
            le = lambda a, b: all(th_le(x, y) for x, y in zip(a, b))
            minimal = [t for t in feas if not any(le(t2, t) and t2 != t for t2 in feas)]
            least = [t for t in feas if all(le(t, t2) for t2 in feas)]
            unfired = sorted({tuple(i + 1 for i in range(T) if not meets(Fr(1, 2), t[i])) for t in feas})
            j = th(Fr(1, 2), 1)
            exp_min = {(j, NEG, NEG), (NEG, NEG, j)}
            good_here = (set(minimal) == exp_min and not least and unfired == [(1,), (1, 2, 3), (3,)]
                         and not le(minimal[0], minimal[1]) and not le(minimal[1], minimal[0]))
            good2 &= good_here
            rows2.append(dict(c=str(c), p=p, n_feasible_threshold_vectors=len(feas),
                              unfired_sets_of_feasible_vectors=[list(x) for x in unfired],
                              minimal_threshold_vectors=sorted([show(x) for x in t] for t in minimal),
                              least_repair_exists=bool(least), as_expected=good_here))
    out["B2_threshold_repairs_of_a_scoped_log"] = dict(cases=rows2, **{"pass": good2})
    ok &= good2
    # ---- B3: control, certificate criterion
    feas = []
    for tau in itertools.product(chain, repeat=T):
        ed = [base[i].with_(thr=tau[i]) for i in range(T)]
        if all(ed[i].fires("last", own[i]) for i in range(T)) and certified(ed, [("last", kq)]):
            feas.append(tau)
    le = lambda a, b: all(th_le(x, y) for x, y in zip(a, b))
    least = [t for t in feas if all(le(t, t2) for t2 in feas)]
    j = th(Fr(1, 2), 1)
    b3 = dict(n_certifying=len(feas), least_repair=[show(x) for x in least[0]] if least else None,
              closed_form=[show(j)] * T)
    b3["pass"] = least == [(j, j, j)]
    out["B3_control_certificate_criterion"] = b3
    ok &= b3["pass"]
    return out, ok


def main():
    a, oka = part_a()
    b, okb = part_b()
    res = dict(
        sentence=("Beyond them [threshold scopes on one detector] least repairs are lost: a role set chosen together with a threshold admits "
                  "two incomparable repairs, and so does the tolerance criterion."),
        method="exhaustive enumeration, exact rational arithmetic, own implementation of Defs. def:scoped, def:fire, "
               "def:repairs; see the docstring of checks/check_lost_least_repairs.py",
        part_A_role_set_and_threshold=a, part_B_tolerance_criterion=b,
        role_set_example_holds=oka, tolerance_example_holds=okb, ALL_PASS=oka and okb)
    json.dump(res, open(os.path.join(HERE, "check_lost_least_repairs.json"), "w"), indent=1)
    print("A (role set + threshold):", "PASS" if oka else "FAIL", "| B (tolerance criterion):", "PASS" if okb else "FAIL")
    for name, r in list(a.items()) + list(b.items()):
        print(f"  {name}: {'pass' if r['pass'] else 'FAIL'}")
    print("wrote", os.path.join(HERE, "check_lost_least_repairs.json"))
    sys.exit(0 if oka and okb else 1)


if __name__ == "__main__":
    main()
