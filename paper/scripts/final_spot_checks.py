#!/usr/bin/env python3
"""Final check: independent spot checks of constructions that the PAPER states (exact arithmetic, small sizes).

    python3 scripts/final_spot_checks.py        writes scripts/final_spot_checks.json; exit 1 if a check fails

Not a replacement for the referee scripts of 03_analysis/theory: each check re-derives one explicit instance of
sec_theory.tex from its statement alone, without importing project code.
Labels after the revision of 2026-11-11 (review round 1); statements that left the paper are those of the technical report.
  G  technical report, Thm. thm:app:rollback-hard(d) (was Thm. thm:rollback-hard(c) of the paper): greedy deletes all
     T = 2L+2 contributions although OPT_c = 2 (both rules, all ties).
  I  technical report, Prop. prop:app:indep (was Thm. thm:cut(c)): independent thresholds cost (1-eps) T times the optimum.
  M  status of Thm. thm:calibration (was part (b)): the exchangeable law that attains min(1, T/(m+1)).
  B  Thm. thm:calibration(a) (was part (c)): two-block rule, false firing <= floor(alpha(m2+1))/(m2+1) on urn laws with ties.
  L  status of Thm. thm:least / Thm. thm:criteria(e): the tolerance criterion has two incomparable minimal repairs
     (+1, -1, +1; c = 1/2).
  C  Thm. thm:criteria(d) (new): the gap family psi_i = k_i = e_i, complaint keys xi (1 + e_j): every restoring rollback
     deletes all T edits, the only restoring weight is W_0, the least repair of the deployed scoping loses no edit and
     the module returns W_E k_i at every own key; every complaint key has nonzero deployed damage.
  X  status of Thm. thm:criteria(b) (new): rollback judged by restoration against thresholds judged by firing (+1, -1, +1).
  K  Thm. thm:lincap(a) (new): on small random logs with independent values, the sets of edits that some restoring
     weight preserves are exactly the sets whose own keys are independent modulo the span of the trace keys.
  N  numbers after Thm. thm:calibration (new): m_2 = 100, alpha = 0.05: 0.436; 95th percentile 8.92 %; l_0.1 = 2 with
     0.037; least sizes 45, 59, 90; 19 for the marginal bound (exact binomial sums).
  P  status of Thm. thm:hard(a) (new; technical report, Remark rem:app:pseudo): Min-Rollback with integer contributions
     in one output dimension by a table over the attainable sums = brute force.
  F  sentence after Thm. thm:calibration: the successor of the rounded product is not the calibrated comparison
     of ratios (float64 example of the referee of part C).
  U  guard band (Remark rem:app:guard, sentence after Thm. thm:least): a band EQUAL to the discrepancy can fire.
"""
import itertools
import json
import math
import os
import random
import sys
from fractions import Fraction as Fr

out, ok = {}, True


def record(name, passed, **info):
    global ok
    ok = ok and passed
    out[name] = dict(passed=bool(passed), **info)
    print(("PASS " if passed else "FAIL ") + name, info)


# ---------------------------------------------------------------------------------------------------------------- G
def greedy_instance(L, c):
    kap = (1 + c) / 2                                   # rational kappa in (c, 1)
    dim = 2 + L                                         # e0, e1, f_1..f_L
    def vec(**kw):
        v = [Fr(0)] * dim
        for k, x in kw.items():
            v[int(k[1:])] = Fr(x)
        return tuple(v)
    items = [vec(i0=Fr(1, 2), i1=2), vec(i0=Fr(1, 2), i1=-2)]
    for j in range(1, L + 1):
        phi = 1 + Fr(j, 4 * L)
        lure = vec(**{"i0": phi, f"i{1 + j}": kap})
        items += [lure, tuple(-x for x in lure)]
    return items


def add(vs, dim):
    return tuple(sum(v[k] for v in vs) for k in range(dim))


def sq(v):
    return sum(x * x for x in v)


def run_greedy(items, c, rule, branch_all=True):
    """returns the set of numbers of deletions over all tie-breaking branches"""
    dim, T = len(items[0]), len(items)
    D2 = sq(add(items, dim))
    results = set()
    stack = [frozenset()]
    seen = set()
    while stack:
        R = stack.pop()
        if R in seen:
            continue
        seen.add(R)
        res = add([items[i] for i in range(T) if i not in R], dim) if len(R) < T else tuple([Fr(0)] * dim)
        if sq(res) <= c * c * D2:
            results.add(len(R))
            continue
        cand = [i for i in range(T) if i not in R]
        if rule == "FO":
            score = {i: sum(a * b for a, b in zip(res, items[i])) for i in cand}
            best = max(score.values())
        else:
            score = {i: -sq(tuple(a - b for a, b in zip(res, items[i]))) for i in cand}
            best = max(score.values())
        for i in cand:
            if score[i] == best:
                stack.append(R | {i})
    return results


def opt(items, c):
    dim, T = len(items[0]), len(items)
    D2 = sq(add(items, dim))
    for r in range(T + 1):
        for R in itertools.combinations(range(T), r):
            keep = [items[i] for i in range(T) if i not in R]
            res = add(keep, dim) if keep else tuple([Fr(0)] * dim)
            if sq(res) <= c * c * D2:
                return r
    return None


g_ok, g_cases = True, 0
for L in (1, 2, 3):
    for c in (Fr(0), Fr(1, 2), Fr(9, 10)):
        it = greedy_instance(L, c)
        T = len(it)
        o = opt(it, c)
        fo, bi = run_greedy(it, c, "FO"), run_greedy(it, c, "BI")
        g_cases += 1
        g_ok = g_ok and o == 2 and fo == {T} and bi == {T} and T == 2 * L + 2
record("G greedy deletes all, OPT = 2", g_ok, cases=g_cases)

# ---------------------------------------------------------------------------------------------------------------- I
i_ok, i_cases = True, 0
for T in (2, 3, 4, 5):
    for eps in (Fr(1, 10), Fr(1, 3), Fr(49, 100)):
        lam = 1 - eps
        # each edit: low (pair covered, edit fires on the complaint) or high (pair uncovered, edit silent on it)
        def phi(state):                                 # state = tuple of 0 / 1 (1 = high)
            return (1 if all(state) else 0) - lam * sum(state)
        def cost(state):
            return (0 if all(state) else 1) + lam * sum(state)
        states = list(itertools.product((0, 1), repeat=T))
        best_phi, best_cost = max(phi(s) for s in states), min(cost(s) for s in states)
        indep = tuple(1 if (1 - lam) > 0 else 0 for _ in range(T))     # per-edit objective: high gives 1 - lam = eps > 0
        i_cases += 1
        i_ok = i_ok and best_phi == 0 and best_cost == 1 and cost(indep) == (1 - eps) * T * best_cost and phi(indep) < 0
record("I independent thresholds: ratio (1 - eps) T, negative gain", i_ok, cases=i_cases)

# ---------------------------------------------------------------------------------------------------------------- M
m_ok, m_cases = True, 0
for m in range(1, 7):
    for T in range(1, 9):
        pop = [[(1 if h == i else 0) - Fr(h, m + 2) for i in range(1, T + 1)] for h in range(1, m + 2)]
        holders = set()
        distinct = True
        for i in range(T):
            col = [pop[h][i] for h in range(m + 1)]
            distinct = distinct and len(set(col)) == len(col)
            mx = max(col)
            if col.count(mx) == 1:
                holders.add(col.index(mx))
        # the fresh query (uniformly random member) fires iff it alone attains the maximum of some coordinate
        p = Fr(len(holders), m + 1)
        m_cases += 1
        m_ok = m_ok and distinct and p == min(Fr(1), Fr(T, m + 1))
record("M least repair: min(1, T/(m+1)) attained without ties", m_ok, cases=m_cases)

# ---------------------------------------------------------------------------------------------------------------- B
rng = random.Random(20261110)
b_ok, b_cases, b_max = True, 0, Fr(0)
for trial in range(400):
    T = rng.choice((1, 2, 3))
    m1, m2 = rng.choice((1, 2)), rng.choice((1, 2, 3))
    alpha = rng.choice((Fr(1, 2), Fr(1, 3), Fr(2, 5), Fr(3, 4)))
    n = m1 + m2 + 1
    pop = [tuple(Fr(rng.randint(0, 3)) for _ in range(T)) for _ in range(n)]          # ties on purpose
    tau0 = tuple(Fr(rng.randint(1, 3)) for _ in range(T))
    rank = math.ceil((1 - alpha) * (m2 + 1))
    bound = Fr(math.floor(alpha * (m2 + 1)), m2 + 1)
    fires = total = 0
    for perm in itertools.permutations(range(n)):
        blk1, blk2, fresh = perm[:m1], perm[m1:m1 + m2], perm[-1]
        base = tuple(max(pop[h][i] for h in blk1) for i in range(T))
        sc = lambda h: max((pop[h][i] - base[i]) / tau0[i] for i in range(T))
        scores = sorted(sc(h) for h in blk2)
        total += 1
        if rank <= m2 and sc(fresh) > scores[rank - 1]:                                 # scale +infinity otherwise
            fires += 1
    p = Fr(fires, total)
    b_cases += 1
    b_max = max(b_max, p - bound)
    b_ok = b_ok and p <= bound
record("B two-block rule: false firing within the bound on urn laws with ties", b_ok, cases=b_cases,
       largest_excess=str(b_max))

# ---------------------------------------------------------------------------------------------------------------- L
contrib, c = [1, -1, 1], Fr(1, 2)
D = abs(sum(contrib))
feas = [set(R) for r in range(4) for R in itertools.combinations(range(3), r)
        if abs(sum(x for i, x in enumerate(contrib) if i not in R)) <= c * D]
minimal = [R for R in feas if not any(S < R for S in feas)]
record("L tolerance criterion: two incomparable minimal repairs", sorted(map(sorted, feas)) == [[0], [0, 1, 2], [2]]
       and sorted(map(sorted, minimal)) == [[0], [2]], feasible=sorted(map(sorted, feas)))

# ---------------------------------------------------------------------------------------------------------------- F
r, t0 = 0.03, 0.41
eta = r / t0
prod = eta * t0
succ = math.nextafter(prod, math.inf)
record("F successor of the rounded product is not the ratio gate", (not (r / t0 > eta)) and prod < r and r >= succ,
       eta=repr(eta), rounded_product=repr(prod), successor=repr(succ), ratio_gate_fires=bool(r / t0 > eta),
       product_gate_fires=bool(r >= succ))

# ---------------------------------------------------------------------------------------------------------------- U
z_repair, disc = Fr(1, 2), Fr(1, 10)
z_serve = z_repair + disc
record("U guard band equal to the discrepancy can fire; a larger one cannot",
       (z_serve >= z_repair + disc) and not (z_serve >= z_repair + disc + Fr(1, 1000)))

# ============================================================== checks added with the revision of 2026-11-11 (review round 1)
def rank_fr(rows):
    """rank of a matrix of Fractions (list of rows), by elimination"""
    m = [list(r) for r in rows]
    rk, col, ncol = 0, 0, (len(m[0]) if m else 0)
    while rk < len(m) and col < ncol:
        piv = next((i for i in range(rk, len(m)) if m[i][col] != 0), None)
        if piv is None:
            col += 1
            continue
        m[rk], m[piv] = m[piv], m[rk]
        for i in range(len(m)):
            if i != rk and m[i][col] != 0:
                f = m[i][col] / m[rk][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[rk])]
        rk += 1
        col += 1
    return rk


def solvable(A, B):
    """is X A = B solvable (A: d x a, B: n x a, as lists of rows)? iff rank [A; B] = rank A (row spaces)"""
    return rank_fr(A + B) == rank_fr(A) if A else all(all(x == 0 for x in r) for r in B)


# ---------------------------------------------------------------------------------------------------------------- C
c_ok, c_cases = True, 0
xi = Fr(1, 3)
for T in range(1, 5):
    for d in range(T, 6):
        n = T                                           # write vectors u_i = e_i in R^T: linearly independent
        ckeys = [[xi * (1 + (1 if a == j else 0)) for a in range(d)] for j in range(d)]      # xi (1 + e_j)
        coord = lambda i, k: k[i]                       # psi_i = k_i = e_i, own coordinate 1
        # deployed damage at a complaint key: sum_i coord_i u_i = (k_1, ..., k_T) != 0
        damage_nonzero = all(any(k[i] != 0 for i in range(T)) for k in ckeys)
        # a rollback keeping J restores iff sum_{i in J} k[i] e_i = 0 at every complaint key iff J is empty
        restoring = [R for r in range(T + 1) for R in itertools.combinations(range(T), r)
                     if all(all(k[i] == 0 for i in range(T) if i not in R) for k in ckeys)]
        only_all = restoring == [tuple(range(T))]
        # a restoring weight W_0 + D' has D' K_Q = 0; K_Q has rank d, so D' = 0
        full_rank = rank_fr(ckeys) == d
        # least repair of the deployed scoping: thresholds just above mu_i = max coordinate on the complaints
        mu = [max(coord(i, k) for k in ckeys) for i in range(T)]
        fires_alone = all(all((coord(j, [1 if a == i else 0 for a in range(d)]) > mu[j]) == (j == i) for j in range(T))
                          for i in range(T))            # at the own key e_i exactly edit i fires: output W_0 k_i + u_i = W_E k_i
        c_cases += 1
        c_ok = c_ok and damage_nonzero and only_all and full_rank and fires_alone and all(m == 2 * xi < 1 for m in mu)
record("C gap family: rollback deletes all, only W_0 restores, least repair loses no edit", c_ok, cases=c_cases)

# ---------------------------------------------------------------------------------------------------------------- X
contrib = [1, -1, 1]                                    # psi_i = k_i = 1; one complaint with the key 1; deployed damage 1
rollback_3_restores = sum(contrib[:2]) == 0             # deleting the third edit: kept contributions 1 and -1
# every edit has the coordinate 1 on the complaint, so a threshold vector under which no edit fires silences all three
certifying_cost = sum(1 for _ in contrib)
record("X mixed criteria: restoring rollback silences 1 pair, every certifying threshold repair 3",
       sum(contrib) == 1 and rollback_3_restores and certifying_cost == 3, rollback_cost=1, certifying_cost=certifying_cost)

# ---------------------------------------------------------------------------------------------------------------- K
rnd = random.Random(20261111)
k_ok, k_cases, k_sets = True, 0, 0
for _ in range(600):
    d, T, a = rnd.randint(2, 4), rnd.randint(1, 3), rnd.randint(0, 3)
    n = T
    KQ = [[Fr(rnd.randint(-2, 2)) for _ in range(a)] for _ in range(d)]                 # d x a (columns: trace keys)
    KE = [[Fr(rnd.randint(-2, 2)) for _ in range(T)] for _ in range(d)]                 # d x T (columns: own keys)
    Psi = [[Fr(rnd.randint(-2, 2)) for _ in range(T)] for _ in range(d)]                # d x T (columns: read-out vectors)
    U = [[Fr(1 if r == c_ else 0) for c_ in range(T)] for r in range(n)]                # write vectors e_i: independent
    coordm = [[sum(Psi[r][i] * KE[r][j] for r in range(d)) for j in range(T)] for i in range(T)]   # <psi_i, k_j>
    if rank_fr(coordm) < T or any(coordm[i][i] <= 0 for i in range(T)):
        continue                                        # hypothesis of the theorem: nonsingular coordinates, own coordinate > 0
    H = coordm                                          # dW K_E = U Psi^T K_E = Psi^T K_E for U = I: required values (n x T)
    rQ = rank_fr([list(r) for r in zip(*KQ)]) if a else 0
    for r in range(T + 1):
        for J in itertools.combinations(range(T), r):
            # D' [K_Q, K_J] = [0, H_J] solvable?  (rows of A = columns of [K_Q, K_J]: use the transposes)
            Acols = [[KQ[x][c_] for x in range(d)] for c_ in range(a)] + [[KE[x][j] for x in range(d)] for j in J]
            Bcols = [[Fr(0)] * n for _ in range(a)] + [[H[x][j] for x in range(n)] for j in J]
            # X A = B with A = [columns] (d x m): solvable iff every linear relation among the columns of A holds for B
            m_ = len(Acols)
            stacked = [Acols[c_] + Bcols[c_] for c_ in range(m_)]                        # relation space of columns
            preservable = (rank_fr(stacked) == rank_fr(Acols)) if m_ else True
            indep_mod = rank_fr(Acols) == rQ + len(J) if m_ else True
            k_sets += 1
            k_ok = k_ok and (preservable == indep_mod)
    k_cases += 1
record("K linear capacity: preservable sets = own keys independent modulo the trace keys", k_ok and k_cases >= 20,
       logs=k_cases, sets=k_sets, statement_holds_on_all=bool(k_ok))

# ---------------------------------------------------------------------------------------------------------------- N
def binom_cdf(n_, p_, x_):
    return sum(math.comb(n_, j) * p_ ** j * (1 - p_) ** (n_ - j) for j in range(0, x_ + 1)) if x_ >= 0 else 0.0


m2, al = 100, 0.05
l_def = math.floor(al * (m2 + 1))
p_exceed = binom_cdf(m2, al, l_def - 1)                              # P(FF > alpha) for continuous scores
lo, hi = 0.0, 1.0
for _ in range(80):                                                  # 95th percentile of Beta(l, m2 + 1 - l)
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if binom_cdf(m2, mid, l_def - 1) > 0.05 else (lo, mid)
l_delta = max(l for l in range(0, m2 + 1) if binom_cdf(m2, al, l - 1) <= 0.1)
sizes = [math.ceil(math.log(1 / dl) / math.log(1 / (1 - al))) for dl in (0.1, 0.05, 0.01)]
record("N numbers after Thm. thm:calibration", l_def == 5 and round(p_exceed, 3) == 0.436 and round(100 * hi, 2) == 8.92
       and l_delta == 2 and round(binom_cdf(m2, al, l_delta - 1), 3) == 0.037 and sizes == [45, 59, 90]
       and math.ceil(1 / al - 1) == 19, l=l_def, p_exceed=round(p_exceed, 4), pct95=round(100 * hi, 3), l_delta=l_delta,
       p_delta=round(binom_cdf(m2, al, l_delta - 1), 4), sizes=sizes)

# ---------------------------------------------------------------------------------------------------------------- P
p_ok, p_cases = True, 0
for _ in range(300):
    T = rnd.randint(1, 8)
    w = [rnd.randint(-4, 4) for _ in range(T)]
    c = Fr(rnd.randint(0, 9), 10)
    D = abs(sum(w))
    best = {0: 0}                                                    # sum of a kept subfamily -> its largest size
    for x in w:
        new = dict(best)
        for s_, k_ in best.items():
            if new.get(s_ + x, -1) < k_ + 1:
                new[s_ + x] = k_ + 1
        best = new
    dp = T - max(k_ for s_, k_ in best.items() if abs(s_) <= c * D)
    brute = min(len(R) for r in range(T + 1) for R in itertools.combinations(range(T), r)
                if abs(sum(x for i, x in enumerate(w) if i not in R)) <= c * D)
    p_cases += 1
    p_ok = p_ok and dp == brute
record("P one output dimension: table over the attainable sums = brute force", p_ok, cases=p_cases)

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_spot_checks.json"), "w"), indent=1)
sys.exit(0 if ok else 1)
