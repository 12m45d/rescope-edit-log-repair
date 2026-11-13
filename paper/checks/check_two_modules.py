#!/usr/bin/env python3
"""Exact check of the only statement that the paper makes about SEVERAL edited modules (sec_theory.tex, paragraph
"What the results do and do not say"):

    "With several modules, raising a threshold can fire another edit [technical report, App. B.2]; certificate
     and least repair are open."

The counterexample is Example ex:app:twomodules (Example B.3, in App. B.2) of the technical report (app_rescope.tex); the prover checked it in
check_a1_locality.py (E1) and the referee of part A re-derived it by hand. This script re-computes it with its own
code, in exact rational arithmetic, and adds two controls.

    python3 checks/check_two_modules.py        writes checks/check_two_modules.json; exit 1 on failure

Model (Def. def:scoped of the paper, scalar keys): a module with pre-edit weight W0 returns, at the key k,
    W0 * k + sum over the edits of the module that fire at k of  g_i * v_i ,       edit i fires iff  psi_i * k >= tau_i
(own coordinate 1, constant response g_i = 1). Two modules are composed: the output h of the first is the key of the
second, so the keys of the second module depend on the log, which is what assumption (H1) excludes.

E1   (report, Example ex:app:twomodules) W0 = 1 in both modules; e1 in module 1 fires where k >= 1, value -5; e2 in
     module 2 fires where h >= 0, value 10; query key k = 2. Expected: pre-edit output 2; scoped log: h = -3, output
     -3, firing set {e1}; after raising tau_1 above 2: h = 2, output 12, firing set {e2}. So raising a threshold makes
     an edit fire that did not fire before: firing is not antitone in the thresholds (Prop. prop:certificate(c) fails).
E1b  (report, Example ex:app:stored) e2 fires where -h >= 1. At k = 2 the scoped log gives h = -3, e2 fires, output
     7; with the value of e2 changed from 10 to 20 the output is 17. On the stored pre-edit keys (2 and 2) e2 does
     not fire: a lineage computed on stored keys misses an edit whose change alters the answer.
C1   control, ONE module: on a grid of instances with two edits in the same module, raising a threshold never adds
     an edit to the firing set (the statement of Prop. prop:certificate(c)).
C2   control, TWO modules: on the same grid with e2 in a second module, the number of instances on which raising
     tau_1 adds e2 to the firing set (it must be positive; E1 is one of them).
"""
import itertools
import json
import os
import sys
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
INF = "inf"


def fires(psi, tau, key):
    """psi * key >= tau; tau = (a, 0) means a, (a, 1) means 'just above a', INF never."""
    if tau == INF:
        return False
    a, plus = tau
    c = Fr(psi) * Fr(key)
    return c > a if plus else c >= a


def module(key, edits, w0=1):
    """edits: list of (name, psi, tau, value). Returns (output, set of firing edits)."""
    out, fired = Fr(w0) * Fr(key), set()
    for name, psi, tau, val in edits:
        if fires(psi, tau, key):
            out += Fr(val)
            fired.add(name)
    return out, fired


def two_modules(k, m1, m2):
    h, f1 = module(k, m1)
    out, f2 = module(h, m2)
    return h, out, f1 | f2


def main():
    res, ok = {}, True
    # ------------------------------------------------------------------ E1
    k = 2
    pre = two_modules(k, [], [])
    e1 = ("e1", 1, (Fr(1), 0), -5)
    e2 = ("e2", 1, (Fr(0), 0), 10)
    before = two_modules(k, [e1], [e2])
    e1_raised = ("e1", 1, (Fr(2), 1), -5)                     # threshold just above the coordinate 2 of the query
    after = two_modules(k, [e1_raised], [e2])
    e1_res = dict(pre_edit_output=str(pre[1]), scoped=dict(h=str(before[0]), output=str(before[1]), fire=sorted(before[2])),
                  after_raising_tau1=dict(h=str(after[0]), output=str(after[1]), fire=sorted(after[2])))
    e1_res["pass"] = (pre[1] == 2 and before[:2] == (Fr(-3), Fr(-3)) and before[2] == {"e1"}
                      and after[:2] == (Fr(2), Fr(12)) and after[2] == {"e2"} and not after[2] <= before[2])
    res["E1_firing_is_not_antitone"] = e1_res
    ok &= e1_res["pass"]
    # ------------------------------------------------------------------ E1b
    e2b = ("e2", -1, (Fr(1), 0), 10)
    e2c = ("e2", -1, (Fr(1), 0), 20)
    out10 = two_modules(k, [e1], [e2b])
    out20 = two_modules(k, [e1], [e2c])
    stored_fire_e2 = fires(-1, (Fr(1), 0), pre[0])            # stored pre-edit key of module 2 is h = 2
    e1b = dict(output_value_10=str(out10[1]), output_value_20=str(out20[1]), fire=sorted(out10[2]),
               e2_fires_on_stored_pre_edit_key=stored_fire_e2)
    e1b["pass"] = out10[1] == 7 and out20[1] == 17 and "e2" in out10[2] and not stored_fire_e2
    res["E1b_frame_rule_fails_on_stored_keys"] = e1b
    ok &= e1b["pass"]
    # ------------------------------------------------------------------ controls on a grid
    keys = [Fr(x) for x in (-2, -1, 0, 1, 2)]
    taus = [(Fr(x), p) for x in (-2, -1, 0, 1, 2) for p in (0, 1)] + [INF]
    order = {t: n for n, t in enumerate(taus)}
    vals = [-5, -1, 1, 10]
    psis = [1, -1]
    n1 = bad1 = n2 = bad2 = 0
    for k, p1, p2, v1, v2, t2 in itertools.product(keys, psis, psis, vals, vals, taus):
        for ta, tb in itertools.combinations(taus, 2):       # tau_1 raised from ta to tb (ta < tb in the chain)
            assert order[ta] < order[tb]
            # one module: both edits read the same key k
            _, fa = module(k, [("e1", p1, ta, v1), ("e2", p2, t2, v2)])
            _, fb = module(k, [("e1", p1, tb, v1), ("e2", p2, t2, v2)])
            n1 += 1
            bad1 += not fb <= fa
            # two modules: e2 reads the output of module 1
            fa2 = two_modules(k, [("e1", p1, ta, v1)], [("e2", p2, t2, v2)])[2]
            fb2 = two_modules(k, [("e1", p1, tb, v1)], [("e2", p2, t2, v2)])[2]
            n2 += 1
            bad2 += not fb2 <= fa2
    res["C1_one_module_grid"] = dict(pairs=n1, firing_set_grew=bad1, **{"pass": bad1 == 0})
    res["C2_two_modules_grid"] = dict(pairs=n2, firing_set_grew=bad2, **{"pass": bad2 > 0})
    ok &= bad1 == 0 and bad2 > 0
    out = dict(sentence="With several modules, raising a threshold can fire another edit [technical report, App. B.2]; "
                        "certificate and least repair are open.",
               what_is_checked="the counterexample only (monotonicity of firing fails with two edited modules); nothing "
                               "is claimed or checked about a certificate or a least repair for several modules",
               results=res, ALL_PASS=bool(ok))
    json.dump(out, open(os.path.join(HERE, "check_two_modules.json"), "w"), indent=1)
    for name, r in res.items():
        print(f"  {name}: {'pass' if r['pass'] else 'FAIL'}")
    print("ALL_PASS:", bool(ok), "| wrote", os.path.join(HERE, "check_two_modules.json"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
