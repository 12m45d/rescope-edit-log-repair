#!/usr/bin/env python3
"""Part 2b: two further checks of the paired differences of E3.
(1) REPLICATION of the stored 2000-resample bootstrap with my own code under the documented seed scheme of the experiments (draw
    multiplicities from numpy default_rng([0, 101]), unit multiplicities from default_rng([0, 102]), integers(0, n, (n_boot, n)); 95 %
    percentile interval; p = share of resamples on the other side of 0, doubled, floor 1 / 2000): the stored interval ends and p must
    then come out exactly.  The multiplicities attach to POSITIONS of the grid, so the order of the units matters: the experiments'
    grid orders the units by their string (fx.metrics.grid: sorted(units, key=str)); the columns are put in that order here.
(2) PRECISE two-sided values (400000 own resamples) for the members of the Holm families whose p is not at the floor, to see
    whether a Holm decision depends on the Monte Carlo error of a 2000-resample p.
-> out/mine_e3_boot.json"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlib import *  # noqa: E402,F401,F403
from p2_e3 import edit_grid, SEL  # noqa: E402


def counts(n, n_boot, rng):
    idx = rng.integers(0, n, size=(n_boot, n))
    c = np.zeros((n_boot, n))
    np.add.at(c, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return c


def replicate(A, B, units, n_boot=2000, seed=0):
    order = sorted(range(len(units)), key=lambda i: str(units[i]))
    A, B = A[:, order], B[:, order]
    nd, nu = A.shape
    cd, cu = counts(nd, n_boot, np.random.default_rng([seed, 101])), counts(nu, n_boot, np.random.default_rng([seed, 102]))
    one = np.ones_like(A)
    f = lambda N: ((cd @ N) * cu).sum(1)                                                        # noqa: E731
    st = 100.0 * (f(A) / f(one) - f(B) / f(one))
    p = 2.0 * min(float((st <= 0).mean()), float((st >= 0).mean()))
    return {"value": 100.0 * float(A.mean() - B.mean()), "ci_lo": float(np.percentile(st, 2.5)), "ci_hi": float(np.percentile(st, 97.5)), "p": min(1.0, max(p, 1.0 / n_boot)),
            "se": float(st.std(ddof=1))}


def main():
    out = {"replicate": {}, "precise": {}}
    mine = jload(OUT + "/mine_e3.json")
    for log in QLOGS + LLOGS:
        b, T = bb(log), T_of(log)
        later = np.arange(LATER_FROM[b], T)
        o = {}
        A, U = edit_grid(log, "rescope", "tf")
        for role, ref in (("Cos", SEL["cos_H2"]), ("Retr", SEL["retr"])):
            Bm, _ = edit_grid(log, ref, "tf")
            o["Htwo" + role + "All"] = replicate(A, Bm, U)
            if len(later):
                o["Htwo" + role + "Later"] = replicate(A[:, later], Bm[:, later], [U[i] for i in later])
        for pair, a, ref in (("PairOne", "rescope", SEL["cos_H3"]), ("PairTwo", "rescope_x", SEL["cos_H3_x"]), ("Retr", "rescope", SEL["retr"])):
            Ah, Uh = edit_grid(log, a, "h3")
            Bh, _ = edit_grid(log, ref, "h3")
            o["Hthree" + pair] = replicate(Ah, Bh, Uh)
            m = mine["logs"][log]["H3"][pair]
            if log not in DEV.values() and m["p"] > 1.5 / m["n_boot"]:
                out["precise"][f"{log} | {pair}"] = {s: crossed_boot_diff(Ah, Bh, n_boot=400000, seed=s, chunk=4000) for s in (11, 12)}
        out["replicate"][log] = o
        forget()
        print(log, {k: (round(v["ci_lo"], 3), round(v["ci_hi"], 3), v["p"]) for k, v in o.items()}, flush=True)
    for k, v in out["precise"].items():
        print("precise", k, [(x["p"], x["n_other_side"]) for x in v.values()])
    json.dump(out, open(OUT + "/mine_e3_boot.json", "w"), indent=1)


if __name__ == "__main__":
    main()
