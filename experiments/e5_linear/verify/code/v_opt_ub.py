"""Verification, extra: what can be checked of the OPTIMUM-CODE numbers without running an exact solver again.

usage: python v_opt_ub.py        -> verify/out/v_opt_ub.json       (CPU, seconds)

The measuring agent's optimum rows (out/opt_rows_<log>.jsonl, key 'last') store, per complaint and tolerance c, a bracket
opt_lb <= OPT_c <= opt_ub and the best deletion set found, R_best.  Checked here with my own contributions (explicit
vectors w_i = u_i <psi_i, k>, last prompt key from the final key cache, fp64):
  ub_feasible   ||sum_{i not in R_best} w_i|| <= c ||D|| and |R_best| = opt_ub: the UPPER bound is a real deletion set;
  consistent    my lower bound mu(c^2 ||D||^2 / gamma) <= opt_ub and opt_lb <= my best rule (LF, FO, BI): no bracket
                contradicts my bounds.
NOT checked: the LOWER bounds opt_lb (branch and bound / cutting planes of the Track C code); a certificate would need
the solver's proof.  The numbers 'certified optimum', 'rule / OPT' and 'bound / OPT' rest on them.
"""
from __future__ import annotations

import v_common as C                    # first: it fixes the number of BLAS threads before numpy is loaded

import os
import time

import numpy as np


def main():
    t0 = time.time()
    out = {"definition": __doc__, "logs": {}}
    for name in C.LOGS:
        F = C.factors(name)
        U, Psi = F["U"], F["Psi"]
        rows = C.probe_rows(name)
        cand = C.candidates(rows)
        comps = {c["group"]: c for c in cand["image"] + cand["text"]}
        mine = {r["group"]: r for r in C.jload(os.path.join(C.OUT, f"v_cond_rollback_{name}.json"))["rows"]}
        opt = [o for o in C.jsonl(os.path.join(C.REPORT_OUT, f"opt_rows_{name}.jsonl")) if o["keydef"] == "last" and "tau" in o]
        cs = [comps[o["group"]] for o in opt]
        Kl, _ = C.last_keys(name, cs)
        res = {"n_rows": len(opt), "per_tol": {}}
        for t in C.TOLS:
            n_feas = n_size = n_cons = n_cert = 0
            worst = None
            for j, o in enumerate(opt):
                q = o["tau"][str(t)]
                c = Psi @ Kl[j].double().numpy()
                Wc = U * c[:, None]
                D = Wc.sum(0)
                kept = np.ones(len(U), dtype=bool)
                kept[np.asarray(q["R_best"], dtype=int)] = False
                r = Wc[kept].sum(0)
                rel = float(np.linalg.norm(r) / np.linalg.norm(D))
                n_feas += int(rel <= t)
                n_size += int(len(set(q["R_best"])) == int(q["opt_ub"]))
                m = mine[o["group"]]["keydefs"]["last"]["tol"][str(t)]
                n_cons += int(m["lb_mu"] <= int(q["opt_ub"]) and int(q["opt_lb"]) <= min(m["lf"], m["fo"], m["bi"]))
                n_cert += int(int(q["opt_lb"]) == int(q["opt_ub"]))
                if worst is None or rel / t > worst:
                    worst = rel / t
            res["per_tol"][str(t)] = {"n_ub_feasible": n_feas, "n_ub_size_equals_set": n_size, "n_consistent_with_my_bounds": n_cons,
                                      "n_trackc_bracket_closed": n_cert, "max_residual_over_tolerance": worst}
        out["logs"][name] = res
        print(name, res)
    out["sec"] = time.time() - t0
    C.jdump(out, os.path.join(C.OUT, "v_opt_ub.json"))


if __name__ == "__main__":
    main()
