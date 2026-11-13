"""E0 (3): the gate-level numbers of M1 / M2 against the theory referee's numbers on the same keys, one difference at a time.

  python e0_referee.py            ->  out/referee.json   (both development logs; CPU, no model)

Referee (rf2_c6_checks.py, section D4, alpha = 5 %; quoted in PROTOCOL.md): stored P11 keys (teacher-forced key at the last
prompt position); calibration on ALL valid REPAIR rows (two-block: 300 random halves; shared scale: all rows); read-out on
ALL valid HELD-OUT rows; no guards; a row fires iff some edit is a candidate; coverage = the sequence's own edit is a candidate.
Ladder (each step changes ONE thing; the last step is M1 / M2 of the gate-level table):
  R0   the referee's stored numbers
  R0r  his script re-run unchanged in a scratch copy (its generator is wrapped only to record the permutations it returns)
  R1h  the library's rule (key cache) on the referee's OWN 300 halves                      (must reproduce R0)
  R1   the same setting, 60000 seeded halves: the expectation over halves (R0 - R1 = Monte Carlo error of his 300 halves);
       shared scale: all REPAIR rows, no randomness
  R2   calibration pool: n = 200 rows drawn at random from REPAIR, 6000 seeded draws (the systematic effect of the pool size)
  R2s  the same on the 20 draws of this experiment (seeds 0 .. 19: draw noise of a 20-draw mean)
  R3   read-out pool: HELD-OUT-DISJOINT instead of HELD-OUT           (20 draws)
  R4   guards on: B = 1 own Rel keys as guards, capped thresholds     (20 draws)
  R5   top-1: coverage = the own edit is the top-1 candidate          (20 draws)  = M1 / M2
Positions: the cached teacher-forced keys are compared bit for bit with the P11 keys the referee used.
Decisions of the ladder are taken on the matrix of library scores (gate.scores) against the library's thresholds; e0_checks.py
verifies the end point R5 against the numbers of rescope.KeyCache.decide.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import e0_common as C

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
import rescope  # noqa: E402
from rescope import KeyCache, add_guards, from_testbed  # noqa: E402

N_HALVES = 60000
N_MANY = 6000
_RUNNER = r'''
import runpy, sys
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


def rerun_referee(scratch):
    """Run an unchanged copy of rf2_c6_checks.py in `scratch` (it writes its JSON next to itself); its generator is wrapped
    only to record the permutations it returns (as the library's test a20 does) -> (fresh JSON, permutations, seconds)."""
    os.makedirs(scratch, exist_ok=True)
    dst = os.path.join(scratch, "rf2_c6_checks.py")
    shutil.copy2(os.path.join(C.REFEREE, "rf2_c6_checks.py"), dst)
    runner = os.path.join(scratch, "_runner.py")
    with open(runner, "w") as f:
        f.write(_RUNNER)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    t0 = time.time()
    r = subprocess.run([sys.executable, runner, dst], capture_output=True, text=True, env=env, cwd=scratch)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    with open(os.path.join(scratch, "rf2_c6_checks.json")) as f:
        fresh = json.load(f)
    with np.load(dst + ".permutations.npz") as z:
        perms = [z[k] for k in sorted(z.files)]
    os.remove(dst + ".permutations.npz")                              # reproducible; 7 MB
    return fresh, perms, time.time() - t0


def max_diff(a, b):
    """Largest absolute difference between the numbers of two JSON values of the same shape (inf if the shapes differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return max([max_diff(a[k], b[k]) for k in a] + [0.0]) if a.keys() == b.keys() else float("inf")
    if isinstance(a, list) and isinstance(b, list):
        return max([max_diff(x, y) for x, y in zip(a, b)] + [0.0]) if len(a) == len(b) else float("inf")
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return 0.0 if a == b else float("inf")
    return abs(a - b)


def run(name: str, fresh, perms):
    t0 = time.time()
    bb = C.LOGS[name]["backbone"]
    kc = KeyCache(bb)
    L = from_testbed(name)
    uids, T = L.meta["uids"], L.T
    ar = np.arange(T)
    probes = kc.log_rows(uids, kind="probe", valid=True)
    pinfo = [kc.index[j] for j in probes]
    rep = np.array([i for i, e in enumerate(pinfo) if e["pool"] == "repair"])
    held = np.array([i for i, e in enumerate(pinfo) if e["pool"] == "heldout"])
    disj = np.array([i for i, e in enumerate(pinfo) if e.get("disjoint")])
    ed = {}
    for m in ("rel", "tgen", "igen"):
        js = kc.log_rows(uids, kind=m)
        assert [kc.index[j]["edit"] for j in js] == list(range(T))
        ed[m] = js
    own = kc.own_keys(uids)
    G0 = C.base_gate(L, "ncoord")                                     # guards: the stored solver keys only
    Gg, _ = add_guards(G0, *own)                                      # + the B = 1 keys of the edits' own Rel prompts
    Rp = G0.scores(kc.keys("tf", probes)).numpy()                     # (n_probes, T) normalised coordinates, teacher-forced key
    Re = {m: G0.scores(kc.keys("tf", ed[m])).numpy() for m in ed}
    n_rep = len(rep)

    # ---- the P11 keys the referee read against the cache's teacher-forced keys of the same sequences, bit for bit
    kd = os.path.join(C.P11, name, "keys")
    with open(os.path.join(kd, "meta.json")) as f:
        meta = json.load(f)
    K11 = np.load(os.path.join(kd, "K_last.npy"))
    by = {kc.index[j]["lib_sid"]: j for j in kc.log_rows(uids, kind=("rel", "tgen", "igen", "probe"))}
    n_in = n_eq = n_lp = 0
    for q in meta["seqs"]:
        j = by.get(q["sid"])
        if j is not None:
            e = kc.index[j]
            n_in += 1
            n_eq += int(np.array_equal(K11[q["n"]], kc.K("tf")[int(e["row_tf"])].numpy()))
            n_lp += int(q["last"] == e["lp_tf"])
    keys = {"n_p11_sequences": len(meta["seqs"]), "n_in_cache": n_in, "keys_bitwise_equal": n_eq, "last_prompt_index_equal": n_lp}
    # ---- the referee's own halves of the fit rows (P11 row numbers -> positions in `probes`)
    sid_pos = {e["lib_sid"]: i for i, e in enumerate(pinfo)}
    pos_of = {q["n"]: sid_pos[q["sid"]] for q in meta["seqs"] if q["kind"] == "probe"}
    fit = np.sort([q["n"] for q in meta["seqs"] if q["kind"] == "probe" and q["pool"] == "repair"])
    his = [np.array([pos_of[x] for x in p]) for p in perms if len(p) == len(fit) and np.array_equal(np.sort(p), fit)]
    assert len(his) == 900 and len(fit) == n_rep and set(his[0].tolist()) == set(rep.tolist())      # 300 halves x 3 alphas; the first 300: alpha = 5 %

    def read(G, test, top1):
        th = G.theta().numpy()
        out = {"firing": 100.0 * float((Rp[test] >= th[None, :]).any(1).mean())}
        for m in ed:
            cand = Re[m] >= th[None, :]
            if top1:
                rk = G.rank_key(torch.from_numpy(Re[m])).numpy()
                top = np.where(cand, rk, -np.inf).argmax(1)
                cov = cand.any(1) & (top == ar)
            else:
                cov = cand[ar, ar]
            out["cov_" + m] = 100.0 * float(cov.mean())
        return out

    def step(cal, specs, test, guards, top1, what):
        """specs: list of ('halves', seed) | ('his', permutation positions) | ('draw', seed): one calibration each."""
        mid = {"two_block": "M1", "shared": "M2"}[cal]
        S = Gg if guards else G0
        acc, capped, etas, bound, n_cal = [], [], [], None, None
        for kind, x in specs:
            if kind == "his":                                         # the referee's halves: block 1 = the first n // 2 rows of his permutation
                n = len(x)
                G, rec = rescope.calibrate_two_block(S, coords=Rp[x], alpha=C.ALPHA, blocks=(np.arange(n // 2), np.arange(n // 2, n)), allow_silence=not guards)
            elif kind == "halves":                                    # all valid REPAIR rows, seeded split
                G, rec = C.calibrate(S, mid, x, coords=Rp[rep], allow_silence=not guards)
            else:                                                     # = splits.sample(repair, n, seed), seeded split
                idx = rep[np.sort(np.random.default_rng(x).choice(n_rep, C.N_CAL, replace=False))]
                G, rec = C.calibrate(S, mid, x, coords=Rp[idx], allow_silence=not guards)
            acc.append(read(G, test, top1))
            capped.append(rec["n_edits_capped"])
            etas.append(rec["eta"])
            bound, n_cal = rec["bound_false_firing"], rec["n"]
        out = {k: C.over_draws([a[k] for a in acc]) for k in acc[0]}
        for v in out.values():
            v.pop("values")
            v["se"] = v["sd"] / float(np.sqrt(v["n_draws"])) if v["n_draws"] > 1 else 0.0
        out.update(what=what, n_draws=len(acc), n_cal=n_cal, n_test=int(len(test)), guards=guards, top1=top1,
                   mean_edits_capped=float(np.mean(capped)), mean_eta=float(np.mean(etas)), bound_false_firing_percent=100.0 * bound)
        raw[(cal, what)] = {k: np.array([a[k] for a in acc]) for k in acc[0]}
        return out

    raw = {}

    def paired(a, b):
        """Two-block minus shared scale on the SAME calibration draws: mean and standard error of the per-draw differences."""
        out = {}
        for k in ("firing", "cov_tgen", "cov_igen", "cov_rel"):
            d = a[k] - b[k]
            out[k] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0, "n_draws": int(d.size)}
        return out

    stored = C.jload(os.path.join(C.REFEREE, "rf2_c6_checks.json"))["REAL"][name]
    d4s = [r for r in stored["D4_hybrid_vs_shared_on_heldout"] if r["alpha"] == 0.05][0]
    d4f = [r for r in fresh["REAL"][name]["D4_hybrid_vs_shared_on_heldout"] if r["alpha"] == 0.05][0]
    res = {"log": name, "T": T, "n_repair_valid": n_rep, "n_heldout_valid": int(len(held)), "n_heldout_disjoint_valid": int(len(disj)),
           "referee": {"n_fit": stored["n_fit"], "n_test": stored["n_test"], "resplits": d4s["resplits"]}, "keys": keys,
           "rerun_max_abs_diff_REAL_section": max_diff(stored, fresh["REAL"][name]), "ladder": {}}
    draws20 = [("draw", s) for s in range(C.N_DRAWS)]
    for cal, lab in (("two_block", "hyb_u=s"), ("shared", "shared_s")):
        def ref_row(d, what):
            return {"what": what, "firing": 100.0 * d[lab]["frac_rows_firing"], "cov_rel": 100.0 * d[lab]["cov_rel"],
                    "cov_tgen": 100.0 * d[lab]["cov_tgen"], "cov_igen": 100.0 * d[lab]["cov_igen"]}
        tb = cal == "two_block"
        lad = {"R0": ref_row(d4s, "referee, stored numbers (300 halves of all REPAIR rows; HELD-OUT; no guards; candidate)"),
               "R0r": ref_row(d4f, "referee's script re-run unchanged")}
        if tb:
            lad["R1h"] = step(cal, [("his", p) for p in his[:300]], held, False, False, "library rule on the referee's own 300 halves (key cache)")
            lad["R1h_next600"] = step(cal, [("his", p) for p in his[300:]], held, False, False,
                                      "library rule on the referee's NEXT 600 halves (drawn by his script for alpha = 10 % and 20 %)")
        lad["R1"] = step(cal, [("halves", s) for s in (range(N_HALVES) if tb else range(1))], held, False, False,
                         f"same setting, {N_HALVES} seeded halves (expectation over halves)" if tb else "library rule in the referee's setting (all REPAIR rows; no randomness)")
        lad["R2"] = step(cal, [("draw", s) for s in range(N_MANY)], held, False, False, f"calibration pool: n = {C.N_CAL} drawn from REPAIR ({N_MANY} draws)")
        lad["R2s"] = step(cal, draws20, held, False, False, f"the {C.N_DRAWS} draws of this experiment (seeds 0 - {C.N_DRAWS - 1})")
        lad["R3"] = step(cal, draws20, disj, False, False, "read-out pool: HELD-OUT-DISJOINT")
        lad["R4"] = step(cal, draws20, disj, True, False, "guards on (capped thresholds)")
        lad["R5"] = step(cal, draws20, disj, True, True, "top-1 coverage = " + ("M1" if tb else "M2"))
        res["ladder"][cal] = lad
        C.say(f"{name} {cal}: " + "; ".join(f"{k} {v['firing'] if isinstance(v['firing'], float) else v['firing']['mean']:.2f} / "
                                            f"{v['cov_tgen'] if isinstance(v['cov_tgen'], float) else v['cov_tgen']['mean']:.1f}" for k, v in lad.items())
              + f" ({time.time() - t0:.0f}s)")
    # ---- how unusual are the referee's 300 halves?  (same rule, same keys: only the halves differ between R0 and R1)
    tb = res["ladder"]["two_block"]
    r1 = raw[("two_block", tb["R1"]["what"])]
    mc = {}
    for k in ("firing", "cov_tgen", "cov_igen"):
        blocks = r1[k].reshape(-1, 300).mean(1)                      # means of disjoint blocks of 300 seeded halves
        se300 = float(r1[k].std(ddof=1) / np.sqrt(300))
        mc[k] = {"referee_300_halves": tb["R0"][k], "library_on_his_halves": tb["R1h"][k]["mean"], "expectation": tb["R1"][k]["mean"],
                 "se_of_a_300_halves_mean": se300, "z_of_referee": (tb["R0"][k] - tb["R1"][k]["mean"]) / se300,
                 "n_blocks_of_300": int(blocks.size), "block_means_min": float(blocks.min()), "block_means_max": float(blocks.max()),
                 "n_blocks_at_least_as_far_from_the_expectation": int((np.abs(blocks - r1[k].mean()) >= abs(tb["R0"][k] - r1[k].mean())).sum())}
    res["referee_monte_carlo"] = mc
    res["library_on_his_halves_max_abs_diff_to_his_numbers"] = max(abs(tb["R1h"][k]["mean"] - tb["R0r"][k]) for k in ("firing", "cov_rel", "cov_tgen", "cov_igen"))
    # ---- two-block minus shared scale, paired on the same calibration draws
    sh = res["ladder"]["shared"]
    one_shared = {k: np.full(1, sh["R1"][k]["mean"]) for k in ("firing", "cov_tgen", "cov_igen", "cov_rel")}
    adv = {"all_REPAIR_rows_referee_readout": {k: {"mean": float(r1[k].mean() - one_shared[k][0]), "se": float(r1[k].std(ddof=1) / np.sqrt(r1[k].size)),
                                                   "n_draws": int(r1[k].size)} for k in one_shared},
           "n200_referee_readout": paired(raw[("two_block", tb["R2"]["what"])], raw[("shared", sh["R2"]["what"])])}
    m12 = {}
    for cal in ("two_block", "shared"):                               # M1 / M2 as defined (guards, top-1, HELD-OUT-DISJOINT) on the 6000 draws
        m12[cal] = step(cal, [("draw", s) for s in range(N_MANY)], disj, True, True, f"M1 / M2 as defined, {N_MANY} draws")
    res["m1_m2_many_draws"] = m12
    adv["n200_M1_minus_M2_as_defined"] = paired(raw[("two_block", m12["two_block"]["what"])], raw[("shared", m12["shared"]["what"])])
    res["two_block_minus_shared"] = adv
    C.say(f"{name}: referee z {({k: round(v['z_of_referee'], 2) for k, v in mc.items()})}; two-block minus shared "
          f"{({a: {k: round(v['mean'], 2) for k, v in b.items()} for a, b in adv.items()})} ({time.time() - t0:.0f}s)")
    res["seconds"] = time.time() - t0
    return res


if __name__ == "__main__":
    scratch = os.path.join(C.OUT, "referee_rerun")
    fresh, perms, sec = rerun_referee(scratch)
    C.say(f"referee script re-run in {sec:.0f}s ({scratch}); {len(perms)} permutations recorded")
    out = {"alpha": C.ALPHA, "n_halves": N_HALVES, "n_many": N_MANY, "referee_script": os.path.join(C.REFEREE, "rf2_c6_checks.py"),
           "referee_rerun_seconds": sec, "code": C.code_fingerprint(), "logs": {name: run(name, fresh, perms) for name in C.LOGS}}
    C.jdump(out, os.path.join(C.OUT, "referee.json"), indent=1)
    C.say(f"wrote {os.path.join(C.OUT, 'referee.json')}")
