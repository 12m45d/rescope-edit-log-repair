"""E9 maintenance of a scoped log: selective revocation (a) and lineage (b) on one development log (GPU).

    gq run --mem <MiB> --prio 1 -- python e9_run.py <log> [stage ...]      stages: base (base runs + lineage), m1, m5, m10, m20, m50 (default: all)

Amendment "E9" of PROTOCOL.md with R19 (c): calibration draw 0, three seeded revocation sets per m in {1, 5, 10, 20, 50}.
Every sequence of every run is ONE forward through the library's Evaluator with a hash of its logits (fx.behave ses.tf_hashed: no memo,
no key cache), so that "bit for bit" is a statement about forwards that ran.
  base runs     the calibrated default method M = fx.methods.rescope(L, draw=0), the deployed weight, the pre-edit weight on the 300 edit
                sequences (Rel, T-Gen, I-Gen) and the valid heldout_disjoint probes; M and the pre-edit weight also on the valid near-scope
                negatives on which the key cache says M releases an edit (or is inside the decision band)
  (a) scoped    fx.methods.mask_edits(L, M, R): thresholds of the edits of R = +inf, nothing else changes
      linear    fx.methods.rollback(L, H, R): the harness's rollback weight (stored rank-one solver terms of R subtracted from W_E)
      same revocation sets R(m, s) = sorted(numpy.random.default_rng([20261111, m, s]).choice(T, m, replace=False)) for both
  (b) lineage   every heldout_disjoint probe and near-scope negative on which M releases an edit: mask that edit, run the row; if another
                edit is released, mask it too, and so on until nothing is released (the chain is recorded)
Resumable: every run is one file under out/hash/ (skipped if present).  One log per process; ends with ses.assert_clean().
"""
from __future__ import annotations
import os as _o, sys as _s  # locations: rescope_paths.py at the repository root is the one configuration point
_d = _o.environ.get("RESCOPE_HOME") or _o.path.dirname(_o.path.realpath(__file__))
while not _o.path.exists(_o.path.join(_d, "rescope_paths.py")) and _o.path.dirname(_d) != _d: _d = _o.path.dirname(_d)
_s.path.insert(0, _d); import rescope_paths as _RP

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E9 = os.path.dirname(HERE)
OUT = os.path.join(E9, "out")
sys.path.insert(0, f"{_RP.ROOT}/02_research/experiments/10_final/common")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import fx  # noqa: E402

ALPHA, N, DRAW = 0.05, 200, 0
MS, N_SETS, SEED = (1, 5, 10, 20, 50), 3, 20261111
SHORT = {"TB-Q100-ae-l2_3": "Q", "TB-L100-ae-l2_3": "L"}
KEEP = ("kind", "csid", "sid", "metric", "edit", "ptype", "hn_type", "edit_b", "query_id", "uid", "strict", "near_edit", "y0_deg", "exact", "token_acc", "n_tok",
        "first_mismatch", "finite", "D1", "tok_loc", "fired", "top_last", "own_fired_last", "own_released_last", "n_cand_edits", "logit_sha1", "released", "lp_shift")


def rev_set(T: int, m: int, s: int) -> list:
    return sorted(int(i) for i in np.random.default_rng([SEED, int(m), int(s)]).choice(T, int(m), replace=False))


def slim(rows):
    return [{k: r[k] for k in KEEP if k in r} for r in rows]


def main():
    name = sys.argv[1]
    stages = sys.argv[2:] or ["base"] + [f"m{m}" for m in MS]
    sh = SHORT[name]
    L = fx.logs.open(name)
    H = fx.logs.load_model(name)
    ses = fx.behave.session(L, H)
    hd = os.path.join(OUT, "hash")
    os.makedirs(hd, exist_ok=True)
    M = fx.methods.rescope(L, alpha=ALPHA, n=N, draw=DRAW)
    rows_e = [j for m in ("rel", "tgen", "igen") for j in L.edit_rows[m]]
    rows_p = L.pool("heldout_disjoint")
    rows_h = L.rows(("hn_i", "hn_q", "hn_x"), valid=True)
    dh = fx.gate.decide(L, M, rows_h, "tf")
    rows_hf = [j for j, f, nr in zip(rows_h, dh["fired"], dh["near"]) if f or nr]
    main_rows = rows_e + rows_p

    def run(tag, method, rows, verbose=400):
        p = os.path.join(hd, f"{sh}__{tag}.jsonl.gz")
        if os.path.exists(p):
            return fx.io.load_rows(p)
        t0 = time.time()
        out = slim(ses.tf_hashed(method, rows, verbose=verbose))
        fx.io.save_rows(p, out)
        print(f"[e9] {name} {tag}: {len(rows)} forwards, {time.time() - t0:.0f}s", flush=True)
        return out

    sets = {m: [rev_set(L.T, m, s) for s in range(N_SETS)] for m in MS}
    fx.io.jdump({"log": name, "alpha": ALPHA, "n": N, "draw": DRAW, "method": {"label": M.label, "digest": M.digest, "key": M.key, "n_capped": fx.methods.n_capped(M),
                                                                              "cal": {k: M.rec["cal"].get(k) for k in ("eta", "n1", "n2", "rank", "bound_false_firing", "n_edits_capped")},
                                                                              "own_prompts": M.rec.get("own_prompts")},
                 "sets": {str(m): v for m, v in sets.items()}, "set_rule": f"sorted(numpy.random.default_rng([{SEED}, m, s]).choice(T, m, replace=False)), s = 0 .. {N_SETS - 1}",
                 "n_edit_sequences": len(rows_e), "n_probe_rows": len(rows_p), "n_hn_valid": len(rows_h), "n_hn_firing_or_near": len(rows_hf),
                 "hn_gate_level": {"n_fired": int(np.sum(dh["fired"])), "n_near": int(np.sum(dh["near"]))}, "code": fx.code_fingerprint()},
                os.path.join(OUT, f"setup_{name}.json"), indent=1)

    # ---------------------------------------------------------------- base runs
    base = run("base_scoped", M, main_rows + rows_hf)
    run("base_preedit", fx.methods.preedit(L), main_rows + rows_hf)
    run("base_deployed", fx.methods.deployed(L), main_rows)

    # ---------------------------------------------------------------- (b) lineage (cheap: first)
    pl = os.path.join(OUT, f"lineage_{name}.json")
    if "base" in stages and not os.path.exists(pl):
        t0 = time.time()
        by = {r["csid"]: r for r in base}
        seq = {L.sid(j): j for j in rows_p + rows_hf}
        state = {c: {"csid": c, "kind": by[c]["kind"], "hn_type": by[c].get("hn_type"), "own": by[c].get("edit"), "masked": [], "released": list(by[c]["released"]),
                     "chain": [{"masked": [], "released": list(by[c]["released"]), "logit_sha1": by[c]["logit_sha1"]}]}
                 for c in seq if by[c]["released"]}
        nfw, rnd = 0, 0
        active = [c for c in state]
        while active:
            rnd += 1
            groups = {}
            for c in active:
                st = state[c]
                if len(st["released"]) != 1:
                    raise RuntimeError(f"{c}: role 'last' top-1 releases one edit, got {st['released']}")
                st["masked"] = sorted(set(st["masked"]) | set(st["released"]))
                groups.setdefault(tuple(st["masked"]), []).append(c)
            for R, cs in sorted(groups.items()):
                Mm = fx.methods.mask_edits(L, M, list(R))
                res = ses.tf_hashed(Mm, [seq[c] for c in cs])
                nfw += len(cs)
                for c, r in zip(cs, res):
                    st = state[c]
                    st["released"] = list(r["released"])
                    st["chain"].append({"masked": list(R), "released": list(r["released"]), "logit_sha1": r["logit_sha1"], "D1": r.get("D1"), "exact": r.get("exact")})
            active = [c for c in active if state[c]["released"]]
            print(f"[e9] {name} lineage round {rnd}: {len(groups)} masks, {nfw} forwards so far, {len(active)} rows still releasing, {time.time() - t0:.0f}s", flush=True)
        dn = fx.gate.decide(L, M, [seq[c] for c in state], "tf")
        for c, nc in zip(state, dn["n_cand"]):
            state[c]["n_cand_gate_level"] = int(nc)
        fx.io.jdump({"log": name, "rows": list(state.values()), "n_forwards": nfw, "n_rounds": rnd, "seconds": time.time() - t0}, pl)

    # ---------------------------------------------------------------- (a) selective revocation
    for m in MS:
        if f"m{m}" not in stages:
            continue
        for s, R in enumerate(sets[m]):
            run(f"scoped_m{m}_s{s}", fx.methods.mask_edits(L, M, R), main_rows)
            Mr = fx.methods.rollback(L, H, R)
            pr = os.path.join(OUT, f"rollback_{name}_m{m}_s{s}.json")
            if not os.path.exists(pr):
                fx.io.jdump({k: Mr.rec.get(k) for k in ("edits", "n", "whole_batches", "rule", "equals_deployed_weight", "equals_pre_edit_weight")} | {"digest": Mr.digest}, pr, indent=1)
            run(f"linear_m{m}_s{s}", Mr, main_rows)
            del Mr
    rec = ses.assert_clean()
    fx.io.jdump({"log": name, "stages": stages, "assert_clean": rec, "stats": dict(ses.stats), "mismatches": ses.mismatches[:20], "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "code": fx.code_fingerprint()}, os.path.join(OUT, f"job_{name}_{'_'.join(stages)}.json"), indent=1)
    print("[e9] done", name, rec, flush=True)


if __name__ == "__main__":
    main()
